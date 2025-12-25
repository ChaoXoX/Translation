"""
书籍翻译智能代理 - 主入口

基于多智能体工作流的长文本领域书籍翻译系统
"""
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from config.settings import get_settings, update_settings
from src.workflow.orchestrator import TranslationOrchestrator
from src.workflow.pipeline import TranslationPipeline
from src.human_loop.interface import HumanLoopInterface, create_human_review_callback
from src.utils.file_utils import load_text_file, save_translation, FileHandler
from src.models.terminology import TerminologyDatabase

# 初始化
app = typer.Typer(help="书籍翻译智能代理 - 基于多智能体工作流的翻译系统")
console = Console()


@app.command()
def translate(
    input_file: Path = typer.Argument(..., help="输入文件路径"),
    output_dir: Path = typer.Option(
        Path("data/output"),
        "--output", "-o",
        help="输出目录"
    ),
    source_lang: str = typer.Option(
        "en",
        "--source", "-s",
        help="源语言"
    ),
    target_lang: str = typer.Option(
        "zh",
        "--target", "-t",
        help="目标语言"
    ),
    domain: str = typer.Option(
        "general",
        "--domain", "-d",
        help="领域 (general, literature, technical, legal, entertainment)"
    ),
    style: str = typer.Option(
        "formal",
        "--style",
        help="翻译风格 (formal, casual, literary)"
    ),
    enable_validation: bool = typer.Option(
        True,
        "--validation/--no-validation",
        help="是否启用回译验证"
    ),
    enable_human_review: bool = typer.Option(
        False,
        "--human-review/--no-human-review",
        help="是否启用人工审核"
    ),
    terminology_file: Optional[Path] = typer.Option(
        None,
        "--terminology", "-T",
        help="术语库文件路径"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="调试模式"
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="快速模式：关闭多版本与验证并增大chunk尺寸"
    )
):
    """
    翻译书籍文本
    
    示例:
        python main.py translate input.txt -o output/ -s en -t zh
    """
    # 配置日志
    if debug:
        logger.add("logs/debug_{time}.log", level="DEBUG")
    else:
        logger.add("logs/translation_{time}.log", level="INFO")
    
    console.print("\n[bold blue]═══ 书籍翻译智能代理 ═══[/bold blue]\n")
    
    # 检查输入文件
    if not input_file.exists():
        console.print(f"[red]错误: 输入文件不存在: {input_file}[/red]")
        raise typer.Exit(1)
    
    # 更新配置
    settings = get_settings()
    settings.translation.source_language = source_lang
    settings.translation.target_language = target_lang
    settings.translation.domain = domain
    settings.translation.style = style
    settings.debug = debug
    
    # 快速模式：减少LLM调用次数
    if fast:
        settings.translation.generate_variants = False
        enable_validation = False
        # 增大chunk以减少单元数（在预处理Agent初始化前生效）
        settings.translation.chunk_size = max(settings.translation.chunk_size, 2200)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载输入文本
    console.print(f"[cyan]加载文件:[/cyan] {input_file}")
    input_text = load_text_file(input_file)
    
    console.print(f"[cyan]文本长度:[/cyan] {len(input_text)} 字符")
    console.print(f"[cyan]翻译方向:[/cyan] {source_lang} → {target_lang}")
    console.print(f"[cyan]领域:[/cyan] {domain}")
    console.print()
    
    # 加载术语库（如果有）
    terminology_db = None
    if terminology_file and terminology_file.exists():
        console.print(f"[cyan]加载术语库:[/cyan] {terminology_file}")
        terminology_db = TerminologyDatabase.load(terminology_file)
    
    # 运行翻译
    asyncio.run(_run_translation(
        input_text=input_text,
        title=input_file.stem,
        output_dir=output_dir,
        terminology_db=terminology_db,
        enable_validation=enable_validation,
        enable_human_review=enable_human_review
    ))


async def _run_translation(
    input_text: str,
    title: str,
    output_dir: Path,
    terminology_db: Optional[TerminologyDatabase] = None,
    enable_validation: bool = True,
    enable_human_review: bool = False
):
    """执行翻译"""
    start_time = datetime.now()
    
    # 创建人机协作接口
    human_interface = None
    human_callback = None
    
    if enable_human_review:
        human_interface = HumanLoopInterface(interactive=True)
        human_callback = await create_human_review_callback(human_interface)
    
    # 创建翻译指挥官
    orchestrator = TranslationOrchestrator(
        terminology_db=terminology_db,
        human_review_callback=human_callback
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]翻译中...", total=None)
        
        try:
            # 执行翻译
            result = await orchestrator.run(
                input_text=input_text,
                title=title,
                enable_validation=enable_validation,
                enable_human_review=enable_human_review
            )
            
            progress.update(task, description="[green]翻译完成!")
            
        except Exception as e:
            progress.update(task, description=f"[red]翻译失败: {e}")
            logger.error(f"Translation failed: {e}")
            # 如果翻译部分完成，尝试保存
            if orchestrator.translation_result and orchestrator.translation_result.units:
                console.print("\n[yellow]翻译已部分完成，尝试保存现有结果...[/yellow]")
                try:
                    orchestrator.translation_result.merge_translations()
                    partial_output = output_dir / f"{title}_partial_{timestamp}.txt"
                    save_translation(
                        orchestrator.translation_result.full_translation,
                        partial_output,
                        format="txt",
                        metadata={"title": title, "source": "translation_agent", "status": "partial"}
                    )
                    console.print(f"[yellow]部分译文已保存到:[/yellow] {partial_output}")
                except:
                    pass
            raise
    
    # 保存结果
    console.print("\n[cyan]保存结果...[/cyan]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存译文
    output_file = output_dir / f"{title}_translated_{timestamp}.txt"
    save_translation(
        result.full_translation,
        output_file,
        format="txt",
        metadata={"title": title, "source": "translation_agent"}
    )
    
    # 保存术语库
    terminology_file = output_dir / f"{title}_terminology_{timestamp}.json"
    orchestrator.save_terminology_database(terminology_file)
    
    # 显示统计
    elapsed = (datetime.now() - start_time).total_seconds()
    state = orchestrator.get_state()
    
    # 检查是否有警告
    status_color = "green" if not state.warnings else "yellow"
    status_text = "翻译完成" if not state.warnings else "翻译完成（有警告）"
    
    console.print(f"\n[bold {status_color}]═══ {status_text} ═══[/bold {status_color}]\n")
    
    if state.warnings:
        console.print("[yellow]警告信息:[/yellow]")
        for warning in state.warnings:
            console.print(f"  ⚠️  {warning}")
        console.print()
    console.print(f"[cyan]总耗时:[/cyan] {elapsed:.1f} 秒")
    console.print(f"[cyan]  - 预处理:[/cyan] {state.preprocessing_time:.1f} 秒")
    console.print(f"[cyan]  - 术语提取:[/cyan] {state.terminology_time:.1f} 秒")
    console.print(f"[cyan]  - 翻译:[/cyan] {state.translation_time:.1f} 秒")
    console.print(f"[cyan]  - 验证:[/cyan] {state.validation_time:.1f} 秒")
    console.print()
    console.print(f"[cyan]处理单元:[/cyan] {state.total_chunks} 个")
    console.print(f"[cyan]译文长度:[/cyan] {len(result.full_translation)} 字符")
    console.print()
    console.print(f"[cyan]输出文件:[/cyan] {output_file}")
    console.print(f"[cyan]术语文件:[/cyan] {terminology_file}")


@app.command()
def analyze(
    input_file: Path = typer.Argument(..., help="输入文件路径"),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="输出文件路径"
    )
):
    """
    分析文本，提取术语和结构
    """
    console.print("\n[bold blue]═══ 文本分析 ═══[/bold blue]\n")
    
    if not input_file.exists():
        console.print(f"[red]错误: 文件不存在: {input_file}[/red]")
        raise typer.Exit(1)
    
    # 加载文本
    text = load_text_file(input_file)
    
    # 分析
    from src.utils.text_utils import TextProcessor
    processor = TextProcessor()
    stats = processor.get_statistics(text)
    
    console.print(f"[cyan]文件:[/cyan] {input_file}")
    console.print(f"[cyan]检测语言:[/cyan] {stats['language']}")
    console.print(f"[cyan]字符数:[/cyan] {stats['char_count']}")
    console.print(f"[cyan]词数:[/cyan] {stats['word_count']}")
    console.print(f"[cyan]句子数:[/cyan] {stats['sentence_count']}")
    console.print(f"[cyan]段落数:[/cyan] {stats['paragraph_count']}")
    
    # 提取潜在术语
    terms = processor.extract_potential_terms(text)
    if terms:
        console.print(f"\n[cyan]潜在术语 ({len(terms)}个):[/cyan]")
        for term in terms[:20]:
            console.print(f"  - {term}")
        if len(terms) > 20:
            console.print(f"  ... 还有 {len(terms) - 20} 个")


@app.command()
def review(
    tasks_file: Path = typer.Argument(..., help="审核任务文件路径"),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="审核结果输出文件"
    )
):
    """
    人工审核模式
    
    加载待审核任务并进行交互式审核
    """
    console.print("\n[bold blue]═══ 人工审核模式 ═══[/bold blue]\n")
    
    if not tasks_file.exists():
        console.print(f"[red]错误: 任务文件不存在: {tasks_file}[/red]")
        raise typer.Exit(1)
    
    interface = HumanLoopInterface(interactive=True)
    
    # 导入任务
    import json
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks_data = json.load(f)
    
    from src.human_loop.interface import ReviewTask, ReviewType
    
    for task_data in tasks_data.get("tasks", []):
        task = ReviewTask(
            task_id=task_data["task_id"],
            review_type=ReviewType(task_data.get("review_type", "translation")),
            source_text=task_data.get("source_text", ""),
            current_translation=task_data.get("current_translation", ""),
            alternatives=task_data.get("alternatives", []),
            context=task_data.get("context", ""),
            confidence=task_data.get("confidence", 0.5)
        )
        interface.add_task(task)
    
    # 处理审核
    results = asyncio.run(interface.process_tasks())
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {task_id: result.model_dump() for task_id, result in results.items()},
                f,
                ensure_ascii=False,
                indent=2,
                default=str
            )
        console.print(f"\n[cyan]结果已保存到:[/cyan] {output_file}")
    
    # 显示摘要
    interface.display_summary()


@app.command()
def demo():
    """
    运行演示翻译
    """
    console.print("\n[bold blue]═══ 演示模式 ═══[/bold blue]\n")
    
    # 示例文本
    sample_text = """
Chapter 1: The Beginning

Yo, let me tell you about growing up in Compton. It wasn't easy, but it made me who I am today. 
My homies and I used to freestyle on the corner, dreaming about making it big in the rap game.

"You got what it takes," my mama always said. She was my biggest supporter, even when times got tough.

The streets taught me a lot - about loyalty, about hustle, about keeping it real. Some of my boys 
caught beef with the wrong people, and that changed everything. But I stayed focused on the music.

That's when I met Dr. Dre. He saw something in me, you know what I'm saying? He said, 
"Kid, you got talent. Let's make some history."

And the rest, as they say, is hip-hop history.
"""
    
    console.print("[cyan]示例文本（说唱自传风格）:[/cyan]")
    console.print(sample_text)
    console.print()
    
    # 确认
    if not typer.confirm("是否开始演示翻译?"):
        raise typer.Exit(0)
    
    # 运行翻译
    output_dir = Path("data/output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(_run_translation(
        input_text=sample_text,
        title="demo_chapter",
        output_dir=output_dir,
        enable_validation=True,
        enable_human_review=False
    ))


@app.command()
def version():
    """显示版本信息"""
    console.print("\n[bold]书籍翻译智能代理[/bold]")
    console.print("版本: 1.0.0")
    console.print("基于: LangChain + LangGraph")
    console.print()


if __name__ == "__main__":
    app()
