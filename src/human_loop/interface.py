"""
人机协作接口

提供人工审核的交互界面和任务管理
"""
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from loguru import logger

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown


class ReviewType(str, Enum):
    """审核类型"""
    TERMINOLOGY = "terminology"
    TRANSLATION = "translation"
    VALIDATION = "validation"


class ReviewPriority(str, Enum):
    """审核优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewTask(BaseModel):
    """审核任务"""
    task_id: str
    review_type: ReviewType
    
    # 内容
    source_text: str
    current_translation: str
    alternatives: List[str] = Field(default_factory=list)
    
    # 上下文
    context: str = ""
    chunk_id: str = ""
    
    # 元数据
    confidence: float = 0.0
    priority: ReviewPriority = ReviewPriority.NORMAL
    
    # 状态
    status: str = "pending"  # pending, completed, skipped
    assigned_to: str = ""
    
    # 时间
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class ReviewResult(BaseModel):
    """审核结果"""
    task_id: str
    action: str  # approve, modify, reject, skip
    
    # 修改后的翻译（如果有）
    approved_translation: str = ""
    
    # 反馈
    feedback: str = ""
    
    # 元数据
    reviewer: str = "human"
    reviewed_at: datetime = Field(default_factory=datetime.now)


class HumanLoopInterface:
    """人机协作接口"""
    
    def __init__(self, interactive: bool = True):
        """
        初始化人机协作接口
        
        Args:
            interactive: 是否使用交互式界面
        """
        self.interactive = interactive
        self.console = Console()
        self.pending_tasks: List[ReviewTask] = []
        self.completed_results: Dict[str, ReviewResult] = {}
        
        # 任务队列文件（用于非交互模式）
        self.task_queue_file: Optional[Path] = None
        self.result_file: Optional[Path] = None
    
    def add_task(self, task: ReviewTask):
        """添加审核任务"""
        self.pending_tasks.append(task)
        logger.info(f"Review task added: {task.task_id} ({task.review_type})")
    
    def add_tasks(self, tasks: List[ReviewTask]):
        """批量添加审核任务"""
        self.pending_tasks.extend(tasks)
        logger.info(f"Added {len(tasks)} review tasks")
    
    async def process_tasks(self) -> Dict[str, ReviewResult]:
        """
        处理所有待审核任务
        
        Returns:
            审核结果字典
        """
        if not self.pending_tasks:
            logger.info("No pending tasks to review")
            return {}
        
        if self.interactive:
            return await self._process_interactive()
        else:
            return await self._process_batch()
    
    async def _process_interactive(self) -> Dict[str, ReviewResult]:
        """交互式处理任务"""
        results = {}
        
        # 按优先级排序
        sorted_tasks = sorted(
            self.pending_tasks,
            key=lambda t: (
                {"critical": 0, "high": 1, "normal": 2, "low": 3}.get(t.priority.value, 2),
                t.created_at
            )
        )
        
        self.console.print("\n[bold blue]═══ 人工审核任务 ═══[/bold blue]\n")
        self.console.print(f"共有 [bold]{len(sorted_tasks)}[/bold] 个任务待审核\n")
        
        for i, task in enumerate(sorted_tasks, 1):
            self.console.print(f"\n[bold cyan]任务 {i}/{len(sorted_tasks)}[/bold cyan]")
            result = await self._review_single_task(task)
            
            if result:
                results[task.task_id] = result
                self.completed_results[task.task_id] = result
                task.status = "completed"
                task.completed_at = datetime.now()
            
            # 询问是否继续
            if i < len(sorted_tasks):
                if not Confirm.ask("\n继续下一个任务?", default=True):
                    self.console.print("[yellow]审核暂停，剩余任务稍后处理[/yellow]")
                    break
        
        # 移除已完成的任务
        self.pending_tasks = [t for t in self.pending_tasks if t.status == "pending"]
        
        return results
    
    async def _review_single_task(self, task: ReviewTask) -> Optional[ReviewResult]:
        """审核单个任务"""
        # 显示任务信息
        self._display_task(task)
        
        # 根据任务类型处理
        if task.review_type == ReviewType.TERMINOLOGY:
            return await self._review_terminology(task)
        elif task.review_type == ReviewType.TRANSLATION:
            return await self._review_translation(task)
        else:
            return await self._review_validation(task)
    
    def _display_task(self, task: ReviewTask):
        """显示任务信息"""
        priority_colors = {
            "critical": "red",
            "high": "yellow",
            "normal": "white",
            "low": "dim"
        }
        color = priority_colors.get(task.priority.value, "white")
        
        # 创建信息表格
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        
        table.add_row("类型", task.review_type.value)
        table.add_row("优先级", f"[{color}]{task.priority.value}[/{color}]")
        table.add_row("置信度", f"{task.confidence:.1%}")
        
        self.console.print(Panel(table, title="[bold]任务信息[/bold]"))
        
        # 显示原文
        self.console.print(Panel(
            task.source_text,
            title="[bold green]原文[/bold green]",
            border_style="green"
        ))
        
        # 显示当前翻译
        self.console.print(Panel(
            task.current_translation,
            title="[bold yellow]当前翻译[/bold yellow]",
            border_style="yellow"
        ))
        
        # 显示上下文（如果有）
        if task.context:
            self.console.print(Panel(
                task.context,
                title="[bold dim]上下文[/bold dim]",
                border_style="dim"
            ))
        
        # 显示备选方案（如果有）
        if task.alternatives:
            alt_text = "\n".join([f"{i+1}. {alt}" for i, alt in enumerate(task.alternatives)])
            self.console.print(Panel(
                alt_text,
                title="[bold blue]备选翻译[/bold blue]",
                border_style="blue"
            ))
    
    async def _review_terminology(self, task: ReviewTask) -> Optional[ReviewResult]:
        """审核术语"""
        self.console.print("\n[bold]请选择操作：[/bold]")
        self.console.print("  1. 接受当前翻译")
        self.console.print("  2. 选择备选方案")
        self.console.print("  3. 输入自定义翻译")
        self.console.print("  4. 跳过")
        
        choice = Prompt.ask("选择", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            return ReviewResult(
                task_id=task.task_id,
                action="approve",
                approved_translation=task.current_translation
            )
        elif choice == "2" and task.alternatives:
            alt_choice = Prompt.ask(
                "选择备选方案编号",
                choices=[str(i+1) for i in range(len(task.alternatives))],
                default="1"
            )
            idx = int(alt_choice) - 1
            return ReviewResult(
                task_id=task.task_id,
                action="modify",
                approved_translation=task.alternatives[idx]
            )
        elif choice == "3":
            custom = Prompt.ask("输入自定义翻译")
            return ReviewResult(
                task_id=task.task_id,
                action="modify",
                approved_translation=custom
            )
        else:
            return ReviewResult(
                task_id=task.task_id,
                action="skip"
            )
    
    async def _review_translation(self, task: ReviewTask) -> Optional[ReviewResult]:
        """审核翻译"""
        self.console.print("\n[bold]请选择操作：[/bold]")
        self.console.print("  1. 接受当前翻译")
        self.console.print("  2. 修改翻译")
        self.console.print("  3. 拒绝（标记需要重新翻译）")
        self.console.print("  4. 跳过")
        
        choice = Prompt.ask("选择", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            return ReviewResult(
                task_id=task.task_id,
                action="approve",
                approved_translation=task.current_translation
            )
        elif choice == "2":
            self.console.print("\n[dim]提示：可以复制当前翻译进行修改[/dim]")
            custom = Prompt.ask("输入修改后的翻译")
            feedback = Prompt.ask("修改原因（可选）", default="")
            return ReviewResult(
                task_id=task.task_id,
                action="modify",
                approved_translation=custom,
                feedback=feedback
            )
        elif choice == "3":
            feedback = Prompt.ask("请说明拒绝原因")
            return ReviewResult(
                task_id=task.task_id,
                action="reject",
                feedback=feedback
            )
        else:
            return ReviewResult(
                task_id=task.task_id,
                action="skip"
            )
    
    async def _review_validation(self, task: ReviewTask) -> Optional[ReviewResult]:
        """审核验证结果"""
        return await self._review_translation(task)
    
    async def _process_batch(self) -> Dict[str, ReviewResult]:
        """批量处理任务（非交互模式）"""
        # 导出任务到文件
        if self.task_queue_file:
            self._export_tasks(self.task_queue_file)
            logger.info(f"Tasks exported to {self.task_queue_file}")
            
            # 等待结果文件
            if self.result_file and self.result_file.exists():
                return self._import_results(self.result_file)
        
        return {}
    
    def _export_tasks(self, filepath: Path):
        """导出任务到文件"""
        tasks_data = [
            {
                "task_id": t.task_id,
                "review_type": t.review_type.value,
                "source_text": t.source_text,
                "current_translation": t.current_translation,
                "alternatives": t.alternatives,
                "context": t.context,
                "confidence": t.confidence,
                "priority": t.priority.value
            }
            for t in self.pending_tasks
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    
    def _import_results(self, filepath: Path) -> Dict[str, ReviewResult]:
        """从文件导入结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        results = {}
        for data in results_data:
            result = ReviewResult(**data)
            results[result.task_id] = result
        
        return results
    
    def export_review_template(self, filepath: Path):
        """导出审核模板（用于人工填写）"""
        template = {
            "instructions": "请在下方填写审核结果，将action设为approve/modify/reject/skip",
            "tasks": []
        }
        
        for task in self.pending_tasks:
            template["tasks"].append({
                "task_id": task.task_id,
                "review_type": task.review_type.value,
                "source_text": task.source_text,
                "current_translation": task.current_translation,
                "alternatives": task.alternatives,
                "context": task.context,
                "confidence": task.confidence,
                "# 以下为需要填写的内容": "",
                "action": "approve",  # approve/modify/reject/skip
                "approved_translation": task.current_translation,
                "feedback": ""
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Review template exported to {filepath}")
    
    def import_review_results(self, filepath: Path) -> Dict[str, ReviewResult]:
        """导入审核结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        for task_data in data.get("tasks", []):
            result = ReviewResult(
                task_id=task_data["task_id"],
                action=task_data.get("action", "skip"),
                approved_translation=task_data.get("approved_translation", ""),
                feedback=task_data.get("feedback", "")
            )
            results[result.task_id] = result
        
        return results
    
    def get_pending_tasks(self) -> List[ReviewTask]:
        """获取待审核任务"""
        return self.pending_tasks
    
    def get_completed_results(self) -> Dict[str, ReviewResult]:
        """获取已完成结果"""
        return self.completed_results
    
    def clear_completed(self):
        """清除已完成结果"""
        self.completed_results = {}
    
    def display_summary(self):
        """显示审核摘要"""
        self.console.print("\n[bold blue]═══ 审核摘要 ═══[/bold blue]\n")
        
        table = Table()
        table.add_column("状态", style="bold")
        table.add_column("数量")
        
        pending = len(self.pending_tasks)
        completed = len(self.completed_results)
        
        table.add_row("待审核", str(pending))
        table.add_row("已完成", str(completed))
        
        self.console.print(table)
        
        if self.completed_results:
            # 统计操作类型
            actions = {}
            for result in self.completed_results.values():
                actions[result.action] = actions.get(result.action, 0) + 1
            
            self.console.print("\n[bold]操作统计：[/bold]")
            for action, count in actions.items():
                self.console.print(f"  {action}: {count}")


async def create_human_review_callback(
    interface: HumanLoopInterface
) -> Callable:
    """
    创建人工审核回调函数
    
    Args:
        interface: 人机协作接口
        
    Returns:
        回调函数
    """
    async def callback(tasks: List[Any]) -> Dict[str, Any]:
        # 转换任务格式
        review_tasks = []
        for task in tasks:
            review_task = ReviewTask(
                task_id=task.task_id if hasattr(task, 'task_id') else task.get("task_id", ""),
                review_type=ReviewType.TERMINOLOGY if "term" in str(task) else ReviewType.TRANSLATION,
                source_text=task.source_text if hasattr(task, 'source_text') else task.get("source_text", ""),
                current_translation=task.current_value if hasattr(task, 'current_value') else task.get("current_value", ""),
                alternatives=task.alternatives if hasattr(task, 'alternatives') else task.get("alternatives", []),
                context=task.context if hasattr(task, 'context') else task.get("context", ""),
                confidence=task.confidence if hasattr(task, 'confidence') else task.get("confidence", 0.5),
                priority=ReviewPriority.HIGH if "high" in str(task) else ReviewPriority.NORMAL
            )
            review_tasks.append(review_task)
        
        interface.add_tasks(review_tasks)
        results = await interface.process_tasks()
        
        # 转换结果格式
        return {
            task_id: {
                "approved_translation": result.approved_translation,
                "action": result.action,
                "feedback": result.feedback
            }
            for task_id, result in results.items()
        }
    
    return callback
