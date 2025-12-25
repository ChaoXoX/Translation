"""
示例：翻译演示

展示完整的翻译工作流
"""
import asyncio
from pathlib import Path
from loguru import logger

# 配置日志
logger.add("demo.log", level="DEBUG")

from src.workflow.orchestrator import TranslationOrchestrator
from src.workflow.pipeline import TranslationPipeline
from src.knowledge.memory import TranslationMemory, StyleGuide, ReaderProfile
from src.human_loop.interface import HumanLoopInterface


# 示例文本：说唱自传风格
SAMPLE_RAP_AUTOBIOGRAPHY = """
Chapter 1: Where I Come From

Yo, let me paint you a picture of my childhood. Growing up in Compton wasn't no fairy tale, 
feel me? The streets had their own rules, and you either learned them quick or got eaten alive.

My pops wasn't around much - he had his own demons to fight. So it was just me, my mama, 
and my little sister trying to make it. Mama worked two jobs - cleaning houses during the day 
and working at the diner at night. She was the real MVP, no cap.

"Marcus," she'd say to me, "you got a gift. Don't let these streets take it from you."

I didn't fully understand what she meant back then. All I knew was that when I picked up that 
pen and started writing rhymes, something magical happened. The pain, the struggle, the hustle - 
it all came pouring out.

My homie Dre-Dog was the first one to notice. We were just kids, maybe thirteen or fourteen, 
freestyling on the corner of Rosecrans and Wilmington. "Yo, Marcus," he said, "you spittin' 
some real talk. That's fire, bruh."

That was my first taste of validation. But the streets had other plans for me.

See, back then, everybody was either banging or hustling. The Crips and the Bloods had the 
whole neighborhood on lock. My cousin Lil' Tone got caught up in that life - caught a beef 
with some dudes from the other side of town. That situation got heated real quick.

I remember the night everything changed. Shots rang out on the block, and when the smoke 
cleared, Lil' Tone was lying on the ground. He survived, but barely. That's when I knew - 
I had to get out. The rap game wasn't just a dream anymore; it was my only way out.

"Keep your head up and your pen moving," Mama told me at the hospital. "God got plans for you."

And you know what? She was right. Three months later, I won my first rap battle at the 
community center. The crowd went crazy. For those three minutes on that stage, I wasn't 
just some broke kid from the hood. I was somebody.

That's when I met Big Mike, the local promoter. He had connections all over LA - from the 
underground clubs to the record labels downtown. "Youngin'," he said, pulling me aside after 
the battle, "you got that it factor. Let me introduce you to some people."

Little did I know, that conversation would change my life forever.

Chapter 2: The Come Up

The next few years were a blur of late nights in the studio, grinding at open mics, and 
building my reputation one bar at a time. Big Mike was true to his word - he got me gigs 
at every club in South Central.

But the music industry? That's a whole different beast. Everybody wants something from you. 
The suits at the record labels talked big, but their contracts were straight-up predatory. 
They wanted to own everything - my music, my image, my soul.

"You gotta be smart about this," Big Mike warned me. "These industry folks will chew you up 
and spit you out if you let 'em."

I learned that lesson the hard way when I signed my first deal. Man, they had me trapped in 
a contract that was basically indentured servitude. Four albums, minimal royalties, and they 
owned all my masters. Classic rookie mistake.

But I ain't gonna lie - those early days had their moments. The first time I heard my song 
on the radio? Bruh, I cried. No shame. My mama was there, and we just held each other and 
cried. All those years of struggle, all those nights wondering if we'd make rent - it felt 
like it was finally paying off.

"You did it, baby," Mama said. "You really did it."

If only it was that simple.
"""


async def demo_translation():
    """演示翻译流程"""
    print("\n" + "="*60)
    print("书籍翻译智能代理 - 演示")
    print("="*60 + "\n")
    
    # 1. 设置翻译记忆和风格
    print("[1] 配置翻译风格和读者画像...")
    
    memory = TranslationMemory()
    
    # 设置风格指南
    style_guide = StyleGuide(
        name="rap_autobiography",
        formality="casual",
        tone="authentic",
        handle_names="transliterate",
        handle_idioms="adapt",
        custom_rules=[
            "保持说唱文化的原汁原味",
            "俚语优先意译，必要时加注释",
            "人名使用音译+首次出现加英文原名"
        ]
    )
    memory.set_style_guide(style_guide)
    
    # 设置读者画像
    reader_profile = ReaderProfile(
        profile_id="rap_fans",
        audience_type="young_adult",
        age_group="young_adult",
        cultural_background="Chinese hip-hop fans",
        preferred_complexity="medium",
        preferred_style="adaptive"
    )
    memory.set_reader_profile(reader_profile)
    
    print("   ✓ 风格指南: 非正式、真实、街头风格")
    print("   ✓ 目标读者: 中国年轻嘻哈爱好者")
    
    # 2. 初始化翻译器
    print("\n[2] 初始化多智能体翻译系统...")
    
    orchestrator = TranslationOrchestrator()
    
    print("   ✓ PreprocessorAgent: 文本预处理")
    print("   ✓ TerminologyAgent: 术语实体管理")
    print("   ✓ TranslatorAgent: 翻译生成")
    print("   ✓ ValidatorAgent: 回译验证")
    
    # 3. 执行翻译
    print("\n[3] 开始翻译流程...")
    print("    输入: 说唱自传 (约2000字)")
    print("    方向: 英语 → 中文")
    
    try:
        result = await orchestrator.run(
            input_text=SAMPLE_RAP_AUTOBIOGRAPHY,
            title="Rap Autobiography Demo",
            enable_validation=True,
            enable_human_review=False
        )
        
        # 4. 显示结果
        print("\n[4] 翻译完成!")
        
        state = orchestrator.get_state()
        print(f"\n   统计信息:")
        print(f"   - 预处理耗时: {state.preprocessing_time:.1f}秒")
        print(f"   - 术语提取耗时: {state.terminology_time:.1f}秒")
        print(f"   - 翻译耗时: {state.translation_time:.1f}秒")
        print(f"   - 验证耗时: {state.validation_time:.1f}秒")
        print(f"   - 总单元数: {state.total_chunks}")
        
        # 显示术语库
        term_db = orchestrator.get_terminology_database()
        print(f"\n   术语库统计:")
        print(f"   - 命名实体: {len(term_db.named_entities)}")
        print(f"   - 领域术语: {len(term_db.domain_terms)}")
        print(f"   - 文化负载词: {len(term_db.cultural_terms)}")
        
        # 显示部分译文
        print("\n" + "="*60)
        print("译文预览 (前1000字符):")
        print("="*60)
        print(result.full_translation[:1000] if result.full_translation else "（无译文）")
        print("...")
        
        # 保存结果
        output_dir = Path("data/output/demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存译文
        with open(output_dir / "translation.txt", 'w', encoding='utf-8') as f:
            f.write(result.full_translation)
        
        # 保存术语库
        term_db.save(output_dir / "terminology.json")
        
        print(f"\n   结果已保存到: {output_dir}/")
        
    except Exception as e:
        print(f"\n   ✗ 翻译失败: {e}")
        logger.exception("Translation demo failed")
        raise
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60 + "\n")


async def demo_terminology_extraction():
    """演示术语提取"""
    print("\n" + "="*60)
    print("术语提取演示")
    print("="*60 + "\n")
    
    from src.agents.terminology import TerminologyAgent
    
    agent = TerminologyAgent()
    
    # 测试文本
    test_text = """
    My homie Dre-Dog and I were spitting bars on the corner. The beef with the Crips 
    got heated after Lil' Tone caught that L. Big Mike was the real OG who kept it 100.
    Mama always said to keep my head up and stay off the streets.
    """
    
    print("输入文本:")
    print(test_text)
    print()
    
    result = await agent.run(
        text=test_text,
        domain="entertainment"
    )
    
    print("\n提取结果:")
    print(f"- 命名实体: {result['extracted_count']['named_entities']}")
    print(f"- 领域术语: {result['extracted_count']['domain_terms']}")
    print(f"- 文化负载词: {result['extracted_count']['cultural_terms']}")
    
    print("\n术语表:")
    for original, translation in result['terminology_dict'].items():
        print(f"  {original} → {translation}")


async def demo_single_translation():
    """演示单段翻译"""
    print("\n" + "="*60)
    print("单段翻译演示")
    print("="*60 + "\n")
    
    from src.agents.translator import TranslatorAgent
    
    agent = TranslatorAgent()
    
    # 测试文本
    test_text = """
    Yo, let me paint you a picture of my childhood. Growing up in Compton wasn't no fairy tale, 
    feel me? The streets had their own rules, and you either learned them quick or got eaten alive.
    """
    
    terminology = {
        "Compton": "康普顿",
        "yo": "嘿"
    }
    
    print("原文:")
    print(test_text)
    print("\n术语表:")
    print(terminology)
    print()
    
    result = await agent.run(
        source_text=test_text,
        terminology=terminology,
        context="说唱自传开篇",
        style="casual",
        target_audience="年轻嘻哈爱好者"
    )
    
    print("\n翻译结果:")
    print(f"状态: {result.status}")
    print(f"置信度: {result.final_confidence:.1%}")
    print(f"\n译文:")
    print(result.best_translation)
    
    if result.versions:
        print(f"\n生成了 {len(result.versions)} 个版本")


if __name__ == "__main__":
    print("\n请选择演示模式:")
    print("1. 完整翻译演示")
    print("2. 术语提取演示")
    print("3. 单段翻译演示")
    
    choice = input("\n请输入选项 (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(demo_translation())
    elif choice == "2":
        asyncio.run(demo_terminology_extraction())
    elif choice == "3":
        asyncio.run(demo_single_translation())
    else:
        print("无效选项")
