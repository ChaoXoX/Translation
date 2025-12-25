"""
翻译总指挥官 (Translation Orchestrator)

协调各Agent完成完整的翻译工作流：
1. 预处理 -> 2. 术语提取 -> 3. 翻译 -> 4. 验证 -> 5. 人机协作
"""
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field

from src.agents import (
    PreprocessorAgent,
    TerminologyAgent,
    TranslatorAgent,
    ValidatorAgent
)
from src.models.document import Document, DocumentChunk
from src.models.terminology import TerminologyDatabase
from src.models.translation import (
    TranslationResult, TranslationUnit, TranslationStatus
)
from config.settings import get_settings


class WorkflowState(BaseModel):
    """工作流状态"""
    stage: str = "idle"  # idle, preprocessing, terminology, translating, validating, human_review, completed
    progress: float = 0.0
    current_chunk: int = 0
    total_chunks: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # 各阶段耗时
    preprocessing_time: float = 0.0
    terminology_time: float = 0.0
    translation_time: float = 0.0
    validation_time: float = 0.0


class HumanReviewTask(BaseModel):
    """人工审核任务"""
    task_id: str
    task_type: str  # terminology, translation, validation
    chunk_id: str = ""
    source_text: str = ""
    current_value: Any = None
    alternatives: List[Any] = Field(default_factory=list)
    context: str = ""
    priority: str = "normal"  # low, normal, high, critical
    status: str = "pending"  # pending, completed, skipped


class TranslationOrchestrator:
    """翻译总指挥官"""
    
    def __init__(
        self,
        terminology_db: Optional[TerminologyDatabase] = None,
        human_review_callback: Optional[Callable] = None
    ):
        self.settings = get_settings()
        
        # 初始化各Agent
        self.preprocessor = PreprocessorAgent()
        self.terminology_agent = TerminologyAgent(terminology_db=terminology_db)
        self.translator = TranslatorAgent()
        self.validator = ValidatorAgent()
        
        # 状态
        self.state = WorkflowState()
        
        # 文档和结果
        self.document: Optional[Document] = None
        self.translation_result: Optional[TranslationResult] = None
        
        # 人机协作
        self.human_review_tasks: List[HumanReviewTask] = []
        self.human_review_callback = human_review_callback
        
        logger.info("TranslationOrchestrator initialized")
    
    async def run(
        self,
        input_text: str,
        title: str = "Untitled",
        enable_validation: bool = True,
        enable_human_review: bool = True
    ) -> TranslationResult:
        """
        执行完整翻译工作流
        
        Args:
            input_text: 输入文本
            title: 文档标题
            enable_validation: 是否启用回译验证
            enable_human_review: 是否启用人工审核
            
        Returns:
            TranslationResult: 翻译结果
        """
        logger.info(f"Starting translation workflow for '{title}'")
        start_time = datetime.now()
        
        try:
            # 阶段1: 预处理
            self.state.stage = "preprocessing"
            preprocessing_start = datetime.now()
            
            self.document = await self.preprocessor.run(input_text, title)
            self.state.total_chunks = self.document.total_chunks
            
            self.state.preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
            logger.info(f"Preprocessing completed: {self.document.total_chunks} chunks")
            
            # 阶段2: 术语提取
            self.state.stage = "terminology"
            terminology_start = datetime.now()
            
            await self._extract_terminology()
            
            self.state.terminology_time = (datetime.now() - terminology_start).total_seconds()
            
            # 处理术语人工审核
            if enable_human_review:
                await self._handle_terminology_review()
            
            # 阶段3: 翻译
            self.state.stage = "translating"
            translation_start = datetime.now()
            
            self.translation_result = TranslationResult(
                result_id=f"result_{self.document.doc_id}",
                doc_id=self.document.doc_id,
                terminology_used=self.terminology_agent.get_terminology_for_translation()
            )
            
            await self._translate_all_chunks()
            
            self.state.translation_time = (datetime.now() - translation_start).total_seconds()
            
            # 阶段4: 验证
            if enable_validation:
                self.state.stage = "validating"
                validation_start = datetime.now()
                
                try:
                    await self._validate_translations()
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
                    logger.warning("Validation failed but translation is complete, will save results anyway")
                    self.state.warnings.append(f"验证失败: {str(e)}")
                
                self.state.validation_time = (datetime.now() - validation_start).total_seconds()
            
            # 阶段5: 人工审核（翻译相关）
            if enable_human_review:
                self.state.stage = "human_review"
                await self._handle_translation_review()
            
            # 完成
            self.state.stage = "completed"
            self.state.progress = 1.0
            
            # 合并最终译文
            self.translation_result.merge_translations()
            self.translation_result.update_progress()
            self.translation_result.completed_at = datetime.now()
            self.translation_result.status = TranslationStatus.FINAL
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Translation workflow completed in {total_time:.1f}s")
            logger.info(f"  - Preprocessing: {self.state.preprocessing_time:.1f}s")
            logger.info(f"  - Terminology: {self.state.terminology_time:.1f}s")
            logger.info(f"  - Translation: {self.state.translation_time:.1f}s")
            logger.info(f"  - Validation: {self.state.validation_time:.1f}s")
            
            # 输出LLM调用统计
            try:
                from rich.table import Table
                table = Table(title="LLM 调用统计")
                table.add_column("Agent", justify="left")
                table.add_column("调用次数", justify="right")
                calls_pre = getattr(self.preprocessor, "llm_calls", 0)
                calls_term = getattr(self.terminology_agent, "llm_calls", 0)
                calls_trans = getattr(self.translator, "llm_calls", 0)
                calls_valid = getattr(self.validator, "llm_calls", 0)
                total_calls = calls_pre + calls_term + calls_trans + calls_valid
                table.add_row("Preprocessor", str(calls_pre))
                table.add_row("Terminology", str(calls_term))
                table.add_row("Translator", str(calls_trans))
                table.add_row("Validator", str(calls_valid))
                table.add_row("合计", str(total_calls))
                from rich.console import Console
                Console().print(table)
                logger.info(f"LLM calls - Pre:{calls_pre} Term:{calls_term} Trans:{calls_trans} Valid:{calls_valid} Total:{total_calls}")
            except Exception as _:
                # 静默失败，避免影响主流程
                pass
            
            return self.translation_result
            
        except Exception as e:
            self.state.errors.append(str(e))
            logger.error(f"Translation workflow failed: {e}")
            raise
    
    async def _extract_terminology(self):
        """提取术语"""
        # 对前几个chunk进行术语提取（建立初始术语库）
        initial_chunks = self.document.chunks[:min(5, len(self.document.chunks))]
        
        for i, chunk in enumerate(initial_chunks):
            progress = (i + 1) / len(initial_chunks) * 0.5
            self.state.progress = progress
            
            await self.terminology_agent.run(
                text=chunk.content,
                domain=self.document.metadata.domain,
                context=chunk.context.context_before
            )
        
        # 后续chunk中识别新术语
        remaining_chunks = self.document.chunks[len(initial_chunks):]
        for i, chunk in enumerate(remaining_chunks):
            progress = 0.5 + (i + 1) / len(remaining_chunks) * 0.5
            self.state.progress = progress
            
            # 只提取，不重复处理已有术语
            await self.terminology_agent.run(
                text=chunk.content,
                domain=self.document.metadata.domain,
                context=chunk.context.context_before
            )
        
        logger.info(f"Terminology extracted: {self.terminology_agent.terminology_db.total_terms} terms")
    
    async def _handle_terminology_review(self):
        """处理术语人工审核"""
        low_confidence_terms = self.terminology_agent.terminology_db.get_low_confidence_entries()
        
        for term in low_confidence_terms:
            task = HumanReviewTask(
                task_id=f"term_review_{term.term_id}",
                task_type="terminology",
                source_text=term.original,
                current_value=term.translation,
                alternatives=term.alternatives,
                context=term.context,
                priority="high" if term.confidence < 0.5 else "normal"
            )
            self.human_review_tasks.append(task)
        
        if self.human_review_tasks and self.human_review_callback:
            await self._wait_for_human_review("terminology")
    
    async def _translate_all_chunks(self):
        """翻译所有chunk"""
        terminology = self.terminology_agent.get_terminology_for_translation()
        total = len(self.document.chunks)
        
        for i, chunk in enumerate(self.document.chunks):
            self.state.current_chunk = i + 1
            self.state.progress = (i + 1) / total
            
            logger.debug(f"Translating chunk {i+1}/{total}")
            
            # 获取上下文
            context = self._get_translation_context(chunk, i)
            
            # 执行翻译
            unit = await self.translator.run(
                source_text=chunk.content,
                terminology=terminology,
                context=context,
                chunk_id=chunk.chunk_id,
                style=self.document.metadata.style,
                target_audience=self.document.metadata.target_audience
            )
            
            self.translation_result.add_unit(unit)
            self.document.mark_chunk_translated(chunk.chunk_id)
            
            # 每完成一个 chunk 保存检查点
            self._save_checkpoint()
    
    def _get_translation_context(self, chunk: DocumentChunk, index: int) -> str:
        """获取翻译上下文"""
        context_parts = []
        
        # 添加前文上下文
        if chunk.context.context_before:
            context_parts.append(f"前文摘要: {chunk.context.context_before}")
        
        # 添加前一个chunk的译文（如果有）
        if index > 0 and self.translation_result and self.translation_result.units:
            prev_unit = self.translation_result.units[-1]
            if prev_unit.best_translation:
                # 只取最后200字符作为上下文
                prev_trans = prev_unit.best_translation[-200:]
                context_parts.append(f"前文译文: ...{prev_trans}")
        
        return "\n".join(context_parts)
    
    async def _validate_translations(self):
        """验证翻译"""
        total = len(self.translation_result.units)
        
        for i, unit in enumerate(self.translation_result.units):
            self.state.progress = (i + 1) / total
            
            logger.debug(f"Validating unit {i+1}/{total}")
            
            result = await self.validator.run(unit)
            
            # 检查是否需要人工审核
            if result.get("needs_human_review"):
                task = HumanReviewTask(
                    task_id=f"trans_review_{unit.unit_id}",
                    task_type="translation",
                    chunk_id=unit.chunk_id,
                    source_text=unit.source_text,
                    current_value=unit.best_translation,
                    context=str(result.get("back_translation", {})),
                    priority="high" if unit.final_confidence < 0.7 else "normal"
                )
                self.human_review_tasks.append(task)
    
    async def _handle_translation_review(self):
        """处理翻译人工审核"""
        translation_tasks = [t for t in self.human_review_tasks if t.task_type == "translation"]
        
        if translation_tasks and self.human_review_callback:
            await self._wait_for_human_review("translation")
    
    async def _wait_for_human_review(self, review_type: str):
        """等待人工审核"""
        pending_tasks = [
            t for t in self.human_review_tasks 
            if t.task_type == review_type and t.status == "pending"
        ]
        
        if not pending_tasks:
            return
        
        logger.info(f"Waiting for human review: {len(pending_tasks)} {review_type} tasks")
        
        if self.human_review_callback:
            # 调用回调函数处理人工审核
            results = await self.human_review_callback(pending_tasks)
            
            # 处理审核结果
            for task_id, result in results.items():
                task = next((t for t in self.human_review_tasks if t.task_id == task_id), None)
                if task:
                    task.status = "completed"
                    
                    # 根据审核类型更新数据
                    if task.task_type == "terminology":
                        await self._apply_terminology_review(task, result)
                    elif task.task_type == "translation":
                        await self._apply_translation_review(task, result)
    
    async def _apply_terminology_review(self, task: HumanReviewTask, result: Dict[str, Any]):
        """应用术语审核结果"""
        if result.get("approved_translation"):
            # 更新术语库
            entry = self.terminology_agent.terminology_db.lookup(task.source_text)
            if entry:
                self.terminology_agent.terminology_db.update_entry(
                    entry.term_id,
                    translation=result["approved_translation"],
                    verified=True,
                    verified_by="human"
                )
    
    async def _apply_translation_review(self, task: HumanReviewTask, result: Dict[str, Any]):
        """应用翻译审核结果"""
        if result.get("approved_translation"):
            # 更新翻译单元
            for unit in self.translation_result.units:
                if unit.unit_id in task.task_id:
                    unit.set_best_translation(result["approved_translation"], confidence=1.0)
                    unit.status = TranslationStatus.APPROVED
                    break
    
    def get_state(self) -> WorkflowState:
        """获取工作流状态"""
        return self.state
    
    def get_human_review_tasks(self) -> List[HumanReviewTask]:
        """获取待审核任务"""
        return [t for t in self.human_review_tasks if t.status == "pending"]
    
    def get_terminology_database(self) -> TerminologyDatabase:
        """获取术语库"""
        return self.terminology_agent.terminology_db
    
    def save_terminology_database(self, filepath: Path):
        """保存术语库"""
        self.terminology_agent.terminology_db.save(filepath)
    
    def load_terminology_database(self, filepath: Path):
        """加载术语库"""
        self.terminology_agent.terminology_db = TerminologyDatabase.load(filepath)
    
    def _save_checkpoint(self):
        """保存检查点"""
        if not self.translation_result:
            return
        try:
            import pickle
            checkpoint_dir = Path("data/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / f"{self.document.doc_id}_checkpoint.pkl"
            
            checkpoint_data = {
                "document": self.document,
                "translation_result": self.translation_result,
                "terminology_db": self.terminology_agent.terminology_db,
                "state": self.state
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.debug(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, doc_id: str) -> bool:
        """加载检查点"""
        try:
            import pickle
            checkpoint_file = Path("data/checkpoints") / f"{doc_id}_checkpoint.pkl"
            if not checkpoint_file.exists():
                return False
            
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.document = checkpoint_data["document"]
            self.translation_result = checkpoint_data["translation_result"]
            self.terminology_agent.terminology_db = checkpoint_data["terminology_db"]
            self.state = checkpoint_data["state"]
            
            logger.info(f"Checkpoint loaded: {len(self.translation_result.units)} units completed")
            return True
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
