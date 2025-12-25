"""
翻译流水线 (Translation Pipeline)

基于LangGraph的多智能体协作工作流实现
提供更细粒度的流程控制和状态管理
"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from enum import Enum
import operator
from datetime import datetime
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, using simplified pipeline")

from src.agents import (
    PreprocessorAgent,
    TerminologyAgent,
    TranslatorAgent,
    ValidatorAgent
)
from src.models.document import Document, DocumentChunk
from src.models.terminology import TerminologyDatabase
from src.models.translation import TranslationResult, TranslationUnit, TranslationStatus


class PipelineStage(str, Enum):
    """流水线阶段"""
    INIT = "init"
    PREPROCESS = "preprocess"
    EXTRACT_TERMINOLOGY = "extract_terminology"
    HUMAN_REVIEW_TERMINOLOGY = "human_review_terminology"
    TRANSLATE = "translate"
    VALIDATE = "validate"
    HUMAN_REVIEW_TRANSLATION = "human_review_translation"
    FINALIZE = "finalize"
    COMPLETE = "complete"


class PipelineState(TypedDict):
    """流水线状态"""
    # 输入
    input_text: str
    title: str
    
    # 配置
    enable_validation: bool
    enable_human_review: bool
    
    # 中间状态
    stage: str
    document: Optional[Document]
    terminology_db: Optional[TerminologyDatabase]
    translation_result: Optional[TranslationResult]
    current_chunk_index: int
    
    # 人工审核
    pending_reviews: List[Dict[str, Any]]
    review_responses: Dict[str, Any]
    
    # 输出
    final_translation: str
    errors: List[str]
    
    # 时间追踪
    stage_times: Dict[str, float]


class TranslationPipeline:
    """翻译流水线"""
    
    def __init__(self):
        # 初始化Agents
        self.preprocessor = PreprocessorAgent()
        self.terminology_agent = TerminologyAgent()
        self.translator = TranslatorAgent()
        self.validator = ValidatorAgent()
        
        # 构建图（如果LangGraph可用）
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
            self.checkpointer = MemorySaver()
        else:
            self.graph = None
            self.checkpointer = None
        
        logger.info("TranslationPipeline initialized")
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""
        # 定义状态图
        workflow = StateGraph(PipelineState)
        
        # 添加节点
        workflow.add_node("preprocess", self._node_preprocess)
        workflow.add_node("extract_terminology", self._node_extract_terminology)
        workflow.add_node("human_review_terminology", self._node_human_review_terminology)
        workflow.add_node("translate", self._node_translate)
        workflow.add_node("validate", self._node_validate)
        workflow.add_node("human_review_translation", self._node_human_review_translation)
        workflow.add_node("finalize", self._node_finalize)
        
        # 设置入口点
        workflow.set_entry_point("preprocess")
        
        # 添加边
        workflow.add_edge("preprocess", "extract_terminology")
        workflow.add_conditional_edges(
            "extract_terminology",
            self._should_review_terminology,
            {
                "review": "human_review_terminology",
                "skip": "translate"
            }
        )
        workflow.add_edge("human_review_terminology", "translate")
        workflow.add_conditional_edges(
            "translate",
            self._should_validate,
            {
                "validate": "validate",
                "skip": "finalize"
            }
        )
        workflow.add_conditional_edges(
            "validate",
            self._should_review_translation,
            {
                "review": "human_review_translation",
                "skip": "finalize"
            }
        )
        workflow.add_edge("human_review_translation", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _node_preprocess(self, state: PipelineState) -> PipelineState:
        """预处理节点"""
        logger.info("Pipeline: Preprocessing")
        start_time = datetime.now()
        
        try:
            document = await self.preprocessor.run(
                state["input_text"],
                state["title"]
            )
            state["document"] = document
            state["stage"] = PipelineStage.PREPROCESS.value
            
        except Exception as e:
            state["errors"].append(f"Preprocess error: {str(e)}")
            logger.error(f"Preprocess failed: {e}")
        
        state["stage_times"]["preprocess"] = (datetime.now() - start_time).total_seconds()
        return state
    
    async def _node_extract_terminology(self, state: PipelineState) -> PipelineState:
        """术语提取节点"""
        logger.info("Pipeline: Extracting terminology")
        start_time = datetime.now()
        
        try:
            document = state["document"]
            
            # 提取术语
            for chunk in document.chunks[:5]:  # 先处理前5个chunk
                await self.terminology_agent.run(
                    text=chunk.content,
                    domain=document.metadata.domain
                )
            
            state["terminology_db"] = self.terminology_agent.terminology_db
            state["stage"] = PipelineStage.EXTRACT_TERMINOLOGY.value
            
            # 检查是否有需要审核的术语
            low_conf = self.terminology_agent.terminology_db.get_low_confidence_entries()
            if low_conf:
                state["pending_reviews"] = [
                    {
                        "type": "terminology",
                        "term_id": term.term_id,
                        "original": term.original,
                        "translation": term.translation,
                        "confidence": term.confidence
                    }
                    for term in low_conf
                ]
                
        except Exception as e:
            state["errors"].append(f"Terminology extraction error: {str(e)}")
            logger.error(f"Terminology extraction failed: {e}")
        
        state["stage_times"]["terminology"] = (datetime.now() - start_time).total_seconds()
        return state
    
    async def _node_human_review_terminology(self, state: PipelineState) -> PipelineState:
        """术语人工审核节点"""
        logger.info("Pipeline: Waiting for terminology review")
        state["stage"] = PipelineStage.HUMAN_REVIEW_TERMINOLOGY.value
        
        # 这里会暂停等待人工输入
        # 在实际实现中，可以通过回调或轮询获取用户输入
        
        # 处理审核结果
        if state.get("review_responses"):
            for term_id, response in state["review_responses"].items():
                if response.get("approved_translation"):
                    self.terminology_agent.terminology_db.update_entry(
                        term_id,
                        translation=response["approved_translation"],
                        verified=True,
                        verified_by="human"
                    )
            
            state["review_responses"] = {}
            state["pending_reviews"] = []
        
        return state
    
    async def _node_translate(self, state: PipelineState) -> PipelineState:
        """翻译节点"""
        logger.info("Pipeline: Translating")
        start_time = datetime.now()
        
        try:
            document = state["document"]
            terminology = self.terminology_agent.get_terminology_for_translation()
            
            # 初始化翻译结果
            if not state.get("translation_result"):
                state["translation_result"] = TranslationResult(
                    result_id=f"result_{document.doc_id}",
                    doc_id=document.doc_id,
                    terminology_used=terminology
                )
            
            result = state["translation_result"]
            
            # 从上次位置继续翻译
            start_index = state.get("current_chunk_index", 0)
            
            for i, chunk in enumerate(document.chunks[start_index:], start=start_index):
                logger.debug(f"Translating chunk {i+1}/{len(document.chunks)}")
                
                unit = await self.translator.run(
                    source_text=chunk.content,
                    terminology=terminology,
                    context="",
                    chunk_id=chunk.chunk_id,
                    style=document.metadata.style,
                    target_audience=document.metadata.target_audience
                )
                
                result.add_unit(unit)
                state["current_chunk_index"] = i + 1
            
            state["stage"] = PipelineStage.TRANSLATE.value
            
        except Exception as e:
            state["errors"].append(f"Translation error: {str(e)}")
            logger.error(f"Translation failed: {e}")
        
        state["stage_times"]["translation"] = (datetime.now() - start_time).total_seconds()
        return state
    
    async def _node_validate(self, state: PipelineState) -> PipelineState:
        """验证节点"""
        logger.info("Pipeline: Validating")
        start_time = datetime.now()
        
        try:
            result = state["translation_result"]
            pending_reviews = []
            
            for unit in result.units:
                validation_result = await self.validator.run(unit)
                
                if validation_result.get("needs_human_review"):
                    pending_reviews.append({
                        "type": "translation",
                        "unit_id": unit.unit_id,
                        "source": unit.source_text,
                        "translation": unit.best_translation,
                        "confidence": unit.final_confidence
                    })
            
            state["pending_reviews"].extend(pending_reviews)
            state["stage"] = PipelineStage.VALIDATE.value
            
        except Exception as e:
            state["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Validation failed: {e}")
        
        state["stage_times"]["validation"] = (datetime.now() - start_time).total_seconds()
        return state
    
    async def _node_human_review_translation(self, state: PipelineState) -> PipelineState:
        """翻译人工审核节点"""
        logger.info("Pipeline: Waiting for translation review")
        state["stage"] = PipelineStage.HUMAN_REVIEW_TRANSLATION.value
        
        # 处理审核结果
        if state.get("review_responses"):
            result = state["translation_result"]
            
            for unit_id, response in state["review_responses"].items():
                if response.get("approved_translation"):
                    for unit in result.units:
                        if unit.unit_id == unit_id:
                            unit.set_best_translation(
                                response["approved_translation"],
                                confidence=1.0
                            )
                            unit.status = TranslationStatus.APPROVED
                            break
            
            state["review_responses"] = {}
            state["pending_reviews"] = [
                r for r in state["pending_reviews"]
                if r.get("type") != "translation"
            ]
        
        return state
    
    async def _node_finalize(self, state: PipelineState) -> PipelineState:
        """完成节点"""
        logger.info("Pipeline: Finalizing")
        
        result = state["translation_result"]
        if result:
            result.merge_translations()
            result.update_progress()
            result.completed_at = datetime.now()
            result.status = TranslationStatus.FINAL
            
            state["final_translation"] = result.full_translation
        
        state["stage"] = PipelineStage.COMPLETE.value
        return state
    
    def _should_review_terminology(self, state: PipelineState) -> str:
        """决定是否需要术语审核"""
        if state.get("enable_human_review") and state.get("pending_reviews"):
            term_reviews = [r for r in state["pending_reviews"] if r.get("type") == "terminology"]
            if term_reviews:
                return "review"
        return "skip"
    
    def _should_validate(self, state: PipelineState) -> str:
        """决定是否需要验证"""
        return "validate" if state.get("enable_validation") else "skip"
    
    def _should_review_translation(self, state: PipelineState) -> str:
        """决定是否需要翻译审核"""
        if state.get("enable_human_review") and state.get("pending_reviews"):
            trans_reviews = [r for r in state["pending_reviews"] if r.get("type") == "translation"]
            if trans_reviews:
                return "review"
        return "skip"
    
    async def run(
        self,
        input_text: str,
        title: str = "Untitled",
        enable_validation: bool = True,
        enable_human_review: bool = False,
        thread_id: str = "default"
    ) -> TranslationResult:
        """
        运行翻译流水线
        
        Args:
            input_text: 输入文本
            title: 文档标题
            enable_validation: 是否启用验证
            enable_human_review: 是否启用人工审核
            thread_id: 线程ID（用于状态持久化）
            
        Returns:
            TranslationResult: 翻译结果
        """
        initial_state: PipelineState = {
            "input_text": input_text,
            "title": title,
            "enable_validation": enable_validation,
            "enable_human_review": enable_human_review,
            "stage": PipelineStage.INIT.value,
            "document": None,
            "terminology_db": None,
            "translation_result": None,
            "current_chunk_index": 0,
            "pending_reviews": [],
            "review_responses": {},
            "final_translation": "",
            "errors": [],
            "stage_times": {}
        }
        
        if LANGGRAPH_AVAILABLE and self.graph:
            # 使用LangGraph执行
            config = {"configurable": {"thread_id": thread_id}}
            
            async for state in self.graph.astream(initial_state, config=config):
                logger.debug(f"Pipeline state: {list(state.keys())}")
            
            # 获取最终状态
            final_state = state
            return final_state.get("translation_result")
        else:
            # 使用简化流程执行
            return await self._run_simple(initial_state)
    
    async def _run_simple(self, state: PipelineState) -> TranslationResult:
        """简化执行流程（不使用LangGraph）"""
        state = await self._node_preprocess(state)
        state = await self._node_extract_terminology(state)
        state = await self._node_translate(state)
        
        if state.get("enable_validation"):
            state = await self._node_validate(state)
        
        state = await self._node_finalize(state)
        
        return state.get("translation_result")
    
    async def resume(
        self,
        thread_id: str,
        review_responses: Dict[str, Any]
    ) -> TranslationResult:
        """
        恢复暂停的流水线（在人工审核后）
        
        Args:
            thread_id: 线程ID
            review_responses: 审核响应
            
        Returns:
            TranslationResult: 翻译结果
        """
        if not LANGGRAPH_AVAILABLE or not self.graph:
            raise NotImplementedError("Resume requires LangGraph")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 获取当前状态
        current_state = self.graph.get_state(config)
        
        # 更新审核响应
        current_state["review_responses"] = review_responses
        
        # 继续执行
        async for state in self.graph.astream(current_state, config=config):
            logger.debug(f"Pipeline resumed, state: {list(state.keys())}")
        
        return state.get("translation_result")
