"""
验证Agent

负责：
1. 回译验证 (TEaR Framework)
2. 自我修正
3. 质量评估
"""
import uuid
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from src.models.translation import (
    TranslationUnit, BackTranslationResult, RefinementResult,
    QualityAssessment, MQMScore, TranslationStatus, Correction
)
from config.prompts import (
    BACK_TRANSLATION_PROMPT,
    TEAR_ESTIMATE_PROMPT,
    TEAR_REFINE_PROMPT,
    QUALITY_ASSESSMENT_PROMPT
)


class ValidatorAgent(BaseAgent):
    """验证Agent"""
    
    def __init__(self, **kwargs):
        # 使用较低温度以获得更一致的评估
        # 验证阶段可用备选更快模型以提速
        from langchain_openai import ChatOpenAI
        from config.settings import get_settings
        _settings = get_settings()
        
        # 智能选择模型：如果是 OpenAI API 且有备用模型则用备用，否则用主模型
        is_openai_api = "openai.com" in _settings.llm.base_url.lower()
        backup_exists = _settings.llm.backup_model and _settings.llm.backup_model.strip()
        
        if is_openai_api and backup_exists:
            model_to_use = _settings.llm.backup_model
        else:
            # 第三方 API（如 DeepSeek）或无备用模型时，使用主模型
            model_to_use = _settings.llm.model_name
        
        fast_llm = ChatOpenAI(
            api_key=_settings.llm.api_key,
            base_url=_settings.llm.base_url,
            model=model_to_use,
            temperature=0.2,
            max_tokens=_settings.llm.max_tokens
        )
        super().__init__(name="ValidatorAgent", llm=fast_llm, temperature=0.2, **kwargs)
        self.back_translation_threshold = self.settings.translation.back_translation_threshold
    
    async def run(
        self,
        unit: TranslationUnit,
        source_text: str = None
    ) -> Dict[str, Any]:
        """
        验证翻译质量
        
        Args:
            unit: 翻译单元
            source_text: 原文（可选，如果unit中没有）
            
        Returns:
            验证结果
        """
        self.update_state("running", "Starting validation", 0.0)
        
        source = source_text or unit.source_text
        translation = unit.best_translation
        
        if not translation:
            self.set_error("No translation to validate")
            return {"error": "No translation to validate"}
        
        try:
            # 1. 回译
            self.update_state("running", "Back-translating", 0.25)
            back_trans_result = await self._back_translate(translation)
            
            # 2. 评估 (TEaR Estimate)
            self.update_state("running", "Estimating quality", 0.5)
            estimate_result = await self._tear_estimate(
                source,
                translation,
                back_trans_result.back_translation
            )
            
            # 更新回译结果
            back_trans_result.semantic_fidelity = estimate_result.get("scores", {}).get("semantic_fidelity", 0)
            back_trans_result.information_completeness = estimate_result.get("scores", {}).get("information_completeness", 0)
            back_trans_result.logical_consistency = estimate_result.get("scores", {}).get("logical_consistency", 0)
            back_trans_result.style_match = estimate_result.get("scores", {}).get("style_match", 0)
            back_trans_result.overall_score = estimate_result.get("overall_score", 0)
            back_trans_result.discrepancies = estimate_result.get("discrepancies", [])
            back_trans_result.assessment = estimate_result.get("assessment", "")
            back_trans_result.calculate_overall_score()
            
            # 3. 如果分数低于阈值，进行修正 (TEaR Refine)
            refinement_result = None
            if back_trans_result.overall_score < self.back_translation_threshold:
                self.update_state("running", "Refining translation", 0.75)
                refinement_result = await self._tear_refine(
                    source,
                    translation,
                    back_trans_result.discrepancies
                )
                
                # 如果有修正，更新翻译单元
                if refinement_result and refinement_result.refined_translation:
                    unit.set_best_translation(
                        refinement_result.refined_translation,
                        refinement_result.final_confidence
                    )
            
            # 4. 质量评估
            self.update_state("running", "Assessing quality", 0.9)
            quality_result = await self._assess_quality(source, unit.best_translation)
            
            # 更新单元状态
            if back_trans_result.passed or (refinement_result and refinement_result.final_confidence >= 0.85):
                unit.status = TranslationStatus.VALIDATED
            else:
                unit.status = TranslationStatus.HUMAN_REVIEW
            
            self.update_state("completed", "Validation completed", 1.0)
            
            result = {
                "back_translation": back_trans_result,
                "refinement": refinement_result,
                "quality_assessment": quality_result,
                "passed": back_trans_result.passed,
                "needs_human_review": unit.status == TranslationStatus.HUMAN_REVIEW,
                "final_translation": unit.best_translation
            }
            
            self.state.result = result
            return result
            
        except Exception as e:
            self.set_error(str(e))
            raise
    
    async def _back_translate(self, translation: str) -> BackTranslationResult:
        """执行回译"""
        prompt = BACK_TRANSLATION_PROMPT.format(
            target_language=self.settings.translation.target_language,
            source_language=self.settings.translation.source_language,
            translation=translation
        )
        
        result = await self.invoke_llm(prompt)
        
        back_translation = ""
        if isinstance(result, dict):
            back_translation = result.get("back_translation", "")
        elif isinstance(result, str):
            back_translation = result
        
        return BackTranslationResult(
            result_id=f"bt_{uuid.uuid4().hex[:8]}",
            unit_id="",  # Will be set by caller
            original_translation=translation,
            back_translation=back_translation
        )
    
    async def _tear_estimate(
        self,
        source_text: str,
        translation: str,
        back_translation: str
    ) -> Dict[str, Any]:
        """TEaR Estimate阶段：评估翻译质量"""
        prompt = TEAR_ESTIMATE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            back_translation=back_translation
        )
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
    
    async def _tear_refine(
        self,
        source_text: str,
        translation: str,
        discrepancies: list
    ) -> RefinementResult:
        """TEaR Refine阶段：修正翻译"""
        prompt = TEAR_REFINE_PROMPT.format(
            source_text=source_text,
            translation=translation,
            discrepancies=discrepancies
        )
        
        result = await self.invoke_llm(prompt)
        
        refinement = RefinementResult(
            result_id=f"rf_{uuid.uuid4().hex[:8]}",
            unit_id="",
            original_translation=translation,
            refined_translation=""
        )
        
        if isinstance(result, dict):
            refinement.refined_translation = result.get("refined_translation", translation)
            
            for corr_data in result.get("corrections", []):
                correction = Correction(
                    original_issue=corr_data.get("original_issue", ""),
                    before=corr_data.get("before", ""),
                    after=corr_data.get("after", ""),
                    correction_type=corr_data.get("correction_type", "")
                )
                refinement.corrections.append(correction)
            
            refinement.remaining_concerns = result.get("remaining_concerns", [])
            refinement.final_confidence = result.get("final_confidence", 0.9)
        
        return refinement
    
    async def _assess_quality(
        self,
        source_text: str,
        translation: str
    ) -> QualityAssessment:
        """使用MQM框架评估翻译质量"""
        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            source_text=source_text,
            translation=translation
        )
        
        result = await self.invoke_llm(prompt)
        
        assessment = QualityAssessment(
            assessment_id=f"qa_{uuid.uuid4().hex[:8]}",
            unit_id="",
            source_text=source_text,
            translation=translation
        )
        
        if isinstance(result, dict):
            mqm_scores = result.get("mqm_scores", {})
            
            # 解析各维度评分
            for dimension in ["accuracy", "fluency", "terminology", "style", "locale"]:
                dim_data = mqm_scores.get(dimension, {})
                score = MQMScore(
                    score=dim_data.get("score", 0) if isinstance(dim_data, dict) else 0,
                    issues=dim_data.get("issues", []) if isinstance(dim_data, dict) else []
                )
                setattr(assessment, dimension, score)
            
            assessment.overall_score = result.get("overall_score", 0)
            assessment.summary = result.get("summary", "")
            assessment.improvement_suggestions = result.get("improvement_suggestions", [])
        
        assessment.calculate_overall_score()
        return assessment
    
    async def validate_terminology_consistency(
        self,
        translation: str,
        terminology: Dict[str, str]
    ) -> Dict[str, Any]:
        """验证术语一致性"""
        prompt = f"""请检查以下译文中的术语使用是否与术语表一致：

译文：
{translation}

术语表：
{terminology}

请检查：
1. 术语表中的术语是否都被正确使用
2. 是否有术语被错误翻译
3. 是否有遗漏的术语

输出格式 (JSON):
```json
{{
    "consistent_terms": ["正确使用的术语"],
    "inconsistent_terms": [
        {{
            "original": "原术语",
            "expected": "期望翻译",
            "found": "实际翻译",
            "location": "位置描述"
        }}
    ],
    "missing_terms": ["未翻译的术语"],
    "overall_consistency": 0.95
}}
```"""
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
    
    async def compare_versions(
        self,
        source_text: str,
        versions: list,
        criteria: str = "综合质量"
    ) -> Dict[str, Any]:
        """比较多个翻译版本"""
        prompt = f"""请比较以下翻译版本，并评选最佳版本：

原文：
{source_text}

翻译版本：
{chr(10).join([f"版本{i+1}: {v}" for i, v in enumerate(versions)])}

评选标准：{criteria}

请从准确性、流畅性、风格匹配度三个维度评估每个版本，并推荐最佳版本。

输出格式 (JSON):
```json
{{
    "evaluations": [
        {{
            "version": 1,
            "accuracy": 9.0,
            "fluency": 8.5,
            "style": 8.0,
            "overall": 8.5,
            "strengths": ["优点"],
            "weaknesses": ["缺点"]
        }}
    ],
    "best_version": 1,
    "recommendation_reason": "推荐理由"
}}
```"""
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
