"""
翻译Agent

负责：
1. 多步骤引导翻译（理解-分解-转换-润色）
2. 多版本生成与融合
3. 翻译质量控制
"""
import uuid
from typing import List, Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from src.models.translation import (
    TranslationUnit, TranslationVersion, TranslationNote,
    UncertainPart, Improvement, TranslationStatus, VersionType
)
from config.prompts import (
    TRANSLATION_STEP1_UNDERSTANDING,
    TRANSLATION_STEP2_DRAFT,
    TRANSLATION_STEP3_POLISH,
    MULTI_VERSION_PROMPT,
    VERSION_FUSION_PROMPT
)


class TranslatorAgent(BaseAgent):
    """翻译Agent"""
    
    def __init__(self, **kwargs):
        super().__init__(name="TranslatorAgent", **kwargs)
        self.generate_variants = self.settings.translation.generate_variants
        self.variant_count = self.settings.translation.variant_count
    
    async def run(
        self,
        source_text: str,
        terminology: Dict[str, str],
        context: str = "",
        chunk_id: str = "",
        style: str = "formal",
        target_audience: str = "general"
    ) -> TranslationUnit:
        """
        执行翻译
        
        Args:
            source_text: 源文本
            terminology: 术语表
            context: 上下文信息
            chunk_id: 对应的chunk ID
            style: 目标风格
            target_audience: 目标读者
            
        Returns:
            TranslationUnit: 翻译单元
        """
        self.update_state("running", "Starting translation", 0.0)
        
        unit_id = f"unit_{uuid.uuid4().hex[:8]}"
        
        unit = TranslationUnit(
            unit_id=unit_id,
            chunk_id=chunk_id,
            source_text=source_text,
            source_language=self.settings.translation.source_language,
            target_language=self.settings.translation.target_language,
            used_terminology=terminology
        )
        
        try:
            # 步骤1：深度理解原文
            self.update_state("running", "Step 1: Understanding", 0.2)
            understanding = await self._step1_understand(source_text, terminology, context)
            unit.understanding = understanding
            
            # 步骤2：生成初始译文
            self.update_state("running", "Step 2: Drafting", 0.4)
            draft_result = await self._step2_draft(
                source_text, understanding, terminology
            )
            draft_translation = ""
            if isinstance(draft_result, dict):
                draft_translation = draft_result.get("draft_translation", "") or draft_result.get("translation", "")
            elif isinstance(draft_result, str):
                draft_translation = draft_result
            
            # 创建初始版本
            draft_version = TranslationVersion(
                version_id=f"v_{uuid.uuid4().hex[:8]}",
                version_type=VersionType.LITERAL,
                translation=draft_translation,
                characteristics="Initial draft translation"
            )
            unit.add_version(draft_version)
            
            # 记录翻译注释
            for note_data in draft_result.get("translation_notes", []):
                note = TranslationNote(
                    source_segment=note_data.get("source_segment", ""),
                    translated_segment=note_data.get("translated_segment", ""),
                    approach=note_data.get("approach", ""),
                    confidence=note_data.get("confidence", 0.8)
                )
                unit.notes.append(note)
            
            # 记录不确定部分
            for uncertain_data in draft_result.get("uncertain_parts", []):
                uncertain = UncertainPart(
                    segment=uncertain_data.get("segment", ""),
                    options=uncertain_data.get("options", []),
                    reason=uncertain_data.get("reason", ""),
                    requires_human_review=True
                )
                unit.uncertain_parts.append(uncertain)
            
            # 步骤3：润色优化
            self.update_state("running", "Step 3: Polishing", 0.6)
            polish_result = await self._step3_polish(
                source_text,
                draft_translation,
                style,
                target_audience
            )
            polished_translation = ""
            if isinstance(polish_result, dict):
                polished_translation = polish_result.get("polished_translation", "") or polish_result.get("translation", "")
            elif isinstance(polish_result, str):
                polished_translation = polish_result
            
            # 记录改进
            for imp_data in polish_result.get("improvements", []):
                improvement = Improvement(
                    before=imp_data.get("before", ""),
                    after=imp_data.get("after", ""),
                    improvement_type=imp_data.get("improvement_type", ""),
                    reason=imp_data.get("reason", "")
                )
                unit.improvements.append(improvement)
            
            # 如果启用多版本生成
            if self.generate_variants:
                self.update_state("running", "Generating variants", 0.8)
                versions = await self._generate_variants(source_text, terminology)
                for v in versions:
                    unit.add_version(v)
                
                # 融合版本
                self.update_state("running", "Fusing versions", 0.9)
                fused_result = await self._fuse_versions(
                    source_text,
                    polished_translation,
                    [v.translation for v in versions]
                )
                
                fused_translation = fused_result.get("fused_translation", polished_translation)
                fused_version = TranslationVersion(
                    version_id=f"v_{uuid.uuid4().hex[:8]}",
                    version_type=VersionType.FUSED,
                    translation=fused_translation,
                    is_recommended=True,
                    recommendation_reason="Fused from multiple versions",
                    accuracy_score=fused_result.get("quality_assessment", {}).get("accuracy", 0.9),
                    fluency_score=fused_result.get("quality_assessment", {}).get("fluency", 0.9),
                    style_score=fused_result.get("quality_assessment", {}).get("style_match", 0.9)
                )
                unit.add_version(fused_version)
                
                final_translation = fused_translation
            else:
                final_translation = polished_translation
            
            # 兜底：如果润色/融合为空，使用起稿版；仍为空则走简易翻译
            if not final_translation.strip():
                if draft_translation.strip():
                    final_translation = draft_translation
                else:
                    final_translation = await self.translate_simple(
                        source_text, terminology
                    )
            
            # 去除模型可能输出的元说明/括注
            final_translation = self._clean_translation(final_translation)
            
            # 设置最佳翻译
            unit.set_best_translation(
                final_translation,
                confidence=polish_result.get("final_confidence", 0.9) if isinstance(polish_result, dict) else 0.9
            )
            unit.status = TranslationStatus.DRAFT
            
            self.update_state("completed", "Translation completed", 1.0)
            self.state.result = unit
            
            logger.info(f"Translation completed for chunk {chunk_id}")
            return unit
            
        except Exception as e:
            self.set_error(str(e))
            unit.status = TranslationStatus.PENDING
            raise

    def _clean_translation(self, text: str) -> str:
        """清理模型输出中的元说明括注"""
        import re
        lines = []
        pattern = re.compile(r"^[（(].{0,120}(术语表|严格遵循|口语化|按.*译|保留原文).*[）)]\s*$")
        for line in text.splitlines():
            if pattern.match(line.strip()):
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        # 去掉多余空行
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned
    
    async def _step1_understand(
        self,
        source_text: str,
        terminology: Dict[str, str],
        context: str
    ) -> Dict[str, Any]:
        """步骤1：深度理解原文"""
        prompt = TRANSLATION_STEP1_UNDERSTANDING.format(
            source_language=self.settings.translation.source_language,
            target_language=self.settings.translation.target_language,
            source_text=source_text,
            terminology=terminology,
            context=context
        )
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
    
    async def _step2_draft(
        self,
        source_text: str,
        understanding: Dict[str, Any],
        terminology: Dict[str, str]
    ) -> Dict[str, Any]:
        """步骤2：生成初始译文"""
        prompt = TRANSLATION_STEP2_DRAFT.format(
            target_language=self.settings.translation.target_language,
            source_text=source_text,
            understanding=understanding,
            terminology=terminology
        )
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
    
    async def _step3_polish(
        self,
        source_text: str,
        draft_translation: str,
        style: str,
        target_audience: str
    ) -> Dict[str, Any]:
        """步骤3：润色优化"""
        prompt = TRANSLATION_STEP3_POLISH.format(
            source_text=source_text,
            draft_translation=draft_translation,
            style=style,
            target_audience=target_audience,
            target_language=self.settings.translation.target_language
        )
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
    
    async def _generate_variants(
        self,
        source_text: str,
        terminology: Dict[str, str]
    ) -> List[TranslationVersion]:
        """生成多个翻译版本"""
        prompt = MULTI_VERSION_PROMPT.format(
            source_text=source_text,
            source_language=self.settings.translation.source_language,
            target_language=self.settings.translation.target_language,
            terminology=terminology,
            variant_count=self.variant_count
        )
        
        result = await self.invoke_llm(prompt)
        
        versions = []
        if isinstance(result, dict) and "versions" in result:
            for v_data in result["versions"]:
                version_type = self._parse_version_type(v_data.get("version_type", ""))
                version = TranslationVersion(
                    version_id=f"v_{uuid.uuid4().hex[:8]}",
                    version_type=version_type,
                    translation=v_data.get("translation", ""),
                    characteristics=v_data.get("characteristics", ""),
                    suitable_for=v_data.get("suitable_for", "")
                )
                versions.append(version)
        
        return versions
    
    async def _fuse_versions(
        self,
        source_text: str,
        base_translation: str,
        variant_translations: List[str]
    ) -> Dict[str, Any]:
        """融合多个翻译版本"""
        # 确保至少有3个版本用于融合
        versions = [base_translation] + variant_translations
        while len(versions) < 3:
            versions.append(base_translation)
        
        prompt = VERSION_FUSION_PROMPT.format(
            source_text=source_text,
            version_1=versions[0],
            version_2=versions[1] if len(versions) > 1 else versions[0],
            version_3=versions[2] if len(versions) > 2 else versions[0]
        )
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {"fused_translation": base_translation}
    
    def _parse_version_type(self, type_str: str) -> VersionType:
        """解析版本类型"""
        type_map = {
            "直译版": VersionType.LITERAL,
            "literal": VersionType.LITERAL,
            "意译版": VersionType.FREE,
            "free": VersionType.FREE,
            "风格化版": VersionType.STYLIZED,
            "stylized": VersionType.STYLIZED
        }
        return type_map.get(type_str.lower(), VersionType.LITERAL)
    
    async def translate_simple(
        self,
        source_text: str,
        terminology: Dict[str, str] = None
    ) -> str:
        """简单翻译（不经过多步骤流程）"""
        terminology = terminology or {}
        
        prompt = f"""请将以下{self.settings.translation.source_language}文本翻译成{self.settings.translation.target_language}：

原文：
{source_text}

术语表（必须使用这些翻译）：
{terminology}

请直接输出翻译结果，不需要其他解释。"""
        
        result = await self.invoke_llm(prompt, parse_json=False)
        return result if isinstance(result, str) else str(result)
