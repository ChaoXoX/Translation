"""
术语与实体Agent

负责：
1. 命名实体识别 (NER)
2. 领域术语提取与规范化
3. 文化负载词识别与翻译策略建议
4. 全局一致性控制
"""
import uuid
from typing import List, Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from src.models.terminology import (
    TerminologyDatabase, NamedEntity, DomainTerm, CulturalTerm,
    EntityType, TranslationStrategy
)
from config.prompts import (
    TERMINOLOGY_EXTRACTION_PROMPT,
    TERMINOLOGY_VERIFICATION_PROMPT,
    SLANG_LOOKUP_PROMPT
)


class TerminologyAgent(BaseAgent):
    """术语与实体Agent"""
    
    def __init__(
        self, 
        terminology_db: Optional[TerminologyDatabase] = None,
        **kwargs
    ):
        super().__init__(name="TerminologyAgent", **kwargs)
        
        # 初始化或使用现有术语库
        if terminology_db:
            self.terminology_db = terminology_db
        else:
            self.terminology_db = TerminologyDatabase(
                db_id=str(uuid.uuid4()),
                project_name="translation_project",
                source_language=self.settings.translation.source_language,
                target_language=self.settings.translation.target_language
            )
    
    async def run(
        self, 
        text: str, 
        domain: str = "general",
        context: str = ""
    ) -> Dict[str, Any]:
        """
        提取并处理术语
        
        Args:
            text: 待处理文本
            domain: 领域
            context: 上下文信息
            
        Returns:
            处理结果
        """
        self.update_state("running", "Extracting terminology", 0.0)
        
        try:
            # 1. 提取术语
            self.update_state("running", "Extracting terms", 0.3)
            extracted = await self._extract_terminology(text, domain)
            
            # 2. 处理命名实体
            self.update_state("running", "Processing named entities", 0.5)
            for entity_data in extracted.get("named_entities", []):
                await self._process_named_entity(entity_data)
            
            # 3. 处理领域术语
            self.update_state("running", "Processing domain terms", 0.6)
            for term_data in extracted.get("domain_terms", []):
                await self._process_domain_term(term_data, domain)
            
            # 4. 处理文化负载词
            self.update_state("running", "Processing cultural terms", 0.7)
            for term_data in extracted.get("cultural_terms", []):
                await self._process_cultural_term(term_data)
            
            # 5. 验证术语
            self.update_state("running", "Verifying terminology", 0.9)
            verification_result = await self._verify_new_terms()
            
            self.update_state("completed", "Terminology extraction completed", 1.0)
            
            result = {
                "extracted_count": {
                    "named_entities": len(extracted.get("named_entities", [])),
                    "domain_terms": len(extracted.get("domain_terms", [])),
                    "cultural_terms": len(extracted.get("cultural_terms", []))
                },
                "database_stats": {
                    "total_terms": self.terminology_db.total_terms,
                    "verified_terms": self.terminology_db.verified_terms
                },
                "needs_human_review": self.terminology_db.get_low_confidence_entries(),
                "terminology_dict": self.terminology_db.to_dict_for_prompt()
            }
            
            self.state.result = result
            return result
            
        except Exception as e:
            self.set_error(str(e))
            raise
    
    async def _extract_terminology(
        self, 
        text: str, 
        domain: str
    ) -> Dict[str, Any]:
        """提取术语"""
        prompt = TERMINOLOGY_EXTRACTION_PROMPT.format(
            text=text,
            domain=domain,
            source_language=self.settings.translation.source_language,
            target_language=self.settings.translation.target_language
        )
        
        result = await self.invoke_llm(prompt)
        return result if isinstance(result, dict) else {}
    
    async def _process_named_entity(self, entity_data: Dict[str, Any]) -> Optional[NamedEntity]:
        """处理命名实体"""
        original = entity_data.get("original", "")
        if not original:
            return None
        
        # 检查是否已存在
        existing = self.terminology_db.lookup(original)
        if existing:
            # 更新出现次数
            existing.occurrence_count += 1
            if entity_data.get("context"):
                existing.contexts.append(entity_data.get("context"))
            return existing
        
        # 创建新实体
        entity_type = self._parse_entity_type(entity_data.get("type", "OTHER"))
        
        entity = NamedEntity(
            term_id=f"entity_{uuid.uuid4().hex[:8]}",
            original=original,
            translation=entity_data.get("suggested_translation", original),
            alternatives=entity_data.get("alternatives", []),
            context=entity_data.get("context", ""),
            entity_type=entity_type,
            confidence=entity_data.get("confidence", 0.8),
            notes=entity_data.get("notes", ""),
            source="auto_extracted"
        )
        
        self.terminology_db.add_named_entity(entity)
        logger.debug(f"Added named entity: {original} -> {entity.translation}")
        
        return entity
    
    async def _process_domain_term(
        self, 
        term_data: Dict[str, Any],
        domain: str
    ) -> Optional[DomainTerm]:
        """处理领域术语"""
        original = term_data.get("original", "")
        if not original:
            return None
        
        # 检查是否已存在
        existing = self.terminology_db.lookup(original)
        if existing:
            existing.occurrence_count += 1
            return existing
        
        # 如果是俚语，尝试查找更准确的含义
        is_slang = "slang" in term_data.get("translation_strategy", "").lower()
        if is_slang:
            slang_info = await self._lookup_slang(
                original, 
                term_data.get("context", ""),
                domain
            )
            if slang_info:
                term_data.update(slang_info)
        
        # 解析翻译策略
        strategy = self._parse_translation_strategy(
            term_data.get("translation_strategy", "literal")
        )
        
        term = DomainTerm(
            term_id=f"term_{uuid.uuid4().hex[:8]}",
            original=original,
            translation=term_data.get("suggested_translation", original),
            domain=domain,
            domain_meaning=term_data.get("domain_meaning", ""),
            general_meaning=term_data.get("general_meaning", ""),
            translation_strategy=strategy,
            is_slang=is_slang,
            confidence=term_data.get("confidence", 0.8),
            source="auto_extracted"
        )
        
        self.terminology_db.add_domain_term(term)
        logger.debug(f"Added domain term: {original} -> {term.translation}")
        
        return term
    
    async def _process_cultural_term(
        self, 
        term_data: Dict[str, Any]
    ) -> Optional[CulturalTerm]:
        """处理文化负载词"""
        original = term_data.get("original", "")
        if not original:
            return None
        
        # 检查是否已存在
        existing = self.terminology_db.lookup(original)
        if existing:
            existing.occurrence_count += 1
            return existing
        
        strategy = self._parse_translation_strategy(
            term_data.get("translation_strategy", "free")
        )
        
        term = CulturalTerm(
            term_id=f"cultural_{uuid.uuid4().hex[:8]}",
            original=original,
            translation=term_data.get("suggested_translation", original),
            cultural_context=term_data.get("cultural_context", ""),
            semantic_explanation=term_data.get("semantic_explanation", ""),
            translation_strategy=strategy,
            confidence=term_data.get("confidence", 0.75),
            source="auto_extracted"
        )
        
        self.terminology_db.add_cultural_term(term)
        logger.debug(f"Added cultural term: {original} -> {term.translation}")
        
        return term
    
    async def _lookup_slang(
        self, 
        term: str, 
        context: str, 
        domain: str
    ) -> Optional[Dict[str, Any]]:
        """查找俚语含义"""
        prompt = SLANG_LOOKUP_PROMPT.format(
            term=term,
            context=context,
            domain=domain
        )
        
        try:
            result = await self.invoke_llm(prompt)
            if isinstance(result, dict):
                return {
                    "domain_meaning": result.get("contextual_meaning", ""),
                    "slang_origin": result.get("origin", ""),
                    "suggested_translation": (
                        result.get("translation_options", [{}])[0].get("translation", term)
                        if result.get("translation_options") else term
                    )
                }
        except Exception as e:
            logger.warning(f"Slang lookup failed for '{term}': {e}")
        
        return None
    
    async def _verify_new_terms(self) -> Dict[str, Any]:
        """验证新术语"""
        unverified = self.terminology_db.get_unverified_entries()
        
        if not unverified:
            return {"verified": 0, "conflicts": []}
        
        # 准备验证请求
        term_list = [
            {
                "original": t.original,
                "translation": t.translation,
                "type": type(t).__name__
            }
            for t in unverified[:20]  # 每次最多验证20个
        ]
        
        # 获取已验证术语作为参考
        reference = {
            t.original: t.translation
            for t in self.terminology_db.named_entities.values()
            if t.verified
        }
        reference.update({
            t.original: t.translation
            for t in self.terminology_db.domain_terms.values()
            if t.verified
        })
        
        prompt = TERMINOLOGY_VERIFICATION_PROMPT.format(
            terminology_list=term_list,
            reference_terminology=reference
        )
        
        try:
            result = await self.invoke_llm(prompt)
            
            if isinstance(result, dict):
                # 处理验证结果
                for verified in result.get("verified_terms", []):
                    original = verified.get("original")
                    if verified.get("status") == "approved":
                        entry = self.terminology_db.lookup(original)
                        if entry:
                            self.terminology_db.verify_entry(
                                entry.term_id, 
                                verified_by="system"
                            )
                
                return {
                    "verified": len(result.get("verified_terms", [])),
                    "conflicts": result.get("conflicts", [])
                }
        except Exception as e:
            logger.warning(f"Term verification failed: {e}")
        
        return {"verified": 0, "conflicts": []}
    
    def _parse_entity_type(self, type_str: str) -> EntityType:
        """解析实体类型"""
        type_map = {
            "PERSON": EntityType.PERSON,
            "LOCATION": EntityType.LOCATION,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "WORK": EntityType.WORK,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT
        }
        return type_map.get(type_str.upper(), EntityType.OTHER)
    
    def _parse_translation_strategy(self, strategy_str: str) -> TranslationStrategy:
        """解析翻译策略"""
        strategy_map = {
            "literal": TranslationStrategy.LITERAL,
            "直译": TranslationStrategy.LITERAL,
            "free": TranslationStrategy.FREE,
            "意译": TranslationStrategy.FREE,
            "transliteration": TranslationStrategy.TRANSLITERATION,
            "音译": TranslationStrategy.TRANSLITERATION,
            "hybrid": TranslationStrategy.HYBRID,
            "音意结合": TranslationStrategy.HYBRID,
            "semantic_compensation": TranslationStrategy.SEMANTIC_COMPENSATION,
            "语义补偿": TranslationStrategy.SEMANTIC_COMPENSATION,
            "literal_with_note": TranslationStrategy.LITERAL_WITH_NOTE,
            "直译加注": TranslationStrategy.LITERAL_WITH_NOTE,
            "adaptation": TranslationStrategy.ADAPTATION,
            "归化": TranslationStrategy.ADAPTATION
        }
        return strategy_map.get(strategy_str.lower(), TranslationStrategy.LITERAL)
    
    def get_terminology_for_translation(self) -> Dict[str, str]:
        """获取用于翻译的术语字典"""
        return self.terminology_db.to_dict_for_prompt()
    
    def get_database(self) -> TerminologyDatabase:
        """获取术语库"""
        return self.terminology_db
    
    async def add_human_verified_term(
        self, 
        original: str, 
        translation: str, 
        term_type: str = "domain"
    ):
        """添加人工验证的术语"""
        if term_type == "entity":
            entity = NamedEntity(
                term_id=f"entity_{uuid.uuid4().hex[:8]}",
                original=original,
                translation=translation,
                verified=True,
                verified_by="human",
                confidence=1.0,
                source="human_input"
            )
            self.terminology_db.add_named_entity(entity)
        elif term_type == "cultural":
            term = CulturalTerm(
                term_id=f"cultural_{uuid.uuid4().hex[:8]}",
                original=original,
                translation=translation,
                verified=True,
                verified_by="human",
                confidence=1.0,
                source="human_input"
            )
            self.terminology_db.add_cultural_term(term)
        else:
            term = DomainTerm(
                term_id=f"term_{uuid.uuid4().hex[:8]}",
                original=original,
                translation=translation,
                verified=True,
                verified_by="human",
                confidence=1.0,
                source="human_input"
            )
            self.terminology_db.add_domain_term(term)
        
        logger.info(f"Human verified term added: {original} -> {translation}")
