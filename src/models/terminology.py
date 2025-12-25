"""
术语数据模型

定义术语库的结构化表示，包括：
- 命名实体 (NER)
- 领域术语
- 文化负载词
- 术语数据库
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class EntityType(str, Enum):
    """命名实体类型"""
    PERSON = "PERSON"           # 人名
    LOCATION = "LOCATION"       # 地名
    ORGANIZATION = "ORGANIZATION"  # 机构名
    WORK = "WORK"               # 作品名
    EVENT = "EVENT"             # 事件名
    PRODUCT = "PRODUCT"         # 产品名
    OTHER = "OTHER"             # 其他


class TranslationStrategy(str, Enum):
    """翻译策略"""
    LITERAL = "literal"           # 直译
    FREE = "free"                 # 意译
    TRANSLITERATION = "transliteration"  # 音译
    HYBRID = "hybrid"             # 音意结合
    SEMANTIC_COMPENSATION = "semantic_compensation"  # 语义补偿
    LITERAL_WITH_NOTE = "literal_with_note"  # 直译加注
    ADAPTATION = "adaptation"     # 改编/归化


class ConfidenceLevel(str, Enum):
    """置信度级别"""
    HIGH = "high"       # 高置信度 (>= 0.9)
    MEDIUM = "medium"   # 中置信度 (0.7-0.9)
    LOW = "low"         # 低置信度 (< 0.7)
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.9:
            return cls.HIGH
        elif score >= 0.7:
            return cls.MEDIUM
        else:
            return cls.LOW


class TerminologyEntry(BaseModel):
    """术语条目基类"""
    term_id: str
    original: str                 # 原文
    translation: str              # 确定的翻译
    
    # 备选翻译
    alternatives: List[str] = Field(default_factory=list)
    
    # 上下文信息
    context: str = ""             # 首次出现的上下文
    contexts: List[str] = Field(default_factory=list)  # 所有出现的上下文
    
    # 元数据
    source: str = ""              # 来源（自动识别/人工添加/外部资源）
    confidence: float = 0.8       # 置信度
    verified: bool = False        # 是否经过验证
    verified_by: str = ""         # 验证者（human/system）
    
    # 使用统计
    occurrence_count: int = 1     # 出现次数
    first_occurrence: int = 0     # 首次出现位置（chunk序号）
    
    # 备注
    notes: str = ""
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        return ConfidenceLevel.from_score(self.confidence)
    
    def needs_review(self) -> bool:
        """是否需要人工审核"""
        return not self.verified and self.confidence_level == ConfidenceLevel.LOW


class NamedEntity(TerminologyEntry):
    """命名实体"""
    entity_type: EntityType = EntityType.OTHER
    
    # 人名特有字段
    full_name: str = ""           # 全名
    nickname: str = ""            # 绰号/别名
    
    # 是否有约定俗成的译法
    has_established_translation: bool = False
    established_source: str = ""  # 约定俗成译法的来源
    
    def __str__(self) -> str:
        return f"Entity[{self.entity_type}]: {self.original} -> {self.translation}"


class DomainTerm(TerminologyEntry):
    """领域术语"""
    domain: str = "general"       # 所属领域
    
    # 含义说明
    domain_meaning: str = ""      # 领域特定含义
    general_meaning: str = ""     # 一般含义
    
    # 翻译策略
    translation_strategy: TranslationStrategy = TranslationStrategy.LITERAL
    
    # 是否是俚语
    is_slang: bool = False
    slang_origin: str = ""        # 俚语来源（如Urban Dictionary）
    
    def __str__(self) -> str:
        return f"Term[{self.domain}]: {self.original} -> {self.translation}"


class CulturalTerm(TerminologyEntry):
    """文化负载词"""
    # 文化背景
    cultural_context: str = ""    # 文化背景说明
    source_culture: str = ""      # 源文化
    
    # 语义解释
    semantic_explanation: str = ""  # 详细语义解释
    
    # 翻译策略
    translation_strategy: TranslationStrategy = TranslationStrategy.FREE
    
    # 是否需要注释
    needs_footnote: bool = False
    footnote_content: str = ""
    
    # 目标文化中的对等表达（如有）
    target_culture_equivalent: str = ""
    
    def __str__(self) -> str:
        return f"Cultural[{self.source_culture}]: {self.original} -> {self.translation}"


class TerminologyDatabase(BaseModel):
    """术语数据库"""
    db_id: str
    project_name: str
    source_language: str = "en"
    target_language: str = "zh"
    
    # 术语集合
    named_entities: Dict[str, NamedEntity] = Field(default_factory=dict)
    domain_terms: Dict[str, DomainTerm] = Field(default_factory=dict)
    cultural_terms: Dict[str, CulturalTerm] = Field(default_factory=dict)
    
    # 快速查找索引（原文 -> term_id）
    _index: Dict[str, str] = PrivateAttr(default_factory=dict)
    
    # 统计信息
    total_terms: int = 0
    verified_terms: int = 0
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def model_post_init(self, __context):
        """初始化后构建索引"""
        self._rebuild_index()
    
    def _rebuild_index(self):
        """重建索引"""
        self._index = {}
        for term_id, entity in self.named_entities.items():
            self._index[entity.original.lower()] = term_id
        for term_id, term in self.domain_terms.items():
            self._index[term.original.lower()] = term_id
        for term_id, term in self.cultural_terms.items():
            self._index[term.original.lower()] = term_id
        self._update_stats()
    
    def _update_stats(self):
        """更新统计信息"""
        self.total_terms = (
            len(self.named_entities) + 
            len(self.domain_terms) + 
            len(self.cultural_terms)
        )
        self.verified_terms = sum(
            1 for e in self.named_entities.values() if e.verified
        ) + sum(
            1 for t in self.domain_terms.values() if t.verified
        ) + sum(
            1 for t in self.cultural_terms.values() if t.verified
        )
    
    def add_named_entity(self, entity: NamedEntity) -> bool:
        """添加命名实体"""
        if entity.original.lower() in self._index:
            return False  # 已存在
        self.named_entities[entity.term_id] = entity
        self._index[entity.original.lower()] = entity.term_id
        self._update_stats()
        self.updated_at = datetime.now()
        return True
    
    def add_domain_term(self, term: DomainTerm) -> bool:
        """添加领域术语"""
        if term.original.lower() in self._index:
            return False
        self.domain_terms[term.term_id] = term
        self._index[term.original.lower()] = term.term_id
        self._update_stats()
        self.updated_at = datetime.now()
        return True
    
    def add_cultural_term(self, term: CulturalTerm) -> bool:
        """添加文化负载词"""
        if term.original.lower() in self._index:
            return False
        self.cultural_terms[term.term_id] = term
        self._index[term.original.lower()] = term.term_id
        self._update_stats()
        self.updated_at = datetime.now()
        return True
    
    def lookup(self, original: str) -> Optional[TerminologyEntry]:
        """查找术语"""
        term_id = self._index.get(original.lower())
        if not term_id:
            return None
        
        if term_id in self.named_entities:
            return self.named_entities[term_id]
        if term_id in self.domain_terms:
            return self.domain_terms[term_id]
        if term_id in self.cultural_terms:
            return self.cultural_terms[term_id]
        return None
    
    def get_translation(self, original: str) -> Optional[str]:
        """获取翻译"""
        entry = self.lookup(original)
        return entry.translation if entry else None
    
    def update_entry(self, term_id: str, **kwargs) -> bool:
        """更新术语条目"""
        entry = None
        if term_id in self.named_entities:
            entry = self.named_entities[term_id]
        elif term_id in self.domain_terms:
            entry = self.domain_terms[term_id]
        elif term_id in self.cultural_terms:
            entry = self.cultural_terms[term_id]
        
        if not entry:
            return False
        
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        entry.updated_at = datetime.now()
        self.updated_at = datetime.now()
        self._update_stats()
        return True
    
    def verify_entry(self, term_id: str, verified_by: str = "human") -> bool:
        """验证术语条目"""
        return self.update_entry(term_id, verified=True, verified_by=verified_by)
    
    def get_unverified_entries(self) -> List[TerminologyEntry]:
        """获取未验证的条目"""
        unverified = []
        for entry in self.named_entities.values():
            if not entry.verified:
                unverified.append(entry)
        for entry in self.domain_terms.values():
            if not entry.verified:
                unverified.append(entry)
        for entry in self.cultural_terms.values():
            if not entry.verified:
                unverified.append(entry)
        return unverified
    
    def get_low_confidence_entries(self) -> List[TerminologyEntry]:
        """获取低置信度条目（需要人工审核）"""
        low_conf = []
        for entry in self.named_entities.values():
            if entry.needs_review():
                low_conf.append(entry)
        for entry in self.domain_terms.values():
            if entry.needs_review():
                low_conf.append(entry)
        for entry in self.cultural_terms.values():
            if entry.needs_review():
                low_conf.append(entry)
        return low_conf
    
    def to_dict_for_prompt(self) -> Dict[str, str]:
        """转换为提示词可用的字典格式"""
        result = {}
        for entry in self.named_entities.values():
            if entry.verified or entry.confidence >= 0.7:
                result[entry.original] = entry.translation
        for entry in self.domain_terms.values():
            if entry.verified or entry.confidence >= 0.7:
                result[entry.original] = entry.translation
        for entry in self.cultural_terms.values():
            if entry.verified or entry.confidence >= 0.7:
                result[entry.original] = entry.translation
        return result
    
    def save(self, filepath: Path):
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: Path) -> "TerminologyDatabase":
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def __str__(self) -> str:
        return f"TermDB[{self.project_name}]: {self.total_terms} terms ({self.verified_terms} verified)"
