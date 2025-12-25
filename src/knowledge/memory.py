"""
翻译记忆模块

管理翻译过程中的各类记忆：
1. 已翻译文本
2. 文档摘要
3. 风格指南
4. 读者偏好
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import json
from loguru import logger
from enum import Enum


class MemoryType(str, Enum):
    """记忆类型"""
    TRANSLATION = "translation"          # 翻译对
    SUMMARY = "summary"                   # 文档摘要
    STYLE_GUIDE = "style_guide"          # 风格指南
    READER_PROFILE = "reader_profile"     # 读者画像
    TERMINOLOGY = "terminology"           # 术语
    CONTEXT = "context"                   # 上下文信息
    FEEDBACK = "feedback"                 # 反馈


class MemoryItem(BaseModel):
    """记忆条目"""
    item_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    
    # 关联信息
    chunk_id: str = ""                    # 关联的chunk ID
    doc_id: str = ""                      # 关联的文档ID
    
    # 元数据
    importance: float = 0.5               # 重要性 (0-1)
    access_count: int = 0                 # 访问次数
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    
    def access(self):
        """记录访问"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class TranslationPair(BaseModel):
    """翻译对"""
    source: str
    target: str
    source_language: str = "en"
    target_language: str = "zh"
    quality_score: float = 0.0
    verified: bool = False


class DocumentSummary(BaseModel):
    """文档摘要"""
    doc_id: str
    title: str
    
    # 内容摘要
    plot_summary: str = ""
    main_characters: List[str] = Field(default_factory=list)
    key_themes: List[str] = Field(default_factory=list)
    key_events: List[str] = Field(default_factory=list)
    
    # 翻译相关
    translation_challenges: List[str] = Field(default_factory=list)
    cultural_notes: List[str] = Field(default_factory=list)


class StyleGuide(BaseModel):
    """翻译风格指南"""
    name: str = "default"
    
    # 基本设置
    formality: str = "formal"             # formal, casual, neutral
    tone: str = "neutral"                 # neutral, friendly, professional
    perspective: str = "third_person"     # first_person, second_person, third_person
    
    # 特殊处理规则
    handle_names: str = "transliterate"   # keep, transliterate, translate
    handle_idioms: str = "adapt"          # literal, adapt, explain
    handle_humor: str = "adapt"           # literal, adapt, omit
    
    # 格式规则
    use_quotes: str = "chinese"           # english ("), chinese ("), keep_original
    number_format: str = "chinese"        # arabic, chinese, mixed
    
    # 自定义规则
    custom_rules: List[str] = Field(default_factory=list)
    
    # 示例
    examples: List[Dict[str, str]] = Field(default_factory=list)


class ReaderProfile(BaseModel):
    """读者画像"""
    profile_id: str
    
    # 基本信息
    audience_type: str = "general"        # general, academic, professional, young_adult
    age_group: str = "adult"              # child, young_adult, adult, elderly
    education_level: str = "general"      # general, academic, professional
    
    # 文化背景
    cultural_background: str = ""
    native_language: str = "zh"
    
    # 阅读偏好
    preferred_complexity: str = "medium"  # simple, medium, complex
    preferred_style: str = "balanced"     # literal, adaptive, creative
    
    # 特殊需求
    accessibility_needs: List[str] = Field(default_factory=list)
    domain_expertise: List[str] = Field(default_factory=list)


class TranslationMemory:
    """翻译记忆管理"""
    
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path
        
        # 记忆存储
        self.memories: Dict[str, MemoryItem] = {}
        
        # 专项存储
        self.translation_pairs: List[TranslationPair] = []
        self.document_summaries: Dict[str, DocumentSummary] = {}
        self.style_guide: StyleGuide = StyleGuide()
        self.reader_profile: Optional[ReaderProfile] = None
        
        # 索引
        self._type_index: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        self._chunk_index: Dict[str, List[str]] = {}
        
        if persist_path and persist_path.exists():
            self.load()
    
    def add_memory(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        chunk_id: str = "",
        doc_id: str = "",
        importance: float = 0.5
    ) -> str:
        """添加记忆"""
        item_id = f"mem_{len(self.memories)}_{memory_type.value}"
        
        item = MemoryItem(
            item_id=item_id,
            memory_type=memory_type,
            content=content,
            chunk_id=chunk_id,
            doc_id=doc_id,
            importance=importance
        )
        
        self.memories[item_id] = item
        self._type_index[memory_type].append(item_id)
        
        if chunk_id:
            if chunk_id not in self._chunk_index:
                self._chunk_index[chunk_id] = []
            self._chunk_index[chunk_id].append(item_id)
        
        logger.debug(f"Memory added: {item_id}")
        return item_id
    
    def add_translation_pair(
        self,
        source: str,
        target: str,
        source_language: str = "en",
        target_language: str = "zh",
        quality_score: float = 0.0,
        verified: bool = False
    ):
        """添加翻译对"""
        pair = TranslationPair(
            source=source,
            target=target,
            source_language=source_language,
            target_language=target_language,
            quality_score=quality_score,
            verified=verified
        )
        self.translation_pairs.append(pair)
        
        # 同时添加到记忆
        self.add_memory(
            MemoryType.TRANSLATION,
            {"source": source, "target": target},
            importance=0.7 if verified else 0.5
        )
    
    def add_document_summary(self, summary: DocumentSummary):
        """添加文档摘要"""
        self.document_summaries[summary.doc_id] = summary
        
        self.add_memory(
            MemoryType.SUMMARY,
            summary.model_dump(),
            doc_id=summary.doc_id,
            importance=0.9
        )
    
    def set_style_guide(self, guide: StyleGuide):
        """设置风格指南"""
        self.style_guide = guide
        
        self.add_memory(
            MemoryType.STYLE_GUIDE,
            guide.model_dump(),
            importance=1.0
        )
    
    def set_reader_profile(self, profile: ReaderProfile):
        """设置读者画像"""
        self.reader_profile = profile
        
        self.add_memory(
            MemoryType.READER_PROFILE,
            profile.model_dump(),
            importance=0.8
        )
    
    def get_memories_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 10
    ) -> List[MemoryItem]:
        """按类型获取记忆"""
        item_ids = self._type_index.get(memory_type, [])
        items = [self.memories[id] for id in item_ids if id in self.memories]
        
        # 按重要性和访问时间排序
        items.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
        
        return items[:limit]
    
    def get_memories_for_chunk(self, chunk_id: str) -> List[MemoryItem]:
        """获取chunk相关的记忆"""
        item_ids = self._chunk_index.get(chunk_id, [])
        return [self.memories[id] for id in item_ids if id in self.memories]
    
    def search_translation_pairs(
        self,
        query: str,
        limit: int = 5
    ) -> List[TranslationPair]:
        """搜索相似翻译对"""
        # 简单的关键词匹配
        query_terms = set(query.lower().split())
        
        scored_pairs = []
        for pair in self.translation_pairs:
            source_terms = set(pair.source.lower().split())
            overlap = len(query_terms & source_terms)
            score = overlap / max(len(query_terms), 1)
            
            if score > 0:
                scored_pairs.append((pair, score))
        
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in scored_pairs[:limit]]
    
    def get_context_for_prompt(
        self,
        chunk_id: str = "",
        include_style: bool = True,
        include_reader: bool = True,
        max_translations: int = 5
    ) -> str:
        """
        获取用于提示词的上下文
        
        Args:
            chunk_id: chunk ID
            include_style: 是否包含风格指南
            include_reader: 是否包含读者画像
            max_translations: 最大翻译记忆数量
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 风格指南
        if include_style and self.style_guide:
            style_text = f"""**翻译风格指南:**
- 正式程度: {self.style_guide.formality}
- 语调: {self.style_guide.tone}
- 人名处理: {self.style_guide.handle_names}
- 习语处理: {self.style_guide.handle_idioms}
"""
            if self.style_guide.custom_rules:
                style_text += "- 特殊规则: " + "; ".join(self.style_guide.custom_rules)
            context_parts.append(style_text)
        
        # 读者画像
        if include_reader and self.reader_profile:
            reader_text = f"""**目标读者:**
- 类型: {self.reader_profile.audience_type}
- 年龄段: {self.reader_profile.age_group}
- 复杂度偏好: {self.reader_profile.preferred_complexity}
"""
            context_parts.append(reader_text)
        
        # 翻译记忆
        if self.translation_pairs:
            recent_pairs = sorted(
                self.translation_pairs,
                key=lambda x: x.quality_score,
                reverse=True
            )[:max_translations]
            
            if recent_pairs:
                tm_text = "**参考翻译:**\n"
                for pair in recent_pairs:
                    tm_text += f"- \"{pair.source[:50]}...\" → \"{pair.target[:50]}...\"\n"
                context_parts.append(tm_text)
        
        # chunk相关记忆
        if chunk_id:
            chunk_memories = self.get_memories_for_chunk(chunk_id)
            if chunk_memories:
                mem_text = "**相关记忆:**\n"
                for mem in chunk_memories[:3]:
                    mem_text += f"- [{mem.memory_type.value}] {str(mem.content)[:100]}...\n"
                context_parts.append(mem_text)
        
        return "\n\n".join(context_parts)
    
    def save(self):
        """保存记忆"""
        if not self.persist_path:
            return
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # 保存记忆
        memories_file = self.persist_path / "memories.json"
        with open(memories_file, 'w', encoding='utf-8') as f:
            json.dump(
                {k: v.model_dump() for k, v in self.memories.items()},
                f,
                ensure_ascii=False,
                indent=2,
                default=str
            )
        
        # 保存翻译对
        pairs_file = self.persist_path / "translation_pairs.json"
        with open(pairs_file, 'w', encoding='utf-8') as f:
            json.dump(
                [p.model_dump() for p in self.translation_pairs],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # 保存风格指南
        style_file = self.persist_path / "style_guide.json"
        with open(style_file, 'w', encoding='utf-8') as f:
            json.dump(self.style_guide.model_dump(), f, ensure_ascii=False, indent=2)
        
        # 保存读者画像
        if self.reader_profile:
            reader_file = self.persist_path / "reader_profile.json"
            with open(reader_file, 'w', encoding='utf-8') as f:
                json.dump(self.reader_profile.model_dump(), f, ensure_ascii=False, indent=2)
        
        # 保存文档摘要
        summaries_file = self.persist_path / "document_summaries.json"
        with open(summaries_file, 'w', encoding='utf-8') as f:
            json.dump(
                {k: v.model_dump() for k, v in self.document_summaries.items()},
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info(f"Memory saved to {self.persist_path}")
    
    def load(self):
        """加载记忆"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        # 加载记忆
        memories_file = self.persist_path / "memories.json"
        if memories_file.exists():
            with open(memories_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.items():
                v['memory_type'] = MemoryType(v['memory_type'])
                self.memories[k] = MemoryItem(**v)
            self._rebuild_indices()
        
        # 加载翻译对
        pairs_file = self.persist_path / "translation_pairs.json"
        if pairs_file.exists():
            with open(pairs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.translation_pairs = [TranslationPair(**p) for p in data]
        
        # 加载风格指南
        style_file = self.persist_path / "style_guide.json"
        if style_file.exists():
            with open(style_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.style_guide = StyleGuide(**data)
        
        # 加载读者画像
        reader_file = self.persist_path / "reader_profile.json"
        if reader_file.exists():
            with open(reader_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.reader_profile = ReaderProfile(**data)
        
        # 加载文档摘要
        summaries_file = self.persist_path / "document_summaries.json"
        if summaries_file.exists():
            with open(summaries_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.document_summaries = {k: DocumentSummary(**v) for k, v in data.items()}
        
        logger.info(f"Memory loaded from {self.persist_path}")
    
    def _rebuild_indices(self):
        """重建索引"""
        self._type_index = {t: [] for t in MemoryType}
        self._chunk_index = {}
        
        for item_id, item in self.memories.items():
            self._type_index[item.memory_type].append(item_id)
            
            if item.chunk_id:
                if item.chunk_id not in self._chunk_index:
                    self._chunk_index[item.chunk_id] = []
                self._chunk_index[item.chunk_id].append(item_id)
    
    def clear(self):
        """清空记忆"""
        self.memories = {}
        self.translation_pairs = []
        self.document_summaries = {}
        self._type_index = {t: [] for t in MemoryType}
        self._chunk_index = {}
