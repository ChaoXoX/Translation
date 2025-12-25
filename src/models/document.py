"""
文档数据模型

定义文档的结构化表示，包括：
- 文档元数据
- 文档切分单元
- 上下文信息
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ParagraphType(str, Enum):
    """段落类型"""
    NARRATIVE = "narrative"       # 叙述
    DIALOGUE = "dialogue"         # 对话
    DESCRIPTION = "description"   # 描写
    EXPOSITION = "exposition"     # 说明
    ARGUMENT = "argument"         # 议论


class DifficultyLevel(str, Enum):
    """翻译难度级别"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DocumentMetadata(BaseModel):
    """文档元数据"""
    title: str = ""
    author: str = ""
    source_language: str = "en"
    target_language: str = "zh"
    
    # 领域和风格
    domain: str = "general"  # 领域：general, literature, technical, legal, entertainment
    style: str = "formal"    # 风格：formal, casual, literary, technical
    formality_level: int = Field(default=3, ge=1, le=5)  # 正式程度 1-5
    
    # 目标读者
    target_audience: str = "general"
    
    # 特殊元素标记
    special_elements: List[str] = Field(default_factory=list)
    
    # 翻译建议
    translation_suggestions: str = ""
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # 自定义元数据
    extra: Dict[str, Any] = Field(default_factory=dict)


class ChunkContext(BaseModel):
    """切分单元的上下文信息"""
    context_before: str = ""      # 前文上下文摘要
    context_after: str = ""       # 后文上下文摘要
    chapter_title: str = ""       # 所属章节标题
    chapter_number: int = 0       # 章节编号
    section_title: str = ""       # 所属小节标题
    
    # 前后相邻chunk的ID（用于上下文追踪）
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


class ChunkMetadata(BaseModel):
    """切分单元的元数据"""
    paragraph_type: ParagraphType = ParagraphType.NARRATIVE
    has_special_terms: bool = False
    estimated_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    
    # 包含的特殊元素
    named_entities: List[str] = Field(default_factory=list)
    domain_terms: List[str] = Field(default_factory=list)
    cultural_terms: List[str] = Field(default_factory=list)
    
    # 字符和词统计
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0


class DocumentChunk(BaseModel):
    """文档切分单元"""
    chunk_id: str
    content: str
    
    # 位置信息
    start_position: int = 0       # 在原文档中的起始位置
    end_position: int = 0         # 在原文档中的结束位置
    sequence_number: int = 0      # 序号
    
    # 上下文
    context: ChunkContext = Field(default_factory=ChunkContext)
    
    # 元数据
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    
    # 处理状态
    is_translated: bool = False
    translation_version: int = 0
    
    def __str__(self) -> str:
        return f"Chunk[{self.chunk_id}]: {self.content[:50]}..."


class Document(BaseModel):
    """文档模型"""
    doc_id: str
    title: str
    raw_content: str
    
    # 元数据
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # 切分后的单元
    chunks: List[DocumentChunk] = Field(default_factory=list)
    
    # 处理状态
    total_chunks: int = 0
    translated_chunks: int = 0
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_chunk(self, chunk: DocumentChunk):
        """添加切分单元"""
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """根据ID获取切分单元"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_progress(self) -> float:
        """获取翻译进度"""
        if self.total_chunks == 0:
            return 0.0
        return self.translated_chunks / self.total_chunks
    
    def mark_chunk_translated(self, chunk_id: str):
        """标记切分单元已翻译"""
        chunk = self.get_chunk_by_id(chunk_id)
        if chunk and not chunk.is_translated:
            chunk.is_translated = True
            chunk.translation_version += 1
            self.translated_chunks += 1
            self.updated_at = datetime.now()
    
    def __str__(self) -> str:
        return f"Document[{self.doc_id}]: {self.title} ({self.translated_chunks}/{self.total_chunks} translated)"


class DocumentSummary(BaseModel):
    """文档摘要（用于Memory）"""
    doc_id: str
    title: str
    
    # 内容摘要
    plot_summary: str = ""         # 情节/内容摘要
    main_characters: List[str] = Field(default_factory=list)   # 主要人物
    key_themes: List[str] = Field(default_factory=list)        # 关键主题
    
    # 风格分析
    narrative_style: str = ""      # 叙事风格
    tone: str = ""                 # 语调
    
    # 翻译相关
    translation_challenges: List[str] = Field(default_factory=list)  # 翻译难点
    style_guide: str = ""          # 翻译风格指南
    
    # 统计信息
    total_words: int = 0
    total_chapters: int = 0


class ReaderProfile(BaseModel):
    """目标读者画像（用于翻译风格调整）"""
    audience_type: str = "general"     # 读者类型
    age_group: str = "adult"           # 年龄段
    expertise_level: str = "general"   # 专业程度
    cultural_background: str = ""      # 文化背景
    
    # 阅读偏好
    preferred_style: str = "balanced"  # 偏好风格
    formality_preference: str = "moderate"  # 正式程度偏好
    
    # 特殊要求
    special_requirements: List[str] = Field(default_factory=list)
