"""
全局配置设置
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class LLMSettings(BaseModel):
    """LLM配置"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o"))
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # 备用模型（用于回译验证等）
    backup_model: str = Field(default_factory=lambda: os.getenv("BACKUP_MODEL", "gpt-4o-mini"))


class TranslationSettings(BaseModel):
    """翻译配置"""
    source_language: str = "en"
    target_language: str = "zh"
    
    # 翻译风格
    style: str = "formal"  # formal, casual, literary, technical
    
    # 领域
    domain: str = "general"  # general, literature, technical, legal, entertainment
    
    # 切分设置
    chunk_size: int = 1500  # 每个翻译单元的最大字符数
    chunk_overlap: int = 200  # 切分重叠区域
    
    # 多版本生成
    generate_variants: bool = True
    variant_count: int = 3
    
    # 回译验证
    enable_back_translation: bool = True
    back_translation_threshold: float = 0.85  # 相似度阈值


class RAGSettings(BaseModel):
    """RAG知识库配置"""
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = str(PROJECT_ROOT / "data" / "vector_db")
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5


class PathSettings(BaseModel):
    """路径配置"""
    input_dir: Path = PROJECT_ROOT / "data" / "input"
    output_dir: Path = PROJECT_ROOT / "data" / "output"
    terminology_dir: Path = PROJECT_ROOT / "data" / "terminology"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    
    def ensure_dirs(self):
        """确保所有目录存在"""
        for path in [self.input_dir, self.output_dir, self.terminology_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)


class HumanLoopSettings(BaseModel):
    """人机协作配置"""
    enabled: bool = True
    
    # 需要人工确认的场景
    confirm_new_terminology: bool = True  # 新术语首次出现
    confirm_cultural_terms: bool = True   # 文化负载词
    confirm_ambiguous_names: bool = True  # 歧义人名
    confirm_low_confidence: bool = True   # 低置信度翻译
    
    # 置信度阈值
    confidence_threshold: float = 0.7


class Settings(BaseModel):
    """全局配置"""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    human_loop: HumanLoopSettings = Field(default_factory=HumanLoopSettings)
    
    # 日志级别
    log_level: str = "INFO"
    
    # 调试模式
    debug: bool = False


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置"""
    return settings


def update_settings(**kwargs):
    """更新配置"""
    global settings
    settings = Settings(**kwargs)
    return settings
