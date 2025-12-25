"""
文件处理工具
"""
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
from loguru import logger

try:
    from ebooklib import epub
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


def load_text_file(filepath: Path) -> str:
    """
    加载文本文件
    
    支持: .txt, .md, .epub
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix in ['.txt', '.md']:
        return _load_plain_text(filepath)
    elif suffix == '.epub':
        return _load_epub(filepath)
    elif suffix == '.json':
        return _load_json_text(filepath)
    else:
        # 尝试作为纯文本加载
        return _load_plain_text(filepath)


def _load_plain_text(filepath: Path) -> str:
    """加载纯文本文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Unable to decode file: {filepath}")


def _load_epub(filepath: Path) -> str:
    """加载EPUB文件"""
    if not EPUB_AVAILABLE:
        raise ImportError("ebooklib is required to load EPUB files. Install with: pip install ebooklib")
    
    if not BS4_AVAILABLE:
        raise ImportError("beautifulsoup4 is required to parse EPUB content. Install with: pip install beautifulsoup4")
    
    book = epub.read_epub(str(filepath))
    
    text_parts = []
    
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            # 解析HTML内容
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # 提取文本
            text = soup.get_text(separator='\n\n')
            text_parts.append(text)
    
    return '\n\n'.join(text_parts)


def _load_json_text(filepath: Path) -> str:
    """从JSON文件加载文本"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持多种JSON格式
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        if 'text' in data:
            return data['text']
        elif 'content' in data:
            return data['content']
        elif 'chapters' in data:
            return '\n\n'.join(data['chapters'])
    elif isinstance(data, list):
        return '\n\n'.join(str(item) for item in data)
    
    raise ValueError("Unable to extract text from JSON file")


def save_translation(
    translation: str,
    output_path: Path,
    format: str = "txt",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    保存翻译结果
    
    Args:
        translation: 翻译文本
        output_path: 输出路径
        format: 输出格式 (txt, md, json)
        metadata: 元数据
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "txt":
        _save_as_txt(translation, output_path, metadata)
    elif format == "md":
        _save_as_markdown(translation, output_path, metadata)
    elif format == "json":
        _save_as_json(translation, output_path, metadata)
    else:
        _save_as_txt(translation, output_path, metadata)
    
    logger.info(f"Translation saved to: {output_path}")


def _save_as_txt(text: str, filepath: Path, metadata: Optional[Dict] = None):
    """保存为纯文本"""
    with open(filepath.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        if metadata:
            f.write(f"# {metadata.get('title', 'Translation')}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("\n" + "="*50 + "\n\n")
        f.write(text)


def _save_as_markdown(text: str, filepath: Path, metadata: Optional[Dict] = None):
    """保存为Markdown"""
    with open(filepath.with_suffix('.md'), 'w', encoding='utf-8') as f:
        if metadata:
            f.write(f"# {metadata.get('title', 'Translation')}\n\n")
            f.write(f"> Generated: {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
        f.write(text)


def _save_as_json(text: str, filepath: Path, metadata: Optional[Dict] = None):
    """保存为JSON"""
    data = {
        "translation": text,
        "metadata": metadata or {},
        "generated_at": datetime.now().isoformat()
    }
    
    with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class FileHandler:
    """文件处理器"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.cwd()
    
    def load(self, filepath: str) -> str:
        """加载文件"""
        full_path = self.base_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
        return load_text_file(full_path)
    
    def save(
        self,
        content: str,
        filepath: str,
        format: str = "txt",
        metadata: Optional[Dict] = None
    ):
        """保存文件"""
        full_path = self.base_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
        save_translation(content, full_path, format, metadata)
    
    def ensure_directory(self, dirpath: str):
        """确保目录存在"""
        full_path = self.base_dir / dirpath if not Path(dirpath).is_absolute() else Path(dirpath)
        full_path.mkdir(parents=True, exist_ok=True)
    
    def list_files(self, dirpath: str, pattern: str = "*") -> list:
        """列出目录中的文件"""
        full_path = self.base_dir / dirpath if not Path(dirpath).is_absolute() else Path(dirpath)
        return list(full_path.glob(pattern))
    
    def save_terminology_db(self, db, filepath: str):
        """保存术语库"""
        full_path = self.base_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
        db.save(full_path)
    
    def load_terminology_db(self, filepath: str):
        """加载术语库"""
        from src.models.terminology import TerminologyDatabase
        full_path = self.base_dir / filepath if not Path(filepath).is_absolute() else Path(filepath)
        return TerminologyDatabase.load(full_path)
    
    def save_translation_result(
        self,
        result,
        output_dir: str,
        prefix: str = "translation"
    ):
        """
        保存完整翻译结果
        
        包括:
        - 完整译文
        - 术语表
        - 各单元详情
        - 质量报告
        """
        output_path = self.base_dir / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整译文
        translation_file = output_path / f"{prefix}_{timestamp}.txt"
        with open(translation_file, 'w', encoding='utf-8') as f:
            f.write(result.full_translation)
        
        # 保存详细结果（JSON）
        details_file = output_path / f"{prefix}_{timestamp}_details.json"
        with open(details_file, 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            details = {
                "result_id": result.result_id,
                "doc_id": result.doc_id,
                "total_units": result.total_units,
                "completed_units": result.completed_units,
                "terminology_used": result.terminology_used,
                "generated_at": datetime.now().isoformat()
            }
            json.dump(details, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Translation result saved to: {output_path}")
        
        return {
            "translation_file": str(translation_file),
            "details_file": str(details_file)
        }
