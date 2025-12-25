"""
文本处理工具
"""
import re
from typing import List, Tuple, Optional
import unicodedata


def clean_text(text: str) -> str:
    """
    清理文本
    
    - 移除多余空白
    - 统一换行符
    - 移除不可见字符
    """
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 移除不可见字符（保留换行和空格）
    text = ''.join(
        char for char in text 
        if unicodedata.category(char) != 'Cc' or char in '\n\t '
    )
    
    # 移除多余空行（保留最多两个连续换行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除行尾空白
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    return text.strip()


def count_words(text: str, language: str = "en") -> int:
    """
    统计词数
    
    Args:
        text: 文本
        language: 语言 (en, zh, etc.)
    
    Returns:
        词数
    """
    if language == "zh":
        # 中文按字符计数
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars)
    else:
        # 英文按空格分词
        words = text.split()
        return len(words)


def count_sentences(text: str) -> int:
    """统计句子数"""
    # 匹配句子结束符
    sentences = re.split(r'[.!?。！？]+', text)
    return len([s for s in sentences if s.strip()])


def split_sentences(text: str, language: str = "en") -> List[str]:
    """
    分割句子
    
    Args:
        text: 文本
        language: 语言
    
    Returns:
        句子列表
    """
    if language == "zh":
        # 中文句子分割
        sentences = re.split(r'([。！？；])', text)
    else:
        # 英文句子分割（处理缩写等）
        # 简单实现，可以用nltk改进
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 合并分隔符
    result = []
    for i, s in enumerate(sentences):
        if s and not re.match(r'^[.!?。！？；]$', s):
            result.append(s.strip())
    
    return result


def detect_language(text: str) -> str:
    """
    简单语言检测
    
    Returns:
        语言代码 (en, zh, etc.)
    """
    # 统计中文字符比例
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(' ', ''))
    
    if total_chars == 0:
        return "unknown"
    
    chinese_ratio = chinese_chars / total_chars
    
    if chinese_ratio > 0.3:
        return "zh"
    else:
        return "en"


def extract_quotes(text: str) -> List[Tuple[str, int, int]]:
    """
    提取引用/对话
    
    Returns:
        列表 [(引用内容, 起始位置, 结束位置)]
    """
    quotes = []
    
    # 英文引号
    for match in re.finditer(r'"([^"]+)"', text):
        quotes.append((match.group(1), match.start(), match.end()))
    
    # 中文引号
    for match in re.finditer(r'"([^"]+)"', text):
        quotes.append((match.group(1), match.start(), match.end()))
    
    # 书名号
    for match in re.finditer(r'《([^》]+)》', text):
        quotes.append((match.group(1), match.start(), match.end()))
    
    return quotes


def normalize_punctuation(text: str, target_lang: str = "zh") -> str:
    """
    规范化标点符号
    
    Args:
        text: 文本
        target_lang: 目标语言
    
    Returns:
        规范化后的文本
    """
    if target_lang == "zh":
        # 英文标点转中文
        replacements = {
            ',': '，',
            '.': '。',
            '!': '！',
            '?': '？',
            ':': '：',
            ';': '；',
            '(': '（',
            ')': '）',
            '"': '"',  # 开引号
        }
    else:
        # 中文标点转英文
        replacements = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '：': ':',
            '；': ';',
            '（': '(',
            '）': ')',
            '"': '"',
            '"': '"',
        }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, source_lang: str = "en", target_lang: str = "zh"):
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def preprocess(self, text: str) -> str:
        """预处理文本"""
        text = clean_text(text)
        return text
    
    def postprocess(self, text: str) -> str:
        """后处理译文"""
        text = clean_text(text)
        text = normalize_punctuation(text, self.target_lang)
        return text
    
    def split_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def get_statistics(self, text: str) -> dict:
        """获取文本统计"""
        return {
            "char_count": len(text),
            "word_count": count_words(text, self.source_lang),
            "sentence_count": count_sentences(text),
            "paragraph_count": len(self.split_paragraphs(text)),
            "language": detect_language(text)
        }
    
    def extract_potential_terms(self, text: str) -> List[str]:
        """提取潜在术语（大写词、专有名词等）"""
        terms = []
        
        if self.source_lang == "en":
            # 提取大写开头的词组
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            terms.extend(capitalized)
            
            # 提取全大写词
            uppercase = re.findall(r'\b[A-Z]{2,}\b', text)
            terms.extend(uppercase)
            
            # 提取引号内容
            quoted = re.findall(r'"([^"]+)"', text)
            terms.extend(quoted)
        
        elif self.source_lang == "zh":
            # 提取书名号内容
            titles = re.findall(r'《([^》]+)》', text)
            terms.extend(titles)
            
            # 提取引号内容
            quoted = re.findall(r'"([^"]+)"', text)
            terms.extend(quoted)
        
        # 去重
        return list(set(terms))
    
    def highlight_differences(
        self, 
        text1: str, 
        text2: str
    ) -> Tuple[str, str]:
        """
        高亮两个文本的差异
        
        Returns:
            (高亮后的text1, 高亮后的text2)
        """
        # 简单实现：标记不同的词
        words1 = text1.split()
        words2 = text2.split()
        
        highlighted1 = []
        highlighted2 = []
        
        max_len = max(len(words1), len(words2))
        
        for i in range(max_len):
            w1 = words1[i] if i < len(words1) else ""
            w2 = words2[i] if i < len(words2) else ""
            
            if w1 != w2:
                highlighted1.append(f"**{w1}**" if w1 else "[缺失]")
                highlighted2.append(f"**{w2}**" if w2 else "[缺失]")
            else:
                highlighted1.append(w1)
                highlighted2.append(w2)
        
        return ' '.join(highlighted1), ' '.join(highlighted2)
