"""
预处理Agent

负责：
1. 文本结构化切分
2. 语域与风格自动识别
3. 上下文信息保留
"""
import uuid
from typing import List, Optional, Tuple
from loguru import logger

from .base_agent import BaseAgent
from src.models.document import (
    Document, DocumentChunk, DocumentMetadata,
    ChunkContext, ChunkMetadata, ParagraphType, DifficultyLevel
)
from config.prompts import STYLE_ANALYSIS_PROMPT, TEXT_CHUNKING_PROMPT


class PreprocessorAgent(BaseAgent):
    """预处理Agent"""
    
    def __init__(self, **kwargs):
        super().__init__(name="PreprocessorAgent", **kwargs)
        self.chunk_size = self.settings.translation.chunk_size
        self.chunk_overlap = self.settings.translation.chunk_overlap
    
    async def run(self, input_data: str, title: str = "Untitled") -> Document:
        """
        预处理文本
        
        Args:
            input_data: 原始文本内容
            title: 文档标题
            
        Returns:
            Document: 处理后的文档对象
        """
        self.update_state("running", "Starting preprocessing", 0.0)
        
        try:
            # 1. 创建文档
            doc = Document(
                doc_id=str(uuid.uuid4()),
                title=title,
                raw_content=input_data
            )
            
            # 2. 分析风格和领域
            self.update_state("running", "Analyzing style and domain", 0.2)
            metadata = await self._analyze_style(input_data[:3000])  # 分析前3000字符
            doc.metadata = metadata
            
            # 3. 切分文本
            self.update_state("running", "Chunking text", 0.5)
            chunks = await self._chunk_text(input_data, doc.metadata)
            
            for chunk in chunks:
                doc.add_chunk(chunk)
            
            self.update_state("completed", "Preprocessing completed", 1.0)
            self.state.result = doc
            
            logger.info(f"Document preprocessed: {len(chunks)} chunks created")
            return doc
            
        except Exception as e:
            self.set_error(str(e))
            raise
    
    async def _analyze_style(self, text: str) -> DocumentMetadata:
        """分析文本风格和领域"""
        prompt = STYLE_ANALYSIS_PROMPT.format(text=text)
        
        result = await self.invoke_llm(prompt)
        
        metadata = DocumentMetadata(
            source_language=self.settings.translation.source_language,
            target_language=self.settings.translation.target_language
        )
        
        if isinstance(result, dict):
            metadata.domain = result.get("domain", "general")
            metadata.style = result.get("style", "formal")
            metadata.formality_level = int(result.get("formality_level", 3))
            metadata.target_audience = result.get("target_audience", "general")
            metadata.special_elements = result.get("special_elements", [])
            metadata.translation_suggestions = result.get("translation_suggestions", "")
        
        return metadata
    
    async def _chunk_text(
        self, 
        text: str, 
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """切分文本"""
        # 首先进行基础切分（按段落）
        paragraphs = self._split_by_paragraphs(text)
        
        # 然后合并成合适大小的chunks
        chunks = self._merge_into_chunks(paragraphs)
        
        # 可选：使用LLM进行智能切分优化
        if len(text) > 10000:
            # 对于长文本，只对边界可能有问题的chunk进行LLM辅助调整
            chunks = await self._optimize_chunks_with_llm(chunks)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落切分"""
        # 按双换行分割
        paragraphs = text.split('\n\n')
        
        # 清理空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _merge_into_chunks(self, paragraphs: List[str]) -> List[DocumentChunk]:
        """将段落合并成合适大小的chunks"""
        chunks = []
        current_content = []
        current_length = 0
        chunk_index = 0
        position = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # 如果当前chunk加上这个段落超过限制
            if current_length + para_length > self.chunk_size and current_content:
                # 保存当前chunk
                chunk_text = '\n\n'.join(current_content)
                chunk = self._create_chunk(
                    chunk_text, 
                    chunk_index, 
                    position,
                    position + len(chunk_text)
                )
                chunks.append(chunk)
                
                position += len(chunk_text) + 2  # +2 for \n\n
                chunk_index += 1
                
                # 保留重叠部分
                overlap_content = self._get_overlap_content(current_content)
                current_content = overlap_content + [para]
                current_length = sum(len(p) for p in current_content)
            else:
                current_content.append(para)
                current_length += para_length
        
        # 处理最后一个chunk
        if current_content:
            chunk_text = '\n\n'.join(current_content)
            chunk = self._create_chunk(
                chunk_text,
                chunk_index,
                position,
                position + len(chunk_text)
            )
            chunks.append(chunk)
        
        # 设置前后chunk关联
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.context.prev_chunk_id = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunk.context.next_chunk_id = chunks[i+1].chunk_id
        
        return chunks
    
    def _get_overlap_content(self, content: List[str]) -> List[str]:
        """获取重叠内容"""
        if not content:
            return []
        
        overlap = []
        overlap_length = 0
        
        for para in reversed(content):
            if overlap_length + len(para) <= self.chunk_overlap:
                overlap.insert(0, para)
                overlap_length += len(para)
            else:
                break
        
        return overlap
    
    def _create_chunk(
        self, 
        content: str, 
        index: int,
        start_pos: int,
        end_pos: int
    ) -> DocumentChunk:
        """创建chunk对象"""
        chunk_id = f"chunk_{index:04d}"
        
        # 分析段落类型和难度
        paragraph_type = self._detect_paragraph_type(content)
        difficulty = self._estimate_difficulty(content)
        
        metadata = ChunkMetadata(
            paragraph_type=paragraph_type,
            estimated_difficulty=difficulty,
            char_count=len(content),
            word_count=len(content.split()),
            sentence_count=content.count('.') + content.count('!') + content.count('?')
        )
        
        context = ChunkContext()
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            start_position=start_pos,
            end_position=end_pos,
            sequence_number=index,
            context=context,
            metadata=metadata
        )
    
    def _detect_paragraph_type(self, text: str) -> ParagraphType:
        """检测段落类型"""
        # 简单的启发式规则
        if '"' in text or '"' in text or '「' in text:
            return ParagraphType.DIALOGUE
        elif text.startswith(('1.', '2.', '-', '•', '*')):
            return ParagraphType.EXPOSITION
        else:
            return ParagraphType.NARRATIVE
    
    def _estimate_difficulty(self, text: str) -> DifficultyLevel:
        """估计翻译难度"""
        # 基于句子长度和特殊字符的简单估计
        avg_sentence_length = len(text) / max(1, text.count('.') + text.count('!') + text.count('?'))
        
        if avg_sentence_length > 100:
            return DifficultyLevel.HARD
        elif avg_sentence_length > 50:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
    
    async def _optimize_chunks_with_llm(
        self, 
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """使用LLM优化切分（可选）"""
        # 这里可以调用LLM检查chunk边界是否合理
        # 为了效率，只检查可能有问题的边界
        # 当前简化实现：直接返回原chunks
        return chunks
    
    async def extract_chapter_info(self, text: str) -> List[dict]:
        """提取章节信息"""
        # 查找章节标记
        import re
        
        chapter_patterns = [
            r'^(Chapter|CHAPTER)\s+(\d+)',
            r'^(第[一二三四五六七八九十\d]+章)',
            r'^(\d+\.)\s+',
        ]
        
        chapters = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in chapter_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    chapters.append({
                        'line_number': i,
                        'title': line.strip(),
                        'match': match.group(0)
                    })
                    break
        
        return chapters
