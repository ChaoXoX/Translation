"""
RAG模块

实现检索增强生成，支持：
1. 术语库检索
2. 参考资料检索
3. 翻译记忆检索
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from loguru import logger

from pydantic import BaseModel, Field

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document as LangChainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, using simplified RAG")

from config.settings import get_settings


class DocumentChunk(BaseModel):
    """文档块"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class RetrievalResult(BaseModel):
    """检索结果"""
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""


class DocumentStore:
    """文档存储"""
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.settings = get_settings()
        self.persist_directory = persist_directory or Path(self.settings.rag.vector_db_path)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        
        # 初始化嵌入模型
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings(
                    api_key=self.settings.llm.api_key,
                    base_url=self.settings.llm.base_url,
                    model=embedding_model
                )
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")
        
        # 简单内存存储（作为后备）
        self.memory_store: List[DocumentChunk] = []
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """添加文档"""
        if not documents:
            return
        
        metadatas = metadatas or [{} for _ in documents]
        
        if LANGCHAIN_AVAILABLE and self.embeddings:
            try:
                # 使用LangChain向量存储
                lc_docs = [
                    LangChainDocument(page_content=doc, metadata=meta)
                    for doc, meta in zip(documents, metadatas)
                ]
                
                if self.vector_store is None:
                    self.vector_store = Chroma.from_documents(
                        documents=lc_docs,
                        embedding=self.embeddings,
                        persist_directory=str(self.persist_directory)
                    )
                else:
                    self.vector_store.add_documents(lc_docs)
                
                logger.info(f"Added {len(documents)} documents to vector store")
                return
            except Exception as e:
                logger.warning(f"Vector store operation failed: {e}")
        
        # 后备：使用内存存储
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            chunk = DocumentChunk(
                chunk_id=f"chunk_{len(self.memory_store) + i}",
                content=doc,
                metadata=meta
            )
            self.memory_store.append(chunk)
        
        logger.info(f"Added {len(documents)} documents to memory store")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """搜索相似文档"""
        if LANGCHAIN_AVAILABLE and self.vector_store:
            try:
                # 使用向量搜索
                if filter_metadata:
                    results = self.vector_store.similarity_search_with_score(
                        query,
                        k=top_k,
                        filter=filter_metadata
                    )
                else:
                    results = self.vector_store.similarity_search_with_score(
                        query,
                        k=top_k
                    )
                
                return [
                    RetrievalResult(
                        content=doc.page_content,
                        score=1 - score,  # 转换为相似度分数
                        metadata=doc.metadata,
                        source="vector_store"
                    )
                    for doc, score in results
                ]
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # 后备：简单关键词搜索
        return self._simple_search(query, top_k)
    
    def _simple_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """简单关键词搜索"""
        query_terms = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.memory_store:
            content_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & content_terms)
            score = overlap / max(len(query_terms), 1)
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # 排序并返回top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
                source="memory_store"
            )
            for chunk, score in scored_chunks[:top_k]
        ]
    
    def clear(self):
        """清空存储"""
        self.memory_store = []
        if self.vector_store:
            # Chroma不支持直接清空，需要重新创建
            self.vector_store = None
    
    def save(self):
        """持久化存储"""
        # 保存内存存储
        memory_file = self.persist_directory / "memory_store.json"
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(
                [chunk.model_dump() for chunk in self.memory_store],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info(f"Document store saved to {self.persist_directory}")
    
    def load(self):
        """加载存储"""
        memory_file = self.persist_directory / "memory_store.json"
        if memory_file.exists():
            with open(memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.memory_store = [DocumentChunk(**item) for item in data]
            logger.info(f"Loaded {len(self.memory_store)} documents from memory store")
        
        # 尝试加载向量存储
        if LANGCHAIN_AVAILABLE and self.embeddings:
            try:
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embeddings
                )
                logger.info("Loaded vector store")
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")


class RAGModule:
    """RAG模块"""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        self.settings = get_settings()
        persist_dir = persist_directory or Path(self.settings.rag.vector_db_path)
        
        # 初始化各类文档存储
        self.terminology_store = DocumentStore(
            persist_directory=persist_dir / "terminology"
        )
        self.reference_store = DocumentStore(
            persist_directory=persist_dir / "reference"
        )
        self.translation_memory_store = DocumentStore(
            persist_directory=persist_dir / "translation_memory"
        )
        
        # 文本分割器
        self.text_splitter = None
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.rag.chunk_size,
                chunk_overlap=self.settings.rag.chunk_overlap
            )
    
    def add_terminology(
        self,
        terms: List[Dict[str, str]],
        domain: str = "general"
    ):
        """
        添加术语到知识库
        
        Args:
            terms: 术语列表 [{"original": "...", "translation": "...", "context": "..."}]
            domain: 领域
        """
        documents = []
        metadatas = []
        
        for term in terms:
            # 构建可搜索的文档
            doc = f"原文: {term.get('original', '')}\n翻译: {term.get('translation', '')}\n说明: {term.get('context', '')}"
            documents.append(doc)
            metadatas.append({
                "type": "terminology",
                "domain": domain,
                "original": term.get("original", ""),
                "translation": term.get("translation", "")
            })
        
        self.terminology_store.add_documents(documents, metadatas)
    
    def add_reference(
        self,
        content: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加参考资料
        
        Args:
            content: 参考内容
            source: 来源
            metadata: 元数据
        """
        # 分割长文档
        if self.text_splitter and len(content) > self.settings.rag.chunk_size:
            chunks = self.text_splitter.split_text(content)
        else:
            chunks = [content]
        
        base_metadata = metadata or {}
        base_metadata["source"] = source
        base_metadata["type"] = "reference"
        
        metadatas = [base_metadata.copy() for _ in chunks]
        
        self.reference_store.add_documents(chunks, metadatas)
    
    def add_translation_memory(
        self,
        source_text: str,
        translation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加翻译记忆
        
        Args:
            source_text: 原文
            translation: 译文
            metadata: 元数据
        """
        doc = f"原文: {source_text}\n译文: {translation}"
        
        meta = metadata or {}
        meta["type"] = "translation_memory"
        meta["source_text"] = source_text
        meta["translation"] = translation
        
        self.translation_memory_store.add_documents([doc], [meta])
    
    def search_terminology(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """搜索术语"""
        filter_meta = {"type": "terminology"}
        if domain:
            filter_meta["domain"] = domain
        
        return self.terminology_store.search(query, top_k, filter_meta)
    
    def search_reference(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """搜索参考资料"""
        return self.reference_store.search(query, top_k, {"type": "reference"})
    
    def search_translation_memory(
        self,
        source_text: str,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """搜索翻译记忆"""
        return self.translation_memory_store.search(
            source_text, 
            top_k, 
            {"type": "translation_memory"}
        )
    
    def search_all(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, List[RetrievalResult]]:
        """搜索所有知识库"""
        return {
            "terminology": self.search_terminology(query, top_k=top_k),
            "reference": self.search_reference(query, top_k=top_k),
            "translation_memory": self.search_translation_memory(query, top_k=top_k)
        }
    
    def get_context_for_translation(
        self,
        source_text: str,
        max_context_length: int = 2000
    ) -> str:
        """
        获取翻译上下文
        
        Args:
            source_text: 待翻译文本
            max_context_length: 最大上下文长度
            
        Returns:
            格式化的上下文字符串
        """
        results = self.search_all(source_text, top_k=3)
        
        context_parts = []
        current_length = 0
        
        # 添加术语
        if results["terminology"]:
            term_context = "**相关术语:**\n"
            for r in results["terminology"]:
                if current_length + len(r.content) < max_context_length:
                    term_context += f"- {r.content}\n"
                    current_length += len(r.content)
            context_parts.append(term_context)
        
        # 添加翻译记忆
        if results["translation_memory"]:
            tm_context = "**翻译记忆:**\n"
            for r in results["translation_memory"]:
                if current_length + len(r.content) < max_context_length:
                    tm_context += f"- {r.content}\n"
                    current_length += len(r.content)
            context_parts.append(tm_context)
        
        # 添加参考资料
        if results["reference"]:
            ref_context = "**参考资料:**\n"
            for r in results["reference"]:
                if current_length + len(r.content) < max_context_length:
                    ref_context += f"- {r.content}\n"
                    current_length += len(r.content)
            context_parts.append(ref_context)
        
        return "\n\n".join(context_parts)
    
    def save(self):
        """保存所有存储"""
        self.terminology_store.save()
        self.reference_store.save()
        self.translation_memory_store.save()
    
    def load(self):
        """加载所有存储"""
        self.terminology_store.load()
        self.reference_store.load()
        self.translation_memory_store.load()
