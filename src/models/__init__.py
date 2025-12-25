"""数据模型模块"""
from .document import Document, DocumentChunk, DocumentMetadata
from .terminology import (
    TerminologyEntry, 
    NamedEntity, 
    DomainTerm, 
    CulturalTerm,
    TerminologyDatabase
)
from .translation import (
    TranslationUnit,
    TranslationVersion,
    TranslationResult,
    BackTranslationResult,
    QualityAssessment
)
