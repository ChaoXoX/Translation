"""
翻译结果数据模型

定义翻译过程中的各种结果结构，包括：
- 翻译单元
- 翻译版本
- 回译结果
- 质量评估
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TranslationStatus(str, Enum):
    """翻译状态"""
    PENDING = "pending"           # 待翻译
    IN_PROGRESS = "in_progress"   # 翻译中
    DRAFT = "draft"               # 初稿完成
    REFINED = "refined"           # 已优化
    VALIDATED = "validated"       # 已验证
    HUMAN_REVIEW = "human_review" # 等待人工审核
    APPROVED = "approved"         # 已批准
    FINAL = "final"               # 最终版本


class VersionType(str, Enum):
    """翻译版本类型"""
    LITERAL = "literal"           # 直译版
    FREE = "free"                 # 意译版
    STYLIZED = "stylized"         # 风格化版
    FUSED = "fused"               # 融合版
    HUMAN_EDITED = "human_edited" # 人工编辑版


class TranslationNote(BaseModel):
    """翻译注释"""
    source_segment: str           # 原文片段
    translated_segment: str       # 对应译文
    approach: str                 # 采用的翻译策略
    confidence: float = 0.9       # 置信度
    note: str = ""                # 备注


class UncertainPart(BaseModel):
    """不确定部分"""
    segment: str                  # 不确定的部分
    options: List[str]            # 可选翻译
    reason: str                   # 不确定原因
    requires_human_review: bool = False


class TranslationVersion(BaseModel):
    """翻译版本"""
    version_id: str
    version_type: VersionType
    translation: str
    
    # 版本特点
    characteristics: str = ""     # 版本特点说明
    suitable_for: str = ""        # 适用场景
    
    # 质量评分
    accuracy_score: float = 0.0
    fluency_score: float = 0.0
    style_score: float = 0.0
    overall_score: float = 0.0
    
    # 是否推荐
    is_recommended: bool = False
    recommendation_reason: str = ""
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)


class Improvement(BaseModel):
    """改进记录"""
    before: str
    after: str
    improvement_type: str  # 流畅性/准确性/风格
    reason: str


class TranslationUnit(BaseModel):
    """翻译单元"""
    unit_id: str
    chunk_id: str                 # 对应的文档chunk ID
    
    # 原文
    source_text: str
    source_language: str = "en"
    target_language: str = "zh"
    
    # 上下文
    context_before: str = ""
    context_after: str = ""
    
    # 理解分析结果
    understanding: Dict[str, Any] = Field(default_factory=dict)
    
    # 翻译版本
    versions: List[TranslationVersion] = Field(default_factory=list)
    
    # 当前最佳翻译
    best_translation: str = ""
    
    # 翻译注释
    notes: List[TranslationNote] = Field(default_factory=list)
    
    # 不确定部分
    uncertain_parts: List[UncertainPart] = Field(default_factory=list)
    
    # 改进记录
    improvements: List[Improvement] = Field(default_factory=list)
    
    # 状态
    status: TranslationStatus = TranslationStatus.PENDING
    
    # 置信度
    final_confidence: float = 0.0
    
    # 使用的术语
    used_terminology: Dict[str, str] = Field(default_factory=dict)
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_version(self, version: TranslationVersion):
        """添加翻译版本"""
        self.versions.append(version)
        self.updated_at = datetime.now()
    
    def set_best_translation(self, translation: str, confidence: float = 0.9):
        """设置最佳翻译"""
        self.best_translation = translation
        self.final_confidence = confidence
        self.updated_at = datetime.now()
    
    def needs_human_review(self) -> bool:
        """是否需要人工审核"""
        return (
            self.status == TranslationStatus.HUMAN_REVIEW or
            self.final_confidence < 0.7 or
            any(p.requires_human_review for p in self.uncertain_parts)
        )


class BackTranslationResult(BaseModel):
    """回译结果"""
    result_id: str
    unit_id: str                  # 对应的翻译单元ID
    
    # 原始翻译
    original_translation: str
    
    # 回译结果
    back_translation: str
    
    # 评估分数 (TEaR框架)
    semantic_fidelity: float = 0.0       # 语义保真度
    information_completeness: float = 0.0  # 信息完整性
    logical_consistency: float = 0.0     # 逻辑一致性
    style_match: float = 0.0             # 风格匹配度
    overall_score: float = 0.0           # 综合评分
    
    # 发现的差异
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 评估说明
    assessment: str = ""
    
    # 是否通过验证
    passed: bool = False
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    
    def calculate_overall_score(self):
        """计算综合评分"""
        self.overall_score = (
            self.semantic_fidelity * 0.4 +
            self.information_completeness * 0.3 +
            self.logical_consistency * 0.2 +
            self.style_match * 0.1
        )
        self.passed = self.overall_score >= 0.85


class Correction(BaseModel):
    """修正记录"""
    original_issue: str
    before: str
    after: str
    correction_type: str


class RefinementResult(BaseModel):
    """优化结果（TEaR Refine阶段）"""
    result_id: str
    unit_id: str
    
    # 优化前后
    original_translation: str
    refined_translation: str
    
    # 修正记录
    corrections: List[Correction] = Field(default_factory=list)
    
    # 剩余问题
    remaining_concerns: List[str] = Field(default_factory=list)
    
    # 最终置信度
    final_confidence: float = 0.0
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)


class MQMScore(BaseModel):
    """MQM维度评分"""
    score: float = 0.0
    issues: List[str] = Field(default_factory=list)


class QualityAssessment(BaseModel):
    """质量评估结果 (基于MQM框架)"""
    assessment_id: str
    unit_id: str
    
    # 原文和译文
    source_text: str
    translation: str
    
    # MQM维度评分
    accuracy: MQMScore = Field(default_factory=MQMScore)
    fluency: MQMScore = Field(default_factory=MQMScore)
    terminology: MQMScore = Field(default_factory=MQMScore)
    style: MQMScore = Field(default_factory=MQMScore)
    locale: MQMScore = Field(default_factory=MQMScore)
    
    # 综合评分
    overall_score: float = 0.0
    
    # 总结
    summary: str = ""
    
    # 改进建议
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now)
    
    def calculate_overall_score(self):
        """计算综合评分"""
        weights = {
            'accuracy': 0.35,
            'fluency': 0.25,
            'terminology': 0.20,
            'style': 0.10,
            'locale': 0.10
        }
        self.overall_score = (
            self.accuracy.score * weights['accuracy'] +
            self.fluency.score * weights['fluency'] +
            self.terminology.score * weights['terminology'] +
            self.style.score * weights['style'] +
            self.locale.score * weights['locale']
        )


class TranslationResult(BaseModel):
    """完整翻译结果"""
    result_id: str
    doc_id: str
    
    # 翻译单元
    units: List[TranslationUnit] = Field(default_factory=list)
    
    # 合并后的完整译文
    full_translation: str = ""
    
    # 质量评估
    quality_assessment: Optional[QualityAssessment] = None
    
    # 统计信息
    total_units: int = 0
    completed_units: int = 0
    human_reviewed_units: int = 0
    
    # 使用的术语表
    terminology_used: Dict[str, str] = Field(default_factory=dict)
    
    # 状态
    status: TranslationStatus = TranslationStatus.PENDING
    
    # 时间戳
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_unit(self, unit: TranslationUnit):
        """添加翻译单元"""
        self.units.append(unit)
        self.total_units = len(self.units)
    
    def update_progress(self):
        """更新进度"""
        self.completed_units = sum(
            1 for u in self.units 
            if u.status in [TranslationStatus.APPROVED, TranslationStatus.FINAL]
        )
        self.human_reviewed_units = sum(
            1 for u in self.units
            if u.status == TranslationStatus.APPROVED
        )
    
    def merge_translations(self) -> str:
        """合并所有翻译单元的译文"""
        translations = []
        # 优先按 chunk 顺序合并，避免 UUID 打乱顺序
        def unit_sort_key(u: TranslationUnit) -> str:
            return getattr(u, "chunk_id", getattr(u, "unit_id", ""))

        for unit in sorted(self.units, key=unit_sort_key):
            text = (unit.best_translation or "").strip()
            if not text:
                # 回退策略：优先使用推荐版本，其次使用最后一个版本，最后使用第一个版本
                chosen = None
                if unit.versions:
                    recommended = [v for v in unit.versions if getattr(v, "is_recommended", False)]
                    if recommended:
                        chosen = recommended[0]
                    else:
                        chosen = unit.versions[-1] or unit.versions[0]
                text = (chosen.translation if chosen else "").strip()
            if text:
                translations.append(text)
        self.full_translation = "\n\n".join(translations)
        return self.full_translation
    
    def get_progress(self) -> float:
        """获取翻译进度"""
        if self.total_units == 0:
            return 0.0
        return self.completed_units / self.total_units
