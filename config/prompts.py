"""
提示词模板库

包含所有Agent使用的高级提示词策略：
- 多步骤引导
- 角色扮演
- 自我反馈
- 回译验证 (TEaR)
"""

# ============================================================
# 预处理Agent提示词
# ============================================================

STYLE_ANALYSIS_PROMPT = """你是一位资深的文本分析专家，擅长识别文本的领域和语体风格。

请分析以下文本，并以JSON格式返回分析结果：

**文本内容:**
{text}

**分析要求:**
1. 领域识别: 判断文本所属领域（如：文学、法律、技术、新闻、娱乐、学术等）
2. 语体风格: 识别文本风格（如：正式、口语化、文学性、专业技术性、幽默诙谐等）
3. 目标读者: 推测文本面向的读者群体
4. 特殊标记: 标注可能需要特殊处理的元素（如俚语、专业术语、文化负载词）

**输出格式 (JSON):**
```json
{{
    "domain": "文本领域",
    "style": "语体风格",
    "formality_level": "正式程度 (1-5)",
    "target_audience": "目标读者",
    "special_elements": ["特殊元素1", "特殊元素2"],
    "translation_suggestions": "翻译建议"
}}
```

请确保分析准确、全面，这将直接影响后续的翻译质量。"""

TEXT_CHUNKING_PROMPT = """你是一位专业的文本结构分析专家。

请将以下文本按照逻辑单元进行切分，确保：
1. 每个单元保持语义完整性
2. 保留必要的上下文信息
3. 标注段落边界和章节信息

**文本内容:**
{text}

**切分要求:**
- 最大单元长度: {max_chunk_size} 字符
- 重叠区域: {overlap_size} 字符
- 保持句子完整性，不在句子中间切分
- 对话段落保持对话完整性

**输出格式 (JSON数组):**
```json
[
    {{
        "chunk_id": 1,
        "content": "切分后的文本内容",
        "context_before": "前文上下文摘要",
        "context_after": "后文上下文摘要（如有）",
        "metadata": {{
            "paragraph_type": "叙述/对话/描写",
            "has_special_terms": true/false,
            "estimated_difficulty": "简单/中等/困难"
        }}
    }}
]
```"""


# ============================================================
# 术语与实体Agent提示词
# ============================================================

TERMINOLOGY_EXTRACTION_PROMPT = """你是一位专业的翻译术语专家，精通{source_language}和{target_language}。

请从以下文本中提取需要特殊处理的术语和实体：

**文本内容:**
{text}

**领域背景:** {domain}
**翻译方向:** {source_language} → {target_language}

**提取要求:**

1. **命名实体 (NER)**
   - 人名（包括绰号、别名）
   - 地名（城市、街道、场所）
   - 机构名（公司、组织、品牌）
   - 作品名（书籍、歌曲、电影）

2. **领域术语**
   - 专业词汇
   - 行业特定表达
   - 特殊语义词汇（如"beef"在说唱中表示"矛盾/争执"）

3. **文化负载词**
   - 无直接对等翻译的表达
   - 俚语和习语
   - 文化特定概念

**输出格式 (JSON):**
```json
{{
    "named_entities": [
        {{
            "original": "原文",
            "type": "PERSON/LOCATION/ORGANIZATION/WORK",
            "context": "出现上下文",
            "suggested_translation": "建议译名",
            "alternatives": ["备选译名1", "备选译名2"],
            "confidence": 0.9,
            "notes": "备注说明"
        }}
    ],
    "domain_terms": [
        {{
            "original": "原文术语",
            "domain_meaning": "领域特定含义",
            "general_meaning": "一般含义",
            "suggested_translation": "建议翻译",
            "translation_strategy": "直译/意译/音译/音意结合",
            "confidence": 0.85
        }}
    ],
    "cultural_terms": [
        {{
            "original": "原文表达",
            "cultural_context": "文化背景说明",
            "semantic_explanation": "语义解释",
            "suggested_translation": "建议翻译",
            "translation_strategy": "意译/语义补偿/直译加注",
            "confidence": 0.8
        }}
    ]
}}
```

请确保提取全面，译名建议准确。对于不确定的翻译，请降低置信度并标注需要人工确认。"""

TERMINOLOGY_VERIFICATION_PROMPT = """你是一位翻译术语审核专家。

请验证以下术语翻译的准确性和一致性：

**待验证术语:**
{terminology_list}

**参考术语库:**
{reference_terminology}

**验证要求:**
1. 检查译名是否符合行业规范
2. 检查与已有术语库是否一致
3. 检查是否存在歧义或错误
4. 对于人名，检查是否有约定俗成的译法

**输出格式 (JSON):**
```json
{{
    "verified_terms": [
        {{
            "original": "原文",
            "approved_translation": "确认的翻译",
            "status": "approved/modified/needs_review",
            "modification_reason": "修改原因（如有）"
        }}
    ],
    "conflicts": [
        {{
            "original": "原文",
            "new_suggestion": "新建议",
            "existing_translation": "已有翻译",
            "recommendation": "建议采用哪个"
        }}
    ]
}}
```"""

SLANG_LOOKUP_PROMPT = """你是一位精通美国流行文化和俚语的语言专家。

请查询以下俚语/非正式表达的准确含义：

**待查询词汇:** {term}
**出现上下文:** {context}
**领域背景:** {domain}

请提供：
1. 该词在当前上下文中的具体含义
2. 词源和使用背景
3. 中文翻译建议（考虑语境适应性）
4. 常见误解或翻译陷阱

**注意:** 如果这是说唱/嘻哈领域的俚语，请参考Urban Dictionary等资源确保准确性。

**输出格式 (JSON):**
```json
{{
    "term": "原词",
    "primary_meaning": "主要含义",
    "contextual_meaning": "上下文特定含义",
    "origin": "词源",
    "usage_notes": "使用说明",
    "translation_options": [
        {{
            "translation": "翻译选项",
            "style": "风格（口语化/正式/中性）",
            "suitability": "适用性评分 (1-5)"
        }}
    ],
    "common_mistakes": ["常见错误翻译1", "常见错误翻译2"]
}}
```"""


# ============================================================
# 翻译Agent提示词 - 多步骤引导
# ============================================================

TRANSLATION_STEP1_UNDERSTANDING = """你是一位资深翻译专家，正在进行{source_language}到{target_language}的翻译工作。

**第一步：深度理解原文**

请仔细分析以下文本的含义：

**原文:**
{source_text}

**已确认术语表:**
{terminology}

**上下文信息:**
{context}

**分析要求:**
1. 梳理文本的核心信息和主旨
2. 识别句子结构（主干成分、从句层级）
3. 标注语气和情感色彩
4. 识别潜在的翻译难点

**输出格式 (JSON):**
```json
{{
    "main_idea": "核心内容概述",
    "sentence_structure": [
        {{
            "sentence": "原句",
            "structure_analysis": "结构分析",
            "key_components": {{
                "subject": "主语",
                "predicate": "谓语",
                "clauses": ["从句1", "从句2"]
            }}
        }}
    ],
    "tone": "语气特征",
    "emotion": "情感色彩",
    "translation_challenges": ["难点1", "难点2"],
    "cultural_notes": "文化注释"
}}
```"""

TRANSLATION_STEP2_DRAFT = """**第二步：生成初始译文**

基于你对原文的理解，现在生成{target_language}译文。

**原文:**
{source_text}

**理解分析:**
{understanding}

**必须使用的术语翻译:**
{terminology}

**翻译要求:**
1. 准确传达原文含义
2. 严格遵循术语表中的翻译
3. 保持原文的语气和风格
4. 确保译文通顺自然

**输出格式 (JSON):**
```json
{{
    "draft_translation": "初始译文",
    "translation_notes": [
        {{
            "source_segment": "原文片段",
            "translated_segment": "对应译文",
            "approach": "采用的翻译策略",
            "confidence": 0.9
        }}
    ],
    "uncertain_parts": [
        {{
            "segment": "不确定的部分",
            "options": ["选项1", "选项2"],
            "reason": "不确定原因"
        }}
    ]
}}
```"""

TRANSLATION_STEP3_POLISH = """**第三步：润色优化译文**

请对初始译文进行润色和优化。

**原文:**
{source_text}

**初始译文:**
{draft_translation}

**目标风格:** {style}
**目标读者:** {target_audience}

**润色要求:**
1. 优化语言表达，使其更加地道自然
2. 调整句式结构，符合{target_language}的表达习惯
3. 确保专有名词和术语翻译一致
4. 检查是否有漏译或误译

**输出格式 (JSON):**
```json
{{
    "polished_translation": "润色后的译文",
    "improvements": [
        {{
            "before": "修改前",
            "after": "修改后",
            "improvement_type": "改进类型（流畅性/准确性/风格）",
            "reason": "修改原因"
        }}
    ],
    "final_confidence": 0.92
}}
```"""


# ============================================================
# 多版本生成提示词
# ============================================================

MULTI_VERSION_PROMPT = """请为以下文本生成{variant_count}个不同风格的翻译版本：

**原文:**
{source_text}

**翻译方向:** {source_language} → {target_language}

**必须遵循的术语表:**
{terminology}

**版本要求:**
1. **直译版**: 尽可能贴近原文结构和表达
2. **意译版**: 注重意思传达，可适当调整表达方式
3. **风格化版**: 在准确基础上，增强文学性或口语化特色

**输出格式 (JSON):**
```json
{{
    "versions": [
        {{
            "version_type": "直译版/意译版/风格化版",
            "translation": "翻译内容",
            "characteristics": "版本特点说明",
            "suitable_for": "适用场景"
        }}
    ],
    "recommendation": {{
        "preferred_version": "推荐版本",
        "reason": "推荐理由"
    }}
}}
```"""

VERSION_FUSION_PROMPT = """请融合以下多个翻译版本，生成最优译文：

**原文:**
{source_text}

**版本1 (直译):**
{version_1}

**版本2 (意译):**
{version_2}

**版本3 (风格化):**
{version_3}

**融合要求:**
1. 取各版本之长
2. 保持术语一致性
3. 确保整体风格统一
4. 追求信达雅的平衡

**输出格式 (JSON):**
```json
{{
    "fused_translation": "融合后的最终译文",
    "fusion_analysis": [
        {{
            "segment": "片段",
            "source_version": "取自哪个版本",
            "reason": "选择原因"
        }}
    ],
    "quality_assessment": {{
        "accuracy": 0.95,
        "fluency": 0.92,
        "style_match": 0.88
    }}
}}
```"""


# ============================================================
# 回译验证提示词 (TEaR Framework)
# ============================================================

BACK_TRANSLATION_PROMPT = """请将以下{target_language}译文回译为{source_language}：

**译文:**
{translation}

**回译要求:**
1. 尽可能准确地还原原意
2. 保持相似的句式结构
3. 不要参考原文，仅基于译文进行回译

**输出格式:**
```json
{{
    "back_translation": "回译结果",
    "notes": "回译过程中的观察"
}}
```"""

TEAR_ESTIMATE_PROMPT = """**TEaR框架 - Estimate阶段**

请比较原文与回译，评估翻译质量：

**原文 (Source):**
{source_text}

**译文 (Translation):**
{translation}

**回译 (Back-translation):**
{back_translation}

**评估维度:**
1. **语义保真度**: 回译与原文的语义一致程度 (0-1)
2. **信息完整性**: 是否有信息丢失或增加 (0-1)
3. **逻辑一致性**: 逻辑关系是否保持 (0-1)
4. **风格匹配度**: 语气和风格是否一致 (0-1)

**输出格式 (JSON):**
```json
{{
    "scores": {{
        "semantic_fidelity": 0.9,
        "information_completeness": 0.95,
        "logical_consistency": 0.88,
        "style_match": 0.85
    }},
    "overall_score": 0.895,
    "discrepancies": [
        {{
            "type": "偏差类型",
            "original_segment": "原文片段",
            "back_translated": "回译片段",
            "issue": "问题描述",
            "severity": "严重程度 (low/medium/high)"
        }}
    ],
    "assessment": "整体评估说明"
}}
```"""

TEAR_REFINE_PROMPT = """**TEaR框架 - Refine阶段**

基于评估结果，请修正译文中的问题：

**原文:**
{source_text}

**当前译文:**
{translation}

**发现的问题:**
{discrepancies}

**修正要求:**
1. 针对每个问题进行修正
2. 保持术语一致性
3. 不影响正确部分的表达
4. 确保修正后更加准确自然

**输出格式 (JSON):**
```json
{{
    "refined_translation": "修正后的译文",
    "corrections": [
        {{
            "original_issue": "原问题",
            "before": "修正前",
            "after": "修正后",
            "correction_type": "修正类型"
        }}
    ],
    "remaining_concerns": ["仍存在的问题（如有）"],
    "final_confidence": 0.95
}}
```"""


# ============================================================
# 人机协作提示词
# ============================================================

HUMAN_REVIEW_REQUEST_PROMPT = """**需要人工审核**

以下翻译决策需要您的确认：

**审核类型:** {review_type}
**原文片段:** {source_segment}
**上下文:** {context}

**系统建议:**
{suggestions}

**请选择或提供您的决定:**
1. 采用系统建议
2. 选择备选方案: {alternatives}
3. 提供自定义翻译

**重要性:** {importance}
**置信度:** {confidence}
"""

HUMAN_FEEDBACK_INTEGRATION_PROMPT = """请根据人工反馈调整翻译：

**原文:**
{source_text}

**原译文:**
{original_translation}

**人工反馈:**
{human_feedback}

**调整要求:**
1. 整合人工反馈意见
2. 确保与已确认术语一致
3. 保持翻译风格统一

**输出格式 (JSON):**
```json
{{
    "adjusted_translation": "调整后的译文",
    "changes_made": "根据反馈所做的修改说明"
}}
```"""


# ============================================================
# 质量评估提示词
# ============================================================

QUALITY_ASSESSMENT_PROMPT = """请按照MQM（Multidimensional Quality Metrics）框架评估以下翻译：

**原文:**
{source_text}

**译文:**
{translation}

**评估维度:**
1. **准确性 (Accuracy)**: 语义准确、无误译漏译
2. **流畅性 (Fluency)**: 语法正确、表达自然
3. **术语一致性 (Terminology)**: 专有名词和术语处理
4. **风格适当性 (Style)**: 符合目标文本类型要求
5. **文化适应性 (Locale)**: 文化转换适当

**输出格式 (JSON):**
```json
{{
    "mqm_scores": {{
        "accuracy": {{
            "score": 9.0,
            "issues": ["问题1", "问题2"]
        }},
        "fluency": {{
            "score": 8.5,
            "issues": []
        }},
        "terminology": {{
            "score": 9.5,
            "issues": []
        }},
        "style": {{
            "score": 8.0,
            "issues": ["风格问题"]
        }},
        "locale": {{
            "score": 8.5,
            "issues": []
        }}
    }},
    "overall_score": 8.7,
    "summary": "总体评价",
    "improvement_suggestions": ["改进建议1", "改进建议2"]
}}
```"""
