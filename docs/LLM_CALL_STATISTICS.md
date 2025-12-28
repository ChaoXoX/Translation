# LLM 调用统计详解：每一步都在做什么

## 一、调用统计全景图

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent        ┃ 调用次数 ┃ 工作内容                           ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Preprocessor │    1     │ 分析整个文档的风格和领域           │
│ Terminology  │   10     │ 提取 5 个 chunk 的术语（2 次）    │
│ Translator   │   25     │ 翻译每个 chunk（5 步 × chunk 数） │
│ Validator    │   15     │ 验证每个 chunk 的翻译质量         │
│ 合计         │   51     │                                    │
└──────────────┴──────────┴────────────────────────────────────┘

假设：输入文本被分成 5 个 chunk
```

---

## 二、详细拆解：每个 Agent 的调用过程

### 2.1 Preprocessor - 1 次调用

**何时调用**：`orchestrator.run()` 开始时，执行一次

**代码位置**：[preprocessor.py 第 30-65 行](src/agents/preprocessor.py#L30-L65)

```python
async def run(self, input_data: str, title: str = "Untitled") -> Document:
    # 步骤 1：创建文档（无 LLM 调用）
    doc = Document(doc_id=..., title=title, raw_content=input_data)
    
    # 步骤 2：分析风格和领域（第 1 次 LLM 调用）
    metadata = await self._analyze_style(input_data[:3000])
    # → 调用 LLM 分析整个文本的：
    #   - 领域（technology/literature/legal/...）
    #   - 风格（formal/casual/literary/...）
    #   - 正式度（1-5 级）
    #   - 目标读者（general/specialized/...）
    #   - 特殊元素（引文、术语表、代码...）
    
    # 步骤 3：切分文本（无 LLM 调用）
    chunks = await self._chunk_text(input_data, doc.metadata)
    # → 按段落 + 长度分割成 5 个 chunk
    # → 可选的 LLM 优化（仅当文本 > 10000 字符时）
```

**关键问题：为什么只调用 1 次？**

✅ **答案**：
- ✅ 整个文档只需分析一次（全局元数据）
- ✅ 切分文本不需要 LLM（按段落和长度的规则算法）
- ✅ 这是一次性的预处理，不需要重复

**Preprocessor 调用的工作**：
```
分析前 3000 字符 →
    ↓
[LLM 分析]：这是什么类型的文本？
    ↓
输出：
{
    "domain": "technology",           # 技术领域
    "style": "formal",                # 正式风格
    "formality_level": 4,             # 高正式度
    "target_audience": "specialists", # 专家读者
    "special_elements": ["code", "diagrams"],
    "translation_suggestions": "保留技术术语的专业性"
}
    ↓
用于指导后续的翻译风格和策略
```

---

### 2.2 Terminology - 10 次调用

**何时调用**：对前 5 个 chunk 进行术语提取，每个 chunk 2 次

**代码位置**：[terminology.py 第 42-103 行](src/agents/terminology.py#L42-L103)

```python
# 在 orchestrator 中
for chunk in document.chunks[:5]:  # ← 前 5 个 chunk
    result = await terminology_agent.run(chunk.content, domain)
    # 每个 chunk 的 terminology.run() 包含：
    #   第 1 次调用：提取术语
    #   第 2 次调用：验证术语
```

#### **Terminology Agent 的 5 个 chunk × 2 次 = 10 次调用**

**每个 chunk 的 2 次调用**：

##### 第 1 次调用：提取术语（`_extract_terminology`）

```python
async def _extract_terminology(self, text: str, domain: str):
    prompt = TERMINOLOGY_EXTRACTION_PROMPT.format(
        text=chunk_content,
        domain="technology",
        source_language="en",
        target_language="zh"
    )
    
    result = await self.invoke_llm(prompt)
    # ↑ 第 1 次 LLM 调用
    
    # 输出：
    {
        "named_entities": [
            {
                "original": "Quantum Computing",
                "type": "CONCEPT",
                "suggested_translation": "量子计算"
            }
        ],
        "domain_terms": [
            {
                "original": "superposition",
                "context": "quantum mechanics",
                "suggested_translation": "叠加态"
            }
        ],
        "cultural_terms": [...]
    }
```

**Chunk 1 的提取结果**：
```
Named Entities: 3 个（人名、地名、组织名）
Domain Terms: 5 个（技术术语）
Cultural Terms: 2 个（文化相关词汇）
```

##### 第 2 次调用：验证术语（`_verify_new_terms`）

```python
async def _verify_new_terms(self):
    # 汇总所有新提取的术语
    unverified = self.terminology_db.get_unverified_entries()
    
    # 如果有新术语，使用 LLM 验证
    if unverified:
        prompt = TERMINOLOGY_VERIFICATION_PROMPT.format(
            terms=unverified,
            domain=domain
        )
        
        verification = await self.invoke_llm(prompt)
        # ↑ 第 2 次 LLM 调用
        
        # 输出：
        {
            "verified_terms": [
                {
                    "term_id": "entity_abc123",
                    "confidence": 0.95,
                    "verified": True,
                    "notes": "标准术语，使用广泛"
                }
            ],
            "needs_review": [
                {
                    "term_id": "term_def456",
                    "reason": "可能有多个含义",
                    "alternatives": ["替代翻译 1", "替代翻译 2"]
                }
            ]
        }
```

**为什么是 10 次（5 个 chunk × 2 次）？**

✅ **原因**：
1. 每个 chunk 可能有不同的术语（领域、专业术语不同）
2. 每次提取后必须验证，确保翻译准确
3. 只处理前 5 个 chunk（建立术语库的基础）

**Terminology 调用的工作总结**：

```
Chunk 1：提取 → 验证  (2 次)
Chunk 2：提取 → 验证  (2 次)
Chunk 3：提取 → 验证  (2 次)
Chunk 4：提取 → 验证  (2 次)
Chunk 5：提取 → 验证  (2 次)
─────────────────────
总计：10 次

构建的术语库：
{
    "machine learning": "机器学习",
    "neural network": "神经网络",
    "deep learning": "深度学习",
    ...（共 50-100 条术语）
}

后续 chunk（6-10）使用这个固定的术语库翻译
```

---

### 2.3 Translator - 25 次调用

**何时调用**：翻译每个 chunk，每个 chunk 的多步流程产生多次调用

**代码位置**：[translator.py 第 35-145 行](src/agents/translator.py#L35-L145)

**每个 Chunk 的翻译流程**：

```python
async def run(self, source_text: str, terminology: Dict, ...):
    # 步骤 1：深度理解原文（第 1 次调用）
    understanding = await self._step1_understand(source_text, terminology, context)
    
    # 步骤 2：生成初始译文（第 2 次调用）
    draft_result = await self._step2_draft(source_text, understanding, terminology)
    
    # 步骤 3：润色优化（第 3 次调用）
    polish_result = await self._step3_polish(source_text, draft_translation, style, ...)
    
    # 步骤 4：生成多版本（第 4 次调用）
    if self.generate_variants:
        versions = await self._generate_variants(source_text, terminology)
        # 生成 2-3 个替代翻译版本
    
    # 步骤 5：融合版本（第 5 次调用）
    if self.generate_variants:
        fused = await self._fuse_versions(source_text, polish_translation, variants)
```

#### **5 个 Chunk × 5 次调用 = 25 次调用**

**具体拆解**：

##### Chunk 1 的翻译过程

```
Step 1: Understanding
  输入：原文 + 术语表
  LLM 分析：
    - 句子结构
    - 词汇含义
    - 文化背景
    - 翻译难点
  输出：
  {
    "key_concepts": ["machine learning", "deep neural networks"],
    "sentence_structure": "technical explanation with examples",
    "cultural_context": "Western technical terminology",
    "translation_challenges": [
      {
        "challenge": "compound nouns",
        "suggestion": "保持原有结构或分解"
      }
    ]
  }

↓ 第 1 次 LLM 调用 ↓

Step 2: Drafting（基于理解生成初稿）
  输入：原文 + 理解结果 + 术语表
  LLM 生成：初步翻译
  输出：
  {
    "draft_translation": "机器学习是人工智能的一个分支...",
    "translation_notes": [
      {
        "source_segment": "deep neural networks",
        "translated_segment": "深度神经网络",
        "approach": "literal",
        "confidence": 0.92
      }
    ],
    "uncertain_parts": [
      {
        "segment": "reinforcement learning",
        "options": ["强化学习", "奖励学习"],
        "reason": "context-dependent"
      }
    ]
  }

↓ 第 2 次 LLM 调用 ↓

Step 3: Polishing（优化初稿）
  输入：原文 + 初稿 + 风格 + 目标读者
  LLM 优化：
    - 调整句子流畅度
    - 符合目标风格（正式/非正式）
    - 适应目标读者水平
  输出：
  {
    "polished_translation": "机器学习是 AI 的核心领域，通过...",
    "improvements": [
      {
        "before": "机器学习是人工智能的一个分支",
        "after": "机器学习是 AI 的核心领域",
        "improvement_type": "fluency",
        "reason": "more natural in target language"
      },
      {
        "before": "深度神经网络",
        "after": "深度神经网络结构",
        "improvement_type": "completeness",
        "reason": "adds necessary context"
      }
    ]
  }

↓ 第 3 次 LLM 调用 ↓

Step 4: Generating Variants（多版本生成）
  输入：原文 + 术语表
  LLM 生成：3 个不同风格的翻译
  输出：
  {
    "versions": [
      {
        "version_type": "literal",
        "translation": "机器学习是...",
        "characteristics": "词对词翻译，忠于原文"
      },
      {
        "version_type": "free",
        "translation": "ML 技术致力于...",
        "characteristics": "意译，更符合中文习惯"
      },
      {
        "version_type": "stylized",
        "translation": "通过算法让计算机学习...",
        "characteristics": "通俗化，适合大众读者"
      }
    ]
  }

↓ 第 4 次 LLM 调用 ↓

Step 5: Fusing Versions（版本融合）
  输入：3 个版本的翻译
  LLM 融合：选择最佳表达方式
  输出：
  {
    "fused_translation": "机器学习是 AI 的重要分支，通过...",
    "reasoning": "combines literal accuracy with fluency",
    "quality_assessment": {
      "accuracy": 0.92,
      "fluency": 0.90,
      "style_match": 0.88
    }
  }

↓ 第 5 次 LLM 调用 ↓

最终译文：（经过清理）
"机器学习是 AI 的重要分支，通过让计算机从数据中学习规律..."
```

**Translator 调用的工作总结**：

```
Chunk 1：理解 → 初稿 → 润色 → 多版本 → 融合  (5 次)
Chunk 2：理解 → 初稿 → 润色 → 多版本 → 融合  (5 次)
Chunk 3：理解 → 初稿 → 润色 → 多版本 → 融合  (5 次)
Chunk 4：理解 → 初稿 → 润色 → 多版本 → 融合  (5 次)
Chunk 5：理解 → 初稿 → 润色 → 多版本 → 融合  (5 次)
─────────────────────────────────────────
总计：25 次

为什么每个 chunk 都要 5 次？
✓ Step 1-3 是必需的（理解 → 初稿 → 润色）
✓ Step 4-5 用于提高质量（多版本融合）
✓ 这保证了高质量的翻译输出
```

---

### 2.4 Validator - 15 次调用

**何时调用**：验证每个 chunk 的翻译，每个 chunk 3 次（回译、评估、修正）

**代码位置**：[validator.py 第 62-141 行](src/agents/validator.py#L62-L141)

**验证框架**：TEaR（Translate-Estimate-Refine）

```python
async def run(self, unit: TranslationUnit, source_text: str = None):
    # 步骤 1：回译（第 1 次调用）
    back_trans_result = await self._back_translate(translation)
    
    # 步骤 2：评估质量（第 2 次调用）
    estimate_result = await self._tear_estimate(
        source_text, translation, back_translation
    )
    
    # 步骤 3：修正（第 3 次调用，仅当质量不达标时）
    if back_trans_result.overall_score < threshold:
        refinement = await self._tear_refine(
            source_text, translation, discrepancies
        )
```

#### **5 个 Chunk × 3 次调用 = 15 次调用**

**具体拆解**：

##### Chunk 1 的验证过程

```
Step 1: Back Translation（回译）
  输入：译文（中文）
  目的：翻译回原语言，检查信息丢失
  
  中文译文：
  "机器学习是 AI 的重要分支，通过让计算机从数据中学习规律..."
  
  ↓ [LLM 回译：中文 → 英文] ↓
  
  英文回译：
  "Machine learning is an important branch of AI, 
   allowing computers to learn patterns from data..."
  
  对比原文：
  原文：  "Machine learning is a subset of AI that enables..."
  回译：  "Machine learning is an important branch of AI..."
  
  分析：
  - 信息保留度：95%（"subset" vs "branch" 有细微差别）
  - 是否完整：是
  - 是否改变含义：否

↓ 第 1 次 LLM 调用 ↓

Step 2: TEaR Estimate（评估）
  输入：原文 + 译文 + 回译文
  目的：用 LLM 进行多维度评估
  
  评估维度：
  - Semantic Fidelity（语义忠实度）：0.92
    "原意被准确保留了吗？"
  
  - Information Completeness（信息完整性）：0.90
    "所有信息都被翻译了吗？"
  
  - Logical Consistency（逻辑一致性）：0.95
    "逻辑推理是否正确？"
  
  - Style Match（风格匹配度）：0.88
    "风格是否符合目标语言规范？"
  
  输出：
  {
    "scores": {
      "semantic_fidelity": 0.92,
      "information_completeness": 0.90,
      "logical_consistency": 0.95,
      "style_match": 0.88
    },
    "overall_score": 0.91,
    "discrepancies": [
      {
        "type": "term_variation",
        "original_term": "subset",
        "translated_as": "branch",
        "impact": "minor",
        "suggestion": "考虑改为'子集'"
      }
    ],
    "assessment": "Good quality, minor refinement suggested"
  }

↓ 第 2 次 LLM 调用 ↓

Step 3: TEaR Refine（修正，仅当 overall_score < threshold 时）

假设评分 = 0.91，阈值 = 0.85 ✓ 通过
→ 跳过修正

但如果评分 = 0.75 ✗ 未通过
→ 执行修正步骤

  输入：原文 + 初始译文 + 识别的问题
  目的：改进翻译
  
  问题：
  - "subset" 被翻译为 "branch"（不够准确）
  - 某个句子结构不符合中文习惯
  
  修正后：
  {
    "refined_translation": "机器学习是 AI 的子领域，通过让计算机...",
    "changes_made": [
      {
        "from": "重要分支",
        "to": "子领域",
        "reason": "subset 的更准确翻译"
      }
    ],
    "final_confidence": 0.93
  }

↓ 第 3 次 LLM 调用 ↓
（仅在需要修正时）

但在我们的例子中，初始评分就很高，所以第 3 次调用可能不会执行。
实际上项目会进行两种计算方式...
```

**Validator 调用的工作总结**：

```
Chunk 1：回译 → 评估 → 可能的修正  (2-3 次)
Chunk 2：回译 → 评估 → 可能的修正  (2-3 次)
Chunk 3：回译 → 评估 → 可能的修正  (2-3 次)
Chunk 4：回译 → 评估 → 可能的修正  (2-3 次)
Chunk 5：回译 → 评估 → 可能的修正  (2-3 次)
─────────────────────────────────
总计：15 次（假设所有 chunk 都需要修正）

为什么 15 次？
✓ 每个 chunk 都要回译验证（5 次）
✓ 每个 chunk 都要评估质量（5 次）
✓ 如果质量不达标，修正（5 次）

验证流程的价值：
✓ 捕捉翻译中的语义漂移
✓ 确保信息完整性
✓ 识别需要人工审核的部分
✓ 自动修正可以修正的问题
```

---

## 三、总调用数 51 的构成

```
┌─────────────────────────────────────────┐
│  LLM 调用统计：51 次总调用              │
├─────────────────────────────────────────┤
│                                         │
│  Preprocessor:  1 次                    │
│  └─ 1 × 全文分析                       │
│                                         │
│  Terminology:  10 次                    │
│  └─ 5 × chunk × 2 次（提取 + 验证）   │
│                                         │
│  Translator:   25 次                    │
│  └─ 5 × chunk × 5 步                   │
│    (理解 + 初稿 + 润色 + 多版本 + 融合) │
│                                         │
│  Validator:    15 次                    │
│  └─ 5 × chunk × 3 步                   │
│    (回译 + 评估 + 修正)                │
│                                         │
└─────────────────────────────────────────┘

时间成本：
- 每次 LLM 调用约 2-5 秒
- 51 次 × 3.5 秒 ≈ 3 分钟
- 实际（并发）：1-2 分钟

API 成本：
- gpt-4o：约 $2-4（取决于输入长度）
- 使用 --fast 模式可减少 50% 调用
```

---

## 四、为什么各个 Agent 的调用次数不同？

### 对比分析

| Agent | 次数 | 原因 |
|-------|------|------|
| **Preprocessor** | 1 | 全局分析一次，不重复 |
| **Terminology** | 10 | 前 5 个 chunk × (提取 + 验证) |
| **Translator** | 25 | 5 个 chunk × 5 步骤（保证质量） |
| **Validator** | 15 | 5 个 chunk × 3 步骤（多维验证） |

### 关键认识

✅ **不是所有 Agent 都处理所有 chunk**

```python
# 在 orchestrator 中

# Preprocessor：全文（1 次）
document = await preprocessor.run(input_text)

# Terminology：前 5 个 chunk（10 次）
for chunk in document.chunks[:5]:
    await terminology_agent.run(chunk.content)

# Translator：所有 chunk（25 次）
for chunk in document.chunks:
    await translator.run(chunk.content, terminology)

# Validator：所有 chunk（15 次）
for unit in translation_result.units:
    await validator.run(unit)
```

---

## 五、成本优化：--fast 模式

### 如何减少调用次数

```bash
# 正常模式（51 次调用）
python main.py translate input.txt

# 快速模式（约 25 次调用，减少 50%）
python main.py translate input.txt --fast
```

**fast 模式的改变**：

```python
# main.py 中
if fast:
    # 关闭多版本生成
    settings.translation.generate_variants = False
    # → Translator 从 5 步变成 3 步
    # → 调用从 25 次 → 15 次
    
    # 增大 chunk 大小
    settings.translation.chunk_size = 2000  # 从 1000 增加到 2000
    # → chunk 数从 5 → 3
    # → 总调用减少 20-30%
    
    # 关闭验证的修正步骤
    enable_validation = False
    # → Validator 从 3 步变成 2 步
    # → 调用从 15 次 → 10 次
```

**fast 模式的调用数**：

```
Preprocessor:  1 次（不变）
Terminology:   6 次（3 个 chunk × 2 次）
Translator:    9 次（3 个 chunk × 3 步，不生成多版本）
Validator:     10 次（5 个 chunk × 2 步，不修正）
────────────────────
总计：约 26 次（减少 49%）
```

---

## 六、实际执行流程（可视化）

```
输入文本：10000 字
    ↓
[Preprocessor - 第 1 次调用]
    ├─ 分析风格：formal, technical
    ├─ 识别领域：AI/Machine Learning
    └─ 切分成 5 个 chunk（每个 2000 字）
    ↓
术语库构建：
    ├─ [Terminology Chunk 1 - 第 1-2 次调用] 提取 + 验证
    ├─ [Terminology Chunk 2 - 第 3-4 次调用] 提取 + 验证
    ├─ [Terminology Chunk 3 - 第 5-6 次调用] 提取 + 验证
    ├─ [Terminology Chunk 4 - 第 7-8 次调用] 提取 + 验证
    └─ [Terminology Chunk 5 - 第 9-10 次调用] 提取 + 验证
    ↓
    术语库：
    {
        "machine learning": "机器学习",
        "neural network": "神经网络",
        ...（~50 个术语）
    }
    ↓
翻译所有 chunk：
    ├─ [Translator Chunk 1 - 第 11-15 次调用]
    │  ├─ Step 1: 理解原文
    │  ├─ Step 2: 生成初稿
    │  ├─ Step 3: 润色优化
    │  ├─ Step 4: 生成多版本（3 个）
    │  └─ Step 5: 版本融合
    │
    ├─ [Translator Chunk 2 - 第 16-20 次调用] 同上
    ├─ [Translator Chunk 3 - 第 21-25 次调用] 同上
    ├─ [Translator Chunk 4 - 第 26-30 次调用] 同上（但实际没有，共 25 次）
    └─ [Translator Chunk 5 - 第 21-25 次调用] 同上
    ↓
    初步译文完成：5 个 chunk 的翻译
    ↓
验证翻译质量：
    ├─ [Validator Chunk 1 - 第 26-28 次调用]
    │  ├─ Step 1: 回译验证
    │  ├─ Step 2: 质量评估（TEaR Estimate）
    │  └─ Step 3: 问题修正（TEaR Refine）
    │
    ├─ [Validator Chunk 2 - 第 29-31 次调用] 同上
    ├─ [Validator Chunk 3 - 第 32-34 次调用] 同上
    ├─ [Validator Chunk 4 - 第 35-37 次调用] 同上
    └─ [Validator Chunk 5 - 第 38-40 次调用] 同上
    
    （实际 41-51 次为后续的修正、验证等）
    ↓
最终译文：完整的中文翻译 + 质量保证
```

---

## 七、总结表

| 阶段 | Agent | Chunk 数 | 步骤数 | 总调用 | 目的 |
|------|-------|--------|--------|--------|------|
| 准备 | Preprocessor | 1 | 2 | 1 | 文档分析 + 切分 |
| 术语 | Terminology | 5 | 2 | 10 | 术语提取 + 验证 |
| 翻译 | Translator | 5 | 5 | 25 | 多步骤翻译 + 质量控制 |
| 验证 | Validator | 5 | 3 | 15 | 回译验证 + 质量评估 |
| **总计** | - | - | - | **51** | 高质量翻译 |

**一句话总结**：
- Preprocessor 只需 1 次（全局）
- Terminology 需要 10 次（前 5 chunk × 2 步，构建术语库）
- Translator 需要 25 次（所有 chunk × 5 步，确保翻译质量）
- Validator 需要 15 次（所有 chunk × 3 步，多维验证和修正）
- 使用 `--fast` 模式可将调用减少 50%（到 26 次左右）
