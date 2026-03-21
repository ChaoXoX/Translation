# 项目结构详解

## 一、目录树总览

```
Translation/
├── README.md                        # 项目说明与快速开始
├── requirements.txt                 # Python 依赖包列表
├── main.py                          # 命令行入口（CLI）
│
├── config/                          # 全局配置层
│   ├── __init__.py
│   ├── settings.py                  # 配置模型（Pydantic）+ 环境变量读取
│   └── prompts.py                   # 所有 Agent 使用的提示词模板
│
├── src/                             # 核心源代码
│   ├── __init__.py
│   ├── agents/                      # 智能体（Agent）层
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Agent 抽象基类
│   │   ├── preprocessor.py          # 预处理 Agent
│   │   ├── terminology.py           # 术语与实体 Agent
│   │   ├── translator.py            # 翻译 Agent
│   │   └── validator.py             # 验证 Agent
│   │
│   ├── workflow/                    # 工作流编排层
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # 顺序式工作流总指挥（当前主用）
│   │   └── pipeline.py              # 基于 LangGraph 的图状态机流水线（扩展备用）
│   │
│   ├── models/                      # 数据模型层（Pydantic）
│   │   ├── __init__.py
│   │   ├── document.py              # 文档与切分单元模型
│   │   ├── terminology.py           # 术语库与术语条目模型
│   │   └── translation.py           # 翻译结果与翻译单元模型
│   │
│   ├── knowledge/                   # 知识库层
│   │   ├── __init__.py
│   │   ├── rag.py                   # RAG（检索增强生成）模块
│   │   └── memory.py                # 翻译记忆管理模块
│   │
│   ├── human_loop/                  # 人机协作层
│   │   ├── __init__.py
│   │   └── interface.py             # 交互式审核界面与任务调度
│   │
│   └── utils/                       # 工具函数层
│       ├── __init__.py
│       ├── text_utils.py            # 文本清洗、统计、切分工具
│       └── file_utils.py            # 文件读写（.txt / .md / .epub）工具
│
├── data/                            # 数据目录（运行时生成）
│   ├── input/                       # 待翻译的输入文本
│   │   └── sample_text.txt          # 示例输入文件
│   ├── output/                      # 译文输出目录
│   ├── terminology/                 # 持久化术语库（JSON）
│   ├── cache/                       # 缓存文件
│   ├── checkpoints/                 # 翻译进度检查点（pickle）
│   └── vector_db/                   # RAG 向量数据库（Chroma/FAISS）
│
├── docs/                            # 项目文档
│   ├── PROJECT_STRUCTURE.md         # 本文件：项目结构详解
│   ├── CHUNK_SPLITTING_EXPLAINED.md # 文本切分策略说明
│   ├── LLM_CALL_STATISTICS.md       # LLM 调用统计说明
│   ├── LANGCHAIN_USAGE_DETAILS.md   # LangChain 使用详情
│   ├── LANGCHAIN_VS_LANGGRAPH.md    # LangChain 与 LangGraph 对比
│   └── LANGGRAPH_DETAILED_EXPLANATION.md  # LangGraph 架构深度解析
│
└── examples/                        # 示例与演示脚本
    └── (demo scripts)
```

---

## 二、各层详细说明

### 2.1 入口层：`main.py`

系统的命令行入口，基于 **[Typer](https://typer.tiangolo.com/)** 框架构建，提供以下子命令：

| 命令 | 说明 |
|------|------|
| `translate` | 翻译书籍文本（核心命令） |
| `analyze`   | 静态分析文本，提取术语与结构信息 |
| `review`    | 人工审核模式，加载待审核任务文件进行交互式审核 |
| `demo`      | 使用内置示例文本运行演示翻译 |
| `version`   | 显示版本信息 |

`translate` 命令的主要参数：

```
--source / -s     源语言（默认 en）
--target / -t     目标语言（默认 zh）
--domain / -d     领域（general / literature / technical / legal / entertainment）
--style           翻译风格（formal / casual / literary）
--validation      启用/禁用回译验证（默认启用）
--human-review    启用/禁用人工审核（默认禁用）
--terminology -T  外部术语库文件路径
--fast            快速模式：关闭多版本生成与验证，扩大 chunk 尺寸
--debug           调试模式，记录详细日志
```

---

### 2.2 配置层：`config/`

#### `config/settings.py`

使用 **Pydantic** `BaseModel` 定义分层配置，自动从环境变量（`.env` 文件）读取敏感信息：

| 配置类 | 作用 |
|--------|------|
| `LLMSettings` | LLM API Key、Base URL、模型名称、温度、备用模型 |
| `TranslationSettings` | 源/目标语言、风格、领域、chunk 大小、多版本开关、回译阈值 |
| `RAGSettings` | Embedding 模型、向量库路径、TopK 召回数 |
| `PathSettings` | 输入/输出/术语/缓存/向量库目录路径 |
| `HumanLoopSettings` | 各类人工确认场景开关、置信度阈值 |
| `Settings` | 以上所有配置的聚合根对象 |

通过 `get_settings()` 获取全局单例，通过 `update_settings()` 更新。

#### `config/prompts.py`

集中管理所有 Agent 使用的提示词模板（字符串常量），包括：

- `STYLE_ANALYSIS_PROMPT` — 文本风格与领域分析
- `TEXT_CHUNKING_PROMPT` — 智能文本切分引导
- `TERMINOLOGY_EXTRACTION_PROMPT` — 术语与命名实体提取
- `TERMINOLOGY_VERIFICATION_PROMPT` — 术语一致性验证
- `SLANG_LOOKUP_PROMPT` — 俚语与文化负载词查询
- `TRANSLATION_STEP1_UNDERSTANDING` — 翻译第一步：深度理解
- `TRANSLATION_STEP2_DRAFT` — 翻译第二步：初稿生成
- `TRANSLATION_STEP3_POLISH` — 翻译第三步：润色优化
- `MULTI_VERSION_PROMPT` — 多版本翻译生成
- `VERSION_FUSION_PROMPT` — 多版本融合
- `BACK_TRANSLATION_PROMPT` — 回译（译文→原语言）
- `TEAR_ESTIMATE_PROMPT` — TEaR 框架：误差估算
- `TEAR_REFINE_PROMPT` — TEaR 框架：修正优化
- `QUALITY_ASSESSMENT_PROMPT` — MQM 质量评估

---

### 2.3 智能体层：`src/agents/`

所有 Agent 继承自 `BaseAgent`，遵循统一接口。

#### `base_agent.py` — 抽象基类

提供所有 Agent 共享的基础能力：

- **LLM 初始化**：自动读取 `settings.llm` 配置，创建 `ChatOpenAI` 实例
- **`invoke_llm()`**：带指数退避重试（最多 3 次）的 LLM 调用方法，支持自动解析 JSON 响应
- **`_parse_json_response()`**：鲁棒 JSON 解析，依次尝试直接解析、代码块提取、正则提取
- **状态管理**：`AgentState`（idle / running / completed / error）+ 进度追踪
- **LLM 调用统计**：`self.llm_calls` 计数器，便于性能分析
- **抽象方法 `run()`**：强制子类实现核心逻辑

#### `preprocessor.py` — 预处理 Agent

**输入**：原始文本字符串 + 文档标题  
**输出**：`Document`（包含元数据 + 有序 `DocumentChunk` 列表）

处理流程：
1. 调用 LLM 分析文本前 3000 字符，识别领域、风格、正式程度、目标读者
2. 调用 LLM 进行智能文本切分（按段落/章节/逻辑块），保留上下文信息
3. 为每个 chunk 生成前后文摘要，标记段落类型（叙述/对话/描写/说明/议论）和翻译难度

#### `terminology.py` — 术语与实体 Agent

**输入**：文本段落 + 领域 + 上下文  
**输出**：更新后的 `TerminologyDatabase`

处理流程：
1. 命名实体识别（人名、地名、组织名等）
2. 领域术语提取与规范化（含翻译建议与置信度）
3. 文化负载词识别（俚语、习语、专有文化词），标注翻译策略
4. 全局一致性维护：相同原文始终对应相同译文

#### `translator.py` — 翻译 Agent

**输入**：源文本 + 术语表 + 上下文 + 风格要求  
**输出**：`TranslationUnit`（含多版本译文、最优译文、注释）

处理流程（多步骤引导）：
1. **理解（Step 1）**：深度理解原文含义、语气、文化背景
2. **初稿（Step 2）**：生成初版译文，遵循术语表
3. **润色（Step 3）**：优化语言流畅度、风格一致性
4. **多版本**（可选）：并行生成多种翻译风格（直译/意译/文学译）
5. **融合**（可选）：LLM 评估多版本并融合最优片段

#### `validator.py` — 验证 Agent

**输入**：`TranslationUnit`（含源文 + 当前最优译文）  
**输出**：验证报告（回译结果、质量评分、修正建议）

处理流程（TEaR 框架）：
1. **回译（Back Translation）**：将译文翻译回源语言
2. **误差估算（T-E）**：比较回译与原文的语义差异
3. **修正（aR）**：针对差异点重新优化译文
4. **MQM 质量评估**：从忠实度、流畅度、术语准确性、风格四个维度打分
5. 判断是否需要人工审核（低置信度时触发）

---

### 2.4 工作流层：`src/workflow/`

#### `orchestrator.py` — 顺序式工作流总指挥（主用）

**架构模式**：命令式顺序调用（Imperative Sequential）

```
输入文本
  │
  ▼ 阶段1: 预处理
  │   PreprocessorAgent.run() → Document（N 个 chunks）
  │
  ▼ 阶段2: 术语提取
  │   TerminologyAgent.run() × N chunks → TerminologyDatabase
  │   [可选] 人工审核低置信度术语
  │
  ▼ 阶段3: 逐块翻译
  │   TranslatorAgent.run() × N chunks → TranslationResult（N 个 units）
  │   每完成一个 chunk 自动保存检查点（pickle）
  │
  ▼ 阶段4: 验证
  │   ValidatorAgent.run() × N units → 质量报告 + 修正
  │   [可选] 低置信度翻译触发人工审核
  │
  ▼ 阶段5: 完成
      TranslationResult.merge_translations() → 完整译文
```

关键特性：
- **断点续传**：每个 chunk 翻译完成后自动 pickle 检查点到 `data/checkpoints/`，下次运行自动加载恢复
- **进度追踪**：`WorkflowState` 记录各阶段耗时和当前进度
- **LLM 调用统计**：运行结束后以 Rich 表格输出各 Agent 的 LLM 调用次数

#### `pipeline.py` — LangGraph 图状态机流水线（扩展备用）

**架构模式**：有向无环图（DAG）+ 状态机

```
PipelineStage 枚举定义各节点：
  INIT → PREPROCESS → EXTRACT_TERMINOLOGY
       → HUMAN_REVIEW_TERMINOLOGY（条件边）
       → TRANSLATE → VALIDATE
       → HUMAN_REVIEW_TRANSLATION（条件边）
       → FINALIZE → COMPLETE
```

与 Orchestrator 的主要区别见 [`docs/LANGGRAPH_DETAILED_EXPLANATION.md`](./LANGGRAPH_DETAILED_EXPLANATION.md)。

---

### 2.5 数据模型层：`src/models/`

所有模型基于 **Pydantic v2** `BaseModel`，保证类型安全与序列化能力。

#### `document.py`

| 类 | 说明 |
|----|------|
| `ParagraphType` | 段落类型枚举（叙述/对话/描写/说明/议论） |
| `DifficultyLevel` | 翻译难度枚举（easy/medium/hard） |
| `DocumentMetadata` | 文档元数据（标题、语言、领域、风格、目标读者） |
| `ChunkContext` | 切分上下文（前文摘要、后文摘要、前一 chunk ID） |
| `ChunkMetadata` | 切分元数据（段落类型、难度、是否包含对话/代码等） |
| `DocumentChunk` | 翻译单元（ID、内容、位置、上下文、元数据、翻译状态） |
| `Document` | 文档根对象（包含元数据 + 有序 chunk 列表） |

#### `terminology.py`

| 类 | 说明 |
|----|------|
| `EntityType` | 实体类型枚举（人名/地名/组织/作品/俚语/文化词/专业术语） |
| `TranslationStrategy` | 翻译策略枚举（音译/意译/直译/保留原文/注释/创译） |
| `NamedEntity` | 命名实体条目（原文、译文、类型、置信度、上下文） |
| `DomainTerm` | 领域术语条目（含领域、子领域、备选译法） |
| `CulturalTerm` | 文化负载词条目（文化背景说明、翻译策略、注释） |
| `TerminologyDatabase` | 术语库（统一存储上述三类条目，支持查询、更新、JSON 序列化） |

#### `translation.py`

| 类 | 说明 |
|----|------|
| `TranslationStatus` | 翻译状态枚举（pending/draft/refined/validated/approved/final） |
| `VersionType` | 版本类型枚举（literal/idiomatic/literary/technical/localized） |
| `TranslationVersion` | 单个翻译版本（译文内容、版本类型、置信度、生成策略） |
| `TranslationNote` | 翻译注释（术语决策、文化解释、不确定点） |
| `UncertainPart` | 不确定片段（原文、备选译法、不确定原因） |
| `BackTranslationResult` | 回译结果（回译文本、语义相似度、差异点） |
| `QualityAssessment` | 质量评估（MQM 各维度分数、总分、改进建议） |
| `TranslationUnit` | 翻译单元（聚合以上所有信息，含最优译文选择逻辑） |
| `TranslationResult` | 翻译结果根对象（包含全部单元，支持合并为完整译文） |

---

### 2.6 知识库层：`src/knowledge/`

#### `rag.py` — 检索增强生成

基于 LangChain + ChromaDB/FAISS 实现向量检索：

- **`DocumentStore`**：向量数据库封装，支持文档入库与语义相似度检索
- **`TranslationMemoryStore`**：翻译记忆专用存储，以"原文→译文"对的形式存储，支持相似原文召回
- **`RAGRetriever`**：统一检索接口，整合文档知识库与翻译记忆，为翻译 Agent 提供参考信息

#### `memory.py` — 翻译记忆管理

轻量级内存管理模块，不依赖向量数据库：

- **`MemoryType`**：记忆类型枚举（翻译对/摘要/风格指南/读者画像/术语/上下文/反馈）
- **`MemoryItem`**：记忆条目（内容、重要性、访问计数、时间戳）
- **`TranslationMemory`**：记忆管理器（按类型存储、按重要性检索、支持 JSON 持久化、自动清理低频记忆）

---

### 2.7 人机协作层：`src/human_loop/`

#### `interface.py`

基于 **[Rich](https://rich.readthedocs.io/)** 库提供终端交互界面：

- **`ReviewType`**：审核类型（术语/翻译/验证）
- **`ReviewPriority`**：优先级（low/normal/high/critical）
- **`ReviewTask`**：审核任务（原文、当前译文、备选译文、上下文、置信度）
- **`ReviewResult`**：审核结果（审核决定、最终译文、审核者注释）
- **`HumanLoopInterface`**：
  - 以富文本面板展示待审核内容
  - 支持"批准/修改/跳过/查看上下文"四类操作
  - 任务队列管理与批量处理
  - 运行结束后输出审核统计摘要
- **`create_human_review_callback()`**：工厂函数，创建可传入 Orchestrator 的回调函数

---

### 2.8 工具函数层：`src/utils/`

#### `text_utils.py`

| 函数/类 | 说明 |
|---------|------|
| `clean_text()` | 清理文本（统一换行符、移除不可见字符、压缩空行） |
| `count_words()` | 统计词数（英文按空格分词，中文按字符计） |
| `detect_language()` | 简单语言检测（中/英/其他） |
| `split_into_sentences()` | 分句工具 |
| `split_into_paragraphs()` | 分段工具 |
| `extract_dialogues()` | 提取对话片段 |
| `TextProcessor` | 文本处理器类（统计分析、潜在术语提取、文本摘要） |

#### `file_utils.py`

| 函数/类 | 说明 |
|---------|------|
| `load_text_file()` | 加载文本文件（支持 .txt / .md / .epub / .json） |
| `save_translation()` | 保存译文（支持 txt / json 格式，可附加元数据） |
| `FileHandler` | 文件处理器类（批量文件扫描、输入/输出目录管理） |

---

## 三、数据流与模块交互

```
┌─────────────────────────────────────────────────────────────┐
│  用户（CLI）                                                  │
│  python main.py translate input.txt -s en -t zh             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  main.py                                                     │
│  ① 解析 CLI 参数                                             │
│  ② 从 config/settings.py 读取/更新全局配置                   │
│  ③ 调用 file_utils.load_text_file() 加载文本                 │
│  ④ 实例化 TranslationOrchestrator                           │
│  ⑤ asyncio.run(_run_translation())                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  src/workflow/orchestrator.py                                │
│  TranslationOrchestrator.run()                              │
│                                                              │
│  [Stage 1] preprocessor.run(raw_text)                       │
│      └─► config/prompts.py (STYLE_ANALYSIS_PROMPT,          │
│               TEXT_CHUNKING_PROMPT)                          │
│      └─► LLM API                                            │
│      └─► → Document (chunks + metadata)                     │
│                                                              │
│  [Stage 2] terminology_agent.run(chunk.content) × N         │
│      └─► config/prompts.py (TERMINOLOGY_EXTRACTION_PROMPT)  │
│      └─► LLM API                                            │
│      └─► → TerminologyDatabase                              │
│      [可选] human_loop/interface.py 术语人工审核             │
│                                                              │
│  [Stage 3] translator.run(chunk, terminology) × N           │
│      └─► config/prompts.py (TRANSLATION_STEP1~3,           │
│               MULTI_VERSION_PROMPT, VERSION_FUSION_PROMPT)  │
│      └─► LLM API × 3~5次/chunk                              │
│      └─► → TranslationUnit (含多版本 + 最优译文)            │
│      └─► _save_checkpoint() → data/checkpoints/            │
│                                                              │
│  [Stage 4] validator.run(unit) × N                          │
│      └─► config/prompts.py (BACK_TRANSLATION_PROMPT,        │
│               TEAR_ESTIMATE_PROMPT, TEAR_REFINE_PROMPT,     │
│               QUALITY_ASSESSMENT_PROMPT)                     │
│      └─► LLM API                                            │
│      └─► 低置信度 → human_loop/interface.py 翻译人工审核    │
│                                                              │
│  [Stage 5] translation_result.merge_translations()          │
│      └─► → full_translation (完整译文字符串)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  main.py（输出）                                             │
│  ① save_translation() → data/output/{title}_translated.txt  │
│  ② orchestrator.save_terminology_database()                 │
│         → data/output/{title}_terminology.json              │
│  ③ 打印耗时统计与 LLM 调用次数                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、配置与环境

### 环境变量（`.env` 文件）

```env
# 必填
OPENAI_API_KEY=your_api_key_here

# 可选，默认值如下
OPENAI_BASE_URL=https://api.openai.com/v1   # 官方 OpenAI 端点；若使用第三方兼容 API
                                             # （如 DeepSeek：https://api.deepseek.com/v1），
                                             # 需将此值替换为对应服务商的 Base URL
MODEL_NAME=gpt-4o                            # 主模型
BACKUP_MODEL=gpt-4o-mini                     # 备用模型（用于验证阶段；第三方 API 不适用时
                                             # 系统会自动回退到主模型）
```

### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `translation.chunk_size` | 1500 | 每个翻译单元的最大字符数 |
| `translation.chunk_overlap` | 200 | chunk 间重叠字符数，保证上下文连贯 |
| `translation.generate_variants` | `True` | 是否生成多版本译文（影响 LLM 调用次数） |
| `translation.variant_count` | 3 | 多版本数量 |
| `translation.back_translation_threshold` | 0.85 | 回译相似度阈值，低于此值触发修正 |
| `human_loop.confidence_threshold` | 0.7 | 触发人工审核的置信度下限 |
| `rag.top_k` | 5 | 向量检索返回的最相似结果数 |

---

## 五、关键设计决策

### 5.1 两种工作流的选择

| 方面 | Orchestrator（当前主用） | Pipeline（LangGraph） |
|------|-------------------------|----------------------|
| 实现方式 | Python 顺序代码 | 有向无环图（DAG） |
| 状态持久化 | 手动 pickle | 自动 MemorySaver |
| 人工审核暂停/恢复 | 需重启流程 | 原生支持 |
| 代码复杂度 | 低，易读 | 高，灵活 |
| 推荐场景 | 批量自动化翻译 | 需要复杂条件路由的场景 |

### 5.2 多步骤翻译策略

翻译 Agent 不直接让 LLM "一步翻译"，而是拆分为 **理解 → 初稿 → 润色** 三步，每步提供专用的提示词，模拟专业译者的工作流程，显著提升长文本翻译质量。

### 5.3 TEaR 验证框架

**T**ranslation → **E**rror estimation → **a**nd **R**efinement：通过回译 + 误差分析的闭环机制，在无需人工介入的情况下自动发现并修正翻译错误。

### 5.4 断点续传机制

为应对长文本翻译中途失败（如 API 超时、网络中断）的场景，每翻译完成一个 chunk 就将完整状态（文档、术语库、翻译结果）序列化为 pickle 文件保存到 `data/checkpoints/`，下次运行同一文件时自动加载恢复，避免重复消耗 Token。

### 5.5 LLM 调用次数估算

对于一篇含 N 个 chunk 的文档（启用多版本 + 验证）：

```
预处理:  2 次（风格分析 1 次 + 智能切分 1 次；复杂文本可能略有增加）
术语:    N × 1 次
翻译:    N × (3步 + 3版本 + 1融合) ≈ N × 7 次
验证:    N × (1回译 + 1估算 + 1修正 + 1评估) = N × 4 次
────────────────────────────────────
合计:    约 2 + N × 12 次 LLM 调用
```

使用 `--fast` 模式可减少至约 `2 + N × 3` 次。
