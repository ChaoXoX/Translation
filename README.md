# 基于多智能体工作流的长文本领域书籍翻译智能代理

## 项目概述

本项目实现了一个基于 LangChain/LangGraph 的多智能体协作翻译系统，专门用于处理长文本书籍的高质量翻译任务。系统采用模块化设计，支持领域术语管理、回译验证、人机协作等高级功能。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Translation Orchestrator                        │
│                        (翻译总指挥官 Agent)                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  Preprocessor │          │  Terminology  │          │  Translation  │
│    Agent      │◄────────►│  & Entity     │◄────────►│  & Refinement │
│  (预处理Agent) │          │    Agent      │          │     Agent     │
└───────────────┘          │ (术语实体Agent)│          │ (翻译优化Agent)│
        │                  └───────────────┘          └───────────────┘
        │                          │                           │
        ▼                          ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│    Text       │          │  Terminology  │          │   Back-Trans  │
│  Chunking     │          │   Database    │          │  Validator    │
│  (文本切分)    │          │   (术语库)     │          │  (回译验证器)  │
└───────────────┘          └───────────────┘          └───────────────┘
        │                          │                           │
        └──────────────────────────┼───────────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Human-in-the-Loop         │
                    │   (人机协作节点)              │
                    └─────────────────────────────┘
```

## 核心功能

### 1. 译前预处理 (Preprocessor Agent)
- 文本结构化切分（按段落/章节/逻辑块）
- 语域与风格自动识别
- 上下文信息保留

### 2. 术语与实体管理 (Terminology & Entity Agent)
- 命名实体识别 (NER)
- 领域术语提取与规范化
- 文化负载词识别与翻译策略建议
- 全局一致性控制

### 3. 翻译与优化 (Translation & Refinement Agent)
- 多步骤引导翻译
- 多版本生成与融合
- 回译验证 (TEaR框架)
- 自我修正机制

### 4. 人机协作
- 关键节点人工介入
- 结构化任务输出
- 翻译决策确认

## 目录结构

```
书籍翻译智能代理/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config/
│   ├── __init__.py
│   ├── settings.py          # 全局配置
│   └── prompts.py           # 提示词模板
├── src/
│   ├── __init__.py
│   ├── models/              # 数据模型
│   │   ├── __init__.py
│   │   ├── document.py      # 文档模型
│   │   ├── terminology.py   # 术语模型
│   │   └── translation.py   # 翻译结果模型
│   ├── agents/              # 智能体
│   │   ├── __init__.py
│   │   ├── base_agent.py    # 基础Agent类
│   │   ├── preprocessor.py  # 预处理Agent
│   │   ├── terminology.py   # 术语实体Agent
│   │   ├── translator.py    # 翻译Agent
│   │   └── validator.py     # 验证Agent
│   ├── workflow/            # 工作流
│   │   ├── __init__.py
│   │   ├── orchestrator.py  # 总指挥官
│   │   └── pipeline.py      # 流水线
│   ├── knowledge/           # 知识库
│   │   ├── __init__.py
│   │   ├── rag.py           # RAG模块
│   │   └── memory.py        # 记忆模块
│   ├── human_loop/          # 人机协作
│   │   ├── __init__.py
│   │   └── interface.py     # 交互接口
│   └── utils/               # 工具函数
│       ├── __init__.py
│       ├── text_utils.py    # 文本处理
│       └── file_utils.py    # 文件处理
├── data/
│   ├── input/               # 输入文本
│   ├── output/              # 输出结果
│   ├── terminology/         # 术语库
│   └── cache/               # 缓存
├── examples/
│   ├── demo_translation.py  # 示例脚本
│   └── sample_text.txt      # 示例文本
├── tests/                   # 测试文件
│   ├── __init__.py
│   ├── test_agents.py
│   └── test_workflow.py
└── main.py                  # 主入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：
```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # 或其他兼容API
MODEL_NAME=gpt-4o
```

### 3. 运行翻译

```bash
python main.py --input data/input/your_book.txt --output data/output/ --source en --target zh
```

## 支持的翻译方向

- 英语 → 中文 (适合说唱/NBA自传等)
- 中文 → 英语
- 中文 → 越南语/泰语 (适合网络文学)
- 其他语言对（需配置对应术语库）

## 许可证

MIT License
