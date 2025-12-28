# LangChain vs LangGraph：详细对比与项目实现

## 一、概念澄清：它们的关系

```
┌─────────────────────────────────────────┐
│        LangChain 框架生态系统            │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────┐                   │
│  │  LangChain Core  │ ← 基础抽象        │
│  └────────┬─────────┘                   │
│           │                             │
│  ┌────────▼──────────┐  ┌──────────┐   │
│  │ LLM Abstraction   │  │ Schemas  │   │
│  │ (ChatOpenAI等)    │  │ (Message)│   │
│  └───────────────────┘  └──────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  LangChain Components            │   │
│  │  ├─ Chains (执行链)              │   │
│  │  ├─ Agents (自主Agent)           │   │
│  │  └─ Prompts (提示工程)           │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  🌟 LangGraph (新增)             │   │
│  │  ├─ StateGraph (状态机)          │   │
│  │  ├─ 节点 & 边 (流程控制)        │   │
│  │  └─ 检查点 (持久化)             │   │
│  └──────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

**关键认识**:
- `LangChain` = 是整个 AI 框架库
- `LangGraph` = LangChain 中的**流程编排子库**（专门管理工作流）
- 关系：LangGraph ⊂ LangChain

---

## 二、LangChain 详解：低层 Agent 框架

### 2.1 LangChain 的核心职责

**LangChain 解决的问题**:
```
原始 OpenAI API 太底层，需要包装：

❌ 直接调用 OpenAI API：
import openai
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "..."}]
)

✅ 用 LangChain 封装：
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("...")
```

### 2.2 LangChain 的四大核心模块

#### A. LLM 抽象层

```python
from langchain_openai import ChatOpenAI

# 一行代码即可支持 OpenAI、DeepSeek 等多家提供商
llm = ChatOpenAI(
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",  # ← 改这里就支持任何 API
    model="gpt-4o"
)

# 无论用哪个提供商，调用方式完全一样
response = llm.invoke("翻译这句话")
```

**优势**：
- 🔌 **API 无关** - 同一段代码支持多个 LLM 提供商
- 🔄 **易切换** - 从 GPT-4 切到 Claude 只需改 base_url
- 📊 **监控集成** - 内置 token 计数、成本追踪

---

#### B. 消息系统

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 结构化消息，而非原始字符串
messages = [
    SystemMessage("你是一个翻译专家"),
    HumanMessage("翻译：Hello"),
    AIMessage("你好")  # 对话历史
]

# LangChain 会自动格式化为 OpenAI 需要的格式
response = llm.invoke(messages)
```

**为什么需要**：
- 👥 **角色清晰** - 系统提示、用户输入、助手响应分离
- 💬 **上下文管理** - 保持对话历史（不丢失之前的回答）
- 🔀 **动态拼接** - 支持链式调用，前一个 Agent 的输出作为后一个的输入

---

#### C. Agents - 自主决策

```python
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain import hub

# 定义工具（能做什么）
tools = [
    Tool(name="查词典", func=lookup_word, description="查询词汇..."),
    Tool(name="回译验证", func=back_translate, description="验证翻译..."),
    Tool(name="查术语表", func=query_terminology, description="查找术语...")
]

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor.from_agent_and_tools(agent, tools)

# Agent 自主决策使用哪个工具
response = executor.invoke({
    "input": "翻译 'quantum computing' 这个术语，需要查一下相关技术文档"
})
# → Agent 会自动选择：先查术语表，再查词典，最后给出最佳翻译
```

**ReAct 框架** (Reasoning + Acting)：
```
思考 (Thought) → 决定用哪个工具
行动 (Action) → 执行工具
观察 (Observation) → 看工具结果
思考 → 是否继续或结束
```

---

#### D. 提示工程 (Prompts)

```python
from langchain_core.prompts import ChatPromptTemplate

# 结构化提示，支持动态变量
prompt_template = ChatPromptTemplate.from_template(
    """你是一个翻译专家，擅长 {domain} 领域翻译。
    
源文本: {source_text}
术语表: {terminology}

请翻译上述文本。"""
)

# 动态注入变量
messages = prompt_template.format_messages(
    domain="医学",
    source_text="The patient has hypertension",
    terminology="hypertension=高血压"
)

response = llm.invoke(messages)
```

---

### 2.3 LangChain 的使用方式

**项目中的使用** (src/agents/base_agent.py)：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class BaseAgent:
    def __init__(self):
        # LangChain 的 LLM 抽象
        self.llm = ChatOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model_name,
            temperature=0.3
        )
    
    async def invoke_llm(self, prompt, system_prompt=None):
        # 使用 LangChain 的消息系统
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        # 调用 LLM（支持异步）
        response = await self.llm.ainvoke(messages)
        return response.content
```

**LangChain 的优势**:
- ✅ **API 统一** - OpenAI、DeepSeek、Anthropic 一套代码
- ✅ **消息管理** - 自动处理对话历史
- ✅ **工具集成** - 方便添加外部工具（知识库、计算器等）
- ✅ **成熟生态** - 社区丰富，集成广泛

---

## 三、LangGraph 详解：高层工作流框架

### 3.1 LangGraph 的核心职责

**LangGraph 解决的问题**:
```
多个 Agent 协作时，谁负责编排？

❌ 不用 LangGraph 的做法：
# main.py - 手动硬编码流程
orchestrator = TranslationOrchestrator()
result1 = await orchestrator.preprocess(text)
result2 = await orchestrator.extract_terminology(result1)
result3 = await orchestrator.translate(result2)
...
# 问题：流程写死了，无法暂停-恢复，无法条件分支

✅ 用 LangGraph 的做法：
workflow = StateGraph(PipelineState)
workflow.add_node("preprocess", preprocess_func)
workflow.add_node("translate", translate_func)
workflow.add_conditional_edges("translate", should_validate, {...})
graph = workflow.compile()
# 问题自动解决：支持任意分支、自动检查点、暂停-恢复
```

### 3.2 LangGraph 的三大核心概念

#### 概念 1：State - 全局状态

```python
from typing import TypedDict, Annotated
import operator

class PipelineState(TypedDict):
    """全局流水线状态（所有节点共享）"""
    # 输入
    input_text: str
    
    # 中间结果
    document: Optional[Document]
    terminology_db: Optional[TerminologyDatabase]
    translation_result: Optional[TranslationResult]
    
    # 人工审核
    pending_reviews: Annotated[List[Dict], operator.add]  # 累积列表
    review_responses: Dict[str, Any]
    
    # 配置
    enable_validation: bool
    
    # 输出
    final_translation: str
```

**状态的作用**:
- 📦 **数据流动** - 从一个节点传到下一个节点
- 🔄 **共享上下文** - 所有节点可访问全局状态
- 💾 **持久化** - 框架自动保存状态到数据库（暂停-恢复）

**State vs 函数参数的对比**:
```python
# ❌ 传统做法（函数参数链）
result1 = preprocess(text)
result2 = terminology(result1)  # 必须接收上一个的输出
result3 = translate(result2)    # 层层依赖，无法跳过

# ✅ LangGraph（全局 State）
messages = [
    SystemMessage(...),
    HumanMessage(...)
]
response = llm.ainvoke(messages)
state["result1"] = response      # 直接读写共享状态
```

---

#### 概念 2：Node - 执行单元

```python
async def node_preprocess(state: PipelineState) -> PipelineState:
    """节点函数的标准形式"""
    # 输入：接收全局状态
    input_text = state["input_text"]
    
    # 处理：执行业务逻辑
    document = await preprocessor.run(input_text)
    
    # 输出：修改并返回状态
    state["document"] = document
    state["stage"] = "preprocess_done"
    
    return state  # 更新的状态会自动流向下一个节点
```

**节点特性**:
- 📥 **输入** = State（上游节点修改的状态）
- 🔧 **处理** = 业务逻辑（调用 Agent、LLM 等）
- 📤 **输出** = 修改后的 State
- 🔗 **链接** = 框架自动传递 State 给下一节点

---

#### 概念 3：Edge - 节点连接规则

##### 3.3a 直接边（顺序连接）

```python
workflow.add_edge("preprocess", "terminology")
# 流程：preprocess 完成 → 立即进入 terminology
```

##### 3.3b 条件边（智能分支）

```python
def should_validate(state: PipelineState) -> str:
    """决策函数：根据状态决定下一步"""
    if state["enable_validation"]:
        return "validate"
    else:
        return "skip_validation"

workflow.add_conditional_edges(
    "translate",           # 源节点
    should_validate,       # 决策函数
    {
        "validate": "validate",           # 返回值 → 目标节点
        "skip_validation": "finalize"
    }
)
```

**流程**:
```
[translate 节点完成]
        ↓
  调用 should_validate(state)
        ↓
  根据 enable_validation 返回值
        ↓
    ┌─────────────────┐
    │                 │
 validate          finalize
```

---

### 3.3 LangGraph 的工作原理

```python
# 第 1 步：定义
workflow = StateGraph(PipelineState)

# 第 2 步：添加所有节点和边
workflow.add_node("preprocess", node_preprocess)
workflow.add_node("translate", node_translate)
workflow.add_edge("preprocess", "translate")

# 第 3 步：编译（变成可执行的图）
graph = workflow.compile()

# 第 4 步：执行（自动处理状态流动）
async for state in graph.astream(
    {"input_text": "..."},
    config={"configurable": {"thread_id": "doc_123"}}
):
    print(f"Current stage: {state['stage']}")
    # LangGraph 自动处理：
    # 1. 状态序列化
    # 2. 调用对应节点函数
    # 3. 接收新状态
    # 4. 检查是否有条件边
    # 5. 路由到下一节点
    # 6. 保存检查点（thread_id 隔离）
```

---

## 四、详细对比表

| 维度 | LangChain | LangGraph |
|------|---------|-----------|
| **层级** | 低层（单个 LLM 调用） | 高层（多个 Agent 协作） |
| **职责** | LLM 接口、消息管理 | 工作流编排、状态管理 |
| **API 范例** | `ChatOpenAI.invoke()` | `StateGraph().compile()` |
| **流程控制** | 否（只是调用 LLM） | 是（节点、边、条件分支） |
| **状态管理** | 无（每次调用独立） | 有（全局 State） |
| **暂停-恢复** | ❌ 不支持 | ✅ 支持（via thread_id） |
| **持久化** | 无 | ✅ 自动（MemorySaver/PostgresSaver） |
| **用途** | 调用 LLM、构建提示 | 编排多个 Agent 的工作流 |

---

## 五、项目的两层架构

### 5.1 第 1 层：LangChain（Agent 内部）

**文件**: `src/agents/base_agent.py`

```python
# 每个 Agent 内部使用 LangChain
class BaseAgent:
    def __init__(self):
        # LangChain 抽象 - 支持多个 LLM 提供商
        self.llm = ChatOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,  # OpenAI / DeepSeek / 其他
            model=settings.llm.model_name
        )
    
    async def invoke_llm(self, prompt, system_prompt=None):
        # LangChain 消息系统
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # 调用 LLM（LangChain 处理 API 差异）
        response = await self.llm.ainvoke(messages)
        return response.content
```

**LangChain 在这里的作用**:
- ✅ 屏蔽 API 细节（一套代码支持 OpenAI/DeepSeek）
- ✅ 处理消息序列化（SystemMessage → {"role": "system", ...}）
- ✅ 支持异步调用（`ainvoke` vs `invoke`）
- ✅ 自动重试、成本追踪

---

### 5.2 第 2a 层：Orchestrator（顺序工作流）

**文件**: `src/workflow/orchestrator.py`（当前使用）

```python
# 不使用 LangGraph，手动编排 Agent
class TranslationOrchestrator:
    async def run(self, input_text):
        # 第 1 阶段
        self.document = await self.preprocessor.run(input_text)
        
        # 第 2 阶段
        await self._extract_terminology()
        
        # 第 3 阶段
        if not checkpoint_loaded:
            await self._translate_all_chunks()
        
        # 第 4 阶段
        if enable_validation:
            await self._validate_translations()
        
        return self.translation_result
```

**特点**:
- 📍 **命令式** - 明确的顺序执行
- 💾 **手动检查点** - `_save_checkpoint()` / `_load_checkpoint()`
- ⚡ **简单直观** - 易于理解和维护
- ❌ **无暂停机制** - 人工审核后必须重新运行

**流程图**:
```
启动
  ↓
[预处理] → checkpoint
  ↓
[术语提取] → checkpoint
  ↓
[翻译所有块] → checkpoint（每个块保存一次）
  ↓
[验证] → checkpoint
  ↓
完成
```

---

### 5.2 第 2b 层：Pipeline（LangGraph 工作流）

**文件**: `src/workflow/pipeline.py`（已实现但未启用）

```python
from langgraph.graph import StateGraph, END

class TranslationPipeline:
    def _build_graph(self) -> StateGraph:
        # 声明式定义工作流
        workflow = StateGraph(PipelineState)
        
        # 添加节点（每个节点是一个 async 函数）
        workflow.add_node("preprocess", self._node_preprocess)
        workflow.add_node("extract_terminology", self._node_extract_terminology)
        workflow.add_node("translate", self._node_translate)
        workflow.add_node("validate", self._node_validate)
        
        # 添加边（连接节点）
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "extract_terminology")
        
        # 条件边（根据状态分支）
        workflow.add_conditional_edges(
            "translate",
            self._should_validate,  # 决策函数
            {
                "validate": "validate",
                "skip": "finalize"
            }
        )
        
        workflow.add_edge("validate", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def run(self, input_text):
        config = {"configurable": {"thread_id": "doc_123"}}
        
        # LangGraph 自动处理：状态流动、检查点、条件分支
        async for state in self.graph.astream(
            {"input_text": input_text},
            config=config
        ):
            pass
        
        return state["translation_result"]
```

**特点**:
- 🕸️ **声明式** - 定义 DAG 结构，框架处理执行
- 💾 **自动检查点** - LangGraph 自动保存
- ⏸️ **暂停-恢复** - 支持中断并从断点继续
- 🔀 **灵活分支** - 支持复杂的条件路由
- 👁️ **可观测** - 内置追踪、可视化

**流程图**:
```
[preprocess]
     ↓
[extract_terminology]
     ↓
   (条件判断：需要人工审核？)
     ↓
  ┌─ YES → [human_review] → (暂停，等待用户输入) → (恢复)
  │
  └─ NO  → (继续)
     ↓
[translate]
     ↓
   (条件判断：需要验证？)
     ↓
  ┌─ YES → [validate] → (条件判断：需要人工审核？)
  │                      ├─ YES → [human_review] → (暂停，等待用户输入)
  │                      └─ NO  → [finalize]
  │
  └─ NO  → [finalize]
     ↓
   END
```

---

## 六、为什么项目目前用 Orchestrator，而不是 LangGraph？

### 6.1 当前决策

```python
# main.py
from src.workflow.orchestrator import TranslationOrchestrator  # ← 这个

# 不是
from src.workflow.pipeline import TranslationPipeline
```

### 6.2 原因分析

| 考虑因素 | Orchestrator | LangGraph |
|---------|------------|----------|
| **开发时间** | ⚡ 快 | 🐢 较慢 |
| **代码复杂度** | ⭐ 低 | ⭐⭐⭐ 中 |
| **学习曲线** | 📚 低 | 📚📚 高 |
| **需求匹配** | ✅ 完全满足 | ⚠️ 过度设计 |

### 6.3 项目的实际需求

```
当前项目的工作流特点：
✅ 完全线性（无复杂分支）
✅ 无需暂停-恢复（用户一次性提交文本，等待最终结果）
✅ 单文档处理（不需要并发）
✅ 简单的有条件执行（enable_validation 开关）
```

**因此**：
```
需求复杂度 < LangGraph 提供的功能
       ↓
使用更简单的 Orchestrator 反而更合适
       ↓
如果未来有新需求，再迁移也不迟
```

---

## 七、三层关系总结

```
┌─────────────────────────────────────────────────────────┐
│ 应用层 (Application)                                     │
│ main.py - CLI 命令入口                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ 编排层 (Orchestration) ← 二选一                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ TranslationOrchestrator (当前使用)                  │
│     └─ 手动顺序执行、手动检查点                        │
│                                                         │
│  ⚪ TranslationPipeline (预留、未使用)                  │
│     └─ 基于 LangGraph 的声明式编排                      │
│                                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Agent 层 (Agents)                                        │
├─────────────────────────────────────────────────────────┤
│  BaseAgent                                              │
│  ├─ PreprocessorAgent                                   │
│  ├─ TerminologyAgent                                    │
│  ├─ TranslatorAgent                                     │
│  └─ ValidatorAgent                                      │
│                                                         │
│  每个 Agent 内部使用 LangChain 调用 LLM                │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ LLM 层 (LangChain Core)                                  │
├─────────────────────────────────────────────────────────┤
│  ChatOpenAI(base_url="...") ← 支持 OpenAI/DeepSeek    │
│  HumanMessage / SystemMessage / AIMessage              │
│  异步调用：await llm.ainvoke(messages)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 八、数据流动示例

### 8.1 LangChain 层（Agent 内部）

```python
# TranslatorAgent 翻译流程

# 第 1 步：构造提示
source_text = "The future is bright"
system_prompt = "You are a professional translator"
prompt = f"Translate: {source_text}"

# 第 2 步：构造消息（LangChain 消息系统）
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=prompt)
]

# 第 3 步：调用 LLM（LangChain 处理）
response = await self.llm.ainvoke(messages)
# LangChain 内部工作：
# 1. 验证 base_url 是否有效
# 2. 添加认证头 (api_key)
# 3. 序列化消息为 OpenAI 格式
# 4. 发送 HTTP 请求
# 5. 接收响应
# 6. 反序列化为 AIMessage

translation = response.content  # "未来很光明"
```

### 8.2 Orchestrator 层（工作流编排）

```python
# TranslationOrchestrator 的工作流

async def run(self, input_text):
    # 第 1 层：LangChain（PreprocessorAgent 内部）
    document = await self.preprocessor.run(input_text)
    # PreprocessorAgent 使用 LangChain 调用 LLM
    # LangChain 处理 API 细节
    
    # 第 2 层：Orchestrator（状态管理）
    self.state.stage = "preprocessing"
    self._save_checkpoint(document)  # 手动保存状态
    
    # 第 3 层：LangChain（TerminologyAgent 内部）
    await self.terminology_agent.run(document.chunks[0].content)
    
    # 第 2 层：Orchestrator（继续编排）
    self.state.stage = "terminology"
    self._save_checkpoint(...)
    
    # 继续链式调用...
```

**数据流**:
```
用户输入
  ↓
[Orchestrator.run()] ─ 编排层
  ├→ [Preprocessor.run()] ─ Agent 层
  │   └→ [ChatOpenAI.ainvoke()] ─ LangChain 层
  │       └→ OpenAI API
  │
  ├→ [TerminologyAgent.run()] ─ Agent 层
  │   └→ [ChatOpenAI.ainvoke()] ─ LangChain 层
  │
  ├→ [TranslatorAgent.run()] ─ Agent 层
  │   └→ [ChatOpenAI.ainvoke()] ─ LangChain 层 ✕ 5 步（理解→草稿→打磨→多版本→融合）
  │
  └→ [ValidatorAgent.run()] ─ Agent 层
      └→ [ChatOpenAI.ainvoke()] ─ LangChain 层 ✕ 3 步（回译→评估→修正）
  
  ↓
最终译文
```

---

## 九、如何阅读项目代码

### 理解流程的两条路线

#### 路线 A：从上往下（用户 → 结果）

```
1. main.py
   ↓
2. TranslationOrchestrator.run()  ← 编排层
   ├─ preprocessor.run()  ← Agent 层
   │   └─ invoke_llm()    ← LangChain 层
   ├─ terminology_agent.run()
   │   └─ invoke_llm()
   ├─ translator.run()
   │   └─ invoke_llm() ✕ 5 次
   └─ validator.run()
       └─ invoke_llm() ✕ 3 次
```

**最重要的文件顺序**:
1. `main.py` - 入口
2. `src/workflow/orchestrator.py` - 工作流编排
3. `src/agents/base_agent.py` - Agent 基类（LangChain 在这里）
4. `config/prompts.py` - 所有提示模板

---

#### 路线 B：从下往上（LLM → 用户）

```
1. LLM 调用
   ↑
2. invoke_llm() (base_agent.py) ← LangChain 层
   ↑
3. Agent.run() (translator.py 等) ← Agent 层
   ↑
4. Orchestrator.run() (orchestrator.py) ← 编排层
   ↑
5. @app.command() translate (main.py) ← 应用层
```

---

## 十、关键代码位置

### LangChain 使用

```
❌ 不要找 LangChain 怎么用
✅ 都在这里：

src/agents/base_agent.py
  ├─ Line 12: from langchain_openai import ChatOpenAI
  ├─ Line 13: from langchain_core.messages import ...
  ├─ Line 45: self.llm = ChatOpenAI(...)
  └─ Line 69-105: invoke_llm() 方法（完整例子）

src/agents/validator.py
  └─ Line 32: ChatOpenAI 的动态模型选择

src/knowledge/rag.py
  ├─ Line 17: OpenAIEmbeddings（如果启用 RAG）
  ├─ Line 18: Chroma / FAISS（向量存储）
  └─ Line 19: RecursiveCharacterTextSplitter（分块）
```

### 编排层使用

```
❌ 不要找编排怎么写
✅ 看这两个对比：

🟢 当前使用（Orchestrator）：
src/workflow/orchestrator.py
  ├─ run() ← 主工作流
  ├─ _extract_terminology()
  ├─ _translate_all_chunks()
  ├─ _validate_translations()
  ├─ _save_checkpoint() ← 手动保存
  └─ _load_checkpoint() ← 手动恢复

⚪ 预留使用（LangGraph）：
src/workflow/pipeline.py
  ├─ _build_graph() ← 定义 DAG
  ├─ _node_preprocess() ← 节点函数
  ├─ _should_validate() ← 决策函数
  └─ resume() ← 自动恢复
```

---

## 总结：一句话理解

| 概念 | 简单解释 |
|-----|--------|
| **LangChain** | 🔌 LLM 插座 - 屏蔽不同 LLM 提供商的 API 差异 |
| **LangGraph** | 🕸️ 流程编排器 - 多个 Agent 协作时的工作流管理 |
| **Orchestrator** | 📋 当前编排方案 - 顺序执行，手动检查点 |
| **项目关系** | Agent (LangChain) → Orchestrator (当前) → 未来迁移到 LangGraph |

