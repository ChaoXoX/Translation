# LangGraph 详细解释：架构对比与项目实现

## 一、核心概念：LangGraph vs Orchestrator

### 1.1 当前项目架构 - Orchestrator（顺序流）

**位置**: `src/workflow/orchestrator.py`

**特点**: 
- 🔄 **顺序执行** - 线性流程：预处理 → 术语 → 翻译 → 验证 → 完成
- 📍 **命令式编程** - 显式调用每个阶段的函数
- 💾 **手动检查点** - `_save_checkpoint()` / `_load_checkpoint()` 由代码管理
- ⚡ **简单直观** - 易读易维护，适合确定流程

```python
# 顺序调用模式
async def run(self, input_text):
    # 第1步
    self.document = await self.preprocessor.run(input_text)
    
    # 第2步
    await self._extract_terminology()
    
    # 第3步（顺序执行，不可跳过）
    await self._translate_all_chunks()
    
    # 第4步
    if enable_validation:
        await self._validate_translations()
    
    # 第5步
    return self.translation_result
```

**缺点**:
- ❌ 人工审核后无法"暂停-恢复" - 需要重新运行整个流程
- ❌ 条件分支复杂 - `if enable_validation` 这样的硬编码
- ❌ 状态管理分散 - WorkflowState 是手动维护的
- ❌ 无法自动持久化状态 - 需要手写 pickle 逻辑

---

### 1.2 设计方案 - LangGraph（状态机 + 图论）

**位置**: `src/workflow/pipeline.py`

**特点**:
- 🕸️ **有向无环图 (DAG)** - 节点 (Node) + 边 (Edge) 定义流程
- 🎯 **条件路由** - 根据状态自动判断下一步（如人工审核跳过/执行）
- 💾 **自动持久化** - `MemorySaver` 或 `PostgresSaver` 自动保存状态
- ⏸️ **暂停-恢复** - 支持在人工审核处暂停，恢复时从断点继续
- 🔗 **声明式编程** - 定义 DAG 结构，框架处理执行逻辑

```python
# 声明式流程 - 使用 StateGraph
workflow = StateGraph(PipelineState)

# 添加节点（每个节点是一个异步函数）
workflow.add_node("preprocess", self._node_preprocess)
workflow.add_node("extract_terminology", self._node_extract_terminology)
workflow.add_node("translate", self._node_translate)
workflow.add_node("validate", self._node_validate)

# 添加边（节点连接）
workflow.add_edge("preprocess", "extract_terminology")

# 条件边 - 根据状态决定下一个节点
workflow.add_conditional_edges(
    "translate",
    self._should_validate,          # 决策函数
    {
        "validate": "validate",      # 如果返回 "validate"
        "skip": "finalize"           # 否则跳到 "finalize"
    }
)

workflow.set_entry_point("preprocess")
workflow.add_edge("finalize", END)

graph = workflow.compile()  # 编译成可执行的图
```

---

## 二、架构对比详表

| 维度 | Orchestrator | LangGraph |
|------|-------------|-----------|
| **执行模式** | 顺序 + 条件 | DAG + 状态机 |
| **流程定义** | 函数调用链 | 节点 + 边的图 |
| **状态管理** | 手动（WorkflowState） | 自动（StateGraph） |
| **暂停-恢复** | ❌ 不支持 | ✅ 原生支持 |
| **持久化** | 手写 pickle | 自动（MemorySaver/PostgresSaver） |
| **条件分支** | if/else 语句 | `add_conditional_edges()` |
| **可观测性** | logger | 内置 trace、可视化 |
| **复杂度** | 低 | 中 |
| **适用场景** | 简单线性流程 | 复杂工作流、需要暂停恢复 |

---

## 三、LangGraph 工作原理（逐步图解）

### 3.1 State - 全局共享状态

```python
class PipelineState(TypedDict):
    """流水线状态"""
    # 输入
    input_text: str
    title: str
    
    # 配置
    enable_validation: bool
    enable_human_review: bool
    
    # 中间状态
    stage: str
    document: Optional[Document]
    terminology_db: Optional[TerminologyDatabase]
    translation_result: Optional[TranslationResult]
    current_chunk_index: int
    
    # 人工审核
    pending_reviews: List[Dict[str, Any]]
    review_responses: Dict[str, Any]
    
    # 输出
    final_translation: str
    errors: List[str]
    
    # 时间追踪
    stage_times: Dict[str, float]
```

**关键点**:
- 📦 所有节点共享同一个 `State` 对象
- 🔄 每个节点**读取**上游节点的输出（从 State 中）
- ✍️ 每个节点**修改** State 并返回更新后的 State
- 🔗 LangGraph 自动传递状态给下一个节点

---

### 3.2 Node - 执行单位

```python
async def _node_preprocess(self, state: PipelineState) -> PipelineState:
    """预处理节点"""
    logger.info("Pipeline: Preprocessing")
    start_time = datetime.now()
    
    try:
        # 执行预处理
        document = await self.preprocessor.run(state["input_text"], state["title"])
        
        # 更新状态
        state["document"] = document
        state["stage"] = PipelineStage.PREPROCESS.value
        state["stage_times"]["preprocess"] = (datetime.now() - start_time).total_seconds()
        
    except Exception as e:
        state["errors"].append(f"Preprocess error: {str(e)}")
    
    return state  # 返回更新后的状态
```

**节点特性**:
- 📥 **输入**: 上一个节点传来的 State
- 🔧 **处理**: 执行业务逻辑（如调用 Agent）
- 📤 **输出**: 修改后的 State
- ⏱️ **异步**: 所有节点都是 `async def`

---

### 3.3 Edge - 节点连接

#### 3.3a 无条件边（直连）

```python
workflow.add_edge("preprocess", "extract_terminology")
```
→ 预处理完成后，直接进入术语提取

#### 3.3b 条件边（智能分支）

```python
workflow.add_conditional_edges(
    "translate",                        # 源节点
    self._should_validate,              # 决策函数
    {
        "validate": "validate",         # 条件 A：进行验证
        "skip": "finalize"              # 条件 B：跳过验证
    }
)
```

**决策函数**:
```python
def _should_validate(self, state: PipelineState) -> str:
    """根据状态决定是否验证"""
    if state["enable_validation"]:
        return "validate"
    else:
        return "skip"
```

**流程**:
1. 翻译节点完成，返回更新的 State
2. LangGraph 调用 `_should_validate(state)` 函数
3. 根据返回值 `"validate"` 或 `"skip"`，路由到不同节点
4. 下个节点继承了当前 State

---

### 3.4 入口点与终点

```python
workflow.set_entry_point("preprocess")  # 图从这里开始
workflow.add_edge("finalize", END)      # 这里结束
```

---

### 3.5 执行流程（可视化）

```
启动
  ↓
[ preprocess ]  ← State: {input_text, title}
  ↓ (直接边)
[ extract_terminology ]  ← State: {document, terminology_db}
  ↓ (条件边 - 检查 enable_human_review)
  ├─ YES → [ human_review_terminology ] → [ translate ]
  └─ NO  → 直接 [ translate ]
              ↓ (条件边 - 检查 enable_validation)
              ├─ YES → [ validate ] → (条件边 - 检查 pending_reviews)
              │         ├─ YES → [ human_review_translation ] → [ finalize ]
              │         └─ NO  → [ finalize ]
              └─ NO  → [ finalize ]
                        ↓
                        END
```

---

## 四、暂停-恢复机制（核心优势）

### 4.1 Orchestrator 中的问题

当用户在"术语人工审核"阶段需要用户输入时：
```python
# 无法暂停！只能这样做：
if enable_human_review:
    await self._handle_terminology_review()  # 阻塞等待用户输入
    # ← 用户需要等待整个函数完成
```

**问题**: 
- 用户提交审核结果后，必须从头运行整个流程
- 如果翻译需要 30 分钟，用户要重新等 30 分钟

---

### 4.2 LangGraph 中的暂停-恢复

```python
# 运行流水线
config = {"configurable": {"thread_id": "doc_123"}}

async for state in graph.astream(initial_state, config=config):
    # 如果到达人工审核节点，流程自动暂停
    if state["stage"] == PipelineStage.HUMAN_REVIEW_TERMINOLOGY.value:
        logger.info(f"Waiting for human review... (thread_id: {thread_id})")
        break  # ← 暂停，等待用户输入
```

**用户提交审核结果后**:
```python
# 恢复流水线
async def resume(thread_id: str, review_responses: Dict):
    config = {"configurable": {"thread_id": thread_id}}
    
    # 获取上次的状态
    current_state = graph.get_state(config)
    
    # 更新审核响应
    current_state["review_responses"] = review_responses
    
    # 从暂停点继续！
    async for state in graph.astream(current_state, config=config):
        pass
    
    return state["translation_result"]
```

**流程**:
1. **第 1 次运行**: preprocess → terminology → **暂停在人工审核**
2. **用户提交审核结果**
3. **第 2 次恢复**: 从人工审核后继续 → 直接进入翻译（跳过了预处理和术语提取！）
4. **结果**: 总耗时 ↓ 50%

---

## 五、项目中的应用场景

### 场景 A：简单线性翻译（✅ Orchestrator 足够）

```
输入 → [预处理] → [术语] → [翻译] → [验证] → 输出
```

**为什么 Orchestrator 够用**:
- 流程完全线性，无分支
- 无需暂停-恢复
- 用户在命令行等待最终结果

**当前实现**: ✅ 生产就绪

---

### 场景 B：带人工审核的复杂流程（❌ Orchestrator 有缺陷，✅ LangGraph 更优）

```
输入 → [预处理] → [术语] → 【人工审核】 ← ← ← ← 用户提交
            ↓
        [翻译] → [验证] → 【人工审核】 ← ← ← ← 用户提交
                     ↓
                  [输出]
```

**Orchestrator 的问题**:
```python
# 第 1 次运行：所有阶段串行执行
await self.run(input_text)  # 花费 1 小时

# 用户在中途提交审核反馈，但...
# 😞 代码会从头开始，预处理和前面的术语提取再做一遍！
# 实际浪费了 30 分钟
```

**LangGraph 的优势**:
```python
# 第 1 次运行：自动暂停在人工审核
async for state in graph.astream(...)  # 花费 20 分钟

# 用户提交审核反馈
await resume(thread_id, review_responses)  # 花费 40 分钟（跳过了预处理）

# 总耗时 = 60 分钟（而不是 90 分钟）
```

---

### 场景 C：多文档并行翻译（🚀 LangGraph 原生支持）

```python
# 同时运行 100 个翻译任务
tasks = []
for doc_id in doc_ids:
    # thread_id 隔离状态，互不干扰
    config = {"configurable": {"thread_id": doc_id}}
    task = graph.astream(initial_state, config=config)
    tasks.append(task)

# LangGraph 会用不同的 thread 存储各自的状态
results = await asyncio.gather(*tasks)
```

**Orchestrator 中要做同样的事**:
```python
# 手动创建 N 个 TranslationOrchestrator 实例
# 手动管理 N 个 checkpoint 目录
# 手动处理状态同步...
# ❌ 太复杂了！
```

---

## 六、为什么项目目前用 Orchestrator？

### 6.1 权衡分析

| 因素 | Orchestrator | LangGraph |
|-----|-------------|-----------|
| **上线时间** | ⚡ 快（1 小时）| 🐢 慢（2-3 小时） |
| **可维护性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **功能完整性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **学习成本** | 📚 低 | 📚📚📚 中等 |
| **生产就绪** | ✅ 是 | ✅ 是（v0.1+） |

### 6.2 项目决策

```python
# pipeline.py 写好了，但没在 CLI 中使用
# src/workflow/orchestrator.py ← main.py 实际使用的是这个

# 原因：
# 1. Orchestrator 足以满足当前需求（线性流程）
# 2. LangGraph 在以下情况才必需：
#    - 需要在任意节点暂停-恢复
#    - 需要支持多个并行的分支路径
#    - 需要自动化的人工审核流程
#    - 需要生产级的 Web UI（如 LangSmith）
```

---

## 七、如何迁移到 LangGraph？

### 7.1 4 步迁移方案

#### 步骤 1：启用 pipeline.py

```python
# main.py
from src.workflow.pipeline import TranslationPipeline  # 改成这个
from src.workflow.orchestrator import TranslationOrchestrator  # 不用这个

async def translate(input_path: str, ...):
    pipeline = TranslationPipeline()
    
    result = await pipeline.run(
        input_text=text,
        enable_human_review=True,  # 启用人工审核
        thread_id=f"doc_{datetime.now().timestamp()}"
    )
```

#### 步骤 2：添加 Web 服务处理暂停-恢复

```python
# web_api.py (新文件)
from fastapi import FastAPI
from src.workflow.pipeline import TranslationPipeline

app = FastAPI()
pipeline = TranslationPipeline()

@app.post("/translate/start")
async def start_translation(text: str):
    """开始翻译"""
    thread_id = generate_thread_id()
    
    # 在后台运行，自动暂停在人工审核
    asyncio.create_task(
        run_pipeline_background(thread_id, text)
    )
    
    return {"thread_id": thread_id}

@app.post("/translate/{thread_id}/review")
async def submit_review(thread_id: str, review_responses: dict):
    """提交人工审核结果"""
    result = await pipeline.resume(thread_id, review_responses)
    return result
```

#### 步骤 3：使用线程隔离状态

```python
# 每个文档有独立的 thread_id
threads = {
    "doc_001": "preprocessing",
    "doc_002": "human_review_terminology",
    "doc_003": "translating"
}

# 互不干扰，可以同时处理 100 个文档
```

#### 步骤 4：持久化检查点

```python
# 使用 PostgresSaver（而非 MemorySaver）
from langgraph.checkpoint.postgres import PostgresSaver

pipeline = TranslationPipeline(
    checkpointer=PostgresSaver(
        conn_string="postgresql://user:pass@localhost/langgraph"
    )
)

# 即使服务重启，也能从断点恢复！
```

---

## 八、代码示例：完整的 LangGraph 使用

### 8.1 最小化例子

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SimpleState(TypedDict):
    text: str
    processed: bool

# 定义节点
async def process_node(state: SimpleState):
    state["text"] = state["text"].upper()
    state["processed"] = True
    return state

# 构建图
workflow = StateGraph(SimpleState)
workflow.add_node("process", process_node)
workflow.set_entry_point("process")
workflow.add_edge("process", END)

graph = workflow.compile()

# 执行
async def main():
    result = await graph.ainvoke({"text": "hello", "processed": False})
    print(result)  # {"text": "HELLO", "processed": True}

asyncio.run(main())
```

### 8.2 带条件分支的例子

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class TextState(TypedDict):
    text: str
    length: int

async def analyze_node(state: TextState):
    state["length"] = len(state["text"])
    return state

async def long_handler(state: TextState):
    state["text"] = state["text"][:50] + "..."
    return state

async def short_handler(state: TextState):
    state["text"] = state["text"].upper()
    return state

def route_handler(state: TextState) -> str:
    """决定下一步"""
    return "long" if state["length"] > 100 else "short"

# 构建图
workflow = StateGraph(TextState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("long", long_handler)
workflow.add_node("short", short_handler)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges(
    "analyze",
    route_handler,
    {
        "long": "long",
        "short": "short"
    }
)
workflow.add_edge("long", END)
workflow.add_edge("short", END)

graph = workflow.compile()
```

### 8.3 带暂停-恢复的例子

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class WorkflowState(TypedDict):
    stage: str
    user_input: str = ""
    result: str = ""

async def stage_one(state: WorkflowState):
    state["stage"] = "waiting_for_input"
    return state

async def stage_two(state: WorkflowState):
    # 使用用户输入
    state["result"] = f"Processed: {state['user_input']}"
    state["stage"] = "completed"
    return state

workflow = StateGraph(WorkflowState)
workflow.add_node("stage_one", stage_one)
workflow.add_node("stage_two", stage_two)

workflow.set_entry_point("stage_one")
workflow.add_edge("stage_one", "stage_two")
workflow.add_edge("stage_two", END)

# 添加检查点保存器
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 第一次运行
async def run_once():
    config = {"configurable": {"thread_id": "123"}}
    async for state in graph.astream({"stage": "init"}, config=config):
        if state["stage"] == "waiting_for_input":
            print("等待用户输入...")
            return state

# 恢复运行
async def resume():
    config = {"configurable": {"thread_id": "123"}}
    
    # 获取当前状态
    current_state = graph.get_state(config)
    
    # 更新用户输入
    current_state["user_input"] = "Hello World"
    
    # 从断点恢复
    async for state in graph.astream(current_state, config=config):
        if state["stage"] == "completed":
            print(f"结果: {state['result']}")
            return state
```

---

## 九、决策树：何时用哪个？

```
┌─ 流程是否完全线性？
│  ├─ YES → 是否需要暂停-恢复？
│  │        ├─ NO  → ✅ Orchestrator (简单高效)
│  │        └─ YES → ⚠️ Orchestrator (可行但不理想)
│  │
│  └─ NO (有分支) → 是否需要暂停-恢复？
│                 ├─ NO  → ⚠️ Orchestrator (可行)
│                 └─ YES → 🚀 LangGraph (推荐)
```

---

## 十、项目的最佳实践建议

### 当前（生产环境）
✅ 继续使用 Orchestrator
- 简单、稳定、易维护
- 满足 MVP 的所有需求

### 阶段 1（用户反馈后）
如果用户反馈"希望翻译中途能修改术语"：
→ 考虑迁移到 LangGraph + 简单 Web UI

### 阶段 2（产品成熟）
如果需要：
- 多用户并发处理多文档
- 复杂的审核工作流
- 自动化的质量反馈循环
→ 完全迁移到 LangGraph + PostgresSaver + LangSmith

---

## 总结对比表

| 指标 | Orchestrator | LangGraph |
|-----|-------------|-----------|
| **现在使用** | ✅ 是 | ❌ 未启用 |
| **代码复杂度** | ⭐ 低 | ⭐⭐⭐ 中 |
| **功能完整性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **暂停-恢复** | ❌ 不支持 | ✅ 原生 |
| **可观测性** | 📊 基础 | 📊📊📊 高级 |
| **迁移难度** | - | 🔧 中等（2-3 天） |
| **何时考虑迁移** | 永远不迁移 | 6+ 个月后有新需求 |

