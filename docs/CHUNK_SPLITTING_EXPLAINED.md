# Chunk 切分详解：代码位置与工作原理

## 一、快速定位代码

### 核心代码位置

| 文件 | 位置 | 功能 |
|------|------|------|
| **src/agents/preprocessor.py** | 第 91-238 行 | 整个切分逻辑 |
| **src/agents/preprocessor.py** | 第 91-110 行 | `_chunk_text()` - 主切分方法 |
| **src/agents/preprocessor.py** | 第 112-121 行 | `_split_by_paragraphs()` - 按段落切分 |
| **src/agents/preprocessor.py** | 第 123-170 行 | `_merge_into_chunks()` - 合并成块 |
| **src/agents/preprocessor.py** | 第 172-184 行 | `_get_overlap_content()` - 处理重叠 |
| **src/agents/preprocessor.py** | 第 186-221 行 | `_create_chunk()` - 创建块对象 |
| **config/settings.py** | 第 29-31 行 | 切分配置参数 |

---

## 二、切分算法详解

### 2.1 整体流程

```python
# src/agents/preprocessor.py 第 91-110 行

async def _chunk_text(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
    """切分文本"""
    # 第 1 步：按段落切分
    paragraphs = self._split_by_paragraphs(text)
    
    # 第 2 步：合并成合适大小的 chunks
    chunks = self._merge_into_chunks(paragraphs)
    
    # 第 3 步：可选的 LLM 优化（长文本）
    if len(text) > 10000:
        chunks = await self._optimize_chunks_with_llm(chunks)
    
    return chunks
```

**流程可视化**：

```
原文本：10000 字
    ↓
[第 1 步：按段落切分] ← _split_by_paragraphs()
    ↓
段落列表：100 个段落（每个段落 50-200 字）
    ↓
[第 2 步：按大小合并] ← _merge_into_chunks()
    ↓
Chunk 列表：5 个 chunk（每个 1500 字，200 字重叠）
    ↓
[第 3 步：LLM 优化] （仅当 > 10000 字符时）
    ↓
最终 Chunk 列表
```

---

### 2.2 第 1 步：按段落切分

**代码**：[preprocessor.py 第 112-121 行](src/agents/preprocessor.py#L112-L121)

```python
def _split_by_paragraphs(self, text: str) -> List[str]:
    """按段落切分"""
    # 按双换行分割（段落分隔符）
    paragraphs = text.split('\n\n')
    
    # 清理空段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs
```

**例子**：

```
原文本：
───────────────────
第一段：Machine learning is a subset of artificial intelligence...

第二段：The field has evolved significantly over the past decade...

第三段：Key techniques include supervised learning, unsupervised learning...
───────────────────

↓ split('\n\n')

段落列表：[
    "Machine learning is a subset of artificial intelligence...",
    "The field has evolved significantly over the past decade...",
    "Key techniques include supervised learning, unsupervised learning..."
]
```

**关键点**：
- ✅ 按**双换行**（`\n\n`）作为段落边界
- ✅ 清理空段落和首尾空格
- ⚠️ 假设输入文本已正确格式化（段落间有空行）

---

### 2.3 第 2 步：按大小合并成 Chunk

**代码**：[preprocessor.py 第 123-170 行](src/agents/preprocessor.py#L123-L170)

```python
def _merge_into_chunks(self, paragraphs: List[str]) -> List[DocumentChunk]:
    """将段落合并成合适大小的chunks"""
    chunks = []
    current_content = []      # 当前 chunk 的段落列表
    current_length = 0        # 当前 chunk 的字符数
    chunk_index = 0
    position = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # ★ 核心判断：当前 chunk + 新段落是否超过 chunk_size？
        if current_length + para_length > self.chunk_size and current_content:
            # 超过限制 → 保存当前 chunk，开始新的
            chunk_text = '\n\n'.join(current_content)
            chunk = self._create_chunk(chunk_text, chunk_index, position, position + len(chunk_text))
            chunks.append(chunk)
            
            position += len(chunk_text) + 2  # +2 for '\n\n'
            chunk_index += 1
            
            # ★ 关键：添加重叠部分到新 chunk
            overlap_content = self._get_overlap_content(current_content)
            current_content = overlap_content + [para]
            current_length = sum(len(p) for p in current_content)
        else:
            # 未超过 → 添加到当前 chunk
            current_content.append(para)
            current_length += para_length
    
    # 处理最后一个 chunk（不足 chunk_size 也要保存）
    if current_content:
        chunk_text = '\n\n'.join(current_content)
        chunk = self._create_chunk(chunk_text, chunk_index, position, position + len(chunk_text))
        chunks.append(chunk)
    
    # ★ 设置前后 chunk 关联（用于上下文）
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.context.prev_chunk_id = chunks[i-1].chunk_id
        if i < len(chunks) - 1:
            chunk.context.next_chunk_id = chunks[i+1].chunk_id
    
    return chunks
```

**配置参数**：[settings.py 第 29-31 行](config/settings.py#L29-L31)

```python
class TranslationSettings(BaseModel):
    chunk_size: int = 1500        # 每个 chunk 的目标大小（字符数）
    chunk_overlap: int = 200      # 相邻 chunk 的重叠大小（字符数）
```

#### **切分演示**：

```
假设：chunk_size = 1500, chunk_overlap = 200

段落列表：
Para 1: 500 字
Para 2: 400 字
Para 3: 600 字
Para 4: 450 字
Para 5: 300 字
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第 1 个 chunk：
  当前长度 = 0
  + Para 1 (500)   → 500 ✓
  + Para 2 (400)   → 900 ✓
  + Para 3 (600)   → 1500 ✓（达到限制）
  + Para 4 (450)   → 1950 ✗（超过限制，不加）
  
  【Chunk_0001】保存：
  ├─ Para 1 (500)
  ├─ Para 2 (400)
  └─ Para 3 (600)
  总长度：1500 字

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第 2 个 chunk：
  【重叠】获取前一个 chunk 的重叠内容（200 字）
  Para 3 最后 200 字 ← 添加到新 chunk 开头
  
  当前长度 = 200（重叠部分）
  + Para 4 (450)   → 650 ✓
  + Para 5 (300)   → 950 ✓
  + Para 6 (...)   → ...继续判断
  
  【Chunk_0002】保存：
  ├─ Para 3 最后 200 字 （重叠）
  ├─ Para 4 (450)
  └─ Para 5 (300)
  总长度：950 字

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

特点：
✅ Chunk_0001 和 Chunk_0002 有 200 字重叠
✅ 确保跨 chunk 边界的上下文连贯性
✅ 翻译时可以参考重叠部分做更好的决策
```

---

### 2.4 第 3 步：处理重叠部分

**代码**：[preprocessor.py 第 172-184 行](src/agents/preprocessor.py#L172-L184)

```python
def _get_overlap_content(self, content: List[str]) -> List[str]:
    """获取重叠内容（从末尾向前）"""
    if not content:
        return []
    
    overlap = []
    overlap_length = 0
    
    # 从后往前添加段落，直到达到 chunk_overlap 大小
    for para in reversed(content):
        if overlap_length + len(para) <= self.chunk_overlap:
            overlap.insert(0, para)  # 插入到前面（保持顺序）
            overlap_length += len(para)
        else:
            break  # 达到重叠大小限制，停止
    
    return overlap
```

**演示**：

```
当前 Chunk 的内容：
  Para 1: 500 字
  Para 2: 400 字  ← 最后
  Para 3: 600 字

chunk_overlap = 200

从后往前遍历：
  Para 3: 600 字 > 200 字（超过）→ 不加 ✗
  Para 2: 400 字，overlap_length = 0 + 400 = 400 > 200 ✗
  Para 1: 无法加（已达到）

结果：overlap = [] （无重叠，因为最后的段落都太大）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

另一个例子：
当前 Chunk 的内容：
  Para 1: 300 字
  Para 2: 100 字  ← 最后
  Para 3: 150 字

chunk_overlap = 200

从后往前遍历：
  Para 3: 150 字，overlap_length = 0 + 150 = 150 ≤ 200 ✓ 加入
  Para 2: 100 字，overlap_length = 150 + 100 = 250 > 200 ✗ 不加

结果：overlap = [Para 3] （150 字重叠）
```

---

### 2.5 第 4 步：创建 Chunk 对象

**代码**：[preprocessor.py 第 186-221 行](src/agents/preprocessor.py#L186-L221)

```python
def _create_chunk(self, content: str, index: int, start_pos: int, end_pos: int) -> DocumentChunk:
    """创建 chunk 对象并分析元数据"""
    chunk_id = f"chunk_{index:04d}"  # chunk_0000, chunk_0001, ...
    
    # 分析段落类型和难度
    paragraph_type = self._detect_paragraph_type(content)
    difficulty = self._estimate_difficulty(content)
    
    # 创建元数据
    metadata = ChunkMetadata(
        paragraph_type=paragraph_type,
        estimated_difficulty=difficulty,
        char_count=len(content),
        word_count=len(content.split()),
        sentence_count=content.count('.') + content.count('!') + content.count('?')
    )
    
    context = ChunkContext()
    
    # 创建 DocumentChunk 对象
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        start_position=start_pos,
        end_position=end_pos,
        sequence_number=index,
        context=context,
        metadata=metadata
    )
```

**Chunk 对象包含的信息**：

```python
DocumentChunk {
    chunk_id: "chunk_0001",           # 全局唯一 ID
    content: "...",                   # 实际文本内容
    sequence_number: 0,               # 顺序号
    
    metadata: {
        char_count: 1500,             # 字符数
        word_count: 250,              # 单词数
        sentence_count: 8,            # 句子数
        paragraph_type: "NARRATIVE",  # 段落类型
        estimated_difficulty: "MEDIUM"# 难度估计
    },
    
    context: {
        prev_chunk_id: "chunk_0000",  # 前一个 chunk
        next_chunk_id: "chunk_0002",  # 后一个 chunk
        context_before: "...",        # 前面的重叠内容
        context_after: "..."          # 后面的重叠内容
    }
}
```

---

## 三、段落类型检测

**代码**：[preprocessor.py 第 223-228 行](src/agents/preprocessor.py#L223-L228)

```python
def _detect_paragraph_type(self, text: str) -> ParagraphType:
    """检测段落类型（启发式规则）"""
    if '"' in text or '"' in text or '「' in text:
        return ParagraphType.DIALOGUE      # 对话（含引号）
    elif text.startswith(('1.', '2.', '-', '•', '*')):
        return ParagraphType.EXPOSITION    # 说明性列表
    else:
        return ParagraphType.NARRATIVE     # 叙述性段落（默认）
```

**类型定义**：[models/document.py](src/models/document.py)

```python
class ParagraphType(str, Enum):
    NARRATIVE = "narrative"       # 叙述文本
    DIALOGUE = "dialogue"         # 对话文本
    EXPOSITION = "exposition"     # 说明文本（列表、步骤）
```

---

## 四、难度估计

**代码**：[preprocessor.py 第 230-240 行](src/agents/preprocessor.py#L230-L240)

```python
def _estimate_difficulty(self, text: str) -> DifficultyLevel:
    """基于句子长度估计翻译难度"""
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = len(text) / max(1, sentence_count)
    
    if avg_sentence_length > 100:
        return DifficultyLevel.HARD        # 长句子 = 高难度
    elif avg_sentence_length > 50:
        return DifficultyLevel.MEDIUM      # 中等长度
    else:
        return DifficultyLevel.EASY        # 短句子 = 低难度
```

**难度等级**：

```python
class DifficultyLevel(str, Enum):
    EASY = "easy"           # 平均句子 < 50 字
    MEDIUM = "medium"       # 平均句子 50-100 字
    HARD = "hard"           # 平均句子 > 100 字
```

**用途**：
- 翻译时调整策略（难的段落需要更多 LLM 调用）
- 人工审核时优先审核难段落
- 成本估算

---

## 五、完整的切分流程示例

### 输入文本

```
Machine learning is a subset of artificial intelligence. It focuses on the development of computer programs that can learn from and make predictions based on data.

The field has evolved significantly over the past decade. Applications range from natural language processing to computer vision. Many organizations now rely on machine learning for critical business decisions.

Key techniques include supervised learning, unsupervised learning, and reinforcement learning. Each has its own advantages and use cases. Practitioners must understand these distinctions to choose the right approach.
```

### 切分过程

```
第 1 步：按 \n\n 分割成段落

Paragraphs = [
    Para_0: "Machine learning is a subset of artificial intelligence. It focuses on the development of computer programs that can learn from and make predictions based on data."
    Para_1: "The field has evolved significantly over the past decade. Applications range from natural language processing to computer vision. Many organizations now rely on machine learning for critical business decisions."
    Para_2: "Key techniques include supervised learning, unsupervised learning, and reinforcement learning. Each has its own advantages and use cases. Practitioners must understand these distinctions to choose the right approach."
]

长度：
  Para_0: 180 字
  Para_1: 240 字
  Para_2: 240 字

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第 2 步：合并成 chunk_size = 1500, chunk_overlap = 200

Chunk 1：
  180 + 240 + 240 = 660 字 < 1500
  全部放入 Chunk 1

Chunk 1 = [Para_0 + Para_1 + Para_2]
Total = 660 字

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第 3 步：如果文本 > 10000 字，进行 LLM 优化

当前文本 = 660 字 < 10000 字
→ 跳过 LLM 优化

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最终结果：

[
    DocumentChunk {
        chunk_id: "chunk_0000",
        sequence_number: 0,
        content: "Machine learning is a subset of artificial intelligence. It focuses on the development of computer programs that can learn from and make predictions based on data.\n\nThe field has evolved significantly over the past decade. Applications range from natural language processing to computer vision. Many organizations now rely on machine learning for critical business decisions.\n\nKey techniques include supervised learning, unsupervised learning, and reinforcement learning. Each has its own advantages and use cases. Practitioners must understand these distinctions to choose the right approach.",
        metadata: {
            char_count: 660,
            word_count: 110,
            sentence_count: 6,
            paragraph_type: "NARRATIVE",
            estimated_difficulty: "MEDIUM"  // avg = 660/6 = 110 > 100
        }
    }
]
```

---

## 六、Chunk 的使用

### 翻译流程中的使用

```python
# orchestrator.py 中

# 第 1 步：预处理得到 chunks
document = await preprocessor.run(input_text)
chunks = document.chunks  # ← 这里就是切分得到的 chunk 列表

# 第 2 步：术语提取（前 5 个）
for chunk in chunks[:5]:
    await terminology_agent.run(chunk.content, domain)

# 第 3 步：翻译（所有 chunk）
for chunk in chunks:
    unit = await translator.run(
        source_text=chunk.content,  # ← 使用 chunk 的内容
        terminology=terminology,
        context=chunk.context.context_before,  # ← 使用前后文
        chunk_id=chunk.chunk_id
    )

# 第 4 步：验证（所有 unit）
for unit in translation_result.units:
    await validator.run(unit, unit.source_text)
```

---

## 七、Chunk 的优势

✅ **段落边界对齐**
- 不会在句子中间切割
- 不会在词中间切割

✅ **重叠保证上下文**
- 翻译时可以参考前后内容
- 避免跨 chunk 边界的翻译不连贯

✅ **可配置**
```python
# 快速模式：更大的 chunk
settings.translation.chunk_size = 2000  # 从 1500 → 2000

# 详细模式：更小的 chunk
settings.translation.chunk_size = 800   # 从 1500 → 800
```

✅ **元数据丰富**
- 知道难度和类型
- 可以区分对待（对话、列表、叙述）

---

## 八、代码位置速查表

| 功能 | 文件 | 行数 | 函数名 |
|------|------|------|--------|
| **整体流程** | preprocessor.py | 91-110 | `_chunk_text()` |
| **按段落切分** | preprocessor.py | 112-121 | `_split_by_paragraphs()` |
| **合并成 chunk** | preprocessor.py | 123-170 | `_merge_into_chunks()` |
| **处理重叠** | preprocessor.py | 172-184 | `_get_overlap_content()` |
| **创建对象** | preprocessor.py | 186-221 | `_create_chunk()` |
| **类型检测** | preprocessor.py | 223-228 | `_detect_paragraph_type()` |
| **难度估计** | preprocessor.py | 230-240 | `_estimate_difficulty()` |
| **配置参数** | settings.py | 29-31 | `TranslationSettings.chunk_size/overlap` |

---

## 总结

**Chunk 切分是一个 3 步过程**：

```
步骤 1: 按双换行切分成段落
  ↓
步骤 2: 合并段落成指定大小的 chunk（有重叠）
  ↓
步骤 3: 分析元数据并设置前后关联

结果：一个精心设计的 chunk 列表
  ✅ 保留段落边界（不会切断句子）
  ✅ 有重叠部分（确保上下文）
  ✅ 包含元数据（类型、难度）
  ✅ 链接相邻 chunk（便于引用上下文）
```

**关键配置参数**：
- `chunk_size = 1500` - 每个 chunk 的目标大小（字符数）
- `chunk_overlap = 200` - 相邻 chunk 的重叠大小（字符数）

这两个参数直接影响翻译的成本、速度和质量！
