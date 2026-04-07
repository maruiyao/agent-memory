# Memory System（记忆管理系统）

一个**适用于任何 AI 应用 / Agent / RAG** 的“记忆管理系统（Memory Manager）”实现：将对话与任务过程中产生的信息，按照认知科学启发的分层结构进行**存储、检索、整合与遗忘**，并可与外部知识库的 **RAG 检索增强**能力协同使用。

本模块的设计与实现主要参考教程《第八章 记忆与检索》（你本地的 `第八章 记忆与检索.md`），并在工程上落地为可复用的 Python 组件。

---

## 核心能力

- **统一入口**：`MemoryManager` 统一管理多种记忆类型的写入、检索、更新、删除、遗忘、整合与统计
- **四类记忆**（可按需启用）：
  - **工作记忆** `WorkingMemory`：短期上下文（纯内存），容量/TTL 管理，适合会话态信息
  - **情景记忆** `EpisodicMemory`：具体事件与会话片段（SQLite 权威存储 + Qdrant 向量索引）
  - **语义记忆** `SemanticMemory`：抽象知识、概念与规则（Qdrant + Neo4j 知识图谱）
  - **感知记忆** `PerceptualMemory`：多模态长期记忆（SQLite + Qdrant；可选 CLIP/CLAP，缺依赖则降级）
- **统一嵌入服务**：`embedding.py` 提供 `get_text_embedder()`，支持 DashScope / 本地 Transformer / TF‑IDF 兜底
- **RAG 管道**：`rag/pipeline.py` 支持“任意格式文档 → Markdown → 分块 → 嵌入 → Qdrant 检索”，并对 PDF 做增强清洗

---

## 架构概览

```text
hello_agents/memory
├── base.py                 # MemoryItem / MemoryConfig / BaseMemory
├── manager.py              # MemoryManager（统一调度）
├── embedding.py            # 统一嵌入（dashscope/local/tfidf）
├── types/                  # 记忆类型层
│   ├── working.py          # WorkingMemory
│   ├── episodic.py         # EpisodicMemory（SQLite + Qdrant）
│   ├── semantic.py         # SemanticMemory（Qdrant + Neo4j）
│   └── perceptual.py       # PerceptualMemory（多模态，按模态分集合）
├── storage/                # 存储后端层
│   ├── document_store.py   # SQLiteDocumentStore（权威存储）
│   ├── qdrant_store.py     # QdrantVectorStore（向量存储）
│   └── neo4j_store.py      # Neo4jGraphStore（知识图谱）
└── rag/                    # RAG 子系统（外部知识）
    ├── document.py         # Document / DocumentProcessor（分块）
    └── pipeline.py         # 端到端 pipeline（转换、清洗、索引、检索、融合）
```

---

## 依赖与安装建议

本模块依赖分为两类：

- **基础依赖**：随框架核心安装即可（`openai/requests/pydantic/...`）
- **可选依赖**（按需启用）：
  - 记忆系统：`qdrant-client`、`neo4j`、`spacy`、`scikit-learn`
  - RAG：`markitdown`、`sentence-transformers/transformers/torch`、`pypdf` 等

如果你在框架根目录使用 pip extras（推荐）：

```bash
pip install -e ".[memory-rag]"
```

---

## 环境变量（.env）

### Qdrant（向量数据库）

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=hello_agents_vectors
QDRANT_DISTANCE=cosine
```

### Neo4j（知识图谱，可选：语义记忆使用）

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

### Embedding（统一嵌入）

```env
# dashscope | local | tfidf
EMBED_MODEL_TYPE=dashscope

# dashscope 默认 text-embedding-v3；local 默认 sentence-transformers/all-MiniLM-L6-v2
EMBED_MODEL_NAME=

# 如果用 dashscope / 其他云端 embedding，需要提供
EMBED_API_KEY=
EMBED_BASE_URL=
```

> 提示：如果你只想“先跑通”，可以把 `EMBED_MODEL_TYPE=local`，并安装 `sentence-transformers`。

---

## 快速使用

### 1）作为通用记忆管理器使用（不绑定任何 Agent 框架）

```python
from hello_agents.memory import MemoryManager

mm = MemoryManager(user_id="user123")

# 写入（也可让 manager 自动分类）
mem_id = mm.add_memory(
    content="用户叫张三，偏好用 Python 写数据分析脚本",
    memory_type="semantic",
    importance=0.8,
    metadata={"source": "chat"}
)

# 检索（跨类型召回）
hits = mm.retrieve_memories("张三 喜好", limit=5)
for m in hits:
    print(m.memory_type, m.importance, m.content)
```

### 2）作为 RAG 文档管道使用（把文件变成可检索知识库）

你可以直接调用 `hello_agents.memory.rag` 暴露的 pipeline 函数（用于摄取、分块、索引、检索）。示例函数包括：

- `load_and_chunk_texts()`：加载并分块（支持多格式文件）
- `index_chunks()`：嵌入并写入 Qdrant
- `search_vectors()` / `search_vectors_expanded()`：向量检索（可选 MQE/HyDE 扩展检索）

> 具体参数请以 `hello_agents/memory/rag/__init__.py` 的导出列表与 `rag/pipeline.py` 为准。

---

## 设计原则（为什么这样做）

- **分层**：把“记忆类型（行为）”与“存储后端（介质）”拆开，方便替换数据库、替换嵌入模型
- **可插拔**：你可以只用 `WorkingMemory` 做短期上下文，也可以启用 `SemanticMemory` 做知识图谱增强
- **工程可用**：权威存储（SQLite）确保可追溯；向量库（Qdrant）提供高召回；图数据库（Neo4j）支持关系推理

