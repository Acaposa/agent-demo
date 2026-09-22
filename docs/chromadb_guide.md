# ChromaDB 使用指南

## ChromaDB 简介

ChromaDB 是一个开源的向量数据库，专为 LLM 应用设计。它的定位是
轻量、易用、可嵌入式，适合快速构建 RAG 原型和中小规模生产系统。

ChromaDB 的核心优势有三个。第一是 API 简洁，几行代码就能完成
集合创建、数据插入和相似度查询。第二是支持持久化，数据存到本地
磁盘，重启不丢失。第三是内置默认嵌入模型，不配置也能跑起来。
缺点是默认模型只支持英文，中文场景必须自定义嵌入函数。

ChromaDB 有两种运行模式。内存模式（EphemeralClient）数据只在
进程内存里，退出即丢失，适合测试。持久化模式（PersistentClient）
数据写到本地目录，重启后仍在，适合生产。

## 核心概念

集合（Collection）是 ChromaDB 的基本组织单位，相当于关系数据库
里的表。一个集合存储一类文档，有独立的嵌入函数和距离度量。

文档（Document）是集合里的一条记录，通常是一个文本片段。每条
文档有一个唯一的 ID、内容、可选的元数据（metadata）。

嵌入（Embedding）是文档对应的向量表示。ChromaDB 可以自动调用
嵌入函数生成，也可以由用户直接提供。

元数据（Metadata）是附加在文档上的键值对，用于过滤和溯源。比如
存 source 字段记录文档来源，存 chunk 字段记录片段序号。

## 创建集合

创建集合时可以指定嵌入函数和距离度量：

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

collection = client.create_collection(
    name="docs",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)