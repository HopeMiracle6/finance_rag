# 中文金融公告/研报 RAG 问答与引用溯源系统

面向中文上市公司公告、财报摘要和研报片段的本地 MVP。系统支持文档解析、文本清洗、chunk 切分、BM25 稀疏检索、Dense 向量检索、Hybrid Search、reranker 重排、RAG 回答生成、引用溯源、自动评测和 Streamlit Demo。

没有配置 API Key 时会自动使用 mock LLM，因此可以先在本地跑通完整流程。

## 技术路线图

1. 文档解析：读取 `PDF / TXT / Markdown`，PDF 按页生成 `RawDocument`。
2. 文本处理：保留金融关键数字、日期、金额、百分比，清理明显空白和页码噪声。
3. Chunk：按整份 PDF 连续文档流切分，支持跨页 overlap，并保留 `doc_id / source_file / page_start / page_end / chunk_id / section_title`。
4. 检索：BM25 使用 `jieba + rank_bm25`，Dense 使用 `sentence-transformers + Chroma`，无本地模型时走 deterministic fallback embedding。
5. 融合：Hybrid 默认使用 Reciprocal Rank Fusion。
6. 重排：优先使用 `FlagEmbedding` reranker；无本地模型时走关键词重排 fallback。
7. 生成：OpenAI-compatible API；无 API Key 时使用 mock LLM。
8. 评测：检索 Recall@K / MRR，RAG 格式、引用、拒答、关键词覆盖、证据命中。

## 目录结构

```text
finance-rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
├── data/
├── src/
├── scripts/
├── demo/
├── outputs/
└── tests/
```

## 安装方法

```bash
pip install -r requirements.txt
```

如需真实下载 BGE embedding / reranker 模型，设置：

```bash
set FINANCE_RAG_ALLOW_MODEL_DOWNLOAD=1
```

默认不主动下载模型，避免本地无网络时卡住；会使用 fallback embedding / reranker 跑通 MVP。

## 快速开始

```bash
python scripts/ingest_docs.py
python scripts/build_bm25_index.py
python scripts/build_dense_index.py

python scripts/ask.py \
  --question "公司净利润增长的主要原因是什么？" \
  --retrieval-mode hybrid \
  --use-reranker true
```

## 数据格式

`RawDocument`：

```json
{"doc_id":"...","source_file":"sample_announcement.txt","file_type":"txt","page":null,"text":"...","metadata":{}}
```

`TextChunk`：

```json
{"chunk_id":"...","doc_id":"...","source_file":"sample_announcement.txt","page":null,"section_title":"二、业绩变动主要原因","text":"...","token_count":256,"metadata":{}}
```


## 当前实现状态

当前项目二已经接入巨潮资讯网公开 PDF 作为 RAG 知识库，不再只使用示例 TXT。已验证的一版本地知识库规模如下：

| 项目 | 数量 |
|---|---:|
| 巨潮公告/报告 PDF | 497 |
| 解析页数 | 4,165 |
| 文档级 chunks | 6,777 |
| 跨页 chunks | 3,459 |
| 异常大跨度页码 | 0 |

切分方式已经从“每页单独切分”改为“整份 PDF 连续切分”。系统会在解析时保留页码标记，切分后为每个 chunk 记录 `page_start/page_end`，因此回答引用可以展示 `91-92` 这类页码范围，适合年报、半年报和投资者关系记录表这类跨页长文档。

由于 Windows OneDrive + 中文长路径下 Chroma/SQLite 可能出现 `disk I/O error`，当前推荐把 Chroma 向量库放在纯英文路径：

```yaml
chroma_persist_dir: D:/finance_rag_chroma
```

## 构建索引

```bash
python scripts/build_bm25_index.py \
  --chunks data/processed/chunks.jsonl \
  --output data/indexes/bm25.pkl

python scripts/build_dense_index.py \
  --chunks data/processed/chunks.jsonl \
  --persist-dir D:/finance_rag_chroma \
  --embedding-model BAAI/bge-m3 \
  --chroma-batch-size 512
```

## 命令行问答

```bash
python scripts/ask.py \
  --question "公司净利润增长的主要原因是什么？" \
  --retrieval-mode hybrid \
  --use-reranker true \
  --top-k 30 \
  --final-top-n 5
```

输出包含问题、结构化回答、引用来源和证据片段。

## 启动 Streamlit Demo

```bash
streamlit run demo/app.py
```

页面支持上传文档、重建索引、选择检索模式、启用/关闭 reranker、查看回答、引用来源和检索排序。

## 自动评测

先生成示例评测集：

```bash
python scripts/make_sample_eval.py
```

运行检索评测：

```bash
python scripts/evaluate_retrieval.py
```

运行 RAG 评测：

```bash
python scripts/evaluate_rag.py
```

报告输出到：

- `outputs/retrieval_report.md`
- `outputs/eval_report.md`

## 如何接入训练得到的 QLoRA 模型

默认配置已经把生成器切到本地 QLoRA：

```yaml
llm:
  provider: local_qlora
  base_model: Qwen/Qwen3-4B
  adapter_path: D:/LLaMA-Factory/saves/qwen3-4b/lora/finance_sft_json
  load_in_4bit: true
  temperature: 0.2
  max_tokens: 1024
```

我在第一个项目的推理配置中找到的 adapter 路径是 `D:\LLaMA-Factory\saves\qwen3-4b\lora\finance_sft_json`。如果你的 adapter 后续导出到其他位置，可以临时用环境变量覆盖：

```powershell
$env:QLORA_ADAPTER_PATH="D:\OneDrive\桌面\2025秋\大模型\中文金融公告解读助手\outputs\qwen3_4b_finance_qlora\final_adapter"
$env:QLORA_BASE_MODEL="Qwen/Qwen3-4B"
$env:QLORA_LOAD_IN_4BIT="true"
```

首次加载本地模型时，如果本机没有缓存 `Qwen/Qwen3-4B`，需要允许下载：

```powershell
$env:FINANCE_RAG_ALLOW_MODEL_DOWNLOAD="1"
```

本地 QLoRA 问答命令：

```bash
python scripts/ask_local_qlora.py \
  --question "公司净利润增长的主要原因是什么？" \
  --retrieval-mode hybrid \
  --use-reranker true
```

如果 adapter 路径不存在、模型未缓存、显存不足或依赖不完整，系统会自动回退到 mock LLM，检索和引用链路仍可运行。

## 对比实验设计

| 实验组 | 检索 | 生成模型 | 目的 |
|---|---|---|---|
| Base Model | 无 | Qwen/Qwen3-4B | 观察基础模型直接回答能力 |
| QLoRA Only | 无 | Qwen3-4B + 金融 QLoRA adapter | 观察领域微调后的直接回答能力 |
| RAG + Base | BM25 + Dense + Hybrid | Qwen/Qwen3-4B | 观察外部证据对基础模型的增益 |
| RAG + QLoRA | BM25 + Dense + Hybrid | Qwen3-4B + 金融 QLoRA adapter | 观察检索证据和领域微调叠加后的效果 |

建议统一使用同一批 `eval_questions.jsonl`，对比格式遵循率、引用存在率、拒答准确率、关键词覆盖率和证据命中率。

## outputs 展示产物

每次调用 `RAGPipeline.ask()` 后，系统会自动追加保存：

- `outputs/qa_results.jsonl`：问题、回答、引用来源、召回片段、模型配置和耗时。
- `outputs/retrieved_contexts.csv`：每个问题召回的 Top-K 文档片段，使用 `utf-8-sig`，方便 Excel 打开。
- `outputs/citation_report.html`：由 `scripts/build_citation_report.py` 生成的引用溯源页面。
- `outputs/eval_report.json` / `outputs/eval_report.md`：由 `scripts/evaluate_rag.py` 生成的本地规则评估报告。
- `outputs/ablation_results.csv`：由 `scripts/build_ablation_results.py` 实际运行生成的消融实验表。

生成展示产物：

```bash
python scripts/build_citation_report.py
python scripts/evaluate_rag.py
python scripts/build_ablation_results.py
```

`ablation_results.csv` 由脚本实际逐题调用 RAG Pipeline 生成，当前三组设置为：向量检索 + Qwen3-4B、向量检索 + reranker + Qwen3-4B、向量检索 + reranker + QLoRA 微调模型。逐题明细保存在 `outputs/ablation_details.jsonl`，用于查看每个问题的回答、引用来源、耗时和规则评估指标。

真实消融实验命令：

```powershell
python scripts/build_ablation_results.py `
  --generator local `
  --top-k 30 `
  --final-top-n 5 `
  --max-tokens 512
```

当前真实实验结果：

| setting | 模型链路 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency |
|---|---|---:|---:|---:|---:|---:|
| baseline | Dense(simple_hash fallback) + Qwen3-4B | 0.0000 | 0.0646 | 0.1000 | 0.0000 | 18.7035s |
| with_reranker | Dense(simple_hash fallback) + keyword reranker + Qwen3-4B | 0.0333 | 0.0896 | 0.1150 | 0.0417 | 10.9209s |
| with_lora | Dense(simple_hash fallback) + keyword reranker + QLoRA | 0.1125 | 0.1125 | 0.1150 | 0.1417 | 29.5828s |

说明：这组实验基于 `outputs/sample_questions.jsonl` 中的 40 条样例问题。当前环境中 `BAAI/bge-m3` 因 torch 版本限制未加载，Dense 使用 `simple_hash` fallback；reranker 本地模型不存在，使用 `keyword_overlap` fallback。

## 检索评测结果

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.2917 | 0.4167 | 0.5278 | 0.7222 | 0.4473 |
| Dense | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid | 0.2778 | 0.3750 | 0.4167 | 0.5278 | 0.3845 |
| Hybrid + Reranker | 0.4306 | 0.5833 | 0.7222 | 0.7778 | 0.5753 |

## Bad Case 分析占位

| Case | 问题 | 现象 | 初步原因 | 后续处理 |
|---|---|---|---|---|
| 1 | 待补充 | 待补充 | 待补充 | 待补充 |

## 后续扩展

1. 接入第一个 QLoRA SFT 模型。
2. 支持长文档多跳问答。
3. 支持更严格的事实一致性评测。
4. 支持 Agent 工具调用。
