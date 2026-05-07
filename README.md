# 中文金融公告/研报 RAG 问答与引用溯源系统

面向中文上市公司公告、财报摘要和研报片段的本地 MVP。系统支持文档解析、文本清洗、chunk 切分、BM25 稀疏检索、Dense 向量检索、Hybrid Search、reranker 重排、RAG 回答生成、引用溯源、自动评测和 Streamlit Demo。

没有配置 API Key 时会自动使用 mock LLM，因此可以先在本地跑通完整流程。

## 技术路线图

1. 文档解析：读取 `PDF / TXT / Markdown`，PDF 按页生成 `RawDocument`。
2. 文本处理：保留金融关键数字、日期、金额、百分比，清理明显空白和页码噪声。
3. Chunk：按标题、段落、句子、固定窗口逐级切分，并保留 `doc_id / source_file / page / chunk_id / section_title`。
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

## 两个项目的数据关系

本项目和“中文金融公告 QLoRA 微调项目”可以使用同一个公开数据来源，但两者的数据形态和用途不同。

项目一是中文金融公告 QLoRA 微调项目。数据主要来自巨潮资讯网公开披露的上市公司公告和定期报告，规模约 500 条。它会把公告内容加工成 `instruction / input / output` 格式的监督微调样本，用于让模型适应中文金融公告问答、摘要、指标解释、风险提示等任务。

项目二是本仓库的中文金融公告/研报 RAG 问答与引用溯源系统。它同样可以使用巨潮资讯网公告、年报、半年报、季度报告、临时公告、风险提示公告、投资者关系活动记录表等公开资料，但这些资料在本项目中不是训练数据，而是 RAG 知识库数据。处理流程是 PDF 解析、文本清洗、chunk 切分、embedding 向量化、向量库索引、Top-K 检索、答案生成和引用溯源。

第二个项目的知识库以巨潮资讯网公开公告、年报、半年报、投资者关系活动记录表为主体。知识库可以包含第一个 QLoRA 项目涉及的部分原始文档，同时也可以额外扩展新的 PDF 文档，用于检索、问答和引用溯源。

两者的关键区别：

| 项目 | 数据来源 | 数据形态 | 数据用途 | 是否训练模型 |
|---|---|---|---|---|
| QLoRA 微调项目 | 巨潮资讯网公告和定期报告 | `instruction / input / output` 样本 | 监督微调生成模型 | 是 |
| RAG 问答与引用溯源系统 | 巨潮资讯网公告、年报、半年报、投资者关系活动记录表等原始文档 | PDF、解析文本、chunks、向量索引 | 构建可检索知识库 | 否 |

可以复用的是项目一下载过的原始公告 PDF 或原始公告文本；不建议直接把项目一构造好的 `instruction / input / output` 训练样本作为 RAG 知识库主体。原因是 instruction 数据已经经过任务化改写，可能丢失原始页码、上下文和披露来源，不利于引用溯源。

RAG 评估问题也应尽量和 QLoRA 训练样本区分。如果评估问题直接复用训练问题，模型可能依靠微调记忆回答，导致无法真实衡量检索、引用和材料外拒答能力。

两个项目的连接方式是：本 RAG 系统可以把项目一微调后的 QLoRA 模型作为答案生成器，但检索证据仍来自本项目构建的知识库。

```text
同一公开数据来源：巨潮资讯网公告 / 定期报告 / 投资者关系活动记录表
        |
        +--> 项目一：QLoRA 微调项目
        |       原始公告文本
        |          -> instruction / input / output
        |          -> Qwen3-4B + LoRA adapter
        |          -> 得到领域生成模型
        |
        +--> 项目二：RAG 问答与引用溯源系统
                原始 PDF / 原始文本
                   -> PDF 解析
                   -> 文本清洗
                   -> chunk 切分
                   -> BM25 / Dense / Hybrid 索引
                   -> Top-K 检索 + reranker
                   -> 调用 Base Model 或项目一 QLoRA 模型生成答案
                   -> 返回答案 + 文件名 + 页码 + chunk_id
```

面试中可以这样解释：

> 我把中文金融公告任务拆成“模型能力适配”和“外部知识检索”两层。第一个项目用约 500 条巨潮资讯网公告构造 instruction 数据，对 Qwen3-4B 做 QLoRA 微调，让模型更熟悉公告摘要、财务指标解释、风险提示和拒答格式。第二个项目不把数据继续当训练集，而是把巨潮公告、定期报告和投资者关系记录表解析成 RAG 知识库，通过 chunk、BM25 + Dense 混合召回、reranker 和引用溯源完成材料内问答。两个项目的连接点是：RAG 系统可以调用第一个项目微调得到的 QLoRA adapter 作为生成器，同时答案依据仍来自可追溯的检索证据，从而兼顾领域表达能力和事实可溯源性。

## 构建索引

```bash
python scripts/build_bm25_index.py \
  --chunks data/processed/chunks.jsonl \
  --output data/indexes/bm25.pkl

python scripts/build_dense_index.py \
  --chunks data/processed/chunks.jsonl \
  --persist-dir data/indexes/chroma
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
- `outputs/rag_eval_report.md`

## 如何接入第一个项目训练得到的 QLoRA 模型

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

`ablation_results.csv` 由脚本实际逐题调用 RAG Pipeline 生成，当前三组设置为：向量检索 + Qwen3-4B、向量检索 + BGE reranker + Qwen3-4B、向量检索 + BGE reranker + QLoRA 微调模型。逐题明细保存在 `outputs/ablation_details.jsonl`，用于查看每个问题的回答、引用来源、耗时和规则评估指标。

## 实验结果表格占位

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| Dense | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| Hybrid | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| Hybrid + Reranker | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

## Bad Case 分析占位

| Case | 问题 | 现象 | 初步原因 | 后续处理 |
|---|---|---|---|---|
| 1 | 待补充 | 待补充 | 待补充 | 待补充 |

## 后续扩展

1. 接入第一个 QLoRA SFT 模型。
2. 支持长文档多跳问答。
3. 支持更严格的事实一致性评测。
4. 支持 Agent 工具调用。

## 简历展示口径

中文金融公告/研报 RAG 问答与引用溯源系统：面向中文上市公司公告、财报摘要和研报片段，构建支持文档解析、chunk 切分、BM25 + Dense 混合召回、reranker 重排、答案引用溯源和自动评测的 RAG 问答系统。设计材料内问答、摘要归纳、信息抽取、材料外拒答和投资建议拒答等测试集，对比 BM25、Dense、Hybrid、Hybrid + Reranker 在 Recall@K、MRR、引用准确率和拒答准确率上的表现，并通过 Streamlit Demo 展示问答结果、证据片段和检索排序。
