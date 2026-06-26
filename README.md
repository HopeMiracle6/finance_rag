# 中文金融公告/研报 RAG 问答与引用溯源系统

面向中文上市公司公告、财报摘要和研报片段的本地 MVP。系统支持文档解析、文本清洗、chunk 切分、BM25 稀疏检索、Dense 向量检索、Hybrid Search、reranker 重排、RAG 回答生成、引用溯源、自动评测和 Streamlit Demo。

默认生成配置为 DeepSeek API 的 `deepseek-v4-pro`，同时保留本地 `Qwen/Qwen3-4B` / QLoRA。正式运行默认启用 strict 模式：缺少 API Key、模型初始化失败或推理失败时直接报错，不再静默返回模板答案。

## 技术路线图

1. 文档解析：读取 `PDF / TXT / Markdown`，PDF 按页生成 `RawDocument`。
2. 文本处理：保留金融关键数字、日期、金额、百分比，清理明显空白和页码噪声。
3. Chunk：按整份 PDF 连续文档流切分，支持跨页 overlap，并保留 `doc_id / source_file / page_start / page_end / chunk_id / section_title`。
4. 检索：BM25 使用 `jieba + rank_bm25`，Dense 使用真实 `BAAI/bge-m3 + sentence-transformers + Chroma`。
5. 融合：Hybrid 默认使用 Reciprocal Rank Fusion。
6. 重排：使用本地 `D:\models\bge-reranker-v2-m3`，通过 `FlagEmbedding` 加载。
7. 生成：支持 DeepSeek V4 Pro、本地 QLoRA 和其他 OpenAI-compatible API。
8. 评测：检索 Recall@K / MRR，RAG 格式、引用、拒答、关键词覆盖、证据命中。
9. 调用审计：记录实际 provider、model、backend、request_id、token 用量、生成耗时和 fallback 状态。

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

### 1. 激活 conda 环境（推荐）

```powershell
conda activate finance_rag
```

环境路径：`D:\Anaconda\envs\finance_rag`，已预装本项目所需依赖。

也可自行创建：

```powershell
conda create -n finance_rag python=3.11 -y
conda activate finance_rag
pip install -r requirements.txt
```

关键推理依赖已按当前 `finance_rag` 环境固定版本。执行以下命令应返回 `No broken requirements found`：

```powershell
python -m pip check
```

默认使用本地缓存/本地目录中的真实 BGE embedding 和 reranker 模型。若本机尚未缓存模型，可临时允许下载：

```powershell
$env:FINANCE_RAG_ALLOW_MODEL_DOWNLOAD="1"
```

当前配置已禁用 embedding / reranker 的静默 fallback：如果 `BAAI/bge-m3` 或本地 `D:\models\bge-reranker-v2-m3` 加载失败，程序会直接报错，避免结果被误写成真实模型。

## 快速开始

### 1. 配置 DeepSeek API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=你的DeepSeek API Key
```

`.env` 已被 `.gitignore` 排除，不要把真实 Key 写入 `.env.example`、README、代码或提交记录。也可以只在当前 PowerShell 会话设置：

```powershell
$env:DEEPSEEK_API_KEY="你的DeepSeek API Key"
```

默认生成配置为：

```yaml
llm:
  provider: deepseek
  base_url: https://api.deepseek.com
  model: deepseek-v4-pro
  strict: true
  thinking_enabled: false
```

项目通过 `python-dotenv` 自动读取根目录 `.env`。RAG 批量问答默认关闭 V4 Pro 思考模式，避免推理 token 占满输出上限；如需开启，可设置 `DEEPSEEK_THINKING=true`。DeepSeek 使用 OpenAI-compatible SDK 调用，官方文档见 <https://api-docs.deepseek.com/>。

### 2. 构建索引并问答

```bash
python scripts/build_bm25_index.py
python scripts/build_dense_index.py

python scripts/ask.py \
  --question "公司净利润增长的主要原因是什么？" \
  --retrieval-mode hybrid \
  --use-reranker true
```

上述命令直接使用当前 `data/processed/kb_chunks.jsonl` 中的正式巨潮知识库。`scripts/ingest_docs.py` 用于处理 `data/raw_docs` 中的临时上传或示例文件，不要在未备份正式 `kb_chunks.jsonl` 时直接运行。

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

由于 Windows OneDrive + 中文长路径下 Chroma/SQLite 可能出现 `disk I/O error`，当前把 Chroma 向量库放在系统临时目录的英文短路径：

```yaml
chroma_persist_dir: C:/Users/MIRACL~1/AppData/Local/Temp/finance_rag_chroma
```

## 构建索引

```bash
python scripts/build_bm25_index.py \
  --chunks data/processed/kb_chunks.jsonl \
  --output data/indexes/bm25.pkl

python scripts/build_dense_index.py \
  --chunks data/processed/kb_chunks.jsonl \
  --persist-dir C:/Users/MIRACL~1/AppData/Local/Temp/finance_rag_chroma \
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

命令行还会显示本次实际生成后端，例如：

```text
生成后端：deepseek / deepseek-v4-pro / openai_sdk / fallback=False
request_id：...
```

当 `strict: true` 时，API Key 缺失或模型调用失败会直接报错，禁止将 mock 模板误记成真实模型结果。

## 启动 Streamlit Demo

```bash
streamlit run demo/app.py
```

页面支持上传文档、重建索引、选择检索模式、启用/关闭 reranker、查看回答、引用来源和检索排序。

## 自动评测

运行检索评测：

```bash
python scripts/evaluate_retrieval.py
```

在双模型逐题明细生成后，分别评估统一的 `hybrid_reranker` 设置：

```powershell
python scripts/evaluate_rag.py `
  --qa-results outputs/ablation_details_qwen.jsonl `
  --setting hybrid_reranker `
  --json-output outputs/eval_report_qwen.json `
  --md-output outputs/eval_report_qwen.md

python scripts/evaluate_rag.py `
  --qa-results outputs/ablation_details_deepseek.jsonl `
  --setting hybrid_reranker `
  --json-output outputs/eval_report_deepseek.json `
  --md-output outputs/eval_report_deepseek.md
```

报告输出到：

- `outputs/retrieval_report.md`
- `outputs/eval_report_qwen.*`
- `outputs/eval_report_deepseek.*`

`scripts/make_sample_eval.py` 只用于创建六题示例集，会覆盖现有正式评测文件，不要在正式实验前运行。

## 如何接入训练得到的 QLoRA 模型

系统支持临时切换到本地 QLoRA。无需修改默认配置，可使用专用命令：

```powershell
python scripts/ask_local_qlora.py `
  --question "公司净利润增长的主要原因是什么？" `
  --retrieval-mode hybrid `
  --use-reranker true
```

本地模型配置为：

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

如果 adapter 路径不存在、模型未缓存、显存不足或依赖不完整，strict 模式会直接报错，不会把 mock 模板答案伪装成本地模型结果。

当前问答记录会保存实际生成信息，而不是仅根据初始化配置推断模型：

```json
{
  "actual_provider": "deepseek",
  "actual_model": "deepseek-v4-pro",
  "backend": "openai_sdk",
  "fallback_used": false,
  "request_id": "...",
  "prompt_tokens": 2300,
  "completion_tokens": 135,
  "total_tokens": 2435,
  "finish_reason": "stop"
}
```

如果显式关闭 strict 模式并发生 fallback，记录中的 `actual_provider` 和 `actual_model` 会变为 `mock`，同时写入错误类型和错误信息。

## 本地 Qwen 与 DeepSeek V4 Pro 实测

实测时间：2026-06-24。两组使用完全相同的检索结果：

```text
问题：启迪设计2024年实现营业收入和同比变动是多少？
检索：Hybrid Search + BGE reranker
top_k：30
final_top_n：5
```

| 项目 | 本地 Qwen3-4B + QLoRA | DeepSeek V4 Pro |
|---|---|---|
| 实际 provider | `local_qlora` | `deepseek` |
| 实际 backend | `transformers+peft` | `openai_sdk` |
| fallback | `false` | `false` |
| 回答结论 | 正确提取 `118,479.29 万元`、同比下降 `4.46%` | 正确提取 `118,479.29 万元`、同比下降 `4.46%` |
| 风格 | 原文复制较多，回答偏长 | 结论更直接、结构更清晰 |
| 生成耗时 | `27.2539s` | `2.5842s` |
| 端到端耗时 | `29.4794s` | `4.7672s` |
| Token 用量 | 本地推理未统计 | prompt `2300`、completion `135`、total `2435` |

两组命中的首条证据相同：

```text
文件：300500_1767170891000_1224913114.pdf
页码：1-2
chunk_id：eefe2ec812f2c439
```

本次单题结果表明：两种模型都能基于正确证据回答；DeepSeek V4 Pro 在该问题上更简洁且延迟更低，本地 QLoRA 的优势是数据不离开本机。该结果只是单题对照，不代表完整评测结论。

## 对比实验设计

| 实验组 | 检索 | 生成模型 | 目的 |
|---|---|---|---|
| Base Model | 无 | Qwen/Qwen3-4B | 观察基础模型直接回答能力 |
| QLoRA Only | 无 | Qwen3-4B + 金融 QLoRA adapter | 观察领域微调后的直接回答能力 |
| RAG + Base | BM25 + Dense + Hybrid | Qwen/Qwen3-4B | 观察外部证据对基础模型的增益 |
| RAG + QLoRA | BM25 + Dense + Hybrid | Qwen3-4B + 金融 QLoRA adapter | 观察检索证据和领域微调叠加后的效果 |
| RAG + DeepSeek | BM25 + Dense + Hybrid + reranker | DeepSeek V4 Pro | 对比云端模型的回答质量、延迟和 token 成本 |

建议统一使用同一批 `eval_questions.jsonl`，对比格式遵循率、引用存在率、拒答准确率、关键词覆盖率和证据命中率。

## outputs 展示产物

每次调用 `RAGPipeline.ask()` 后，系统会自动追加保存：

- `outputs/qa_results.jsonl`：问题、回答、引用来源、召回片段，以及实际 provider/model/backend、request_id、token、fallback 和耗时。
- `outputs/retrieved_contexts.csv`：每个问题召回的 Top-K 文档片段，使用 `utf-8-sig`，方便 Excel 打开。
- `outputs/eval_report_qwen.*` / `outputs/eval_report_deepseek.*`：两个模型在 `hybrid_reranker` 设置下的规则评估。
- `outputs/ablation_results_qwen.csv` / `outputs/ablation_results_deepseek.csv`：双模型三组消融摘要。
- `outputs/ablation_details_qwen.jsonl` / `outputs/ablation_details_deepseek.jsonl`：各 120 条逐题回答、引用和调用审计。
- `outputs/latency_report_qwen.csv` / `outputs/latency_report_deepseek.csv`：逐题端到端延迟。
- `outputs/citation_report_qwen.html` / `outputs/citation_report_deepseek.html`：双模型引用溯源页面。

本轮对两个生成模型使用完全相同的三组检索设置：

- `dense`
- `dense_reranker`
- `hybrid_reranker`

Qwen 实验命令：

```powershell
python scripts/build_ablation_results.py `
  --generator qwen `
  --settings dense,dense_reranker,hybrid_reranker `
  --top-k 30 `
  --final-top-n 5 `
  --max-tokens 512 `
  --output outputs/ablation_results_qwen.csv `
  --details-output outputs/ablation_details_qwen.jsonl `
  --latency-output outputs/latency_report_qwen.csv
```

DeepSeek 实验命令：

```powershell
python scripts/build_ablation_results.py `
  --generator deepseek `
  --settings dense,dense_reranker,hybrid_reranker `
  --top-k 30 `
  --final-top-n 5 `
  --max-tokens 512 `
  --output outputs/ablation_results_deepseek.csv `
  --details-output outputs/ablation_details_deepseek.jsonl `
  --latency-output outputs/latency_report_deepseek.csv
```

当前真实模型配置：

| 组件 | 当前配置 |
|---|---|
| Dense embedding | `BAAI/bge-m3`，backend=`sentence_transformers` |
| Reranker | `D:\models\bge-reranker-v2-m3`，backend=`FlagEmbedding` |
| 本地生成 | `Qwen/Qwen3-4B + finance QLoRA`，4-bit |
| API 生成 | `deepseek-v4-pro`，非思考模式 |
| Chroma 持久化目录 | `C:/Users/MIRACL~1/AppData/Local/Temp/finance_rag_chroma` |
| 知识库 chunks | `data/processed/kb_chunks.jsonl`，共 6,777 条 |
| fallback 策略 | embedding、reranker、LLM 均启用 strict，禁止静默 fallback |

## 双模型全量消融结果

实测时间：2026-06-24。每个模型运行 40 题 × 3 组设置，共 120 条回答。

### Qwen3-4B + QLoRA

| setting | 检索/重排 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency |
|---|---|---:|---:|---:|---:|---:|
| dense | Dense | 0.4708 | 0.4708 | 0.3000 | 0.7417 | 31.6880s |
| dense_reranker | Dense + reranker | 0.4792 | 0.4792 | 0.3650 | 0.8938 | 32.1128s |
| hybrid_reranker | BM25 + Dense + RRF + reranker | 0.5625 | 0.5625 | 0.3650 | 0.8938 | 28.4964s |

### DeepSeek V4 Pro

| setting | 检索/重排 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency |
|---|---|---:|---:|---:|---:|---:|
| dense | Dense | 0.6750 | 0.7021 | 0.3000 | 0.7167 | 4.1142s |
| dense_reranker | Dense + reranker | 0.7646 | 0.7729 | 0.3650 | 0.8688 | 4.7683s |
| hybrid_reranker | BM25 + Dense + RRF + reranker | 0.7729 | 0.7812 | 0.3650 | 0.8438 | 4.4654s |

两组共 240 条回答，fallback 均为 0。DeepSeek 的 120 条记录均包含 request_id，共消耗 prompt 214,775 tokens、completion 30,381 tokens、total 245,156 tokens；其中 3 条回答因达到 512 token 上限而以 `finish_reason=length` 结束。Qwen 为本地推理，不产生 API token 费用。

## 检索评测结果

说明：当前检索链路已切换为 `finance_rag` 环境中的真实模型：`BAAI/bge-m3` 通过 `sentence_transformers` 加载，reranker 使用本地 `D:\models\bge-reranker-v2-m3` 并通过 `FlagEmbedding` 加载；Chroma 已迁移到非 OneDrive 的英文短路径，避免 `disk I/O error`。

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.2917 | 0.4167 | 0.5278 | 0.7222 | 0.4473 |
| Dense | 0.4028 | 0.5278 | 0.6111 | 0.7778 | 0.5368 |
| Hybrid | 0.4861 | 0.6944 | 0.7778 | 0.8750 | 0.6510 |
| Hybrid + Reranker | 0.5694 | 0.8333 | 0.8750 | 0.9167 | 0.7378 |

## 双模型 RAG 评估

以下报告只比较两个模型在最佳统一设置 `hybrid_reranker` 下的 40 个问题：

| 指标 | Qwen3-4B + QLoRA | DeepSeek V4 Pro |
|---|---:|---:|
| 问题总数 | 40 | 40 |
| 平均回答长度 | 948.425 | 483.125 |
| 平均引用数量 | 4.5 | 4.5 |
| 无引用回答比例 | 0.1 | 0.1 |
| 平均延迟秒数 | 28.4964 | 4.4654 |
| possible_unfaithful 数量 | 17 | 13 |

DeepSeek 在本轮规则指标和延迟上整体领先，平均端到端延迟约为 Qwen 的 1/6.38；Qwen 的主要问题是回答偏长、复制证据和格式损坏。注意这些指标是答案要点和字符串规则，不是 RAGAS，也可能低估单位换算、同义表达和语义拒答。

## Bad Case 分析

完整分析见 `outputs/bad_cases.md`，当前 5 个真实 bad case 概览如下：

| Case | 问题类型 | 失败现象 | 初步原因 | 后续处理 |
|---|---|---|---|---|
| 1 | 主体检索 | 三花智控未召回，Qwen使用其他公司数字回答 | 缺少公司名和公告类型过滤 | 增加主体硬过滤 |
| 2 | 原因分析 | 天赐材料召回到其他公司 | 原因类语义宽泛 | 主体过滤 + query rewrite |
| 3 | 生成格式 | Qwen 对减值问题输出错误事件类型和损坏 JSON | QLoRA 模板与 RAG Prompt 不一致 | 结构化输出校验 |
| 4 | 财务表格 | 第四季度数据无法稳定提取 | 需要表格解析和差额计算 | 建立字段级索引 |
| 5 | 投资建议 | 两个模型拒答措辞均不完全稳定 | 当前只清空引用，仍调用 LLM | Pipeline 硬短路拒答 |

## 后续扩展

1. 增加公司名、股票代码和公告类型的检索硬过滤。
2. 对财务表格做结构化抽取和季度差额计算。
3. 将投资建议问题改为 Pipeline 级直接拒答。
4. 接入 RAGAS 或人工标注评测，替代部分字符串规则。
