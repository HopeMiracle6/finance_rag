# outputs 展示目录

本目录保存中文金融公告 RAG 系统的可复现实验产物。

## 2026-06-24 双模型实验

Qwen3-4B + QLoRA 与 DeepSeek V4 Pro 使用同一批 40 个问题，分别运行以下三组设置：

- `dense`
- `dense_reranker`
- `hybrid_reranker`

每个模型共生成 120 条回答。两组均启用 strict 模式，本轮 fallback 数量均为 0。

## 文件说明

- `sample_questions.jsonl`：40 个评测问题及答案要点。
- `retrieval_report.md`：不经过生成模型的 BM25、Dense、Hybrid、Hybrid + reranker 检索指标。
- `ablation_results_qwen.csv` / `ablation_results_deepseek.csv`：两种模型的三组消融摘要。
- `ablation_details_qwen.jsonl` / `ablation_details_deepseek.jsonl`：逐题回答、引用、调用审计和规则指标。
- `latency_report_qwen.csv` / `latency_report_deepseek.csv`：逐题端到端延迟。
- `eval_report_qwen.json` / `eval_report_qwen.md`：Qwen 在 `hybrid_reranker` 设置下的 40 题规则评估。
- `eval_report_deepseek.json` / `eval_report_deepseek.md`：DeepSeek 在 `hybrid_reranker` 设置下的 40 题规则评估。
- `citation_report_qwen.html` / `citation_report_deepseek.html`：120 条回答的引用可视化。
- `bad_cases.md`：基于本轮双模型结果整理的典型失败案例。
- `qa_results.jsonl` / `retrieved_contexts.csv`：交互式问答持续追加的运行记录，默认不提交。
- `screenshots/`：Streamlit 截图目录，目前仍待补充真实截图。

DeepSeek 的 120 条记录均包含 request_id，总 token 为 245,156；本地 Qwen 不产生 API token 费用。
