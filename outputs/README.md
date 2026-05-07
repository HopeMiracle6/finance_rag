# outputs 展示目录

这里保存中文金融公告/研报 RAG 问答与引用溯源系统的运行产物，便于复现实验和面试展示。

## 文件说明

- `demo_cases.md`：可演示的典型问答案例。
- `sample_questions.jsonl`：评估用样例问题，包含事实问答、摘要、风险提示和拒答类问题。
- `qa_results.jsonl`：每次 RAG 问答自动追加保存的完整结果。
- `retrieved_contexts.csv`：每次问答召回的 Top-K 文档片段。
- `citation_report.html`：由问答结果生成的引用溯源可视化报告。
- `eval_report.json` / `eval_report.md`：本地规则评估结果。
- `ablation_results.csv`：演示版消融实验结果。
- `latency_report.csv`：问答耗时记录。
- `screenshots/`：Streamlit Demo 截图放置目录。
- `logs/`：运行日志放置目录。

当前目录中的样例内容用于展示格式；真实运行后，`qa_results.jsonl` 和 `retrieved_contexts.csv` 会持续追加新记录。
