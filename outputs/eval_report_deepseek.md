# RAG 本地评估报告

| 指标 | 数值 |
|---|---:|
| 问题总数 | 40 |
| 平均回答长度 | 483.125 |
| 平均引用数量 | 4.5 |
| 无引用回答比例 | 0.1 |
| 平均延迟秒数 | 4.4654 |
| possible_unfaithful 数量 | 13 |

## 按问题类型统计

| 类型 | 数量 | 平均引用数量 | 平均延迟秒数 | possible_unfaithful |
|---|---:|---:|---:|---:|
| fact_qa | 15 | 5.0 | 4.3197 | 8 |
| analysis | 11 | 5.0 | 4.6761 | 2 |
| risk_summary | 10 | 5.0 | 4.8982 | 3 |
| investment_advice | 4 | 0.0 | 3.3501 | 0 |

> 当前为本地规则评估；后续可接入 RAGAS 计算 faithfulness、answer_relevancy、context_precision 等指标。
