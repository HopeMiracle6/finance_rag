# RAG 本地评估报告

| 指标 | 数值 |
|---|---:|
| 问题总数 | 40 |
| 平均回答长度 | 948.425 |
| 平均引用数量 | 4.5 |
| 无引用回答比例 | 0.1 |
| 平均延迟秒数 | 28.4964 |
| possible_unfaithful 数量 | 17 |

## 按问题类型统计

| 类型 | 数量 | 平均引用数量 | 平均延迟秒数 | possible_unfaithful |
|---|---:|---:|---:|---:|
| fact_qa | 15 | 5.0 | 30.3107 | 9 |
| analysis | 11 | 5.0 | 24.9436 | 4 |
| risk_summary | 10 | 5.0 | 25.2219 | 4 |
| investment_advice | 4 | 0.0 | 39.6491 | 0 |

> 当前为本地规则评估；后续可接入 RAGAS 计算 faithfulness、answer_relevancy、context_precision 等指标。
