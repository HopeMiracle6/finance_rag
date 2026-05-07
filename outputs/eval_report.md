# RAG 本地评估报告

| 指标 | 数值 |
|---|---:|
| 问题总数 | 26 |
| 平均回答长度 | 538.9231 |
| 平均引用数量 | 3.6154 |
| 无引用回答比例 | 0.2308 |
| 平均延迟秒数 | 20.3784 |
| possible_unfaithful 数量 | 20 |

## 按问题类型统计

| 类型 | 数量 | 平均引用数量 | 平均延迟秒数 | possible_unfaithful |
|---|---:|---:|---:|---:|
| fact_qa | 8 | 4.25 | 15.8419 | 8 |
| analysis | 6 | 5.0 | 16.5996 | 6 |
| risk_summary | 6 | 5.0 | 30.121 | 6 |
| investment_advice | 6 | 0.0 | 20.4631 | 0 |

> 当前为本地规则评估；后续可接入 RAGAS 计算 faithfulness、answer_relevancy、context_precision 等指标。
