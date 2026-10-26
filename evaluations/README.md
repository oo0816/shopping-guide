# 业务评测

business_cases.jsonl 包含25组、52轮开发用业务对话。预算、品牌、偏好保留、商品指代、切换需求、空结果、异常恢复和会话隔离分别有明确断言。

在 shopping-agent 目录运行缓存重放：

    python -X utf8 -B -m evaluations.run_business_evals --mode cached

需要补齐真实模型缓存时显式开启在线模式，并设置调用上限：

    python -X utf8 -B -m evaluations.run_business_evals --mode online --max-llm-calls 60 --max-embedding-calls 30

只重测一个场景：

    python -X utf8 -B -m evaluations.run_business_evals --mode cached --case-id B18

默认只使用真实模型缓存，不发送 API 请求。在线模式也优先复用缓存，失败不自动付费重试。调用限额同时覆盖需求解析、上下文解析、回答生成和查询向量。会话存入临时数据库，保留现有用户会话。

每次输出 runs/business_模式_时间.json 和同名 Markdown。JSON 包含各轮实际输出、断言、模型公开配置、数据和代码哈希、API次数及在线原始JSON回答。Markdown列出失败和待人工复核的理由与摘录。没有使用预置正确模型答案冒充实际模型输出。

报告区分已通过、失败、缓存缺失、调用预算耗尽。完整通过率以全部选中用例为分母；只看已执行用例的通过率时必须同时查看执行覆盖率。代码退出状态：0表示全部通过，1表示存在失败，2表示没有失败但有未执行场景。

B24明确注入一次上下文网络失败，验证状态不被污染和随后重试，不是对供应商真实故障率的测量。每个场景首个失败后停止后续轮次，避免错误状态造成级联误判。

自动结果只证明已标注的状态、条件、SKU和引用约束。推荐是否值得采纳、引用是否真正支持结论、参数限定是否适用于具体SKU，需要另外复核。该集用于开发，不是独立最终测试集。

首次实际运行结果及问题见 business_findings.md。


用途排序修复后，相对价格用例不再假定固定序号总是较贵或较便宜；根据实际展示SKU和独立商品表检查替代是否存在。原基线结果仍保留。四组定向复测可用 --case-id B01 --case-id B14 --case-id B15 --case-id B18 选择，默认cached模式不产生新调用。


## 逐项语义审核专项

    python -X utf8 -B -m evaluations.run_fit_evals
    python -X utf8 -B -m evaluations.run_fit_evals --online --max-llm-calls 14

默认只读实际模型缓存。fit_review_cases.jsonl为14条受控证据，分别统计规则、原始模型和程序保护后的状态；不是额外的真实商品资料或独立最终测试集。业务用例检查新审核需指定--fit-review semantic，业务运行器默认rules保留历史基线。详见fit_review_findings.md。


## LangGraph 迁移一致性

    .\.venv\Scripts\python.exe -X utf8 -B -m evaluations.run_graph_evals

默认对照B01、B18共2组3轮，分别运行旧会话流程与新图，比较需求、卡片、带引用说明及澄清/焦点/相对价格条件。新增请求额度固定为0；缺缓存记not_run_cache，绝不转在线。可重复 --case-id 选择其他用例；故障注入用例另记未运行，本运行器不代替完整业务评测。

结果保存在 runs/graph_cached_*.json，含版本、代码摘要、缓存缺失和差异项。最新默认对照2组通过，LLM及Embedding新增调用均为0。它衡量迁移一致性，不是推荐准确率。原业务和语义专项运行器保持原用法。
