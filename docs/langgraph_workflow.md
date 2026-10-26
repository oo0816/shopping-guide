# LangGraph 并行实现

旧入口 `main.py` 保留。新入口 `graph_main.py` 使用真正的 StateGraph 节点，复用已有模型客户端、缓存、预算筛选、检索和引用护栏。没有把完整的旧 `run()` 包成一个节点，也没有改成多个自由对话 Agent。

## 运行

在 `shopping-agent` 根目录执行。独立环境已经建立；新机器可运行 `python -m venv .venv`，再执行 `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`。当前验证环境为 Windows / Python 3.13，直接依赖见 `requirements.txt`，完整版本见 `requirements-lock.txt`。

首轮选购（此例已有缓存，未加 `--online`）：

```powershell
.\.venv\Scripts\python.exe -X utf8 graph_main.py "预算1000元，想买双跑步鞋用于夏天通勤，希望轻便一些，品牌不限。" --session-id shoes-graph --stream
```

新问题或新追问缺缓存时，需要显式加 `--online` 才允许请求模型或问题向量。不会重新计算所有商品向量。`--method keyword` 切换关键词检索；`--fit-review rules` 保留规则基线，默认仍为 semantic。

澄清后恢复：

```powershell
.\.venv\Scripts\python.exe -X utf8 graph_main.py "预算1000元，希望轻便，适合夏天通勤。" --session-id clarify-graph --stream
.\.venv\Scripts\python.exe -X utf8 graph_main.py "买跑步鞋。" --session-id clarify-graph --stream
.\.venv\Scripts\python.exe -X utf8 graph_main.py --session-id clarify-graph --state --json
```

三条命令可以分别启动进程。第一条暂停，第二条通过 `Command(resume=...)` 恢复，第三条只查看状态。完成后可用同一 ID 继续追问。无商品也会暂停，等用户明确调整条件，不自行提高预算或换品牌。

失败重试：

```powershell
.\.venv\Scripts\python.exe -X utf8 graph_main.py --session-id shoes-graph --retry
# 若失败原因是缺缓存，并且准备允许请求模型：
.\.venv\Scripts\python.exe -X utf8 graph_main.py --session-id shoes-graph --retry --online --stream
```

`--retry` 只用于失败且未完成的节点，不能同时提交新问题。等待澄清时直接输入补充文本。存在失败节点时拒绝用新问题覆盖；想独立开始可换一个 ID。重试沿用原来的检索方式和审核模式，不能通过切换参数绕过待完成的语义审核。

`--stream` 将节点完成/暂停事件写入 stderr；`--json` 的 stdout 仍是完整 JSON。这里的 stream 是流程进度，不是模型 token 流。

## 模块和节点

| 图节点/状态 | 复用的代码 | 职责 |
| --- | --- | --- |
| `ShoppingState` | `app/graph_workflow.py` | 当前问题、结构化需求、候选、证据、审核结果、会话记忆和近6轮记录 |
| `requirements` | `NeedsExtractor`、`DialogueExtractor`、`apply_plan` | 首轮提取或增量更新；保留原文依据和指代检查 |
| `product_search` | `find_candidates`、`app/catalog.py` | 按实际 SKU 预算、品牌、类型筛选 |
| `evidence_retrieval` | `ShoppingAssistant.retrieve_candidates` | 共享原检索、规则分组、最多3个候选和选定 SKU 事实 |
| `fit_review` | `FitReviewer`、`apply_review` | 偏好逐项审核和范围保护；未通过不进入输出 |
| `recommendation` | `review_answer` 或 `ShoppingAssistant.answer_candidates` | 从已校验矩阵生成说明；规则/详情分支继续使用原 RAG 护栏 |
| `human_input` | LangGraph `interrupt()` | 缺信息、指代不清、无法验证硬条件或无候选时暂停；用户回复后回到需求解析 |
| `commit_turn` | 原会话状态字段结构 | 完成一轮后保存需求、实际卡片顺序、焦点、相对价格约束和简短历史 |

条件边决定需求是否需要澄清、筛选是否有商品，以及是否需要语义审核。当前没有“审核失败后自由改写直到通过”的循环：无效引用或审核失败会阻止输出，保留失败节点供修复后重试。有限制/未知是合法审核结果，可输出部分结论和缺失项。

原 `ShoppingAssistant.run()` 继续调用共享的检索和回答方法；这两处只做方法提取，原提示词、筛选规则和模型缓存键保持一致。其余业务模块和原始数据未搬动。

## 持久化和重试边界

- 新 SQLite checkpointer：`data/processed/shopping_graph_checkpoints.sqlite`，不写旧 `shopping_sessions.sqlite`。旧 ID 的历史仍通过旧入口读取；新图不会自动导入旧会话，即使 ID 相同也相互独立。
- checkpointer 保存的是节点边界的完整状态；不是答案缓存，也不替代向量索引。模型密钥不放入 State。原需求、回答、审核、Embedding 缓存继续复用。
- 暂停时 `checkpoint_saved=true`、`state_saved=false`：图进度已落盘，但该业务轮尚未完成提交。收到补充后，先记录这次澄清/空结果，再继续新一轮。
- 同一图会话使用进程文件锁，拒绝并发覆盖。不同会话仍各自独立；这是本地 SQLite 实现，尚未迁移为服务端多实例存储。
- 只有不调用模型的本地 `product_search` 遇到 SQLite busy/locked 才自动尝试第二次。可能调用模型的节点不自动重试，避免 API 成功而缓存写失败时再次计费。
- 手动重试保留已完成节点；失败节点本身从头执行，尽量命中原缓存。一个节点内若已发生部分 API 请求，不承诺严格一次计费。无效输出不会静默降级成成功。
- 检查点保留当时商品/证据快照，不代表实时价格库存；资料或程序大幅变更后应开始新会话。尚未增加 checkpoint 清理策略或 Web 服务。

LangGraph 机制参考：[interrupt 与恢复](https://docs.langchain.com/oss/python/langgraph/interrupts)、[持久化](https://docs.langchain.com/oss/python/langgraph/persistence)。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -X utf8 -B -m evaluations.run_graph_evals
```

原120项加新增17项，共137项离线测试通过。新增覆盖新旧结果一致、真实节点事件、SQLite 重启读取、澄清与空结果恢复、焦点详情、会话隔离、并发保护、审核阻断、失败步骤重试及已完成步骤不重复执行。

缓存对照使用现有 B01、B18，共2组3轮，比较结构化需求、卡片、审核说明、澄清问题和焦点/相对价格约束，新旧一致且新增 LLM/Embedding 请求均为0。它衡量迁移一致性，不是新的推荐质量评测，也不是完整25组业务回归。其他用例可加 `--case-id`，缺缓存会记为未运行，不会偷偷改成在线。

另已用两个独立 CLI 进程验证真实商品“缺类型 → 补充跑步鞋”恢复，第三个进程查看到第2轮、partial、无待执行节点。记录位于 `runs/graph_cli_resume_20261025.json`。这些验证未调用付费 API，也未重建商品索引。
