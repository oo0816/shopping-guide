# shopping-guide · 有据选购

基于 RAG 与 LangGraph 的同品类导购系统。将用户需求转为 SKU 硬条件筛选，检索商品证据，逐项审核适配程度，展示推荐理由、限制条件和缺失信息。

## 已实现

- 多轮需求更新、商品指代和澄清。
- 预算、品牌和品类筛选，逐项适配审核及商品/SKU引用范围保护。
- LangGraph 节点编排、SQLite检查点、暂停恢复及失败步骤重试。
- 本地聊天网页，在线模型调用需显式允许。
- 可选接入 Go 电商接口，以实时 SKU 价格、库存和状态筛选候选。

电商集成对应 [xzxg-shop fork](https://github.com/oo0816/xzxg-shop/tree/feat/shopping-integration)，底座来自 [LYP-leo/xzxg-shop](https://github.com/LYP-leo/xzxg-shop)。此仓库维护 Python 导购、证据审核与评测；当前尚未完成账号绑定、购物车和订单闭环。

## 环境与启动

在仓库根目录运行（本地验证环境为 Windows / Python 3.13）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests
```

本地已有数据准备完成后启动网页：

```powershell
.\.venv\Scripts\python.exe -X utf8 server.py
```

打开 `http://127.0.0.1:8000`。密钥按 `.env.example` 在本机配置，真实密钥不提交。

**新克隆环境不包含本机完整商品资料、向量索引、模型缓存和会话数据库。** 测试使用隔离夹具，可以离线执行；完整演示需要自行准备有权使用的商品JSON，通过 `scripts/import_products.py --source <资料目录>` 导入。向量构建与模型/Embedding检查脚本位于 `scripts/`，在线调用可能计费；不要把新克隆环境当作已有缓存的演示环境。

## 可选实时商品模式

先运行匹配版本的 Go 电商后端（8080），再在此仓库的 PowerShell 中运行：

```powershell
$env:SHOP_CATALOG_URL = 'http://127.0.0.1:8080/api/v1'
.\.venv\Scripts\python.exe -X utf8 -B -m uvicorn app.api:app --host 127.0.0.1 --port 8001
```

此模式仍需本地证据资料与已匹配的商品/SKU身份。Go接口失败时不回退历史价格，身份或规格变化时要求同步资料。查询结果不是库存预留或有效交易报价，历史会话快照仍需在交易前重新校验。

## 验证与文档

2026-10-26：157项Python离线测试通过；本机与Go接口实际联调覆盖294个商品、973个SKU。该结果证明接入和已定义行为，不代表推荐准确率或生产负载能力。

- [LangGraph流程与恢复](docs/langgraph_workflow.md)
- [网页入口](docs/web_entry.md)
- [评测说明](evaluations/README.md)
- [适配审核验证](evaluations/fit_review_findings.md)

文档中的历史测试数量和开发集结果按各自日期理解，不作为独立盲测成绩。完整集成说明见电商fork的 `docs/shopping_integration.md`。
