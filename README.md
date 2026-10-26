# project-agent

面向 AI 客服产品的竞品调研系统。从自己的 L5 LangGraph 学习项目开始，逐步补齐原文证据、事实核验、定向补查和可靠执行。

## 当前阶段

目前是独立运行基线：研究、写作、评审、人工确认、SQLite 审批恢复和多轮追问已存在。网页正文证据链、独立 Verifier、竞品案例集和服务化属于后续开发任务。

现有源码暂放在 `l5-langgraph/`，沿用熟悉的目录结构。

## 本机离线运行

```powershell
Set-Location 'C:\Users\13077\Desktop\project-agent\l5-langgraph'
.\.venv\Scripts\python.exe -B main.py --demo --stream
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m evaluations.run_evals --demo --limit 1
.\.venv\Scripts\python.exe -B -m evaluations.run_conversation_evals --demo
```

新环境使用 `python -m venv .venv` 建立环境，再安装 `requirements-lock.txt`。密钥通过环境变量提供；离线演示与测试无需密钥。

## 独立仓库迁移

- 八类兼容测试改为本仓库内的行为断言，保留首次通过、修改后通过、次数上限、连续问题、超时、拒绝、空输入和 trace 场景。它们不再导入相邻的旧版测试源码。
- 评测默认运行 LangGraph 后端；原版对照需要另外提供旧版目录，例如 `--backend both --legacy-dir 'D:\path\to\l5-multi-agent'`。
- 历史评测说明保留在 `l5-langgraph/docs/evals.md`，其中结果属于迁移前记录，不代表当前独立仓库的新评测。
- `.venv`、数据库、运行产物和密钥文件通过 `.gitignore` 排除；本机已有数据可继续保留。

## 下一步

先准备 10 个竞品调研案例，再实现网页正文读取、证据保存与原文定位；之后加入 Verifier 和定向补查。各阶段同时验收功能、故障行为与个人对核心代码的理解。

## 基线验证

迁移验证包含 51 项离线测试（43 项原有测试、8 项独立编写的兼容场景），两个离线评测入口，以及缺失旧版目录时的明确报错。演示产物不代表真实模型内容质量，未运行付费在线评测。
