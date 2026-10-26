# 浏览器聊天入口

## 启动

在 `C:\Users\13077\Desktop\project-agent\shopping-agent` 目录打开 PowerShell，运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 server.py
```

打开 <http://127.0.0.1:8000>。服务只监听本机；运行窗口按 Ctrl+C 停止。新环境先创建 `.venv`，再使用其中的 Python 安装 `requirements-lock.txt`。原 `main.py` 和 `graph_main.py` 继续可用。

## 怎么用

1. 点击示例，或直接输入预算、商品类型和用途，再发送。
2. 默认只复用缓存。新问题缺缓存时会保存失败进度；需要调用模型时，勾选“允许在线模型调用”，点击“重试未完成的步骤”。勾选会允许本轮请求 LLM 和问题 Embedding，可能产生费用。
3. 信息不全时直接补充，例如“买跑步鞋。”；无商品时可明确修改预算，不会自动放宽条件。
4. 商品卡展示的是筛选后的具体 SKU；可展开“查看依据”读取摘录。有限制和待确认的偏好会单独展示。
5. 可以继续问“第二双怎么样”等问题。新追问通常需要在线解析，不代表之前回答失效。
6. 刷新页面自动恢复当前会话。浏览器只记住会话编号，历史从 SQLite 读取，页面显示最近6个已提交轮次及当前未完成轮次。也可以手动输入编号恢复已有 LangGraph 会话。
7. “开始新的选购”生成新编号，不删除旧记录。失败未完成时，先重试或新开选购，避免覆盖待执行步骤。

旧 `main.py --session-id` 的会话库与 LangGraph 库独立；网页不会自动读取旧入口的会话。网页直接使用现有 LangGraph 库，因此可继续 `graph_main.py` 创建的会话。

## 文件和接口

| 文件 | 用途 |
| --- | --- |
| `server.py` | 本机 Uvicorn 启动入口 |
| `app/api.py` | FastAPI 路由、参数校验、浏览器响应整理 |
| `web/index.html` | 页面结构 |
| `web/style.css` | 桌面和窄屏布局 |
| `web/app.js` | 发送、重试、恢复及引用展示，无前端构建依赖 |
| `tests/test_api.py` | 隔离数据库的 HTTP 集成测试 |
| `scripts/check_web_browser.cjs` | 可选真实浏览器验收，使用 Node 原生 CDP，不依赖 npm 包 |

- `GET /api/health`：健康检查，不调用模型。
- `POST /api/chat`：`{"session_id":"demo","query":"买跑步鞋","online":false,"method":"vector"}`。
- `POST /api/retry`：`{"session_id":"demo","online":false}`，只重试失败步骤。
- `GET /api/sessions/{session_id}`：只读当前结果和最近历史。
- `/docs`：FastAPI 接口说明；默认 Swagger 资源来自 CDN，需要联网。聊天页面资源均在本地。

业务状态（澄清、无商品、缺缓存、审核失败）在 HTTP 200 的 `status` 中返回，便于保留和展示进度。参数格式错误为422，会话执行冲突为409，存储不可用为503。在线权限默认为false，未知请求字段拒绝；密钥仍只在服务端 `.env`，不由浏览器接收或提交。

网页不展示 checkpoint 原文、模型原始审核输出和本地资料路径。商品名称、模型文字及引用使用 `textContent` 渲染，避免当作 HTML 执行。服务限制本机 Host 和同源页面；当前是单用户本地入口，还没有账号、服务端费用额度、请求队列或公网部署配置。页面显示请求处理状态，不提供实时 token 流。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -q
# 先启动服务，再运行以下可选验收（需要 Node 22+、Chrome 或 Edge）：
node scripts/check_web_browser.cjs
```

149项离线测试通过，其中新增12项 HTTP 测试，覆盖真实商品夹具、历史隔离、澄清补充、失败重试、默认不在线、输入校验和浏览器边界。

真实无头 Chrome 已验证：缓存商品选购、澄清后继续、刷新恢复、手动恢复编号、桌面及390px窄屏、输入框不遮挡聊天、离线缺缓存后的显式重试和 HTML 文本安全展示。所有浏览器业务请求均为 `online=false`，没有调用付费模型 API。验收记录及截图在 `runs/web_browser_check.json`、`runs/web_desktop.png`、`runs/web_mobile.png`。

这些是入口和迁移验证，不是推荐准确率或生产负载验收。
