"use strict";
const $ = (id) => document.getElementById(id);
const state = { session: "", busy: false, latest: null };
const statusLabels = { completed: "这些商品值得比较", partial: "有匹配点，也有需要确认的地方", insufficient_evidence: "找到候选，但推荐依据还不充分", needs_clarification: "先补充一点信息", no_products: "当前条件下没有找到商品", generation_disabled: "这一步需要新的模型调用", generation_failed: "暂时没能完成这一步", invalid_answer: "这次回答未通过检查", failed: "流程暂未完成" };
const fitLabels = { supported: "支持", limited: "有限制", unknown: "待确认" };
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}
function newId() { return "web-" + crypto.randomUUID(); }
function remember(id) {
  state.session = id;
  $("session-id").value = id;
  try { localStorage.setItem("shopping.session", id); } catch (_) { /* Storage is optional. */ }
}
function notice(text, isError = false) {
  $("notice").textContent = text;
  $("notice").hidden = !text;
  $("notice").classList.toggle("error", isError);
}
function controls() {
  const blocked = Boolean(state.latest?.resumable && !state.latest?.paused);
  $("send").disabled = state.busy || blocked;
  $("send").textContent = state.busy ? "处理中…" : "发送 ↑";
  $("query").disabled = state.busy || blocked;
  $("retry").hidden = !blocked;
  $("retry").disabled = state.busy;
  $("new-session").disabled = state.busy;
  $("session-id").disabled = state.busy;
  $("restore-form").querySelector("button").disabled = state.busy;
  $("online").disabled = state.busy;
  document.querySelectorAll("[data-example]").forEach((button) => { button.disabled = state.busy; });
  $("connection").textContent = state.busy ? "正在处理，请稍等…" : state.latest?.paused ? "等待你补充条件" : blocked ? "进度已保存，可以重试" : "会话保存在本机";
}
async function request(path, body) {
  const response = await fetch(path, body === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  let data;
  try { data = await response.json(); } catch (_) { throw new Error("服务响应异常，请恢复会话查看进度。"); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "输入未通过检查，请确认会话编号和问题长度。");
  return data;
}
function userMessage(text) {
  $("messages").append(element("div", "message user", text));
  $("welcome").hidden = true;
}
function readable(text, cards) {
  let result = String(text);
  cards.forEach((card, index) => { result = result.replaceAll(card.product_id, `候选 ${index + 1}`); });
  return result;
}
function assistantMessage(result) {
  if (!result || result.status === "empty") return;
  const cards = result.candidates || [];
  const message = element("article", "message assistant");
  const heading = element("div", "assistant-heading");
  heading.append(element("span", "", "据"), element("div", "", "有据选购"));
  message.append(heading, element("div", "answer-status", statusLabels[result.status] || "正在整理你的选择"));
  const needs = result.requirements;
  if (needs) {
    const tags = [...(needs.sub_categories || [])];
    if (needs.max_price != null) tags.push(`预算 ≤ ${needs.max_price} 元`);
    if (result.relative_price_ceiling != null) tags.push(`本轮价格 ≤ ${result.relative_price_ceiling} 元`);
    message.append(element("div", "need-tags", tags.join(" · ")));
  }
  for (const text of [...(result.questions || []), ...(result.unsupported_constraints || [])]) message.append(element("p", "answer-text", text));
  if (result.message) message.append(element("p", "answer-text", result.message));
  if (result.review_pending) message.append(element("p", "answer-text", "以下商品符合筛选条件，适配审核尚未完成。"));
  if (cards.length) {
    const grid = element("div", "product-grid");
    cards.forEach((card, index) => {
      const item = element("section", "product-card");
      item.append(element("div", "card-index", `候选 ${String(index + 1).padStart(2, "0")} · ${card.brand || card.sub_category || ""}`), element("h3", "", card.title));
      if (card.sku) {
        const price = Number(card.sku.price);
        item.append(element("div", "price", Number.isFinite(price) ? `¥ ${price.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "价格待确认"));
        item.append(element("div", "specs", Object.entries(card.sku.properties || {}).map(([key, value]) => `${key}：${value}`).join(" / ")));
      }
      const fits = element("div", "fit-tags");
      (card.fit_review || []).forEach((check) => fits.append(element("span", `tag ${check.status}`, `${check.requirement} · ${fitLabels[check.status] || "待确认"}`)));
      item.append(fits);
      grid.append(item);
    });
    message.append(grid);
  }
  const answer = result.answer || {};
  (answer.claims || []).forEach((claim) => {
    message.append(element("p", "answer-text", readable(claim.text, cards)));
    if (claim.citations?.length) {
      const details = element("details", "citations");
      details.append(element("summary", "", `查看依据 · ${claim.citations.length} 条摘录`));
      claim.citations.forEach((citation) => details.append(element("blockquote", "", citation.quote)));
      message.append(details);
    }
  });
  if (answer.missing_information?.length) {
    const missing = element("div", "missing");
    missing.append(element("strong", "", "还不能确定的地方"));
    const list = element("ul");
    answer.missing_information.forEach((text) => list.append(element("li", "", readable(text, cards))));
    missing.append(list);
    message.append(missing);
  }
  if (result.error) message.append(element("p", "answer-text", result.error));
  $("messages").append(message);
  $("welcome").hidden = true;
}
function renderSession(result) {
  $("messages").replaceChildren();
  $("welcome").hidden = false;
  const history = result.history || [];
  history.forEach((turn) => { userMessage(turn.query); assistantMessage(turn.result); });
  const last = history.at(-1)?.result;
  if (result.status !== "empty" && (!last || last.turn !== result.turn || last.status !== result.status || last.query !== result.query)) {
    if (result.query) userMessage(result.query);
    assistantMessage(result);
  }
  state.latest = result;
  controls();
  if (result.paused) notice("条件已保存，请在下方补充信息后继续。");
  else if (result.resumable) notice("进度已保存。如果是缺缓存，先勾选允许在线调用，再重试未完成步骤。", true);
  else notice("");
}
async function restore(id) {
  const result = await request(`/api/sessions/${encodeURIComponent(id)}`);
  remember(id);
  renderSession(result);
  requestAnimationFrame(() => { const pane = document.querySelector(".workspace"); pane.scrollTop = pane.scrollHeight; });
}
async function submit(retry = false) {
  if (state.busy) return;
  const query = $("query").value.trim();
  if (!retry && !query) return;
  state.busy = true;
  controls();
  notice("");
  if (!retry) userMessage(query);
  try {
    const body = { session_id: state.session, online: $("online").checked };
    if (!retry) body.query = query;
    const result = await request(retry ? "/api/retry" : "/api/chat", body);
    $("query").value = "";
    try { await restore(state.session); }
    catch (_) { state.latest = result; assistantMessage(result); notice("结果已返回，历史记录加载失败，可点击恢复会话重新读取。", true); }
  } catch (error) {
    try { await restore(state.session); } catch (_) { /* Never auto-resend a possibly billable request. */ }
    notice(error.message + " 请先查看当前进度，避免重复发送。", true);
  } finally {
    state.busy = false;
    controls();
    if (!$("query").disabled) $("query").focus();
    const pane = document.querySelector(".workspace");
    pane.scrollTo({ top: pane.scrollHeight, behavior: "smooth" });
  }
}
$("chat-form").addEventListener("submit", (event) => { event.preventDefault(); submit(); });
$("query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); $("chat-form").requestSubmit(); }
});
$("retry").addEventListener("click", () => submit(true));
$("new-session").addEventListener("click", () => {
  remember(newId());
  $("query").value = "";
  $("online").checked = false;
  renderSession({ status: "empty", history: [] });
  $("query").focus();
});
$("restore-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const id = $("session-id").value.trim();
  state.busy = true;
  controls();
  try { await restore(id); } catch (error) { $("session-id").value = state.session; notice(error.message, true); }
  finally { state.busy = false; controls(); }
});
document.querySelectorAll("[data-example]").forEach((button) => button.addEventListener("click", () => { $("query").value = button.dataset.example; $("query").focus(); }));
(async function boot() {
  let id;
  try { id = localStorage.getItem("shopping.session"); } catch (_) { /* Storage is optional. */ }
  remember(id || newId());
  state.busy = true;
  controls();
  try { await restore(state.session); } catch (error) { notice("连接失败：" + error.message, true); }
  finally { state.busy = false; controls(); }
})();
