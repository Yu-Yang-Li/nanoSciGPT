(function () {
  "use strict";

  const pages = window.COURSE_PAGES || [];
  const storageKey = "ai4s-dual-track-answers-v1";
  const token = new URLSearchParams(location.hash.replace(/^#/, "")).get("token") || "";
  const apiBase = "./api";
  const state = {
    pageIndex: Math.min(Math.max(Number(new URLSearchParams(location.search).get("p") || 1) - 1, 0), pages.length - 1),
    answers: loadAnswers(),
    promptDirty: false,
    conversationId: createId(),
    runningJob: null,
    connected: false
  };

  const el = (id) => document.getElementById(id);
  const ui = {
    pageTitle: el("pageTitle"), pageMinutes: el("pageMinutes"), pageCounter: el("pageCounter"), pageRail: el("pageRail"),
    prevPage: el("prevPage"), nextPage: el("nextPage"), lessonScroll: el("lessonScroll"), stagePath: el("stagePath"),
    lessonHeading: el("lessonHeading"), lessonClaim: el("lessonClaim"), demoCanvas: el("demoCanvas"), demoCaption: el("demoCaption"),
    fieldGrid: el("fieldGrid"), whyNext: el("whyNext"), stopBoundary: el("stopBoundary"), fillExample: el("fillExample"),
    connectionState: el("connectionState"), chatScroll: el("chatScroll"), promptEditor: el("promptEditor"), promptStatus: el("promptStatus"),
    rebuildPrompt: el("rebuildPrompt"), copyPrompt: el("copyPrompt"), sendPrompt: el("sendPrompt"), newConversation: el("newConversation"),
    runStatus: el("runStatus"), runElapsed: el("runElapsed"), toast: el("toast")
  };

  function createId() {
    return globalThis.crypto && crypto.randomUUID ? crypto.randomUUID() : `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function loadAnswers() {
    try { return JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { return {}; }
  }

  function saveAnswers() {
    localStorage.setItem(storageKey, JSON.stringify(state.answers));
  }

  function toast(message) {
    ui.toast.textContent = message;
    ui.toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => ui.toast.classList.remove("show"), 1600);
  }

  function renderRail() {
    ui.pageRail.replaceChildren(...pages.map((page, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `rail-page ${index < state.pageIndex ? "done" : ""} ${index === state.pageIndex ? "active" : ""}`;
      button.title = `P${page.id} · ${page.title}`;
      button.innerHTML = `<i></i><span>${String(page.id).padStart(2, "0")}</span>`;
      button.addEventListener("click", () => showPage(index));
      return button;
    }));
  }

  function pageAnswers(page) {
    return state.answers[String(page.id)] || {};
  }

  function renderFields(page) {
    const values = pageAnswers(page);
    ui.fieldGrid.replaceChildren(...page.fields.map((field) => {
      const wrapper = document.createElement("div");
      wrapper.className = `field-control ${field.size || ""}`;
      const label = document.createElement("label");
      label.htmlFor = `field-${page.id}-${field.key}`;
      label.textContent = field.label;
      let input;
      if (field.type === "select") {
        input = document.createElement("select");
        field.options.forEach((option) => {
          const item = document.createElement("option"); item.value = option; item.textContent = option; input.appendChild(item);
        });
      } else if (field.type === "textarea") {
        input = document.createElement("textarea");
        input.placeholder = field.placeholder || "";
      } else {
        input = document.createElement("input"); input.type = "text"; input.placeholder = field.placeholder || "";
      }
      input.id = label.htmlFor;
      input.value = values[field.key] || "";
      input.dataset.key = field.key;
      input.addEventListener("input", () => {
        state.answers[String(page.id)] = state.answers[String(page.id)] || {};
        state.answers[String(page.id)][field.key] = input.value;
        saveAnswers();
        if (!state.promptDirty) rebuildPrompt();
        else ui.promptStatus.textContent = "左侧输入已变化 · 点击重建";
      });
      wrapper.append(label, input);
      return wrapper;
    }));
  }

  function contextPageIds(pageId) {
    if (pageId <= 4) return Array.from({ length: pageId }, (_, i) => i + 1);
    if (pageId <= 10) return [...new Set([4, pageId])];
    if (pageId <= 16) return [...new Set([4, 10, pageId])];
    return [...new Set([4, 10, 16, pageId >= 21 ? 20 : pageId])];
  }

  function buildInputSummary(page) {
    const sections = [];
    contextPageIds(page.id).forEach((pageId) => {
      const sourcePage = pages[pageId - 1];
      const values = pageAnswers(sourcePage);
      const lines = sourcePage.fields.map((field) => {
        const value = values[field.key];
        return value ? `${field.label}：${value}` : "";
      }).filter(Boolean);
      if (lines.length) sections.push(`P${pageId} ${sourcePage.stage}\n${lines.join("\n")}`);
    });
    return sections.length ? sections.join("\n\n") : "尚未填写；请把缺失信息标记为信息不足。";
  }

  function defaultPrompt(page) {
    return `${window.AI4S_VISIBLE_CONTRACT}\n\n【当前页】P${page.id} · ${page.title}\n${page.prompt}\n\n【我在网页中提供的输入】\n${buildInputSummary(page)}`;
  }

  function rebuildPrompt() {
    const page = pages[state.pageIndex];
    ui.promptEditor.value = defaultPrompt(page);
    state.promptDirty = false;
    ui.promptStatus.textContent = "已根据左侧输入生成";
  }

  function showPage(index) {
    if (index < 0 || index >= pages.length) return;
    state.pageIndex = index;
    const page = pages[index];
    history.replaceState({}, "", `${location.pathname}?p=${page.id}${location.hash}`);
    ui.pageTitle.textContent = `P${page.id} · ${page.stage}`;
    ui.pageMinutes.textContent = `${page.time} min`;
    ui.pageCounter.textContent = `${String(page.id).padStart(2, "0")}/25`;
    ui.stagePath.textContent = page.stage;
    ui.lessonHeading.textContent = page.title;
    ui.lessonClaim.textContent = page.claim;
    ui.demoCaption.textContent = page.caption || "非真实训练或实验";
    ui.whyNext.textContent = page.whyNext;
    ui.stopBoundary.textContent = page.boundary;
    renderRail();
    renderFields(page);
    window.renderAI4SSimulation(ui.demoCanvas, page);
    state.promptDirty = false;
    rebuildPrompt();
    ui.prevPage.disabled = index === 0;
    ui.nextPage.disabled = index === pages.length - 1;
    const active = ui.pageRail.querySelector(".active");
    if (active && window.innerWidth <= 860) active.scrollIntoView({ block: "nearest", inline: "nearest" });
    ui.lessonScroll.scrollTop = 0;
  }

  function fillExample() {
    const page = pages[state.pageIndex];
    state.answers[String(page.id)] = state.answers[String(page.id)] || {};
    page.fields.forEach((field) => { state.answers[String(page.id)][field.key] = field.example || ""; });
    saveAnswers();
    renderFields(page);
    rebuildPrompt();
    toast("已载入教学示例；请替换为自己的课题");
  }

  function message(role, text, meta = "") {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "你" : "AI";
    const body = document.createElement("div");
    body.className = "message-body";
    const content = document.createElement("p");
    content.textContent = text;
    body.appendChild(content);
    if (meta) {
      const foot = document.createElement("div"); foot.className = "message-meta"; foot.textContent = meta; body.appendChild(foot);
    }
    wrapper.append(avatar, body);
    ui.chatScroll.appendChild(wrapper);
    ui.chatScroll.scrollTop = ui.chatScroll.scrollHeight;
    return wrapper;
  }

  function systemMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "message system-message";
    const body = document.createElement("div"); body.className = "message-body";
    const content = document.createElement("p"); content.textContent = text; body.appendChild(content); wrapper.appendChild(body);
    ui.chatScroll.appendChild(wrapper); ui.chatScroll.scrollTop = ui.chatScroll.scrollHeight;
  }

  function setConnection(kind, label) {
    ui.connectionState.className = `connection ${kind}`;
    ui.connectionState.querySelector("strong").textContent = label;
  }

  async function healthCheck() {
    try {
      const response = await fetch(`${apiBase}/health`, { cache: "no-store" });
      if (!response.ok) throw new Error("bridge unavailable");
      const health = await response.json();
      state.connected = Boolean(health.ok && health.codex_found && health.router_reachable && token);
      if (state.connected) setConnection("connected", "已连接");
      else if (token) setConnection("error", "CLI不可用");
      else setConnection("static", "展示模式 · 可复制");
    } catch (_) {
      state.connected = false;
      setConnection("static", "展示模式 · 可复制");
    }
    ui.sendPrompt.disabled = !state.connected || Boolean(state.runningJob);
  }

  function authHeaders(json = false) {
    const headers = { "X-AI4S-Token": token };
    if (json) headers["Content-Type"] = "application/json";
    return headers;
  }

  async function sendPrompt() {
    const prompt = ui.promptEditor.value.trim();
    if (!prompt) { toast("Prompt不能为空"); return; }
    if (!state.connected) { toast("公开展示模式可复制Prompt；真实调用需在本机启动桥接"); await healthCheck(); return; }
    message("user", prompt, `P${pages[state.pageIndex].id} · 完整可见Prompt`);
    const thinking = message("assistant", "正在通过Codex CLI处理", "真实调用 · scnet/GLM-5.3");
    thinking.classList.add("thinking");
    ui.sendPrompt.disabled = true;
    ui.runStatus.textContent = "正在运行";
    ui.runElapsed.textContent = "计时中";
    try {
      const response = await fetch(`${apiBase}/jobs`, {
        method: "POST",
        headers: authHeaders(true),
        body: JSON.stringify({ prompt, conversation_id: state.conversationId })
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "无法创建Codex任务");
      state.runningJob = payload.job.id;
      const job = await pollJob(payload.job.id);
      thinking.remove();
      if (job.status === "succeeded") {
        message("assistant", typeof job.result === "string" ? job.result : JSON.stringify(job.result, null, 2), `已运行 · ${job.elapsed_seconds}s · exit 0`);
        ui.runStatus.textContent = "已运行 · 待按内容判断是否已评测";
        ui.runElapsed.textContent = `${job.elapsed_seconds}s`;
      } else {
        const error = message("assistant", `真实调用未完成：${job.error || job.status}`, `状态：${job.status} · 未自动回退其他模型`);
        error.classList.add("error-message");
        ui.runStatus.textContent = "运行失败 · 保留错误";
        ui.runElapsed.textContent = `${job.elapsed_seconds || 0}s`;
      }
    } catch (error) {
      thinking.remove();
      const item = message("assistant", `连接或调用失败：${error.message}`, "真实错误已保留 · 未加载模拟回答");
      item.classList.add("error-message");
      ui.runStatus.textContent = "连接失败";
      ui.runElapsed.textContent = "—";
    } finally {
      state.runningJob = null;
      ui.sendPrompt.disabled = !state.connected;
    }
  }

  async function pollJob(jobId) {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 650));
      const response = await fetch(`${apiBase}/jobs/${jobId}`, { headers: authHeaders(), cache: "no-store" });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "无法读取任务状态");
      if (["succeeded", "failed", "cancelled"].includes(payload.job.status)) return payload.job;
      const elapsed = payload.job.started_at ? Math.max(0, Math.round(Date.now() / 1000 - payload.job.started_at)) : 0;
      ui.runElapsed.textContent = `${elapsed}s`;
    }
  }

  function newConversation() {
    if (state.runningJob) { toast("当前调用结束后再新建对话"); return; }
    state.conversationId = createId();
    ui.chatScroll.innerHTML = "";
    message("assistant", "新对话已建立。当前页Prompt仍完整可见；发送后将创建新的Codex会话。", "设计状态 · 新会话");
    ui.runStatus.textContent = "设计 · 尚未运行";
    ui.runElapsed.textContent = "—";
  }

  ui.prevPage.addEventListener("click", () => showPage(state.pageIndex - 1));
  ui.nextPage.addEventListener("click", () => showPage(state.pageIndex + 1));
  ui.fillExample.addEventListener("click", fillExample);
  ui.rebuildPrompt.addEventListener("click", rebuildPrompt);
  ui.copyPrompt.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(ui.promptEditor.value); toast("Prompt已复制"); }
    catch (_) { ui.promptEditor.select(); document.execCommand("copy"); toast("Prompt已复制"); }
  });
  ui.promptEditor.addEventListener("input", () => { state.promptDirty = true; ui.promptStatus.textContent = "已手动编辑"; });
  ui.promptEditor.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); sendPrompt(); } });
  ui.sendPrompt.addEventListener("click", sendPrompt);
  ui.newConversation.addEventListener("click", newConversation);
  window.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (event.key === "ArrowRight" || event.key === "PageDown") showPage(state.pageIndex + 1);
    if (event.key === "ArrowLeft" || event.key === "PageUp") showPage(state.pageIndex - 1);
  });

  showPage(state.pageIndex);
  healthCheck();
  setInterval(healthCheck, 15000);
})();
