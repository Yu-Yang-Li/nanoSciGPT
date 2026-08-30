(function () {
  "use strict";

  const node = (label, sub = "", cls = "") => `<div class="flow-node ${cls}">${label}${sub ? `<small>${sub}</small>` : ""}</div>`;
  const arrow = () => `<span class="flow-arrow" aria-hidden="true"></span>`;
  const buttons = (items, selected = 0, attr = "data-choice") => `<div class="sim-button-row">${items.map((item, index) => `<button type="button" class="sim-button ${index === selected ? "selected" : ""}" ${attr}="${index}">${item}</button>`).join("")}</div>`;
  const metric = (label, width, value, cls = "") => `<div class="metric-row ${cls}"><span>${label}</span><div class="metric-track"><div class="metric-fill" style="width:${width}%"></div></div><strong>${value}</strong></div>`;
  const check = (label, status, done = false) => `<div class="check-row ${done ? "done" : ""}"><i>${done ? "✓" : "·"}</i><strong>${label}</strong><span>${status}</span></div>`;

  function bindChoice(root, callback) {
    root.querySelectorAll("[data-choice]").forEach((button) => {
      button.addEventListener("click", () => {
        root.querySelectorAll("[data-choice]").forEach((item) => item.classList.remove("selected"));
        button.classList.add("selected");
        callback(Number(button.dataset.choice), button);
      });
    });
  }

  function executionGap(root) {
    root.innerHTML = `<div class="sim-stack">
      <p class="sim-title">一句模糊指令，会得到什么？</p>
      <div class="flow-row" data-flow>${["读取数据", "选择模型", "生成分数", "输出候选", "撰写报告"].map((item, i) => `${node(item, "", i === 0 ? "active" : "")}${i < 4 ? arrow() : ""}`).join("")}</div>
      <div class="check-list" data-warnings hidden>${check("数据身份", "不明")}${check("评价器", "未定义")}${check("真实反馈", "没有")}${check("结论边界", "缺失")}</div>
      <div class="sim-button-row"><button class="sim-button selected" type="button" data-run>模拟Agent执行</button></div>
      <p class="sim-note" data-result>工程流程：准备执行　·　科学任务：尚未定义</p>
    </div>`;
    let timer = null;
    root.querySelector("[data-run]").addEventListener("click", () => {
      if (timer) return;
      const nodes = [...root.querySelectorAll(".flow-node")];
      let index = 0;
      nodes.forEach((item) => item.classList.remove("active", "good"));
      root.querySelector("[data-warnings]").hidden = true;
      timer = setInterval(() => {
        if (index > 0) nodes[index - 1].classList.replace("active", "good");
        if (index < nodes.length) nodes[index].classList.add("active");
        index += 1;
        if (index > nodes.length) {
          clearInterval(timer); timer = null;
          root.querySelector("[data-warnings]").hidden = false;
          root.querySelector("[data-result]").textContent = "工程流程：已执行　·　科学任务：尚未定义";
        }
      }, 260);
    });
  }

  function dualAxis(root) {
    root.innerHTML = `<div class="sim-stack">
      <div class="axis-wrap"><span class="axis-label-y">科研过程长度</span><span class="axis-label-x">能力复用半径</span><i class="axis-dot" data-dot></i></div>
      <div class="sim-button-row axis-controls"><label>共享任务 <input data-x type="range" min="8" max="92" value="28"></label><label>真实反馈 <input data-y type="range" min="8" max="92" value="22"></label></div>
      <p class="sim-note" data-label>当前建议：专用模型 + 人工/工具工作流</p>
    </div>`;
    const dot = root.querySelector("[data-dot]");
    const x = root.querySelector("[data-x]");
    const y = root.querySelector("[data-y]");
    const update = () => {
      dot.style.left = `${x.value}%`; dot.style.bottom = `${y.value}%`;
      const right = Number(x.value) > 50; const high = Number(y.value) > 50;
      root.querySelector("[data-label]").textContent = `当前建议：${right ? "基座迁移" : "专用模型"} + ${high ? "反馈闭环" : "工具工作流"}（不是成熟度排名）`;
    };
    x.addEventListener("input", update); y.addEventListener("input", update); update();
  }

  function evidenceLadder(root) {
    const labels = ["模型输出", "已运行", "已评测", "已外部验证", "科学结论"];
    root.innerHTML = `<div class="sim-stack"><div class="evidence-ladder" data-ladder>${labels.map((label, i) => `<div class="evidence-step ${i <= 2 ? "reached" : ""}">${label}</div>`).join("")}</div>${buttons(["只有模型自评", "独立benchmark", "真实实验/观测"], 1)}<p class="sim-note" data-result>当前最高：已评测；不能升级为外部验证。</p></div>`;
    bindChoice(root, (index) => {
      const reached = [1, 2, 3][index];
      root.querySelectorAll(".evidence-step").forEach((item, i) => item.classList.toggle("reached", i <= reached));
      root.querySelector("[data-result]").textContent = index === 0 ? "SELF-CONFIRMATION RISK：同一模型不能独立证明自己。" : index === 1 ? "当前最高：已评测；不能升级为外部验证。" : "当前最高：已外部验证；仍需研究者解释科学结论。";
    });
  }

  function entryCard(root) {
    const items = ["科学问题", "科学对象", "数据单位", "现有数据", "目标输出", "当前瓶颈", "最低评价器", "真实反馈", "结论边界"];
    root.innerHTML = `<div class="sim-stack"><p class="sim-title">九字段课题入口卡</p><div class="check-list" data-list>${items.map((item, i) => check(item, i < 6 ? "已填写" : "待补", i < 6)).join("")}</div><div class="sim-button-row"><button type="button" class="sim-button selected" data-check>检查课题入口</button></div><p class="sim-note" data-result>入口尚未完整，但允许明确写“无反馈”。</p></div>`;
    root.querySelector("[data-check]").addEventListener("click", () => {
      root.querySelectorAll(".check-row").forEach((row) => { row.classList.add("done"); row.querySelector("i").textContent = "✓"; row.lastElementChild.textContent = row.querySelector("strong").textContent === "真实反馈" ? "无（合法约束）" : "已填写"; });
      root.querySelector("[data-result]").textContent = "入口卡可继续：优先专用模型；暂不构成闭环。";
    });
  }

  function baselineSplit(root) {
    root.innerHTML = `<div class="sim-stack"><p class="sim-title">同一V0，换一种划分会发生什么？</p>${buttons(["随机划分", "同源簇留出", "时间留出"], 0)}<div data-metrics>${metric("表面得分", 91, "0.91")}${metric("失败敏感指标", 84, "0.84")}</div><p class="sim-note" data-result>随机划分可能把相似对象同时放进训练和测试。</p></div>`;
    const states = [[91,84,"0.91","0.84","高分不等于可泛化：先检查泄漏。"],[72,53,"0.72","0.53","对象留出暴露新家族失效区。"],[66,48,"0.66","0.48","时间留出检验未来数据漂移。"]];
    bindChoice(root, (index) => {
      const s = states[index]; const bars = root.querySelectorAll(".metric-fill"); const values = root.querySelectorAll(".metric-row strong");
      bars[0].style.width = `${s[0]}%`; bars[1].style.width = `${s[1]}%`; values[0].textContent = s[2]; values[1].textContent = s[3]; root.querySelector("[data-result]").textContent = s[4];
    });
  }

  function pretraining(root) {
    root.innerHTML = `<div class="sim-stack"><p class="sim-title">蛋白质序列（教学示例）</p><div class="token-row"><span class="token">M</span><span class="token">A</span><span class="token mask" data-mask>[MASK]</span><span class="token">K</span><span class="token">L</span></div><div class="flow-row">${node("1 编码", "学习上下文")}${arrow()}${node("2 恢复", "遮盖目标")}${arrow()}${node("3 表征", "等待探针")}</div><div class="sim-button-row"><button type="button" class="sim-button selected" data-restore>播放预训练</button></div><div class="flow-row" data-downstream style="opacity:.35">${node("任务A", "结构预测")}${node("任务B", "功能注释")}</div><p class="sim-note" data-result>预训练损失下降，尚不能证明迁移有效。</p></div>`;
    root.querySelector("[data-restore]").addEventListener("click", () => {
      const mask = root.querySelector("[data-mask]"); mask.textContent = "R"; mask.classList.remove("mask"); mask.classList.add("good"); root.querySelector("[data-downstream]").style.opacity = "1"; root.querySelector("[data-result]").textContent = "下一步：冻结表征，用两个简单探针检验是否可迁移。";
    });
  }

  function transfer(root) {
    root.innerHTML = `<div class="sim-stack"><div class="flow-row">${node("共享骨干", "预训练表征", "active")}${arrow()}${node("任务A", "+0.10", "good")}${node("任务B", "−0.04", "bad")}${node("任务C", "+0.06", "good")}</div>${buttons(["100%标签", "10%标签", "1%标签"], 0)}<div data-metrics>${metric("任务A", 82, "0.82")}${metric("任务B", 64, "0.64")}${metric("任务C", 74, "0.74")}</div><p class="sim-note" data-result>平均迁移为正，但任务B负迁移：不能只报平均值。</p></div>`;
    const values = [[82,64,74],[75,55,68],[60,48,57]];
    bindChoice(root, (index) => root.querySelectorAll(".metric-fill").forEach((bar, i) => { bar.style.width = `${values[index][i]}%`; bar.parentElement.nextElementSibling.textContent = `0.${values[index][i]}`; }));
  }

  function capabilityToggles(root) {
    root.innerHTML = `<div class="sim-stack"><div class="flow-row">${node("共享表征", "已具备", "good")}${arrow()}${node("第二模态", "关闭")}${node("生成能力", "关闭")}${node("统一接口", "关闭")}</div><div class="sim-button-row" data-toggles><label class="sim-button"><input type="checkbox" value="模态"> 第二模态</label><label class="sim-button"><input type="checkbox" value="生成"> 生成</label><label class="sim-button"><input type="checkbox" value="统一"> 统一接口</label></div><div class="check-list" data-cost>${check("新增评价器", "0")}${check("新增对齐/约束", "0")}${check("新增失效区", "0")}</div><p class="sim-note" data-result>最小够用接口：共享表征。</p></div>`;
    root.querySelectorAll("[data-toggles] input").forEach((input) => input.addEventListener("change", () => {
      const count = root.querySelectorAll("[data-toggles] input:checked").length;
      root.querySelectorAll("[data-cost] .check-row span").forEach((item) => item.textContent = String(count));
      root.querySelector("[data-result]").textContent = count ? `已打开${count}项能力：每项都必须补独立评价与停止条件。` : "最小够用接口：共享表征。";
    }));
  }

  function routeDecision(root) {
    const routes = [["继续专用", "一个任务 / V0够用"],["调用现成", "接口匹配 / 达到门槛"],["适配基座", "多个共享任务 / 少量标签"],["训练新基座", "长期复用 + 数据与维护"]];
    root.innerHTML = `<div class="sim-stack"><div class="route-grid">${routes.map((r,i)=>`<button type="button" class="route-card ${i===0?"selected":""}" data-choice="${i}"><strong>${r[0]}</strong><small>${r[1]}</small></button>`).join("")}</div><p class="sim-note" data-result>当前证据只支持：继续专用模型。</p></div>`;
    bindChoice(root, (index, button) => { root.querySelectorAll(".route-card").forEach((card)=>card.classList.remove("selected")); button.classList.add("selected"); root.querySelector("[data-result]").textContent = index === 3 ? "训练新基座仍锁定：需证明前三条路线均不足。" : `已选择${routes[index][0]}：下一步补最小验证证据。`; });
  }

  function compiledCard(root) {
    root.innerHTML = `<div class="sim-stack"><div class="check-list">${check("问题—I/O—基线", "一致", true)}${check("预训练—迁移合同", "信息不足")}${check("路线—成本—维护", "一致", true)}${check("生成—独立评价器", "不采用", true)}${check("停止—降级—下一步", "一致", true)}</div><div class="flow-row">${node("DESIGN_READY", "专用路线也可READY", "good")}</div><p class="sim-note">设计卡是可执行判断，不是模型训练证据。</p></div>`;
  }

  function actionMachine(root) {
    root.innerHTML = `<div class="sim-stack"><div class="flow-row">${node("输入", "版本v1")}${arrow()}${node("工具执行", "等待", "active")}${arrow()}${node("机器输出", "—")}${arrow()}${node("评价器", "—")}</div>${buttons(["执行成功 + 评价通过", "执行成功 + 科学未通过", "工具失败"],0)}<p class="sim-note" data-result>运行状态与科学状态分开记录。</p></div>`;
    bindChoice(root,(index)=>{ const nodes=root.querySelectorAll(".flow-node"); nodes.forEach(n=>n.classList.remove("good","bad","warn")); if(index===0){nodes[1].classList.add("good");nodes[2].classList.add("good");nodes[3].classList.add("good");root.querySelector("[data-result]").textContent="工具SUCCESS；评价PASS。";}else if(index===1){nodes[1].classList.add("good");nodes[2].classList.add("good");nodes[3].classList.add("bad");root.querySelector("[data-result]").textContent="工具SUCCESS；科学结果FAIL——这是有效反证。";}else{nodes[1].classList.add("bad");root.querySelector("[data-result]").textContent="工具ERROR；本轮不可评价，必须保留错误。";}});
  }

  function toolContracts(root) {
    root.innerHTML = `<div class="sim-stack"><div class="flow-row">${node("数据表", "ID + 300 K")}${arrow()}${node("脚本A", "期望 °C", "warn")}${arrow()}${node("模型", "调用成功")}${arrow()}${node("评价器", "错误输入")}</div><div class="sim-button-row"><button type="button" class="sim-button selected" data-fix>补齐单位与ID合同</button></div><p class="sim-note" data-result>SILENT ERROR：API成功，但参数含义错误。</p></div>`;
    root.querySelector("[data-fix]").addEventListener("click",()=>{const nodes=root.querySelectorAll(".flow-node");nodes[1].classList.replace("warn","good");nodes[1].innerHTML="显式转换<small>K → °C；ID对齐</small>";nodes[2].classList.add("good");nodes[3].classList.add("good");root.querySelector("[data-result]").textContent="工具链可追溯；仍未形成反馈闭环。";});
  }

  function feedbackLoop(root) {
    root.innerHTML = `<div class="sim-stack">${buttons(["固定自动化", "反馈驱动"],0)}<div class="flow-row" data-flow>${node("第1轮", "参数A", "active")}${arrow()}${node("第2轮", "参数A")}${arrow()}${node("第3轮", "参数A")}</div><p class="sim-note" data-result>运行三轮仍用参数A：这是循环，不是闭环。</p></div>`;
    bindChoice(root,(index)=>{const nodes=root.querySelectorAll(".flow-node");if(index===0){["参数A","参数A","参数A"].forEach((v,i)=>nodes[i].innerHTML=`第${i+1}轮<small>${v}</small>`);root.querySelector("[data-result]").textContent="运行三轮仍用参数A：这是循环，不是闭环。";}else{["参数A","反馈→参数B","越界→停止"].forEach((v,i)=>{nodes[i].innerHTML=`第${i+1}轮<small>${v}</small>`;nodes[i].className=`flow-node ${i===2?"bad":"good"}`;});root.querySelector("[data-result]").textContent="反馈改变第2轮；安全域在第3轮触发停止。";}});
  }

  function researchState(root) {
    root.innerHTML = `<div class="sim-stack"><div class="flow-row">${node("H1-v1", "支持E1", "active")}${arrow()}${node("新证据E2", "反证到达", "warn")}${arrow()}${node("H1-v2", "尚未生成")}</div><div class="check-list">${check("失败结果", "保留",true)}${check("版本触发依据", "E2",true)}${check("未决问题", "仍存在",true)}</div><div class="sim-button-row"><button type="button" class="sim-button selected" data-version>根据E2创建新版本</button></div><p class="sim-note" data-result>旧假设和证据仍在，不允许覆盖。</p></div>`;
    root.querySelector("[data-version]").addEventListener("click",()=>{const n=root.querySelectorAll(".flow-node")[2];n.classList.add("good");n.innerHTML="H1-v2<small>由E2触发；保留v1</small>";root.querySelector("[data-result]").textContent="跨轮状态保存了假设为什么改变。";});
  }

  function responsibility(root) {
    const rows=[["问题设定","仅人工"],["模型/代码","自动并通知"],["高风险设备","执行前批准"],["关键证据","仅人工"],["最终签署","仅人工"]];
    root.innerHTML=`<div class="sim-stack"><div class="check-list">${rows.map(([a,b])=>check(a,b,true)).join("")}</div>${buttons(["全部自动", "平衡权限", "全部人工"],1)}<p class="sim-note" data-result>自动化执行可逆步骤；人在高风险和结论门槛负责。</p></div>`;
    bindChoice(root,(index)=>{root.querySelector("[data-result]").textContent=["风险不可接受：问题、设备和结论越权。","自动化执行可逆步骤；人在高风险和结论门槛负责。","安全但自动化价值接近零：需要重新分配可逆步骤。"][index];});
  }

  function systemLevels(root) {
    const levels=["L1 可执行步骤","L2 工具工作流","L3 反馈闭环","L4 持续研究状态"];
    root.innerHTML=`<div class="sim-stack"><div class="evidence-ladder">${levels.map((label,i)=>`<button type="button" class="evidence-step ${i<2?"reached":""}" data-level="${i}">${label}</button>`).join("")}</div><div class="sim-button-row"><label class="sim-button"><input type="checkbox" data-feedback> 反馈改变下一步</label><label class="sim-button"><input type="checkbox" data-state> 失败与假设跨轮</label></div><p class="sim-note" data-result>当前最高：可追溯工具工作流。</p></div>`;
    const update=()=>{const f=root.querySelector("[data-feedback]").checked;const s=root.querySelector("[data-state]").checked;const max=s&&f?3:f?2:1;root.querySelectorAll(".evidence-step").forEach((n,i)=>n.classList.toggle("reached",i<=max));root.querySelector("[data-result]").textContent=`当前最高：${levels[max].replace(/^L\d\s/,"")}。${s&&!f?"（状态存在但反馈闭环缺失，仍降级）":""}`;};root.querySelectorAll("input").forEach(i=>i.addEventListener("change",update));
  }

  function systemMap(root) {
    const items=[["科学问题","人的设定"],["模型工具","能力输出"],["Agent行动","调用与记录"],["独立评价器","可以否决"],["真实反馈","改变下一步"],["状态更新","保存为什么"],["人工关口","授权/签署"]];
    root.innerHTML=`<div class="sim-stack"><div class="flow-row">${items.map((r,i)=>`${node(r[0],r[1],i===3?"good":"")}${i<items.length-1?arrow():""}`).join("")}</div>${buttons(["模型输出直连结论", "经过评价器与反馈"],1)}<p class="sim-note" data-result>正确链：模型能力经过独立评价器，反馈再更新行动。</p></div>`;
    bindChoice(root,(index)=>{root.querySelector("[data-result]").textContent=index===0?"链路断开：MODEL OUTPUT ≠ SCIENTIFIC CLAIM。":"正确链：模型能力经过独立评价器，反馈再更新行动。";});
  }

  function weatherDemo(root) {
    const colors=["#dff4fb","#c9eaf6","#a9ddeb","#82cada","#5ab6ca","#389eb7","#2185a1","#126b87","#09516e"];
    root.innerHTML=`<div class="sim-stack"><div class="flow-row"><div><p class="sim-title">3×3 合成天气网格 · t4</p><div class="grid-demo">${colors.map((c,i)=>`<span class="grid-cell" style="background:${c}">${(12.2+i*.3).toFixed(1)}</span>`).join("")}</div></div><div style="min-width:220px;display:grid;gap:12px">${metric("持续性MAE",70,"0.700")}${metric("趋势MAE",20,"0.200")}${metric("训练最大增量",60,"0.600")}${metric("测试增量",70,"0.700")}</div></div><div class="sim-button-row"><button type="button" class="sim-button selected" data-weather>播放：数据 → 基线 → 评价 → 修正</button></div><p class="sim-note" data-result>尚未运行；左侧只是确定性HTML演示。</p></div>`;
    const messages=["读取45行合成fixture；身份=synthetic_teaching_only。","持续性MAE=0.700；趋势MAE=0.200。","反证：测试增量0.700超过训练最大0.600。","下一步：真实课题按时间/区域留出并增加OOD门槛。"];
    root.querySelector("[data-weather]").addEventListener("click",()=>{let i=0;const result=root.querySelector("[data-result]");const timer=setInterval(()=>{result.textContent=messages[i];i+=1;if(i===messages.length)clearInterval(timer);},480);});
  }

  function blueprintDraft(root) {
    root.innerHTML=`<div class="sim-stack"><div class="flow-row">${node("P4入口卡","已有输入","good")}${arrow()}${node("P10模型卡","已有路线","good")}${arrow()}${node("P16闭环卡","已有层级","good")}${arrow()}${node("系统蓝图","等待生成")}</div><div class="route-grid"><div class="route-card selected"><strong>来自原设计</strong><small>8项</small></div><div class="route-card"><strong>未定义</strong><small>3项</small></div><div class="route-card"><strong>建议·未验证</strong><small>2项</small></div><div class="route-card"><strong>已运行证据</strong><small>0项</small></div></div><p class="sim-note">Codex只能整理已有输入；建议不能自动升级为事实。</p></div>`;
  }

  function auditGates(root) {
    const gates=[["模型路线","单任务新基座"],["反馈真实性","REPLAY冒充在线"],["评价器独立","模型自评"],["结论与责任","候选冒充发现"]];
    root.innerHTML=`<div class="sim-stack"><div class="route-grid">${gates.map(([a,b],i)=>`<button type="button" class="route-card" data-gate="${i}"><strong>${a}</strong><small>${b}</small></button>`).join("")}</div><p class="sim-note" data-result>点击四道门，定位并修正注入问题。</p></div>`;
    root.querySelectorAll("[data-gate]").forEach(btn=>btn.addEventListener("click",()=>{btn.classList.toggle("selected");btn.querySelector("small").textContent=btn.classList.contains("selected")?"已降级 / 补证据":"待审计";const count=root.querySelectorAll(".route-card.selected").length;root.querySelector("[data-result]").textContent=count===4?"BLUEPRINT_REVIEWED：已审查，不等于系统验证。":`已通过 ${count}/4 道门。`; }));
  }

  function timeline(root, kind) {
    const configs={
      protein:{
        title:"蛋白质模型能力角色",
        steps:[
          ["专用预测","AlphaFold2","从序列到结构","结构输出不能自动推出功能","共享表征"],
          ["预训练迁移","UniRep / TAPE / ESM","跨序列任务复用表示","基座性要由迁移合同和从头训练对照证明","条件生成"],
          ["多模态生成","ESM3","在条件下生成序列或结构","共享表示不等于生物机制","约束候选"],
          ["条件候选","RFdiffusion","按结构约束生成候选","候选仍需独立计算与实验评价","外部验证"]
        ]
      },
      weather:{
        title:"天气与气候模型能力角色",
        steps:[
          ["专用预报","Pangu / GraphCast","给定分析初值做天气预报","依赖初始场和传统系统，不是自主系统","跨任务适配"],
          ["预训练适配","ClimaX","把共享表示迁移到多个任务","跨任务能力要逐项写清训练与评价合同","多任务基座"],
          ["多任务基座","Aurora","面向多类地球系统任务适配","平均优势不能掩盖区域、极端和OOD失败","物理混合"],
          ["混合模拟","NeuralGCM","学习模块与物理方程协同","必须检查守恒、长期稳定和误差累积","独立评价"]
        ]
      },
      scientist:{
        title:"AI Scientist系统能力角色",
        steps:[
          ["工具编排","Coscientist / ChemCrow","规划并调用科学工具","工具执行成功不等于形成反馈闭环","真实反馈"],
          ["真实反馈","SAMPLE / 小来 / StarWhisper","实验或观测结果改变下一步","REPLAY、计算反馈和在线反馈必须分开标注","持续状态"],
          ["持续过程","The AI Scientist / Robin","跨轮保存假设、反证与失败","长流程或多Agent数量不等于可信科学家","人工关口"]
        ]
      }
    };
    const config=configs[kind];
    const steps=config.steps;
    root.innerHTML=`<div class="knowledge-map" data-knowledge-map>
      <div class="knowledge-map__header"><div><span class="knowledge-map__eyebrow">KNOWLEDGE ROLE MAP</span><strong>${config.title}</strong></div><span>点击节点查看科学边界</span></div>
      <div class="knowledge-map__track" role="list" aria-label="${config.title}">
        ${steps.map(([phase,model,role,boundary,next],i)=>`<div class="knowledge-map__step" role="listitem">
          <button type="button" class="knowledge-node ${i===0?"selected":""}" data-knowledge-node="${i}" aria-pressed="${i===0?"true":"false"}">
            <span class="knowledge-node__index">${String(i+1).padStart(2,"0")}</span><span class="knowledge-node__phase">${phase}</span><strong>${model}</strong><small>${role}</small>
          </button>${i<steps.length-1?`<div class="knowledge-edge" aria-hidden="true"><span>${next}</span><i></i></div>`:""}
        </div>`).join("")}
      </div>
      <div class="knowledge-inspector" aria-live="polite"><span class="knowledge-inspector__index" data-knowledge-index>01</span><div><span>当前节点的科学边界</span><strong data-knowledge-boundary>${steps[0][3]}</strong></div><p data-knowledge-role>${steps[0][2]}</p></div>
      <p class="knowledge-map__note">这是能力角色与评价关系图，不是代码谱系、产品谱系或历史演化图。</p>
    </div>`;
    const nodes=[...root.querySelectorAll("[data-knowledge-node]")];
    const select=(index)=>{
      nodes.forEach((button,i)=>{const active=i===index;button.classList.toggle("selected",active);button.setAttribute("aria-pressed",String(active));});
      root.querySelector("[data-knowledge-index]").textContent=String(index+1).padStart(2,"0");
      root.querySelector("[data-knowledge-boundary]").textContent=steps[index][3];
      root.querySelector("[data-knowledge-role]").textContent=steps[index][2];
    };
    nodes.forEach((button,index)=>button.addEventListener("click",()=>select(index)));
  }

  function evidencePack(root) {
    const items=["双轨系统蓝图","真实/脱敏输入","Prompt与完整轨迹","可执行评价器与结果","失败/反证","反馈后的修改","能/不能声称"];
    root.innerHTML=`<div class="sim-stack"><div class="check-list">${items.map((item,i)=>`<button type="button" class="check-row ${i<3?"done":""}" data-pack="${i}"><i>${i<3?"✓":"·"}</i><strong>${item}</strong><span>${i<3?"已准备":"待补"}</span></button>`).join("")}</div><p class="sim-note" data-result>EVIDENCE_PACK_INCOMPLETE · 3/7</p></div>`;
    root.querySelectorAll("[data-pack]").forEach(btn=>btn.addEventListener("click",()=>{btn.classList.toggle("done");btn.querySelector("i").textContent=btn.classList.contains("done")?"✓":"·";btn.lastElementChild.textContent=btn.classList.contains("done")?"已准备":"待补";const count=root.querySelectorAll(".check-row.done").length;root.querySelector("[data-result]").textContent=count===7?"EVIDENCE_PACK_READY · 包含失败与边界":"EVIDENCE_PACK_INCOMPLETE · "+count+"/7";}));
  }

  function journey(root) {
    const items=["模糊问题","入口卡","模型原型","闭环原型","审查蓝图","证据包计划"];
    root.innerHTML=`<div class="sim-stack"><div class="flow-row">${items.map((item,i)=>`${node(item,i===0?"起点":"",i===0?"active":"")}${i<items.length-1?arrow():""}`).join("")}</div><div class="sim-button-row"><button type="button" class="sim-button selected" data-play>回放90分钟</button></div><p class="sim-note" data-result>课程完成不按“大”或“自动”排名。</p></div>`;
    root.querySelector("[data-play]").addEventListener("click",()=>{const nodes=[...root.querySelectorAll(".flow-node")];let i=0;nodes.forEach(n=>n.className="flow-node");const timer=setInterval(()=>{nodes[i].classList.add("good");i+=1;if(i===nodes.length){clearInterval(timer);root.querySelector("[data-result]").textContent="终点：一项可执行、可否决、可停止的最小行动。";}},250);});
  }

  window.renderAI4SSimulation = function renderAI4SSimulation(root, page) {
    const renderers = {
      "execution-gap": executionGap,
      "dual-axis": dualAxis,
      "evidence-ladder": evidenceLadder,
      "entry-card": entryCard,
      "baseline-split": baselineSplit,
      pretraining,
      transfer,
      "capability-toggles": capabilityToggles,
      "route-decision": routeDecision,
      "compiled-card": compiledCard,
      "action-machine": actionMachine,
      "tool-contracts": toolContracts,
      "feedback-loop": feedbackLoop,
      "research-state": researchState,
      responsibility,
      "system-levels": systemLevels,
      "system-map": systemMap,
      "weather-demo": weatherDemo,
      "blueprint-draft": blueprintDraft,
      "audit-gates": auditGates,
      "protein-timeline": (target) => timeline(target, "protein"),
      "weather-timeline": (target) => timeline(target, "weather"),
      "scientist-timeline": (target) => timeline(target, "scientist"),
      "evidence-pack": evidencePack,
      journey
    };
    (renderers[page.sim] || compiledCard)(root);
  };
})();
