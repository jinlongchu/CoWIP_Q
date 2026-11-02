"use strict";

const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const pct = value => `${Math.round(value * 100)}%`;
const colors = {a:"#4b7cc6",b:"#c9954b",k:"#008b78",actual:"#008b78",reference:"#c6a269",target:"#aab7bf"};
const controllerLabels = {
  m4:"M4 Q ensemble (PI off)", madqn:"MADQN Q ensemble",
  pi:"Coordinated PI", manual:"Manual cap control"
};

// Translate messages at display time so the running simulation need not reset.
function englishMessage(value) { return String(value); }

const descriptions = {
  m4:"Mean action Q-values from five independent models. PI residual guidance is off.",
  madqn:"Mean local action Q-values from five consistently retrained MADQN models.",
  pi:"The project's coordinated PI controller adjusts all three workshops online.",
  manual:"Set a target cap for each workshop. Each cap changes by at most one card per step."
};
const eventNames = {finished:"Finished output",transfer:"Stage complete",buffer:"Buffer entry",blocked:"Blocked",processing:"Processing",decision:"Decision",control:"Configuration"};
let state = null;
let playing = false;
let busy = false;
let timer = null;
let inFlight = null;
let conditionsDirty = false;
let toastTimer = null;

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {cache:"no-store"} : {
    method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(englishMessage(result.error || "The simulation service is unavailable."));
  return result;
}

function showError(error) {
  $("error").textContent = englishMessage(error.message || String(error));
  $("error").hidden = false;
  playing = false;
  clearTimeout(timer);
  renderButtons();
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3000);
}

async function mutate(path, payload) {
  busy = true;
  renderButtons();
  $("error").hidden = true;
  const operation = request(path, payload);
  inFlight = operation;
  try {
    state = await operation;
    render();
    return state;
  } finally {
    if (inFlight === operation) inFlight = null;
    busy = false;
    renderButtons();
  }
}

async function pauseAndWait() {
  playing = false;
  clearTimeout(timer);
  renderButtons();
  if (inFlight) await inFlight.catch(() => {});
}

function renderButtons() {
  for (const id of ["step","reset","apply-caps","apply-conditions","strategy"]) $(id).disabled = busy || !state;
  $("step").disabled ||= Boolean(state?.done);
  $("play").disabled = !state || Boolean(state?.done);
  $("play").innerHTML = playing ? '<span aria-hidden="true">Ⅱ</span> Pause' : '<span aria-hidden="true">▶</span> Resume';
  if (state?.summary.steps === 0 && !playing) $("play").innerHTML = '<span aria-hidden="true">▶</span> Start';
  if (state?.done) $("play").innerHTML = '<span aria-hidden="true">✓</span> Run complete';
  $("run-status").textContent = state?.done ? "Completed" : playing ? "Running" : "Paused";
  $("status-dot").className = `status-dot${state?.done ? " done" : playing ? " running" : ""}`;
}

async function playStep() {
  if (!playing || state?.done) return;
  try {
    await mutate("/api/step", {count:1});
    if (state.done) {
      playing = false;
      toast("All 360 decisions are complete. You can export the full run.");
    }
    renderButtons();
    if (playing) timer = setTimeout(playStep, Number($("speed").value));
  } catch (error) { showError(error); }
}

function clock(minute) {
  const total = Math.round(minute);
  return `${Math.floor(total / 60).toString().padStart(2,"0")}:${(total % 60).toString().padStart(2,"0")}`;
}

function renderCell(cell) {
  const letter = ["A","B","K"][cell.id];
  const shortName = ["A processing","B processing","Assembly"][cell.id];
  const stages = cell.stages.map(stage => {
    const machines = stage.machines.map(machine => {
      const label = machine.status === "busy" ? "Busy" : machine.status === "blocked" ? "Blocked" : "Idle";
      const title = machine.job ? `${machine.job}; ${label}; ${fmt(machine.remaining,1)} min remaining` : `Machine ${machine.machine} idle`;
      return `<div class="machine ${machine.status}" title="${esc(title)}"><div class="machine-top"><i class="machine-dot"></i><span>${label}${stage.machines.length > 1 ? ` · M${machine.machine}` : ""}</span></div><div class="job-id">${machine.job ? esc(machine.job) : "Waiting"}</div><div class="machine-progress"><div style="width:${pct(machine.progress)}"></div></div></div>`;
    }).join("");
    const dots = '<i class="queue-square"></i>'.repeat(Math.min(stage.queue_count,5));
    return `<div class="station"><div class="station-label"><span>${letter === "K" ? "S" : letter}${stage.number}</span><span class="station-mean">${fmt(stage.mean_minutes,0)}m</span></div>${machines}<div class="queue-label" title="${esc(stage.queue_jobs.length ? stage.queue_jobs.join(" / ") : "Queue empty")}">Queue <b>${stage.queue_count}</b><span class="queue-list">${dots}</span></div></div>`;
  }).join("");
  return `<div class="cell-title"><span class="cell-title-left"><span class="cell-letter">${letter}</span>${shortName}</span><span class="wip-label">WIP <strong>${cell.wip}</strong> / cap ${cell.cap}</span></div><div class="stations">${stages}</div><div class="cell-summary"><span>Busy ${pct(cell.busy)}</span><div class="util-track"><span style="width:${pct(cell.busy)}"></span></div><span>Blocked ${pct(cell.blocked)}</span></div>`;
}

function renderBuffer(buffer, index) {
  let slots = "";
  for (let i=0; i<buffer.capacity; i++) slots += `<span class="buffer-slot${i < buffer.count ? " filled" : ""}" title="${esc(buffer.jobs[i] || "Empty slot")}">${i < buffer.count ? ["A","B"][index] : "·"}</span>`;
  return `<div class="buffer-title">Buffer ${["A","B"][index]}</div><div class="buffer-slots">${slots}</div><div class="buffer-count"><strong>${buffer.count}</strong> / ${buffer.capacity} parts</div>`;
}

function render() {
  const m = state.metrics;
  $("clock").textContent = clock(state.minute);
  $("timeline-progress").style.width = `${state.minute / state.horizon * 100}%`;
  const scheduleTimes = [600,1200,1800,2400,3000];
  $("timeline-markers").innerHTML = state.conditions.schedule_enabled ? scheduleTimes.map(t => `<span class="timeline-marker${state.minute >= t ? " passed" : ""}" style="left:${t / state.horizon * 100}%" title="Scheduled condition change: ${t} min"></span>`).join("") : "";
  $("kpi-throughput").textContent = fmt(m.throughput_moving_kits_per_hour);
  $("kpi-reference").textContent = fmt(m.tracking_target_kits_per_hour);
  $("kpi-target").textContent = fmt(m.target_kits_per_hour,0);
  $("kpi-wip").textContent = fmt(m.total_wip_component_equiv,0);
  $("kpi-mean-wip").textContent = state.summary.steps ? fmt(m.mean_wip_component_equiv) : "—";
  $("kpi-nrmse").textContent = fmt(state.summary.nrmse,4);
  $("kpi-reward").textContent = fmt(state.summary.reward_x1000,2);
  $("kpi-finished").textContent = m.kits_completed;
  $("finished-count").textContent = m.kits_completed;
  $("cell-a").innerHTML = renderCell(state.cells[0]);
  $("cell-b").innerHTML = renderCell(state.cells[1]);
  $("cell-k").innerHTML = renderCell(state.cells[2]);
  $("buffer-a").innerHTML = renderBuffer(state.buffers[0],0);
  $("buffer-b").innerHTML = renderBuffer(state.buffers[1],1);
  const a = state.buffers[0].count, b = state.buffers[1].count;
  $("pair-status").textContent = a && b ? (state.cells[2].wip >= state.cells[2].cap ? "Parts ready / waiting for cap capacity" : "Matching parts ready") : !a && !b ? "Waiting for A / B parts" : `Waiting for ${a ? "B" : "A"} parts`;
  const actionNames = ["Decrease","Hold","Increase"];
  $("policy-caption").textContent = `${controllerLabels[state.controller]}${state.summary.intervened ? " · Custom experiment" : " · Standard configuration"}`;
  $("action-caption").textContent = `Latest: ${state.last_actions.map(a => actionNames[a]).join(" / ")} · Adjustments: ${state.summary.adjustment_points}`;
  $("decision-strategy").textContent = controllerLabels[state.controller];
  $("decision-rows").innerHTML = state.cells.map((cell,i) => {
    const values = state.last_q ? `<span class="q-values">${state.last_q[i].map((q,j) => `<span class="q-value${q == null ? " illegal" : j === state.last_actions[i] ? " selected" : ""}" title="Q-value for ${actionNames[j]}">${q == null ? "N/A" : fmt(q,3)}</span>`).join(" / ")}</span>` : '<span class="muted">—</span>';
    return `<div class="decision-row"><span class="decision-name">${["Workshop A","Workshop B","Assembly"][i]}</span>${values}<span class="action-chip">${actionNames[state.last_actions[i]]}</span></div>`;
  }).join("");
  $("cost-tracking").textContent = fmt(state.reward_parts?.tracking_cost == null ? null : state.reward_parts.tracking_cost * 1000,3);
  $("cost-inventory").textContent = fmt(state.reward_parts?.inventory_cost == null ? null : state.reward_parts.inventory_cost * 1000,3);
  $("cost-action").textContent = fmt(state.reward_parts?.action_cost == null ? null : state.reward_parts.action_cost * 1000,3);
  $("event-count").textContent = `${state.events.length} entries`;
  $("event-rows").innerHTML = state.events.map(event => `<tr><td>${fmt(event.minute,2)} min</td><td><span class="event-kind ${event.kind}">${esc(eventNames[event.kind] || event.kind)}</span></td><td>${esc(englishMessage(event.message))}</td></tr>`).join("");
  $("strategy").innerHTML = state.controllers.map(controller => `<option value="${controller.id}"${controller.id === state.controller ? " selected" : ""}${controller.available ? "" : " disabled"}>${esc(controllerLabels[controller.id] || englishMessage(controller.label))}${controller.available ? "" : " (models missing)"}</option>`).join("");
  $("strategy-description").textContent = descriptions[state.controller];
  $("manual-controls").hidden = state.controller !== "manual";
  if (!conditionsDirty) syncConditions();
  $("connection").textContent = `Connected · seed ${state.seed}`;
  document.querySelector(".connection-dot").classList.remove("offline");
  renderCharts();
  renderButtons();
}

function syncConditions() {
  $("target").value = state.conditions.target;
  const gamma = String(state.conditions.gamma[0]);
  if (["1","2","4","9"].includes(gamma)) $("variability").value = gamma;
  $("slowdown").value = state.conditions.b_slowdown_percent;
  $("slowdown-label").textContent = `${state.conditions.b_slowdown_percent}%`;
}

function chart(id, series, cap = false) {
  const container = $(id);
  const data = state.history;
  const width = Math.max(200,container.clientWidth);
  const height = container.clientHeight;
  const pad = {left:34,right:16,top:10,bottom:30};
  const maxMinute = Math.max(120,state.minute);
  const maxValue = cap ? 8 : Math.max(2,...data.flatMap(point => series.map(s => point[s.key]))) * 1.16;
  const x = minute => pad.left + minute / maxMinute * (width-pad.left-pad.right);
  const y = value => height-pad.bottom-value / maxValue * (height-pad.top-pad.bottom);
  let svg = `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true">`;
  for(let i=0;i<=4;i++) {
    const value = maxValue * i / 4;
    const py = y(value);
    svg += `<line x1="${pad.left}" y1="${py}" x2="${width-pad.right}" y2="${py}" stroke="#eaf0f2" stroke-width="1"/><text x="${pad.left-9}" y="${py+3}" text-anchor="end">${cap ? fmt(value,0) : fmt(value,1)}</text>`;
  }
  const ticks = width < 400 ? 3 : 4;
  for(let i=0;i<=ticks;i++) {
    const value = maxMinute * i / ticks;
    svg += `<text x="${x(value)}" y="${height-10}" text-anchor="${i===0 ? "start" : i===ticks ? "end" : "middle"}">${fmt(value/60,value < 120 ? 1 : 0)}h</text>`;
  }
  for(const s of series) {
    let path = "";
    data.forEach((point,i) => {
      if(i === 0) path += `M${x(point.minute)},${y(point[s.key])}`;
      else if(cap || s.key === "target") path += `H${x(point.minute)}V${y(point[s.key])}`;
      else path += `L${x(point.minute)},${y(point[s.key])}`;
    });
    svg += `<path class="chart-path" d="${path}" stroke="${s.color}"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`;
    if(data.length === 1) svg += `<circle cx="${x(data[0].minute)}" cy="${y(data[0][s.key])}" r="2.5" fill="${s.color}"/>`;
  }
  svg += `</svg><div class="chart-tooltip" hidden></div>`;
  container.innerHTML = svg;
  const tooltip = container.querySelector(".chart-tooltip");
  container.onpointermove = event => {
    const box = container.getBoundingClientRect();
    const px = event.clientX - box.left;
    const minute = Math.max(0, Math.min(maxMinute,(px-pad.left)/(width-pad.left-pad.right)*maxMinute));
    let nearest = data[0];
    for(const point of data) if(Math.abs(point.minute-minute) < Math.abs(nearest.minute-minute)) nearest = point;
    tooltip.innerHTML = `${fmt(nearest.minute,0)} min<br>${series.map(s => `${esc(s.label)}　${fmt(nearest[s.key],cap ? 0 : 2)}`).join("<br>")}`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(0,Math.min(width-tooltip.offsetWidth-4,px+10))}px`;
    tooltip.style.top = `${Math.max(0,Math.min(height-tooltip.offsetHeight,event.clientY-box.top-25))}px`;
  };
  container.onpointerleave = () => { tooltip.hidden = true; };
}

function renderCharts() {
  if(!state) return;
  chart("throughput-chart",[
    {key:"target",label:"Current target",color:colors.target,dash:"4 4"},
    {key:"reference",label:"Rolling reference",color:colors.reference,dash:"5 4"},
    {key:"throughput",label:"Actual throughput",color:colors.actual}
  ]);
  chart("cap-chart",[
    {key:"cap_a",label:"A cap",color:colors.a},
    {key:"cap_b",label:"B cap",color:colors.b},
    {key:"cap_assembly",label:"Assembly cap",color:colors.k}
  ],true);
}

$("play").addEventListener("click", async () => {
  if(playing) { await pauseAndWait(); return; }
  if(!state || state.done) return;
  playing = true;
  renderButtons();
  playStep();
});
$("step").addEventListener("click", async () => {
  try { await pauseAndWait(); await mutate("/api/step",{count:1}); } catch(error) { showError(error); }
});
$("reset").addEventListener("click", async () => {
  try {
    await pauseAndWait();
    const seed = Number($("seed").value);
    if(!Number.isSafeInteger(seed) || seed < 0 || seed > 2000000000) throw new Error("The random seed must be an integer from 0 to 2000000000.");
    conditionsDirty = false;
    await mutate("/api/reset",{seed,controller:$("strategy").value,schedule:$("schedule").checked});
    toast("Reset complete. The same seed uses the same random processing samples.");
  } catch(error) { showError(error); }
});
$("strategy").addEventListener("change", async event => {
  const controller = event.target.value;
  try { await pauseAndWait(); await mutate("/api/configure",{controller}); toast("Policy changed. Click Resume to continue."); }
  catch(error) { showError(error); if(state) render(); }
});
$("apply-caps").addEventListener("click", async () => {
  try {
    await pauseAndWait();
    const caps = [$("cap-a"),$("cap-b"),$("cap-k")].map(input => Number(input.value));
    if(caps.some(c => !Number.isInteger(c) || c < 0 || c > 8)) throw new Error("All three caps must be integers from 0 to 8.");
    await mutate("/api/configure",{controller:"manual",manual_caps:caps});
    toast("Targets applied. Caps will move toward them from the next decision.");
  } catch(error) { showError(error); }
});
for(const id of ["target","variability","slowdown"]) $(id).addEventListener("input",() => {
  conditionsDirty = true;
  $("slowdown-label").textContent = `${$("slowdown").value}%`;
});
$("apply-conditions").addEventListener("click", async () => {
  try {
    await pauseAndWait();
    const target = Number($("target").value);
    if(!Number.isInteger(target) || target < 1 || target > 20) throw new Error("Target throughput must be an integer from 1 to 20.");
    const payload = {target,gamma:Number($("variability").value),b_slowdown_percent:Number($("slowdown").value)};
    await mutate("/api/configure",payload);
    conditionsDirty = false;
    syncConditions();
    toast("Conditions updated. Jobs already in process retain their sampled processing times.");
  } catch(error) { showError(error); }
});
document.querySelectorAll(".nav-link").forEach(link => link.addEventListener("click",() => {
  document.querySelectorAll(".nav-link").forEach(item => item.classList.toggle("active",item===link));
}));
new ResizeObserver(renderCharts).observe($("throughput-chart"));
new ResizeObserver(renderCharts).observe($("cap-chart"));

(async function init() {
  try {
    state = await request("/api/state");
    $("seed").value = state.seed;
    $("schedule").checked = state.conditions.schedule_enabled;
    render();
  } catch(error) {
    $("connection").textContent = "Local simulator disconnected";
    document.querySelector(".connection-dot").classList.add("offline");
    showError(error);
  }
})();
