'use strict';

const KIND_LABELS = { R: '电阻', V: '电压源', I: '电流源' };
const PARAM_UNITS = { R: 'Ω', V: 'V', I: 'A' };
const OBS_PLACEHOLDERS = {
  voltage: '非接地节点编号，如 2',
  current: '元件名称，如 R1',
};

const EXAMPLE_NETLIST = [
  '# 示例：含电压源与电流源的直流电路',
  'V1 1 0 12',
  'R1 1 2 6',
  'R2 2 0 6',
  'R3 2 3 12',
  'I1 0 3 0.5',
].join('\n');

const netlistEl = document.getElementById('netlist');
const solveBtn = document.getElementById('solve-btn');
const exampleBtn = document.getElementById('example-btn');
const clearBtn = document.getElementById('clear-btn');
const errorEl = document.getElementById('error');
const resultsEl = document.getElementById('results');
const voltageTbody = document.querySelector('#voltage-table tbody');
const currentTbody = document.querySelector('#current-table tbody');
const obsTypeEl = document.getElementById('obs-type');
const obsTargetEl = document.getElementById('obs-target');
const sensBtn = document.getElementById('sens-btn');
const sensOutputEl = document.getElementById('sens-output');
const sensSummaryEl = document.getElementById('sens-summary');
const sensTbody = document.querySelector('#sens-table tbody');
const thNode1El = document.getElementById('th-node1');
const thNode2El = document.getElementById('th-node2');
const thLoadEl = document.getElementById('th-load');
const thBtn = document.getElementById('th-btn');
const thOutputEl = document.getElementById('th-output');
const thSummaryEl = document.getElementById('th-summary');
const thTbody = document.querySelector('#th-table tbody');
const measRowsEl = document.getElementById('meas-rows');
const measAddBtn = document.getElementById('meas-add');
const diagBtn = document.getElementById('diag-btn');
const diagOutputEl = document.getElementById('diag-output');
const diagSummaryEl = document.getElementById('diag-summary');
const diagTbody = document.querySelector('#diag-table tbody');
const diagDetailsEl = document.getElementById('diag-details');

// 数值格式化：去掉无意义的尾零，极大/极小值用科学计数法
function fmt(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return String(x);
  if (Math.abs(x) < 1e-12) return '0';
  const a = Math.abs(x);
  if (a >= 1e6 || a < 1e-4) return x.toExponential(4);
  return String(Number(x.toPrecision(6)));
}

// 在 fmt 基础上显式标注正负号
function fmtSigned(x) {
  const s = fmt(x);
  if (s === '0' || s.startsWith('-')) return s;
  return `+${s}`;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
  resultsEl.hidden = true;
  sensOutputEl.hidden = true;
  thOutputEl.hidden = true;
  diagOutputEl.hidden = true;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = '';
}

function addRow(tbody, cells) {
  const tr = document.createElement('tr');
  for (const { text, isNum, sign } of cells) {
    const td = document.createElement('td');
    td.textContent = text;
    if (isNum) td.className = 'num';
    if (sign > 0) td.classList.add('pos');
    else if (sign < 0) td.classList.add('neg');
    tr.appendChild(td);
  }
  tbody.appendChild(tr);
}

function renderResults(data) {
  voltageTbody.textContent = '';
  currentTbody.textContent = '';

  addRow(voltageTbody, [
    { text: '0（地）' },
    { text: '0', isNum: true },
  ]);
  for (const item of data.node_voltages) {
    addRow(voltageTbody, [
      { text: String(item.node) },
      { text: fmt(item.voltage), isNum: true },
    ]);
  }

  for (const item of data.currents) {
    addRow(currentTbody, [
      { text: item.name },
      { text: KIND_LABELS[item.kind] || item.kind },
      { text: `${item.from} → ${item.to}` },
      { text: fmt(item.current), isNum: true },
    ]);
  }

  resultsEl.hidden = false;
}

async function solve() {
  clearError();
  solveBtn.disabled = true;
  try {
    const resp = await fetch('/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ netlist: netlistEl.value }),
    });
    let data;
    try {
      data = await resp.json();
    } catch {
      showError(`服务器返回异常（HTTP ${resp.status}）`);
      return;
    }
    // 出错时仅提示原因，输入内容保留在文本框中
    if (!data.ok) {
      showError(data.error || '求解失败');
      return;
    }
    renderResults(data);
  } catch {
    showError('无法连接服务器，请检查服务是否正常运行');
  } finally {
    solveBtn.disabled = false;
  }
}

solveBtn.addEventListener('click', solve);
exampleBtn.addEventListener('click', () => {
  netlistEl.value = EXAMPLE_NETLIST;
  clearError();
});
clearBtn.addEventListener('click', () => {
  netlistEl.value = '';
  clearError();
  resultsEl.hidden = true;
  sensOutputEl.hidden = true;
  thOutputEl.hidden = true;
  diagOutputEl.hidden = true;
  netlistEl.focus();
});
netlistEl.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') solve();
});

// ---------- 灵敏度分析 ----------

function renderSensitivity(data) {
  sensTbody.textContent = '';
  const obs = data.observation;
  const label = obs.type === 'voltage'
    ? `节点 ${obs.node} 电压`
    : `元件 ${obs.name} 电流（${obs.from} → ${obs.to}）`;
  sensSummaryEl.textContent =
    `观测量：${label}，当前值 ${fmt(obs.value)} ${obs.unit}。` +
    '下表为观测量对每个元件数值的局部导数，以及该数值改为原值 1.01 倍时' +
    '观测量的一阶变化量（按变化量绝对值降序）。';

  for (const row of data.sensitivities) {
    addRow(sensTbody, [
      { text: row.name },
      { text: KIND_LABELS[row.kind] || row.kind },
      { text: `${fmt(row.value)} ${PARAM_UNITS[row.kind] || ''}`, isNum: true },
      {
        text: `${fmtSigned(row.derivative)} ${row.derivative_unit}`,
        isNum: true,
        sign: row.derivative,
      },
      {
        text: `${fmtSigned(row.delta)} ${row.delta_unit}`,
        isNum: true,
        sign: row.delta,
      },
    ]);
  }
  sensOutputEl.hidden = false;
}

async function sensitivity() {
  clearError();
  const target = obsTargetEl.value.trim();
  if (!target) {
    showError(obsTypeEl.value === 'voltage'
      ? '请填写要观测的非接地节点编号'
      : '请填写要观测电流的元件名称');
    return;
  }
  const observation = obsTypeEl.value === 'voltage'
    ? { type: 'voltage', node: target }
    : { type: 'current', element: target };

  sensBtn.disabled = true;
  try {
    const resp = await fetch('/sensitivity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ netlist: netlistEl.value, observation }),
    });
    let data;
    try {
      data = await resp.json();
    } catch {
      showError(`服务器返回异常（HTTP ${resp.status}）`);
      return;
    }
    // 出错时仅提示原因，网表与观测对象输入均保留
    if (!data.ok) {
      showError(data.error || '灵敏度分析失败');
      return;
    }
    renderSensitivity(data);
  } catch {
    showError('无法连接服务器，请检查服务是否正常运行');
  } finally {
    sensBtn.disabled = false;
  }
}

sensBtn.addEventListener('click', sensitivity);
obsTypeEl.addEventListener('change', () => {
  obsTargetEl.placeholder = OBS_PLACEHOLDERS[obsTypeEl.value] || '';
  obsTargetEl.focus();
});
obsTargetEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sensitivity();
});

// ---------- 两端口戴维南等效分析 ----------

function renderThevenin(data) {
  thTbody.textContent = '';
  const th = data.thevenin;
  thSummaryEl.textContent =
    `端口：节点 ${th.node_p}（+）→ 节点 ${th.node_n}（−），` +
    `负载电阻 RL = ${fmt(th.load)} Ω。` +
    '负载电流正方向为第一节点 → 第二节点（负值表示实际方向相反）；' +
    '交换两个端点时电压、电流变号，功率不变。';

  const rows = [
    { label: '开路电压 Voc', value: th.voc, unit: th.voc_unit, sign: true,
      note: `V(${th.node_p}) − V(${th.node_n})，端口开路时` },
    { label: '等效电阻 Rth', value: th.rth, unit: th.rth_unit, sign: false,
      note: '独立源置零后注入 1 A 试验电流求得；被理想电压源短接时为 0' },
    { label: '端口电压 VL', value: th.vl, unit: th.vl_unit, sign: true,
      note: '负载两端电压，第一节点侧为 +，VL = IL × RL' },
    { label: '负载电流 IL', value: th.il, unit: th.il_unit, sign: true,
      note: '正方向：第一节点 → 第二节点，IL = Voc / (Rth + RL)' },
    { label: '负载功率 PL', value: th.pl, unit: th.pl_unit, sign: false,
      note: 'PL = VL × IL，恒为非负' },
  ];
  for (const row of rows) {
    addRow(thTbody, [
      { text: row.label },
      {
        text: `${row.sign ? fmtSigned(row.value) : fmt(row.value)} ${row.unit}`,
        isNum: true,
        sign: row.sign ? row.value : 0,
      },
      { text: row.note },
    ]);
  }
  thOutputEl.hidden = false;
}

async function thevenin() {
  clearError();
  const node1 = thNode1El.value.trim();
  const node2 = thNode2El.value.trim();
  const load = thLoadEl.value.trim();
  if (!node1 || !node2) {
    showError('请填写两个不同的端口节点编号（允许节点 0）');
    return;
  }
  if (!load) {
    showError('请填写正的有限负载电阻（单位 Ω）');
    return;
  }

  thBtn.disabled = true;
  try {
    const resp = await fetch('/thevenin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ netlist: netlistEl.value, node1, node2, load }),
    });
    let data;
    try {
      data = await resp.json();
    } catch {
      showError(`服务器返回异常（HTTP ${resp.status}）`);
      return;
    }
    // 出错时仅提示原因，网表与端口输入均保留
    if (!data.ok) {
      showError(data.error || '戴维南等效分析失败');
      return;
    }
    renderThevenin(data);
  } catch {
    showError('无法连接服务器，请检查服务是否正常运行');
  } finally {
    thBtn.disabled = false;
  }
}

thBtn.addEventListener('click', thevenin);
for (const el of [thNode1El, thNode2El, thLoadEl]) {
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') thevenin();
  });
}

// ---------- 单电阻断路定位 ----------

const MAX_MEAS = 12;

function updateMeasButtons() {
  const rows = measRowsEl.querySelectorAll('.meas-row');
  measAddBtn.disabled = rows.length >= MAX_MEAS;
  for (const row of rows) {
    row.querySelector('.meas-remove').disabled = rows.length <= 1;
  }
}

function addMeasRow() {
  if (measRowsEl.querySelectorAll('.meas-row').length >= MAX_MEAS) return;
  const div = document.createElement('div');
  div.className = 'meas-row';
  div.innerHTML =
    '<label>节点 <input type="text" class="meas-node" inputmode="numeric" ' +
    'size="4" placeholder="如 2" autocomplete="off"></label>' +
    '<label>实测电压 (V) <input type="text" class="meas-voltage" ' +
    'inputmode="decimal" size="8" placeholder="如 5.0" autocomplete="off"></label>' +
    '<label>容差 (V) <input type="text" class="meas-tol" inputmode="decimal" ' +
    'size="6" placeholder="如 0.1" autocomplete="off"></label>' +
    '<button type="button" class="meas-remove" title="删除该测点">×</button>';
  div.querySelector('.meas-remove').addEventListener('click', () => {
    div.remove();
    updateMeasButtons();
  });
  measRowsEl.appendChild(div);
  updateMeasButtons();
}

measAddBtn.addEventListener('click', addMeasRow);
addMeasRow();  // 初始保留一行

function renderDiagnosis(data) {
  diagTbody.textContent = '';
  diagDetailsEl.textContent = '';

  const c = data.conclusion;
  if (c.type === 'normal') {
    const r = data.results.find((x) => x.hypothesis === c.hypothesis);
    diagSummaryEl.textContent =
      `结论：电路正常——仅「正常电路」假设与实测数据匹配` +
      `（得分 ${fmt(r.score)} ≤ 1）。`;
  } else if (c.type === 'open') {
    const r = data.results.find((x) => x.hypothesis === c.hypothesis);
    diagSummaryEl.textContent =
      `结论：电阻 ${c.element} 断路——仅「${c.hypothesis}」假设与实测数据匹配` +
      `（得分 ${fmt(r.score)} ≤ 1）。`;
  } else if (c.type === 'ambiguous') {
    diagSummaryEl.textContent =
      `结论：无法区分——以下 ${c.candidates.length} 个假设均与实测数据匹配` +
      `（得分均 ≤ 1）：${c.candidates.join('、')}。` +
      '可增加测点或收紧容差以进一步区分。';
  } else {
    diagSummaryEl.textContent =
      '结论：没有任何假设与实测数据匹配（所有可求解假设的得分均 > 1），' +
      '实测数据不符合「正常或单电阻断路」模型。';
  }

  for (const r of data.results) {
    if (r.solvable) {
      addRow(diagTbody, [
        { text: r.hypothesis },
        { text: fmt(r.score), isNum: true },
        {
          text: r.matched ? '匹配（≤ 1）' : '不匹配（> 1）',
          sign: r.matched ? 1 : -1,
        },
        { text: '各测点 |误差|÷容差 的最大值' },
      ]);
    } else {
      addRow(diagTbody, [
        { text: r.hypothesis },
        { text: '—', isNum: true },
        { text: '无法判定' },
        { text: r.reason },
      ]);
    }
  }

  for (const r of data.results) {
    if (!r.solvable) continue;
    const h = document.createElement('h3');
    h.textContent = `假设「${r.hypothesis}」的测点预测（得分 ${fmt(r.score)}）`;
    diagDetailsEl.appendChild(h);
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const label of ['节点', '实测 (V)', '预测 (V)', '误差 (V)',
                         '容差 (V)', '|误差|÷容差']) {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    for (const p of r.predictions) {
      addRow(tbody, [
        { text: String(p.node) },
        { text: fmt(p.measured), isNum: true },
        { text: fmt(p.predicted), isNum: true },
        { text: fmtSigned(p.error), isNum: true, sign: p.error },
        { text: fmt(p.tolerance), isNum: true },
        { text: fmt(p.ratio), isNum: true, sign: p.ratio > 1 ? -1 : 0 },
      ]);
    }
    table.appendChild(tbody);
    diagDetailsEl.appendChild(table);
  }
  diagOutputEl.hidden = false;
}

async function diagnose() {
  clearError();
  const rows = [...measRowsEl.querySelectorAll('.meas-row')];
  if (rows.length === 0) {
    showError('请至少添加 1 条测点');
    return;
  }
  const measurements = [];
  for (const row of rows) {
    const node = row.querySelector('.meas-node').value.trim();
    const voltage = row.querySelector('.meas-voltage').value.trim();
    const tolerance = row.querySelector('.meas-tol').value.trim();
    if (!node || !voltage || !tolerance) {
      showError('请完整填写每条测点的节点、实测电压与容差（或删除多余测点行）');
      return;
    }
    measurements.push({ node, voltage, tolerance });
  }

  diagBtn.disabled = true;
  try {
    const resp = await fetch('/diagnose-open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ netlist: netlistEl.value, measurements }),
    });
    let data;
    try {
      data = await resp.json();
    } catch {
      showError(`服务器返回异常（HTTP ${resp.status}）`);
      return;
    }
    // 出错时仅提示原因，网表与全部测点输入均保留
    if (!data.ok) {
      showError(data.error || '断路定位失败');
      return;
    }
    renderDiagnosis(data);
  } catch {
    showError('无法连接服务器，请检查服务是否正常运行');
  } finally {
    diagBtn.disabled = false;
  }
}

diagBtn.addEventListener('click', diagnose);
