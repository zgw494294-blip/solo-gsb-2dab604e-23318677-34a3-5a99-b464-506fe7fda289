"""直流电路网表的解析、校验与改进节点分析（MNA）求解。

网表格式（每行一个元件，# 开头为注释）：
    名称 起点 终点 数值
    R 开头：电阻，单位欧姆，数值必须为正
    V 开头：理想电压源，单位伏特，规定 起点电位 - 终点电位 = 数值
    I 开头：理想电流源，单位安培，正方向为起点 -> 终点
节点 0 为接地节点。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

GROUND = 0              # 接地节点编号
MAX_NODES = 12          # 非接地节点数上限
MAX_ELEMENTS = 30       # 元件数上限
MAX_NODE_ID = 1_000_000  # 节点编号上限（防止无意义输入）
MAX_NAME_LEN = 16       # 元件名称长度上限

_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
VALID_KINDS = ('R', 'V', 'I')


class NetlistError(Exception):
    """网表格式或数值非法。"""


class NoUniqueSolutionError(Exception):
    """电路约束导致不存在唯一解。"""


@dataclass
class Element:
    name: str
    kind: str   # 'R' / 'V' / 'I'
    n1: int     # 起点
    n2: int     # 终点
    value: float


def _parse_node(token: str, lineno: int) -> int:
    try:
        node = int(token, 10)
    except ValueError:
        raise NetlistError(
            f'第 {lineno} 行：节点编号 "{token}" 非法，应为非负整数') from None
    if node < 0:
        raise NetlistError(f'第 {lineno} 行：节点编号 {node} 非法，应为非负整数')
    if node > MAX_NODE_ID:
        raise NetlistError(f'第 {lineno} 行：节点编号 {node} 过大（上限 {MAX_NODE_ID}）')
    return node


def parse_netlist(text: str) -> tuple[list[Element], list[int]]:
    """解析并校验网表，返回 (元件列表, 排序后的非接地节点列表)。"""
    elements: list[Element] = []
    seen: dict[str, int] = {}  # 小写名称 -> 首次出现的行号

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise NetlistError(
                f'第 {lineno} 行：每行应为 4 个字段「名称 起点 终点 数值」，'
                f'实际为 {len(parts)} 个')
        name, tok_n1, tok_n2, tok_val = parts

        if len(name) > MAX_NAME_LEN or not _NAME_RE.match(name):
            raise NetlistError(
                f'第 {lineno} 行：名称 "{name}" 非法，应以字母开头，'
                f'仅含字母、数字、下划线，且不超过 {MAX_NAME_LEN} 个字符')
        kind = name[0].upper()
        if kind not in VALID_KINDS:
            raise NetlistError(
                f'第 {lineno} 行：名称 "{name}" 应以 R（电阻）、V（电压源）'
                f'或 I（电流源）开头')
        key = name.lower()
        if key in seen:
            raise NetlistError(
                f'第 {lineno} 行：名称 "{name}" 与第 {seen[key]} 行的元件重复')
        seen[key] = lineno

        n1 = _parse_node(tok_n1, lineno)
        n2 = _parse_node(tok_n2, lineno)
        if n1 == n2:
            raise NetlistError(
                f'第 {lineno} 行：{name} 的起点与终点相同（均为节点 {n1}），不允许')

        try:
            value = float(tok_val)
        except ValueError:
            raise NetlistError(
                f'第 {lineno} 行：数值 "{tok_val}" 不是合法的浮点数') from None
        if not math.isfinite(value):
            raise NetlistError(f'第 {lineno} 行：数值 "{tok_val}" 必须为有限值')
        if kind == 'R' and value <= 0:
            raise NetlistError(
                f'第 {lineno} 行：电阻 {name} 的阻值必须为正数（当前为 {value}）')

        elements.append(Element(name=name, kind=kind, n1=n1, n2=n2, value=value))

    if not elements:
        raise NetlistError('网表为空：请至少输入一个元件')
    if len(elements) > MAX_ELEMENTS:
        raise NetlistError(
            f'元件数量为 {len(elements)} 个，超过上限 {MAX_ELEMENTS} 个')

    nodes = sorted({n for e in elements for n in (e.n1, e.n2)} - {GROUND})
    if len(nodes) > MAX_NODES:
        raise NetlistError(
            f'非接地节点数量为 {len(nodes)} 个，超过上限 {MAX_NODES} 个')
    return elements, nodes


class _UnionFind:
    """并查集，用于电路拓扑检查。"""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


def _check_unique_solution(elements: list[Element], nodes: list[int]) -> None:
    """MNA 方程组有唯一解的拓扑充要条件：

    1. 每个节点都能经电阻/电压源到达地（否则电位悬浮，或存在电流源割集）；
    2. 电压源（含经地）不构成回路（否则 KVL 矛盾或回路电流不确定）。
    """
    uf = _UnionFind()
    for e in elements:
        if e.kind in ('R', 'V'):
            uf.union(e.n1, e.n2)
    floating = [nd for nd in nodes if not uf.connected(nd, GROUND)]
    if floating:
        raise NoUniqueSolutionError(
            '无唯一解：节点 {} 未通过电阻或电压源与地（节点 0）连通，'
            '电位悬浮无法确定（也可能存在仅由电流源构成的割集）'.format(
                '、'.join(map(str, floating))))

    uf_v = _UnionFind()
    for e in elements:
        if e.kind != 'V':
            continue
        if uf_v.connected(e.n1, e.n2):
            raise NoUniqueSolutionError(
                f'无唯一解：电压源 {e.name}（{e.n1}→{e.n2}）与其他电压源构成回路，'
                'KVL 相互矛盾或回路电流无法唯一确定')
        uf_v.union(e.n1, e.n2)


def _assemble_mna(elements: list[Element], nodes: list[int]):
    """组装 MNA 方程组 A x = z。

    返回 (A, z, index, vsources)：index 为非接地节点 -> 未知量序号的映射，
    vsources 为电压源列表（其 起点->终点 电流为附加未知量）。
    """
    index = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    vsources = [e for e in elements if e.kind == 'V']
    m = len(vsources)
    size = n + m
    A = [[0.0] * size for _ in range(size)]
    z = [0.0] * size

    # 电阻与电流源盖章
    for e in elements:
        a = index.get(e.n1)  # 接地节点不在 index 中，视为 None
        b = index.get(e.n2)
        if e.kind == 'R':
            g = 1.0 / e.value
            if a is not None:
                A[a][a] += g
            if b is not None:
                A[b][b] += g
            if a is not None and b is not None:
                A[a][b] -= g
                A[b][a] -= g
        elif e.kind == 'I':
            # 电流由起点流向终点：起点流出、终点注入
            if a is not None:
                z[a] -= e.value
            if b is not None:
                z[b] += e.value

    # 电压源盖章：附加未知量为其 起点->终点 的电流
    for k, e in enumerate(vsources):
        row = n + k
        a = index.get(e.n1)
        b = index.get(e.n2)
        if a is not None:
            A[a][row] += 1.0
            A[row][a] += 1.0
        if b is not None:
            A[b][row] -= 1.0
            A[row][b] -= 1.0
        z[row] = e.value  # V(起点) - V(终点) = 数值

    return A, z, index, vsources


def _solve_mna(elements: list[Element], nodes: list[int]):
    """组装并求解 MNA 方程，返回 (x, A, index, vsources)。"""
    A, z, index, vsources = _assemble_mna(elements, nodes)

    _check_unique_solution(elements, nodes)

    size = len(z)
    a_mat = np.array(A, dtype=float)
    b_vec = np.array(z, dtype=float)
    try:
        x = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError as exc:
        raise NoUniqueSolutionError(
            '无唯一解：电路方程组奇异（可能存在仅由电流源构成的割集等约束冲突）'
        ) from exc
    x = [float(v) for v in x]

    # 残差校验，防止病态矩阵给出不可靠结果
    residual = max(
        (abs(sum(A[i][j] * x[j] for j in range(size)) - z[i])
         for i in range(size)),
        default=0.0)
    scale = max(1.0, max((abs(v) for v in z), default=0.0))
    if any(not math.isfinite(v) for v in x) or residual > 1e-6 * scale:
        raise NoUniqueSolutionError('数值求解失败：电路可能无唯一解或严重病态')

    return x, A, index, vsources


def solve_circuit(elements: list[Element],
                  nodes: list[int]) -> tuple[list[dict], list[dict]]:
    """用改进节点分析求解，返回 (节点电压列表, 元件电流列表)。

    元件电流的正方向均为 起点 -> 终点。
    """
    x, _, index, vsources = _solve_mna(elements, nodes)
    n = len(nodes)

    voltages = [{'node': nd, 'voltage': x[index[nd]]} for nd in nodes]

    currents = []
    k = 0
    for e in elements:
        v1 = x[index[e.n1]] if e.n1 in index else 0.0
        v2 = x[index[e.n2]] if e.n2 in index else 0.0
        if e.kind == 'R':
            cur = (v1 - v2) / e.value
        elif e.kind == 'I':
            cur = e.value
        else:  # 'V'：MNA 附加未知量即 起点->终点 电流
            cur = x[n + k]
            k += 1
        currents.append({
            'name': e.name,
            'kind': e.kind,
            'from': e.n1,
            'to': e.n2,
            'current': cur,
        })
    return voltages, currents


# 元件数值的单位（用于灵敏度结果的单位标注）
PARAM_UNITS = {'R': 'Ω', 'V': 'V', 'I': 'A'}


def compute_sensitivity(elements: list[Element], nodes: list[int],
                        obs_type: str, obs_target):
    """对当前 MNA 方程求导，计算观测量对每个元件数值的局部灵敏度。

    obs_type 为 'voltage'（观测非接地节点电压）或 'current'（观测元件
    起点->终点 电流）；obs_target 相应为节点编号（int）或元件名称（str，
    不区分大小写）。

    对元件数值 p：A·(dx/dp) = dz/dp − (dA/dp)·x，其中电阻只改变导纳
    矩阵 A（g = 1/R，dg/dR = −1/R²），电压源与电流源只改变右端 z。

    返回 (observation, rows)：observation 为观测量描述及当前值；
    rows 每项为某元件的导数与「数值改为原值 1.01 倍」的一阶变化量
    （= 导数 × 0.01 × 原值），已按变化量绝对值降序排列。
    """
    # 先确认观测对象存在，再求解电路
    obs_element = None
    if obs_type == 'voltage':
        if obs_target == GROUND:
            raise NetlistError('节点 0 为接地节点，电压恒为 0，不能作为观测量')
        if obs_target not in nodes:
            raise NetlistError(
                f'观测节点 {obs_target} 不存在：当前网表的非接地节点为 '
                f'{"、".join(map(str, nodes))}')
    elif obs_type == 'current':
        obs_element = next(
            (e for e in elements
             if e.name.lower() == str(obs_target).strip().lower()), None)
        if obs_element is None:
            names = '、'.join(e.name for e in elements)
            raise NetlistError(
                f'网表中不存在名为 "{obs_target}" 的元件，无法观测其电流'
                f'（现有元件：{names}）')
    else:
        raise NetlistError(f'未知观测类型 "{obs_type}"')

    x, A, index, vsources = _solve_mna(elements, nodes)
    n = len(nodes)
    size = n + len(vsources)
    vsrc_pos = {id(e): k for k, e in enumerate(vsources)}

    def xv(node: int) -> float:
        return x[index[node]] if node in index else 0.0

    # 观测量的当前值与单位
    if obs_type == 'voltage':
        obs_unit = 'V'
        observation = {'type': 'voltage', 'node': obs_target,
                       'value': xv(obs_target), 'unit': obs_unit}
    else:
        obs_unit = 'A'
        if obs_element.kind == 'R':
            obs_value = (xv(obs_element.n1) - xv(obs_element.n2)) / obs_element.value
        elif obs_element.kind == 'I':
            obs_value = obs_element.value
        else:  # 'V'
            obs_value = x[n + vsrc_pos[id(obs_element)]]
        observation = {'type': 'current', 'name': obs_element.name,
                       'from': obs_element.n1, 'to': obs_element.n2,
                       'value': obs_value, 'unit': obs_unit}

    a_mat = np.array(A, dtype=float)
    rows = []
    for p in elements:
        # 右端 rhs = dz/dp − (dA/dp)·x
        rhs = [0.0] * size
        if p.kind == 'R':
            a = index.get(p.n1)
            b = index.get(p.n2)
            dg = -1.0 / (p.value * p.value)
            va = x[a] if a is not None else 0.0
            vb = x[b] if b is not None else 0.0
            if a is not None:
                rhs[a] -= dg * (va - vb)
            if b is not None:
                rhs[b] -= dg * (vb - va)
        elif p.kind == 'V':
            # 电压源数值只出现在自己的约束行右端
            rhs[n + vsrc_pos[id(p)]] += 1.0
        else:  # 'I'：z[起点] -= J，z[终点] += J
            a = index.get(p.n1)
            b = index.get(p.n2)
            if a is not None:
                rhs[a] -= 1.0
            if b is not None:
                rhs[b] += 1.0

        dx = np.linalg.solve(a_mat, np.array(rhs, dtype=float))
        dx = [float(v) for v in dx]

        if obs_type == 'voltage':
            deriv = dx[index[obs_target]]
        elif obs_element.kind == 'R':
            a = index.get(obs_element.n1)
            b = index.get(obs_element.n2)
            dva = dx[a] if a is not None else 0.0
            dvb = dx[b] if b is not None else 0.0
            deriv = (dva - dvb) / obs_element.value
            if p is obs_element:
                # 计入自身阻值的直接影响：i = (v1 − v2)/R 的分母含 R
                deriv -= ((xv(obs_element.n1) - xv(obs_element.n2))
                          / (obs_element.value * obs_element.value))
        elif obs_element.kind == 'I':
            # 电流源电流即其数值本身
            deriv = 1.0 if p is obs_element else 0.0
        else:  # 观测电压源电流：MNA 附加未知量
            deriv = dx[n + vsrc_pos[id(obs_element)]]

        # 数值改为原值 1.01 倍时观测量的一阶变化量（零值电源该量为 0）
        delta = deriv * 0.01 * p.value
        rows.append({
            'name': p.name,
            'kind': p.kind,
            'value': p.value,
            'derivative': deriv,
            'derivative_unit': f'{obs_unit}/{PARAM_UNITS[p.kind]}',
            'delta': delta,
            'delta_unit': obs_unit,
        })

    rows.sort(key=lambda r: -abs(r['delta']))
    return observation, rows


def compute_thevenin(elements: list[Element], nodes: list[int],
                     node_p: int, node_n: int, load: float) -> dict:
    """对指定的两个节点做两端口戴维南等效分析。

    端口按 第一节点 node_p（+）到 第二节点 node_n（−）定义：

    - 开路电压 Voc = V(node_p) − V(node_n)，由原电路求解；
    - 等效电阻 Rth：独立源右端置零（电压源短路、电流源开路，保留理想电压源
      的零伏约束行），从第二节点向第一节点注入 1 A 试验电流，再解同一 MNA
      系数矩阵，Rth = V(node_p) − V(node_n)；端口被理想电压源短接时为 0 Ω；
    - 负载（R_L > 0）接在两端口之间，电流正方向为 node_p -> node_n：
      I_L = Voc / (Rth + R_L)，V_L = I_L · R_L，P_L = V_L · I_L。
    """
    all_nodes = set(nodes) | {GROUND}
    missing = [nd for nd in (node_p, node_n) if nd not in all_nodes]
    if missing:
        exist = '、'.join(map(str, sorted(all_nodes)))
        raise NetlistError(
            f'节点 {"、".join(map(str, missing))} 不存在：当前网表中的节点为 {exist}')
    if node_p == node_n:
        raise NetlistError(
            f'两个端口端点不能相同（均为节点 {node_p}），请选择两个不同的节点')
    try:
        load = float(load)
    except (TypeError, ValueError):
        raise NetlistError(
            f'负载电阻 "{load}" 不是合法的数值，应为正的有限浮点数') from None
    if not math.isfinite(load) or load <= 0:
        raise NetlistError(
            f'负载电阻必须为正的有限值（当前为 {load}）')

    # 开路电压：直接求原电路的节点解
    x, A, index, _ = _solve_mna(elements, nodes)

    def xv(node: int) -> float:
        return x[index[node]] if node in index else 0.0

    voc = xv(node_p) - xv(node_n)

    # 等效电阻：独立源置零（z=0，电压源保留零伏约束行），从第二节点向
    # 第一节点注入 1 A：电流在 node_p 注入、从 node_n 抽出
    size = len(x)
    z_test = [0.0] * size
    if node_p in index:
        z_test[index[node_p]] += 1.0
    if node_n in index:
        z_test[index[node_n]] -= 1.0

    a_mat = np.array(A, dtype=float)
    try:
        xt = np.linalg.solve(a_mat, np.array(z_test, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise NoUniqueSolutionError(
            '无唯一解：独立源置零后的等效电路方程组奇异，无法计算等效电阻'
        ) from exc
    xt = [float(v) for v in xt]

    residual = max(
        (abs(sum(A[i][j] * xt[j] for j in range(size)) - z_test[i])
         for i in range(size)),
        default=0.0)
    if any(not math.isfinite(v) for v in xt) or residual > 1e-6:
        raise NoUniqueSolutionError('数值求解失败：等效电阻电路可能严重病态')

    def xvt(node: int) -> float:
        return xt[index[node]] if node in index else 0.0

    rth = xvt(node_p) - xvt(node_n)
    # 理想电压源短接端口时应为 0，钳掉浮点残差造成的微小负值
    if -1e-9 < rth < 0:
        rth = 0.0

    il = voc / (rth + load)          # 正方向：第一节点 -> 第二节点
    vl = il * load
    pl = vl * il

    return {
        'node_p': node_p,
        'node_n': node_n,
        'voc': voc, 'voc_unit': 'V',
        'rth': rth, 'rth_unit': 'Ω',
        'load': load, 'load_unit': 'Ω',
        'vl': vl, 'vl_unit': 'V',
        'il': il, 'il_unit': 'A',
        'pl': pl, 'pl_unit': 'W',
    }


# 单电阻断路定位的测点数量上限（与非接地节点数上限一致）
MAX_MEASUREMENTS = MAX_NODES


def _validate_measurements(measurements: list[dict],
                           nodes: list[int]) -> list[dict]:
    """校验测点列表，返回规范化后的 [{'node', 'voltage', 'tolerance'}]。"""
    if not 1 <= len(measurements) <= MAX_MEASUREMENTS:
        raise NetlistError(
            f'测点数量为 {len(measurements)} 条，'
            f'应为 1～{MAX_MEASUREMENTS} 条不重复的非接地节点实测电压')
    seen = set()
    checked = []
    for i, m in enumerate(measurements, 1):
        node, voltage, tol = m['node'], m['voltage'], m['tolerance']
        if node == GROUND:
            raise NetlistError(
                f'第 {i} 条测点：节点 0 为接地节点，电压恒为 0，不能作为测点')
        if node not in nodes:
            raise NetlistError(
                f'第 {i} 条测点：节点 {node} 不存在：当前网表的非接地节点为 '
                f'{"、".join(map(str, nodes))}')
        if node in seen:
            raise NetlistError(f'第 {i} 条测点：节点 {node} 与前面的测点重复')
        seen.add(node)
        if not math.isfinite(voltage):
            raise NetlistError(
                f'第 {i} 条测点：实测电压必须为有限值（当前为 {voltage}，单位 V）')
        if not math.isfinite(tol) or tol <= 0:
            raise NetlistError(
                f'第 {i} 条测点：容差必须为正的有限值（当前为 {tol}，单位 V）')
        checked.append({'node': node, 'voltage': voltage, 'tolerance': tol})
    return checked


def diagnose_open(elements: list[Element], nodes: list[int],
                  measurements: list[dict]) -> dict:
    """单电阻断路定位：枚举「正常电路」与「每个电阻单独断路」的假设。

    measurements 为 [{'node': 非接地节点, 'voltage': 实测电压,
    'tolerance': 绝对容差}]（1～12 条、节点不重复、电压有限、容差为正的
    有限值，单位均为 V）。

    断路即删除该电阻支路，保留原节点集合重新组装 MNA（不以大电阻近似）。
    正常电路必须有唯一解，否则抛出 NoUniqueSolutionError；断路后无唯一解的
    假设记录原因（solvable=False）后继续检查其余假设。

    每个可求解假设的得分为各测点 |预测 − 实测| ÷ 容差 的最大值，≤ 1 为匹配。
    返回 {'results', 'matches', 'conclusion'}：results 中可求解假设按
    （得分, 假设名称）排序，无唯一解的假设按名称排在最后；conclusion 的
    type 为 normal（仅正常电路匹配）/ open（仅某电阻断路匹配）/
    ambiguous（多个匹配，无法区分）/ none（无匹配，不符合单断路模型）。
    """
    checked = _validate_measurements(measurements, nodes)

    # 正常电路必须有唯一解，否则诊断前提不成立
    _solve_mna(elements, nodes)

    hypotheses = [('正常电路', None, elements)]
    for r in elements:
        if r.kind == 'R':
            hypotheses.append(
                (f'{r.name} 断路', r.name, [e for e in elements if e is not r]))

    solvable = []
    unsolvable = []
    for name, open_element, variant in hypotheses:
        try:
            x, _, index, _ = _solve_mna(variant, nodes)
        except NoUniqueSolutionError as exc:
            unsolvable.append({
                'hypothesis': name, 'element': open_element,
                'solvable': False, 'reason': str(exc),
            })
            continue
        predictions = []
        score = 0.0
        for m in checked:
            predicted = x[index[m['node']]]
            error = predicted - m['voltage']
            ratio = abs(error) / m['tolerance']
            score = max(score, ratio)
            predictions.append({
                'node': m['node'],
                'measured': m['voltage'],
                'predicted': predicted,
                'error': error,
                'tolerance': m['tolerance'],
                'ratio': ratio,
            })
        solvable.append({
            'hypothesis': name, 'element': open_element,
            'solvable': True, 'predictions': predictions,
            'score': score, 'matched': score <= 1.0,
        })

    solvable.sort(key=lambda r: (r['score'], r['hypothesis']))
    unsolvable.sort(key=lambda r: r['hypothesis'])
    matched = [r for r in solvable if r['matched']]

    if len(matched) == 1:
        only = matched[0]
        conclusion = {
            'type': 'normal' if only['element'] is None else 'open',
            'hypothesis': only['hypothesis'],
            'element': only['element'],
        }
    elif matched:
        conclusion = {
            'type': 'ambiguous',
            'candidates': [r['hypothesis'] for r in matched],
        }
    else:
        conclusion = {'type': 'none'}

    return {
        'results': solvable + unsolvable,
        'matches': [r['hypothesis'] for r in matched],
        'conclusion': conclusion,
    }
