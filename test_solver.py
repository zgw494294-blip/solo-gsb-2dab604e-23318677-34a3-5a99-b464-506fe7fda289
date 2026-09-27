"""solver.py 的单元测试。

本环境未安装 NumPy，因此注入一个纯 Python 的最小 numpy 替身，
仅实现 solver.py 用到的接口（array / linalg.solve / linalg.LinAlgError）。
在装有真实 NumPy 的环境中运行时，替身同样能保证接口语义一致。
"""

import sys
import types


# ---------- 最小 numpy 替身 ----------
class _LinAlgError(Exception):
    pass


def _gauss_solve(A, b):
    """高斯消元（部分主元），奇异时抛出 _LinAlgError。"""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            raise _LinAlgError('Singular matrix')
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _fake_array(obj, dtype=None):
    if obj and isinstance(obj[0], (list, tuple)):
        return [list(row) for row in obj]
    return list(obj)


fake_np = types.ModuleType('numpy')
fake_np.array = _fake_array
fake_np.linalg = types.SimpleNamespace(solve=_gauss_solve, LinAlgError=_LinAlgError)
sys.modules['numpy'] = fake_np

import solver  # noqa: E402
from solver import (NetlistError, NoUniqueSolutionError, compute_sensitivity,  # noqa: E402
                    compute_thevenin, diagnose_open, parse_netlist,
                    solve_circuit)

# ---------- 测试工具 ----------
PASSED = 0
FAILED = 0


def check(cond, label):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f'  ok  {label}')
    else:
        FAILED += 1
        print(f'FAIL  {label}')


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def solve_text(text):
    elements, nodes = parse_netlist(text)
    return solve_circuit(elements, nodes)


def currents_by_name(currents):
    return {c['name']: c['current'] for c in currents}


def expect_error(text, exc_type, keyword, label):
    try:
        solve_text(text)
    except exc_type as exc:
        check(keyword in str(exc), f'{label}（提示包含「{keyword}」）-> {exc}')
        return
    except Exception as exc:  # noqa: BLE001
        check(False, f'{label}（抛出了错误类型 {type(exc).__name__}: {exc}）')
        return
    check(False, f'{label}（未抛出 {exc_type.__name__}）')


# ---------- 正常求解 ----------
print('== 正常求解 ==')

# 分压电路：V(1)=10, V(2)=5；R1、R2 电流均为 1A；V1 电流(1->0)为 -1A
v, c = solve_text('V1 1 0 10\nR1 1 2 5\nR2 2 0 5')
volt = {item['node']: item['voltage'] for item in v}
cur = currents_by_name(c)
check(close(volt[1], 10) and close(volt[2], 5), '分压电路节点电压')
check(close(cur['R1'], 1) and close(cur['R2'], 1), '分压电路电阻电流')
check(close(cur['V1'], -1), '电压源电流方向（起点->终点为 -1A）')

# 电流源：2A 从地注入节点 1，经 4Ω 电阻到地，V(1)=8V
v, c = solve_text('I1 0 1 2\nR1 1 0 4')
volt = {item['node']: item['voltage'] for item in v}
cur = currents_by_name(c)
check(close(volt[1], 8), '电流源电路节点电压')
check(close(cur['R1'], 2) and close(cur['I1'], 2), '电流源电路元件电流')

# 混合电路（手算：V=[12, 7.5, 13.5]，i_V1=-0.75，i_R3=-0.5）
v, c = solve_text('V1 1 0 12\nR1 1 2 6\nR2 2 0 6\nR3 2 3 12\nI1 0 3 0.5')
volt = {item['node']: item['voltage'] for item in v}
cur = currents_by_name(c)
check(close(volt[1], 12) and close(volt[2], 7.5) and close(volt[3], 13.5),
      '混合电路节点电压')
check(close(cur['V1'], -0.75) and close(cur['R1'], 0.75)
      and close(cur['R2'], 1.25) and close(cur['R3'], -0.5)
      and close(cur['I1'], 0.5), '混合电路元件电流')

# 0V 电压源相当于短路/电流表：V(1)=0，I1 的 1A 全部经 V1 流回地
v, c = solve_text('V1 1 0 0\nR1 1 0 5\nI1 0 1 1')
volt = {item['node']: item['voltage'] for item in v}
cur = currents_by_name(c)
check(close(volt[1], 0) and close(cur['V1'], 1) and close(cur['R1'], 0),
      '0V 电压源作电流表')

# 注释、空行、大小写、负值
v, c = solve_text('# 注释行\n\nr1 1 0 8\nv1 1 0 -4\ni2 0 1 -0.5\n')
volt = {item['node']: item['voltage'] for item in v}
cur = currents_by_name(c)
check(close(volt[1], -4), '小写名称与负电压源')
check(close(cur['r1'], -0.5) and close(cur['i2'], -0.5) and close(cur['v1'], 0),
      '负电流源方向')

# 电流输出顺序与输入一致
_, c = solve_text('R2 1 0 4\nR1 1 0 4\nV9 1 0 8')
check([x['name'] for x in c] == ['R2', 'R1', 'V9'], '电流结果顺序与输入一致')

# ---------- 格式 / 数值非法 ----------
print('== 格式与数值校验 ==')
expect_error('', NetlistError, '网表为空', '空网表')
expect_error('# 只有注释\n', NetlistError, '网表为空', '仅注释')
expect_error('R1 1 0', NetlistError, '4 个字段', '字段不足')
expect_error('R1 1 0 5 extra', NetlistError, '4 个字段', '字段过多')
expect_error('X1 1 0 5', NetlistError, 'R（电阻）、V（电压源）或 I（电流源）', '非法首字母')
expect_error('1R 1 0 5', NetlistError, '名称', '数字开头的名称')
expect_error('R1 x 0 5', NetlistError, '节点编号', '非法节点')
expect_error('R1 -1 0 5', NetlistError, '非负整数', '负节点编号')
expect_error('R1 1 1 5', NetlistError, '起点与终点相同', '起点终点相同')
expect_error('R1 1 0 abc', NetlistError, '浮点数', '非法数值')
expect_error('R1 1 0 1e999', NetlistError, '有限值', '无穷大数值')
expect_error('R1 1 0 nan', NetlistError, '有限值', 'NaN 数值')
expect_error('R1 1 0 0', NetlistError, '正数', '零电阻')
expect_error('R1 1 0 -5', NetlistError, '正数', '负电阻')
expect_error('R1 1 0 5\nR1 1 0 6', NetlistError, '重复', '名称重复')
expect_error('R1 1 0 5\nr1 0 1 6', NetlistError, '重复', '名称重复（大小写）')
expect_error('\n'.join(f'R{i} 1 0 {i+1}' for i in range(1, 32)),
             NetlistError, '30', '元件数超过 30')
expect_error('\n'.join(f'R{i} {i} 0 1' for i in range(1, 14)),
             NetlistError, '12', '非接地节点超过 12')

# ---------- 无唯一解 ----------
print('== 无唯一解 ==')
expect_error('R1 1 2 5\nV1 3 0 1', NoUniqueSolutionError, '悬浮', '浮动节点')
expect_error('I1 1 0 1', NoUniqueSolutionError, '悬浮', '电流源割集（悬浮）')
expect_error('R1 1 0 5\nI1 1 2 1\nI2 2 1 1', NoUniqueSolutionError, '悬浮',
             '部分节点悬浮')
expect_error('V1 1 0 5\nV2 1 0 3', NoUniqueSolutionError, '回路', '电压源并联矛盾')
expect_error('V1 1 2 5\nV2 2 3 3\nV3 1 3 8\nR1 1 0 10',
             NoUniqueSolutionError, '回路', '电压源回路')

# ---------- 灵敏度分析 ----------
print('== 灵敏度分析 ==')


def sens_text(text, obs_type, target):
    elements, nodes = parse_netlist(text)
    return compute_sensitivity(elements, nodes, obs_type, target)


def sens_by_name(rows):
    return {r['name']: r for r in rows}


# 分压电路：V(2) = 10·R2/(R1+R2) = 5
# dV2/dR1 = -0.5 V/Ω，dV2/dR2 = +0.5 V/Ω，dV2/dV1 = +0.5 V/V
DIV = 'V1 1 0 10\nR1 1 2 5\nR2 2 0 5'
obs, rows = sens_text(DIV, 'voltage', 2)
d = sens_by_name(rows)
check(obs['type'] == 'voltage' and obs['node'] == 2
      and close(obs['value'], 5) and obs['unit'] == 'V',
      '灵敏度：观测量（节点电压）当前值')
check(close(d['R1']['derivative'], -0.5) and close(d['R2']['derivative'], 0.5)
      and close(d['V1']['derivative'], 0.5), '灵敏度：节点电压对各元件导数')
check(close(d['R1']['delta'], -0.025) and close(d['R2']['delta'], 0.025)
      and close(d['V1']['delta'], 0.05), '灵敏度：1% 变化量 = 导数 × 0.01 × 原值')
check([r['name'] for r in rows] == ['V1', 'R1', 'R2'],
      '灵敏度：结果按变化量绝对值降序')
check(d['R1']['derivative_unit'] == 'V/Ω' and d['V1']['derivative_unit'] == 'V/V'
      and d['R1']['delta_unit'] == 'V', '灵敏度：导数与变化量单位标注')

# 观测电阻电流：i_R1 = V1/(R1+R2) = 1A
# di/dR1 = -0.1（含自身阻值直接影响；不计直接影响会错误得到 +0.1）
obs, rows = sens_text(DIV, 'current', 'R1')
d = sens_by_name(rows)
check(obs['type'] == 'current' and obs['name'] == 'R1'
      and close(obs['value'], 1) and obs['unit'] == 'A',
      '灵敏度：观测量（电阻电流）当前值')
check(close(d['R1']['derivative'], -0.1),
      '灵敏度：电阻电流对自身阻值导数（计入直接影响）')
check(close(d['R2']['derivative'], -0.1) and close(d['V1']['derivative'], 0.1),
      '灵敏度：电阻电流对其他元件导数')
check(d['R1']['derivative_unit'] == 'A/Ω' and d['V1']['derivative_unit'] == 'A/V',
      '灵敏度：电流观测的导数单位')

# 观测电压源电流：i_V1 = -V1/(R1+R2) = -1A，di/dR1 = +0.1
obs, rows = sens_text(DIV, 'current', 'V1')
d = sens_by_name(rows)
check(close(obs['value'], -1) and close(d['R1']['derivative'], 0.1)
      and close(d['V1']['derivative'], -0.1),
      '灵敏度：电压源电流及其导数')

# 观测电流源电流：对自身数值导数为 1，其余为 0
obs, rows = sens_text('I1 0 1 2\nR1 1 0 4', 'current', 'I1')
d = sens_by_name(rows)
check(close(obs['value'], 2) and close(d['I1']['derivative'], 1)
      and close(d['R1']['derivative'], 0),
      '灵敏度：电流源电流对自身导数为 1、其余为 0')
check(close(d['I1']['delta'], 0.02) and close(d['R1']['delta'], 0),
      '灵敏度：电流源 1% 变化量')

# 零值电源：导数照常给出，1% 变化量为 0
obs, rows = sens_text('V1 1 0 0\nR1 1 0 5\nI1 0 1 1', 'voltage', 1)
d = sens_by_name(rows)
check(close(d['V1']['derivative'], 1) and close(d['V1']['delta'], 0),
      '灵敏度：零值电压源导数非零、变化量为 0')

# 电流源对节点电压的导数：V(1) = I1 × R1，dV/dI1 = R = 5
obs, rows = sens_text('R1 1 0 5\nI1 0 1 1', 'voltage', 1)
d = sens_by_name(rows)
check(close(d['I1']['derivative'], 5) and close(d['I1']['delta'], 0.05),
      '灵敏度：电流源对节点电压的导数（dV/dI = R）')

# 元件名称不区分大小写
obs, rows = sens_text(DIV, 'current', 'r1')
check(obs['name'] == 'R1', '灵敏度：元件名称不区分大小写')

# 观测对象非法 / 不存在
def expect_sens_error(text, obs_type, target, keyword, label):
    try:
        sens_text(text, obs_type, target)
    except NetlistError as exc:
        check(keyword in str(exc), f'{label}（提示包含「{keyword}」）-> {exc}')
        return
    except Exception as exc:  # noqa: BLE001
        check(False, f'{label}（抛出了错误类型 {type(exc).__name__}: {exc}）')
        return
    check(False, f'{label}（未抛出 NetlistError）')


expect_sens_error(DIV, 'voltage', 0, '接地', '观测接地节点')
expect_sens_error(DIV, 'voltage', 7, '不存在', '观测不存在的节点')
expect_sens_error(DIV, 'current', 'R9', '不存在', '观测不存在的元件')
expect_sens_error(DIV, 'power', 2, '未知观测类型', '未知观测类型')

# 网表本身无唯一解时仍按原逻辑报错
try:
    sens_text('I1 1 0 1', 'voltage', 1)
except NoUniqueSolutionError as exc:
    check('悬浮' in str(exc), f'灵敏度：沿用无唯一解反馈 -> {exc}')
else:
    check(False, '灵敏度：无唯一解时未抛出 NoUniqueSolutionError')

# 与有限差分对比（混合电路，覆盖三种观测量 × 全部元件）
print('== 灵敏度：有限差分交叉验证 ==')
MIX = 'V1 1 0 12\nR1 1 2 6\nR2 2 0 6\nR3 2 3 12\nI1 0 3 0.5'


def obs_value_of(elements, nodes, obs_type, target):
    if obs_type == 'voltage':
        voltages, _ = solve_circuit(elements, nodes)
        return {v['node']: v['voltage'] for v in voltages}[target]
    _, currents = solve_circuit(elements, nodes)
    return {c['name']: c['current'] for c in currents}[target]


fd_ok = True
for obs_type, target in [('voltage', 3), ('current', 'R3'),
                         ('current', 'V1'), ('current', 'I1')]:
    elements, nodes = parse_netlist(MIX)
    _, rows = compute_sensitivity(elements, nodes, obs_type, target)
    base = obs_value_of(elements, nodes, obs_type, target)
    for row in rows:
        h = 1e-6
        perturbed = [solver.Element(e.name, e.kind, e.n1, e.n2,
                                    e.value + h if e.name == row['name']
                                    else e.value)
                     for e in elements]
        fd = (obs_value_of(perturbed, nodes, obs_type, target) - base) / h
        if not close(fd, row['derivative'], tol=1e-4):
            fd_ok = False
            print(f'  不匹配：观测 {obs_type} {target}，对 {row["name"]} '
                  f'解析导数 {row["derivative"]}，有限差分 {fd}')
check(fd_ok, '解析导数与有限差分一致（混合电路，三种观测量）')

# ---------- 两端口戴维南等效分析 ----------
print('== 两端口戴维南等效分析 ==')


def th_text(text, n1, n2, load):
    elements, nodes = parse_netlist(text)
    return compute_thevenin(elements, nodes, n1, n2, load)


# 分压电路，端口 2->0：Voc=5V，Rth=5∥5=2.5Ω；接 2.5Ω 负载时
# VL=2.5V，IL=1A，PL=2.5W
r = th_text(DIV, 2, 0, 2.5)
check(close(r['voc'], 5) and close(r['rth'], 2.5),
      '戴维南：分压电路开路电压与等效电阻')
check(close(r['vl'], 2.5) and close(r['il'], 1) and close(r['pl'], 2.5),
      '戴维南：负载端口电压、电流、功率')
check(r['voc_unit'] == 'V' and r['rth_unit'] == 'Ω' and r['vl_unit'] == 'V'
      and r['il_unit'] == 'A' and r['pl_unit'] == 'W' and r['load_unit'] == 'Ω',
      '戴维南：结果单位标注')

# 同一端口接 7.5Ω：IL = 5/10 = 0.5A，VL = 3.75V，PL = 1.875W
r = th_text(DIV, 2, 0, 7.5)
check(close(r['il'], 0.5) and close(r['vl'], 3.75) and close(r['pl'], 1.875),
      '戴维南：不同负载下的端口量（IL = Voc/(Rth+RL)）')

# 反向选择端点 0->2：Voc、VL、IL 变号，Rth 与功率不变
rf = th_text(DIV, 0, 2, 2.5)
check(close(rf['voc'], -5) and close(rf['rth'], 2.5)
      and close(rf['vl'], -2.5) and close(rf['il'], -1)
      and close(rf['pl'], 2.5),
      '戴维南：反向端点电压电流变号、电阻功率不变')

# 跨电阻端口 1->2：Voc=V(1)-V(2)=5V，Rth=5∥5=2.5Ω
r = th_text(DIV, 1, 2, 2.5)
check(close(r['voc'], 5) and close(r['rth'], 2.5),
      '戴维南：两个非接地节点之间的端口')

# 端口与理想电压源并联（1->0）：Rth=0Ω，VL=Voc=10V，IL=Voc/RL
r = th_text(DIV, 1, 0, 10)
check(close(r['voc'], 10) and close(r['rth'], 0)
      and close(r['vl'], 10) and close(r['il'], 1) and close(r['pl'], 10),
      '戴维南：端口被理想电压源短接时 Rth=0Ω')

# 0V 电压源同样短接端口：Voc=0，Rth=0，VL=IL=PL=0
r = th_text('V1 1 0 0\nR1 1 0 5\nI1 0 1 1', 1, 0, 4)
check(close(r['voc'], 0) and close(r['rth'], 0) and close(r['vl'], 0)
      and close(r['il'], 0) and close(r['pl'], 0),
      '戴维南：0V 电压源短接端口')

# 电流源电路 I1 0 1 2, R1 1 0 4：1->0 端口 Voc=8V，Rth=4Ω
r = th_text('I1 0 1 2\nR1 1 0 4', 1, 0, 4)
check(close(r['voc'], 8) and close(r['rth'], 4)
      and close(r['vl'], 4) and close(r['il'], 1) and close(r['pl'], 4),
      '戴维南：电流源电路（独立电流源置零为开路）')

# 混合电路，端口 3->2：Voc=13.5-7.5=6V；I1 置零后开路，节点 3 仅经
# R3=12Ω 连到节点 2，故 Rth=12Ω；接 4Ω 得 IL=0.375A、VL=1.5V、PL=0.5625W
r = th_text(MIX, 3, 2, 4)
check(close(r['voc'], 6) and close(r['rth'], 12)
      and close(r['vl'], 1.5) and close(r['il'], 0.375) and close(r['pl'], 0.5625),
      '戴维南：混合电路非接地节点端口')

# 混合电路，端口 3->0：Voc=13.5V；V1 置零短路（节点 1 接地）、I1 开路，
# Rth = R3 + (R1∥R2) = 12 + 3 = 15Ω
r = th_text(MIX, 3, 0, 2.4)
check(close(r['voc'], 13.5) and close(r['rth'], 15)
      and close(r['il'], 13.5 / 17.4) and close(r['vl'], 2.4 * 13.5 / 17.4)
      and close(r['pl'], 2.4 * (13.5 / 17.4) ** 2),
      '戴维南：混合电路对地端口与负载量')

# 纯电阻（无独立源）网络：Voc=0，Rth 仍为两端等效电阻（10∥(10+10)=20/3Ω）
r = th_text('R1 1 0 10\nR2 1 2 10\nR3 2 0 10', 1, 0, 5)
check(close(r['voc'], 0) and close(r['rth'], 20 / 3)
      and close(r['vl'], 0) and close(r['il'], 0) and close(r['pl'], 0),
      '戴维南：无独立源时 Voc=0、Rth 照常求出')

# 功率恒为非负（Voc 为负、端口被电压源短接 Rth=0 时 PL 仍为正）
r = th_text('V1 1 0 -10\nR1 1 0 10', 1, 0, 10)
check(close(r['voc'], -10) and close(r['rth'], 0)
      and r['pl'] >= 0 and close(r['pl'], 10),
      '戴维南：负开路电压下功率仍为非负')


def expect_th_error(text, n1, n2, load, keyword, label):
    try:
        th_text(text, n1, n2, load)
    except NetlistError as exc:
        check(keyword in str(exc), f'{label}（提示包含「{keyword}」）-> {exc}')
        return
    except Exception as exc:  # noqa: BLE001
        check(False, f'{label}（抛出了错误类型 {type(exc).__name__}: {exc}）')
        return
    check(False, f'{label}（未抛出 NetlistError）')


# 节点不存在（含接地节点 0 以外的编号）
expect_th_error(DIV, 7, 0, 1, '不存在', '戴维南：第一节点不存在')
expect_th_error(DIV, 2, 9, 1, '不存在', '戴维南：第二节点不存在')
# 端点相同（允许 0，但不能两个都是 0）
expect_th_error(DIV, 2, 2, 1, '不能相同', '戴维南：两端点相同（非接地）')
expect_th_error(DIV, 0, 0, 1, '不能相同', '戴维南：两端点相同（均为地）')
# 负载非法
expect_th_error(DIV, 2, 0, 0, '正的有限', '戴维南：零负载')
expect_th_error(DIV, 2, 0, -3, '正的有限', '戴维南：负负载')
expect_th_error(DIV, 2, 0, float('inf'), '正的有限', '戴维南：无穷负载')
expect_th_error(DIV, 2, 0, float('nan'), '正的有限', '戴维南：NaN 负载')
expect_th_error(DIV, 2, 0, 'abc', '合法的数值', '戴维南：非数值负载')

# 原网表无唯一解时沿用原报错
try:
    th_text('I1 1 0 1', 1, 0, 1)
except NoUniqueSolutionError as exc:
    check('悬浮' in str(exc), f'戴维南：无唯一解网表沿用原反馈 -> {exc}')
else:
    check(False, '戴维南：无唯一解时未抛出 NoUniqueSolutionError')

# ---------- 单电阻断路定位 ----------
print('== 单电阻断路定位 ==')


def diag_text(text, measurements):
    elements, nodes = parse_netlist(text)
    return diagnose_open(elements, nodes, measurements)


def results_by_name(result):
    return {r['hypothesis']: r for r in result['results']}


# 分压电路 DIV：正常 V(2)=5；R1 断路 V(2)=0；R2 断路 V(2)=10
r = diag_text(DIV, [{'node': 2, 'voltage': 5.0, 'tolerance': 0.1}])
d = results_by_name(r)
check(r['conclusion']['type'] == 'normal'
      and r['conclusion']['hypothesis'] == '正常电路',
      '断路定位：实测符合正常电路时结论为正常')
check(d['正常电路']['matched'] and close(d['正常电路']['score'], 0),
      '断路定位：正常假设得分 0 且匹配')
check(not d['R1 断路']['matched'] and close(d['R1 断路']['score'], 50)
      and not d['R2 断路']['matched']
      and close(d['R2 断路']['score'], 50),
      '断路定位：不匹配的断路假设得分 > 1')
check(r['matches'] == ['正常电路'], '断路定位：matches 仅含匹配假设')
check(r['results'][0]['hypothesis'] == '正常电路',
      '断路定位：结果按得分升序排列')

# 带符号误差：R1 断路预测 V(2)=0，实测 5 → 误差 = 预测 − 实测 = −5
pred = d['R1 断路']['predictions'][0]
check(pred['node'] == 2 and close(pred['predicted'], 0)
      and close(pred['error'], -5) and close(pred['measured'], 5)
      and close(pred['tolerance'], 0.1) and close(pred['ratio'], 50),
      '断路定位：预测值、带符号误差与容差')

# R1 断路：实测 V(2)=0 → 仅「R1 断路」匹配
r = diag_text(DIV, [{'node': 2, 'voltage': 0.0, 'tolerance': 0.1}])
check(r['conclusion']['type'] == 'open'
      and r['conclusion']['element'] == 'R1',
      '断路定位：实测符合 R1 断路时指出断路元件')

# 断路后无唯一解的假设单列原因，其余假设继续检查
NET3 = 'V1 1 0 10\nR1 1 2 5\nR2 2 0 5\nR3 2 3 8'
r = diag_text(NET3, [{'node': 2, 'voltage': 5.0, 'tolerance': 0.1}])
d = results_by_name(r)
check(not d['R3 断路']['solvable'] and '悬浮' in d['R3 断路']['reason'],
      '断路定位：断路后悬浮的假设单列原因')
check(d['正常电路']['solvable'] and d['R1 断路']['solvable']
      and d['R2 断路']['solvable'],
      '断路定位：无唯一解假设不影响其余假设求解')
check(r['conclusion']['type'] == 'normal',
      '断路定位：含不可求解假设时仍能得出正常结论')
check(r['results'][-1]['hypothesis'] == 'R3 断路',
      '断路定位：无唯一解的假设排在结果最后')

# 测点被电压源钳位：所有假设预测一致 → 多个匹配，无法区分
r = diag_text(DIV, [{'node': 1, 'voltage': 10.0, 'tolerance': 0.1}])
check(r['conclusion']['type'] == 'ambiguous'
      and len(r['conclusion']['candidates']) == 3,
      '断路定位：多个假设同时匹配时提示无法区分')
check([x['hypothesis'] for x in r['results']]
      == ['R1 断路', 'R2 断路', '正常电路'],
      '断路定位：得分相同按假设名称排序')

# 实测值与任何假设都不符 → 不符合单断路模型
r = diag_text(DIV, [{'node': 2, 'voltage': 3.0, 'tolerance': 0.1}])
check(r['conclusion']['type'] == 'none' and r['matches'] == [],
      '断路定位：无匹配时说明不符合单断路模型')

# 多测点：得分取各测点 |误差|/容差 的最大值
r = diag_text(DIV, [{'node': 1, 'voltage': 10.05, 'tolerance': 0.1},
                    {'node': 2, 'voltage': 5.0, 'tolerance': 0.1}])
d = results_by_name(r)
check(close(d['正常电路']['score'], 0.5)
      and r['conclusion']['type'] == 'normal',
      '断路定位：多测点得分取最大归一化误差')

# 得分决定排序：正常 0.1、R1 断路 4.9、R2 断路 5.1
r = diag_text(DIV, [{'node': 2, 'voltage': 4.9, 'tolerance': 1.0}])
check([x['hypothesis'] for x in r['results']]
      == ['正常电路', 'R1 断路', 'R2 断路'],
      '断路定位：可求解假设按得分升序排列')

# 无电阻的网表：仅枚举正常电路
r = diag_text('V1 1 0 4\nI1 0 1 0.5', [{'node': 1, 'voltage': 4.0,
                                        'tolerance': 0.1}])
check(len(r['results']) == 1 and r['conclusion']['type'] == 'normal',
      '断路定位：无电阻网表只检查正常电路')


def expect_diag_error(text, measurements, keyword, label,
                      exc_type=NetlistError):
    try:
        diag_text(text, measurements)
    except exc_type as exc:
        check(keyword in str(exc), f'{label}（提示包含「{keyword}」）-> {exc}')
        return
    except Exception as exc:  # noqa: BLE001
        check(False, f'{label}（抛出了错误类型 {type(exc).__name__}: {exc}）')
        return
    check(False, f'{label}（未抛出 {exc_type.__name__}）')


M = lambda n, v, t: {'node': n, 'voltage': v, 'tolerance': t}  # noqa: E731
expect_diag_error(DIV, [], '1～12', '断路定位：测点数量为 0')
expect_diag_error(DIV, [M(1, 1, 1)] * 13, '1～12', '断路定位：测点超过 12 条')
expect_diag_error(DIV, [M(0, 0, 1)], '接地', '断路定位：测点为接地节点')
expect_diag_error(DIV, [M(7, 1, 1)], '不存在', '断路定位：测点节点不存在')
expect_diag_error(DIV, [M(2, 5, 1), M(2, 5, 1)], '重复',
                  '断路定位：测点节点重复')
expect_diag_error(DIV, [M(2, 5, 0)], '正的有限', '断路定位：零容差')
expect_diag_error(DIV, [M(2, 5, -0.1)], '正的有限', '断路定位：负容差')
expect_diag_error(DIV, [M(2, 5, float('nan'))], '正的有限',
                  '断路定位：NaN 容差')
expect_diag_error(DIV, [M(2, 5, float('inf'))], '正的有限',
                  '断路定位：无穷容差')
expect_diag_error(DIV, [M(2, float('inf'), 1)], '有限',
                  '断路定位：无穷实测电压')
expect_diag_error(DIV, [M(2, float('nan'), 1)], '有限',
                  '断路定位：NaN 实测电压')

# 正常电路无唯一解时诊断前提不成立，直接报错
expect_diag_error('I1 1 0 1', [M(1, 1, 1)], '悬浮',
                  '断路定位：正常电路无唯一解', NoUniqueSolutionError)

print(f'\n通过 {PASSED} 项，失败 {FAILED} 项')
sys.exit(1 if FAILED else 0)
