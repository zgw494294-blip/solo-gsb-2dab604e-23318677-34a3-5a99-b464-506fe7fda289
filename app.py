"""Flask 入口：提供页面与 /solve、/sensitivity、/thevenin、/diagnose-open 接口。"""

import math

from flask import Flask, jsonify, render_template, request

from solver import (NetlistError, NoUniqueSolutionError, compute_sensitivity,
                    compute_thevenin, diagnose_open, parse_netlist,
                    solve_circuit)

app = Flask(__name__)

# 防止异常输入撑爆内存，网表文本本身很小
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024


def solve_netlist(text: str) -> dict:
    """解析并求解网表文本，返回可 JSON 序列化的结果字典。"""
    elements, nodes = parse_netlist(text)
    voltages, currents = solve_circuit(elements, nodes)
    return {'ok': True, 'node_voltages': voltages, 'currents': currents}


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/solve')
def solve():
    payload = request.get_json(silent=True) or {}
    text = payload.get('netlist', '')
    if not isinstance(text, str):
        return jsonify(ok=False, error='请求格式错误：netlist 必须是字符串'), 400
    try:
        result = solve_netlist(text)
    except (NetlistError, NoUniqueSolutionError) as exc:
        return jsonify(ok=False, error=str(exc))
    return jsonify(**result)


def _parse_observation(payload: dict) -> tuple[str, object]:
    """从请求体提取观测量，返回 (类型, 目标)。"""
    obs = payload.get('observation')
    if not isinstance(obs, dict):
        raise NetlistError(
            '请求格式错误：observation 必须是对象，形如 '
            '{"type": "voltage", "node": 2} 或 '
            '{"type": "current", "element": "R1"}')
    obs_type = obs.get('type')
    if obs_type == 'voltage':
        raw = obs.get('node')
        try:
            node = int(str(raw).strip(), 10)
        except (ValueError, TypeError):
            raise NetlistError(
                f'观测节点编号 "{raw}" 非法，应为非负整数') from None
        if node < 0:
            raise NetlistError(f'观测节点编号 {node} 非法，应为非负整数')
        return 'voltage', node
    if obs_type == 'current':
        name = obs.get('element')
        if not isinstance(name, str) or not name.strip():
            raise NetlistError('请给出要观测电流的元件名称（element 字段）')
        return 'current', name.strip()
    raise NetlistError(
        f'观测类型 "{obs_type}" 非法：只能是 "voltage"（节点电压）'
        '或 "current"（元件电流）')


@app.post('/sensitivity')
def sensitivity():
    payload = request.get_json(silent=True) or {}
    text = payload.get('netlist', '')
    if not isinstance(text, str):
        return jsonify(ok=False, error='请求格式错误：netlist 必须是字符串'), 400
    try:
        elements, nodes = parse_netlist(text)
        obs_type, obs_target = _parse_observation(payload)
        observation, rows = compute_sensitivity(
            elements, nodes, obs_type, obs_target)
    except (NetlistError, NoUniqueSolutionError) as exc:
        return jsonify(ok=False, error=str(exc))
    return jsonify(ok=True, observation=observation, sensitivities=rows)


def _parse_port(payload: dict) -> tuple[int, int, float]:
    """从请求体提取戴维南端口参数，返回 (第一节点, 第二节点, 负载电阻)。"""
    ports = []
    for key, label in (('node1', '第一节点'), ('node2', '第二节点')):
        raw = payload.get(key)
        if isinstance(raw, bool):
            raise NetlistError(f'{label}编号非法，应为非负整数（允许节点 0）')
        try:
            node = int(str(raw).strip(), 10)
        except (ValueError, TypeError):
            raise NetlistError(
                f'{label}编号 "{raw}" 非法，应为非负整数（允许节点 0）') from None
        if node < 0:
            raise NetlistError(f'{label}编号 {node} 非法，应为非负整数（允许节点 0）')
        ports.append(node)

    raw_load = payload.get('load')
    if isinstance(raw_load, bool):
        raise NetlistError('负载电阻非法，应为正的有限数值（单位 Ω）')
    try:
        load = float(str(raw_load).strip())
    except (ValueError, TypeError):
        raise NetlistError(
            f'负载电阻 "{raw_load}" 不是合法的数值，应为正的有限数值（单位 Ω）'
        ) from None
    if not math.isfinite(load) or load <= 0:
        raise NetlistError(
            f'负载电阻必须为正的有限值（当前为 {raw_load}，单位 Ω）')
    return ports[0], ports[1], load


@app.post('/thevenin')
def thevenin():
    payload = request.get_json(silent=True) or {}
    text = payload.get('netlist', '')
    if not isinstance(text, str):
        return jsonify(ok=False, error='请求格式错误：netlist 必须是字符串'), 400
    try:
        elements, nodes = parse_netlist(text)
        node1, node2, load = _parse_port(payload)
        result = compute_thevenin(elements, nodes, node1, node2, load)
    except (NetlistError, NoUniqueSolutionError) as exc:
        return jsonify(ok=False, error=str(exc))
    return jsonify(ok=True, thevenin=result)


def _parse_measurements(payload: dict) -> list[dict]:
    """从请求体提取测点列表，返回 [{'node', 'voltage', 'tolerance'}]。"""
    raw = payload.get('measurements')
    if not isinstance(raw, list):
        raise NetlistError(
            '请求格式错误：measurements 必须是数组，形如 '
            '[{"node": 2, "voltage": 5.0, "tolerance": 0.1}, ...]（1～12 条）')
    if not 1 <= len(raw) <= 12:
        raise NetlistError(f'测点数量为 {len(raw)} 条，应为 1～12 条')
    measurements = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise NetlistError(
                f'第 {i} 条测点必须是对象，形如 '
                '{"node": 2, "voltage": 5.0, "tolerance": 0.1}')
        raw_node = item.get('node')
        if isinstance(raw_node, bool):
            raise NetlistError(f'第 {i} 条测点：节点编号非法，应为非负整数')
        try:
            node = int(str(raw_node).strip(), 10)
        except (ValueError, TypeError):
            raise NetlistError(
                f'第 {i} 条测点：节点编号 "{raw_node}" 非法，'
                '应为非负整数') from None
        if node < 0:
            raise NetlistError(
                f'第 {i} 条测点：节点编号 {node} 非法，应为非负整数')

        values = {}
        for key, label in (('voltage', '实测电压'), ('tolerance', '容差')):
            raw_val = item.get(key)
            if isinstance(raw_val, bool):
                raise NetlistError(
                    f'第 {i} 条测点：{label}非法，应为数值（单位 V）')
            try:
                values[key] = float(str(raw_val).strip())
            except (ValueError, TypeError):
                raise NetlistError(
                    f'第 {i} 条测点：{label} "{raw_val}" 不是合法的数值'
                    '（单位 V）') from None
        measurements.append({'node': node, 'voltage': values['voltage'],
                             'tolerance': values['tolerance']})
    return measurements


@app.post('/diagnose-open')
def diagnose_open_endpoint():
    payload = request.get_json(silent=True) or {}
    text = payload.get('netlist', '')
    if not isinstance(text, str):
        return jsonify(ok=False, error='请求格式错误：netlist 必须是字符串'), 400
    try:
        elements, nodes = parse_netlist(text)
        measurements = _parse_measurements(payload)
        result = diagnose_open(elements, nodes, measurements)
    except (NetlistError, NoUniqueSolutionError) as exc:
        return jsonify(ok=False, error=str(exc))
    return jsonify(ok=True, **result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
