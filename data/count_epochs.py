"""历史数据统计辅助脚本。

统计 Inno.txt 的唯一秒标识，并可对速度积分估算里程。当前路径和目标轨迹仍是早期
Windows 环境的硬编码值，因此它不属于标准数据校验或训练流程；正式数据检查应使用
tools/validate_dataset.py。运行前必须手动修改 BASE_DIR 和 TARGET。
"""

import os
import re
import numpy as np

BASE_DIR = r"F:\AAA-lulul\科研工作\毕业设计\A-KIT-main\A-KIT-main\data_lj"
TARGET = 'data013325'

def count_epochs_in_file(filepath):
    ids = set()
    pattern = re.compile(r'^\s*(\d+)\b')
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    ids.add(m.group(1))
    except FileNotFoundError:
        return ids
    return ids


def compute_distance_from_vel(filepath):
    """读取 vel 文件（首列时间，后面为速度分量），对速度模进行时间积分得到里程（米）。"""
    if not os.path.exists(filepath):
        return 0.0
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.replace(',', ' ').split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    # 非数字 token 跳过整行
                    vals = []
                    break
            if vals:
                rows.append(vals)
    if len(rows) < 2:
        return 0.0
    arr = np.asarray(rows, dtype=float)
    # 排序并确保时间在第一列
    arr = arr[np.argsort(arr[:, 0])]
    t = arr[:, 0]
    vel_cols = arr[:, 1:]
    if vel_cols.size == 0:
        return 0.0
    # 若只有一维速度，则直接取绝对值
    if vel_cols.ndim == 1 or vel_cols.shape[1] == 1:
        speed = np.abs(vel_cols.flatten())
    else:
        speed = np.linalg.norm(vel_cols, axis=1)
    # 积分：trapezoidal
    try:
        dist = float(np.trapz(speed, t))
    except Exception:
        # 若时间戳不合适，使用差分
        dt = np.diff(t)
        if dt.size == 0:
            return 0.0
        dist = float(np.sum((speed[:-1] + speed[1:]) / 2 * dt))
    return dist

# find data* subfolders
all_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith('data')]
all_dirs.sort()

results = {}
for d in all_dirs:
    folder = os.path.join(BASE_DIR, d)
    infile = os.path.join(folder, 'Inno.txt')
    if not os.path.exists(infile):
        # fallback to any .txt file
        txts = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
        if txts:
            infile = os.path.join(folder, txts[0])
        else:
            results[d] = set()
            distances = globals().get('distances', {})
            distances[d] = 0.0
            globals()['distances'] = distances
            continue
    ids = count_epochs_in_file(infile)
    results[d] = ids
    # 计算对应文件夹的里程（优先找 Vel.txt）
    velfile = os.path.join(folder, 'Vel.txt')
    if not os.path.exists(velfile):
        # 尝试查找名称中包含 vel 的文件（忽略大小写），否则第一个 .txt
        vel_candidates = [f for f in os.listdir(folder) if 'vel' in f.lower()]
        if vel_candidates:
            velfile = os.path.join(folder, vel_candidates[0])
        else:
            txts = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
            velfile = os.path.join(folder, txts[0]) if txts else ''
    dist = 0.0
    if velfile:
        try:
            dist = compute_distance_from_vel(velfile)
        except Exception:
            dist = 0.0
    distances = globals().get('distances', {})
    distances[d] = dist
    globals()['distances'] = distances

# count for data013325
if TARGET in results:
    count_target = len(results[TARGET])
else:
    count_target = 0

# 目标里程
distances = globals().get('distances', {})
target_dist = float(distances.get(TARGET, 0.0))

# select other four data folders (excluding TARGET)
others = [d for d in all_dirs if d != TARGET]
selected_others = others[:4]

sum_others = sum(len(results.get(d, set())) for d in selected_others)
sum_others_dist = sum(float(distances.get(d, 0.0)) for d in selected_others)

total_all = sum(len(v) for v in results.values())
total_distance_all = sum(float(v) for v in distances.values())

print(f"Found data folders: {all_dirs}")
print(f"Epochs in {TARGET}: {count_target}")
print(f"Selected other folders (first 4 excluding {TARGET}): {selected_others}")
print(f"Total epochs in those 4 folders: {sum_others}")
print(f"Total epochs across all found data folders: {total_all}")
print(f"里程（{TARGET}）：{target_dist:.3f} m")
print(f"其余四组里程总和: {sum_others_dist:.3f} m")
print(f"全部文件夹总里程: {total_distance_all:.3f} m")

# Optionally write a small report
report = os.path.join(BASE_DIR, 'epoch_counts_report.txt')
with open(report, 'w', encoding='utf-8') as rf:
    rf.write(f"Folder\tEpochCount\n")
    for d in all_dirs:
        rf.write(f"{d}\t{len(results.get(d, set()))}\t{float(distances.get(d,0.0)):.3f}\n")
    rf.write(f"\nTarget {TARGET}: {count_target}\n")
    rf.write(f"Selected others: {selected_others}\n")
    rf.write(f"Sum others: {sum_others}\n")
    rf.write(f"Total all: {total_all}\n")
    rf.write(f"\nTarget distance (m): {target_dist:.3f}\n")
    rf.write(f"Selected others distance sum (m): {sum_others_dist:.3f}\n")
    rf.write(f"Total distance all (m): {total_distance_all:.3f}\n")

print(f"Report written to: {report}")
