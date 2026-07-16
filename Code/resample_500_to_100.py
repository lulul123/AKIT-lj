#!/usr/bin/env python3
"""
resample_500_to_100.py

将 500Hz 的 IMU/Vel 文本文件下采样到 100Hz，保证每秒输出恰好 100 个样本（seq_len=100）
用法示例：
  python resample_500_to_100.py --input path/to/imu_500hz.txt --output path/to/imu_100hz.txt --method pick
  python resample_500_to_100.py --dir path/to/dataset --pattern "imu.txt"  # 在目录中查找并处理

支持两种下采样方法：
 - pick: 每隔 factor=5 选择一个样本（速度快，保留原始时间戳）
 - avg: 每 5 个样本做平均（平滑时间戳和数值）

注意：脚本只保留那些在某一整数秒内恰好有 500 个样本的秒，其他秒会被跳过并打印提示。
"""

import os
import argparse
from collections import defaultdict
import numpy as np


def resample_file(input_path, output_path, source_rate=500, target_rate=100, method='pick', float_fmt='%.6f'):
    assert source_rate % target_rate == 0, 'source_rate must be divisible by target_rate'
    factor = source_rate // target_rate

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    # 读取文件
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]

    if not lines:
        raise ValueError('Empty input file')

    header = lines[0]
    data_lines = lines[1:]

    # 按整数秒分组
    groups = defaultdict(list)  # sec -> list of (timestamp, data_list)
    for ln in data_lines:
        if not ln.strip():
            continue
        parts = ln.split()
        try:
            ts = float(parts[0])
        except Exception:
            # 无效行跳过
            continue
        values = [float(x) for x in parts[1:]] if len(parts) > 1 else []
        sec = int(ts)
        groups[sec].append((ts, values))

    kept_seconds = []
    downsampled = []  # list of (ts, values)
    skipped = 0
    exact_match_found = False

    # 首先尝试严格匹配 source_rate 的秒
    for sec in sorted(groups.keys()):
        arr = groups[sec]
        if len(arr) == source_rate:
            exact_match_found = True
            arr.sort(key=lambda x: x[0])
            if method == 'pick':
                for i in range(target_rate):
                    idx = i * factor
                    ts, vals = arr[idx]
                    downsampled.append((ts, vals))
            else:  # avg
                for i in range(target_rate):
                    slice_items = arr[i*factor:(i+1)*factor]
                    ts_mean = sum([it[0] for it in slice_items]) / factor
                    if slice_items and slice_items[0][1]:
                        vals_arr = np.array([it[1] for it in slice_items], dtype=np.float64)
                        vals_mean = vals_arr.mean(axis=0).tolist()
                    else:
                        vals_mean = []
                    downsampled.append((ts_mean, vals_mean))
            kept_seconds.append(sec)
        else:
            # 记录为候选但先跳过
            continue

    # 如果没有严格匹配的秒，则采用弹性策略：只要每秒样本数 >= target_rate，就做等间隔下采样
    if not exact_match_found:
        downsampled = []
        kept_seconds = []
        for sec in sorted(groups.keys()):
            arr = groups[sec]
            n = len(arr)
            if n < target_rate:
                skipped += 1
                continue

            arr.sort(key=lambda x: x[0])
            # 计算等间隔索引或分段
            if method == 'pick':
                for i in range(target_rate):
                    # 选择接近等分点的索引
                    idx = int(round(i * n / target_rate))
                    if idx >= n:
                        idx = n - 1
                    ts, vals = arr[idx]
                    downsampled.append((ts, vals))
            else:  # avg
                for i in range(target_rate):
                    start = int(np.floor(i * n / target_rate))
                    end = int(np.floor((i + 1) * n / target_rate))
                    if end <= start:
                        end = min(start + 1, n)
                    slice_items = arr[start:end]
                    ts_mean = sum([it[0] for it in slice_items]) / len(slice_items)
                    if slice_items and slice_items[0][1]:
                        vals_arr = np.array([it[1] for it in slice_items], dtype=np.float64)
                        vals_mean = vals_arr.mean(axis=0).tolist()
                    else:
                        vals_mean = []
                    downsampled.append((ts_mean, vals_mean))
            kept_seconds.append(sec)

    if not downsampled:
        raise RuntimeError('No valid seconds with enough samples found; nothing to write.')

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as fw:
        fw.write(header + '\n')
        for ts, vals in downsampled:
            if vals:
                line = ' '.join([float_fmt % ts] + [float_fmt % v for v in vals])
            else:
                line = float_fmt % ts
            fw.write(line + '\n')

    print(f'Input: {input_path}')
    print(f'Output: {output_path}')
    print(f'Seconds processed (kept): {len(kept_seconds)}')
    print(f'Seconds skipped (not enough samples): {skipped}')


def process_directory(directory, pattern=None, source_rate=500, target_rate=100, method='pick'):
    """在目录内查找匹配文件并生成下采样文件，输出名后缀 _100hz"""
    files = []
    for root, _, filenames in os.walk(directory):
        for fn in filenames:
            if pattern:
                if pattern in fn:
                    files.append(os.path.join(root, fn))
            else:
                # 默认处理 .txt 文件
                if fn.lower().endswith('.txt'):
                    files.append(os.path.join(root, fn))

    if not files:
        print('No files found to process')
        return

    for fp in files:
        out_fp = fp.replace('.txt', '_100hz.txt')
        try:
            resample_file(fp, out_fp, source_rate=source_rate, target_rate=target_rate, method=method)
        except Exception as e:
            print(f'Failed to process {fp}: {e}')


def resample_pair_imu_vel(imu_path, vel_path, out_imu_path, out_vel_path,
                          source_rate=500, target_rate=100, method='pick', float_fmt='%.6f'):
    """
    按 vel 时间戳范围配对 imu 与 vel（要求时间戳完全一致），然后对配对成功的样本按秒下采样到 target_rate。
    输出两个文件：out_imu_path 和 out_vel_path，时间戳一一对应。
    """
    assert source_rate >= target_rate and target_rate > 0, 'invalid rates'

    # 读取 imu
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f'IMU file not found: {imu_path}')
    if not os.path.exists(vel_path):
        raise FileNotFoundError(f'Vel file not found: {vel_path}')

    with open(imu_path, 'r', encoding='utf-8', errors='ignore') as f:
        imu_lines = [ln.rstrip('\n') for ln in f]
    with open(vel_path, 'r', encoding='utf-8', errors='ignore') as f:
        vel_lines = [ln.rstrip('\n') for ln in f]

    if not imu_lines or not vel_lines:
        raise ValueError('Empty imu or vel file')

    imu_header = imu_lines[0]
    vel_header = vel_lines[0]
    imu_data_lines = imu_lines[1:]
    vel_data_lines = vel_lines[1:]

    # 构建时间戳到数据映射
    imu_map = {}
    for ln in imu_data_lines:
        if not ln.strip():
            continue
        parts = ln.split()
        try:
            ts = float(parts[0])
        except Exception:
            continue
        vals = [float(x) for x in parts[1:]] if len(parts) > 1 else []
        imu_map[ts] = vals

    vel_map = {}
    for ln in vel_data_lines:
        if not ln.strip():
            continue
        parts = ln.split()
        try:
            ts = float(parts[0])
        except Exception:
            continue
        vals = [float(x) for x in parts[1:]] if len(parts) > 1 else []
        vel_map[ts] = vals

    # 以 vel 时间戳为准，筛选出在 imu 中也存在的时间戳（完全匹配）
    paired_ts = [ts for ts in sorted(vel_map.keys()) if ts in imu_map]

    if not paired_ts:
        raise RuntimeError('No matching timestamps between vel and imu')

    # 按整数秒分组配对样本
    from collections import defaultdict
    sec_groups = defaultdict(list)  # sec -> list of (ts, imu_vals, vel_vals)
    for ts in paired_ts:
        sec = int(ts)
        sec_groups[sec].append((ts, imu_map[ts], vel_map[ts]))

    # 下采样/对每秒做 pick 或 avg，确保每秒输出 target_rate
    out_imu = []  # list of (ts, vals)
    out_vel = []
    skipped_seconds = 0
    for sec in sorted(sec_groups.keys()):
        arr = sec_groups[sec]
        arr.sort(key=lambda x: x[0])
        n = len(arr)
        if n < target_rate:
            skipped_seconds += 1
            continue

        if n == source_rate:
            # 可以直接按 factor 取样
            factor = source_rate // target_rate
            if method == 'pick':
                for i in range(target_rate):
                    idx = i * factor
                    if idx >= n:
                        idx = n - 1
                    ts, imu_vals, vel_vals = arr[idx]
                    out_imu.append((ts, imu_vals))
                    out_vel.append((ts, vel_vals))
            else:  # avg
                for i in range(target_rate):
                    slice_items = arr[i*factor:(i+1)*factor]
                    ts_mean = sum([it[0] for it in slice_items]) / len(slice_items)
                    imu_avg = np.mean([it[1] for it in slice_items], axis=0).tolist() if slice_items and slice_items[0][1] else []
                    vel_avg = np.mean([it[2] for it in slice_items], axis=0).tolist() if slice_items and slice_items[0][2] else []
                    out_imu.append((ts_mean, imu_avg))
                    out_vel.append((ts_mean, vel_avg))
        else:
            # 弹性选择：按等间隔索引选取 target_rate 样本
            if method == 'pick':
                for i in range(target_rate):
                    idx = int(np.floor(i * n / target_rate))
                    if idx >= n:
                        idx = n - 1
                    ts, imu_vals, vel_vals = arr[idx]
                    out_imu.append((ts, imu_vals))
                    out_vel.append((ts, vel_vals))
            else:
                for i in range(target_rate):
                    start = int(np.floor(i * n / target_rate))
                    end = int(np.floor((i + 1) * n / target_rate))
                    if end <= start:
                        end = min(start + 1, n)
                    slice_items = arr[start:end]
                    ts_mean = sum([it[0] for it in slice_items]) / len(slice_items)
                    imu_avg = np.mean([it[1] for it in slice_items], axis=0).tolist() if slice_items and slice_items[0][1] else []
                    vel_avg = np.mean([it[2] for it in slice_items], axis=0).tolist() if slice_items and slice_items[0][2] else []
                    out_imu.append((ts_mean, imu_avg))
                    out_vel.append((ts_mean, vel_avg))

    if not out_imu:
        raise RuntimeError('No output samples after pairing/downsampling')

    # 写入输出文件（使用原始 header）
    with open(out_imu_path, 'w', encoding='utf-8') as f:
        f.write(imu_header + '\n')
        for ts, vals in out_imu:
            if vals:
                line = ' '.join([('%.6f' % ts)] + [('%.6f' % v) for v in vals])
            else:
                line = ('%.6f' % ts)
            f.write(line + '\n')

    with open(out_vel_path, 'w', encoding='utf-8') as f:
        f.write(vel_header + '\n')
        for ts, vals in out_vel:
            if vals:
                line = ' '.join([('%.6f' % ts)] + [('%.6f' % v) for v in vals])
            else:
                line = ('%.6f' % ts)
            f.write(line + '\n')

    print(f'Paired input IMU: {imu_path}  Vel: {vel_path}')
    print(f'Paired output IMU: {out_imu_path}  Vel: {out_vel_path}')
    print(f'Seconds skipped (samples < {target_rate}): {skipped_seconds}')


def main():
    parser = argparse.ArgumentParser(description='Resample 500Hz text data to 100Hz per-second windows')
    parser.add_argument('--input', type=str, help='input file path')
    parser.add_argument('--output', type=str, help='output file path')
    parser.add_argument('--dir', type=str, help='process all files in directory')
    parser.add_argument('--pattern', type=str, default=None, help='filename pattern to match when using --dir')
    parser.add_argument('--method', type=str, default='pick', choices=['pick', 'avg'], help='downsample method')
    parser.add_argument('--source_rate', type=int, default=500, help='source sampling rate (default 500)')
    parser.add_argument('--target_rate', type=int, default=100, help='target sampling rate (default 100)')
    parser.add_argument('--imu', type=str, help='input IMU file path')
    parser.add_argument('--vel', type=str, help='input Vel file path')
    parser.add_argument('--out_imu', type=str, help='output IMU file path')
    parser.add_argument('--out_vel', type=str, help='output Vel file path')

    args = parser.parse_args()

    if args.input:
        out = args.output if args.output else args.input.replace('.txt', '_100hz.txt')
        resample_file(args.input, out, source_rate=args.source_rate, target_rate=args.target_rate, method=args.method)
    elif args.dir:
        process_directory(args.dir, pattern=args.pattern, source_rate=args.source_rate, target_rate=args.target_rate, method=args.method)
    elif args.imu and args.vel and args.out_imu and args.out_vel:
        resample_pair_imu_vel(args.imu, args.vel, args.out_imu, args.out_vel,
                              source_rate=args.source_rate, target_rate=args.target_rate, method=args.method)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
