#!/usr/bin/env python3
"""Parse a dxbenchmark profiler.json into a clean summary.

dxbenchmark writes profiler.json with one entry per stage per job:
  "<stage>[npu_x][Device_y][Job_z]" -> [{"start": <ns>, "end": <ns>}]
Durations are nanoseconds; duration_ms = (end-start)/1e6.

Throughput (FPS) is the true measured rate = n_jobs / (max_end - min_start),
which for an async pipeline is gated by the bottleneck stage, not the serial
sum of stages. Per-stage p50/mean/p90 come from the per-job durations.

Usage: analyze_profiler.py <profiler.json> [label]
Prints a JSON summary to stdout.
"""
import json, re, sys, statistics as st

KEY = re.compile(r'^(.*?)\[(npu_\d+)\]\[(Device_\d+)\]\[Job_(\d+)\]$')


def pct(a, q):
    if not a:
        return None
    s = sorted(a)
    i = min(len(s) - 1, int(len(s) * q))
    return s[i]


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    d = json.load(open(path))

    stage_durs = {}          # stage -> [duration_ms]
    core_durs = {}           # "Inference Core N" -> [duration_ms]
    task_jobs = {}           # core -> set(job_id) for NPU Task
    all_start, all_end = [], []

    for k, v in d.items():
        m = KEY.match(k)
        if not m:
            continue
        stage, core, dev, job = m.group(1), m.group(2), m.group(3), int(m.group(4))
        rec = v[0] if isinstance(v, list) and v else v
        if not (isinstance(rec, dict) and 'start' in rec and 'end' in rec):
            continue
        dur = (rec['end'] - rec['start']) / 1e6  # ms
        stage_durs.setdefault(stage, []).append(dur)
        if stage.startswith('Inference Core'):
            core_durs.setdefault(stage, []).append(dur)
        if stage == 'NPU Task':
            task_jobs.setdefault(core, set()).add(job)
            all_start.append(rec['start'])
            all_end.append(rec['end'])

    n_jobs = sum(len(s) for s in task_jobs.values())
    window_s = (max(all_end) - min(all_start)) / 1e9 if all_end else 0
    fps = n_jobs / window_s if window_s else 0

    def stat(name):
        a = stage_durs.get(name, [])
        if not a:
            return None
        return {'mean_ms': round(st.mean(a), 4),
                'p50_ms': round(st.median(a), 4),
                'p90_ms': round(pct(a, 0.90), 4),
                'min_ms': round(min(a), 4),
                'max_ms': round(max(a), 4),
                'n': len(a)}

    out = {
        'label': label,
        'n_jobs': n_jobs,
        'window_s': round(window_s, 4),
        'fps': round(fps, 2),
        'active_cores': sorted(core_durs.keys()),
        'stages': {s: stat(s) for s in
                   ['H2D', 'D2H', 'NPU Task', 'NPU Input Format Handler',
                    'NPU Output Format Handler', 'Buffer Pool Wait',
                    'Framework Overhead', 'Service Process Wait'] if s in stage_durs},
        'inference_per_core': {c: stat(c) for c in sorted(core_durs.keys())},
    }
    # aggregate inference across cores (pure NPU compute)
    inf_all = [x for c in core_durs for x in core_durs[c]]
    if inf_all:
        out['inference_all_cores'] = {'mean_ms': round(st.mean(inf_all), 4),
                                      'p50_ms': round(st.median(inf_all), 4),
                                      'n': len(inf_all)}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
