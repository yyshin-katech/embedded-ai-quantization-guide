#!/usr/bin/env python3
"""Parse a dxbenchmark profiler.json for the DETR hybrid pipeline into a clean stage
summary. Unlike the crossover analyze_profiler.py (NPU-only), DETR's .dxnn has BOTH an
npu_0 task (CNN backbone + first encoder self-attention, INT8) AND a cpu_0 task (the rest
of the transformer, FP32 via ORT) -> keys look like
  "<stage>[npu_0][Device_0][Job_z]"   and   "CPU Task[cpu_0][Device_-1][Job_z]".
So the regex is widened to npu_\\d+|cpu_\\d+ and Device_-?\\d+.

FPS = n(NPU Task jobs) / (max_end - min_start over ALL stages) = true async pipeline rate,
gated by the bottleneck stage (for DETR that is the host CPU Task, not the NPU or PCIe).

Usage: analyze_detr_regime.py <profiler.json> [label]"""
import json
import re
import statistics as st
import sys

KEY = re.compile(r'^(.*?)\[((?:npu|cpu)_\d+)\]\[(Device_-?\d+)\]\[Job_(\d+)\]$')


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    d = json.load(open(path))

    stage_durs, core_durs = {}, {}
    npu_jobs = set()
    all_start, all_end = [], []

    for k, v in d.items():
        m = KEY.match(k)
        if not m:
            continue
        stage, grp, _dev, job = m.group(1), m.group(2), m.group(3), int(m.group(4))
        rec = v[0] if isinstance(v, list) and v else v
        if not (isinstance(rec, dict) and 'start' in rec and 'end' in rec):
            continue
        dur = (rec['end'] - rec['start']) / 1e6  # ms
        stage_durs.setdefault(stage, []).append(dur)
        if stage.startswith('Inference Core'):
            core_durs.setdefault(stage, []).append(dur)
        all_start.append(rec['start'])
        all_end.append(rec['end'])
        if stage == 'NPU Task':
            npu_jobs.add(job)

    window_s = (max(all_end) - min(all_start)) / 1e9 if all_end else 0
    fps = len(npu_jobs) / window_s if window_s else 0

    def stat(name):
        a = stage_durs.get(name, [])
        if not a:
            return None
        return {'mean_ms': round(st.mean(a), 3), 'p50_ms': round(st.median(a), 3),
                'min_ms': round(min(a), 3), 'max_ms': round(max(a), 3), 'n': len(a)}

    order = ['CPU Task', 'CPU Dispatch Wait', 'Framework Overhead', 'NPU Task',
             'H2D', 'Inference', 'NPU Input Format Handler', 'D2H',
             'NPU Output Format Handler', 'Buffer Pool Wait', 'Service Process Wait']
    out = {'label': label, 'n_npu_jobs': len(npu_jobs), 'window_s': round(window_s, 3),
           'fps': round(fps, 3),
           'stages': {s: stat(s) for s in order if s in stage_durs},
           'inference_per_core': {c: stat(c) for c in sorted(core_durs.keys())}}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
