#!/usr/bin/env python3
"""Build the crossover SSOT (crossover_summary.json) from local raw artifacts.

Reads (all under ../ relative to this script's crossover/ root):
  raw/corescale/corescale.csv   seq + async core-count sweep (1/2/3 cores)
  raw/bufsweep/bufsweep.csv     dxrun --buffer-count sweep
  results/{rn50,y26,y5s}_npuall.json   analyze_profiler.py stage summaries
  raw/{rn50,y26,y5s}_npuall/stdout.txt  dxbenchmark stdout (input/output bytes)

Emits results/crossover_summary.json — the single source of truth cited by the
report. Regime is classified purely from the profiler p50 stage split:
  compute-bound  if Inference p50 > D2H p50   (NPU compute is the bottleneck)
  D2H-bound      otherwise                    (PCIe output transfer bottleneck)

Core-count sweep maps dxrun -n to physical cores:
  1 core = -n 1 (NPU_0) · 2 cores = -n 4 (NPU_0/1) · 3 cores = -n 0 (NPU_ALL).
"""
import csv, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {'resnet50': 'rn50', 'yolo26n': 'y26', 'yolov5s': 'y5s'}
BYTES = re.compile(r'input\s+([\d,]+)\s+bytes,\s+output\s+([\d,]+)\s+bytes')


def read_corescale():
    d = {}
    with open(os.path.join(ROOT, 'raw/corescale/corescale.csv')) as f:
        for r in csv.DictReader(f):
            m = d.setdefault(r['model'], {'seq': None, 'async': {}})
            if r['mode'] == 'seq':
                m['seq'] = float(r['fps'])
            else:
                m['async'][int(r['cores'])] = float(r['fps'])
    return d


def read_bufsweep():
    d = {}
    with open(os.path.join(ROOT, 'raw/bufsweep/bufsweep.csv')) as f:
        for r in csv.DictReader(f):
            d.setdefault(r['model'], {})[int(r['buffer_count'])] = float(r['fps'])
    return d


def read_bytes(tag):
    with open(os.path.join(ROOT, f'raw/{tag}_npuall/stdout.txt')) as f:
        for line in f:
            m = BYTES.search(line)
            if m:
                return (int(m.group(1).replace(',', '')),
                        int(m.group(2).replace(',', '')))
    return (None, None)


def main():
    cs, bs = read_corescale(), read_bufsweep()
    out = {'note': 'DX-M1 host-bound vs NPU-bound crossover; same DX-M1 + same '
                   'Pi5, model varied. dxrun -b async, --buffer-count 8 for the '
                   'core-count sweep. Regime from dxbenchmark profiler p50.',
           'core_map': {'1 core': '-n 1 (NPU_0)', '2 cores': '-n 4 (NPU_0/1)',
                        '3 cores': '-n 0 (NPU_ALL)'},
           'models': {}}
    for name, tag in MODELS.items():
        prof = json.load(open(os.path.join(ROOT, f'results/{tag}_npuall.json')))
        inb, outb = read_bytes(tag)
        st = prof['stages']
        infer = prof['inference_all_cores']['p50_ms']
        d2h = st['D2H']['p50_ms']
        h2d = st['H2D']['p50_ms']
        a = cs[name]['async']
        scaling = round(a[3] / a[1], 3)
        job = {c.split()[-1]: prof['inference_per_core'][c]['n']
               for c in prof['inference_per_core']}
        out['models'][name] = {
            'input_bytes': inb, 'output_bytes': outb,
            'seq_fps': cs[name]['seq'],
            'async_fps': {'1core': a[1], '2core': a[2], '3core': a[3]},
            'core_scaling_3c_over_1c': scaling,
            'async_over_seq': round(a[3] / cs[name]['seq'], 3),
            'profiler_p50_ms': {'H2D': h2d, 'Inference': infer, 'D2H': d2h},
            'profiler_fps_internal': prof['fps'],
            'core_job_dist': job,
            'regime': 'compute-bound' if infer > d2h else 'D2H-bound',
            'infer_over_d2h': round(infer / d2h, 2),
            'd2h_over_infer': round(d2h / infer, 2),
            'buffer_sweep_fps': bs.get(name, {}),
        }
    # cross-model ratios (SSOT for the report's headline numbers)
    M = out['models']
    out['headline'] = {
        'output_bytes_ratio_y26_over_rn50':
            round(M['yolo26n']['output_bytes'] / M['resnet50']['output_bytes'], 1),
        'd2h_ratio_y26_over_rn50':
            round(M['yolo26n']['profiler_p50_ms']['D2H'] /
                  M['resnet50']['profiler_p50_ms']['D2H'], 1),
        'async3c_ratio_rn50_over_y26':
            round(M['resnet50']['async_fps']['3core'] /
                  M['yolo26n']['async_fps']['3core'], 2),
        'y5s_lighter_compute_than_rn50':
            M['yolov5s']['profiler_p50_ms']['Inference'] <
            M['resnet50']['profiler_p50_ms']['Inference'],
        'y5s_slower_than_rn50_factor':
            round(M['resnet50']['async_fps']['3core'] /
                  M['yolov5s']['async_fps']['3core'], 1),
    }
    p = os.path.join(ROOT, 'results/crossover_summary.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
