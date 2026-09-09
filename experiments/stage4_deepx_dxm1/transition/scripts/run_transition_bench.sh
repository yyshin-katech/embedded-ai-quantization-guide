#!/usr/bin/env bash
# DX-M1 regime transition sweep (Pi-side harness).
#
# For each synthetic fixed-compute / swept-output model tr_c<cout>, at each core
# count (1c=-n 1, 2c=-n 4, 3c=-n 0) run dxbenchmark with the profiler on and reduce
# profiler.json -> per-stage p50 + measured async FPS + per-core job distribution.
#
# The trunk compute is FIXED across the sweep (Inference p50 ~= const); only the head
# expander changes the output (D2H) size.  Sweeping cout therefore walks the model from
# compute-bound (Inference > D2H, core-scaling ~=3x) to D2H-bound (D2H > Inference,
# core-scaling ~=1x), tracing the transition curve the crossover axis (274bffa) proved
# exists at the extremes but did not trace.
#
# Emits (small, version-controlled): raw/<model>/analyzed_n{1,2,3}.json, raw/corescale.csv
# Keeps (large, gitignored):         raw/<model>/profiler_n3.json  (3-core raw, archival)
#
# Usage (on the Pi):  bash run_transition_bench.sh [TIME] [WARMUP]
set -u
BASE=~/dxm1_transition
MODELS=$BASE/models
RAW=$BASE/raw
ANALYZE=$BASE/analyze_profiler.py
TIME=${1:-5}
WARMUP=${2:-2}
TMP=/tmp/dxm1_cs
mkdir -p "$RAW" "$TMP"

# core-count -> dxbenchmark -n arg :  1core=NPU_0(-n1)  2core=NPU_0/1(-n4)  3core=NPU_ALL(-n0)
declare -A NARG=( [1]=1 [2]=4 [3]=0 )

CSV=$RAW/corescale.csv
echo "name,cout,nominal_bytes,fps_1c,fps_2c,fps_3c,scale_3c_1c,scale_2c_1c" > "$CSV"

# sweep in ascending cout order
COUTS="5 20 41 82 163 327 490 653 980 1276 2551 5102"
for c in $COUTS; do
  m="tr_c$c"
  mdir="$MODELS/$m"
  [ -d "$mdir" ] || { echo "!! missing $mdir"; continue; }
  odir="$RAW/$m"; mkdir -p "$odir"
  nominal=$(( c * 14 * 14 * 4 ))
  declare -A FPS
  for cores in 1 2 3; do
    n=${NARG[$cores]}
    # profiler run (verbose): writes profiler.json to CWD
    ( cd "$BASE" && rm -f profiler.json && \
      dxbenchmark --dir "$mdir" -n "$n" -t "$TIME" --warmup "$WARMUP" -v >/dev/null 2>&1 )
    prof="$BASE/profiler.json"
    if [ -s "$prof" ]; then
      python3 "$ANALYZE" "$prof" "${m}_n${cores}c" > "$odir/analyzed_n${cores}.json"
      FPS[$cores]=$(python3 -c "import json;print(json.load(open('$odir/analyzed_n${cores}.json'))['fps'])")
      [ "$cores" = "3" ] && cp "$prof" "$odir/profiler_n3.json"
    else
      FPS[$cores]=0
    fi
  done
  s31=$(python3 -c "print(round(${FPS[3]}/${FPS[1]},3) if ${FPS[1]} else 0)")
  s21=$(python3 -c "print(round(${FPS[2]}/${FPS[1]},3) if ${FPS[1]} else 0)")
  echo "$m,$c,$nominal,${FPS[1]},${FPS[2]},${FPS[3]},$s31,$s21" >> "$CSV"
  # progress line with the key stages from the 3-core profile
  read INF H2D D2H <<<"$(python3 -c "
import json;d=json.load(open('$odir/analyzed_n3.json'))
s=d['stages'];ia=d['inference_all_cores']
print(round(ia['p50_ms'],3),round(s['H2D']['p50_ms'],3),round(s['D2H']['p50_ms'],3))")"
  printf '  %-9s cout=%-5s out=%8dB  Inf=%-6s H2D=%-6s D2H=%-6s  fps 1/2/3c=%s/%s/%s  3c/1c=%s\n' \
    "$m" "$c" "$nominal" "$INF" "$H2D" "$D2H" "${FPS[1]}" "${FPS[2]}" "${FPS[3]}" "$s31"
done
echo "=== done -> $CSV ==="
cat "$CSV"
