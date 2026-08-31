#!/usr/bin/env python3
"""Sample Pi5 host-board power via `vcgencmd pmic_read_adc` = Sum(I_rail*V_rail).
NOTE: covers only Pi5 *internal* rails; the M.2 DX-M1 card draws from EXT5V
upstream of these rails and is NOT captured here. Usage: psample.py <dur_s> <label>"""
import subprocess, re, time, sys, json, statistics as st
dur, label = float(sys.argv[1]), sys.argv[2]
CUR = re.compile(r'(\w+)_A current\(\d+\)=([\d.]+)A')
VLT = re.compile(r'(\w+)_V volt\(\d+\)=([\d.]+)V')
def sample():
    r = subprocess.run(['vcgencmd','pmic_read_adc'],capture_output=True,text=True).stdout
    cur = {m.group(1): float(m.group(2)) for m in CUR.finditer(r)}
    vlt = {m.group(1): float(m.group(2)) for m in VLT.finditer(r)}
    per = {k: cur[k]*vlt[k] for k in cur if k in vlt}
    return sum(per.values()), per
watts, per_acc = [], {}
t0 = time.time(); first = True
while time.time()-t0 < dur:
    p, per = sample()
    if first: first = False; continue
    watts.append(p)
    for k,w in per.items(): per_acc.setdefault(k,[]).append(w)
out = {'label':label,'n':len(watts),'dur_s':round(dur,1),
       'watt_mean':round(st.mean(watts),3),'watt_p50':round(st.median(watts),3),
       'watt_min':round(min(watts),3),'watt_max':round(max(watts),3),
       'per_rail_w_mean':{k:round(st.mean(v),4) for k,v in sorted(per_acc.items())}}
print(json.dumps(out))
