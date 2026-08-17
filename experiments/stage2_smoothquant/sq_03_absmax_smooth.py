#!/usr/bin/env python3
# sq_03_absmax_smooth.py — §4.4 "전/후 activation 분포 비교"(초안 smooth_check) 실측.
# 대상 transformer Linear의 입력 채널별 absmax를, 스무딩 '전'(raw)과 '후'(× pre_quant_scale)로
# 비교해 spike(max/median 비)가 실제로 눌리는지 본다. 초안 기대: ratio 31.7x → 2.96x.
import sys, json, numpy as np, torch
import modelopt.torch.quantization as mtq
import sq_common as C

# --- 대상 Linear 선택: encoder.layers.0 의 self-attn projection 우선 ---
def pick_target(model):
    cand = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "encoder.layers.0" in name and "proj" in name:
            cand.append(name)
    if not cand:  # 폴백: 첫 encoder Linear
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "encoder" in name:
                cand.append(name); break
    return cand[0]

def get_module(model, name):
    m = model
    for p in name.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

# 순환성 회피: smoothquant 캘리브 셋과 absmax 측정 셋을 '분리'한다.
all_ids = C.img_ids()
step = max(1, len(all_ids) // 64)
calib_ids = all_ids[::step][:64]                 # smoothquant 캘리브(균등 64장)
cset = set(calib_ids)
measure_ids = [i for i in all_ids[123:400] if i not in cset][:24]   # 측정용 분리 24장
calib_pvs = [C.pixel_values(i)[0] for i in calib_ids]
measure_pvs = [C.pixel_values(i)[0] for i in measure_ids]
print(f"calib {len(calib_ids)}장 / 측정 {len(measure_ids)}장 (분리, 교집합 {len(cset & set(measure_ids))})", flush=True)

# --- 전: FP 모델에서 대상 Linear 입력의 채널별 absmax 수집(측정 셋) ---
model = C.load_model()
target = pick_target(model)
Cin = get_module(model, target).in_features
print(f"대상 Linear: {target}  (in_features={Cin})", flush=True)

acc = torch.zeros(Cin)
def pre_hook(mod, inp):
    x = inp[0].detach().float().cpu()          # [.., Cin]
    a = x.abs().amax(dim=tuple(range(x.ndim - 1)))
    global acc
    acc = torch.maximum(acc, a)
h = get_module(model, target).register_forward_pre_hook(pre_hook)
with torch.no_grad():
    for pv in measure_pvs:
        model(pixel_values=pv)
h.remove()
before = acc.clone()
del model; torch.cuda.empty_cache()

# --- 후: smoothquant로 양자화한 모델에서 pre_quant_scale(=1/s) 추출(별도 calib 셋) ---
model2 = C.load_model()
fwd = C.forward_loop_factory(calib_pvs)
model2 = mtq.quantize(model2, mtq.INT8_SMOOTHQUANT_CFG, fwd)
qmod = get_module(model2, target)

# input_quantizer의 pre_quant_scale를 견고하게 탐색 (버전별 속성명 대비)
pqs = None
iq = getattr(qmod, "input_quantizer", None)
for holder in (iq, qmod):
    if holder is None:
        continue
    for attr in ("pre_quant_scale", "_pre_quant_scale"):
        v = getattr(holder, attr, None)
        if v is not None:
            pqs = v.detach().float().cpu().reshape(-1); break
    if pqs is None:  # 버퍼로 등록된 경우
        for n, b in holder.named_buffers():
            if "pre_quant_scale" in n and b.numel() == Cin:
                pqs = b.detach().float().cpu().reshape(-1); break
    if pqs is not None:
        break

def ratio(v):
    v = v[v > 0]
    return float(v.max() / v.median())

rec = {"target": target, "in_features": int(Cin),
       "before": {"max": round(float(before.max()), 3), "median": round(float(before.median()), 3),
                  "ratio": round(ratio(before), 2)}}

if pqs is not None and pqs.numel() == Cin:
    after = before * pqs                     # 스무딩된 activation absmax
    rec["pre_quant_scale_found"] = True
    rec["after"] = {"max": round(float(after.max()), 3), "median": round(float(after.median()), 3),
                    "ratio": round(ratio(after), 2)}
    rec["pre_quant_scale_stats"] = {"min": round(float(pqs.min()), 4), "max": round(float(pqs.max()), 4)}
    rec["ratio_before_to_after"] = f"{rec['before']['ratio']}x -> {rec['after']['ratio']}x"
else:
    rec["pre_quant_scale_found"] = False
    rec["note"] = "pre_quant_scale 미발견 — 속성명/폴딩 위치 재확인 필요"

json.dump(rec, open("experiments/stage2_smoothquant/sq_03_absmax.json", "w"), indent=2)
print(json.dumps(rec, indent=2, ensure_ascii=False))
print("SQ03_DONE")
