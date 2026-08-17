#!/usr/bin/env python3
# sq_04_alpha_sweep.py — SmoothQuant migration strength α 스윕 (자기완결).
# 같은 eval 서브셋(앞 N장 고정)에서 FP32 / INT8_DEFAULT(max) / SmoothQuant(α∈{0.5,1.0}) 를 재어
#  ① modelopt 프리셋 기본 α=1.0  vs  ② 논문 권장 α=0.5  의 회복 차이를 격리한다.
# 겸사: dict-override({"method":"smoothquant","alpha":a})가 실제로 동작하는지 실증
#       → 초안 §4.4의 "⚠️ 확인 필요"(alpha 지정 위치)를 데이터로 종결.
# 프리셋은 모듈 전역이라 copy.deepcopy로 복제 후 override(전역 오염 방지).
#   사용법: sq_04_alpha_sweep.py [--limit N] [--calib K]
import sys, json, copy, time, torch
import modelopt
import modelopt.torch.quantization as mtq
import sq_common as C

N = 500; ncalib = 100
for a in sys.argv[1:]:
    if a.startswith("--limit"): N = int(a.split("=")[1])
    elif a.startswith("--calib"): ncalib = int(a.split("=")[1])

ids = C.img_ids()[:N]                       # 앞 N장 고정 (전 팔 동일 eval)
calib_pvs, calib_ids = C.calib_pixel_values(ncalib)   # 대표 100장(균등, eval과 독립)
fwd = C.forward_loop_factory(calib_pvs)
print(f"eval {len(ids)}장(앞 N 고정) | calib {len(calib_ids)}장 | dev {C.DEV}", flush=True)

out = {"eval_images": len(ids), "calib_images": len(calib_ids),
       "modelopt": modelopt.__version__, "arms": {}}

def run(cfg, tag):
    m = C.load_model()
    if cfg is not None:
        t = time.time(); m = mtq.quantize(m, cfg, fwd)
        print(f"  {tag} quantize {time.time()-t:.1f}s", flush=True)
    r = C.evaluate(m, ids, tag=tag)
    print(f"  {tag}: {r}", flush=True)
    del m; torch.cuda.empty_cache()
    return r

# --- FP32 기준 & INT8_DEFAULT(대조군, max) ---
print("\n=== [FP32] ===", flush=True)
out["arms"]["fp32"] = run(None, "fp32")
print("\n=== [INT8_DEFAULT] (max, no smooth) ===", flush=True)
out["arms"]["int8_default"] = run(mtq.INT8_DEFAULT_CFG, "int8_default")

# --- SmoothQuant α 스윕 (deepcopy → dict override) ---
alphas = [0.5, 1.0]                          # 0.5=논문 권장, 1.0=modelopt 프리셋 기본
for al in alphas:
    cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    cfg["algorithm"] = {"method": "smoothquant", "alpha": al}
    print(f"\n=== [SmoothQuant α={al}] (dict-override) ===", flush=True)
    out["arms"][f"sq_a{al}"] = run(cfg, f"sq_a{al}")

# --- 회복률 분해 ---
fp = out["arms"]["fp32"]["mAP"]; d0 = out["arms"]["int8_default"]["mAP"]
gap = round(fp - d0, 4)
out["gap_fp32_minus_default"] = gap
out["recovery_pct_by_alpha"] = {
    str(al): (round((out["arms"][f"sq_a{al}"]["mAP"] - d0) / gap * 100, 1) if gap else None)
    for al in alphas
}
out["mAP_by_alpha"] = {str(al): out["arms"][f"sq_a{al}"]["mAP"] for al in alphas}
# dict-override가 실제로 α를 바꿨는지(두 α의 mAP가 달라야 override가 먹은 것): 실증
out["alpha_override_worked"] = (out["arms"]["sq_a0.5"]["mAP"] != out["arms"]["sq_a1.0"]["mAP"])
json.dump(out, open("experiments/stage2_smoothquant/sq_04_alpha_sweep.json", "w"), indent=2)
print("\nGAP:", gap, "| RECOVERY% by α:", json.dumps(out["recovery_pct_by_alpha"]),
      "| override_worked:", out["alpha_override_worked"], flush=True)
print("SQ04_DONE")
