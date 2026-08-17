#!/usr/bin/env python3
# sq_02_triplet_map.py — 자기일관 3원 COCO mAP: FP32 / INT8_DEFAULT(대조) / INT8_SMOOTHQUANT(처리).
# 두 INT8 프리셋은 비트폭·weight축(per-ch)·activation축(per-tensor)이 동일하고
# 오직 algorithm("max" vs "smoothquant")만 다르다 → SmoothQuant 효과를 격리한다.
# 매 팔마다 모델을 새로 로드(quantize가 in-place라 오염 방지), 동일 eval 셋·동일 calib 셋 사용.
#   사용법: sq_02_triplet_map.py [--limit N] [--calib K]
import sys, json, time, torch
import modelopt.torch.quantization as mtq
import sq_common as C

limit = None; ncalib = 100
for a in sys.argv[1:]:
    if a.startswith("--limit"): limit = int(a.split("=")[1])
    elif a.startswith("--calib"): ncalib = int(a.split("=")[1])

ids = C.img_ids(limit)
calib_pvs, calib_ids = C.calib_pixel_values(ncalib)
fwd = C.forward_loop_factory(calib_pvs)
print(f"eval 이미지: {len(ids)} | calib: {len(calib_ids)}장(균등간격) | device: {C.DEV}", flush=True)

out = {"eval_images": len(ids), "calib_images": len(calib_ids),
       "modelopt": __import__("modelopt").__version__, "arms": {}}

# --- 팔 1: FP32 (sanity ≈ 0.4207) ---
print("\n=== [FP32] baseline ===", flush=True)
m = C.load_model()
out["arms"]["fp32"] = C.evaluate(m, ids, tag="fp32")
print("FP32:", out["arms"]["fp32"], flush=True)
del m; torch.cuda.empty_cache()

# --- 팔 2: INT8_DEFAULT (per-tensor act, max calib, NO smoothing) = 대조군 ---
print("\n=== [INT8_DEFAULT] per-tensor activation, max calib (대조군) ===", flush=True)
m = C.load_model()
t = time.time()
m = mtq.quantize(m, mtq.INT8_DEFAULT_CFG, fwd)
print(f"  quantize(max) 완료 {time.time()-t:.1f}s", flush=True)
out["arms"]["int8_default"] = C.evaluate(m, ids, tag="int8_default")
print("INT8_DEFAULT:", out["arms"]["int8_default"], flush=True)
del m; torch.cuda.empty_cache()

# --- 팔 3: INT8_SMOOTHQUANT (same granularity + smoothing) = 처리군 ---
print("\n=== [INT8_SMOOTHQUANT] per-tensor activation + smoothing (처리군) ===", flush=True)
m = C.load_model()
t = time.time()
m = mtq.quantize(m, mtq.INT8_SMOOTHQUANT_CFG, fwd)
print(f"  quantize(smoothquant) 완료 {time.time()-t:.1f}s", flush=True)
out["arms"]["int8_smoothquant"] = C.evaluate(m, ids, tag="int8_smoothquant")
print("INT8_SMOOTHQUANT:", out["arms"]["int8_smoothquant"], flush=True)
del m; torch.cuda.empty_cache()

# --- 델타 ---
fp = out["arms"]["fp32"]["mAP"]
d0 = out["arms"]["int8_default"]["mAP"]
sq = out["arms"]["int8_smoothquant"]["mAP"]
out["deltas"] = {
    "fp32": fp,
    "int8_default_drop": round(d0 - fp, 4),
    "smoothquant_drop": round(sq - fp, 4),
    "smoothquant_recovery_vs_default": round(sq - d0, 4),   # 핵심 수치
    "smoothquant_recovery_pct_of_gap": (round((sq - d0) / (fp - d0) * 100, 1) if (fp - d0) != 0 else None),
}
json.dump(out, open("experiments/stage2_smoothquant/sq_02_triplet_map.json", "w"), indent=2)
print("\nDELTAS:", json.dumps(out["deltas"]), flush=True)
print("SQ02_DONE")
