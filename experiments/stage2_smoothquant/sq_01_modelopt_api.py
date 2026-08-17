#!/usr/bin/env python3
# sq_01_modelopt_api.py — modelopt 실제 API를 기록해 §4.4 초안의 "⚠️ 확인 필요"를 해소한다.
# 초안 가정: config["algorithm"]={"method":"smoothquant","alpha":0.5}  ← 0.45.0에선 틀림.
# 실측: algorithm은 최상위 문자열 "smoothquant"; quant_cfg는 리스트 형식.
import json, inspect
import modelopt
import modelopt.torch.quantization as mtq

def axes(cfg):
    d = {}
    for e in cfg["quant_cfg"]:
        if isinstance(e, dict) and e.get("quantizer_name") in ("*weight_quantizer", "*input_quantizer"):
            d[e["quantizer_name"]] = e.get("cfg")
    return d

rec = {
    "modelopt_version": modelopt.__version__,
    "quantize_signature": str(inspect.signature(mtq.quantize)),
    "cfg_presets_with_CFG": [c for c in dir(mtq) if "CFG" in c],
    "INT8_SMOOTHQUANT_CFG": {
        "exists": hasattr(mtq, "INT8_SMOOTHQUANT_CFG"),
        "top_keys": list(mtq.INT8_SMOOTHQUANT_CFG.keys()),
        "algorithm": mtq.INT8_SMOOTHQUANT_CFG.get("algorithm"),
        "axes": axes(mtq.INT8_SMOOTHQUANT_CFG),
    },
    "INT8_DEFAULT_CFG": {
        "exists": hasattr(mtq, "INT8_DEFAULT_CFG"),
        "algorithm": mtq.INT8_DEFAULT_CFG.get("algorithm"),
        "axes": axes(mtq.INT8_DEFAULT_CFG),
    },
    "draft_override_valid": None,  # 아래에서 판정
}
# 초안의 dict-algorithm override가 현행에서 유효한지: algorithm이 문자열이면 초안 예시는 부정확
rec["draft_override_valid"] = not isinstance(mtq.INT8_SMOOTHQUANT_CFG.get("algorithm"), str)

json.dump(rec, open("experiments/stage2_smoothquant/sq_01_api.json", "w"), indent=2)
print(json.dumps(rec, indent=2))
print("SQ01_DONE")
