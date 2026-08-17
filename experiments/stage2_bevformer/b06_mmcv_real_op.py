#!/usr/bin/env python3
"""
b06 — [Tier B / 레거시 venv ~/bevf-legacy] mmcv 실제 CUDA op 검증.

Tier A(b05)는 mmcv 폴백 함수를 "복사"해 분해를 봤다. b06은 mmcv를 실제 설치해
  B1) MultiScaleDeformableAttnFunction(진짜 CUDA 커널)이 CUDA 12.8/드라이버 595에서 로드·실행되는가
      → 이게 Tier B 전체의 go/no-go (레거시 mmcv CUDA op가 정본 CUDA에서 사는가)
  B2) 그 커스텀 op를 torch.onnx.export 하면 무엇이 나오는가
      - 커스텀 도메인 노드(mmcv::MMCVMultiScaleDeformableAttention, 플러그인 필요)인가
      - 아니면 onnx export 시 자동으로 순수 PyTorch 분해로 폴백하는가 (§4.6.2 핵심 단정)
를 실측한다.  실행:  ~/bevf-legacy/bin/python b06_mmcv_real_op.py
"""
import os, json, warnings
import torch

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
res = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ---- mmcv 로드 (컴파일된 _ext .so 로드 = ABI/CUDA 호환 1차 관문) ----
try:
    import mmcv
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
    res["mmcv"] = mmcv.__version__
    print(f"mmcv {mmcv.__version__}  (_ext 로드 성공)")
except Exception as e:
    res["mmcv_import_error"] = f"{type(e).__name__}: {e}"
    print(f"### mmcv import 실패(=Tier B 벽): {res['mmcv_import_error']}")
    with open(os.path.join(OUT, "b06_mmcv_real_op_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    raise SystemExit(1)

# ---- 테스트 텐서 (b05와 동일 shape) ----
bs, num_heads, embed_dims = 1, 8, 32
num_levels, num_points, num_queries = 4, 4, 100
shapes = [(8, 8), (4, 4), (2, 2), (1, 1)]
num_value = sum(H * W for H, W in shapes)
spatial_shapes = torch.as_tensor(shapes, dtype=torch.long)
level_start = torch.cat([spatial_shapes.new_zeros(1),
                         spatial_shapes.prod(1).cumsum(0)[:-1]])

def make(dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    value = torch.randn(bs, num_value, num_heads, embed_dims, generator=g).to(dev)
    loc = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2, generator=g).to(dev)
    w = torch.rand(bs, num_queries, num_heads, num_levels, num_points, generator=g).to(dev)
    return value, loc, w

# ---- B1: CUDA 커널 실행 + CPU 폴백과 수치 일치 ----
print("\n=== B1: 진짜 CUDA 커널 ===")
try:
    v, loc, w = make("cuda")
    ss, ls = spatial_shapes.cuda(), level_start.cuda()
    out_cuda = MultiScaleDeformableAttnFunction.apply(v, ss, ls, loc, w, 64)
    out_ref = multi_scale_deformable_attn_pytorch(v, ss, loc, w)  # 순수 PyTorch 기준
    diff = (out_cuda - out_ref).abs().max().item()
    res["B1"] = {"ok": True, "out_shape": list(out_cuda.shape), "max_abs_diff_vs_pytorch": diff}
    print(f"  CUDA op OK  out={tuple(out_cuda.shape)}  |CUDA-pytorch|max={diff:.2e}  (일치=커널 정상)")
except Exception as e:
    res["B1"] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:300]}"}
    print(f"  CUDA op 실패: {res['B1']['err']}")

# ---- B2: 실제 MODULE 의 ONNX export 거동 (§4.6.2 핵심) ----
# mmcv 1.7.1 은 symbolic/is_in_onnx_export 이 없음 → 커스텀 ONNX 노드/플러그인 경로가 "내장돼 있지 않다".
#   - CUDA 텐서로 export: 모듈이 CUDA Function 분기(line 353) → symbolic 없어 export 실패
#   - CPU  텐서로 export: 모듈이 pytorch 폴백 분기(line 357) → 표준 op 분해(= b05, GridSample×levels)
# 즉 "바닐라 mmcv 로 BEVFormer 를 export 하는 유일한 길 = CPU 폴백 분기 강제 → 분해".
print("\n=== B2: 실제 MODULE export (CPU 폴백 vs CUDA Function) ===")
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention

class RealModule(nn.Module):
    """진짜 mmcv MultiScaleDeformableAttention 모듈 (proj 포함)."""
    def __init__(self, dev):
        super().__init__()
        torch.manual_seed(0)
        self.msda = MultiScaleDeformableAttention(
            embed_dims=256, num_heads=8, num_levels=4, num_points=4,
            batch_first=True).to(dev).eval()
        self.register_buffer("ss", spatial_shapes.to(dev))
        self.register_buffer("lsi", level_start.to(dev))
    def forward(self, query, value, reference_points):
        return self.msda(query=query, value=value, reference_points=reference_points,
                         spatial_shapes=self.ss, level_start_index=self.lsi)

def make_module_inputs(dev):
    g = torch.Generator(device="cpu").manual_seed(1)
    query = torch.randn(bs, num_queries, 256, generator=g).to(dev)
    value = torch.randn(bs, num_value, 256, generator=g).to(dev)
    ref = torch.rand(bs, num_queries, num_levels, 2, generator=g).to(dev)  # 정규화 [0,1]
    return query, value, ref

import onnx
from collections import Counter
for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    path = os.path.join(OUT, f"_real_module_{dev}.onnx")
    rec = {"device": dev, "ok": False}
    try:
        mod = RealModule(dev)
        qi, vi, ri = make_module_inputs(dev)
        torch.onnx.export(mod, (qi, vi, ri), path, opset_version=16,
                          input_names=["query", "value", "reference_points"])
        m = onnx.load(path)
        c = Counter(n.op_type for n in m.graph.node)
        domains = sorted({(n.domain or "ai.onnx") for n in m.graph.node})
        custom = [n.op_type for n in m.graph.node
                  if (n.domain or "ai.onnx") not in ("ai.onnx", "")]
        graph_inputs = [i.name for i in m.graph.input]
        # value/reference_points 가 살아있나? 사라졌으면 = MSDeformAttn 출력이 상수로 baked
        dropped = [x for x in ("value", "reference_points") if x not in graph_inputs]
        rec.update(ok=True, n_nodes=sum(c.values()), n_gridsample=c.get("GridSample", 0),
                   domains=domains, custom_nodes=custom, graph_inputs=graph_inputs,
                   dropped_inputs=dropped, top_ops=c.most_common(8))
        if dropped:
            kind = f"🔴 silent-wrong: {dropped} 가 상수로 baked(그래프 입력에서 사라짐)"
        elif custom:
            kind = "커스텀 노드 " + str(custom)
        else:
            kind = f"표준 분해(GridSample={c.get('GridSample',0)}=num_levels)"
        print(f"  [{dev:4s}] export OK  nodes={sum(c.values())}  inputs={graph_inputs}  -> {kind}")
        os.remove(path)
        for e in (".onnx.data", ".data"):
            if os.path.exists(path + e): os.remove(path + e)
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[:280]}"
        print(f"  [{dev:4s}] FAIL {rec['err']}")
    res.setdefault("B2", {})[dev] = rec

with open(os.path.join(OUT, "b06_mmcv_real_op_result.json"), "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("\n저장: b06_mmcv_real_op_result.json")
