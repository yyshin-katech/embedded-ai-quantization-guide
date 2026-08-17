#!/usr/bin/env python3
"""
b01 — grid_sample (4D) ONNX export opset sweep.

§4.6.1 / §5.1 표의 단정 검증:
  - opset <16 에서 aten::grid_sampler 가 "not supported" 로 깨지는가
  - opset >=16 에서 GridSample 노드로 export 되는가 (4D = 2D 공간 샘플링)
  - torch 2.11 기본 dynamo=True 경로는 opset_version 을 무시하는가 (DETR 실측과 동일한가)

각 (opset, 경로) 조합을 시도하고 정확한 에러 원문/성공 여부/emit된 opset/GridSample 노드 유무를 기록.
"""
import json, os, sys, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")  # 스윕 노이즈 억제 (에러는 예외로 따로 잡음)
OUT = os.path.dirname(os.path.abspath(__file__))


class GridSample4D(nn.Module):
    # BEVFormer bilinear 샘플링 설정 고정
    def forward(self, feat, grid):
        return F.grid_sample(feat, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)


def graph_op_types(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    ops = [n.op_type for n in m.graph.node]
    opset = {i.domain or "ai.onnx": i.version for i in m.opset_import}
    return ops, opset


def try_export(dynamo, opset, tag):
    feat = torch.randn(1, 8, 16, 16)          # [N,C,H,W]
    grid = torch.rand(1, 10, 10, 2) * 2 - 1   # [N,Hout,Wout,2] in [-1,1]
    path = os.path.join(OUT, f"_gs4d_{tag}.onnx")
    rec = {"path": "legacy" if not dynamo else "dynamo",
           "req_opset": opset, "ok": False, "err": None,
           "emit_opset": None, "has_gridsample": None, "n_nodes": None}
    try:
        kw = dict(dynamo=dynamo)
        if not dynamo:
            kw["opset_version"] = opset
        else:
            kw["opset_version"] = opset  # dynamo 가 실제로 무시하는지 확인용
            kw["verbose"] = False
        torch.onnx.export(GridSample4D(), (feat, grid), path, **kw)
        ops, emit = graph_op_types(path)
        rec["ok"] = True
        rec["emit_opset"] = emit.get("ai.onnx")
        rec["has_gridsample"] = ("GridSample" in ops)
        rec["n_nodes"] = len(ops)
        rec["op_types"] = sorted(set(ops))
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if os.path.exists(path):
            os.remove(path)
        for ext in (".onnx.data", ".data"):
            p2 = path + ext
            if os.path.exists(p2):
                os.remove(p2)
    return rec


def main():
    print(f"torch {torch.__version__}")
    results = []
    print("\n=== LEGACY (dynamo=False) — opset_version 존중 ===")
    for op in (9, 11, 13, 14, 15, 16, 17, 18, 20):
        r = try_export(False, op, f"legacy_op{op}")
        results.append(r)
        if r["ok"]:
            print(f"  opset {op:2d}: OK  emit={r['emit_opset']}  GridSample={r['has_gridsample']}  nodes={r['n_nodes']}")
        else:
            print(f"  opset {op:2d}: FAIL {r['err']}")

    print("\n=== DYNAMO (dynamo=True, torch 2.11 기본) — opset_version 무시 여부 ===")
    for op in (11, 17):
        r = try_export(True, op, f"dynamo_op{op}")
        results.append(r)
        if r["ok"]:
            print(f"  req opset {op:2d}: OK  emit={r['emit_opset']}  GridSample={r['has_gridsample']}  nodes={r['n_nodes']}")
        else:
            print(f"  req opset {op:2d}: FAIL {r['err']}")

    with open(os.path.join(OUT, "b01_grid_sample_4d_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: b01_grid_sample_4d_result.json")


if __name__ == "__main__":
    main()
