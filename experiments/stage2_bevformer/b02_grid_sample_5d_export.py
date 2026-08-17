#!/usr/bin/env python3
"""
b02 — grid_sample (5D 볼류메트릭) ONNX export.

§4.6.1 / §5.1 표의 단정 검증:
  - opset 16/17 에서 5D 는 "5D volumetric" 류로 깨지는가 (4D만 지원)
  - opset 20 에서 5D 가 표준으로 export 되는가 (torch 2.11 legacy)
  - dynamo 경로에서 5D 거동
성공한 export 파일은 b03(런타임: ORT/TRT)이 재사용하도록 남긴다.
"""
import json, os, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))


class GridSample5D(nn.Module):
    def forward(self, feat, grid):
        # feat [N,C,D,H,W], grid [N,Dout,Hout,Wout,3]
        return F.grid_sample(feat, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)


def graph_info(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    ops = [n.op_type for n in m.graph.node]
    opset = {i.domain or "ai.onnx": i.version for i in m.opset_import}
    return sorted(set(ops)), opset.get("ai.onnx")


def try_export(dynamo, opset, keep):
    feat = torch.randn(1, 4, 6, 8, 8)          # [N,C,D,H,W]
    grid = torch.rand(1, 5, 8, 8, 3) * 2 - 1   # [N,Do,Ho,Wo,3] volumetric
    tag = ("dynamo" if dynamo else "legacy") + f"_op{opset}"
    path = os.path.join(OUT, f"_gs5d_{tag}.onnx")
    rec = {"path": "dynamo" if dynamo else "legacy", "req_opset": opset,
           "ok": False, "err": None, "emit_opset": None,
           "op_types": None, "saved": None}
    try:
        kw = dict(dynamo=dynamo, opset_version=opset)
        if dynamo:
            kw["verbose"] = False
        torch.onnx.export(GridSample5D(), (feat, grid), path, **kw)
        ops, emit = graph_info(path)
        rec.update(ok=True, emit_opset=emit, op_types=ops)
        if keep:
            rec["saved"] = os.path.basename(path)
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {str(e)[:400]}"
    finally:
        if not (rec["ok"] and keep) and os.path.exists(path):
            os.remove(path)
        for ext in (".onnx.data", ".data"):
            if os.path.exists(path + ext) and not (rec["ok"] and keep):
                os.remove(path + ext)
    return rec


def main():
    print(f"torch {torch.__version__}\n")
    results = []
    print("=== LEGACY (dynamo=False), 5D ===")
    for op in (16, 17, 18, 20):
        r = try_export(False, op, keep=(op == 20))  # 20 성공하면 런타임 테스트용 보존
        results.append(r)
        print(f"  opset {op:2d}: " + ("OK  emit=%s  ops=%s%s" % (
            r["emit_opset"], r["op_types"], "  [saved %s]" % r["saved"] if r["saved"] else "")
            if r["ok"] else "FAIL " + r["err"]))

    print("\n=== DYNAMO (dynamo=True), 5D ===")
    for op in (17, 20):
        r = try_export(True, op, keep=(op == 20))
        results.append(r)
        print(f"  req opset {op:2d}: " + ("OK  emit=%s  ops=%s%s" % (
            r["emit_opset"], r["op_types"], "  [saved %s]" % r["saved"] if r["saved"] else "")
            if r["ok"] else "FAIL " + r["err"]))

    with open(os.path.join(OUT, "b02_grid_sample_5d_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n저장: b02_grid_sample_5d_result.json")
    print("보존된 5D onnx:", [f for f in os.listdir(OUT) if f.startswith("_gs5d") and f.endswith(".onnx")])


if __name__ == "__main__":
    main()
