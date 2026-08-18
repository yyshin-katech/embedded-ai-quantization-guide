#!/usr/bin/env python3
"""Finding #1 fix: AI Hub's ONNX frontend rejects an ORT-quantized INT8 QDQ model
whose output tensor `logits` appears in BOTH graph.value_info AND graph IO
(an ONNX spec violation ORT's quantizer introduces via shape-inference). ORT and
TensorRT tolerate it; AI Hub's compile job fails with:
    Tensors {'logits'} occur in value_info but also in model IO.
Fix = strip the IO-colliding value_info entries. Graph computation is unchanged
(value_info is only redundant shape annotation for already-declared IO tensors).

Usage: python clean_valueinfo_for_aihub.py in.onnx out.onnx
"""
import sys, onnx
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
io = set(i.name for i in g.input) | set(o.name for o in g.output)
before = len(g.value_info)
keep = [v for v in g.value_info if v.name not in io]
removed = [v.name for v in g.value_info if v.name in io]
del g.value_info[:]
g.value_info.extend(keep)
onnx.checker.check_model(m)          # PASS after fix
onnx.save(m, dst)
print(f"value_info {before} -> {len(keep)} (removed IO-colliding: {removed})")
print(f"checker PASS, saved {dst}")
