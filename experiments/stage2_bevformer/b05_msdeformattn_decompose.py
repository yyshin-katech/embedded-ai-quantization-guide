#!/usr/bin/env python3
"""
b05 — MSDeformAttn 표준 op 분해 export (§4.6.2 (a) 검증).

단정:
  - mmcv `multi_scale_deformable_attn_pytorch`(순수 PyTorch 폴백)가 표준 op 분해다
  - 이걸 export 하면 opset 16+ 에서 표준 그래프가 나온다 (커스텀 op 없이)
  - 레벨마다 grid_sample 이 1회씩 호출된다 → GridSample 노드가 num_levels 개
  - 노드 수가 폭증한다 (논리적 1개 op → 수십 개 표준 op)

mmcv 의존 없이, mmcv 원본과 동일한 폴백 함수를 그대로 옮겨 검증한다.
"""
import os, json, warnings
import torch, torch.nn as nn, torch.nn.functional as F

warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """mmcv.ops.multi_scale_deform_attn 의 순수 PyTorch 폴백 (원본 동일)."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H, W) in enumerate(value_spatial_shapes):
        value_l_ = (value_list[level].flatten(2).transpose(1, 2)
                    .reshape(bs * num_heads, embed_dims, H, W))
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
              .sum(-1).view(bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()


class MSDA(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.shapes = shapes
    def forward(self, value, sampling_locations, attention_weights):
        return multi_scale_deformable_attn_pytorch(
            value, self.shapes, sampling_locations, attention_weights)


def main():
    print(f"torch {torch.__version__}\n")
    bs, num_heads, embed_dims = 1, 8, 32
    num_levels, num_points, num_queries = 4, 4, 100
    shapes = [(8, 8), (4, 4), (2, 2), (1, 1)]     # FPN 4레벨(작게)
    num_value = sum(H * W for H, W in shapes)       # 85

    value = torch.randn(bs, num_value, num_heads, embed_dims)
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)

    # 수치 sanity: forward 동작 확인
    out = MSDA(shapes)(value, sampling_locations, attention_weights)
    print(f"forward OK  out={tuple(out.shape)}  (기대 [bs,num_queries,num_heads*embed_dims]=[{bs},{num_queries},{num_heads*embed_dims}])")

    results = {}
    for op in (13, 16, 17):
        path = os.path.join(OUT, f"_msda_op{op}.onnx")
        rec = {"req_opset": op, "ok": False, "err": None}
        try:
            torch.onnx.export(MSDA(shapes), (value, sampling_locations, attention_weights),
                              path, opset_version=op, dynamo=False)
            import onnx
            m = onnx.load(path)
            ops = [n.op_type for n in m.graph.node]
            from collections import Counter
            c = Counter(ops)
            rec.update(ok=True, n_nodes=len(ops), n_gridsample=c.get("GridSample", 0),
                       emit_opset=m.opset_import[0].version, top_ops=c.most_common(8))
            print(f"\nopset {op}: OK  nodes={len(ops)}  GridSample={c.get('GridSample',0)} "
                  f"(기대=num_levels={num_levels})  emit={rec['emit_opset']}")
            print(f"   op 분포: {dict(c.most_common(10))}")
            os.remove(path)
            for e in (".onnx.data", ".data"):
                if os.path.exists(path + e): os.remove(path + e)
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"\nopset {op}: FAIL {rec['err']}")
        results[f"opset{op}"] = rec

    with open(os.path.join(OUT, "b05_msdeformattn_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n저장: b05_msdeformattn_result.json")


if __name__ == "__main__":
    main()
