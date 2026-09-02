#!/usr/bin/env python3
"""One-time: extract the deployable 2-output DETR from detr_sim.onnx.

detr_sim.onnx exports FOUR graph outputs: logits [1,100,92], pred_boxes [1,100,4],
and two export leftovers — onnx::MatMul_3387 [1,100,256] and encoder_hidden_states
[1,850,256]. The latter alone is 850*256*4 = 870,400 B, ~24x the real output payload
(logits+boxes = 38,400 B). Left in, they would (a) bloat every D2H transfer and wreck
the regime measurement and (b) risk confusing dx_com's IO contract. onnx.utils.extract_model
prunes to exactly the pixel_values -> (logits, pred_boxes) subgraph.
"""
import argparse
import onnx
from onnx.utils import extract_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/yuyeong/embedded-ai-quantization-guide/_workspace/stage2/detr_sim.onnx")
    ap.add_argument("--dst", default="/home/yuyeong/dxm1_detr/detr_2out.onnx")
    args = ap.parse_args()
    extract_model(args.src, args.dst,
                  input_names=["pixel_values"],
                  output_names=["logits", "pred_boxes"])
    m = onnx.load(args.dst, load_external_data=False)
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print("wrote", args.dst)
    print("nodes:", len(m.graph.node), "outputs:", outs)


if __name__ == "__main__":
    main()
