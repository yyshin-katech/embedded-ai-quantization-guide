#!/usr/bin/env python3
"""Score a COCO-format predictions.json against instances_val2017.json, restricted
to the eval subset's image_ids (from eval_meta.json). Prints the 12 COCOeval stats
and writes a small json. Same scorer for FP32 and every INT8 .dxnn variant, so the
numbers are 1:1 comparable (quantization isolated)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ann", default="/home/yuyeong/embedded-ai-quantization-guide/_workspace/coco/annotations/instances_val2017.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = [m["image_id"] for m in json.load(open(os.path.join(ROOT, "results", "eval_meta.json")))["images"]]
    coco = COCO(args.ann)
    dt = coco.loadRes(args.pred)
    ev = COCOeval(coco, dt, "bbox")
    ev.params.imgIds = ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats  # 12 standard COCO metrics
    res = {"tag": args.tag, "n_images": len(ids),
           "mAP_50_95": round(float(s[0]), 4), "mAP_50": round(float(s[1]), 4),
           "mAP_75": round(float(s[2]), 4), "mAP_s": round(float(s[3]), 4),
           "mAP_m": round(float(s[4]), 4), "mAP_l": round(float(s[5]), 4),
           "AR_1": round(float(s[6]), 4), "AR_10": round(float(s[7]), 4),
           "AR_100": round(float(s[8]), 4), "AR_s": round(float(s[9]), 4),
           "AR_m": round(float(s[10]), 4), "AR_l": round(float(s[11]), 4)}
    out = args.out or os.path.join(ROOT, "results", f"map_{args.tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2), file=sys.stderr)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
