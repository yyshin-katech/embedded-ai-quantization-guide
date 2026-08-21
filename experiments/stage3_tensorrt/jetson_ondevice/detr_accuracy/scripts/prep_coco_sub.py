#!/usr/bin/env python3
# prep_coco_sub.py (HOST) — assemble the on-board eval subset: the first N sorted COCO
# val2017 image_ids (HEAD; disjoint from the calibration tail 100 in detr_sym_export.py),
# copy their JPEGs into a staging dir + write manifest.json, then tar for scp to the board.
import json, os, shutil, sys, tarfile
from pycocotools.coco import COCO

ANN = "_workspace/coco/annotations/instances_val2017.json"
IMG = "_workspace/coco/val2017"
STAGE = "_workspace/stage3_jetson_detr_acc/coco_sub"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

coco = COCO(ANN)
ids = sorted(coco.getImgIds())[:N]                     # head, disjoint from calib tail
os.makedirs(STAGE, exist_ok=True)
manifest = []
for iid in ids:
    fn = coco.loadImgs(iid)[0]["file_name"]
    shutil.copy(os.path.join(IMG, fn), os.path.join(STAGE, fn))
    manifest.append({"image_id": int(iid), "file_name": fn})
json.dump(manifest, open(os.path.join(STAGE, "manifest.json"), "w"))
print("staged %d images -> %s" % (len(manifest), STAGE))

tar_path = "_workspace/stage3_jetson_detr_acc/coco_sub.tar"
with tarfile.open(tar_path, "w") as t:
    t.add(STAGE, arcname="coco_sub")
print("tar ->", tar_path, round(os.path.getsize(tar_path) / 1e6, 1), "MB")
