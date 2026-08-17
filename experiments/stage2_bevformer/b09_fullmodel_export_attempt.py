"""
B3 / Tier B — BEVFormer-tiny 전체 모델 ONNX export 시도 (모델 레벨 벽 실측)
op 레벨(b01~b06)은 이미 확정. 여기선 "바닐라 BEVFormer 전체를 torch.onnx.export로
내보내면 정확히 어디서 막히나"를 실측 로그로 남긴다.

핵심 가설(코드 구조에서): forward_test/simple_test 는
  - img_metas(list[dict], 비텐서)를 소비하고
  - can_bus/lidar2img 를 forward 内部에서 reference point 계산에 쓰며(=기능적 입력)
  - self.prev_frame_info 로 시간축 재귀(stateful) → 트레이스 불가
그래서 텐서-in/텐서-out 그래프가 아니다. DerryHub 포크가 별도 래퍼로 이걸 텐서화한다.

CPU export 강제(B2 실측: mmcv 커스텀 op은 CUDA에서 상수 baked → CPU만 유효).
150초 alarm으로 바운드(전체 모델 CPU 트레이스가 길어지면 노드 폭증으로 판정).
"""
import os, sys, signal, json, traceback, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter, collate, MMDataParallel
import projects.mmdet3d_plugin  # registry 등록 트리거
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

RESULT = {"stage": "B3 fullmodel export", "device": "cpu(forced)", "outcome": None, "detail": None,
          "graph_inputs": None, "n_nodes": None}

class TimeBox(Exception): pass
def _alarm(sig, frm): raise TimeBox("export exceeded 150s wall (CPU full-model trace)")
signal.signal(signal.SIGALRM, _alarm); signal.alarm(150)

cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny.py')
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, 'ckpts/bevformer_tiny_epoch_24.pth', map_location='cpu')
model.eval().cpu()
print("[build] BEVFormer-tiny 빌드+체크포인트 로드 OK (cpu)")

# 실제 val 샘플 1개 확보
ds = build_dataset(cfg.data.test)
sample = ds[0]
img_dc = sample['img'][0]            # DataContainer
metas_dc = sample['img_metas'][0]    # DataContainer
img = img_dc.data.unsqueeze(0).cpu() # [1, num_cam, 3, H, W]
img_metas = [metas_dc.data]          # list[dict]  (simple_test 는 img_metas=list 기대)
print(f"[sample] img {tuple(img.shape)}, img_metas keys = {sorted(img_metas[0].keys())}")
# lidar2img / can_bus 가 실제로 head 내부 입력인지 확인
for k in ('lidar2img', 'can_bus'):
    v = img_metas[0].get(k, None)
    print(f"         img_metas['{k}'] present = {v is not None} "
          f"(type={type(v).__name__})")

class ExportWrap(nn.Module):
    """img 만 텐서 입력으로 두고 img_metas/prev_bev 는 baked(=포크 래퍼가 텐서화하는 대상)."""
    def __init__(self, m, metas):
        super().__init__(); self.m = m; self.metas = metas
    def forward(self, img):
        # 시간축 재귀 없이 단일 프레임 (prev_bev=None) — 그래도 head 는 img_metas[lidar2img/can_bus] 소비
        outs = self.m.pts_bbox_head(self.m.extract_feat(img=img, img_metas=self.metas),
                                    self.metas, prev_bev=None)
        return outs['bev_embed']

wrap = ExportWrap(model, img_metas).eval()
try:
    with torch.no_grad():
        y = wrap(img)
    print(f"[fwd] CPU forward OK, bev_embed {tuple(y.shape)} — 이제 export 시도")
except Exception as e:
    RESULT["outcome"] = "forward_failed"
    RESULT["detail"] = f"{type(e).__name__}: {e}"
    print("[fwd] FORWARD 실패:", RESULT["detail"]); traceback.print_exc()
    signal.alarm(0); open('b09_result.json','w').write(json.dumps(RESULT, indent=2, default=str)); sys.exit(0)

onnx_path = 'bevformer_tiny_fullmodel_attempt.onnx'
try:
    with torch.no_grad():
        torch.onnx.export(wrap, (img,), onnx_path, opset_version=16,
                          input_names=['img'], output_names=['bev_embed'],
                          do_constant_folding=False, verbose=False)
    signal.alarm(0)
    import onnx
    m = onnx.load(onnx_path)
    gi = [i.name for i in m.graph.input]
    nn_ = len(m.graph.node)
    RESULT.update(outcome="exported_but_check_inputs", graph_inputs=gi, n_nodes=nn_,
                  detail=f"export 성공. 그래프 입력={gi} (img_metas가 입력에서 사라졌으면 baked). 노드 {nn_}")
    print("[export] 성공:", RESULT["detail"])
except TimeBox as e:
    RESULT.update(outcome="timeboxed", detail=str(e))
    print("[export] 바운드 초과:", e)
except Exception as e:
    signal.alarm(0)
    RESULT.update(outcome="export_failed", detail=f"{type(e).__name__}: {str(e)[:600]}")
    print("[export] 실패:", RESULT["detail"]); traceback.print_exc()

open('b09_result.json','w').write(json.dumps(RESULT, indent=2, default=str))
print("[done] b09_result.json 기록")
