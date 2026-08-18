# create_data_bevdet.py → mini 패치 (BEVDet info pkl 생성)

정본 `tools/create_data_bevdet.py`는 `v1.0-trainval` 하드코딩 + `create_groundtruth_database`(무거운
LiDAR seg) 호출이라 nuScenes-**mini** 스모크엔 그대로 못 쓴다. 아래 3곳만 패치한 사본
`tools/create_data_bevdet_mini.py`를 만들어 실행했다.

## 패치 3곳

```diff
 if __name__ == '__main__':
     dataset = 'nuscenes'
-    version = 'v1.0-trainval'
+    version = 'v1.0-mini'
     root_path = './data/nuscenes'
     extra_tag = 'bevdetv3-nuscenes'          # ← v3 (문서 초안의 v2가 아님; dev3.0 실제 태그)
     ...

 def add_ann_adj_info(extra_tag):
-    nuscenes_version = 'v1.0-trainval'
+    nuscenes_version = 'v1.0-mini'
     ...

     # [mini 패치] create_groundtruth_database는 학습 증강용 — FP32 eval에 불필요하므로 생략
-    create_groundtruth_database('NuScenesDataset', root_path, extra_tag,
-                                f'{root_path}/{extra_tag}_infos_train.pkl')
+    # create_groundtruth_database(...)   # 주석처리
```

`tools/data_converter/nuscenes_converter.py::create_nuscenes_infos`는 이미 `v1.0-mini`를 지원
(`available_vers=['v1.0-trainval','v1.0-test','v1.0-mini']`, `mini_train`/`mini_val` 분기) → converter 자체는
무패치.

## 실행

```bash
SP=~/bevf-legacy/lib/python3.10/site-packages
cd ~/capstone-bev/BEVDet
export PYTHONPATH=$PWD                          # ★ tools.data_converter import 위해 repo root 필수
                                                 #    (`python tools/x.py`는 repo root를 sys.path에 안 넣음)
export LD_LIBRARY_PATH=$SP/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
~/bevf-legacy/bin/python tools/create_data_bevdet_mini.py
```

## 데이터 심링크 (재다운로드 회피 — §4.6 BEVFormer가 쓰던 mini 재사용)

```bash
cd ~/capstone-bev/BEVDet/data/nuscenes
ln -s ~/bevformer_work/BEVFormer/data/nuscenes/{maps,samples,sweeps,v1.0-mini} .
# samples 706M · sweeps 4.4G · maps 5.6M — 동일 nuScenes-mini 원본
```

## 산출

| 파일 | samples | scenes | split | bytes |
|------|--------:|-------:|-------|------:|
| `bevdetv3-nuscenes_infos_train.pkl` | 323 | 8 | mini_train | 6,252,981 |
| `bevdetv3-nuscenes_infos_val.pkl`   |  81 | 2 | mini_val   | 1,839,463 |

config `bevdet-r50.py`의 test `ann_file='bevdetv3-nuscenes_infos_val.pkl'`과 파일명 정합.

**캐비앗:** mini 81 val + init 가중치 → 절대 mAP 문헌비교 불가. 파이프라인 관통(walking skeleton) 검증용.
