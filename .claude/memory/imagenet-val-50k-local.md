---
name: imagenet-val-50k-local
description: "진짜 ImageNet val 50,000장이 ~/stage1-work/data/에 있다 (원본 tar + synset 재배치 + 무손실 전처리 캐시, 총 27GB) — 라벨 매핑·전처리 규약 포함, 재다운로드 불필요"
metadata: 
  node_type: memory
  type: reference
---

ImageNet-1k validation 전량이 현 머신([[machine-ai-lap-rtx3080]])에 있다. **저장소 밖**
(`~/stage1-work/data/`, 27GB; `val_full` 6.4GB 확인)이라 커밋 대상 아님. 2·3·4단계 정확도 측정에 재사용.

| 경로 | 내용 |
|------|------|
| `data/imagenet/ILSVRC2012_img_val.tar` | 원본 6,744,924,160 B, MD5 `29b22e2961454d5413ddabcf34fc5622`(공식값 일치) |
| `data/val_full/<wnid>/*.JPEG` | synset 폴더로 재배치, 1000×50 = 50,000장 |
| `data/labels/val_synset_map.txt` | `파일명 wnid 인덱스` — devkit `meta.mat` + ground_truth에서 유도 |
| `data/labels/val.txt` | Caffe 배포판 라벨(독립 출처). devkit 유도분과 50,000건 전부 일치 교차검증 |
| `data/cache/{squash,tv}.npy` | 전처리 캐시 각 7.1GB, NHWC uint8 memmap. 24장 표본에서 직접 디코딩과 비트 동일 |
| `data/cache/calib{200,1000}_idx.npy` | 캘리브 분할(클래스별 첫 장). calib200 ⊂ calib1000 |

**라벨 인덱스 규약:** 정렬된 WNID 0-based = torchvision `ImageFolder` 규약. 다른 규약을 섞으면
top-1이 0.1% 근처로 무너진다.

**전처리 두 종류를 반드시 구분** (`~/stage1-work/prep_cache.py`):
- `crop_tv` = 짧은 변 256 bilinear + center crop 224 → **torchvision 공식 규약. ResNet18 FP32 top-1
  69.81% = 공개값 69.758%와 0.05%p 일치.** 정확도 주장엔 이걸 쓴다.
- `crop_squash` = 종횡비 무시 256×256 + `[16:240]` crop → 가이드 문서의 `preprocess()`. **−1.07%p 손실**
  (p=1.6e-14), 양자화 손실보다 9배 크다.

출처는 사용자 Synology 공유 링크(MD5 검증 통과). **다시 받을 필요 없다.** 실측 결과는
[[stage1-quantization-hands-on]].
