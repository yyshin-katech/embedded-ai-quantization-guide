#!/usr/bin/env python3
"""QAT 회복 실험의 빠진 대조군 — "추가 학습" 효과와 "양자화 인식 학습" 효과를 분리한다.

qat_recovery.py는 FP32(학습 없음) vs QAT(2 에폭 학습)를 비교했다. 그런데 QAT 팔만
val 40,000장으로 추가 학습을 받았으므로, QAT가 FP32를 이긴 것이 '양자화를 학습으로
극복했다'인지 '같은 분포로 더 학습했을 뿐'인지 구분되지 않는다.

이 스크립트는 fake-quant만 제거하고 나머지를 전부 동일하게 맞춘 FP32 파인튜닝을 돌린다
(같은 분할·같은 step 수·같은 optimizer/LR/schedule/batch/epoch/전처리).
읽을 값은 하나다: QAT가 FP32-finetune보다 높은가 낮은가.
  - 비슷하다  → QAT의 이득은 전부 '추가 학습'이고 양자화 인식은 기여가 없다
  - 더 낮다   → 그 차이가 이 구성에서 양자화가 실제로 가져간 대가다

🔴 QAT 수치를 상수로 박지 않는다. 같은 배치로 돌린 qat_recovery.py의 JSON을 읽는다 —
   배치를 96→48로 바꿨을 때 낡은 69.13%(BS=96)와 비교하면 조용히 오답이 나온다.
"""
import json, sys, time, numpy as np, torch, torch.nn as nn, torchvision
from qat_recovery import CACHE, DEV, MEAN, STD, BS_TRAIN, BS_EVAL, EPOCHS, RESULT_JSON, evaluate


def load_qat_result():
    """같은 (배치, 에폭)으로 끝난 QAT 실행 결과만 받아들인다."""
    if not RESULT_JSON.exists():
        sys.exit(f"❌ {RESULT_JSON.name}이 없다. 먼저 같은 배치로 qat_recovery.py를 돌려라:\n"
                 f"   python3 qat_recovery.py 2>&1 | tee qat_recovery_bs{BS_TRAIN}.log")
    r = json.load(open(RESULT_JSON))
    if (r["bs_train"], r["epochs"]) != (BS_TRAIN, EPOCHS):
        sys.exit(f"❌ 설정 불일치 — JSON은 BS={r['bs_train']}/EP={r['epochs']}, "
                 f"지금은 BS={BS_TRAIN}/EP={EPOCHS}. 두 팔은 같은 설정이어야 비교가 성립한다.")
    print(f"QAT 참조값 ({RESULT_JSON.name}, BS={r['bs_train']}·EP={r['epochs']}): "
          f"FP32 {r['fp32']:.2f}% → PTQ {r['ptq']:.2f}% → QAT {r['qat']:.2f}%\n")
    return r


def main():
    qat = load_qat_result()
    QAT_EP1, QAT_FP32 = qat["qat"], qat["fp32"]
    imgs = np.load(CACHE / "squash.npy", mmap_mode="r")
    labels = np.load(CACHE / "labels.npy").astype(np.int64)
    mean, std = MEAN.to(DEV), STD.to(DEV)

    # qat_recovery.py와 비트 단위로 같은 분할 (rng seed 0, 클래스당 40/10)
    rng = np.random.default_rng(0)
    tr, ev = [], []
    for c in range(1000):
        perm = rng.permutation(np.where(labels == c)[0])
        tr.extend(perm[:40].tolist()); ev.extend(perm[40:].tolist())
    tr = np.array(sorted(tr), dtype=np.int64); ev = np.array(sorted(ev), dtype=np.int64)
    assert not (set(tr.tolist()) & set(ev.tolist()))
    print(f"학습 {len(tr)}장 / 평가 {len(ev)}장 (서로소 ✅, qat_recovery.py와 동일 분할)\n")

    W = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=W).to(DEV)

    # 🔴 .eval() 필수 — 빼면 BatchNorm이 running stats 대신 배치 통계를 써서
    #    학습 전 기준선이 68.52% → 68.04%로 어긋난다(1차 실행에서 실제로 겪음).
    model.eval()
    base = evaluate(lambda x: model(x), imgs, labels, ev, mean, std).mean() * 100
    print(f"(1) FP32 학습 전            top-1 = {base:.2f}%   "
          f"← qat_recovery.py의 {QAT_FP32:.2f}%와 일치해야 함")
    if abs(base - QAT_FP32) > 0.05:
        print(f"    ⚠️ {QAT_FP32:.2f}%와 {base-QAT_FP32:+.2f}%p 어긋났다 — "
              f"분할·전처리·eval 모드를 먼저 의심하라")

    # QAT와 동일한 학습 설정 — fake-quant만 없다
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    steps = len(tr) // BS_TRAIN
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps)

    for ep in range(EPOCHS):
        model.train(); perm = np.random.default_rng(ep).permutation(tr)
        t0, run, seen = time.time(), 0.0, 0
        for i in range(steps):
            sel = np.sort(perm[i * BS_TRAIN:(i + 1) * BS_TRAIN])
            x = torch.from_numpy(np.ascontiguousarray(imgs[sel])).to(DEV)
            x = x.permute(0, 3, 1, 2).float().div_(255.0).sub_(mean).div_(std)
            y = torch.from_numpy(labels[sel]).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y)
            loss.backward(); opt.step(); sched.step()
            run += loss.item(); seen += 1
            if i % 100 == 0:
                print(f"    ep{ep} step {i:4d}/{steps}  loss={run/seen:.4f}", flush=True)
        model.eval()
        ft = evaluate(lambda x: model(x), imgs, labels, ev, mean, std).mean() * 100
        print(f"(2) FP32 finetune ep{ep}      top-1 = {ft:.2f}%  Δ(학습 전)={ft-base:+.2f}%p  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\n=== 분리 결과 ===")
    print(f"FP32 학습 전        {base:.2f}%")
    print(f"FP32 + 파인튜닝     {ft:.2f}%   ({ft-base:+.2f}%p)  ← 순수 '추가 학습' 효과")
    print(f"QAT (fake-quant)   {QAT_EP1:.2f}%   ({QAT_EP1-base:+.2f}%p)")
    print(f"\nQAT − FP32finetune = {QAT_EP1-ft:+.2f}%p  ← 같은 학습량에서 양자화가 가져간 대가")
    if abs(QAT_EP1 - ft) < 0.30:
        print("→ 두 팔이 사실상 같다. QAT의 이득은 '추가 학습'이고, 양자화 인식 자체의 기여는 못 잰다.")
    elif QAT_EP1 < ft:
        print("→ QAT가 낮다. 그 격차가 학습으로도 못 메운 양자화 손실이다.")
    else:
        print("→ QAT가 높다. 정규화 효과 가설 — 반복 측정 없이는 단정 불가.")


if __name__ == "__main__":
    main()
