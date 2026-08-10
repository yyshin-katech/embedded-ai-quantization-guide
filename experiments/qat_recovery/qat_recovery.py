#!/usr/bin/env python3
"""QAT 회복 실험 — "QAT는 PTQ가 잃은 정확도를 되찾는다"를 실제로 시험한다.

가이드 2.5절은 QAT/STE를 설명하지만, 실습은 합성 텐서로 gradient가 통과하는지만 본다.
진짜 val 셋이 생겼으니 주장 자체를 측정할 수 있다.

⚠️ 한계 (반드시 같이 읽어야 함): ImageNet train split이 없어 **val을 쪼개서** 학습한다.
   클래스당 40장 학습 / 10장 평가로 서로소 분할하므로 평가셋 누수는 없지만,
   val 분포로 학습한 모델이라 절대 top-1을 문헌값과 비교하면 안 된다.
   여기서 읽을 것은 FP32 → PTQ → QAT 세 값의 **상대 관계**뿐이다.

양자화 구성은 PTQ 실험과 맞춘다: weight per-channel 대칭 INT8 + activation per-tensor
비대칭 UINT8(입력 후 각 블록 출력). STE는 clamp 범위 밖 gradient를 0으로 죽인다.
"""
import json, os, pathlib, time, numpy as np, torch, torch.nn as nn, torchvision

WORK = pathlib.Path(os.path.expanduser("~/stage1-work"))
CACHE = WORK / "data" / "cache"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# 배치 96/250 → 48/128. 결과 때문이 아니라 하드웨어 때문이었다: 이 머신의 RTX 3060이
# 이 워크로드에서 Xid 79(GPU has fallen off the bus)로 3회 죽었다(08-04, 08-10 ×2).
#    🔴 그런데 배치 축소는 **무효한 레버로 판명됐다**(2026-08-10 실측 반증).
#       BS 96→48로 반감해도 최대 전력은 138W → 129.7W(약 8W)만 줄었고 — 배치가 작아지면
#       GPU가 커널을 더 자주 띄워 빈자리를 메운다 — BS=48에서도 ep1 step 500/833에서 죽었다.
#       생존 시간만 40초 → 190초로 늘었을 뿐이다. 텔레메트리 98행 중 73행이
#       `0x4`(SW Power Cap): 부하 26초 뒤 ~130W 상한에 붙어 죽을 때까지 안 떨어졌다.
#       → 진짜 레버는 배치가 아니라 전력 상한(`sudo nvidia-smi -pl`)과 팬 하한이다.
#       상세는 memory `gpu-xid79-fallen-off-bus` / `HANDOFF.md`.
#    ⚠️ 배치를 바꾸면 step 수가 2배(416→833)로 늘어 학습 궤적이 달라진다.
#       QAT 팔과 대조군 팔은 **반드시 같은 배치**로 돌려야 비교가 성립한다.
#    환경변수로 덮어쓸 수 있다: QAT_BS_TRAIN=96 python3 qat_recovery.py
BS_TRAIN = int(os.environ.get("QAT_BS_TRAIN", 48))
BS_EVAL = int(os.environ.get("QAT_BS_EVAL", 128))
EPOCHS = int(os.environ.get("QAT_EPOCHS", 2))
# 🔴 캘리브레이션 장수를 배치와 분리한다. 예전엔 `range(0, 20*BS_EVAL, BS_EVAL)`이라
#    BS_EVAL을 250→128로 줄이자 관측 이미지가 5,000→2,560장으로 같이 반토막 나
#    PTQ top-1이 68.46%→68.37%로 움직였다(배치는 결과를 바꿔선 안 되는 노브다).
CALIB_N = int(os.environ.get("QAT_CALIB_N", 5000))
RESULT_JSON = WORK / f"qat_recovery_result_bs{BS_TRAIN}.json"


class FakeQuantSTE(torch.autograd.Function):
    """대칭/비대칭 fake-quant + STE. clamp 밖은 gradient를 0으로 죽인다."""
    @staticmethod
    def forward(ctx, x, scale, zp, q_min, q_max):
        u = x / scale + zp
        q = torch.clamp(torch.round(u), q_min, q_max)
        ctx.save_for_backward((u >= q_min) & (u <= q_max))
        return (q - zp) * scale

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return g * mask.to(g.dtype), None, None, None, None


def fq_weight_per_channel(w, bits=8):
    q_max = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = torch.clamp(flat.detach().abs().max(dim=1).values / q_max, min=1e-12)
    shape = [-1] + [1] * (w.dim() - 1)
    return FakeQuantSTE.apply(w, scale.reshape(shape), 0, -q_max, q_max)


class ActFakeQuant(nn.Module):
    """activation per-tensor 비대칭 UINT8. 관측 구간엔 EMA로 min/max를 모은다."""
    def __init__(self, momentum=0.05):
        super().__init__()
        self.register_buffer("lo", torch.tensor(float("nan")))
        self.register_buffer("hi", torch.tensor(float("nan")))
        self.momentum, self.observing = momentum, True

    def forward(self, x):
        if self.observing:
            lo, hi = x.detach().amin(), x.detach().amax()
            if torch.isnan(self.lo):
                self.lo, self.hi = lo, hi
            else:
                self.lo += self.momentum * (lo - self.lo)
                self.hi += self.momentum * (hi - self.hi)
        if torch.isnan(self.lo):
            return x
        lo = torch.minimum(self.lo, torch.zeros_like(self.lo))
        hi = torch.maximum(self.hi, torch.zeros_like(self.hi))
        scale = torch.clamp((hi - lo) / 255.0, min=1e-12)
        zp = torch.round(-lo / scale)
        return FakeQuantSTE.apply(x, scale, zp, 0, 255)


class QConv2d(nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(x, fq_weight_per_channel(self.weight), self.bias)


class QLinear(nn.Linear):
    def forward(self, x):
        return nn.functional.linear(x, fq_weight_per_channel(self.weight), self.bias)


def quantize_model(model):
    """Conv2d/Linear를 fake-quant 버전으로 교체하고, 블록 출력에 act fake-quant를 끼운다."""
    def swap(mod):
        for name, ch in mod.named_children():
            if isinstance(ch, nn.Conv2d):
                q = QConv2d(ch.in_channels, ch.out_channels, ch.kernel_size, ch.stride,
                            ch.padding, ch.dilation, ch.groups, ch.bias is not None)
                q.load_state_dict(ch.state_dict()); setattr(mod, name, q)
            elif isinstance(ch, nn.Linear):
                q = QLinear(ch.in_features, ch.out_features, ch.bias is not None)
                q.load_state_dict(ch.state_dict()); setattr(mod, name, q)
            else:
                swap(ch)
    swap(model)
    # activation 양자화 지점: 입력 + 각 stage 출력 + avgpool 출력
    model.qin = ActFakeQuant()
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        setattr(model, f"q_{lname}", ActFakeQuant())
    model.q_pool = ActFakeQuant()
    return model


def qforward(m, x):
    x = m.qin(x)
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.q_layer1(m.layer1(x)); x = m.q_layer2(m.layer2(x))
    x = m.q_layer3(m.layer3(x)); x = m.q_layer4(m.layer4(x))
    x = m.q_pool(torch.flatten(m.avgpool(x), 1))
    return m.fc(x)


def set_observing(m, flag):
    for mod in m.modules():
        if isinstance(mod, ActFakeQuant):
            mod.observing = flag


@torch.no_grad()
def evaluate(fwd, imgs, labels, idx, mean, std):
    hits = np.zeros(len(idx), dtype=bool)
    for s in range(0, len(idx), BS_EVAL):
        sel = idx[s:s + BS_EVAL]
        x = torch.from_numpy(np.ascontiguousarray(imgs[sel])).to(DEV)
        x = x.permute(0, 3, 1, 2).float().div_(255.0).sub_(mean).div_(std)
        hits[s:s + len(sel)] = fwd(x).argmax(1).cpu().numpy() == labels[sel]
    return hits


def main():
    imgs = np.load(CACHE / "squash.npy", mmap_mode="r")
    labels = np.load(CACHE / "labels.npy").astype(np.int64)
    mean, std = MEAN.to(DEV), STD.to(DEV)

    # 클래스당 40/10 서로소 분할
    rng = np.random.default_rng(0)
    tr, ev = [], []
    for c in range(1000):
        w = np.where(labels == c)[0]
        perm = rng.permutation(w)
        tr.extend(perm[:40].tolist()); ev.extend(perm[40:].tolist())
    tr = np.array(sorted(tr), dtype=np.int64); ev = np.array(sorted(ev), dtype=np.int64)
    assert not (set(tr.tolist()) & set(ev.tolist()))
    print(f"학습 {len(tr)}장 / 평가 {len(ev)}장 (서로소 확인 ✅), device={DEV}\n")

    W = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    # ---- (1) FP32 기준선 ----
    fp32 = torchvision.models.resnet18(weights=W).eval().to(DEV)
    h = evaluate(lambda x: fp32(x), imgs, labels, ev, mean, std)
    fp32_top1 = h.mean() * 100
    print(f"(1) FP32                     top-1 = {fp32_top1:.2f}%")

    # ---- (2) PTQ: fake-quant만 씌우고 학습 없음 (activation 범위만 관측) ----
    ptq = quantize_model(torchvision.models.resnet18(weights=W)).eval().to(DEV)
    set_observing(ptq, True)
    with torch.no_grad():                      # 캘리브레이션: 학습셋 앞 CALIB_N장 (배치와 무관)
        for s in range(0, CALIB_N, BS_EVAL):
            sel = tr[s:min(s + BS_EVAL, CALIB_N)]
            x = torch.from_numpy(np.ascontiguousarray(imgs[sel])).to(DEV)
            qforward(ptq, x.permute(0, 3, 1, 2).float().div_(255.0).sub_(mean).div_(std))
    set_observing(ptq, False)
    ptq_top1 = evaluate(lambda x: qforward(ptq, x), imgs, labels, ev, mean, std).mean() * 100
    print(f"(2) PTQ (fake-quant, 학습 없음) top-1 = {ptq_top1:.2f}%  Δ={ptq_top1-fp32_top1:+.2f}%p")

    # ---- (3) QAT: 같은 구성에서 STE로 파인튜닝 ----
    qat = quantize_model(torchvision.models.resnet18(weights=W)).to(DEV)
    qat.load_state_dict(ptq.state_dict())      # PTQ와 같은 시작점(관측된 act 범위 포함)
    set_observing(qat, False)                  # 범위 고정 후 weight만 학습
    opt = torch.optim.SGD(qat.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    steps_per_epoch = len(tr) // BS_TRAIN
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)

    for ep in range(EPOCHS):
        qat.train(); perm = np.random.default_rng(ep).permutation(tr)
        t0, run, seen = time.time(), 0.0, 0
        for i in range(steps_per_epoch):
            sel = np.sort(perm[i * BS_TRAIN:(i + 1) * BS_TRAIN])
            x = torch.from_numpy(np.ascontiguousarray(imgs[sel])).to(DEV)
            x = x.permute(0, 3, 1, 2).float().div_(255.0).sub_(mean).div_(std)
            y = torch.from_numpy(labels[sel]).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = lossf(qforward(qat, x), y)
            loss.backward(); opt.step(); sched.step()
            run += loss.item(); seen += 1
            if i % 100 == 0:
                print(f"    ep{ep} step {i:4d}/{steps_per_epoch}  loss={run/seen:.4f}", flush=True)
        qat.eval()
        t1 = evaluate(lambda x: qforward(qat, x), imgs, labels, ev, mean, std).mean() * 100
        print(f"(3) QAT epoch {ep}            top-1 = {t1:.2f}%  Δ(FP32)={t1-fp32_top1:+.2f}%p  "
              f"Δ(PTQ)={t1-ptq_top1:+.2f}%p   [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n요약: FP32 {fp32_top1:.2f}% → PTQ {ptq_top1:.2f}% → QAT {t1:.2f}%")
    loss_pp = fp32_top1 - ptq_top1
    rec = (t1 - ptq_top1) / loss_pp * 100 if loss_pp > 0 else float("nan")
    print(f"PTQ 손실 {loss_pp:.2f}%p 중 QAT가 회복한 비율 = {rec:.1f}%")
    if loss_pp < 0.10:
        print(f"   🔴 이 회복률은 읽지 말 것 — 분모({loss_pp:.2f}%p)가 측정 노이즈(±0.05%p) 크기다.")
    print("⚠️ val을 쪼개 학습했으므로 절대값은 문헌과 비교 불가 — 상대 관계만 유효")
    print("⚠️ QAT 팔만 추가 학습을 받았다 — '추가 학습' 효과와 분리하려면 "
          "qat_control_finetune.py(동일 설정, fake-quant만 제거)를 같은 배치로 돌려야 한다.")

    json.dump({"bs_train": BS_TRAIN, "bs_eval": BS_EVAL, "epochs": EPOCHS, "calib_n": CALIB_N,
               "fp32": round(fp32_top1, 4), "ptq": round(ptq_top1, 4), "qat": round(t1, 4),
               "n_train": len(tr), "n_eval": len(ev)},
              open(RESULT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {RESULT_JSON.name} (대조군이 이 파일을 읽는다)")


if __name__ == "__main__":
    main()
