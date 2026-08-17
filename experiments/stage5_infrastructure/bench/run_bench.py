# bench/run_bench.py  (CLI 엔트리포인트: config.yaml 순회 → results/*.json)
# 학습가이드 §4-5 그대로 — itertools.product 3중 순회 + --only-backend.
# 검증 정정 1건: 문서의 load_backend(key)는 zero-arg 생성이지만, TRT는 INT8 캘리브레이터가
# 데이터에 의존하므로 loader.calib_feed를 주입한다(문서 §4-5의 `from data import Loader` 전제를
# 실제 배선으로 옮긴 것). 나머지 순회/제외/봉인 로직은 문서와 동일.
import argparse
import importlib
import itertools
import pathlib
import sys
import yaml

BACKENDS = {
    "trt":   ("backends.trt",   "TRTBackend"),
    "tidl":  ("backends.tidl",  "TIDLBackend"),
    "qnn":   ("backends.qnn",   "QNNBackend"),
    "drpai": ("backends.drpai", "DRPAIBackend"),
}


def load_backend(key, loader=None, calib_cache=None):
    mod, cls = BACKENDS[key]
    C = getattr(importlib.import_module(mod), cls)
    if key == "trt":
        # 데이터 의존부(INT8 캘리브 피드) 주입 — 문서 §4-5의 데이터층을 실제로 연결.
        return C(in_name="input",
                 calib_feed=(loader.calib_feed(200) if loader else None),
                 calib_cache=calib_cache)
    return C()


def is_excluded(backend, precision, exclude):
    for rule in exclude or []:
        if rule.get("backend") == backend and rule.get("precision") == precision:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only-backend", default=None,
                    help="config를 무시하고 이 백엔드만 (예: CI에서 trt만)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg.get("defaults", {})
    warmup, iters = d.get("warmup", 20), d.get("iters", 200)
    eval_n = d.get("eval_n", 5000)
    exclude = cfg.get("exclude", [])
    backends = [args.only_backend] if args.only_backend else cfg["backends"]

    from data import Loader, Evaluator  # noqa  (§4-5: 프로젝트 데이터층)

    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    cache_dir = pathlib.Path(".trt_cache"); cache_dir.mkdir(exist_ok=True)
    n_ok = n_skip = 0
    # 3중 순회 = 매트릭스의 셀 하나하나
    for m, prec, bk in itertools.product(cfg["models"], cfg["precisions"], backends):
        if is_excluded(bk, prec, exclude):
            print(f"skip (excluded): {m['name']} × {bk} × {prec}")
            n_skip += 1
            continue
        loader = Loader(m, eval_n=eval_n)
        calib_cache = str(cache_dir / f"{m['name']}__{bk}__{prec}.calib")
        be = load_backend(bk, loader=loader, calib_cache=calib_cache)
        try:
            be.build(m["onnx"], prec, m.get("calib"))
            res = be.measure(m["name"], prec, loader, Evaluator(),
                             warmup=warmup, iters=iters)
        except NotImplementedError as e:
            # stub 백엔드(보드 없음): 회색 결과로 봉인하고 계속
            from backends.base import BenchResult
            res = BenchResult(m["name"], be.soc_name, prec,
                              float("nan"), float("nan"), float("nan"), 0.0,
                              notes=f"not implemented: {e}")
        fn = be.collect(res)            # base.py의 공통 collect
        print(f"wrote {fn}  (lat={res.latency_ms} acc={res.accuracy} note={res.notes})")
        n_ok += 1
    print(f"done: {n_ok} results, {n_skip} skipped")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
