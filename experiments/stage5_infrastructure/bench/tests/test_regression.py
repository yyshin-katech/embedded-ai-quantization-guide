# bench/tests/test_regression.py — 학습가이드 §4-7 그대로(검증: 정정 없이 통과/실패 재현).
# (A) 임계값 회귀 + (B) pytest-regressions 골든 파일 이중화.
# 검증 정정(문서 밖 메타): 문서는 pytest-regressions "v3.0+"라 하나 2026-08-17 PyPI 최신은
#   2.11.0(v3.0 미존재). dataframe_regression fixture는 2.11.0에 정상 존재 → 코드는 그대로 동작.
import glob
import json
import pathlib
import pandas as pd
import pytest

MAP_DROP_TOLERANCE = 0.01      # 1%p 절대 하락 허용치
LAT_REGRESS_FACTOR = 1.10      # latency 10% 이상 악화도 실패(선택)

BASELINE = pathlib.Path(__file__).parent / "baseline_matrix.csv"


def _load_current() -> pd.DataFrame:
    rows = [json.load(open(p)) for p in glob.glob("results/*.json")]
    assert rows, "results/*.json 없음 — 벤치를 먼저 실행하라"
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def merged():
    base = pd.read_csv(BASELINE)
    cur = _load_current()
    key = ["model", "soc", "precision"]
    return base.merge(cur, on=key, suffixes=("_base", "_cur"))


# --- (A) 임계값 기반 회귀 ---
def test_no_map_regression(merged):
    """accuracy(top-1/mAP)가 baseline 대비 1%p 이상 하락한 케이스가 있으면 실패."""
    m = merged.dropna(subset=["accuracy_base", "accuracy_cur"]).copy()
    m["drop"] = m["accuracy_base"] - m["accuracy_cur"]
    bad = m[m["drop"] > MAP_DROP_TOLERANCE]
    assert bad.empty, (
        "accuracy 회귀 감지:\n"
        + bad[["model", "soc", "precision",
               "accuracy_base", "accuracy_cur", "drop"]].to_string(index=False)
    )


def test_no_latency_regression(merged):
    """latency가 baseline 대비 10% 이상 악화되면 실패(선택)."""
    m = merged.dropna(subset=["latency_ms_base", "latency_ms_cur"])
    bad = m[m["latency_ms_cur"] > m["latency_ms_base"] * LAT_REGRESS_FACTOR]
    assert bad.empty, (
        "Latency 회귀 감지:\n"
        + bad[["model", "soc", "precision",
               "latency_ms_base", "latency_ms_cur"]].to_string(index=False)
    )


# --- (B) 골든 파일 비교 (pytest-regressions) ---
def test_matrix_matches_golden(dataframe_regression):
    """현재 매트릭스를 골든과 통째로 비교. 의도적 변경은 --force-regen으로만 갱신.

    dataframe_regression fixture는 pytest-regressions(2026-08-17 최신 2.11.0)가 제공.
    최초 실행(또는 --force-regen) 시 tests/test_regression/ 아래 골든을 생성, 이후 비교.
    NaN(보드필요) 행은 골든 비교에서 흔들림이 없어 그대로 둔다.
    """
    cur = _load_current().sort_values(["model", "soc", "precision"]).reset_index(drop=True)
    dataframe_regression.check(
        cur[["model", "soc", "precision", "latency_ms", "accuracy"]],
        default_tolerance=dict(atol=1e-3, rtol=1e-3),
    )
