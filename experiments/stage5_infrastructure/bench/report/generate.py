# bench/report/generate.py — 학습가이드 §4-6 실측 검증본(2026-08-17, AI-LAP/RTX3080).
# results/*.json → pandas 피벗 → 마크다운 + CSV + HTML 3종. CSV는 회귀 baseline 소스.
#
# 정정(6): 문서 원안의 pivot_table은 dropna 기본값(True)이라 값이 전부 NaN인 행을
#   조용히 버린다 → stub 백엔드(tidl/qnn/drpai)의 '보드필요' 회색 행이 matrix.md/html에서
#   통째로 사라진다. 그런데 문서 §5-1은 "회색(보드필요) 행도 매트릭스에 남긴다 — 빈칸이
#   아니라 명시적 '보드필요'가 정직한 보고"라고 명시한다. 즉 §4-6 코드가 §5-1 원칙을
#   스스로 위반한다(실행해 봐야 드러나는 자기모순). → 모든 pivot에 dropna=False.
#   CSV(long-form)는 원래도 6행 전부 보존하므로 회귀 baseline엔 영향 없다.
import glob
import json
import pathlib
import pandas as pd

RESULTS_DIR = "results"
OUT_DIR = pathlib.Path("report")


def collect(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """results/*.json → 하나의 DataFrame (긴 형식)."""
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/*.json"))]
    return pd.DataFrame(rows)


def _highlight_html(df_long: pd.DataFrame) -> str:
    """HTML 대시보드: 행별 최저 latency를 초록. pandas Styler는 jinja2 필요(3절)."""
    lat = df_long.pivot_table(index=["model", "soc"], columns="precision",
                              values="latency_ms", dropna=False)  # 정정(6): 회색 행 보존
    styler = (
        lat.style
        .format("{:.4f}", na_rep="보드필요")   # NaN 셀 = 명시적 '보드필요'(§5-1 정직성)
        .highlight_min(axis=1, color="#c6efce")
        .set_caption("Latency (ms, median) — 행별 최저값 강조")
        .set_table_styles([
            {"selector": "th", "props": [("background", "#f2f2f2"),
                                         ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "right"),
                                         ("padding", "4px 10px")]},
        ])
    )
    return styler.to_html()


def main():
    df = collect()
    if df.empty:
        raise SystemExit("no results/*.json — run_bench.py 먼저 실행")
    OUT_DIR.mkdir(exist_ok=True)

    # (1) CSV — 정렬해 저장 (회귀 baseline 소스). 정렬은 diff 안정성.
    key = ["model", "soc", "precision"]
    df_sorted = df.sort_values(key).reset_index(drop=True)
    df_sorted.to_csv(OUT_DIR / "matrix.csv", index=False)

    # (2) 사람이 읽는 피벗 (정정(6): dropna=False로 '보드필요' 회색 행 보존 — §5-1 원칙)
    lat = df.pivot_table(index=["model", "soc"], columns="precision",
                         values="latency_ms", dropna=False)
    acc = df.pivot_table(index=["model", "soc"], columns="precision",
                         values="accuracy", dropna=False)

    # (3) Markdown
    md = [
        "# 성능 매트릭스 (자동 생성)\n",
        f"> 총 {len(df)} 케이스 · TRT {[v for v in df['trt_version'].unique() if v]}\n",
        "## Latency (ms, median)\n", lat.round(4).to_markdown(), "\n",
        "## Accuracy (top-1)\n", acc.round(4).to_markdown(), "\n",
    ]
    (OUT_DIR / "matrix.md").write_text("\n".join(md))

    # (4) HTML 대시보드
    (OUT_DIR / "matrix.html").write_text(_highlight_html(df))
    print("wrote report/matrix.csv, matrix.md, matrix.html")


if __name__ == "__main__":
    main()
