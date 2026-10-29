"""검증 구간(2025-01 ~) 상세 — 튜닝 1위 4개 동일비중을 중심으로.

- 변동성 국면: 연도별 실현변동성, 60일 이동 변동성 그래프
- 검증 구간 낙폭, 월별 수익, 가장 큰 하락 구간에서의 전략별 수익
- S1·S4 거래 목록, 전략별 포지션 그래프

튜닝 1위 숫자는 optimize_daily.py 결과(results/optimize_daily.json)와 같다.
결과: week02/results/oos_analysis.json, week02/results/figs/8~10_*.png
실행: python oos_analysis.py   (week02 폴더에서)
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtest_daily import RESULTS, load, run, s0_buyhold, stats  # noqa: E402
from optimize_daily import FIGS, IS, NAMES, OOS, S3, s1, s2, s4  # noqa: E402

TUNED = {"S1": dict(lookback=5, mode="lo"), "S2": dict(window=22, cap=1.0, power=0.5),
         "S3": dict(delta=20, decay=1, gate=True, flip=1), "S4": dict(n=3, entry=30, trend=100, exit_n=10)}
DEFAULT = {"S1": dict(lookback=12, mode="ls"), "S2": dict(window=22, cap=1.0, power=1.0),
           "S3": dict(delta=10, decay=4, gate=True, flip=1), "S4": dict(n=2, entry=5, trend=200, exit_n=5)}
COLORS = {"S1": "#1f77b4", "S2": "#2ca02c", "S3": "#d62728", "S4": "#9467bd"}


def positions(df, params):
    s3 = S3(df)
    return {"S1": s1(df, **params["S1"]), "S2": s2(df, **params["S2"]),
            "S3": s3(**params["S3"]), "S4": s4(df, **params["S4"])}


def trade_list(o: pd.DataFrame) -> list[dict]:
    """포지션이 0 이 아닌 구간 = 거래 1건 (시가 진입 ~ 시가 청산)."""
    out, start, acc = [], None, 1.0
    pos = np.sign(o["pos"])
    for d, p, r in zip(o.index, pos.values, o["ret"].values):
        if start is not None and p != cur:
            out.append({"진입일": str(start.date()), "청산일": str(d.date()), "방향": "롱" if cur > 0 else "숏",
                        "수익률%": round((acc - 1) * 100, 2)})
            start = None
        if p != 0 and start is None:
            start, acc, cur = d, 1.0, p
        if start is not None:
            acc *= 1 + r
        cur = p if start is not None else 0
    if start is not None:
        out.append({"진입일": str(start.date()), "청산일": "보유 중", "방향": "롱" if cur > 0 else "숏",
                    "수익률%": round((acc - 1) * 100, 2)})
    return out


def main():
    df = load()
    bh = run(df, s0_buyhold(df))["ret"]
    tuned = {k: run(df, p) for k, p in positions(df, TUNED).items()}
    dflt = {k: run(df, p) for k, p in positions(df, DEFAULT).items()}
    combo_t = pd.concat([o["ret"] for o in tuned.values()], axis=1).mean(axis=1)
    combo_d = pd.concat([o["ret"] for o in dflt.values()], axis=1).mean(axis=1)
    rep = {}

    # 1. 변동성 국면
    vol_y = (bh.loc["2001":].groupby(bh.loc["2001":].index.year).std() * np.sqrt(252) * 100).round(1)
    rank = vol_y.rank(ascending=False).astype(int)
    rep["연도별 실현변동성%"] = vol_y.to_dict()
    rep["변동성 순위(2001~2026, 큰 순)"] = {y: f"{rank[y]}/{len(rank)}" for y in (2025, 2026)}
    rep["학습 구간 평균 변동성%"] = round(float(bh.loc[IS].std() * np.sqrt(252) * 100), 1)
    rep["검증 구간 변동성%"] = round(float(bh.loc[OOS].std() * np.sqrt(252) * 100), 1)
    big = bh.loc[OOS].abs()
    rep["검증 구간 하루 ±3% 넘은 날"] = int((big > 0.03).sum())
    rep["학습 구간 하루 ±3% 넘은 날 (연평균)"] = round(float((bh.loc[IS].abs() > 0.03).sum() / 24), 1)

    # 2. 검증 구간 가장 큰 하락 (매수보유 고점→저점)
    e = (1 + bh.loc[OOS]).cumprod()
    trough = (e / e.cummax() - 1).idxmin()
    peak = e.loc[:trough].idxmax()
    win = slice(peak + pd.Timedelta(days=1), trough)
    seg = {"매수보유": bh, "튜닝 4개 동일비중": combo_t, "기본값 4개 동일비중": combo_d,
           **{f"{NAMES[k]} (튜닝)": o["ret"] for k, o in tuned.items()}}
    rep["검증 구간 최대 하락"] = {"고점": str(peak.date()), "저점": str(trough.date()),
                           "구간 수익률%": {k: round(((1 + r.loc[win]).prod() - 1) * 100, 1) for k, r in seg.items()}}

    # 3. 월별 수익 (검증 구간)
    month = {k: ((1 + r.loc[OOS]).groupby(r.loc[OOS].index.to_period("M")).prod() - 1) * 100 for k, r in seg.items()}
    rep["검증 월별%"] = {k: {str(p): round(v, 1) for p, v in m.items()} for k, m in month.items()}

    # 4. 전략별 검증 지표 (튜닝)
    rep["튜닝 전략 검증 지표"] = {NAMES[k]: stats(o.loc[OOS, "ret"], o.loc[OOS, "pos"], o.loc[OOS, "cost"]) for k, o in tuned.items()}
    rep["튜닝 전략 학습 지표"] = {NAMES[k]: stats(o.loc[IS, "ret"], o.loc[IS, "pos"], o.loc[IS, "cost"]) for k, o in tuned.items()}
    one = pd.Series(1.0, index=combo_t.index)
    rep["튜닝 4개 동일비중"] = {"IS": stats(combo_t.loc[IS], one.loc[IS]), "OOS": stats(combo_t.loc[OOS], one.loc[OOS])}
    rep["검증 상관"] = pd.DataFrame({NAMES[k]: o.loc[OOS, "ret"] for k, o in tuned.items()}).corr().round(2).to_dict()
    rep["검증 거래 S1"] = trade_list(tuned["S1"].loc[OOS])
    rep["검증 거래 S4"] = trade_list(tuned["S4"].loc[OOS])

    (RESULTS / "oos_analysis.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # 그래프 8: 60일 변동성
    fig, ax = plt.subplots(figsize=(10, 3.5))
    rv = bh.rolling(60).std() * np.sqrt(252) * 100
    ax.plot(rv.loc["2001":], color="#555", lw=1)
    ax.axvspan(pd.Timestamp("2025-01-02"), df.index[-1], color="#ff7f0e", alpha=.15, label="검증 구간")
    ax.axhline(rep["학습 구간 평균 변동성%"], color="k", ls="--", lw=.8, label=f"2001–2024 평균 {rep['학습 구간 평균 변동성%']}%")
    ax.set_title("⑧ 코스피200선물 60일 실현변동성 (연환산 %)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "8_rolling_vol.png"); plt.close(fig)

    # 그래프 9: 검증 구간 누적 + 낙폭 (튜닝 동일비중 중심)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for lab, r, c, w in [("매수보유", bh, "#888", 1.2), ("기본값 4개 동일비중", combo_d, "#bcbd22", 1.2),
                         ("튜닝 4개 동일비중", combo_t, "#ff7f0e", 2.2)]:
        q = (1 + r.loc[OOS]).cumprod()
        a1.plot(q, color=c, lw=w, label=lab)
        a2.plot((q / q.cummax() - 1) * 100, color=c, lw=w)
    for a in (a1, a2):
        a.axvspan(peak, trough, color="red", alpha=.08)
    a1.set_title("⑨ 검증 구간 누적수익(배)과 낙폭(%) — 빨간 띠 = 매수보유 최대 하락 구간"); a1.legend(fontsize=8); a1.grid(alpha=.3); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "9_oos_combo_drawdown.png"); plt.close(fig)

    # 그래프 10: 튜닝 전략 포지션 (검증 구간)
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    px = df.loc[OOS, "close"]
    axes[0].plot(px, color="#333", lw=1); axes[0].set_title("⑩ 검증 구간 코스피200선물 종가와 튜닝 전략 포지션", fontsize=10)
    for ax, (k, o) in zip(axes[1:], tuned.items()):
        p = o.loc[OOS, "pos"]
        ax.fill_between(p.index, p.values, 0, step="post", color=COLORS[k], alpha=.5)
        ax.set_ylim(-1.1, 1.1); ax.set_ylabel(k, rotation=0, labelpad=15); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "10_oos_positions.png"); plt.close(fig)

    print(json.dumps({k: rep[k] for k in ("변동성 순위(2001~2026, 큰 순)", "학습 구간 평균 변동성%", "검증 구간 변동성%",
                                          "검증 구간 하루 ±3% 넘은 날", "학습 구간 하루 ±3% 넘은 날 (연평균)", "검증 구간 최대 하락")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
