"""6주차 그림 — backtest6.py 가 부름.

  1_equity.png        A·B·C·D 누적 순손익 (같은 기간)
  2_tail.png          최대 낙폭 · 최악의 하루 · 일간 ES95 (설정별 막대)
  3_day_<날짜>.png    D(6주차 전체)에서 분할 매수·분할 청산·손절이 한 날에 같이 작동한 예
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for k in ("axes.edgecolor", "xtick.color", "ytick.color"):
        plt.rcParams[k] = MUTED
    plt.rcParams["axes.labelcolor"] = INK
    return plt


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def make(df, F, plans, allT: dict, days: list, out: Path):
    plt = _plt()
    out.mkdir(parents=True, exist_ok=True)
    real = lambda T: T[~T["shadow"].astype(bool)] if (len(T) and "shadow" in T) else T     # 가상 매매는 뺌
    keep = {k: real(v) for k, v in allT.items() if k.split(" ")[0] in COL}

    # 1. 누적 순손익
    fig, ax = plt.subplots(figsize=(9, 4.2))
    start = "20250901" if days[0] < "20250901" < days[-1] else days[0]
    n_before = {n: int((T.groupby("ep")["date"].first() < start).sum()) for n, T in keep.items()}
    ends = []
    for name, T in keep.items():
        k = name.split(" ")[0]
        daily = T.groupby("date")["net"].sum().reindex(days, fill_value=0.0).cumsum() / 1e4
        daily = daily[daily.index >= start]
        x = pd.to_datetime(daily.index)
        ax.plot(x, daily.values, color=COL[k], lw=2, label=name)
        ends.append((daily.values[-1], name, x[-1]))
    ends.sort()
    gap = (max(e[0] for e in ends) - min(e[0] for e in ends) + 1) * 0.1
    ys = []
    for v, name, xe in ends:
        y = v if not ys or v - ys[-1] >= gap else ys[-1] + gap
        ys.append(y)
        ax.annotate(f"{name}  {v:,.0f}만", (xe, v), xytext=(xe + pd.Timedelta(days=6), y), va="center", fontsize=8.5, color=INK)
    _clean(ax)
    ax.set_ylabel("누적 순손익 (만 원)")
    ax.set_title("위험관리 전·후 — 같은 신호·같은 비용·최대 2단위", loc="left", fontsize=11, color=INK)
    if start != days[0]:
        ax.text(0.0, -0.13, f"{days[0][:4]}-{days[0][4:6]} ~ 2025-08 은 그림에서 뺌 (거래 {min(n_before.values())}~{max(n_before.values())}건, "
                "1분 상관이 0.5 밑이라 대부분 진입 안 함) — 누적은 그 기간 손익을 포함", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    span = pd.to_datetime(days[-1]) - pd.to_datetime(start)
    ax.set_xlim(right=pd.to_datetime(days[-1]) + span * 0.28)
    fig.tight_layout(); fig.savefig(out / "1_equity.png", dpi=140); plt.close(fig)

    # 2. 꼬리
    from backtest6 import stats
    S = {n: stats(T, days) for n, T in keep.items()}
    met = [("최대낙폭원", "최대 낙폭"), ("최악의하루원", "최악의 하루"), ("일간ES95원", "하위 5% 날 평균 (ES95)")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (m, lab) in zip(axs, met):
        names = list(S)
        vals = [-S[n][m] / 1e4 for n in names]
        ax.bar(range(len(names)), vals, color=[COL[n.split(" ")[0]] for n in names], width=0.62, edgecolor="white", lw=2)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(names)), [n.split(" ")[0] for n in names])
        ax.set_title(lab + " (만 원 손실)", fontsize=9.5, color=INK, loc="left")
        _clean(ax)
    fig.text(0.01, 0.01, "A 없음 · B 4주차 안전장치 · C B+분할 · D 6주차 전체 · E 개선 전체(11/22)", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out / "2_tail.png", dpi=140); plt.close(fig)

    # 3. 작동 예시 하루 — D 에서 2차 진입·분할 청산·손절(또는 보호)이 같은 날
    TD = next((T for n, T in keep.items() if n.startswith("D")), None)
    if TD is None or TD.empty:
        return
    g = TD.groupby("date")
    cand = [d for d, x in g if (x["tag"] == "2차").any() and (x["exit"] == "분할청산1").any()
            and x["exit"].isin(["가격손절", "금액손절", "시간손절", "이익보호"]).any()]
    cand = cand or [d for d, x in g if (x["tag"] == "2차").any() and (x["exit"] == "분할청산1").any()]
    for d in cand[-2:]:
        day_plot(plt, F, plans, TD[TD["date"] == d], d, out)


def day_plot(plt, F, plans, T, d, out):
    f = F[F.index.strftime("%Y%m%d") == d]
    pl = plans.get(pd.Timestamp(d))
    a, m = pl.plan.entry, pl.plan.exit
    T = T.copy(); T["t_in"] = pd.to_datetime(T["t_in"]); T["t_out"] = pd.to_datetime(T["t_out"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(f.index, f["z"], color=INK, lw=1.2)
    side = -1 if int(T["dir"].iloc[0]) > 0 else 1        # 롱 스프레드면 z 음수 쪽 선
    for y, lab, ls in ((a, "진입 a", "--"), (a + 0.5, "2차 a+0.5", ":"), (a + 1.5, "손절 a+1.5", "-"),
                       ((a + m) / 2, "분할청산 (a+m)/2", ":"), (m, "수렴 m", "--")):
        ax.axhline(side * y, color=MUTED, lw=0.8, ls=ls)
        ax.text(0.995, side * y, lab, transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, va="bottom", ha="right")
    zat = lambda t: float(f["z"].reindex([t], method="nearest").iloc[0]) if len(f) else np.nan
    for j, (_, r) in enumerate(T.sort_values("t_in").iterrows()):
        ax.scatter([r.t_in], [zat(r.t_in)], marker="^" if r.tag != "2차" else "D", s=60, color=COL["A"], zorder=3,
                   edgecolor="white", lw=1.5)
        ax.annotate(f"{r.tag} 진입 {r.t_in:%H:%M}", (r.t_in, zat(r.t_in)), xytext=(-80, -18 - 16 * j), textcoords="offset points",
                    fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.scatter([r.t_out], [zat(r.t_out)], marker="v", s=60, color=COL["B"], zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(f"{r.tag} {r.exit} {r.t_out:%H:%M} {r.net / 1e4:+.1f}만", (r.t_out, zat(r.t_out)), xytext=(14, 24 + 18 * j),
                    textcoords="offset points", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    t0, t1 = T["t_in"].min() - pd.Timedelta(minutes=60), T["t_out"].max() + pd.Timedelta(minutes=60)
    ax.set_xlim(max(t0, f.index[0]), min(t1, f.index[-1]))
    _clean(ax)
    ax.set_ylabel("z (σ_eq 단위)")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(f"{d} — 6주차 규칙이 작동한 하루 (진입 a={a}, 수렴 m={m}, 반감기 {pl.fit.half_life:.0f}분)",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout(); fig.savefig(out / f"3_day_{d}.png", dpi=140); plt.close(fig)
