"""코스피200선물 일봉 전략 4개 — 2001~2024 로 숫자를 고르고 2025-01 이후(고를 때 안 본 구간)에서 확인한다.

- 학습(IS): 2001-01-02 ~ 2024-12-30. 전략별 격자의 모든 조합을 돌려 IS 샤프 상위 3개를 고른다
  (S4 는 거래 30건 미만 조합 제외 — 표본이 너무 작으면 샤프가 우연히 커짐. S1~S3 는 늘 포지션이 있거나 수백 번 바뀜)
- 검증(OOS): 2025-01-02 ~ 마지막 날. 고른 조합과 문헌 기본값, 매수보유를 같은 비용으로 비교
- 체결·비용·수정가는 backtest_daily.py 와 같다 (다음날 시가, 1틱, 0.00145%, 롤오버 2회)

결과: week02/results/optimize_daily.json, week02/results/figs/*.png
실행: python optimize_daily.py   (week02 폴더에서)
"""
import itertools
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtest_daily import (RESULTS, by_year, decay_linear, load, month_end_flags, rsi, run,  # noqa: E402
                            s0_buyhold, stats, trades, ts_rank)

FIGS = Path(__file__).resolve().parent / "results" / "figs"
IS = slice("2001-01-02", "2024-12-30")
OOS = slice("2025-01-02", None)
MIN_TRADES = 30
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "figure.dpi": 110})


# ── 숫자를 바꿀 수 있게 일반화한 전략 (기본값 = 문헌/IQC 값) ──

def s1(df, lookback=12, mode="ls"):
    me = month_end_flags(df.index)
    m = (1 + df["ret"].fillna(0)).cumprod()[me]
    sig = np.sign(m / m.shift(lookback) - 1)
    if mode == "lo":  # 롱/현금
        sig = sig.clip(lower=0)
    return sig.reindex(df.index).ffill()


def s2(df, window=22, cap=1.0, power=1.0):
    """비중 = c / (직전 window 일 실현분산)^power. power 1 = 분산(논문), 0.5 = 표준편차."""
    r = df["ret"].fillna(0)
    ends = df.index[month_end_flags(df.index)]
    rv = (r.rolling(window).var() * window)[ends]
    mret = (1 + r).cumprod()[ends].pct_change()
    raw = mret / rv.shift(1) ** power
    c = mret.expanding().std() / raw.expanding().std()   # 그 시점까지 데이터로만 맞춤
    w = (c / rv ** power).clip(upper=cap)
    return w.reindex(df.index).ffill()


class S3:
    """IQC #5. 거래조건(20·60·0.95·−0.01·252·0.8)은 고정하고 신호 쪽 숫자만 바꾼다."""

    def __init__(self, df):
        self.df = df
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        self.x = ((df["high"] - df["close"]) - (df["close"] - df["low"])) / rng
        ret = df["ret"]
        sd20, sd60 = ret.rolling(20).std(), ret.rolling(60).std()
        self.on = (sd20 > sd60.rolling(60).mean() * 0.95) & (
            (ret.rolling(60).mean() > -0.01) | (ts_rank(sd20, 252) > 0.8))

    def __call__(self, delta=10, decay=4, gate=True, flip=1):
        sig = flip * np.sign(decay_linear(self.x.diff(delta), decay))
        return sig.where(self.on).ffill().fillna(0) if gate else sig.fillna(0)


def s4(df, n=2, entry=5, trend=200, exit_n=5):
    c = df["aclose"].values
    up = pd.Series(c).rolling(trend).mean().values if trend else np.full(len(c), -np.inf)
    ex = pd.Series(c).rolling(exit_n).mean().values
    r = rsi(df["aclose"], n).values
    pos, out = 0.0, np.zeros(len(c))
    for i in range(len(c)):
        if pos == 0 and c[i] > up[i] and r[i] < entry:
            pos = 1.0
        elif pos == 1 and c[i] > ex[i]:
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=df.index)


GRIDS = {
    "S1": (s1, {"lookback": list(range(1, 13)), "mode": ["ls", "lo"]}, {"lookback": 12, "mode": "ls"}),
    "S2": (s2, {"window": [22, 44, 66, 126], "cap": [1.0, 1.5, 2.0], "power": [1.0, 0.5]},
           {"window": 22, "cap": 1.0, "power": 1.0}),
    "S3": (None, {"delta": [1, 3, 5, 10, 20], "decay": [1, 2, 4, 8], "gate": [True, False], "flip": [1, -1]},
           {"delta": 10, "decay": 4, "gate": True, "flip": 1}),
    "S4": (s4, {"n": [2, 3, 4], "entry": [2, 5, 10, 15, 20, 25, 30], "trend": [0, 100, 150, 200], "exit_n": [3, 5, 10]},
           {"n": 2, "entry": 5, "trend": 200, "exit_n": 5}),
}
NAMES = {"S1": "S1 시계열모멘텀", "S2": "S2 변동성관리", "S3": "S3 IQC 반전", "S4": "S4 RSI 눌림목"}


def label(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def evaluate(df, pos):
    o = run(df, pos)
    a, b = o.loc[IS], o.loc[OOS]
    return o, stats(a["ret"], a["pos"], a["cost"]), stats(b["ret"], b["pos"], b["cost"])


def main():
    df = load()
    FIGS.mkdir(parents=True, exist_ok=True)
    s3 = S3(df)
    bh = run(df, s0_buyhold(df))
    report = {"학습": "2001-01-02 ~ 2024-12-30", "검증": f"2025-01-02 ~ {df.index[-1].date()}",
              "매수보유": {"IS": stats(bh.loc[IS, "ret"], bh.loc[IS, "pos"]), "OOS": stats(bh.loc[OOS, "ret"], bh.loc[OOS, "pos"])},
              "전략": {}}
    grid_rows, chosen = [], {}

    for key, (fn, grid, default) in GRIDS.items():
        f = s3 if key == "S3" else fn
        rows = []
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid.keys(), combo))
            pos = f(df, **p) if key != "S3" else f(**p)
            o, si, so = evaluate(df, pos)
            rows.append({"key": key, "params": p, "IS": si, "OOS": so, "_pos": pos, "_o": o})
        ok = [r for r in rows if key != "S4" or (r["IS"]["거래수"] or 0) >= MIN_TRADES]
        top = sorted(ok, key=lambda r: r["IS"]["샤프"], reverse=True)[:3]
        dflt = next(r for r in rows if r["params"] == default)
        is_sh = np.array([r["IS"]["샤프"] for r in ok])
        oos_sh = np.array([r["OOS"]["샤프"] for r in ok])
        report["전략"][key] = {
            "조합 수": len(rows), "조건 통과": len(ok),
            "IS 샤프 분포(중앙값/상위10%)": [round(float(np.median(is_sh)), 2), round(float(np.quantile(is_sh, 0.9)), 2)],
            "IS-OOS 샤프 순위상관": round(float(pd.Series(is_sh).rank().corr(pd.Series(oos_sh).rank())), 2),
            "기본값": {"params": label(default), "IS": dflt["IS"], "OOS": dflt["OOS"]},
            "상위3": [{"params": label(r["params"]), "IS": r["IS"], "OOS": r["OOS"]} for r in top],
        }
        chosen[key] = {"default": dflt, "best": top[0], "top": top}
        grid_rows += rows
        print(key, "기본", dflt["IS"]["샤프"], "→ 최고", top[0]["IS"]["샤프"], label(top[0]["params"]),
              "| OOS", dflt["OOS"]["샤프"], "→", top[0]["OOS"]["샤프"])

    # 조합: 전략별 1위끼리 동일비중 / 기본값끼리 동일비중
    for name, pick in (("기본값 4개 동일비중", "default"), ("튜닝 1위 4개 동일비중", "best")):
        r = pd.concat([chosen[k][pick]["_o"]["ret"] for k in GRIDS], axis=1).mean(axis=1)
        one = pd.Series(1.0, index=r.index)
        report[name] = {"IS": stats(r.loc[IS], one.loc[IS]), "OOS": stats(r.loc[OOS], one.loc[OOS]),
                        "IS 연도별%": by_year(r.loc[IS]), "OOS 연도별%": by_year(r.loc[OOS])}
        chosen[name] = r

    out = RESULTS / "optimize_daily.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    figures(df, bh, chosen, grid_rows)
    print(f"→ {out}, {FIGS}")


# ── 그래프 ──

def eq(r):
    return (1 + r).cumprod()


def figures(df, bh, chosen, rows):
    keys = list(GRIDS)
    colors = {"S0": "#888888", "S1": "#1f77b4", "S2": "#2ca02c", "S3": "#d62728", "S4": "#9467bd", "C": "#ff7f0e"}

    # 1. 기본값 누적수익 (IS, 로그)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq(bh.loc[IS, "ret"]), color=colors["S0"], label="매수보유", lw=1.2)
    for k in keys:
        ax.plot(eq(chosen[k]["default"]["_o"].loc[IS, "ret"]), color=colors[k], label=NAMES[k], lw=1.2)
    ax.plot(eq(chosen["기본값 4개 동일비중"].loc[IS]), color=colors["C"], label="4개 동일비중", lw=2)
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}배")
    ax.yaxis.set_major_formatter(fmt); ax.yaxis.set_minor_formatter(fmt)
    ax.set_title("① 문헌 기본값 — 누적수익 (2001–2024, 로그)"); ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "1_default_equity.png"); plt.close(fig)

    # 2. 낙폭
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for lab, r, c in [("매수보유", bh.loc[IS, "ret"], colors["S0"]),
                      ("4개 동일비중", chosen["기본값 4개 동일비중"].loc[IS], colors["C"]),
                      (NAMES["S2"], chosen["S2"]["default"]["_o"].loc[IS, "ret"], colors["S2"])]:
        e = eq(r); ax.fill_between(e.index, (e / e.cummax() - 1) * 100, 0, alpha=.35, color=c, label=lab)
    ax.set_title("② 낙폭 (%) — 기본값"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "2_drawdown.png"); plt.close(fig)

    # 3. 연도별 수익 히트맵
    yr = pd.DataFrame({"매수보유": by_year(bh.loc[IS, "ret"]),
                       **{NAMES[k]: by_year(chosen[k]["default"]["_o"].loc[IS, "ret"]) for k in keys},
                       "동일비중": by_year(chosen["기본값 4개 동일비중"].loc[IS])})
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(yr.values, cmap="RdYlGn", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(range(yr.shape[1]), yr.columns, rotation=30, fontsize=8); ax.set_yticks(range(len(yr)), yr.index, fontsize=8)
    for i, j in itertools.product(range(yr.shape[0]), range(yr.shape[1])):
        ax.text(j, i, f"{yr.iat[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("③ 연도별 수익률 (%) — 기본값"); fig.colorbar(im, shrink=.6)
    fig.tight_layout(); fig.savefig(FIGS / "3_yearly_heatmap.png"); plt.close(fig)

    # 4. 상관
    cr = pd.DataFrame({NAMES[k]: chosen[k]["default"]["_o"].loc[IS, "ret"] for k in keys}).corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(4), cr.columns, rotation=30, fontsize=8); ax.set_yticks(range(4), cr.columns, fontsize=8)
    for i, j in itertools.product(range(4), range(4)):
        ax.text(j, i, f"{cr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("④ 일수익률 상관 — 기본값"); fig.tight_layout(); fig.savefig(FIGS / "4_correlation.png"); plt.close(fig)

    # 5. 파라미터 지형 (IS 샤프)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    by = {k: [r for r in rows if r["key"] == k] for k in keys}
    ax = axes[0, 0]
    for mode, c in (("ls", colors["S1"]), ("lo", "#17becf")):
        rs = [r for r in by["S1"] if r["params"]["mode"] == mode]
        ax.plot([r["params"]["lookback"] for r in rs], [r["IS"]["샤프"] for r in rs], "o-", color=c,
                label="롱/숏" if mode == "ls" else "롱/현금")
    ax.axhline(0.41, color="#888", ls="--", lw=1, label="매수보유"); ax.set_xlabel("룩백(개월)"); ax.set_title("S1 룩백별 IS 샤프"); ax.legend(fontsize=8)
    heat(axes[0, 1], by["S2"], "window", "cap", {"power": 1.0}, "S2 (분산) 창×상한 IS 샤프")
    heat(axes[1, 0], by["S3"], "delta", "decay", {"gate": True, "flip": 1}, "S3 (원래 방향·조건 켬) delta×decay IS 샤프")
    heat(axes[1, 1], by["S4"], "entry", "trend", {"n": 2, "exit_n": 5}, "S4 (RSI2·SMA5 청산) 진입×추세 IS 샤프")
    fig.suptitle("⑤ 파라미터 지형 — 2001–2024 샤프"); fig.tight_layout(); fig.savefig(FIGS / "5_param_landscape.png"); plt.close(fig)

    # 6. IS vs OOS 샤프 (모든 조합)
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in keys:
        rs = [r for r in by[k] if k != "S4" or (r["IS"]["거래수"] or 0) >= MIN_TRADES]
        ax.scatter([r["IS"]["샤프"] for r in rs], [r["OOS"]["샤프"] for r in rs], s=10, alpha=.5, color=colors[k], label=NAMES[k])
        b = chosen[k]["best"]; ax.scatter(b["IS"]["샤프"], b["OOS"]["샤프"], s=120, marker="*", color=colors[k], edgecolor="k")
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
    ax.set_xlabel("학습 샤프 (2001–2024)"); ax.set_ylabel("검증 샤프 (2025–)"); ax.set_title("⑥ 학습 vs 검증 샤프 (★ = 학습 1위)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "6_is_vs_oos.png"); plt.close(fig)

    # 7. 검증 구간 누적수익
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, pick, title in ((axes[0], "default", "기본값"), (axes[1], "best", "튜닝 1위")):
        ax.plot(eq(bh.loc[OOS, "ret"]), color=colors["S0"], label="매수보유")
        for k in keys:
            ax.plot(eq(chosen[k][pick]["_o"].loc[OOS, "ret"]), color=colors[k], label=NAMES[k])
        comb = "기본값 4개 동일비중" if pick == "default" else "튜닝 1위 4개 동일비중"
        ax.plot(eq(chosen[comb].loc[OOS]), color=colors["C"], lw=2, label="4개 동일비중")
        ax.set_title(f"⑦ 검증 구간 누적 — {title}"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "7_oos_equity.png"); plt.close(fig)


def heat(ax, rows, xk, yk, fixed, title):
    rs = [r for r in rows if all(r["params"][k] == v for k, v in fixed.items())]
    xs = sorted({r["params"][xk] for r in rs}); ys = sorted({r["params"][yk] for r in rs})
    z = np.full((len(ys), len(xs)), np.nan)
    for r in rs:
        z[ys.index(r["params"][yk]), xs.index(r["params"][xk])] = r["IS"]["샤프"]
    im = ax.imshow(z, cmap="RdYlGn", vmin=-0.6, vmax=0.6, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xs)), xs); ax.set_yticks(range(len(ys)), ys); ax.set_xlabel(xk); ax.set_ylabel(yk); ax.set_title(title, fontsize=9)
    for i, j in itertools.product(range(len(ys)), range(len(xs))):
        if not np.isnan(z[i, j]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=.8)


if __name__ == "__main__":
    main()
