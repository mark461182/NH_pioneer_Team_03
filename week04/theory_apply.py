"""오늘 공부한 이론 셋을 4주차 데이터에 붙이기 (README 6-14)

  python theory_apply.py   → results/theory_var.csv, theory_hawkes.json, theory_kelly.json, theory_intensity.csv

A. 포트폴리오 분산 — 스프레드 분산 분해
   σ_s² = σ_A² + β²σ_B² − 2βρσ_Aσ_B   (s = rA − β·rB, 일간)
   분기마다 식으로 계산한 σ_s 와 실제 잔차 표준편차를 맞춰 보고, ρ 가 떨어지면 σ_s 가 얼마나 커지는지
   → 「일간 상관 < 0.3 이면 페어 멈춤」 필터의 근거

B. 호크스 — 1분 거래량이 「몰렸다 비었다」 하나 (1분봉으로 하는 이산 근사)
   N_t = 그 분 거래량 ÷ 그 시각 평균(장 초반·막판이 원래 많은 U자 모양을 뺌) ÷ 그날 평균(바쁜 날 효과를 뺌)
   N_t = μ + Σ_{k=1..30} φ_k N_{t−k}  (날 안에서만, 최소제곱)  ← 이산 호크스(INAR(∞)) 근사
   분기비(branching ratio) n = Σφ_k : 사건 하나가 평균 몇 개를 더 부르나. 0 이면 포아송, 1 에 가까우면 거의 다 연쇄
   φ_k 를 지수 α·e^{−(k−1)/τ} 로 맞춰 「식는 시간」 τ(분)
   그리고 직전 30분 거래량으로 만든 강도 λ_t 가 높은 분에 들어간 거래가 더 나쁜지 (호가 튐·역선택이 몰릴 때인지)

C. 켈리 — 얼마나 걸어야 하나
   1단위(하이닉스 1 + 삼성 β 맞춤) 거래 손익 X 를 「1단위 증거금」 으로 나눈 수익률 r 로 보고
   연속 근사 켈리 f* = E[r]/Var[r]  (자본 대비 1단위 증거금 비중), 하프켈리 = f*/2
   비용 뒤·비용 전, 기존 OU·보정 OU 각각. 추정 오차: E[r] 의 95% 하한으로 한 「보수 켈리」 도 같이
   증거금률은 모름 → 주식선물 증거금률을 15% 로 가정 (바뀌면 f* 가 비례해서 바뀜)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MULT = 10
MARGIN_RATE = 0.15


# ── A ────────────────────────────────────────────────────────
def part_a() -> pd.DataFrame:
    F = pd.read_csv(OUT / "signal_daily.csv", index_col=0, parse_dates=True).dropna(subset=["rA", "rB", "beta"])
    rows = []
    for q, X in F.groupby(F.index.to_period("Q")):
        if len(X) < 20:
            continue
        sa, sb = X.rA.std(), X.rB.std()
        rho = X.rA.corr(X.rB)
        b = X.beta.mean()
        pred = math.sqrt(max(sa ** 2 + b ** 2 * sb ** 2 - 2 * b * rho * sa * sb, 0))
        real = (X.rA - X.beta * X.rB).std()
        # 같은 σ·β 에서 ρ 만 0.9 였다면
        at90 = math.sqrt(max(sa ** 2 + b ** 2 * sb ** 2 - 2 * b * 0.9 * sa * sb, 0))
        rows.append({"분기": str(q), "일수": len(X), "σ삼성%": round(sa * 100, 2), "σ하닉%": round(sb * 100, 2), "β": round(b, 2),
                     "ρ": round(rho, 2), "σ스프레드_식%": round(pred * 100, 2), "σ스프레드_실제%": round(real * 100, 2),
                     "ρ=0.9일때%": round(at90 * 100, 2), "배수(ρ실제/0.9)": round(pred / at90, 2) if at90 > 0 else None})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "theory_var.csv", index=False, encoding="utf-8-sig")
    return A


# ── B ────────────────────────────────────────────────────────
def hawkes_fit(v: pd.Series, dates: pd.Series, K: int = 30) -> dict:
    X, Y = [], []
    for _, idx in v.groupby(dates).indices.items():
        x = v.values[idx]
        if len(x) <= K + 5:
            continue
        L = np.column_stack([x[K - k:len(x) - k] for k in range(1, K + 1)])
        X.append(L); Y.append(x[K:])
    X, Y = np.vstack(X), np.concatenate(Y)
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    mu, phi = coef[0], coef[1:]
    n = float(phi.sum())
    k = np.arange(1, K + 1)
    pos = phi > 0
    tau = float("nan")
    if pos.sum() >= 3:
        sl, ic = np.polyfit(k[pos] - 1, np.log(phi[pos]), 1)
        tau = -1 / sl if sl < 0 else float("nan")
    disp = float(Y.var() / Y.mean())
    return {"분기비_n": round(n, 3), "식는시간_τ분": round(tau, 1), "φ1": round(float(phi[0]), 3), "φ5": round(float(phi[4]), 3),
            "φ15": round(float(phi[14]), 3), "분산/평균(포아송=1)": round(disp, 2), "표본분": int(len(Y))}


def part_b(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    out = {}
    lam = {}
    for leg, col in (("삼성", "a_volume"), ("하이닉스", "b_volume")):
        v = d[col].astype(float)
        season = v.groupby(d["time"]).transform("mean")
        z = v / season.replace(0, np.nan)
        z = z.fillna(0.0)
        z = z / z.groupby(d["date"]).transform("mean").replace(0, np.nan)   # 그날 전체가 바쁜 날(날 효과)도 빼야 진짜 「몰림」
        z = z.fillna(0.0)
        out[leg] = {}
        for y, idx in z.groupby(d["date"].str[:4]).indices.items():
            out[leg][y] = hawkes_fit(z.iloc[idx], d["date"].iloc[idx])
        # 강도: 직전 30분 (현재 제외) 거래량의 지수가중합 → 그날 안에서만
        lam[leg] = z.groupby(d["date"]).transform(lambda s: s.shift(1).ewm(halflife=5, min_periods=5).mean())
    L = pd.DataFrame({"lam_a": lam["삼성"].values, "lam_b": lam["하이닉스"].values}, index=d.index)
    return out, L


# ── C ────────────────────────────────────────────────────────
def kelly(X: pd.DataFrame, col: str) -> dict:
    notional = X.qa * X.pa_in * MULT + X.qb * X.pb_in * MULT
    r = X[col] / (notional * MARGIN_RATE)
    m, s = r.mean(), r.std()
    lo = m - 1.96 * s / math.sqrt(len(r))
    f = m / s ** 2 if s > 0 else float("nan")
    fl = max(lo, 0) / s ** 2 if s > 0 else float("nan")
    return {"거래": int(len(r)), "거래당수익률%(증거금대비)": round(m * 100, 3), "표준편차%": round(s * 100, 2),
            "켈리_f*": round(max(f, 0), 3), "켈리_원값": round(f, 3), "하프켈리": round(max(f, 0) / 2, 3),
            "보수켈리(95%하한)": round(fl, 3)}


def part_c() -> dict:
    res = {}
    for k in ("orig", "noise"):
        p = OUT / f"ou_noise_trades_{k}.csv"
        if not p.exists():
            continue
        T = pd.read_csv(p)
        for (name, per), X in T.groupby(["설정", "구간"]):
            if len(X) < 10:
                continue
            res.setdefault("보정 OU" if k == "noise" else "기존 OU", {}).setdefault(name, {})[per] = {
                "비용뒤": kelly(X, "net"), "비용전": kelly(X, "gross")}
    return res


def part_b_trades(L: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame | None:
    """진입한 분의 강도(하이닉스·삼성 평균)로 나눠, 강도 높을 때 들어간 거래가 더 나쁜지."""
    p = OUT / "ou_noise_trades_noise.csv"
    if not p.exists():
        return None
    T = pd.read_csv(p, parse_dates=["t_in"])
    lam = (L["lam_a"] + L["lam_b"]) / 2
    T["강도"] = T["t_in"].map(lam)
    T = T.dropna(subset=["강도"])
    # 같은 달 안에서 강도 순위로 3등분 (가격 수준·기간이 섞이지 않게)
    T["월"] = T["t_in"].dt.to_period("M")
    T["강도구간"] = pd.cut(T.groupby("월")["강도"].rank(pct=True), [0, 1 / 3, 2 / 3, 1], labels=["낮음", "중간", "높음"], include_lowest=True)
    g = T.groupby(["강도구간"], observed=True).agg(거래=("gross", "size"), 비용전평균=("gross", "mean"), 비용평균=("cost", "mean"),
                                                 순평균=("net", "mean"), 승률=("net", lambda x: (x > 0).mean()))
    g[["비용전평균", "비용평균", "순평균"]] = g[["비용전평균", "비용평균", "순평균"]].round(0)
    g["승률"] = g["승률"].round(3)
    g["비용전/비용"] = (g["비용전평균"] / g["비용평균"]).round(3)
    g.to_csv(OUT / "theory_intensity.csv", encoding="utf-8-sig")
    return g


def main():
    A = part_a()
    print("A. 분산 분해\n", A.to_string(index=False))
    d = pd.read_csv(OUT / "pair_1m_full.csv", dtype={"date": str, "time": str})
    d = d[d["time"] < "153000"].reset_index(drop=True)
    d.index = pd.to_datetime(d["date"] + d["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    H, L = part_b(d)
    (OUT / "theory_hawkes.json").write_text(json.dumps(H, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nB. 호크스\n", pd.DataFrame({(l, y): v for l, Y in H.items() for y, v in Y.items()}).to_string())
    g = part_b_trades(L, d)
    if g is not None:
        print("\nB-2. 진입 강도별 (보정 OU, 전 설정)\n", g.to_string())
    C = part_c()
    (OUT / "theory_kelly.json").write_text(json.dumps(C, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nC. 켈리\n", json.dumps(C, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
