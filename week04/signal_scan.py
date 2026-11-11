"""4주차 2차 — 분기별 관계 분석 → 일 단위 신호 찾기 (학습 2022~2024) → 고정 → 검증 2025~2026 한 번 (README 6-10)

  python signal_scan.py     → results/signal_scan.json, results/signal_scan_*.csv

왜 일 단위인가 (분기별 분석, results/quarterly_corr.csv):
  같은 분기에서도 상관이 1분 0.13 → 5분 0.29 → 30분 0.48 → 일간 0.74 로 커짐 (2022Q1). **Epps 효과** — 삼성 선물은
  2022년에 1분 중 64% 가 가격이 안 바뀔 만큼 거래가 듬성듬성해 짧은 간격의 상관이 낮게 보였음. 6-7절의 「2022~25 상관 0.2,
  2026만 페어」 는 1분 기준이라 그렇게 보인 것이고, **일간으로는 2022년부터 0.7 안팎으로 같이 움직였음.**
  → 두 종목의 관계는 일 단위에 있음. 1분 교차상관(선도·후행)은 k=1 에서 0.02~0.05 로 하이닉스가 아주 약간 먼저지만, 비용 대비 0.

데이터: 1분봉에서 하루 시가(첫 봉)·종가(15:29 봉) — results/pair_daily.csv
잔차 수익률: e = rA − β·rB, β = 직전 60거래일 일간 수익률 회귀 (그날 정보 안 씀)

───────── 후보 신호 (돌리기 전에 목록 고정) ─────────
  예측 대상 = 신호를 본 **다음** 시점부터의 잔차 수익률 (실제로 거래할 수 있는 구간)
  S1 1일 반전      : e(t, 종가→종가)              → e(t+1, 시가→종가)          음(−)이면 반전
  S2 5일 반전      : Σe(t−4..t) / σ               → e(t+1 시가 → t+5 종가)   (겹치지 않게 5일 간격 표본)
  S3 20일 모멘텀   : Σe(t−19..t) / σ              → e(t+1 시가 → t+5 종가)
  S4 선도(하이닉스) : rB(t, 종가→종가)              → e(t+1, 시가→종가)          하이닉스가 오른 다음 날 삼성이 따라오나
  S5 갭 반전       : 갭 잔차 g(t) = 시가 잔차(t−1 종가→t 시가) → e(t, 시가→종가)    시가에 알고 같은 날 장중에 거래
  S6 상관 이탈     : (20일 상관 − 120일 상관) 이 낮을 때의 |5일 반전| 강도 차이 — S2 를 상관 낮은 날/높은 날로 나눠 봄
판정 (학습 2022-01 ~ 2024-12 만):  |t| ≥ 2  그리고  2022·2023·2024 세 해 모두 같은 부호  → 「후보」
  후보만 검증 2025-01 ~ 2026-11 에서 한 번: 같은 부호이고 비용 뒤 순손익 > 0 이면 「채택」
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import pair_core as pc

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
IS_END, OOS_START = "2024-12-31", "2025-01-01"


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else float("nan")


def slope_t(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 20:
        return float("nan"), float("nan"), len(x)
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    res = y - y.mean() - b * xc
    se = math.sqrt(float((res ** 2).sum()) / (len(x) - 2) / float((xc ** 2).sum()))
    return b, b / se, len(x)


def build() -> pd.DataFrame:
    D = pd.read_csv(OUT / "pair_daily.csv", index_col=0, parse_dates=True)
    la_o, la_c, lb_o, lb_c = (np.log(D[c]) for c in ("ao", "ac", "bo", "bc"))
    rA, rB = la_c.diff(), lb_c.diff()                                  # 종가→종가
    beta = pd.Series(np.nan, index=D.index)
    for i in range(60, len(D)):
        x, y = rB.iloc[i - 60:i], rA.iloc[i - 60:i]
        m = x.notna() & y.notna()
        beta.iloc[i] = float((x[m] * y[m]).sum() / (x[m] ** 2).sum())
    F = pd.DataFrame(index=D.index)
    F["beta"] = beta
    F["rA"], F["rB"] = rA, rB
    F["e_cc"] = rA - beta * rB                                          # 오늘 잔차 (그날 β 는 어제까지로 추정)
    F["e_gap"] = (la_o - la_c.shift(1)) - beta * (lb_o - lb_c.shift(1))  # 밤사이 잔차
    F["e_oc"] = (la_c - la_o) - beta * (lb_c - lb_o)                    # 장중 잔차 (시가→종가)
    F["sig_e"] = F["e_cc"].rolling(60).std().shift(1)
    F["corr20"] = rA.rolling(20).corr(rB)
    F["corr120"] = rA.rolling(120).corr(rB)
    # 미래 (거래 구간)
    F["y1"] = F["e_oc"].shift(-1)                                       # 다음 날 시가→종가
    nxt_open = (la_o.shift(-1) - beta * lb_o.shift(-1))
    F["y5"] = (la_c.shift(-5) - beta * lb_c.shift(-5)) - nxt_open       # 다음 날 시가 → 5일 뒤 종가 (β 고정)
    F["ac"], F["bc"] = D["ac"], D["bc"]
    return F


def scan(F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    F["s1"] = F["e_cc"]
    F["s2"] = F["e_cc"].rolling(5).sum() / F["sig_e"]
    F["s3"] = F["e_cc"].rolling(20).sum() / F["sig_e"]
    F["s4"] = F["rB"]
    F["s5"] = F["e_gap"]
    F["y_s5"] = F["e_oc"]                                              # 갭은 같은 날 시가→종가
    every5 = np.arange(len(F)) % 5 == 0
    specs = [("S1 1일 반전", "s1", "y1", None), ("S2 5일 반전", "s2", "y5", every5), ("S3 20일 모멘텀", "s3", "y5", every5),
             ("S4 하이닉스 선도", "s4", "y1", None), ("S5 갭 반전", "s5", "y_s5", None)]
    rows = []
    yrs = F.index.year
    for name, s, y, sub in specs:
        G = F if sub is None else F[sub]
        gy = G.index.year
        r = {"신호": name}
        for per, m in (("학습", G.index <= IS_END), ("검증", G.index >= OOS_START)):
            b, t, n = slope_t(G.loc[m, s], G.loc[m, y])
            r[f"{per}_기울기"], r[f"{per}_t"], r[f"{per}_n"] = round(b, 4), round(t, 2), n
        for yy in (2022, 2023, 2024, 2025, 2026):
            b, t, n = slope_t(G.loc[gy == yy, s], G.loc[gy == yy, y])
            r[f"{yy}_t"] = round(t, 2)
        sgn = [np.sign(r[f"{yy}_t"]) for yy in (2022, 2023, 2024)]
        r["후보"] = bool(abs(r["학습_t"]) >= 2 and len(set(sgn)) == 1)
        rows.append(r)
    # S6: S2 를 상관 이탈(20일 − 120일) 위·아래로 나눔
    G = F[every5].copy()
    G["gap_corr"] = G["corr20"] - G["corr120"]
    for per, m in (("학습", G.index <= IS_END), ("검증", G.index >= OOS_START)):
        H = G[m]
        lo = H["gap_corr"] <= H["gap_corr"].median()
        for lab, mm in (("상관 이탈 큼", lo), ("상관 유지", ~lo)):
            b, t, n = slope_t(H.loc[mm, "s2"], H.loc[mm, "y5"])
            rows.append({"신호": f"S6 5일 반전 | {lab}", f"{per}_기울기": round(b, 4), f"{per}_t": round(t, 2), f"{per}_n": n})
    return pd.DataFrame(rows)


def main():
    F = build()
    T = scan(F)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    T.to_csv(OUT / "signal_scan.csv", index=False, encoding="utf-8-sig")
    F.to_csv(OUT / "signal_daily.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
