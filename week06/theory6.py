"""6주차 — 위험관리 값의 근거 숫자 (README 3절 표의 「이유」 칸)

  python theory6.py   → results/theory6.json

1) OU 로 본 손절선·2차 진입선 — 표준화 OU  dy = −κy dt + √(2κ) dW  (y = 스프레드/σ_eq)
   척도함수 S(y) = ∫₀^y e^{u²/2} du 로  P(a 에서 출발해 m 보다 b 에 먼저 닿음) = (S(a) − S(m)) / (S(b) − S(m))
2) 시간 손절 — 같은 OU 를 몬테카를로로: a 에서 m 까지 첫 도달 시간 분포 (반감기 단위)
3) 실제 데이터 — A(없음) 와 A1(가격손절)의 같은 진입을 짝지어 「손절이 없었다면」 을 봄 · 보유시간별 손익
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def S(y: float, n: int = 4001) -> float:
    u = np.linspace(0, y, n)
    f = np.exp(u * u / 2)
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(u)))


def p_hit(a: float, m: float, b: float) -> float:
    """a 에서 출발, b(>a) 에 m(<a) 보다 먼저 닿을 확률."""
    return (S(a) - S(m)) / (S(b) - S(m))


def fpt_hl(a: float, m: float, n: int = 20000, dt_hl: float = 0.01, cap_hl: float = 20, seed: int = 0) -> np.ndarray:
    """a → m 첫 도달 시간 (반감기 단위). κ = ln2 / 반감기, 반감기 = 1 로 둠."""
    rng = np.random.default_rng(seed)
    k = math.log(2)
    y = np.full(n, a)
    t = np.full(n, np.nan)
    alive = np.ones(n, bool)
    steps = int(cap_hl / dt_hl)
    sd = math.sqrt(2 * k * dt_hl)
    for i in range(steps):
        y[alive] += -k * y[alive] * dt_hl + sd * rng.standard_normal(alive.sum())
        hit = alive & (y <= m)
        t[hit] = (i + 1) * dt_hl
        alive &= ~hit
        if not alive.any():
            break
    t[alive] = cap_hl
    return t


def main():
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["설정"]
    a, m = summ["진입선a_중앙"], summ["청산선m_중앙"]
    out = {"기준": {"진입a_중앙": a, "청산m_중앙": m, "반감기_중앙분": summ["반감기_중앙분"]}}
    out["가격손절_도달확률"] = {f"a+{g}": round(p_hit(a, m, a + g), 4) for g in (0.5, 1.0, 1.5, 2.0)}
    out["2차진입선_도달확률"] = round(p_hit(a, m, a + 0.5), 4)
    out["2차_추가이익_σ"] = 0.5                                     # 같은 m 까지 돌아오면 2차는 0.5σ 더 벌어진 곳에서 출발
    t = fpt_hl(a, m)
    out["수렴시간_반감기단위"] = {"중앙": round(float(np.median(t)), 2), "평균": round(float(t.mean()), 2),
                         "P(>1.5HL)": round(float((t > 1.5).mean()), 3), "P(>3HL)": round(float((t > 3).mean()), 3),
                         "P(>5HL)": round(float((t > 5).mean()), 3)}
    t2 = fpt_hl(a + 1.5, m, seed=1)
    out["손절선에서_수렴까지_반감기단위_중앙"] = round(float(np.median(t2)), 2)

    # 실제: A1 의 가격손절 거래 ↔ 같은 시각에 들어간 A 거래
    A = pd.read_csv(OUT / "trades_A.csv", parse_dates=["t_in", "t_out"])
    A1 = pd.read_csv(OUT / "trades_A1.csv", parse_dates=["t_in", "t_out"])
    st = A1[A1["exit"] == "가격손절"]
    j = st.merge(A[["t_in", "net", "exit", "hold_min"]], on="t_in", suffixes=("_손절", "_그대로"))
    out["가격손절_짝비교"] = {"손절건": int(len(st)), "짝지은건": int(len(j)),
                         "손절로_확정한_손익원": int(j["net_손절"].sum()), "그대로_들고갔으면원": int(j["net_그대로"].sum()),
                         "그대로가_나았던비율": round(float((j["net_그대로"] > j["net_손절"]).mean()), 3) if len(j) else None,
                         "그대로_최악원": int(j["net_그대로"].min()) if len(j) else None,
                         "손절_최악원": int(j["net_손절"].min()) if len(j) else None}
    # 보유시간 — A 거래를 그날 최대보유(=반감기×3) 넘었나로 나눔
    import pickle
    plans, _ = pickle.loads(sorted((HERE / "cache").glob("plans_2022*.pkl"))[-1].read_bytes())
    mh = A["t_in"].dt.normalize().map(lambda d: plans[d].params.max_hold if plans[d].ok else np.nan)
    long_ = A["hold_min"] > mh
    out["보유시간_A"] = {"3HL_넘은거래": int(long_.sum()), "넘은거래_평균순손익원": int(A.loc[long_, "net"].mean()),
                     "안넘은거래_평균순손익원": int(A.loc[~long_, "net"].mean()),
                     "넘은거래_승률": round(float((A.loc[long_, "net"] > 0).mean()), 3)}
    D = pd.read_csv(OUT / "trades_D.csv", dtype={"date": str})
    dd = D.groupby("date")["net"].sum()
    out["D_하루손익"] = {"최악원": int(dd.min()), "−60만_닿은날": int((dd <= -600_000).sum()),
                       "−30만_넘은날": int((dd <= -300_000).sum()), "거래한날": int(len(dd))}
    E = pd.read_csv(OUT / "risk_events_D.csv")
    first_dd = E[(E["규칙"] == "낙폭한도") & E["내용"].astype(str).str.contains("그날 중단")]["t"].min()
    out["D_낙폭중단_처음"] = first_dd
    (OUT / "theory6.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
