"""OU 보정판을 원래 설정(243일, 학습 183 / 검증 60, 가격비 스프레드)에 그대로 넣어 비교 (README 6-13)

  python backtest_ou_noise.py   → results/ou_noise_backtest.json

바꾸는 것은 pair_core.fit_ou_days 하나 — 「AR(1) 그대로」 → 「OU + 호가 튐 분리」(ou_noise.fit_noise).
나머지(진입·청산선 공식, 창 = 반감기×5, 보유 = 반감기×3, 손절, 비용, 체결 지연, 추세필터)는 모두 같음.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_pair as bp
import ou_noise as on
import pair_core as pc

OUT = Path(__file__).resolve().parent / "results"
ORIG = pc.fit_ou_days


def fit_noise_days(days, min_n: int = 200, min_len: int = 30):
    xs = [np.asarray(x, float) for x in days if len(np.asarray(x)) >= min_len]
    if sum(len(x) for x in xs) < min_n:
        return pc.OUFit(False, why="표본 부족")
    g = on.autocov_days(xs)
    f, _ = on.fit_noise(g)
    return f


def run(fit_fn):
    pc.fit_ou_days = fit_fn
    df = bp.load(OUT / "pair_1m.csv")
    base, c_mkt = pc.Params(), pc.Costs()
    c_pas = replace(c_mkt, passive="auto")
    days = sorted(df["date"].unique())
    is_days, oos_days = days[:-60], days[-60:]
    pair_df = pd.concat({"a": df["a_close"], "b": df["b_close"]}, axis=1).dropna()
    V = pc.SpreadView(pair_df, "ratio", None, base)
    A_, B_ = df["a_close"], df["b_close"]
    out, TR = {}, []
    P = {"mkt": pc.plan_days(A_, B_, base, c_mkt, view=V), "pas": pc.plan_days(A_, B_, base, c_pas, view=V)}
    print("plans done", flush=True)
    for name, cc, pas, tf in (("OU 시장가", c_mkt, False, False), ("OU 시장가·추세필터", c_mkt, False, True),
                              ("OU 비싼다리 지정가·추세필터", c_pas, True, True)):
        plans = P["pas" if pas else "mkt"]
        f = pc.features_ou(A_, B_, plans, trend_filter=True, view=V)
        if not tf:
            f = f.drop(columns=["block_long", "block_short"])
        ok = [p for p in plans.values() if p.ok]
        res = {"거래가능일비율": round(len(ok) / len(plans), 3),
               "반감기_중앙분": round(float(np.median([p.fit.half_life for p in plans.values() if p.fit and p.fit.ok])), 1),
               "진입z_중앙": float(np.median([p.plan.entry for p in ok])) if ok else None}
        for per, dd in (("학습", is_days), ("검증", oos_days)):
            m = f.index.strftime("%Y%m%d").isin(dd)
            T = bp.simulate(df, f[m], base, cc, plans=plans, passive=("auto" if pas else None))
            if len(T):
                TR.append(T.assign(설정=name, 구간=per))
            s = bp.stats(T, dd)
            res[per] = {k: s.get(k) for k in ("거래", "총비용전원", "평균비용원", "총순손익원", "일간샤프", "최대낙폭원", "평균보유분")}
        out[name] = res
    pc.fit_ou_days = ORIG
    out["_trades"] = TR
    return out


def main():
    import sys
    if len(sys.argv) > 1:                       # 병렬 실행용: orig | noise 하나만
        k = sys.argv[1]
        r = run(ORIG if k == "orig" else fit_noise_days)
        pd.concat(r.pop("_trades")).to_csv(OUT / f"ou_noise_trades_{k}.csv", index=False, encoding="utf-8-sig")
        (OUT / f"ou_noise_backtest_{k}.json").write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8")
        return
    R = {"기존 (AR(1) 그대로)": json.loads((OUT / "ou_noise_backtest_orig.json").read_text(encoding="utf-8")),
         "보정 (OU + 호가 튐 분리)": json.loads((OUT / "ou_noise_backtest_noise.json").read_text(encoding="utf-8"))}
    (OUT / "ou_noise_backtest.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rows = []
    for k, v in R.items():
        for n, r in v.items():
            for per in ("학습", "검증"):
                rows.append({"OU": k, "설정": n, "구간": per, "거래가능일": r["거래가능일비율"], "반감기": r["반감기_중앙분"],
                             "진입z": r["진입z_중앙"], **r[per]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
