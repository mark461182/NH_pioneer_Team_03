"""삼성전자–SK하이닉스 주식선물 페어 — 1분봉 백테스트 (실행기와 같은 pair_core 규칙)

  python backtest_pair.py [--data results/pair_1m.csv] [--oos-days 60]

체결 가정  : 신호가 난 분의 **다음 분 시가**, 다리마다 1틱 불리 + 수수료 0.003%(편도, 모름 → 가정)
계약 수    : 하이닉스 1계약 고정, 삼성은 진입 시점 β·가격으로 맞춤 (pair_core.hedge_qty)
구간       : 마지막 --oos-days 영업일 = 검증, 그 앞 = 학습

───────── 숫자 고르는 규칙 (2026-11-07, 데이터 받기 전에 고정) ─────────
  격자      : 진입 z ∈ {1.5, 2.0, 2.5, 3.0} × 창 W ∈ {60, 120, 240}분, 청산 0.5·손절 3.5 등 나머지는 고정
  자격      : 학습 거래 ≥ 30  그리고  (진입 − 청산) ≥ 학습 구간 「비용 하한 z」 중앙값
              비용 하한 z = 왕복 비용(원) ÷ (삼성 다리 금액 × 스프레드 σ)  — 이만큼 수렴해야 비용을 메움
  고르기    : 자격 있는 조합 중 **학습 구간 일간 샤프(비용 뒤) 1위**
  검증      : 고른 조합과 기본값(2.0, 120) 두 개만 검증 구간에서 **한 번** 계산. 나머지 조합의 검증값은 안 봄
  자격 있는 조합이 없으면 → 「이 비용에선 이 전략이 성립 안 함」 이 결론 (모의 실행은 기본값으로 실행 확인만)

───────── 11/08 추가: OU 모드와 비싼다리 지정가 (README 3-7·3-8) ─────────
  OU 모드   : 고르지 않음. 날마다 전날까지 2일 스프레드로 OU(κ, σ_eq) 추정 → 비용을 넣은 분당 기대이익 최대
              진입선 (pair_core.ou_plan). 반감기 3~120분 밖·비용 넘는 진입선 없음·기대 주기 780분 초과면 그날 쉼
  지정가    : 하이닉스를 신호 분 종가 지정가로 2분 대기, 뚫고 지나간 거래가 있을 때만 체결로 봄
  비교      : 검증 구간에서 아래 여섯 가지만 한 번씩 — 기본값·시장가 / 격자 고른 것·시장가 / OU·시장가 / OU·지정가
              / OU·시장가·추세필터 / OU·지정가·추세필터   (추세필터는 11/08 추가, README 3-9)
              판정 기준(미리 고정):
                ① 추세필터 채택 여부는 **학습 구간**에서: 「OU·지정가·추세필터」 학습 샤프 > 「OU·지정가」 학습 샤프 → 채택
                ② 채택된 쪽(필터 있음/없음)의 「OU·지정가」 가 검증 구간 순손익 > 0 이고 일간 샤프가 기본값보다 높으면
                   모의 실행은 그 설정으로, 아니면 「비용 앞에서 페어가 안 됨」 을 결론으로 쓰고 기본값으로 실행 확인만
  스프레드 정의 (11/08, README 3-11) : β 하루 고정 / 가격비(β=1) / 칼만 β 셋 중 「OU·지정가」 학습 구간 샤프 1위를
              먼저 고르고, 이후 ③~⑦ 은 모두 그 정의로. 검증 구간엔 고른 정의와 기준(β 고정)만 나란히 실음
  ⑦ 리스크 레이어 (11/08, README 3-10) : 고르지 않는 안전장치 — ①에서 채택된 설정에 얹어 학습→검증을 이어서 한 번 돌림.
              성과 비교가 아니라 「무엇을 막았나 · 최대낙폭 · 예측 VaR99 사후검증(Kupiec)」 을 봄

출력  results/backtest_grid.csv (학습 구간 격자)  results/backtest_summary.json  results/backtest_trades.csv
      results/figs/*.png
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import pair_core as pc
from risk_layer import RiskConfig, RiskLayer, kupiec

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ENTRIES = (1.5, 2.0, 2.5, 3.0)
WINDOWS = (60, 120, 240)
DEFAULT = (2.0, 120)


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"date": str, "time": str})
    d.index = pd.to_datetime(d["date"] + d["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    return d


def simulate(df: pd.DataFrame, feat: pd.DataFrame, p: pc.Params, c: pc.Costs, mult: int = pc.MULT,
             plans: dict | None = None, passive_b: bool = False, passive_bars: int = 2,
             risk: "RiskLayer | None" = None, passive: str | None = None) -> pd.DataFrame:
    """feat = features()/features_ou() 결과. 신호는 그 분 종가로 판정.

    시장가 체결 : 다음 분 시가, 다리마다 c.slip_ticks 틱 불리 + 수수료
    plans       : 날짜 → DayPlan (OU 모드). 그날 ok=False 면 진입 안 함, ok 면 그날 숫자(진입·청산·손절·보유)로 바꿈
    passive     : 지정가로 먼저 걸 다리 "a"/"b"/"auto"(단위당 틱 비용이 큰 쪽, pair_core.passive_leg_of). passive="auto" 는 "b".
                  그 다리를 신호 분 종가에 지정가로 걸고, 다음 `passive_bars` 분 안에 그 가격을 **뚫고 지나간**
                  거래가 있으면(매수: 저가 < 지정가, 매도: 고가 > 지정가) 지정가에 체결된 것으로 봄 (닿기만 하면 안 침 —
                  대기열 뒤쪽이라 보수적으로). 체결된 다음 분 시가에 다른 다리 시장가. 안 체결되면 거래 없음(놓친 신호).
                  청산은 「수렴」 일 때만 같은 방식, 못 하면 그다음 분 시가에 두 다리 시장가.
                  (삼성 9계약 같은 큰 수량도 한 번에 체결된다고 봄 — 1분봉으론 대기열 잔량을 모름, 낙관적인 가정)
    """
    passive = passive or ("b" if passive_b else None)
    cp = replace(c, passive=passive) if passive else c
    sig = pc.Signal(p)
    F = feat
    A_o, B_o = df["a_open"].reindex(F.index).values, df["b_open"].reindex(F.index).values
    HI = {k: (df[f"{k}_high"].reindex(F.index).values if f"{k}_high" in df else df[f"{k}_open"].reindex(F.index).values) for k in "ab"}
    LO = {k: (df[f"{k}_low"].reindex(F.index).values if f"{k}_low" in df else df[f"{k}_open"].reindex(F.index).values) for k in "ab"}
    OP = {"a": A_o, "b": B_o}
    idx, z, corr, pa, pb = F.index, F["z"].values, F["corr"].values, F["a"].values, F["b"].values
    beta, sd = F["beta"].values, F["sd"].values
    bl = F["block_long"].values if "block_long" in F else np.zeros(len(F), bool)
    bs = F["block_short"].values if "block_short" in F else np.zeros(len(F), bool)
    day = idx.normalize()
    s_mkt = c.slip_ticks
    blocked = risk_zero = 0
    trades, pend, skip_day, missed = [], None, False, 0
    n = len(F)
    for i in range(n):
        new_day = i == 0 or day[i] != day[i - 1]
        last_of_day = i == n - 1 or day[i + 1] != day[i]
        if new_day and plans is not None:
            pl = plans.get(day[i])
            skip_day = not (pl is not None and pl.ok)
            if not skip_day:
                sig.p = pl.params
                if risk is not None:          # 리스크 레이어를 쓰면 거래 손실 한도 = 꼬리 예산, 하루 한도 = 그 두 배
                    sig.p = replace(pl.params, loss_limit=risk.cfg.budget, day_loss_limit=2 * risk.cfg.budget)
                    risk.note_sigma(str(day[i].date()), pl.fit.sigma_eq)
        # 1) 걸려 있는 주문 처리
        if pend is not None:
            k = pend["kind"]
            if k == "enter_mkt":
                pk = replace(sig.p, unit_b=pend["k"])
                qa = pc.hedge_qty(pend["beta"], A_o[i], B_o[i], pk)
                if qa > 0:
                    sig.opened(pc.Position(pend["dir"], qa, pk.unit_b, A_o[i], B_o[i], idx[i], pend["z"], pend["beta"]))
                    sig.pos.risk = pend.get("risk", {})
                    sig.pos.s_in = (s_mkt, s_mkt)
                    sig.pos.floor = pc.cost_floor_z(pend["sd"], qa, A_o[i], sig.p.unit_b, B_o[i], c, mult)
                pend = None
            elif k == "enter_wait":                       # 비싼 다리 지정가 대기
                L = pend["leg"]
                hit = (LO[L][i] < pend["lim"]) if pend["side"] == "buy" else (HI[L][i] > pend["lim"])
                if hit:
                    pend = {**pend, "kind": "enter_hedge"}
                elif i >= pend["until"]:
                    missed += 1
                    pend = None
            elif k == "enter_hedge":                      # 지정가 체결 다음 분 → 다른 다리 시장가
                L = pend["leg"]
                xa = pend["lim"] if L == "a" else A_o[i]
                xb = pend["lim"] if L == "b" else B_o[i]
                pk = replace(sig.p, unit_b=pend["k"])
                qa = pend["qa"] if L == "a" else pc.hedge_qty(pend["beta"], xa, xb, pk)
                sig.opened(pc.Position(pend["dir"], qa, pk.unit_b, xa, xb, idx[i], pend["z"], pend["beta"]))
                sig.pos.risk = pend.get("risk", {})
                sig.pos.s_in = (0.0, s_mkt) if L == "a" else (s_mkt, 0.0)
                sig.pos.pleg = L
                sig.pos.floor = pc.cost_floor_z(pend["sd"], qa, xa, sig.p.unit_b, xb, cp, mult)
                pend = None
            elif k == "exit_mkt":
                trades.append(close(sig, idx[i], A_o[i], B_o[i], pend["why"], c, mult, (s_mkt, s_mkt), risk))
                pend = None
            elif k == "exit_wait":
                L = pend["leg"]
                hit = (LO[L][i] < pend["lim"]) if pend["side"] == "buy" else (HI[L][i] > pend["lim"])
                if hit:
                    pend = {**pend, "kind": "exit_hedge", "why": pend["why"] + "(지정가)"}
                elif i >= pend["until"]:
                    pend = {"kind": "exit_mkt", "why": pend["why"] + "(지정가 못 해 시장가)"}
            elif k == "exit_hedge":
                L = pend["leg"]
                xa = pend["lim"] if L == "a" else A_o[i]
                xb = pend["lim"] if L == "b" else B_o[i]
                so = (0.0, s_mkt) if L == "a" else (s_mkt, 0.0)
                trades.append(close(sig, idx[i], xa, xb, pend["why"], c, mult, so, risk))
                pend = None
        if last_of_day:
            if sig.pos is not None:                       # 하루 끝 — 그 분 종가로 정리
                trades.append(close(sig, idx[i], pa[i], pb[i], "데이터 끝", c, mult, (s_mkt, s_mkt), risk))
            pend = None
            continue
        if pend is not None:
            continue
        # 2) 판정
        blk = frozenset(([+1] if bl[i] else []) + ([-1] if bs[i] else []))
        d = sig.decide(idx[i], z[i], corr[i], pa[i], pb[i], mult, blk)
        if d.reason.startswith("추세 반대") and not skip_day:
            blocked += 1
        if d.action == "enter" and not skip_day and math.isfinite(beta[i]):
            info = {"dir": d.dir, "z": z[i], "beta": beta[i], "sd": sd[i], "k": sig.p.unit_b, "risk": {}}
            if risk is not None and plans is not None:      # 리스크 레이어: 이번 거래 단위 수
                pl = plans[day[i]]
                q1 = pc.hedge_qty(beta[i], pa[i], pb[i], replace(sig.p, unit_b=1))
                cu = pc.round_trip_cost(q1, 1, pa[i], pb[i], cp, mult)
                kk, why = risk.size(day=str(day[i].date()), qa=q1, pa=pa[i], pb=pb[i], mult=mult,
                                    sigma_eq=pl.fit.sigma_eq, entry=sig.p.entry, stop=sig.p.stop, tail=pl.tail,
                                    cost_unit=cu)
                if kk == 0:
                    risk_zero += 1
                    continue
                info["k"], info["risk"] = kk, why
            if passive:
                qa_e = pc.hedge_qty(beta[i], pa[i], pb[i], replace(sig.p, unit_b=info["k"]))
                L = pc.passive_leg_of(qa_e, pa[i], info["k"], pb[i], cp)
                a_side = "buy" if d.dir > 0 else "sell"            # 롱 스프레드 = 삼성 매수·하이닉스 매도
                side = a_side if L == "a" else pc.opposite(a_side)
                pend = {"kind": "enter_wait", "leg": L, "lim": pa[i] if L == "a" else pb[i], "side": side,
                        "qa": qa_e, "until": i + passive_bars, **info}
            else:
                pend = {"kind": "enter_mkt", **info}
        elif d.action == "exit":
            L = getattr(sig.pos, "pleg", None)
            if passive and L and d.reason == "수렴":
                a_side = "sell" if sig.pos.dir > 0 else "buy"       # 청산은 진입의 반대
                side = a_side if L == "a" else pc.opposite(a_side)
                pend = {"kind": "exit_wait", "leg": L, "lim": pa[i] if L == "a" else pb[i], "side": side,
                        "until": i + passive_bars, "why": d.reason}
            else:
                pend = {"kind": "exit_mkt", "why": d.reason}
    T = pd.DataFrame(trades)
    T.attrs["missed"] = missed
    T.attrs["trend_blocked"] = blocked      # 추세 필터가 막은 분 수 (같은 괴리가 여러 분 이어지면 여러 번 셈)
    T.attrs["risk_zero"] = risk_zero        # 리스크 레이어가 0단위로 막은 신호 수
    return T


def close(sig: pc.Signal, t, xa: float, xb: float, why: str, c: pc.Costs, mult: int, s_out: tuple,
          risk=None) -> dict:
    """s_out = (삼성, 하이닉스) 청산 슬리피지 틱. 진입 슬리피지는 pos.s_in."""
    pos = sig.pos
    gross = pos.pnl(xa, xb, mult)
    sa_in, sb_in = getattr(pos, "s_in", (c.slip_ticks, c.slip_ticks))
    cost = (pc.leg_cost(pos.qa, pos.pa, c, mult, sa_in) + pc.leg_cost(pos.qa, xa, c, mult, s_out[0])
            + pc.leg_cost(pos.qb, pos.pb, c, mult, sb_in) + pc.leg_cost(pos.qb, xb, c, mult, s_out[1]))
    net = gross - cost
    sig.closed(t, net)
    rk = getattr(pos, "risk", {}) or {}
    if risk is not None:
        risk.on_close(net, str(t.date()))
    return {"date": t.strftime("%Y%m%d"), "t_in": pos.t_in, "t_out": t, "dir": pos.dir, "qa": pos.qa, "qb": pos.qb,
            "k": pos.qb, "var99_pred": rk.get("VaR99원"), "es975_pred": rk.get("ES975원"),
            "beta": round(pos.beta, 4), "z_in": round(pos.z_in, 2), "pa_in": pos.pa, "pb_in": pos.pb,
            "pa_out": xa, "pb_out": xb, "hold_min": (t - pos.t_in).total_seconds() / 60, "exit": why,
            "passive_leg": getattr(pos, "pleg", None), "gross": round(gross), "cost": round(cost), "net": round(net), "floor_z": round(getattr(pos, "floor", np.nan), 3)}


def stats(T: pd.DataFrame, days: list) -> dict:
    if T.empty:
        return {"거래": 0}
    daily = T.groupby("date")["net"].sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    sh = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else float("nan")
    return {"거래": int(len(T)), "거래일": len(days), "승률": round(float((T.net > 0).mean()), 3),
            "평균순손익원": round(float(T.net.mean())), "평균비용원": round(float(T.cost.mean())),
            "총순손익원": round(float(T.net.sum())), "총비용전원": round(float(T.gross.sum())),
            "일간샤프": round(float(sh), 2), "최대낙폭원": round(float((eq - eq.cummax()).min())),
            "평균보유분": round(float(T.hold_min.mean()), 1), "비용하한z_중앙값": round(float(T.floor_z.median()), 3),
            "청산사유": T["exit"].value_counts().to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "pair_1m.csv"))
    ap.add_argument("--oos-days", type=int, default=60)
    a = ap.parse_args()
    df = load(Path(a.data))
    base, c_mkt = pc.Params(), pc.Costs()
    c_pas = replace(c_mkt, passive="auto")          # 비싼 다리(단위당 틱 비용이 큰 쪽)를 지정가로 — README 3-8
    days = sorted(df["date"].unique())
    is_days, oos_days = days[:-a.oos_days], days[-a.oos_days:]
    print(f"데이터 {days[0]} ~ {days[-1]} {len(days)}일 · 학습 {len(is_days)}일 / 검증 {len(oos_days)}일 ({oos_days[0]} ~)")
    in_is = lambda f: f.index.strftime("%Y%m%d").isin(is_days)
    OUT.mkdir(exist_ok=True)

    betas = pc.daily_betas(df["a_close"], df["b_close"], base.beta_days)

    # ── A. 격자 (시장가) — 학습 구간에서만 고름 ──
    feats = {w: pc.features(df["a_close"], df["b_close"], replace(base, window=w), betas) for w in WINDOWS}
    rows = []
    for w in WINDOWS:
        f_is = feats[w][in_is(feats[w])]
        for e in ENTRIES:
            T = simulate(df, f_is, replace(base, entry=e, window=w), c_mkt)
            s_ = stats(T, is_days)
            floor = s_.get("비용하한z_중앙값", float("nan"))
            ok = s_.get("거래", 0) >= 30 and (e - base.exit) >= floor
            rows.append({"진입z": e, "창분": w, **{k: v for k, v in s_.items() if k != "청산사유"}, "자격": ok})
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "backtest_grid.csv", index=False, encoding="utf-8-sig")
    print(G.to_string(index=False))
    elig = G[G["자격"]]
    chosen = None if elig.empty else tuple(elig.sort_values("일간샤프", ascending=False).iloc[0][["진입z", "창분"]])

    # ── B0. 스프레드 정의 고르기 (README 3-11) — 「OU · 비싼다리 지정가」 의 **학습 구간** 샤프로만 ──
    A_, B_ = df["a_close"], df["b_close"]
    pair_df = pd.concat({"a": A_, "b": B_}, axis=1).dropna()
    spread_rows, views = [], {}
    for mode in pc.SPREAD_MODES:
        V = pc.SpreadView(pair_df, mode, betas if mode == "beta" else None, base)
        views[mode] = V
        pl_ = pc.plan_days(A_, B_, base, c_pas, view=V)
        f_ = pc.features_ou(A_, B_, pl_, view=V)
        T_ = simulate(df, f_[in_is(f_)], base, c_pas, plans=pl_, passive="auto")
        st_ = stats(T_, is_days)
        okd = [x for x in pl_.values() if x.ok]
        spread_rows.append({"정의": mode, "학습_거래": st_.get("거래", 0), "학습_일간샤프": st_.get("일간샤프"),
                            "학습_총순손익원": st_.get("총순손익원"), "거래가능일비율": round(len(okd) / len(pl_), 3),
                            "반감기_중앙값분": round(float(np.median([x.fit.half_life for x in okd])), 1) if okd else None,
                            "비용σ_중앙값": round(float(np.median([x.plan.cost_sig for x in okd])), 2) if okd else None})
    S_ = pd.DataFrame(spread_rows)
    S_.to_csv(OUT / "spread_compare_is.csv", index=False, encoding="utf-8-sig")
    print(S_.to_string(index=False))
    cand = S_.dropna(subset=["학습_일간샤프"])
    spread_mode = str(cand.sort_values("학습_일간샤프", ascending=False).iloc[0]["정의"]) if len(cand) else "beta"
    V = views[spread_mode]

    # ── B. OU — 고르지 않음. 날마다 전날까지 데이터로 공식이 정함 (위에서 고른 스프레드 정의로) ──
    plans_mkt = pc.plan_days(A_, B_, base, c_mkt, view=V)
    plans_pas = pc.plan_days(A_, B_, base, c_pas, view=V)
    P = pd.DataFrame([pl.row() for pl in plans_mkt.values()])
    P2 = pd.DataFrame([pl.row() for pl in plans_pas.values()])
    P = P.merge(P2[["date", "ok", "cost_sigma", "entry", "cycle_min"]], on="date", suffixes=("", "_지정가"))
    P.to_csv(OUT / "ou_days.csv", index=False, encoding="utf-8-sig")
    f_ou_mkt = pc.features_ou(A_, B_, plans_mkt, trend_filter=True, view=V)
    f_ou_pas = pc.features_ou(A_, B_, plans_pas, trend_filter=True, view=V)
    no_trend = lambda f: f.drop(columns=["block_long", "block_short"])
    ou_info = {"반감기_중앙값분": round(float(P["half_life_min"].median()), 1),
               "반감기_사분위": [round(float(x), 1) for x in P["half_life_min"].quantile([0.25, 0.75])],
               "비용σ_중앙값_시장가": round(float(P["cost_sigma"].median()), 2),
               "비용σ_중앙값_지정가": round(float(P["cost_sigma_지정가"].median()), 2),
               "거래가능일비율_시장가": round(float(P["ok"].mean()), 3),
               "거래가능일비율_지정가": round(float(P["ok_지정가"].mean()), 3),
               "OU진입선_중앙값_시장가": P.loc[P["ok"], "entry"].median() if P["ok"].any() else None,
               "OU진입선_중앙값_지정가": P.loc[P["ok_지정가"], "entry_지정가"].median() if P["ok_지정가"].any() else None,
               "거래안함_사유": P.loc[~P["ok"], "why"].str.split(" ").str[0].value_counts().to_dict(),
               "아침추세_있던날비율": round(float((P["day_trend"] != 0).mean()), 3)}

    # ── 검증: 미리 정한 네 가지만, 한 번씩 ──
    variants = [("기본값 2.0/120 · 시장가", feats[DEFAULT[1]], replace(base, entry=DEFAULT[0], window=DEFAULT[1]), c_mkt, None, False)]
    if chosen is not None:
        variants.append((f"격자 {chosen[0]}/{int(chosen[1])} · 시장가", feats[int(chosen[1])],
                         replace(base, entry=chosen[0], window=int(chosen[1])), c_mkt, None, False))
    if spread_mode != "beta":     # 기준: 원래 계획(β 하루 고정)의 OU·지정가 — 검증 구간에서 고른 정의와 나란히
        pl_b = pc.plan_days(A_, B_, base, c_pas, view=views["beta"])
        variants.append(("OU · 비싼다리 지정가 [스프레드 β고정 — 기준]",
                         no_trend(pc.features_ou(A_, B_, pl_b, view=views["beta"])), base, c_pas, pl_b, True))
    variants += [("OU · 시장가", no_trend(f_ou_mkt), base, c_mkt, plans_mkt, False),
                 ("OU · 비싼다리 지정가", no_trend(f_ou_pas), base, c_pas, plans_pas, True),
                 ("OU · 시장가 · 추세필터", f_ou_mkt, base, c_mkt, plans_mkt, False),
                 ("OU · 비싼다리 지정가 · 추세필터", f_ou_pas, base, c_pas, plans_pas, True)]
    summary = {"데이터": [days[0], days[-1]], "학습일": len(is_days), "검증일": len(oos_days),
               "비용가정": {"슬리피지틱_편도_다리마다": c_mkt.slip_ticks, "수수료율_편도": c_mkt.fee_rate,
                         "지정가_다리_슬리피지틱": 0.0, "지정가_다리": "auto (계약수×1틱 큰 쪽)"},
               "격자_고른조합": None if chosen is None else {"진입z": chosen[0], "창분": int(chosen[1])},
               "스프레드정의": {"고른것": spread_mode, "학습구간비교": spread_rows},
               "OU": ou_info, "비교": {}}
    all_T = []
    for name, f, prm, cc, plans, pas in variants:
        res = {}
        for per, m, dd in (("학습", in_is(f), is_days), ("검증", ~in_is(f), oos_days)):
            T = simulate(df, f[m], prm, cc, plans=plans, passive=("auto" if pas else None))
            res[per] = {**stats(T, dd), "놓친신호(지정가 미체결)": T.attrs.get("missed", 0),
                        "추세필터_막은분": T.attrs.get("trend_blocked", 0)}
            if len(T):
                all_T.append(T.assign(조합=name, 구간=per))
        summary["비교"][name] = res
    # 추세 필터 채택은 **학습 구간**에서만 판단 (미리 정한 규칙): ④ 대비 ⑥ 학습 샤프가 높으면 채택
    s4 = summary["비교"]["OU · 비싼다리 지정가"]["학습"].get("일간샤프", float("nan"))
    s6 = summary["비교"]["OU · 비싼다리 지정가 · 추세필터"]["학습"].get("일간샤프", float("nan"))
    adopt = bool(s6 > s4)
    summary["추세필터_채택"], summary["추세필터_판단근거"] = adopt, f"학습 샤프 필터없음 {s4} vs 필터 {s6}"

    # ⑦ 리스크 레이어 — 고르지 않음(안전장치라 항상 켬). 채택된 설정(④ 또는 ⑥)에 얹어 크기만 바꿈
    #    학습→검증을 **이어서** 한 번에 돌림 (낙폭 상태·σ 기록이 이어지도록), 표는 구간별로 나눠 셈
    f7 = f_ou_pas if adopt else no_trend(f_ou_pas)
    name7 = "OU · 비싼다리 지정가" + (" · 추세필터" if adopt else "") + " · 리스크레이어"
    rl = RiskLayer(RiskConfig())
    T7 = simulate(df, f7, base, c_pas, plans=plans_pas, passive="auto", risk=rl)
    res7 = {}
    for per, dd in (("학습", is_days), ("검증", oos_days)):
        Tp = T7[T7["date"].isin(dd)] if len(T7) else T7
        st = stats(Tp, dd)
        if len(Tp):
            st["평균단위k"] = round(float(Tp["k"].mean()), 2)
            st["단위분포"] = Tp["k"].value_counts().sort_index().to_dict()
            v = Tp.dropna(subset=["var99_pred"])
            st["VaR99_사후검증"] = kupiec(len(v), int((-v["net"] > v["var99_pred"]).sum()), 0.01)
        res7[per] = st
        if len(Tp):
            all_T.append(Tp.assign(조합=name7, 구간=per))
    res7["리스크레이어_0단위로막은분"] = T7.attrs.get("risk_zero", 0)
    res7["마지막상태"] = {"누적손익원": round(rl.st.equity), "고점대비원": round(rl.st.equity - rl.st.peak), "상태": rl.st.mode}
    summary["비교"][name7] = res7
    summary["리스크레이어_설정"] = asdict(RiskConfig())
    if chosen is None:
        summary["격자_결론"] = "자격 있는 조합 없음 — 시장가 비용에서 비용 하한을 넘는 진입선이 없거나 거래 30건 미만"
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if all_T:
        pd.concat(all_T).to_csv(OUT / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    try:
        figures(df, feats, G, all_T, chosen or DEFAULT, oos_days, P)
    except Exception as ex:
        print("그림 생략:", ex)


def figures(df, feats, G, all_T, combo, oos_days, P=None):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 한글 글꼴 없는 환경의 경고 숨김
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figs = OUT / "figs"
    figs.mkdir(exist_ok=True)

    H = G.pivot(index="창분", columns="진입z", values="일간샤프")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(H.values, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(H.columns)), [str(x) for x in H.columns]); ax.set_yticks(range(len(H.index)), [str(x) for x in H.index])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("진입 z"); ax.set_ylabel("창(분)"); ax.set_title("학습 구간 일간 샤프 (비용 뒤)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(figs / "1_grid_sharpe.png", dpi=130); plt.close(fig)

    if all_T:
        T = pd.concat(all_T)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for name, g in T.groupby("조합"):
            g = g.sort_values("t_out")
            ax.plot(g["t_out"], g["net"].cumsum() / 1e4, label=name)
        ax.axvline(pd.to_datetime(oos_days[0]), color="gray", ls="--", lw=1)
        ax.text(pd.to_datetime(oos_days[0]), ax.get_ylim()[1], " 검증 →", va="top", fontsize=9, color="gray")
        ax.set_ylabel("누적 순손익 (만원)"); ax.set_title("누적 순손익 — 하이닉스 1계약 기준"); ax.legend()
        fig.tight_layout(); fig.savefig(figs / "2_equity.png", dpi=130); plt.close(fig)

    w = int(combo[1])
    f = feats[w]
    last = f[f.index.strftime("%Y%m%d") == oos_days[-1]]
    if len(last):
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].plot(last.index, last["a"] / last["a"].iloc[0], label="삼성전자")
        ax[0].plot(last.index, last["b"] / last["b"].iloc[0], label="SK하이닉스")
        ax[0].legend(); ax[0].set_title(f"{oos_days[-1]} 가격(시가=1) 과 z (창 {w}분)")
        ax[1].plot(last.index, last["z"], color="k", lw=1)
        for y in (combo[0], -combo[0]):
            ax[1].axhline(y, color="tab:red", ls="--", lw=1)
        for y in (0.5, -0.5):
            ax[1].axhline(y, color="tab:green", ls=":", lw=1)
        ax[1].set_ylabel("z"); fig.tight_layout(); fig.savefig(figs / "3_last_day_z.png", dpi=130); plt.close(fig)

    if P is not None and len(P):
        Q = P.copy(); Q["date"] = pd.to_datetime(Q["date"])
        fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
        ax[0].plot(Q["date"], Q["half_life_min"], ".-"); ax[0].set_ylabel("반감기(분)"); ax[0].set_title("날마다 OU 추정 (전날까지 2일)")
        ax[1].plot(Q["date"], Q["cost_sigma"], ".-", label="시장가"); ax[1].plot(Q["date"], Q["cost_sigma_지정가"], ".-", label="비싼다리 지정가")
        ax[1].set_ylabel("왕복 비용 (σ_eq 배)"); ax[1].legend()
        ax[2].plot(Q["date"], Q["entry"], ".", label="시장가"); ax[2].plot(Q["date"], Q["entry_지정가"], ".", label="비싼다리 지정가")
        ax[2].set_ylabel("OU 최적 진입선"); ax[2].legend()
        fig.tight_layout(); fig.savefig(figs / "4_ou_days.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
