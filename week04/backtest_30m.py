"""4주차 추가 — 30분봉에서 두 방향: ① OU 평균회귀 vs ② 볼린저 밴드 돌파(추세추종)

  python backtest_30m.py [--data results/pair_1m.csv] [--oos-days 60]

왜 30분봉인가 (README 7절): 1분봉의 「되돌림」은 대부분 1분 안에 끝나는 호가 튐(1분 뒤 자기상관 −0.27, 2분 뒤 0)이라
REST 로 신호를 보고 들어가면 이미 사라져 있었고, 비용 전 이익이 비용의 1/26~1/50 이었음. 30분 움직임은 훨씬 커서
같은 비용이 상대적으로 작아짐. 그리고 1분봉 백테스트에서 「아침 추세」 가 47.7% 날에 잡혔음 — 스프레드가 한 방향으로
이어지는 일이 잦았다면 되돌림이 아니라 **추세를 따라가는** 쪽이 맞을 수도 있음 → 두 방향을 같은 조건에서 비교.

───────── 규칙 (2026-11-08, 30분봉 결과 보기 전에 고정 — 격자 탐색 없음) ─────────
공통
  봉       1분봉을 30분으로 묶음 (09:00 ~ 15:29, 하루 13봉; 15:30~ 단일가 직전 5분은 버림)
  스프레드  s = ln(삼성) − β·ln(하이닉스), β = 전날까지 20거래일 30분 수익률 회귀 (하루 고정) — 시장 방향을 빼려고
            (1분봉에서 고른 「가격비」는 β=1 이라 하이닉스 쪽 시장 노출이 남음 → 추세추종에선 시장 베팅이 돼 버림)
  크기     1단위 = 하이닉스 1계약 + β 로 맞춘 삼성 (pair_core.hedge_qty)
  체결     신호 봉 종가로 판정 → **다음 봉 시가**에 두 다리 시장가, 다리마다 1틱 + 수수료 0.003% (1분봉과 같은 비용 가정)
  보유     밤을 넘겨도 됨 (30분봉이라 하루 안에 끝내라는 제약은 빼고, 밤사이 갭 손익도 그대로 셈)
  구간     마지막 60영업일 = 검증. 학습/검증 숫자를 따로 보고, 숫자를 고르지 않으니 검증은 「다른 기간에도 같은가」 확인용
① OU 평균회귀 (pair_core 의 OU 식을 30분 단위로)
  날마다 전날까지 20일 스프레드(날마다 평균 뺌)로 AR(1) → κ(1/봉), σ_eq, 반감기(봉)
  비용(σ_eq 배) 넣은 분당→봉당 기대이익 최대 진입선 a·청산 m (pair_core.ou_plan), 손절 a+1.5, 최대 보유 반감기×3 봉
  z = (s − 직전 W봉 평균)/σ_eq, W = 반감기×5 (8~60봉)
  쉬는 날: 회귀 안 보임 · 반감기 < 0.5봉 또는 > 26봉(이틀) · 비용 넘는 진입선 없음
② 볼린저 밴드 돌파 — 교과서 값 그대로 (20봉, ±2σ)
  밴드 = 직전 20봉 스프레드 평균 ± 2 × 표준편차 (현재 봉 제외)
  진입: 종가가 위 밴드를 **위로 뚫으면** 롱 스프레드(삼성 매수·하이닉스 매도), 아래 밴드를 아래로 뚫으면 숏 스프레드
  청산: 종가가 중앙선(20봉 평균)을 반대로 넘으면. 반대쪽 밴드를 뚫으면 청산 후 반대로 진입
  (손절·보유 상한 없음 — 추세추종의 손절 역할을 중앙선이 함)
판정: 학습·검증 둘 다 순손익 > 0 이어야 「성립」.  하나라도 아니면 「30분봉에서도 불성립」.

───────── 11/08 추가 (위 결과를 본 뒤, 이 셋째 규칙은 그 전에 고정) ─────────
③ 변동성 돌파 — 「옵션 매수처럼」: 조용할 땐 안 하고, 흔들림이 커지기 시작할 때 방향 상관없이 올라탐.
  손실은 작게 끊고(옵션 프리미엄처럼) 이익은 길게 둠(옵션 매수의 볼록한 손익).
  압축   직전 20봉 표준편차(밴드 폭)가 직전 120봉의 밴드 폭 중 하위 20% 이하 = 「조용함」
  진입   압축이 최근 3봉 안에 있었고, 이번 봉 종가가 ±2σ 밴드를 뚫으면 → 뚫은 방향으로 (위로 = 롱 스프레드)
  청산   ① 종가가 중앙선을 다시 넘어오면(작은 손실로 끊음 = 프리미엄)
         ② 유리한 쪽 최고점에서 진입 때 σ 의 2배 만큼 되돌리면(이익 지키며 길게 = 추적 손절)
  보는 것  손익 분포의 왜도(옵션 매수면 +), 상위 10% 거래가 전체 이익에서 차지하는 몫, 진입 뒤 흔들림이 실제로 커졌나

───────── 11/09 추가 ④ 「트레이더라면」 — ①②③ 을 다 본 뒤 설계. 같은 데이터로 돌리므로 **표본 내 결과** ─────────
④ 추세추종 + 국면 필터 + 비용 최소화 + 갭 통제 (README 6-7)
  국면   전날까지 20거래일 스프레드 **일별 평균**에 직선을 맞춘 기울기 × 20일 ≥ 그 20일 일별 평균의 표준편차 → 추세 ±1.
         국면 「없음」 이면 그날 신규 진입 없음 (횡보기 거르기)
  진입   30분 볼린저(20봉, ±2σ) 돌파 **국면과 같은 방향만**. 삼성 다리는 돌파 봉 종가에 지정가 → 다음 봉 안에 그 가격을
         뚫고 지나간 거래가 있어야 체결(1분봉 백테스트와 같은 보수적 가정), 체결되면 하이닉스 시장가. 못 하면 놓침
  청산   추적 손절만 — 유리한 쪽 최고점(종가)에서 진입 때 20봉 σ 의 1.5배 되돌리면 다음 봉 시가 시장가. 중앙선 청산·뒤집기 없음
  크기   밤을 넘기는 포지션은 **절반**(삼성 계약 반올림, 하이닉스 1 → 그대로. 1단위라 절반을 못 만들면 15:00 봉에 청산)
         ※ 1단위 기준이라 「절반」 은 삼성 다리만 줄어 헤지가 깨짐 → 대신 **15:00 봉 종가에 그날 손익이 음수면 청산, 양수면 유지**
         (「이기는 포지션만 밤을 넘긴다」 — 갭 위험을 이익 쿠션이 있는 쪽으로만 짐)
  한도   하루 실현손실 −60만 원 → 그날 끝.  고점 대비 −250만 원 → 5거래일 쉼
  판정   없음 — 표본 내 설계라 「성립/불성립」 을 말하지 않음. 11월부터 새 데이터로 앞으로 검증할 **가설**로만 기록
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import pair_core as pc

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BAR = 30
BETA_DAYS = 20
FIT_DAYS = 20
BB_N, BB_K = 20, 2.0


def to_bars(df1: pd.DataFrame, minutes: int = BAR) -> pd.DataFrame:
    d = df1[df1["time"].str.zfill(6) < "153000"].copy()
    key = d.index.floor(f"{minutes}min")
    agg = {}
    for k in "ab":
        agg[f"{k}_open"] = d[f"{k}_open"].groupby(key).first()
        agg[f"{k}_high"] = d[f"{k}_high"].groupby(key).max()
        agg[f"{k}_low"] = d[f"{k}_low"].groupby(key).min()
        agg[f"{k}_close"] = d[f"{k}_close"].groupby(key).last()
    out = pd.DataFrame(agg).dropna()
    out["date"] = out.index.strftime("%Y%m%d")
    return out


def daily_beta(df: pd.DataFrame, days: int) -> pd.Series:
    """날짜 → 전날까지 `days` 일 30분 수익률 회귀 β (그날 첫 봉 갭 제외)."""
    ra = pc.intraday_returns(df["a_close"]); rb = pc.intraday_returns(df["b_close"])
    dates = sorted(set(df.index.normalize()))
    out = {}
    for i, d in enumerate(dates):
        if i < days:
            out[d] = float("nan"); continue
        m = (ra.index >= dates[i - days]) & (ra.index < d)
        x, y = rb[m], ra[m]
        ok = x.notna() & y.notna()
        out[d] = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum()) if ok.sum() > 100 else float("nan")
    return pd.Series(out)


class Book:
    """포지션 하나 — 다음 봉 시가 체결, 비용은 pair_core.Costs."""

    def __init__(self, c: pc.Costs, mult: int = pc.MULT):
        self.c, self.mult = c, mult
        self.pos = None
        self.trades = []

    def open(self, t, dir_, beta, pa, pb, why, z=float("nan")):
        qa = pc.hedge_qty(beta, pa, pb, pc.Params())
        self.pos = {"t_in": t, "dir": dir_, "qa": qa, "qb": 1, "pa": pa, "pb": pb, "beta": beta, "why_in": why, "z_in": z}

    def close(self, t, pa, pb, why):
        p = self.pos
        gross = p["dir"] * (p["qa"] * (pa - p["pa"]) - p["qb"] * (pb - p["pb"])) * self.mult
        cost = (pc.leg_cost(p["qa"], p["pa"], self.c, self.mult) + pc.leg_cost(p["qa"], pa, self.c, self.mult)
                + pc.leg_cost(p["qb"], p["pb"], self.c, self.mult) + pc.leg_cost(p["qb"], pb, self.c, self.mult))
        self.trades.append({**p, "t_out": t, "pa_out": pa, "pb_out": pb, "why_out": why,
                            "overnight": p["t_in"].date() != t.date(), "hold_bars": None,
                            "gross": round(gross), "cost": round(cost), "net": round(gross - cost),
                            "date": t.strftime("%Y%m%d")})
        self.pos = None


def run_ou(df: pd.DataFrame, betas: pd.Series, c: pc.Costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    la, lb = np.log(df["a_close"].values), np.log(df["b_close"].values)
    day = df.index.normalize()
    dates = sorted(set(day))
    plans, rows = {}, []
    for i, d in enumerate(dates):
        bt = float(betas.get(d, float("nan")))
        if i < FIT_DAYS or not math.isfinite(bt):
            plans[d] = None; continue
        xs = [la[day == p] - bt * lb[day == p] for p in dates[i - FIT_DAYS:i]]
        fit = pc.fit_ou_days(xs, min_n=150, min_len=5)
        last = df[day == dates[i - 1]].iloc[-1]
        qa = pc.hedge_qty(bt, last["a_close"], last["b_close"], pc.Params())
        row = {"date": str(d.date()), "beta": round(bt, 4), "ok": False}
        if fit.ok:
            cs = pc.round_trip_cost(qa, 1, last["a_close"], last["b_close"], c) / (qa * last["a_close"] * pc.MULT * fit.sigma_eq)
            row.update(half_life_bars=round(fit.half_life, 2), cost_sigma=round(cs, 3))
            if 0.5 <= fit.half_life <= 26:
                pl = pc.ou_plan(fit, cs)
                if pl.ok:
                    plans[d] = {"beta": bt, "fit": fit, "plan": pl, "W": int(min(60, max(8, round(5 * fit.half_life)))),
                                "max_hold": max(2, int(round(3 * fit.half_life)))}
                    row.update(ok=True, entry=pl.entry, exit=pl.exit)
                else:
                    row["why"] = pl.why
            else:
                row["why"] = f"반감기 {fit.half_life:.1f}봉"
        else:
            row["why"] = fit.why
        plans.setdefault(d, None)
        rows.append(row)
    bk = Book(c)
    A_o, B_o = df["a_open"].values, df["b_open"].values
    pend, held = None, 0
    idx = df.index
    for i in range(len(df)):
        d = day[i]
        if pend is not None:                                   # 이번 봉 시가 체결
            kind, info = pend
            if kind in ("exit", "flip") and bk.pos:
                bk.close(idx[i], A_o[i], B_o[i], info["why"])
            if kind == "enter":
                bk.open(idx[i], info["dir"], info["beta"], A_o[i], B_o[i], info["why"], info["z"])
                held = 0
            pend = None
        pl = plans.get(d)
        if bk.pos is not None:
            held += 1
        if pl is None:
            if bk.pos is not None and i < len(df) - 1:     # 쉬는 날로 넘어왔으면 정리
                pend = ("exit", {"why": "쉬는 날 정리"})
            continue
        W, se, bt = pl["W"], pl["fit"].sigma_eq, pl["beta"]
        if i < W:
            continue
        x = la[i - W:i + 1] - bt * lb[i - W:i + 1]
        z = (x[-1] - x[:-1].mean()) / se
        a, m, stop = pl["plan"].entry, pl["plan"].exit, pl["plan"].entry + 1.5
        if i == len(df) - 1:
            continue
        if bk.pos is not None:
            dr = bk.pos["dir"]
            if z * dr >= -m:
                pend = ("exit", {"why": "수렴"})
            elif abs(z) >= stop and z * dr < 0:
                pend = ("exit", {"why": "z 손절"})
            elif held >= pl["max_hold"]:
                pend = ("exit", {"why": "보유 상한"})
        elif abs(z) < stop:
            if z >= a:
                pend = ("enter", {"dir": -1, "beta": bt, "why": f"z {z:+.2f} ≥ {a}", "z": z})
            elif z <= -a:
                pend = ("enter", {"dir": +1, "beta": bt, "why": f"z {z:+.2f} ≤ −{a}", "z": z})
    if bk.pos is not None:
        bk.close(idx[-1], df["a_close"].iloc[-1], df["b_close"].iloc[-1], "데이터 끝")
    return pd.DataFrame(bk.trades), pd.DataFrame(rows)


def run_bb(df: pd.DataFrame, betas: pd.Series, c: pc.Costs) -> pd.DataFrame:
    la, lb = np.log(df["a_close"].values), np.log(df["b_close"].values)
    day = df.index.normalize()
    bt_bar = np.array([betas.get(d, np.nan) for d in day], float)
    bk = Book(c)
    A_o, B_o = df["a_open"].values, df["b_open"].values
    idx = df.index
    pend = None
    for i in range(len(df)):
        if pend is not None:
            kind, info = pend
            if kind in ("exit", "flip") and bk.pos:
                bk.close(idx[i], A_o[i], B_o[i], info["why"])
            if kind in ("enter", "flip"):
                bk.open(idx[i], info["dir"], info["beta"], A_o[i], B_o[i], info["why"], info.get("z", np.nan))
            pend = None
        bt = bt_bar[i]
        if i < BB_N + 1 or not math.isfinite(bt) or i == len(df) - 1:
            continue
        x = la[i - BB_N - 1:i + 1] - bt * lb[i - BB_N - 1:i + 1]     # 그날 β 로 다시 계산 (β 바뀌어도 밴드 일관)
        win_now, win_prev = x[-BB_N - 1:-1], x[-BB_N - 2:-2]
        mid, sd = win_now.mean(), win_now.std(ddof=1)
        midp, sdp = win_prev.mean(), win_prev.std(ddof=1)
        up, dn, upp, dnp = mid + BB_K * sd, mid - BB_K * sd, midp + BB_K * sdp, midp - BB_K * sdp
        cur, prev = x[-1], x[-2]
        brk_up = cur > up and prev <= upp
        brk_dn = cur < dn and prev >= dnp
        pb_ = (cur - mid) / sd if sd > 0 else np.nan
        if bk.pos is None:
            if brk_up:
                pend = ("enter", {"dir": +1, "beta": bt, "why": "위 밴드 돌파", "z": pb_})
            elif brk_dn:
                pend = ("enter", {"dir": -1, "beta": bt, "why": "아래 밴드 돌파", "z": pb_})
        else:
            dr = bk.pos["dir"]
            if (dr > 0 and brk_dn) or (dr < 0 and brk_up):
                pend = ("flip", {"dir": -dr, "beta": bt, "why": "반대 밴드 돌파 → 뒤집기", "z": pb_})
            elif (dr > 0 and cur < mid) or (dr < 0 and cur > mid):
                pend = ("exit", {"why": "중앙선 복귀"})
    if bk.pos is not None:
        bk.close(idx[-1], df["a_close"].iloc[-1], df["b_close"].iloc[-1], "데이터 끝")
    return pd.DataFrame(bk.trades)


def run_squeeze(df: pd.DataFrame, betas: pd.Series, c: pc.Costs, look: int = 120, q: float = 0.2,
                recent: int = 3, trail: float = 2.0) -> pd.DataFrame:
    la, lb = np.log(df["a_close"].values), np.log(df["b_close"].values)
    day = df.index.normalize()
    bt_bar = np.array([betas.get(d, np.nan) for d in day], float)
    bk = Book(c)
    A_o, B_o = df["a_open"].values, df["b_open"].values
    idx = df.index
    pend, sq_hist, ext, sd_in = None, [], None, None
    n = len(df)
    sq_flag = np.zeros(n, bool)
    for i in range(n):
        if pend is not None:
            kind, info = pend
            if kind == "exit" and bk.pos:
                bk.trades_extra = None
                bk.close(idx[i], A_o[i], B_o[i], info["why"])
                bk.trades[-1].update(vol_ratio=info.get("vr"))
            if kind == "enter":
                bk.open(idx[i], info["dir"], info["beta"], A_o[i], B_o[i], info["why"], info.get("z", np.nan))
                bk.pos["i_in"], ext, sd_in = i, None, info["sd"]
            pend = None
        bt = bt_bar[i]
        if i < BB_N + look + 2 or not math.isfinite(bt) or i == n - 1:
            continue
        x = la[i - BB_N - look - 1:i + 1] - bt * lb[i - BB_N - look - 1:i + 1]     # 그날 β 로 다시 계산
        cur, prev = x[-1], x[-2]
        # 직전 look 봉 각각의 20봉 표준편차 (현재 봉 제외) → 압축 판단
        sds = np.array([x[j - BB_N:j].std(ddof=1) for j in range(len(x) - look - 1, len(x))])   # 마지막이 현재 기준 밴드
        sd, sd_prev = sds[-1], sds[-2]
        mid = x[-BB_N - 1:-1].mean(); midp = x[-BB_N - 2:-2].mean()
        sq_flag[i] = sd <= np.quantile(sds[:-1], q)
        recently = sq_flag[max(0, i - recent):i].any() or sq_flag[i]
        up, dn = mid + BB_K * sd, mid - BB_K * sd
        upp, dnp = midp + BB_K * sd_prev, midp - BB_K * sd_prev
        if bk.pos is None:
            if recently and cur > up and prev <= upp:
                pend = ("enter", {"dir": +1, "beta": bt, "why": "압축 뒤 위로 돌파", "sd": sd, "z": (cur - mid) / sd})
            elif recently and cur < dn and prev >= dnp:
                pend = ("enter", {"dir": -1, "beta": bt, "why": "압축 뒤 아래로 돌파", "sd": sd, "z": (cur - mid) / sd})
        else:
            dr = bk.pos["dir"]
            fav = dr * cur
            ext = fav if ext is None else max(ext, fav)
            k_in = bk.pos["i_in"]
            vr = None
            if i - k_in >= 1:
                seg = x[-(i - k_in) - 1:]
                vr = float(np.abs(np.diff(seg)).mean() / max(np.abs(np.diff(x[-BB_N - 1 - (i - k_in):-(i - k_in)])).mean(), 1e-12))
            if dr * (cur - mid) < 0:
                pend = ("exit", {"why": "중앙선 복귀(작은 손실)", "vr": vr})
            elif ext - fav >= trail * sd_in:
                pend = ("exit", {"why": "추적 손절(이익 지킴)", "vr": vr})
    if bk.pos is not None:
        bk.close(idx[-1], df["a_close"].iloc[-1], df["b_close"].iloc[-1], "데이터 끝")
    return pd.DataFrame(bk.trades)


def daily_regime(df: pd.DataFrame, betas: pd.Series, days: int = 20) -> pd.Series:
    """날짜 → 추세 국면(+1/−1/0): 전날까지 `days` 일 스프레드 일별 평균의 직선 기울기 × days ≥ 그 평균들의 σ."""
    la, lb = np.log(df["a_close"]), np.log(df["b_close"])
    day = df.index.normalize()
    dates = sorted(set(day))
    out = {}
    for i, d in enumerate(dates):
        bt = betas.get(d, np.nan)
        if i < days or not math.isfinite(bt):
            out[d] = 0; continue
        means = np.array([float((la[day == p] - bt * lb[day == p]).mean()) for p in dates[i - days:i]])
        t = np.arange(days, dtype=float)
        slope = np.polyfit(t, means, 1)[0]
        out[d] = int(np.sign(slope)) if abs(slope * days) >= means.std(ddof=1) else 0
    return pd.Series(out)


def run_trader(df: pd.DataFrame, betas: pd.Series, c: pc.Costs, trail: float = 1.5,
               day_loss: float = 600_000, dd_stop: float = 2_500_000, rest_days: int = 5) -> tuple[pd.DataFrame, dict]:
    la, lb = np.log(df["a_close"].values), np.log(df["b_close"].values)
    day = df.index.normalize()
    dates = sorted(set(day))
    regime = daily_regime(df, betas)
    bt_bar = np.array([betas.get(d, np.nan) for d in day], float)
    rg_bar = np.array([regime.get(d, 0) for d in day], int)
    A_o, B_o, A_h, A_l = df["a_open"].values, df["b_open"].values, df["a_high"].values, df["a_low"].values
    hhmm = df.index.strftime("%H%M")
    bk = Book(replace(c, passive="a"))
    bk.c_mkt = c
    idx = df.index
    pend, ext = None, None
    day_pnl, equity, peak, rest_until = {}, 0.0, 0.0, None
    missed = blocked_regime = 0
    n = len(df)
    for i in range(n):
        d = day[i]
        if pend is not None:
            kind, info = pend
            if kind == "exit" and bk.pos:
                bk.close(idx[i], A_o[i], B_o[i], info["why"])
                t = bk.trades[-1]
                # 진입이 삼성 지정가였으면 삼성 진입 슬리피지 0 으로 비용 정정
                t["cost"] = int(round(pc.leg_cost(t["qa"], t["pa"], c, bk.mult, 0.0) + pc.leg_cost(t["qa"], t["pa_out"], c, bk.mult)
                                      + pc.leg_cost(t["qb"], t["pb"], c, bk.mult) + pc.leg_cost(t["qb"], t["pb_out"], c, bk.mult)))
                t["net"] = t["gross"] - t["cost"]
                day_pnl[d] = day_pnl.get(d, 0.0) + t["net"]
                equity += t["net"]; peak = max(peak, equity)
                if equity - peak <= -dd_stop:
                    k = dates.index(d)
                    rest_until = dates[min(len(dates) - 1, k + rest_days)]
            elif kind == "enter_wait":                           # 삼성 지정가: 이번 봉 안에 뚫고 지나갔나
                lim, side = info["lim"], info["a_side"]
                hit = (A_l[i] < lim) if side == "buy" else (A_h[i] > lim)
                if hit:
                    bk.open(idx[i], info["dir"], info["beta"], lim, B_o[i], info["why"], info["z"])
                    bk.pos["sd_in"], ext = info["sd"], None
                else:
                    missed += 1
            pend = None
        bt, rg = bt_bar[i], rg_bar[i]
        if i < BB_N + 1 or not math.isfinite(bt) or i == n - 1:
            continue
        x = la[i - BB_N - 1:i + 1] - bt * lb[i - BB_N - 1:i + 1]
        mid, sd = x[-BB_N - 1:-1].mean(), x[-BB_N - 1:-1].std(ddof=1)
        midp, sdp = x[-BB_N - 2:-2].mean(), x[-BB_N - 2:-2].std(ddof=1)
        cur, prev = x[-1], x[-2]
        if bk.pos is not None:
            dr = bk.pos["dir"]
            fav = dr * cur
            ext = fav if ext is None else max(ext, fav)
            if ext - fav >= trail * bk.pos["sd_in"]:
                pend = ("exit", {"why": "추적 손절"})
            elif hhmm[i] == "1500":                              # 마지막 봉 전: 밤 넘길지
                pa_now, pb_now = df["a_close"].values[i], df["b_close"].values[i]
                unreal = dr * (bk.pos["qa"] * (pa_now - bk.pos["pa"]) - bk.pos["qb"] * (pb_now - bk.pos["pb"])) * bk.mult
                if unreal <= 0:
                    pend = ("exit", {"why": "장마감 — 지는 포지션은 밤 안 넘김"})
            continue
        if rest_until is not None and d <= rest_until:
            continue
        if day_pnl.get(d, 0.0) <= -day_loss:
            continue
        brk_up = cur > mid + BB_K * sd and prev <= midp + BB_K * sdp
        brk_dn = cur < mid - BB_K * sd and prev >= midp - BB_K * sdp
        if not (brk_up or brk_dn):
            continue
        want = +1 if brk_up else -1
        if rg == 0 or rg != want:
            blocked_regime += 1
            continue
        a_side = "buy" if want > 0 else "sell"
        pend = ("enter_wait", {"dir": want, "beta": bt, "lim": df["a_close"].values[i], "a_side": a_side,
                               "why": f"국면 {rg:+d} · {'위' if want > 0 else '아래'} 돌파", "z": (cur - mid) / sd, "sd": sd})
    if bk.pos is not None:
        bk.close(idx[-1], df["a_close"].iloc[-1], df["b_close"].iloc[-1], "데이터 끝")
    T = pd.DataFrame(bk.trades)
    meta = {"놓친신호(지정가)": missed, "국면불일치로_건너뜀": blocked_regime,
            "국면분포": {str(k): int(v) for k, v in regime.value_counts().sort_index().items()}}
    return T, meta


def payoff_shape(T: pd.DataFrame) -> dict:
    """옵션 매수 같은 손익인가 — 왜도, 상위 10% 거래 몫, 진입 뒤 흔들림 배율."""
    if len(T) < 5:
        return {}
    g = T["gross"].astype(float)
    top = g.sort_values(ascending=False)
    k = max(1, len(g) // 10)
    out = {"비용전_왜도": round(float(g.skew()), 2),
           "상위10%_거래_비용전합": int(top.iloc[:k].sum()), "나머지_비용전합": int(top.iloc[k:].sum()),
           "이긴거래_평균": int(g[g > 0].mean()) if (g > 0).any() else 0,
           "진거래_평균": int(g[g <= 0].mean()) if (g <= 0).any() else 0}
    if "vol_ratio" in T:
        v = T["vol_ratio"].dropna()
        if len(v):
            out["진입뒤_흔들림_배율_중앙"] = round(float(v.median()), 2)
    return out


def stats(T: pd.DataFrame, days: list) -> dict:
    if T.empty:
        return {"거래": 0}
    daily = T.groupby("date")["net"].sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    sh = daily.mean() / daily.std() * math.sqrt(252) if daily.std() > 0 else float("nan")
    hold = (pd.to_datetime(T["t_out"]) - pd.to_datetime(T["t_in"])).dt.total_seconds() / 3600
    return {"거래": int(len(T)), "승률": round(float((T.net > 0).mean()), 3),
            "비용전_합계원": int(T.gross.sum()), "비용_합계원": int(T.cost.sum()), "순손익_합계원": int(T.net.sum()),
            "거래당_비용전원": int(T.gross.mean()), "거래당_비용원": int(T.cost.mean()),
            "일간샤프": round(float(sh), 2), "최대낙폭원": int((eq - eq.cummax()).min()),
            "평균보유시간": round(float(hold.mean()), 1), "밤넘긴비율": round(float(T["overnight"].mean()), 3),
            "청산사유": T["why_out"].value_counts().to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "pair_1m.csv"))
    ap.add_argument("--oos-days", type=int, default=60)
    a = ap.parse_args()
    d1 = pd.read_csv(a.data, dtype={"date": str, "time": str})
    d1.index = pd.to_datetime(d1["date"] + d1["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    df = to_bars(d1)
    days = sorted(df["date"].unique())
    is_days, oos_days = days[:-a.oos_days], days[-a.oos_days:]
    c = pc.Costs()
    betas = daily_beta(df, BETA_DAYS)
    # 30분 잔차 자기상관 — 1분봉(−0.27)과 비교
    ra, rb = pc.intraday_returns(df["a_close"]), pc.intraday_returns(df["b_close"])
    bt = df.index.normalize().map(betas).astype(float)
    e = (ra - bt * rb).dropna()
    T_ou, P_ou = run_ou(df, betas, c)
    T_bb = run_bb(df, betas, c)
    T_sq = run_squeeze(df, betas, c)
    T_tr, meta_tr = run_trader(df, betas, c)
    summ = {"봉": f"{BAR}분", "데이터": [days[0], days[-1]], "학습일": len(is_days), "검증일": len(oos_days),
            "30분_잔차자기상관": {"1봉": round(float(e.autocorr(1)), 3), "2봉": round(float(e.autocorr(2)), 3),
                             "3봉": round(float(e.autocorr(3)), 3)},
            "OU_계획": {"거래가능일": int(P_ou["ok"].sum()), "전체일": len(P_ou),
                      "반감기_중앙값_봉": float(P_ou["half_life_bars"].median()),
                      "비용σ_중앙값": float(P_ou["cost_sigma"].median()),
                      "진입선_중앙값": float(P_ou.loc[P_ou["ok"], "entry"].median()) if P_ou["ok"].any() else None},
            "결과": {}}
    for name, T in (("① OU 평균회귀", T_ou), ("② 볼린저 돌파 추세추종", T_bb), ("③ 변동성 돌파 (옵션 매수형)", T_sq),
                    ("④ 트레이더 — 국면 필터·삼성 지정가·추적 손절·갭 통제 [표본 내]", T_tr)):
        r = {}
        for per, dd in (("학습", is_days), ("검증", oos_days)):
            Tp = T[T["date"].isin(dd)] if len(T) else T
            r[per] = {**stats(Tp, dd), **payoff_shape(Tp)}
        if name.startswith("④"):
            r["판정"] = "판정 없음 — ①②③ 결과를 본 뒤 설계한 표본 내 가설. 11월 이후 새 데이터로 앞으로 검증"
            r.update(meta_tr)
        else:
            ok = all(r[p].get("순손익_합계원", -1) > 0 for p in ("학습", "검증"))
            r["판정"] = "성립" if ok else "불성립"
        summ["결과"][name] = r
    OUT.mkdir(exist_ok=True)
    (OUT / "backtest_30m_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.concat([T_ou.assign(전략="OU"), T_bb.assign(전략="볼린저"), T_sq.assign(전략="변동성돌파"), T_tr.assign(전략="트레이더")]).to_csv(OUT / "backtest_30m_trades.csv", index=False, encoding="utf-8-sig")
    P_ou.to_csv(OUT / "ou_days_30m.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    try:
        import logging
        import matplotlib
        matplotlib.use("Agg")
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for nm, T in (("① OU 평균회귀", T_ou), ("② 볼린저 돌파 추세추종", T_bb), ("③ 변동성 돌파", T_sq), ("④ 트레이더 [표본 내]", T_tr)):
            if len(T):
                g = T.sort_values("t_out")
                ax.plot(pd.to_datetime(g["t_out"]), g["net"].cumsum() / 1e4, label=f"{nm} (비용 뒤)")
                ax.plot(pd.to_datetime(g["t_out"]), g["gross"].cumsum() / 1e4, ls=":", label=f"{nm} (비용 전)")
        ax.axvline(pd.to_datetime(oos_days[0]), color="gray", ls="--", lw=1)
        ax.set_ylabel("누적 손익 (만원)"); ax.set_title("30분봉 — ①②③ 사전등록 · ④ 표본 내 설계 (하이닉스 1계약 기준)"); ax.legend(fontsize=8)
        (OUT / "figs").mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig(OUT / "figs" / "5_30m_two_ways.png", dpi=130); plt.close(fig)
    except Exception as ex:
        print("그림 생략:", ex)


if __name__ == "__main__":
    main()
