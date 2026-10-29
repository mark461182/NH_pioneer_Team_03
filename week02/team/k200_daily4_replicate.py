"""팀원 전략 「코스피200 선물 일봉 4전략 동일비중」 독립 재현 — 기본값 / 튜닝 두 경우.

명세: 팀원 통합본 부록 A (식), 4-2 (비용), 4-3 (지표). 숫자는 고르지 않고 문서에 적힌 값 그대로 쓴다.
데이터: 국내_차트(일) A016C000, cidx_yn=1, 원시가 + occurs1_diff_rt (team/k200_demo_daily_with_diff.csv)
        팀원 데이터는 2000년부터(실전 서버), 우리는 2021-12-27부터(모의 서버). 겹치는 날짜의 값은 부록 B 기준값과 일치 확인.
목적: 팀원이 보고한 검증 구간(2025-01-02 ~) 성과가 다른 사람이 따로 짠 코드로도 나오는지 확인.

실행: python team/k200_daily4_replicate.py
"""
import numpy as np
import pandas as pd

TICK, FEE = 0.05, 0.0000145
OOS_START = "2025-01-02"

PARAMS = {
    "기본": dict(s1=(12, "ls"), s2=(22, 1.0, 1.0), s3=(10, 4, True, 1), s4=(2, 5, 200, 5)),
    "튜닝": dict(s1=(5, "lo"), s2=(22, 1.0, 0.5), s3=(20, 1, True, 1), s4=(3, 30, 100, 10)),
}


def load(path="team/k200_demo_daily_with_diff.csv"):
    d = pd.read_csv(path, parse_dates=["dt"]).sort_values("dt").reset_index(drop=True)
    d["prev"] = d["close"] - d["diff"]
    d["ret"] = d["diff"] / d["prev"]
    d.loc[0, "ret"] = np.nan
    gap = ((d["close"] - d["close"].shift(1)) - d["diff"]).round(2).fillna(0.0)
    d["roll"] = (gap.abs() > 1e-9).astype(int)
    d.loc[0, "roll"] = 0
    adj = np.zeros(len(d))                                  # 마지막 날 0, 뒤에서부터 누적
    for t in range(len(d) - 1, 0, -1):
        adj[t - 1] = adj[t] + gap.iloc[t]
    d["aclose"] = d["close"] + adj
    d["I"] = (1 + d["ret"].fillna(0)).cumprod()
    d["is_me"] = d["dt"].dt.to_period("M") != d["dt"].shift(-1).dt.to_period("M")   # 데이터상 월말
    return d


def s1(d, lookback, mode):
    me = d.index[d["is_me"]]
    M = d.loc[me, "I"]
    sig = np.sign(M / M.shift(lookback) - 1)
    if mode == "lo":
        sig = sig.clip(lower=0)
    t = pd.Series(np.nan, index=d.index)
    t.loc[me] = sig
    return t.ffill().fillna(0.0)


def s2(d, window, cap, power):
    r = d["ret"].fillna(0.0)
    rv = r.rolling(window).var(ddof=1) * window
    me = d.index[d["is_me"]]
    RV = rv.loc[me].reset_index(drop=True)
    I = d.loc[me, "I"].reset_index(drop=True)
    R = I / I.shift(1) - 1
    raw = R / RV.shift(1) ** power
    c = R.expanding().std(ddof=1) / raw.expanding().std(ddof=1)
    w = np.minimum(c / RV ** power, cap)
    t = pd.Series(np.nan, index=d.index)
    t.loc[me] = w.to_numpy()
    return t.ffill().fillna(0.0)


def decay_linear(s, n):
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda a: np.nan if np.isnan(a).any() else (a * w).sum() / w.sum(), raw=True)


def ts_rank(s, n):
    def f(a):
        if np.isnan(a).any():
            return np.nan
        return (a[:-1] < a[-1]).sum() / (n - 1)
    return s.rolling(n).apply(f, raw=True)


def s3(d, delta, decay, gate, flip):
    H, L, C = d["high"], d["low"], d["close"]
    x = ((H - C) - (C - L)) / (H - L).replace(0, np.nan)
    sig = flip * np.sign(decay_linear(x - x.shift(delta), decay))
    sd20 = d["ret"].rolling(20).std(ddof=1)
    sd60 = d["ret"].rolling(60).std(ddof=1)
    on = (sd20 > sd60.rolling(60).mean() * 0.95) & ((d["ret"].rolling(60).mean() > -0.01) | (ts_rank(sd20, 252) > 0.8))
    if gate:
        return sig.where(on).ffill().fillna(0.0)
    return sig.fillna(0.0)


def rsi(c, n):
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dd).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def s4(d, n, entry, trend, exit_n):
    c = d["aclose"]
    up = c.rolling(trend).mean() if trend > 0 else pd.Series(-np.inf, index=d.index)
    ex = c.rolling(exit_n).mean()
    R = rsi(c, n)
    pos, out = 0, []
    for t in range(len(d)):
        if pos == 0 and c.iloc[t] > up.iloc[t] and R.iloc[t] < entry:
            pos = 1
        elif pos == 1 and c.iloc[t] > ex.iloc[t]:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=d.index, dtype=float)


def net_returns(d, target):
    hold = target.shift(1).fillna(0.0)
    before = hold.shift(1).fillna(0.0)
    gross = before * (d["open"] - d["prev"]) / d["prev"] + hold * (d["close"] - d["open"]) / d["prev"]
    per = (TICK + FEE * d["open"]) / d["prev"]
    cost = (hold - before).abs() * per + d["roll"] * 2 * before.abs() * per
    return (gross - cost).fillna(0.0), hold


def metrics(r):
    r = r.dropna()
    E = (1 + r).cumprod()
    n = len(r)
    return dict(누적=E.iloc[-1] - 1, CAGR=E.iloc[-1] ** (252 / n) - 1, 변동성=r.std(ddof=1) * np.sqrt(252),
                샤프=r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std(ddof=1) > 0 else np.nan,
                MDD=(E / E.cummax() - 1).min(), 일수=n)


def run_case(d, p):
    tg = {"S1": s1(d, *p["s1"]), "S2": s2(d, *p["s2"]), "S3": s3(d, *p["s3"]), "S4": s4(d, *p["s4"])}
    nets, holds = {}, {}
    for k, t in tg.items():
        nets[k], holds[k] = net_returns(d, t)
    nets["동일비중"] = pd.concat([nets[k] for k in ("S1", "S2", "S3", "S4")], axis=1).mean(axis=1)
    bh, _ = net_returns(d, pd.Series(1.0, index=d.index))
    nets["매수보유"] = bh
    return nets, holds


def fmt(x, pct=True):
    return "-" if pd.isna(x) else (f"{x * 100:+.1f}%" if pct else f"{x:.2f}")


def main():
    d = load()
    oos = d["dt"] >= OOS_START
    print(f"데이터 {d['dt'].min().date()} ~ {d['dt'].max().date()} ({len(d)}일), 롤오버 감지 {d['roll'].sum()}번")
    print(f"검증 구간 {OOS_START} ~ {d.loc[oos, 'dt'].max().date()} ({oos.sum()}일)  ※ 팀원 문서는 2026-10-26까지\n")
    rows = []
    for case, p in PARAMS.items():
        nets, holds = run_case(d, p)
        for k in ("매수보유", "S1", "S2", "S3", "S4", "동일비중"):
            if case == "튜닝" and k == "매수보유":
                continue
            m = metrics(nets[k][oos])
            rows.append([case if k != "매수보유" else "-", k, fmt(m["누적"]), fmt(m["CAGR"]), fmt(m["변동성"]),
                         fmt(m["샤프"], False), fmt(m["MDD"])])
        seg = (d["dt"] >= "2026-06-22")
        print(f"[{case}] 2026-06-22 이후 구간 수익: " + ", ".join(
            f"{k} {fmt((1 + nets[k][seg]).prod() - 1)}" for k in ("매수보유", "동일비중", "S1", "S2", "S3", "S4")))
    print()
    out = pd.DataFrame(rows, columns=["경우", "전략", "누적", "CAGR", "변동성", "샤프", "최대낙폭"])
    print(out.to_string(index=False))
    out.to_csv("team/replicate_oos.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
