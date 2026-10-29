"""코스피200선물 일봉 전략 4개 백테스트 (2001-01-02 ~ 2024-12-30).

모든 숫자는 출처가 있는 값만 쓰고, 이 데이터로 고른 값은 없다.

| 전략 | 출처 | 숫자 |
| --- | --- | --- |
| S1 시계열 모멘텀 | Moskowitz·Ooi·Pedersen (2012, JFE) | 12개월 수익률 부호, 1개월 보유(월말 리밸런싱) |
| S2 변동성 관리 롱 | Moreira·Muir (2017, JF) 식(1)·표 | 비중 = c / 직전 달 실현분산, 월 리밸런싱, 비중 상한 1(무레버리지 사양) |
| S3 IntradayRange 반전 | 건준 IQC 2026 알파 #5 (IS Sharpe 2.25) | 식의 20·60·0.95·0.01·252·0.8·10·4 그대로 |
| S4 RSI(2) 눌림목 | Connors·Alvarez (2008) | 종가 > SMA200 이고 RSI(2) < 5 매수, 종가 > SMA5 청산, 손절 없음 |

체결·비용
- 신호는 그날 종가로 판정, 다음 거래일 시가 체결 (week02 기존 규칙과 같음)
- 슬리피지: 한쪽 1틱(0.05pt) — NH 운영 예시 저장소와 week02 기존 백테스트 가정
- 수수료: 0.00145% — NH선물 Open API 코스피선물 기존 수수료 (2025-08 헤럴드경제 이벤트 기사의 「기존」 값)
- 분기 롤오버: 보유 중이면 만기일에 옛 월물 청산 + 새 월물 진입 = 2번 체결 비용
- 수익률은 보유 월물 기준 (diff_rt 누적 수정가, fetch_daily.py 참고). 증거금 이자는 넣지 않음
- 2000년은 지표 준비 기간 (가장 긴 창: S3 의 252일 순위 + 20일 표준편차)

실행: python backtest_daily.py   (week02 폴더에서)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
TICK = 0.05
FEE = 0.0000145
START, END = "2001-01-02", "2024-12-30"


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "k200_1d_continuous.csv", dtype={"date": str})
    df.index = pd.to_datetime(df.pop("date"))
    df["prev"] = df["close"] - df["diff"]       # 보유 월물의 전일 종가 (롤오버 날도 맞음)
    df["ret"] = df["diff"] / df["prev"]          # 보유 월물 종가 대비 수익률
    df.iloc[0, df.columns.get_loc("ret")] = np.nan
    adj = df["adj_offset"]
    for c in ("open", "high", "low", "close"):
        df["a" + c] = df[c] + adj                # 수정가 (이동평균·RSI 용)
    return df


def month_end_flags(idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(idx.to_period("M"), index=idx)
    return s != s.shift(-1)


# ── 전략: 각자 「그날 종가 기준 목표 포지션」 을 돌려준다 (다음날 시가 체결) ──

def s1_tsmom(df: pd.DataFrame) -> pd.Series:
    me = month_end_flags(df.index)
    idx = (1 + df["ret"].fillna(0)).cumprod()
    m = idx[me]
    sig = np.sign(m / m.shift(12) - 1)          # 지난 12개월 수익률 부호
    return sig.reindex(df.index).ffill()


def s2_volmanaged(df: pd.DataFrame, cap: float = 1.0) -> pd.Series:
    me = month_end_flags(df.index)
    month = df.index.to_period("M")
    r = df["ret"].fillna(0)
    rv = r.groupby(month).apply(lambda x: ((x - x.mean()) ** 2).sum())   # 그 달의 실현분산
    mret = (1 + r).groupby(month).prod() - 1                               # 그 달 보유 수익률
    raw = mret / rv.shift(1)                                               # f / RV²(전달), c 없이
    # c: 「관리 포트폴리오 표준편차 = 매수보유 표준편차」 (논문 정의) 를 그 시점까지 데이터로만 맞춘다
    c = mret.expanding().std() / raw.expanding().std()
    w = (c / rv).clip(upper=cap)                                           # 다음 달에 쓸 비중
    w.index = w.index.to_timestamp(how="end").normalize()
    out = pd.Series(np.nan, index=df.index)
    ends = df.index[me]
    out.loc[ends] = w.reindex(ends.to_period("M").to_timestamp(how="end").normalize()).values
    return out.ffill()


def ts_rank(s: pd.Series, n: int) -> pd.Series:
    # BRAIN ts_rank: 지난 n일 안에서 오늘 값의 순위 (0~1)
    return s.rolling(n).apply(lambda x: (x[:-1] < x[-1]).sum() / (n - 1), raw=True)


def decay_linear(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)


def s3_iqc_range(df: pd.DataFrame, gate: bool = True) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    x = ((df["high"] - df["close"]) - (df["close"] - df["low"])) / rng
    sig = np.sign(decay_linear(x.diff(10), 4))
    ret = df["ret"]
    sd20, sd60 = ret.rolling(20).std(), ret.rolling(60).std()
    on = (sd20 > sd60.rolling(60).mean() * 0.95) & ((ret.rolling(60).mean() > -0.01) | (ts_rank(sd20, 252) > 0.8))
    # BRAIN trade_when(조건, 알파, -1): 조건이 참인 날만 새 값, 거짓이면 직전 값 유지, exit=-1 이라 강제 청산 없음
    return sig.where(on).ffill().fillna(0) if gate else sig.fillna(0)


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()   # Wilder 평활
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def s4_rsi2(df: pd.DataFrame, entry: float = 5) -> pd.Series:
    c = df["aclose"]
    sma200, sma5, r2 = c.rolling(200).mean(), c.rolling(5).mean(), rsi(c, 2)
    pos, out = 0.0, []
    for i in range(len(c)):
        if pos == 0 and c.iloc[i] > sma200.iloc[i] and r2.iloc[i] < entry:
            pos = 1.0
        elif pos == 1 and c.iloc[i] > sma5.iloc[i]:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=df.index)


def s0_buyhold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


# ── 손익 ──

def run(df: pd.DataFrame, target: pd.Series, ticks: float = 1.0) -> pd.DataFrame:
    """target(종가 판정) → 다음날 시가에 체결. 일 수익률(비용 차감)과 포지션을 돌려준다."""
    hold = target.shift(1).fillna(0)            # 오늘 시가부터 들고 있는 포지션
    before = hold.shift(1).fillna(0)            # 어제 종가에 들고 있던 포지션 (밤사이 보유)
    prev = df["prev"]
    gross = before * (df["open"] - prev) / prev + hold * (df["close"] - df["open"]) / prev
    per_trade = (ticks * TICK + FEE * df["open"]) / prev
    cost = (hold - before).abs() * per_trade + df["roll"] * 2 * before.abs() * per_trade
    out = pd.DataFrame({"pos": hold, "gross": gross, "cost": cost})
    out["ret"] = out["gross"] - out["cost"]
    return out


def trades(pos: pd.Series, ret: pd.Series) -> list[float]:
    """포지션이 0 이 아닌 값으로 바뀌어 유지되는 구간 = 거래 1건, 구간 복리 수익."""
    out, cur, acc = [], 0.0, 1.0
    for p, r in zip(pos.values, ret.values):
        if p != cur and cur != 0:
            out.append(acc - 1)
            acc = 1.0
        if p != 0:
            acc *= 1 + r
        cur = p
    if cur != 0:
        out.append(acc - 1)
    return out


def stats(r: pd.Series, pos: pd.Series, cost: pd.Series | None = None) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = eq / eq.cummax() - 1
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    t = trades(np.sign(pos), r)
    always = bool((pos != 0).all())  # 늘 들고 있으면(S0·S2) 거래 단위가 없음
    return {
        "누적수익률%": round((eq.iloc[-1] - 1) * 100, 1),
        "연수익률(CAGR)%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
        "연변동성%": round(r.std() * np.sqrt(252) * 100, 2),
        "샤프": round(r.mean() / r.std() * np.sqrt(252), 2),
        "최대낙폭%": round(dd.min() * 100, 1),
        "월수익률 왜도": round(float(m.skew()), 2),
        "보유비율%": round((pos != 0).mean() * 100, 1),
        "평균 |포지션|": round(pos.abs().mean(), 2),
        "거래수": None if always else len(t),
        "승률%": None if always or not t else round(np.mean([x > 0 for x in t]) * 100, 1),
        "비용합(%p, 단순합)": None if cost is None else round(cost.sum() * 100, 1),
    }


def by_year(r: pd.Series) -> dict:
    y = (1 + r).groupby(r.index.year).prod() - 1
    return {int(k): round(v * 100, 1) for k, v in y.items()}


def main() -> None:
    df = load()
    strategies = {
        "S0 매수보유(비교용)": s0_buyhold(df),
        "S1 시계열모멘텀 12-1": s1_tsmom(df),
        "S2 변동성관리 롱": s2_volmanaged(df),
        "S3 IQC IntradayRange 반전": s3_iqc_range(df),
        "S4 RSI(2) 눌림목": s4_rsi2(df),
    }
    runs = {k: run(df, v) for k, v in strategies.items()}
    win = slice(START, END)

    report = {"기간": f"{START} ~ {END}", "가정": {"슬리피지_틱_한쪽": 1, "수수료": FEE, "체결": "종가 판정 → 다음날 시가"}}
    report["전체"], report["연도별%"], report["구간"] = {}, {}, {}
    for k, o in runs.items():
        o = o.loc[win]
        s = stats(o["ret"], o["pos"], o["cost"])
        report["전체"][k] = s
        report["연도별%"][k] = by_year(o["ret"])
        report["구간"][k] = {
            name: stats(o["ret"].loc[a:b], o["pos"].loc[a:b])["샤프"]
            for name, (a, b) in {"2001-2012": ("2001", "2012"), "2013-2024": ("2013", "2024")}.items()
        }

    rets = pd.DataFrame({k: o.loc[win, "ret"] for k, o in runs.items()})
    four = rets.drop(columns="S0 매수보유(비교용)")
    report["상관(일수익률)"] = four.corr().round(2).to_dict()
    combo = four.mean(axis=1)
    report["4전략 동일비중"] = stats(combo, pd.Series(1.0, index=combo.index))
    report["4전략 동일비중"]["연도별%"] = by_year(combo)

    # 숫자를 바꿔 결과가 얼마나 흔들리는지만 본다 (고르는 데 쓰지 않음)
    sens = {}
    for n in (3, 6, 9, 12):
        me = month_end_flags(df.index)
        idx = (1 + df["ret"].fillna(0)).cumprod()[me]
        sig = np.sign(idx / idx.shift(n) - 1).reindex(df.index).ffill()
        sens[f"S1 룩백 {n}개월"] = stats(run(df, sig).loc[win, "ret"], sig.loc[win])["샤프"]
    sens["S2 상한 1.5"] = stats(run(df, s2_volmanaged(df, 1.5)).loc[win, "ret"], s2_volmanaged(df, 1.5).loc[win])["샤프"]
    sens["S3 거래조건 끔"] = stats(run(df, s3_iqc_range(df, gate=False)).loc[win, "ret"], s3_iqc_range(df, False).loc[win])["샤프"]
    for e in (2, 10):
        p = s4_rsi2(df, e)
        sens[f"S4 RSI2 < {e}"] = stats(run(df, p).loc[win, "ret"], p.loc[win])["샤프"]
    for k, v in strategies.items():
        if k.startswith("S0"):
            continue
        sens[f"{k[:2]} 슬리피지 2틱"] = stats(run(df, v, ticks=2).loc[win, "ret"], v.loc[win])["샤프"]
    report["민감도(샤프)"] = sens

    # 요청 범위(2024) 밖 — 참고용으로만 따로 둔다
    report["참고: 2025-01 ~ 2026-10"] = {
        k: {kk: stats(o.loc["2025":, "ret"], o.loc["2025":, "pos"])[kk] for kk in ("누적수익률%", "샤프", "최대낙폭%")}
        for k, o in runs.items()
    }

    out = RESULTS / "backtest_daily.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({k: (1 + o.loc[win, "ret"]).cumprod() for k, o in runs.items()}).round(4).to_csv(
        RESULTS / "backtest_daily_equity.csv")
    print(json.dumps(report["전체"], ensure_ascii=False, indent=1))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
