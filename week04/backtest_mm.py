"""마켓메이킹 흉내 — 1분봉으로 하는 거친 근사 (README 6-9). 한 종목, 한 호가씩.

  python backtest_mm.py            → results/mm_summary.json

**1분봉으로 마켓메이킹을 정확히 셀 수는 없습니다.** 호가창·대기열·초 단위 체결이 없기 때문. 아래 가정은 체결은 보수적으로,
역선택은 데이터 그대로 반영하도록 잡았고, 결과는 「모양」 을 보는 용도입니다.

규칙 (돌리기 전에 고정)
  호가    매 분 끝에 직전 종가 기준 매수호가 = 종가 − 1틱, 매도호가 = 종가 + 1틱 을 1계약씩 걺
  체결    다음 1분봉이 그 가격을 **뚫고 지나가야** 체결 (저가 < 매수호가 → 매수 체결, 고가 > 매도호가 → 매도 체결). 닿기만 하면 미체결
          (대기열 맨 뒤 가정). 한 봉에 양쪽 다 체결되면 둘 다 (왕복 2틱 수입). 한 봉에 같은 쪽 최대 1계약
  재고    |재고| ≤ 3. 재고가 +면 매수호가 안 냄(매도만), −면 매도호가만 → 재고를 0 으로 되돌리는 쪽으로만 호가
  마감    15:25 봉에서 남은 재고는 종가에 시장가 정리(1틱 불리) — 밤을 안 넘김
  비용    체결마다 수수료 0.003% (호가 비용은 없음 — 받는 쪽이므로)
  손익    체결가 기준 실현 + 마감 정리. 1계약 = 10주
기록    하루 체결 수, 틱 수입(체결 때마다 +1틱), 역선택(체결 뒤 5분 가격이 내 체결가 대비 얼마나 불리하게 갔나), 재고 손익
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pair_core as pc

OUT = Path(__file__).resolve().parent / "results"
MULT = 10
MAX_INV = 3
FEE = 0.00003


def run(df: pd.DataFrame, col: str) -> pd.DataFrame:
    o, h, l, c = (df[f"{col}_{k}"].values for k in ("open", "high", "low", "close"))
    t = df["time"].values
    day = df["date"].values
    n = len(df)
    inv, cash = 0, 0.0
    rows = []
    fills = []        # (i, side, px) 역선택 측정용
    done_day = None
    for i in range(1, n):
        if day[i] != day[i - 1]:
            inv, cash = 0, 0.0          # 날마다 평평하게 시작 (전날 마감에 정리했음)
            continue
        if day[i] == done_day:
            continue
        px0 = c[i - 1]
        tick = pc.tick_size(px0)
        bid, ask = px0 - tick, px0 + tick
        nb = ns = 0
        if inv <= 0 and l[i] < bid:          # 매수 체결
            inv += 1; cash -= bid * MULT; cash -= bid * MULT * FEE; nb = 1; fills.append((i, +1, bid))
        if inv >= 0 or ns == 0:
            pass
        if (inv - nb) >= 0 and h[i] > ask:   # 매도 체결 (매수와 같은 봉이면 둘 다 — 재고 기준은 봉 시작 재고)
            inv -= 1; cash += ask * MULT; cash -= ask * MULT * FEE; ns = 1; fills.append((i, -1, ask))
        inv = max(-MAX_INV, min(MAX_INV, inv))
        last_bar = (i + 1 == n) or day[i + 1] != day[i] or t[i] >= "152500"
        if last_bar and not (i + 1 < n and day[i + 1] == day[i] and t[i + 1] >= "152500" and t[i] >= "152500" and False):
            pass
        if last_bar and (i + 1 == n or day[i + 1] != day[i] or t[i] == "152500"):
            if inv != 0:                     # 마감 정리: 1틱 불리하게
                px = c[i] - tick if inv > 0 else c[i] + tick
                cash += inv * px * MULT; cash -= abs(inv) * px * MULT * FEE
                rows.append({"date": day[i], "t": t[i], "kind": "마감정리", "qty": inv, "px": px})
                inv = 0
            rows.append({"date": day[i], "t": t[i], "kind": "일손익", "pnl": cash, "fills": sum(1 for f in fills if day[f[0]] == day[i])})
            cash = 0.0
            done_day = day[i]
    R = pd.DataFrame([r for r in rows if r["kind"] == "일손익"])
    # 역선택: 체결 뒤 5분 종가가 체결가 대비 (매수면 하락이 불리)
    adv = []
    for i, s, px in fills:
        if i + 5 < n and day[i + 5] == day[i]:
            adv.append(-s * (c[i + 5] - px))
    R.attrs["adverse_5m"] = float(np.mean(adv)) if adv else float("nan")
    R.attrs["fills"] = len(fills)
    return R


def main():
    df = pd.read_csv(OUT / "pair_1m_full.csv", dtype={"date": str, "time": str})
    df = df[df["time"] < "153000"].reset_index(drop=True)
    summ = {"가정": "1분봉 근사 — 체결은 호가를 뚫고 지나갈 때만, 한 봉 한쪽 1계약, 재고 ±3, 15:25 시장가 정리, 수수료 0.003%", "종목": {}}
    for name, col in (("삼성전자 선물", "a"), ("SK하이닉스 선물", "b")):
        R = run(df, col)
        R["yr"] = R["date"].str[:4]
        tick_mean = df.groupby(df["date"].str[:4])[f"{col}_close"].mean().apply(pc.tick_size)
        by = R.groupby("yr").agg(일수=("pnl", "size"), 체결=("fills", "sum"), 순손익원=("pnl", "sum"), 일평균원=("pnl", "mean"))
        by["체결당_원"] = (by["순손익원"] / by["체결"]).round(0)
        by["틱_원"] = tick_mean.reindex(by.index).values * MULT
        summ["종목"][name] = {"총체결": int(R.attrs["fills"]), "하루평균체결": round(R["fills"].mean(), 1),
                            "총순손익원": int(R["pnl"].sum()), "일평균원": int(R["pnl"].mean()),
                            "일손익_t": round(float(R["pnl"].mean() / R["pnl"].std() * np.sqrt(len(R))), 2),
                            "플러스일비율": round(float((R["pnl"] > 0).mean()), 3),
                            "체결뒤5분_역선택_원(1계약)": round(R.attrs["adverse_5m"] * MULT),
                            "연도별": json.loads(by.round(0).to_json(orient="index", force_ascii=False))}
    OUT.joinpath("mm_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
