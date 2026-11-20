"""장 시작 괴리 전략 — 첫 체결 한 번만이 아니라 시간 창을 넓혀 여러 번 진입하면 빈도·손익이 어떻게 되는지 (9/11~11/18).
  적정가 = 전일 15:00~15:19 (KRX300/코스피200) 비율 중앙값 × 코스피200 선물.  신호 = 체결 분 종가 괴리 |bp| ≥ th
  진입 = 신호 다음 체결 분의 시가(두 상품 모두), 괴리 반대 방향 KRX300 + 코스피200 헤지. 청산 = h분 뒤 종가. 보유 중엔 새 신호 무시. 총비용 3틱.
실행: python3 open_freq.py"""
import json, pandas as pd, numpy as np
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); TICK = 0.2; days = sorted({d for d in x.index.normalize()}); DAY = {}; prev = None
for d in days:
    J = x[x.index.normalize() == d].join(k[k.index.normalize() == d], how="inner", rsuffix="_k")
    late = J[(J.index.strftime("%H:%M") >= "15:00") & (J.index.strftime("%H:%M") <= "15:19")]; r_today = np.log(late.close / late.close_k).median() if len(late) >= 5 else None
    if prev is not None and len(J) > 5:
        J = J.copy(); J["R"] = np.exp(prev); J["dev"] = (J.close / (J.R * J.close_k) - 1) * 1e4; DAY[d] = J
    if r_today is not None: prev = r_today
nd = len(DAY)
def bt(a, b, th, h, cost=3.0, adapt=False, max_per_day=99):
    P = []
    for d, J in DAY.items():
        hm = J.index.strftime("%H:%M"); busy = None; n = 0; idx = J.index
        base = 0.0
        for i in range(len(J) - 1):
            t = idx[i]
            if adapt and i >= 5: base = J.dev.iloc[max(0, i - 30):i].median()       # 오늘 지금까지의 수준을 빼고 봄
            if hm[i] < a or hm[i] > b or (busy is not None and t < busy) or n >= max_per_day: continue
            s = J.dev.iloc[i] - (base if adapt else 0)
            if abs(s) < th: continue
            e = J.iloc[i + 1]; side = -np.sign(s); out = J[:idx[i + 1] + pd.Timedelta(minutes=h)].iloc[-1]
            pnl = side * ((out.close - e.open) - e.R * (out.close_k - e.open_k)) / TICK - cost
            P.append((d, hm[i], abs(s), pnl)); busy = idx[i + 1] + pd.Timedelta(minutes=h); n += 1
    return pd.DataFrame(P, columns=["day", "hm", "sig", "pnl"])
def show(label, T):
    if len(T) < 5: print(f"  {label:46s} 거래 {len(T)}"); return
    dp = T.groupby("day").pnl.sum().reindex(list(DAY)).fillna(0); hf = len(dp) // 2
    print(f"  {label:46s} 거래 {len(T):4d} (하루 {len(T)/nd:4.1f}) 평균 {T.pnl.mean():+6.1f}틱 중앙 {T.pnl.median():+6.1f} 승률 {(T.pnl>0).mean()*100:3.0f}% t {T.pnl.mean()/T.pnl.std()*np.sqrt(len(T)):+5.1f} | 하루 합 평균 {dp.mean():+6.1f}틱, 플러스인 날 {(dp>0).mean()*100:3.0f}%, 최악의 날 {dp.min():+5.0f} | 앞/뒤 절반 {dp[:hf].sum():+5.0f}/{dp[hf:].sum():+5.0f}")
print(f"날 수 {nd}.  손익 단위 틱(1틱 = 1만원/계약), 총비용 3틱 차감 후\n[1] 시간 창별 (괴리 ≥10bp, 10분 보유, 전일 비율 기준)")
for a, b in (("08:45", "08:59"), ("09:00", "09:14"), ("09:15", "09:29"), ("09:30", "09:59"), ("10:00", "11:59"), ("12:00", "15:00")): show(f"{a}~{b}", bt(a, b, 10, 10))
print("[2] 08:45~09:29 창, 문턱·보유시간별")
for th in (8, 12, 16, 24):
    for h in (5, 15): show(f"괴리 ≥{th}bp, {h}분 보유", bt("08:45", "09:29", th, h))
print("[3] 같은 창, 하루 최대 진입 횟수 제한 (괴리 ≥10bp, 10분 보유)")
for mx in (1, 2, 3): show(f"하루 최대 {mx}번", bt("08:45", "09:29", 10, 10, max_per_day=mx))
print("[4] 기준을 '오늘 직전 30개 체결 분의 괴리 중앙값'으로 바꿔 하루 종일 (괴리 ≥ th, 10분 보유) — 전일 비율이 낮 동안 어긋나는 문제 제거")
for th in (10, 15, 20): show(f"09:10~15:00, 괴리 ≥{th}bp", bt("09:10", "15:00", th, 10, adapt=True))
