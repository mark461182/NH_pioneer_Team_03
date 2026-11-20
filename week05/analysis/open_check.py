"""장 시작 괴리 재검증 — 신호(첫 체결의 괴리)를 본 '다음 체결 분'의 첫 체결가에 진입 (같은 분 시각 어긋남 효과 제거).
적정가 = 전일 15:00~15:19 (KRX300/코스피200) 비율 중앙값 × 코스피200 선물. 실행: python3 open_check.py"""
import json, pandas as pd, numpy as np
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); TICK = 0.2; days = sorted(set(x.index.date)); rows = []; prev = None; allm = []
for d in days:
    xd = x[x.index.date == d]; kd = k[k.index.date == d]; J = xd.join(kd, how="inner", rsuffix="_k")
    late = J[(J.index.strftime("%H:%M") >= "15:00") & (J.index.strftime("%H:%M") <= "15:19")]; r_today = np.log(late.close / late.close_k).median() if len(late) >= 5 else None
    if prev is not None and len(J) > 3:
        R = np.exp(prev); J = J.copy(); J["dev_o"] = (J.open / (R * J.open_k) - 1) * 1e4; J["dev_c"] = (J.close / (R * J.close_k) - 1) * 1e4; J["day"] = d; J["n"] = range(len(J)); allm.append(J)
        a, b = J.iloc[0], J.iloc[1]; sig = a.dev_c; side = -np.sign(sig)          # 첫 체결 분의 종가 괴리가 신호, 다음 체결 분 시가에 진입
        row = {"day": d, "t_sig": J.index[0].strftime("%H:%M"), "t_in": J.index[1].strftime("%H:%M"), "sig": sig, "left": b.dev_o, "wait": (J.index[1] - J.index[0]).seconds / 60}
        for h in (5, 15, 30):
            e = J[:J.index[1] + pd.Timedelta(minutes=h)].iloc[-1]; row[f"pnl{h}"] = side * ((e.close - b.open) - R * (e.close_k - b.open_k)) / TICK
        rows.append(row)
    if r_today is not None: prev = r_today
O = pd.DataFrame(rows).set_index("day"); nd = len(O)
print("날 수", nd, "| 신호(첫 체결 분 종가 괴리) |bp| 중앙", round(O.sig.abs().median(), 1), "→ 진입 때(다음 체결 분 시가) 같은 방향으로 남은 괴리 평균", round(float((O.left * np.sign(O.sig)).mean()), 1), "bp, 중앙", round(float((O.left * np.sign(O.sig)).median()), 1), "| 진입까지 기다린 분 중앙", O.wait.median())
print("신호 시각:", O.t_sig.value_counts().sort_index().head(6).to_dict(), "…")
def stat(p, label):
    p = p.dropna(); print(f"  {label:42s} 거래 {len(p):3d}  평균 {p.mean():+6.1f}틱  중앙 {p.median():+6.1f}  승률 {(p>0).mean()*100:3.0f}%  t {p.mean()/p.std()*np.sqrt(len(p)):+5.1f}  최악 {p.min():+6.1f}")
for th in (0, 10, 20):
    s = O[O.sig.abs() >= th]
    for h in (5, 15, 30): stat(s[f"pnl{h}"] - 3, f"|신호|≥{th}bp, {h}분 보유, 헤지, 총비용 3틱")
A = pd.concat(allm); A["hm"] = A.index.strftime("%H:%M")
print("\n시간대별 괴리 크기 (전일 비율 기준, 체결 분 종가, |bp| 중앙) 와 '한 분 뒤에도 같은 부호로 남는 비율·폭':")
A["nxt_o"] = A.groupby("day").dev_o.shift(-1)
for a, b in (("08:45", "08:59"), ("09:00", "09:04"), ("09:05", "09:14"), ("09:15", "09:59"), ("10:00", "11:59"), ("12:00", "15:19")):
    s = A[(A.hm >= a) & (A.hm <= b)].dropna(subset=["nxt_o"]); big = s[s.dev_c.abs() >= 10]
    print(f"  {a}~{b}: 체결 분 {len(s):5d}, |괴리| 중앙 {s.dev_c.abs().median():5.1f}bp | |괴리|≥10bp {len(big):4d}건 → 다음 체결 분 시가에 같은 방향으로 남은 폭 평균 {(big.nxt_o*np.sign(big.dev_c)).mean():+5.1f}bp (신호 평균 {big.dev_c.abs().mean():.1f})")
print("\n주의: 낮 시간 괴리에는 그날 비율이 전일과 달라진 몫(지수 구성 차이 하루 10bp + 베이시스 변화)이 섞여 있음")
