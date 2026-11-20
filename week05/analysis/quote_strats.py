"""KRX300 선물 — 걸린 호가를 치는(시장가) 쪽 전략 후보를 1분봉으로 시험 (9/11~11/18, 12월물).
호가 자료가 없어 체결가로만 봄. 비용은 '틱 수' 가정으로 훑음: 진입은 (다음 분 첫 체결가 + c틱), 청산은 (h분 뒤 체결가 − c틱).
실행: python3 quote_strats.py"""
import json, pandas as pd, numpy as np
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); TICK = 0.2; MULT = 5e4; days = sorted(set(x.index.date)); nd = len(days)
def stat(p, label):          # p: 거래별 손익(틱)
    p = pd.Series(p, dtype=float)
    if len(p) < 5: print(f"  {label:44s} 거래 {len(p)}"); return
    print(f"  {label:44s} 거래 {len(p):5d} (하루 {len(p)/nd:4.1f})  평균 {p.mean():+6.2f}틱  승률 {(p>0).mean()*100:3.0f}%  t {p.mean()/p.std()*np.sqrt(len(p)):+5.1f}  합 {p.sum()*TICK*MULT/1e4:+8.0f}만원/1계약")
# ── A. 뒤늦은 호가 치기: 코스피200 이 움직였는데 KRX300 이 덜 따라왔을 때 그 방향으로 진입
print("[A] 코스피200 대비 덜 따라온 KRX300 을 다음 분 첫 체결가에 따라 사기/팔기 (09:00~15:10)")
T = []
for d in days:
    xd = x[x.index.date == d]; kd = k[k.index.date == d]
    grid = pd.date_range(pd.Timestamp(d) + pd.Timedelta(hours=9), pd.Timestamp(d) + pd.Timedelta(hours=15, minutes=19), freq="1min")
    g = pd.DataFrame({"xc": xd.close.reindex(grid), "xo": xd.open.reindex(grid), "kc": kd.close.reindex(grid).ffill(), "ko": kd.open.reindex(grid)}, index=grid)
    g["xl"] = g.xc.ffill(); g["ratio"] = np.log(g.xc / g.kc)                       # 체결된 분의 비율만
    g["m"] = g.ratio.rolling(30, min_periods=8).median().shift(1).ffill()           # 직전 30분(체결 분) 중앙값, 앞을 보지 않음
    g["dev"] = (np.log(g.xl / g.kc) - g.m) * 1e4                                    # +면 KRX300 이 비쌈
    g["rk"] = np.log(g.kc).diff() * 1e4
    g["nxo"] = g.xo.shift(-1)                                                       # 다음 분 첫 체결가 (없으면 NaN → 진입 불가)
    g["nko"] = g.ko.shift(-1)
    for h in (3, 5, 10): g[f"x{h}"] = g.xl.shift(-h); g[f"k{h}"] = g.kc.shift(-h)
    g["day"] = d; T.append(g)
G = pd.concat(T); G = G[G.index.strftime("%H:%M") <= "15:09"]
print("  신호 뒤 '다음 분에 KRX300 체결이 있어 진입 가능'한 비율:", f"{G[G.dev.abs()>=10].nxo.notna().mean()*100:.0f}% (|괴리|≥10bp 인 분 기준)")
for th in (8, 12, 16, 24):
    s = G[(G.dev.abs() >= th) & G.nxo.notna()].copy(); side = -np.sign(s.dev)                # 싸면 사고 비싸면 팜
    left = (-(np.log(s.nxo / s.nko) - s.m) * 1e4 * side * -1)                                # 진입 시점에 남아 있는 괴리(유리한 방향 +)
    print(f" |괴리|≥{th}bp: 신호 {len(s)} (하루 {len(s)/nd:.1f}) | 신호 시점 괴리 평균 {s.dev.abs().mean():.1f}bp → 다음 분 첫 체결 때 남은 괴리 {(-(np.log(s.nxo/s.nko)-s.m)*1e4*side).mean():+.1f}bp")
    for h in (3, 10):
        raw = (s[f"x{h}"] - s.nxo) * side / TICK; hedge = (np.log(s[f"k{h}"] / s.nko) * s.nxo) * side / TICK
        for c in (0, 1, 2):
            stat((raw - 2 * c).dropna(), f"  {h:2d}분 보유, 비용 편도 {c}틱, 헤지 없음")
        stat((raw - hedge - 2 * 1.5).dropna(), f"  {h:2d}분 보유, 코스피200 헤지, 총비용 3틱")
# ── B. 장 시작: 첫 체결이 코스피200 기준 적정가에서 벗어나 있나
print("\n[B] 장 시작 첫 체결가의 괴리 (적정가 = 전일 15:00~15:19 비율 중앙값 × 그 시각 코스피200 선물)")
rows = []; prev = None
for d in days:
    xd = x[x.index.date == d]; kd = k[k.index.date == d]; J = xd.join(kd, how="inner", rsuffix="_k")
    late = J[(J.index.strftime("%H:%M") >= "15:00") & (J.index.strftime("%H:%M") <= "15:19")]; r_today = np.log(late.close / late.close_k).median() if len(late) >= 5 else None
    if prev is not None and len(xd):
        t0 = xd.index[0]; kk = kd[:t0]
        if len(kk):
            fair = np.exp(prev) * kk.close.iloc[-1]; dev0 = (xd.open.iloc[0] / fair - 1) * 1e4
            nxt = xd[xd.index > t0]; row = {"day": d, "t0": t0.strftime("%H:%M"), "dev0": dev0, "vol0": xd.volume.iloc[0]}
            for h in (5, 15, 30):
                a = xd[:t0 + pd.Timedelta(minutes=h)]; b = kd[:t0 + pd.Timedelta(minutes=h)]
                row[f"dev{h}"] = (a.close.iloc[-1] / (np.exp(prev) * b.close.iloc[-1]) - 1) * 1e4
                row[f"pnl{h}"] = -np.sign(dev0) * ((a.close.iloc[-1] - xd.open.iloc[0]) - np.exp(prev) * (b.close.iloc[-1] - kk.close.iloc[-1])) / TICK   # 괴리 반대로 진입, 코스피200 헤지
            rows.append(row)
    if r_today is not None: prev = r_today
O = pd.DataFrame(rows).set_index("day")
print("  날 수", len(O), "| 첫 체결 |괴리| bp: 중앙", round(O.dev0.abs().median(), 1), "75%", round(O.dev0.abs().quantile(.75), 1), "최대", round(O.dev0.abs().max(), 1), "| 첫 체결 분 거래량 중앙", O.vol0.median())
print("  괴리 크기 변화(중앙 |bp|): 첫 체결", round(O.dev0.abs().median(), 1), "→ 5분 뒤", round(O.dev5.abs().median(), 1), "→ 15분 뒤", round(O.dev15.abs().median(), 1), "→ 30분 뒤", round(O.dev30.abs().median(), 1), "| 첫 괴리와 30분 뒤 괴리 상관", round(float(O.dev0.corr(O.dev30)), 2))
for th in (0, 10, 20, 40):
    s = O[O.dev0.abs() >= th]
    for h in (5, 30): stat((s[f"pnl{h}"] - 3).dropna(), f"|첫 괴리|≥{th}bp, 반대로 진입·{h}분 보유·헤지, 총비용 3틱")
# ── C. 튄 체결 뒤: 직전 체결가에서 크게 벗어난 체결이 나온 다음 분
print("\n[C] 코스피200 은 안 움직였는데 KRX300 체결가만 튄 뒤 (그 분 코스피200 |변화|<5bp, KRX300 괴리 변화 ≥ th)")
G["ddev"] = G.dev.diff()
for th in (10, 15, 25):
    s = G[(G.rk.abs() < 5) & (G.ddev.abs() >= th) & G.nxo.notna() & G.xc.notna()].copy(); side = -np.sign(s.ddev)
    raw = (s.x3 - s.nxo) * side / TICK
    print(f" 튄 폭≥{th}bp: {len(s)}건 (하루 {len(s)/nd:.1f}) | 다음 분 첫 체결에서 이미 되돌아온 비율 {((s.nxo - s.xc)*side > 0).mean()*100:.0f}%, 그 폭 평균 {((s.nxo - s.xc)*side/TICK).mean():+.1f}틱")
    for c in (0, 1): stat((raw - 2 * c).dropna(), f"  다음 분 첫 체결가에 반대로 진입·3분 보유, 편도 {c}틱")
