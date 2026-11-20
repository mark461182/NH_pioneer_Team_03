"""KRX300 선물(12월물 A086C000) 시장 분석 — 유동성·거래 시간대·코스피200과의 관계.
실행: python3 analyze.py   (data/ 는 krx_dl.py 로 받음. NH 국내 모의 서버 시세, 조회만)
"""
import json, pandas as pd, numpy as np
pd.set_option("display.width", 200)
def L(n, f):
    d = pd.DataFrame(json.load(open(f"data/{n}_{f}.json")))
    d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.drop(columns="timestamp")
A, B = pd.Timestamp("2026-09-11"), pd.Timestamp("2026-11-18 23:59")   # 12월물이 근월물인 구간, 11/19(수능, 1시간 순연)는 시간대 분석에서 뺌
MULT = {"krx300": 50000, "k200": 250000, "mini": 50000, "kq150": 10000}
D = {n: L(n, "1d").loc[A:B] for n in MULT}; M = {n: L(n, "1m").loc[A:B] for n in ("krx300", "k200", "mini")}
snap = json.load(open("data/snapshot_20261119_1720.json"))
print("[0] 11/19 장 마감 뒤 스냅샷")
for n in ("krx300", "k200", "mini", "kq150"):
    p = snap[n]["price"]; print(f"  {n:7s} 종가 {p['last']:8.2f} 거래량 {p['volume']:7d} 미결제 {p['openInterest']:7d} 거래량/미결제 {p['volume']/p['openInterest']:.2f}  현물 {p['spot']} 이론가 {p['theory']} 실제베이시스 {p['basisActual']}")
for k in ("inv_08", "inv_01", "inv_05"):
    s, c = snap[k]
    if s == 200:
        tot = sum(i["buy"] for i in c["investors"]) or 1
        print(" ", k, c.get("instrument"), {i["group"]: f"{(i['buy']+i['sell'])/2/tot*100:.0f}%" for i in c["investors"] if (i["buy"] + i["sell"]) / 2 / tot > 0.005})
print("\n[1] 일별 규모 (9/11~11/18, 거래일", len(D["krx300"]), ")")
T = pd.DataFrame({n: D[n].volume for n in MULT}); N = pd.DataFrame({n: D[n].volume * D[n].close * MULT[n] / 1e8 for n in MULT})
print("  계약수  중앙:", T.median().round(0).to_dict(), "| 최소~최대 krx300:", int(T.krx300.min()), "~", int(T.krx300.max()))
print("  명목(억원) 중앙:", N.median().round(0).to_dict())
print("  krx300 명목 / k200 명목:", f"{(N.krx300/N.k200).median()*100:.2f}%", "| / 미니:", f"{(N.krx300/N.mini).median()*100:.1f}%", "| / 코스닥150:", f"{(N.krx300/N.kq150).median()*100:.0f}%")
c = D["krx300"].close; print("  1계약 명목(억):", round(c.iloc[-1] * 5e4 / 1e8, 2), "| 1틱 0.2pt = 1만원 =", round(0.2 / c.iloc[-1] * 1e4, 2), "bp | k200 1틱 0.05 =", round(0.05 / D['k200'].close.iloc[-1] * 1e4, 2), "bp, 미니 0.02 =", round(0.02 / D['k200'].close.iloc[-1] * 1e4, 2), "bp")
r = pd.DataFrame({n: np.log(D[n].close).diff() for n in MULT}).dropna()
print("  일간 변동성(연율 %):", (r.std() * np.sqrt(252) * 100).round(1).to_dict(), "| 일중 범위 (고-저)/종가 중앙 %:", {n: round(float(((D[n].high - D[n].low) / D[n].close).median() * 100), 2) for n in MULT})
print("  일간 수익률 상관:", r.corr().loc["krx300"].round(3).to_dict())
X = np.c_[np.ones(len(r)), r.k200, r.kq150]; b = np.linalg.lstsq(X, r.krx300, rcond=None)[0]; e = r.krx300 - X @ b
print(f"  회귀 krx300 = {b[1]:.2f}·k200 + {b[2]:.2f}·kq150, R² {1-e.var()/r.krx300.var():.3f}, 잔차 일 σ {e.std()*1e4:.0f}bp  (주의: krx300 종가는 15:19 전후 마지막 체결, k200 은 15:45 → 시각 불일치)")
print("\n[2] 일중 거래 분포 (krx300)")
m = M["krx300"]; g = m.groupby(m.index.date)
print("  하루 체결 있는 분:", g.size().describe()[["min", "50%", "max"]].round(0).to_dict(), "/ 411분 | k200:", int(M["k200"].groupby(M["k200"].index.date).size().median()))
print("  첫 체결 시각 중앙:", g.apply(lambda x: x.index[0].hour * 60 + x.index[0].minute).median() // 60, ":", int(g.apply(lambda x: x.index[0].hour * 60 + x.index[0].minute).median() % 60), "| 마지막 체결 시각:", g.apply(lambda x: x.index[-1].strftime("%H:%M")).value_counts().head(4).to_dict())
print("  15:20 이후 체결 있는 날:", int(g.apply(lambda x: (x.index.strftime("%H:%M") >= "15:21").any()).sum()), "/", len(g), "| 08:45~08:59 체결 있는 날:", int(g.apply(lambda x: (x.index.strftime("%H:%M") < "09:00").any()).sum()))
print("  체결 있는 분의 계약수: 중앙", m.volume.median(), "평균", round(m.volume.mean(), 1), "| 1계약짜리 분", f"{(m.volume==1).mean()*100:.0f}%", "| 상위 1% 분이 차지하는 거래량", f"{m.volume.nlargest(len(m)//100).sum()/m.volume.sum()*100:.0f}%")
hh = m.groupby(m.index.floor("30min").strftime("%H:%M")); nd = len(g)
print("  30분 구간별:  시작 | 체결분/30 | 계약/일 | 거래량 비중")
for k, v in hh:
    print(f"     {k}   {len(v)/nd:5.1f}     {v.volume.sum()/nd:7.0f}    {v.volume.sum()/m.volume.sum()*100:5.1f}%")
gap = g.apply(lambda x: pd.Series(x.index).diff().dt.total_seconds().div(60).dropna()).reset_index(drop=True)
print("  체결 사이 간격(분): 중앙", gap.median(), "90%", gap.quantile(.9), "최대", gap.max())
print("\n[3] 가격 움직임 (krx300 체결 분 기준)")
dp = g.close.diff().dropna() / 0.2; dpn = dp[dp != 0]
print("  분간 가격변화(틱): |변화| 중앙", dp.abs().median(), "평균", round(dp.abs().mean(), 1), "| 0틱 비율", f"{(dp==0).mean()*100:.0f}%")
x = pd.concat([d.close.diff().dropna() for _, d in g]); cov = np.mean([(d.close.diff().dropna().values[1:] * d.close.diff().dropna().values[:-1]).mean() for _, d in g if len(d) > 5])
print("  1차 자기상관:", round(float(np.mean([d.close.diff().autocorr(1) for _, d in g if len(d) > 20])), 3), "| Roll 추정 스프레드:", (f"{2*np.sqrt(-cov):.2f}pt = {2*np.sqrt(-cov)/0.2:.1f}틱 = {2*np.sqrt(-cov)/c.iloc[-1]*1e4:.1f}bp" if cov < 0 else "음의 자기공분산 없음"))
print("  분봉 (고-저) 틱: 중앙", ((m.high - m.low) / 0.2).median(), "평균", round(float(((m.high - m.low) / 0.2).mean()), 1))
for n in ("k200", "mini"):
    q = M[n]; gq = q.groupby(q.index.date); tk = 0.05 if n == "k200" else 0.02
    print(f"  비교 {n}: 1차 자기상관", round(float(np.mean([d.close.diff().autocorr(1) for _, d in gq if len(d) > 20])), 3), "분봉 (고-저) 틱 중앙", round(float(((q.high - q.low) / tk).median()), 1))
print("\n[4] 코스피200 선물과의 관계 (09:00~15:19, 1분 격자, 체결 없으면 직전가)")
out = []
for d in sorted(set(m.index.date)):
    grid = pd.date_range(pd.Timestamp(d) + pd.Timedelta(hours=9), pd.Timestamp(d) + pd.Timedelta(hours=15, minutes=19), freq="1min")
    P = pd.DataFrame({n: M[n].close.reindex(M[n].index.union(grid)).ffill().reindex(grid) for n in M}); P["tr"] = M["krx300"].volume.reindex(grid).fillna(0) > 0; P["d"] = d; out.append(P.dropna())
ALL = pd.concat(out)
def R(k):
    return pd.concat([np.log(g_[["krx300", "k200", "mini"]]).diff(k).iloc[k::k] for _, g_ in ALL.groupby("d")]).dropna()
for k in (1, 5, 15, 30, 60):
    q = R(k); print(f"  {k:3d}분 수익률 상관 krx300~k200 {q.krx300.corr(q.k200):.3f}  mini~k200 {q.mini.corr(q.k200):.3f}  σ(bp) krx300 {q.krx300.std()*1e4:.1f} k200 {q.k200.std()*1e4:.1f}  베타 {np.polyfit(q.k200, q.krx300, 1)[0]:.2f}")
q = pd.concat([np.log(g_[["krx300", "k200"]]).diff().assign(d=d_) for d_, g_ in ALL.groupby("d")]).dropna()
print("  시차 상관 corr(k200 r_t, krx300 r_{t+k}):", " ".join(f"{k:+d}:{pd.concat([pd.DataFrame({'x': g_.k200, 'y': g_.krx300.shift(-k)}) for _, g_ in q.groupby('d')]).dropna().corr().iloc[0,1]:+.2f}" for k in range(-3, 6)))
ALL["ratio"] = np.log(ALL.krx300 / ALL.k200); dm = ALL.groupby("d").ratio.agg(["mean", "std", "min", "max"])
print("  log(krx300/k200): 일평균의 범위", round(float(dm['mean'].min()), 4), "~", round(float(dm['mean'].max()), 4), "| 일중 σ 중앙", f"{dm['std'].median()*1e4:.1f}bp", "| 일중 (최대-최소) 중앙", f"{((dm['max']-dm['min']).median())*1e4:.0f}bp")
dev = ALL.ratio - ALL.groupby("d").ratio.transform(lambda s: s.rolling(60, min_periods=20).mean()); dv = dev.dropna()
print("  60분 이동평균 대비 이탈(bp): σ", f"{dv.std()*1e4:.1f}", "| |이탈|>10bp 비율", f"{(dv.abs()>1e-3).mean()*100:.1f}%", "| 이탈 1분 자기상관", round(float(dv.autocorr(1)), 3), "5분", round(float(dv.autocorr(5)), 3), "15분", round(float(dv.autocorr(15)), 3))
print("  비용 기준: krx300 왕복 2틱(시장가 진입·청산, 스프레드 1틱 가정) =", round(0.4 / c.iloc[-1] * 1e4, 2), "bp + k200/미니 헤지 다리")
