"""mm_sim.py 와 같은 모의를 30초봉으로 (호가를 30초마다 고침). 9/11~11/18.
  적정가 F = (직전 60개 체결 봉의 KRX300/코스피200 비율 중앙값) × 그 30초 시작 때 코스피200 선물가
  매 30초 시작에 F∓d틱 에 1계약씩, 30초 끝에 취소. 체결 = KRX300 저가/고가가 내 가격을 뚫고 지나감.
실행: python3 mm_sim_30s.py"""
import json, pandas as pd, numpy as np
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_30s.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); TICK = 0.2; days = sorted({d for d in x.index.normalize()}); out = []
for d in days:
    xd = x[x.index.normalize() == d]; kd = k[k.index.normalize() == d]
    g = pd.date_range(d + pd.Timedelta(hours=9, minutes=1), d + pd.Timedelta(hours=15, minutes=15), freq="30s")
    G = pd.DataFrame({"xo": xd.open, "xh": xd.high, "xl": xd.low, "xc": xd.close, "xv": xd.volume}).reindex(g); G["ko"] = kd.open.reindex(g); G["kc"] = kd.close.reindex(g).ffill(); G["ko"] = G.ko.fillna(G.kc.shift(1))
    ratio = np.log(xd.close / kd.close.reindex(xd.index)).dropna(); m = ratio.rolling(60, min_periods=20).median().shift(1)
    G["m"] = m.reindex(m.index.union(g)).ffill().reindex(g); G["F"] = np.exp(G.m) * G.ko; G["xlast"] = xd.close.reindex(xd.index.union(g)).ffill().reindex(g)
    for h in (0, 6, 20): G[f"Fv{h}"] = np.exp(G.m) * G.kc.shift(-h); G[f"xv{h}"] = G.xlast.shift(-h)
    G["day"] = d; out.append(G.dropna(subset=["F"]))
G = pd.concat(out); nd = G.day.nunique()
r30 = np.log(k.close).groupby(k.index.date).diff().dropna()
print("30초 구간", len(G), "일수", nd, "| KRX300 체결 있는 구간", int(G.xv.notna().sum()), f"({G.xv.notna().mean()*100:.0f}%)", "| 코스피200 30초 표준편차", round(float(r30.std() * 1e4), 1), "bp =", round(float(r30.std() * 1e4 / 0.59), 1), "틱")
def fills(d_):
    bid = np.floor((G.F - d_ * TICK) / TICK + 1e-9) * TICK; ask = np.ceil((G.F + d_ * TICK) / TICK - 1e-9) * TICK
    hb = (G.xl < bid - 1e-9).fillna(False); ha = (G.xh > ask + 1e-9).fillna(False)
    return pd.concat([G[hb].assign(px=bid[hb], sd=1), G[ha].assign(px=ask[ha], sd=-1)]).sort_index()
print(f"\n{'거리 d':>6s} {'체결/일':>7s} | 체결 후 손익(틱, 코스피200 환산): {'30초 끝':>7s} {'3분':>6s} {'10분':>6s} | KRX300 이후가 10분 | {'승률':>4s} {'불리체결':>6s} {'불리폭(틱)':>8s} {'일별 t':>6s}")
for d_ in (1, 2, 3, 5, 8, 12, 16, 24, 32):
    f = fills(d_)
    if len(f) < 10: print(f"{d_:6d} {len(f)/nd:7.1f}"); continue
    mo = {h: (f[f"Fv{h}"] - f.px) * f.sd / TICK for h in (0, 6, 20)}; mx = (f.xv20 - f.px) * f.sd / TICK; kmv = (f.kc - f.ko) / f.ko * 1e4 * f.sd / 0.59
    dayp = mo[20].groupby(f.day).sum().reindex(sorted(G.day.unique())).fillna(0)
    print(f"{d_:6d} {len(f)/nd:7.1f} |                               {mo[0].mean():+7.2f} {mo[6].mean():+6.2f} {mo[20].mean():+6.2f} | {mx.mean():+13.2f}    | {(mo[20]>0).mean()*100:3.0f}% {(kmv<0).mean()*100:5.0f}% {(-kmv).mean():+8.1f} {dayp.mean()/dayp.std()*np.sqrt(len(dayp)):+6.1f}")
print("\n거리별 손익을 '받는 폭(d) − 불리하게 움직인 폭'으로 보면: 30초 끝 손익 ≈ d − 불리폭. 1분 갱신 때 5틱 거리의 불리폭은 12.6틱이었음")
f = fills(5); mo = (f.Fv20 - f.px) * f.sd / TICK
print("d=5틱 시간대별 (건수, 10분 손익):", {h: (int((f.index.hour == h).sum()), round(float(mo[f.index.hour == h].mean()), 1)) for h in range(9, 16)})
calm = G.kc.pct_change().abs().rolling(20).mean().shift(1) * 1e4; f["calm"] = calm.reindex(f.index).values
q = f.calm.quantile([.25, .5, .75]).values
for nm, m_ in (("직전 10분이 가장 잔잔한 1/4", f.calm <= q[0]), ("가장 요동친 1/4", f.calm >= q[2])):
    s = f[m_]; print(f"d=5틱, {nm}: {len(s)}건, 10분 손익 {((s.Fv20 - s.px) * s.sd / TICK).mean():+.2f}틱")
