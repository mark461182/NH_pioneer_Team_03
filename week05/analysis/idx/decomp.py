"""KRX300 − 코스피200 일별 수익률 차이를 종목 그룹별로 분해 (11/18 ETF 보유수량 고정, 비중은 주가 따라 변동).
실행: ~/nh_pioneer/.venv/bin/python decomp.py   (px.py 먼저)"""
import pandas as pd, numpy as np, json
pd.set_option("display.width", 200)
T = pd.read_csv("active.csv", dtype={"code": str}).set_index("code"); P = pd.read_csv("px.csv", index_col=0, parse_dates=True)
ETF = P[["069500", "292190", "229200"]]; P = P[T.index].ffill()
P = P.loc["2026-09-10":"2026-11-19"]; r = P.pct_change().iloc[1:]
def wts(col):                                  # 11/18 비중 → 보유수량 고정으로 매일 비중 역산
    sh = T[col] / P.loc["2026-11-18"]; v = P * sh; return v.div(v.sum(axis=1), axis=0).shift(1).iloc[1:]
W3, W2 = wts("w300"), wts("w200"); r3, r2 = (W3 * r).sum(axis=1), (W2 * r).sum(axis=1); d = r3 - r2
print(f"기간 {r.index[0].date()}~{r.index[-1].date()} ({len(r)}일)")
print(f"합성 지수 일간 σ: KRX300 {r3.std()*100:.2f}%  코스피200 {r2.std()*100:.2f}%  | 상관 {r3.corr(r2):.4f} | 베타(KRX300~코스피200) {np.polyfit(r2, r3, 1)[0]:.3f}")
print(f"수익률 차이(KRX300−코스피200): 일 σ {d.std()*1e4:.1f}bp, 최대 |차이| {d.abs().max()*1e4:.0f}bp, 기간 누적 {((1+r3).prod()/(1+r2).prod()-1)*100:+.2f}%p  (KRX300 {((1+r3).prod()-1)*100:+.1f}%, 코스피200 {((1+r2).prod()-1)*100:+.1f}%)")
e = ETF.loc[P.index].pct_change().iloc[1:]
print(f"검증: 합성 코스피200 vs KODEX200 상관 {r2.corr(e['069500']):.4f} | ETF 로 본 차이(292190−069500) 일 σ {(e['292190']-e['069500']).std()*1e4:.1f}bp, 합성 차이와 상관 {d.corr(e['292190']-e['069500']):.2f}")
G = T.grp.copy(); G[["005930"]] = "삼성전자"; G[["000660"]] = "SK하이닉스"; G = G.replace({"공통": "공통(반도체 2종 제외)"})
C = ((W3 - W2) * r).T.groupby(G).sum().T; AW = (W3 - W2).T.groupby(G).sum().T.mean()
tab = pd.DataFrame({"종목수": G.value_counts(), "평균 비중차(%p)": AW * 100, "기여 일σ(bp)": C.std() * 1e4, "차이 분산 중 몫(%)": C.apply(lambda s: s.cov(d)) / d.var() * 100, "누적 기여(bp)": C.sum() * 1e4})
print("\n[그룹별 분해]  차이 = Σ (KRX300 비중 − 코스피200 비중) × 종목 수익률"); print(tab.round(2).to_string())
only = T.index[T.w200 == 0]; rx = (W3[only].div(W3[only].sum(axis=1), axis=0) * r[only]).sum(axis=1); a = W3[only].sum(axis=1).mean() * 100
print(f"\n[한 줄 근사]  차이 ≈ {a:.2f}% × (KRX300에만 있는 125종목 수익률 − 코스피200 수익률)")
print(f"  125종목 바구니: 일 σ {rx.std()*100:.2f}%, 코스피200 과 상관 {rx.corr(r2):.2f}, 베타 {np.polyfit(r2, rx, 1)[0]:.2f} | (바구니−코스피200) 일 σ {(rx-r2).std()*100:.2f}%")
ap = a / 100 * (rx - r2); print(f"  근사와 실제 차이의 상관 {ap.corr(d):.3f}, 근사 일 σ {ap.std()*1e4:.1f}bp (실제 {d.std()*1e4:.1f}bp)")
print(f"  125종목 바구니 vs KODEX 코스닥150 수익률 상관 {rx.corr(e['229200']):.2f}")
print(f"  차이 d 와 코스피200 수익률의 상관 {d.corr(r2):+.2f}  → 코스피200 이 1% 오르는 날 KRX300 은 평균 {np.polyfit(r2, d, 1)[0]*100:+.2f}bp 차이")
print("\n[차이가 컸던 날 5]"); big = d.abs().nlargest(5).index
print(pd.DataFrame({"KRX300": r3[big] * 100, "코스피200": r2[big] * 100, "차이bp": d[big] * 1e4, "125종목": rx[big] * 100, "삼성전자": r.loc[big, "005930"] * 100, "하이닉스": r.loc[big, "000660"] * 100}).round(2).to_string())
# 선물과 비교
def L(n):
    x = pd.DataFrame(json.load(open(f"../data/{n}_1d.json"))); x.index = pd.to_datetime(x.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.normalize(); return x.close[x.volume > 0]
f = pd.DataFrame({"x": L("krx300"), "k": L("k200")}).pct_change().dropna(); fd = (f.x - f.k).reindex(d.index).dropna()
print(f"\n[선물과 비교, {len(fd)}일]  선물 일간 차이(KRX300선물−코스피200선물, 종가 기준) σ {fd.std()*1e4:.0f}bp  vs  지수(합성) 차이 σ {d[fd.index].std()*1e4:.1f}bp | 둘의 상관 {fd.corr(d[fd.index]):.2f}")
print(f"  → 선물 차이의 분산 중 지수 구성 차이로 설명되는 몫: {fd.corr(d[fd.index])**2*100:.0f}%")

# 시각을 맞춘 선물 비교: 매일 KRX300 선물의 15:20 이전 마지막 체결과, 같은 분의 코스피200 선물 가격
def M(n):
    x = pd.DataFrame(json.load(open(f"../data/{n}_1m.json"))); x.index = pd.to_datetime(x.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return x.close
mx, mk = M("krx300"), M("k200"); rows = {}
for day, g in mx.groupby(mx.index.normalize()):
    if day == pd.Timestamp("2026-11-19"): g = g[g.index.strftime("%H:%M") <= "16:20"]
    else: g = g[g.index.strftime("%H:%M") <= "15:20"]
    kk = mk[:g.index[-1]]
    if len(g) and len(kk) and kk.index[-1].normalize() == day: rows[day] = (g.iloc[-1], kk.iloc[-1])
S = pd.DataFrame(rows, index=["x", "k"]).T.pct_change().dropna(); sd = (S.x - S.k).reindex(d.index).dropna()
print(f"  시각 맞춘 선물 차이 (같은 분의 가격끼리, {len(sd)}일): σ {sd.std()*1e4:.0f}bp | 지수 차이와 상관 {sd.corr(d[sd.index]):.2f} → 지수 구성으로 설명되는 몫 {sd.corr(d[sd.index])**2*100:.0f}%")
print(f"  선물 차이에서 지수 차이를 뺀 나머지(선물 시장 고유) σ {(sd-d[sd.index]).std()*1e4:.0f}bp")
