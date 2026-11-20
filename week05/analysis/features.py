"""KRX300 선물의 퀀트 관점 특징 모음 (9/11~11/18, 12월물). 실행: python3 features.py"""
import json, pandas as pd, numpy as np
pd.set_option("display.width", 220)
def L(n, f):
    d = pd.DataFrame(json.load(open(f"data/{n}_{f}.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.drop(columns="timestamp")
A, B = "2026-09-11", "2026-11-18 23:59"
x, k = L("krx300", "1m").loc[A:B], L("k200", "1m").loc[A:B]; dx, dk = L("krx300", "1d").loc[A:B], L("k200", "1d").loc[A:B]; dx, dk = dx[dx.volume > 0], dk[dk.volume > 0]
dx.index, dk.index = dx.index.normalize(), dk.index.normalize()
print("[A] 투자자별 (수량, 매수+매도 기준 점유율)")
I = pd.DataFrame(json.load(open("data/investor_daily.json"))); I = I[I.group != "total"]; I["tv"] = I.buy + I.sell; I["net"] = I.buy - I.sell
for prd, nm in (("08", "KRX300"), ("01", "코스피200"), ("05", "미니"), ("06", "코스닥150")):
    g = I[I.prd == prd]; sh = g.groupby("group").tv.sum() / g.tv.sum() * 100
    print(f"  {nm:7s} {g.date.nunique()}일:", {a: round(b, 1) for a, b in sh.sort_values(ascending=False).items() if b >= 0.3})
g = I[I.prd == "08"].pivot_table(index="date", columns="group", values=["net", "tv"], aggfunc="sum").fillna(0); net = g["net"]; tv = g["tv"]
print("  KRX300 일별 순매수(계약): 외국인 |순매수| 중앙", net.foreign.abs().median(), "최대", net.foreign.abs().max(), "| 증권·선물 중앙", net.securitiesFutures.abs().median(), "| 개인 중앙", net.individual.abs().median())
print("  외국인 순매수 ↔ 증권 순매수 상관", round(float(net.foreign.corr(net.securitiesFutures)), 2), "| 외국인 |순매수| / 외국인 거래량(매수+매도) 중앙", f"{(net.foreign.abs()/tv.foreign).median()*100:.1f}%", "| 누적 순매수(36일): 외국인", int(net.foreign.sum()), "증권", int(net.securitiesFutures.sum()), "개인", int(net.individual.sum()))
print("  외국인 점유율 일별 범위", f"{(tv.foreign/tv.sum(axis=1)).min()*100:.0f}~{(tv.foreign/tv.sum(axis=1)).max()*100:.0f}%", "| 9월 평균", f"{(tv.foreign/tv.sum(axis=1))[:'2026-09-30'].mean()*100:.0f}%", "10월", f"{(tv.foreign/tv.sum(axis=1))['2026-10-01':'2026-10-31'].mean()*100:.0f}%", "11월", f"{(tv.foreign/tv.sum(axis=1))['2026-11-01':].mean()*100:.0f}%")
print("\n[B] 거래량의 성격")
v = pd.DataFrame({"x": dx.volume, "k": dk.volume, "rng": (dk.high - dk.low) / dk.close, "absr": dk.close.pct_change().abs()}).dropna()
print("  일 거래량 추이(주별 평균 계약):", dx.volume.resample("W").mean().round(0).astype(int).tolist())
print("  KRX300 거래량 ↔ 코스피200 거래량 상관", round(float(v.x.corr(v.k)), 2), "| ↔ 코스피200 일중 범위", round(float(v.x.corr(v.rng)), 2), "| 요일별 평균:", dx.volume.groupby(dx.index.dayofweek).mean().round(0).astype(int).tolist())
J = x.join(k, how="outer", rsuffix="_k"); J = J[(J.index.strftime("%H:%M") >= "09:00") & (J.index.strftime("%H:%M") <= "15:19")]; J["volume"] = J.volume.fillna(0); J["kabs"] = ((J.high_k - J.low_k) / J.close_k * 1e4)
print("  분 단위: KRX300 거래량 ↔ 코스피200 그 분 범위 상관", round(float(J.volume.corr(J.kabs)), 2), "| ↔ 코스피200 거래량", round(float(J.volume.corr(J.volume_k)), 2))
qs = J.groupby(pd.qcut(J.kabs, 5, labels=["가장 잔잔", "2", "3", "4", "가장 요동"]), observed=True).agg(체결확률=("volume", lambda s: (s > 0).mean()), 평균계약=("volume", "mean")); print("  코스피200 1분 범위 5분위별 KRX300:", qs.round(2).to_dict())
vv = x.volume; print("  체결 있는 분의 계약수 분위: 25%", vv.quantile(.25), "50%", vv.quantile(.5), "75%", vv.quantile(.75), "95%", vv.quantile(.95), "99%", vv.quantile(.99), "최대", vv.max())
big = x[x.volume >= vv.quantile(.99)]; print("  상위 1% 분(≥", int(vv.quantile(.99)), "계약)의 시각:", big.index.strftime("%H:%M").value_counts().head(6).to_dict())
print("  하루 거래량 중 09:00~09:10 몫", f"{x[(x.index.strftime('%H:%M')>='09:00')&(x.index.strftime('%H:%M')<'09:10')].volume.sum()/x.volume.sum()*100:.1f}%", "| 코스피200 은", f"{k[(k.index.strftime('%H:%M')>='09:00')&(k.index.strftime('%H:%M')<'09:10')].volume.sum()/k.volume.sum()*100:.1f}%")
print("\n[C] 엉뚱한 가격 체결 (코스피200 선물의 그 분 고가·저가에 그날 비율을 곱한 범위를 벗어난 정도)")
out = []
for d, g in J.dropna(subset=["close", "close_k"]).groupby(lambda t: t.date()):
    ratio = (g.close / g.close_k).rolling(31, center=True, min_periods=8).median()
    hi, lo = g.high_k * ratio, g.low_k * ratio; ex = np.maximum(g.high - hi, lo - g.low).clip(lower=0) / g.close * 1e4
    out.append(pd.DataFrame({"ex": ex, "vol": g.volume, "side": np.where(g.high - hi > lo - g.low, "위", "아래")}))
E = pd.concat(out).dropna(); nd = len(out)
for th in (5, 10, 20, 50, 100):
    s = E[E.ex > th]; print(f"  {th:3d}bp 넘게 벗어난 분: {len(s):5d}건 (하루 {len(s)/nd:5.1f}건, 체결 분의 {len(s)/len(E)*100:4.1f}%), 그 분 거래량 중앙 {s.vol.median() if len(s) else 0}")
s = E[E.ex > 20]; print("  20bp 초과 건의 방향:", s.side.value_counts().to_dict(), "| 시각대:", s.index.strftime("%H").value_counts().sort_index().to_dict())
print("  일봉 고가·저가 비교: (KRX300 일중 범위 − 코스피200 일중 범위) %p 분포", ((dx.high - dx.low) / dx.close * 100 - (dk.high - dk.low) / dk.close * 100).describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_dict())
print("\n[D] 변동성·시간대")
def rv(df, a, b):
    g = df[(df.index.strftime("%H:%M") >= a) & (df.index.strftime("%H:%M") < b)]; r = np.log(g.close).groupby(g.index.date).diff().dropna(); return r.std() * 1e4
print("  코스피200 선물 1분 σ(bp) 시간대별:", {a: round(float(rv(k, a, b)), 1) for a, b in (("08:45", "09:00"), ("09:00", "09:30"), ("09:30", "11:00"), ("11:00", "13:30"), ("13:30", "15:20"), ("15:20", "15:46"))})
oc = pd.DataFrame({"x_gap": dx.open / dx.close.shift(1) - 1, "k_gap": dk.open / dk.close.shift(1) - 1, "x_id": dx.close / dx.open - 1, "k_id": dk.close / dk.open - 1}).dropna()
print("  밤사이 갭 σ: KRX300", f"{oc.x_gap.std()*100:.2f}%", "코스피200", f"{oc.k_gap.std()*100:.2f}%", "| 장중(시가→종가) σ: KRX300", f"{oc.x_id.std()*100:.2f}%", "코스피200", f"{oc.k_id.std()*100:.2f}%", "| 갭이 하루 분산에서 차지하는 몫: KRX300", f"{oc.x_gap.var()/(oc.x_gap.var()+oc.x_id.var())*100:.0f}%")
fo = x.groupby(x.index.date).apply(lambda g: g.index[0]); k_at = pd.Series({d: k[:t].close.iloc[-1] for d, t in fo.items() if len(k[:t])}); x_open = x.groupby(x.index.date).open.first()
gap_k = pd.Series({d: k[k.index.date == d].open.iloc[0] for d in fo.index if (k.index.date == d).any()})
print("  KRX300 첫 체결 시각 분포:", fo.apply(lambda t: t.strftime("%H:%M")).value_counts().sort_index().head(8).to_dict(), "…  | 08:45 정각 첫 체결", int((fo.apply(lambda t: t.strftime('%H:%M')) == "08:45").sum()), "/", len(fo), "일")
print("  첫 체결까지 코스피200 이 이미 움직인 폭 |%|: 중앙", f"{((k_at/gap_k-1).abs()*100).median():.2f}", "최대", f"{((k_at/gap_k-1).abs()*100).max():.2f}")
print("\n[E] 가격 눈금")
px = pd.concat([x.open, x.high, x.low, x.close]); last = (px * 10 % 10).round().astype(int)
print("  체결가 소수 첫째 자리 분포(%):", (last.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(), "| 정수(.0) 체결", f"{(last==0).mean()*100:.1f}% (고르면 20%)")
print("  가격대: 기간 최고", dx.high.max(), "최저", dx.low.min(), "| 1틱 bp:", round(0.2 / dx.close.max() * 1e4, 2), "~", round(0.2 / dx.close.min() * 1e4, 2), "| 가격제한폭 ±8%(1단계) =", round(dx.close.iloc[-1] * 0.08 / 0.2), "틱")
print("\n[F] 만기·월물")
print("  12월물 거래 시작: 거래량>0 첫날", L("krx300", "1d").query("volume>0").index[0].date(), "| 그 전 9월물 만기 2026-09-10 | 3월물(A0873000) 거래량 0, 미결제 0, 스프레드 종목(D086C73S) 호가 없음")
print("  → 월물 교체는 만기 직전 며칠에 몰릴 것으로 보이나 9월 교체 구간 데이터 없음(만기 지난 월물은 조회 안 됨)")
