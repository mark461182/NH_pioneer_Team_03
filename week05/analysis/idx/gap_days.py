"""그래프에서 두 선이 벌어진 날이 '삼성전자·SK하이닉스 외 종목이 따로 움직인 날'인지 확인.
  선물 쪽: 하루 중 (KRX300 선물 % − 코스피200 선물 %) — 첫 공통 체결 분 기준, 그래프와 같은 정의
  지수 쪽: 구성 종목의 시가→종가 수익률로 만든 (KRX300 − 코스피200) 장중 차이와 그 분해
실행: ~/nh_pioneer/.venv/bin/python gap_days.py"""
import json, pandas as pd, numpy as np, yfinance as yf, warnings, os; warnings.filterwarnings("ignore")
T = pd.read_csv("active.csv", dtype={"code": str}).set_index("code"); mkt = pd.read_csv("mkt.csv", index_col=0, dtype=str).iloc[:, 0]; mkt.index = mkt.index.astype(str).str.zfill(6)
if not os.path.exists("ohlc_open.csv"):
    tk = [c + "." + mkt.get(c, "KS") for c in T.index]; d = yf.download(tk, start="2026-09-01", end="2026-11-20", progress=False, auto_adjust=True)
    o = d["Open"]; c = d["Close"]; o.columns = [x.split(".")[0] for x in o.columns]; c.columns = [x.split(".")[0] for x in c.columns]; o.to_csv("ohlc_open.csv"); c.to_csv("ohlc_close.csv")
O = pd.read_csv("ohlc_open.csv", index_col=0, parse_dates=True); C = pd.read_csv("ohlc_close.csv", index_col=0, parse_dates=True); O.columns = O.columns.astype(str).str.zfill(6); C.columns = C.columns.astype(str).str.zfill(6)
cols = [c for c in T.index if c in C.columns]; O, C = O[cols].loc["2026-09-11":"2026-11-18"], C[cols].loc["2026-09-11":"2026-11-18"]; T = T.loc[cols]
ri = (C / O - 1)                                                # 종목별 장중(시가→종가) 수익률
def W(col): sh = T[col] / C.iloc[-1]; v = O * sh; return v.div(v.sum(axis=1), axis=0)
W3, W2 = W("w300"), W("w200"); A = (W3 - W2) * ri
G = T.grp.copy(); G[["005930", "000660"]] = "반도체 2종(삼성전자·SK하이닉스)"; G = G.replace({"공통": "공통 나머지", "KRX300에만(코스닥150)": "KRX300에만", "KRX300에만(그 외)": "KRX300에만", "코스피200에만": "공통 나머지"})
Cg = A.T.groupby(G).sum().T * 1e4; idx_diff = A.sum(axis=1) * 1e4
only = T.index[T.w200 == 0]; r_only = (W3[only].div(W3[only].sum(axis=1), axis=0) * ri[only]).sum(axis=1) * 100; r_k200 = (W2 * ri).sum(axis=1) * 100
semi = ri[["005930", "000660"]].mean(axis=1) * 100; rest_common = T.index[(T.w200 > 0) & ~T.index.isin(["005930", "000660"])]; r_rest = (W2[rest_common].div(W2[rest_common].sum(axis=1), axis=0) * ri[rest_common]).sum(axis=1) * 100
def L(n):
    d = pd.DataFrame(json.load(open(f"../data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); rows = {}
for d, J in x.join(k, how="inner", rsuffix="_k").groupby(lambda t: t.normalize()):
    df = ((J.close / J.close.iloc[0]) - (J.close_k / J.close_k.iloc[0])) * 1e4
    rows[d] = {"끝 간격": df.iloc[-15:].median(), "평균|간격|": df.abs().mean(), "흔들림": df.diff().std(), "최대|간격|": df.abs().max(), "거래량": J.volume.sum(), "k200범위": (J.high_k.max() / J.low_k.min() - 1) * 100}
F = pd.DataFrame(rows).T; F.index = pd.to_datetime(F.index)
M = F.join(pd.DataFrame({"지수 차이": idx_diff, "KRX300에만 기여": Cg["KRX300에만"], "반도체 기여": Cg["반도체 2종(삼성전자·SK하이닉스)"], "공통 나머지 기여": Cg["공통 나머지"], "125종목−코스피200(%p)": r_only - r_k200, "반도체 평균%": semi, "공통 나머지%": r_rest, "125종목%": r_only}), how="inner").dropna()
print("날 수", len(M)); pd.set_option("display.width", 220)
print("\n[1] 선물 그래프의 '끝 간격'(bp, 장 끝 15분 중앙값) 과 지수 쪽 숫자의 상관")
for c in ("지수 차이", "KRX300에만 기여", "반도체 기여", "공통 나머지 기여", "125종목−코스피200(%p)"): print(f"  끝 간격 ~ {c:22s} 상관 {M['끝 간격'].corr(M[c]):+.2f}")
b = np.polyfit(M["지수 차이"], M["끝 간격"], 1); res = M["끝 간격"] - np.polyval(b, M["지수 차이"])
print(f"  끝 간격 = {b[0]:.2f} × 지수 차이 {b[1]:+.1f}  | 끝 간격 표준편차 {M['끝 간격'].std():.1f}bp, 지수 차이 {M['지수 차이'].std():.1f}bp, 설명 안 되는 나머지 {res.std():.1f}bp")
print("\n[2] 지수 차이(장중)의 분해: 분산 중 몫 —", {c: f"{Cg[c].cov(idx_diff)/idx_diff.var()*100:.0f}%" for c in Cg.columns})
print("\n[3] 그래프가 '안 겹쳐 보이는 정도'(평균|간격|) 와의 상관")
for c in ("|지수 차이|", "거래량", "k200범위", "흔들림"):
    v = M["지수 차이"].abs() if c == "|지수 차이|" else M[c]; print(f"  평균|간격| ~ {c:10s} 상관 {M['평균|간격|'].corr(v):+.2f}")
print("\n[4] 평균|간격| 큰 날 10개")
S = M.sort_values("평균|간격|", ascending=False).head(10)[["평균|간격|", "끝 간격", "지수 차이", "KRX300에만 기여", "반도체 기여", "반도체 평균%", "공통 나머지%", "125종목%", "거래량", "k200범위"]]; S.index = S.index.strftime("%m-%d"); print(S.round(1).to_string())
print("\n[5] 평균|간격| 작은 날 5개"); S = M.sort_values("평균|간격|").head(5)[["평균|간격|", "끝 간격", "지수 차이", "KRX300에만 기여", "반도체 평균%", "125종목%", "거래량", "k200범위"]]; S.index = S.index.strftime("%m-%d"); print(S.round(1).to_string())
