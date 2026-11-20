"""구성 종목 일별 종가(야후) 내려받기 → px.csv. 실행: ~/nh_pioneer/.venv/bin/python px.py"""
import pandas as pd, yfinance as yf
T = pd.read_csv("active.csv", dtype={"code": str}).set_index("code"); q = pd.read_csv("kq150_w.csv", dtype={"code": str}).code.tolist()
def dl(tk):
    d = yf.download(tk, start="2026-06-01", end="2026-11-20", progress=False, auto_adjust=True, threads=True)["Close"]
    return d.dropna(axis=1, how="all")
ks = [c for c in T.index if T.w200[c] > 0]; kq = [c for c in T.index if T.w200[c] == 0 and c in q]; un = [c for c in T.index if c not in ks and c not in kq]
A = dl([c + ".KS" for c in ks + un]); B = dl([c + ".KQ" for c in kq + un] + ["069500.KS", "292190.KS", "229200.KS"])
P = pd.concat([A, B.drop(columns=[c for c in B.columns if c.replace(".KQ", ".KS") in A.columns and c.endswith(".KQ")])], axis=1)
mkt = {c.split(".")[0]: c.split(".")[1] for c in P.columns}; P.columns = [c.split(".")[0] for c in P.columns]; P = P.loc[:, ~P.columns.duplicated()]
P.to_csv("px.csv"); pd.Series(mkt).to_csv("mkt.csv")
miss = [c for c in T.index if c not in P.columns]
print("받은 종목", P.shape, "| 못 받은 종목", len(miss), [(c, T.name[c], round(T.w300[c], 2), round(T.w200[c], 2)) for c in miss][:12], "| 못 받은 비중 KRX300", round(T.w300[miss].sum(), 2), "코스피200", round(T.w200[miss].sum(), 2))
print("그 외 종목 시장:", pd.Series({c: mkt.get(c) for c in un}).value_counts().to_dict()); print(P[["069500", "292190", "229200", "005930"]].tail(3))
