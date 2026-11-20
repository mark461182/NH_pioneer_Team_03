"""KRX300 선물보다 먼저 움직이는 것이 있는지 — 1분 시차 분석.
[1] 국내 선물(46일)  [2] ES·NQ(10/26~11/13)  [3] 야후 현물·지수·환율(11/11~11/18, 6일)
KRX300 은 체결이 드물어 '직전가'로 채우면 뭐든 앞서 보이므로, 목표를 둘로 나눠 봄:
  (가) 코스피200 선물의 다음 분 수익률 (KRX300 적정가의 95%)   (나) KRX300 을 다음 분 첫 체결가에 사서 h분 뒤 체결가까지
실행: python3 leadlag.py"""
import json, pandas as pd, numpy as np
pd.set_option("display.width", 220)
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
def LO(n):
    R = json.load(open(f"../data/{n}_1m.json")); t = pd.to_datetime([r["occurs1_ref_dt"] + r["occurs1_exch_tm"].zfill(6) for r in R], format="%Y%m%d%H%M%S")
    return pd.DataFrame({"close": [float(r["occurs1_close_pric"]) for r in R]}, index=t).sort_index()
x = L("krx300"); NH = {"k200": L("k200").close, "kq150": L("kq150").close, "삼성전자선물": L("sec_f").close, "하이닉스선물": L("hyn_f").close}
OV = {"ES": LO("es").close, "NQ": LO("nq").close}
Y = pd.read_csv("data/yahoo_1m_close.csv", index_col=0, parse_dates=True); Y = Y.rename(columns={"sec": "삼성전자", "hyn": "SK하이닉스", "sksq": "SK스퀘어", "kodex200": "KODEX200", "kodex_kq150": "KODEX코스닥150", "kodex_semi": "KODEX반도체", "usdkrw": "달러원", "kospi": "코스피지수", "kosdaq": "코스닥지수", "nikkei": "닛케이", "nq_y": "NQ(야후)", "es_y": "ES(야후)"})
def panel(series, days, a="09:01", b="15:19", limit=3):
    out = []
    for d in days:
        grid = pd.date_range(f"{d} {a}", f"{d} {b}", freq="1min"); g = pd.DataFrame(index=grid)
        for n, s in series.items():
            sd = s[s.index.normalize() == pd.Timestamp(d)]
            g[n] = np.log(sd.reindex(sd.index.union(grid)).ffill(limit=limit).reindex(grid)).diff()
        xd = x[x.index.normalize() == pd.Timestamp(d)]; g["x_open_next"] = xd.open.reindex(grid).shift(-1); g["x_last"] = xd.close.reindex(xd.index.union(grid)).ffill().reindex(grid)
        for h in (3, 10): g[f"x_fwd{h}"] = np.log(g.x_last.shift(-h) / g.x_open_next)
        g["k_fwd1"] = g["k200"].shift(-1); g = g.replace([np.inf, -np.inf], np.nan); num = g.columns.difference(["x_open_next", "x_last"]); g[num] = g[num] * 1e4; g["day"] = d; out.append(g.iloc[1:])
    return pd.concat(out)
def report(P, names, title):
    print(f"\n{title}  (분 수 {len(P)}, 일수 {P.day.nunique()})")
    print(f"  {'후보':14s} {'동시 상관':>8s} | corr(후보 r_t, 코스피200 r_t+k):  {'k=-2':>6s} {'-1':>6s} {'+1':>6s} {'+2':>6s} {'+3':>6s} | 코스피200 다음 분 예측(자기 수익률 통제) 계수·t | KRX300 다음 체결가 진입 3분·10분 예측 t")
    for n in names:
        q = P[list(dict.fromkeys([n, "k200", "day"]))].dropna(); cc = {}
        for kk in (-2, -1, 1, 2, 3):
            v = pd.concat([pd.DataFrame({"a": g[n], "b": g["k200"].shift(-kk)}) for _, g in q.groupby("day")]).dropna(); cc[kk] = v.a.corr(v.b)
        z = P[list(dict.fromkeys([n, "k200", "k_fwd1"]))].dropna(); X = np.c_[np.ones(len(z)), z["k200"], z[n]] if n != "k200" else np.c_[np.ones(len(z)), z["k200"]]
        b, res = np.linalg.lstsq(X, z.k_fwd1, rcond=None)[:2]; e = z.k_fwd1 - X @ b; se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * e.var()); tk = b[-1] / se[-1]
        ts = []
        for h in (3, 10):
            w = P[list(dict.fromkeys([n, "k200", f"x_fwd{h}"]))].dropna(); X2 = np.c_[np.ones(len(w)), w["k200"], w[n]] if n != "k200" else np.c_[np.ones(len(w)), w["k200"]]
            b2 = np.linalg.lstsq(X2, w[f"x_fwd{h}"], rcond=None)[0]; e2 = w[f"x_fwd{h}"] - X2 @ b2; se2 = np.sqrt(np.diag(np.linalg.pinv(X2.T @ X2)) * e2.var()); ts.append((b2[-1], b2[-1] / se2[-1], len(w)))
        print(f"  {n:14s} {q[n].corr(q['k200']):8.2f} |                                 {cc[-2]:+6.3f} {cc[-1]:+6.3f} {cc[1]:+6.3f} {cc[2]:+6.3f} {cc[3]:+6.3f} | {b[-1]:+7.3f}  t {tk:+5.1f} | 3분 계수 {ts[0][0]:+6.3f} t {ts[0][1]:+5.1f} · 10분 계수 {ts[1][0]:+6.3f} t {ts[1][1]:+5.1f} (n={ts[0][2]})")
days = sorted({str(d) for d in x.index.date})
P1 = panel(NH, days); report(P1, ["k200", "kq150", "삼성전자선물", "하이닉스선물"], "[1] 국내 선물 9/11~11/18")
d2 = [d for d in days if "2026-10-26" <= d <= "2026-11-13"]; P2 = panel({**NH, **OV}, d2); report(P2, ["k200", "ES", "NQ"], "[2] 미국 지수선물 (한국 낮 시간, 10/26~11/13)")
d3 = [d for d in days if d >= "2026-11-11"]; ys = {c: Y[c].dropna() for c in Y.columns}; P3 = panel({**NH, **ys}, d3, b="14:58")
al = {k_: pd.concat([pd.DataFrame({"a": g["KODEX200"], "b": g["k200"].shift(-k_)}) for _, g in P3.groupby("day")]).dropna().corr().iloc[0, 1] for k_ in (-1, 0, 1)}
print("\n야후 시각 정렬 확인: corr(KODEX200 r_t, 코스피200선물 r_t+k) k=-1,0,+1 →", {k_: round(v, 2) for k_, v in al.items()})
report(P3, ["k200", "삼성전자", "SK하이닉스", "SK스퀘어", "KODEX200", "KODEX반도체", "KODEX코스닥150", "코스피지수", "코스닥지수", "달러원", "닛케이", "NQ(야후)", "ES(야후)"], "[3] 현물·지수·환율 (야후, 11/11~11/18)")
print("\n읽는 법: k=+1 열이 크면 후보가 1분 먼저. k=-1 열이 크면 코스피200 선물이 먼저. t 는 |2| 넘어야 의미.")

# ── ES·NQ 가 크게 움직인 분 뒤: KRX300 을 다음 분 첫 체결가에 같은 방향으로 진입했을 때 (bp, 1틱=0.59bp)
print("\n[4] ES·NQ 가 한 분에 크게 움직인 뒤 (10/26~11/13, 15일)")
print("  한국 낮 시간 1분 표준편차(bp): ES", round(P2.ES.std(), 1), "NQ", round(P2.NQ.std(), 1), "코스피200", round(P2.k200.std(), 1))
P2["us"] = (P2.ES + P2.NQ) / 2; P2["k_fwd3"] = sum(P2.groupby("day").k200.shift(-i) for i in (1, 2, 3))
for q_ in (0.80, 0.90, 0.95, 0.98):
    th = P2.us.abs().quantile(q_); s = P2[(P2.us.abs() >= th)].dropna(subset=["x_fwd3"]); sg = np.sign(s.us); p3 = s.x_fwd3 * sg; p10 = s.x_fwd10 * sg
    print(f"  |ES·NQ 평균|≥{th:4.1f}bp (상위 {100-q_*100:.0f}%): 진입 가능 {len(s):4d}건 (하루 {len(s)/15:4.1f}) | 그 분 코스피200 이미 {((s.k200)*sg).mean():+5.1f}bp | KRX300 3분 {p3.mean():+5.2f}bp (t {p3.mean()/p3.std()*np.sqrt(len(s)):+4.1f}, 승률 {(p3>0).mean()*100:.0f}%) · 10분 {p10.mean():+5.2f}bp (t {p10.mean()/p10.std()*np.sqrt(len(s.dropna(subset=['x_fwd10']))):+4.1f}) | 코스피200 이후 3분 {(s.k_fwd3*sg).mean():+5.2f}bp")
half = sorted(P2.day.unique()); a_, b_ = half[:8], half[8:]
for nm, dd in (("앞 8일", a_), ("뒤 7일", b_)):
    s = P2[P2.day.isin(dd) & (P2.us.abs() >= P2.us.abs().quantile(.9))].dropna(subset=["x_fwd3"]); p3 = s.x_fwd3 * np.sign(s.us); print(f"  상위 10% 신호를 {nm}로 나누면: {len(s)}건, KRX300 3분 {p3.mean():+.2f}bp, t {p3.mean()/p3.std()*np.sqrt(len(s)):+.1f}")
