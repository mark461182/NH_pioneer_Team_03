"""당일 매매 규칙 훑기 — KODEX 200 (069500) 야후 60분봉 약 2년 + 5분봉 60일.
'앞 구간 수익률의 부호'로 '뒤 구간'에 진입(같은 방향=추종 기준)했을 때의 평균 손익. 음수면 반대로 하면 그만큼 플러스.
실행: ~/nh_pioneer/.venv/bin/python intraday_scan.py"""
import pandas as pd, numpy as np, yfinance as yf, warnings; warnings.filterwarnings("ignore")
def get(iv, per):
    d = yf.download("069500.KS", period=per, interval=iv, progress=False, auto_adjust=False); d.columns = [c[0] for c in d.columns]
    d.index = d.index.tz_convert("Asia/Seoul").tz_localize(None); return d[d.Volume > 0]
H = get("60m", "730d"); F = get("5m", "60d"); H.to_csv("data/kodex200_60m.csv"); F.to_csv("data/kodex200_5m.csv")
print("60분봉", H.index[0].date(), "~", H.index[-1].date(), "일수", H.index.normalize().nunique(), "| 봉 시각:", sorted(set(H.index.strftime("%H:%M")))[:9])
print("5분봉 ", F.index[0].date(), "~", F.index[-1].date(), "일수", F.index.normalize().nunique())
def day_table(D, cuts):
    rows = {}; prev_close = None
    for d, g in D.groupby(D.index.normalize()):
        if len(g) < 5: prev_close = g.Close.iloc[-1]; continue
        r = {"open": g.Open.iloc[0], "close": g.Close.iloc[-1], "prev": prev_close}
        for c in cuts:
            s = g[g.index.strftime("%H:%M") < c]; r[c] = s.Close.iloc[-1] if len(s) else np.nan
        rows[d] = r; prev_close = g.Close.iloc[-1]
    return pd.DataFrame(rows).T.astype(float).dropna()
def test(T, name, a0, a1, b0, b1, cost_bp=1.5):
    sig = np.log(T[a1] / T[a0]); fwd = np.log(T[b1] / T[b0]) * 1e4; p = np.sign(sig) * fwd; p = p[sig != 0]
    h = len(p) // 2; t = p.mean() / p.std() * np.sqrt(len(p))
    big = p[sig.abs()[p.index] >= sig.abs().quantile(.5)]
    print(f"  {name:38s} n={len(p):4d} 상관 {sig.corr(fwd):+.2f} | 추종 평균 {p.mean():+6.1f}bp 승률 {(p>0).mean()*100:3.0f}% t {t:+5.1f} | 앞 절반 {p[:h].mean():+6.1f} 뒤 절반 {p[h:].mean():+6.1f} | 신호 큰 절반만 {big.mean():+6.1f}bp (t {big.mean()/big.std()*np.sqrt(len(big)):+4.1f})")
    return p
print("\n[60분봉, 약 2년]  (KODEX200 왕복 비용 대략 1~2bp, KRX300 선물 왕복 2틱 = 1.2bp + 수수료)")
T = day_table(H, ["10:00", "11:00", "13:00", "14:00", "15:00"]); print("  날 수", len(T), "| 하루 시가→종가 표준편차", round(float((np.log(T.close / T.open)).std() * 1e4)), "bp, 갭 표준편차", round(float((np.log(T.open / T.prev)).std() * 1e4)), "bp")
R = {}
R["gap_oc"] = test(T, "밤사이 갭 → 시가~종가", "prev", "open", "open", "close")
R["gap_h1"] = test(T, "밤사이 갭 → 첫 1시간(09~10시)", "prev", "open", "open", "10:00")
R["gap_rest"] = test(T, "밤사이 갭 → 10시~종가", "prev", "open", "10:00", "close")
R["h1_rest"] = test(T, "첫 1시간 → 10시~종가", "open", "10:00", "10:00", "close")
R["h1_last"] = test(T, "첫 1시간 → 마지막 구간(15시~종가)", "open", "10:00", "15:00", "close")
R["am_pm"] = test(T, "오전(시가~13시) → 13시~종가", "open", "13:00", "13:00", "close")
R["to14_last"] = test(T, "시가~14시 → 14시~종가", "open", "14:00", "14:00", "close")
R["to15_last"] = test(T, "시가~15시 → 15시~종가", "open", "15:00", "15:00", "close")
R["pc10_rest"] = test(T, "전일종가~10시 → 10시~종가", "prev", "10:00", "10:00", "close")
R["pc15_last"] = test(T, "전일종가~15시 → 15시~종가", "prev", "15:00", "15:00", "close")
T["pd"] = T.close.shift(1); T["pp"] = T.prev.shift(1); T2 = T.dropna()
R["prev_oc"] = test(T2, "전일 하루 수익률 → 오늘 시가~종가", "pp", "pd", "open", "close")
print("\n[5분봉, 60일]")
T5 = day_table(F, ["09:15", "09:30", "10:00", "11:00", "13:00", "14:30", "15:00"])
test(T5, "첫 15분 → 09:15~종가", "open", "09:15", "09:15", "close"); test(T5, "첫 30분 → 09:30~종가", "open", "09:30", "09:30", "close"); test(T5, "첫 30분 → 09:30~11:00", "open", "09:30", "09:30", "11:00")
test(T5, "밤사이 갭 → 첫 30분", "prev", "open", "open", "09:30"); test(T5, "밤사이 갭 → 시가~종가", "prev", "open", "open", "close"); test(T5, "전일종가~14:30 → 14:30~종가", "prev", "14:30", "14:30", "close"); test(T5, "첫 30분 → 마지막 30분(15:00~)", "open", "09:30", "15:00", "close")
pd.DataFrame(R).to_csv("data/intraday_scan_60m.csv")
