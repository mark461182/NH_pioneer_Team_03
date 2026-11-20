"""시가≈종가인 날과 크게 움직인 날을 장 초반에 구분할 징후가 있는지.
  목표 = 초반 창이 끝난 시점부터 종가까지 움직인 크기 |종가/그 시점 가격 − 1| (방향 무시, 초반 움직임 자체는 빼고 봄)
  [1] KODEX 200 60분봉 약 2년 (초반 창 = 첫 1시간)   [2] KODEX 200 5분봉 60일 (초반 창 = 첫 15분·30분)   [3] 선물 1분봉 45일 (KRX300 고유 징후)
실행: python3 early_signs.py   (data/kodex200_60m.csv, kodex200_5m.csv 는 intraday_scan.py 가 받음)"""
import json, pandas as pd, numpy as np
pd.set_option("display.width", 200)
def days(D, cut):
    R = {}; prev = None
    for d, g in D.groupby(D.index.normalize()):
        e = g[g.index.strftime("%H:%M") < cut]; l = g[g.index.strftime("%H:%M") >= cut]
        if len(e) and len(l) and prev is not None:
            p = e.Close.iloc[-1]
            R[d] = {"갭 크기": abs(g.Open.iloc[0] / prev["c"] - 1) * 100, "초반 움직임 크기": abs(p / g.Open.iloc[0] - 1) * 100, "초반 고저 범위": (e.High.max() / e.Low.min() - 1) * 100, "초반 거래량": e.Volume.sum(),
                    "전일 고저 범위": prev["rng"], "전일 움직임 크기": prev["absr"], "이후 움직임": abs(g.Close.iloc[-1] / p - 1) * 100, "이후 고저 범위": (l.High.max() / l.Low.min() - 1) * 100, "시가→종가": abs(g.Close.iloc[-1] / g.Open.iloc[0] - 1) * 100,
                    "초반 방향 유지": np.sign(p / g.Open.iloc[0] - 1) * (g.Close.iloc[-1] / p - 1) * 100}
        prev = {"c": g.Close.iloc[-1], "rng": (g.High.max() / g.Low.min() - 1) * 100, "absr": abs(g.Close.iloc[-1] / g.Open.iloc[0] - 1) * 100}
    T = pd.DataFrame(R).T; T["초반 거래량(20일 평균 대비)"] = T["초반 거래량"] / T["초반 거래량"].rolling(20).mean().shift(1); T["최근 20일 평균 범위"] = T["전일 고저 범위"].rolling(20).mean(); return T.drop(columns="초반 거래량").dropna()
FE = ["초반 고저 범위", "초반 움직임 크기", "갭 크기", "초반 거래량(20일 평균 대비)", "전일 고저 범위", "전일 움직임 크기", "최근 20일 평균 범위"]
def report(T, title):
    print(f"\n{title}  (날 수 {len(T)})  이후 움직임 중앙 {T['이후 움직임'].median():.2f}%, 상위 1/4 경계 {T['이후 움직임'].quantile(.75):.2f}%")
    big = T["이후 움직임"] >= T["이후 움직임"].quantile(.75); small = T["이후 움직임"] <= T["이후 움직임"].quantile(.25)
    print(f"  {'징후':28s} {'순위상관':>7s} {'앞 절반':>7s} {'뒤 절반':>7s} | 징후 하위 1/3 → 상위 1/3 일 때:  이후 움직임 중앙(%)      크게 움직인 날 비율      거의 제자리인 날 비율")
    h = len(T) // 2
    for f in FE:
        r = T[f].rank().corr(T["이후 움직임"].rank()); r1 = T[f][:h].rank().corr(T["이후 움직임"][:h].rank()); r2 = T[f][h:].rank().corr(T["이후 움직임"][h:].rank())
        lo, hi = T[f] <= T[f].quantile(1 / 3), T[f] >= T[f].quantile(2 / 3)
        print(f"  {f:28s} {r:+7.2f} {r1:+7.2f} {r2:+7.2f} |                         {T['이후 움직임'][lo].median():.2f} → {T['이후 움직임'][hi].median():.2f}            {big[lo].mean()*100:3.0f}% → {big[hi].mean()*100:3.0f}%             {small[lo].mean()*100:3.0f}% → {small[hi].mean()*100:3.0f}%")
    X = np.c_[np.ones(len(T)), np.log(T[["초반 고저 범위", "최근 20일 평균 범위"]].clip(lower=0.01))]; y = np.log(T["이후 고저 범위"].clip(lower=0.01)); b = np.linalg.lstsq(X, y, rcond=None)[0]
    X1 = np.c_[np.ones(len(T)), np.log(T[["최근 20일 평균 범위"]])]; b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    print(f"  이후 고저 범위를 설명하는 정도(R²): 최근 20일 평균만 {1-(y-X1@b1).var()/y.var():.2f} → 초반 고저 범위를 더하면 {1-(y-X@b).var()/y.var():.2f}")
    print(f"  방향: 초반 방향이 그 뒤에도 이어진 날 {(T['초반 방향 유지']>0).mean()*100:.0f}%, 평균 {T['초반 방향 유지'].mean():+.3f}% | 초반 범위 상위 1/3 인 날만: {(T['초반 방향 유지'][T['초반 고저 범위']>=T['초반 고저 범위'].quantile(2/3)]>0).mean()*100:.0f}%")
H = pd.read_csv("data/kodex200_60m.csv", index_col=0, parse_dates=True); F = pd.read_csv("data/kodex200_5m.csv", index_col=0, parse_dates=True)
report(days(H, "10:00"), "[1] KODEX 200 60분봉, 초반 = 09:00~10:00, 이후 = 10:00~종가")
report(days(F, "09:15"), "[2a] KODEX 200 5분봉, 초반 = 첫 15분"); report(days(F, "09:30"), "[2b] KODEX 200 5분봉, 초반 = 첫 30분")
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); R = {}; prev = None
for d in sorted({t for t in x.index.normalize()}):
    xd = x[x.index.normalize() == d]; kd = k[k.index.normalize() == d]; hm = lambda g: g.index.strftime("%H:%M"); J = xd.join(kd, how="inner", rsuffix="_k")
    late = J[(hm(J) >= "15:00") & (hm(J) <= "15:19")]; r_today = np.log(late.close / late.close_k).median() if len(late) >= 5 else None
    xe = xd[hm(xd) < "09:15"]; ke = kd[hm(kd) < "09:15"]; kl = kd[(hm(kd) >= "09:15") & (hm(kd) <= "15:19")]
    if prev is not None and len(xe) and len(kl):
        R[d] = {"KRX300 첫 체결 괴리 크기(bp)": abs(xd.open.iloc[0] / (np.exp(prev) * kd[:xd.index[0]].close.iloc[-1]) - 1) * 1e4, "KRX300 초반 거래량(~09:15)": xe.volume.sum(), "KRX300 초반 체결 분 수": len(xe), "KRX300 첫 체결 시각(분)": (xd.index[0] - d).seconds / 60 - 525,
                "코스피200선물 초반 범위(%)": (ke.high.max() / ke.low.min() - 1) * 100, "코스피200선물 초반 거래량": ke.volume.sum(), "이후": abs(kl.close.iloc[-1] / kl.open.iloc[0] - 1) * 100}
    if r_today is not None: prev = r_today
T = pd.DataFrame(R).T.dropna(); print(f"\n[3] 선물 1분봉 {len(T)}일 — 초반(~09:15) 징후와 '09:15~15:19 코스피200 선물 움직임 크기'의 순위상관  (45일이라 ±0.30 안쪽은 우연 범위)")
for c in T.columns[:-1]: print(f"  {c:30s} {T[c].rank().corr(T['이후'].rank()):+.2f}")
