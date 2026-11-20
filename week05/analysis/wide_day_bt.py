"""첫 1시간(09~10시) 범위가 넓은 날, 10시 이후 전략 백테스트.
  넓은 날 = 첫 1시간 고저 범위 ÷ (직전 20거래일 첫 1시간 범위의 중앙값) ≥ k   — 10시에 알 수 있는 값만 사용
  A 추종: 10시에 첫 1시간 방향으로 진입 → 종가 청산      B 역추종: 반대로
  C 돌파: 10시 이후 첫 1시간 고가(저가)를 넘으면 그 가격에 진입 → 종가 청산 (같은 봉에서 양쪽 다 넘으면 손실 쪽으로 계산)
  D 돌파+손절: C 에 첫 1시간 범위 중간값 손절 추가
  비용 왕복 3bp. [1] KODEX 200 60분봉 (최근 250/125/62일)  [2] 5분봉 60일  [3] 선물 1분봉 45일, KRX300 선물 체결가로
실행: python3 wide_day_bt.py"""
import json, pandas as pd, numpy as np
pd.set_option("display.width", 220); COST = 3.0
def build(D, cut="10:00"):
    R = {}
    for d, g in D.groupby(D.index.normalize()):
        e = g[g.index.strftime("%H:%M") < cut]; l = g[g.index.strftime("%H:%M") >= cut]
        if len(e) == 0 or len(l) < 2: continue
        hi, lo, o, p, c = e.High.max(), e.Low.min(), e.Open.iloc[0], l.Open.iloc[0], l.Close.iloc[-1]; mid = (hi + lo) / 2
        up = l.High > hi; dn = l.Low < lo; brk = stop = 0.0; side = 0
        for i in range(len(l)):
            if side == 0:
                if up.iloc[i] and dn.iloc[i]: side = 9; brk = -(hi / lo - 1) * 1e4; stop = brk; break          # 한 봉에서 양쪽 → 최악 가정
                if up.iloc[i]: side = 1; ent = hi; rest = l.iloc[i:]
                elif dn.iloc[i]: side = -1; ent = lo; rest = l.iloc[i:]
                if side in (1, -1):
                    brk = side * (c / ent - 1) * 1e4
                    hit = (rest.Low.iloc[1:] < mid) if side == 1 else (rest.High.iloc[1:] > mid)          # 진입 봉 이후 봉에서 손절 판정
                    stop = side * (mid / ent - 1) * 1e4 if hit.any() else brk
                    break
        R[d] = {"rng": (hi / lo - 1) * 100, "dir": np.sign(p / o - 1), "follow": np.sign(p / o - 1) * (c / p - 1) * 1e4, "brk": brk, "stop": stop, "brk_on": side != 0, "abs": abs(c / p - 1) * 1e4, "pos": (p - lo) / (hi - lo) if hi > lo else .5}
    T = pd.DataFrame(R).T.astype(float); T["rel"] = T.rng / T.rng.rolling(20).median().shift(1); return T.dropna()
def line(p, label, n_all):
    p = p.dropna()
    if len(p) < 5: return f"    {label:22s} 거래 {len(p)}"
    h = len(p) // 2
    return f"    {label:22s} 거래 {len(p):3d} ({len(p)/n_all*100:3.0f}%의 날) 평균 {p.mean():+7.1f}bp 중앙 {p.median():+7.1f} 승률 {(p>0).mean()*100:3.0f}% t {p.mean()/p.std()*np.sqrt(len(p)):+5.1f} 합 {p.sum()/100:+6.1f}% 최악 {p.min():+6.0f} | 앞/뒤 절반 평균 {p[:h].mean():+6.1f}/{p[h:].mean():+6.1f}"
def report(T, title):
    print(f"\n{title}  날 수 {len(T)}")
    for nm, m in (("넓은 날 (평소의 1.5배↑)", T.rel >= 1.5), ("넓은 날 (1.25배↑)", T.rel >= 1.25), ("넓은 날 (1.0배↑)", T.rel >= 1.0), ("좁은 날 (0.8배↓)", T.rel <= 0.8), ("모든 날", T.rel > 0)):
        S = T[m]; print(f"  {nm}: {len(S)}일, 10시 이후 움직임 크기 중앙 {S['abs'].median():.0f}bp")
        print(line(S.follow[S.dir != 0] - COST, "A 추종", len(S))); print(line(-S.follow[S.dir != 0] - COST, "B 역추종", len(S)))
        print(line(S.brk[S.brk_on == 1] - COST, "C 돌파", len(S))); print(line(S.stop[S.brk_on == 1] - COST, "D 돌파+중간값 손절", len(S)))
H = pd.read_csv("data/kodex200_60m.csv", index_col=0, parse_dates=True); T = build(H)
for nm, n in (("[1a] 60분봉 최근 250일", 250), ("[1b] 60분봉 최근 125일", 125), ("[1c] 60분봉 최근 62일", 62)): report(T.iloc[-n:], f"{nm} ({T.index[-n].date()}~)")
F = pd.read_csv("data/kodex200_5m.csv", index_col=0, parse_dates=True); report(build(F), "[2] 5분봉 60일 (돌파 판정이 5분 단위로 더 정확)")
def L(n):
    d = pd.DataFrame(json.load(open(f"data/{n}_1m.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300"), L("k200"); R = {}
for d in sorted({t for t in x.index.normalize()}):
    kd = k[k.index.normalize() == d]; xd = x[x.index.normalize() == d]; hm = kd.index.strftime("%H:%M"); e = kd[(hm >= "09:00") & (hm < "10:00")]; xl = xd[(xd.index.strftime("%H:%M") >= "10:00") & (xd.index.strftime("%H:%M") <= "15:19")]
    if len(e) < 30 or len(xl) < 10: continue
    sgn = np.sign(e.close.iloc[-1] / e.open.iloc[0] - 1); R[d] = {"rng": (e.high.max() / e.low.min() - 1) * 100, "follow": sgn * (xl.close.iloc[-1] / xl.open.iloc[0] - 1) * 1e4, "abs": abs(xl.close.iloc[-1] / xl.open.iloc[0] - 1) * 1e4}
X = pd.DataFrame(R).T; X["rel"] = X.rng / X.rng.rolling(10).median().shift(1); X = X.dropna()
print(f"\n[3] 선물 1분봉 {len(X)}일 — 신호는 코스피200 선물 09~10시 범위(직전 10일 중앙값 대비), 매매는 KRX300 선물 10시 이후 첫 체결가 → 15:19 이전 마지막 체결가, 왕복 2틱(1.2bp)+수수료로 3bp")
for nm, m in (("넓은 날 (1.25배↑)", X.rel >= 1.25), ("넓은 날 (1.0배↑)", X.rel >= 1.0), ("좁은 날 (0.8배↓)", X.rel <= 0.8), ("모든 날", X.rel > 0)):
    S = X[m]; print(f"  {nm}: {len(S)}일, 이후 움직임 크기 중앙 {S['abs'].median():.0f}bp"); print(line(S.follow - COST, "A 추종", len(S))); print(line(-S.follow - COST, "B 역추종", len(S)))
