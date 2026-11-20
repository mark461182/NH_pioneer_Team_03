"""장 시작 지정가 전략 — 호가를 5초마다 고쳤다면 (근사).
  과거 43일은 30초봉뿐이라 30초 안의 코스피200 경로를 모름 → 내 매수호가는 그 30초 동안 [R×코스피200 저가 − d, R×코스피200 고가 − d] 사이를 움직였다고 보고
    확실한 체결: KRX300 저가가 구간 맨 아래(R×저가 − d)보다도 낮음      가능한 체결: KRX300 저가가 구간 맨 위(R×고가 − d)보다 낮음
    체결가: 확실한 체결은 구간 중간(R×(고+저)/2 − d) 으로 가정, 헤지는 같은 값에 대응하는 코스피200 가격
  나머지(청산 09:15~09:19 시장가 −7틱, 헤지 1.5틱)는 open_limit_bt.py 와 같음.  실행: python3 open_limit_fast.py"""
import json, pandas as pd, numpy as np
def L(n, f):
    d = pd.DataFrame(json.load(open(f"data/{n}_{f}.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300", "30s"), L("k200", "30s"); x1, k1 = L("krx300", "1m"), L("k200", "1m"); T = 0.2; DAYS = {}; prev = None
for d in sorted({t for t in x.index.normalize()}):
    J = x1[x1.index.normalize() == d].join(k1[k1.index.normalize() == d], how="inner", rsuffix="_k"); hm = J.index.strftime("%H:%M"); late = J[(hm >= "15:00") & (hm <= "15:19")]
    r_today = np.log(late.close / late.close_k).median() if len(late) >= 5 else None
    if prev is not None:
        xd = x[x.index.normalize() == d]; kd = k[k.index.normalize() == d]; g = pd.date_range(d + pd.Timedelta(hours=8, minutes=45), d + pd.Timedelta(hours=9, minutes=19, seconds=30), freq="30s")
        G = pd.DataFrame({"xl": xd.low, "xh": xd.high, "xc": xd.close}).reindex(g); G["kc"] = kd.close.reindex(g).ffill(); G["kh"] = kd.high.reindex(g).fillna(G.kc); G["kl"] = kd.low.reindex(g).fillna(G.kc); G["ko"] = kd.open.reindex(g).fillna(G.kc.shift(1))
        ex = G[G.index.strftime("%H:%M") >= "09:15"]; xe = ex.xc.median(); ke = ex.kc.dropna().iloc[-1] if ex.kc.notna().any() else np.nan
        if not np.isnan(xe) and not np.isnan(ke): DAYS[d] = (G, np.exp(prev), xe, ke)
    if r_today is not None: prev = r_today
def bt(d_, mode):
    out = []
    for day, (G, R, xe, ke) in DAYS.items():
        W = G[G.index.strftime("%H:%M") < "09:00"]
        if mode == "30초": lo_b = hi_b = np.floor((R * W.ko - d_ * T) / T + 1e-9) * T; lo_a = hi_a = np.ceil((R * W.ko + d_ * T) / T - 1e-9) * T; kref = W.kc
        else: lo_b, hi_b = R * W.kl - d_ * T, R * W.kh - d_ * T; lo_a, hi_a = R * W.kl + d_ * T, R * W.kh + d_ * T; kref = (W.kh + W.kl) / 2
        thr_b = lo_b if mode in ("30초", "확실") else hi_b; thr_a = hi_a if mode in ("30초", "확실") else lo_a
        for side, hit in ((1, (W.xl < thr_b - 1e-9)), (-1, (W.xh > thr_a + 1e-9))):
            h = hit[hit.fillna(False)]
            if len(h):
                t = h.index[0]; kf = kref[t]; p = (R * kf - side * d_ * T) if mode != "30초" else (lo_b[t] if side == 1 else lo_a[t])
                out.append({"day": day, "pnl": side * ((xe - side * 7 * T) - p) / T - side * R * (ke - kf) / T - 1.5})
    return pd.DataFrame(out)
print(f"날 수 {len(DAYS)}  (08:45~08:59 주문, 손익 틱)\n{'거리':>4s} | {'30초 갱신(앞 결과)':^34s} | {'5초 갱신 근사 — 확실한 체결만':^34s} | {'5초 갱신 근사 — 가능한 체결 전부':^34s}")
for d_ in (5, 10, 15, 20, 30, 40):
    cells = []
    for mode in ("30초", "확실", "가능"):
        Z = bt(d_, mode); cells.append(f"{len(Z):3d}건 평균 {Z.pnl.mean():+6.1f} 승률 {(Z.pnl>0).mean()*100:3.0f}% t {Z.pnl.mean()/Z.pnl.std()*np.sqrt(len(Z)):+4.1f}" if len(Z) > 3 else f"{len(Z):3d}건")
    print(f"{d_:4d} | " + " | ".join(cells))
print("\n이론값: 호가가 시세를 바로 따라가면 체결 직후 유리폭 = d틱, 여기서 청산 7틱 + 헤지 1.5틱을 빼면 기대 손익 ≈ d − 8.5틱 (비율 변화는 평균 0 의 잡음)")
