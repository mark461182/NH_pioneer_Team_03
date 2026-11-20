"""장 시작 넓은 스프레드 안쪽에 지정가를 거는 전략 백테스트 (30초봉, 9/11~11/18, 44일).
  적정가 F = 전일 15:00~15:19 (KRX300/코스피200) 비율 중앙값 × 그 30초 시작 때 코스피200 선물가
  08:45부터 끝 시각까지 30초마다 매수 F−d틱, 매도 F+d틱 에 1계약씩 다시 걺. 한쪽이 체결되면 그쪽은 그날 끝.
  체결 = 그 30초 KRX300 저가(고가)가 내 가격을 '뚫음'(엄격) 또는 '닿음'(느슨).
  헤지 = 체결된 30초봉 종가에 코스피200 반대 포지션. 청산 = 09:15~09:19 KRX300 체결가 중앙값 ∓ 7틱(스프레드 절반, 시장가 청산), 코스피200 도 같은 시각 청산.
  비용 = 헤지 왕복 1.5틱. 손익 단위 틱(1틱 = 1만원/계약).   실행: python3 open_limit_bt.py"""
import json, pandas as pd, numpy as np
pd.set_option("display.width", 220)
def L(n, f):
    d = pd.DataFrame(json.load(open(f"data/{n}_{f}.json"))); d.index = pd.to_datetime(d.timestamp, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None); return d.loc["2026-09-11":"2026-11-18 23:59"]
x, k = L("krx300", "30s"), L("k200", "30s"); x1, k1 = L("krx300", "1m"), L("k200", "1m"); T = 0.2; HALF = 7; HEDGE = 1.5; DAYS = {}; prev = None
for d in sorted({t for t in x.index.normalize()}):
    J = x1[x1.index.normalize() == d].join(k1[k1.index.normalize() == d], how="inner", rsuffix="_k"); hm = J.index.strftime("%H:%M")
    late = J[(hm >= "15:00") & (hm <= "15:19")]; r_today = np.log(late.close / late.close_k).median() if len(late) >= 5 else None
    if prev is not None:
        xd = x[x.index.normalize() == d]; kd = k[k.index.normalize() == d]
        g = pd.date_range(d + pd.Timedelta(hours=8, minutes=45), d + pd.Timedelta(hours=9, minutes=19, seconds=30), freq="30s")
        G = pd.DataFrame({"xl": xd.low, "xh": xd.high, "xc": xd.close, "xv": xd.volume}).reindex(g); G["kc"] = kd.close.reindex(g).ffill(); G["ko"] = kd.open.reindex(g).fillna(G.kc.shift(1))
        ex = G[(G.index.strftime("%H:%M") >= "09:15")]; xe = ex.xc.median(); ke = ex.kc.dropna().iloc[-1] if ex.kc.notna().any() else np.nan
        if not np.isnan(xe) and not np.isnan(ke): DAYS[d] = (G, np.exp(prev), xe, ke)
    if r_today is not None: prev = r_today
nd = len(DAYS); print("날 수", nd)
pre = [(G.xv[G.index.strftime("%H:%M") < "09:00"].fillna(0).sum(), G.xv[G.index.strftime("%H:%M") < "09:00"].notna().sum()) for G, *_ in DAYS.values()]
print("08:45~08:59 KRX300: 하루 체결 계약 중앙", np.median([a for a, b in pre]), "| 체결 있는 30초 구간 중앙", np.median([b for a, b in pre]), "/ 30 | 체결이 아예 없던 날", sum(1 for a, b in pre if a == 0))
def bt(d_, end="09:00", strict=True, exit_half=HALF):
    out = []
    for day, (G, R, xe, ke) in DAYS.items():
        W = G[G.index.strftime("%H:%M") < end]; F = R * W.ko; bid = np.floor((F - d_ * T) / T + 1e-9) * T; ask = np.ceil((F + d_ * T) / T - 1e-9) * T
        hb = (W.xl < bid - 1e-9) if strict else (W.xl <= bid + 1e-9); ha = (W.xh > ask + 1e-9) if strict else (W.xh >= ask - 1e-9)
        for side, hit, px in ((1, hb, bid), (-1, ha, ask)):
            h = hit[hit.fillna(False)]
            if len(h):
                t = h.index[0]; p = px[t]; kf = W.kc[t]
                pnl = side * ((xe - side * exit_half * T) - p) / T - side * R * (ke - kf) / T - HEDGE
                out.append({"day": day, "t": t.strftime("%H:%M:%S"), "side": side, "px": p, "pnl": pnl, "edge_at_fill": side * (R * kf - p) / T, "drift": side * (xe - R * ke) / T})
    return pd.DataFrame(out)
def show(lbl, Z):
    if len(Z) < 4: print(f"  {lbl:34s} 체결 {len(Z)}"); return
    dp = Z.groupby("day").pnl.sum().reindex(list(DAYS)).fillna(0); h = len(dp) // 2
    print(f"  {lbl:34s} 체결 {len(Z):3d} ({Z.day.nunique():2d}일) 평균 {Z.pnl.mean():+6.1f} 중앙 {Z.pnl.median():+6.1f} 승률 {(Z.pnl>0).mean()*100:3.0f}% t {Z.pnl.mean()/Z.pnl.std()*np.sqrt(len(Z)):+5.1f} 최악 {Z.pnl.min():+5.0f} 합 {Z.pnl.sum():+6.0f} | 앞/뒤 절반 합 {dp[:h].sum():+5.0f}/{dp[h:].sum():+5.0f} | 체결 직후 유리폭 {Z.edge_at_fill.mean():+5.1f}, 청산 때 비율 변화 몫 {Z.drift.mean():+5.1f}")
print("\n[1] 08:45~08:59 만, 엄격한 체결, 시장가 청산(−7틱)");
for d_ in (5, 10, 15, 20, 30, 40, 60): show(f"적정가 ∓{d_}틱", bt(d_))
print("[2] 같은 조건, 느슨한 체결(닿으면 체결)");
for d_ in (10, 20, 30, 40): show(f"적정가 ∓{d_}틱", bt(d_, strict=False))
print("[3] 08:45~09:04 까지 걸어 둠, 엄격");
for d_ in (10, 20, 30, 40): show(f"적정가 ∓{d_}틱", bt(d_, end="09:05"))
print("[4] 청산을 호가 중간값에 했다고 가정(−0틱) — 상한, 08:45~08:59 엄격");
for d_ in (10, 20, 30): show(f"적정가 ∓{d_}틱", bt(d_, exit_half=0))
Z = bt(20); print("\n[5] ∓20틱 체결 목록 (엄격, 08:45~08:59)"); Z["day"] = Z.day.dt.strftime("%m-%d"); print(Z.round(1).to_string(index=False))
