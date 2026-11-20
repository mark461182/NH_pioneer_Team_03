"""시장조성자 호가 '바깥'에 지정가를 걸어 두고, 체결되면 바로 시장조성자 호가를 쳐서 청산하는 방식 — 11/20 호가 녹화(5초)로 계산.
  내 매수 = 상대 최우선 매수호가 − d틱, 내 매도 = 상대 최우선 매도호가 + d틱 (5초마다 상대 호가에 맞춰 다시 걺)
  체결 = 다음 5초에 거래가 있었고 마지막 체결가가 내 가격 이하(이상).  청산 = 체결을 안 다음 기록의 상대 최우선 호가에 반대 주문.
실행: python3 book_outside.py"""
import pandas as pd, numpy as np, json, sys
D = pd.read_csv("book/book_20261120_flat.csv", index_col=0, parse_dates=True); T = 0.2
D = D[(D.hm >= "09:05") & (D.hm <= "15:20") & D.xb.notna() & D.xa.notna()].copy(); D["dv"] = D.xvol.diff().fillna(0); N, N2 = D.shift(-1), D.shift(-2)
hrs = (D.index[-1] - D.index[0]).total_seconds() / 3600; print(f"구간 {D.index[0]:%H:%M} ~ {D.index[-1]:%H:%M} ({hrs:.1f}시간), 기록 {len(D)}, 거래 있던 5초 구간 {int((D.dv>0).sum())}개")
# 호가 단계 간격과 깊이
lv = []
for ln in open(sys.argv[1] if len(sys.argv) > 1 else "../results/20261120/book_20261120.jsonl"):
    try: r = json.loads(ln)["A086C000"]
    except Exception: continue
    if len(r["b"]) >= 5 and len(r["a"]) >= 5:
        b = sorted(r["b"], reverse=True); a = sorted(r["a"]); lv.append(((b[0][0] - b[4][0]) / T, (a[4][0] - a[0][0]) / T, sum(q for _, q in b), sum(q for _, q in a)))
lv = np.array(lv); print(f"보이는 5단계: 최우선에서 5번째까지 거리 중앙 매수 {np.median(lv[:,0]):.0f}틱 / 매도 {np.median(lv[:,1]):.0f}틱, 5단계 잔량 합 중앙 {np.median(lv[:,2]):.0f} / {np.median(lv[:,3]):.0f}계약")
thr = (N.xlast - D.xb) / T; tha = (D.xa - N.xlast) / T; trd = N.dv > 0
print("거래가 있던 구간에서 마지막 체결가가 직전 상대 호가를 얼마나 뚫었나(틱, +면 호가보다 더 깊이):")
deep = pd.concat([(-thr[trd & (N.xlast <= D.xb)]), (-tha[trd & (N.xlast >= D.xa)])]); print("  분포:", {f"≥{k}틱": int((deep >= k).sum()) for k in (0, 1, 2, 3, 5, 8, 10, 15, 20)}, "| 최대", round(float(deep.max()), 0))
print(f"\n{'d':>3s} {'체결':>5s} {'시간당':>6s} | 청산 손익(틱): {'평균':>6s} {'중앙':>6s} {'승률':>5s} {'최악':>6s} {'최고':>6s} | {'합(틱)':>7s} | 체결 순간 상대 호가는 어디로 갔나: 내 가격보다 위(유리) 비율")
for d_ in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
    bid = D.xb - d_ * T; ask = D.xa + d_ * T
    fb = trd & (N.xlast <= bid + 1e-9); fa = trd & (N.xlast >= ask - 1e-9)
    pl = pd.concat([(N2.xb[fb] - bid[fb]) / T, (ask[fa] - N2.xa[fa]) / T]).dropna()
    if len(pl) == 0: print(f"{d_:3d} {0:5d}"); continue
    print(f"{d_:3d} {len(pl):5d} {len(pl)/hrs:6.1f} |               {pl.mean():+6.1f} {pl.median():+6.1f} {(pl>0).mean()*100:4.0f}% {pl.min():+6.0f} {pl.max():+6.0f} | {pl.sum():+7.0f} | {(pl>0).mean()*100:.0f}%")
print("\n체결 시각 예 (d=5):"); bid = D.xb - 5 * T; ask = D.xa + 5 * T; fb = trd & (N.xlast <= bid + 1e-9); fa = trd & (N.xlast >= ask - 1e-9)
for t in list(D.index[fb | fa])[:10]:
    s = "매수" if fb[t] else "매도"; my = bid[t] if fb[t] else ask[t]; ex = N2.xb[t] if fb[t] else N2.xa[t]
    print(f"  {t:%H:%M:%S} {s} 내 가격 {my:.1f} | 직전 상대 호가 {D.xb[t]}/{D.xa[t]} → 체결가 {N.xlast[t]} ({int(N.dv[t])}계약) → 다음 상대 호가 {N2.xb[t]}/{N2.xa[t]} | 청산 {ex} 손익 {((ex-my) if fb[t] else (my-ex))/T:+.0f}틱 | 그 5초 코스피200 {((N.kb[t]+N.ka[t])/(D.kb[t]+D.ka[t])-1)*1e4:+.1f}bp")

# 체결된 5초 동안 코스피200 이 내게 불리한 쪽으로 얼마나 움직였는지로 나눠 보기 (불리 = 매수 체결인데 하락, 매도 체결인데 상승)
print("\n[나눠 보기] 체결이 난 5초 동안 코스피200 이 내게 불리하게 움직인 폭별 청산 손익 (틱)")
kch = (np.log((N.kb + N.ka) / (D.kb + D.ka)) * 1e4)
print(f"{'d':>3s} | {'불리 2bp 미만':>22s} | {'2~5bp':>22s} | {'5bp 이상':>22s}")
for d_ in (2, 3, 5, 8, 10):
    bid = D.xb - d_ * T; ask = D.xa + d_ * T; fb = trd & (N.xlast <= bid + 1e-9); fa = trd & (N.xlast >= ask - 1e-9)
    pl = pd.concat([(N2.xb[fb] - bid[fb]) / T, (ask[fa] - N2.xa[fa]) / T]); adv = pd.concat([-kch[fb], kch[fa]]); ok = pl.notna(); pl, adv = pl[ok], adv[ok]
    cell = lambda m: f"{int(m.sum()):3d}건 평균 {pl[m].mean():+6.1f} 승률 {(pl[m]>0).mean()*100:3.0f}%" if m.sum() else "  0건"
    print(f"{d_:3d} | {cell(adv < 2)} | {cell((adv >= 2) & (adv < 5))} | {cell(adv >= 5)}")
allk = kch.abs(); print(f"참고: 전체 5초 구간 중 코스피200 이 2bp 미만으로 움직인 비율 {(allk<2).mean()*100:.0f}%, 5bp 이상 {(allk>=5).mean()*100:.0f}%")
