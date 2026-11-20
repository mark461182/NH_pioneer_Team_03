"""11/20 호가 녹화(5초)로 본 ① 실제 체결이 호가 대는 쪽에 유리했나 ② 스프레드 안쪽에 내 호가를 넣었다면.
적정가 = 직전 30분 (KRX300 호가 중간값 / 코스피200 호가 중간값) 비율 중앙값 × 코스피200 호가 중간값.
실행: python3 book_mm.py   (book_an.py 가 만든 book/book_20261120_flat.csv 사용)"""
import pandas as pd, numpy as np, sys
D = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "book/book_20261120_flat.csv", index_col=0, parse_dates=True); T = 0.2
D = D[(D.hm >= "09:00") & D.xb.notna() & D.xa.notna() & D.kb.notna()].copy(); D["xmid"] = (D.xa + D.xb) / 2; D["kmid"] = (D.ka + D.kb) / 2
D["m"] = np.log(D.xmid / D.kmid).rolling(360, min_periods=60).median().shift(1); D["fair"] = np.exp(D.m) * D.kmid; D = D.dropna(subset=["fair"]); D["dv"] = D.xvol.diff().fillna(0)
for h in (1, 2, 6, 12, 60): D[f"f{h}"] = D.fair.shift(-h)
hrs = (D.index[-1] - D.index[0]).total_seconds() / 3600; print(f"구간 {D.index[0]:%H:%M} ~ {D.index[-1]:%H:%M} ({hrs:.1f}시간), 기록 {len(D)}")
# ① 실제 체결: 직전 호가 기준으로 어느 쪽이 쳤는지 → 호가 대던 쪽의 손익
P = D.shift(1); tr = D[D.dv > 0].copy(); tr["pb"], tr["pa"], tr["pf"] = P.xb.reindex(tr.index), P.xa.reindex(tr.index), P.fair.reindex(tr.index); tr = tr.dropna(subset=["pb", "pa", "pf"])
tr["side"] = np.where(tr.xlast >= tr.pa, -1, np.where(tr.xlast <= tr.pb, 1, 0))          # 호가 대던 쪽 기준: +1 = 매수 체결됨(상대가 팔았음), -1 = 매도 체결됨
q = tr[tr.side != 0]
print(f"\n① 호가를 친 체결 구간 {len(q)}개, {int(q.dv.sum())}계약 (호가 사이 체결 {int((tr.side==0).sum())}개는 제외)")
print(f"   친 쪽이 낸 폭(체결가와 직전 적정가의 거리): 중앙 {((q.pf - q.xlast) * q.side / T).median():+.1f}틱, 평균 {((q.pf - q.xlast) * q.side / T).mean():+.1f}틱")
print("   호가 대던 쪽의 손익(틱) = (이후 적정가 − 체결가) × 방향:")
for h, nm in ((1, "5초 뒤"), (2, "10초 뒤"), (6, "30초 뒤"), (12, "60초 뒤"), (60, "5분 뒤")):
    v = ((q[f"f{h}"] - q.xlast) * q.side / T).dropna(); w = q.dv.reindex(v.index)
    print(f"     {nm:6s}: 평균 {v.mean():+5.2f}  계약가중 {np.average(v, weights=w):+5.2f}  중앙 {v.median():+5.2f}  플러스 비율 {(v>0).mean()*100:3.0f}%  t {v.mean()/v.std()*np.sqrt(len(v)):+5.1f}")
kmv = (np.log(D.kmid / D.kmid.shift(1)) * 1e4).reindex(q.index) * -q.side
print(f"   체결이 난 5초 동안 코스피200 이 친 쪽에 유리하게 움직인 폭: 평균 {kmv.mean():+.1f}bp ({kmv.mean()/0.59:+.1f}틱), 그런 비율 {(kmv>0).mean()*100:.0f}% | 그 5초에 코스피200 이 3bp 넘게 움직인 체결 {((D.kmid.pct_change().abs()*1e4).reindex(q.index)>3).mean()*100:.0f}% (전체 5초 구간에선 {((D.kmid.pct_change().abs()*1e4)>3).mean()*100:.0f}%)")
# ② 스프레드 안쪽에 내 호가: 매 5초 적정가 ∓ d틱, 다음 5초에 상대가 내 가격 이하(이상)로 체결했으면 체결
print("\n② 내 호가를 적정가 ∓ d틱에 걸고 5초마다 고칠 때 (체결 = 다음 5초에 거래가 있었고 마지막 체결가가 내 가격을 건드림, 1계약)")
print(f"   {'d':>3s} {'내가 최우선인 비율':>10s} {'체결 수':>6s} {'시간당':>6s} | 체결 후 손익(틱): {'5초':>6s} {'30초':>6s} {'60초':>6s} {'5분':>6s} | {'60초 승률':>7s} {'t':>5s} | 코스피200 헤지 1.5틱 뺀 60초 손익 합")
N = D.shift(-1)
for d_ in (1, 2, 3, 4, 5, 6, 8):
    bid = np.floor((D.fair - d_ * T) / T + 1e-9) * T; ask = np.ceil((D.fair + d_ * T) / T - 1e-9) * T
    fb = (N.dv > 0) & (N.xlast <= bid + 1e-9); fa = (N.dv > 0) & (N.xlast >= ask - 1e-9)
    F = pd.concat([D[fb].assign(px=bid[fb], sd=1), D[fa].assign(px=ask[fa], sd=-1)]).sort_index()
    best = ((bid > D.xb + 1e-9) & (ask < D.xa - 1e-9)).mean()
    if len(F) < 5: print(f"   {d_:3d} {best*100:9.0f}% {len(F):6d}"); continue
    mo = {h: ((F[f"f{h + 1}"] if f"f{h + 1}" in F else F.fair.shift(-h - 1)) - F.px) * F.sd / T for h in (1,)}
    g = lambda h: ((D.fair.shift(-h).reindex(F.index).values - F.px.values) * F.sd.values / T)
    m5, m30, m60, m300 = g(2), g(7), g(13), g(61); v = pd.Series(m60).dropna()
    print(f"   {d_:3d} {best*100:9.0f}% {len(F):6d} {len(F)/hrs:6.1f} |                 {np.nanmean(m5):+6.2f} {np.nanmean(m30):+6.2f} {np.nanmean(m60):+6.2f} {np.nanmean(m300):+6.2f} | {(v>0).mean()*100:6.0f}% {v.mean()/v.std()*np.sqrt(len(v)):+5.1f} | {(v-1.5).sum():+7.0f}틱 ({len(v)}건)")
print("\n주의: 5초 사이에 무슨 일이 있었는지는 모름(마지막 체결가와 누적 거래량만 봄). 내 주문이 있었다면 상대의 행동이 달라졌을 수 있음. 하루 오전 한 번의 자료.")
