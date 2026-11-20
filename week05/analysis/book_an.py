"""11/20 호가 녹화(5초 간격) 분석 — KRX300 선물 스프레드·잔량, 코스피200 대비 호가가 늦는 정도, 체결이 어느 쪽 호가에서 났는지.
실행: python3 book_an.py [파일]   (book_rec.py 가 만든 jsonl)"""
import json, sys, pandas as pd, numpy as np
pd.set_option("display.width", 220)
fn = sys.argv[1] if len(sys.argv) > 1 else "book/book_20261120.jsonl"; R = []
for ln in open(fn):
    try: r = json.loads(ln)
    except Exception: continue
    x, k, m = r["A086C000"], r["A016C000"], r["A056C000"]
    def bb(o): return (max(p for p, q in o["b"]) if o["b"] else np.nan, min(p for p, q in o["a"]) if o["a"] else np.nan)
    def qty(o, side, px): return next((q for p, q in o[side] if p == px), np.nan)
    xb, xa = bb(x); kb, ka = bb(k); mb, ma = bb(m)
    R.append({"t": pd.Timestamp(r["t"]).tz_localize(None), "xb": xb, "xa": xa, "xbq": qty(x, "b", xb), "xaq": qty(x, "a", xa), "xlast": x.get("last"), "xvol": x.get("vol"), "oi": x.get("oi"), "spot": x.get("spot"), "theory": x.get("theory"),
              "kb": kb, "ka": ka, "klast": k.get("last"), "kvol": k.get("vol"), "mb": mb, "ma": ma, "xbt": x.get("bt"), "nlev_b": len(x["b"]), "nlev_a": len(x["a"]),
              "depth_b": sum(q for p, q in x["b"]), "depth_a": sum(q for p, q in x["a"])})
D = pd.DataFrame(R).set_index("t"); D["hm"] = D.index.strftime("%H:%M"); T = 0.2
print("기록", len(D), D.index[0].strftime("%H:%M:%S"), "~", D.index[-1].strftime("%H:%M:%S"), "| 간격 중앙", round(float(pd.Series(D.index).diff().dt.total_seconds().median()), 1), "초")
pre = D[D.hm < "08:45"]; print("08:45 전: KRX300 양쪽 호가 있는 기록", int((pre.xb.notna() & pre.xa.notna()).sum()), "/", len(pre))
first_q = D[D.xb.notna() & D.xa.notna()].index[0]; first_tr = D[D.xvol > 0].index[0]; print("KRX300 양쪽 호가 처음 보인 시각", first_q.strftime("%H:%M:%S"), "| 첫 체결 확인 시각", first_tr.strftime("%H:%M:%S"), "첫 체결가", D.xlast[first_tr])
S = D[(D.hm >= "09:00") & D.xb.notna() & D.xa.notna() & D.kb.notna() & D.ka.notna()].copy()
S["sp"] = (S.xa - S.xb) / T; S["ksp"] = (S.ka - S.kb) / 0.05; S["xmid"] = (S.xa + S.xb) / 2; S["kmid"] = (S.ka + S.kb) / 2
print(f"\n[1] KRX300 스프레드 (09:00 이후 {len(S)}건): 중앙 {S.sp.median():.0f}틱 = {S.sp.median()*T/S.xmid.median()*1e4:.1f}bp | 분위 10% {S.sp.quantile(.1):.0f} · 25% {S.sp.quantile(.25):.0f} · 75% {S.sp.quantile(.75):.0f} · 90% {S.sp.quantile(.9):.0f} · 최소 {S.sp.min():.0f} · 최대 {S.sp.max():.0f}틱")
print("    스프레드 분포(틱→비율%):", {f"{a}~{b}": round(float(((S.sp >= a) & (S.sp <= b)).mean() * 100), 1) for a, b in ((1, 2), (3, 5), (6, 10), (11, 15), (16, 25), (26, 999))})
print(f"    코스피200 선물 스프레드 중앙 {S.ksp.median():.0f}틱 = {S.ksp.median()*0.05/S.kmid.median()*1e4:.2f}bp | 미니 {((D.ma-D.mb)/0.02).median():.0f}틱")
print(f"    최우선 잔량 중앙: 매수 {S.xbq.median():.0f} / 매도 {S.xaq.median():.0f}계약 | 보이는 5단계 합: 매수 {S.depth_b.median():.0f} / 매도 {S.depth_a.median():.0f}계약 | 호가 단계 수 중앙 {S.nlev_b.median():.0f}/{S.nlev_a.median():.0f}")
print("    시간대별 스프레드 중앙(틱):", S.groupby(S.index.floor("30min").strftime("%H:%M")).sp.median().round(0).astype(int).to_dict())
S["ratio"] = np.log(S.xmid / S.kmid); S["m"] = S.ratio.rolling(360, min_periods=60).median().shift(1); S["fair"] = np.exp(S.m) * S.kmid; S = S.dropna(subset=["fair"])
S["dev"] = (S.xmid / S.fair - 1) * 1e4; print(f"\n[2] 호가 중간값이 코스피200 환산 적정가(직전 30분 비율 기준)에서 벗어난 정도: 표준편차 {S.dev.std():.1f}bp, |괴리| 중앙 {S.dev.abs().median():.1f}bp, 95% {S.dev.abs().quantile(.95):.1f}bp")
S["edge_buy"] = (S.fair - S.xa) / T; S["edge_sell"] = (S.xb - S.fair) / T          # +면 호가가 적정가를 넘어 들어와 있음(치면 그만큼 유리)
e = np.maximum(S.edge_buy, S.edge_sell); print(f"    매도호가가 적정가보다 싸거나 매수호가가 적정가보다 비싼 순간(호가를 치면 유리): {(e>0).mean()*100:.1f}% 의 기록, 그때 유리한 폭 중앙 {e[e>0].median() if (e>0).any() else 0:.1f}틱, 최대 {e.max():.1f}틱")
for th in (1, 2, 3, 5): print(f"      유리한 폭 ≥{th}틱: {int((e>=th).sum())}건 ({(e>=th).mean()*100:.2f}%)")
print(f"    적정가에서 가까운 쪽 호가까지 거리(틱): 중앙 {(-e).median():.1f} → 호가를 치려면 평균 이만큼 불리하게 사야 함")
S["dk"] = np.log(S.kmid).diff() * 1e4; S["dx"] = np.log(S.xmid).diff() * 1e4
print("\n[3] 코스피200 호가 중간값이 움직인 뒤 KRX300 호가 중간값이 따라오는 속도 (5초 단위 시차 상관)")
print("    corr(코스피200 변화 t, KRX300 변화 t+k):", " ".join(f"k={k_}:{S.dk.corr(S.dx.shift(-k_)):+.2f}" for k_ in (-2, -1, 0, 1, 2, 3, 6)))
big = S[S.dk.abs() >= S.dk.abs().quantile(.95)]; sg = np.sign(big.dk)
resp = {k_: float(((np.log(S.xmid.shift(-k_)) - np.log(S.xmid.shift(1))).reindex(big.index) * 1e4 * sg).mean()) for k_ in (0, 1, 2, 3, 6, 12)}
print(f"    코스피200 이 5초에 크게 움직인 {len(big)}건(평균 {(big.dk*sg).mean():.1f}bp): KRX300 호가 중간값의 누적 반응(bp) — 같은 5초 {resp[0]:.1f}, 5초 뒤 {resp[1]:.1f}, 10초 뒤 {resp[2]:.1f}, 15초 뒤 {resp[3]:.1f}, 30초 뒤 {resp[6]:.1f}, 60초 뒤 {resp[12]:.1f}")
A = D[D.hm >= "09:00"].copy(); A["dv"] = A.xvol.diff(); tr = A[(A.dv > 0)].copy(); pb, pa = A.xb.shift(1), A.xa.shift(1); tr["pb"] = pb.reindex(tr.index); tr["pa"] = pa.reindex(tr.index)
tr = tr.dropna(subset=["pb", "pa"]); tr["where"] = np.where(tr.xlast >= tr.pa, "매도호가 이상(매수가 침)", np.where(tr.xlast <= tr.pb, "매수호가 이하(매도가 침)", "호가 사이"))
print(f"\n[4] 체결이 난 5초 구간 {len(tr)}개 (계약 {int(tr.dv.sum())}): 마지막 체결가가 직전 호가의 어디였나 (구간 수 비율 / 계약 비율)")
for w, g in tr.groupby("where"): print(f"    {w:22s} {len(g)/len(tr)*100:4.0f}% / {g.dv.sum()/tr.dv.sum()*100:4.0f}%   구간당 계약 중앙 {g.dv.median():.0f}")
tr["psp"] = (tr.pa - tr.pb) / T; print(f"    체결 직전 스프레드 중앙 {tr.psp.median():.0f}틱 (전체 중앙 {S.sp.median():.0f}틱) | 체결 구간 사이 간격 중앙 {pd.Series(tr.index).diff().dt.total_seconds().median():.0f}초")
D.to_csv("book/book_20261120_flat.csv")
