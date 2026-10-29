"""팀원 24년 데이터로 리스크 레이어를 학습 구간(2001~2024)까지 확인한다.

확인할 것 (팀원 차트를 보고 미리 정한 4가지)
  1. 2008년: 원안이 버틴 해에 브레이크가 수익을 깎는가
  2. 2020년 코로나: 브레이크가 손실을 막았는가
  3. 2011~2024 횡보: 브레이크 비용(놓친 수익)은 얼마인가
  4. S3를 브레이크에서 빼면 나은가
  + v1(실제 계좌 낙폭)의 영구 정지 결함이 실제 데이터에서 일어나는가
  + VaR 사후검증(Kupiec)이 24년에서도 FHS만 통과하는가

실행: python team/risk_is_check.py [--csv results/k200_1d_continuous.csv]  → team/risk_is_report.md
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k200_daily4_replicate as K  # noqa: E402
import k200_daily4_improved as KI  # noqa: E402
import risk_layer as R  # noqa: E402


def fmt(r, m):
    x = K.metrics(r[m])
    return [f"{x['누적'] * 100:+.0f}%", f"{x['변동성'] * 100:.1f}%", f"{x['샤프']:.2f}", f"{x['MDD'] * 100:.1f}%"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/k200_1d_continuous.csv")
    ap.add_argument("--case", default="튜닝")
    ap.add_argument("--out", default="team/risk_is_report.md")
    a = ap.parse_args()
    d, src = KI.load_any(a.csv)
    d = d.set_index("dt", drop=False)
    IS = (d["dt"] >= "2001-01-02") & (d["dt"] <= "2024-12-30")
    OOS = d["dt"] >= "2025-01-02"
    nets, _ = K.run_case(d.reset_index(drop=True), K.PARAMS[a.case])
    P = {k: pd.Series(nets[k].to_numpy(), index=d.index) for k in ("S1", "S2", "S3", "S4")}
    bh = pd.Series(nets["매수보유"].to_numpy(), index=d.index)
    c = pd.DataFrame(P).mean(axis=1)
    P2 = R.calibrate(c, ~OOS)
    r, w = R.apply_limits(d, c, True, True, True, P2)
    r1, _ = R.apply_limits(d, c, True, False, False, P2)
    r3, _ = R.apply_limits(d, c, False, False, True, P2)
    rv1, wv1 = R.apply_limits(d, c, True, True, True, R.V1)
    c124 = pd.DataFrame({k: P[k] for k in ("S1", "S2", "S4")}).mean(axis=1)
    r124, _ = R.apply_limits(d, c124, True, True, True, R.calibrate(c124, ~OOS))
    rx = 0.75 * r124 + 0.25 * P["S3"]

    md = [f"# 리스크 레이어 — 24년 데이터 확인 ({a.case})\n", f"데이터 {src} ({d['dt'].min().date()} ~ {d['dt'].max().date()})\n",
          f"v2 한도(학습 구간에서 계산): ES {P2['es_limit']:.2%} (k={P2['k']:.2f}) · 낙폭 ½선 {P2['dd_half']:.1%} · 0선 {P2['dd_zero']:.1%} · 복귀 {P2['dd_resume']:.1%}\n"]
    rows = []
    for nm, s in (("원안", c), ("① 변동성 타깃", r1), ("③ ES 한도", r3), ("①+②+③ (v2)", r), ("①+②+③ (v1 판단값)", rv1),
                  ("S3 제외 브레이크", rx), ("매수보유", bh)):
        rows.append([nm] + fmt(s, IS) + fmt(s, OOS))
    md.append("## 1. 학습(2001~2024) vs 검증(2025~)\n")
    md.append(pd.DataFrame(rows, columns=["구성", "학습 누적", "학습 변동성", "학습 샤프", "학습 낙폭",
                                          "검증 누적", "검증 변동성", "검증 샤프", "검증 낙폭"]).to_markdown(index=False) + "\n")
    z = wv1[wv1 == 0]
    md.append(f"v1 정지(노출 0) {len(z)}일: {z.index.min().date() if len(z) else '-'} 이후 끝까지 복귀 못 함\n" if len(z) else "v1 정지 없음\n")

    md.append("## 2. 위기 구간\n")
    rows = []
    for nm, s_, e_ in (("2002-04~2003-03 IT 버블 후", "2002-04-01", "2003-03-31"), ("2008 금융위기", "2008-01-01", "2008-12-31"),
                       ("2011 하반기 유럽위기", "2011-07-01", "2011-12-31"), ("2020-02~04 코로나", "2020-02-01", "2020-04-30"),
                       ("2026-06~10 검증 급락", "2026-06-22", "2026-10-26")):
        m = (d["dt"] >= s_) & (d["dt"] <= e_)
        rows.append([nm] + [fmt(bh, m)[0]] + fmt(c, m)[::3] + fmt(r, m)[::3] + [f"{w[m].mean():.2f}"])
    md.append(pd.DataFrame(rows, columns=["구간", "매수보유", "원안 수익", "원안 낙폭", "브레이크 수익", "브레이크 낙폭", "평균 노출"]).to_markdown(index=False) + "\n")

    md.append("## 3. 기간별\n")
    rows = []
    for nm, s_, e_ in (("2001~2010", "2001-01-02", "2010-12-31"), ("2011~2024", "2011-01-01", "2024-12-30")):
        m = (d["dt"] >= s_) & (d["dt"] <= e_)
        rows.append([nm] + fmt(c, m) + fmt(r, m))
    md.append(pd.DataFrame(rows, columns=["기간", "원안 누적", "원안 변동성", "원안 샤프", "원안 낙폭",
                                          "브레이크 누적", "브레이크 변동성", "브레이크 샤프", "브레이크 낙폭"]).to_markdown(index=False) + "\n")

    md.append("## 4. 연도별 수익 (%)\n")
    rows = []
    for y, g in pd.DataFrame({"매수보유": bh, "원안": c, "브레이크": r, "w": w})[IS | OOS].groupby(lambda t: t.year):
        rows.append([y] + [f"{((1 + g[k]).prod() - 1) * 100:+.0f}" for k in ("매수보유", "원안", "브레이크")] + [f"{g['w'].mean():.2f}"])
    md.append(pd.DataFrame(rows, columns=["연도", "매수보유", "원안", "브레이크", "평균 노출"]).to_markdown(index=False) + "\n")

    md.append("## 5. VaR95 사후검증 (Kupiec)\n")
    rows = []
    for nm, m in (("학습", IS), ("검증", OOS)):
        for lab, s in (("원안", c), ("브레이크", r)):
            for mod in ("hist", "fhs"):
                n, x, _, p = R.kupiec(s[m], model=mod)
                rows.append([nm, lab, "과거분포" if mod == "hist" else "FHS", f"{x}/{n}", f"{n * 0.05:.0f}", f"{p:.2f}", "통과" if p >= 0.05 else "실패"])
    md.append(pd.DataFrame(rows, columns=["구간", "대상", "모형", "초과", "기대", "p", "판정"]).to_markdown(index=False) + "\n")

    E = (1 + R.apply_limits(d, c, True, False, True, P2)[0]).cumprod()
    dd = E / E.cummax() - 1
    md.append(f"② 낙폭 한도: 그림자 곡선 최저 {dd[IS].min():.1%} ({dd[IS].idxmin().date()}), ½선 아래 {int((dd <= P2['dd_half']).sum())}일\n")
    text = "\n".join(md)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
