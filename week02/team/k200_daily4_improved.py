"""팀원 전략(일봉 4전략 동일비중)에 수정안을 얹은 백테스트 — 원안 vs 수정안.

수정안 (근거: 문헌에 이미 있는 규칙. 이 데이터로 숫자를 고르지 않음)
  ① S1 변동성 스케일링 복원 — Moskowitz·Ooi·Pedersen(2012) 원식: 비중 = sign × min(1, σ_target / σ_22)
     σ_target = 20% (코스피200 24년 실현변동성 23.1%에 가까운 둥근 값), 레버리지 없음(상한 1)
  ② 조합 변동성 타깃 — Moreira·Muir(2017) 방식을 조합 전체에: 노출 = min(1, 10% / 조합 σ_22)
     10% = 팀원 학습구간 조합 변동성(11.9%)에 가까운 둥근 값
  ③ S3 제외 — 학습 근거가 가장 약한 전략(학습 샤프 0.33, 비용 87%p)을 빼고 ①②로 급락 방어가 되는지
  σ_22 는 전일까지의 22일 실현변동성(연율). 노출 변경에도 틱 비용을 물린다.

데이터: 팀원 CSV(k200_1d_continuous.csv, 2000~)가 있으면 그걸, 없으면 모의 서버 일봉(2021-12~).
구간: 팀원과 같게 학습 2001-01-02~2024-12-30 / 검증 2025-01-02~. (모의 데이터면 학습은 2023~2024만)

실행:
  python team/k200_daily4_improved.py                                  # 모의 서버 데이터
  python team/k200_daily4_improved.py --csv results/k200_1d_continuous.csv   # 팀원 데이터
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k200_daily4_replicate as K  # noqa: E402

S1_TARGET, COMBO_TARGET, WIN = 0.20, 0.10, 22


def load_any(path):
    if path and os.path.exists(path):
        raw = pd.read_csv(path)
        cols = {c.lower(): c for c in raw.columns}
        pick = lambda *names: next((cols[n] for n in names if n in cols), None)  # noqa: E731
        df = pd.DataFrame({
            "dt": pd.to_datetime(raw[pick("dt", "date", "ref_dt")].astype(str)),
            "open": raw[pick("open", "open_pric")], "high": raw[pick("high", "high_pric")],
            "low": raw[pick("low", "low_pric")], "close": raw[pick("close", "close_pric")],
            "diff": raw[pick("diff", "diff_rt")]})
        tmp = "team/_loaded_daily.csv"
        df.sort_values("dt").to_csv(tmp, index=False)
        return K.load(tmp), path
    return K.load(), "team/k200_demo_daily_with_diff.csv (모의 서버)"


def sigma22(r):
    return r.rolling(WIN).std(ddof=1) * np.sqrt(252)


def vt(target, sig):
    return np.minimum(1.0, target / sig.shift(1)).fillna(0.0)


def combo_vt(d, c, target=COMBO_TARGET):
    w = vt(target, sigma22(c))
    per = (K.TICK + K.FEE * d["open"]) / d["prev"]
    return c * w - (w - w.shift(1)).abs().fillna(0.0) * per, w


def build(d, p):
    nets, _ = K.run_case(d, p)
    t1 = K.s1(d, *p["s1"])
    n1v, _ = K.net_returns(d, t1 * vt(S1_TARGET, sigma22(d["ret"])))
    S = {k: nets[k] for k in ("S1", "S2", "S3", "S4")}
    orig = pd.DataFrame(S).mean(axis=1)
    s1v = pd.DataFrame({**S, "S1": n1v}).mean(axis=1)
    no3 = pd.DataFrame({k: S[k] for k in ("S1", "S2", "S4")}).mean(axis=1)
    no3_s1v = pd.DataFrame({"S1": n1v, "S2": S["S2"], "S4": S["S4"]}).mean(axis=1)
    out = {
        "원안: 4개 동일비중": orig,
        "① S1 변동성 스케일링": s1v,
        "② 조합 변동성 타깃": combo_vt(d, orig)[0],
        "①+②": combo_vt(d, s1v)[0],
        "S3 제외 (비교용)": no3,
        "③ S3 제외 + ①+②": combo_vt(d, no3_s1v)[0],
        "매수보유": nets["매수보유"],
    }
    return out, combo_vt(d, no3_s1v)[1]


def table(series, mask):
    rows = []
    for name, r in series.items():
        m = K.metrics(r[mask])
        rows.append([name, f"{m['누적'] * 100:+.1f}%", f"{m['CAGR'] * 100:+.1f}%", f"{m['변동성'] * 100:.1f}%",
                     f"{m['샤프']:.2f}", f"{m['MDD'] * 100:.1f}%"])
    return pd.DataFrame(rows, columns=["구성", "누적", "CAGR", "변동성", "샤프", "최대낙폭"])


def worst_years(series, d, mask):
    rows = []
    for name, r in series.items():
        rr = r[mask]
        yrs = (1 + rr).groupby(d.loc[mask, "dt"].dt.year).prod() - 1
        rows.append([name] + [f"{v * 100:+.1f}%" for v in yrs.values])
    return pd.DataFrame(rows, columns=["구성"] + [str(y) for y in yrs.index])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/k200_1d_continuous.csv")
    ap.add_argument("--is-start", default="2001-01-02")
    ap.add_argument("--is-end", default="2024-12-30")
    ap.add_argument("--oos-start", default="2025-01-02")
    ap.add_argument("--out", default="team/improved_results.md")
    a = ap.parse_args()

    d, src = load_any(a.csv)
    is_start = max(pd.Timestamp(a.is_start), d["dt"].min() + pd.Timedelta(days=400))   # 지표 준비기간
    IS = (d["dt"] >= is_start) & (d["dt"] <= a.is_end)
    OOS = d["dt"] >= a.oos_start
    md = [f"# 팀원 전략 + 수정안 백테스트\n", f"데이터: {src}  ({d['dt'].min().date()} ~ {d['dt'].max().date()}, {len(d)}일)",
          f"학습 {d.loc[IS, 'dt'].min().date()} ~ {d.loc[IS, 'dt'].max().date()} ({IS.sum()}일) · "
          f"검증 {d.loc[OOS, 'dt'].min().date()} ~ {d.loc[OOS, 'dt'].max().date()} ({OOS.sum()}일)\n",
          "수정안 규칙은 결과 전에 고정 (S1 σ목표 20%, 조합 σ목표 10%, 창 22일, 레버리지 없음). 숫자는 이 데이터로 고르지 않음.\n"]
    for case in ("기본", "튜닝"):
        series, w = build(d, K.PARAMS[case])
        md.append(f"## [{case}] 학습 구간\n")
        md.append(table(series, IS).to_markdown(index=False) + "\n")
        md.append(f"## [{case}] 검증 구간 (한 번만 확인)\n")
        md.append(table(series, OOS).to_markdown(index=False) + "\n")
        md.append(f"### [{case}] 연도별 수익 (검증)\n")
        md.append(worst_years(series, d, OOS).to_markdown(index=False) + "\n")
        md.append(f"③의 조합 노출(검증 구간): 평균 {w[OOS].mean():.2f}, 최소 {w[OOS].min():.2f}, "
                  f"노출 1.0 미만인 날 비율 {(w[OOS] < 1).mean():.0%}\n")
    text = "\n".join(md)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
