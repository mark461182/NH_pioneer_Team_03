"""선물 고유 위험 측정 — 증거금 위험 · 갭(오버나이트) 위험 · 스트레스 슬리피지 · 정수 계약 변환.
팀원 전략(튜닝 4전략 동일비중)에 리스크 레이어(①변동성 타깃 ②낙폭 한도 ③ES 한도)를 얹은 뒤 측정한다.

가정 (전부 보고서에 명시)
  증거금률   : 21% (KRX 2026-08-03 조정, 위탁·유지 구분 미확인 → 유지증거금률은 위탁의 2/3로 가정 = 14%)
  명목금액   : 지수 × 25만 원 (코스피200 선물 1계약)
  자본금     : 1계약 명목의 100% / 50% / 30% 세 경우 (레버리지 1배 / 2배 / 3.3배)
  스트레스   : 일중 변동폭(고가−저가)이 직전 60일 중앙값의 2배를 넘는 날은 슬리피지 5틱 (평소 1틱)
  미니선물   : 승수 5만 원 (1계약 = 코스피200 선물 0.2계약). 노출을 미니 계약 수로 반올림.

실행: python team/risk_futures.py [--csv 팀원CSV] → team/risk_futures_report.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k200_daily4_improved as KI  # noqa: E402
import k200_daily4_replicate as K  # noqa: E402
import risk_layer as R  # noqa: E402

MARGIN, MAINT, MULT, MINI_MULT = 0.21, 0.14, 250_000, 50_000
STRESS_TICKS = 5


def combo_exposure(d, p):
    nets, holds = K.run_case(d.reset_index(drop=True), p)
    idx = d.index
    nets = {k: pd.Series(v.to_numpy(), index=idx) for k, v in nets.items()}
    holds = {k: pd.Series(v.to_numpy(), index=idx) for k, v in holds.items()}
    combo = pd.DataFrame({k: nets[k] for k in ("S1", "S2", "S3", "S4")}).mean(axis=1)
    expo = pd.DataFrame({k: holds[k] for k in ("S1", "S2", "S3", "S4")}).mean(axis=1)   # 오늘 보유 비중(부호 포함)
    return combo, expo, nets


def margin_analysis(d, ret_series, expo_series, capital_ratio, OOS):
    """자본 = 1계약 명목 × capital_ratio. 일별 손익(원)과 증거금 대비 자본 상태."""
    px0 = d.loc[OOS, "prev"].iloc[0]
    capital0 = px0 * MULT * capital_ratio
    r = ret_series[OOS]
    eq = capital0 * (1 + r).cumprod()
    notional = (d.loc[OOS, "close"] * MULT * expo_series[OOS].abs())
    req = notional * MARGIN                            # 위탁증거금
    maint = notional * MAINT
    margin_call = (eq < maint)
    worst_ratio = (eq / req.replace(0, np.nan)).min()
    dd = (eq / eq.cummax() - 1).min()
    return dict(자본비율=f"{capital_ratio:.0%}", 레버리지=f"{1 / capital_ratio:.1f}배", 초기자본=f"{capital0 / 1e4:,.0f}만 원",
                최종자본=f"{eq.iloc[-1] / 1e4:,.0f}만 원", 최대낙폭_자본대비=f"{dd * 100:.1f}%",
                자본_대_위탁증거금_최소=f"{worst_ratio:.2f}" if not np.isnan(worst_ratio) else "-",
                추가증거금_발생일=int(margin_call.sum()),
                최악하루_원=f"{(r * eq.shift(1).fillna(capital0)).min() / 1e4:,.0f}만 원")


def gap_split(d, expo, OOS):
    gap = (d["open"] - d["prev"]) / d["prev"]
    intra = (d["close"] - d["open"]) / d["prev"]
    before = expo.shift(1).fillna(0.0)                 # 밤사이 들고 있던 비중 = 전날 보유
    hold = expo
    g, i = before * gap, hold * intra
    rows = []
    for name, s in (("밤사이(갭) 손익", g[OOS]), ("장중 손익", i[OOS]), ("합계", (g + i)[OOS])):
        rows.append(dict(구분=name, 연변동성=f"{s.std(ddof=1) * np.sqrt(252) * 100:.1f}%",
                         VaR95=f"{-np.quantile(s, 0.05) * 100:.2f}%", VaR99=f"{-np.quantile(s, 0.01) * 100:.2f}%",
                         ES97_5=f"{R.es_hist(s) * 100:.2f}%", 최악=f"{s.min() * 100:.2f}% ({s.idxmin().date()})"))
    share = g[OOS].var() / (g[OOS].var() + i[OOS].var())
    u = pd.DataFrame({"구분": ["기초자산(노출 1) 갭", "기초자산(노출 1) 장중"],
                      "연변동성": [f"{gap[OOS].std() * np.sqrt(252) * 100:.1f}%", f"{intra[OOS].std() * np.sqrt(252) * 100:.1f}%"],
                      "VaR95": [f"{-np.quantile(gap[OOS], .05) * 100:.2f}%", f"{-np.quantile(intra[OOS], .05) * 100:.2f}%"],
                      "VaR99": [f"{-np.quantile(gap[OOS], .01) * 100:.2f}%", f"{-np.quantile(intra[OOS], .01) * 100:.2f}%"],
                      "ES97_5": [f"{R.es_hist(gap[OOS]) * 100:.2f}%", f"{R.es_hist(intra[OOS]) * 100:.2f}%"],
                      "최악": [f"{gap[OOS].min() * 100:.2f}% ({gap[OOS].idxmin().date()})", f"{intra[OOS].min() * 100:.2f}% ({intra[OOS].idxmin().date()})"]})
    return pd.DataFrame(rows), u, share


def stress_slippage(d, p, OOS):
    """스트레스 날(일중 변동폭 > 60일 중앙값 × 2)에 슬리피지 5틱으로 다시 계산."""
    rng = d["high"] - d["low"]
    stress = (rng > rng.rolling(60).median().shift(1) * 2).fillna(False)
    base_tick = K.TICK
    nets, holds = K.run_case(d.reset_index(drop=True), p)
    # 비용 재계산: 원래 비용(1틱)에 스트레스 날 추가 4틱
    idx = d.index
    out = {}
    for k in ("S1", "S2", "S3", "S4"):
        h = pd.Series(holds[k].to_numpy(), index=idx)
        before = h.shift(1).fillna(0.0)
        extra = (h - before).abs() * ((STRESS_TICKS - 1) * base_tick / d["prev"]) * stress.astype(float)
        extra = extra.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[k] = pd.Series(nets[k].to_numpy(), index=idx) - extra
    combo_s = pd.DataFrame(out).mean(axis=1)
    combo = pd.DataFrame({k: pd.Series(nets[k].to_numpy(), index=idx) for k in ("S1", "S2", "S3", "S4")}).mean(axis=1)
    m0, m1 = K.metrics(combo[OOS]), K.metrics(combo_s[OOS])
    return pd.DataFrame([["평소 1틱", f"{m0['누적'] * 100:+.1f}%", f"{m0['샤프']:.2f}", f"{m0['MDD'] * 100:.1f}%"],
                         [f"스트레스 날 {STRESS_TICKS}틱 ({int(stress[OOS].sum())}일)", f"{m1['누적'] * 100:+.1f}%", f"{m1['샤프']:.2f}", f"{m1['MDD'] * 100:.1f}%"]],
                        columns=["슬리피지 가정", "누적", "샤프", "최대낙폭"]), combo, combo_s


def integer_contracts(d, combo, expo_w, OOS, capital_ratio):
    """노출(비중×한도계수)을 미니선물 정수 계약으로 반올림했을 때의 성과.
    손익은 K.net_returns 와 같은 방식(전일 보유×갭 + 당일 보유×장중 − 변경·롤오버 비용)으로 직접 계산한다."""
    px = d["prev"]
    capital = d.loc[OOS, "prev"].iloc[0] * MULT * capital_ratio
    mini = (expo_w * capital / (px * MINI_MULT)).round()
    hold = mini * px * MINI_MULT / capital                 # 실제 노출 비중
    before = hold.shift(1).fillna(0.0)
    gap = (d["open"] - d["prev"]) / d["prev"]
    intra = (d["close"] - d["open"]) / d["prev"]
    per = (K.TICK + K.FEE * d["open"]) / d["prev"]
    r = before * gap + hold * intra - (hold - before).abs() * per - d["roll"] * 2 * before.abs() * per
    frac_before = expo_w.shift(1).fillna(0.0)
    r_frac = frac_before * gap + expo_w * intra - (expo_w - frac_before).abs() * per - d["roll"] * 2 * frac_before.abs() * per
    m_f, m_i = K.metrics(r_frac[OOS]), K.metrics(r[OOS])
    return dict(자본비율=f"{capital_ratio:.0%}", 미니계약_평균=f"{mini[OOS].abs().mean():.1f}", 미니계약_최대=int(mini[OOS].abs().max()),
                노출0인날=int((mini[OOS] == 0).sum()), 소수비중_샤프=f"{m_f['샤프']:.2f}", 정수계약_샤프=f"{m_i['샤프']:.2f}",
                소수비중_낙폭=f"{m_f['MDD'] * 100:.1f}%", 정수계약_낙폭=f"{m_i['MDD'] * 100:.1f}%",
                소수비중_누적=f"{m_f['누적'] * 100:+.1f}%", 정수계약_누적=f"{m_i['누적'] * 100:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/k200_1d_continuous.csv")
    ap.add_argument("--oos-start", default="2025-01-02")
    ap.add_argument("--out", default="team/risk_futures_report.md")
    a = ap.parse_args()
    d, src = KI.load_any(a.csv)
    d = d.set_index("dt", drop=False)
    OOS = d["dt"] >= a.oos_start
    p = K.PARAMS["튜닝"]
    combo, expo, nets = combo_exposure(d, p)
    P2 = R.calibrate(combo, ~OOS)
    lim_r, w = R.apply_limits(d, combo, True, True, True, P2)
    expo_lim = expo * w

    md = [f"# 선물 고유 위험 측정 — 팀원 전략(튜닝 4전략) 원안 vs 리스크 레이어 적용\n",
          f"데이터 {src} · 검증 구간 {a.oos_start} ~ {d['dt'].max().date()} ({int(OOS.sum())}일)\n",
          f"가정: 증거금률 {MARGIN:.0%} (KRX 2026-08-03 조정값, 위탁·유지 구분 미확인), 유지증거금 {MAINT:.0%} (가정), "
          f"승수 {MULT:,}원, 미니 {MINI_MULT:,}원, 스트레스 날 슬리피지 {STRESS_TICKS}틱\n"]

    md.append("## 1. 증거금 위험 — 자본금을 얼마로 잡느냐에 따라 같은 전략이 다르게 보인다\n")
    for label, r, e in (("원안", combo, expo), ("리스크 레이어 적용", lim_r, expo_lim)):
        rows = [margin_analysis(d, r, e, cr, OOS) for cr in (1.0, 0.5, 0.3)]
        md.append(f"### {label}\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    md.append("자본÷위탁증거금이 1 아래로 가면 신규 주문 불가, 자본이 유지증거금 아래로 가면 추가 증거금(마진콜). "
              "원안 비중은 노출 1 기준이라 자본이 명목의 100%일 때는 레버리지 1배다.\n")

    md.append("## 2. 갭(오버나이트) 위험 — 손절이 닿지 않는 구간\n")
    for label, e in (("원안", expo), ("리스크 레이어 적용", expo_lim)):
        t, u, share = gap_split(d, e, OOS)
        md.append(f"### {label} (전략 손익 기준)\n\n" + t.to_markdown(index=False) + "\n")
        md.append(f"전략 손익 분산 중 밤사이 비중 {share:.0%}\n")
    md.append("### 기초자산 자체 (노출 1 기준)\n\n" + u.to_markdown(index=False) + "\n")
    md.append("일봉 전략은 포지션을 밤새 들고 가므로 손절이 있어도 갭만큼은 그대로 맞는다. "
              "장중 손절이 보호하는 건 장중 손익뿐이다. 밤사이 위험을 줄이는 방법은 노출을 줄이는 것(리스크 레이어)뿐이다.\n")

    md.append("## 3. 스트레스 슬리피지 — 급변일 체결은 1틱이 아니다\n")
    t, c0, c1 = stress_slippage(d, p, OOS)
    md.append(t.to_markdown(index=False) + "\n")
    m0, m1 = K.metrics(R.apply_limits(d, c0, True, True, True, P2)[0][OOS]), K.metrics(R.apply_limits(d, c1, True, True, True, P2)[0][OOS])
    md.append(f"리스크 레이어 적용 후: 평소 1틱 샤프 {m0['샤프']:.2f} / 낙폭 {m0['MDD'] * 100:.1f}%  →  스트레스 5틱 샤프 {m1['샤프']:.2f} / 낙폭 {m1['MDD'] * 100:.1f}%\n")

    md.append("## 4. 정수 계약 변환 — 소수 비중은 주문할 수 없다\n")
    rows = [integer_contracts(d, combo, expo_lim, OOS, cr) for cr in (1.0, 0.5, 0.3)]
    md.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    md.append("리스크 레이어 노출(0~1)을 미니코스피200 선물 계약 수로 반올림. 자본이 작을수록 계약 수가 0 아니면 1로 튀어 "
              "백테스트와 실제가 벌어진다.\n")

    md.append("## 5. 종합 — 측정값과 한도\n")
    final = R.measure_block("리스크 레이어 적용", lim_r[OOS])
    base = R.measure_block("원안", combo[OOS])
    md.append(pd.DataFrame([base, final]).T.rename(columns={0: "원안", 1: "리스크 레이어"}).to_markdown() + "\n")
    text = "\n".join(md)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
