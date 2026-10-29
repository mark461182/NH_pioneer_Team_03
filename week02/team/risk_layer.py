"""리스크 레이어 — 팀원 전략(일봉 4전략)의 일별 수익률에 위험 측정과 한도 규칙을 얹는다.

측정 (보고용)
  VaR 95/99 (과거분포 · 정규 · 코니시-피셔), ES 97.5 (과거분포), 왜도·첨도, 최악 하루, 최장 연속 손실일,
  최대낙폭과 회복 기간, VaR 사후검증(Kupiec: 초과 횟수가 기대치와 맞는가),
  공분산 기반 위험 기여도(동일비중일 때 각 전략이 조합 변동성의 몇 %를 만드는가)

한도 규칙 (모두 레버리지 없음 = 노출 ≤ 1)
  ① 변동성 타깃  : 노출 = min(1, 10% / 조합 σ22)
  ② 낙폭 한도    : 낙폭이 ½선 이하면 노출 ×0.5, 0선 이하면 0, 복귀선 위로 올라오면 1 (히스테리시스)
                   v2: 세 선을 학습 구간에서 계산(calibrate) — "연 변동성 10%·엣지 0인 전략이 1년 동안
                       운만으로 겪는 최대낙폭"의 분포(학습 구간 표준화 잔차 부트스트랩 1만 경로)에서
                       ½선 = 상위 10%(10년에 한 번), 0선 = 상위 1%(100년에 한 번 → 전략 고장 의심), 복귀선 = 중앙값
                   v2: 낙폭은 실제 계좌가 아니라 '①+③만 적용한 그림자 자산곡선'으로 잰다.
                       실제 계좌로 재면 노출 0일 때 회복할 수 없어 영원히 꺼지는 결함이 있음(v1)
  ③ ES 한도      : FHS-ES97.5(일) > 한도 이면 노출 = min(노출, 한도 / ES)
                   v2: 한도 = k × 10%/√252, k = 학습 구간 표준화 잔차의 ES97.5 ÷ 표준편차
                       ("①과 같은 위험 예산을 꼬리 단위로 환산" — ①과 ③이 같은 10%에서 나옴)
  v1(판단값: −10/−15/−8%, 2%)은 비교용으로 남김
                   FHS = 필터드 히스토리컬 시뮬레이션: EWMA(λ=0.94) 변동성으로 표준화한 과거 250일 잔차의 꼬리평균 × 오늘 σ
                   (Barone-Adesi·Giannopoulos·Vosper 1999; 과거분포 ES는 사후검증 실패 → 리스크_보완방안.md 1절)
  적용 순서 ①→②→③, 최종 노출 = 셋의 곱이 아니라 min (한도는 '가장 엄격한 것'이 결정)
  노출 변경에 틱 비용을 물린다.

실행:  python team/risk_layer.py [--csv 팀원CSV]   → team/risk_report.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k200_daily4_replicate as K  # noqa: E402
import k200_daily4_improved as KI  # noqa: E402

VOL_TARGET, DD_HALF, DD_ZERO, DD_RESUME, ES_LIMIT, ES_WIN = 0.10, -0.10, -0.15, -0.08, 0.02, 250
V1 = dict(es_limit=0.02, dd_half=-0.10, dd_zero=-0.15, dd_resume=-0.08, shadow=False, 근거="판단값")
N_PATHS, HORIZON, SEED = 10_000, 252, 0


def calibrate(c, is_mask, vol_target=None):
    """학습 구간(is_mask)만으로 ③ ES 한도와 ② 낙폭 선을 계산한다. 검증 구간은 보지 않는다."""
    vt = VOL_TARGET if vol_target is None else vol_target
    sig = ewma_sigma(c).shift(1)
    z = (c / sig).replace([np.inf, -np.inf], np.nan)[is_mask].dropna()
    z = z[z != 0]                                         # 전략이 아직 안 돌던 날 제외
    z = (z - z.mean()) / z.std(ddof=1)                    # 엣지 0, 분산 1로 정규화 → 꼬리 모양만 남김
    k = es_hist(z, 0.975)
    sd = vt / np.sqrt(252)
    rng = np.random.default_rng(SEED)
    paths = rng.choice(z.to_numpy(), size=(N_PATHS, HORIZON)) * sd
    E = np.cumprod(1 + paths, axis=1)
    mdd = (E / np.maximum.accumulate(E, axis=1) - 1).min(axis=1)
    return dict(es_limit=k * sd, dd_half=np.quantile(mdd, 0.10), dd_zero=np.quantile(mdd, 0.01),
                dd_resume=np.quantile(mdd, 0.50), shadow=True, k=k, n=len(z),
                근거=f"학습 구간 {len(z)}일 잔차, 부트스트랩 {N_PATHS:,}경로")


# ---------------------------------------------------------------- 측정

def var_es(r, a=0.95):
    r = r.dropna()
    hist_var = -np.quantile(r, 1 - a)
    mu, sd = r.mean(), r.std(ddof=1)
    z = st.norm.ppf(1 - a)
    norm_var = -(mu + z * sd)
    s, k = st.skew(r), st.kurtosis(r)                     # k = 초과첨도
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    cf_var = -(mu + zcf * sd)
    return hist_var, norm_var, cf_var


def es_hist(r, a=0.975):
    r = r.dropna()
    q = np.quantile(r, 1 - a)
    return -r[r <= q].mean()


LAMBDA = 0.94


def ewma_sigma(r, lam=LAMBDA):
    """RiskMetrics EWMA 일변동성. 반환값 σ_t 는 t일까지의 정보로 계산(t+1일 예측용)."""
    return np.sqrt(r.fillna(0).pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def fhs_series(r, a=0.975, win=ES_WIN, kind="es"):
    """FHS VaR/ES (전일까지 정보로 오늘 예측). z = r/σ_{t-1} 표준화 잔차의 과거 win일 분위수 × σ_{t-1}."""
    idx0 = r.index
    r = r.dropna()
    sig = ewma_sigma(r).shift(1)
    z = (r / sig).replace([np.inf, -np.inf], np.nan)
    if kind == "es":
        qz = z.rolling(win).apply(lambda a_: es_hist(pd.Series(a_), a), raw=True)
    else:
        qz = -z.rolling(win).quantile(1 - a)
    return (qz.shift(1) * sig).reindex(idx0)


def kupiec(r, a=0.95, win=250, model="hist"):
    """굴러가는 VaR(전일까지 정보)로 다음 날을 예측 → 초과 횟수 사후검증. model: hist | fhs"""
    r = r.dropna()
    if model == "fhs":
        var = fhs_series(r, a, win, kind="var")
    else:
        var = -r.rolling(win).quantile(1 - a).shift(1)
    ok = var.notna()
    exc = (r[ok] < -var[ok]).sum()
    n = int(ok.sum())
    if n == 0:
        return n, exc, np.nan, np.nan
    p = 1 - a
    x = exc
    # Kupiec LR_uc
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = -2 * (((n - x) * np.log(1 - p) + x * np.log(p)) - ((n - x) * np.log(1 - x / n) + x * np.log(x / n))) if 0 < x < n else np.nan
    pval = 1 - st.chi2.cdf(lr, 1) if not np.isnan(lr) else np.nan
    return n, int(exc), lr, pval


def drawdown_stats(r):
    E = (1 + r.fillna(0)).cumprod()
    peak = E.cummax()
    dd = E / peak - 1
    mdd = dd.min()
    t_trough = dd.idxmin()
    t_peak = E.loc[:t_trough].idxmax()
    after = E.loc[t_trough:]
    rec = after[after >= peak.loc[t_trough]]
    t_rec = rec.index[0] if len(rec) else None
    return mdd, t_peak, t_trough, t_rec, dd


def risk_contrib(parts, w=None):
    X = parts.dropna()
    w = np.full(X.shape[1], 1 / X.shape[1]) if w is None else np.asarray(w)
    C = np.cov(X.T, ddof=1)
    pv = w @ C @ w
    mrc = C @ w
    rc = w * mrc / pv
    return pd.Series(rc, index=X.columns), np.sqrt(pv * 252), pd.DataFrame(np.corrcoef(X.T), index=X.columns, columns=X.columns)


def streaks(r):
    neg = (r < 0).astype(int)
    best, cur = 0, 0
    for v in neg:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------- 한도 규칙

def apply_limits(d, c, use_vol=True, use_dd=True, use_es=True, p=None):
    """c: 조합 일수익(비용 차감, 노출 1). p: 한도 숫자(None이면 V1). 반환: 한도 적용 수익, 노출 시계열."""
    p = V1 if p is None else p
    n = len(c)
    sig = (c.rolling(22).std(ddof=1) * np.sqrt(252)).shift(1)
    es = fhs_series(c, 0.975, ES_WIN, kind="es")       # FHS-ES (전일까지 정보)
    out = np.zeros(n)
    w = np.ones(n)
    E, peak, dd_state = 1.0, 1.0, 1.0           # dd_state: 1 / 0.5 / 0 (히스테리시스 상태)
    Es, peak_s = 1.0, 1.0                       # 그림자 자산곡선 (①+③만 적용)
    per = ((K.TICK + K.FEE * d["open"]) / d["prev"]).to_numpy()
    per = np.where(np.isfinite(per), per, 0.0)          # 첫 행(전일 종가 없음) 방어
    cv = c.to_numpy()
    for t in range(n):
        f = 1.0
        if use_vol and not np.isnan(sig.iloc[t]) and sig.iloc[t] > 0:
            f = min(f, VOL_TARGET / sig.iloc[t])
        if use_es and not np.isnan(es.iloc[t]) and es.iloc[t] > p["es_limit"]:
            f = min(f, p["es_limit"] / es.iloc[t])
        f13 = f
        if use_dd:
            dd = (Es / peak_s - 1) if p["shadow"] else (E / peak - 1)
            if dd <= p["dd_zero"]:
                dd_state = 0.0
            elif dd <= p["dd_half"]:
                dd_state = min(dd_state, 0.5)
            elif dd > p["dd_resume"]:
                dd_state = 1.0
            f = min(f, dd_state)
        w[t] = f
        cost = abs(f - (w[t - 1] if t else 0.0)) * per[t]
        out[t] = f * cv[t] - cost
        E *= 1 + out[t]
        peak = max(peak, E)
        Es *= 1 + f13 * cv[t]
        peak_s = max(peak_s, Es)
    return pd.Series(out, index=c.index), pd.Series(w, index=c.index)


# ---------------------------------------------------------------- 보고

def measure_block(name, r):
    r = r.dropna()
    hv95, nv95, cf95 = var_es(r, 0.95)
    hv99, nv99, cf99 = var_es(r, 0.99)
    es = es_hist(r, 0.975)
    mdd, tp, tt, tr, _ = drawdown_stats(r)
    n, exc, lr, p = kupiec(r)
    nf, excf, _, pf = kupiec(r, model="fhs")
    return dict(구성=name, 일수=len(r), 연변동성=f"{r.std(ddof=1) * np.sqrt(252) * 100:.1f}%",
                왜도=f"{st.skew(r):+.2f}", 초과첨도=f"{st.kurtosis(r):.1f}",
                VaR95_과거=f"{hv95 * 100:.2f}%", VaR95_정규=f"{nv95 * 100:.2f}%", VaR95_CF=f"{cf95 * 100:.2f}%",
                VaR99_과거=f"{hv99 * 100:.2f}%", ES97_5=f"{es * 100:.2f}%", 최악하루=f"{r.min() * 100:.2f}%",
                최장연속손실=streaks(r), 최대낙폭=f"{mdd * 100:.1f}%",
                낙폭기간=f"{tp.date()}→{tt.date()}" + (f"→회복 {tr.date()}" if tr is not None else " (미회복)"),
                VaR95_사후검증_과거분포=f"초과 {exc}/{n} (기대 {n * 0.05:.0f}), p={p:.2f}" if n else "-",
                VaR95_사후검증_FHS=f"초과 {excf}/{nf} (기대 {nf * 0.05:.0f}), p={pf:.2f}" if nf else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/k200_1d_continuous.csv")
    ap.add_argument("--oos-start", default="2025-01-02")
    ap.add_argument("--out", default="team/risk_report.md")
    a = ap.parse_args()
    d, src = KI.load_any(a.csv)
    d = d.set_index("dt", drop=False)
    OOS = d["dt"] >= a.oos_start
    md = [f"# 리스크 레이어 보고\n", f"데이터 {src} ({d['dt'].min().date()} ~ {d['dt'].max().date()})  · 검증 구간 {a.oos_start} ~\n",
          f"한도 규칙: 변동성 타깃 {VOL_TARGET:.0%} · ES97.5(FHS: EWMA λ={LAMBDA}, 잔차 {ES_WIN}일) · 낙폭 한도. "
          "v2 = ③·②의 숫자를 학습 구간(검증 시작 전)에서 계산 + 낙폭은 그림자 자산곡선으로 측정. v1 = 판단값(비교용)\n"]
    for case in ("기본", "튜닝"):
        nets, holds = K.run_case(d.reset_index(drop=True), K.PARAMS[case])
        nets = {k: pd.Series(v.to_numpy(), index=d.index) for k, v in nets.items()}
        parts = pd.DataFrame({k: nets[k] for k in ("S1", "S2", "S3", "S4")})
        combo = parts.mean(axis=1)
        rc, pvol, corr = risk_contrib(parts[OOS])
        md.append(f"## [{case}] 1. 위험 기여도 (검증 구간, 동일비중)\n")
        md.append("조합 연변동성 " + f"{pvol * 100:.1f}%  ·  각 전략이 조합 분산에 기여한 비율:\n")
        md.append(pd.DataFrame({"전략": rc.index, "위험기여": [f"{v:.0%}" for v in rc.values],
                                "단독 연변동성": [f"{parts[k][OOS].std(ddof=1) * np.sqrt(252) * 100:.1f}%" for k in rc.index]}).to_markdown(index=False) + "\n")
        md.append("상관 (검증):\n\n" + corr.round(2).to_markdown() + "\n")
        P2 = calibrate(combo, ~OOS)
        md.append(f"### [{case}] v2 한도 숫자 (학습 구간에서 계산, {P2['근거']})\n")
        md.append(pd.DataFrame([["③ ES 한도", f"{V1['es_limit']:.2%}", f"{P2['es_limit']:.2%}", f"k = {P2['k']:.2f} × 10%/√252"],
                                ["② ½선", f"{V1['dd_half']:.1%}", f"{P2['dd_half']:.1%}", "운만으로 1년 최대낙폭 상위 10%"],
                                ["② 0선", f"{V1['dd_zero']:.1%}", f"{P2['dd_zero']:.1%}", "상위 1%"],
                                ["② 복귀선", f"{V1['dd_resume']:.1%}", f"{P2['dd_resume']:.1%}", "중앙값"],
                                ["② 낙폭 측정", "실제 계좌", "그림자 곡선(①+③)", "노출 0이면 영구 정지되는 결함 제거"]],
                               columns=["항목", "v1 (판단값)", "v2 (학습 구간)", "v2 근거"]).to_markdown(index=False) + "\n")
        # 한도 적용 변형
        variants = {"원안 (노출 1)": combo,
                    "① 변동성 타깃": apply_limits(d, combo, True, False, False, P2)[0],
                    "② 낙폭 한도": apply_limits(d, combo, False, True, False, P2)[0],
                    "③ ES 한도": apply_limits(d, combo, False, False, True, P2)[0],
                    "①+②+③ (v2)": apply_limits(d, combo, True, True, True, P2)[0],
                    "①+②+③ (v1, 비교용)": apply_limits(d, combo, True, True, True, V1)[0],
                    "매수보유": nets["매수보유"]}
        md.append(f"## [{case}] 2. 한도 규칙 적용 — 성과 (검증 구간)\n")
        rows = []
        for name, r in variants.items():
            m = K.metrics(r[OOS])
            rows.append([name, f"{m['누적'] * 100:+.1f}%", f"{m['변동성'] * 100:.1f}%", f"{m['샤프']:.2f}", f"{m['MDD'] * 100:.1f}%"])
        md.append(pd.DataFrame(rows, columns=["구성", "누적", "변동성", "샤프", "최대낙폭"]).to_markdown(index=False) + "\n")
        _, w_all = apply_limits(d, combo, True, True, True, P2)
        md.append(f"①+②+③ (v2) 노출: 평균 {w_all[OOS].mean():.2f}, 0인 날 {(w_all[OOS] == 0).sum()}일, 1 미만인 날 {(w_all[OOS] < 1).mean():.0%}\n")
        md.append(f"## [{case}] 3. 위험 측정 (검증 구간)\n")
        rows = [measure_block(n, r[OOS]) for n, r in variants.items()]
        md.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    text = "\n".join(md)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
