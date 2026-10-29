"""리스크 레이어 — 한 번에 전부 계산해서 results/risk_layer_* 로 저장한다.

입력 (week02/results/)
  k200_1d_continuous.csv             팀원 fetch_daily.py 산출 연결 일봉 (date, open, high, low, close, diff, ...)
  strategy_returns_tuned.csv (선택)  4전략 비용 차감 일수익률 (dt, S1, S2, S3, S4). 있으면 그대로 쓰고,
                                     없으면 k200_daily4_replicate.py(팀원 optimize_daily.py 튜닝 1위 규칙·인자를
                                     재구현한 것)로 만들어 results/risk_layer_strategy_returns.csv 에 저장한다.
                                     재현 확인: 팀원 문서의 학습·검증 표(기본·튜닝, S1~S4)와 소수점까지 일치.

출력 (week02/results/)
  risk_layer_daily.csv       날짜, 원안 수익, ①②③ 각 노출, 최종 노출, 결정한 규칙, 레이어 후 수익, 그림자 낙폭
  risk_layer_summary.json    보고서 표의 숫자 전부
  risk_layer_strategy_returns.csv   (재구현으로 만든 경우) 4전략 일수익률
  risk_layer_*.png           누적·낙폭, 노출, 연도별, 위기 구간, 민감도

규칙·수식: risk_layer.py (①②③, FHS, 부트스트랩 임계값, Kupiec), risk_futures.py (증거금·갭·슬리피지·정수 계약)
키·계좌 정보는 쓰지 않는다 (시세 파일만 읽음).

실행: python team/risk_layer_run.py [--csv results/k200_1d_continuous.csv] [--returns results/strategy_returns_tuned.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k200_daily4_replicate as K  # noqa: E402
import k200_daily4_improved as KI  # noqa: E402
import risk_layer as R  # noqa: E402
import risk_futures as F  # noqa: E402

IS_START, IS_END, OOS_START = "2001-01-02", "2024-12-30", "2025-01-02"
SENS_VOL, SENS_ES, SENS_DD = (0.08, 0.10, 0.12, 0.15), (0.015, 0.02, 0.03), ((-0.10, -0.15, -0.08), (-0.07, -0.12, -0.05), (-0.15, -0.20, -0.12))
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", gray="#8a8984", text="#0b0b0b", sub="#52514e", grid="#e6e5e1")


def metrics(r):
    m = K.metrics(r)
    return dict(누적=round(float(m["누적"]), 4), CAGR=round(float(m["CAGR"]), 4), 변동성=round(float(m["변동성"]), 4),
                샤프=round(float(m["샤프"]), 3), 최대낙폭=round(float(m["MDD"]), 4), 최악하루=round(float(r.min()), 4), 일수=int(m["일수"]))


def load_returns(d, path, out_dir):
    if path and os.path.exists(path):
        s = pd.read_csv(path, parse_dates=["dt"]).set_index("dt")[["S1", "S2", "S3", "S4"]].reindex(d.index).fillna(0.0)
        bh = K.net_returns(d, pd.Series(1.0, index=d.index))[0]
        return s, bh, f"{path} (팀원 제공)"
    nets, holds = K.run_case(d.reset_index(drop=True), K.PARAMS["튜닝"])
    s = pd.DataFrame({k: pd.Series(nets[k].to_numpy(), index=d.index) for k in ("S1", "S2", "S3", "S4")})
    bh = pd.Series(nets["매수보유"].to_numpy(), index=d.index)
    s.to_csv(os.path.join(out_dir, "risk_layer_strategy_returns.csv"), index_label="dt")
    return s, bh, f"k200_daily4_replicate.py 재구현 (팀원 optimize_daily.py 튜닝 1위: {K.PARAMS['튜닝']})"


def decide_rule(w, f1, f2, f3):
    if w >= 1:
        return "없음"
    if f3 < f1 and np.isclose(w, f3):
        return "③ES"
    if np.isclose(w, f1):
        return "①변동성"
    return "②낙폭"


def sensitivity(d, c, OOS):
    saved = (R.VOL_TARGET,)
    rows = []
    for vt in SENS_VOL:
        for es in SENS_ES:
            for dd in SENS_DD:
                R.VOL_TARGET = vt
                p = dict(es_limit=es, dd_half=dd[0], dd_zero=dd[1], dd_resume=dd[2], shadow=True)
                r, w = R.apply_limits(d, c, True, True, True, p)
                m = K.metrics(r[OOS])
                rows.append(dict(변동성목표=vt, ES한도=es, 낙폭반=dd[0], 낙폭0=dd[1], 복귀=dd[2],
                                 검증샤프=None if np.isnan(m["샤프"]) else round(float(m["샤프"]), 3),
                                 검증낙폭=round(float(m["MDD"]), 4), 검증누적=round(float(m["누적"]), 4)))
    R.VOL_TARGET = saved[0]
    return rows


def charts(d, c, r, w, bh, rule, IS, OOS, sens, out_dir, P2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": PAL["grid"], "axes.labelcolor": PAL["sub"],
                         "xtick.color": PAL["sub"], "ytick.color": PAL["sub"], "grid.color": PAL["grid"], "axes.grid": True, "grid.linewidth": 0.6})
    m = IS | OOS
    E = pd.DataFrame({"매수보유": (1 + bh[m]).cumprod(), "원안 (노출 1)": (1 + c[m]).cumprod(), "리스크 레이어": (1 + r[m]).cumprod()})
    dd = E / E.cummax() - 1
    col = {"매수보유": PAL["gray"], "원안 (노출 1)": PAL["blue"], "리스크 레이어": PAL["orange"]}

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for k in E:
        ax[0].plot(E.index, E[k], color=col[k], lw=1.6 if k != "매수보유" else 1.2, label=k)
        ax[1].plot(dd.index, dd[k] * 100, color=col[k], lw=1.4 if k != "매수보유" else 1.0)
    ax[0].set_yscale("log"); ax[0].set_title("누적 수익 (로그) — 원안 vs 리스크 레이어", color=PAL["text"]); ax[0].legend(frameon=False)
    ax[0].axvspan(pd.Timestamp(OOS_START), E.index[-1], color=PAL["yellow"], alpha=0.08, lw=0)
    ax[1].set_title("고점 대비 낙폭 (%)", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "risk_layer_equity.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))
    vol = (c.rolling(22).std(ddof=1) * np.sqrt(252))[m]
    ax[0].plot(vol.index, vol * 100, color=PAL["blue"], lw=1.2); ax[0].axhline(R.VOL_TARGET * 100, color=PAL["orange"], ls="--", lw=1)
    ax[0].set_title(f"원안 조합의 22일 실현변동성 (%) — 점선 = 목표 {R.VOL_TARGET:.0%}", color=PAL["text"])
    ax[1].fill_between(w[m].index, w[m], 0, color=PAL["orange"], alpha=0.5, lw=0); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("리스크 레이어 노출 (0~1)", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "risk_layer_exposure.png"), dpi=130); plt.close(fig)

    yr = pd.DataFrame({"원안": c[m], "리스크 레이어": r[m]}).groupby(lambda t: t.year).apply(lambda g: (1 + g).prod() - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(yr)); bw = 0.4
    ax.bar(x - bw / 2, yr["원안"], bw, color=PAL["blue"], label="원안"); ax.bar(x + bw / 2, yr["리스크 레이어"], bw, color=PAL["orange"], label="리스크 레이어")
    ax.set_xticks(x); ax.set_xticklabels(yr.index, rotation=45); ax.axhline(0, color=PAL["sub"], lw=0.8); ax.legend(frameon=False)
    ax.set_title("연도별 수익률 (%)", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "risk_layer_yearly.png"), dpi=130); plt.close(fig)

    S = pd.DataFrame(sens)
    fig, axs = plt.subplots(1, len(SENS_DD), figsize=(12, 3.8), sharey=True)
    for ax, dd_ in zip(axs, SENS_DD):
        g = S[S["낙폭반"] == dd_[0]].pivot(index="변동성목표", columns="ES한도", values="검증샤프")
        im = ax.imshow(g.values.astype(float), cmap="Blues", vmin=1.5, vmax=3.0, aspect="auto")
        ax.set_xticks(range(len(g.columns))); ax.set_xticklabels([f"{v:.1%}" for v in g.columns]); ax.set_yticks(range(len(g.index))); ax.set_yticklabels([f"{v:.0%}" for v in g.index])
        ax.set_xlabel("ES 한도"); ax.set_title(f"낙폭 {dd_[0]:.0%}/{dd_[1]:.0%}/{dd_[2]:.0%}", color=PAL["text"], fontsize=10); ax.grid(False)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]; ax.text(j, i, "정지" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8, color=PAL["text"] if (np.isnan(v) or v < 2.5) else "white")
    axs[0].set_ylabel("변동성 목표"); fig.suptitle("민감도 — 검증 구간 샤프 (한도 숫자 36가지)", color=PAL["text"])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "risk_layer_sensitivity.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/k200_1d_continuous.csv")
    ap.add_argument("--returns", default="results/strategy_returns_tuned.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d, src = KI.load_any(a.csv)
    d = d.set_index("dt", drop=False)
    IS = (d["dt"] >= IS_START) & (d["dt"] <= IS_END)
    OOS = d["dt"] >= OOS_START
    S, bh, src_ret = load_returns(d, a.returns, a.out_dir)
    c = S.mean(axis=1)

    # ---- 한도 숫자 (학습 구간) · 노출 · 수익
    P2 = R.calibrate(c, ~OOS)
    r, w = R.apply_limits(d, c, True, True, True, P2)
    _, w1 = R.apply_limits(d, c, True, False, False, P2)
    _, w2 = R.apply_limits(d, c, False, True, False, P2)
    _, w3 = R.apply_limits(d, c, False, False, True, P2)
    rv1, wv1 = R.apply_limits(d, c, True, True, True, R.V1)
    r1, r3 = R.apply_limits(d, c, True, False, False, P2)[0], R.apply_limits(d, c, False, False, True, P2)[0]
    r2 = R.apply_limits(d, c, False, True, False, P2)[0]
    Es = (1 + R.apply_limits(d, c, True, False, True, P2)[0]).cumprod()
    shadow_dd = Es / Es.cummax() - 1
    rule = pd.Series([decide_rule(w.iloc[t], w1.iloc[t], w2.iloc[t], w3.iloc[t]) for t in range(len(w))], index=w.index)
    daily = pd.DataFrame({"원안수익": c, "노출_①변동성": w1, "노출_②낙폭": w2, "노출_③ES": w3, "최종노출": w, "결정규칙": rule,
                          "레이어후수익": r, "그림자낙폭": shadow_dd, "구간": np.where(OOS, "검증", np.where(IS, "학습", "준비"))})
    daily.round(6).to_csv(os.path.join(a.out_dir, "risk_layer_daily.csv"), index_label="dt")

    # ---- 요약
    def block(mask):
        out = {"성과": {k: metrics(v[mask]) for k, v in (("원안", c), ("①변동성타깃", r1), ("②낙폭한도", r2), ("③ES한도", r3), ("①+②+③_v2", r), ("①+②+③_v1판단값", rv1), ("매수보유", bh))}}
        out["위험측정"] = {k: R.measure_block(k, v[mask]) for k, v in (("원안", c), ("①+②+③_v2", r))}
        kp = {}
        for lab, s in (("원안", c), ("리스크레이어", r)):
            for mod in ("hist", "fhs"):
                n, x, lr, p = R.kupiec(s[mask], model=mod)
                kp[f"{lab}_{'과거분포' if mod == 'hist' else 'FHS'}"] = dict(초과=int(x), 일수=int(n), 기대=round(n * 0.05, 1), p=None if np.isnan(p) else round(float(p), 3), 판정="통과" if p >= 0.05 else "실패")
        out["VaR95_사후검증"] = kp
        rc = rule[mask].value_counts(normalize=True)
        out["결정규칙_비율"] = {k: round(float(rc.get(k, 0.0)), 3) for k in ("③ES", "①변동성", "②낙폭", "없음")}
        out["노출"] = dict(평균=round(float(w[mask].mean()), 3), 최소=round(float(w[mask].min()), 3), 노출0일수=int((w[mask] == 0).sum()), 노출1미만비율=round(float((w[mask] < 1).mean()), 3))
        out["전략별"] = {k: dict(원안=metrics(S[k][mask]), 레이어후=metrics((S[k] * w)[mask])) for k in S}
        corr = S[mask].corr().round(3)
        rcb, pvol, _ = R.risk_contrib(S[mask])
        out["상관"] = corr.to_dict(); out["위험기여"] = {k: round(float(v), 3) for k, v in rcb.items()}
        per = ((K.TICK + K.FEE * d["open"]) / d["prev"]).replace([np.inf, -np.inf], 0).fillna(0)
        out["레이어_추가비용_누적"] = round(float(((w - w.shift(1)).abs() * per)[mask].sum()), 5)
        return out

    summ = {"데이터": src, "일수익률_출처": src_ret, "기간": dict(데이터=[str(d["dt"].min().date()), str(d["dt"].max().date())], 학습=[IS_START, IS_END], 검증=[OOS_START, str(d["dt"].max().date())]),
            "한도숫자": {"v2_학습구간계산": {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else (v if isinstance(v, str) else bool(v)))) for k, v in P2.items()},
                      "v1_판단값": {k: v for k, v in R.V1.items() if k != "근거"},
                      "공통": dict(변동성목표=R.VOL_TARGET, 창=22, EWMA_lambda=R.LAMBDA, FHS창=R.ES_WIN, ES수준=0.975, 부트스트랩=dict(경로=R.N_PATHS, 기간일=R.HORIZON, seed=R.SEED), 비용=dict(틱=K.TICK, 수수료율=K.FEE)),
                      "v2_설명": "ES한도 = k×변동성목표/√252, k=학습구간 표준화잔차 ES97.5/표준편차. 낙폭선 = 엣지0·연10% 전략의 1년 최대낙폭 부트스트랩 분포 상위10%(½)·1%(0)·중앙값(복귀). 낙폭은 ①+③만 적용한 그림자 자산곡선으로 측정"},
            "검증구간": block(OOS), "학습구간": block(IS)}
    z = wv1[wv1 == 0]
    summ["v1_영구정지"] = dict(노출0일수=int(len(z)), 시작=str(z.index.min().date()) if len(z) else None, 끝=str(z.index.max().date()) if len(z) else None)
    summ["②낙폭한도_작동"] = dict(그림자낙폭최저=round(float(shadow_dd[IS | OOS].min()), 4), 날짜=str(shadow_dd[IS | OOS].idxmin().date()), 반선이하일수=int((shadow_dd <= P2["dd_half"]).sum()), 영선이하일수=int((shadow_dd <= P2["dd_zero"]).sum()))
    crisis = {}
    for nm, s_, e_ in (("2002-04~2003-03", "2002-04-01", "2003-03-31"), ("2008", "2008-01-01", "2008-12-31"), ("2011H2", "2011-07-01", "2011-12-31"), ("2020-02~04", "2020-02-01", "2020-04-30"), ("2026-06~10", "2026-06-22", str(d["dt"].max().date()))):
        mm = (d["dt"] >= s_) & (d["dt"] <= e_)
        if mm.sum():
            crisis[nm] = dict(매수보유=metrics(bh[mm]), 원안=metrics(c[mm]), 레이어=metrics(r[mm]), 평균노출=round(float(w[mm].mean()), 3))
    summ["위기구간"] = crisis
    yrs = {}
    for y, g in pd.DataFrame({"매수보유": bh, "원안": c, "레이어": r, "w": w})[IS | OOS].groupby(lambda t: t.year):
        yrs[int(y)] = dict(매수보유=round(float((1 + g["매수보유"]).prod() - 1), 4), 원안=round(float((1 + g["원안"]).prod() - 1), 4), 레이어=round(float((1 + g["레이어"]).prod() - 1), 4), 평균노출=round(float(g["w"].mean()), 3))
    summ["연도별"] = yrs

    # ---- 선물 고유 (검증 구간)
    _, expo, _ = F.combo_exposure(d, K.PARAMS["튜닝"])
    expo_lim = expo * w
    summ["증거금"] = {"가정": dict(증거금률=F.MARGIN, 유지증거금률=F.MAINT, 승수=F.MULT, 미니승수=F.MINI_MULT),
                   "원안": [F.margin_analysis(d, c, expo, cr, OOS) for cr in (1.0, 0.5, 0.3)], "레이어": [F.margin_analysis(d, r, expo_lim, cr, OOS) for cr in (1.0, 0.5, 0.3)]}
    gap = {}
    for lab, e in (("원안", expo), ("레이어", expo_lim)):
        t, u, share = F.gap_split(d, e, OOS)
        gap[lab] = dict(표=t.to_dict(orient="records"), 밤사이분산비중=round(float(share), 3))
    gap["기초자산"] = u.to_dict(orient="records")
    summ["갭위험"] = gap
    t, c0, c1 = F.stress_slippage(d, K.PARAMS["튜닝"], OOS)
    m0, m1 = K.metrics(R.apply_limits(d, c0, True, True, True, P2)[0][OOS]), K.metrics(R.apply_limits(d, c1, True, True, True, P2)[0][OOS])
    summ["스트레스슬리피지"] = dict(틱=F.STRESS_TICKS, 원안=t.to_dict(orient="records"), 레이어=dict(평소샤프=round(float(m0["샤프"]), 3), 스트레스샤프=round(float(m1["샤프"]), 3), 평소낙폭=round(float(m0["MDD"]), 4), 스트레스낙폭=round(float(m1["MDD"]), 4)))
    summ["정수계약"] = [F.integer_contracts(d, c, expo_lim, OOS, cr) for cr in (1.0, 0.5, 0.3)]

    # ---- 민감도 36가지
    sens = sensitivity(d, c, OOS)
    ok = [s["검증샤프"] for s in sens if s["검증샤프"] is not None]
    summ["민감도36"] = dict(정상계산=len(ok), 영구정지=36 - len(ok), 검증샤프_범위=[round(min(ok), 3), round(max(ok), 3)],
                        검증낙폭_범위=[round(min(s["검증낙폭"] for s in sens if s["검증샤프"] is not None), 4), round(max(s["검증낙폭"] for s in sens if s["검증샤프"] is not None), 4)],
                        주="변동성목표 4 × ES한도 3 × 낙폭선 3 = 36가지, 낙폭은 그림자 곡선으로 측정(v2 방식). 검증 구간 샤프·낙폭", 표=sens)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o
    with open(os.path.join(a.out_dir, "risk_layer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(clean(summ), f, ensure_ascii=False, indent=1)
    if not a.no_charts:
        charts(d, c, r, w, bh, rule, IS, OOS, sens, a.out_dir, P2)
    print(f"데이터 {src}\n일수익률 {src_ret}\n한도 v2: ES {P2['es_limit']:.2%} (k={P2['k']:.2f}) · 낙폭 {P2['dd_half']:.1%}/{P2['dd_zero']:.1%}/{P2['dd_resume']:.1%}")
    for nm in ("학습구간", "검증구간"):
        s = summ[nm]["성과"]
        print(f"[{nm}] 원안 샤프 {s['원안']['샤프']} 낙폭 {s['원안']['최대낙폭']:.1%} | 레이어 샤프 {s['①+②+③_v2']['샤프']} 낙폭 {s['①+②+③_v2']['최대낙폭']:.1%} | 결정규칙 {summ[nm]['결정규칙_비율']}")
    print(f"저장: {a.out_dir}/risk_layer_daily.csv, risk_layer_summary.json" + ("" if a.no_charts else ", risk_layer_*.png"))


if __name__ == "__main__":
    main()
