"""README 요약용 그림 3장 → results/figs/6_epps.png, 7_bounce.png, 8_ou_noise.png"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent / "results"; F = OUT / "figs"; F.mkdir(exist_ok=True)

# 6. 봉 길이별 상관 (Epps)
Q = pd.read_csv(OUT / "quarterly_corr.csv")
fig, ax = plt.subplots(figsize=(10, 4))
for c, m in (("1분", "o"), ("5분", "s"), ("30분", "^"), ("일간", "D")):
    ax.plot(Q["분기"], Q[c], marker=m, label=c)
ax.set_ylim(0, 1); ax.set_ylabel("삼성–하이닉스 수익률 상관"); ax.legend(ncol=4, loc="lower right")
ax.set_title("같은 분기라도 봉이 짧을수록 상관이 낮게 잡힌다 (Epps 효과)"); ax.tick_params(axis="x", rotation=60); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(F / "6_epps.png", dpi=130); plt.close(fig)

# 7. 호가 튐: 1분 잔차 자기상관 + 30분 잔차 반전(바로 vs 1분 뒤)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(["1분 뒤", "2분 뒤", "5분 뒤"], [-0.268, -0.006, 0.008], color=["#c44", "#999", "#999"])
ax[0].axhline(0, color="k", lw=.8); ax[0].set_title("1분 잔차 자기상관 — 되돌림은 1분 안에 끝남"); ax[0].set_ylim(-0.3, 0.05)
ax[1].bar(["신호 봉 종가에 바로\n(불가능)", "1분 뒤 시가\n(실제)", "왕복 비용"], [1038, -160, -13154], color=["#7a7", "#c44", "#555"])
ax[1].axhline(0, color="k", lw=.8); ax[1].set_title("30분 잔차 반전, 거래당 원 (학습 2022~24)"); ax[1].set_ylabel("원")
fig.tight_layout(); fig.savefig(F / "7_bounce.png", dpi=130); plt.close(fig)

# 8. OU 보정 전후
S = json.loads((OUT / "ou_noise_summary.json").read_text(encoding="utf-8"))
ys = ["2022", "2023", "2024", "2025", "2026"]
B = {k: json.loads((OUT / f"ou_noise_backtest_{k}.json").read_text(encoding="utf-8")) for k in ("orig", "noise")}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
x = np.arange(len(ys)); w = .38
ax[0].bar(x - w / 2, [S[y]["반감기분_중앙_기존"] for y in ys], w, label="기존 AR(1)")
ax[0].bar(x + w / 2, [S[y]["반감기분_중앙_보정"] for y in ys], w, label="보정 (OU + 호가 튐)")
ax[0].set_xticks(x, ys); ax[0].set_ylabel("반감기 중앙값(분)"); ax[0].legend(); ax[0].set_title("반감기가 약 3배")
labs, o, n = [], [], []
for name in ("OU 시장가", "OU 시장가·추세필터"):
    for per in ("학습", "검증"):
        labs.append(f"{name.replace('OU ', '')}\n{per}"); o.append(B["orig"][name][per]["총비용전원"] / 1e4); n.append(B["noise"][name][per]["총비용전원"] / 1e4)
x = np.arange(len(labs))
ax[1].bar(x - w / 2, o, w, label="기존"); ax[1].bar(x + w / 2, n, w, label="보정")
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(x, labs, fontsize=8); ax[1].set_ylabel("비용 전 합계(만 원)"); ax[1].legend(); ax[1].set_title("비용 전 손익 (243일)")
fig.tight_layout(); fig.savefig(F / "8_ou_noise.png", dpi=130); plt.close(fig)
print("ok")
