"""4주차 리스크 레이어 — 페어 한 번에 몇 「단위」 를 걸지 정함 (1단위 = 하이닉스 1계약 + β 로 맞춘 삼성 계약).

2주차 리스크 레이어(team/risk_layer.py)와 같은 생각을 페어에 옮김:
신호·진입선은 건드리지 않고, **크기만** 정함. 규칙 셋 중 가장 엄격한 것을 따르고, 모두 **그 거래 전까지의 정보로만** 정함.

    ① ES 예산    한 거래의 꼬리 손실 예상치(ES97.5, 보유시간 H분) × 단위 수 ≤ 거래당 예산
                 단위당 손실 = max( ES 손실 , 손절선까지 손실 ) + 왕복 비용     ← 손절이 있어도 갭으로 넘을 수 있어 큰 쪽
    ② 변동성 국면 오늘 σ_eq 가 지난 20일 중앙값의 1.5배 넘으면 단위 수 절반   ← OU 추정이 흔들리는 날
    ③ 낙폭 한도   실현 누적손익이 고점 대비 −150만 원 → 절반, −250만 원 → 그날 중단·다음 날부터 절반,
                 고점 대비 −100만 원 위로 회복하면 원래대로
    ④ 증거금      (실행기만) 주문가능수량 API 로 두 다리 다 들어갈 수 있는 만큼만

사후검증: 거래마다 예측한 VaR99 를 남겨, 실제 손실이 넘은 횟수를 Kupiec 검정으로 봄 (2주차와 같은 방법).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RiskConfig:
    budget: float = 300_000        # 거래당 꼬리 손실 예산 (원)
    k_max: int = 1                 # 최대 단위 수 — 기본은 키우지 않음(2주차처럼 「줄이기만」). 키우기는 검증 뒤 --k-max 로
    vol_ratio: float = 1.5         # ② 오늘 σ_eq / 20일 중앙값 이 넘으면 절반
    vol_days: int = 20
    dd_half: float = -1_500_000    # ③ 고점 대비
    dd_stop: float = -2_500_000
    dd_recover: float = -1_000_000
    tail_days: int = 5             # ① 꼬리 추정에 쓰는 지난 거래일 수
    var_q: float = 0.99
    es_q: float = 0.975


# ── 꼬리 추정 ────────────────────────────────────────────────
def spread_tail(day_spreads: list[np.ndarray], h: int, var_q: float = 0.99, es_q: float = 0.975) -> dict:
    """지난 며칠의 1분 스프레드(그날 β 로 계산) → H분 변화의 꼬리 (로그 스프레드 단위).

    H분 변화는 **같은 날 안에서만** 겹쳐 만듦 (밤사이 갭 제외 — 장마감 전에 청산하므로).
    방향을 모르니(롱/숏) 양쪽 꼬리 중 큰 쪽.
    """
    ch = []
    for x in day_spreads:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) > h:
            ch.append(x[h:] - x[:-h])
    if not ch:
        return {"ok": False, "n": 0}
    d = np.concatenate(ch)
    if len(d) < 200:
        return {"ok": False, "n": len(d)}
    out = {"ok": True, "n": int(len(d)), "h": h}
    for side, v in (("up", d), ("down", -d)):            # up: 스프레드가 오를 때 손해 보는 쪽(숏 스프레드)
        var = float(np.quantile(v, var_q))
        tail = v[v >= np.quantile(v, es_q)]
        out[f"var_{side}"], out[f"es_{side}"] = var, float(tail.mean())
    out["var"] = max(out["var_up"], out["var_down"])
    out["es"] = max(out["es_up"], out["es_down"])
    return out


def kupiec(n: int, x: int, p: float) -> dict:
    """VaR 초과 횟수 x / n 이 기대 비율 p 와 맞는지 (Kupiec 1995 POF). p값이 작으면 VaR 가 틀림."""
    if n == 0:
        return {"n": 0}
    ph = x / n
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = ((n - x) * math.log(1 - ph) if x < n else 0.0) + (x * math.log(ph) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    pval = math.erfc(math.sqrt(max(lr, 0) / 2))            # 자유도 1 카이제곱 꼬리
    return {"n": n, "초과": x, "기대": round(n * p, 2), "LR": round(lr, 3), "p값": round(pval, 3)}


# ── 레이어 ───────────────────────────────────────────────────
@dataclass
class RiskState:
    equity: float = 0.0            # 실현 누적손익 (원)
    peak: float = 0.0
    mode: str = "normal"           # normal | half | stop
    stop_day: str = ""             # stop 이 걸린 날 (그날만 중단)
    sigmas: list = field(default_factory=list)   # 지난 날들 σ_eq (② 기준)
    last_sigma_day: str = ""


class RiskLayer:
    def __init__(self, cfg: RiskConfig = RiskConfig(), path: Path | None = None):
        self.cfg, self.path = cfg, path
        self.st = RiskState()
        if path is not None and path.exists():
            self.st = RiskState(**json.loads(path.read_text(encoding="utf-8")))

    def save(self):
        if self.path is not None:
            self.path.write_text(json.dumps(asdict(self.st), ensure_ascii=False, indent=1), encoding="utf-8")

    # ③ 낙폭 상태
    def _dd_mode(self, day: str) -> str:
        """주문 전 확인: 중단은 그날만 → 다음 날 첫 확인에서 절반으로."""
        s = self.st
        if s.mode == "stop" and s.stop_day != day:
            s.mode = "half"
        return s.mode

    def _after_close(self, day: str):
        """거래가 끝날 때마다 낙폭으로 상태를 바꿈 (중단은 새로 손실이 나서 선을 넘을 때만)."""
        c, s = self.cfg, self.st
        dd = s.equity - s.peak
        if dd <= c.dd_stop:
            s.mode, s.stop_day = "stop", day
        elif dd <= c.dd_half and s.mode == "normal":
            s.mode = "half"
        elif dd > c.dd_recover and s.mode == "half":
            s.mode = "normal"

    def note_sigma(self, day: str, sigma_eq: float):
        """그날 σ_eq 를 기록 (② 의 기준이 될 과거값). 같은 날 두 번은 안 넣음."""
        if day != self.st.last_sigma_day and math.isfinite(sigma_eq):
            self.st.sigmas = (self.st.sigmas + [float(sigma_eq)])[-self.cfg.vol_days:]
            self.st.last_sigma_day = day

    def size(self, *, day: str, qa: int, pa: float, pb: float, mult: int, sigma_eq: float, entry: float,
             stop: float, tail: dict, cost_unit: float, margin_units: int | None = None) -> tuple[int, dict]:
        """단위 수 k 와 이유. qa = 1단위의 삼성 계약 수, cost_unit = 1단위 왕복 비용(원)."""
        c = self.cfg
        a_notional = qa * pa * mult
        loss_stop = max(stop - entry, 0) * sigma_eq * a_notional
        loss_es = tail.get("es", float("nan")) * a_notional if tail.get("ok") else float("nan")
        loss_unit = max(x for x in (loss_stop, loss_es) if math.isfinite(x)) + cost_unit if (
            math.isfinite(loss_stop) or math.isfinite(loss_es)) else float("inf")
        k_es = int(min(c.k_max, max(0, math.floor(c.budget / loss_unit)))) if loss_unit > 0 else 0
        why = {"단위당꼬리손실원": round(loss_unit), "ES손실원": None if not math.isfinite(loss_es) else round(loss_es),
               "손절손실원": round(loss_stop), "①ES예산": k_es}
        k = k_es
        past = self.st.sigmas[:-1] if self.st.last_sigma_day == day else self.st.sigmas   # 오늘 것은 빼고
        if len(past) >= 5:
            ratio = sigma_eq / float(np.median(past))
            why["σ비율"] = round(ratio, 2)
            if ratio > c.vol_ratio:
                k = k // 2
                why["②변동성"] = "절반"
        mode = self._dd_mode(day)
        why["③낙폭"] = f"{mode} (고점 대비 {self.st.equity - self.st.peak:,.0f}원)"
        if mode == "half":
            k = k // 2 if k > 1 else k
        elif mode == "stop":
            k = 0
        if margin_units is not None:
            why["④증거금단위"] = margin_units
            k = min(k, margin_units)
        why["k"] = k
        if tail.get("ok"):
            why["VaR99원"] = round(tail["var"] * a_notional * k)
            why["ES975원"] = round(tail["es"] * a_notional * k)
        return k, why

    def on_close(self, net: float, day: str):
        s = self.st
        s.equity += net
        s.peak = max(s.peak, s.equity)
        self._after_close(day)
        self.save()
