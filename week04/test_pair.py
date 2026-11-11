"""4주차 시험 — 키·네트워크 없이.  python test_pair.py  (week04 폴더에서)

손계산
  1. 계약 수: β 0.6, 삼성 100,000, 하이닉스 550,000 → 550,000 ÷ (0.6 × 100,000) = 9.17 → 9계약
  2. 비용: 삼성 9계약 편도 = 9×10×(100 + 3) = 9,270원, 하이닉스 1계약 편도 = 10×(1,000 + 16.5) = 10,165원
          왕복 = 2 × 19,435 = 38,870원.  σ 0.001 이면 1σ 값 = 9×100,000×10×0.001 = 9,000원 → 비용 하한 z = 4.32
  3. 손익: 롱 스프레드(삼성 9 매수·하이닉스 1 매도) 삼성 +200원, 하이닉스 +1,000원 → 10×(9×200 − 1×1,000) = 8,000원
통계·신호
  4. β 추정: 참 β 0.6 인 가짜 데이터에서 0.55~0.65
  5. 신호: z −2.1 진입(롱) → −0.4 수렴 청산 / 반대로 더 벌어지면 z 손절 / 15:15 장마감 / 상관 낮으면 진입 안 함
두 다리 주문
  6. 정상: 두 다리 전량 체결
  7. 하이닉스 거부 → 삼성 체결분을 되돌림 (한 다리만 남지 않음)
  8. 삼성 처음엔 절반만 체결 → 재주문으로 전량
  9. 청산(unwind=False)에서 한 다리 거부 → 되돌리지 않고 남은 수량만 알림
실행기
 10. 진입 → 청산 → trades.csv 한 줄, 계좌 0
 11. 레그 실패 → 신규 중지, trades.csv 에 「레그실패」, 계좌 0
 12. 청산 한 다리 실패 → 다음 루프에 남은 다리만 다시 → 계좌 0
 13. 재시작 → state.json 으로 포지션 이어 관리
 14. 잔고 불일치 → 신규 중지
백테스트
 15. 강한 평균회귀 + 비용 0 이면 이익, 같은 데이터에 비용을 크게 주면 손실 (방향 확인 — 비용 큰 쪽은 하루 손실 한도로 거래가 일찍 멈춤)
── 11/08 수정안 ──
OU (2번)
 16. 첫 도달 시간 공식: 출발점 근처 ±a 는 순수 확산 a²/2 와 같음, a→m 은 몬테카를로와 10% 안
 17. 날마다 수준이 크게 바뀌어도 κ 추정 (참 κ 0.05 → 0.04~0.06)
 18. 비용이 크면 최적 진입선이 멀어짐, 비용이 너무 크면 「거래 안 함」, 청산선은 0 이상
 19. plan_days: 그날 숫자 = OU 계획 (보유 상한 = 반감기 × 3)
하이닉스 지정가 (1번)
 20. 지정가 체결 → 삼성 헤지 / 미체결 → 노출 없음(놓침) / 헤지 거부 → 하이닉스 되돌림
 21. 실행기: 지정가 체결 → 포지션, 미체결 → trades.csv 「신호놓침」·포지션 없음
 22. 백테스트 지정가 모드: 놓친 신호 수를 셈
추세 필터 (11/08)
 23. 아침: 3일 날평균이 같은 방향으로 크게 움직이면 그날 추세 ±1, 평평하면 0
 24. 신호: 막힌 방향 진입은 「추세 반대」, 청산은 영향 없음
 25. 장중: 스프레드가 꾸준히 오르면(한 시간 +3%) 숏 스프레드(하락에 거는 진입)만 막힘
 26. 실행기: 막힌 진입은 기록만, 포지션 없음
리스크 레이어 (11/08)
 27. 꼬리 추정: 1분 변화 σ 인 무작위 걸음이면 16분 VaR99 ≈ 2.326 × 4σ (10% 안)
 28. Kupiec: 100번 중 0번 초과(기대 1) → LR 2.01, p 0.156 / 1번 → LR 0, p 1
 29. ① ES 예산 (상한 3으로 시험): 단위당 꼬리손실 66,000원 → 예산 50만 원이면 7 → 상한 3, 예산 10만 원이면 1
 30. ② σ 가 지난 중앙값의 2배 → 3단위가 1단위로
 31. ③ 낙폭: −160만 원 → 절반, 더 잃어 −260만 원 → 그날 0, 다음 날 절반, 회복 → 원래대로
 32. 백테스트 리스크 레이어: 단위 k·예측 VaR 기록, 사후검증 계산
 33. 실행기: 예산이 너무 작으면 0단위(주문 없음), 주문가능수량이 적으면 그만큼만
스프레드 정의 (11/08)
 34. 칼만 β: 참 β 가 0.5 → 0.8 로 바뀌면 따라감 (마지막 0.7~0.9), 앞부분만 넣어도 같은 값(미래 정보 없음)
 35. 칼만 스프레드의 1분 변화 = rA − (직전 β)·rB, β 가 바뀌어도 튀지 않음
 36. 가격비: β = 1, 스프레드 = ln A − ln B
 37. 이동평균·이동기울기 빠른 계산 = 직접 계산
 38. 세 정의 모두 plan_days 가 거래하는 날을 만듦
비싼 다리 자동 선택 (11/08 실제 가격 반영)
 39. 실제 가격(삼성 213,000원 1틱 500 × 9 / 하이닉스 1,300,000원 1틱 1,000 × 1) → 삼성 / 예전 가정(10만·55만) → 하이닉스
 40. 왕복 비용 손계산: 삼성 지정가면 삼성 슬리피지 0 → 2×[9×10×(0+6.39) + 10×(1,000+39)] = 21,930원
 41. 삼성 9계약 중 4계약만 체결 → 하이닉스 0.44계약은 못 붙임 → 삼성 4계약 되돌림, 노출 없음
 42. 백테스트 auto: 실제 가격대 데이터에서 지정가 다리가 삼성(a)
 43. 실행기 --passive auto: 첫 주문이 삼성 지정가
"""
import argparse
import datetime as dt
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

time.sleep = lambda s: None          # 체결 대기 건너뜀
import pair_core as pc               # noqa: E402
import pair_spread as ps             # noqa: E402

P = pc.Params()
ok = 0


def check(name, cond, info=""):
    global ok
    assert cond, f"{name} 실패 {info}"
    ok += 1
    print(f"[통과] {name}")


# ── 손계산 ──
check("1 계약 수 9", pc.hedge_qty(0.6, 100_000, 550_000, P) == 9)
C = pc.Costs()
check("2a 삼성 편도 9,270원", round(pc.leg_cost(9, 100_000, C)) == 9270)
check("2b 하이닉스 편도 10,165원", round(pc.leg_cost(1, 550_000, C)) == 10165)
check("2c 왕복 38,870원", round(pc.round_trip_cost(9, 1, 100_000, 550_000, C)) == 38870)
check("2d 비용 하한 z 4.32", abs(pc.cost_floor_z(0.001, 9, 100_000, 1, 550_000, C) - 4.319) < 0.01)
pos = pc.Position(+1, 9, 1, 100_000, 550_000, pd.Timestamp("2026-11-10 10:00"), -2.1, 0.6)
check("3 손익 8,000원", pos.pnl(100_200, 551_000) == 8000)

# ── β ──
rng = np.random.default_rng(0)
idx = pd.date_range("2026-11-02 09:00", periods=395, freq="min")
idx = idx.append([idx + pd.Timedelta(days=k) for k in range(1, 8)])
rb = rng.normal(0, 0.001, len(idx)); ra = 0.6 * rb + rng.normal(0, 0.0004, len(idx))
a, b = pd.Series(100_000 * np.exp(np.cumsum(ra)), idx), pd.Series(550_000 * np.exp(np.cumsum(rb)), idx)
beta = pc.estimate_beta(pc.intraday_returns(a), pc.intraday_returns(b))
check("4 β 추정 0.55~0.65", 0.55 < beta < 0.65, beta)

# ── 신호 ──
T0 = pd.Timestamp("2026-11-10 10:00")
s = pc.Signal(P)
d1 = s.decide(T0, -2.1, 0.8, 100_000, 550_000)
check("5a z −2.1 → 롱 진입", d1.action == "enter" and d1.dir == +1)
s.opened(pc.Position(+1, 9, 1, 100_000, 550_000, T0, -2.1, 0.6))
check("5b z −1.0 보유", s.decide(T0 + pd.Timedelta(minutes=5), -1.0, 0.8, 100_000, 550_000).action == "hold")
check("5c z −0.4 수렴 청산", s.decide(T0 + pd.Timedelta(minutes=6), -0.4, 0.8, 100_000, 550_000).reason == "수렴")
check("5d z −3.6 손절", s.decide(T0 + pd.Timedelta(minutes=6), -3.6, 0.8, 100_000, 550_000).reason == "z 손절")
check("5e 15:15 장마감", s.decide(pd.Timestamp("2026-11-10 15:15"), -1.0, 0.8, 100_000, 550_000).reason == "장마감 청산")
s2 = pc.Signal(P)
check("5f 상관 0.3 → 진입 안 함", s2.decide(T0, -2.5, 0.3, 100_000, 550_000).action == "none")
check("5g 09:10 → 진입 안 함", s2.decide(pd.Timestamp("2026-11-10 09:10"), -2.5, 0.8, 1, 1).action == "none")


# ── 두 다리 ──
class Book:
    """주문마다 체결 비율·거부를 정할 수 있는 가짜 브로커."""

    def __init__(self, px=None, reject=(), first_partial=None):
        self.px = px or {"A": 100_000, "B": 550_000}
        self.reject, self.first_partial = set(reject), dict(first_partial or {})
        self.orders, self.net, self.n = {}, {"A": 0, "B": 0}, 0

    def send(self, sym, side, qty, price=None):
        if sym in self.reject:
            raise RuntimeError(f"거부 {sym}")
        self.n += 1
        no = f"N{self.n}"
        self.sent = getattr(self, "sent", []) + [(sym, side, qty, price)]
        if price is not None and getattr(self, "no_limit_fill", False):   # 지정가가 안 체결되는 시장
            self.orders[no] = (0, 0.0)
            return no
        q = qty
        if sym in self.first_partial:
            q = int(qty * self.first_partial.pop(sym))
        self.orders[no] = (q, self.px[sym])
        self.net[sym] += q if side == "buy" else -q
        return no

    def filled(self, no):
        return self.orders.get(no, (0, 0.0))

    def cancel(self, no, sym, side):
        pass

    def net_positions(self):
        return dict(self.net)

    def max_qty(self, sym, side, px):
        return getattr(self, "max_q", {}).get(sym, 999)


legs = [pc.Leg("B", "sell", 1), pc.Leg("A", "buy", 9)]
bk = Book(); r = pc.PairExecutor(bk).execute(legs, "t")
check("6 정상 두 다리", r.ok and bk.net == {"A": 9, "B": -1})
bk = Book(reject={"B"}); r = pc.PairExecutor(bk).execute(legs, "t")
check("7 하이닉스 거부 → 삼성 되돌림", (not r.ok) and bk.net == {"A": 0, "B": 0} and "되돌림 완료" in r.note, (bk.net, r.note))
bk = Book(first_partial={"A": 0.5}); r = pc.PairExecutor(bk).execute(legs, "t")
check("8 절반 체결 → 재주문 전량", r.ok and bk.net == {"A": 9, "B": -1}, bk.net)
bk = Book(); bk.net = {"A": 9, "B": -1}; bk.reject = {"B"}
r = pc.PairExecutor(bk).execute([pc.Leg("B", "buy", 1), pc.Leg("A", "sell", 9)], "t", unwind=False)
check("9 청산 한 다리 거부 → 되돌리지 않음", (not r.ok) and bk.net == {"A": 0, "B": -1} and not r.unwound, bk.net)


# ── 실행기 ──
CLOCK = [dt.datetime(2026, 11, 10, 10, 0, 30, tzinfo=ps.KST)]
ps.now_kst = lambda: CLOCK[0]


class FakeRunner(ps.Runner):
    """특징 계산 대신 정해 둔 z 를 줌 (실행기 흐름만 시험)."""
    script: list = []

    def features(self):
        t = pd.Timestamp(CLOCK[0].replace(tzinfo=None)).floor("min") - pd.Timedelta(minutes=1)
        z, corr = self.script[0] if self.script else (0.0, 0.8)
        return pd.DataFrame({"a": [100_000.0], "b": [550_000.0], "beta": [0.6], "z": [z], "corr": [corr], "sd": [0.001]},
                            index=[t])


def args(**kw):
    a = argparse.Namespace(entry=2.0, exit=0.5, stop=3.5, window=120, loss_limit=300_000, day_loss=600_000,
                           unit_b=1, fee_rate=0.00003, stop_at="15:30", ou=False, passive_b=False, passive_wait=30,
                           trend_filter=False, risk=False, risk_budget=500_000, k_max=3, spread="beta", passive="none")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


A, B = {"sym": "A", "name": "삼성전자", "mult": 10}, {"sym": "B", "name": "SK하이닉스", "mult": 10}
Q = lambda s: {"last": 100_000.0, "bid": 99_900.0, "ask": 100_000.0} if s == "A" else {"last": 550_000.0, "bid": 549_000.0, "ask": 550_000.0}


def runner(tmp, bk, script, **kw):
    FakeRunner.script = script
    return FakeRunner(args(**kw), ps.Rec(Path(tmp)), A, B, None, Q, bk, pd.DataFrame())


def tick(r, minute, z, corr=0.8):
    CLOCK[0] = CLOCK[0].replace(minute=minute)
    r.script[:] = [(z, corr)]
    r.step()


def kinds(tmp):
    import json
    return [json.loads(l)["kind"] for l in (Path(tmp) / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def case(fn):
    tmp = tempfile.mkdtemp()
    try:
        fn(tmp)
    finally:
        shutil.rmtree(tmp)


def t10(tmp):
    bk = Book(); r = runner(tmp, bk, [])
    tick(r, 1, -2.2); tick(r, 2, -1.0)
    bk.px = {"A": 100_200, "B": 551_000}
    tick(r, 3, -0.3)
    tr = pd.read_csv(Path(tmp) / "trades.csv")
    check("10 진입→청산 기록·계좌 0", len(tr) == 1 and tr.iloc[0]["손익_비용전원"] == 8000 and bk.net == {"A": 0, "B": 0},
          (tr.to_dict("records"), bk.net))


def t11(tmp):
    bk = Book(reject={"B"}); r = runner(tmp, bk, [])
    tick(r, 1, -2.2)
    tr = pd.read_csv(Path(tmp) / "trades.csv")
    bk.reject = set(); tick(r, 15, -2.5)
    check("11 레그실패 → 신규중지·계좌 0", r.sig.blocked and tr.iloc[0]["구분"] == "레그실패" and bk.net == {"A": 0, "B": 0}
          and r.sig.pos is None, (r.sig.blocked, bk.net))


def t12(tmp):
    bk = Book(); r = runner(tmp, bk, [])
    tick(r, 1, -2.2)
    bk.reject = {"B"}; tick(r, 2, -0.3)
    check("12a 청산 한 다리 실패 → 남은 다리 기억", r.exit_left == [{"sym": "B", "side": "buy", "qty": 1}] and bk.net == {"A": 0, "B": -1},
          (r.exit_left, bk.net))
    r.reconcile(force=True)
    check("12b 청산 중 잔고 대조 일치", not r.sig.blocked)
    bk.reject = set(); tick(r, 3, 0.0)
    check("12c 다음 루프에 다시 → 계좌 0", not r.exit_left and r.sig.pos is None and bk.net == {"A": 0, "B": 0}, bk.net)


def t13(tmp):
    bk = Book(); r = runner(tmp, bk, [])
    tick(r, 1, -2.2)
    r2 = runner(tmp, bk, [])
    check("13a 재시작 → 포지션 복구", r2.sig.pos is not None and r2.sig.pos.qa == 9)
    tick(r2, 5, -0.2)
    check("13b 복구 뒤 청산", r2.sig.pos is None and bk.net == {"A": 0, "B": 0})


def t14(tmp):
    bk = Book(); bk.net = {"A": 3, "B": 0}; r = runner(tmp, bk, [])
    r.reconcile(force=True); tick(r, 1, -2.5)
    check("14 잔고 불일치 → 신규 중지", r.sig.blocked and r.sig.pos is None and "잔고불일치" in kinds(tmp))


for f in (t10, t11, t12, t13, t14):
    case(f)


# ── 백테스트 방향 ──
import backtest_pair as bt  # noqa: E402

rng = np.random.default_rng(3)
rows, lb, u = [], math.log(550_000), 0.0
for dd in pd.bdate_range("2026-06-01", periods=12):
    for m in range(395):
        t = dd + pd.Timedelta(hours=9, minutes=m)
        rbb = rng.normal(0, 0.003); lb += rbb
        u += -0.15 * u + rng.normal(0, 0.001)            # 강한 평균회귀 괴리 (상관 약 0.86)
        la = math.log(100_000) + 0.6 * (lb - math.log(550_000)) + u
        rows.append((t, math.exp(la), math.exp(lb)))
D = pd.DataFrame(rows, columns=["t", "a", "b"]).set_index("t")
df = pd.DataFrame({"a_open": D["a"].shift(1).fillna(D["a"]), "a_close": D["a"], "b_open": D["b"].shift(1).fillna(D["b"]),
                   "b_close": D["b"]}, index=D.index)
df["date"] = df.index.strftime("%Y%m%d")
p = pc.Params(beta_days=3, window=60)
feat = pc.features(df["a_close"], df["b_close"], p)
free = bt.simulate(df, feat, p, pc.Costs(slip_ticks=0, fee_rate=0))
dear = bt.simulate(df, feat, p, pc.Costs(slip_ticks=30, fee_rate=0.001))
check("15 비용 0 이익 / 비용 큼 손실", len(free) > 20 and len(dear) > 0 and free.net.sum() > 0 and dear.net.sum() < 0, (len(free), free.net.sum() if len(free) else None, len(dear), dear.net.sum() if len(dear) else None))


# ── 16~22: 11/08 수정안 ──
check("16a ±a 근처 = a²/2", abs(pc.mfpt_out(0, 0.1) - 0.005) < 2e-4)
rng = np.random.default_rng(7); dtau = 0.001
y = np.full(3000, 2.0); tt = np.zeros(3000); alive = np.ones(3000, bool)
while alive.any():
    y[alive] += -y[alive] * dtau + math.sqrt(2 * dtau) * rng.normal(size=alive.sum()); tt[alive] += dtau
    alive &= y > 0.5
check("16b 2→0.5 몬테카를로 10% 안", abs(pc.mfpt_down(2, 0.5) / tt.mean() - 1) < 0.10, (pc.mfpt_down(2, 0.5), tt.mean()))

days = []
for k in range(5):
    x = [rng.normal(0, 0.05)]                                       # 날마다 수준이 크게 다름
    for _ in range(390):
        x.append(x[-1] + 0.05 * (x[0] - x[-1]) + rng.normal(0, 0.001))
    days.append(np.array(x))
fit = pc.fit_ou_days(days)
check("17 κ 추정 0.04~0.06 (참 0.05)", fit.ok and 0.04 < fit.kappa < 0.06, fit)

f0 = pc.OUFit(True, 0.05, 0.0, 0.001, math.log(2) / 0.05, 2000)
p_lo, p_hi, p_no = pc.ou_plan(f0, 0.5), pc.ou_plan(f0, 2.0), pc.ou_plan(f0, 9.0)
check("18 비용↑ → 진입선↑, 너무 크면 안 함, 청산 ≥ 0", p_lo.entry < p_hi.entry and not p_no.ok and p_lo.exit >= 0 and p_hi.exit >= 0,
      (p_lo, p_hi, p_no))

idx = pd.date_range("2026-11-02 09:00", periods=390, freq="min")
idx = idx.append([idx + pd.Timedelta(days=k) for k in range(1, 9)])
rb = rng.normal(0, 0.002, len(idx)); lb = np.cumsum(rb)
u = np.zeros(len(idx))
for i in range(1, len(idx)):
    u[i] = u[i - 1] * 0.92 + rng.normal(0, 0.0015)
sa = pd.Series(100_000 * np.exp(0.6 * lb + u), idx); sb = pd.Series(550_000 * np.exp(lb), idx)
plans = pc.plan_days(sa, sb, pc.Params(), pc.Costs(slip_ticks=0.2, fee_rate=0.0))
okp = [pl for pl in plans.values() if pl.ok]
check("19 plan_days 숫자 연결", okp and all(pl.params.entry == pl.plan.entry and
      pl.params.max_hold == int(min(120, max(10, round(3 * pl.fit.half_life)))) for pl in okp),
      [pl.row() for pl in plans.values()][-2:])

bk = Book(); r = pc.PairExecutor(bk).execute_passive_first(pc.Leg("B", "sell", 1), 550_000, pc.Leg("A", "buy", 9), "t")
check("20a 지정가 체결 → 헤지", r.ok and bk.net == {"A": 9, "B": -1})
bk = Book(); bk.no_limit_fill = True
r = pc.PairExecutor(bk).execute_passive_first(pc.Leg("B", "sell", 1), 550_000, pc.Leg("A", "buy", 9), "t")
check("20b 미체결 → 노출 없음", r.missed and bk.net == {"A": 0, "B": 0})
bk = Book(); bk.reject = {"A"}
r = pc.PairExecutor(bk).execute_passive_first(pc.Leg("B", "sell", 1), 550_000, pc.Leg("A", "buy", 9), "t")
check("20c 헤지 거부 → 하이닉스 되돌림", (not r.ok) and (not r.missed) and bk.net == {"A": 0, "B": 0}, bk.net)


def t21(tmp):
    bk = Book(); r = runner(tmp, bk, [], passive_b=True)
    tick(r, 1, -2.2)
    a_ok = r.sig.pos is not None and bk.net == {"A": 9, "B": -1}
    tick(r, 2, -0.3)
    bk2 = Book(); bk2.no_limit_fill = True
    tmp2 = tempfile.mkdtemp()
    r2 = runner(tmp2, bk2, [], passive_b=True); tick(r2, 1, -2.2)
    tr2 = pd.read_csv(Path(tmp2) / "trades.csv"); shutil.rmtree(tmp2)
    check("21 실행기 지정가: 체결→포지션→청산 / 미체결→신호놓침", a_ok and bk.net == {"A": 0, "B": 0} and r2.sig.pos is None
          and tr2.iloc[0]["구분"] == "신호놓침" and bk2.net == {"A": 0, "B": 0})


case(t21)
dfp = df.assign(b_high=df["b_close"], b_low=df["b_close"])
Tp = bt.simulate(dfp, feat, p, pc.Costs(slip_ticks_b=0.0), passive_b=True)
check("22 백테스트 지정가 모드 — 놓친 신호 셈", "missed" in Tp.attrs and Tp.attrs["missed"] >= 0)


# ── 23~26: 추세 필터 ──
def make_pair(drift_per_day=0.0, intraday_slope=0.0, days=7, seed=11):
    g = np.random.default_rng(seed)
    ix = pd.date_range("2026-11-02 09:00", periods=390, freq="min")
    ix = ix.append([ix + pd.Timedelta(days=k) for k in range(1, days)])
    lbb = np.cumsum(g.normal(0, 0.002, len(ix)))
    dayno = (ix.normalize() - ix[0].normalize()).days.values
    mins = (ix - ix.normalize()).seconds.values / 60 - 540
    uu = np.zeros(len(ix))
    for i in range(1, len(ix)):
        uu[i] = uu[i - 1] * 0.92 + g.normal(0, 0.0015)
    spread = uu + drift_per_day * dayno + intraday_slope * mins * (dayno == dayno.max())   # 장중 추세는 마지막 날만
    return (pd.Series(100_000 * np.exp(0.6 * lbb + spread), ix), pd.Series(550_000 * np.exp(lbb), ix))


cz = pc.Costs(slip_ticks=0.2, fee_rate=0.0)
sa, sb = make_pair(drift_per_day=0.01)
pl_up = pc.plan_days(sa, sb, pc.Params(), cz)
sa0, sb0 = make_pair(drift_per_day=0.0)
pl_flat = pc.plan_days(sa0, sb0, pc.Params(), cz)
last_up, last_flat = list(pl_up.values())[-1], list(pl_flat.values())[-1]
check("23 아침 추세: 하루 +1% 씩 → +1, 평평 → 0", last_up.ok and last_up.day_trend == 1 and last_flat.day_trend == 0,
      (last_up.row(), last_flat.row()))

s3 = pc.Signal(P)
check("24a 막힌 방향 진입 → 추세 반대", s3.decide(T0, 2.5, 0.8, 1, 1, blocked_dirs=frozenset({-1})).reason.startswith("추세 반대"))
check("24b 안 막힌 방향은 진입", s3.decide(T0, -2.5, 0.8, 1, 1, blocked_dirs=frozenset({-1})).action == "enter")
s3.opened(pc.Position(-1, 9, 1, 100_000, 550_000, T0, 2.5, 0.6))
check("24c 청산엔 영향 없음", s3.decide(T0 + pd.Timedelta(minutes=3), 0.2, 0.8, 1, 1, blocked_dirs=frozenset({-1, 1})).action == "exit")

sa2, sb2 = make_pair(intraday_slope=0.0005)                         # 마지막 날 장중 분당 +0.05% (한 시간 +3%)
pl2 = pc.plan_days(sa2, sb2, pc.Params(), cz)
f2 = pc.features_ou(sa2, sb2, pl2, trend_filter=True)
lastday = f2[f2.index.normalize() == f2.index.normalize()[-1]]
late = lastday[lastday.index.strftime("%H%M") >= "1100"]
check("25 장중 오름 추세 → 숏 스프레드만 막힘", late["block_short"].mean() > 0.9 and late["block_long"].mean() < 0.1,
      (late["block_short"].mean(), late["block_long"].mean(), late["slope_sig"].median()))


class TrendRunner(FakeRunner):
    def features(self):
        f = super().features()
        f["block_short"], f["block_long"], f["slope_sig"] = True, False, 3.0
        return f


def t26(tmp):
    bk = Book()
    TrendRunner.script = []
    r = TrendRunner(args(), ps.Rec(Path(tmp)), A, B, None, Q, bk, pd.DataFrame())
    tick(r, 1, 2.4)
    check("26 실행기: 막힌 진입 → 기록만", r.sig.pos is None and bk.n == 0 and "추세필터" in kinds(tmp))


case(t26)

# ── 27~33: 리스크 레이어 ──
import risk_layer as rl  # noqa: E402

g = np.random.default_rng(5)
walks = [np.cumsum(g.normal(0, 0.001, 390)) for _ in range(60)]
tl = rl.spread_tail(walks, 16)
check("27 꼬리: VaR99 ≈ 2.326×4σ", tl["ok"] and abs(tl["var"] / (2.326 * 0.004) - 1) < 0.10, tl)
k0, k1 = rl.kupiec(100, 0, 0.01), rl.kupiec(100, 1, 0.01)
check("28 Kupiec", abs(k0["LR"] - 2.01) < 0.01 and abs(k0["p값"] - 0.156) < 0.002 and k1["LR"] == 0 and k1["p값"] == 1.0, (k0, k1))

tail0 = {"ok": True, "es": 0.004, "var": 0.003}
kw = dict(day="2026-11-10", qa=9, pa=100_000, pb=550_000, mult=10, sigma_eq=0.002, entry=1.5, stop=3.0,
          tail=tail0, cost_unit=30_000)
k_big, w_big = rl.RiskLayer(rl.RiskConfig(budget=500_000, k_max=3)).size(**kw)
k_small, _ = rl.RiskLayer(rl.RiskConfig(budget=100_000, k_max=3)).size(**kw)
check("29 ES 예산 → 3 / 1", k_big == 3 and w_big["①ES예산"] == 3 and w_big["단위당꼬리손실원"] == 66_000 and k_small == 1,
      (w_big, k_small))

K3 = rl.RiskConfig(budget=500_000, k_max=3)
L = rl.RiskLayer(K3)
for dd_, sg in enumerate([0.001, 0.0011, 0.0009, 0.001, 0.001]):
    L.note_sigma(f"2026-11-0{dd_ + 1}", sg)
k_v, w_v = L.size(**{**kw, "sigma_eq": 0.002, "day": "2026-11-10"})
check("30 σ 2배 → 절반", w_v["②변동성"] == "절반" and k_v == 1, w_v)

L = rl.RiskLayer(K3)
k_a, _ = L.size(**kw)
L.on_close(-1_600_000, "2026-11-10"); k_b, _ = L.size(**kw)
L.on_close(-1_000_000, "2026-11-10"); k_c, _ = L.size(**kw)
k_d, _ = L.size(**{**kw, "day": "2026-11-11"})
L.on_close(2_000_000, "2026-11-11"); k_e, _ = L.size(**{**kw, "day": "2026-11-11"})
check("31 낙폭 3→1→0→1→3", (k_a, k_b, k_c, k_d, k_e) == (3, 1, 0, 1, 3), (k_a, k_b, k_c, k_d, k_e, L.st))

sa3, sb3 = make_pair(days=9)
pl3 = pc.plan_days(sa3, sb3, pc.Params(), cz)
f3 = pc.features_ou(sa3, sb3, pl3)
df3 = pd.DataFrame({"a_open": sa3.shift(1).fillna(sa3), "a_close": sa3, "b_open": sb3.shift(1).fillna(sb3), "b_close": sb3,
                    "b_high": sb3, "b_low": sb3})
T3 = bt.simulate(df3, f3, pc.Params(), cz, plans=pl3, risk=rl.RiskLayer(K3))
okk = len(T3) > 0 and T3["k"].between(1, 3).all() and T3["var99_pred"].notna().all()
v3 = rl.kupiec(len(T3), int((-T3["net"] > T3["var99_pred"]).sum()), 0.01) if len(T3) else {}
check("32 백테스트 리스크 레이어 기록", okk and "p값" in v3, (len(T3), T3.head(2).to_dict("records") if len(T3) else None))


class PlanRunner(FakeRunner):
    """OU 계획을 손으로 넣은 실행기."""
    def __init__(self, *a_, **k_):
        super().__init__(*a_, **k_)
        self.plan = pc.DayPlan(pd.Timestamp("2026-11-10"), True, 0.6, pc.OUFit(True, 0.05, 0.0, 0.002, 13.9, 999),
                               pc.OUPlan(True, 2.0, 0.0, 0.01, 60, 0.5), 9, 70, replace_p(), tail=tail0)


def replace_p():
    from dataclasses import replace as rp
    return rp(P, entry=2.0, exit=0.0, stop=3.5)


def t33(tmp):
    bk = Book(); PlanRunner.script = []
    r = PlanRunner(args(risk=True, risk_budget=20_000), ps.Rec(Path(tmp)), A, B, None, Q, bk, pd.DataFrame())
    tick(r, 1, -2.2)
    tr = pd.read_csv(Path(tmp) / "trades.csv")
    a_ok = r.sig.pos is None and bk.n == 0 and tr.iloc[0]["구분"] == "리스크레이어0단위"
    tmp2 = tempfile.mkdtemp()
    bk2 = Book(); bk2.max_q = {"B": 2, "A": 99}
    r2 = PlanRunner(args(risk=True, risk_budget=10_000_000), ps.Rec(Path(tmp2)), A, B, None, Q, bk2, pd.DataFrame())
    tick(r2, 1, -2.2); shutil.rmtree(tmp2)
    check("33 실행기 리스크: 0단위 / 증거금 2단위", a_ok and r2.sig.pos is not None and r2.sig.pos.qb == 2 and bk2.net["B"] == -2,
          (a_ok, None if r2.sig.pos is None else (r2.sig.pos.qa, r2.sig.pos.qb)))


case(t33)

# ── 34~38: 스프레드 정의 ──
g = np.random.default_rng(21)
ix = pd.date_range("2026-11-02 09:00", periods=390, freq="min")
ix = ix.append([ix + pd.Timedelta(days=k) for k in range(1, 8)])
rbk = g.normal(0, 0.002, len(ix))
btrue = np.where(np.arange(len(ix)) < len(ix) // 2, 0.5, 0.8)
rak = btrue * rbk + g.normal(0, 0.0005, len(ix))
ka, kb_ = pd.Series(100_000 * np.exp(np.cumsum(rak)), ix), pd.Series(550_000 * np.exp(np.cumsum(rbk)), ix)
K = pc.kalman_beta(ka, kb_)
K2 = pc.kalman_beta(ka.iloc[:1500], kb_.iloc[:1500])
check("34 칼만 β 0.5→0.8 따라감·미래정보 없음", 0.7 < K["beta"].iloc[-1] < 0.9 and 0.4 < K["beta"].iloc[1000] < 0.6
      and np.allclose(K["beta"].values[:1500], K2["beta"].values), (K["beta"].iloc[[1000, -1]].tolist()))
ra_, rb_ = pc.intraday_returns(ka).values, pc.intraday_returns(kb_).values
dsv = np.diff(K["s"].values)
exp_ = np.nan_to_num(ra_[1:] - K["beta"].values[1:] * rb_[1:])
check("35 칼만 스프레드 변화 = rA − β·rB", np.allclose(dsv, exp_), np.abs(dsv - exp_).max())
Vr = pc.SpreadView(pd.DataFrame({"a": ka, "b": kb_}), "ratio", None, P)
check("36 가격비 β=1", Vr.day_beta(None) == 1.0 and np.allclose(Vr.x(1.0, slice(0, 5)), np.log(ka.values[:5]) - np.log(kb_.values[:5])))
xx = g.normal(0, 1, 300).cumsum()
rm, rs = pc._roll_mean(xx, 20), pc._roll_slope(xx, 20)
ok_m = np.isclose(rm[50], xx[30:50].mean())
ok_s = np.isclose(rs[50], np.polyfit(np.arange(20), xx[30:50], 1)[0])
check("37 이동평균·기울기", ok_m and ok_s, (rm[50], xx[30:50].mean(), rs[50]))
sa4, sb4 = make_pair(days=8)
okn = {m: sum(pl.ok for pl in pc.plan_days(sa4, sb4, pc.Params(), cz, spread=m).values()) for m in pc.SPREAD_MODES}
check("38 세 정의 모두 거래일", all(v > 0 for v in okn.values()), okn)

# ── 39~43: 비싼 다리 자동 ──
ca = pc.Costs(passive="auto")
check("39 비싼 다리: 실제 가격 → 삼성, 예전 가정 → 하이닉스",
      pc.passive_leg_of(9, 213_000, 1, 1_300_000, ca) == "a" and pc.passive_leg_of(9, 100_000, 1, 550_000, ca) == "b")
rt = pc.round_trip_cost(9, 1, 213_000, 1_300_000, ca)
check("40 왕복 비용 21,930원", round(rt) == 21_930, rt)
bk = Book(first_partial={"A": 4 / 9})
r = pc.PairExecutor(bk).execute_passive_first(pc.Leg("A", "buy", 9), 213_000, pc.Leg("B", "sell", 1), "t")
check("41 삼성 4/9 체결 → 되돌림·노출 없음", r.missed and bk.net == {"A": 0, "B": 0}, (bk.net, r.note))
g = np.random.default_rng(31)
ix = pd.date_range("2026-11-02 09:00", periods=390, freq="min")
ix = ix.append([ix + pd.Timedelta(days=k) for k in range(1, 9)])
lbb = np.cumsum(g.normal(0, 0.002, len(ix))); uu = np.zeros(len(ix))
for i in range(1, len(ix)):
    uu[i] = uu[i - 1] * 0.9 + g.normal(0, 0.0015)
A5 = pd.Series(213_000 * np.exp(0.67 * lbb + uu), ix).round(-2); B5 = pd.Series(1_300_000 * np.exp(lbb), ix).round(-3)
df5 = pd.DataFrame({"a_open": A5.shift(1).fillna(A5), "a_close": A5, "a_high": A5 * 1.001, "a_low": A5 * 0.999,
                    "b_open": B5.shift(1).fillna(B5), "b_close": B5, "b_high": B5 * 1.001, "b_low": B5 * 0.999})
pl5 = pc.plan_days(A5, B5, pc.Params(), ca)
f5 = pc.features_ou(A5, B5, pl5)
T5 = bt.simulate(df5, f5, pc.Params(), pc.Costs(), plans=pl5, passive="auto")
check("42 백테스트 auto → 삼성 지정가", len(T5) > 0 and set(T5["passive_leg"].dropna()) == {"a"}, (len(T5), T5["passive_leg"].value_counts().to_dict() if len(T5) else None))


def t43(tmp):
    bk = Book(); FakeRunner.script = []
    QQ = lambda s: {"last": 213_000.0, "bid": 212_500.0, "ask": 213_000.0} if s == "A" else {"last": 1_300_000.0, "bid": 1_299_000.0, "ask": 1_300_000.0}
    r = FakeRunner(args(passive="auto"), ps.Rec(Path(tmp)), A, B, None, QQ, bk, pd.DataFrame())
    tick(r, 1, -2.2)
    first = bk.sent[0]
    check("43 실행기 auto → 첫 주문 삼성 지정가(매수1호가)", first[0] == "A" and first[1] == "buy" and first[3] == 212_500.0
          and r.sig.pos is not None, bk.sent)


case(t43)
print(f"모두 통과 ({ok}개)")
