"""6주차 시험 — 키·네트워크 없이 돎.  python test6.py   (또는 pytest test6.py)

손계산으로 결과를 미리 적어 두고 엔진·감시·실행기가 그대로 하는지 봅니다.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import tempfile

import pandas as pd

import pair_core as pc
import rules6 as r6
from monitor6 import Watch, WatchCfg

P = replace(pc.Params(), entry=1.5, exit=0.0, max_hold=60, break_corr=0.2)   # 진입 1.5, 수렴 0, 보유 60분
T0 = pd.Timestamp("2026-11-23 10:00")


def eng(rules=None, **kw):
    e = r6.Engine6(rules or r6.Rules6(**kw), p=P, mult=10)
    e.set_day(T0.date(), P)
    return e


def t(m):
    return T0 + pd.Timedelta(minutes=m)


def fill(e, o, m, z, pa=60_000.0, pb=600_000.0):
    """주문을 바로 체결시킨 셈 (1단위 = 삼성 10 + 하이닉스 1)."""
    if o.kind == "open":
        e.opened(t(m), o.units, 10 * o.units, o.units, pa, pb, z, o.dir, o.tag)
    else:
        e.closed(t(m), o.tids, -1000.0, o.why, z)


# ── 엔진 ─────────────────────────────────────────────────────
def test_split_entry_exit_protect():
    e = eng()
    o = e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)                    # z −1.6 ≤ −1.5 → 1차 1단위 롱 스프레드
    assert [(x.kind, x.units, x.dir, x.tag) for x in o] == [("open", 1, +1, "1차")]
    fill(e, o[0], 1, -1.6)
    assert e.on_bar(t(2), -1.8, 0.9, 6e4, 6e5) == []            # a+0.5 = 2.0 전 → 아무것도 안 함
    o = e.on_bar(t(3), -2.05, 0.9, 6e4, 6e5)                   # −2.05 ≤ −2.0 → 2차
    assert [(x.kind, x.tag) for x in o] == [("open", "2차")]
    fill(e, o[0], 4, -2.05)
    assert e.units == 2
    o = e.on_bar(t(10), -0.7, 0.9, 6e4, 6e5)                   # 중간선 (1.5+0)/2 = 0.75 안 → 1차 트랜치만 청산
    assert o[0].why == "분할청산1" and o[0].tids == (1,)
    fill(e, o[0], 11, -0.7)
    assert e.units == 1 and e.partial_done
    o = e.on_bar(t(15), -1.55, 0.9, 6e4, 6e5)                  # 다시 진입선 1.5 밖 → 이익 보호
    assert o[0].why == "이익보호" and o[0].tids == (2,)


def test_split_then_converge():
    e = eng()
    fill(e, e.on_bar(t(0), 1.6, 0.9, 6e4, 6e5)[0], 1, 1.6)      # 숏 스프레드
    assert e.dir == -1
    fill(e, e.on_bar(t(2), 2.1, 0.9, 6e4, 6e5)[0], 3, 2.1)
    o = e.on_bar(t(5), -0.01, 0.9, 6e4, 6e5)                   # 한 번에 수렴선까지 → 둘 다 수렴
    assert o[0].why == "수렴" and set(o[0].tids) == {1, 2}


def test_price_stop_and_no_entry_beyond_stop():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    o = e.on_bar(t(2), -3.0, 0.9, 6e4, 6e5)                    # 1.5 + 1.5 = 3.0 → 가격 손절
    assert o[0].why == "가격손절"
    fill(e, o[0], 3, -3.0)
    assert e.on_bar(t(20), -3.1, 0.9, 6e4, 6e5) == []           # 손절선 밖에서는 새로 안 들어감


def test_no_risk_rides_blowout():
    e = eng(r6.no_risk())
    o = e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)
    assert o[0].units == 2 and o[0].tag == "한번에"            # 위험관리 없음: 2단위 한 번에
    fill(e, o[0], 1, -1.6)
    assert e.on_bar(t(2), -5.0, 0.0, 6e4, 6e5) == []            # z −5, 상관 0 이어도 안 나옴
    assert e.on_bar(t(59), -5.0, 0.0, 6e4, 6e5) == []
    assert e.on_bar(pd.Timestamp("2026-11-23 15:15"), -5.0, 0.9, 6e4, 6e5)[0].why == "장마감"


def test_money_stop():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 60_000, 600_000)[0], 1, -1.6, 60_000, 600_000)
    # 롱 스프레드 = 삼성 10 매수 · 하이닉스 1 매도. 삼성 −3,000원 → −3,000×10×10 = −30만 → 한도(1단위 × 30만) 닿음
    o = e.on_bar(t(2), -2.5, 0.9, 57_000, 600_000)
    assert o[0].why == "금액손절"


def test_time_stop():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    assert e.on_bar(t(60), -1.6, 0.9, 6e4, 6e5) == []           # 59분 보유
    assert e.on_bar(t(61), -1.6, 0.9, 6e4, 6e5)[0].why == "시간손절"


def test_second_tranche_only_early():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    assert e.on_bar(t(31), -2.1, 0.9, 6e4, 6e5) == []           # 최대 보유 60분의 절반(30분) 지나면 2차 안 함


def test_day_loss_limit():
    e = eng(day_loss=600_000)
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    e.closed(t(5), (1,), -650_000, "수렴", -0.1)
    assert e.halted.startswith("하루손실한도")
    assert e.on_bar(t(30), -1.7, 0.9, 6e4, 6e5) == []
    e.set_day(T0.date() + pd.Timedelta(days=1), P)               # 다음 날 풀림
    assert e.halted == ""


def test_day_stop_count():
    e = eng()
    for k in range(2):
        fill(e, e.on_bar(t(20 * k), -1.6, 0.9, 6e4, 6e5)[0], 20 * k + 1, -1.6)
        e.closed(t(20 * k + 2), tuple(x.tid for x in e.tranches), -10_000, "가격손절", -3.0)
    assert e.halted.startswith("하루손절횟수")


def test_drawdown_modes():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    e.closed(t(2), (1,), -1_600_000, "수렴", 0)                  # 고점 0 → −160만: half
    assert e.dd_mode == "half" and e.cap_units() == 1
    e.halted = ""                                                # (하루 한도는 따로 — 여기선 낙폭만 봄)
    e.last_exit = None
    fill(e, e.on_bar(t(30), -1.6, 0.9, 6e4, 6e5)[0], 31, -1.6)
    assert e.on_bar(t(32), -2.2, 0.9, 6e4, 6e5) == []            # 2차 자리지만 최대 1단위
    assert any(x["규칙"] == "분할제한" for x in e.events)
    e.closed(t(33), (2,), -1_000_000, "가격손절", -3)             # −260만 → stop
    assert e.dd_mode == "stop" and e.cap_units() == 0
    e.set_day(T0.date() + pd.Timedelta(days=1), P)
    assert e.dd_mode == "half"                                   # 다음 날 1단위로 재개
    e.equity = e.peak - 900_000
    e._limits(t(0))
    assert e.dd_mode == "normal"                                 # −100만 위로 회복


def test_block_stops_entry_not_exit():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    e.block(t(1), "연결끊김")
    assert e.on_bar(t(2), -0.0, 0.9, 6e4, 6e5)[0].why == "수렴"   # 청산은 막지 않음
    fill(e, e.on_bar(t(2), 0.0, 0.9, 6e4, 6e5)[0], 3, 0.0)
    assert e.on_bar(t(30), -1.7, 0.9, 6e4, 6e5) == []           # 신규는 막음
    e.unblock(t(31))
    assert e.on_bar(t(32), -1.7, 0.9, 6e4, 6e5)[0].kind == "open"


def test_state_roundtrip():
    e = eng()
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    s = e.to_state()
    e2 = r6.Engine6(r6.Rules6(), p=P)
    e2.load_state(s)
    assert e2.units == 1 and e2.dir == 1 and e2.tranches[0].t_in == t(1)


# ── 감시 ─────────────────────────────────────────────────────
class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def watch():
    c, log, blk = Clock(), [], []
    w = Watch(WatchCfg(), ("A", "B"), log=lambda k, **kw: log.append((k, kw)),
              block=lambda why: blk.append(("block", why)), unblock=lambda why: blk.append(("unblock", why)),
              clock=c, hhmm=lambda: "10:00")
    return w, c, log, blk


def test_watch_disconnect_and_refill():
    w, c, log, blk = watch()
    w.on_state("live"); w.on_tick("A", 1); w.on_tick("B", 1)
    w.on_state("reconnecting")
    c.now += 19; w.check()
    assert blk == []
    c.now += 2; w.check()
    assert blk[-1][0] == "block" and "연결끊김" in blk[-1][1]
    w.on_state("live"); w.on_tick("A", 1); w.on_tick("B", 1); w.check()
    assert blk[-1][0] == "unblock" and w.need_refill


def test_watch_stale_quotes():
    w, c, log, blk = watch()
    w.on_state("live"); w.on_tick("A", 1); w.on_tick("B", 1)
    c.now += 100; w.on_tick("A", 1); c.now += 25; w.check()
    assert "시세정지" in w.issues and "['B']" in w.issues["시세정지"]
    w.on_tick("B", 1); w.check()
    assert "시세정지" not in w.issues and blk[-1][0] == "unblock"


def test_watch_reject_is_sticky():
    w, c, log, blk = watch()
    w.on_state("live"); w.on_tick("A", 1); w.on_tick("B", 1)
    w.track("123", "A", "buy", 9)
    w.on_order({"id": "123", "status": "rejected", "errorCode": "E99"})
    assert "주문거부" in w.issues
    c.now += 5; w.on_tick("A", 1); w.on_tick("B", 1); w.check()
    assert "주문거부" in w.issues and blk[-1][0] == "block"      # 자동으로 안 풀림
    w.ack()
    assert w.issues == {} and blk[-1][0] == "unblock"


def test_watch_unfilled():
    w, c, log, blk = watch()
    w.on_state("live"); w.on_tick("A", 1); w.on_tick("B", 1)
    w.track("9", "B", "sell", 1)
    c.now += 11; w.on_tick("A", 1); w.on_tick("B", 1); w.check()
    assert any(k == "미체결경보" for k, _ in log) and "미체결" not in w.issues
    c.now += 20; w.on_tick("A", 1); w.on_tick("B", 1); w.check()
    assert "미체결" in w.issues
    w2, c2, _, _ = watch()
    w2.track("9", "B", "sell", 1)
    w2.on_order({"id": "9", "status": "filled", "filled": 1, "remaining": 0})
    c2.now += 40; w2.check()
    assert "미체결" not in w2.issues


def test_tick_stop_fires_once():
    w, c, log, blk = watch()
    fired = []
    w.urgent_exit = fired.append
    mu, se = math.log(60_000) - math.log(600_000), 0.001
    w.set_guard({"dir": +1, "mu": mu, "se": se, "entry": 1.5, "stop_gap": 1.5, "loss_lim": 0, "mult": 10,
                 "tranches": [(10, 1, 60_000, 600_000)]})
    w.on_tick("A", 60_000); w.on_tick("B", 600_000)
    assert fired == []
    w.on_tick("A", 60_000 * math.exp(-0.0031))                     # z = −3.1 ≤ −3.0
    w.on_tick("A", 60_000 * math.exp(-0.0035))
    assert fired == ["틱 가격손절"]                                 # 한 번만


def test_tick_money_stop():
    w, *_ = watch()
    fired = []
    w.urgent_exit = fired.append
    w.set_guard({"dir": +1, "mu": 0, "se": 1.0, "entry": 1.5, "stop_gap": 99, "loss_lim": 300_000, "mult": 10,
                 "tranches": [(10, 1, 60_000, 600_000)]})
    w.on_tick("B", 600_000); w.on_tick("A", 57_100)                 # −2,900 × 10 × 10 = −29만 → 아직
    assert fired == []
    w.on_tick("A", 57_000)                                         # −30만
    assert fired == ["틱 금액손절"]


# ── 11/22 개선 규칙 ───────────────────────────────────────────
def test_anchor_stop_catches_drifting_mean():
    """이동평균이 포지션을 따라와 이동 z 는 −1.0 인데, 진입 시점 평균 기준으로는 −3.1 → 고정 z 손절."""
    for on, want in ((True, "고정z손절"), (False, None)):
        e = eng(anchor_stop=on)
        o = e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5, s=-0.0016, se=0.001)        # μ₀ = s − z·σ = 0
        fill(e, o[0], 1, -1.6)
        assert abs(e.mu0) < 1e-12
        o = e.on_bar(t(20), -1.0, 0.9, 6e4, 6e5, s=-0.0031, se=0.001)
        assert (o[0].why if o else None) == want


def test_anchor_second_tranche():
    e = eng(anchor_stop=True)
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5, s=-0.0016, se=0.001)[0], 1, -1.6)
    o = e.on_bar(t(5), -1.7, 0.9, 6e4, 6e5, s=-0.0021, se=0.001)           # 이동 z −1.7 (2차 자리 아님), 고정 z −2.1
    assert [(x.kind, x.tag) for x in o] == [("open", "2차")]


def test_pct_money_stop():
    e = eng(r6.improved(gap_skip=None, skip_expiry=False))
    fill(e, e.on_bar(t(0), -1.6, 0.9, 60_000, 600_000)[0], 1, -1.6, 60_000, 600_000)
    assert e.ref_unit == 12_000_000                                        # (10×60,000 + 1×600,000) × 10
    assert e.loss_limit() == 156_000                                       # 1.3%
    assert e.on_bar(t(2), -1.8, 0.9, 58_500, 600_000) == []                # −15만
    assert e.on_bar(t(3), -1.8, 0.9, 58_400, 600_000)[0].why == "금액손절"   # −16만


def test_shadow_resume():
    R = r6.improved(gap_skip=None, skip_expiry=False, shadow_n=2, dd_half_pct=0.005, dd_stop_pct=0.01,
                    dd_recover_pct=0.003, day_stops=None, day_loss_pct=None)
    e = eng(R)
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    e.closed(t(2), (1,), -200_000, "수렴", 0)                               # 1단위 명목 1,200만의 1% = −12만 넘음
    assert e.dd_mode == "stop" and e.shadow and e.cap_units() == 1
    for k, net in ((1, 50_000), (2, 10_000)):
        o = e.on_bar(t(20 * k), -1.6, 0.9, 6e4, 6e5)
        fill(e, o[0], 20 * k + 1, -1.6)
        assert e.tranches[0].shadow
        e.closed(t(20 * k + 2), tuple(x.tid for x in e.tranches), net, "수렴", 0)
        assert e.equity == -200_000 and e.day_pnl == -200_000              # 가상 손익은 안 들어감
    assert e.dd_mode == "half" and not e.shadow and e.peak == e.equity     # 최근 2건 합 +6만 → 재개


def test_manual_stop_stays_until_resume():
    e = eng(r6.improved(gap_skip=None, skip_expiry=False, dd_resume="manual", dd_stop_pct=0.01, dd_half_pct=0.005,
                        dd_recover_pct=0.003))
    fill(e, e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5)[0], 1, -1.6)
    e.closed(t(2), (1,), -200_000, "수렴", 0)
    e.set_day(T0.date() + pd.Timedelta(days=1), P)
    assert e.dd_mode == "stop" and e.on_bar(t(24 * 60 + 30), -1.6, 0.9, 6e4, 6e5) == []
    e.resume(t(24 * 60 + 31))
    assert e.dd_mode == "half" and e.on_bar(t(24 * 60 + 32), -1.6, 0.9, 6e4, 6e5)[0].kind == "open"


def test_day_skip_gap_and_expiry():
    import datetime as dt
    R = r6.improved()
    assert r6.is_expiry(dt.date(2026, 11, 12)) and not r6.is_expiry(dt.date(2026, 11, 5))   # 2026-11 둘째 목요일 = 12일
    assert r6.day_skip(R, dt.date(2026, 11, 12), 0.1).startswith("만기일")
    assert r6.day_skip(R, dt.date(2026, 11, 23), -2.5).startswith("갭필터")
    assert r6.day_skip(R, dt.date(2026, 11, 23), 1.9) == ""
    e = r6.Engine6(R, p=P, mult=10)
    e.set_day(T0.date(), P, "갭필터 (밤사이 스프레드 갭 -2.5σ)")
    assert e.on_bar(t(0), -1.6, 0.9, 6e4, 6e5) == []


def test_watch_own_cancel_is_not_unfilled():
    w, c, log, blk = watch()
    w.on_state("live"); w.on_tick("A", 1); w.on_tick("B", 1)
    w.track("7", "A", "buy", 9)
    w.cancelled("7")                                                       # 실행기가 6초 뒤 취소하고 재주문
    c.now += 40; w.on_tick("A", 1); w.on_tick("B", 1); w.check()
    assert "미체결" not in w.issues


def test_watch_down_for():
    w, c, *_ = watch()
    assert w.down_for() == 0
    w.on_state("live"); w.on_state("reconnecting"); c.now += 75
    assert w.down_for() == 75
    w.on_state("live")
    assert w.down_for() == 0


def test_tick_anchor_stop():
    w, *_ = watch()
    fired = []
    w.urgent_exit = fired.append
    base = math.log(60_000) - math.log(600_000)
    w.set_guard({"dir": +1, "mu": base - 0.0025, "mu0": base, "se": 0.001, "entry": 1.5, "stop_gap": 1.5, "loss_lim": 0,
                 "mult": 10, "tranches": [(10, 1, 60_000, 600_000)]})     # 이동평균은 −2.5σ 만큼 따라옴
    w.on_tick("B", 600_000); w.on_tick("A", 60_000 * math.exp(-0.0031))   # 이동 z −0.6, 고정 z −3.1
    assert fired == ["틱 고정z손절"]


# ── 실행기 (종이 체결, 네트워크 없음) ─────────────────────────
class _A:
    entry, exit, stop, window, loss_limit, day_loss, unit_b = 2.0, 0.5, 3.5, 120, 300_000, 600_000, 1
    fee_rate, ou, spread, trend_filter, passive, passive_b, risk = 0.00003, False, "ratio", False, "none", False, False
    units_max, loss_stop_pct, day_loss_pct, stop_at, ack = 2, 0.013, 0.026, "15:30", False


Q = {"SA": {"last": 60_000.0, "bid": 59_900.0, "ask": 60_000.0}, "SB": {"last": 600_000.0, "bid": 599_000.0, "ask": 600_000.0}}
ROW = pd.Series({"beta": 1.0, "z": -2.1, "a": 60_000.0, "b": 600_000.0, "corr": 0.9, "spread": 0.0, "sd": 0.001})


def mk_runner(d, day="x", broker=None, ack=False):
    import pair_spread as ps
    from runner6 import Runner6
    a = _A(); a.ack = ack
    broker = broker or pc.PaperFillBroker(lambda s: (Q[s]["bid"], Q[s]["ask"]))
    R = ps.Rec(Path(d) / day)
    logs = []
    R.log = lambda k, **kw: logs.append((k, kw))
    w, c, *_ = watch()
    run = Runner6(a, R, {"sym": "SA", "mult": 10}, {"sym": "SB", "mult": 10}, None, lambda s: Q[s], broker,
                  pd.DataFrame(columns=["a", "b"]), w)
    run.ex.sleep = lambda s: None
    return run, broker, w, c, logs


def test_runner_split_orders_and_reconcile():
    with tempfile.TemporaryDirectory() as d:
        run, broker, w, *_ = mk_runner(d)
        run._open(r6.Order("open", units=1, dir=+1, tag="1차"), ROW, t(0))
        run._open(r6.Order("open", units=1, dir=+1, tag="2차"), ROW, t(1))
        assert run.eng.units == 2 and run.mine() == {"SA": 20, "SB": -2}
        assert broker.net_positions() == {"SA": 20, "SB": -2}
        assert len(w.orders) == 4                                   # 다리 4개 주문 모두 감시 목록에
        run._close((1,), "분할청산1", t(5))
        assert run.mine() == {"SA": 10, "SB": -1} and broker.net_positions() == {"SA": 10, "SB": -1}
        assert run.eng.partial_done
        tr = pd.read_csv(Path(d) / "x" / "trades.csv")
        assert list(tr["트랜치"]) == ["1차"]


def test_runner_margin_check():
    class B(pc.PaperFillBroker):
        cap = {"SA": 10, "SB": 5}

        def max_qty(self, sym, side, px):
            return self.cap[sym]

    with tempfile.TemporaryDirectory() as d:
        b = B(lambda s: (Q[s]["bid"], Q[s]["ask"]))
        run, _, _, _, logs = mk_runner(d, broker=b)
        run._open(r6.Order("open", units=2, dir=+1, tag="한번에"), ROW, t(0))   # 삼성 10계약 = 1단위만 가능
        assert run.eng.units == 1 and any(k == "증거금확인" and kw["가능단위"] == 1 for k, kw in logs)
        b.cap = {"SA": 3, "SB": 5}
        run._open(r6.Order("open", units=1, dir=+1, tag="2차"), ROW, t(1))      # 0단위 → 주문 안 냄
        assert run.eng.units == 1 and any(k == "진입보류" for k, _ in logs)


def test_runner_state_carries_across_days():
    with tempfile.TemporaryDirectory() as d:
        run, *_ = mk_runner(d, "20261123")
        run.eng.equity, run.eng.peak, run.eng.dd_mode = -2_600_000.0, 0.0, "stop"
        run.halt("레그실패 — 시험")
        run.eng.blocked = "연결끊김: 시험"                                     # 자동 해제 항목은 날을 안 넘김
        run._save()
        run2, _, w2, *_ = mk_runner(d, "20261124")                              # 다음 날 (새 날짜 폴더)
        assert run2.eng.equity == -2_600_000.0 and run2.eng.dd_mode == "stop"
        assert run2.eng.blocked == "레그실패 — 시험" and "사람확인" in w2.issues
        run3, *_ = mk_runner(d, "20261125", ack=True)                           # HTS 확인 뒤 --ack
        assert run3.eng.blocked == "" and run3.eng.dd_mode == "half" and run3.eng.peak == run3.eng.equity


def test_runner_one_leg_escalation():
    import time as _t
    with tempfile.TemporaryDirectory() as d:
        run, _, w, _, logs = mk_runner(d)
        run._open(r6.Order("open", units=1, dir=+1, tag="1차"), ROW, t(0))
        left = [{"sym": "SA", "side": "sell", "qty": 10}]                       # 하이닉스만 청산되고 삼성 매수 10계약이 남음
        run.pending_close = {"tids": [1], "why": "수렴", "left": left, "fills": {}, "t0": _t.time() - 5, "ref": {"SA": 60_000.0}}
        assert run._naked() == 1
        w.price["SA"] = 58_000.0                                                # −2,000 × 10 × 10 = −20만 ≥ 기준(금액 손절 30만 ÷ 2 = 15만)
        assert run._naked() == 2 and run.pending_close["fast"]
        run.pending_close["t0"] = _t.time() - 61
        assert run._naked() == 3 and "한 다리" in run.eng.blocked and run.sticky


def test_runner_one_leg_limit_scales_with_units():
    """2단위를 닫다가 한 다리만 남으면 기준은 30만 (1단위 15만 × 2) — 1분 정상 변동(12만~14만)에서 안 걸림."""
    import time as _t
    with tempfile.TemporaryDirectory() as d:
        run, _, w, _, logs = mk_runner(d)
        run._open(r6.Order("open", units=1, dir=+1, tag="1차"), ROW, t(0))
        run._open(r6.Order("open", units=1, dir=+1, tag="2차"), ROW, t(1))
        left = [{"sym": "SA", "side": "sell", "qty": 20}]                       # 삼성 매수 20계약이 남음
        run.pending_close = {"tids": [1, 2], "why": "수렴", "left": left, "fills": {}, "t0": _t.time() - 5, "ref": {"SA": 60_000.0}}
        w.price["SA"] = 59_000.0                                                # −1,000 × 20 × 10 = −20만 < 30만
        assert run._naked() == 1
        w.price["SA"] = 58_400.0                                                # −32만 ≥ 30만
        assert run._naked() == 2


def test_runner_stream_down_escalation():
    with tempfile.TemporaryDirectory() as d:
        run, broker, w, c, logs = mk_runner(d)
        run._open(r6.Order("open", units=1, dir=+1, tag="1차"), ROW, t(0))
        run._open(r6.Order("open", units=1, dir=+1, tag="2차"), ROW, t(1))
        w.on_state("live"); w.on_state("reconnecting")
        c.now += 59; run._stream_down()
        assert run.poll == 20
        c.now += 2; run._stream_down()
        assert run.poll == 5 and run.eng.units == 2                            # 61초: REST 5초, 아직 안 줄임
        c.now += 120; run._stream_down()
        assert run.eng.units == 1 and [x.tag for x in run.eng.tranches] == ["1차"]   # 181초: 2차만 청산
        assert broker.net_positions() == {"SA": 10, "SB": -1}
        w.on_state("live"); run._stream_down()
        assert run.poll == 20


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print("통과", f.__name__)
        except Exception as e:
            bad += 1
            import traceback; traceback.print_exc()
            print("실패", f.__name__, e)
    print(f"{len(fns) - bad}/{len(fns)} 통과")
    sys.exit(1 if bad else 0)
