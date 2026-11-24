"""open_limit.py 시험. 실행: python3 -m pytest week05/test_open_limit.py -q"""
import datetime as dt

from open_limit import K200, KRX300, MINI, OpenLimit, Params, Quote, ReplayBroker, ask_price, bid_price, quote_from_snap

R = 4.0
P = Params(ratio=R, d_ticks=20)


def snap(hms, kb=800.0, ka=800.05, xb=3100.0, xa=3300.0, last=3200.0, vol=0, mb=799.9, ma=800.1):
    return {"t": f"2026-11-23T{hms}.000+09:00", KRX300: {"b": [[xb, 5]] if xb else [], "a": [[xa, 5]] if xa else [], "last": last, "vol": vol},
            K200: {"b": [[kb, 3]] if kb else [], "a": [[ka, 3]] if ka else []}, MINI: {"b": [[mb, 3]] if mb else [], "a": [[ma, 3]] if ma else []}}


def run(snaps, p=P, broker=None):
    br = (broker or ReplayBroker()); log = []; eng = OpenLimit(p, br, log=lambda k, **kw: log.append((k, kw)), sleep=lambda s: None)
    for s in snaps:
        br.advance(s); eng.step(quote_from_snap(s))
    return eng, br, log


def test_prices_round_away_from_fair():
    assert bid_price(3200.1, 20) == 3196.0          # 3196.1 → 내림
    assert ask_price(3200.1, 20) == 3204.2          # 3204.1 → 올림
    assert bid_price(3200.0, 20) == 3196.0 and ask_price(3200.0, 20) == 3204.0


def test_no_order_before_start():
    eng, br, _ = run([snap("08:44:55")])
    assert eng.state == "대기" and not br.sent


def test_quotes_both_sides_at_fair_minus_plus_d():
    eng, br, _ = run([snap("08:45:00")])
    fair = R * 800.025
    assert {(s[2], s[4]) for s in br.sent} == {("buy", bid_price(fair, 20)), ("sell", ask_price(fair, 20))}
    assert eng.state == "호가중"


def test_requote_only_when_fair_moves_enough():
    eng, br, _ = run([snap("08:45:00"), snap("08:45:05", kb=800.05, ka=800.10), snap("08:45:10", kb=801.0, ka=801.05)])
    assert len(br.sent) == 4                         # 처음 2개 + 크게 움직인 뒤 다시 2개 (0.05pt = 1틱 이동은 그대로 둠)
    assert len(br.resting) == 2


def test_buy_fill_hedges_with_mini_and_cancels_other_side():
    fair = R * 800.025; b = bid_price(fair, 20)
    eng, br, log = run([snap("08:45:00"), snap("08:48:00", last=b - 0.4, vol=1)])
    assert eng.state == "보유" and eng.entry["side"] == "buy" and eng.entry["price"] == b
    assert eng.hedge == {"side": "sell", "qty": 4, "price": 799.9, "주문번호": eng.hedge["주문번호"], "주문수량": 4}
    assert not br.resting                            # 반대쪽 매도 지정가는 취소됨


def test_no_fill_without_trade_even_if_last_is_low():
    eng, _, _ = run([snap("08:45:00", last=3000.0, vol=7), snap("08:45:05", last=3000.0, vol=7)])
    assert eng.state == "호가중"                      # 누적 거래량이 안 늘었으면 체결 아님


def test_stop_cancels_everything_when_unfilled():
    eng, br, _ = run([snap("08:45:00"), snap("08:59:30")])
    assert eng.state == "철수확인" and not br.resting           # 취소를 내고, 다음 번에 체결 여부를 한 번 더 확인
    br.advance(snap("08:59:33")); eng.step(quote_from_snap(snap("08:59:33")))
    assert eng.state == "끝" and eng.pnl()["거래"] == 0


def test_exit_at_0915_closes_both_legs_and_pnl_matches():
    fair = R * 800.025; b = bid_price(fair, 20)
    eng, br, _ = run([snap("08:45:00"), snap("08:48:00", last=b, vol=1), snap("09:00:00", vol=1), snap("09:15:00", xb=3199.0, xa=3202.0, vol=1, mb=799.9, ma=800.1)])
    assert eng.state == "끝" and len(eng.exits) == 2
    res = eng.pnl()
    krx = (3199.0 - b) * 50_000                      # 매수 b → 매수호가 3199.0 에 매도
    mini = (799.9 - 800.1) * 4 * 50_000              # 799.9 매도 → 800.1 매수
    assert abs(res["손익"] - (krx + mini)) < 1e-6 and res["남은 포지션"] == {}


def test_sell_fill_side():
    fair = R * 800.025; a = ask_price(fair, 20)
    eng, _, _ = run([snap("08:45:00"), snap("08:50:00", last=a + 1.0, vol=2)])
    assert eng.entry["side"] == "sell" and eng.hedge["side"] == "buy" and eng.hedge["price"] == 800.1


def test_wide_k200_quote_pulls_orders():
    eng, br, _ = run([snap("08:45:00"), snap("08:45:05", kb=795.0, ka=805.0)])
    assert not br.resting and eng.state == "호가중"


def test_side_skipped_when_it_would_cross_the_book():
    eng, br, _ = run([snap("08:45:00", xb=3100.0, xa=3190.0)])       # 내 매수 목표(약 3196)가 상대 매도호가 3190 위 → 매수는 안 냄
    assert [s[2] for s in br.sent] == ["sell"]


def test_hedge_reject_is_reported_not_hidden():
    fair = R * 800.025; b = bid_price(fair, 20)
    eng, _, _ = run([snap("08:45:00"), snap("08:48:00", last=b, vol=1, mb=None, ma=None)])
    assert eng.state == "보유" and eng.hedge["qty"] == 0 and "헤지 주문 거부" in eng.note


def test_one_fill_per_day():
    fair = R * 800.025; b = bid_price(fair, 20)
    eng, br, _ = run([snap("08:45:00"), snap("08:48:00", last=b, vol=1), snap("08:50:00", last=b - 5, vol=3)])
    assert len([s for s in br.sent if s[1] == KRX300]) == 2          # 처음 건 매수·매도뿐, 체결 뒤 새 지정가 없음


class StickyBroker(ReplayBroker):
    """취소 요청이 먹지 않는 브로커 — 취소와 체결이 엇갈리는 경우를 흉내."""
    def cancel(self, no, sym, side):
        pass


def test_fill_on_an_order_we_tried_to_cancel_is_still_caught():
    fair = R * 800.025; b0 = bid_price(fair, 20)
    snaps = [snap("08:45:00"), snap("08:45:05", kb=801.0, ka=801.05),          # 적정가 이동 → 옛 주문 취소(먹지 않음) 후 새로 걺
             snap("08:45:10", kb=801.0, ka=801.05, last=b0 + 2.0, vol=1)]      # 옛 매수가(b0)보다는 높고 새 매수가보다는 낮은 체결 → 새 주문만 체결
    eng, br, _ = run(snaps, broker=StickyBroker())
    assert eng.state == "보유" and eng.entry["side"] == "buy"
    eng2, br2, _ = run([snap("08:45:00"), snap("08:45:05", kb=801.0, ka=801.05), snap("08:45:10", kb=801.0, ka=801.05, last=b0 - 0.2, vol=1)], broker=StickyBroker())
    assert eng2.entry["qty"] == 2                                                # 취소한 줄 알았던 옛 주문까지 체결 → 2계약으로 잡고
    assert eng2.hedge["주문수량"] == 8                                            # 헤지도 그만큼


def test_stop_then_late_fill_becomes_a_position():
    fair = R * 800.025; b = bid_price(fair, 20)
    eng, br, _ = run([snap("08:45:00"), snap("08:59:30"), snap("08:59:33", last=b, vol=1)], broker=StickyBroker())
    assert eng.state == "보유" and eng.hedge["qty"] == 4                          # 취소가 늦어 체결된 것을 놓치지 않고 헤지


def test_both_sides_filled_ends_flat_without_hedge():
    fair = R * 800.025; b = bid_price(fair, 20); a = ask_price(fair, 20)
    br = StickyBroker(); eng = OpenLimit(P, br, sleep=lambda s: None)
    s0 = snap("08:45:00"); br.advance(s0); eng.step(quote_from_snap(s0))
    nos = {o[2]: o[0] for o in br.sent}
    br.orders[nos["buy"]] = (1, b); br.orders[nos["sell"]] = (1, a)               # 같은 구간에 양쪽 다 체결
    s1 = snap("08:45:05", vol=2); br.snap = s1; eng.step(quote_from_snap(s1))
    assert eng.state == "끝" and eng.hedge is None
    assert abs(eng.pnl()["손익"] - (a - b) * 50_000) < 1e-6 and eng.pnl()["남은 포지션"] == {}


def test_unfilled_market_hedge_is_reported():
    class NoFill(ReplayBroker):
        def send(self, sym, side, qty, price=None):
            no = super().send(sym, side, qty, price)
            if price is None and sym == MINI:
                self.orders[no] = (0, 0.0)                                       # 헤지 시장가가 접수만 되고 체결 안 됨
            return no
    fair = R * 800.025; b = bid_price(fair, 20)
    eng, _, _ = run([snap("08:45:00"), snap("08:48:00", last=b, vol=1)], broker=NoFill())
    assert eng.hedge["qty"] == 0 and "헤지 미완료" in eng.note and eng.pnl()["손익"] is None


# ── 2026-11-23 모의계좌 첫 실행에서 드러난 문제들 ─────────────────────────
class PaperLike(ReplayBroker):
    """모의 서버처럼: KRX300 시장가는 거부, 정해 둔 주문번호의 취소는 정해 둔 횟수만큼 거부."""

    def __init__(self, refuse_cancel=None):
        super().__init__(); self.refuse = dict(refuse_cancel or {}); self.cancel_calls = []

    def send(self, sym, side, qty, price=None):
        if sym == KRX300 and price is None:
            raise RuntimeError("(시장가)호가유형별 호가입력제한오류")
        return super().send(sym, side, qty, price)

    def cancel(self, no, sym, side):
        self.cancel_calls.append(no)
        if self.refuse.get(no, 0) > 0:
            self.refuse[no] -= 1
            raise RuntimeError("원주문 접수확인 전이거나 잔량이 없는 경우")
        super().cancel(no, sym, side)


def test_krx300_exit_is_a_crossing_limit_not_a_market_order():
    fair = R * 800.025; s = ask_price(fair, 20)
    eng, br, log = run([snap("08:45:00"), snap("08:48:00", last=s + 0.4, vol=1),
                        snap("09:15:00", xb=3210.0, xa=3212.0, mb=803.0, ma=803.2)], broker=PaperLike())
    assert eng.state == "끝" and eng.pnl()["남은 포지션"] == {}
    no, sym, side, qty, price = [x for x in br.sent if x[1] == KRX300][-1]
    assert (side, price) == ("buy", round(3212.0 + P.cross_ticks * 0.2, 1))     # 매도호가를 넘기는 지정가
    assert br.filled(no) == (1, 3212.0)                                         # 체결은 상대 호가에


def test_exit_retries_and_keeps_the_hedge_until_krx300_is_flat():
    fair = R * 800.025; s = ask_price(fair, 20)
    eng, br, log = run([snap("08:45:00"), snap("08:48:00", last=s + 0.4, vol=1),
                        snap("09:15:00", xb=None, xa=None), snap("09:15:03", xb=None, xa=None)], broker=PaperLike())
    assert eng.state == "청산중" and eng.net() == {KRX300: -1, MINI: 4}          # KRX300 호가가 없으면 미니도 그대로 둠
    br.advance(snap("09:15:06", xb=3210.0, xa=3212.0)); eng.step(quote_from_snap(snap("09:15:06", xb=3210.0, xa=3212.0)))
    assert eng.state == "끝" and eng.pnl()["남은 포지션"] == {} and eng.pnl()["손익"] is not None


def test_exit_gives_up_after_the_try_limit_and_says_so():
    fair = R * 800.025; s = ask_price(fair, 20)
    p = Params(ratio=R, d_ticks=20, max_exit_tries=2)
    eng, _, _ = run([snap("08:45:00"), snap("08:48:00", last=s + 0.4, vol=1), snap("09:15:00", xb=None, xa=None),
                     snap("09:15:03", xb=None, xa=None)], p=p, broker=PaperLike())
    assert eng.state == "끝" and "청산 미완료" in eng.note and eng.pnl()["손익"] is None


def test_refused_cancel_keeps_the_order_and_blocks_a_second_one_on_that_side():
    br = PaperLike(refuse_cancel={"RP00002": 1})                                 # 첫 매도 지정가의 취소가 한 번 거부됨
    eng, br, log = run([snap("08:45:00"), snap("08:45:03", kb=801.0, ka=801.05), snap("08:45:06", kb=801.0, ka=801.05)], broker=br)
    sells = [x for x in br.sent if x[2] == "sell"]
    assert [k for k, _ in log].count("취소거부") == 1
    assert len(sells) == 2 and br.cancel_calls.count("RP00002") == 2             # 두 번째 시도에서 취소된 뒤에야 새 매도
    assert sum(1 for v in br.resting.values() if v[1] == "sell") == 1            # 살아 있는 매도는 하나뿐


def test_fill_on_an_order_whose_cancel_was_refused_is_hedged():
    fair = R * 800.025; s = ask_price(fair, 20)
    br = PaperLike(refuse_cancel={"RP00002": 99})
    eng, br, _ = run([snap("08:45:00"), snap("08:45:03", kb=801.0, ka=801.05), snap("08:45:06", kb=801.0, ka=801.05, last=s + 0.2, vol=1)], broker=br)
    assert eng.state == "보유" and eng.entry == {"side": "sell", "qty": 1, "price": s, "t": eng.entry["t"]}
    assert eng.hedge["side"] == "buy" and eng.hedge["qty"] == 4


def test_stop_does_not_finish_while_an_order_is_still_alive():
    br = PaperLike(refuse_cancel={"RP00001": 2})
    eng, br, _ = run([snap("08:45:00"), snap("08:59:30"), snap("08:59:33")], broker=br)
    assert eng.state == "철수확인" and len(br.resting) == 1                      # 취소가 두 번 거부돼 아직 살아 있음
    br.advance(snap("08:59:36")); eng.step(quote_from_snap(snap("08:59:36")))
    br.advance(snap("08:59:39")); eng.step(quote_from_snap(snap("08:59:39")))
    assert eng.state == "끝" and not br.resting


def test_late_opposite_fill_while_holding_is_netted_at_exit():
    fair = R * 800.025; b = bid_price(fair, 20); s = ask_price(fair, 20)
    br = PaperLike(refuse_cancel={"RP00002": 99})                                # 매도 지정가가 끝까지 안 빠짐
    eng, br, _ = run([snap("08:45:00"), snap("08:48:00", last=b - 0.2, vol=1), snap("08:50:00", last=s + 0.2, vol=2),
                      snap("09:15:00", vol=2, mb=803.0, ma=803.2)], broker=br)
    assert eng.state == "끝" and eng.pnl()["남은 포지션"] == {}                   # KRX300 은 이미 0, 미니 헤지만 정리
    assert [x for x in br.sent if x[1] == KRX300 and x[0] not in ("RP00001", "RP00002")] == []
