"""5주차 — 장 시작 지정가 전략을 NH 호가에 붙여 돌리는 실행 스크립트.

    python run_open_limit.py --app-dir <nh-futures-open-api 폴더>             # 예행연습: 실시간 호가, 주문 없음(체결은 가정)
    python run_open_limit.py --app-dir <...> --orders                        # NH 국내 **모의계좌**에 실제 주문

- 시세는 `--quote-env`(기본 real, 조회만), 주문은 4주차 `nh_pair.NHPaperBroker` 로 **모의(paper)만** 냅니다.
- 시작할 때 전 거래일 15:00~15:19 의 1분봉으로 비율을 계산합니다(`--ratio` 로 직접 줄 수도 있음).
- `--orders` 일 때는 시작 전에 모의계좌에 KRX300·미니 포지션이 없는지, 주문가능수량이 되는지 확인하고 아니면 주문 없이 끝냅니다.
- 끝나면 남은 지정가를 다시 취소하고, 모의계좌에 KRX300 포지션이 남아 있으면 상대 호가를 넘기는 지정가로 정리합니다.
  KRX300 이 0 이 된 뒤에만 미니를 시장가로 정리합니다(KRX300 을 못 정리하면 헤지를 남겨 둠).
- 결과: `results/<날짜>/live_events.jsonl`(사건), `live_book.jsonl`(조회한 호가, book_rec.py 와 같은 형식), `live_summary.json`
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

from open_limit import K200, KRX300, MINI, OpenLimit, Params, Quote, ReplayBroker, quote_from_snap

KST = dt.timezone(dt.timedelta(hours=9))


def now_kst() -> dt.datetime:
    return dt.datetime.now(KST).replace(tzinfo=None)


def hms(s: str) -> dt.time:
    return dt.datetime.strptime(s, "%H:%M:%S").time()


# ── 시세 ────────────────────────────────────────────────────────
def take_snap(F, Q, env: str) -> dict:
    """세 종목 호가와 KRX300·코스피200 현재가를 조회해 book_rec.py 와 같은 모양으로 돌려줌."""
    row = {"t": now_kst().isoformat(timespec="milliseconds") + "+09:00"}
    for sym in (KRX300, K200, MINI):
        b = Q.book(F.price_bid(sym, environment=env), asked=sym)
        row[sym] = {"bt": b.get("time"), "a": [(x["price"], x["quantity"]) for x in b.get("asks", [])],
                    "b": [(x["price"], x["quantity"]) for x in b.get("bids", [])]}
    for sym in (KRX300, K200):
        p = Q.quote(F.price(sym, environment=env), asked=sym)
        row[sym].update(last=p.get("last"), vol=p.get("volume"), oi=p.get("openInterest"), spot=p.get("spot"), theory=p.get("theory"))
    return row


def minute_closes(F, sym: str, day: str, env: str) -> dict[str, float]:
    """그 날 15:00~15:19 의 분별 종가 {HHMM: 값}."""
    r = F.chart_minute(sym, start_date=day, end_date=day, start_time="150000", end_time="151959", environment=env)
    out = {}
    for x in r.get("Occurs1") or []:
        tm = str(x.get("occurs1_exch_tm", "")).strip().zfill(6)[:4]
        if str(x.get("occurs1_ref_dt", "")).strip() == day and "1500" <= tm <= "1519":
            try:
                out[tm] = float(str(x.get("occurs1_close_pric")).strip())
            except (TypeError, ValueError):
                pass
    return out


def prev_ratio(F, env: str, today: dt.date, log) -> tuple[float, str, int]:
    """가장 최근 거래일의 15:00~15:19, 두 상품이 같은 분에 찍힌 종가 비율의 중앙값."""
    for back in range(1, 8):
        d = today - dt.timedelta(days=back)
        if d.weekday() >= 5:
            continue
        day = d.strftime("%Y%m%d")
        try:
            a, b = minute_closes(F, KRX300, day, env), minute_closes(F, K200, day, env)
        except Exception as e:
            log("비율조회실패", 날짜=day, 오류=str(e)[:200])
            continue
        common = sorted(set(a) & set(b))
        log("비율조회", 날짜=day, KRX300_분수=len(a), 코스피200_분수=len(b), 공통=len(common))
        if len(common) >= 5:
            return statistics.median(a[m] / b[m] for m in common), day, len(common)
    raise RuntimeError("최근 거래일의 15:00~15:19 공통 분봉이 5개 미만 — 비율을 계산할 수 없음")


# ── 주문내역을 끝 페이지까지 읽는 모의 브로커 ─────────────────────────
def paged_broker(nh, F):
    class PagedBroker(nh.NHPaperBroker):
        """주문·취소가 많아 주문내역이 여러 페이지로 나뉘어도 내 주문을 놓치지 않게 함."""

        def _history(self):
            if self._hist is None or time.time() - self._hist_t > 0.8:
                rows, key, seen = [], "", set()
                self._hist_full = False
                for _ in range(15):
                    r = self.F.order_history(next_key=key, environment=nh.ORDER_ENV)
                    got = r.get("Occurs1") or []
                    fresh = [x for x in got if (str(x.get("occurs1_ord_no", "")).strip(), str(x.get("occurs1_org_ord_no", "")).strip()) not in seen]
                    for x in fresh:
                        seen.add((str(x.get("occurs1_ord_no", "")).strip(), str(x.get("occurs1_org_ord_no", "")).strip()))
                    rows += fresh
                    key = str(r.get("nxt_key") or "").strip()
                    if not key or not fresh:
                        self._hist_full = True               # 끝 페이지까지 읽음 (15페이지 상한에 걸린 게 아님)
                        break
                self._hist, self._hist_t = rows, time.time()
            return self._hist

        def _row(self, no):
            for x in self._history():
                if str(x.get("occurs1_ord_no", "")).strip() == no:
                    return x
            return None

        def send(self, sym, side, qty, price=None):
            no = super().send(sym, side, qty, price)
            self._sent_at[no] = time.time()
            return no

        def open_qty(self, no):
            """그 주문의 미체결 잔량. 주문내역에 아직 없으면 살아 있는 것으로 봄."""
            x = self._row(no)
            if x is None:
                return 1
            rem = nh._n(x.get("occurs1_rem_ord_qty"))
            return int(rem) if rem == rem else 0

        def alive(self, no):
            """잔량이 남은 주문인지. 주문내역에 없으면: 내역을 끝까지 못 읽었거나 낸 지 60초 안이면 살아 있다고 봄."""
            if self._row(no) is None:
                return not self._hist_full or time.time() - self._sent_at.get(no, 0.0) < 60
            return self.open_qty(no) > 0

        def unacked(self, no):
            """모의 서버 「미접수」: 잔량은 있는데 거래소 접수 시각이 비어 있음. 이 상태에서는 취소가 5039 로 거부됨."""
            x = self._row(no)
            return bool(x) and self.open_qty(no) > 0 and not str(x.get("occurs1_exch_rcv_tm", "")).strip().strip("0")

        def fill_time(self, no):
            """거래소 체결 시각(occurs1_exch_exec_tm, HHMMSSmmm). 체결 전이면 None."""
            x = self._row(no)
            s = str((x or {}).get("occurs1_exch_exec_tm", "")).strip()
            if not s.strip("0") or not s.isdigit():
                return None
            if len(s) not in (8, 9):                         # 명세는 HHMMSSmmm. 앞자리 0 이 빠지면 8자리 — 그 밖의 길이는 믿지 않고 호가 시각을 씀
                return None
            s = s.zfill(9)
            return dt.time(int(s[:2]), int(s[2:4]), int(s[4:6]))

        def cancel(self, no, sym, side):
            x = self._row(no)
            if x is None:                                # 원본은 여기서 조용히 넘어가 취소가 안 나갔음
                raise RuntimeError("주문내역에 아직 없는 주문 — 취소를 내지 못함")
            rem = nh._n(x.get("occurs1_rem_ord_qty"))
            if rem == rem and rem > 0:
                self.F.cancel(sym, org_ord_no=no, trd_div=nh.SIDE[side], ord_qty=str(int(rem)), environment=nh.ORDER_ENV)
                self._hist = None                        # 다음 조회는 취소가 반영된 내역으로

    b = PagedBroker(F)
    b._sent_at, b._hist_full = {}, False
    return b


def watch_leftovers(eng, broker, F, Q, a, log, today) -> dict:
    """실행이 끝난 뒤에도 서버에 살아 있는 내 주문(모의 서버 「미접수」 등)이 있으면 끝내지 않고 지켜봄.

    poll 마다: 새로 드러난 체결 반영 → 살아 있는 주문 취소 재시도 → 모의계좌에 포지션이 있으면 바로 정리
    (앞선 정리 주문이 아직 살아 있으면 겹쳐 내지 않음). 살아 있는 주문도 포지션도 없으면 끝. `--watch-until` 이 한도.
    """
    end = hms(a.watch_until)
    mine = lambda: [(no, KRX300, side, "지정가") for no, side, _ in eng.limits] + [(t[0], t[1], t[2], "정리") for t in eng.takes]
    started, rounds, left, alive, fails, mismatch = False, 0, {}, [], 0, 0
    while now_kst().time() < end:
        try:                                              # 조회 한도·타임아웃 한 번으로 감시가 멈추지 않게
            eng._t = now_kst().strftime("%H:%M:%S")
            broker._hist = None
            late = eng._new_fills()
            for side, got, px, no in late:
                ft = eng._fill_time(no, Quote(now_kst(), None, None)).strftime("%H:%M:%S")
                eng.fills.append((KRX300, side, got, px, "설계밖체결"))
                eng.offplan.append((no, side, got, px, ft))
                log("늦은체결", sym=KRX300, side=side, 수량=got, 체결가=px, 주문번호=no, 체결시각=ft, 설계밖="실행 시간 뒤 체결")
                eng._note("실행 시간 뒤 체결 — 설계 밖, 즉시 청산")
            eng._late_takes()
            alive = [x for x in mine() if broker.alive(x[0])]
            pos = broker.net_positions()
            left = {s: pos.get(s, 0) for s in (KRX300, MINI) if pos.get(s, 0)}
            fails = 0
            if not alive and not left:
                if started:
                    log("감시끝", 회수=rounds, 이유="살아 있는 주문·포지션 없음")
                break
            if not started:
                started = True
                log("감시시작", 살아있는주문=[(no, side, kind, "미접수" if broker.unacked(no) else "") for no, _, side, kind in alive],
                    모의계좌_포지션=left, 한도=a.watch_until)
            for no, sym, side, _ in alive:
                try:
                    broker.cancel(no, sym, side)
                except Exception as e:
                    if rounds % 30 == 0:                  # 기록이 너무 길어지지 않게 5분에 한 번만 적음
                        log("취소거부", 주문번호=no, side=side, 미접수=broker.unacked(no), 오류=str(e)[:120])
            mine_net = {s: v for s, v in eng.net().items() if v}
            if left != mine_net and mismatch == 0:        # 잔고 반영이 늦을 수 있음 — 한 번은 기다려 보고, 계속 다르면 잔고 기준
                mismatch = 1
                log("잔고불일치", 모의계좌_포지션=left, 체결기록_순포지션=mine_net, 처리="한 회차 기다림")
            elif left:
                mismatch = 0
                sym = KRX300 if left.get(KRX300) else MINI
                n = left[sym]
                reduce = "sell" if n > 0 else "buy"
                if not any(s == sym and sd == reduce and k == "정리" for _, s, sd, k in alive):   # 줄이는 쪽 정리 주문이 살아 있을 때만 기다림
                    try:
                        q = quote_from_snap(take_snap(F, Q, a.quote_env))
                    except Exception as e:
                        log("시세실패", 오류=str(e)[:200]); q = None
                    eng._take(sym, reduce, abs(n), "설계밖청산", q)
            else:
                mismatch = 0
        except Exception as e:
            fails += 1
            log("감시오류", 연속=fails, 오류=str(e)[:200])
            if fails % 30 == 0:
                log("경고", 내용=f"감시 조회가 {fails}번 연속 실패 — HTS 에서 확인 필요")
        rounds += 1
        time.sleep(a.watch_poll)
    else:
        if started:
            log("경고", 내용=f"감시 한도 {a.watch_until} 까지 정리되지 않음 — 살아 있는 주문 {[x[0] for x in alive]}, 포지션 {left}. HTS 에서 확인 필요")
    if started:
        try:
            pos = broker.net_positions()
            left = {s: pos.get(s, 0) for s in (KRX300, MINI) if pos.get(s, 0)}
            log("잔고대조", 모의계좌_포지션={s: pos.get(s, 0) for s in (KRX300, MINI)}, 시점="감시 뒤")
        except Exception as e:
            log("감시오류", 오류=str(e)[:200], 시점="감시 뒤 잔고대조")
    return left if started else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--nh-pair-dir", default=str(Path(__file__).resolve().parent.parent / "week04"), help="nh_pair.py 가 있는 폴더")
    ap.add_argument("--orders", action="store_true", help="모의계좌에 실제 주문을 냄. 없으면 주문 없이 체결을 가정")
    ap.add_argument("--quote-env", default="real", choices=("real", "paper"))
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--requote", type=int, default=6)
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--ratio", type=float)
    ap.add_argument("--start", default="08:45:00")
    ap.add_argument("--stop", default="08:59:30")
    ap.add_argument("--exit", default="09:15:00")
    ap.add_argument("--watch-until", default="15:44:00", help="끝난 뒤 살아 있는 주문·포지션을 지켜보는 한도 (정규장 15:45 마감 전)")
    ap.add_argument("--watch-poll", type=float, default=10.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()

    today = now_kst().date()
    out = Path(a.out) / today.strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    tag = "live" if a.orders else "dry"
    ev = open(out / f"{tag}_events.jsonl", "a", encoding="utf-8")
    bk = open(out / f"{tag}_book.jsonl", "a", encoding="utf-8")

    def log(kind, **kw):
        row = {"사건": kind, "at": now_kst().strftime("%H:%M:%S.%f")[:-3], **kw}
        ev.write(json.dumps(row, ensure_ascii=False, default=str) + "\n"); ev.flush()
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)

    sys.path.insert(0, a.nh_pair_dir)
    import nh_pair as nh
    F = nh.load_app(a.app_dir)
    from backend.domestic_futureoption import domestic_quotes as Q

    log("시작", 주문=("모의계좌" if a.orders else "없음(예행연습)"), 시세환경=a.quote_env, d=a.d, 다시거는기준=a.requote, 주기=a.poll,
        시작=a.start, 철수=a.stop, 청산=a.exit)
    try:
        if a.ratio:
            ratio, rday, n = a.ratio, "직접 지정", 0
        else:
            ratio, rday, n = prev_ratio(F, a.quote_env, today, log)
        log("비율", 값=round(ratio, 6), 기준일=rday, 공통분=n)
    except Exception as e:
        log("중단", 이유="비율 계산 실패", 오류=str(e)[:300])
        return 2
    p = Params(ratio=ratio, d_ticks=a.d, requote_ticks=a.requote, start=hms(a.start), stop=hms(a.stop), exit=hms(a.exit))

    if a.orders:
        assert nh.ORDER_ENV == "paper"
        broker = paged_broker(nh, F)
        try:
            pos = broker.net_positions()
            mine = {s: pos.get(s, 0) for s in (KRX300, MINI)}
            first = take_snap(F, Q, a.quote_env)
            kb = max((x[0] for x in first[K200]["b"]), default=None)
            ka = min((x[0] for x in first[K200]["a"]), default=None)
            ref = ratio * (kb + ka) / 2 if kb and ka else (first[KRX300].get("last") or 0)
            mref = (first[K200].get("last") or 0)
            can = {"KRX300 매수": broker.max_qty(KRX300, "buy", ref), "KRX300 매도": broker.max_qty(KRX300, "sell", ref),
                   "미니 매수": broker.max_qty(MINI, "buy", mref), "미니 매도": broker.max_qty(MINI, "sell", mref)}
            log("사전점검", 모의계좌_포지션=mine, 주문가능수량=can)
            if any(mine.values()):
                log("중단", 이유="모의계좌에 KRX300·미니 포지션이 이미 있음 — 주문 없이 종료")
                return 3
            if min(can["KRX300 매수"], can["KRX300 매도"]) < p.qty or min(can["미니 매수"], can["미니 매도"]) < p.hedge_qty:
                log("중단", 이유="주문가능수량 부족 — 주문 없이 종료")
                return 3
        except Exception as e:
            log("중단", 이유="사전점검 실패 — 주문 없이 종료", 오류=str(e)[:300])
            return 3
    else:
        broker = ReplayBroker()

    eng = OpenLimit(p, broker, log=lambda kind, **kw: log(kind, **kw))
    hard_end = (dt.datetime.combine(today, p.exit) + dt.timedelta(minutes=5)).time()
    fails = 0
    while eng.state != "끝" and now_kst().time() < hard_end:
        t0 = time.time()
        try:
            snap = take_snap(F, Q, a.quote_env)
            bk.write(json.dumps(snap, ensure_ascii=False) + "\n"); bk.flush()
            if not a.orders:
                broker.advance(snap)
            q = quote_from_snap(snap)
            fails = 0
        except Exception as e:                       # 시세를 못 받으면 적정가를 모르는 것 — 걸어 둔 주문을 빼게 함
            fails += 1
            log("시세실패", 연속=fails, 오류=str(e)[:200])
            q = Quote(now_kst(), None, None)
        try:
            eng.step(q)
        except Exception as e:
            log("판단오류", 오류=str(e)[:300], 상태=eng.state)
        wait = a.poll
        if eng.state == "보유":                      # 들고 기다리는 동안은 조회를 줄임 (조회 한도 초과 EGW020 방지). 청산 30초 전부터 다시 촘촘히
            to_exit = (dt.datetime.combine(today, p.exit) - now_kst()).total_seconds()
            wait = a.poll if to_exit < 30 else min(10.0, max(a.poll, to_exit - 30))
        time.sleep(max(0.2, wait - (time.time() - t0)))
    if eng.state == "청산중":
        eng._note("청산 미완료 — 실행 시간 끝")

    # ── 마무리: 남은 지정가 취소 → 모의계좌 잔고 대조. 남은 주문·포지션 정리는 아래 감시 단계가 맡음
    #    (감시는 줄이는 쪽 정리 주문이 살아 있으면 겹쳐 내지 않음 — 여기서 따로 정리 주문을 내면 그 확인을 건너뜀) ──
    eng._t = now_kst().strftime("%H:%M:%S")
    eng._cancel_all("마무리")
    left = {}
    if a.orders:
        time.sleep(2)
        try:
            for no, side, price in eng.limits:       # 취소가 안 먹은 주문이 남아 있으면 한 번 더
                try:
                    if broker.alive(no):
                        broker.cancel(no, KRX300, side)
                except Exception:
                    pass
            pos = broker.net_positions()
            left = {s: pos.get(s, 0) for s in (KRX300, MINI) if pos.get(s, 0)}
            log("잔고대조", 모의계좌_포지션={s: pos.get(s, 0) for s in (KRX300, MINI)}, 시점="실행 끝")
        except Exception as e:
            log("마무리오류", 오류=str(e)[:300])

    def write_summary(left, watched=None):
        res = eng.pnl()
        summary = {"날짜": str(today), "주문": ("모의계좌" if a.orders else "없음(예행연습, 체결은 가정)"), "비율": ratio, "비율_기준일": rday, "d": a.d,
                   "다시거는기준": a.requote, "지정가_횟수": len(eng.limits), "상태": eng.state, "체결": eng.fills, "진입": eng.entry, "헤지": eng.hedge,
                   "청산": eng.exits, "설계밖": eng.offplan, "결과": res, "모의계좌_남은포지션": left, "감시": watched}
        (out / f"{tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return res

    res = write_summary(left)
    if a.orders:                                     # 서버에 아직 살아 있는 내 주문이 있으면 끝내지 않고 지켜봄 (2026-11-26: 09:16 에 체결된 미접수 주문)
        try:
            w = watch_leftovers(eng, broker, F, Q, a, log, today)
            if w is not None:
                left = w
                res = write_summary(left, watched=f"감시함 (~{now_kst().strftime('%H:%M:%S')})")
        except Exception as e:
            log("감시오류", 오류=str(e)[:300])
    log("끝", 결과=res, 지정가_횟수=len(eng.limits), 모의계좌_남은포지션=left)
    return 0


if __name__ == "__main__":
    sys.exit(main())
