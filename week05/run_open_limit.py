"""5주차 — 장 시작 지정가 전략을 NH 호가에 붙여 돌리는 실행 스크립트.

    python run_open_limit.py --app-dir <nh-futures-open-api 폴더>             # 예행연습: 실시간 호가, 주문 없음(체결은 가정)
    python run_open_limit.py --app-dir <...> --orders                        # NH 국내 **모의계좌**에 실제 주문

- 시세는 `--quote-env`(기본 real, 조회만), 주문은 4주차 `nh_pair.NHPaperBroker` 로 **모의(paper)만** 냅니다.
- 시작할 때 전 거래일 15:00~15:19 의 1분봉으로 비율을 계산합니다(`--ratio` 로 직접 줄 수도 있음).
- `--orders` 일 때는 시작 전에 모의계좌에 KRX300·미니 포지션이 없는지, 주문가능수량이 되는지 확인하고 아니면 주문 없이 끝냅니다.
- 끝나면 남은 지정가를 다시 취소하고, 모의계좌에 KRX300·미니 포지션이 남아 있으면 시장가로 정리합니다.
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
                for _ in range(15):
                    r = self.F.order_history(next_key=key, environment=nh.ORDER_ENV)
                    got = r.get("Occurs1") or []
                    fresh = [x for x in got if (str(x.get("occurs1_ord_no", "")).strip(), str(x.get("occurs1_org_ord_no", "")).strip()) not in seen]
                    for x in fresh:
                        seen.add((str(x.get("occurs1_ord_no", "")).strip(), str(x.get("occurs1_org_ord_no", "")).strip()))
                    rows += fresh
                    key = str(r.get("nxt_key") or "").strip()
                    if not key or not fresh:
                        break
                self._hist, self._hist_t = rows, time.time()
            return self._hist

    return PagedBroker(F)


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
        time.sleep(max(0.2, a.poll - (time.time() - t0)))

    # ── 마무리: 남은 지정가 취소 → 늦은 체결 확인 → 모의계좌 잔고 대조·정리 ──
    eng._t = now_kst().strftime("%H:%M:%S")
    eng._cancel_all("마무리")
    left = {}
    if a.orders:
        time.sleep(2)
        try:
            for no, side, price in eng.limits:       # 취소가 안 먹은 주문이 남아 있으면 한 번 더
                try:
                    broker.cancel(no, KRX300, side)
                except Exception:
                    pass
            time.sleep(2)
            late = eng._new_fills()
            if late:
                log("마무리_늦은체결", 내역=[(s, g, px, no) for s, g, px, no in late])
                for s, g, px, no in late:
                    eng.fills.append((KRX300, s, g, px, "지정가체결(마무리에서 발견)"))
            pos = broker.net_positions()
            left = {s: pos.get(s, 0) for s in (KRX300, MINI) if pos.get(s, 0)}
            log("잔고대조", 모의계좌_포지션={s: pos.get(s, 0) for s in (KRX300, MINI)})
            for sym, n in left.items():              # 시작할 때 0 이었으므로 남은 것은 이 실행이 만든 포지션
                e = eng._market(sym, "sell" if n > 0 else "buy", abs(n), "마무리정리")
                if e["qty"] < abs(n):
                    log("경고", 내용=f"{sym} {abs(n) - e['qty']}계약이 정리되지 않음 — HTS 에서 확인 필요")
            if left:
                time.sleep(2)
                pos = broker.net_positions()
                left = {s: pos.get(s, 0) for s in (KRX300, MINI) if pos.get(s, 0)}
                log("잔고재대조", 모의계좌_포지션={s: pos.get(s, 0) for s in (KRX300, MINI)})
        except Exception as e:
            log("마무리오류", 오류=str(e)[:300])
    res = eng.pnl()
    summary = {"날짜": str(today), "주문": ("모의계좌" if a.orders else "없음(예행연습, 체결은 가정)"), "비율": ratio, "비율_기준일": rday, "d": a.d,
               "다시거는기준": a.requote, "지정가_횟수": len(eng.limits), "상태": eng.state, "체결": eng.fills, "진입": eng.entry, "헤지": eng.hedge,
               "청산": eng.exits, "결과": res, "모의계좌_남은포지션": left}
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("끝", 결과=res, 지정가_횟수=len(eng.limits), 모의계좌_남은포지션=left)
    return 0


if __name__ == "__main__":
    sys.exit(main())
