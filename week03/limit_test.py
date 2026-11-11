"""MHI 지정가 체결 시험 — NH 모의계좌 (11/05 시장가 거부 7032 후속).

순서
  1. 기준가: NH 실전 키 MHIX26 현재가 (시세 열렸으면) → 안 되면 Yahoo ^HSI 1분봉 마지막 값(약 15분 지연)
  2. 매수 지정가 = 기준가 + OFFSET  (바로 체결될 만큼 유리하게)
  3. 30초 기다렸다가 주문내역 조회 → 체결이면 매도 지정가 = 기준가 − OFFSET 로 청산
                                     → 미체결이면 취소
  4. 모든 응답을 limit_test_log.jsonl 에 저장 (README 6-3·8-1 근거)

실행:  python limit_test.py --app-dir C:\\Users\\admin\\Downloads\\nh-futures-open-api
주문은 모의(paper) 서버로만 나갑니다.
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

SYM = "MHIX26"
ENV = "paper"          # 바꾸지 말 것
WAIT = 30              # 초
LOG = Path(__file__).resolve().parent / "limit_test_log.jsonl"
HIDE = ("acnt", "acct", "account", "ac_no", "pwd", "token", "key", "user_id", "ip_addr", "mac")


def clean(x):
    if isinstance(x, dict):
        return {k: ("***" if any(h in k.lower() for h in HIDE) else clean(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, str):          # 오류 문구 안의 키 값도 가림 (설정 파일 파싱 오류가 키 줄을 인용함)
        return re.sub(r"(app_?(?:key|secret)\W{0,6})[A-Za-z0-9+/=_-]{6,}", r"\1***", x, flags=re.I)
    return x


def log(kind, **kw):
    rec = {"t": dt.datetime.now().strftime("%H:%M:%S"), "kind": kind, **clean(kw)}
    print(f"[{rec['t']} KST] {kind:8s} " + " ".join(f"{k}={v}" for k, v in kw.items() if k != "raw"))
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def ref_price(F):
    try:
        r = F.price([("FHKE", SYM)], environment="real")
        rows = r.get("Occurs1") or []
        p = float(rows[0].get("occurs1_last_pric"))
        if p > 0:
            return p, "NH 실전 MHIX26 현재가"
    except Exception as e:
        log("실전시세실패", 오류=str(e)[:150])
    import yfinance as yf
    d = yf.download("^HSI", period="1d", interval="1m", progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return float(d["Close"].dropna().iloc[-1]), "Yahoo ^HSI (약 15분 지연, 지수)"


def send(F, A, side, px, why):
    try:
        r = F.order(SYM, qt_div_cd="1", trd_div_cd=A.SIDE_CODE[side], glb_ord_tp_cd="2",
                    ord_qty="1", ord_pric=str(int(round(px))), stop_pric="", environment=ENV)
        no = str(r.get("ord_no", "")).strip()
        log("주문", side=side, 가격=int(round(px)), 이유=why, 주문번호=no,
            오류=(str(r.get("err_cd", "")).strip() + " " + str(r.get("err_msg", "")).strip()).strip(), raw=r)
        return no
    except Exception as e:
        log("주문거부", side=side, 가격=int(round(px)), 오류=str(e)[:300])
        return ""


def status(F, no):
    try:
        r = F.order_history(inq_div="0", prd_cd="MHI", environment=ENV)
        rows = r.get("Occurs1") or []
        mine = [x for x in rows if str(x.get("occurs1_ord_no", "")).strip() == no]
        row = mine[0] if mine else {}
        s = {k: str(row.get(f"occurs1_{k}", "")).strip() for k in
             ("ord_sts", "ord_qty", "exec_qty", "ord_rem_qty", "ord_pric", "exec_pric", "exec_tm")}
        log("조회", 주문번호=no, 찾음=bool(mine), 상태코드=s["ord_sts"], 체결수량=s["exec_qty"],
            잔량=s["ord_rem_qty"], 체결가=s["exec_pric"], raw=row or rows[-3:])
        return s
    except Exception as e:
        log("조회실패", 오류=str(e)[:200])
        return {}


def cancel(F, A, side, px, no):
    try:
        r = F.order(SYM, qt_div_cd="3", trd_div_cd=A.SIDE_CODE[side], glb_ord_tp_cd="2",
                    ord_qty="1", ord_pric=str(int(round(px))), stop_pric="", org_ord_no=no, environment=ENV)
        log("취소", 원주문=no, 오류=(str(r.get("err_cd", "")).strip() + " " + str(r.get("err_msg", "")).strip()).strip(), raw=r)
    except Exception as e:
        log("취소실패", 원주문=no, 오류=str(e)[:200])


def filled(s):
    try:
        return int(float(s.get("exec_qty") or 0)) >= 1 or s.get("ord_sts") == "6"
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--offset", type=float, default=150, help="기준가 대비 유리하게 둘 폭(pt)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.app_dir).resolve()))
    from backend.overseas_futureoption import overseas_futureoption_functions as F
    from backend.overseas_futureoption import overseas_account as A

    if input(f"모의계좌에 {SYM} 지정가 1계약 매수 시험(미체결이면 취소, 체결이면 지정가 매도). 진행하려면 YES: ").strip() != "YES":
        return
    px, src = ref_price(F)
    log("기준가", 가격=round(px, 1), 출처=src, 여유폭=a.offset)

    buy_px = px + a.offset
    no = send(F, A, "buy", buy_px, "지정가 체결 시험")
    if not no:
        log("끝", 결과="매수 주문이 접수되지 않음 — 위 오류를 8-1절에")
        return
    time.sleep(WAIT)
    s = status(F, no)
    if not filled(s):
        cancel(F, A, "buy", buy_px, no)
        time.sleep(5)
        status(F, no)
        log("끝", 결과=f"지정가 {WAIT}초 미체결 → 취소. 모의 서버가 HKEX 지정가를 체결하지 않는 것으로 보임")
        return

    px2, src2 = ref_price(F)
    sell_px = px2 - a.offset
    no2 = send(F, A, "sell", sell_px, "체결 시험 청산")
    time.sleep(WAIT)
    s2 = status(F, no2) if no2 else {}
    if no2 and not filled(s2):
        cancel(F, A, "sell", sell_px, no2)
        log("끝", 결과="매수는 체결, 매도 미체결 → 매도 취소. **HTS 에서 MHI 1계약 직접 청산 필요**")
        return
    log("끝", 결과=f"매수 체결가 {s.get('exec_pric')} / 매도 체결가 {s2.get('exec_pric')} — 지정가로 실행 가능")


if __name__ == "__main__":
    main()
