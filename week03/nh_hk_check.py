"""NH선물 해외 실전 키로 항셍(HSI)·미니항셍(MHI) 시세가 오는지 확인하고 1분봉을 CSV로 저장.

**읽기 전용** — 주문 함수는 불러오는 순간 막아 둠. 계좌·주문 API 호출 없음.

준비
  1) NH선물 Open API 앱 저장소를 받는다
       git clone https://github.com/PetterChanHyuk/nh-futures-open-api
       cd nh-futures-open-api  (그 README 대로 가상환경·패키지 설치)
  2) ~/NHFutures/config/config.yaml (Windows: %USERPROFILE%\\NHFutures\\config\\config.yaml)
     의 real.overseas.app_key / app_secret 에 **해외 실전** 키 입력
  3) 이 파일을 앱 저장소 폴더 안에 두고 실행
       python nh_hk_check.py            # 시세 확인 + 최근 5일 1분봉
       python nh_hk_check.py --days 10  # 기간 늘리기

결과
  out_hk/check.txt          어떤 호출이 성공/실패했는지 (키 값은 안 찍음)
  out_hk/<종목>_1m.csv      1분봉 원문 (시각 = 응답 그대로, KST)
"""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from backend.overseas_futureoption import overseas_futureoption_functions as F

# 주문 계열은 실수로라도 못 부르게 막는다
for name in dir(F):
    if any(w in name.lower() for w in ("order", "cancel", "modify", "amend")):
        setattr(F, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("이 스크립트는 주문 금지")))

ENV = "real"
SYMS = [("FHKE", "HSIX26"), ("FHKE", "MHIX26")]   # 11월물. 12월로 넘어가면 Z26
OUT = Path("out_hk")
LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def try_call(label, fn):
    try:
        r = fn()
        log(f"[성공] {label}")
        return r
    except Exception as e:  # 오류 문구에 EGW016(권한 없음)·ETR029(종목 없음) 등이 들어 있음
        log(f"[실패] {label} → {str(e)[:200]}")
        return None


def minutes(ex, sym, days):
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    rows, key, pages = [], getattr(F, "FIRST_PAGE", ""), 0
    while pages < 40:
        r = F.chart_minute(sym, exch_cd=ex, inq_strt_dt=start.strftime("%Y%m%d"),
                           inq_ed_dt=end.strftime("%Y%m%d"), next_key=key, environment=ENV)
        got = r.get("Occurs1") or []
        lo = start.strftime("%Y%m%d")
        keep = [x for x in got if str(x.get("occurs1_ref_dt", "")) >= lo]
        rows += keep
        pages += 1
        key = str(r.get("nxt_key") or "").strip()
        if not got or not key or len(keep) < len(got):   # 최신→과거 순이라 시작일 넘으면 끝
            break
    seen, out = set(), []
    for x in rows:
        k = (x.get("occurs1_ref_dt"), x.get("occurs1_exch_tm"))
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--env", default="real", help="real(기본) / paper")
    ap.add_argument("--sym", action="append", help="거래소:종목, 예) FHKE:HSIZ26 (여러 번 가능)")
    a = ap.parse_args()
    global ENV, SYMS
    ENV = a.env
    if a.sym:
        SYMS = [tuple(x.split(":", 1)) for x in a.sym]
    OUT.mkdir(exist_ok=True)
    log(f"실행 {dt.datetime.now():%Y-%m-%d %H:%M:%S} · 환경 {ENV} · 기간 {a.days}일")
    for ex, sym in SYMS:
        r = try_call(f"{sym} 현재가", lambda: F.price([(ex, sym)], environment=ENV))
        if r:
            o = (r.get("Occurs1") or [{}])[0]
            if not str(o.get("occurs1_last_pric", "")).strip():
                log("    ↳ 응답은 왔지만 현재가가 비어 있음 = 종목이 없거나 이 서버에 시세 없음")
            log(f"    현재가 {str(o.get('occurs1_last_pric', '')).strip()} · 영업일 {o.get('occurs1_biz_dt')}")
        r = try_call(f"{sym} 호가", lambda: F.bid_price(sym, exch_cd=ex, environment=ENV))
        if r:
            log(f"    매도1 {str(r.get('ask_pric_1', '')).strip()} / 매수1 {str(r.get('bid_pric_1', '')).strip()}")
        rows = try_call(f"{sym} 1분봉", lambda: minutes(ex, sym, a.days))
        if rows:
            p = OUT / f"{sym}_1m.csv"
            cols = sorted({k for r in rows for k in r})
            with p.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            log(f"    {len(rows)}줄 저장 → {p}")
    (OUT / "check.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\n요약 → {OUT / 'check.txt'}  (이 폴더만 보내주면 됨, config.yaml·tokens.json 은 절대 X)")


if __name__ == "__main__":
    sys.exit(main())
