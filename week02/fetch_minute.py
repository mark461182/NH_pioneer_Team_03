"""국내 선물 근월물 연결 1분봉(주간)을 하루씩 받아 CSV 로 저장한다. 기본은 코스피200 선물.

- 만기 지난 월물은 분봉 조회가 빈 답이라, 연결선물(`cidx_yn=1`)로 받는다.
  근월물이 바뀌는 날 가격이 이어 붙는다 (check_frequency.py 는 당일 청산이라 날을 넘는 보유가 없다).
- 이어받기 키(`nxt_key`)로 넘기면 같은 줄이 다시 와서, 하루씩 날짜를 바꿔 부른다.
- 빈 답이 연속 5영업일 나오면 더 과거는 없다고 본다 (휴장일도 빈 답이라 한 번에 멈추지 않음).

실행: python fetch_minute.py [마지막날 YYYYMMDD] [종목코드] [저장이름]   (week02 폴더에서, 기본: 어제, 코스피200, k200)
"""
import csv
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from nh_client import NHClient

SYMBOL = "A016C000"  # 연결선물 조회의 기준 종목 (코스피200 F 202612)
RESULTS = Path(__file__).resolve().parent / "results"
MAX_EMPTY_STREAK = 5
MAX_DAYS = 400


def day_bars(client: NHClient, day: str, sym: str = SYMBOL) -> list[dict]:
    body = client.post("/trade/v1/domestic/chart-minute", {
        "sym": sym, "inq_strt_dt": day, "inq_ed_dt": day,
        "strt_tm": "000000", "ed_tm": "235959", "dcnt": "1",
        "nght_tp": "1", "cidx_yn": "1", "fut_cidx_exp_mod": "0",
        "nxt_key": "99999999", "req_qty": "2000",
    })
    return [r for r in body.get("Occurs1") or [] if r.get("occurs1_ref_dt") == day]


def main() -> None:
    last = (datetime.strptime(sys.argv[1], "%Y%m%d").date() if len(sys.argv) > 1
            else date.today() - timedelta(days=1))
    sym = sys.argv[2] if len(sys.argv) > 2 else SYMBOL
    out = RESULTS / f"{sys.argv[3] if len(sys.argv) > 3 else 'k200'}_1m_continuous.csv"
    client = NHClient("real")  # 시세 조회만. 주문 없음
    rows, empty, d, asked = [], 0, last, 0
    while empty < MAX_EMPTY_STREAK and asked < MAX_DAYS:
        if d.weekday() < 5:
            got = day_bars(client, d.strftime("%Y%m%d"), sym)
            asked += 1
            empty = 0 if got else empty + 1
            rows += got
            time.sleep(0.25)
        d -= timedelta(days=1)

    rows.sort(key=lambda r: (r["occurs1_ref_dt"], r["occurs1_exch_tm"]))
    RESULTS.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "open", "high", "low", "close", "volume", "opint"])
        for r in rows:
            w.writerow([r["occurs1_ref_dt"], r["occurs1_exch_tm"], r["occurs1_open_pric"],
                        r["occurs1_high_pric"], r["occurs1_low_pric"], r["occurs1_close_pric"],
                        r["occurs1_cum_trd_qty"], r["occurs1_opint"]])
    days = sorted({r["occurs1_ref_dt"] for r in rows})
    print(f"조회 {asked}영업일, 봉 있는 날 {len(days)} ({days[0]} ~ {days[-1]}), {len(rows)}봉 → {out}")


if __name__ == "__main__":
    main()
