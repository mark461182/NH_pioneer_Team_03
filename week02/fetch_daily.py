"""코스피200선물 연결 일봉(주간)을 받을 수 있는 만큼 과거부터 받아 CSV 로 저장한다.

- 국내_차트(일)은 한 번에 최대 3000건이고, nxt_key 로는 같은 구간이 다시 와서 조회 종료일을 당겨 가며 받는다.
  조회 종료일 당일은 답에 들어오지 않는다 (end=20140731 → 20140730 까지).
- 원시 연결(fut_cidx_exp_mod=0)은 롤오버 날(분기 만기 다음 거래일) 월물이 바뀌며 가격이 튄다.
  `diff_rt` 는 「같은 월물」의 전일 대비라서, 롤오버 날만 「원시 종가 차이」와 어긋난다 (2014~2026 49번, 전부 분기 만기 다음날).
  → 보유 월물 기준 수정가 = diff_rt 누적. 수정가 = 원시가 + adj_offset (마지막 날 offset 0).
- 만기수정(=1) 연결은 쓰지 않는다: 2014~2026 에 차이가 4번만 바뀌어 분기 롤오버를 반영하지 못한다.

실행: python fetch_daily.py   (week02 폴더에서) [조회 종료일 YYYYMMDD, 그날 제외]   (기본 20261027)
"""
import csv
import sys
import time
from pathlib import Path

from nh_client import NHClient

SYMBOL = "A016C000"  # 연결선물 조회의 기준 종목 (코스피200 F)
RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "k200_1d_continuous.csv"


def window(client: NHClient, end: str) -> dict[str, dict]:
    body = client.post("/trade/v1/domestic/chart-daily", {
        "sym": SYMBOL, "inq_strt_dt": "19900101", "inq_ed_dt": end, "nght_tp": "1",
        "cidx_yn": "1", "fut_cidx_exp_mod": "0", "nxt_key": "99999999", "req_qty": "3000",
    })
    return {r["occurs1_ref_dt"]: r for r in body.get("Occurs1") or []
            if float(r.get("occurs1_close_pric") or 0) > 0}


def main() -> None:
    end = sys.argv[1] if len(sys.argv) > 1 else "20261027"
    client = NHClient("real")  # 시세 조회만 (모의 서버에는 긴 과거 일봉이 없을 수 있어 실전 시세 키 사용)
    raw: dict[str, dict] = {}
    while True:
        got = window(client, end)
        new = {d: r for d, r in got.items() if d not in raw}
        if not new:
            break
        raw.update(new)
        print(f"{min(new)} ~ {max(new)} {len(new)}일")
        if len(got) < 3000:
            break
        end = min(new)
        time.sleep(0.3)

    dates = sorted(raw)
    close = {d: float(raw[d]["occurs1_close_pric"]) for d in dates}
    # 뒤에서부터: offset(t-1) = offset(t) + (원시 종가 차이 − 같은 월물 차이)
    offset = {dates[-1]: 0.0}
    roll = {dates[0]: 0}
    for prev, d in zip(reversed(dates[:-1]), reversed(dates[1:])):
        gap = round((close[d] - close[prev]) - float(raw[d]["occurs1_diff_rt"]), 2)
        roll[d] = int(abs(gap) > 1e-9)
        offset[prev] = round(offset[d] + gap, 2)

    RESULTS.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume", "opint", "diff", "adj_offset", "roll"])
        for d in dates:
            r = raw[d]
            w.writerow([d, r["occurs1_open_pric"], r["occurs1_high_pric"], r["occurs1_low_pric"],
                        r["occurs1_close_pric"], r["occurs1_cum_trd_qty"], r["occurs1_opint"],
                        r["occurs1_diff_rt"], f"{offset[d]:.2f}", roll[d]])
    print(f"{len(dates)}일 ({dates[0]} ~ {dates[-1]}), 롤오버 {sum(roll.values())}번 → {OUT}")


if __name__ == "__main__":
    main()
