"""한 월물(기본: 코스피200 2026년 12월물 A016C000)의 일봉(주간)을 받아 CSV 로 저장한다.

앱 백테스트는 연결선물이 아니라 그 월물 한 종목의 일봉으로 셈한다. compare_app_backtest.py 가
같은 데이터로 맞대 보려고 쓴다. 한 번에 최대 3000건이라 조회 종료일을 당겨 가며 받는다.

실행: python fetch_contract_daily.py [종목코드] [저장이름]   (week02 폴더에서, 기본: A016C000, k200_202612_1d)
"""
import csv
import sys
from pathlib import Path

from nh_client import NHClient

RESULTS = Path(__file__).resolve().parent / "results"
ENDS = ("20261030", "20251231", "20250630")


def main() -> None:
    sym = sys.argv[1] if len(sys.argv) > 1 else "A016C000"
    out = RESULTS / f"{sys.argv[2] if len(sys.argv) > 2 else 'k200_202612_1d'}.csv"
    client = NHClient("real")  # 시세 조회만. 주문 없음
    rows: dict[str, dict] = {}
    for end in ENDS:
        body = client.post("/trade/v1/domestic/chart-daily", {
            "sym": sym, "inq_strt_dt": "20230101", "inq_ed_dt": end, "nght_tp": "1",
            "cidx_yn": "0", "fut_cidx_exp_mod": "0", "nxt_key": "99999999", "req_qty": "3000",
        })
        for r in body.get("Occurs1") or []:
            if float(r.get("occurs1_close_pric") or 0) > 0:
                rows[r["occurs1_ref_dt"]] = r
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for day in sorted(rows):
            r = rows[day]
            w.writerow([day, r["occurs1_open_pric"], r["occurs1_high_pric"], r["occurs1_low_pric"],
                        r["occurs1_close_pric"], r.get("occurs1_cum_trd_qty") or r.get("occurs1_trd_qty") or ""])
    print(f"{len(rows)}일 ({min(rows)} ~ {max(rows)}) → {out}")


if __name__ == "__main__":
    main()
