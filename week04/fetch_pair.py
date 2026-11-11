"""삼성전자·SK하이닉스 주식선물 연결 1분봉(주간)을 하루씩 받아 CSV 로 저장 — 백테스트 입력.

  python fetch_pair.py --app-dir <nh-futures-open-api 폴더> [--days 260] [--last YYYYMMDD]

- 실전 키로 시세만 조회합니다 (주문 없음).
- 만기 지난 월물은 분봉이 빈 답이라 연결선물(cidx_yn=1)로 받습니다 — 2주차 fetch_minute.py 와 같은 방식.
  두 종목은 만기일이 같아(매월 둘째 목요일) 같은 날 월물이 바뀝니다.
- 빈 답이 5영업일 연속이면 더 과거는 없다고 봅니다 (휴장일도 빈 답).
출력: results/pair_1m.csv  (date, time, a_open..a_close, b_open..b_close — 두 종목 다 있는 분만)
"""
import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

import nh_pair as nh

OUT = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--days", type=int, default=260, help="받을 영업일 수 (최대)")
    ap.add_argument("--last", default=None, help="마지막 날 YYYYMMDD (기본: 어제)")
    ap.add_argument("--sym-a"); ap.add_argument("--sym-b")
    ap.add_argument("--quote-env", choices=("real", "paper"), default="real",
                    help="시세를 받을 키 (기본 실전). 실전 국내 키가 없고 모의 키에 시세 권한이 있으면 paper")
    ap.add_argument("--out", default="pair_1m.csv", help="results/ 안 저장 파일 이름")
    ap.add_argument("--max-empty", type=int, default=5, help="빈 답이 이만큼 연속이면 더 과거는 없다고 봄")
    ap.add_argument("--no-continuous", action="store_true",
                    help="연결선물(cidx_yn=1)이 빈 답이면 이 월물 그대로 받기 (그 월물 상장 뒤 기간만)")
    a = ap.parse_args()

    F = nh.load_app(a.app_dir)
    nh.set_quote_env(a.quote_env)
    today = dt.date.today()
    if a.sym_a and a.sym_b:
        A, B = {"sym": a.sym_a, "name": nh.NAME_A}, {"sym": a.sym_b, "name": nh.NAME_B}
    else:
        rows = nh.master_rows(a.app_dir)
        A, B = nh.pick_contract(rows, nh.NAME_A, today), nh.pick_contract(rows, nh.NAME_B, today)
    print(f"A {A['name']} {A['sym']} / B {B['name']} {B['sym']}")

    last = dt.datetime.strptime(a.last, "%Y%m%d").date() if a.last else today - dt.timedelta(days=1)
    cont = not a.no_continuous
    frames, d, asked, empty = [], last, 0, 0
    out_path = OUT / a.out
    if out_path.exists():                              # 이어받기: 저장된 가장 이른 날 앞부터
        prev = pd.read_csv(out_path, dtype={"date": str, "time": str})
        prev.index = pd.to_datetime(prev["date"] + prev["time"].str.zfill(6), format="%Y%m%d%H%M%S")
        frames.append(prev.drop(columns=["date", "time"]))
        d = dt.datetime.strptime(prev["date"].min(), "%Y%m%d").date() - dt.timedelta(days=1)
        print(f"이어받기: {a.out} 에 {prev['date'].nunique()}일 있음 → {d} 부터 거꾸로")

    def save():
        if not frames:
            return
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated()]
        df.insert(0, "date", df.index.strftime("%Y%m%d"))
        df.insert(1, "time", df.index.strftime("%H%M%S"))
        OUT.mkdir(exist_ok=True)
        df.to_csv(out_path, index=False)
        return df

    import atexit
    atexit.register(save)                              # 어떤 이유로 끝나도 받은 만큼은 저장
    while asked < a.days and empty < a.max_empty:
        if d.weekday() < 5:
            day = d.strftime("%Y%m%d")
            fa = nh.day_bars(F, A["sym"], day, continuous=cont)
            fb = nh.day_bars(F, B["sym"], day, continuous=cont)
            if cont and asked < 3 and not (len(fa) and len(fb)):      # 처음 며칠이 비면 연결선물 미지원으로 보고 전환
                fa2, fb2 = nh.day_bars(F, A["sym"], day), nh.day_bars(F, B["sym"], day)
                if len(fa2) and len(fb2):
                    print(f"  {day} 연결선물 빈 답 → 월물 그대로 받기로 전환 (그 월물 상장 뒤 기간만 나옴)")
                    cont, fa, fb = False, fa2, fb2
            asked += 1
            if len(fa) and len(fb):
                both = fa.add_prefix("a_").join(fb.add_prefix("b_"), how="inner")
                frames.append(both)
                empty = 0
            else:
                empty += 1
            if asked % 20 == 0:
                print(f"  {day} 까지 {asked}영업일, 봉 있는 날 {len(frames)}", flush=True)
                save()                                  # 중간 저장 — 끊겨도 이어받기
            time.sleep(0.3)
        d -= dt.timedelta(days=1)

    if not frames:
        raise SystemExit("받은 봉이 없음 — 실전 키 [국내 시세조회] 권한·종목코드(--sym-a/--sym-b)·날짜(--last)를 확인")
    df = save()
    days = df["date"].unique()
    print(f"조회 {asked}영업일 · 봉 있는 날 {len(days)} ({days[0]} ~ {days[-1]}) · {len(df)}분 → results/{a.out}"
          + (f"  (빈 답 {empty}일 연속으로 멈춤 — 그 앞은 데이터 없음)" if empty >= a.max_empty else ""))


if __name__ == "__main__":
    main()
