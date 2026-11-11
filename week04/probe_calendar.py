"""캘린더 스프레드(같은 종목 월물끼리) 가 되는지 먼저 확인 — 시세 조회만, 주문 없음

  python probe_calendar.py --app-dir <nh-futures-open-api> [--days 60]
  → results/cal_master.csv   삼성전자·SK하이닉스 줄 전부 (선물·스프레드·옵션 구분값 그대로)
    results/cal_<종목코드>.csv 월물별 1분봉 (연결 아님, 그 월물 자체)
    results/cal_probe.txt      월물별 「하루 거래량·거래 있는 분 비율」 요약

알고 싶은 것
  ① 마스터에 월물 스프레드 종목(한 번에 두 월물을 주문하는 종목)이 있나 → fo_tp·opt_tp 값으로 구분
  ② 3월물(원월물) 1분봉이 API 로 받아지나, 거래가 얼마나 되나
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

import nh_pair as nh

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def all_rows(app_dir: str) -> pd.DataFrame:
    nh.load_app(app_dir)
    from backend.master.master_download import download_all
    from backend.master.master_store import load_all
    try:
        download_all()
    except Exception as e:
        print("마스터 내려받기 실패 — 받아 둔 파일 사용:", str(e)[:150])
    rows = []
    for lm in load_all():
        if lm.table is None:
            continue
        for raw in lm.table.rows:
            r = dict(zip(lm.table.fields, raw))
            nm = " ".join(str(r.get("name_kor", "")).split())
            if nm.startswith(nh.NAME_A + " ") or nm.startswith(nh.NAME_B + " ") or "스프레드" in nm and (nh.NAME_A in nm or nh.NAME_B in nm):
                r["_table"] = lm.table.name
                r["name_kor"] = nm
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--days", type=int, default=60, help="최근 몇 영업일")
    ap.add_argument("--quote-env", choices=("real", "paper"), default="real")
    a = ap.parse_args()
    nh.set_quote_env(a.quote_env)
    OUT.mkdir(exist_ok=True)
    F = nh.load_app(a.app_dir)

    M = all_rows(a.app_dir)
    M.to_csv(OUT / "cal_master.csv", index=False, encoding="utf-8-sig")
    lines = [f"마스터 줄 {len(M)}개 (삼성전자·SK하이닉스)"]
    show = [c for c in ("_table", "series", "name_kor", "fo_tp", "opt_tp", "exp_ym", "ctrt_size", "cme_tp") if c in M.columns]
    if len(M):
        lines.append(M[show].drop_duplicates().to_string(index=False))

    # 옵션이 아닌 줄(선물·스프레드)만, 주간만
    cand = M.copy()
    if "fo_tp" in cand:
        cand = cand[cand["fo_tp"] != "O"]
    if "cme_tp" in cand:
        cand = cand[cand["cme_tp"].fillna("day") != "night"]
    if "opt_tp" in cand:
        cand = cand[~cand["opt_tp"].isin(["C", "P", "2", "3"])]   # 콜·풋으로 보이는 값은 뺌 (값 모르면 남김)
    syms = list(dict.fromkeys(cand["series"].astype(str).str.strip())) if len(cand) else []
    lines.append(f"\n1분봉 받을 종목 {len(syms)}개: {syms}")

    days, d = [], dt.date.today() - dt.timedelta(days=1)
    while len(days) < a.days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d -= dt.timedelta(days=1)
    days.reverse()

    summary = []
    for s in syms:
        nm = cand.loc[cand["series"].astype(str).str.strip() == s, "name_kor"].iloc[0]
        frames = []
        for day in days:
            try:
                f = nh.day_bars(F, s, day)
            except Exception as e:
                print(f"  {s} {day} 실패: {str(e)[:100]}")
                continue
            if len(f):
                f = f.copy()
                f["vol1"] = f["volume"].diff().fillna(f["volume"]).clip(lower=0)   # 누적 → 분당
                frames.append(f)
        if not frames:
            summary.append({"종목": s, "이름": nm, "받은날": 0})
            print(f"{s} {nm}: 데이터 없음")
            continue
        X = pd.concat(frames)
        X.to_csv(OUT / f"cal_{s}.csv", encoding="utf-8-sig")
        per_day = X.groupby(X.index.date)
        summary.append({"종목": s, "이름": nm, "받은날": per_day.ngroups,
                        "하루거래량_중앙": int(per_day["vol1"].sum().median()),
                        "거래있는분_비율": round(float((X["vol1"] > 0).mean()), 3),
                        "하루봉수_중앙": int(per_day.size().median()),
                        "첫날": str(X.index[0].date()), "끝날": str(X.index[-1].date())})
        print(summary[-1])
    S = pd.DataFrame(summary)
    lines.append("\n" + S.to_string(index=False))
    (OUT / "cal_probe.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-1:]))
    print("\n끝 — results/cal_probe.txt, cal_master.csv, cal_*.csv 를 Claude 에게 주세요")


if __name__ == "__main__":
    sys.exit(main())
