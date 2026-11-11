"""results/pair_1m_older.csv + results/pair_1m.csv → results/pair_1m_full.csv (겹치는 분은 최근 파일 우선)"""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent / "results"
a = pd.read_csv(OUT / "pair_1m_older.csv", dtype={"date": str, "time": str})
b = pd.read_csv(OUT / "pair_1m.csv", dtype={"date": str, "time": str})
f = pd.concat([a, b]).drop_duplicates(["date", "time"], keep="last").sort_values(["date", "time"])
f.to_csv(OUT / "pair_1m_full.csv", index=False)
print(f"{f.date.nunique()}일, {len(f):,}분 → results/pair_1m_full.csv")
