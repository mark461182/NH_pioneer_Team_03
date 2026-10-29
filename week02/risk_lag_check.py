"""리스크 레이어 실행 시점 점검 — 노출을 하루 늦게 적용하면 결과가 얼마나 바뀌는가.

risk_layer_run.py 는 전날 종가까지의 정보로 정한 노출을 오늘 수익 전체(밤사이 갭 포함)에 곱한다.
실제로는 노출을 다음 날 시가에 바꾸므로, 가장 보수적으로 「하루 늦게」 곱해 본다 (노출 변경 비용은 무시).

실행: python risk_lag_check.py   (week02 폴더에서, risk_layer_run.py 결과를 읽음)
결과: results/risk_lag_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return {"누적": round(eq.iloc[-1] - 1, 4), "변동성": round(r.std() * np.sqrt(252), 4),
            "샤프": round(r.mean() / r.std() * np.sqrt(252), 3),
            "최대낙폭": round((eq / eq.cummax() - 1).min(), 4), "최악하루": round(r.min(), 4)}


def main() -> None:
    d = pd.read_csv(RESULTS / "risk_layer_daily.csv", parse_dates=["dt"]).set_index("dt")
    out = {}
    for part in ("학습", "검증"):
        x = d[d["구간"] == part]
        late = x["최종노출"].shift(1).fillna(1) * x["원안수익"]
        out[part] = {"원안": metrics(x["원안수익"]), "레이어(그날 적용)": metrics(x["레이어후수익"]),
                     "레이어(하루 늦게)": metrics(late)}
    (RESULTS / "risk_lag_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
