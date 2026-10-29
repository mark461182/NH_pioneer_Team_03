"""앱 시뮬레이션 실행 기록을 한 주 단위로 정리한다 — 매주 금 16:30(KST) VM 에서 자동으로 돈다.

NH FUTURES API 앱(백엔드)이 남긴 기록 파일(strategy-trades.jsonl)과 실행 상태(API)를 읽어
live/<ISO 주>/ 에 원본 사본과 요약을 쓰고, live/README.md 의 주별 표를 다시 만든다.

- 원본은 고치지 않는다. 그 주의 줄만 골라 그대로 복사한다(journal.jsonl).
- 요약은 기록에 적힌 값(netPl·fee·체결 사유)을 더하기만 한다. 지어내는 값은 없다.
- 기록의 record 칸(paper·sim·real)을 그대로 옮겨, 앱 시뮬레이션과 계좌 주문을 섞지 않는다.

실행: python3 live/weekly_report.py [--week 2026-W44] [--api http://127.0.0.1:47100]
      [--journal ~/NHFutures/state/strategy-trades.jsonl]   (저장소 루트에서, 표준 라이브러리만 사용)
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
LIVE = Path(__file__).resolve().parent
RECORD_LABEL = {"paper": "앱 시뮬레이션 (종이 체결, 계좌 주문 없음)", "sim": "모의계좌 주문", "real": "실전계좌 주문"}


def week_bounds(label: str) -> tuple[datetime, datetime]:
    """ISO 주 「2026-W44」 → 그 주 월 00:00 ~ 다음 월 00:00 (KST)."""
    year, week = label.split("-W")
    monday = date.fromisocalendar(int(year), int(week), 1)
    start = datetime(monday.year, monday.month, monday.day, tzinfo=KST)
    return start, start + timedelta(days=7)


def this_week() -> str:
    year, week, _ = datetime.now(KST).isocalendar()
    return f"{year}-W{week:02d}"


def line_time(row: dict) -> int | None:
    for key in ("time", "closedAt", "endedAt", "placedAt"):
        if isinstance(row.get(key), (int, float)):
            return int(row[key])
    return None


def kst(ms: int | None) -> str:
    return datetime.fromtimestamp(ms / 1000, KST).strftime("%Y-%m-%d %H:%M:%S") if ms else ""


def read_journal(paths: list[Path], start: datetime, end: datetime) -> list[str]:
    lo, hi = start.timestamp() * 1000, end.timestamp() * 1000
    kept = []
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            when = line_time(row)
            if when is not None and lo <= when < hi:
                kept.append(raw)
    return kept


def fetch(api: str, path: str) -> dict | None:
    try:
        with urllib.request.urlopen(api + path, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 — 앱이 꺼져 있어도 기록 파일 요약은 낸다
        return None


def summarize(rows: list[dict]) -> dict:
    runs: dict[str, dict] = {}
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[row.get("runId", "")].append(row)
    for run_id, items in sorted(by_run.items()):
        trades = [r for r in items if r["kind"] == "trade"]
        fills = [r for r in items if r["kind"] == "fill"]
        events = Counter(r.get("event") for r in items if r["kind"] == "event")
        net = [r.get("netPl") or 0.0 for r in trades]
        runs[run_id] = {
            "전략 버전": items[0].get("versionId", ""),
            "종목": items[0].get("code", ""),
            "세션": items[0].get("session", ""),
            "기록 갈래": sorted({r.get("record", "") for r in items}),
            "신호": events.get("signal", 0),
            "주문": sum(1 for r in items if r["kind"] == "order"),
            "체결": len(fills),
            "체결 사유": dict(Counter(f"{r.get('intent')}·{r.get('reason')}" for r in fills)),
            "거래": len(trades),
            "승": sum(1 for v in net if v > 0),
            "패": sum(1 for v in net if v < 0),
            "순손익(원)": round(sum(net)),
            "수수료(원)": round(sum(r.get("fee") or 0.0 for r in trades)),
            "이벤트": dict(events),
        }
    return runs


def write_week(label: str, api: str, journals: list[Path]) -> dict:
    start, end = week_bounds(label)
    raw = read_journal(journals, start, end)
    rows = [json.loads(r) for r in raw]
    out = LIVE / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "journal.jsonl").write_text("".join(r + "\n" for r in raw), encoding="utf-8")

    with (out / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["runId", "record", "side", "quantity", "openedAt", "closedAt", "avgEntry", "avgExit", "netPl", "fee"])
        for r in (x for x in rows if x["kind"] == "trade"):
            w.writerow([r.get("runId"), r.get("record"), r.get("side"), r.get("quantity"), kst(r.get("openedAt")),
                        kst(r.get("closedAt")), r.get("avgEntry"), r.get("avgExit"), round(r.get("netPl") or 0), round(r.get("fee") or 0)])

    state = fetch(api, "/api/strategy/runs")
    if state is not None:
        (out / "runs.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        for run in state.get("runs") or []:
            snap = fetch(api, f"/api/strategy/runs/{run['id']}")
            if snap is not None:
                (out / f"snapshot_{run['id']}.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "주": label, "구간(KST)": f"{start:%Y-%m-%d} ~ {(end - timedelta(days=1)):%Y-%m-%d}",
        "만든 시각(KST)": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "기록 줄": len(raw), "앱 상태 읽음": state is not None,
        "실행 상태": {r["id"]: r.get("stateLabel") for r in (state or {}).get("runs") or []},
        "실행별": summarize(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def write_index() -> None:
    weeks = sorted((json.loads(p.read_text(encoding="utf-8")) for p in LIVE.glob("*-W*/summary.json")),
                   key=lambda s: s["주"], reverse=True)
    lines = [
        "# 실행 기록 — 매주 자동 정리",
        "",
        "> **이 파일과 아래 폴더는 VM 이 매주 금 16:30(KST) 자동으로 만듭니다.** 사람이 고치지 않습니다.",
        "> 만드는 코드: [weekly_report.py](weekly_report.py). 실행 환경: GCP VM 에 상주하는 NH FUTURES API 앱 백엔드.",
        "> 「기록 갈래」가 `paper` 인 줄은 **앱 시뮬레이션**(실전 시세로 판정, 종이 체결)이며 **계좌 주문이 아닙니다.**",
        "> 손익은 기록에 적힌 `netPl` 을 더한 값이고, 손실도 그대로 적습니다.",
        "",
        "| 주 | 구간 | 실행 | 기록 갈래 | 신호 | 체결 | 거래 (승·패) | 순손익(원) | 상태 | 폴더 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in weeks:
        runs = s["실행별"] or {"(기록 없음)": None}
        for run_id, r in runs.items():
            state = s["실행 상태"].get(run_id, "")
            if r is None:
                lines.append(f"| {s['주']} | {s['구간(KST)']} | 기록 없음 | | | | | | | [{s['주']}]({s['주']}/) |")
                continue
            kinds = ", ".join(RECORD_LABEL.get(k, k) for k in r["기록 갈래"])
            lines.append(f"| {s['주']} | {s['구간(KST)']} | `{run_id}` | {kinds} | {r['신호']} | {r['체결']} | "
                         f"{r['거래']} ({r['승']}·{r['패']}) | {r['순손익(원)']:,} | {state} | [{s['주']}]({s['주']}/) |")
    lines += ["", "폴더마다: `journal.jsonl`(그 주 기록 원본 사본) · `trades.csv` · `summary.json` · `runs.json`·`snapshot_<실행>.json`(정리 시각의 실행 상태와 조건 표)", ""]
    (LIVE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default=this_week())
    ap.add_argument("--api", default="http://127.0.0.1:47100")
    ap.add_argument("--journal", default=str(Path.home() / "NHFutures" / "state" / "strategy-trades.jsonl"))
    a = ap.parse_args()
    journal = Path(a.journal).expanduser()
    summary = write_week(a.week, a.api, [journal.with_suffix(".jsonl.old"), journal])
    write_index()
    print(json.dumps({k: v for k, v in summary.items() if k != "실행별"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
