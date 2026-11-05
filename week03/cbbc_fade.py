"""CBBC 벽 돌파 되돌리기 — 장중 실행기 (3주차, Team-03)

규칙 (백테스트와 같은 정의)
  1. 지도: 장 시작 전 HKEX 공개 CBBC 전체 목록(HSI 기초)에서 50pt 구간별 물량을 계산
     물량 = 발행수량 × 미결제(%) ÷ 교환비율 ÷ 50  (HSI 선물 계약 상당)
  2. 이벤트: 장중(09:30~16:00 HKT) HSI 지수 1분봉이 09:30 시가 너머 50pt 경계를 그날 처음 뚫은 분
       위로 경계 e 를 뚫음 → 구간 [e, e+50) 의 베어 물량을 봄
       아래로 경계 e 를 뚫음 → 구간 [e-50, e) 의 불 물량을 봄
  3. 신호: 그 구간 물량 ≥ 300 이면 **돌파 반대 방향** 으로 MHI 1계약 (위로 뚫으면 매도, 아래로 뚫으면 매수)
  4. 청산: 이벤트 분 + 30분. 한 번에 한 포지션. 이벤트+30분이 16:00 을 넘으면 진입 안 함
  5. 주문: NH 는 HKEX 시장가를 받지 않음 (7032, 11/05 확인) → **유리한 지정가** 로 냄
       매수 = 기준가 + 150pt, 매도 = 기준가 − 150pt. 모의 서버는 지정가가 아니라 그 시점 가격으로 체결함
       (11/05 시험: 매수 지정가 24,942 → 24,783 체결, 매도 지정가 24,635 → 24,791 체결)

실행 모드
  python cbbc_fade.py                     # 기본: 신호·가상 체결만 기록 (키 불필요, 주문 없음)
  python cbbc_fade.py --paper-order --app-dir <nh-futures-open-api 폴더>
                                          # 신호마다 **NH 모의계좌** MHI 지정가(±150pt) 주문 (실전 주문은 코드상 불가)
  python cbbc_fade.py --test-order --app-dir <...>
                                          # 모의계좌 MHI 1계약 매수 → 60초 뒤 매도, 체결되는지만 확인
  python cbbc_fade.py --price nh --app-dir <...>
                                          # 가격을 NH 실전 키 HSI 선물 1분봉으로 (지연 없음, [해외 시세] 신청이 API 에 반영돼야 함)

가격 출처
  yf (기본): Yahoo ^HSI 1분봉 — 무료지만 **약 15분 지연**. 지연만큼 늦게 진입하고 청산 시각(이벤트+30분)은 그대로 →
             실제 보유는 약 15분. 백테스트에서 이 지연판은 따로 계산해 둠 (README 4절).
  nh: NH 실전 해외 키로 HSIX26 1분봉. 지수와 선물의 차이(베이시스)는 지연된 Yahoo 지수와 겹치는 분으로 추정해 뺌.

기록: results/<날짜>/ 에 map.csv(그날 지도), fulllist_raw.csv(HKEX 원본), log.jsonl(모든 일), trades.csv(거래 요약)
"""
import argparse
import datetime as dt
import io
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HKT = dt.timezone(dt.timedelta(hours=8))
FULL = "https://www.hkex.com.hk/eng/cbbc/search/cbbcFullList.csv"
UA = {"User-Agent": "Mozilla/5.0 (team03-cbbc-study)"}
B = 50            # 구간 폭 pt
WALL = 300        # 벽 기준 물량 (HSI 선물 계약 상당)
HOLD = 30         # 이벤트 후 청산까지 분
MAX_LAG = 25      # 이벤트를 이만큼(분) 늦게 알았으면 진입 안 함
COST = 3          # 왕복 비용 가정 pt (스프레드 1 + 슬리피지 1 + 수수료 1)
SYM_ORDER = "MHIX26"   # 12월물로 넘어가면 MHIZ26
SYM_PRICE = "HSIX26"
ORDER_ENV = "paper"    # 주문은 모의 서버로만. 바꾸지 말 것
POLL = 30              # 초
LIMIT_OFFSET = 150     # 지정가 여유폭 pt — 기준가가 Yahoo 15분 지연이어도 바로 체결되도록

OUT = Path(__file__).resolve().parent / "results" / dt.datetime.now(HKT).strftime("%Y%m%d")


# ───────── 기록 ─────────
SECRET = ("acnt", "acct", "account", "ac_no", "pwd", "passwd", "token", "key", "user_id", "ip_addr", "mac")


def clean(x):
    """응답 원문에서 계좌번호·비번·토큰 같은 칸을 지움 (결과는 공개 저장소에 올라감)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not any(w in k.lower() for w in SECRET)}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def log(kind, **kw):
    kw = clean(kw)
    rec = {"t": dt.datetime.now(HKT).isoformat(timespec="seconds"), "kind": kind, **kw}
    with (OUT / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    msg = " ".join(f"{k}={v}" for k, v in kw.items() if k not in ("raw",))
    print(f"[{rec['t'][11:19]} HKT] {kind:8s} {msg}")


# ───────── 지도 ─────────
def load_map() -> dict:
    raw = urllib.request.urlopen(urllib.request.Request(FULL, headers=UA), timeout=60).read().decode("utf-16")
    (OUT / "fulllist_raw.csv").write_text(raw, encoding="utf-8")
    lines = raw.splitlines()
    stamp = lines[0].strip()
    h = [i for i, l in enumerate(lines) if l.startswith("CBBC Code")][0]
    df = pd.read_csv(io.StringIO("\n".join(lines[h:])), sep="\t", dtype=str, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    df = df[df["UL"] == "HSI"]
    num = lambda s: pd.to_numeric(s.str.replace(",", ""), errors="coerce")
    L = pd.DataFrame({"code": df["CBBC Code"], "side": df["Bull/Bear"].str.lower(), "call": num(df["Call Level"]),
                      "ratio": num(df["Entitlement Ratio^"]), "issue": num(df["Total Issue Size"]),
                      "os": num(df["O/S (%)"])}).dropna()
    L["qty"] = L["issue"] * L["os"] / 100 / L["ratio"] / 50
    L["bucket"] = np.floor(L["call"] / B) * B
    M = L.groupby(["side", "bucket"])["qty"].sum().reset_index()
    M.to_csv(OUT / "map.csv", index=False)
    log("지도", 목록시각=stamp, 종목수=len(L), 벽수=int((M["qty"] >= WALL).sum()))
    if dt.datetime.now(HKT).strftime("%H:%M") > "09:20":
        log("주의", 내용="장중에 받은 목록 — 이미 녹아웃된 CBBC 는 목록에서 빠져 있어 지도가 덜 정확함. 다음엔 09:20 HKT 전에 시작")
    return {(s, float(b)): q for s, b, q in M.itertuples(index=False)}


# ───────── 가격 ─────────
def yf_bars() -> pd.DataFrame:
    import yfinance as yf
    d = yf.download("^HSI", interval="1m", period="1d", progress=False, auto_adjust=False)
    if d is None or d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d.index = d.index.tz_convert(HKT)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    return d.iloc[:-1]            # 마지막 봉은 아직 덜 찬 봉일 수 있음


class NHPrice:
    """NH 실전 HSI 선물 1분봉 − 베이시스 → 지수 상당 가격."""

    def __init__(self, F):
        self.F = F
        self.basis = None

    def bars(self) -> pd.DataFrame:
        today = dt.datetime.now(HKT).strftime("%Y%m%d")
        r = self.F.chart_minute(SYM_PRICE, exch_cd="FHKEX", inq_strt_dt=today, inq_ed_dt=today, environment="real")
        rows = r.get("Occurs1") or []
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows)
        idx = pd.to_datetime(d["occurs1_ref_dt"] + d["occurs1_exch_tm"].str.zfill(6), format="%Y%m%d%H%M%S")
        f = pd.DataFrame({c.title(): pd.to_numeric(d[f"occurs1_{c}_pric"]) for c in ("open", "high", "low", "close")})
        f.index = (idx - pd.Timedelta(hours=1)).dt.tz_localize(HKT)    # NH 시각 = KST → HKT
        f = f[~f.index.duplicated()].sort_index().iloc[:-1]
        y = yf_bars()
        both = f["Close"].reindex(y.index).dropna()
        if len(both) >= 5:
            self.basis = float((both - y["Close"].reindex(both.index)).tail(30).median())
        if self.basis is None:
            return pd.DataFrame()                                      # 베이시스 모르면 판단 보류
        return f - self.basis


# ───────── 이벤트 ─────────
def events(g: pd.DataFrame, Q: dict) -> list[dict]:
    t = g.index.strftime("%H:%M")
    g = g[(t >= "09:30") & (t < "16:00")]
    if g.empty:
        return []
    o = float(g["Open"].iloc[0])
    base = np.floor(o / B) * B
    out = []
    for j in range(1, 41):
        e = base + B * j
        hit = g.index[g["High"] >= e]
        if len(hit):
            out.append({"time": hit[0], "dir": "up", "edge": e, "qty": Q.get(("bear", e), 0.0), "open": o})
        e = base - B * (j - 1)
        if e < o:
            hit = g.index[g["Low"] <= e]
            if len(hit):
                out.append({"time": hit[0], "dir": "down", "edge": e, "qty": Q.get(("bull", e - B), 0.0), "open": o})
    return sorted(out, key=lambda x: (x["time"], x["edge"]))


# ───────── 주문 (모의만) ─────────
class Broker:
    def __init__(self, F, A, live: bool):
        self.F, self.A, self.live = F, A, live

    def send(self, side: str, why: str, ref: float):
        """ref = 기준가(지수 상당). 매수는 ref+LIMIT_OFFSET, 매도는 ref−LIMIT_OFFSET 지정가."""
        px = None if ref != ref else int(round(ref + (LIMIT_OFFSET if side == "buy" else -LIMIT_OFFSET)))
        if not self.live:
            log("가상주문", side=side, 이유=why, 기준가=ref, 지정가=px)
            return None
        assert ORDER_ENV == "paper"
        if px is None:
            log("주문보류", side=side, 이유=why, 오류="기준가 없음 (가격 받기 실패) — HTS 에서 직접 확인")
            return None
        try:
            r = self.F.order(SYM_ORDER, qt_div_cd="1", trd_div_cd=self.A.SIDE_CODE[side], glb_ord_tp_cd="2",
                             ord_qty="1", ord_pric=str(px), stop_pric="", environment=ORDER_ENV)
            log("모의주문", side=side, 이유=why, 지정가=px, 주문번호=str(r.get("ord_no", "")).strip(),
                오류=str(r.get("err_cd", "")).strip() + " " + str(r.get("err_msg", "")).strip(), raw=r)
            return r
        except Exception as e:
            log("주문실패", side=side, 이유=why, 오류=str(e)[:300])
            return None

    def fills(self):
        if not self.live:
            return
        try:
            r = self.F.order_history(inq_div="2", prd_cd="MHI", environment=ORDER_ENV)
            rows = r.get("Occurs1") or []
            log("체결조회", 건수=len(rows), raw=rows[-5:])
        except Exception as e:
            log("체결조회실패", 오류=str(e)[:200])


def load_app(app_dir: str | None):
    if not app_dir:
        sys.exit("--app-dir 에 nh-futures-open-api 폴더 경로를 주세요")
    sys.path.insert(0, str(Path(app_dir).resolve()))
    from backend.overseas_futureoption import overseas_futureoption_functions as F
    from backend.overseas_futureoption import overseas_account as A
    return F, A


# ───────── 메인 루프 ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-order", action="store_true", help="신호마다 NH 모의계좌 주문")
    ap.add_argument("--test-order", action="store_true", help="모의 MHI 1계약 매수→60초 뒤 매도 시험")
    ap.add_argument("--price", choices=("yf", "nh"), default="yf")
    ap.add_argument("--app-dir")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    F = A = None
    if a.paper_order or a.test_order or a.price == "nh":
        F, A = load_app(a.app_dir)

    if a.test_order:
        if input(f"모의계좌({ORDER_ENV})에 {SYM_ORDER} 1계약 매수 후 60초 뒤 매도합니다. 진행하려면 YES: ").strip() != "YES":
            return
        b = Broker(F, A, live=True)
        g = yf_bars()
        ref = float(g["Close"].iloc[-1]) if len(g) else float("nan")
        b.send("buy", "체결 시험", ref)
        time.sleep(60)
        b.fills()
        b.send("sell", "체결 시험 청산", ref)
        time.sleep(20)
        b.fills()
        return

    log("시작", 모드="모의주문" if a.paper_order else "신호만(주문 없음)", 가격=a.price, 벽=WALL, 보유분=HOLD)
    Q = load_map()
    broker = Broker(F, A, live=a.paper_order)
    src = NHPrice(F) if a.price == "nh" else None
    seen, pos, trades = set(), None, []

    while True:
        now = dt.datetime.now(HKT)
        try:
            g = src.bars() if src else yf_bars()
        except Exception as e:
            log("가격실패", 오류=str(e)[:200])
            g = pd.DataFrame()
        last = float(g["Close"].iloc[-1]) if len(g) else np.nan
        last_t = g.index[-1] if len(g) else None

        # 청산
        if pos and now >= pos["exit_at"]:
            broker.send("buy" if pos["side"] == "sell" else "sell", "30분 청산", last)
            pos["exit_px"] = last
            sgn = 1 if pos["side"] == "buy" else -1
            pos["pnl_pt_virtual"] = sgn * (last - pos["entry_px"]) - COST if last == last else np.nan
            log("청산", **{k: v for k, v in pos.items() if k != "exit_at"})
            trades.append(pos)
            pd.DataFrame(trades).to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")
            broker.fills()
            pos = None

        # 새 이벤트
        for ev in events(g, Q) if len(g) else []:
            key = (ev["dir"], ev["edge"])
            if key in seen:
                continue
            seen.add(key)
            lag = (now - ev["time"].to_pydatetime()).total_seconds() / 60
            wall = ev["qty"] >= WALL
            log("이벤트", 시각=ev["time"].strftime("%H:%M"), 방향=ev["dir"], 경계=ev["edge"], 물량=round(ev["qty"], 1),
                벽=wall, 늦게앎_분=round(lag, 1))
            if not wall:
                continue
            exit_at = ev["time"].to_pydatetime() + dt.timedelta(minutes=HOLD)
            if pos is not None:
                log("건너뜀", 이유="이미 포지션 있음")
            elif lag > MAX_LAG:
                log("건너뜀", 이유=f"{lag:.0f}분 늦게 앎")
            elif exit_at.strftime("%H:%M") > "16:00":
                log("건너뜀", 이유="청산 시각이 16:00 넘음")
            else:
                side = "sell" if ev["dir"] == "up" else "buy"
                broker.send(side, f"{ev['dir']} {ev['edge']:.0f} 돌파, 물량 {ev['qty']:.0f}", last)
                pos = {"event_t": ev["time"].strftime("%H:%M"), "dir": ev["dir"], "edge": ev["edge"],
                       "qty": round(ev["qty"], 1), "side": side, "entry_t": now.strftime("%H:%M:%S"),
                       "entry_px": last, "lag_min": round(lag, 1), "exit_at": exit_at}
                broker.fills()

        if now.strftime("%H:%M") >= "16:05" and pos is None:
            log("종료", 거래수=len(trades),
                가상손익_pt=round(float(np.nansum([t["pnl_pt_virtual"] for t in trades])), 1) if trades else 0,
                마지막봉=str(last_t))
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
