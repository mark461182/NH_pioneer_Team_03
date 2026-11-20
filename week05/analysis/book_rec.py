"""KRX300·코스피200·미니 선물 호가/현재가 녹화 (조회만, 주문 함수 안 부름). 앱 로컬 API 를 5초마다 돌림.
실행(VM): python3 book_rec.py [끝시각 HHMM, 기본 1550] [최대 횟수]  → ~/nh_week05/krx300/book/book_YYYYMMDD.jsonl
"""
import json, urllib.request, datetime as D, time, os, sys
B = 'http://127.0.0.1:47100/api/domestic'; K = D.timezone(D.timedelta(hours=9))
END = sys.argv[1] if len(sys.argv) > 1 else '1550'; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
OUT = os.path.expanduser('~/nh_week05/krx300/book'); os.makedirs(OUT, exist_ok=True)


def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=8) as f: return json.loads(f.read())
    except Exception as e: return {'err': str(e)[:120]}


SY = ('A086C000', 'A016C000', 'A056C000'); i = 0
f = open(f"{OUT}/book_{D.datetime.now(K):%Y%m%d}.jsonl", 'a')
while D.datetime.now(K).strftime('%H%M') < END and i < N:
    t0 = time.time(); row = {'t': D.datetime.now(K).isoformat(timespec='milliseconds')}
    for s in SY:
        b = post('/price-bid', {'sym': s})
        row[s] = {'bt': b.get('time'), 'a': [(x['price'], x['quantity']) for x in b.get('asks', [])],
                  'b': [(x['price'], x['quantity']) for x in b.get('bids', [])], 'e': b.get('err')}
    for s in SY[:2]:
        p = post('/price', {'sym': s})
        row[s].update(last=p.get('last'), vol=p.get('volume'), oi=p.get('openInterest'), spot=p.get('spot'), theory=p.get('theory'))
    f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush(); i += 1
    time.sleep(max(0.2, 5 - (time.time() - t0)))
