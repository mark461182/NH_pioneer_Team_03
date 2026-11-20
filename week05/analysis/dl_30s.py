"""KRX300·코스피200 선물 30초봉 내려받기 (chart-minute 의 dcnt=0, 조회만). 실행(VM): python3 dl_30s.py"""
import json, urllib.request, datetime as D, time, os, collections
B = 'http://127.0.0.1:47100/api/domestic'; K = D.timezone(D.timedelta(hours=9)); OUT = os.path.expanduser('~/nh_week05/krx300/data')
def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
    for i in range(3):
        try:
            with urllib.request.urlopen(r, timeout=30) as f: return 200, json.loads(f.read())
        except urllib.error.HTTPError as e: res = (e.code, e.read().decode()[:150])
        except Exception as e: res = (0, str(e)[:150])
        time.sleep(2)
    return res
def day(sym, ds):
    rows = {}; key = None
    for _ in range(14):
        body = {'sym': sym, 'inq_strt_dt': ds, 'inq_ed_dt': ds, 'dcnt': '0'}
        if key: body['nxt_key'] = key
        s, c = post('/chart-minute', body)
        if s != 200: return rows, (s, str(c)[:80])
        b = c.get('bars') or []; new = [x for x in b if x['timestamp'] not in rows]
        for x in b: rows[x['timestamp']] = x
        key = str(c.get('nextKey') or '').strip(); time.sleep(0.25)
        if not new or not key or key[:8] < ds: break
    return rows, None
for n, sym in (('krx300', 'A086C000'), ('k200', 'A016C000')):
    allb = {}; bad = 0; d = D.date(2026, 11, 19); errs = []
    while d >= D.date(2026, 9, 11) and bad < 7:
        if d.weekday() < 5:
            r, e = day(sym, d.strftime('%Y%m%d'))
            if e: errs.append((str(d), e))
            allb.update(r); bad = bad + 1 if not r else 0
        d -= D.timedelta(days=1)
    R = [allb[k] for k in sorted(allb)]; json.dump(R, open(f'{OUT}/{n}_30s.json', 'w'))
    dd = lambda b: D.datetime.fromtimestamp(b['timestamp'] / 1000, K).date(); cnt = collections.Counter(map(dd, R)); vol = collections.Counter()
    for b in R: vol[dd(b)] += b['volume']
    days = sorted(cnt); print(n, len(R), '일수', len(days), days[0] if days else '', '~', days[-1] if days else '', '하루 봉 중앙', sorted(cnt.values())[len(cnt)//2] if cnt else 0, '| 11/18 거래량 합', vol.get(D.date(2026, 11, 18)), '오류', errs[:3], flush=True)
print('DONE', flush=True)
