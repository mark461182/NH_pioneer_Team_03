"""투자자별 매매 일별 내려받기 (KRX300=08, 코스피200=01, 미니=05, 코스닥150=06; 수량 기준, 조회만). 실행(VM): python3 inv_dl.py"""
import json, urllib.request, datetime as D, time, os
B = 'http://127.0.0.1:47100/api/domestic'; OUT = os.path.expanduser('~/nh_week05/krx300/data')
def post(p, body):
    r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=15) as f: return json.loads(f.read())
    except Exception as e: return {'err': str(e)[:100]}
rows = []; d = D.date(2026, 11, 19)
while d >= D.date(2026, 9, 1):
    if d.weekday() < 5:
        for prd in ('08', '01', '05', '06'):
            c = post('/trend-investor', {'prd': prd, 'ref_dt': d.strftime('%Y%m%d'), 'inq_tp': '1'}); time.sleep(0.25)
            for i in c.get('investors', []):
                if i['buy'] or i['sell']: rows.append({'date': str(d), 'prd': prd, 'group': i['group'], 'buy': i['buy'], 'sell': i['sell']})
    d -= D.timedelta(days=1)
json.dump(rows, open(OUT + '/investor_daily.json', 'w')); ds = sorted({r['date'] for r in rows if r['prd'] == '08'})
print('행', len(rows), '| KRX300 자료 있는 날', len(ds), ds[:1], ds[-1:])
