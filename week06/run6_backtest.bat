@echo off
rem NH pioneer week06 - tests, before/after backtest, theory numbers, replay demos (no API needed)
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
echo [1/4] tests
%PY% test6.py > results\06_test.txt 2>&1
powershell -NoProfile -Command "Get-Content -Encoding UTF8 results\06_test.txt | Select-Object -Last 1"
echo [2/4] before/after backtest (first run about 10 min, then about 2 min)
%PY% backtest6.py > results\06_backtest.txt 2>&1
echo [3/4] theory numbers
%PY% theory6.py > results\06_theory.txt 2>&1
echo [4/4] replay demos
%PY% demo6.py --date 20251110 > results\06_demo1.txt 2>&1
%PY% demo6.py --date 20260212 --reject-nth 2 > results\06_demo2.txt 2>&1
powershell -NoProfile -Command "Get-Content -Encoding UTF8 results\06_backtest.txt | Select-Object -First 20"
echo done. results\compare.csv, results\figs\*.png, results\demo_*\log.jsonl
pause
