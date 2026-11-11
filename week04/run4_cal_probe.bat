@echo off
rem NH pioneer week04 - calendar spread probe (quotes only, no orders)
set QENV=real
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set APP=%USERPROFILE%\Downloads\nh-futures-open-api
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
echo calendar probe (about 3-6 min, quote env = %QENV%)
%PY% probe_calendar.py --app-dir "%APP%" --quote-env %QENV% --days 60 > results\04_cal_probe.txt 2>&1
powershell -NoProfile -Command "Get-Content -Encoding UTF8 'results\04_cal_probe.txt' | Select-Object -Last 12"
echo.
echo done. tell Claude it is finished.
pause
