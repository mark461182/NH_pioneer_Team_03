@echo off
rem NH pioneer week06 - live run on NH domestic PAPER account (quotes: real key, orders: paper key)
rem usage: run6_paper.bat            normal
rem        run6_paper.bat --ack      after checking HTS when stopped by reject / unfilled / leg failure
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=%USERPROFILE%\anaconda3\python.exe
set APP=%USERPROFILE%\Downloads\nh-futures-open-api
%PY% runner6.py --app-dir "%APP%" --paper-order %*
pause
