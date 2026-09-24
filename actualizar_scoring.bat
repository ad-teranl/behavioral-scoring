@echo off
title Behavioral Scoring - Recalcular scoring
pushd "%~dp0"
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
python score_cartera.py
echo.
echo Scoring actualizado. Ya puedes abrir iniciar_dashboard.bat
popd
pause
