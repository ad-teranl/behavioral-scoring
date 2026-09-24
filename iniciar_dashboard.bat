@echo off
title Behavioral Scoring Dashboard
pushd "%~dp0"
if not exist "app\Inicio.py" (
    echo ERROR: este archivo debe estar en la raiz de 2026_behavioral_scoring_prueba
    echo Carpeta actual: %CD%
    pause
    exit /b
)
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
python -m streamlit run app\Inicio.py
popd
pause
