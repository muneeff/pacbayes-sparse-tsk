@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" call tools\01_setup_environment.cmd || exit /b 1
call ".venv\Scripts\activate.bat" || exit /b 1
python -m pytest tests\v3 || exit /b 1
python scripts\validate_v3.py || exit /b 1
echo Validation completed successfully.
exit /b 0
