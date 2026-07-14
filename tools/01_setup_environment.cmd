@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  python -m venv .venv || exit /b 1
)

call ".venv\Scripts\activate.bat" || exit /b 1

if not exist ".venv\.dependencies_v4_installed" (
  echo Installing frozen dependencies once...
  python -m pip install --upgrade pip || exit /b 1
  python -m pip install -r requirements-lock.txt || exit /b 1
  python -m pip install -e . --no-deps || exit /b 1
  type nul > ".venv\.dependencies_v4_installed"
) else (
  echo Dependencies already installed; skipping reinstall.
)

python -c "import pacbayes_tsk, pandas, matplotlib; print('Python environment OK')" || exit /b 1
exit /b 0
