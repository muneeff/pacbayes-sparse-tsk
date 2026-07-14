@echo off
setlocal
cd /d "%~dp0\.."

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python is not available in the active environment.
  exit /b 1
)

set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

python scripts\run_development_v3.py ^
  --development configs\v3\fixed_k_sensitivity_v4.yaml ^
  --output results\development\fixed_k_sensitivity_v4 ^
  --workers 4
if errorlevel 1 exit /b 1

python scripts\summarize_fixed_k_sensitivity_v4.py ^
  --development configs\v3\fixed_k_sensitivity_v4.yaml ^
  --input results\development\fixed_k_sensitivity_v4 ^
  --output results\development\fixed_k_sensitivity_v4 ^
  --paper-root paper
if errorlevel 1 exit /b 1

python -m pytest tests\v3
if errorlevel 1 exit /b 1

echo.
echo Fixed-K sensitivity completed successfully.
echo Results: results\development\fixed_k_sensitivity_v4
echo Paper assets: paper\tables and paper\figures
endlocal
