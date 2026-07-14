@echo off
setlocal
cd /d "%~dp0.."
where python >nul 2>&1 || (echo ERROR: Python was not found. Activate fgan_env first.& exit /b 1)
python scripts\statistical_analysis_v4_2.py --repo-root . --bootstrap-resamples 20000 || exit /b 1
echo Statistical tables and confidence intervals were generated under results\statistics\v4_2 and paper\tables.
exit /b 0
