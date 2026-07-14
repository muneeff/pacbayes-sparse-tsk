@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" call tools\01_setup_environment.cmd || exit /b 1
call ".venv\Scripts\activate.bat" || exit /b 1
python tools\generate_manuscript_assets_v4.py --repo-root . --paper-root paper || exit /b 1
echo Generated LaTeX tables and PNG figures under paper\tables and paper\figures.
exit /b 0
