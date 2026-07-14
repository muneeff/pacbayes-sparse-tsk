@echo off
setlocal
cd /d "%~dp0.."
call tools\00_verify_prerequisites.cmd || exit /b 1
call tools\01_setup_environment.cmd || exit /b 1
call tools\02_generate_manuscript_assets.cmd || exit /b 1
call tools\03_validate_code_and_results.cmd || exit /b 1
call tools\04_compile_manuscript.cmd || exit /b 1
echo.
echo Local build completed. GitHub was NOT modified.
exit /b 0
