@echo off
setlocal
cd /d "%~dp0.."
call tools\00_verify_prerequisites.cmd || exit /b 1
call tools\01_setup_environment.cmd || exit /b 1
call tools\02_generate_manuscript_assets.cmd || exit /b 1
call tools\07_run_statistical_analysis_v4_2.cmd || exit /b 1
call tools\08_run_posthoc_energy_benchmarks_v4_3.cmd || exit /b 1
call tools\09_extract_fuzzy_rules_v4_3.cmd || exit /b 1
call tools\03_validate_code_and_results.cmd || exit /b 1
call tools\04_compile_manuscript.cmd || exit /b 1
echo.
echo Complete V4.3 local build finished. The locked PJM gate was NOT rerun or modified.
exit /b 0
