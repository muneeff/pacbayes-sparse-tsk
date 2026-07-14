@echo off
setlocal
cd /d "%~dp0.."
python scripts\extract_fuzzy_rule_examples_v4_3.py --project-root . || exit /b 1
echo Fuzzy-rule examples generated.
exit /b 0
