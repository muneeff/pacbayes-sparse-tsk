@echo off
setlocal
cd /d "%~dp0.."
echo [1/4] Checking Git...
where git >nul 2>&1 || (echo ERROR: Git is not installed or not in PATH.& exit /b 1)
git --version

echo [2/4] Checking Python...
where python >nul 2>&1 || (echo ERROR: Python is not installed or not in PATH.& exit /b 1)
python --version

echo [3/4] Checking LaTeX...
where latexmk >nul 2>&1 && (latexmk -v | findstr /i "Latexmk" & goto latex_ok)
where pdflatex >nul 2>&1 || (echo ERROR: Install MiKTeX or TeX Live and ensure pdflatex is in PATH.& exit /b 1)
where bibtex >nul 2>&1 || (echo ERROR: bibtex is required for the bibliography.& exit /b 1)
:latex_ok

echo [4/4] Project root:
cd

echo.
echo All required commands were found.
exit /b 0
