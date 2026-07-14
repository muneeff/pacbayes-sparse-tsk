@echo off
setlocal
cd /d "%~dp0..\paper"

where latexmk >nul 2>&1
if not errorlevel 1 (
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
  copy /y main.pdf PACBayes_TSK_Manuscript_V4.pdf >nul
  echo Manuscript built: paper\PACBayes_TSK_Manuscript_V4.pdf
  exit /b 0
)

where pdflatex >nul 2>&1 || (echo ERROR: pdflatex was not found.& exit /b 1)
where bibtex >nul 2>&1 || (echo ERROR: bibtex was not found.& exit /b 1)

pdflatex -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
bibtex main || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error main.tex || exit /b 1
copy /y main.pdf PACBayes_TSK_Manuscript_V4.pdf >nul

echo Manuscript built: paper\PACBayes_TSK_Manuscript_V4.pdf
exit /b 0
