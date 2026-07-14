@echo off
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
  echo Usage:
  echo tools\05_publish_github.cmd https://github.com/OWNER/REPOSITORY.git "Your Name" "you@example.com"
  exit /b 1
)

set "REMOTE_URL=%~1"
set "AUTHOR_NAME=%~2"
set "AUTHOR_EMAIL=%~3"

where git >nul 2>&1 || (echo ERROR: Git is not installed or not in PATH.& exit /b 1)

if not exist ".git" git init -b main || exit /b 1
git branch -M main || exit /b 1

if not "%AUTHOR_NAME%"=="" git config user.name "%AUTHOR_NAME%"
if not "%AUTHOR_EMAIL%"=="" git config user.email "%AUTHOR_EMAIL%"

git config user.name >nul 2>&1 || (echo ERROR: Git user.name is not configured.& exit /b 1)
git config user.email >nul 2>&1 || (echo ERROR: Git user.email is not configured.& exit /b 1)

git add . || exit /b 1
git diff --cached --quiet
if errorlevel 1 (
  git commit -m "Release V4 reproducibility code, frozen results, and manuscript sources" || exit /b 1
) else (
  echo No new staged changes to commit.
)

git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin "%REMOTE_URL%" || exit /b 1
) else (
  git remote set-url origin "%REMOTE_URL%" || exit /b 1
)

git remote -v
git push -u origin main || exit /b 1

echo GitHub upload completed.
exit /b 0
