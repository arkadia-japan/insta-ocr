@echo off
setlocal EnableExtensions

if /i "%~1"=="--worker" goto :main
cmd /k call "%~f0" --worker
exit /b %errorlevel%

:main
cd /d "%~dp0"

set "VENV_PY=.venv311\Scripts\python.exe"
set "SERVICE_ACCOUNT_FILE=C:\Users\yoona\secure\shorts-analyzer-490407-b7632627fff3.json"
set "SPREADSHEET_ID=1kFFRfOkcgtp0a4SZZ5q5hxMXLNAwlAGO6-bCEROAgCo"
set "URL_COLUMN=G"
set "GOOGLE_API_PYTHON=C:\Python313\python.exe"
set "LOG_DIR=C:\Users\Public\shorts_visual_transcriber_runtime\logs"
set "LOG_FILE=%LOG_DIR%\sheet_sync_latest.log"
set "EXIT_CODE=1"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul
>"%LOG_FILE%" (
  echo ==== Shorts Visual Transcriber Sheet Sync ====
  echo Started: %DATE% %TIME%
  echo Project root: %CD%
  echo Mode: retry error rows
  echo.
)

if not exist "%VENV_PY%" (
  echo Virtual environment Python was not found: %VENV_PY%
  >>"%LOG_FILE%" echo Virtual environment Python was not found: %VENV_PY%
  goto :error
)

if not exist "%SERVICE_ACCOUNT_FILE%" (
  echo Service account JSON was not found: %SERVICE_ACCOUNT_FILE%
  >>"%LOG_FILE%" echo Service account JSON was not found: %SERVICE_ACCOUNT_FILE%
  goto :error
)

set "SHEETS_SYNC_GOOGLE_PYTHON=%GOOGLE_API_PYTHON%"

echo Retrying error rows...
echo Log file: %LOG_FILE%
echo.
>>"%LOG_FILE%" echo Retrying error rows.
"%VENV_PY%" -m app.sheets_sync ^
  --service-account-file "%SERVICE_ACCOUNT_FILE%" ^
  --spreadsheet-id "%SPREADSHEET_ID%" ^
  --url-column "%URL_COLUMN%" ^
  --retry-errors >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" goto :error

echo Sync finished successfully.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOG_FILE%' -Tail 80"
goto :success

:error
echo.
echo Sheet sync failed with exit code %EXIT_CODE%.
echo Showing the last log lines from %LOG_FILE%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOG_FILE%' -Tail 120"
echo.
echo This window stays open so the error can be inspected.
exit /b %EXIT_CODE%

:success
echo.
echo Press any key to close this window.
pause >nul
exit /b 0
