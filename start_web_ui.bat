@echo off
setlocal EnableExtensions

if /i "%~1"=="--worker" goto :main
if /i "%~1"=="--dry-run" set "DRY_RUN=1"

if not defined DRY_RUN (
  cmd /k call "%~f0" --worker
  exit /b %errorlevel%
)

:main
cd /d "%~dp0"

set "APP_URL=http://localhost:8501"
set "HEALTH_URL=http://localhost:8501/_stcore/health"
set "VENV_DIR=.venv311"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

call :resolve_runtime_root
if errorlevel 1 goto :error

call :prepare_log
if errorlevel 1 goto :error

call :configure_paddle_env
if errorlevel 1 goto :error

call :resolve_bootstrap_python
if errorlevel 1 goto :error

if not defined DRY_RUN (
  call :check_existing_server
  if not errorlevel 1 (
    echo Web UI is already running. Opening browser...
    call :log "Web UI is already running."
    start "" "%APP_URL%"
    goto :success
  )
)

if not exist "%VENV_PY%" (
  echo [1/5] Creating virtual environment...
  call :log "[1/5] Creating virtual environment."
  if defined DRY_RUN (
    echo [DRY RUN] %BOOTSTRAP_CMD% -m venv "%VENV_DIR%"
  ) else (
    call %BOOTSTRAP_CMD% -m venv "%VENV_DIR%" >>"%LOG_FILE%" 2>&1 || goto :error
  )
)

echo [2/5] Checking required packages...
call :log "[2/5] Checking required packages."
if defined DRY_RUN (
  call :print_run_python -c "import streamlit, faster_whisper, cv2, yt_dlp, numpy, paddleocr, paddle"
) else (
  call :run_python -c "import streamlit, faster_whisper, cv2, yt_dlp, numpy, paddleocr, paddle" >>"%LOG_FILE%" 2>&1
  if errorlevel 1 (
    echo [3/5] Installing required packages...
    call :log "[3/5] Installing required packages."
    call :run_python -m pip install --upgrade pip >>"%LOG_FILE%" 2>&1 || goto :error
    call :run_python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ >>"%LOG_FILE%" 2>&1 || goto :error
    call :run_python -m pip install -r requirements.txt >>"%LOG_FILE%" 2>&1 || goto :error
  ) else (
    echo Dependencies are already installed.
    call :log "Dependencies are already installed."
  )
)

echo [4/5] Opening browser...
call :log "[4/5] Opening browser."
if defined DRY_RUN (
  echo [DRY RUN] Browser wait helper will open %APP_URL% after the health check succeeds.
  call :print_run_python -m streamlit run web_ui.py
  goto :success
)

start "" powershell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command "$health='%HEALTH_URL%'; $app='%APP_URL%'; for ($i = 0; $i -lt 60; $i++) { try { $response = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -eq 200) { Start-Process $app; exit 0 } } catch {}; Start-Sleep -Seconds 1 }; Start-Process $app"

echo [5/5] Starting Web UI...
echo Web UI: %APP_URL%
echo Log file: %LOG_FILE%
echo This window stays open after errors so you can inspect the log.
echo Press Ctrl+C in this window to stop the app.
call :log "[5/5] Starting Web UI."
call :run_python -m streamlit run web_ui.py >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%errorlevel%"
call :log "Streamlit exit code: %EXIT_CODE%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Streamlit exited with code %EXIT_CODE%.
  echo Showing the last lines from %LOG_FILE%...
  call :show_log_tail
  echo.
  endlocal & exit /b %EXIT_CODE%
)

echo Web UI stopped.
goto :success

:resolve_runtime_root
if defined SHORTS_VISUAL_TRANSCRIBER_RUNTIME (
  call :try_runtime_root "%SHORTS_VISUAL_TRANSCRIBER_RUNTIME%"
  if not errorlevel 1 exit /b 0
)

if defined PUBLIC (
  call :try_runtime_root "%PUBLIC%\shorts_visual_transcriber_runtime"
  if not errorlevel 1 exit /b 0
)

call :try_runtime_root "C:\Users\Public\shorts_visual_transcriber_runtime"
if not errorlevel 1 exit /b 0

call :try_runtime_root "C:\shorts_visual_transcriber_runtime"
if not errorlevel 1 exit /b 0

echo ASCII runtime directory could not be created.
exit /b 1

:try_runtime_root
set "RUNTIME_ROOT=%~1"
if defined DRY_RUN exit /b 0
if not exist "%RUNTIME_ROOT%" mkdir "%RUNTIME_ROOT%" >nul 2>nul
if exist "%RUNTIME_ROOT%\" exit /b 0
exit /b 1

:prepare_log
set "LOG_DIR=%RUNTIME_ROOT%\logs"
set "LOG_FILE=%LOG_DIR%\web_ui_latest.log"
if defined DRY_RUN (
  echo Log file: %LOG_FILE%
  exit /b 0
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul
if not exist "%LOG_DIR%\" exit /b 1
>"%LOG_FILE%" (
  echo ==== Shorts Visual Transcriber Web UI ====
  echo Started: %DATE% %TIME%
  echo Project root: %CD%
  echo App URL: %APP_URL%
  echo.
)
exit /b 0

:configure_paddle_env
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
set "PADDLE_PDX_CACHE_HOME=%RUNTIME_ROOT%\paddlex_cache"
set "PADDLE_OCR_BASE_DIR=%RUNTIME_ROOT%\paddleocr"
if defined DRY_RUN exit /b 0
if not exist "%PADDLE_PDX_CACHE_HOME%" mkdir "%PADDLE_PDX_CACHE_HOME%" >nul 2>nul
if not exist "%PADDLE_OCR_BASE_DIR%" mkdir "%PADDLE_OCR_BASE_DIR%" >nul 2>nul
if not exist "%PADDLE_PDX_CACHE_HOME%\" exit /b 1
if not exist "%PADDLE_OCR_BASE_DIR%\" exit /b 1
exit /b 0

:log
if defined DRY_RUN exit /b 0
>>"%LOG_FILE%" echo %~1
exit /b 0

:check_existing_server
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; try { $response = Invoke-WebRequest -Uri '%HEALTH_URL%' -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>nul
exit /b %errorlevel%

:show_log_tail
if not exist "%LOG_FILE%" exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path '%LOG_FILE%' -Tail 80"
exit /b 0

:resolve_bootstrap_python
where py >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP_CMD=py -3.11"
  exit /b 0
)

where python >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP_CMD=python"
  exit /b 0
)

echo Python 3.11 was not found.
exit /b 1

:run_python
if exist "%VENV_PY%" (
  call "%VENV_PY%" %*
) else (
  call %BOOTSTRAP_CMD% %*
)
exit /b %errorlevel%

:print_run_python
if exist "%VENV_PY%" (
  echo [DRY RUN] "%VENV_PY%" %*
) else (
  echo [DRY RUN] %BOOTSTRAP_CMD% %*
)
exit /b 0

:error
set "EXIT_CODE=%errorlevel%"
if "%EXIT_CODE%"=="0" set "EXIT_CODE=1"
echo.
echo Startup failed.
if defined LOG_FILE (
  echo Log file: %LOG_FILE%
  call :show_log_tail
)
echo This window stays open so the error can be inspected.
endlocal & exit /b %EXIT_CODE%

:success
endlocal & exit /b 0
