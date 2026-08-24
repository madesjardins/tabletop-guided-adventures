@echo off
REM Launch Tabletop Guided Adventures application
REM Uses uv to run inside the project virtual environment

REM Load environment variables from .env if it exists (e.g. HF_TOKEN)
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

echo Launching Tabletop Guided Adventures...
uv run python\launch_ttga.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo ============================================================
    echo Application exited with error code: %errorlevel%
    echo ============================================================
    pause
) else (
    echo Application closed successfully.
    timeout /t 3
)
