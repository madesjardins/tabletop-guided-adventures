@echo off
REM Launch Tabletop Guided Adventures application
REM Activates conda environment and runs the main application

echo Activating conda environment: ttga_env
call conda activate ttga_env

if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'ttga_env'
    echo Please ensure the environment exists: conda create -n ttga_env python=3.13
    pause
    exit /b 1
)

echo Launching Tabletop Guided Adventures...
python python\launch_ttga.py

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
