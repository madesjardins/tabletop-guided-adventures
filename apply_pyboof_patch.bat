@echo off
REM Apply PyBoof mmap bug fix patch
REM This script automatically detects the PyBoof installation location and applies the patch

echo ============================================================
echo PyBoof mmap Bug Fix Patcher
echo ============================================================
echo.

REM Get PyBoof installation path using Python
echo Detecting PyBoof installation location...
python -c "import pyboof; import os; print(os.path.dirname(pyboof.__file__))" > pyboof_path.tmp 2>nul

if errorlevel 1 (
    echo ERROR: Could not find PyBoof installation.
    echo Please ensure PyBoof is installed: pip install pyboof
    del pyboof_path.tmp 2>nul
    pause
    exit /b 1
)

set /p PYBOOF_DIR=<pyboof_path.tmp
del pyboof_path.tmp

echo PyBoof found at: %PYBOOF_DIR%
echo.

REM Check if __init__.py exists
if not exist "%PYBOOF_DIR%\__init__.py" (
    echo ERROR: __init__.py not found in PyBoof directory
    pause
    exit /b 1
)

REM Check specifically for the buggy line in the Windows section (after 'if os.name')
echo Checking if patch is needed...
powershell -Command "$content = Get-Content '%PYBOOF_DIR%\__init__.py' -Raw; if ($content -match '(?s)if os\.name == ''nt'':\s+mmap_file = mmap\.mmap') { exit 0 } else { exit 1 }"
if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo Patch already applied!
    echo ============================================================
    echo The PyBoof mmap bug fix has already been applied to:
    echo %PYBOOF_DIR%\__init__.py
    echo.
    echo No action needed.
    pause
    exit /b 0
)

echo Found buggy code in Windows section that needs patching...

REM Create backup
echo Creating backup...
copy "%PYBOOF_DIR%\__init__.py" "%PYBOOF_DIR%\__init__.py.backup" >nul
if errorlevel 1 (
    echo ERROR: Failed to create backup
    pause
    exit /b 1
)
echo Backup created: %PYBOOF_DIR%\__init__.py.backup

REM Apply patch using PowerShell for better text replacement
echo.
echo Applying patch...
powershell -Command "$content = Get-Content '%PYBOOF_DIR%\__init__.py' -Raw; $content = $content -replace '        mmap_file = mmap\.mmap\(pbg\.mmap_fid\.fileno\(\), length=0\)', '        pbg.mmap_file = mmap.mmap(pbg.mmap_fid.fileno(), length=0)'; Set-Content '%PYBOOF_DIR%\__init__.py' -Value $content -NoNewline"

if errorlevel 1 (
    echo ERROR: Failed to apply patch
    echo Restoring backup...
    copy "%PYBOOF_DIR%\__init__.py.backup" "%PYBOOF_DIR%\__init__.py" >nul
    pause
    exit /b 1
)

REM Verify patch was applied
findstr /C:"pbg.mmap_file = mmap.mmap(pbg.mmap_fid.fileno(), length=0)" "%PYBOOF_DIR%\__init__.py" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Patch verification failed
    echo Restoring backup...
    copy "%PYBOOF_DIR%\__init__.py.backup" "%PYBOOF_DIR%\__init__.py" >nul
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Patch applied successfully!
echo ============================================================
echo.
echo Fixed file: %PYBOOF_DIR%\__init__.py
echo Backup saved: %PYBOOF_DIR%\__init__.py.backup
echo.
echo The PyBoof mmap bug has been fixed. You can now use MicroQR detection.
echo.
pause
