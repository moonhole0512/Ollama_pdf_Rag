@echo off
setlocal

REM --- Port Configuration ---
set "DEFAULT_PORT=8000"
set /p "USER_PORT=Enter the port number to use (default: %DEFAULT_PORT%): "
if not defined USER_PORT (
    set "APP_PORT=%DEFAULT_PORT%"
) else (
    set "APP_PORT=%USER_PORT%"
)
echo.
echo Starting server on port %APP_PORT%
echo.


REM --- Backend Setup ---
echo [Backend] Setting up Python environment...
set "BACKEND_DIR=%~dp0backend"
set "VENV_DIR=%BACKEND_DIR%\venv"

if not exist "%VENV_DIR%" (
    echo [Backend] Virtual environment not found. Creating it now...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [Backend] Failed to create virtual environment. Please check if Python is installed.
        exit /b %errorlevel%
    )
    echo [Backend] Virtual environment created.
)

echo [Backend] Activating virtual environment and installing packages...
call "%VENV_DIR%\Scripts\activate.bat"
pip install -r "%BACKEND_DIR%\requirements.txt" --log "%BACKEND_DIR%\pip_install.log"
if %errorlevel% neq 0 (
    echo [Backend] Failed to install packages. Check pip_install.log for details.
    exit /b %errorlevel%
)
echo [Backend] Python environment is ready.
echo.


REM --- Frontend Setup ---
echo [Frontend] Setting up Node.js environment and building frontend...
set "FRONTEND_DIR=%~dp0frontend"

if not exist "%FRONTEND_DIR%\node_modules" (
    echo [Frontend] Node.js dependencies not found. Installing...
    cmd /c "cd %FRONTEND_DIR% && npm install"
    if %errorlevel% neq 0 (
        echo [Frontend] Failed to install Node.js packages.
        exit /b %errorlevel%
    )
    echo [Frontend] Node.js dependencies installed.
) else (
    echo [Frontend] Node.js dependencies already installed.
)

echo [Frontend] Building frontend for production...
cmd /c "cd %FRONTEND_DIR% && npm run build"
if %errorlevel% neq 0 (
    echo [Frontend] Failed to build frontend.
    exit /b %errorlevel%
)
echo [Frontend] Frontend built successfully.
echo.


REM --- Start Server ---
echo Starting FastAPI server...
echo Visit http://localhost:%APP_PORT% in your browser.
set "PYTHONPATH=%~dp0"
uvicorn backend.main:app --host 0.0.0.0 --port %APP_PORT%

endlocal
pause
