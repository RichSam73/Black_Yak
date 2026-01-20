@echo off
chcp 65001 >nul
echo ============================================
echo   PDF Translate Note - Docker Runner
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo [ERROR] .env file not found.
    echo Please copy .env.example to .env and fill in your API keys.
    echo.
    echo   copy .env.example .env
    echo   notepad .env
    echo.
    pause
    exit /b 1
)

REM Check if image exists, build if not
docker image inspect pdf-translate-note >nul 2>&1
if errorlevel 1 (
    echo [INFO] Building Docker image...
    docker build -t pdf-translate-note .
    if errorlevel 1 (
        echo [ERROR] Docker build failed.
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Starting PDF Translate Note...
echo [INFO] Open browser: http://localhost:7000
echo [INFO] Press Ctrl+C to stop
echo.

docker run -it --rm -p 7000:7000 --env-file .env pdf-translate-note
