@echo off
setlocal
cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:8080"

rem Do not start a second NiceGUI instance when SAMcloud is already running.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$bindListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, 8080); try { $bindListener.Start(); $bindListener.Stop(); exit 1 } catch {}; try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri 'http://127.0.0.1:8080/' -ErrorAction Stop | Out-Null; exit 0 } catch {}; $connections = @(Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue | Where-Object { ($_.State -in @('Listen', 'Bound') -or [string]::IsNullOrEmpty([string]$_.State)) -and $_.OwningProcess -ne 0 }); $processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }); Write-Host 'Port 8080 is occupied, but SAMcloud did not answer at http://127.0.0.1:8080/.'; if ($processIds.Count -eq 0) { Write-Host 'Windows did not report an owning PID for port 8080.' } else { foreach ($processId in $processIds) { $process = Get-CimInstance Win32_Process -Filter ('ProcessId = {0}' -f $processId) -ErrorAction SilentlyContinue; if ($process) { Write-Host ('PID {0}: {1}' -f $processId, $process.CommandLine) } else { Write-Host ('PID {0}: command line unavailable' -f $processId) } } }; Write-Host 'Close the process above, for example: taskkill /PID <pid> /T /F, then try again.'; exit 2"
set "PORT_CHECK_EXIT=%errorlevel%"
if "%PORT_CHECK_EXIT%" == "0" (
    echo SAMcloud is already running on port 8080.
    echo Opening %APP_URL% in your browser...
    start "" "%APP_URL%"
    pause
    exit /b 0
)
if "%PORT_CHECK_EXIT%" == "2" (
    pause
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo Starting SAMcloud...
echo Open %APP_URL% in your browser if it does not open automatically.
"%~dp0venv\Scripts\python.exe" "%~dp0main.py"
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%" == "0" (
    echo.
    echo SAMcloud stopped with exit code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
