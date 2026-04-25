@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo Installing Tesseract (UB Mannheim)...
winget install --id UB-Mannheim.TesseractOCR --exact --accept-package-agreements --accept-source-agreements --silent
if errorlevel 1 (
  echo Winget install failed.
  goto :fail
)

set "TESS=%ProgramFiles%\Tesseract-OCR\tesseract.exe"
if exist "%TESS%" goto :found
set "TESS=%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
if exist "%TESS%" goto :found

for /f "delims=" %%I in ('where tesseract 2^>nul') do (
  set "TESS=%%I"
  goto :found
)

echo Could not find tesseract.exe after install.
goto :fail

:found
for %%I in ("%TESS%") do set "TESSDIR=%%~dpI"
set "TESSDATA=%TESSDIR%tessdata"
set "URD=%TESSDATA%\urd.traineddata"

if not exist "%TESSDATA%" mkdir "%TESSDATA%"

if not exist "%URD%" (
  echo Downloading urd.traineddata...
  where curl >nul 2>nul
  if not errorlevel 1 (
    curl -L -o "%URD%" "https://github.com/tesseract-ocr/tessdata/raw/main/urd.traineddata"
  ) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest 'https://github.com/tesseract-ocr/tessdata/raw/main/urd.traineddata' -OutFile '%URD%'"
  )
)

echo.
echo Tesseract path: %TESS%
echo Installed languages:
"%TESS%" --list-langs

echo.
echo Done. Confirm that "urd" appears above.
pause
exit /b 0

:fail
echo.
echo Failed. Try running this .bat as Administrator.
pause
exit /b 1
