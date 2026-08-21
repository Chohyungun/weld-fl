@echo off
REM ===========================================================================
REM publish.bat — publish.sh 를 Git Bash 로 실행하는 Windows 래퍼.
REM
REM   저장소는 만들지 않는다. GitHub 에서 직접 만든 빈 저장소 URL 을 넘겨라.
REM
REM   사용:
REM     scripts\publish.bat <원격URL>            미리보기
REM     scripts\publish.bat <원격URL> --apply    실제 수행
REM ===========================================================================
setlocal

if "%~1"=="" (
  echo.
  echo   원격 URL 이 필요합니다.
  echo.
  echo   사용법:
  echo     scripts\publish.bat https://github.com/USER/weld-fl.git
  echo     scripts\publish.bat https://github.com/USER/weld-fl.git --apply
  echo.
  exit /b 1
)

set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"
if not exist "%BASH%" (
  echo.
  echo   Git Bash 를 찾지 못했습니다. Git for Windows 설치를 확인하세요.
  echo   확인한 경로:
  echo     %ProgramFiles%\Git\bin\bash.exe
  echo     %ProgramFiles(x86)%\Git\bin\bash.exe
  echo     %LOCALAPPDATA%\Programs\Git\bin\bash.exe
  echo.
  exit /b 1
)

pushd "%~dp0\.."
"%BASH%" scripts/publish.sh %*
set "RC=%ERRORLEVEL%"
popd

endlocal & exit /b %RC%
