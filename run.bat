@echo off
REM ═══════════════════════════════════════════════════════════════
REM  CuaOS Launcher (Windows)
REM  Usage:
REM    run.bat                    — Mission Control with Fara-7B (default)
REM    run.bat fara               — Mission Control with Fara-7B
REM    run.bat ui-tars            — Mission Control with UI-TARS-1.5-7B
REM    run.bat qwen3              — Mission Control with Qwen3-VL-8B
REM    run.bat [model] terminal   — Terminal mode (no GUI)
REM ═══════════════════════════════════════════════════════════════

set "VENV=%~dp0.venv\Scripts"
set "HF_HOME=L:\models\huggingface"

REM Parse model argument
set "CUAOS_MODEL=fara-7b"
set "ENTRY=gui_mission_control.py"

if /i "%~1"=="fara" (
    set "CUAOS_MODEL=fara-7b"
    shift
)
if /i "%~1"=="ui-tars" (
    set "CUAOS_MODEL=ui-tars-1.5-7b"
    shift
)
if /i "%~1"=="qwen3" (
    set "CUAOS_MODEL=qwen3-vl-8b"
    shift
)
if /i "%~1"=="terminal" (
    set "ENTRY=main.py"
)

echo [CuaOS] Model:  %CUAOS_MODEL%
echo [CuaOS] Entry:  %ENTRY%
echo [CuaOS] HF dir: %HF_HOME%
echo.

"%VENV%\python.exe" "%~dp0%ENTRY%"
pause
