@echo off
setlocal
pushd "%~dp0.."
".venv\Scripts\python.exe" -B -X utf8 "tools\training_monitor.py" --open
if errorlevel 1 (
 echo Monitor did not start. Check the message above.
 pause
)
popd
