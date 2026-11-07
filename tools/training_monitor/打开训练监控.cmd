@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1
".venv\Scripts\python.exe" -B -X utf8 -m tools.training_monitor --port 18765 --open
if errorlevel 1 (
 echo Monitor did not start. Check the message above.
 pause
)
popd
