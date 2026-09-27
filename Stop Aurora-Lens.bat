@echo off
setlocal EnableExtensions
aurora-lens stop %*
exit /b %ERRORLEVEL%
