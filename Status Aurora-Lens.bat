@echo off
setlocal EnableExtensions
aurora-lens status %*
exit /b %ERRORLEVEL%
