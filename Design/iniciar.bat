@echo off
REM ============================================================
REM  ToonVerse - arranca el backend del modelo y abre Chrome
REM ============================================================
cd /d "%~dp0"
echo Iniciando servidor del modelo (cartoon_v3_local.pt)...
start "ToonVerse server" py server.py
echo Esperando a que cargue el modelo...
timeout /t 12 /nobreak >nul
echo Abriendo Chrome en http://localhost:8000
start chrome "http://localhost:8000"
echo.
echo La pagina ya esta abierta. Deja la ventana del servidor abierta.
echo Para detener: cierra la ventana "ToonVerse server".
