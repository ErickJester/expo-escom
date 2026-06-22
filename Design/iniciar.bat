@echo off
cd /d "%~dp0"
title ToonVerse - Servidor Local
echo.
echo  ============================================
echo    ToonVerse - Clasificador Neural
echo  ============================================
echo.
echo  Iniciando servidor en http://localhost:8000
echo  El navegador se abrira solo en unos segundos.
echo.
echo  Para DETENER: cierra esta ventana o pulsa Ctrl+C.
echo.
set OPEN_BROWSER=1

where py >nul 2>nul && goto :runpy
where python >nul 2>nul && goto :runpython
echo  [ERROR] No se encontro Python. Instala Python 3 y reintenta.
pause
goto :eof

:runpy
py server.py
goto :end

:runpython
python server.py
goto :end

:end
echo.
echo  El servidor se detuvo.
pause
