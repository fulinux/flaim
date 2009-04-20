@ECHO OFF

setlocal

set solution=flaim-projects.sln
set operation=Build
set build=Release
set platform=Win32
set program=%0

:next_arg
shift
if "%0" == ""           goto do_build
if "%0" == "clean"      ((set operation=Clean) && goto next_arg)
if "%0" == "build"      ((set operation=Build) && goto next_arg)
if "%0" == "debug"      ((set build=Debug)     && goto next_arg)
if "%0" == "release"    ((set build=Release)   && goto next_arg)
if "%0" == "win32"      ((set platform=Win32)  && goto next_arg)
if "%0" == "32"         ((set platform=Win32)  && goto next_arg)
if "%0" == "win64"      ((set platform=x64)    && goto next_arg)
if "%0" == "64"         ((set platform=x64)    && goto next_arg)
goto help

:do_build
echo %operation%ing %solution% "%build%|%platform%" build...
devenv flaim-projects.sln /%operation% "%build%|%platform%"

goto done

:help
echo Usage: %program% [Build^|Clean] [Release^|Debug] [[Win]32^|[Win]64]
echo Builds the "Release|Win32" configuration by default.

:done
endlocal
