@ECHO OFF

setlocal

set solution=flaim.sln
set operation=Build

set build="Release"
if     "%1" == "debug"   (set build="Debug" && goto do_build)
if     "%1" == "release" (set build="Release" && goto do_build)
if NOT "%1" == ""         goto help

:do_build
echo %operation%ing %solution% %build% build...
devenv flaim-projects.sln /%operation% %build%

goto done

:help
echo Usage: %0 [release^|debug]
echo %operation%s the "release" build by default.

:done
endlocal
