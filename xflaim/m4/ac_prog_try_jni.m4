# AC_PROG_TRY_JNI([quiet])
# ------------------------
# AC_PROG_TRY_JNI tests for the existence of the three
# tools required to build Java Native Interface (JNI) 
# modules: javac, javah, and jar
#
# If one or more of the tools are not found, and the "quiet"
# parameter was not passed, then it prints a very visible 
# message to STDOUT and to the log file indicating that the
# build process will continue without JNI support.
#
# In the process, the JAVAC, JAVAH and JAR environment 
# variables are made precious to Autoconf. You can use them 
# in your Makefile.in files with @JAVAC@, @JAVAH@ and @JAR@.
#
# If all three tools are present, then the ac_prog_have_jni
# environment variable is set to 'yes', otherwise it's set to
# 'no'.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JNI],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_PROG_TRY_JAVAC([quiet])dnl
AC_PROG_TRY_JAVAH([quiet])dnl
AC_PROG_TRY_JAR([quiet])dnl
m4_ifvaln([$1],,
[ac_prog_have_jni=yes
if test -z "$JAVAC"; then ac_prog_have_jni=no; fi
if test -z "$JAVAH"; then ac_prog_have_jni=no; fi
if test -z "$JAR"; then ac_prog_have_jni=no; fi
if test "x$ac_prog_have_jni" = xno; then
  AC_MSG_WARN([
  -----------------------------------------
   Some required Java Native Interface
   tools missing - continuing without 
   project JNI support.
  -----------------------------------------])
fi])dnl
])
