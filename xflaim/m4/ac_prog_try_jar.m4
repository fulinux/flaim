# AC_PROG_TRY_JAR([quiet])
# ------------------------
# AC_PROG_TRY_JAR tests for an existing jar program. If the JAR
# environment variable is empty, then it searches for a jar program
# in the system search path.
#
# If no arguments are given to this macro, and no jar program 
# can be found, it prints a very visible message to STDOUT and
# to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. (Technically, 
# any passing any value in the first argument has the same effect 
# as passing "quiet".)
#
# Makes JAR precious to Autoconf. You can use JAR in your
# Makefile.in files with @JAR@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAR],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAR], [Java archive utility])dnl
AC_CHECK_PROGS([JAR], [jar$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$JAR"; then
  AC_MSG_WARN([
  -----------------------------------------
   No Java jar utility found - continuing
   without Java Archive support.
  -----------------------------------------])
fi])dnl
])
