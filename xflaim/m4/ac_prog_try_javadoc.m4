# AC_PROG_TRY_JAVADOC([quiet])
# ----------------------------
# AC_PROG_TRY_JAVADOC tests for an existing javadoc generator.
# If the JAVADOC environment variable is not set, it searches the
# system path to find it. 
#
# If no arguments are given to this macro, and no doxygen 
# program can be found, it prints a very visible message 
# to STDOUT and to the config.log file. If the "quiet" 
# argument is passed, then only the normal "check" line
# is displayed. (Technically, any passing any value in 
# the first argument has the same effect as "quiet".)
#
# Makes JAVADOC precious to Autoconf. You can use the JAVADOC 
# variable in your Makefile.in files with @JAVADOC@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVADOC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVADOC], [Java source documentation utility])dnl
AC_CHECK_PROGS([JAVADOC], [javadoc$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$JAVADOC"; then
  AC_MSG_WARN([
  -----------------------------------------
   No javadoc program found - continuing
   without javadoc documentation support.
  -----------------------------------------])
fi])dnl
])
