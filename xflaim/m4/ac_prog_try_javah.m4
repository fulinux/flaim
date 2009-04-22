# AC_PROG_TRY_JAVAH([quiet])
# --------------------------
# AC_PROG_TRY_JAVAH looks for an existing Java native header (JNI)
# generator. If the JAVAH environment variable is not set, it looks
# in the system path for a javah program. 
#
# If no arguments are given to this macro, and no javah program
# can be found, it prints a very visible message to STDOUT and 
# to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. (Technically, 
# any passing any value in the first argument has the same effect
# as "quiet".)
#
# Makes JAVAH precious to Autoconf. You can use the JAVAH
# variable in your Makefile.in files with @JAVAH@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVAH],
[AC_REQUIRE([AC_CANONICAL_SYSTEM])dnl
AC_REQUIRE([AC_PROG_CPP])dnl
AC_ARG_VAR([JAVAH], [Java header utility])dnl
AC_CHECK_PROGS([JAVAH], [javah])
m4_ifvaln([$1],,
[if test -z "$DOXYGEN"; then
  AC_MSG_WARN([
  -----------------------------------------
   No javah utility found - continuing
   without Java Header utility support.
  -----------------------------------------])
fi])dnl
])
