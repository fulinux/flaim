# AC_PROG_TRY_JAVAC([quiet])
# --------------------------
# AC_PROG_TRY_JAVAC looks for an existing Java compiler. If the
# JAVAC environment variable is not set, it searches the system 
# path for a Java compiler, beginning with the free ones.
#
# If no arguments are given to this macro, and no Java compiler
# can be found, it prints a very visible message to STDOUT and 
# to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. (Technically,
# any passing any value in the first argument has the same effect
# as "quiet".)
#
# Makes JAVAC precious to Autoconf. You can use the JAVAC 
# variable in your Makefile.in files with @JAVAC@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVAC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVAC], [Java compiler])dnl
AC_CHECK_PROGS([JAVAC], ["gcj$EXEEXT -C" guavac$EXEEXT jikes$EXEEXT javac$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$DOXYGEN"; then
  AC_MSG_WARN([
  -----------------------------------------
   No Java compiler found - continuing
   without Java compiler support.
  -----------------------------------------])
fi])dnl
])
