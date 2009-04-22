# AC_PROG_TRY_JAVA([quiet])
# -------------------------
# AC_PROG_TRY_JAVA looks for an existing JAVA virtual machine. If 
# the JAVA environment variable is empty, it searches the system 
# path for a java program.
#
# If no arguments are given to this macro, and no java virtual
# machine can be found, it prints a very visible message to STDOUT
# and to the config.log file. If the "quiet" argument is passed,
# then only the normal "check" line is displayed. (Technically, 
# any passing any value in the first argument has the same effect
# as "quiet".)
#
# Makes JAVA precious to Autoconf. You can use the JAVA variable
# in your Makefile.in files with @JAVA@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVA],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVA], [Java virtual machine])dnl
AC_CHECK_PROGS([JAVA], [kaffe$EXEEXT java$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$DOXYGEN"; then
  AC_MSG_WARN([
  -----------------------------------------
   No Doxygen program found - continuing
   without Doxygen documentation support.
  -----------------------------------------])
fi])dnl
])
