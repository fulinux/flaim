# AC_PROG_TRY_DOXYGEN([quiet])
# ----------------------------
# AC_PROG_TRY_DOXYGEN tests for an existing doxygen source
# documentation program. It sets or uses the environment 
# variable DOXYGEN. 
#
# If no arguments are given to this macro, and no doxygen 
# program can be found, it prints a very visible message 
# to STDOUT and to the config.log file. If the "quiet" 
# argument is passed, then only the normal "check" line
# is displayed. (Technically, any passing any value in 
# the first argument has the same effect as "quiet".)
#
# Makes DOXYGEN precious to Autoconf. You can use DOXYGEN in
# your Makefile.in files with @DOXYGEN@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_DOXYGEN],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([DOXYGEN], [Doxygen source doc generation program])dnl
AC_CHECK_PROGS([DOXYGEN], [doxygen$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$DOXYGEN"; then
  AC_MSG_WARN([
  -----------------------------------------
   No Doxygen program found - continuing
   without Doxygen documentation support.
  -----------------------------------------])
fi])dnl
])
