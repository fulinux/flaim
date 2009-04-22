# AC_PROG_TRY_CSVM([quiet])
# -------------------------
# AC_PROG_TRY_CSVM looks for an existing CSharp virtual machine. 
# If the CSVM environment variable is not already set, it looks
# in the system path for a program named mono, and then for one
# named cs (the Microsoft CSharp VM).
#
# If no arguments are given to this macro, and no CSharp VM can
# be found, it prints a very visible message to STDOUT and to 
# the config.log file. If the "quiet" argument is passed, then
# only the normal "check" line is displayed. (Technically, any
# passing any value in the first argument has the same effect 
# as "quiet".)
#
# Makes CSVM precious to Autoconf. You can use CSVM in your 
# Makefile.in with @CSVM@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_CSVM],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([CSVM], [CSharp virtual machine])dnl
AC_CHECK_PROGS([CSVM], [mono$EXEEXT cs$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$CSVM"; then
  AC_MSG_WARN([
  -----------------------------------------
   No CSharp VM found - continuing without
   CSharp Virtual Machine support.
  -----------------------------------------])
fi])dnl
])
