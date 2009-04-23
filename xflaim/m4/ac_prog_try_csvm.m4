# AC_PROG_TRY_CSVM(["quiet"])
# ---------------------------
# AC_PROG_TRY_CSVM tests for an existing CSharp virtual machine. 
# It sets or uses the environment variable CSVM.
#
# If no arguments are given to this macro, and no CSharp virtual
# machine can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the CSVM variable precious to Autoconf. You can 
# use the CSVM variable in your Makefile.in files with 
# @CSVM@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-23
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_CSVM],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([CSVM], [CSharp virtual machine])dnl
AC_CHECK_PROGS([CSVM], [mono$EXEEXT cs$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$CSVM"; then
  AC_MSG_WARN([No CSharp virtual machine found - continuing without CSVM support])
fi])dnl
])
