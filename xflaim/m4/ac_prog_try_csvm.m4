# AC_PROG_TRY_CSVM(["quiet"])
# ---------------------------
# AC_PROG_TRY_CSVM tests for an existing CSharp virtual machine. 
# It sets or uses the environment variable CSVM.
#
# If no arguments are given to this macro, and no CSharp virtual
# machine can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the CSVM variable precious to Autoconf. You can 
# use the CSVM variable in your Makefile.in files with 
# @CSVM@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_CSVM],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([CSVM], [CSharp virtual machine])dnl
AC_CHECK_PROGS([CSVM], [mono$EXEEXT cs$EXEEXT])
ifelse([$1],,
[if test -z "$CSVM"; then
  AC_MSG_WARN([CSharp VM not found - continuing without CSharp VM])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# AC_PROG_TRY_CSVM
