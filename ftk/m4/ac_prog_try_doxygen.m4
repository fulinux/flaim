# AC_PROG_TRY_DOXYGEN(["quiet"])
# ------------------------------
# AC_PROG_TRY_DOXYGEN tests for an existing doxygen source
# documentation program. It sets or uses the environment 
# variable DOXYGEN. 
#
# If no arguments are given to this macro, and no doxygen 
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the DOXYGEN variable precious to Autoconf. You can 
# use the DOXYGEN variable in your Makefile.in files with 
# @DOXYGEN@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_DOXYGEN],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([DOXYGEN], [Doxygen source doc generation program])dnl
AC_CHECK_PROGS([DOXYGEN], [doxygen$EXEEXT])
ifelse([$1],,
[if test -z "$DOXYGEN"; then
  AC_MSG_WARN([Doxygen program not found - continuing without Doxygen])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])
