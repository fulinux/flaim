# AC_PROG_TRY_JAVA(["quiet"])
# ---------------------------
# AC_PROG_TRY_JAVA test for an existing JAVA virtual machine.
# It uses or sets the environment variable JAVA.
#
# If no arguments are given to this macro, and no java virtual
# machine can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the JAVA variable precious to Autoconf. You can 
# use the JAVA variable in your Makefile.in files with 
# @JAVA@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVA],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVA], [Java virtual machine])dnl
AC_CHECK_PROGS([JAVA], [kaffe$EXEEXT java$EXEEXT])
ifelse([$1],,
[if test -z "$JAVA"; then
  AC_MSG_WARN([Java VM not found - continuing without JVM])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])
