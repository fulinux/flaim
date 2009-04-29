# AC_PROG_TRY_JAR(["quiet"])
# --------------------------
# AC_PROG_TRY_JAR tests for an existing Java ARchive program.i
# It sets or uses the environment variable JAR.
#
# If no arguments are given to this macro, and no Java jar
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the JAR variable precious to Autoconf. You can 
# use the JAR variable in your Makefile.in files with 
# @JAR@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAR],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAR], [Java archive utility])dnl
AC_CHECK_PROGS([JAR], [jar$EXEEXT])
ifelse([$1],,
[if test -z "$JAR"; then
  AC_MSG_WARN([Java ARchive program not found - continuing without jar])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# AC_PROG_TRY_JAR
