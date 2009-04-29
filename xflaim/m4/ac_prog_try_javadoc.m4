# AC_PROG_TRY_JAVADOC(["quiet"])
# ------------------------------
# AC_PROG_TRY_JAVADOC tests for an existing javadoc generator.
# It uses or sets the environment variable JAVADOC.
#
# If no arguments are given to this macro, and no javadoc 
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the JAVADOC variable precious to Autoconf. You can 
# use the JAVADOC variable in your Makefile.in files with 
# @JAVADOC@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVADOC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVADOC], [Java source documentation utility])dnl
AC_CHECK_PROGS([JAVADOC], [javadoc$EXEEXT])
ifelse([$1],,
[if test -z "$JAVADOC"; then
  AC_MSG_WARN([Javadoc program not found - continuing without javadoc])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# AC_PROG_TRY_JAVADOC
