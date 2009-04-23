# AC_PROG_TRY_JAVADOC(["quiet"])
# ------------------------------
# AC_PROG_TRY_JAVADOC tests for an existing javadoc generator.
# It uses or sets the environment variable JAVADOC.
#
# If no arguments are given to this macro, and no javadoc 
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the JAVADOC variable precious to Autoconf. You can 
# use the JAVADOC variable in your Makefile.in files with 
# @JAVADOC@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-23
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVADOC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVADOC], [Java source documentation utility])dnl
AC_CHECK_PROGS([JAVADOC], [javadoc$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$JAVADOC"; then
  AC_MSG_WARN([No javadoc program found - continuing without javadoc support])
fi])dnl
])
