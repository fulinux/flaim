# AC_PROG_TRY_JAVAH(["quiet"])
# ----------------------------
# AC_PROG_TRY_JAVAH tests for an existing Java native header (JNI)
# generator. It uses or sets the environment variable JAVAH.
#
# If no arguments are given to this macro, and no javah
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the JAVAH variable precious to Autoconf. You can 
# use the JAVAH variable in your Makefile.in files with 
# @JAVAH@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-23
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVAH],
[AC_REQUIRE([AC_CANONICAL_SYSTEM])dnl
AC_REQUIRE([AC_PROG_CPP])dnl
AC_ARG_VAR([JAVAH], [Java header utility])dnl
AC_CHECK_PROGS([JAVAH], [javah])
m4_ifvaln([$1],,
[if test -z "$DOXYGEN"; then
  AC_MSG_WARN([No javah utility found - continuing wihtout javah support])
fi])dnl
])
