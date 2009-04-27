# AC_PROG_TRY_JAVAC(["quiet"])
# ----------------------------
# AC_PROG_TRY_JAVAC tests for an existing Java compiler. It uses
# or sets the environment variable JAVAC.
#
# If no arguments are given to this macro, and no Java
# compiler can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the JAVAC variable precious to Autoconf. You can 
# use the JAVAC variable in your Makefile.in files with 
# @JAVAC@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_JAVAC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([JAVAC], [Java compiler])dnl
AC_CHECK_PROGS([JAVAC], ["gcj$EXEEXT -C" guavac$EXEEXT jikes$EXEEXT javac$EXEEXT])
ifelse([$1],,
[if test -z "$JAVAC"; then
  AC_MSG_WARN([Java compiler not found - continuing without javac])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])
