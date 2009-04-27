# AC_PROG_TRY_CSC(["quiet"])
# --------------------------
# AC_PROG_TRY_CSC tests for an existing CSharp compiler. It sets
# or uses the environment variable CSC.
#
# It checks for a Mono CSharp compiler (msc) and then for a 
# Microsoft CSharp compiler (csc).
#
# If no arguments are given to this macro, and no CSharp
# compiler can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed.
#
# Makes the CSC variable precious to Autoconf. You can 
# use the CSC variable in your Makefile.in files with 
# @CSC@.
#
# NOTE: Currently, passing any value in the first argument has 
#       the same effect as passing "quiet", however, you should
#       not rely on this, as all other words are reserved.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-27
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_CSC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([CSC], [CSharp compiler])dnl
AC_CHECK_PROGS([CSC], [mcs$EXEEXT csc$EXEEXT])
ifelse([$1],,
[if test -z "$CSC"; then
  AC_MSG_WARN([CSharp compiler not found - continuing without CSharp])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])
