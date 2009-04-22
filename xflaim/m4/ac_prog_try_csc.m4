# AC_PROG_TRY_CSC([quiet])
# ------------------------
# AC_PROG_TRY_CSC looks for an existing CSharp compiler. If the 
# CSC variable is empty, it checks for a Mono CSharp compiler
# and then for a Microsoft CSharp compiler.
#
# If no arguments are given to this macro, and no CSharp compiler
# can be found, it prints a very visible message to STDOUT and to
# the config.log file. If the "quiet" argument is passed, then 
# only the normal "check" line is displayed. (Technically, any 
# passing any value in the first argument has the same effect as
# passing "quiet".)
#
# Makes CSC precious to Autoconf. You can use CSC variable in
# your Makefile.in with @CSC@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-04-22
# License:  AllPermissive
#
AC_DEFUN([AC_PROG_TRY_CSC],
[AC_REQUIRE([AC_EXEEXT])dnl
AC_ARG_VAR([CSC], [CSharp compiler])dnl
AC_CHECK_PROGS([CSC], [mcs$EXEEXT csc$EXEEXT])
m4_ifvaln([$1],,
[if test -z "$CSC"; then
  AC_MSG_WARN([
  -----------------------------------------
   No CSharp compiler found - continuing
   without CSHARP compiler support.
  -----------------------------------------])
fi])dnl
])
