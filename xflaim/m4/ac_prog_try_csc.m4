dnl @synopsis AC_PROG_TRY_CSC
dnl
dnl AC_PROG_TRY_CSC looks for an existing CSharp compiler. It sets
dnl and/or uses the environment variable CSC, then tests for the 
dnl Mono CSharp compiler.
dnl
dnl You can use the CSC variable in your Makefile.in, with @CSC@.
dnl
dnl @category CSharp
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_CSC],[
AC_REQUIRE([AC_EXEEXT])dnl
test -z "$CSC" && AC_CHECK_PROGS([CSC], [mcs$EXEEXT csc$EXEEXT])
if test -n "$CSC"; then
  AC_PROG_CSC_WORKS
fi])
