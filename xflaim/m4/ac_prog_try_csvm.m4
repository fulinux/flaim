dnl @synopsis AC_PROG_TRY_CSVM
dnl
dnl AC_PROG_TRY_CSVM looks for an existing CSharp virtual machine. 
dnl It sets and/or uses the environment variable CSC, then tests 
dnl for the Mono CSharp compiler.
dnl
dnl If and when a CSVM is located, it's then tested via 
dnl AC_PROG_CSVM_WORKS.
dnl
dnl You can use the CSVM variable in your Makefile.in, with @CSVM@.
dnl
dnl @category CSharp
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_CSVM],[
AC_REQUIRE([AC_EXEEXT])dnl
test -z "$CSVM" && AC_CHECK_PROGS([CSVM], [mono$EXEEXT cs$EXEEXT])
if test -n "$CSVM"; then
  AC_PROG_CSVM_WORKS
fi])
