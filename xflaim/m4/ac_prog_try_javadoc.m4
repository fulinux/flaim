dnl @synopsis AC_PROG_TRY_JAVADOC
dnl
dnl AC_PROG_TRY_JAVADOC tests for an existing javadoc generator.
dnl It uses and/or sets the environment variable JAVADOC, then
dnl tests in sequence various common javadoc generator.
dnl
dnl You can use the JAVADOC variable in your Makefile.in, with
dnl @JAVADOC@.
dnl
dnl @category Java
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_JAVADOC],[
AC_REQUIRE([AC_EXEEXT])dnl
if test -z "$JAVAPREFIX"; then
  test -z "$JAVADOC" && AC_CHECK_PROGS([JAVADOC], [javadoc$EXEEXT])
else
  test -z "$JAVADOC" && AC_CHECK_PROGS([JAVADOC], [javadoc$EXEEXT], [$JAVAPREFIX])
fi])
