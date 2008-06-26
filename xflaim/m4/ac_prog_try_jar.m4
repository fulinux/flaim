dnl @synopsis AC_PROG_TRY_JAR
dnl
dnl AC_PROG_TRY_JAR tests for an existing jar program. It sets and/or
dnl uses the environment variable JAR then tests in sequence various
dnl common jar programs.
dnl
dnl You can use the JAR variable in your Makefile.in, with @JAR@.
dnl
dnl @category Java
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_JAR],[
AC_REQUIRE([AC_EXEEXT])dnl
if test -z "$JAVAPREFIX"; then
  test -z "$JAR" && AC_CHECK_PROGS([JAR], [jar$EXEEXT])
else
  test -z "$JAR" && AC_CHECK_PROGS([JAR], [jar$EXEEXT], [$JAVAPREFIX])
fi])
