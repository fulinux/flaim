dnl @synopsis AC_PROG_TRY_JAVAC
dnl
dnl AC_PROG_TRY_JAVAC looks for an existing Java compiler. It sets 
dnl and/or uses the environment variable JAVAC, then tests for 
dnl various known java compilers, beginning with free ones.
dnl
dnl You can use the JAVAC variable in your Makefile.in, with @JAVAC@.
dnl
dnl @category Java
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_JAVAC],[
AC_REQUIRE([AC_EXEEXT])dnl
if test -z "$JAVAPREFIX"; then
  test -z "$JAVAC" && AC_CHECK_PROGS([JAVAC], ["gcj$EXEEXT -C" guavac$EXEEXT jikes$EXEEXT javac$EXEEXT])
else
  test -z "$JAVAC" && AC_CHECK_PROGS([JAVAC], ["gcj$EXEEXT -C" guavac$EXEEXT jikes$EXEEXT javac$EXEEXT], [$JAVAPREFIX])
fi
if test -n "$JAVAC"; then
  AC_PROG_JAVAC_WORKS
fi])
