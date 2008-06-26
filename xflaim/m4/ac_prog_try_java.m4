dnl @synopsis AC_PROG_TRY_JAVA
dnl
dnl AC_PROG_TRY_JAVA looks for an existing JAVA virtual machine. It 
dnl sets and/or uses the environment variable JAVA, then tests for
dnl various Java virtual machines, beginning with the free ones.
dnl
dnl If and when a JVM is located, it's then tested via 
dnl AC_PROG_JAVA_WORKS.
dnl
dnl You can use the JAVA variable in your Makefile.in, with @JAVA@.
dnl
dnl @category Java
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_JAVA],[
AC_REQUIRE([AC_EXEEXT])dnl
if test -z "$JAVAPREFIX"; then
  test -z "$JAVA" && AC_CHECK_PROGS([JAVA], [kaffe$EXEEXT java$EXEEXT])
else
  test -z "$JAVA" && AC_CHECK_PROGS([JAVA], [kaffe$EXEEXT java$EXEEXT], [$JAVAPREFIX])
fi
if test -n "$JAVA"; then
  AC_PROG_JAVA_WORKS
fi])
