dnl @synopsis AC_PROG_TRY_JAVAH
dnl
dnl AC_PROG_TRY_JAVAH looks for an existing Java native header (JNI)
dnl generator. It sets and/or uses the environment variable JAVAH, 
dnl then tests for the javah utility, beginning with free ones.
dnl
dnl You can use the JAVAH variable in your Makefile.in, with @JAVAH@.
dnl
dnl @category Java
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-23
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_TRY_JAVAH],[
AC_REQUIRE([AC_CANONICAL_SYSTEM])dnl
AC_REQUIRE([AC_PROG_CPP])dnl
AC_PATH_PROG(JAVAH,javah)
if test x"`eval 'echo $ac_cv_path_JAVAH'`" != x ; then
  AC_TRY_CPP([#include <jni.h>],,[
    ac_save_CPPFLAGS="$CPPFLAGS"
changequote(, )dnl
    ac_dir=`echo $ac_cv_path_JAVAH | sed 's,\(.*\)/[^/]*/[^/]*$,\1/include,'`
    ac_machdep=`echo $build_os | sed 's,[-0-9].*,,' | sed 's,cygwin,win32,'`
changequote([, ])dnl
    CPPFLAGS="$ac_save_CPPFLAGS -I$ac_dir -I$ac_dir/$ac_machdep"
    AC_TRY_CPP([#include <jni.h>],
               ac_save_CPPFLAGS="$CPPFLAGS",
               AC_MSG_WARN([unable to include <jni.h>]))
    CPPFLAGS="$ac_save_CPPFLAGS"])
fi])
