dnl @synopsis AC_PROG_CSC_WORKS
dnl
dnl Internal use ONLY.
dnl
dnl Note: This is part of the set of autoconf M4 macros for CSharp
dnl programs. It is VERY IMPORTANT that you download the whole set,
dnl some macros depend on other.
dnl
dnl @category CSharp
dnl @author John Calcote <john.calcote@gmail.com>
dnl @version 2008-06-24
dnl @license GPLWithACException

AC_DEFUN([AC_PROG_CSC_WORKS],[
AC_CACHE_CHECK([if $CSC works], ac_cv_prog_csc_works, [
CSC_TEST=test.cs
TEST_EXE=test.exe
cat << \EOF > $CSC_TEST
/* [#]line __oline__ "configure" */
public class Test { static void Main() {} }
EOF
if AC_TRY_COMMAND([$CSC $CSCFLAGS $CSC_TEST]) >/dev/null 2>&1; then
  ac_cv_prog_csc_works=yes
else
  AC_MSG_ERROR([The CSharp compiler $CSC failed (see config.log)])
  echo "configure: failed program was:" >&AC_FD_CC
  cat $CSC_TEST >&AC_FD_CC
fi
rm -f $CSC_TEST $TEST_EXE
])
AC_PROVIDE([$0])dnl
])
