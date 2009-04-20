#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.62])
AC_INIT([XFLAIM],[5.2],[flaim-users@forge.novell.com])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([-Wall -Werror])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_LANG(C++)

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/xflaim.h])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_LN_S
AC_PROG_INSTALL

# Checks for optional programs.
AC_PROG_TRY_CSC
AC_PROG_TRY_CSVM
AC_PROG_TRY_JAVAC
AC_PROG_TRY_JAVAH
AC_PROG_TRY_JAVADOC
AC_PROG_TRY_JAR
AC_PROG_TRY_DOXYGEN

# Configure FTKLIB, FTKINC, FTK_LTLIB and FTK_INCLUDE
FLM_FTK_SEARCH

# Check for Java compiler.
have_java=yes 
if test -z "$JAVAC"; then have_java=no; fi
if test -z "$JAVAH"; then have_java=no; fi
if test -z "$JAR"; then have_java=no; fi
if test "x$have_java" = xno; then
  echo "-----------------------------------------" >&AS_MESSAGE_LOG_FD
  echo " Some Java tools not found - continuing" >&AS_MESSAGE_LOG_FD
  echo " without XFLAIM JNI support." >&AS_MESSAGE_LOG_FD
  echo "-----------------------------------------" >&AS_MESSAGE_LOG_FD
fi
AM_CONDITIONAL([HAVE_JAVA], [test "x$have_java" = xyes])

# Check for CSharp compiler.
if test -z "$CSC"; then
  echo "-----------------------------------------" >&AS_MESSAGE_LOG_FD
  echo " No CSharp compiler found - continuing" >&AS_MESSAGE_LOG_FD
  echo " without XFLAIM CSHARP support." >&AS_MESSAGE_LOG_FD
  echo "-----------------------------------------" >&AS_MESSAGE_LOG_FD
fi
AM_CONDITIONAL([HAVE_CSHARP], [test -n "$CSC"])

# Check for doxygen program.
if test -z "$DOXYGEN"; then
  echo "-----------------------------------------" >&AS_MESSAGE_LOG_FD
  echo " No Doxygen program found - continuing" >&AS_MESSAGE_LOG_FD
  echo " without Doxygen documentation support." >&AS_MESSAGE_LOG_FD
  echo "-----------------------------------------" >&AS_MESSAGE_LOG_FD
fi
AM_CONDITIONAL([HAVE_DOXYGEN], [test "x$DOXYGEN" != x])

# Check for pthreads
ACX_PTHREAD(
 [AC_DEFINE([HAVE_PTHREAD],[1],[Define if you have POSIX threads libraries and header files.])
  LIBS="$PTHREAD_LIBS $LIBS"
  CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
  CXXFLAGS="$CXXFLAGS $PTHREAD_CXXFLAGS"
  CC="$PTHREAD_CC"
  CXX="$PTHREAD_CXX"])

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([atexit strerror])

# Configure for large files, even on 32-bit systems
AC_SYS_LARGEFILE

# Configure DEBUG source code, if requested.
if test "x$debug" = xyes; then
  AC_DEFINE([FLM_DEBUG], [], [Define to enable FLAIM debug features])
fi

# Configure compiler and platform flags and options.
# This stuff was in the original makefile...
# Personally, I believe -O2 does as good a job as 
# a hand-coded set of optimization flags...
gcc_opt_flags="-O -foptimize-sibling-calls
 -fstrength-reduce -fcse-follow-jumps -fcse-skip-blocks
 -frerun-cse-after-loop -frerun-loop-opt -fgcse
 -fgcse-lm -fgcse-sm -fdelete-null-pointer-checks
 -fexpensive-optimizations -fregmove -fsched-interblock
 -fsched-spec -fcaller-saves -fpeephole2 -freorder-blocks
 -freorder-functions -falign-functions -falign-jumps
 -falign-loops -falign-labels -fcrossjumping"

# Enable g++ specific compiler flags.
# Currently, enabling all warnings displays the type-pun error dozens
# of times, and of course enabling warnings as werrors breaks the build
#if test "x$CXX" = "xg++"; then
#  CXXFLAGS="$CXXFLAGS -Wall -Werror"
#  CXXFLAGS="$CXXFLAGS $gcc_opt_flags"
#fi

# Configure supported platforms' compiler and linker flags.
case $host in
  sparc-*-solaris*)
    LDFLAGS="$LDFLAGS -R /usr/lib/lwp"
    if "x$CXX" != "xg++"; then
      if "x$debug" = xno; then
        CXXFLAGS="$CXXFLAGS -xO3"
      fi
      SUN_STUDIO_COMPILER=`"$CXX" -V | grep "Sun C++"`
      if "x$SUN_STUDIO_COMPILER" = "xSun C++"; then
        CXXFLAGS="$CXXFLAGS -errwarn=%all -errtags -erroff=hidef,inllargeuse,doubunder"
      fi
    fi ;;

  *-apple-darwin*)
    AC_DEFINE([OSX], [], 
      [Define if building on Apple OSX.]) ;;

  *-*-aix*)
    if "x$CXX" != "xg++"; then
      CXXFLAGS="$CXXFLAGS -qstrict"
    fi ;;

  *-*-hpux*)
    if "x$CXX" != "xg++"; then
      # Disable "Placement operator delete invocation is not yet implemented" warning
      CXXFLAGS="$CXXFLAGS +W930"
    fi ;;
esac

AC_CONFIG_FILES([Makefile
                 docs/Makefile
                 docs/doxygen/Makefile
                 docs/doxygen/doxyfile
                 obs/Makefile
                 obs/xflaim.spec
                 sample/Makefile
                 src/Makefile
                 src/cs/Makefile
                 src/cs/wrapper/Makefile
                 src/java/Makefile
                 src/java/wrapper/Makefile
                 util/Makefile
                 libxflaim.pc])
AC_OUTPUT

echo "
  XFLAIM XML Database Library ($PACKAGE_NAME) version $PACKAGE_VERSION
  Prefix.........: $prefix
  Debug Build....: $debug
  C++ Compiler...: $CXX $CXXFLAGS $CPPFLAGS
  Linker.........: $LD $LDFLAGS $LIBS
  FTK Library....: ${FTKLIB:-INSTALLED}
  FTK Include....: ${FTKINC:-INSTALLED}
  CSharp Compiler: ${CSC:-NONE} $CSCFLAGS
  CSharp VM......: ${CSVM:-NONE}
  Java Compiler..: ${JAVAC:-NONE} $JAVACFLAGS
  JavaH Utility..: ${JAVAH:-NONE} $JAVAHFLAGS
  Jar Utility....: ${JAR:-NONE} $JARFLAGS 
  Javadoc Utility: ${JAVADOC:-NONE}
  Doxygen........: ${DOXYGEN:-NONE}
"
