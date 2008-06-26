#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.62])
AC_INIT([SQLFLAIM], [6.0], [flaim-users@forge.novell.com])
AM_INIT_AUTOMAKE([-Wall -Werror])
LT_PREREQ([2.2])
LT_INIT([dlopen])

AC_LANG(C++)

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/flaimsql.h])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_INSTALL

# Checks for optional programs.
AC_PROG_TRY_DOXYGEN

# Configure FTKLIB and FTKINC variables.
AC_ARG_VAR([FTKLIB], [The PATH wherein libflaimtk.la can be found.])
AC_ARG_VAR([FTKINC], [The PATH wherein flaimtk.h can be found.])

# Ensure that both or neither FTK paths were specified.
if (test -n "$FTKLIB" && test -z "$FTKINC") || \
   (test -n "$FTKINC" && test -z "$FTKLIB"); then
  AC_MSG_ERROR([Specify both FTKINC and FTKLIB, or neither.])
fi 

# Not specified? Check for FTK in standard places.
if test -z "$FTKLIB"; then
  # Check for flaim tool kit as a sub-project.
  if test -d "$srcdir/ftk"; then
    AC_CONFIG_SUBDIRS([ftk])
    FTKINC='$(top_srcdir)/ftk/src'
    FTKLIB='$(top_builddir)/ftk/src'
  else
    # Check for flaim tool kit as a super-project.
    if test -d "$srcdir/../ftk"; then
      FTKINC='$(top_srcdir)/../ftk/src'
      FTKLIB='$(top_builddir)/../ftk/src'
    fi
  fi
fi

# Still empty? Check for *installed* flaim tool kit.
if test -z "$FTKLIB"; then
  AC_CHECK_LIB([flaimtk], [ftkFastChecksum], 
    [AC_CHECK_HEADERS([flaimtk.h])
     LIBS="-lflaimtk $LIBS"],
    [AC_MSG_ERROR([No FLAIM Toolkit found. Terminating.])])
fi

# AC_SUBST command line variables from FTKLIB and FTKINC.
if test -n "$FTKLIB"; then
  AC_SUBST([FTK_LTLIB], ["$FTKLIB/libflaimtk.la"])
  AC_SUBST([FTK_INCLUDE], ["-I$FTKINC"])
fi

# Configure options: --enable-debug[=no].
AC_ARG_ENABLE([debug],
  [AS_HELP_STRING([--enable-debug],
    [enable debug code (default is no)])],
  [debug="$withval"], [debug=no])

# Check for doxygen program.
if test -z "$DOXYGEN"; then
  echo "-----------------------------------------"
  echo " No Doxygen program found - continuing"
  echo " without Doxygen documentation support."
  echo "-----------------------------------------"
fi
AM_CONDITIONAL([HAVE_DOXYGEN], [test "x$DOXYGEN" != x])

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Checks for library functions.

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

# Configure DEBUG source code, if requested.
if test "x$debug" = xyes; then
  AC_DEFINE([FLM_DEBUG], [], [Define to enable FLAIM debug features])
fi

# Configure global pre-processor definitions.
AC_DEFINE([_REENTRANT], [], [Define for reentrant code])
AC_DEFINE([_LARGEFILE64_SOURCE], [], [Define for 64-bit data files])
AC_DEFINE([_LARGEFILE_SOURCE], [], [Define for 64-bit data files])

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
      CXXFLAGS="$CXXFLAGS -qthreaded -qstrict -q$build_bits"
      LDFLAGS="$LDFLAGS -q$build_bits -X$build_bits"
    fi ;;

  *-*-hpux*)
    if "x$CXX" != "xg++"; then
      # Disable "Placement operator delete invocation is not yet implemented" warning
      CXXFLAGS="$CXXFLAGS +W930"
    fi ;;
esac

AC_CONFIG_FILES([Makefile
                 obs/sqlflaim.spec
                 obs/Makefile
                 src/Makefile])
AC_OUTPUT

echo "
  FLAIM SQL Database Library ($PACKAGE_NAME) version $PACKAGE_VERSION
  Prefix.........: $prefix
  Debug Build....: $debug
  C++ Compiler...: $CXX $CXXFLAGS $CPPFLAGS
  Linker.........: $LD $LDFLAGS $LIBS
  FTK Library....: ${FTKLIB:-INSTALLED}
  FTK Include....: ${FTKINC:-INSTALLED}
  Doxygen........: ${DOXYGEN:-NONE}
"

