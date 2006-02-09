#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_INIT([xflaim], [5.0], [xflaim-bug@novell.com])
AM_INIT_AUTOMAKE

AC_PREREQ(2.59)

AC_CONFIG_SRCDIR([src/xflaim.h])
AC_CONFIG_HEADER([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL

# Checks for libraries.
AC_CHECK_LIB([ncurses], [initscr])
AC_CHECK_LIB([rt], [aio_cancel])
AC_CHECK_LIB([dl], [dlopen])
AC_CHECK_LIB([m], [log10])
AC_CHECK_LIB([nsl], [gethostname])
AC_CHECK_LIB([pthread], [pthread_create])
AC_CHECK_LIB([socket], [socket])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([arpa/inet.h fcntl.h limits.h malloc.h netdb.h netinet/in.h stddef.h stdlib.h string.h strings.h sys/mount.h sys/socket.h sys/statvfs.h sys/time.h sys/vfs.h unistd.h utime.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_MODE_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_STRUCT_TM
AC_C_VOLATILE

# Checks for library functions.
AC_FUNC_LSTAT
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_FUNC_MKTIME
AC_FUNC_MMAP
AC_FUNC_STAT
AC_CHECK_FUNCS([atexit fdatasync ftruncate getcwd gethrtime gettimeofday localtime_r memmove memset mkdir munmap realpath rmdir strcasecmp strchr strerror strncasecmp strrchr strstr])

#
# Generic DEBUG support - default is OFF
#   ON means with symbols, no optimization, DEBUG is defined
#   OFF means with symbols, full optimization, NDEBUG is defined

AC_ARG_ENABLE(debug,
  AS_HELP_STRING([--enable-debug], [Turn on debugging (default is OFF)]),
  [case "${enableval}" in
    yes) debug=yes ;;
    no)  debug=no  ;;
    *)   AC_MSG_ERROR(bad value ${enableval} for --enable-debug) ;;
  esac], [debug=no])
AM_CONDITIONAL(DEBUG, test x$debug = xyes)
AM_CONDITIONAL(NDEBUG, test x$debug = xno)
if test x$debug = xyes; then
  AC_DEFINE(DEBUG, 1, [Defined for debug builds])
else
  AC_DEFINE(NDEBUG, 1, [Defined for non-debug builds])
fi

#
# Set C/C++ compiler-specific warning/optimization/option flags

CPPFLAGS="$CPPFLAGS -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64"

if test x$GCC = xyes; then
  # GCC: turn on all warnings, set warnings as errors
  #   disable optimization for debug builds
  #   enable high-level optimizations for non-debug builds
  CXXFLAGS="$CXXFLAGS -Wall -Werror"
  if test x$debug = xyes; then
    CXXFLAGS="$CXXFLAGS -O0"
  else
    CXXFLAGS="$CXXFLAGS -O3 -fgcse-sm"
  fi
elif $CC -V 2>&1 | grep "WorkShop Compilers"; then
  # Solaris Workshop: enable optimizations for debug builds
  CXXFLAGS="$CXXFLAGS -errwarn=%all -errtags -erroff=hidef,inllargeuse"
  if test x$debug = xno; then
    CXXFLAGS="$CXXFLAGS -fast"
    if echo $CC | grep "xarch=v9"; then
      # -fast sets -xarch=v8 disabling 64-bit mode, enable it again
      CXXFLAGS="$CXXFLAGS -xarch=v9"
    fi
  fi
elif echo $host_os | grep -i "osf" >/dev/null; then
  # OSF: enable standard C (shouldn't be necessary)
  #   enable optimizations for non-debug builds
  CXXFLAGS="$CXXFLAGS -std"
  if test x$debug = xno; then
    CXXFLAGS="$CXXFLAGS -O"
  fi
else
  # everything else: take a wild guess (shouldn't need this)
  if test x$debug = xno; then
    CXXFLAGS="$CXXFLAGS -O"
  fi
fi

AC_CONFIG_FILES([Makefile src/Makefile util/Makefile sample/Makefile])
AC_OUTPUT

