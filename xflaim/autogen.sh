#!/bin/sh
# Run this to generate all the initial makefiles, etc.

srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`
cd $srcdir
PROJECT=xflaim
TEST_TYPE=-f
FILE=src/xflaim.h

DIE=0

(autoconf --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have autoconf installed to compile $PROJECT."
	echo "Download the appropriate package for your distribution,"
	echo "or get the source tarball at ftp://ftp.gnu.org/pub/gnu/"
	DIE=1
}

(automake --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have automake installed to compile $PROJECT."
	echo "Download the appropriate package for your distribution,"
	echo "or get the source tarball at ftp://ftp.gnu.org/pub/gnu/"
	DIE=1
}

if test "$DIE" -eq 1; then
	exit 1
fi

test $TEST_TYPE $FILE || {
	echo "You must run this script in the top-level $PROJECT directory"
	exit 1
}

aclocal $ACLOCAL_FLAGS
echo "aclocal done"
libtoolize --force --copy
echo "libtoolize done"
autoheader
echo "autoheader done"
automake --add-missing --copy $am_opt
echo "automake done"
autoconf
echo "autoconf done"
cd $ORIGDIR

echo 
echo "Now type 'configure' and 'make' to compile $PROJECT. You can do this" 
echo "in a separate build directory if you wish"
