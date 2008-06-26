EXTRA_DIST = ha_flaim.cpp ha_flaim.h rflread.cpp rflread.h viewrfl.cpp

## Utility Programs

bin_PROGRAMS = flmcheckdb flmrebuild flmview flmdbshell flmgigatest

flmcheckdb_SOURCES = checkdb.cpp
flmcheckdb_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)
flmcheckdb_LDADD = libutil.la ../src/libflaim.la

flmrebuild_SOURCES = rebuild.cpp
flmrebuild_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)
flmrebuild_LDADD = libutil.la ../src/libflaim.la

flmview_SOURCES = \
 flmarg.h \
 viewblk.cpp \
 view.cpp \
 viewdisp.cpp \
 viewedit.cpp \
 viewfhdr.cpp \
 view.h \
 viewlfil.cpp \
 viewlhdr.cpp \
 viewmenu.cpp \
 viewsrch.cpp
flmview_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)
flmview_LDADD = libutil.la ../src/libflaim.la

flmdbshell_SOURCES = \
 dbshell.cpp \
 dbshell.h \
 flm_edit.cpp \
 flm_edit.h
flmdbshell_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)
flmdbshell_LDADD = libutil.la ../src/libflaim.la

flmgigatest_SOURCES = gigatest.cpp gigatest.h
flmgigatest_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)
flmgigatest_LDADD = libutil.la ../src/libflaim.la

## Utility Convenience Library

noinst_LTLIBRARIES = libutil.la

libutil_la_SOURCES = \
 flm_dlst.cpp \
 flm_dlst.h \
 flm_lutl.cpp \
 flm_lutl.h \
 sharutil.cpp \
 sharutil.h
libutil_la_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)

## Check Programs

check_PROGRAMS = ut_basictest
check_HEADERS = flmunittest.h

ut_basictest_SOURCES = basic_test.cpp flmunittest.cpp
ut_basictest_CPPFLAGS = -I$(top_srcdir)/src $(FTK_INCLUDE)
ut_basictest_LDADD = libutil.la ../src/libflaim.la

## Unit Tests

TESTS = ut_basictest

clean-local:
	-rm -rf samplebackup samplerestore.*
