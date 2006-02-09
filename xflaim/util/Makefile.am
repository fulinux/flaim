noinst_LTLIBRARIES = libutilsup.la
libutilsup_la_SOURCES = flm_dlst.cpp flm_lutl.cpp ftx.cpp sharutil.cpp
libutilsup_la_CPPFLAGS = -I$(top_srcdir)/src
libutilsup_la_LDFLAGS = -static

bin_PROGRAMS = checkdb rebuild view xshell

checkdb_SOURCES = checkdb.cpp
checkdb_CPPFLAGS = -I$(top_srcdir)/src
checkdb_LDADD = ../src/libxflaim.la libutilsup.la

rebuild_SOURCES = rebuild.cpp
rebuild_CPPFLAGS = -I$(top_srcdir)/src
rebuild_LDADD = ../src/libxflaim.la libutilsup.la

view_SOURCES = \
  view.cpp \
  viewblk.cpp \
  viewdisp.cpp \
  viewedit.cpp \
  viewhdr.cpp \
  viewlfil.cpp \
  viewmenu.cpp \
  viewsrch.cpp
view_CPPFLAGS = -I$(top_srcdir)/src
view_LDADD = ../src/libxflaim.la libutilsup.la

xshell_SOURCES = fdomedt.cpp xshell.cpp fshell.cpp
xshell_CPPFLAGS = -I$(top_srcdir)/src
xshell_LDADD = ../src/libxflaim.la libutilsup.la

