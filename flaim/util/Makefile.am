noinst_LTLIBRARIES = libutilsup.la
libutilsup_la_SOURCES = \
  flm_dlst.cpp \
  flm_lutl.cpp \
  ftx.cpp \
  ftxunix.cpp \
  sharutil.cpp \
  wpscrnkb.cpp
libutilsup_la_CPPFLAGS = -I$(top_srcdir)/src
libutilsup_la_LDFLAGS = -static

bin_PROGRAMS = checkdb rebuild view sample dbshell

checkdb_SOURCES = checkdb.cpp
checkdb_CPPFLAGS = -I$(top_srcdir)/src
checkdb_LDADD = ../src/libflaim.la libutilsup.la

rebuild_SOURCES = rebuild.cpp
rebuild_CPPFLAGS = -I$(top_srcdir)/src
rebuild_LDADD = ../src/libflaim.la libutilsup.la

view_SOURCES = \
  view.cpp \
  viewblk.cpp \
  viewdisp.cpp \
  viewedit.cpp \
  viewfhdr.cpp \
  viewlhdr.cpp \
  viewlfil.cpp \
  viewmenu.cpp \
  viewsrch.cpp
view_CPPFLAGS = -I$(top_srcdir)/src
view_LDADD = ../src/libflaim.la libutilsup.la

sample_SOURCES = sample.cpp
sample_CPPFLAGS = -I$(top_srcdir)/src
sample_LDADD = ../src/libflaim.la libutilsup.la

dbshell_SOURCES = dbshell.cpp flm_edit.cpp
dbshell_CPPFLAGS = -I$(top_srcdir)/src
dbshell_LDADD = ../src/libflaim.la libutilsup.la

