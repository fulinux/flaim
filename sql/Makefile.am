ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = GNUMakefile README.W32 win32

SUBDIRS = $(subdirs) src obs

doc_DATA = AUTHORS ChangeLog COPYING INSTALL NEWS README

rpms srcrpm: dist
	$(MAKE) -C obs $(AM_MAKEFLAGS) $@
	rpmarch=`rpm --showrc | grep ^build\ arch | sed 's/\(.*: \)\(.*\)/\2/'`; \
	test -z $$rpmarch || ( mv $$rpmarch/* .; rm -rf $$rpmarch )
	-rm -rf $(distdir)

dist-hook:
	-rm -rf `find $(distdir) -name .svn`

.PHONY: srcrpm rpms

