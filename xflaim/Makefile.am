ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = GNUMakefile INSTALL.W32 debian netware

SUBDIRS = $(subdirs) src util sample docs obs

doc_DATA = AUTHORS ChangeLog COPYING INSTALL NEWS README

rpms srcrpm: dist
	$(MAKE) -C obs $(AM_MAKEFLAGS) $@
	rpmarch=`rpm --showrc | grep ^build\ arch | sed 's/\(.*: \)\(.*\)/\2/'`; \
	test -z $$rpmarch || ( mv $$rpmarch/* .; rm -rf $$rpmarch )
	-rm -rf $(distdir)

dist-hook:
	-rm -rf `find $(distdir) -name .svn`

.PHONY: srcrpm rpms

