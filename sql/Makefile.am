ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = GNUMakefile README.W32 win32

SUBDIRS = src obs

doc_DATA = AUTHORS ChangeLog COPYING INSTALL NEWS README

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libflaimsql.pc

RPM = rpm

rpms srcrpm: dist
	cd obj && $(MAKE) $(AM_MAKEFLAGS) $@
	rpmarch=`$(RPM) --showrc | grep "^build arch" | sed 's/\(.*: \)\(.*\)/\2/'`; \
	test -z "$$rpmarch" || ( mv $$rpmarch/* .; rm -rf $$rpmarch )
	rm -rf $(distdir)

dist-hook:
	rm -rf `find $(distdir) -name .svn`

.PHONY: srcrpm rpms
