ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = GNUMakefile README.W32 debian netware win32

SUBDIRS = src util sample docs obs

doc_DATA = AUTHORS ChangeLog COPYING INSTALL NEWS README

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libxflaim.pc

RPM = rpm

rpms srcrpm: dist
	cd obs && $(MAKE) $(AM_MAKEFLAGS) $@
	rpmarch=`$(RPM) --showrc | grep "^build arch" | sed 's/\(.*: \)\(.*\)/\2/'`; \
	test -z "$$rpmarch" || ( mv $$rpmarch/* .; rm -rf $$rpmarch )
	rm -rf $(distdir)

dist-hook:
	rm -rf `find $(distdir) -name .svn`

.PHONY: srcrpm rpms
