ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = README.W32 tools win32

SUBDIRS = ftk flaim sql xflaim

rpms srcrpm:
	for dir in $(SUBDIRS); do \
 	  $(MAKE) -C $$dir $@; \
	done

dist-hook:
	-rm -rf `find $(distdir) -name .svn`

.PHONY: rpms srcrpm

