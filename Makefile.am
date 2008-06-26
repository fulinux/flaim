ACLOCAL_AMFLAGS = -I m4

SUBDIRS = ftk flaim sql xflaim

rpms srcrpm:
	for dir in $(SUBDIRS); do \
 	  $(MAKE) -C $$dir $@; \
	done

.PHONY: rpms srcrpm

