#-------------------------------------------------------------------------
# Desc:	Platform identification and configuration
# Tabs:	3
#
#		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
#
#		This program is free software; you can redistribute it and/or
#		modify it under the terms of version 2 of the GNU General Public
#		License as published by the Free Software Foundation.
#
#		This program is distributed in the hope that it will be useful,
#		but WITHOUT ANY WARRANTY; without even the implied warranty of
#		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#		GNU General Public License for more details.
#
#		You should have received a copy of the GNU General Public License
#		along with this program; if not, contact Novell, Inc.
#
#		To contact Novell about this file by physical or electronic mail,
#		you may find current contact information at www.novell.com
#
# $Id: $
#-------------------------------------------------------------------------

# -- Subversion Revision --

calc_svn_revision =

ifneq (,$(findstring ignore-local-mods,$(MAKECMDGOALS)))
	submake_targets += ignore-local-mods
	ignore_local_mods = 1
endif

ifneq (,$(findstring ilm,$(MAKECMDGOALS)))
	submake_targets += ilm
	ignore_local_mods = 1
endif

ifdef ignore_local_mods
	local_mods_ok = 1
else
	local_mods_ok =
endif

ifneq (,$(findstring dist,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring rpm,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring pathinfo,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring changelog,$(MAKECMDGOALS)))

	calc_svn_revision = 1

	# Get the info for this directory

	ifndef svn_user
      $(error Must define svn_user=<user> in environment or as a parameter)
	endif

	ifndef svn_rev
      $(error Must define svn_rev=<low[:high]> in environment or as a parameter)
	endif

	svnrevs = $(subst :, ,$(svn_rev))
	svn_low_rev = $(word 1,$(svnrevs))
	svn_high_rev = $(word 2,$(svnrevs))

	svnurl0 := $(shell svn info)
	svnurl1 = $(subst URL: ,URL:,$(svnurl0))
	svnurl2 = $(filter URL:%,$(svnurl1))
	svnurl3 = $(subst URL:,,$(svnurl2))
	svnurl = $(subst ://,://$(svn_user)@,$(svnurl3))
endif

ifdef calc_svn_revision

	# Get the info for all files.

	ifndef local_mods_ok
		srevision := $(shell svnversion . -n)

		ifneq (,$(findstring M,$(srevision)))
         $(error Local modifications found - please check in before making distro)
		endif

		ifneq (,$(findstring :,$(srevision)))
         $(error Mixed revisions in repository - please update before making distro)
		endif
	endif

	numdigits = $(words $(subst 9,9 ,$(subst 8,8 ,$(subst 7,7 ,\
						$(subst 6,6 ,$(subst 5,5 ,$(subst 4,4 ,$(subst 3,3 ,\
						$(subst 2,2 ,$(subst 1,1 ,$(subst 0,0 ,$(1))))))))))))
	revision0 := $(shell svn info -R)
	revision1 = $(subst Last Changed Rev: ,LastChangedRev:,$(revision0))
	revision2 = $(filter LastChangedRev:%,$(revision1))
	revision3 = $(subst LastChangedRev:,,$(revision2))
	revision4 = $(sort $(revision3))
	revision5 = $(foreach num,$(revision4),$(call numdigits,$(num)):$(num))
	revision6 = $(sort $(revision5))
	revision7 = $(word $(words $(revision6)),$(revision6))
	svn_revision = $(word 2,$(subst :, ,$(revision7)))

else
	ifeq "$(wildcard SVNRevision.*)" ""
		svn_revision = 0
	else
		svn_revision = $(word 2,$(subst ., ,$(wildcard SVNRevision.*)))
	endif
endif

ifeq "$(svn_high_rev)" ""
	svn_high_rev = $(svn_revision)
endif

# -- Paths initializations --

ifndef rpm_build_root
	ifneq (,$(DESTDIR))
		rpm_build_root = $(DESTDIR)/
	else
		rpm_build_root =
	endif
endif

# -- Target variables --

target_build_type =
usenativecc = yes
target_os_family =
target_processor_family =
target_word_size =
requested_word_size =
win_target =
unix_target =
netware_target =
submake_targets =

# -- Enable command echoing --

ifneq (,$(findstring verbose,$(MAKECMDGOALS)))
	submake_targets += verbose
	ec =
else
	ec = @
endif

# -- Determine the host operating system --

ifndef host_os_family
	ifneq (,$(findstring WIN,$(OS)))
		host_os_family = win
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring Win,$(OS)))
		host_os_family = win
	endif
endif

ifndef host_os_family
	ifeq (,$(OSTYPE))
		ifneq (,$(RPM_OS))
			OSTYPE = $(RPM_OS)
		endif
	endif
	
	ifeq (,$(OSTYPE))
		OSTYPE := $(shell uname -s)
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring Linux,$(OSTYPE)))
		host_os_family = linux
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring linux,$(OSTYPE)))
		host_os_family = linux
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring solaris,$(OSTYPE)))
		host_os_family = solaris
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring SunOS,$(OSTYPE)))
		host_os_family = solaris
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring darwin,$(OSTYPE)))
		host_os_family = osx
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring Darwin,$(OSTYPE)))
		host_os_family = osx
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring aix,$(OSTYPE)))
		host_os_family = aix
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring hpux,$(OSTYPE)))
		host_os_family = hpux
	endif
endif

ifndef host_os_family
   $(error Host operating system could not be determined.  You may need to export OSTYPE from the environment.)
endif

# -- Target build type --

ifndef target_build_type
	ifneq (,$(findstring debug,$(MAKECMDGOALS)))
		submake_targets += debug
		target_build_type = debug
	endif
endif

ifndef target_build_type
	ifneq (,$(findstring release,$(MAKECMDGOALS)))
		submake_targets += release
		target_build_type = release
	endif
endif

ifndef target_build_type
	target_build_type = release
endif

# -- Use non-native (i.e., gcc) compiler on Solaris, etc.

ifneq (,$(findstring usegcc,$(MAKECMDGOALS)))
	submake_targets += usegcc
	usenativecc = no
endif

# -- Override platform default word size? --

ifneq (,$(findstring 64bit,$(MAKECMDGOALS)))
	submake_targets += 64bit
	requested_word_size = 64
endif

ifneq (,$(findstring 32bit,$(MAKECMDGOALS)))
	submake_targets += 32bit
	requested_word_size = 32
endif

# -- Target operating system --

ifndef target_os_family
	ifeq ($(host_os_family),linux)
		unix_target = yes
		target_os_family = linux
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),solaris)
		unix_target = yes
		target_os_family = solaris
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),osx)
		unix_target = yes
		target_os_family = osx
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),aix)
		unix_target = yes
		target_os_family = aix
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),hpux)
		unix_target = yes
		target_os_family = hpux
	endif
endif

ifneq (,$(findstring nlm,$(MAKECMDGOALS)))
	submake_targets += nlm
	netware_target = yes
	target_os_family = netware
	host_os_family = win
endif

ifndef target_os_family
	ifeq ($(host_os_family),win)
		win_target = yes
		target_os_family = win
	endif
endif

ifndef target_os_family
   $(error Target operating system could not be determined)
endif

# -- Host word size and processor --

host_native_word_size =
host_processor_family =
host_supported_word_sizes =

ifneq (,$(PROCESSOR_ARCHITECTURE))
	HOSTTYPE = $(PROCESSOR_ARCHITECTURE)
endif

ifeq (,$(HOSTTYPE))
	ifneq (,$(RPM_ARCH))
		HOSTTYPE = $(RPM_ARCH)
	endif
endif

ifeq (,$(HOSTTYPE))
	HOSTTYPE := $(shell uname -p)
	ifneq (,$(findstring nvalid,$(HOSTTYPE)))
		HOSTTYPE := $(shell uname -m)
	endif
endif

ifeq (,$(HOSTTYPE))
	HOSTYPE := $(shell uname -p)
endif

ifeq (,$(HOSTTYPE))
	$(error HOSTTYPE environment variable has not been set)
endif

ifndef host_native_word_size
	ifneq (,$(findstring x86_64,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring x86,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 32
		host_supported_word_sizes = 32
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring 86,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 32
		host_supported_word_sizes = 32
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring ia64,$(HOSTTYPE)))
		host_processor_family = ia64
		host_native_word_size = 64
		host_supported_word_sizes = 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring s390x,$(HOSTTYPE)))
		host_processor_family = s390
		host_native_word_size = 64
		host_supported_word_sizes = 31 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring s390,$(HOSTTYPE)))
		host_processor_family = s390
		host_native_word_size = 31
		host_supported_word_sizes = 31
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring ppc64,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring ppc,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 32
		host_supported_word_sizes = 32
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring sparc,$(HOSTTYPE)))
		host_processor_family = sparc
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring powerpc,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 32
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring Power,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 32
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring rs6000,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring hppa,$(HOSTTYPE)))
		host_processor_family = hppa
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring 9000,$(HOSTTYPE)))
		host_processor_family = hppa
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
   $(error Unable to determine host word size. $(HOSTTYPE))
endif

# -- Target word size and processor --

ifneq (,$(findstring nlm,$(MAKECMDGOALS)))
	target_processor_family = x86
	target_word_size = 32
	target_supported_word_sizes = 32
else
	target_processor_family = $(host_processor_family)
	target_word_size = $(host_native_word_size)
	target_supported_word_sizes = $(host_supported_word_sizes)
endif

ifdef requested_word_size
	ifneq (,$(findstring $(requested_word_size),$(target_supported_word_sizes)))
		target_word_size = $(requested_word_size)
	else
      $(error Unsupported target word size)
	endif
endif

# -- Helper functions --

define normpath
$(strip $(subst \,/,$(1)))
endef

ifeq (win,$(host_os_family))
define hostpath
$(strip $(subst /,\,$(1)))
endef
else
define hostpath
$(strip $(1))
endef
endif

ifeq (win,$(host_os_family))
define ppath
$(strip $(subst \,\\,$(subst /,\,$(1))))
endef
else
define ppath
$(strip $(1))
endef
endif

ifeq (win,$(host_os_family))
   define create_archive
		-$(ec)$(call rmcmd,$(2))
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) a -ttar -r $(call hostpath,$(2)).tar $(call hostpath,$(3))"
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) a -tgzip -r $(call hostpath,$(2)).tar.gz $(call hostpath,$(2)).tar"
		$(ec)cmd /C "cd $(call hostpath,$(1)) && del $(call hostpath,$(2)).tar"
   endef

   define extract_archive
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) x -y $(call hostpath,$(2)).tar.gz
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) x -y $(call hostpath,$(2)).tar
   endef
else
   define create_archive
		-$(ec)$(call rmcmd,$(2))
		$(ec)tar cf $(2).tar -C $(1) $(3)
		$(ec)gzip -f $(2).tar
		$(ec)chmod 775 $(2).tar.gz
   endef

   define extract_archive
		$(ec)gunzip $(strip $(1))/$(2).tar.gz
		$(ec)tar xvf $(strip $(1))/$(2).tar -C $(1)
   endef
endif

# Platform-specific commands, directories, etc.

ifeq ($(host_os_family),win)
	allprereqs  = $(call hostpath,$+)
	copycmd = copy /Y $(call hostpath,$(1)) $(call hostpath,$(2)) 1>NUL
	dircopycmd = xcopy /Y /E /V /I $(call hostpath,$(1)) $(call hostpath,$(2))
	rmcmd = if exist $(call hostpath,$(1)) del /Q $(call hostpath,$(1)) 1>NUL
	rmdircmd = if exist $(call hostpath,$(1)) rmdir /q /s $(call hostpath,$(1)) 1>NUL
	mkdircmd = -if not exist $(call hostpath,$(1)) mkdir $(call hostpath,$(1))
	runtest = cmd /C "cd $(call hostpath,$(test_dir)) && $(1) -d"
	topdir := $(call normpath,$(shell chdir))
else
	allprereqs = $+
	copycmd = cp -f $(1) $(2)
	dircopycmd = cp -rf $(1) $(2)
	rmcmd = rm -f $(1)
	rmdircmd = rm -rf $(1)
	mkdircmd = mkdir -p $(1)
	runtest = sh -c "cd $(test_dir); ./$(1) -d; exit"
	topdir := $(shell pwd)
endif

# If this is an un-tar'd or un-zipped source package, the tools directory
# will be subordinate to the top directory.  Otherwise, it will be
# a sibling to the top directory - which is how it is set up in the
# subversion repository.

ifeq "$(wildcard $(topdir)/tools*)" ""
	tooldir := $(dir $(topdir))tools/$(host_os_family)
else
	tooldir := $(topdir)/tools/$(host_os_family)
endif

# -- Utility variables --

em :=
sp := $(em) $(em)
percent := \045
dollar := \044
question := \077
asterisk := \052
dash := \055

# -- Tools --

ifdef unix_target
	gprintf = printf
else
	gprintf = $(call hostpath,$(tooldir)/printf.exe)
endif

# -- misc. targets --

.PHONY : status
status:
	$(ec)$(gprintf) "===============================================================================\n"
	$(ec)$(gprintf) "SVN Revision.................... $(svn_revision)\n"
	$(ec)$(gprintf) "Host Operating System Family.... $(host_os_family)\n"
	$(ec)$(gprintf) "Top Directory................... $(call ppath,$(topdir))\n"
	$(ec)$(gprintf) "Target Operating System Family.. $(target_os_family)\n"
	$(ec)$(gprintf) "Target Processor Family......... $(target_processor_family)\n"
	$(ec)$(gprintf) "Target Word Size................ $(target_word_size)\n"
	$(ec)$(gprintf) "Target Build Type............... $(target_build_type)\n"
	$(ec)$(gprintf) "Target Path..................... $(call ppath,$(target_path))\n"
	$(ec)$(gprintf) "Compiler........................ $(call ppath,$(compiler))\n"
	$(ec)$(gprintf) "Librarian....................... $(call ppath,$(libr))\n"
	$(ec)$(gprintf) "Defines......................... $(strip $(ccdefs))\n"
	$(ec)$(gprintf) "===============================================================================\n"
