#-------------------------------------------------------------------------
# Desc:	GNU makefile for FLAIM library and utilities
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
# $Id: Makefile 3105 2006-01-11 11:14:10 -0700 (Wed, 11 Jan 2006) ahodgkinson $
#-------------------------------------------------------------------------

#############################################################################
#
# Sample Usage:
#
#		make clean debug all
#
#############################################################################

# -- Include --

-include config.in

# -- Project --

project_name = flaim
project_desc = An extensible, flexible, adaptable, embeddable database engine

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

# -- Versions --

major_version = 4
minor_version = 8

version = $(major_version).$(minor_version).$(svn_revision)

so_current = 2
so_revision = 0
so_age = 1
shared_lib_version =

package_release_num = 1

# -- Paths initializations --

install_prefix = /usr

# -- RPM, SPEC file names

package_proj_name = lib$(project_name)
package_proj_name_and_ver = $(package_proj_name)-$(version)

# -- Determine if we are only cleaning --

util_targets = checkdb rebuild view sample dbshell
test_targets = basictest
all_targets = java rpms install libs all allutils test $(util_targets) $(test_targets)
found_targets = $(foreach target,$(MAKECMDGOALS),$(if $(findstring $(target),$(all_targets)),$(target),))

ifneq (,$(findstring clean,$(MAKECMDGOALS)))
	do_clean = 1
	ifeq ($(if $(findstring 0,$(words $(found_targets))),1,0),0)
      $(error Cannot specify other targets with clean target)
	endif
else
	do_clean = 0
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

# Files and Directories

ifeq ($(target_word_size),64)
	ifeq ($(target_os_family),linux)
		lib_dir_name = lib64
	endif
endif

ifndef lib_dir_name
	lib_dir_name = lib
endif

lib_install_dir = $(rpm_build_root)$(install_prefix)/$(lib_dir_name)
include_install_dir = $(rpm_build_root)$(install_prefix)/include
pkgconfig_install_dir = $(lib_install_dir)/pkgconfig
build_output_dir = $(topdir)/build
docs_output_dir = $(build_output_dir)/docs

target_path = $(build_output_dir)/$(target_os_family)-$(target_processor_family)-$(target_word_size)/$(target_build_type)

package_dir = $(target_path)/package
spec_dir = $(package_dir)/SPECS
spec_file = $(spec_dir)/$(package_proj_name).spec
package_sources_dir = $(package_dir)/SOURCES
package_bin_dir = $(package_dir)/BIN
package_build_dir = $(package_dir)/BUILD
package_rpms_dir = $(package_dir)/RPMS
package_srpms_dir = $(package_dir)/SRPMS
pkgconfig_file_name = $(package_proj_name).pc
pkgconfig_file = $(package_dir)/$(pkgconfig_file_name)

package_stage_parent_dir = $(package_dir)/stage
package_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)
package_bin_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)/$(target_os_family)-$(target_processor_family)-$(target_word_size)/$(target_build_type)
package_lib_stage_dir = $(package_bin_stage_dir)/lib
package_shared_lib_stage_dir = $(package_lib_stage_dir)/shared
package_static_lib_stage_dir = $(package_lib_stage_dir)/static
package_util_stage_dir = $(package_bin_stage_dir)/util
package_inc_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)/include

src_package_dir = $(package_sources_dir)
bin_package_dir = $(package_bin_dir)

src_package_base_name = $(package_proj_name_and_ver)
bin_package_base_name = $(package_proj_name_and_ver)-$(target_os_family)-$(target_processor_family)-$(target_word_size)-bin

src_package_name=$(src_package_base_name).tar.gz
bin_package_name=$(bin_package_base_name).tar.gz

rpm_name = $(package_proj_name_and_ver)-$(package_release_num).$(HOSTTYPE).rpm
srpm_name = $(package_proj_name_and_ver)-$(package_release_num).src.rpm
develrpm_name = $(package_proj_name)-devel-$(version)-$(package_release_num).$(HOSTTYPE).rpm

inc_dirs = src util
util_dir = $(target_path)/util
test_dir = $(target_path)/test
sample_dir = $(target_path)/sample
lib_dir = $(target_path)/$(lib_dir_name)
shared_lib_dir = $(lib_dir)/shared
static_lib_dir = $(lib_dir)/static

util_obj_dir = $(util_dir)/obj
test_obj_dir = $(test_dir)/obj
sample_obj_dir = $(sample_dir)/obj

lib_obj_dir = $(static_lib_dir)/obj
ifdef win_target
	lib_sobj_dir = $(shared_lib_dir)/obj
else
	lib_sobj_dir = $(lib_obj_dir)
endif

# -- Tools --

libr =
exe_linker =
shared_linker =
compiler =

# Compiler definitions and flags

ccflags =
ccdefs =

ifeq ($(target_word_size),64)
	ccdefs += FLM_64BIT
endif

##############################################################################
#   Win settings
##############################################################################
ifdef win_target
	exe_suffix = .exe
	obj_suffix = .obj
	lib_prefix =
	static_lib_suffix = .lib
	shared_lib_suffix = .dll
	libr = lib.exe
	exe_linker = link.exe
	shared_linker = link.exe
	compiler = cl.exe

	# Compiler defines and flags

	ccflags += /nologo /c /GF /GR /J /MD /W4 /WX /Zi /Zp1
	ccdefs += _CRT_SECURE_NO_DEPRECATE
	ccdefs += WIN32_LEAN_AND_MEAN
	ccdefs += WIN32_EXTRA_LEAN

	ifeq ($(target_build_type),debug)
		ccflags += /Ob1 /Od /RTC1 /Wp64
		ccdefs += FLM_DEBUG
	else
		ccflags += /O2
	endif

	# Linker switches

	shared_link_flags = \
		/DLL \
		/DEBUG /PDB:$(call hostpath,$(@:.dll=.pdb)) \
		/map:$(call hostpath,$(@:.dll=.map)) \
		/INCREMENTAL:NO \
		/NOLOGO \
		/OUT:$(call hostpath,$@)

	exe_link_flags = \
		/DEBUG /PDB:$(call hostpath,$(@:.exe=.pdb)) \
		/map:$(call hostpath,$(@:.exe=.map)) \
		/INCREMENTAL:NO \
		/FIXED:NO \
		/NOLOGO \
		/OUT:$(call hostpath,$@)

	# Libraries that our various components need to link against

	lib_link_libs = imagehlp.lib user32.lib rpcrt4.lib wsock32.lib
	exe_link_libs = $(lib_link_libs)

	# Convert the list of defines into a proper set of command-line params

	ifdef ccdefs
		ccdefine = $(foreach def,$(strip $(ccdefs)),/D$(def))
	endif

	# Same thing for the include dirs

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),/I$(call hostpath,$(inc_dir)))

	# Concatenate everything into the ccflags variable

	ccflags += $(ccdefine) $(ccinclude)
endif

##############################################################################
#   Linux/Unix settings
##############################################################################
ifdef unix_target

	ifneq ($(so_age),0)
		shared_lib_version = $(so_current).$(so_revision).$(so_age)
	else
		ifneq ($(so_revision),0)
			shared_lib_version = $(so_current).$(so_revision)
		else
			shared_lib_version = $(so_current)
		endif
	endif
	
	exe_suffix =
	obj_suffix = .o
	lib_prefix = lib
	static_lib_suffix = .a
	shared_lib_suffix = .so.$(shared_lib_version)
	
	compiler = g++
	exe_linker = g++
	shared_linker = g++

	ifeq ($(target_os_family),osx)
		libr = libtool
	else
		libr = ar
	endif

	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),solaris)
			compiler = CC
			exe_linker = CC
			shared_linker = ld
		endif
	endif

	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),aix)
			compiler = xlC_r
			exe_linker = xlC_r
			shared_linker = xlC_r
		endif
	endif

	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),hpux)
			compiler = aCC
			exe_linker = aCC
			shared_linker = aCC
		endif
	endif

	# Compiler defines and flags

	ifeq ($(compiler),g++)
		ccflags += -Wall -Werror -fPIC
		ifneq ($(target_processor_family),ia64)
			ccflags += -m$(target_word_size)
		endif
	endif

	ifeq ($(target_os_family),linux)

		# Must support 64 bit file sizes - even for 32 bit builds.

		ccdefs += N_PLAT_UNIX _LARGEFILE64_SOURCE _FILE_OFFSET_BITS=64

		ifeq ($(target_build_type),release)
			ccflags += \
				-O \
				-foptimize-sibling-calls \
				-fstrength-reduce -fcse-follow-jumps \
				-fcse-skip-blocks \
				-frerun-cse-after-loop \
				-frerun-loop-opt \
				-fgcse \
				-fgcse-lm \
				-fgcse-sm \
				-fdelete-null-pointer-checks \
				-fexpensive-optimizations \
				-fregmove \
				-fsched-interblock \
				-fsched-spec \
				-fcaller-saves \
				-fpeephole2 \
				-freorder-blocks \
				-freorder-functions \
				-falign-functions \
				-falign-jumps \
				-falign-loops \
				-falign-labels \
				-fcrossjumping
		endif
	endif

	ifeq ($(target_os_family),solaris)
		ifeq ($(usenativecc),yes)
			ccflags += -KPIC -errwarn=%all -errtags -erroff=hidef,inllargeuse
			ifeq ($(target_word_size),64)
				ccflags += -xarch=generic64
			else
				# Must support 64 bit file sizes - even for 32 bit builds.

				ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
				ccflags += -xarch=generic
			endif
		endif
	endif

	ifeq ($(target_os_family),aix)
		ifeq ($(usenativecc),yes)
			ccflags += -qthreaded
			ifeq ($(target_word_size),64)
				ccflags += -q64
			else
				# Must support 64 bit file sizes - even for 32 bit builds.

				ccflags += -q32
				ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
			endif
		endif
	endif

	ifeq ($(target_os_family),hpux)
		ifeq ($(usenativecc),yes)
			# Must support 64 bit file sizes - even for 32 bit builds.

			ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
		endif
	endif

	ccdefs += _REENTRANT

	ifeq ($(target_build_type),debug)
		ccdefs += FLM_DEBUG
		ccflags += -g
	endif

	# Convert the list of defines into a proper set of command-line params

	ifdef ccdefs
		ccdefine = $(foreach def,$(strip $(ccdefs)),-D$(def))
	endif

	# Same thing for the include dirs

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),-I$(inc_dir))

	# Concatenate everything into the ccflags variable

	ccflags += $(ccdefine) $(ccinclude)

	# Linker switches

	shared_link_flags =
	link_flags = -o $@

	ifeq ($(compiler),g++)
		ifneq ($(target_processor_family),ia64)
			shared_link_flags += -m$(target_word_size)
			link_flags += -m$(target_word_size)
		endif
	endif

	lib_link_libs = -lpthread
	exe_link_libs = -lpthread

	ifeq ($(target_os_family),linux)
		lib_link_libs += -lrt -lstdc++ -ldl
		exe_link_libs += -lrt -lstdc++ -ldl -lncurses
		shared_link_flags += -shared -Wl,-Bsymbolic -fpic \
			-Wl,-soname,$(@F).1 -o $@
	endif

	ifeq ($(target_os_family),solaris)
		link_flags += -R /usr/lib/lwp
		shared_link_flags += -G -pic -o $@
		
		ifeq ($(usenativecc),yes)
			ifeq ($(target_word_size),64)
				link_flags += -xarch=generic64
			else
				link_flags += -xarch=generic
			endif
		endif

		lib_link_libs += -lm -lc -ldl -lsocket -lnsl -lrt
		exe_link_libs += -lm -lc -ldl -lsocket -lnsl -lrt -lcurses
	endif

	ifeq ($(target_os_family),aix)
		lib_link_libs += -lm -lc
		exe_link_libs += -lm -lc -lcurses
	endif

	ifeq ($(target_os_family),hpux)
		lib_link_libs += -lm -lc -lrt
		exe_link_libs += -lm -lc -lrt -lcurses
	endif

	ifeq ($(target_os_family),osx)
		shared_lib_suffix = -$(major_version).$(so_current).dylib
		lib_link_libs += -lstdc++ -ldl
		exe_link_libs += -lstdc++ -ldl -lncurses
		shared_link_flags += -dynamiclib
		shared_link_flags += -current_version $(major_version).$(so_current).$(so_revision)
		shared_link_flags += -compatibility_version $(major_version).$(so_current).0
		shared_link_flags += -o $@
	endif

	exe_link_flags = $(link_flags)
endif

##############################################################################
#   NetWare settings
##############################################################################
ifdef netware_target
	exe_suffix = .nlm
	obj_suffix = .obj
	lib_prefix =
	static_lib_suffix = .lib
	shared_lib_suffix = .nlm

	ifdef WATCOM
		wc_dir = $(WATCOM)
	endif

	ifdef watcom
		wc_dir = $(watcom)
	endif

	ifndef wc_dir
		wc_dir = $(WC_DIR)
	endif

	ifndef wc_dir
      $(error Target operating system could not be determined)
	endif

	libr = "$(call normpath,$(strip $(wc_dir)))/binnt/wlib.exe"
	exe_linker = "$(call normpath,$(strip $(wc_dir)))/binnt/wlink.exe"
	shared_linker = "$(call normpath,$(strip $(wc_dir)))/binnt/wlink.exe"
	compiler = "$(call normpath,$(wc_dir))/binnt/wpp386.exe"

	ifneq ($(build),release)
		ccdefs += FLM_DEBUG
	endif

	ccflags += /ez /6s /w4 /za /zp1 /zq /zm /s /ei /of+ /we /bt=NETWARE

	ifeq ($(build),release)
		ccflags += /oair
	else
		ccflags += /hc
	endif

	libflags += /b /q /p=256
	link_flags = /m /l /v /s

	export include = $(foreach inc_dir,$(strip $(inc_dirs)),$(call hostpath,$(inc_dir));)
	export INCLUDE = $(include)
	export wpp386 = /d$(subst $(sp), /d,$(strip $(ccdefs))) $(ccflags)
	export wcc386 = /d$(subst $(sp), /d,$(strip $(ccdefs))) $(ccflags)

   define make_lis_file_cmd
		$(ec)$(gprintf) "option verbose\n" > $(4)
		$(ec)$(gprintf) "option stack=32k\n" >> $(4)
		$(ec)$(gprintf) "option synchronize\n" >> $(4)
		$(ec)$(gprintf) "option nod\n" >> $(4)
		$(ec)$(gprintf) "option map\n" >> $(4)
		$(ec)$(gprintf) "option start=FlmLoad, exit=FlmUnload, caseexact\n" >> $(4)
		$(ec)$(gprintf) "option nodefaultlibs\n" >> $(4)
		$(ec)$(gprintf) "option screenname 'NONE'\n" >> $(4)
		$(ec)$(gprintf) "option threadname '$(2)'\n" >> $(4)
		$(ec)$(gprintf) "debug all debug novell\n" >> $(4)
		$(ec)$(gprintf) "form novell nlm '$(2)'\n" >> $(4)
		$(ec)$(gprintf) "name $(call ppath,$(1)/$(2)$(exe_suffix))\n" >> $(4)

		$(ec)$(gprintf) "import __WSAFDIsSet\n" >> $(4)
		$(ec)$(gprintf) "import ActivateScreen\n" >> $(4)
		$(ec)$(gprintf) "import Alloc\n" >> $(4)
		$(ec)$(gprintf) "import AllocateResourceTag\n" >> $(4)
		$(ec)$(gprintf) "import atomic_dec\n" >> $(4)
		$(ec)$(gprintf) "import atomic_inc\n" >> $(4)
		$(ec)$(gprintf) "import atomic_xchg\n" >> $(4)
		$(ec)$(gprintf) "import BitTest\n" >> $(4)
		$(ec)$(gprintf) "import CEvaluateExpression\n" >> $(4)
		$(ec)$(gprintf) "import CFindLoadModuleHandle\n" >> $(4)
		$(ec)$(gprintf) "import CheckKeyStatus\n" >> $(4)
		$(ec)$(gprintf) "import ClearScreen\n" >> $(4)
		$(ec)$(gprintf) "import CloseFile\n" >> $(4)
		$(ec)$(gprintf) "import CloseScreen\n" >> $(4)
		$(ec)$(gprintf) "import CMovB\n" >> $(4)
		$(ec)$(gprintf) "import CMoveFast\n" >> $(4)
		$(ec)$(gprintf) "import ConvertPathString\n" >> $(4)
		$(ec)$(gprintf) "import ConvertSecondsToTicks\n" >> $(4)
		$(ec)$(gprintf) "import ConvertTicksToSeconds\n" >> $(4)
		$(ec)$(gprintf) "import CpuCurrentProcessor\n" >> $(4)
		$(ec)$(gprintf) "import CreateDirectory\n" >> $(4)
		$(ec)$(gprintf) "import CreateFile\n" >> $(4)
		$(ec)$(gprintf) "import CSetD\n" >> $(4)
		$(ec)$(gprintf) "import DebuggerSymbolList\n" >> $(4)
		$(ec)$(gprintf) "import DeleteDirectory\n" >> $(4)
		$(ec)$(gprintf) "import DirectorySearch\n" >> $(4)
		$(ec)$(gprintf) "import DirectReadFile\n" >> $(4)
		$(ec)$(gprintf) "import DirectReadFile\n" >> $(4)
		$(ec)$(gprintf) "import DirectWriteFile\n" >> $(4)
		$(ec)$(gprintf) "import DirectWriteFile\n" >> $(4)
		$(ec)$(gprintf) "import DirectWriteFileNoWait\n" >> $(4)
		$(ec)$(gprintf) "import DirectWriteFileNoWait\n" >> $(4)
		$(ec)$(gprintf) "import DisableInputCursor\n" >> $(4)
		$(ec)$(gprintf) "import DisplayScreenTextWithAttribute\n" >> $(4)
		$(ec)$(gprintf) "import DOSFirstByteBitMap\n" >> $(4)
		$(ec)$(gprintf) "import EnableInputCursor\n" >> $(4)
		$(ec)$(gprintf) "import EnterDebugger\n" >> $(4)
		$(ec)$(gprintf) "import EraseFile\n" >> $(4)
		$(ec)$(gprintf) "import ExpandFileInContiguousBlocks\n" >> $(4)
		$(ec)$(gprintf) "import ExpandFileInContiguousBlocks\n" >> $(4)
		$(ec)$(gprintf) "import ExportPublicSymbol\n" >> $(4)
		$(ec)$(gprintf) "import FindAndLoadNLM\n" >> $(4)
		$(ec)$(gprintf) "import Free\n" >> $(4)
		$(ec)$(gprintf) "import FreeLimboVolumeSpace\n" >> $(4)
		$(ec)$(gprintf) "import FreeLimboVolumeSpace\n" >> $(4)
		$(ec)$(gprintf) "import GetCacheBufferSize\n" >> $(4)
		$(ec)$(gprintf) "import GetClosestSymbol\n" >> $(4)
		$(ec)$(gprintf) "import GetCurrentClock\n" >> $(4)
		$(ec)$(gprintf) "import GetCurrentNumberOfCacheBuffers\n" >> $(4)
		$(ec)$(gprintf) "import GetCurrentTime\n" >> $(4)
		$(ec)$(gprintf) "import GetEntryFromPathStringBase\n" >> $(4)
		$(ec)$(gprintf) "import GetEntryFromPathStringBase\n" >> $(4)
		$(ec)$(gprintf) "import GetFileSize\n" >> $(4)
		$(ec)$(gprintf) "import GetKey\n" >> $(4)
		$(ec)$(gprintf) "import GetNLMAllocMemoryCounts\n" >> $(4)
		$(ec)$(gprintf) "import GetOriginalNumberOfCacheBuffers\n" >> $(4)
		$(ec)$(gprintf) "import GetProductMajorVersionNumber\n" >> $(4)
		$(ec)$(gprintf) "import GetRunningProcess\n" >> $(4)
		$(ec)$(gprintf) "import GetScreenSize\n" >> $(4)
		$(ec)$(gprintf) "import GetSyncClockFields\n" >> $(4)
		$(ec)$(gprintf) "import GetSystemConsoleScreen\n" >> $(4)
		$(ec)$(gprintf) "import ImportPublicSymbol\n" >> $(4)
		$(ec)$(gprintf) "import kCreateThread\n" >> $(4)
		$(ec)$(gprintf) "import kCurrentThread\n" >> $(4)
		$(ec)$(gprintf) "import kDelayThread\n" >> $(4)
		$(ec)$(gprintf) "import kDestroyThread\n" >> $(4)
		$(ec)$(gprintf) "import kExitThread\n" >> $(4)
		$(ec)$(gprintf) "import kGetThreadName\n" >> $(4)
		$(ec)$(gprintf) "import kGetThreadName\n" >> $(4)
		$(ec)$(gprintf) "import KillMe\n" >> $(4)
		$(ec)$(gprintf) "import kMutexAlloc\n" >> $(4)
		$(ec)$(gprintf) "import kMutexFree\n" >> $(4)
		$(ec)$(gprintf) "import kMutexLock\n" >> $(4)
		$(ec)$(gprintf) "import kMutexUnlock\n" >> $(4)
		$(ec)$(gprintf) "import kQueCount\n" >> $(4)
		$(ec)$(gprintf) "import kReturnCurrentProcessorID\n" >> $(4)
		$(ec)$(gprintf) "import kScheduleThread\n" >> $(4)
		$(ec)$(gprintf) "import kSemaphoreAlloc\n" >> $(4)
		$(ec)$(gprintf) "import kSemaphoreExamineCount\n" >> $(4)
		$(ec)$(gprintf) "import kSemaphoreFree\n" >> $(4)
		$(ec)$(gprintf) "import kSemaphoreSignal\n" >> $(4)
		$(ec)$(gprintf) "import kSemaphoreTimedWait\n" >> $(4)
		$(ec)$(gprintf) "import kSemaphoreWait\n" >> $(4)
		$(ec)$(gprintf) "import kSetThreadLoadHandle\n" >> $(4)
		$(ec)$(gprintf) "import kSetThreadName\n" >> $(4)
		$(ec)$(gprintf) "import kYieldIfTimeSliceUp\n" >> $(4)
		$(ec)$(gprintf) "import kYieldThread\n" >> $(4)
		$(ec)$(gprintf) "import LoadModule\n" >> $(4)
		$(ec)$(gprintf) "import LoadRules\n" >> $(4)
		$(ec)$(gprintf) "import MapFileHandleToFCB\n" >> $(4)
		$(ec)$(gprintf) "import MapPathToDirectoryNumber\n" >> $(4)
		$(ec)$(gprintf) "import MapPathToDirectoryNumber\n" >> $(4)
		$(ec)$(gprintf) "import MapVolumeNameToNumber\n" >> $(4)
		$(ec)$(gprintf) "import ModifyDirectoryEntry\n" >> $(4)
		$(ec)$(gprintf) "import ModifyDirectoryEntry\n" >> $(4)
		$(ec)$(gprintf) "import MountVolume\n" >> $(4)
		$(ec)$(gprintf) "import NDSCreateStreamFile\n" >> $(4)
		$(ec)$(gprintf) "import NDSDeleteStreamFile\n" >> $(4)
		$(ec)$(gprintf) "import NDSOpenStreamFile\n" >> $(4)
		$(ec)$(gprintf) "import NWLocalToUnicode\n" >> $(4)
		$(ec)$(gprintf) "import NWUnicodeToLocal\n" >> $(4)
		$(ec)$(gprintf) "import OpenFile\n" >> $(4)
		$(ec)$(gprintf) "import OpenScreen\n" >> $(4)
		$(ec)$(gprintf) "import OutputToScreen\n" >> $(4)
		$(ec)$(gprintf) "import PositionInputCursor\n" >> $(4)
		$(ec)$(gprintf) "import PositionOutputCursor\n" >> $(4)
		$(ec)$(gprintf) "import ReadFile\n" >> $(4)
		$(ec)$(gprintf) "import RenameEntry\n" >> $(4)
		$(ec)$(gprintf) "import RestartServer\n" >> $(4)
		$(ec)$(gprintf) "import ReturnResourceTag\n" >> $(4)
		$(ec)$(gprintf) "import ReturnVolumeMappingInformation\n" >> $(4)
		$(ec)$(gprintf) "import ReturnVolumeMappingInformation\n" >> $(4)
		$(ec)$(gprintf) "import RevokeFileHandleRights\n" >> $(4)
		$(ec)$(gprintf) "import SetCursorStyle\n" >> $(4)
		$(ec)$(gprintf) "import SetFileSize\n" >> $(4)
		$(ec)$(gprintf) "import SetFileSize\n" >> $(4)
		$(ec)$(gprintf) "import SGUIDCreate\n" >> $(4)
		$(ec)$(gprintf) "import SizeOfAllocBlock\n" >> $(4)
		$(ec)$(gprintf) "import StopBell\n" >> $(4)
		$(ec)$(gprintf) "import SwitchToDirectFileMode\n" >> $(4)
		$(ec)$(gprintf) "import SwitchToDirectFileMode\n" >> $(4)
		$(ec)$(gprintf) "import UngetKey\n" >> $(4)
		$(ec)$(gprintf) "import UnImportPublicSymbol\n" >> $(4)
		$(ec)$(gprintf) "import UnloadRules\n" >> $(4)
		$(ec)$(gprintf) "import VMGetDirectoryEntry\n" >> $(4)
		$(ec)$(gprintf) "import WorkToDoListHead\n" >> $(4)
		$(ec)$(gprintf) "import WriteFile\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_bind\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_closesocket\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_gethostbyaddr\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_gethostbyname\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_gethostname\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_htonl\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_htons\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_inet_addr\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_inet_ntoa\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_listen\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_recv\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_select\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_send\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_setsockopt\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_shutdown\n" >> $(4)
		$(ec)$(gprintf) "import WS2_32_socket\n" >> $(4)
		$(ec)$(gprintf) "import WSAAccept\n" >> $(4)
		$(ec)$(gprintf) "import WSACleanup\n" >> $(4)
		$(ec)$(gprintf) "import WSAConnect\n" >> $(4)
		$(ec)$(gprintf) "import WSAGetLastError\n" >> $(4)
		$(ec)$(gprintf) "import WSAStartup\n" >> $(4)
		$(ec)$(gprintf) "file $(subst $(sp),\nfile ,$(call ppath,$(3)))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(ctorarst))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(dtorarst))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(undefed))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(undefmbd))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(pure_err))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(stablcl))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(stabact))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(stabactv))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(stabmod))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(prwdata))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(moddtorr))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib(stabadt))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib(i8d))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib(i8m))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib(i8s))\n" >> $(4)
		$(ec)$(gprintf) "library $(call ppath,$(flaim_static_lib))\n" >> $(4)
   endef

   define flm_exe_link_cmd
		$(call make_lis_file_cmd,$(1),$(2),$(3),$(call hostpath,$(1)/$(2).lis))
		$(ec)$(call hostpath,$(exe_linker)) @$(call hostpath,$(1)/$(2).lis)
		$(ec)$(call rmcmd,$(target_path)/$(1).lis)
   endef

endif

# -- File lists --

flaim_src = \
	$(patsubst src/%.cpp,%.cpp,$(wildcard src/*.cpp))

utilsup_src = \
	flm_dlst.cpp \
	flm_lutl.cpp \
	ftx.cpp \
	ftxunix.cpp \
	sharutil.cpp \
	wpscrnkb.cpp

checkdb_src = \
	checkdb.cpp \
	$(utilsup_src)

rebuild_src = \
	rebuild.cpp \
	$(utilsup_src)

view_src = \
	view.cpp \
	viewblk.cpp \
	viewdisp.cpp \
	viewedit.cpp \
	viewfhdr.cpp \
	viewlhdr.cpp \
	viewlfil.cpp \
	viewmenu.cpp \
	viewsrch.cpp \
	$(utilsup_src)

sample_src = \
	sample.cpp

dbshell_src = \
	dbshell.cpp \
	flm_edit.cpp \
	$(utilsup_src)

ut_basictest_src = \
	flmunittest.cpp \
	basic_test.cpp \
	$(utilsup_src)

# -- FLAIM library --

flaim_static_obj = $(patsubst %.cpp,$(lib_obj_dir)/%$(obj_suffix),$(flaim_src))
flaim_shared_obj = $(patsubst %.cpp,$(lib_sobj_dir)/%$(obj_suffix),$(flaim_src))
flaim_static_lib = $(static_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
ifndef netware_target
	flaim_shared_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)
	flaim_shared_imp_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
endif

# -- Unit tests  --

ut_basictest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_basictest_src))

# -- Utilities  --

checkdb_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(checkdb_src))
checkdb_exe = $(util_dir)/checkdb$(exe_suffix)

rebuild_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(rebuild_src))
rebuild_exe = $(util_dir)/rebuild$(exe_suffix)

view_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(view_src))
view_exe = $(util_dir)/view$(exe_suffix)

dbshell_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(dbshell_src))
dbshell_exe = $(util_dir)/dbshell$(exe_suffix)

sample_obj = $(patsubst %.cpp,$(sample_obj_dir)/%$(obj_suffix),$(sample_src))
sample_exe = $(sample_dir)/sample$(exe_suffix)

# -- Make system pattern search paths --

vpath %.cpp src util sample

# -- Default target --

.PHONY : libs
libs: status clean dircheck $(flaim_static_lib) $(flaim_shared_lib)

# -- *.cpp -> *$(obj_suffix) --

ifdef win_target
define cpp_to_obj_template
$$($(1)_obj_dir)/%$$(obj_suffix) : %.cpp
	$$(ec)$$(compiler) $$(ccflags) /Fo$$(call hostpath,$$@) $$(call hostpath,$$<)
endef
endif

ifdef unix_target
define cpp_to_obj_template
$$($(1)_obj_dir)/%$$(obj_suffix) : %.cpp
	$$(ec)$$(gprintf) "$$<\n"
	$$(ec)$$(compiler) $$(ccflags) -c $$< -o $$@
endef
endif

ifdef netware_target
define cpp_to_obj_template
$$($(1)_obj_dir)/%$(obj_suffix) : %.cpp
	$$(ec)$$(gprintf) "$$(notdir $$(strip $$@))\n"
	$$(ec)$$(call hostpath,$$(compiler)) $$(call hostpath,$$<) /fo=$$(call hostpath,$$@)
endef
endif

$(foreach tmpl,lib util sample test,$(eval $(call cpp_to_obj_template,$(tmpl))))

ifdef win_target
$(lib_sobj_dir)/%$(obj_suffix) : %.cpp
	$(ec)$(compiler) $(ccflags) /DFLM_SRC /DFLM_DLL \
		/Fd$(call hostpath,$(lib_sobj_dir)/tmp.pdb) \
		/Fo$(call hostpath,$@) $(call hostpath,$<)
endif

# -- flaim.lib and libflaim.a --

$(flaim_static_lib) : $(flaim_static_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(libr) /NOLOGO $(call hostpath,$+) /OUT:$(call hostpath,$@)
endif
ifdef unix_target
	$(ec)rm -f $@
ifeq ($(target_os_family),osx)
	$(ec)$(libr) -static -o $@ $+
else
	$(ec)$(libr) -rcs $@ $+
endif
endif
ifdef netware_target
	$(ec)dir /s/b  $(call hostpath,$(lib_obj_dir)/*$(obj_suffix)) > $(call hostpath,$(static_lib_dir)/flmlib.lis)
	$(ec)$(call hostpath,$(libr)) $(libflags) $(call hostpath,$(flaim_static_lib)) @$(call hostpath,$(static_lib_dir)/flmlib.lis)
endif

# -- flaim.dll and libflaim.so --

$(flaim_shared_lib) : $(flaim_shared_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(exe_linker) $(call hostpath,$+) $(shared_link_flags) $(lib_link_libs)
endif
ifdef unix_target
	$(ec)rm -f $@
	$(ec)$(shared_linker) $+ $(shared_link_flags) $(lib_link_libs)
endif

# -- Utility link command --

ifndef flm_exe_link_cmd
   define flm_exe_link_cmd
		$(ec)$(exe_linker) $(exe_link_flags) $(allprereqs) $(exe_link_libs)
   endef
endif

# -- checkdb --

.PHONY : checkdb
checkdb: status clean dircheck $(flaim_static_lib) $(checkdb_exe)
$(checkdb_exe): $(checkdb_obj) $(flaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),checkdb,$(checkdb_obj))

# -- rebuild --

.PHONY : rebuild
rebuild: status clean dircheck $(flaim_static_lib) $(rebuild_exe)
$(rebuild_exe): $(rebuild_obj) $(flaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),rebuild,$(rebuild_obj))

# -- view --

.PHONY : view
view: status clean dircheck $(flaim_static_lib) $(view_exe)
$(view_exe): $(view_obj) $(flaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),view,$(view_obj))

# -- dbshell --

.PHONY : dbshell
dbshell: status clean dircheck $(flaim_static_lib) $(dbshell_exe)
$(dbshell_exe): $(dbshell_obj) $(flaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),dbshell,$(dbshell_obj))

# -- sample --

.PHONY : sample
ifndef netware_target
sample: status clean dircheck $(flaim_static_lib) $(sample_exe)
$(sample_exe): $(sample_obj) $(flaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(sample_dir),sample,$(sample_obj))
endif

# -- basictest --

.PHONY : basictest
basictest: status clean dircheck $(flaim_static_lib) $(test_dir)/basictest$(exe_suffix)
$(test_dir)/basictest$(exe_suffix): $(ut_basictest_obj) $(flaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir)basictest,$(ut_basictest_obj))

# -- version --

define make_version_files
	$(ec)$(gprintf) "$(version)" > $(1)/VERSION
	$(ec)$(gprintf) " " > $(1)/SVNRevision.$(svn_revision)
	$(ec)$(gprintf) "Version files created.\n"
endef

# -- srcdist --

.PHONY : srcdist
srcdist: status clean dircheck docs spec
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating source package (SVN revision $(svn_revision)) ...\n"
   endif
	-$(ec)$(call rmdircmd,$(package_stage_dir))
	$(ec)$(call mkdircmd,$(package_stage_dir))
	$(ec)$(call make_version_files,$(package_stage_dir))
	$(ec)$(call copycmd,Makefile,$(package_stage_dir))
	$(ec)$(call copycmd,COPYING,$(package_stage_dir))
	$(ec)$(call copycmd,Doxyfile,$(package_stage_dir))
	$(ec)$(call dircopycmd,src,$(package_stage_dir)/src)
	$(ec)$(call dircopycmd,util,$(package_stage_dir)/util)
	$(ec)$(call dircopycmd,$(dir $(topdir))tools,$(package_stage_dir)/tools)
ifneq ($(host_os_family),win)
	-$(ec)rm -rf `find $(package_stage_dir) -name .svn`
endif
	$(ec)$(call dircopycmd,$(docs_output_dir),$(package_stage_dir)/docs)
ifeq ($(host_os_family),win)
	$(ec)$(call copycmd,make.exe,$(package_stage_dir))
endif
	$(ec)$(call create_archive,$(package_stage_parent_dir), \
		$(src_package_dir)/$(src_package_base_name), \
		$(package_proj_name_and_ver))
	$(ec)$(call copycmd,$(package_proj_name).changes,$(package_stage_dir))
	$(ec)$(call rmdircmd,$(package_stage_dir))
	$(ec)$(gprintf) "Source package created.\n"

# -- bindist --

.PHONY : bindist
bindist: status clean dircheck all binpackage
	$(ec)$(gprintf) ""

# -- binpackage --

.PHONY : binpackage
binpackage: status
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating binary package (SVN revision $(svn_revision)) ...\n"
   endif
	-$(ec)$(call rmdircmd,$(package_stage_dir))
	$(ec)$(call mkdircmd,$(package_stage_dir))
	$(ec)$(call mkdircmd,$(package_inc_stage_dir))
	$(ec)$(call mkdircmd,$(package_shared_lib_stage_dir))
	$(ec)$(call mkdircmd,$(package_static_lib_stage_dir))
	$(ec)$(call mkdircmd,$(package_util_stage_dir))
	$(ec)$(call make_version_files,$(package_stage_dir))
	$(ec)$(call copycmd,COPYING,$(package_stage_dir))
	$(ec)$(call copycmd,src/flaim.h,$(package_inc_stage_dir))
	$(ec)$(call copycmd,$(flaim_static_lib),$(package_static_lib_stage_dir))
ifdef flaim_shared_lib
	$(ec)$(call copycmd,$(flaim_shared_lib),$(package_shared_lib_stage_dir))
endif
ifdef win_target
	$(ec)$(call copycmd,$(flaim_shared_imp_lib),$(package_shared_lib_stage_dir))
endif
	$(ec)$(call copycmd,$(checkdb_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(rebuild_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(view_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(dbshell_exe),$(package_util_stage_dir))
	$(ec)$(call create_archive,$(package_stage_parent_dir), \
		$(bin_package_dir)/$(bin_package_base_name), \
		$(package_proj_name_and_ver))
	$(ec)$(call rmdircmd,$(package_stage_parent_dir))
	$(ec)$(gprintf) "Binary package created.\n"

# -- dist --

.PHONY : dist
dist: status clean dircheck srcdist
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating distribution (SVN revision $(svn_revision)) ...\n"
   endif
	$(ec)$(call copycmd,$(src_package_dir)/$(src_package_name),$(package_dir))
	$(ec)$(call extract_archive,$(package_dir),$(src_package_base_name))
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) clean
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) $(submake_targets) all
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) $(submake_targets) binpackage package_dir="$(package_dir)"
	$(ec)$(call rmdircmd,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call rmcmd,$(package_dir)/$(src_package_name))
	$(ec)$(gprintf) "Distribution created.\n"

# -- Change log --

.PHONY : changelog
changelog:
	$(ec)$(gprintf) "Creating change log for SVN revisions $(svn_low_rev)-$(svn_high_rev) ...\n"
	$(ec)$(gprintf) "Using SVN user $(svn_user) ...\n"
	$(ec)$(gprintf) "Using SVN URL $(svnurl) ...\n"
	$(ec)svn log $(svnurl) -v -r $(svn_low_rev):$(svn_high_rev) > $(package_sources_dir)/$(package_proj_name_and_ver).tar.log
	$(ec)$(gprintf) "Change log created.\n"

# -- install --

.PHONY : install
install: libs pkgconfig
ifneq ($(host_os_family),win)
	$(ec)$(gprintf) "Installing ...\n"
	mkdir -p $(lib_install_dir)/pkgconfig
	mkdir -p $(include_install_dir)
	install --mode=644 $(flaim_shared_lib) $(lib_install_dir)
	install --mode=644 $(flaim_static_lib) $(lib_install_dir)
	install --mode=644 $(pkgconfig_file) $(pkgconfig_install_dir)
	install --mode=644 src/flaim.h $(include_install_dir)
	-ldconfig $(lib_install_dir) -l -v $(lib_install_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)
	$(ec)$(gprintf) "Installation complete.\n"
endif

# -- uninstall --

.PHONY : uninstall
uninstall:
ifneq ($(host_os_family),win)
	$(ec)$(gprintf) "Uninstalling ...\n"
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)*$(shared_lib_suffix)*
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
	-rm -rf $(pkgconfig_install_dir)/$(pkgconfig_file_name)
	-rm -rf $(include_install_dir)/flaim.h
	$(ec)$(gprintf) "Uninstalled.\n"
endif

# -- spec file --

.PHONY : spec
spec: dircheck
	$(ec)$(gprintf) "Creating spec file ...\n"
	$(ec)$(gprintf) "Name: $(package_proj_name)\n" > $(spec_file)
	$(ec)$(gprintf) "$(percent)define prefix $(install_prefix)\n" >> $(spec_file)
	$(ec)$(gprintf) "BuildRequires: gcc-c++ libstdc++ libstdc++-devel\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: $(project_desc)\n" >> $(spec_file)
	$(ec)$(gprintf) "URL: http://forge.novell.com/modules/xfmod/project/$(question)flaim\n" >> $(spec_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(spec_file)
	$(ec)$(gprintf) "Release: $(package_release_num)\n" >> $(spec_file)
	$(ec)$(gprintf) "License: GPL\n" >> $(spec_file)
	$(ec)$(gprintf) "Vendor: Novell, Inc.\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "Source: $(package_proj_name_and_ver).tar.gz\n" >> $(spec_file)
	$(ec)$(gprintf) "BuildRoot: $(percent){_tmppath}/$(percent){name}-$(percent){version}-build\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description\n" >> $(spec_file)
	$(ec)$(gprintf) "FLAIM is an embeddable cross-platform database engine that provides a\n" >> $(spec_file)
	$(ec)$(gprintf) "rich, powerful, easy-to-use feature set. It is the database engine used\n" >> $(spec_file)
	$(ec)$(gprintf) "by Novell eDirectory. It has proven to be highly scalable, reliable,\n" >> $(spec_file)
	$(ec)$(gprintf) "and robust. It is available on a wide variety of 32 bit and 64 bit\n" >> $(spec_file)
	$(ec)$(gprintf) "platforms.\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "Authors:\n" >> $(spec_file)
	$(ec)$(gprintf) "$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)\n" >> $(spec_file)
	$(ec)$(gprintf) "    Daniel Sanders\n" >> $(spec_file)
	$(ec)$(gprintf) "    Andrew Hodgkinson\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)package devel\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: FLAIM static library and header file\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "Provides: $(package_proj_name)-devel\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description devel\n" >> $(spec_file)
	$(ec)$(gprintf) "FLAIM is an embeddable cross-platform database engine that provides a\n" >> $(spec_file)
	$(ec)$(gprintf) "rich, powerful, easy-to-use feature set. It is the database engine used\n" >> $(spec_file)
	$(ec)$(gprintf) "by Novell eDirectory. It has proven to be highly scalable, reliable,\n" >> $(spec_file)
	$(ec)$(gprintf) "and robust. It is available on a wide variety of 32 bit and 64 bit\n" >> $(spec_file)
	$(ec)$(gprintf) "platforms.\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "Authors:\n" >> $(spec_file)
	$(ec)$(gprintf) "$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)\n" >> $(spec_file)
	$(ec)$(gprintf) "    Daniel Sanders\n" >> $(spec_file)
	$(ec)$(gprintf) "    Andrew Hodgkinson\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)prep\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)setup -q\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)build\n" >> $(spec_file)
	$(ec)$(gprintf) "$(MAKE) lib_dir_name=$(percent){_lib} $(submake_targets) libs\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)install\n" >> $(spec_file)
	$(ec)$(gprintf) "$(MAKE) rpm_build_root=$(dollar)RPM_BUILD_ROOT install_prefix=$(percent){prefix} lib_dir_name=$(percent){_lib} $(submake_targets) install\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)clean\n" >> $(spec_file)
	$(ec)$(gprintf) "rm -rf $(dollar)RPM_BUILD_ROOT\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)defattr(-,root,root)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)doc COPYING VERSION\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(asterisk)$(shared_lib_suffix)$(asterisk)\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files devel\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(static_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/pkgconfig/$(pkgconfig_file_name)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/include/flaim.h\n" >> $(spec_file)
	$(ec)$(gprintf) "Created spec file.\n"

# -- PKG-CONFIG --

.PHONY : pkgconfig
pkgconfig: dircheck
	$(ec)$(gprintf) "prefix=$(install_prefix)\n" > $(pkgconfig_file)
	$(ec)$(gprintf) "exec_prefix=$(dollar){prefix}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "libdir=$(dollar){exec_prefix}/$(lib_dir_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "includedir=$(dollar){prefix}/include\n\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Name: $(package_proj_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Description: $(project_desc)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Libs: $(lib_link_libs) -lflaim -L$(dollar){libdir}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Cflags: -I$(dollar){includedir}\n" >> $(pkgconfig_file)

# -- SRCRPM --

.PHONY : srcrpm
srcrpm: srcdist spec
	$(ec)$(gprintf) "Creating source RPM ...\n"
	$(ec)rpmbuild --define="_topdir $(package_dir)" --quiet -bs $(spec_file)
	$(ec)$(gprintf) "Source RPM created.\n"

# -- RPMS --

.PHONY : rpms
rpms: dist spec
	$(ec)$(gprintf) "Creating source and binary RPMs ...\n"
	$(ec)rpmbuild --define="_topdir $(package_dir)" --quiet -ba $(spec_file)
	$(ec)find $(package_dir) -name *.rpm | xargs chmod 775
	$(ec)$(gprintf) "Source and binary RPMs created.\n"

# -- Documentation --

.PHONY : docs
docs: status clean dircheck
	$(ec)$(gprintf) "Creating documentation ...\n"
	$(ec)doxygen Doxyfile
	$(ec)$(gprintf) "Documentation created.\n"

.PHONY : dircheck
dircheck:
	$(ec)$(call mkdircmd,$(util_obj_dir))
	$(ec)$(call mkdircmd,$(test_obj_dir))
	$(ec)$(call mkdircmd,$(sample_obj_dir))
	$(ec)$(call mkdircmd,$(lib_obj_dir))
ifneq ($(lib_sobj_dir),$(lib_obj_dir))
	$(ec)$(call mkdircmd,$(lib_sobj_dir))
endif
	$(ec)$(call mkdircmd,$(util_dir))
	$(ec)$(call mkdircmd,$(test_dir))
	$(ec)$(call mkdircmd,$(sample_dir))
	$(ec)$(call mkdircmd,$(static_lib_dir))
	$(ec)$(call mkdircmd,$(shared_lib_dir))
	$(ec)$(call mkdircmd,$(package_dir))
	$(ec)$(call mkdircmd,$(spec_dir))
	$(ec)$(call mkdircmd,$(package_sources_dir))
	$(ec)$(call mkdircmd,$(package_bin_dir))
	$(ec)$(call mkdircmd,$(package_build_dir))
	$(ec)$(call mkdircmd,$(package_rpms_dir))
	$(ec)$(call mkdircmd,$(package_srpms_dir))

# -- phony targets --

.PHONY : all
all: libs allutils
	$(ec)$(gprintf) ""

.PHONY : allutils
allutils: status clean dircheck $(util_targets)
	$(ec)$(gprintf) ""

.PHONY : test
test:	status clean dircheck $(flaim_static_lib) $(test_targets)
	$(ec)$(call runtest,basictest)

.PHONY : debug
debug:
	$(ec)$(gprintf) ""

.PHONY : release
release:
	$(ec)$(gprintf) ""

.PHONY : flm_dbg_log
flm_dbg_log:
	$(ec)$(gprintf) ""

.PHONY : usegcc
usegcc:
	$(ec)$(gprintf) ""

.PHONY : 32bit
32bit:
	$(ec)$(gprintf) ""

.PHONY : 64bit
64bit:
	$(ec)$(gprintf) ""

.PHONY : win
win:
	$(ec)$(gprintf) ""

.PHONY : linux
linux:
	$(ec)$(gprintf) ""

.PHONY : solaris
solaris:
	$(ec)$(gprintf) ""

.PHONY : osx
osx:
	$(ec)$(gprintf) ""

.PHONY : nlm
nlm:
	$(ec)$(gprintf) ""

.PHONY : verbose
verbose:
	$(ec)$(gprintf) ""

.PHONY : check
check:
	$(ec)$(gprintf) ""

.PHONY : TAGS
TAGS:
	$(ec)$(gprintf) ""

.PHONY : info
info:
	$(ec)$(gprintf) ""

.PHONY : ignore-local-mods
ignore-local-mods:
	$(ec)$(gprintf) ""

.PHONY : ilm
ilm:
	$(ec)$(gprintf) ""

.PHONY : installcheck
installcheck:
	$(ec)$(gprintf) ""

.PHONY : clean
clean:
ifeq ($(do_clean),1)
	$(ec)$(gprintf) "\n"
	$(ec)$(gprintf) "Cleaning $(target_path) ...\n"
	-$(ec)$(call rmdircmd,$(target_path))
	-$(ec)$(call rmcmd,*.pch)
	$(ec)$(gprintf) "\n"
endif

.PHONY : distclean
	-$(ec)$(call rmcmd,*.pch)

.PHONY : mostlyclean
mostlyclean : clean
	$(ec)$(gprintf) ""

.PHONY : maintainer-clean
maintainer-clean:
	-$(ec)$(call rmdircmd,$(build_output_dir))
	-$(ec)$(call rmcmd,*.pch)

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
