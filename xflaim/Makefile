#-------------------------------------------------------------------------
# Desc:	GNU makefile for XFLAIM library and utilities
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
# $Id: Makefile 3136 2006-01-25 12:19:01 -0700 (Wed, 25 Jan 2006) dsanders $
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

project_name = xflaim
project_desc = An extensible, flexible, adaptable, embeddable XML database engine

# -- Versions --

major_version = 5
minor_version = 0

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

version = $(major_version).$(minor_version).$(svn_revision)

so_current = 1
so_revision = 0
so_age = 0
suffix_version =

package_release_num = 1

# -- Paths initializations --

ifndef rpm_build_root
	ifneq (,$(DESTDIR))
		rpm_build_root = $(DESTDIR)/
	else
		rpm_build_root =
	endif
endif

install_prefix = /usr

# -- RPM, SPEC file names

package_proj_name = lib$(project_name)
package_proj_name_and_ver = $(package_proj_name)-$(version)

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
package_archive_format = tar

# -- Enable command echoing --

ifneq (,$(findstring verbose,$(MAKECMDGOALS)))
	submake_targets += verbose
	ec =
else
	ec = @
endif

# -- Determine if we are only cleaning --

util_targets = checkdb rebuild view sample xshell
test_targets = basictest binarytest colldeftest dictchangetest dictdeftest \
		dirtyexittest1 dirtyexittest2 domnodetest enctest importtest \
		indexdeftest indextest1 indextest2 indextest3 \
		namespacetest metaphonetest regressiontest rfltest sortkeytest \
		sortkeytest2 xpathtest1 xpathtest2
all_targets = java rpms install libs all allutils test $(util_targets) $(test_targets)
found_targets = $(foreach target,$(MAKECMDGOALS),$(if $(findstring $(target),$(all_targets)),$(target),))

ifneq (,$(findstring clean,$(MAKECMDGOALS)))
	clean_only = $(if $(findstring 0,$(words $(found_targets))),1,0)
	do_clean = 1
	ifeq ($(clean_only),0)
      $(error Cannot specify other targets with clean target)
	endif
else
	clean_only = 0
	do_clean = 0
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

ifeq (,$(OSTYPE))
	ifneq (,$(RPM_OS))
		HOSTTYPE = $(RPM_OS)
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
	ifneq (,$(findstring darwin,$(OSTYPE)))
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
   $(error Host operating system could not be determined.  You may need to export HOSTTYPE from the environment.)
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
	package_archive_format = zip
	host_os_family = win
endif

ifndef target_os_family
	ifeq ($(host_os_family),win)
		win_target = yes
		target_os_family = win
		package_archive_format = zip
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
   $(error HOSTTYPE environment variable has not been set)
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
	$(error Unable to determine host word size)
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

ifeq ($(package_archive_format),tar)
   define create_archive
		-$(ec)$(call rmcmd,$(2))
		$(ec)tar zcf $(2) -C $(1) $(3)
		$(ec)chmod 775 $(2)
   endef

   define extract_archive
		$(ec)tar zxvf $(strip $(1))/$(2) -C $(1)
   endef
else
   define create_archive
		-$(ec)$(call rmcmd,$(2))
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) a -tzip -r $(call hostpath,$(2)) $(call hostpath,$(3))"
   endef
	
   define extract_archive
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) x -y $(call hostpath,$(2))
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
package_dir = $(build_output_dir)/package
spec_dir = $(package_dir)/SPECS
spec_file = $(spec_dir)/$(package_proj_name).spec
package_sources_dir = $(package_dir)/SOURCES
package_build_dir = $(package_dir)/BUILD
package_rpms_dir = $(package_dir)/RPMS
package_srpms_dir = $(package_dir)/SRPMS
pkgconfig_file_name = $(package_proj_name).pc
pkgconfig_file = $(package_dir)/$(pkgconfig_file_name)

target_path = $(build_output_dir)/$(target_os_family)-$(target_processor_family)-$(target_word_size)/$(target_build_type)

package_stage_parent_dir = $(package_dir)/stage
package_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)
package_bin_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)/$(target_os_family)-$(target_processor_family)-$(target_word_size)/$(target_build_type)
package_lib_stage_dir = $(package_bin_stage_dir)/lib
package_shared_lib_stage_dir = $(package_lib_stage_dir)/shared
package_static_lib_stage_dir = $(package_lib_stage_dir)/static
package_util_stage_dir = $(package_bin_stage_dir)/util
package_inc_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)/include
src_package_prefix = $(package_proj_name_and_ver)
bin_package_prefix = $(package_proj_name_and_ver)-$(target_os_family)-$(target_processor_family)-$(target_word_size)-bin

ifeq ($(package_archive_format),tar)
	src_package_name=$(src_package_prefix).tar.gz
	bin_package_name=$(bin_package_prefix).tar.gz
else
	src_package_name=$(src_package_prefix).zip
	bin_package_name=$(bin_package_prefix).zip
endif

java_output_dir = $(target_path)/java
java_class_output_dir = $(target_path)/java/classes

ifdef JAVA_HOME
	jhome := $(subst \,/,$(JAVA_HOME))
endif

inc_dirs = src util

ifdef jhome
	jni_src_dir = java/jni
	inc_dirs += $(subst $(sp),@,"$(jhome)/include")
	inc_dirs += $(subst $(sp),@,"$(jhome)/include/win32")
	inc_dirs += $(subst $(sp),@,"$(jhome)/include/linux")
	inc_dirs += $(jni_src_dir)
endif

xflaim_jni_jar = $(java_output_dir)/xflaimjni.jar

util_dir = $(target_path)/util
test_dir = $(target_path)/test
sample_dir = $(target_path)/sample
lib_dir = $(target_path)/$(lib_dir_name)
shared_lib_dir = $(lib_dir)/shared
static_lib_dir = $(lib_dir)/static

util_obj_dir = $(util_dir)/obj
test_obj_dir = $(test_dir)/obj
sample_obj_dir = $(sample_dir)/obj

lib_obj_dir = $(lib_dir)/obj

# -- Utility variables --

em :=
sp := $(em) $(em)
percent := \045
dollar := \044
question := \077
asterisk := \052
dash := \055

# -- Tools --

libr =
linker =
compiler =

ifdef unix_target
	gprintf = printf
else
	gprintf = $(call hostpath,$(tooldir)/printf.exe)
endif

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
	linker = link.exe
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
	exe_link_libs =

	# Convert the list of defines into a proper set of command-line params

	ifdef ccdefs
		ccdefine = $(foreach def,$(strip $(ccdefs)),/D$(def))
	endif

	# Same thing for the include dirs

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),/I$(subst @,$(sp),$(call hostpath,$(inc_dir))))

	# Concatenate everything into the ccflags variable

	ccflags += $(ccdefine) $(ccinclude)
endif

##############################################################################
#   Linux/Unix settings
##############################################################################
ifdef unix_target
	exe_suffix =
	obj_suffix = .o
	lib_prefix = lib
	static_lib_suffix = .a
	shared_lib_suffix = .so

	ifneq ($(so_age),0)
		suffix_version = .$(so_current).$(so_revision).$(so_age)
	else
		ifneq ($(so_revision),0)
			suffix_version = .$(so_current).$(so_revision)
		else
			suffix_version = .$(so_current)
		endif
	endif
	compiler = g++
	linker = g++
	
	ifeq ($(target_os_family),osx)
		libr = libtool
	else
		libr = ar
	endif
	
	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),solaris)
			compiler = CC
			linker = CC
		endif
	endif
	
	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),aix)
			compiler = xlC_r
			linker = xlC_r
		endif
	endif
	
	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),hpux)
			compiler = aCC
			linker = aCC
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
				
				ccflags += -xarch=generic 
				ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
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

	ifeq ($(target_os_family),osx)
		ccdefs += OSX
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

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),-I$(subst @,$(sp),$(inc_dir)))

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
	exe_link_libs =

	ifeq ($(target_os_family),linux)
		lib_link_libs += -lrt -lstdc++ -ldl
		exe_link_libs += -lncurses
		shared_link_flags += -shared -Wl,-Bsymbolic -fpic \
			-Wl,-soname,$(@F).1 -o $@
	else
		exe_link_libs += -lcurses
	endif

	ifeq ($(target_os_family),solaris)
		link_flags += -R /usr/lib/lwp
		ifeq ($(usenativecc),yes)
			ifeq ($(target_word_size),64)
				link_flags += -xarch=generic64
			else
				link_flags += -xarch=generic
			endif
		endif

		lib_link_libs += -lm -lc -ldl -lsocket -lnsl -lrt
	endif

	ifeq ($(target_os_family),aix)
		lib_link_libs += -lm -lc
	endif
	
	ifeq ($(target_os_family),hpux)
		lib_link_libs += -lm -lc -lrt
	endif
	
	ifeq ($(target_os_family),osx)
		lib_link_libs += -lstdc++ -ldl
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
	linker = "$(call normpath,$(strip $(wc_dir)))/binnt/wlink.exe"
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
		
		$(ec)$(gprintf) "file $(subst $(sp),\nfile ,$(call ppath,$(3)))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib)\n" >> $(4)
		$(ec)$(gprintf) "library $(call ppath,$(xflaim_static_lib))\n" >> $(4)
   endef
	
   define flm_exe_link_cmd
		$(call make_lis_file_cmd,$(1),$(2),$(3),$(call hostpath,$(1)/$(2).lis))
		$(ec)$(call hostpath,$(linker)) @$(call hostpath,$(1)/$(2).lis)
		$(ec)$(call rmcmd,$(target_path)/$(1).lis)
   endef
	
endif

# -- File lists --

xflaim_src = \
	$(patsubst src/%.cpp,%.cpp,$(wildcard src/*.cpp))

xflaim_jni_src = \
	$(patsubst java/jni/%.cpp,%.cpp,$(wildcard java/jni/*.cpp))
	
util_common_src = \
	flm_dlst.cpp \
	flm_lutl.cpp \
	ftx.cpp \
	sharutil.cpp

checkdb_src = \
	checkdb.cpp \
	$(util_common_src)

rebuild_src = \
	rebuild.cpp \
	$(util_common_src)

view_src = \
	view.cpp \
	viewblk.cpp \
	viewdisp.cpp \
	viewedit.cpp \
	viewhdr.cpp \
	viewlfil.cpp \
	viewmenu.cpp \
	viewsrch.cpp \
	$(util_common_src)
	
sample_src = \
	sample.cpp \
	$(util_common_src)

xshell_src = \
	fdomedt.cpp \
	xshell.cpp \
	fshell.cpp \
	$(util_common_src)
	
ut_basictest_src = \
	flmunittest.cpp \
	basictestsrv.cpp
	
ut_binarytest_src = \
	flmunittest.cpp \
	binarytest.cpp
	
ut_colldeftest_src = \
	flmunittest.cpp \
	colldeftestsrv.cpp
	
ut_dictchangetest_src = \
	flmunittest.cpp \
	dictchangetest.cpp
	
ut_dictdeftest_src = \
	flmunittest.cpp \
	dictdeftestsrv.cpp
	
ut_dirtyexittest1_src = \
	flmunittest.cpp \
	dirtyexittest1srv.cpp
	
ut_dirtyexittest2_src = \
	flmunittest.cpp \
	dirtyexittest2srv.cpp
	
ut_domnodetest_src = \
	flmunittest.cpp \
	domnodetestsrv.cpp
	
ut_enctest_src = \
	flmunittest.cpp \
	enctestsrv.cpp
	
ut_importtest_src = \
	flmunittest.cpp \
	importtestsrv.cpp
	
ut_indexdeftest_src = \
	flmunittest.cpp \
	indexdeftestsrv.cpp
	
ut_indextest1_src = \
	flmunittest.cpp \
	indextest1srv.cpp
	
ut_indextest2_src = \
	flmunittest.cpp \
	indextest2srv.cpp
	
ut_indextest3_src = \
	flmunittest.cpp \
	indextest3.cpp
	
ut_metaphonetest_src = \
	flmunittest.cpp \
	metaphonetestsrv.cpp
	
ut_namespacetest_src = \
	flmunittest.cpp \
	namespacetestsrv.cpp
	
ut_regressiontest_src = \
	flmunittest.cpp \
	regressiontest.cpp
	
ut_rfltest_src = \
	flmunittest.cpp \
	rfltestsrv.cpp
	
ut_sortkeytest_src = \
	flmunittest.cpp \
	sortkeytest.cpp
	
ut_sortkeytest2_src = \
	flmunittest.cpp \
	sortkeytest2.cpp
	
ut_xpathtest1_src = \
	flmunittest.cpp \
	xpathtest1srv.cpp
	
ut_xpathtest2_src = \
	flmunittest.cpp \
	xpathtest2srv.cpp
	
# -- XFLAIM library --

xflaim_obj = $(patsubst %.cpp,$(lib_obj_dir)/%$(obj_suffix),$(xflaim_src))
xflaim_jni_obj = $(patsubst %.cpp,$(lib_obj_dir)/%$(obj_suffix),$(xflaim_jni_src))
xflaim_static_lib = $(static_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
ifndef netware_target
   xflaim_shared_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)$(suffix_version)
   xflaim_shared_imp_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
endif

# -- Unit tests  --

ut_basictest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_basictest_src))
ut_binarytest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_binarytest_src))
ut_colldeftest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_colldeftest_src))
ut_dictchangetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dictchangetest_src))
ut_dictdeftest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dictdeftest_src))
ut_dirtyexittest1_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dirtyexittest1_src))
ut_dirtyexittest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dirtyexittest2_src))
ut_domnodetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_domnodetest_src))
ut_enctest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_enctest_src))
ut_importtest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_importtest_src))
ut_indexdeftest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indexdeftest_src))
ut_indextest1_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indextest1_src))
ut_indextest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indextest2_src))
ut_indextest3_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indextest3_src))
ut_metaphonetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_metaphonetest_src))
ut_namespacetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_namespacetest_src))
ut_regressiontest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_regressiontest_src))
ut_rfltest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_rfltest_src))
ut_sortkeytest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_sortkeytest_src))
ut_sortkeytest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_sortkeytest2_src))
ut_xpathtest1_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_xpathtest1_src))
ut_xpathtest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_xpathtest2_src))

# -- Java --

jni_java_src = $(wildcard java/xflaim/*.java)

# -- Utilities  --

checkdb_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(checkdb_src))
checkdb_exe = $(util_dir)/checkdb$(exe_suffix)

rebuild_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(rebuild_src))
rebuild_exe = $(util_dir)/rebuild$(exe_suffix)

view_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(view_src))
view_exe = $(util_dir)/view$(exe_suffix)

xshell_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(xshell_src))
xshell_exe = $(util_dir)/xshell$(exe_suffix)

sample_obj = $(patsubst %.cpp,$(sample_obj_dir)/%$(obj_suffix),$(sample_src))
sample_exe = $(sample_dir)/sample$(exe_suffix)

# -- Make system pattern search paths -- 

vpath %.cpp src util sample java/jni 
vpath %.java jni/java/xflaim

# -- Default target --

.PHONY : libs
ifeq ($(clean_only),0)
libs: status clean dircheck $(xflaim_static_lib) $(xflaim_shared_lib)
endif

# -- JNI (Java) --

ifeq ($(clean_only),0)
ifdef jhome
$(xflaim_jni_jar) : $(jni_java_src)
	$(ec)$(gprintf) "Compiling JNI java sources ...\n"
	$(ec)$(call hostpath,$(jhome)/bin/javac) -g -d $(java_class_output_dir) $(call hostpath,$(jni_java_src))
	$(ec)$(gprintf) "Creating JNI jar file ...\n"
	$(ec)$(call rmcmd,$(xflaim_jni_jar))
	$(ec)$(call hostpath,$(jhome)/bin/jar) cf $(call hostpath,$(xflaim_jni_jar)) \
		-C $(call hostpath,$(java_class_output_dir)) .
	$(ec)$(gprintf) "Creating JNI header files ...\n"
	$(ec)$(call hostpath,$(jhome)/bin/javah) -d $(jni_src_dir) \
		-classpath $(call hostpath,$(xflaim_jni_jar)) \
		$(foreach jclass,$(notdir $(basename $(jni_java_src))),xflaim.$(jclass))
endif
endif

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

# -- xflaim.lib and libxflaim.a --

ifeq ($(clean_only),0)
$(xflaim_static_lib) : $(xflaim_obj)
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
	$(ec)$(call hostpath,$(libr)) $(libflags) $(call hostpath,$(xflaim_static_lib)) @$(call hostpath,$(static_lib_dir)/flmlib.lis)
endif
endif

# -- xflaim.dll and libxflaim.so --

ifeq ($(clean_only),0)
$(xflaim_shared_lib) : $(xflaim_obj) $(xflaim_jni_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(linker) $(call hostpath,$+) $(shared_link_flags) $(lib_link_libs)
endif
ifdef unix_target
	$(ec)rm -f $@
ifeq ($(target_os_family),linux)
	$(ec)$(linker) $+ $(shared_link_flags) $(lib_link_libs)
endif
endif
endif

# -- xflaim.lib import library --

ifeq ($(clean_only),0)
$(xflaim_shared_imp_lib) : $(xflaim_shared_lib)
	$(ec)$(gprintf) "Building $@ ...\n"
endif

# -- Utility link command --

ifndef flm_exe_link_cmd
   define flm_exe_link_cmd
		$(ec)$(linker) $(exe_link_flags) $(allprereqs) $(lib_link_libs) $(exe_link_libs)
   endef
endif

# -- checkdb --

.PHONY : checkdb
ifeq ($(clean_only),0)
checkdb: status clean dircheck libs $(checkdb_exe)
$(checkdb_exe): $(checkdb_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),checkdb,$(checkdb_obj))
endif

# -- rebuild --

.PHONY : rebuild
ifeq ($(clean_only),0)
rebuild: status clean dircheck libs $(rebuild_exe)
$(rebuild_exe): $(rebuild_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),rebuild,$(rebuild_obj))
endif

# -- view --

.PHONY : view
ifeq ($(clean_only),0)
view: status clean dircheck libs $(view_exe)
$(view_exe): $(view_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),view,$(view_obj))
endif

# -- sample --

.PHONY : sample
ifeq ($(clean_only),0)
sample: status clean dircheck libs $(sample_exe)
$(sample_exe): $(sample_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(sample_dir),dbshell,$(sample_obj))
	$(ec)$(call copycmd,sample/xmlfiles/*.xml,$(sample_dir))
endif

# -- xshell --

.PHONY : xshell
ifeq ($(clean_only),0)
xshell: status clean dircheck libs $(xshell_exe)
$(xshell_exe): $(xshell_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(sample_dir),sample,$(sample_obj))
endif

# -- basictest --

.PHONY : basictest
ifeq ($(clean_only),0)
basictest: status clean dircheck $(test_dir)/basictest$(exe_suffix)
$(test_dir)/basictest$(exe_suffix): $(ut_basictest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),basictest,$(ut_basictest_obj))
endif

# -- binarytest --

.PHONY : binarytest
ifeq ($(clean_only),0)
binarytest: status clean dircheck $(test_dir)/binarytest$(exe_suffix)
$(test_dir)/binarytest$(exe_suffix): $(ut_binarytest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),binarytest,$(ut_binarytest_obj))
endif

# -- colldeftest --

.PHONY : colldeftest
ifeq ($(clean_only),0)
colldeftest: status clean dircheck $(test_dir)/colldeftest$(exe_suffix)
$(test_dir)/colldeftest$(exe_suffix): $(ut_colldeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),colldeftest,$(ut_colldeftest_obj))
endif

# -- dictchangetest --

.PHONY : dictchangetest
ifeq ($(clean_only),0)
dictchangetest: status clean dircheck $(test_dir)/dictchangetest$(exe_suffix)
$(test_dir)/dictchangetest$(exe_suffix): $(ut_dictchangetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dictchangetest,$(ut_dictchangetest_obj))
endif

# -- dictdeftest --

.PHONY : dictdeftest
ifeq ($(clean_only),0)
dictdeftest: status clean dircheck $(test_dir)/dictdeftest$(exe_suffix)
$(test_dir)/dictdeftest$(exe_suffix): $(ut_dictdeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dictdeftest,$(ut_dictdeftest_obj))
endif

# -- dirtyexittest1 --

.PHONY : dirtyexittest1
ifeq ($(clean_only),0)
dirtyexittest1: status clean dircheck $(test_dir)/dirtyexittest1$(exe_suffix)
$(test_dir)/dirtyexittest1$(exe_suffix): $(ut_dirtyexittest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dirtyexittest1,$(ut_dirtyexittest1_obj))
endif

# -- dirtyexittest2 --

.PHONY : dirtyexittest2
ifeq ($(clean_only),0)
dirtyexittest2: status clean dircheck $(test_dir)/dirtyexittest2$(exe_suffix)
$(test_dir)/dirtyexittest2$(exe_suffix): $(ut_dirtyexittest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dirtyexittest2,$(ut_dirtyexittest2_obj))
endif

# -- domnodetest --

.PHONY : domnodetest
ifeq ($(clean_only),0)
domnodetest: status clean dircheck $(test_dir)/domnodetest$(exe_suffix)
$(test_dir)/domnodetest$(exe_suffix): $(ut_domnodetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),domnodetest,$(ut_domnodetest_obj))
endif

# -- enctest --

.PHONY : enctest
ifeq ($(clean_only),0)
enctest: status clean dircheck $(test_dir)/enctest$(exe_suffix)
$(test_dir)/enctest$(exe_suffix): $(ut_enctest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),enctest,$(ut_enctest_obj))
endif

# -- importtest --

.PHONY : importtest
ifeq ($(clean_only),0)
importtest: status clean dircheck $(test_dir)/importtest$(exe_suffix)
$(test_dir)/importtest$(exe_suffix): $(ut_importtest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),importtest,$(ut_importtest_obj))
endif

# -- indexdeftest --

.PHONY : indexdeftest
ifeq ($(clean_only),0)
indexdeftest: status clean dircheck $(test_dir)/indexdeftest$(exe_suffix)
$(test_dir)/indexdeftest$(exe_suffix): $(ut_indexdeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indexdeftest,$(ut_indexdeftest_obj))
endif

# -- indextest1 --

.PHONY : indextest1
ifeq ($(clean_only),0)
indextest1: status clean dircheck $(test_dir)/indextest1$(exe_suffix)
$(test_dir)/indextest1$(exe_suffix): $(ut_indextest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indextest1,$(ut_indextest1_obj))
endif

# -- indextest2 --

.PHONY : indextest2
ifeq ($(clean_only),0)
indextest2: status clean dircheck $(test_dir)/indextest2$(exe_suffix)
$(test_dir)/indextest2$(exe_suffix): $(ut_indextest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indextest2,$(ut_indextest2_obj))
endif

# -- indextest3 --

.PHONY : indextest3
ifeq ($(clean_only),0)
indextest3: status clean dircheck $(test_dir)/indextest3$(exe_suffix)
$(test_dir)/indextest3$(exe_suffix): $(ut_indextest3_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indextest3,$(ut_indextest3_obj))
endif

# -- metaphonetest --

.PHONY : metaphonetest
ifeq ($(clean_only),0)
metaphonetest: status clean dircheck $(test_dir)/metaphonetest$(exe_suffix)
$(test_dir)/metaphonetest$(exe_suffix): $(ut_metaphonetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),metaphonetest,$(ut_metaphonetest_obj))
endif

# -- namespacetest --

.PHONY : namespacetest
ifeq ($(clean_only),0)
namespacetest: status clean dircheck $(test_dir)/namespacetest$(exe_suffix)
$(test_dir)/namespacetest$(exe_suffix): $(ut_namespacetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),namespacetest,$(ut_namespacetest_obj))
endif

# -- regressiontest --

.PHONY : regressiontest
ifeq ($(clean_only),0)
regressiontest: status clean dircheck $(test_dir)/regressiontest$(exe_suffix)
$(test_dir)/regressiontest$(exe_suffix): $(ut_regressiontest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),regressiontest,$(ut_regressiontest_obj))
endif

# -- rfltest --

.PHONY : rfltest
ifeq ($(clean_only),0)
rfltest: status clean dircheck $(test_dir)/rfltest$(exe_suffix)
$(test_dir)/rfltest$(exe_suffix): $(ut_rfltest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),rfltest,$(ut_rfltest_obj))
endif

# -- sortkeytest --

.PHONY : sortkeytest
ifeq ($(clean_only),0)
sortkeytest: status clean dircheck $(test_dir)/sortkeytest$(exe_suffix)
$(test_dir)/sortkeytest$(exe_suffix): $(ut_sortkeytest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),sortkeytest,$(ut_sortkeytest_obj))
endif

# -- sortkeytest2 --

.PHONY : sortkeytest2
ifeq ($(clean_only),0)
sortkeytest2: status clean dircheck $(test_dir)/sortkeytest2$(exe_suffix)
$(test_dir)/sortkeytest2$(exe_suffix): $(ut_sortkeytest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),sortkeytest2,$(ut_sortkeytest2_obj))
endif

# -- xpathtest1 --

.PHONY : xpathtest1
ifeq ($(clean_only),0)
xpathtest1: status clean dircheck $(test_dir)/xpathtest1$(exe_suffix)
$(test_dir)/xpathtest1$(exe_suffix): $(ut_xpathtest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),xpathtest1,$(ut_xpathtest1_obj))
endif

# -- xpathtest2 --

.PHONY : xpathtest2
ifeq ($(clean_only),0)
xpathtest2: status clean dircheck $(test_dir)/xpathtest2$(exe_suffix)
$(test_dir)/xpathtest2$(exe_suffix): $(ut_xpathtest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),xpathtest2,$(ut_xpathtest2_obj))
endif

# -- java (JNI jar file) --

.PHONY : java
ifeq ($(clean_only),0)
java: status clean dircheck $(xflaim_jni_jar) $(xflaim_shared_lib)
endif

# -- version --

define make_version_files
	$(ec)$(gprintf) "$(version)" > $(1)/VERSION
	$(ec)$(gprintf) " " > $(1)/SVNRevision.$(svn_revision)
	$(ec)$(gprintf) "Version files created.\n"
endef

# -- srcdist --

.PHONY : srcdist
ifeq ($(clean_only),0)
srcdist: status dircheck docs spec
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
	$(ec)$(call dircopycmd,sample,$(package_stage_dir)/sample)
	$(ec)$(call dircopycmd,java,$(package_stage_dir)/java)
	$(ec)$(call dircopycmd,$(docs_output_dir),$(package_stage_dir)/docs)
	$(ec)$(call dircopycmd,$(dir $(topdir))tools,$(package_stage_dir)/tools)
ifneq ($(host_os_family),win)
	-$(ec)rm -rf `find $(package_stage_dir) -name .svn`
endif
ifeq ($(host_os_family),win)
	$(ec)$(call copycmd,make.exe,$(package_stage_dir))
endif
	$(ec)$(call create_archive,$(package_stage_parent_dir), \
		$(package_dir)/$(src_package_name), \
		$(package_proj_name_and_ver))
	$(ec)$(call copycmd,$(package_proj_name).changes,$(package_dir))
	$(ec)$(call rmdircmd,$(package_stage_dir))
	$(ec)$(gprintf) "Source package created.\n"
endif

# -- bindist --

.PHONY : bindist
ifeq ($(clean_only),0)
bindist: status dircheck all binpackage
	$(ec)$(gprintf) ""
endif

.PHONY : binpackage
ifeq ($(clean_only),0)
binpackage: status clean
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
	$(ec)$(call copycmd,src/xflaim.h,$(package_inc_stage_dir))
	$(ec)$(call copycmd,$(xflaim_static_lib),$(package_static_lib_stage_dir))
ifdef xflaim_shared_lib
	$(ec)$(call copycmd,$(xflaim_shared_lib),$(package_shared_lib_stage_dir))
endif
ifdef win_target
	$(ec)$(call copycmd,$(xflaim_shared_imp_lib),$(package_shared_lib_stage_dir))
endif
	$(ec)$(call copycmd,$(checkdb_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(rebuild_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(view_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(dbshell_exe),$(package_util_stage_dir))
	$(ec)$(call create_archive,$(package_stage_parent_dir), \
		$(package_dir)/$(bin_package_name), \
		$(package_proj_name_and_ver))
	$(ec)$(call rmdircmd,$(package_stage_parent_dir))
	$(ec)$(gprintf) "Binary package created.\n"
endif

# -- dist --

.PHONY : dist
ifeq ($(clean_only),0)
dist: status dircheck srcdist
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating distribution (SVN revision $(svn_revision)) ...\n"
   endif
	$(ec)$(call extract_archive,$(package_dir),$(src_package_name))
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) clean
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) $(submake_targets) all
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) $(submake_targets) binpackage package_dir="$(package_dir)"
	$(ec)$(call rmdircmd,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(gprintf) "Distribution created.\n"
endif

# -- changelog --

.PHONY : changelog
ifeq ($(clean_only),0)
changelog:
	$(ec)$(gprintf) "Creating change log for SVN revisions $(svn_low_rev)-$(svn_high_rev) ...\n"
	$(ec)$(gprintf) "Using SVN user $(svn_user) ...\n"
	$(ec)$(gprintf) "Using SVN URL $(svnurl) ...\n"
	$(ec)svn log $(svnurl) -v -r $(svn_low_rev):$(svn_high_rev) > $(package_sources_dir)/$(package_proj_name_and_ver).tar.log
	$(ec)$(gprintf) "Change log created.\n"
endif

# -- install --

.PHONY : install
ifeq ($(clean_only),0)
install: libs pkgconfig
ifneq ($(host_os_family),win)
	$(ec)$(gprintf) "Installing ...\n"
	mkdir -p $(lib_install_dir)/pkgconfig
	mkdir -p $(include_install_dir)
	install --mode=644 $(xflaim_shared_lib) $(lib_install_dir)
	install --mode=644 $(xflaim_static_lib) $(lib_install_dir)
	install --mode=644 $(pkgconfig_file) $(pkgconfig_install_dir)
	install --mode=644 src/xflaim.h $(include_install_dir)
	-ldconfig $(lib_install_dir) -l -v $(lib_install_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)$(suffix_version)
	$(ec)$(gprintf) "Installation complete.\n"
endif
endif

# -- uninstall --

.PHONY : uninstall
ifeq ($(clean_only),0)
uninstall: 
ifneq ($(host_os_family),win)
	$(ec)$(gprintf) "Uninstalling ...\n"
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)*
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
	-rm -rf $(pkgconfig_install_dir)/$(pkgconfig_file_name)
	-rm -rf $(include_install_dir)/xflaim.h
	$(ec)$(gprintf) "Uninstalled.\n"
endif
endif

# -- spec file --

.PHONY : spec
ifeq ($(clean_only),0)
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
	$(ec)$(gprintf) "XFLAIM is an embeddable cross-platform XML database engine that\n" >> $(spec_file)
	$(ec)$(gprintf) "provides a rich, powerful, easy-to-use feature set. It has proven to be\n" >> $(spec_file)
	$(ec)$(gprintf) "highly scalable, reliable, and robust. It is available on a wide\n" >> $(spec_file)
	$(ec)$(gprintf) "variety of 32 bit and 64 bit platforms.\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "Authors:\n" >> $(spec_file)
	$(ec)$(gprintf) "$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)\n" >> $(spec_file)
	$(ec)$(gprintf) "    Daniel Sanders\n" >> $(spec_file)
	$(ec)$(gprintf) "    Andrew Hodgkinson\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)package devel\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: XFLAIM static library and header file\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "Provides: $(package_proj_name)-devel\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description devel\n" >> $(spec_file)
	$(ec)$(gprintf) "XFLAIM is an embeddable cross-platform XML database engine that\n" >> $(spec_file)
	$(ec)$(gprintf) "provides a rich, powerful, easy-to-use feature set. It has proven to be\n" >> $(spec_file)
	$(ec)$(gprintf) "highly scalable, reliable, and robust. It is available on a wide\n" >> $(spec_file)
	$(ec)$(gprintf) "variety of 32 bit and 64 bit platforms.\n" >> $(spec_file)
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
	$(ec)$(gprintf) "make lib_dir_name=$(percent){_lib} libs\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)install\n" >> $(spec_file)
	$(ec)$(gprintf) "make rpm_build_root=$(dollar)RPM_BUILD_ROOT install_prefix=$(percent){prefix} lib_dir_name=$(percent){_lib} install\n" >> $(spec_file)
	$(ec)$(gprintf) "rm -rf $(build_output_dir)\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)clean\n" >> $(spec_file)
	$(ec)$(gprintf) "rm -rf $(dollar)RPM_BUILD_ROOT\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)defattr(-,root,root)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)doc COPYING VERSION\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(shared_lib_suffix)$(asterisk)\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files devel\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(static_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/pkgconfig/$(pkgconfig_file_name)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/include/xflaim.h\n" >> $(spec_file)
	$(ec)$(gprintf) "Created spec file.\n"
endif

# -- PKG-CONFIG --

.PHONY : pkgconfig
ifeq ($(clean_only),0)
pkgconfig: dircheck
	$(ec)$(gprintf) "prefix=$(install_prefix)\n" > $(pkgconfig_file)
	$(ec)$(gprintf) "exec_prefix=$(dollar){prefix}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "libdir=$(dollar){exec_prefix}/$(lib_dir_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "includedir=$(dollar){prefix}/include\n\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Name: $(package_proj_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Description: $(project_desc)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Libs: $(lib_link_libs) -lxflaim -L$(dollar){libdir}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Cflags: -I$(dollar){includedir}\n" >> $(pkgconfig_file)
endif

# -- SRCRPM --

.PHONY : srcrpm
ifeq ($(clean_only),0)
srcrpm: dist spec
	$(ec)$(gprintf) "Creating source RPM ...\n"
	$(ec)rpmbuild --define="_topdir $(homedir)/$(package_dir)" --quiet -bs $(spec_file)
	$(ec)$(gprintf) "Source RPM created.\n"
endif

# -- RPMS --

.PHONY : rpms
ifeq ($(clean_only),0)
rpms: dist spec
	$(ec)$(gprintf) "Creating source and binary RPMs ...\n"
	$(ec)rpmbuild --define="_topdir $(homedir)/$(package_dir)" --quiet -ba $(spec_file)
	$(ec)find $(package_dir) -name *.rpm | xargs chmod 775
	$(ec)$(gprintf) "Source and binary RPMs created.\n"
endif

# -- Documentation --

.PHONY : docs
ifeq ($(clean_only),0)
docs: status dircheck
	$(ec)$(gprintf) "Creating documentation ...\n"
	$(ec)doxygen Doxyfile
	$(ec)$(gprintf) "Documentation created.\n"
endif

# -- misc. targets --

.PHONY : status
ifeq ($(clean_only),0)
status:
	$(ec)$(gprintf) "===============================================================================\n"
	$(ec)$(gprintf) "SVN Revision ................... $(revision)\n"
	$(ec)$(gprintf) "Host Operating System Family ... $(host_os_family)\n"
	$(ec)$(gprintf) "Top Directory .................. $(call ppath,$(topdir))\n"
	$(ec)$(gprintf) "Target Operating System Family . $(target_os_family)\n"
	$(ec)$(gprintf) "Target Processor Family ........ $(target_processor_family)\n"
	$(ec)$(gprintf) "Target Word Size ............... $(target_word_size)\n"
	$(ec)$(gprintf) "Target Build Type .............. $(target_build_type)\n"
	$(ec)$(gprintf) "Target Path .................... $(call ppath,$(target_path))\n"
	$(ec)$(gprintf) "Compiler ....................... $(call ppath,$(compiler))\n"
	$(ec)$(gprintf) "Librarian ...................... $(call ppath,$(libr))\n"
	$(ec)$(gprintf) "Java Home ...................... $(call ppath,$(jhome))\n"
	$(ec)$(gprintf) "Defines ........................ $(strip $(ccdefs))\n"
	$(ec)$(gprintf) "===============================================================================\n"
endif

.PHONY : dircheck
ifeq ($(clean_only),0)
dircheck:
	$(ec)$(call mkdircmd,$(util_obj_dir))
	$(ec)$(call mkdircmd,$(test_obj_dir))
	$(ec)$(call mkdircmd,$(sample_obj_dir))
	$(ec)$(call mkdircmd,$(lib_obj_dir))
	$(ec)$(call mkdircmd,$(java_output_dir))
	$(ec)$(call mkdircmd,$(java_class_output_dir))
	$(ec)$(call mkdircmd,$(util_dir))
	$(ec)$(call mkdircmd,$(test_dir))
	$(ec)$(call mkdircmd,$(sample_dir))
	$(ec)$(call mkdircmd,$(static_lib_dir))
	$(ec)$(call mkdircmd,$(shared_lib_dir))
	$(ec)$(call mkdircmd,$(package_dir))
	$(ec)$(call mkdircmd,$(spec_dir))
	$(ec)$(call mkdircmd,$(package_sources_dir))
	$(ec)$(call mkdircmd,$(package_build_dir))
	$(ec)$(call mkdircmd,$(package_rpms_dir))
	$(ec)$(call mkdircmd,$(package_srpms_dir))
endif

# -- phony targets --

.PHONY : all
ifeq ($(clean_only),0)
all: libs allutils
	$(ec)$(gprintf) ""
endif

.PHONY : allutils
ifeq ($(clean_only),0)
allutils: status dircheck libs $(util_targets)
	$(ec)$(gprintf) ""
endif

.PHONY : test
ifeq ($(clean_only),0)
test:	status dircheck $(xflaim_static_lib) $(test_targets)
	$(ec)$(call copycmd,util/xmlfiles/*.xml,$(test_dir))
	$(ec)$(call runtest,basictest)
	$(ec)$(call runtest,binarytest)
	$(ec)$(call runtest,colldeftest)
	$(ec)$(call runtest,dictchangetest)
	$(ec)$(call runtest,dictdeftest)
	$(ec)$(call runtest,dirtyexittest1)
	$(ec)$(call runtest,dirtyexittest2)
	$(ec)$(call runtest,domnodetest)
	$(ec)$(call runtest,enctest)
	$(ec)$(call runtest,importtest)
	$(ec)$(call runtest,indexdeftest)
	$(ec)$(call runtest,indextest1)
	$(ec)$(call runtest,indextest2)
	$(ec)$(call runtest,indextest3)
	$(ec)$(call runtest,namespacetest)
	$(ec)$(call runtest,metaphonetest)
	$(ec)$(call runtest,regressiontest)
	$(ec)$(call runtest,rfltest)
	$(ec)$(call runtest,sortkeytest)
	$(ec)$(call runtest,sortkeytest2)
	$(ec)$(call runtest,xpathtest1)
	$(ec)$(call runtest,xpathtest2)
endif

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
	
.PHONY : mostlyclean
mostlyclean : clean
	$(ec)$(gprintf) ""

.PHONY : clean
clean: status
ifeq ($(do_clean),1)
	$(ec)$(gprintf) "\n"
	$(ec)$(gprintf) "Cleaning $(target_path) ...\n"
	-$(ec)$(call rmdircmd,$(target_path))
	-$(ec)$(call rmcmd,*.pch)
	$(ec)$(gprintf) "\n"
endif
	
.PHONY : distclean
	-$(ec)$(call rmcmd,*.pch)
	
.PHONY : maintainer-clean
maintainer-clean:
	-$(ec)$(call rmdircmd,$(build_output_dir))
	-$(ec)$(call rmcmd,*.pch)
