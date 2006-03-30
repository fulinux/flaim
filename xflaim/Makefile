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
	ignore_local_mods = 1
endif

ifneq (,$(findstring ilm,$(MAKECMDGOALS)))
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

ifneq (,$(findstring zips,$(MAKECMDGOALS)))
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
	
	svnurl0 = $(shell svn info)
	svnurl1 = $(subst URL: ,URL:,$(svnurl0))
	svnurl2 = $(filter URL:%,$(svnurl1))
	svnurl3 = $(subst URL:,,$(svnurl2))
	svnurl = $(subst ://,://$(svn_user)@,$(svnurl3))
endif
	
ifdef calc_svn_revision

	# Get the info for all files.
	
	ifndef local_mods_ok	
		srevision = $(shell svnversion . -n)

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
	revision0 = $(shell svn info -R)
	revision1 = $(subst Last Changed Rev: ,LastChangedRev:,$(revision0))
	revision2 = $(filter LastChangedRev:%,$(revision1))
	revision3 = $(subst LastChangedRev:,,$(revision2))
	revision4 = $(sort $(revision3))
	revision5 = $(foreach num,$(revision4),$(call numdigits,$(num)):$(num))
	revision6 = $(sort $(revision5))
	revision7 = $(word $(words $(revision6)),$(revision6))
	revision = $(word 2,$(subst :, ,$(revision7)))
	
else
	ifeq "$(wildcard SVNRevision.*)" ""
		revision = 0
	else
		revision = $(word 2,$(subst ., ,$(wildcard SVNRevision.*)))
	endif
endif

ifeq "$(svn_high_rev)" ""
	svn_high_rev = $(revision)
endif

version = $(major_version).$(minor_version).$(revision)

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
target_processor =
target_word_size =
requested_target_word_size =
win_target =
unix_target =

# -- Enable command echoing --

ifneq (,$(findstring verbose,$(MAKECMDGOALS)))
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
   $(error Host operating system could not be determined.  You may need to export HOSTTYPE from the environment.)
endif

# -- Target build type --

ifndef target_build_type
	ifneq (,$(findstring debug,$(MAKECMDGOALS)))
		target_build_type = debug
	endif
endif

ifndef target_build_type
	ifneq (,$(findstring release,$(MAKECMDGOALS)))
		target_build_type = release
	endif
endif

ifndef target_build_type
	target_build_type = release
endif

# -- Use non-native (i.e., gcc) compiler on Solaris, etc.

ifneq (,$(findstring usegcc,$(MAKECMDGOALS)))
	usenativecc = no
endif

# -- Override platform default word size? --

ifneq (,$(findstring 64bit,$(MAKECMDGOALS)))
	requested_target_word_size = 64
endif

ifneq (,$(findstring 32bit,$(MAKECMDGOALS)))
	requested_target_word_size = 32
endif

# -- Target operating system and processor architecture --

ifndef target_os_family
	ifeq ($(host_os_family),linux)
		unix_target = yes
		target_os_family = linux
		
		ifeq (,$(HOSTTYPE))
			ifneq (,$(RPM_ARCH))
				HOSTTYPE = $(RPM_ARCH)
			endif
         $(error HOSTTYPE environment variable has not been set)
		endif
		
		ifeq (,$(HOSTTYPE))
         $(error HOSTTYPE environment variable has not been set)
		endif
		
		ifdef requested_target_word_size
			ifneq (,$(findstring x86,$(HOSTTYPE)))
				ifeq ($(requested_target_word_size),64)
					target_processor = x86
					target_word_size = 64
				else
					target_processor = x86
					target_word_size = 32
				endif
			else
            $(error Platform not supported)
			endif
		else
			ifneq (,$(findstring x86,$(HOSTTYPE)))
				ifneq (,$(findstring x86_64,$(HOSTTYPE)))
					target_processor = x86
					target_word_size = 64
				else
					target_processor = x86
					target_word_size = 32
				endif
			else
				ifneq (,$(findstring 86,$(HOSTTYPE)))
					target_processor = x86
					target_word_size = 32
				else
					ifneq (,$(findstring s390,$(HOSTTYPE)))
						ifneq (,$(findstring s390x,$(HOSTTYPE)))
							target_processor = s390x
							target_word_size = 64
						else
							target_processor = s390
							target_word_size = 31
						endif
					else
						ifneq (,$(findstring ia64,$(HOSTTYPE)))
							target_processor = ia64
							target_word_size = 64
						else
							ifneq (,$(findstring ppc64,$(HOSTTYPE)))
								target_processor = powerpc
								target_word_size = 64
							else
								ifneq (,$(findstring ppc,$(HOSTTYPE)))
									target_processor = powerpc
									target_word_size = 32
								else
                           $(error Platform not supported)
								endif
							endif
						endif
					endif
				endif
			endif
		endif
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),solaris)
		unix_target = yes
		target_os_family = solaris
		
		ifeq (,$(HOSTTYPE))
         $(error HOSTTYPE environment variable has not been set)
		endif
		
		ifdef requested_target_word_size
			ifneq (,$(findstring sparc,$(HOSTTYPE)))
				ifeq ($(requested_target_word_size),64)
					target_processor = sparc
					target_word_size = 64
				else
					target_processor = sparc
					target_word_size = 32
				endif
			else
				ifeq ($(requested_target_word_size),64)
					target_processor = x86
					target_word_size = 64
				else
					target_processor = x86
					target_word_size = 32
				endif
			endif
		else
			ifneq (,$(findstring sparc,$(HOSTTYPE)))
				target_processor = sparc
				target_word_size = 32
			else
				target_processor = x86
				target_word_size = 32
			endif
		endif
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),osx)
		unix_target = yes
		target_os_family = osx

		ifeq (,$(HOSTTYPE))
         $(error HOSTTYPE environment variable has not been set)
		endif
		
		ifdef requested_target_word_size
			ifneq (,$(findstring powerpc,$(HOSTTYPE)))
				ifeq ($(requested_target_word_size),64)
					target_processor = powerpc
					target_word_size = 64
				else
					target_processor = powerpc
					target_word_size = 32
				endif
			else
            $(error Platform not supported)
			endif
		else
			ifneq (,$(findstring powerpc,$(HOSTTYPE)))
				target_processor = powerpc
				target_word_size = 32
			else
            $(error Platform not supported)
			endif
		endif
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),win)
		win_target = yes
		target_os_family = win
		target_processor = x86
		target_word_size = 32
	endif
endif
	
ifndef target_os_family
   $(error Target operating system could not be determined)
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
build_output_dir = build
package_dir = $(build_output_dir)/package
spec_dir = $(package_dir)/SPECS
spec_file = $(spec_dir)/$(package_proj_name).spec
package_sources_dir = $(package_dir)/SOURCES
package_build_dir = $(package_dir)/BUILD
package_rpms_dir = $(package_dir)/RPMS
package_srpms_dir = $(package_dir)/SRPMS
package_zips_dir = $(package_dir)/ZIPS
pkgconfig_file_name = $(package_proj_name).pc
pkgconfig_file = $(package_dir)/$(pkgconfig_file_name)

target_path = $(build_output_dir)/$(target_os_family)-$(target_processor)-$(target_word_size)/$(target_build_type)

zip_stage_parent_dir = $(package_build_dir)/$(package_proj_name_and_ver)/stage
zip_bin_stage_dir = $(zip_stage_parent_dir)/$(package_proj_name_and_ver)/$(target_os_family)-$(target_processor)-$(target_word_size)/$(target_build_type)
zip_lib_stage_dir = $(zip_bin_stage_dir)/lib
zip_shared_lib_stage_dir = $(zip_lib_stage_dir)/shared
zip_static_lib_stage_dir = $(zip_lib_stage_dir)/static
zip_util_stage_dir = $(zip_bin_stage_dir)/util
zip_inc_stage_dir = $(zip_stage_parent_dir)/$(package_proj_name_and_ver)/include
zip_stage_dir = $(zip_stage_parent_dir)/$(package_proj_name_and_ver)

src_zip_name=$(package_proj_name_and_ver)-src.zip
bin_zip_name=$(package_proj_name_and_ver)-bin.zip

java_output_dir = $(target_path)/java
java_class_output_dir = $(target_path)/java/classes

ifdef JAVA_HOME
	jhome := $(subst \,/,$(JAVA_HOME))
endif

inc_dirs = src util

ifdef jhome
	jni_inc_dir = $(target_path)/java/include
	inc_dirs += $(subst $(sp),@,"$(jhome)/include")
	inc_dirs += $(subst $(sp),@,"$(jhome)/include/win32")
	inc_dirs += $(subst $(sp),@,"$(jhome)/include/linux")
	inc_dirs += $(jni_inc_dir)
endif

static_obj_dir = $(target_path)/static_obj
xflaim_jni_jar = $(java_output_dir)/xflaimjni.jar
obj_dir = $(target_path)/obj
util_dir = $(target_path)/util
test_dir = $(target_path)/test
sample_dir = $(target_path)/sample
shared_lib_dir = $(target_path)/lib/shared
static_lib_dir = $(target_path)/lib/static

# -- Utility variables --

em :=
sp := $(em) $(em)
percent := \045
dollar := \044
question := \077
asterisk := \052
dash := \055

ifeq ($(host_os_family),win)
	allprereqs  = $(subst /,\,$+)
	copycmd = copy /Y $(subst /,\,$(1)) $(subst /,\,$(2))
	dircopycmd = xcopy /Y /E /V /I $(subst /,\,$(1)) $(subst /,\,$(2))
	rmcmd = del /Q $(subst /,\,$(1)) 2>NUL
	rmdircmd = if exist $(subst /,\,$(1)) rmdir /q /s $(subst /,\,$(1)) 2>NUL
	mkdircmd = -if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1))
	runtest = cmd /C "cd $(subst /,\,$(test_dir)) && $(1) -d"
	homedir := $(subst \,/,$(shell chdir))
else
	allprereqs = $+
	copycmd = cp -f $(1) $(2)
	dircopycmd = cp -rf $(1) $(2)
	rmcmd = rm -f $(1)
	rmdircmd = rm -rf $(1)
	mkdircmd = mkdir -p $(1)
	runtest = sh -c "cd $(test_dir); ./$(1) -d; exit"
	homedir := $(shell pwd)
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

# -- Tools --

libr =
linker =
compiler =

gprintf = printf

# Compiler definitions and flags

ccflags =
ccdefs =

ifneq (,$(findstring flm_dbg_log,$(MAKECMDGOALS)))
	ccdefs += FLM_DBG_LOG
endif

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
	
	ifeq ($(target_build_type),debug)
		ccflags += /Ob1 /Od /RTC1 /Wp64
		ccdefs += FLM_DEBUG
	else
		ccflags += /O2
	endif
	
	# Linker switches

	shared_link_flags = \
		/DLL \
		/DEBUG /PDB:$(subst /,\,$(@:.dll=.pdb)) \
		/map:$(subst /,\,$(@:.dll=.map)) \
		/INCREMENTAL:NO \
		/NOLOGO \
		/OUT:$(subst /,\,$@)
	
	util_link_flags = \
		/DEBUG /PDB:$(subst /,\,$(@:.exe=.pdb)) \
		/map:$(subst /,\,$(@:.exe=.map)) \
		/INCREMENTAL:NO \
		/FIXED:NO \
		/NOLOGO \
		/OUT:$(subst /,\,$@)

	# Libraries that our various components need to link against
	
	lib_link_libs = imagehlp.lib user32.lib rpcrt4.lib wsock32.lib
	exe_link_libs =

	# Convert the list of defines into a proper set of command-line params

	ifdef ccdefs
		ccdefine = $(foreach def,$(strip $(ccdefs)),/D$(def))
	endif

	# Same thing for the include dirs

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),/I$(subst @,$(sp),$(subst /,\,$(inc_dir))))

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
	
	# Compiler defines and flags

	ifeq ($(compiler),g++)
		ccflags += -Wall -Werror -fPIC
		ifneq ($(target_processor),ia64)		
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
			endif
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

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),-I$(subst @,$(sp),$(call hostpath,$(inc_dir))))

	# Concatenate everything into the ccflags variable

	ccflags += $(ccdefine) $(ccinclude)

	# Linker switches
	
	shared_link_flags =
	link_flags = -o $@
	
	ifeq ($(compiler),g++)
		ifneq ($(target_processor),ia64)		
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
			endif
		endif

		lib_link_libs += -lm -lc -ldl -lsocket -lnsl -lrt
	endif

	ifeq ($(target_os_family),osx)
		lib_link_libs += -lstdc++ -ldl
	endif
	
	util_link_flags = $(link_flags)
endif

# -- File lists --

xflaim_src = \
	$(wildcard src/*.cpp)

xflaim_jni_src = \
	$(wildcard jni/native/src/*.cpp)
	
utilsup_src = \
	flm_dlst.cpp \
	flm_lutl.cpp \
	ftx.cpp \
	sharutil.cpp

checkdb_src = \
	checkdb.cpp

rebuild_src = \
	rebuild.cpp

view_src = \
	view.cpp \
	viewblk.cpp \
	viewdisp.cpp \
	viewedit.cpp \
	viewhdr.cpp \
	viewlfil.cpp \
	viewmenu.cpp \
	viewsrch.cpp
	
sample_src = \
	sample.cpp

xshell_src = \
	fdomedt.cpp \
	xshell.cpp \
	fshell.cpp
	
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

xflaim_obj = $(patsubst src/%.cpp,$(obj_dir)/%$(obj_suffix),$(xflaim_src))
xflaim_jni_obj = $(patsubst jni/native/src/%.cpp,$(obj_dir)/%$(obj_suffix),$(xflaim_jni_src))
xflaim_static_lib = $(static_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
xflaim_shared_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)$(suffix_version)
xflaim_shared_imp_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
xflaim_jni_shared_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)jni$(shared_lib_suffix)$(suffix_version)

# -- Unit tests  --

ut_basictest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_basictest_src))
ut_binarytest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_binarytest_src))
ut_colldeftest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_colldeftest_src))
ut_dictchangetest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_dictchangetest_src))
ut_dictdeftest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_dictdeftest_src))
ut_dirtyexittest1_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_dirtyexittest1_src))
ut_dirtyexittest2_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_dirtyexittest2_src))
ut_domnodetest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_domnodetest_src))
ut_enctest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_enctest_src))
ut_importtest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_importtest_src))
ut_indexdeftest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_indexdeftest_src))
ut_indextest1_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_indextest1_src))
ut_indextest2_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_indextest2_src))
ut_indextest3_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_indextest3_src))
ut_metaphonetest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_metaphonetest_src))
ut_namespacetest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_namespacetest_src))
ut_regressiontest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_regressiontest_src))
ut_rfltest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_rfltest_src))
ut_sortkeytest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_sortkeytest_src))
ut_sortkeytest2_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_sortkeytest2_src))
ut_xpathtest1_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_xpathtest1_src))
ut_xpathtest2_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_xpathtest2_src))

# -- Java --

jni_java_src = $(wildcard jni/java/xflaim/*.java)

# -- Utilities  --

checkdb_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(checkdb_src))
checkdb_exe = $(util_dir)/checkdb$(exe_suffix)

rebuild_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(rebuild_src))
rebuild_exe = $(util_dir)/rebuild$(exe_suffix)

view_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(view_src))
view_exe = $(util_dir)/view$(exe_suffix)

xshell_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(xshell_src))
xshell_exe = $(util_dir)/xshell$(exe_suffix)

sample_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(sample_src))
sample_exe = $(sample_dir)/sample$(exe_suffix)

utilsup_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(utilsup_src))

# -- Make system pattern search paths -- 

vpath %.cpp src util sample jni/native/src 
vpath %.java jni/java/xflaim

# -- Default target --

.PHONY : libs
libs: status dircheck $(xflaim_static_lib) $(xflaim_shared_lib)

# -- JNI (Java) --

ifdef jhome
$(xflaim_jni_jar) : $(jni_java_src)
	$(ec)$(gprintf) "Compiling JNI java sources ...\n"
	$(ec)$(jhome)/bin/javac -g -d $(java_class_output_dir) $(call hostpath,$(jni_java_src))
	$(ec)$(gprintf) "Creating JNI jar file ...\n"
	$(ec)$(call rmcmd,$(xflaim_jni_jar))
	$(ec)$(jhome)/bin/jar cf $(call hostpath,$(xflaim_jni_jar)) \
		-C $(call hostpath,$(java_class_output_dir)) .
	$(ec)$(gprintf) "Creating JNI header files ...\n"
	$(ec)$(jhome)/bin/javah -d $(jni_inc_dir) \
		-classpath $(call hostpath,$(xflaim_jni_jar)) \
		$(foreach jclass,$(notdir $(basename $(jni_java_src))),xflaim.$(jclass))
endif

# -- *.cpp -> *$(obj_suffix) --

$(obj_dir)/%$(obj_suffix) : %.cpp
ifdef win_target
	$(ec)$(compiler) $(ccflags) /Fd$(subst /,\,$(obj_dir))\tmp.pdb \
		/Fo$(subst /,\,$@) $(subst /,\,$<)
else
	$(ec)$(gprintf) "$<\n"
	$(ec)$(compiler) $(ccflags) -c $< -o $@
endif

# -- xflaim.lib and libxflaim.a --

$(xflaim_static_lib) : $(xflaim_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(libr) /NOLOGO $(subst /,\,$+) /OUT:$(subst /,\,$@)	
else
	$(ec)rm -f $@
ifeq ($(target_os_family),osx)
	$(ec)$(libr) -static -o $@ $+
else
	$(ec)$(libr) -rcs $@ $+
endif
endif

# -- xflaim.dll and libxflaim.so --

$(xflaim_shared_lib) : $(xflaim_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(linker) $(subst /,\,$+) $(shared_link_flags) $(lib_link_libs)
else
	$(ec)rm -f $@
ifeq ($(target_os_family),linux)
	$(ec)$(linker) $+ $(shared_link_flags) $(lib_link_libs)
endif
endif

# -- xflaim.lib import library --

$(xflaim_shared_imp_lib) : $(xflaim_shared_lib)
	$(ec)$(gprintf) "Building $@ ...\n"

# -- xflaimjni.dll and libxflaimjni.so --

$(xflaim_jni_shared_lib) : $(xflaim_obj) $(xflaim_jni_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(linker) $(subst /,\,$+) $(shared_link_flags) $(lib_link_libs)
else
	$(ec)rm -f $@
ifeq ($(target_os_family),linux)
	$(ec)$(linker) $+ $(shared_link_flags) $(lib_link_libs)
endif
endif

# -- Utility link command --

ifndef flm_util_link_cmd
   define flm_util_link_cmd
		$(ec)$(linker) $(util_link_flags) $(allprereqs) $(lib_link_libs) $(exe_link_libs)
   endef
endif

# -- checkdb --

.PHONY : checkdb
checkdb: status dircheck libs $(checkdb_exe)
$(checkdb_exe): $(checkdb_obj) $(utilsup_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- rebuild --

.PHONY : rebuild
rebuild: status dircheck libs $(rebuild_exe)
$(rebuild_exe): $(rebuild_obj) $(utilsup_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- view --

.PHONY : view
view: status dircheck libs $(view_exe)
$(view_exe): $(view_obj) $(utilsup_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- sample --

.PHONY : sample
sample: status dircheck libs $(sample_exe)
$(sample_exe): $(sample_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	$(ec)$(call copycmd,sample/xmlfiles/*.xml,$(sample_dir))

# -- xshell --

.PHONY : xshell
xshell: status dircheck libs $(xshell_exe)
$(xshell_exe): $(xshell_obj) $(utilsup_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- basictest --

.PHONY : basictest
basictest: status dircheck $(test_dir)/basictest$(exe_suffix)
$(test_dir)/basictest$(exe_suffix): $(ut_basictest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- binarytest --

.PHONY : binarytest
binarytest: status dircheck $(test_dir)/binarytest$(exe_suffix)
$(test_dir)/binarytest$(exe_suffix): $(ut_binarytest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- colldeftest --

.PHONY : colldeftest
colldeftest: status dircheck $(test_dir)/colldeftest$(exe_suffix)
$(test_dir)/colldeftest$(exe_suffix): $(ut_colldeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- dictchangetest --

.PHONY : dictchangetest
dictchangetest: status dircheck $(test_dir)/dictchangetest$(exe_suffix)
$(test_dir)/dictchangetest$(exe_suffix): $(ut_dictchangetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- dictdeftest --

.PHONY : dictdeftest
dictdeftest: status dircheck $(test_dir)/dictdeftest$(exe_suffix)
$(test_dir)/dictdeftest$(exe_suffix): $(ut_dictdeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- dirtyexittest1 --

.PHONY : dirtyexittest1
dirtyexittest1: status dircheck $(test_dir)/dirtyexittest1$(exe_suffix)
$(test_dir)/dirtyexittest1$(exe_suffix): $(ut_dirtyexittest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- dirtyexittest2 --

.PHONY : dirtyexittest2
dirtyexittest2: status dircheck $(test_dir)/dirtyexittest2$(exe_suffix)
$(test_dir)/dirtyexittest2$(exe_suffix): $(ut_dirtyexittest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- domnodetest --

.PHONY : domnodetest
domnodetest: status dircheck $(test_dir)/domnodetest$(exe_suffix)
$(test_dir)/domnodetest$(exe_suffix): $(ut_domnodetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- enctest --

.PHONY : enctest
enctest: status dircheck $(test_dir)/enctest$(exe_suffix)
$(test_dir)/enctest$(exe_suffix): $(ut_enctest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- importtest --

.PHONY : importtest
importtest: status dircheck $(test_dir)/importtest$(exe_suffix)
$(test_dir)/importtest$(exe_suffix): $(ut_importtest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- indexdeftest --

.PHONY : indexdeftest
indexdeftest: status dircheck $(test_dir)/indexdeftest$(exe_suffix)
$(test_dir)/indexdeftest$(exe_suffix): $(ut_indexdeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- indextest1 --

.PHONY : indextest1
indextest1: status dircheck $(test_dir)/indextest1$(exe_suffix)
$(test_dir)/indextest1$(exe_suffix): $(ut_indextest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- indextest2 --

.PHONY : indextest2
indextest2: status dircheck $(test_dir)/indextest2$(exe_suffix)
$(test_dir)/indextest2$(exe_suffix): $(ut_indextest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- indextest3 --

.PHONY : indextest3
indextest3: status dircheck $(test_dir)/indextest3$(exe_suffix)
$(test_dir)/indextest3$(exe_suffix): $(ut_indextest3_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- metaphonetest --

.PHONY : metaphonetest
metaphonetest: status dircheck $(test_dir)/metaphonetest$(exe_suffix)
$(test_dir)/metaphonetest$(exe_suffix): $(ut_metaphonetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- namespacetest --

.PHONY : namespacetest
namespacetest: status dircheck $(test_dir)/namespacetest$(exe_suffix)
$(test_dir)/namespacetest$(exe_suffix): $(ut_namespacetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- regressiontest --

.PHONY : regressiontest
regressiontest: status dircheck $(test_dir)/regressiontest$(exe_suffix)
$(test_dir)/regressiontest$(exe_suffix): $(ut_regressiontest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- rfltest --

.PHONY : rfltest
rfltest: status dircheck $(test_dir)/rfltest$(exe_suffix)
$(test_dir)/rfltest$(exe_suffix): $(ut_rfltest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- sortkeytest --

.PHONY : sortkeytest
sortkeytest: status dircheck $(test_dir)/sortkeytest$(exe_suffix)
$(test_dir)/sortkeytest$(exe_suffix): $(ut_sortkeytest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- sortkeytest2 --

.PHONY : sortkeytest2
sortkeytest2: status dircheck $(test_dir)/sortkeytest2$(exe_suffix)
$(test_dir)/sortkeytest2$(exe_suffix): $(ut_sortkeytest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- xpathtest1 --

.PHONY : xpathtest1
xpathtest1: status dircheck $(test_dir)/xpathtest1$(exe_suffix)
$(test_dir)/xpathtest1$(exe_suffix): $(ut_xpathtest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- xpathtest2 --

.PHONY : xpathtest2
xpathtest2: status dircheck $(test_dir)/xpathtest2$(exe_suffix)
$(test_dir)/xpathtest2$(exe_suffix): $(ut_xpathtest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)
	
# -- jni --

.PHONY : jni
jni: status dircheck $(xflaim_jni_jar) $(xflaim_jni_shared_lib)
	
# -- dist --

.PHONY : dist
dist: status dircheck
	$(ec)$(gprintf) "Creating package (SVN Revision $(revision)) ...\n"
	-$(ec)$(call rmdircmd,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call mkdircmd,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(gprintf) "$(version)" > $(package_dir)/$(package_proj_name_and_ver)/VERSION
	$(ec)$(gprintf) " " > $(package_dir)/$(package_proj_name_and_ver)/SVNRevision.$(revision)
	$(ec)$(call copycmd,Makefile,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call copycmd,COPYING,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call copycmd,Doxyfile,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call dircopycmd,src,$(package_dir)/$(package_proj_name_and_ver)/src)
	$(ec)$(call dircopycmd,util,$(package_dir)/$(package_proj_name_and_ver)/util)
	$(ec)$(call dircopycmd,sample,$(package_dir)/$(package_proj_name_and_ver)/sample)
ifdef win_target
	$(ec)$(call copycmd,make.exe,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call copycmd,printf.exe,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)cmd /C "cd $(subst /,\,$(homedir)/$(package_dir)) && \
			$(subst /,\,$(homedir)/7za a -tzip -r ZIPS/$(src_zip_name)) \
			$(package_proj_name_and_ver)"
else
	$(ec)tar zcf $(package_sources_dir)/$(package_proj_name_and_ver).tar.gz \
			-C $(package_dir) $(package_proj_name_and_ver)
	$(ec)chmod 775 $(package_sources_dir)/$(package_proj_name_and_ver).tar.gz
endif
	$(ec)$(call copycmd,$(package_proj_name).changes,$(package_dir))
	-$(ec)$(call rmdircmd,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(gprintf) "Package created.\n"
	
# -- changelog --

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
ifndef win_target
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

# -- uninstall --

.PHONY : uninstall
uninstall: 
ifndef win_target
	$(ec)$(gprintf) "Uninstalling ...\n"
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)*
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
	-rm -rf $(pkgconfig_install_dir)/$(pkgconfig_file_name)
	-rm -rf $(include_install_dir)/xflaim.h
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
	$(ec)$(gprintf) "Libs: $(lib_link_libs) -lxflaim -L$(dollar){libdir}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Cflags: -I$(dollar){includedir}\n" >> $(pkgconfig_file)

# -- SRCRPM --

.PHONY : srcrpm
srcrpm: dist spec
	$(ec)$(gprintf) "Creating source RPM ...\n"
	$(ec)rpmbuild --define="_topdir $(homedir)/$(package_dir)" --quiet -bs $(spec_file)
	$(ec)$(gprintf) "Source RPM created.\n"
	
# -- RPMS --

.PHONY : rpms
rpms: dist spec
	$(ec)$(gprintf) "Creating source and binary RPMs ...\n"
	$(ec)rpmbuild --define="_topdir $(homedir)/$(package_dir)" --quiet -ba $(spec_file)
	$(ec)find $(package_dir) -name *.rpm | xargs chmod 775
	$(ec)$(gprintf) "Source and binary RPMs created.\n"
	
# -- ZIPS --

.PHONY : zips
zips: dist
ifdef win_target
	$(ec)$(gprintf) "Creating source and binary ZIPs ...\n"
	$(ec)$(call copycmd,$(package_zips_dir)/$(src_zip_name),$(package_build_dir))
	$(ec)7za x -o$(package_build_dir) $(package_build_dir)/$(src_zip_name)
	$(ec)$(call rmcmd,$(package_build_dir)/$(src_zip_name))
	$(ec)make -C $(package_build_dir)/$(package_proj_name_and_ver) all
	$(ec)$(call mkdircmd,$(zip_stage_dir))
	$(ec)$(call mkdircmd,$(zip_inc_stage_dir))
	$(ec)$(call mkdircmd,$(zip_shared_lib_stage_dir))
	$(ec)$(call mkdircmd,$(zip_static_lib_stage_dir))
	$(ec)$(call mkdircmd,$(zip_util_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/VERSION,\
		$(zip_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/COPYING,\
		$(zip_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/src/xflaim.h,\
		$(zip_inc_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(xflaim_static_lib), \
		$(zip_static_lib_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(xflaim_shared_lib), \
		$(zip_shared_lib_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(xflaim_shared_imp_lib), \
		$(zip_shared_lib_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(checkdb_exe),\
		$(zip_util_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(rebuild_exe),\
		$(zip_util_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(view_exe),\
		$(zip_util_stage_dir))
	$(ec)$(call copycmd,$(package_build_dir)/$(package_proj_name_and_ver)/$(xshell_exe),\
		$(zip_util_stage_dir))
	$(ec)cmd /C "cd $(subst /,\,$(zip_stage_parent_dir)) && \
			$(subst /,\,$(homedir)/7za a -tzip -r $(homedir)/$(package_build_dir)/$(bin_zip_name)) \
			$(package_proj_name_and_ver)"
	$(ec)$(call copycmd,$(package_build_dir)/$(bin_zip_name),$(package_zips_dir))
	$(ec)$(call rmcmd,$(package_build_dir)/$(bin_zip_name))
	-$(ec)$(call rmdircmd,$(package_build_dir)/$(package_proj_name_and_ver))
	$(ec)$(gprintf) "Source and binary ZIPs created.\n"
endif
	
# -- misc. targets --

.PHONY : status
status:
	$(ec)$(gprintf) "===============================================================================\n"
	$(ec)$(gprintf) "SVN Revision = $(revision)\n"
	$(ec)$(gprintf) "Host Operating System Family = $(host_os_family)\n"
	$(ec)$(gprintf) "Directory = $(homedir)\n"
	$(ec)$(gprintf) "Target Operating System Family = $(target_os_family)\n"
	$(ec)$(gprintf) "Target Processor = $(target_processor)\n"
	$(ec)$(gprintf) "Target Word Size = $(target_word_size)\n"
	$(ec)$(gprintf) "Target Build Type = $(target_build_type)\n"
	$(ec)$(gprintf) "Target Path = $(target_path)\n"
	$(ec)$(gprintf) "Compiler = $(compiler)\n"
	$(ec)$(gprintf) "Librarian = $(libr)\n"
	$(ec)$(gprintf) "Java Home = $(jhome)\n"
	$(ec)$(gprintf) "Defines = $(strip $(ccdefs))\n"
	$(ec)$(gprintf) "===============================================================================\n"

.PHONY : dircheck
dircheck:
	$(ec)$(call mkdircmd,$(obj_dir))
	$(ec)$(call mkdircmd,$(static_obj_dir))
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
	$(ec)$(call mkdircmd,$(package_zips_dir))

# -- phony targets --

.PHONY : all
all: libs allutils
	$(ec)$(gprintf) ""

.PHONY : allutils
allutils: status dircheck libs checkdb rebuild view sample xshell
	$(ec)$(gprintf) ""

.PHONY : test
test:	status dircheck $(xflaim_static_lib) \
		basictest binarytest colldeftest dictchangetest dictdeftest \
		dirtyexittest1 dirtyexittest2 domnodetest enctest importtest \
		indexdeftest indextest1 indextest2 indextest3 \
		namespacetest metaphonetest regressiontest rfltest sortkeytest \
		sortkeytest2 xpathtest1 xpathtest2
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
	-$(ec)$(call rmdircmd,$(build_output_dir))
	-$(ec)$(call rmcmd,*.pch)
	
.PHONY : distclean
	-$(ec)$(call rmcmd,*.pch)
	
.PHONY : mostlyclean
mostlyclean : clean
	$(ec)$(gprintf) ""

.PHONY : maintainer-clean
maintainer-clean:
	-$(ec)$(call rmdircmd,$(build_output_dir))
	-$(ec)$(call rmcmd,*.pch)
