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

# -- Project --

project_name = flaim
project_desc = An extensible, flexible, adaptable, embeddable database engine

# -- Versions --

major_version = 4
minor_version = 8
ifneq (,$(findstring dist,$(MAKECMDGOALS)))

	# Get the info for all files.
	
	revision = $(shell svnversion . -n)
	
	ifneq (,$(findstring M,$(revision)))
      $(error Local modifications found - please check in before making distro)
	endif

	ifneq (,$(findstring :,$(revision)))
      $(error Mixed revisions in repository - please update before making distro)
	endif
else
	ifeq "$(wildcard SVNRevision.*)" ""
		revision = 0
	else
		revision = $(word 2,$(subst ., ,$(wildcard SVNRevision.*)))
	endif
endif
version = $(major_version).$(minor_version).$(revision)

rpm_release_num = 1

# -- Paths initializations --

rpm_build_root =
install_prefix = /usr

# -- RPM, SPEC file names

rpm_proj_name = lib$(project_name)
rpm_proj_name_and_ver = $(rpm_proj_name)-$(version)

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
   $(error Host operating system could not be determined)
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
				ifneq (,$(findstring i386,$(HOSTTYPE)))
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
spec_file = $(package_dir)/$(rpm_proj_name).spec
pkgconfig_file_name = $(rpm_proj_name).pc
pkgconfig_file = $(package_dir)/$(pkgconfig_file_name)

target_path = $(build_output_dir)/$(target_os_family)-$(target_processor)-$(target_word_size)/$(target_build_type)

inc_dirs = src util
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

ifeq ($(host_os_family),win)
	allprereqs  = $(subst /,\,$+)
	copycmd = copy /Y $(subst /,\,$(1)) $(subst /,\,$(2)) >NUL
	rmcmd = del /Q $(subst /,\,$(1))
	rmdircmd = cmd /C "rmdir /q /s $(subst /,\,$(1))"
	mkdircmd = -if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1)) >NUL
	runtest = cmd /C "cd $(subst /,\,$(test_dir)) && $(1) -d"
else
	allprereqs = $+
	copycmd = cp -f $(1) $(2)
	rmcmd = rm -f $(1)
	rmdircmd = rm -rf $(1)
	mkdircmd = mkdir -p $(1)
	runtest = sh -c "cd $(test_dir); ./$(1) -d; exit"
endif

# -- Tools --

libr =
linker =
compiler =

ifdef unix_target
	gprintf = printf
else
	gprintf = gprintf
endif

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
	lib_suffix = .lib
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

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),/I$(subst /,\,$(inc_dir)))

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
	lib_suffix = .a
	shared_lib_suffix = .so
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
				-funit-at-a-time \
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

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),-I$(inc_dir))

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

flaim_src = \
	$(wildcard src/*.cpp)

utilsup_src = \
	flm_dlst.cpp \
	flm_lutl.cpp \
	ftx.cpp \
	ftxunix.cpp \
	sharutil.cpp \
	wpscrnkb.cpp

checkdb_src = \
	checkdb.cpp

rebuild_src = \
	rebuild.cpp

view_src = \
	view.cpp \
	viewblk.cpp \
	viewdisp.cpp \
	viewedit.cpp \
	viewfhdr.cpp \
	viewlhdr.cpp \
	viewlfil.cpp \
	viewmenu.cpp \
	viewsrch.cpp
	
sample_src = \
	sample.cpp

ut_basictest_src = \
	flmunittest.cpp \
	basic_test.cpp
	
# -- FLAIM library --

flaim_obj = $(patsubst src/%.cpp,$(obj_dir)/%$(obj_suffix),$(flaim_src))
static_flaim_lib = $(static_lib_dir)/$(lib_prefix)$(project_name)$(lib_suffix)
shared_flaim_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)-$(version)$(shared_lib_suffix)

# -- Unit tests  --

ut_basictest_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(ut_basictest_src))

# -- Utilities  --

checkdb_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(checkdb_src))
rebuild_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(rebuild_src))
view_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(view_src))
sample_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(sample_src))
utilsup_obj = $(patsubst %.cpp,$(obj_dir)/%$(obj_suffix),$(utilsup_src))

# -- Make system pattern search paths -- 

vpath %.cpp src util sample

# -- Default target --

.PHONY : libs
libs: status dircheck $(static_flaim_lib) $(shared_flaim_lib)

# -- *.cpp -> *$(obj_suffix) --

$(obj_dir)/%$(obj_suffix) : %.cpp
ifdef win_target
	$(ec)$(compiler) $(ccflags) /Fd$(subst /,\,$(obj_dir))\tmp.pdb \
		/Fo$(subst /,\,$@) $(subst /,\,$<)
else
	$(ec)$(gprintf) "$<\n"
	$(ec)$(compiler) $(ccflags) -c $< -o $@
endif

# -- flaim.lib and libflaim.a --

$(static_flaim_lib) : $(flaim_obj)
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

# -- flaim.dll and libflaim.so --

$(shared_flaim_lib) : $(flaim_obj)
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
checkdb: status dircheck $(util_dir)/checkdb$(exe_suffix)
$(util_dir)/checkdb$(exe_suffix): $(checkdb_obj) $(utilsup_obj) $(static_flaim_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- rebuild --

.PHONY : rebuild
rebuild: status dircheck $(util_dir)/rebuild$(exe_suffix)
$(util_dir)/rebuild$(exe_suffix): $(rebuild_obj) $(utilsup_obj) $(static_flaim_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- view --

.PHONY : view
view: status dircheck $(util_dir)/view$(exe_suffix)
$(util_dir)/view$(exe_suffix): $(view_obj) $(utilsup_obj) $(static_flaim_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- sample --

.PHONY : sample
sample: status dircheck $(sample_dir)/sample$(exe_suffix)
$(sample_dir)/sample$(exe_suffix): $(sample_obj) $(static_flaim_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- basictest --

.PHONY : basictest
basictest: status dircheck $(test_dir)/basictest$(exe_suffix)
$(test_dir)/basictest$(exe_suffix): $(ut_basictest_obj) $(utilsup_obj) $(static_flaim_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(flm_util_link_cmd)

# -- dist --

.PHONY : dist
dist: status dircheck spec
ifndef win_target
	$(ec)$(gprintf) "Creating package (SVN Revision $(revision)) ...\n"
	-$(ec)$(call rmdircmd,$(package_dir))/*.gz
	$(ec)$(call mkdircmd,$(package_dir)/$(rpm_proj_name_and_ver))
	$(ec)$(gprintf) "$(version)" > $(package_dir)/$(rpm_proj_name_and_ver)/VERSION
	$(ec)$(gprintf) " " > $(package_dir)/$(rpm_proj_name_and_ver)/SVNRevision.$(revision)
	$(ec)$(call copycmd,Makefile,$(package_dir)/$(rpm_proj_name_and_ver))
	$(ec)$(call copycmd,COPYING,$(package_dir)/$(rpm_proj_name_and_ver))
	$(ec)$(call copycmd,Doxyfile,$(package_dir)/$(rpm_proj_name_and_ver))
	$(ec)$(call mkdircmd,$(package_dir)/$(rpm_proj_name_and_ver)/src)
	$(ec)$(call copycmd,src/*.cpp,$(package_dir)/$(rpm_proj_name_and_ver)/src)
	$(ec)$(call copycmd,src/*.h,$(package_dir)/$(rpm_proj_name_and_ver)/src)
	$(ec)$(call mkdircmd,$(package_dir)/$(rpm_proj_name_and_ver)/util)
	$(ec)$(call copycmd,util/*.cpp,$(package_dir)/$(rpm_proj_name_and_ver)/util)
	$(ec)$(call copycmd,util/*.h,$(package_dir)/$(rpm_proj_name_and_ver)/util)
	$(ec)tar zcf $(package_dir)/$(rpm_proj_name_and_ver).tar.gz -C $(package_dir) $(rpm_proj_name_and_ver)
	$(ec)$(call copycmd,$(rpm_proj_name).changes,$(package_dir))
	-$(ec)$(call rmdircmd,$(package_dir)/$(rpm_proj_name_and_ver))
	$(ec)$(gprintf) "Package created.\n"
endif
	
# -- install --

.PHONY : install
install: libs pkgconfig
ifndef win_target
	$(ec)$(gprintf) "Installing ...\n"
	mkdir -p $(lib_install_dir)/pkgconfig
	mkdir -p $(include_install_dir)
	install --mode=644 $(shared_flaim_lib) $(lib_install_dir)
	ln -fs $(lib_prefix)$(project_name)-$(version)$(shared_lib_suffix) $(lib_install_dir)/$(lib_prefix)$(project_name)-$(major_version).$(minor_version)$(shared_lib_suffix)
	ln -fs $(lib_prefix)$(project_name)-$(version)$(shared_lib_suffix) $(lib_install_dir)/$(lib_prefix)$(project_name)-$(major_version)$(shared_lib_suffix)
	ln -fs $(lib_prefix)$(project_name)-$(version)$(shared_lib_suffix) $(lib_install_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)
	install --mode=644 $(static_flaim_lib) $(lib_install_dir)
	install --mode=644 $(pkgconfig_file) $(pkgconfig_install_dir)
	install --mode=644 src/flaim.h $(include_install_dir)
	$(ec)$(gprintf) "Installation complete.\n"
endif

# -- uninstall --

.PHONY : uninstall
uninstall: 
ifndef win_target
	$(ec)$(gprintf) "Uninstalling ...\n"
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)-$(version)$(shared_lib_suffix)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)-$(major_version)$(shared_lib_suffix)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)-$(major_version).$(minor_version)$(shared_lib_suffix)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(lib_suffix)
	-rm -rf $(pkgconfig_install_dir)/$(pkgconfig_file_name)
	-rm -rf $(include_install_dir)/flaim.h
	$(ec)$(gprintf) "Uninstalled.\n"
endif

# -- spec file --


.PHONY : spec
spec: dircheck
	$(ec)$(gprintf) "# neededforbuild  libstdc++-devel libstdc++ gcc-c++\n" > $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)define prefix $(install_prefix)\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "BuildRequires: libstdc++-devel libstdc++ gcc-c++\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: $(project_desc)\n" >> $(spec_file)
	$(ec)$(gprintf) "Name: $(rpm_proj_name)\n" >> $(spec_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(spec_file)
	$(ec)$(gprintf) "Release: $(rpm_release_num)\n" >> $(spec_file)
	$(ec)$(gprintf) "Copyright: GPL\n" >> $(spec_file)
	$(ec)$(gprintf) "Vendor: Novell, Inc.\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "Source: $(rpm_proj_name_and_ver).tar.gz\n" >> $(spec_file)
	$(ec)$(gprintf) "BuildRoot: /var/tmp/$(rpm_proj_name_and_ver)-buildroot\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)package devel\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: FLAIM static library and header file\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "Provides: $(rpm_proj_name)-devel\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description devel\n" >> $(spec_file)
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
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)-$(version)$(shared_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)-$(major_version).$(minor_version)$(shared_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)-$(major_version)$(shared_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(shared_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files devel\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/pkgconfig/$(pkgconfig_file_name)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/include/flaim.h\n" >> $(spec_file)

# -- PKG-CONFIG --

.PHONY : pkgconfig
pkgconfig: dircheck
	$(ec)$(gprintf) "prefix=$(install_prefix)\n" > $(pkgconfig_file)
	$(ec)$(gprintf) "exec_prefix=$(dollar){prefix}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "libdir=$(dollar){exec_prefix}/$(lib_dir_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "name: $(rpm_proj_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Description: $(project_desc)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Libs: $(lib_link_libs) -lflaim -L$(dollar){libdir}\n" >> $(pkgconfig_file)

# -- SRCRPM --

.PHONY : srcrpm
srcrpm: dist
	$(ec)$(gprintf) "Creating source RPM ...\n"
	$(ec)rpmbuild -ts --quiet $(package_dir)/$(project_name)-$(version).tar.gz
	$(ec)$(call copycmd,$(shell rpm --eval %{_srcrpmdir})/$(project_name)-$(version)*.src.rpm $(package_dir))
	$(ec)$(gprintf) "Source RPM created.\n"
	
# -- Documentation --

.PHONY : docs
docs: status dircheck
	$(ec)$(gprintf) "Creating documentation ...\n"
	$(ec)doxygen Doxyfile
	$(ec)$(gprintf) "Documentation created.\n"
	
# -- misc. targets --

.PHONY : status
status:
	$(ec)$(gprintf) "===============================================================================\n"
	$(ec)$(gprintf) "SVN Revision = $(revision)\n"
	$(ec)$(gprintf) "Host Operating System Family = $(host_os_family)\n"
	$(ec)$(gprintf) "Target Operating System Family = $(target_os_family)\n"
	$(ec)$(gprintf) "Target Processor = $(target_processor)\n"
	$(ec)$(gprintf) "Target Word Size = $(target_word_size)\n"
	$(ec)$(gprintf) "Target Build Type = $(target_build_type)\n"
	$(ec)$(gprintf) "Target Path = $(target_path)\n"
	$(ec)$(gprintf) "Compiler = $(compiler)\n"
	$(ec)$(gprintf) "Librarian = $(libr)\n"
	$(ec)$(gprintf) "Defines = $(strip $(ccdefs))\n"
	$(ec)$(gprintf) "===============================================================================\n"

.PHONY : dircheck
dircheck:
	$(ec)$(call mkdircmd,$(obj_dir))
	$(ec)$(call mkdircmd,$(util_dir))
	$(ec)$(call mkdircmd,$(test_dir))
	$(ec)$(call mkdircmd,$(sample_dir))
	$(ec)$(call mkdircmd,$(static_lib_dir))
	$(ec)$(call mkdircmd,$(shared_lib_dir))
	$(ec)$(call mkdircmd,$(package_dir))

# -- phony targets --

.PHONY : all
all: libs allutils
	$(ec)$(gprintf) ""

.PHONY : allutils
allutils: status dircheck checkdb rebuild view sample
	$(ec)$(gprintf) ""

.PHONY : test
test:	status dircheck $(static_flaim_lib) basictest
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
	
.PHONY : installcheck
installcheck:
	$(ec)$(gprintf) ""
	
.PHONY : clean
clean:
	-$(ec)$(call rmdircmd,$(target_path))
	-$(ec)$(call rmcmd *.pch)
	
.PHONY : distclean
	-$(ec)$(call rmcmd *.pch)
	
.PHONY : mostlyclean
mostlyclean : clean
	$(ec)$(gprintf) ""

.PHONY : maintainer-clean
maintainer-clean:
	-$(ec)$(call rmdircmd,$(build_output_dir))
	-$(ec)$(call rmcmd *.pch)
