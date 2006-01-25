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

# -- Target variables --

target_build_type =
usenativecc = yes
target_os_family =
target_processor =
target_word_size =
requested_target_word_size =
win_target =
unix_target =
build_output_dir = ./build

# -- Enable command echoing

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
					$(error Platform not supported)
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

# -- Utility variables --

em :=
sp := $(em) $(em)
comma := ,

ifeq ($(host_os_family),win)
	allprereqs  = $(subst /,\,$+)
	copycmd = copy /Y $(subst /,\,$(1)) $(subst /,\,$(2)) >NUL
	mkdircmd = -if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1)) >NUL
	runtest = cmd /C "cd $(subst /,\,$(test_dir)) && $(1) -d"
else
	allprereqs = $+
	copycmd = cp -f $(1) $(2)
	mkdircmd = mkdir -p $(1)
	runtest = sh -c "cd $(test_dir); ./$(1) -d; exit"
endif

# -- Target path --

target_path = $(build_output_dir)/$(target_os_family)-$(target_processor)-$(target_word_size)/$(target_build_type)

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

# Directories

inc_dirs = src util smi
obj_dir = $(target_path)/obj
util_dir = $(target_path)/util
test_dir = $(target_path)/test
sample_dir = $(target_path)/sample
shared_lib_dir = $(target_path)/lib/shared
static_lib_dir = $(target_path)/lib/static

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
	
	link_libs = imagehlp.lib user32.lib rpcrt4.lib wsock32.lib

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

		ifeq ($(target_word_size),64)
			ccflags += -m64
		else
			ccflags += -m32
		endif
	endif
	
	ifeq ($(target_os_family),linux)
	
		# Must support 64 bit file sizes - even for 32 bit builds.
	
		ccdefs += N_PLAT_UNIX _LARGEFILE64_SOURCE _FILE_OFFSET_BITS=64
		
		ifeq ($(target_build_type),release)
			ccflags += \
				-O \
				-fforce-mem \
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
		ifeq ($(target_word_size),64)
			link_flags += -m64
		else
			link_flags += -m32
		endif
	endif
	
	link_libs = -lpthread

	ifeq ($(target_os_family),linux)
		link_libs += -lncurses -lrt -lstdc++ -ldl
		shared_link_flags += -shared -Wl,-Bsymbolic -fpic \
			-Wl,-soname,$(@F).1 -o $@
	else
		link_libs += -lcurses
	endif

	ifeq ($(target_os_family),solaris)
		link_flags += -R /usr/lib/lwp
		ifeq ($(usenativecc),yes)
			ifeq ($(target_word_size),64)
				link_flags += -xarch=generic64
			endif
		endif
		link_libs += -lm -lc -ldl -lsocket -lnsl -lrt
	endif

	ifeq ($(target_os_family),osx)
		link_libs += -lstdc++ -ldl
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
static_flaim_lib = $(static_lib_dir)/$(lib_prefix)flaim$(lib_suffix)
shared_flaim_lib = $(shared_lib_dir)/$(lib_prefix)flaim$(shared_lib_suffix)

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

.PHONY : all
all: status dircheck $(static_flaim_lib) $(shared_flaim_lib) allutils

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
	$(ec)$(linker) $(subst /,\,$+) $(shared_link_flags) $(link_libs)
else
	$(ec)rm -f $@
ifeq ($(target_os_family),linux)
	$(ec)$(linker) $+ $(shared_link_flags) $(link_libs)
endif
endif

# -- Utility link command --

ifndef flm_util_link_cmd
   define flm_util_link_cmd
		$(ec)$(linker) $(util_link_flags) $(allprereqs) $(link_libs)
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

# -- misc. targets --

status:
	$(ec)$(gprintf) "===============================================================================\n"
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

dircheck:
	$(ec)$(call mkdircmd,$(obj_dir))
	$(ec)$(call mkdircmd,$(util_dir))
	$(ec)$(call mkdircmd,$(test_dir))
	$(ec)$(call mkdircmd,$(sample_dir))
	$(ec)$(call mkdircmd,$(static_lib_dir))
	$(ec)$(call mkdircmd,$(shared_lib_dir))

# -- phony targets --

.PHONY : allutils
allutils: status dircheck checkdb rebuild view sample

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

.PHONY : clean
clean:
	$(ec)$(gprintf) "Cleaning $(target_path) ...\n"
ifdef win_target
	-$(ec)if exist $(subst /,\,$(target_path)) rd /s /q $(subst /,\,$(target_path)) >NUL
	-$(ec)if exist *.pch del /Q *.pch >NUL
else
	-$(ec)rm -rf $(target_path) 2>/dev/null
endif
