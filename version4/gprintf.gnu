#*****************************************************************************
#File:   gprintf.gnu
#Desc:   GNU makefile for gprintf utility
#
#
# $Log$
# Revision 1.2  2005/11/10 22:58:48  dsanders
# Check in open-source changes at head branch - taken from rosalind base.
#
# Revision 1.0  2000/06/28 13:46:40  andy
# Initial revision
#
#  
#     Rev 1.0   28 Jun 2000 13:46:40   andy
#  Initial revision.
#  
#---------------------------------------------------------------------------
#
#	Copyright (C) Unpublished Work of Novell, Inc. 
#	All Rights Reserved.
#
#	This work is an unpublished work and contains confidential, 
#	proprietary and trade secret information of Novell, Inc. Access 
#	to this work is restricted to (i) Novell, Inc. employees who have 
#	a need to know how to perform tasks within the scope of their 
#	assignments and (ii) entities other than Novell, Inc. who have 
#	entered into appropriate license agreements. No part of this work 
#	may be used, practiced, performed, copied, distributed, revised, 
#	modified, translated, abridged, condensed, expanded, collected, 
#	compiled, linked, recast, transformed or adapted without the
#	prior written consent of Novell, Inc. Any use or exploitation of 
#	this work without authorization could subject the perpetrator to 
#	criminal and civil liability.
#----------------------------------------------------------------------------

# -- includes --

# -- misc. declarations --
.PHONY : all clean
.SUFFIXES : .cpp .c .h .hpp .obj .rsp

# -- variables --

build_os =
env_ok = 1
error_str =
win32_target =

# -- includes --

# -- OS --

ifeq ($(OS),WINNT)
	build_os = WINNT
endif

ifeq ($(OS),Windows_NT)
	build_os = WINNT
endif

ifndef build_os
	error_str = Unsupported operating system
	env_ok =
endif

# -- build (debug, release, etc.) --

ifndef build
	build = release
endif

# -- platform --

ifndef platform
	platform = vc6
endif

ifeq ($(platform),vc6)
	win32_target = 1
endif

# -- default directories --

ifndef vc_dir
	vc_dir = c:/msdev6/vc98
endif

ifndef flmroot
	flmroot = c:/flaim/$(flm_dir)
endif

# -- Files --

gprintf_src = gprintf.cpp
gprintf_obj = $(patsubst %.cpp,$(flmroot)/$(build)/$(platform)/%.obj,$(gprintf_src))

# -- linker definitions -- 

kernel_libs=
link_flags=

ifdef win32_target
	kernel_libs = user32.lib mpr.lib libcmt.lib libcpmt.lib \
		oldnames.lib kernel32.lib imagehlp.lib wsock32.lib advapi32.lib

	link_flags =  /fixed:no /nologo /machine:i386

	ifeq ($(build),debug)
		kernel_libs += msvcrtd.lib 
		link_flags += /debug
	else
		kernel_libs += msvcrt.lib
	endif
endif

# -- utility variables -- 
em :=
sp := $(em) $(em)
comma := ,

# -- setup include path --
inc = $(vc_dir)/include;$(flmroot)/util

# -- compiler definitions -- 
ccdefs =
ccincs =
ccflags =
libflags =

ifdef win32_target

	ccdefs += WIN32 WIN32_LEAN_AND_MEAN WIN32_EXTRA_LEAN

	ifeq ($(build),debug)
		ccdefs += DEBUG
	endif

	ccincs += /I"$(subst ;,"$(sp)/I",$(inc))"
	ccincs += /I$(vc_dir)/include

	ccflags += /nologo /c /G3s /Zp1 /Gf /J \
		/MT /W3 /YX /Oy-

	ifeq ($(build),release)
		ccflags += /Ox /Gy
	else
		ccflags += /Z7 /Od /Ob1
	endif

	libflags += /nologo

endif

# -- tool names -- 
libr =
linker =
cc =
ccp =

ifdef win32_target
	libr = $(subst \,/,$(strip $(vc_dir)))/bin/lib.exe
	linker = $(subst \,/,$(strip $(vc_dir)))/bin/link.exe
	cc = $(subst \,/,$(strip $(vc_dir)))/bin/cl.exe
	ccp = $(subst \,/,$(strip $(vc_dir)))/bin/cl.exe
endif

ifndef libr
	error_str = Librarian not defined
	env_ok =
endif

ifndef cc
	error_str = C compiler not defined
	env_ok =
endif

ifndef ccp
	error_str = C++ compiler not defined
	env_ok =
endif

# -- make system pattern search paths -- 

vpath %.c $(inc)
vpath %.cpp $(inc)

# -- pattern rules --

ifdef env_ok

$(flmroot)/$(build)/$(platform)/%.obj : %.cpp
	@$(subst /,\\,$(strip $(ccp))) $(ccflags) /Fo$@ /D$(subst $(sp), /D,$(strip $(ccdefs))) $(ccincs) $<

$(flmroot)/$(build)/$(platform)/gprintf.exe: $(gprintf_obj)
	@echo $(gprintf_obj) > gprintf.lis
	@echo $(kernel_libs) >> gprintf.lis
	@$(linker) $(link_flags) /out:$@ @gprintf.lis
	@del gprintf.lis

else

@echo Environment not configured properly.
@echo $(error_str)

endif


