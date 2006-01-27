//------------------------------------------------------------------------------
// Desc:	Contains prototypes for memory functions.
//
// Tabs:	3
//
//		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: ftkmem.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FTKMEM_H
#define FTKMEM_H

extern "C"
{
#if defined( FLM_UNIX)
	
	#ifdef HAVE_CONFIG_H
                #include "config.h"
        #endif	
	
	#if !defined ( USE_ALT_MEM_MANAGER)
		#include <stdlib.h>
	#endif
#else
	#include <malloc.h>
#endif
}

#ifdef USE_ALT_MEM_MANAGER

	// Mappings

	#define os_malloc			f_alt_malloc
	#define os_realloc		f_alt_realloc
	#define os_free			f_alt_free

#else

	#define os_malloc			malloc
	#define os_realloc		realloc
	#define os_free			free

#endif

typedef struct F_MemHdrTag
{
	FLMUINT			uiDataSize;
#ifdef FLM_DEBUG
	const char *	pszFileName;
	FLMINT			iLineNumber;
	FLMBOOL			bAllocFromNewOp;
	FLMUINT			uiAllocationId;
	FLMUINT			uiAllocCnt;
	FLMUINT *		puiStack;
#endif
#if FLM_ALIGN_SIZE == 8
	FLMUINT			uiDummy;
#endif
} F_MEM_HDR;

#define F_GET_ALLOC_PTR( pDataPtr) \
	(FLMBYTE *)((FLMBYTE *)(pDataPtr) - sizeof( F_MEM_HDR))

#define F_GET_DATA_PTR( pAllocPtr) \
	(FLMBYTE *)((FLMBYTE *)(pAllocPtr) + sizeof( F_MEM_HDR))

#define F_GET_MEM_DATA_SIZE( pDataPtr) \
	(((F_MEM_HDR *)(F_GET_ALLOC_PTR( pDataPtr)))->uiDataSize)

// Picket fence

#define F_PICKET_FENCE			"FFFFFFFF"
#if defined( FLM_DEBUG)
#define F_PICKET_FENCE_SIZE	8
#else
#define F_PICKET_FENCE_SIZE	0
#endif

#ifdef FLM_DEBUG
RCODE f_allocImp(								// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr,
	FLMBOOL			bAllocFromNewOp,
	const char *	pszFileName,
	FLMINT			iLineNumber);

RCODE f_callocImp(							// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber);

RCODE f_reallocImp(							// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber);

RCODE f_recallocImp(							// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber);

void f_resetStackInfoImp(					// Source: flalloc.cpp
	void *			pvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber);

#define f_alloc(uiSize,ppvPtr) \
				f_allocImp(uiSize,(void **)(ppvPtr),FALSE,__FILE__,__LINE__)

#define f_calloc(uiSize,ppvPtr) \
				f_callocImp(uiSize,(void **)(ppvPtr),__FILE__,__LINE__)

#define f_realloc(uiSize,ppvPtr) \
				f_reallocImp(uiSize,(void **)(ppvPtr),__FILE__,__LINE__)

#define f_recalloc(uiSize,ppvPtr) \
				f_recallocImp(uiSize,(void **)(ppvPtr),__FILE__,__LINE__)

#define f_resetStackInfo(pvPtr) \
				f_resetStackInfoImp(pvPtr,__FILE__,__LINE__)

#else
RCODE f_allocImp(								// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr);

RCODE f_callocImp(							// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr);

RCODE f_reallocImp(							// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr);

RCODE f_recallocImp(							// Source: flalloc.cpp
	FLMUINT			uiSize,
	void **			ppvPtr);

#define f_alloc(uiSize,ppvPtr) \
				f_allocImp(uiSize,(void **)(ppvPtr))

#define f_calloc(uiSize,ppvPtr) \
				f_callocImp(uiSize,(void **)(ppvPtr))

#define f_realloc(uiSize,ppvPtr) \
				f_reallocImp(uiSize,(void **)(ppvPtr))

#define f_recalloc(uiSize,ppvPtr) \
				f_recallocImp(uiSize,(void **)(ppvPtr))

#define f_resetStackInfo(pvPtr)

#endif

#define f_free(ppvPtr) f_freeImp( (void **)ppvPtr,FALSE)

void f_freeImp(								// Source: flalloc.cpp
	void **			ppvPtr,
	FLMBOOL			bFreeFromDeleteOp);

FINLINE FLMUINT f_msize(
	void *			pvPtr)
{
#if defined( FLM_UNIX) || defined( USE_ALT_MEM_MANAGER)
	return( pvPtr ? F_GET_MEM_DATA_SIZE( (pvPtr)) : 0);
#elif defined ( FLM_NLM)
		return( pvPtr ? msize( (F_GET_ALLOC_PTR( (pvPtr)))) : 0);
#else
		return( pvPtr ? _msize( (F_GET_ALLOC_PTR( (pvPtr)))) : 0);
#endif
}

void f_memoryInit( void);					// Source: flalloc.cpp

void f_memoryCleanup( void);				// Source: flalloc.cpp

FLMUINT * memWalkStack( void);

void logMemLeak(
	F_MEM_HDR *		pHdr);

#ifdef FLM_DEBUG
	#define f_new				new( __FILE__, __LINE__)
#else
	#define f_new				new
#endif

#endif // FTKMEM_H
