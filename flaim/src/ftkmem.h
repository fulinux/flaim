//-------------------------------------------------------------------------
// Desc:	Memory allocation - definitions.
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
// $Id: ftkmem.h 12299 2006-01-19 15:01:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef FTKMEM_H
#define FTKMEM_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

	RCODE f_alloc(
		FLMUINT			uiSize,
		void **			ppvPtr,
		const char *	pszFile,
		int				iLine);

	#define f_alloc( uiSize, ppvPtr) \
		f_alloc( (uiSize), (void **)(ppvPtr), __FILE__, __LINE__)

	RCODE f_calloc(
		FLMUINT			uiSize,
		void **			ppvPtr,
		const char *	pszFile,
		int				iLine);

	#define f_calloc( uiSize, ppvPtr) \
		f_calloc( (uiSize), (void **)(ppvPtr), __FILE__, __LINE__)

	RCODE f_realloc(
		FLMUINT			uiSize,
		void **			ppvPtr,
		const char *	pszFile,
		int				iLine);

	#define f_realloc( uiSize, ppvPtr) \
		f_realloc( (uiSize), (void **)(ppvPtr), __FILE__, __LINE__)

	RCODE f_recalloc(
		FLMUINT			uiSize,
		void **			ppvPtr,
		const char *	pszFile,
		int				iLine);

	#define f_recalloc( uiSize, ppvPtr) \
		f_recalloc( (uiSize), (void **)(ppvPtr), __FILE__, __LINE__)

	void f_free(
		void **			ppvPtr);

	#define f_free( ppvPtr) \
		f_free( (void **)ppvPtr)

	FLMUINT f_msize(
		void *			pvPtr);

	FLMUINT * memWalkStack( void);

	typedef struct F_MemHdrTag
	{
		FLMUINT			uiDataSize;
	#ifdef FLM_DEBUG
		const char *	pszFileName;
		FLMINT			iLineNumber;
		FLMUINT			uiAllocationId;
		FLMUINT			uiAllocCnt;
		FLMUINT *		puiStack;
	#endif
	#if !defined( FLM_DEBUG) && FLM_ALIGN_SIZE == 8
		FLMUINT			uiDummy;
	#endif
	} F_MEM_HDR;

	void logMemLeak(
		F_MEM_HDR *		pHdr);

#include "fpackoff.h"

#endif
