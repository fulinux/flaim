//-------------------------------------------------------------------------
// Desc:	Extended cache manager structures.
// Tabs:	3
//
//		Copyright (c) 2001-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ecache.h 12245 2006-01-19 14:29:51 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef ECACHE_H
#define ECACHE_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

typedef FLMUINT64					ESMADDR64;

// Prototypes

#ifdef FLM_NLM
	#pragma pack(push, 1)
#endif

typedef struct
{
	FLMUINT64		ui64TotalExtendedMemory;
	FLMUINT64		ui64RemainingExtendedMemory;
	FLMUINT32		ui32TotalMemoryBelow4Gig;
} ESMQueryInfo;

#ifdef FLM_NLM
	#pragma pack(pop)
#endif

typedef FLMUINT (* ESM_ALLOC_FUNC)(
	FLMUINT64			size,
	FLMUINT				options,
	ESMADDR64 *			esmAddress);

typedef FLMUINT (* ESM_FREE_FUNC)(
	ESMADDR64 			esmAddress);

typedef FLMUINT (* ESM_QUERY_FUNC)(
	FLMUINT32			ui32BufferSize,
	ESMQueryInfo *		pBuffer);

typedef FLMUINT (* ESM_ALLOC_WIN_FUNC)(
	FLMUINT32			ui32Size,
	FLMUINT32 *			pui32LogicalAddress,
	FLMUINT32			ui32Caller);

typedef FLMUINT (* ESM_FREE_WIN_FUNC)(
	FLMUINT32			ui32LogicalAddress,
	FLMUINT32			ui32Caller);

typedef FLMUINT (* ESM_MAP_MEM_FUNC)(
	FLMUINT32 			ui32WindowAddress,
	ESMADDR64			esmAddress,
	FLMUINT32			ui32Size);

typedef struct
{
	ESMADDR64		esmAddr;
	FLMUINT			uiStartBlkAddr;
} ECACHE_HDR;

/****************************************************************************
Desc: 	Class which provides database caching in extended server memory
****************************************************************************/
class FlmECache : public F_Base
{
public:

	// Constructor and Destructor

	FlmECache();

	virtual ~FlmECache();

	// Setup

	FLMBOOL setupECache(
		FLMUINT		uiBlockSize,
		FLMUINT		uiMaxFileSize);

	// Storage methods

	void putBlock(
		FLMUINT		uiBlockAddr,
		FLMBYTE *	pucBlock,
		FLMBOOL		bCalcChecksum = FALSE);

	void invalidateBlock(
		FLMUINT		uiBlockAddr);

	// Retrieval methods

	RCODE getBlock(
		FLMUINT		uiBlockAddr,
		FLMBYTE *	pucBlock,
		FLMUINT		uiLength);

	// Statistics

	void getStats(
		FLM_ECACHE_USAGE *		pUsage,
		FLMBOOL						bAddToCurrent = FALSE);

	FINLINE FLMUINT getPageSize( void)
	{
		return( m_uiPageSize);
	}

private:

	void cleanup( void);

	void getPosition(
		FLMUINT			uiBlockAddr,
		FLMUINT *		puiBlkOffsetInPage,
		FLMUINT *		puiExpectedPageStartOffset,
		ECACHE_HDR **	ppECacheHdr);

	FLMBOOL mapToWindow(
		ECACHE_HDR *	pHeader);

	FLMUINT					m_uiDbBlockSize;
	F_MUTEX					m_hMutex;
	ECACHE_HDR *			m_pAllocTable;
	FLMUINT					m_uiAllocTableSize;
	FLMUINT					m_uiPageSize;
	void *					m_pvWindow;
	FLMUINT64				m_ui64MappedESMAddr;
	FLMUINT64				m_ui64MaxFileSize;
	FLMUINT64				m_ui64BytesAllocated;
	FLMUINT64				m_ui64CacheHits;
	FLMUINT64				m_ui64CacheFaults;
	ESM_ALLOC_FUNC			m_fnESMAlloc;
	ESM_FREE_FUNC			m_fnESMFree;
	ESM_QUERY_FUNC			m_fnESMQuery;
	ESM_ALLOC_WIN_FUNC	m_fnESMAllocWindow;
	ESM_FREE_WIN_FUNC		m_fnESMFreeWindow;
	ESM_MAP_MEM_FUNC		m_fnESMMapMemory;
};

#include "fpackoff.h"

#endif // ECACHE_H
