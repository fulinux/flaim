//-------------------------------------------------------------------------
// Desc:	Extended cache manager.
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
// $Id: ecache.cpp 12245 2006-01-19 14:29:51 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef ECACHE_TEST
/****************************************************************************
Desc: Routine to simulate the ESMAlloc function on platforms that do not
		support ESM
****************************************************************************/
FLMUINT testESMAllocFunc(
	FLMUINT64			size,
	FLMUINT				options,
	ESMADDR64 *			esmAddress)
{
	FLMUINT		uiErr = 0;
	void *		pvAlloc;

	F_UNREFERENCED_PARM( options);

	if( RC_BAD( f_alloc( (FLMUINT)size, &pvAlloc)))
	{
		uiErr = 1;
		goto Exit;
	}

	*esmAddress = (FLMUINT64)pvAlloc;

Exit:

	return( uiErr);
}

/****************************************************************************
Desc: Routine to simulate the ESMFree function on platforms that do not
		support ESM
****************************************************************************/
FLMUINT testESMFreeFunc(
	ESMADDR64 			esmAddress)
{
	void *		pvAlloc = (void *)((FLMUINT)esmAddress);

	f_free( &pvAlloc);
	return( 0);
}

/****************************************************************************
Desc: Routine to simulate the ESMQuery function on platforms that do not
		support ESM
****************************************************************************/
FLMUINT testESMQueryFunc(
	FLMUINT32			ui32BufferSize,
	ESMQueryInfo *		pBuffer)
{
	F_UNREFERENCED_PARM( ui32BufferSize);

	f_memset( pBuffer, 0, sizeof( ESMQueryInfo));
	pBuffer->ui64TotalExtendedMemory = FLM_MAX_UINT64;
	pBuffer->ui64RemainingExtendedMemory = FLM_MAX_UINT64;
	pBuffer->ui32TotalMemoryBelow4Gig = FLM_MAX_UINT32;
	return( 0);
}

/****************************************************************************
Desc: Routine to simulate the ESMAllocWindow function on platforms that 
		do not support ESM
****************************************************************************/
FLMUINT testESMAllocWinFunc(
	FLMUINT32		ui32Size,
	FLMUINT32 *		pui32LogicalAddress,
	FLMUINT32		ui32Caller)
{
	F_UNREFERENCED_PARM( ui32Caller);
	*pui32LogicalAddress = FLM_MAX_UINT32;
	return( 0);

}

/****************************************************************************
Desc: Routine to simulate the ESMFreeWindow function on platforms that do not
		support ESM
****************************************************************************/
FLMUINT testESMFreeWinFunc(
	FLMUINT32			ui32LogicalAddress,
	FLMUINT32			ui32Caller)
{
	F_UNREFERENCED_PARM( ui32LogicalAddress);
	F_UNREFERENCED_PARM( ui32Caller);
	return( 0);
}

/****************************************************************************
Desc: Routine to simulate the ESMMapMemory function on platforms that do not
		support ESM
****************************************************************************/
FLMUINT testESMMapMemoryFunc(
	FLMUINT32 			ui32WindowAddress,
	ESMADDR64			esmAddress,
	FLMUINT32			ui32Size)
{
	F_UNREFERENCED_PARM( ui32WindowAddress);
	F_UNREFERENCED_PARM( esmAddress);
	F_UNREFERENCED_PARM( ui32Size);

	flmAssert( 0);
	return( 0);
}

#endif // ECACHE_TEST

/****************************************************************************
Desc: Constructor
****************************************************************************/
FlmECache::FlmECache()
{
	m_uiDbBlockSize = 0;
	m_uiPageSize = 0;
	m_ui64MaxFileSize = 0;
	m_hMutex = F_MUTEX_NULL;
	m_pAllocTable = NULL;
	m_uiAllocTableSize = 0;
	m_pvWindow = NULL;
	m_ui64MappedESMAddr = 0;
	m_ui64BytesAllocated = 0;
	m_ui64CacheHits = 0;
	m_ui64CacheFaults = 0;
	m_fnESMAlloc = NULL;
	m_fnESMFree = NULL;
	m_fnESMQuery = NULL;
	m_fnESMAllocWindow = NULL;
	m_fnESMFreeWindow = NULL;
	m_fnESMMapMemory = NULL;
}

/****************************************************************************
Desc: Destructor
****************************************************************************/
FlmECache::~FlmECache()
{
	cleanup();
}

/****************************************************************************
Desc: Frees all allocated resources
****************************************************************************/
void FlmECache::cleanup( void)
{
	FLMUINT			uiLoop;
	ECACHE_HDR *	pECache;

	if( m_pAllocTable)
	{
		for( uiLoop = 0, pECache = m_pAllocTable;
			uiLoop < m_uiAllocTableSize; uiLoop++, pECache++)
		{
			if( pECache->esmAddr != 0)
			{
				if( m_fnESMFree( pECache->esmAddr) != 0)
				{
					flmAssert( 0);
				}
				pECache->esmAddr = 0;
			}
		}

		f_free( &m_pAllocTable);
	}

	if( m_pvWindow)
	{
		m_fnESMFreeWindow( (FLMUINT32)((FLMUINT)m_pvWindow), 0);
		m_pvWindow = NULL;
		m_ui64MappedESMAddr = 0;
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc: Allocates resources needed by the ECache object
****************************************************************************/
FLMBOOL FlmECache::setupECache(
	FLMUINT			uiBlockSize,
	FLMUINT			uiMaxFileSize)
{
	FLMBOOL			bSetupOk = FALSE;
	ESMQueryInfo	esmQueryInfo;

	flmAssert( VALID_BLOCK_SIZE( uiBlockSize));
	flmAssert( uiMaxFileSize && (uiMaxFileSize % 4096) == 0);
	flmAssert( m_pAllocTable == NULL);

#if defined( FLM_NLM) && !defined( ECACHE_TEST)

	if( (m_fnESMAlloc = (ESM_ALLOC_FUNC)ImportPublicSymbol(
		(unsigned long)f_getNLMHandle(),
		(unsigned char *)"\x08" "ESMAlloc")) == NULL)
	{
		goto Exit;
	}

	if( (m_fnESMFree = (ESM_FREE_FUNC)ImportPublicSymbol(
		(unsigned long)f_getNLMHandle(),
		(unsigned char *)"\x07" "ESMFree")) == NULL)
	{
		goto Exit;
	}

	if( (m_fnESMQuery = (ESM_QUERY_FUNC)ImportPublicSymbol(
		(unsigned long)f_getNLMHandle(),
		(unsigned char *)"\x08" "ESMQuery")) == NULL)
	{
		goto Exit;
	}

	if( (m_fnESMAllocWindow = (ESM_ALLOC_WIN_FUNC)ImportPublicSymbol(
		(unsigned long)f_getNLMHandle(),
		(unsigned char *)"\x0E" "ESMAllocWindow")) == NULL)
	{
		goto Exit;
	}

	if( (m_fnESMFreeWindow = (ESM_FREE_WIN_FUNC)ImportPublicSymbol(
		(unsigned long)f_getNLMHandle(),
		(unsigned char *)"\x0D" "ESMFreeWindow")) == NULL)
	{
		goto Exit;
	}

	if( (m_fnESMMapMemory = (ESM_MAP_MEM_FUNC)ImportPublicSymbol(
		(unsigned long)f_getNLMHandle(),
		(unsigned char *)"\x0C" "ESMMapMemory")) == NULL)
	{
		goto Exit;
	}

#elif defined( ECACHE_TEST)

	m_fnESMAlloc = testESMAllocFunc;
	m_fnESMFree = testESMFreeFunc;
	m_fnESMQuery = testESMQueryFunc;
	m_fnESMAllocWindow = testESMAllocWinFunc;
	m_fnESMFreeWindow = testESMFreeWinFunc;
	m_fnESMMapMemory = testESMMapMemoryFunc;

#endif

	if (!m_fnESMQuery)
	{
		goto Exit;
	}

	// Query to see if the machine has any ESM memory.

	if( m_fnESMQuery( (unsigned)sizeof( ESMQueryInfo), &esmQueryInfo) != 0)
	{
		goto Exit;
	}

	// If no ESM, fail with a "not implemented" error

	if( !esmQueryInfo.ui64TotalExtendedMemory)
	{
		goto Exit;
	}

	// Determine the system's memory page size

#ifdef FLM_NLM
	m_uiPageSize = (4 * 1024 * 1024); // NetWare uses a fixed page size
#endif

	// Allocate the lookup table.  Set m_uiAllocTableSize to
	// 64 GB / 4MB (max supported memory on NetWare / min alloc
	// size in extended memory).

	flmAssert( m_uiPageSize != 0);
	m_uiAllocTableSize = (FLMUINT)(esmQueryInfo.ui64TotalExtendedMemory /
		(FLMUINT64)m_uiPageSize);

	if( m_uiAllocTableSize == 0)
	{
		goto Exit;
	}

	if (RC_BAD( f_calloc( 
		m_uiAllocTableSize * sizeof( ECACHE_HDR), &m_pAllocTable)))
	{
		goto Exit;
	}

	// Allocate a mutex

	if( RC_BAD( f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	// Round the maximum database file size up so that it
	// is a multiple of the allocation page size.  This makes
	// it easier to calculate the location of pages that belong
	// to a particular file.

	m_ui64MaxFileSize = uiMaxFileSize;
	if( m_ui64MaxFileSize % (FLMUINT64)m_uiPageSize)
	{
		m_ui64MaxFileSize += ((FLMUINT64)m_uiPageSize) - 
			(m_ui64MaxFileSize % (FLMUINT64)m_uiPageSize);
	}
	m_uiDbBlockSize = uiBlockSize;
	bSetupOk = TRUE;

Exit:

	if( !bSetupOk)
	{
		cleanup();
	}

	return( bSetupOk);
}

/****************************************************************************
Desc: Retreives a block or partial block from ECache
****************************************************************************/
RCODE FlmECache::getBlock(
	FLMUINT		uiBlockAddr,
	FLMBYTE *	pucBlock,
	FLMUINT		uiLength)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiFileNum = FSGetFileNumber( uiBlockAddr);
	FLMUINT			uiExpectedStartOffset;
	FLMUINT			uiBlkOffsetInPage;
	FLMBYTE *		pucSrcBlk;
	FLMBOOL			bMutexLocked = FALSE;
	ECACHE_HDR *	pHeader = NULL;

#ifdef FLM_DEBUG
	flmAssert( uiLength >= BH_OVHD);
	flmAssert( uiLength <= m_uiDbBlockSize);
#else
	F_UNREFERENCED_PARM( uiLength);
#endif

	getPosition( uiBlockAddr, &uiBlkOffsetInPage,
		&uiExpectedStartOffset, &pHeader);

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( !pHeader->esmAddr ||
		pHeader->uiStartBlkAddr != 
			FSBlkAddress( uiFileNum, uiExpectedStartOffset))
	{
		m_ui64CacheFaults++;
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Map the ESM page into our window

	if( !mapToWindow( pHeader))
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Copy the block into the destination buffer

	pucSrcBlk = ((FLMBYTE *)m_pvWindow) + uiBlkOffsetInPage;
	if( FB2UD( &pucSrcBlk[ BH_ADDR]) != 0)
	{
		FLMUINT		uiEncSize = getEncryptSize( pucSrcBlk);

		if( uiEncSize >= BH_OVHD && uiEncSize <= m_uiDbBlockSize)
		{
			f_memcpy( pucBlock, pucSrcBlk, uiEncSize);
			if( uiFileNum > 0 && uiFileNum < MAX_DATA_FILE_NUM_VER40 &&
				(GET_BH_ADDR( pucBlock) & 0xFFFFFF00) != 
					(uiBlockAddr & 0xFFFFFF00))
			{
				goto Invalidate_Block;
			}
			m_ui64CacheHits++;
		}
		else
		{
Invalidate_Block:

			// Invalidate the block

			UD2FBA( 0, pucSrcBlk); 
			m_ui64CacheFaults++;
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}
	else
	{
		m_ui64CacheFaults++;
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc: Puts a block or partial block into ECache
****************************************************************************/
void FlmECache::putBlock(
	FLMUINT		uiBlockAddr,
	FLMBYTE *	pucBlock,
	FLMBOOL		bCalcChecksum)
{
	FLMUINT			uiFileNum = FSGetFileNumber( uiBlockAddr);
	FLMUINT			uiExpectedStartOffset;
	FLMUINT			uiBlkOffsetInPage;
	FLMUINT			uiEncSize = getEncryptSize( pucBlock);
	FLMBYTE *		pucDestBlk;
	FLMBOOL			bMutexLocked = FALSE;
	FLMBOOL			bBlockWritten = FALSE;
	ECACHE_HDR *	pHeader = NULL;

	getPosition( uiBlockAddr, &uiBlkOffsetInPage,
		&uiExpectedStartOffset, &pHeader);

	if( uiEncSize < BH_OVHD || uiEncSize > m_uiDbBlockSize)
	{
		goto Exit;
	}

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( !pHeader->esmAddr ||
		pHeader->uiStartBlkAddr != 
			FSBlkAddress( uiFileNum, uiExpectedStartOffset))
	{
		if( !pHeader->esmAddr)
		{
			if( m_fnESMAlloc( m_uiPageSize, 0, &pHeader->esmAddr) != 0)
			{
				pHeader->esmAddr = 0;
				goto Exit;
			}
			m_ui64BytesAllocated += m_uiPageSize;
		}

		// Map the page into our local window

		if( !mapToWindow( pHeader))
		{
			goto Exit;
		}

		// Fill the block with zeros

		f_memset( m_pvWindow, 0, m_uiPageSize);
		
		// Update the cache header

		pHeader->uiStartBlkAddr = FSBlkAddress( uiFileNum, 
			uiExpectedStartOffset);
	}
	else
	{
		// Map the page into our local window

		if( !mapToWindow( pHeader))
		{
			goto Exit;
		}
	}

	flmAssert( uiFileNum == FSGetFileNumber( pHeader->uiStartBlkAddr));

	pucDestBlk = ((FLMBYTE *)m_pvWindow) + uiBlkOffsetInPage;
	f_memcpy( pucDestBlk, pucBlock, uiEncSize);
	bBlockWritten = TRUE;

	if( bCalcChecksum)
	{
		if( RC_BAD( BlkCheckSum( pucDestBlk, 
			CHECKSUM_SET, uiBlockAddr, uiEncSize)))
		{
			// Invalidate this block since we had an error.
			UD2FBA( 0, pucDestBlk); 
			goto Exit;
		}
	}

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;

Exit:

	// If we had an error, invalidate the entire page

	if( !bBlockWritten && pHeader && pHeader->esmAddr != 0)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( m_hMutex);
			bMutexLocked = TRUE;
		}

		// Invalidate the entire page

		m_fnESMFree( pHeader->esmAddr);
		pHeader->esmAddr = 0;
		pHeader->uiStartBlkAddr = 0;
		m_ui64BytesAllocated -= m_uiPageSize;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc: Invalidates a block in ECache so that it will not be returned via
		a call to getBlock()
****************************************************************************/
void FlmECache::invalidateBlock(
	FLMUINT		uiBlockAddr)
{
	FLMUINT			uiFileNum = FSGetFileNumber( uiBlockAddr);
	FLMUINT			uiExpectedStartOffset;
	FLMUINT			uiBlkOffsetInPage;
	FLMBOOL			bMutexLocked = FALSE;
	FLMBOOL			bBlockWritten = FALSE;
	ECACHE_HDR *	pHeader = NULL;

	getPosition( uiBlockAddr, &uiBlkOffsetInPage,
		&uiExpectedStartOffset, &pHeader);

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( pHeader->esmAddr != 0 &&
		pHeader->uiStartBlkAddr == 
			FSBlkAddress( uiFileNum, uiExpectedStartOffset))
	{
		flmAssert( uiBlkOffsetInPage < m_uiPageSize);
		flmAssert( (uiBlkOffsetInPage % m_uiDbBlockSize) == 0);

		// Map the page into our local window

		if( mapToWindow( pHeader))
		{
			// Set the block address to 0

			bBlockWritten = TRUE;
			UD2FBA( 0, ((FLMBYTE *)m_pvWindow) + uiBlkOffsetInPage); 
		}
	}

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;
}

/****************************************************************************
Desc: Returns ECache statistics
****************************************************************************/
void FlmECache::getStats(
	FLM_ECACHE_USAGE *	pUsage,
	FLMBOOL					bAddToCurrent)
{
	ESMQueryInfo			esmQueryInfo;

	if( !bAddToCurrent)
	{
		f_memset( pUsage, 0, sizeof( FLM_ECACHE_USAGE));
	}

	// Query to get ESM info

	if( m_fnESMQuery( (unsigned)sizeof( ESMQueryInfo), &esmQueryInfo) == 0)
	{
		pUsage->ui64TotalExtendedMemory = 
			esmQueryInfo.ui64TotalExtendedMemory;
		pUsage->ui64RemainingExtendedMemory = 
			esmQueryInfo.ui64RemainingExtendedMemory;
	}

	f_mutexLock( m_hMutex);

	pUsage->ui64TotalBytesAllocated += m_ui64BytesAllocated;
	pUsage->ui64CacheHits += m_ui64CacheHits;
	pUsage->ui64CacheFaults += m_ui64CacheFaults;

	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc: Determines the position of a block in ECache (memory page, etc.)
****************************************************************************/
void FlmECache::getPosition(
	FLMUINT			uiBlockAddr,
	FLMUINT *		puiBlkOffsetInPage,
	FLMUINT *		puiExpectedPageStartOffset,
	ECACHE_HDR **	ppECacheHdr)
{
	FLMUINT		uiPage;
	FLMUINT		uiFileNum = FSGetFileNumber( uiBlockAddr);
	FLMUINT		uiFileOffset = FSGetFileOffset( uiBlockAddr);
	FLMUINT		uiTableOffset;
	FLMUINT64	ui64BitAddr;

	ui64BitAddr = ((FLMUINT64)uiFileNum * m_ui64MaxFileSize) +
			(FLMUINT64)uiFileOffset;

	uiPage = (FLMUINT)(ui64BitAddr / (FLMUINT64)m_uiPageSize);

	uiTableOffset = uiPage % m_uiAllocTableSize;

	flmAssert( uiTableOffset < m_uiAllocTableSize);

	*ppECacheHdr = &m_pAllocTable[ uiTableOffset];

	*puiBlkOffsetInPage = uiFileOffset % m_uiPageSize;
	flmAssert( (*puiBlkOffsetInPage % m_uiDbBlockSize) == 0);

	*puiExpectedPageStartOffset = 
		(uiFileOffset / m_uiPageSize) * m_uiPageSize;
}

/****************************************************************************
Desc:		Maps a specified ESM page into our local address space
Notes:	This method assumes that the ECache mutex is locked
****************************************************************************/
FLMBOOL FlmECache::mapToWindow(
	ECACHE_HDR *	pHeader)
{
	FLMUINT64	ui64ESMAddr = pHeader->esmAddr;
	FLMBOOL		bMapped = FALSE;

	flmAssert( ui64ESMAddr != 0);

	if( ui64ESMAddr == m_ui64MappedESMAddr)
	{
		bMapped = TRUE;
		goto Exit;
	}

	if( !m_pvWindow)
	{
		FLMUINT32	ui32LogicalAddr;

		if( m_fnESMAllocWindow( (unsigned)m_uiPageSize, &ui32LogicalAddr, 0) != 0)
		{
			goto Exit;
		}

		m_pvWindow = (void *)((FLMUINT)ui32LogicalAddr);
	}

#ifdef ECACHE_TEST
	m_pvWindow = (void *)ui64ESMAddr;
#else
	if( m_fnESMMapMemory( (FLMUINT32)((FLMUINT)m_pvWindow),
		ui64ESMAddr, (unsigned)m_uiPageSize) != 0)
	{
		flmAssert( 0);
		m_ui64MappedESMAddr = 0;
		goto Exit;
	}

	m_ui64MappedESMAddr = ui64ESMAddr;
	bMapped = TRUE;
#endif

Exit:

	if( !bMapped)
	{
		// Invalidate the entire page
		m_fnESMFree( pHeader->esmAddr);
		pHeader->esmAddr = 0;
		pHeader->uiStartBlkAddr = 0;
		m_ui64BytesAllocated -= m_uiPageSize;
	}

	return( bMapped);
}
