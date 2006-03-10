//-------------------------------------------------------------------------
// Desc:	Memory management using fixed-size allocators - for dealing with
//			memory fragmentation issues.
// Tabs:	3
//
//		Copyright (c) 2004-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flfixed.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_NLM
	extern "C"
	{
		extern LONG	gv_lAllocRTag;
	}
#endif

/****************************************************************************
Desc:
****************************************************************************/
F_SlabManager::F_SlabManager()
{
	m_hMutex = F_MUTEX_NULL;
	m_pFirstInSlabList = NULL;
	m_pLastInSlabList = NULL;
	m_uiTotalSlabs = 0;
	m_uiAvailSlabs = 0;
	m_uiInUseSlabs = 0;
	m_pLowPrealloc = NULL;
	m_pHighPrealloc = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_SlabManager::~F_SlabManager()
{
	
	flmAssert( !m_uiInUseSlabs);
	flmAssert( m_uiAvailSlabs == m_uiTotalSlabs);
	
	freeAllSlabs();
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::setup(
	FLMUINT 		uiPreallocSize,
	FLMUINT		uiMinSlabSize)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiSysSlabSize = 0;
	FLMUINT		uiSlabSize = uiMinSlabSize;
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	// Determine the slab size

#ifdef FLM_WIN
	{
		SYSTEM_INFO		sysInfo;

		GetSystemInfo( &sysInfo);
		uiSysSlabSize = sysInfo.dwAllocationGranularity;
	}
#endif

	if( !uiSysSlabSize)
	{
		uiSysSlabSize = uiSlabSize;
	}

	// Round the given slab size up to the closest operating 
	// system slab size so we don't waste any memory.

	if( uiSlabSize % uiSysSlabSize)
	{
		m_uiSlabSize = ((uiSlabSize / uiSysSlabSize) + 1) * uiSysSlabSize;
	}
	else
	{
		m_uiSlabSize = uiSlabSize;
	}
	
	// Pre-allocate the requested amount of memory from the system
	
	if( uiPreallocSize)
	{
		if( RC_BAD( rc = resize( uiPreallocSize)))
		{
			goto Exit;
		}
	}
		
Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::resize(
	FLMUINT 				uiNumBytes,
	FLMUINT *			puiActualSize)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiSlabsNeeded;
	void *			pSlab;

	f_mutexLock( m_hMutex);
	
	if( puiActualSize)
	{
		*puiActualSize = 0;
	}
	
	uiSlabsNeeded = (uiNumBytes / m_uiSlabSize) + 
						 ((uiNumBytes % m_uiSlabSize) ? 1 : 0);
						 
	if( !uiSlabsNeeded && !m_uiInUseSlabs)
	{
		freeAllSlabs();
	}
	else if( m_uiTotalSlabs > uiSlabsNeeded)
	{
		// Do the best we can to free slabs.  We can only get rid of
		// slabs that aren't in use.
		
		if( RC_BAD( rc = sortSlabList()))
		{
			goto Exit;
		}
		
		while( m_pLastInSlabList && m_uiTotalSlabs > uiSlabsNeeded)
		{
			pSlab = m_pLastInSlabList;
			if( (m_pLastInSlabList = ((SLABHEADER *)pSlab)->pPrev) != NULL)
			{
				((SLABHEADER *)m_pLastInSlabList)->pNext = NULL;
			}
			else
			{
				m_pFirstInSlabList = NULL;
			}
			
			releaseSlabToSystem( pSlab);
			
			flmAssert( m_uiTotalSlabs);
			flmAssert( m_uiInUseSlabs);
			
			m_uiAvailSlabs--;
			m_uiTotalSlabs--;
		}
		
		if( !m_uiTotalSlabs)
		{
			flmAssert( !m_pFirstInSlabList);
			flmAssert( !m_pLastInSlabList);
			
			m_pLowPrealloc = NULL;
			m_pHighPrealloc = NULL;
		}
		else if( !uiNumBytes)
		{
			// Set the low and high pre-allocation pointers to NULL so that
			// slabs will be released back to the system when they are returned
			// to the slab manager.
			
			m_pLowPrealloc = NULL;
			m_pHighPrealloc = NULL;
		}
		else
		{
			if( m_pFirstInSlabList < m_pLowPrealloc)
			{
				m_pLowPrealloc = m_pFirstInSlabList;
			}
			
			if( m_pLastInSlabList > m_pHighPrealloc)
			{
				m_pHighPrealloc = m_pLastInSlabList;
			}
		}
	}
	else
	{
		// Allocate the required number of slabs
		
		while( m_uiTotalSlabs < uiSlabsNeeded)
		{
			if( (pSlab = allocSlabFromSystem()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
			
			if( !m_pLowPrealloc || pSlab < m_pLowPrealloc)
			{
				m_pLowPrealloc = pSlab;
			}
			
			if( pSlab > m_pHighPrealloc)
			{
				m_pHighPrealloc = pSlab;
			}
	
			// Touch every byte in the slab so that the operating system is
			// forced to immediately assign physical memory.
	
			f_memset( pSlab, 0, m_uiSlabSize);
			
			// Link the slab into the avail list
			
			if( m_pFirstInSlabList)
			{
				((SLABHEADER *)m_pFirstInSlabList)->pPrev = pSlab;
			}
			
			((SLABHEADER *)pSlab)->pNext = m_pFirstInSlabList;
			m_pFirstInSlabList = pSlab;
			
			if( !m_pLastInSlabList)
			{
				m_pLastInSlabList = pSlab;
			}
			
			m_uiTotalSlabs++;
			m_uiAvailSlabs++;
		}
	}
	
	if( puiActualSize)
	{
		*puiActualSize = m_uiTotalSlabs * m_uiSlabSize;
	}
	
Exit:

	if( RC_BAD( rc))
	{
		freeAllSlabs();
	}

	f_mutexUnlock( m_hMutex);
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::allocSlab(
	void **				ppSlab)
{
	RCODE		rc = FERR_OK;

	f_mutexLock( m_hMutex);
	
	if( m_pFirstInSlabList)
	{
		*ppSlab = m_pFirstInSlabList;
		if( (m_pFirstInSlabList = 
				((SLABHEADER *)m_pFirstInSlabList)->pNext) != NULL)
		{
			((SLABHEADER *)m_pFirstInSlabList)->pPrev = NULL;
		}
		else
		{
			m_pLastInSlabList = NULL;
		}
		
		((SLABHEADER *)*ppSlab)->pNext = NULL;
		
		flmAssert( m_uiAvailSlabs);
		m_uiAvailSlabs--;
		m_uiInUseSlabs++;
	}
	else
	{
		flmAssert( !m_uiAvailSlabs);
		
		if( (*ppSlab = allocSlabFromSystem()) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		m_uiTotalSlabs++;
		m_uiInUseSlabs++;
	}
	
Exit:

	f_mutexUnlock( m_hMutex);
	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
void F_SlabManager::freeSlab(
	void **				ppSlab)
{
	flmAssert( ppSlab && *ppSlab);
	
	f_mutexLock( m_hMutex);

	// There's no guarantee that out-of-band allocations will
	// fall outside of the preallocated address space, but
	// this is the best we can do.  The FLAIM cache system
	// is generally well-behaved.  Thus, even if an out-of-band
	// allocation falls within the prealloc range, we won't
	// exceed our allocation limits by too much.  It is better
	// to hold onto a little extra memory than to fail an allocation
	// request.
	
	if( m_pLowPrealloc && *ppSlab >= m_pLowPrealloc && *ppSlab <= m_pHighPrealloc)
	{
		((SLABHEADER *)*ppSlab)->pPrev = NULL;
		if( (((SLABHEADER *)*ppSlab)->pNext = m_pFirstInSlabList) != NULL)
		{
			((SLABHEADER *)m_pFirstInSlabList)->pPrev = *ppSlab;
		}
		else
		{
			m_pLastInSlabList = *ppSlab;
		}
		
		m_pFirstInSlabList = *ppSlab;
		*ppSlab = NULL;

		flmAssert( m_uiInUseSlabs);		
		m_uiInUseSlabs--;
		m_uiAvailSlabs++;
	}
	else
	{
		releaseSlabToSystem( *ppSlab);
		*ppSlab = NULL;
		
		flmAssert( m_uiTotalSlabs);
		flmAssert( m_uiInUseSlabs);
		
		m_uiTotalSlabs--;
		m_uiInUseSlabs--;
	}
	
	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_SlabManager::freeAllSlabs( void)
{
	void *			pNextSlab;
	SLABHEADER *	pSlabHeader;

	while( m_pFirstInSlabList)
	{
		pSlabHeader = (SLABHEADER *)m_pFirstInSlabList;
		pNextSlab = pSlabHeader->pNext;
		releaseSlabToSystem( m_pFirstInSlabList);
		m_pFirstInSlabList = pNextSlab;
	}
	
	m_uiTotalSlabs = 0;
	m_uiAvailSlabs = 0;
	m_pLowPrealloc = NULL;
	m_pHighPrealloc = NULL;
	m_pLastInSlabList = NULL;
}
	
/****************************************************************************
Desc:
****************************************************************************/
void * F_SlabManager::allocSlabFromSystem( void)
{
	void *		pSlab;
	
#if defined( FLM_OSX) && !defined( MAP_ANONYMOUS)
	#define MAP_ANONYMOUS		MAP_ANON
#endif
	
#ifdef FLM_WIN
	pSlab = VirtualAlloc( NULL,
		(DWORD)m_uiSlabSize, MEM_COMMIT, PAGE_READWRITE);
#elif defined( FLM_UNIX) && !defined( FLM_SOLARIS)
	if( (pSlab = mmap( 0, m_uiSlabSize, 
		PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED)
	{
		return( NULL);
	}
	
	// We don't use mmap on Solaris because of the amount of address
	// space it consumes for red-zones and rounding.  The following is from
	// the Solaris mmap man page:
	//	
	// The mmap() function aligns based on the length of  the  map-
	// ping.  When  determining  the  amount of space to add to the
	// address space, mmap() includes two  8-Kbyte  pages,  one  at
	// each  end  of the mapping that are not mapped and are there-
	// fore used as "red-zone" pages.  Attempts to reference  these
	// pages result in access violations.
	//	
	// The size requested is incremented by the 16 Kbytes for these
	// pages  and is then subject to rounding constraints. The con-
	// straints are:
	//
	// o	For 32-bit processes:
	//
	//     If length > 4 Mbytes
	//             round to 4-Mbyte multiple
	//     elseif length > 512 Kbytes
	//             round to 512-Kbyte multiple
	//     else
	//             round to 64-Kbyte multiple
	//
	// o	For 64-bit processes:
	//
	//     If length > 4 Mbytes
	//             round to 4-Mbyte multiple
	//     else
	//             round to 1-Mbyte multiple
	//
	// The net result is that for a 32-bit process:
	//
	// o	If an mmap() request is made for 4 Mbytes,  it  results
	//		in 4 Mbytes + 16 Kbytes and is rounded up to 8 Mbytes.
	//
	// o	If an mmap() request is made for 512 Kbytes, it results
	//		in 512 Kbytes + 16 Kbytes and is rounded up to 1 Mbyte.
	//
	// o	If an mmap() request is made for 1 Mbyte, it results in
	//		1 Mbyte + 16 Kbytes and is rounded up to 1.5 Mbytes.
	//
	// o	Each 8-Kbyte mmap request "consumes" 64 Kbytes of  vir-
	//		tual address space.
	
#else
	if( RC_BAD( f_alloc( m_uiSlabSize, &pSlab)))
	{
		return( NULL);
	}
#endif

	return( pSlab);
}
		
/****************************************************************************
Desc:
****************************************************************************/
void F_SlabManager::releaseSlabToSystem(
	void *		pSlab)
{
	flmAssert( pSlab);
	
#ifdef FLM_WIN
	VirtualFree( pSlab, 0, MEM_RELEASE);
#elif defined( FLM_UNIX) && !defined( FLM_SOLARIS)
	munmap( pSlab, m_uiSlabSize);
#else
	f_free( &pSlab);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT F_SlabManager::slabAddrCompareFunc(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	void *		pSlab1 = (((void **)pvBuffer)[ uiPos1]);
	void *		pSlab2 = (((void **)pvBuffer)[ uiPos2]);

	flmAssert( pSlab1 != pSlab2);

	if( pSlab1 < pSlab2)
	{
		return( -1);
	}

	return( 1);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_SlabManager::slabAddrSwapFunc(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	void **		ppSlab1 = &(((void **)pvBuffer)[ uiPos1]);
	void **		ppSlab2 = &(((void **)pvBuffer)[ uiPos2]);
	void *		pTmp;

	pTmp = *ppSlab1;
	*ppSlab1 = *ppSlab2;
	*ppSlab2 = pTmp;
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::sortSlabList( void)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLoop;
	void **			pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	void *			smallSortBuf[ SMALL_SORT_BUF_SIZE];
	void *			pCurSlab;
	void *			pPrevSib;

	if( m_uiAvailSlabs <= 1)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiAvailSlabs;

	// Sort the avail list according to the starting memory addresses of the
	// slabs

	if( uiMaxSortEntries <= SMALL_SORT_BUF_SIZE)
	{
		pSortBuf = smallSortBuf;
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiMaxSortEntries * sizeof( void *), &pSortBuf)))
		{
			goto Exit;
		}
	}
	
	pCurSlab = m_pFirstInSlabList;

	while( pCurSlab)
	{
		flmAssert( uiSortEntries != uiMaxSortEntries);
		pSortBuf[ uiSortEntries++] = pCurSlab;
		pCurSlab = ((SLABHEADER *)pCurSlab)->pNext;
	}
	
	flmAssert( uiSortEntries == uiMaxSortEntries);

	// Quick sort

	flmAssert( uiSortEntries);

	f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
		F_SlabManager::slabAddrCompareFunc,
		F_SlabManager::slabAddrSwapFunc);

	// Re-link the items in the list according to the new 
	// sort order

	m_pFirstInSlabList = NULL;
	m_pLastInSlabList = NULL;
	
	pCurSlab = NULL;
	pPrevSib = NULL;

	for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
	{
		pCurSlab = pSortBuf[ uiLoop];
		((SLABHEADER *)pCurSlab)->pNext = NULL;
		((SLABHEADER *)pCurSlab)->pPrev = NULL;

		if( pPrevSib)
		{
			((SLABHEADER *)pCurSlab)->pPrev = pPrevSib;
			((SLABHEADER *)pPrevSib)->pNext = pCurSlab;
		}
		else
		{
			m_pFirstInSlabList = pCurSlab;
		}

		pPrevSib = pCurSlab;
	}
	
	m_pLastInSlabList = pCurSlab;

Exit:

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_FixedAlloc::F_FixedAlloc()
{
	m_pSlabManager = NULL;
	m_pFirstBlock = NULL;
	m_pLastBlock = NULL;
	m_pFirstBlockWithAvailCells = NULL;
	m_pLastBlockWithAvailCells = NULL;
	m_uiBlocksWithAvailCells = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalFreeCells = 0;
	m_fnCanRelocate = NULL;
	m_fnRelocate = NULL;
	m_puiTotalBytesAllocated = NULL;
	m_uiSlabSize = 0;

	m_hLocalMutex = F_MUTEX_NULL;
	m_phMutex = NULL;
	
	m_uiAllocatedSlabs = 0;
	m_uiAllocatedCells = 0;
	m_uiAllocatedCellWatermark = 0;
	m_uiEverFreedCells = 0;
}

/****************************************************************************
Desc:	Destructor for F_FixedAlloc.  checks for memory leaks, and
		frees all memory in use.
****************************************************************************/
F_FixedAlloc::~F_FixedAlloc()
{
#ifdef FLM_DEBUG
	testForLeaks();
#endif

	freeAll();
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
	}

	if( m_hLocalMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hLocalMutex);
	}
}

/****************************************************************************
Desc:	Setup method for any setup that can fail 
****************************************************************************/
RCODE F_FixedAlloc::setup(
	F_SlabManager *	pSlabManager,
	FLMBOOL				bUseMutex,
	FLMUINT				uiCellSize,
	FLMUINT *			puiTotalBytesAllocated)
{
	RCODE			rc = FERR_OK;

	flmAssert( pSlabManager);
	flmAssert( uiCellSize);
	
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	m_uiCellSize = uiCellSize;
	m_puiTotalBytesAllocated = puiTotalBytesAllocated;
	m_uiSlabSize = m_pSlabManager->getSlabSize();

	// Get the alloc-aligned versions of all the sizes

	m_uiBlockHeaderSize = getAllocAlignedSize( sizeof( BLOCK));
	m_uiCellHeaderSize = getAllocAlignedSize( sizeof( CELLHEADER));
	m_uiCellSize = getAllocAlignedSize( m_uiCellSize);

	// Ensure that there's enough space for our overhead

	flmAssert( m_uiCellSize >= sizeof( CELLAVAILNEXT));

	m_uiSizeOfCellAndHeader = m_uiCellHeaderSize + m_uiCellSize;

	m_uiCellsPerBlock =
		(m_uiSlabSize - m_uiBlockHeaderSize) /
		m_uiSizeOfCellAndHeader;

	flmAssert( m_uiCellsPerBlock);
	flmAssert( (m_uiCellsPerBlock * m_uiCellSize) < m_uiSlabSize);
		
	if( bUseMutex)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hLocalMutex)))
		{
			goto Exit;
		}
		
		m_phMutex = &m_hLocalMutex;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Setup method for any setup that can fail 
****************************************************************************/
RCODE F_FixedAlloc::setup(
	F_SlabManager *	pSlabManager,
	F_MUTEX *			phMutex,
	FLMUINT				uiCellSize,
	FLMUINT *			puiTotalBytesAllocated)
{
	RCODE		rc = FERR_OK;
	
	if( RC_BAD( rc = setup( pSlabManager, (FLMBOOL)FALSE, 
		uiCellSize, puiTotalBytesAllocated)))
	{
		goto Exit;
	}
	
	if( phMutex)
	{
		m_phMutex = phMutex;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Private, internal method to fetch a cell
****************************************************************************/
void * F_FixedAlloc::getCell( void)
{
	BLOCK *			pBlock = NULL;
	FLMBYTE *		pCell = NULL;
	CELLHEADER *	pHeader;

	// If there's a block that has an avail cell, that one gets priority

	if( (pBlock = m_pFirstBlockWithAvailCells) != NULL)
	{
		flmAssert( pBlock->ui32AvailCellCount <= m_uiTotalFreeCells);
		flmAssert( m_uiTotalFreeCells);
		flmAssert( pBlock->ui32AllocatedCells < m_uiCellsPerBlock);

		pCell = m_pFirstBlockWithAvailCells->pLocalAvailCellListHead;
		flmAssert( pCell);

		pHeader = (CELLHEADER *)((FLMBYTE *)pCell - m_uiCellHeaderSize);

		pBlock->ui32AllocatedCells++;
		pBlock->ui32AvailCellCount--;
		m_uiTotalFreeCells--;
		
		// An avail cell holds as its contents the next pointer in the avail chain.
		// Avail chains do not span blocks.

		pBlock->pLocalAvailCellListHead = ((CELLAVAILNEXT *)pCell)->pNextInList;

		// If there are no other avail cells in this block at this point,
		// then we need to unlink the block from the
		// blocks-with-avail-cells list, headed by m_pFirstBlockWithAvailCells

		if( !pBlock->pLocalAvailCellListHead)
		{
			// Save a copy of the block we're going to unlink

			BLOCK * pBlockToUnlink = pBlock;

			// Need to keep the NULLNESS of the content of the cell consistent
			// with the block's ui32AvailCellCount being equal to 0

			flmAssert( !pBlockToUnlink->ui32AvailCellCount);

			// There can't be a pPrevBlockWithAvailCells since
			// we're positioned to the first one

			flmAssert( !pBlockToUnlink->pPrevBlockWithAvailCells);				

			// Update m_pFirstBlockWithAvailCells to point to the next one

			if( (m_pFirstBlockWithAvailCells =
				pBlockToUnlink->pNextBlockWithAvailCells) == NULL)
			{
				flmAssert( m_pLastBlockWithAvailCells == pBlockToUnlink);
				m_pLastBlockWithAvailCells = NULL;
			}

			// Unlink from blocks-with-avail-cells list

			if( pBlockToUnlink->pNextBlockWithAvailCells)
			{
				pBlockToUnlink->
					pNextBlockWithAvailCells->pPrevBlockWithAvailCells =
					pBlockToUnlink->pPrevBlockWithAvailCells;
				pBlockToUnlink->pNextBlockWithAvailCells = NULL;
				
				flmAssert( !pBlockToUnlink->pPrevBlockWithAvailCells);
			}

			// Decrement the block count

			flmAssert( m_uiBlocksWithAvailCells);
			m_uiBlocksWithAvailCells--;
		}
	}
	else
	{
		// If our m_pFirstBlock is completely full, or there is no
		// m_pFirstBlock, it is time to allocate a new block

		if( !m_pFirstBlock ||
			(m_pFirstBlock->ui32NextNeverUsedCell == m_uiCellsPerBlock))
		{
			BLOCK *			pNewBlock;

			if( (pNewBlock = getAnotherBlock()) == NULL)
			{
				goto Exit;
			}

			if( m_pFirstBlock)
			{
				pNewBlock->pNext = m_pFirstBlock;
				m_pFirstBlock->pPrev = pNewBlock;
			}
			else
			{
				m_pLastBlock = pNewBlock;
			}

			m_pFirstBlock = pNewBlock;
		}

		pBlock = m_pFirstBlock;
		pBlock->ui32AllocatedCells++;
		flmAssert( pBlock->ui32AllocatedCells <= m_uiCellsPerBlock);
		
		pHeader = (CELLHEADER *)
			((FLMBYTE *)pBlock + m_uiBlockHeaderSize +
				(m_uiSizeOfCellAndHeader * m_pFirstBlock->ui32NextNeverUsedCell));

		pCell = ((FLMBYTE *)pHeader + m_uiCellHeaderSize);		
		m_pFirstBlock->ui32NextNeverUsedCell++;
	}

	pHeader->pContainingBlock = pBlock;

#ifdef FLM_DEBUG
	if (gv_FlmSysData.bTrackLeaks && gv_FlmSysData.bStackWalk)
	{
		pHeader->puiStack = memWalkStack();
	}
	else
	{
		pHeader->puiStack = NULL;
	}
#endif

	m_uiAllocatedCells++;
	m_uiAllocatedCellWatermark = 
		f_max( m_uiAllocatedCells, m_uiAllocatedCellWatermark);

Exit:

	return( pCell);
}

/****************************************************************************
Desc:	Public method to free a cell of memory back to the system. 
****************************************************************************/
void F_FixedAlloc::freeCell(
	void *		pCell,
	FLMBOOL		bMutexLocked,
	FLMBOOL		bFreeIfEmpty,
	FLMBOOL *	pbFreedBlock)
{
	CELLAVAILNEXT *	pCellContents;
	CELLHEADER *		pHeader;
	BLOCK *				pBlock;
	FLMBOOL				bUnlockMutex = FALSE;

	if( pbFreedBlock)
	{
		*pbFreedBlock = FALSE;
	}

	if( !pCell)
	{
		return;
	}
	
	if( !bMutexLocked && m_phMutex)
	{
		f_mutexLock( *m_phMutex);
		bUnlockMutex = TRUE;
	}

	pCellContents = (CELLAVAILNEXT *)pCell;
	pHeader = (CELLHEADER *)(((FLMBYTE *)pCell) - m_uiCellHeaderSize);
	pBlock = pHeader->pContainingBlock;

	flmAssert( pBlock);
	flmAssert( pBlock->pvAllocator == (void *)this);

	pHeader->pContainingBlock = NULL;
	
#ifdef FLM_DEBUG
	if( pHeader->puiStack)
	{
		os_free( pHeader->puiStack);
		pHeader->puiStack = NULL;
	}
#endif

	// Should always be set on a free
	
	flmAssert( m_pFirstBlock);
	
	// Add the cell to the pBlock's free list

	pCellContents->pNextInList = pBlock->pLocalAvailCellListHead;

#ifdef FLM_DEBUG
	// Write out a string that's easy to see in memory when debugging

	f_strcpy( pCellContents->szDebugPattern, "FREECELL");
#endif

	flmAssert( pCell);
	pBlock->pLocalAvailCellListHead = (FLMBYTE *)pCell;
	pBlock->ui32AvailCellCount++;

	flmAssert( pBlock->ui32AllocatedCells);
	pBlock->ui32AllocatedCells--;

	// If there's no chain, make this one the first

	if( !m_pFirstBlockWithAvailCells)
	{
		m_pFirstBlockWithAvailCells = pBlock;
		m_pLastBlockWithAvailCells = pBlock;
		flmAssert( !pBlock->pNextBlockWithAvailCells);
		flmAssert( !pBlock->pPrevBlockWithAvailCells);
		m_uiBlocksWithAvailCells++;
		m_bAvailListSorted = TRUE;
	}
	else if( pBlock->ui32AvailCellCount == 1)
	{
		// This item is not linked in to the chain, so link it in

		if( m_bAvailListSorted && pBlock > m_pFirstBlockWithAvailCells)
		{
			m_bAvailListSorted = FALSE;
		}

		pBlock->pNextBlockWithAvailCells = m_pFirstBlockWithAvailCells;
		pBlock->pPrevBlockWithAvailCells = NULL;
		m_pFirstBlockWithAvailCells->pPrevBlockWithAvailCells = pBlock;
		m_pFirstBlockWithAvailCells = pBlock;
		m_uiBlocksWithAvailCells++;
	}

	// Adjust counter, because the cell is now considered free

	m_uiTotalFreeCells++;

	// If this block is now totally avail

	if( pBlock->ui32AvailCellCount == m_uiCellsPerBlock)
	{
		flmAssert( !pBlock->ui32AllocatedCells);

		// If we have met our threshold for being able to free a block

		if( m_uiTotalFreeCells >= m_uiCellsPerBlock || bFreeIfEmpty)
		{
			freeBlock( pBlock);
			if( pbFreedBlock)
			{
				*pbFreedBlock = TRUE;
			}
		}
		else if( pBlock != m_pFirstBlockWithAvailCells)
		{
			// Link the block to the front of the avail list so that
			// it can be freed quickly at some point in the future

			if( pBlock->pPrevBlockWithAvailCells)
			{
				pBlock->pPrevBlockWithAvailCells->pNextBlockWithAvailCells =
					pBlock->pNextBlockWithAvailCells;
			}

			if( pBlock->pNextBlockWithAvailCells)
			{
				pBlock->pNextBlockWithAvailCells->pPrevBlockWithAvailCells =
					pBlock->pPrevBlockWithAvailCells;
			}
			else
			{
				flmAssert( m_pLastBlockWithAvailCells == pBlock);
				m_pLastBlockWithAvailCells = pBlock->pPrevBlockWithAvailCells;
			}

			if( m_pFirstBlockWithAvailCells)
			{
				m_pFirstBlockWithAvailCells->pPrevBlockWithAvailCells = pBlock;
			}

			pBlock->pPrevBlockWithAvailCells = NULL;
			pBlock->pNextBlockWithAvailCells = m_pFirstBlockWithAvailCells;
			m_pFirstBlockWithAvailCells = pBlock;
		}
	}

	m_uiAllocatedCells--;
	m_uiEverFreedCells++;

	if( bUnlockMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:	Grabs another slab of memory from the operating system
****************************************************************************/
F_FixedAlloc::BLOCK * F_FixedAlloc::getAnotherBlock( void)
{
	BLOCK *	pBlock = NULL;
	
	if( RC_BAD( m_pSlabManager->allocSlab( (void **)&pBlock)))
	{
		goto Exit;
	}

	if (m_puiTotalBytesAllocated)
	{
		(*m_puiTotalBytesAllocated) += m_uiSlabSize;
	}

	// Initialize the block header fields

	f_memset( pBlock, 0, sizeof( BLOCK));
	pBlock->pvAllocator = (void *)this;
	m_uiAllocatedSlabs++;

Exit:
	
	return( pBlock);
}

/****************************************************************************
Desc:	Private internal method to free an unused empty block back to the OS.
****************************************************************************/
void F_FixedAlloc::freeBlock(
	BLOCK *			pBlock)
{
	flmAssert( pBlock);
	flmAssert( !pBlock->ui32AllocatedCells);
	flmAssert( pBlock->pvAllocator == this);
	
	// Unlink from all-blocks-list

	if( pBlock->pNext)
	{
		pBlock->pNext->pPrev = pBlock->pPrev;
	}
	else
	{
		m_pLastBlock = pBlock->pPrev;
	}

	if( pBlock->pPrev)
	{
		pBlock->pPrev->pNext = pBlock->pNext;
	}
	else
	{
		m_pFirstBlock = pBlock->pNext;
	}

	// Unlink from blocks-with-avail-cells list

	if( pBlock->pNextBlockWithAvailCells)
	{
		pBlock->pNextBlockWithAvailCells->pPrevBlockWithAvailCells =
			pBlock->pPrevBlockWithAvailCells;
	}
	else
	{
		m_pLastBlockWithAvailCells = pBlock->pPrevBlockWithAvailCells;
	}

	if( pBlock->pPrevBlockWithAvailCells)
	{
		pBlock->pPrevBlockWithAvailCells->pNextBlockWithAvailCells =
			pBlock->pNextBlockWithAvailCells;
	}
	else
	{
		m_pFirstBlockWithAvailCells = pBlock->pNextBlockWithAvailCells;
	}

	flmAssert( m_uiBlocksWithAvailCells);
	m_uiBlocksWithAvailCells--;
	flmAssert( m_uiTotalFreeCells >= pBlock->ui32AvailCellCount);
	m_uiTotalFreeCells -= pBlock->ui32AvailCellCount;
	m_uiAllocatedSlabs--;

	if (m_puiTotalBytesAllocated)
	{
		flmAssert( *m_puiTotalBytesAllocated >= m_uiSlabSize);
		(*m_puiTotalBytesAllocated) -= m_uiSlabSize; 
	}
	
	m_pSlabManager->freeSlab( (void **)&pBlock);
}

/****************************************************************************
Desc:	Public method to free all the memory in the system.  
****************************************************************************/
void F_FixedAlloc::freeAll( void)
{
	BLOCK *		pFreeMe;

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}

	while( m_pFirstBlock)
	{
		pFreeMe = m_pFirstBlock;
		m_pFirstBlock = m_pFirstBlock->pNext;
		freeBlock( pFreeMe);
	}

	flmAssert( !m_uiTotalFreeCells);

	m_pFirstBlock = NULL;
	m_pLastBlock = NULL;
	m_pFirstBlockWithAvailCells = NULL;
	m_pLastBlockWithAvailCells = NULL;
	m_uiBlocksWithAvailCells = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalFreeCells = 0;
	m_uiAllocatedSlabs = 0;
	m_uiAllocatedCells = 0;
	
	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/
void F_FixedAlloc::getStats(
	FLM_ALLOC_USAGE *	pUsage)
{
	f_memset( pUsage, 0, sizeof( FLM_ALLOC_USAGE));

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}

	pUsage->ui64Slabs = m_uiAllocatedSlabs;
	pUsage->ui64SlabBytes = m_uiAllocatedSlabs * m_uiSlabSize;
	pUsage->ui64AllocatedCells = m_uiAllocatedCells;
	pUsage->ui64FreeCells = m_uiTotalFreeCells;

	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:		If a relocation callback function has been registered, and memory 
			can be compressed, the avail list will be compressed
****************************************************************************/ 
void F_FixedAlloc::defragmentMemory( void)
{
	BLOCK *			pCurBlock;
	BLOCK *			pPrevSib;
	CELLHEADER *	pCellHeader;
	FLMBOOL			bBlockFreed;
	FLMBYTE *		pucOriginal;
	FLMBYTE *		pucReloc = NULL;
	FLMUINT			uiLoop;
	BLOCK **			pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	BLOCK *			smallSortBuf[ SMALL_SORT_BUF_SIZE];

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}

	if( !m_fnRelocate || m_uiTotalFreeCells < m_uiCellsPerBlock)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiBlocksWithAvailCells;

	// Re-sort the blocks in the avail list according to
	// their memory addresses to help reduce logical fragmentation

	if( !m_bAvailListSorted && uiMaxSortEntries > 1)
	{
		if( uiMaxSortEntries <= SMALL_SORT_BUF_SIZE)
		{
			pSortBuf = smallSortBuf;
		}
		else
		{
			if( RC_BAD( f_alloc( uiMaxSortEntries * sizeof( BLOCK *), &pSortBuf)))
			{
				goto Exit;
			}
		}

		pCurBlock = m_pFirstBlockWithAvailCells;

		while( pCurBlock)
		{
			flmAssert( uiSortEntries != uiMaxSortEntries);
			pSortBuf[ uiSortEntries++] = pCurBlock;
			pCurBlock = pCurBlock->pNextBlockWithAvailCells;
		}

		// Quick sort

		flmAssert( uiSortEntries);

		f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
			F_FixedAlloc::blockAddrCompareFunc,
			F_FixedAlloc::blockAddrSwapFunc);

		// Re-link the items in the list according to the new 
		// sort order

		m_pFirstBlockWithAvailCells = NULL;
		m_pLastBlockWithAvailCells = NULL;

		pCurBlock = NULL;
		pPrevSib = NULL;

		for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
		{
			pCurBlock = pSortBuf[ uiLoop];
			pCurBlock->pNextBlockWithAvailCells = NULL;
			pCurBlock->pPrevBlockWithAvailCells = NULL;

			if( pPrevSib)
			{
				pCurBlock->pPrevBlockWithAvailCells = pPrevSib;
				pPrevSib->pNextBlockWithAvailCells = pCurBlock;
			}
			else
			{
				m_pFirstBlockWithAvailCells = pCurBlock;
			}

			pPrevSib = pCurBlock;
		}

		m_pLastBlockWithAvailCells = pCurBlock;
		m_bAvailListSorted = TRUE;
	}

	// Process the avail list (which should be sorted unless
	// we are too low on memory)

	pCurBlock = m_pLastBlockWithAvailCells;

	while( pCurBlock)
	{
		if( m_uiTotalFreeCells < m_uiCellsPerBlock)
		{
			// No need to continue ... we aren't above the
			// free cell threshold

			goto Exit;
		}

		pPrevSib = pCurBlock->pPrevBlockWithAvailCells;

		if( pCurBlock == m_pFirstBlockWithAvailCells ||
				!pCurBlock->ui32AvailCellCount)
		{
			// We've either hit the beginning of the avail list or
			// the block that we are now positioned on has been
			// removed from the avail list.  In either case,
			// we are done.

			break;
		}

		if( pCurBlock->ui32AvailCellCount == m_uiCellsPerBlock ||
			pCurBlock->ui32NextNeverUsedCell == pCurBlock->ui32AvailCellCount)
		{
			freeBlock( pCurBlock);
			pCurBlock = pPrevSib;
			continue;
		}

		for( uiLoop = 0; uiLoop < pCurBlock->ui32NextNeverUsedCell &&
			pCurBlock != m_pFirstBlockWithAvailCells &&
			m_uiTotalFreeCells >= m_uiCellsPerBlock; uiLoop++)
		{
			pCellHeader = (CELLHEADER *)
				((FLMBYTE *)pCurBlock + m_uiBlockHeaderSize +
					(uiLoop * m_uiSizeOfCellAndHeader));

			if( pCellHeader->pContainingBlock)
			{
				// If pContainingBlock is non-NULL, the cell is currently allocated

				flmAssert( pCellHeader->pContainingBlock == pCurBlock);

				pucOriginal = ((FLMBYTE *)pCellHeader + m_uiCellHeaderSize);

				if( !m_fnCanRelocate || m_fnCanRelocate( pucOriginal))
				{
					if( (pucReloc = (FLMBYTE *)getCell()) == NULL)
					{
						goto Exit;
					}

					f_memcpy( pucReloc, pucOriginal, m_uiCellSize);
					m_fnRelocate( pucOriginal, pucReloc);
					freeCell( pucOriginal, TRUE, TRUE, &bBlockFreed);
					
					if( bBlockFreed)
					{
						break;
					}
				}
			}
		}

		pCurBlock = pPrevSib;
	}

Exit:

	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
void F_FixedAlloc::incrementTotalBytesAllocated(
	FLMUINT					uiCount)
{
	if( m_puiTotalBytesAllocated)
	{
		if( m_phMutex)
		{
			f_mutexLock( *m_phMutex);
		}
		
		*m_puiTotalBytesAllocated += uiCount;	
		
		if( m_phMutex)
		{
			f_mutexUnlock( *m_phMutex);
		}
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
FLMUINT F_FixedAlloc::getTotalBytesAllocated( void)
{
	FLMUINT		uiTotal = 0;
	
	if( m_puiTotalBytesAllocated)
	{
		if( m_phMutex)
		{
			f_mutexLock( *m_phMutex);
		}
		
		uiTotal = *m_puiTotalBytesAllocated;	
		
		if( m_phMutex)
		{
			f_mutexUnlock( *m_phMutex);
		}
	}
	
	return( uiTotal);
}
	
/****************************************************************************
Desc:		
****************************************************************************/ 
void F_FixedAlloc::decrementTotalBytesAllocated(
	FLMUINT					uiCount)
{
	if( m_puiTotalBytesAllocated)
	{
		if( m_phMutex)
		{
			f_mutexLock( *m_phMutex);
		}
		
		flmAssert( *m_puiTotalBytesAllocated >= uiCount);
		*m_puiTotalBytesAllocated -= uiCount;	
		
		if( m_phMutex)
		{
			f_mutexUnlock( *m_phMutex);
		}
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
void F_FixedAlloc::freeUnused( void)
{
	BLOCK *			pBlock;

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}

	if( (pBlock = m_pFirstBlockWithAvailCells) != NULL &&
		!pBlock->ui32AllocatedCells)
	{
		freeBlock( pBlock);
	}

	if( (pBlock = m_pFirstBlock) != NULL &&
		!pBlock->ui32AllocatedCells)
	{
		freeBlock( pBlock);
	}

	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:	Debug method to do mem leak testing.  Any cells allocated via
		allocCell but not freed via freeCell() will be triggered here.
****************************************************************************/ 
#ifdef FLM_DEBUG
void F_FixedAlloc::testForLeaks( void)
{
	BLOCK *			pBlockRover = m_pFirstBlock;
	CELLHEADER *	pHeader;
	FLMUINT			uiLoop;
	F_MEM_HDR		memHeader;

	// Test for leaks

	while( pBlockRover)
	{
		for( uiLoop = 0; uiLoop < pBlockRover->ui32NextNeverUsedCell; uiLoop++)
		{			
			pHeader = (CELLHEADER*)
				((FLMBYTE*)pBlockRover + m_uiBlockHeaderSize +
					(uiLoop * m_uiSizeOfCellAndHeader));

			// Nonzero here means we have a leak

			if( pHeader->pContainingBlock)
			{
				// We have a leak, so let's call logMemLeak with the
				// appropriate header passed in

				f_memset( &memHeader, 0, sizeof( F_MEM_HDR));
				memHeader.uiDataSize = m_uiCellSize;
				memHeader.puiStack = pHeader->puiStack;
				logMemLeak( &memHeader);
			}
		}

		pBlockRover = pBlockRover->pNext;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/ 
F_BufferAlloc::~F_BufferAlloc()
{
	FLMUINT	uiLoop;

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}
	
	for (uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->Release();
			m_ppAllocators[ uiLoop] = NULL;
		}
	}

	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
	}
	
	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_BufferAlloc::setup(
	F_SlabManager *	pSlabManager,
	F_MUTEX *			phMutex,
	FLMUINT * 			puiTotalBytesAllocated)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiSize;
	
	flmAssert( pSlabManager);
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	m_puiTotalBytesAllocated = puiTotalBytesAllocated;
	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( (m_ppAllocators[ uiLoop] = f_new F_FixedAlloc) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		switch (uiLoop)
		{
			case 0: 
				uiSize = CELL_SIZE_0;
				break;
			case 1: 
				uiSize = CELL_SIZE_1;
				break;
			case 2: 
				uiSize = CELL_SIZE_2;
				break;
			case 3: 
				uiSize = CELL_SIZE_3;
				break;
			case 4: 
				uiSize = CELL_SIZE_4;
				break;
			case 5: 
				uiSize = CELL_SIZE_5;
				break;
			case 6: 
				uiSize = CELL_SIZE_6;
				break;
			case 7: 
				uiSize = CELL_SIZE_7;
				break;
			case 8:
				uiSize = CELL_SIZE_8;
				break;
			case 9: 
				uiSize = CELL_SIZE_9;
				break;
			case 10:
				uiSize = CELL_SIZE_10;
				break;
			case 11: 
				uiSize = CELL_SIZE_11;
				break;
			case 12: 
				uiSize = CELL_SIZE_12;
				break;
			case 13: 
				uiSize = CELL_SIZE_13;
				break;
			case 14: 
				uiSize = CELL_SIZE_14;
				break;
			case 15: 
				uiSize = CELL_SIZE_15;
				break;
			case 16: 
				uiSize = CELL_SIZE_16;
				break;
			case 17: 
				uiSize = CELL_SIZE_17;
				break;
			case 18: 
				uiSize = CELL_SIZE_18;
				break;
			case 19: 
				uiSize = CELL_SIZE_19;
				break;
			case 20: 
				uiSize = CELL_SIZE_20;
				break;
			case 21: 
				uiSize = CELL_SIZE_21;
				break;
			default:
				uiSize = 0;
				flmAssert( 0);
				break;
		}

		if (RC_BAD( rc = m_ppAllocators[ uiLoop]->setup( 
			pSlabManager, (FLMBOOL)FALSE, uiSize, puiTotalBytesAllocated)))
		{
			goto Exit;
		}
	}
	
	m_phMutex = phMutex;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_BufferAlloc::setRelocationFuncs(
	FLM_CAN_RELOC_FUNC	fnCanRelocate,
	FLM_RELOC_FUNC			fnRelocate)
{
	FLMUINT		uiLoop;

	flmAssert( fnCanRelocate);
	flmAssert( fnRelocate);

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}
	
	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->setRelocationFuncs( 
				fnCanRelocate, fnRelocate);
		}

		uiLoop++;
	}
	
	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_BufferAlloc::allocBuf(
	FLMUINT				uiSize,
	void *				pvInitialData,
	FLMUINT				uiDataSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL *			pbAllocatedOnHeap)
{
	RCODE					rc = FERR_OK;
	F_FixedAlloc *		pAllocator = getAllocator( uiSize);

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}
	
	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	if( pAllocator)
	{
		flmAssert( pAllocator->getCellSize() >= uiSize);

		if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( 
			pvInitialData, uiDataSize)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiSize, ppucBuffer)))
		{
			goto Exit;
		}
		
		if( m_puiTotalBytesAllocated)
		{
			(*m_puiTotalBytesAllocated) += f_msize( *ppucBuffer);
		}
		
		if( pvInitialData)
		{
			f_memcpy( *ppucBuffer, pvInitialData, uiDataSize);
		}
		
		if( pbAllocatedOnHeap)
		{
			*pbAllocatedOnHeap = TRUE;
		}
	}
	
Exit:
	
	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_BufferAlloc::reallocBuf(
	FLMUINT				uiOldSize,
	FLMUINT				uiNewSize,
	void *				pvInitialData,
	FLMUINT				uiDataSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL *			pbAllocatedOnHeap)
{
	RCODE					rc = FERR_OK;
	FLMBYTE *			pucTmp;
	F_FixedAlloc *		pOldAllocator;
	F_FixedAlloc *		pNewAllocator;
	FLMBOOL				bMutexLocked = FALSE;

	flmAssert( uiNewSize);

	if( !uiOldSize)
	{
		rc = allocBuf( uiNewSize, pvInitialData, uiDataSize, 
			ppucBuffer, pbAllocatedOnHeap);
		goto Exit;
	}
	
	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
		bMutexLocked = TRUE;
	}

	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	pOldAllocator = getAllocator( uiOldSize);
	pNewAllocator = getAllocator( uiNewSize);

	if( pOldAllocator)
	{
		if( pNewAllocator)
		{
			if( pOldAllocator == pNewAllocator)
			{
				// The allocation will still fit in the same cell

				goto Exit;
			}

			if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = f_alloc( uiNewSize, &pucTmp)))
			{
				goto Exit;
			}
			
			if( m_puiTotalBytesAllocated)
			{
				(*m_puiTotalBytesAllocated) += f_msize( pucTmp);
			}
			
			if( pbAllocatedOnHeap)
			{
				*pbAllocatedOnHeap = TRUE;
			}
		}

		f_memcpy( pucTmp, *ppucBuffer, f_min( uiOldSize, uiNewSize));
		pOldAllocator->freeCell( *ppucBuffer);
		*ppucBuffer = pucTmp;
	}
	else
	{
		if( pNewAllocator)
		{
			if( m_puiTotalBytesAllocated)
			{
				FLMUINT	uiAllocSize = f_msize( *ppucBuffer);
				
				flmAssert( *m_puiTotalBytesAllocated >= uiAllocSize);
				(*m_puiTotalBytesAllocated) -= uiAllocSize;
			}
			
			if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( 
				*ppucBuffer, f_min( uiOldSize, uiNewSize))) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			f_free( ppucBuffer);
			*ppucBuffer = pucTmp;
		}
		else
		{
			FLMUINT		uiAllocSize = f_msize( *ppucBuffer);
	
			flmAssert( uiOldSize > m_ppAllocators[ NUM_BUF_ALLOCATORS - 1]->getCellSize());
			flmAssert( uiNewSize > m_ppAllocators[ NUM_BUF_ALLOCATORS - 1]->getCellSize());
			
			if( RC_BAD( rc = f_realloc( uiNewSize, ppucBuffer)))
			{
				goto Exit;
			}
			
			if( m_puiTotalBytesAllocated)
			{
				flmAssert( *m_puiTotalBytesAllocated >= uiAllocSize);
				(*m_puiTotalBytesAllocated) -= uiAllocSize;
				(*m_puiTotalBytesAllocated) += f_msize( *ppucBuffer);
			}
			
			if( pbAllocatedOnHeap)
			{
				*pbAllocatedOnHeap = TRUE;
			}
		}
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( *m_phMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_BufferAlloc::freeBuf(
	FLMUINT				uiSize,
	FLMBYTE **			ppucBuffer)
{
	F_FixedAlloc *		pAllocator = getAllocator( uiSize);
	
	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}

	if( pAllocator)
	{
		pAllocator->freeCell( *ppucBuffer, FALSE, TRUE, NULL);
		*ppucBuffer = NULL;
	}
	else
	{
		if( m_puiTotalBytesAllocated)
		{
			FLMUINT		uiAllocSize = f_msize( *ppucBuffer);
			
			flmAssert( *m_puiTotalBytesAllocated >= uiAllocSize);
			(*m_puiTotalBytesAllocated) -= uiAllocSize;
		}
		
		f_free( ppucBuffer);
	}
	
	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_BufferAlloc::defragmentMemory( void)
{
	FLMUINT	uiLoop;

	if( m_phMutex)
	{
		f_mutexLock( *m_phMutex);
	}
	
	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->defragmentMemory();
			m_ppAllocators[ uiLoop]->freeUnused();
		}

		uiLoop++;
	}
	
	if( m_phMutex)
	{
		f_mutexUnlock( *m_phMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
FLMUINT F_BufferAlloc::getTrueSize(
	FLMUINT				uiSize,
	FLMBYTE *			pucBuffer)
{
	FLMUINT				uiTrueSize;
	F_FixedAlloc *		pAllocator;

	if( !uiSize)
	{
		uiTrueSize = 0;
	}
	else if( (pAllocator = getAllocator( uiSize)) != NULL)
	{
		uiTrueSize = pAllocator->getCellSize();
	}
	else
	{
		uiTrueSize = f_msize( pucBuffer);
	}

	return( uiTrueSize);
}

/****************************************************************************
Desc:
****************************************************************************/ 
F_FixedAlloc * F_BufferAlloc::getAllocator(
	FLMUINT				uiSize)
{
	F_FixedAlloc *		pAllocator;

	flmAssert( uiSize);
	
	if( uiSize <= CELL_SIZE_10)
	{
		if( uiSize <= CELL_SIZE_4)
		{
			if( uiSize <= CELL_SIZE_2)
			{
				if( uiSize <= CELL_SIZE_0)
				{
					pAllocator = (F_FixedAlloc *)m_ppAllocators [0];
				}
				else
				{
					pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_1
															? m_ppAllocators [1]
															: m_ppAllocators [2]);
				}
			}
			else
			{
				pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_3
														? m_ppAllocators [3]
														: m_ppAllocators [4]);
			}
		}
		else if( uiSize <= CELL_SIZE_7)
		{
			if( uiSize <= CELL_SIZE_5)
			{
				pAllocator = (F_FixedAlloc *)m_ppAllocators [5];
			}
			else
			{
				pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_6
														? m_ppAllocators [6]
														: m_ppAllocators [7]);
			}
		}
		else
		{
			if( uiSize <= CELL_SIZE_8)
			{
				pAllocator = (F_FixedAlloc *)m_ppAllocators [8];
			}
			else
			{
				pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_9
														? m_ppAllocators [9]
														: m_ppAllocators [10]);
			}
		}
	}
	else if( uiSize <= CELL_SIZE_16)
	{
		if( uiSize <= CELL_SIZE_13)
		{
			if( uiSize <= CELL_SIZE_11)
			{
				pAllocator = (F_FixedAlloc *)m_ppAllocators [11];
			}
			else
			{
				pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_12
														? m_ppAllocators [12]
														: m_ppAllocators [13]);
			}
		}
		else
		{
			if( uiSize <= CELL_SIZE_14)
			{
				pAllocator = (F_FixedAlloc *)m_ppAllocators [14];
			}
			else
			{
				pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_15
														? m_ppAllocators [15]
														: m_ppAllocators [16]);
			}
		}
	}
	else if( uiSize <= CELL_SIZE_19)
	{
		if( uiSize <= CELL_SIZE_17)
		{
			pAllocator = (F_FixedAlloc *)m_ppAllocators [17];
		}
		else
		{
			pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_18
													? m_ppAllocators [18]
													: m_ppAllocators [19]);
		}
	}
	else if( uiSize <= CELL_SIZE_21)
	{
		pAllocator = (F_FixedAlloc *)(uiSize <= CELL_SIZE_20
												? m_ppAllocators [20]
												: m_ppAllocators [21]);
	}
	else
	{
		pAllocator = NULL;
	}

	return( pAllocator);
}
