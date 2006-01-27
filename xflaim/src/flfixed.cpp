//------------------------------------------------------------------------------
// Desc:	Special allocators for making many fixed-size allocations.
//
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
// $Id: flfixed.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_SOLARIS
	#include <fcntl.h>
#endif

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
	m_uiTotalBytesAllocated = 0;
	m_pFirstInSlabList = NULL;
	m_pLastInSlabList = NULL;
	m_uiTotalSlabs = 0;
	m_uiAvailSlabs = 0;
	m_uiInUseSlabs = 0;
	m_uiPreallocSlabs = 0;
#ifdef FLM_SOLARIS
	m_DevZero = -1;
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
F_SlabManager::~F_SlabManager()
{
	
	flmAssert( !m_uiInUseSlabs);
	flmAssert( m_uiAvailSlabs == m_uiTotalSlabs);

	freeAllSlabs();
	
	flmAssert( !m_uiTotalBytesAllocated);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
	
#ifdef FLM_SOLARIS
	if( m_DevZero > 0)
	{
		close( m_DevZero);
	}
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::setup(
	FLMUINT 				uiPreallocSize)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiSysSlabSize = 0;
	FLMUINT		uiSlabSize = 64 * 1024;
	
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

#ifdef FLM_SOLARIS
	if( (m_DevZero = open( "/dev/zero", O_RDWR)) == -1)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
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
		lockMutex();

		if( RC_BAD( rc = resize( uiPreallocSize, NULL, TRUE)))
		{
			goto Exit;
		}

		unlockMutex();
	}
		
Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::resize(
	FLMUINT 			uiNumBytes,
	FLMUINT *		puiActualSize,
	FLMBOOL			bMutexLocked)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiSlabsNeeded;
	void *			pSlab;
	FLMBOOL			bUnlockMutex = FALSE;

	if( !bMutexLocked)
	{
		lockMutex();
		bUnlockMutex = TRUE;
	}
	
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
	}
	else
	{
		// Allocate the required number of slabs
		
		while( m_uiTotalSlabs < uiSlabsNeeded)
		{
			if( (pSlab = allocSlabFromSystem()) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
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

	m_uiPreallocSlabs = m_uiTotalSlabs;
	
Exit:

	if( RC_BAD( rc))
	{
		freeAllSlabs();
	}

	if( bUnlockMutex)
	{
		unlockMutex();
	}
	
	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_SlabManager::allocSlab(
	void **				ppSlab,
	FLMBOOL				bMutexLocked)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bUnlockMutex = FALSE;

	if( !bMutexLocked)
	{
		lockMutex();
		bUnlockMutex = TRUE;
	}
	
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
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
		
		m_uiTotalSlabs++;
		m_uiInUseSlabs++;
	}
	
Exit:

	if( bUnlockMutex)
	{
		unlockMutex();
	}
	
	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
void F_SlabManager::freeSlab(
	void **				ppSlab,
	FLMBOOL				bMutexLocked)
{
	FLMBOOL				bUnlockMutex = FALSE;
	
	flmAssert( ppSlab && *ppSlab);
	
	if( !bMutexLocked)
	{
		lockMutex();
		bUnlockMutex = TRUE;
	}

	if( m_uiTotalSlabs <= m_uiPreallocSlabs)
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
	
	if( bUnlockMutex)
	{
		unlockMutex();
	}
}

/****************************************************************************
Desc:	Assumes that the mutex is locked
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
		m_uiTotalSlabs--;
		m_uiAvailSlabs--;
	}
	
	flmAssert( !m_uiAvailSlabs);
	m_pLastInSlabList = NULL;
}
	
/****************************************************************************
Desc:	Assumes that the mutex is locked
****************************************************************************/
void * F_SlabManager::allocSlabFromSystem( void)
{
	void *		pSlab;
	
#ifdef FLM_WIN
	pSlab = VirtualAlloc( NULL,
		(DWORD)m_uiSlabSize, MEM_COMMIT, PAGE_READWRITE);
#elif defined( FLM_SOLARIS)
	if( (pSlab = mmap( 0, m_uiSlabSize, 
		PROT_READ | PROT_WRITE, MAP_PRIVATE, m_DevZero, 0)) == MAP_FAILED)
	{
		return( NULL);
	}
#elif defined( FLM_UNIX)

#ifndef MAP_ANONYMOUS
	#define MAP_ANONYMOUS	MAP_ANON
#endif

	if( (pSlab = mmap( 0, m_uiSlabSize, 
		PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED)
	{
		return( NULL);
	}
#else
	if( RC_BAD( f_alloc( m_uiSlabSize, &pSlab)))
	{
		return( NULL);
	}
#endif

	incrementTotalBytesAllocated( m_uiSlabSize, TRUE);

	return( pSlab);
}
		
/****************************************************************************
Desc:	Assumes that the mutex is locked
****************************************************************************/
void F_SlabManager::releaseSlabToSystem(
	void *		pSlab)
{
	flmAssert( pSlab);
	
#ifdef FLM_WIN
	VirtualFree( pSlab, 0, MEM_RELEASE);
#elif defined( FLM_SOLARIS)
	munmap( (char *)pSlab, m_uiSlabSize);
#elif defined( FLM_UNIX)
	munmap( pSlab, m_uiSlabSize);
#else
	f_free( &pSlab);
#endif

	decrementTotalBytesAllocated( m_uiSlabSize, TRUE);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_CACHE_PROTECT	
void F_SlabManager::protectSlab(
	void *		pSlab)
{
#ifdef FLM_WIN
	(void)pSlab;
	DWORD		dOldProtect;
	VirtualProtect( pSlab, m_uiSlabSize, PAGE_READONLY, &dOldProtect);
	flmAssert( dOldProtect == PAGE_READWRITE);
#elif defined( FLM_UNIX)
	mprotect( pSlab, m_uiSlabSize, PROT_READ);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_CACHE_PROTECT	
void F_SlabManager::unprotectSlab(
	void *		pSlab)
{
#ifdef FLM_WIN
	(void)pSlab;
	DWORD		dOldProtect;
	VirtualProtect( pSlab, m_uiSlabSize, PAGE_READWRITE, &dOldProtect);
	flmAssert( dOldProtect == PAGE_READONLY);
#elif defined( FLM_UNIX)
	mprotect( pSlab, m_uiSlabSize, PROT_READ | PROT_WRITE);
#endif
}
#endif

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
	RCODE				rc = NE_XFLM_OK;
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
	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pRelocator = NULL;
	m_pFirstSlabWithAvailCells = NULL;
	m_pLastSlabWithAvailCells = NULL;
	m_uiSlabsWithAvailCells = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalFreeCells = 0;
	m_uiSlabSize = 0;
	m_pUsageStats = NULL;
	
#ifdef FLM_CACHE_PROTECT	
	m_bMemProtectionEnabled = FALSE;
#endif
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
}

/****************************************************************************
Desc:	Setup method for any setup that can fail 
****************************************************************************/
RCODE F_FixedAlloc::setup(
	IF_Relocator *			pRelocator,
	F_SlabManager *		pSlabManager,
	FLMBOOL					bMemProtect,
	FLMUINT					uiCellSize,
	XFLM_SLAB_USAGE *		pUsageStats)
{
	RCODE			rc = NE_XFLM_OK;

	flmAssert( pSlabManager);
	flmAssert( uiCellSize);
	flmAssert( pUsageStats != NULL);
	
	m_pUsageStats = pUsageStats;
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	m_pRelocator = pRelocator;
	m_uiCellSize = uiCellSize;
	m_uiSlabSize = m_pSlabManager->getSlabSize();

	// Get the alloc-aligned versions of all the sizes

	m_uiSlabHeaderSize = getAllocAlignedSize( sizeof( SLAB));
	if (pRelocator)
	{
		m_uiCellHeaderSize = getAllocAlignedSize( sizeof( CELLHEADER));
	}
	else
	{
		m_uiCellHeaderSize = getAllocAlignedSize( sizeof( CELLHEADER2));
	}
	m_uiCellSize = getAllocAlignedSize( m_uiCellSize);

	// Ensure that there's enough space for our overhead

	flmAssert( m_uiCellSize >= sizeof( CELLAVAILNEXT));

	m_uiSizeOfCellAndHeader = m_uiCellHeaderSize + m_uiCellSize;

	m_uiCellsPerSlab =
		(m_uiSlabSize - m_uiSlabHeaderSize) /
		m_uiSizeOfCellAndHeader;

	flmAssert( m_uiCellsPerSlab);
	flmAssert( m_uiCellsPerSlab <= FLM_MAX_UINT16);
	flmAssert( (m_uiCellsPerSlab * m_uiCellSize) < m_uiSlabSize);
	
#ifdef FLM_CACHE_PROTECT	
	m_bMemProtectionEnabled = bMemProtect;
#else
	F_UNREFERENCED_PARM( bMemProtect);
#endif
	
	return( rc);
}

/****************************************************************************
Desc:	Private, internal method to fetch a cell
****************************************************************************/
void * F_FixedAlloc::getCell(
	IF_Relocator *		pRelocator)
{
	SLAB *			pSlab = NULL;
	FLMBYTE *		pCell = NULL;
	CELLHEADER *	pHeader;

	// If there's a slab that has an avail cell, that one gets priority

	if( (pSlab = m_pFirstSlabWithAvailCells) != NULL)
	{
#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( pSlab, TRUE);
#endif
		
		flmAssert( pSlab->ui16AvailCellCount <= m_uiTotalFreeCells);
		flmAssert( m_uiTotalFreeCells);
		flmAssert( pSlab->ui16AllocatedCells < m_uiCellsPerSlab);

		pCell = m_pFirstSlabWithAvailCells->pLocalAvailCellListHead;
		flmAssert( pCell);

		pHeader = (CELLHEADER *)((FLMBYTE *)pCell - m_uiCellHeaderSize);
		pSlab->ui16AllocatedCells++;
		pSlab->ui16AvailCellCount--;
		m_uiTotalFreeCells--;
		
		// An avail cell holds as its contents the next pointer in the avail chain.
		// Avail chains do not span slabs.

		pSlab->pLocalAvailCellListHead = ((CELLAVAILNEXT *)pCell)->pNextInList;

		// If there are no other avail cells in this slab at this point,
		// then we need to unlink the slab from the
		// slabs-with-avail-cells list, headed by m_pFirstSlabWithAvailCells

		if( !pSlab->pLocalAvailCellListHead)
		{
			// Save a copy of the slab we're going to unlink

			SLAB * pSlabToUnlink = pSlab;

			// Need to keep the NULLNESS of the content of the cell consistent
			// with the slab's ui16AvailCellCount being equal to 0

			flmAssert( !pSlabToUnlink->ui16AvailCellCount);

			// There can't be a pPrevSlabWithAvailCells since
			// we're positioned to the first one

			flmAssert( !pSlabToUnlink->pPrevSlabWithAvailCells);				

			// Update m_pFirstSlabWithAvailCells to point to the next one

			if( (m_pFirstSlabWithAvailCells =
				pSlabToUnlink->pNextSlabWithAvailCells) == NULL)
			{
				flmAssert( m_pLastSlabWithAvailCells == pSlabToUnlink);
				m_pLastSlabWithAvailCells = NULL;
			}

			// Unlink from slabs-with-avail-cells list

			if( pSlabToUnlink->pNextSlabWithAvailCells)
			{
#ifdef FLM_CACHE_PROTECT	
				unprotectSlab( pSlabToUnlink->pNextSlabWithAvailCells, TRUE);
#endif
				
				pSlabToUnlink->
					pNextSlabWithAvailCells->pPrevSlabWithAvailCells =
					pSlabToUnlink->pPrevSlabWithAvailCells;

#ifdef FLM_CACHE_PROTECT	
				protectSlab( pSlabToUnlink->pNextSlabWithAvailCells, TRUE);
#endif
				pSlabToUnlink->pNextSlabWithAvailCells = NULL;
			}

			// Decrement the slab count

			flmAssert( m_uiSlabsWithAvailCells);
			m_uiSlabsWithAvailCells--;
		}
	}
	else
	{
		// If our m_pFirstSlab is completely full, or there is no
		// m_pFirstSlab, it is time to allocate a new slab

		if( !m_pFirstSlab ||
			(m_pFirstSlab->ui16NextNeverUsedCell == m_uiCellsPerSlab))
		{
			SLAB *			pNewSlab;

			if( (pNewSlab = getAnotherSlab()) == NULL)
			{
				goto Exit;
			}

			if( m_pFirstSlab)
			{
#ifdef FLM_CACHE_PROTECT	
				unprotectSlab( pNewSlab, TRUE);
#endif
				pNewSlab->pNext = m_pFirstSlab;
#ifdef FLM_CACHE_PROTECT	
				protectSlab( pNewSlab, TRUE);
#endif

#ifdef FLM_CACHE_PROTECT	
				unprotectSlab( m_pFirstSlab, TRUE);
#endif
				m_pFirstSlab->pPrev = pNewSlab;
#ifdef FLM_CACHE_PROTECT	
				protectSlab( m_pFirstSlab, TRUE);
#endif
			}
			else
			{
				m_pLastSlab = pNewSlab;
			}

			m_pFirstSlab = pNewSlab;
		}

		pSlab = m_pFirstSlab;

#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( pSlab, TRUE);
#endif
		pSlab->ui16AllocatedCells++;
		
#ifdef FLM_CACHE_PROTECT	
		flmAssert( pSlab->ui16AllocatedCells <= m_uiCellsPerSlab);
#endif
		
		pHeader = (CELLHEADER *)
				((FLMBYTE *)pSlab + m_uiSlabHeaderSize +
					(m_uiSizeOfCellAndHeader * m_pFirstSlab->ui16NextNeverUsedCell));

		pCell = ((FLMBYTE *)pHeader + m_uiCellHeaderSize);		
		m_pFirstSlab->ui16NextNeverUsedCell++;
	}

	pHeader->pContainingSlab = pSlab;

#ifdef FLM_DEBUG
	if (gv_XFlmSysData.bTrackLeaks && gv_XFlmSysData.bStackWalk)
	{
		pHeader->puiStack = memWalkStack();
	}
	else
	{
		pHeader->puiStack = NULL;
	}
#endif
	if (!m_pRelocator)
	{
		((CELLHEADER2 *)pHeader)->pRelocator = pRelocator;
	}

#ifdef FLM_CACHE_PROTECT	
	protectSlab( pSlab, TRUE);
#endif
	
	m_pUsageStats->ui64AllocatedCells++;

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
	FLMBOOL *	pbFreedSlab)
{
	CELLAVAILNEXT *	pCellContents;
	CELLHEADER *		pHeader;
	SLAB *				pSlab;
	FLMBOOL				bUnlockMutex = FALSE;
#ifdef FLM_CACHE_PROTECT	
	FLMBOOL				bProtectSlab = FALSE;
#endif

	if( pbFreedSlab)
	{
		*pbFreedSlab = FALSE;
	}

	if( !pCell)
	{
		return;
	}
	
	if( !bMutexLocked)
	{
		m_pSlabManager->lockMutex();
		bUnlockMutex = TRUE;
	}

	pCellContents = (CELLAVAILNEXT *)pCell;
	pHeader = (CELLHEADER *)(((FLMBYTE *)pCell) - m_uiCellHeaderSize);
	pSlab = pHeader->pContainingSlab;

	// Memory corruption detected!

	if( !pSlab || pSlab->pvAllocator != (void *)this)
	{
		flmAssert( 0);
		goto Exit;
	}

#ifdef FLM_CACHE_PROTECT	
	unprotectSlab( pSlab, TRUE);
	bProtectSlab = TRUE;
#endif
	
	pHeader->pContainingSlab = NULL;
#ifdef FLM_DEBUG
	if( pHeader->puiStack)
	{
		os_free( pHeader->puiStack);
		pHeader->puiStack = NULL;
	}
#endif

	// Should always be set on a free
	
	flmAssert( m_pFirstSlab);
	
	// Add the cell to the pSlab's free list

	pCellContents->pNextInList = pSlab->pLocalAvailCellListHead;

#ifdef FLM_DEBUG
	// Write out a string that's easy to see in memory when debugging

	f_strcpy( pCellContents->szDebugPattern, (FLMBYTE*)"FREECELL");
#endif

	flmAssert( pCell);
	pSlab->pLocalAvailCellListHead = (FLMBYTE *)pCell;
	pSlab->ui16AvailCellCount++;

	flmAssert( pSlab->ui16AllocatedCells);
	pSlab->ui16AllocatedCells--;

	// If there's no chain, make this one the first

	if( !m_pFirstSlabWithAvailCells)
	{
		m_pFirstSlabWithAvailCells = pSlab;
		m_pLastSlabWithAvailCells = pSlab;
		flmAssert( !pSlab->pNextSlabWithAvailCells);
		flmAssert( !pSlab->pPrevSlabWithAvailCells);
		m_uiSlabsWithAvailCells++;
		m_bAvailListSorted = TRUE;
	}
	else if( pSlab->ui16AvailCellCount == 1)
	{
		// This item is not linked in to the chain, so link it in

		if( m_bAvailListSorted && pSlab > m_pFirstSlabWithAvailCells)
		{
			m_bAvailListSorted = FALSE;
		}

		pSlab->pNextSlabWithAvailCells = m_pFirstSlabWithAvailCells;
		pSlab->pPrevSlabWithAvailCells = NULL;

#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( m_pFirstSlabWithAvailCells, TRUE);
#endif
		m_pFirstSlabWithAvailCells->pPrevSlabWithAvailCells = pSlab;
#ifdef FLM_CACHE_PROTECT	
		protectSlab( m_pFirstSlabWithAvailCells, TRUE);
#endif
		m_pFirstSlabWithAvailCells = pSlab;
		m_uiSlabsWithAvailCells++;
	}

	// Adjust counter, because the cell is now considered free

	m_uiTotalFreeCells++;

	// If this slab is now totally avail

	if( pSlab->ui16AvailCellCount == m_uiCellsPerSlab)
	{
		flmAssert( !pSlab->ui16AllocatedCells);

		// If we have met our threshold for being able to free a slab

		if( m_uiTotalFreeCells >= m_uiCellsPerSlab || bFreeIfEmpty)
		{
#ifdef FLM_CACHE_PROTECT	
			protectSlab( pSlab, TRUE);
			bProtectSlab = FALSE;
#endif

			freeSlab( pSlab);

			if( pbFreedSlab)
			{
				*pbFreedSlab = TRUE;
			}
		}
		else if( pSlab != m_pFirstSlabWithAvailCells)
		{
			// Link the slab to the front of the avail list so that
			// it can be freed quickly at some point in the future

			if( pSlab->pPrevSlabWithAvailCells)
			{
				pSlab->pPrevSlabWithAvailCells->pNextSlabWithAvailCells =
					pSlab->pNextSlabWithAvailCells;
			}

			if( pSlab->pNextSlabWithAvailCells)
			{
				pSlab->pNextSlabWithAvailCells->pPrevSlabWithAvailCells =
					pSlab->pPrevSlabWithAvailCells;
			}
			else
			{
				flmAssert( m_pLastSlabWithAvailCells == pSlab);
				m_pLastSlabWithAvailCells = pSlab->pPrevSlabWithAvailCells;
			}

			if( m_pFirstSlabWithAvailCells)
			{
				m_pFirstSlabWithAvailCells->pPrevSlabWithAvailCells = pSlab;
			}

			pSlab->pPrevSlabWithAvailCells = NULL;
			pSlab->pNextSlabWithAvailCells = m_pFirstSlabWithAvailCells;
			m_pFirstSlabWithAvailCells = pSlab;
		}
	}

	m_pUsageStats->ui64AllocatedCells--;

Exit:

#ifdef FLM_CACHE_PROTECT	
	if( bProtectSlab)
	{
		protectSlab( pSlab, TRUE);
	}
#endif
	
	if( bUnlockMutex)
	{
		m_pSlabManager->unlockMutex();
	}

	return;
}

/****************************************************************************
Desc:	Grabs another slab of memory from the operating system
****************************************************************************/
F_FixedAlloc::SLAB * F_FixedAlloc::getAnotherSlab( void)
{
	SLAB *	pSlab = NULL;
	
	if( RC_BAD( m_pSlabManager->allocSlab( (void **)&pSlab, TRUE)))
	{
		goto Exit;
	}

	// Initialize the slab header fields

	f_memset( pSlab, 0, sizeof( SLAB));
	pSlab->pvAllocator = (void *)this;
	m_pUsageStats->ui64Slabs++;

#ifdef FLM_CACHE_PROTECT	
	if( m_bMemProtectionEnabled)
	{
		m_pSlabManager->protectSlab( pSlab);
	}
#endif
	
Exit:
	
	return( pSlab);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_CACHE_PROTECT	
void F_FixedAlloc::protectSlab(
	SLAB *			pSlab,
	FLMBOOL			bMutexLocked)
{
	FLMBOOL			bUnlockMutex = FALSE;

	if( !m_bMemProtectionEnabled)
	{
		return;
	}
	
	if( !bMutexLocked)
	{
		m_pSlabManager->lockMutex();
		bUnlockMutex = TRUE;
	}
	
	flmAssert( pSlab->pvAllocator == this);
	flmAssert( pSlab->ui16UnprotectCount);

	pSlab->ui16UnprotectCount--;
	
	if( !pSlab->ui16UnprotectCount)
	{
		m_pSlabManager->protectSlab( pSlab);
	}
	
	if( bUnlockMutex)
	{
		m_pSlabManager->unlockMutex();
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_CACHE_PROTECT	
void F_FixedAlloc::unprotectSlab(
	SLAB *			pSlab,
	FLMBOOL			bMutexLocked)
{
	FLMBOOL			bUnlockMutex = FALSE;
	
	if( !m_bMemProtectionEnabled)
	{
		return;
	}

	flmAssert( pSlab->pvAllocator == this);
	
	if( !bMutexLocked)
	{
		m_pSlabManager->lockMutex();
		bUnlockMutex = TRUE;
	}
	
	if( !pSlab->ui16UnprotectCount)
	{
		m_pSlabManager->unprotectSlab( pSlab);
	}
	
	pSlab->ui16UnprotectCount++;
	
	if( bUnlockMutex)
	{
		m_pSlabManager->unlockMutex();
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_CACHE_PROTECT	
void F_FixedAlloc::protectCell(
	void *			pvCell)
{
	CELLHEADER *	pCellHeader;

	m_pSlabManager->lockMutex();
	pCellHeader = (CELLHEADER *)((FLMBYTE *)pvCell - m_uiCellHeaderSize);
	protectSlab( pCellHeader->pContainingSlab, TRUE);
	m_pSlabManager->unlockMutex();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_CACHE_PROTECT	
void F_FixedAlloc::unprotectCell(
	void *			pvCell)
{
	CELLHEADER *	pCellHeader;

	m_pSlabManager->lockMutex();
	pCellHeader = (CELLHEADER *)((FLMBYTE *)pvCell - m_uiCellHeaderSize);
	unprotectSlab( pCellHeader->pContainingSlab, TRUE);
	m_pSlabManager->unlockMutex();
}
#endif

/****************************************************************************
Desc:	Private internal method to free an unused empty slab back to the OS.
****************************************************************************/
void F_FixedAlloc::freeSlab(
	SLAB *			pSlab)
{
#ifdef FLM_DEBUG
	CELLAVAILNEXT *		pAvailNext = NULL;
	FLMUINT32				ui32AvailCount = 0;
#endif

	flmAssert( pSlab);
#ifdef FLM_CACHE_PROTECT	
	flmAssert( !pSlab->ui16UnprotectCount);
#endif

	// Memory corruption detected!

	if( pSlab->ui16AllocatedCells || pSlab->pvAllocator != this)
	{
		flmAssert( 0);
		return;
	}

#ifdef FLM_DEBUG
	// Walk the avail chain as a sanity check

	pAvailNext = (CELLAVAILNEXT *)pSlab->pLocalAvailCellListHead;
	while( pAvailNext)
	{
		ui32AvailCount++;
		pAvailNext = (CELLAVAILNEXT *)pAvailNext->pNextInList;
	}

	flmAssert( pSlab->ui16AvailCellCount == ui32AvailCount);
	flmAssert( pSlab->ui16NextNeverUsedCell >= ui32AvailCount);
#endif
	
	// Unlink from all-slabs-list

	if( pSlab->pNext)
	{
#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( pSlab->pNext, TRUE);
#endif
		pSlab->pNext->pPrev = pSlab->pPrev;
#ifdef FLM_CACHE_PROTECT	
		protectSlab( pSlab->pNext, TRUE);
#endif
	}
	else
	{
		m_pLastSlab = pSlab->pPrev;
	}

	if( pSlab->pPrev)
	{
#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( pSlab->pPrev, TRUE);
#endif
		pSlab->pPrev->pNext = pSlab->pNext;
#ifdef FLM_CACHE_PROTECT	
		protectSlab( pSlab->pPrev, TRUE);
#endif
	}
	else
	{
		m_pFirstSlab = pSlab->pNext;
	}

	// Unlink from slabs-with-avail-cells list

	if( pSlab->pNextSlabWithAvailCells)
	{
#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( pSlab->pNextSlabWithAvailCells, TRUE);
#endif
		pSlab->pNextSlabWithAvailCells->pPrevSlabWithAvailCells =
			pSlab->pPrevSlabWithAvailCells;
#ifdef FLM_CACHE_PROTECT	
		protectSlab( pSlab->pNextSlabWithAvailCells, TRUE);
#endif
	}
	else
	{
		m_pLastSlabWithAvailCells = pSlab->pPrevSlabWithAvailCells;
	}

	if( pSlab->pPrevSlabWithAvailCells)
	{
#ifdef FLM_CACHE_PROTECT	
		unprotectSlab( pSlab->pPrevSlabWithAvailCells, TRUE);
#endif
		pSlab->pPrevSlabWithAvailCells->pNextSlabWithAvailCells =
			pSlab->pNextSlabWithAvailCells;
#ifdef FLM_CACHE_PROTECT	
		protectSlab( pSlab->pPrevSlabWithAvailCells, TRUE);
#endif
	}
	else
	{
		m_pFirstSlabWithAvailCells = pSlab->pNextSlabWithAvailCells;
	}

	flmAssert( m_uiSlabsWithAvailCells);
	m_uiSlabsWithAvailCells--;
	flmAssert( m_uiTotalFreeCells >= pSlab->ui16AvailCellCount);
	m_uiTotalFreeCells -= pSlab->ui16AvailCellCount;
	m_pUsageStats->ui64Slabs--;
	
#ifdef FLM_CACHE_PROTECT	
	unprotectSlab( pSlab, TRUE);
#endif
	m_pSlabManager->freeSlab( (void **)&pSlab, TRUE);
}

/****************************************************************************
Desc:	Public method to free all the memory in the system.  
****************************************************************************/
void F_FixedAlloc::freeAll( void)
{
	SLAB *		pFreeMe;

	m_pSlabManager->lockMutex();

	while( m_pFirstSlab)
	{
		pFreeMe = m_pFirstSlab;
		m_pFirstSlab = m_pFirstSlab->pNext;
		freeSlab( pFreeMe);
	}

	flmAssert( !m_uiTotalFreeCells);

	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pFirstSlabWithAvailCells = NULL;
	m_pLastSlabWithAvailCells = NULL;
	m_uiSlabsWithAvailCells = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalFreeCells = 0;
	f_memset( m_pUsageStats, 0, sizeof( XFLM_SLAB_USAGE));
	
	m_pSlabManager->unlockMutex();	
}

/****************************************************************************
Desc:		If a relocation callback function has been registered, and memory 
			can be compressed, the avail list will be compressed
****************************************************************************/ 
void F_FixedAlloc::defragmentMemory( void)
{
	SLAB *			pCurSlab;
	SLAB *			pPrevSib;
	CELLHEADER *	pCellHeader;
	FLMBOOL			bSlabFreed;
	FLMBYTE *		pucOriginal;
	FLMBYTE *		pucReloc = NULL;
	FLMUINT			uiLoop;
	SLAB **			pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	SLAB *			smallSortBuf[ SMALL_SORT_BUF_SIZE];

	m_pSlabManager->lockMutex();
	
	if( m_uiTotalFreeCells < m_uiCellsPerSlab)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiSlabsWithAvailCells;

	// Re-sort the slabs in the avail list according to
	// their memory addresses to help reduce logical fragmentation

	if( !m_bAvailListSorted && uiMaxSortEntries > 1)
	{
		if( uiMaxSortEntries <= SMALL_SORT_BUF_SIZE)
		{
			pSortBuf = smallSortBuf;
		}
		else
		{
			if( RC_BAD( f_alloc( uiMaxSortEntries * sizeof( SLAB *), &pSortBuf)))
			{
				goto Exit;
			}
		}

		pCurSlab = m_pFirstSlabWithAvailCells;

		while( pCurSlab)
		{
			flmAssert( uiSortEntries != uiMaxSortEntries);
			pSortBuf[ uiSortEntries++] = pCurSlab;
			pCurSlab = pCurSlab->pNextSlabWithAvailCells;
		}

		// Quick sort

		flmAssert( uiSortEntries);

		f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
			F_FixedAlloc::slabAddrCompareFunc,
			F_FixedAlloc::slabAddrSwapFunc);

		// Re-link the items in the list according to the new 
		// sort order

		m_pFirstSlabWithAvailCells = NULL;
		m_pLastSlabWithAvailCells = NULL;

		pCurSlab = NULL;
		pPrevSib = NULL;

		for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
		{
			pCurSlab = pSortBuf[ uiLoop];
#ifdef FLM_CACHE_PROTECT	
			unprotectSlab( pCurSlab, TRUE);
#endif
			
			pCurSlab->pNextSlabWithAvailCells = NULL;
			pCurSlab->pPrevSlabWithAvailCells = NULL;

			if( pPrevSib)
			{
				pCurSlab->pPrevSlabWithAvailCells = pPrevSib;
#ifdef FLM_CACHE_PROTECT	
				unprotectSlab( pPrevSib, TRUE);
#endif
				pPrevSib->pNextSlabWithAvailCells = pCurSlab;
#ifdef FLM_CACHE_PROTECT	
				protectSlab( pPrevSib, TRUE);
#endif
			}
			else
			{
				m_pFirstSlabWithAvailCells = pCurSlab;
			}

#ifdef FLM_CACHE_PROTECT	
			protectSlab( pCurSlab, TRUE);
#endif
			pPrevSib = pCurSlab;
		}

		m_pLastSlabWithAvailCells = pCurSlab;
		m_bAvailListSorted = TRUE;
	}

	// Process the avail list (which should be sorted unless
	// we are too low on memory)

	pCurSlab = m_pLastSlabWithAvailCells;

	while( pCurSlab)
	{
		if( m_uiTotalFreeCells < m_uiCellsPerSlab)
		{
			// No need to continue ... we aren't above the
			// free cell threshold

			goto Exit;
		}

		pPrevSib = pCurSlab->pPrevSlabWithAvailCells;

		if( pCurSlab == m_pFirstSlabWithAvailCells ||
				!pCurSlab->ui16AvailCellCount)
		{
			// We've either hit the beginning of the avail list or
			// the slab that we are now positioned on has been
			// removed from the avail list.  In either case,
			// we are done.

			break;
		}

		if( pCurSlab->ui16AvailCellCount == m_uiCellsPerSlab ||
			pCurSlab->ui16NextNeverUsedCell == pCurSlab->ui16AvailCellCount)
		{
			freeSlab( pCurSlab);
			pCurSlab = pPrevSib;
			continue;
		}

		for( uiLoop = 0; uiLoop < pCurSlab->ui16NextNeverUsedCell &&
			pCurSlab != m_pFirstSlabWithAvailCells &&
			m_uiTotalFreeCells >= m_uiCellsPerSlab; uiLoop++)
		{
			IF_Relocator *	pRelocator;

			pCellHeader = (CELLHEADER *)
				((FLMBYTE *)pCurSlab + m_uiSlabHeaderSize +
					(uiLoop * m_uiSizeOfCellAndHeader));
			if ((pRelocator = m_pRelocator) == NULL)
			{
				pRelocator = ((CELLHEADER2 *)pCellHeader)->pRelocator;
			}

			if( pCellHeader->pContainingSlab)
			{

				// If pContainingSlab is non-NULL, the cell is currently allocated

				flmAssert( pCellHeader->pContainingSlab == pCurSlab);

				pucOriginal = ((FLMBYTE *)pCellHeader + m_uiCellHeaderSize);

				if( pRelocator->canRelocate( pucOriginal))
				{
					if( (pucReloc = (FLMBYTE *)getCell( pRelocator)) == NULL)
					{
						goto Exit;
					}

#ifdef FLM_CACHE_PROTECT	
					unprotectSlab( ((CELLHEADER *)(pucReloc - 
								m_uiCellHeaderSize))->pContainingSlab, TRUE);
#endif
							
					f_memcpy( pucReloc, pucOriginal, m_uiCellSize);
					pRelocator->relocate( pucOriginal, pucReloc);

#ifdef FLM_CACHE_PROTECT	
					protectSlab( ((CELLHEADER *)(pucReloc - 
								m_uiCellHeaderSize))->pContainingSlab, TRUE);
#endif
							
					freeCell( pucOriginal, TRUE, TRUE, &bSlabFreed);
					
					if( bSlabFreed)
					{
						break;
					}
				}
			}
		}

		pCurSlab = pPrevSib;
	}

Exit:

	m_pSlabManager->unlockMutex();

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
void F_FixedAlloc::freeUnused( void)
{
	SLAB *			pSlab;

	m_pSlabManager->lockMutex();

	if( (pSlab = m_pFirstSlabWithAvailCells) != NULL &&
		!pSlab->ui16AllocatedCells)
	{
		freeSlab( pSlab);
	}

	if( (pSlab = m_pFirstSlab) != NULL &&
		!pSlab->ui16AllocatedCells)
	{
		freeSlab( pSlab);
	}

	m_pSlabManager->unlockMutex();
}

/****************************************************************************
Desc:	Debug method to do mem leak testing.  Any cells allocated via
		allocCell but not freed via freeCell() will be triggered here.
****************************************************************************/ 
#ifdef FLM_DEBUG
void F_FixedAlloc::testForLeaks( void)
{
	SLAB *			pSlabRover = m_pFirstSlab;
	CELLHEADER *	pHeader;
	FLMUINT			uiLoop;
	F_MEM_HDR		memHeader;

	// Test for leaks

	while( pSlabRover)
	{
		for( uiLoop = 0; uiLoop < pSlabRover->ui16NextNeverUsedCell; uiLoop++)
		{
			pHeader = (CELLHEADER *)
				((FLMBYTE *)pSlabRover + m_uiSlabHeaderSize +
					(uiLoop * m_uiSizeOfCellAndHeader));

			// Nonzero here means we have a leak

			if( pHeader->pContainingSlab)
			{
				// We have a leak, so let's call logMemLeak with the
				// appropriate header passed in

				f_memset( &memHeader, 0, sizeof( F_MEM_HDR));
				memHeader.uiDataSize = m_uiCellSize;
				memHeader.puiStack = pHeader->puiStack;
				logMemLeak( &memHeader);
			}
		}

		pSlabRover = pSlabRover->pNext;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/ 
F_BufferAlloc::~F_BufferAlloc()
{
	FLMUINT	uiLoop;

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
}
	
/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_BufferAlloc::setup(
	F_SlabManager *		pSlabManager,
	FLMBOOL					bMemProtect,
	XFLM_SLAB_USAGE *		pUsageStats)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiSize;
	
	flmAssert( pSlabManager);
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( (m_ppAllocators[ uiLoop] = f_new F_FixedAlloc) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
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
				rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
				goto Exit;
		}

		if (RC_BAD( rc = m_ppAllocators[ uiLoop]->setup( NULL,
			pSlabManager, bMemProtect, uiSize, pUsageStats)))
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
RCODE F_BufferAlloc::allocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiSize,
	void *				pvInitialData,
	FLMUINT				uiDataSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL *			pbAllocatedOnHeap)
{
	RCODE					rc = NE_XFLM_OK;
	F_FixedAlloc *		pAllocator = getAllocator( uiSize);

	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	if( pAllocator)
	{
		flmAssert( pAllocator->getCellSize() >= uiSize);

		if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( pRelocator, 
			pvInitialData, uiDataSize)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiSize, ppucBuffer)))
		{
			goto Exit;
		}
		
		m_pSlabManager->incrementTotalBytesAllocated( 
			f_msize( *ppucBuffer), FALSE);
		
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
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_BufferAlloc::reallocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiOldSize,
	FLMUINT				uiNewSize,
	void *				pvInitialData,
	FLMUINT				uiDataSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL *			pbAllocatedOnHeap)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE *			pucTmp;
	F_FixedAlloc *		pOldAllocator;
	F_FixedAlloc *		pNewAllocator;
	FLMBOOL				bLockedMutex = FALSE;

	flmAssert( uiNewSize);

	if( !uiOldSize)
	{
		rc = allocBuf( pRelocator, uiNewSize, pvInitialData, uiDataSize, 
			ppucBuffer, pbAllocatedOnHeap);
		goto Exit;
	}

	pOldAllocator = getAllocator( uiOldSize);
	pNewAllocator = getAllocator( uiNewSize);

	if( pOldAllocator && pOldAllocator == pNewAllocator)
	{
		// The allocation will still fit in the same cell

		goto Exit;
	}
	
	m_pSlabManager->lockMutex();
	bLockedMutex = TRUE;

	if( pbAllocatedOnHeap)
	{
		*pbAllocatedOnHeap = FALSE;
	}
	
	if( pOldAllocator)
	{
		if( pNewAllocator)
		{
			flmAssert( pOldAllocator != pNewAllocator);

			if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( pRelocator,
										NULL, 0, TRUE)) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = f_alloc( uiNewSize, &pucTmp)))
			{
				goto Exit;
			}
			
			m_pSlabManager->incrementTotalBytesAllocated( 
				f_msize( pucTmp), FALSE);
			
			if( pbAllocatedOnHeap)
			{
				*pbAllocatedOnHeap = TRUE;
			}
		}

		f_memcpy( pucTmp, *ppucBuffer, f_min( uiOldSize, uiNewSize));
		pOldAllocator->freeCell( *ppucBuffer, TRUE);
		*ppucBuffer = pucTmp;
	}
	else
	{
		if( pNewAllocator)
		{
			if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( pRelocator, 
				*ppucBuffer, f_min( uiOldSize, uiNewSize))) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}

			m_pSlabManager->decrementTotalBytesAllocated( 
					f_msize( *ppucBuffer), TRUE);			
			f_free( ppucBuffer);
			*ppucBuffer = pucTmp;
		}
		else
		{
			FLMUINT		uiOldAllocSize = f_msize( *ppucBuffer);
	
			flmAssert( uiOldSize > m_ppAllocators[ NUM_BUF_ALLOCATORS - 1]->getCellSize());
			flmAssert( uiNewSize > m_ppAllocators[ NUM_BUF_ALLOCATORS - 1]->getCellSize());
			
			if( RC_BAD( rc = f_realloc( uiNewSize, ppucBuffer)))
			{
				goto Exit;
			}
			
			m_pSlabManager->decrementTotalBytesAllocated( 
				uiOldAllocSize, TRUE);
			m_pSlabManager->incrementTotalBytesAllocated( 
				f_msize( *ppucBuffer), TRUE);
			
			if( pbAllocatedOnHeap)
			{
				*pbAllocatedOnHeap = TRUE;
			}
		}
	}

Exit:

	if( bLockedMutex)
	{
		m_pSlabManager->unlockMutex();
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
	
	if( pAllocator)
	{
		pAllocator->freeCell( *ppucBuffer, FALSE, TRUE, NULL);
		*ppucBuffer = NULL;
	}
	else
	{
		m_pSlabManager->decrementTotalBytesAllocated( 
			f_msize( *ppucBuffer), FALSE);			
		f_free( ppucBuffer);
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_BufferAlloc::defragmentMemory( void)
{
	FLMUINT	uiLoop;

	for( uiLoop = 0; uiLoop < NUM_BUF_ALLOCATORS; uiLoop++)
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->defragmentMemory();
			m_ppAllocators[ uiLoop]->freeUnused();
		}

		uiLoop++;
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
/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_MultiAlloc::setup(
	F_SlabManager *		pSlabManager,
	FLMBOOL					bMemProtect,
	FLMUINT *				puiCellSizes,
	XFLM_SLAB_USAGE *		pUsageStats)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;
	FLMUINT		uiCellCount;

	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	uiCellCount = 0;
	while( puiCellSizes[ uiCellCount])
	{
		uiCellCount++;
	}

	if( !uiCellCount)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_INVALID_PARM);
		goto Exit;
	}

	f_qsort( puiCellSizes, 0, uiCellCount - 1, 
		flmQSortUINTCompare, flmQSortUINTSwap);

	if( RC_BAD( rc = f_alloc( 
		sizeof( FLMUINT *) * (uiCellCount + 1), &m_puiCellSizes)))
	{
		goto Exit;
	}
	
	m_pSlabManager->incrementTotalBytesAllocated( 
		f_msize( m_puiCellSizes), FALSE);
	
	f_memcpy( m_puiCellSizes, puiCellSizes, 
		(uiCellCount + 1) * sizeof( FLMUINT));

	// Set up the allocators

	if( RC_BAD( rc = f_calloc( 
		sizeof( F_FixedAlloc *) * (uiCellCount + 1), &m_ppAllocators)))
	{
		goto Exit;
	}
	
	m_pSlabManager->incrementTotalBytesAllocated( 
		f_msize( m_ppAllocators), FALSE);

	uiLoop = 0;
	while( m_puiCellSizes[ uiLoop])
	{
		if( (m_ppAllocators[ uiLoop] = f_new F_FixedAlloc) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_ppAllocators[ uiLoop]->setup( NULL,
			pSlabManager, bMemProtect, m_puiCellSizes[ uiLoop], pUsageStats)))
		{
			goto Exit;
		}

		uiLoop++;
	}

Exit:

	if( RC_BAD( rc))
	{
		cleanup();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_MultiAlloc::cleanup( void)
{
	FLMUINT		uiLoop = 0;

	if( !m_puiCellSizes || !m_ppAllocators)
	{
		goto Exit;
	}

	while( m_puiCellSizes[ uiLoop])
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->Release();
			m_ppAllocators[ uiLoop] = NULL;
		}

		uiLoop++;
	}

Exit:

	if( m_puiCellSizes)
	{
		m_pSlabManager->decrementTotalBytesAllocated( 
			f_msize( m_puiCellSizes), FALSE);
		f_free( &m_puiCellSizes);
	}
	
	if( m_ppAllocators)
	{
		m_pSlabManager->decrementTotalBytesAllocated( 
			f_msize( m_ppAllocators), FALSE);
		f_free( &m_ppAllocators);
	}
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
		m_pSlabManager = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_MultiAlloc::allocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL				bMutexLocked)
{
	RCODE					rc = NE_XFLM_OK;
	F_FixedAlloc *		pAllocator = getAllocator( uiSize);

	flmAssert( pAllocator);
	flmAssert( pAllocator->getCellSize() >= uiSize);

	if( (*ppucBuffer = (FLMBYTE *)pAllocator->allocCell( pRelocator, 
		NULL, 0, bMutexLocked)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
RCODE F_MultiAlloc::reallocBuf(
	IF_Relocator *		pRelocator,
	FLMUINT				uiNewSize,
	FLMBYTE **			ppucBuffer,
	FLMBOOL				bMutexLocked)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE *			pucTmp;
	F_FixedAlloc *		pOldAllocator;
	F_FixedAlloc *		pNewAllocator;
	FLMBOOL				bLockedMutex = FALSE;

	flmAssert( uiNewSize);

	if( !(*ppucBuffer))
	{
		rc = allocBuf( pRelocator, uiNewSize, ppucBuffer);
		goto Exit;
	}

	pOldAllocator = getAllocator( *ppucBuffer);
	pNewAllocator = getAllocator( uiNewSize);

	if( pOldAllocator == pNewAllocator)
	{
		// The allocation will still fit in the same cell

		goto Exit;
	}
	
	if( !bMutexLocked)
	{
		m_pSlabManager->lockMutex();
		bLockedMutex = TRUE;
	}

	if( (pucTmp = (FLMBYTE *)pNewAllocator->allocCell( pRelocator, *ppucBuffer, 
		f_min( uiNewSize, pOldAllocator->m_uiCellSize), TRUE)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pOldAllocator->freeCell( *ppucBuffer, TRUE);
	*ppucBuffer = pucTmp;
	
Exit:

	if( bLockedMutex)
	{
		m_pSlabManager->unlockMutex();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/ 
void F_MultiAlloc::defragmentMemory( void)
{
	FLMUINT		uiLoop = 0;

	while( m_puiCellSizes[ uiLoop])
	{
		if( m_ppAllocators[ uiLoop])
		{
			m_ppAllocators[ uiLoop]->defragmentMemory();
			m_ppAllocators[ uiLoop]->freeUnused();
		}

		uiLoop++;
	}
}

/****************************************************************************
Desc:
****************************************************************************/ 
F_FixedAlloc * F_MultiAlloc::getAllocator(
	FLMUINT				uiSize)
{
	F_FixedAlloc *		pAllocator = NULL;
	FLMUINT				uiLoop;

	flmAssert( uiSize);

	for( uiLoop = 0; m_puiCellSizes[ uiLoop]; uiLoop++)
	{
		if( m_puiCellSizes[ uiLoop] >= uiSize)
		{
			pAllocator = m_ppAllocators[ uiLoop];
			break;
		}
	}

	return( pAllocator);
}

/****************************************************************************
Desc:
****************************************************************************/ 
F_FixedAlloc * F_MultiAlloc::getAllocator(
	FLMBYTE *			pucBuffer)
{
	F_FixedAlloc::CELLHEADER *	pHeader;
	F_FixedAlloc::SLAB *			pSlab;
	F_FixedAlloc *					pAllocator = NULL;

	m_pSlabManager->lockMutex();
	
	pHeader = (F_FixedAlloc::CELLHEADER *)(pucBuffer - 
			F_FixedAlloc::getAllocAlignedSize( 
			sizeof( F_FixedAlloc::CELLHEADER2)));
	pSlab = pHeader->pContainingSlab;
	pAllocator = (F_FixedAlloc *)pSlab->pvAllocator;

	m_pSlabManager->unlockMutex();
	return( pAllocator);
}

/****************************************************************************
Desc:
****************************************************************************/ 
#ifdef FLM_CACHE_PROTECT	
void F_MultiAlloc::protectBuffer(
	void *			pvBuffer,
	FLMBOOL			bMutexLocked)
{
	F_FixedAlloc::CELLHEADER *	pHeader;
	F_FixedAlloc::SLAB *			pSlab;
	F_FixedAlloc *					pAllocator = NULL;
	FLMBYTE *						pucBuffer = (FLMBYTE *)pvBuffer;

	if( !bMutexLocked)
	{
		m_pSlabManager->lockMutex();
	}
	
	pHeader = (F_FixedAlloc::CELLHEADER *)(pucBuffer - 
			F_FixedAlloc::getAllocAlignedSize( 
			sizeof( F_FixedAlloc::CELLHEADER2)));
	pSlab = pHeader->pContainingSlab;
	pAllocator = (F_FixedAlloc *)pSlab->pvAllocator;
	pAllocator->protectSlab( pSlab, TRUE);

	if( !bMutexLocked)
	{
		m_pSlabManager->unlockMutex();
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/ 
#ifdef FLM_CACHE_PROTECT	
void F_MultiAlloc::unprotectBuffer(
	void *			pvBuffer,
	FLMBOOL			bMutexLocked)
{
	F_FixedAlloc::CELLHEADER *	pHeader;
	F_FixedAlloc::SLAB *			pSlab;
	F_FixedAlloc *					pAllocator = NULL;
	FLMBYTE *						pucBuffer = (FLMBYTE *)pvBuffer;

	if( !bMutexLocked)
	{
		m_pSlabManager->lockMutex();
	}
	
	pHeader = (F_FixedAlloc::CELLHEADER *)(pucBuffer - 
		F_FixedAlloc::getAllocAlignedSize( 
		sizeof( F_FixedAlloc::CELLHEADER2)));
	pSlab = pHeader->pContainingSlab;
	pAllocator = (F_FixedAlloc *)pSlab->pvAllocator;
	pAllocator->unprotectSlab( pSlab, TRUE);

	if( !bMutexLocked)
	{
		m_pSlabManager->unlockMutex();
	}
}
#endif
