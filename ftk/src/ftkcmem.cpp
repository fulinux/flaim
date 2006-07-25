//------------------------------------------------------------------------------
// Desc: Block cache allocator
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id: $
//------------------------------------------------------------------------------

#include "ftksys.h"

/****************************************************************************
Desc:
****************************************************************************/
typedef struct SLABINFO
{
	void *			pvSlab;
	SLABINFO *		pPrevInList;
	SLABINFO *		pNextInList;
	SLABINFO *		pPrevSlabWithAvail;
	SLABINFO *		pNextSlabWithAvail;
	FLMUINT8			ui8NextNeverUsed;
	FLMUINT8			ui8AvailBlocks;
	FLMUINT8			ui8FirstAvail;
	FLMUINT8			ui8AllocatedBlocks;
#define F_ALLOC_MAP_BYTES	4
#define F_ALLOC_MAP_BITS	(F_ALLOC_MAP_BYTES * 8)
	FLMBYTE			ucAllocMap[ F_ALLOC_MAP_BYTES];
} SLABINFO;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct AVAILBLOCK
{
	FLMUINT8			ui8NextAvail;
} AVAILBLOCK;

/****************************************************************************
Desc:
****************************************************************************/
class F_CacheAlloc : public F_Object
{
public:

	F_CacheAlloc();

	virtual ~F_CacheAlloc();

	RCODE FLMAPI setup(
		IF_SlabManager *		pSlabManager,
		IF_Relocator *			pRelocator,
		FLMUINT					uiBlockSize,
		FLM_SLAB_USAGE *		pUsageStats,
		FLMUINT *				puiTotalBytesAllocated);

	RCODE FLMAPI allocBlock(
		void **					ppvBlock);

	void FLMAPI freeBlock(
		void **					ppvBlock);

	void FLMAPI freeUnused( void);

	void FLMAPI freeAll( void);

	void FLMAPI defragmentMemory( void);
	
private:

	void cleanup( void);

	RCODE getCell(
		SLABINFO **				ppSlab,
		void **					ppvCell);

	RCODE getAnotherSlab(
		SLABINFO **				ppSlab);
		
	void freeSlab(
		SLABINFO **				ppSlab);

	void freeCell(
		SLABINFO **				ppSlab,
		void **					ppvCell);

	IF_SlabManager *			m_pSlabManager;
	IF_Relocator *				m_pRelocator;
	IF_FixedAlloc *			m_pInfoAllocator;
	SLABINFO *					m_pFirstSlab;
	SLABINFO *					m_pLastSlab;
	SLABINFO *					m_pFirstSlabWithAvail;
	SLABINFO *					m_pLastSlabWithAvail;
	FLMBOOL						m_bAvailListSorted;
	FLMUINT						m_uiSlabSize;
	FLMUINT						m_uiBlockSize;
	FLMUINT						m_uiBlocksPerSlab;
	FLMUINT						m_uiSlabsWithAvail;
	FLMUINT						m_uiTotalAvailBlocks;
	FLM_SLAB_USAGE *			m_pUsageStats;
	FLMUINT *					m_puiTotalBytesAllocated;
	F_MUTEX						m_hMutex;
	
friend class F_SlabInfoRelocator;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_SlabInfoRelocator : public IF_Relocator
{
public:

	F_SlabInfoRelocator()
	{
		m_pCacheAlloc = NULL;
	}
	
	virtual ~F_SlabInfoRelocator()
	{
	}

	void FLMAPI relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc);

	FLMBOOL FLMAPI canRelocate(
		void *	pvOldAlloc);
		
private:

	F_CacheAlloc *		m_pCacheAlloc;
		
friend class F_CacheAlloc;
};

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI slabInfoAddrCompareFunc(
	void *					pvBuffer,
	FLMUINT					uiPos1,
	FLMUINT					uiPos2)
{
	SLABINFO *		pSlab1 = (((SLABINFO **)pvBuffer)[ uiPos1]);
	SLABINFO *		pSlab2 = (((SLABINFO **)pvBuffer)[ uiPos2]);

	f_assert( pSlab1 != pSlab2);

	if( pSlab1->pvSlab < pSlab2->pvSlab)
	{
		return( -1);
	}

	return( 1);
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI slabInfoAddrSwapFunc(
	void *					pvBuffer,
	FLMUINT					uiPos1,
	FLMUINT					uiPos2)
{
	SLABINFO **		ppSlab1 = &(((SLABINFO **)pvBuffer)[ uiPos1]);
	SLABINFO **		ppSlab2 = &(((SLABINFO **)pvBuffer)[ uiPos2]);
	SLABINFO *		pTmp;

	pTmp = *ppSlab1;
	*ppSlab1 = *ppSlab2;
	*ppSlab2 = pTmp;
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_CacheAlloc::F_CacheAlloc()
{
	m_pSlabManager = NULL;
	m_pRelocator = NULL;
	m_pInfoAllocator = NULL;
	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pFirstSlabWithAvail = NULL;
	m_pLastSlabWithAvail = NULL;
	m_bAvailListSorted = FALSE;
	m_uiSlabSize = 0;
	m_uiBlockSize = 0;
	m_uiBlocksPerSlab = 0;
	m_uiSlabsWithAvail = 0;
	m_uiTotalAvailBlocks = 0;
	m_pUsageStats = NULL;
	m_puiTotalBytesAllocated = NULL;
	m_hMutex = F_MUTEX_NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_CacheAlloc::~F_CacheAlloc()
{
	cleanup();
}

/****************************************************************************
Desc:
****************************************************************************/
void F_CacheAlloc::cleanup( void)
{
	freeAll();
	
	if( m_pInfoAllocator)
	{
		m_pInfoAllocator->Release();
		m_pInfoAllocator = NULL;
	}
	
	if( m_pSlabManager)
	{
		m_pSlabManager->Release();
		m_pSlabManager = NULL;
	}
	
	if( m_pRelocator)
	{
		m_pRelocator->Release();
		m_pRelocator = NULL;
	}
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CacheAlloc::setup(
	IF_SlabManager *			pSlabManager,
	IF_Relocator *				pRelocator,
	FLMUINT						uiBlockSize,
	FLM_SLAB_USAGE *			pUsageStats,
	FLMUINT *					puiTotalBytesAllocated)
{
	RCODE							rc = NE_FLM_OK;
	F_SlabInfoRelocator *	pSlabInfoRelocator = NULL;

	f_assert( pSlabManager);
	f_assert( pRelocator);
	f_assert( uiBlockSize);
	f_assert( pUsageStats);
	
	if( uiBlockSize != 4096 && uiBlockSize != 8192)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_PARM);
		goto Exit;
	}
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	m_pUsageStats = pUsageStats;
	m_puiTotalBytesAllocated = puiTotalBytesAllocated;
	
	m_pSlabManager = pSlabManager;
	m_pSlabManager->AddRef();
	
	m_pRelocator = pRelocator;
	m_pRelocator->AddRef();
	
	m_uiBlockSize = uiBlockSize;
	m_uiSlabSize = m_pSlabManager->getSlabSize();
	
	m_uiBlocksPerSlab = m_uiSlabSize / m_uiBlockSize;
	f_assert( F_ALLOC_MAP_BITS >= m_uiBlocksPerSlab); 
	
	if( RC_BAD( rc = FlmAllocFixedAllocator( &m_pInfoAllocator)))
	{
		goto Exit;
	}
	
	if( (pSlabInfoRelocator = f_new F_SlabInfoRelocator) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	pSlabInfoRelocator->m_pCacheAlloc = this;
	
	if( RC_BAD( rc = m_pInfoAllocator->setup( FALSE, m_pSlabManager, 
		pSlabInfoRelocator, sizeof( SLABINFO), 
		m_pUsageStats, puiTotalBytesAllocated)))
	{
		goto Exit;
	}

Exit:

	if( pSlabInfoRelocator)
	{
		pSlabInfoRelocator->Release();
	}
	
	if( RC_BAD( rc))
	{
		cleanup();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_CacheAlloc::allocBlock(
	void **				ppvBlock)
{
	RCODE					rc = NE_FLM_OK;
	FLMBOOL				bMutexLocked = FALSE;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( RC_BAD( rc = getCell( NULL, ppvBlock)))
	{
		goto Exit;
	}
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_CacheAlloc::freeBlock(
	void **				ppvBlock)
{
	SLABINFO *			pSlab = NULL;
//	FLMBYTE *			pucBlock = (FLMBYTE *)*ppvBlock;
	FLMBOOL				bMutexLocked = FALSE;
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	f_assert( 0); // VISIT: Need to locate the slab!!!!
	
	freeCell( &pSlab, ppvBlock);
	
	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CacheAlloc::getCell(
	SLABINFO **		ppSlab,
	void **			ppvCell)
{
	RCODE				rc = NE_FLM_OK;
	AVAILBLOCK *	pAvailBlock;
	SLABINFO *		pSlab = NULL;
	FLMBYTE *		pCell = NULL;

#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	// If there is a slab that has an avail cell, that one gets priority

	if( (pSlab = m_pFirstSlabWithAvail) != NULL)
	{
		f_assert( pSlab->ui8AvailBlocks <= m_uiTotalAvailBlocks);
		f_assert( m_uiTotalAvailBlocks);
		f_assert( pSlab->ui8AllocatedBlocks < m_uiBlocksPerSlab);
		f_assert( !f_isBitSet( pSlab->ucAllocMap, pSlab->ui8FirstAvail));

		pAvailBlock = (AVAILBLOCK *)(((FLMBYTE *)pSlab->pvSlab) + 
									(pSlab->ui8FirstAvail * m_uiBlockSize));

		pSlab->ui8AllocatedBlocks++;
		pSlab->ui8AvailBlocks--;
		m_uiTotalAvailBlocks--;
		
		f_setBit( pSlab->ucAllocMap, pSlab->ui8FirstAvail);
		
		// A free block holds as its contents the next pointer in the free chain.
		// Free chains do not span slabs.

		pSlab->ui8FirstAvail = pAvailBlock->ui8NextAvail;

		// If there are no other free blocks in this slab, we need to unlink 
		// the slab from the slabs-with-avail-blocks list
		
		if( !pSlab->ui8AvailBlocks)
		{
			// Save a copy of the slab we're going to unlink

			SLABINFO * 		pSlabToUnlink = pSlab;

			f_assert( !pSlabToUnlink->ui8AvailBlocks);
			f_assert( !pSlabToUnlink->pPrevSlabWithAvail);				

			// Update m_pFirstSlabWithAvail to point to the next one

			if( (m_pFirstSlabWithAvail =
				pSlabToUnlink->pNextSlabWithAvail) == NULL)
			{
				f_assert( m_pLastSlabWithAvail == pSlabToUnlink);
				m_pLastSlabWithAvail = NULL;
			}

			// Unlink from slabs-with-avail-cells list

			if( pSlabToUnlink->pNextSlabWithAvail)
			{
				pSlabToUnlink->pNextSlabWithAvail->pPrevSlabWithAvail =
					pSlabToUnlink->pPrevSlabWithAvail;
				pSlabToUnlink->pNextSlabWithAvail = NULL;
			}

			// Decrement the slab count

			f_assert( m_uiSlabsWithAvail);
			m_uiSlabsWithAvail--;
		}
		
		pCell = (FLMBYTE *)pAvailBlock;
	}
	else
	{
		if( !m_pFirstSlab ||
			 (m_pFirstSlab->ui8NextNeverUsed == m_uiBlocksPerSlab))
		{
			SLABINFO *		pNewSlab;
			
			if( RC_BAD( rc = getAnotherSlab( &pNewSlab)))
			{
				goto Exit;
			}
			
			if( m_pFirstSlab)
			{
				pNewSlab->pNextInList = m_pFirstSlab;
				m_pFirstSlab->pPrevInList = pNewSlab;
			}
			else
			{
				m_pLastSlab = pNewSlab;
			}

			m_pFirstSlab = pNewSlab;
		}

		pSlab = m_pFirstSlab;
		pSlab->ui8AllocatedBlocks++;
		
		pCell = (((FLMBYTE *)pSlab->pvSlab) + 
									(pSlab->ui8NextNeverUsed * m_uiBlockSize));

		f_setBit( pSlab->ucAllocMap, pSlab->ui8NextNeverUsed);									
		pSlab->ui8NextNeverUsed++;
	}

	if( m_pUsageStats)
	{
		m_pUsageStats->ui64AllocatedCells++;
	}
	
	if( ppSlab)
	{
		*ppSlab = pSlab;
	}
	
	*ppvCell = pCell;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_CacheAlloc::freeCell(
	SLABINFO **		ppSlab,
	void **			ppvCell)
{
	SLABINFO *		pSlab = *ppSlab;
	FLMBYTE *		pCell = (FLMBYTE *)*ppvCell;
	AVAILBLOCK *	pAvailBlock = (AVAILBLOCK *)pCell;
	FLMUINT			uiBlockNum;
	
#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif

	// Make sure the cell and slab look sane

	if( !pSlab || !pCell || pCell < pSlab->pvSlab || 
		(pCell + m_uiBlockSize) > ((FLMBYTE *)pSlab->pvSlab + m_uiSlabSize))
	{
		f_assert( 0);
		goto Exit;
	}
	
	// Verify that the cell address is on a block boundary
	
	f_assert( ((pCell - (FLMBYTE *)pSlab->pvSlab) & m_uiBlockSize) == 0);
	
	// Determine the block number

	uiBlockNum = (pCell - (FLMBYTE *)pSlab->pvSlab) / m_uiBlockSize;
	
	// Make sure the block is valie
	
	f_assert( uiBlockNum < m_uiBlocksPerSlab);
	f_assert( f_isBitSet( pSlab->ucAllocMap, uiBlockNum));
	
	// Clear the "allocated" bit
	
	f_clearBit( pSlab->ucAllocMap, uiBlockNum);
	
	// Should always be non-null on a free
	
	f_assert( m_pFirstSlab);
	
	// Add the cell to the slab's free list

	pAvailBlock->ui8NextAvail = pSlab->ui8FirstAvail;
	pSlab->ui8FirstAvail = uiBlockNum;
	pSlab->ui8AvailBlocks++;

	f_assert( pSlab->ui8AllocatedBlocks);
	pSlab->ui8AllocatedBlocks--;

	// If there's no chain, make this one the first

	if( !m_pFirstSlabWithAvail)
	{
		m_pFirstSlabWithAvail = pSlab;
		m_pLastSlabWithAvail = pSlab;
		f_assert( !pSlab->pNextSlabWithAvail);
		f_assert( !pSlab->pPrevSlabWithAvail);
		m_uiSlabsWithAvail++;
		m_bAvailListSorted = TRUE;
	}
	else if( pSlab->ui8AvailBlocks == 1)
	{
		// This item is not linked in to the chain, so link it in

		if( m_bAvailListSorted && 
			 pSlab->pvSlab > m_pFirstSlabWithAvail->pvSlab)
		{
			m_bAvailListSorted = FALSE;
		}

		pSlab->pNextSlabWithAvail = m_pFirstSlabWithAvail;
		pSlab->pPrevSlabWithAvail = NULL;
		m_pFirstSlabWithAvail->pPrevSlabWithAvail = pSlab;
		m_pFirstSlabWithAvail = pSlab;
		m_uiSlabsWithAvail++;
	}

	// Adjust counter, because the block is now available

	m_uiTotalAvailBlocks++;

	// If this slab is now totally avail

	if( pSlab->ui8AvailBlocks == m_uiBlocksPerSlab)
	{
		f_assert( !pSlab->ui8AllocatedBlocks);

		// If we have met our threshold for being able to free a slab

		if( m_uiTotalAvailBlocks >= m_uiBlocksPerSlab)
		{
			freeSlab( &pSlab);
		}
		else if( pSlab != m_pFirstSlabWithAvail)
		{
			// Link the slab to the front of the avail list so that
			// it can be freed quickly at some point in the future

			if( pSlab->pPrevSlabWithAvail)
			{
				pSlab->pPrevSlabWithAvail->pNextSlabWithAvail =
					pSlab->pNextSlabWithAvail;
			}

			if( pSlab->pNextSlabWithAvail)
			{
				pSlab->pNextSlabWithAvail->pPrevSlabWithAvail =
					pSlab->pPrevSlabWithAvail;
			}
			else
			{
				f_assert( m_pLastSlabWithAvail == pSlab);
				m_pLastSlabWithAvail = pSlab->pPrevSlabWithAvail;
			}

			if( m_pFirstSlabWithAvail)
			{
				m_pFirstSlabWithAvail->pPrevSlabWithAvail = pSlab;
			}

			pSlab->pPrevSlabWithAvail = NULL;
			pSlab->pNextSlabWithAvail = m_pFirstSlabWithAvail;
			m_pFirstSlabWithAvail = pSlab;
		}
	}
	
	if( m_pUsageStats)
	{
		m_pUsageStats->ui64AllocatedCells--;
	}

	*ppSlab = pSlab;
	*ppvCell = NULL;
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_CacheAlloc::getAnotherSlab(
	SLABINFO **		ppSlab)
{
	RCODE				rc = NE_FLM_OK;
	SLABINFO *		pSlab = NULL;
	
#ifdef FLM_DEBUG
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}
#endif
			
	if( (pSlab = (SLABINFO *)m_pInfoAllocator->allocCell( 
		NULL, NULL)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pSlabManager->allocSlab( &pSlab->pvSlab)))
	{
		m_pInfoAllocator->freeCell( &pSlab);
		goto Exit;
	}

	f_memset( pSlab, 0, sizeof( SLABINFO));
	
	if( m_pUsageStats)
	{
		m_pUsageStats->ui64Slabs++;
	}
	
	if( m_puiTotalBytesAllocated)
	{
		(*m_puiTotalBytesAllocated) += m_uiSlabSize;
	}
	
	*ppSlab = pSlab;

Exit:
	
	return( rc);
}

/****************************************************************************
Desc:	Private internal method to free an unused empty slab back to the OS.
****************************************************************************/
void F_CacheAlloc::freeSlab(
	SLABINFO **			ppSlab)
{
	SLABINFO *			pSlab = *ppSlab;
	FLMUINT				uiLoop;

	f_assert( pSlab);
	
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexLocked( m_hMutex);
	}

	if( pSlab->ui8AllocatedBlocks)
	{
		// Memory corruption detected!

		f_assert( 0);
		return;
	}
	
	// Make sure all "allocated" bits have been cleared
	
	for( uiLoop = 0; uiLoop < F_ALLOC_MAP_BYTES; uiLoop++)
	{
		if( pSlab->ucAllocMap[ uiLoop])
		{
			f_assert( 0);
			return;
		}
	}

	// Unlink from all-slabs-list

	if( pSlab->pNextInList)
	{
		pSlab->pNextInList->pPrevInList = pSlab->pPrevInList;
	}
	else
	{
		m_pLastSlab = pSlab->pPrevInList;
	}

	if( pSlab->pPrevInList)
	{
		pSlab->pPrevInList->pNextInList = pSlab->pNextInList;
	}
	else
	{
		m_pFirstSlab = pSlab->pNextInList;
	}

	// Unlink from slabs-with-avail-cells list

	if( pSlab->pNextSlabWithAvail)
	{
		pSlab->pNextSlabWithAvail->pPrevSlabWithAvail =
			pSlab->pPrevSlabWithAvail;
	}
	else
	{
		m_pLastSlabWithAvail = pSlab->pPrevSlabWithAvail;
	}

	if( pSlab->pPrevSlabWithAvail)
	{
		pSlab->pPrevSlabWithAvail->pNextSlabWithAvail =
			pSlab->pNextSlabWithAvail;
	}
	else
	{
		m_pFirstSlabWithAvail = pSlab->pNextSlabWithAvail;
	}

	f_assert( m_uiSlabsWithAvail);
	m_uiSlabsWithAvail--;
	
	f_assert( m_uiTotalAvailBlocks >= pSlab->ui8AvailBlocks);
	m_uiTotalAvailBlocks -= pSlab->ui8AvailBlocks;
	m_pSlabManager->freeSlab( (void **)&pSlab->pvSlab);
	m_pInfoAllocator->freeCell( &pSlab);
	
	if( m_pUsageStats)
	{
		f_assert( m_pUsageStats->ui64Slabs);
		m_pUsageStats->ui64Slabs--;
	}
	
	if( m_puiTotalBytesAllocated)
	{
		f_assert( (*m_puiTotalBytesAllocated) >= m_uiSlabSize);
		(*m_puiTotalBytesAllocated) -= m_uiSlabSize;
	}
	
	*ppSlab = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_CacheAlloc::freeAll( void)
{
	SLABINFO *		pFreeMe;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}

	while( m_pFirstSlab)
	{
		pFreeMe = m_pFirstSlab;
		m_pFirstSlab = m_pFirstSlab->pNextInList;
		freeSlab( &pFreeMe);
	}

	f_assert( !m_uiTotalAvailBlocks);

	m_pFirstSlab = NULL;
	m_pLastSlab = NULL;
	m_pFirstSlabWithAvail = NULL;
	m_pLastSlabWithAvail = NULL;
	m_uiSlabsWithAvail = 0;
	m_bAvailListSorted = TRUE;
	m_uiTotalAvailBlocks = 0;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/ 
void FLMAPI F_CacheAlloc::freeUnused( void)
{
	SLABINFO *		pSlab;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
	}

	if( (pSlab = m_pFirstSlabWithAvail) != NULL &&
		!pSlab->ui8AllocatedBlocks)
	{
		freeSlab( &pSlab);
	}

	if( (pSlab = m_pFirstSlab) != NULL && !pSlab->ui8AllocatedBlocks)
	{
		freeSlab( &pSlab);
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_CacheAlloc::defragmentMemory( void)
{
	RCODE				rc = NE_FLM_OK;
	SLABINFO *		pCurSlab;
	SLABINFO *		pPrevSib;
	FLMUINT			uiLoop;
	SLABINFO **		pSortBuf = NULL;
	FLMUINT			uiMaxSortEntries;
	FLMUINT			uiSortEntries = 0;
#define SMALL_SORT_BUF_SIZE 256
	SLABINFO *		smallSortBuf[ SMALL_SORT_BUF_SIZE];
	FLMBOOL			bMutexLocked = FALSE;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_assertMutexNotLocked( m_hMutex);
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}
	
	if( m_uiTotalAvailBlocks < m_uiBlocksPerSlab)
	{
		goto Exit;
	}

	uiMaxSortEntries = m_uiSlabsWithAvail;

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
			if( RC_BAD( rc = f_alloc( uiMaxSortEntries * sizeof( SLABINFO *),
				&pSortBuf)))
			{
				goto Exit;
			}
		}

		pCurSlab = m_pFirstSlabWithAvail;

		while( pCurSlab)
		{
			f_assert( uiSortEntries != uiMaxSortEntries);
			
			pSortBuf[ uiSortEntries++] = pCurSlab;
			pCurSlab = pCurSlab->pNextSlabWithAvail;
		}

		// Quick sort

		f_assert( uiSortEntries);

		f_qsort( (FLMBYTE *)pSortBuf, 0, uiSortEntries - 1, 
			slabInfoAddrCompareFunc, slabInfoAddrSwapFunc);

		// Re-link the items in the list according to the new 
		// sort order

		m_pFirstSlabWithAvail = NULL;
		m_pLastSlabWithAvail = NULL;

		pCurSlab = NULL;
		pPrevSib = NULL;

		for( uiLoop = 0; uiLoop < uiSortEntries; uiLoop++)
		{
			pCurSlab = pSortBuf[ uiLoop];
			
			pCurSlab->pNextSlabWithAvail = NULL;
			pCurSlab->pPrevSlabWithAvail = NULL;

			if( pPrevSib)
			{
				pCurSlab->pPrevSlabWithAvail = pPrevSib;
				pPrevSib->pNextSlabWithAvail = pCurSlab;
			}
			else
			{
				m_pFirstSlabWithAvail = pCurSlab;
			}

			pPrevSib = pCurSlab;
		}

		m_pLastSlabWithAvail = pCurSlab;
		m_bAvailListSorted = TRUE;
	}

	// Process the avail list (which should be sorted unless
	// we are too low on memory)

	pCurSlab = m_pLastSlabWithAvail;

	while( pCurSlab)
	{
		if( m_uiTotalAvailBlocks < m_uiBlocksPerSlab)
		{
			// No need to continue ... we aren't above the
			// free cell threshold

			goto Exit;
		}

		pPrevSib = pCurSlab->pPrevSlabWithAvail;

		if( pCurSlab == m_pFirstSlabWithAvail || !pCurSlab->ui8AvailBlocks)
		{
			// We've either hit the beginning of the avail list or
			// the slab that we are now positioned on has been
			// removed from the avail list.  In either case,
			// we are done.

			break;
		}

		if( pCurSlab->ui8AvailBlocks == m_uiBlocksPerSlab ||
			pCurSlab->ui8NextNeverUsed == pCurSlab->ui8AvailBlocks)
		{
			freeSlab( &pCurSlab);
			pCurSlab = pPrevSib;
			continue;
		}

		for( uiLoop = 0; uiLoop < pCurSlab->ui8NextNeverUsed &&
			pCurSlab != m_pFirstSlabWithAvail &&
			m_uiTotalAvailBlocks >= m_uiBlocksPerSlab; uiLoop++)
		{
			FLMBYTE *	pucBlock;
			
			pucBlock = (FLMBYTE *)(pCurSlab->pvSlab) + (uiLoop * m_uiBlockSize);
			
			if( f_isBitSet( pCurSlab->ucAllocMap, uiLoop))
			{
				FLMBYTE *	pucReloc = NULL;
				SLABINFO *	pRelocSlab;
				
				if( m_pRelocator->canRelocate( pucBlock))
				{
					if( RC_BAD( rc = getCell( &pRelocSlab, (void **)&pucReloc)))
					{
						goto Exit;
					}
					
					f_memcpy( pucReloc, pucBlock, m_uiBlockSize);
					m_pRelocator->relocate( pucBlock, pucReloc);

					freeCell( &pCurSlab, (void **)&pucBlock);
					
					if( !pCurSlab)
					{
						break;
					}
				}
			}
		}

		pCurSlab = pPrevSib;
	}
	
	// Defragment the slab info list
	
	m_pInfoAllocator->defragmentMemory();
	
Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	if( pSortBuf && pSortBuf != smallSortBuf)
	{
		f_free( &pSortBuf);
	}
}

/****************************************************************************
Desc:		
****************************************************************************/
FLMBOOL F_SlabInfoRelocator::canRelocate(
	void *		pvAlloc)
{
	F_UNREFERENCED_PARM( pvAlloc);
	return( TRUE);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_SlabInfoRelocator::relocate(
	void *			pvOldAlloc,
	void *			pvNewAlloc)
{
	SLABINFO *		pOldSlabInfo = (SLABINFO *)pvOldAlloc;
	SLABINFO *		pNewSlabInfo = (SLABINFO *)pvNewAlloc;
	F_CacheAlloc *	pCacheAlloc = m_pCacheAlloc;
	
	if( pOldSlabInfo->pPrevInList)
	{
		f_assert( pOldSlabInfo != pCacheAlloc->m_pFirstSlab); 
		pOldSlabInfo->pPrevInList->pNextInList = pNewSlabInfo;
	}
	else
	{
		f_assert( pOldSlabInfo == pCacheAlloc->m_pFirstSlab); 
		pCacheAlloc->m_pFirstSlab = pNewSlabInfo;
	}
	
	if( pOldSlabInfo->pNextInList)
	{
		f_assert( pOldSlabInfo != pCacheAlloc->m_pLastSlab);
		pOldSlabInfo->pNextInList->pPrevInList = pNewSlabInfo;
	}
	else
	{
		f_assert( pOldSlabInfo == pCacheAlloc->m_pLastSlab);
		pCacheAlloc->m_pLastSlab = pNewSlabInfo;
	}
	
	if( pOldSlabInfo->pPrevSlabWithAvail)
	{
		f_assert( pOldSlabInfo != pCacheAlloc->m_pFirstSlabWithAvail);
		pOldSlabInfo->pPrevSlabWithAvail->pNextSlabWithAvail = pNewSlabInfo;
	}
	else
	{
		f_assert( pOldSlabInfo == pCacheAlloc->m_pFirstSlabWithAvail);
		pCacheAlloc->m_pFirstSlabWithAvail = pNewSlabInfo;
	}
	
	if( pOldSlabInfo->pNextSlabWithAvail)
	{
		f_assert( pOldSlabInfo != pCacheAlloc->m_pLastSlabWithAvail);
		pOldSlabInfo->pNextSlabWithAvail->pPrevSlabWithAvail = pNewSlabInfo;
	}
	else
	{
		f_assert( pOldSlabInfo == pCacheAlloc->m_pLastSlabWithAvail);
		pCacheAlloc->m_pLastSlabWithAvail = pNewSlabInfo;
	}
	
#ifdef FLM_DEBUG
	f_memset( pOldSlabInfo, 0, sizeof( SLABINFO));
#endif
}
