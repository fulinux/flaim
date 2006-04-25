//-------------------------------------------------------------------------
// Desc:	Memory management using fixed-size allocators - for dealing with
//			memory fragmentation issues - definitions.
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
// $Id: flfixed.h 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef FLFIXED_H
#define FLFIXED_H

#include "fpackon.h"

/****************************************************************************
Desc:
****************************************************************************/
class F_SlabManager : public F_Base
{
public:

	F_SlabManager();

	virtual ~F_SlabManager();

	RCODE setup(
		FLMUINT 				uiPreallocSize,
		FLMUINT				uiMinSlabSize = 64 * 1024);
		
	RCODE allocSlab(
		void **				ppSlab);
		
	void freeSlab(
		void **				ppSlab);
		
	FINLINE FLMUINT getSlabSize( void)
	{
		return( m_uiSlabSize);
	}
	
	RCODE resize(
		FLMUINT 				uiNumBytes,
		FLMUINT *			puiActualSize = NULL);

private:

	void freeAllSlabs( void);
	
	void * allocSlabFromSystem( void);
	
	void releaseSlabToSystem(
		void *				pSlab);

	RCODE sortSlabList( void);

	typedef struct SLABHEADER
	{
		void *				pPrev;
		void *				pNext;
	} SLABHEADER;

	static FLMINT slabAddrCompareFunc(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2);

	static void slabAddrSwapFunc(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2);
	
	F_MUTEX					m_hMutex;
	void *					m_pFirstInSlabList;
	void *					m_pLastInSlabList;
	FLMUINT					m_uiSlabSize;
	FLMUINT					m_uiTotalSlabs;
	FLMUINT					m_uiAvailSlabs;
	FLMUINT					m_uiInUseSlabs;
	void *					m_pLowPrealloc;
	void *					m_pHighPrealloc;
};

// Cell sizes for buffer allocator

#define CELL_SIZE_0	64
#define CELL_SIZE_1	128
#define CELL_SIZE_2	192
#define CELL_SIZE_3	320
#define CELL_SIZE_4	512
#define CELL_SIZE_5	672
#define CELL_SIZE_6	832
#define CELL_SIZE_7	1088
#define CELL_SIZE_8	1344
#define CELL_SIZE_9	1760
#define CELL_SIZE_10	2176
#define CELL_SIZE_11	2848
#define CELL_SIZE_12	3520
#define CELL_SIZE_13	4608
#define CELL_SIZE_14	5152
#define CELL_SIZE_15	5696
#define CELL_SIZE_16 8164
#define CELL_SIZE_17 13068
#define CELL_SIZE_18 16340
#define CELL_SIZE_19 21796
#define CELL_SIZE_20 32700
#define CELL_SIZE_21 65420

#define NUM_BUF_ALLOCATORS	22

typedef struct FLM_ALLOC_USAGE
{
	FLMUINT64			ui64Slabs;
	FLMUINT64			ui64SlabBytes;
	FLMUINT64			ui64AllocatedCells;
	FLMUINT64			ui64FreeCells;
} FLM_ALLOC_USAGE;

typedef void (* FLM_RELOC_FUNC)(
	void *				pvOldAlloc,
	void *				pvNewAlloc);

typedef FLMBOOL (* FLM_CAN_RELOC_FUNC)(
	void *				pvOldAlloc);

typedef void (* FLM_ALLOC_INIT_FUNC)(
	void *				pvAlloc);
	
/****************************************************************************
Desc:	Class to provide an efficient means of providing many allocations
		of a fixed size.
****************************************************************************/
class F_FixedAlloc : public F_Base
{
public:

	F_FixedAlloc();

	virtual ~F_FixedAlloc();

	RCODE setup(
		F_SlabManager *	pSlabManager,
		FLMBOOL				bUseMutex,
		FLMUINT				uiCellSize,
		FLMUINT *			puiTotalBytesAllocated);

	RCODE setup(
		F_SlabManager *	pSlabManager,
		F_MUTEX *			phMutex,
		FLMUINT				uiCellSize,
		FLMUINT *			puiTotalBytesAllocated);
		
	FINLINE void setRelocationFuncs(
		FLM_CAN_RELOC_FUNC	fnCanRelocate,
		FLM_RELOC_FUNC			fnRelocate)
	{
		flmAssert( fnCanRelocate);
		flmAssert( fnRelocate);

		m_fnCanRelocate = fnCanRelocate;
		m_fnRelocate = fnRelocate;
	}

	FINLINE void * allocCell(
		void *				pvInitialData = NULL,
		FLMUINT				uiDataSize = 0)
	{
		void *	pvCell;

		if( m_phMutex)
		{
			f_mutexLock( *m_phMutex);
		}

		pvCell = getCell();
		
		if( pvCell && pvInitialData)
		{
			f_memcpy( pvCell, pvInitialData, uiDataSize);
		}
	
		if( m_phMutex)
		{
			f_mutexUnlock( *m_phMutex);
		}
	
		return( pvCell);
	}

	FINLINE void * allocCell(
		FLM_ALLOC_INIT_FUNC	fnAllocInit)
	{
		void *	pvCell;

		if( m_phMutex)
		{
			f_mutexLock( *m_phMutex);
		}

		pvCell = getCell();
		
		if( pvCell && fnAllocInit)
		{
			fnAllocInit( pvCell);
		}
	
		if( m_phMutex)
		{
			f_mutexUnlock( *m_phMutex);
		}
	
		return( pvCell);
	}
	
	FINLINE void freeCell( 
		void *		ptr)
	{
		freeCell( ptr, FALSE, FALSE, NULL);
	}

	void freeUnused( void);

	void freeAll( void);

	void getStats(
		FLM_ALLOC_USAGE *		pUsage);

	FINLINE FLMUINT getCellSize( void)
	{
		return( m_uiCellSize);
	}
	
	void defragmentMemory( void);
	
	void incrementTotalBytesAllocated(
		FLMUINT					uiCount);
	
	void decrementTotalBytesAllocated(
		FLMUINT					uiCount);
		
	FLMUINT getTotalBytesAllocated( void);
	
	F_MUTEX * getMutex( void)
	{
		return( m_phMutex);
	}

	typedef struct BLOCK
	{
		void *		pvAllocator;
		BLOCK *		pNext;
		BLOCK *		pPrev;
		BLOCK *		pNextBlockWithAvailCells;
		BLOCK *		pPrevBlockWithAvailCells;
		FLMBYTE *	pLocalAvailCellListHead;
		FLMUINT32	ui32NextNeverUsedCell;
		FLMUINT32	ui32AvailCellCount;
		FLMUINT32	ui32AllocatedCells;
	} BLOCK;

	typedef struct CELLHEADER
	{
		BLOCK *		pContainingBlock;
#ifdef FLM_DEBUG
		FLMUINT *	puiStack;
#endif
	} CELLHEADER;

	typedef struct CELLAVAILNEXT
	{
		FLMBYTE *	pNextInList;
#ifdef FLM_DEBUG
		char			szDebugPattern[ 8];
#endif
	} CELLAVAILNEXT;

private:

	void * getCell( void);

	BLOCK * getAnotherBlock( void);

	static FINLINE FLMUINT getAllocAlignedSize(
		FLMUINT		uiAskedForSize)
	{
		return( (uiAskedForSize + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN));
	}

	void freeBlock( 
		BLOCK *			pBlock);

	void freeCell(
		void *		pCell,
		FLMBOOL		bMutexLocked,
		FLMBOOL		bFreeIfEmpty,
		FLMBOOL *	pbFreedBlock);

#ifdef FLM_DEBUG
	void testForLeaks( void);
#endif

	FINLINE static FLMINT blockAddrCompareFunc(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2)
	{
		BLOCK *		pBlock1 = (((BLOCK **)pvBuffer)[ uiPos1]);
		BLOCK *		pBlock2 = (((BLOCK **)pvBuffer)[ uiPos2]);

		flmAssert( pBlock1 != pBlock2);

		if( pBlock1 < pBlock2)
		{
			return( -1);
		}

		return( 1);
	}

	FINLINE static void blockAddrSwapFunc(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2)
	{
		BLOCK **		ppBlock1 = &(((BLOCK **)pvBuffer)[ uiPos1]);
		BLOCK **		ppBlock2 = &(((BLOCK **)pvBuffer)[ uiPos2]);
		BLOCK *		pTmp;

		pTmp = *ppBlock1;
		*ppBlock1 = *ppBlock2;
		*ppBlock2 = pTmp;
	}

	F_SlabManager *		m_pSlabManager;
	BLOCK *					m_pFirstBlock;
	BLOCK *					m_pLastBlock;
	BLOCK *					m_pFirstBlockWithAvailCells;
	BLOCK *					m_pLastBlockWithAvailCells;
	FLMBOOL					m_bAvailListSorted;
	FLMUINT					m_uiBlocksWithAvailCells;
	FLMUINT					m_uiBlockHeaderSize;
	FLMUINT					m_uiCellHeaderSize;
	FLMUINT					m_uiCellSize;
	FLMUINT					m_uiSizeOfCellAndHeader; 
	FLMUINT					m_uiTotalFreeCells;
	FLMUINT					m_uiCellsPerBlock;

	F_MUTEX					m_hLocalMutex;
	F_MUTEX *				m_phMutex;

	FLM_CAN_RELOC_FUNC	m_fnCanRelocate;
	FLM_RELOC_FUNC			m_fnRelocate;
	FLMUINT *				m_puiTotalBytesAllocated;
	FLMUINT					m_uiSlabSize;
	
	// Members specifically for stats

	FLMUINT					m_uiAllocatedSlabs;
	FLMUINT					m_uiAllocatedCells;
	FLMUINT					m_uiAllocatedCellWatermark;
	FLMUINT					m_uiEverFreedCells;

friend class F_BufferAlloc;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_BufferAlloc : public F_Base
{
public:

	F_BufferAlloc()
	{
		f_memset( m_ppAllocators, 0, sizeof( m_ppAllocators));
		m_pSlabManager = NULL;
		m_puiTotalBytesAllocated = NULL;
		m_phMutex = NULL;
	}

	virtual ~F_BufferAlloc();

	RCODE setup(
		F_SlabManager *		pSlabManager,
		F_MUTEX *				phMutex,
		FLMUINT * 				puiTotalBytesAllocated);

	void setRelocationFuncs(
		FLM_CAN_RELOC_FUNC	fnCanRelocate,
		FLM_RELOC_FUNC			fnRelocate);

	RCODE allocBuf(
		FLMUINT				uiSize,
		void *				pvInitialData,
		FLMUINT				uiDataSize,
		FLMBYTE **			ppucBuffer,
		FLMBOOL *			pbAllocatedOnHeap);

	RCODE reallocBuf(
		FLMUINT				uiOldSize,
		FLMUINT				uiNewSize,
		void *				pvInitialData,
		FLMUINT				uiDataSize,
		FLMBYTE **			ppucBuffer,
		FLMBOOL *			pbAllocatedOnHeap);

	void freeBuf(
		FLMUINT				uiSize,
		FLMBYTE **			ppucBuffer);

	FLMUINT getTrueSize(
		FLMUINT				uiSize,
		FLMBYTE *			pucBuffer);

	void defragmentMemory( void);
	
private:

	F_FixedAlloc * getAllocator(
		FLMUINT				uiSize);

	F_SlabManager *		m_pSlabManager;
	F_FixedAlloc *			m_ppAllocators[ NUM_BUF_ALLOCATORS];
	FLMUINT *				m_puiTotalBytesAllocated;
	F_MUTEX *				m_phMutex;
};

#include "fpackoff.h"

#endif // FLFIXED_H
