//------------------------------------------------------------------------------
// Desc: Special allocators for making many fixed-size allocations.
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
// $Id: flfixed.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FLFIXED_H
#define FLFIXED_H

// Cell sizes for buffer allocator

#define CELL_SIZE_0			16
#define CELL_SIZE_1			32
#define CELL_SIZE_2			64
#define CELL_SIZE_3			128
#define CELL_SIZE_4			192
#define CELL_SIZE_5			320
#define CELL_SIZE_6			512
#define CELL_SIZE_7			672
#define CELL_SIZE_8			832
#define CELL_SIZE_9			1088
#define CELL_SIZE_10			1344
#define CELL_SIZE_11			1760
#define CELL_SIZE_12			2176
#define CELL_SIZE_13			2848
#define CELL_SIZE_14			3520
#define CELL_SIZE_15			4608
#define CELL_SIZE_16			5152
#define CELL_SIZE_17			5696
#define CELL_SIZE_18 		8164
#define CELL_SIZE_19 		13068
#define CELL_SIZE_20 		16340
#define CELL_SIZE_21 		21796
#define MAX_CELL_SIZE		CELL_SIZE_21

#define NUM_BUF_ALLOCATORS	22

/****************************************************************************
Desc:
****************************************************************************/
class F_SlabManager : public XF_RefCount, public XF_Base
{
public:

	F_SlabManager();

	virtual ~F_SlabManager();

	RCODE setup(
		FLMUINT 				uiPreallocSize);
		
	RCODE allocSlab(
		void **				ppSlab,
		FLMBOOL				bMutexLocked);
		
	void freeSlab(
		void **				ppSlab,
		FLMBOOL				bMutexLocked);
		
	RCODE resize(
		FLMUINT 				uiNumBytes,
		FLMUINT *			puiActualSize = NULL,
		FLMBOOL				bMutexLocked = FALSE);

	FINLINE void incrementTotalBytesAllocated(
		FLMUINT					uiCount,
		FLMBOOL					bMutexLocked)
	{
		if( !bMutexLocked)
		{
			lockMutex();
		}
		
		m_uiTotalBytesAllocated += uiCount;	
		
		if( !bMutexLocked)
		{
			unlockMutex();
		}
	}

	FINLINE void decrementTotalBytesAllocated(
		FLMUINT					uiCount,
		FLMBOOL					bMutexLocked)
	{
		if( !bMutexLocked)
		{
			lockMutex();
		}
		
		flmAssert( m_uiTotalBytesAllocated >= uiCount);
		m_uiTotalBytesAllocated -= uiCount;	
		
		if( !bMutexLocked)
		{
			unlockMutex();
		}
	}

	FINLINE FLMUINT getSlabSize( void)
	{
		return( m_uiSlabSize);
	}

	FINLINE FLMUINT getTotalSlabs( void)
	{
		return( m_uiTotalSlabs);
	}
	
	FINLINE void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
	
	FINLINE void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}
	
	FINLINE FLMUINT totalBytesAllocated( void)
	{
		return( m_uiTotalBytesAllocated);
	}

	FINLINE FLMUINT availSlabs( void)
	{
		return( m_uiAvailSlabs);
	}
	
#ifdef FLM_CACHE_PROTECT	
	void protectSlab(
		void *			pSlab);

	void unprotectSlab(
		void *			pSlab);
#endif
		
private:

	void freeAllSlabs( void);
	
	void * allocSlabFromSystem( void);
	
	void releaseSlabToSystem(
		void *				pSlab);

	RCODE sortSlabList( void);

	typedef struct
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
	FLMUINT					m_uiTotalBytesAllocated;
	void *					m_pFirstInSlabList;
	void *					m_pLastInSlabList;
	FLMUINT					m_uiSlabSize;
	FLMUINT					m_uiTotalSlabs;
	FLMUINT					m_uiAvailSlabs;
	FLMUINT					m_uiInUseSlabs;
	FLMUINT					m_uiPreallocSlabs;
#ifdef FLM_SOLARIS
	int						m_DevZero;
#endif

friend class F_FixedAlloc;
};

/****************************************************************************
Desc:	Class with two virtual functions - canRelocate and relocate.
****************************************************************************/
class IF_Relocator
{
public:
	virtual void relocate(
		void *	pvOldAlloc,
		void *	pvNewAlloc) = 0;

	virtual FLMBOOL canRelocate(
		void *	pvOldAlloc) = 0;
};

/****************************************************************************
Desc:	Class to provide an efficient means of providing many allocations
		of a fixed size.
****************************************************************************/
class F_FixedAlloc : public XF_RefCount, public XF_Base
{
public:

	F_FixedAlloc();

	virtual ~F_FixedAlloc();

	RCODE setup(
		IF_Relocator *			pRelocator,
		F_SlabManager *		pSlabManager,
		FLMBOOL					bMemProtect,
		FLMUINT					uiCellSize,
		XFLM_SLAB_USAGE *		pUsageStats);

	FINLINE void * allocCell(
		IF_Relocator *		pRelocator,
		void *				pvInitialData = NULL,
		FLMUINT				uiDataSize = 0,
		FLMBOOL				bMutexLocked = FALSE)
	{
		void *	pvCell;
		
		flmAssert( pRelocator);

		if( !bMutexLocked)
		{
			m_pSlabManager->lockMutex();
		}
		
		if( (pvCell = getCell( pRelocator)) == NULL)
		{
			goto Exit;
		}
		
		if( uiDataSize == sizeof( FLMUINT *))
		{
			*((FLMUINT *)pvCell) = *((FLMUINT *)pvInitialData); 
		}
		else if( uiDataSize)
		{
			f_memcpy( pvCell, pvInitialData, uiDataSize);
		}
		
	Exit:
		
		if( !bMutexLocked)
		{
			m_pSlabManager->unlockMutex();
		}
		
		return( pvCell);
	}

	FINLINE void freeCell( 
		void *		ptr,
		FLMBOOL		bMutexLocked)
	{
		freeCell( ptr, bMutexLocked, FALSE, NULL);
	}

	void freeUnused( void);

	void freeAll( void);

	FINLINE FLMUINT getCellSize( void)
	{
		return( m_uiCellSize);
	}
	
	void defragmentMemory( void);
	
#ifdef FLM_CACHE_PROTECT	
	void protectCell(
		void *					pvCell);
	
	void unprotectCell(
		void *					pvCell);
#endif
		
	typedef struct Slab
	{
		void *		pvAllocator;
		Slab *		pNext;
		Slab *		pPrev;
		Slab *		pNextSlabWithAvailCells;
		Slab *		pPrevSlabWithAvailCells;
		FLMBYTE *	pLocalAvailCellListHead;
		FLMUINT16	ui16NextNeverUsedCell;
		FLMUINT16	ui16AvailCellCount;
		FLMUINT16	ui16AllocatedCells;
#ifdef FLM_CACHE_PROTECT	
		FLMUINT32	ui16UnprotectCount;
#endif
	} SLAB;

	typedef struct CELLHEADER
	{
		SLAB *			pContainingSlab;
#ifdef FLM_DEBUG
		FLMUINT *		puiStack;
#endif
	} CELLHEADER;

	typedef struct CELLHEADER2
	{
		CELLHEADER		cellHeader;
		IF_Relocator *	pRelocator;
	} CELLHEADER2;

	typedef struct CellAvailNext
	{
		FLMBYTE *	pNextInList;
#ifdef FLM_DEBUG
		FLMBYTE		szDebugPattern[ 8];
#endif
	} CELLAVAILNEXT;

private:

#ifdef FLM_CACHE_PROTECT	
	void protectSlab(
		SLAB *			pSlab,
		FLMBOOL			bMutexLocked);
		
	void unprotectSlab(
		SLAB *			pSlab,
		FLMBOOL			bMutexLocked);
#endif
		
	void * getCell(
		IF_Relocator *		pRelocator);

	SLAB * getAnotherSlab( void);

	static FINLINE FLMUINT getAllocAlignedSize(
		FLMUINT		uiAskedForSize)
	{
		return( (uiAskedForSize + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN));
	}

	void freeSlab( 
		SLAB *			pSlab);

	void freeCell(
		void *		pCell,
		FLMBOOL		bMutexLocked,
		FLMBOOL		bFreeIfEmpty,
		FLMBOOL *	pbFreedSlab);

#ifdef FLM_DEBUG
	void testForLeaks( void);
#endif

	FINLINE static FLMINT slabAddrCompareFunc(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2)
	{
		SLAB *		pSlab1 = (((SLAB **)pvBuffer)[ uiPos1]);
		SLAB *		pSlab2 = (((SLAB **)pvBuffer)[ uiPos2]);

		flmAssert( pSlab1 != pSlab2);

		if( pSlab1 < pSlab2)
		{
			return( -1);
		}

		return( 1);
	}

	FINLINE static void slabAddrSwapFunc(
		void *		pvBuffer,
		FLMUINT		uiPos1,
		FLMUINT		uiPos2)
	{
		SLAB **		ppSlab1 = &(((SLAB **)pvBuffer)[ uiPos1]);
		SLAB **		ppSlab2 = &(((SLAB **)pvBuffer)[ uiPos2]);
		SLAB *		pTmp;

		pTmp = *ppSlab1;
		*ppSlab1 = *ppSlab2;
		*ppSlab2 = pTmp;
	}

	F_SlabManager *		m_pSlabManager;
	SLAB *					m_pFirstSlab;
	SLAB *					m_pLastSlab;
	SLAB *					m_pFirstSlabWithAvailCells;
	SLAB *					m_pLastSlabWithAvailCells;
	IF_Relocator *			m_pRelocator;
	FLMBOOL					m_bAvailListSorted;
	FLMUINT					m_uiSlabsWithAvailCells;
	FLMUINT					m_uiSlabHeaderSize;
	FLMUINT					m_uiCellHeaderSize;
	FLMUINT					m_uiCellSize;
	FLMUINT					m_uiSizeOfCellAndHeader; 
	FLMUINT					m_uiTotalFreeCells;
	FLMUINT					m_uiCellsPerSlab;
	FLMUINT					m_uiSlabSize;
	
	// Members specifically for stats
	
	XFLM_SLAB_USAGE *		m_pUsageStats;

	// Memory protection

#ifdef FLM_CACHE_PROTECT	
	FLMBOOL					m_bMemProtectionEnabled;
#endif
	
friend class F_BufferAlloc;
friend class F_MultiAlloc;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_BufferAlloc : public XF_RefCount, public XF_Base
{
public:

	F_BufferAlloc()
	{
		f_memset( m_ppAllocators, 0, sizeof( m_ppAllocators));
		m_pSlabManager = NULL;
	}

	virtual ~F_BufferAlloc();

	RCODE setup(
		F_SlabManager *		pSlabManager,
		FLMBOOL					bMemProtect,
		XFLM_SLAB_USAGE *		pUsageStats);

	RCODE allocBuf(
		IF_Relocator *		pRelocator,
		FLMUINT				uiSize,
		void *				pvInitialData,
		FLMUINT				uiDataSize,
		FLMBYTE **			ppucBuffer,
		FLMBOOL *			pbAllocatedOnHeap = NULL);

	RCODE reallocBuf(
		IF_Relocator *		pRelocator,
		FLMUINT				uiOldSize,
		FLMUINT				uiNewSize,
		void *				pvInitialData,
		FLMUINT				uiDataSize,
		FLMBYTE **			ppucBuffer,
		FLMBOOL *			pbAllocatedOnHeap = NULL);

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
};

/****************************************************************************
Desc:
****************************************************************************/
class F_MultiAlloc : public XF_RefCount, public XF_Base
{
public:

	F_MultiAlloc()
	{
		m_pSlabManager = NULL;
		m_puiCellSizes = NULL;
		m_ppAllocators = NULL;
	}

	~F_MultiAlloc()
	{
		cleanup();
	}

	RCODE setup(
		F_SlabManager *		pSlabManager,
		FLMBOOL					bMemProtect,
		FLMUINT *				puiCellSizes,
		XFLM_SLAB_USAGE *		pUsageStats);

	RCODE allocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiSize,
		FLMBYTE **				ppucBuffer,
		FLMBOOL					bMutexLocked = FALSE);

	RCODE reallocBuf(
		IF_Relocator *			pRelocator,
		FLMUINT					uiNewSize,
		FLMBYTE **				ppucBuffer,
		FLMBOOL					bMutexLocked = FALSE);

	FINLINE void freeBuf(
		FLMBYTE **				ppucBuffer)
	{
		if( ppucBuffer && *ppucBuffer)
		{
			getAllocator( *ppucBuffer)->freeCell( *ppucBuffer, FALSE);
			*ppucBuffer = NULL;
		}
	}

	void defragmentMemory( void);

	FINLINE FLMUINT getTrueSize(
		FLMBYTE *				pucBuffer)
	{
		return( getAllocator( pucBuffer)->getCellSize());
	}

#ifdef FLM_CACHE_PROTECT	
	void protectBuffer(
		void *					pvBuffer,
		FLMBOOL					bMutexLocked = FALSE);

	void unprotectBuffer(
		void *					pvBuffer,
		FLMBOOL					bMutexLocked = FALSE);
#endif

	FINLINE void lockMutex( void)
	{
		m_pSlabManager->lockMutex();
	}

	FINLINE void unlockMutex( void)
	{
		m_pSlabManager->unlockMutex();
	}
		
private:

	F_FixedAlloc * getAllocator(
		FLMUINT					uiSize);

	F_FixedAlloc * getAllocator(
		FLMBYTE *				pucBuffer);

	void cleanup( void);

	F_SlabManager *			m_pSlabManager;
	FLMUINT *					m_puiCellSizes;
	F_FixedAlloc **			m_ppAllocators;
};

#endif // FLFIXED_H
