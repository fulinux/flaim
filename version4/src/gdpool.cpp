//-------------------------------------------------------------------------
// Desc:	Memory pool routines.
// Tabs:	3
//
//		Copyright (c) 1992-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gdpool.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_UNIX) || defined( FLM_NLM) || defined( FLM_WIN)

	#define PTR_IN_MBLK(p,bp,offs)	(((FLMBYTE *)(p) > (FLMBYTE *)(bp)) && \
												 ((FLMBYTE *)(p) <= (FLMBYTE *)(bp) + (offs)))
#else
	#error Platform not supported
#endif

FSTATIC RCODE GedPoolFreeToMark(
	POOL * 		pPool,
	void *		markPtr);

/****************************************************************************
Desc: Initializes a memory pool
****************************************************************************/
void GedPoolInit(
		POOL *		pPool,
		FLMUINT 		uiBlkSize)
{
	pPool->uiBytesAllocated = 0;
	pPool->lblk = NULL;
	pPool->pPoolStats = NULL;
	pPool->uiBlkSize = uiBlkSize;
}

/****************************************************************************
Desc: Returns a "marker" to the current offset in the memory pool
****************************************************************************/
void * GedPoolMark(
	POOL *		pPool)
{
	return (void *)((pPool->lblk)
					  ? (FLMBYTE *) pPool->lblk + pPool->lblk->uiFreeOfs
					  : (FLMBYTE *)GedPoolAlloc( pPool, 1));
}

/****************************************************************************
Desc: Determine what the initial block size should be for smart pools.	
****************************************************************************/
FINLINE void SetInitialSmartPoolBlkSize( 
	POOL *	pPool)
{
	/* 
		Determine starting block size:
		1) average of bytes allocated / # of frees/resets (average size needed)
		2) add 10% - to minimize extra allocs 
	*/
	pPool->uiBlkSize = (pPool->pPoolStats->uiAllocBytes / pPool->pPoolStats->uiCount);

	pPool->uiBlkSize += (pPool->uiBlkSize / 10);

	if( pPool->uiBlkSize < 512)
		pPool->uiBlkSize = 512;
}

/****************************************************************************
Desc: Update the bytes allocated and alloc count for this pool's POOL_STATS
****************************************************************************/
FINLINE void UpdateSmartPoolStats(
	POOL *	pPool)
{
	if( pPool->uiBytesAllocated)
	{
		POOL_STATS * pStats = pPool->pPoolStats;
		if( (pStats->uiAllocBytes + pPool->uiBytesAllocated) >= 0xFFFF0000)
		{
			pStats->uiAllocBytes = (pStats->uiAllocBytes / pStats->uiCount) * 100;
			pStats->uiCount = 100;
		}
		else
		{
			pStats->uiAllocBytes += pPool->uiBytesAllocated;
			pStats->uiCount++;
		}
		pPool->uiBytesAllocated = 0;
	}
}

/*API~*******************************************************************
Desc:	Initialize a smart pool memory structure. A smart pool is one that
		will adjust it's block allocation size based on statistics it 
		gathers within the POOL_STATS structure. For each pool that user 
		wants to use smart memory management a global POOL_STATS structure
		should be declared. The POOL_STATS structure is used to track the 
		total bytes allocated and determine what the correct pool block
		size should be.
*************************************************************************/
void GedSmartPoolInit(
	POOL *			pPool,
	POOL_STATS *	pPoolStats) 
{
	pPool->lblk    = NULL;
	pPool->uiBytesAllocated = 0;
	pPool->pPoolStats = pPoolStats;	

	if( pPoolStats && pPoolStats->uiCount)
	{
		SetInitialSmartPoolBlkSize( pPool);
	}
	else
	{
		pPool->uiBlkSize = 2048;
	}
}

/*API~***********************************************************************
Desc:	Allocates a block of memory from a memory pool.
Note:	If the number of bytes is more than the what is left in the
		current block then a new block will be allocated and the lbkl element
		of the PMS will be updated.
*END************************************************************************/
void * GedPoolAlloc(
	POOL * 		pPool,
	FLMUINT		uiSize)
{
	MBLK *		blk = pPool->lblk;
	MBLK *		old_lblk = blk;
	FLMBYTE *	freePtr;
	FLMUINT		uiBlkSize;

	// Adjust the size to a machine word boundary 
	// NOTE: ORed and ANDed 0x800.. & 0x7FFF to prevent partial 
	// stalls on Netware.
	
	if( uiSize & (FLM_ALLOC_ALIGN | 0x80000000))
	{
		uiSize = ((uiSize + FLM_ALLOC_ALIGN) & (~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
	}

	// Check if room in block

	if( ! blk || uiSize > blk->uiFreeSize)
	{
		// Check if previous block has space for allocation

		if( blk && blk->pPrevBlk != NULL && uiSize <= blk->pPrevBlk->uiFreeSize)
		{
			blk = blk->pPrevBlk;
			goto Exit;
		}

		/* Not enough memory in block - allocate new block */

		/* Determine the block size:
			1) start with max of last block size, initial pool size, or alloc size
			2) if this is an extra block alloc then increase the size by 1/2
			3) adjust size to include blk header */

		uiBlkSize = (blk) ? blk->uiBlkSize : pPool->uiBlkSize;
		uiBlkSize = f_max( uiSize, uiBlkSize);

		if( blk && (uiBlkSize == blk->uiBlkSize) && uiBlkSize <= 32769)
			uiBlkSize += uiBlkSize / 2;
		
		uiBlkSize += sizeof( MBLK);
 
		if( RC_BAD( f_alloc( uiBlkSize, &blk)))
			return( NULL);

		// Initialize the block elements
		
		blk->uiBlkSize = uiBlkSize;
		blk->uiFreeOfs = sizeof( MBLK);	
		blk->uiFreeSize = uiBlkSize - sizeof( MBLK);

		pPool->lblk = blk;
		blk->pPrevBlk = old_lblk;
	}

Exit:
	freePtr = (FLMBYTE *) blk;
	freePtr += blk->uiFreeOfs;
	blk->uiFreeOfs += uiSize;
	blk->uiFreeSize -= uiSize;

	pPool->uiBytesAllocated += uiSize;
	return( (void *) freePtr);
}

/*API~********************************************************************
Desc:	Allocates memory from a pool and initializes all bytes to zero.
*END*********************************************************************/
void * GedPoolCalloc(
	POOL * 		pPool,
  	FLMUINT		uiSize)
{
	void *		ptr;

	if( (ptr = GedPoolAlloc( pPool, uiSize)) != NULL)
	{
		f_memset( ptr, 0, uiSize);
	}
	return ptr;
}

/*API~***********************************************************************
Desc:	Releases all memory allocated to a pool.
Note:	All memory allocated to the pool is returned to the operating system.
*END************************************************************************/
RCODE GedPoolFree(
	POOL *		pPool)
{
	MBLK *		blk = pPool->lblk;
	MBLK *		prevBlk;

	while( blk)
	{
		prevBlk = blk->pPrevBlk;
		f_free( &blk);
		blk = prevBlk;
	}

	pPool->lblk = NULL;

	// For Smart Pools update pool statictics

	if( pPool->pPoolStats)
	{
		UpdateSmartPoolStats( pPool);
	}

	return( FERR_OK);
}

/*API~***********************************************************************
Desc:	Resets memory blocks allocated to a pool.
Note:	Will reset the free space in the first memory block, and if
		any extra blocks exist they will be freed (destroyed).
*END************************************************************************/
RCODE GedPoolReset(
	POOL *		pPool,
	void *		markPtr)
{
	MBLK *		blk = pPool->lblk;
	MBLK *		prevBlk;

	if( ! blk)
		return( FERR_OK);

	// For Smart Pools update pool statictics

	if( pPool->pPoolStats)
	{
		UpdateSmartPoolStats( pPool);
	}

	if( markPtr)
	{
		return( GedPoolFreeToMark( pPool, markPtr));
	}

	// Free all blocks except last one in chain -- which is really
	// the first block allocated.  This will help us keep memory from 
	// getting fragmented.

	while( blk->pPrevBlk)
	{
		prevBlk = blk->pPrevBlk;
		f_free( &blk);
		blk = prevBlk;
	}

	if( (blk->uiBlkSize - sizeof(MBLK)) > pPool->uiBlkSize)
	{
		// The first block was not the default size, so FREE it

		f_free( &blk);
		pPool->lblk = NULL;
	}
	else
	{
		// Reset the allocation pointers in the first block
		
		blk->uiFreeOfs  = sizeof( MBLK);
		blk->uiFreeSize = blk->uiBlkSize - sizeof( MBLK);
		pPool->lblk = blk;

#ifdef FLM_MEM_CHK
		{
			FLMBYTE * 	ptr = (FLMBYTE *) blk;
			ptr += blk->uiFreeOfs;
			f_memset( ptr, 'r', blk->uiFreeSize);
		}
#endif
	}

	// On smart pools, adjust the initial block size on pool resets

	if( pPool->pPoolStats)
	{
		SetInitialSmartPoolBlkSize( pPool);
	}

	return( FERR_OK);
}

/****************************************************************************
Desc:	Frees memory until the markPtr is found.
****************************************************************************/
FSTATIC RCODE GedPoolFreeToMark(
	POOL *		pPool,
	void *		markPtr)
{
	MBLK *		blk = pPool->lblk;
	MBLK *		prevBlk;

	pPool->lblk = NULL;							/* Initialize PMS to no BLOCKS */

	while(  blk)								/* Free all allocated blks in chain */
	{
		prevBlk = blk->pPrevBlk;     		/* Save pointer to prev block */

		/* Check for mark point */
		if( PTR_IN_MBLK( markPtr, blk, blk->uiBlkSize))
		{
			FLMUINT  uiOldFreeOfs = blk->uiFreeOfs;

			/* Reset freeOfs and freeSize variables */
			blk->uiFreeOfs = (FLMUINT)((FLMBYTE *)markPtr - (FLMBYTE *)blk);
			blk->uiFreeSize = blk->uiBlkSize - blk->uiFreeOfs;

#if defined( FLM_MEM_CHK) || defined( MEM_TEST)
			{
			/*	memset the memory so someone pointing to it will get a error.*/
			FLMBYTE * 	ptr = (FLMBYTE *) blk;
			ptr += blk->uiFreeOfs;
			f_memset( ptr, 'r', blk->uiFreeSize);	// Set memory to 'r' for Reset
			}
#endif

			// For Smart Pools deduct the bytes allocated since pool mark
			
			if( pPool->pPoolStats)
			{
				flmAssert( uiOldFreeOfs >= blk->uiFreeOfs);
				pPool->uiBytesAllocated -= (uiOldFreeOfs - blk->uiFreeOfs);
			}

			break;
		}

		if( pPool->pPoolStats)
		{
			pPool->uiBytesAllocated -= (blk->uiFreeOfs - sizeof( MBLK));
		}

		f_free( &blk);
		blk = prevBlk;							/* Point to previous block */
	}

	if( blk)
		pPool->lblk = blk;

	return( FERR_OK);
}


