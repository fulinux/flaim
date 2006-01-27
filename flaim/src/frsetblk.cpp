//-------------------------------------------------------------------------
// Desc:	Result set blocks
// Tabs:	3
//
//		Copyright (c) 1996-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frsetblk.cpp 12320 2006-01-19 15:53:51 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Public:	FResultSetBlk
Desc:		Constructor
Notes:
****************************************************************************/

FResultSetBlk::FResultSetBlk()
{
	m_pNext = m_pPrev = NULL;
	reset();
}


/****************************************************************************
Public:	reset
Desc:		Reset a block so it can be reused.
****************************************************************************/
void FResultSetBlk::reset( void)
{
	flmAssert( !m_pNext && !m_pPrev);
	
	// Initialize all of the member variables
	// between this constructor, SetBuffer() and Setup().

	m_BlockHeader.ui64FilePos = RSBLK_UNSET_FILE_POS;
	m_BlockHeader.uiEntryCount = 0;
	// m_BlockHeader.uiBlockSize = RSBLK_BLOCK_SIZE; set in Setup()
	//m_BlockHeader.FirstBlock = ?;		set in Setup().
	//m_BlockHeader.LastBlock = ?;		set in Setup().
	
	m_ppFileHdl64 = NULL;
	m_uiBlkEntryPosition = RS_POSITION_NOT_SET;
	m_iEntryPos = 0;
	//m_uiEntrySize =			set in Setup().
	//m_bEntriesInOrder = 	set in Setup().
	//m_bFixedEntrySize;		set in Setup().
	m_bDuplicateFound = FALSE;
	m_bPositioned = FALSE;
	m_bModifiedEntry = FALSE;
	m_pBlockBuf = NULL;
}

/****************************************************************************
Public:	Setup
Desc:		Setup the result set block - passing in needed member variables
Notes:	Must be called.
****************************************************************************/

void FResultSetBlk::Setup(
	F_64BitFileHandle **
						ppFileHdl64,	// file handle to use for temp file.
	RSET_COMPARE_FUNC_p 				// Zero or record compare function.
						fnCompare,		// Returns (FLMINT) -1, 0 or 1 values.
	void *			UserValue,		// UserValue for callback.
	FLMUINT			uiEntrySize,
	FLMBOOL			bFirstInList,	// Use RSBLK_IS_FIRST_IN_LIST for true.
	FLMBOOL			bDropDuplicates,	// If TRUE drop duplicates
	FLMBOOL			bEntriesInOrder)	// TRUE when entries are in order.
{
	flmAssert( ppFileHdl64 != NULL);

	m_ppFileHdl64 = ppFileHdl64;
	m_fnCompare = fnCompare;
	m_UserValue = UserValue;
	m_uiEntrySize = uiEntrySize;
	m_BlockHeader.bFirstBlock = bFirstInList;
	m_BlockHeader.bLastBlock = FALSE;		// Set for real in the flush call.

	m_bFixedEntrySize = m_uiEntrySize ? TRUE : FALSE;	// Friend member

	if( !m_uiEntrySize )
	{
		m_uiEntrySize = BLKOFFSET_SIZE + LENGTH_SIZE;
	}
	m_bDropDuplicates = bDropDuplicates;
	m_bEntriesInOrder = bEntriesInOrder;

	// Other variables have been setup from the constructor call.
}


/****************************************************************************
Public:	SetBuffer
Desc:		The buffer is NOT allocated the by the result set block object.
			Setup the pBuffer and associated variables.  Read in the data
			for this block if necessary.  If NULL is passed in as pBuffer
			then this block is not the active block anymore.
Notes:	Must be called before other methods below are called.
****************************************************************************/

RCODE		FResultSetBlk::SetBuffer(
	FLMBYTE *		pBuffer,				// Working buffer or NULL 
	FLMUINT			uiBufferLength)	// Default value is RSBLK_BLOCK_SIZE.
{
	RCODE				rc = FERR_OK;

	// If a buffer is defined then read in the data from disk.

	f_yieldCPU();
	
	if( pBuffer)
	{
		m_pBlockBuf = pBuffer;
		// Is there data already in the block?
		if( ! m_BlockHeader.uiEntryCount)
		{
			// uiBlockSize is the final block size after squeeze.
			// uiLengthRemaining is working value of bytes available.

			m_BlockHeader.uiBlockSize = uiBufferLength;
			m_uiLengthRemaining = uiBufferLength;
			m_pNextEntryPtr = m_pBlockBuf;
			if( m_bFixedEntrySize)
			{
				m_pEndPoint = m_pBlockBuf;
			}
			else	// variable length entries
			{
				m_pEndPoint = m_pBlockBuf + uiBufferLength;
			}
		}
		else
		{
			// Read in the data if necessary.
			if( RC_BAD( rc = Read()))		// Sets bPositioned to TRUE on success.
				goto Exit;
		}
		// GWBUG: 46817
		// The block is now in focus
		m_bPositioned = TRUE;
	}
	else // inactivating block so the buffer can be reused.
	{
		// Check if the block has been modified

		if( m_bModifiedEntry )
		{
			// Is this a lone block?
			if( !m_BlockHeader.bLastBlock || !m_BlockHeader.bFirstBlock )
			{
				if( RC_BAD( rc = Write()))
					goto Exit;
			}
			m_bModifiedEntry = FALSE;
		}
		// GWBUG: 46817
		// The block is now out of focus 
		
		m_bPositioned = FALSE;
		m_pNextEntryPtr = m_pEndPoint = m_pBlockBuf = NULL;
	}
Exit:
	return( rc);
}


/****************************************************************************
Public:	AddEntry
Desc:		Add a variable length entry to the result set.  If fixed length
			entry then call AddEntry for fixed length entries.
Ret:     RCODE - FERR_OK or FERR_EOF_HIT when full 
Notes:
****************************************************************************/

RCODE FResultSetBlk::AddEntry(
	FLMBYTE *		pEntry,
	FLMUINT			uiEntryLength)
{
	RCODE				rc = FERR_OK;
	BLKOFFSET		uEntryOffset;
	FLMUINT			uiAlignLength;			// Length taking into account alignment.

	flmAssert( NULL != m_pBlockBuf );

	// Was setup called for fixed length entries?

	if( m_bFixedEntrySize )
	{
		rc = AddEntry( pEntry );
		goto Exit;
	}

	uiAlignLength = (uiEntryLength + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN);

	// Check to see if the current buffer will overflow. 

	if( m_uiLengthRemaining < uiAlignLength + BLKOFFSET_SIZE + LENGTH_SIZE )
	{
		// Caller should call Flush and setup correctly what to do next.
		
		rc = RC_SET( FERR_EOF_HIT );
		goto Exit;
	}

	// Copy entry and compute the offset value for pNextEntryPtr.

	m_pEndPoint -= uiAlignLength;
	f_memcpy( m_pEndPoint, pEntry, uiEntryLength );

	uEntryOffset = (BLKOFFSET) (m_pEndPoint - m_pBlockBuf);
	SetOffset( uEntryOffset, m_pNextEntryPtr );
	SetLength( uiEntryLength, m_pNextEntryPtr );
	m_pNextEntryPtr += (BLKOFFSET_SIZE + LENGTH_SIZE);

	m_uiLengthRemaining -= (uiAlignLength + BLKOFFSET_SIZE + LENGTH_SIZE );
	m_BlockHeader.uiEntryCount++;

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Add an fixed length entry to the result set.  
****************************************************************************/

RCODE FResultSetBlk::AddEntry(
	FLMBYTE *		pEntry)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiEntrySize = m_uiEntrySize;	// Optimization

	// Check that setup was called for fixed length entries.

	flmAssert( m_bFixedEntrySize );

	// Check to see if the current buffer is full. 

	if( m_uiLengthRemaining < uiEntrySize )
	{
		// Caller should call Flush and setup correctly what to do next.

		rc = RC_SET( FERR_EOF_HIT );
		goto Exit;
	}

	f_memcpy( m_pNextEntryPtr, pEntry, uiEntrySize );
	m_BlockHeader.uiEntryCount++;
	m_pNextEntryPtr		+= uiEntrySize;
	m_pEndPoint				+= uiEntrySize;
	m_uiLengthRemaining	-= uiEntrySize;

Exit:
	return( rc);
}


/****************************************************************************
Public:	ModifyEntry
Desc:		Modify the current entry being references.
Ret:     RCODE - FERR_OK always.  Will assert on entry size mismatch.
Notes:	The size of each block cannot be modified.  This is to allow
			writing to the same location on disk and not waste disk memory.
****************************************************************************/

RCODE FResultSetBlk::ModifyEntry(
	FLMBYTE *		pEntry,
	FLMUINT			uiEntryLength)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pCurEntry;

	F_UNREFERENCED_PARM( uiEntryLength);

	flmAssert( NULL != m_pBlockBuf );

	// The incoming entry MUST be the same size.

	if( m_bFixedEntrySize )
	{
		// Assert that the entry length must be zero.
		// If not - still use m_uiEntrySize;

		flmAssert( 0 == uiEntryLength );

		// Position to the current item.

		pCurEntry = &m_pBlockBuf[ m_iEntryPos * m_uiEntrySize ];
		f_memcpy( pCurEntry, pEntry, m_uiEntrySize );		
	}
	else
	{
		// Variable Length
		FLMUINT		uiCurEntryLength; // uiAlignLength;

		pCurEntry = &m_pBlockBuf[ m_iEntryPos * m_uiEntrySize ];
		uiCurEntryLength = GetLength( pCurEntry );

		// We cannot support changing the entry size at this time.

		flmAssert( uiEntryLength == uiCurEntryLength );

		pCurEntry = m_pBlockBuf + GetOffset( pCurEntry );

		f_memcpy( pCurEntry, pEntry, uiCurEntryLength );
	}

	m_bModifiedEntry = TRUE;
//Exit:
	return( rc);
}


/****************************************************************************
Public:	Flush
Desc:		The block is full and need to flush the block to disk.  If 
			bForceWrite is FALSE then will not write block to disk.
Ret:     RCODE
Notes:	
****************************************************************************/

RCODE FResultSetBlk::Flush(
	FLMBOOL		bLastBlockInList,		// Last block in a block list.
	FLMBOOL		bForceWrite)			// if TRUE write out to disk.
{
	RCODE			rc = FERR_OK;

	flmAssert( NULL != m_pBlockBuf );// Insure SetBuffer was called.

	// Set the Block header information, sort, kill dups and flush
	
	SqueezeSpace();						// Squeeze out wasted space.

	if( !m_bEntriesInOrder)				// Are entries NOT in order?
	{
		rc = SortAndRemoveDups();		// Remove duplicate entries
		if( RC_BAD(rc))
			goto Exit;
	}
	m_bEntriesInOrder = TRUE;			// Entries are now in order.

	m_BlockHeader.bLastBlock = bLastBlockInList;	

	if( bForceWrite)
	{
		if( RC_BAD( rc = Write()))
			goto Exit;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		If there is length remaining, squeeze out additional space.
****************************************************************************/
void FResultSetBlk::SqueezeSpace()
{
	// Fixed Entry Size?
	
	if(  m_bFixedEntrySize)				
	{
		// Yes, no need to squeeze out any space.
		goto Exit;
	}

	// Is there room to shift things down? 
	// Don't do if no entries or if less than 64 bytes or no entries.

	if( (m_uiLengthRemaining >= 64 ) && m_BlockHeader.uiEntryCount )
	{
		FLMUINT		uiBytesToMoveUp;
		FLMBYTE *	pEntry;

		uiBytesToMoveUp = m_uiLengthRemaining;
		m_uiLengthRemaining = 0;

		// Overlapping memory move call.

		flmAssert( (m_pBlockBuf + m_BlockHeader.uiBlockSize) > m_pEndPoint );
		flmAssert( uiBytesToMoveUp < m_BlockHeader.uiBlockSize );

		f_memmove( m_pEndPoint - uiBytesToMoveUp, m_pEndPoint,
			(FLMUINT) ((m_pBlockBuf + m_BlockHeader.uiBlockSize ) - m_pEndPoint ));

		m_BlockHeader.uiBlockSize -= uiBytesToMoveUp;
		m_pEndPoint -= uiBytesToMoveUp;

		// Change all of the offsets for every entry.  This is expensive.

		for( pEntry = m_pBlockBuf;
			  pEntry < m_pNextEntryPtr; 
			  pEntry += BLKOFFSET_SIZE + LENGTH_SIZE )
		{
			BLKOFFSET		uEntryOffset;

			uEntryOffset = GetOffset( pEntry ) - ((BLKOFFSET) uiBytesToMoveUp);
			SetOffset( uEntryOffset, pEntry );
	  }
	}
Exit:
	return;
}

/****************************************************************************
Desc:		Sort the current block and remove all duplicates.
****************************************************************************/
RCODE FResultSetBlk::SortAndRemoveDups()
{
	RCODE			rc = FERR_OK;
	
	// Nothing to do if one or zero entries in the block.
	
	if( (m_BlockHeader.uiEntryCount <= 1 ) || (! m_fnCompare ))
	{
		goto Exit;
	}

	m_bDuplicateFound = FALSE;
	if( RC_BAD( rc = QuickSort( 0, m_BlockHeader.uiEntryCount - 1 )))
		goto Exit;

	/*
	Some users of result sets may not have any duplicates to remove
	or may want the side effect of having duplicates to further
	process the entries like for sorting tracker records.  It is up
	to the compare routine to never return RS_EQUALS in this case.
	*/
	/*
	This algorithm is tuned for the case where there are zero or few 
	duplicate records.  Removing duplicates is expensive in this design.
	*/

	if( m_bDropDuplicates && m_bDuplicateFound )
	{
		FLMUINT		uiEntriesRemaining;
		FLMBYTE *	pBase;
		FLMBYTE *	pEntry;
		FLMBYTE *	pNextEntry;
		FLMINT		iCompare;

		pBase = pEntry = m_pBlockBuf;
		
		for( uiEntriesRemaining = m_BlockHeader.uiEntryCount - 1
			; uiEntriesRemaining > 0
			; uiEntriesRemaining-- )
		{
			pNextEntry = pEntry + m_uiEntrySize;
		
			if( m_bFixedEntrySize)
			{
				rc = m_fnCompare( pEntry, m_uiEntrySize, 
									  pNextEntry, m_uiEntrySize,
									  m_UserValue, &iCompare );
			}
			else
			{
				rc = m_fnCompare( pBase + GetOffset( pEntry ), 
									  GetLength( pEntry ),
									  pBase + GetOffset( pNextEntry), 
									  GetLength( pNextEntry ),
									  m_UserValue, &iCompare );
			}
			if( RC_BAD(rc))
				goto Exit;

			if( iCompare == RS_EQUALS )
			{
				RemoveEntry( pEntry );
				
				// Leave pEntry alone - everyone will scoot down
			}
			else
			{
				pEntry += m_uiEntrySize;
			}
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Remove the current entry from the block.
****************************************************************************/
void FResultSetBlk::RemoveEntry(
	FLMBYTE *		pEntry)
{
	if( m_bFixedEntrySize )
	{
		// Don't like moving zero bytes - check first.

		if( pEntry + m_uiEntrySize < m_pEndPoint )
		{
			// This is really easy - just memmove everyone down.

			f_memmove( pEntry, pEntry + m_uiEntrySize, 
						(FLMUINT) (m_pEndPoint - pEntry ) - m_uiEntrySize);
		}
		m_BlockHeader.uiEntryCount--;
		m_BlockHeader.uiBlockSize -= m_uiEntrySize;
		m_pEndPoint -= m_uiEntrySize;
	}
	else // Variable length - much harder 
	{
		/*
		Example - remove entry  3 below...

		[entryOfs1:len][entryOfs2:len][entryOfs3:len][entryOfs4:len]
		[entryData1][entryData2][entryData3][entryData4]

		Need to reduce EntryOfs1 and entryOfs2 by m_uiEntrySize+entryLen3.
		because these entries are stored AFTER entry 3 - entries are first
		stored going from the back of the block to the front of the block.
		Need to reduce Ofs4 by OFFSET_SIZE.
		*/
		BLKOFFSET	uDeletedOffset = GetOffset( pEntry );
		BLKOFFSET	uTempOffset;
		FLMUINT		uiDeletedLength = GetLength( pEntry );
		FLMBYTE *	pCurEntry;
		FLMUINT		uiPos;
		FLMUINT		uiMoveBytes;

		flmAssert( m_BlockHeader.uiBlockSize >=
						(uDeletedOffset + uiDeletedLength ));

		// Don't move zero bytes - don't know what will happen x-platform.
		uiMoveBytes = (FLMUINT) 
			( m_BlockHeader.uiBlockSize - (uDeletedOffset + uiDeletedLength ));
		
		if( uiMoveBytes)
		{
			// First move down the variable length entry data.
			f_memmove( m_pBlockBuf + uDeletedOffset,
						  m_pBlockBuf + uDeletedOffset + uiDeletedLength,
						  uiMoveBytes );
		}
		flmAssert( m_BlockHeader.uiBlockSize >= (FLMUINT) 
							((pEntry + m_uiEntrySize) - m_pBlockBuf) );
		
		uiMoveBytes = (FLMUINT) (m_BlockHeader.uiBlockSize - 
							((pEntry + m_uiEntrySize) - m_pBlockBuf));
		if( uiMoveBytes )
		{
			f_memmove( pEntry, pEntry + m_uiEntrySize, uiMoveBytes );
		}
		m_BlockHeader.uiBlockSize -= uiDeletedLength + m_uiEntrySize;

		// Adjust the offset values.

		m_BlockHeader.uiEntryCount--;
		for( uiPos = 0, pCurEntry = m_pBlockBuf
			; uiPos < m_BlockHeader.uiEntryCount
			; uiPos++, pCurEntry += m_uiEntrySize )
		{
			// Assume that the offsets are NOT in descending order.	
			// This will help in the future additional adding and deleting
			// to an existing result set.

			uTempOffset = GetOffset( pCurEntry );
			if( uTempOffset > uDeletedOffset)
			{
				uTempOffset -= (BLKOFFSET) uiDeletedLength;
			}
			uTempOffset -= (BLKOFFSET)m_uiEntrySize;
			SetOffset( uTempOffset, pCurEntry );
		}
	}
	return;
}

/****************************************************************************
Desc:		Quick sort an array of values.
Notes:	Optimized the above quicksort algorithm.  On page 559 the book
			suggests that "The worst case can sometimes be avioded by choosing
			more carefully the record for final placement at each state."
			This algorithm picks a mid point for the compare value.  Doing
			this helps the worst case where the entries are in order.  In Order
			tests went from 101 seconds down to 6 seconds!
			This helps the 'in order' sorts from worst case Order(N^^2)/2 with 
			the normal quickSort to Order(NLog2 N) for the worst case.
			Also optimized the number of recursions to Log2 N from (N-2).
			Will recurse the SMALLER side and will iterate to the top of
			the routinefor the LARGER side.  Follow comments below.
****************************************************************************/
RCODE FResultSetBlk::QuickSort(
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	FLMBYTE *		pEntryTbl = m_pBlockBuf;
	FLMBYTE *		pCurEntry;
	FLMUINT			uiLBPos, uiUBPos, uiMIDPos;
	FLMUINT			uiLeftItems, uiRightItems;
	FLMINT			iCompare;
	FLMUINT			uiEntrySize = m_uiEntrySize;
	RCODE				rc = FERR_OK;	// Set in case only one entry.
	FLMBYTE			ucaSwapBuffer[MAX_FIXED_ENTRY_SIZE];
#define RECURSIVE_CALL
	// The problem with using the non-recursive code is that
	// for formula to compute the maximum number of levels
	// is so complex we cannot guess on a value.
#ifndef RECURSIVE_CALL
	FLMUINT			uiaLowerStack[64];
	FLMUINT			uiaUpperStack[64];
	FLMUINT			uiStackPos;
#endif

#define	RS_SWAP(pTbl,pos1,pos2)	{ \
	f_memcpy( ucaSwapBuffer, &pTbl[pos2*uiEntrySize], uiEntrySize); \
	f_memcpy( &pTbl[ pos2 * uiEntrySize ], &pTbl[ pos1 * uiEntrySize ], uiEntrySize ); \
	f_memcpy( &pTbl[ pos1 * uiEntrySize ], ucaSwapBuffer, uiEntrySize ); }

#ifndef RECURSIVE_CALL
	// Setup the stack.
	uiStackPos = 1;
	uiaLowerStack[0] = uiLowerBounds;
	uiaUpperStack[0] = uiUpperBounds;

	while( uiStackPos )
	{
		uiStackPos--;
		uiLowerBounds = uiaLowerStack[ uiStackPos ];
		uiUpperBounds = uiaUpperStack[ uiStackPos ];
#endif

Iterate_Larger_Half:

		uiUBPos = uiUpperBounds;
		uiLBPos = uiLowerBounds;
		uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
		pCurEntry = &pEntryTbl[ uiMIDPos * uiEntrySize ];
		for( ;;)
		{
			while( (uiLBPos == uiMIDPos)				// Don't compare with target
				||  (((rc = EntryCompare( &pEntryTbl[ uiLBPos * uiEntrySize ],
													pCurEntry,
													&iCompare )) == FERR_OK)
					 && (iCompare == RS_LESS_THAN )))
			{
				if( uiLBPos >= uiUpperBounds) break;
				uiLBPos++;
			}
			if(RC_BAD(rc))						// Check for error
				goto Exit;

			while( (uiUBPos == uiMIDPos)				// Don't compare with target
				||  (((rc = EntryCompare( pCurEntry,
													&pEntryTbl[ uiUBPos * uiEntrySize ],
													&iCompare )) == FERR_OK)
					 && (iCompare == RS_LESS_THAN )))
			{
				if( uiUBPos == 0)				// Check for underflow condition
				{
					break;
				}
				uiUBPos--;
			}
			if(RC_BAD(rc))						// Check for error
				goto Exit;
			
			if( uiLBPos < uiUBPos )			// Interchange and continue loop.
			{
				/* Interchange [uiLBPos] with [uiUBPos]. */

				RS_SWAP( pEntryTbl, uiLBPos, uiUBPos );
				uiLBPos++;						// Scan from left to right.
				uiUBPos--;						// Scan from right to left.
			}
			else									// Past each other - done
			{
				break;
			}
		}
		/* 5 cases to check.
			1) UB < MID < LB - Don't need to do anything.
			2) MID < UB < LB - swap( UB, MID )
			3) UB < LB < MID - swap( LB, MID )
			4) UB = LB < MID - swap( LB, MID ) - At first position
			5) MID < UB = LB - swap( UB, MID ) - At last position
		*/

		/* Check for swap( LB, MID ) - cases 3 and 4 */

		if( uiLBPos < uiMIDPos )
		{
			/* Interchange [uiLBPos] with [uiMIDPos] */

			RS_SWAP( pEntryTbl, uiMIDPos, uiLBPos );
			uiMIDPos = uiLBPos;
		}
		else if( uiMIDPos < uiUBPos )		/* cases 2 and 5 */
		{
			/* Interchange [uUBPos] with [uiMIDPos] */

			RS_SWAP( pEntryTbl, uiMIDPos, uiUBPos );
			uiMIDPos = uiUBPos;
		}

		/* To save stack space - recurse the SMALLER Piece.  For the larger
			piece goto the top of the routine.  Worst case will be
			(Log2 N)  levels of recursion.

			Don't recurse in the following cases:
			1) We are at an end.  Just loop to the top.
			2) There are two on one side.  Compare and swap.  Loop to the top.
				Don't swap if the values are equal.  There are many recursions
				with one or two entries.  This doesn't speed up any so it is
				commented out.
		*/
		/* Check the left piece. */

		uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
								? uiMIDPos - uiLowerBounds		// 2 or more
								: 0;
		uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
								? uiUpperBounds - uiMIDPos 		// 2 or more
								: 0;
		/*
		A removed optimization was here.
		If two left or right items then check and swap.
		I didn't see any improvement in time using this code.
		*/

		if( uiLeftItems < uiRightItems )
		{
			/* Recurse on the LEFT side and goto the top on the RIGHT side. */

			if( uiLeftItems )
			{
#ifdef RECURSIVE_CALL
				// Recursive call.
				if( RC_BAD( rc = QuickSort( uiLowerBounds, uiMIDPos - 1 )))
					goto Exit;
#else
				uiaLowerStack[ uiStackPos   ] = uiLowerBounds;
				uiaUpperStack[ uiStackPos++ ] = uiMIDPos - 1;
#endif
			}
			uiLowerBounds = uiMIDPos + 1;
			goto Iterate_Larger_Half;
		}
		else if( uiLeftItems )	// Compute a truth table to figure out this check.
		{
			/* Recurse on the RIGHT side and goto the top for the LEFT side. */

			if( uiRightItems )
			{
				// Recursive call.

#ifdef RECURSIVE_CALL
				if( RC_BAD( rc = QuickSort( uiMIDPos + 1, uiUpperBounds )))
					goto Exit;
#else
				uiaLowerStack[ uiStackPos   ] = uiMIDPos + 1;
				uiaUpperStack[ uiStackPos++ ] = uiUpperBounds;
#endif
			}
			uiUpperBounds = uiMIDPos - 1;
			goto Iterate_Larger_Half;
		}
#ifndef RECURSIVE_CALL
	}
#endif
Exit:	
	return( rc);
}

/****************************************************************************
Public:	FRSDefaultCompare
Desc:		Default compare that compares the data like memcmp() from left
			to right and then by length where smaller is less than bigger.
Ret:     *piCompare = -1 RS_LESS_THAN; 0 = RS_EQUALS; 1 RS_GREATER_THAN 
			RCODE = any code from user defined compare routine.
			non-zero rcode will terminate sort/compare.
Notes:	Do NOT make this routine static.  This routine needs to be public.
****************************************************************************/
RCODE FRSDefaultCompare(
	void *		pData1,
	FLMUINT		uiLength1,
	void *		pData2,
	FLMUINT		uiLength2,
	void *		UserValue,
	FLMINT *		piCompare)
{
	FLMUINT		uiMinLength = f_min( uiLength1, uiLength2 );
	FLMINT		iCompareValue;
	int			siMemcmpValue;

	F_UNREFERENCED_PARM( UserValue);

	if( (siMemcmpValue = f_memcmp( pData1, pData2, (FLMUINT) uiMinLength )) == 0)
	{
		// Both equal up to the minimum length.  Is there additional data? 

		if( uiLength1 != uiLength2 )
		{
			iCompareValue = (uiLength1 < uiLength2) 
				? RS_LESS_THAN : RS_GREATER_THAN;
		}
		else
		{
			iCompareValue = RS_EQUALS;
		}
	}
	else 
	{
		iCompareValue = (siMemcmpValue < 0) ? RS_LESS_THAN : RS_GREATER_THAN;
	}
	*piCompare = iCompareValue;
	return FERR_OK;
}

/****************************************************************************
Desc:		Write this block to disk.  Adjust variables.
****************************************************************************/
RCODE FResultSetBlk::Write()
{
	FLMUINT			uiTotalBytesWritten;
	FLMUINT64		ui64FilePos;
	FLMUINT			uiBytesToWrite,
						uiBytesWritten;
	RCODE				rc;

	// By this time there better be something to write...

	// The file should be opened by default.

	if( m_BlockHeader.ui64FilePos == RSBLK_UNSET_FILE_POS )
	{
		if( RC_BAD(rc = (*m_ppFileHdl64)->Size( &m_BlockHeader.ui64FilePos )))
			goto Exit;
	}

	// Write out the block header definition.

	rc = (*m_ppFileHdl64)->Write(
						 m_BlockHeader.ui64FilePos,
						 sizeof( FBlockHeader), &m_BlockHeader, 
						 &uiBytesWritten );
	if( RC_BAD( rc))
		goto Exit;

	ui64FilePos = m_BlockHeader.ui64FilePos + 
						(FLMUINT64)sizeof( FBlockHeader);

	for( uiTotalBytesWritten = 0
		; uiTotalBytesWritten < m_BlockHeader.uiBlockSize
		; uiTotalBytesWritten += uiBytesWritten )
	{
		if( uiTotalBytesWritten + (MAX_WRITE_BYTES) > m_BlockHeader.uiBlockSize)
		{
			uiBytesToWrite = m_BlockHeader.uiBlockSize - uiTotalBytesWritten;
		}
		else
		{
			uiBytesToWrite = MAX_WRITE_BYTES;
			// Write at most 60K at a time.
		}
		rc = (*m_ppFileHdl64)->Write(
						 ui64FilePos,
						 uiBytesToWrite,
						 &m_pBlockBuf[ uiTotalBytesWritten ],
						 &uiBytesWritten );
		if( RC_BAD( rc))
			goto Exit;
		ui64FilePos += uiBytesWritten;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Read in the specified block into memory.
Ret:     FERR_OK, FERR_EOF_HIT or I/O error
****************************************************************************/
RCODE FResultSetBlk::Read()
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesRead, uiBytesToRead;
	FLMUINT		uiTotalBytesRead;
	FLMUINT64	ui64FilePos = m_BlockHeader.ui64FilePos;
	FBlockHeader BlockHeader;

	// Nothing to do???

	if( RSBLK_UNSET_FILE_POS == ui64FilePos)
		goto Exit;

	// First read the block header in.

	rc = (*m_ppFileHdl64)->Read( ui64FilePos, sizeof( FBlockHeader ),
						(void *) &BlockHeader, &uiBytesRead );
	if( RC_BAD(rc)) 
		goto Exit;

	// Verify that the block header data is the same.
	// This is the best we can do to verify that the file handle
	// is not junky.

	if( (BlockHeader.ui64FilePos != m_BlockHeader.ui64FilePos )
	 || (BlockHeader.uiEntryCount != m_BlockHeader.uiEntryCount ))
	{
		// Returning data error because there is a big possibility
		// that the file handles have become corrupt.
		rc = RC_SET( FERR_FAILURE );
		goto Exit;
	}

	ui64FilePos += sizeof( FBlockHeader );

	for( uiTotalBytesRead = 0
		; uiTotalBytesRead < m_BlockHeader.uiBlockSize
		; uiTotalBytesRead += uiBytesRead )
	{
		if( uiTotalBytesRead + (MAX_WRITE_BYTES) > m_BlockHeader.uiBlockSize)
		{
			uiBytesToRead = m_BlockHeader.uiBlockSize - uiTotalBytesRead;
		}
		else
		{
			uiBytesToRead = MAX_WRITE_BYTES;
		}
		rc = (*m_ppFileHdl64)->Read(
							ui64FilePos,
							uiBytesToRead,
							&m_pBlockBuf[ uiTotalBytesRead ],
							&uiBytesRead );
		if( RC_BAD(rc)) 
			goto Exit;
		ui64FilePos += (FLMUINT64)uiBytesRead;
	}

Exit:
	if( RC_OK(rc))
	{
		m_bPositioned = TRUE;
		m_iEntryPos = -1;
	}
	return( rc);
}

/****************************************************************************
Desc:		Copies the current entry into the user buffer.  Checks for overflow.
****************************************************************************/
RCODE FResultSetBlk::CopyCurrentEntry(
	FLMBYTE *	pBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc = FERR_OK;	// Must be initailized
	FLMUINT		uiEntrySize;
	FLMBYTE *	pEntry;

	flmAssert( NULL != pBuffer );

	// Copy the current entry.  This is a shared routine
	// because the code to copy an entry is a little complicated.

	uiEntrySize = m_uiEntrySize;
	pEntry = &m_pBlockBuf[ m_iEntryPos * uiEntrySize ];
	
	if( !m_bFixedEntrySize )
	{
		uiEntrySize = GetLength( pEntry );
		pEntry = m_pBlockBuf + GetOffset( pEntry );
	}
	
	if( uiBufferLength && (uiEntrySize > uiBufferLength))
	{
		uiEntrySize = uiBufferLength;
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW );
		// Fall through into memcpy.
	}
	f_memcpy( pBuffer, pEntry, (FLMUINT) uiEntrySize );
	if( puiReturnLength)
	{
		*puiReturnLength = uiEntrySize;
	}
//Exit:
	return( rc);
}

/****************************************************************************
Public:	GetCurrent
Desc:		Return the Current entry reference in the result set.
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_NOT_FOUND	- Not positioned anywhere in the result set.
			FERR_EOF_HIT		- Positioned past the last entry
			FERR_BOF_HIT		- Positioned before the first entry.
****************************************************************************/
RCODE FResultSetBlk::GetCurrent(
	FLMBYTE *	pBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc;

	flmAssert( NULL != m_pBlockBuf );
	if( !m_bPositioned )
	{
		rc = RC_SET( FERR_NOT_FOUND );
		goto Exit;
	}

	// Check for EOF and BOF conditions - otherwise return current.
	
	if( m_iEntryPos >= (FLMINT) m_BlockHeader.uiEntryCount )
	{
		rc = RC_SET( FERR_EOF_HIT );
		goto Exit;
	}
	if( m_iEntryPos == -1)
	{
		rc = RC_SET( FERR_BOF_HIT );
		goto Exit;
	}
	rc = CopyCurrentEntry( pBuffer, uiBufferLength, puiReturnLength );
	
Exit:
	return( rc);
}

/****************************************************************************
Public:	GetNextPtr
Desc:		Return a pointer to the next reference in the result set.  
			If the result set is not positioned then the first entry will 
			be returned.
			
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_NOT_FOUND	- zero records in the result set.
			FERR_EOF_HIT		- Positioned past the last entry
****************************************************************************/
RCODE FResultSetBlk::GetNextPtr(
	FLMBYTE **		ppBuffer,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiEntrySize;
	FLMBYTE *		pEntry;

	flmAssert( ppBuffer != NULL );
	flmAssert( puiReturnLength != NULL );
	flmAssert( m_bPositioned );

	// Are we on the last entry or past the last entry?

	if( m_iEntryPos + 1 >= (FLMINT) m_BlockHeader.uiEntryCount )
	{
		m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
		rc = RC_SET( FERR_EOF_HIT );
		goto Exit;
	}
	m_iEntryPos++;				// Else position to next entry

	uiEntrySize = m_uiEntrySize;
	pEntry = &m_pBlockBuf[ m_iEntryPos * uiEntrySize ];
	
	if( !m_bFixedEntrySize )
	{
		uiEntrySize = GetLength( pEntry );
		pEntry = m_pBlockBuf + GetOffset( pEntry );
	}
	
	*ppBuffer = pEntry;
	*puiReturnLength = uiEntrySize;

Exit:
	return( rc);
}


/****************************************************************************
Public:	GetPrev
Desc:		Return the previous reference in the result set.  If the result set
			is not positioned then the last entry will be returned.
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_BOF_HIT		- Positioned past the first entry
****************************************************************************/
RCODE FResultSetBlk::GetPrev(
	FLMBYTE *		pBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bPositioned );

	// If not positioned then position past last entry.
	if( m_iEntryPos == -1)
	{
		m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
	}			
	// Are we on the first entry or before the first entry?

	if( m_iEntryPos == 0)
	{
		m_iEntryPos = -1;				// Just set it
		rc = RC_SET( FERR_BOF_HIT );
		goto Exit;
	}

	m_iEntryPos--;				// position to previous entry.

	rc = CopyCurrentEntry( pBuffer, uiBufferLength, puiReturnLength );

Exit:
	return( rc);
}

/****************************************************************************
Public:	SetPosition
Desc:		Set the current entry position for this block.
In:		uiPosition - Zero based position value for this block
			or RS_POSITION_NOT_SET if invalid position (past the end).
Ret:		FERR_OK			- Returned OK
			FERR_EOF_HIT		- Positioned past the end of the block.
Note:		The state must be that the data in the block has been
			read in.  Otherwise, will set position to first or last.
****************************************************************************/

RCODE FResultSetBlk::SetPosition(
	FLMUINT			uiPosition)
{
	RCODE				rc= FERR_OK;

	// Buffer must be set or SetBuffer() will set iEntryPos back to -1.
	flmAssert( m_bPositioned );	

	if( uiPosition == RS_POSITION_NOT_SET )
	{
		m_iEntryPos = -1;
		goto Exit;
	}
	flmAssert( uiPosition >= m_uiBlkEntryPosition );

	// Convert to a zero based number relative to this block.

	if( uiPosition >= m_uiBlkEntryPosition )
		uiPosition -= m_uiBlkEntryPosition;
	else
		uiPosition = 0;			// Handle if assert condition is not in debug.

	if( uiPosition >= m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( FERR_EOF_HIT );
		m_iEntryPos = m_BlockHeader.uiEntryCount;
	}
	else
	{
		m_iEntryPos = (FLMINT) uiPosition;
	}

Exit:
	return( rc);
}

/****************************************************************************
Public:	FindMatch
Desc:		Find the matching entry within the block using the compare routine.
			This does a binary search on entries.  Watch the (out) variable.
Out:		*piCompare = RS_EQUALS - match found OR entry would compare between
											  the low and high entries in the block.
							= RS_GREATER_THAN - match entry is greater than
												the highest entry in the block.
							= RS_LESS_THAN - match entry is less than the
												lowest entry in the block.
Ret:		FERR_OK			- Returned OK
			FERR_NOT_FOUND	- match not found
Notes:	One side effect is that m_iEntryPos is set to the matched
			entry or some random entry if not found is returned.
****************************************************************************/

RCODE FResultSetBlk::FindMatch(		// Find and return an etnry that 
												// matches in this block.
	FLMBYTE *	pMatchEntry,			// Entry to match
	FLMUINT		uiMatchEntryLength,	// Variable length of above entry 
	FLMBYTE *	pFoundEntry,			// (out) Entry to return
	FLMUINT *	puiFoundEntryLength,	// (out) Length of entry returned
	RSET_COMPARE_FUNC_p 					// Record compare function.
					fnCompare,				// Returns (FLMINT) -1, 0 or 1 values.
	void *		UserValue,				// UserValue for callback.
	FLMINT	*	piCompare)				// See comments above.
{
	RCODE			rc = FERR_OK;
	FLMINT		iCompare;				// Return from CompareEntry
	FLMUINT		uiLow, uiHigh, uiMid, uiLimit;

	uiLow = 0;
	uiHigh = uiLimit = GetNumberOfEntries() - 1;
	if( ! uiMatchEntryLength)			// Set the match entry length.
		uiMatchEntryLength = m_uiEntrySize;
	
	// Check the first and last entries in the block.
	// Copy the current entry if found. 

	if( RC_BAD( rc = CompareEntry( pMatchEntry, uiMatchEntryLength, uiLow,
								fnCompare, UserValue, &iCompare )))
		goto Exit;

	if( iCompare != RS_GREATER_THAN )
	{
		if( iCompare == RS_LESS_THAN )
		{
			rc = RC_SET( FERR_NOT_FOUND );
		}

		*piCompare = iCompare;
		goto Exit;
	}
	if( RC_BAD( rc =  CompareEntry( pMatchEntry, uiMatchEntryLength, uiHigh,
											fnCompare, UserValue, &iCompare )))
		goto Exit;

	if( iCompare != RS_LESS_THAN )
	{
		if( iCompare == RS_GREATER_THAN )
		{
			rc = RC_SET( FERR_NOT_FOUND );
		}
		*piCompare = iCompare;
		goto Exit;
	}

	// Set the iCompare to equals because
	// the match entry sorts within the block somewhere.  
	// Binary search the entries in the block.  May still
	// not find the matching entry.

	*piCompare = RS_EQUALS;

	for( ;; )									// Initialize low and high above.
	{
		uiMid = (uiLow + uiHigh) >> 1;	// (uiLow + uiHigh) / 2

		if( RC_BAD( rc = CompareEntry( pMatchEntry, uiMatchEntryLength, uiMid,
											fnCompare, UserValue, &iCompare )))
			goto Exit;
		if( iCompare == RS_EQUALS )
		{
			// Found Match!  All set up to return.
			goto Exit;
		}
		// Check if we are done - where wLow equals uHigh.

		if( uiLow >= uiHigh)
			break;								// Done - item not found.

		if( iCompare == RS_LESS_THAN )
		{
			if( uiMid == 0)					// Way too high?
				break;	
			uiHigh = uiMid - 1;				// Too high
		}
		else
		{
			if( uiMid == uiLimit)			// Done - Hit the top
				break;	
			uiLow = uiMid + 1;				// Too low
		}
	}
	// On break set we did not find the matching entry.

	rc = RC_SET( FERR_NOT_FOUND );

Exit:
	if( RC_OK(rc))
	{
		rc = CopyCurrentEntry( pFoundEntry, 0, puiFoundEntryLength );
	}
	return( rc);
}


/****************************************************************************
Desc:		Compare the buffer entry with entry identifies by 
			uiEntryPos.
Out:		*piCompare = RS_EQUALS - match found OR entry value would compare
											  between the low and high entries.
							= RS_GREATER_THAN - match entry is greater than
												the highest entry in the block.
							= RS_LESS_THAN - match entry is less than the
												lowest entry in the block.
****************************************************************************/

RCODE FResultSetBlk::CompareEntry(	// Compares match entry with entry
												// identified by uiEntryPos.
	FLMBYTE *	pMatchEntry,			// Entry to match
	FLMUINT		uiMatchEntryLength,	// Variable length of pMatchEntry.
	FLMUINT		uiEntryPos,				// Position of entry in block.
	RSET_COMPARE_FUNC_p 					// Record compare function.
					fnCompare,				// Returns (FLMINT) -1, 0 or 1 values.
	void *		UserValue,				// UserValue for callback.
	FLMINT *		piCompare)				// Return from compare.
{
	RCODE			rc;
	FLMBYTE *	pEntry;
	FLMUINT		uiEntrySize;

	F_UNREFERENCED_PARM( fnCompare);
	F_UNREFERENCED_PARM( UserValue);

	// Position to the entry.

	m_iEntryPos = (FLMINT) uiEntryPos;
	uiEntrySize = m_uiEntrySize;
	pEntry = &m_pBlockBuf[ m_iEntryPos * uiEntrySize ];
	
	if( !m_bFixedEntrySize )
	{
		uiEntrySize = GetLength( pEntry );
		pEntry = m_pBlockBuf + GetOffset( pEntry );
	}

	rc = m_fnCompare( pMatchEntry, uiMatchEntryLength,
							pEntry, uiEntrySize,
							m_UserValue, piCompare );

	return( rc);
}
