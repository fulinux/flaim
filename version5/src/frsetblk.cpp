//------------------------------------------------------------------------------
// Desc: Result set block routines
//
// Tabs:	3
//
//		Copyright (c) 1996-2000, 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frsetblk.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"
#include "frset.h"

/*****************************************************************************
Desc:
******************************************************************************/
FResultSetBlk::FResultSetBlk()
{
	m_pNext = m_pPrev = NULL;
	m_pCompare = NULL;
	reset();
}

/*****************************************************************************
Desc:	Reset a block so it can be reused.
******************************************************************************/
void FResultSetBlk::reset( void)
{
	flmAssert( !m_pNext && !m_pPrev);

	// Initialize all of the member variables
	// between this constructor, SetBuffer() and Setup().

	m_BlockHeader.ui64FilePos = RSBLK_UNSET_FILE_POS;
	m_BlockHeader.uiEntryCount = 0;
	m_ppFileHdl64 = NULL;
	m_ui64BlkEntryPosition = RS_POSITION_NOT_SET;
	m_iEntryPos = 0;
	m_bDuplicateFound = FALSE;
	m_bPositioned = FALSE;
	m_bModifiedEntry = FALSE;
	m_pucBlockBuf = NULL;
}

/*****************************************************************************
Desc:
******************************************************************************/
void FResultSetBlk::Setup(
	F_64BitFileHandle **		ppFileHdl64,	// file handle to use for temp file.
	IF_ResultSetCompare *	pCompare,
	FLMUINT						uiEntrySize,
	FLMBOOL						bFirstInList,
	FLMBOOL						bDropDuplicates,	// If TRUE drop duplicates
	FLMBOOL						bEntriesInOrder)	// TRUE when entries are in order.
{
	flmAssert( ppFileHdl64);
	m_ppFileHdl64 = ppFileHdl64;

	if( m_pCompare)
	{
		m_pCompare->Release();
	}

	if( (m_pCompare = pCompare) != NULL)
	{
		m_pCompare->AddRef();
	}

	m_uiEntrySize = uiEntrySize;
	m_BlockHeader.bFirstBlock = bFirstInList;
	m_BlockHeader.bLastBlock = FALSE;
	m_bFixedEntrySize = m_uiEntrySize ? TRUE : FALSE;

	if( !m_uiEntrySize)
	{
		m_uiEntrySize = sizeof( F_VAR_HEADER);
	}

	m_bDropDuplicates = bDropDuplicates;
	m_bEntriesInOrder = bEntriesInOrder;
}

/*****************************************************************************
Desc:		The buffer is NOT allocated the by the result set block object.
			Setup the pucBuffer and associated variables.  Read in the data
			for this block if necessary.  If NULL is passed in as pucBuffer
			then this block is not the active block anymore.
Notes:	Must be called before other methods below are called.
*****************************************************************************/
RCODE		FResultSetBlk::SetBuffer(
	FLMBYTE *		pucBuffer,			// Working buffer or NULL
	FLMUINT			uiBufferLength)	// Default value is RSBLK_BLOCK_SIZE.
{
	RCODE				rc = NE_XFLM_OK;

	// If a buffer is defined then read in the data from disk.

	if( pucBuffer)
	{
		m_pucBlockBuf = pucBuffer;
		if( !m_BlockHeader.uiEntryCount)
		{
			// uiBlockSize is the final block size after squeeze.
			// uiLengthRemaining is working value of bytes available.

			m_BlockHeader.uiBlockSize = uiBufferLength;
			m_uiLengthRemaining = uiBufferLength;

			if( m_bFixedEntrySize)
			{
				m_pucEndPoint = m_pucBlockBuf;
			}
			else
			{
				m_pucEndPoint = m_pucBlockBuf + uiBufferLength;
			}
		}
		else
		{
			// Read in the data if necessary.

			if( RC_BAD( rc = Read()))
			{
				goto Exit;
			}
		}

		// The block is now in focus

		m_bPositioned = TRUE;
	}
	else
	{
		// Deactivating block so the buffer can be reused.
		// Check if the block has been modified

		if( m_bModifiedEntry)
		{
			// Is this a lone block?

			if( !m_BlockHeader.bLastBlock || !m_BlockHeader.bFirstBlock)
			{
				if( RC_BAD( rc = Write()))
				{
					goto Exit;
				}
			}
			m_bModifiedEntry = FALSE;
		}

		// The block is now out of focus

		m_bPositioned = FALSE;
		m_pucEndPoint = m_pucBlockBuf = NULL;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Add a variable length entry to the result set.  If fixed length
			entry then call AddEntry for fixed length entries.
*****************************************************************************/
RCODE FResultSetBlk::AddEntry(
	FLMBYTE *		pucEntry,
	FLMUINT			uiEntryLength)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiAlignLength;
	F_VAR_HEADER *	pEntry;

	flmAssert( m_pucBlockBuf);

	// Was setup called for fixed length entries?

	if( m_bFixedEntrySize )
	{
		rc = AddEntry( pucEntry );
		goto Exit;
	}

	uiAlignLength = (uiEntryLength + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN);

	// Check to see if the current buffer will overflow.

	if( m_uiLengthRemaining < uiAlignLength + sizeof( F_VAR_HEADER))
	{
		// Caller should call Flush and setup correctly what to do next.

		rc = RC_SET( NE_XFLM_EOF_HIT );
		goto Exit;
	}

	// Copy entry and compute the offset value for pNextEntryPtr.

	m_pucEndPoint -= uiAlignLength;
	f_memcpy( m_pucEndPoint, pucEntry, uiEntryLength );

	pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_BlockHeader.uiEntryCount;
	pEntry->ui32Offset = (FLMUINT32)(m_pucEndPoint - m_pucBlockBuf);
	pEntry->ui32Length = (FLMUINT32)uiEntryLength;

	m_uiLengthRemaining -= (uiAlignLength + sizeof( F_VAR_HEADER));
	m_BlockHeader.uiEntryCount++;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Add a fixed length entry to the result set.
*****************************************************************************/
RCODE FResultSetBlk::AddEntry(
	FLMBYTE *	pucEntry)
{
	RCODE		rc = NE_XFLM_OK;

	// Check that setup was called for fixed length entries.

	flmAssert( m_bFixedEntrySize);

	// Check to see if the current buffer is full.

	if( m_uiLengthRemaining < m_uiEntrySize)
	{
		// Caller should call Flush and setup correctly what to do next.

		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	f_memcpy( m_pucBlockBuf + (m_uiEntrySize * m_BlockHeader.uiEntryCount),
		pucEntry, m_uiEntrySize);
	m_BlockHeader.uiEntryCount++;
	m_pucEndPoint += m_uiEntrySize;
	m_uiLengthRemaining -= m_uiEntrySize;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Modify the current entry being references.
Notes:	The size of each block cannot be modified.  This is to allow
			writing to the same location on disk and not waste disk memory.
*****************************************************************************/
RCODE FResultSetBlk::ModifyEntry(
	FLMBYTE *	pucEntry,
	FLMUINT		uiEntryLength)
{
	RCODE	rc = NE_XFLM_OK;

	F_UNREFERENCED_PARM( uiEntryLength);

	flmAssert( m_pucBlockBuf);

	// The incoming entry MUST be the same size.

	if( m_bFixedEntrySize )
	{
		// Assert that the entry length must be zero.
		// If not - still use m_uiEntrySize;

		flmAssert( !uiEntryLength);

		// Copy over the current item.

		f_memcpy( &m_pucBlockBuf [m_iEntryPos * m_uiEntrySize],
						pucEntry, m_uiEntrySize );
	}
	else
	{
		// Variable Length

		F_VAR_HEADER *	pCurEntry;

		pCurEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;

		// We cannot support changing the entry size at this time.

		flmAssert( uiEntryLength == (FLMUINT)pCurEntry->ui32Length);

		f_memcpy( m_pucBlockBuf + pCurEntry->ui32Offset,
				pucEntry, uiEntryLength);
	}

	m_bModifiedEntry = TRUE;
	return( rc);
}

/*****************************************************************************
Desc:		The block is full and need to flush the block to disk.  If
			bForceWrite is FALSE then will not write block to disk.
*****************************************************************************/
RCODE FResultSetBlk::Flush(
	FLMBOOL		bLastBlockInList,		// Last block in a block list.
	FLMBOOL		bForceWrite)			// if TRUE write out to disk.
{
	RCODE	rc = NE_XFLM_OK;

	// Make sure SetBuffer was called

	flmAssert( m_pucBlockBuf);
	SqueezeSpace();

	if( !m_bEntriesInOrder)
	{
		// Remove duplicate entries.

		if( RC_BAD( rc = SortAndRemoveDups()))
		{
			goto Exit;
		}
	}

	m_bEntriesInOrder = TRUE;
	m_BlockHeader.bLastBlock = bLastBlockInList;

	if( bForceWrite)
	{
		if( RC_BAD( rc = Write()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	If there is length remaining, squeeze out additional space.
*****************************************************************************/
void FResultSetBlk::SqueezeSpace( void)
{
	FLMUINT	uiPos;

	// Fixed Entry Size?

	if( m_bFixedEntrySize)
	{
		// Yes, no need to squeeze out any space.

		goto Exit;
	}

	// Is there room to shift things down?
	// Don't do if no entries or if less than 64 bytes.

	if( m_uiLengthRemaining >= 64 && m_BlockHeader.uiEntryCount)
	{
		FLMUINT			uiBytesToMoveUp;
		F_VAR_HEADER *	pEntry;

		uiBytesToMoveUp = m_uiLengthRemaining;
		m_uiLengthRemaining = 0;

		// Overlapping memory move call.

		flmAssert( (m_pucBlockBuf + m_BlockHeader.uiBlockSize) > m_pucEndPoint );
		flmAssert( uiBytesToMoveUp < m_BlockHeader.uiBlockSize );

		f_memmove( m_pucEndPoint - uiBytesToMoveUp, m_pucEndPoint,
			(FLMUINT) ((m_pucBlockBuf + m_BlockHeader.uiBlockSize ) - m_pucEndPoint ));

		m_BlockHeader.uiBlockSize -= uiBytesToMoveUp;
		m_pucEndPoint -= uiBytesToMoveUp;

		// Change all of the offsets for every entry.  This is expensive.

		for( uiPos = 0, pEntry = (F_VAR_HEADER *)m_pucBlockBuf;
			  uiPos < m_BlockHeader.uiEntryCount;
			  pEntry++, uiPos++)
		{
			pEntry->ui32Offset -= (FLMUINT32)uiBytesToMoveUp;
		}
	}

Exit:

	return;
}

/*****************************************************************************
Desc:	Sort the current block and remove all duplicates.
*****************************************************************************/
RCODE FResultSetBlk::SortAndRemoveDups( void)
{
	RCODE	rc = NE_XFLM_OK;

	// Nothing to do if one or zero entries in the block.

	if( m_BlockHeader.uiEntryCount <= 1 || !m_pCompare)
	{
		goto Exit;
	}

	m_bDuplicateFound = FALSE;
	if( RC_BAD( rc = QuickSort( 0, m_BlockHeader.uiEntryCount - 1)))
	{
		goto Exit;
	}

	// Some users of result sets may not have any duplicates to remove
	// or may want the side effect of having duplicates to further
	// process the entries like for sorting tracker records.  It is up
	// to the compare routine to never return 0 in this case.

	// This algorithm is tuned for the case where there are zero or few
	// duplicate records.  Removing duplicates is expensive in this design.

	if( m_bDropDuplicates && m_bDuplicateFound)
	{
		FLMUINT	uiEntriesRemaining;
		FLMINT	iCompare;

		if( m_bFixedEntrySize)
		{
			FLMBYTE *	pucEntry;
			FLMBYTE *	pucNextEntry;

			pucEntry = m_pucBlockBuf;
			for( uiEntriesRemaining = m_BlockHeader.uiEntryCount - 1
				; uiEntriesRemaining > 0
				; uiEntriesRemaining-- )
			{
				pucNextEntry = pucEntry + m_uiEntrySize;

				if( RC_BAD( rc = m_pCompare->compare( pucEntry, m_uiEntrySize,
									  pucNextEntry, m_uiEntrySize,
									  &iCompare)))
				{
					goto Exit;
				}

				if( iCompare == 0)
				{
					RemoveEntry( pucEntry);

					// Leave pucEntry alone - everyone will scoot down
				}
				else
				{
					pucEntry += m_uiEntrySize;
				}
			}
		}
		else
		{
			F_VAR_HEADER *	pEntry = (F_VAR_HEADER *)m_pucBlockBuf;
			F_VAR_HEADER *	pNextEntry;

			for( uiEntriesRemaining = m_BlockHeader.uiEntryCount - 1
				; uiEntriesRemaining > 0
				; uiEntriesRemaining-- )
			{
				pNextEntry = pEntry + 1;

				if( RC_BAD( rc = m_pCompare->compare( m_pucBlockBuf + pEntry->ui32Offset,
									  (FLMUINT)pEntry->ui32Length,
									  m_pucBlockBuf + pNextEntry->ui32Offset,
									  (FLMUINT)pNextEntry->ui32Length,
									  &iCompare)))
				{
					goto Exit;
				}

				if( iCompare == 0)
				{
					RemoveEntry( (FLMBYTE *)pEntry);

					// Leave pEntry alone - everyone will scoot down
				}
				else
				{
					pEntry++;
				}
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Remove the current entry from the block.
*****************************************************************************/
void FResultSetBlk::RemoveEntry(
	FLMBYTE *	pucEntry)
{
	if( m_bFixedEntrySize)
	{
		// Don't like moving zero bytes - check first.

		if( pucEntry + m_uiEntrySize < m_pucEndPoint)
		{
			// This is really easy - just memmove everyone down.

			f_memmove( pucEntry, pucEntry + m_uiEntrySize,
						(FLMUINT)(m_pucEndPoint - pucEntry) - m_uiEntrySize);
		}

		m_BlockHeader.uiEntryCount--;
		m_BlockHeader.uiBlockSize -= m_uiEntrySize;
		m_pucEndPoint -= m_uiEntrySize;
	}
	else
	{
		// Variable length entries - much harder

		// Example - remove entry  3 below...

		// [entryOfs1:len][entryOfs2:len][entryOfs3:len][entryOfs4:len]
		// [entryData1][entryData2][entryData3][entryData4]

		// Need to reduce EntryOfs1 and entryOfs2 by m_uiEntrySize+entryLen3.
		// because these entries are stored AFTER entry 3 - entries are first
		// stored going from the back of the block to the front of the block.
		// Need to reduce Ofs4 by OFFSET_SIZE.

		F_VAR_HEADER *	pEntry = (F_VAR_HEADER *)pucEntry;
		FLMUINT			uiDeletedOffset = (FLMUINT)pEntry->ui32Offset;
		FLMUINT			uiTempOffset;
		FLMUINT			uiDeletedLength = (FLMUINT)pEntry->ui32Length;
		F_VAR_HEADER *	pCurEntry;
		FLMUINT			uiPos;
		FLMUINT			uiMoveBytes;

		flmAssert( m_BlockHeader.uiBlockSize >=
						(uiDeletedOffset + uiDeletedLength ));

		uiMoveBytes = (FLMUINT)
			(m_BlockHeader.uiBlockSize - (uiDeletedOffset + uiDeletedLength));

		if( uiMoveBytes)
		{

			// First move down the variable length entry data.

			f_memmove( m_pucBlockBuf + uiDeletedOffset,
						  m_pucBlockBuf + uiDeletedOffset + uiDeletedLength,
						  uiMoveBytes );
		}

		flmAssert( m_BlockHeader.uiBlockSize >=
							(FLMUINT)((FLMBYTE *)(&pEntry[1]) - m_pucBlockBuf) );

		uiMoveBytes = m_BlockHeader.uiBlockSize -
							(FLMUINT)((FLMBYTE *)(&pEntry [1]) - m_pucBlockBuf);

		if( uiMoveBytes)
		{
			f_memmove( pEntry, &pEntry[1], uiMoveBytes );
		}

		m_BlockHeader.uiBlockSize -= (uiDeletedLength + sizeof( F_VAR_HEADER));

		// Adjust the offset values.

		m_BlockHeader.uiEntryCount--;

		for( uiPos = 0, pCurEntry = (F_VAR_HEADER *)m_pucBlockBuf
			; uiPos < m_BlockHeader.uiEntryCount
			; uiPos++, pCurEntry++)
		{
			// Assume that the offsets are NOT in descending order.
			// This will help in the future additional adding and deleting
			// to an existing result set.

			uiTempOffset = (FLMUINT)pCurEntry->ui32Offset;
			if (uiTempOffset > uiDeletedOffset)
			{
				uiTempOffset -= uiDeletedLength;
			}
			uiTempOffset -= sizeof( F_VAR_HEADER);
			pCurEntry->ui32Offset = (FLMUINT32)uiTempOffset;
		}
	}
}

/*****************************************************************************
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
			the routine for the LARGER side.  Follow comments below.
*****************************************************************************/
RCODE FResultSetBlk::QuickSort(
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucEntryTbl = m_pucBlockBuf;
	FLMBYTE *		pucCurEntry;
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	FLMINT			iCompare;
	FLMUINT			uiEntrySize = m_uiEntrySize;
	FLMBYTE			ucaSwapBuffer[MAX_FIXED_ENTRY_SIZE];

#define	RS_SWAP(pTbl,pos1,pos2)	{ \
	f_memcpy( ucaSwapBuffer, &pTbl[pos2*uiEntrySize], uiEntrySize); \
	f_memcpy( &pTbl[ pos2 * uiEntrySize ], &pTbl[ pos1 * uiEntrySize ], uiEntrySize ); \
	f_memcpy( &pTbl[ pos1 * uiEntrySize ], ucaSwapBuffer, uiEntrySize ); }

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pucCurEntry = &pucEntryTbl[ uiMIDPos * uiEntrySize ];

	for (;;)
	{
		// Don't compare with target

		while( uiLBPos == uiMIDPos ||
				 (RC_OK( rc = EntryCompare( &pucEntryTbl[ uiLBPos * uiEntrySize],
												pucCurEntry,
												&iCompare)) &&
				  iCompare < 0))
		{
			if( uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		if( RC_BAD( rc))
		{
			goto Exit;
		}

		// Don't compare with target

		while( uiUBPos == uiMIDPos ||
				 (RC_OK( rc = EntryCompare( pucCurEntry,
												&pucEntryTbl[uiUBPos * uiEntrySize],
												&iCompare)) &&
				  iCompare < 0))
		{
			// Check for underflow

			if( !uiUBPos)
			{
				break;
			}

			uiUBPos--;
		}

		if (RC_BAD( rc))
		{
			goto Exit;
		}

		// Interchange and continue loop

		if( uiLBPos < uiUBPos)
		{
			// Interchange [uiLBPos] with [uiUBPos].

			RS_SWAP( pucEntryTbl, uiLBPos, uiUBPos );
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else
		{
			// Past each other - done

			break;
		}
	}

	// 5 cases to check.
	// 1) UB < MID < LB - Don't need to do anything.
	// 2) MID < UB < LB - swap( UB, MID )
	// 3) UB < LB < MID - swap( LB, MID )
	// 4) UB = LB < MID - swap( LB, MID ) - At first position
	// 5) MID < UB = LB - swap( UB, MID ) - At last position

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos)
	{
		// Interchange [uiLBPos] with [uiMIDPos]

		RS_SWAP( pucEntryTbl, uiMIDPos, uiLBPos );
		uiMIDPos = uiLBPos;
	}
	else if (uiMIDPos < uiUBPos)
	{
		// Cases 2 and 5
		// Interchange [uUBPos] with [uiMIDPos]

		RS_SWAP( pucEntryTbl, uiMIDPos, uiUBPos );
		uiMIDPos = uiUBPos;
	}

	// To save stack space - recurse the SMALLER Piece.  For the larger
	// piece goto the top of the routine.  Worst case will be
	// (Log2 N)  levels of recursion.

	// Don't recurse in the following cases:
	// 1) We are at an end.  Just loop to the top.
	// 2) There are two on one side.  Compare and swap.  Loop to the top.
	//		Don't swap if the values are equal.  There are many recursions
	//		with one or two entries.  This doesn't speed up any so it is
	//		commented out.

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if( uiLeftItems < uiRightItems)
	{
		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if( uiLeftItems)
		{
			// Recursive call.

			if( RC_BAD( rc = QuickSort( uiLowerBounds, uiMIDPos - 1)))
			{
				goto Exit;
			}
		}

		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems)
	{
		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if( uiRightItems)
		{
			// Recursive call.

			if( RC_BAD( rc = QuickSort( uiMIDPos + 1, uiUpperBounds)))
			{
				goto Exit;
			}
		}

		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Write this block to disk.
*****************************************************************************/
RCODE FResultSetBlk::Write()
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiBytesWritten;

	// By this time there better be something to write...
	// The file should be opened by default.

	if( m_BlockHeader.ui64FilePos == RSBLK_UNSET_FILE_POS)
	{
		if( RC_BAD(rc = (*m_ppFileHdl64)->Size( &m_BlockHeader.ui64FilePos)))
		{
			goto Exit;
		}
	}

	// Write out the block header definition.

	if( RC_BAD( rc = (*m_ppFileHdl64)->Write(
						 m_BlockHeader.ui64FilePos,
						 sizeof( F_BLOCK_HEADER), &m_BlockHeader,
						 &uiBytesWritten)))
	{
		goto Exit;
	}

	// Write out the data buffer

	if( RC_BAD( rc = (*m_ppFileHdl64)->Write(
						 m_BlockHeader.ui64FilePos + sizeof( F_BLOCK_HEADER),
						 m_BlockHeader.uiBlockSize,
						 m_pucBlockBuf,
						 &uiBytesWritten)))
	{
		goto Exit;
	}

Exit:

	return rc;
}

/*****************************************************************************
Desc:	Read in the specified block into memory.
*****************************************************************************/
RCODE FResultSetBlk::Read()
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiBytesRead;
	F_BLOCK_HEADER	BlockHeader;

	// Nothing to do?

	if (m_BlockHeader.ui64FilePos == RSBLK_UNSET_FILE_POS)
	{
		goto Exit;
	}

	// First read the block header in.

	if (RC_BAD( rc = (*m_ppFileHdl64)->Read( m_BlockHeader.ui64FilePos,
						sizeof( F_BLOCK_HEADER ),
						(void *)&BlockHeader, &uiBytesRead)))
	{
		goto Exit;
	}

	// Verify that the block header data is the same.
	// This is the best we can do to verify that the file handle
	// is not junky.

	if (BlockHeader.ui64FilePos != m_BlockHeader.ui64FilePos ||
		 BlockHeader.uiEntryCount != m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}

	// Read in the data buffer

	if (RC_BAD( rc = (*m_ppFileHdl64)->Read(
						m_BlockHeader.ui64FilePos + sizeof( F_BLOCK_HEADER),
						m_BlockHeader.uiBlockSize,
						m_pucBlockBuf, &uiBytesRead)))
	{
		goto Exit;
	}

Exit:

	if (RC_OK(rc))
	{
		m_bPositioned = TRUE;
		m_iEntryPos = -1;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Copies the current entry into the user buffer.  Checks for overflow.
*****************************************************************************/
RCODE FResultSetBlk::CopyCurrentEntry(
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE				rc = NE_XFLM_OK;	// Must be initailized
	FLMUINT			uiEntrySize;
	F_VAR_HEADER *	pEntry;
	FLMBYTE *		pucEntry;

	flmAssert( pucBuffer);

	// Copy the current entry.  This is a shared routine
	// because the code to copy an entry is a little complicated.

	if( !m_bFixedEntrySize)
	{
		pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;
		uiEntrySize = pEntry->ui32Length;
		pucEntry = m_pucBlockBuf + pEntry->ui32Offset;
	}
	else
	{
		uiEntrySize = m_uiEntrySize;
		pucEntry = &m_pucBlockBuf[ m_iEntryPos * uiEntrySize ];
	}

	if( uiBufferLength && (uiEntrySize > uiBufferLength))
	{
		uiEntrySize = uiBufferLength;
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW );
		// Fall through into memcpy.
	}

	f_memcpy( pucBuffer, pucEntry, uiEntrySize);

	if( puiReturnLength)
	{
		*puiReturnLength = uiEntrySize;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Return the Current entry reference in the result set.
*****************************************************************************/
RCODE FResultSetBlk::GetCurrent(
	FLMBYTE *	pBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc;

	flmAssert( m_pucBlockBuf);
	if( !m_bPositioned )
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND );
		goto Exit;
	}

	// Check for EOF and BOF conditions - otherwise return current.

	if (m_iEntryPos >= (FLMINT) m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	if( m_iEntryPos == -1)
	{
		rc = RC_SET( NE_XFLM_BOF_HIT );
		goto Exit;
	}

	rc = CopyCurrentEntry( pBuffer, uiBufferLength, puiReturnLength );

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return a pointer to the next reference in the result set.
		If the result set is not positioned then the first entry will
		be returned.
*****************************************************************************/
RCODE FResultSetBlk::GetNextPtr(
	FLMBYTE **	ppucBuffer,
	FLMUINT *	puiReturnLength)
{
	RCODE	rc = NE_XFLM_OK;

	flmAssert( ppucBuffer);
	flmAssert( puiReturnLength);
	flmAssert( m_bPositioned);

	// Are we on the last entry or past the last entry?

	if (m_iEntryPos + 1 >= (FLMINT) m_BlockHeader.uiEntryCount)
	{
		m_iEntryPos = (FLMINT)m_BlockHeader.uiEntryCount;
		rc = RC_SET( NE_XFLM_EOF_HIT );
		goto Exit;
	}

	// Position to the next entry

	m_iEntryPos++;

	if (!m_bFixedEntrySize)
	{
		F_VAR_HEADER *	pEntry;

		pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;
		*puiReturnLength = (FLMUINT)pEntry->ui32Length;
		*ppucBuffer =  m_pucBlockBuf + pEntry->ui32Offset;
	}
	else
	{
		*puiReturnLength = m_uiEntrySize;
		*ppucBuffer = &m_pucBlockBuf[ m_iEntryPos * m_uiEntrySize];
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the previous reference in the result set.  If the result set
		is not positioned then the last entry will be returned.
*****************************************************************************/
RCODE FResultSetBlk::GetPrev(
	FLMBYTE *	pucBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE	rc = NE_XFLM_OK;

	flmAssert( m_bPositioned);

	// If not positioned then position past last entry.

	if (m_iEntryPos == -1)
	{
		m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
	}

	// Are we on the first entry or before the first entry?

	if (m_iEntryPos == 0)
	{
		m_iEntryPos = -1;
		rc = RC_SET( NE_XFLM_BOF_HIT);
		goto Exit;
	}

	m_iEntryPos--;				// position to previous entry.

	if (RC_BAD( rc = CopyCurrentEntry( pucBuffer, uiBufferLength,
									puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Set the current entry position for this block.
*****************************************************************************/
RCODE FResultSetBlk::SetPosition(
	FLMUINT64	ui64Position)
{
	RCODE	rc= NE_XFLM_OK;

	// Buffer must be set or SetBuffer() will set iEntryPos back to -1.

	flmAssert( m_bPositioned);

	if( ui64Position == RS_POSITION_NOT_SET)
	{
		m_iEntryPos = -1;
		goto Exit;
	}
	
	flmAssert( ui64Position >= m_ui64BlkEntryPosition);

	// Convert to a zero based number relative to this block.

	if (ui64Position >= m_ui64BlkEntryPosition)
	{
		ui64Position -= m_ui64BlkEntryPosition;
	}
	else
	{
		ui64Position = 0;
	}

	if (ui64Position >= m_BlockHeader.uiEntryCount)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		m_iEntryPos = m_BlockHeader.uiEntryCount;
	}
	else
	{
		m_iEntryPos = (FLMINT)ui64Position;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Find the matching entry within the block using the compare routine.
			This does a binary search on entries.  Watch the (out) variable.
Out:		*piCompare = 0		-	match found OR entry would compare between
										the low and high entries in the block.
							 > 0	-	match entry is greater than
										the highest entry in the block.
							 < 0	-	match entry is less than the
										lowest entry in the block.
Notes:	One side effect is that m_iEntryPos is set to the matched
			entry or some random entry if not found is returned.
*****************************************************************************/
RCODE FResultSetBlk::FindMatch(		// Find and return an etnry that
												// matches in this block.
	FLMBYTE *	pucMatchEntry,			// Entry to match
	FLMUINT		uiMatchEntryLength,	// Variable length of above entry
	FLMBYTE *	pucFoundEntry,			// (out) Entry to return
	FLMUINT *	puiFoundEntryLength,	// (out) Length of entry returned
	FLMINT	*	piCompare)				// See comments above.
{
	RCODE			rc = NE_XFLM_OK;
	FLMINT		iCompare;				// Return from CompareEntry
	FLMUINT		uiLow;
	FLMUINT		uiHigh;
	FLMUINT		uiMid;
	FLMUINT		uiLimit;

	uiLow = 0;
	uiHigh = uiLimit = m_BlockHeader.uiEntryCount - 1;

	// Set the match entry length

	if (!uiMatchEntryLength)
	{
		uiMatchEntryLength = m_uiEntrySize;
	}

	// Check the first and last entries in the block.
	// Copy the current entry if found.

	if( RC_BAD( rc = CompareEntry( pucMatchEntry, uiMatchEntryLength, uiLow,
								&iCompare)))
	{
		goto Exit;
	}

	if( iCompare <= 0)
	{
		if( iCompare < 0)
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
		}
		else
		{
			if( pucFoundEntry)
			{
				rc = CopyCurrentEntry( pucFoundEntry, 0, puiFoundEntryLength);
			}
		}

		*piCompare = iCompare;
		goto Exit;
	}

	if( RC_BAD( rc =  CompareEntry( pucMatchEntry, uiMatchEntryLength, uiHigh,
											&iCompare )))
	{
		goto Exit;
	}

	if (iCompare >= 0)
	{
		if (iCompare > 0)
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
		}
		else
		{
			rc = CopyCurrentEntry( pucFoundEntry, 0, puiFoundEntryLength);
		}
		*piCompare = iCompare;
		goto Exit;
	}

	// Set the iCompare to equals because
	// the match entry sorts within the block somewhere.
	// Binary search the entries in the block.  May still
	// not find the matching entry.

	*piCompare = 0;
	for( ;;)
	{
		uiMid = (uiLow + uiHigh) >> 1;	// (uiLow + uiHigh) / 2

		if( RC_BAD( rc = CompareEntry( pucMatchEntry, uiMatchEntryLength, uiMid,
											&iCompare)))
		{
			goto Exit;
		}

		if( iCompare == 0)
		{
			// Found Match!  All set up to return.

			if( pucFoundEntry)
			{
				rc = CopyCurrentEntry( pucFoundEntry, 0, puiFoundEntryLength);
			}

			goto Exit;
		}

		// Check if we are done - where uiLow >= uiHigh.

		if( uiLow >= uiHigh)
		{
			// Done - item not found

			break;
		}

		if( iCompare < 0)
		{
			// Way too high?

			if( !uiMid)
			{
				break;
			}

			// Too high

			uiHigh = uiMid - 1;
		}
		else
		{
			if( uiMid == uiLimit)
			{
				// Done - hit the top
				break;
			}

			// Too low

			uiLow = uiMid + 1;
		}
	}

	// On break set we did not find the matching entry.

	rc = RC_SET( NE_XFLM_NOT_FOUND);

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Compare the buffer entry with entry identifies by
			uiEntryPos.
Out:		*piCompare = 0	-	match found OR entry value would compare
									between the low and high entries.
						  > 0	-	match entry is greater than
									the highest entry in the block.
						  < 0	-	match entry is less than the
									lowest entry in the block.
*****************************************************************************/
RCODE FResultSetBlk::CompareEntry(	// Compares match entry with entry
												// identified by uiEntryPos.
	FLMBYTE *	pucMatchEntry,			// Entry to match
	FLMUINT		uiMatchEntryLength,	// Variable length of pMatchEntry.
	FLMUINT		uiEntryPos,				// Position of entry in block.
	FLMINT *		piCompare)				// Return from compare.
{
	FLMBYTE *	pucEntry;
	FLMUINT		uiEntrySize;

	// Position to the entry.

	m_iEntryPos = (FLMINT) uiEntryPos;

	if (!m_bFixedEntrySize)
	{
		F_VAR_HEADER *	pEntry;

		pEntry = ((F_VAR_HEADER *)m_pucBlockBuf) + m_iEntryPos;
		uiEntrySize = (FLMUINT)pEntry->ui32Length;
		pucEntry = m_pucBlockBuf + pEntry->ui32Offset;
	}
	else
	{
		uiEntrySize = m_uiEntrySize;
		pucEntry = &m_pucBlockBuf[ m_iEntryPos * uiEntrySize ];
	}

	return( m_pCompare->compare( pucMatchEntry, uiMatchEntryLength,
							pucEntry, uiEntrySize,
							piCompare));
}

/*****************************************************************************
Desc:	Make sure the state reflects what we have in the blocks.
*****************************************************************************/
void FResultSetBlk::adjustState(
	FLMUINT			uiBlkBufferSize)
{
	F_VAR_HEADER *		pVarHdr;
	FLMUINT				uiTotalSize = 0;
	FLMBYTE *			pucFromPos;
	FLMBYTE *			pucToPos;
	FLMUINT				uiBytesMoved;
	FLMUINT				uiPos;

	// Are the entries in the block fixed length or variable length?

	if( m_bFixedEntrySize)
	{
		// Fixed Length.

		m_uiLengthRemaining = uiBlkBufferSize - 
										(m_BlockHeader.uiEntryCount * m_uiEntrySize);
		m_ui64BlkEntryPosition = 0;
		m_pucEndPoint = m_pucBlockBuf + (m_BlockHeader.uiEntryCount * m_uiEntrySize);
	}
	else
	{
		// Variable length Entries.
		// We may need to move the entries around.  First, determine if the block is full.

		if( m_BlockHeader.uiBlockSize < uiBlkBufferSize)
		{
			uiTotalSize = m_BlockHeader.uiBlockSize -
								(sizeof(F_VAR_HEADER) * m_BlockHeader.uiEntryCount);
			
			pucFromPos = m_pucBlockBuf + (sizeof(F_VAR_HEADER) * m_BlockHeader.uiEntryCount);

			pucToPos = (m_pucBlockBuf + uiBlkBufferSize) - uiTotalSize;

			f_memmove( pucToPos, pucFromPos, uiTotalSize);

			for( uiBytesMoved = (pucToPos - pucFromPos),
							uiPos = 0,
							pVarHdr = (F_VAR_HEADER *)m_pucBlockBuf;
				  uiPos < m_BlockHeader.uiEntryCount;
				  pVarHdr++, uiPos++)
			{
				pVarHdr->ui32Offset += (FLMUINT32)uiBytesMoved;
			}

			m_pucEndPoint = pucToPos;
			m_uiLengthRemaining = uiBlkBufferSize - m_BlockHeader.uiBlockSize;
			m_ui64BlkEntryPosition = pucToPos - m_pucBlockBuf;
		}
		else
		{
			m_uiLengthRemaining = 0;
		}
	}

	m_BlockHeader.uiBlockSize = uiBlkBufferSize;
}

/*****************************************************************************
Desc:	truncate the file to the current file position.
*****************************************************************************/
RCODE FResultSetBlk::Truncate(
	FLMBYTE *			pszPath)
{
	RCODE				rc = NE_XFLM_OK;

	if( RC_BAD( rc = (*m_ppFileHdl64)->Truncate( 
		m_BlockHeader.ui64FilePos)))
	{
		goto Exit;
	}

	(*m_ppFileHdl64)->Close( FALSE);

	if( RC_BAD( rc = (*m_ppFileHdl64)->Open( ( char *)pszPath)))
	{
		goto Exit;
	}

	m_BlockHeader.ui64FilePos = RSBLK_UNSET_FILE_POS;

Exit:

	return( rc);
}
