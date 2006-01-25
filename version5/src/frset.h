//------------------------------------------------------------------------------
// Desc:	Result sets
//
// Tabs:	3
//
//		Copyright (c) 1996-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frset.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FRSET_H
#define FRSET_H

// Forward declarations

class FResultSet;
class FResultSetBlk;

/*****************************************************************************
*****
** 	Definitions
*****
*****************************************************************************/

#define	RSBLK_BLOCK_SIZE		(1024 * 512)

// Need to allocate on the stack the maximum fixed entry size
// in the recursive quicksort routine.

#define	MAX_FIXED_ENTRY_SIZE			64

#define	RS_POSITION_NOT_SET		FLM_MAX_UINT64

/*****************************************************************************
*****
** 	Result Set Block Definitions
*****
*****************************************************************************/

// Block Size Limits:
//		1) 4 GB (whatever we can represent in 32 bits)
//		2) Three blocks are allocated as a maximum during the merge phase.
//			Two of these blocks are freed after the merge phase.
//
// Block Layout:
//		Block Header Structure
//		Variable Length Entries
//			An array of [Offset][Length] entry items (F_VAR_HEADER)
//			The variable length entries are stored from the end back to front.
//			The length of a variable length entry must be less than the block size.
//		Fixed Length Entries
//			Each entry is stored from the first of the buffer to the last.

typedef struct FVarHeaderTag
{
	FLMUINT32	ui32Offset;
	FLMUINT32	ui32Length;
} F_VAR_HEADER;

/*===========================================================================

							Block Header Definition

  Desc:	Actually stored as the first section of each block.
			We can write this structure because the same process will
			read the block header i.e. portability is not a problem.
===========================================================================*/

typedef struct FBlockHeaderTag
{
	FLMUINT64	ui64FilePos;			// ~0 or file position
	FLMUINT		uiEntryCount;			// Number of entries in block
	FLMUINT		uiBlockSize;			// Total Block size in memory or on disk
	FLMBOOL		bFirstBlock;			// TRUE=first block in chain
	FLMBOOL		bLastBlock;				// TRUE=last block in chain
} F_BLOCK_HEADER;

#define	RSBLK_UNSET_FILE_POS		(~((FLMUINT64)0))

/*===========================================================================

							Result Set Block Class Definition
							Move to another file if FResultSet is more public.
							Source: FRSETBLK.CPP

Note: Anything member variable in comments is for the future.
===========================================================================*/
class	FResultSetBlk : public XF_RefCount, public XF_Base
{
public:

	// Constructor

	FResultSetBlk();

	// Destructor

	FINLINE ~FResultSetBlk()
	{
		if (m_pNext)
		{
			m_pNext->m_pPrev = m_pPrev;
		}
		
		if( m_pPrev)
		{
			m_pPrev->m_pNext = m_pNext;
		}
		
		if (m_pCompare)
		{
			m_pCompare->Release();
		}
	}

	// Setup and termination methods

	void reset( void);

	void Setup(
		F_64BitFileHandle **		ppFileHdl64,	// file handle to use for temp file.
		IF_ResultSetCompare *	pCompare,
		FLMUINT						uiEntrySize,	// Entry size if fixed.
		FLMBOOL						bFirstInList,	// Use RSBLK_IS_FIRST_IN_LIST or
															// RSBLK_NOT_FIRST_IN_LIST
		FLMBOOL						bDropDuplicates,	// If TRUE drop duplicates
		FLMBOOL						bEntriesInOrder);	// TRUE when entries are in order.

	RCODE		SetBuffer(					// Set the current working buffer
		FLMBYTE *		pBuffer,
		FLMUINT			uiBufferSize = RSBLK_BLOCK_SIZE);

	FINLINE FLMUINT	BytesUsedInBuffer( void)
	{
		if (m_bEntriesInOrder)
		{
			return( m_BlockHeader.uiBlockSize);
		}
		else
		{
			return( m_BlockHeader.uiBlockSize - m_uiLengthRemaining);
		}
	}

	// Entry Add and Sort Methods

	RCODE AddEntry(						// Variable or fixed length entry coming in
		FLMBYTE *	pEntry,				// Entry cannot be all zero values.
		FLMUINT		uiEntryLength );	// If length is zero then ignore entry.

	RCODE ModifyEntry(					// Modify current entry.
		FLMBYTE *	pEntry,				// Points to entry buffer
		FLMUINT		uiEntryLength = 0);	// Zero value means fixed length.
												// Existing entry must be the same length
												// and sort in the same place.


	FINLINE RCODE Finalize(
		FLMBOOL		bForceWrite)
	{
		return Flush( TRUE, bForceWrite);
	}

	RCODE Flush(							// Sort and flush block to disk if more data
		FLMBOOL		bLastBlockInList,
		FLMBOOL		bForceWrite);

	// Methods to read entries.

	RCODE GetCurrent(						// Return current entry
		FLMBYTE *	pBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	FINLINE RCODE GetNext(
		FLMBYTE *	pucBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength)
	{
		// Are we on the last entry or past the last entry?

		if (m_iEntryPos + 1 >= (FLMINT)m_BlockHeader.uiEntryCount)
		{
			m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
			return RC_SET( NE_XFLM_EOF_HIT);
		}

		m_iEntryPos++;				// Else position to next entry

		return CopyCurrentEntry( pucBuffer, uiBufferLength, puiReturnLength);
	}

	RCODE GetNextPtr(						// Get a pointer to the next entry.
												// Internal call not to be exposed.
		FLMBYTE * *	ppBuffer,
		FLMUINT *	puiReturnLength);

	RCODE GetPrev(							// Position to previous entry and return
		FLMBYTE *	pBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	FINLINE FLMUINT64 GetPosition( void)
	{
		return( (!m_bPositioned ||
								m_iEntryPos == -1 ||
								m_iEntryPos == (FLMINT)m_BlockHeader.uiEntryCount
								? RS_POSITION_NOT_SET
								: m_ui64BlkEntryPosition + (FLMUINT64)m_iEntryPos));
	}


	RCODE SetPosition(					// Set the input position.  Returns
												// SUCCESS or NE_XFLM_NOT_FOUND if position bad.
		FLMUINT64	ui64Position );	// (in) Position or RS_POSITION_NOT_SET

	RCODE	FindMatch(						// Find and return an etnry that
												// matches in this block.
		FLMBYTE *	pMatchEntry,			// Entry to match
		FLMUINT		uiMatchEntryLength,	// Variable length of above entry
		FLMBYTE *	pFoundEntry,			// (out) Entry to return
		FLMUINT *	puiFoundEntryLength,	// (out) Length of entry returned
		FLMINT *		piCompare);			// See comments above.

	void adjustState(
		FLMUINT			uiBlkBufferSize);

	RCODE Truncate(
		FLMBYTE *		pszPath);

private:

	RCODE AddEntry(						// Fixed length entry coming in.
		FLMBYTE *	pucEntry);			// Entry cannot be all zero values.

	void SqueezeSpace( void);			// Squeeze out space if variable length.

	RCODE SortAndRemoveDups( void);	// Sort entries in a block and remove dups.

	void RemoveEntry(						// Remove an entry.
		FLMBYTE *	pucEntry);

	RCODE QuickSort(						// The great quick sort algorithm.
		FLMUINT		uiLowerBounds,
		FLMUINT		uiUpperBounds);

	FINLINE RCODE EntryCompare(
		FLMBYTE *		pucLeftEntry,
		FLMBYTE *		pucRightEntry,
		FLMINT *			piCompare)
	{
		RCODE				rc;

		if( m_bFixedEntrySize)
		{
			rc = m_pCompare->compare( pucLeftEntry,  m_uiEntrySize,
						pucRightEntry, m_uiEntrySize, piCompare);
		}
		else
		{
			rc = m_pCompare->compare(
						m_pucBlockBuf + ((F_VAR_HEADER *)pucLeftEntry)->ui32Offset,
						((F_VAR_HEADER *)pucLeftEntry)->ui32Length,
						m_pucBlockBuf + ((F_VAR_HEADER *)pucRightEntry)->ui32Offset,
						((F_VAR_HEADER *)pucRightEntry)->ui32Length,
						piCompare);
		}
		if (*piCompare == 0)
		{
			m_bDuplicateFound = TRUE;
		}
		return rc;
	}

	RCODE CopyCurrentEntry(				// Copy current entry to pBuffer.
		FLMBYTE *	pBuffer,
		FLMUINT		uiBufferLength,
		FLMUINT *	puiReturnLength);

	RCODE CompareEntry(					// Compares match entry with entry
												// identified by uiEntryPos.
		FLMBYTE *	pMatchEntry,		// Entry to match
		FLMUINT		uiMatchEntryLength,// Variable length of pMatchEntry.
		FLMUINT		uiEntryPos,			// Position of entry in block.
		FLMINT *		piCompare);			// Return from compare.

	// I/O Methods

	RCODE Write();							// Write a block to disk
	RCODE Read();							// Read a block of data from disk

	// Member Variables

	F_BLOCK_HEADER	m_BlockHeader;			// Block Header - written to disk.

	IF_ResultSetCompare *		m_pCompare;

	// Variables to track data in the block.

	FLMBYTE *		m_pucBlockBuf;			// Block buffer.  Allocated by owner
	FLMBYTE *		m_pucEndPoint;			// Entry data runs from rear to front

	FResultSetBlk *	m_pNext;				// Next RSBLK in a chain.
	FResultSetBlk *	m_pPrev;				// Previous RSBlk in a chain.

	F_64BitFileHandle **
						m_ppFileHdl64;			// Points to a 64-bit file handle

	FLMUINT64		m_ui64BlkEntryPosition;// Position of first entry in block.
	FLMUINT			m_uiLengthRemaining;	// Bytes left between entry pointers
													// and pEndPoint (start of entry data.)

	// Flags and counters

	FLMINT			m_iEntryPos;			// Entry position when reading.
	FLMUINT			m_uiEntrySize;			// Entry size used in the block (non-0).

	FLMBOOL			m_bEntriesInOrder;	// TRUE = entries are in order.
	FLMBOOL			m_bFixedEntrySize;	// TRUE = process fixed length entries.
	FLMBOOL			m_bPositioned;			// TRUE = we are positioned through
													// ResultSetFirst() or ResultSetLast().
	FLMBOOL			m_bModifiedEntry;		// 1+ entries in blk modifed or deleted.
	FLMBOOL			m_bDuplicateFound;	// TRUE = duplicate found when sorting.
	FLMBOOL			m_bDropDuplicates;	// TRUE drop duplicate entries.
friend class FResultSet;
};

/*****************************************************************************
Desc:	Result set class
*****************************************************************************/
class FResultSet : public IF_ResultSet, public XF_Base
{
public:

	FResultSet();
	FResultSet(
		FLMUINT		uiBlkSize);

	virtual ~FResultSet();							// No errors returned when closing files
															// or freeing memory.

	RCODE XFLMAPI setupResultSet(
		const char *				pszPath,			// Default I/O path to use for the result
															// set files.  There may be 2 files when
															// a lot of data is to be merged.
		IF_ResultSetCompare *	pCompare,		// Compare callback object
		FLMUINT						uiEntrySize,	// Size of entry if fixed length or zero
															// if variable length.  Assert will be
															// called if entry size changed when setup
															// is called multiple times.
		FLMBOOL						bDropDuplicates = TRUE,
		FLMBOOL						bEntriesInOrder = FALSE,
															// TRUE if entries are in order.
		const char *				pszInputFileName = NULL);	
															// Use this file if needed.  If specified
															// AND it exists, the file will be opened and read.
															// If you don't want to use an existing file, you
															// should call resetResultSet immediately after this call.
															// That will delete the existing file.

	FINLINE void XFLMAPI setSortStatus(
		IF_ResultSetSortStatus *	pSortStatus)
	{
		if (m_pSortStatus)
		{
			m_pSortStatus->Release();
			m_pSortStatus = NULL;
		}
		if ((m_pSortStatus = pSortStatus) != NULL)
		{
			m_pSortStatus->AddRef();
		}
	}

	FINLINE FLMUINT64 XFLMAPI getTotalEntries( void)
	{
		FResultSetBlk	*	pBlk = m_pFirstRSBlk;
		FLMUINT64			ui64TotalEntries = 0;

		for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->m_pNext)
		{
			ui64TotalEntries += pBlk->m_BlockHeader.uiEntryCount;
		}
		return ui64TotalEntries;
	}

	// Methods for building a result set.

	RCODE XFLMAPI addEntry(				// Add fixed or variable length entry.
		const void *	pvEntry,				// Points to entry buffer
		FLMUINT			uiEntryLength = 0);	// zero value means fixed length

	RCODE XFLMAPI finalizeResultSet(	// Commit all entries added.
		FLMUINT64 *		pui64TotalEntries = NULL);// (out) returns total entries.

	// Reading Entries from a Result Set

	RCODE XFLMAPI getFirst(					// Returns first entry
		void *		pvEntryBuffer,
		FLMUINT		uiBufferLength = 0,
		FLMUINT *	puiEntryLength = NULL);

	RCODE XFLMAPI getNext(					// Returns next entry
		void *		pvEntryBuffer,
		FLMUINT		uiBufferLength = 0,
		FLMUINT *	puiEntryLength = NULL);

	RCODE XFLMAPI getLast(					// Returns last entry
		void *		pvEntryBuffer,
		FLMUINT		uiBufferLength = 0,
		FLMUINT *	puiEntryLength = NULL);

	RCODE XFLMAPI getPrev(					// Returns previous entry
		void *		pvEntryBuffer,
		FLMUINT		uiBufferLength = 0,
		FLMUINT *	puiEntryLength = NULL);

	RCODE XFLMAPI getCurrent(				// Returns current entry
		void *		pvEntryBuffer,			// Buffer to hold entry
		FLMUINT		uiBufferLength = 0,	// Length of buffer to hold entry
		FLMUINT *	puiEntryLength = NULL);	// (out) Length of returned entry

	FINLINE RCODE XFLMAPI modifyCurrent(
		const void *	pvEntry,
		FLMUINT			uiEntryLength = 0)	// If zero entry is fixed length.
	{
		return( m_pCurRSBlk->ModifyEntry( (FLMBYTE *)pvEntry, uiEntryLength));
	}

	FINLINE RCODE XFLMAPI findMatch(
		const void *	pvMatchEntry,	// Fixed length entry to match.
		void *			pvFoundEntry)	// (out) Buffer to return full entry to.
	{
		return( findMatch( pvMatchEntry, m_uiEntrySize,
								pvFoundEntry, NULL));
	}

	RCODE XFLMAPI findMatch(						// Find and return an etnry that
															// matches in the result set (variable).
		const void *	pvMatchEntry,				// Entry to match
		FLMUINT			uiMatchEntryLength,		// Variable length of above entry
		void *			pvFoundEntry,				// (out) Entry to return
		FLMUINT *		puiFoundEntryLength);	// (out) Length of entry returned
		
	FINLINE FLMUINT64 XFLMAPI getPosition( void)
	{
		return( (!m_pCurRSBlk
								? RS_POSITION_NOT_SET
								: m_pCurRSBlk->GetPosition()));
	}

	RCODE XFLMAPI setPosition(
		FLMUINT64		ui64Position);

	RCODE XFLMAPI resetResultSet(
		FLMBOOL			bDelete = TRUE);

	RCODE XFLMAPI flushToFile( void);

private:

	// Private Methods

	FINLINE FLMUINT64 NumberOfBlockChains( void)
	{
		FLMUINT64			ui64Count = 0;
		FResultSetBlk *	pBlk = m_pFirstRSBlk;

		for (; pBlk ; pBlk = pBlk->m_pNext)
		{
			if (pBlk->m_BlockHeader.bFirstBlock)
			{
				ui64Count++;
			}
		}
		return ui64Count;
	}

	RCODE MergeSort();				// Sort/Merge all blks into single list.

	RCODE GetNextPtr(						// Return ptr to next entry.
		FResultSetBlk **ppCurBlk,
		FLMBYTE *	*	ppBuffer,
		FLMUINT *		puiReturnLength);

	RCODE UnionBlkLists(
		FResultSetBlk *pLeftBlk,			// (IN) Left block
		FResultSetBlk *pRightBlk = NULL);// (IN) May be NULL

	RCODE CopyRemainingItems(				// Copy remaining items from a block list.
		FResultSetBlk *pCurBlk);			// (IN) Inputblock

	void CloseFile(							// Close the input file.
		F_64BitFileHandle **		ppFileHdl64,
		FLMBOOL						bDelete = TRUE);

	RCODE OpenFile(							// Open the input file
		F_64BitFileHandle **		ppFileHdl64);

	FResultSetBlk *SelectMidpoint(		// Select the midpoint between blocks.
		FResultSetBlk *pLowBlk,
		FResultSetBlk *pHighBlk,
		FLMBOOL			bPickHighIfNeighbors);

	RCODE XFLMAPI setupFromFile( void);


	// Callback objects

	IF_ResultSetCompare *		m_pCompare;

	IF_ResultSetSortStatus *	m_pSortStatus;
	FLMUINT64						m_ui64EstTotalUnits;	// Estimated total number of units to do.
	FLMUINT64						m_ui64UnitsDone;		// Units completed

	FLMUINT			m_uiEntrySize;			// Fixed length entry size or 0 for var.

	FLMUINT64		m_ui64TotalEntries;	// Total number of entries.

	FResultSetBlk *m_pCurRSBlk,			 // Current result set block.
					  *m_pFirstRSBlk,			// Points to first of merge list
					  *m_pLastRSBlk;			// Points to last of merge list

	char				m_szIoDefaultPath[ F_PATH_MAX_SIZE],// Copy of default path from setup().
						m_szIoFilePath1[ F_PATH_MAX_SIZE],	// File created for result set.
						m_szIoFilePath2[ F_PATH_MAX_SIZE];	// File create for result set merge.

	F_64BitFileHandle *
						m_pFileHdl641;			// File handles.
	F_64BitFileHandle	*
						m_pFileHdl642;

	FLMBYTE *		m_pucBlockBuf1;		// Buffer for initial loading of entries
	FLMBYTE *		m_pucBlockBuf2;		// Buffer for merge step.
	FLMBYTE *		m_pucBlockBuf3;		// Buffer for merge step.
	FLMUINT			m_uiBlockBuf1Len;

	FLMBOOL			m_bFile1Opened;		// TRUE when m_IoFile1 is opened.
	FLMBOOL			m_bFile2Opened;		// TRUE when m_IoFile2 is opened.
	FLMBOOL			m_bOutput2ndFile;		// TRUE if output is 2nd file
	FLMBOOL			m_bInitialAdding;		// TRUE when user adding entries.
	FLMBOOL			m_bFinalizeCalled;	// TRUE after finalize step.
	FLMBOOL			m_bSetupCalled;		// TRUE after setup has been called.
	FLMBOOL			m_bDropDuplicates;	// TRUE drop duplicate entries.
	FLMBOOL			m_bAppAddsInOrder;	// TRUE entries are added in order.
	FLMBOOL			m_bEntriesInOrder;	// TRUE entries are in order
	FLMUINT			m_uiBlkSize;			// Default is RSBLK_BLOCK_SIZE

friend class FResultSetBlk;
};

#endif		// ifndef FRSET_H
