//------------------------------------------------------------------------------
// Desc:	Result set routines
//
// Tabs:	3
//
//		Copyright (c) 1996-1998, 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frset.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

// Make sure that the extension is in lower case characters.

#define		FRSET_FILENAME_EXTENSION		"frs"

/*
** Sorting Result Sets:
**
	New algorithm 7/2/97.  This is a good one!
	Below are refinements to the existing result set code.

	1) We now have two files that are used in the merge-sort process.
		The first file is used to hold all of the blocks created when
		adding entries into the result set.  The second file is used for
		the first and thereafter odd merge steps.  The first file is then
		truncated and used for each even merge step.  At the end of the
		merge one of the files will be deleted.  Three buffers are used
		during merge and only one will remain after the merge is done.
		This is safer than the previous method and uses a little less
		disk space.  There are many small improvements that can be made.

	2) The result set code now takes a buffer and a length and has no
		knowledge of the data in the result set.  In fact, the data may
		be fixed length or variable length.  We removed the record cache
		from the result set and made a record cache manager.

  Future enhancements to consider:
	1) Do a 3, 4 or N-Way merge.
		This will greatly increase memory allocations but save a lot of time
		reading and writing when the number of entries is large.
	2) Use 3 buffers on the initial load.  This is really doing the first
		phase of the merge when adding entries.  The algorithm would add
		entries to two buffers, and when full merge to the third buffer and
		write out two sorted buffer.
	3) Don't write out the last block - use it as the first block of the
		merge when not complete.  This will save a write and read on each
		pass.  In addition, the I/O cache may be helped out.
		In addition, the last block of each phase should be used first
		on the next phase.

  Old Notes:
		Duplicate Entries:
			Duplicate entries are very difficult for a general purpose sorter
			to find.  In some cases the user would want to compare only these
			fields and not these to determine a duplicate.  This result set
			code lets the user pass in a callback routine to determine if
			two entries are the same and if one should be dropped.  The user
			could pass in NULL to cause all duplicates to be retained.
		Quick Sort Algorithm:
			This algorithm, in FRSETBLK.CPP is a great algorithm that Scott
			came up with.  It will recurse only Log(base2)N times (number of
			bits needed to represent N).  This is a breakthrough because
			all sorting algorithms I have seen will recurse N-1 times if
			the data is in order or in reverse order.  This will crash the
			stack for a production quality sort.
		Variable Length
			This sorting engine (result set) supports variable length and
			fixed length data.  There is very low overhead for variable
			length support.  This sorting engine can be used for a variety
			of tasks.

  Example:

	All numbers are logical block numbers.

	Adding		Pass 1		Pass2			Pass3				Pass4
	Phase 		File 2		File 1		File 2			File 1
	File 1		(created)	(truncated)	(truncated)		(truncated)
	=========	=========	=========	===========		====================
	1				10 (1+2)		14 (10+11)	16 (14+15)		17 Final file (16+9)
	2
	3				11 (3+4)		15 (12+13)	9
	4
	5				12 (5+6)		9
	6
	7				13 (7+8)
	8
	9				9
*/

/*****************************************************************************
Desc:
*****************************************************************************/
FResultSet::FResultSet()
{
	m_pCompare = NULL;
	m_pSortStatus = NULL;
	m_ui64EstTotalUnits = 0;
	m_ui64UnitsDone = 0;

	m_uiEntrySize = 0;
	m_ui64TotalEntries = 0;
	m_pCurRSBlk = NULL;
	m_pFirstRSBlk = NULL;
	m_pLastRSBlk = NULL;

	f_memset( &m_szIoDefaultPath[0], 0, F_PATH_MAX_SIZE);

	m_pucBlockBuf1 = NULL;
	m_pucBlockBuf2 = NULL;
	m_pucBlockBuf3 = NULL;
	m_uiBlockBuf1Len = 0;
	m_bFile1Opened = FALSE;
	m_bFile2Opened = FALSE;
	m_pFileHdl641 = NULL;
	m_pFileHdl642 = NULL;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bFinalizeCalled = FALSE;
	m_bSetupCalled = FALSE;
	m_uiBlkSize = RSBLK_BLOCK_SIZE;
}

/*****************************************************************************
Desc:
*****************************************************************************/
FResultSet::FResultSet(
	FLMUINT			uiBlkSize)
{
	m_pCompare = NULL;
	m_pSortStatus = NULL;
	m_ui64EstTotalUnits = 0;
	m_ui64UnitsDone = 0;

	m_uiEntrySize = 0;
	m_ui64TotalEntries = 0;
	m_pCurRSBlk = NULL;
	m_pFirstRSBlk = NULL;
	m_pLastRSBlk = NULL;

	f_memset( &m_szIoDefaultPath[0], 0, F_PATH_MAX_SIZE);

	m_pucBlockBuf1 = NULL;
	m_pucBlockBuf2 = NULL;
	m_pucBlockBuf3 = NULL;
	m_uiBlockBuf1Len = 0;
	m_bFile1Opened = FALSE;
	m_bFile2Opened = FALSE;
	m_pFileHdl641 = NULL;
	m_pFileHdl642 = NULL;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bFinalizeCalled = FALSE;
	m_bSetupCalled = FALSE;
	m_uiBlkSize = uiBlkSize;
}

/*****************************************************************************
Desc:
*****************************************************************************/
FResultSet::~FResultSet()
{
	FResultSetBlk *pCurRSBlk;
	FResultSetBlk *pNextRSBlk;

	// Free up the result set block chain.

	for( pCurRSBlk = m_pFirstRSBlk; pCurRSBlk; pCurRSBlk = pNextRSBlk)
	{
		FLMUINT		uiCount;

		pNextRSBlk = pCurRSBlk->m_pNext;
		uiCount = pCurRSBlk->Release();
		flmAssert( !uiCount);
	}

	// Set list to NULL for debugging in memory.

	m_pFirstRSBlk = NULL;
	m_pLastRSBlk = NULL;
	m_pCurRSBlk = NULL;

	// Free up all of the block buffers in the list.

	f_free( &m_pucBlockBuf1);
	f_free( &m_pucBlockBuf2);
	f_free( &m_pucBlockBuf3);

	// Close all opened files

	CloseFile( &m_pFileHdl641 );
	CloseFile( &m_pFileHdl642 );

	if( m_pCompare)
	{
		m_pCompare->Release();
	}

	if( m_pSortStatus)
	{
		m_pSortStatus->Release();
	}
}

/*****************************************************************************
Desc:	Reset the result set so it can be reused.
*****************************************************************************/
RCODE XFLMAPI FResultSet::resetResultSet(
	FLMBOOL		bDelete)
{
	RCODE					rc = NE_XFLM_OK;
	FResultSetBlk *	pCurRSBlk;
	FResultSetBlk *	pNextRSBlk;

	// Free up the result set block chain - except for the first one.

	for( pCurRSBlk = m_pFirstRSBlk; pCurRSBlk; pCurRSBlk = pNextRSBlk)
	{
		FLMUINT		uiCount;

		pNextRSBlk = pCurRSBlk->m_pNext;
		if( pCurRSBlk != m_pFirstRSBlk)
		{
			uiCount = pCurRSBlk->Release();
			flmAssert( !uiCount);
		}
	}

	// Free up all of the block buffers in the list, except for the first one.

	f_free( &m_pucBlockBuf2);
	f_free( &m_pucBlockBuf3);

	if( !m_pucBlockBuf1 || m_uiBlockBuf1Len != m_uiBlkSize)
	{
		if( m_pucBlockBuf1)
		{
			f_free( &m_pucBlockBuf1);
		}

		if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf1)))
		{
			goto Exit;
		}

		m_uiBlockBuf1Len = m_uiBlkSize;
	}

	// Close all opened files

	CloseFile( &m_pFileHdl641, bDelete );
	CloseFile( &m_pFileHdl642 );
	m_bFile1Opened = m_bFile2Opened = FALSE;
	m_pFileHdl641 = m_pFileHdl642 = NULL;

	// Reset some other variables

	if( m_pSortStatus)
	{
		m_pSortStatus->Release();
		m_pSortStatus = NULL;
	}

	m_ui64EstTotalUnits = 0;
	m_ui64UnitsDone = 0;
	m_ui64TotalEntries = 0;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bEntriesInOrder = m_bAppAddsInOrder;
	m_bFinalizeCalled = FALSE;

	// If we don't have a block, allocate it.  Otherwise
	// reset the one we have left.

	if( !m_pFirstRSBlk)
	{
		if( (m_pFirstRSBlk = f_new FResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	else
	{
		m_pFirstRSBlk->reset();
	}

	m_pLastRSBlk = m_pCurRSBlk = m_pFirstRSBlk;
	(void)m_pFirstRSBlk->Setup( &m_pFileHdl641, m_pCompare,
		m_uiEntrySize, TRUE, m_bDropDuplicates,
		m_bEntriesInOrder);
	(void) m_pFirstRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlockBuf1Len);

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Setup the result set with all of the needed input values.
		This method must only be called once.
*****************************************************************************/
RCODE XFLMAPI FResultSet::setupResultSet(
	const char *				pszDirPath,
	IF_ResultSetCompare *	pCompare,
	FLMUINT						uiEntrySize,
	FLMBOOL						bDropDuplicates,
	FLMBOOL						bEntriesInOrder,
	const char *				pszInputFileName)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bNewBlock = FALSE;
	FLMBOOL	bNewBuffer = FALSE;

	flmAssert( !m_bSetupCalled );
	flmAssert( uiEntrySize <= MAX_FIXED_ENTRY_SIZE);

	// Perform all of the allocations first.

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = f_new FResultSetBlk;

	// Allocation Error?

	if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_XFLM_MEM );
		goto Exit;
	}

	bNewBlock = TRUE;
	m_pCurRSBlk->Setup( &m_pFileHdl641, pCompare,
			uiEntrySize, TRUE, bDropDuplicates, bEntriesInOrder);

	// Allocate only the first buffer - other buffers only used in merge.

	if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf1)))
	{
		goto Exit;
	}

	m_uiBlockBuf1Len = m_uiBlkSize;
	bNewBuffer = TRUE;
	(void) m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlockBuf1Len);

	// Set the input variables.

	if( pszDirPath)
	{
		f_strcpy( m_szIoDefaultPath, pszDirPath);
	}

	if( m_pCompare)
	{
		m_pCompare->Release();
	}

	if( (m_pCompare = pCompare) != NULL)
	{
		m_pCompare->AddRef();
	}

	m_uiEntrySize = uiEntrySize;
	m_bDropDuplicates = bDropDuplicates;
	m_bEntriesInOrder = m_bAppAddsInOrder = bEntriesInOrder;
	
	// If a filename was passed in, then we will try to open it and read whatever
	// data it holds into the result set.  If the file does not exist, it will not
	// be created at this time.

	if( pszInputFileName)
	{
		f_strcpy( m_szIoFilePath1, m_szIoDefaultPath);

		if( RC_BAD( rc = gv_pFileSystem->pathAppend( 
			m_szIoFilePath1, pszInputFileName)))
		{
			goto Exit;
		}

		f_strcat( m_szIoFilePath1, "." FRSET_FILENAME_EXTENSION);

		if( RC_BAD( rc = setupFromFile()))
		{
			goto Exit;
		}
	}

Exit:

	// Free allocations on any error

	if( RC_BAD(rc))
	{
		if( bNewBlock)
		{
			if( m_pCurRSBlk)
			{
				m_pCurRSBlk->Release();
				m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;
			}
		}

		if( bNewBuffer)
		{
			f_free( &m_pucBlockBuf1);
			m_uiBlockBuf1Len = 0;
		}
	}
	else
	{
		m_bSetupCalled = TRUE;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Attempt to establish the result set from an existing file.
*****************************************************************************/
RCODE XFLMAPI FResultSet::setupFromFile( void)
{
	RCODE						rc = NE_XFLM_OK;
	FResultSetBlk *		pNextRSBlk;
	FLMUINT					uiOffset;
	FLMUINT					uiBytesRead;
	F_BLOCK_HEADER			BlkHdr;

	flmAssert( !m_bSetupCalled);

	if( (m_pFileHdl641 = f_new F_64BitFileHandle) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pFileHdl641->Open( m_szIoFilePath1)))
	{
		if( rc == NE_XFLM_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = m_pFileHdl641->Create( m_szIoFilePath1)))
			{
				rc = NE_XFLM_OK;
				m_pFileHdl641->Release();
				m_pFileHdl641 = NULL;
				goto Exit;
			}
		}
		else
		{
			rc = NE_XFLM_OK;
			m_pFileHdl641->Release();
			m_pFileHdl641 = NULL;
			goto Exit;
		}
	}

	m_bFile1Opened = TRUE;

	// Release the current set of blocks.
	
	while( m_pFirstRSBlk)
	{
		m_pCurRSBlk = m_pFirstRSBlk;
		m_pFirstRSBlk = m_pFirstRSBlk->m_pNext;
		m_pCurRSBlk->Release();
	}

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;

	// Allocate the buffer that we will use to read the data in.

	if( !m_pucBlockBuf1)
	{
		if( RC_BAD( rc = f_calloc( m_uiBlkSize, &m_pucBlockBuf1)))
		{
			goto Exit;
		}
		m_uiBlockBuf1Len = m_uiBlkSize;
	}
	else
	{
		f_memset( m_pucBlockBuf1, 0, m_uiBlkSize);
	}

	// Now read every block in the file and create a FResultSetBlk chain.

	f_memset( (void *)&BlkHdr, 0, sizeof(	F_BLOCK_HEADER));

	for( uiOffset = 0;;)
	{
		// Read the block header

		if( RC_BAD( rc = m_pFileHdl641->Read( 
			BlkHdr.ui64FilePos + BlkHdr.uiBlockSize + uiOffset,
			sizeof( F_BLOCK_HEADER), &BlkHdr, &uiBytesRead)))
		{
			if( rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_IO_END_OF_FILE)
			{
				rc = NE_XFLM_OK;
				break;
			}

			goto Exit;
		}

		// Put the previous block out of fous.

		if( m_pCurRSBlk)
		{
			if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( NULL, m_uiBlkSize)))
			{
				goto Exit;
			}
		}

		// Allocate a new RSBlk and link into the result block list.
		
		if( (pNextRSBlk = f_new FResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM );
			goto Exit;
		}

		if( !m_pFirstRSBlk)
		{
			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}
		else
		{
			m_pCurRSBlk->m_pNext = pNextRSBlk;
			pNextRSBlk->m_pPrev = m_pCurRSBlk;
			m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}

		m_pCurRSBlk->Setup( &m_pFileHdl641, m_pCompare, m_uiEntrySize,
			BlkHdr.bFirstBlock, m_bDropDuplicates, !m_bInitialAdding);

		f_memcpy( (void *)&m_pCurRSBlk->m_BlockHeader, 
			(void *)&BlkHdr, sizeof(F_BLOCK_HEADER));

		// Process the block...

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}

		m_pCurRSBlk->adjustState( m_uiBlkSize);
		uiOffset = sizeof(F_BLOCK_HEADER);
	}

	// If the file is empty or just created, we won't have a RS Block yet.

	if( !m_pCurRSBlk)
	{
		// Allocate a new RSBlk

		if( (pNextRSBlk = f_new FResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM );
			goto Exit;
		}

		if( !m_pFirstRSBlk)
		{
			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}
		else
		{
			m_pCurRSBlk->m_pNext = pNextRSBlk;
			pNextRSBlk->m_pPrev = m_pCurRSBlk;
			m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		}

		m_pCurRSBlk->Setup(  &m_pFileHdl641, m_pCompare,
				m_uiEntrySize, m_bInitialAdding, m_bDropDuplicates,
				!m_bInitialAdding );

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}
	else
	{
		// Resize the file.

		if( RC_BAD(rc = m_pCurRSBlk->Truncate( (FLMBYTE *)m_szIoFilePath1)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Write the current block and close the file.  Call this function befor
		calling resetResultSet so that it can be reused.
*****************************************************************************/
RCODE XFLMAPI FResultSet::flushToFile()
{
	RCODE					rc = NE_XFLM_OK;

	flmAssert( m_bFile1Opened);

	// Flush to disk what ever we have.

	if( RC_BAD( rc = m_pCurRSBlk->Flush( m_bInitialAdding, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( NULL, m_uiBlkSize)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Interface to add a variable length entry to the result set.
Notes:	Public method used by application and by the internal sort
			and merge steps during finalize.  The user must never add an
			entry that is larger than the block size.
*****************************************************************************/
RCODE XFLMAPI FResultSet::addEntry(
	const void *	pvEntry,
	FLMUINT			uiEntryLength)				// If zero then entry is fixed length
{
	RCODE	rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( !m_bFinalizeCalled);

	rc = m_pCurRSBlk->AddEntry( (FLMBYTE *)pvEntry, uiEntryLength);

	// See if current block is full

	if( rc == NE_XFLM_EOF_HIT)
	{
		FResultSetBlk *			pNextRSBlk;
		F_64BitFileHandle **		ppFileHdl64;

		if( m_bInitialAdding && !m_bFile1Opened)
		{
			// Need to create and open the output file?
			// In a merge we may be working on the 2nd file and NOT the 1st.
			// There just isn't a better place to open the 1st file.

			if( RC_BAD(rc = OpenFile( &m_pFileHdl641)))
			{
				goto Exit;
			}
		}

		ppFileHdl64 = (m_bOutput2ndFile) ? &m_pFileHdl642 : &m_pFileHdl641;

		// Always flush to disk (TRUE) from here.

		if( RC_BAD( rc = m_pCurRSBlk->Flush( m_bInitialAdding, TRUE)))
		{
			goto Exit;
		}

		(void) m_pCurRSBlk->SetBuffer( NULL, m_uiBlkSize);

		// Adding the current block is complete so allocate a new
		// block object and link it into the list.
		// We must continue to use this same block buffer.

		// Allocate a new RSBlk and link into the result block list.

		if( (pNextRSBlk = f_new FResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM );
			goto Exit;
		}

		m_pCurRSBlk->m_pNext = pNextRSBlk;
		pNextRSBlk->m_pPrev = m_pCurRSBlk;
		m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		m_pCurRSBlk->Setup(  ppFileHdl64, m_pCompare,
				m_uiEntrySize, m_bInitialAdding, m_bDropDuplicates,
				!m_bInitialAdding );

		// Reset all of the buffer pointers and values.

		(void)m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlockBuf1Len);

		// Make the callback only during the merge phase.

		if( !m_bInitialAdding && m_pSortStatus)
		{
			if( m_ui64EstTotalUnits <= ++m_ui64UnitsDone )
			{
				m_ui64EstTotalUnits = m_ui64UnitsDone;
			}

			if( RC_BAD( rc = m_pSortStatus->reportSortStatus( m_ui64EstTotalUnits,
									m_ui64UnitsDone)))
			{
				goto Exit;
			}
		}

		// Add the entry again.  This call should never fail because of space.
		// If it does fail then the entry is larger than the buffer size.

		if( RC_BAD( rc = m_pCurRSBlk->AddEntry( 
			(FLMBYTE *)pvEntry, uiEntryLength)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			}

			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Done adding entries.  Sort all of the entries and perform a merge.
*****************************************************************************/
RCODE XFLMAPI FResultSet::finalizeResultSet(
	FLMUINT64 *		pui64TotalEntries)
{
	RCODE		rc = NE_XFLM_OK;
	FLMBOOL	bMergeSort;

	// Avoid being called more than once.

	flmAssert( !m_bFinalizeCalled);
	flmAssert( m_bSetupCalled );

	// Not a bug - but for future possibilities just check
	// if there is more than one block and if so then
	// the while() loop merge sort needs to be called.

	bMergeSort = (m_pFirstRSBlk != m_pLastRSBlk) ? TRUE : FALSE;

	// Force the write to disk if bMergeSort is TRUE.

	if( RC_BAD(rc = m_pCurRSBlk->Finalize( bMergeSort)))
	{
		goto Exit;
	}

	m_bInitialAdding = FALSE;

	// If the entries are in order fixup the block chain and we are done.

	if( m_bEntriesInOrder)
	{
		FResultSetBlk	*	pBlk;

		if( NumberOfBlockChains() > 1)
		{
			// Entries already in order - need to fixup the blocks.

			for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->m_pNext)
			{
				pBlk->m_BlockHeader.bFirstBlock = FALSE;
				pBlk->m_BlockHeader.bLastBlock = FALSE;
			}

			m_pFirstRSBlk->m_BlockHeader.bFirstBlock = TRUE;
			m_pLastRSBlk->m_BlockHeader.bLastBlock = TRUE;
			m_pCurRSBlk = NULL;
		}

		goto Exit;
	}

	// Compute total number of blocks.

	if( m_pSortStatus)
	{
		// Estimate total number of unit blocks to be written.

		FLMUINT64	ui64Units = NumberOfBlockChains();
		FLMUINT64	ui64Loops;

		m_ui64EstTotalUnits = 0;
		for( ui64Loops = ui64Units; ui64Loops > 1;
			  ui64Loops = (ui64Loops + 1) / 2 )
		{
			m_ui64EstTotalUnits += ui64Units;
		}
	}

	// Do the merge sort.
	// Keep looping until we have only one block in the result set list.

	while( NumberOfBlockChains() > 1)
	{
		// Allocate two more buffers.  Merge will open the 2nd file.
		// Exit will free these allocations and close one of the files.

		// Are the 2nd and 3rd buffers allocated?

		if( !m_pucBlockBuf2)
		{
			if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf2)))
			{
				goto Exit;
			}
		}

		if( !m_pucBlockBuf3)
		{
			if( RC_BAD( rc = f_alloc( m_uiBlkSize, &m_pucBlockBuf3)))
			{
				goto Exit;
			}
		}

		// Swap which file is selected as the output file.

		m_bOutput2ndFile = m_bOutput2ndFile ? FALSE : TRUE;

		// Here is the magical call that does all of the work!

		if( RC_BAD( rc = MergeSort()))
		{
			goto Exit;
		}
	}

Exit:

	// If we did a merge sort of multiple blocks then
	// free the first and second buffers and close one of the files.

	if( RC_BAD(rc))
	{
		f_free( &m_pucBlockBuf1);
		m_uiBlockBuf1Len = 0;
	}

	f_free( &m_pucBlockBuf2);
	f_free( &m_pucBlockBuf3);

	// Close the non-output opened file.  Close both on error.
	// If m_bFile2Opened then we did a merge - close one file

	if( m_bFile2Opened || RC_BAD( rc))
	{
		if( m_bOutput2ndFile || RC_BAD( rc))
		{
			if( m_bFile1Opened)
			{
				m_pFileHdl641->Close( TRUE);
				m_bFile1Opened = FALSE;
			}

			if( m_pFileHdl641)
			{
				m_pFileHdl641->Release();
				m_pFileHdl641 = NULL;
			}
		}

		if( !m_bOutput2ndFile || RC_BAD( rc))
		{
			if( m_bFile2Opened)
			{
				m_pFileHdl642->Close( TRUE);
				m_bFile2Opened = FALSE;
			}

			if( m_pFileHdl642)
			{
				m_pFileHdl642->Release();
				m_pFileHdl642 = NULL;
			}
		}
	}

	if( RC_OK(rc))
	{
		FLMUINT64			ui64Pos;
		FResultSetBlk *	pRSBlk;

		m_bFinalizeCalled = TRUE;
		m_bEntriesInOrder = TRUE;

		m_ui64TotalEntries = getTotalEntries();

		// Set the return value for total entries.

		if( pui64TotalEntries)
		{
			*pui64TotalEntries = m_ui64TotalEntries;
		}

		if( !m_ui64TotalEntries)
		{
			if( m_pCurRSBlk)
			{
				m_pCurRSBlk->Release();
			}

			m_pCurRSBlk = NULL;
			m_pFirstRSBlk = NULL;
			m_pLastRSBlk = NULL;
			f_free( &m_pucBlockBuf1);
			m_uiBlockBuf1Len = 0;
		}

		// Set the ui64BlkEntryPosition values in each block.

		for( ui64Pos = 0, pRSBlk = m_pFirstRSBlk;
				pRSBlk;
				pRSBlk = pRSBlk->m_pNext)
		{
			pRSBlk->m_ui64BlkEntryPosition = ui64Pos;
			ui64Pos += pRSBlk->m_BlockHeader.uiEntryCount;
		}

		// Resize the buffer to save space if only one block & in memory.

		if( m_pFirstRSBlk == m_pLastRSBlk && m_pCurRSBlk)
		{
			FLMBYTE *	pucNewBlk;
			FLMUINT		uiLen = m_pCurRSBlk->BytesUsedInBuffer();

			if( uiLen != m_uiBlockBuf1Len)
			{
				if( RC_OK( rc = f_alloc( uiLen, &pucNewBlk)))
				{
					f_memcpy( pucNewBlk, m_pucBlockBuf1, uiLen);
					f_free( &m_pucBlockBuf1);
					m_pucBlockBuf1 = pucNewBlk;
					m_uiBlockBuf1Len = uiLen;
				}
			}

			// Need to always do the SetBuffer, because it causes the
			// result set to get positioned.

			if( RC_OK( rc))
			{
				rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, uiLen);
			}
		}
	}

	// else on error finalize leaves the block list in an awful state.

	return( rc);
}

/*****************************************************************************
Desc:	Perform a Merge Sort on a list of result set blocks.  This new
		algorithm uses two files for the sort.  The end result may
		be one of the two files.  At the end of the sort all old result set
		block objects will be freed and only one result set block object
		will be left.  This RSBlk object will be used for reading the
		entries.  At this point there are at least 'N' result set block
		objects that will be merged into ('N'/2) block objects.
*****************************************************************************/
RCODE FResultSet::MergeSort( void)
{
	RCODE							rc = NE_XFLM_OK;
	FResultSetBlk *			pBlkList = NULL;
	FResultSetBlk *			pTempBlk;
	FResultSetBlk *			pLeftBlk;
	FResultSetBlk *			pRightBlk;
	F_64BitFileHandle **		ppFileHdl64;

	// Set output file and truncate it.

	// OpenFilex() Closes and creats a new file.
	// This is prefered over truncating the file because
	// if a database gets truncated we will be blamed.

	rc = (m_bOutput2ndFile)
			? OpenFile( &m_pFileHdl642)
			: OpenFile( &m_pFileHdl641);

	if( RC_BAD( rc))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	ppFileHdl64 = ( m_bOutput2ndFile ) ? &m_pFileHdl642 : &m_pFileHdl641;

	// Get the list to the RS blocks

	pBlkList = m_pFirstRSBlk;

	// Form an empty list to build.

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;

	// Read and UNION one or two blocks at a time getting rid of duplicates.
	// Reading the entries when performing a union of only one block
	// is a lot of work for nothing - but it simplifies the code.

	pTempBlk = pBlkList;
	while (pTempBlk)
	{
		pLeftBlk = pTempBlk;
		pRightBlk = pTempBlk->m_pNext;

		while( pRightBlk && !pRightBlk->m_BlockHeader.bFirstBlock)
		{
			pRightBlk = pRightBlk->m_pNext;
		}

		// Allocate a new result set block list and link into the new list.

		if( (m_pCurRSBlk = f_new FResultSetBlk) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		if( !m_pLastRSBlk)
		{
			// First time

			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk;
		}
		else
		{
			m_pLastRSBlk->m_pNext = m_pCurRSBlk;
			m_pCurRSBlk->m_pPrev = m_pLastRSBlk;
			m_pLastRSBlk = m_pCurRSBlk;
		}

		m_pCurRSBlk->Setup(  ppFileHdl64, m_pCompare,
				m_uiEntrySize, TRUE, m_bDropDuplicates, TRUE);

		// Output to block buffer 1

		(void)m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize );
		if( RC_BAD( rc = pLeftBlk->SetBuffer( m_pucBlockBuf2, m_uiBlkSize)))
		{
			goto Exit;
		}

		if( pRightBlk)
		{
			if( RC_BAD( rc = pRightBlk->SetBuffer( m_pucBlockBuf3, m_uiBlkSize)))
			{
				goto Exit;
			}
		}

		// pRightBlk may be NULL - will move left block to output.
		// Output leftBlk and rightBlk to the output block (m_pCurRSBlk)

		if( RC_BAD(rc = UnionBlkLists( pLeftBlk, pRightBlk)))
		{
			goto Exit;
		}

		// Setup for the next loop.

		pTempBlk = pRightBlk ? pRightBlk->m_pNext : NULL;
		while( pTempBlk && !pTempBlk->m_BlockHeader.bFirstBlock)
		{
			pTempBlk = pTempBlk->m_pNext;
		}
	}

Exit:

	// Free the working block list.

	pTempBlk = pBlkList;
	while( pTempBlk)
	{
		FLMUINT	uiTemp;

		pRightBlk = pTempBlk->m_pNext;
		uiTemp = pTempBlk->Release();
		flmAssert( uiTemp == 0);
		pTempBlk = pRightBlk;
	}

	return( rc);
}

/*****************************************************************************
Desc:	Return the Current entry reference in the result set.
*****************************************************************************/
RCODE XFLMAPI FResultSet::getCurrent(
	void *		pvBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bFinalizeCalled);

	if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
	}
	else
	{
		rc = m_pCurRSBlk->GetCurrent( (FLMBYTE *)pvBuffer, uiBufferLength,
										puiReturnLength );
	}

	return( rc);
}

/*****************************************************************************
Desc:	Return the next reference in the result set.  If the result set
		is not positioned then the first entry will be returned.
*****************************************************************************/
RCODE XFLMAPI FResultSet::getNext(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bFinalizeCalled);

	// Make sure we are positioned to a block.

	if( !m_pCurRSBlk)
	{
		m_pCurRSBlk = m_pFirstRSBlk;
		if( !m_pCurRSBlk)
		{
			rc = RC_SET( NE_XFLM_EOF_HIT);
			goto Exit;
		}

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	rc = m_pCurRSBlk->GetNext( (FLMBYTE *)pvBuffer, uiBufferLength,
										puiReturnLength );

	// Position to the next block?

	if( rc == NE_XFLM_EOF_HIT)
	{
		if( m_pCurRSBlk->m_pNext)
		{
			m_pCurRSBlk->SetBuffer( NULL);
			m_pCurRSBlk = m_pCurRSBlk->m_pNext;

			if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pCurRSBlk->GetNext( 
				(FLMBYTE *)pvBuffer, uiBufferLength, puiReturnLength)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the previous reference in the result set.  If the result set
		is not positioned then the last entry will be returned.
*****************************************************************************/
RCODE XFLMAPI FResultSet::getPrev(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	flmAssert( m_bFinalizeCalled);

	// Make sure we are positioned to a block.

	if( !m_pCurRSBlk)
	{
		if( (m_pCurRSBlk = m_pLastRSBlk) == NULL)
		{
			rc = RC_SET( NE_XFLM_BOF_HIT);
			goto Exit;
		}

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	rc = m_pCurRSBlk->GetPrev( (FLMBYTE *)pvBuffer, uiBufferLength,
										puiReturnLength );

	// Position to the previous block?

	if( rc == NE_XFLM_BOF_HIT)
	{
		if( m_pCurRSBlk->m_pPrev)
		{
			m_pCurRSBlk->SetBuffer( NULL);
			m_pCurRSBlk = m_pCurRSBlk->m_pPrev;
			if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pCurRSBlk->GetPrev( (FLMBYTE *)pvBuffer,
											uiBufferLength,
											puiReturnLength)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the first reference in the result set.
*****************************************************************************/
RCODE XFLMAPI FResultSet::getFirst(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	flmAssert( m_bFinalizeCalled);

	if( m_pCurRSBlk != m_pFirstRSBlk)
	{
		if( m_pCurRSBlk)
		{
			m_pCurRSBlk->SetBuffer( NULL);
		}

		m_pCurRSBlk = m_pFirstRSBlk;

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}
	else if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
		goto Exit;
	}

	if( RC_BAD( rc = m_pCurRSBlk->GetNext( (FLMBYTE *)pvBuffer,
		uiBufferLength, puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return the last reference in the result set.
*****************************************************************************/
RCODE XFLMAPI FResultSet::getLast(
	void *			pvBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc = NE_XFLM_OK;

	flmAssert( m_bFinalizeCalled);

	if( m_pCurRSBlk != m_pLastRSBlk)
	{
		if( m_pCurRSBlk)
		{
			m_pCurRSBlk->SetBuffer( NULL);
		}

		m_pCurRSBlk = m_pLastRSBlk;

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}
	else if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	if( RC_BAD( rc = m_pCurRSBlk->GetPrev( (FLMBYTE *) pvBuffer,
		uiBufferLength, puiReturnLength)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Find the matching entry in the result set using the compare routine.
		This does a binary search on the list of blocks.
*****************************************************************************/
RCODE XFLMAPI FResultSet::findMatch(
	const void *	pvMatchEntry,			// Entry to match
	FLMUINT			uiMatchEntryLength,	// Variable length of above entry
	void *			pvFoundEntry,			// (out) Entry to return
	FLMUINT *		puiFoundEntryLength)	// (out) Length of entry returned
{
	RCODE					rc = NE_XFLM_OK;
	FLMINT				iBlkCompare;		// 0 if key is/would be in block.
	FResultSetBlk *	pLowBlk;				// Used for locating block.
	FResultSetBlk *	pHighBlk;			// Low and High are exclusive.

	flmAssert( m_bFinalizeCalled);

	// If not positioned anywhere, position to the midpoint.
	// Otherwise, start on the current block we are on.

	if( !m_pCurRSBlk)
	{
		// m_pFirstRSBlk will be NULL if no entries.

		if( !m_pFirstRSBlk)
		{
			rc = RC_SET( NE_XFLM_NOT_FOUND);
			goto Exit;
		}

		if( m_pFirstRSBlk == m_pLastRSBlk)
		{
			m_pCurRSBlk = m_pFirstRSBlk;
		}
		else
		{
			m_pCurRSBlk = SelectMidpoint( m_pFirstRSBlk, m_pLastRSBlk, FALSE);
		}

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	// Set the exclusive low block and high block.

	pLowBlk = m_pFirstRSBlk;
	pHighBlk = m_pLastRSBlk;

	// Loop until the correct block is found.

	for( ;;)
	{
		// Two return value returned: rc and iBlkCompare.
		// FindMatch returns NE_XFLM_OK if the entry if found in the block.
		//	It returns NE_XFLM_NOT_FOUND if not found in the block.
		// uiCompare returns 0 if entry would be within the block.
		// otherwise < 0 if previous blocks should be checked
		// and > 0 if next blocks should be checked.

		rc = m_pCurRSBlk->FindMatch(
									(FLMBYTE *) pvMatchEntry, uiMatchEntryLength,
									(FLMBYTE *) pvFoundEntry, puiFoundEntryLength,
									&iBlkCompare );

		// Found match or should key be within the block.

		if( RC_OK(rc) || iBlkCompare == 0)
		{
			goto Exit;
		}

		if( iBlkCompare < 0)
		{
			// Done if the low block
			// Keep NE_XFLM_NOT_FOUND return code

			if( m_pCurRSBlk == pLowBlk)
			{
				goto Exit;
			}

			// Set the new high block

			pHighBlk = m_pCurRSBlk->m_pPrev;
		}
		else
		{
			// Done if we are at the high block
			// Keep the NE_XFLM_NOT_FOUND return code

			if( m_pCurRSBlk == pHighBlk)
			{
				goto Exit;
			}

			pLowBlk = m_pCurRSBlk->m_pNext;
		}

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( NULL)))
		{
			goto Exit;
		}

		m_pCurRSBlk = SelectMidpoint( pLowBlk, pHighBlk, FALSE);

		// Need to set the working buffer.

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Select the midpoint between two different blocks in a list.
		Entries should not be the same value.
*****************************************************************************/
FResultSetBlk * FResultSet::SelectMidpoint(
	FResultSetBlk *	pLowBlk,
	FResultSetBlk *	pHighBlk,
	FLMBOOL				bPickHighIfNeighbors)
{
	FLMUINT				uiCount;
	FResultSetBlk *	pTempBlk;

	// If the same then return.

	if( pLowBlk == pHighBlk)
	{
		pTempBlk = pLowBlk;
		goto Exit;
	}

	// Check if neighbors and use the boolean flag.

	if( pLowBlk->m_pNext == pHighBlk)
	{
		pTempBlk = (FResultSetBlk *)(bPickHighIfNeighbors
											 ? pHighBlk
											 : pLowBlk);
		goto Exit;
	}

	// Count the total blocks exclusive between low and high and add one.
	// Check pTempBlk against null to not crash.

	for( pTempBlk = pLowBlk, uiCount = 1;
		  pTempBlk && (pTempBlk != pHighBlk);
		  uiCount++)
	{
		pTempBlk = pTempBlk->m_pNext;
	}

	// Check for implementation error - pTempBlk is NULL and handle.

	if( !pTempBlk)
	{
		flmAssert( 0);
		pTempBlk = pLowBlk;
		goto Exit;
	}

	// Loop to the middle item
	// Divide count by 2

	uiCount >>= 1;
	for( pTempBlk = pLowBlk; uiCount > 0; uiCount--)
	{
		pTempBlk = pTempBlk->m_pNext;
	}

Exit:

	return( pTempBlk);
}

/*****************************************************************************
Desc:	Set the current entry position.
*****************************************************************************/
RCODE XFLMAPI FResultSet::setPosition(
	FLMUINT64		ui64Position)
{
	RCODE					rc = NE_XFLM_OK;
	FResultSetBlk *	pInitialBlk = m_pCurRSBlk;

	flmAssert( m_bFinalizeCalled);

	if( ui64Position == RS_POSITION_NOT_SET)
	{
		// Set out of focus

		if( m_pCurRSBlk)
		{
			if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( NULL)))
			{
				goto Exit;
			}
		}

		m_pCurRSBlk = NULL;
		goto Exit;
	}

	if( !m_pCurRSBlk)
	{
		m_pCurRSBlk = m_pFirstRSBlk;
	}

	// Check for empty result set.

	if( !m_pCurRSBlk)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	if( ui64Position < m_pCurRSBlk->m_ui64BlkEntryPosition)
	{
		// Go backwards looking for the correct block.

		do
		{
			m_pCurRSBlk = m_pCurRSBlk->m_pPrev;
			flmAssert( m_pCurRSBlk);
		}
		while( ui64Position < m_pCurRSBlk->m_ui64BlkEntryPosition);
	}
	else if( ui64Position >= m_pCurRSBlk->m_ui64BlkEntryPosition +
								  m_pCurRSBlk->m_BlockHeader.uiEntryCount)
	{
		// Go forward looking for the correct block.

		do
		{
			if( !m_pCurRSBlk->m_pNext)
			{
				// Will set rc to EOF in SetPosition below.

				break;
			}

			m_pCurRSBlk = m_pCurRSBlk->m_pNext;
		}
		while( ui64Position >= m_pCurRSBlk->m_ui64BlkEntryPosition +
									m_pCurRSBlk->m_BlockHeader.uiEntryCount);
	}

	// Need working buffer out of focus.

	if( pInitialBlk != m_pCurRSBlk)
	{
		if( pInitialBlk)
		{
			if( RC_BAD( rc = pInitialBlk->SetBuffer( NULL)))
			{
				goto Exit;
			}
		}

		// Need working buffer into focus.

		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pucBlockBuf1, m_uiBlkSize)))
		{
			goto Exit;
		}
	}

	// Now we are positioned to the correct block.

	if( RC_BAD( rc = m_pCurRSBlk->SetPosition( ui64Position)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Return a pointer to the next entry in the list.
*****************************************************************************/
RCODE FResultSet::GetNextPtr(
	FResultSetBlk **	ppCurBlk,
	FLMBYTE **			ppucBuffer,
	FLMUINT *			puiReturnLength)
{
	RCODE					rc = NE_XFLM_OK;
	FResultSetBlk *	pCurBlk = *ppCurBlk;
	FResultSetBlk *	pNextBlk;
	FLMBYTE *			pucBuffer;

	flmAssert( pCurBlk);

	while( RC_BAD( rc = pCurBlk->GetNextPtr( ppucBuffer, puiReturnLength)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			if( pCurBlk->m_pNext)
			{
				pNextBlk = pCurBlk->m_pNext;
				if( !pNextBlk->m_BlockHeader.bFirstBlock)
				{
					pucBuffer = pCurBlk->m_pucBlockBuf;
					pCurBlk->SetBuffer( NULL );
					pCurBlk = pNextBlk;
					if( RC_BAD( rc = pCurBlk->SetBuffer( pucBuffer, m_uiBlkSize)))
					{
						goto Exit;
					}
					*ppCurBlk = pCurBlk;
					continue;
				}
			}
		}

		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Union two block lists into a result output list.  This may be
			called to union two result sets or to perform the initial merge-sort
			on a create result set.

			Performing an N-way merge would be fast when we have over 10K
			of entries.  However, the code is more complex.
*****************************************************************************/
RCODE FResultSet::UnionBlkLists(
	FResultSetBlk *	pLeftBlk,
	FResultSetBlk *	pRightBlk)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE *	pucLeftEntry;
	FLMBYTE *	pucRightEntry;
	FLMUINT		uiLeftLength;
	FLMUINT		uiRightLength;

	// If no right block then copy all of the items from the left block
	// to the output block.  We could optimize this in the future.

	if( !pRightBlk)
	{
		rc = CopyRemainingItems( pLeftBlk);
		goto Exit;
	}

	// Now the fun begins.  Read entries from both lists and union
	// while checking the order of the entries.

	if( RC_BAD( rc = GetNextPtr( &pLeftBlk, &pucLeftEntry, &uiLeftLength)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			rc = CopyRemainingItems( pRightBlk);
		}

		goto Exit;
	}

	if( RC_BAD( rc = GetNextPtr( &pRightBlk, &pucRightEntry, &uiRightLength)))
	{
		if( rc == NE_XFLM_EOF_HIT)
		{
			rc = CopyRemainingItems( pLeftBlk);
		}

		goto Exit;
	}

	for (;;)
	{
		FLMINT	iCompare;

		if( RC_BAD(rc = m_pCompare->compare( pucLeftEntry, uiLeftLength,
				pucRightEntry, uiRightLength, &iCompare )))
		{
			goto Exit;
		}

		if( iCompare < 0)
		{
			// Take the left item.

			if( RC_BAD(rc = addEntry( pucLeftEntry, uiLeftLength)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = GetNextPtr( &pLeftBlk, 
				&pucLeftEntry, &uiLeftLength)))
			{
				if( rc != NE_XFLM_EOF_HIT)
				{
					goto Exit;
				}

				if( RC_BAD( rc = addEntry( pucRightEntry, uiRightLength)))
				{
					goto Exit;
				}

				// Left entries are done - read all of the right entries.

				rc = CopyRemainingItems( pRightBlk);
				goto Exit;
			}
		}
		else
		{
			// If equals then drop the right item and continue comparing left.
			// WARNING: Don't try to optimize for equals because when one
			// list runs out the remaining duplicate entries must be dropped.
			// Continuing to compare the duplicate item is the correct way.

			if( iCompare > 0 || !m_bDropDuplicates)
			{
				// Take the right item.

				if( RC_BAD(rc = addEntry( pucRightEntry, uiRightLength)))
				{
					goto Exit;
				}
			}

			if( RC_BAD(rc = GetNextPtr( &pRightBlk, 
				&pucRightEntry, &uiRightLength)))
			{
				if( rc != NE_XFLM_EOF_HIT)
				{
					goto Exit;
				}

				if( RC_BAD(rc = addEntry( pucLeftEntry, uiLeftLength)))
				{
					goto Exit;
				}

				// Right entries are done - read all of the left entries.

				rc = CopyRemainingItems( pLeftBlk);
				goto Exit;
			}
		}
	}

Exit:

	if( RC_OK( rc))
	{
		// Flush out the output entries.

		rc = m_pCurRSBlk->Finalize( TRUE );
		m_pCurRSBlk->SetBuffer( NULL);
		m_pCurRSBlk = NULL;

		if( m_pSortStatus)
		{
			RCODE	rc2;
			
			++m_ui64UnitsDone;
			if( RC_BAD( rc2 = m_pSortStatus->reportSortStatus( m_ui64EstTotalUnits,
											m_ui64UnitsDone)))
			{
				if( RC_OK( rc))
				{
					rc = rc2;
				}
			}
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:	Copy the remaining items from a block list to the output.
*****************************************************************************/
RCODE FResultSet::CopyRemainingItems(
	FResultSetBlk *	pCurBlk)
{
	RCODE					rc;
	FLMBYTE *			pucEntry;
	FLMUINT				uiLength;

	while( RC_OK( rc = GetNextPtr( &pCurBlk, &pucEntry, &uiLength)))
	{
		if( RC_BAD( rc = addEntry( pucEntry, uiLength)))
		{
			goto Exit;
		}
	}

	if( rc == NE_XFLM_EOF_HIT)
	{
		rc = NE_XFLM_OK;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:	Closes and deletes one of two files.
*****************************************************************************/
void FResultSet::CloseFile(
	F_64BitFileHandle **		ppFileHdl64,
	FLMBOOL						bDelete)
{
	if( ppFileHdl64 == &m_pFileHdl641)
	{
		if( m_bFile1Opened)
		{
			m_pFileHdl641->Close( bDelete);
			m_bFile1Opened = FALSE;
		}

		if( m_pFileHdl641)
		{
			m_pFileHdl641->Release();
			m_pFileHdl641 = NULL;
		}
	}
	else
	{
		if( m_bFile2Opened)
		{
			m_pFileHdl642->Close( TRUE);
			m_bFile2Opened = FALSE;
		}

		if( m_pFileHdl642)
		{
			m_pFileHdl642->Release();
			m_pFileHdl642 = NULL;
		}
	}
}

/*****************************************************************************
Desc:	Close the file if previously opened and creates the file.
*****************************************************************************/
RCODE FResultSet::OpenFile(
	F_64BitFileHandle **		ppFileHdl64)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL *	pbFileOpened;
	char *		pszDirPath;

	// Will close and delete if opened, else will do nothing.

	CloseFile( ppFileHdl64);

	if( ppFileHdl64 == &m_pFileHdl641)
	{
		pbFileOpened = &m_bFile1Opened;
		pszDirPath = &m_szIoFilePath1 [0];
	}
	else
	{
		pbFileOpened = &m_bFile2Opened;
		pszDirPath = &m_szIoFilePath2 [0];
	}

	f_strcpy( pszDirPath, m_szIoDefaultPath);

	if( (*ppFileHdl64 = f_new F_64BitFileHandle) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = (*ppFileHdl64)->CreateUnique( pszDirPath,
										FRSET_FILENAME_EXTENSION)))
	{
		(*ppFileHdl64)->Release();
		*ppFileHdl64 = NULL;
		goto Exit;
	}

	*pbFileOpened = TRUE;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Create and empty data vector and return it's interface...
*****************************************************************************/
RCODE XFLMAPI F_DbSystem::createIFResultSet(
	IF_ResultSet **	ppResultSet)
{
	RCODE	rc = NE_XFLM_OK;

	if( (*ppResultSet = f_new FResultSet) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
Exit:

	return( rc);
}
