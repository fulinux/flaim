//-------------------------------------------------------------------------
// Desc:	Result sets
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
// $Id: frset.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

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
***** 
** 	Setup/Shutdown Routines
***** 
*****************************************************************************/

FResultSet::FResultSet()
{
	// Let's just initialize all member variables.

	m_fnCompare = NULL;					// Setup
	m_UserValue = (void *) 0;		// Setup
	m_fnCallback = NULL;
	m_CallbackInfo.UserValue = (void *)0;
	m_CallbackInfo.ui64EstTotalUnits = 0;
	m_CallbackInfo.ui64UnitsDone = 0;

	m_uiEntrySize = 0;
	m_ui64TotalEntries = 0;
	m_pCurRSBlk = m_pFirstRSBlk = m_pLastRSBlk = NULL;
	
	f_memset( m_szDefaultPath, 0, F_PATH_MAX_SIZE);
	m_pBlockBuf1 = m_pBlockBuf2 = m_pBlockBuf3 = NULL;
	m_uiBlockBuf1Len = 0;
	m_bFile1Opened = m_bFile2Opened = FALSE;
	m_pFileHdl641 = m_pFileHdl642 = NULL;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bFinalizeCalled = FALSE;
	m_bSetupCalled = FALSE;
}


FResultSet::~FResultSet()
{
	FResultSetBlk *pCurRSBlk;
	FResultSetBlk *pNextRSBlk;
	
	// Free up the result set block chain.
	
	for( pCurRSBlk = m_pFirstRSBlk; pCurRSBlk; pCurRSBlk = pNextRSBlk )
	{
		FLMUINT		uiCount;

		pNextRSBlk = pCurRSBlk->GetNext();
		uiCount = pCurRSBlk->Release();
		flmAssert( uiCount == 0);
	}
	// Set list to NULL for debugging in memory.
	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;

	// Free up all of the block buffers in the list.

	f_free( &m_pBlockBuf1);
	f_free( &m_pBlockBuf2);
	f_free( &m_pBlockBuf3);

	// Close all opened files
	
	CloseFile( &m_pFileHdl641 );
	CloseFile( &m_pFileHdl642 );
}


/****************************************************************************
Public:  reset
Desc:    Reset the result set so it can be reused.
****************************************************************************/
RCODE FResultSet::reset( void)
{
	RCODE					rc = FERR_OK;
	FResultSetBlk *	pCurRSBlk;
	FResultSetBlk *	pNextRSBlk;
	
	// Free up the result set block chain - except for the first one.
	
	for( pCurRSBlk = m_pFirstRSBlk; pCurRSBlk; pCurRSBlk = pNextRSBlk )
	{
		FLMUINT		uiCount;

		pNextRSBlk = pCurRSBlk->GetNext();
		if (pCurRSBlk != m_pFirstRSBlk)
		{
			uiCount = pCurRSBlk->Release();
			flmAssert( uiCount == 0);
		}
	}

	// Free up all of the block buffers in the list, except for the first one.

	f_free( &m_pBlockBuf2);
	f_free( &m_pBlockBuf3);

	// We want a buffer that is at least RSBLK_BLOCK_SIZE.

	if (!m_pBlockBuf1 || m_uiBlockBuf1Len < RSBLK_BLOCK_SIZE)
	{
		if (m_pBlockBuf1)
		{
			f_free( &m_pBlockBuf1);
		}
		if( RC_BAD( rc = f_calloc( RSBLK_BLOCK_SIZE, &m_pBlockBuf1)))
		{
			goto Exit;
		}
		m_uiBlockBuf1Len = RSBLK_BLOCK_SIZE;
	}

	// Close all opened files
	
	CloseFile( &m_pFileHdl641 );
	CloseFile( &m_pFileHdl642 );
	m_bFile1Opened = m_bFile2Opened = FALSE;
	m_pFileHdl641 = m_pFileHdl642 = NULL;

	// Reset some other variables

	m_fnCallback = NULL;
	m_CallbackInfo.UserValue = (void *)0;
	m_CallbackInfo.ui64EstTotalUnits = 0;
	m_CallbackInfo.ui64UnitsDone = 0;
	m_ui64TotalEntries = 0;
	m_bOutput2ndFile = FALSE;
	m_bInitialAdding = TRUE;
	m_bEntriesInOrder = m_bAppAddsInOrder;
	m_bFinalizeCalled = FALSE;

	// If we don't have a block, allocate it.  Otherwise
	// reset the one we have left.

	if (!m_pFirstRSBlk)
	{
		if ((m_pFirstRSBlk = f_new FResultSetBlk) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	else
	{
		m_pFirstRSBlk->reset();
	}
	m_pLastRSBlk = m_pCurRSBlk = m_pFirstRSBlk;

	(void)m_pFirstRSBlk->Setup( &m_pFileHdl641, m_fnCompare, m_UserValue,
		m_uiEntrySize, RSBLK_IS_FIRST_IN_LIST, m_bDropDuplicates,
		m_bEntriesInOrder);
	(void) m_pFirstRSBlk->SetBuffer( m_pBlockBuf1, m_uiBlockBuf1Len);
Exit:
	return( rc);
}


/****************************************************************************
Public:  Setup
Desc:    Setup the result set with all of the needed input values.
			This method must only be called once.  
Ret:     FERR_OK        - Created FERR_OKfully
			WERR_MEM       - Allocation error
Notes:	Handles all error conditions.
****************************************************************************/

RCODE FResultSet::Setup(
	const char *			pszIoPath,
	RSET_COMPARE_FUNC_p	fnCompare,
	void *	      		UserValue, 
	FLMUINT					uiEntrySize,
	FLMBOOL					bDropDuplicates,
	FLMBOOL					bEntriesInOrder)
{
	RCODE          		rc = FERR_OK;
	FLMBOOL					bNewBlock = FALSE;
	FLMBOOL					bNewBuffer = FALSE;

	flmAssert( !m_bSetupCalled );
	flmAssert( uiEntrySize <= MAX_FIXED_ENTRY_SIZE );

	// Perform all of the allocations first.

	m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = f_new FResultSetBlk;

	// Allocation Error?
	
	if( ! m_pCurRSBlk )
	{
		rc = RC_SET( FERR_MEM );
		goto Exit;
	}
	bNewBlock = TRUE;
	m_pCurRSBlk->Setup( &m_pFileHdl641, fnCompare, UserValue,
			uiEntrySize, RSBLK_IS_FIRST_IN_LIST, bDropDuplicates, bEntriesInOrder );

	// Allocate only the first buffer - other buffers only used in merge.

	if( RC_BAD( rc = f_calloc( RSBLK_BLOCK_SIZE, &m_pBlockBuf1)))
		goto Exit;
	m_uiBlockBuf1Len = RSBLK_BLOCK_SIZE;
	bNewBuffer = TRUE;	
	(void) m_pCurRSBlk->SetBuffer( m_pBlockBuf1);

	// Set the input variables.

	if( pszIoPath)
	{
		f_strcpy( m_szDefaultPath, pszIoPath);
	}
	
	m_fnCompare = fnCompare;
	m_UserValue = UserValue;
	m_uiEntrySize = uiEntrySize;
	m_bDropDuplicates = bDropDuplicates;
	m_bEntriesInOrder = m_bAppAddsInOrder = bEntriesInOrder;

Exit:

	// Free allocations on any error
	
	if( RC_BAD(rc))
	{
		if( bNewBlock)
		{
			m_pCurRSBlk->Release();
			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk = NULL;
		}
		if( bNewBuffer)
		{
			f_free( &m_pBlockBuf1);
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
***** 
** 	Methods for Building a Result Set
***** 
*****************************************************************************/

/****************************************************************************
Public:	AddEntry
Desc:		Interface to add a variable length entry to the result set.  
Ret:     RCODE - I/O error
Notes:	Public method used by application and by the internal sort
			and merge steps during finalize.  The user must never add an
			entry that is larger than the block size. 
****************************************************************************/

RCODE FResultSet::AddEntry(
	void *		pEntry,
	FLMUINT		uiEntryLength)				// If zero then entry is fixed length
{
	RCODE			rc;

	flmAssert( m_bSetupCalled );
	flmAssert( !m_bFinalizeCalled );
	
	rc = m_pCurRSBlk->AddEntry( (FLMBYTE *) pEntry, uiEntryLength );

	if( rc == FERR_EOF_HIT )					// Current block is full.
	{
		FResultSetBlk *			pNextRSBlk;
		F_64BitFileHandle **		ppFileHdl64;

		if( m_bInitialAdding && !m_bFile1Opened )
		{
			// Need to create and open the output file?
			// In a merge we may be working on the 2nd file and NOT the 1st.
			// There just isn't a better place to open the 1st file.

			if( RC_BAD(rc = OpenFile( &m_pFileHdl641 )))
				goto Exit;
		}
		ppFileHdl64 = ( m_bOutput2ndFile ) ? &m_pFileHdl642 : &m_pFileHdl641;

		// Always flush to disk (TRUE) from here.

		if( RC_BAD( rc = m_pCurRSBlk->Flush( m_bInitialAdding, TRUE )))
			goto Exit;
		(void) m_pCurRSBlk->SetBuffer( NULL );

		// Adding the current block is complete so allocate a new 
		// block object and link it into the list.
		// We must continue to use this same block buffer.

		// Allocate a new RSBlk and link into the result block list.

		pNextRSBlk = f_new FResultSetBlk;
		if( ! pNextRSBlk )
		{
			rc = RC_SET( FERR_MEM );
			goto Exit;
		}
		m_pCurRSBlk->SetNext( pNextRSBlk );
		pNextRSBlk->SetPrev( m_pCurRSBlk );
		m_pLastRSBlk = m_pCurRSBlk = pNextRSBlk;
		m_pCurRSBlk->Setup(  ppFileHdl64, m_fnCompare, 
				m_UserValue, m_uiEntrySize, m_bInitialAdding, m_bDropDuplicates, 
				!m_bInitialAdding );

		// Reset all of the buffer pointers and values.

		(void) m_pCurRSBlk->SetBuffer( m_pBlockBuf1 );

		// Make the callback only during the merge phase.
	
		if( !m_bInitialAdding && m_fnCallback )
		{
			if( m_CallbackInfo.ui64EstTotalUnits <= 
					++m_CallbackInfo.ui64UnitsDone )
			{
				m_CallbackInfo.ui64EstTotalUnits = 
					m_CallbackInfo.ui64UnitsDone;
			}		
			(void) m_fnCallback( &m_CallbackInfo );
		}

		// Add the entry again.  This call should never fail because of space.
		// If it does fail then the entry is larger than the buffer size.

		if( RC_BAD( rc = m_pCurRSBlk->AddEntry( (FLMBYTE *) pEntry, uiEntryLength )))
		{
			if( rc == FERR_EOF_HIT )
			{
				flmAssert( FALSE );					// Force assert for testing
				rc = RC_SET( FERR_FAILURE );
			}
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Public:	Finalize
Desc:		Done adding entries.  Sort all of the entries and perform a merge.
Ret:     WERR_OK
Notes:	This algorithm is tricky and there are many variations that we
			could make to it.  We have tried a lot of variations.  In the
			future this method may be replaced by CreateIterator().
Caution: On any error the result set is in a bad state and should be dumped.
****************************************************************************/

RCODE FResultSet::Finalize(
	FLMUINT64 *		pui64TotalEntries)	// (OUT) Returns total number of entries.
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bMergeSort;

	// Avoid being called more than once.
	flmAssert( !m_bFinalizeCalled);
	flmAssert( m_bSetupCalled );

	// Not a bug - but for future possibilities just check
	// if there is more than one block and if so then
	// the while() loop merge sort needs to be called.

	bMergeSort = (m_pFirstRSBlk != m_pLastRSBlk) ? TRUE : FALSE;

	// Force the write to disk if bMergeSort is TRUE.

	if( RC_BAD(rc = m_pCurRSBlk->Finalize( bMergeSort )))	
		goto Exit;

	m_bInitialAdding = FALSE;

	// If the entries are in order fixup the block chain and we are done.
	if( m_bEntriesInOrder )
	{
		FResultSetBlk	*	pBlk;
		
		if( NumberOfBlockChains() > (FLMUINT64)1 )
		{
			// Entries already in order - need to fixup the blocks.
			for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->GetNext() )
			{
				pBlk->SetFirstInChain( FALSE);
				pBlk->SetLastInChain( FALSE );
			}
			m_pFirstRSBlk->SetFirstInChain( TRUE );
			m_pLastRSBlk->SetLastInChain( TRUE );
			m_pCurRSBlk = NULL;
		}
		goto Exit;
	}

	// Compute total number of blocks.
	
	if( m_fnCallback)
	{
		// Estimate total number of unit blocks to be written.
		FLMUINT64	ui64Units = NumberOfBlockChains();
		FLMUINT64	ui64Loops;
		
		m_CallbackInfo.ui64EstTotalUnits = 0;
		for( ui64Loops = ui64Units; ui64Loops > (FLMUINT64)1;
			ui64Loops = (ui64Loops + (FLMUINT64)1) / (FLMUINT64)2 )
		{
			m_CallbackInfo.ui64EstTotalUnits += ui64Units;
		}
	}

	// Do the merge sort.
	// Keep looping until we have only one block in the result set list.

	while( NumberOfBlockChains() > (FLMUINT64)1)
	{
		// Allocate two more buffers.  Merge will open the 2nd file.
		// Exit will free these allocations and close one of the files.

		// Are the 2nd and 3rd buffers allocated?

		if( !m_pBlockBuf2)
		{
			if( RC_BAD( rc = f_calloc( RSBLK_BLOCK_SIZE, &m_pBlockBuf2)))
				goto Exit;
		}
		if( !m_pBlockBuf3)
		{
			if( RC_BAD( rc = f_calloc( RSBLK_BLOCK_SIZE, &m_pBlockBuf3)))
				goto Exit;
		}
		// Swap which file is selected as the output file.

		m_bOutput2ndFile = m_bOutput2ndFile ? FALSE : TRUE;

		// Here is the magical call that does all of the work!

		if( RC_BAD( rc = MergeSort()))
			goto Exit;
	}

Exit:

	// If we did a merge sort of multiple blocks then
	// free the first and second buffers and close one of the files.

	if( RC_BAD(rc))
	{
		f_free( &m_pBlockBuf1);
		m_uiBlockBuf1Len = 0;
	}
	f_free( &m_pBlockBuf2);
	f_free( &m_pBlockBuf3);

	// Close the non-output opened file.  Close both on error.

	// If m_bFile2Opened then we did a merge - close one file

	if( m_bFile2Opened || RC_BAD(rc))
	{
		if( m_bOutput2ndFile || RC_BAD(rc) )
		{
			if( m_bFile1Opened )
			{
				m_pFileHdl641->Close( TRUE);
				m_bFile1Opened = FALSE;
			}
			if (m_pFileHdl641)
			{
				m_pFileHdl641->Release();
				m_pFileHdl641 = NULL;
			}
		}
		if( !m_bOutput2ndFile || RC_BAD(rc) )
		{
			if( m_bFile2Opened )
			{
				m_pFileHdl642->Close( TRUE);
				m_bFile2Opened = FALSE;
			}

			if (m_pFileHdl642)
			{
				m_pFileHdl642->Release();
				m_pFileHdl642 = NULL;
			}
		}
	}

	if( RC_OK(rc))
	{
		FLMUINT			 uiPos;
		FResultSetBlk * pRSBlk;

		m_bFinalizeCalled = TRUE;		// Used for asserts.
		m_bEntriesInOrder = TRUE;
	
		m_ui64TotalEntries = GetTotalEntries();
		
		// Set the return value for total entries.

		if( pui64TotalEntries)
		{
			*pui64TotalEntries = m_ui64TotalEntries;
		}

		if( !m_ui64TotalEntries)
		{
			if( m_pCurRSBlk )
				m_pCurRSBlk->Release();
			m_pCurRSBlk = m_pFirstRSBlk = m_pLastRSBlk = NULL;
			f_free( &m_pBlockBuf1);
			m_uiBlockBuf1Len = 0;
		}

		// Set the uiBlkEntryPosition values in each block.

		for( uiPos = 0, pRSBlk = m_pFirstRSBlk; 
				pRSBlk; 
				pRSBlk = pRSBlk->GetNext() )
		{
			pRSBlk->SetInitialPosition( uiPos );
			uiPos += pRSBlk->GetNumberOfEntries();
		}

		// Resize the buffer to save space if only one block & in memory.
		if( (m_pFirstRSBlk == m_pLastRSBlk) && m_pCurRSBlk )
		{
			FLMBYTE *		pNewBlk;
			FLMUINT			uiLen = m_pCurRSBlk->BytesUsedInBuffer();

			if (uiLen != m_uiBlockBuf1Len)
			{
				rc = f_alloc( uiLen, &pNewBlk);
				if( RC_OK(rc))
				{
					f_memcpy( pNewBlk, m_pBlockBuf1, uiLen);
					f_free( &m_pBlockBuf1);
					m_pBlockBuf1 = pNewBlk;
					m_uiBlockBuf1Len = uiLen;
				}
			}

			// Need to always do the SetBuffer, because it causes the
			// result set to get positioned.

			if (RC_OK( rc))
			{
				rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1, uiLen);
			}
		}
	}
	// else on error finalize leaves the block list in an awful state.

	return( rc);
}

/****************************************************************************
Desc:		Perform a Merge Sort on a list of result set blocks.  This new
			algorithm uses two files for the sort.  The end result may
			be one of the two files.  At the end of the sort all old result set
			block objects will be freed and only one result set block object
			will be left.  This RSBlk object will be used for reading the
			entries.  At this point there are at least 'N' result set block
			objects that will be merged into ('N'/2) block objects.
****************************************************************************/
RCODE FResultSet::MergeSort()
{
	RCODE							rc = FERR_OK;
	FResultSetBlk	*			pBlkList = NULL,
						*			pTempBlk,
						*			pLeftBlk,
						*			pRightBlk;
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
		flmAssert( 0);
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
	while( pTempBlk )
	{
		pLeftBlk = pTempBlk;
		pRightBlk = pTempBlk->GetNext();		// May be NULL so watch out!
		while( pRightBlk && (!pRightBlk->IsFirstInChain()) )
		{
			pRightBlk = pRightBlk->GetNext();
		}
		// Allocate a new result set block list and link into the new list.
	
		m_pCurRSBlk = f_new FResultSetBlk;
		if( ! m_pCurRSBlk )
		{
			rc = RC_SET( FERR_MEM );
			goto Exit;
		}
		if( ! m_pLastRSBlk )		// First Time?
		{
			m_pFirstRSBlk = m_pLastRSBlk = m_pCurRSBlk;
		}
		else
		{
			m_pLastRSBlk->SetNext( m_pCurRSBlk );
			m_pCurRSBlk->SetPrev( m_pLastRSBlk);
			m_pLastRSBlk = m_pCurRSBlk;
		}
		m_pCurRSBlk->Setup(  ppFileHdl64, m_fnCompare, m_UserValue,
				m_uiEntrySize, RSBLK_IS_FIRST_IN_LIST, m_bDropDuplicates,
				RSBLK_ENTRIES_IN_ORDER );
		
		// Output to block buffer 1
		(void) m_pCurRSBlk->SetBuffer( m_pBlockBuf1 );
		if( RC_BAD( rc = pLeftBlk->SetBuffer( m_pBlockBuf2 )))
			goto Exit;
		if( pRightBlk)
		{
			if( RC_BAD( rc = pRightBlk->SetBuffer( m_pBlockBuf3 )))
				goto Exit;
		}

		// pRightBlk may be NULL - will move left block to output.
		// Output leftBlk and rightBlk to the output block (m_pCurRSBlk)

		if( RC_BAD(rc = UnionBlkLists( pLeftBlk, pRightBlk )))
			goto Exit;

		// Setup for the next loop.
		pTempBlk = pRightBlk ? pRightBlk->GetNext() : NULL;
		while( pTempBlk && (!pTempBlk->IsFirstInChain()) )
		{
			pTempBlk = pTempBlk->GetNext();
		}
	}
Exit:

	// Free the working block list.
	pTempBlk = pBlkList;
	while( pTempBlk )
	{	
		FLMUINT		uiTemp;

		pRightBlk = pTempBlk->GetNext();
		uiTemp = pTempBlk->Release();
		flmAssert( uiTemp == 0);
		pTempBlk = pRightBlk;
	}

	return( rc);
}


/*****************************************************************************
***** 
** 	Reading Result Set Entries
***** 
*****************************************************************************/


/****************************************************************************
Public:	GetCurrent
Desc:		Return the Current entry reference in the result set.
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_NOT_FOUND	- Not positioned anywhere in the result set.
			FERR_EOF_HIT		- Positioned past the last entry
			FERR_BOF_HIT		- Positioned before the first entry.
****************************************************************************/

RCODE FResultSet::GetCurrent(
	void *		vpBuffer,
	FLMUINT		uiBufferLength,
	FLMUINT *	puiReturnLength)
{
	RCODE			rc;

	flmAssert( m_bFinalizeCalled );

	if( !m_pCurRSBlk )
	{
		rc = RC_SET( FERR_NOT_FOUND );
	}
	else
	{
		rc = m_pCurRSBlk->GetCurrent( (FLMBYTE *) vpBuffer, uiBufferLength, 
										puiReturnLength );
	}
	return( rc);
}

/****************************************************************************
Public:	GetNext
Desc:		Return the next reference in the result set.  If the result set
			is not positioned then the first entry will be returned.
			
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_EOF_HIT		- Positioned past the last entry
****************************************************************************/

RCODE FResultSet::GetNext(
	void *			vpBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	flmAssert( m_bFinalizeCalled );

	// Make sure we are positioned to a block.
	if( ! m_pCurRSBlk )
	{
		m_pCurRSBlk = m_pFirstRSBlk;
		if( ! m_pCurRSBlk )
		{
			rc = RC_SET( FERR_EOF_HIT );
			goto Exit;
		}
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}

	rc = m_pCurRSBlk->GetNext( (FLMBYTE *) vpBuffer, uiBufferLength, 
										puiReturnLength );
	
	// Position to the next block?
	if( rc == FERR_EOF_HIT )
	{
		if( m_pCurRSBlk->GetNext() != NULL )
		{
			m_pCurRSBlk->SetBuffer( NULL );
			m_pCurRSBlk = m_pCurRSBlk->GetNext();
			if( RC_BAD( rc= m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
				goto Exit;
			if( RC_BAD( rc = m_pCurRSBlk->GetNext( (FLMBYTE *) vpBuffer, uiBufferLength, 
											puiReturnLength )))
				goto Exit;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Public:	GetPrev
Desc:		Return the previous reference in the result set.  If the result set
			is not positioned then the last entry will be returned.
			
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_BOF_HIT		- Positioned before the first entry
****************************************************************************/

RCODE FResultSet::GetPrev(
	void *			vpBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	flmAssert( m_bFinalizeCalled );

	// Make sure we are positioned to a block.
	if( !m_pCurRSBlk )
	{
		m_pCurRSBlk = m_pLastRSBlk;
		if( ! m_pCurRSBlk )
		{
			rc = RC_SET( FERR_BOF_HIT );
			goto Exit;
		}
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}

	rc = m_pCurRSBlk->GetPrev( (FLMBYTE *) vpBuffer, uiBufferLength, 
										puiReturnLength );
	
	// Position to the previous block?
	if( rc == FERR_BOF_HIT )
	{
		if( m_pCurRSBlk->GetPrev() != NULL )
		{
			m_pCurRSBlk->SetBuffer( NULL );
			m_pCurRSBlk = m_pCurRSBlk->GetPrev();
			if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
				goto Exit;
			rc = m_pCurRSBlk->GetPrev( (FLMBYTE *) vpBuffer, uiBufferLength,
											puiReturnLength );
			if( RC_BAD(rc))
				goto Exit;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Public:	GetFirst
Desc:		Return the first reference in the result set.
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_NOT_FOUND	- zero records in the result set.
****************************************************************************/

RCODE FResultSet::GetFirst(
	void *			vpBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	flmAssert( m_bFinalizeCalled );

	if( m_pCurRSBlk != m_pFirstRSBlk )
	{
		if( m_pCurRSBlk)
			m_pCurRSBlk->SetBuffer( NULL );
		m_pCurRSBlk = m_pFirstRSBlk;
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}
	else if( ! m_pCurRSBlk )
	{
		rc = RC_SET( FERR_NOT_FOUND );
		goto Exit;
	}

	rc = m_pCurRSBlk->GetNext( (FLMBYTE *) vpBuffer, uiBufferLength, 
										puiReturnLength );
Exit:
	return( rc);
}

/****************************************************************************
Public:	GetLast
Desc:		Return the last reference in the result set.
Ret:		FERR_OK			- Returned the current recRef[].
			FERR_CONV_DEST_OVERFLOW - buffer is not big enough for data
			FERR_NOT_FOUND	- zero records in the result set.
****************************************************************************/

RCODE FResultSet::GetLast(
	void *			vpBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	RCODE				rc;

	flmAssert( m_bFinalizeCalled );

	if( m_pCurRSBlk != m_pLastRSBlk )
	{
		if( m_pCurRSBlk)
			m_pCurRSBlk->SetBuffer( NULL );
		m_pCurRSBlk = m_pLastRSBlk;
		if( RC_BAD(rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}
	else if( ! m_pCurRSBlk )
	{
		rc = RC_SET( FERR_EOF_HIT );
		goto Exit;
	}

	rc = m_pCurRSBlk->GetPrev( (FLMBYTE *) vpBuffer, uiBufferLength, 
										puiReturnLength );
Exit:
	return( rc);
}

/****************************************************************************
Public:	FindMatch
Desc:		Find the matching entry in the result set using the compare routine.
			This does a binary search on the list of blocks.
Ret:		FERR_OK			- Returned OK
			FERR_NOT_FOUND	- match not found
****************************************************************************/

RCODE FResultSet::FindMatch(		// Find and return an etnry that 
											// matches in the result set (variable).
	void *			vpMatchEntry,	// Entry to match
	FLMUINT			uiMatchEntryLength,	// Variable length of above entry 
	void *			vpFoundEntry,	// (out) Entry to return
	FLMUINT *		puiFoundEntryLength,	// (out) Length of entry returned
	RSET_COMPARE_FUNC_p 				// Record compare function.
						fnCompare,		// Returns (FLMINT) -1, 0 or 1 values.
	void *		UserValue)		// UserValue for callback.
{
	RCODE				rc;
	FLMINT			iBlkCompare;			// RS_EQUALS if key is/would be in block.
	FResultSetBlk *pLowBlk;				// Used for locating block.
	FResultSetBlk *pHighBlk;			// Low and High are exclusive.

	flmAssert( m_bFinalizeCalled );

	// If not positioned anywhere, position to the midpoint.
	// Otherwise, start on the current block we are on.
	if( ! m_pCurRSBlk )
	{
		if( ! m_pFirstRSBlk )			// Will be null if no entries.
		{
			rc = RC_SET( FERR_NOT_FOUND );
			goto Exit;
		}
		if( m_pFirstRSBlk == m_pLastRSBlk )
			m_pCurRSBlk = m_pFirstRSBlk;
		else
		{
			m_pCurRSBlk = SelectMidpoint( m_pFirstRSBlk, m_pLastRSBlk, FALSE );
		}
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}

	// Set the exclusive low block and high block.

	pLowBlk = m_pFirstRSBlk;
	pHighBlk = m_pLastRSBlk;

	// Loop until the correct block is found.

	for(;;)
	{
		// Two return value returned: rc and iBlkCompare.
		// blk->FindMatch returns FERR_OK if the entry if found in the block.
		//						returns FERR_NOT_FOUND if not found in the block.
		// uiCompare returns RS_EQUALS if entry would be within the block.
		// otherwise RS_LESS_THAN if previous blocks should be checked
		// and RS_GREATER_THAN if next blocks should be checked.

		rc = m_pCurRSBlk->FindMatch( 
									(FLMBYTE *) vpMatchEntry, uiMatchEntryLength,
									(FLMBYTE *) vpFoundEntry, puiFoundEntryLength,
									fnCompare, UserValue,
									&iBlkCompare );
		// Found match or should key be within the block.
		if( RC_OK(rc) || (RS_EQUALS == iBlkCompare ))
		{
			goto Exit;
		}
		if( RS_LESS_THAN == iBlkCompare )
		{
			if( m_pCurRSBlk == pLowBlk )		// done if the low block
				goto Exit;							// keep FERR_NOT_FOUND value

			pHighBlk = m_pCurRSBlk->GetPrev();// Set the new high block
		}
		else	// RS_GREATER_THAN == iBlkCompare
		{
			if( m_pCurRSBlk == pHighBlk )		// done if we are at the high block
				goto Exit;							// keep FERR_NOT_FOUND value
			
			pLowBlk = m_pCurRSBlk->GetNext();
		}
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( NULL )))
			goto Exit;
			
		m_pCurRSBlk = SelectMidpoint( pLowBlk, pHighBlk, FALSE );

		// GWBUG 46817 - need to set the working buffer.
		
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}
Exit:
	return( rc);
}


/****************************************************************************
Desc:		Select the midpoint between two different blocks in a list.
			Entries should not be the same value.
****************************************************************************/

FResultSetBlk * FResultSet::SelectMidpoint(
	FResultSetBlk *pLowBlk,
	FResultSetBlk *pHighBlk,
	FLMBOOL			bPickHighIfNeighbors)
{
	int				siCount;
	FResultSetBlk *pTempBlk;

	// If the same then return.

	if( pLowBlk == pHighBlk )
	{
		pTempBlk = pLowBlk;
		goto Exit;
	}
	
	// Check if neighbors and use the boolean flag.

	if( pLowBlk->GetNext() == pHighBlk)
	{
		if( bPickHighIfNeighbors )
			pTempBlk = pHighBlk;
		else
			pTempBlk = pLowBlk;
		goto Exit;
	}

	// Count the total blocks exclusive between low and high and add one.
	// Check pTempBlk against null to not crash.

	for( pTempBlk = pLowBlk, siCount = 1;
		  pTempBlk && (pTempBlk != pHighBlk );
		  siCount++ )
	{
		pTempBlk = pTempBlk->GetNext();
	}

	// Check for implementation error - pTempBlk is NULL and handle.

	flmAssert( NULL != pTempBlk );
	if( ! pTempBlk )					// on error
	{
		pTempBlk = pLowBlk;			// Just position to low block to be safe.
		goto Exit;
	}

	// Loop to the middle item.
	siCount = siCount >> 1;			// Divide by two.
	for( pTempBlk = pLowBlk; siCount > 0; siCount-- )
	{
		pTempBlk = pTempBlk->GetNext();
	}
Exit:
	return pTempBlk;
}

/****************************************************************************
Public:	SetPosition
Desc:		Set the current entry position.  
In:		uiPosition - Zero based position value or RS_POSITION_NOT_SET
			to set back to the beginning or end.
Ret:		FERR_OK			- Returned OK
			FERR_EOF_HIT		- Positioned to the end of the result set.
****************************************************************************/

RCODE FResultSet::SetPosition(
	FLMUINT			uiPosition)
{
	RCODE				rc = FERR_OK;
	FResultSetBlk * pInitialBlk = m_pCurRSBlk;
	
	flmAssert( m_bFinalizeCalled );

	if( uiPosition == RS_POSITION_NOT_SET )
	{
		// GWBUG 46817 - set out of focus.
		if( m_pCurRSBlk )
		{
			if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( NULL )))
				goto Exit;
		}
		m_pCurRSBlk = NULL;
		goto Exit;
	}
	// else position to the correct block that holds uiPosition.


	if( !m_pCurRSBlk )
	{
		m_pCurRSBlk = m_pFirstRSBlk;
	}
	// Check for empty result set.
	if( !m_pCurRSBlk )
	{
		rc = RC_SET( FERR_EOF_HIT );
		goto Exit;
	}
	
	if( uiPosition < m_pCurRSBlk->GetInitialPosition() )
	{
		// Go backwards looking for the correct block.

		do
		{
			m_pCurRSBlk = m_pCurRSBlk->GetPrev();
			flmAssert( NULL != m_pCurRSBlk );
		}
		while( uiPosition < m_pCurRSBlk->GetInitialPosition() );
	}
	else if( uiPosition >=
			m_pCurRSBlk->GetInitialPosition() + m_pCurRSBlk->GetNumberOfEntries() )
	{
		// Go forward looking for the correct block.
		
		do
		{
			if( ! m_pCurRSBlk->GetNext() )
			{
				// Will set rc to EOF in SetPosition below.
				break;
			}
			m_pCurRSBlk = m_pCurRSBlk->GetNext();
		}
		while( uiPosition >=
			m_pCurRSBlk->GetInitialPosition() + m_pCurRSBlk->GetNumberOfEntries() );
	}

	// GWBUG 46817 - need working buffer out of focus.
	if( pInitialBlk != m_pCurRSBlk )
	{
		if( pInitialBlk)
		{
			if( RC_BAD( rc = pInitialBlk->SetBuffer( NULL )))
				goto Exit;
		}

		// GWBUG 46817 - need working buffer into focus.
		if( RC_BAD( rc = m_pCurRSBlk->SetBuffer( m_pBlockBuf1 )))
			goto Exit;
	}
	// Now we are positioned to the correct block.

	rc = m_pCurRSBlk->SetPosition( uiPosition );

Exit:
	return( rc);
}
	
/****************************************************************************
Desc:		Return a pointer to the next entry in the list.
****************************************************************************/
RCODE FResultSet::GetNextPtr(
	FResultSetBlk **	ppCurBlk,
	FLMBYTE **			ppBuffer,
	FLMUINT *			puiReturnLength)
{
	RCODE				rc;
	FResultSetBlk *pCurBlk = *ppCurBlk;
	FResultSetBlk *pNextBlk;
	FLMBYTE *		pBuffer;

	flmAssert( NULL != pCurBlk );

	while( RC_BAD( rc = pCurBlk->GetNextPtr( ppBuffer, puiReturnLength)))
	{
		if( rc == FERR_EOF_HIT )
		{
			if( pCurBlk->GetNext())
			{
				pNextBlk = pCurBlk->GetNext();
				if( !pNextBlk->IsFirstInChain() )
				{
					pBuffer = pCurBlk->GetBuffer();
					pCurBlk->SetBuffer( NULL );
					pCurBlk = pNextBlk;
					if( RC_BAD( rc = pCurBlk->SetBuffer( pBuffer )))
						goto Exit;
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
	
/****************************************************************************
Desc:		Union two block lists into a result output list.  This may be
			called to union two result sets or to perform the initial merge-sort
			on a create result set.  
			
			Performaing an N-way merge would be fast when we have over 10K
			of entries.  However, the code is more complex.
Notes:	Intersection and cross product routines are very similar
			to this routine.
****************************************************************************/
RCODE FResultSet::UnionBlkLists(
	FResultSetBlk *pLeftBlk,			// (IN) Left block
	FResultSetBlk *pRightBlk)			// (IN) Right block - may be NULL
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pLeftEntry, 
			 *			pRightEntry;
	FLMUINT			uiLeftLength, uiRightLength;

	// If no right block then copy all of the items from the left block
	// to the output block.  We could optimize this in the future.

	if( ! pRightBlk )
	{
		rc = CopyRemainingItems( pLeftBlk );
		goto Exit;
	}

	// Now the fun begins.  Read entries from both lists and union 
	// while checking the order of the entries.


	if( RC_BAD(rc = GetNextPtr( &pLeftBlk, &pLeftEntry, &uiLeftLength )))
	{
		if( rc == FERR_EOF_HIT )
		{
			rc = CopyRemainingItems( pRightBlk );
		}
		goto Exit;
	}
	if( RC_BAD(rc = GetNextPtr( &pRightBlk, &pRightEntry, &uiRightLength )))
	{
		if( rc == FERR_EOF_HIT )
		{
			rc = CopyRemainingItems( pLeftBlk );
		}
		goto Exit;
	}

	for(;;)
	{
		FLMINT		iCompare;
		if( RC_BAD( rc = m_fnCompare( pLeftEntry, uiLeftLength,
												pRightEntry, uiRightLength,
												m_UserValue, &iCompare)))
		{
			goto Exit;
		}

		if( iCompare == RS_LESS_THAN )
		{
			// Take the left item.

			if( RC_BAD(rc = AddEntry( pLeftEntry, uiLeftLength )))
				goto Exit;

			if( RC_BAD(rc = GetNextPtr( &pLeftBlk, &pLeftEntry, &uiLeftLength )))
			{
				if( rc != FERR_EOF_HIT )
					goto Exit;

				if( RC_BAD(rc = AddEntry( pRightEntry, uiRightLength )))
					goto Exit;
	
				// Left entries are done - read all of the right entries.

				rc = CopyRemainingItems( pRightBlk );
				goto Exit;
			}
		}
		else
		{
			// If equals then drop the right item and continue comparing left.
			// WARNING: Don't try to optimize for equals because when one
			// list runs out the remaining duplicate entries must be dropped.
			// Continuing to compare the duplicate item is the correct way.

			if( (iCompare == RS_GREATER_THAN) || !m_bDropDuplicates )
			{
				// Take the right item. 

				if( RC_BAD(rc = AddEntry( pRightEntry, uiRightLength )))
					goto Exit;
			}
		
			if( RC_BAD(rc = GetNextPtr( &pRightBlk, &pRightEntry, &uiRightLength)))
			{
				if( rc != FERR_EOF_HIT )
					goto Exit;

				if( RC_BAD(rc = AddEntry( pLeftEntry, uiLeftLength )))
					goto Exit;
		
				// Right entries are done - read all of the left entries.
	
				rc = CopyRemainingItems( pLeftBlk );
				goto Exit;
			}
		}
	}
Exit:
	if( RC_OK(rc))
	{
		// Flush out the output entries.
		rc = m_pCurRSBlk->Finalize( TRUE );
		m_pCurRSBlk->SetBuffer( NULL );
		m_pCurRSBlk = NULL;

		if( m_fnCallback )
		{
			++m_CallbackInfo.ui64UnitsDone; 
			(void) m_fnCallback( &m_CallbackInfo );
		}
	}

	return( rc);
}

	
/****************************************************************************
Desc:		Copy the remaining items from a block list to the output.
****************************************************************************/
RCODE FResultSet::CopyRemainingItems(
	FResultSetBlk *		pCurBlk)
{
	RCODE				rc;
	FLMBYTE *		pEntry;
	FLMUINT			uiLength;

	while( RC_OK(rc = GetNextPtr( &pCurBlk, &pEntry, &uiLength )))
	{
		if( RC_BAD(rc = AddEntry( pEntry, uiLength )))
			goto Exit;
	}
	if( rc == FERR_EOF_HIT )
	{
		rc = FERR_OK;
	}
Exit:
	return( rc);
}


/****************************************************************************
Desc:		Closes and deletes one of two files.
****************************************************************************/
void FResultSet::CloseFile(
	F_64BitFileHandle **		ppFileHdl64)
{
	if( ppFileHdl64 == &m_pFileHdl641)
	{
		if( m_bFile1Opened)
		{
			m_pFileHdl641->Close( TRUE);
			m_bFile1Opened = FALSE;
		}
		if (m_pFileHdl641)
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
		if (m_pFileHdl642)
		{
			m_pFileHdl642->Release();
			m_pFileHdl642 = NULL;
		}
	}
	return;
}

/****************************************************************************
Desc:		Close the file if previously opened and creates the file.
****************************************************************************/
RCODE FResultSet::OpenFile(
	F_64BitFileHandle **		ppFileHdl64)
{ 
	RCODE			rc = FERR_OK;
	FLMBOOL *	pbFileOpened;
	char *		pszIoPath;

	// Will close and delete if opened, else will do nothing.

	CloseFile( ppFileHdl64 );

	if( ppFileHdl64 == &m_pFileHdl641 )
	{
		pbFileOpened = &m_bFile1Opened;
		pszIoPath = &m_szFilePath1[ 0];
	}
	else
	{
		pbFileOpened = &m_bFile2Opened;
		pszIoPath = &m_szFilePath2[ 0];
	}
	
	f_strcpy( pszIoPath, m_szDefaultPath);

	if( (*ppFileHdl64 = f_new F_64BitFileHandle) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = (*ppFileHdl64)->CreateUnique( pszIoPath, "frs")))
	{
		(*ppFileHdl64)->Release();
		*ppFileHdl64 = NULL;
		goto Exit;
	}

	*pbFileOpened = TRUE;

Exit:

	return( rc);
}
