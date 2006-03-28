//-------------------------------------------------------------------------
// Desc:	Result sets - class definitions
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
// $Id: frset.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FRSET_H
#define FRSET_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Need forward declaration.

class FResultSet;
class F_64BitFileHandle;

/*****************************************************************************
***** 
** 	Definitions
***** 
*****************************************************************************/

// The default block size can be over 64K for 32-bit machines.
// We have tested a block size over 400K bytes (9/97).
// Three blocks are required during the merge phase if more than
// one block is used in the result set.  The block may be resized at finalize.
// 0xE000 == 56K bytes
// 0x6000 = 24K bytes

#define	RSBLK_BLOCK_SIZE		0xE000

// Need to allocate on the stack the maximum fixed entry size
// in the recursive quicksort routine.

#define	MAX_FIXED_ENTRY_SIZE			64


// Values to be used with the *piCompare variables in the compare routine.

#define	RS_LESS_THAN				(-1)
#define	RS_EQUALS					(0)
#define	RS_GREATER_THAN			(1)


// Values for udPosition in GetPosition() and SetPosition().

#define	RS_POSITION_NOT_SET		((FLMUINT)0xFFFFFFFF)

/*****************************************************************************
***** 
** 	Compare Routines
**		The compare routine is used so that the user may specify how to
**		sort the entries in the result set.  If a non-zero RCODE is 
** 	returned from the compare routine then the error will bubble up
**		to the caller of the merge/finalize routine.
***** 
*****************************************************************************/


typedef RCODE (* RSET_COMPARE_FUNC_p)(
	void *				vpData1,
	FLMUINT				uiLength1,
	void *				vpData2,
	FLMUINT				uiLength2,
	void *				UserValue,
	FLMINT *				piCompare);

RCODE FRSDefaultCompare(
	void *				vpData1,
	FLMUINT				uiLength1,
	void *				vpData2,
	FLMUINT				uiLength2,
	void *				UserValue,
	FLMINT *				piCompare);

/*****************************************************************************
***** 
** 	Callback Function
	The callback function is used to callback the application
	during the merge process.  The merge process can take a long
	time if there are millions of entries.  The initial value of
	udEstTotalUnits may be slightly less than the ending 
	result of udUnitsDone.  
**
***** 
*****************************************************************************/

typedef struct RSET_CB_INFO
{
	void *		UserValue;
	FLMUINT64	ui64EstTotalUnits;	// Estimated total number of units to do.
	FLMUINT64	ui64UnitsDone;			// Units completed

} RSET_CB_INFO;

typedef FLMINT (* RSET_CB_FUNC_p)( RSET_CB_INFO *);

/*****************************************************************************
***** 
** 	Forward References
***** 
*****************************************************************************/

class FResultSet;
class FResultSetBlk;


/*****************************************************************************
***** 
** 	Result Set Block Definitions
***** 
*****************************************************************************/

	// Block Size Limits:
	//		1) Allows over 64K entries per block providing (int) is 32 bits.
	//		2) The block size cannot be over 64K for 16 bit applications.
	//			A 400K block size has been tested on WIN95(10/97).
	//		3) Three blocks are allocated as a maximum during the merge phase.
	//			Two of these blocks are freed after the merge phase.
	//
	// Block Layout:
	//		Block Header Structure
	//		Variable Length Entries
	//			An array of [Offset][Length] entry items
	//			The variable length entries are stored from the end back to front.
	//			The length of a variable length entry must be less than the block size.
	//		Fixed Length Entries
	//			Each entry is stored from the first of the buffer to the last.
	//
	//
	// BLKOFFSET_SIZE cannot be 4 for 16-bit code.  Must be 4 if block size > 64K.
	// The macros below are in mixed case because they may turn into methods
	// if we would want to input the block size in the result set setup call.
	// SEE RSBLK_BLOCK_SIZE in frset.h

	#if RSBLK_BLOCK_SIZE <= 0xFFFF
		#define	BLKOFFSET_SIZE						2
		typedef FLMUINT16		BLKOFFSET;
		#define	GetOffset(p)		( FB2UW( p))
		#define	SetOffset(ofs,p)	( UW2FBA( ofs, p ))
	#else
		typedef FLMUINT32		BLKOFFSET;
		#define	BLKOFFSET_SIZE						4
		#define	GetOffset(p)		( FB2UD( p))
		#define	SetOffset(ofs,p)	( UD2FBA( ofs, p ))
	#endif

	#define	LENGTH_SIZE				2
	#define	GetLength(p)			((FLMUINT)( FB2UW(&(p)[BLKOFFSET_SIZE])))
	#define	SetLength(len,p)		( UW2FBA( ((FLMUINT16)len), &(p)[BLKOFFSET_SIZE] ))


	// Write bytes used to write less than 64K at a time because
	// Write and Read only support up to 64K size buffer.

	#define	MAX_WRITE_BYTES		0x8000

/****************************************************************************

							Block Header Definition

  Desc:	Actually stored as the first section of each block.  
			We can write this structure because the same process will
			read the block header i.e. portability is not a problem.
****************************************************************************/

typedef struct FBlockHeader 
{
	FLMUINT64	ui64FilePos;			// RSBLK_UNSET_FILE_POS or file position
	FLMUINT		uiEntryCount,			// Number of entries in block
					uiBlockSize;			// Total Block size in memory or on disk
	FLMBOOL		bFirstBlock,			// TRUE=first block in chain
					bLastBlock;				// TRUE=last block in chain
} FBlockHeader;

#define RSBLK_UNSET_FILE_POS			(~((FLMUINT64)0))
#define RSBLK_IS_FIRST_IN_LIST		TRUE
#define RSBLK_ENTRIES_IN_ORDER		TRUE

/****************************************************************************
Desc:
****************************************************************************/
class	FResultSetBlk : public F_Base
{
public:

	FResultSetBlk();

	virtual ~FResultSetBlk()
	{
		if( m_pNext)
		{
			m_pNext->SetPrev( m_pPrev);
		}
		
		if( m_pPrev)
		{
			m_pPrev->SetNext( m_pNext);
		}
	}

	void reset( void);

	void Setup(
		F_64BitFileHandle **	ppFileHdl64,
		RSET_COMPARE_FUNC_p	fnCompare,
		void *					UserValue,
		FLMUINT					uiEntrySize,
		FLMBOOL					bFirstInList,
		FLMBOOL					bDropDuplicates,
		FLMBOOL					bEntriesInOrder);

	RCODE SetBuffer(
		FLMBYTE *				pBuffer,
		FLMUINT					uiBufferSize = RSBLK_BLOCK_SIZE);

	FINLINE FLMBYTE * GetBuffer( void)
	{ 
		return m_pBlockBuf;
	}

	FINLINE FLMUINT BytesUsedInBuffer( void)
	{ 
		if( m_bEntriesInOrder)
		{
			return( m_BlockHeader.uiBlockSize);
		}
		else
		{
			return( m_BlockHeader.uiBlockSize - m_uiLengthRemaining);
		}
	}

	RCODE AddEntry(
		FLMBYTE *				pEntry,
		FLMUINT					uiEntryLength );

	RCODE ModifyEntry(
		FLMBYTE *				pEntry,
		FLMUINT					uiEntryLength = 0);

	FINLINE FLMUINT GetNumberOfEntries( void)
	{ 
		return( m_BlockHeader.uiEntryCount); 
	}

	FINLINE RCODE Finalize(
		FLMBOOL					bForceWrite)
	{
		return( Flush( TRUE, bForceWrite));
	}

	RCODE Flush(
		FLMBOOL					bLastBlockInList,
		FLMBOOL					bForceWrite);

	FINLINE FResultSetBlk * GetNext( void)
	{
		return( m_pNext); 
	}
	
	FINLINE FResultSetBlk * GetPrev( void)
	{
		return( m_pPrev); 
	}

	FINLINE void SetNext(
		FResultSetBlk *		pRSBlk)
	{
		m_pNext = pRSBlk; 
	}

	FINLINE void SetPrev(
		FResultSetBlk *		pRSBlk )
	{
		m_pPrev = pRSBlk; 
	}

	RCODE GetCurrent(
		FLMBYTE *				pBuffer,
		FLMUINT					uiBufferLength,
		FLMUINT *				puiReturnLength);
	
	FINLINE RCODE GetNext(
		FLMBYTE *				pBuffer,
		FLMUINT					uiBufferLength,
		FLMUINT *				puiReturnLength)
	{
		// Are we on the last entry or past the last entry?

		if( m_iEntryPos + 1 >= (FLMINT) m_BlockHeader.uiEntryCount )
		{
			m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
			return( RC_SET( FERR_EOF_HIT));
		}

		m_iEntryPos++;
		return( CopyCurrentEntry( pBuffer, uiBufferLength, puiReturnLength));
	}

	RCODE GetNextPtr(
		FLMBYTE **				ppBuffer,
		FLMUINT *				puiReturnLength);

	RCODE GetPrev(
		FLMBYTE *				pBuffer,
		FLMUINT					uiBufferLength,
		FLMUINT *				puiReturnLength);

	FINLINE RCODE GetPosition(
		FLMUINT *				puiPosition)
	{
		if( !m_bPositioned || (m_iEntryPos == -1) ||
			 (m_iEntryPos == (FLMINT) m_BlockHeader.uiEntryCount))
		{
			*puiPosition = RS_POSITION_NOT_SET;
		}
		else if( puiPosition)
		{
			*puiPosition = m_uiBlkEntryPosition + m_iEntryPos;
		}

		return( FERR_OK);
	}

	RCODE SetPosition(
		FLMUINT					uiPosition);

	RCODE FindMatch( 
		FLMBYTE *				pMatchEntry,
		FLMUINT					uiMatchEntryLength, 
		FLMBYTE *				pFoundEntry,
		FLMUINT *				puiFoundEntryLength,
		RSET_COMPARE_FUNC_p	fnCompare,
		void *					UserValue,
		FLMINT *					piCompare);

	FINLINE FLMBOOL IsFirstInChain( void)
	{
		if( m_BlockHeader.bFirstBlock)
		{
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}

	FINLINE void SetFirstInChain(
		FLMBOOL					bIsFirstInChain)
	{
		m_BlockHeader.bFirstBlock = bIsFirstInChain;
	}

	FINLINE void SetLastInChain(
		FLMBOOL					bIsLastInChain)
	{
		m_BlockHeader.bLastBlock = bIsLastInChain;
	}

	FINLINE FLMUINT GetInitialPosition( void)
	{ 
		return( m_uiBlkEntryPosition);
	}
	
	FINLINE void SetInitialPosition(
		FLMUINT					uiBlkEntryPosition)
	{ 
		m_uiBlkEntryPosition = uiBlkEntryPosition;
	}

private:

	RCODE AddEntry(
		FLMBYTE *				pEntry);

	void SqueezeSpace();
	
	RCODE SortAndRemoveDups( void);

	void RemoveEntry(
		FLMBYTE *				pEntry);

	RCODE QuickSort(
		FLMUINT					uiLowerBounds,
		FLMUINT					uiUpperBounds);

	FINLINE RCODE EntryCompare(
		FLMBYTE *				pLeftEntry,
		FLMBYTE *				pRightEntry,
		FLMINT *					piCompare)
	{
		RCODE		rc;

		if( m_bFixedEntrySize)
		{
			rc = m_fnCompare( pLeftEntry,  m_uiEntrySize,
					pRightEntry, m_uiEntrySize, m_UserValue, piCompare);
		}
		else
		{
			rc = m_fnCompare( m_pBlockBuf + GetOffset( pLeftEntry), 
					GetLength( pLeftEntry ), m_pBlockBuf + GetOffset( pRightEntry),
					GetLength( pRightEntry ), m_UserValue, piCompare);
		}
		
		if( *piCompare == RS_EQUALS)
		{
			m_bDuplicateFound = TRUE;
		}
		
		return( rc);
	}
	
	RCODE CopyCurrentEntry(
		FLMBYTE *				pBuffer,
		FLMUINT					uiBufferLength,
		FLMUINT *				puiReturnLength);

	RCODE CompareEntry(
		FLMBYTE *				pMatchEntry,
		FLMUINT					uiMatchEntryLength,
		FLMUINT					uiEntryPos,
		RSET_COMPARE_FUNC_p	fnCompare,
		void *					UserValue,
		FLMINT *					piCompare);

	RCODE Write( void);
	
	RCODE Read( void);

	FBlockHeader			m_BlockHeader;
	RSET_COMPARE_FUNC_p 	m_fnCompare;
	void *					m_UserValue;
	FLMBYTE *				m_pBlockBuf;
	FLMBYTE *				m_pNextEntryPtr;
	FLMBYTE *				m_pEndPoint;
	FResultSetBlk *		m_pNext;
	FResultSetBlk *		m_pPrev;
	F_64BitFileHandle **	m_ppFileHdl64;
	FLMUINT					m_uiBlkEntryPosition;
	FLMUINT					m_uiLengthRemaining;
	FLMINT					m_iEntryPos;
	FLMUINT					m_uiEntrySize;
	FLMBOOL					m_bEntriesInOrder;
	FLMBOOL					m_bFixedEntrySize;
	FLMBOOL					m_bPositioned;
	FLMBOOL					m_bModifiedEntry;
	FLMBOOL					m_bDuplicateFound;
	FLMBOOL					m_bDropDuplicates;
};

/*****************************************************************************
Desc:
*****************************************************************************/
class FResultSet : public F_Base
{
public:

	FResultSet();
	
	virtual ~FResultSet();
	
	RCODE Setup(
		const char *				pIoPath,
		RSET_COMPARE_FUNC_p		fnCompare,
		void *	   				UserValue,
		FLMUINT						uiEntrySize,
		FLMBOOL						bDropDuplicates = TRUE,
		FLMBOOL						bEntriesInOrder = FALSE);

	FINLINE void SetCallback(	  
		RSET_CB_FUNC_p 			fnCallback,
		void *						UserValue)
	{
		m_CallbackInfo.UserValue = UserValue;
		m_fnCallback = fnCallback;
	}
		
	FINLINE FLMUINT64 GetTotalEntries( void)
	{
		FResultSetBlk	*	pBlk = m_pFirstRSBlk;
		FLMUINT64			ui64TotalEntries = 0;

		for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->GetNext())
		{
			ui64TotalEntries += pBlk->GetNumberOfEntries();
		}
		
		return( ui64TotalEntries);
	}

	RCODE AddEntry(
		void *						vpEntry,
		FLMUINT						uiEntryLength = 0);

	RCODE Finalize(
		FLMUINT64 *					pui64TotalEntries = NULL);

	FINLINE RCODE ModifyCurrent(
		void *						pEntry,
		FLMUINT						uiEntryLength)
	{
		return( m_pCurRSBlk->ModifyEntry( (FLMBYTE *) pEntry, uiEntryLength));
	}

	RCODE GetCurrent(
		void *						vpBuffer,
		FLMUINT						uiBufferLength = 0,
		FLMUINT *					puiReturnLength = NULL);
	
	RCODE GetNext(
		void *						vpBuffer,
		FLMUINT						uiBufferLength = 0,
		FLMUINT *					puiReturnLength = NULL);
	
	RCODE GetPrev(
		void *						vpBuffer,
		FLMUINT						uiBufferLength = 0,
		FLMUINT *					puiReturnLength = NULL);
	
	RCODE GetFirst(
		void *						vpBuffer,
		FLMUINT						uiBufferLength = 0,
		FLMUINT *					puiReturnLength = NULL);
	
	RCODE GetLast(
		void *						vpBuffer,
		FLMUINT						uiBufferLength = 0,
		FLMUINT *					puiReturnLength = NULL);

	RCODE FindMatch( 
		void *						vpMatchEntry,
		FLMUINT						uiMatchEntryLength, 
		void *						vpFoundEntry,
		FLMUINT *					puiFoundEntryLength,
		RSET_COMPARE_FUNC_p 		fnCompare,
		void *						UserValue);
		
	FINLINE RCODE FindMatch(
		void *						vpMatchEntry,
		void *						vpFoundEntry,
		RSET_COMPARE_FUNC_p		fnCompare,
		void *						UserValue)
	{
		return( FindMatch( vpMatchEntry, (FLMUINT) m_uiEntrySize,
				vpFoundEntry, NULL, fnCompare, UserValue));
	}

	FINLINE RCODE GetPosition(
		FLMUINT *					puiPosition)
	{
		if( !m_pCurRSBlk)
		{
			if( puiPosition)
			{
				*puiPosition = RS_POSITION_NOT_SET;
			}
			
			return( FERR_OK);
		}

		return( m_pCurRSBlk->GetPosition( puiPosition));
	}
	
	RCODE SetPosition(
		FLMUINT						uiPosition);

	RCODE reset( void);

private:

	FINLINE FLMUINT64 NumberOfBlockChains( void)
	{ 
		FLMUINT64			ui64Count = 0;
		FResultSetBlk *	pBlk = m_pFirstRSBlk;

		for( ; pBlk ; pBlk = pBlk->GetNext())
		{
			if( pBlk->IsFirstInChain())
			{
				ui64Count++;
			}
		}
		
		return( ui64Count);
	}

	RCODE MergeSort( void);
	
	RCODE GetNextPtr(
		FResultSetBlk **			ppCurBlk,
		FLMBYTE **					ppBuffer,
		FLMUINT *					puiReturnLength);
	
	RCODE UnionBlkLists(
		FResultSetBlk *			pLeftBlk,
		FResultSetBlk *			pRightBlk = NULL);

	RCODE CopyRemainingItems(
		FResultSetBlk *			pCurBlk);	
	
	void CloseFile(
		F_64BitFileHandle **		ppFileHdl64);

	RCODE OpenFile(
		F_64BitFileHandle **		ppFileHdl64);

	FResultSetBlk * SelectMidpoint(
		FResultSetBlk *			pLowBlk,
		FResultSetBlk *			pHighBlk,
		FLMBOOL						bPickHighIfNeighbors);

	RSET_COMPARE_FUNC_p			m_fnCompare;
	void *							m_UserValue;
	RSET_CB_FUNC_p 				m_fnCallback;
	RSET_CB_INFO					m_CallbackInfo;
	FLMUINT							m_uiEntrySize;
	FLMUINT64						m_ui64TotalEntries;
	FResultSetBlk *				m_pCurRSBlk;
	FResultSetBlk *				m_pFirstRSBlk;
	FResultSetBlk *				m_pLastRSBlk;
	char								m_szDefaultPath[ F_PATH_MAX_SIZE];
	char								m_szFilePath1[ F_PATH_MAX_SIZE];
	char								m_szFilePath2[ F_PATH_MAX_SIZE];
	F_64BitFileHandle *			m_pFileHdl641;
	F_64BitFileHandle	*			m_pFileHdl642;
	FLMBYTE *						m_pBlockBuf1;
	FLMBYTE *						m_pBlockBuf2;
	FLMBYTE *						m_pBlockBuf3;
	FLMUINT							m_uiBlockBuf1Len;
	FLMBOOL							m_bFile1Opened;
	FLMBOOL							m_bFile2Opened;
	FLMBOOL							m_bOutput2ndFile;
	FLMBOOL							m_bInitialAdding;
	FLMBOOL							m_bFinalizeCalled;
	FLMBOOL							m_bSetupCalled;
	FLMBOOL							m_bDropDuplicates;
	FLMBOOL							m_bAppAddsInOrder;
	FLMBOOL							m_bEntriesInOrder;
};					

#include "fpackoff.h"

#endif
