//-------------------------------------------------------------------------
// Desc:	Routines used during query to traverse through index b-trees - definitions.
// Tabs:	3
//
//		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fscursor.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FSCURSOR_H
#define FSCURSOR_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

class F_Base;

typedef struct KeyPosition
{
	FLMUINT			uiKeyLen;
	FLMUINT			uiRecordId;
	FLMBOOL			bExclusiveKey;

	// State information
	FLMUINT			uiRefPosition;
	FLMUINT			uiDomain;
	FLMUINT			uiBlockTransId;
	FLMUINT			uiBlockAddr;
	FLMUINT			uiCurElm;
	DIN_STATE		DinState;

	// Stack and key information
	BTSK *			pStack;
	FLMBOOL			bStackInUse;
	// VISIT: Add bIsUntilKey boolean.
	BTSK				Stack [BH_MAX_LEVELS];
	FLMBYTE			pKey [MAX_KEY_SIZ + 4];	// + 4 is for safety
} KEYPOS;

typedef struct Key_Set *	KEYSET_p;
typedef struct Key_Set
{
	KEYPOS			fromKey;
	KEYPOS			untilKey;
	KEYSET_p			pNext;
	KEYSET_p			pPrev;

} KEYSET;


/****************************************************************************
Desc:	File system implementation of a cursor for an index.
****************************************************************************/
class FSIndexCursor : public F_Base
{
public:

	FSIndexCursor();
	
	virtual ~FSIndexCursor();

	// Reset the cursor back to an empty state.
	void reset();

	// Reset the transaction on this cursor.
	RCODE resetTransaction( 
		FDB *			pDb);

	// Release all b-tree blocks back to the cache.
	void releaseBlocks( void);

	RCODE	setupKeys(
		FDB *				pDb,
		IXD_p				pIxd,
		QPREDICATE_p * ppQPredicateList,
		FLMBOOL *		pbDoRecMatch,
		FLMBOOL *		pbDoKeyMatch,
		FLMUINT *		puiLeafBlocksBetween,
		FLMUINT *		puiTotalKeys,	
		FLMUINT *		puiTotalRefs,	
		FLMBOOL *		pbTotalsEstimated);

	RCODE	setupKeys(
		FDB *			pDb,
		IXD_p       pIxd,
		FLMBYTE *	pFromKey,
		FLMUINT		uiFromKeyLen,
		FLMUINT		uiFromRecordId,
		FLMBYTE *	pUntilKey,
		FLMUINT		uiUntilKeyLen,
		FLMUINT		uiUntilRecordId,
		FLMBOOL		bExclusiveUntil);

	RCODE unionKeys(
		FSIndexCursor * pFSCursor);

	RCODE intersectKeys(
		FDB *			pDb,
		FSIndexCursor * pFSCursor);

	FLMBOOL compareKeyRange(
		FLMBYTE *	pFromKey,
		FLMUINT		uiFromKeyLen,
		FLMBOOL		bExclusiveFrom,
		FLMBYTE *	pUntilKey,
		FLMUINT		uiUntilKeyLen,
		FLMBOOL		bExclusiveUntil,
		FLMBOOL *	pbUntilKeyInSet,
		FLMBOOL *	pbUntilGreaterThan);
	
	// Returns FERR_OK, FERR_BOF_HIT, FERR_EOF_HIT or error
	RCODE currentKey(
		FDB *				pDb,
		FlmRecord **	pPrecordKey,
		FLMUINT *		puiRecordId);
	
	// Returns FERR_OK, FERR_BOF_HIT, FERR_EOF_HIT or error
	RCODE currentKeyBuf(
		FDB *				pDb,
		POOL *			pPool,
		FLMBYTE **		ppKeyBuf,
		FLMUINT *		puiKeyLen,
		FLMUINT *		puiRecordId,
		FLMUINT *		puiContainerId);
	
	// Returns FERR_OK, FERR_BOF_HIT or error
	RCODE firstKey(
		FDB *				pDb,
		FlmRecord **	pPrecordKey,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_EOF_HIT or error
	RCODE lastKey(
		FDB *				pDb,
		FlmRecord **	pPrecordKey,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_EOF_HIT or error
	RCODE	nextKey(
		FDB *				pDb,
		FlmRecord **	pPrecordKey,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_BOF_HIT or error
	RCODE	prevKey(
		FDB *				pDb,
		FlmRecord **	pPrecordKey,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_EOF_HIT or error
	RCODE	nextRef(
		FDB *				pDb,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_BOF_HIT or error
	RCODE	prevRef(
		FDB *				pDb,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_NOT_FOUND if not found or error.
	RCODE positionTo(
		FDB *				pDb,
		FLMBYTE *		pKey,
		FLMUINT			uiKeyLen,
		FLMUINT			uiRecordId = 0);

	// Returns FERR_OK, FERR_NOT_FOUND if not found or error.
	RCODE positionToDomain(
		FDB *				pDb,
		FLMBYTE *		pKey,
		FLMUINT			uiKeyLen,
		FLMUINT			uiDomain);

	// Does this index support native absolute positioning?
	FLMBOOL isAbsolutePositionable()
	{
		return (m_pIxd->uiFlags & IXD_POSITIONING) ? TRUE : FALSE;
	}

	// Set absolute position (if not supported returns FERR_FAILURE).
	// uiPosition of zero positions to BOF, ~0 to EOF, one based value.
	RCODE setAbsolutePosition(
		FDB *				pDb,
		FLMUINT			uiRefPosition);

	// Get absolute position (if not supported returns FERR_FAILURE).
	// uiPosition of zero positions to BOF, ~0 to EOF, one based value.
	RCODE getAbsolutePosition(
		FDB *				pDb,
		FLMUINT *		puiRefPosition);

	// Get the total number of reference with all from/until sets.
	// Does not have to support absolute positioning.
	RCODE getTotalReferences(
		FDB *				pDb,
		FLMUINT *		puiTotalRefs,
		FLMBOOL *		pbTotalEstimated);

	RCODE savePosition( void);

	RCODE restorePosition( void);

	RCODE	getFirstLastKeys(
		FLMBYTE **		ppFirstKey,
		FLMUINT *		puiFirstKeyLen,
		FLMBYTE **		ppLastKey,
		FLMUINT *		puiLastKeyLen,
		FLMBOOL *		pbLastKeyExclusive);

protected:

	KEYSET *	getFromUntilSets( void) 
	{
		return m_pFirstSet;
	}

private:

	void freeSets( void);

	RCODE useNewDb( 
		FDB *	pDb);

	FLMBOOL FSCompareKeyPos(
		KEYSET *			pSet1,
		KEYSET *			pSet2,
		FLMBOOL *		pbFromKeysLessThan,
		FLMBOOL *		pbUntilKeysGreaterThan);

	RCODE setKeyPosition(
		FDB *				pDb,
		FLMBOOL			bGoingForward,
		KEYPOS *			pInKeyPos,
		KEYPOS *			pOutKeyPos);

	RCODE reposition(
		FDB *				pDb,
		FLMBOOL			bCanPosToNextKey,
		FLMBOOL			bCanPosToPrevKey,
		FLMBOOL *		pbKeyGone,
		FLMBOOL			bCanPosToNextRef,
		FLMBOOL			bCanPosToPrevRef,
		FLMBOOL *		pbRefGone);

	void releaseKeyBlocks( 
		KEYPOS *			pKeyPos)
	{
		if( pKeyPos->bStackInUse)
		{
			FSReleaseStackCache( pKeyPos->Stack, BH_MAX_LEVELS, FALSE);
			pKeyPos->bStackInUse = FALSE;
		}
	}

	RCODE checkTransaction(
		FDB *				pDb)
	{
		return (RCODE) ((m_uiCurrTransId != pDb->LogHdr.uiCurrTransID ||
			m_uiBlkChangeCnt != pDb->uiBlkChangeCnt)
				? resetTransaction( pDb) 
				: FERR_OK);
	}

	RCODE	setupForPositioning(
		FDB *			pDb);

	// Save the current key position into pSaveKeyPos
	void saveCurrKeyPos(
		KEYPOS *		pSaveKeyPos);

	// Restore the current key position from pSaveKeyPos
	void restoreCurrKeyPos(
		KEYPOS *		pSaveKeyPos);

	// Returns FERR_OK, FERR_NOT_FOUND if no key set.
	RCODE getKeySet(
		FLMBYTE *		pKey,
		FLMUINT			uiKeyLen,
		KEYSET **		ppKeySet);

	// Database information
	FLMUINT				m_uiCurrTransId;
	FLMUINT				m_uiBlkChangeCnt;
	FLMBOOL				m_bIsUpdateTrans;
	FLMUINT				m_uiIndexNum;
	LFILE	*				m_pLFile;
	IXD *					m_pIxd;

	// Contains a list of all of the FROM/UNTIL sets
	KEYSET *				m_pFirstSet;
	
	// State information.
	KEYSET *				m_pCurSet;
	FLMBOOL				m_bAtBOF;			// Before the first key.
	FLMBOOL				m_bAtEOF;			// After the last key.

	KEYPOS				m_curKeyPos;		// Current key position
	KEYPOS *				m_pSavedPos;		// Saved position
	KEYSET				m_DefaultSet;		// Single minimum FROM/UNTIL key
};

typedef struct RecPosition
{
	FLMUINT			uiRecordId;
	FLMUINT			uiBlockTransId;
	FLMUINT			uiBlockAddr;
	BTSK *			pStack;
	FLMBOOL			bStackInUse;
	FLMBOOL			bExclusiveKey;			// True if an UNTIL key.
	BTSK				Stack [BH_MAX_LEVELS];
	FLMBYTE			pKey [DIN_KEY_SIZ];
} RECPOS;

// The record set will always have inclusive FROM/UNTIL values.
typedef struct Record_Set * RECSET_p;
typedef struct Record_Set
{
	RECPOS			fromKey;
	RECPOS			untilKey;
	RECSET_p			pNext;
	RECSET_p			pPrev;

} RECSET;

/****************************************************************************
Desc:	File system implementation of a cursor for a data container.
****************************************************************************/
class FSDataCursor: public F_Base
{
public:

	FSDataCursor();
	
	virtual ~FSDataCursor();

	// Reset this cursor back to an initial state.
	void reset();

	// Reset the transaction on this cursor.
	RCODE resetTransaction( 
		FDB *				pDb);

	void releaseBlocks( void);

	void setContainer( FLMUINT uiContainer)
	{
		m_uiContainer = uiContainer;
	}

	RCODE	setupRange(
		FDB *				pDb,
		FLMUINT			uiContainer,
		FLMUINT			uiLowRecordId,
		FLMUINT			uiHighRecordId,
		FLMUINT *		puiLeafBlocksBetween,
		FLMUINT *		puiTotalRecords,		
		FLMBOOL *		pbTotalsEstimated);

	RCODE unionRange(
		FSDataCursor * pFSCursor);

	RCODE intersectRange(
		FSDataCursor * pFSCursor);

	RCODE currentRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);
	
	// Returns FERR_OK, FERR_BOF_HIT or error
	RCODE firstRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_EOF_HIT or error
	RCODE lastRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_EOF_HIT or error
	RCODE	nextRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_BOF_HIT or error
	RCODE	prevRec(
		FDB *				pDb,
		FlmRecord **	pPrecord,
		FLMUINT *		puiRecordId);

	// Returns FERR_OK, FERR_NOT_FOUND or error
	RCODE positionTo(
		FDB *				pDb,
		FLMUINT			uiRecordId);

	// Returns FERR_OK, FERR_EOF_HIT or error
	RCODE positionToOrAfter(
		FDB *				pDb,
		FLMUINT *		puiRecordId);

	RCODE savePosition( void);

	RCODE restorePosition( void);

protected:

	RECSET *	getFromUntilSets( void) 
	{
		return m_pFirstSet;
	}

private:

	void freeSets( void);

	void releaseRecBlocks( 
		RECPOS *			pRecPos)
	{
		if( pRecPos->bStackInUse)
		{
			FSReleaseStackCache( pRecPos->Stack, BH_MAX_LEVELS, FALSE);
			pRecPos->bStackInUse = FALSE;
		}
	}
	RCODE setRecPosition(
		FDB *				pDb,
		FLMBOOL			bGoingForward,
		RECPOS *			pInRecPos,
		RECPOS *			pOutRecPos);

	RCODE reposition(
		FDB *				pDb,
		FLMBOOL			bCanPosToNextRec,
		FLMBOOL			bCanPosToPrevRec,
		FLMBOOL *		pbRecordGone);

	FLMBOOL FSCompareRecPos(
		RECSET  *		pSet1,
		RECSET  *		pSet2,
		FLMBOOL *		pbFromKeysLessThan,
		FLMBOOL *		pbUntilKeysGreaterThan);

	RCODE checkTransaction(
		FDB *				pDb)
	{
		return (RCODE) ((m_uiCurrTransId != pDb->LogHdr.uiCurrTransID ||
			m_uiBlkChangeCnt != pDb->uiBlkChangeCnt)
				? resetTransaction( pDb) 
				: FERR_OK);
	}
	
	// Database information
	FLMUINT				m_uiCurrTransId;
	FLMUINT				m_uiBlkChangeCnt;
	FLMBOOL				m_bIsUpdateTrans;
	FLMUINT				m_uiContainer;
	LFILE	*				m_pLFile;

	// Contains a list of all of the FROM/UNTIL sets
	RECSET *				m_pFirstSet;
	
	// State information.
	RECSET *				m_pCurSet;
	FLMBOOL				m_bAtBOF;			// Before the first key.
	FLMBOOL				m_bAtEOF;			// After the last key.

	RECPOS				m_curRecPos;		// Current key position
	RECPOS *				m_pSavedPos;	  	// Saved position
	RECSET				m_DefaultSet;		// Single minimum FROM/UNTIL key

};

#include "fpackoff.h"

#endif
