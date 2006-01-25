//-------------------------------------------------------------------------
// Desc:	Check database indexes for logical corruptions.
// Tabs:	3
//
//		Copyright (c) 1992,1994-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flchkix.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE chkVerifyTrackerCounts(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndexNum,
	FLMUINT			uiKeyCount,
	FLMUINT			uiRefCount);

FSTATIC RCODE chkIsCountIndex(
	STATE_INFO *	pStateInfo,
	FLMUINT			uiIndexNum,
	FLMBOOL *		pbIsCountIndex);

FSTATIC RCODE chkResolveRSetMissingKey(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIxRefDrn);

FSTATIC RCODE chkVerifyDelNonUniqueRec(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiRecDrn,
	FLMUINT *		puiRecContainerRV,
	FLMBOOL *		pbDelRecRV);

FSTATIC RCODE chkVerifyKeyExists(
	FDB *				pDb,
	LFILE *			pLFile,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiRefDrn,
	FLMBOOL *		pbFoundRV);

FSTATIC RCODE chkAddDelKeyRef(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndexNum,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDrn,
	FLMUINT			uiFlags);

FSTATIC RCODE chkGetKeySource(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDrn,
	FLMUINT *		puiRecordContainerRV,
	FLMBOOL *		pbKeyInRecRV,
	FLMBOOL *		pbKeyInIndexRV);

FSTATIC RCODE chkReportIxError(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	eCorruptionType	eCorruption,
	FLMUINT				uiErrIx,
	FLMUINT				uiErrDrn,
	FLMBYTE *			pucErrKey,
	FLMUINT				uiErrKeyLen,
	FLMBOOL *			pbFixErrRV);

RCODE chkGetNextRSKey(
	IX_CHK_INFO *	pIxChkInfo);

FSTATIC RCODE chkVerifyKeyNotUnique(
	STATE_INFO *	pStateInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT *		puiRefCountRV);

FSTATIC RCODE chkStartUpdate(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo);

FSTATIC RCODE chkEndUpdate(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo);

FLMINT chkRSCallbackFunc(
	RSET_CB_INFO *	pCBInfo);

RCODE chkCompareIxRSEntries(
	void *		vpData1,
	FLMUINT		uiLength1,
	void *		vpData2,
	FLMUINT		uiLength2,
	void *		UserValue,
	FLMINT *		piCompare);


/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE chkRSGetNext(
	void *			pRSet,
	FLMBYTE *		pBuffer,
	FLMUINT			uiBufferLength,
	FLMUINT *		puiReturnLength)
{
	FLMUINT		uiReturnLen;
	RCODE			rc;

	rc = ((FResultSet *)pRSet)->GetNext( (void *)pBuffer,
		uiBufferLength, &uiReturnLen);
	*puiReturnLength = (rc == FERR_OK) ? uiReturnLen : 0;

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE chkKeyToTree(
	IXD *				pIxd,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FlmRecord **	ppKeyRV)
{
	return flmIxKeyOutput( pIxd, pucKey, uiKeyLen, ppKeyRV, FALSE);
}

/********************************************************************
Desc: Verifies the key and reference counts against the counts that
		are stored in the tracker record.
*********************************************************************/
FSTATIC RCODE chkVerifyTrackerCounts(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndexNum,
	FLMUINT			uiKeyCount,
	FLMUINT			uiRefCount
	)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = pStateInfo->pDb;
	FlmRecord *			pRecord = NULL;
	eCorruptionType	eCorruption;
	FLMUINT				uiTrackerKeyCount = 0;
	FLMUINT				uiTrackerRefCount = 0;
	void *				pvField;

	// Retrieve the tracker record from record cache.

	if (RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL, FLM_TRACKER_CONTAINER,
								uiIndexNum, TRUE, NULL, NULL, &pRecord)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
		rc = FERR_OK;
	}
	else
	{
		if ((pvField = pRecord->find( pRecord->root(), FLM_KEY_TAG)) != NULL)
		{
			if (RC_BAD( rc = pRecord->getUINT( pvField, &uiTrackerKeyCount)))
			{
				goto Exit;
			}
		}
		if ((pvField = pRecord->find( pRecord->root(), FLM_REFS_TAG)) != NULL)
		{
			if (RC_BAD( rc = pRecord->getUINT( pvField, &uiTrackerRefCount)))
			{
				goto Exit;
			}
		}
	}

	// See if the counts match what we got from the tracker record.

	if (uiKeyCount != uiTrackerKeyCount || uiRefCount != uiTrackerRefCount)
	{

		// Log an error.

		eCorruption = (eCorruptionType)((uiKeyCount != uiTrackerKeyCount)
													? FLM_KEY_COUNT_MISMATCH
													: FLM_REF_COUNT_MISMATCH);

		if (RC_BAD( rc = chkReportError( pIxChkInfo->pDbInfo, eCorruption,
									LOCALE_INDEX, uiIndexNum, LF_INDEX,
									0xFF, 0, 0, 0, 0, 0xFFFF, 0, NULL)))
		{
			goto Exit;
		}
	}
Exit:
	if (pRecord)
	{
		pRecord->Release();
	}
	return( rc);
}

/********************************************************************
Desc: Determine if an index is an index that keeps key and reference
		counts.
*********************************************************************/
FSTATIC RCODE chkIsCountIndex(
	STATE_INFO *	pStateInfo,
	FLMUINT			uiIndexNum,
	FLMBOOL *		pbIsCountIndex
	)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = pStateInfo->pDb;
	IXD *		pIxd;

	if( RC_BAD( rc = fdictGetIndex( pDb->pDict,
		pDb->pFile->bInLimitedMode, uiIndexNum,
		NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}
	*pbIsCountIndex = (FLMBOOL)((pIxd->uiFlags & IXD_COUNT)
										 ? (FLMBOOL)TRUE
										 : (FLMBOOL)FALSE);
Exit:
	return( rc);
}
	
/********************************************************************
Desc: Verifies the current index key against the result set.
*********************************************************************/
RCODE chkVerifyIXRSet(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIxRefDrn
	)
{
	FLMINT		iCmpVal = 0;
	FLMUINT		uiIteration = 0;
	FLMBOOL		bRSetEmpty = FALSE;
	RCODE			rc = FERR_OK;
	RS_IX_KEY *	pCurrRSKey;
	RS_IX_KEY *	pPrevRSKey;

	if (!pIxChkInfo->pCurrRSKey)
	{
		pIxChkInfo->pCurrRSKey = &pIxChkInfo->IxKey1;
		pIxChkInfo->pPrevRSKey = &pIxChkInfo->IxKey2;
	}

	/* Compare index and result set keys */
	
	while( !bRSetEmpty)
	{
		if( pIxChkInfo->bGetNextRSKey)
		{
			/*
			Get the next key from the result set.  If the result set
			is empty, then pIxChkInfo->uiRSKeyLen will be set to
			zero, forcing the problem to be resolved below.
			*/

			if( RC_BAD( rc = chkGetNextRSKey( pIxChkInfo)))
			{
				if( rc == FERR_EOF_HIT || rc == FERR_NOT_FOUND)
				{
					/*
					Set bRSetEmpty to TRUE so that the loop will exit after the
					current key is resolved.  Otherwise, conflict resolution on
					the current key will be repeated forever (infinite loop).
					*/

					bRSetEmpty = TRUE;
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				/* Updated statistics */
				
				pIxChkInfo->pDbInfo->pProgress->ui64NumKeysExamined++;
			}
		}
		pCurrRSKey = pIxChkInfo->pCurrRSKey;
		pPrevRSKey = pIxChkInfo->pPrevRSKey;

		if( pCurrRSKey->uiRSKeyLen == 0 || bRSetEmpty)
		{
			/*
			We don't have a key because we got an EOF when
			reading the result set.  Need to resolve the
			fact that the result set does not have a key
			that is found in the index.  Set iCmpVal to
			1 to force this resolution.
			*/

			iCmpVal = 1;
		}
		else
		{
			/*
			Compare the index key and result set key.
			*/

			iCmpVal = chkCompareKeySet( 
				pCurrRSKey->uiRSIxNum,
				&(pCurrRSKey->pucRSKeyBuf[ RS_KEY_OFFSET]),
				pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD,
				pCurrRSKey->uiRSRefDrn,
				pStateInfo->pLogicalFile->pLFile->uiLfNum,
				pStateInfo->pCurKey,
				pStateInfo->uiCurKeyLen,
				uiIxRefDrn);
		}

		if( iCmpVal < 0)
		{

			// If a comparison is done where the keys from the result set
			// don't match what we got from the index, we will forego verifying
			// the tracker counts.  Verifying of tracker counts can only
			// occur if we have an otherwise clean check of the index keys.

			pIxChkInfo->bCheckCounts = FALSE;

			/*
			The result set key is less than the index key.  This could mean
			that the result set key needs to be added to the index.
			*/

			if(( RC_BAD( rc = chkResolveIXMissingKey( pStateInfo,
											pIxChkInfo))) ||
				(pIxChkInfo->pDbInfo->bReposition))
			{
				/*
				If the key was added to the index (bReposition == TRUE)
				or we got some other error, we don't want to get the next
				result set key.
				*/

				pIxChkInfo->bGetNextRSKey = FALSE;
				goto Exit;
			}
			else
			{
				/*
				False alarm.  The index is missing the key because of
				a concurrent update.  We want to get the next RS key.
				*/

				pIxChkInfo->bGetNextRSKey = TRUE;
			}
		}
		else if( iCmpVal > 0)
		{

			// If a comparison is done where the keys from the result set
			// don't match what we got from the index, we will forego verifying
			// the tracker counts.  Verifying of tracker counts can only
			// occur if we have an otherwise clean check of the index keys.

			pIxChkInfo->bCheckCounts = FALSE;

			/*
			The result set key is greater than the index key.  This could mean
			that the index key needs to be deleted from the index.  Whether we
			delete the index key or not, we don't need to get the next result
			set key, but we do want to reposition and get the next index key.
			*/

			pIxChkInfo->bGetNextRSKey = FALSE;
			if(( RC_BAD( rc = chkResolveRSetMissingKey( pStateInfo,
										pIxChkInfo, uiIxRefDrn))) ||
				(pIxChkInfo->pDbInfo->bReposition))
			{
				goto Exit;
			}
			break;
		}
		else
		{

			/*
			The index and result set keys are equal.  We want to get
			the next result set and index keys.
			*/

			pIxChkInfo->bGetNextRSKey = TRUE;

			// Determine if we have switched indexes.  If so, verify the key
			// and reference counts against the counts in the tracker record.

			if (pCurrRSKey->uiRSIxNum != pPrevRSKey->uiRSIxNum)
			{
				if (pIxChkInfo->bCheckCounts)
				{

					// Verify the key and reference counts against tracker record.

					if (RC_BAD( rc = chkVerifyTrackerCounts( pStateInfo,
													pIxChkInfo,
													pPrevRSKey->uiRSIxNum,
													pIxChkInfo->uiRSIxKeyCount,
													pIxChkInfo->uiRSIxRefCount)))
					{
						goto Exit;
					}
				}

				// Determine if the new index is one that supports counts.

				if (RC_BAD( rc = chkIsCountIndex( pStateInfo,
											pCurrRSKey->uiRSIxNum,
											&pIxChkInfo->bCheckCounts)))
				{
					goto Exit;
				}
				if (pIxChkInfo->bCheckCounts)
				{

					// Set the counts to one.

					pIxChkInfo->uiRSIxKeyCount = 1;
					pIxChkInfo->uiRSIxRefCount = 1;
				}
			}
			else
			{
				if (pIxChkInfo->bCheckCounts)
				{

					// Always increment the reference count.

					pIxChkInfo->uiRSIxRefCount++;

					// See if the key changed.

					if (pCurrRSKey->uiRSKeyLen !=
						 pPrevRSKey->uiRSKeyLen ||
						 (pCurrRSKey->uiRSKeyLen > RS_KEY_OFFSET &&
						  f_memcmp( &pCurrRSKey->pucRSKeyBuf [RS_KEY_OFFSET],
										&pPrevRSKey->pucRSKeyBuf [RS_KEY_OFFSET],
										pCurrRSKey->uiRSKeyLen - RS_KEY_OFFSET) != 0))
					{
						pIxChkInfo->uiRSIxKeyCount++;
					}
					else
					{

						// If the keys are the same, at least the DRNs better
						// be different.

						flmAssert( pCurrRSKey->uiRSRefDrn !=
									  pPrevRSKey->uiRSRefDrn);
					}
				}
			}
			break;
		}

		/* Call the yield function periodically */

		uiIteration++;
		if( !(uiIteration & 0x1F) )
		{
			f_yieldCPU();
		}
	}

Exit:

	return( rc);
}


/********************************************************************
Desc: Retrieves the next key from the sorted result set
*********************************************************************/
RCODE chkGetNextRSKey(
	IX_CHK_INFO *	pIxChkInfo
	)
{
	RCODE			rc = FERR_OK;
	RS_IX_KEY *	pCurrRSKey;

	// Swap current and last key pointers - this allows us to always keep
	// the last key without having to memcpy the keys.

	pCurrRSKey = pIxChkInfo->pCurrRSKey;
	pIxChkInfo->pCurrRSKey = pIxChkInfo->pPrevRSKey;
	pIxChkInfo->pPrevRSKey = pCurrRSKey;
	pCurrRSKey = pIxChkInfo->pCurrRSKey;

	/* Get the next key */

	if( RC_BAD( rc = chkRSGetNext( pIxChkInfo->pRSet,
		pCurrRSKey->pucRSKeyBuf,
		MAX_KEY_SIZ + RS_KEY_OVERHEAD,
		&(pCurrRSKey->uiRSKeyLen))))
	{
		goto Exit;
	}

	/* Verify that the key meets the minimum length requirements */

	flmAssert( pCurrRSKey->uiRSKeyLen >= RS_KEY_OVERHEAD);

	/* Extract the index number and reference DRN */

	pCurrRSKey->uiRSIxNum =
		(FLMUINT)FB2UW( &(pCurrRSKey->pucRSKeyBuf[ RS_IX_OFFSET]));
	pCurrRSKey->uiRSRefDrn =
		(FLMUINT)FB2UD( &(pCurrRSKey->pucRSKeyBuf[ RS_REF_OFFSET]));

Exit:

	return( rc);

}

/********************************************************************
Desc: Resolves the case of a key found in the result set but not in
		the current index.
*********************************************************************/
RCODE
	chkResolveIXMissingKey(
		STATE_INFO *	pStateInfo,
		IX_CHK_INFO *	pIxChkInfo
	)
{
	FLMBOOL		bKeyInRec;
	FLMBOOL		bKeyInIndex;
	RCODE			rc = FERR_OK;
	FLMBOOL		bFixCorruption = FALSE;
	RS_IX_KEY *	pCurrRSKey = pIxChkInfo->pCurrRSKey;

	/*
	Determine if the record generates the key and if the
	key is found in the index.
	*/

	if( RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo,
		pCurrRSKey->uiRSIxNum, &(pCurrRSKey->pucRSKeyBuf[ RS_KEY_OFFSET]),
		(FLMUINT)(pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
		pCurrRSKey->uiRSRefDrn, NULL,
		&bKeyInRec,	&bKeyInIndex)))
	{
		if( rc == FERR_INDEX_OFFLINE)
		{
			rc = FERR_OK;
		}
		goto Exit;
	}

	/*
	If the record does not generate the key or the key+ref is in the index,
	the index is not corrupt.
	*/

	if( !bKeyInRec || bKeyInIndex)
	{
		pIxChkInfo->pDbInfo->pProgress->ui64NumConflicts++;
		goto Exit;
	}

	/*
	Otherwise, the index is corrupt.
	*/

	/* Update statistics */
	
	pIxChkInfo->pDbInfo->pProgress->ui64NumRecKeysNotFound++;
	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexCorruptions++;

	/* Report the error */
	
	if( RC_BAD( rc = chkReportIxError( pStateInfo, pIxChkInfo,
		FLM_DRN_NOT_IN_KEY_REFSET, pCurrRSKey->uiRSIxNum,
		pCurrRSKey->uiRSRefDrn,
		&(pCurrRSKey->pucRSKeyBuf[ RS_KEY_OFFSET]),
		(FLMUINT)(pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
		&bFixCorruption)))
	{
		goto Exit;
	}

	/*
	Exit if the application does not want to repair the corruption.
	*/

	if( !bFixCorruption)
	{
		/* Set the logical corruption flag */
		pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
		goto Exit;
	}

	/* Fix the corruption */
	
	/* Update statistics */

	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexRepairs++;
		
	/* Add the key. */
	
	if( RC_OK( rc = chkAddDelKeyRef(
		pStateInfo, pIxChkInfo, pCurrRSKey->uiRSIxNum,
		&(pCurrRSKey->pucRSKeyBuf[ RS_KEY_OFFSET]),
		(FLMUINT)(pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
		pCurrRSKey->uiRSRefDrn, 0)))
	{
		pIxChkInfo->pDbInfo->bReposition = TRUE;
		goto Exit;
	}
	else
	{
		if( rc == FERR_NOT_UNIQUE)
		{
			/*
			A subsequent record probably also generates this key,
			but the index is a unique index so we were not allowed
			to add the missing key + ref to the index. This record
			should probably be deleted.
			*/

			if( RC_OK( rc = chkResolveNonUniqueKey( pStateInfo,
				pIxChkInfo, pCurrRSKey->uiRSIxNum,
				&(pCurrRSKey->pucRSKeyBuf[ RS_KEY_OFFSET]),
				(FLMUINT)(pCurrRSKey->uiRSKeyLen - RS_KEY_OVERHEAD),
				pCurrRSKey->uiRSRefDrn)))
			{
				pIxChkInfo->pDbInfo->bReposition = TRUE;
				goto Exit;
			}
		}
		else
		{
			/* Set the logical corruption flag */
			pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
		}
	}

Exit:

	return( rc);
}


/********************************************************************
Desc: Resolves the case of a key found in the current index but not
		in the result set.
*********************************************************************/
FSTATIC RCODE
	chkResolveRSetMissingKey(
		STATE_INFO *	pStateInfo,
		IX_CHK_INFO *	pIxChkInfo,
		FLMUINT			uiIxRefDrn
	)
{
	FLMBOOL						bKeyInRec;
	FLMBOOL						bKeyInIndex;
	RCODE							rc = FERR_OK;
	FLMBOOL						bFixCorruption = FALSE;

	/*
	See if the key is found in the index and/or generated
	by the record.
	*/

	if( RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo,
		pStateInfo->pLogicalFile->pLFile->uiLfNum,
		pStateInfo->pCurKey, pStateInfo->uiCurKeyLen,
		uiIxRefDrn, NULL, &bKeyInRec, &bKeyInIndex)))
	{
		if( rc == FERR_INDEX_OFFLINE)
		{
			rc = FERR_OK;
		}
		goto Exit;
	}

	/*
	If the key is generated by the record or the key is not found
	in the index, the index is not corrupt.
	*/
	
	if( bKeyInRec || !bKeyInIndex)
	{
		pIxChkInfo->pDbInfo->pProgress->ui64NumConflicts++;
		goto Exit;
	}

	/*
	Otherwise, the index is corrupt.
	*/

	/* Update statistics */
	
	pIxChkInfo->pDbInfo->pProgress->ui64NumKeysNotFound++;
	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexCorruptions++;

	/* Report the error */

	if( RC_BAD( rc = chkReportIxError( pStateInfo, pIxChkInfo,
		FLM_IX_KEY_NOT_FOUND_IN_REC, 
		pStateInfo->pLogicalFile->pLFile->uiLfNum, uiIxRefDrn,
		pStateInfo->pCurKey, pStateInfo->uiCurKeyLen,
		&bFixCorruption)))
	{
		goto Exit;
	}

	/*
	Exit if the application does not want to repair the corruption.
	*/

	if( !bFixCorruption)
	{
		/* Set the logical corruption flag */
		pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
		goto Exit;
	}

	/* Fix the corruption */
	
	/* Update statistics */

	pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexRepairs++;
		
	/* Delete the reference from the index */
	
	if( RC_OK( rc = chkAddDelKeyRef(
		pStateInfo, pIxChkInfo, 
		pStateInfo->pLogicalFile->pLFile->uiLfNum,
		pStateInfo->pCurKey, pStateInfo->uiCurKeyLen,
		uiIxRefDrn, KREF_DELETE_FLAG)))
	{
		pIxChkInfo->pDbInfo->bReposition = TRUE;
	}
	else
	{
		/* Set the logical corruption flag */
		pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
	}

Exit:

	return( rc);
}


/********************************************************************
Desc: Resolves the case of multiple references associated with a
		key in a unique index.
*********************************************************************/
RCODE
	chkResolveNonUniqueKey(
		STATE_INFO *	pStateInfo,
		IX_CHK_INFO *	pIxChkInfo,
		FLMUINT			uiIndex,
		FLMBYTE *		pucKey,
		FLMUINT			uiKeyLen,
		FLMUINT			uiDrn
	)
{
	FDB *							pDb = pStateInfo->pDb;
	LFILE *						pRecLFile = NULL;
	FLMBOOL						bDeleteRec = FALSE;
	FLMUINT						uiRecContainer;
	RCODE							rc = FERR_OK;
	RCODE							rc2 = FERR_OK;
	FLMBOOL						bFixCorruption = FALSE;
	FlmRecord *					pOldRecord = NULL;

	/*
	Verify that the record violates the constraints of the unique
	index and should be deleted.
	*/
	
	if( RC_BAD( rc = chkVerifyDelNonUniqueRec( pStateInfo, pIxChkInfo,
		uiIndex, pucKey, uiKeyLen, uiDrn, &uiRecContainer, &bDeleteRec)))
	{
		goto Exit;
	}

	if( bDeleteRec)
	{
		/* Update statistics */

		pIxChkInfo->pDbInfo->pProgress->ui64NumNonUniqueKeys++;
		pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexCorruptions++;

		/* Report the error */
	
		if( RC_BAD( rc = chkReportIxError( pStateInfo, pIxChkInfo,
			FLM_NON_UNIQUE_ELM_KEY_REF, uiIndex, uiDrn,
			pucKey, uiKeyLen, &bFixCorruption)))
		{
			goto Exit;
		}

		if( !bFixCorruption)
		{
			/* Set the logical corruption flag */
			pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
			goto Exit;
		}

		/*
		Delete the record that generated the non-unique
		reference.
		*/
		
		/* Update statistics */

		pIxChkInfo->pDbInfo->pProgress->uiLogicalIndexRepairs++;
			
		/*
		Start an update transaction, if necessary.
		*/

		if( RC_BAD( rc = chkStartUpdate( pStateInfo, pIxChkInfo)))
		{
			goto Exit;
		}

		/*
		Re-verify that the record should be deleted.
		*/

		if( RC_BAD( rc = chkVerifyDelNonUniqueRec( pStateInfo, pIxChkInfo,
			uiIndex, pucKey, uiKeyLen, uiDrn, &uiRecContainer, &bDeleteRec)))
		{
			goto Exit;
		}

		if( bDeleteRec == TRUE)
		{
			void *		pvMark;
							
			/*
			Mark the DB's temporary pool.
			*/
			
			pvMark = GedPoolMark( &(pDb->TempPool));

			/*
			Call the internal delete function, passing boolean flags
			indicating that missing keys should not prevent the
			record deletion and that the record validator callback
			should not be called.
			*/

			if( RC_BAD( rc = fdictGetContainer( pDb->pDict,
				uiRecContainer, &pRecLFile)))
			{
				goto Exit;
			}

			rc = flmDeleteRecord( pDb, pRecLFile,
				uiDrn, &pOldRecord, TRUE);

			if (gv_FlmSysData.EventHdrs [F_EVENT_UPDATES].pEventCBList)
			{
				flmUpdEventCallback( pDb,
						F_EVENT_DELETE_RECORD, (HFDB)pDb, rc, uiDrn,
						uiRecContainer, NULL, pOldRecord);
			}

			/*
			Reset the DB's temporary pool.  The flmDeleteRecord
			call allocates space for the record that is being deleted.
			*/
			
			GedPoolReset( &(pDb->TempPool), pvMark);
			
			if( RC_BAD( rc))
			{
				/*
				If the record had already been deleted, continue the
				check without reporting the error.
				*/
				if( rc == FERR_NOT_FOUND)
				{
					rc = FERR_OK;

					/* Update statistics */

					pIxChkInfo->pDbInfo->pProgress->uiNumProblemsFixed++;
				}
				else
				{
					/* Set the logical corruption flag */
					pIxChkInfo->pDbInfo->pProgress->bLogicalIndexCorrupt = TRUE;
				}
				goto Exit;
			}
			
			/* Update statistics */

			pIxChkInfo->pDbInfo->pProgress->uiNumProblemsFixed++;
		}
	}
	else
	{
		/* Increment the conflict counter */
		pIxChkInfo->pDbInfo->pProgress->ui64NumConflicts++;
	}

Exit:

	/*
	End the update.  chkEndUpdate will be a no-op if an update
	transaction was not started.
	*/

	rc2 = chkEndUpdate( pStateInfo, pIxChkInfo);

	if( pOldRecord)
	{
		pOldRecord->Release();
	}

	return( (RCODE)((rc != FERR_OK) ? (RCODE)rc : (RCODE)rc2));
}


/********************************************************************
Desc: Verifies that the specified record should be deleted because it
		generates key(s) which violate the constraints of a unique
		index.
*********************************************************************/
FSTATIC RCODE chkVerifyDelNonUniqueRec(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiRecDrn,
	FLMUINT *		puiRecContainerRV,
	FLMBOOL *		pbDelRecRV
	)
{
	FLMBOOL						bKeyInRec;
	FLMBOOL						bRecRefdByKey;
	FLMUINT						uiRefCount;
	FLMUINT						uiRecContainer;
	RCODE							rc = FERR_OK;

	*pbDelRecRV = FALSE;
	*puiRecContainerRV = 0;

	/*
	See if the key is found in the index and/or generated
	by the record.
	*/

	if( RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo,
		uiIndex,	pucKey, uiKeyLen, uiRecDrn,
		&uiRecContainer, &bKeyInRec, &bRecRefdByKey)))
	{
		if( rc == FERR_INDEX_OFFLINE)
		{
			rc = FERR_OK;
		}
		goto Exit;
	}
		
	*puiRecContainerRV = uiRecContainer;

	if( bKeyInRec == TRUE)
	{
		/* Verify that the key is not unique */

		if( RC_BAD( rc = chkVerifyKeyNotUnique(
			pStateInfo, uiIndex, pucKey,
			uiKeyLen, &uiRefCount)))
		{
			goto Exit;
		}

		if( uiRefCount > 1)
		{
			/*
			The unique index has multiple references for the
			specified key.  Since the current record generates
			a non-unique key, it should be deleted even if
			it is not one of the records referenced by the
			key.  Of course, if it is already referenced by
			the key, deleting the record will reduce the
			number of references associated with the key
			by one.
			*/

			*pbDelRecRV = TRUE;
		}
		else if( uiRefCount == 1 && bRecRefdByKey == FALSE)
		{
			/*
			The unique index already has a key corresponding
			to the key being generated by the current record.
			However, the record is not referenced from the
			unique index.  The record should still be
			deleted since it generates a non-unique key.
			*/

			*pbDelRecRV = TRUE;
		}
	}

Exit:

	return( rc);
}


/********************************************************************
Desc: Determines if a key is generated by the current record
		and/or if the key is found in the current index
*********************************************************************/
FSTATIC RCODE chkGetKeySource(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	FLMUINT			uiIndex,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiDrn,
	FLMUINT *		puiRecordContainerRV,
	FLMBOOL *		pbKeyInRecRV,
	FLMBOOL *		pbKeyInIndexRV
	)
{
	FlmRecord *	pRecord = NULL;
	FDB *			pDb = pStateInfo->pDb;
	LFILE *		pLFile;
	LFILE *		pIxLFile;
	REC_KEY *	pKeys = NULL;
	REC_KEY *	pTempKey = NULL;
	IXD *			pIxd;
	FLMBYTE		ucRecKeyBuf[ MAX_KEY_SIZ];
	FLMUINT		uiRecKeyLen;
	FLMUINT		uiKeyCount;
	FLMBOOL		bResetKRef = FALSE;
	void *		pIxPoolMark;
	void *		pDbPoolMark;
	FLMUINT		uiContainerNum;
	RCODE			rc = FERR_OK;

	/*
	Initialize return values.
	*/

	*pbKeyInRecRV = FALSE;
	*pbKeyInIndexRV = FALSE;
	
	if( puiRecordContainerRV)
	{
		*puiRecordContainerRV = 0;
	}

	/* Initialize variables */

	pIxPoolMark = GedPoolMark( &(pIxChkInfo->pool));

	/*
	Need to mark the DB's temporary pool.  The index code allocates
	memory for new CDL entries from the DB pool.  If the pool is not
	reset, it grows during the check and becomes VERY large.
	*/

	pDbPoolMark = GedPoolMark( &(pDb->TempPool));


	/* Set up the KRef so that flmGetRecKeys will work */
	
	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}
	bResetKRef = TRUE;

	/* Get the LFile and IXD of the index */

	if( RC_BAD( rc = fdictGetIndex( pDb->pDict,
		pDb->pFile->bInLimitedMode, uiIndex,
		&pIxLFile, &pIxd)))
	{
		// Return FERR_INDEX_OFFLINE error.
		goto Exit;
	}

	if ((uiContainerNum = pIxd->uiContainerNum) == 0)
	{

		// Container number is always the last two bytes of the key.

		flmAssert( uiKeyLen > getIxContainerPartLen( pIxd));
		uiContainerNum = getContainerFromKey( pucKey, uiKeyLen);
	}

	// Get the LFile of the record that caused the error

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainerNum, &pLFile)))
	{
		goto Exit;
	}

	// Set the record container return value

	if( puiRecordContainerRV)
	{
		*puiRecordContainerRV = uiContainerNum;
	}

	/* See if the key is in the index. */
		
	if( RC_BAD( rc = chkVerifyKeyExists( pDb,
		pIxLFile, pucKey, uiKeyLen, uiDrn, pbKeyInIndexRV)))
	{
		goto Exit;
	}

	/* Read the record */

	if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
		pLFile->uiLfNum, uiDrn, FALSE, NULL, NULL, &pRecord)))
	{
		if (rc != FERR_NOT_FOUND)
			goto Exit;

		// NOTE: Deliberately not bringing in to cache if not found there.

		if( RC_BAD( rc = FSReadRecord( pDb, pLFile, uiDrn,
								&pRecord, NULL, NULL)))
		{
			if( rc == FERR_NOT_FOUND)
			{
				*pbKeyInRecRV = FALSE;
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
	}

	if( pRecord)
	{
		/* Generate record keys */

		if (RC_BAD( rc = flmGetRecKeys( pDb, pIxd, pRecord,
									pRecord->getContainerID(),
									TRUE, &(pIxChkInfo->pool), &pKeys)))
		{
			goto Exit;
		}

		uiKeyCount = 0;
		pTempKey = pKeys;
		while( pTempKey != NULL)
		{
			/* Build the collated keys for each key tree. */

			if( RC_BAD( rc = KYTreeToKey( pDb, pIxd,
				pTempKey->pKey, pTempKey->pKey->getContainerID(),
				ucRecKeyBuf, &uiRecKeyLen, 0)))
			{
				goto Exit;
			}

			if( KYKeyCompare( pucKey, uiKeyLen,
				ucRecKeyBuf, uiRecKeyLen) == BT_EQ_KEY)
			{
				*pbKeyInRecRV = TRUE;
				break;
			}
			pTempKey = pTempKey->pNextKey;
			uiKeyCount++;

			/*
			Release the CPU periodically to prevent CPU hog
			problems.
			*/

			f_yieldCPU();
		}
	}

Exit:

	if (pKeys)
	{
		pTempKey = pKeys;
		while (pTempKey)
		{
			pTempKey->pKey->Release();
			pTempKey = pTempKey->pNextKey;
		}
	}

	if( pRecord)
	{
		pRecord->Release();
	}

	// Remove any keys added to the KRef
	
	if (bResetKRef)
	{
		KYAbortCurrentRecord( pDb);
	}

	// Reset the DB's temporary pool
	
	GedPoolReset( &(pDb->TempPool), pDbPoolMark);

	// Reset the index check pool
	
	GedPoolReset( &(pIxChkInfo->pool), pIxPoolMark);

	return( rc);
}


/********************************************************************
Desc: Verify that a key is (or is not) found in an index.
*********************************************************************/
FSTATIC RCODE chkVerifyKeyExists(
	FDB *				pDb,
	LFILE *			pLFile,
	FLMBYTE *		pucKey,
	FLMUINT			uiKeyLen,
	FLMUINT			uiRefDrn,
	FLMBOOL *		pbFoundRV)
{
	RCODE			rc = FERR_OK;	/* Return code */
	BTSK			stackBuf[ BH_MAX_LEVELS ];	/* Stack to hold b-tree variables */
	BTSK *		stack = stackBuf;				/* Points to proper stack frame	 */
	FLMUINT		uiDinDomain = DIN_DOMAIN( uiRefDrn) + 1; /* Lower bounds  */
	FLMBYTE		ucBtKeyBuf[ MAX_KEY_SIZ ];	/* Key buffer pointed to by stack */
	DIN_STATE	dinState;
	FLMUINT		uiTmpDrn;

	*pbFoundRV = FALSE;
	f_memset( &dinState, 0, sizeof( DIN_STATE));

	/*	Initialize stack cache. */

	FSInitStackCache( &(stackBuf[ 0]), BH_MAX_LEVELS);
	stack = stackBuf;
	stack->pKeyBuf = ucBtKeyBuf;

	/* Search for the key. */
	
	if( RC_BAD( rc = FSBtSearch( pDb, 
		pLFile, &stack, pucKey, uiKeyLen, uiDinDomain)))
	{
		goto Exit;
	}

	if( stack->uiCmpStatus == BT_EQ_KEY)
	{
		uiTmpDrn = uiRefDrn;

		/* Reading the current element, position to or after uiTmpDrn */
		rc = FSRefSearch( stack, &dinState, &uiTmpDrn);

		/* If the entry was not found, returns FERR_FAILURE */

		if( rc == FERR_OK)
		{
			*pbFoundRV = TRUE;
		}
		else if( rc != FERR_FAILURE)
		{
			goto Exit;
		}
		else /* rc == FERR_FAILURE */
		{
			rc = FERR_OK;
		}
	}

Exit:

	/* Free the stack cache */

	FSReleaseStackCache( stackBuf, BH_MAX_LEVELS, FALSE);
	return( rc );
}


/********************************************************************
Desc: Compares a composite key (index, ref, key) for equality.
Note: Since index references are sorted in decending order, a
		composite key with a lower ref DRN will sort after a key
		with a higher ref DRN.
*********************************************************************/
FLMINT
	chkCompareKeySet(
		FLMUINT		uiIxNum1,
		FLMBYTE *	pData1,
		FLMUINT		uiLength1,
		FLMUINT		uiDrn1,
		FLMUINT		uiIxNum2,
		FLMBYTE *	pData2,
		FLMUINT		uiLength2,
		FLMUINT		uiDrn2
	)
{
	FLMINT	iCmpVal = RS_EQUALS;
	FLMUINT	uiMinLen;


	/* Compare index numbers */

	if( uiIxNum1 > uiIxNum2)
	{
		iCmpVal = RS_GREATER_THAN;
		goto Exit;
	}
	else if( uiIxNum1 < uiIxNum2)
	{
		iCmpVal = RS_LESS_THAN;
		goto Exit;
	}

	/* Compare keys */

	uiMinLen = (FLMUINT)(uiLength1 < uiLength2) ? uiLength1 : uiLength2;
	iCmpVal = f_memcmp( pData1, pData2, uiMinLen);
	if( iCmpVal == 0)
	{
		/* Compare references */

		if( uiLength1 == uiLength2)
		{
			/*
			A key with a lower ref DRN will sort after a key
			with a higher ref DRN.
			*/

			if( uiDrn1 > uiDrn2)
			{
				iCmpVal = RS_LESS_THAN;
			}
			else if( uiDrn1 < uiDrn2)
			{
				iCmpVal = RS_GREATER_THAN;
			}
			else
			{
				iCmpVal = RS_EQUALS;
				goto Exit;
			}
		}
		else if( uiLength1 > uiLength2)
		{
			iCmpVal = RS_GREATER_THAN;
		}
		else
		{
			iCmpVal = RS_LESS_THAN;
		}
	}
	else
	{
		iCmpVal = (FLMINT)((iCmpVal > 0)
						? (FLMINT)RS_GREATER_THAN
						: (FLMINT)RS_LESS_THAN);
	}
	
Exit:

	return( iCmpVal);
}


/***************************************************************************
Desc:	This routine adds or deletes an index key and/or reference.
*****************************************************************************/
FSTATIC RCODE
	chkAddDelKeyRef(
		STATE_INFO *	pStateInfo,
		IX_CHK_INFO *	pIxChkInfo,
		FLMUINT			uiIndexNum,
		FLMBYTE *		pucKey,
		FLMUINT			uiKeyLen,
		FLMUINT			uiDrn,
		FLMUINT			uiFlags
	)
{
	RCODE				rc = FERR_OK;
	RCODE				rc2 = FERR_OK;
	FLMBYTE			ucKeyBuf[ sizeof( KREF_ENTRY) + MAX_KEY_SIZ];
	KREF_ENTRY *	pKrefEntry = (KREF_ENTRY *)(&ucKeyBuf[ 0]);
	IXD *				pIxd;
	LFILE *			pLFile;
	FLMBOOL			bStartedUpdate = FALSE;
	FLMBOOL			bKeyInRec;
	FLMBOOL			bKeyInIndex;


	/* Start an update transaction, if necessary */

	if( RC_BAD( rc = chkStartUpdate( pStateInfo, pIxChkInfo)))
	{
		goto Exit;
	}
	bStartedUpdate = TRUE;
	
	/* Look up the LFILE and IXD for the index. */

	if( RC_BAD( rc = fdictGetIndex( pStateInfo->pDb->pDict,
		pStateInfo->pDb->pFile->bInLimitedMode, uiIndexNum,
		&pLFile, &pIxd)))
	{
		// Shouldn't get FERR_INDEX_OFFLINE in here.
		goto Exit;
	}

	/* Verify that the state has not changed */

	if( RC_BAD( rc = chkGetKeySource( pStateInfo, pIxChkInfo,
		uiIndexNum, pucKey, uiKeyLen, uiDrn, NULL,
		&bKeyInRec, &bKeyInIndex)))
	{
		goto Exit;
	}

	if( (bKeyInIndex == TRUE && ((uiFlags & KREF_DELETE_FLAG) != 0)) ||
		(bKeyInIndex == FALSE && uiFlags == 0))
	{

		/* Setup the KrefEntry structure */

		flmAssert( uiIndexNum > 0 && uiIndexNum < FLM_UNREGISTERED_TAGS); // Sanity check
		f_memcpy( &(ucKeyBuf[ sizeof( KREF_ENTRY)]), pucKey, uiKeyLen);
		pKrefEntry->ui16KeyLen = (FLMUINT16)uiKeyLen;
		pKrefEntry->ui16IxNum = (FLMUINT16)uiIndexNum;
		pKrefEntry->uiDrn = uiDrn;
		pKrefEntry->uiTrnsSeq = 1;
		pKrefEntry->uiFlags = uiFlags;

		if( (pIxd->uiFlags & IXD_UNIQUE) != 0)
		{
			/*
			Do not allow duplicate keys to be added to a unique index.
			*/
			pKrefEntry->uiFlags |= KREF_UNIQUE_KEY;
		}

		/* Add or delete the key/reference. */

		if( RC_BAD( rc = FSRefUpdate( pStateInfo->pDb, pLFile, pKrefEntry)))
		{
			goto Exit;
		}

		/* Update statistics */
		pIxChkInfo->pDbInfo->pProgress->uiNumProblemsFixed++;
	}

Exit:

	/* End the update. */

	if( bStartedUpdate == TRUE)
	{
		if( RC_BAD( rc2 = chkEndUpdate( pStateInfo, pIxChkInfo)))
		{
			goto Exit;
		}
	}

	rc = (RCODE)((rc != FERR_OK) ? (RCODE)rc : (RCODE)rc2);

	return( rc);
}

/********************************************************************
Desc: Populates the CORRUPT_INFO structure and calls the user's
		callback routine.
*********************************************************************/
FSTATIC RCODE chkReportIxError(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	eCorruptionType	eCorruption,
	FLMUINT				uiErrIx,
	FLMUINT				uiErrDrn,
	FLMBYTE *			pucErrKey,
	FLMUINT				uiErrKeyLen,
	FLMBOOL *			pbFixErrRV
	)
{
	FDB *							pDb = pStateInfo->pDb;
	POOL *						pTmpPool;
	IXD *							pIxd;
	LFILE *						pLFile;
	void *						pIxPoolMark;
	void *						pDbPoolMark = NULL;
	FLMBOOL						bResetKRef = FALSE;
	RCODE							rc = FERR_OK;
	CORRUPT_INFO				CorruptInfo;
	FLMUINT						uiContainerNum;

	f_memset( &CorruptInfo, 0, sizeof( CORRUPT_INFO));

	// Mark the index check pool
	
	pIxPoolMark = GedPoolMark( &(pIxChkInfo->pool));
	pTmpPool = &(pIxChkInfo->pool);

	// Need to mark the DB's temporary pool.  The index code allocates
	// memory for new CDL entries from the DB pool.  If the pool is not
	// reset, it grows during the check and becomes VERY large.

	pDbPoolMark = GedPoolMark( &(pDb->TempPool));

	// Set up the KRef so that flmGetRecKeys will work
	
	if( RC_BAD( rc = KrefCntrlCheck( pDb)))
	{
		goto Exit;
	}
	bResetKRef = TRUE;

	// Report the error

	CorruptInfo.eErrLocale = LOCALE_INDEX;
	CorruptInfo.eCorruption = eCorruption;
	CorruptInfo.uiErrLfNumber = uiErrIx;
	CorruptInfo.uiErrDrn = uiErrDrn;
	CorruptInfo.uiErrElmOffset = pStateInfo->uiElmOffset;

	if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
		pDb->pFile->bInLimitedMode, uiErrIx,
		NULL, &pIxd, TRUE)))
	{
		goto Exit;
	}

	/* Generate the key tree using the key that caused the error */

	if( RC_BAD( rc = chkKeyToTree( pIxd, pucErrKey, uiErrKeyLen,
		&(CorruptInfo.pErrIxKey))))
	{
		goto Exit;
	}

	/* Get the LFile */
	
	if ((uiContainerNum = pIxd->uiContainerNum) == 0)
	{

		// Container number is always the last two bytes of the key.

		flmAssert( uiErrKeyLen > getIxContainerPartLen( pIxd));
		uiContainerNum = getContainerFromKey( pucErrKey, uiErrKeyLen);
	}

	/* Get the LFile */
	
	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, uiContainerNum, &pLFile)))
	{
		goto Exit;
	}

	/* Read the record */

	if( RC_BAD( rc = flmRcaRetrieveRec( pDb, NULL,
		pLFile->uiLfNum, uiErrDrn, FALSE, NULL, NULL, &(CorruptInfo.pErrRecord))))
	{
		if (rc != FERR_NOT_FOUND)
			goto Check_Error;

		// NOTE: Deliberately not bringing in to cache if not found there.

		if( RC_BAD( rc = FSReadRecord( pDb, pLFile, uiErrDrn,
								&(CorruptInfo.pErrRecord), NULL, NULL)))
		{
Check_Error:
			/*
			Record may have been deleted or cannot be returned
			because of an old view error.
			*/

			if( rc == FERR_NOT_FOUND)
			{
				rc = FERR_OK;
			}
			else if( FlmErrorIsFileCorrupt( rc))
			{
				pIxChkInfo->pDbInfo->pProgress->bPhysicalCorrupt = TRUE;
				rc = FERR_OK;
				goto Exit;
			}
			else
			{
				goto Exit;
			}
		}
	}
	
	/* Generate index keys for the current index and record */

	if( CorruptInfo.pErrRecord != NULL)
	{
		if (RC_BAD( rc = flmGetRecKeys( pDb, pIxd,
			CorruptInfo.pErrRecord,
			CorruptInfo.pErrRecord->getContainerID(),
			TRUE, pTmpPool,
			&(CorruptInfo.pErrRecordKeyList))))
		{
			goto Exit;
		}
	}

	*pbFixErrRV = FALSE;
	if ((pIxChkInfo->pDbInfo->fnStatusFunc) &&
		 (RC_OK( pIxChkInfo->pDbInfo->LastStatusRc)))
	{
		pIxChkInfo->pDbInfo->LastStatusRc =
				(*pIxChkInfo->pDbInfo->fnStatusFunc)( FLM_PROBLEM_STATUS,
									(void *)&CorruptInfo,
									(void *)pbFixErrRV,
									pIxChkInfo->pDbInfo->pProgress->AppArg);
	}
Exit:

	if( CorruptInfo.pErrRecord)
	{
		CorruptInfo.pErrRecord->Release();
	}

	if( CorruptInfo.pErrIxKey)
	{
		CorruptInfo.pErrIxKey->Release();
	}

	if( CorruptInfo.pErrRecordKeyList)
	{
		REC_KEY *	pTempKey = CorruptInfo.pErrRecordKeyList;

		while (pTempKey)
		{
			pTempKey->pKey->Release();
			pTempKey = pTempKey->pNextKey;
		}
	}

	// Remove any keys added to the KRef

	if (bResetKRef)
	{
		KYAbortCurrentRecord( pDb);
	}

	// Reset the DB's temporary pool
	
	GedPoolReset( &(pDb->TempPool), pDbPoolMark);

	// Reset the index check pool
	
	GedPoolReset( &(pIxChkInfo->pool), pIxPoolMark);
	return( rc);
}


/***************************************************************************
Desc:	This routine verifies that a key is not unique
*****************************************************************************/
FSTATIC RCODE
	chkVerifyKeyNotUnique(
		STATE_INFO *	pStateInfo,
		FLMUINT			uiIndex,
		FLMBYTE *		pucKey,
		FLMUINT			uiKeyLen,
		FLMUINT *		puiRefCountRV
	)
{
	FDB *			pDb = pStateInfo->pDb;
	RCODE			rc = FERR_OK;
	FlmRecord *	pKeyTree = NULL;
	IXD *			pIxd;
	FLMUINT		uiRefDrn;

	*puiRefCountRV = 0;

	/* Get the IXD */

	if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
					pDb->pFile->bInLimitedMode,
					uiIndex, NULL, &pIxd)))
	{
		goto Exit;
	}

	/*
	This routine should not be called unless the index is
	a unique index.
	*/

	flmAssert( ((pIxd->uiFlags & IXD_UNIQUE) != 0));
	
	/*
	Generate the key tree from the collation key.
	*/

	if( RC_BAD( rc = chkKeyToTree( pIxd, pucKey, uiKeyLen, &pKeyTree)))
	{
		goto Exit;
	}

	// Count up to the first two references for the key.

	if (RC_BAD( rc = FlmKeyRetrieve( (HFDB)pDb,
							uiIndex, pKeyTree->getContainerID(),
							pKeyTree, 0, FO_EXACT,
							NULL, &uiRefDrn)))
	{

		// If the key is NOT found, the problem no longer exists.

		if ((rc == FERR_NOT_FOUND) ||
			 (rc == FERR_BOF_HIT) ||	
			 (rc == FERR_EOF_HIT))
		{
			rc = FERR_OK;
		}
		goto Exit;
	}

	// Found at least one reference.

	*puiRefCountRV = 1;

	// Go exclusive of the last key/reference found to see if there
	// are more references for the key.

	if (RC_BAD( rc = FlmKeyRetrieve( (HFDB)pDb,
							uiIndex, pKeyTree->getContainerID(),
							pKeyTree, uiRefDrn, FO_KEY_EXACT | FO_EXCL,
							NULL, &uiRefDrn)))
	{
		if ((rc == FERR_NOT_FOUND) ||
			 (rc == FERR_BOF_HIT) ||
			 (rc == FERR_EOF_HIT))
		{
			rc = FERR_OK;
		}
		goto Exit;
	}

	// May be more references, but it is sufficient to know that there
	// are at least two.

	*puiRefCountRV = 2;
Exit:
	return( rc);
}


/***************************************************************************
Desc:	
*****************************************************************************/
FSTATIC RCODE chkStartUpdate(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo
	)
{
	FDB *		pDb = pStateInfo->pDb;
	FLMBOOL	bAbortedReadTrans = FALSE;
	RCODE		rc = FERR_OK;
	RCODE		rc2 = FERR_OK;

	if( flmGetDbTransType( pDb) == FLM_READ_TRANS)
	{
		/* Free the KrefCntrl */

		KrefCntrlFree( pDb);

		/* Abort the read transaction */

		if( RC_BAD( rc = flmAbortDbTrans( pDb)))
		{
			goto Exit;
		}
		bAbortedReadTrans = TRUE;

		/* Try to start an update transaction */
	
		if( RC_BAD( rc = flmBeginDbTrans( pDb,
			FLM_UPDATE_TRANS, pIxChkInfo->pDbInfo->uiMaxLockWait, 
			FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
		pIxChkInfo->pDbInfo->bStartedUpdateTrans = TRUE;
	}

	if( RC_BAD( pIxChkInfo->pDbInfo->LastStatusRc))
	{
		rc = pIxChkInfo->pDbInfo->LastStatusRc;
		goto Exit;
	}

Exit:

	/*
	If something went wrong after the update transaction was started,
	abort the transaction.
	*/

	if( RC_BAD( rc))
	{
		if( pIxChkInfo->pDbInfo->bStartedUpdateTrans == TRUE)
		{
			(void)flmAbortDbTrans( pDb);
			pIxChkInfo->pDbInfo->bStartedUpdateTrans = FALSE;
		}
	}

	/*
	Re-start the read transaction.
	*/
	
	if( bAbortedReadTrans == TRUE &&
		pIxChkInfo->pDbInfo->bStartedUpdateTrans == FALSE)
	{
		rc2 = flmBeginDbTrans( pDb, FLM_READ_TRANS, 0, FLM_DONT_POISON_CACHE);
	}

	rc = (RCODE)((rc != FERR_OK) ? (RCODE)rc : (RCODE)rc2);

	return( rc);
}


/***************************************************************************
Desc:	
*****************************************************************************/
FSTATIC RCODE
	chkEndUpdate(
		STATE_INFO *	pStateInfo,
		IX_CHK_INFO *	pIxChkInfo
	)
{
	RCODE		rc = FERR_OK;
	RCODE		rc2 = FERR_OK;

	if( pIxChkInfo->pDbInfo->bStartedUpdateTrans == TRUE)
	{
		/*
		Commit the update transaction that was started.  If the transaction
		started by the application, do not commit it.
		*/
		
		if( RC_BAD( rc = flmCommitDbTrans( pStateInfo->pDb, 0, FALSE)))
		{
			goto Exit;
		}
		pIxChkInfo->pDbInfo->bStartedUpdateTrans = FALSE;
	}

Exit:
	
	/* Re-start read transaction */
	
	if( flmGetDbTransType( pStateInfo->pDb) == FLM_NO_TRANS)
	{
		rc2 = flmBeginDbTrans( pStateInfo->pDb, 
			FLM_READ_TRANS, 0, FLM_DONT_POISON_CACHE);
	}

	rc = (RCODE)((rc != FERR_OK) ? (RCODE)rc : (RCODE)rc2);

	return( rc);
}


/***************************************************************************
Desc:	Initializes a result set for use by the logical check code
*****************************************************************************/
RCODE chkRSInit(
	const char *	pIoPath,
	void **			ppvRSetRV)
{
	RCODE				rc = FERR_OK;
	FResultSet *	pRSet;

	if( (pRSet = f_new FResultSet) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pRSet->Setup( pIoPath,
		chkCompareIxRSEntries, 0, 0, TRUE, FALSE)))
	{
		goto Exit;
	}

	*ppvRSetRV = (void *)pRSet;

Exit:
	
	return( rc);
}

/***************************************************************************
Desc:	Sorts a result set.
*****************************************************************************/
RCODE
	chkRSFinalize(
		IX_CHK_INFO *	pIxChkInfo,
		FLMUINT64 *		pui64TotalEntries
	)
{
	FResultSet *				pRSet = (FResultSet *)(pIxChkInfo->pRSet);
	DB_CHECK_PROGRESS *		pProgress = pIxChkInfo->pDbInfo->pProgress;
	DB_CHECK_PROGRESS			saveInfo;
	RCODE							rc = FERR_OK;

	/*
	Save the current check phase information.
	*/

	f_memcpy( &saveInfo, pProgress, sizeof( DB_CHECK_PROGRESS));
	
	/*
	Set information for the result set sort phase.
	*/
	
	pProgress->iCheckPhase = CHECK_RS_SORT;
	pProgress->bStartFlag = TRUE;
	pProgress->ui64NumRSUnits = 0;
	pProgress->ui64NumRSUnitsDone = 0;

	pRSet->SetCallback( chkRSCallbackFunc, (void *)pIxChkInfo);

	if( RC_BAD( rc = pRSet->Finalize( pui64TotalEntries)))
	{
		goto Exit;
	}

Exit:

	(void)pRSet->SetCallback( NULL, 0);

	/*
	Reset the pProgress information.
	*/

	f_memcpy( pProgress, &saveInfo, sizeof( DB_CHECK_PROGRESS));
	pProgress->bStartFlag = TRUE;

	return( rc);
}

/***************************************************************************
Desc:	Compares result set entries during the finalization stage to allow
		the result set to be sorted and to remove duplicates.
*****************************************************************************/
RCODE chkCompareIxRSEntries(
	void *		vpData1,
	FLMUINT		uiLength1,
	void *		vpData2,
	FLMUINT		uiLength2,
	void *		UserValue,
	FLMINT *		piCompare)
{
	FLMBYTE *	pucData1 = (FLMBYTE *)vpData1;
	FLMBYTE *	pucData2 = (FLMBYTE *)vpData2;
	FLMUINT		uiIxNum1;
	FLMUINT		uiIxNum2;
	FLMUINT		uiDrn1;
	FLMUINT		uiDrn2;

	F_UNREFERENCED_PARM( UserValue);

	uiIxNum1 = (FLMUINT)FB2UW( &(pucData1[ RS_IX_OFFSET]));
	uiIxNum2 = (FLMUINT)FB2UW( &(pucData2[ RS_IX_OFFSET]));
	uiDrn1 = (FLMUINT)FB2UD( &(pucData1[ RS_REF_OFFSET]));
	uiDrn2 = (FLMUINT)FB2UD( &(pucData2[ RS_REF_OFFSET]));

	*piCompare = chkCompareKeySet(
		uiIxNum1, &(pucData1[ RS_KEY_OFFSET]),
		uiLength1 - RS_KEY_OVERHEAD, uiDrn1,
		uiIxNum2, &(pucData2[ RS_KEY_OFFSET]),
		uiLength2 - RS_KEY_OVERHEAD, uiDrn2);

	return( FERR_OK);
}


/***************************************************************************
Desc:	Callback for result set sort progress.
*****************************************************************************/
FLMINT chkRSCallbackFunc(
	RSET_CB_INFO *		pCBInfo)
{
	IX_CHK_INFO *				pIxChkInfo = (IX_CHK_INFO *)pCBInfo->UserValue;
	DB_CHECK_PROGRESS *		pProgress = pIxChkInfo->pDbInfo->pProgress;
	FLMINT						iRetVal = 0;

	f_yieldCPU();

	/*
	Set the status values.
	*/
	
	pProgress->ui64NumRSUnits = pCBInfo->ui64EstTotalUnits;
	pProgress->ui64NumRSUnitsDone = pCBInfo->ui64UnitsDone;

	/*
	Call the progress callback.
	*/

	if (RC_BAD( chkCallProgFunc( pIxChkInfo->pDbInfo)))
	{
		iRetVal = -1;
		goto Exit;
	}

Exit:

	pProgress->bStartFlag = FALSE;
	return( iRetVal);
}
