//-------------------------------------------------------------------------
// Desc:	Key and reference building routines.
// Tabs:	3
//
//		Copyright (c) 1990-1992,1994-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kybuild.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE flmProcessIndexedFld(
	FDB *			pDb,
	IXD_p			pUseIxd,
	IFD *			pIfdChain,
	void **		ppPathFlds,
	FLMUINT		uiLeafFieldLevel,
	FLMUINT		uiAction,
	FLMUINT		uiContainerNum,
	FLMUINT		uiDrn,
	FLMBOOL *	pbHadUniqueKeys,
	FlmRecord *	pRecord,
	void *		pvField);

/****************************************************************************
Desc:		Main driver for processing the fields in a record.
****************************************************************************/
RCODE flmProcessRecFlds(
	FDB *			pDb,						/* Operation context. */
	IXD_p			pIxd,
	FLMUINT		uiContainerNum,		/* Database container number. */
	FLMUINT		uiDrn,					/* Record Drn */
	FlmRecord *	pRecord,					/* Record to build keys for */
	FLMUINT		uiAction,				/* Either DEL or ADD */
	FLMBOOL		bPurgedFldsOk,
	FLMBOOL *	pbHadUniqueKeys)
{
	RCODE			rc = FERR_OK;
	void *		pathFlds[ GED_MAXLVLNUM + 1];
	FLMUINT		uiLeafFieldLevel;
	void *		pvField;

	/* Process each field in the record. */
	pvField = pRecord->root();
	for(;;)
	{
		FLMUINT		uiItemType;
		IFD *			pIfdChain;
		FLMUINT		uiTagNum = pRecord->getFieldID( pvField);
		FLMUINT		uiFldState;
		FLMBOOL		bFldEncrypted;
		FLMUINT		uiEncFlags = 0;
		FLMUINT		uiEncId = 0;
		FLMUINT		uiEncState;

		if( RC_BAD( rc = fdictGetField( pDb->pDict, uiTagNum, &uiItemType,
												&pIfdChain, &uiFldState)))
		{
			// Fill diagnostic error data.
			pDb->Diag.uiInfoFlags |=
				(FLM_DIAG_FIELD_NUM | FLM_DIAG_FIELD_TYPE);
			pDb->Diag.uiFieldNum = uiTagNum;
			pDb->Diag.uiFieldType = (FLMUINT)pRecord->getDataType( pvField);
			goto Exit;
		}
		
		// Check for encryption.
		bFldEncrypted = pRecord->isEncryptedField( pvField);
		if (bFldEncrypted)
		{
			// May still proceed if the field is already encrypted.
			uiEncFlags = pRecord->getEncFlags( pvField);
			
			if (!(uiEncFlags & FLD_HAVE_ENCRYPTED_DATA) &&
				 !pDb->pFile->bInLimitedMode)
			{
				uiEncId =  pRecord->getEncryptionID( pvField);
	
				if (RC_BAD( rc = fdictGetEncInfo( pDb,
															 uiEncId,
															 NULL,
															 &uiEncState)))
				{
					// Fill diagnostic error data.
					pDb->Diag.uiInfoFlags |=
						(FLM_DIAG_FIELD_NUM | FLM_DIAG_ENC_ID);
					pDb->Diag.uiFieldNum = uiTagNum;
					pDb->Diag.uiEncId = uiEncId;
					goto Exit;
				}
	
				// Check the state of the Encryption Record.
				if (uiEncState == ITT_ENC_STATE_PURGE)
				{
					// EncDef record has been marked as 'purged'. So, user is not
					// allowed to add new fields that are encrypted with this EncDef Id.
					pDb->Diag.uiInfoFlags |= (FLM_DIAG_FIELD_NUM | FLM_DIAG_ENC_ID);
					pDb->Diag.uiFieldNum = uiTagNum;
					pDb->Diag.uiEncId = uiEncId;
					rc = RC_SET( FERR_PURGED_ENCDEF_FOUND);
					goto Exit;
				}
			}
			else if (!(uiEncFlags & FLD_HAVE_ENCRYPTED_DATA) &&
						pDb->pFile->bInLimitedMode)
			{
				rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
				goto Exit;
			}

		}

		uiLeafFieldLevel = (FLMINT)pRecord->getLevel( pvField);
		pathFlds[ uiLeafFieldLevel] = pvField;

		/* Check the field state */
		if( uiFldState == ITT_FLD_STATE_PURGE && bPurgedFldsOk == FALSE)
		{
			pDb->Diag.uiInfoFlags |= FLM_DIAG_FIELD_NUM;
			pDb->Diag.uiFieldNum = uiTagNum;
			rc = RC_SET( FERR_PURGED_FLD_FOUND);
			goto Exit;
		}
		else if( (uiFldState == ITT_FLD_STATE_CHECKING 
				 || uiFldState == ITT_FLD_STATE_UNUSED)
		&& ! (uiAction & KREF_DEL_KEYS)
			/* Don't change states while reindexing, because the FDB maynot be in a good state. */
		&& ! (uiAction & KREF_INDEXING_ONLY))	
		{	/* Because a occurance of this field was found update the field's
				state to be 'active' */
			if( RC_BAD( rc = flmChangeItemState( pDb, uiTagNum, ITT_FLD_STATE_ACTIVE)))
			{
				goto Exit;
			}

			// If this is an encrypted field, see if we need to update the state of
			// the EncDef record too.
			if (bFldEncrypted)
			{
				if (( uiEncState == ITT_ENC_STATE_CHECKING ||
						uiEncState == ITT_ENC_STATE_UNUSED) &&
						!(uiAction & KREF_DEL_KEYS) &&
						!(uiAction & KREF_INDEXING_ONLY))
				{
					if( RC_BAD( rc = flmChangeItemState( pDb, uiEncId, ITT_ENC_STATE_ACTIVE)))
					{
						goto Exit;
					}
				}
			}
		}

		if( uiItemType != pRecord->getDataType( pvField) &&
			uiTagNum < FLM_DICT_FIELD_NUMS)
		{
			rc = RC_SET( FERR_BAD_FIELD_TYPE);
			pDb->Diag.uiInfoFlags |=
				(FLM_DIAG_FIELD_NUM | FLM_DIAG_FIELD_TYPE);
			pDb->Diag.uiFieldNum = uiTagNum;
			pDb->Diag.uiFieldType = (FLMUINT)pRecord->getDataType( pvField);
			goto Exit;
		}

		if( pRecord->getDataType( pvField) == FLM_BLOB_TYPE && 
			(!(uiAction & KREF_INDEXING_ONLY)))
		{
			if( RC_BAD( rc = flmBlobPlaceInTransactionList(
					pDb, ((uiAction & KREF_DEL_KEYS)
							? BLOB_DELETE_ACTION : BLOB_ADD_ACTION),
					pRecord, pvField)))
			{
				goto Exit;
			}
		}

		if( pIfdChain)
		{
			if (RC_BAD( rc = flmProcessIndexedFld( pDb, pIxd, pIfdChain, pathFlds,
				uiLeafFieldLevel, uiAction, uiContainerNum, uiDrn, pbHadUniqueKeys,
				pRecord, pvField)))
			{
				goto Exit;
			}
		}
		
		if( (pvField = pRecord->next( pvField)) == NULL)
		{
			break;
		}
	}

Exit:

	// Build and add the compound keys to the KREF table

	if( RC_OK( rc))
	{
		rc = KYBuildCmpKeys( pDb, uiAction, uiContainerNum,
						uiDrn, pbHadUniqueKeys, pRecord);
	}
	
	return( rc);
}


/***************************************************************************
Desc:	See if a field's path matches the path in the IFD.
*****************************************************************************/
FLMBOOL flmCheckIfdPath(
	IFD *			pIfd,
	FlmRecord *	pRecord,
	void **		ppPathFlds,
	FLMUINT		uiLeafFieldLevel,
	void *		pvLeafField,
	void **		ppvContextField	// Return's context field of field in record
											// i.e., the first field in the IFD.  Only
											// returned if the context matches.
	)
{
	FLMBOOL		bMatched = FALSE;
	void *		pvContextField;
	FLMINT		iParentPos;
	FLMUINT *	puiIfdFldPathCToP;

	// Check the field path to see if field is in context.

	pvContextField = pvLeafField;
	puiIfdFldPathCToP = &pIfd->pFieldPathCToP [1];
	iParentPos = (FLMINT)uiLeafFieldLevel - 1;
	while (*puiIfdFldPathCToP && iParentPos >= 0)
	{
		pvContextField = ppPathFlds [iParentPos];

		// Check for FLM_ANY_FIELD (wild_tag) and skip it.

		if (*puiIfdFldPathCToP == FLM_ANY_FIELD)
		{

			// Look at next field in IFD path to see if it matches
			// the current field.  If it does, continue from there.

			if (*(puiIfdFldPathCToP + 1))
			{
				if (pRecord->getFieldID( pvContextField) ==
							*(puiIfdFldPathCToP + 1))
				{

					// Skip wild card and field that matched.

					puiIfdFldPathCToP += 2;
				}

				// Go to next field in path being evaluated no matter
				// what.  If it didn't match, we continue looking at
				// the wild card.  If it did match, we go to the next
				// field in the path.

				iParentPos--;
			}
			else
			{

				// Rest of path is an automatic match - had wildcard
				// at top of IFD path.

				// It's not really necessary to increment this, but
				// it is more efficient because of the comparisons
				// that are done when we exit this loop.

				puiIfdFldPathCToP++;
				pvContextField = ppPathFlds [0];
				break;
			}
		}
		else if (pRecord->getFieldID( pvContextField) != *puiIfdFldPathCToP)
		{

			// Field does not match current field in IFD.
			// This jump to Exit will return FALSE.  bMatched is FALSE at
			// this point.

			goto Exit;
		}
		else
		{

			// Go up a level in the record and the IFD path - to parent.

			iParentPos--;
			puiIfdFldPathCToP++;
		}
	}

	// If we got to the end of the field path in the IFD, we have a
	// match.

	if (!(*puiIfdFldPathCToP) ||
		  (*puiIfdFldPathCToP == FLM_ANY_FIELD &&
		  !(*(puiIfdFldPathCToP + 1))))
	{
		*ppvContextField = pvContextField;
		bMatched = TRUE;
	}
Exit:
	return( bMatched);
}

/****************************************************************************
Desc:		Processes a field in a record - indexing, blob, etc.
****************************************************************************/
FSTATIC RCODE flmProcessIndexedFld(
	FDB *					pDb,
	IXD_p					pUseIxd,
	IFD *					pIfdChain,
	void **				ppPathFlds,
	FLMUINT				uiLeafFieldLevel,
	FLMUINT				uiAction,
	FLMUINT				uiContainerNum,
	FLMUINT				uiDrn,
	FLMBOOL *			pbHadUniqueKeys,
	FlmRecord *			pRecord,
	void *				pvField)
{
	RCODE					rc = FERR_OK;
	IFD_p					pIfd;
	IXD_p					pIxd;
	void *				pRootContext;
	const FLMBYTE *	pValue;
	const FLMBYTE *	pExportValue;
	FLMUINT				uiValueLen;
	FLMUINT				uiKeyLen;
	FLMBYTE				pTmpKeyBuf[ MAX_KEY_SIZ ];

	pTmpKeyBuf[0] = '\0';

	for ( pIfd = pIfdChain; pIfd; pIfd = pIfd->pNextInChain)
	{
		if( pUseIxd)
		{
			if( pUseIxd->uiIndexNum == pIfd->uiIndexNum)
			{
				pIxd = pUseIxd;
			}
			else
			{
				continue;
			}
		}
		else
		{
			pIxd = pIfd->pIxd;
			
			// If index is offline or on a different container, skip it.
			// NOTE: if pIxd->uiContainerNum is zero, the index is indexing
			// ALL containers.

			if (pIxd->uiContainerNum)
			{
				if (pIxd->uiContainerNum != uiContainerNum)
				{
					continue;
				}

				if( pIxd->uiFlags & IXD_OFFLINE)
				{
					if( uiDrn > pIxd->uiLastDrnIndexed)
					{
						continue;
					}
					// Else index the key.
				}
			}
			else
			{

				// uiContainerNum == 0, indexing all containers

				if( pIxd->uiFlags & IXD_OFFLINE)
				{
					if( uiContainerNum > pIxd->uiLastContainerIndexed ||
						 uiContainerNum == pIxd->uiLastContainerIndexed &&
						 uiDrn > pIxd->uiLastDrnIndexed)
					{
						continue;
					}
					// Else index the key.
				}
			}
		}

		// See if field path matches what is defined in the IFD.

		if (!flmCheckIfdPath( pIfd, pRecord, ppPathFlds, uiLeafFieldLevel,
				pvField, &pRootContext))
		{
			// Skip this field.
			continue;
		}

		// Field passed the path verification.  Now output the KEY.

		if( pIfd->uiFlags & IFD_COMPOUND)
		{
			/* Compound Key. */

			if (RC_BAD( rc = KYCmpKeyAdd2Lst( pDb, pIxd, pIfd, 
														 pvField, pRootContext)))
				goto Exit;
		}
		else if (pIfd->uiFlags & IFD_CONTEXT)
		{
			FLMBYTE KeyBuf [4];

				/* Context key (tag number). */

			KeyBuf [0] = KY_CONTEXT_PREFIX;
			flmUINT16ToBigEndian( (FLMUINT16)pRecord->getFieldID( pvField ), &KeyBuf [1]);

			if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
									pIfd, uiAction, uiDrn,
									pbHadUniqueKeys, KeyBuf, KY_CONTEXT_LEN,
									TRUE, FALSE, FALSE)))
			{
				goto Exit;
			}
		}
		else if ((pIfd->uiFlags & IFD_SUBSTRING) &&
					(pRecord->getDataType( pvField) == FLM_TEXT_TYPE))
		{
			FLMBOOL		bFirstSubstring = TRUE;
			FLMUINT		uiLanguage = pIxd->uiLanguage;
			
			// An encrypted field, in limited mode means we use the encrypted data instead.
			if (pRecord->isEncryptedField( pvField) && pDb->pFile->bInLimitedMode)
			{
				pValue = pRecord->getEncryptionDataPtr( pvField);
				uiValueLen = pRecord->getEncryptedDataLength( pvField);

				if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
											pIfd, uiAction,  uiDrn, pbHadUniqueKeys,
											pValue, uiValueLen, FALSE, bFirstSubstring, TRUE)))
				{
					goto Exit;
				}
			}
			else
			{
				pExportValue = pValue = pRecord->getDataPtr( pvField);
				uiValueLen = pRecord->getDataLength( pvField);

				/* Loop for each word in the text field adding it to the table. */

				while (KYSubstringParse( &pValue, &uiValueLen, 
												pIfd->uiFlags, pIfd->uiLimit,
												(FLMBYTE *) pTmpKeyBuf, &uiKeyLen) == TRUE)
				{
					if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
												pIfd, uiAction,  uiDrn, pbHadUniqueKeys,
												(FLMBYTE *) pTmpKeyBuf, uiKeyLen, FALSE, 
												bFirstSubstring, FALSE)))
					{
						break;
					}
	
					if( (uiValueLen == 1 && 
							!(uiLanguage >= FIRST_DBCS_LANG && 
								uiLanguage <= LAST_DBCS_LANG)))
					{
						break;
					}
	
					bFirstSubstring = FALSE;
				}
	
				if (RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}
		else if ((pIfd->uiFlags & IFD_EACHWORD) &&
					(pRecord->getDataType( pvField) == FLM_TEXT_TYPE))
		{
			// An encrypted field, in limited mode means we use the encrypted data instead.
			if (pRecord->isEncryptedField( pvField) && pDb->pFile->bInLimitedMode)
			{
				pValue = pRecord->getEncryptionDataPtr( pvField);
				uiValueLen = pRecord->getEncryptedDataLength( pvField);

				if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
										pIfd, uiAction, uiDrn, pbHadUniqueKeys,
										pValue, uiValueLen, FALSE, FALSE, TRUE)))
				{
					goto Exit;
				}
			}
			else
			{
				pExportValue = pValue = pRecord->getDataPtr( pvField);
				uiValueLen = pRecord->getDataLength( pvField);

				/* Loop for each word in the text field adding it to the table. */

				while (KYEachWordParse( &pValue, &uiValueLen,
												pIfd->uiLimit,
												(FLMBYTE *) pTmpKeyBuf, &uiKeyLen) == TRUE)
				{
					if (RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
											pIfd, uiAction, uiDrn, pbHadUniqueKeys,
											(FLMBYTE *) pTmpKeyBuf, uiKeyLen, FALSE, FALSE,
											FALSE)))
					{
						break;
					}
				}
	
				if (RC_BAD( rc))
				{
					goto Exit;
				}
			}
		}
		else
		{
			/* Index field content - entire field. */
			FLMBOOL			bEncryptedKey = FALSE;
			
			// An encrypted field, in limited mode means we use the encrypted data instead.
			if (pRecord->isEncryptedField( pvField) && pDb->pFile->bInLimitedMode)
			{
				pExportValue = pValue = pRecord->getEncryptionDataPtr( pvField);
				uiValueLen = pRecord->getEncryptedDataLength( pvField);
				bEncryptedKey = TRUE;
			}
			else
			{
				pExportValue = pValue = pRecord->getDataPtr( pvField);
				uiValueLen = pRecord->getDataLength( pvField);
			}

			if( RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
					pIfd, uiAction, uiDrn, pbHadUniqueKeys,
					pExportValue, uiValueLen,
					FALSE, FALSE, bEncryptedKey)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Add an index key to the buffers
****************************************************************************/
RCODE KYAddToKrefTbl(
	FDB *					pDb,
	IXD_p					pIxd,
	FLMUINT				uiContainerNum,
	IFD_p					pIfd,
	FLMUINT				uiAction,
	FLMUINT				uiDrn,
	FLMBOOL *			pbHadUniqueKeys,
	const FLMBYTE *	pKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bAlreadyCollated,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bFldIsEncrypted)
{
	RCODE					rc = FERR_OK;
	KREF_ENTRY_p		pKref;
	FLMBYTE *			pKrefKey;
	FLMUINT				uiKrefKeyLen;
	FLMUINT				uiSizeNeeded;
	KREF_CNTRL_p		pKrefCntrl = &pDb->KrefCntrl;

	/*
	Removed if( !uiKeyLen) goto Exit;
	As of Oct98, we need to store keys that have no value.
	This may or may not have an inpact on GroupWise (doubtful).
	On single value fields, we will store a one byte x03 value.
	A Collated key will not be zero length.
	*/

#ifdef FLM_NLM
	// GWBug to give up CPU when indexing tons of fields for a record.
	if( (pKrefCntrl->uiCount & 0x7F) == 0x7F)
		f_yieldCPU();
#endif

	/* If the table is FULL, commit the keys or expand the table */

	if (pKrefCntrl->uiCount == pKrefCntrl->uiKrefTblSize)
	{
		FLMUINT		uiAllocSize;
		FLMUINT		uiOrigKrefTblSize = pKrefCntrl->uiKrefTblSize;

		if( pKrefCntrl->uiKrefTblSize > 0x8000 / sizeof( KREF_ENTRY_p))
		{
			pKrefCntrl->uiKrefTblSize += 4096;
		}
		else
		{
			pKrefCntrl->uiKrefTblSize *= 2;
		}

		// GWBUG #30146 1/13/97: Let the table grow until memory error.

		uiAllocSize = pKrefCntrl->uiKrefTblSize * sizeof( KREF_ENTRY_p );

		if( RC_BAD( rc = f_realloc( uiAllocSize, &pKrefCntrl->pKrefTbl)))
		{
			pKrefCntrl->uiKrefTblSize = uiOrigKrefTblSize;
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	/* Get the collated key. */

	if (bAlreadyCollated)
	{
		/* Compound keys are already collated. */

		pKrefKey = (FLMBYTE *)pKey;
		uiKrefKeyLen = uiKeyLen;
	}
	else
	{
		pKrefKey = pKrefCntrl->pKrefKeyBuf;
		uiKrefKeyLen = (pIxd->uiContainerNum)
							? MAX_KEY_SIZ
							: MAX_KEY_SIZ - getIxContainerPartLen( pIxd);

		if( RC_BAD( rc = KYCollateValue( pKrefKey, &uiKrefKeyLen,
				pKey, uiKeyLen, pIfd->uiFlags, pIfd->uiLimit,
				NULL, NULL,
				(FLMUINT)((pIxd->uiLanguage != 0xFFFF)
							 ? pIxd->uiLanguage
							 : pDb->pFile->FileHdr.uiDefaultLanguage),
							FALSE, bFirstSubstring, FALSE, NULL, NULL,
							bFldIsEncrypted)))
		{
			goto Exit;
		}
	}

	// If indexing all containers, add the container
	// number.

	if (!pIxd->uiContainerNum)
	{
		appendContainerToKey( pIxd, uiContainerNum, pKrefKey, &uiKrefKeyLen);
	}

	/*
	Allocate memory for the key's KREF and the key itself.
	We allocate one extra byte so we can NULL terminate the key
	below.  The extra NULL character is to ensure that the compare
	in the qsort routine will work.  GedPoolAlloc machine aligns.
	*/

	uiSizeNeeded = sizeof( KREF_ENTRY) + uiKrefKeyLen + 1;

	if( (pKref = (KREF_ENTRY_p)
			 GedPoolAlloc( pKrefCntrl->pPool, uiSizeNeeded )) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pKrefCntrl->pKrefTbl [ pKrefCntrl->uiCount++ ] = pKref;
	pKrefCntrl->uiTotalBytes += uiSizeNeeded;

	/* Fill in all of the fields in the KREF structure. */

	flmAssert( pIxd->uiIndexNum > 0 &&
		pIxd->uiIndexNum < FLM_UNREGISTERED_TAGS); // Sanity check
	pKref->ui16IxNum = (FLMUINT16)pIxd->uiIndexNum;
	pKref->uiDrn = uiDrn;
	if (uiAction & KREF_DEL_KEYS)
	{
		pKref->uiFlags = ((uiAction & KREF_MISSING_KEYS_OK)
							  			  ? (FLMUINT)(KREF_DELETE_FLAG |
										  				 KREF_MISSING_OK)
							  			  : (FLMUINT)(KREF_DELETE_FLAG));
	}
	else
	{
		pKref->uiFlags = 0;
	}
	if( pIxd->uiFlags & IXD_UNIQUE)
	{
		*pbHadUniqueKeys = TRUE;
		pKref->uiFlags |= KREF_UNIQUE_KEY;
	}
	if (bFldIsEncrypted)
	{
		pKref->uiFlags |= KREF_ENCRYPTED_KEY;
	}
	pKref->ui16KeyLen = (FLMUINT16)uiKrefKeyLen;
	pKref->uiTrnsSeq = pKrefCntrl->uiTrnsSeqCntr;

	// Null terminate the key so compare in qsort will work
	
  	pKrefKey[ uiKrefKeyLen++ ] = '\0';

	// Copy the key to just after the KREF structure
	
 	f_memcpy( (FLMBYTE *) (&pKref [1]), pKrefKey, uiKrefKeyLen);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Encrypt a field in the pRecord.
****************************************************************************/
RCODE flmEncryptField(
	FDICT *			pDict,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiEncId,
	POOL *			pPool)
{
	RCODE				rc = FERR_OK;
	F_CCS *			pCcs;
	FLMUINT			uiEncLength;
	FLMBYTE *		pucEncBuffer;
	FLMBYTE *		pucDataBuffer = NULL;
	FLMUINT			uiCheckLength;
	void *			pvMark;
#ifdef FLM_DEBUG
	FLMBOOL			bOk;
	FLMUINT			uiLoop;
#endif

	pvMark = GedPoolMark( pPool);

#ifdef FLM_CHECK_RECORD
	if (RC_BAD( rc = pRecord->checkRecord()))
	{
		goto Exit;
	}
#endif

	if ( !pRecord->isEncryptedField( pvField))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FLD_NOT_ENCRYPTED);
		goto Exit;
	}

	pCcs = (F_CCS *)pDict->pIttTbl[ uiEncId].pvItem;

	flmAssert( pCcs);

	uiEncLength = pRecord->getEncryptedDataLength( pvField);

	pucDataBuffer = (FLMBYTE *)GedPoolAlloc( pPool, uiEncLength);

	if (!pucDataBuffer)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pucEncBuffer = (FLMBYTE *)pRecord->getEncryptionDataPtr( pvField);
	uiCheckLength = uiEncLength;

#ifdef FLM_DEBUG
	// Preset the buffer to a known value so we can check
	// it after the encryption.  It should NOT be the same!

	f_memset( pucEncBuffer, 'B', uiEncLength);
#endif

	// We copy the data into a buffer that is as large as the encrypted data
	// because the encryption algorithm is expecting to get a buffer
	// that does not need to be padded to the nearest 16 byte boundary.
	f_memcpy( pucDataBuffer,
				 pRecord->getDataPtr( pvField),
				 pRecord->getDataLength( pvField));

	if (RC_BAD( rc = pCcs->encryptToStore( pucDataBuffer, uiEncLength,
		pucEncBuffer, &uiCheckLength)))
	{
		goto Exit;
	}

	if (uiCheckLength != uiEncLength)
	{
		rc = RC_SET( FERR_DATA_SIZE_MISMATCH);
		goto Exit;
	}

#ifdef FLM_DEBUG
	bOk = FALSE;
	for (uiLoop = 0; uiLoop < uiEncLength; uiLoop++)
	{
		if( pucEncBuffer[ uiLoop] != 'B')
		{
			bOk = TRUE;
			break;
		}
	}

	if ( !bOk)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}
#endif

	pRecord->setEncFlags( pvField, FLD_HAVE_DECRYPTED_DATA |
											 FLD_HAVE_ENCRYPTED_DATA);

#ifdef FLM_CHECK_RECORD
	if (RC_BAD( rc = pRecord->checkRecord()))
	{
		goto Exit;
	}
#endif

Exit:

	GedPoolReset( pPool, pvMark);

	return( rc);

}


/****************************************************************************
Desc:	Decrypt an encrypted field in the pRecord.
****************************************************************************/
RCODE flmDecryptField(
	FDICT *			pDict,
	FlmRecord *		pRecord,
	void *			pvField,
	FLMUINT			uiEncId,
	POOL *			pPool)
{
	RCODE				rc = FERR_OK;
	F_CCS *			pCcs;
	FLMUINT			uiEncLength;
	FLMBYTE *		pucEncBuffer = NULL;
	FLMBYTE *		pucDataBuffer = NULL;
	FLMUINT			uiCheckLength;
	void *			pvMark = NULL;

	pvMark = GedPoolMark( pPool);

#ifdef FLM_CHECK_RECORD
	if (RC_BAD( rc = pRecord->checkRecord()))
	{
		goto Exit;
	}
#endif

	if ( !pRecord->isEncryptedField( pvField))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FLD_NOT_ENCRYPTED);
		goto Exit;
	}

	pCcs = (F_CCS *)pDict->pIttTbl[ uiEncId].pvItem;

	flmAssert( pCcs);

	uiEncLength = pRecord->getEncryptedDataLength( pvField);
	pucDataBuffer = (FLMBYTE *)GedPoolAlloc( pPool, uiEncLength);
	
	if (!pucDataBuffer)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pucEncBuffer = (FLMBYTE *)pRecord->getEncryptionDataPtr( pvField);
	uiCheckLength = uiEncLength;

	if( RC_BAD( rc = pCcs->decryptFromStore( pucEncBuffer, uiEncLength,
		pucDataBuffer, &uiCheckLength)))
	{
		goto Exit;
	}

	if (uiCheckLength != uiEncLength)
	{
		rc = RC_SET( FERR_DATA_SIZE_MISMATCH);
		goto Exit;
	}

	f_memcpy( pRecord->getDataPtr( pvField),
				 pucDataBuffer,
				 pRecord->getDataLength(pvField));

	pRecord->setEncFlags( pvField, FLD_HAVE_DECRYPTED_DATA |
											 FLD_HAVE_ENCRYPTED_DATA);

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = pRecord->checkRecord()))
	{
		goto Exit;
	}
#endif

Exit:

	GedPoolReset( pPool, pvMark);
	return( rc);
}
