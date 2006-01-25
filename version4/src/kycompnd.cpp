//-------------------------------------------------------------------------
// Desc:	Compound key building routines.
// Tabs:	3
//
//		Copyright (c) 1990-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kycompnd.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE KYCmpKeyElmBld(
	FDB *				pDb,
	IXD_p				pIxd,
	FLMUINT			uiContainerNum,
	IFD_p				pIfd,
	FLMUINT			uiAction,
	FLMUINT			uiDrn,
	FLMBOOL *		pbHadUniqueKeys,
	FLMUINT			uiCdlEntry,
	FLMUINT			uiCompoundPos,
	FLMBYTE *		pKeyBuf,
	FLMUINT			uiKeyLen,
	FLMBYTE *		pLowUpBuf,
	FLMUINT			uiLuLen,
	FlmRecord *		pRecord,
	FLD_CONTEXT *	pFldContext);

/****************************************************************************
Desc:	Add an field into the CDL (Compound Data List) for this ISK.
****************************************************************************/
RCODE KYCmpKeyAdd2Lst(
	FDB *			pDb,
	IXD_p			pIxd,				/* Index definition. */
	IFD_p			pIfd,				/* Index field definition. */
	void *		pvField,				/* Field whose value is part of the key. */
	void *		pRootContext)	/* Points to root context of field path. */
{
	CDL_p			pCdl;
	KREF_CNTRL_p pKrefCntrl;
	CDL_p  *		ppCdlTbl;
	RCODE			rc = FERR_OK;
	FLMUINT		uiCdlEntry;
	FLMUINT		uiIxEntry;

	/* OCT98, Need to handle case of zero length data coming in. */
 	pKrefCntrl = &pDb->KrefCntrl;
	ppCdlTbl = pKrefCntrl->ppCdlTbl;
	flmAssert( (ppCdlTbl != NULL) );

	/* Figure out which CDL and index entry to use. */

	uiIxEntry = (FLMUINT) (pIxd - pDb->pDict->pIxdTbl);
	uiCdlEntry = (FLMUINT) (pIfd - pDb->pDict->pIfdTbl);

	/*
	2/25/99 - Removed code to not add the field if a duplicate
	value is found.  This dropped index keys with multiple contexts.
	*/

	if( (pCdl = (CDL_p)GedPoolAlloc( &pDb->TempPool,
												sizeof( CDL))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	flmAssert( (pKrefCntrl->pIxHasCmpKeys != NULL) );

	pKrefCntrl->pIxHasCmpKeys [uiIxEntry] = TRUE;
	pCdl->pField = pvField;
	pCdl->pRootContext = pRootContext;

	/* Insert at first of CDL list */

	pCdl->pNext = ppCdlTbl [uiCdlEntry];
	ppCdlTbl [uiCdlEntry] = pCdl;
	pKrefCntrl->bHaveCompoundKey = TRUE;

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Called when an entire record has been processed by the key 
			building functions. Builds and add all compound keys to the table.
****************************************************************************/
RCODE KYBuildCmpKeys(
	FDB *				pDb,
	FLMUINT			uiAction,
	FLMUINT			uiContainerNum,
	FLMUINT			uiDrn,
	FLMBOOL *		pbHadUniqueKeys,
	FlmRecord *		pRecord)
{
	KREF_CNTRL_p	pKrefCntrl = &pDb->KrefCntrl;
	CDL_p  *			ppCdlTbl = pKrefCntrl->ppCdlTbl;
	FLMBYTE *		pKeyBuf = pKrefCntrl->pKrefKeyBuf;
	FLMBYTE *		pIxHasCmpKeys = pKrefCntrl->pIxHasCmpKeys;
	IXD_p				pIxd;
	IFD_p				pIfd;
	IFD_p				pFirstIfd;
	FLMUINT			uiFirstCdlEntry;
	FLMUINT			uiCdlEntry;
	RCODE				rc = FERR_OK;
	FLMBOOL			bBuildCmpKeys;
	FLMUINT			uiIxEntry;
	FLMUINT			uiTotalIndexes;
  	FLMUINT			uiIfdCnt;
	FLMUINT			uiKeyLen;
	FLMBYTE    		LowUpBuf [MAX_LOWUP_BUF];
	FLD_CONTEXT		fldContext;
	FDICT *			pDict = pDb->pDict;

	LowUpBuf[0] = '\0';

	if( pKrefCntrl->bHaveCompoundKey == FALSE)
		goto Exit;
	pKrefCntrl->bHaveCompoundKey = FALSE;
	flmAssert( (pKeyBuf != NULL) && (pIxHasCmpKeys != NULL ));

	// Loop through all of the indexes looking for a CDL entry.
	// VISIT: We need to find the indexes faster than looping!

	uiTotalIndexes = pDict->uiIxdCnt;
  	for (uiIxEntry = 0; uiIxEntry < uiTotalIndexes; uiIxEntry++)
	{
		// See if the index has compound keys to build.

		if( !pIxHasCmpKeys [uiIxEntry])
		{
			continue;
		}
		pIxd = pDict->pIxdTbl + uiIxEntry;
		pIxHasCmpKeys [uiIxEntry] = FALSE;		// Clear the flag.
		bBuildCmpKeys = TRUE;

		// Make sure that all required fields are present.

		pFirstIfd = pIfd = pIxd->pFirstIfd;
		uiCdlEntry = uiFirstCdlEntry = (FLMUINT) (pFirstIfd - pDict->pIfdTbl);
		for (uiIfdCnt = 0;
			  uiIfdCnt < pIxd->uiNumFlds;
			  pIfd++, uiCdlEntry++, uiIfdCnt++)
		{
			FLMUINT		uiCompoundPos;
			FLMBOOL 		bHitFound;
			
			// Loop on each compound field piece looking for REQUIRED field
			// without any data - then we don't have to build a key.
			
			bHitFound = (pIfd->uiFlags & IFD_REQUIRED_PIECE) ? FALSE : TRUE;
			uiCompoundPos = pIfd->uiCompoundPos;
			for(;;)
			{
				if( !bHitFound)
				{
					if( ppCdlTbl [uiCdlEntry])
					{
						bHitFound = TRUE;					// Loop through all ixds
					}	
				}
				if( (pIfd->uiFlags & IFD_LAST) 
				 || ((pIfd+1)->uiCompoundPos != uiCompoundPos))
					break;
				pIfd++;
				uiCdlEntry++;
				uiIfdCnt++;
			}
			if( !bHitFound)
			{
				bBuildCmpKeys = FALSE;
				break;
			}
		}
		
		//  Build the individual compound keys.

		if( bBuildCmpKeys)
      {
			uiKeyLen = 0;
			f_memset( &fldContext, 0, sizeof(FLD_CONTEXT));

			if( RC_BAD(rc = KYCmpKeyElmBld( pDb, pIxd, uiContainerNum,
										pFirstIfd,
										uiAction, uiDrn, pbHadUniqueKeys,
										uiFirstCdlEntry, 0, pKeyBuf,
										uiKeyLen, LowUpBuf, 0,
										pRecord, &fldContext)))
			{
				goto Exit;
			}
		}

		/* Reset the CDL pointers to NULL. */
		/* VISIT: It would be faster to
			memset the whole thing in a single call. */

		f_memset( (void *) (&ppCdlTbl [ uiFirstCdlEntry ]),
					 0, sizeof(CDL_p) * pIxd->uiNumFlds);
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Build all compound keys for a record.
Notes:	This routine is recursive in nature.  Will recurse the number of
			levels you have defined in the compound field.  Please note that luLen
			and uiKeyLen are never modified.  This is so the while loop does not have
			to reset them for every repeating field in the cdl link.
****************************************************************************/
FSTATIC RCODE KYCmpKeyElmBld(
	FDB *				pDb,
	IXD_p				pIxd,					// Index definition.
	FLMUINT			uiContainerNum,
	IFD_p				pIfd,					// Index field definition.
	FLMUINT			uiAction,
	FLMUINT			uiDrn,
	FLMBOOL *		pbHadUniqueKeys,
	FLMUINT			uiCdlEntry,			// CDL entry for the IFD. 
  	FLMUINT			uiCompoundPos,		// Compound Piece number - zero based
	FLMBYTE *		pKeyBuf,				// Key buffer to build the key in
	FLMUINT			uiKeyLen,			// Total length left in the key buffer
	FLMBYTE *		pLowUpBuf,			// For POST compound keys place bits here.
	FLMUINT    		uiLuLen,				// Length used in pLowUpBuf.
	FlmRecord *		pRecord,				// Record being indexed.
	FLD_CONTEXT *	pFldContext			// State to verify all fields are siblings.
	)
{
	RCODE					rc = FERR_OK;
	CDL_p	*				pCdlTbl = pDb->KrefCntrl.ppCdlTbl;
	CDL_p					pCdl = pCdlTbl [uiCdlEntry];
	FLMBYTE *			pTmpBuf = NULL;
	void *				pvMark = NULL;
	IFD_p					pNextIfdPiece;
	void *				pvField;
	void *				pSaveParentAnchor;
	FLMUINT				uiNextCdlEntry;
	FLMBOOL				bBuiltKeyPiece;
	FLMUINT				uiElmLen;
  	FLMUINT     		uiPostFlag;
	FLMUINT    			uiPostLen;
	FLMUINT		  		uiTempLuLen;
	FLMUINT				uiPieceLuLen;
	FLMUINT				uiNextPiecePos;			// 0 if this is the last piece.
	FLMUINT				uiLanguage;
	FLMUINT				uiMaxKeySize = (pIxd->uiContainerNum)
											? MAX_KEY_SIZ
											: MAX_KEY_SIZ - getIxContainerPartLen( pIxd);
	FLMBOOL				bFldIsEncrypted = FALSE;

	if ((uiLanguage = pIxd->uiLanguage) == 0xFFFF)
	{
		uiLanguage = pDb->pFile->FileHdr.uiDefaultLanguage;
	}

	// Test for compound key being tons of levels.  Still need to code for.
	flmAssert( uiCompoundPos < MAX_COMPOUND_PIECES);

	/* Set if this piece is part of post. */

	uiPostFlag = IFD_IS_POST_TEXT( pIfd);

	/* Add the DELIMITER, except on the first key element. */

	if( uiCompoundPos != 0)
	{
		IFD_p			pPrevIfd = pIfd - 1;		// Works because IFD is on first
														// if they repeat.
		if( (uiLanguage >= FIRST_DBCS_LANG) &&
			 (uiLanguage <= LAST_DBCS_LANG) &&
			 (IFD_GET_FIELD_TYPE( pPrevIfd) == FLM_TEXT_TYPE) &&
			 (!(pPrevIfd->uiFlags & IFD_CONTEXT)))
		{
			pKeyBuf [uiKeyLen++] = 0;
		}
		pKeyBuf [uiKeyLen++] = COMPOUND_MARKER;
	}

	// Determine the next IFD compound piece.

	for( pNextIfdPiece = (IFD_p)NULL,
			uiNextCdlEntry = uiCdlEntry + 1,
			uiNextPiecePos = 0
		; ((pIfd+uiNextPiecePos)->uiFlags & IFD_LAST) == 0
		; )
	{
		if( (pIfd+uiNextPiecePos)->uiCompoundPos !=
			 (pIfd+uiNextPiecePos+1)->uiCompoundPos)
		{
			pNextIfdPiece = pIfd + uiNextPiecePos + 1;
			uiNextCdlEntry = uiCdlEntry + uiNextPiecePos + 1;
			break;
		}

		if( !pCdl)
		{
			pIfd++;
			pCdl = pCdlTbl [ ++uiCdlEntry];
			uiNextCdlEntry = uiCdlEntry + 1;
		}
		else
			uiNextPiecePos++;
	}

	pSaveParentAnchor = pFldContext->pParentAnchor;
	bBuiltKeyPiece = FALSE;	

	/* Loop on each CDL, but do at least once. */

	while( pCdl || !bBuiltKeyPiece)
	{
		// Restore context values for each iteration.
		pFldContext->pParentAnchor = pSaveParentAnchor;

		/*
		If there is a field to process, verify that its path is
		relative to the previous non-null compound pieces.
		*/
		if( pCdl)
		{
			pvField = pCdl->pField;
			
			// Validate the current and previous root contexts.

			if( KYValidatePathRelation( pRecord, pCdl->pRootContext, pvField,
										pFldContext, uiCompoundPos) == FERR_FAILURE)
			{
				// This field didn't pass the test, get the next field.
				goto Next_CDL_Field;
			}
		}
		else
		{
			pvField = NULL;
		}
		bBuiltKeyPiece = TRUE;
		uiPostLen = uiElmLen = 0;
		uiTempLuLen = uiLuLen;

		if( pCdl && (pIfd->uiFlags & (IFD_EACHWORD|IFD_SUBSTRING)) 
			&& (pRecord->getDataType( pvField) == FLM_TEXT_TYPE) 
			&& pRecord->getDataLength( pvField)
			&&	((!pRecord->isEncryptedField( pvField)
				|| (pRecord->isEncryptedField( pvField)
				&&  pDb->pFile->bInLimitedMode))))
		{
			const FLMBYTE *	pText = pRecord->getDataPtr( pvField);
			FLMUINT				uiTextLen = pRecord->getDataLength( pvField);
			FLMUINT	 			uiWordLen;
			FLMBOOL				bReturn;
			FLMBOOL				bFirstSubstring = (pIfd->uiFlags & IFD_SUBSTRING)
												? TRUE : FALSE;

			if( !pTmpBuf)
			{
				pvMark = GedPoolMark( &pDb->TempPool);
				if( (pTmpBuf = (FLMBYTE *)GedPoolAlloc( &pDb->TempPool,
					(FLMUINT)MAX_KEY_SIZ + 8)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Cleanup1;
				}
			}

			// Loop on each WORD in the value

			for(;;)
			{
				bReturn = (pIfd->uiFlags & IFD_EACHWORD)
								? (FLMBOOL) KYEachWordParse( &pText, &uiTextLen,
												pIfd->uiLimit,
												pTmpBuf, &uiWordLen)
								: (FLMBOOL) KYSubstringParse( &pText, &uiTextLen,
												pIfd->uiFlags, pIfd->uiLimit,
												pTmpBuf, &uiWordLen);
				if( !bReturn)
				{
					break;
				}

				uiTempLuLen = uiLuLen;

				// Compute number of bytes left

				uiElmLen = uiMaxKeySize - uiKeyLen - uiTempLuLen;
				if( RC_BAD( rc = KYCollateValue( &pKeyBuf [uiKeyLen], &uiElmLen,
											pTmpBuf, uiWordLen,
											pIfd->uiFlags, pIfd->uiLimit,
											NULL, &uiPieceLuLen, uiLanguage, TRUE,
											bFirstSubstring, FALSE, NULL)))
				{
					goto Exit;
				}

				bFirstSubstring = FALSE;
				if( uiPostFlag)
				{
					uiElmLen -= uiPieceLuLen;
					f_memcpy( &pLowUpBuf [uiTempLuLen], 
										&pKeyBuf[ uiKeyLen + uiElmLen ], uiPieceLuLen);
					uiTempLuLen += uiPieceLuLen;
				}

				if( !pNextIfdPiece)
				{

					// All ISKs have been added so now output the key

					if( uiTempLuLen )
					{
						uiPostLen = KYCombPostParts( pKeyBuf,
															(FLMUINT)(uiKeyLen + uiElmLen),
															pLowUpBuf, uiTempLuLen,
															uiLanguage,
															(FLMUINT)(pIfd->uiFlags) );
					}

					if( RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
											pIfd, uiAction, uiDrn, pbHadUniqueKeys,
											pKeyBuf,
											(FLMUINT)(uiKeyLen + uiElmLen + uiPostLen),
											TRUE, FALSE, FALSE)))
					{
						goto Cleanup1;
					}
				}
				else if( RC_BAD( rc))
				{
					goto Cleanup1;
				}
				else
				{

					// RECURSIVE CALL to the Next ISK provided no overflow

					if( RC_BAD( rc = KYCmpKeyElmBld( pDb, pIxd, uiContainerNum,
													pNextIfdPiece,
													uiAction, uiDrn, pbHadUniqueKeys,
													uiNextCdlEntry, 
													uiCompoundPos + 1, pKeyBuf,
													(FLMUINT)(uiKeyLen + uiElmLen), pLowUpBuf,
													uiTempLuLen, pRecord, pFldContext)))
					{
						goto Cleanup1;
					}
				}
			
				if( (pIfd->uiFlags & IFD_SUBSTRING) && 
					(uiTextLen == 1 && 
						!(uiLanguage >= FIRST_DBCS_LANG && 
							uiLanguage <= LAST_DBCS_LANG)))
				{
					break;
				}
			}

Cleanup1:

			if (RC_BAD( rc))
			{
				goto Exit;
			}
		}
		else							/* NOT an EACHWORD attribute */
		{
			if( pvField)				/* May not have any data at any level */
			{
				if( pIfd->uiFlags & IFD_CONTEXT)
				{
					pKeyBuf [uiKeyLen] = KY_CONTEXT_PREFIX;
					intToByte( (FLMUINT16)pRecord->getFieldID( pvField), &pKeyBuf [uiKeyLen + 1]);
		   		uiKeyLen += KY_CONTEXT_LEN;
				}
				else if( pRecord->getDataLength( pvField))
				{
					const FLMBYTE *	pExportValue = pRecord->getDataPtr( pvField);
					FLMUINT				uiDataLength = pRecord->getDataLength( pvField);
					
					if (pRecord->isEncryptedField( pvField) &&
						 pDb->pFile->bInLimitedMode)
					{
						pExportValue = pRecord->getEncryptionDataPtr( pvField);
						uiDataLength = pRecord->getEncryptedDataLength( pvField);
						bFldIsEncrypted = TRUE;
					}

       			/* Compute number of bytes left. */

					uiElmLen = uiMaxKeySize - uiKeyLen - uiLuLen;
					if( RC_BAD( rc = KYCollateValue( &pKeyBuf [uiKeyLen], &uiElmLen,
									pExportValue,
									uiDataLength, pIfd->uiFlags,
									pIfd->uiLimit, NULL, &uiPieceLuLen, 
									uiLanguage, TRUE, FALSE, FALSE, NULL, NULL,
									bFldIsEncrypted)))
					{
						goto Exit;
					}

					if( uiPostFlag )
					{
						uiElmLen -= uiPieceLuLen;
						f_memcpy( &pLowUpBuf [uiTempLuLen], 
											&pKeyBuf [uiKeyLen + uiElmLen], uiPieceLuLen);
						uiTempLuLen += uiPieceLuLen;
					}
				}
			}

			if( !pNextIfdPiece)
			{

				/* All IFDs have been added so now output the key. */

				if( uiTempLuLen)
				{
					uiPostLen = KYCombPostParts( pKeyBuf,
														(FLMUINT)(uiKeyLen + uiElmLen),
														pLowUpBuf, uiTempLuLen,
														uiLanguage, (FLMUINT)(pIfd->uiFlags));
				}

				if( RC_BAD( rc = KYAddToKrefTbl( pDb, pIxd, uiContainerNum,
											pIfd, uiAction, uiDrn, pbHadUniqueKeys,
											pKeyBuf,
											(FLMUINT)(uiKeyLen + uiElmLen + uiPostLen),
											TRUE, FALSE, bFldIsEncrypted)))
				{
					goto Exit;
				}
			}
			else if( RC_BAD( rc))
			{
				goto Exit;
			}
			else
			{

				/* RECURSIVE CALL to the Next ISK provided no overflow. */

				if( RC_BAD( rc = KYCmpKeyElmBld( pDb, pIxd, uiContainerNum,
												pNextIfdPiece,
												uiAction, uiDrn, pbHadUniqueKeys,
												uiNextCdlEntry, 
												uiCompoundPos + 1, pKeyBuf,
												(FLMUINT)(uiKeyLen + uiElmLen), pLowUpBuf,
												uiTempLuLen, pRecord, pFldContext)))
				{
					goto Exit;
				}
			}
		}
Next_CDL_Field:
		
		if( pCdl)
		{
			pCdl = pCdl->pNext;
		}

		// If the CDL list is empty, goto the next IFD if same uiCompoundPos.

		while ((!pCdl)
			&& ((pIfd->uiFlags & IFD_LAST) == 0)
			&&  (pIfd->uiCompoundPos == (pIfd+1)->uiCompoundPos))
		{
			pIfd++;
			pCdl = pCdlTbl [++uiCdlEntry];
		}
		
		/*
		Here is the tough part of the new compound indexing strategy (Aug98).
		If all fields failed the validate field path test and this piece of
		the compound key is required, then goto exit NOW which will not
		build any key with the previous built key pieces.
		*/

		if( !pCdl && !bBuiltKeyPiece && ((pIfd->uiFlags & IFD_OPTIONAL) == 0))
		{
			goto Exit;
		}
	}	// end while( pCdl || !bBuiltKeyPiece)

Exit:

	if( pvMark)
	{
		GedPoolReset( &pDb->TempPool, pvMark);
	}

	return( rc);
 }


/****************************************************************************
Desc:		Validate that the current field is related to the other fields
			in the compound key index.  The context (left-most) fields of the
			field paths must all be siblings of each other in order to 
			be related.
Notes:	If the GEDCOM implementation changes to where finding the level
			of a field is expensive, we need to set local level variables.
****************************************************************************/
RCODE KYValidatePathRelation(
	FlmRecord *		pRecord,
	void *			pCurContext,		// Current compound piece context.
	void *			pCurFld,				// Current compound field.
	FLD_CONTEXT *	pFldContext,		/* Points to field path state.
													->pParentAnchor is used as the parent
													of related siblings.  There can only
													be one parent anchor per compound
													set.  All remaining fields must
													be a child of this
													parent anchor.*/
	FLMUINT			uiCompoundPos)		// Compound piece position
{
	RCODE			rc = FERR_OK;
	void *		pCurParent;
	FLMUINT		uiPrevCompoundPos;
	FLMBOOL		bMatchedContext;

	// If too many compound level, just exit and don't check.
	if( uiCompoundPos >= MAX_COMPOUND_PIECES)
	{
		goto Exit;
	}

	pCurParent = pRecord->parent( pCurContext);

	// First time in is the easy case - just set the parent anchor.
	// A value of NULL is OK.

	if( uiCompoundPos == 0)
	{
		pFldContext->pParentAnchor = pCurParent;
		goto Exit;
	}

	bMatchedContext = FALSE;
	// uiCompoundPos used at exit to save state.
	uiPrevCompoundPos = uiCompoundPos;
	while( uiPrevCompoundPos--)
	{
		if( pFldContext->rootContexts[ uiPrevCompoundPos] == pCurContext)
		{
			// Check this field against the current field values.

			rc = KYVerifyMatchingPaths( pRecord, pCurContext, pCurFld, 
						pFldContext->leafFlds[ uiPrevCompoundPos]);

			// Return failure on any failure.  Otherwise continue.
			if( rc == FERR_FAILURE)
			{
				goto Exit;
			}
			bMatchedContext = TRUE;
		}
	}
	if( bMatchedContext)
	{
		/*
		If we had some base relation match, there is no need to 
		verify that the parents are the same.
		*/
		goto Exit;
	}

	// Verify that the parent anchor equals the parent of pCurContext.
	// GedParent() supports passing a NULL value.

	if( pFldContext->pParentAnchor != pCurParent)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

Exit:
	// Set the state variables for this compound position.
	if( RC_OK(rc))
	{
		pFldContext->rootContexts[ uiCompoundPos ] = pCurContext;
		pFldContext->leafFlds[ uiCompoundPos] = pCurFld;
	}
	return( rc);
}

/****************************************************************************
Desc:		Verify that two paths with a common context match paths.
			If the tag of pCurContext has a previous match in the compound
			key, the field should also match (more of a relational validation).
		This means that for keys (A.B.C.D  AND A.B.C.E) the 'A.B.C' fields should 
		be the same field.  ALL previous field pieces must be checked for this.
		This could be (but isn't being) done by finding the best match" 
		and only comparing the current with the best match.

	Hard Example:
		Do these fields match - A.B.D.E.F and A.C.D.E.G?
		We don't want to keep the field path of the two fields around because
		this is more state than we need right now.  These match only if the
		'A's are the same field.

			A						A
				B						C
					D						D
						E						E
							F						G
Ret:		FERR_OK or FERR_FAILURE
****************************************************************************/
RCODE KYVerifyMatchingPaths(
	FlmRecord *	pRecord,
	void *		pCurContext,			// Same value as pMatchFld's context.
	void *		pCurFld,					// Current field
	void *		pMatchFld)				// Some field from a previous piece.
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCurLevel;
	FLMUINT		uiMatchLevel;
	FLMBOOL		bMismatchFound = FALSE;

	// If a field equals a context then don't bother to check.

	if( (pCurContext == pCurFld) || (pCurContext == pMatchFld))
	{
		goto Exit;
	}

	// Go up the parent line until levels match.
	
	uiCurLevel = pRecord->getLevel( pCurFld);
	uiMatchLevel = pRecord->getLevel( pMatchFld);
	flmAssert( pRecord->getLevel( pCurContext) < uiCurLevel);

	while( uiCurLevel != uiMatchLevel)
	{
		if( uiCurLevel > uiMatchLevel)
		{
			pCurFld = pRecord->parent( pCurFld);
			uiCurLevel--;
		}
		else
		{
			pMatchFld = pRecord->parent( pMatchFld);
			uiMatchLevel--;
		}
	}
	// Go up until you hit the matching context.

	while( pCurFld != pCurContext)
	{
		if( pRecord->getFieldID( pCurFld) == pRecord->getFieldID( pMatchFld))
		{
			// If the fields are NOT the same we MAY have a mismatch.
			if( pCurFld != pMatchFld)
			{
				bMismatchFound = TRUE;
			}
		}
		else
		{
			// Tags are different - start over checking
			bMismatchFound = FALSE;
		}
		// Go to the next parent.
		pCurFld = pRecord->parent( pCurFld);
		pMatchFld = pRecord->parent( pMatchFld);
	}
	if( bMismatchFound)
	{
		rc = RC_SET( FERR_FAILURE);
	}

Exit:
	return( rc);
}


/****************************************************************************
Desc:		Combine the bits from all POST text keys.
****************************************************************************/
FLMUINT KYCombPostParts(
	FLMBYTE *	pKeyBuf,
	FLMUINT	  	uiKeyLen,
	FLMBYTE *	pLowUpBuf,
	FLMUINT	  	uiLuLen,
	FLMUINT		uiLanguage,
	FLMUINT		uiIfdAttr
	)
{
	FLMUINT		wReturnLen;

	if( !uiLuLen)
		return( 0);	 				/* Don't add any more if no pLowUpBuf[] */

	wReturnLen = (FLMUINT)(uiLuLen + 2);
	if( (uiLanguage >= FIRST_DBCS_LANG) && 
		 (uiLanguage <= LAST_DBCS_LANG) &&
		 ((uiIfdAttr & 0x0F) == FLM_TEXT_TYPE) &&
		 (!(uiIfdAttr & IFD_CONTEXT )))
	{
		pKeyBuf [uiKeyLen++] = 0;				/* Add two bytes */
		wReturnLen++;
	}
	pKeyBuf [uiKeyLen++] = END_COMPOUND_MARKER;
	
	f_memcpy( &pKeyBuf [uiKeyLen], pLowUpBuf, uiLuLen );	/* Move pLowUpBuf[] */
	pKeyBuf [uiKeyLen + uiLuLen] = (FLMBYTE) uiLuLen;			/* Last byte is uiLuLen */
	
	return( wReturnLen );
}
