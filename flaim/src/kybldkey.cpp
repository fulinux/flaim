//-------------------------------------------------------------------------
// Desc:	Build keys for searching in a query.
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
// $Id: kybldkey.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMBOOL flmFindWildcard(
	FLMBYTE *	pValue,
	FLMUINT *	puiCharPos);

FSTATIC RCODE flmAddKeyPiece(
	FLMUINT		uiMaxKeySize,
	IFD *			pIfd,
	FLMBOOL		bDoMatchBegin,
	FLMBYTE *	pFromKey,
	FLMUINT *	puiFromKeyPos,
	FLMBOOL		bFromAtFirst,
	FLMBYTE *	pUntilKey,
	FLMUINT *	puiUntilKeyPos,
	FLMBOOL		bUntilAtEnd,
	FLMBYTE *	pBuf,
	FLMUINT		uiBufLen,
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbDoneBuilding);

FSTATIC RCODE flmAddTextPiece(
	FLMUINT		uiMaxKeySize,
	IFD *			pIfd,
	FLMBOOL		bCaseInsensitive,
	FLMBOOL		bDoMatchBegin,
	FLMBOOL		bDoFirstSubstring,
	FLMBOOL		bTrailingWildcard,
	FLMBYTE *	pFromKey,
	FLMUINT *	puiFromKeyPos,
	FLMBOOL		bFromAtFirst,
	FLMBYTE *	pUntilKey,
	FLMUINT *	puiUntilKeyPos,
	FLMBOOL		bUntilAtEnd,
	FLMBYTE *	pBuf,
	FLMUINT		uiBufLen,
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbDoneBuilding,
	FLMBOOL *	pbOriginalCharsLost);

FSTATIC FLMBOOL flmSelectBestSubstr(
	FLMBYTE **		ppValue,
	FLMUINT *		puiValueLen,
	FLMUINT			uiIfdFlags,
	FLMBOOL *		pbTrailingWildcard);

FSTATIC FLMUINT flmCountCharacters(
	FLMBYTE * 		pValue,
	FLMUINT			uiValueLen,
	FLMUINT			uiMaxToCount,			
	FLMUINT			uiIfdFlags);

FSTATIC void flmUintToBCD( 
	FLMUINT			uiValue, 
	FLMBYTE *		pNumberBuf, 
	FLMUINT *		puiValueLen);

FSTATIC void flmIntToBCD( 
	FLMINT			iValue, 
	FLMBYTE *		pNumberBuf, 
	FLMUINT *		puiValueLen);

/****************************************************************************
Desc:		Build the from and until keys given a field list with operators and
			values and an index.
Notes:	The knowledge of query definitions is limited in these routines.
****************************************************************************/
RCODE flmBuildFromAndUntilKeys(
	IXD_p				pIxd,
	QPREDICATE **	ppQPredicate,		// List of field predicates that are parallel
												// with the IFD list for the index.  
												// Same number of elements as the IFD list.
	FLMBYTE *	pFromKey,				// From key to build
	FLMUINT *	puiFromKeyLen,			// return fromKey length
	FLMBYTE *	pUntilKey,				// Until key to build
	FLMUINT *	puiUntilKeyLen,		// return untilKey length.
	FLMBOOL *	pbDoRecMatch,			// [out] Leave alone or set to TRUE.
	FLMBOOL *	pbDoKeyMatch,			// [out] Default = TRUE, Change when needed.
	FLMBOOL *	pbExclusiveUntilKey)	// [out] Leave alone or set to TRUE.
{
	RCODE       rc = FERR_OK;
	QPREDICATE *pCurPred;
	IFD *			pIfd = pIxd->pFirstIfd;
	FLMUINT		uiLanguage = pIxd->uiLanguage;
	FLMUINT		uiIfdCnt = pIxd->uiNumFlds;
	FLMUINT		uiFromKeyPos = 0;
	FLMUINT		uiUntilKeyPos = 0;
	FLMBOOL		bFromAtFirst;
	FLMBOOL		bUntilAtEnd;
	FLMBOOL		bDataTruncated;
	FLMBOOL		bDoneBuilding;
	FLMBOOL		bMustNotDoKeyMatch = FALSE;
	FLMBOOL		bDoKeyMatch = FALSE;
	FLMBOOL		bOriginalCharsLost;
	FLMBOOL		bDBCSLanguage = (uiLanguage >= FIRST_DBCS_LANG) && 
										 (uiLanguage <= LAST_DBCS_LANG) ? TRUE : FALSE;
	FLMBYTE		pNumberBuf[8];
	FLMUINT		uiTempLen;
	FLMUINT		uiMaxKeySize = (pIxd->uiContainerNum)
										? MAX_KEY_SIZ
										: MAX_KEY_SIZ - getIxContainerPartLen( pIxd);

	bDataTruncated = bDoneBuilding = FALSE;
	*puiFromKeyLen = *puiUntilKeyLen = 0;
	uiFromKeyPos = uiUntilKeyPos = 0;
	*pbExclusiveUntilKey = TRUE;		// Will almost always build an exclusive key.
	
	for( ; !bDoneBuilding && uiIfdCnt--; ppQPredicate++, pIfd++)
	{
		// Add the compound marker if not the first piece.

		if( pIfd->uiCompoundPos)
		{
			IFD *		pPrevIfd = (pIfd - 1);

			// Add the compound markers for this key piece.

			if( bDBCSLanguage
			 && (IFD_GET_FIELD_TYPE( pPrevIfd) == FLM_TEXT_TYPE)
			 && (!((pPrevIfd)->uiFlags & IFD_CONTEXT)))
			{
				pFromKey[ uiFromKeyPos++] = 0;
				pUntilKey[ uiUntilKeyPos++] = 0;
			}
			pFromKey [uiFromKeyPos++] = COMPOUND_MARKER;
			pUntilKey [uiUntilKeyPos++] = COMPOUND_MARKER;
		}

		bFromAtFirst = bUntilAtEnd = FALSE;
		pCurPred = *ppQPredicate;

		if( !pCurPred)
		{
			/*
			There is not a predicate that matches this compound key piece.  
			Done processing, yet may need to look for a predicate
			that will force a doKeyMatch or a doRecMatch.
			*/

			if( RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize,
						pIfd, FALSE, pFromKey, &uiFromKeyPos, TRUE,
						pUntilKey, &uiUntilKeyPos, TRUE, NULL, 0,
						&bDataTruncated, &bDoneBuilding)))
			{
				goto Exit;
			}
			continue;
		}

		// Handle special cases for indexing context and/or exists predicate.

		else if( pIfd->uiFlags & IFD_CONTEXT)
		{
			// Indexed only the TAG.  Simple to set the tag as the key.

			if( RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE,
						pFromKey, &uiFromKeyPos, FALSE,
						pUntilKey, &uiUntilKeyPos, FALSE, NULL, 0,
						&bDataTruncated, &bDoneBuilding)))
			{
				goto Exit;
			}

			// If we don't have an exists predicate we need to read the record.
			if( pCurPred->eOperator != FLM_EXISTS_OP)
			{
				bMustNotDoKeyMatch = TRUE;
			}
			continue;
		}
		else
		{
			FLMBOOL		bMatchedBadOperator = FALSE;
			switch( pCurPred->eOperator)
			{
				case FLM_EXISTS_OP:
				case FLM_NE_OP:
					bMatchedBadOperator = TRUE;
					bUntilAtEnd = TRUE;
					bFromAtFirst = TRUE;
					break;
				default:
					if( pCurPred->bNotted)
					{
						bMatchedBadOperator = TRUE;
						bUntilAtEnd = TRUE;
						bFromAtFirst = TRUE;
					}
					break;
			}

			if( bMatchedBadOperator)
			{
				// Does exist is a FIRST to LAST for this piece.

				if( RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE,
							pFromKey, &uiFromKeyPos, bFromAtFirst,
							pUntilKey, &uiUntilKeyPos, bUntilAtEnd, NULL, 0,
							&bDataTruncated, &bDoneBuilding)))
				{
					goto Exit;
				}
				continue;
			}
		}

		switch( IFD_GET_FIELD_TYPE( pIfd))
		{
			/*
			Build TEXT type piece
			*/
			case FLM_TEXT_TYPE:
			{
				FLMBOOL		bCaseInsensitive = (FLMBOOL) 
					((pCurPred->pVal->uiFlags & FLM_NOCASE) ? TRUE : FALSE);
				FLMBOOL		bDoFirstSubstring = (FLMBOOL)
					((pIfd->uiFlags & IFD_SUBSTRING) ? TRUE : FALSE);

				// True bDoMatchBegin generates high values in UNTIL key.
				FLMBOOL		bDoMatchBegin;
				FLMBOOL		bDoSubstringSearch;
				FLMBOOL		bTrailingWildcard;		// If trailing spaces remain
				FLMBYTE *	pValue = (FLMBYTE *) pCurPred->pVal->val.pucBuf;
				FLMUINT		uiValueLen = pCurPred->pVal->uiBufLen;

				bDoMatchBegin = bDoSubstringSearch = bTrailingWildcard = FALSE;
				switch( pCurPred->eOperator)
				{
					// The difference between MATCH and EQ_OP is that EQ does
					// not support wildcards inbedded in the search key.

					case FLM_MATCH_OP:
					case FLM_MATCH_BEGIN_OP:

						if( pCurPred->eOperator == FLM_MATCH_BEGIN_OP)
						{
							bDoKeyMatch = bDoMatchBegin = TRUE;
						}
						if( pCurPred->pVal->uiFlags & FLM_WILD)
						{
							if( !bDoFirstSubstring)
							{
								FLMBOOL		bFoundWildcard = 
													flmFindWildcard( pValue, &uiValueLen);

								bDoKeyMatch = TRUE;

								if( pCurPred->eOperator == FLM_MATCH_OP)
								{
									bTrailingWildcard = bDoMatchBegin = bFoundWildcard;
								}
								else
								{
									bTrailingWildcard = bDoMatchBegin = TRUE;
								}
							}
							else
							{	
								// If this is a substring index look for a 
								// better 'contains' string to search for. 
								// We don't like "A*BCDEFG" searches.
								
								bTrailingWildcard = 
									(pCurPred->eOperator == FLM_MATCH_BEGIN_OP) 
										? TRUE 
										: FALSE;

								if( flmSelectBestSubstr( &pValue, &uiValueLen, 
																pIfd->uiFlags, &bTrailingWildcard))
								{
									bDoMatchBegin = bTrailingWildcard;
									bMustNotDoKeyMatch = TRUE;
									bDoFirstSubstring = FALSE;
								}
								else if( bTrailingWildcard)
								{
									bDoKeyMatch = bDoMatchBegin = TRUE;
								}
							}
						}
						break;

					case FLM_CONTAINS_OP:
					case FLM_MATCH_END_OP:

						// Normal text index this piece goes from first to last.
						if( !bDoFirstSubstring) //!(pIfd->uiFlags & IFD_SUBSTRING))
						{
							bFromAtFirst = TRUE;
							bUntilAtEnd = TRUE;
						}
						else
						{
							bDoFirstSubstring = TRUE;
							bDoSubstringSearch = TRUE;

							// SPACE/Hyphen rules on SUBSTRING index.
							// If the search string starts with " _asdf" then we must do
							// a record match so "Z asdf" matches and "Zasdf" doesn't.
							// We won't touch key match even though it MAY return
							// FLM_TRUE when in fact the key may or may not match.
							// VISIT: MatchBegin and Contains could also optimize the
							// trailing space by adding the space ONLY to the UNTIL key.

							if( uiValueLen &&
								(	(*pValue == ASCII_SPACE 
									&& (pIfd->uiFlags & IFD_MIN_SPACES))
								|| (*pValue == ASCII_UNDERSCORE 
									&& (pIfd->uiFlags & IFD_NO_UNDERSCORE))))
							{
								*pbDoRecMatch = TRUE;
							}						
							
							// Take the flags from the pVal and NOT from the predicate. 
							if( pCurPred->pVal->uiFlags & FLM_WILD)
							{
								/*
								Select the best substring.  The case of 
								"A*BCD*E*FGHIJKLMNOP" will look for "FGHIJKLMNOP".
								and TURN OFF doKeyMatch and SET doRecMatch.
								*/

								bTrailingWildcard = 
									(pCurPred->eOperator == FLM_CONTAINS_OP)
										? TRUE 
										: FALSE;

								if( flmSelectBestSubstr( &pValue, &uiValueLen, 
															pIfd->uiFlags, &bTrailingWildcard))
								{
									bDoMatchBegin = bTrailingWildcard;
									bMustNotDoKeyMatch = TRUE;
									bDoFirstSubstring = FALSE;
								}
								if( bTrailingWildcard)
								{
									bDoKeyMatch = bDoMatchBegin = TRUE;
								}
							}
							if( bDoFirstSubstring)
							{
								// Setting bDoMatchBegin creates a UNTIL key 
								// with trailing 0xFF values.
								if( pCurPred->eOperator == FLM_CONTAINS_OP)
								{
									bDoKeyMatch = TRUE;	// Because of subcollation values
									bDoMatchBegin = TRUE;// Sets high values in UNTIL key.	
								}
							}
							
							// Special case: Single character contains/MEnd in a substr ix.
							if( !bDBCSLanguage && flmCountCharacters( pValue, uiValueLen, 
								2, pIfd->uiFlags) < 2)
							{
								bDoKeyMatch = bFromAtFirst = bUntilAtEnd = TRUE;
							}
						}
						break;

					// No wild card support for the operators below.
					case FLM_EQ_OP:
						break;
					case FLM_GE_OP:
					case FLM_GT_OP:
						bUntilAtEnd = TRUE;
						break;
					case FLM_LE_OP:
						bFromAtFirst = TRUE;
						break;
					case FLM_LT_OP:
						bFromAtFirst = TRUE;
						*pbExclusiveUntilKey = TRUE;
						break;
					default:
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
				}

				// If index is case insensitive, but search is case sensitive
				// we must NOT do a key match - we would fail things we should
				// not be failing.
				
				if ((pIfd->uiFlags & IFD_UPPER) && !bCaseInsensitive)
				{
					bMustNotDoKeyMatch = TRUE;
				}
				if( RC_BAD( rc = flmAddTextPiece( uiMaxKeySize,
							pIfd, bCaseInsensitive, bDoMatchBegin, 
							bDoFirstSubstring, bTrailingWildcard,
							pFromKey, &uiFromKeyPos, bFromAtFirst,
							pUntilKey, &uiUntilKeyPos, bUntilAtEnd,
							pValue, uiValueLen,
							/*pbExclusiveUntilKey, */
							&bDataTruncated, &bDoneBuilding, &bOriginalCharsLost)))
				{
					goto Exit;
				}
				if (bOriginalCharsLost)
				{
					bMustNotDoKeyMatch = TRUE;
				}
				break;
			}
			// Build NUMBER or CONTEXT type piece
			// VISIT: Add a true number type so we don't have to build a NODE.

			case FLM_NUMBER_TYPE:
			case FLM_CONTEXT_TYPE:
				switch( pCurPred->pVal->eType)
				{
					case FLM_INT32_VAL:
					{
						FLMINT	iValue = pCurPred->pVal->val.iVal;
						if( pCurPred->eOperator == FLM_GT_OP)
						{
							iValue++;
						}
						if( IFD_GET_FIELD_TYPE( pIfd) == FLM_NUMBER_TYPE)
						{
							flmIntToBCD( iValue, pNumberBuf, &uiTempLen);
						}
						else
						{
							UD2FBA( iValue, pNumberBuf);
							uiTempLen = 4;
						}
						break;
					}
					case FLM_UINT32_VAL:
					case FLM_REC_PTR_VAL:
					{
						FLMUINT	uiValue = pCurPred->pVal->val.uiVal;
						if( pCurPred->eOperator == FLM_GT_OP)
						{
							uiValue++;
						}
						if( IFD_GET_FIELD_TYPE( pIfd) == FLM_NUMBER_TYPE)
						{
							flmUintToBCD( uiValue, pNumberBuf, &uiTempLen);
						}
						else
						{
							UD2FBA( uiValue, pNumberBuf);
							uiTempLen = 4;
						}
						break;
					}
					default:
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
				}
				switch( pCurPred->eOperator)
				{
					case FLM_EQ_OP:
						break;
					case FLM_GE_OP:
					case FLM_GT_OP:
						bUntilAtEnd = TRUE;
						break;
					case FLM_LE_OP:
						bFromAtFirst = TRUE;
						break;
					case FLM_LT_OP:
						bFromAtFirst = TRUE;
						*pbExclusiveUntilKey = TRUE;
						break;
					default:
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
				}

				if( RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, FALSE,
							pFromKey, &uiFromKeyPos, bFromAtFirst,
							pUntilKey, &uiUntilKeyPos, bUntilAtEnd,
							(FLMBYTE *) pNumberBuf, uiTempLen,
							&bDataTruncated, &bDoneBuilding)))
				{
					goto Exit;
				}

				break;

			/*
			Build BINARY type piece
			*/

			case FLM_BINARY_TYPE:
			{
				FLMBOOL	bMatchBegin = FALSE;

				switch( pCurPred->eOperator)
				{
					case FLM_MATCH_BEGIN_OP:
						bMatchBegin = TRUE;
						break;
					case FLM_EQ_OP:
						break;
					case FLM_GE_OP:
						bUntilAtEnd = TRUE;
						break;
					case FLM_GT_OP:
						bUntilAtEnd = TRUE;
						bDoKeyMatch = TRUE;
						break;
					case FLM_LE_OP:
						bFromAtFirst = TRUE;
						break;
					case FLM_LT_OP:
						bFromAtFirst = TRUE;
						*pbExclusiveUntilKey = TRUE;
						break;
					default:
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
				}
				if( RC_BAD( rc = flmAddKeyPiece( uiMaxKeySize, pIfd, bMatchBegin,
							pFromKey, &uiFromKeyPos, bFromAtFirst,
							pUntilKey, &uiUntilKeyPos, bUntilAtEnd,
							pCurPred->pVal->val.pucBuf, pCurPred->pVal->uiBufLen,
							&bDataTruncated, &bDoneBuilding)))
				{
					goto Exit;
				}

				break;
			}

			default:
				flmAssert(0);		// Unsupported
				break;
		}
		if( bDataTruncated)
		{
			bMustNotDoKeyMatch = TRUE;
		}
	}

	// Really rare case where FROM/UNTIL keys are exactly the same.
	if( !bDoneBuilding && (uiIfdCnt + 1 == 0) &&
		 uiUntilKeyPos < uiMaxKeySize - 2)
	{
		// Always make the until key exclusive.
		// *pbExclusiveUntilKey = FALSE;
		pUntilKey[ uiUntilKeyPos++ ] = 0xFF;
		pUntilKey[ uiUntilKeyPos++ ] = 0xFF;
	}


Exit:
	if( bMustNotDoKeyMatch)
	{
		*pbDoKeyMatch = FALSE;
		*pbDoRecMatch = TRUE;
	}
	else if( bDoKeyMatch || !pIxd->uiContainerNum)
	{
		*pbDoKeyMatch = TRUE;
	}
	// Special case for building FIRST/LAST keys.
	if( !uiFromKeyPos)
	{
		*pFromKey = '\0';
		uiFromKeyPos = 1;
	}
	if( !uiUntilKeyPos)
	{
		f_memset( pUntilKey, 0xFF, uiMaxKeySize - 2);
		uiUntilKeyPos = uiMaxKeySize - 2;
	}
	*puiFromKeyLen  = uiFromKeyPos;
	*puiUntilKeyLen = uiUntilKeyPos;
	return( rc);
}

/****************************************************************************
Desc:		Truncate the length of the text buffer on the first wild card.
****************************************************************************/
FSTATIC FLMBOOL flmFindWildcard(
	FLMBYTE *	pVal,
	FLMUINT *	puiCharPos)			// [in/out] always returns charPos of * or length
{
	FLMBOOL		bHaveChar = FALSE;
	FLMBYTE *	pSaveVal = pVal;
	FLMUINT 		uiObjLength;
	FLMUINT		uiLen = *puiCharPos;

	for( ; 
		  *pVal; 
		  pVal += uiObjLength, uiLen = (uiObjLength < uiLen) ? uiLen - uiObjLength : 0)
	{
		switch( (FLMUINT)(GedTextObjType( *pVal)))
		{
			case ASCII_CHAR_CODE:  		// 0nnnnnnn
				if( *pVal == ASCII_WILDCARD)
				{
					bHaveChar = TRUE;
					goto Exit;
				}
				uiObjLength = 1;

				// Check for '*' or '\\' after an escape character.
				if( *pVal == ASCII_BACKSLASH && 
					(*(pVal + 1) == ASCII_WILDCARD 
					|| *(pVal + 1) == ASCII_BACKSLASH))
				{
					uiObjLength++;
				}
				break;

			case WHITE_SPACE_CODE:		// 110nnnnn
				uiObjLength = 1;
				break;
			case CHAR_SET_CODE:	  		// 10nnnnnn
			case UNK_EQ_1_CODE:
			case OEM_CODE:
				uiObjLength = 2;
				break;
			case UNICODE_CODE:			// Unconvertable UNICODE code
			case EXT_CHAR_CODE:
				uiObjLength = 3;
				break;
			case UNK_GT_255_CODE:
				uiObjLength = 1 + sizeof( FLMUINT16) + FB2UW( pVal + 1);
				break;
			case UNK_LE_255_CODE:
				uiObjLength = 2 + (FLMUINT16)*(pVal + 1);
				break;
			default:							// should NEVER happen: bug if does
				uiObjLength = 1;
				break;						// Should not really return an error
		}
	}
Exit:
	*puiCharPos = (FLMUINT)(pVal - pSaveVal);
	return bHaveChar;
}

/****************************************************************************
Desc:		Add a key piece to the from and until key.  Text fields are not 
			handled in this routine because of their complexity.
Notes:	The goal of this code is to build a the collated compound piece
			for the 'from' and 'until' key only once instead of twice.
****************************************************************************/
FSTATIC RCODE flmAddKeyPiece(
	FLMUINT		uiMaxKeySize,
	IFD *			pIfd,
	FLMBOOL		bDoMatchBegin,
	FLMBYTE *	pFromKey,
	FLMUINT *	puiFromKeyPos,
	FLMBOOL		bFromAtFirst,
	FLMBYTE *	pUntilKey,
	FLMUINT *	puiUntilKeyPos,
	FLMBOOL		bUntilAtEnd,
	FLMBYTE *	pBuf,
	FLMUINT		uiBufLen,
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbDoneBuilding)
{
	RCODE       rc = FERR_OK;
	FLMUINT		uiFromKeyPos = *puiFromKeyPos;
	FLMUINT		uiUntilKeyPos = *puiUntilKeyPos;
	FLMBYTE *	pDestKey;
	FLMUINT		uiDestKeyLen;

	if( pIfd->uiCompoundPos == 0 && bFromAtFirst && bUntilAtEnd)
	{
		// Special case for the first piece - FIRST to LAST - zero length keys.
		// so that the caller can get the number of references for the entire index.
		// VISIT: May want to set the from key to have 1 byte and set high values
		//	for the until key.  This way the caller never checks this special case.

		*pbDoneBuilding = TRUE;
		goto Exit;
	}

	// Handle the CONTEXT exception here - this is not done in kyCollate.

	if( pIfd->uiFlags & IFD_CONTEXT)
	{
		pFromKey [uiFromKeyPos] = KY_CONTEXT_PREFIX;
		flmUINT16ToBigEndian( (FLMUINT16) pIfd->uiFldNum, &pFromKey [uiFromKeyPos + 1]);
		uiFromKeyPos += KY_CONTEXT_LEN;

		if( uiUntilKeyPos + KY_CONTEXT_LEN < uiMaxKeySize)
		{
			pUntilKey [uiUntilKeyPos] = KY_CONTEXT_PREFIX;
			flmUINT16ToBigEndian( (FLMUINT16) pIfd->uiFldNum, &pUntilKey [uiUntilKeyPos + 1]);
			uiUntilKeyPos += KY_CONTEXT_LEN;
		}
		goto Exit;
	}

	if( bFromAtFirst)
	{
		if( bUntilAtEnd)
		{
			// Not the first piece and need to go from first to last.
			*pbDoneBuilding = TRUE;

			if( uiUntilKeyPos < uiMaxKeySize - 2)
			{
				if( uiUntilKeyPos > 0)
				{
					// Instead of filling the key with 0xFF, increment the marker.
					pUntilKey [uiUntilKeyPos - 1]++;
				}
				else
				{
					f_memset( pUntilKey, 0xFF, uiMaxKeySize - 2);
					uiUntilKeyPos = uiMaxKeySize - 2;
				}
			}
			goto Exit;
		}

		if( uiUntilKeyPos >= uiMaxKeySize - 2)
		{
			goto Exit;
		}
		// Have a LAST key but no FROM key.
		pDestKey = pUntilKey + uiUntilKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiUntilKeyPos;
	}
	else
	{
		pDestKey = pFromKey + uiFromKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiFromKeyPos;
	}
		
	rc = KYCollateValue( pDestKey, &uiDestKeyLen, 
								(FLMBYTE *)pBuf, uiBufLen,
								pIfd->uiFlags, pIfd->uiLimit, NULL, NULL, 
								0, TRUE, FALSE, FALSE, pbDataTruncated);

	if( rc == FERR_CONV_DEST_OVERFLOW)
	{
		rc = FERR_OK;
	}
	else if( RC_BAD( rc))
	{
		goto Exit;
	}

	// If we just built the FROM key, we may want to copy to the UNTIL key.
	if( pDestKey == pFromKey + uiFromKeyPos)
	{
		uiFromKeyPos += uiDestKeyLen;

		// Unless the UNTIL key is full, the length is at or less than FROM key.
		if( !bUntilAtEnd)
		{
			if( uiUntilKeyPos  + uiDestKeyLen <= uiMaxKeySize)
			{
				f_memcpy( &pUntilKey[ uiUntilKeyPos], pDestKey, uiDestKeyLen);
				uiUntilKeyPos += uiDestKeyLen;
			}

			if( bDoMatchBegin)
			{
				flmAssert( IFD_GET_FIELD_TYPE( pIfd) == FLM_BINARY_TYPE);

				if( uiUntilKeyPos < MAX_KEY_SIZ - 2)
				{
				// Optimization - only need to set a single byte to 0xFF.
				// We can do this because this routine does not deal with text key
				// pieces and binary, number and context will never have 0xFF bytes.

					pUntilKey[ uiUntilKeyPos++] = 0xFF;
				}
				
				// We don't need to set *pbDoneBuilding = TRUE, because we may
				// be able to continue building the from key
			}
		}
		else
		{
			if( uiUntilKeyPos > 0)
			{
				// Instead of filling the key with 0xFF, increment the marker.
				pUntilKey [uiUntilKeyPos - 1]++;
			}
			else
			{
				// Optimization - only need to set a single byte to 0xFF.
				// We can do this because this routine does not deal with text key
				// pieces and binary, number and context will never have 0xFF bytes.

				flmAssert( IFD_GET_FIELD_TYPE( pIfd) != FLM_TEXT_TYPE);

				*pUntilKey = 0xFF;
				uiUntilKeyPos++;
			}
		}
	}
	else
	{
		uiUntilKeyPos += uiDestKeyLen;
	}

Exit:
	// Set the FROM and UNTIL key length return values.
	*puiFromKeyPos = uiFromKeyPos;
	*puiUntilKeyPos = uiUntilKeyPos;
	return( rc);
}

/****************************************************************************
Desc:		Add a text piece to the from and until key.  Some of the code is
			the same with AddKeyPiece above.
Notes:	The goal of this code is to build a the collated compound piece
			for the 'from' and 'until' key only once instead of twice.
****************************************************************************/
FSTATIC RCODE flmAddTextPiece(
	FLMUINT		uiMaxKeySize,
	IFD *			pIfd,
	FLMBOOL		bCaseInsensitive,
	FLMBOOL		bDoMatchBegin,
	FLMBOOL		bDoFirstSubstring,
	FLMBOOL		bTrailingWildcard,
	FLMBYTE *	pFromKey,
	FLMUINT *	puiFromKeyPos,
	FLMBOOL		bFromAtFirst,
	FLMBYTE *	pUntilKey,
	FLMUINT *	puiUntilKeyPos,
	FLMBOOL		bUntilAtEnd,
	FLMBYTE *	pBuf,
	FLMUINT		uiBufLen,
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbDoneBuilding,
	FLMBOOL *	pbOriginalCharsLost
	)
{
	RCODE       rc = FERR_OK;
	FLMUINT		uiFromKeyPos = *puiFromKeyPos;
	FLMUINT		uiUntilKeyPos = *puiUntilKeyPos;
	FLMUINT		uiLanguage = pIfd->pIxd->uiLanguage;
	FLMBYTE *	pDestKey;
	FLMUINT		uiDestKeyLen;
	FLMUINT		uiCollationLen = 0;
	FLMUINT		uiCaseLen;
	FLMBOOL		bIsDBCS = (uiLanguage >= FIRST_DBCS_LANG &&
								  uiLanguage <= LAST_DBCS_LANG)
								  ? TRUE
								  : FALSE;

	*pbOriginalCharsLost = FALSE;
	if( pIfd->uiCompoundPos == 0 && bFromAtFirst && bUntilAtEnd)
	{
		// Special case for the first piece - FIRST to LAST - zero length keys.
		// so that the caller can get the number of references for the entire index.
		// VISIT: May want to set the from key to have 1 byte and set high values
		//	for the until key.  This way the caller never checks this special case.

		*pbDoneBuilding = TRUE;
		goto Exit;
	}
	if( bFromAtFirst)
	{
		if( bUntilAtEnd)
		{
			// Not the first piece and need to go from first to last.
			*pbDoneBuilding = TRUE;

			if( uiUntilKeyPos < uiMaxKeySize - 2)
			{
				// Instead of filling the key with 0xFF, increment the marker.
				pUntilKey [uiUntilKeyPos - 1]++;
			}
			goto Exit;
		}
		if( uiUntilKeyPos >= uiMaxKeySize - 2)
		{
			goto Exit;
		}
		// Have a LAST key but no FROM key.
		pDestKey = pUntilKey + uiUntilKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiUntilKeyPos;
	}
	else		// Handle below if UNTIL key is LAST.
	{
		pDestKey = pFromKey + uiFromKeyPos;
		uiDestKeyLen = uiMaxKeySize - uiFromKeyPos;
	}
	// Add IFD_ESC_CHAR to the ifd flags because
	// the search string must have BACKSLASHES and '*' escaped.

	rc = KYCollateValue( pDestKey, &uiDestKeyLen, 
						(FLMBYTE *)pBuf, uiBufLen,
						pIfd->uiFlags | IFD_ESC_CHAR, pIfd->uiLimit,
						&uiCollationLen, &uiCaseLen,
						uiLanguage, TRUE, bDoFirstSubstring, 
						bTrailingWildcard, pbDataTruncated,
						pbOriginalCharsLost);
	if( rc == FERR_CONV_DEST_OVERFLOW)
		rc = FERR_OK;
	else if( RC_BAD( rc))
		goto Exit;

	if( pIfd->uiFlags & IFD_POST)
	{
		uiDestKeyLen -= uiCaseLen;
	}
	else
	{
		// Special case: The index is NOT an upper index and the search is
		// case-insensitive.
		// The FROM key must have lower case values and the UNTIL must be the
		// upper case values. This will be true for Asian indexes also.
		
		if (uiDestKeyLen &&
			 (bIsDBCS ||
			  (!(pIfd->uiFlags & IFD_UPPER) && bCaseInsensitive)))
		{

			// Subtract off all but the case marker.
			// Remember that for DBCS (Asian) the case marker is two bytes.

			uiDestKeyLen -= (uiCaseLen -
									((FLMUINT)(bIsDBCS
												  ? (FLMUINT)2
												  : (FLMUINT)1)));

			// NOTE: SC_LOWER is only used in GREEK indexes, which is why
			// we use it here instead of SC_MIXED.

			pDestKey[ uiDestKeyLen - 1] = (FLMBYTE)
				(( uiLanguage != (FLMUINT) GR_LANG) ? 
						COLL_MARKER | SC_MIXED : COLL_MARKER | SC_LOWER);
			// Once the FROM key has been approximated, we are done building.
			*pbDoneBuilding = TRUE;
		}
	}

	// Copy or move pieces of the FROM key into the UNTIL key.

	if( pDestKey == pFromKey + uiFromKeyPos)
	{
		if( uiUntilKeyPos < uiMaxKeySize - 2)
		{
			if (!bUntilAtEnd)
			{
				if( bDoMatchBegin)
				{
					if (uiCollationLen)
					{
						f_memcpy( &pUntilKey[ uiUntilKeyPos], pDestKey, uiCollationLen);
						uiUntilKeyPos += uiCollationLen;
					}
					
					// Fill the rest of the key with high values.
					f_memset( &pUntilKey[ uiUntilKeyPos], 0xFF, (uiMaxKeySize - 2) - uiUntilKeyPos);
					uiUntilKeyPos = uiMaxKeySize - 2;
					// Don't need to set the done building flag to TRUE.  
				}
				else if (uiDestKeyLen)
				{
					if( !bDoFirstSubstring)
					{
						f_memcpy( &pUntilKey[ uiUntilKeyPos], pDestKey, uiDestKeyLen);
						uiUntilKeyPos += uiDestKeyLen;
					}
					else
					{
						// Do two copies so that the first substring byte is gone.
						f_memcpy( &pUntilKey[ uiUntilKeyPos], pDestKey, uiCollationLen);
						uiUntilKeyPos += uiCollationLen;
						if( bIsDBCS)
							uiCollationLen++;
						uiCollationLen++;
						f_memcpy( &pUntilKey[ uiUntilKeyPos], pDestKey + uiCollationLen,
										uiDestKeyLen - uiCollationLen);
						uiUntilKeyPos += (uiDestKeyLen - uiCollationLen);
					}

					// Special case again : raw case in index and search comparison.
					// Case has already been completely removed if it is a post index,
					// so no need to change the marker byte.

					if (!(pIfd->uiFlags & IFD_POST) &&
						 (bIsDBCS || 
							(!(pIfd->uiFlags & IFD_UPPER) && bCaseInsensitive)))
					{
						// Add 1 to make sure the until key is higher than the upper value.
						pUntilKey[ uiUntilKeyPos - 1] = (COLL_MARKER | SC_UPPER) + 1;
					}
				}
			}
			else
			{
				if( uiUntilKeyPos > 0)
				{
					// Instead of filling the key with 0xFF, increment the marker.
					pUntilKey [uiUntilKeyPos - 1]++;
				}
				else
				{
					// Keys can have 0xFF values in them, so it is not sufficient to
					// set only uiDestKeyLen bytes to 0xFF.  We must set the entire
					// key.

					f_memset( pUntilKey, 0xFF, uiMaxKeySize - 2);
					uiUntilKeyPos = uiMaxKeySize - 2;
				}
			}
		}
		uiFromKeyPos += uiDestKeyLen;
	}
	else
	{
		// We just built the UNTIL key.  The FROM key doesn't need to be built.
		uiUntilKeyPos += uiDestKeyLen;
	}

Exit:
	// Set the FROM and UNTIL keys
	*puiFromKeyPos = uiFromKeyPos;
	*puiUntilKeyPos = uiUntilKeyPos;
	return( rc);
}

/****************************************************************************
Desc:		Select the best substring for a CONTAINS or MATCH_END search.
			Look below for the algorithm.
****************************************************************************/
FSTATIC FLMBOOL flmSelectBestSubstr(	// Returns TRUE if NOT using first of key
	FLMBYTE **		ppValue,					// [in/out]
	FLMUINT *		puiValueLen,			// [in/out]
	FLMUINT			uiIfdFlags,
	FLMBOOL *		pbTrailingWildcard)		// [in] change if found a wildcard
{
	FLMBYTE *		pValue = *ppValue;
	FLMBYTE *		pCurValue;
	FLMBYTE *		pBest;
	FLMBOOL			bBestTerminatesWithWildCard = *pbTrailingWildcard;
	FLMUINT			uiCurLen;
	FLMUINT			uiBestNumChars;
	FLMUINT			uiBestValueLen;
	FLMUINT			uiWildcardPos = 0;
	FLMUINT			uiTargetNumChars;
	FLMUINT			uiNumChars;
	FLMBOOL			bNotUsingFirstOfString = FALSE;

#define	GOOD_ENOUGH_CHARS			16

	// There may not be any wildcards at all.  Find the first one.
	if( flmFindWildcard( pValue, &uiWildcardPos))
	{

		bBestTerminatesWithWildCard = TRUE;
		pBest = pValue;
		pCurValue = pValue + uiWildcardPos + 1; 
		uiCurLen = *puiValueLen - (uiWildcardPos + 1);

		uiBestValueLen = uiWildcardPos;
		uiBestNumChars = flmCountCharacters( pValue, uiWildcardPos,
			GOOD_ENOUGH_CHARS, uiIfdFlags);
		uiTargetNumChars = uiBestNumChars + uiBestNumChars;

		/*
		Here is the great FindADoubleLengthThatIsBetter algorithm.
		Below are the values to pick a next better contains key.
			First Key Size			Next Key Size that will be used
				1				* 2		2		// Single char searches are REALLY BAD
				2				* 2		4
				3				* 2		6
				4				* 2		8
				...						...
		At each new key piece, increment the target length by 2 so that it
		will be even harder to find a better key.  
		*/
		while( uiBestNumChars < GOOD_ENOUGH_CHARS && *pCurValue)
		{
			if( flmFindWildcard( pCurValue, &uiWildcardPos))
			{
				uiNumChars = flmCountCharacters( pCurValue, uiWildcardPos,
					GOOD_ENOUGH_CHARS, uiIfdFlags);
				if( uiNumChars >= uiTargetNumChars)
				{
					pBest = pCurValue;
					uiBestValueLen = uiWildcardPos;
					uiBestNumChars = uiNumChars;
					uiTargetNumChars = uiNumChars + uiNumChars;
				}
				else
				{
					uiTargetNumChars += 2;
				}
				pCurValue = pCurValue + uiWildcardPos + 1;
				uiCurLen -= uiWildcardPos + 1;
			}
			else
			{
				// Check the last section that may or may not have trailing *.
				uiNumChars = flmCountCharacters( pCurValue, uiCurLen,
					GOOD_ENOUGH_CHARS, uiIfdFlags);
				if( uiNumChars >= uiTargetNumChars)
				{
					pBest = pCurValue;
					uiBestValueLen = uiCurLen;
					bBestTerminatesWithWildCard = *pbTrailingWildcard;
				}
				break;
			}
		}
		if( pBest != *ppValue)
		{
			bNotUsingFirstOfString = TRUE;
		}
		*ppValue = pBest;
		*puiValueLen = uiBestValueLen;
		*pbTrailingWildcard = bBestTerminatesWithWildCard;
	}

//Exit:
	return bNotUsingFirstOfString;
}

/****************************************************************************
Desc:		Returns true if this text will generate a single characater key.
****************************************************************************/
FSTATIC FLMUINT flmCountCharacters(		// Returns number of characters in key
	FLMBYTE * 		pValue,
	FLMUINT			uiValueLen,
	FLMUINT			uiMaxToCount,			
	FLMUINT			uiIfdFlags)
{

	FLMUINT		uiNumChars = 0;
	FLMUINT 		uiObjLength;

	for( uiObjLength = 0; 
		  uiNumChars <= uiMaxToCount && uiValueLen; 
		  pValue += uiObjLength,
			uiValueLen = (uiValueLen >= uiObjLength) ? uiValueLen - uiObjLength : 0)
	{
		switch( (FLMUINT)(GedTextObjType( *pValue)))
		{
			case ASCII_CHAR_CODE:  		// 0nnnnnnn
				uiObjLength = 1;
				if( *pValue == ASCII_SPACE)
				{
					// Ignore if using space rules.
					// VISIT: Need to address ending spaces before a wildcard.
					if( uiIfdFlags & (IFD_MIN_SPACES | IFD_NO_SPACE))
						break;
					uiNumChars++;
				}
				else if( *pValue == ASCII_UNDERSCORE)
				{
					// Ignore if using the underscore space rules.
					// VISIT: Need to address ending spaces before a wildcard.
					if( uiIfdFlags & IFD_NO_UNDERSCORE)
						break;
					uiNumChars++;
				}
				else if( *pValue == ASCII_DASH)
				{
					if( uiIfdFlags & IFD_NO_DASH)
						break;
					uiNumChars++;
				}
				else if( *pValue == ASCII_BACKSLASH && 
					(*(pValue + 1) == ASCII_WILDCARD 
					|| *(pValue + 1) == ASCII_BACKSLASH))
				{
					uiObjLength++;
					uiNumChars++;
				}
				else
				{
					uiNumChars++;
				}
				break;

			case WHITE_SPACE_CODE:		// 110nnnnn
				uiObjLength = 1;
				uiNumChars++;
				break;
			case CHAR_SET_CODE:	  		// 10nnnnnn
			case UNK_EQ_1_CODE:
			case OEM_CODE:
				uiObjLength = 2;
				uiNumChars++;
				break;
			case UNICODE_CODE:			// Unconvertable UNICODE code
			case EXT_CHAR_CODE:
				uiObjLength = 3;
				uiNumChars++;
				break;
			case UNK_GT_255_CODE:
				uiObjLength = 1 + sizeof( FLMUINT16) + FB2UW( pValue + 1);
				break;
			case UNK_LE_255_CODE:
				uiObjLength = 2 + (FLMUINT16)*(pValue + 1);
				break;
			default:							// should NEVER happen: bug if does
				uiObjLength = 1;
				break;						// Should not really return an error
		}
	}
	return uiNumChars;
}
/****************************************************************************
Desc:		Cheating routine to blast out an internal number without using
			the pool to allocate space.
****************************************************************************/
FSTATIC void flmUintToBCD( 
	FLMUINT			uiNum, 
	FLMBYTE *		pNumberBuf, 
	FLMUINT *		puiValueLen)
{

	FLMBYTE		ucNibStk[ F_MAX_NUM_BUF + 1];	/* spare byte for odd BCD counts */
	FLMBYTE *	pucNibStk;

	/* push spare (undefined) nibble for possible half-used terminating byte */

	pucNibStk = &ucNibStk[ 1];

	/* push terminator nibble -- popped last */

	*pucNibStk++ = 0x0F;

	/* push digits */
	/* do 32 bit division until we get down to 16 bits */

	while( uiNum >= 10)
	{
		*pucNibStk++ = (FLMBYTE)(uiNum % 10);	/* push BCD nibbles in reverse order */
		uiNum /= 10;
	}
	*pucNibStk++ = (FLMBYTE)uiNum;			/* push last nibble of number */

	*puiValueLen = (pucNibStk - ucNibStk) >> 1;

	/* Pop stack and pack nibbles into byte stream a pair at a time */

	do
	{
		*pNumberBuf++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);	/* spare stack byte stops seg wrap */

	return;
}

/****************************************************************************
Desc:		Cheating routine to blast out an internal number without using
			the pool to allocate space.
			Code taken from gnbcd.cpp.
****************************************************************************/
FSTATIC void flmIntToBCD( 
	FLMINT			iNum, 
	FLMBYTE *		pNumberBuf, 
	FLMUINT *		puiValueLen)
{
	FLMUINT		uiNum;
	FLMBYTE		ucNibStk[ F_MAX_NUM_BUF + 1];	/* spare byte for odd BCD counts */
	FLMBYTE *	pucNibStk;
	FLMINT		iNegFlag;

	// Initialize byte 0.  It doesn't really matter that we do this, but it
	// keeps purify quiet - we don't end up accessing unitialized memory.

	ucNibStk [0] = 0;

	// push spare (undefined) nibble for possible half-used terminating byte
	// push terminator nibble -- popped last

	ucNibStk [1] = 0x0F;
	pucNibStk = &ucNibStk[ 2];

	/* separate sign from magnituted; (FLMUINT)un = +/- n & flag */

	uiNum = ((iNegFlag = iNum < 0) != 0) ?	-iNum : iNum;

	/* push digits */
	/* do 32 bit division until we get down to 16 bits */

	while( uiNum >= 10)
	{
		*pucNibStk++ = (FLMBYTE)(uiNum % 10);/* push BCD nibbles in reverse order */
		uiNum /= 10;
	}
	*pucNibStk++ = (FLMBYTE)uiNum;			/* push last nibble of number */

	if( iNegFlag)
	{
		*pucNibStk++ = 0x0B;						/* push sign nibble last */
	}

	*puiValueLen = (pucNibStk - ucNibStk) >> 1;

	/* Pop stack and pack nibbles into byte stream a pair at a time */

	do
	{
		*pNumberBuf++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);	/* spare stack byte stops seg wrap */

	return;
}
