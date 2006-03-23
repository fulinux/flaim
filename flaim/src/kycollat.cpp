//-------------------------------------------------------------------------
// Desc:	Key collation routines.
// Tabs:	3
//
//		Copyright (c) 1991-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kycollat.cpp 12313 2006-01-19 15:14:44 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE KYFormatText(
	const FLMBYTE *	psVal,
	FLMUINT				uiSrcLen,
	FLMBOOL				bMinSpaces,
	FLMBOOL				bNoUnderscore,
	FLMBOOL				bNoSpace,
	FLMBOOL				bNoDash,
	FLMBOOL				bEscChar,
	FLMBOOL				bInputTruncated,
	FLMBYTE *			psDestBuf,
	FLMUINT *			puiDestLen);

/****************************************************************************
Desc:	Create an index key given a keyTree and index definition. This routine
		works on a normal data tree - used in FlmKeyBuild. where
     	a data record is traversed with field paths being checked.

Ret:	FERR_OK, FERR_MEM, FERR_DATA_CONVERSION
****************************************************************************/
RCODE KYTreeToKey(
	FDB_p				pDb,
	IXD_p				pIxd,
	FlmRecord *		pRecord,
	FLMUINT			uiContainerNum,
	FLMBYTE * 		pKeyBuf,
	FLMUINT *		puiKeyLenRV,
	FLMUINT			uiFlags)
{
	RCODE					rc = FERR_OK;
	IFD_p					pIfd;
	void *				pvMatchField;
	FLMBYTE *			pToKey = pKeyBuf;
	const FLMBYTE *	pExportPtr;
	FLMUINT				uiToKeyLen;
	FLMUINT				uiTotalLen;
	FLMINT				nth;
	FLMINT				iMissingFlds;
	FLMUINT     		uiIskPostFlag;
	FLMUINT    			uiLuLen;
	FLMUINT				uiPieceLuLen;
	FLMUINT				uiLanguage;
	FLMUINT				uiIsPost = 0;
	FLMBOOL				bIsAsianCompound;
	FLMBOOL				bIsCompound;
	FLMBYTE    			LowUpBuf [MAX_LOWUP_BUF];
	FLMUINT				uiMaxKeySize = (pIxd->uiContainerNum)
										? MAX_KEY_SIZ
										: MAX_KEY_SIZ - getIxContainerPartLen( pIxd);
	
	if ((uiLanguage = pIxd->uiLanguage) == 0xFFFF)
	{
		uiLanguage = pDb->pFile->FileHdr.uiDefaultLanguage;
	}
	
	uiLuLen = 0;
	iMissingFlds = 0;
	uiTotalLen = 0;

	pIfd = pIxd->pFirstIfd;
	bIsCompound = (pIfd->uiFlags & IFD_COMPOUND) ? TRUE : FALSE;
	for (;;pIfd++)
	{

		/* Set if IFD is post. */

		uiIsPost |= (FLMUINT) (uiIskPostFlag = (FLMUINT)IFD_IS_POST_TEXT( pIfd));

  		bIsAsianCompound =((uiLanguage >= FIRST_DBCS_LANG) && 
								 (uiLanguage <= LAST_DBCS_LANG) && 
								 (IFD_GET_FIELD_TYPE( pIfd) == FLM_TEXT_TYPE) &&
								 (!(pIfd->uiFlags & IFD_CONTEXT)));
		nth = 1;
		uiToKeyLen = 0;

		// Find matching node in the tree - if not found skip and continue.

FIND_NXT:
		if( (pvMatchField = pRecord->find( pRecord->root(), pIfd->uiFldNum, nth)) != NULL)
		{

			/* Match was found, now if flagged, validate its parent path. */

			if( uiFlags & KY_PATH_CHK_FLAG)
			{
				FLMUINT *	puiFieldPath;
				void *		pTempField = pvMatchField;
				FLMUINT		uiCurrentFld;

				puiFieldPath = pIfd->pFieldPathCToP;
				for( uiCurrentFld = 1; puiFieldPath [uiCurrentFld]; uiCurrentFld++)
				{
					if( ((pTempField = pRecord->parent( pTempField)) == NULL) ||
						 (pRecord->getFieldID( pTempField) != puiFieldPath [uiCurrentFld]))
					{
						nth++;
						goto FIND_NXT;
					}
				}
			}
			
      	/* Convert the node's key value to the index type. */

			/* Compute maximum. bytes remaining. */

			uiToKeyLen = uiMaxKeySize - uiTotalLen;

			/* Take the tag and make it the key. */

			if( pIfd->uiFlags & IFD_CONTEXT)
			{

        		/* Output the tag number. */

				*pToKey = KY_CONTEXT_PREFIX;
				flmUINT16ToBigEndian( (FLMUINT16) pRecord->getFieldID( pvMatchField), &pToKey [1]);
				uiToKeyLen = KY_CONTEXT_LEN;
			}
			else
			{
				pExportPtr = pRecord->getDataPtr( pvMatchField);

				if( RC_BAD( rc = KYCollateValue( pToKey, &uiToKeyLen, pExportPtr,
							pRecord->getDataLength( pvMatchField),
							pIfd->uiFlags, pIfd->uiLimit,
							NULL, &uiPieceLuLen, uiLanguage, bIsCompound, 
							(FLMBOOL) ((pIfd->uiFlags & IFD_SUBSTRING)
									? (pRecord->isLeftTruncated( pvMatchField) 
										? FALSE : TRUE)
									: FALSE),
							pRecord->isRightTruncated( pvMatchField), NULL)))
				{
					goto Exit;
				}

				if( pRecord->isRightTruncated( pvMatchField))
				{
					/*
					VISIT: This is a bug in f_tocoll.cpp that if we
					fix all text indexes could be corrupt.  If the string
					is EXACTLY the length of the truncation length then
					it should, but doesn't, set the truncation flag.  
					The code didn't match the design intent.
					*/
					f_memmove( &pToKey[ uiToKeyLen - uiPieceLuLen + 1],	
								  &pToKey[ uiToKeyLen - uiPieceLuLen], uiPieceLuLen);
					pToKey[ uiToKeyLen - uiPieceLuLen] = COLL_TRUNCATED;
					uiToKeyLen++;
				}

				if( uiIskPostFlag)		/* uiPieceLuLen cannot be 0 */
				{
					uiToKeyLen -= uiPieceLuLen;
					f_memcpy( &LowUpBuf [uiLuLen], 
									&pToKey [uiToKeyLen], uiPieceLuLen );
					uiLuLen += uiPieceLuLen;
				}
			}
		}		/* ifend gedFind==NULL */

		//	Check here if key found else the fields are missing.

		if( uiToKeyLen)
		{
			iMissingFlds = 0;
			pToKey    += uiToKeyLen;
			uiTotalLen += uiToKeyLen;

			// Go to the last IFD with the same compound position.

			while( ((pIfd->uiFlags & IFD_LAST) == 0)
			   &&   (pIfd->uiCompoundPos == (pIfd+1)->uiCompoundPos))
			{
				pIfd++;
			}
		}
		else											/* no matching field found */
		{
			// Continue if there are still fields with same compound position.
			
			if( ((pIfd->uiFlags & IFD_LAST) == 0)
		   &&   (pIfd->uiCompoundPos == (pIfd+1)->uiCompoundPos))
			{
				continue;
			}
		
			iMissingFlds++;
			if( bIsAsianCompound)
				iMissingFlds++;
   	}

		// Check if done. 

		
		if( pIfd->uiFlags & IFD_LAST)
   	 	break;

		if( bIsCompound)
		{
			if( bIsAsianCompound)					/* Output 2 bytes for marker */
			{
				*pToKey++ = 0;
				uiTotalLen++;
			}	
			*pToKey++ = COMPOUND_MARKER;
			uiTotalLen++;
		}
		else if( uiToKeyLen )							/* multi-field index key */
			break;
	} /* FOR LOOP END */
	
	/*
	Back up iMissingFlds-1 because last
	field does not have compound marker.
	Add 4 bytes of foxes for high values.
	*/

	if( iMissingFlds && (uiFlags & KY_HIGH_FLAG) && (bIsCompound))
	{

		/*
		Ignore the last one or two iMissingFlds values because a compound
		marker was not added to the end of the key.
		*/

		if( bIsAsianCompound)
			iMissingFlds--;
		uiTotalLen -= --iMissingFlds;
		pToKey    -= iMissingFlds;

		/*
		Fill with high values to the end of the buffer.
		It is easy for double byte ASIAN collation values to all be 0xFF.
		*/

		if( uiTotalLen < uiMaxKeySize)
		{
			f_memset( pToKey, 0xFF, uiMaxKeySize - uiTotalLen );
			pToKey += (uiMaxKeySize - uiTotalLen);
			uiTotalLen += (uiMaxKeySize - uiTotalLen);
		}
	}
	else if( uiIsPost)				/* else take care of post index */
	{
		uiTotalLen += KYCombPostParts( pKeyBuf, uiTotalLen, LowUpBuf, uiLuLen,
											  uiLanguage, (FLMUINT)(pIfd->uiFlags));
	}

	// Add container number to the key if the index is on all containers.

	if (!pIxd->uiContainerNum)
	{
		appendContainerToKey( pIxd, uiContainerNum, pKeyBuf, &uiTotalLen);
	}
	*puiKeyLenRV = uiTotalLen;
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Build a collated key value piece.
****************************************************************************/
RCODE KYCollateValue(
	FLMBYTE *			pDest,
	FLMUINT *			puiDestLenRV,
	const FLMBYTE *	pSrc,
	FLMUINT				uiSrcLen,
	FLMUINT				uiFlags,
	FLMUINT				uiLimit,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiLuLenRV,
	FLMUINT				uiLanguage,
	FLMBOOL				bCompoundPiece,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bInputTruncated,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL				bFldIsEncrypted)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiDestLen;
	FLMUINT		uiDataType = uiFlags & 0x0F;

	// Treat an encrypted field as binary for collation purposes.
	if (bFldIsEncrypted)
	{
		uiDataType = FLM_BINARY_TYPE;
	}
	
	if( puiLuLenRV)
	{
		*puiLuLenRV = 0;
	}

	if( (uiDestLen = *puiDestLenRV) == 0)
	{
		return( RC_SET( FERR_KEY_OVERFLOW));
	}

	if( uiDataType == FLM_TEXT_TYPE)
	{
		FLMUINT	uiCharLimit;
		FLMBYTE	byTmpBuf[ MAX_KEY_SIZ + 8];		// OK on the stack.

		if(uiFlags & (IFD_MIN_SPACES | IFD_NO_UNDERSCORE | 
							IFD_NO_SPACE | IFD_NO_DASH | IFD_ESC_CHAR)) 
		{
			if( RC_BAD( rc = KYFormatText( 
					pSrc, uiSrcLen,
					(uiFlags & IFD_MIN_SPACES) ? TRUE : FALSE,
					(uiFlags & IFD_NO_UNDERSCORE) ? TRUE : FALSE,
					(uiFlags & IFD_NO_SPACE) ? TRUE : FALSE,
					(uiFlags & IFD_NO_DASH) ? TRUE : FALSE,
					(uiFlags & IFD_ESC_CHAR) ? TRUE : FALSE,
					bInputTruncated, byTmpBuf, &uiSrcLen)))
			{
				goto Exit;
			}
			pSrc = (FLMBYTE *) byTmpBuf;
		}

		uiCharLimit = uiLimit ? uiLimit : IFD_DEFAULT_LIMIT;
		if( (uiLanguage >= FIRST_DBCS_LANG ) && (uiLanguage <= LAST_DBCS_LANG))
		{
			rc = AsiaFlmTextToColStr( pSrc, uiSrcLen, pDest, &uiDestLen,
								(uiFlags & IFD_UPPER), puiCollationLen, puiLuLenRV, 
								uiCharLimit, bFirstSubstring, pbDataTruncated);
		}
		else
		{
			rc = FTextToColStr( pSrc, uiSrcLen, pDest, &uiDestLen,
								(uiFlags & IFD_UPPER), puiCollationLen, puiLuLenRV, 
								uiLanguage, uiCharLimit, bFirstSubstring,
								pbOriginalCharsLost, pbDataTruncated);
		}
	}
	
	// TRICKY: uiDestLen could be set to zero if text and no value.

	if( !uiSrcLen || !uiDestLen)
	{
		if( !bCompoundPiece)
		{
			// Zero length key. Any value under 0x1F would work.
			if( (uiLanguage >= FIRST_DBCS_LANG ) && 
				 (uiLanguage <= LAST_DBCS_LANG))
			{
				pDest [0] = 0;
				pDest [1] = NULL_KEY_MARKER;
				uiDestLen = 2;
			}
			else
			{
				pDest [0] = NULL_KEY_MARKER;
				uiDestLen = 1;
			}
		}
		else
		{
			uiDestLen = 0;
		}
		goto Exit;
	}

 	switch (uiDataType)
	{
		case FLM_TEXT_TYPE:
			break;

		case FLM_NUMBER_TYPE:				/* parse internal number; generate key */
		{
			FLMBYTE *			pOutput = pDest + 1;		/* First byte holds sign/magnitude */
			const FLMBYTE *	pTempSrc = pSrc;
			FLMUINT				uiBytesOutput = 1;		/* Allow for sign/magnitude byte */
			FLMUINT				uiMaxOutLen = uiDestLen;
			FLMINT				iHiInNibble = 1;
			FLMINT				iHiOutNibble = 1;
			FLMUINT				uiSigSign = SIG_POS;		/* hi bit causes + to collate before */
			FLMUINT				uiMagnitude = COLLATED_NUM_EXP_BIAS - 1;
			FLMBYTE				byValue;

			for (rc = FERR_OK;;)
			{
				switch(							/* Determine what we're pointing at */
					byValue = (iHiInNibble++ & 1)	/* test if high/low nibble */
						?	(FLMBYTE)(*pTempSrc >> 4)		/* high: shift for math below */
						:	(FLMBYTE)(*pTempSrc++ & 0x0F) /* low: mask off high & point next */
				){
					case 0x0B:					/* Negative Sign code */
						uiSigSign = 0;
						continue;
					case 0x0A:  				/* Ignore for now - not implemented */
					case 0x0C:
					case 0x0D:
					case 0x0E:
						continue;
					case 0x0F:					/* Terminator */
						*pDest = (FLMBYTE)(uiSigSign |
										((uiSigSign ? uiMagnitude : ~uiMagnitude) & 0x7F));
						goto NumDone;
					default:						/* Numeric digits */
						uiMagnitude++;			/* Determine the magnitude as we go */
						if( uiSigSign)			/* positive number */
							byValue += COLLATED_DIGIT_OFFSET;
						else						/* invert for key collation */
							byValue = (FLMBYTE)((COLLATED_DIGIT_OFFSET + 9) - byValue);
						if( iHiOutNibble++ & 1)	/* test high/low output nibble */
						{
							/* need another byte; check length */

							if( uiBytesOutput++ == uiMaxOutLen)
							{
								uiBytesOutput = 0;
								rc = RC_SET( FERR_KEY_OVERFLOW);
								goto NumDone;
							}

							*pOutput = (FLMBYTE)((byValue << 4)	/* store high nibble */
								| 0x0F);				/* pre-set terminator (may be last)*/
						}
						else
						{
							*pOutput++ &= (FLMBYTE)(byValue | 0xF0);/* reset low nib-high wasn't last */
						}
						continue;
				}
			}
NumDone:
			uiDestLen = uiBytesOutput;
		}
			break;

		case FLM_BINARY_TYPE:
		{
			FLMUINT				uiLength = uiSrcLen;
			const FLMBYTE *	tmpSrc = pSrc;
			FLMBYTE *			tmpDest = pDest;
			FLMBOOL				bTruncated = FALSE;

			if( uiLength >= uiLimit)
			{
				uiLength = uiLimit;
				bTruncated = TRUE;
			}

			if( uiDestLen < (uiLength << 1))
			{
				// Compute length so will not overflow

				uiLength = (FLMUINT)(uiDestLen >> 1);
			}
			else
			{
				uiDestLen = (FLMUINT)(uiLength << 1);
			}

			// Convert each byte to two bytes

			while( uiLength--)
			{
				*tmpDest++ = (FLMBYTE)(COLLS + ((*tmpSrc) >> 4));
				*tmpDest++ = (FLMBYTE)(COLLS + ((*tmpSrc++) & 0x0F));
			}

			if( bTruncated)
			{
				*tmpDest++ = COLL_TRUNCATED;
			}
			break;
		}

		case FLM_CONTEXT_TYPE:
		{
			if( uiDestLen < 5)
			{
				uiDestLen = 0;
				rc = RC_SET( FERR_KEY_OVERFLOW);
			}
			else
			{
				*pDest = 0x1F;
				flmUINT32ToBigEndian( FB2UD( pSrc), pDest + 1);
				uiDestLen = 5;
				rc = FERR_OK;
			}
			break;
		}

		default:
			rc = RC_SET( FERR_CONV_BAD_DEST_TYPE);
			break;
	}

Exit:

	*puiDestLenRV = uiDestLen;
	return( rc);
}

/****************************************************************************
Desc:		Format text removing leading and trailing spaces.  Treat 
			underscores as spaces.  As options, remove all spaces and dashes.
Ret:		FERR_OK always.  WIll truncate so text will fill MAX_KEY_SIZ.
			Allocate 8 more than MAX_KEY_SIZ for psDestBuf.
Visit:	Pass in uiLimit and pass back a truncated flag when the
			string is truncated.  This was not done because we will have
			to get the exact truncated count that is done in f_tocoll.cpp
			and that could introduce some bugs.
****************************************************************************/
FSTATIC RCODE KYFormatText(
	const FLMBYTE *	psVal,			// Points to value source 
	FLMUINT				uiSrcLen,		// Length of the key-NOT NULL TERMINATED
												// Booleans below are zero or NON-zero
	FLMBOOL				bMinSpaces,		// Remove leading/trailing/multiple spaces
	FLMBOOL				bNoUnderscore,	// Convert underscore to space
	FLMBOOL				bNoSpace,		// Remove all spaces
	FLMBOOL				bNoDash,			// Remove all dashes (hyphens)
	FLMBOOL				bEscChar,		// Literal '*' or '\\' char after '\\' esc char
	FLMBOOL				bInputTruncated,// TRUE if input key data is truncated.
	FLMBYTE *			psDestBuf,		// (out) Destination buffer
	FLMUINT *			puuiDestLen)	// (out) Length of key in destination buffer.
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		psDestPtr = psDestBuf;
	FLMBYTE			ucValue;
	FLMBYTE			objType;
	FLMUINT			uiCurPos = 0;
	FLMUINT			uiDestPos = 0;
	FLMUINT			uiOldDestPos = 0;
	FLMUINT 			objLength;
	FLMBOOL			bLastCharWasSpace = bMinSpaces;

	for( ; uiCurPos < uiSrcLen && uiDestPos < MAX_KEY_SIZ - 1; uiCurPos += objLength )
	{
		ucValue = psVal [uiCurPos];
		objLength = 1;
		uiOldDestPos = uiDestPos;
		objType = (FLMBYTE)(GedTextObjType( ucValue));

		switch( objType)
		{
			case ASCII_CHAR_CODE:  			/* 0nnnnnnn */

				if( (ucValue == ASCII_SPACE) || ((ucValue == ASCII_UNDERSCORE) && bNoUnderscore))
				{
					if( bLastCharWasSpace || bNoSpace)
					{
						break;
					}
					// Sets to true if we want to minimize spaces.
					bLastCharWasSpace = bMinSpaces;
					ucValue = ASCII_SPACE;
				}
				// FYI: There are about 13 different UNICODE hyphen characters.
				else if( (ucValue == ASCII_DASH) && bNoDash)
				{
					break;
				}
				else
				{
					if( (ucValue == ASCII_BACKSLASH) && bEscChar && 
							(psVal [uiCurPos+1] == ASCII_WILDCARD || psVal [uiCurPos+1] == ASCII_BACKSLASH))
					{
						ucValue = psVal [uiCurPos+1];
						objLength++;
					}
					bLastCharWasSpace = FALSE;
				}
				psDestPtr[ uiDestPos++] = ucValue;
				break;
			case WHITE_SPACE_CODE:			/* 110nnnnn */
				if( bLastCharWasSpace || bNoSpace)
				{
					break;
				}
				// Sets to true if we want to minimize spaces.
				bLastCharWasSpace = bMinSpaces;
				psDestPtr[ uiDestPos++] = ASCII_SPACE;
				break;
			case CHAR_SET_CODE:	  			/* 10nnnnnn */
			case UNK_EQ_1_CODE:
			case OEM_CODE:
				bLastCharWasSpace = FALSE;
				psDestPtr[ uiDestPos++] = psVal [uiCurPos];
				psDestPtr[ uiDestPos++] = psVal [uiCurPos+1];
				objLength = 2;
				break;
			case UNICODE_CODE:				/* Unconvertable UNICODE code */
			case EXT_CHAR_CODE:				/* Full extended character */
				bLastCharWasSpace = FALSE;
				psDestPtr[ uiDestPos++] = psVal [uiCurPos];
				psDestPtr[ uiDestPos++] = psVal [uiCurPos+1];
				psDestPtr[ uiDestPos++] = psVal [uiCurPos+2];
				objLength = 3;
				break;
			case UNK_GT_255_CODE:
				bLastCharWasSpace = FALSE;
				objLength = 1 + sizeof( FLMUINT) + FB2UW( &psVal [uiCurPos + 1]);
				break;
			case UNK_LE_255_CODE:
				bLastCharWasSpace = FALSE;
				objLength = 2 + (FLMUINT) (psVal [uiCurPos+1]);
				break;
			default:				/* should NEVER happen: same as other code like this. */
				psDestPtr[ uiDestPos++] = psVal [uiCurPos];
				bLastCharWasSpace = FALSE;
				break;
		}
	}

	// On overflow - back out of the last character.
	if( uiDestPos >= MAX_KEY_SIZ - 1)
	{
		uiDestPos = uiOldDestPos;
		bLastCharWasSpace = FALSE;
	}
	// Handle the trailing space if present.
	// bLastCharWasSpace cannot be set to true if bNoSpace is true.
	if( bLastCharWasSpace && uiDestPos && !bInputTruncated)
	{
		uiDestPos--;
	}
	psDestPtr[ uiDestPos] = '\0';
	*puuiDestLen = (FLMUINT) uiDestPos;

//Exit:
	return( rc);
}

