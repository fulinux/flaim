//-------------------------------------------------------------------------
// Desc:	Generate a key from a collated key.
// Tabs:	3
//
//		Copyright (c) 1999-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flkeymak.cpp 12263 2006-01-19 14:43:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/********************************************************************
Desc: Build a responce tree of NODEs for the key output.
*********************************************************************/
RCODE flmIxKeyOutput(
	IXD_p				pIxd,
	FLMBYTE *		pucFromKey,
	FLMUINT			uiKeyLen,
	FlmRecord **	ppKeyRV,			// Returns key
	FLMBOOL			bFullFldPaths)	// If true add full field paths
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pKey = NULL;
	void *			pvField;
	FLMBYTE			ucKeyBuf[ MAX_KEY_SIZ + 12];
	FLMBYTE *		pucToKey = &ucKeyBuf[ 0];
	FLMBYTE *		pucPostBuf = NULL;
	IFD_p				pIfd;
	FLMUINT			uiLongValue;
	FLMUINT			uiToKeyLen;
	FLMUINT			uiLanguage = pIxd->uiLanguage;
	FLMUINT			uiFromKeyLen;
	FLMUINT 			uiFromRemaining;
	FLMUINT			uiPostLen;
	FLMUINT			uiPostPos;
	FLMUINT			uiTempFromKeyLen;
	FLMUINT			uiFldType;
	FLMUINT			uiDataType;
	FLMBOOL			bDataRightTruncated;
	FLMBOOL			bFirstSubstring;
	FLMBOOL			bSigSign;
	FLMBYTE			ucTemp;
	FLMUINT			uiContainer;
	FLMUINT			uiMaxKeySize;

	// If the index is on all containers, see if this key has
	// a container component.  If so, strip it off.

	if( (uiContainer = pIxd->uiContainerNum) == 0)
	{
		FLMUINT	uiContainerPartLen = getIxContainerPartLen( pIxd);

		if (uiKeyLen <= uiContainerPartLen)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_BTREE_ERROR);
			goto Exit;
		}
		uiContainer = getContainerFromKey( pucFromKey, uiKeyLen);

		// Subtract off the bytes for the container part.

		uiKeyLen -= uiContainerPartLen;
		uiMaxKeySize = MAX_KEY_SIZ - uiContainerPartLen;
	}
	else
	{
		uiMaxKeySize = MAX_KEY_SIZ;
	}

	// Old code did this.
	flmAssert( uiLanguage != 0xFFFF);

	if (*ppKeyRV)
	{
		if( (*ppKeyRV)->isReadOnly() || (*ppKeyRV)->isCached())
		{
			(*ppKeyRV)->Release();
			*ppKeyRV = NULL;
		}
		else
		{
			(*ppKeyRV)->clear();
		}
	}

	if( (pKey = *ppKeyRV) == NULL)
	{
		if( (pKey = f_new FlmRecord) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		*ppKeyRV = pKey;
	}

	pKey->setContainerID( uiContainer);

	uiFromKeyLen = uiFromRemaining = uiKeyLen;
	pIfd = pIxd->pFirstIfd;

	// If post index, get post low/up section.

	if( pIfd->uiFlags & IFD_POST )
	{

		// Last byte has low/upper length

		uiPostLen = pucFromKey[ uiFromKeyLen - 1 ];
		pucPostBuf = &pucFromKey[ uiFromKeyLen - uiPostLen - 1 ];
		uiPostPos = 0;
	}

	// Allocate [key] root field 
	// VISIT: Removed to be more pure.

	if (RC_BAD( rc = pKey->insertLast( 0, FLM_KEY_TAG, FLM_CONTEXT_TYPE, NULL)))
	{
		goto Exit;
	}

	// Loop for each compound piece of key

	for( ;;)								
	{
	   FLMBOOL	bIsAsianCompound;
	   FLMUINT	uiMarker;

		bDataRightTruncated = bFirstSubstring = FALSE;

  		bIsAsianCompound = (FLMBOOL)(((uiLanguage >= FIRST_DBCS_LANG) && 
												(uiLanguage <= LAST_DBCS_LANG) &&
												(IFD_GET_FIELD_TYPE( pIfd) == FLM_TEXT_TYPE) &&
												(!(pIfd->uiFlags & IFD_CONTEXT)))
											  ? (FLMBOOL)TRUE
											  : (FLMBOOL)FALSE);
	   
		uiMarker = (FLMUINT)((bIsAsianCompound)
									? (FLMUINT)((FLMUINT)(*pucFromKey << 8) +
													*(pucFromKey+1))
									: (FLMUINT) *pucFromKey);
		uiFldType = (FLMUINT) IFD_GET_FIELD_TYPE( pIfd);
		uiDataType = IFD_GET_FIELD_TYPE( pIfd);	
		
		// Hit a compound marker or end of key marker
		// Check includes COMPOUND_MARKER & END_COMPOUND_MARKER

		if( uiMarker <= NULL_KEY_MARKER)		
		{

			// If the field is required or single field then generate an empty node.

			if( ((pIfd->uiFlags & IFD_OPTIONAL) == 0) ||
				 (uiFldType == FLM_TEXT_TYPE) ||
				 (uiFldType == FLM_BINARY_TYPE) ||
				 ((pIfd->uiFlags & IFD_LAST) && !pIfd->uiCompoundPos ))
			{
				if( RC_BAD( rc = flmBuildKeyPaths( pIfd, pIfd->uiFldNum,
											uiDataType, bFullFldPaths, pKey, &pvField)))
					goto Exit;
			}
			if( uiMarker == END_COMPOUND_MARKER)	// Used for post keys
				break;

			uiFromKeyLen = 0;		// This piece is zero - skip it - may be others
		}
		else
		{

			// If compound key or if only field used in index
			// output the key elements field number or else 'NA'

			if( pIfd->uiFlags & IFD_CONTEXT)
			{
				if( RC_BAD( rc = flmBuildKeyPaths( pIfd,
											(FLMUINT)byteToInt( &pucFromKey [1]),
											uiDataType, bFullFldPaths, pKey, &pvField)))
				{
					goto Exit;
				}
				uiFromKeyLen = KY_CONTEXT_LEN;
			}

			else
			{
				if( RC_BAD( rc = flmBuildKeyPaths( pIfd, pIfd->uiFldNum,
											uiDataType, bFullFldPaths, pKey, &pvField)))
				{
					goto Exit;
				}

				// Grab only the Nth section of key if compound key
				//	Null out key if uiToKeyLen gets 0
				
				UD2FBA( 0, pucToKey);

				switch( uiDataType)
				{
					case FLM_TEXT_TYPE:

						uiTempFromKeyLen = uiFromKeyLen;
						uiToKeyLen = FColStrToText( pucFromKey, &uiTempFromKeyLen, pucToKey,
												uiLanguage, pucPostBuf, &uiPostPos, 
												&bDataRightTruncated, &bFirstSubstring);
						uiFromKeyLen = uiTempFromKeyLen;
						break;

					case FLM_NUMBER_TYPE:
					{
						FLMUINT		uiFirstColNibble;		// Current collated nibble
						FLMUINT		uiFirstNumNibble;		// Current output nibble
						FLMBYTE *	pucOutPtr;				// Output pointer
						FLMBYTE *	pucColPtr;
						FLMUINT		uiBytesProcessed;

						// Start at byte after sign/magnitude byte

						pucColPtr = pucFromKey + 1;
						uiBytesProcessed = 1;
						uiFirstColNibble = 1;

						// Determine the sign of the number

						pucOutPtr = pucToKey;
						if( (bSigSign = (*pucFromKey & SIG_POS)) == 0)
						{
							*pucOutPtr = 0xB0;
							uiFirstNumNibble = 0;
						}
						else
						{
							uiFirstNumNibble = 1;
						}

						// Parse through the collated number outputting data
						// to the buffer as we go.

						for( ;;)
						{
							// Determine what we are pointing at

							if( (ucTemp = *pucColPtr) <= COMPOUND_MARKER)
							{
								break;
							}

							if( uiFirstColNibble++ & 1)
							{
								ucTemp >>= 4;
							}
							else
							{
								ucTemp &= 0x0F;
								pucColPtr++;
								uiBytesProcessed++;
							}

							// A hex F signifies the end of a collated number with an
							// odd number of nibbles

							if( ucTemp == 0x0F)
							{
								break;
							}

							// Convert collated number nibble to BCD nibble
							// and lay it in buffer

							ucTemp -= COLLATED_DIGIT_OFFSET;

							// Is number negative?

							if( !bSigSign)
							{
								// Negative values are ~ed

								ucTemp = (FLMBYTE)(10 -(ucTemp + 1));
							}

							if( uiFirstNumNibble++ & 1)
							{
								*pucOutPtr = (FLMBYTE)(ucTemp << 4);
							}
							else
							{
								*pucOutPtr++ += ucTemp;
							}

							if( uiBytesProcessed == uiFromKeyLen)
							{
								break;
							}
						}

						// Append Terminator code to internal number

						*pucOutPtr++ |= (uiFirstNumNibble & 1) ? 0xFF : 0x0F;
						uiToKeyLen = (FLMUINT) (pucOutPtr - pucToKey);
						uiFromKeyLen = uiBytesProcessed;
						rc = FERR_OK;
						break;
					}

					case FLM_BINARY_TYPE:
					{
						FLMUINT			uiMaxLength;
						FLMBYTE *		pucSrc = pucFromKey;

						uiMaxLength = ((uiFromKeyLen >> 1) < uiMaxKeySize)
												? (FLMUINT)(uiFromKeyLen >> 1)
												: (FLMUINT)uiMaxKeySize;
						uiToKeyLen = 0;
						while( (uiToKeyLen < uiMaxLength) && ((ucTemp = *pucSrc) >= COLLS))
						{

							// Take two bytes from source to make one byte in dest

							pucToKey[ uiToKeyLen++] =
								(FLMBYTE)(((ucTemp - COLLS) << 4) + (*(pucSrc + 1) - COLLS));
							pucSrc += 2;
						}

						if( (uiToKeyLen < (uiFromKeyLen >> 1)) && (*pucSrc >= COLLS))
						{
							rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						}
						else
						{
							rc = FERR_OK;
							uiFromKeyLen = uiToKeyLen << 1;

							// FLAIM has a bug where the binary fields don't have
							// the COLL_TRUNCATED value on truncated values.
							// The good news is that we know the true length of
							// binary fields.
							if( *pucSrc == COLL_TRUNCATED)
							{
								uiFromKeyLen++;
								bDataRightTruncated = TRUE;
							}
							else if( uiToKeyLen >= pIfd->uiLimit) 
							{
								bDataRightTruncated = TRUE;
							}
						}
						break;
					}

					case FLM_CONTEXT_TYPE:
					default:
						uiFromKeyLen = 5;

						uiLongValue = (FLMUINT)byteToLong( pucFromKey + 1);
						UD2FBA( (FLMUINT32)uiLongValue, pucToKey);
						uiToKeyLen = 4;
						break;
				}

				if( RC_BAD( rc))
				{
					goto Exit;
				}

				// Allocate and Copy Value into the node

				if( uiToKeyLen)
				{
					FLMBYTE *	pucValue;

					if( RC_BAD(rc = pKey->allocStorageSpace( pvField,
								uiDataType, uiToKeyLen, 0, 0, 0, &pucValue, NULL)))
					{
						goto Exit;
					}
					
					f_memcpy( pucValue, pucToKey, uiToKeyLen);
				}

				// Set first sub-string and truncated flags.

				if( (pIfd->uiFlags & IFD_SUBSTRING) && !bFirstSubstring)
				{
					pKey->setLeftTruncated( pvField, TRUE);
				}
				if( bDataRightTruncated)
				{
					pKey->setRightTruncated( pvField, TRUE);
				}
			}
		}

		// Compute variables for next section of compound key
		// Add 1 for compound marker if still is stuff in key

		if( uiFromRemaining != uiFromKeyLen)
		{
			uiFromKeyLen += (FLMUINT)(bIsAsianCompound ? (FLMUINT)2 : (FLMUINT)1);
		}
		pucFromKey += uiFromKeyLen;						/* Position to end of section */
		if( (uiFromKeyLen = (uiFromRemaining -= uiFromKeyLen)) == 0)
		{
			break;										/* Finished with this key */
		}
		while( ((pIfd->uiFlags & IFD_LAST) == 0) 
		   &&   (pIfd->uiCompoundPos == (pIfd+1)->uiCompoundPos))
		{
			pIfd++;
		}
		if( pIfd->uiFlags & IFD_LAST)
		{
			break;
		}
		
		pIfd++;
	}

	// Check if we have one field left. 

	if( (pIfd->uiFlags & IFD_LAST) == 0)
	{
		while( (pIfd->uiFlags & IFD_LAST) == 0) 
		{
			pIfd++;
		}
		if( (pIfd->uiFlags & IFD_OPTIONAL) == 0)
		{
			if( RC_BAD( rc = flmBuildKeyPaths( pIfd, pIfd->uiFldNum,
										uiDataType, bFullFldPaths, pKey, &pvField)))
			{
				goto Exit;
			}
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	This module will read all references of an index key.
		The references will be output number defined as REFS_PER_NODE
****************************************************************************/
RCODE flmBuildKeyPaths(
	IFD_p			pIfd,
	FLMUINT		uiFldNum,
	FLMUINT		uiDataType,
	FLMBOOL		bFullFldPaths,
	FlmRecord *	pKey,
	void **		ppvField)
{
	RCODE			rc = FERR_OK;
	void *		pvField;
	void *		pvParentField;
	void *		pvChildField;
	FLMUINT *	pFieldPath;
	FLMUINT		uiTempDataType;
	FLMUINT		uiFieldPos;
	FLMUINT		uiTargetFieldID;

	// Easy case first.
	if( !bFullFldPaths)
	{
		rc = pKey->insertLast( 1, uiFldNum, uiDataType, &pvField);
		goto Exit;
	}

	pFieldPath = pIfd->pFieldPathPToC;
	pvParentField = pKey->root();
	uiFieldPos = 0;
	// Loop finding field matches.

	pvField = pKey->find( pvParentField, pFieldPath[ uiFieldPos]);
	if( pvField)
	{
		pvParentField = pvField;
		uiFieldPos++;
		uiTargetFieldID = pFieldPath[ uiFieldPos];

		// Loop finding matching children from this point on.

		for( pvChildField = pKey->firstChild( pvParentField); pvChildField; )
		{
			if( pKey->getFieldID( pvChildField) == uiTargetFieldID)
			{
				// On the child field?
				if( pFieldPath[ uiFieldPos + 1] == 0)
				{
					pvField = pvChildField;
					// Set the data type in case the data length is zero.
					pKey->allocStorageSpace( pvField, uiDataType, 0, 0, 0, 0, NULL, NULL);
					break;
				}
				pvParentField = pvChildField;
				uiFieldPos++;
				uiTargetFieldID = pFieldPath[ uiFieldPos];
				pvChildField = pKey->firstChild( pvParentField);
			}
			else
			{
				pvChildField = pKey->nextSibling( pvChildField);
			}
		}
	}

	// Insert the rest of the field path down to the value field (uiFieldPos==0).

	uiTempDataType = FLM_CONTEXT_TYPE;
	for( ; pFieldPath[ uiFieldPos]; uiFieldPos++)
	{
		// Add the real data type for the last field, otherwise set as context.
		if( pFieldPath[ uiFieldPos + 1] == 0)
		{
			uiTempDataType = uiDataType;
		}

		if( RC_BAD( rc = pKey->insert( pvParentField, INSERT_LAST_CHILD,
								pFieldPath[ uiFieldPos], uiTempDataType, &pvField)))
		{
			goto Exit;
		}
		pvParentField = pvField;
	}

Exit:
	*ppvField = pvField;
	return( rc);
}
