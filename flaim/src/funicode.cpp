//-------------------------------------------------------------------------
// Desc:	Unicode functions.
// Tabs:	3
//
//		Copyright (c) 1999-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: funicode.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMUINT flmUnicodeToWP(
	const FLMUNICODE *	puzUniStr,	
	FLMUINT16 *				pWPChr);

/****************************************************************************
Desc: 	Returns the size of buffer needed to hold the unicode string in 
			FLAIM's storage format.
****************************************************************************/
FLMEXP FLMUINT FLMAPI FlmGetUnicodeStorageLength(
	const FLMUNICODE *	puzStr)
{
	FLMBYTE		chrSet;
	FLMUINT		uiStorageLength = 0;
	FLMUINT		uniLength;
	FLMUINT16	wp60Buf[12];

	flmAssert( puzStr != NULL);

	// Two passes are needed to store a UNICODE string:
	//   1st pass determines the storage length (via FlmGetUnicodeStorageLength)
	//   2nd pass stores the string into FLAIMs internal text format
	//	  (via FlmUnicode2Storage).

	do
	{
		//  Cannot check for A..Z because flmUnicodeToWP may convert
		//  multiple Unicode characters into 1 WP char - (D-slash)
		//  This 'complex' convert code is defined out.
		//
		//  Personally, I don't think this should ever be done, but the
		//  conversions must be looked at.  The hard part of all of this
		//  is deciding if we should have perfect UNI-->WP60-->UNI where
		//  the 2nd UNI is exactly the same as the first.
		//
		//  For the NDS project, this code MUST have exact conversions.

		if( *puzStr < 0x20)
		{
			uniLength = 1;
			uiStorageLength += 3;
		}				
		else
		{
			// This is a speed good optimization.

			if( *puzStr < 0x7F)
			{
				uiStorageLength++;
				puzStr++;
				continue;
			}

			uniLength = flmUnicodeToWP( puzStr, wp60Buf);

			if( !uniLength)
			{
				uiStorageLength += 3;
				uniLength = 1;
			}
			else
			{
				if( (chrSet = (FLMBYTE) (wp60Buf[0] >> 8)) == 0)
				{
					uiStorageLength++;
				}
				else
				{
					uiStorageLength += (chrSet <= 63) ? 2 : 3;
				}
			}
		}
		puzStr += uniLength;

	} while( *puzStr != 0 );

	return( uiStorageLength);
}

/****************************************************************************
Desc: 	Copies and formats a Unicode string into FLAIM's storage format.
			The Unicode string must be in little endian format.
			Unicode values that are not represented as WordPerfect 6.x characters
			are preserved as non-WP characters.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmUnicode2Storage(
	const FLMUNICODE *	puzStr,
	FLMUINT *				puiBufLength,
	FLMBYTE *				pBuf)
{
	FLMBYTE			chrSet;
	FLMUINT16		wp60Buf[ 12];
	FLMUINT			uiStorageLength = 0;
	FLMUINT			uniLength;

	flmAssert( puzStr != NULL);
	flmAssert( pBuf != NULL);

	do
	{
		if( *puzStr < 0x20 )
		{
			// Output the character as an unconvertable unicode character.

			*pBuf++ = UNICODE_CODE;
			*pBuf++ = *puzStr >> 8;
			*pBuf++ = (FLMBYTE) *puzStr;
			uniLength = 1;
			uiStorageLength += 3;
		}
		else
		{
			if( *puzStr < 0x7F)
			{
				uiStorageLength++;
				*pBuf++ = (FLMBYTE)*puzStr++;
				continue;
			}

			uniLength = flmUnicodeToWP( puzStr, wp60Buf);

			if( !uniLength)
			{
				*pBuf++ = UNICODE_CODE;
				*pBuf++ = *puzStr >> 8;
				*pBuf++ = (FLMBYTE)*puzStr;
				uniLength = 1;
				uiStorageLength += 3;
			}
			else
			{
				chrSet = wp60Buf[0] >> 8;

				if( chrSet == 0)
				{
					*pBuf++ = (FLMBYTE) wp60Buf[0];
					uiStorageLength++;
				}
				else if( chrSet <= 63)
				{
					*pBuf++ = CHAR_SET_CODE | chrSet;
					*pBuf++ = (FLMBYTE) wp60Buf[0];
					uiStorageLength += 2;
				}
				else
				{
					*pBuf++ = EXT_CHAR_CODE;
					*pBuf++ = chrSet;
					*pBuf++ = (FLMBYTE) wp60Buf[0];
					uiStorageLength += 3;
				}
			}
		}
		puzStr += uniLength;

		// Make sure input buffer was large enough

		if( *puiBufLength < uiStorageLength)
		{
			return( RC_SET( FERR_CONV_DEST_OVERFLOW));
		}

	} while( *puzStr != 0);

	*puiBufLength = uiStorageLength;
	return( FERR_OK );
}

/****************************************************************************
Desc:		Convert  from Unicode to 1 and only 1 WP60 character
Ret:		Conversion Count - 0 means Unicode character could not be converted.
Notes:	See commented out code below this for real neat multiple character
			conversions.  We don't really want this so that the original
			UNICODE characters are preserved on get/put as much as possible.
			Code copied from WPTEXT\WPCHU.C in WpChUUniToWPLang() because 
			of the multiple character conversion and that we only do one 
			character at a time for both interfaces.  
			Called from the UNICODE put routine above and QuickFinder 
			UNICODE to WP conversion.
****************************************************************************/
FSTATIC FLMUINT flmUnicodeToWP(
	const FLMUNICODE *	pUniStr,
	FLMUINT16 *				pWPChr)
{
	FLMUINT		uiReturnLen = 1;
	FLMUNICODE	uzUniChar = *pUniStr;
	FLMINT16		max;
	FLMINT16		min;
	FLMINT16		temp;
	FLMUINT16 *	tablePtr;
	FLMUINT16	tblChr;

	if( uzUniChar < 127)
	{
		*pWPChr = uzUniChar;
		goto Exit;
	}

	tablePtr = (FLMUINT16 *) WP_UTOWP60;

	// Value we should use ... max = UTOWP60_ENTRIES - 1;
	// Bug introduced before Nov99 where UTOWP60_ENTRIES is actually 1502 
	// and the value of 2042 was used.  Through debugging, all values in the 
	// table from 1021 to 1502 were never converted to WP character.  So, in order
	// to search correctly on these values we must preserve the WRONG conversion
	// of these characters (Unicode x222E on).  The new max table size is 1021 so
	// max will be set to 1020 to work correctly.

	max = 1020;
	min = 0;

	do
	{
		temp = (min+max) >> 1;
		tblChr = *(tablePtr+(temp*2));
		if( tblChr < uzUniChar )
		{
			min = temp+1;
		}
		else if( tblChr > uzUniChar )
		{
			max = temp-1;
		}
		else
		{
			*pWPChr = *(tablePtr + (temp*2) + 1);
			goto Exit;
		}

	} while( min <= max);

	uiReturnLen = 0;
	
Exit:

	return( uiReturnLen );
}

/****************************************************************************
Desc: 	Converts storage formats to UNICODE.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmStorage2Unicode(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,
	FLMUINT *			puiStrBufLen,
	FLMUNICODE *		puzStrBuf)
{
	FLMUNICODE *	tablePtr;
	FLMBYTE			c;
	FLMUINT			bytesProcessed = 0;
	FLMUINT			bytesOutput = 0;
	FLMUINT			outputData;
	FLMUINT			maxOutLen;
	FLMBYTE			objType;
	FLMUINT			objLength = 0;
	FLMBYTE			tempBuf[ 80];
	FLMBYTE			chrSet, chrVal;
	FLMUNICODE		newChrVal;
	RCODE				rc = FERR_OK;

	// If the value is a number, convert to text first

	if( uiValueType != FLM_TEXT_TYPE)
	{
		if( pucValue == NULL)
		{
			uiValueLength = 0;
		}
		else
		{
			if( uiValueType == FLM_NUMBER_TYPE)
			{
				uiValueLength = sizeof( tempBuf);
				rc = GedNumToText( pucValue, tempBuf, &uiValueLength);
			}
			else	
			{
				rc = RC_SET( FERR_CONV_ILLEGAL);
				goto Exit;
			}

			if( RC_BAD(rc))
			{
				goto Exit;
			}

			pucValue = &tempBuf[ 0];
		}
	}

	maxOutLen = *puiStrBufLen;
	outputData = ((puzStrBuf != NULL) && (maxOutLen > 1));

	if( outputData)
	{
		maxOutLen -= 2;
	}

	// Parse through the string outputting data to the buffer as we go

	while( bytesProcessed < uiValueLength)
	{
		// Determine what we are pointing at

		c = *pucValue;
		objType = GedTextObjType( c);
		switch( objType)
		{
			case ASCII_CHAR_CODE:
				objLength = 1;
				if( outputData)
				{
					if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto GedGetUNICODE_Output;
					}

					*puzStrBuf++ = c;
				}
				bytesOutput += 2;
				break;

			case CHAR_SET_CODE:
				objLength = 2;
				if( outputData)
				{
					if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto GedGetUNICODE_Output;
					}

					// Convert WP to UNICODE
					
					chrSet = c & 0x3F;
					chrVal = *(pucValue + 1);
				
					goto ConvertWPToUni;
				}

				bytesOutput += 2;
				break;
				
			case WHITE_SPACE_CODE:
				objLength = 1;

				if( outputData)
				{
					if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto GedGetUNICODE_Output;
					}

					if( c == (WHITE_SPACE_CODE | NATIVE_TAB))
					{
						*puzStrBuf = (FLMUNICODE) 9;
					}
					else if( c == (WHITE_SPACE_CODE | NATIVE_LINEFEED))
					{
						*puzStrBuf = (FLMUNICODE) 10;
					}
					else if( c == (WHITE_SPACE_CODE | HARD_RETURN))
					{
						*puzStrBuf = (FLMUNICODE) 13;
					}
					else
					{
						*puzStrBuf = (FLMUNICODE) 0x20;
					}

					puzStrBuf++;
				}

				bytesOutput += 2;
				break;

			case EXT_CHAR_CODE:
				objLength = 3;
				if( outputData)
				{
					if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto GedGetUNICODE_Output;
					}

					// Convert back from WP to UNICODE

					chrSet = *(pucValue + 1);
					chrVal = *(pucValue + 2);

ConvertWPToUni:

					// Code taken from _WpChWPToUni() in WPTEXT\WPCHU.C
					// There should always be a chrSet value.
					
					if( (chrSet < WP60toUni_MAX) && 
						((tablePtr = WP60toUni[ chrSet ]) != 0 ))
					{
						FLMUNICODE *	pCpxUniStr;
						
						newChrVal = tablePtr[ chrVal];

						if ((newChrVal & WPCH_LOMASK) == 0xF000)
						{
							/*
							**  Does character convert to many Unicode chars?
							**  Yes: Use complex character table
							**       Move to the correct location in the table
							*/

							pCpxUniStr = WP60toCpxUni[chrSet];
							pCpxUniStr += (newChrVal & WPCH_HIMASK) * WPCH_MAX_COMPLEX;

							while( *pCpxUniStr)
							{
								if( outputData)
								{
									if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
									{
										rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
										goto GedGetUNICODE_Output;
									}
									*puzStrBuf++ = *pCpxUniStr++;
								}
								bytesOutput += 2;
							}
						}
						else
						{
							if( outputData)
							{
								if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
								{
									rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
									goto GedGetUNICODE_Output;
								}
								*puzStrBuf++ = newChrVal; 
							}
							bytesOutput += 2;		
						}
					}
					else
					{
						// Big extended WP char
						
						if( outputData)
						{
							if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
							{
								rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
								goto GedGetUNICODE_Output;
							}

							*puzStrBuf++ = 0x03;
						}

						bytesOutput += 2;		
					}
				}
				break;

			case OEM_CODE:

				// We always just skip OEM codes

				objLength = 2;
				break;

			case UNICODE_CODE:
				objLength = 3;
				if( outputData)
				{
					if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto GedGetUNICODE_Output;
					}

					*puzStrBuf++ = (*(pucValue + 1) << 8) + *(pucValue + 2);
				}
				bytesOutput += 2;
				break;
	
			case UNK_EQ_1_CODE:
				objLength = 2;
				if( outputData)
				{
					if( (maxOutLen < 2) || (bytesOutput > maxOutLen - 2))
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto GedGetUNICODE_Output;
					}
					*puzStrBuf++ = *(pucValue+1);
				}
				bytesOutput += 2;
				break;

			default:
				flmAssert(0);
				bytesProcessed = uiValueLength;
				break;
		}
		pucValue += objLength;
		bytesProcessed += objLength;
	}

	// Add TWO terminating NULL characters, but DO NOT increment the
	// bytesOutput counter!

GedGetUNICODE_Output:

	if( outputData)
	{
		*puzStrBuf = 0;
	}

	*puiStrBufLen = bytesOutput;

Exit:

	return( rc);
}
