//-------------------------------------------------------------------------
// Desc:	Native text conversion routines.
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
// $Id: fnative.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define NON_DISPLAYABLE_CHAR			0xFF

/****************************************************************************
Desc: 	Returns the size of buffer needed to hold the native string in 
			FLAIM's storage format.
****************************************************************************/
FLMUINT FlmGetNativeStorageLength(
	const char *	pszStr)
{
	RCODE				rc;
	FLMUINT			uiStorageLength;

	rc = FlmNative2Storage( pszStr, &uiStorageLength, NULL);
	flmAssert( rc == FERR_OK);

	return uiStorageLength;
}

/****************************************************************************
Desc:		Copies and formats a native 8-bit null terminated string into a 
			caller supplied buffer, It converts the string into an internal FLAIM 
			TEXT string.
****************************************************************************/
RCODE FlmNative2Storage(
	const char *		pszNativeString, 
	FLMUINT *			puiStorageLen,
	FLMBYTE *			pStorageBuffer)
{
	FLMBOOL				bGetLengthPass;
	const char *		ptr;
	FLMBYTE				c;
	FLMUINT				uiLength = 0;

	// Are we determining the needed length or converting the data

	bGetLengthPass = ( pStorageBuffer) ? FALSE : TRUE;

	/* Parse through the string */

	ptr = pszNativeString;
	for( ;;)
	{

		/* Put the character in a local variable for speed */

		c = f_toascii( *ptr);

		/* See if we are at the end of the string */

		if( !c)
		{
			/* We have reached end of the string, return the storage length */

			*puiStorageLen = uiLength;
			return( FERR_OK);
		}
		else
		{
			if( c < ASCII_SPACE)
			{

				/* If it is a tab, carriage return, or linefeed, output */
				/* a whitespace code for indexing and each word purposes */

				if( c == ASCII_TAB)
				{
					if( bGetLengthPass)
						uiLength++;
					else
						*pStorageBuffer++ = WHITE_SPACE_CODE | NATIVE_TAB;
				}
				else if( c == ASCII_NEWLINE)
				{
					if( bGetLengthPass)
						uiLength++;
					else
						*pStorageBuffer++ = WHITE_SPACE_CODE | NATIVE_LINEFEED;
				}
				else if( c == ASCII_CR)
				{
					if( bGetLengthPass)
						uiLength++;
					else
						*pStorageBuffer++ = WHITE_SPACE_CODE | HARD_RETURN;
				}
				else
				{
					if( bGetLengthPass)
					{
							uiLength += 2;
					}		
					else
					{
						/* Output the character as an unknown byte if no WP char found */
						*pStorageBuffer++ = UNK_EQ_1_CODE | NATIVE_TYPE;
						*pStorageBuffer++ = c;
					}
				}
			}
			else if( c < 127)
			{
				/* For now assume < 127 means character set zero.  */
				/* Value 127 is very sacred in WP land and is really an */
				/* extended character */

				if( bGetLengthPass)
					uiLength++;
				else
					*pStorageBuffer++ = c;
			}
			else
			{
				if( bGetLengthPass)
				{
					uiLength += 2;
				}
				else
				{
					*pStorageBuffer++ = OEM_CODE;
					*pStorageBuffer++ = c;
				}
			}
			/* Increment our pointer past the character just handled */

			ptr++;
		}
	}
}

/****************************************************************************
Desc: 	Convert a storage text string into a native string
****************************************************************************/
RCODE FlmStorage2Native(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue, 
	FLMUINT *			puiOutBufLenRV,
	char *				pOutBuffer)
{
	RCODE					rc = FERR_OK;
	const FLMBYTE *	ptr = pucValue;
	char *				outPtr;
	FLMBYTE				c;
	FLMUINT				bytesProcessed;
	FLMUINT				bytesOutput;
	FLMUINT				valLength = uiValueLength;
	FLMUINT				outputData;
	FLMUINT				maxOutLen = 0;
	FLMBYTE				objType;
	FLMUINT				objLength = 0;
	FLMBYTE				TempBuf[ 80];
	FLMUINT				length;

	/* If the input is not a TEXT or a NUMBER node, return an error for now. */

	if( (uiValueType == FLM_BINARY_TYPE) || (uiValueType == FLM_CONTEXT_TYPE))
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	/* If the node is a number, convert to text first */

	if( uiValueType != FLM_TEXT_TYPE)
	{
		if( ptr == NULL)
		{
			valLength = 0;
		}
		else
		{
			valLength = sizeof( TempBuf);
			flmAssert( uiValueType == FLM_NUMBER_TYPE);
			if (RC_BAD( rc = GedNumToText( ptr, TempBuf, &valLength)))
			{
				goto Exit;
			}
			ptr = &TempBuf[ 0];
		}
	}

	outputData = ((pOutBuffer != NULL) && (*puiOutBufLenRV));
	if( outputData)
	{
		maxOutLen = *puiOutBufLenRV - 1;
	}
	
	bytesProcessed = 0;
	bytesOutput = 0;
	outPtr = pOutBuffer;

	while( bytesProcessed < valLength)
	{
		c = *ptr;
		objType = (FLMBYTE)GedTextObjType( c);
		
		switch( objType)
		{
			case ASCII_CHAR_CODE:
				objLength = 1;
				if( outputData)
				{
					if( bytesOutput < maxOutLen)
						*outPtr++ = f_tonative( c);
					else
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				bytesOutput++;
				break;
			case CHAR_SET_CODE:
				objLength = 2;
				if( outputData)
				{
					if( bytesOutput < maxOutLen)
					{
						/**-----------------------------------------
						***  Can only convert to native if the char
						***  has been stored as a extended NATIVE.
						***  FLM Doesn't support code pages
						***  or alt WP to native mappings at all!
						***  OLD CODE BELOW...
						***    wpchr = (((FLMUINT16) (c & 0x3F)) << 8)+  CharSet
						***             *(ptr + 1);						 CharVal
						***    WpCh6Getnative( wpchr, outPtr);
						***    outPtr++;
						***----------------------------------------*/
						
						if( (c & (~objType)) == 0)
							*outPtr++ = f_tonative( *(ptr + 1));
						else
							*outPtr++ = NON_DISPLAYABLE_CHAR;
					}
					else
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				bytesOutput++;
				break;
			case WHITE_SPACE_CODE:
				objLength = 1;

				/* ALWAYS OUTPUT A SPACE WHEN WE SEE A WHITE_SPACE_CODE */
				/* UNLESS IT IS A HYPHEN */

				if( outputData)
				{
					if( bytesOutput < maxOutLen)
					{
						c &= (~WHITE_SPACE_MASK);
						if(	(c == HARD_HYPHEN) ||
								(c == HARD_HYPHEN_EOL) ||
								(c == HARD_HYPHEN_EOP)
						)
							c = ASCII_DASH;					/* Minus sign -- character set zero */
						else if( c == NATIVE_TAB)
							c = ASCII_TAB;
						else if( c == NATIVE_LINEFEED)
							c = ASCII_NEWLINE;
						else if( c == HARD_RETURN)
							c = ASCII_CR;
						else
							c = ASCII_SPACE;					/* Space */
						*outPtr++ = f_tonative( c);
					}
					else
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				bytesOutput++;
				break;
			case UNK_GT_255_CODE:
			case UNK_LE_255_CODE:
				if( objType == UNK_GT_255_CODE)
				{
					length = FB2UW( ptr + 1);
					objLength = (FLMUINT16)(1 + sizeof( FLMUINT16) + length);
				}
				else
				{
					length = (FLMUINT16)*(ptr + 1);
					objLength = (FLMUINT16)(2 + length);
				}

				/* Skip it if it is not a NATIVE code */

				if( (c & (~objType)) == NATIVE_TYPE)
				{
					if( outputData)
					{
						if( (maxOutLen < length) || (bytesOutput > maxOutLen - length))
						{
							rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
							goto Native_Output;
						}
						if( objType == UNK_LE_255_CODE)
							f_memcpy( outPtr, ptr + 2, length);
						else
							f_memcpy( outPtr, ptr + 1 + sizeof( FLMUINT16), length);
						outPtr += length;
					}
					bytesOutput += length;
				}
				break;
			case UNK_EQ_1_CODE:
				objLength = 2;

				/* Skip it if it is not a NATIVE code */

				if( (c & (~objType)) == NATIVE_TYPE)
				{
					if( outputData)
					{
						if( bytesOutput < maxOutLen)
							*outPtr++ = f_tonative( *(ptr + 1));
						else
						{
							rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
							goto Native_Output;
						}
					}
					bytesOutput++;
				}
				break;
			case EXT_CHAR_CODE:
				objLength = 3;
				if( outputData)
				{
					if( bytesOutput < maxOutLen)
					{
						/**-----------------------------------------
						***  Can no longer convert to native
						***  because toolkit will not support
						***  code pages or alt WP->native mappings
						***  OLD CODE BELOW...
						***    wpchr = (((FLMUINT16) (*(ptr+1))) << 8) +  ** Character set
						***             *(ptr + 2);										 ** Character
						***    WpCh6Getnative( wpchr, outPtr);
						***    outPtr++;
						***----------------------------------------*/

						*outPtr += NON_DISPLAYABLE_CHAR;
					}
					else
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				bytesOutput++;
				break;

			case OEM_CODE:
				objLength = 2;
				if( outputData)
				{
					/**  This code takes the original OEM 8-bit code.
					***  This bothers me (Scott) a bit (good pun!) because the
					***  character could have come from a different code page.
					***  In the far future, 8-bit codes could be remembered
					***  with the original code page that was used to create
					***  the text.  This way the database could have good
					***  storage and recall of the information.  All databases
					***  share this common problem and none solve it well except
					***  for Lotus Notes which have their own character set like
					***  WP.
					**/

					if( bytesOutput < maxOutLen)
						*outPtr++ = f_tonative( *(ptr + 1));
					else
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				bytesOutput++;
				break;


			case UNICODE_CODE:
				objLength = 3;
				if( outputData)
				{
					if( bytesOutput < maxOutLen )
						*outPtr++ = UNICODE_UNCONVERTABLE_CHAR;
					else
					{
						rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
						goto Native_Output;
					}
				}
				bytesOutput++;
				break;

			default:

				/* These codes should NEVER HAPPEN -- bug if we get here */

				break;
		}
		ptr += objLength;
		bytesProcessed += objLength;
	}

	/* Add a terminating NULL character, but DO NOT increment the */
	/* bytesOutput counter! */

Native_Output:
	if( outputData)
		*outPtr = 0;
	*puiOutBufLenRV = bytesOutput;

Exit:
	return( rc);
}
