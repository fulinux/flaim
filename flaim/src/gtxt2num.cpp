//-------------------------------------------------------------------------
// Desc:	Convert internal text to internal number
// Tabs:	3
//
//		Copyright (c) 1992-1994,1996-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gtxt2num.cpp 12309 2006-01-19 15:09:04 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:		This routine converts an internal text string to a number.
Ret:		SUCCESS or FERR_CONV_BAD_DIGIT or FERR_CONV_DEST_OVERFLOW.
Notes:	If the buffer pointer is NULL, the routine just determines how
			much buffer space is needed to return the text in number
			format.
****************************************************************************/
RCODE GedTextToNum(
	FLMBYTE *	textStr,		/* Pointer to buffer containing TEXT */
	FLMUINT		textLen,		/* Length of text (in bytes) */
	FLMBYTE *	buffer,		/* Pointer to buffer where number data is to be 
											returned */
	FLMUINT	*	bufLenRV)	/* Return length -- on input contains buffer size */
{
	FLMBYTE *	outPtr;
	FLMBYTE		c;
	FLMUINT		bytesProcessed;
	FLMUINT		bytesOutput;
	FLMUINT		outputData;
	FLMUINT		maxOutLen;
	FLMUINT		objType;
	FLMUINT		objLength;
	FLMBOOL		firstNibble;
	FLMBOOL  	have1Num;
	FLMBOOL  	insideNum;
	FLMBOOL		haveSign;

	maxOutLen = *bufLenRV;
	outputData = ((buffer != NULL) && (maxOutLen));
	bytesProcessed =
	bytesOutput		= 0;
	outPtr = buffer;

	/* Parse through the string outputting data to the buffer */
	/* as we go. */

	haveSign  = 
	have1Num  =
	insideNum = 0;
	firstNibble = 1;
	if( textStr == NULL)
		textLen = 0;

	for( 
		; bytesProcessed < textLen
		; textStr += objLength, bytesProcessed += objLength)
	{

		/* Determine what we are pointing at */

		c = *textStr;
		objType = (FLMBYTE)GedTextObjType( c);		/* Don't put this in if() below */

		if( objType == ASCII_CHAR_CODE)
		{
			objLength = 1;
			if( (c == ASCII_SPACE) || (c == ASCII_TAB) || (c == ASCII_NEWLINE) || (c == ASCII_CR))
			{
				if( insideNum)
					have1Num = 1;
				break;
			}
			/* Code below was a break - now skips leading zeros */
			
			if( (c == ASCII_ZERO) && (!insideNum)) /* Ignore leading zeroes */
				continue;
			
			if( (c >= ASCII_ZERO) && (c <= ASCII_NINE))
			{
				if( !insideNum)
				{
					insideNum = 1;
					haveSign = 1;
				}
				c -= ASCII_ZERO;
			}

			/* Handle sign characters ('+', '-') */

			else if( ((c == ASCII_PLUS) || (c == ASCII_MINUS)) && (!haveSign) && (!insideNum))
			{
				haveSign = 1;
				if( c == ASCII_MINUS)
					c = 0x0B;
			}
			else
				return( RC_SET( FERR_CONV_BAD_DIGIT));

			if( outputData)
			{
				if( (firstNibble) && (bytesOutput == maxOutLen))
					return( RC_SET( FERR_CONV_DEST_OVERFLOW));

				if( firstNibble)
				{
					c <<= 4;
					*outPtr = c;
				}
				else
				{
					*outPtr = (FLMBYTE)(*outPtr + c);
					outPtr++;
				}
			}
			if( firstNibble)
				bytesOutput++;
			firstNibble = !firstNibble;
		
		}
		else switch( objType)
		{
			case WHITE_SPACE_CODE:
				objLength = 1;
				break;

			/* Skip the unkown codes for now */

			case UNK_GT_255_CODE:
				objLength = (1 + sizeof( FLMUINT16) + FB2UW( textStr + 1));
				break;
			case UNK_LE_255_CODE:
				objLength = (2 + (FLMUINT16)*(textStr + 1));
				break;
			case UNK_EQ_1_CODE:
				objLength = 2;
				break;
			case CHAR_SET_CODE:
			case EXT_CHAR_CODE:
			case OEM_CODE:
			case UNICODE_CODE:
			// Should not hit default.
			default:
				return( RC_SET( FERR_CONV_BAD_DIGIT));
		}
	}

 /* Interpret empty number or all zeroes as single zero */

	if( (!insideNum) && (!have1Num))
	{
		if( outputData)
		{
			if( (firstNibble) && (bytesOutput == maxOutLen))
				return( RC_SET( FERR_CONV_DEST_OVERFLOW));
			if( firstNibble)
			{
				*outPtr = 0x00;
			}
			else
				outPtr++;
		}
		if( firstNibble)
			bytesOutput++;
		firstNibble = !firstNibble;
	}

	/* Add Terminator code to the end of the number */

	if( outputData)
	{
		if( (firstNibble) && (bytesOutput == maxOutLen))
			return( RC_SET( FERR_CONV_DEST_OVERFLOW));
		if( firstNibble)
			*outPtr = 0xFF;
		else
			*outPtr = (FLMBYTE)(*outPtr + 0x0F);
	}
	if( firstNibble)
		bytesOutput++;
	*bufLenRV = bytesOutput;
	return( FERR_OK);
}
