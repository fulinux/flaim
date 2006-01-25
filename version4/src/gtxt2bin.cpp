//-------------------------------------------------------------------------
// Desc:	Convert internal text to binary.
// Tabs:	3
//
//		Copyright (c) 1992-1993,1996-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gtxt2bin.cpp 12309 2006-01-19 15:09:04 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	This routine converts an internal text string to binary.  Every
		two hex digits becomes one byte of binary data.
Ret:	SUCCESS or FERR_CONV_BAD_DIGIT or FERR_CONV_DEST_OVERFLOW.
Notes:	If the buffer pointer is NULL, the routine just determines how
			much buffer space is needed to return the text in binary
			format.
****************************************************************************/
RCODE GedTextToBin(
	const FLMBYTE *	textStr,
	FLMUINT				textLen,
	FLMBYTE *			buffer,
	FLMUINT *			bufLenRV)
{
	FLMBYTE *	outPtr;
	FLMBYTE		c;
	FLMUINT		bytesProcessed;
	FLMUINT		bytesOutput;
	FLMUINT		outputData;
	FLMUINT		maxOutLen;
	FLMBYTE		objType;
	FLMUINT		objLength;
	FLMBYTE		firstNibble;

	maxOutLen = *bufLenRV;
	outputData = ((buffer != NULL) && (maxOutLen));
	bytesProcessed = 0;
	bytesOutput = 0;
	outPtr = buffer;

	/* Parse through the string outputting data to the buffer */
	/* as we go. */

	firstNibble = 1;
	if( textStr == NULL)
		textLen = 0;
	while( bytesProcessed < textLen)
	{
		/* Determine what we are pointing at */

		c = *textStr;
		objType = (FLMBYTE)GedTextObjType( c);
		switch( objType)
		{
			case ASCII_CHAR_CODE:
				objLength = 1;
				if( ((c >= ASCII_ZERO) && (c <= ASCII_NINE)) ||
						((c >= ASCII_UPPER_A) && (c <= ASCII_UPPER_F)) ||
						((c >= ASCII_LOWER_A) && (c <= ASCII_LOWER_F)))
				{
					if( outputData)
					{
						if( (firstNibble) && (bytesOutput == maxOutLen))
							return( RC_SET( FERR_CONV_DEST_OVERFLOW));

						/* Convert character to its binary nibble */

						if( (c >= ASCII_ZERO) && (c <= ASCII_NINE))
							c -= ASCII_ZERO;
						else if( (c >= ASCII_UPPER_A) && (c <= ASCII_UPPER_F))
							c = (FLMBYTE)(c - ASCII_UPPER_A + 10);
						else
							c = (FLMBYTE)(c - ASCII_LOWER_A + 10);
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
				else
				{
					/* White space can be ignored */

					if( (c != ASCII_SPACE) && (c != ASCII_TAB) && (c != ASCII_NEWLINE) && (c != ASCII_CR))
						return( RC_SET( FERR_CONV_BAD_DIGIT));
				}
				break;
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
			// default shouldn't happen - if it does just bail.
			default: 
				return( RC_SET( FERR_CONV_BAD_DIGIT));
		}
		textStr += objLength;
		bytesProcessed += objLength;
	}

	*bufLenRV = bytesOutput;
	return( FERR_OK);
}
