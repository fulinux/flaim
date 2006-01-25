//-------------------------------------------------------------------------
// Desc:	Convert internal number to internal text format.
// Tabs:	3
//
//		Copyright (c) 1992-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gnum2txt.cpp 12309 2006-01-19 15:09:04 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:		This routine converts an internal number to an internal (ASCII) text.
			Support for FT_NUMBER and FT_REAL numeric types.
****************************************************************************/
RCODE GedNumToText(
	const FLMBYTE *	num,
	FLMBYTE *			buffer,
	FLMUINT 	*			bufLenRV)
{
	FLMBYTE *			outPtr;
	FLMBYTE				c;
	FLMBYTE				c1 = 0;
	FLMUINT				bytesOutput;
	FLMUINT				outputData;
	FLMUINT				maxOutLen;
	FLMBYTE				done;
	FLMBYTE				firstNibble;
	FLMBYTE				lastChar;
	const FLMBYTE *	pExp = NULL;
	FLMBYTE				parseExponent = 0;
	FLMBYTE				firstNibbleAtExp = 0;
	FLMBYTE				firstDigit = 0;

	maxOutLen = *bufLenRV;
	outputData = ((buffer != NULL) && (maxOutLen));
	bytesOutput = 0;
	outPtr = buffer;

	/* Parse through the string outputting data to the buffer */
	/* as we go. */

	done = (num == NULL);					/* Sets to TRUE if NULL else FALSE	*/
	firstNibble = 1;
	lastChar = 0xFF;

	while( !done)
	{
continue_loop:

		if( firstNibble)						/* Rather not do a ? : here because */
		{											/* of the num++ in the : portion		*/
			c = (FLMBYTE)(*num >> 4);
		}
		else
		{
			c  = (FLMBYTE)(*num++ & 0x0F);
		}
		firstNibble = !firstNibble;

		if( c <= 9)								/* Check common case before switch */
		{
			if( parseExponent)			 	/* Exponent number? */
				firstDigit++;
			c1 = (FLMBYTE)( ASCII_ZERO + c);		/* Normal decimal value */
		}
		else switch( c)
		{
			case 0x0A:
				c1 = ASCII_DOT;
				break;
			case 0x0B:
				c1 = ASCII_DASH;
				break;
			case 0x0C:  /* Ignore for now - imaginary numbers not implemented */
				c1 = 0;							/* Set c1 to zero if no output */
				break;

			case 0x0D:
				c1 = ASCII_SLASH;
				break;
			case 0x0E:
				/* For real numbers the exponent appears first */
				/* This was done to make it easier for building keys */
				if( !parseExponent)
				{
					parseExponent++;			/* 1=need to output 1st digit */
					pExp = num;					/* Set state to reparse exponent */
					if( firstNibble)			/* If set to one */
						pExp--;
					firstNibbleAtExp = (FLMBYTE)(firstNibble ^ 1);

					/* Parse to the end of the exponent area - most 5 nibbles */
					for (;;)
					{
						if( firstNibble)
						{
							if( (*num >> 4) == 0x0F)
								break;
						}
						else
						{
							if( (*num++ & 0x0F) == 0x0F)
								break;
						}
						firstNibble = !firstNibble;
					}
					firstNibble = !firstNibble;	/* Don't forget this! */
					goto continue_loop;		/* 'continue' is vauge - use a goto */
				}
				else
				{
					c1 = ASCII_UPPER_E;
					parseExponent = 0;		/* Clear flag */
				}
				break;
			case 0x0F:
				c1 = 0;							/* Set c1 to zero if no output */
				if( !parseExponent)			/* Done if no exponent or done /w exp*/
					done = TRUE;
				break;
			/* default checked before switch */
		/*	default:
		**		c1 = ASCII_ZERO + c;
		**		break;
		*/
		}

		/* If we got a character, put into output buffer (or just count) */

		if( c1)
		{
			/* If the last character was an exponent and the current */
			/* character is not a minus sign, insert a plus (+) */

			if( (lastChar == ASCII_UPPER_E) && (c1 != ASCII_MINUS))
			{
				if( outputData)
				{
					if( bytesOutput < maxOutLen)
						*outPtr++ = ASCII_PLUS;
					else
						return( RC_SET( FERR_CONV_DEST_OVERFLOW));
				}
				bytesOutput++;
			}
			if( outputData)
			{
				if( bytesOutput < maxOutLen)
					*outPtr++ = c1;
				else
					return( RC_SET( FERR_CONV_DEST_OVERFLOW));
			}
			bytesOutput++;
			/* If exponent (real) number output decimal place */
			if( firstDigit == 1)
			{
				firstDigit++;						/* Set to != 1 */
				if( outputData)
				{
					if( bytesOutput < maxOutLen)
						*outPtr++ = ASCII_DOT;
					else
						return( RC_SET( FERR_CONV_DEST_OVERFLOW));
				}
				bytesOutput++;
			}
			lastChar = c1;
		}
		else if( parseExponent)					/* Hit last trailing 'F' in num */
		{
			num = pExp;								/* Restore state */
			firstNibble = firstNibbleAtExp;
			/* Go again parsing the exponent */
		}
	}
	*bufLenRV = bytesOutput;
	return( FERR_OK);
}
