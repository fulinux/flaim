//-------------------------------------------------------------------------
// Desc:	String collation for Asian languages.
// Tabs:	3
//
//		Copyright (c) 1993-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kyasia1.cpp 12312 2006-01-19 15:14:03 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:  	Convert a text string to a collated string.
****************************************************************************/
RCODE AsiaFlmTextToColStr(
	const FLMBYTE *	Str,					/* Points to the internal TEXT string */
	FLMUINT 				StrLen,				/* Length of the internal TEXT string */
	FLMBYTE *			ColStr,				/* Output collated string */
	FLMUINT *			ColStrLenRV,		/* Collated string length return value */
													/* Input value is MAX num of bytes in buffer*/
	FLMUINT				UppercaseFlag,		/* Set if to convert to uppercase */
	FLMUINT *			puiCollationLen,	/* Returns the collation bytes length */
	FLMUINT *			puiCaseLen,			/* Returns length of case bytes */
	FLMUINT				uiCharLimit,		/* Max number of characters in this key piece */
	FLMBOOL				bFirstSubstring,	/* TRUE is this is the first substring key */
	FLMBOOL *			pbDataTruncated)
{
	RCODE					rc = FERR_OK; 
	const FLMBYTE *	pszStrEnd;				/* Points to the end of the String */
	FLMUINT				Length;					/* Temporary variable for length */
	FLMUINT 				uiTargetColLen = *ColStrLenRV - 12;/* 6=ovhd,6=worst char*/
	FLMBYTE				SubColBuf[ MAX_SUBCOL_BUF+1];	/* Holds Sub-col values (diac) */
	FLMBYTE				LowUpBuf[ MAX_LOWUP_BUF+MAX_LOWUP_BUF+2];	/* 2 case bits/wpchar */
	FLMUINT				ColLen;					/* Return value of collated length */
	FLMUINT				SubColBitPos;			/* Index into Sub-collation buffer */
	FLMUINT 				LowUpBitPos;			/* Lower upper bit position */
	FLMUINT				Flags;					/* Clear all bit flags */
	FLMUINT16			NextWpChar;				/* next WP character */
	FLMUINT16			UnicodeChar;			/* Unicode character */
	FLMUINT16			ColValue;				/* Set to zero for the first time */
	FLMBOOL				bDataTruncated = FALSE;
	
	ColLen = SubColBitPos = LowUpBitPos = Flags = 0;
	UnicodeChar = ColValue = 0;

	// Don't allow any key component to exceed 256 bytes regardless of the
	// user-specified character or byte limit.  The goal is to prevent
	// any single key piece from consuming too much of the key (which is
	// limited to 640 bytes) and thus "starving" other pieces, resulting
	// in a key overflow error.

	if( uiTargetColLen > 256)
	{
		uiTargetColLen = 256;
	}

	// Make sure SubColBuf and LowUpBuf are set to 0's

	f_memset( SubColBuf, 0, sizeof( SubColBuf));
	f_memset( LowUpBuf,  0, sizeof( LowUpBuf));

	pszStrEnd = &Str[ StrLen];
	NextWpChar = 0;

	while( (Str < pszStrEnd) || NextWpChar || UnicodeChar)
	{
		FLMUINT16	WpChar;				/* Current WP character */
		FLMUINT 		ObjLength;
		FLMUINT16	SubColVal;			/* Sub-collated value (diacritic) */
		FLMBYTE		CaseFlags;
		
		// Get the next character from the TEXT String.  NOTE: OEM characters
		// will be returned as character set ZERO, the character will be
		// greater than 127.
		
		WpChar = NextWpChar;

		for( NextWpChar = 0;
			  (!WpChar || !NextWpChar) && !UnicodeChar && (Str < pszStrEnd);
			  Str += ObjLength )	  		/* Inc Str to skip what its pointing at*/
		{
			FLMBYTE		ObjType;
			FLMBYTE		CurByte;
			FLMUINT16	CurWpChar = 0;
			
			CurByte = *Str;
			ObjType = (FLMBYTE)(GedTextObjType( CurByte));
			ObjLength = 1;
			switch( ObjType)
			{
				case ASCII_CHAR_CODE:  			/* 0nnnnnnn */
					CurWpChar = (FLMUINT16)CurByte; /* Character set zero is assumed */
					break;
				case CHAR_SET_CODE:	  			/* 10nnnnnn */
					ObjLength = 2;
					CurWpChar =						
						(FLMUINT16)(((FLMUINT16)(CurByte & (~CHAR_SET_MASK)) << 8)	/* Char set */
						+ (FLMUINT16)*(Str + 1));	/* Character */
					break;
				case WHITE_SPACE_CODE:			/* 110nnnnn */
					CurByte &= (~WHITE_SPACE_MASK);
					CurWpChar = ((CurByte == HARD_HYPHEN) ||
									 (CurByte == HARD_HYPHEN_EOL) ||
									 (CurByte == HARD_HYPHEN_EOP)
									)
									?	0x2D 			/* Minus sign - character set zero*/
									:	0x20;			/* Space -- character set zero */
					break;

				/* Skip all of the unknown stuff */
				case UNK_GT_255_CODE:
					ObjLength = 1 + sizeof( FLMUINT16) + FB2UW( Str + 1);
					break;
				case UNK_LE_255_CODE:
					ObjLength = 2 + (FLMUINT16)*(Str + 1);
					break;
				case UNK_EQ_1_CODE:
					ObjLength = 2;
					break;
				case EXT_CHAR_CODE:
					ObjLength = 3;
					CurWpChar =
						(FLMUINT16)(((FLMUINT16)*(Str + 1) << 8)		/* Character set */
						+ (FLMUINT16)*(Str + 2));			/* Character */
					break;
				case OEM_CODE:						/* Should never get these */
					ObjLength = 2;

					/* OEM characters are always >= 128. */
					/* We use character set zero to process them. */
					CurWpChar = (FLMUINT16)*(Str + 1);
					break;
				case UNICODE_CODE:			/* Unconvertable UNICODE code */
					ObjLength = 3;
					UnicodeChar = (FLMUINT16)(((FLMUINT16)*(Str + 1) << 8)	/* Char set */
									 + (FLMUINT16)*(Str + 2));			/* Character */
					CurWpChar = 0;
					break;

				default:							/* should NEVER happen: bug if does */
					continue;
			}	/* end switch */

			if( !WpChar)						/* Change which needs changing */
				WpChar = CurWpChar;
			else
				NextWpChar = CurWpChar; 
		}	/* end of FOR statement */


		/**-----------------------------------------------------------
		*** If we didn't get a character, break out of the outer
		*** processing loop.
		***----------------------------------------------------------*/

		if( !WpChar && !UnicodeChar)
			break;											/* leave WHILE statement */

		if( WpChar)
		{
			if( fwpAsiaGetCollation( WpChar, NextWpChar, ColValue, &ColValue,
							&SubColVal, &CaseFlags, (FLMUINT16)UppercaseFlag ) == 2)
			{
				/* Took the NextWpChar value */
				NextWpChar = 0;				/* Force to skip this value */
			}
		}
		else // use the UnicodeChar value for this pass
		{
			// This handles all of the UNICODE characters that could not
			// be converted to WP characters - which will include most
			// of the Asian characters.

			CaseFlags = 0;
			if (UnicodeChar < 0x20)
			{
				ColValue = 0xFFFF;

				// Setting SubColVal to a high code will ensure
				// that the code that the UnicodeChar will be stored
				// in its full 16 bits in the sub-collation area.

				SubColVal = 0xFFFF;

				// NOTE: UnicodeChar SHOULD NOT be set to zero here.
				// It will be set to zero below.

			}
			else
			{
				ColValue = UnicodeChar;
				SubColVal = 0;
				UnicodeChar = 0;
			}
		}

		/* Store the values in 2 bytes */
		ColStr[ ColLen++ ] = (FLMBYTE)(ColValue >> 8);
		ColStr[ ColLen++ ] = (FLMBYTE)(ColValue & 0xFF);

		if( SubColVal)
		{
			Flags |= HAD_SUB_COLLATION;
			if( SubColVal <= 31)			/* 5 bit - store bits 10 */
			{
				SET_BIT( SubColBuf, SubColBitPos);
				SubColBitPos += 1 + 1;	/* Stores a zero */
				SETnBITS( 5, SubColBuf, SubColBitPos, SubColVal);
	
				SubColBitPos += 5;
			}	
			else 									/* 2 bytes - store bits 110 or 11110 */
			{
				FLMUINT		 Temp;
				
				SET_BIT( SubColBuf, SubColBitPos);
				SubColBitPos++;
				SET_BIT( SubColBuf, SubColBitPos);
				SubColBitPos++;
				
				if( !WpChar && UnicodeChar)				/* Store as "11110" */
				{
					SubColVal = UnicodeChar;
					UnicodeChar = 0;
					SET_BIT( SubColBuf, SubColBitPos );
					SubColBitPos++;
					SET_BIT( SubColBuf, SubColBitPos );
					SubColBitPos++;
				}
				SubColBitPos++;				/* Skip past the zero */
			
				/* Go to the next byte boundary to write the WP char */
				SubColBitPos = (SubColBitPos + 7) & (~7);
				Temp = BYTES_IN_BITS( SubColBitPos);

				/* Need to store HIGH-Low - PC format is Low-high! */
				SubColBuf[ Temp  ] = (FLMBYTE) (SubColVal >> 8);
				SubColBuf[ Temp+1] = (FLMBYTE) (SubColVal);

				SubColBitPos += 16;
			}
		}
		else
			SubColBitPos++;

		/* Save case information - always 2 bits worth for asian */

		if( CaseFlags & 0x02)
		{
			SET_BIT( LowUpBuf, LowUpBitPos);
		}
		LowUpBitPos++;
		
		if( CaseFlags & 0x01)
		{
			SET_BIT( LowUpBuf, LowUpBitPos);
		}
		LowUpBitPos++;
		
		// Check to see if ColLen is within 1 byte of max
		
		if( (ColLen >= uiCharLimit) ||
			 (ColLen + BYTES_IN_BITS( SubColBitPos) + 
					 BYTES_IN_BITS( LowUpBitPos) >= uiTargetColLen))
		{
			// Still something left?
			
			if ((Str < pszStrEnd) || NextWpChar || UnicodeChar)
			{
				bDataTruncated = TRUE;
			}
			
			// Hit the max. number of characters
			
			break;
		}			
	}

	if( puiCollationLen)
	{
		*puiCollationLen = ColLen;
	}

	// Add the first substring marker - also serves as making the string non-null.
	if( bFirstSubstring)
	{
		ColStr[ ColLen++ ] = 0;
		ColStr[ ColLen++ ] = COLL_FIRST_SUBSTRING;
	}

	if( bDataTruncated)
	{
		ColStr[ ColLen++ ] = 0;
		ColStr[ ColLen++ ] = COLL_TRUNCATED;
	}

	if( !ColLen && !SubColBitPos)
	{
		if( puiCaseLen)
		{
			*puiCaseLen = 0;
		}
		goto Exit;
	}

	// Done putting the String into 3 sections - build the COLLATED KEY

	if( Flags & HAD_SUB_COLLATION)
	{
		ColStr[ ColLen++ ] = 0;
		ColStr[ ColLen++ ] = COLL_MARKER | SC_SUB_COL;
		
		// Move the Sub-collation (diacritics) into the collating String
		
		Length = (FLMUINT)(BYTES_IN_BITS(SubColBitPos));
		f_memcpy( &ColStr[ColLen], SubColBuf, Length);
		ColLen += Length;
	}

	// Always represent the marker as 2 bytes and case bits in asia

	ColStr[ ColLen++ ] = 0;
	ColStr[ ColLen++ ] = COLL_MARKER | SC_MIXED;

	Length = (FLMUINT)(BYTES_IN_BITS( LowUpBitPos ));
	f_memcpy( &ColStr[ ColLen ], LowUpBuf, Length);
	if( puiCaseLen)
	{
		*puiCaseLen = (FLMUINT)(Length + 2);
	}
	ColLen += Length;

Exit:

	if( pbDataTruncated)
	{
		*pbDataTruncated = bDataTruncated;
	}

	*ColStrLenRV = (FLMUINT)ColLen;
	return( rc);
}
