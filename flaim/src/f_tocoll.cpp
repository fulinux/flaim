//-------------------------------------------------------------------------
// Desc:	Collation routines for indexing.
// Tabs:	3
//
//		Copyright (c) 1991-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: f_tocoll.cpp 12245 2006-01-19 14:29:51 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

// Returns TRUE if upper case, FALSE if lower case.

FINLINE FLMBOOL charIsUpper(
	FLMUINT16	ui16Char)
{
	return( (FLMBOOL)((ui16Char < 0x7F)
							? (FLMBOOL)((ui16Char >= ASCII_LOWER_A &&
											 ui16Char <= ASCII_LOWER_Z)
											 ? (FLMBOOL)FALSE
											 : (FLMBOOL)TRUE)
							: fwpIsUpper( ui16Char)));
}

extern FLMBYTE		fwp_dia60Tbl[];				// Diacritic conversions	
extern FLMBYTE		fwp_alefSubColTbl[];			// Arabic sub collation table.
extern FLMBYTE		fwp_ar2BitTbl[];				// Arabic 2 bit table.

/****************************************************************************
Desc:  	Convert a text string to a collated string.
			If FERR_CONV_DEST_OVERFLOW is returned the string is truncated as
			best as it can be.  The caller must decide to return the error up
			or deal with the truncation.
Return:	RCODE = SUCCESS or FERR_CONV_DEST_OVERFLOW
VISIT:	If the string is EXACTLY the length of the truncation 
			length then it should, but doesn't, set the truncation flag.  
			The code didn't match the design intent.  Fix next major
			version.
****************************************************************************/
RCODE FTextToColStr(
	const FLMBYTE *	pucStr,					// Points to the internal TEXT string
	FLMUINT 				uiStrLen,				// Length of the internal TEXT string
	FLMBYTE *			pucCollatedStr,		// Returns collated string
	FLMUINT *			puiCollatedStrLen,	// Returns total collated string length
														// Input is maximum bytes in buffer
	FLMUINT  			uiUppercaseFlag,		// Set if to convert to uppercase
	FLMUINT *			puiCollationLen,		// Returns the collation bytes length
	FLMUINT *			puiCaseLen,				// Returns length of case bytes
	FLMUINT				uiLanguage,				// Language
	FLMUINT				uiCharLimit,			// Max number of characters in this key piece
	FLMBOOL				bFirstSubstring,		// TRUE is this is the first substring key
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL *			pbDataTruncated)
{
	RCODE					rc = FERR_OK;
	const FLMBYTE *	pucStrEnd;				// Points to the end of the string
	FLMUINT16			ui16Base;				// Value of the base character
	FLMUINT16			ui16SubColVal;			// Sub-collated value (diacritic)
	FLMUINT 				uiObjLength = 0;
	FLMUINT 				uiLength;				// Temporary variable for length
	FLMUINT 				uiTargetColLen = *puiCollatedStrLen - 8;	// 4=ovhd,4=worse char
	FLMUINT				uiObjType;
	FLMBOOL				bDataTruncated = FALSE;

	// Need to increase the buffer sizes to not overflow.
	// Characaters without COLL values will take up 3 bytes in
	// the ucSubColBuf[] and easily overflow the buffer.
	// Hard coded the values so as to minimize changes.

	FLMBYTE		ucSubColBuf[ MAX_SUBCOL_BUF + 301];	// Holds sub-collated values(diac)
	FLMBYTE		ucCaseBits[ MAX_LOWUP_BUF + 81];		// Holds case bits
	FLMUINT16	ui16WpChr;			// Current WP character
	FLMUNICODE	unichr = 0;			// Current unconverted Unicode character
	FLMUINT16	ui16WpChr2;			// 2nd character if any; default 0 for US lang
	FLMUINT		uiColLen;			// Return value of collated length
	FLMUINT		uiSubColBitPos;	// Sub-collation bit position
	FLMUINT	 	uiCaseBitPos;		// Case bit position
	FLMUINT		uiFlags;				// Clear all bit flags
	FLMBOOL		bHebrewArabic = FALSE;	// Set if language is hebrew, arabic, farsi
	FLMBOOL		bTwoIntoOne;

	uiColLen = 0;
	uiSubColBitPos = 0;
	uiCaseBitPos = 0;
	uiFlags = 0;
	ui16WpChr2 = 0;

	// Don't allow any key component to exceed 256 bytes regardless of the
	// user-specified character or byte limit.  The goal is to prevent
	// any single key piece from consuming too much of the key (which is
	// limited to 640 bytes) and thus "starving" other pieces, resulting
	// in a key overflow error.

	if( uiTargetColLen > 256)
	{
		uiTargetColLen = 256;
	}

	// Code below sets ucSubColBuf[] and ucCaseBits[] values to zero.

	if (uiLanguage != US_LANG)
	{
		if (uiLanguage == AR_LANG ||		// Arabic
			 uiLanguage == FA_LANG ||		// Farsi - persian
			 uiLanguage == HE_LANG ||		// Hebrew
			 uiLanguage == UR_LANG)			// Urdu
		{
			bHebrewArabic = TRUE;
		}
	}
	pucStrEnd = &pucStr [uiStrLen];

	while (pucStr < pucStrEnd)
	{

		// Set the case bits and sub-collation bits to zero when
		// on the first bit of the byte.

		if (!(uiCaseBitPos & 0x07))
		{
			ucCaseBits [uiCaseBitPos >> 3] = 0;
		}
		if (!(uiSubColBitPos & 0x07))
		{
			ucSubColBuf [uiSubColBitPos >> 3] = 0;
		}

		// Get the next character from the TEXT string.

		for (ui16WpChr = ui16SubColVal = 0;	// Default sub-collation value
			  !ui16WpChr && pucStr < pucStrEnd;
			  pucStr += uiObjLength)
		{
			FLMBYTE	ucChar = *pucStr;
				
			uiObjType = GedTextObjType( ucChar);
			switch (uiObjType)
			{
				case ASCII_CHAR_CODE:  			// 0nnnnnnn
					uiObjLength = 1;

					// Character set zero is assumed.

					ui16WpChr = (FLMUINT16)ucChar;
					continue;
				case CHAR_SET_CODE:	  			// 10nnnnnn
					uiObjLength = 2;

					// Character set followed by character

					ui16WpChr = (((FLMUINT16)(ucChar & (~CHAR_SET_MASK)) << 8)
							+ (FLMUINT16)*(pucStr + 1));
					continue;
				case WHITE_SPACE_CODE:			// 110nnnnn
					uiObjLength = 1;
					ucChar &= (~WHITE_SPACE_MASK);
					ui16WpChr = (ucChar == HARD_HYPHEN ||
									 ucChar == HARD_HYPHEN_EOL ||
									 ucChar == HARD_HYPHEN_EOP)
									? (FLMUINT16)0x2D	// Minus sign -- character set 0
									: (FLMUINT16)0x20;// Space -- character set zero
					continue;

				// Skip all of the unknown stuff

				case UNK_GT_255_CODE:
					uiObjLength = 3 + FB2UW( pucStr + 1);
					continue;
				case UNK_LE_255_CODE:
					uiObjLength = 2 + (FLMUINT16)*(pucStr + 1);
					continue;
				case UNK_EQ_1_CODE:
					uiObjLength = 2;
					continue;
				case EXT_CHAR_CODE:
					uiObjLength = 3;

					// Character set followed by character

					ui16WpChr = (((FLMUINT16)*(pucStr + 1) << 8)
							  + (FLMUINT16)*(pucStr + 2));
					continue;
				case OEM_CODE:

					// OEM characters are always >= 128
					// Use character set zero to process them.

					uiObjLength = 2;
					ui16WpChr = (FLMUINT16)*(pucStr + 1);
					continue;
				case UNICODE_CODE:			// Unconvertable UNICODE code
					uiObjLength = 3;

					// Unicode character followed by unicode character set

					unichr = (FLMUINT16)(((FLMUINT16)*(pucStr + 1) << 8)
								+ (FLMUINT16)*(pucStr + 2));
					ui16WpChr = UNK_UNICODE_CODE;
					continue;
				default:

					// Should not happen, but don't return an error

					flmAssert( 0);
					continue;
			}
		}

		// If we didn't get a character, break out of while loop.

		if (!ui16WpChr)
		{
			break;
		}

		// fwpCheckDoubleCollation modifies ui16WpChr if a digraph or a double
		// character sequence is found.  If a double character is found, pucStr
		// is incremented and ui16WpChr2 is set to 1.  If a digraph is found,
		// pucStr is not changed, but ui16WpChr contains the first character and
		// ui16WpChr2 contains the second character of the digraph.

		if (uiLanguage != US_LANG)
		{
			ui16WpChr2 = fwpCheckDoubleCollation( &ui16WpChr, &bTwoIntoOne,
											&pucStr, uiLanguage);
		}

		// Save the case bit

		if (!uiUppercaseFlag)
		{

			// charIsUpper returns TRUE if upper case, 0 if lower case.

			if (!charIsUpper( ui16WpChr))
			{
				uiFlags |= HAD_LOWER_CASE;
			}
			else
			{

				// Set if upper case.

				SET_BIT( ucCaseBits, uiCaseBitPos);
			}
			uiCaseBitPos++;
		}

		// Handle OEM characters, non-collating characters,
		// characters with subcollating values, double collating
		// values.

		// Get the collated value from the WP character-if not collating value

		if ((pucCollatedStr[ uiColLen++] =
				(FLMBYTE)(fwpGetCollation( ui16WpChr, uiLanguage))) >= COLS11)
		{
			FLMUINT	uiTemp;

			// Save OEM characters just like non-collating characters

			// If lower case, convert to upper case.

			if (!charIsUpper( ui16WpChr))
			{
				ui16WpChr &= ~1;
			}

			// No collating value given for this WP char.
			// Save original WP char (2 bytes) in subcollating
			// buffer.

			// 1110 is a new code that will store an insert over
			// the character OR a non-convertable unicode character.
			// Store with the same alignment as "store_extended_char"
			// below.

			// 11110 is code for unmappable UNICODE value.
			// A value 0xFE will be the collation value.  The sub-collation
			// value will be 0xFFFF followed by the UNICODE value.
			// Be sure to eat an extra case bit.

			// See specific Hebrew and Arabic comments in the
			//	switch statement below.

			// Set the next byte that follows in the sub collation buffer.

			ucSubColBuf [(uiSubColBitPos + 8) >> 3] = 0;

			if (bHebrewArabic && (pucCollatedStr [uiColLen-1] == COLS0_ARABIC))
			{

				// Store first bit of 1110, fall through & store remaining 3 bits

				SET_BIT( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;

				// Don't store collation value

				uiColLen--;
			}
			else if (unichr)
			{
				ui16WpChr = unichr;
				unichr = 0;

				// Store 11 out of 11110

				SET_BIT( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				SET_BIT( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				if (!uiUppercaseFlag)
				{
					ucCaseBits [(uiCaseBitPos + 7) >> 3] = 0;

					// Set upper case bit.

					SET_BIT( ucCaseBits, uiCaseBitPos);
					uiCaseBitPos++;
				}
			}				
store_extended_char:

			// Set the next byte that follows in the sub collation buffer.

			ucSubColBuf [(uiSubColBitPos + 8) >> 3] = 0;
			ucSubColBuf [(uiSubColBitPos + 16) >> 3] = 0;
			uiFlags |= HAD_SUB_COLLATION;

			// Set 110 bits in sub-collation - continued from above.
			// No need to explicitly set the zero, but must increment
			// for it.

			SET_BIT( ucSubColBuf, uiSubColBitPos);
			uiSubColBitPos++;
			SET_BIT( ucSubColBuf, uiSubColBitPos);
			uiSubColBitPos += 2;

			// store_aligned_word: This label is not referenced.
			// Go to the next byte boundary to write the character.

			uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
			uiTemp = BYTES_IN_BITS( uiSubColBitPos);

			// Need to big-endian - so it will sort correctly.

			ucSubColBuf [uiTemp] = (FLMBYTE)(ui16WpChr >> 8);
			ucSubColBuf [uiTemp + 1] = (FLMBYTE)(ui16WpChr);
			uiSubColBitPos += 16;
			ucSubColBuf [uiSubColBitPos >> 3] = 0;
		}
		else
		{
			// Had a collation value

			// Add the lower/uppercase bit if a mixed case output.

			// If not lower ASCII set - check diacritic value for sub-collation

			if (!(ui16WpChr & 0xFF00))
			{

				// ASCII character set - set a single 0 bit - just need to
				// increment to do this.

				uiSubColBitPos++;
			}
			else
			{
				FLMBYTE	ucTmpChar = (FLMBYTE)ui16WpChr;
				FLMBYTE	ucCharSet = (FLMBYTE)(ui16WpChr >> 8);

				// Convert char to uppercase because case information
				// is stored above.  This will help
				// ensure that the "ETA" doesn't sort before "eta"

				if (!charIsUpper(ui16WpChr))
				{
					ui16WpChr &= ~1;
				}

				switch (ucCharSet)
				{
					case CHSMUL1:	// Multinational 1

						// If we cannot break down a char into base and
						// diacritic we cannot combine the charaacter
						// later when converting back the key.  In that case,
						// write the entire WP char in the sub-collation area.

						if (fwpCh6Brkcar( ui16WpChr, &ui16Base, &ui16SubColVal))
						{
							goto store_extended_char;
						}

						// Write the FLAIM diacritic sub-collation value.
						// Prefix is 2 bits "10".  Remember to leave
						// "111" alone for the future.
						// NOTE: The "unlaut" character must sort after the "ring"
						// character.

						ui16SubColVal = ((ui16SubColVal & 0xFF) == umlaut	&&
											  (uiLanguage == SU_LANG || 
												uiLanguage == SV_LANG || 
												uiLanguage == CZ_LANG || 
												uiLanguage == SL_LANG))
							?	(FLMUINT16)(fwp_dia60Tbl[ ring] + 1)
							:	(FLMUINT16)(fwp_dia60Tbl[ ui16SubColVal & 0xFF]);
				
store_sub_col:
						// Set the next byte that follows in the sub collation buffer.

						ucSubColBuf[ (uiSubColBitPos + 8) >> 3] = 0;
						uiFlags |= HAD_SUB_COLLATION;

						// Set the 10 bits - no need to explicitly set the zero, but
						// must increment for it.

						SET_BIT( ucSubColBuf, uiSubColBitPos);
						uiSubColBitPos += 2;

						// Set sub-collation bits.

						SETnBITS( 5, ucSubColBuf, uiSubColBitPos, ui16SubColVal);
						uiSubColBitPos += 5;
						break;
						
					case CHSGREK:		// Greek

						if (ucTmpChar >= 52  ||		// Keep case bit for 52-69 else ignore
          				 ui16WpChr == 0x804 ||	// [ 8,4] BETA Medial | Terminal
							 ui16WpChr == 0x826) 	// [ 8,38] SIGMA terminal
						{
							goto store_extended_char;
						}
							
						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
						
					case CHSCYR:
						if (ucTmpChar >= 144)
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;

						// VISIT: Georgian covers 208-249 - no collation defined yet

						break;
						
					case CHSHEB:		// Hebrew

						// Three sections in Hebrew:
						//		0..26 - main characters
						//		27..83 - accents that apear over previous character
						//		84..118- dagesh (ancient) hebrew with accents

						// Because the ancient is only used for sayings & scriptures
						// we will support a collation value and in the sub-collation
						// store the actual character because sub-collation is in 
						// character order.

            		if (ucTmpChar >= 84)		// Save ancient - value 84 and above
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
						
					case CHSARB1:		// Arabic 1

						// Three sections in Arabic:						
						//		00..37  - accents that display OVER a previous character
						//		38..46  - symbols 
						//		47..57  - numbers
						//		58..163 - characters
						//		164     - hamzah accent
						//		165..180- common characters with accents
						//		181..193- ligatures - common character combinations
						//		194..195- extensions - throw away when sorting
					
						if (ucTmpChar <= 46)
						{
							goto store_extended_char;	// save original character
						}

						if (pucCollatedStr[ uiColLen-1] == COLS10a+1)	// Alef?
						{	
							ui16SubColVal = (ucTmpChar >= 165)
								? (FLMUINT16)(fwp_alefSubColTbl[ ucTmpChar - 165 ])
								: (FLMUINT16)7;			// Alef subcol value
							goto store_sub_col;
						}
						if (ucTmpChar >= 181)			// Ligatures - char combination
						{
							goto store_extended_char;	// save original character
						}

						if (ucTmpChar == 64)				// taa exception
						{
							ui16SubColVal = 8;
							goto store_sub_col;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
						
					case CHSARB2:			// Arabic 2

						// There are some characters that share the same slot
						// Check the bit table if above character 64
						
						if (ucTmpChar >= 64 &&
							 fwp_ar2BitTbl[(ucTmpChar - 64) >> 3] & 
								(0x80 >> (ucTmpChar & 0x07)))
						{
							goto store_extended_char;	// Will save original
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;

					default:

						// Increment bit position to set a zero bit.

						uiSubColBitPos++;
						break;
				}
			}


			// Now let's worry about double character sorting

			if (ui16WpChr2)
			{
				if (pbOriginalCharsLost)
				{
					*pbOriginalCharsLost = TRUE;
				}

				// Set the next byte that follows in the sub collation buffer.

				ucSubColBuf [(uiSubColBitPos + 7) >> 3] = 0;

				if (bTwoIntoOne)
				{
					
					// Sorts after character in ui16WpChr after call to
					// fwpCheckDoubleCollation
					// Write the char 2 times so lower/upper bits are correct.
					// Could write infinite times because of collation rules.

					pucCollatedStr[ uiColLen] = ++pucCollatedStr[ uiColLen-1];
					uiColLen++;

					// If original was upper case, set one more upper case bit

					if (!uiUppercaseFlag)
					{
						ucCaseBits[ (uiCaseBitPos + 7) >> 3] = 0;
						if (!charIsUpper( (FLMUINT16) *(pucStr - 1)))
						{
							uiFlags |= HAD_LOWER_CASE;
						}
						else
						{
							SET_BIT( ucCaseBits, uiCaseBitPos);
						}
						uiCaseBitPos++;
					}

					// Take into account the diacritical space

					uiSubColBitPos++;
				}
				else
				{

					// We have a digraph, get second collation value

					pucCollatedStr[ uiColLen++] =
						(FLMBYTE)(fwpGetCollation( ui16WpChr2, uiLanguage));

					// Normal case, assume no diacritics set

					uiSubColBitPos++;

					// If first was upper, set one more upper bit.

					if (!uiUppercaseFlag)
					{
						ucCaseBits [(uiCaseBitPos + 7) >> 3] = 0;
						if (charIsUpper( ui16WpChr))
						{
							SET_BIT( ucCaseBits, uiCaseBitPos);
						}
						uiCaseBitPos++;

						// no need to reset the uiFlags
					}
				}
			}
		}

		// Check to see if uiColLen is at some overflow limit.

		if (uiColLen >= uiCharLimit ||
			 uiColLen + BYTES_IN_BITS( uiSubColBitPos) + 
						  BYTES_IN_BITS( uiCaseBitPos) >= uiTargetColLen)
		{

			// We hit the maximum number of characters.

			if (pucStr < pucStrEnd)
			{
				bDataTruncated = TRUE;
			}
			break;
		}
	}

	// END OF WHILE LOOP

	if (puiCollationLen)
	{
		*puiCollationLen = uiColLen;
	}

	// Add the first substring marker - also serves as making the string non-null.

	if (bFirstSubstring)
	{
		pucCollatedStr [uiColLen++] = COLL_FIRST_SUBSTRING;
	}

	if (bDataTruncated)
	{
		pucCollatedStr[ uiColLen++ ] = COLL_TRUNCATED;
	}

	// 10/20/98 - Add code to return NOTHING if no values found.

	if (!uiColLen && !uiSubColBitPos)
	{
		if (puiCaseLen)
		{
			*puiCaseLen = 0;
		}
		goto Exit;
	}

	// Store extra zero bit in the sub-collation area for Hebrew/Arabic

	if (bHebrewArabic)
	{
		uiSubColBitPos++;
	}

	// Done putting the string into 4 sections - build the COLLATED KEY
	// Don't set uiUppercaseFlag earlier than here because SC_LOWER may be zero

	uiUppercaseFlag = (uiLanguage == GR_LANG) ? SC_LOWER : SC_UPPER;

	// The default terminating characters is (COLL_MARKER|SC_UPPER)
	// Did we write anything to the subcollation area?

	if (uiFlags & HAD_SUB_COLLATION)
	{
		// Writes out a 0x7

		pucCollatedStr [uiColLen++] = COLL_MARKER | SC_SUB_COL;

		// Move the sub-collation into the collating string

		uiLength = BYTES_IN_BITS( uiSubColBitPos);
		f_memcpy( &pucCollatedStr[uiColLen], ucSubColBuf, uiLength);
		uiColLen += uiLength;
	}

	// Move the upper/lower case stuff - force bits for Greek ONLY
	// This is such a small size that a memcpy is not worth it

	if (uiFlags & HAD_LOWER_CASE)
	{
		FLMUINT		uiNumBytes = BYTES_IN_BITS( uiCaseBitPos);
		FLMBYTE *	pucCasePtr = ucCaseBits;

		// Output the 0x5
	
		pucCollatedStr [uiColLen++] = (FLMBYTE)(COLL_MARKER | SC_MIXED);
		if (puiCaseLen)
		{
			*puiCaseLen = uiNumBytes + 1;
		}

		if (uiUppercaseFlag == SC_LOWER)
		{

			// Negate case bits for languages (like GREEK) that sort
			// upper case before lower case.

			while (uiNumBytes--)
			{
				pucCollatedStr [uiColLen++] = ~(*pucCasePtr++);
			}
		}
		else
		{
			while (uiNumBytes--)
			{
				pucCollatedStr [uiColLen++] = *pucCasePtr++;
			}
		}
	}
	else
	{

		// All characters are either upper or lower case, as determined
		// by uiUppercaseFlag.

		pucCollatedStr [uiColLen++] = (FLMBYTE)(COLL_MARKER | uiUppercaseFlag);
		if( puiCaseLen)
		{
			*puiCaseLen = 1;
		}
	}
Exit:

	// Set length return value.

	if( pbDataTruncated)
	{
		*pbDataTruncated = bDataTruncated;
	}

	*puiCollatedStrLen = uiColLen;
	return( rc);
}
