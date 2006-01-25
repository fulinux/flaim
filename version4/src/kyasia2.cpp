//-------------------------------------------------------------------------
// Desc:	Convert collated string to WP string - for Asian languages.
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
// $Id: kyasia2.cpp 12312 2006-01-19 15:14:03 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define	SET_CASE_BIT		0x01
#define	SET_KATAKANA_BIT	0x01
#define	SET_WIDTH_BIT		0x02
#define	COLS_ASIAN_MARK_VAL		0x40		/* With out 0x100 */


extern	FLMUINT16	colToWPChr[];	/* Converts collated value to WP character */
extern	FLMBYTE		ml1_COLtoD[];	/* Diacritic conversions */
extern	FLMBYTE		KanaSubColTbl[];
/* Position in the table+1 is subColValue */
extern BYTE_WORD_TBL fwp_Ch24ColTbl[];	

FLMBYTE		ColToKanaTbl[ 48 ] /* Only 48 values + 0x40, 0x41, 0x42 (169..171) */
= {
	 0,	/* a=0, A=1 */
	 2,	/* i=2, I=3 */
	 4,	/* u=4, U=5, VU=83 */
	 6,	/* e=6, E=7 */
 	 8,	/* o=8, O=9 */
 	84,	/* KA=10, GA=11, ka=84 - remember voicing table is optimized */
 			/*                       so that zero value is position and  */
 			/*                       if voice=1 and no 0 is changed to 0 */
 	12,	/* KI=12, GI=13 */
 	14,	/* KU=14, GU=15 */
 	85,	/* KE=16, GE=17, ke=85 */
 	18,	/* KO=18, GO=19 */
/*10*/
 	20,	/* SA=20, ZA=21 */
 	22,	/* SHI=22, JI=23 */
 	24,	/* SU=24, ZU=25 */
 	26,	/* SE=26, ZE=27 */
 	28,	/* SO=28, ZO=29 */
 	30,	/* TA=30, DA=31 */
	32,	/* CHI=32, JI=33 */
	34,	/* tsu=34, TSU=35, ZU=36 */
	37,	/* TE=37, DE=38 */
	39,	/* TO=39, DO=40 */
/*20*/
	41,	/* NA */
	42,	/* NI */
	43,	/* NU */
	44,	/* NE */
	45,	/* NO */		
	46,	/* HA, BA, PA */
	49,	/* HI, BI, PI */
	52,	/* FU, BU, PU */
	55,	/* HE, BE, PE */
	58,	/* HO, BO, PO */
/*30*/	
	61,	/* MA */
	62,	/* MI */
	63,	/* MU */
	64,	/* ME */
	65,	/* MO */
	66,	/* ya, YA */
	68,	/* yu, YU */
	70,	/* yo, YO */
	72,	/* RA */
	73,	/* RI */
/*40*/	
	74,	/* RU */
	75,	/* RE */
	76,	/* RO */
	77,	/* wa, WA */
	79,	/* WI */
	80,	/* WE */
	81,	/* WO */
	82		/*  N */
};

/***************************************************************************
Desc:		Get the original string from an asian collation string
Ret:		Length of the word string in bytes
****************************************************************************/

FLMUINT		AsiaConvertColStr(
	FLMBYTE *	CollatedStr,			/* Points to the Flaim collated string */
	FLMUINT *	CollatedStrLenRV,		/* Length of the Flaim collated string */
	FLMBYTE *	WordStr,			  		/* Output string to build - WP word string */
	FLMBOOL *	pbDataTruncated,		/* Set to TRUE if data was truncated */
	FLMBOOL *	pbFirstSubstring)		/* Set to TRUE if marker exists */
{
	FLMBYTE *	pWordStr = WordStr;	/* Points to the word string data area */
	FLMUINT		Length = *CollatedStrLenRV;/* May optimize as a register */
	FLMUINT		CollStrPos = 0;		/* Position in CollatedStr[] */
	FLMBOOL		bHadExtended = FALSE;
	FLMUINT		WordStrLen;
	FLMUINT16	ColChar;					/* 2 byte value for asian */

	while( Length)
	{
		FLMBYTE	CharVal, CharSet;
		CharSet = CollatedStr[ CollStrPos ];
		CharVal = CollatedStr[ CollStrPos + 1 ];
		ColChar = (FLMUINT16)((CharSet << 8) + CharVal);
		
		if( ColChar <= MAX_COL_OPCODE)
			break;

		CollStrPos += 2;
		Length -= 2;
		if( CharSet == 0)				/* Normal Latin/Greek/Cyrillic value */
		{
			ColChar = colToWPChr[ CharVal - COLLS ];
		}
		else if( CharSet == 1)		/* katakana or hiragana character */
		{
			if( CharVal > sizeof( ColToKanaTbl ))	/* Special cases below */
			{
				if( CharVal == COLS_ASIAN_MARK_VAL)			/* dakuten */
					ColChar = 0x240a;
				else if( CharVal == COLS_ASIAN_MARK_VAL + 1)	/* handakuten */
					ColChar = 0x240b;
				else if( CharVal == COLS_ASIAN_MARK_VAL + 2)	/* chuuten */
					ColChar = 0x2405;
				else
					ColChar = 0xFFFF;			/* error */
			}
			else
			{
				ColChar = (FLMUINT16)(0x2600 + ColToKanaTbl[ CharVal ]);
			}
		}
		else if( CharSet != 0xFF || CharVal != 0xFF)	// Asian characters
		{
			// Insert zeroes that will be treated as a signal for
			// uncoverted unicode characters later on.  NOTE: Cannot
			// use 0xFFFF, because we need to be able to detect this
			// case in the sub-collation stuff, and we don't want
			// to confuse it with the 0xFFFF that may have been inserted
			// in another case.
			// THIS IS A REALLY BAD HACK, BUT IT IS THE BEST WE CAN DO
			// FOR NOW!
			*pWordStr++ = 0;
			*pWordStr++ = 0;
			bHadExtended = TRUE;
		}
		/* else does not have a collation value - found in sub-collation part */
		
		UW2FBA( ColChar, pWordStr );		/* Put the uncollation value back */
		pWordStr += 2;
	}

	UW2FBA( 0, pWordStr);			/* NULL Terminate the string */
	WordStrLen = (FLMUINT) (pWordStr - WordStr);

	/**--------------------------------------------------------------------
	***  Parse through the sub-collation and case information.
	***  Watch out for COMP CollStrPosT indexes-doesn't have case info after
	***  Here are values for some of the codes:
	***   [ 0x01] - end for fields case info follows - for COMP POST indexes
	***   [ 0x02] - compound marker
	***   [ 0x05] - case bits follow
	***   [ 0x06] - case information is all uppercase
	***   [ 0x07] - beginning of sub-collation information
	***	[ 0x08] - first substring field that is made
	***	[ 0x09] - truncation marker for text and binary
	*** 
	***  Asian chars the case information should always be there and not
	***  compressed out.  This is because the case information could change
	***  the actual width of the character from 0x26xx to charset 11.
	***-------------------------------------------------------------------*/

	/**
	***  Does truncation marker or sub-collation follow?
	**/
	if( Length)
	{
		ColChar = (FLMUINT16)((CollatedStr[CollStrPos] << 8) +
									CollatedStr[CollStrPos+1]);

		// First substring is before truncated.
		if( ColChar == COLL_FIRST_SUBSTRING)
		{
			if( pbFirstSubstring)
				*pbFirstSubstring = TRUE;		// Don't need to initialize to FALSE.
			Length -= 2;
			CollStrPos += 2;
			ColChar = (FLMUINT16)((CollatedStr[CollStrPos] << 8) +
										CollatedStr[CollStrPos+1]);
		}
		if( ColChar == COLL_TRUNCATED)
		{
			if( pbDataTruncated)
				*pbDataTruncated = TRUE;		// Don't need to initialize to FALSE.
			Length -= 2;
			CollStrPos += 2;
			ColChar = (FLMUINT16)((CollatedStr[CollStrPos] << 8) +
										CollatedStr[CollStrPos+1]);
		}
		if( ColChar == (COLL_MARKER | SC_SUB_COL))
		{
			FLMUINT 	TempLen;

			/* Do another pass on the word string adding diacritics/voicings */
			CollStrPos += 2;
			Length -= 2;
			TempLen = AsiaParseSubCol( WordStr, &WordStrLen, 
												&CollatedStr[ CollStrPos ]);
			CollStrPos += TempLen;
			Length -= TempLen;
		}
		else
			goto check_case;
	}
	
	/**
	***  Does the case info follow? - It may not because of post indexes
	**/
	if( Length)
	{
		ColChar = (FLMUINT16)((CollatedStr[CollStrPos] << 8) +
									CollatedStr[CollStrPos+1]);
check_case:
		if( ColChar == (COLL_MARKER | SC_MIXED))
		{
			CollStrPos += 2;
			CollStrPos += AsiaParseCase( WordStr, &WordStrLen,
											&CollatedStr[CollStrPos]);

			// Set bHadExtended to FALSE, because they will have
			// been taken care of in this pass.

			bHadExtended = FALSE;
		}
	}

	// Change embedded zeroes to 0xFFFFs

	if (bHadExtended)
	{
		FLMUINT		uiCnt;
		FLMBYTE *	pucTmp;

		for (uiCnt = 0, pucTmp = WordStr;
			  uiCnt < WordStrLen;
			  uiCnt += 2, pucTmp += 2)
		{
			if (FB2UW( pucTmp) == 0)
			{
				UW2FBA( 0xFFFF, pucTmp);
			}
		}
	}

	/* Follow marker is 2 bytes if post otherwise will be 1 byte */

	/* Should make a pass and count the extended characters */

	*CollatedStrLenRV = CollStrPos; 	/* value should be on 0x01 or 0x02 flag */
	return( WordStrLen);					/* Return the length of the word string */
}

/****************************************************************************
Desc:		Combine the diacritic 5 and 16 bit values to an existing word string.
Ret:		FLMUINT - Number of bytes parsed
Notes:	For each bit in the sub-collation section:
	0 - no subcollation information
	10 - take next 5 bits - will tell about diacritics or japanese vowel
	110 - align to next byte & take word value as extended character

****************************************************************************/

FLMUINT	AsiaParseSubCol(
	FLMBYTE *	WordStr,						/* Existing word string to modify */
	FLMUINT *	puiWordStrLen,				/* Wordstring length in bytes */
	FLMBYTE *	SubColBuf					/* Diacritic values in 5 bit sets */
	)
{
	FLMUINT 		SubColBitPos = 0;
	FLMUINT 		NumWords = *puiWordStrLen >> 1;
	FLMUINT16 	Diac;
	FLMUINT16 	WpChar;

	/* For each word in the word string ... */
	while( NumWords--)
	{

		// Have to skip 0, because it is not accounted for
		// in the sub-collation bits.  It was inserted when we
		// encountered unconverted unicode characters (Asian).
		// Will be converted to something else later on.
		// SEE NOTE ABOVE.

		if (FB2UW( WordStr) == 0)
		{
			WordStr += 2;
			continue;
		}

		/* This macro DOESN'T increment bitPos */
		if( TEST1BIT( SubColBuf, SubColBitPos))
		{
			/**
			***  Bits 10 - take next 5 bits
			***  Bits 110 align and take next word
			***  Bits 11110 align and take unicode value
			**/
			
			SubColBitPos++;
			if( ! TEST1BIT( SubColBuf, SubColBitPos))
			{
				SubColBitPos++;
				Diac = (FLMUINT16)(GETnBITS( 5, SubColBuf, SubColBitPos));
				SubColBitPos += 5;

				if( (WpChar = FB2UW( WordStr )) < 0x100)
				{
					if( (WpChar >= 'A') && (WpChar <= 'Z'))
					{
	
						/* Convert to WP diacritic and combine characters */
						fwpCh6Cmbcar( &WpChar, WpChar, (FLMUINT16) ml1_COLtoD[Diac] );
						/* Even if cmbcar fails, WpChar is still set to a valid value */
					}
					else							/* Symbols from charset 0x24 */
					{
						WpChar = (FLMUINT16)(0x2400 + fwp_Ch24ColTbl[ Diac - 1 ].ByteValue);
					}	
				}
				else if( WpChar >= 0x2600)		/* Katakana */
				{
					/**
					***  Voicings - will allow to select original char
					***		000 - some 001 are changed to 000 to save space
					***		001 - set if large char (uppercase)
					***		010 - set if voiced
					***		100 - set if half voiced
					***
					***  Should NOT match voicing or wouldn't be here!
					**/

					FLMBYTE CharVal = (FLMBYTE)(WpChar & 0xFF);
					
					/* Try exceptions first so don't access out of bounds */

					if( CharVal == 84)
						WpChar = (FLMUINT16)(0x2600 +
												((Diac == 1)
												? (FLMUINT16)10
												: (FLMUINT16)11));

					else if( CharVal == 85)
						WpChar = (FLMUINT16)(0x2600 +
												((Diac == 1)
												 ? (FLMUINT16)16
												 : (FLMUINT16)17));

					/* Try the next 2 slots, if not then value is 83,84 or 85 */
					
					else if( KanaSubColTbl[ CharVal + 1 ] == Diac )
						WpChar++;
					else if( /* (Diac == 5) && ZU is an exception! */
								(KanaSubColTbl[ CharVal + 2 ] == Diac ))
						WpChar += 2;

					/* last exception below */

					else if( CharVal == 4)
							WpChar = 0x2600 + 83;

					/* else leave alone! - invalid storage */
				}

				UW2FBA( WpChar, WordStr );		/* Set if changed or not */
			}
			else		/* "110" */
			{
				FLMUINT    Temp;

				SubColBitPos++;				/* Skip second '1' */
				
				if( TEST1BIT( SubColBuf, SubColBitPos))	/* 11?10 ? */
				{
					/* Unconvertable UNICODE character */
					/* The format will be 4 bytes, 0xFF, 0xFF, 2 byte Unicode */

					shiftN( WordStr, (FLMUINT16)(NumWords + NumWords + 4), 2 );
					WordStr += 2;				/* Skip the 0xFFFF for now */
					SubColBitPos += 2;		/* Skip next "11" */
					(*puiWordStrLen) += 2;
				}
				SubColBitPos++;			/* Skip the zero */

				/* Round up to next byte */
				SubColBitPos = (SubColBitPos + 7) & (~7);
				Temp = BYTES_IN_BITS( SubColBitPos );
				WordStr[1] = SubColBuf[ Temp ];				/* Character set */
				WordStr[0] = SubColBuf[ Temp + 1 ];			/* Character */
				SubColBitPos += 16;
			}
		}
		else
			SubColBitPos++;					/* Be sure to increment this! */

		WordStr += 2;							/* Next WP character */
	}

	return( BYTES_IN_BITS( SubColBitPos ));
}

/****************************************************************************
Desc:		The case bits for asia are:
				Latin/Greek/Cyrillic
					01 - case bit set if character is uppercase
					10 - double wide character in CS 0x25xx, 0x26xx and 0x27xx
				Japanese
					00 - double wide hiragana 0x255e..25b0
					01 - double wide katakana 0x2600..2655
					10 - single wide symbols from charset 11 that map to CS24??
					11 - single wide katakana from charset 11
Ret:		
Notes:	This is tricky to really understand the inputs.
	This looks at the bits according to the current character value.
****************************************************************************/

FLMUINT	AsiaParseCase(
	FLMBYTE *	WordStr,			  	/* Existing word string to modify */
	FLMUINT *	WordStrLenRV,	  	/* Length of the WordString in bytes */
	FLMBYTE *	pCaseBits	  		/* Lower/upper case bit string */
	)
{
	FLMUINT		WordStrLen = *WordStrLenRV;
	FLMUINT		uiWordCnt;
	FLMUINT		uiExtraBytes = 0;
	FLMUINT16	WpChar;
	FLMBYTE		TempByte = 0;
	FLMBYTE		MaskByte;
	
	/* For each character in the word string ... */

	for(  uiWordCnt = WordStrLen >> 1,/* Total number of words in word string */
			MaskByte = 0;					/* Force first time to get a byte */
				
			uiWordCnt--;)					/* Test */
	{
		FLMBYTE	CharSet, CharVal;
		
		WpChar = FB2UW( WordStr );		/* Get the next character */

		// Must skip any 0xFFFFs or zeroes that were inserted.

		if (WpChar == 0xFFFF || WpChar == 0)
		{
			// Put back 0xFFFF in case it was a zero.

			UW2FBA( 0xFFFF, WordStr);
			WordStr += 2;
			uiExtraBytes += 2;
			continue;
		}
		if( MaskByte == 0)				/* Time to get another byte */
		{
			TempByte = *pCaseBits++;
			MaskByte = 0x80;
		}
		CharSet = (FLMBYTE)(WpChar >> 8);
		CharVal = (FLMBYTE)(WpChar & 0xFF);
		
		if( WpChar < 0x2400 )			/*** SINGLE WIDE - NORMAL CHARACTERS ***/
		{
			if( TempByte & MaskByte)	/* convert to double wide? */
			{
				/**
				***  Latin/greek/cyrillic 
				***  Convert to uppercase double wide char
				**/
			
				if( CharSet == 0)			/* Latin - uppercase */
				{
					/* May convert to 0x250F (Latin) or CS24 */
					if( WpChar >= 'A' && WpChar <= 'Z')
						WpChar = (FLMUINT16)(WpChar - 0x30 + 0x250F);	/* Convert to double wide*/
					else
						HanToZenkaku( WpChar, 0, &WpChar );
				}
				else if( CharSet == 8)	/* Greek */
				{
					if( CharVal > 38)		/* Adjust for spaces in greek */
						CharVal -= 2;
					if( CharVal > 4)
						CharVal -= 2;	
						
					WpChar = (FLMUINT16)((CharVal >> 1) + 0x265E);
				}
				else if( CharSet == 10)	/* Cyrillic */
				{
					WpChar = (FLMUINT16)((CharVal >> 1) + 0x2700);
				}
				else
					HanToZenkaku( WpChar, 0, &WpChar );
					
				CharSet = (FLMBYTE)(WpChar >> 8);	/* Less code this way */
				CharVal = (FLMBYTE)(WpChar & 0xFF);
			}

			MaskByte >>= 1;					/* Next bit */

			if( ( TempByte & MaskByte) == 0)	/* Change to lower case? */
			{
				switch( CharSet)			/* Convert WpChar to lower case */
				{
				case	0:
					WpChar |= 0x20;		/* Bit zero only if lower case */
					break;
				case	1:
					if( CharVal >= 26)	/* in upper/lower case region? */
						WpChar++;
					break;
				case	8:
					if( CharVal <= 69)	/* All lowercase after 69 */
						WpChar++;
					break;
				case	10:
					if( CharVal <= 199)	/* No cases after 199 */
						WpChar++;
					break;
				case	0x25:
				case	0x26:
												/* should be double wide latin or greek */
					WpChar += 0x20;		/* Add offset to convert to lowercase */
					break;
				case	0x27:					/* double wide cyrillic only */
					WpChar += 0x30;		/* Add offset to convert to lowercase */
					break;
				}
			}
		}
		
		else										/***  JAPANESE CHARACTERS  ***/
		{
			if( TempByte & MaskByte)		/* Original chars from CharSet 11 */
			{
				if( CharSet == 0x26)			/* Convert to ZenToHankaku */
				{
				FLMUINT16	NextChar = 0;

					WpChar = ZenToHankaku( WpChar, &NextChar );

					if( NextChar)				/* Move everone down */
					{
						uiWordCnt++;
						shiftN( WordStr, uiWordCnt + uiWordCnt + 2, 2 );
						UW2FBA( WpChar, WordStr );
						WordStr += 2;
						WpChar = NextChar;	/* Store this below */
						
						*WordStrLenRV = *WordStrLenRV + 2;	/* Adjust length */
						/* Don't change WordStrLen - returns # bits used */
					}
				}
				else if( CharSet == 0x24)
				{
					WpChar = ZenToHankaku( WpChar, (FLMUINT16 *) 0 );
				}
				MaskByte >>= 1;				/* Eat next bit! */			
			}
			else
			{
				MaskByte >>= 1;					/* Next bit */
				if( (TempByte & MaskByte) == 0)	/* Convert to hiragana? */
				{
					/* kanji will also fall through here */
					if( CharSet == 0x26)
						WpChar = (FLMUINT16)(0x255E + CharVal);	/* Convert to hiragana */
				}
			}
		}
		UW2FBA( WpChar, WordStr );
		WordStr += 2;
		MaskByte >>= 1;
	}
	uiWordCnt = WordStrLen - uiExtraBytes;	// Should be 2 bits for each character.
		
	return( BYTES_IN_BITS( uiWordCnt ));
}
