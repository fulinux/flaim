//------------------------------------------------------------------------------
// Desc:	Header for collation routines
//
// Tabs:	3
//
//		Copyright (c) 1991-1992, 1994-2000, 2002-2006 Novell, Inc.
//		All Rights Reserved.
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
// $Id: fcollate.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FCOLLATE_H
#define FCOLLATE_H

// Character set #'s are same as high byte values
// except for algorithmic set.

#define CHSASCI					0			// ASCII
#define CHSMUL1					1			// Multinational 1
#define CHSMUL2					2			// Multinational 2
#define CHSBOXD					3			// Box drawing
#define CHSSYM1					4			// Typographic Symbols
#define CHSSYM2					5			// Iconic Symbols
#define CHSMATH					6			// Math
#define CHMATHX					7			// Math Extension
#define CHSGREK					8			// Greek
#define CHSHEB						9			// Hebrew
#define CHSCYR						10			// Cyrillic
#define CHSKANA					11			// Japanese Kana
#define CHSUSER					12			// User-defined
#define CHSARB1					13			// Arabic
#define CHSARB2					14			// Arabic script

#define NCHSETS					15			// # of character sets (excluding Asian)
#define ACHSETS					0x0E0		// Maximum character set value - Asian
#define ACHSMIN					0x024		// Minimum character set value - Asian
#define ACHCMAX					0x0FE		// Maxmimum character value in Asian sets

//	LAST_LANG - Used for tables than contain items for each language.
// *_DBCS_LANG - Start and end points for double byte languages.

#define LAST_LANG 				(XFLM_LA_LANG + 1) // Last language marker
#define FIRST_DBCS_LANG			(XFLM_JP_LANG)
#define LAST_DBCS_LANG			(XFLM_LA_LANG)

// Collating Sequence Equates
// NOTE:The collating sequence MUST start at 32 (20h).  This allows for the
//  	handling of nulls and control characters.

#define COLLS						32					// first collating number (space/end of line)
#define COLS1						(COLLS + 9)		// quotes
#define COLS2						(COLS1 + 5)		// parens
#define COLS3						(COLS2 + 6)		// money
#define COLS4						(COLS3 + 6)		// math ops
#define COLS5						(COLS4 + 8)		// math others
#define COLS6						(COLS5 + 14)	// others: %#&@\_|~
#define COLS7						(COLS6 + 13)	// greek
#define COLS8						(COLS7 + 25)	// numbers
#define COLS9						(COLS8 + 10)	// alphabet
// Three below will overlap each other
#define COLS10						(COLS9 + 60)	// cyrillic
#define COLS10h					(COLS9 + 42)	// hebrew - writes over european & cyrilic
#define COLS10a					(COLS10h + 28)	// arabic - inclusive from 198(C6)-252(FC)

#define COLS11						253				//	End of list - arabic goes to the end

#define COLS0_ARABIC				COLS11			// Set if arabic accent marking
#define COLS0_HEBREW				COLS11			// Set if hebrew accent marking

#define COLSOEM					254				//  OEM character in upper range - non-collatable
															//  Phase COLSOEM out - not used!
#define COLS0_UNICODE			254				//  Use this for UNICODE
#define COLS0						255				//  graphics/misc - chars without a collate value

// Definitions for diacritics.

#define grave						0
#define centerd					1
#define tilde						2
#define circum						3
#define crossb						4
#define slash						5
#define acute						6
#define umlaut						7
#define macron						8

#define aposab						9
#define aposbes					10
#define aposba						11

#define ring						14
#define dota						15
#define dacute						16
#define cedilla					17
#define ogonek						18
#define caron						19
#define stroke						20

#define breve						22
#define dotlesi					239
#define dotlesj					25

#define gacute						83		// greek acute
#define gdia						84		// greek diaeresis
#define gactdia					85		// acute diaeresis
#define ggrvdia					86		// grave diaeresis
#define ggrave						87		// greek grave
#define gcircm						88		// greek circumflex
#define gsmooth					89		// smooth breathing
#define grough						90		// rough breathing
#define giota						91		// iota subscript
#define gsmact						92		// smooth breathing acute
#define grgact						93		// rough breathing acute
#define gsmgrv						94		// smooth breathing grave
#define grggrv						95		// rough breathing grave
#define gsmcir						96		// smooth breathing circumflex
#define grgcir						97		// rough breathing circumflex
#define gactio						98		// acute iota
#define ggrvio						99		// grave iota
#define gcirio						100	// circumflex iota
#define gsmio						101	// smooth iota
#define grgio						102	// rough iota
#define gsmaio						103	// smooth acute iota
#define grgaio						104	// rough acute iota
#define gsmgvio					105	// smooth grave iota
#define grggvio					106	// rough grave iota
#define gsmcio						107	// smooth circumflex iota
#define grgcio						108	// rough circumflex iota
#define ghprime					81		// high prime
#define glprime					82		// low prime

#define racute						200	// russian acute
#define rgrave						201	// russian grave
#define rrtdesc					204	// russian right descender
#define rogonek					205	// russian ogonek
#define rmacron					206	// russian macron

// GWBUG 30,645 - Had 200 and 70 - the 200 for max subcol was not
// enough for the sset and JP characters - computed wrong.
// This crashed the process that was building a key of sset characaters.

#define MAX_SUBCOL_BUF			(500)	// (((MAX_KEY_SIZ / 4) * 3 + fluff
#define MAX_CASE_BYTES	  		(150) // ((MAX_KEY_SIZ -(MAX_KEY_SIZ / 8)) / 8) * 2

// Flags

#define HAD_SUB_COLLATION		0x01	// Set if had sub-collating values-diacritics
#define HAD_LOWER_CASE			0x02	// Set if you hit a lowercase character


#define COLL_FIRST_SUBSTRING	0x03	// First substring marker

#define COLL_MARKER 				0x04		// Marks place of sub-collation

#define SC_LOWER					0x00		// Only lowercase characters exist
#define SC_MIXED					0x01		// Lower/uppercase flags follow in next byte
#define SC_UPPER					0x02		// Only upper characters exist
#define SC_SUB_COL				0x03		// Sub-collation follows (diacritics|extCh)

#define COLL_TRUNCATED			0x0C		// This key piece has been truncated from original

#define UNK_UNICODE_CODE		0xFFFE	// Used for collation

#define TRUNCATED_FLAG					0x8000
#define EXCLUSIVE_LT_FLAG				0x4000
#define EXCLUSIVE_GT_FLAG				0x2000
#define SEARCH_KEY_FLAG					0x1000
#define KEY_COMPONENT_LENGTH_MASK	0x0FFF
#define KEY_LOW_VALUE					0x0FFE
#define KEY_HIGH_VALUE					0x0FFF

FINLINE FLMBOOL isKeyComponentLTExclusive(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & EXCLUSIVE_LT_FLAG) ? TRUE : FALSE);
}

FINLINE FLMBOOL isKeyComponentGTExclusive(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & EXCLUSIVE_GT_FLAG) ? TRUE : FALSE);
}

FINLINE FLMBOOL isKeyComponentTruncated(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & TRUNCATED_FLAG) ? TRUE : FALSE);
}

FINLINE FLMBOOL isSearchKeyComponent(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FB2UW( pucKeyComponent) & SEARCH_KEY_FLAG) ? TRUE : FALSE);
}

FINLINE FLMUINT getKeyComponentLength(
	const FLMBYTE *	pucKeyComponent)
{
	return( (FLMUINT)(FB2UW( pucKeyComponent)) & KEY_COMPONENT_LENGTH_MASK);
}

// Prototypes

RCODE flmUTF8ToColText(								// Source: fcollate.cpp
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucCollatedStr,
	FLMUINT *			puiCollatedStrLen,
	FLMBOOL  			bCaseInsensitive,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiCaseLen,
	FLMUINT				uiLanguage,
	FLMUINT				uiCharLimit,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bDataTruncated,
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL *			pbDataTruncated);

RCODE flmColText2StorageText(						// Source: fcollate.cpp
	const FLMBYTE *	pucColStr,
	FLMUINT			uiColStrLen,
	FLMBYTE *		pucStorageBuf,
	FLMUINT *		puiStorageLen,
	FLMUINT	   	uiLang,
	FLMBOOL *		pbDataTruncated,
	FLMBOOL *		pbFirstSubstring);

RCODE flmAsiaUTF8ToColText(						// Source: fcollate.cpp
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucColStr,
	FLMUINT *			puiColStrLen,
	FLMBOOL				bCaseInsensitive,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiCaseLen,
	FLMUINT				uiCharLimit,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bDataTruncated,
	FLMBOOL *			pbDataTruncated);

RCODE flmWPCheckDoubleCollation(					// Source: fcollate.cpp
	IF_PosIStream *	pIStream,
	FLMBOOL				bUnicodeStream,
	FLMBOOL				bAllowTwoIntoOne,
	FLMUNICODE *		puzChar,
	FLMUNICODE *		puzChar2,
	FLMBOOL *			pbTwoIntoOne,
	FLMUINT				uiLanguage);

FLMUINT16 flmWPAsiaGetCollation(					// Source: fcollate.cpp
	FLMUINT16	ui16WpChar,
	FLMUINT16	ui16NextWpChar,
	FLMUINT16   ui16PrevColValue,
	FLMUINT16 *	pui16ColValue,
	FLMUINT16 * pui16SubColVal,
	FLMBYTE *	pucCaseBits,
	FLMBOOL		bUppercaseFlag);

FLMUINT16 flmWPGetCollation(						// Source: fcollate.cpp
	FLMUINT16	ui16WpChar,
	FLMUINT		uiLanguage);

FLMUINT16 flmWPUpper(								// Source: fcollate.cpp
	FLMUINT16	ui16WpChar);

FLMUINT16 flmWPLower(								// Source: fcollate.cpp
	FLMUINT16	ui16WpChar);

FLMBOOL flmWPIsUpper(								// Source: fcollate.cpp
	FLMUINT16	ui16WpChar);

FLMBOOL flmWPBrkcar(									// Source: fcollate.cpp
	FLMUINT16		ui16WpChar,
	FLMUINT16 *		pui16BaseChar,
	FLMUINT16 *		pui16DiacriticChar);

FLMUINT16 flmWPGetSubCol(							// Source: fcollate.cpp
	FLMUINT16		ui16WPValue,
	FLMUINT16		ui16ColValue,
	FLMUINT			uiLanguage);

RCODE flmInitCharMappingTables( void);			// Source: funicode.cpp

#ifdef DEF_FLM_UNI_GLOBALS
	#define UNIG_EXTERN
#else
	#define UNIG_EXTERN		extern
#endif

UNIG_EXTERN FLMUINT16 * gv_pUnicodeToWP60
#ifdef DEF_FLM_UNI_GLOBALS
	= NULL
#endif
	;

UNIG_EXTERN FLMUINT16 * gv_pWP60ToUnicode
#ifdef DEF_FLM_UNI_GLOBALS
	= NULL
#endif
	;

UNIG_EXTERN FLMUINT gv_uiMinUniChar
#ifdef DEF_FLM_UNI_GLOBALS
	= 0
#endif
	;

UNIG_EXTERN FLMUINT gv_uiMaxUniChar
#ifdef DEF_FLM_UNI_GLOBALS
	= 0
#endif
	;

UNIG_EXTERN FLMUINT gv_uiMinWPChar
#ifdef DEF_FLM_UNI_GLOBALS
	= 0
#endif
	;

UNIG_EXTERN FLMUINT gv_uiMaxWPChar
#ifdef DEF_FLM_UNI_GLOBALS
	= 0
#endif
	;
/****************************************************************************
Desc:		Convert a Unicode character to its WP equivalent
Ret:		Returns TRUE if the character could be converted
****************************************************************************/
FINLINE FLMBOOL flmUnicodeToWP(
	FLMUNICODE		uUniChar,		// Unicode character to convert
	FLMUINT16 *		pui16WPChar)	// Returns 0 or WPChar converted.
{
	if( uUniChar <= 127)
	{
		// Character is in the ASCII conversion range

		*pui16WPChar = uUniChar;
		return( TRUE);
	}

	if( uUniChar < gv_uiMinUniChar || uUniChar > gv_uiMaxUniChar)
	{
		*pui16WPChar = 0;
		return( FALSE);
	}

	if( (*pui16WPChar = gv_pUnicodeToWP60[ uUniChar - gv_uiMinUniChar]) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Convert a WP character to its Unicode equivalent
****************************************************************************/
FINLINE RCODE flmWPToUnicode(
	FLMUINT16		ui16WPChar,
	FLMUNICODE *	puUniChar)
{
	if( ui16WPChar <= 127)
	{
		// Character is in the ASCII conversion range

		*puUniChar = (FLMUNICODE)ui16WPChar;
		return( NE_XFLM_OK);
	}

	if( ui16WPChar < gv_uiMinWPChar || ui16WPChar > gv_uiMaxWPChar)
	{
		return( RC_SET_AND_ASSERT( NE_XFLM_CONV_ILLEGAL));
	}

	*puUniChar = gv_pWP60ToUnicode[ ui16WPChar - gv_uiMinWPChar];
	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:	Reads the next character from the storage buffer
****************************************************************************/
FINLINE RCODE flmGetCharFromUTF8Buf(
	const FLMBYTE **		ppucBuf,
	const FLMBYTE *		pucEnd,
	FLMUNICODE *			puChar)
{
	const FLMBYTE *	pucBuf = *ppucBuf;
	FLMUINT				uiMaxLen = pucEnd ? (FLMUINT)(pucEnd - *ppucBuf) : 3;

	if( !uiMaxLen)
	{
		*puChar = 0;
		return( NE_XFLM_OK);
	}
	
	if( pucBuf[ 0] <= 0x7F)
	{
		if( (*puChar = (FLMUNICODE)pucBuf[ 0]) != 0)
		{
			(*ppucBuf)++;
		}
		return( NE_XFLM_OK);
	}

	if( uiMaxLen < 2 || (pucBuf[ 1] >> 6) != 0x02)
	{
		return( RC_SET( NE_XFLM_BAD_UTF8));
	}

	if( (pucBuf[ 0] >> 5) == 0x06)
	{
		*puChar = 
			(FLMUNICODE)(((FLMUNICODE)( pucBuf[ 0] - 0xC0) << 6) +
							(FLMUNICODE)(pucBuf[ 1] - 0x80));
		(*ppucBuf) += 2;
		return( NE_XFLM_OK);
	}

	if( uiMaxLen < 3 ||
		 (pucBuf[ 0] >> 4) != 0x0E ||
		 (pucBuf[ 2] >> 6) != 0x02)
	{
		return( RC_SET( NE_XFLM_BAD_UTF8));
	}

	*puChar = 
		(FLMUNICODE)(((FLMUNICODE)(pucBuf[ 0] - 0xE0) << 12) +
			((FLMUNICODE)(pucBuf[ 1] - 0x80) << 6) +
						(FLMUNICODE)(pucBuf[ 2] - 0x80));
	(*ppucBuf) += 3;

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc: 	Convert a Unicode character to UTF-8
*****************************************************************************/
FINLINE RCODE flmUni2UTF8(
	FLMUNICODE		uChar,
	FLMBYTE *		pucBuf,
	FLMUINT *		puiBufSize)
{
	if( uChar <= 0x007F)
	{
		if( pucBuf)
		{
			if( *puiBufSize < 1)
			{
				return( RC_SET( NE_XFLM_CONV_DEST_OVERFLOW));
			}

			*pucBuf = (FLMBYTE)uChar;
		}
		*puiBufSize = 1;
	}
	else if( uChar <= 0x07FF)
	{
		if( pucBuf)
		{
			if( *puiBufSize < 2)
			{
				return( RC_SET( NE_XFLM_CONV_DEST_OVERFLOW));
			}

			*pucBuf++ = (FLMBYTE)(0xC0 | (FLMBYTE)(uChar >> 6));
			*pucBuf = (FLMBYTE)(0x80 | (FLMBYTE)(uChar & 0x003F));
		}
		*puiBufSize = 2;
	}
	else
	{
		if( pucBuf)
		{
			if( *puiBufSize < 3)
			{
				return( RC_SET( NE_XFLM_CONV_DEST_OVERFLOW));
			}

			*pucBuf++ = (FLMBYTE)(0xE0 | (FLMBYTE)(uChar >> 12));
			*pucBuf++ = (FLMBYTE)(0x80 | (FLMBYTE)((uChar & 0x0FC0) >> 6));
			*pucBuf = (FLMBYTE)(0x80 | (FLMBYTE)(uChar & 0x003F));
		}
		*puiBufSize = 3;
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:		Reads the next UTF-8 character from a UTF-8 buffer
Notes:	This routine assumes that the destination buffer can hold at least
			three bytes
****************************************************************************/
FINLINE RCODE flmGetUTF8CharFromUTF8Buf(
	FLMBYTE **		ppucBuf,
	FLMBYTE *		pucEnd,
	FLMBYTE *		pucDestBuf,
	FLMUINT *		puiLen)
{
	FLMBYTE *	pucBuf = *ppucBuf;
	FLMUINT		uiMaxLen = pucEnd ? (FLMUINT)(pucEnd - *ppucBuf) : 3;

	if( !uiMaxLen || !pucBuf[ 0])
	{
		*puiLen = 0;
		return( NE_XFLM_OK);
	}
	
	if( pucBuf[ 0] <= 0x7F)
	{
		*pucDestBuf = pucBuf[ 0];
		(*ppucBuf)++;
		*puiLen = 1;
		return( NE_XFLM_OK);
	}

	if( uiMaxLen < 2 || (pucBuf[ 1] >> 6) != 0x02)
	{
		return( RC_SET( NE_XFLM_BAD_UTF8));
	}

	if( (pucBuf[ 0] >> 5) == 0x06)
	{
		pucDestBuf[ 0] = pucBuf[ 0];
		pucDestBuf[ 1] = pucBuf[ 1];
		(*ppucBuf) += 2;
		*puiLen = 2;
		return( NE_XFLM_OK);
	}

	if( uiMaxLen < 3 ||
		 (pucBuf[ 0] >> 4) != 0x0E || 
		 (pucBuf[ 2] >> 6) != 0x02)
	{
		return( RC_SET( NE_XFLM_BAD_UTF8));
	}

	pucDestBuf[ 0] = pucBuf[ 0];
	pucDestBuf[ 1] = pucBuf[ 1];
	pucDestBuf[ 2] = pucBuf[ 2];
	(*ppucBuf) += 3;
	*puiLen = 3;

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE flmGetUTF8Length(
	const FLMBYTE *	pucBuf,
	FLMUINT				uiBufLen,
	FLMUINT *			puiBytes,
	FLMUINT *			puiChars)
{
	const FLMBYTE *	pucStart = pucBuf;
	const FLMBYTE *	pucEnd = uiBufLen ? (pucStart + uiBufLen) : NULL;
	FLMUINT				uiChars = 0;

	if (!pucBuf)
	{
		goto Exit;
	}

	while( (!pucEnd || pucBuf < pucEnd) && *pucBuf)
	{
		if( *pucBuf <= 0x7F)
		{
			pucBuf++;
			uiChars++;
			continue;
		}
	
		if( (pucEnd && pucBuf + 1 >= pucEnd) ||
			 (pucBuf[ 1] >> 6) != 0x02)
		{
			return( RC_SET( NE_XFLM_BAD_UTF8));
		}
	
		if( ((*pucBuf) >> 5) == 0x06)
		{
			pucBuf += 2;
			uiChars++;
			continue;
		}
	
		if( (pucEnd && pucBuf + 2 >= pucEnd) ||
			 (pucBuf[ 0] >> 4) != 0x0E || 
			 (pucBuf[ 2] >> 6) != 0x02)
		{
			return( RC_SET( NE_XFLM_BAD_UTF8));
		}
		
		pucBuf += 3;
		uiChars++;
	}

Exit:

	*puiChars = uiChars;
	if (pucEnd && pucBuf == pucEnd)
	{
		*puiBytes = (FLMUINT)(pucBuf - pucStart);
	}
	else
	{
		// Hit a null byte
		*puiBytes = (FLMUINT)(pucBuf - pucStart) + 1;
	}

	return( NE_XFLM_OK);
}

RCODE	flmUnicode2UTF8(								// Source: funicode.cpp
	FLMUNICODE *	puzStr,
	FLMUINT			uiStrLen,
	FLMBYTE *		pucBuf,
	FLMUINT *		puiBufLength);
	
#endif
