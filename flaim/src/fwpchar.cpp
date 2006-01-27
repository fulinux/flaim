//-------------------------------------------------------------------------
// Desc:	WP character routines.
// Tabs:	3
//
//		Copyright (c) 1991-1992,1994-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fwpchar.cpp 12301 2006-01-19 15:02:55 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

static char fwp_langtbl[LAST_LANG+LAST_LANG]
		 = {
	'U', 'S',	/* English, United States  */
	'A', 'F',	/* Afrikaans               */
	'A', 'R',	/* Arabic                  */
	'C', 'A',	/* Catalan                 */
	'H', 'R',	/* Croatian                */
	'C', 'Z',	/* Czech                   */
	'D', 'K',	/* Danish                  */
	'N', 'L',	/* Dutch                   */
	'O', 'Z',	/* English, Australia      */
	'C', 'E',	/* English, Canada         */
	'U', 'K',	/* English, United Kingdom */
	'F', 'A',	/* Farsi                   */
	'S', 'U',	/* Finnish                 */
	'C', 'F',	/* French, Canada          */
	'F', 'R',	/* French, France          */
	'G', 'A',	/* Galician                */		
	'D', 'E',	/* German, Germany         */
	'S', 'D',	/* German, Switzerland     */
	'G', 'R',	/* Greek                   */
	'H', 'E',	/* Hebrew                  */
	'M', 'A',	/* Hungarian               */
	'I', 'S',	/* Icelandic               */
	'I', 'T',	/* Italian                 */
	'N', 'O',	/* Norwegian               */
	'P', 'L',	/* Polish                  */
	'B', 'R',	/* Portuguese, Brazil      */
	'P', 'O',	/* Portuguese, Portugal    */
	'R', 'U',	/* Russian                 */
	'S', 'L',	/* Slovak                  */
	'E', 'S',	/* Spanish                 */
	'S', 'V',	/* Swedish                 */
	'Y', 'K',	/* Ukrainian               */
	'U', 'R',	/* Urdu                    */
	'T', 'K',	/* Turkey                  */
	'J', 'P',	/* Japanese						*/
	'K', 'R',	/* Korean						*/
	'C', 'T',	/* Chinese-Traditional		*/
	'C', 'S',	/* Chinese-Simplified		*/
	'L', 'A'	/* Future asian language	*/
/* Removed in conjunction with change in wps6.h */
/*	'T', 'A'		Taiwanese - really CS!	*/
	};

/*
	fwp_caseConvertableRange[] defines the range of characters within the set 
	which are case convertible.
*/

static FLMBYTE fwp_caseConvertableRange[] = {
	26,241,												/* Multinational 1				*/
	0,0,													/* Multinational 2				*/
	0,0,													/* Box Drawing						*/
	0,0,													/* Symbol 1							*/
	0,0,													/* Symbol 2							*/
	0,0,													/* Math 1							*/
	0,0,													/* Math 2							*/
	0,69,													/* Greek 1							*/
	0,0,													/* Hebrew							*/
	0,199,												/* Cyrillic							*/
	0,0,													/* Japanese Kana					*/
	0,0,													/* User-defined					*/
	0,0,													/* Not defined						*/
	0,0,													/* Not defined						*/
	0,0,													/* Not defined						*/
};

/****************************************************************************
Desc:	getNextCharState can be thought of as a 2 dimentional array with
		i and j as the row and column indicators respectively.  If a value
		exists at the intersection of i and j, it is returned.  Sparse array 
		techniques are used to minimize memory usage.
****************************************************************************/
FINLINE FLMUINT16 getNextCharState(
	FLMUINT		i,
	FLMUINT		j)
{
	FLMUINT		k, x;

	for( k = fwp_indexi[ x =
			(i > START_COL) ? (START_ALL) : i ]; /* adjust so don't use full tables */
		  k <= (FLMUINT) (fwp_indexi[ x + 1] - 1);
		  k++ )
	{
			// FIXUP_AREA_SIZE should be 24.
		if(  j == fwp_indexj[ k])
		{
			return( fwp_valuea[ (i > START_COL) 
				?	(k + (FIXUP_AREA_SIZE * (i - START_ALL))) 
				: k]);
		}
	}

	return(0);
}

/****************************************************************************
Desc:	Determine the language number from the 2 byte language code
****************************************************************************/
FLMUINT FlmLanguage(
	char  *	pszLanguageCode)
{
	char		cFirstChar  = *pszLanguageCode;
	char		cSecondChar = *(pszLanguageCode+1);
	FLMUINT	uiTablePos;

	for (uiTablePos = 0; uiTablePos < (LAST_LANG+LAST_LANG); uiTablePos += 2 )
	{
		if (fwp_langtbl [uiTablePos] == cFirstChar &&
			 fwp_langtbl [uiTablePos + 1] == cSecondChar)
		{

			// Return uiTablePos div 2

			return( uiTablePos >> 1);
		}
	}

	// Language not found, return default US language

	return( US_LANG);
}

/****************************************************************************
Desc:	Determine the language code from the language number
****************************************************************************/
void FlmGetLanguage(
	FLMUINT	uiLangNum,
	char  *	pszLanguageCode)
{

	// iLangNum could be negative

	if (uiLangNum >= LAST_LANG)
	{
		uiLangNum = US_LANG;
	}

	uiLangNum += uiLangNum;
	*pszLanguageCode++ = fwp_langtbl [uiLangNum];
	*pszLanguageCode++ = fwp_langtbl [uiLangNum + 1];
	*pszLanguageCode = 0;
}

/****************************************************************************
Desc:	Converts a character to upper case (if possible)
****************************************************************************/
FLMUINT16 fwpCh6Upper(
	FLMUINT16	ui16WpChar)
{
	if (ui16WpChar < 256)
	{
		if (ui16WpChar >= ASCII_LOWER_A && ui16WpChar <= ASCII_LOWER_Z) 
		{

			// Return ASCII upper case

			return( ui16WpChar & 0xdf);
		}
	}
	else
	{	
		FLMBYTE	ucCharSet = ui16WpChar >> 8;

		if (ucCharSet == CHSMUL1)
		{
			FLMBYTE	ucChar = ui16WpChar & 0xFF;

			if (ucChar >= fwp_caseConvertableRange[ (CHSMUL1-1) * 2] &&
				 ucChar <= fwp_caseConvertableRange[ ((CHSMUL1-1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if (ucCharSet == CHSGREK)
		{
			if ((ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSGREK-1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if (ucCharSet == CHSCYR)
		{
			if ((ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSCYR-1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if (ui16WpChar >= Lower_JP_a)
		{

			// Possible double byte character set alphabetic character?

			if (ui16WpChar <= Lower_JP_z)
			{

				// Japanese?

				ui16WpChar = (ui16WpChar - Lower_JP_a) + Upper_JP_A;
			}
			else if (ui16WpChar >= Lower_KR_a && ui16WpChar <= Lower_KR_z)
			{

				// Korean?

				ui16WpChar = (ui16WpChar - Lower_KR_a) + Upper_KR_A;
			}
			else if (ui16WpChar >= Lower_CS_a && ui16WpChar <= Lower_CS_z)
			{

				// Chinese Simplified?

				ui16WpChar = (ui16WpChar - Lower_CS_a) + Upper_CS_A;
			}
			else if (ui16WpChar >= Lower_CT_a && ui16WpChar <= Lower_CT_z)
			{

				// Chinese Traditional?

				ui16WpChar = (ui16WpChar - Lower_CT_a) + Upper_CT_A;
			}
		}
	}

	// Return original character - original not in lower case.

	return( ui16WpChar);
}

/****************************************************************************
Desc:	Checks to see if WP character is upper case
****************************************************************************/
FLMBOOL fwpIsUpper(
	FLMUINT16	ui16WpChar
	)
{
	FLMBYTE	ucChar;
	FLMBYTE	ucCharSet;

	// Get character

	ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

	// Test if ASCII character set

	if (!(ui16WpChar & 0xFF00))
	{
		return( (ucChar >= ASCII_LOWER_A && ucChar <= ASCII_LOWER_Z)
				  ? FALSE
				  : TRUE);
	}

	// Get the character set

	ucCharSet = (FLMBYTE) (ui16WpChar >> 8);

	// CHSMUL1 == Multinational 1 character set
	// CHSGREK == Greek character set
	// CHSCYR == Cyrillic character set

	if ((ucCharSet == CHSMUL1 && ucChar >= 26 && ucChar <= 241) ||
		 (ucCharSet == CHSGREK && ucChar <= 69) ||
		 (ucCharSet == CHSCYR && ucChar <= 199))
	{	
		return( (ucChar & 1) ? FALSE : TRUE);
	}

	// Don't care that double ss is lower

	return( TRUE);
}

/****************************************************************************
Desc:	Converts a character to lower case (if possible)
****************************************************************************/
FLMUINT16 fwpCh6Lower(
	FLMUINT16	ui16WpChar)
{
	if (ui16WpChar < 256)
	{
		if (ui16WpChar >= ASCII_UPPER_A && ui16WpChar <= ASCII_UPPER_Z) 
		{
			return( ui16WpChar | 0x20);
		}
	}
	else
	{	
		FLMBYTE	ucCharSet = ui16WpChar >> 8;

		if (ucCharSet == CHSMUL1)
		{
			FLMBYTE	ucChar = ui16WpChar & 0xFF;

			if (ucChar >= fwp_caseConvertableRange[ (CHSMUL1-1) * 2] &&
				 ucChar <= fwp_caseConvertableRange[ ((CHSMUL1-1) * 2) + 1] )
			{
				return( ui16WpChar | 1);
			}
		}
		else if (ucCharSet == CHSGREK)
		{
			if ((ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSGREK-1) * 2) + 1])
			{
				return( ui16WpChar | 1);
			}
		}
		else if (ucCharSet == CHSCYR)
		{
			if ((ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSCYR-1) * 2) + 1])
			{
				return( ui16WpChar | 1);
			}
		}
		else if (ui16WpChar >= Upper_JP_A)
		{
			// Possible double byte character set alphabetic character?

			if (ui16WpChar <= Upper_JP_Z)
			{

				// Japanese?

				ui16WpChar = ui16WpChar - Upper_JP_A + Lower_JP_a;
			}
			else if (ui16WpChar >= Upper_KR_A && ui16WpChar <= Upper_KR_Z)
			{

				// Korean?

				ui16WpChar = ui16WpChar - Upper_KR_A + Lower_KR_a;
			}
			else if (ui16WpChar >= Upper_CS_A && ui16WpChar <= Upper_CS_Z)
			{

				// Chinese Simplified?

				ui16WpChar = ui16WpChar - Upper_CS_A + Lower_CS_a;
			}
			else if (ui16WpChar >= Upper_CT_A && ui16WpChar <= Upper_CT_Z)
			{

				// Chinese Traditional?

				ui16WpChar = ui16WpChar - Upper_CT_A + Lower_CT_a;
			}
		}
	}

	// Return original character, original not in upper case

	return(ui16WpChar);
}

/****************************************************************************
Desc:	Break a WP character into a base and a diacritical char.
Ret: 	TRUE - if not found
		FALSE - if found
****************************************************************************/
FLMBOOL fwpCh6Brkcar(
	FLMUINT16		ui16WpChar, 
	FLMUINT16 *		pui16BaseChar,
	FLMUINT16 *		pui16DiacriticChar)
{
	BASE_DIACRITP	pBaseDiacritic;
	FLMINT			iTableIndex;

	if ((pBaseDiacritic = fwp_car60_c[ HI(ui16WpChar)]) == 0)
	{
		return( TRUE);
	}

	iTableIndex = ((FLMBYTE)ui16WpChar) - pBaseDiacritic->start_char;
	if (iTableIndex < 0 ||
		 iTableIndex > pBaseDiacritic->char_count ||
		 pBaseDiacritic->table [iTableIndex].base == (FLMBYTE)0xFF)
	{
		return( TRUE);
	}

	if ((HI( ui16WpChar) != CHSMUL1) ||
		 ((fwp_ml1_cb60[ ((FLMBYTE) ui16WpChar) >> 3] >>
			(7 - (ui16WpChar & 0x07))) & 0x01))
	{

		// normal case, same base as same as characters

		*pui16BaseChar = (ui16WpChar & 0xFF00) |
								pBaseDiacritic->table [iTableIndex].base;
		*pui16DiacriticChar = (ui16WpChar & 0xFF00) |
								pBaseDiacritic->table[iTableIndex].diacrit;
	}
	else
	{

		// Multi-national where base is ascii value.

		*pui16BaseChar = pBaseDiacritic->table [iTableIndex].base;
		*pui16DiacriticChar = (ui16WpChar & 0xFF00) |
										pBaseDiacritic->table[iTableIndex].diacrit;
	}
	return( FALSE);
}

/****************************************************************************
Desc:	Take a base and a diacritic and compose a WP character.
		Note on base character: i's and j's must be dotless i's and j's (for
		those which use them) or they will not be found.
Ret: 	TRUE - if not found
		FALSE  - if found
Notes: ascii characters with diacriticals are in multi-national if anywhere;
		 all other base chars with diacritics are found in their own sets.
****************************************************************************/
FLMBOOL fwpCh6Cmbcar(
	FLMUINT16 *	pui16WpChar, 
	FLMUINT16	ui16BaseChar, 
	FLMINT16		ui16DiacriticChar)
{
	FLMUINT					uiRemaining;
	FLMBYTE					ucCharSet;
	FLMBYTE					ucChar;
	BASE_DIACRITP			pBaseDiacritic;
	BASE_DIACRIT_TABLEP	pTable;

	ucCharSet = HI( ui16BaseChar);
	if (ucCharSet > fwp_max_car60_size)
	{
		return( TRUE);
	}

	// Is base ASCII?  If so, look in multinational 1

	if (!ucCharSet)
	{
		ucCharSet = CHSMUL1;
	}

	if ((pBaseDiacritic = fwp_car60_c[ucCharSet]) == 0)
	{
		return( TRUE);
	}

	ucChar = LO( ui16BaseChar);
	ui16DiacriticChar = LO( ui16DiacriticChar);
	pTable = pBaseDiacritic->table;
	for (uiRemaining = pBaseDiacritic->char_count;
		  uiRemaining;
		  uiRemaining--, pTable++ )
	{

		// Same base?

		if (pTable->base == ucChar &&
			 (pTable->diacrit & 0x7F) == ui16DiacriticChar)
		{

			// Same diacritic?

			*pui16WpChar = (FLMUINT16) (((FLMUINT16) ucCharSet << 8) + 
					(pBaseDiacritic->start_char +
					 (FLMUINT16)(pTable - pBaseDiacritic->table)));
			return( FALSE);
		}
	}
	return( TRUE);
}

/**************************************************************************
Desc:	Find the collating value of a WP character
ret:	Collating value (COLS0 is high value - undefined WP char)
***********************************************************************/
FLMUINT16 fwpGetCollation(
	FLMUINT16	ui16WpChar,
	FLMUINT		uiLanguage)
{
	FLMUINT16		ui16State;
	FLMBYTE			ucCharVal;
	FLMBYTE			ucCharSet;
	FLMBOOL			bHebrewArabicFlag = FALSE;
	TBL_B_TO_BP *	pColTbl = fwp_col60Tbl;

	// State ONLY for non-US

	if (uiLanguage != US_LANG)
	{
		if (uiLanguage == AR_LANG ||		// Arabic
			 uiLanguage == FA_LANG ||		// Farsi - persian
			 uiLanguage == HE_LANG ||		// Hebrew
			 uiLanguage == UR_LANG) 		// Urdu
		{
			pColTbl = fwp_HebArabicCol60Tbl;
			bHebrewArabicFlag = TRUE;
		}	
		else
		{

			// check if uiLanguage candidate for alternate double collating

			ui16State = getNextCharState( START_COL, uiLanguage);
			if (0 != (ui16State = getNextCharState( (ui16State
							?	ui16State		// look at special case languages
							:	START_ALL),		// look at US and European
							(FLMUINT) ui16WpChar)))
			{
				return( ui16State);
			}
		}
	}

	ucCharVal = (FLMBYTE)ui16WpChar;
	ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
	
	// This is an optimized version of f_b_bp_citrp() inline for performance

	do
	{
		if (pColTbl->key == ucCharSet)
		{
			FLMBYTE *	pucColVals;	// table of collating values

			pucColVals = pColTbl->charPtr;

			// Check if the value is in the range of collated chars

			// Above lower range of table?

			if (ucCharVal >= *pucColVals)
			{

				// Make value zero based to index

				ucCharVal -= *pucColVals++;

				// Below maximum number of table entries?

				if (ucCharVal < *pucColVals++)
				{

					// Return collated value.

					return( pucColVals[ ucCharVal]);
				}
			}
		}

		// Go to next table entry

		pColTbl++;
	} while (pColTbl->key != 0xFF);

	if (bHebrewArabicFlag)
	{
		if (ucCharSet == CHSHEB ||
			 ucCharSet == CHSARB1 ||
			 ucCharSet == CHSARB2)
		{

			// Same as COLS0_HEBREW

			return( COLS0_ARABIC);
		}
	}

	// Defaults for characters that don't have a collation value.

	return( COLS0);
}

/****************************************************************************
Desc:	Check for double characters that sort as 1 (like ch in Spanish) or
		1 character that should sort as 2 (like � sorts as ae in French).
Return:	0 = nothing changes.  Otherwise, *pui16WpChar is the first
			character, and the return value contains the 2nd character.
			In addition, *pbTwoIntoOne will be TRUE if we should take two
			characters and treat as one (i.e, change the collation on the
			outside to one more than the collation of the first character).
****************************************************************************/
FLMUINT16 fwpCheckDoubleCollation(
	FLMUINT16 *			pui16WpChar,
	FLMBOOL *			pbTwoIntoOne,
	const FLMBYTE **	ppucInputStr,
	FLMUINT				uiLanguage)
{
	FLMUINT16	ui16CurState;
	FLMUINT16	ui16WpChar;
	FLMUINT16	ui16SecondChar;
	FLMUINT16	ui16LastChar = 0;
	FLMUINT		uiInLen;
	FLMBOOL		bUpperFlag;

	ui16WpChar = *pui16WpChar;
	bUpperFlag = fwpIsUpper( ui16WpChar);	

	uiInLen = 0;
	ui16SecondChar = 0;

	// Primer read

	if ((ui16CurState = getNextCharState( 0, uiLanguage)) == 0)
	{
		goto Exit;
	}
	for (;;)
	{
		switch (ui16CurState)
		{
			case INSTSG:
				*pui16WpChar = ui16SecondChar = (FLMUINT16)f_toascii( 's');
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case INSTAE:
				if (bUpperFlag)
				{
					*pui16WpChar = (FLMUINT16)f_toascii( 'A');
					ui16SecondChar = (FLMUINT16)f_toascii( 'E');
				}
				else
				{
					*pui16WpChar = (FLMUINT16)f_toascii( 'a');
					ui16SecondChar = (FLMUINT16)f_toascii( 'e');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case INSTIJ:
				if (bUpperFlag)
				{
					*pui16WpChar = (FLMUINT16)f_toascii( 'I');
					ui16SecondChar = (FLMUINT16)f_toascii( 'J');
				}
				else
				{
					*pui16WpChar = (FLMUINT16)f_toascii( 'i');
					ui16SecondChar = (FLMUINT16)f_toascii( 'j');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case INSTOE:
				if (bUpperFlag)
				{
					*pui16WpChar = (FLMUINT16)f_toascii( 'O');
					ui16SecondChar = (FLMUINT16)f_toascii( 'E');
				}
				else
				{
					*pui16WpChar = (FLMUINT16)f_toascii( 'o');
					ui16SecondChar = (FLMUINT16)f_toascii( 'e');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case WITHAA:
				*pui16WpChar = (FLMUINT16)(bUpperFlag
													? (FLMUINT16)0x122
													: (FLMUINT16)0x123);
				(*ppucInputStr)++;
				break;
			case AFTERC:
				*pui16WpChar = (FLMUINT16)(bUpperFlag
													? (FLMUINT16)f_toascii( 'C')
													: (FLMUINT16)f_toascii( 'c'));
				ui16SecondChar = ui16LastChar;
				*pbTwoIntoOne = TRUE;
				(*ppucInputStr)++;
				goto Exit;
			case AFTERH:
				*pui16WpChar = (FLMUINT16)(bUpperFlag
													? (FLMUINT16)f_toascii( 'H')
													: (FLMUINT16)f_toascii( 'h'));
				ui16SecondChar = ui16LastChar;
				*pbTwoIntoOne = TRUE;
				(*ppucInputStr)++;
				goto Exit;
			case AFTERL:
				*pui16WpChar = (FLMUINT16)(bUpperFlag
													? (FLMUINT16)f_toascii( 'L')
													: (FLMUINT16)f_toascii( 'l'));
				ui16SecondChar = ui16LastChar;
				*pbTwoIntoOne = TRUE;
				(*ppucInputStr)++;
				goto Exit;
			default:
				// Handles STATE1 through STATE11 also
				break;
		}

		if ((ui16CurState = getNextCharState( ui16CurState,
									fwpCh6Lower( ui16WpChar))) == 0)
		{
			goto Exit;
		}
		ui16LastChar = ui16WpChar;
		ui16WpChar = (FLMUINT16) *((*ppucInputStr) + (uiInLen++)); 
	}
		
Exit:

	return( ui16SecondChar);
}
