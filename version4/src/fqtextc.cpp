//-------------------------------------------------------------------------
// Desc:	Query text comparison
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fqtextc.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

// From COLTBL.cpp

extern FLMBYTE  fwp_dia60Tbl[];		/* Diacritic conversions */
extern FLMBYTE  fwp_alefSubColTbl[];
extern FLMBYTE  fwp_ar2BitTbl[];

#define COMPARE_COLLATION				1
#define COMPARE_COL_AND_SUBCOL		2
#define COMPARE_VALUE					3

#define NULL_SUB_COL_CHECK			NULL
#define NULL_CASE_CHECK				NULL
#define NULL_WILD_CARD_CHECK		NULL

FSTATIC FLMINT	flmTextCompareSingleChar(
	FLMBYTE **		ppLeftText,	
	FLMUINT *		puiLeftLen,	
	FLMUINT *		puiLeftWpChar2,
	FLMBYTE **		ppRightText,
	FLMUINT *		puiRightLen,
	FLMUINT *		puiRightWpChar2,
	FLMINT *			piSubColCompare,
	FLMINT *			piCaseCompare,	
	FLMBOOL *		pbHitWildCard,
	FLMINT			iCompareType,	
	FLMUINT16 *		pui16ColVal,
	FLMUINT			uiFlags,
	FLMUINT			uiLangId);


FSTATIC FLMUINT16	flmTextGetSubCol(
	FLMUINT16		ui16WPValue,
	FLMUINT16		ui16ColValue,
	FLMUINT			uiLangId);

/****************************************************************************
Desc: 	
****************************************************************************/
FINLINE FLMUINT flmCharTypeAnsi7( 
	FLMUINT16	ui16Char)
{
	if( (ui16Char >= ASCII_LOWER_A && ui16Char <= ASCII_LOWER_Z) ||
		 (ui16Char >= ASCII_UPPER_A && ui16Char <= ASCII_UPPER_Z) ||
		 (ui16Char >= ASCII_ZERO && ui16Char <= ASCII_NINE))
	{
		return SDWD_CHR;
	}
	if( ui16Char == 0x27)
		return WDJN_CHR;
	
	if( ui16Char <= 0x2B)
		return DELI_CHR;

	if( ui16Char == ASCII_COMMA ||
		 ui16Char == ASCII_DASH ||
		 ui16Char == ASCII_DOT ||
		 ui16Char == ASCII_SLASH ||
		 ui16Char == ASCII_COLON ||
		 ui16Char == ASCII_AT ||
		 ui16Char == ASCII_BACKSLASH ||
		 ui16Char == ASCII_UNDERSCORE)
		return WDJN_CHR;
	return DELI_CHR;
}


/*API~***********************************************************************
Name :	FlmStrCmp
Desc :	Compare two unicode strings. This comparison uses the collation
			rules that are defined for the specified language.
Return:	Signed value of compare.  
			<0 if less than, 0 if equal, >0 if greater than
			The case of returning 1 may be in using wild cards which
			only need to return a does not match value.
*END************************************************************************/
FLMINT  FlmStrCmp(
	FLMUINT					uiCompFlags,		
	FLMUINT					byLang,
	const FLMUNICODE *	uzStr1,
	const FLMUNICODE *	uzStr2)
{
	FLMINT		iCmp;
	POOL			Pool;
	NODE	*		pNd1;
	NODE	*		pNd2;
	RCODE			rc;

	GedPoolInit( &Pool, 256);

	if( (pNd1 = GedNodeMake( &Pool, 1, &rc)) == NULL ||
		 (pNd2 = GedNodeMake( &Pool, 1, &rc)) == NULL)
	{
		flmAssert( 0);
		iCmp = 1;
      goto Exit;	
	}

	if( RC_BAD( rc = GedPutUNICODE( &Pool, pNd1, uzStr1)))
	{
		flmAssert( RC_OK( rc));
		iCmp = 1;
		goto Exit;
	}

	if( RC_BAD( rc = GedPutUNICODE( &Pool, pNd2, uzStr2)))
	{
		flmAssert( RC_OK( rc));
		iCmp = -1;
		goto Exit;
	}

	// Handle null string cases.

	if( GedValLen( pNd1) == 0)
	{
		iCmp = 1;
		goto Exit;
	}
	else if( GedValLen( pNd2) == 0)
	{
		iCmp = -1;
		goto Exit;
	}

	// VISIT: need to add support for the IGNORE_DASH and IGNORE_SPACE options.

	iCmp = flmTextCompare( (FLMBYTE *)GedValPtr( pNd1), GedValLen( pNd1), 
			(FLMBYTE *)GedValPtr( pNd2), GedValLen( pNd2), uiCompFlags, byLang);

Exit:
	GedPoolFree( &Pool);
	return iCmp;
}

/****************************************************************************
Desc:  	Compare two entire strings.  There is some debate how this routine
			should compare the sub-collation values when wild cards are used.
			THIS DOES NOT ALLOW WILD CARDS.
Return:	Signed value of compare.  
			<0 if less than, 0 if equal, >0 if greater than
			The case of returning 1 may be in using wild cards which
			only need to return a does not match value.
****************************************************************************/
FLMINT flmTextCompare(										
	FLMBYTE *	pLeftBuf,
	FLMUINT		uiLeftLen,
	FLMBYTE *	pRightBuf,
	FLMUINT		uiRightLen,
	FLMUINT		uiFlags,
	FLMUINT		uiLang)
{
	FLMINT		iCompare = 0;
	FLMINT		iSubColCompare = 0;		// MUST BE INITIALIZED
	FLMINT *		pSubColCompare;
	FLMINT		iCaseCompare = 0;			// MUST BE INITIALIZED
	FLMINT *		pCaseCompare;
	FLMUINT		uiLeadingSpace;
	FLMUINT		uiTrailingSpace;
	FLMUINT16	ui16ColVal = 0;				// Needed for asian collation
	FLMUINT16	ui16WPChar;
	FLMUINT16	ui16UniChar;
	FLMUINT		uiLeftWpChar2 = 0;
	FLMUINT		uiRightWpChar2 = 0;
	
	uiTrailingSpace = uiLeadingSpace = 
		(uiFlags & FLM_MIN_SPACES) ? FLM_NO_SPACE : 0;
	pCaseCompare = (uiFlags & FLM_NOCASE) ? NULL : &iCaseCompare;
	pSubColCompare = &iSubColCompare;

	// Handle NULL buffers first.

	if (!pLeftBuf)
	{
		if (pRightBuf)
		{
			iCompare = -1;
		}
		goto Exit;
	}

	while ((uiLeftLen || uiLeftWpChar2) &&
			 (uiRightLen || uiRightWpChar2))
	{
		if ((iCompare = flmTextCompareSingleChar(
								&pLeftBuf, &uiLeftLen, &uiLeftWpChar2,
								&pRightBuf, &uiRightLen, &uiRightWpChar2,
								pSubColCompare, pCaseCompare, NULL_WILD_CARD_CHECK,
								COMPARE_COLLATION, &ui16ColVal, 
								uiFlags | uiLeadingSpace, uiLang)) != 0)
		{
			goto Exit;
		}
		uiLeadingSpace = 0;
	}

	// EQUAL - as far as the collation values are concerned and one
	// or both of the strings is at the end.

	if (uiLeftLen || uiLeftWpChar2)
	{
		uiLeftLen -= flmTextGetValue( pLeftBuf, uiLeftLen, &uiLeftWpChar2,
								uiFlags | uiTrailingSpace, &ui16WPChar, &ui16UniChar);

		if (uiLeftLen || ui16WPChar || ui16UniChar)
		{
			iCompare = 1;
		}
	}
	else if (uiRightLen || uiRightWpChar2)
	{
		uiRightLen -= flmTextGetValue( pRightBuf, uiRightLen, &uiRightWpChar2,
			uiFlags | uiTrailingSpace, &ui16WPChar, &ui16UniChar);
		if (uiRightLen || ui16WPChar || ui16UniChar)
		{
			iCompare = -1;
		}
	}
	if (iCompare == 0)
	{

		// All collation bytes equal - return subcollation/case difference.

		iCompare = (iSubColCompare != 0) ? iSubColCompare : iCaseCompare;
	}

Exit:

	return iCompare;
}

/****************************************************************************
Desc:  	Match two entire strings.  
Return:	FLM_TRUE or FLM_FALSE
Notes:	This code calls the collation routine because in the future there
			will be equal conditions with different unicode characters.

DOCUMENTATION DEALING WITH WILD CARDS AND SPACE RULES.

	The space rules are not obvious when dealing with wild cards.  
	This will outline the rules that are being applied so that we can
	do a regression test when this code changes.

	Rule #1:	Return same result if leading or trailing wild card is added.
				The underscore is also the space character in these examples
				and the MIN_SPACES rule is being applied.

	Format:	DataString Operator SearchString

	Example:	if     A == A      A_ == A      A == A_     A_ == A_  
				then   A == A*     A_ == A*     A == A_*    A_ == A_*
				and    A == *A     A_ == *A     A == *A_    A_ == *A_
				and    A == *A*    A_ == *A*    A == *A_*   A_ == *A_*
				where 'A' represent a string of any characters.

	Strictly put, the query Field == A_* can be broken down to
		Field == A || Field == A_*
	where the space after 'A' should not be treated as a trailing space.

	In addition we can apply the space before the string with the same results,
	but we are not going to handle the case of *_A correctly.
	This is because the query *_A should be expanded to 
		Field == A || Field == *_A
	where the space before 'A' should not be treated as a leading space.
	When we need to find "_A" in a search string then we will expand the
	query to handle this.


	Rule #2:	The spaces before a trailing truncation are NOT to be treated
				as trailing spaces if there are remaining bytes in the data string.

	Example:	(A_B == A_*) but (AB != A_*)


	Rule #3:	Space value(s) without anything other value are equal to no values.
	Example:	(" " == "")


	Rule #4: Trim leading/trailing spaces before and after wild cards. 
				SMI does this when formatting.

		_* and *_ same as *			so A == _* and A = *_ but A != *_*


	Additional wildcard cases to test for:

	Wildcard cases to handle.
		(ABBBBC == A*BC)			Hits the goto Compare_Again case three times.
		(ABBBBD != A*B)			Stuff still remains in dataString
		(ABBBBC != A*BCD)			Stuff still remains in searchString

****************************************************************************/
FLMUINT flmTextMatch(										
	FLMBYTE *	pLeftBuf,
	FLMUINT		uiLeftLen,
	FLMBYTE *	pRightBuf,
	FLMUINT		uiRightLen,
	FLMUINT		uiFlags,
	FLMBOOL		bLeadingWildCard,
	FLMBOOL		bTrailingWildCard,
	FLMUINT		uiLang)
{
	FLMINT		iCompare = 0;
	FLMUINT		uiLeadingSpace;
	FLMUINT		uiTrailingSpace;
	FLMBOOL		bHitWildCard;
	FLMBOOL		bHasWildCardPos;
	FLMBOOL *	pbHitWildCard;
	FLMUINT		uiValueLen;
	FLMUINT16	ui16WPChar;
	FLMUINT16	ui16UniChar;
	FLMUINT16	ui16Tmp1;
	FLMUINT16	ui16Tmp2;
	FLMINT		iCompareType;
	FLMUINT		uiLeftWpChar2 = 0;
	FLMUINT		uiRightWpChar2 = 0;
	// LWCP = Last Wild Card Position - used for wild card state
	FLMBYTE *	pLWCPLeftBuf = NULL;
	FLMBYTE *	pLWCPRightBuf = NULL;
	FLMUINT		uiLWCPLeftLen = 0;
	FLMUINT		uiLWCPRightLen = 0;
	FLMUINT		uiLWCPLeftWpChar2 = 0;
	FLMUINT		uiLWCPRightWpChar2 = 0;

	if( uiFlags & FLM_COMPARE_COLLATED_VALUES)
	{
		iCompareType = COMPARE_COLLATION;
	}
	else
	{
		iCompareType = (uiFlags & FLM_NOCASE) 
								? COMPARE_COL_AND_SUBCOL : COMPARE_VALUE;
	}

	// Handle NULL buffers first - don't test for zero length values yet.

	if (!pLeftBuf)
	{
		if (pRightBuf)
		{
			iCompare = -1;
		}
		goto Exit;
	}

	bHitWildCard = bHasWildCardPos = FALSE;
	uiLeadingSpace = uiTrailingSpace = 
		(uiFlags & FLM_MIN_SPACES) ? FLM_NO_SPACE : 0;
	pbHitWildCard = (uiFlags & FLM_WILD) ? &bHitWildCard : NULL;

	if (bLeadingWildCard)
	{
		goto Leading_Wild_Card;
	}

	while (!iCompare &&
			(uiLeftLen || uiLeftWpChar2) &&
			(uiRightLen || uiRightWpChar2))
	{
		iCompare = flmTextCompareSingleChar(
								&pLeftBuf, &uiLeftLen, &uiLeftWpChar2,
								&pRightBuf, &uiRightLen, &uiRightWpChar2,
								NULL_SUB_COL_CHECK, NULL_CASE_CHECK, pbHitWildCard, 
								iCompareType, NULL, 
								uiFlags | uiLeadingSpace, uiLang);

		uiLeadingSpace = 0;
		if (bHitWildCard)
		{
Leading_Wild_Card:
			bHitWildCard = FALSE;
			bHasWildCardPos = FALSE;		// Turn off last wildcard.

			// If right side is done, we are done.

			if (!uiRightLen && !uiRightWpChar2)
			{
				uiLeftLen = 0;
				uiLeftWpChar2 = 0;
				break;
			}

			// Save state on the RIGHT to handle the sick case of search key 
			// "b*aH" being able to match "baaaaaaaaaH" (Lambda Case)
			// LWCP = LastWildCardPosition

			pLWCPRightBuf = pRightBuf;
			uiLWCPRightLen = uiRightLen;
			uiLWCPRightWpChar2 = uiRightWpChar2;

			// Find first matching character on the left side.

Compare_Again:

			iCompare = -1;
			while (iCompare && (uiLeftLen || uiLeftWpChar2))
			{
				iCompare = flmTextCompareSingleChar(
								&pLeftBuf, &uiLeftLen, &uiLeftWpChar2,
								&pRightBuf, &uiRightLen, &uiRightWpChar2,
								NULL_SUB_COL_CHECK, NULL_CASE_CHECK, NULL_WILD_CARD_CHECK,
								iCompareType, NULL, 
								uiFlags | uiLeadingSpace, uiLang);
				
				uiLeadingSpace = 0;

				// Done with the right side?  Return iCompare value.

				if (!uiRightLen && !uiRightWpChar2)
				{
					break;
				}

				// Values different and still have stuff on left?

				if (iCompare && (uiLeftLen || uiLeftWpChar2))
				{
					// Advance the left if there is anything left
					uiValueLen = flmTextGetValue( pLeftBuf, uiLeftLen,
													&uiLeftWpChar2,
													uiFlags, &ui16Tmp1, &ui16Tmp2);
					pLeftBuf += uiValueLen;
					uiLeftLen -= uiValueLen;
				}
			}

			// Save state on the LEFT 

			if (uiLeftLen || uiLeftWpChar2)
			{
				pLWCPLeftBuf = pLeftBuf;
				uiLWCPLeftLen = uiLeftLen;
				uiLWCPLeftWpChar2 = uiLeftWpChar2;
				bHasWildCardPos = TRUE;
			}

			// EQUAL - as far as the collation values are concerned.
		}
	}

	if (iCompare == 0)
	{
		// In here because LEFT and/or RIGHT are out of bytes.
		// Check for trailing spaces if MIN_SPACES.

		if (uiLeftLen || uiLeftWpChar2)
		{
			if (!bTrailingWildCard)
			{
				uiLeftLen -= flmTextGetValue( pLeftBuf, uiLeftLen,
										&uiLeftWpChar2,
										uiFlags | uiTrailingSpace, &ui16WPChar,
										&ui16UniChar);
				
				if (uiLeftLen || ui16WPChar || ui16UniChar)
				{
					iCompare = 1;
				}
			}
		}
		else if (uiRightLen || uiRightWpChar2)
		{
			uiRightLen -= flmTextGetValue( pRightBuf, uiRightLen, &uiRightWpChar2,
									uiFlags | uiTrailingSpace, &ui16WPChar, &ui16UniChar);

			// Equals if right just had a trailing wild card. (else case)

			if (uiRightLen || !pbHitWildCard || ui16WPChar != '*')
			{				
				if (uiRightLen || ui16WPChar || ui16UniChar)
				{
					iCompare = -1;
				}
			}
		}
	}

	// Handle the embedded wild card case.

	if (iCompare != 0 && bHasWildCardPos)
	{

		// Restore wild card state.

		pLeftBuf = pLWCPLeftBuf;
		uiLeftLen = uiLWCPLeftLen;
		uiLeftWpChar2 = uiLWCPLeftWpChar2;
		pRightBuf = pLWCPRightBuf;
		uiRightLen = uiLWCPRightLen;
		uiRightWpChar2 = uiLWCPRightWpChar2;
		bHasWildCardPos = FALSE;

		goto Compare_Again;
	}

Exit:

	return (!iCompare ? FLM_TRUE : FLM_FALSE);
}

/****************************************************************************
Desc:  	Compare only the leading left and right characters according
			to the many flags that are passed in.  This routine operates
			to save and set state for the calling routine.
TODO:
			This routine does NOT support Asian, Hebrew, or Arabic language
			collations.  In addition, fwpCheckDoubleCollation() is not called for other non-US
			lanagues.  There is still a lot of work to do!  This is our
			default US compare and it is not very good for JP.

Return:	Signed value of compare.  
			<0 if less than, 0 if equal, >0 if greater than.
Asian Notes:
			The asian compare takes two characters and may use one or both.
			This makes the algorithm complex so we may have to build full
			tests to see what we broke.
NDS Notes:
			The right side (search string) is already formatted according
			to the space/dash rules of the syntax.  
****************************************************************************/
FSTATIC FLMINT	flmTextCompareSingleChar(
	FLMBYTE **		ppLeftText,		// [in] Points to current value.
											// [out] Points to next character if equals.
	FLMUINT *		puiLeftLen,		// [in] Bytes remaining in text string.
											// [out] Bytes remaining in text string.
	FLMUINT *		puiLeftWpChar2,// Second left character - for double characters
	FLMBYTE **		ppRightText,	// [in] Points to current value.
											// [out] Points to next character if equals.
	FLMUINT *		puiRightLen,	// [in] Bytes remaining in text string.
											// [out] Bytes remaining in text string.
	FLMUINT *		puiRightWpChar2,// Second right character - for double characters.
	FLMINT *			piSubColCompare,//[in] If NULL disregard the subcollation
											// values if collation values are equal.
											// [out] If equals is returned, value is
											// set ONLY if the signed value of comparing
											// the sub-collation values is not equal.
											// See lengthy unicode compare below.
	FLMINT *			piCaseCompare,	// [in] If NULL disregard the case bits
											// if collation values are equal.  Japanese
											// values are an exception to this rule.
											// [out] If equals is returned, value is
											// set ONLY if the signed value of comparing 
											// the case values is not equal.
	FLMBOOL *		pbHitWildCard,	// [in] If NULL then do not look for wild
											// cards in the right text string.
											// [out] If non-null, a wild card (*,?) will
											// be looked for on the RIGHT SIDE ONLY.
											// If '?' is found 0 will be returned and
											// pointers are advanced.  If '*' is found,
											// this value will be set to TRUE and the
											// right side is advanced.  If no wild 
											// card is found the value will not be set.
	FLMINT			iCompareType,	// COMPARE_COLLATION, COMPARE_COL_AND_SUBCOL, COMPARE_VALUE
	FLMUINT16 * 	pui16ColVal,	// Needed for asian collation compare.
	FLMUINT			uiFlags,			// FLM_* flags
	FLMUINT			uiLangId)		// FLAIM/WordPerfect Lanaguge id.
{
	FLMBYTE *		pLeftText = *ppLeftText;
	FLMBYTE *		pRightText = *ppRightText;
	FLMINT			iCompare = 0;
	FLMUINT			uiRightFlags = uiFlags;
	FLMUINT16		ui16LeftWPChar;
	FLMUINT16		ui16LeftUniChar;
	FLMUINT16		ui16RightWPChar;
	FLMUINT16		ui16RightUniChar;
	FLMUINT			uiLeftValueLen;
	FLMUINT			uiRightValueLen;
	FLMUINT16		ui16LeftCol;
	FLMUINT16		ui16RightCol;
	FLMUINT			uiLeftWpChar2 = *puiLeftWpChar2;
	FLMUINT			uiRightWpChar2 = *puiRightWpChar2;
	FLMBOOL			bLeftTwoIntoOne;
	FLMBOOL			bRightTwoIntoOne;

	// Get the next character from the TEXT string.  NOTE: OEM characters
	// will be returned as a UNICODE character.  A unicode character here
	// is a value that cannot be converted to the WP set (no good collation value)..

	uiLeftValueLen = flmTextGetValue( pLeftText, *puiLeftLen, &uiLeftWpChar2,
									uiFlags, &ui16LeftWPChar, &ui16LeftUniChar);
	uiRightValueLen = flmTextGetValue( pRightText, *puiRightLen, &uiRightWpChar2,
							uiRightFlags, &ui16RightWPChar, &ui16RightUniChar);

	// At this point, the double character, if any, should have been consumed.

	flmAssert( !uiLeftWpChar2 && !uiRightWpChar2);

	// Check for the following escape characters: "\\" "*" and "\\" "\\"

	if( ui16RightWPChar == ASCII_BACKSLASH)
	{
		if( pRightText[ uiRightValueLen ] == ASCII_BACKSLASH)
		{
			uiRightValueLen++;
		}
		else if( pRightText[ uiRightValueLen ] == ASCII_WILDCARD)
		{
			ui16RightWPChar = ASCII_WILDCARD;
			uiRightValueLen++;
		}
	}
	// Checking for wild cards in the right string? (Always a WP character)
	else if( pbHitWildCard)	
	{

		// The '*' wildcard means to match zero or many characters.
		// The sick case of "A*B" compared to "A**B" should be considered.

		if( ui16RightWPChar == ASCII_WILDCARD)
		{
			// Eat all duplicate wild cards.
			while( pRightText[ uiRightValueLen] == ASCII_WILDCARD)
			{
				uiRightValueLen++;
			}

			// Advance the right value.  Keep left value alone.  
			// Return equals (default).

			*pbHitWildCard = TRUE;

			// Don't advance the left value.

			uiLeftValueLen = 0;
			uiLeftWpChar2 = *puiLeftWpChar2;
			goto Exit;
		}
	}

	// First section is to compare just WP values.
	
	if( ui16LeftWPChar && ui16RightWPChar)
	{
		FLMUINT16	ui16LeftSubCol;
		FLMUINT16	ui16RightSubCol;

		if (iCompareType == COMPARE_VALUE)
		{

			// Check the obvious case of equal WP values.

			if( ui16LeftWPChar != ui16RightWPChar) 
			{
				iCompare = -1;
			}
			goto Exit;
		}

		// JP compare code.

		if (uiLangId >= FIRST_DBCS_LANG && uiLangId <= LAST_DBCS_LANG)
		{
			FLMUINT		uiNextLeftLen;
			FLMUINT		uiNextRightLen;
			FLMUINT16	ui16NextLeftWPChar;
			FLMUINT16	ui16NextRightWPChar;
			FLMUINT16	ui16ColVal = pui16ColVal ? *pui16ColVal : 0;
			FLMBYTE		ucLeftCaseValue;
			FLMBYTE		ucRightCaseValue;

			// Should have already consumed double character, if any

			flmAssert( !uiLeftWpChar2 && !uiRightWpChar2);
			uiNextLeftLen  = flmTextGetValue( pLeftText+uiLeftValueLen, 
										*puiLeftLen, &uiLeftWpChar2, uiFlags,
										&ui16NextLeftWPChar, &ui16LeftUniChar);
			uiNextRightLen = flmTextGetValue( pRightText+uiRightValueLen,
									*puiRightLen, &uiRightWpChar2, uiFlags,
									&ui16NextRightWPChar, &ui16RightUniChar);

			// nextL/R WPChar may be zero.

			if (fwpAsiaGetCollation( ui16LeftWPChar, ui16NextLeftWPChar, ui16ColVal,
							&ui16LeftCol, &ui16LeftSubCol, &ucLeftCaseValue, FALSE) == 2)
			{
				uiLeftValueLen += uiNextLeftLen;
			}
			if (fwpAsiaGetCollation( ui16RightWPChar, ui16NextRightWPChar, ui16ColVal,
							&ui16RightCol, &ui16RightSubCol, &ucRightCaseValue, FALSE) == 2)
			{
				uiRightValueLen += uiNextRightLen;
			}
			
			// Compare all of the stuff now.

			if (ui16LeftCol == ui16RightCol)
			{
				if( (iCompareType == COMPARE_COL_AND_SUBCOL) ||
					 (piSubColCompare && (*piSubColCompare == 0)))
				{
					if( ui16LeftSubCol != ui16RightSubCol)
					{
						if( iCompareType == COMPARE_COL_AND_SUBCOL)
						{	
							iCompare = -1;
							goto Exit;
						}

						// At this point piSubColCompare cannot be NULL.

						*piSubColCompare = (ui16LeftSubCol < ui16RightSubCol) ? -1 : 1;

						// Write over the case compare value

						if( piCaseCompare )
						{
							*piCaseCompare = *piSubColCompare;
						}
					}
				}
				if (iCompareType != COMPARE_COL_AND_SUBCOL)
				{

					// Check case?

					if (piCaseCompare && (*piCaseCompare == 0))
					{
						if( ucLeftCaseValue != ucRightCaseValue)
						{
							*piCaseCompare = ucLeftCaseValue < ucRightCaseValue?-1:1;
						}
					}
				}
			}
			else
			{
				iCompare = (ui16LeftCol < ui16RightCol) ? -1 : 1;
			}
			goto Exit;
		}

		flmAssert( !uiLeftWpChar2 && !uiRightWpChar2);

		if (uiLangId != US_LANG)
		{
			const FLMBYTE *	pucTmp;

			pucTmp = pLeftText + uiLeftValueLen;
			uiLeftWpChar2 = fwpCheckDoubleCollation( &ui16LeftWPChar, &bLeftTwoIntoOne,
										&pucTmp, uiLangId);
			uiLeftValueLen = (FLMUINT)(pucTmp - pLeftText);

			pucTmp = pRightText + uiRightValueLen;
			uiRightWpChar2 = fwpCheckDoubleCollation( &ui16RightWPChar, &bRightTwoIntoOne,
										&pucTmp, uiLangId);
			uiRightValueLen = (FLMUINT)(pucTmp - pRightText);

			// See if we got the same double character

			if (uiLeftWpChar2 == uiRightWpChar2 &&
				 ui16LeftWPChar == ui16RightWPChar)
			{
				uiLeftWpChar2 = 0;
				uiRightWpChar2 = 0;
				goto Exit;
			}
		}
		else if (ui16LeftWPChar == ui16RightWPChar)
		{

			// Same WP character

			goto Exit;
		}

		ui16LeftCol = fwpGetCollation( ui16LeftWPChar, uiLangId);

		// Handle two characters collating as one.

		if (uiLeftWpChar2 && bLeftTwoIntoOne)
		{
			ui16LeftCol++;
		}

		ui16RightCol = fwpGetCollation( ui16RightWPChar, uiLangId);

		// Handle two characters collating as one.

		if (uiRightWpChar2 && bRightTwoIntoOne)
		{
			ui16RightCol++;
		}

		if( ui16LeftCol == ui16RightCol)
		{
			// Should we bother to check subcollation? - don't bother with 7-bit

			if( (  (iCompareType == COMPARE_COL_AND_SUBCOL) 
				 || (piSubColCompare && (*piSubColCompare == 0))) 
			&&  ((ui16LeftWPChar | ui16RightWPChar) & 0xFF00))	// Non-ascii
			{
				ui16LeftSubCol = flmTextGetSubCol( ui16LeftWPChar, 
															ui16LeftCol, uiLangId);
				ui16RightSubCol= flmTextGetSubCol( ui16RightWPChar, 
															ui16RightCol, uiLangId);

				if (!piCaseCompare)
				{

					// If the sub-collation value is the original
					// character, it means that the collation could not
					// distinguish the characters and sub-collation is being
					// used to do it.  However, this creates a problem when the
					// characters are the same character except for case.  In that
					// scenario, we incorrectly return a not-equal when we are
					// doing a case-insensitive comparison.  So, at this point,
					// we need to use the sub-collation for the upper-case of the
					// character instead of the sub-collation for the character
					// itself.

					if (ui16LeftSubCol == ui16LeftWPChar)
					{
						ui16LeftSubCol = flmTextGetSubCol(
													fwpCh6Upper( ui16LeftWPChar),
													ui16LeftCol, uiLangId);
					}
					if (ui16RightSubCol == ui16RightWPChar)
					{
						ui16RightSubCol= flmTextGetSubCol(
													fwpCh6Upper( ui16RightWPChar),
													ui16RightCol, uiLangId);
					}
				}

				// YES - go for it...
				
				if( ui16LeftSubCol != ui16RightSubCol)
				{
					if( iCompareType == COMPARE_COL_AND_SUBCOL)
					{	
						iCompare = (ui16LeftSubCol < ui16RightSubCol) ? -1 : 1;
						goto Exit;
					}
					// At this point piSubColCompare cannot be NULL.
					*piSubColCompare = (ui16LeftSubCol < ui16RightSubCol) ? -1 : 1;
					/* Write over the case compare value */
					if( piCaseCompare )
					{
						*piCaseCompare = *piSubColCompare;
					}
				}
				// ? goto Exit???
			}

			if( iCompareType == COMPARE_COL_AND_SUBCOL)
			{
				goto Exit;
			}
			// Check case?

			if( piCaseCompare && (*piCaseCompare == 0))
			{

				// fwpIsUpper() only returns FALSE (lower) or TRUE (not-lower)

				FLMBOOL	bLeftUpper = fwpIsUpper( ui16LeftWPChar);
				FLMBOOL	bRightUpper = fwpIsUpper( ui16RightWPChar);

				if (bLeftUpper != bRightUpper)
				{
					*piCaseCompare = !bLeftUpper ? -1 : 1;
				}
				// ? else - don't know why they would be the same.
			}
		}
		else
		{
			iCompare = (ui16LeftCol < ui16RightCol) ? -1 : 1;
		}
		goto Exit;

	}	// end of working with BOTH WP characters

	/*else*/
	if( ui16LeftUniChar && ui16RightUniChar)
	{
		// Compare two (non-convertable) UNICODE values.

		// Check the obvious case of equal UNICODE values.
		if( ui16LeftUniChar == ui16RightUniChar)
		{
			goto Exit;
		}

		// Compare subcollation or compare value?
		if( iCompareType != COMPARE_COLLATION) 
		{
			iCompare = -1;
			goto Exit;
		}

		/*
		For non-asian - we store these values in the sub-collcation area.
		We should return the differece in sub-collation values - but this 
		may not work for all compares.
		For asian compares, most values we have a collation value.  
		This is a BIG differece in comparing asian values.

		If we want sub-collation compare then set it, otherwise set main
		iCompare value.
		*/

		if( piSubColCompare )
		{
			if( *piSubColCompare == 0)
			{
				*piSubColCompare = ui16LeftUniChar < ui16RightUniChar ? -1 : 1;
			}
		}
		else
		{
			// Treat as the collation value - this is different than the index.

			iCompare = ui16LeftUniChar < ui16RightUniChar ? -1 : 1;
		}
		goto Exit;
	}
	/*else*/

	// Compare subcollation or compare value?
	if( iCompareType != COMPARE_COLLATION) 
	{
		iCompare = -1;
		goto Exit;
	}

	// Check for no left character.
	if( !ui16LeftWPChar && !ui16LeftUniChar)
	{
		// No left character.  check if no right character.

		if( ui16RightWPChar || ui16RightUniChar)
		{
			iCompare = -1;
		}
		/* else returns equals. */
	}

	// Check for no right character.
	else if( !ui16RightWPChar && !ui16RightUniChar)
	{
		iCompare = 1;
	}

	/*
	What remains is one WP char and one Unicode char.
	Remember the sub-collation comment above.  Some WP char may not
	have a collation value (COLS0) so in US sort these values may be
	equal and have different sub-collation values.  YECH!!!!

	The unicode value will always have collation value of COLS0 (0xFF)
	and subcollation value of 11110 [unicodeValue]
	The WP value could be anything & if collation value is COLS0 will
	have a subcollation value os 1110 [WPValue]
	
	So, we have to check to see of the WP collation value is COLS0.  
	If not iCompare is used.  If both represent high collation then
	the WP value will always have a lower sub-collation value.

	The (not so obvious) code would be to code up...
	iCompare = ui16LeftWPChar ? -1 : 1;
	if we didn't care about sub-collation (and we may not care).
	
	This is easier to over code than have ?: operators for the two cases.
	*/

	else if( ui16LeftWPChar)
	{
		// Remember - unicode subcol is always COLS0.

		if( fwpGetCollation( ui16LeftWPChar, uiLangId) == COLS0)
		{
			if( piSubColCompare && (*piSubColCompare == 0))
			{
				*piSubColCompare = -1;
			}
		}
		else
		{
			iCompare = -1;
		}
	}
	else
	{
		// left=unicode, right=WP
		// Remember - unicode subcol is always COLS0 for non-asian.

		if( fwpGetCollation( ui16RightWPChar, uiLangId) == COLS0)
		{
			if( piSubColCompare && (*piSubColCompare == 0))
			{
				*piSubColCompare = 1;
			}
		}
		else
		{
			iCompare = 1;
		}
	}
Exit:

	if( !iCompare )
	{
		// Position to the next values if equal

		*puiLeftLen -= uiLeftValueLen;
		*ppLeftText  = pLeftText + uiLeftValueLen;
		*puiLeftWpChar2 = uiLeftWpChar2;
		*puiRightLen -= uiRightValueLen;
		*ppRightText = pRightText + uiRightValueLen;
		*puiRightWpChar2 = uiRightWpChar2;
	}
	return iCompare;
}

/****************************************************************************
Desc:  	Return the next WP or unicode character value and parsing type.
****************************************************************************/
FLMUINT flmTextGetCharType(
	const FLMBYTE *	pText,
	FLMUINT 				uiLen,
	FLMUINT16 *			pui16WPValue,
	FLMUNICODE *		puzUniValue, 
	FLMUINT	 *			pType)	
{
	FLMUINT				uiReturnLen;
	FLMUINT16			wpValue;
	FLMUNICODE			uniValue;
	FLMUINT				uiCharSet;

	uiReturnLen = flmTextGetValue( pText, uiLen, NULL,
		FLM_MIN_SPACES, pui16WPValue, puzUniValue);
	wpValue = *pui16WPValue;
	uniValue = *puzUniValue;

	if( wpValue)
	{
		if( wpValue < 0x080)
		{
			*pType = flmCharTypeAnsi7( wpValue);
			goto Exit;
		}
		uiCharSet = (FLMUINT) (wpValue >> 8);
		
		if( uiCharSet == 1 ||
			 uiCharSet == 2 ||
			 (uiCharSet >= 8 && uiCharSet <= 11))
		{
			*pType = SDWD_CHR;
			goto Exit;
		}
		*pType = DELI_CHR;
	}
	else
	{
		// For now all unicode is a delimeter
		*pType = DELI_CHR;
	}
Exit:
	return uiReturnLen;
}

/****************************************************************************
Desc:  	Return the next WP or unicode character value.
Return:	Number of bytes formatted to return the character value.
Note:		This code must be fast so some compromises have been made
			in respect to maintenance.
			DON"T CHEAT.  This routine returns the number of spaces
			skipped over if FLM_MIN_SPACE or FLM_NO_SPACE is turned on.
			White space checking does NOT applity to WP spaces.  Only
			to the 0x20 space.
****************************************************************************/
FLMUINT flmTextGetValue(
	const FLMBYTE *		pText,			// [in] Points to current value.
	FLMUINT 					uiLen,			// [in] Bytes remaining in text.
	FLMUINT *				puiWpChar2,		// Was there a double character?
	FLMUINT					uiFlags,			// [in] 
	FLMUINT16 *				pui16WPValue,	// [out] WP Character value or 0 if unicode.
	FLMUNICODE *			puzUniValue)	// [out] Unicode or OEM value if 
													// *pui16WPChar is zero.
{
	FLMUINT			uiReturnLength = 0;
	FLMUINT			uiObjectLength;
	FLMUINT16		ui16CurValue;		// Current working (WPish) value.
	FLMUNICODE		uzUniValue;

	uiReturnLength = 0;
	ui16CurValue = 0;
	uzUniValue = 0;

	if (puiWpChar2 && *puiWpChar2)
	{
		ui16CurValue = (FLMUINT16)(*puiWpChar2);
		*puiWpChar2 = 0;
		uiObjectLength = 0;
		goto Check_White_Space;
	}

	while (uiLen && !ui16CurValue && !uzUniValue)
	{
		ui16CurValue = (FLMUINT16) *pText;

		switch( GedTextObjType( ui16CurValue ))
		{
			case ASCII_CHAR_CODE:  			/* 0nnnnnnn */
				uiObjectLength = 1;

Check_White_Space:

				// Do all of the bIgnore* stuff here.  
				// WHITE SPACE CODE doesn't apply.

				if( ui16CurValue == (FLMUINT16) ASCII_UNDERSCORE && (uiFlags & FLM_NO_UNDERSCORE))
				{
					ui16CurValue = (FLMUINT16) ASCII_SPACE;
				}
				if( ui16CurValue == (FLMUINT16) ASCII_SPACE)
				{
					if( uiFlags & FLM_NO_SPACE)
					{
						ui16CurValue = 0;
					}
					else if( uiFlags & FLM_MIN_SPACES)
					{
						// Eat up the remaining spaces and underscores (if NO_UNDERSCORES).
						while( (pText[ uiObjectLength] == ASCII_SPACE
							  || (   pText[ uiObjectLength] == ASCII_UNDERSCORE 
								  && (uiFlags & FLM_NO_UNDERSCORE)))
						 &&  uiObjectLength < uiLen)
						{
							uiObjectLength++;
						}
					}		
				}
				else if( ui16CurValue == ASCII_DASH && (uiFlags & FLM_NO_DASH))
				{
					ui16CurValue = 0;
				}
				break;

			case CHAR_SET_CODE:	  			/* 10nnnnnn - Character Set | Char */
				uiObjectLength = 2;
				ui16CurValue = (FLMUINT16)
							(((FLMUINT16)(ui16CurValue & (~CHAR_SET_MASK)) << 8)	
							+ (FLMUINT16)*(pText + 1));	/* Character */
				break;

			case WHITE_SPACE_CODE:			/* 110nnnnn */
			{
				FLMBYTE		ucTmpByte;

				uiObjectLength = 1;
				ucTmpByte = *pText & (~WHITE_SPACE_MASK);

				ui16CurValue = ((ucTmpByte == HARD_HYPHEN) ||
								  (ucTmpByte == HARD_HYPHEN_EOL) ||
								  (ucTmpByte == HARD_HYPHEN_EOP))
								? 	(FLMUINT16) 0x2D 		/* Minus sign */
								:	(FLMUINT16) 0x20;		/* Space */
				break;
			}

			case EXT_CHAR_CODE:				/* Full extended character */
				uiObjectLength = 3;
				ui16CurValue = (FLMUINT16)(((FLMUINT16)*(pText + 1) << 8)	/* Char set */
							  + (FLMUINT16) *(pText + 2));					/* Character */
				break;

			case UNICODE_CODE:				/* Unconvertable UNICODE code */
	
				uiObjectLength = 3;
				ui16CurValue = 0;
				uzUniValue = (FLMUINT16)(((FLMUINT16)*(pText + 1) << 8)	/* Char set */
							  + (FLMUINT16)*(pText + 2));		/* Character */
				break;

			case OEM_CODE:
				uiObjectLength = 2;			/* OEM characters are always >= 128.*/
													/* Make this a unicode character */
				ui16CurValue = 0;
				uzUniValue = (FLMUINT16) *(pText + 1);
				break;


			/* Skip all of the unknown stuff */
			case UNK_GT_255_CODE:
				uiObjectLength = (FLMUINT16)(1 + sizeof( FLMUINT16) + FB2UW( pText + 1));
				break;
			case UNK_LE_255_CODE:
				uiObjectLength = 2 + (FLMUINT16)*(pText + 1);
				break;
			case UNK_EQ_1_CODE:
				uiObjectLength = 2;
				break;
			default:							/* should NEVER happen: bug if does */
												/* Coded to skip remaining data. */
				ui16CurValue = 0;
				uiObjectLength = uiLen;
				break;						/* just give up. */
		}	/* End of switch */

		uiReturnLength += uiObjectLength;
		pText += uiObjectLength;
		uiLen -= uiObjectLength;
	}

//Exit:
	*pui16WPValue = ui16CurValue;
	*puzUniValue = uzUniValue;
	return uiReturnLength;
}


/****************************************************************************
Desc:  	Return the sub-collation value of a WPText character.
			Unconvered Unicode values always have a sub-collation
			value of 11110 + Unicode Value.
****************************************************************************/

FSTATIC FLMUINT16	flmTextGetSubCol(
	FLMUINT16		ui16WPValue,		// [in] WP Character value.
	FLMUINT16		ui16ColValue,		// [in] Collation Value (for arabic)
	FLMUINT			uiLangId)		// [in] WP Language ID.
{
	FLMUINT16		ui16SubColVal;
	FLMBYTE			byCharVal;
	FLMBYTE			byCharSet;
	FLMUINT16		ui16Base;

	// Easy case first.

	ui16SubColVal = 0;
	if( (ui16WPValue & 0xFF00 ) == 0)
	{
		goto Exit;
	}

	// From here down default ui16SubColVal is WP value.

	ui16SubColVal = ui16WPValue;
	
	byCharVal = (FLMBYTE) ui16WPValue;
	byCharSet = (FLMBYTE) (ui16WPValue >> 8);
	
	/**--------------------------------------------------
	***  Convert char to uppercase because case information
	***  is stored above.  This will help
	***  insure that the "ETA" doesn't sort before "eta"
	***  could use is lower code here for added performance.
	***-------------------------------------------------*/
	
	/* This just happens to work with all WP character values. */
	if (!fwpIsUpper( ui16WPValue))
	{
		ui16WPValue &= ~1;
	}

	switch( byCharSet)
	{
		case	CHSMUL1:
			/**--------------------------------------------------
			***  If you cannot break down a char into base and
			***  diacritic then you cannot combine the charaacter
			***  later when converting back the key.  So, write
			***  the entire WP char in the sub-collation area.
			***  We can ONLY SUPPORT MULTINATIONAL 1 for brkcar()
			***-------------------------------------------------*/

			if( fwpCh6Brkcar( ui16WPValue, &ui16Base, &ui16SubColVal))
			{

				// WordPerfect character cannot be broken down.
				// If we had a collation value other than 0xFF (COLS0), don't
				// return a sub-collation value.  This will allow things like
				// upper and lower AE digraphs to compare properly.

				if (ui16ColValue != COLS0)
				{
					ui16SubColVal = 0;
				}
				goto Exit;
			}
			
			/**-------------------------------------------------
			*** Write the FLAIM diacritic sub-collation value.
			*** Prefix is 2 bits "10".  Remember to leave
			*** "111" alone for the future.
			*** Bug 11/16/92 = was only writing a "1" and not "10" 
			***------------------------------------------------*/
			ui16SubColVal = (
					(ui16SubColVal & 0xFF) == umlaut		/* Def in charset.h */
					&& ( (uiLangId == SU_LANG) || 
						  (uiLangId == SV_LANG) || 
						  (uiLangId == CZ_LANG) || 
						  (uiLangId == SL_LANG) 
						)
					)
				?	(FLMUINT16)(fwp_dia60Tbl[ ring] + 1)	/* umlaut must be after ring above*/
				:	(FLMUINT16)(fwp_dia60Tbl[ ui16SubColVal & 0xFF]);
	
			break;

		case	CHSGREK:
			/**------------
			***  Greek
			***-----------*/
			if( (byCharVal >= 52)  ||	/* Keep case bit for 52-69 else ignore*/
          	 (ui16WPValue == 0x804) ||	/*[ 8,4] BETA Medial | Terminal*/
				 (ui16WPValue == 0x826)) 	/*[ 8,38] SIGMA termainal */
			{
				ui16SubColVal = ui16WPValue;
			}
			/* else no subcollation to worry about */
			break;
			
		case	CHSCYR:
			if( byCharVal >= 144)
			{
				ui16SubColVal = ui16WPValue;
			}
			/* else no subcollation to worry about */

			/* VISIT: Georgian covers 208-249 - no collation defined yet */
			break;
			
		case	CHSHEB:					/* Hebrew */
			/**-----------------------------------------------------------
			***  Three sections in Hebrew:
			***   0..26 - main characters
			***  27..83 - accents that apear over previous character
			***  84..118- dagesh (ancient) hebrew with accents
			***
			***  Because the ancient is only used for sayings & scriptures
			***  we will support a collation value and in the sub-collation
			***  store the actual character because sub-collation is in 
			***  character order.
			***----------------------------------------------------------*/

         if( byCharVal >= 84) 		/* Save ancient - value 84 and above */
			{
				ui16SubColVal = ui16WPValue;
			}
			break;
			
		case	CHSARB1:					/* Arabic 1 */
			/**-------------------------------------------------------
			***  Three sections in Arabic:						
			***  00..37  - accents that display OVER a previous character
			***  38..46  - symbols 
			***  47..57  - numbers
			***  58..163 - characters
			***  164     - hamzah accent
			***  165..180- common characters with accents
			***  181..193- ligatures - common character combinations
			***  194..195- extensions - throw away when sorting
			***------------------------------------------------------*/
		
			if( byCharVal <= 46 )
			{
				ui16SubColVal = ui16WPValue;
			}
			else
			{
				if( ui16ColValue == COLS10a+1)	/* Alef? */
				{	
					ui16SubColVal = (byCharVal >= 165)
						? (FLMUINT16)(fwp_alefSubColTbl[ byCharVal - 165 ])
						: (FLMUINT16)7;							/* Alef subcol value */
				}
				else
				{
					if( byCharVal >= 181)				/* Ligatures - char combination*/
					{
						ui16SubColVal = ui16WPValue;
					}
					else if( byCharVal == 64)			/* taa exception */
					{
						ui16SubColVal = 8;
					}
				}
			}
			break;
			
		case	CHSARB2:					/* Arabic 2 */
			/* There are some characters that share the same slot */
			/* Check the bit table if above character 64 */
			
			if ((byCharVal >= 64) &&
				 (fwp_ar2BitTbl[(byCharVal-64)>> 3] & (0x80 >> (byCharVal&0x07))))
			{
				ui16SubColVal = ui16WPValue;
			}	
			break;

	} /* end switch  */

Exit:
	return ui16SubColVal;
}
