//-------------------------------------------------------------------------
// Desc:	Eachword/substring parsing for eachword/substring indexing.
// Tabs:	3
//
//		Copyright (c) 1990-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kyeword.cpp 12313 2006-01-19 15:14:44 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	Substring-ize the string in a node.  Normalize spaces and hyphens if
		told to.  Example: ABC  DEF
			ABC DEF
			BC DEF
			C DEF
			DEF
VISIT: This needs a lot of word to decide what to do with Kanji and
		word joining charactings.  Need to use the routines in fqtextc.cpp
		to determine the character type.
****************************************************************************/
FLMBOOL KYSubstringParse(
	const FLMBYTE **	ppText,				// [in][out] points to text
	FLMUINT  *	 		puiTextLen,			// [in][out] length of text
	FLMUINT				uiIfdFlags,			// [in] flags
	FLMUINT				uiLimitParm,		// [in] Max characters
	FLMBYTE *	  		pKeyBuf,				// [out] key buffer to fill
  	FLMUINT *			puiKeyLen)			// [out] returns length
{
	const FLMBYTE *	pText = *ppText;
	FLMUINT				uiLen = *puiTextLen;
	FLMUINT				uiWordLen = 0;
	FLMUINT				uiLimit = uiLimitParm ? uiLimitParm : IFD_DEFAULT_SUBSTRING_LIMIT;
	FLMUINT				uiFlags = 0;
	FLMUINT				uiLeadingSpace = FLM_NO_SPACE;

	FLMBOOL				bIgnoreSpaceDefault = (uiIfdFlags & IFD_NO_SPACE) ? TRUE : FALSE;
	FLMBOOL				bIgnoreSpace = TRUE;
	FLMBOOL				bIgnoreDash   = (uiIfdFlags & IFD_NO_DASH) ? TRUE : FALSE;
	FLMBOOL				bMinSpaces    = (uiIfdFlags & (IFD_MIN_SPACES | IFD_NO_SPACE)) ? TRUE : FALSE;
	FLMBOOL				bNoUnderscore = (uiIfdFlags & IFD_NO_UNDERSCORE) ? TRUE : FALSE;
	FLMBOOL				bFirstCharacter = TRUE;

	// Set uiFlags
	if( bIgnoreSpaceDefault)
		uiFlags |= FLM_NO_SPACE;
	if( bIgnoreDash)
		uiFlags |= FLM_NO_DASH;
	if( bNoUnderscore)
		uiFlags |= FLM_NO_UNDERSCORE;
	if( uiIfdFlags & IFD_MIN_SPACES)
		uiFlags |= FLM_MIN_SPACES;

	/*
	The limit must return one more than requested in order
	for the text to collation routine to set the truncated flag.
	*/
	uiLimit++;

	while( uiLen && uiLimit--)
	{
		FLMBYTE			ch = *pText;
		FLMUINT16		ui16WPValue;
		FLMUNICODE		ui16UniValue;
		FLMUINT			uiCharLen;

		if( (ch & ASCII_CHAR_MASK) == ASCII_CHAR_CODE)
		{
			if( ch == ASCII_UNDERSCORE && bNoUnderscore)
			{
				ch = ASCII_SPACE;
			}
			if( ch == ASCII_SPACE && bMinSpaces)
			{
				if( !bIgnoreSpace)
				{
					pKeyBuf[ uiWordLen++ ] = ASCII_SPACE;
				}
				bIgnoreSpace = TRUE;
				pText++;
				uiLen--;
				continue;
			}
			ui16WPValue = (FLMUINT16) ch;
			uiCharLen = 1;
		}
		else
		{
			if( (uiCharLen = flmTextGetValue( pText, uiLen, NULL,
									uiFlags | uiLeadingSpace, 
									&ui16WPValue, &ui16UniValue)) == 0)
				break;
			flmAssert( uiCharLen <= uiLen);
		}
		uiLeadingSpace = 0;
		bIgnoreSpace = bIgnoreSpaceDefault;
		uiLen -= uiCharLen;
		while( uiCharLen--)
		{
			pKeyBuf[ uiWordLen++ ] = *pText++;
		}

		// If on the first word position to start on next character
		// for the next call.
		if( bFirstCharacter)
		{
			bFirstCharacter = FALSE;
			// First character - set return value.
			*ppText = pText;
			*puiTextLen = uiLen;
		}
	}
	pKeyBuf[ uiWordLen ] = '\0';
	// Case of all spaces - the FALSE will trigger indexing is done.
	*puiKeyLen = (FLMUINT)uiWordLen;
	return( ( uiWordLen) ? TRUE : FALSE);
}

/****************************************************************************
Desc:	Keyword-ize the information in a node - node is assumed to be a
		TEXT node.
VISIT: This needs a lot of work to decide what to do with Kanji and
		word joining charactings.  Need to use the routines in fqtextc.cpp
		to determine the character type.  Also, the code should be redone to
		be like the substring code above instead of count the buffer.
****************************************************************************/
FLMBOOL KYEachWordParse(
	const FLMBYTE **	pText,
	FLMUINT  *	 		puiTextLen,
	FLMUINT				uiLimitParm,		// [in] Max characters
	FLMBYTE *	  		pKeyBuf,				// [out] Buffer of at least MAX_KEY_SIZ
  	FLMUINT  *			puiKeyLen)
{
	const FLMBYTE *	pKey = NULL;
	const FLMBYTE *	pTmpKey;
	FLMUINT				uiLimit = uiLimitParm ? uiLimitParm : IFD_DEFAULT_SUBSTRING_LIMIT;
	FLMUINT				uiLen;
	FLMUINT				uiBytesProcessed = 0;
	FLMBOOL				bSkippingDelim = TRUE;
	FLMBOOL				bHaveWord = FALSE;
	FLMUINT				uiWordLen = 0;
	FLMUINT16			ui16WPValue;
	FLMUNICODE			ui16UniValue;
	FLMUINT				uiCharLen;
	FLMUINT				uiType;

	uiLen = *puiTextLen;
	pTmpKey = *pText;
	while ((uiBytesProcessed < uiLen) && (!bHaveWord) && uiLimit)
	{
		uiCharLen = flmTextGetCharType( pTmpKey, uiLen,
									&ui16WPValue, &ui16UniValue, &uiType);

		/* Determine how to handle what we got. */

		if (bSkippingDelim)
		{

			/*
			If we were skipping delimiters, and we run into a non-delimiter
			character, set the bSkippingDelim flag to FALSE to indicate the
			beginning of a word.
			*/

			if (uiType & SDWD_CHR)
			{
				pKey = pTmpKey;
				uiWordLen = uiCharLen;
				bSkippingDelim = FALSE;
				uiLimit--;
			}
		}
		else
		{

			/*
			If we were NOT skipping delimiters, and we run into a delimiter
			output the word.
			*/

			if (uiType & (DELI_CHR | WDJN_CHR))
				bHaveWord = TRUE;
			else
			{
				uiWordLen += uiCharLen;
				uiLimit--;
			}
		}

		/* Increment str to skip past what we are pointing at. */

		pTmpKey += uiCharLen;
		uiBytesProcessed += uiCharLen;
	}

	*pText = pTmpKey;
	*puiTextLen -= uiBytesProcessed;

	/* Return the word, if any. */

	if (uiWordLen)
	{
		*puiKeyLen = uiWordLen;
		f_memcpy( pKeyBuf, pKey, uiWordLen);
	}

	return( ( uiWordLen) ? TRUE : FALSE);
}
