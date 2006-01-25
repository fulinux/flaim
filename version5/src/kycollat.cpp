//------------------------------------------------------------------------------
// Desc:	Index collation routines
//
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
// $Id: kycollat.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE KYFormatUTF8Text(
	IF_PosIStream *	pIStream,
	FLMUINT				uiFlags,
	FLMUINT				uiCompareRules,
	F_DynaBuf *			pDynaBuf);

FSTATIC FLMBOOL flmAddMetaphone(
	const char *	pszStr,
	const char *	pszAltStr,
	FLMBYTE *		pszMeta,
	FLMUINT *		puiMetaOffset,
	FLMBYTE *		pszAltMeta,
	FLMUINT *		puiAltMetaOffset);

FSTATIC void flmMetaStrToNum(
	FLMBYTE *		pszMeta,
	FLMUINT *		puiMeta);

#ifdef FLM_DEBUG
	typedef struct
	{
		const char *	pszWord;
		FLMUINT			uiMeta;
		FLMUINT			uiAltMeta;
	} METAPHONE_MAPPING;

	static METAPHONE_MAPPING gv_MetaTestTable[] =
	{
		{ "ghislane",				0x4680, 0x4680 },
		{ "ghiradelli",			0x4AC6, 0x4AC6 },
		{ "hugh",					0x3000, 0x3000 },
		{ "san francisco",		0xB82A, 0xB82A },
		{ "van wagner",			0x2858, 0x2858 },
		{ "vanwagner",				0x2858, 0x2858 },
		{ "gnome",					0x8700, 0x8700 },
		{ "write",					0xAC00, 0xAC00 },
		{ "dumb",					0xC700, 0xC700 },
		{ "caesar",					0xBBA0, 0xBBA0 },
		{ "chianti",				0x58C0, 0x58C0 },
		{ "michael",				0x7560, 0x7D60 },
		{ "chemistry",				0x57BC, 0x57BC },
		{ "chorus",					0x5AB0, 0x5AB0 },
		{ "mchugh",					0x7500, 0x7500 },
		{ "czerny",					0xBA80, 0xDA80 },
		{ "focaccia",				0x25D0, 0x25D0 },
		{ "mcclellan",				0x7566, 0x7566 },
		{ "bellocchio",			0x96D0, 0x96D0 },
		{ "bacchus",				0x95B0, 0x95B0 },
		{ "accident",				0x15BC, 0x15BC },
		{ "accede",					0x15BC, 0x15BC },
		{ "succeed",				0xB5BC, 0xB5BC },
		{ "bacci",					0x9D00, 0x9D00 },
		{ "mac caffrey",			0x752A, 0x752A },
		{ "edge",					0x1400, 0x1400 },
		{ "edgar",					0x1C5A, 0x1C5A },
		{ "laugh",					0x6200, 0x6200 },
		{ "caugh",					0x5200, 0x5200 },
		{ "cagney",					0x5580, 0x5580 },
		{ "tagliaro",				0xC56A, 0xC6A0 },
		{ "biaggi",					0x9400, 0x9500 },
		{ "jose",					0x3B00, 0x3B00 },
		{ "yankelovich",			0x1856, 0x1856 },
		{ "bajador",				0x94CA, 0x93CA },
		{ "cabrillo",				0x59A6, 0x59A0 },
		{ "campbell",				0x5796, 0x5796 },
		{ "rogier",					0xA400, 0xA4A0 },
		{ "hochmeier",				0x357A, 0x357A },
		{ "island",					0x168C, 0x168C },
		{ "isle",					0x1600, 0x1600 },
		{ "sugar",					0xD5A0, 0xB5A0 },
		{ "herb",					0x3A90, 0x3A90 },
		{ "mannheim",				0x7870, 0x7870 },
		{ "snider",					0xB8CA, 0xD8CA },
		{ "schneider",				0xD8CA, 0xB8CA },
		{ "smith",					0xB700, 0xD7C0 },
		{ "schmidt",				0xD7C0, 0xB7C0 },
		{ "school",					0xB560, 0xB560 },
		{ "schenker",				0xD85A, 0xB585 },
		{ "resnais",				0xAB80, 0xAB8B },
		{ "artois",					0x1AC0, 0x1ACB },
		{ "celebration",			0xB69A, 0xB69A },
		{ "thomas",					0xC7B0, 0xC7B0 },
		{ "uomo",					0x1700, 0x1700 },
		{ "womo",					0x1700, 0x2700 },
		{ "arnow",					0x1A80, 0x1A82 },
		{ "arnoff",					0x1A82, 0x1A82 },
		{ "filipowicz",			0x269C, 0x2692 },
		{ "breaux",					0x9A00, 0x9A00 },
		{ "zhao",					0x4000, 0x4000 },
		{ NULL,						0x0000, 0x0000 }
	};
#endif

/****************************************************************************
Desc:	Build a collated key value piece.
****************************************************************************/
RCODE KYCollateValue(
	FLMBYTE *			pucDest,
	FLMUINT *			puiDestLen,
	IF_PosIStream *	pIStream,
	FLMUINT				uiDataType,
	FLMUINT				uiFlags,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLimit,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiLuLen,
	FLMUINT				uiLanguage,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bDataTruncated,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbOriginalCharsLost)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiDestLen;
	F_BufferIStream	bufferStream;
	FLMUINT				uiCharLimit;
	FLMUINT				uiLength;
	FLMBYTE *			pucTmpDest;
	FLMUINT				uiBytesRead;
	FLMBOOL				bHaveData = TRUE;
	FLMUNICODE			uChar;
	FLMBYTE				ucDynaBuf[ 64];
	F_DynaBuf			dynaBuf( ucDynaBuf, sizeof( ucDynaBuf));

	if (puiLuLen)
	{
		*puiLuLen = 0;
	}

	if ((uiDestLen = *puiDestLen) == 0)
	{
		rc = RC_SET( NE_XFLM_KEY_OVERFLOW);
		goto Exit;
	}

	if (uiDataType != XFLM_TEXT_TYPE)
	{
		if( !pIStream->remainingSize())
		{
			bHaveData = FALSE;
		}
	}
	else
	{
		FLMUINT64	ui64SavePosition = pIStream->getCurrPosition();

		if( RC_BAD( rc = flmReadUTF8CharAsUnicode( 
			pIStream, &uChar)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				bHaveData = FALSE;
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
		{
			goto Exit;
		}

		// The text is expected to be 0-terminated UTF-8

		if ((uiFlags & ICD_ESC_CHAR) ||
			 (uiCompareRules &
				(XFLM_COMP_COMPRESS_WHITESPACE |
				 XFLM_COMP_NO_WHITESPACE |
				 XFLM_COMP_NO_UNDERSCORES |
				 XFLM_COMP_NO_DASHES |
				 XFLM_COMP_WHITESPACE_AS_SPACE |
				 XFLM_COMP_IGNORE_LEADING_SPACE |
				 XFLM_COMP_IGNORE_TRAILING_SPACE)))
		{
			dynaBuf.truncateData( 0);
			if (RC_BAD( rc = KYFormatUTF8Text( pIStream,
					uiFlags, uiCompareRules, &dynaBuf)))
			{
				goto Exit;
			}
			if (RC_BAD( rc = bufferStream.open( dynaBuf.getBufferPtr(),
											dynaBuf.getDataLength())))
			{
				goto Exit;
			}
			pIStream = &bufferStream;
		}

		uiCharLimit = uiLimit ? uiLimit : ICD_DEFAULT_LIMIT;

		if( (uiLanguage >= FIRST_DBCS_LANG ) && (uiLanguage <= LAST_DBCS_LANG))
		{
			if( RC_BAD( rc = flmAsiaUTF8ToColText( pIStream, pucDest, &uiDestLen,
								(uiCompareRules & XFLM_COMP_CASE_INSENSITIVE)
								? TRUE
								: FALSE,
								puiCollationLen, puiLuLen,
								uiCharLimit, bFirstSubstring,
								bDataTruncated, pbDataTruncated)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = flmUTF8ToColText( pIStream, pucDest, &uiDestLen,
								(uiCompareRules & XFLM_COMP_CASE_INSENSITIVE)
								? TRUE
								: FALSE,
								puiCollationLen, puiLuLen,
								uiLanguage, uiCharLimit, bFirstSubstring,
								bDataTruncated,
								pbOriginalCharsLost, pbDataTruncated)))
			{
				goto Exit;
			}
		}
	}

	// TRICKY: uiDestLen could be set to zero if text and no value.

	if (!bHaveData || !uiDestLen)
	{
		uiDestLen = 0;
		goto Exit;
	}

 	switch (uiDataType)
	{
		case XFLM_TEXT_TYPE:
			break;

		case XFLM_NUMBER_TYPE:
		{
			FLMBYTE	ucTmpBuf [FLM_MAX_NUM_BUF_SIZE];
			
			uiLength = (FLMUINT)pIStream->remainingSize();
			
			flmAssert( uiLength <= sizeof( ucTmpBuf));

			if (RC_BAD( rc = pIStream->read( ucTmpBuf, uiLength, &uiBytesRead)))
			{
				goto Exit;
			}
			flmAssert( uiBytesRead == uiLength);
			if (RC_BAD( rc = flmStorageNum2CollationNum( ucTmpBuf,
										uiBytesRead, pucDest, &uiDestLen)))
			{
				goto Exit;
			}
			break;
		}

		case XFLM_BINARY_TYPE:
		{
			uiLength = (FLMUINT)pIStream->remainingSize();
			pucTmpDest = pucDest;

			if (uiLength >= uiLimit)
			{
				uiLength = uiLimit;
				bDataTruncated = TRUE;
			}

			// We don't want any single key piece to "pig out" more
			// than 256 bytes of the key

			if (uiDestLen > 256)
			{
				uiDestLen = 256;
			}

			if (uiLength > uiDestLen)
			{

				// Compute length so will not overflow

				uiLength = uiDestLen;
				bDataTruncated = TRUE;
			}
			else
			{
				uiDestLen = uiLength;
			}

			// Store as is.

			if (RC_BAD( rc = pIStream->read( pucTmpDest, uiDestLen, &uiBytesRead)))
			{
				goto Exit;
			}

			if (bDataTruncated && pbDataTruncated)
			{
				*pbDataTruncated = TRUE;
			}
			break;
		}

		default:
		{
			rc = RC_SET( NE_XFLM_CANNOT_INDEX_DATA_TYPE);
			break;
		}
	}

Exit:

	*puiDestLen = uiDestLen;
	return( rc);
}

/****************************************************************************
Desc:		Format text removing leading and trailing spaces.  Treat
			underscores as spaces.  As options, remove all spaces and dashes.
Ret:		NE_XFLM_OK always.  WIll truncate so text will fill MAX_KEY_SIZ.
			Allocate 8 more than MAX_KEY_SIZ for psDestBuf.
Visit:	Pass in uiLimit and pass back a truncated flag when the
			string is truncated.  This was not done because we will have
			to get the exact truncated count that is done in f_tocoll.cpp
			and that could introduce some bugs.
****************************************************************************/
FSTATIC RCODE KYFormatUTF8Text(
	IF_PosIStream *	pIStream,
	FLMUINT				uiFlags,					// ICD flags
	FLMUINT				uiCompareRules,		// ICD compare rules
	F_DynaBuf *			pDynaBuf)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiFirstSpaceCharPos = FLM_MAX_UINT;
	FLMUNICODE	uChar;
	FLMUINT		uiSize;
	FLMUINT		uiStrSize = 0;
	FLMBYTE *	pucTmp;

	if( !pIStream->remainingSize())
	{
		pDynaBuf->truncateData( 0);
		goto Exit;
	}

	for (;;)
	{
		if (RC_BAD( rc = flmReadUTF8CharAsUnicode( pIStream, &uChar)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
		if ((uChar = flmConvertChar( uChar, uiCompareRules)) == 0)
		{
			continue;
		}

		if (uChar == ASCII_SPACE)
		{
			if (uiCompareRules &
				 (XFLM_COMP_COMPRESS_WHITESPACE |
				  XFLM_COMP_IGNORE_TRAILING_SPACE))
			{
				
				// Remember the position of the first space.
				// When we come to the end of the spaces, we may reset
				// the size to compress out spaces if necessary.  Or,
				// we may opt to get rid of all of them.

				if (uiFirstSpaceCharPos == FLM_MAX_UINT)
				{
					uiFirstSpaceCharPos = uiStrSize;
				}
			}
		}
		else
		{
			
			// Once we hit a non-space character, we can turn off the
			// ignore leading spaces flag.
			
			uiCompareRules &= (~(XFLM_COMP_IGNORE_LEADING_SPACE));
			
			// See if we need to compress spaces.
			
			if (uiFirstSpaceCharPos != FLM_MAX_UINT)
			{
				
				// Output exactly one ASCII_SPACE character if we are compressing
				// spaces.  If we are not compressing spaces, then the only other
				// way uiFirstSpaceCharPos would have been set is if we were
				// ignoring trailing spaces.  In that case, since the spaces
				// were not trailing spaces, we need to leave them as is.
				
				if (uiCompareRules & XFLM_COMP_COMPRESS_WHITESPACE)
				{
					
					// A space will already have been encoded into the string.
					// Since we know a space takes exactly one byte in the UTF8
					// space, we can simply set our pointer one byte past where
					// the last non-space character was found.
					
					uiStrSize = uiFirstSpaceCharPos + 1;
					pDynaBuf->truncateData( uiStrSize);
				}
				uiFirstSpaceCharPos = FLM_MAX_UINT;
			}
			
			// If we are allowing escaped characters, backslash is treated
			// always as an escape character.  Whatever follows the
			// backslash is the character we need to process.

			if (uChar == ASCII_BACKSLASH && (uiFlags & ICD_ESC_CHAR))
			{
				if (RC_BAD( rc = flmReadUTF8CharAsUnicode( pIStream, &uChar)))
				{
					if (rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
		}
		
		// Output the character - need at most three bytes
		
		if (RC_BAD( rc = pDynaBuf->allocSpace( 3, (void **)&pucTmp)))
		{
			goto Exit;
		}
		uiSize = 3;
		if (RC_BAD( rc = flmUni2UTF8( uChar, pucTmp, &uiSize)))
		{
			goto Exit;
		}
		uiStrSize += uiSize;
		pDynaBuf->truncateData( uiStrSize);
	}

	// If uiFirstSpaceCharPos != FLM_MAX_UINT, it means that all of the
	// characters at the end of the string were spaces.  If we
	// are ignoring trailing spaces, we need to truncate the string so
	// they will be ignored.  Otherwise, we need to compress them into
	// a single space.
	
	if (uiFirstSpaceCharPos != FLM_MAX_UINT)
	{
		if (uiCompareRules & XFLM_COMP_IGNORE_TRAILING_SPACE)
		{
			uiStrSize = uiFirstSpaceCharPos;
		}
		else
		{
			flmAssert( uiCompareRules & XFLM_COMP_COMPRESS_WHITESPACE);
			
			// A space will already have been encoded into the string.
			// Since we know a space takes exactly one byte in the UTF8
			// space, we can simply set our pointer one byte past where
			// the last non-space character was found.

			uiStrSize = uiFirstSpaceCharPos + 1;
		}
		pDynaBuf->truncateData( uiStrSize);
	}
	
	// Terminate the UTF-8 string
	
	if (RC_BAD( rc = pDynaBuf->appendByte( 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC FLMBOOL flmAddMetaphone(
	const char *		pszStr,
	const char *		pszAltStr,
	FLMBYTE *			pszMeta,
	FLMUINT *			puiMetaOffset,
	FLMBYTE *			pszAltMeta,
	FLMUINT *			puiAltMetaOffset)
{
	FLMBOOL		bDone = FALSE;

	if( pszStr)
	{
		while( *pszStr)
		{
			if( *puiMetaOffset < 4)
			{
				pszMeta[ (*puiMetaOffset)++] = *pszStr;
			}

			if( !pszAltStr && pszAltMeta && *puiAltMetaOffset < 4)
			{
				pszAltMeta[ (*puiAltMetaOffset)++] = *pszStr;
			}

			if( *puiMetaOffset == 4 && *puiAltMetaOffset == 4)
			{
				bDone = TRUE;
				break;
			}

			pszStr++;
		}
	}

	if( pszAltStr)
	{
		while( *pszAltStr)
		{
			if( *puiAltMetaOffset < 4)
			{
				pszAltMeta[ (*puiAltMetaOffset)++] = *pszAltStr;
			}

			if( *puiMetaOffset == 4 && *puiAltMetaOffset == 4)
			{
				bDone = TRUE;
				break;
			}

			pszAltStr++;
		}
	}

	return( bDone);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void flmMetaStrToNum(
	FLMBYTE *		pszMeta,
	FLMUINT *		puiMeta)
{
	FLMUINT		uiMeta = 0;
	FLMUINT		uiOffset = 0;

	for( ;;)
	{
		if( *pszMeta)
		{
			switch( *pszMeta)
			{
				case '0':
					break;
				case 'A':
					uiMeta += 1;
					break;
				case 'F':
					uiMeta += 2;
					break;
				case 'H':
					uiMeta += 3;
					break;
				case 'J':
					uiMeta += 4;
					break;
				case 'K':
					uiMeta += 5;
					break;
				case 'L':
					uiMeta += 6;
					break;
				case 'M':
					uiMeta += 7;
					break;
				case 'N':
					uiMeta += 8;
					break;
				case 'P':
					uiMeta += 9;
					break;
				case 'R':
					uiMeta += 10;
					break;
				case 'S':
					uiMeta += 11;
					break;
				case 'T':
					uiMeta += 12;
					break;
				case 'X':
					uiMeta += 13;
					break;
				default:
					flmAssert( 0);
			}

			pszMeta++;
		}

		if( ++uiOffset == 4)
		{
			flmAssert( *pszMeta == 0);
			break;
		}
		uiMeta <<= 4;
	}

	*puiMeta = uiMeta;
}

/****************************************************************************
Desc:		Generate the metaphone and alternate metaphone keys for a given
			input string
Notes:	Lawrence Philips' Metaphone Algorithm is an algorithm which returns
			the rough approximation of how an English word sounds.  Rather
			than returning the character representation of the encoded word,
			this routine returns a 16-bit numeric representation.
****************************************************************************/
RCODE flmGetNextMetaphone(
	IF_IStream *	pIStream,
	FLMUINT *		puiMetaphone,
	FLMUINT *		puiAltMetaphone)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiInputOffset = 0;
	FLMUINT			uiInputLen = 0;
	FLMUINT			uiLast;
	FLMUINT			uiLoop;
	FLMUINT			uiMetaOffset = 0;
	FLMUINT			uiAltMetaOffset = 0;
	FLMBOOL			bSlavoGermanic = FALSE;
	FLMBOOL			bHavePrefix = FALSE;
#define MAX_METAPHONE_INPUT_CHARS		32
	FLMUNICODE		uzRealInputBuffer[ MAX_METAPHONE_INPUT_CHARS + 6];
	FLMUNICODE *	uzInput = &uzRealInputBuffer [5];
	FLMUNICODE		uChar;
	FLMBYTE			ucMeta[ 5];
	FLMBYTE			ucAltMeta[ 5];

	// Tack on five extra spaces at the beginning of the real buffer, so that we
	// can safely access characters before the beginning of the string:
	// i.e., the uzInput [uiInputLen - n] comparisons.  NOTE: n never
	// gets to be more than 5.

	for( uiLoop = 0; uiLoop < 5; uiLoop++)
	{
		uzRealInputBuffer [uiLoop] = FLM_UNICODE_SPACE;
	}

	*puiMetaphone = 0;

	if( puiAltMetaphone)
	{
		*puiAltMetaphone = 0;
	}

	// Get the first word from the stream

	for( ;;)
	{
		if( RC_BAD( rc = flmReadUTF8CharAsUnicode( 
			pIStream, &uChar)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				if( uiInputLen)
				{
					rc = NE_XFLM_OK;
					break;
				}
			}

			goto Exit;
		}

		if( gv_XFlmSysData.pXml->isWhitespace( uChar))
		{
			if( !uiInputLen)
			{
				continue;
			}
			else
			{
				// Handle the special cases of "san ", "van ", "von ",
				// and "mac ".  Since these are common name prefixes 
				// handled by the metaphone algorithm, we want to continue
				// getting the rest of the name.

				if( !bHavePrefix && uiInputLen == 3 &&
					 (f_uninativencmp( uzInput, "san", 3) == 0 ||
					  f_uninativencmp( uzInput, "van", 3) == 0 ||
					  f_uninativencmp( uzInput, "von", 3) == 0 ||
					  f_uninativencmp( uzInput, "mac", 3) == 0))
				{
					uzInput[ uiInputLen++] = FLM_UNICODE_SPACE;
					bHavePrefix = TRUE;
					continue;
				}
				else
				{
					if( bHavePrefix && uiInputLen == 4)
					{
						// Since there wasn't anything following the "prefix",
						// the trailing space needs to be removed

						uiInputLen--;
					}
					break;
				}
			}
		}

		if( uiInputLen < (MAX_METAPHONE_INPUT_CHARS - 5))
		{
			uzInput[ uiInputLen++] = f_unitolower( uChar);

			if( !bSlavoGermanic &&
				 (uChar == FLM_UNICODE_w ||
				  uChar == FLM_UNICODE_k ||
				  (uiInputLen > 1 && uChar == FLM_UNICODE_z && 
					uzInput[ uiInputLen - 2] == FLM_UNICODE_c) ||
				  (uiInputLen >= 4 && uChar == FLM_UNICODE_z &&
					uzInput[ uiInputLen - 2] == FLM_UNICODE_t &&
					uzInput[ uiInputLen - 3] == FLM_UNICODE_i &&
					uzInput[ uiInputLen - 4] == FLM_UNICODE_w)))
			{
				bSlavoGermanic = TRUE;
			}
		}
	}

	// Tack on five extra spaces to the end of the string so that
	// the algorithm below can access characters beyond the end safely.

	for( uiLoop = 0; uiLoop < 5; uiLoop++)
	{
		uzInput[ uiInputLen + uiLoop] = FLM_UNICODE_SPACE;
	}

	uzInput[ uiInputLen + 5] = 0;
	uiLast = uiInputLen - 1;

	// Skip the first letter of the following sequences when
	// they are found at the beginning of the word

	if( f_uninativencmp( &uzInput[ uiInputOffset], "gn", 2) == 0 ||
		f_uninativencmp( &uzInput[ uiInputOffset], "kn", 2) == 0 ||
		f_uninativencmp( &uzInput[ uiInputOffset], "pn", 2) == 0 ||
		f_uninativencmp( &uzInput[ uiInputOffset], "wr", 2) == 0 ||
		f_uninativencmp( &uzInput[ uiInputOffset], "ps", 2) == 0)
	{
		uiInputOffset++;
	}
	else if( uzInput[ uiInputOffset] == FLM_UNICODE_x)
	{
		// An initial 'X' is pronounced as a 'Z' which maps to 'S'

		if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
			ucAltMeta, &uiAltMetaOffset))
		{
			goto Done;
		}

		uiInputOffset++;
	}

	while( uiMetaOffset < 4 || uiAltMetaOffset < 4)
	{
		if( uiInputOffset >= uiInputLen)
		{
			break;
		}

		switch( uzInput[ uiInputOffset])
		{
			case FLM_UNICODE_a:
			case FLM_UNICODE_e:
			case FLM_UNICODE_i:
			case FLM_UNICODE_o:
			case FLM_UNICODE_u:
			case FLM_UNICODE_y:
			{
				if( !uiInputOffset)
				{
					// All initial vowels map to 'A'

					if( flmAddMetaphone( "A", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}

				uiInputOffset++;
				break;
			}
               
			case FLM_UNICODE_b:
			{
				//"-mb", e.g", "dumb", already skipped over...

				if( flmAddMetaphone( "P", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_b)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}
				break;
			}

			case FLM_UNICODE_c_CEDILLA:
			{
				if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				uiInputOffset++;
				break;
			}

			case FLM_UNICODE_c:
			{
				// Various Germanic

				if( uiInputOffset && !f_isvowel( uzInput[ uiInputOffset - 2]) &&
					f_uninativencmp( &uzInput[ uiInputOffset], "ach", 3) == 0 &&
					((f_uninativencmp( &uzInput[ uiInputOffset + 2], "i", 1) != 0) && 
					((f_uninativencmp( &uzInput[ uiInputOffset + 2], "e", 1) != 0) ||
					f_uninativencmp( &uzInput[ uiInputOffset - 2], "bacher", 6) == 0 ||
					f_uninativencmp( &uzInput[ uiInputOffset - 2], "macher", 6) == 0)))
				{       
					if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				// Special case of "caesar"

				if( !uiInputOffset && 
					f_uninativencmp( &uzInput[ uiInputOffset], "caesar", 6) == 0)
				{
					if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset +=2;
					break;
				}

				// Italian "chianti"

				if( f_uninativencmp( &uzInput[ uiInputOffset], "chia", 4) == 0)
				{
					if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "ch", 2) == 0)
            {       
					// Handle case of "Michael"

					if( uiInputOffset && 
						f_uninativencmp( &uzInput[ uiInputOffset], "chae", 4) == 0)
					{
						if( flmAddMetaphone( "K", "X", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset +=2;
						break;
					}

					// Greek roots such as "chemistry" and "chorus"

					if( !uiInputOffset && 
						(f_uninativencmp( &uzInput[ uiInputOffset + 1], "harac", 5) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "haris", 5) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "hor", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "hym", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "hia", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "hem", 3) == 0) &&
						f_uninativencmp( &uzInput[ uiInputOffset + 1], "chore", 5) != 0)
					{
						if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 2;
						break;
					}

					// Germanic, Greek 'CH' -> 'KH'

					if( f_uninativencmp( &uzInput[ 0], "van ", 4) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "von ", 4) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "sch", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset - 2], "orches", 6) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset - 2], "archit", 6) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset - 2], "orchid", 6) == 0 ||
						 uzInput[ uiInputOffset + 2] == FLM_UNICODE_t ||
						 uzInput[ uiInputOffset + 2] == FLM_UNICODE_s ||
						 ((uzInput[ uiInputOffset - 1] == FLM_UNICODE_a ||
						   uzInput[ uiInputOffset - 1] == FLM_UNICODE_o ||
						   uzInput[ uiInputOffset - 1] == FLM_UNICODE_u ||
						   uzInput[ uiInputOffset - 1] == FLM_UNICODE_e ||
						   !uiInputOffset) &&
						  (uzInput[ uiInputOffset + 2] == FLM_UNICODE_l ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_r ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_n ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_m ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_b ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_h ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_f ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_v ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_w ||
						   uzInput[ uiInputOffset + 2] == FLM_UNICODE_SPACE)))
					{
						if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{  
						if( uiInputOffset)
						{
							if( f_uninativencmp( &uzInput[ 0], "mc", 2) == 0)
							{
								// Names such as "McHugh"

								if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
							else
							{
								if( flmAddMetaphone( "X", "K", ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
						}
						else
						{
							if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
					}

					uiInputOffset += 2;
					break;
				}

				// "czerny"

				if( f_uninativencmp( &uzInput[ uiInputOffset], "cz", 2) == 0 &&
					f_uninativencmp( &uzInput[ uiInputOffset - 2], "wicz", 4) != 0)
				{
					if( flmAddMetaphone( "S", "X", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset + 1], "cia", 3) == 0)
				{
					// Words such as "focaccia"

					if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 3;
					break;
				}

				// Double 'C', but not if in a name such as "McClellan"

				if( f_uninativencmp( &uzInput[ uiInputOffset], "cc", 2) == 0 &&
					!(uiInputOffset == 1 && uzInput[ 0] == FLM_UNICODE_m))
				{
					// "bellocchio" but not "bacchus"

					if( (uzInput[ uiInputOffset + 2] == FLM_UNICODE_i ||
						  uzInput[ uiInputOffset + 2] == FLM_UNICODE_e ||
						  uzInput[ uiInputOffset + 2] == FLM_UNICODE_h) &&
						 f_uninativencmp( &uzInput[ uiInputOffset + 2], "hu", 2) != 0)
					{
						// "accident", "accede", "succeed"

						if( (uiInputOffset == 1 && 
							 uzInput[ uiInputOffset - 1] == FLM_UNICODE_a) ||
							 f_uninativencmp( &uzInput[ uiInputOffset - 1], "uccee", 5) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset - 1], "ucces", 5) == 0)
						{
							if( flmAddMetaphone( "KS", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						else
						{
							// "bacci", "bertucci", and other Italian words

							if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						uiInputOffset += 3;
						break;
					}
					else
					{
						// Pierce's rule

						if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 2;
						break;
					}
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "ck", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "cg", 2) == 0 || 
					 f_uninativencmp( &uzInput[ uiInputOffset], "cq", 2) == 0)
				{
					if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "ci", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "ce", 2) == 0 || 
					 f_uninativencmp( &uzInput[ uiInputOffset], "cy", 2) == 0)
				{
					// Italian vs. English

					if( f_uninativencmp( &uzInput[ uiInputOffset], "cio", 3) == 0 || 
						 f_uninativencmp( &uzInput[ uiInputOffset], "cie", 3) == 0 || 
						 f_uninativencmp( &uzInput[ uiInputOffset], "cia", 3) == 0)
					{
						if( flmAddMetaphone( "S", "X", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}

					uiInputOffset += 2;
					break;
				}

				// else

				if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}
                       
				// Name such as "Mac Caffrey", "Mac Gregor"

				if( f_uninativencmp( &uzInput[ uiInputOffset + 1], " c", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], " q", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], " g", 2) == 0)
				{
					uiInputOffset += 3;
				}
				else
				{
					if( (uzInput[ uiInputOffset + 1] == FLM_UNICODE_c ||
						  uzInput[ uiInputOffset + 1] == FLM_UNICODE_k ||	
						  uzInput[ uiInputOffset + 1] == FLM_UNICODE_q) &&
						 !(f_uninativencmp( &uzInput[ uiInputOffset + 1], "ce", 2) == 0 ||
						  f_uninativencmp( &uzInput[ uiInputOffset + 1], "ci", 2) == 0))
					{
						uiInputOffset += 2;
					}
					else
					{
						uiInputOffset += 1;
					}
				}

				break;
			}

			case FLM_UNICODE_d:
			{
				if( f_uninativencmp( &uzInput[ uiInputOffset], "dg", 2) == 0)
				{
					if( uzInput[ uiInputOffset + 2] == FLM_UNICODE_i ||
						 uzInput[ uiInputOffset + 2] == FLM_UNICODE_e ||
						 uzInput[ uiInputOffset + 2] == FLM_UNICODE_y)
					{
						// "edge"

						if( flmAddMetaphone( "J", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 3;
						break;
					}
					else
					{
						// "edgar"

						if( flmAddMetaphone( "TK", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 2;
						break;
					}
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "dt", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "dd", 2) == 0)
				{
					if( flmAddMetaphone( "T", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}
                    
				// else

				if( flmAddMetaphone( "T", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				uiInputOffset++;
				break;
			}

			case FLM_UNICODE_f:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_f)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset += 1;
				}

				if( flmAddMetaphone( "F", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_g:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_h)
				{
					if( uiInputOffset > 0 && 
						!f_isvowel( uzInput[ uiInputOffset - 1]))
					{
						if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 2;
						break;
					}

					if( uiInputOffset < 3)
					{
						// "ghislane", "ghiradelli"

						if( !uiInputOffset)
						{ 
							if( uzInput[ uiInputOffset + 2] == FLM_UNICODE_i)
							{
								if( flmAddMetaphone( "J", NULL, ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
							else
							{
								if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
							uiInputOffset += 2;
							break;
						}
					}

					// Parker's rule (with some further refinements) - "hugh"

					if( (uiInputOffset && 
						  (uzInput[ uiInputOffset - 2] == FLM_UNICODE_b ||
							uzInput[ uiInputOffset - 2] == FLM_UNICODE_h ||
							uzInput[ uiInputOffset - 2] == FLM_UNICODE_d)) ||
						 (uiInputOffset > 2 && // "bough"
						  (uzInput[ uiInputOffset - 3] == FLM_UNICODE_b ||
							uzInput[ uiInputOffset - 3] == FLM_UNICODE_h ||
							uzInput[ uiInputOffset - 3] == FLM_UNICODE_d)) ||
						 (uiInputOffset > 3 && // "broughton"
						  (uzInput[ uiInputOffset - 4] == FLM_UNICODE_b ||
							uzInput[ uiInputOffset - 4] == FLM_UNICODE_h)))
					{
						uiInputOffset += 2;
						break;
					}
					else
					{
						// "laugh", "McLaughlin", "cough", "gough", "rough", "tough"

						if( uiInputOffset > 2 && 
							 uzInput[ uiInputOffset - 1] == FLM_UNICODE_u &&
							 (uzInput[ uiInputOffset - 3] == FLM_UNICODE_c ||
							  uzInput[ uiInputOffset - 3] == FLM_UNICODE_g ||
							  uzInput[ uiInputOffset - 3] == FLM_UNICODE_l ||
							  uzInput[ uiInputOffset - 3] == FLM_UNICODE_r ||
							  uzInput[ uiInputOffset - 3] == FLM_UNICODE_t))
						{
							if( flmAddMetaphone( "F", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						else if( uiInputOffset && uzInput[ uiInputOffset - 1] != FLM_UNICODE_i)
						{
							if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}

						uiInputOffset += 2;
						break;
					}
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_n)
				{
					if( uiInputOffset == 1 && f_isvowel( uzInput[ 0]) && 
						!bSlavoGermanic)
					{
						if( flmAddMetaphone( "KN", "N", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						// Not "cagney", etc.

						if( f_uninativencmp( &uzInput[ uiInputOffset + 2], "ey", 2) != 0 && 
							uzInput[ uiInputOffset + 1] != FLM_UNICODE_y && 
							!bSlavoGermanic)
						{
							if( flmAddMetaphone( "N", "KN", ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						else
						{
							if( flmAddMetaphone( "KN", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
					}

					uiInputOffset += 2;
					break;
				}

				// "tagliaro"

				if( f_uninativencmp( &uzInput[ uiInputOffset + 1], "li", 2) == 0 &&
					!bSlavoGermanic)
				{
					if( flmAddMetaphone( "KL", "L", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				// Words starting with "ges", "gep", "gel", "gie", etc.

				if( !uiInputOffset && 
					(uzInput[ uiInputOffset + 1] == FLM_UNICODE_y ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "es", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "ep", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "eb", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "el", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "ey", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "ib", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "il", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "in", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "ie", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "ei", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset + 1], "er", 2) == 0))
				{
					if( flmAddMetaphone( "K", "J", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				// -ger-,  -gy-

				if( (f_uninativencmp( &uzInput[ uiInputOffset + 1], "er", 2) == 0 ||
					  uzInput[ uiInputOffset + 1] == FLM_UNICODE_y) &&
					 !(f_uninativencmp( &uzInput[ 0], "danger", 6) == 0 ||
						f_uninativencmp( &uzInput[ 0], "ranger", 6) == 0 ||
						f_uninativencmp( &uzInput[ 0], "manger", 6) == 0) &&
					 !(uzInput[ uiInputOffset - 1] == FLM_UNICODE_e ||
						uzInput[ uiInputOffset - 1] == FLM_UNICODE_i) &&
					 !(f_uninativencmp( &uzInput[ uiInputOffset - 1], "rgy", 3) == 0 ||
						f_uninativencmp( &uzInput[ uiInputOffset - 1], "ogy", 3) == 0))
				{
					if( flmAddMetaphone( "K", "J", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				// Italian words such as "biaggi"

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_e ||
					 uzInput[ uiInputOffset + 1] == FLM_UNICODE_i ||
					 uzInput[ uiInputOffset + 1] == FLM_UNICODE_y ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 1], "aggi", 4) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 1], "oggi", 4) == 0)
				{
					// Obvious Germanic

					if( f_uninativencmp( &uzInput[ 0], "van ", 4) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "von ", 4) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "sch", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "et", 2) == 0)
					{
						if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						// Always soft if French ending

						if( f_uninativencmp( &uzInput[ uiInputOffset + 1], "ier ", 4) == 0)
						{
							if( flmAddMetaphone( "J", NULL, ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						else
						{
							if( flmAddMetaphone( "J", "K", ucMeta, &uiMetaOffset, 
								ucAltMeta, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						uiInputOffset += 2;
						break;
					}
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_g)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset += 1;
				}

				if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_h:
			{
				// Only keep if first and if before a vowel or between two vowels

				if( (!uiInputOffset || f_isvowel( uzInput[ uiInputOffset - 1])) &&
					f_isvowel( uzInput[ uiInputOffset + 1]))
				{
					if( flmAddMetaphone( "H", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
				}
				else
				{
					// Take care of "HH"

					uiInputOffset++;
				}
				break;
			}

			case FLM_UNICODE_j:
			{
				// Obvious Spanish such as "Jose" and "San Jacinto"

				if( f_uninativencmp( &uzInput[ uiInputOffset], "jose", 4) == 0 ||
					f_uninativencmp( &uzInput[ 0], "san ", 4) == 0)
				{
					if( (!uiInputOffset && uzInput[ uiInputOffset + 4] == FLM_UNICODE_SPACE) ||
						f_uninativencmp( &uzInput[ 0], "san ", 4) == 0)
					{
						if( flmAddMetaphone( "H", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( flmAddMetaphone( "J", "H", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					uiInputOffset++;
					break;
				}

				if( !uiInputOffset && 
					f_uninativencmp( &uzInput[ uiInputOffset], "jose", 4) != 0)
				{
					// Yankelovich / Jankelowicz

					if( flmAddMetaphone( "J", "A", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}
				else
				{
					// Spanish pronunciation of words such as "bajador"

					if( f_isvowel( uzInput[ uiInputOffset - 1]) && 
						!bSlavoGermanic && 
						  (uzInput[ uiInputOffset + 1] == FLM_UNICODE_a || 
							uzInput[ uiInputOffset + 1] == FLM_UNICODE_o))
					{
						if( flmAddMetaphone( "J", "H", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( uiInputOffset == uiLast)
						{
							if( flmAddMetaphone( "J", NULL, ucMeta, &uiMetaOffset, 
								NULL, &uiAltMetaOffset))
							{
								goto Done;
							}
						}
						else
						{
							if( uzInput[ uiInputOffset + 1] != FLM_UNICODE_l &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_t &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_k &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_s &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_n &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_m &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_b &&
								 uzInput[ uiInputOffset + 1] != FLM_UNICODE_z &&
								 uzInput[ uiInputOffset - 1] != FLM_UNICODE_s &&
								 uzInput[ uiInputOffset - 1] != FLM_UNICODE_k &&
								 uzInput[ uiInputOffset - 1] != FLM_UNICODE_l)
							{
								if( flmAddMetaphone( "J", NULL, ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
						}
					}
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_j)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}
				break;
			}

			case FLM_UNICODE_k:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_k)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}
				break;
			}

			case FLM_UNICODE_l:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_l)
				{
					// Spanish words such as "cabrillo" and "gallegos"

					if( (uiInputOffset == (uiInputLen - 3) && 
							(f_uninativencmp( &uzInput[ uiInputOffset - 1], "illo", 4) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset - 1], "illa", 4) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset - 1], "alle", 4) == 0)) ||
						 ((f_uninativencmp( &uzInput[ uiLast - 1], "as", 2) == 0 ||
						   f_uninativencmp( &uzInput[ uiLast - 1], "os", 2) == 0 ||
							uzInput[ uiLast] == FLM_UNICODE_a ||
							uzInput[ uiLast] == FLM_UNICODE_o) &&
						  f_uninativencmp( &uzInput[ uiInputOffset - 1], "alle", 4) == 0))
					{
						if( flmAddMetaphone( "L", NULL, ucMeta, &uiMetaOffset, 
							NULL, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 2;
						break;
					}
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "L", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_m:
			{
				if( (f_uninativencmp( &uzInput[ uiInputOffset - 1], "umb", 3) == 0 &&
						((uiInputOffset + 1) == uiLast || 
							f_uninativencmp( &uzInput[ uiInputOffset + 2], "er", 2) == 0)) ||
					uzInput[ uiInputOffset + 1] == FLM_UNICODE_m)	// "dumb", "thumb", etc.
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "M", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_n:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_n)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "N", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_n_TILDE:
			{
				if( flmAddMetaphone( "N", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				uiInputOffset++;
				break;
			}

			case FLM_UNICODE_p:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_h)
				{
					if( flmAddMetaphone( "F", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				// Account for "Campbell", "raspberry", etc.

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_p ||
					 uzInput[ uiInputOffset + 1] == FLM_UNICODE_b)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "P", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_q:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_q)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset += 1;
				}

				if( flmAddMetaphone( "K", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_r:
			{
				// French words such as "rogier".  Excludes "Hochmeier"

				if( uiInputOffset == uiLast && 
					!bSlavoGermanic &&
					f_uninativencmp( &uzInput[ uiInputOffset - 2], "ie", 2) == 0 &&
					f_uninativencmp( &uzInput[ uiInputOffset - 4], "me", 2) != 0 &&
					f_uninativencmp( &uzInput[ uiInputOffset - 4], "ma", 2) != 0)
				{
					if( flmAddMetaphone( NULL, "R", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}
				else
				{
					if( flmAddMetaphone( "R", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_r)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}
				break;
			}

			case FLM_UNICODE_s:
			{
				// Special cases of "island", "isle", "carlisle", "carlysle"
				
				if( f_uninativencmp( &uzInput[ uiInputOffset - 1], "isl", 3) == 0 ||
					f_uninativencmp( &uzInput[ uiInputOffset - 1], "ysl", 3) == 0)
				{
					uiInputOffset++;
					break;
				}

				// Special case of 'sugar-'

				if( !uiInputOffset && 
					f_uninativencmp( &uzInput[ uiInputOffset], "sugar", 5) == 0)
				{
					if( flmAddMetaphone( "X", "S", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset++;
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "sh", 2) == 0)
				{
					// Germanic

					if( f_uninativencmp( &uzInput[ uiInputOffset + 1], "heim", 4) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "hoek", 4) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "holm", 4) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "holz", 4) == 0)
					{
						if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}

					uiInputOffset += 2;
					break;
				}

				// Italian and Armenian

				if( f_uninativencmp( &uzInput[ uiInputOffset], "sio", 3) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "sia", 3) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "sian", 4) == 0)
				{
					if( !bSlavoGermanic)
					{
						if( flmAddMetaphone( "S", "X", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}

					uiInputOffset += 3;
					break;
				}

				// German & Anglicisations such as "Smith" matching "Schmidt" and
				// "Snider" matching "Schneider"

				if( (!uiInputOffset && 
					  (uzInput[ uiInputOffset + 1] == FLM_UNICODE_m ||
					   uzInput[ uiInputOffset + 1] == FLM_UNICODE_n ||
						uzInput[ uiInputOffset + 1] == FLM_UNICODE_l ||
						uzInput[ uiInputOffset + 1] == FLM_UNICODE_w)) ||
					 uzInput[ uiInputOffset + 1] == FLM_UNICODE_z)
				{
					if( flmAddMetaphone( "S", "X", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_z)
					{
						uiInputOffset += 2;
					}
					else
					{
						uiInputOffset++;
					}
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "sc", 2) == 0)
				{
					// Schlesinger's rule

					if( uzInput[ uiInputOffset + 2] == FLM_UNICODE_h)
					{
						// Words of Dutch origin such as "school" and "schooner"

						if( f_uninativencmp( &uzInput[ uiInputOffset + 3], "oo", 2) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset + 3], "er", 2) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset + 3], "en", 2) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset + 3], "uy", 2) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset + 3], "ed", 2) == 0 ||
							 f_uninativencmp( &uzInput[ uiInputOffset + 3], "em", 2) == 0)
						{
							// "Schermerhorn", "Schenker"

							if( f_uninativencmp( &uzInput[ uiInputOffset + 3], "er", 2) == 0 ||
								 f_uninativencmp( &uzInput[ uiInputOffset + 3], "en", 2) == 0)
							{
								if( flmAddMetaphone( "X", "SK", ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
							else
							{
								if( flmAddMetaphone( "SK", NULL, ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}

							uiInputOffset += 3;
							break;
						}
						else
						{
							if( !uiInputOffset && !f_isvowel( uzInput[ 3]) && 
								uzInput[ 3] != FLM_UNICODE_w)
							{
								if( flmAddMetaphone( "X", "S", ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}
							else
							{
								if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
									ucAltMeta, &uiAltMetaOffset))
								{
									goto Done;
								}
							}

							uiInputOffset += 3;
							break;
						}
					}

					if( uzInput[ uiInputOffset + 2] == FLM_UNICODE_i ||
						 uzInput[ uiInputOffset + 2] == FLM_UNICODE_e ||
						 uzInput[ uiInputOffset + 2] == FLM_UNICODE_y)
					{
						if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}

						uiInputOffset += 3;
						break;
					}

					if( flmAddMetaphone( "SK", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 3;
					break;
				}

				// French words such as "resnais" and "artois"

				if( uiInputOffset == uiLast &&
					(f_uninativencmp( &uzInput[ uiInputOffset - 2], "ai", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 2], "oi", 2) == 0))
				{
					if( flmAddMetaphone( NULL, "S", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}
				else
				{
					if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_s ||
					uzInput[ uiInputOffset + 1] == FLM_UNICODE_z)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}
				break;
			}

			case FLM_UNICODE_t:
			{
				if( f_uninativencmp( &uzInput[ uiInputOffset], "tion", 4) == 0)
				{
					if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 3;
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "tia", 3) == 0 ||
					f_uninativencmp( &uzInput[ uiInputOffset], "tch", 3) == 0)
				{
					if( flmAddMetaphone( "X", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 3;
					break;
				}

				if( f_uninativencmp( &uzInput[ uiInputOffset], "th", 2) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "tth", 3) == 0)
				{
					// Special cases of "Thomas", "Thames", or Germanic

					if( f_uninativencmp( &uzInput[ uiInputOffset + 2], "om", 2) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 2], "am", 2) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "van ", 4) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "von ", 4) == 0 ||
						 f_uninativencmp( &uzInput[ 0], "sch", 3) == 0)
					{
						if( flmAddMetaphone( "T", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( flmAddMetaphone( "0", "T", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}

					uiInputOffset += 2;
					break;
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_t ||
					uzInput[ uiInputOffset] == FLM_UNICODE_d)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "T", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_v:
			{
				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_v)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}

				if( flmAddMetaphone( "F", NULL, ucMeta, &uiMetaOffset, 
					ucAltMeta, &uiAltMetaOffset))
				{
					goto Done;
				}

				break;
			}

			case FLM_UNICODE_w:
			{
				if( f_uninativencmp( &uzInput[ uiInputOffset], "wr", 2) == 0)
				{
					if( flmAddMetaphone( "R", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}

				if( !uiInputOffset &&
					(f_isvowel( uzInput[ uiInputOffset + 1]) || 
					 f_uninativencmp( &uzInput[ uiInputOffset], "wh", 2) == 0))
				{
					// "Wasserman" should match "Vasserman"

					if( f_isvowel( uzInput[ uiInputOffset + 1]))
					{
						if( flmAddMetaphone( "A", "F", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						// "Uomo" should match "Womo"

						if( flmAddMetaphone( "A", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
				}

				// "Arnow" should match "Arnoff"

				if( (uiInputOffset == uiLast && 
					 f_isvowel( uzInput[ uiInputOffset - 1])) ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 1], "ewski", 5) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 1], "ewsky", 5) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 1], "owski", 5) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset - 1], "owsky", 5) == 0 ||
					 f_uninativencmp( &uzInput[ 0], "sch", 3) == 0)
				{
					if( flmAddMetaphone( NULL, "F", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset++;
					break;
				}

				// Polish names and words such as "Filipowicz"

				if( f_uninativencmp( &uzInput[ uiInputOffset], "wicz", 4) == 0 ||
					 f_uninativencmp( &uzInput[ uiInputOffset], "witz", 4) == 0)
				{
					if( flmAddMetaphone( "TS", "FX", ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset +=4;
					break;
				}

				// otherwise, skip the current character

				uiInputOffset++;
				break;
			}

			case FLM_UNICODE_x:
			{
				// French words such as "breaux"

				if( !(uiInputOffset == uiLast && 
						(f_uninativencmp( &uzInput[ uiInputOffset - 3], "iau", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset - 3], "eau", 3) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset - 2], "au", 2) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset - 2], "ou", 2) == 0)))
				{
					if( flmAddMetaphone( "KS", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}
				}

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_c ||
					 uzInput[ uiInputOffset + 1] == FLM_UNICODE_x)
				{
					uiInputOffset += 2;
				}
				else
				{
					uiInputOffset++;
				}
				break;
			}

			case FLM_UNICODE_z:
			{
				// Chinese pinyin such as "Zhao"

				if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_h)
				{
					if( flmAddMetaphone( "J", NULL, ucMeta, &uiMetaOffset, 
						ucAltMeta, &uiAltMetaOffset))
					{
						goto Done;
					}

					uiInputOffset += 2;
					break;
				}
				else
				{
					if( f_uninativencmp( &uzInput[ uiInputOffset + 1], "zo", 2) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "zi", 2) == 0 ||
						 f_uninativencmp( &uzInput[ uiInputOffset + 1], "za", 2) == 0 ||
						 (bSlavoGermanic && uiInputOffset && 
							uzInput[ uiInputOffset - 1] != FLM_UNICODE_t))
					{
						if( flmAddMetaphone( "S", "TS", ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}
					else
					{
						if( flmAddMetaphone( "S", NULL, ucMeta, &uiMetaOffset, 
							ucAltMeta, &uiAltMetaOffset))
						{
							goto Done;
						}
					}

					if( uzInput[ uiInputOffset + 1] == FLM_UNICODE_z)
					{
						uiInputOffset += 2;
					}
					else
					{
						uiInputOffset++;
					}
				}

				break;
			}

			default:
			{
				uiInputOffset++;
				break;
			}
		}
	}

Done:

	ucMeta[ uiMetaOffset] = 0;
	flmMetaStrToNum( ucMeta, puiMetaphone);

	if( puiAltMetaphone)
	{
		ucAltMeta[ uiAltMetaOffset] = 0;
		flmMetaStrToNum( ucAltMeta, puiAltMetaphone);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Verifies that the metaphone routines are generating the correct
			codes for a hard-coded set of words.
****************************************************************************/
#ifdef FLM_DEBUG
RCODE flmVerifyMetaphoneRoutines( void)
{
	RCODE						rc = NE_XFLM_OK;
	METAPHONE_MAPPING *	pMetaMap = gv_MetaTestTable;
	F_BufferIStream		bufferStream;
	FLMUINT					uiMeta;
	FLMUINT					uiAltMeta;

	for( ;;)
	{
		if( !pMetaMap->pszWord)
		{
			break;
		}

		if( RC_BAD( rc = bufferStream.open( 
			(FLMBYTE *)pMetaMap->pszWord, f_strlen( pMetaMap->pszWord))))
		{
			goto Exit;
		}

		if( RC_BAD( rc = flmGetNextMetaphone( &bufferStream, 
			&uiMeta, &uiAltMeta)))
		{
			goto Exit;
		}

		if( uiMeta != pMetaMap->uiMeta ||
			uiAltMeta != pMetaMap->uiAltMeta)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
			goto Exit;
		}

		bufferStream.close();
		pMetaMap++;
	}

Exit:

	flmAssert( RC_OK( rc));
	return( rc);
}
#endif
