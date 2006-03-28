//-------------------------------------------------------------------------
// Desc:	Miscellaneous functions.
// Tabs:	3
//
//		Copyright (c) 1995-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fmisc.cpp 12266 2006-01-19 14:45:33 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	FLAIM language code table
****************************************************************************/
static char flmLangTable[ LAST_LANG + LAST_LANG] = 
{
	'U', 'S',								// English, United States
	'A', 'F',								// Afrikaans
	'A', 'R',								// Arabic
	'C', 'A',								// Catalan
	'H', 'R',								// Croatian
	'C', 'Z',								// Czech
	'D', 'K',								// Danish
	'N', 'L',								// Dutch
	'O', 'Z',								// English, Australia
	'C', 'E',								// English, Canada
	'U', 'K',								// English, United Kingdom
	'F', 'A',								// Farsi
	'S', 'U',								// Finnish
	'C', 'F',								// French, Canada
	'F', 'R',								// French, France
	'G', 'A',								// Galician		
	'D', 'E',								// German, Germany
	'S', 'D',								// German, Switzerland
	'G', 'R',								// Greek
	'H', 'E',								// Hebrew
	'M', 'A',								// Hungarian
	'I', 'S',								// Icelandic
	'I', 'T',								// Italian
	'N', 'O',								// Norwegian
	'P', 'L',								// Polish
	'B', 'R',								// Portuguese, Brazil
	'P', 'O',								// Portuguese, Portugal
	'R', 'U',								// Russian
	'S', 'L',								// Slovak
	'E', 'S',								// Spanish
	'S', 'V',								// Swedish
	'Y', 'K',								// Ukrainian
	'U', 'R',								// Urdu
	'T', 'K',								// Turkey
	'J', 'P',								// Japanese
	'K', 'R',								// Korean
	'C', 'T',								// Chinese-Traditional
	'C', 'S',								// Chinese-Simplified
	'L', 'A'									// Future asian language
};

/****************************************************************************
Desc:	Determine the language number from the 2 byte language code
****************************************************************************/
FLMEXP FLMUINT FLMAPI FlmLanguage(
	char *	pszLanguageCode)
{
	char		cFirstChar  = *pszLanguageCode;
	char		cSecondChar = *(pszLanguageCode + 1);
	FLMUINT	uiTablePos;

	for (uiTablePos = 0; uiTablePos < (LAST_LANG+LAST_LANG); uiTablePos += 2 )
	{
		if (flmLangTable[ uiTablePos] == cFirstChar &&
			 flmLangTable[ uiTablePos + 1] == cSecondChar)
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
FLMEXP void FLMAPI FlmGetLanguage(
	FLMUINT	uiLangNum,
	char *	pszLanguageCode)
{
	if (uiLangNum >= LAST_LANG)
	{
		uiLangNum = US_LANG;
	}

	uiLangNum += uiLangNum;
	*pszLanguageCode++ = flmLangTable[ uiLangNum];
	*pszLanguageCode++ = flmLangTable[ uiLangNum + 1];
	*pszLanguageCode = 0;
}

/****************************************************************************
Desc:	Returns TRUE if the passed in RCODE indicates that a corruption
		has occured in a FLAIM database file.
****************************************************************************/
FLMEXP FLMBOOL FLMAPI FlmErrorIsFileCorrupt(
	RCODE			rc)
{
	FLMBOOL		b = FALSE;

	switch( rc)
	{
		case FERR_BTREE_ERROR :
		case FERR_DATA_ERROR :
		case FERR_DD_ERROR :
		case FERR_NOT_FLAIM :
		case FERR_PCODE_ERROR :
		case FERR_BLOCK_CHECKSUM :
		case FERR_INCOMPLETE_LOG :
		case FERR_KEY_NOT_FOUND :
		case FERR_NO_REC_FOR_KEY:
			b = TRUE;
			break;
		default :
			break;
	}

	return( b);
}

/****************************************************************************
Desc:	Returns specific information about the most recent error that
		occured within FLAIM.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmGetDiagInfo(
	HFDB				hDb,
	eDiagInfoType	eDiagCode,
	void *			pvDiagInfo)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb;

	if ((pDb = (FDB *)hDb) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	fdbUseCheck( pDb);

	/* Now, copy over the data into the users variable */
	switch( eDiagCode)
	{
		case FLM_GET_DIAG_INDEX_NUM :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_INDEX_NUM))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiIndexNum;
			}
			break;

		case FLM_GET_DIAG_DRN :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_DRN))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiDrn;
			}
			break;

		case FLM_GET_DIAG_FIELD_NUM :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_FIELD_NUM))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiFieldNum;
			}
			break;

		case FLM_GET_DIAG_FIELD_TYPE :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_FIELD_TYPE))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiFieldType;
			}
			break;

		case FLM_GET_DIAG_ENC_ID :
			if (!(pDb->Diag.uiInfoFlags & FLM_DIAG_ENC_ID))
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}
			else
			{
				*((FLMUINT *)pvDiagInfo) = pDb->Diag.uiEncId;
			}
			break;
		default:
			flmAssert( 0);
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;

	}

Exit:
	if( pDb)
	{
		fdbUnuse( pDb);
	}
	return( rc);
}


/****************************************************************************
Desc:	Get the total bytes represented by a particular block address.
****************************************************************************/
FLMUINT64 FSGetSizeInBytes(
	FLMUINT	uiMaxFileSize,
	FLMUINT	uiBlkAddress)
{
	FLMUINT	uiFileNum;
	FLMUINT	uiFileOffset;
	FLMUINT64	ui64Size;

	uiFileNum = FSGetFileNumber( uiBlkAddress);
	uiFileOffset = FSGetFileOffset( uiBlkAddress);
	if( uiFileNum > 1)
	{
		ui64Size = (FLMUINT64)(((FLMUINT64)uiFileNum - (FLMUINT64)1) *
											(FLMUINT64)uiMaxFileSize +
											(FLMUINT64)uiFileOffset);
	}
	else
	{
		ui64Size = (FLMUINT64)uiFileOffset;
	}
	return( ui64Size);
}

/****************************************************************************
Desc:	Converts a UNICODE string consisting of 7-bit ASCII characters to
		a 7-bit ASCII string.  The conversion is done in place, so that
		only one buffer is needed
*****************************************************************************/
RCODE flmUnicodeToAscii(
	FLMUNICODE *	puzString) // Unicode in, Ascii out
{
	FLMBYTE *	pucDest;

	pucDest = (FLMBYTE *)puzString;
	while( *puzString)
	{
		if( *puzString > 0x007F)
		{
			*pucDest = 0xFF;
		}
		else
		{
			*pucDest = (FLMBYTE)*puzString;
		}
		pucDest++;
		puzString++;
	}
	*pucDest = '\0';

	return( FERR_OK);
}
