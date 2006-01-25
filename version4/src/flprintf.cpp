//-------------------------------------------------------------------------
// Desc:	sprintf type functionality.
// Tabs:	3
//
//		Copyright (c) 2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flprintf.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

// Percent formating prefixes

#define FLM_PREFIX_NONE				0
#define FLM_PREFIX_MINUS 			1
#define FLM_PREFIX_PLUS				2
#define FLM_PREFIX_POUND 			3

// Width and Precision flags

#define FLM_PRINTF_MINUS_FLAG		0x0001
#define FLM_PRINTF_PLUS_FLAG		0x0002
#define FLM_PRINTF_SPACE_FLAG		0x0004
#define FLM_PRINTF_POUND_FLAG		0x0008
#define FLM_PRINTF_ZERO_FLAG		0x0010
#define FLM_PRINTF_SHORT_FLAG		0x0020
#define FLM_PRINTF_LONG_FLAG		0x0040
#define FLM_PRINTF_DOUBLE_FLAG	0x0080
#define FLM_PRINTF_INT64_FLAG		0x0100
#define FLM_PRINTF_COMMA_FLAG		0x0200

FSTATIC void flmSprintfParseArgs(
	const char *		pszFormat,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

FSTATIC void flmProcessFormatString(
	FLMUINT					uiLen,
	F_SPRINTF_INFO *		pInfo,
	...);

FSTATIC FLMUINT flmPrintNumber(
	FLMUINT64			ui64Val,
	FLMUINT				uiBase,
	FLMBOOL				bUpperCase,
	FLMBOOL				bCommas,
	char *				pszBuf);

FSTATIC void flmSprintfProcessFieldInfo(
	const char **		ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args);

FSTATIC void flmSprintfCharFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

FSTATIC void flmSprintfErrorFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

FSTATIC void flmSprintfNotHandledFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

FSTATIC void flmSprintfNumberFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

/****************************************************************************
Desc:		Parameter 'format' points to text following a '%' sign. Process
			legal field information.	Leave 'format' pointing at the format
			specifier char.
****************************************************************************/
FSTATIC void flmSprintfProcessFieldInfo(
	const char **		ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args)
{
	const char *		pszFormat = *ppszFormat;

	// Process flags

	*puiFlags = 0;
	for( ;;)
	{
		switch( *pszFormat)
		{
			case '-':
				*puiFlags |= FLM_PRINTF_MINUS_FLAG;
				break;

			case '+':
				*puiFlags |= FLM_PRINTF_PLUS_FLAG;
				break;

			case ' ':
				*puiFlags |= FLM_PRINTF_SPACE_FLAG;
				break;

			case '#':
				*puiFlags |= FLM_PRINTF_POUND_FLAG;
				break;

			case '0':
				*puiFlags |= FLM_PRINTF_ZERO_FLAG;
				break;

			case ',':
				*puiFlags |= FLM_PRINTF_COMMA_FLAG;
				break;

			default:
				goto NoMoreFlags;
		}

		pszFormat++;
	}

NoMoreFlags:

	// Process width

	*puiWidth = 0;
	if( *pszFormat == '*')
	{
		*puiWidth = f_va_arg( *args, unsigned int);
		pszFormat++;
	}
	else
	{
		while( *pszFormat >= '0' && *pszFormat <= '9')
		{
			*puiWidth = (*puiWidth * 10) + (*pszFormat - '0');
			pszFormat++;
		}
	}

	// Process precision

	*puiPrecision = 0;
	if( *pszFormat == '.')
	{
		pszFormat++;
		if( *pszFormat == '*')
		{
			*puiPrecision = f_va_arg( *args, unsigned int);
			pszFormat++;
		}
		else while( *pszFormat >= '0' && *pszFormat <= '9')
		{
			*puiPrecision = (*puiPrecision * 10) + (*pszFormat - '0');
			pszFormat++;
		}
	}

	// Size modifiers
	switch( *pszFormat)
	{
		case 'I':
			if( pszFormat[ 1] == '6' && pszFormat[ 2] == '4')
			{
				*puiFlags |= FLM_PRINTF_INT64_FLAG;
				pszFormat += 3;
			}
			else
			{
				flmAssert( 0);
			}
			break;

		case 'L':
			*puiFlags |= FLM_PRINTF_DOUBLE_FLAG;
			pszFormat++;
			break;

		case 'l':
			*puiFlags |= FLM_PRINTF_LONG_FLAG;
			pszFormat++;
			break;

		case 'h':
			*puiFlags |= FLM_PRINTF_SHORT_FLAG;
			pszFormat++;
			break;
	}

	*ppszFormat = pszFormat;
	return;
}

/****************************************************************************
Desc:		Handle text portions of the format string
****************************************************************************/
FSTATIC void flmProcessFormatString(
	FLMUINT					uiLen,
	F_SPRINTF_INFO *		pInfo,
	...)
{
	f_va_list	args;

	f_va_start( args, pInfo);
	if( uiLen)
	{
		flmSprintfStringFormatter( 0, uiLen, uiLen, 0, pInfo, &args);
	}
	f_va_end( args);
}

/****************************************************************************
Desc:		Parse arguments in format string, calling appropriate handlers
****************************************************************************/
FSTATIC void flmSprintfParseArgs(
	const char *		pszFormat,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	char				ucFormatChar;
	FLMUINT			uiFlags;
	FLMUINT			uiWidth;
	FLMUINT			uiPrecision;
	const char *	pszTextStart = pszFormat;

	while( (ucFormatChar = *pszFormat++) != 0)
	{
		if( ucFormatChar != '%')
		{
			continue;
		}

		uiWidth = (FLMUINT)(pszFormat - pszTextStart - 1);
		flmProcessFormatString( uiWidth, pInfo, pszTextStart);

		flmSprintfProcessFieldInfo( &pszFormat, &uiWidth,
			&uiPrecision, &uiFlags, args);

		ucFormatChar = (unsigned char)*pszFormat++;
		switch( ucFormatChar)
		{
			case '%':
			case 'c':
				flmSprintfCharFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			case 'd':
			case 'i':
			case 'o':
			case 'u':
			case 'x':
			case 'X':
			case 'p':
				if( ucFormatChar == 'i')
				{
					ucFormatChar = 'd';
				}

				flmSprintfNumberFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			case 's':
			case 'S':
			case 'U':
				flmSprintfStringFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			case 'e':
			case 'E':
				flmSprintfErrorFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			default:
				flmSprintfNotHandledFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;
		}
		pszTextStart = pszFormat;
	}

	flmProcessFormatString( (FLMUINT)(pszFormat - pszTextStart - 1),
		pInfo, pszTextStart);
}

/****************************************************************************
Desc:		Default string formatter.
****************************************************************************/
void flmSprintfStringFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	static char			pszNullPointerStr[] = "<null>";
	FLMUINT				uiOutputLen;
	FLMUINT				uiCount;
	FLMUNICODE *		pUnicode;
	char *				pszStr = f_va_arg( *args, char *);
	char *				pszDest = pInfo->pszDestStr;

	if( !pszStr)
	{
		uiOutputLen = f_strlen( pszNullPointerStr);
	}
	else if( ucFormatChar == 'S')
	{
		uiOutputLen = *pszStr++;
	}
	else
	{
		if( ucFormatChar == 'U')
		{
			uiOutputLen = 0;
			pUnicode = (FLMUNICODE *)pszStr;
			while( *pUnicode)
			{
				if( *pUnicode >= 32 && *pUnicode <= 127)
				{
					uiOutputLen++;
				}
				else
				{
					uiOutputLen += 7;
				}
				pUnicode++;
			}
		}
		else
		{
			uiOutputLen = f_strlen( pszStr);
		}
	}

	if( uiPrecision > 0 && uiOutputLen > uiPrecision)
	{
		uiOutputLen = uiPrecision;
	}

	uiCount = uiWidth - uiOutputLen;

	if( uiOutputLen < uiWidth && !(uiFlags & FLM_PRINTF_MINUS_FLAG))
	{
		// Right justify

		f_memset( pszDest, ' ', uiCount);
		pszDest += uiCount;
	}

	if( !pszStr)
	{
		f_memcpy( pszDest, pszNullPointerStr, uiOutputLen);
		pszDest += uiOutputLen;
	}
	else if( ucFormatChar == 'U')
	{
		FLMUINT		uiBytesOutput = 0;

		for( pUnicode = (FLMUNICODE *)pszStr;
			uiBytesOutput < uiOutputLen && *pUnicode; pUnicode++)
		{
			if( *pUnicode >= 32 && *pUnicode <= 127)
			{
				*pszDest++ = (char)*pUnicode;
				uiBytesOutput++;
			}
			else
			{
				char			szTmpBuf[ 8];
				FLMUINT		uiTmpLen;
				char *		pszTmp;

				szTmpBuf[ 0] = '~';
				szTmpBuf[ 1] = '[';
				uiTmpLen = flmPrintNumber( (FLMUINT64)(*pUnicode),
					16, TRUE, FALSE, &szTmpBuf[ 2]);
				uiTmpLen += 2;
				szTmpBuf[ uiTmpLen] = ']';
				szTmpBuf[ uiTmpLen + 1] = 0;

				pszTmp = szTmpBuf;
				while( *pszTmp && uiBytesOutput < uiOutputLen)
				{
					*pszDest++ = *pszTmp;
					pszTmp++;
					uiBytesOutput++;
				}
			}
		}
	}
	else
	{
		f_memcpy( pszDest, pszStr, uiOutputLen);
		pszDest += uiOutputLen;
	}

	if( uiOutputLen < uiWidth && (uiFlags & FLM_PRINTF_MINUS_FLAG))
	{
		// Left justify

		f_memset( pszDest, ' ', uiCount);
		pszDest += uiCount;
	}

	*pszDest = 0;
	uiCount = (FLMUINT)(pszDest - pInfo->pszDestStr);
	pInfo->pszDestStr = pszDest;
}

/****************************************************************************
Desc:		Converts a number to a string
****************************************************************************/
FSTATIC FLMUINT flmPrintNumber(
	FLMUINT64		ui64Val,
	FLMUINT			uiBase,
	FLMBOOL			bUpperCase,
	FLMBOOL			bCommas,
	char *			pszBuf)
{
	char			ucChar;
	FLMUINT		uiOffset = 0;
	FLMUINT		uiDigitCount = 0;
	FLMUINT		uiLoop;

	// We don't support commas on bases other than 10

	if( uiBase != 10)
	{
		bCommas = FALSE;
	}

	// Build the number string from the value

	for( ;;)
	{
		ucChar = (char)(ui64Val % (FLMUINT64)uiBase);
		pszBuf[ uiOffset++] = (char)(ucChar > 9
			? ucChar + (bUpperCase ? 'A' : 'a') - 10
						: ucChar + '0');
		uiDigitCount++;

		if( (ui64Val = (ui64Val / (FLMUINT64)uiBase)) == (FLMUINT64)0)
		{
			break;
		}

		if( bCommas && (uiDigitCount % 3) == 0)
		{
			pszBuf[ uiOffset++] = ',';
		}
	}

	// Reverse the string

	for( uiLoop = 0; uiLoop < uiOffset / 2; uiLoop++)
	{
		ucChar = pszBuf[ uiLoop];
		pszBuf[ uiLoop] = pszBuf[ uiOffset - uiLoop - 1];
		pszBuf[ uiOffset - uiLoop - 1] = ucChar;
	}

	pszBuf[ uiOffset] = 0;
	return( uiOffset);
}

/****************************************************************************
Desc:		Default number formatter.
			Format: %[flags][width][.prec]'E'
****************************************************************************/
FSTATIC void flmSprintfNumberFormatter(
	char					ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	FLMUINT		uiCount;
	FLMUINT		uiPrefix = FLM_PREFIX_NONE;
	FLMUINT		uiLength;
	FLMUINT		uiBase = 10;
	char			ucNumberBuffer[ 64];
	FLMBOOL		bUpperCase = FALSE;
	FLMBOOL		bCommas = FALSE;
	char *		pszTmp;
	char *		pszDest = pInfo->pszDestStr;
	FLMUINT64	ui64Val;

	if( ucFormatChar == 'p')
	{
		ui64Val = (FLMUINT64)((FLMUINT)f_va_arg( *args, void *));
		uiFlags |= FLM_PRINTF_POUND_FLAG;
	}
	else  if( ucFormatChar != 'd')
	{
		// Unsigned number

		if( uiFlags & FLM_PRINTF_SHORT_FLAG)
		{
			ui64Val = (FLMUINT64)((unsigned int)f_va_arg( *args, int));
		}
		else if( uiFlags & (FLM_PRINTF_LONG_FLAG | FLM_PRINTF_DOUBLE_FLAG))
		{
			ui64Val = (FLMUINT64)((unsigned long int)f_va_arg( *args, long int));
		}
		else if( uiFlags & FLM_PRINTF_INT64_FLAG)
		{
			ui64Val = f_va_arg( *args, FLMUINT64);
		}
		else
		{
			ui64Val = (FLMUINT64)((unsigned int)f_va_arg( *args, int));
		}
	}
	else
	{
		// Signed number

		if( uiFlags & FLM_PRINTF_SHORT_FLAG)
		{
			ui64Val = (FLMUINT64)((FLMINT64)f_va_arg( *args, int));
		}
		else if( uiFlags & (FLM_PRINTF_LONG_FLAG | FLM_PRINTF_DOUBLE_FLAG))
		{
			ui64Val = (FLMUINT64)((FLMINT64)f_va_arg( *args, long int));
		}
		else if( uiFlags & FLM_PRINTF_INT64_FLAG)
		{
			ui64Val = (FLMUINT64)f_va_arg( *args, FLMINT64);
		}
		else
		{
			ui64Val = (FLMUINT64)((FLMINT64)f_va_arg( *args, int));
		}
	}

	switch( ucFormatChar)
	{
		case 'd':
		{
			if( ((FLMINT64)ui64Val) < 0)
			{
				uiPrefix = FLM_PREFIX_MINUS;
				if( uiWidth > 0)
				{
					uiWidth--;
				}
				ui64Val = (FLMUINT64)(-(FLMINT64)ui64Val);
			}
			else if( uiFlags & FLM_PRINTF_PLUS_FLAG)
			{
				uiPrefix = FLM_PREFIX_PLUS;
				if( uiWidth > 0)
				{
					uiWidth--;
				}
			}
			break;
		}

		case 'o':
		{
			uiBase = 8;
			break;
		}

		case 'x':
		case 'X':
		case 'p':
		{
			if( (uiFlags & FLM_PRINTF_POUND_FLAG) && ui64Val != (FLMUINT64)0)
			{
				uiPrefix = FLM_PREFIX_POUND;
				if( uiWidth > 1)
				{
					uiWidth -= 2;
				}
			}
			uiBase = 16;
			break;
		}
	}

	if( ucFormatChar == 'X')
	{
		bUpperCase = TRUE;
	}

	if( (uiFlags & FLM_PRINTF_COMMA_FLAG) && uiBase == 10)
	{
		bCommas = TRUE;
	}

	uiLength = flmPrintNumber( ui64Val, uiBase, bUpperCase,
		bCommas, ucNumberBuffer);

	if( uiWidth < uiLength)
	{
		uiWidth = uiLength;
	}

	if( uiFlags & FLM_PRINTF_ZERO_FLAG)
	{
		// Zero fill

		uiPrecision = uiWidth;
	}
	else if( !(uiFlags & FLM_PRINTF_MINUS_FLAG))
	{
		// Right justify

		while( uiWidth > uiLength && uiWidth > uiPrecision)
		{
			*pszDest++ = ' ';
			uiWidth--;
		}
	}

	// Handle the prefix (if any)

	switch( uiPrefix)
	{
		case FLM_PREFIX_NONE:
			break;

		case FLM_PREFIX_MINUS:
			*pszDest++ = '-';
			break;

		case FLM_PREFIX_PLUS:
			*pszDest++ = '+';
			break;

		case FLM_PREFIX_POUND:
		{
			*pszDest++ = '0';
			*pszDest++ = 'x';
			break;
		}

		default:
			flmAssert( 0);
			break;
	}

	// Handle the precision

	if( bCommas && uiPrecision && (uiPrecision % 4) == 0)
	{
		uiPrecision--;
	}

	while( uiLength < uiPrecision)
	{
		if( bCommas && (uiPrecision % 4) == 0)
		{
			*pszDest++ = ',';
			uiPrecision--;
			uiWidth--;
			continue;
		}

		*pszDest++ = '0';
		uiPrecision--;
		uiWidth--;
	}

	// Output the number

	for( uiCount = uiLength, pszTmp = &ucNumberBuffer[ 0];
		uiCount > 0; uiCount--)
	{
		*pszDest++ = *pszTmp++;
	}

	if( uiFlags & FLM_PRINTF_MINUS_FLAG)
	{
		// Left justify
		while( uiLength < uiWidth)
		{
			*pszDest++ = ' ';
			uiWidth--;
		}
	}

	*pszDest = 0;
	pInfo->pszDestStr = pszDest;
}

/****************************************************************************
Desc:		Default character formatter.
			Prints the character specified by VALUE in 'c', or the '%' character.
			Format: %[flags][width][.prec]'c'
				flags	= <not used>
				width	= <not used>
				prec	= <not used>
****************************************************************************/
FSTATIC void flmSprintfCharFormatter(
	char					ucFormatChar,
	FLMUINT				, // uiWidth,
	FLMUINT				, // uiPrecision,
	FLMUINT				, // uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	*pInfo->pszDestStr++ = (char)((ucFormatChar == '%')
										? '%'
										: f_va_arg( *args, int));
	*pInfo->pszDestStr = 0;
}

/****************************************************************************
Desc:	Default error formatter.
****************************************************************************/
FSTATIC void flmSprintfErrorFormatter(
	char					ucFormatChar,
	FLMUINT				, // uiWidth,
	FLMUINT				, // uiPrecision,
	FLMUINT				, // uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	FLMUINT			uiErrorCode = (FLMUINT)f_va_arg( *args, unsigned);

	if( ucFormatChar == 'e')
	{
		pInfo->pszDestStr +=
			f_sprintf( pInfo->pszDestStr, "%s (%04X)",
				FlmErrorString( (RCODE)uiErrorCode),
				(unsigned)uiErrorCode);
	}
	else
	{
		pInfo->pszDestStr +=
			f_sprintf( pInfo->pszDestStr, "%04X", (unsigned)uiErrorCode);
	}
}

/****************************************************************************
Desc:		Unknown format handler
****************************************************************************/
FSTATIC void flmSprintfNotHandledFormatter(
	char					, // ucFormatChar,
	FLMUINT				, // uiWidth,
	FLMUINT				, // uiPrecision,
	FLMUINT				, // uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			) // args)
{
	flmAssert( 0);
	*pInfo->pszDestStr++ = '?';
	*pInfo->pszDestStr = 0;
}

/****************************************************************************
Desc:		FLAIM's vsprintf
****************************************************************************/
FLMINT f_vsprintf(
	char *			pszDestStr,
	const char *	pszFormat,
	f_va_list *		args)
{
	F_SPRINTF_INFO info;

	info.pszDestStr = pszDestStr;
	flmSprintfParseArgs( pszFormat, &info, args);
	*info.pszDestStr = 0;

	return( (FLMINT)(info.pszDestStr - pszDestStr));
}

/****************************************************************************
Desc:		FLAIM's sprintf
****************************************************************************/
FLMINT f_sprintf(
	char *			pszDestStr,
	const char *	pszFormat,
	...)
{
	FLMINT		iLen;
	f_va_list	args;

	f_va_start(args, pszFormat);
	iLen = f_vsprintf( pszDestStr, pszFormat, &args);
	f_va_end(args);

	return( iLen);
}
