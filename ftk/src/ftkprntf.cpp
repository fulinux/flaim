//------------------------------------------------------------------------------
// Desc: sprintf and vsprintf
//
// Tabs:	3
//
//		Copyright (c) 2002-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flprintf.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

FSTATIC void f_sprintfParseArgs(
	FLMBYTE *			pszFormat,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args);

FSTATIC void f_processFormatString(
	FLMUINT					uiLen,
	F_SPRINTF_INFO *		pInfo,
	...);

FSTATIC FLMUINT f_printNumber(
	FLMUINT64			ui64Val,
	FLMUINT				uiBase,
	FLMBOOL				bUpperCase,
	FLMBOOL				bCommas,
	FLMBYTE *			pszBuf);

/****************************************************************************
Desc:		Parameter 'format' points to text following a '%' sign. Process
			legal field information.	Leave 'format' pointing at the format
			specifier char.
****************************************************************************/
void f_sprintfProcessFieldInfo(
	FLMBYTE **			ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args)
{
	FLMBYTE *		pszFormat = *ppszFormat;

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
				f_assert( 0);
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
FSTATIC void f_processFormatString(
	FLMUINT					uiLen,
	F_SPRINTF_INFO *		pInfo,
	...)
{
	f_va_list	args;

	f_va_start( args, pInfo);
	if( uiLen)
	{
		f_sprintfStringFormatter( 0, uiLen, uiLen, 0, pInfo, &args);
	}
	f_va_end( args);
}

/****************************************************************************
Desc:		Parse arguments in format string, calling appropriate handlers
****************************************************************************/
FSTATIC void f_sprintfParseArgs(
	FLMBYTE *			pszFormat,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	FLMBYTE		ucFormatChar;
	FLMUINT		uiFlags;
	FLMUINT		uiWidth;
	FLMUINT		uiPrecision;
	FLMBYTE *	pszTextStart = pszFormat;

	while( (ucFormatChar = (FLMBYTE)*pszFormat++) != 0)
	{
		if( ucFormatChar != '%')
		{
			continue;
		}

		uiWidth = (FLMUINT)(pszFormat - pszTextStart - 1);
		f_processFormatString( uiWidth, pInfo, pszTextStart);

		f_sprintfProcessFieldInfo( &pszFormat, &uiWidth,
			&uiPrecision, &uiFlags, args);

		ucFormatChar = (unsigned char)*pszFormat++;
		switch( ucFormatChar)
		{
			case '%':
			case 'c':
				f_sprintfCharFormatter( ucFormatChar, uiWidth,
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

				f_sprintfNumberFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			case 's':
			case 'S':
			case 'U':
				f_sprintfStringFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			case 'e':
			case 'E':
				f_sprintfErrorFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;

			default:
				f_sprintfNotHandledFormatter( ucFormatChar, uiWidth,
					uiPrecision, uiFlags, pInfo, args);
				break;
		}
		pszTextStart = pszFormat;
	}

	f_processFormatString( (FLMUINT)(pszFormat - pszTextStart - 1),
		pInfo, pszTextStart);
}

/****************************************************************************
Desc:		Default string formatter.
****************************************************************************/
void f_sprintfStringFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	static const char	pszNullPointerStr[] = "<null>";
	FLMUINT				uiOutputLen;
	FLMUINT				uiCount;
	FLMUNICODE *		pUnicode;
	const char *		pszStr = f_va_arg( *args, char *);
	FLMBYTE *			pszDest = pInfo->pszDestStr;

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
				*pszDest++ = (FLMBYTE)(*pUnicode);
				uiBytesOutput++;
			}
			else
			{
				FLMBYTE		szTmpBuf[ 8];
				FLMUINT		uiTmpLen;
				FLMBYTE *	pszTmp;

				szTmpBuf[ 0] = '~';
				szTmpBuf[ 1] = '[';
				uiTmpLen = f_printNumber( (FLMUINT64)(*pUnicode),
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
FSTATIC FLMUINT f_printNumber(
	FLMUINT64		ui64Val,
	FLMUINT			uiBase,
	FLMBOOL			bUpperCase,
	FLMBOOL			bCommas,
	FLMBYTE *		pszBuf)
{
	FLMBYTE		ucChar;
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
		ucChar = (FLMBYTE)(ui64Val % uiBase);
		pszBuf[ uiOffset++] = (FLMBYTE)(ucChar > 9
			? ucChar + (bUpperCase ? 'A' : 'a') - 10
						: ucChar + '0');
		uiDigitCount++;

		if( (ui64Val = (ui64Val / uiBase)) == 0)
		{
			break;
		}

		if( bCommas && (uiDigitCount % 3) == 0)
		{
			pszBuf[ uiOffset++] = (FLMBYTE)',';
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
void f_sprintfNumberFormatter(
	FLMBYTE				ucFormatChar,
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
	FLMBYTE		ucNumberBuffer[ 64];
	FLMBOOL		bUpperCase = FALSE;
	FLMBOOL		bCommas = FALSE;
	FLMBYTE *	pszTmp;
	FLMBYTE *	pszDest = pInfo->pszDestStr;
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
			if( (uiFlags & FLM_PRINTF_POUND_FLAG) && ui64Val)
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

	uiLength = f_printNumber( ui64Val, uiBase, bUpperCase,
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
			f_assert( 0);
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
void f_sprintfCharFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				, // uiWidth,
	FLMUINT				, // uiPrecision,
	FLMUINT				, // uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			args)
{
	*pInfo->pszDestStr++ = (FLMBYTE)((ucFormatChar == '%')
										? '%'
										: f_va_arg( *args, int));
	*pInfo->pszDestStr = 0;
}

/****************************************************************************
Desc:	Default error formatter.
****************************************************************************/
void f_sprintfErrorFormatter(
	FLMBYTE				ucFormatChar,
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
			f_sprintf( (char *)pInfo->pszDestStr, "0x%04X",
				(unsigned)uiErrorCode);
	}
	else
	{
		pInfo->pszDestStr +=
			f_sprintf( (char *)pInfo->pszDestStr, "0x%04X",
			(unsigned)uiErrorCode);
	}
}

/****************************************************************************
Desc:		Unknown format handler
****************************************************************************/
void f_sprintfNotHandledFormatter(
	FLMBYTE				, // ucFormatChar,
	FLMUINT				, // uiWidth,
	FLMUINT				, // uiPrecision,
	FLMUINT				, // uiFlags,
	F_SPRINTF_INFO *	pInfo,
	f_va_list *			) // args)
{
	f_assert( 0);
	*pInfo->pszDestStr++ = '?';
	*pInfo->pszDestStr = 0;
}

/****************************************************************************
Desc:		FLAIM's vsprintf
****************************************************************************/
FLMINT FLMAPI f_vsprintf(
	char *			pszDestStr,
	const char *	pszFormat,
	f_va_list *		args)
{
	F_SPRINTF_INFO info;

	info.pszDestStr = (FLMBYTE *)pszDestStr;
	f_sprintfParseArgs( (FLMBYTE *)pszFormat, &info, args);
	*info.pszDestStr = 0;

	return( (FLMINT)(info.pszDestStr - (FLMBYTE *)pszDestStr));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_sprintf(
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

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_printf(
	const char *	pszFormat,
	...)
{
#ifndef FLM_RING_ZERO_NLM
	FLMINT		iLen;
	va_list		args;

	va_start(args, pszFormat);
	iLen = vprintf( pszFormat, args);
	va_end(args);

	return( iLen);
#else
	F_UNREFERENCED_PARM( pszFormat);
	return( 0);
#endif
}
