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

/****************************************************************************
Desc:		Parameter 'format' points to text following a '%' sign. Process
			legal field information.	Leave 'format' pointing at the format
			specifier char.
****************************************************************************/
void F_Printf::processFieldInfo(
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
void F_Printf::processFormatString(
	FLMUINT					uiLen,
	...)
{
	f_va_list	args;

	f_va_start( args, uiLen);
	if( uiLen)
	{
		stringFormatter( 0, uiLen, uiLen, 0, &args);
	}
	f_va_end( args);
}

/****************************************************************************
Desc:		Parse arguments in format string, calling appropriate handlers
****************************************************************************/
void F_Printf::parseArgs(
	const char *		pszFormat,
	f_va_list *			args)
{
	char				cFormatChar;
	FLMUINT			uiFlags;
	FLMUINT			uiWidth;
	FLMUINT			uiPrecision;
	const char *	pszTextStart = pszFormat;

	while( (cFormatChar = *pszFormat++) != 0)
	{
		if( cFormatChar != '%')
		{
			continue;
		}

		uiWidth = (FLMUINT)(pszFormat - pszTextStart - 1);
		processFormatString( uiWidth, pszTextStart);

		processFieldInfo( &pszFormat, &uiWidth,
			&uiPrecision, &uiFlags, args);

		cFormatChar = *pszFormat++;
		switch( cFormatChar)
		{
			case '%':
			case 'c':
				charFormatter( cFormatChar, args);
				break;

			case 'd':
			case 'i':
			case 'o':
			case 'u':
			case 'x':
			case 'X':
			case 'p':
				if( cFormatChar == 'i')
				{
					cFormatChar = 'd';
				}

				numberFormatter( cFormatChar, uiWidth, uiPrecision, uiFlags, args);
				break;

			case 'B':
			case 'F':
			
				// If we are not outputting a log message, colors are simply
				// stripped out.
				
				if (m_pLogMsg)
				{
					colorFormatter( cFormatChar, (eColorType)uiWidth, uiFlags);
				}
				break;
				
			case 's':
			case 'S':
			case 'U':
				stringFormatter( cFormatChar, uiWidth, uiPrecision, uiFlags, args);
				break;

			case 'e':
			case 'E':
				errorFormatter( args);
				break;

			default:
				notHandledFormatter();
				break;
		}
		pszTextStart = pszFormat;
	}

	processFormatString( (FLMUINT)(pszFormat - pszTextStart - 1), pszTextStart);
}

/****************************************************************************
Desc:		Default string formatter.
****************************************************************************/
void F_Printf::stringFormatter(
	char					cFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	f_va_list *			args)
{
	static const char	pszNullPointerStr[] = "<null>";
	FLMUINT				uiOutputLen;
	FLMUINT				uiCount;
	FLMUNICODE *		pUnicode;
	const char *		pszStr = f_va_arg( *args, char *);

	if (!pszStr)
	{
		uiOutputLen = f_strlen( pszNullPointerStr);
	}
	else if (cFormatChar == 'S')
	{
		uiOutputLen = *pszStr++;
	}
	else
	{
		if (cFormatChar == 'U')
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

	if (uiPrecision > 0 && uiOutputLen > uiPrecision)
	{
		uiOutputLen = uiPrecision;
	}

	uiCount = uiWidth - uiOutputLen;

	if (uiOutputLen < uiWidth && !(uiFlags & FLM_PRINTF_MINUS_FLAG))
	{
		// Right justify
		memsetChar( ' ', uiCount);
	}

	if( !pszStr)
	{
		outputStr( pszNullPointerStr, uiOutputLen);
	}
	else if (cFormatChar == 'U')
	{
		FLMUINT		uiBytesOutput = 0;

		for( pUnicode = (FLMUNICODE *)pszStr;
			uiBytesOutput < uiOutputLen && *pUnicode; pUnicode++)
		{
			if( *pUnicode >= 32 && *pUnicode <= 127)
			{
				outputChar( (char)(*pUnicode));
				uiBytesOutput++;
			}
			else
			{
				char			szTmpBuf[ 8];
				FLMUINT		uiTmpLen;

				szTmpBuf[ 0] = '~';
				szTmpBuf[ 1] = '[';
				uiTmpLen = printNumber( (FLMUINT64)(*pUnicode),
					16, TRUE, FALSE, &szTmpBuf[ 2]);
				uiTmpLen += 2;
				szTmpBuf[ uiTmpLen] = ']';
				szTmpBuf[ uiTmpLen + 1] = 0;
				
				if ((uiBytesOutput = uiTmpLen + 2) > uiOutputLen)
				{
					uiBytesOutput = uiOutputLen;
				}

				outputStr( szTmpBuf, uiBytesOutput); 
			}
		}
	}
	else
	{
		outputStr( pszStr, uiOutputLen);
	}

	if (uiOutputLen < uiWidth && (uiFlags & FLM_PRINTF_MINUS_FLAG))
	{
		// Left justify

		memsetChar( ' ', uiCount);
	}
}

/****************************************************************************
Desc:	Output the current log buffer - only called when logging.
****************************************************************************/
void F_Printf::outputLogBuffer( void)
{
	if (m_uiCharOffset)
	{
		m_szLogBuf [m_uiCharOffset] = 0;
		
		m_pLogMsg->appendString( m_szLogBuf);
		
		// Reset to start filling from the beginning of the buffer.
		
		m_uiCharOffset = 0;
	}
}

/****************************************************************************
Desc:		Change colors - may only push or pop a color on to the color stack.
****************************************************************************/
void F_Printf::colorFormatter(
	char			cFormatChar,
	eColorType	eColor,
	FLMUINT		uiFlags)
{
	
	// Color formatting is ignored if there is not a log message object.
	
	if (m_pLogMsg)
	{
		
		// Before changing colors, output the current log buffer.
		
		outputLogBuffer();
	
		if (cFormatChar == 'F')	// Foreground color
		{
			if (uiFlags & FLM_PRINTF_PLUS_FLAG)
			{
				m_pLogMsg->pushForegroundColor();
			}
			else if (uiFlags & FLM_PRINTF_MINUS_FLAG)
			{
				m_pLogMsg->popForegroundColor();
			}
			else if (m_eCurrentForeColor != eColor)
			{
				m_eCurrentForeColor = eColor;
				m_pLogMsg->changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
			}
		}
		else	// cFormatChar == 'B' - background color
		{
			if (uiFlags & FLM_PRINTF_PLUS_FLAG)
			{
				m_pLogMsg->pushBackgroundColor();
			}
			else if (uiFlags & FLM_PRINTF_MINUS_FLAG)
			{
				m_pLogMsg->popBackgroundColor();
			}
			else if (m_eCurrentBackColor != eColor)
			{
				m_eCurrentBackColor = eColor;
				m_pLogMsg->changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
			}
		}
	}
}

/****************************************************************************
Desc:		Converts a number to a string
****************************************************************************/
FLMUINT F_Printf::printNumber(
	FLMUINT64		ui64Val,
	FLMUINT			uiBase,
	FLMBOOL			bUpperCase,
	FLMBOOL			bCommas,
	char *			pszBuf)
{
	char			cChar;
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
		cChar = (char)(ui64Val % uiBase);
		pszBuf[ uiOffset++] = (char)(cChar > 9
			? cChar + (char)(bUpperCase ? 'A' : 'a') - 10
			: cChar + '0');
		uiDigitCount++;

		if( (ui64Val = (ui64Val / uiBase)) == 0)
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
		cChar = pszBuf[ uiLoop];
		pszBuf[ uiLoop] = pszBuf[ uiOffset - uiLoop - 1];
		pszBuf[ uiOffset - uiLoop - 1] = cChar;
	}

	pszBuf[ uiOffset] = 0;
	return( uiOffset);
}

/****************************************************************************
Desc:		Default number formatter.
			Format: %[flags][width][.prec]'E'
****************************************************************************/
void F_Printf::numberFormatter(
	char					cFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	f_va_list *			args)
{
	FLMUINT		uiPrefix = FLM_PREFIX_NONE;
	FLMUINT		uiLength;
	FLMUINT		uiBase = 10;
	char			cNumberBuffer[ 64];
	FLMBOOL		bUpperCase = FALSE;
	FLMBOOL		bCommas = FALSE;
	FLMUINT64	ui64Val;

	if (cFormatChar == 'p')
	{
		ui64Val = (FLMUINT64)((FLMUINT)f_va_arg( *args, void *));
		uiFlags |= FLM_PRINTF_POUND_FLAG;
	}
	else if (cFormatChar != 'd')
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

		if (uiFlags & FLM_PRINTF_SHORT_FLAG)
		{
			ui64Val = (FLMUINT64)((FLMINT64)f_va_arg( *args, int));
		}
		else if (uiFlags & (FLM_PRINTF_LONG_FLAG | FLM_PRINTF_DOUBLE_FLAG))
		{
			ui64Val = (FLMUINT64)((FLMINT64)f_va_arg( *args, long int));
		}
		else if (uiFlags & FLM_PRINTF_INT64_FLAG)
		{
			ui64Val = (FLMUINT64)f_va_arg( *args, FLMINT64);
		}
		else
		{
			ui64Val = (FLMUINT64)((FLMINT64)f_va_arg( *args, int));
		}
	}

	switch (cFormatChar)
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
			if ((uiFlags & FLM_PRINTF_POUND_FLAG) && ui64Val)
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

	if (cFormatChar == 'X')
	{
		bUpperCase = TRUE;
	}

	if ((uiFlags & FLM_PRINTF_COMMA_FLAG) && uiBase == 10)
	{
		bCommas = TRUE;
	}

	uiLength = printNumber( ui64Val, uiBase, bUpperCase, bCommas, cNumberBuffer);

	if (uiWidth < uiLength)
	{
		uiWidth = uiLength;
	}

	if (uiFlags & FLM_PRINTF_ZERO_FLAG)
	{
		// Zero fill

		uiPrecision = uiWidth;
	}
	else if (!(uiFlags & FLM_PRINTF_MINUS_FLAG))
	{
		// Right justify

		while (uiWidth > uiLength && uiWidth > uiPrecision)
		{
			outputChar( ' ');
			uiWidth--;
		}
	}

	// Handle the prefix (if any)

	switch (uiPrefix)
	{
		case FLM_PREFIX_NONE:
			break;

		case FLM_PREFIX_MINUS:
			outputChar( '-');
			break;

		case FLM_PREFIX_PLUS:
			outputChar( '+');
			break;

		case FLM_PREFIX_POUND:
		{
			outputStr( "0x", 2);
			break;
		}

		default:
			f_assert( 0);
			break;
	}

	// Handle the precision

	if (bCommas && uiPrecision && (uiPrecision % 4) == 0)
	{
		uiPrecision--;
	}

	while (uiLength < uiPrecision)
	{
		if (bCommas && (uiPrecision % 4) == 0)
		{
			outputChar( ',');
			uiPrecision--;
			uiWidth--;
			continue;
		}

		outputChar( '0');
		uiPrecision--;
		uiWidth--;
	}

	// Output the number

	outputStr( cNumberBuffer, uiLength);	

	if (uiFlags & FLM_PRINTF_MINUS_FLAG)
	{
		// Left justify
		if (uiWidth > uiLength)
		{
			memsetChar( ' ', (uiWidth - uiLength));
		}
	}
}

/****************************************************************************
Desc:		Default character formatter.
			Prints the character specified by VALUE in 'c', or the '%' character.
			Format: %[flags][width][.prec]'c'
				flags	= <not used>
				width	= <not used>
				prec	= <not used>
****************************************************************************/
void F_Printf::charFormatter(
	char					cFormatChar,
	f_va_list *			args)
{
	char	cChar = (char)((cFormatChar == '%')
								? (char)'%'
								: (char)f_va_arg( *args, int));
	outputChar( cChar);
}

/****************************************************************************
Desc:	Default error formatter.
****************************************************************************/
void F_Printf::errorFormatter(
	f_va_list *			args)
{
	numberFormatter( 'X', 4, 0,
		FLM_PRINTF_SHORT_FLAG | FLM_PRINTF_ZERO_FLAG | FLM_PRINTF_POUND_FLAG,
		args);
}

/****************************************************************************
Desc:		Unknown format handler
****************************************************************************/
void F_Printf::notHandledFormatter( void)
{
	f_assert( 0);
	outputChar( '?');
}

/****************************************************************************
Desc:		FLAIM's vsprintf
****************************************************************************/
FLMINT FLMAPI F_Printf::strvPrintf(
	char *			pszDestStr,
	const char *	pszFormat,
	f_va_list *		args)
{
	m_pszDestStr = pszDestStr;
	m_pLogMsg = NULL;
	parseArgs( pszFormat, args);
	*m_pszDestStr = 0;

	return( (FLMINT)(m_pszDestStr - pszDestStr));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_Printf::strPrintf(
	char *			pszDestStr,
	const char *	pszFormat,
	...)
{
	f_va_list	args;

	m_pszDestStr = pszDestStr;
	m_pLogMsg = NULL;
	f_va_start(args, pszFormat);
	parseArgs( pszFormat, &args);
	f_va_end(args);
	*m_pszDestStr = 0;

	return( (FLMINT)(m_pszDestStr - pszDestStr));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_Printf::logvPrintf(
	IF_LogMessageClient *	pLogMsg,
	const char *				pszFormat,
	f_va_list *					args)
{
	m_pszDestStr = NULL;
	m_uiNumLogChars = 0;
	m_uiCharOffset = 0;
	m_pLogMsg = pLogMsg;
	m_eCurrentForeColor = FLM_LIGHTGRAY;
	m_eCurrentBackColor = FLM_BLACK;
	m_pLogMsg->changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
	parseArgs( pszFormat, args);
	outputLogBuffer();

	return( (FLMINT)m_uiNumLogChars);
}
			
/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_Printf::logPrintf(
	IF_LogMessageClient *	pLogMsg,
	const char *				pszFormat,
	...)
{
	f_va_list	args;

	m_pszDestStr = NULL;
	m_uiNumLogChars = 0;
	m_uiCharOffset = 0;
	m_pLogMsg = pLogMsg;
	m_eCurrentForeColor = FLM_BLACK;
	m_eCurrentBackColor = FLM_LIGHTGRAY;
	m_pLogMsg->changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
	f_va_start(args, pszFormat);
	parseArgs( pszFormat, &args);
	f_va_end(args);
	outputLogBuffer();

	return( m_uiNumLogChars);
}

/****************************************************************************
Desc:		FLAIM's vsprintf
****************************************************************************/
FLMINT FLMAPI f_vsprintf(
	char *			pszDestStr,
	const char *	pszFormat,
	f_va_list *		args)
{
	F_Printf		formatter;
	
	return( formatter.strvPrintf( pszDestStr, pszFormat, args));
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
	F_Printf		formatter;

	f_va_start(args, pszFormat);
	iLen = formatter.strvPrintf( pszDestStr, pszFormat, &args);
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
	FLMINT			iLen;
	f_va_list		args;
	char				szTmpBuf[ 512];
	F_Printf			formatter;

	f_va_start( args, pszFormat);
	iLen = formatter.strvPrintf( szTmpBuf, pszFormat, &args);
	f_va_end(args);
	
#ifndef FLM_RING_ZERO_NLM
	fprintf( stdout, szTmpBuf);
	fflush( stdout);
#endif

	return( iLen);
}

