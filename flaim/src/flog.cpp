//-------------------------------------------------------------------------
// Desc:	Message logging.
// Tabs:	3
//
//		Copyright (c) 2001-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flog.cpp 12331 2006-01-23 10:19:55 -0700 (Mon, 23 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

// Static functions

FSTATIC void flmLogProcessFieldInfo(
	const char **		ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args);

FSTATIC RCODE flmLogProcessFormatString(
	FLMUINT				uiLen,
	F_LogMessage *		pLogMessage, ...);

FSTATIC RCODE flmLogParsePrintfArgs(
	const char *		pszFormat,
	f_va_list *			args,
	F_LogMessage *		pLogMessage);

FSTATIC RCODE flmLogStringFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

FSTATIC FLMUINT flmLogPrintNumber(
	FLMUINT				uiNumber,
	FLMUINT				uiBase,
	char *				pszBuffer);

FSTATIC RCODE flmLogNumberFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

FSTATIC RCODE flmLogErrorFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

FSTATIC RCODE flmLogColorFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

FSTATIC RCODE flmLogCharFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

FSTATIC RCODE flmLogNotHandledFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

// Percent formating prefixes

#define P_NONE			0
#define P_MINUS 		1
#define P_PLUS			2
#define P_POUND 		3

// Width and precision flags

#define MINUS_FLAG	0x0001
#define PLUS_FLAG		0x0002
#define SPACE_FLAG	0x0004
#define POUND_FLAG	0x0008
#define ZERO_FLAG		0x0010
#define SHORT_FLAG	0x0020
#define LONG_FLAG		0x0040
#define DOUBLE_FLAG 	0x0080

// Format handlers

typedef RCODE (*FORMATHANDLER)(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWdth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args);

typedef struct FORMATTERTABLE
{
	FORMATHANDLER		formatTextHandler;
	FORMATHANDLER		percentHandler;
	FORMATHANDLER		lowerCaseHandlers[ 26];
	FORMATHANDLER		upperCaseHandlers[ 26];
} FORMATTERTABLE;

/****************************************************************************
Desc:	Default formatter table
****************************************************************************/
FSTATIC FORMATTERTABLE flmLogFormatHandlers = 
{
	flmLogStringFormatter,
	flmLogCharFormatter,
	{
		/* a */ flmLogNotHandledFormatter,
		/* b */ flmLogNotHandledFormatter,
		/* c */ flmLogCharFormatter,
		/* d */ flmLogNumberFormatter,
		/* e */ flmLogErrorFormatter,
		/* f */ flmLogNotHandledFormatter,
		/* g */ flmLogNotHandledFormatter,
		/* h */ flmLogNotHandledFormatter,
		/* i */ flmLogNotHandledFormatter,
		/* j */ flmLogNotHandledFormatter,
		/* k */ flmLogNotHandledFormatter,
		/* l */ flmLogNotHandledFormatter,
		/* m */ flmLogNotHandledFormatter,
		/* n */ flmLogNotHandledFormatter,
		/* o */ flmLogNumberFormatter,
		/* p */ flmLogNotHandledFormatter,
		/* q */ flmLogNotHandledFormatter,
		/* r */ flmLogNotHandledFormatter,
		/* s */ flmLogStringFormatter,
		/* t */ flmLogNotHandledFormatter,
		/* u */ flmLogNumberFormatter,
		/* v */ flmLogNotHandledFormatter,
		/* w */ flmLogNotHandledFormatter,
		/* x */ flmLogNumberFormatter,
		/* y */ flmLogNotHandledFormatter,
		/* z */ flmLogNotHandledFormatter,
	},
	{
		/* A */ flmLogNotHandledFormatter,
		/* B */ flmLogNotHandledFormatter,
		/* C */ flmLogColorFormatter,
		/* D */ flmLogNotHandledFormatter,
		/* E */ flmLogErrorFormatter,
		/* F */ flmLogNotHandledFormatter,
		/* G */ flmLogNotHandledFormatter,
		/* H */ flmLogNotHandledFormatter,
		/* I */ flmLogNotHandledFormatter,
		/* J */ flmLogNotHandledFormatter,
		/* K */ flmLogNotHandledFormatter,
		/* L */ flmLogNotHandledFormatter,
		/* M */ flmLogNotHandledFormatter,
		/* N */ flmLogNotHandledFormatter,
		/* O */ flmLogNotHandledFormatter,
		/* P */ flmLogNotHandledFormatter,
		/* Q */ flmLogNotHandledFormatter,
		/* R */ flmLogNotHandledFormatter,
		/* S */ flmLogStringFormatter,
		/* T */ flmLogNotHandledFormatter,
		/* U */ flmLogNotHandledFormatter,
		/* V */ flmLogNotHandledFormatter,
		/* W */ flmLogNotHandledFormatter,
		/* X */ flmLogNumberFormatter,
		/* Y */ flmLogNotHandledFormatter,
		/* Z */ flmLogNotHandledFormatter,
	}
};

/****************************************************************************
Desc:	*ppszFormat points to text following a '%' sign.  Process legal field
		information.  Leave *ppszFormat pointing at the format specifier char.
****************************************************************************/
FSTATIC void flmLogProcessFieldInfo(
	const char **		ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args)
{
	const char *		pszTmp = *ppszFormat;

	/* process flags */
	*puiFlags = 0;

	while( *pszTmp == '-' || *pszTmp == '+' || *pszTmp == ' ' ||
		*pszTmp == '#' || *pszTmp == '0')
	{
		switch( *pszTmp)
		{
			case '-':
				*puiFlags |= MINUS_FLAG;
				break;
			case '+':
				*puiFlags |= PLUS_FLAG;
				break;
			case ' ':
				*puiFlags |= SPACE_FLAG;
				break;
			case '#':
				*puiFlags |= POUND_FLAG;
				break;
			case '0':
				*puiFlags |= ZERO_FLAG;
				break;
		}
		pszTmp++;
	}

	/* process width */
	
	*puiWidth = 0;
	if( *pszTmp == '*')
	{
		*puiWidth = f_va_arg( *args, unsigned int);
		pszTmp++;
	}
	else while( *pszTmp >= '0' && *pszTmp <= '9')
	{
		*puiWidth = (*puiWidth * 10) + (*pszTmp - '0');
		pszTmp++;
	}

	/* process precision */

	*puiPrecision = 0;
	if( *pszTmp == '.')
	{
		pszTmp++;
		if( *pszTmp == '*')
		{
			*puiPrecision = f_va_arg( *args, unsigned int);
			pszTmp++;
		}
		else while( *pszTmp >= '0' && *pszTmp <= '9')
		{
			*puiPrecision = (*puiPrecision * 10) + (*pszTmp - '0');
			pszTmp++;
		}
	}

	/* size modifiers */

	switch( *pszTmp)
	{
		case 'L':
			*puiFlags |= DOUBLE_FLAG;
			pszTmp++;
			break;
		case 'l':
			*puiFlags |= LONG_FLAG;
			pszTmp++;
			break;
		case 'h':
			*puiFlags |= SHORT_FLAG;
			pszTmp++;
			break;
	}

	*ppszFormat = pszTmp;
}

/****************************************************************************
Desc:	Handle text portions of the format string
****************************************************************************/
FSTATIC RCODE flmLogProcessFormatString(
	FLMUINT				uiLen,
	F_LogMessage *		pLogMessage, ...)
{
	RCODE			rc = FERR_OK;
	f_va_list	args;

	f_va_start( args, pLogMessage);
	if( uiLen && flmLogFormatHandlers.formatTextHandler)
	{
		rc = flmLogFormatHandlers.formatTextHandler( 
			0, uiLen, uiLen, 0, pLogMessage, (f_va_list *)&args);
	}
	f_va_end(args);

	return( rc);
}

/****************************************************************************
Desc:	Parse arguments in format string, calling appropriate handlers
****************************************************************************/
FSTATIC RCODE flmLogParsePrintfArgs(
	const char *		pszFormat,
	f_va_list *			args,
	F_LogMessage *		pLogMessage)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			ucChar;
	FLMUINT			uiFlags;
	FLMUINT			uiWidth;
	FLMUINT			uiPrecision;
	const char *	pszTextStart = pszFormat;
	FORMATHANDLER	fnHandler;

	while( (ucChar = *pszFormat++) != 0)
	{
		if( ucChar != '%')
		{
			// Handle invalid characters
			if( ucChar < ASCII_SPACE || ucChar > ASCII_TILDE)
			{
				uiWidth = (FLMUINT)(pszFormat - pszTextStart - 1);

				if( uiWidth)
				{
					if( RC_BAD( rc = flmLogProcessFormatString( uiWidth,
						pLogMessage, pszTextStart)))
					{
						goto Exit;
					}
				}

				// Only call newline() if ASCII_NEWLINE character.
				// We will skip ASCII_CR characters

				if( ucChar == ASCII_NEWLINE)
				{
					pLogMessage->newline();
				}
				pszTextStart = pszFormat;
			}

			continue;
		}

		uiWidth = (FLMUINT)(pszFormat - pszTextStart - 1);
		if( RC_BAD( rc = flmLogProcessFormatString( uiWidth, 
			pLogMessage, pszTextStart)))
		{
			goto Exit;
		}

		flmLogProcessFieldInfo( &pszFormat, &uiWidth, 
			&uiPrecision, &uiFlags, args);

		if( (ucChar = *pszFormat++) >= 'a' && ucChar <= 'z')
		{
			fnHandler = flmLogFormatHandlers.lowerCaseHandlers[ ucChar - 'a'];
		}
		else if( ucChar >= 'A' && ucChar <= 'Z')
		{
			fnHandler = flmLogFormatHandlers.upperCaseHandlers[ ucChar - 'A'];
		}
		else if( ucChar == '%')
		{
			fnHandler = flmLogFormatHandlers.percentHandler;
		}
		else
		{
			fnHandler = flmLogNotHandledFormatter;
		}

		if( RC_BAD( rc = fnHandler( ucChar, uiWidth, 
			uiPrecision, uiFlags, pLogMessage, args)))
		{
			goto Exit;
		}
		pszTextStart = pszFormat;
	}

	if( RC_BAD( rc = flmLogProcessFormatString( 
		(FLMUINT)(pszFormat - pszTextStart - 1),
		pLogMessage, pszTextStart)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Default string formatter.
			Prints the ascii string specified by ADDRESS in 's'.
			Prints length preceeded string specified by ADDRESS in 'S'.
			Prints unicode string specified by ADDRESS in 'U'.
			Format: %[flags][width][.prec]'s'|'S'|'U'
				flags	= '-' left justifies if string length < width
				width	= minimum number of characters to print
				prec	= maximum number of characters to print
****************************************************************************/
FSTATIC RCODE flmLogStringFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args)
{
	RCODE					rc = FERR_OK;
	const char *		pszNullPointerStr = "<null>";
	FLMUINT				uiLength;
	FLMUINT				uiCount;
	char					szOutputBuf[ 128];
	char *				pszDest = &szOutputBuf[ 0];
	FLMUINT				uiMaxLen = sizeof( szOutputBuf) - 1;
	const char *		pszString = f_va_arg( *args, char *);
	void *				pAllocBuf = NULL;

	if( uiWidth >= uiMaxLen)
	{
		// Need to allocate a temporary buffer

		uiMaxLen = uiWidth;
		if( RC_BAD( rc = f_alloc( (FLMUINT)(uiMaxLen + 1), &pAllocBuf)))
		{
			goto Exit;
		}
		
		pszDest = (char *)pAllocBuf;
	}

	if( !pszString)
	{
		uiLength = f_strlen( pszNullPointerStr);
	}
	else if( ucFormatChar == 'S')
	{
		uiLength = (FLMUINT)(*pszString);
		pszString++;
	}
	else
	{
		uiLength = ucFormatChar == 0 ? uiWidth : f_strlen( pszString);
	}

	if( uiPrecision > 0 && uiLength > uiPrecision)
	{
		uiLength = uiPrecision;
	}
	uiCount = uiWidth - uiLength;

	if( uiLength < uiWidth && !(uiFlags & MINUS_FLAG))
	{ 
		// right justify
		f_memset( pszDest, ' ', uiCount);
		pszDest += uiCount;
	}

	if( !pszString)
	{
		f_memcpy( pszDest, pszNullPointerStr, uiLength);
	}
	else
	{
		f_memcpy( pszDest, pszString, uiLength);
	}

	pszDest += uiLength;

	// left justify

	if( uiLength < uiWidth && (uiFlags & MINUS_FLAG))
	{ 
		f_memset( pszDest, ' ', uiCount);
		pszDest += uiCount;
	}

	*pszDest = 0;
	pLogMessage->appendString( szOutputBuf);

Exit:

	if( pAllocBuf)
	{
		f_free( &pAllocBuf);
	}

	return( rc);
}

/****************************************************************************
Desc:	This is used by printf to output numbers.  It uses recursion to
		separate the numbers down to individual digits and then calls 
		PrintDigit to output each digit.
****************************************************************************/
FSTATIC FLMUINT flmLogPrintNumber(
	FLMUINT				uiNumber,
	FLMUINT				uiBase,
	char *				pszBuffer)
{
	FLMBYTE	ucChar = (FLMBYTE)( uiNumber % uiBase);
	FLMUINT	uiIndex = uiNumber / uiBase;

	uiIndex = uiIndex ? flmLogPrintNumber( uiIndex, uiBase, pszBuffer) : 0;
	pszBuffer[ uiIndex] = (FLMBYTE)(ucChar > 9 
						? ucChar + 'a' - 10 
						: ucChar + '0');

	return( uiIndex + 1);
}

/****************************************************************************
Desc:	Default number formatter.
			Prints the number specified by VALUE in 'd', 'o', 'u', 'x', or 'X'
			Format: %[flags][width][.prec]'E'
				flags	= 'h' value is uint16
					'l' value is uint32
					'-' left align result
					'+' print plus sign if positive
					'#' print '0x' in front of hex numbers
					'0' zero-fill
			width	= minimum number of characters to print
			prec	= maximum number of characters to print (truncates or rounds)
****************************************************************************/
FSTATIC RCODE flmLogNumberFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args)
{
	FLMUINT			uiPrefix = P_NONE;
	FLMUINT			uiLength;
	FLMUINT			uiBase = 10;
	FLMUINT			uiLoop;
	char				szNumberBuf[ 32];
	char				szOutputBuf[ 128];
	char *			pszDest = &szOutputBuf[ 0];
	FLMUINT			uiMaxLen = sizeof( szOutputBuf) - 1;
	char *			pszTmp;
	FLMUINT			uiArg;

	if( uiFlags & SHORT_FLAG)
	{
		uiArg = (FLMUINT)f_va_arg( *args, int);
	}
	else if( uiFlags & LONG_FLAG)
	{
		uiArg = (FLMUINT)f_va_arg( *args, long int);
	}
	else
	{
		uiArg = (FLMUINT)f_va_arg( *args, int);
	}

	switch( ucFormatChar)
	{
		case 'd':
			if( (long)uiArg < 0)
			{	
				// handle negatives
				uiPrefix = P_MINUS;
				if( uiWidth > 0)
				{
					uiWidth--;
				}
				uiArg = (FLMUINT)(-((long)uiArg));
			}
			else if( uiFlags & PLUS_FLAG)
			{
				uiPrefix = P_PLUS;
				if( uiWidth > 0)
				{
					uiWidth--;
				}
			}
			break;

		case 'o':
			uiBase = 8;
			break;

		case 'x':
		case 'X':
			if( (uiFlags & POUND_FLAG) != 0 && uiArg)
			{
				uiPrefix = P_POUND;
				if( uiWidth > 1)
				{
					uiWidth -= 2;
				}
			}
			uiBase = 16;
			break;
	}

	uiLength = flmLogPrintNumber( uiArg, uiBase, szNumberBuf);
	szNumberBuf[ uiLength] = 0;

	if( ucFormatChar == 'X')
	{
		pszTmp = &szNumberBuf[ 0];
		while( *pszTmp)
		{
			if( (*pszTmp >= 'a') && (*pszTmp <= 'z'))
			{
				*pszTmp = (FLMBYTE)((*pszTmp - 'a') + 'A');
			}
			pszTmp++;
		}
	}

	if( uiWidth < uiLength)
	{
		uiWidth = uiLength;
	}

	if( uiFlags & ZERO_FLAG)
	{
		// zero fill
		uiPrecision = uiWidth;
	}
	else if( !(uiFlags & MINUS_FLAG))
	{
		// right justify
		while( uiWidth > uiLength && uiWidth > uiPrecision && uiMaxLen > 0)
		{
			*pszDest++ = ' ';
			uiMaxLen--;
			uiWidth--;
		}
	}

	// handle the prefix if any

	if( uiMaxLen)
	{
		switch( uiPrefix)
		{
			case P_MINUS:
				*pszDest++ = '-';
				uiMaxLen--;
				break;
			case P_PLUS:
				*pszDest++ = '+';
				uiMaxLen--;
				break;
			case P_POUND:
				*pszDest++ = '0';
				uiMaxLen--;
				*pszDest++ = ucFormatChar;
				uiMaxLen--;
				break;
		}
	}

	// handle the precision
	while( uiLength < uiPrecision && uiMaxLen)
	{
		*pszDest++ = '0';
		uiMaxLen--;
		uiPrecision--;
		uiWidth--;
	}

	// print the number
	for( uiLoop = uiLength, pszTmp = &szNumberBuf[ 0];
			uiLoop > 0 && uiMaxLen; uiLoop--, uiMaxLen--)
	{
		*pszDest++ = *pszTmp++;
	}

	if( uiFlags & MINUS_FLAG)
	{
		// left justify
		while( uiLength < uiWidth && uiMaxLen > 0)
		{
			*pszDest++ = ' ';
			uiMaxLen--;
			uiWidth--;
		}
	}

	*pszDest = 0;
	pLogMessage->appendString( szOutputBuf);

	return( FERR_OK);
}

/****************************************************************************
Desc:	Default error formatter.
****************************************************************************/
FSTATIC RCODE flmLogErrorFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args)
{
	RCODE				rc = FERR_OK;
	char				szOutputBuf[ 128];
	FLMUINT			uiErrorCode = (FLMUINT)f_va_arg( *args, unsigned);

	F_UNREFERENCED_PARM( ucFormatChar);
	F_UNREFERENCED_PARM( uiWidth);
	F_UNREFERENCED_PARM( uiPrecision);
	F_UNREFERENCED_PARM( uiFlags);

	if( uiErrorCode < 0x0000FFFF)
	{
		f_sprintf( szOutputBuf, "%s (0x%4.4X, %u)",
			FlmErrorString( (RCODE)uiErrorCode),
			(unsigned)uiErrorCode, (unsigned)uiErrorCode);
	}
	else
	{
		f_sprintf( szOutputBuf, "0x%8.8X, %d",
			(unsigned)uiErrorCode, (unsigned)uiErrorCode);
	}
	
	pLogMessage->appendString( szOutputBuf);

	return( rc);
}

/****************************************************************************
Desc:	Color formatter
****************************************************************************/
FSTATIC RCODE flmLogColorFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args)
{
	RCODE					rc = FERR_OK;

	F_UNREFERENCED_PARM( ucFormatChar);
	F_UNREFERENCED_PARM( args);

	if( uiFlags & PLUS_FLAG)
	{
		// Push a color onto the stack

		if( !uiWidth)
		{
			// Foreground

			pLogMessage->pushForegroundColor();
		}
		else
		{
			// Background

			pLogMessage->pushBackgroundColor();
		}
	}
	else if( uiFlags & MINUS_FLAG)
	{
		// Pop a color from the color stack

		if( !uiWidth)
		{
			// Foreground

			pLogMessage->popForegroundColor();
		}
		else
		{
			// Background

			pLogMessage->popBackgroundColor();
		}
	}
	else
	{
		FlmColorType	eForeground = (FlmColorType)(uiWidth + 1);
		FlmColorType	eBackground = (FlmColorType)(uiPrecision + 1);

		// Set a new foreground and/or background color

		if( eForeground >= FLM_NUM_COLORS || eBackground >= FLM_NUM_COLORS)
		{
			goto Exit;
		}

		pLogMessage->setColor( eForeground, eBackground);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Default character formatter.
			Prints the character specified by VALUE in 'c', or the '%' character.
			Format: %[flags][width][.prec]'c'
				flags	= <not used>
				width	= <not used>
				prec	= <not used>
****************************************************************************/
FSTATIC RCODE flmLogCharFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args)
{
	char			ucCharBuf[ 2];

	F_UNREFERENCED_PARM( uiWidth);
	F_UNREFERENCED_PARM( uiPrecision);
	F_UNREFERENCED_PARM( uiFlags);

	ucCharBuf[ 0] =  (FLMBYTE)((ucFormatChar == '%') 
								? '%' 
								: f_va_arg( *args, int));
	ucCharBuf[ 1] = 0;
	pLogMessage->appendString( ucCharBuf);

	return( FERR_OK);
}

/****************************************************************************
Desc:	Unhandled format strings
****************************************************************************/
FSTATIC RCODE flmLogNotHandledFormatter(
	FLMBYTE				ucFormatChar,
	FLMUINT				uiWidth,
	FLMUINT				uiPrecision,
	FLMUINT				uiFlags,
	F_LogMessage *		pLogMessage,
	f_va_list *			args)
{
	F_UNREFERENCED_PARM( ucFormatChar);
	F_UNREFERENCED_PARM( uiWidth);
	F_UNREFERENCED_PARM( uiPrecision);
	F_UNREFERENCED_PARM( uiFlags);
	F_UNREFERENCED_PARM( args);

	pLogMessage->appendString( "<no formatter>");
	return( FERR_OK);
}

/****************************************************************************
Desc:	Main entry point for printf functionality.
****************************************************************************/
void flmLogPrintf(
	F_LogMessage *		pLogMessage,
	const char *		szFormatStr, ...)
{
	f_va_list			args;

	f_va_start( args, szFormatStr);
	(void)flmLogParsePrintfArgs( szFormatStr, (f_va_list *)&args, pLogMessage);
	f_va_end( args);
}

/****************************************************************************
Desc:	Printf routine that accepts a va_list argument
****************************************************************************/
void flmLogVPrintf( 
	F_LogMessage *		pLogMessage,
	const char *		szFormatStr,
	f_va_list *			args)
{
	(void)flmLogParsePrintfArgs( szFormatStr, args, pLogMessage);
}

/****************************************************************************
Desc:	Returns an F_LogMessage object if logging is enabled for the
		specified message type
****************************************************************************/
F_LogMessage * flmBeginLogMessage(
	FlmLogMessageType			eMsgType,
	FlmLogMessageSeverity	eMsgSeverity)
{
	F_LogMessage *		pNewMsg = NULL;

	if( gv_FlmSysData.pLogger)
	{
		pNewMsg = gv_FlmSysData.pLogger->beginMessage( eMsgType, eMsgSeverity);
	}

	return( pNewMsg);
}

/****************************************************************************
Desc:	F_Logger constructor
****************************************************************************/
F_Logger::F_Logger()
{
	m_hMutex = F_MUTEX_NULL;
	m_bSetupCalled = FALSE;
	m_pbEnabledList = NULL;
}

/****************************************************************************
Desc:	F_Logger destructor
****************************************************************************/
F_Logger::~F_Logger()
{
	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
		f_mutexDestroy( &m_hMutex);
	}

	if( m_pbEnabledList)
	{
		f_free( &m_pbEnabledList);
	}
}

/****************************************************************************
Desc:	Set up the logger object
****************************************************************************/
RCODE F_Logger::setupLogger( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( !m_bSetupCalled);

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_calloc( 
		((FLMUINT)FLM_NUM_MESSAGE_TYPES) * sizeof( FLMBOOL), &m_pbEnabledList)))
	{
		goto Exit;
	}

	m_bSetupCalled = TRUE;

Exit:

	if( RC_BAD( rc))
	{
		if( m_hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &m_hMutex);
		}

		if( m_pbEnabledList)
		{
			f_free( &m_pbEnabledList);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns TRUE if the logger has been set up
****************************************************************************/
FLMBOOL F_Logger::loggerIsSetup( void)
{
	return( m_bSetupCalled);
}

/****************************************************************************
Desc:	Turns logging on for a specific message type
****************************************************************************/
void F_Logger::enableMessageType(
	FlmLogMessageType		eMsgType)
{
	FLMUINT		uiSlot = (FLMUINT)eMsgType;

	flmAssert( m_bSetupCalled);

	if( uiSlot < ((FLMUINT)FLM_NUM_MESSAGE_TYPES))
	{
		m_pbEnabledList[ uiSlot] = TRUE;
	}
}

/****************************************************************************
Desc:	Turns logging on for all message types
****************************************************************************/
void F_Logger::enableAllMessageTypes( void)
{
	FLMUINT		uiLoop;

	flmAssert( m_bSetupCalled);

	for( uiLoop = 0; uiLoop < ((FLMUINT)FLM_NUM_MESSAGE_TYPES); uiLoop++)
	{
		m_pbEnabledList[ uiLoop] = TRUE;
	}
}

/****************************************************************************
Desc:	Turs logging off for a specific message type
****************************************************************************/
void F_Logger::disableMessageType(
	FlmLogMessageType		eMsgType)
{
	FLMUINT		uiSlot = (FLMUINT)eMsgType;

	flmAssert( m_bSetupCalled);

	if( uiSlot < ((FLMUINT)FLM_NUM_MESSAGE_TYPES))
	{
		m_pbEnabledList[ uiSlot] = FALSE;
	}
}

/****************************************************************************
Desc:	Turns logging off for all message types
****************************************************************************/
void F_Logger::disableAllMessageTypes( void)
{
	flmAssert( m_bSetupCalled);
	f_memset( m_pbEnabledList, 0, 
		((FLMUINT)FLM_NUM_MESSAGE_TYPES) * sizeof( FLMBOOL));
}

/****************************************************************************
Desc:	Returns TRUE if the specified message type is being logged
****************************************************************************/
FLMBOOL F_Logger::messageTypeEnabled(
	FlmLogMessageType		eMsgType)
{
	FLMUINT		uiSlot = (FLMUINT)eMsgType;

	flmAssert( m_bSetupCalled);

	if( uiSlot < ((FLMUINT)FLM_NUM_MESSAGE_TYPES))
	{
		return( m_pbEnabledList[ uiSlot]);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_Logger::lockLogger( void)
{
	flmAssert( m_bSetupCalled);
	f_mutexLock( m_hMutex);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_Logger::unlockLogger( void)
{
	flmAssert( m_bSetupCalled);
	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:	Pushes the current foreground color onto the color stack
****************************************************************************/
void F_LogMessage::pushForegroundColor( void)
{
	if( m_uiForeColors < F_MAX_COLOR_STACK_SIZE)
	{
		m_eForeColors[ F_MAX_COLOR_STACK_SIZE - 
			(++m_uiForeColors)] = m_eCurrentForeColor;
	}
	else
	{
		m_uiForeColors++;
	}
}

/****************************************************************************
Desc:	Pushes the current background color onto the color stack
****************************************************************************/
void F_LogMessage::pushBackgroundColor( void)
{
	if( m_uiBackColors < F_MAX_COLOR_STACK_SIZE)
	{
		m_eBackColors[ F_MAX_COLOR_STACK_SIZE - 
			(++m_uiBackColors)] = m_eCurrentBackColor;
	}
	else
	{
		m_uiBackColors++;
	}
}

/****************************************************************************
Desc:	Pops the foreground color off of the top of the color stack
****************************************************************************/
void F_LogMessage::popForegroundColor( void)
{
	FlmColorType	eForeColor = m_eCurrentForeColor;

	// Pop a color from the color stack

	if( m_uiForeColors)
	{
		if( m_uiForeColors <= F_MAX_COLOR_STACK_SIZE)
		{
			eForeColor = m_eForeColors[ 
					F_MAX_COLOR_STACK_SIZE - m_uiForeColors];
		}
		m_uiForeColors--;
	}

	setColor( eForeColor, m_eCurrentBackColor);
}

/****************************************************************************
Desc:	Pops the background color off of the top of the color stack
****************************************************************************/
void F_LogMessage::popBackgroundColor( void)
{
	FlmColorType	eBackColor = m_eCurrentBackColor;

	// Pop a color from the color stack

	if( m_uiBackColors)
	{
		if( m_uiBackColors <= F_MAX_COLOR_STACK_SIZE)
		{
			eBackColor = m_eBackColors[ 
					F_MAX_COLOR_STACK_SIZE - m_uiBackColors];
		}
		m_uiBackColors--;
	}

	setColor( m_eCurrentForeColor, eBackColor);
}

/****************************************************************************
Desc: Sets the foreground and background colors of a message
****************************************************************************/
void F_LogMessage::setColor(
	FlmColorType	eForeColor,
	FlmColorType	eBackColor)
{
	if( eForeColor != m_eCurrentForeColor ||
		eBackColor != m_eCurrentBackColor)
	{
		m_eCurrentForeColor = eForeColor;
		m_eCurrentBackColor = eBackColor;
		changeColor( m_eCurrentForeColor, m_eCurrentBackColor);
	}
}

/****************************************************************************
Desc:	Ends a logging message
****************************************************************************/
void flmEndLogMessage(
	F_LogMessage **		ppLogMessage)
{
	if( *ppLogMessage)
	{
		(*ppLogMessage)->endMessage();
		(*ppLogMessage)->Release();
		*ppLogMessage = NULL;
	}
}
