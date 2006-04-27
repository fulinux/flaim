//------------------------------------------------------------------------------
// Desc:	Contains routines for logging messages from within FLAIM.
//
// Tabs:	3
//
//		Copyright (c) 2001-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flog.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

static F_MUTEX					gv_hLoggerMutex = F_MUTEX_NULL;
static FLMUINT					gv_uiPendingLogMessages = 0;
static IF_LoggerClient *	gv_pLogger = NULL;

FSTATIC void f_logProcessFormatString(
	FLMUINT						uiLen,
	IF_LogMessageClient *	pLogMessage, ...);

FSTATIC void f_logParsePrintfArgs(
	FLMBYTE *					pszFormat,
	f_va_list *					args,
	IF_LogMessageClient *	pLogMessage);

FSTATIC void f_logStringFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args);

FSTATIC void f_logNumberFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args);

FSTATIC void f_logErrorFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args);

FSTATIC void f_logColorFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args);

FSTATIC void f_logCharFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args);

FSTATIC void f_logNotHandledFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args);

/****************************************************************************
Desc:	Handle text portions of the format string
****************************************************************************/
FSTATIC void f_logProcessFormatString(
	FLMUINT						uiLen,
	IF_LogMessageClient *	pLogMessage, ...)
{
	f_va_list	args;

	f_va_start( args, pLogMessage);
	if( uiLen)
	{
		f_logStringFormatter( 0, uiLen, uiLen, 0, pLogMessage, &args);
	}
	f_va_end( args);
}

/****************************************************************************
Desc:	Parse arguments in format string, calling appropriate handlers
****************************************************************************/
FSTATIC void f_logParsePrintfArgs(
	FLMBYTE *					pszFormat,
	f_va_list *					args,
	IF_LogMessageClient *	pLogMessage)
{
	FLMBYTE			ucChar;
	FLMUINT			uiFlags;
	FLMUINT			uiWidth;
	FLMUINT			uiPrecision;
	FLMBYTE *		pszTextStart = pszFormat;

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
					f_logProcessFormatString( uiWidth,
						pLogMessage, pszTextStart);
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
		f_logProcessFormatString( uiWidth, pLogMessage, pszTextStart);

		f_sprintfProcessFieldInfo( &pszFormat, &uiWidth,
			&uiPrecision, &uiFlags, args);

		ucChar = (unsigned char)*pszFormat++;
		switch( ucChar)
		{
			case '%':
			case 'c':
				f_logCharFormatter( ucChar, uiWidth,
					uiPrecision, uiFlags, pLogMessage, args);
				break;
			case 'C':
				f_logColorFormatter( ucChar, uiWidth,
					uiPrecision, uiFlags, pLogMessage, args);
				break;
			case 'd':
			case 'o':
			case 'u':
			case 'x':
			case 'X':
				f_logNumberFormatter( ucChar, uiWidth,
					uiPrecision, uiFlags, pLogMessage, args);
				break;
			case 's':
			case 'S':
				f_logStringFormatter( ucChar, uiWidth,
					uiPrecision, uiFlags, pLogMessage, args);
				break;
			case 'e':
				f_logErrorFormatter( ucChar, uiWidth,
					uiPrecision, uiFlags, pLogMessage, args);
				break;
			default:
				f_logNotHandledFormatter( ucChar, uiWidth,
					uiPrecision, uiFlags, pLogMessage, args);
				break;
		}

		pszTextStart = pszFormat;
	}

	f_logProcessFormatString( (FLMUINT)(pszFormat - pszTextStart - 1),
		pLogMessage, pszTextStart);
}

/****************************************************************************
Desc:	Default string formatter.
****************************************************************************/
FSTATIC void f_logStringFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args)
{
	FLMBYTE				ucOutputBuf[ 128];
	FLMBYTE *			pszDest = &ucOutputBuf[ 0];
	FLMUINT				uiMaxLen = sizeof( ucOutputBuf) - 1;
	void *				pAllocBuf = NULL;
	F_SPRINTF_INFO		info;

	if( uiWidth >= uiMaxLen)
	{
		// Need to allocate a temporary buffer

		uiMaxLen = uiWidth;
		if (RC_BAD( f_alloc( (FLMUINT)(uiMaxLen + 1), &pAllocBuf)))
		{
			uiWidth = uiMaxLen;
		}
		else
		{
			pszDest = (FLMBYTE *)pAllocBuf;
		}
	}

	f_memset( &info, 0, sizeof( info));
	info.pszDestStr = pszDest;

	f_sprintfStringFormatter( ucFormatChar, uiWidth, uiPrecision,
		uiFlags, &info, args);

	pLogMessage->appendString( (char *)pszDest);

	if( pAllocBuf)
	{
		f_free( &pAllocBuf);
	}
}

/****************************************************************************
Desc:	Default number formatter.
****************************************************************************/
FSTATIC void f_logNumberFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args)
{
	FLMBYTE				ucOutputBuf[ 128];
	F_SPRINTF_INFO		info;

	f_memset( &info, 0, sizeof( info));
	info.pszDestStr = &ucOutputBuf[ 0];
	f_sprintfNumberFormatter( ucFormatChar, uiWidth,
		uiPrecision, uiFlags, &info, args);
	pLogMessage->appendString( (char *)ucOutputBuf);
}

/****************************************************************************
Desc:	Default error formatter.
****************************************************************************/
FSTATIC void f_logErrorFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args)
{
	FLMBYTE				ucOutputBuf[ 128];
	F_SPRINTF_INFO		info;

	f_memset( &info, 0, sizeof( info));
	info.pszDestStr = &ucOutputBuf[ 0];
	f_sprintfErrorFormatter( ucFormatChar, uiWidth,
		uiPrecision, uiFlags, &info, args);
	pLogMessage->appendString( (char *)ucOutputBuf);
}

/****************************************************************************
Desc:	Color formatter
****************************************************************************/
FSTATIC void f_logColorFormatter(
	FLMBYTE,						// ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					// args
	)
{
	if( uiFlags & FLM_PRINTF_PLUS_FLAG)
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
	else if( uiFlags & FLM_PRINTF_MINUS_FLAG)
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
		eColorType	eForeground = (eColorType)(uiWidth + 1);
		eColorType	eBackground = (eColorType)(uiPrecision + 1);

		// Set a new foreground and/or background color

		if( eForeground >= FLM_NUM_COLORS || eBackground >= FLM_NUM_COLORS)
		{
			goto Exit;
		}

		pLogMessage->changeColor( eForeground, eBackground);
	}

Exit:

	return;
}

/****************************************************************************
Desc:	Default character formatter.
			Prints the character specified by VALUE in 'c', or the '%' character.
			Format: %[flags][width][.prec]'c'
				flags	= <not used>
				width	= <not used>
				prec	= <not used>
****************************************************************************/
FSTATIC void f_logCharFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args)
{
	FLMBYTE				ucOutputBuf[ 32];
	F_SPRINTF_INFO		info;

	f_memset( &info, 0, sizeof( info));
	info.pszDestStr = &ucOutputBuf[ 0];
	f_sprintfCharFormatter( ucFormatChar, uiWidth,
		uiPrecision, uiFlags, &info, args);
	pLogMessage->appendString( (char *)ucOutputBuf);
}

/****************************************************************************
Desc:	Unhandled format strings
****************************************************************************/
FSTATIC void f_logNotHandledFormatter(
	FLMBYTE						ucFormatChar,
	FLMUINT						uiWidth,
	FLMUINT						uiPrecision,
	FLMUINT						uiFlags,
	IF_LogMessageClient *	pLogMessage,
	f_va_list *					args)
{
	FLMBYTE				ucOutputBuf[ 64];
	F_SPRINTF_INFO		info;

	f_memset( &info, 0, sizeof( info));
	info.pszDestStr = &ucOutputBuf[ 0];
	f_sprintfNotHandledFormatter( ucFormatChar, uiWidth,
		uiPrecision, uiFlags, &info, args);
	pLogMessage->appendString( (char *)ucOutputBuf);
}

/****************************************************************************
Desc:	Main entry point for printf functionality.
****************************************************************************/
void f_logPrintf(
	IF_LogMessageClient *	pLogMessage,
	const char *				szFormatStr, ...)
{
	f_va_list				args;

	f_va_start( args, szFormatStr);
	f_logParsePrintfArgs( (FLMBYTE *)szFormatStr, &args, pLogMessage);
	f_va_end( args);
}

/****************************************************************************
Desc:	Printf routine that accepts a va_list argument
****************************************************************************/
void FLMAPI f_logVPrintf(
	IF_LogMessageClient *	pLogMessage,
	const char *				szFormatStr,
	f_va_list *					args)
{
	f_logParsePrintfArgs( (FLMBYTE *)szFormatStr, args, pLogMessage);
}

/****************************************************************************
Desc:	Returns an IF_LogMessageClient object if logging is enabled for the
		specified message type
****************************************************************************/
IF_LogMessageClient * FLMAPI f_beginLogMessage(
	FLMUINT		uiMsgType)
{
	IF_LogMessageClient *		pNewMsg = NULL;

	f_mutexLock( gv_hLoggerMutex);
	
	if( !gv_pLogger)
	{
		goto Exit;
	}
		
	if( (pNewMsg = gv_pLogger->beginMessage( uiMsgType)) != NULL)
	{
		gv_uiPendingLogMessages++;
	}
	
Exit:

	f_mutexUnlock( gv_hLoggerMutex);
	return( pNewMsg);
}

/****************************************************************************
Desc:		Logs information about an error
****************************************************************************/
void FLMAPI f_logError(
	RCODE				rc,
	const char *	pszDoing,
	const char *	pszFileName,
	FLMINT			iLineNumber)
{
	FLMBYTE *					pszMsgBuf = NULL;
	IF_LogMessageClient *	pLogMsg = NULL;

	if( (pLogMsg = f_beginLogMessage( 0)) != NULL)
	{
		if( RC_OK( f_alloc( 512, &pszMsgBuf)))
		{
			if( pszFileName)
			{
				f_sprintf( (char *)pszMsgBuf,
					"Error %s: %e, File=%s, Line=%d.",
					pszDoing, rc, pszFileName, (int)iLineNumber);
			}
			else
			{
				f_sprintf( (char *)pszMsgBuf, "Error %s: %e.", pszDoing, rc);
			}

			pLogMsg->changeColor( FLM_YELLOW, FLM_BLACK);
			pLogMsg->appendString( (char *)pszMsgBuf);
		}
		
		f_endLogMessage( &pLogMsg);
	}

	if( pszMsgBuf)
	{
		f_free( &pszMsgBuf);
	}
}

/****************************************************************************
Desc:	Ends a logging message
****************************************************************************/
void FLMAPI f_endLogMessage(
	IF_LogMessageClient **		ppLogMessage)
{
	if( *ppLogMessage)
	{
		f_mutexLock( gv_hLoggerMutex);
		flmAssert( gv_uiPendingLogMessages);
		
		(*ppLogMessage)->endMessage();
		(*ppLogMessage)->Release();
		*ppLogMessage = NULL;
		
		gv_uiPendingLogMessages--;
		f_mutexUnlock( gv_hLoggerMutex);
	}
}
