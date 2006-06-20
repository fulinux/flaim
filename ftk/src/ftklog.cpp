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

FSTATIC FLMUINT traceFormatNumber(
	FLMUINT64					ui64Number,
	FLMINT						uiBase,
	char *						pszDest);

#define FORMAT_MINUS_FLAG		0x0001
#define FORMAT_PLUS_FLAG		0x0002
#define FORMAT_SPACE_FLAG		0x0004
#define FORMAT_POUND_FLAG		0x0008
#define FORMAT_ZERO_FLAG		0x0010
#define FORMAT_SHORT_FLAG		0x0020
#define FORMAT_LONG_FLAG		0x0040
#define FORMAT_DOUBLE_FLAG 	0x0080
#define FORMAT_INT64_FLAG		0x0100

// Special strings to embed in trace() calls.

#define FORE_BLACK			"%0F"
#define FORE_BLUE				"%1F"
#define FORE_GREEN			"%2F"
#define FORE_CYAN				"%3F"
#define FORE_RED 				"%4F"
#define FORE_MAGENTA			"%5F"
#define FORE_BROWN			"%6F"
#define FORE_LIGHTGRAY		"%7F"
#define FORE_DARKGRAY		"%8F"
#define FORE_LIGHTBLUE		"%9F"
#define FORE_LIGHTGREEN 	"%10F"
#define FORE_LIGHTCYAN		"%11F"
#define FORE_LIGHTRED		"%12F"
#define FORE_LIGHTMAGENTA	"%13F"
#define FORE_YELLOW			"%14F"
#define FORE_WHITE			"%15F"

#define BACK_BLACK			"%0B"
#define BACK_BLUE				"%1B"
#define BACK_GREEN			"%2B"
#define BACK_CYAN				"%3B"
#define BACK_RED 				"%4B"
#define BACK_MAGENTA			"%5B"
#define BACK_BROWN			"%6B"
#define BACK_LIGHTGRAY		"%7B"
#define BACK_DARKGRAY		"%8B"
#define BACK_LIGHTBLUE		"%9B"
#define BACK_LIGHTGREEN 	"%10B"
#define BACK_LIGHTCYAN		"%11B"
#define BACK_LIGHTRED		"%12B"
#define BACK_LIGHTMAGENTA	"%13B"
#define BACK_YELLOW			"%14B"
#define BACK_WHITE			"%15B"

#define PUSH_FORE_COLOR		"%+F"
#define POP_FORE_COLOR		"%-F"

#define PUSH_BACK_COLOR		"%+B"
#define POP_BACK_COLOR		"%-B"

// Trace categories reserved for users

#define USER_CATEGORY1		0x8000000
#define USER_CATEGORY2		0x4000000
#define USER_CATEGORY3		0x2000000
#define USER_CATEGORY4		0x1000000
#define USER_CATEGORY5		0x0800000
#define USER_CATEGORY6		0x0400000
#define USER_CATEGORY7		0x0200000
#define USER_CATEGORY8		0x0100000

/****************************************************************************
Desc:
****************************************************************************/
class F_Trace : public F_Object
{
public:

	F_Trace();
	
	virtual ~F_Trace();

	FLMINT FLMAPI AddRef( void);
	
	FLMINT FLMAPI Release( void);

	FINLINE void enableCategory(
		FLMUINT	uiCategory)
	{
		m_uiEnabledCategories |= uiCategory;
	}

	FINLINE void disableCategory(
		FLMUINT	uiCategory)
	{
		m_uiEnabledCategories &= (~(uiCategory));
	}

	FINLINE FLMBOOL categoryEnabled(
		FLMUINT	uiCategory)
	{
		return( ((m_uiEnabledCategories & uiCategory) == uiCategory)
				  ? TRUE
				  : FALSE);
	}

	FINLINE FLMUINT getEnabledCategories( void)
	{
		return( m_uiEnabledCategories);
	}

	void trace(
		FLMUINT			uiCategory,
		const char *	pszFormat,
		...);

	FLMBOOL setMultiThreaded( void);

	void setPipe(
		F_Trace *		pTracePipe);

	void lock( void);

	void unlock( void);

	void outputText(
		FLMUINT			uiCategory,
		FLMUINT			uiForeColor,
		FLMUINT			uiBackColor,
		const char *	pszString);

	virtual void outputString(
		FLMUINT			uiCategory,
		FLMUINT			uiForeColor,
		FLMUINT			uiBackColor,
		const char *	pszString) = 0;

private:

	void processFieldInfo(
		const char **	ppszFormat,
		FLMUINT *		puiWidth,
		FLMUINT *		puiPrecision,
		FLMUINT *		puiFlags,
		f_va_list *		args);

	void processStringText(
		FLMUINT			uiLen,
		...);

	void traceOutputArgs(
		const char *	pszFormat,
		f_va_list *		args);

	void formatColor(
		FLMUINT	uiChar,
		FLMUINT	uiColor,
		FLMUINT	uiFlags);

	void formatString(
		FLMUINT		uiFormatChar,
		FLMUINT		uiWidth,
		FLMUINT		uiPrecision,
		FLMUINT		uiFlags,
		f_va_list *	args);

	void formatNumber(
		FLMUINT		uiFormatChar,
		FLMUINT		uiWidth,
		FLMUINT		uiPrecision,
		FLMUINT		uiFlags,
		f_va_list *	args);

	void formatChar(
		FLMUINT		uiFormatChar,
		f_va_list *	args);

	void formatNotHandled( void);

	void outputCurrentText(
		FLMUINT	uiForeColor,
		FLMUINT	uiBackColor);

	F_MUTEX			m_hMutex;
	FLMUINT			m_uiLockCnt;
#ifdef FLM_DEBUG
	FLMUINT			m_uiLockThreadId;
#endif
	FLMUINT			m_uiEnabledCategories;
	F_Trace *		m_pTracePipe;

	// Variables used to do the printf stuff

#define MAX_FORMAT_STR_SIZE		1000

	char			m_szDestStr [MAX_FORMAT_STR_SIZE];
	char *		m_pszDestStr;
	FLMUINT		m_uiMaxLen;
	FLMUINT		m_uiForeColorDepth;
	FLMUINT		m_uiBackColorDepth;
	FLMUINT		m_uiForeColorStack [8];
	FLMUINT		m_uiBackColorStack [8];
	FLMUINT		m_uiCurrentForeColor;
	FLMUINT		m_uiCurrentBackColor;
	FLMUINT		m_uiCategory;

};

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
	FLMUINT						uiMsgType,
	eLogMessageSeverity		eMsgSeverity)
{
	IF_LogMessageClient *		pNewMsg = NULL;

	f_mutexLock( gv_hLoggerMutex);
	
	if( !gv_pLogger)
	{
		goto Exit;
	}
		
	if( (pNewMsg = gv_pLogger->beginMessage( uiMsgType, eMsgSeverity)) != NULL)
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

	if( (pLogMsg = f_beginLogMessage( 0, F_ERR_MESSAGE)) != NULL)
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
		f_assert( gv_uiPendingLogMessages);
		
		(*ppLogMessage)->endMessage();
		(*ppLogMessage)->Release();
		*ppLogMessage = NULL;
		
		gv_uiPendingLogMessages--;
		f_mutexUnlock( gv_hLoggerMutex);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_Trace::F_Trace() 
{
	m_hMutex = F_MUTEX_NULL;
	m_uiLockCnt = 0;
#ifdef FLM_DEBUG
	m_uiLockThreadId = 0;
#endif
	m_uiEnabledCategories = 0;
	m_pTracePipe = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_Trace::~F_Trace()
{
	flmAssert( m_uiLockCnt == 0);
	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
	if (m_pTracePipe)
	{
		m_pTracePipe->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_Trace::AddRef( void)
{
	return( f_atomicInc( &m_refCnt));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_Trace::Release( void)
{
	FLMINT	iRefCnt;

	if( (iRefCnt = f_atomicDec( &m_refCnt)) == 0)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::trace(
	FLMUINT			uiCategory,
	const char *	pszFormat,
	...
	)
{
	f_va_list	args;

	f_va_start( args, pszFormat);

	// First see if the stuff is enabled.

	if (uiCategory & m_uiEnabledCategories)
	{
		lock();

		// Leave room for terminator.

		m_uiMaxLen = sizeof( m_szDestStr) - 1;
		m_pszDestStr = &m_szDestStr [0];
		m_uiCurrentForeColor = FLM_CURRENT_COLOR;
		m_uiCurrentBackColor = FLM_CURRENT_COLOR;
		m_uiForeColorDepth = 0;
		m_uiBackColorDepth = 0;
		m_uiCategory = uiCategory;
		traceOutputArgs( pszFormat, (f_va_list *)&args);
		outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
		unlock();
	}
	f_va_end( args);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL F_Trace::setMultiThreaded( void)
{
	FLMBOOL	bOk = TRUE;

	if (m_hMutex == F_MUTEX_NULL)
	{
		if (RC_BAD( f_mutexCreate( &m_hMutex)))
		{
			bOk = FALSE;
		}
	}
	return( bOk);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::setPipe(
	F_Trace *	pTracePipe)
{
	lock();
	if (m_pTracePipe && m_pTracePipe != pTracePipe)
	{
		m_pTracePipe->Release();
	}
	if ((m_pTracePipe = pTracePipe) != NULL)
	{
		m_pTracePipe->AddRef();
	}
	unlock();
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::lock( void)
{
	if (m_uiLockCnt)
	{
#ifdef FLM_DEBUG
		flmAssert( m_uiLockThreadId == f_threadId());
#endif
		m_uiLockCnt++;
	}
	else if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		m_uiLockCnt++;
#ifdef FLM_DEBUG
		m_uiLockThreadId = f_threadId();
#endif
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::unlock( void)
{
	if (m_uiLockCnt)
	{
#ifdef FLM_DEBUG
		flmAssert( m_uiLockThreadId == f_threadId());
#endif
		m_uiLockCnt--;
		if (!m_uiLockCnt && m_hMutex != F_MUTEX_NULL)
		{
			f_mutexUnlock( m_hMutex);
#ifdef FLM_DEBUG
			m_uiLockThreadId = 0;
#endif
		}
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::outputText(
	FLMUINT			uiCategory,
	FLMUINT			uiForeColor,
	FLMUINT			uiBackColor,
	const char *	pszString)
{
	lock();
	outputString( uiCategory, uiForeColor, uiBackColor, pszString);
	unlock();
	
	if (m_pTracePipe)
	{
		if (uiCategory & m_pTracePipe->getEnabledCategories())
		{
			m_pTracePipe->outputText( uiCategory, uiForeColor, uiBackColor,
								pszString);
		}
	}
}

/****************************************************************************
Desc:		Parameter 'ppszFormat' points to text following a '%' sign. Process
			legal field information. Leave 'ppszFormat' pointing at the format
			specifier character.
****************************************************************************/
void F_Trace::processFieldInfo(
	const char **		ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args)
{
	const char *		pszTmp = *ppszFormat;

	// Process flags

	*puiFlags = 0;
	for (;;)
	{
		switch (*pszTmp)
		{
			case '-': *puiFlags |= FORMAT_MINUS_FLAG; break;
			case '+': *puiFlags |= FORMAT_PLUS_FLAG; break;
			case ' ': *puiFlags |= FORMAT_SPACE_FLAG; break;
			case '#': *puiFlags |= FORMAT_POUND_FLAG; break;
			case '0': *puiFlags |= FORMAT_ZERO_FLAG; break;
			default:
				goto Out1;
		}
		pszTmp++;
	}
	
Out1:

	// Process width

	*puiWidth = 0;
	if (*pszTmp == '*')
	{
		*puiWidth = f_va_arg( *args, unsigned int);
		++pszTmp;
	}
	else while (*pszTmp >= '0' && *pszTmp <= '9')
	{
		*puiWidth = (*puiWidth * 10) + (*pszTmp - '0');
		++pszTmp;
	}

	// Process precision

	*puiPrecision = 0;
	if (*pszTmp == '.')
	{
		++pszTmp;
		if (*pszTmp == '*')
		{
			*puiPrecision = f_va_arg( *args, unsigned int);
			++pszTmp;
		}
		else while (*pszTmp >= '0' && *pszTmp <= '9')
		{
			*puiPrecision = (*puiPrecision * 10) + (*pszTmp - '0');
			++pszTmp;
		}
	}

	// Size modifiers

	switch( *pszTmp)
	{
		case 'L': *puiFlags |= FORMAT_DOUBLE_FLAG; ++pszTmp; break;
		case 'l': *puiFlags |= FORMAT_LONG_FLAG; ++pszTmp; break;
		case 'h': *puiFlags |= FORMAT_SHORT_FLAG; ++pszTmp; break;
		case 'I':
		{
			if( pszTmp[1] == '6' && pszTmp[2] == '4')
			{
				*puiFlags |= FORMAT_INT64_FLAG;
				pszTmp += 3;
			}
			break;
		}
	}
	
	*ppszFormat = pszTmp;
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::processStringText(
	FLMUINT	uiLen,
	...)
{
	f_va_list	args;

	f_va_start(args, uiLen);
	if (uiLen)
	{
		formatString( 0, uiLen, uiLen, 0, (f_va_list *)&args);
	}
	f_va_end(args);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::traceOutputArgs(
	const char *	pszFormat,
	f_va_list *		args)
{
	FLMUINT			uiChar;
	FLMUINT			uiWidth;
	FLMUINT			uiPrecision;
	FLMUINT			uiFlags;
	const char *	pszTextStart = pszFormat;

	while ((uiChar = (FLMUINT)*pszFormat++) != 0)
	{
		if (uiChar != '%')
		{
			continue;
		}
		uiWidth = pszFormat - pszTextStart - 1;
		processStringText( uiWidth, pszTextStart);
		processFieldInfo( &pszFormat, &uiWidth, &uiPrecision, &uiFlags, args);
		uiChar = (FLMUINT)*pszFormat++;

		switch (uiChar)
		{
			case 'c':
			case '%':
				formatChar( uiChar, args);
				break;
			case 'B':
			case 'F':
				formatColor( uiChar, uiWidth, uiFlags);
				break;
			case 'd':
			case 'o':
			case 'u':
			case 'x':
			case 'X':
				formatNumber( uiChar, uiWidth, uiPrecision, uiFlags, args);
				break;
			case 's':
			case 'S':
			case 'U':
				formatString( uiChar, uiWidth, uiPrecision, uiFlags, args);
				break;

			default:
				formatNotHandled();
				break;
		}
		pszTextStart = pszFormat;
	}
	processStringText( (FLMUINT)(pszFormat - pszTextStart) - 1, pszTextStart);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_Trace::formatColor(
	FLMUINT	uiChar,
	FLMUINT	uiColor,
	FLMUINT	uiFlags)
{
	FLMUINT	uiPrevForeColor = m_uiCurrentForeColor;
	FLMUINT	uiPrevBackColor = m_uiCurrentBackColor;

	if (uiChar == 'F')	// Foreground color
	{
		if ((uiFlags & FORMAT_PLUS_FLAG) && m_uiForeColorDepth < 8)
		{
			m_uiForeColorStack [m_uiForeColorDepth++] = m_uiCurrentForeColor;
		}
		else if ((uiFlags & FORMAT_MINUS_FLAG) && m_uiForeColorDepth > 0)
		{
			m_uiCurrentForeColor = m_uiForeColorStack [--m_uiForeColorDepth];
		}
		else
		{
			m_uiCurrentForeColor = uiColor;
		}
	}
	else	// uiChar == 'B' - background color
	{
		if ((uiFlags & FORMAT_PLUS_FLAG) && m_uiBackColorDepth < 8)
		{
			m_uiBackColorStack [m_uiBackColorDepth++] = m_uiCurrentBackColor;
		}
		else if ((uiFlags & FORMAT_MINUS_FLAG) && m_uiBackColorDepth > 0)
		{
			m_uiCurrentBackColor = m_uiBackColorStack [--m_uiBackColorDepth];
		}
		else
		{
			m_uiCurrentBackColor = uiColor;
		}
	}

	// If the color changed, output the current text with the last
	// colors we had.

	if ((m_uiCurrentForeColor != uiPrevForeColor) ||
		 (m_uiCurrentBackColor != uiPrevBackColor))
	{
		outputCurrentText( uiPrevForeColor, uiPrevBackColor);
	}
}

/****************************************************************************
Desc:		String formatter.
			Outputs the asciiz string specified by ADDRESS in 's'.
			Outputs length preceeded string specified by ADDRESS in 'S'.
			Outputs unicode string specified by ADDRESS in 'U'.
			Format: %[flags][width][.prec]'s'|'S'|'U'
				flags	= '-' left justifies if string length < width
				width	= minimum number of characters to print
				prec	= maximum number of characters to print
****************************************************************************/
void F_Trace::formatString(
	FLMUINT		uiFormatChar,
	FLMUINT		uiWidth,
	FLMUINT		uiPrecision,
	FLMUINT		uiFlags,
	f_va_list *	args)
{
	F_SPRINTF_INFO		info;

	info.pszDestStr = (FLMBYTE *)m_pszDestStr;
	f_sprintfStringFormatter( (FLMBYTE)uiFormatChar, uiWidth, 
		uiPrecision, uiFlags, &info, args);
	m_pszDestStr = (char *)info.pszDestStr;
}

/****************************************************************************
Desc:		This is used by printf and sprintf to output numbers.  It uses
			recursion to separate the numbers down to individual digits and
			output them in the proper order.
****************************************************************************/
FSTATIC FLMUINT traceFormatNumber(
	FLMUINT64	ui64Number,
	FLMINT		uiBase,
	char *		pszDest)
{
	char			c = (char)(ui64Number % uiBase);
	FLMUINT64	ui64Index = ui64Number / uiBase;

	ui64Index = ui64Index
				 ? traceFormatNumber( ui64Index, uiBase, pszDest)
				 : 0;
	pszDest [ui64Index] = (char)((c > 9)
										? c + 'a' - 10
										: c + '0');
	return (FLMUINT)(ui64Index + 1);
}

// Percent formating prefixes

#define P_NONE				0
#define P_MINUS 			1
#define P_PLUS				2
#define P_POUND 			3

/****************************************************************************
Desc:		Number formatter.
			Formats the number specified by VALUE in 'd', 'o', 'u', 'x', or 'X'
			Format: %[flags][width][.prec]'E'
				flags	= 'h' value is uint16
				  'l' value is uint32
				  '-' left align result
				  '+' print plus sign if positive
				  '#' print '0x' in front of hex numbers
				  '0' zero-fill
				width	= minimum number of characters to output
				prec	= maximum number of characters to output (truncates or rounds)
****************************************************************************/
void F_Trace::formatNumber(
	FLMUINT		uiFormatChar,
	FLMUINT		uiWidth,
	FLMUINT		uiPrecision,
	FLMUINT		uiFlags,
	f_va_list *	args)
{
	FLMUINT		uiCount;
	FLMUINT		uiPrefix = P_NONE;
	FLMUINT		uiLength;
	FLMUINT		uiBase = 10;
	char			szNumberBuffer [64];
	char *		pszStr;
	FLMUINT64	ui64Arg;

	if (uiFlags & FORMAT_SHORT_FLAG)
	{
		ui64Arg = f_va_arg( *args, int);
	}
	else if (uiFlags & FORMAT_LONG_FLAG)
	{
		ui64Arg = f_va_arg( *args, long int);
	}
	else if ( uiFlags & FORMAT_INT64_FLAG)
	{
		ui64Arg = f_va_arg( *args, FLMUINT64);
	}
	else
	{
		ui64Arg = f_va_arg( *args, int);
	}

	switch (uiFormatChar)
	{
		case 'd':
			if ((long)ui64Arg < 0)
			{
				// handle negatives

				uiPrefix = P_MINUS;
				if (uiWidth > 0)
				{
					uiWidth--;
				}
				ui64Arg = -(long)ui64Arg;
			}
			else if (uiFlags & FORMAT_PLUS_FLAG)
			{
				uiPrefix = P_PLUS;
				if (uiWidth > 0)
					uiWidth--;
			}
			break;

		case 'o':
			uiBase = 8;
			break;

		case 'x':
		case 'X':
			if (uiFlags & FORMAT_POUND_FLAG && ui64Arg)
			{
				uiPrefix = P_POUND;
				if (uiWidth > 1)
				{
					uiWidth -= 2;
				}
			}
			uiBase = 16;
			break;
	}
	uiLength = traceFormatNumber( ui64Arg, uiBase, szNumberBuffer);
	szNumberBuffer [uiLength] = 0;
	if (uiFormatChar == 'X')
	{
		char *	p = &szNumberBuffer [0];
		while (*p)
		{
			if ((*p >= 'a') && (*p <= 'z'))
			{
				*p = (char)(*p - 'a' + 'A');
			}
			p++;
		}
	}
	if (uiWidth < uiLength)
	{
		uiWidth = uiLength;
	}

	if (uiFlags & FORMAT_ZERO_FLAG)
	{
		uiPrecision = uiWidth; // zero fill
	}
	else if (!(uiFlags & FORMAT_MINUS_FLAG))
	{
		// Right justify.

		while ((uiWidth > uiLength) &&
				 (uiWidth > uiPrecision))
		{
			if (!m_uiMaxLen)
			{
				outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
			}
			*m_pszDestStr++ = ' ';
			m_uiMaxLen--;
			--uiWidth;
		}
	}

	// handle the prefix if any

	if (!m_uiMaxLen)
	{
		outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
	}
	switch (uiPrefix)
	{
		case P_MINUS:
			*m_pszDestStr++ = '-';
			m_uiMaxLen--;
			break;
		case P_PLUS:
			*m_pszDestStr++ = '+';
			m_uiMaxLen--;
			break;
		case P_POUND:
			*m_pszDestStr++ = '0';
			m_uiMaxLen--;
			if (!m_uiMaxLen)
			{
				outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
			}
			*m_pszDestStr++ = (char)uiFormatChar;
			m_uiMaxLen--;
			break;
	}

	// handle the precision

	while (uiLength < uiPrecision)
	{
		if (!m_uiMaxLen)
		{
			outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
		}
		*m_pszDestStr++ = '0';
		m_uiMaxLen--;
		--uiPrecision;
		--uiWidth;
	}

	// print out the number

	for (uiCount = uiLength, pszStr = szNumberBuffer;
		  uiCount > 0;
		  uiCount--, m_uiMaxLen--)
	{
		if (!m_uiMaxLen)
		{
			outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
		}
		*m_pszDestStr++ = *pszStr++;
	}

	if (uiFlags & FORMAT_MINUS_FLAG)
	{
		while (uiLength < uiWidth) // left justify
		{
			if (!m_uiMaxLen)
			{
				outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
			}
			*m_pszDestStr++ = ' ';
			m_uiMaxLen--;
			--uiWidth;
		}
	}
}

/****************************************************************************
Desc:		Character formatter.
			FOrmats the character specified by VALUE in 'c', or the '%' character.
			Format: %[flags][width][.prec]'c'
				flags	= <not used>
				width	= <not used>
				prec	= <not used>
****************************************************************************/
void F_Trace::formatChar(
	FLMUINT		uiFormatChar,
	f_va_list *	args)
{
	char c = (uiFormatChar == '%')
				? '%'
				: f_va_arg( *args, int);

	if (!m_uiMaxLen)
	{
		outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
	}
	*m_pszDestStr++ = c;
	m_uiMaxLen--;
}


/****************************************************************************
Desc:		Handles all formatting stuff that is invalid.
****************************************************************************/
void F_Trace::formatNotHandled( void)
{
	const char * pszNoFormatter = "<no formatter>";
	FLMUINT	uiLen = f_strlen( pszNoFormatter);
	FLMUINT	uiTmpLen;

	while (uiLen)
	{
		if (!m_uiMaxLen)
		{
			outputCurrentText( m_uiCurrentForeColor, m_uiCurrentBackColor);
		}
		if ((uiTmpLen = uiLen) > m_uiMaxLen)
		{
			uiTmpLen = m_uiMaxLen;
		}
		f_memcpy( m_pszDestStr, (void *)pszNoFormatter, uiTmpLen);
		m_pszDestStr += uiTmpLen;
		m_uiMaxLen -= uiTmpLen;
		uiLen -= uiTmpLen;
		pszNoFormatter += uiTmpLen;
	}
}

/****************************************************************************
Desc:		Outputs the current text that we have buffered.
****************************************************************************/
void F_Trace::outputCurrentText(
	FLMUINT	uiForeColor,
	FLMUINT	uiBackColor)
{
	FLMUINT	uiLen = (FLMUINT)(m_pszDestStr - &m_szDestStr [0]);

	if (uiLen)
	{
		m_szDestStr [uiLen] = 0;
		outputText( m_uiCategory, uiForeColor, uiBackColor, m_szDestStr);
		m_uiMaxLen = sizeof( m_szDestStr) - 1;
		m_pszDestStr = &m_szDestStr [0];
	}
}

/****************************************************************************
Desc:	Initialize the toolkit logger
****************************************************************************/
RCODE f_loggerInit( void)
{
	RCODE	rc = NE_FLM_OK;

	if (RC_BAD( rc = f_mutexCreate( &gv_hLoggerMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Shutdown the toolkit logger
****************************************************************************/
void f_loggerShutdown( void)
{
	if (gv_pLogger)
	{
		gv_pLogger->Release();
		gv_pLogger = NULL;
	}
	if (gv_hLoggerMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_hLoggerMutex);
	}
}

void f_setLoggerClient(
	IF_LoggerClient *	pLogger)
{
	f_mutexLock( gv_hLoggerMutex);
	if (gv_pLogger)
	{
		gv_pLogger->Release();
	}
	if ((gv_pLogger = pLogger) != NULL)
	{
		gv_pLogger->AddRef();
	}
	f_mutexUnlock( gv_hLoggerMutex);
}
