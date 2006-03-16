//-------------------------------------------------------------------------
// Desc:	Trace class.
// Tabs:	3
//
//		Copyright (c) 1999-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftrace.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMUINT traceFormatNumber(
	FLMUINT64	ui64Number,
	FLMINT		uiBase,
	char *		pszDest);

/****************************************************************************
Public: Constructors	
****************************************************************************/
FlmTrace::FlmTrace() 
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
Public: 	Destructor
****************************************************************************/
FlmTrace::~FlmTrace()
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
Public:	addRef
Desc:		Add a reference to this object.
****************************************************************************/
FLMINT FlmTrace::AddRef( void)
{
	FLMINT	iRefCnt;

	lock();
	iRefCnt = ++m_i32RefCnt;
	unlock();
	return( iRefCnt);
}

/****************************************************************************
Public:	release
Desc:		Removes a reference to this object.
****************************************************************************/
FLMINT FlmTrace::Release( void)
{
	FLMINT	iRefCnt;

	lock();
	iRefCnt = --m_i32RefCnt;
	unlock();

	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Public:	FlmTrace::trace
Desc:		Outputs the formatted message to the appropriate places.
****************************************************************************/
void FlmTrace::trace(
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
Public:	FlmTrace::setMultiThreaded
Desc:		Enables this as a multi-threaded trace object.
****************************************************************************/
FLMBOOL FlmTrace::setMultiThreaded( void)
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
Public:	FlmTrace::setPipe
Desc:		Sets a FlmTrace object to pipe output to.
****************************************************************************/
void FlmTrace::setPipe(
	FlmTrace *	pTracePipe
	)
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
Public:	FlmTrace::lock
Desc:		Locks the object for multi-threaded use.
****************************************************************************/
void FlmTrace::lock( void)
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
Public:	FlmTrace::unlock
Desc:		Unlocks the object for multi-threaded use.
****************************************************************************/
void FlmTrace::unlock( void)
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
Public:	FlmTrace::outputText
Desc:		Calls outputString and then forwards to pipe for output.
****************************************************************************/
void FlmTrace::outputText(
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
void FlmTrace::processFieldInfo(
	const char **		ppszFormat,
	FLMUINT *			puiWidth,
	FLMUINT *			puiPrecision,
	FLMUINT *			puiFlags,
	f_va_list *			args)
{
	const char *		pszTmp = *ppszFormat;

	/* process flags */

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

	// process width

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

	// process precision

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

	// size modifiers

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
Desc:		Handle text portions of the text string.
****************************************************************************/
void FlmTrace::processStringText(
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
Desc:		Parse arguments in pszFormat, calling approopriate handlers.
****************************************************************************/
void FlmTrace::traceOutputArgs(
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
Desc:		Change colors - may only push or pop a color on to the color stack.
****************************************************************************/
void FlmTrace::formatColor(
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
void FlmTrace::formatString(
	FLMUINT		uiFormatChar,
	FLMUINT		uiWidth,
	FLMUINT		uiPrecision,
	FLMUINT		uiFlags,
	f_va_list *	args)
{
	F_SPRINTF_INFO		info;

	info.pszDestStr = m_pszDestStr;
	flmSprintfStringFormatter( (FLMBYTE)uiFormatChar, uiWidth, 
		uiPrecision, uiFlags, &info, args);
	m_pszDestStr = info.pszDestStr;
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
void FlmTrace::formatNumber(
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
void FlmTrace::formatChar(
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
void FlmTrace::formatNotHandled( void)
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
void FlmTrace::outputCurrentText(
	FLMUINT	uiForeColor,
	FLMUINT	uiBackColor
	)
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
