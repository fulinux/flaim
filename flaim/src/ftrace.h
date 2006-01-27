//-------------------------------------------------------------------------
// Desc:	Trace class - definitions.
// Tabs:	3
//
//		Copyright (c) 1999-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftrace.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FTRACE_H
#define FTRACE_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

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

// Standard trace categories

// Categories reserved for users

#define USER_CATEGORY1		0x8000000
#define USER_CATEGORY2		0x4000000
#define USER_CATEGORY3		0x2000000
#define USER_CATEGORY4		0x1000000
#define USER_CATEGORY5		0x0800000
#define USER_CATEGORY6		0x0400000
#define USER_CATEGORY7		0x0200000
#define USER_CATEGORY8		0x0100000

/****************************************************************************
Desc: 	Abstract base class which provides the record interface that
			FLAIM uses to access and manipulate all records.
****************************************************************************/

class FlmTrace : public F_Base
{
public:

	FlmTrace();
	
	virtual ~FlmTrace();

	FLMUINT AddRef( void);
	FLMUINT Release( void);

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
		FlmTrace *		pTracePipe);

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
	FlmTrace *		m_pTracePipe;

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

#include "fpackoff.h"

#endif
