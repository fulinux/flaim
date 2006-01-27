//-------------------------------------------------------------------------
// Desc:	Message logging - definitions.
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
// $Id: flog.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FLOG_H
#define FLOG_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Special defines for use in the format string of flmLogPrintf

#define F_BLACK			"%0C"
#define F_BLUE				"%1C"
#define F_GREEN			"%2C"
#define F_CYAN				"%3C"
#define F_RED 				"%4C"
#define F_PURPLE			"%5C"
#define F_BROWN			"%6C"
#define F_LIGHTGRAY		"%7C"
#define F_DARKGRAY		"%8C"
#define F_LIGHTBLUE		"%9C"
#define F_LIGHTGREEN 	"%10C"
#define F_LIGHTCYAN		"%11C"
#define F_LIGHTRED		"%12C"
#define F_LIGHTPURPLE	"%13C"
#define F_YELLOW			"%14C"
#define F_WHITE			"%15C"

#define F_PUSHFORECOLOR	"%+0C"
#define F_PUSHBACKCOLOR	"%+1C"
#define F_POPFORECOLOR	"%-0C"
#define F_POPBACKCOLOR	"%-1C"

#define F_PUSHCOLOR		F_PUSHFORECOLOR F_PUSHBACKCOLOR	
#define F_POPCOLOR		F_POPFORECOLOR F_POPBACKCOLOR

#define F_BLUE_ON_WHITE	"%1.15C"

// Logging functions for use within FLAIM

F_LogMessage * flmBeginLogMessage(
	FlmLogMessageType			eMsgType,
	FlmLogMessageSeverity	eMsgSeverity);

void flmLogPrintf(
	F_LogMessage *	pLogMessage,
	const char *		pszFormatStr, ...);

void flmLogVPrintf( 
	F_LogMessage *		pLogMessage,
	const char *		szFormatStr,
	f_va_list *			args);

/****************************************************************************
Desc:	Ends a logging message
****************************************************************************/
void flmEndLogMessage(
	F_LogMessage **		ppLogMessage);

#include "fpackoff.h"

#endif
