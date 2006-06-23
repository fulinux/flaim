//------------------------------------------------------------------------------
// Desc:	This file contains the logging routines.  They use the
//			IF_Logger_Client and IF_LogMessage_Client classes.
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
// $Id$
//------------------------------------------------------------------------------

#ifndef FLOG_H
#define FLOG_H

// Logging functions for use within FLAIM

IF_LogMessageClient * flmBeginLogMessage(
	eLogMessageType	eMsgType);

void flmEndLogMessage(
	IF_LogMessageClient **	ppLogMessage);

/*============================================================================
								Debug Logging Functions
============================================================================*/

#ifdef FLM_DBG_LOG

	void scaLogWrite(
		F_Database *	pDatabase,
		FLMUINT			uiWriteAddress,
		FLMBYTE *		pucBlkBuf,
		FLMUINT			uiBufferLen,
		FLMUINT			uiBlockSize,
		char *			pszEvent);

	void flmDbgLogWrite(
		F_Database *	pDatabase,
		FLMUINT			uiBlkAddress,
		FLMUINT			uiWriteAddress,
		FLMUINT64		ui64TransId,
		char *			pszEvent);

	void flmDbgLogUpdate(
		F_Database *	pDatabase,
		FLMUINT64		ui64TransId,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		RCODE				rc,
		char *			pszEvent);

	void flmDbgLogMsg(
		char *		pszMsg);

	void flmDbgLogInit( void);
	void flmDbgLogExit( void);
	void flmDbgLogFlush( void);

#endif	// #ifdef FLM_DBG_LOG

#endif 		// #ifndef FLOG_H
