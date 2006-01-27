//-------------------------------------------------------------------------
// Desc:	Object reference tracker - definitions.
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
// $Id: fobjtrck.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FOBJTRCK_H
#define FOBJTRCK_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/*
Forward references
*/

class F_ObjRefTracker;

/*
Callbacks
*/

typedef RCODE (* ADDR_FMT_HOOK)(	// Address formatter / translator
	F_ObjRefTracker *		pRefTracker,		// Reference tracker object
	void *					pAddress,			// Pointer to the address
	FLMBYTE *				pucBuffer,			// Buffer for formatted address
	FLMUINT					uiSize,				// Size of buffer
	void *					pvUserData);		// User-supplied callback data

/****************************************************************************
	Object reference tracker
****************************************************************************/

class F_ObjRefTracker
{
public:

	F_ObjRefTracker( void);
	
	virtual ~F_ObjRefTracker( void);

	RCODE setup(
		const char *		pszObjName,
		FLMBOOL				bLogToFile = FALSE);

	void trackRef(
		void *				pReferenceID,
		void *				pSubrefID = NULL);
		
	void untrackRef(
		void *				referenceID,
		void *				subrefID = NULL);
		
	void checkForUnreleasedRefs(
		FLMUINT *			puiCount = NULL);

	void setAddressFormatter(
		ADDR_FMT_HOOK 		pFunc,
		void *				pvUserData);

	FINLINE void setModuleHandle( 
		void *				pModHandle) 
	{ 
		m_pModHandle = pModHandle;
	}
	
private:

	F_MUTEX						m_hRefListMutex;
	F_ListMgr *					m_pListManager;
	F_ListNode					m_lnode;

	FLMUINT						m_lOptions;
	FLMUINT					   m_lCallStackDepth;

#define FORTRACK_MAX_OBJ_NAME_LEN		63
	char	 						m_pszObjName[ FORTRACK_MAX_OBJ_NAME_LEN + 1];
	F_FileSystemImp *			m_pFileSystem;
	FLMBOOL						m_bLocalFS;		// Was the file system allocated for this object?
	ADDR_FMT_HOOK				m_pAddrFmtHook;
	void *						m_pUserData;
	void *						m_pModHandle;
	char							m_pLogPath[ F_PATH_MAX_SIZE];

	void formatAddress(
		char *				pucBuffer,
		FLMUINT				uiSize,
		void *				pAddress);

	static void getCallStack(
		void *				stack[],
		FLMUINT				uiCount,
		FLMUINT				uiSkip);
		
	void logError(
		const char *		pucMessage);

	RCODE logMessage(
		const char *		pucMessage,
		F_FileHdl *			pFileHdl,
		FLMUINT &			uiFileCursor);
};

#	define CTRC_STACK_SIZE 20

class TrackingRecord : public F_ListItem
{
public:

	TrackingRecord( void * pReferenceID, void * pSubrefID)
	{
		m_pReferenceID = pReferenceID;
		m_pSubrefID = pSubrefID;
		m_uiThreadID = f_threadId();
		f_memset( m_stack, 0, sizeof(m_stack));
	}
	
	virtual ~TrackingRecord()
	{
	}

	void * getReferenceID()
	{
		return m_pReferenceID;
	}
	void * getSubrefID()
	{
		return m_pSubrefID;
	}
	FLMUINT getThreadID()
	{
		return m_uiThreadID;
	}
	void * getStack()
	{
		return m_stack;
	}
	
private:

	void *					m_pReferenceID;
	void *					m_pSubrefID;
	FLMUINT					m_uiThreadID;
	void *					m_stack[CTRC_STACK_SIZE+1];
};

#include "fpackoff.h"

#endif //FORTRACK_H
