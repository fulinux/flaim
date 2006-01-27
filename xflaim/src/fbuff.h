//------------------------------------------------------------------------------
// Desc:	This file contains the F_IOBuffer and F_IOBufferMgr classes.
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
// $Id: fbuff.h 3123 2006-01-24 17:19:50 -0700 (Tue, 24 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#ifndef FBUFF_H
#define FBUFF_H

class F_IOBuffer;
class F_IOBufferMgr;

typedef void (* WRITE_COMPLETION_CB)(
	F_IOBuffer *		pWriteBuffer);

/*===========================================================================
Class:	F_IOBufferMgr
===========================================================================*/
class F_IOBufferMgr : public XF_RefCount, public XF_Base
{
public:

	// Constructor

	F_IOBufferMgr();

	// Destructor

	virtual ~F_IOBufferMgr();

	RCODE waitForAllPendingIO( void);

	FINLINE void setMaxBuffers(
		FLMUINT			uiMaxBuffers)
	{
		m_uiMaxBuffers = uiMaxBuffers;
	}

	FINLINE void setMaxBytes(
		FLMUINT			uiMaxBytes)
	{
		m_uiMaxBufferBytesToUse = uiMaxBytes;
	}

	FINLINE void enableKeepBuffer( void)
	{
		m_bKeepBuffers = TRUE;
	}

	RCODE getBuffer(
		F_IOBuffer **		ppIOBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT				uiBlockSize);

	FINLINE FLMBOOL havePendingIO( void)
	{
		return( m_pFirstPending ? TRUE : FALSE);
	}

	FINLINE FLMBOOL haveUsed( void)
	{
		return( m_pFirstUsed ? TRUE : FALSE);
	}

private:

	// Private methods and variables

	F_IOBuffer *		m_pFirstPending;
	F_IOBuffer *		m_pFirstAvail;
	F_IOBuffer *		m_pFirstUsed;
	FLMUINT				m_uiMaxBuffers;
	FLMUINT				m_uiMaxBufferBytesToUse;
	FLMUINT				m_uiBufferBytesInUse;
	FLMUINT				m_uiBuffersInUse;
	RCODE					m_completionRc;
	FLMBOOL				m_bKeepBuffers;

	void linkToList(
		F_IOBuffer **	ppListHead,
		F_IOBuffer *	pIOBuffer);

	void unlinkFromList(
		F_IOBuffer *	pIOBuffer);

friend class F_IOBuffer;

};

/*===========================================================================
Class:	F_IOBuffer
===========================================================================*/
class F_IOBuffer : public XF_RefCount, public XF_Base
{
#define MAX_BUFFER_BLOCKS	16
public:

	typedef enum
	{
		MGR_LIST_NONE,
		MGR_LIST_AVAIL,
		MGR_LIST_PENDING,
		MGR_LIST_USED
	} eBufferMgrList;

	// Constructor

	F_IOBuffer();

	// Destructor

	~F_IOBuffer();

	RCODE setupBuffer(
		FLMUINT	uiBufferSize,
		FLMUINT	uiBlockSize);

	FINLINE FLMBYTE * getBuffer( void)
	{
		return( m_pucBuffer);
	}

	FINLINE FLMUINT getBufferSize( void)
	{
		return( m_uiBufferSize);
	}

	FINLINE FLMUINT getBlockSize( void)
	{
		return( m_uiBlockSize);
	}

	void notifyComplete(
		RCODE			rc);

	FINLINE void setCompletionCallback(
		WRITE_COMPLETION_CB 	fnCompletion)
	{
		m_fnCompletion = fnCompletion;
	}

	FINLINE void startTimer(
		XFLM_DB_STATS *	pDbStats)
	{
		if ((m_pDbStats = pDbStats) != NULL)
		{
			m_ui64ElapMilli = 0;
			f_timeGetTimeStamp( &m_StartTime);
		}
	}

	void endTimer( void);

	FINLINE FLMUINT64 getElapTime( void)
	{
		return( m_ui64ElapMilli);
	}

	FINLINE XFLM_DB_STATS * getDbStats( void)
	{
		return( m_pDbStats);
	}

	FINLINE void setCompletionCallbackData(
		FLMUINT	uiBlockNumber,
		void *	pvData)
	{
		flmAssert( uiBlockNumber < MAX_BUFFER_BLOCKS);
		m_UserData [uiBlockNumber] = pvData;
	}

	FINLINE void * getCompletionCallbackData(
		FLMUINT	uiBlockNumber)
	{
		flmAssert( uiBlockNumber < MAX_BUFFER_BLOCKS);
		return( m_UserData [uiBlockNumber]);
	}

	FINLINE RCODE getCompletionCode( void)
	{
		return( m_completionRc);
	}

	FINLINE eBufferMgrList getList( void)
	{
		return( m_eList);
	}

	void makePending( void);

#ifdef FLM_WIN
	FINLINE OVERLAPPED * getOverlapped( void)
	{
		return( &m_Overlapped);
	}

	FINLINE void setFileHandle(
		HANDLE	FileHandle)
	{
		m_FileHandle = FileHandle;
	}
#endif

#if defined( FLM_LINUX) || defined( FLM_SOLARIS) || defined( FLM_OSX)
	FINLINE struct aiocb * getAIOStruct( void)
	{
		return( &m_aio);
	}
#endif

#ifdef FLM_NLM
	void signalComplete(
		RCODE		rc);
#endif

private:

	// Only called by the buffer manager

	RCODE setupIOBuffer(
		F_IOBufferMgr *	pIOBufferMgr);

	FLMBOOL isIOComplete( void);

	RCODE waitToComplete( void);

	// Private methods and variables

	F_IOBufferMgr *		m_pIOBufferMgr;
	FLMBYTE *				m_pucBuffer;
	void *					m_UserData [MAX_BUFFER_BLOCKS];
	FLMUINT					m_uiBufferSize;
	FLMUINT					m_uiBlockSize;
	eBufferMgrList			m_eList;
	FLMBOOL					m_bDeleteOnNotify;
#ifdef FLM_WIN
	HANDLE					m_FileHandle;
	OVERLAPPED				m_Overlapped;
#endif
#if defined( FLM_LINUX) || defined( FLM_SOLARIS) || defined( FLM_OSX)
	struct aiocb			m_aio;
#endif
#ifdef FLM_NLM
	F_SEM						m_hSem;
#endif
	F_IOBuffer *			m_pNext;
	F_IOBuffer *			m_pPrev;
	WRITE_COMPLETION_CB	m_fnCompletion;
	XFLM_DB_STATS *		m_pDbStats;
	RCODE						m_completionRc;
	F_TMSTAMP				m_StartTime;
	FLMUINT64				m_ui64ElapMilli;

	friend class F_IOBufferMgr;
};

#endif 		// #ifndef FBUFF_H
