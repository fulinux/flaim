//-------------------------------------------------------------------------
// Desc:	Cross platform toolkit for threads - definitions
// Tabs:	3
//
//		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftkthrd.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FTKTHRD_H
#define FTKTHRD_H

#if defined( FLM_UNIX) 
	#include <pthread.h>
	#include <unistd.h>			// defines _POSIX_THREADS
	#ifndef  _POSIX_THREADS
		#define  _POSIX_THREADS
	#endif
#endif 

#if defined( FLM_NLM) || defined( FLM_WIN) || defined( FLM_UNIX)
	#define F_MULTI_THREADED
#endif

#ifdef FLM_WIN
	#define f_threadId()	(FLMUINT)_threadid
#endif

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#ifdef FLM_NLM
	extern "C"
	{
		// WARNING! The following come from mpktypes.h

		void *		kCurrentThread(void);
	}
	#define f_threadId()	((FLMUINT)kCurrentThread())
#endif

#ifdef FLM_UNIX
	FINLINE FLMUINT f_threadId(void)
	{
	#ifdef  _POSIX_THREADS
		pthread_t thrd = pthread_self();
	#endif

	#if defined( SCO)
		return( (FLMUINT) thrd.field2);
	#else
		return( (FLMUINT) thrd);
	#endif
	}
#endif

#define F_THREAD_MIN_STACK_SIZE			(16 * 1024)
#define F_THREAD_DEFAULT_STACK_SIZE		(16 * 1024)

// Forward declarations

class F_Thread;
class F_ThreadMgr;

// Thread types

#define FLM_DEFAULT_THREAD_GROUP						1
#define FLM_CHECKPOINT_THREAD_GROUP					2
#define FLM_BACKGROUND_INDEXING_THREAD_GROUP		3
#define FLM_DB_THREAD_GROUP							4

/****************************************************************************
Desc: 	Class for managing a set of threads
****************************************************************************/
class F_ThreadMgr : public F_Base
{
public:

	// Constructors

	F_ThreadMgr()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pThreadList = NULL;
		m_uiNumThreads = 0;
	}

	// Destructor

	virtual ~F_ThreadMgr();

	// Setup

	RCODE setupThreadMgr( void);

	// Shutdown

	void shutdownThreadGroup(
		FLMUINT			uiThreadGroup);

	void setThreadShutdownFlag(
		FLMUINT			uiThreadId);

	// Search

	RCODE findThread(
		F_Thread **		ppThread,
		FLMUINT			uiThreadGroup,
		FLMUINT			uiAppId = 0,
		FLMBOOL			bOkToFindMe = TRUE);

	RCODE getNextGroupThread(
		F_Thread **		ppThread,
		FLMUINT			uiThreadGroup,
		FLMUINT *		puiThreadId);

	// Statistics

	RCODE getThreadInfo(
		POOL *				pPool,
		F_THREAD_INFO **	ppThreadInfo,
		FLMUINT *			puiNumThreads);

	FLMUINT getThreadGroupCount(
		FLMUINT			uiThreadGroup);
		
	FINLINE void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
	
	FINLINE void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}

	void unlinkThread(
		F_Thread *		pThread,
		FLMBOOL			bMutexLocked);

private:

	F_MUTEX			m_hMutex;
	F_Thread *		m_pThreadList;
	FLMUINT			m_uiNumThreads;

friend class F_Thread;
};

typedef enum
{
	FLM_THREAD_STATUS_UNKNOWN = 0,
	FLM_THREAD_STATUS_INITIALIZING,
	FLM_THREAD_STATUS_RUNNING,
	FLM_THREAD_STATUS_SLEEPING,
	FLM_THREAD_STATUS_TERMINATING,
	FLM_THREAD_STATUS_STARTING_TRANS,
	FLM_THREAD_STATUS_COMMITTING_TRANS,
	FLM_THREAD_STATUS_ABORTING_TRANS
} eFlmThreadStatus;

typedef RCODE (* F_THREAD_FUNC)(F_Thread *);

/****************************************************************************
Desc: 	Class for creating and managing a thread
****************************************************************************/
class F_Thread : public F_Base
{
public:

	F_Thread()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pszThreadName = NULL;
		m_pszThreadStatus = NULL;
		m_uiStatusBufLen = 0;
		m_pPrev = NULL;
		m_pNext = NULL;
		cleanupThread();
	}

	virtual ~F_Thread()
	{
		stopThread();
		cleanupThread();
	}

	FLMINT AddRef( 
		FLMBOOL bMutexLocked);

	FINLINE FLMINT AddRef( void)
	{
		return( AddRef( FALSE));
	}

	FLMINT Release( 
		FLMBOOL bMutexLocked);

	FINLINE FLMINT Release( void)
	{
		return( Release( FALSE));
	}

	RCODE startThread(
		F_THREAD_FUNC	fnThread,
		const char *	pszThreadName = NULL,
		FLMUINT			uiThreadGroup = FLM_DEFAULT_THREAD_GROUP,
		FLMUINT			uiAppId = 0,
		void *			pvParm1 = NULL,
		void *			pvParm2 = NULL,
		FLMUINT        uiStackSize = F_THREAD_DEFAULT_STACK_SIZE);

	void stopThread( void);

	FINLINE FLMUINT getThreadId( void)
	{
		return( m_uiThreadId);
	}

	FINLINE FLMBOOL getShutdownFlag( void)
	{
		return( m_bShutdown);
	}

	FINLINE RCODE getExitCode( void)
	{
		return( m_exitRc);
	}

	FINLINE void * getParm1( void)
	{
		return( m_pvParm1);
	}

	FINLINE void setParm1(
		void *		pvParm)
	{
		m_pvParm1 = pvParm;
	}

	FINLINE void * getParm2( void)
	{
		return( m_pvParm2);
	}

	FINLINE void setParm2(
		void *		pvParm)
	{
		m_pvParm2 = pvParm;
	}

	FINLINE void setShutdownFlag( void)
	{
		m_bShutdown = TRUE;
	}

	FINLINE FLMBOOL isThreadRunning( void)
	{
		return( m_bRunning);
	}

	void setThreadStatus(
		const char *		pszBuffer, ...);

	void setThreadStatus(
		eFlmThreadStatus	eGenericStatus);

	FINLINE void setThreadAppId(
		FLMUINT		uiAppId)
	{
		f_mutexLock( m_hMutex);
		m_uiAppId = uiAppId;
		f_mutexUnlock( m_hMutex);
	}

	FINLINE FLMUINT getThreadAppId( void)
	{
		return( m_uiAppId);
	}

	FINLINE FLMUINT getThreadGroup( void)
	{
		return( m_uiThreadGroup);
	}

	void cleanupThread( void);

	void _setThreadStatus(
		const char *		pszStatus);

	F_MUTEX				m_hMutex;
	F_Thread *			m_pPrev;
	F_Thread *			m_pNext;
	char *				m_pszThreadName;
	char *				m_pszThreadStatus;
	FLMUINT				m_uiStatusBufLen;
	FLMBOOL				m_bShutdown;
	F_THREAD_FUNC		m_fnThread;
	FLMBOOL				m_bRunning;
	FLMUINT				m_uiStackSize;
	void *				m_pvParm1;
	void *				m_pvParm2;
	FLMUINT				m_uiThreadId;
	FLMUINT				m_uiThreadGroup;
	FLMUINT				m_uiAppId;
	FLMUINT				m_uiStartTime;
	RCODE					m_exitRc;

friend class F_ThreadMgr;
};

RCODE f_threadCreate(
	F_Thread **			ppThread,
	F_THREAD_FUNC		fnThread,
	const char *		pszThreadName = NULL,
	FLMUINT				uiThreadGroup = 0,
	FLMUINT				uiAppId = 0,
	void *				pvParm1 = NULL,
	void *				pvParm2 = NULL,
	FLMUINT				uiStackSize = (16 * 1024));
							
void f_threadDestroy(
	F_Thread **			ppThread);

#include "fpackoff.h"

#endif
