//-------------------------------------------------------------------------
// Desc:	Cross platform toolkit for threads.
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
// $Id: ftkthrd.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_UNIX)
	#include <signal.h>
#endif

#ifdef FLM_NLM
	static void * threadStub(
		void *	pvUnused,
		void *	pvThread);
#elif defined( FLM_WIN)
	static unsigned __stdcall threadStub(
		void *	pvThread);
#elif defined( FLM_UNIX)
	extern "C"
	{
		static void * threadStub(
			void *	pvThread);
	}
#endif

/****************************************************************************
Desc:		Add a Reference to this object.
****************************************************************************/
FLMINT F_Thread::AddRef(
	FLMBOOL		bMutexLocked)
{
	FLMINT		iRefCnt;

	if( !bMutexLocked)
	{
		f_mutexLock( m_hMutex);
	}

	iRefCnt = F_Base::AddRef();

	if( !bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:		Removes a reference to this object.
****************************************************************************/
FLMINT F_Thread::Release( 
	FLMBOOL		bMutexLocked)
{
	FLMINT		iRefCnt;
	FLMBOOL		bUnlockMutex = FALSE;

	if( !bMutexLocked && m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bUnlockMutex = TRUE;
		bMutexLocked = TRUE;
	}

	flmAssert( getRefCount() > 0);
	iRefCnt = flmAtomicDec( &m_refCnt, m_hMutex, bMutexLocked);

	if( bUnlockMutex)
	{
		f_mutexUnlock( m_hMutex);
	}

	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:    Performs various setup work and starts a new thread
****************************************************************************/
RCODE F_Thread::startThread(
	F_THREAD_FUNC	fnThread, 
	const char *	pszThreadName,
	FLMUINT			uiThreadGroup,
	FLMUINT			uiAppId,
	void *			pvParm1,
	void *			pvParm2,
	FLMUINT        uiStackSize)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bManagerMutexLocked = FALSE;
#ifdef FLM_NLM
	void *			hThread = NULL;
#endif
#ifdef FLM_WIN
	unsigned			uiThreadId;
#endif
#if defined( FLM_UNIX)
	#if defined( _POSIX_THREADS)
		pthread_attr_t	thread_attr;
		pthread_t		uiThreadId;
	#else
		threadid_p		uiThreadId;
	#endif
#endif

	flmAssert( fnThread != NULL && m_fnThread == NULL);

	m_fnThread = fnThread;
	m_pvParm1 = pvParm1;
	m_pvParm2 = pvParm2;

	// Initialize the thread's mutex

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	// Set the stack size

	m_uiStackSize = (uiStackSize < F_THREAD_MIN_STACK_SIZE)
							? F_THREAD_MIN_STACK_SIZE
							: uiStackSize;

	// Set the thread name

	if( pszThreadName && *pszThreadName)
	{
		FLMUINT		uiNameLen = f_strlen( pszThreadName) + 1;

		if( RC_BAD( rc = f_alloc( uiNameLen, &m_pszThreadName)))
		{
			goto Exit;
		}

		f_memcpy( m_pszThreadName, pszThreadName, uiNameLen);
	}

	// Set the thread group ID and the application-specified thread ID

	m_uiThreadGroup = uiThreadGroup;
	m_uiAppId = uiAppId;

	// Set the thread's state to "running" -- if we fail to
	// start the thread, this will be set back to false when
	// the cleanupThread() method is called below.  We set this
	// to TRUE here so that the stopThread() method won't get
	// stuck in an infinite loop if the thread was never started.

	m_bRunning = TRUE;

	// Lock the thread manager's mutex.

	f_mutexLock( gv_FlmSysData.pThreadMgr->m_hMutex);
	bManagerMutexLocked = TRUE;

	// Increment the active thread count

	gv_FlmSysData.pThreadMgr->m_uiNumThreads++;

	// Link the thread into the manager's list.  We can't link threads in order
	// by thread ID at this point, because we don't know what the new thread's
	// ID will be.

	if( gv_FlmSysData.pThreadMgr->m_pThreadList)
	{
		gv_FlmSysData.pThreadMgr->m_pThreadList->m_pPrev = this;
	}

	m_pNext = gv_FlmSysData.pThreadMgr->m_pThreadList;
	gv_FlmSysData.pThreadMgr->m_pThreadList = this;

	// Increment the reference count of the thread object now
	// that it is linked into the thread manager's list.

	AddRef();
	
	// Start the thread

#ifdef FLM_WIN
	if( _beginthreadex( 
		NULL, (unsigned int)m_uiStackSize, threadStub,
		(void *)this, 0, &uiThreadId) == 0) // 0 indicates a failure
	{
		rc = RC_SET( FERR_THREAD_ERR);
		goto Exit;
	}
	m_uiThreadId = (FLMUINT)uiThreadId;
#elif defined( FLM_NLM)
	if( (hThread = kCreateThread( 
		(BYTE *)((m_pszThreadName)
			? (BYTE *)m_pszThreadName 
			: (BYTE *)"NDSDB"),
		threadStub, NULL, (LONG)m_uiStackSize,
		(void *)this)) == NULL)
	{
		rc = RC_SET( FERR_THREAD_ERR);
		goto Exit;
	}
	m_uiThreadId = (FLMUINT)hThread;

	if( kSetThreadLoadHandle( hThread, (LONG)f_getNLMHandle()) != 0)
	{
		(void)kDestroyThread( hThread);
		rc = RC_SET( FERR_THREAD_ERR);
		goto Exit;
	}
			
   if( kScheduleThread( hThread) != 0)
	{
		(void)kDestroyThread( hThread);
		rc = RC_SET( FERR_THREAD_ERR);
		goto Exit;
	}
#elif defined( FLM_UNIX)
	#ifdef  _POSIX_THREADS
		pthread_attr_init( &thread_attr);
		pthread_attr_setdetachstate( &thread_attr, PTHREAD_CREATE_DETACHED);
		
		if (pthread_create( &uiThreadId, &thread_attr,
				threadStub, this) != 0)
		{
			rc = RC_SET( FERR_THREAD_ERR);
			goto Exit;
		}
	#else
		m_uiStackSize = f_max( m_uiStackSize, thr_minstack());
		m_uiStackSize = f_max( m_uiStackSize, thr_min_stack());

		if( thr_create( (void*)NULL, (size_t)uiStackSize,
			threadStub, this, (long)0, &uiThreadId) != 0)
		{
			rc = RC_SET( FERR_THREAD_ERR);
			goto Exit;
		}
	#endif

		m_uiThreadId = (FLMUINT)uiThreadId;

	#ifdef _POSIX_THREADS
		pthread_attr_destroy( &thread_attr);     
	#endif
#endif

	// Code is not designed to handle a thread ID of 0

	flmAssert( m_uiThreadId != 0);

	// Unlock the thread manager's mutex.

	f_mutexUnlock( gv_FlmSysData.pThreadMgr->m_hMutex);
	bManagerMutexLocked = FALSE;

Exit:

	if( RC_BAD( rc))
	{
		// Unlink the thread from the manager's list.  This call
		// won't do anything if the thread was not linked above.

		gv_FlmSysData.pThreadMgr->unlinkThread( this, bManagerMutexLocked);

		// Reset the thread object back to its initial state

		cleanupThread();
	}

	if( bManagerMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.pThreadMgr->m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:    Stop a running thread
****************************************************************************/
void F_Thread::stopThread( void)
{
	// Set the shutdown flag and wait for the thread's
	// status to be something other than "running"

	m_bShutdown = TRUE;
	while( m_bRunning)
	{
		f_sleep( 10);
	}

	// Reset the shutdown flag in case this object is re-used.

	m_bShutdown = FALSE;
}

/****************************************************************************
Desc:    Begins a new thread of execution and calls the passed function.
			Performs generic thread init and cleanup functions.
****************************************************************************/
#ifdef FLM_NLM
void * threadStub(
	void *	pvUnused,
	void *	pvThread)
#elif defined( FLM_WIN)
unsigned __stdcall threadStub(
	void *	pvThread)
#elif defined( FLM_UNIX)
void * threadStub(
	void *	pvThread)
#endif
{
	F_Thread *		pThread = (F_Thread *)pvThread;

#ifdef FLM_NLM
	F_UNREFERENCED_PARM( pvUnused);
#endif

#ifdef FLM_UNIX
	// Block all signals (main thread will handle all signals)

	sigset_t mask;
	sigfillset(&mask);
	pthread_sigmask(SIG_SETMASK, &mask, 0);
#endif

	// Lock the manager's mutex

	gv_FlmSysData.pThreadMgr->lockMutex();

	// At this point, the thread ID must match.

	flmAssert( pThread->m_uiThreadId == f_threadId());

	// Set the start time

	f_timeGetSeconds( &pThread->m_uiStartTime);

	// Unlock the manager's mutex

	gv_FlmSysData.pThreadMgr->unlockMutex();

	// Call the thread's function

	pThread->m_exitRc = pThread->m_fnThread( pThread);

	// Add a temporary reference to the thread object so
	// it doesn't go away when we unlink it from the
	// manager

	pThread->AddRef();

	// Unlink the thread from the thread manager.

	gv_FlmSysData.pThreadMgr->unlinkThread( pThread, FALSE);

	// Set the running flag to FALSE

	pThread->m_bRunning = FALSE;

	// Release the temporary reference to the thread.  Once the
	// reference is release, pThread must not be accessed because
	// the object may have gone away.

	pThread->Release();
	pThread = NULL;

	// Terminate the thread

#if defined( FLM_WIN)
	_endthreadex( 0);
	return( 0);
#elif defined( FLM_NLM)
	kExitThread( NULL);
#endif

#if defined( FLM_NLM) || defined( FLM_UNIX)
	return( NULL);
#endif
}

/****************************************************************************
Desc:    Frees any resources allocated to the thread and resets member
			variables to their initial state
****************************************************************************/
void F_Thread::cleanupThread( void)
{
	flmAssert( !m_pPrev && !m_pNext);
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}

	if( m_pszThreadName)
	{
		f_free( &m_pszThreadName);
	}

	if( m_pszThreadStatus)
	{
		f_free( &m_pszThreadStatus);
	}

	m_uiStatusBufLen = 0;
	m_bShutdown = FALSE;
	m_fnThread = NULL;
	m_bRunning = FALSE;
	m_uiStackSize = 0;
	m_pvParm1 = NULL;
	m_pvParm2 = NULL;
	m_uiThreadId = 0;
	m_uiThreadGroup = FLM_DEFAULT_THREAD_GROUP;
	m_uiAppId = 0;
	m_uiStartTime = 0;
	m_exitRc = FERR_OK;
}

/****************************************************************************
Desc:    Set the thread's status
****************************************************************************/
void F_Thread::_setThreadStatus(
	const char *	pszStatus)
{
	FLMUINT		uiStatusLen = f_strlen( pszStatus) + 1;

	if( m_uiStatusBufLen < uiStatusLen)
	{
		FLMUINT		uiAllocSize = uiStatusLen < 128 ? 128 : uiStatusLen;

		if( m_pszThreadStatus != NULL)
		{
			f_free( &m_pszThreadStatus);
		}
		m_uiStatusBufLen = 0;

		if( RC_BAD( f_alloc( uiAllocSize, &m_pszThreadStatus)))
		{
			m_pszThreadStatus = NULL;
			goto Exit;
		}

		m_uiStatusBufLen = uiAllocSize;
	}

	f_mutexLock( m_hMutex);
	f_memcpy( m_pszThreadStatus, pszStatus, uiStatusLen);
	f_mutexUnlock( m_hMutex);

Exit:

	return;
}

/****************************************************************************
Desc:    Set the thread's status
****************************************************************************/
void F_Thread::setThreadStatus(
	const char *	pszFormat, ...)
{
	char			pucBuffer[ 128];
	f_va_list	args;

	f_va_start( args, pszFormat);
	f_vsprintf( pucBuffer, pszFormat, &args);
	f_va_end( args);

	_setThreadStatus( pucBuffer);
}

/****************************************************************************
Desc:    Set the thread's status to a generic string
****************************************************************************/
void F_Thread::setThreadStatus(
	eFlmThreadStatus	eGenericStatus)
{
	const char *		pszStatus = NULL;

	switch( eGenericStatus)
	{
		case FLM_THREAD_STATUS_INITIALIZING:
			pszStatus = "Initializing";
			break;

		case FLM_THREAD_STATUS_RUNNING:
			pszStatus = "Running";
			break;

		case FLM_THREAD_STATUS_SLEEPING:
			pszStatus = "Sleeping";
			break;

		case FLM_THREAD_STATUS_TERMINATING:
			pszStatus = "Terminating";
			break;

		case FLM_THREAD_STATUS_STARTING_TRANS:
			pszStatus = "Starting transaction";
			break;

		case FLM_THREAD_STATUS_COMMITTING_TRANS:
			pszStatus = "Committing transaction";
			break;

		case FLM_THREAD_STATUS_ABORTING_TRANS:
			pszStatus = "Aborting transaction";
			break;

		case FLM_THREAD_STATUS_UNKNOWN:
		default:
			pszStatus = "Unknown";
			break;
	}

	if( pszStatus)
	{
		setThreadStatus( pszStatus);
	}
}

/****************************************************************************
Desc:		Allocates resources needed by the thread manager
****************************************************************************/
RCODE F_ThreadMgr::setupThreadMgr( void)
{
	RCODE		rc = FERR_OK;	

	flmAssert( m_hMutex == F_MUTEX_NULL);

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Removes a thread from the thread manager's list
Notes:	This routine assumes that the manager's mutex is already locked.
****************************************************************************/
void F_ThreadMgr::unlinkThread(
	F_Thread *		pThread,
	FLMBOOL			bMutexIsLocked)
{
	// Lock the thread manager's mutex

	if( !bMutexIsLocked)
	{
		f_mutexLock( m_hMutex);
	}

	// If the thread isn't linked into the list,
	// don't do anything

	if( !pThread->m_pPrev && !pThread->m_pNext &&
		m_pThreadList != pThread)
	{
		goto Exit;
	}

	// Decrement the active thread count

	flmAssert( m_uiNumThreads);
	m_uiNumThreads--;

	if( pThread->m_pPrev)
	{
		pThread->m_pPrev->m_pNext = pThread->m_pNext;
	}
	else
	{
		m_pThreadList = pThread->m_pNext;
	}

	if( pThread->m_pNext)
	{
		pThread->m_pNext->m_pPrev = pThread->m_pPrev;
	}

	pThread->m_pNext = NULL;
	pThread->m_pPrev = NULL;

	// Release the thread object

	pThread->Release();

Exit:

	if( !bMutexIsLocked)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:		Signals all threads in a thread group to shut down and waits
			for them to terminate.
****************************************************************************/
void F_ThreadMgr::shutdownThreadGroup(
	FLMUINT		uiThreadGroup)
{
	F_Thread *			pThread;
	FLMUINT				uiCount;

	for( ;;)
	{
		f_mutexLock( m_hMutex);

		uiCount = 0;
		pThread = m_pThreadList;
		while( pThread)
		{
			if( pThread->m_uiThreadGroup == uiThreadGroup)
			{
				pThread->setShutdownFlag();
				uiCount++;
			}
			pThread = pThread->m_pNext;
		}

		f_mutexUnlock( m_hMutex);

		if( !uiCount)
		{
			break;
		}

		// The threads will automatically unlink themselves from
		// the manager before they terminate.  Just sleep for
		// a few milliseconds and look through the list again to
		// verify that there are no more threads in the group.

		f_sleep( 200);
	}
}

/****************************************************************************
Desc:		Signals a thread to shut down.
****************************************************************************/
void F_ThreadMgr::setThreadShutdownFlag(
	FLMUINT			uiThreadId)
{
	F_Thread *			pThread;

	flmAssert( uiThreadId != 0);

	f_mutexLock( m_hMutex);
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->m_uiThreadId == uiThreadId)
		{
			pThread->setShutdownFlag();
			break;
		}
		pThread = pThread->m_pNext;
	}

	f_mutexUnlock( m_hMutex);
}

/****************************************************************************
Desc:		Allocates an array of F_THREAD_INFO structures and populates them
			with information about the threads being managed by this object.
****************************************************************************/
RCODE F_ThreadMgr::getThreadInfo(
	POOL *					pPool,
	F_THREAD_INFO **		ppThreadInfo,
	FLMUINT *				puiNumThreads)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiOffset;
	FLMUINT				uiLoop;
	FLMUINT				uiSubLoop;
	FLMUINT				uiLen;
	FLMBOOL				bMutexLocked = FALSE;
	F_THREAD_INFO *	pThreadInfo = NULL;
	F_THREAD_INFO		tmpThreadInfo;
	F_Thread *			pCurThread;
	void *				pvMark = GedPoolMark( pPool);

	*ppThreadInfo = NULL;
	*puiNumThreads = 0;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( m_uiNumThreads == 0)
	{
		goto Exit;
	}

	if( (pThreadInfo = (F_THREAD_INFO *)GedPoolCalloc( pPool, 
		sizeof( F_THREAD_INFO) * m_uiNumThreads)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	uiOffset = 0;
	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		flmAssert( uiOffset < m_uiNumThreads);
		f_mutexLock( pCurThread->m_hMutex);

		pThreadInfo[ uiOffset].uiThreadId = pCurThread->m_uiThreadId;
		pThreadInfo[ uiOffset].uiThreadGroup = pCurThread->m_uiThreadGroup;
		pThreadInfo[ uiOffset].uiAppId = pCurThread->m_uiAppId;
		pThreadInfo[ uiOffset].uiStartTime = pCurThread->m_uiStartTime;

		if( pCurThread->m_pszThreadName)
		{
			uiLen = f_strlen( pCurThread->m_pszThreadName) + 1;

			if( ( pThreadInfo[ uiOffset].pszThreadName = 
				(char *)GedPoolCalloc( pPool, uiLen)) != NULL)
			{
				f_memcpy( pThreadInfo[ uiOffset].pszThreadName,
					pCurThread->m_pszThreadName, uiLen);
			}
		}

		if( pCurThread->m_pszThreadStatus)
		{
			uiLen = f_strlen( pCurThread->m_pszThreadStatus) + 1;

			if( ( pThreadInfo[ uiOffset].pszThreadStatus = 
				(char *)GedPoolCalloc( pPool, uiLen)) != NULL)
			{
				f_memcpy( pThreadInfo[ uiOffset].pszThreadStatus,
					pCurThread->m_pszThreadStatus, uiLen);
			}
		}

		f_mutexUnlock( pCurThread->m_hMutex);
		uiOffset++;
		pCurThread = pCurThread->m_pNext;
	}

	flmAssert( uiOffset == m_uiNumThreads);
	*puiNumThreads = m_uiNumThreads;

	f_mutexUnlock( m_hMutex);
	bMutexLocked = FALSE;

	// Sort the list by thread ID

	for( uiLoop = 0; uiLoop < *puiNumThreads; uiLoop++)
	{
		for( uiSubLoop = uiLoop + 1; uiSubLoop < *puiNumThreads; uiSubLoop++)
		{
			if( pThreadInfo[ uiLoop].uiThreadId > 
				pThreadInfo[ uiSubLoop].uiThreadId)
			{
				f_memcpy( &tmpThreadInfo, 
					&pThreadInfo[ uiLoop], sizeof( F_THREAD_INFO));
				f_memcpy( &pThreadInfo[ uiLoop], 
					&pThreadInfo[ uiSubLoop], sizeof( F_THREAD_INFO));
				f_memcpy( &pThreadInfo[ uiSubLoop], 
					&tmpThreadInfo, sizeof( F_THREAD_INFO));
			}
		}
	}

	*ppThreadInfo = pThreadInfo;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pvMark);
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:		Finds a thread based on user-specified identifiers
****************************************************************************/
RCODE F_ThreadMgr::findThread(
	F_Thread **		ppThread,
	FLMUINT			uiThreadGroup,
	FLMUINT			uiAppId,
	FLMBOOL			bOkToFindMe)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bMutexLocked = FALSE;
	F_Thread *			pCurThread;

	*ppThread = NULL;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( m_uiNumThreads == 0)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		f_mutexLock( pCurThread->m_hMutex);

		if( pCurThread->m_uiThreadGroup == uiThreadGroup &&
			pCurThread->m_uiAppId == uiAppId)
		{
			if( bOkToFindMe || 
				(!bOkToFindMe && pCurThread->m_uiThreadId != f_threadId()))
			{
				// Found a match.

				pCurThread->AddRef( TRUE);
				*ppThread = pCurThread;
				f_mutexUnlock( pCurThread->m_hMutex);
				goto Exit;
			}
		}
	
		f_mutexUnlock( pCurThread->m_hMutex);
		pCurThread = pCurThread->m_pNext;
	}

	rc = RC_SET( FERR_NOT_FOUND);

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:		Finds a thread based on user-specified identifiers
****************************************************************************/
RCODE F_ThreadMgr::getNextGroupThread(
	F_Thread **		ppThread,
	FLMUINT			uiThreadGroup,
	FLMUINT *		puiThreadId)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bMutexLocked = FALSE;
	F_Thread *		pCurThread;
	F_Thread *		pFoundThread = NULL;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( m_uiNumThreads == 0)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	pCurThread = m_pThreadList;
	while( pCurThread)
	{
		if( pCurThread->m_uiThreadGroup == uiThreadGroup &&
			pCurThread->m_uiThreadId > *puiThreadId)
		{
			// The threads are not kept in order by thread ID in the
			// manager's list.  So, we need to make sure we get the
			// thread with the next ID beyond the ID passed into the
			// routine.

			if( !pFoundThread || 
				pCurThread->m_uiThreadId < pFoundThread->m_uiThreadId)
			{
				pFoundThread = pCurThread;
			}
		}
	
		pCurThread = pCurThread->m_pNext;
	}

	if( !pFoundThread)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	pFoundThread->AddRef();
	*ppThread = pFoundThread;
	*puiThreadId = pFoundThread->m_uiThreadId;

Exit:

	if( RC_BAD( rc))
	{
		*ppThread = NULL;
		*puiThreadId = 0xFFFFFFFF;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:		Returns a count of the number of threads in a specified group
****************************************************************************/
FLMUINT F_ThreadMgr::getThreadGroupCount(
	FLMUINT			uiThreadGroup)
{
	F_Thread *		pThread;
	FLMUINT			uiCount;

	f_mutexLock( m_hMutex);

	uiCount = 0;
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->m_uiThreadGroup == uiThreadGroup)
		{
			uiCount++;
		}
		pThread = pThread->m_pNext;
	}

	f_mutexUnlock( m_hMutex);
	return( uiCount);
}

/****************************************************************************
Desc:    Allocate a thread object and start the thread
****************************************************************************/
RCODE f_threadCreate(
	F_Thread **			ppThread,
	F_THREAD_FUNC		fnThread,
	const char *		pszThreadName,
	FLMUINT				uiThreadGroup,
	FLMUINT				uiAppId,
	void *				pvParm1,
	void *				pvParm2,
	FLMUINT				uiStackSize)
{
	RCODE				rc = FERR_OK;
	F_Thread *		pThread = NULL;

	if( ppThread)
	{
		*ppThread = NULL;
	}

	if( (pThread = f_new F_Thread) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pThread->startThread(
		fnThread, pszThreadName, uiThreadGroup, uiAppId, 
		pvParm1, pvParm2, uiStackSize)))
	{
		goto Exit;
	}

	if( ppThread)
	{
		*ppThread = pThread;

		// Set pThread to NULL so that the object won't be released
		// below.  The application has indicated (by passing in a
		// non-NULL ppThread) that it wants to keep a reference to
		// the thread.

		pThread = NULL;
	}

Exit:

	if( pThread)
	{
		pThread->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Deletes a thread object and sets the passed-in pointer to NULL
Notes:	Should not be used on threads that were started with the
			auto-destroy flag set to TRUE
****************************************************************************/
void f_threadDestroy(
	F_Thread **		ppThread)
{
	// Shutdown the thread

	if( *ppThread != NULL)
	{
		(*ppThread)->stopThread();
		(*ppThread)->Release();
		*ppThread = NULL;
	}
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_ThreadMgr::~F_ThreadMgr()
{
	F_Thread *		pTmpThread;

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		pTmpThread = m_pThreadList;
		while( pTmpThread)
		{
			pTmpThread->setShutdownFlag();
			pTmpThread = pTmpThread->m_pNext;
		}

		while( m_pThreadList)
		{
			f_mutexUnlock( m_hMutex);
			f_sleep( 50);
			f_mutexLock( m_hMutex);
		}
		f_mutexUnlock( m_hMutex);
		f_mutexDestroy( &m_hMutex);
	}
}
