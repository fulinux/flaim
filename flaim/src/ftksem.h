//-------------------------------------------------------------------------
// Desc:	Cross platform toolkit for mutexes and semaphores - definitions.
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
// $Id: ftksem.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FTKSEM_H
#define FTKSEM_H

#if defined( FLM_UNIX) 
	#include <pthread.h>
	#include <unistd.h>
	#ifndef  _POSIX_THREADS
		#define  _POSIX_THREADS
	#endif
#endif 

/*****************************************************************************

					 					Mutexes

*****************************************************************************/

/* prototypes for the mutex functions **********************************/

#ifndef FLM_NLM							// Inlined on this platform
	RCODE f_mutexCreate(
		F_MUTEX *	phMutex);

	void f_mutexDestroy(
		F_MUTEX *	phMutex);
#endif

#if defined( FLM_NLM)

	FINLINE RCODE f_mutexCreate(
		F_MUTEX *	phMutex)
	{
		if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"NOVDB")) == F_MUTEX_NULL)
		{
			return( RC_SET( FERR_MEM));
		}

		return( FERR_OK);
	}

	FINLINE void f_mutexDestroy(
		F_MUTEX *	phMutex)
	{
		if (*phMutex != F_MUTEX_NULL)
		{
			(void)kMutexFree( (MUTEX)(*phMutex));
			*phMutex = F_MUTEX_NULL;
		}
	}

	FINLINE void f_mutexLock( 
		F_MUTEX		hMutex)
	{
		(void)kMutexLock( (MUTEX)hMutex);
	}

	FINLINE void f_mutexUnlock(
		F_MUTEX		hMutex)
	{
		(void)kMutexUnlock( (MUTEX)hMutex);
	}

#elif defined( FLM_WIN)

	FINLINE void f_mutexLock(
		F_MUTEX		hMutex)
	{
		(void)EnterCriticalSection( (CRITICAL_SECTION *)hMutex);
	}

	FINLINE void f_mutexUnlock(
		F_MUTEX		hMutex)
	{
		(void)LeaveCriticalSection( (CRITICAL_SECTION *)hMutex);
	}

#elif defined( FLM_UNIX)

	void f_mutexLock(
		F_MUTEX		hMutex);
	
	void f_mutexUnlock(
		F_MUTEX		hMutex);

#endif


/*****************************************************************************

					 					Semaphores

*****************************************************************************/

/* pass this define to semwait if you want to wait forever.  may cause hung */
/* machines or proccesses                                                                                                                                       */
#define F_SEM_WAITFOREVER			(0xFFFFFFFF)

#if defined( FLM_WIN)
	typedef HANDLE					F_SEM;
	typedef HANDLE *				F_SEM_p;
	#define F_SEM_NULL			NULL

#elif defined( FLM_NLM)
	typedef SEMAPHORE				F_SEM;
	typedef SEMAPHORE *			F_SEM_p;
	#define F_SEM_NULL			0

#elif defined( FLM_UNIX)

	/* Added by R. Ganesan because Event Semaphores are not the same as
		Mutex Semaphores. Event Semaphores can be signalled without being
		locked. Event Semaphores need to have a genuine wait till they are
		signalled.

		Note: If semaphore.h is not available; this can be implemented in
		terms of condition variables. Condition variables can also be used
		if it is desired that multiple signals == one signal.
	*/

	#if defined( FLM_AIX) || defined( FLM_OSX)
	
	// OS X only has named semaphores, not unamed ones.  If does, however
	// have condition variables and mutexes, so we'll just use the AIX
	// code (and get timed waits as a bonus...)

		typedef struct
		{
			pthread_mutex_t lock;
			pthread_cond_t  cond;
			int             count;
		} sema_t;

		int sema_init( sema_t * sem);
		void sema_destroy( sema_t * sem);
		void p_operation_cleanup( void * arg);
		int sema_wait( sema_t * sem);
		int sema_timedwait( sema_t * sem, unsigned int uiTimeout);
		int sema_signal( sema_t * sem);

	#else
		#include <semaphore.h>
		
		// Note for future reference: We had problems in the AIX build for
		// eDir 8.8 with open being redefined to open64 in some places
		// because we needed support for large files and this was causing
		// problems with FlmBlobImp::open().  The redefinition happens in
		// fcntl.h, and only fposix.cpp needs to include it.  Unfortunately,
		// semaphore.h also includes fcntl.h, and most flaim files end up 
		// including this ftksem.h.  This means that if we ever enable
		// large file support on other unix's, we might bump into these
		// problems again.
		
	#endif
	
	typedef F_SEM *				F_SEM_p;
	#define F_SEM_NULL			NULL

#else
	#error Unsupported platform
#endif

/* prototypes for the semaphore functions **********************************/

#if defined( FLM_NLM)

	FINLINE RCODE f_semCreate(
		F_SEM *		phSem)
	{
		if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"NOVDB", 0)) == F_SEM_NULL)
		{
			return( RC_SET( FERR_MEM));
		}

		return( FERR_OK);
	}

	FINLINE void f_semDestroy(
		F_SEM *		phSem)
	{
		if (*phSem != F_SEM_NULL)
		{
			(void)kSemaphoreFree( (SEMAPHORE)(*phSem));
			*phSem = F_SEM_NULL;
		}
	}

	FINLINE RCODE f_semWait(
		F_SEM			hSem,
		FLMUINT		uiTimeout)
	{
		RCODE			rc = FERR_OK;

		if( uiTimeout == F_SEM_WAITFOREVER)
		{
			if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
			{
				rc = RC_SET( FERR_MUTEX_UNABLE_TO_LOCK);
			}
		}
		else
		{
			if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
			{
				rc = RC_SET( FERR_MUTEX_UNABLE_TO_LOCK);
			}
		}

		return( rc);
	}

	FINLINE void f_semSignal(
		F_SEM			hSem)
	{
		(void)kSemaphoreSignal( (SEMAPHORE)hSem);
	}

#elif defined( FLM_WIN)

	FINLINE RCODE f_semCreate(
		F_SEM *		phSem)
	{
		if( (*phSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL, 
			0, 10000, NULL )) == NULL)
		{
			return( RC_SET( FERR_MUTEX_OPERATION_FAILED));
		}

		return FERR_OK;
	}												

	FINLINE void f_semDestroy(
		F_SEM *		phSem)
	{
		if (*phSem != F_SEM_NULL)
		{
			CloseHandle( *phSem);
			*phSem = F_SEM_NULL;
		}
	}

	FINLINE RCODE f_semWait(
		F_SEM			hSem,
		FLMUINT		uiTimeout)
	{
		if( WaitForSingleObject( hSem, uiTimeout ) == WAIT_OBJECT_0)
		{
			return( FERR_OK);
		}
		else
		{
			return( RC_SET( FERR_MUTEX_UNABLE_TO_LOCK));
		}
	}

	FINLINE void f_semSignal( 
		F_SEM			hSem)
	{
		(void)ReleaseSemaphore( hSem, 1, NULL);
	}

#elif defined( FLM_UNIX)

	void f_semDestroy(
		F_SEM *	phSem);

	RCODE f_semCreate(
		F_SEM *	phSem);

	RCODE f_semWait(
		F_SEM		hSem,
		FLMUINT	uiTimeout);

	FINLINE void f_semSignal( 
		F_SEM			hSem)
	{
#if defined( FLM_AIX) || defined( FLM_OSX)
		(void)sema_signal( (sema_t *)hSem);
#else
		(void)sem_post( (sem_t *)hSem);
#endif
	}

#else
	#error Platform undefined
#endif

#endif
