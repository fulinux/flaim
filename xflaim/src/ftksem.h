//------------------------------------------------------------------------------
// Desc:	This file defines the needed prototypes, defines, macros needed
//			for FLAIM's Semaphore and Mutex functions.
//
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
// $Id: ftksem.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FTKSEM_H
#define FTKSEM_H

#ifndef FLM_NLM
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
			return RC_SET( NE_XFLM_MEM);
		}

		return NE_XFLM_OK;
	}

	FINLINE void f_mutexDestroy(
		F_MUTEX *	phMutex)
	{
		if (*phMutex != F_MUTEX_NULL)
		{
			if( kMutexFree( (MUTEX)(*phMutex)))
			{
				flmAssert( 0);
			}
			
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

	FINLINE void f_assertMutexLocked(
		F_MUTEX)
	{
	}

#elif defined( FLM_WIN)

	#ifndef __STDDEF_H
		#include <stddef.h>
	#endif

	FINLINE void f_mutexLock(
		F_MUTEX		hMutex)
	{
		while( flmAtomicExchange( 
			&(((F_INTERLOCK *)hMutex)->locked), 1) != 0)
		{
#ifdef FLM_DEBUG
			flmAtomicInc( &(((F_INTERLOCK *)hMutex)->waitCount));
#endif
			Sleep( 0);
		}

#ifdef FLM_DEBUG
		flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == 0);
		((F_INTERLOCK *)hMutex)->uiThreadId = _threadid;
		flmAtomicInc( &(((F_INTERLOCK *)hMutex)->lockedCount));
#endif
	}
	
	FINLINE void f_mutexUnlock(
		F_MUTEX		hMutex)
	{
		flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
#ifdef FLM_DEBUG
		flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
		((F_INTERLOCK *)hMutex)->uiThreadId = 0;
#endif
		flmAtomicExchange( &(((F_INTERLOCK *)hMutex)->locked), 0);
	}

	FINLINE void f_assertMutexLocked(
		F_MUTEX		hMutex)
	{
#ifdef FLM_DEBUG
		flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
		flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
#else
		F_UNREFERENCED_PARM( hMutex);
#endif
	}

#elif defined( FLM_UNIX)

	FINLINE void f_mutexLock(
		F_MUTEX		hMutex)
	{
		(void)pthread_mutex_lock( hMutex);
	}

	FINLINE void f_mutexUnlock(
		F_MUTEX		hMutex)
	{
		(void)pthread_mutex_unlock( hMutex);
	}

	FINLINE void f_assertMutexLocked(
		F_MUTEX)
	{
	}

#endif

/*****************************************************************************
					 					Semaphores
*****************************************************************************/

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

	typedef struct
	{
		pthread_mutex_t lock;
		pthread_cond_t  cond;
		int             count;
	} sema_t;

	typedef sema_t *		F_SEM;
	typedef F_SEM *		F_SEM_p;
	#define F_SEM_NULL	NULL

	int sema_signal(
		sema_t *			sem);

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
			return RC_SET( NE_XFLM_MEM);
		}

		return NE_XFLM_OK;
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
		RCODE			rc = NE_XFLM_OK;

		if( uiTimeout == F_SEM_WAITFOREVER)
		{
			if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
			{
				rc = RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE);
			}
		}
		else
		{
			if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
			{
				rc = RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE);
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
			return( RC_SET( NE_XFLM_COULD_NOT_CREATE_SEMAPHORE));
		}

		return NE_XFLM_OK;
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
			return( NE_XFLM_OK);
		}
		else
		{
			return( RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE));
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
		(void)sema_signal( hSem);
	}

#else
	#error Platform undefined
#endif

#endif 		// #ifndef FTKSEM_H
