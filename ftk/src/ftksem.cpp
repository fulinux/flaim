//------------------------------------------------------------------------------
// Desc:	This file contains mutex and semaphore functions
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
// $Id: ftksem.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
typedef struct
{
	pthread_mutex_t		lock;
	pthread_cond_t			cond;
	int						count;
} sema_t;
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WIN)
RCODE FLMAPI f_mutexCreate(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if( (*phMutex = (F_MUTEX)malloc( sizeof( F_INTERLOCK))) == F_MUTEX_NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	((F_INTERLOCK *)(*phMutex))->locked = 0;
#ifdef FLM_DEBUG
	((F_INTERLOCK *)(*phMutex))->uiThreadId = 0;
	((F_INTERLOCK *)(*phMutex))->lockedCount = 0;
	((F_INTERLOCK *)(*phMutex))->waitCount = 0;
#endif

	return( NE_FLM_OK);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WIN)
void FLMAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
RCODE FLMAPI f_mutexCreate(
	F_MUTEX *			phMutex)
{
	RCODE								rc = NE_FLM_OK;
	pthread_mutexattr_t *		pMutexAttr = NULL;

	f_assert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)malloc( 
		sizeof( pthread_mutex_t))) == F_MUTEX_NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

#if defined( FLM_DEBUG) && defined( FLM_LINUX)
	{
		pthread_mutexattr_t			mutexAttr;
	
		if( !pthread_mutexattr_init( &mutexAttr))
		{
			pMutexAttr = &mutexAttr;
			pthread_mutexattr_settype( pMutexAttr, PTHREAD_MUTEX_ERRORCHECK_NP);
		}
	}
#endif

	if( pthread_mutex_init( (pthread_mutex_t *)*phMutex, pMutexAttr) != 0)
	{
		// NOTE: Cannot call f_free because we had to use malloc up above due
		// to the fact that the memory subsystem uses a mutex before itis
		// completely ready to go.

		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
		rc = RC_SET( NE_FLM_COULD_NOT_CREATE_MUTEX);
		goto Exit;
	}

Exit:

	if( pMutexAttr)
	{
		pthread_mutexattr_destroy( pMutexAttr);
	}

	return( rc);
}
#endif
			  
/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
RCODE FLMAPI f_mutexCreate(
	F_MUTEX *			phMutex)
{
	RCODE					rc = NE_FLM_OK;
	lwp_mutex_t			defaultMutex = DEFAULTMUTEX;

	f_assert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)malloc( sizeof( lwp_mutex_t))) == F_MUTEX_NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	f_memcpy( *phMutex, &defaultMutex, sizeof( lwp_mutex_t));

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
void FLMAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		pthread_mutex_destroy( (pthread_mutex_t *)*phMutex);

		// NOTE: Cannot call f_free because we had to use malloc up above due
		// to the fact that the memory subsystem uses a mutex before it is
		// completely ready to go.

		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
void FLMAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	f_assert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
void FLMAPI f_mutexLock(
	F_MUTEX		hMutex)
{
	(void)pthread_mutex_lock( (pthread_mutex_t *)hMutex);
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
void FLMAPI f_mutexLock(
	F_MUTEX		hMutex)
{
	for( ;;)
	{
		if( _lwp_mutex_lock( (lwp_mutex_t *)hMutex) == 0)
		{
			break;
		}
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
void FLMAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	(void)pthread_mutex_unlock( (pthread_mutex_t *)hMutex);
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS)
void FLMAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	_lwp_mutex_unlock( (lwp_mutex_t *)hMutex);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
void FLMAPI f_assertMutexLocked(
	F_MUTEX)
{
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FLMAPI f_mutexCreate(
	F_MUTEX *	phMutex)
{
	if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"FTK_MUTEX")) == F_MUTEX_NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}

	return( NE_FLM_OK);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FLMAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	if (*phMutex != F_MUTEX_NULL)
	{
		(void)kMutexFree( (MUTEX)(*phMutex));
		*phMutex = F_MUTEX_NULL;
	}
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FLMAPI f_mutexLock( 
	F_MUTEX		hMutex)
{
	(void)kMutexLock( (MUTEX)hMutex);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FLMAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	(void)kMutexUnlock( (MUTEX)hMutex);
}
#endif
	
/****************************************************************************
Desc:	Initializes a semaphore handle on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE int sema_init(
	sema_t *			pSem)
{
	int iErr = 0;

	if( (iErr = pthread_mutex_init( &pSem->lock, NULL)) < 0)
	{
		goto Exit;
	}

	if( (iErr = pthread_cond_init( &pSem->cond, NULL)) < 0)
	{
		pthread_mutex_destroy( &pSem->lock);
		goto Exit;
	}

	pSem->count = 0;

Exit:

	return( iErr);
}
#endif

/****************************************************************************
Desc:	Frees a semaphore handle on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE void sema_destroy(
	sema_t *			pSem)
{
	pthread_mutex_destroy( &pSem->lock);
	pthread_cond_destroy( &pSem->cond);
}
#endif

/****************************************************************************
Desc:	Waits for a semaphore to be signaled on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE int _sema_wait(
	sema_t *			pSem)
{
	int	iErr = 0;

	pthread_mutex_lock( &pSem->lock);
	while( !pSem->count)
	{
		if( (iErr = pthread_cond_wait( &pSem->cond, &pSem->lock)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
			}
			else
			{
				f_assert( 0);
				goto Exit;
			}
		}
	}

	pSem->count--;
	f_assert( pSem->count >= 0);

Exit:

	pthread_mutex_unlock( &pSem->lock);
	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits for a semaphore to be signaled on Solaris
****************************************************************************/
#if defined( FLM_SOLARIS)
FINLINE int _sema_wait(
	sema_t *			pSem)
{
	int	iErr = 0;

	for( ;;)
	{
		if( (iErr = sema_wait( pSem)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
				continue;
			}
			else
			{
				f_assert( 0);
				goto Exit;
			}
		}

		break;
	}

Exit:

	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits a specified number of milliseconds for a semaphore
		to be signaled on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
FINLINE int _sema_timedwait(
	sema_t *			pSem,
	unsigned int	msecs)
{
	int					iErr = 0;
   struct timeval		now;
	struct timespec	abstime;

   // If timeout is F_SEM_WAITFOREVER, do sem_wait.

   if( msecs == F_SEM_WAITFOREVER)
   {
      iErr = _sema_wait( pSem);
      return( iErr);
   }

   gettimeofday( &now, NULL);
	abstime.tv_sec = now.tv_sec + ((msecs) ? (msecs / 1000) : 0);
	abstime.tv_nsec = ( now.tv_usec + ((msecs % 1000) *	1000)) * 1000;

	pthread_mutex_lock( &pSem->lock);

Restart:

	while( !pSem->count)
	{
		if( (iErr = pthread_cond_timedwait( &pSem->cond,
			&pSem->lock, &abstime)) != 0)
		{
			if( iErr == EINTR)
			{
				iErr = 0;
				goto Restart;
			}
			goto Exit;
		}
	}

	pSem->count--;
	f_assert( pSem->count >= 0);

Exit:

	pthread_mutex_unlock( &pSem->lock);
	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits a specified number of milliseconds for a semaphore
		to be signaled on UNIX
****************************************************************************/
#if defined( FLM_SOLARIS)
FINLINE int _sema_timedwait(
	sema_t *			pSem,
	unsigned int	msecs)
{
	int					iErr = 0;
   struct timeval		now;
	struct timespec	abstime;

   // If timeout is F_SEM_WAITFOREVER, do sem_wait.

   if( msecs == F_SEM_WAITFOREVER)
   {
      iErr = _sema_wait( pSem);
      return( iErr);
   }

   gettimeofday( &now, NULL);
	abstime.tv_sec = now.tv_sec + ((msecs) ? (msecs / 1000) : 0);
	abstime.tv_nsec = ( now.tv_usec + ((msecs % 1000) *	1000)) * 1000;

Restart:

	if( (iErr = sema_timedwait( pSem, &abstime)) != 0)
	{
		if( iErr == EINTR)
		{
			iErr = 0;
			goto Restart;
		}
		goto Exit;
	}

Exit:

	return( iErr);
}
#endif

/****************************************************************************
Desc:	Signals a semaphore on UNIX
****************************************************************************/
#if (defined( FLM_UNIX) || defined( FLM_LIBC_NLM)) && !defined( FLM_SOLARIS)
int sema_signal(
	sema_t *			pSem)
{
	pthread_mutex_lock( &pSem->lock);
	pSem->count++;
	f_assert( pSem->count > 0);
	pthread_cond_signal( &pSem->cond);
	pthread_mutex_unlock( &pSem->lock);

	return( 0);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE f_semCreate(
	F_SEM *		phSem)
{
	RCODE	rc = NE_FLM_OK;

	f_assert( phSem != NULL);

	if( RC_BAD( rc = f_alloc( sizeof( sema_t), phSem)))
	{
		goto Exit;
	}

#if defined( FLM_SOLARIS)
	if( sema_init( (sema_t *)*phSem, 0, USYNC_THREAD, NULL) < 0) 
#else
	if( sema_init( (sema_t *)*phSem) < 0)
#endif
	{
		f_free( phSem);
		*phSem = F_SEM_NULL;
		rc = RC_SET( NE_FLM_COULD_NOT_CREATE_SEMAPHORE);
		goto Exit;
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
void f_semDestroy(
	F_SEM  *		phSem)
{
	f_assert( phSem != NULL);

	if (*phSem != F_SEM_NULL)
	{
		sema_destroy( (sema_t *)*phSem);
		f_free( phSem);
		*phSem = F_SEM_NULL;
	}
}
#endif

/****************************************************************************
Desc:   Get the lock on a semaphore - p operation
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	RCODE			rc	= NE_FLM_OK;

	f_assert( hSem != F_SEM_NULL);

	//catch the F_SEM_WAITFOREVER flag so we can directly call _sema_wait
	//instead of passing F_SEM_WAITFOREVER through to _sema_timedwait.
	//Note that on AIX the datatype of the uiTimeout (in the timespec
	//struct) is surprisingly a signed int, which makes this catch
	//essential.

	if( uiTimeout == F_SEM_WAITFOREVER)
	{
		if( _sema_wait( (sema_t *)hSem))
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}
	else
	{
		if( _sema_timedwait( (sema_t *)hSem, (unsigned int)uiTimeout))
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}

	return( rc);
}
#endif

/****************************************************************************
Desc:   Get the lock on a semaphore - p operation
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
void FLMAPI f_semSignal(
	F_SEM			hSem)
{
#if defined( FLM_SOLARIS)
	sema_post( (sema_t *)hSem);
#else
	sema_signal( (sema_t *)hSem);
#endif
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FLMAPI f_semCreate(
	F_SEM *		phSem)
{
	if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"FTK_SEM", 0)) == F_SEM_NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FLMAPI f_semDestroy(
	F_SEM *		phSem)
{
	if (*phSem != F_SEM_NULL)
	{
		(void)kSemaphoreFree( (SEMAPHORE)(*phSem));
		*phSem = F_SEM_NULL;
	}
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
RCODE FLMAPI f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	RCODE			rc = NE_FLM_OK;
	
	if( uiTimeout == F_SEM_WAITFOREVER)
	{
		if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}
	else
	{
		if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}
	
	return( rc);
}
#endif

/*************************************************************************
Desc:
*************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void FLMAPI f_semSignal(
	F_SEM			hSem)
{
	(void)kSemaphoreSignal( (SEMAPHORE)hSem);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FLMAPI f_mutexLock(
	F_MUTEX			hMutex)
{
	F_INTERLOCK *		pInterlock = (F_INTERLOCK *)hMutex;

#ifdef FLM_DEBUG
	if( pInterlock->locked)
	{
		f_assert( pInterlock->uiThreadId != _threadid);
	}
#endif

	while( f_atomicExchange( &pInterlock->locked, 1) != 0)
	{
#ifdef FLM_DEBUG
		f_atomicInc( &pInterlock->waitCount);
#endif
		Sleep( 0);
	}

#ifdef FLM_DEBUG
	f_assert( pInterlock->uiThreadId == 0);
	pInterlock->uiThreadId = _threadid;
	f_atomicInc( &pInterlock->lockedCount);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FLMAPI f_mutexUnlock(
	F_MUTEX		hMutex)
{
	F_INTERLOCK *		pInterlock = (F_INTERLOCK *)hMutex;

	f_assert( pInterlock->locked == 1);
#ifdef FLM_DEBUG
	f_assert( pInterlock->uiThreadId == _threadid);
	pInterlock->uiThreadId = 0;
#endif
	f_atomicExchange( &pInterlock->locked, 0);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FLMAPI f_assertMutexLocked(
	F_MUTEX		hMutex)
{
#ifdef FLM_DEBUG
	f_assert( ((F_INTERLOCK *)hMutex)->locked == 1);
	f_assert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
#else
	F_UNREFERENCED_PARM( hMutex);
#endif
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
RCODE FLMAPI f_semCreate(
	F_SEM *		phSem)
{
	if( (*phSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL,
		0, 10000, NULL )) == NULL)
	{
		return( RC_SET( NE_FLM_COULD_NOT_CREATE_SEMAPHORE));
	}

	return NE_FLM_OK;
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FLMAPI f_semDestroy(
	F_SEM *		phSem)
{
	if (*phSem != F_SEM_NULL)
	{
		CloseHandle( *phSem);
		*phSem = F_SEM_NULL;
	}
}
#endif
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
RCODE FLMAPI f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	if( WaitForSingleObject( hSem, uiTimeout ) == WAIT_OBJECT_0)
	{
		return( NE_FLM_OK);
	}
	else
	{
		return( RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE));
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_WIN
void FLMAPI f_semSignal(
	F_SEM			hSem)
{
	(void)ReleaseSemaphore( hSem, 1, NULL);
}
#endif
