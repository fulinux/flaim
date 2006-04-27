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
#if defined( FLM_UNIX)
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
	flmAssert( phMutex != NULL);

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
	flmAssert( phMutex != NULL);

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
#if defined( FLM_UNIX)
RCODE FLMAPI f_mutexCreate(
	F_MUTEX *			phMutex)
{
	RCODE								rc = NE_FLM_OK;
	pthread_mutexattr_t *		pMutexAttr = NULL;

	flmAssert( phMutex != NULL);

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
#if defined( FLM_UNIX)
void FLMAPI f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

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
Desc:	Initializes a semaphore handle on UNIX
****************************************************************************/
#if defined( FLM_UNIX)
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
#if defined( FLM_UNIX)
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
#if defined( FLM_UNIX)
FINLINE int sema_wait(
	sema_t *			pSem)
{
	int	iErr = 0;

	pthread_mutex_lock( &pSem->lock);
	while( !pSem->count)
	{
		if( (iErr = pthread_cond_wait( &pSem->cond, &pSem->lock)))
		{
			if( iErr == EINTR)
			{
				iErr = 0;
			}
			else
			{
				goto Exit;
			}
		}
	}

	pSem->count--;
	flmAssert( pSem->count >= 0);

Exit:

	pthread_mutex_unlock( &pSem->lock);
	return( iErr);
}
#endif

/****************************************************************************
Desc:	Waits a specified number of milliseconds for a semaphore
		to be signaled on UNIX
****************************************************************************/
#if defined( FLM_UNIX)
FINLINE int sema_timedwait(
	sema_t *			pSem,
	unsigned int	msecs)
{
   struct timeval		now;
	struct timespec	abstime;
	int					iErr = 0;

   // If timeout is F_SEM_WAITFOREVER, do sem_wait.

   if( msecs == F_SEM_WAITFOREVER)
   {
      iErr = sema_wait( pSem);
      return( iErr);
   }

	pthread_mutex_lock( &pSem->lock);

   gettimeofday( &now, NULL);
	abstime.tv_sec = now.tv_sec + ((msecs) ? (msecs / 1000) : 0);
	abstime.tv_nsec = ( now.tv_usec + ((msecs % 1000) *	1000)) * 1000;

Restart:

	while( !pSem->count)
	{
		if( (iErr = pthread_cond_timedwait( &pSem->cond,
			&pSem->lock, &abstime)))
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
	flmAssert( pSem->count >= 0);

Exit:

	pthread_mutex_unlock( &pSem->lock);
	return( iErr);
}
#endif

/****************************************************************************
Desc:	Signals a semaphore on UNIX
****************************************************************************/
#if defined( FLM_UNIX)
int sema_signal(
	sema_t *			pSem)
{
	pthread_mutex_lock( &pSem->lock);
	pSem->count++;
	flmAssert( pSem->count > 0);
	pthread_cond_signal( &pSem->cond);
	pthread_mutex_unlock( &pSem->lock);

	return( 0);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX)
RCODE f_semCreate(
	F_SEM *		phSem)
{
	RCODE	rc = NE_FLM_OK;

	flmAssert( phSem != NULL);

	if( RC_BAD( rc = f_alloc( sizeof( sema_t), phSem)))
	{
		goto Exit;
	}

	if( sema_init( (sema_t *)*phSem) < 0)
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
#if defined( FLM_UNIX)
void f_semDestroy(
	F_SEM  *		phSem)
{
	flmAssert( phSem != NULL);

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
#if defined( FLM_UNIX)
RCODE f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	RCODE			rc	= NE_FLM_OK;

	flmAssert( hSem != F_SEM_NULL);

	//catch the F_SEM_WAITFOREVER flag so we can directly call sema_wait
	//instead of passing F_SEM_WAITFOREVER through to sema_timedwait.
	//Note that on AIX the datatype of the uiTimeout (in the timespec
	//struct) is surprisingly a signed int, which makes this catch
	//essential.

	if( uiTimeout == F_SEM_WAITFOREVER)
	{
		if( sema_wait( (sema_t *)hSem))
		{
			rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}
	else
	{
		if( sema_timedwait( (sema_t *)hSem, (unsigned int)uiTimeout))
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
#if defined( FLM_UNIX)
FINLINE void FLMAPI f_semSignal(
	F_SEM			hSem)
{
	(void)sema_signal( (sema_t *)hSem);
}
#endif

/****************************************************************************
Desc:   Get the lock on a semaphore - p operation
****************************************************************************/
#if defined( FLM_WATCOM_NLM)
int gv_DummyFtksem(void)
{
	return( 0);
}
#endif
