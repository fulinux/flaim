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

#include "flaimsys.h"

#if defined( FLM_WIN)

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_mutexCreate(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

	if( (*phMutex = (F_MUTEX)os_malloc( sizeof( F_INTERLOCK))) == F_MUTEX_NULL)
	{
		return( RC_SET( NE_XFLM_MEM));
	}

	(*phMutex)->locked = 0;
#ifdef FLM_DEBUG
	(*phMutex)->uiThreadId = 0;
	(*phMutex)->lockedCount = 0;
	(*phMutex)->waitCount = 0;
#endif

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
void f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		os_free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}

#elif defined( FLM_UNIX)

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_mutexCreate(
	F_MUTEX *			phMutex)
{
	RCODE								rc = NE_XFLM_OK;
	pthread_mutexattr_t *		pMutexAttr = NULL;

	flmAssert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)os_malloc( 
		sizeof( pthread_mutex_t))) == F_MUTEX_NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
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

	if( pthread_mutex_init( *phMutex, pMutexAttr) != 0)
	{
		// NOTE: Cannot call f_free because we had to use os_malloc up above due
		// to the fact that the memory subsystem uses a mutex before itis
		// completely ready to go.

		os_free( *phMutex);
		*phMutex = F_MUTEX_NULL;
		rc = RC_SET( NE_XFLM_COULD_NOT_CREATE_MUTEX);
		goto Exit;
	}

Exit:

	if( pMutexAttr)
	{
		pthread_mutexattr_destroy( pMutexAttr);
	}

	return( rc);
}
			  
/****************************************************************************
Desc:
****************************************************************************/
void f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		pthread_mutex_destroy( *phMutex);

		// NOTE: Cannot call f_free because we had to use os_malloc up above due
		// to the fact that the memory subsystem uses a mutex before it is
		// completely ready to go.

		os_free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}

/****************************************************************************
Desc:	Initializes a semaphore handle on UNIX
****************************************************************************/
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

/****************************************************************************
Desc:	Frees a semaphore handle on UNIX
****************************************************************************/
FINLINE void sema_destroy(
	sema_t *			pSem)
{
	pthread_mutex_destroy( &pSem->lock);
	pthread_cond_destroy( &pSem->cond);
}

/****************************************************************************
Desc:	Waits for a semaphore to be signaled on UNIX
****************************************************************************/
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

/****************************************************************************
Desc:	Waits a specified number of milliseconds for a semaphore
		to be signaled on UNIX
****************************************************************************/
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

/****************************************************************************
Desc:	Signals a semaphore on UNIX
****************************************************************************/
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

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_semCreate(
	F_SEM *		phSem)
{
	RCODE	rc = NE_XFLM_OK;

	flmAssert( phSem != NULL);

	if( RC_BAD( rc = f_alloc( sizeof( sema_t), phSem)))
	{
		goto Exit;
	}

	if( sema_init( *phSem) < 0)
	{
		f_free( phSem);
		*phSem = F_SEM_NULL;
		rc = RC_SET( NE_XFLM_COULD_NOT_CREATE_SEMAPHORE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void f_semDestroy(
	F_SEM  *		phSem)
{
	flmAssert( phSem != NULL);

	if (*phSem != F_SEM_NULL)
	{
		sema_destroy( *phSem);
		f_free( phSem);
		*phSem = F_SEM_NULL;
	}
}

/****************************************************************************
Desc:   Get the lock on a semaphore - p operation
****************************************************************************/
RCODE f_semWait(
	F_SEM			hSem,
	FLMUINT		uiTimeout)
{
	RCODE			rc	= NE_XFLM_OK;

	flmAssert( hSem != F_SEM_NULL);

	//catch the F_SEM_WAITFOREVER flag so we can directly call sema_wait
	//instead of passing F_SEM_WAITFOREVER through to sema_timedwait.
	//Note that on AIX the datatype of the uiTimeout (in the timespec
	//struct) is surprisingly a signed int, which makes this catch
	//essential.

	if( uiTimeout == F_SEM_WAITFOREVER)
	{
		if( sema_wait( hSem))
		{
			rc = RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}
	else
	{
		if( sema_timedwait( hSem, (unsigned int)uiTimeout))
		{
			rc = RC_SET( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE);
		}
	}

	return( rc);
}
#endif

#if defined( FLM_WATCOM_NLM)
	int gv_DummyFtksem(void)
	{
		return( 0);
	}
#endif
