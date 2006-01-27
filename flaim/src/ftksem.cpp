//-------------------------------------------------------------------------
// Desc:	Cross platform toolkit for mutexes and semaphores.
// Tabs:	3
//
//		Copyright (c) 2000-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftksem.cpp 12299 2006-01-19 15:01:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_AIX) || defined( FLM_OSX)

/****************************************************************************
Desc:		Initializes a semaphore handle on AIX
****************************************************************************/
int sema_init( sema_t * sem)
{
	int iErr = 0;

	if( (iErr = pthread_mutex_init( &sem->lock, NULL)) < 0)
	{
		goto Exit;
	}

	if( (iErr = pthread_cond_init( &sem->cond, NULL)) < 0)
	{
		pthread_mutex_destroy( &sem->lock);
		goto Exit;
	}

	sem->count = 0;

Exit:

	return( iErr);
}

/****************************************************************************
Desc:		Frees a semaphore handle on AIX
****************************************************************************/
void sema_destroy( sema_t * sem)
{
	pthread_mutex_destroy( &sem->lock);
	pthread_cond_destroy( &sem->cond);
}

/****************************************************************************
Desc:		Waits for a semaphore to be signaled on AIX
****************************************************************************/
int sema_wait( sema_t * sem)
{
	int	iErr = 0;

	pthread_mutex_lock( &sem->lock);
	while( !sem->count)
	{
		if( (iErr = pthread_cond_wait( &sem->cond, &sem->lock)))
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
	sem->count--;
	flmAssert( sem->count >= 0);

Exit:

	pthread_mutex_unlock( &sem->lock);
	return( iErr);
}

/****************************************************************************
Desc:		Waits a specified number of miliseconds for a semaphore
			to be signaled on AIX
****************************************************************************/
int sema_timedwait( sema_t * sem, unsigned int msecs)
{
   struct timeval		now;
	struct timespec	abstime;
	int					iErr = 0;

   // If timeout is F_SEM_WAITFOREVER, do sem_wait.

   if( msecs == F_SEM_WAITFOREVER)
   {
      iErr = sema_wait( sem);
      return( iErr);
   }

	pthread_mutex_lock( &sem->lock);

Restart:

   gettimeofday( &now, NULL);
	abstime.tv_sec = now.tv_sec + ((msecs) ? (msecs / 1000) : 0);
	abstime.tv_nsec = ( now.tv_usec + ((msecs % 1000) *	1000)) * 1000;

	while( !sem->count)
	{
		if( (iErr = pthread_cond_timedwait( &sem->cond, 
			&sem->lock, &abstime)))
		{
			if( iErr == EINTR)
			{
				iErr = 0;
				goto Restart;
			}
			goto Exit;
		}
	}
	sem->count--;
	flmAssert( sem->count >= 0);

Exit:

	pthread_mutex_unlock( &sem->lock);
	return( iErr);
}

/****************************************************************************
Desc:		Signals a semaphore on AIX
****************************************************************************/
int sema_signal( sema_t * sem)
{
	pthread_mutex_lock( &sem->lock);
	sem->count++;
	flmAssert( sem->count > 0);
	pthread_cond_signal( &sem->cond);
	pthread_mutex_unlock( &sem->lock);

	return( 0);
}

#endif // FLM_AIX || FLM_OSX

/****************************************************************************/
/****************************************************************************/
				
#if defined( FLM_WIN)

/****************************************************************************/
/****************************************************************************/

/****************************************************************************
Desc:		create a semaphore handle for use later
Notes:	initial state of the semaphore is unlocked.
****************************************************************************/
RCODE f_mutexCreate(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)os_malloc( 
		sizeof( CRITICAL_SECTION))) == F_MUTEX_NULL)
	{
		return( RC_SET( FERR_MUTEX_OPERATION_FAILED));
	}

	InitializeCriticalSection( (CRITICAL_SECTION *)*phMutex);
	return( FERR_OK);
}												

/****************************************************************************
Desc:		Destroy a semaphore that was created previously through f_mutexCreate
****************************************************************************/
void f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{

		DeleteCriticalSection( (CRITICAL_SECTION *)*phMutex);

		// NOTE: Cannot call f_free because we had to use os_malloc up above
		// due to the fact that the memory subsystem uses a mutex before it is
		// completely ready to go.

		os_free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}

#elif defined( FLM_UNIX)

/****************************************************************************/
/****************************************************************************/


/****************************************************************************
Desc:   create a semaphore handle for use later 
Notes:  initial state is unlocked
****************************************************************************/
RCODE f_mutexCreate(
	F_MUTEX *	phMutex)
{
	RCODE		rc = FERR_OK;

	flmAssert( phMutex != NULL);

	// NOTE: Cannot call f_alloc because the memory initialization needs
	// to be able to set up mutexes.

	if ((*phMutex = (F_MUTEX)os_malloc( 
		sizeof( pthread_mutex_t))) == F_MUTEX_NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( pthread_mutex_init( (pthread_mutex_t *)*phMutex, NULL) != 0)
		// SOLARIS2 needs attr==NULL, if default mutex attributes are used
	{

		// NOTE: Cannot call f_free because we had to use os_malloc up above
		// due to the fact that the memory subsystem uses a mutex before it is
		// completely ready to go.

		os_free( *phMutex);
		*phMutex = F_MUTEX_NULL;
		rc = RC_SET( FERR_MUTEX_OPERATION_FAILED);
		goto Exit;
	}

Exit:

	return( rc);
}												

/****************************************************************************
Desc:   create a semaphore handle for use later 
****************************************************************************/
void f_mutexDestroy(
	F_MUTEX *	phMutex)
{
	flmAssert( phMutex != NULL);

	if (*phMutex != F_MUTEX_NULL)
	{
		pthread_mutex_destroy( (pthread_mutex_t *)*phMutex);

		// NOTE: Cannot call f_free because we had to use os_malloc up above
		// due to the fact that the memory subsystem uses a mutex before it is
		// completely ready to go.

		os_free( *phMutex);
		*phMutex = F_MUTEX_NULL;
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
void f_mutexLock(
	F_MUTEX		hMutex)
{
	while( pthread_mutex_lock( (pthread_mutex_t *)hMutex) != 0);
}

/****************************************************************************
Desc: 
****************************************************************************/
void f_mutexUnlock(
	F_MUTEX		hMutex)
{
	(void)pthread_mutex_unlock( (pthread_mutex_t *)hMutex);
}

/****************************************************************************
Desc:   create a semaphore handle for use later 
Notes:  initial state is locked
****************************************************************************/
RCODE f_semCreate(
	F_SEM *		phSem)
{
	RCODE	rc = FERR_OK;

	flmAssert( phSem != NULL);

#if defined( FLM_AIX) || defined( FLM_OSX)
	if( RC_BAD( rc = f_alloc( sizeof( sema_t), phSem)))
	{
		goto Exit;
	}
#else
	if( RC_BAD( rc = f_alloc( sizeof( sem_t), phSem)))
	{
		goto Exit;
	}
#endif

#if defined( FLM_AIX) || defined( FLM_OSX)
	if( sema_init( (sema_t *)*phSem) < 0)
#else
	if( sem_init( (sem_t *)*phSem, 0, 0) < 0)
#endif
	{
		f_free( phSem);
		*phSem = F_SEM_NULL;
		rc = RC_SET( FERR_SEM_OPERATION_FAILED);
		goto Exit;
	}

Exit:

	return( rc);
}		

/****************************************************************************
Desc:   Destroy a semaphore
****************************************************************************/
void f_semDestroy(
	F_SEM  *		phSem)
{
	flmAssert( phSem != NULL);
	
	if (*phSem != F_SEM_NULL)
	{
#if defined( FLM_AIX) || defined( FLM_OSX)
		sema_destroy( (sema_t *)*phSem);
#else
		sem_destroy( (sem_t *)*phSem);
#endif
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
	RCODE			rc	= FERR_OK;

	flmAssert( hSem != F_SEM_NULL);

#if defined( FLM_AIX) || defined( FLM_OSX)

	// Catch the F_SEM_WAITFOREVER flag so we can directly call sema_wait
	// instead of passing F_SEM_WAITFOREVER through to sema_timedwait.
	// Note that on AIX the datatype of the uiTimeout (in the timespec
	// struct) is surprisingly a signed int, which makes this catch
	// essential.
	
	if( uiTimeout == F_SEM_WAITFOREVER)
	{
		if( sema_wait( (sema_t *)hSem))
		{
			rc = RC_SET( FERR_SEM_UNABLE_TO_LOCK);
		}
	}
	else
	{
		if( sema_timedwait( (sema_t *)hSem, uiTimeout))
		{
			rc = RC_SET( FERR_SEM_UNABLE_TO_LOCK);
		}
	}
#else
	if( !uiTimeout)
	{
		if( sem_trywait( (sem_t *)hSem) != 0)
		{
			rc = RC_SET( FERR_SEM_UNABLE_TO_LOCK);
		}
	}
	else
	{
Restart:
		if( sem_wait( (sem_t *)hSem) != 0)
		{
			// Defect 243865.  Retry sem_wait on EINTR signal.
			// This is at best a temporary fix.  Signals are handled explicitly
			// in NDS.  Unfortunately DirXML loads a JVM into the DHOST
			// address space. Signal handling for JVM threads gets really messy. 
			// Hopefully, DirXML will be moved into a separate process
			// for Tao SP1. The sem_wait fix is needed till that time.

			if( errno == EINTR)
			{
			  goto Restart;
			}
			rc = RC_SET( FERR_SEM_UNABLE_TO_LOCK);
		}
	}

#endif

	return( rc);
}
#endif

#if defined( FLM_NLM) && !defined( __MWERKS__)

int gv_DummyFtksem(void)
{
	return( 0);
}

#endif
