//-------------------------------------------------------------------------
// Desc:	Cross platform toolkit for named semaphores.
// Tabs:	3
//
//		Copyright (c) 2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftknsem.cpp 12330 2006-01-23 10:07:04 -0700 (Mon, 23 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_WIN)

/****************************************************************************
Desc:   create or open a named semaphore (WIN implementation)
****************************************************************************/
F_NamedSemaphore::F_NamedSemaphore(
	const char *		pszName,
	FLMUINT				uiMaxCount,
	eNamedSemFlags		eFlags) : m_hSem(  F_SEM_NULL), m_bInitialized( FALSE)
							  
{
	HANDLE hTemp;

	hTemp = CreateSemaphore (NULL, uiMaxCount, uiMaxCount, pszName);
	
	// Check to see if the create failed, or if the semaphore already existed
	// but we asked to create a new one...
	
	if( (hTemp == NULL) ||
		  ((eFlags == OpenOnly) && (GetLastError() == ERROR_ALREADY_EXISTS)) )
	{
		goto Exit;
	}

	// If the semaphore already exists, then create will return a handle to it.
	// but will also set the error code to ERROR_ALREADY_EXISTS.  This only
	// matters if we've set the eFlags to OpenOnly.  Otherwise, we'll just
	// reset the error to success ...
	
	SetLastError( 0);

	m_hSem = hTemp;
	m_bInitialized = TRUE;
	
Exit:

	if( !m_bInitialized)
	{
		CloseHandle( hTemp);
		m_hSem = NULL;
	}
}

/****************************************************************************
Desc:   Named Semaphore destructor (WIN implementation)
****************************************************************************/
F_NamedSemaphore::~F_NamedSemaphore()
{
	if( m_bInitialized)
	{
		CloseHandle( m_hSem);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NamedSemaphore::wait()
{
	DWORD		udError;
	
	if( !m_bInitialized)
	{
		return( RC_SET( FERR_NAMED_SEMAPHORE_ERR));
	}

	udError = WaitForSingleObject( m_hSem, INFINITE);	

	return( (udError == WAIT_OBJECT_0)
				? FERR_OK
				: MapWinErrorToFlaim(udError, FERR_NAMED_SEMAPHORE_ERR) );
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NamedSemaphore::signal(
	FLMUINT		uiCount)
{

	RCODE			rc = FERR_OK;
	
	if( !m_bInitialized)
	{
		return( RC_SET( FERR_NAMED_SEMAPHORE_ERR));
	}

	if( !ReleaseSemaphore( m_hSem, uiCount, NULL))
	{
		rc = MapWinErrorToFlaim(GetLastError(), FERR_NAMED_SEMAPHORE_ERR);
	}
	
	return( rc);
}

/****************************************************************************
Desc: Remove the semaphore from memory

Note: Several objects may actually refer to the same semaphore.  This
		function only needs to be called once per semaphore, not once per
		object.  Once this function has been called, attempts by other 
		objects to reference this semaphore could result in errors or other
		unplesantness.  This is really only needed on UNIX and Netware; Win
		handles this sort of thing automatically.
****************************************************************************/
RCODE F_NamedSemaphore::destroy( void)
{
	// On Windows, semphore objects are automatically destroyed after the
	// last handle has been closed.  All this function does then is 
	// close the handle, and set m_bInitialized to false.  This effectively
	// keeps the current process from accessing the semaphore, but other
	// processes are still able to use it...
	
	CloseHandle( m_hSem);
	m_bInitialized = FALSE;
	return( FERR_OK);
}

#elif defined( FLM_UNIX)

#include <sys/ipc.h>
#include <sys/sem.h>

/****************************************************************************
Desc:   create or open a named semaphore (Unix implementation)
****************************************************************************/
F_NamedSemaphore::F_NamedSemaphore(
	const char	*		pszName,
	FLMUINT				uiMaxCount,
	eNamedSemFlags		eFlags) : m_hSem( 0), m_bInitialized( FALSE)
{
	FLMBOOL			bSemExists;
	struct sembuf	sops;

	// This switch is just for error checking.  These three cases are handled by
	// the logic below, but if eFlags is something else, we're in trouble...
	switch (eFlags)
	{
		case OpenOnly:
		case CreateOnly:
		case OpenOrCreate:
			break;
		default:
			flmAssert( 0);
	}

	// Check to see if the semaphore already exists...
	
	if( (m_hSem = semget( NameToUnixKey( pszName), 1, 0666)) != -1)
	{
		//It doesn't...
		
		bSemExists = true;

		//If we were supposed to create a new semaphore, then fail...
		
		if( eFlags == CreateOnly)
		{
			goto Exit;
		}
	}
	else
	{
		bSemExists = false;
		
		// Now, if the OpenOnly flag was set, we've got a problem...
		
		if (eFlags == OpenOnly)
		{
			goto Exit;
		}
		else
		{
			if( (m_hSem = semget (NameToUnixKey(pszName), 1,
										  IPC_CREAT | IPC_EXCL | 0666)) == -1)
			{
				goto Exit;
			}
		}
	}

	// If we created a new semaphore, then we want to initialize it's value to
	// uiMaxCount.  If we just opened an existing one, then we want to leave
	// it alone...

	if( !bSemExists)
	{
		sops.sem_num = 0;
		sops.sem_op = (unsigned)uiMaxCount;
		sops.sem_flg = 0;
		
		if( semop( m_hSem, &sops, 1) == -1)
		{
			goto Exit;
		}
	}

	m_bInitialized = TRUE;
	
Exit:
	
	return;
}

/****************************************************************************
Desc:   Named Semaphore destructor (Unix implementation)
****************************************************************************/
F_NamedSemaphore::~F_NamedSemaphore()
{
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NamedSemaphore::wait()
{
	RCODE				rc = FERR_OK;
	struct sembuf	sops;

	sops.sem_num = 0;
	sops.sem_op = -1;
	sops.sem_flg = 0;

	if( semop( m_hSem, &sops, 1) == -1)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_NAMED_SEMAPHORE_ERR);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NamedSemaphore::signal(
	FLMUINT			uiCount)
{
	RCODE 			rc = FERR_OK;
	struct sembuf 	sops;
	
	sops.sem_num = 0;
	sops.sem_op = 1;
	sops.sem_flg = 0;

	while( uiCount--)
	{
		if( semop( m_hSem, &sops, 1) == -1)
		{
			rc = MapErrnoToFlaimErr( errno, FERR_NAMED_SEMAPHORE_ERR);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Remove the semaphore from memory (Unix implementation)

Note: Several objects may actually refer to the same semaphore.  This
		function only needs to be called once per semaphore, not once per
		object.  Once this function has been called, attempts by other 
		objects to reference this semaphore could result in errors or other
		unplesantness.  This is really only needed on UNIX and Netware; WIN
		handles this sort of thing automatically.
****************************************************************************/
RCODE F_NamedSemaphore::destroy( void)
{
	RCODE		rc = FERR_OK;

	// Call semctl() to remove the semaphore.  If any other processes are
	// waiting on this semaphore, the wait() will fail.
	
	if( semctl( m_hSem, 0, IPC_RMID, NULL) == -1)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_NAMED_SEMAPHORE_ERR);
	}

	return( rc);
}

/****************************************************************************
Desc:   A private function that converts a string into a 32 bit
		  key value (Unix only)
****************************************************************************/
FLMUINT32 F_NamedSemaphore::NameToUnixKey(
	const char *	pszName)
{
	FLMBYTE			szLastInt[ 4] = { 0, 0, 0, 0};
	FLMUINT32 *		pui32Current;
	FLMUINT32		ui32Sum = 0;
	FLMUINT			uiTemp = 0;
	FLMUINT			uiLoop;

	// Basically, we're going to divide the name up into 4 character groups,
	// treat each of those groups as an unsigned int and add them all together.
	// For the last group - since it has a 75% chance of not being 4 bytes - 
	// we add 0's to the end until it is 4 bytes.

	for( uiLoop = 0; (uiLoop + 3) < f_strlen( pszName); uiLoop += 4)
	{
		pui32Current = (FLMUINT32 *)&pszName[ uiLoop];
		ui32Sum += *pui32Current;
	}
	
	// The prev loop took care of most of the string, but we have at most
	// three chars left in pszName that need to be added to ui32Sum...
	
	while( uiLoop < f_strlen( pszName))
	{
		szLastInt[ uiTemp] = pszName[ uiLoop];
		uiLoop++;
		uiTemp++;
	}

	pui32Current = (FLMUINT32 *)szLastInt;
	ui32Sum += *pui32Current;

	return( ui32Sum);
}

#elif defined( FLM_NLM)
	FLMUINT uiFTKNSEM_Dummy;
#else
	#error "Unsupported Platform"	
#endif
