//-------------------------------------------------------------------------
// Desc:	Server context class.
// Tabs:	3
//
//		Copyright (c) 1998-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_sctx.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
*****************************************************************************/
FSV_SCTX::FSV_SCTX( void)
{
	m_uiSessionToken = 0;
	m_uiCacheSize = FSV_DEFAULT_CACHE_SIZE;
	m_bSetupCalled = FALSE;
	m_paSessions = NULL;
	m_hMutex = F_MUTEX_NULL;
	m_szServerBasePath[ 0] = '\0';
	m_pLogFunc = NULL;
}

/****************************************************************************
Desc:
*****************************************************************************/
FSV_SCTX::~FSV_SCTX( void)
{
	FLMUINT	uiSlot;

	if( m_bSetupCalled)
	{
		/*
		Clean up and free the session table.
		*/

		for( uiSlot = 0; uiSlot < m_uiMaxSessions; uiSlot++)
		{
			if( m_paSessions[ uiSlot] != NULL)
			{
				m_paSessions[ uiSlot]->Release();
			}
		}

		f_free( &m_paSessions);
	
		/*
		Free the session semaphore.
		*/
		
		(void)f_mutexDestroy( &m_hMutex);
	}
}


/****************************************************************************
Desc:	Configures and initializes the server context.
*****************************************************************************/
RCODE FSV_SCTX::Setup(
	FLMUINT			uiMaxSessions,
	const char *	pszServerBasePath,
	FSV_LOG_FUNC 	pLogFunc)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiSlot;

	/*
	Make sure that setup has not been called.
	*/
	
	flmAssert( m_bSetupCalled == FALSE);

	/*
	If zero was passed as the value of uiMaxSessions,
	use the default.
	*/

	if( !uiMaxSessions)
	{
		m_uiMaxSessions = FSV_DEFAULT_MAX_CONNECTIONS;
	}
	else
	{
		m_uiMaxSessions = uiMaxSessions;
	}

	/*
	Initialize the session table.
	*/

	if( RC_BAD( rc = f_alloc( sizeof( FSV_SESN *) * m_uiMaxSessions,
								&m_paSessions)))
	{
		goto Exit;
	}
	
	for( uiSlot = 0; uiSlot < m_uiMaxSessions; uiSlot++)
	{
		m_paSessions[ uiSlot] = NULL;
	}
	
	/*
	Initialize the context mutex
	*/
	
	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	/*
	Set the server's home path.
	*/

	if( pszServerBasePath)
	{
		f_strcpy( m_szServerBasePath, pszServerBasePath);
	}
	else
	{
		m_szServerBasePath[ 0] = '\0';
	}

	/*
	Set the logging function.
	*/

	m_pLogFunc = pLogFunc;

	/*
	Set the setup flag.
	*/

	m_bSetupCalled = TRUE;

Exit:

	/*
	Clean up any allocations if an error was encountered.
	*/
	
	if( RC_BAD( rc))
	{
		if( m_paSessions != NULL)
		{
			f_free( &m_paSessions);
		}

		if( m_hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &m_hMutex);
		}
	}

	return( rc);
}

  
/****************************************************************************
Desc:	Creates a new session object and adds it to the session table.
*****************************************************************************/
RCODE FSV_SCTX::OpenSession(
	FLMUINT			uiVersion,
	FLMUINT			uiFlags,
	FLMUINT *		puiIdRV,
	FSV_SESN **		ppSessionRV)
{
	FLMUINT		uiSlot;
	FLMUINT		uiCurrTime;
	FLMBOOL		bLocked = FALSE;
	FSV_SESN	*	pSession = NULL;
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Initialize the Id
	*/

	*puiIdRV = 0;

	/*
	Create a new session object
	*/

	if( (pSession = f_new FSV_SESN) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Allocate the session object
	*/

	if( RC_BAD( rc = pSession->Setup( this, uiVersion, uiFlags)))
	{
		goto Exit;
	}

	/*
	Lock the context mutex
	*/

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	/*
	Find an empty slot in the table.
	*/

	for( uiSlot = 0; uiSlot < m_uiMaxSessions; uiSlot++)
	{
		if( !m_paSessions[ uiSlot])
		{
			break;
		}
	}

	if( uiSlot >= m_uiMaxSessions)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Assign the session to the table slot.
	*/

	m_paSessions[ uiSlot] = pSession;

	/*
	Increment the session token.
	*/

	m_uiSessionToken++;

	/*
	If the session token is 0xFFFF, reset it to 1.  Because FSV_INVALID_ID
	is 0xFFFFFFFF, it is important to reset the session token so that
	a session will never be assigned an invalid ID.
	*/

	if( m_uiSessionToken == 0xFFFF)
	{
		m_uiSessionToken = 1;
	}

	/*
	Set the session's ID.
	*/

	*puiIdRV = uiSlot | (m_uiSessionToken << 16);
	pSession->setId( *puiIdRV);

	/*
	Set the session's cookie using the current time.
	*/

	f_timeGetSeconds( &uiCurrTime);
	pSession->setCookie( uiCurrTime);

	/*
	Unlock the context mutex
	*/

	f_mutexUnlock( m_hMutex);
	bLocked = FALSE;

Exit:

	if( RC_BAD( rc))
	{
		if( pSession)
		{
			pSession->Release();
			pSession = NULL;
		}
	}
	else
	{
		if( ppSessionRV)
		{
			*ppSessionRV = pSession;
		}
	}

	if( bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}


/****************************************************************************
Desc:	Closes a session and removes it from the session table.
*****************************************************************************/
RCODE FSV_SCTX::CloseSession(
	FLMUINT		uiId)
{
	FLMUINT		uiSlot = (0x0000FFFF & uiId);
	FLMBOOL		bLocked = FALSE;
	FSV_SESN	*	pSession = NULL;
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Lock the context mutex
	*/

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	/*
	Make sure that the slot is valid.
	*/

	if( uiSlot >= m_uiMaxSessions)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	/*
	Get a pointer to the table entry.
	*/
	
	if( (pSession = m_paSessions[ uiSlot]) == NULL)
	{
		// Session already closed
		goto Exit;
	}

	/*
	Verify the session ID.
	*/

	if( pSession->getId() != uiId)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Free the session.
	*/

	pSession->Release();

	/*
	Reset the table entry.
	*/
	
	m_paSessions[ uiSlot] = NULL;

Exit:

	if( bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}


/****************************************************************************
Desc:	Returns a pointer to a specific session.
*****************************************************************************/
RCODE FSV_SCTX::GetSession(
	FLMUINT			uiId,
	FSV_SESN **		ppSession)
{
	FLMUINT		uiSlot = (0x0000FFFF & uiId);
	FLMBOOL		bLocked = FALSE;
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Lock the context mutex
	*/

	f_mutexLock( m_hMutex);
	bLocked = TRUE;

	/*
	Make sure that the slot is valid.
	*/

	if( uiSlot >= m_uiMaxSessions)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	/*
	Get a pointer to the entry in the session table.
	*/
	
	if( (*ppSession = m_paSessions[ uiSlot]) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Verify the session ID.
	*/

	if( (*ppSession)->getId() != uiId)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	if( bLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Sets the server's base (relative) path
*****************************************************************************/
RCODE	FSV_SCTX::SetBasePath(
	const char *		pszServerBasePath)
{
	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Lock the context mutex
	*/

	f_mutexLock( m_hMutex);

	/*
	Set the server's base path.
	*/

	if( pszServerBasePath)
	{
		f_strcpy( m_szServerBasePath, pszServerBasePath);
	}
	else
	{
		m_szServerBasePath[ 0] = '\0';
	}

	/*
	Unlock the context mutex
	*/

	f_mutexUnlock( m_hMutex);
	return( FERR_OK);
}


/****************************************************************************
Desc:	Copies the server's base path into a user-supplied path location
*****************************************************************************/
RCODE FSV_SCTX::GetBasePath(
	char *	pszServerBasePath)
{
	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Lock the context mutex
	*/

	f_mutexLock( m_hMutex);

	/*
	Copy the base path.
	*/

	f_strcpy( pszServerBasePath, m_szServerBasePath);

	/*
	Unlock the context mutex
	*/

	f_mutexUnlock( m_hMutex);
	return( FERR_OK);
}


/****************************************************************************
Desc:	Builds and IO_PATH given a file's URL.  The file must be located on
		the server.  This routine assumes that the host and port match the
		servers host and port.
*****************************************************************************/
RCODE FSV_SCTX::BuildFilePath(
	const FLMUNICODE *	puzUrlString,
	char *					pszFilePathRV)
{
	RCODE				rc = FERR_OK;
	char				szBasePath[ F_PATH_MAX_SIZE];
	FUrl				Url;
	char *			pucAsciiUrl;
	const char *	pszFile;
	POOL				tmpPool;

	/*
	Initialize a temporary pool.
	*/

	GedPoolInit( &tmpPool, 256);

	/*
	Attempt to convert the UNICODE URL to a native string
	*/

	if( RC_BAD( rc = fcsConvertUnicodeToNative( &tmpPool,
		puzUrlString, &pucAsciiUrl)))
	{
		goto Exit;
	}

	/*
	Parse the URL.
	*/

	if( RC_BAD( rc = Url.SetUrl( pucAsciiUrl)))
	{
		goto Exit;
	}

	pszFile = Url.GetFile();

	if( Url.GetRelative())
	{
		/*
		Get the server's base path.
		*/

		GetBasePath( szBasePath);

		/*
		Build the database path.
		*/
		
		f_strcpy( pszFilePathRV, szBasePath);
		if( RC_BAD( rc = f_pathAppend( pszFilePathRV, pszFile)))
		{
			goto Exit;
		}
	}
	else
	{
		/*
		Absolute path.  Use the path "as-is."
		*/

		f_strcpy( pszFilePathRV, pszFile);
	}

Exit:

	GedPoolFree( &tmpPool);
	return( rc);
}


/****************************************************************************
Desc:	Sets the server's temporary directory.
*****************************************************************************/
RCODE FSV_SCTX::SetTempDir(
	const char *	pszTempDir)
{
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Set the temporary directory.  There is no need to lock the context semaphore
	because the state of the context is not being changed.
	*/

	if( RC_BAD( rc = FlmConfig(
		FLM_TMPDIR, (void *)pszTempDir, 0)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/******************************************************************************
Desc: Logs a message
*****************************************************************************/
void FSV_SCTX::LogMessage(
	FSV_SESN *		pSession,
	const char *	pucMsg,
	RCODE				rc,
	FLMUINT			uiMsgSeverity)
{
	if( m_pLogFunc)
	{
		f_mutexLock( m_hMutex);
		m_pLogFunc( pucMsg, rc, uiMsgSeverity, (void *)pSession);
		f_mutexUnlock( m_hMutex);
	}
}
