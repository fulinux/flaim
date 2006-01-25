//-------------------------------------------------------------------------
// Desc:	TCP handler
// Tabs:	3
//
//		Copyright (c) 1999-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_tcph.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

RCODE fsvTcpListener(
	F_Thread *		pThread);

RCODE fsvTcpClientHandler(
	F_Thread *		pThread);

RCODE fsvTcpVulture(
	F_Thread *		pThread);

RCODE fsvTcpAcceptConnection(
	F_MUTEX *			phHandlerSem,
	FCS_TCP * 			pClient);

#define FSV_MAX_TCP_HANDLERS			64

FLMBOOL				gv_bTcpAllowConnections = TRUE;
FLMBOOL				gv_bTcpRunning = FALSE;
F_Thread *			gv_TcpHandlers[ FSV_MAX_TCP_HANDLERS];
F_Thread *			gv_pTcpListenerThrd = NULL;

/****************************************************************************
Desc:
****************************************************************************/
RCODE	fsvStartTcpListener(
	FLMUINT		uiPort)
{
	RCODE		rc = FERR_OK;

	if( RC_BAD( rc = f_threadCreate( &gv_pTcpListenerThrd,
		fsvTcpListener, "DB TCP Listener", 
		FLM_DEFAULT_THREAD_GROUP, 0, (void *)uiPort)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void fsvShutdownTcpListener( void)
{
	f_threadDestroy( &gv_pTcpListenerThrd);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvTcpListener(
	F_Thread *		pThread)
{
	RCODE						rc = FERR_OK;
	FCS_TCP_SERVER *		pServer;
	FCS_TCP *				pClient = NULL;
	F_Thread *				pVultureThread = NULL;
	F_MUTEX					hHandlerSem = F_MUTEX_NULL;
	FLMUINT					uiLoop;
	FLMUINT					uiPort = (FLMUINT)pThread->getParm1();

	/* Initialize TCP */

	if( (pServer = f_new FCS_TCP_SERVER) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pServer->bind( uiPort)))
	{
#ifdef FSV_LOGGING
		fsvLogHandlerMessage( NULL, "TCPH: Unable to bind to port.",
			rc, FSV_LOG_ERROR);
#endif
		gv_bTcpAllowConnections = FALSE;
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < FSV_MAX_TCP_HANDLERS; uiLoop++)
	{
		gv_TcpHandlers[ uiLoop] = NULL;
	}

	if( RC_BAD( rc = f_mutexCreate( &hHandlerSem)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_threadCreate( &pVultureThread,
		fsvTcpVulture, "DB TCP Vulture", 
		FLM_DEFAULT_THREAD_GROUP, 0, (void *)(&hHandlerSem))))
	{
		goto Exit;
	}

	gv_bTcpRunning = TRUE;

	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			goto Exit;
		}

		if( pClient == NULL)
		{
			if( (pClient = f_new FCS_TCP) == NULL)
			{
#ifdef FSV_LOGGING
				fsvLogHandlerMessage( NULL,
					"TCPH: Unable to create a new connection",
					rc, FSV_LOG_ERROR);
#endif
				f_sleep( 100);
				continue;
			}
		}

		/*
		See if a client is waiting to connect.
		*/

		if( RC_OK( rc = pServer->connectClient( pClient, 2, 1200)))
		{
			if( RC_BAD( fsvTcpAcceptConnection( &hHandlerSem, pClient)))
			{
				pClient->Release();
			}
			pClient = NULL;
		}
		else
		{
			if( rc != FERR_SVR_READ_TIMEOUT)
			{
#ifdef FSV_LOGGING
				fsvLogHandlerMessage( NULL,
					"TCPH: Error listening for connections.",
					rc, FSV_LOG_ERROR);

				fsvLogHandlerMessage( NULL, "TCPH: Attempting to reset.",
					rc, FSV_LOG_ERROR);
#endif

				/*
				Drop the current client.
				*/
				
				pClient->Release();
				pClient = NULL;
				
				/*
				Re-initialize the listener.
				*/

				pServer->Release();
				pServer = f_new FCS_TCP_SERVER;
				flmAssert( pServer != NULL);

				if( RC_BAD( rc = pServer->bind( uiPort)))
				{
#ifdef FSV_LOGGING
					fsvLogHandlerMessage( NULL, "TCPH: Unable to bind to port.",
						rc, FSV_LOG_ERROR);
#endif
					gv_bTcpAllowConnections = FALSE;
				}
				else
				{
#ifdef FSV_LOGGING
					fsvLogHandlerMessage( NULL, "TCPH: Server reset completed.",
						rc, FSV_LOG_EVENT);
#endif
				}
			}
		}
	}

Exit:

	// Shut down all threads and free any allocated resources
	
	f_threadDestroy( &pVultureThread);

	if( pClient)
	{
		pClient->Release();
	}

	if( pServer)
	{
		pServer->Release();
	}

	gv_bTcpRunning = FALSE;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvTcpVulture(
	F_Thread *		pThread)
{
	F_MUTEX *	phHandlerMutex = (F_MUTEX *)pThread->getParm1();
	FLMUINT		uiLoop;
	FLMUINT		uiActiveThreads;
	FLMUINT		uiRetry;
	FLMBOOL		bShutdown = FALSE;

	while( !bShutdown)
	{
		if( pThread->getShutdownFlag())
		{
			bShutdown = TRUE;
			break;
		}

		f_mutexLock( *phHandlerMutex);

		uiActiveThreads = 0;
		uiLoop = 0;
		while( uiLoop < FSV_MAX_TCP_HANDLERS)
		{
			if( gv_TcpHandlers[ uiLoop] != NULL)
			{
				if( !gv_TcpHandlers[ uiLoop]->isThreadRunning())
				{
#ifdef FSV_LOGGING
					fsvLogHandlerMessage( NULL, "TCPH: Thread resources discarded.",
						FERR_OK, FSV_LOG_DEBUG);
#endif
					f_threadDestroy( &(gv_TcpHandlers[ uiLoop]));
				}
				else
				{
					uiActiveThreads++;
				}
			}
			uiLoop++;
		}

		f_mutexUnlock( *phHandlerMutex);

		for( uiLoop = 0; uiLoop < 100; uiLoop++)
		{
			if( pThread->getShutdownFlag())
			{
				bShutdown = TRUE;
				break;
			}
			f_sleep( 100);
		}
	}

	uiRetry = 0;
	while( uiRetry < 60)
	{
		uiActiveThreads = 0;
		uiLoop = 0;
		while( uiLoop < FSV_MAX_TCP_HANDLERS)
		{
			if( gv_TcpHandlers[ uiLoop] != NULL)
			{
				if( !gv_TcpHandlers[ uiLoop]->isThreadRunning())
				{
					f_threadDestroy( &(gv_TcpHandlers[ uiLoop]));
				}
				else
				{
					gv_TcpHandlers[ uiLoop]->setShutdownFlag();
					uiActiveThreads++;
				}
			}
			uiLoop++;
		}

		if( uiActiveThreads == 0)
		{
			break;
		}
		f_sleep( 1000);
		uiRetry++;
	}

	return( FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvTcpAcceptConnection(
	F_MUTEX *			phHandlerMutex,
	FCS_TCP * 			pClient)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bMutexLocked = FALSE;
	FLMUINT			uiLoop;
	F_Thread *		pClientThrd;
	
	f_mutexLock( *phHandlerMutex);
	bMutexLocked = TRUE;

	uiLoop = 0;
	while( uiLoop < FSV_MAX_TCP_HANDLERS)
	{
		if( gv_TcpHandlers[ uiLoop] == NULL)
		{
			break;
		}
		uiLoop++;
	}

	if( uiLoop < FSV_MAX_TCP_HANDLERS)
	{
		if( RC_BAD( rc = f_threadCreate( &pClientThrd,
			fsvTcpClientHandler, "DB TCP Handler",
			FLM_DEFAULT_THREAD_GROUP, 0, pClient)))
		{
			goto Exit;
		}
		gv_TcpHandlers[ uiLoop] = pClientThrd;
	}
	else
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( *phHandlerMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE fsvTcpClientHandler(
	F_Thread *			pThread)
{
	RCODE				rc = FERR_OK;
	FCS_TCP *		pClient = (FCS_TCP *)pThread->getParm1();
	FCS_IPIS	*		pIpIStream;
	FCS_IPOS	*		pIpOStream = NULL;
	FCS_DIS *		pDataIStream = NULL;
	FCS_DOS *		pDataOStream = NULL;
	FLMUINT			uiSessionId = FCS_INVALID_ID;
	POOL				pool;
#ifdef FSV_LOGGING
	FLMBYTE			pucLogBuf[ 256];
#endif

	/*
	Initialize the scratch pool
	*/

	GedPoolInit( &pool, 2048);

#ifdef FSV_LOGGING
	f_sprintf( pucLogBuf, "TCPH: Connection accepted from %s",
		pClient->peerIpNameTxt());
	fsvLogHandlerMessage( NULL, pucLogBuf,
		FERR_OK, FSV_LOG_EVENT);
#endif

	/*
	Allocate required objects.
	*/

	if( (pIpIStream = f_new FCS_IPIS( pClient)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (pIpOStream = f_new FCS_IPOS( pClient)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (pDataIStream = f_new FCS_DIS) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (pDataOStream = f_new FCS_DOS) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			goto Exit;
		}

		if( RC_BAD( rc = pClient->socketPeekRead( 5)))
		{
			if( rc != FERR_SVR_READ_TIMEOUT)
			{
				goto Exit;
			}
		}
		else
		{

			/*
			Configure the data input stream.
			*/
			
			if( RC_BAD( rc = pDataIStream->setup( pIpIStream)))
			{
				goto Exit;
			}

			/*
			Configure the data output stream.
			*/
			
			if( RC_BAD( rc = pDataOStream->setup( pIpOStream)))
			{
				goto Exit;
			}

			/*
			Process the request.
			*/

			if( RC_BAD( rc = fsvProcessRequest( pDataIStream,
				pDataOStream, &pool, &uiSessionId)))
			{
				goto Exit;
			}
		}
	}

Exit:
	
#ifdef FSV_LOGGING
	if( pClient)
	{
		f_sprintf( pucLogBuf, "TCPH: %s disconnected.",
			pClient->peerIpNameTxt());
	}
	else
	{
		f_sprintf( pucLogBuf, "TCPH: <UNKNOWN> disconnected.");
	}

	fsvLogHandlerMessage( NULL, pucLogBuf,
		FERR_OK, FSV_LOG_EVENT);
#endif

	if( pDataIStream)
	{
		pDataIStream->Release();
	}

	if( pDataOStream)
	{
		pDataOStream->Release();
	}

	if( pIpIStream)
	{
		pIpIStream->Release();
	}

	if( pIpOStream)
	{
		pIpOStream->Release();
	}

	if( pClient)
	{
		pClient->Release();
	}

	if( RC_BAD( rc) && uiSessionId != FCS_INVALID_ID)
	{
		FSV_SCTX *	pServerContext = NULL;

		// Close the session and release any resources 
		// held by the client (open transactions, etc.)

		if( RC_OK( fsvGetGlobalContext( &pServerContext)))
		{
			pServerContext->CloseSession( uiSessionId);
		}

#ifdef FSV_LOGGING
		fsvLogHandlerMessage( NULL,
			"Session discarded.", FERR_OK, FSV_LOG_DEBUG);
#endif
	}

	GedPoolFree( &pool);
	return( rc);
}
