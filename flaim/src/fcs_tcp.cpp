//-------------------------------------------------------------------------
// Desc:	TCP/IP networking.
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_tcp.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

// These must be defined BEFORE any includes.  Unfortunately, this
// also means that we can't use our FLM_HPUX define because it hasn't
// been set yet...

#if defined( __hpux) || defined( hpux) 
	#define _XOPEN_SOURCE_EXTENDED 1
	#define _INCLUDE_HPUX_SOURCE
#endif

#include "flaimsys.h"

#if defined( FLM_NLM) && !defined ( __MWERKS__)
	// Disable errors for "expression for 'while' is always false"
	// Needed for FD_SET macro
	#pragma warning 555 9 
#endif

#ifdef FLM_WIN
	#pragma warning(disable : 4127)	// conditional expression is constant (from FD_SET())
#endif

/********************************************************************
Desc: Constructor
*********************************************************************/
FCS_TCP::FCS_TCP( void)
{
	m_pszIp[ 0] = '\0';
	m_pszName[ 0] = '\0';
	m_pszPeerIp[ 0] = '\0';
	m_pszPeerName[ 0] = '\0';
	m_uiIOTimeout = 10;
	m_iSocket = INVALID_SOCKET;
	m_ulRemoteAddr = 0;
	m_bInitialized = FALSE;
	m_bConnected = FALSE;

#ifndef FLM_UNIX
	if( !WSAStartup( MAKEWORD(2, 0), &m_wsaData))
	{
		m_bInitialized = TRUE;
	}
#endif
}


/********************************************************************
Desc: Destructor
*********************************************************************/
FCS_TCP::~FCS_TCP( void )
{
	if( m_bConnected)
	{
		close();
	}

#ifndef FLM_UNIX
	if( m_bInitialized)
	{
		WSACleanup();
	}
#endif
}

/********************************************************************
Desc: Gets information about the local host machine.
*********************************************************************/
RCODE FCS_TCP::_GetLocalInfo( void)
{
	struct hostent *		pHostEnt;
	FLMUINT32				ui32IPAddr;
	RCODE						rc = FERR_OK;

	m_pszIp[ 0] = '\0';
	m_pszName[ 0] = '\0';

	if( m_pszName[ 0] == '\0')
	{
		if( gethostname( m_pszName, (unsigned)sizeof( m_pszName)))
		{
			rc = RC_SET( FERR_SVR_SOCK_FAIL);
			goto Exit;
		}
	}

	if( m_pszIp[ 0] == '\0' &&
		(pHostEnt = gethostbyname( m_pszName)) != NULL)
	{
		ui32IPAddr = (FLMUINT32)(*((unsigned long *)pHostEnt->h_addr));
		if( ui32IPAddr != (FLMUINT32)-1)
		{
			struct in_addr			InAddr;

			InAddr.s_addr = ui32IPAddr;
			f_strcpy( m_pszIp, inet_ntoa( InAddr));
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Gets information about the remote machine.
*********************************************************************/
RCODE FCS_TCP::_GetRemoteInfo( void)
{
	struct sockaddr_in 	SockAddrIn;
	char *					InetAddr = NULL;
	struct hostent	*		HostsName;
	RCODE						rc = FERR_OK;

	m_pszPeerIp[ 0] = '\0';
	m_pszPeerName[ 0] = '\0';

	SockAddrIn.sin_addr.s_addr = (unsigned)m_ulRemoteAddr;

	/*
	inet_ntoa() - converts a 32-bit value in in_addr format into an ASCII
	string representing the address in dotted notation.
	VISIT:
	NetWare: Macro in arpa/inet.h. "Apps with multiple threads should use
	NWinet_ntoa instead of inet_ntoa.  Then we can get rid of the semaphore!
	*/
	
	InetAddr = inet_ntoa( SockAddrIn.sin_addr );
	f_strcpy( m_pszPeerIp, InetAddr );
	
	/*
	Try to get the peer's host name by looking up his IP
	address.  If found, copy IP Host name "BEVIS@NOVELL.COM" to TCPInfo
	otherwise, use his IP address as IP name.
	VISIT:
	Netware: "If your app has multiple threads, use either NWgethostbyaddr
	or NetDBgethostbyaddr().  This does the blocking?  This may be done
	already in netdb.h - it is hard to tell.
	*/

	HostsName = gethostbyaddr( (char *)&SockAddrIn.sin_addr.s_addr,
		(unsigned)sizeof( unsigned long), AF_INET );

	if( HostsName != NULL)
	{
		f_strcpy( m_pszPeerName, (char*) HostsName->h_name );
	}
	else
	{
		if (!InetAddr)
		{
			InetAddr = inet_ntoa( SockAddrIn.sin_addr);
		}
		f_strcpy( m_pszPeerName, InetAddr );
	}
	
	return( rc);
}

/********************************************************************
Desc: Tests for socket data readiness
*********************************************************************/
RCODE FCS_TCP::_SocketPeek(
	FLMINT			iTimeoutVal,
	FLMBOOL			bPeekRead
	)
{
	struct timeval		TimeOut;
	int					iMaxDescs;
	fd_set				GenDescriptors;
	fd_set *				DescrRead;
	fd_set *				DescrWrt;
	RCODE					rc = FERR_OK;

	if( m_iSocket != INVALID_SOCKET)
	{
		FD_ZERO( &GenDescriptors );
		FD_SET( m_iSocket, &GenDescriptors );

		iMaxDescs = (int)(m_iSocket + 1);
		DescrRead = bPeekRead ? &GenDescriptors : NULL;
		DescrWrt  = bPeekRead ? NULL : &GenDescriptors;

		TimeOut.tv_sec = (long)iTimeoutVal;
		TimeOut.tv_usec = (long)0;

		if( select( iMaxDescs, DescrRead, DescrWrt, NULL, &TimeOut) < 0 )
		{
			rc = RC_SET( FERR_SVR_SELECT_ERR);
			goto Exit;
		}
		else
		{
			if( !FD_ISSET( m_iSocket, &GenDescriptors))
			{
				rc = bPeekRead 
					? RC_SET( FERR_SVR_READ_TIMEOUT)
					: RC_SET( FERR_SVR_WRT_TIMEOUT);
			}
		}
	}
	else
	{
		rc = RC_SET( FERR_SVR_CONNECT_FAIL);
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Writes data to the connection.
*********************************************************************/
RCODE FCS_TCP::write(
	FLMBYTE *		pucDataBuffer,
	FLMUINT			uiDataCnt,
	FLMUINT *		puiWrtCnt)
{
	FLMUINT	uiPartialCnt;
	FLMUINT	uiToWrite;
	FLMUINT	uiHaveWritten = 0;
	RCODE		rc = FERR_OK;

	if( m_iSocket == INVALID_SOCKET)
	{
		rc = RC_SET( FERR_SVR_CONNECT_FAIL);
	}

	uiToWrite = uiDataCnt;
	*puiWrtCnt = 0;
	while( uiToWrite > 0)
	{
		/* The internal write call checks the arguments. */
		
		if( RC_BAD( rc = _write( pucDataBuffer, 
			uiToWrite, &uiPartialCnt)))
		{
			goto Exit;
		}

		pucDataBuffer += uiPartialCnt;
		uiHaveWritten += uiPartialCnt;
		uiToWrite = (FLMUINT)(uiDataCnt - uiHaveWritten);
		*puiWrtCnt = uiHaveWritten;
	}

Exit:

	return( rc);
}


RCODE FCS_TCP::_write(
	FLMBYTE *		pucBuffer,
	FLMUINT			uiDataCnt,
	FLMUINT			*puiWrtCnt)
{
	FLMINT			iRetryCount = 0;
	FLMINT			iWrtCnt = 0;
	RCODE				rc = FERR_OK;

	flmAssert( m_iSocket != INVALID_SOCKET && pucBuffer && uiDataCnt);

Retry:

	*puiWrtCnt = 0;
	if ( RC_OK( rc = _SocketPeek( m_uiIOTimeout, FALSE)))
	{
		iWrtCnt = send( m_iSocket, (char *)pucBuffer, (int)uiDataCnt, 0 );
		switch ( iWrtCnt )
		{
			case -1:
				*puiWrtCnt = 0;
				rc = RC_SET( FERR_SVR_WRT_FAIL);
				break;

			case 0:
				rc = RC_SET( FERR_SVR_DISCONNECT);
				break;

			default:
				*puiWrtCnt = (FLMUINT)iWrtCnt;
				break;
		}
	}

	if( RC_BAD( rc) && rc != FERR_SVR_WRT_TIMEOUT)
	{
#ifndef FLM_UNIX
		FLMINT iSockErr = WSAGetLastError();
#else
		FLMINT iSockErr = errno;
#endif

#if defined( FLM_WIN) || defined( FLM_NLM)
		if( iSockErr == WSAECONNABORTED)
#else
		if( iSockErr == ECONNABORTED)
#endif
		{
			rc = RC_SET( FERR_SVR_DISCONNECT);
		}
#if defined( FLM_WIN) || defined( FLM_NLM)
		else if( iSockErr == WSAEWOULDBLOCK && iRetryCount < 5)
#else
		else if( iSockErr == EWOULDBLOCK && iRetryCount < 5)
#endif
		{
			iRetryCount++;
			f_sleep( (FLMUINT)(100 * iRetryCount));
			goto Retry;
		}
	}

	return( rc);
}

/********************************************************************
Desc: Reads data from the connection
*********************************************************************/
RCODE FCS_TCP::read(
	FLMBYTE *		pucBuffer,
   FLMUINT			uiDataCnt,
	FLMUINT *		puiReadCnt)
{
	FLMINT		iReadCnt = 0;
	RCODE			rc = FERR_OK;

	flmAssert( m_bConnected && pucBuffer && uiDataCnt);

	if( RC_OK( rc = _SocketPeek( m_uiIOTimeout, TRUE)))
	{
		iReadCnt = (FLMINT)recv( m_iSocket, 
			(char *)pucBuffer, (int)uiDataCnt, 0);
		switch ( iReadCnt)
		{
			case -1:
				iReadCnt = 0;
#if defined( FLM_WIN) || defined( FLM_NLM)
				if ( WSAGetLastError() == WSAECONNRESET)
#else
				if( errno == ECONNRESET)
#endif
				{
					rc = RC_SET( FERR_SVR_DISCONNECT);
				}
				else
				{
					rc = RC_SET( FERR_SVR_READ_FAIL);
				}
				break;

			case 0:
					rc = RC_SET( FERR_SVR_DISCONNECT);
				break;

			default:
				break;
		}
	}

	if( puiReadCnt)
	{
		*puiReadCnt = (FLMUINT)iReadCnt;
	}

	return( rc);
}

/********************************************************************
Desc: Reads data from the connection - Timeout valkue is zero, no error
      is generated if timeout occurs.
*********************************************************************/
RCODE FCS_TCP::readNoWait(
	FLMBYTE *		pucBuffer,
   FLMUINT			uiDataCnt,
	FLMUINT *		puiReadCnt)
{
	FLMINT		iReadCnt = 0;
	RCODE			rc = FERR_OK;

	flmAssert( m_bConnected && pucBuffer && uiDataCnt);

	if( puiReadCnt)
	{
		*puiReadCnt = 0;
	}

	if( RC_OK( rc = _SocketPeek( (FLMUINT)0, TRUE)))
	{
		iReadCnt = recv( m_iSocket, (char *)pucBuffer, (int)uiDataCnt, 0);
		switch ( iReadCnt)
		{
			case -1:
				*puiReadCnt = 0;
#if defined( FLM_WIN) || defined( FLM_NLM)
				if ( WSAGetLastError() == WSAECONNRESET)
#else
				if( errno == ECONNRESET)
#endif
				{
					rc = RC_SET( FERR_SVR_DISCONNECT);
				}
				else
				{
					rc = RC_SET( FERR_SVR_READ_FAIL);
				}
				goto Exit;

			case 0:
				rc = RC_SET( FERR_SVR_DISCONNECT);
				goto Exit;

			default:
				break;
		}
	}
	else if (rc == FERR_SVR_READ_TIMEOUT)
	{
		rc = FERR_OK;
	}

	if( puiReadCnt)
	{
		*puiReadCnt = (FLMUINT)iReadCnt;
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Reads data and does not return until all requested data has
		been read or a timeout error has been encountered.
*********************************************************************/
RCODE FCS_TCP::readAll(
	FLMBYTE *		pucBuffer,
	FLMUINT			uiDataCnt,
   FLMUINT *		puiReadCnt)
{
	FLMUINT		uiToRead = 0;
	FLMUINT		uiHaveRead = 0;
	FLMUINT		uiPartialCnt;
	RCODE			rc = FERR_OK;

	flmAssert( m_bConnected && pucBuffer && uiDataCnt);

	uiToRead = uiDataCnt;
	while( uiToRead)
	{
		if( RC_BAD( rc = read( pucBuffer, uiToRead, &uiPartialCnt)))
		{
			goto Exit;
		}

		pucBuffer += uiPartialCnt;
		uiHaveRead += uiPartialCnt;
		uiToRead = (FLMUINT)(uiDataCnt - uiHaveRead);

		if( puiReadCnt)
		{
			*puiReadCnt = uiHaveRead;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Enables or disables Nagle's algorithm
*********************************************************************/
RCODE FCS_TCP::setTcpDelay(
	FLMBOOL			bOn)
{
	RCODE			rc = FERR_OK;

	int				iOn;

	if( m_iSocket != INVALID_SOCKET)
	{
		iOn = bOn ? 1 : 0;

		if( (setsockopt( m_iSocket, IPPROTO_TCP, TCP_NODELAY, (char *)&iOn,
			(unsigned)sizeof( iOn) )) < 0)
		{
			rc = RC_SET( FERR_SVR_SOCKOPT_FAIL);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_SVR_ALREADY_CLOSED);
	}

Exit:

	return( rc);
}

/********************************************************************
Desc: Closes any open connections
*********************************************************************/
void FCS_TCP::close(
	FLMBOOL			bForce)
{
	if( m_iSocket == INVALID_SOCKET)
	{
		goto Exit;
	}

#ifdef FLM_NLM
	F_UNREFERENCED_PARM( bForce);
#else
	if( !bForce)
	{
		char					ucTmpBuf[ 128];
		struct timeval		tv;
		fd_set				fds;
		fd_set				fds_read;
		fd_set				fds_err;

		// Close our half of the connection

		shutdown( m_iSocket, 1);

		// Set up to wait for readable data on the socket

		FD_ZERO( &fds);
		FD_SET( m_iSocket, &fds);

		tv.tv_sec = 10;
		tv.tv_usec = 0;

		fds_read = fds;
		fds_err = fds;

		// Wait for data or an error

		while( select( m_iSocket + 1, &fds_read, NULL, &fds_err, &tv) > 0)
		{
			if( recv( m_iSocket, ucTmpBuf, sizeof( ucTmpBuf), 0) <= 0)
			{
				break;
			}
			fds_read = fds;
			fds_err = fds;
		}

		shutdown( m_iSocket, 2);
	}
#endif

#ifndef FLM_UNIX
	closesocket( m_iSocket);
#else
	::close( m_iSocket);
#endif

Exit:

	m_iSocket = INVALID_SOCKET;
	m_bConnected = FALSE;
}

/********************************************************************
Desc: Creates a client object
*********************************************************************/
FCS_TCP_CLIENT::FCS_TCP_CLIENT( void) : FCS_TCP()
{
	m_bConnected = FALSE;
}

/********************************************************************
Desc: Closes any connections and frees client resources
*********************************************************************/
FCS_TCP_CLIENT::~FCS_TCP_CLIENT( void )
{
	(void)close();
}

/********************************************************************
Desc: Opens a new connection
*********************************************************************/
RCODE FCS_TCP_CLIENT::openConnection(
	const char  *	pucHostName,
	FLMUINT			uiPort,
	FLMUINT			uiConnectTimeout,
	FLMUINT			uiDataTimeout)
{
	FLMINT					iSockErr;
	FLMINT    				iTries;
	FLMINT					iMaxTries = 5;
	struct sockaddr_in	address;
	struct hostent *		pHostEntry;
	unsigned long			ulIPAddr;
	RCODE						rc = FERR_OK;

	flmAssert( !m_bConnected);
	m_iSocket = INVALID_SOCKET;

	if( pucHostName && pucHostName[ 0] != '\0')
	{
		ulIPAddr = inet_addr( (char *)pucHostName);
		if( ulIPAddr == (unsigned long)INADDR_NONE)
		{
			pHostEntry = gethostbyname( (char *)pucHostName);

			if( !pHostEntry)
			{
				rc = RC_SET( FERR_SVR_NOIP_ADDR);
				goto Exit;
			}
			else
			{
				ulIPAddr = *((unsigned long *)pHostEntry->h_addr);
			}

		}
	}
	else
	{
		ulIPAddr = inet_addr( (char *)"127.0.0.1");
	}

	/******************************************************/
	/* Fill in the Socket structure with family type		*/
	/******************************************************/

	f_memset( (char*)&address, 0, sizeof( struct sockaddr_in));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = (unsigned)ulIPAddr;
	address.sin_port = htons( (unsigned short)uiPort);
	
	/*
	Allocate a socket, then attempt to connect to it!
	*/

	if( (m_iSocket = socket( AF_INET, 
		SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
	{
		rc = RC_SET( FERR_SVR_SOCK_FAIL);
		goto Exit;
	}

	/******************************************************/
	/* Now attempt to connect with the specified 			*/
	/*      partner host, time-out if connection				*/
	/*      doesn't complete within alloted time				*/
	/******************************************************/
#ifdef FLM_WIN
	/*
	**
	*/
	if ( uiConnectTimeout )
	{
		if ( uiConnectTimeout < 5 )
		{
			iMaxTries = (iMaxTries * uiConnectTimeout) / 5;
			uiConnectTimeout = 5;
		}
	}
	else
	{
		iMaxTries = 1;
	}
#endif	

	for( iTries = 0; iTries < iMaxTries; iTries++ )
	{			
		iSockErr = 0;
		if( connect( m_iSocket, (struct sockaddr *)(void *)&address,
			(unsigned)sizeof(struct sockaddr)) >= 0)
		{
			/* SUCCESS! */
			break;
		}

		#ifndef FLM_UNIX
			iSockErr = WSAGetLastError();
		#else
			iSockErr = errno;
		#endif

#ifdef FLM_WIN
		/* 
		In WIN, we sometimes get WSAEINVAL when, if we keep
		trying, we will eventually connect.  Therefore,
		here we'll treat WSAEINVAL as EINPROGRESS.
		*/

		if( iSockErr == WSAEINVAL)
		{
#ifndef FLM_UNIX
			closesocket( m_iSocket);
#else
			::close( m_iSocket);
#endif
			if( (m_iSocket = socket( AF_INET, 
				SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
			{
				rc = RC_SET( FERR_SVR_SOCK_FAIL);
				goto Exit;
			}
#if defined( FLM_WIN) || defined( FLM_NLM)
			iSockErr = WSAEINPROGRESS;
#else
			iSockErr = EINPROGRESS;
#endif
			continue;
		}
#endif

#if defined( FLM_WIN) || defined( FLM_NLM)
		if( iSockErr == WSAEISCONN )
#else
		if( iSockErr == EISCONN )
#endif
		{
			break;
		}
#if defined( FLM_WIN) || defined( FLM_NLM)
		else if( iSockErr == WSAEWOULDBLOCK)
#else
		else if( iSockErr == EWOULDBLOCK)
#endif
		{
			/* 
			** Let's wait a split second to give the connection
         ** request a chance. 
         */

			f_sleep( 100 );
			continue;
		}
#if defined( FLM_WIN) || defined( FLM_NLM)
		else if( iSockErr == WSAEINPROGRESS)
#else
		else if( iSockErr == EINPROGRESS)
#endif
		{
			if( RC_OK( rc = _SocketPeek( uiConnectTimeout, FALSE)))
			{
				/* 
				** Let's wait a split second to give the connection
            ** request a chance. 
            */

				f_sleep( 100 );
				continue;
			}
		}
		rc = RC_SET( FERR_SVR_CONNECT_FAIL);
	}

	if( RC_BAD( rc))
	{
		if( m_iSocket != INVALID_SOCKET)
		{
#ifndef FLM_UNIX
			closesocket( m_iSocket);
#else
			::close( m_iSocket);
#endif
			m_iSocket = INVALID_SOCKET;
		}
		goto Exit;
	}

	m_uiIOTimeout = uiDataTimeout;
	
	setTcpDelay( TRUE);
	m_bConnected = TRUE;

Exit:

	return( rc);
}
	
/********************************************************************
Desc: Constructor
*********************************************************************/
FCS_TCP_SERVER::FCS_TCP_SERVER( void) : FCS_TCP()
{
	m_bBound = FALSE;
}

/********************************************************************
Desc: Destructor
*********************************************************************/
FCS_TCP_SERVER::~FCS_TCP_SERVER( void)
{
	if( m_bBound)
	{
		close( TRUE);
	}
}

/********************************************************************
Desc: Bind to a port prior to listening for connections
*********************************************************************/
RCODE	FCS_TCP_SERVER::bind(
	FLMUINT		uiBindPort,
	FLMBYTE *	pucBindAddr)
{
	struct sockaddr_in 	address;
	RCODE						rc = FERR_OK;

	if( m_bBound)
	{
		rc = RC_SET( FERR_SVR_SOCK_FAIL);
		goto Exit;
	}

	if( (m_iSocket = socket( AF_INET, 
		SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
	{
		rc = RC_SET( FERR_SVR_SOCK_FAIL);
		goto Exit;
	}

	f_memset( &address, 0, sizeof( address));
	address.sin_family = AF_INET;
	if( !pucBindAddr)
	{
		address.sin_addr.s_addr = htonl( INADDR_ANY);
	}
	else
	{
		address.sin_addr.s_addr = inet_addr( (char *)pucBindAddr);
	}
	address.sin_port = htons( (unsigned short)uiBindPort);

	// Bind to the address+port

	if( ::bind( m_iSocket, (struct sockaddr *)(void *)&address, 
		(unsigned)sizeof( address)) != 0)
	{
		rc = RC_SET( FERR_SVR_BIND_FAIL);
		goto Exit;
	}

	/*
	** Bind succeeded, 
	** listen() prepares a socket to accept a connection and specifies a
	** queue limit for incoming connections.  The accept() accepts the connection.
	** Listen returns immediatly.
	
	** Duane: Note for NetWare I spoke with Sravan Vadlakonda in San Jose, 
	** Netware allows 32 not 5 as the max.  We set this high because the 
	** nonpreemptive nature of NLMs means we might not get back to this 
	** thread in time to accept all of the pending connections.  As of
	** Aug 97 the tcpip.nlm displays an error when we don't clean the q
	** of pending connections fast enough.
	*/

#ifdef FLM_NLM
	if( listen( m_iSocket, 32 ) < 0)
#endif
	{
		if( listen( m_iSocket, 5 ) < 0)
		{
			rc = RC_SET( FERR_SVR_LISTEN_FAIL);
			goto Exit;
		}
	}

	/*
	Disable the packet send delay.
	*/

	setTcpDelay( TRUE);
	m_bBound = TRUE;

Exit:

	if( RC_BAD( rc) && m_iSocket != INVALID_SOCKET)
	{
#ifndef FLM_UNIX
		closesocket( m_iSocket);
#else
		::close( m_iSocket);
#endif
		m_iSocket = INVALID_SOCKET;
	}

	return( rc);
}
	
/********************************************************************
Desc: Wait for and accept a client connection
*********************************************************************/
RCODE FCS_TCP_SERVER::connectClient(
	FCS_TCP *	pClient,
	FLMINT		uiConnectTimeout,
	FLMINT		uiDataTimeout)
{
	SOCKET					iSocket;
#if defined( FLM_UNIX)
	socklen_t				iAddrLen;
#else
	int						iAddrLen;
#endif
	struct sockaddr_in 	address;
	RCODE						rc = FERR_OK;

	if( !m_bBound)
	{
		rc = RC_SET( FERR_SVR_BIND_FAIL);
		goto Exit;
	}

	if( RC_BAD( rc = _SocketPeek( uiConnectTimeout, TRUE)))
	{
		goto Exit;
	}

	iAddrLen = (int)sizeof( struct sockaddr);
	if( (iSocket = accept( m_iSocket, 
		(struct sockaddr *)(void *)&address, &iAddrLen)) == INVALID_SOCKET)
	{
		rc = RC_SET( FERR_SVR_ACCEPT_FAIL);
		goto Exit;
	}

	pClient->m_ulRemoteAddr = address.sin_addr.s_addr;
	pClient->m_iSocket = iSocket;
	pClient->m_bConnected = TRUE;
	pClient->m_uiIOTimeout = uiDataTimeout;
	pClient->setTcpDelay( TRUE);

Exit:

	return( rc);
}
