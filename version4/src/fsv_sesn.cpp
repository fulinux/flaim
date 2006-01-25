//-------------------------------------------------------------------------
// Desc:	Server session class.
// Tabs:	3
//
//		Copyright (c) 1998-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_sesn.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
*****************************************************************************/
FSV_SESN::FSV_SESN( void)
{
	m_pServerContext = NULL;
	m_hDb = HFDB_NULL;
	m_uiSessionId = FCS_INVALID_ID;
	m_uiCookie = 0;
	m_uiFlags = 0;
	m_pBIStream = NULL;
	m_pBOStream = NULL;
	m_bSetupCalled = FALSE;
	m_uiClientProtocolVersion = 0;
	GedPoolInit( &m_wireScratchPool, 2048);
}


/****************************************************************************
Desc:
*****************************************************************************/
FSV_SESN::~FSV_SESN( void)
{
	FLMUINT			uiLoop;

	if( m_bSetupCalled)
	{
		/*
		Free iterator resources.
		*/
		
		for( uiLoop = 0; uiLoop < MAX_SESN_ITERATORS; uiLoop++)
		{
			if( m_IteratorList[ uiLoop] != HFCURSOR_NULL)
			{
				(void)FlmCursorFree( &m_IteratorList[ uiLoop]);
			}
		}

		/*
		Close the database
		*/

		if( m_hDb != HFDB_NULL)
		{
			(void)FlmDbClose( &m_hDb);
		}

		/*
		Free the buffer streams
		*/

		if( m_pBIStream)
		{
			m_pBIStream->Release();
		}

		if( m_pBOStream)
		{
			m_pBOStream->Release();
		}
	}

	GedPoolFree( &m_wireScratchPool);
}


/****************************************************************************
Desc:	Configures and initializes the session.
*****************************************************************************/
RCODE FSV_SESN::Setup(
	FSV_SCTX *	pServerContext,
	FLMUINT		uiVersion,
	FLMUINT		uiFlags)
{
	FLMUINT	uiLoop;
	RCODE		rc = FERR_OK;

	/*
	Make sure that setup has not been called.
	*/
	
	flmAssert( m_bSetupCalled == FALSE);

	/*
	Verify that the requested version is supported.
	*/

	if( uiVersion > FCS_VERSION_1_1_1)
	{
		rc = RC_SET( FERR_UNSUPPORTED_VERSION);
		goto Exit;
	}
	m_uiClientProtocolVersion = uiVersion;
	

	/*
	Set the server context.
	*/

	m_pServerContext = pServerContext;

	/*
	Initialize the iterator list
	*/

	for( uiLoop = 0; uiLoop < MAX_SESN_ITERATORS; uiLoop++)
	{
		m_IteratorList[ uiLoop] = HFCURSOR_NULL;
	}

	m_bSetupCalled = TRUE;
	m_uiFlags = uiFlags;

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Opens the requested database.
*****************************************************************************/
RCODE FSV_SESN::OpenDatabase(
	FLMUNICODE *	puzDbPath,
	FLMUNICODE *	puzDataDir,
	FLMUNICODE *	puzRflDir,
	FLMUINT			uiOpenFlags)
{
	RCODE			rc = FERR_OK;
	char *		pszDbPath = NULL;
	char *		pszDataDir;
	char *		pszRflDir;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_hDb == HFDB_NULL);

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE * 3, &pszDbPath)))
	{
		goto Exit;
	}
	pszDataDir = pszDbPath + F_PATH_MAX_SIZE;
	pszRflDir = pszDataDir + F_PATH_MAX_SIZE;

	/*
	Perform some sanity checking.
	*/

	if( !puzDbPath)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	/*
	Convert the UNICODE URL to a server path.
	*/

	if( RC_BAD( rc = m_pServerContext->BuildFilePath( 
		puzDbPath, pszDbPath)))
	{
		goto Exit;
	}

	/*
	Convert the data directory
	*/

	if( puzDataDir)
	{
		if( RC_BAD( rc = m_pServerContext->BuildFilePath( 
			puzDataDir, pszDataDir)))
		{
			goto Exit;
		}
	}
	else
	{
		pszDataDir = NULL;
	}
	
	/*
	Convert the RFL path
	*/

	if( puzRflDir)
	{
		if( RC_BAD( rc = m_pServerContext->BuildFilePath( 
			puzRflDir, pszRflDir)))
		{
			goto Exit;
		}
	}
	else
	{
		*pszRflDir = 0;
	}
	
	/*
	Open the database.
	*/

	if( RC_BAD( rc = FlmDbOpen( pszDbPath, pszDataDir, pszRflDir,
		uiOpenFlags, NULL, &m_hDb)))
	{
		goto Exit;
	}

Exit:

	if (pszDbPath)
	{
		f_free( &pszDbPath);
	}

	/*
	Free resources
	*/
	
	if( RC_BAD( rc))
	{
		if( m_hDb != HFDB_NULL)
		{
			(void)FlmDbClose( &m_hDb);
		}
	}

	return( rc);
}


/****************************************************************************
Desc:	Creates a new database.
*****************************************************************************/
RCODE FSV_SESN::CreateDatabase(
	FLMUNICODE *	puzDbPath,
	FLMUNICODE *	puzDataDir,
	FLMUNICODE *	puzRflDir,
	FLMUNICODE *	puzDictPath,
	FLMUNICODE *	puzDictBuf,
	CREATE_OPTS *	pCreateOpts)
{
	RCODE			rc = FERR_OK;
	POOL			tmpPool;
	char *		pucDictBuf = NULL;
	char *		pszDbPath = NULL;
	char *		pszDataDir;
	char *		pszRflDir;
	char *		pszDictPath;

	/*
	Initialize a temporary pool.
	*/

	GedPoolInit( &tmpPool, 1024);
	
	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);
	flmAssert( m_hDb == HFDB_NULL);

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE * 4, &pszDbPath)))
	{
		goto Exit;
	}
	pszDataDir = pszDbPath + F_PATH_MAX_SIZE;
	pszRflDir = pszDataDir + F_PATH_MAX_SIZE;
	pszDictPath = pszRflDir + F_PATH_MAX_SIZE;

	/*
	Perform some sanity checking.
	*/

	if( !puzDbPath)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	/*
	Convert the DB URL to a server path.
	*/

	if( RC_BAD( rc = m_pServerContext->BuildFilePath( puzDbPath, pszDbPath)))
	{
		goto Exit;
	}
	
	/*
	Convert the dictionary URL to a server path.
	*/

	if( puzDictPath)
	{
		if( RC_BAD( rc =
			m_pServerContext->BuildFilePath( puzDictPath, pszDictPath)))
		{
			goto Exit;
		}
	}
	else
	{
		pszDictPath = NULL;
	}

	/*
	Convert the data directory
	*/

	if( puzDataDir)
	{
		if( RC_BAD( rc = m_pServerContext->BuildFilePath( 
			puzDataDir, pszDataDir)))
		{
			goto Exit;
		}
	}
	else
	{
		pszDataDir = NULL;
	}

	/*
	Convert the RFL path
	*/

	if( puzRflDir)
	{
		if( RC_BAD( rc = m_pServerContext->BuildFilePath( 
			puzRflDir, pszRflDir)))
		{
			goto Exit;
		}
	}
	else
	{
		*pszRflDir = 0;
	}

	/*
	Attempt to convert the UNICODE dictionary buffer to a native string
	*/

	if( puzDictBuf)
	{
		if( RC_BAD( rc = fcsConvertUnicodeToNative( &tmpPool,
			puzDictBuf, &pucDictBuf)))
		{
			goto Exit;
		}
	}

	/*
	Create the database.
	*/

	if( RC_BAD( rc = FlmDbCreate( pszDbPath, pszDataDir, pszRflDir,
		pszDictPath, pucDictBuf, pCreateOpts, &m_hDb)))
	{
		goto Exit;
	}

Exit:

	if (pszDbPath)
	{
		f_free( &pszDbPath);
	}

	/*
	Free resources
	*/

	if( RC_BAD( rc))
	{
		if( m_hDb != HFDB_NULL)
		{
			(void)FlmDbClose( &m_hDb);
		}
	}

	GedPoolFree( &tmpPool);
	return( rc);
}


/****************************************************************************
Desc:	Closes the database.
*****************************************************************************/
RCODE	FSV_SESN::CloseDatabase( void)
{
	RCODE				rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Close the database.
	*/

	if( m_hDb != HFDB_NULL)
	{
		if( RC_BAD( rc = FlmDbClose( &m_hDb)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Creates a new iterator and adds it to the session's iterator list.
*****************************************************************************/
RCODE FSV_SESN::InitializeIterator(
	FLMUINT *	puiIteratorIdRV,
	HFDB			hDb,
	FLMUINT		uiContainer,
	HFCURSOR *	phIteratorRV)
{
	HFCURSOR		hIterator = HFCURSOR_NULL;
	FLMUINT		uiSlot;
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Set the iterator id.
	*/

	*puiIteratorIdRV = FCS_INVALID_ID;

	/*
	Find a slot in the session's iterator table
	*/

	for( uiSlot = 0; uiSlot < MAX_SESN_ITERATORS; uiSlot++)
	{
		if( m_IteratorList[ uiSlot] == HFCURSOR_NULL)
		{
			break;
		}
	}

	/*
	Too many open iterators
	*/

	if( uiSlot == MAX_SESN_ITERATORS)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	/*
	Initialize a new iterator (cursor).
	*/
	
	if( RC_BAD( rc = FlmCursorInit( hDb, uiContainer, &hIterator)))
	{
		goto Exit;
	}

	/*
	Add the iterator to the iterator list.
	*/

	m_IteratorList[ uiSlot] = hIterator;
	*puiIteratorIdRV = uiSlot;

Exit:

	/*
	Free resources
	*/

	if( RC_BAD( rc))
	{
		if( hIterator != HFCURSOR_NULL)
		{
			(void)FlmCursorFree( &hIterator);
		}
	}
	else
	{
		if( phIteratorRV)
		{
			*phIteratorRV = hIterator;
		}
	}

	return( rc);
}


/****************************************************************************
Desc:	Frees the specified iterator and removes it from the session's
		iterator list.
*****************************************************************************/
RCODE FSV_SESN::FreeIterator(
	FLMUINT		uiIteratorId)
{
	HFCURSOR		hIterator = HFCURSOR_NULL;
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	/*
	Find the iterator in the resource bag and remove it.
	*/

	if( uiIteratorId >= MAX_SESN_ITERATORS ||
		m_IteratorList[ uiIteratorId] == HFCURSOR_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	hIterator = m_IteratorList[ uiIteratorId];
	m_IteratorList[ uiIteratorId] = HFCURSOR_NULL;
	
	/*
	Free the iterator.
	*/

	if( RC_BAD( rc = FlmCursorFree( &hIterator)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Returns the specified iterator's handle
*****************************************************************************/
RCODE	FSV_SESN::GetIterator(
	FLMUINT		uiIteratorId,
	HFCURSOR *	phIteratorRV)
{
	RCODE			rc = FERR_OK;

	/*
	Make sure that setup has been called.
	*/
	
	flmAssert( m_bSetupCalled == TRUE);

	if( uiIteratorId >= MAX_SESN_ITERATORS ||
		m_IteratorList[ uiIteratorId] == HFCURSOR_NULL)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	*phIteratorRV = m_IteratorList[ uiIteratorId];

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Returns a pointer to the buffer input stream
*****************************************************************************/
RCODE FSV_SESN::GetBIStream(
	FCS_BIOS **		ppBIStream)
{
	RCODE		rc = FERR_OK;

	*ppBIStream = NULL;

	if( !m_pBIStream)
	{
		m_pBIStream = f_new FCS_BIOS;
		if( !m_pBIStream)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	*ppBIStream = m_pBIStream;

Exit:

	return( rc);
}
	

/****************************************************************************
Desc:	Returns a pointer to the buffer output stream
*****************************************************************************/
RCODE FSV_SESN::GetBOStream(
	FCS_BIOS **		ppBOStream)
{
	RCODE		rc = FERR_OK;

	*ppBOStream = NULL;

	if( !m_pBOStream)
	{
		m_pBOStream = f_new FCS_BIOS;
		if( !m_pBOStream)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	*ppBOStream = m_pBOStream;

Exit:

	return( rc);
}
