//-------------------------------------------------------------------------
// Desc:	Database restore.
// Tabs:	3
//
//		Copyright (c) 2001-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frestore.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
****************************************************************************/
F_RflUnknownStream::F_RflUnknownStream() 
{
	m_pRfl = NULL;
	m_bStartedWriting = FALSE;
	m_bInputStream = FALSE;
	m_bSetupCalled = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_RflUnknownStream::~F_RflUnknownStream() 
{
	if (m_bSetupCalled)
	{
		(void)close();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_RflUnknownStream::setup(
		F_Rfl *			pRfl,
		FLMBOOL			bInputStream)
{
	RCODE			rc = FERR_OK;

	flmAssert( !m_bSetupCalled);

	if (!pRfl)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}
	m_pRfl = pRfl;
	m_bInputStream = bInputStream;
	m_bSetupCalled = TRUE;
	m_bStartedWriting = FALSE;

Exit:
	return( rc);
}
	
/****************************************************************************
Public: close
****************************************************************************/
RCODE F_RflUnknownStream::close( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);

	// There is nothing to do for input streams, because the RFL
	// code handles skipping over any unknown data that may not have
	// been read yet.
	// For output streams, we need to call the endLoggingUnknown
	// routine so that the last packet gets written out.

	if (!m_bInputStream)
	{
		if (m_bStartedWriting)
		{
			m_bStartedWriting = FALSE;
			if (RC_BAD( rc = m_pRfl->endLoggingUnknown()))
			{
				goto Exit;
			}
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Public: read - will return FERR_EOF_HIT when there is no more data to read.
****************************************************************************/
RCODE F_RflUnknownStream::read(
	FLMUINT			uiLength,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);

	if (!m_bInputStream)
	{

		// Cannot read from an output stream.

		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if (RC_BAD( rc = m_pRfl->readUnknown( uiLength, (FLMBYTE *)pvBuffer,
										puiBytesRead)))
	{
		goto Exit;
	}

Exit:
	return( rc);
}

/****************************************************************************
Public: write
****************************************************************************/
RCODE F_RflUnknownStream::write(
	FLMUINT			uiLength,
	void *			pvBuffer)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( m_pRfl);

	if (m_bInputStream)
	{

		// Cannot write to an input stream.

		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Need to start logging on the first write.

	if (!m_bStartedWriting)
	{
		if (RC_BAD( rc = m_pRfl->startLoggingUnknown()))
		{
			goto Exit;
		}
		m_bStartedWriting = TRUE;
	}

	// Log the data.

	if (RC_BAD( rc = m_pRfl->logUnknown( (FLMBYTE *)pvBuffer, uiLength)))
	{
		goto Exit;
	}
Exit:
	return( rc);
}

/*API~***********************************************************************
Desc:	Returns an unknown stream object - suitable for writing unknown
		streams into the roll-forward log.
*END************************************************************************/
FLMEXP RCODE FLMAPI FlmDbGetUnknownStreamObj(
	HFDB						hDb,
	F_UnknownStream **	ppUnknownStream)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = (FDB *)hDb;
	F_RflUnknownStream *	pUnkStream = NULL;

	flmAssert( pDb);
	flmAssert( ppUnknownStream);

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// This is only valid on 4.3 and greater.

	if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
	{
		goto Exit;	// Will return FERR_OK and a NULL pointer.
	}

	// Must be in an update transaction.

	if (pDb->uiTransType == FLM_NO_TRANS)
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}
	if (pDb->uiTransType != FLM_UPDATE_TRANS)
	{
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	// Allocate the stream object we want.

	if ((pUnkStream = f_new F_RflUnknownStream) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Setup the unknown stream object.

	if (RC_BAD( rc = pUnkStream->setup( pDb->pFile->pRfl, FALSE)))
	{
		goto Exit;
	}

Exit:
	if (RC_BAD( rc) && pUnkStream)
	{
		pUnkStream->Release();
		pUnkStream = NULL;
	}
	*ppUnknownStream = (F_UnknownStream *)pUnkStream;
	return( rc);
}

/*
*** R_RestoreFS methods
*/

/****************************************************************************
Public: Constructor
****************************************************************************/
F_FSRestore::F_FSRestore() 
{
	m_pFileHdl = NULL;
	m_pFileHdl64 = NULL;
	m_ui64Offset = 0;
	m_bSetupCalled = FALSE;
	m_szDbPath[ 0] = 0;
	m_uiDbVersion = 0;
	m_szBackupSetPath[ 0] = 0;
	m_szRflDir[ 0] = 0;
	m_bOpen = FALSE;
}

/****************************************************************************
Public: Destructor
****************************************************************************/
F_FSRestore::~F_FSRestore() 
{
	if( m_bOpen)
	{
		(void)close();
	}
}

/****************************************************************************
Public: setup
****************************************************************************/
RCODE F_FSRestore::setup(
	const char *		pucDbPath,
	const char *		pucBackupSetPath,
	const char *		pucRflDir)
{
	flmAssert( !m_bSetupCalled);
	flmAssert( pucDbPath);
	flmAssert( pucBackupSetPath);

	f_strcpy( m_szDbPath, pucDbPath);
	f_strcpy( m_szBackupSetPath, pucBackupSetPath);

	if( pucRflDir)
	{
		f_strcpy( m_szRflDir, pucRflDir);
	}
	

	m_bSetupCalled = TRUE;
	return( FERR_OK);
}

/****************************************************************************
Public: openBackupSet
****************************************************************************/
RCODE F_FSRestore::openBackupSet( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( !m_pFileHdl64);

	if( (m_pFileHdl64 = f_new F_64BitFileHandle) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pFileHdl64->Open( m_szBackupSetPath)))
	{
		m_pFileHdl64->Release();
		m_pFileHdl64 = NULL;
		goto Exit;
	}

	m_ui64Offset = 0;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Public: openRflFile
****************************************************************************/
RCODE F_FSRestore::openRflFile(
	FLMUINT			uiFileNum)
{
	RCODE				rc = FERR_OK;
	char				szRflPath[ F_PATH_MAX_SIZE];
	char				szDbPrefix[ F_FILENAME_SIZE];
	char				szBaseName[ F_FILENAME_SIZE];
	FLMBYTE *		pBuf = NULL;
	FILE_HDR			fileHdr;
	LOG_HDR			logHdr;
	F_FileHdl *		pFileHdl = NULL;

	flmAssert( m_bSetupCalled);
	flmAssert( uiFileNum);
	flmAssert( !m_pFileHdl);

	/*
	Read the database header to determine the version number
	*/
	
	if( !m_uiDbVersion)
	{
		if (RC_BAD( rc = f_alloc( 2048, &pBuf)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Open( 
			m_szDbPath, F_IO_RDWR | F_IO_SH_DENYNONE | F_IO_DIRECT,
			&pFileHdl)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = flmReadAndVerifyHdrInfo( NULL, pFileHdl,
			pBuf, &fileHdr, &logHdr, NULL)))
		{
			goto Exit;
		}

		pFileHdl->Close();
		pFileHdl->Release();
		pFileHdl = NULL;

		m_uiDbVersion = fileHdr.uiVersionNum;
	}

	/*
	Generate the log file name.
	*/

	if( RC_BAD( rc = rflGetDirAndPrefix( 
		m_uiDbVersion, m_szDbPath, m_szRflDir, szRflPath, szDbPrefix)))
	{
		goto Exit;
	}

	rflGetBaseFileName( m_uiDbVersion, szDbPrefix, uiFileNum, szBaseName);
	f_pathAppend( szRflPath, szBaseName);

	/* 
	Open the file.
	*/

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->OpenBlockFile( 
		szRflPath, F_IO_RDWR | F_IO_SH_DENYNONE | F_IO_DIRECT,
		512, &m_pFileHdl)))
	{
		goto Exit;
	}

	m_bOpen = TRUE;
	m_ui64Offset = 0;

Exit:

	if( pBuf)
	{
		f_free( &pBuf);
	}

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Public: openIncFile
****************************************************************************/
RCODE F_FSRestore::openIncFile(
	FLMUINT			uiFileNum)
{
	RCODE			rc = FERR_OK;
	char			szIncPath[ F_PATH_MAX_SIZE];
	char			szIncFile[ F_FILENAME_SIZE];

	flmAssert( m_bSetupCalled);
	flmAssert( !m_pFileHdl64);

	/*
	Since this is a non-interactive restore, we will "guess"
	that incremental backups are located in the same parent
	directory as the main backup set.  We will further assume
	that the incremental backup sets have been named XXXXXXXX.INC,
	where X is a hex digit.
	*/

	if( RC_BAD( rc = f_pathReduce( m_szBackupSetPath, 
		szIncPath, NULL)))
	{
		goto Exit;
	}

	f_sprintf( szIncFile, "%08X.INC", (unsigned)uiFileNum);
	f_pathAppend( szIncPath, szIncFile);

	if( (m_pFileHdl64 = f_new F_64BitFileHandle) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pFileHdl64->Open( szIncPath)))
	{
		m_pFileHdl64->Release();
		m_pFileHdl64 = NULL;
		goto Exit;
	}

	m_ui64Offset = 0;
	m_bOpen = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Public: read
****************************************************************************/
RCODE F_FSRestore::read(
	FLMUINT			uiLength,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	FLMUINT		uiBytesRead = 0;
	RCODE			rc = FERR_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( m_pFileHdl || m_pFileHdl64);

	if( m_pFileHdl64)
	{
		if( RC_BAD( rc = m_pFileHdl64->Read( m_ui64Offset, 
			uiLength, pvBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = m_pFileHdl->Read( (FLMUINT)m_ui64Offset,
			uiLength, pvBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
	}

Exit:

	m_ui64Offset += uiBytesRead;

	if( puiBytesRead)
	{
		*puiBytesRead = uiBytesRead;
	}

	return( rc);
}

/****************************************************************************
Public: close
****************************************************************************/
RCODE F_FSRestore::close( void)
{
	flmAssert( m_bSetupCalled);

	if( m_pFileHdl64)
	{
		m_pFileHdl64->Release();
		m_pFileHdl64 = NULL;
	}

	if( m_pFileHdl)
	{
		m_pFileHdl->Release();
		m_pFileHdl = NULL;
	}

	m_bOpen = FALSE;
	m_ui64Offset = 0;

	return( FERR_OK);
}
