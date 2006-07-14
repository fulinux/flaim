//------------------------------------------------------------------------------
// Desc:	This include file contains the methods for the super file class.
//
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
// $Id: $
//------------------------------------------------------------------------------

#include "ftksys.h"

/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileHdl::F_SuperFileHdl( void)
{
	m_pSuperFileClient = NULL;
	m_pCFileHdl = NULL;
	m_pBlockFileHdl = NULL;
	m_uiBlockFileNum = 0;
	m_bBlockFileDirty = FALSE;
	m_bCFileDirty = FALSE;
	m_uiBlockSize = 0;
	m_uiExtendSize = (8 * 1024 * 1024);
	m_uiMaxAutoExtendSize = f_getMaxFileSize();
	m_uiFileOpenFlags = 0;
	m_uiFileCreateFlags = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileHdl::~F_SuperFileHdl()
{
	if( m_pCFileHdl)
	{
		if( m_bCFileDirty)
		{
			f_assert( 0);
			m_pCFileHdl->flush();
		}
		
		m_pCFileHdl->Release();
	}
	
	if( m_pBlockFileHdl)
	{
		if( m_bBlockFileDirty)
		{
			m_pBlockFileHdl->flush();
		}
		
		m_pBlockFileHdl->Release();
	}
	
	if( m_pSuperFileClient)
	{
		m_pSuperFileClient->Release();
	}
	
	if( m_pFileHdlCache)
	{
		m_pFileHdlCache->Release();
	}
}

/****************************************************************************
Desc:	Configures the super file object
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::setup(
	IF_SuperFileClient *		pSuperFileClient,
	IF_FileHdlCache *			pFileHdlCache,
	FLMUINT						uiFileOpenFlags,
	FLMUINT						uiFileCreateFlags)
{
	RCODE			rc = NE_FLM_OK;

	f_assert( !m_pSuperFileClient);
	
	m_pSuperFileClient = pSuperFileClient;
	m_pSuperFileClient->AddRef();

	if( (m_pFileHdlCache = pFileHdlCache) != NULL)
	{
		m_pFileHdlCache->AddRef();
	}
	else
	{
		if( RC_BAD( rc = f_getFileSysPtr()->allocFileHandleCache( 
			8, 120, &m_pFileHdlCache)))
		{
			goto Exit;
		}
	}
	
	m_uiFileOpenFlags = uiFileOpenFlags;
	m_uiFileCreateFlags = uiFileCreateFlags;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc: Creates a file
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::createFile(
	FLMUINT			uiFileNumber,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	char				szFilePath[ F_PATH_MAX_SIZE];
	IF_FileHdl *	pFileHdl = NULL;

	f_assert( m_uiBlockSize);
	
	// If the file creation flags are not set we won't allow this operation
	// to continue
	
	if( !m_uiFileCreateFlags)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_ILLEGAL_OP);
		goto Exit;
	}
	
	// See if we already have an open file handle (or if we can open the file).
	// If so, truncate the file and use it.

	if( RC_BAD( rc = getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		if( rc != NE_FLM_IO_PATH_NOT_FOUND)
		{
			goto Exit;
		}
		
		rc = NE_FLM_OK;
	}
	
	if( !pFileHdl)
	{
		if( RC_BAD( rc = m_pSuperFileClient->getFilePath( 
			uiFileNumber, szFilePath)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pFileHdlCache->createFile( szFilePath,
			m_uiFileCreateFlags, &pFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pFileHdl->truncate()))
		{
			goto Exit;
		}
	}
	
	if( ppFileHdl)
	{
		*ppFileHdl = pFileHdl;
		pFileHdl = NULL;
	}
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Reads a database block into a buffer
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::readBlock(
	FLMUINT			uiBlkAddress,
	FLMUINT			uiBytesToRead,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	f_assert( m_uiBlockSize);

	if( RC_BAD( rc = getFileHdl(
		m_pSuperFileClient->getFileNumber( uiBlkAddress), FALSE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->read(
		m_pSuperFileClient->getFileOffset( uiBlkAddress), uiBytesToRead,
		pvBuffer, puiBytesRead)))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: Writes a block to the database
****************************************************************************/
RCODE F_SuperFileHdl::writeBlock(
	FLMUINT				uiBlkAddress,
	FLMUINT				uiBytesToWrite,
	IF_IOBuffer *		pIOBuffer)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	f_assert( m_uiBlockSize);

	if( RC_BAD( rc = getFileHdl(
		m_pSuperFileClient->getFileNumber( uiBlkAddress), TRUE, &pFileHdl)))
	{
		if( rc != NE_FLM_IO_PATH_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = createFile( 
			m_pSuperFileClient->getFileNumber( uiBlkAddress), &pFileHdl)))
		{
			goto Exit;
		}
	}

	pFileHdl->setExtendSize( m_uiExtendSize);
	pFileHdl->setMaxAutoExtendSize( m_uiMaxAutoExtendSize);
	
	rc = pFileHdl->write( m_pSuperFileClient->getFileOffset( uiBlkAddress),
		uiBytesToWrite, pIOBuffer);

	pIOBuffer = NULL;		

	if( RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pIOBuffer)
	{
		f_assert( RC_BAD( rc));
		pIOBuffer->notifyComplete( rc);
	}

	return( rc);
}

/****************************************************************************
Desc: Writes a block to the database
****************************************************************************/
RCODE F_SuperFileHdl::writeBlock(
	FLMUINT				uiBlkAddress,
	FLMUINT				uiBytesToWrite,
	const void *		pvBuffer,
	FLMUINT *			puiBytesWritten)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	f_assert( m_uiBlockSize);

	if( RC_BAD( rc = getFileHdl(
		m_pSuperFileClient->getFileNumber( uiBlkAddress), TRUE, &pFileHdl)))
	{
		if( rc != NE_FLM_IO_PATH_NOT_FOUND)
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = createFile( 
			m_pSuperFileClient->getFileNumber( uiBlkAddress), &pFileHdl)))
		{
			goto Exit;
		}
	}

	pFileHdl->setExtendSize( m_uiExtendSize);
	pFileHdl->setMaxAutoExtendSize( m_uiMaxAutoExtendSize);
	
	if( RC_BAD( rc = pFileHdl->write( 
		m_pSuperFileClient->getFileOffset( uiBlkAddress), uiBytesToWrite,
		pvBuffer, puiBytesWritten)))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Flush dirty files to disk.
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::flush( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( m_pCFileHdl && m_bCFileDirty)
	{
		if( RC_BAD( rc = m_pCFileHdl->flush()))
		{
			goto Exit;
		}
		
		m_bCFileDirty = FALSE;
	}
	
	if( m_pBlockFileHdl && m_bBlockFileDirty)
	{
		if( RC_BAD( rc = m_pBlockFileHdl->flush()))
		{
			goto Exit;
		}
		
		m_bBlockFileDirty = FALSE;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Truncates back to an end of file block address.
****************************************************************************/
RCODE	FLMAPI F_SuperFileHdl::truncateFile(
	FLMUINT			uiEOFBlkAddress)
{
	RCODE 			rc = NE_FLM_OK;
	FLMUINT			uiFileNumber = m_pSuperFileClient->getFileNumber( uiEOFBlkAddress);
	FLMUINT			uiBlockOffset = m_pSuperFileClient->getFileOffset( uiEOFBlkAddress);
	IF_FileHdl *	pFileHdl = NULL;

	// Truncate the current block file.

	if( RC_BAD( rc = getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->truncate( uiBlockOffset)))
	{
		goto Exit;
	}

	// Truncate all of the block files beyon the end-of-file address

	for( ;;)
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		
		if( RC_BAD( getFileHdl( ++uiFileNumber, TRUE, &pFileHdl)))
		{
			break;
		}

		if( RC_BAD( rc = pFileHdl->truncate()))
		{
			goto Exit;
		}
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Truncates back to an end of file block address.
****************************************************************************/
RCODE	FLMAPI F_SuperFileHdl::truncateFile(
	FLMUINT			uiFileNumber,
	FLMUINT			uiOffset)
{
	RCODE 			rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	if( RC_BAD( rc = getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->truncate( uiOffset)))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Truncate to zero length any files between the specified start
		and end files.
****************************************************************************/
void FLMAPI F_SuperFileHdl::truncateFiles(
	FLMUINT			uiStartFileNum,
	FLMUINT			uiEndFileNum)
{
	FLMUINT			uiFileNumber;
	IF_FileHdl *	pFileHdl = NULL;

	for( uiFileNumber = uiStartFileNum; 
		  uiFileNumber <= uiEndFileNum; 
		  uiFileNumber++)
	{
		if( RC_OK( getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
		{
			pFileHdl->truncate();
			pFileHdl->Release();
		}
	}
}

/****************************************************************************
Desc:	Returns the physical size of a file
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::getFileSize(
	FLMUINT			uiFileNumber,
	FLMUINT64 *		pui64FileSize)
{
	RCODE 			rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	*pui64FileSize = 0;

	if( RC_BAD( rc = getFileHdl( uiFileNumber, FALSE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->size( pui64FileSize)))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the path of a file given its file number
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::getFilePath(
	FLMUINT			uiFileNumber,
	char *			pszIoPath)
{
	return( m_pSuperFileClient->getFilePath( uiFileNumber, pszIoPath));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FLMAPI F_SuperFileHdl::canDoAsync( void)
{
	FLMBOOL		bCanDoAsync = FALSE;
	
	if( m_pCFileHdl)
	{
		bCanDoAsync = m_pCFileHdl->canDoAsync();
	}
	else
	{
		IF_FileHdl *		pFileHdl = NULL;
		
		if( RC_OK( getFileHdl( 0, FALSE, &pFileHdl)))
		{
			bCanDoAsync = pFileHdl->canDoAsync();
			pFileHdl->Release();
		}
	}
	
	return( bCanDoAsync);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::releaseFiles( void)
{
	RCODE			rc = NE_FLM_OK;
	
	if( RC_BAD( rc = flush()))
	{
		goto Exit;
	}
	
	if( m_pCFileHdl)
	{
		m_pCFileHdl->Release();
		m_pCFileHdl = NULL;
	}
	
	if( m_pBlockFileHdl)
	{
		m_pBlockFileHdl->Release();
		m_pBlockFileHdl = NULL;
		m_uiBlockFileNum = 0;
	}

	m_pFileHdlCache->closeUnusedFiles();
	
Exit:

	return( rc);
}
	
/****************************************************************************
Desc:	Returns a file handle given the file's number
****************************************************************************/
RCODE FLMAPI F_SuperFileHdl::getFileHdl(
	FLMUINT						uiFileNum,
	FLMBOOL						bForUpdate,
	IF_FileHdl **				ppFileHdl)
{
	RCODE							rc = NE_FLM_OK;
	IF_FileHdl *				pFileHdl = NULL;
	char							szFilePath[ F_PATH_MAX_SIZE];
	
	f_assert( *ppFileHdl == NULL);
	
	if( !uiFileNum)
	{
		if( !m_pCFileHdl)
		{
			if( RC_BAD( rc = m_pSuperFileClient->getFilePath(  
				uiFileNum, szFilePath)))
			{
				goto Exit;
			}
		
			if( RC_BAD( rc = m_pFileHdlCache->openFile( szFilePath,
				m_uiFileOpenFlags, &pFileHdl)))
			{
				goto Exit;
			}

			m_pCFileHdl = pFileHdl;
			m_pCFileHdl->AddRef();
		}
		else
		{
			pFileHdl = m_pCFileHdl;
			pFileHdl->AddRef();
		}
		
		if( bForUpdate)
		{
			m_bCFileDirty = TRUE;
		}
	}
	else
	{
		if( m_pBlockFileHdl)
		{
			if( m_uiBlockFileNum != uiFileNum)
			{
				if( m_bBlockFileDirty)
				{
					m_pBlockFileHdl->flush();
					m_bBlockFileDirty = FALSE;
				}
				
				m_pBlockFileHdl->Release();
				m_pBlockFileHdl = NULL;
				m_uiBlockFileNum = 0;
			}
		}
		
		if( !m_pBlockFileHdl)
		{
			if( RC_BAD( rc = m_pSuperFileClient->getFilePath(  
				uiFileNum, szFilePath)))
			{
				goto Exit;
			}
		
			if( RC_BAD( rc = m_pFileHdlCache->openFile( szFilePath,
				m_uiFileOpenFlags, &pFileHdl)))
			{
				goto Exit;
			}

			m_uiBlockFileNum = uiFileNum;
			m_pBlockFileHdl = pFileHdl;
			m_pBlockFileHdl->AddRef();
		}
		else
		{
			pFileHdl = m_pBlockFileHdl;
			pFileHdl->AddRef();
		}
		
		if( bForUpdate)
		{
			m_bBlockFileDirty = TRUE;
		}
	}

	*ppFileHdl = pFileHdl;
	pFileHdl = NULL;

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}
