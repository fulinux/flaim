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
	f_memset( &m_CheckedOutFileHdls[ 0], 0, sizeof( m_CheckedOutFileHdls));
	m_pCheckedOutFileHdls = &m_CheckedOutFileHdls [0];
	m_uiCkoArraySize = MAX_CHECKED_OUT_FILE_HDLS + 1;
	m_uiBlockSize = 0;
	m_uiExtendSize = (8 * 1024 * 1024);
	m_uiMaxAutoExtendSize = f_getMaxFileSize();
	m_uiLowestDirtySlot = 1;
	m_uiHighestDirtySlot = 0;
	m_uiHighestUsedSlot = 0;
	m_uiHighestFileNumber = 0;
	m_bMinimizeFlushes = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileHdl::~F_SuperFileHdl()
{
	releaseFiles( TRUE);
	
	if( m_pSuperFileClient)
	{
		m_pSuperFileClient->Release();
	}
}

/****************************************************************************
Desc:	Configures the super file object
****************************************************************************/
RCODE F_SuperFileHdl::setup(
	IF_SuperFileClient *		pSuperFileClient)
{
	f_assert( !m_pSuperFileClient);
	
	m_pSuperFileClient = pSuperFileClient;
	m_pSuperFileClient->AddRef();
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc: Creates a file
****************************************************************************/
RCODE F_SuperFileHdl::createFile(
	FLMUINT			uiFileNumber)
{
	RCODE				rc = NE_FLM_OK;
	char				szFilePath[ F_PATH_MAX_SIZE];
	IF_FileHdl *	pFileHdl = NULL;

	// Sanity checks

	flmAssert( m_uiBlockSize);

	// See if we already have an open file handle (or if we can open the file).
	// If so, truncate the file and use it.

	if( RC_OK( rc = getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		rc = pFileHdl->truncate( 0);
		pFileHdl = NULL;
		goto Exit;
	}
	else if( rc != NE_FLM_IO_PATH_NOT_FOUND)
	{
		goto Exit;
	}

	// Build the file path

	if( RC_BAD( rc = m_pSuperFileClient->getFilePath( uiFileNumber, szFilePath)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_getFileSysPtr()->createFile( szFilePath,
		FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_DIRECT | FLM_IO_SH_DENYNONE,
		&pFileHdl)))
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
Desc:	Reads a database block into a buffer
****************************************************************************/
RCODE F_SuperFileHdl::readBlock(
	FLMUINT			uiBlkAddress,
	FLMUINT			uiBytesToRead,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	flmAssert( m_uiBlockSize);

	if( RC_BAD( rc = getFileHdl(
		m_pSuperFileClient->getFileNumber( uiBlkAddress), FALSE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->sectorRead(
		m_pSuperFileClient->getFileOffset( uiBlkAddress), uiBytesToRead,
		pvBuffer, puiBytesRead)))
	{
		if (rc != NE_FLM_IO_END_OF_FILE && rc != NE_FLM_MEM)
		{
			releaseFile( m_pSuperFileClient->getFileNumber( uiBlkAddress), TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Writes a block to the database
****************************************************************************/
RCODE F_SuperFileHdl::writeBlock(
	FLMUINT			uiBlkAddress,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	IF_IOBuffer *	pIOBuffer,
	FLMUINT *		puiBytesWritten)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	flmAssert( m_uiBlockSize);

Get_Handle:

	if( RC_BAD( rc = getFileHdl(
		m_pSuperFileClient->getFileNumber(uiBlkAddress), TRUE, &pFileHdl)))
	{
		if (rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			if (RC_BAD( rc = createFile( m_pSuperFileClient->getFileNumber( 
					uiBlkAddress))))
			{
				goto Exit;
			}
			else
			{
				goto Get_Handle;
			}
		}
		goto Exit;
	}

	pFileHdl->setExtendSize( m_uiExtendSize);
	pFileHdl->setMaxAutoExtendSize( m_uiMaxAutoExtendSize);
	if( RC_BAD( rc = pFileHdl->sectorWrite(
		m_pSuperFileClient->getFileOffset( uiBlkAddress), uiBytesToWrite,
		pvBuffer, pIOBuffer, puiBytesWritten)))
	{
		if (rc != NE_FLM_IO_DISK_FULL && rc != NE_FLM_MEM)
		{
			releaseFile( m_pSuperFileClient->getFileNumber( uiBlkAddress), TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads data from the database header
****************************************************************************/
RCODE F_SuperFileHdl::readHeader(
	FLMUINT			uiOffset,
	FLMUINT			uiBytesToRead,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl;

#ifdef FLM_DEBUG
	if( m_uiBlockSize)
	{
		// Note: Block size may not be set because we are in the process of
		// opening the file for the first time and we don't know the block
		// size until after the header has been read.

		flmAssert( (FLMUINT)(uiOffset + uiBytesToRead) <= m_uiBlockSize);
	}
#endif

	if( RC_BAD( rc = getFileHdl( 0, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->read( uiOffset,
		uiBytesToRead, pvBuffer, puiBytesRead)))
	{
		if (rc != NE_FLM_IO_END_OF_FILE && rc != NE_FLM_MEM)
		{
			releaseFile( (FLMUINT)0, TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes data to the database header
****************************************************************************/
RCODE F_SuperFileHdl::writeHeader(
	FLMUINT			uiOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWritten)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl;

#ifdef FLM_DEBUG
	if( m_uiBlockSize)
	{
		flmAssert( (FLMUINT)(uiOffset + uiBytesToWrite) <= m_uiBlockSize);
	}
#endif

	if( RC_BAD( rc = getFileHdl( 0, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->write( uiOffset,
		uiBytesToWrite, pvBuffer, puiBytesWritten)))
	{
		if (rc != NE_FLM_IO_DISK_FULL && rc != NE_FLM_MEM)
		{
			releaseFile( (FLMUINT)0, TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Releases all file handle objects and optionally closes the files
****************************************************************************/
RCODE F_SuperFileHdl::releaseFile(
	FLMUINT		uiFileNum,
	FLMBOOL		bCloseFile)
{
	RCODE								rc = NE_FLM_OK;
	CHECKED_OUT_FILE_HDL *		pCkoFileHdl;
	FLMUINT							uiSlot;

	pCkoFileHdl = getCkoFileHdlPtr( uiFileNum, &uiSlot);
	if( pCkoFileHdl->uiFileNumber == uiFileNum)
	{
		if( RC_BAD( rc = releaseFile( pCkoFileHdl, bCloseFile)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Releases all file handle objects and optionally closes the files
****************************************************************************/
RCODE F_SuperFileHdl::releaseFiles(
	FLMBOOL		bCloseFiles)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLoop;

	for( uiLoop = 0; uiLoop <= m_uiHighestUsedSlot; uiLoop++)
	{
		if( RC_BAD( rc = releaseFile(
			&m_CheckedOutFileHdls[ uiLoop], bCloseFiles)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Releases a file handle object
****************************************************************************/
RCODE F_SuperFileHdl::releaseFile(
	CHECKED_OUT_FILE_HDL *		pCkoFileHdl,
	FLMBOOL							bCloseFile)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = pCkoFileHdl->pFileHdl;

	if( pFileHdl)
	{
		if( pCkoFileHdl->bDirty)
		{
			(void)pFileHdl->flush();
		}

		if( bCloseFile)
		{
			FLMUINT		uiRefCnt;

			uiRefCnt = pFileHdl->Release();
			flmAssert( uiRefCnt == 0);
		}

		clearCkoFileHdl( pCkoFileHdl);
	}

	return( rc);
}

/****************************************************************************
Desc:	Copy one CKO array into another.
****************************************************************************/
void F_SuperFileHdl::copyCkoFileHdls(
	CHECKED_OUT_FILE_HDL *	pSrcCkoArray,
	FLMUINT						uiSrcHighestUsedSlot)
{
	FLMUINT	uiNewSlot;
	FLMUINT	uiSrcSlot;

	// Zeroeth element is always copied.

	f_memcpy( m_pCheckedOutFileHdls, pSrcCkoArray,
					sizeof( CHECKED_OUT_FILE_HDL));

	// Memset the rest of the destination array to zero.

	f_memset( &m_pCheckedOutFileHdls[1], 0, sizeof( CHECKED_OUT_FILE_HDL) *
					(m_uiCkoArraySize - 1));

	m_uiHighestUsedSlot = 0;
	m_uiLowestDirtySlot = 1;
	m_uiHighestDirtySlot = 0;
	
	for (uiSrcSlot = 1, pSrcCkoArray++;
		  uiSrcSlot <= uiSrcHighestUsedSlot;
		  uiSrcSlot++, pSrcCkoArray++)
	{
		if (pSrcCkoArray->pFileHdl && pSrcCkoArray->uiFileNumber)
		{
			uiNewSlot = pSrcCkoArray->uiFileNumber % (m_uiCkoArraySize - 1) + 1;

			// Only overwrite the destination one if the file number is
			// lower than the one already there

			if (pSrcCkoArray->uiFileNumber <
				 m_pCheckedOutFileHdls [uiNewSlot].uiFileNumber ||
				 !m_pCheckedOutFileHdls [uiNewSlot].uiFileNumber)
			{
				if (m_pCheckedOutFileHdls [uiNewSlot].uiFileNumber)
				{
					releaseFile( &m_pCheckedOutFileHdls [uiNewSlot], FALSE);
				}
				
				f_memcpy( &m_pCheckedOutFileHdls [uiNewSlot], pSrcCkoArray,
							 sizeof( CHECKED_OUT_FILE_HDL));
				if (uiNewSlot > m_uiHighestUsedSlot)
				{
					m_uiHighestUsedSlot = uiNewSlot;
				}
				
				if (m_uiHighestFileNumber < pSrcCkoArray->uiFileNumber)
				{
					m_uiHighestFileNumber = pSrcCkoArray->uiFileNumber;
				}
				
				if (pSrcCkoArray->bDirty)
				{
					if (m_uiLowestDirtySlot > m_uiHighestDirtySlot)
					{
						m_uiLowestDirtySlot =
						m_uiHighestDirtySlot = uiNewSlot;
					}
					else if( m_uiHighestDirtySlot < uiNewSlot)
					{
						m_uiHighestDirtySlot = uiNewSlot;
					}
					else if (m_uiLowestDirtySlot < uiNewSlot)
					{
						m_uiLowestDirtySlot = uiNewSlot;
					}
				}
			}
			else
			{
				releaseFile( pSrcCkoArray, FALSE);
			}
		}
	}
}

/****************************************************************************
Desc:	Disable flush minimizing.
****************************************************************************/
void F_SuperFileHdl::disableFlushMinimize( void)
{

	// Copy the allocated array back into the fixed array.
	// This doesn't necessarily copy all of the file handles.

	if( m_pCheckedOutFileHdls != &m_CheckedOutFileHdls [0])
	{
		CHECKED_OUT_FILE_HDL *	pOldCkoArray = m_pCheckedOutFileHdls;
		FLMUINT						uiOldHighestUsedSlot = m_uiHighestUsedSlot;

		m_pCheckedOutFileHdls = &m_CheckedOutFileHdls [0];
		m_uiCkoArraySize = MAX_CHECKED_OUT_FILE_HDLS + 1;
		copyCkoFileHdls( pOldCkoArray, uiOldHighestUsedSlot);

		f_free( &pOldCkoArray);
	}
	
	m_bMinimizeFlushes = FALSE;
}

/****************************************************************************
Desc:	Flush dirty files to disk.
****************************************************************************/
RCODE F_SuperFileHdl::flush( void)
{
	RCODE		rc = NE_FLM_OK;
	FLMUINT	uiLoop;

	// Flush all dirty files

	for( uiLoop = m_uiLowestDirtySlot;
		  uiLoop <= m_uiHighestDirtySlot;
		  uiLoop++)
	{
		if( m_pCheckedOutFileHdls[ uiLoop].bDirty)
		{
			RCODE	tmpRc;

			if( RC_BAD( tmpRc =
						m_pCheckedOutFileHdls[ uiLoop].pFileHdl->flush()))
			{
				rc = tmpRc;
				releaseFile( &m_pCheckedOutFileHdls [uiLoop], TRUE);
			}
			
			m_pCheckedOutFileHdls[ uiLoop].bDirty = FALSE;
		}
	}
	
	m_uiLowestDirtySlot = 1;
	m_uiHighestDirtySlot = 0;
	
	return( rc);
}

/****************************************************************************
Desc:	Truncates back to an end of file block address.
		This may only be called from reduce() because there cannot
		be any other cases to reduce a 3x block file.
****************************************************************************/
RCODE	F_SuperFileHdl::truncateFile(
	FLMUINT			uiEOFBlkAddress)
{
	RCODE 			rc = NE_FLM_OK;
	FLMUINT			uiFileNumber = m_pSuperFileClient->getFileNumber( uiEOFBlkAddress);
	FLMUINT			uiBlockOffset = m_pSuperFileClient->getFileOffset( uiEOFBlkAddress);
	IF_FileHdl *	pFileHdl;

	// Truncate the current block file.

	if( RC_BAD( rc = getFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->truncate( uiBlockOffset)))
	{
		releaseFile( uiFileNumber, TRUE);
		goto Exit;
	}

	// Visit the rest of the high block files until a NULL file hdl is hit.

	for( ;;)
	{
		if( RC_BAD( getFileHdl( ++uiFileNumber, TRUE, &pFileHdl)))
		{
			break;
		}

		if( RC_BAD( rc = pFileHdl->truncate( (FLMUINT)0 )))
		{
			releaseFile( uiFileNumber, TRUE);
			goto Exit;
		}

		if( RC_BAD( rc = releaseFile( uiFileNumber, TRUE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Truncate to zero length any files between the specified start
		and end files.
****************************************************************************/
void F_SuperFileHdl::truncateFiles(
	FLMUINT	uiStartFileNum,
	FLMUINT	uiEndFileNum)
{
	FLMUINT			uiFileNumber;
	IF_FileHdl *	pFileHdl;

	for( uiFileNumber = uiStartFileNum;
		  uiFileNumber <= uiEndFileNum;
		  uiFileNumber++ )
	{
		if( RC_OK( getFileHdl( uiFileNumber, TRUE, &pFileHdl )))
		{
			(void)pFileHdl->truncate( (FLMUINT)0 );
			(void)releaseFile( uiFileNumber, TRUE);
		}
	}
}

/****************************************************************************
Desc:	Returns the physical size of a file
****************************************************************************/
RCODE F_SuperFileHdl::getFileSize(
	FLMUINT			uiFileNumber,
	FLMUINT64 *		pui64FileSize)
{
	RCODE 			rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;

	*pui64FileSize = 0;

	// Get the file handle.

	if( RC_BAD( rc = getFileHdl( uiFileNumber, FALSE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->size( pui64FileSize)))
	{
		releaseFile( uiFileNumber, TRUE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns the path of a file given its file number
****************************************************************************/
RCODE F_SuperFileHdl::getFilePath(
	FLMUINT			uiFileNumber,
	char *			pszIoPath)
{
	return( m_pSuperFileClient->getFilePath( uiFileNumber, pszIoPath));
}

/****************************************************************************
Desc:	Reallocates the checked out file handle array.
****************************************************************************/
RCODE F_SuperFileHdl::reallocCkoArray(
	FLMUINT						uiFileNum)
{
	RCODE							rc = NE_FLM_OK;
	FLMUINT						uiNewSize;
	CHECKED_OUT_FILE_HDL *	pNewCkoArray;
	CHECKED_OUT_FILE_HDL *	pOldCkoArray;
	FLMUINT						uiOldHighestUsedSlot;

	if (uiFileNum < m_uiHighestFileNumber)
	{
		uiFileNum = m_uiHighestFileNumber;
	}
	
	uiNewSize = uiFileNum + 128;

	if (RC_BAD( rc = f_calloc( sizeof( CHECKED_OUT_FILE_HDL) * uiNewSize,
									&pNewCkoArray)))
	{
		goto Exit;
	}

	pOldCkoArray = m_pCheckedOutFileHdls;
	uiOldHighestUsedSlot = m_uiHighestUsedSlot;

	m_pCheckedOutFileHdls = pNewCkoArray;
	m_uiCkoArraySize = uiNewSize;

	copyCkoFileHdls( pOldCkoArray, uiOldHighestUsedSlot);

	// Can't free the old one until after the copy!

	if (pOldCkoArray != &m_CheckedOutFileHdls [0])
	{
		f_free( &pOldCkoArray);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns a file handle given the file's number
****************************************************************************/
RCODE F_SuperFileHdl::getFileHdl(
	FLMUINT						uiFileNum,
	FLMBOOL						bGetForUpdate,
	IF_FileHdl **				ppFileHdl)
{
	RCODE							rc = NE_FLM_OK;
	IF_FileHdl *				pFileHdl = NULL;
	CHECKED_OUT_FILE_HDL *	pCkoFileHdl;
	char							szFilePath[ F_PATH_MAX_SIZE];
	FLMUINT						uiSlot;

	pCkoFileHdl = getCkoFileHdlPtr( uiFileNum, &uiSlot);
	if( pCkoFileHdl->uiFileNumber != uiFileNum &&
		 pCkoFileHdl->pFileHdl)
	{
		if( pCkoFileHdl->bDirty && m_bMinimizeFlushes)
		{
			flmAssert( pCkoFileHdl->uiFileNumber);
			if (RC_BAD( reallocCkoArray( uiFileNum)))
			{
				goto Exit;
			}
			pCkoFileHdl = getCkoFileHdlPtr( uiFileNum, &uiSlot);

			// Better have reallocated so that the new slot for
			// the file number has nothing in it.

			flmAssert( !pCkoFileHdl->uiFileNumber &&
						  !pCkoFileHdl->pFileHdl);
		}
		else
		{
			if( RC_BAD( rc = releaseFile( pCkoFileHdl, FALSE)))
			{
				goto Exit;
			}
		}
	}

	if( !pCkoFileHdl->pFileHdl)
	{
		if (!pFileHdl)
		{
			// Build the file path

			if( RC_BAD( rc = m_pSuperFileClient->getFilePath(  
				uiFileNum, szFilePath)))
			{
				goto Exit;
			}

			// Open the file

			if( RC_BAD( rc = f_getFileSysPtr()->openFile( szFilePath,
				FLM_IO_RDWR | FLM_IO_SH_DENYNONE | FLM_IO_DIRECT,
				&pFileHdl)))
			{
				goto Exit;
			}
		}

		pCkoFileHdl->pFileHdl = pFileHdl;
		pFileHdl = NULL;
		pCkoFileHdl->uiFileNumber = uiFileNum;
		pCkoFileHdl->bDirty = FALSE;
		
		if( m_uiHighestUsedSlot < uiSlot)
		{
			m_uiHighestUsedSlot = uiSlot;
		}
		
		if (m_uiHighestFileNumber < uiFileNum)
		{
			m_uiHighestFileNumber = uiFileNum;
		}
	}

	*ppFileHdl = pCkoFileHdl->pFileHdl;
	if( bGetForUpdate)
	{
		pCkoFileHdl->bDirty = TRUE;
		if (m_uiLowestDirtySlot > m_uiHighestDirtySlot)

		{
			m_uiLowestDirtySlot =
			m_uiHighestDirtySlot = uiSlot;
		}
		else if( m_uiHighestDirtySlot < uiSlot)
		{
			m_uiHighestDirtySlot = uiSlot;
		}
		else if (m_uiLowestDirtySlot < uiSlot)
		{
			m_uiLowestDirtySlot = uiSlot;
		}
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}
