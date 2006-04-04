//------------------------------------------------------------------------------
// Desc:	This include file contains the methods for FLAIM's
// 		super file class.
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
// $Id: fsuperfl.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC char base24ToDigit(
	FLMUINT	uiBaseValue);

/****************************************************************************
Desc:
****************************************************************************/
F_FileIdList::F_FileIdList()
{
	m_hMutex = F_MUTEX_NULL;
	m_uiFileIdTblSize = 0;
	m_puiFileIdTbl = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_FileIdList::~F_FileIdList()
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}

	if( m_puiFileIdTbl)
	{
		for( FLMUINT uiLoop = 0; uiLoop < m_uiFileIdTblSize; uiLoop++)
		{
			if( m_puiFileIdTbl[ uiLoop])
			{
				(void)gv_XFlmSysData.pFileHdlMgr->removeFileHdls(
					m_puiFileIdTbl[ uiLoop]);
			}
		}

		f_free( &m_puiFileIdTbl);
	}
}

/****************************************************************************
Desc: Allocates the mutex used by the file ID list object
****************************************************************************/
RCODE F_FileIdList::setup( void)
{
	RCODE			rc = NE_XFLM_OK;

	flmAssert( m_hMutex == F_MUTEX_NULL);

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Translates a database file number into a file ID.
****************************************************************************/
RCODE F_FileIdList::getFileId(
	FLMUINT		uiFileNumber,
	FLMUINT *	puiFileId)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bMutexLocked = TRUE;
	FLMUINT		uiLoop = 0;

	f_mutexLock( m_hMutex);
	bMutexLocked = TRUE;

	if( uiFileNumber >= m_uiFileIdTblSize)
	{
		FLMUINT	uiOldTableSize = m_uiFileIdTblSize;

		/*
		Re-size the table
		*/

		if( RC_BAD( rc = f_recalloc( (uiFileNumber + 1) * sizeof( FLMUINT),
			&m_puiFileIdTbl)))
		{
			goto Exit;
		}
		m_uiFileIdTblSize = uiFileNumber + 1;

		for( uiLoop = uiOldTableSize; uiLoop < m_uiFileIdTblSize; uiLoop++)
		{
			m_puiFileIdTbl[ uiLoop] = gv_XFlmSysData.pFileHdlMgr->getUniqueId();
		}
	}

	*puiFileId = m_puiFileIdTbl[ uiFileNumber];

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileHdl::F_SuperFileHdl( void)
{
	m_pFileIdList = NULL;
	m_pszDbFileName = NULL;
	m_pszDataFileNameBase = NULL;
	f_memset( &m_CheckedOutFileHdls[ 0], 0, sizeof( m_CheckedOutFileHdls));
	m_pCheckedOutFileHdls = &m_CheckedOutFileHdls [0];
	m_uiCkoArraySize = MAX_CHECKED_OUT_FILE_HDLS + 1;
	m_uiBlockSize = 0;
	m_uiExtendSize = XFLM_DEFAULT_FILE_EXTEND_SIZE;
	m_uiMaxAutoExtendSize = gv_XFlmSysData.uiMaxFileSize;
	m_uiLowestDirtySlot = 1;
	m_uiHighestDirtySlot = 0;
	m_uiHighestUsedSlot = 0;
	m_uiHighestFileNumber = 0;
	m_bMinimizeFlushes = FALSE;
	m_bSetupCalled = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_SuperFileHdl::~F_SuperFileHdl()
{
	/*
	Release any file handles still being held and close the files.
	*/

	if( m_bSetupCalled)
	{
		(void)ReleaseFiles( TRUE);
	}

	/*
	Release the ID list
	*/

	if( m_pFileIdList)
	{
		m_pFileIdList->Release( FALSE);
	}

	if (m_pszDbFileName)
	{
		f_free( &m_pszDbFileName);
	}
}

/****************************************************************************
Desc:	Configures the super file object
****************************************************************************/
RCODE F_SuperFileHdl::Setup(
	F_FileIdList *		pFileIdList,
	const char *		pszDbFileName,
	const char *		pszDataDir)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiNameLen;
	FLMUINT		uiDataNameLen;
	char			szDir [F_PATH_MAX_SIZE];
	char			szBaseName [F_FILENAME_SIZE];

	flmAssert( !m_bSetupCalled);

	if( !pszDbFileName && *pszDbFileName == 0)
	{
		rc = RC_SET( NE_XFLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	if( !pFileIdList)
	{
		if( (m_pFileIdList = f_new F_FileIdList) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pFileIdList->setup()))
		{
			FLMUINT		uiRefCnt;

			uiRefCnt = m_pFileIdList->Release( FALSE);
			flmAssert( !uiRefCnt);
			m_pFileIdList = NULL;
			goto Exit;
		}
	}
	else
	{
		pFileIdList->AddRef( FALSE);
		m_pFileIdList = pFileIdList;
	}

	uiNameLen = f_strlen( pszDbFileName);
	if (pszDataDir && *pszDataDir)
	{
		if (RC_BAD( rc = gv_pFileSystem->pathReduce( 
			pszDbFileName, szDir, szBaseName)))
		{
			goto Exit;
		}
		f_strcpy( szDir, pszDataDir);
		if (RC_BAD( rc = gv_pFileSystem->pathAppend( 
			szDir, szBaseName)))
		{
			goto Exit;
		}
		uiDataNameLen = f_strlen( szDir);

		if (RC_BAD( rc = f_alloc( (uiNameLen + 1) + (uiDataNameLen + 1),
									&m_pszDbFileName)))
		{
			goto Exit;
		}

		f_memcpy( m_pszDbFileName, pszDbFileName, uiNameLen + 1);
		m_pszDataFileNameBase = m_pszDbFileName + uiNameLen + 1;
		F_DbSystem::getDbBasePath( m_pszDataFileNameBase, szDir, &m_uiDataExtOffset);
		m_uiExtOffset = uiNameLen - (uiDataNameLen - m_uiDataExtOffset);
	}
	else
	{
		if (RC_BAD( rc = f_alloc( (uiNameLen + 1) * 2, &m_pszDbFileName)))
		{
			goto Exit;
		}

		f_memcpy( m_pszDbFileName, pszDbFileName, uiNameLen + 1);
		m_pszDataFileNameBase = m_pszDbFileName + uiNameLen + 1;
		F_DbSystem::getDbBasePath( m_pszDataFileNameBase, 
			m_pszDbFileName, &m_uiDataExtOffset);
		m_uiExtOffset = m_uiDataExtOffset;
	}

	m_bSetupCalled = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc: Creates a file
****************************************************************************/
RCODE F_SuperFileHdl::CreateFile(
	FLMUINT			uiFileNumber)
{
	char				szFilePath[ F_PATH_MAX_SIZE];
	F_FileHdl *		pFileHdl = NULL;
	FLMUINT			uiFileId;
	RCODE				rc = NE_XFLM_OK;

	/*
	Sanity checks
	*/

	flmAssert( m_bSetupCalled && m_uiBlockSize);
	flmAssert( uiFileNumber <= MAX_LOG_BLOCK_FILE_NUMBER);

	/*
	See if we already have an open file handle (or if we can open the file).
	If so, truncate the file and use it.
	*/

	if( RC_OK( rc = GetFileHdl( uiFileNumber, TRUE, (IF_FileHdl **)&pFileHdl)))
	{
		rc = pFileHdl->Truncate( 0);
		pFileHdl = NULL; // Don't want to release the file handle at exit
		goto Exit;
	}
	else if( rc != NE_XFLM_IO_PATH_NOT_FOUND)
	{
		goto Exit;
	}

	// The file was not found above.  Allocate a new file handle.

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pFileHdl->SetBlockSize( m_uiBlockSize);

	// Configure the file handle.

	if( RC_BAD( rc = m_pFileIdList->getFileId( uiFileNumber, &uiFileId)))
	{
		goto Exit;
	}

	flmAssert( uiFileId); // File ID should always be non-zero

	pFileHdl->setupFileHdl( uiFileId, FALSE);

	// Build the file path

	if( RC_BAD( rc = GetFilePath( uiFileNumber, szFilePath)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->Create( szFilePath,
		XFLM_IO_RDWR | XFLM_IO_EXCL | XFLM_IO_DIRECT | XFLM_IO_SH_DENYNONE)))
	{
		goto Exit;
	}

	// Insert into the file handle manager

	gv_XFlmSysData.pFileHdlMgr->insertInUsedList( FALSE, pFileHdl, TRUE);

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
RCODE F_SuperFileHdl::ReadBlock(
	FLMUINT			uiBlkAddress,
	FLMUINT			uiBytesToRead,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	IF_FileHdl *	pFileHdl = NULL;
	RCODE				rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled && m_uiBlockSize);

	if( RC_BAD( rc = GetFileHdl(
		FSGetFileNumber( uiBlkAddress), FALSE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->SectorRead(
		FSGetFileOffset( uiBlkAddress), uiBytesToRead,
		pvBuffer, puiBytesRead)))
	{
		if (rc != NE_XFLM_IO_END_OF_FILE && rc != NE_XFLM_MEM)
		{
			ReleaseFile( FSGetFileNumber( uiBlkAddress), TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Writes a block to the database
****************************************************************************/
RCODE F_SuperFileHdl::WriteBlock(
	FLMUINT			uiBlkAddress,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT			uiBufferSize,
	F_IOBuffer *	pIOBuffer,
	FLMUINT *		puiBytesWritten)
{
	IF_FileHdl *	pFileHdl = NULL;
	RCODE				rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled && m_uiBlockSize);

Get_Handle:
	if( RC_BAD( rc = GetFileHdl(
		FSGetFileNumber( uiBlkAddress), TRUE, &pFileHdl)))
	{
		if (rc == NE_XFLM_IO_PATH_NOT_FOUND)
		{
			if (RC_BAD( rc = CreateFile( FSGetFileNumber( uiBlkAddress))))
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
	if( RC_BAD( rc = pFileHdl->SectorWrite(
		FSGetFileOffset( uiBlkAddress), uiBytesToWrite,
		pvBuffer, uiBufferSize, pIOBuffer, puiBytesWritten)))
	{
		if (rc != NE_XFLM_IO_DISK_FULL && rc != NE_XFLM_MEM)
		{
			ReleaseFile( FSGetFileNumber( uiBlkAddress), TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads data from the database header
****************************************************************************/
RCODE F_SuperFileHdl::ReadHeader(
	FLMUINT			uiOffset,
	FLMUINT			uiBytesToRead,
	void *			pvBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_XFLM_OK;
	IF_FileHdl *	pFileHdl;

#ifdef FLM_DEBUG
	if( m_uiBlockSize)
	{
		/*
		Note: Block size may not be set because we are in the process of
		opening the file for the first time and we don't know the block
		size until after the header has been read.
		*/

		flmAssert( (FLMUINT)(uiOffset + uiBytesToRead) <= m_uiBlockSize);
	}
#endif

	if( RC_BAD( rc = GetFileHdl( 0, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->Read( uiOffset,
		uiBytesToRead, pvBuffer, puiBytesRead)))
	{
		if (rc != NE_XFLM_IO_END_OF_FILE && rc != NE_XFLM_MEM)
		{
			ReleaseFile( (FLMUINT)0, TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes data to the database header
****************************************************************************/
RCODE F_SuperFileHdl::WriteHeader(
	FLMUINT			uiOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWritten)
{
	RCODE				rc = NE_XFLM_OK;
	IF_FileHdl *	pFileHdl;

#ifdef FLM_DEBUG
	if( m_uiBlockSize)
	{
		flmAssert( (FLMUINT)(uiOffset + uiBytesToWrite) <= m_uiBlockSize);
	}
#endif

	if( RC_BAD( rc = GetFileHdl( 0, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->Write( uiOffset,
		uiBytesToWrite, pvBuffer, puiBytesWritten)))
	{
		if (rc != NE_XFLM_IO_DISK_FULL && rc != NE_XFLM_MEM)
		{
			ReleaseFile( (FLMUINT)0, TRUE);
		}
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Releases all file handle objects and optionally closes the files
****************************************************************************/
RCODE F_SuperFileHdl::ReleaseFile(
	FLMUINT		uiFileNum,
	FLMBOOL		bCloseFile)
{
	RCODE								rc = NE_XFLM_OK;
	CHECKED_OUT_FILE_HDL *		pCkoFileHdl;
	FLMUINT							uiSlot;

	pCkoFileHdl = getCkoFileHdlPtr( uiFileNum, &uiSlot);
	if( pCkoFileHdl->uiFileNumber == uiFileNum)
	{
		if( RC_BAD( rc = ReleaseFile( pCkoFileHdl, bCloseFile)))
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
RCODE F_SuperFileHdl::ReleaseFiles(
	FLMBOOL		bCloseFiles)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;

	flmAssert( m_bSetupCalled);

	for( uiLoop = 0; uiLoop <= m_uiHighestUsedSlot; uiLoop++)
	{
		if( RC_BAD( rc = ReleaseFile(
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
RCODE F_SuperFileHdl::ReleaseFile(
	CHECKED_OUT_FILE_HDL *		pCkoFileHdl,
	FLMBOOL							bCloseFile)
{
	RCODE				rc = NE_XFLM_OK;
	F_FileHdl *		pFileHdl = (F_FileHdl *)pCkoFileHdl->pFileHdl;

	if( pFileHdl)
	{
		flmAssert( pFileHdl->getFileId());

		if( pCkoFileHdl->bDirty)
		{
			(void)pFileHdl->Flush();
		}

		if( bCloseFile)
		{
			FLMUINT		uiRefCnt;

			/*
			We must remove this handle from all lists and release
			the file handle.
			*/

			gv_XFlmSysData.pFileHdlMgr->removeFileHdls( pFileHdl->getFileId());
			uiRefCnt = pFileHdl->Release();
			flmAssert( uiRefCnt == 0);			// pFileHdl should have been freed.
		}
		else
		{

			// Link out of the used list and move to the available list.

			gv_XFlmSysData.pFileHdlMgr->makeAvailAndRelease( FALSE, pFileHdl);

			// NOTE: makeAvailAndRelease will perform a release on the
			// file handle object for the caller.
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
	FLMUINT						uiSrcHighestUsedSlot
	)
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
					ReleaseFile( &m_pCheckedOutFileHdls [uiNewSlot], FALSE);
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
				ReleaseFile( pSrcCkoArray, FALSE);
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

	if (m_pCheckedOutFileHdls != &m_CheckedOutFileHdls [0])
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
RCODE F_SuperFileHdl::Flush( void)
{
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiLoop;

	// Flush all dirty files

	for (uiLoop = m_uiLowestDirtySlot;
		  uiLoop <= m_uiHighestDirtySlot;
		  uiLoop++)
	{
		if( m_pCheckedOutFileHdls[ uiLoop].bDirty)
		{
			RCODE	tmpRc;

			if (RC_BAD( tmpRc =
						m_pCheckedOutFileHdls[ uiLoop].pFileHdl->Flush()))
			{
				rc = tmpRc;
				ReleaseFile( &m_pCheckedOutFileHdls [uiLoop], TRUE);
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
RCODE	F_SuperFileHdl::TruncateFile(
	FLMUINT			uiEOFBlkAddress)
{
	RCODE 			rc = NE_XFLM_OK;
	FLMUINT			uiFileNumber = (FLMUINT)FSGetFileNumber( uiEOFBlkAddress);
	FLMUINT			uiBlockOffset = (FLMUINT)FSGetFileOffset( uiEOFBlkAddress);
	IF_FileHdl *	pFileHdl;

	/*
	Truncate the current block file.
	*/

	if( RC_BAD( rc = GetFileHdl( uiFileNumber, TRUE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->Truncate( uiBlockOffset)))
	{
		ReleaseFile( uiFileNumber, TRUE);
		goto Exit;
	}

	/*
	Visit the rest of the high block files until a NULL file hdl is hit.
	*/

	for( ;;)
	{
		if( RC_BAD( GetFileHdl( ++uiFileNumber, TRUE, &pFileHdl)))
		{
			break;
		}

		if( RC_BAD( rc = pFileHdl->Truncate( (FLMUINT)0 )))
		{
			ReleaseFile( uiFileNumber, TRUE);
			goto Exit;
		}

		if( RC_BAD( rc = ReleaseFile( uiFileNumber, TRUE)))
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
void F_SuperFileHdl::TruncateFiles(
	FLMUINT	uiStartFileNum,
	FLMUINT	uiEndFileNum)
{
	FLMUINT			uiFileNumber;
	IF_FileHdl *	pFileHdl;

	for( uiFileNumber = uiStartFileNum;
		  uiFileNumber <= uiEndFileNum;
		  uiFileNumber++ )
	{
		if( RC_OK( GetFileHdl( uiFileNumber, TRUE, &pFileHdl )))
		{
			(void)pFileHdl->Truncate( (FLMUINT)0 );
			(void)ReleaseFile( uiFileNumber, TRUE);
		}
	}
}

/****************************************************************************
Desc:	Returns the physical size of a file
****************************************************************************/
RCODE F_SuperFileHdl::GetFileSize(
	FLMUINT			uiFileNumber,
	FLMUINT64 *		pui64FileSize)
{
	IF_FileHdl *	pFileHdl = NULL;
	RCODE 			rc = NE_XFLM_OK;

	flmAssert( m_bSetupCalled);
	flmAssert( pui64FileSize);

	*pui64FileSize = 0;

	// Get the file handle.

	if( RC_BAD( rc = GetFileHdl( uiFileNumber, FALSE, &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->Size( pui64FileSize)))
	{
		ReleaseFile( uiFileNumber, TRUE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns the path of a file given its file number
****************************************************************************/
RCODE F_SuperFileHdl::GetFilePath(
	FLMUINT			uiFileNumber,
	char *			pszIoPath)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiExtOffset;

	// Sanity checks

	flmAssert( m_bSetupCalled);

	if (!uiFileNumber)
	{
		f_strcpy( pszIoPath, m_pszDbFileName);
		goto Exit;
	}

	if( uiFileNumber <= MAX_DATA_BLOCK_FILE_NUMBER)
	{
		f_memcpy( pszIoPath, m_pszDataFileNameBase, m_uiDataExtOffset);
		uiExtOffset = m_uiDataExtOffset;
	}
	else
	{
		f_memcpy( pszIoPath, m_pszDbFileName, m_uiExtOffset);
		uiExtOffset = m_uiExtOffset;
	}

	// Modify the file's extension.

	bldSuperFileExtension( uiFileNumber, &pszIoPath[ uiExtOffset]);

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reallocates the checked out file handle array.
****************************************************************************/
RCODE F_SuperFileHdl::reallocCkoArray(
	FLMUINT	uiFileNum
	)
{
	RCODE							rc = NE_XFLM_OK;
	FLMUINT						uiNewSize;
	CHECKED_OUT_FILE_HDL *	pNewCkoArray;
	CHECKED_OUT_FILE_HDL *	pOldCkoArray;
	FLMUINT						uiOldHighestUsedSlot;

	if (uiFileNum < m_uiHighestFileNumber)
	{
		uiFileNum = m_uiHighestFileNumber;
	}
	uiNewSize = uiFileNum + 128;

	// Reallocate so we can guarantee that all of the current file
	// numbers will copy and there is room for this new one as well.

	if (uiNewSize > MAX_LOG_BLOCK_FILE_NUMBER + 1)
	{
		flmAssert( uiFileNum <= MAX_LOG_BLOCK_FILE_NUMBER);
		uiNewSize = MAX_LOG_BLOCK_FILE_NUMBER + 1;
	}

	// No need to call f_calloc, because copyCkoFileHdls will initialize
	// it below.

	if (RC_BAD( rc = f_alloc( sizeof( CHECKED_OUT_FILE_HDL) * uiNewSize,
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
RCODE F_SuperFileHdl::GetFileHdl(
	FLMUINT			uiFileNum,
	FLMBOOL			bGetForUpdate,
	IF_FileHdl **	ppFileHdl)
{
	F_FileHdl *					pFileHdl = NULL;
	FLMUINT						uiFileId;
	CHECKED_OUT_FILE_HDL *	pCkoFileHdl;
	char							szFilePath[ F_PATH_MAX_SIZE];
	FLMUINT						uiSlot;
	RCODE							rc = NE_XFLM_OK;

	/*
	Get the file handle
	*/

	pCkoFileHdl = getCkoFileHdlPtr( uiFileNum, &uiSlot);
	if( pCkoFileHdl->uiFileNumber != uiFileNum &&
		 pCkoFileHdl->pFileHdl)
	{
		if (pCkoFileHdl->bDirty && m_bMinimizeFlushes)
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
			if( RC_BAD( rc = ReleaseFile( pCkoFileHdl, FALSE)))
			{
				goto Exit;
			}
		}
	}

	if( !pCkoFileHdl->pFileHdl)
	{
		/*
		Get the file ID
		*/

		if( RC_BAD( rc = m_pFileIdList->getFileId( uiFileNum, &uiFileId)))
		{
			goto Exit;
		}

		/*
		Look for an available file handle if not opening exclusive.
		NOTE: AddRef() performed for caller by findAvail if a file
				handle is found.
		*/

		gv_XFlmSysData.pFileHdlMgr->findAvail( uiFileId, FALSE, &pFileHdl);
			
		if (!pFileHdl)
		{
			/*
			Allocate a new file handle, open the file and
			link into the used directory.
			*/

			if ((pFileHdl = f_new F_FileHdl) == NULL)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}

			/*
			If m_uiBlockSize is 0, direct I/O will not be used
			*/

			pFileHdl->SetBlockSize( m_uiBlockSize);

			flmAssert( uiFileId); // File ID must be non-zero

			pFileHdl->setupFileHdl( uiFileId, FALSE);

			// Build the file path

			if( RC_BAD( rc = GetFilePath( uiFileNum, szFilePath)))
			{
				goto Exit;
			}

			// Open the file

			if( RC_BAD( rc = pFileHdl->Open( szFilePath,
				XFLM_IO_RDWR | XFLM_IO_SH_DENYNONE | XFLM_IO_DIRECT)))
			{
				goto Exit;
			}

			// Insert into the manager

			gv_XFlmSysData.pFileHdlMgr->insertInUsedList( FALSE, pFileHdl, TRUE);
		}

		pCkoFileHdl->pFileHdl = (IF_FileHdl *)pFileHdl;
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

/****************************************************************************
Desc:		Generates a file name given a super file number.
			Adds ".xx" to pFileExtension.  Use lower case characters.
Notes:	This is a base 24 alphanumeric value where
			{ a, b, c, d, e, f, i, l, o, r, u, v } values are removed.
****************************************************************************/
void bldSuperFileExtension(
	FLMUINT		uiFileNum,
	char *		pszFileExtension)
{
	char	ucLetter;

	if (uiFileNum <= MAX_DATA_BLOCK_FILE_NUMBER - 1536)
	{
		// No additional letter - File numbers 1 to 511
		// This is just like pre-4.3 numbering.
		ucLetter = 0;
	}
	else if (uiFileNum <= MAX_DATA_BLOCK_FILE_NUMBER - 1024)
	{
		// File numbers 512 to 1023
		ucLetter = 'r';
	}
	else if (uiFileNum <= MAX_DATA_BLOCK_FILE_NUMBER - 512)
	{
		// File numbers 1024 to 1535
		ucLetter = 's';
	}
	else if (uiFileNum <= MAX_DATA_BLOCK_FILE_NUMBER)
	{
		// File numbers 1536 to 2047
		ucLetter = 't';
	}
	else if (uiFileNum <= MAX_LOG_BLOCK_FILE_NUMBER - 1536)
	{
		// File numbers 2048 to 2559
		ucLetter = 'v';
	}
	else if (uiFileNum <= MAX_LOG_BLOCK_FILE_NUMBER - 1024)
	{
		// File numbers 2560 to 3071
		ucLetter = 'w';
	}
	else if (uiFileNum <= MAX_LOG_BLOCK_FILE_NUMBER - 512)
	{
		// File numbers 3072 to 3583
		ucLetter = 'x';
	}
	else
	{
		flmAssert( uiFileNum <= MAX_LOG_BLOCK_FILE_NUMBER);

		// File numbers 3584 to 4095
		ucLetter = 'z';
	}

	*pszFileExtension++ = '.';
	*pszFileExtension++ = (char)(base24ToDigit( (uiFileNum & 511) / 24));
	*pszFileExtension++ = (char)(base24ToDigit( (uiFileNum & 511) % 24));
	*pszFileExtension++ = ucLetter;
	*pszFileExtension   = 0;
}

/****************************************************************************
Desc:		Turn a base 24 value into a native alphanumeric value.
Notes:	This is a base 24 alphanumeric value where
			{a, b, c, d, e, f, i, l, o, r, u, v } values are removed.
****************************************************************************/
FSTATIC char base24ToDigit(
	FLMUINT	uiValue)
{
	flmAssert( uiValue <= 23);

	if( uiValue <= 9)
	{
		uiValue += (FLMUINT) NATIVE_ZERO;
	}
	else
	{
		uiValue = f_toascii(uiValue) - 10 + (FLMUINT)f_toascii('g');
		if( uiValue >= (FLMUINT)'i')
		{
			uiValue++;
			if( uiValue >= (FLMUINT)'l')
			{
				uiValue++;
				if( uiValue >= (FLMUINT)'o')
				{
					uiValue++;
					if( uiValue >= (FLMUINT)'r')
					{
						uiValue++;
						if( uiValue >= (FLMUINT)'u')
						{
							uiValue++;
							if( uiValue >= (FLMUINT)'v')
							{
								uiValue++;
							}
						}
					}
				}
			}
		}
	}
	return (char)uiValue;
}
