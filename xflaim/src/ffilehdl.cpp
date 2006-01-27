//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileHdlMgr and F_FileHdl classes.
//
// Tabs:	3
//
//		Copyright (c) 1997-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ffilehdl.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Name:	f_filecpy
Desc:	quick and easy way to write a string to a file.  The contents of
		pszSourceFile becomes pszData.
****************************************************************************/
RCODE f_filecpy(
	const char *	pszSourceFile,
	const char *	pszData)
{
	IF_FileHdl *	pFileHdl = NULL;
	F_FileSystem	fileSystem;
	RCODE				rc = NE_XFLM_OK;

	//if it exists, delete it
	if (RC_OK( rc = fileSystem.Exists( pszSourceFile)))
	{
		if ( RC_BAD( rc = fileSystem.Delete( pszSourceFile)))
		{
			goto Exit;
		}
	}

	if ( RC_BAD( rc = fileSystem.Create( pszSourceFile, XFLM_IO_RDWR,
		&pFileHdl)))
	{
		goto Exit;
	}

	{
		FLMUINT uiBytesWritten = 0;
		if ( RC_BAD( rc = pFileHdl->Write(
			0,
			f_strlen( pszData),
			(void *)pszData,
			&uiBytesWritten)))
		{
			goto Exit;
		}
	}
Exit:
	if ( pFileHdl)
	{
		pFileHdl->Close();
		pFileHdl->Release();
		pFileHdl = NULL;
	}
	return rc;
}

/****************************************************************************
Name:	f_filecat
Desc:	quick and easy way to append a string to a file.  The contents of
		pszData are appended to pszSourceFile.
****************************************************************************/
RCODE f_filecat(
	const char *	pszSourceFile,
	const char *	pszData)
{
	IF_FileHdl *		pFileHdl = NULL;
	F_FileSystem		fileSystem;
	RCODE					rc = NE_XFLM_OK;

	if (RC_BAD( rc = fileSystem.Exists( pszSourceFile)))
	{
		//create file if it doesn't already exist
		if ( rc == NE_XFLM_IO_PATH_NOT_FOUND)
		{
			rc = fileSystem.Create( pszSourceFile, XFLM_IO_RDWR,
				&pFileHdl);
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		rc = fileSystem.Open( pszSourceFile,
			XFLM_IO_RDWR, &pFileHdl);
	}

	if (RC_BAD( rc))
	{
		goto Exit;
	}

	{
		FLMUINT64 	ui64FileSize = 0;
		FLMUINT 		uiBytesWritten = 0;

		if ( RC_BAD( rc = pFileHdl->Size( &ui64FileSize)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pFileHdl->Write( ui64FileSize, f_strlen( pszData),
			(void *)pszData, &uiBytesWritten)))
		{
			goto Exit;
		}
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Close();
		pFileHdl->Release();
		pFileHdl = NULL;
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Initializes variables
****************************************************************************/
F_FileHdlMgr::F_FileHdlMgr()
{
	m_hMutex = F_MUTEX_NULL;

	//m_uiOpenThreshold = ~0;	<- this creates a compiler warning!
	m_uiOpenThreshold = 0xFFFF;			// No limit - this should be enough
	FLM_SECS_TO_TIMER_UNITS( 30 * 60, m_uiMaxAvailTime);	// 30 minutes
	m_bIsSetup = FALSE;

	m_uiFileIdCounter = 0;

	m_pFirstAvail = NULL;
	m_pLastAvail = NULL;
	m_uiNumAvail = 0;

	m_pFirstUsed = NULL;
	m_pLastUsed = NULL;
	m_uiNumUsed = 0;

}

/****************************************************************************
Desc:	Setup the File handle manager.
****************************************************************************/
RCODE F_FileHdlMgr::setupFileHdlMgr(
	FLMUINT		uiOpenThreshold,
	FLMUINT		uiMaxAvailTime)
{
	RCODE	rc = NE_XFLM_OK;

	// Need to allocate a mutex

	if (RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	m_uiOpenThreshold = uiOpenThreshold;
	m_uiMaxAvailTime = uiMaxAvailTime;
	m_bIsSetup = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Return the next available file handle that matches the uiFileId.
****************************************************************************/
void F_FileHdlMgr::findAvail(
	FLMUINT			uiFileId,			// Desired FileHdr's ID
	FLMBOOL			bReadOnlyFlag,		// TRUE if file is read only
	F_FileHdl **	ppFileHdl)			// [out] returned FileHdl object.
{
	F_FileHdl *	pFileHdl;

	lockMutex( FALSE);
	pFileHdl = m_pFirstAvail;
	while (pFileHdl)
	{
		if (pFileHdl->m_uiFileId == uiFileId &&
			 pFileHdl->m_bOpenedReadOnly == bReadOnlyFlag)
		{

			// Move this file handle out of the available list into
			// the used list.

			// NOTE: To prevent this file handle from being freed this code
			// performs an AddRef while its being relinked.  This reference
			// will be kept for the caller.

			pFileHdl->AddRef();			// LOCK WHILE MOVING FILE HDL

			removeFromList( TRUE,
				pFileHdl, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
			insertInList( TRUE, pFileHdl, FALSE,
				&m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);

			// NOTE: DO NOT CALL RELEASE -- Keep reference for caller.

			break;
		}
		pFileHdl = pFileHdl->m_pNext;
	}
	unlockMutex( FALSE);
	*ppFileHdl = pFileHdl;
}

/****************************************************************************
Desc: Make the specified F_FileHdl available for someone else to use.
****************************************************************************/
void F_FileHdlMgr::makeAvailAndRelease(
	FLMBOOL			bMutexAlreadyLocked,
	F_FileHdl *		pFileHdl)			// FileHdl to move to the avail list.
{
	pFileHdl->m_uiAvailTime = (FLMUINT)FLM_GET_TIMER();

	lockMutex( bMutexAlreadyLocked);

	// NOTE: To prevent this file handle from being freed this code
	// performs an AddRef/Release while its being relinked.

	pFileHdl->AddRef();			// LOCK WHILE MOVING FILE HDL

	removeFromList( TRUE,
			pFileHdl, &m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);
	insertInList( TRUE, pFileHdl, TRUE,
			&m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);

	pFileHdl->Release();			// UNLOCK NOW THAT MOVE IS DONE!

	// Release the caller's reference to the file handle

	pFileHdl->Release();

	unlockMutex( bMutexAlreadyLocked);
}

/****************************************************************************
Desc:	Remove (close&free) all FileHdl's that have the specified FileId.
		Remove from the avail and used lists.
****************************************************************************/
void F_FileHdlMgr::removeFileHdls(
 	FLMUINT			uiFileId
	)
{
	F_FileHdl *	pFileHdl;
	F_FileHdl *	pNextFileHdl;

	lockMutex( FALSE);

	// Free all matching file handles in the available list.

	pFileHdl = m_pFirstAvail;
	while (pFileHdl)
	{
		pNextFileHdl = pFileHdl->m_pNext;
		if (pFileHdl->m_uiFileId == uiFileId)
		{
			removeFromList( TRUE,
				pFileHdl, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
		}
		pFileHdl = pNextFileHdl;
	}

	// Free all matching file handles in the used list.

	pFileHdl = m_pFirstUsed;
	while (pFileHdl)
	{
		pNextFileHdl = pFileHdl->m_pNext;
		if (pFileHdl->m_uiFileId == uiFileId)
		{
			removeFromList( TRUE,
				pFileHdl, &m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);
		}
		pFileHdl = pNextFileHdl;
	}

	unlockMutex( FALSE);
}

/****************************************************************************
Desc:	Remove all handles from the avail list.
****************************************************************************/
void F_FileHdlMgr::freeAvailList(
	FLMBOOL	bMutexAlreadyLocked)
{
	F_FileHdl *	pFileHdl;
	F_FileHdl *	pNextFileHdl;

	lockMutex( bMutexAlreadyLocked);
	pFileHdl = m_pFirstAvail;
	while (pFileHdl)
	{
		pFileHdl->m_bInList = FALSE;
		pNextFileHdl = pFileHdl->m_pNext;
		pFileHdl->Release();
		pFileHdl = pNextFileHdl;
	}
	m_pFirstAvail = NULL;
	m_pLastAvail = NULL;
	m_uiNumAvail = 0;
	unlockMutex( bMutexAlreadyLocked);
}

/****************************************************************************
Desc:	Remove all handles from the used list.
****************************************************************************/
void F_FileHdlMgr::freeUsedList(
	FLMBOOL	bMutexAlreadyLocked)
{
	F_FileHdl *	pFileHdl;
	F_FileHdl *	pNextFileHdl;

	lockMutex( bMutexAlreadyLocked);
	pFileHdl = m_pFirstUsed;
	while (pFileHdl)
	{
		pFileHdl->m_bInList = FALSE;
		pNextFileHdl = pFileHdl->m_pNext;
		pFileHdl->Release();
		pFileHdl = pNextFileHdl;
	}
	m_pFirstUsed = NULL;
	m_pLastUsed = NULL;
	m_uiNumUsed = 0;
	unlockMutex( bMutexAlreadyLocked);
}

/****************************************************************************
Desc:	Insert a handle into either the avail or used list.
****************************************************************************/
void F_FileHdlMgr::insertInList(
	FLMBOOL			bMutexAlreadyLocked,
	F_FileHdl *		pFileHdl,
	FLMBOOL			bInsertAtEnd,
	F_FileHdl **	ppFirst,
	F_FileHdl **	ppLast,
	FLMUINT *		puiCount
	)
{
	lockMutex( bMutexAlreadyLocked);

	flmAssert( !pFileHdl->m_bInList);

	if (bInsertAtEnd)
	{
		pFileHdl->m_pNext = NULL;
		if ((pFileHdl->m_pPrev = *ppLast) != NULL)
		{
			pFileHdl->m_pPrev->m_pNext = pFileHdl;
		}
		else
		{
			*ppFirst = pFileHdl;
		}
		*ppLast = pFileHdl;
	}
	else
	{
		pFileHdl->m_pPrev = NULL;
		if ((pFileHdl->m_pNext = *ppFirst) != NULL)
		{
			pFileHdl->m_pNext->m_pPrev = pFileHdl;
		}
		else
		{
			*ppLast = pFileHdl;
		}
		*ppFirst = pFileHdl;
	}
	(*puiCount)++;
	pFileHdl->m_bInList = TRUE;
	pFileHdl->AddRef();
	unlockMutex( bMutexAlreadyLocked);
}

/****************************************************************************
Desc:	Remove a handle into either the avail or used list.
****************************************************************************/
void F_FileHdlMgr::removeFromList(
	FLMBOOL			bMutexAlreadyLocked,
	F_FileHdl *		pFileHdl,
	F_FileHdl **	ppFirst,
	F_FileHdl **	ppLast,
	FLMUINT *		puiCount
	)
{
	lockMutex( bMutexAlreadyLocked);

	flmAssert( pFileHdl->m_bInList);
	if (pFileHdl->m_pNext)
	{
		pFileHdl->m_pNext->m_pPrev = pFileHdl->m_pPrev;
	}
	else
	{
		*ppLast = pFileHdl->m_pPrev;
	}
	if (pFileHdl->m_pPrev)
	{
		pFileHdl->m_pPrev->m_pNext = pFileHdl->m_pNext;
	}
	else
	{
		*ppFirst = pFileHdl->m_pNext;
	}
	flmAssert( *puiCount);
	(*puiCount)--;
	pFileHdl->m_bInList = FALSE;
	pFileHdl->Release();
	unlockMutex( bMutexAlreadyLocked);
}

/****************************************************************************
Desc:	Check items in the avail list and if over a certain age then
		remove them from the avail list.  This will cause file handles
		that have been opened for a long time to be closed.  Also added
		code to reduce the total number of file handles if it is more
		than the open threshold.
****************************************************************************/
void F_FileHdlMgr::checkAgedFileHdls(
	FLMUINT	uiMinTimeOpened
	)
{
	FLMUINT	uiTime;
	FLMUINT	uiMaxAvailTicks;

	uiTime = (FLMUINT)FLM_GET_TIMER();

	FLM_SECS_TO_TIMER_UNITS( uiMinTimeOpened, uiMaxAvailTicks);

	lockMutex( FALSE);

	// Loop while the open count is greater than the open threshold.

	while (m_uiNumAvail && (m_uiNumAvail + m_uiNumUsed > m_uiOpenThreshold))
	{

		// Release until the threshold is down.

		releaseOneAvail( TRUE);
	}

	// Reduce all items older than the specified time.

	while (m_pFirstAvail)
	{

		// All file handles are in order of oldest first.
		// m_uiMaxAvailTime may be a zero value.

		if (FLM_ELAPSED_TIME( uiTime, m_pFirstAvail->m_uiAvailTime) <
							uiMaxAvailTicks)
		{
			break;					// All files are newer so we are done.
		}
		removeFromList( TRUE,
			m_pFirstAvail, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
	}

	unlockMutex( FALSE);
}

/****************************************************************************
Desc:	Do a partial copy from one file into another file.
****************************************************************************/
RCODE flmCopyPartial(
	IF_FileHdl *	pSrcFileHdl,			// Source file handle.
	FLMUINT64		ui64SrcOffset,			// Offset to start copying from.
	FLMUINT64		ui64SrcSize,				// Bytes to copy
	IF_FileHdl *	pDestFileHdl,			// Destination file handle
	FLMUINT64		ui64DestOffset,			// Destination start offset.
	FLMUINT64 *		pui64BytesCopiedRV)		// Returns number of bytes copied
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucBuffer = NULL;
	FLMUINT			uiAllocSize = 65536;
	FLMUINT			uiBytesToRead;
	FLMUINT64		ui64CopySize;
	FLMUINT64		ui64FileOffset;
	FLMUINT			uiBytesRead;
	FLMUINT			uiBytesWritten;

	ui64CopySize = ui64SrcSize;
	*pui64BytesCopiedRV = 0;

	// Set the buffer size for use during the file copy

	if( ui64CopySize < uiAllocSize)
	{
		uiAllocSize = (FLMUINT)ui64CopySize;
	}

	// Allocate a buffer

	if( RC_BAD( rc = f_alloc( uiAllocSize, &pucBuffer)))
	{
		goto Exit;
	}

	// Position the file pointers

	if( RC_BAD( rc = pSrcFileHdl->Seek( ui64SrcOffset, XFLM_IO_SEEK_SET,
								&ui64FileOffset)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDestFileHdl->Seek( ui64DestOffset, XFLM_IO_SEEK_SET,
								&ui64FileOffset)))
	{
		goto Exit;
	}

	// Begin copying the data

	while( ui64CopySize)
	{
		if( ui64CopySize > uiAllocSize)
		{
			uiBytesToRead = uiAllocSize;
		}
		else
		{
			uiBytesToRead = (FLMUINT)ui64CopySize;
		}
		
		rc = pSrcFileHdl->Read( XFLM_IO_CURRENT_POS, uiBytesToRead,
										pucBuffer, &uiBytesRead);
										
		if (rc == NE_XFLM_IO_END_OF_FILE)
		{
			rc = NE_XFLM_OK;
		}
		
		if (RC_BAD( rc))
		{
			rc = RC_SET( NE_XFLM_IO_COPY_ERR);
			goto Exit;
		}

		uiBytesWritten = 0;
		if( RC_BAD( rc = pDestFileHdl->Write( XFLM_IO_CURRENT_POS, uiBytesRead,
									pucBuffer, &uiBytesWritten)))
		{
			if (rc == NE_XFLM_IO_DISK_FULL)
			{
				*pui64BytesCopiedRV += uiBytesWritten;
			}
			else
			{
				rc = RC_SET( NE_XFLM_IO_COPY_ERR);
			}
			
			goto Exit;
		}
		
		*pui64BytesCopiedRV += uiBytesWritten;

		if( uiBytesRead < uiBytesToRead)
		{
			rc = RC_SET( NE_XFLM_IO_END_OF_FILE);
			goto Exit;
		}

		ui64CopySize -= uiBytesRead;
	}
	
Exit:

	if (pucBuffer)
	{
		(void)f_free( &pucBuffer);
	}

	return( rc);
}
