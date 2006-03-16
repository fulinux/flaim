//-------------------------------------------------------------------------
// Desc:	File handle class.
// Tabs:	3
//
//		Copyright (c) 1997-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ffilehdl.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

RCODE	gv_CriticalFSError = FERR_OK;

/****************************************************************************
Desc:		Close all used and unused files in the file handle manager.
Note:		This routine will tend to process as much as possible and keep
			error codes around until complete.
****************************************************************************/
RCODE flmCloseAllFiles( void)
{
	RCODE				rc;
	F_FileHdlMgr *	pFileHdlMgr;

	pFileHdlMgr = gv_FlmSysData.pFileHdlMgr;

	// Close all available files.  This will have to be done again
	// because a file could be sent to the avail list between this call
	// the the lockSem() call below.
	
	for(;;)
	{
		if( pFileHdlMgr->ReleaseOneAvail() != FERR_OK )
		{
			break;
		}
	}
	
	// Visit all of the used file handles and unlink them from the used
	// list.  This will cause them to go away when the file handle is
	// released.

	rc = pFileHdlMgr->ReleaseUsedFiles();
	
	// Hold on to return code.	
	
	// Close all available files again in case some used files joined

	for(;;)
	{
		if( pFileHdlMgr->ReleaseOneAvail() != FERR_OK )
		{
			break;
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Quick and easy way to write a string to a file.  The contents of
		pszSourceFile becomes pszData.
****************************************************************************/
RCODE f_filecpy(
	const char *	pszSourceFile,
	const char *	pszData)
{
	RCODE				rc = FERR_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_OK( rc = gv_FlmSysData.pFileSystem->Exists( pszSourceFile)))
	{
		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Delete( pszSourceFile)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Create( 
		pszSourceFile, F_IO_RDWR, &pFileHdl)))
	{
		goto Exit;
	}
	
	{
		FLMUINT 	uiBytesWritten = 0;
		
		if( RC_BAD( rc = pFileHdl->Write( 0, f_strlen( pszData), pszData,
			&uiBytesWritten)))
		{
			goto Exit;
		}		
	}
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Close();
		pFileHdl->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Quick and easy way to append a string to a file.  The contents of
		pszData are appended to pszSourceFile.
****************************************************************************/ 
RCODE f_filecat(
	const char *	pszSourceFile,
	const char *	pszData)
{
	RCODE				rc = FERR_OK;
	F_FileHdl *		pFileHdl = NULL;
	FLMUINT 			uiFileSize = 0;
	FLMUINT 			uiBytesWritten = 0;

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Exists( pszSourceFile)))
	{
		if( rc == FERR_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Create( 
				pszSourceFile, F_IO_RDWR, &pFileHdl)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Open( 
			pszSourceFile, F_IO_RDWR, &pFileHdl)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pFileHdl->Size( &uiFileSize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->Write( uiFileSize, f_strlen( pszData),
		pszData, &uiBytesWritten)))
	{
		goto Exit;
	}
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Close();
		pFileHdl->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Initializes variables
****************************************************************************/
F_FileHdlMgr::F_FileHdlMgr(
	F_MUTEX *	phMutex)		
{
	m_phMutex = phMutex;
	m_uiOpenThreshold = 0xFFFF;
	FLM_SECS_TO_TIMER_UNITS( 30 * 60, m_uiMaxAvailTime);
	m_bIsSetup = FALSE;
	m_uiFileIdCounter = 0;
}

/****************************************************************************
Desc:		Setup values for the File handle manager.
****************************************************************************/
RCODE F_FileHdlMgr::Setup(
	FLMUINT		uiOpenThreshold,
	FLMUINT		uiMaxAvailTime)
{
	RCODE			rc = FERR_OK;

	// Critical section may be needed here because some platforms (we think)
	// may not set and get a variable as an atomic unit.

	m_uiOpenThreshold = uiOpenThreshold;
	m_uiMaxAvailTime = uiMaxAvailTime;
	
	if( !m_bIsSetup)
	{
		rc = m_ListMgr.Setup( m_LNodes, FHM_LNODE_COUNT );
		m_bIsSetup = TRUE;
	}

	return( rc);
}

/****************************************************************************
Desc:		Returns a unique id that can be assigned to a F_FileHdlImp object.	
****************************************************************************/
FLMUINT F_FileHdlMgr::GetUniqueId()
{
	FLMUINT			uiTemp;
	F_MutexRef		MutexRef( m_phMutex );

	MutexRef.Lock();
	uiTemp = ++m_uiFileIdCounter;
	MutexRef.Unlock();

	return( uiTemp);
}

/****************************************************************************
Desc:		Return the next available file handle that matches the uiFileId.
			Remove all old opened file handles that have been available for
			more the the max avail time.  This will help to not keep files
			opened for a long time if they are not used.
****************************************************************************/
RCODE F_FileHdlMgr::FindAvail(
	F_MutexRef *		pMutexRef,
	FLMUINT				uiFileId,
	FLMBOOL				bReadOnlyFlag,
	F_FileHdlImp **	ppFileHdl)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl;
	FLMBOOL				bFound = FALSE;

	*ppFileHdl = NULL;
	pMutexRef->Lock();

	pFileHdl = (F_FileHdlImp *) m_ListMgr.GetItem( FHM_AVAIL_LIST, 0 );
	while( pFileHdl)
	{
		if( (pFileHdl->GetFileId() == uiFileId) && 
			(pFileHdl->IsOpenedReadOnly() == bReadOnlyFlag ))
		{
			// Move this file handle out of the available list into
			// the used list. 
			//
			//	NOTE: To prevent this file handle from being freed this code 
			//	performs an AddRef while its being relinked.  This reference
			//	will be kept for the caller.
			
			pFileHdl->AddRef();
			bFound = TRUE;

			if( RC_OK( rc = pFileHdl->RemoveFromList( FHM_AVAIL_LIST)))
			{
				m_ListMgr.InsertAtFirst( FHM_USED_LIST, pFileHdl);
				*ppFileHdl = pFileHdl;
			}

			// NOTE: DO NOT CALL RELEASE -- Keep reference for caller.
			
			break;
		}

		// Next pFileHdl in the list.

		pFileHdl = (F_FileHdlImp *)pFileHdl->GetNextListItem( FHM_AVAIL_LIST);
	}

	if( !bFound)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	pMutexRef->Unlock();
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlMgr::InsertNew(
	F_MutexRef *		pMutexRef,
	F_FileHdlImp *		pFileHdl)
{
	pMutexRef->Lock();
	m_ListMgr.InsertAtEnd( FHM_USED_LIST, (F_ListItem *)pFileHdl);
	pMutexRef->Unlock();
	return( FERR_OK);
}

/****************************************************************************
Desc:		Make the specified F_FileHdlImp available for someone else to use.
****************************************************************************/
RCODE F_FileHdlMgr::MakeAvailAndRelease(
	F_MutexRef *		pMutexRef,
	F_FileHdlImp *		pFileHdl)
{
	RCODE				rc = FERR_OK;
	FLMINT			iRefCnt;
	
	pFileHdl->SetAvailTime();
	pMutexRef->Lock();

	// NOTE: To prevent this file handle from being freed this code 
	// performs an AddRef/Release while its being relinked. 
			
	pFileHdl->AddRef();

	if( RC_OK( rc = pFileHdl->RemoveFromList( FHM_USED_LIST)))
	{
		m_ListMgr.InsertAtEnd( FHM_AVAIL_LIST, (F_ListItem *) pFileHdl );
	}

	pFileHdl->Release();	

	// Release the caller's reference to the file handle
	
	iRefCnt = pFileHdl->Release();
	flmAssert( iRefCnt == 1);

	pMutexRef->Unlock();
	return( rc);
}

/****************************************************************************
Desc:		Remove the input FileHdl from the avil and used lists.
****************************************************************************/
RCODE F_FileHdlMgr::Remove(
	F_FileHdlImp *		pFileHdl)
{
	RCODE				rc;
	F_MutexRef		MutexRef( m_phMutex );

	MutexRef.Lock();
	rc = pFileHdl->RemoveFromList( FLM_ALL_LISTS);
	MutexRef.Unlock();

	return( rc);
}

/****************************************************************************
Desc:		Remove (close&free) all FileHdl's that have the specified FileId.
			Remove from the avail and used lists.
****************************************************************************/

RCODE F_FileHdlMgr::Remove(
	F_MutexRef *	pMutexRef,
 	FLMUINT			uiFileId,
	FLMBOOL			bFreeUsedFiles)	// Should used files be freed
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl;
	F_FileHdlImp *		pNextFileHdl;

	pMutexRef->Lock();

	// Free all matching file handles in the available list.
	
	for( pFileHdl = (F_FileHdlImp *) m_ListMgr.GetItem( FHM_AVAIL_LIST, 0 );
		  pFileHdl;
		  pFileHdl = pNextFileHdl )
	{
		pNextFileHdl = (F_FileHdlImp *) pFileHdl->GetNextListItem( FHM_AVAIL_LIST );
		if( pFileHdl->GetFileId() == uiFileId )
		{
			// Match found - remove from list.
		
			if( RC_BAD( rc = pFileHdl->RemoveFromList( FHM_AVAIL_LIST )))
				goto Exit;
		}
	}

	if( bFreeUsedFiles == TRUE)
	{
		// Free all matching file handles in the used list.

		for( pFileHdl = (F_FileHdlImp *) m_ListMgr.GetItem( FHM_USED_LIST, 0 );
			  pFileHdl;
			  pFileHdl = pNextFileHdl )
		{
			pNextFileHdl = (F_FileHdlImp *) pFileHdl->GetNextListItem( FHM_USED_LIST );
			if( pFileHdl->GetFileId() == uiFileId )
			{
				// Match found - remove from list.
			
				if( RC_BAD( rc = pFileHdl->RemoveFromList( FHM_USED_LIST )))
					goto Exit;
			}
		}
	}

Exit:

	pMutexRef->Unlock();
	return( rc);
}

/****************************************************************************
Desc:		Check items in the avail list and if over the input certain age then
			remove them from the avail list.  
****************************************************************************/
RCODE F_FileHdlMgr::CheckAgedItems(
	FLMUINT		uiMinTimeOpened)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiMaxAvailTime = m_uiMaxAvailTime;
	F_MutexRef	MutexRef( m_phMutex );

	m_uiMaxAvailTime = uiMinTimeOpened;
	rc = CheckAgedItems( &MutexRef );
	m_uiMaxAvailTime = uiMaxAvailTime;

	return( rc);
}

/****************************************************************************
Desc:		Returns the total number of opened file handles. This includes all
			FileHdls found within both the USED and AVAIL list.
****************************************************************************/
FLMUINT F_FileHdlMgr::GetOpenedFiles( void)
{
	F_MutexRef	MutexRef( m_phMutex);
	FLMUINT		uiTemp;

	MutexRef.Lock();
	uiTemp = m_ListMgr.GetCount( FLM_ALL_LISTS);
	MutexRef.Unlock();

	return uiTemp;
}

/****************************************************************************
Desc:		Release one available file handle.
****************************************************************************/
RCODE F_FileHdlMgr::ReleaseOneAvail(
	F_MutexRef *	pMutexRef)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl;

	pMutexRef->Lock();

	// Free all matching file handles in the available list.
	
	pFileHdl = (F_FileHdlImp *) m_ListMgr.GetItem( FHM_AVAIL_LIST, 0 );

	if( pFileHdl != NULL)
	{
		rc = pFileHdl->RemoveFromList( FHM_AVAIL_LIST );
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
	}
	
	pMutexRef->Unlock();
	return( rc);
}

/****************************************************************************
Desc:		Release ALL used files so that they close when released.
****************************************************************************/
RCODE F_FileHdlMgr::ReleaseUsedFiles( void)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl;
	F_FileHdlImp *		pNextFileHdl;
	F_MutexRef			MutexRef( m_phMutex);

	MutexRef.Lock();

	// Free all matching file handles in the used list.

	for( pFileHdl = (F_FileHdlImp *) m_ListMgr.GetItem( FHM_USED_LIST, 0 );
		  pFileHdl;
		  pFileHdl = pNextFileHdl)
	{
		pNextFileHdl = (F_FileHdlImp *)pFileHdl->GetNextListItem( FHM_USED_LIST );

		if( RC_BAD( rc = pFileHdl->RemoveFromList( FHM_USED_LIST )))
		{
			goto Exit;
		}
	}

Exit:

	MutexRef.Unlock();
	return( rc);
}

/****************************************************************************
Desc:		Check items in the avail list and if over a certain age then
			remove them from the avail list.  This will cause file handles
			that have been opened for a long time to be closed.  Also added
			code to reduce the total number of file handles if it is more
			than the open threshold.
****************************************************************************/
RCODE F_FileHdlMgr::CheckAgedItems(
	F_MutexRef *		pMutexRef)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl;
	FLMUINT				uiTime;
	FLMUINT				uiMaxAvailTicks;
	
	uiTime = (FLMUINT)FLM_GET_TIMER();

	pMutexRef->Lock();

	FLM_SECS_TO_TIMER_UNITS( m_uiMaxAvailTime, uiMaxAvailTicks);

	// Loop while the open count is greater than the open threshold.

	while( m_ListMgr.GetCount( FLM_ALL_LISTS) > m_uiOpenThreshold)
	{
		// Release until the threshold is down.  Only returns FERR_OK
		// if it released one avail handle, otherwise returns FERR_NOT_FOUND.

		if( RC_BAD( rc = ReleaseOneAvail( pMutexRef )))
		{
			if( FERR_NOT_FOUND == rc)
				rc = FERR_OK;
			break;
		}
	}
	
	// Reduce all items older than the specified time.

	for(;;)
	{
		FLMUINT	uiTmp;

		pFileHdl = (F_FileHdlImp *) m_ListMgr.GetItem( FHM_AVAIL_LIST, 0 );
		if( !pFileHdl )
			break;

		// All file handles are in order of oldest first.
		// m_uiMaxAvailTime may be a zero value.

		uiTmp = pFileHdl->GetAvailTime();
		if( uiMaxAvailTicks > FLM_ELAPSED_TIME( uiTime, uiTmp) )
		{
			// All files are newer so we are done.
			
			break;
		}
		
		// Remove the file handle from the list and get first item.

		if( RC_BAD( rc = pFileHdl->RemoveFromList( FHM_AVAIL_LIST)))
		{
			goto Exit;
		}
	}
	
Exit:

	pMutexRef->Unlock();
	return( rc);
}

/****************************************************************************
Desc:		Do a partial copy from one file into another file.
****************************************************************************/
RCODE flmCopyPartial(
	F_FileHdl *		pSrcFileHdl,
	FLMUINT			uiSrcOffset,
	FLMUINT			uiSrcSize,
	F_FileHdl *		pDestFileHdl,
	FLMUINT			uiDestOffset,
	FLMUINT *		puiBytesCopiedRV)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucBuffer = NULL;
	FLMUINT			uiAllocSize = 65536;
	FLMUINT			uiBytesToRead;
	FLMUINT			uiCopySize;
	FLMUINT			uiFileOffset;
	FLMUINT			uiBytesRead;
	FLMUINT			uiBytesWritten;

	uiCopySize = uiSrcSize;
	*puiBytesCopiedRV = 0;

	// Set the buffer size for use during the file copy

	if (uiCopySize < uiAllocSize)
	{
		uiAllocSize = uiCopySize;
	}

	// Allocate a buffer

	if( RC_BAD( rc = f_alloc( uiAllocSize, &pucBuffer)))
	{
		goto Exit;
	}

	// Position the file pointers

	if( RC_BAD( rc = pSrcFileHdl->Seek( uiSrcOffset, F_IO_SEEK_SET,
								&uiFileOffset)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDestFileHdl->Seek( uiDestOffset, F_IO_SEEK_SET,
								&uiFileOffset)))
	{
		goto Exit;
	}

	// Begin copying the data

	while (uiCopySize)
	{
		if (uiCopySize > uiAllocSize)
		{
			uiBytesToRead = uiAllocSize;
		}
		else
		{
			uiBytesToRead = uiCopySize;
		}
		
		rc = pSrcFileHdl->Read( F_IO_CURRENT_POS, uiBytesToRead,
										pucBuffer, &uiBytesRead);
										
		if( rc == FERR_IO_END_OF_FILE)
		{
			rc = FERR_OK;
		}
		
		if( RC_BAD( rc))
		{
			rc = RC_SET( FERR_IO_COPY_ERR);
			goto Exit;
		}

		uiBytesWritten = 0;
		if( RC_BAD( rc = pDestFileHdl->Write( F_IO_CURRENT_POS, uiBytesRead,
									pucBuffer, &uiBytesWritten)))
		{
			if( rc == FERR_IO_DISK_FULL)
			{
				*puiBytesCopiedRV += uiBytesWritten;
			}
			else
			{
				rc = RC_SET( FERR_IO_COPY_ERR);
			}
			
			goto Exit;
		}
		
		*puiBytesCopiedRV += uiBytesWritten;

		if( uiBytesRead < uiBytesToRead)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
			goto Exit;
		}

		uiCopySize -= uiBytesRead;
	}
	
Exit:

	if (pucBuffer)
	{
		(void)f_free( &pucBuffer);
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Setup (Initialize) a FileHdl object
****************************************************************************/
RCODE F_FileHdlImp::Setup(							
	FLMUINT			uiFileId)
{
	RCODE		rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	m_uiFileId = uiFileId;

	if (uiFileId)
	{
		F_ListItem::Setup( gv_FlmSysData.pFileHdlMgr->GetListMgr(),
									m_LNode, FHM_LNODE_COUNT);
	}

Exit:

	return( rc);
}
