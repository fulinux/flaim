//-------------------------------------------------------------------------
// Desc:	Windows I/O
// Tabs:	3
//
//		Copyright (c) 1999-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fwin.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_WIN)

extern RCODE		gv_CriticalFSError;

FSTATIC RCODE _DeleteFile(
	const char *	path);


/***************************************************************************
Desc:	Maps WIN errors to IO errors.
***************************************************************************/
RCODE MapWinErrorToFlaim(
	DWORD		udErrCode,
	RCODE		defaultRc)
{

	/* Switch on passed in error code value */

	switch (udErrCode)
	{
		case ERROR_NOT_ENOUGH_MEMORY:
		case ERROR_OUTOFMEMORY:
			return( RC_SET( FERR_MEM));
		case ERROR_BAD_NETPATH:
		case ERROR_BAD_PATHNAME:
		case ERROR_DIRECTORY:
		case ERROR_FILE_NOT_FOUND:
		case ERROR_INVALID_DRIVE:
		case ERROR_INVALID_NAME:
		case ERROR_NO_NET_OR_BAD_PATH:
		case ERROR_PATH_NOT_FOUND:
			return( RC_SET( FERR_IO_PATH_NOT_FOUND));

		case ERROR_ACCESS_DENIED:
		case ERROR_SHARING_VIOLATION:
		case ERROR_FILE_EXISTS:
		case ERROR_ALREADY_EXISTS:
			return( RC_SET( FERR_IO_ACCESS_DENIED));

		case ERROR_BUFFER_OVERFLOW:
		case ERROR_FILENAME_EXCED_RANGE:
			return( RC_SET( FERR_IO_PATH_TOO_LONG));

		case ERROR_DISK_FULL:
		case ERROR_HANDLE_DISK_FULL:
			return( RC_SET( FERR_IO_DISK_FULL));

		case ERROR_CURRENT_DIRECTORY:
		case ERROR_DIR_NOT_EMPTY:
			return( RC_SET( FERR_IO_MODIFY_ERR));

		case ERROR_DIRECT_ACCESS_HANDLE:
		case ERROR_INVALID_HANDLE:
		case ERROR_INVALID_TARGET_HANDLE:
			return( RC_SET( FERR_IO_BAD_FILE_HANDLE));

		case ERROR_HANDLE_EOF:
			return( RC_SET( FERR_IO_END_OF_FILE));

		case ERROR_OPEN_FAILED:
			return( RC_SET( FERR_IO_OPEN_ERR));

		case ERROR_CANNOT_MAKE:
			return( RC_SET( FERR_IO_PATH_CREATE_FAILURE));

		case ERROR_LOCK_FAILED:
		case ERROR_LOCK_VIOLATION:
			return( RC_SET( FERR_IO_FILE_LOCK_ERR));

		case ERROR_NEGATIVE_SEEK:
		case ERROR_SEEK:
		case ERROR_SEEK_ON_DEVICE:
			return( RC_SET( FERR_IO_SEEK_ERR));

		case ERROR_NO_MORE_FILES:
		case ERROR_NO_MORE_SEARCH_HANDLES:
			return( RC_SET( FERR_IO_NO_MORE_FILES));

		case ERROR_TOO_MANY_OPEN_FILES:
			return( RC_SET( FERR_IO_TOO_MANY_OPEN_FILES));

		case NO_ERROR:
			return( FERR_OK);

		case ERROR_DISK_CORRUPT:
		case ERROR_DISK_OPERATION_FAILED:
		case ERROR_FILE_CORRUPT:
		case ERROR_FILE_INVALID:
		case ERROR_NOT_SAME_DEVICE:
		case ERROR_IO_DEVICE:
		default:
			return( RC_SET( defaultRc));

   }
}

/****************************************************************************
Desc:		Default Constructor for F_FileHdl class
****************************************************************************/
F_FileHdlImp::F_FileHdlImp()
{
	m_FileHandle = INVALID_HANDLE_VALUE;
	m_uiBlockSize = 0;
	m_uiBytesPerSector = 0;
	m_uiNotOnSectorBoundMask = 0;
	m_uiGetSectorBoundMask = 0;
	m_bDoDirectIO = FALSE;
	m_uiExtendSize = 0;
	m_uiMaxAutoExtendSize = gv_FlmSysData.uiMaxFileSize;
	m_pucAlignedBuff = NULL;
	m_uiAlignedBuffSize = 0;
	m_uiCurrentPos = 0;
	m_bCanDoAsync = FALSE;			// Change to TRUE when we want to do async writes.
	m_Overlapped.hEvent = NULL;
}

/****************************************************************************
Desc:		Destructor for F_FileHdl class
****************************************************************************/
F_FileHdlImp::~F_FileHdlImp()
{
	// Close file if it was open.

	if( m_bFileOpened)
	{
		(void)Close();
	}
	if (m_pucAlignedBuff)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		flmAssert( gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated >=
							m_uiAlignedBuffSize);
		gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated -=
			m_uiAlignedBuffSize;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);

		(void)VirtualFree( m_pucAlignedBuff, 0, MEM_RELEASE);
		m_pucAlignedBuff = NULL;
		m_uiAlignedBuffSize = 0;
	}
	if (m_Overlapped.hEvent)
	{
		CloseHandle( m_Overlapped.hEvent);
	}
}

/***************************************************************************
Desc:		Open or create a file.
***************************************************************************/
RCODE F_FileHdlImp::OpenOrCreate(
	const char *	pFileName,
   FLMUINT			uiAccess,
	FLMBOOL			bCreateFlag)
{
	char			szSaveFileName[ F_PATH_MAX_SIZE];
	RCODE			rc = FERR_OK;
	DWORD			udAccessMode = 0;
	DWORD			udShareMode = 0;
	DWORD			udCreateMode = 0;
	DWORD			udAttrFlags = 0;
	DWORD			udErrCode;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	m_bDoDirectIO = (uiAccess & F_IO_DIRECT) ? TRUE : FALSE;

	if( uiAccess & F_IO_DELETE_ON_CLOSE)
	{
		if( !m_pszIoPath)
		{
			if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszIoPath)))
			{
				goto Exit;
			}
		}
		f_strcpy( m_pszIoPath, pFileName);
		m_bDeleteOnClose = TRUE;
	}
	else
	{
		m_bDeleteOnClose = FALSE;
	}

	// Save the file name in case we have to create the directory

	if ((bCreateFlag) && (uiAccess & F_IO_CREATE_DIR))
	{
		f_strcpy( (char *)(&szSaveFileName), (char *)pFileName);
	}

	// If doing direct IO, need to get the sector size.

	if (m_bDoDirectIO)
	{
		if (!m_uiBlockSize)
		{
			m_bDoDirectIO = FALSE;
		}
		else
		{
			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->GetSectorSize( 
				pFileName, &m_uiBytesPerSector)))
			{
				goto Exit;
			}
			m_uiNotOnSectorBoundMask = m_uiBytesPerSector - 1;
			m_uiGetSectorBoundMask = ~m_uiNotOnSectorBoundMask;

			// Can't do direct IO if the block size isn't a multiple of
			// the sector size.

			if (m_uiBlockSize < m_uiBytesPerSector ||
				 m_uiBlockSize % m_uiBytesPerSector != 0)
			{
				m_bDoDirectIO = FALSE;
			}
		}
	}

	// Only enable asynchronous writes if direct I/O is enabled.

	if (m_bDoDirectIO)
	{
		m_bCanDoAsync = gv_FlmSysData.bOkToDoAsyncWrites;
	}

	/*
	Set up the file characteristics requested by caller.
	*/

   if (uiAccess & F_IO_SH_DENYRW)
   {
      udShareMode = 0;
      uiAccess &= ~F_IO_SH_DENYRW;
   }
   else if (uiAccess & F_IO_SH_DENYWR)
   {
      udShareMode = FILE_SHARE_READ;
      uiAccess &= ~F_IO_SH_DENYWR;
   }
	else if (uiAccess & F_IO_SH_DENYNONE)
   {
      udShareMode = (FILE_SHARE_READ | FILE_SHARE_WRITE);
      uiAccess &= ~F_IO_SH_DENYNONE;
   }
	else
	{
      udShareMode = (FILE_SHARE_READ | FILE_SHARE_WRITE);
	}

	/* Begin setting the CreateFile flags and fields */

   udAttrFlags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS;
	if (m_bDoDirectIO)
	{
		udAttrFlags |= FILE_FLAG_NO_BUFFERING;
	}
	if (m_bCanDoAsync)
	{
		udAttrFlags |= FILE_FLAG_OVERLAPPED;
	}

   if (bCreateFlag)
   {
   	if (uiAccess & F_IO_EXCL)
		{
	  		udCreateMode = CREATE_NEW;
		}
		else
		{
		   udCreateMode = CREATE_ALWAYS;
		}
   }
	else
   {
		udCreateMode = OPEN_EXISTING;
   }

   udAccessMode = GENERIC_READ | GENERIC_WRITE;

   if ( (!bCreateFlag) && (uiAccess & F_IO_RDONLY) )
      udAccessMode = GENERIC_READ;

Retry_Create:

	/* Try to create or open the file */

	if( (m_FileHandle = CreateFile( (LPCTSTR)pFileName, udAccessMode,
					udShareMode, NULL, udCreateMode,
					udAttrFlags, NULL)) == INVALID_HANDLE_VALUE)
	{
		udErrCode = GetLastError();
		if ((udErrCode == ERROR_PATH_NOT_FOUND) && (uiAccess & F_IO_CREATE_DIR))
		{
			char	szTemp[ F_PATH_MAX_SIZE];
			char	ioDirPath[ F_PATH_MAX_SIZE];

			uiAccess &= ~F_IO_CREATE_DIR;

			/* Remove the file name for which we are creating the directory. */

			if( RC_OK( f_pathReduce( szSaveFileName, ioDirPath, szTemp)))
			{
				if( RC_OK( rc = gv_FlmSysData.pFileSystem->CreateDir( ioDirPath)))
				{
					goto Retry_Create;
				}
				else
				{
					goto Exit;
				}
			}
		}
		rc = MapWinErrorToFlaim( udErrCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)(m_bDoDirectIO
												? (RCODE)FERR_DIRECT_CREATING_FILE
												: (RCODE)FERR_CREATING_FILE)
								  : (RCODE)(m_bDoDirectIO
												? (RCODE)FERR_DIRECT_OPENING_FILE
												: (RCODE)FERR_OPENING_FILE)));
		goto Exit;
	}
Exit:
	if (RC_BAD( rc))
	{
		m_FileHandle = INVALID_HANDLE_VALUE;
	}
   return( rc );
}

/****************************************************************************
Desc:		Create a file 
****************************************************************************/
RCODE F_FileHdlImp::Create(
	const char *	pszIoPath,
	FLMUINT			uiIoFlags )
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bFileOpened == FALSE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = OpenOrCreate( pszIoPath, uiIoFlags, TRUE)))
	{
		goto Exit;
	}

	m_bFileOpened = TRUE;
	m_uiCurrentPos = 0;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlImp::CreateUnique(
	char *			pszIoPath,
	const char *	pszFileExtension,
	FLMUINT			uiIoFlags)
{
	RCODE			rc = FERR_OK;
	char *		pszTmp;
	FLMBOOL		bModext = TRUE;
	FLMUINT		uiBaseTime = 0;
	char			ucHighByte = 0;
	char			szFileName[ F_FILENAME_SIZE];
	char			szDirPath[ F_PATH_MAX_SIZE];
	char			szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT		uiCount;

	szFileName[0] = '\0';
	szTmpPath[0] = '\0';
	flmAssert( !m_bFileOpened);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	f_strcpy( szDirPath, pszIoPath);

   // Search backwards replacing trailing spaces with NULLs.

	pszTmp = (char *) szDirPath;
	pszTmp += (f_strlen( pszTmp) - 1);
	while( pszTmp >= (char *) szDirPath && (*pszTmp == 0x20))
	{
		*pszTmp = 0;
		pszTmp--;
	}

	if (pszTmp >= (char *) szDirPath && *pszTmp != '\\')
	{
		pszTmp++;
		*pszTmp++ = '\\';
	}
	else
	{
		pszTmp++;
	}
	*pszTmp = 0;

	if ((pszFileExtension) && (f_strlen( pszFileExtension) >= 3))
	{
		bModext = FALSE;
	}

	uiCount = 0;
	do
	{
		f_pathCreateUniqueName( &uiBaseTime,  szFileName, pszFileExtension,
										&ucHighByte, bModext);

		// Need to strcpy to the buffer b/c it is uninitialized

		f_strcpy( szTmpPath, szDirPath);
		f_pathAppend( szTmpPath, szFileName);

		rc = Create( szTmpPath, uiIoFlags | F_IO_EXCL);

		if (rc == FERR_IO_DISK_FULL)
		{
			(void)_DeleteFile( szTmpPath);
			goto Exit;
		}

		if ((rc == FERR_IO_PATH_NOT_FOUND) || (rc == FERR_IO_INVALID_PASSWORD))
		{
			goto Exit;
		}
	} while ((rc != FERR_OK) && (uiCount++ < 10));

   // Check if the path was created

   if ((uiCount >= 10) && (rc != FERR_OK))
   {
		rc = RC_SET( FERR_IO_PATH_CREATE_FAILURE);
		goto Exit;
   }

	m_bFileOpened = TRUE;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

	// Created file name needs to be returned.

	f_strcpy( pszIoPath, szTmpPath);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Open a file
****************************************************************************/
RCODE F_FileHdlImp::Open(
	const char *	pszIoPath,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = FERR_OK;

	flmAssert( m_bFileOpened == FALSE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	// Loop on error open conditions.
	
	for(;;)
	{
		if( RC_OK( rc = OpenOrCreate( pszIoPath, uiIoFlags, FALSE)))
			break;

		if( rc != FERR_IO_TOO_MANY_OPEN_FILES )
		{
			goto Exit;
		}

		// If for some reason we cannot open the file, then
		// try to close some other file handle in the list.

		if( RC_BAD( gv_FlmSysData.pFileHdlMgr->ReleaseOneAvail()))
		{
			goto Exit;
		}
	}

	m_bFileOpened = TRUE;
	m_uiCurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & F_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Close a file
****************************************************************************/
RCODE F_FileHdlImp::Close( void)
{
	FLMBOOL	bDeleteAllowed = TRUE;
	RCODE		rc = FERR_OK;

	if( !m_bFileOpened)
	{
		goto Exit;
	}

#ifdef FLM_DEBUG
	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		bDeleteAllowed = FALSE;
	}
#endif
	if (!CloseHandle( m_FileHandle))
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_CLOSING_FILE);
		goto Exit;
	}

	m_FileHandle = INVALID_HANDLE_VALUE;
	m_bFileOpened = m_bOpenedReadOnly = m_bOpenedExclusive = FALSE;

	if( m_bDeleteOnClose)
	{
		flmAssert( m_pszIoPath );

		if( bDeleteAllowed)
		{
			(void)_DeleteFile( m_pszIoPath);
		}
		m_bDeleteOnClose = FALSE;
	}

	if( m_pszIoPath)
	{
		f_free( &m_pszIoPath);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Does nothing.
****************************************************************************/
RCODE F_FileHdlImp::Flush()
{
	RCODE	rc = FERR_OK;

	if (m_bDoDirectIO)
	{
		rc = GET_FS_ERROR();
	}
	else
	{
		if (!FlushFileBuffers( m_FileHandle))
  		{
			rc = MapWinErrorToFlaim( GetLastError(), FERR_FLUSHING_FILE);
		}
	}
	return( rc);
}

/****************************************************************************
Desc:		Allocate an aligned buffer.
****************************************************************************/
RCODE F_FileHdlImp::AllocAlignBuffer( void)
{
	RCODE	rc = FERR_OK;

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = RoundToNextSector( 64 * 1024);
	if ((m_pucAlignedBuff = (FLMBYTE *)VirtualAlloc( NULL,
								(DWORD)m_uiAlignedBuffSize,
								MEM_COMMIT, PAGE_READWRITE)) == NULL)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_MEM);
		goto Exit;
	}
	f_mutexLock( gv_FlmSysData.hShareMutex);
	gv_FlmSysData.SCacheMgr.Usage.uiTotalBytesAllocated +=
		m_uiAlignedBuffSize;
	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Position and do a single read operation.
****************************************************************************/
RCODE F_FileHdlImp::DoOneRead(
	DWORD			udReadOffset,
	DWORD			udBytesToRead,
	LPVOID		pvReadBuffer,
	LPDWORD		pudBytesRead
	)
{
	RCODE				rc = FERR_OK;
	OVERLAPPED *	pOverlapped;
	LONG				lDummy;
	DWORD				udErr;
	
	// Position the file to the specified offset.

	if (!m_bCanDoAsync)
	{
		lDummy = 0;
		if (SetFilePointer( m_FileHandle, (LONG)udReadOffset,
					&lDummy, FILE_BEGIN) == 0xFFFFFFFF)
		{
			rc = MapWinErrorToFlaim( GetLastError(),
						FERR_POSITIONING_IN_FILE);
			goto Exit;
		}
		pOverlapped = NULL;
	}
	else
	{
		if (!m_Overlapped.hEvent)
		{
			if ((m_Overlapped.hEvent = CreateEvent( NULL, TRUE,
													FALSE, NULL)) == NULL)
			{
				rc = MapWinErrorToFlaim( GetLastError(),
								FERR_SETTING_UP_FOR_READ);
				goto Exit;
			}
		}
		pOverlapped = &m_Overlapped;
		pOverlapped->Offset = udReadOffset;
		pOverlapped->OffsetHigh = 0;
		if (!ResetEvent( pOverlapped->hEvent))
		{
			rc = MapWinErrorToFlaim( GetLastError(),
						FERR_SETTING_UP_FOR_READ);
			goto Exit;
		}
	}

	// Do the read

	if (!ReadFile( m_FileHandle, pvReadBuffer,
						udBytesToRead, pudBytesRead, pOverlapped))
	{
		udErr = GetLastError();
		if (udErr == ERROR_IO_PENDING && m_bCanDoAsync)
		{
			if (!GetOverlappedResult( m_FileHandle, pOverlapped,
						pudBytesRead, TRUE))
			{
				rc = MapWinErrorToFlaim( GetLastError(), FERR_READING_FILE);
				goto Exit;
			}
		}
		else
		{
			rc = MapWinErrorToFlaim( udErr, FERR_READING_FILE);
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Read from a file - reads using aligned buffers and offsets - only
			sector boundaries
****************************************************************************/
RCODE F_FileHdlImp::DirectRead(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
	FLMBOOL		bBuffHasFullSectors,
   FLMUINT *	puiBytesReadRV)
{
	RCODE				rc = FERR_OK;
	DWORD				udBytesRead;
	FLMBYTE *		pucReadBuffer;
	FLMBYTE *		pucDestBuffer;
	FLMUINT			uiMaxBytesToRead;
	FLMBOOL			bHitEOF;
	
	flmAssert( m_bFileOpened);

	if( puiBytesReadRV)
	{
		*puiBytesReadRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiReadOffset == F_IO_CURRENT_POS)
		uiReadOffset = m_uiCurrentPos;

	// This loop does multiple reads (if necessary) to get all of the
	// data.  It uses aligned buffers and reads at sector offsets.

	pucDestBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{

		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if ((uiReadOffset & m_uiNotOnSectorBoundMask) ||
			 (((FLMUINT)pucDestBuffer) & m_uiNotOnSectorBoundMask) ||
			 ((uiBytesToRead & m_uiNotOnSectorBoundMask) &&
			  (!bBuffHasFullSectors)))
		{
			if (!m_pucAlignedBuff)
			{
				if (RC_BAD( rc = AllocAlignBuffer()))
				{
					goto Exit;
				}
			}
			pucReadBuffer = m_pucAlignedBuff;

			// Must read enough bytes to cover all of the sectors that
			// contain the data we are trying to read.  The value of
			// (uiReadOffset & m_uiNotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round that up to the next sector
			// to get the total number of bytes we are going to read.

			uiMaxBytesToRead = RoundToNextSector( uiBytesToRead +
									  (uiReadOffset & m_uiNotOnSectorBoundMask));

			// Can't read more than the aligned buffer will hold.

			if (uiMaxBytesToRead > m_uiAlignedBuffSize)
			{
				uiMaxBytesToRead = m_uiAlignedBuffSize;
			}
		}
		else
		{
			uiMaxBytesToRead = RoundToNextSector( uiBytesToRead);
			flmAssert( uiMaxBytesToRead >= uiBytesToRead);
			pucReadBuffer = pucDestBuffer;
		}

		bHitEOF = FALSE;
		if (RC_BAD( rc = DoOneRead(
										(DWORD)TruncateToPrevSector( uiReadOffset),
										 (DWORD)uiMaxBytesToRead,
										 (LPVOID)pucReadBuffer,
										 &udBytesRead)))
		{
			goto Exit;
		}
		if (udBytesRead < (DWORD)uiMaxBytesToRead)
		{
			bHitEOF = TRUE;
		}

		// If the offset we want to read from is not on a sector
		// boundary, increment the read buffer pointer to the
		// offset where the data we need starts and decrement the
		// bytes read by the difference between the start of the
		// sector and the actual read offset.

		if (uiReadOffset & m_uiNotOnSectorBoundMask)
		{
			pucReadBuffer += (uiReadOffset & m_uiNotOnSectorBoundMask);
			flmAssert( (FLMUINT)udBytesRead >= m_uiBytesPerSector);
			udBytesRead -= (DWORD)(uiReadOffset & m_uiNotOnSectorBoundMask);
		}

		// If bytes read is more than we actually need, truncate it back
		// so that we only copy what we actually need.

		if ((FLMUINT)udBytesRead > uiBytesToRead)
		{
			udBytesRead = (DWORD)uiBytesToRead;
		}
		uiBytesToRead -= (FLMUINT)udBytesRead;
		if( puiBytesReadRV)
		{
			(*puiBytesReadRV) += (FLMUINT)udBytesRead;
		}
		m_uiCurrentPos = uiReadOffset + (FLMUINT)udBytesRead;

		// If using a different buffer for reading, copy the
		// data read into the destination buffer.

		if (pucDestBuffer != pucReadBuffer)
		{
			f_memcpy( pucDestBuffer, pucReadBuffer, udBytesRead);
		}
		if (!uiBytesToRead)
			break;

		// Still more to read - did we hit EOF above?

		if (bHitEOF)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
			break;
		}
		pucDestBuffer += udBytesRead;
		uiReadOffset += (FLMUINT)udBytesRead;
	}

Exit:
	return( rc );
}

/****************************************************************************
Desc:		Read from a file
****************************************************************************/
RCODE F_FileHdlImp::Read(
	FLMUINT		uiReadOffset,
	FLMUINT		uiBytesToRead,	
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE			rc = FERR_OK;
	DWORD			udBytesRead;
	
	// Do the direct IO call if enabled.

	if (m_bDoDirectIO)
	{
		rc = DirectRead( uiReadOffset, uiBytesToRead,
									pvBuffer, FALSE, puiBytesReadRV);
		goto Exit;
	}

	// If not doing direct IO, a single read call will do.

	flmAssert( m_bFileOpened);
	if( puiBytesReadRV)
	{
		*puiBytesReadRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiReadOffset == F_IO_CURRENT_POS)
		uiReadOffset = m_uiCurrentPos;

	if (RC_BAD( rc = DoOneRead( (DWORD)uiReadOffset,
										 (DWORD)uiBytesToRead,
										 (LPVOID)pvBuffer,
										 &udBytesRead)))
	{
		goto Exit;
	}

	if( puiBytesReadRV)
	{
		*puiBytesReadRV = (FLMUINT)udBytesRead;
	}

	m_uiCurrentPos = uiReadOffset + (FLMUINT)udBytesRead;

	if (udBytesRead < (DWORD)uiBytesToRead)
	{
		rc = RC_SET( FERR_IO_END_OF_FILE);
		goto Exit;
	}

Exit:
	return( rc );
}

/****************************************************************************
Desc:		Sets current position of file.
****************************************************************************/
RCODE F_FileHdlImp::Seek(
	FLMUINT		uiOffset,
	FLMINT		iWhence,
	FLMUINT *	puiNewOffset)			// [out] new file offset
{
	RCODE	rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	switch (iWhence)
	{
		case F_IO_SEEK_CUR:
			m_uiCurrentPos += uiOffset;
			break;
		case F_IO_SEEK_SET:
			m_uiCurrentPos = uiOffset;
			break;
		case F_IO_SEEK_END:
			rc = Size( &m_uiCurrentPos );
			if ( rc )
			{
				goto Exit;
			}
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
	}
	*puiNewOffset = m_uiCurrentPos;
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Return the size of the file
****************************************************************************/
RCODE F_FileHdlImp::Size(
	FLMUINT *	puiSize)
{
	RCODE		rc;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if ((*puiSize = (FLMUINT)GetFileSize( m_FileHandle, NULL )) ==
											0xFFFFFFFF)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_GETTING_FILE_SIZE);
		goto Exit;
	}
Exit:
	return( rc );
}

/****************************************************************************
Desc:		Returns m_uiCurrentPos
****************************************************************************/
RCODE F_FileHdlImp::Tell(
	FLMUINT *	puiOffset)
{
	RCODE	rc;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	*puiOffset = m_uiCurrentPos;

Exit:
	
	return( rc );
}

/****************************************************************************
Desc:		Truncate the file to the indicated size
WARNING: Direct IO methods are calling this method.  Make sure that all changes
			to this method work in direct IO mode.
****************************************************************************/
RCODE F_FileHdlImp::Truncate(
	FLMUINT	uiSize)
{
	RCODE		rc = FERR_OK;
	LONG		lDummy;

	flmAssert( m_bFileOpened == TRUE);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	// Position the file to the nearest sector below the read offset.

	lDummy = 0;
	if (SetFilePointer( m_FileHandle,
				(LONG)uiSize, &lDummy, FILE_BEGIN) == 0xFFFFFFFF)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_POSITIONING_IN_FILE);
		goto Exit;
	}

   // Set the new file size.

	if (!SetEndOfFile( m_FileHandle))
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_TRUNCATING_FILE);
		goto Exit;
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Handles when a file is extended in direct IO mode.  May extend the
			file some more.  Will always call FlushFileBuffers to ensure that
			the new file size gets written out.
****************************************************************************/
RCODE F_FileHdlImp::extendFile(
	FLMUINT	uiEndOfLastWrite,	// Must be on a sector boundary
	FLMUINT	uiMaxBytesToExtend,
	FLMBOOL	bFlush
	)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiTotalBytesToExtend;
	FLMUINT			uiBytesToWrite;
	DWORD				udBytesWritten;
	LONG				lDummy;
	DWORD				udErr;
	OVERLAPPED *	pOverlapped;

	if ((uiTotalBytesToExtend = uiMaxBytesToExtend) != 0)
	{
		if (uiEndOfLastWrite > m_uiMaxAutoExtendSize)
		{
			uiTotalBytesToExtend = 0;
		}
		else
		{

			// Don't extend beyond maximum file size.

			if (m_uiMaxAutoExtendSize - uiEndOfLastWrite < uiTotalBytesToExtend)
			{
				uiTotalBytesToExtend = m_uiMaxAutoExtendSize - uiEndOfLastWrite;
			}

			// If the extend size is not on a sector boundary, round it down.

			uiTotalBytesToExtend = TruncateToPrevSector( uiTotalBytesToExtend);
		}
	}

	if (uiTotalBytesToExtend)
	{

		// Allocate an aligned buffer if we haven't already.

		if (!m_pucAlignedBuff)
		{
			if (RC_BAD( rc = AllocAlignBuffer()))
			{
				goto Exit;
			}
		}
		f_memset( m_pucAlignedBuff, 0, m_uiAlignedBuffSize);
	}

	// Extend the file until we run out of bytes to write.

	while (uiTotalBytesToExtend)
	{
		if ((uiBytesToWrite = m_uiAlignedBuffSize) > uiTotalBytesToExtend)
		{
			uiBytesToWrite = uiTotalBytesToExtend;
		}
		if (!m_bCanDoAsync)
		{
			lDummy = 0;
			if (SetFilePointer( m_FileHandle, (LONG)uiEndOfLastWrite,
						&lDummy, FILE_BEGIN) == 0xFFFFFFFF)
			{
				rc = MapWinErrorToFlaim( GetLastError(),
							FERR_POSITIONING_IN_FILE);
				goto Exit;
			}
			pOverlapped = NULL;
		}
		else
		{
			pOverlapped = &m_Overlapped;
			if (!pOverlapped->hEvent)
			{
				if ((pOverlapped->hEvent = CreateEvent( NULL, TRUE,
														FALSE, NULL)) == NULL)
				{
					rc = MapWinErrorToFlaim( GetLastError(),
								FERR_SETTING_UP_FOR_WRITE);
					goto Exit;
				}
			}
			pOverlapped->Offset = (DWORD)uiEndOfLastWrite;
			pOverlapped->OffsetHigh = 0;
			if (!ResetEvent( pOverlapped->hEvent))
			{
				rc = MapWinErrorToFlaim( GetLastError(),
								FERR_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
		}

		// Do the write

		if (!WriteFile( m_FileHandle, (LPVOID)m_pucAlignedBuff,
							(DWORD)uiBytesToWrite, &udBytesWritten,
						  pOverlapped))
		{
			udErr = GetLastError();
			if (udErr == ERROR_IO_PENDING && m_bCanDoAsync)
			{
				if (!GetOverlappedResult( m_FileHandle, pOverlapped,
							&udBytesWritten, TRUE))
				{
					rc = MapWinErrorToFlaim( GetLastError(),
								FERR_WRITING_FILE);

				}
			}
			else
			{
				rc = MapWinErrorToFlaim( udErr, FERR_WRITING_FILE);
			}
		}
		if (RC_BAD( rc))
		{

			// Don't care if it is a disk full error, because
			// extending the file is optional work.

			if (rc == FERR_IO_DISK_FULL)
			{
				rc = FERR_OK;
				break;
			}
			goto Exit;
		}

		// NO more room on disk, but that's OK because we were only
		// extending the file beyond where it needed to be.  If that
		// fails, we will just flush what we have done so far.

		if (udBytesWritten < (DWORD)uiBytesToWrite)
		{
			break;
		}
		uiTotalBytesToExtend -= uiBytesToWrite;
		uiEndOfLastWrite += uiBytesToWrite;
	}

	// Flush the file buffers to ensure that the file size gets written
	// out.

	if (bFlush)
	{
		if (!FlushFileBuffers( m_FileHandle))
  		{
			rc = MapWinErrorToFlaim( GetLastError(), FERR_FLUSHING_FILE);
			goto Exit;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Write to a file using direct IO
****************************************************************************/
RCODE F_FileHdlImp::DirectWrite(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
   const void *	pvBuffer,
	F_IOBuffer *	pBufferObj,
	FLMBOOL			bBuffHasFullSectors,
	FLMBOOL			bZeroFill,
   FLMUINT *		puiBytesWrittenRV)
{
	RCODE				rc = FERR_OK;
	DWORD				udBytesRead;
	DWORD				udBytesWritten;
	FLMBYTE *		pucWriteBuffer;
	FLMBYTE *		pucSrcBuffer;
	FLMUINT			uiMaxBytesToWrite;
	FLMUINT			uiCurrFileSize;
	FLMUINT			uiBytesBeingOutput;
	OVERLAPPED *	pOverlapped;
	LONG				lDummy;
	DWORD				udErr;
	FLMBOOL			bExtendFile = FALSE;
	FLMBOOL			bDoAsync = (pBufferObj != NULL) 
										? TRUE 
										: FALSE;
	FLMBOOL			bDidAsync = FALSE;
	FLMUINT			uiLastWriteOffset;
	FLMUINT			uiLastWriteSize;
	
	flmAssert( m_bFileOpened);

#ifdef FLM_DEBUG
	if (bDoAsync)
	{
		flmAssert( m_bCanDoAsync);
	}
#endif

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}

	// Determine if the write will extend the file beyond its
	// current size.  If so, we will need to call FlushFileBuffers

	if ((uiCurrFileSize = (FLMUINT)GetFileSize( m_FileHandle, NULL )) ==
											0xFFFFFFFF)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_GETTING_FILE_SIZE);
		goto Exit;
	}
	if (uiWriteOffset + uiBytesToWrite > uiCurrFileSize &&
		 m_uiExtendSize != (FLMUINT)(~0))
	{
		bExtendFile = TRUE;

		if (uiWriteOffset > uiCurrFileSize)
		{

			// Fill in the empty space.

			if (RC_BAD( rc = extendFile( uiCurrFileSize,
									RoundToNextSector( uiWriteOffset - uiCurrFileSize),
									FALSE)))
			{
				goto Exit;
			}
		}

		// Can't do asynchronous if we are going to extend the file.

		bDoAsync = FALSE;
	}

	// This loop is for direct IO - must make sure we use
	// aligned buffers.

	pucSrcBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{

		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if ((uiWriteOffset & m_uiNotOnSectorBoundMask) ||
			 (((FLMUINT)pucSrcBuffer) & m_uiNotOnSectorBoundMask) ||
			 ((uiBytesToWrite & m_uiNotOnSectorBoundMask) &&
			  (!bBuffHasFullSectors)))
		{

			// Cannot be using a temporary write buffer if we are doing
			// asynchronous writes!

			flmAssert( !bDoAsync || !m_bCanDoAsync);
			if (!m_pucAlignedBuff)
			{
				if (RC_BAD( rc = AllocAlignBuffer()))
				{
					goto Exit;
				}
			}
			pucWriteBuffer = m_pucAlignedBuff;

			// Must write enough bytes to cover all of the sectors that
			// contain the data we are trying to write out.  The value of
			// (uiWriteOffset & m_uiNotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round to the next sector to get the
			// total number of bytes we are going to write.

			uiMaxBytesToWrite = RoundToNextSector( uiBytesToWrite +
									  (uiWriteOffset & m_uiNotOnSectorBoundMask));

			// Can't write more than the aligned buffer will hold.

			if (uiMaxBytesToWrite > m_uiAlignedBuffSize)
			{
				uiMaxBytesToWrite = m_uiAlignedBuffSize;
				uiBytesBeingOutput = uiMaxBytesToWrite -
										(uiWriteOffset & m_uiNotOnSectorBoundMask);
			}
			else
			{
				uiBytesBeingOutput = uiBytesToWrite;
			}

			// If the write offset is not on a sector boundary, we must
			// read at least the first sector into the buffer.

			if (uiWriteOffset & m_uiNotOnSectorBoundMask)
			{

				// Read the first sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if (RC_BAD( rc = DoOneRead(
										(DWORD)TruncateToPrevSector( uiWriteOffset),
										(DWORD)m_uiBytesPerSector,
										(LPVOID)pucWriteBuffer,
										&udBytesRead)))
				{
					goto Exit;
				}
			}

			// If we are writing more than one sector, and the last sector's
			// worth of data we are writing out is only a partial sector,
			// we must read in this sector as well.

			if ((uiMaxBytesToWrite > m_uiBytesPerSector) &&
				 (uiMaxBytesToWrite > uiBytesToWrite) &&
				 (!bBuffHasFullSectors))
			{

				// Read the last sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if (RC_BAD( rc = DoOneRead( 
								(DWORD)(TruncateToPrevSector( uiWriteOffset)) +
								(DWORD)(uiMaxBytesToWrite - m_uiBytesPerSector),
								(DWORD)m_uiBytesPerSector,
								(LPVOID)(&pucWriteBuffer [uiMaxBytesToWrite - m_uiBytesPerSector]),
								&udBytesRead)))
				{
					if (rc == FERR_IO_END_OF_FILE)
					{
						rc = FERR_OK;
						f_memset( &pucWriteBuffer [uiMaxBytesToWrite - m_uiBytesPerSector],
										0, m_uiBytesPerSector);
					}
					else
					{
						goto Exit;
					}
				}
			}

			// Finally, copy the data from the source buffer into the
			// write buffer.

			f_memcpy( &pucWriteBuffer [uiWriteOffset & m_uiNotOnSectorBoundMask],
								pucSrcBuffer, uiBytesBeingOutput);
		}
		else
		{
			uiMaxBytesToWrite = RoundToNextSector( uiBytesToWrite);
			uiBytesBeingOutput = uiBytesToWrite;
			pucWriteBuffer = pucSrcBuffer;
			if( bZeroFill && uiMaxBytesToWrite > uiBytesToWrite)
			{
				f_memset( &pucWriteBuffer [uiBytesToWrite], 0,
							uiMaxBytesToWrite - uiBytesToWrite);
			}
		}

		// Position the file to the nearest sector below the write offset.

		uiLastWriteOffset = TruncateToPrevSector( uiWriteOffset);
		if (!m_bCanDoAsync)
		{
			lDummy = 0;
			if (SetFilePointer( m_FileHandle, (LONG)uiLastWriteOffset,
						&lDummy, FILE_BEGIN) == 0xFFFFFFFF)
			{
				rc = MapWinErrorToFlaim( GetLastError(),
							FERR_POSITIONING_IN_FILE);
				goto Exit;
			}
			pOverlapped = NULL;
		}
		else
		{
			if (!pBufferObj)
			{
				pOverlapped = &m_Overlapped;
			}
			else
			{
				pOverlapped = pBufferObj->getOverlapped();
				pBufferObj->setFileHandle( m_FileHandle);
			}

			if (!pOverlapped->hEvent)
			{
				if ((pOverlapped->hEvent = CreateEvent( NULL, TRUE,
														FALSE, NULL)) == NULL)
				{
					rc = MapWinErrorToFlaim( GetLastError(),
								FERR_SETTING_UP_FOR_WRITE);
					goto Exit;
				}
			}
			pOverlapped->Offset = (DWORD)uiLastWriteOffset;
			pOverlapped->OffsetHigh = 0;
			if (!ResetEvent( pOverlapped->hEvent))
			{
				rc = MapWinErrorToFlaim( GetLastError(),
								FERR_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
		}

		// Do the write

		uiLastWriteSize = uiMaxBytesToWrite;
		if (!WriteFile( m_FileHandle, (LPVOID)pucWriteBuffer,
							(DWORD)uiMaxBytesToWrite, &udBytesWritten,
						  pOverlapped))
		{
			udErr = GetLastError();
			if (udErr == ERROR_IO_PENDING && m_bCanDoAsync)
			{

				// If an async structure was passed in, we better have
				// been able to finish in a single write operation.
				// Otherwise, we are in deep trouble because we are not
				// set up to do multiple async write operations within
				// a single call.

				if( bDoAsync)
				{
					pBufferObj->makePending();
					bDidAsync = TRUE;
					break;
				}

				if (!GetOverlappedResult( m_FileHandle, pOverlapped,
							&udBytesWritten, TRUE))
				{
					rc = MapWinErrorToFlaim( GetLastError(),
								FERR_WRITING_FILE);
					goto Exit;
				}
			}
			else
			{
				rc = MapWinErrorToFlaim( udErr, FERR_WRITING_FILE);
				goto Exit;
			}
		}

		if (udBytesWritten < (DWORD)uiMaxBytesToWrite)
		{
			rc = RC_SET( FERR_IO_DISK_FULL);
			goto Exit;
		}

		uiBytesToWrite -= uiBytesBeingOutput;
		if( puiBytesWrittenRV)
		{
			(*puiBytesWrittenRV) += uiBytesBeingOutput;
		}
		m_uiCurrentPos = uiWriteOffset + uiBytesBeingOutput;
		if (!uiBytesToWrite)
		{
			break;
		}

		pucSrcBuffer += uiBytesBeingOutput;
		uiWriteOffset += uiBytesBeingOutput;
	}

	// See if we extended the file.  If so, we may want to extend it some
	// more and then also do a flush.

	if (bExtendFile)
	{

		// NOTE: uiLastWriteOffset + uiLastWrite is guaranteed to be
		// on a sector boundary.

		if (RC_BAD( rc = extendFile(
								uiLastWriteOffset + uiLastWriteSize,
								m_uiExtendSize, TRUE)))
		{
			goto Exit;
		}
	}

Exit:

	if( !bDidAsync && pBufferObj)
	{
		pBufferObj->notifyComplete( rc);
	}

	return( rc );
}

/****************************************************************************
Desc:		Write to a file
****************************************************************************/
RCODE F_FileHdlImp::Write(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWrittenRV)
{
	RCODE				rc = FERR_OK;
	DWORD				udBytesWritten;
	OVERLAPPED *	pOverlapped;
	LONG				lDummy;
	DWORD				udErr;

	if (m_bDoDirectIO)
	{
		rc = DirectWrite( uiWriteOffset, uiBytesToWrite, pvBuffer,
								NULL, FALSE, FALSE, puiBytesWrittenRV);
		goto Exit;
	}

	// If not doing direct IO, a single write call will do.

	flmAssert( m_bFileOpened);

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if (uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}

	// Position the file.

	if (!m_bCanDoAsync)
	{
		lDummy = 0;
		if (SetFilePointer( m_FileHandle, (LONG)uiWriteOffset,
					&lDummy, FILE_BEGIN) == 0xFFFFFFFF)
		{
			rc = MapWinErrorToFlaim( GetLastError(),
						FERR_POSITIONING_IN_FILE);
			goto Exit;
		}
		pOverlapped = NULL;
	}
	else
	{
		if (!m_Overlapped.hEvent)
		{
			if ((m_Overlapped.hEvent = CreateEvent( NULL, TRUE,
													FALSE, NULL)) == NULL)
			{
				rc = MapWinErrorToFlaim( GetLastError(),
							FERR_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
		}
		pOverlapped = &m_Overlapped;
		pOverlapped->Offset = (DWORD)uiWriteOffset;
		pOverlapped->OffsetHigh = 0;
		if (!ResetEvent( pOverlapped->hEvent))
		{
			rc = MapWinErrorToFlaim( GetLastError(),
						FERR_SETTING_UP_FOR_WRITE);
			goto Exit;
		}
	}

	if (!WriteFile( m_FileHandle, (LPVOID)pvBuffer,
						(DWORD)uiBytesToWrite, &udBytesWritten,
					   pOverlapped))
	{
		udErr = GetLastError();
		if (udErr == ERROR_IO_PENDING && m_bCanDoAsync)
		{
			if (!GetOverlappedResult( m_FileHandle, pOverlapped,
						&udBytesWritten, TRUE))
			{
				rc = MapWinErrorToFlaim( GetLastError(),
							FERR_WRITING_FILE);
				goto Exit;
			}
		}
		else
		{
			rc = MapWinErrorToFlaim( udErr, FERR_WRITING_FILE);
			goto Exit;
		}
	}

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = (FLMUINT)udBytesWritten;
	}

	m_uiCurrentPos = uiWriteOffset + (FLMUINT)udBytesWritten;
	if (udBytesWritten < (DWORD)uiBytesToWrite)
	{
		rc = RC_SET( FERR_IO_DISK_FULL);
		goto Exit;
	}

Exit:
	return( rc );
}

/****************************************************************************
Desc:	Deletes a file.
****************************************************************************/
FSTATIC RCODE _DeleteFile(
	const char *	path)
{
	RCODE			rc = FERR_OK;

	/* Delete the file */
   if( DeleteFile( (LPTSTR)path) == FALSE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_DELETING_FILE);
	}

	return( rc);
} 

#else
	#if defined( FLM_NLM) && !defined( __MWERKS__)
		int gv_iXxxxxDummy( void)
		{
			return( 0);
		}
	#endif
#endif // #if defined( FLM_WIN)

