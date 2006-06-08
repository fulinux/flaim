//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileHdl class on Windows platforms.
//
// Tabs:	3
//
//		Copyright (c) 1999-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fwin.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

#if defined( FLM_WIN)

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdl::F_FileHdl()
{
	m_bFileOpened = FALSE;
	m_bDeleteOnRelease = FALSE;
	m_bOpenedReadOnly = FALSE;
	m_pszFileName = NULL;
	m_FileHandle = INVALID_HANDLE_VALUE;
	m_uiBytesPerSector = 0;
	m_ui64NotOnSectorBoundMask = 0;
	m_ui64GetSectorBoundMask = 0;
	m_bDoDirectIO = FALSE;
	m_uiExtendSize = 0;
	m_uiMaxAutoExtendSize = FLM_MAXIMUM_FILE_SIZE;
	m_pucAlignedBuff = NULL;
	m_uiAlignedBuffSize = 0;
	m_ui64CurrentPos = 0;
	m_bOpenedInAsyncMode = FALSE;
	m_Overlapped.hEvent = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdl::~F_FileHdl()
{
	if( m_bFileOpened)
	{
		(void)close();
	}
	
	if (m_pucAlignedBuff)
	{
		(void)VirtualFree( m_pucAlignedBuff, 0, MEM_RELEASE);
		m_pucAlignedBuff = NULL;
		m_uiAlignedBuffSize = 0;
	}
	
	if (m_Overlapped.hEvent)
	{
		CloseHandle( m_Overlapped.hEvent);
	}
	
	if (m_pszFileName)
	{
		f_free( &m_pszFileName);
	}
}

/***************************************************************************
Desc:	Open or create a file.
***************************************************************************/
RCODE F_FileHdl::openOrCreate(
	const char *	pszFileName,
   FLMUINT			uiAccess,
	FLMBOOL			bCreateFlag)
{
	RCODE					rc = NE_FLM_OK;
	char					szSaveFileName[ F_PATH_MAX_SIZE];
	DWORD					udAccessMode = 0;
	DWORD					udShareMode = 0;
	DWORD					udCreateMode = 0;
	DWORD					udAttrFlags = 0;
	DWORD					udErrCode;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	m_bDoDirectIO = (uiAccess & FLM_IO_DIRECT) ? TRUE : FALSE;

	// Save the file name in case we have to create the directory

	if ((bCreateFlag) && (uiAccess & FLM_IO_CREATE_DIR))
	{
		f_strcpy( szSaveFileName, pszFileName);
	}

	// If doing direct IO, need to get the sector size.

	if (m_bDoDirectIO)
	{
		if (RC_BAD( rc = pFileSystem->getSectorSize(
			pszFileName, &m_uiBytesPerSector)))
		{
			goto Exit;
		}
		
		m_ui64NotOnSectorBoundMask = m_uiBytesPerSector - 1;
		m_ui64GetSectorBoundMask = ~m_ui64NotOnSectorBoundMask;
	}

	// Only enable asynchronous writes if direct I/O is enabled.

	if (m_bDoDirectIO && pFileSystem->canDoAsync())
	{
		m_bOpenedInAsyncMode = TRUE;
	}

	// Set up the file characteristics requested by caller.

   if (uiAccess & FLM_IO_SH_DENYRW)
   {
      udShareMode = 0;
      uiAccess &= ~FLM_IO_SH_DENYRW;
   }
   else if (uiAccess & FLM_IO_SH_DENYWR)
   {
      udShareMode = FILE_SHARE_READ;
      uiAccess &= ~FLM_IO_SH_DENYWR;
   }
	else if (uiAccess & FLM_IO_SH_DENYNONE)
   {
      udShareMode = (FILE_SHARE_READ | FILE_SHARE_WRITE);
      uiAccess &= ~FLM_IO_SH_DENYNONE;
   }
	else
	{
      udShareMode = (FILE_SHARE_READ | FILE_SHARE_WRITE);
	}

	// Begin setting the CreateFile flags and fields

   udAttrFlags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS;
	if (m_bDoDirectIO)
	{
		udAttrFlags |= FILE_FLAG_NO_BUFFERING;
	}
	
	if (m_bOpenedInAsyncMode)
	{
		udAttrFlags |= FILE_FLAG_OVERLAPPED;
	}

   if (bCreateFlag)
   {
   	if (uiAccess & FLM_IO_EXCL)
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

   if( (!bCreateFlag) && (uiAccess & FLM_IO_RDONLY))
	{
      udAccessMode = GENERIC_READ;
	}

Retry_Create:

	// Try to create or open the file

	if( (m_FileHandle = CreateFile( (LPCTSTR)pszFileName, udAccessMode,
					udShareMode, NULL, udCreateMode,
					udAttrFlags, NULL)) == INVALID_HANDLE_VALUE)
	{
		udErrCode = GetLastError();
		if ((udErrCode == ERROR_PATH_NOT_FOUND) && (uiAccess & FLM_IO_CREATE_DIR))
		{
			char		szTemp[ F_PATH_MAX_SIZE];
			char		szDirPath[ F_PATH_MAX_SIZE];

			uiAccess &= ~FLM_IO_CREATE_DIR;

			// Remove the file name for which we are creating the directory.

			if( RC_OK( pFileSystem->pathReduce( szSaveFileName, 
				szDirPath, szTemp)))
			{
				if( RC_OK( rc = pFileSystem->createDir( szDirPath)))
				{
					goto Retry_Create;
				}
				else
				{
					goto Exit;
				}
			}
		}
		
		rc = f_mapPlatformError( udErrCode,
						(RCODE)(bCreateFlag
								  ? (RCODE)(m_bDoDirectIO
												? (RCODE)NE_FLM_DIRECT_CREATING_FILE
												: (RCODE)NE_FLM_CREATING_FILE)
								  : (RCODE)(m_bDoDirectIO
												? (RCODE)NE_FLM_DIRECT_OPENING_FILE
												: (RCODE)NE_FLM_OPENING_FILE)));
		goto Exit;
	}
	
Exit:

	if( RC_BAD( rc))
	{
		m_FileHandle = INVALID_HANDLE_VALUE;
	}
	
   return( rc);
}

/****************************************************************************
Desc:	Create a file
****************************************************************************/
RCODE F_FileHdl::create(
	const char *	pszFileName,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = NE_FLM_OK;

	f_assert( m_bFileOpened == FALSE);

	if( uiIoFlags & FLM_IO_DELETE_ON_RELEASE)
	{
		f_assert( m_pszFileName == NULL);

		if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
		{
			goto Exit;
		}

		f_strcpy( m_pszFileName, pszFileName);
		m_bDeleteOnRelease = TRUE;
	}
	else
	{
		m_bDeleteOnRelease = FALSE;
	}

	if( RC_BAD( rc = openOrCreate( pszFileName, uiIoFlags, TRUE)))
	{
		goto Exit;
	}
	
	m_bFileOpened = TRUE;
	m_ui64CurrentPos = 0;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	if( RC_BAD( rc) && m_bDeleteOnRelease && m_pszFileName)
	{
		f_free( &m_pszFileName);
	}

	return rc;
}

/****************************************************************************
Desc:	Create a unique file name in the specified directory
****************************************************************************/
RCODE F_FileHdl::createUnique(
	char *				pszDirName,
	const char *		pszFileExtension,
	FLMUINT				uiIoFlags)
{
	RCODE					rc = NE_FLM_OK;
	char *				pszTmp;
	FLMBOOL				bModext = TRUE;
	FLMUINT				uiBaseTime = 0;
	FLMBYTE				ucHighByte = 0;
	char					szFileName[ F_FILENAME_SIZE];
	char					szDirPath[ F_PATH_MAX_SIZE];
	char					szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT				uiCount;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	szFileName[0] = '\0';
	szTmpPath[0] = '\0';
	f_assert( m_bFileOpened == FALSE);

	if( uiIoFlags & FLM_IO_DELETE_ON_RELEASE)
	{
		f_assert( m_pszFileName == NULL);
		m_bDeleteOnRelease = TRUE;
	}
	else
	{
		m_bDeleteOnRelease = FALSE;
	}
	
	f_strcpy( szDirPath, pszDirName);

   // Search backwards replacing trailing spaces with NULLs.

	pszTmp = (char *) szDirPath;
	pszTmp += (f_strlen( pszTmp) - 1);
	while( pszTmp >= (char *) szDirPath && (*pszTmp == 0x20))
	{
		*pszTmp = 0;
		pszTmp--;
	}

	// Append a backslash if one isn't already there

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
		pFileSystem->pathCreateUniqueName( &uiBaseTime, szFileName, 
			pszFileExtension, &ucHighByte, bModext);

		f_strcpy( szTmpPath, szDirPath);
		pFileSystem->pathAppend( szTmpPath, szFileName);
		if( m_pszFileName)
		{
			f_free( &m_pszFileName);
		}

		rc = create( szTmpPath, uiIoFlags | FLM_IO_EXCL);
		if (rc == NE_FLM_IO_DISK_FULL)
		{
			DeleteFile( (LPTSTR)szTmpPath);
			goto Exit;
		}
		if ((rc == NE_FLM_IO_PATH_NOT_FOUND) || (rc == NE_FLM_IO_INVALID_PASSWORD))
		{
			goto Exit;
		}
	} while ((rc != NE_FLM_OK) && (uiCount++ < 10));

   // Check if the path was created

   if ((uiCount >= 10) && (rc != NE_FLM_OK))
   {
		rc = RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE);
		goto Exit;
   }
	m_bFileOpened = TRUE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;

	// Created file name needs to be returned
	
	f_strcpy( pszDirName, szTmpPath);

Exit:

	if( RC_BAD( rc) && m_pszFileName)
	{
		f_free( &m_pszFileName);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Open a file
****************************************************************************/
RCODE F_FileHdl::open(
	const char *	pszFileName,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = NE_FLM_OK;

	f_assert( m_bFileOpened == FALSE);

	if( uiIoFlags & FLM_IO_DELETE_ON_RELEASE)
	{
		f_assert( m_pszFileName == NULL);

		if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
		{
			goto Exit;
		}

		f_strcpy( m_pszFileName, pszFileName);
		m_bDeleteOnRelease = TRUE;
	}
	else
	{
		m_bDeleteOnRelease = FALSE;
	}

	if( RC_BAD( rc = openOrCreate( pszFileName, uiIoFlags, FALSE)))
	{
		goto Exit;
	}

	m_bFileOpened = TRUE;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & FLM_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	if( RC_BAD( rc) && m_bDeleteOnRelease && m_pszFileName)
	{
		f_free( &m_pszFileName);
	}

	return rc;
}

/****************************************************************************
Desc:	Close a file
****************************************************************************/
RCODE FLMAPI F_FileHdl::close( void)
{
	FLMBOOL	bDeleteAllowed = TRUE;
	RCODE		rc = NE_FLM_OK;

	if( !m_bFileOpened)
	{
		goto Exit;
	}

	if( !CloseHandle( m_FileHandle))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_CLOSING_FILE);
		goto Exit;
	}

	m_FileHandle = INVALID_HANDLE_VALUE;
	m_bFileOpened = m_bOpenedReadOnly = m_bOpenedExclusive = FALSE;

	if( m_bDeleteOnRelease)
	{
		f_assert( NULL != m_pszFileName );

		if( bDeleteAllowed)
		{
			DeleteFile( (LPTSTR)m_pszFileName);
		}
		
		m_bDeleteOnRelease = FALSE;
		f_free( &m_pszFileName);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Flush IO to disk
****************************************************************************/
RCODE FLMAPI F_FileHdl::flush( void)
{
	RCODE		rc = NE_FLM_OK;

	if( !m_bDoDirectIO)
	{
		if( !FlushFileBuffers( m_FileHandle))
  		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_FLUSHING_FILE);
		}
	}
	return( rc);
}

/****************************************************************************
Desc:	Allocate an aligned buffer.
****************************************************************************/
RCODE F_FileHdl::allocAlignedBuffer( void)
{
	RCODE	rc = NE_FLM_OK;

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = roundToNextSector( 64 * 1024);
	if ((m_pucAlignedBuff = (FLMBYTE *)VirtualAlloc( NULL,
								(DWORD)m_uiAlignedBuffSize,
								MEM_COMMIT, PAGE_READWRITE)) == NULL)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_MEM);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Position and do a single read operation.
****************************************************************************/
RCODE F_FileHdl::doOneRead(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,
	void *			pvReadBuffer,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	OVERLAPPED *	pOverlapped;
	LARGE_INTEGER	liTmp;

	// Position the file to the specified offset.

	if (!m_bOpenedInAsyncMode)
	{
		liTmp.QuadPart = ui64ReadOffset;
		if( !SetFilePointerEx( m_FileHandle, liTmp, NULL, FILE_BEGIN))
		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
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
				rc = f_mapPlatformError( GetLastError(),
								NE_FLM_SETTING_UP_FOR_READ);
				goto Exit;
			}
		}
		
		pOverlapped = &m_Overlapped;
		pOverlapped->Offset = (DWORD)(ui64ReadOffset & 0xFFFFFFFF);
		pOverlapped->OffsetHigh = (DWORD)(ui64ReadOffset >> 32);
		
		if( !ResetEvent( pOverlapped->hEvent))
		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_SETTING_UP_FOR_READ);
			goto Exit;
		}
	}
	
	// Do the read

	if( !ReadFile( m_FileHandle, pvReadBuffer, uiBytesToRead, 
		puiBytesRead, pOverlapped))
	{
		DWORD		udErr = GetLastError();
		
		if( udErr == ERROR_IO_PENDING && m_bOpenedInAsyncMode)
		{
			if( !GetOverlappedResult( m_FileHandle, 
				pOverlapped, puiBytesRead, TRUE))
			{
				rc = f_mapPlatformError( GetLastError(), NE_FLM_READING_FILE);
				goto Exit;
			}
		}
		else
		{
			rc = f_mapPlatformError( udErr, NE_FLM_READING_FILE);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Read from a file - reads using aligned buffers and offsets - only
		sector boundaries
****************************************************************************/
RCODE F_FileHdl::directRead(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,
   void *			pvBuffer,
	FLMBOOL			bBuffHasFullSectors,
   FLMUINT *		puiBytesReadRV)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesRead;
	FLMBYTE *		pucReadBuffer;
	FLMBYTE *		pucDestBuffer;
	FLMUINT			uiMaxBytesToRead;
	FLMBOOL			bHitEOF;

	f_assert( m_bFileOpened);

	if( puiBytesReadRV)
	{
		*puiBytesReadRV = 0;
	}

	if( ui64ReadOffset == FLM_IO_CURRENT_POS)
	{
		ui64ReadOffset = m_ui64CurrentPos;
	}

	// This loop does multiple reads (if necessary) to get all of the
	// data.  It uses aligned buffers and reads at sector offsets.

	pucDestBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{

		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if ((ui64ReadOffset & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT64)pucDestBuffer) & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT64)uiBytesToRead & m_ui64NotOnSectorBoundMask) &&
			  (!bBuffHasFullSectors)))
		{
			if (!m_pucAlignedBuff)
			{
				if (RC_BAD( rc = allocAlignedBuffer()))
				{
					goto Exit;
				}
			}
			pucReadBuffer = m_pucAlignedBuff;

			// Must read enough bytes to cover all of the sectors that
			// contain the data we are trying to read.  The value of
			// (ui64ReadOffset & m_ui64NotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round that up to the next sector
			// to get the total number of bytes we are going to read.

			uiMaxBytesToRead = roundToNextSector( uiBytesToRead +
									  (ui64ReadOffset & m_ui64NotOnSectorBoundMask));

			// Can't read more than the aligned buffer will hold.

			if (uiMaxBytesToRead > m_uiAlignedBuffSize)
			{
				uiMaxBytesToRead = m_uiAlignedBuffSize;
			}
		}
		else
		{
			uiMaxBytesToRead = roundToNextSector( uiBytesToRead);
			f_assert( uiMaxBytesToRead >= uiBytesToRead);
			pucReadBuffer = pucDestBuffer;
		}

		bHitEOF = FALSE;
		if (RC_BAD( rc = doOneRead( truncateToPrevSector( ui64ReadOffset),
			uiMaxBytesToRead, pucReadBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
		
		if( uiBytesRead < uiMaxBytesToRead)
		{
			bHitEOF = TRUE;
		}

		// If the offset we want to read from is not on a sector
		// boundary, increment the read buffer pointer to the
		// offset where the data we need starts and decrement the
		// bytes read by the difference between the start of the
		// sector and the actual read offset.

		if (ui64ReadOffset & m_ui64NotOnSectorBoundMask)
		{
			pucReadBuffer += (ui64ReadOffset & m_ui64NotOnSectorBoundMask);
			f_assert( uiBytesRead >= m_uiBytesPerSector);
			uiBytesRead -= (ui64ReadOffset & m_ui64NotOnSectorBoundMask);
		}

		// If bytes read is more than we actually need, truncate it back
		// so that we only copy what we actually need.

		if( uiBytesRead > uiBytesToRead)
		{
			uiBytesRead = uiBytesToRead;
		}
		
		uiBytesToRead -= uiBytesRead;
		
		if( puiBytesReadRV)
		{
			(*puiBytesReadRV) += uiBytesRead;
		}
		
		m_ui64CurrentPos = ui64ReadOffset + uiBytesRead;

		// If using a different buffer for reading, copy the
		// data read into the destination buffer.

		if (pucDestBuffer != pucReadBuffer)
		{
			f_memcpy( pucDestBuffer, pucReadBuffer, uiBytesRead);
		}
		
		if (!uiBytesToRead)
		{
			break;
		}

		// Still more to read - did we hit EOF above?

		if (bHitEOF)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
			break;
		}
		
		pucDestBuffer += uiBytesRead;
		ui64ReadOffset += uiBytesRead;
	}

Exit:
	return( rc );
}

/****************************************************************************
Desc:	Read from a file
****************************************************************************/
RCODE FLMAPI F_FileHdl::read(
	FLMUINT64	ui64ReadOffset,
	FLMUINT		uiBytesToRead,
   void *		pvBuffer,
   FLMUINT *	puiBytesReadRV)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiBytesRead;

	// Do the direct IO call if enabled.

	if (m_bDoDirectIO)
	{
		rc = directRead( ui64ReadOffset, uiBytesToRead,
									pvBuffer, FALSE, puiBytesReadRV);
		goto Exit;
	}

	// If not doing direct IO, a single read call will do.

	f_assert( m_bFileOpened);
	if( puiBytesReadRV)
	{
		*puiBytesReadRV = 0;
	}

	if( ui64ReadOffset == FLM_IO_CURRENT_POS)
	{
		ui64ReadOffset = m_ui64CurrentPos;
	}

	if( RC_BAD( rc = doOneRead( ui64ReadOffset, uiBytesToRead,
		pvBuffer, &uiBytesRead)))
	{
		goto Exit;
	}

	if( puiBytesReadRV)
	{
		*puiBytesReadRV = uiBytesRead;
	}

	m_ui64CurrentPos = ui64ReadOffset + uiBytesRead;

	if (uiBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_FLM_IO_END_OF_FILE);
		goto Exit;
	}

Exit:
	return( rc );
}

/****************************************************************************
Desc:	Sets current position of file.
****************************************************************************/
RCODE FLMAPI F_FileHdl::seek(
	FLMUINT64			ui64Offset,
	FLMINT				iWhence,
	FLMUINT64 *			pui64NewOffset)
{
	RCODE	rc = NE_FLM_OK;

	switch (iWhence)
	{
		case FLM_IO_SEEK_CUR:
		{
			m_ui64CurrentPos += ui64Offset;
			break;
		}
		
		case FLM_IO_SEEK_SET:
		{
			m_ui64CurrentPos = ui64Offset;
			break;
		}
		
		case FLM_IO_SEEK_END:
		{
			if( RC_BAD( rc = size( &m_ui64CurrentPos )))
			{
				goto Exit;
			}
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
	if( pui64NewOffset)
	{
		*pui64NewOffset = m_ui64CurrentPos;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Return the size of the file
****************************************************************************/
RCODE FLMAPI F_FileHdl::size(
	FLMUINT64 *		pui64Size)
{
	RCODE					rc = NE_FLM_OK;
	LARGE_INTEGER		liTmp;
	
	if( !GetFileSizeEx( m_FileHandle, &liTmp))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_GETTING_FILE_SIZE);
		goto Exit;
	}
	
	*pui64Size = liTmp.QuadPart;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_FileHdl::tell(
	FLMUINT64 *		pui64Offset)
{
	*pui64Offset = m_ui64CurrentPos;
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:		Truncate the file to the indicated size
WARNING: Direct IO methods are calling this method.  Make sure that all changes
			to this method work in direct IO mode.
****************************************************************************/
RCODE FLMAPI F_FileHdl::truncate(
	FLMUINT64		ui64Size)
{
	RCODE					rc = NE_FLM_OK;
	LARGE_INTEGER		liTmp;

	f_assert( m_bFileOpened);

	// Position the file to the nearest sector below the read offset.
	
	liTmp.QuadPart = ui64Size;
	if( !SetFilePointerEx( m_FileHandle, liTmp, NULL, FILE_BEGIN))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_POSITIONING_IN_FILE);
		goto Exit;
	}
		
   // Set the new file size.

	if( !SetEndOfFile( m_FileHandle))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_TRUNCATING_FILE);
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
RCODE F_FileHdl::extendFile(
	FLMUINT64		ui64EndOfLastWrite,
	FLMUINT			uiMaxBytesToExtend,
	FLMBOOL			bFlush)
{
	RCODE				rc = NE_FLM_OK;
	OVERLAPPED *	pOverlapped;
	FLMUINT			uiTotalBytesToExtend;
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiBytesWritten;
	LARGE_INTEGER	liTmp;

	if ((uiTotalBytesToExtend = uiMaxBytesToExtend) != 0)
	{
		if (ui64EndOfLastWrite > m_uiMaxAutoExtendSize)
		{
			uiTotalBytesToExtend = 0;
		}
		else
		{
			// Don't extend beyond maximum file size.

			if (m_uiMaxAutoExtendSize - ui64EndOfLastWrite < uiTotalBytesToExtend)
			{
				uiTotalBytesToExtend = m_uiMaxAutoExtendSize - ui64EndOfLastWrite;
			}

			// If the extend size is not on a sector boundary, round it down.

			uiTotalBytesToExtend = truncateToPrevSector( uiTotalBytesToExtend);
		}
	}

	if (uiTotalBytesToExtend)
	{
		// Allocate an aligned buffer if we haven't already.

		if (!m_pucAlignedBuff)
		{
			if (RC_BAD( rc = allocAlignedBuffer()))
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
		
		if (!m_bOpenedInAsyncMode)
		{
			liTmp.QuadPart = ui64EndOfLastWrite;
			if( !SetFilePointerEx( m_FileHandle, liTmp, NULL, FILE_BEGIN))
			{
				rc = f_mapPlatformError( GetLastError(), 
							NE_FLM_POSITIONING_IN_FILE);
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
					rc = f_mapPlatformError( GetLastError(),
								NE_FLM_SETTING_UP_FOR_WRITE);
					goto Exit;
				}
			}
			
			pOverlapped->Offset = (DWORD)(ui64EndOfLastWrite & 0xFFFFFFFF);
			pOverlapped->OffsetHigh = (DWORD)(ui64EndOfLastWrite >> 32);
			
			if (!ResetEvent( pOverlapped->hEvent))
			{
				rc = f_mapPlatformError( GetLastError(),
							NE_FLM_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
		}
		
		// Do the write

		if( !WriteFile( m_FileHandle, m_pucAlignedBuff,
			uiBytesToWrite, &uiBytesWritten, pOverlapped))
		{
			flmAssert( 0);
			rc = f_mapPlatformError( GetLastError(), NE_FLM_WRITING_FILE);
			
			// Don't care if it is a disk full error, because
			// extending the file is optional work.

			if( rc == NE_FLM_IO_DISK_FULL)
			{
				rc = NE_FLM_OK;
				break;
			}
			
			goto Exit;
		}

		// NO more room on disk, but that's OK because we were only
		// extending the file beyond where it needed to be.  If that
		// fails, we will just flush what we have done so far.

		if( uiBytesWritten < uiBytesToWrite)
		{
			break;
		}
		
		uiTotalBytesToExtend -= uiBytesToWrite;
		ui64EndOfLastWrite += uiBytesToWrite;
	}

	// Flush the file buffers to ensure that the file size gets written
	// out.

	if( bFlush)
	{
		if( !FlushFileBuffers( m_FileHandle))
  		{
			rc = f_mapPlatformError( GetLastError(), NE_FLM_FLUSHING_FILE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Write to a file using direct IO
****************************************************************************/
RCODE F_FileHdl::directWrite(
	FLMUINT64			ui64WriteOffset,
	FLMUINT				uiBytesToWrite,
   const void *		pvBuffer,
	IF_IOBuffer *		pBufferObj,
	FLMBOOL				bBuffHasFullSectors,
	FLMBOOL				bZeroFill,
   FLMUINT *			puiBytesWrittenRV)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesRead;
	FLMUINT			uiBytesWritten;
	FLMBYTE *		pucWriteBuffer;
	FLMBYTE *		pucSrcBuffer;
	FLMUINT			uiMaxBytesToWrite;
	FLMUINT64		ui64CurrFileSize;
	FLMUINT			uiBytesBeingOutput;
	OVERLAPPED *	pOverlapped;
	DWORD				udErr;
	FLMBOOL			bExtendFile = FALSE;
	FLMBOOL			bWaitForWrite = (pBufferObj == NULL)
										? TRUE
										: FALSE;
	FLMUINT64		ui64LastWriteOffset;
	FLMUINT			uiLastWriteSize;
	LARGE_INTEGER	liTmp;

	f_assert( m_bFileOpened);

	if( !bWaitForWrite)
	{
		f_assert( m_bOpenedInAsyncMode);
	}
	
	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = 0;
	}

	if( ui64WriteOffset == FLM_IO_CURRENT_POS)
	{
		ui64WriteOffset = m_ui64CurrentPos;
	}

	// Determine if the write will extend the file beyond its
	// current size.  If so, we will need to call FlushFileBuffers

	if( !GetFileSizeEx( m_FileHandle, &liTmp))
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_GETTING_FILE_SIZE);
		goto Exit;
	}
	
	ui64CurrFileSize = liTmp.QuadPart;
	
	if( ui64WriteOffset + uiBytesToWrite > ui64CurrFileSize &&
		 m_uiExtendSize != (FLMUINT)(~0))
	{
		bExtendFile = TRUE;

		if( ui64WriteOffset > ui64CurrFileSize)
		{

			// Fill in the empty space.

			if (RC_BAD( rc = extendFile( ui64CurrFileSize,
				roundToNextSector( ui64WriteOffset - ui64CurrFileSize), FALSE)))
			{
				goto Exit;
			}
		}

		// Can't do asynchronous if we are going to extend the file.

		bWaitForWrite = TRUE;
	}

	// This loop is for direct IO - must make sure we use
	// aligned buffers.

	pucSrcBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{

		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if ((ui64WriteOffset & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT)pucSrcBuffer) & m_ui64NotOnSectorBoundMask) ||
			 ((uiBytesToWrite & m_ui64NotOnSectorBoundMask) &&
			  (!bBuffHasFullSectors)))
		{

			// Cannot do an async write if we have to use a temporary buffer
			
			bWaitForWrite = TRUE;

			if (!m_pucAlignedBuff)
			{
				if (RC_BAD( rc = allocAlignedBuffer()))
				{
					goto Exit;
				}
			}
			pucWriteBuffer = m_pucAlignedBuff;

			// Must write enough bytes to cover all of the sectors that
			// contain the data we are trying to write out.  The value of
			// (ui64WriteOffset & m_ui64NotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round to the next sector to get the
			// total number of bytes we are going to write.

			uiMaxBytesToWrite = roundToNextSector( uiBytesToWrite +
									  (ui64WriteOffset & m_ui64NotOnSectorBoundMask));

			// Can't write more than the aligned buffer will hold.

			if (uiMaxBytesToWrite > m_uiAlignedBuffSize)
			{
				uiMaxBytesToWrite = m_uiAlignedBuffSize;
				uiBytesBeingOutput = uiMaxBytesToWrite -
										(ui64WriteOffset & m_ui64NotOnSectorBoundMask);
			}
			else
			{
				uiBytesBeingOutput = uiBytesToWrite;
			}

			// If the write offset is not on a sector boundary, we must
			// read at least the first sector into the buffer.

			if (ui64WriteOffset & m_ui64NotOnSectorBoundMask)
			{

				// Read the first sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if (RC_BAD( rc = doOneRead( truncateToPrevSector( ui64WriteOffset),
					m_uiBytesPerSector, pucWriteBuffer, &uiBytesRead)))
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

				if (RC_BAD( rc = doOneRead(
					(truncateToPrevSector( ui64WriteOffset)) +
						(uiMaxBytesToWrite - m_uiBytesPerSector),
					m_uiBytesPerSector,
					(&pucWriteBuffer [uiMaxBytesToWrite - m_uiBytesPerSector]),
					&uiBytesRead)))
				{
					if (rc == NE_FLM_IO_END_OF_FILE)
					{
						rc = NE_FLM_OK;
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

			f_memcpy( &pucWriteBuffer[ ui64WriteOffset & m_ui64NotOnSectorBoundMask],
								pucSrcBuffer, uiBytesBeingOutput);
		}
		else
		{
			uiMaxBytesToWrite = roundToNextSector( uiBytesToWrite);
			uiBytesBeingOutput = uiBytesToWrite;
			pucWriteBuffer = pucSrcBuffer;
			if( bZeroFill && uiMaxBytesToWrite > uiBytesToWrite)
			{
				f_memset( &pucWriteBuffer [uiBytesToWrite], 0,
							uiMaxBytesToWrite - uiBytesToWrite);
			}
		}

		// Position the file to the nearest sector below the write offset.

		ui64LastWriteOffset = truncateToPrevSector( ui64WriteOffset);
		if( !m_bOpenedInAsyncMode)
		{
			liTmp.QuadPart = ui64LastWriteOffset;
			if( !SetFilePointerEx( m_FileHandle, liTmp, NULL, FILE_BEGIN))
			{
				rc = f_mapPlatformError( GetLastError(),
							NE_FLM_POSITIONING_IN_FILE);
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
				pOverlapped = ((F_IOBuffer *)pBufferObj)->getOverlapped();
				((F_IOBuffer *)pBufferObj)->setFileHandle( m_FileHandle);
			}

			if (!pOverlapped->hEvent)
			{
				if ((pOverlapped->hEvent = CreateEvent( NULL, TRUE,
														FALSE, NULL)) == NULL)
				{
					rc = f_mapPlatformError( GetLastError(),
								NE_FLM_SETTING_UP_FOR_WRITE);
					goto Exit;
				}
			}
			
			pOverlapped->Offset = (DWORD)(ui64LastWriteOffset & 0xFFFFFFFF);
			pOverlapped->OffsetHigh = (DWORD)(ui64LastWriteOffset >> 32);
			
			if (!ResetEvent( pOverlapped->hEvent))
			{
				rc = f_mapPlatformError( GetLastError(),
								NE_FLM_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
		}

		// Do the write

		uiLastWriteSize = uiMaxBytesToWrite;
		if( !WriteFile( m_FileHandle, (LPVOID)pucWriteBuffer,
							(DWORD)uiMaxBytesToWrite, &uiBytesWritten,
						  pOverlapped))
		{
			udErr = GetLastError();
			if( udErr == ERROR_IO_PENDING && m_bOpenedInAsyncMode)
			{
				// If an async structure was passed in, we better have
				// been able to finish in a single write operation.
				// Otherwise, we are in deep trouble because we are not
				// set up to do multiple async write operations within
				// a single call.

				if( !bWaitForWrite)
				{
					pBufferObj->makePending();
					break;
				}

				if (!GetOverlappedResult( m_FileHandle, pOverlapped,
							&uiBytesWritten, TRUE))
				{
					rc = f_mapPlatformError( GetLastError(),
								NE_FLM_WRITING_FILE);
					goto Exit;
				}
			}
			else
			{
				flmAssert( 0);
				rc = f_mapPlatformError( udErr, NE_FLM_WRITING_FILE);
				goto Exit;
			}
		}

		if (uiBytesWritten < uiMaxBytesToWrite)
		{
			rc = RC_SET( NE_FLM_IO_DISK_FULL);
			goto Exit;
		}

		uiBytesToWrite -= uiBytesBeingOutput;
		
		if( puiBytesWrittenRV)
		{
			(*puiBytesWrittenRV) += uiBytesBeingOutput;
		}
		
		m_ui64CurrentPos = ui64WriteOffset + uiBytesBeingOutput;
		
		if (!uiBytesToWrite)
		{
			break;
		}

		pucSrcBuffer += uiBytesBeingOutput;
		ui64WriteOffset += uiBytesBeingOutput;
	}

	// See if we extended the file.  If so, we may want to extend it some
	// more and then also do a flush.

	if (bExtendFile)
	{
		// NOTE: ui64LastWriteOffset + uiLastWrite is guaranteed to be
		// on a sector boundary.

		if (RC_BAD( rc = extendFile(
								ui64LastWriteOffset + uiLastWriteSize,
								m_uiExtendSize, TRUE)))
		{
			goto Exit;
		}
	}

Exit:

	if( pBufferObj && !pBufferObj->isPending())
	{
		pBufferObj->notifyComplete( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:	Write to a file
****************************************************************************/
RCODE FLMAPI F_FileHdl::write(
	FLMUINT64		ui64WriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWrittenRV)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesWritten;
	OVERLAPPED *	pOverlapped;
	DWORD				udErr;
	LARGE_INTEGER	liTmp;

	if (m_bDoDirectIO)
	{
		rc = directWrite( ui64WriteOffset, uiBytesToWrite, pvBuffer,
								NULL, FALSE, FALSE, puiBytesWrittenRV);
		goto Exit;
	}

	// If not doing direct IO, a single write call will do.

	f_assert( m_bFileOpened);

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = 0;
	}

	if( ui64WriteOffset == FLM_IO_CURRENT_POS)
	{
		ui64WriteOffset = m_ui64CurrentPos;
	}

	// Position the file.

	if (!m_bOpenedInAsyncMode)
	{
		liTmp.QuadPart = ui64WriteOffset;
		if( !SetFilePointerEx( m_FileHandle, liTmp, NULL, FILE_BEGIN))
		{
			rc = f_mapPlatformError( GetLastError(),
						NE_FLM_POSITIONING_IN_FILE);
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
				rc = f_mapPlatformError( GetLastError(),
							NE_FLM_SETTING_UP_FOR_WRITE);
				goto Exit;
			}
		}
		
		pOverlapped = &m_Overlapped;
		pOverlapped->Offset = (DWORD)(ui64WriteOffset & 0xFFFFFFFF);
		pOverlapped->OffsetHigh = (DWORD)(ui64WriteOffset >> 32);
		
		if( !ResetEvent( pOverlapped->hEvent))
		{
			rc = f_mapPlatformError( GetLastError(),
						NE_FLM_SETTING_UP_FOR_WRITE);
			goto Exit;
		}
	}

	if (!WriteFile( m_FileHandle, (LPVOID)pvBuffer,
						(DWORD)uiBytesToWrite, &uiBytesWritten,
					   pOverlapped))
	{
		udErr = GetLastError();
		if (udErr == ERROR_IO_PENDING && m_bOpenedInAsyncMode)
		{
			if (!GetOverlappedResult( m_FileHandle, pOverlapped,
						&uiBytesWritten, TRUE))
			{
				flmAssert( 0);
				rc = f_mapPlatformError( GetLastError(), NE_FLM_WRITING_FILE);
				goto Exit;
			}
		}
		else
		{
			flmAssert( 0);
			rc = f_mapPlatformError( udErr, NE_FLM_WRITING_FILE);
			goto Exit;
		}
	}

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = uiBytesWritten;
	}

	m_ui64CurrentPos = ui64WriteOffset + uiBytesWritten;
	
	if (uiBytesWritten < uiBytesToWrite)
	{
		rc = RC_SET( NE_FLM_IO_DISK_FULL);
		goto Exit;
	}

Exit:
	return( rc );
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_FileHdl::lock( void)
{
	RCODE				rc = NE_FLM_OK;

	// Lock the first byte in file

	if( !LockFile( m_FileHandle, 0, 0, 1, 1))
	{
		rc = RC_SET( NE_FLM_IO_FILE_LOCK_ERR);
		goto Exit;
	}

Exit:

	return( rc);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_FileHdl::unlock( void)
{
	RCODE				rc = NE_FLM_OK;

	// Unlock the first byte in file

	if( !UnlockFile( m_FileHandle, 0, 0, 1, 1))
	{
		rc = RC_SET( NE_FLM_IO_FILE_LOCK_ERR);
		goto Exit;
	}

Exit:

	return( rc);
}
		
/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI f_yieldCPU( void)
{
	Sleep( 0);
}

/**********************************************************************
Desc:
**********************************************************************/
RCODE FLMAPI f_chdir(
	const char *		pszDir)
{
	RCODE		rc = NE_FLM_OK;
	
	if( _chdir( pszDir) != 0)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/**********************************************************************
Desc:
**********************************************************************/
RCODE FLMAPI f_getcwd(
	char *			pszDir)
{
	RCODE		rc = NE_FLM_OK;
	
	if( _getcwd( pszDir, F_PATH_MAX_SIZE) == NULL)
	{
		*pszDir = 0;
		rc = f_mapPlatformError( GetLastError(), NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
Exit:

	return( rc);
}

#endif // FLM_WIN

/****************************************************************************
Desc:	Deletes a file
****************************************************************************/
#if defined( FLM_WATCOM_NLM) || defined( FLM_OSX)
int gv_ftkwinDummy(void)
{
	return( 0);
}
#endif
