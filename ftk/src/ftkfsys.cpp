//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileSystem class.
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
// $Id: ffilesys.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

/****************************************************************************
Desc:    Create a file, return a file handle to created file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->create( pszFileName, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (IF_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Create a block-oriented file, return a file handle to created file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createBlockFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	FLMUINT			uiBlockSize,
	IF_FileHdl **	ppFileHdl)
{
	RCODE			rc = NE_FLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	pFileHdl->setBlockSize( uiBlockSize);
	
	if (RC_BAD( rc = pFileHdl->create( pszFileName, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (IF_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:	Create a unique file, return a file handle to created file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createUniqueFile(
	const char *	pszDirName,
	const char *	pszFileExtension,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc;
	F_FileHdl *		pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = pFileHdl->createUnique( pszDirName, 
			pszFileExtension,	uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (IF_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc: Open a file, return a file handle to opened file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::openFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE			rc = NE_FLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = pFileHdl->open( pszFileName, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (IF_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:	Open a block-oriented file, return a file handle to opened file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::openBlockFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	FLMUINT			uiBlockSize,
	IF_FileHdl **	ppFileHdl)
{
	RCODE			rc = NE_FLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	pFileHdl->setBlockSize( uiBlockSize);
	
	if (RC_BAD( rc = pFileHdl->open( pszFileName, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (IF_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Open a directory, return a file handle to opened directory.
****************************************************************************/
RCODE FLMAPI F_FileSystem::openDir(
	const char *	pszDirName,
	const char *	pszPattern,
	IF_DirHdl **	ppDirHdl)
{
	RCODE			rc = NE_FLM_OK;
	F_DirHdl *	pDirHdl = NULL;

	if ((pDirHdl = f_new F_DirHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = pDirHdl->openDir( pszDirName, pszPattern)))
	{
		pDirHdl->Release();
		pDirHdl = NULL;
	}

Exit:

	*ppDirHdl = (IF_DirHdl *)pDirHdl;
	return( rc);
}

/****************************************************************************
Desc:    Create a directory.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createDir(
	const char *	pszDirName)
{
	RCODE			rc = NE_FLM_OK;
	F_DirHdl *	pDirHdl = NULL;

	if ((pDirHdl = f_new F_DirHdl) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	rc = pDirHdl->createDir( pszDirName);

Exit:

	if (pDirHdl)
	{
		pDirHdl->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Remove a directory
****************************************************************************/
RCODE FLMAPI F_FileSystem::removeDir(
	const char *	pszDirName,
	FLMBOOL			bClear)
{
	RCODE				rc = NE_FLM_OK;
	IF_DirHdl *		pDirHdl = NULL;
	char				szFilePath[ F_PATH_MAX_SIZE];

	if( bClear)
	{
		if( RC_BAD( rc = openDir( pszDirName, (char *)"*", &pDirHdl)))
		{
			goto Exit;
		}

		for ( rc = pDirHdl->next(); RC_OK( rc) ; rc = pDirHdl->next())
		{
			pDirHdl->currentItemPath( szFilePath);
			if( !pDirHdl->currentItemIsDir())
			{
				if( RC_BAD( rc = deleteFile( szFilePath)))
				{
					if( rc == NE_FLM_IO_PATH_NOT_FOUND || 
						 rc == NE_FLM_IO_INVALID_FILENAME)
					{
						rc = NE_FLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
			else
			{
				if( RC_BAD( rc = removeDir( szFilePath, bClear)))
				{
					if( rc == NE_FLM_IO_PATH_NOT_FOUND ||
						 rc == NE_FLM_IO_INVALID_FILENAME)
					{
						rc = NE_FLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
		}

		// Need to release the directory handle so the
		// directory will be closed when we try to delete it
		// below.

		pDirHdl->Release();
		pDirHdl = NULL;
	}

	if( RC_BAD( rc = removeDir( pszDirName)))
	{
		goto Exit;
	}

Exit:

	if (pDirHdl)
	{
		pDirHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:	Determine if a file or directory exists.
****************************************************************************/
RCODE FLMAPI F_FileSystem::doesFileExist(
	const char *	pszPath)
{
#if defined( FLM_NLM)

	return( flmNetWareTestIfFileExists( pszPath));

#elif defined( FLM_WIN)

	DWORD		dwFileAttr = GetFileAttributes( (LPTSTR)pszPath);

	if( dwFileAttr == (DWORD)-1)
		return RC_SET( NE_FLM_IO_PATH_NOT_FOUND);

	return NE_FLM_OK;

#else

   if( access( pszPath, F_OK) == -1)
	{
		return( MapErrnoToFlaimErr( errno, NE_FLM_CHECKING_FILE_EXISTENCE));
	}

	return( NE_FLM_OK);
	
#endif
}

/****************************************************************************
Desc:    Get the time stamp of the last modification to this file.
Notes:
	puiTimeStamp is assumed to point to a DWORD.

NLM Notes:
	We could call MapPathToDirectoryNumber and GetDirectoryEntry directly.
	This works, providing that the high byte of the directory entry (returned
	by MapPathToDirectoryNumber) is masked off.  Otherwise, GetDirectoryEntry
	will generate an abend.
	We have opted to call a higher level function, GetEntryFromPathStringBase,
	which calls the lower level functions for us.
****************************************************************************/
RCODE FLMAPI F_FileSystem::getFileTimeStamp(
	const char *	pszPath,
	FLMUINT *		puiTimeStamp)
{
#if defined( FLM_NLM)
	
	return( flmNetWareGetFileTimeStamp( pszPath, puiTimeStamp));
 
#elif defined( FLM_WIN)

	WIN32_FIND_DATA find_data;
	FILETIME			ftLocalFileTime;
	SYSTEMTIME		stLastFileWriteTime;
	HANDLE			hSearch = INVALID_HANDLE_VALUE;
	RCODE				rc = NE_FLM_OK;
	F_TMSTAMP		tmstamp;

	hSearch = FindFirstFile( (LPTSTR)pszPath, &find_data);
	if( hSearch == INVALID_HANDLE_VALUE)
	{
		rc = MapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		switch( rc)
		{
	   	case NE_FLM_IO_NO_MORE_FILES:
				rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
				goto Exit;
			default:
				goto Exit;
		}	/* End switch. */
	}

	// Convert it to a local time, so we can adjust based on our own
	// GroupWise time zone.
		
	if( FileTimeToLocalFileTime( &(find_data.ftLastWriteTime),
											&ftLocalFileTime) == FALSE)
	{
		rc = MapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

	// Convert the local time to a system time so we can map it into
	// a GroupWise Date\Time structure
		
	if( FileTimeToSystemTime( &ftLocalFileTime,
									   &stLastFileWriteTime) == FALSE)
	{
		rc = MapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

   /* Fill the Time Stamp structure */
   f_memset( &tmstamp, 0, sizeof( F_TMSTAMP));
   tmstamp.hour		= (FLMBYTE)stLastFileWriteTime.wHour;
   tmstamp.minute		= (FLMBYTE)stLastFileWriteTime.wMinute;
   tmstamp.second		= (FLMBYTE)stLastFileWriteTime.wSecond;
	tmstamp.hundredth = (FLMBYTE)stLastFileWriteTime.wMilliseconds;
   tmstamp.year		= (FLMUINT16)stLastFileWriteTime.wYear;
	tmstamp.month		= (FLMBYTE)(stLastFileWriteTime.wMonth - 1);
   tmstamp.day			= (FLMBYTE)stLastFileWriteTime.wDay;

   /* Convert and return the file time stamp as seconds since January 1, 1970 */
   f_timeDateToSeconds( &tmstamp, puiTimeStamp);

Exit:

	if( hSearch != INVALID_HANDLE_VALUE)
	{
	   FindClose( hSearch);
	}

	if( RC_OK(rc))
	{
		*puiTimeStamp = f_localTimeToUTC( *puiTimeStamp);
	}

   return( rc);

#else

	struct stat   	filestatus;

	if( stat( pszPath, &filestatus) == -1)
	{
       return( MapErrnoToFlaimErr( errno, NE_FLM_GETTING_FILE_INFO));
	}

	*puiTimeStamp = (FLMUINT)filestatus.st_mtime; // st_mtime is UTC
	return NE_FLM_OK;

#endif
}

/****************************************************************************
Desc: Determine if a path is a directory.
****************************************************************************/
FLMBOOL FLMAPI F_FileSystem::isDir(
	const char *		pszDirName)
{
#if defined( FLM_WIN)

	DWORD	FileAttr = GetFileAttributes( (LPTSTR)pszDirName);
	
	if( FileAttr == 0xFFFFFFFF)
	{
		return( FALSE);
	}

	return (FileAttr & FILE_ATTRIBUTE_DIRECTORY) ? TRUE : FALSE;

#else

	struct stat     filestatus;

	if( stat( (char *)pszDirName, &filestatus) == -1)
	{
		return FALSE;
	}

	return ( S_ISDIR( filestatus.st_mode)) ? TRUE : FALSE;
#endif
}

/****************************************************************************
Desc:    Delete a file or directory
****************************************************************************/
RCODE FLMAPI F_FileSystem::deleteFile(
	const char *		pszFileName)
{
#if defined( FLM_NLM)

	return( flmNetWareDeleteFile( pszFileName));

#elif defined( FLM_WIN)

	if( DeleteFile( (LPTSTR)pszFileName) == FALSE)
	{
		return( MapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}
	
	return( NE_FLM_OK);

#else

	struct stat FileStat;

	if( stat( (char *)pszFileName, &FileStat) == -1)
	{
		return( MapErrnoToFlaimErr( errno, NE_FLM_GETTING_FILE_INFO));
	}

	// Ensure that the path does NOT designate a directory for deletion
	
	if( S_ISDIR(FileStat.st_mode))
	{
		return( RC_SET( NE_FLM_IO_ACCESS_DENIED));
	}

	// Delete the file
	
	if( unlink( (char *)pszFileName) == -1)
	{
       return( MapErrnoToFlaimErr( errno, NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);
	 
#endif
}

/****************************************************************************
Desc:	Copy a file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::copyFile(
	const char *	pszSrcFileName,	// Name of source file to be copied.
	const char *	pszDestFileName,	// Name of destination file.
	FLMBOOL			bOverwrite,			// Overwrite destination file?
	FLMUINT64 *		pui64BytesCopied)	// Returns number of bytes copied.
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pSrcFileHdl = NULL;
	IF_FileHdl *	pDestFileHdl = NULL;
	FLMBOOL			bCreatedDest = FALSE;
	FLMUINT64		ui64SrcSize;

	// See if the destination file exists.  If it does, see if it is
	// OK to overwrite it.  If so, delete it.

	if (doesFileExist( pszDestFileName) == NE_FLM_OK)
	{
		if (!bOverwrite)
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		if (RC_BAD( rc = deleteFile( pszDestFileName)))
		{
			goto Exit;
		}
	}

	// Open the source file.

	if( RC_BAD( rc = openFile( pszSrcFileName, 
			FLM_IO_RDONLY | FLM_IO_SH_DENYNONE, &pSrcFileHdl)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pSrcFileHdl->size( &ui64SrcSize)))
	{
		goto Exit;
	}

	// Create the destination file.

	if( RC_BAD( rc = createFile( pszDestFileName, 
			FLM_IO_RDWR | FLM_IO_SH_DENYNONE, &pDestFileHdl)))
	{
		goto Exit;
	}
	bCreatedDest = TRUE;

	// Do the copy.

	if( RC_BAD( rc = f_copyPartial( pSrcFileHdl, 0, ui64SrcSize, 
				pDestFileHdl, 0, pui64BytesCopied)))
	{
		goto Exit;
	}
	
Exit:

	if( pSrcFileHdl)
	{
		pSrcFileHdl->close();
		pSrcFileHdl->Release();
	}
	
	if( pDestFileHdl)
	{
		pDestFileHdl->close();
		pDestFileHdl->Release();
	}
	
	if( RC_BAD( rc))
	{
		if( bCreatedDest)
		{
			(void)deleteFile( pszDestFileName);
		}
		
		*pui64BytesCopied = 0;
	}
	
	return( rc);
}

/****************************************************************************
Desc: Rename a file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::renameFile(
	const char *		pszFileName,
	const char *		pszNewFileName)
{
#if defined( FLM_NLM)

	return( flmNetWareRenameFile( pszFileName, pszNewFileName));

#elif defined( FLM_WIN)

	DWORD			error;
	RCODE			rc = NE_FLM_OK;
	FLMUINT64	ui64BytesCopied;

   // Try to move the file by doing a rename first, otherwise copy the file

	if( (MoveFile( (LPTSTR)pszFileName, (LPTSTR)pszNewFileName)) != TRUE)
	{
		error = GetLastError();
		switch( error)
		{
			case ERROR_NOT_SAME_DEVICE:
			case ERROR_NO_MORE_FILES:
			case NO_ERROR:
				if( copyFile( pszFileName, pszNewFileName, TRUE, &ui64BytesCopied))
				{
					rc = RC_SET( NE_FLM_IO_COPY_ERR);
				}
				else
				{
					rc = F_FileSystem::deleteFile( pszFileName);
				}
				break;
			default:
				rc = MapPlatformError( error, NE_FLM_RENAMING_FILE);
				break;
		}
	}

	return( rc);

#else

	RCODE			rc;
	FLMBOOL		bSrcIsDir;
	FLMUINT64	ui64BytesCopied;

	if( RC_BAD( rc = unix_TargetIsDir( (char*)pszFileName, &bSrcIsDir)))
	{
		return( rc);
	}

	errno = 0;

	if( RC_BAD( unix_RenameSafe( pszFileName, pszNewFileName)))
	{
		switch( errno)
		{
			case EXDEV:
			{
				if( bSrcIsDir)
				{
					return( RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE));
				}
				else
				{
					if( copyFile( pszFileName, pszNewFileName, TRUE, &ui64BytesCopied))
					{
						return( RC_SET( NE_FLM_IO_COPY_ERR));
					}
					
					F_FileSystem::deleteFile( pszFileName);
					return( NE_FLM_OK);
				}
			}

			default:
			{
				if( errno == ENOENT)
				{
					return( RC_SET( NE_FLM_IO_RENAME_FAILURE));
				}
				else
				{
					return( MapErrnoToFlaimErr( errno, NE_FLM_RENAMING_FILE));
				}
			}
		}
	}

	return( NE_FLM_OK);
#endif
}

/****************************************************************************
Desc: Get the sector size (not supported on all platforms).
****************************************************************************/
RCODE FLMAPI F_FileSystem::getSectorSize(
	const char *	pszFileName,
	FLMUINT *		puiSectorSize)
{
#ifdef FLM_NLM

	F_UNREFERENCED_PARM( pszFileName);
	*puiSectorSize = NETWARE_SECTOR_SIZE;
	return( NE_FLM_OK);
	
#elif defined( FLM_WIN)

	RCODE			rc = NE_FLM_OK;
	DWORD			udSectorsPerCluster;
	DWORD			udBytesPerSector;
	DWORD			udNumberOfFreeClusters;
	DWORD			udTotalNumberOfClusters;
	char			szVolume [256];
	char *		pszVolume;
	FLMUINT		uiLen;

	if (!pszFileName)
	{
		pszVolume = NULL;
	}
	else
	{
		pathParse( pszFileName, NULL, szVolume, NULL, NULL);
		if (!szVolume [0])
		{
			pszVolume = NULL;
		}
		else
		{
			uiLen = f_strlen( szVolume);
			if (szVolume [uiLen - 1] == ':')
			{
				szVolume [uiLen] = '\\';
				szVolume [uiLen + 1] = 0;
			}
			pszVolume = &szVolume [0];
		}
	}

	if (!GetDiskFreeSpace( (LPCTSTR)pszVolume, &udSectorsPerCluster,
			&udBytesPerSector, &udNumberOfFreeClusters,
			&udTotalNumberOfClusters))
	{
		rc = MapPlatformError( GetLastError(), NE_FLM_INITIALIZING_IO_SYSTEM);
		*puiSectorSize = 0;
		goto Exit;
	}
	*puiSectorSize = (FLMUINT)udBytesPerSector;
	
Exit:

	return( rc);

#else
	F_UNREFERENCED_PARM( pszFileName);
	*puiSectorSize = (FLMUINT)sysconf( _SC_PAGESIZE);
	return( NE_FLM_OK);
#endif
}

/****************************************************************************
Desc: Set the Read-Only Attribute (not supported on all platforms).
****************************************************************************/
RCODE F_FileSystem::setReadOnly(
	const char *	pszFileName,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = NE_FLM_OK;

#if defined( FLM_UNIX)
	struct stat		filestatus;

	if( stat( (char *)pszFileName, &filestatus))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
	if ( bReadOnly)
	{
		filestatus.st_mode &= ~S_IWUSR;
	}
	else
	{
		filestatus.st_mode |= S_IWUSR;
	}
	
	if ( chmod( (char *)pszFileName, filestatus.st_mode))
	{
		rc = RC_SET( NE_FLM_FAILURE);
		goto Exit;
	}
	
#elif defined( FLM_WIN)

	DWORD				dwAttr;

	dwAttr = GetFileAttributes( (LPTSTR)pszFileName);
	if( dwAttr == (DWORD)-1)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
	if ( bReadOnly)
	{
		dwAttr |= F_IO_FA_RDONLY;
	}
	else
	{
		dwAttr &= ~F_IO_FA_RDONLY;
	}
	
	if( !SetFileAttributes( (LPTSTR)pszFileName, dwAttr))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
#elif defined( FLM_NLM)

	if ( RC_BAD( rc = flmNetWareSetReadOnly( pszFileName, bReadOnly)))
	{
		goto Exit;
	}
#else
	rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc: stat tpath to see if it is a directory
****************************************************************************/
#if defined( FLM_UNIX)
RCODE F_FileSystem::unix_TargetIsDir(
	const char	*	tpath,
	FLMBOOL *		isdir)
{
	struct stat		sbuf;
	RCODE				rc = NE_FLM_OK;

	*isdir = 0;
	if( stat(tpath, &sbuf) < 0)
	{
		rc = MapErrnoToFlaimErr( errno, NE_FLM_IO_ACCESS_DENIED);
	}
	else if( (sbuf.st_mode & S_IFMT) == S_IFDIR)
	{
		*isdir = 1;
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:	Rename an existing file (typically an "X" locked file to an
		unlocked file) using a safe (non-race) method.  To ensure that
		an existing file is not being overwritten by a rename operation,
		we will first create a new file with the desired name (using the
		CREAT and EXCL options, (ensuring a unique file name)).  Then,
		the source file will be renamed to new name.
****************************************************************************/
#if defined( FLM_UNIX)
RCODE F_FileSystem::unix_RenameSafe(
	const char *	pszSrcFile,
	const char *	pszDestFile)
{
	RCODE			rc = NE_FLM_OK;
	struct stat	temp_stat_buf;
	
	errno = 0;
	if( stat( pszDestFile, &temp_stat_buf) != -1)
	{
		// If we were able to stat it, then the file obviously exists...
		
		rc = RC_SET( NE_FLM_IO_RENAME_FAILURE);
		goto Exit;
	}
	else
	{
		if (errno != ENOENT)
		{
			// ENOENT means the file didn't exist, which is what we were
			// hoping for.
			
			rc = MapErrnoToFlaimErr( errno, NE_FLM_IO_RENAME_FAILURE);
			goto Exit;
		}
	}

	errno = 0;
	if( rename( pszSrcFile, pszDestFile) != 0)
	{
		rc = MapErrnoToFlaimErr( errno, NE_FLM_IO_RENAME_FAILURE);
	}

Exit:

	return( rc);
}
#endif

/****************************************************************************
Desc:		Initializes variables
****************************************************************************/
F_FileHdlMgr::F_FileHdlMgr()
{
	m_hMutex = F_MUTEX_NULL;

	m_uiOpenThreshold = 0xFFFF;
	FLM_SECS_TO_TIMER_UNITS( 30 * 60, m_uiMaxAvailTime);
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
	RCODE			rc = NE_FLM_OK;

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
	FLMUINT			uiFileId,
	FLMBOOL			bReadOnlyFlag,
	F_FileHdl **	ppFileHdl)
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

			pFileHdl->AddRef();

			removeFromList( TRUE,
				pFileHdl, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
				
			insertInList( TRUE, pFileHdl, FALSE,
				&m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);

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
	F_FileHdl *		pFileHdl)
{
	pFileHdl->m_uiAvailTime = (FLMUINT)FLM_GET_TIMER();

	lockMutex( bMutexAlreadyLocked);

	pFileHdl->AddRef();

	removeFromList( TRUE,
			pFileHdl, &m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);
			
	insertInList( TRUE, pFileHdl, TRUE,
			&m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);

	pFileHdl->Release();
	pFileHdl->Release();

	unlockMutex( bMutexAlreadyLocked);
}

/****************************************************************************
Desc:	Remove (close&free) all FileHdl's that have the specified FileId.
		Remove from the avail and used lists.
****************************************************************************/
void F_FileHdlMgr::removeFileHdls(
 	FLMUINT			uiFileId)
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
	FLMBOOL			bMutexAlreadyLocked)
{
	F_FileHdl *		pFileHdl;
	F_FileHdl *		pNextFileHdl;

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
	FLMBOOL			bMutexAlreadyLocked)
{
	F_FileHdl *		pFileHdl;
	F_FileHdl *		pNextFileHdl;

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
	FLMUINT *		puiCount)
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
	FLMUINT *		puiCount)
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
	FLMUINT	uiMinTimeOpened)
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

		if( FLM_ELAPSED_TIME( uiTime, 
				m_pFirstAvail->m_uiAvailTime) < uiMaxAvailTicks)
		{
			// All files are newer so we are done.
			
			break;
		}
		
		removeFromList( TRUE,
			m_pFirstAvail, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
	}

	unlockMutex( FALSE);
}

/****************************************************************************
Desc:	Do a partial copy from one file into another file.
****************************************************************************/
RCODE FLMAPI f_copyPartial(
	IF_FileHdl *	pSrcFileHdl,			// Source file handle.
	FLMUINT64		ui64SrcOffset,			// Offset to start copying from.
	FLMUINT64		ui64SrcSize,			// Bytes to copy
	IF_FileHdl *	pDestFileHdl,			// Destination file handle
	FLMUINT64		ui64DestOffset,		// Destination start offset.
	FLMUINT64 *		pui64BytesCopiedRV)	// Returns number of bytes copied
{
	RCODE				rc = NE_FLM_OK;
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

	if( RC_BAD( rc = pSrcFileHdl->seek( ui64SrcOffset, FLM_IO_SEEK_SET,
								&ui64FileOffset)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDestFileHdl->seek( ui64DestOffset, FLM_IO_SEEK_SET,
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
		
		rc = pSrcFileHdl->read( FLM_IO_CURRENT_POS, uiBytesToRead,
										pucBuffer, &uiBytesRead);
										
		if (rc == NE_FLM_IO_END_OF_FILE)
		{
			rc = NE_FLM_OK;
		}
		
		if (RC_BAD( rc))
		{
			rc = RC_SET( NE_FLM_IO_COPY_ERR);
			goto Exit;
		}

		uiBytesWritten = 0;
		if( RC_BAD( rc = pDestFileHdl->write( FLM_IO_CURRENT_POS, uiBytesRead,
									pucBuffer, &uiBytesWritten)))
		{
			if (rc == NE_FLM_IO_DISK_FULL)
			{
				*pui64BytesCopiedRV += uiBytesWritten;
			}
			else
			{
				rc = RC_SET( NE_FLM_IO_COPY_ERR);
			}
			
			goto Exit;
		}
		
		*pui64BytesCopiedRV += uiBytesWritten;

		if( uiBytesRead < uiBytesToRead)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
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

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI f_filecpy(
	const char *	pszSourceFile,
	const char *	pszData)
{
	RCODE				rc = NE_FLM_OK;
	IF_FileHdl *	pFileHdl = NULL;
	F_FileSystem	fileSystem;
	FLMUINT 			uiBytesWritten = 0;

	if( RC_OK( rc = fileSystem.doesFileExist( pszSourceFile)))
	{
		if( RC_BAD( rc = fileSystem.deleteFile( pszSourceFile)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = fileSystem.createFile( pszSourceFile, FLM_IO_RDWR,
		&pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->write( 0, f_strlen( pszData), (void *)pszData,
		&uiBytesWritten)))
	{
		goto Exit;
	}
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->close();
		pFileHdl->Release();
		pFileHdl = NULL;
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI f_filecat(
	const char *	pszSourceFile,
	const char *	pszData)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileHdl *		pFileHdl = NULL;
	F_FileSystem		fileSystem;
	FLMUINT64 			ui64FileSize = 0;
	FLMUINT 				uiBytesWritten = 0;

	if (RC_BAD( rc = fileSystem.doesFileExist( pszSourceFile)))
	{
		if( rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = fileSystem.createFile( pszSourceFile, FLM_IO_RDWR,
				&pFileHdl)))
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
		if( RC_BAD( rc = fileSystem.openFile( pszSourceFile,
			FLM_IO_RDWR, &pFileHdl)))
		{
			goto Exit;
		}
	}

	if ( RC_BAD( rc = pFileHdl->size( &ui64FileSize)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->write( ui64FileSize, f_strlen( pszData),
		(void *)pszData, &uiBytesWritten)))
	{
		goto Exit;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->close();
		pFileHdl->Release();
		pFileHdl = NULL;
	}
	
	return( rc);
}
