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

#include "flaimsys.h"

/****************************************************************************
Desc:    Create a file, return a file handle to created file.
****************************************************************************/
RCODE XFLMAPI F_FileSystem::Create(
	const char *	pszFileName,		// Name of file to be created
	FLMUINT			uiIoFlags,			// Access amd Mode flags
	IF_FileHdl **	ppFileHdl)			// Returns open file handle object.
{
	RCODE			rc = NE_XFLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->Create( pszFileName, uiIoFlags)))
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
RCODE XFLMAPI F_FileSystem::CreateBlockFile(
	const char *	pszFileName,		// Name of file to be created
	FLMUINT			uiIoFlags,			// Access amd Mode flags
	FLMUINT			uiBlockSize,		// Block size for file
	IF_FileHdl **	ppFileHdl)			// Returns open file handle object.
{
	RCODE			rc = NE_XFLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	pFileHdl->SetBlockSize( uiBlockSize);
	
	if (RC_BAD( rc = pFileHdl->Create( pszFileName, uiIoFlags)))
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
RCODE XFLMAPI F_FileSystem::CreateUnique(
	const char *	pszDirName,			// Directory where file is to be created.
	const char *	pszFileExtension,	// Extension to be used on file.
	FLMUINT			uiIoFlags,			// Access and Mode flags.
	IF_FileHdl **	ppFileHdl)			// Returns open file handle object.
{
	RCODE				rc;
	F_FileHdl *		pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if( RC_BAD( rc = pFileHdl->CreateUnique( pszDirName, pszFileExtension,	uiIoFlags)))
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
RCODE XFLMAPI F_FileSystem::Open(
	const char *	pszFileName,	// Name of file to be opened.
	FLMUINT			uiIoFlags,		// Access and Mode flags.
	IF_FileHdl **	ppFileHdl)		// Returns open file handle object.
{
	RCODE			rc = NE_XFLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = pFileHdl->Open( pszFileName, uiIoFlags)))
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
RCODE XFLMAPI F_FileSystem::OpenBlockFile(
	const char *	pszFileName,		// Name of file to be opened.
	FLMUINT			uiIoFlags,		// Access and Mode flags.
	FLMUINT			uiBlockSize,	// Block size for file
	IF_FileHdl **	ppFileHdl)		// Returns open file handle object.
{
	RCODE			rc = NE_XFLM_OK;
	F_FileHdl *	pFileHdl = NULL;

	if ((pFileHdl = f_new F_FileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	pFileHdl->SetBlockSize( uiBlockSize);
	
	if (RC_BAD( rc = pFileHdl->Open( pszFileName, uiIoFlags)))
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
RCODE XFLMAPI F_FileSystem::OpenDir(
	const char *	pszDirName,			// Directory to be opened.
	const char *	pszPattern,			// File name pattern.
	IF_DirHdl **	ppDirHdl)			// Returns open directory handle object.
{
	RCODE			rc = NE_XFLM_OK;
	F_DirHdl *	pDirHdl = NULL;

	if ((pDirHdl = f_new F_DirHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = pDirHdl->OpenDir( pszDirName, pszPattern)))
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
RCODE XFLMAPI F_FileSystem::CreateDir(
	const char *	pszDirName)		// Directory to be created.
{
	RCODE			rc = NE_XFLM_OK;
	F_DirHdl *	pDirHdl = NULL;

	if ((pDirHdl = f_new F_DirHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	rc = pDirHdl->CreateDir( pszDirName);

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
RCODE XFLMAPI F_FileSystem::RemoveDir(
	const char *	pszDirName,	// Directory to delete
	FLMBOOL			bClear)		// OK to delete files if dir is not empty?
{
	RCODE				rc = NE_XFLM_OK;
	IF_DirHdl *		pDirHdl = NULL;
	char				szFilePath[ F_PATH_MAX_SIZE];

	if( bClear)
	{
		// Note: '*' = all files
		if( RC_BAD( rc = OpenDir( pszDirName, (char *)"*", &pDirHdl)))
		{
			goto Exit;
		}

		// Delete everything in the directory
		for ( rc = pDirHdl->Next(); RC_OK( rc) ; rc = pDirHdl->Next())
		{
			pDirHdl->CurrentItemPath( szFilePath);
			if( !pDirHdl->CurrentItemIsDir())
			{
				if( RC_BAD( rc = Delete( szFilePath)))
				{
					if( rc == NE_XFLM_IO_PATH_NOT_FOUND || rc == NE_XFLM_IO_INVALID_FILENAME)
					{
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
			}
			else
			{
				if( RC_BAD( rc = RemoveDir( szFilePath, bClear)))
				{
					if( rc == NE_XFLM_IO_PATH_NOT_FOUND || rc == NE_XFLM_IO_INVALID_FILENAME)
					{
						rc = NE_XFLM_OK;
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

	if ((pDirHdl = f_new F_DirHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pDirHdl->RemoveDir( pszDirName)))
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
RCODE XFLMAPI F_FileSystem::Exists(
	const char *	pszPath)			// Name of file or directory to check.
{
#if defined( FLM_NLM)

	return( flmNetWareTestIfFileExists( pszPath));

#elif defined( FLM_WIN)

	DWORD		dwFileAttr = GetFileAttributes( (LPTSTR)pszPath);

	if( dwFileAttr == (DWORD)-1)
		return RC_SET( NE_XFLM_IO_PATH_NOT_FOUND);

	return NE_XFLM_OK;

#else

   if( access( pszPath, F_OK) == -1)
	{
		return( MapErrnoToFlaimErr( errno, NE_XFLM_CHECKING_FILE_EXISTENCE));
	}

	return( NE_XFLM_OK);
	
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
RCODE XFLMAPI F_FileSystem::GetTimeStamp(
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
	RCODE				rc = NE_XFLM_OK;
	F_TMSTAMP		tmstamp;

	hSearch = FindFirstFile( (LPTSTR)pszPath, &find_data);
	if( hSearch == INVALID_HANDLE_VALUE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), NE_XFLM_OPENING_FILE);
		switch( rc)
		{
	   	case NE_XFLM_IO_NO_MORE_FILES:
				rc = RC_SET( NE_XFLM_IO_PATH_NOT_FOUND);
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
		rc = MapWinErrorToFlaim( GetLastError(), NE_XFLM_OPENING_FILE);
		goto Exit;
	}

	// Convert the local time to a system time so we can map it into
	// a GroupWise Date\Time structure
		
	if( FileTimeToSystemTime( &ftLocalFileTime,
									   &stLastFileWriteTime) == FALSE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), NE_XFLM_OPENING_FILE);
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
	   FindClose( hSearch);

	if( RC_OK(rc))
		*puiTimeStamp = flmLocalToUTC( *puiTimeStamp);

   return( rc);

#else

	struct stat   	filestatus;

	if( stat( pszPath, &filestatus) == -1)
	{
       return( MapErrnoToFlaimErr( errno, NE_XFLM_GETTING_FILE_INFO));
	}

	*puiTimeStamp = (FLMUINT)filestatus.st_mtime; // st_mtime is UTC
	return NE_XFLM_OK;

#endif
}

/****************************************************************************
Desc: Determine if a path is a directory.
****************************************************************************/
FLMBOOL XFLMAPI F_FileSystem::IsDir(
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
RCODE XFLMAPI F_FileSystem::Delete(
	const char *		pszFileName)
{
#if defined( FLM_NLM)

	return( flmNetWareDeleteFile( pszFileName));

#elif defined( FLM_WIN)

	if( DeleteFile( (LPTSTR)pszFileName) == FALSE)
	{
		return( MapWinErrorToFlaim( GetLastError(), NE_XFLM_IO_DELETING_FILE));
	}
	
	return( NE_XFLM_OK);

#else

	struct stat FileStat;

	if( stat( (char *)pszFileName, &FileStat) == -1)
	{
		return( MapErrnoToFlaimErr( errno, NE_XFLM_GETTING_FILE_INFO));
	}

	// Ensure that the path does NOT designate a directory for deletion
	
	if( S_ISDIR(FileStat.st_mode))
	{
		return( RC_SET( NE_XFLM_IO_ACCESS_DENIED));
	}

	// Delete the file
	
	if( unlink( (char *)pszFileName) == -1)
	{
       return( MapErrnoToFlaimErr( errno, NE_XFLM_IO_DELETING_FILE));
	}

	return( NE_XFLM_OK);
	 
#endif
}

/****************************************************************************
Desc:	Copy a file.
****************************************************************************/
RCODE XFLMAPI F_FileSystem::Copy(
	const char *	pszSrcFileName,	// Name of source file to be copied.
	const char *	pszDestFileName,	// Name of destination file.
	FLMBOOL			bOverwrite,			// Overwrite destination file?
	FLMUINT64 *		pui64BytesCopied)	// Returns number of bytes copied.
{
	RCODE				rc = NE_XFLM_OK;
	IF_FileHdl *	pSrcFileHdl = NULL;
	IF_FileHdl *	pDestFileHdl = NULL;
	FLMBOOL			bCreatedDest = FALSE;
	FLMUINT64		ui64SrcSize;

	/*
	See if the destination file exists.  If it does, see if it is
	OK to overwrite it.  If so, delete it.
	*/

	if (Exists( pszDestFileName) == NE_XFLM_OK)
	{
		if (!bOverwrite)
		{
			rc = RC_SET( NE_XFLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		if (RC_BAD( rc = Delete( pszDestFileName)))
		{
			goto Exit;
		}
	}

	// Open the source file.

	if( RC_BAD( rc = Open( pszSrcFileName, XFLM_IO_RDONLY | XFLM_IO_SH_DENYNONE,
								&pSrcFileHdl)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pSrcFileHdl->Size( &ui64SrcSize)))
	{
		goto Exit;
	}

	// Create the destination file.

	if( RC_BAD( rc = Create( pszDestFileName, XFLM_IO_RDWR | XFLM_IO_SH_DENYNONE,
								&pDestFileHdl)))
	{
		goto Exit;
	}
	bCreatedDest = TRUE;

	// Do the copy.

	if( RC_BAD( rc = flmCopyPartial( pSrcFileHdl, 0, ui64SrcSize, 
				pDestFileHdl, 0, pui64BytesCopied)))
	{
		goto Exit;
	}
	
Exit:

	if( pSrcFileHdl)
	{
		pSrcFileHdl->Close();
		pSrcFileHdl->Release();
	}
	
	if( pDestFileHdl)
	{
		pDestFileHdl->Close();
		pDestFileHdl->Release();
	}
	
	if( RC_BAD( rc))
	{
		if( bCreatedDest)
		{
			(void)Delete( pszDestFileName);
		}
		
		*pui64BytesCopied = 0;
	}
	
	return( rc);
}

/****************************************************************************
Desc: Rename a file.
****************************************************************************/
RCODE XFLMAPI F_FileSystem::Rename(
	const char *		pszFileName,
	const char *		pszNewFileName)
{
#if defined( FLM_NLM)

	return( flmNetWareRenameFile( pszFileName, pszNewFileName));

#elif defined( FLM_WIN)

	DWORD			error;
	RCODE			rc = NE_XFLM_OK;
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
				if( Copy( pszFileName, pszNewFileName, TRUE, &ui64BytesCopied))
				{
					rc = RC_SET( NE_XFLM_IO_COPY_ERR);
				}
				else
				{
					rc = F_FileSystem::Delete( pszFileName);
				}
				break;
			default:
				rc = MapWinErrorToFlaim( error, NE_XFLM_RENAMING_FILE);
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
					return( RC_SET( NE_XFLM_IO_PATH_CREATE_FAILURE));
				}
				else
				{
					if( Copy( pszFileName, pszNewFileName, TRUE, &ui64BytesCopied))
					{
						return( RC_SET( NE_XFLM_IO_COPY_ERR));
					}
					
					F_FileSystem::Delete( pszFileName);
					return( NE_XFLM_OK);
				}
			}

			default:
			{
				if( errno == ENOENT)
				{
					return( RC_SET( NE_XFLM_IO_RENAME_FAILURE));
				}
				else
				{
					return( MapErrnoToFlaimErr( errno, NE_XFLM_RENAMING_FILE));
				}
			}
		}
	}

	return( NE_XFLM_OK);
#endif
}

/****************************************************************************
Desc: Get the sector size (not supported on all platforms).
****************************************************************************/
RCODE XFLMAPI F_FileSystem::GetSectorSize(
	const char *	pszFileName,
	FLMUINT *		puiSectorSize)
{
#ifdef FLM_NLM

	F_UNREFERENCED_PARM( pszFileName);
	*puiSectorSize = NETWARE_SECTOR_SIZE;
	return( NE_XFLM_OK);
	
#elif defined( FLM_WIN)

	RCODE			rc = NE_XFLM_OK;
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
		rc = MapWinErrorToFlaim( GetLastError(), NE_XFLM_INITIALIZING_IO_SYSTEM);
		*puiSectorSize = 0;
		goto Exit;
	}
	*puiSectorSize = (FLMUINT)udBytesPerSector;
	
Exit:

	return( rc);

#else
	F_UNREFERENCED_PARM( pszFileName);
	*puiSectorSize = (FLMUINT)sysconf( _SC_PAGESIZE);
	return( NE_XFLM_OK);
#endif
}

/****************************************************************************
Desc: Set the Read-Only Attribute (not supported on all platforms).
****************************************************************************/
RCODE F_FileSystem::SetReadOnly(
	const char *	pszFileName,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = NE_XFLM_OK;

#if defined( FLM_UNIX)
	struct stat		filestatus;

	if( stat( (char *)pszFileName, &filestatus))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
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
		rc = RC_SET( NE_XFLM_FAILURE);
		goto Exit;
	}
	
#elif defined( FLM_WIN)

	DWORD				dwAttr;

	dwAttr = GetFileAttributes( (LPTSTR)pszFileName);
	if( dwAttr == (DWORD)-1)
	{
		rc = RC_SET( NE_XFLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
	if ( bReadOnly)
	{
		dwAttr |= XF_IO_FA_RDONLY;
	}
	else
	{
		dwAttr &= ~XF_IO_FA_RDONLY;
	}
	
	if( !SetFileAttributes( (LPTSTR)pszFileName, dwAttr))
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
		goto Exit;
	}
#elif defined( FLM_NLM)

	if ( RC_BAD( rc = flmNetWareSetReadOnly( pszFileName, bReadOnly)))
	{
		goto Exit;
	}
#else
	rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
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
	RCODE				rc = NE_XFLM_OK;

	*isdir = 0;
	if( stat(tpath, &sbuf) < 0)
	{
		rc = MapErrnoToFlaimErr( errno, NE_XFLM_IO_ACCESS_DENIED);
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

/*************************************************************************
 * There appears to be a bug in newer versions of glibc (it was first
 * noticed in RedHat 8) where the behavior of the reaname() function has
 * changed.  According to the man page, rename should allow you to 
 * overwrite an existing directory (ie: rename( "OldDir", "NewDir") 
 * should succeed, even if NewDir already exists).  In order to 
 * avoid having #ifdefs for individual Linux distributions, we
 * decided not to rely on this behavior, and simply do the 
 * existence test ourselves.  Thus, the race condition that this 
 * function was supposed to avoid is theoretically possible, but as long
 * as this function only gets called while the database is closed, it won't
 * be a problem.              RGM 5 June 2003
 ***************************************************************************/
RCODE F_FileSystem::unix_RenameSafe(
	const char *	pszSrcFile,
	const char *	pszDestFile)
{
	RCODE			rc = NE_XFLM_OK;
	struct stat	temp_stat_buf;
	
	errno = 0;
	if( stat( pszDestFile, &temp_stat_buf) != -1)
	{
		// If we were able to stat it, then the file obviously exists...
		
		rc = RC_SET( NE_XFLM_IO_RENAME_FAILURE);
		goto Exit;
	}
	else
	{
		if (errno != ENOENT)
		{
			// ENOENT means the file didn't exist, which is what we were
			// hoping for.
			
			rc = MapErrnoToFlaimErr( errno, NE_XFLM_IO_RENAME_FAILURE);
			goto Exit;
		}
	}

	errno = 0;
	if( rename( pszSrcFile, pszDestFile) != 0)
	{
		rc = MapErrnoToFlaimErr( errno, NE_XFLM_IO_RENAME_FAILURE);
	}

Exit:

	return( rc);
}
#endif
