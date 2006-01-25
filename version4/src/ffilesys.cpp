//-------------------------------------------------------------------------
// Desc:	File system class.
// Tabs:	3
//
//		Copyright (c) 1998-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ffilesys.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc: Convert local time to UTC
Note: This code was basically copied from prtime.c
****************************************************************************/
FINLINE FLMUINT flmLocalToUTC(
	FLMUINT	uiSeconds)
{
	return( uiSeconds + f_timeGetLocalOffset());
}

/****************************************************************************
Desc:    Create a file, return a file handle to created file.
****************************************************************************/
RCODE F_FileSystemImp::Create(
	const char *	pFilePath,
	FLMUINT				uiIoFlags,
	F_FileHdl **		ppFileHdl)
{
	RCODE				rc = FERR_OK;
	F_FileHdlImp *	pFileHdl = NULL;

	if (RC_BAD( rc = getFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->Create( pFilePath, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (F_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Create a block-oriented file, return a file handle to created file.
****************************************************************************/
RCODE F_FileSystemImp::CreateBlockFile(
	const char *			pFilePath,
	FLMUINT					uiIoFlags,
	FLMUINT					uiBlockSize,
	F_FileHdlImp **		ppFileHdl)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl = NULL;

	if (RC_BAD( rc = getFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

#ifdef FLM_WIN
	pFileHdl->SetBlockSize( uiBlockSize);
#else
	F_UNREFERENCED_PARM( uiBlockSize);
#endif

	if (RC_BAD( rc = pFileHdl->Create( pFilePath, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Create a unique file, return a file handle to created file.
****************************************************************************/
RCODE F_FileSystemImp::CreateUnique(
	char *				pszDirPath,
	const char *		pszFileExtension,
	FLMUINT				uiIoFlags,
	F_FileHdl **		ppFileHdl)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl = NULL;

	if( RC_BAD( rc = getFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileHdl->CreateUnique( 
		pszDirPath, pszFileExtension, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (F_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Open a file, return a file handle to opened file.
****************************************************************************/
RCODE F_FileSystemImp::Open(
	const char *		pFilePath,
	FLMUINT				uiIoFlags,
	F_FileHdl **		ppFileHdl)
{
	RCODE					rc = FERR_OK;
	F_FileHdlImp *		pFileHdl = NULL;

	if (RC_BAD( rc = getFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->Open( pFilePath, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = (F_FileHdl *)pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Open a block-oriented file, return a file handle to opened file.
****************************************************************************/
RCODE F_FileSystemImp::OpenBlockFile(
	const char *			pFilePath,
	FLMUINT					uiIoFlags,
	FLMUINT					uiBlockSize,
	F_FileHdlImp **		ppFileHdl)
{
	RCODE				rc = FERR_OK;
	F_FileHdlImp *	pFileHdl = NULL;

	if (RC_BAD( rc = getFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

#ifdef FLM_WIN
	pFileHdl->SetBlockSize( uiBlockSize);
#else
	F_UNREFERENCED_PARM( uiBlockSize);
#endif

	if (RC_BAD( rc = pFileHdl->Open( pFilePath, uiIoFlags)))
	{
		pFileHdl->Release();
		pFileHdl = NULL;
		goto Exit;
	}

Exit:

	*ppFileHdl = pFileHdl;
	return( rc);
}

/****************************************************************************
Desc:    Open a directory, return a file handle to opened directory.
****************************************************************************/
RCODE F_FileSystemImp::OpenDir(
	const char *		pszDirPath,
	const char *		pszPattern,
	F_DirHdl **			ppDirHdl)
{
	RCODE				rc = FERR_OK;
	F_DirHdl *		pDirHdl = NULL;

	if( (pDirHdl = f_new F_DirHdlImp) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pDirHdl->OpenDir( pszDirPath, pszPattern)))
	{
		pDirHdl->Release();
		pDirHdl = NULL;
	}

Exit:

	*ppDirHdl = pDirHdl;
	return( rc);
}

/****************************************************************************
Desc:		Create a directory (and parent directories if necessary).
****************************************************************************/
RCODE F_FileSystemImp::CreateDir(
	const char *			pszDirPath)
{
	char *		pszParentDir = NULL;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &pszParentDir)))
	{
		goto Exit;
	}

	// Discover the parent directory of the given one

	if( RC_BAD( rc = f_pathReduce( pszDirPath, pszParentDir, NULL)))
	{
		goto Exit;
	}

	// If f_pathReduce couldn't reduce the path at all, then an invalid
	// path was supplied.

	if( f_strcmp( pszDirPath, pszParentDir) == 0)
	{
		rc = RC_SET( FERR_IO_INVALID_PATH);
		goto Exit;
	}

	// If a parent directory was found, and it doesn't already exist, create it

	if( *pszParentDir)
	{
		// If the "parent" is actually a regular file we need to return an error

		if( RC_OK( gv_FlmSysData.pFileSystem->Exists( pszParentDir)))
		{
			if( !gv_FlmSysData.pFileSystem->IsDir( pszParentDir))
			{
				rc = RC_SET( FERR_IO_ACCESS_DENIED);
				goto Exit;
			}
		}

		// Recurse on the parent directory

		else if( RC_BAD( rc = CreateDir( pszParentDir)))
		{
			goto Exit;
		}
	}

#if defined( FLM_WIN)

	if( !CreateDirectory((LPTSTR)pszDirPath, NULL))
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_CREATING_FILE);
	}

#elif defined( FLM_UNIX)

	// Create a new directory with mode parameter 0777.  From the linux
	// man (2) mkdir page:

	//		mode specifies the permissions to use. It is modified by the
	//		process's umask in the usual way: the permissions of the created
	//		file are (mode & ~umask).
	
	if( mkdir( pszDirPath, 0700) == -1 )
	{
		rc = MapErrnoToFlaimErr( errno, FERR_CREATING_FILE);
	}

#elif defined( FLM_NLM)
	
	rc = _CreateDir( pszDirPath);

#endif
	 
Exit:
	if (pszParentDir)
	{
		f_free( &pszParentDir);
	}
	return (rc);
}

/****************************************************************************
Desc:		NetWare implementation to create a directory
Input:
	pPathToDirectory = fully-qualified path to the directory to be created
	Examples:
		In this example, the directory 'myDir' would be created:
		sys:\system\myDir

		Note that ConvertPathString doesn't support server names.  So
		it returns an error on paths like:
			server-name/volume:\directory_1

Return:
	FERR_OK = success, otherwise there was an error
	Note that the parent directory must exist.  If it doesn't, an error will 
	occur.
****************************************************************************/
#ifdef FLM_NLM
RCODE F_FileSystemImp::_CreateDir(
	const char *		pPathToDirectory)
{
	FLMBYTE		pucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		pucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lNewDirectoryID;
	void			*pNotUsed;
	LONG			lErrorCode;
	RCODE			rc = FERR_OK;
	
	f_strcpy( (char *)&pucPseudoLNamePath[1], pPathToDirectory);
	pucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pPathToDirectory);
	
	if( (lErrorCode = ConvertPathString( 0, 0, pucPseudoLNamePath, &lVolumeID,		
		&lPathID, pucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}

	if( (lErrorCode = CreateDirectory( 0, lVolumeID, lPathID, pucLNamePath,
		lLNamePathCount, LONGNameSpace, MaximumDirectoryAccessBits,
		&lNewDirectoryID, &pNotUsed)) != 0)
	{
		goto Exit;
	}

Exit:

	if( lErrorCode)
	{
		rc = MapNWtoFlaimError( lErrorCode, FERR_CREATING_FILE);
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:    Remove a directory.
****************************************************************************/
RCODE F_FileSystemImp::RemoveDir(
	const char *		pszDirPath,
	FLMBOOL				bClear)
{
	RCODE				rc = FERR_OK;
	F_DirHdl *		pDirHdl = NULL;
	char				szFilePath[ F_PATH_MAX_SIZE];

	if( bClear)
	{
		if( RC_BAD( rc = OpenDir( pszDirPath, "*", &pDirHdl)))
		{
			goto Exit;
		}

		// Delete everything in the directory
		
		for( rc = pDirHdl->Next(); RC_OK( rc) ; rc = pDirHdl->Next())
		{
			pDirHdl->CurrentItemPath( szFilePath);
			if( !pDirHdl->CurrentItemIsDir())
			{
				if( RC_BAD( rc = Delete( szFilePath)))
				{
					if( rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
					{
						rc = FERR_OK;
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
					if( rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
					{
						rc = FERR_OK;
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

	if( RC_BAD( rc = RemoveEmptyDir( pszDirPath)))
	{
		goto Exit;
	}

Exit:

	if( pDirHdl)
	{
		pDirHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Remove an empty directory
****************************************************************************/
RCODE F_FileSystemImp::RemoveEmptyDir(
	const char *			pszDirPath)
{
#if defined( FLM_WIN)

	if( !RemoveDirectory( (LPTSTR)pszDirPath))
	{
		return( MapWinErrorToFlaim( GetLastError(), FERR_DELETING_FILE));
	}

	return( FERR_OK);

#elif defined( FLM_UNIX)

	 if( rmdir( pszDirPath) == -1 )
	 {
		 return( MapErrnoToFlaimErr( errno, FERR_DELETING_FILE));
	 }

    return( FERR_OK);

#elif defined( FLM_NLM)
	RCODE			rc = FERR_OK;
	FLMBYTE		pucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		pucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lErrorCode;
	
	f_strcpy( (char *)&pucPseudoLNamePath[1], pszDirPath);
	pucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pszDirPath);
	
	if( (lErrorCode = ConvertPathString( 0, 0, pucPseudoLNamePath, &lVolumeID,		
		&lPathID, pucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}

	if( (lErrorCode = DeleteDirectory( 0, lVolumeID, lPathID, pucLNamePath,
		lLNamePathCount, LONGNameSpace)) != 0)
	{
		goto Exit;
	}

Exit:

	if( lErrorCode)
	{
		rc = MapNWtoFlaimError( lErrorCode, FERR_DELETING_FILE);
	}
	
	return( rc);
#endif
}

/****************************************************************************
Desc:    Determine if a file or directory exists.
****************************************************************************/
RCODE F_FileSystemImp::Exists(
	const char *			pPath)
{
#if defined( FLM_NLM)
	return( NWTestIfFileExists( pPath));
#elif defined( FLM_WIN)
	DWORD		dwFileAttr = GetFileAttributes( (LPTSTR)pPath);

	if( dwFileAttr == (DWORD)-1)
	{
		return( RC_SET( FERR_IO_PATH_NOT_FOUND));
	}

	return( FERR_OK);

#else

	// Check for the file's existence

   if( access( pPath, F_OK) == -1)
	{
		return( MapErrnoToFlaimErr( errno, FERR_CHECKING_FILE_EXISTENCE));
	}

	return( FERR_OK);
#endif
}

/****************************************************************************
Desc:    Get the time stamp of the last modification to this file.
****************************************************************************/
RCODE F_FileSystemImp::GetTimeStamp(
	const char *		pPath,
	FLMUINT *			puiTimeStamp)
{
#if defined( FLM_NLM)
	FLMUINT		uiTmp;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	LONG			lErrorCode;
	struct DirectoryStructure * 
					pFileInfo = NULL;
	RCODE			rc = FERR_OK;

	flmAssert( puiTimeStamp );
	*puiTimeStamp = 0;
	
	f_strcpy( (char *)&ucPseudoLNamePath[1], pPath);
	ucPseudoLNamePath[ 0] = (FLMBYTE)f_strlen( pPath );
	
	if( (lErrorCode = ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID,    
		&lPathID, ucLNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}
	
	if( (lErrorCode = GetEntryFromPathStringBase( 0, lVolumeID, 0, ucLNamePath,
		lLNamePathCount, LONGNameSpace, LONGNameSpace, &pFileInfo, 
		&lDirectoryID)) != 0)
	{
		goto Exit;
	}

	if( pFileInfo)
	{
		FLMUINT			uiTime;
		FLMUINT			uiDate;
		F_TMSTAMP		dateTime;
		LONG				DayMask = 0x001F; 
		LONG				MonthMask = 0x01E0; 
		LONG				YearMask = 0xFE00;
		LONG				SecMask = 0x001F; 
		LONG				MinMask = 0x07E0;
		LONG				HourMask = 0xF800;
		
		//Get the low-order 16 bits
		
		uiTime = (FLMUINT)pFileInfo->DLastUpdatedDateAndTime;
		
		//Get the high-order 16 bits
		
		uiDate = (FLMUINT)(pFileInfo->DLastUpdatedDateAndTime >> 16);

		f_memset( &dateTime, 0, sizeof( dateTime));
		dateTime.second = (FLMBYTE) ((uiTime & SecMask) * 2);
		dateTime.minute = (FLMBYTE) ((uiTime & MinMask) >> 5);
		dateTime.hour = (FLMBYTE) ((uiTime & HourMask) >> 11);
		dateTime.day = (FLMBYTE) (uiDate & DayMask);
		dateTime.month = (FLMBYTE) ((uiDate & MonthMask) >> 5)-1;
		dateTime.year = (FLMUINT16)(((uiDate & YearMask) >> 9) + 1980);
		
		f_timeDateToSeconds( &dateTime, &uiTmp);
		*puiTimeStamp = uiTmp;
		*puiTimeStamp = flmLocalToUTC(*puiTimeStamp);
	}

Exit:

	if( lErrorCode != 0 )
	{
		rc = MapNWtoFlaimError(lErrorCode, FERR_PARSING_FILE_NAME);
	}
	
	return( rc);
	
#elif defined( FLM_WIN)

	RCODE					rc = FERR_OK;
	WIN32_FIND_DATA	find_data;
	FILETIME				ftLocalFileTime;
	SYSTEMTIME			stLastFileWriteTime;
	HANDLE				hSearch = INVALID_HANDLE_VALUE;
	F_TMSTAMP			tmstamp;	

	hSearch = FindFirstFile( (LPTSTR)pPath, &find_data);
	if( hSearch == INVALID_HANDLE_VALUE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_OPENING_FILE);
		
		switch( rc)
		{
	   	case FERR_IO_NO_MORE_FILES:
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
				goto Exit;
			default:
				goto Exit;
		}
	}

	if( FileTimeToLocalFileTime( &(find_data.ftLastWriteTime),
											&ftLocalFileTime) == FALSE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_OPENING_FILE);
		goto Exit;
	}

	if( FileTimeToSystemTime( &ftLocalFileTime, &stLastFileWriteTime) == FALSE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_OPENING_FILE);
		goto Exit;
	}

   f_memset( &tmstamp, 0, sizeof( F_TMSTAMP));
	
   tmstamp.hour = (FLMBYTE)stLastFileWriteTime.wHour;
   tmstamp.minute = (FLMBYTE)stLastFileWriteTime.wMinute;
   tmstamp.second = (FLMBYTE)stLastFileWriteTime.wSecond;
	tmstamp.hundredth = (FLMBYTE)stLastFileWriteTime.wMilliseconds;
   tmstamp.year = (FLMUINT16)stLastFileWriteTime.wYear;
	tmstamp.month = (FLMBYTE)(stLastFileWriteTime.wMonth - 1);
   tmstamp.day = (FLMBYTE)stLastFileWriteTime.wDay;

   f_timeDateToSeconds( &tmstamp, puiTimeStamp);

Exit:

	if( hSearch != INVALID_HANDLE_VALUE)
	{
	   FindClose( hSearch);
	}

	if( RC_OK( rc))
	{
		*puiTimeStamp = flmLocalToUTC( *puiTimeStamp);
	}

   return( rc);

#else

	struct stat   	filestatus;

	if( stat( pPath, &filestatus) == -1)
	{
       return( MapErrnoToFlaimErr( errno, FERR_GETTING_FILE_INFO));
	}

	*puiTimeStamp = (FLMUINT)filestatus.st_mtime;
	return( FERR_OK);

#endif   
}  

/****************************************************************************
Desc:    Determine if a path is a directory.
****************************************************************************/
FLMBOOL F_FileSystemImp::IsDir(
	const char *		pPath)
{
	FLMBOOL		bIsDir = FALSE;

#if defined( FLM_NLM)
	LONG			lIsFile;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;

	f_strcpy( (char *)&ucPseudoLNamePath[1], pPath);
	ucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pPath );
	if( ConvertPathString( 0, 0, ucPseudoLNamePath, &lVolumeID, &lPathID,      
		ucLNamePath, &lLNamePathCount) == 0)
	{
		if( MapPathToDirectoryNumber( 0, lVolumeID, 0, ucLNamePath, 
			lLNamePathCount, LONGNameSpace, &lDirectoryID, &lIsFile) == 0)
		{
			bIsDir = (FLMBOOL)((lIsFile == 0) ? (FLMBOOL)TRUE : (FLMBOOL)FALSE);
		}
	}
	
	return( bIsDir);

#elif defined( FLM_WIN)

	DWORD		FileAttr = GetFileAttributes( (LPTSTR)pPath);
	
	if( FileAttr == 0xFFFFFFFF)
	{
		return( bIsDir);
	}

	return( FileAttr & FILE_ATTRIBUTE_DIRECTORY) ? TRUE : FALSE;

#else

	struct stat     filestatus;

	if( stat( pPath, &filestatus) == -1)
	{
		return bIsDir;
	}

	return( S_ISDIR( filestatus.st_mode)) ? TRUE : FALSE;
#endif
}

/****************************************************************************
Desc:    Delete a file or directory
****************************************************************************/
RCODE F_FileSystemImp::Delete(
	const char *		pPath)
{
#if defined( FLM_NLM)

	return( NWDeleteFile( pPath));

#elif defined( FLM_WIN)

	if( DeleteFile( (LPTSTR)pPath) == FALSE)
	{
		return( MapWinErrorToFlaim( GetLastError(), FERR_DELETING_FILE));
	}
	
	return FERR_OK;

#else

	struct stat FileStat;

	if( stat( pPath, &FileStat) == -1)
	{
       return( MapErrnoToFlaimErr( errno, FERR_GETTING_FILE_INFO));
	}

	if( S_ISDIR(FileStat.st_mode))
	{
		return( RC_SET( FERR_IO_ACCESS_DENIED));
	}

	if( unlink( pPath) == -1)
	{
       return( MapErrnoToFlaimErr( errno, FERR_DELETING_FILE));
	}

    return( FERR_OK);
#endif
}

/****************************************************************************
Desc:    Copy a file.
****************************************************************************/
RCODE F_FileSystemImp::Copy(
	const char *			pSrcFilePath,
	const char *			pDestFilePath,
	FLMBOOL					bOverwrite,
	FLMUINT *				puiBytesCopied)
{
	RCODE				rc = FERR_OK;
	F_FileHdl *		pSrcFileHdl = NULL;
	F_FileHdl *		pDestFileHdl = NULL;
	FLMBOOL			bCreatedDest = FALSE;
	FLMUINT			uiSrcSize;

	// See if the destination file exists.  If it does, see if it is
	// OK to overwrite it.  If so, delete it.

	if( Exists( pDestFilePath) == FERR_OK)
	{
		if( !bOverwrite)
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
		
		if( RC_BAD( rc = Delete( pDestFilePath)))
		{
			goto Exit;
		}
	}

	// Open the source file

	if( RC_BAD( rc = Open( pSrcFilePath, F_IO_RDONLY | F_IO_SH_DENYNONE,
							&pSrcFileHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pSrcFileHdl->Size( &uiSrcSize)))
	{
		goto Exit;
	}

	// Create the destination file

	if( RC_BAD( rc = Create( pDestFilePath, F_IO_RDWR | F_IO_SH_DENYNONE,
									&pDestFileHdl)))
	{
		goto Exit;
	}
	
	bCreatedDest = TRUE;

	// Do the copy

	if( RC_BAD( rc = flmCopyPartial( pSrcFileHdl, 0, uiSrcSize, 
				pDestFileHdl, 0, puiBytesCopied)))
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
			(void)Delete( pDestFilePath);
		}
		
		*puiBytesCopied = 0;
	}
	
	return( rc);
}

/****************************************************************************
Desc:    Rename a file.
****************************************************************************/
RCODE F_FileSystemImp::Rename(
	const char *		pFilePath,
	const char *		pNewFilePath)
{
#if defined( FLM_NLM)

	return( NWRenameFile( pFilePath, pNewFilePath));

#elif defined( FLM_WIN)
	DWORD			error;
	RCODE			rc = FERR_OK;
	FLMUINT		uiBytesCopied;

   // Try to move the file by doing a rename first, otherwise copy the file

	if( (MoveFile( (LPTSTR)pFilePath, (LPTSTR)pNewFilePath)) != TRUE) 
	{
		error = GetLastError();
		
		switch( error)
		{
			case ERROR_NOT_SAME_DEVICE:
			case ERROR_NO_MORE_FILES:
			case NO_ERROR:
			{
				if( Copy( pFilePath, pNewFilePath, TRUE, &uiBytesCopied))
				{
					rc = RC_SET( FERR_IO_COPY_ERR);
				}
				else
				{
					rc = F_FileSystemImp::Delete( pFilePath);
				}
				
				break;
			}
			
			default:
			{
				rc = MapWinErrorToFlaim( error, FERR_RENAMING_FILE);
				break;
			}
		}
	}

	return( rc);

#else
	RCODE			rc = FERR_OK;
	FLMBOOL		bSrcIsDir;
	FLMUINT		uiBytesCopied;

	if( RC_BAD( rc = targetIsDir( (const char *)pFilePath, &bSrcIsDir)))
	{
		return( rc);
	}

	errno = 0;

	if( RC_BAD( renameSafe( pFilePath, pNewFilePath)))
	{
		switch( errno)
		{
			case EXDEV:
			{
				if( bSrcIsDir)
				{
					return( RC_SET( FERR_IO_PATH_CREATE_FAILURE));
				}
				else
				{
					if( Copy( pFilePath, pNewFilePath, TRUE, &uiBytesCopied))
					{
						return( RC_SET( FERR_IO_COPY_ERR));
					}
					
					F_FileSystemImp::Delete( pFilePath);
					return( FERR_OK);
				}
			}

			default:
			{
				if( errno == ENOENT)
				{
					return( RC_SET( FERR_IO_RENAME_FAILURE));
				}
				else
				{
					return( MapErrnoToFlaimErr( errno, FERR_RENAMING_FILE));
				}
			}
		}
	}

	return( FERR_OK);
#endif
}

/****************************************************************************
Desc:    Get the sector size (not supported on all platforms).
****************************************************************************/
RCODE F_FileSystemImp::GetSectorSize(
	const char *		pFileName,
	FLMUINT *			puiSectorSize)
{
#ifdef FLM_NLM

	F_UNREFERENCED_PARM( pFileName);
	*puiSectorSize = 512;
	return( FERR_OK);
	
#elif defined( FLM_WIN)

	RCODE			rc = FERR_OK;
	DWORD			udSectorsPerCluster;
	DWORD			udBytesPerSector;
	DWORD			udNumberOfFreeClusters;
	DWORD			udTotalNumberOfClusters;
	char			szVolume [256];
	char *		pszVolume;
	FLMUINT		uiLen;

	if( !pFileName)
	{
		pszVolume = NULL;
	}
	else
	{
		f_pathParse( pFileName, NULL, szVolume, NULL, NULL);
		
		if( !szVolume [0])
		{
			pszVolume = NULL;
		}
		else
		{
			uiLen = f_strlen( szVolume);
			
			if( szVolume [uiLen - 1] == ':')
			{
				szVolume [uiLen] = '\\';
				szVolume [uiLen + 1] = 0;
			}
			pszVolume = &szVolume [0];
		}
	}

	if( !GetDiskFreeSpace( (LPCTSTR)pszVolume, &udSectorsPerCluster,
			&udBytesPerSector, &udNumberOfFreeClusters,
			&udTotalNumberOfClusters))
	{
		rc = MapWinErrorToFlaim( GetLastError(),
					FERR_INITIALIZING_IO_SYSTEM);
		*puiSectorSize = 0;
		goto Exit;
	}
	
	*puiSectorSize = (FLMUINT)udBytesPerSector;
	
Exit:

	return( rc);

#else

	F_UNREFERENCED_PARM( pFileName);
	*puiSectorSize = 0;
	return( FERR_OK);
	
#endif
}

/****************************************************************************
Desc:	stat tpath to see if it is a directory
****************************************************************************/
#if defined( FLM_UNIX)
RCODE F_FileSystemImp::targetIsDir(
	const char	*		pszPath,
	FLMBOOL *			pbIsDir)
{
	RCODE				rc = FERR_OK;
	struct stat		sbuf;

	*pbIsDir = FALSE;
	
	if( stat( pszPath, &sbuf) < 0)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_IO_ACCESS_DENIED);
	}
	else if( (sbuf.st_mode & S_IFMT) == S_IFDIR)
	{
		*pbIsDir = TRUE;
	}
	
	return( rc);
}
#endif

/****************************************************************************
Desc:		Rename an existing file (typically an "X" locked file to an
			unlocked file) using a safe (non-race) method.  To ensure that
			an existing file is not being overwritten by a rename operation,
			we will first create a new file with the desired name (using the
			CREAT and EXCL options, (ensuring a unique file name)).  Then,
			the source file will be renamed to new name.
****************************************************************************/
#if defined( FLM_UNIX)
RCODE F_FileSystemImp::renameSafe(
	const char *		pszSrcFile,
	const char *		pszDestFile)
{
	RCODE       	rc = FERR_OK;
	struct stat 	temp_stat_buf;

	// There appears to be a bug in newer versions of glibc (it was first
	// noticed in RedHat 8) where the behavior of the reaname() function has
	// changed.  According to the man page, rename should allow you to
	// overwrite an existing directory (ie: rename( "OldDir", "NewDir")
	// should succeed, even if NewDir already exists).  In order to
	// avoid having #ifdefs for individual Linux distributions, we
	// decided not to rely on this behavior, and simply do the
	// existence test ourselves.  Thus, the race condition that this
	// function was supposed to avoid is theoretically possible, but as long
	// as this function only gets called while the database is closed, it won't
	// be a problem.
 
	errno = 0;
	if( stat( pszDestFile, &temp_stat_buf) != -1)
	{
		// If we were able to stat it, then the file obviously exists...
		
		rc = RC_SET( FERR_IO_RENAME_FAILURE);
		goto Exit;
	}
	else
	{
		if( errno != ENOENT)
		{
			// ENOENT means the file didn't exist, which is what we were
			// hoping for
			
			rc = MapErrnoToFlaimErr( errno, FERR_IO_RENAME_FAILURE);
			goto Exit;
		}
	}

	errno = 0;
	if( rename( pszSrcFile, pszDestFile) != 0)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_IO_RENAME_FAILURE);
	}

Exit:

	return( rc);
}
#endif
