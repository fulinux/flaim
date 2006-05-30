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

#define	FHM_AVAIL_LIST		0
#define	FHM_USED_LIST		1
#define	FHM_LNODE_COUNT	2

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void f_hexToNative(
	FLMBYTE		ucHexVal,
	char *		pszNativeChar)
{
	*pszNativeChar = (char)(ucHexVal < 10
										? ucHexVal + NATIVE_ZERO
										: (ucHexVal - 10) + NATIVE_LOWER_A);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void f_setupTime(
	FLMUINT *	puiBaseTime,
	FLMBYTE *	pbyHighByte)
{
	FLMUINT		uiSdTime = 0;
	f_timeGetSeconds( &uiSdTime);
	*pbyHighByte = (FLMBYTE)(uiSdTime >> 24);
	uiSdTime = uiSdTime << 5;
	if( *puiBaseTime < uiSdTime)
		*puiBaseTime = uiSdTime;
}

/****************************************************************************
Desc:
****************************************************************************/
class F_FileSystem : public IF_FileSystem
{
public:

	F_FileSystem()
	{
		m_bCanDoAsync = FALSE;
	}

	virtual ~F_FileSystem()
	{
	}
	
	RCODE setup( void);

	RCODE FLMAPI createFile(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		IF_FileHdl **			ppFile);

	RCODE FLMAPI createBlockFile(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		FLMUINT					uiBlockSize,
		IF_FileHdl **			ppFile);

	RCODE FLMAPI createUniqueFile(
		char *					pszPath,
		const char *			pszFileExtension,
		FLMUINT					uiIoFlags,
		IF_FileHdl **			ppFile);

	RCODE FLMAPI createLockFile(
		const char *			pszPath,
		IF_FileHdl **			ppLockFileHdl);
		
	RCODE FLMAPI openFile(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		IF_FileHdl **			ppFile);

	RCODE FLMAPI openBlockFile(
		const char *			pszFileName,
		FLMUINT					uiIoFlags,
		FLMUINT					uiBlockSize,
		IF_FileHdl **			ppFile);

	RCODE FLMAPI openDir(
		const char *			pszDirName,
		const char *			pszPattern,
		IF_DirHdl **			ppDir);

	RCODE FLMAPI createDir(
		const char *			pszDirName);

	RCODE FLMAPI removeDir(
		const char *			pszDirName,
		FLMBOOL					bClear = FALSE);

	RCODE FLMAPI doesFileExist(
		const char *			pszFileName);

	FLMBOOL FLMAPI isDir(
		const char *			pszFileName);

	RCODE FLMAPI getFileTimeStamp(
		const char *			pszFileName,
		FLMUINT *				puiTimeStamp);

	RCODE FLMAPI deleteFile(
		const char *			pszFileName);

	RCODE FLMAPI deleteMultiFileStream(
		const char *			pszDirectory,
		const char *			pszBaseName);
		
	RCODE FLMAPI copyFile(
		const char *			pszSrcFileName,
		const char *			pszDestFileName,
		FLMBOOL					bOverwrite,
		FLMUINT64 *				pui64BytesCopied);

	RCODE FLMAPI copyPartialFile(
		IF_FileHdl *			pSrcFileHdl,
		FLMUINT64				ui64SrcOffset,
		FLMUINT64				ui64SrcSize,
		IF_FileHdl *			pDestFileHdl,
		FLMUINT64				ui64DestOffset,
		FLMUINT64 *				pui64BytesCopiedRV);
	
	RCODE FLMAPI renameFile(
		const char *			pszFileName,
		const char *			pszNewFileName);

	void FLMAPI pathParse(
		const char *			pszPath,
		char *					pszServer,
		char *					pszVolume,
		char *					pszDirPath,
		char *					pszFileName);

	RCODE FLMAPI pathReduce(
		const char *			pszSourcePath,
		char *					pszDestPath,
		char *					pszString);

	RCODE FLMAPI pathAppend(
		char *					pszPath,
		const char *			pszPathComponent);

	RCODE FLMAPI pathToStorageString(
		const char *			pszPath,
		char *					pszString);

	void FLMAPI pathCreateUniqueName(
		FLMUINT *				puiTime,
		char *					pszFileName,
		const char *			pszFileExt,
		FLMBYTE *				pHighChars,
		FLMBOOL					bModext);

	FLMBOOL FLMAPI doesFileMatch(
		const char *			pszFileName,
		const char *			pszTemplate);

	RCODE FLMAPI getSectorSize(
		const char *			pszFileName,
		FLMUINT *				puiSectorSize);

	RCODE FLMAPI setReadOnly(
		const char *			pszFileName,
		FLMBOOL					bReadOnly);

	FLMBOOL FLMAPI canDoAsync( void);
		
private:

	RCODE removeEmptyDir(
		const char *			pszDirName);

#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	RCODE renameSafe(
		const char *			pszSrcFile,
		const char *			pszDestFile);

	RCODE targetIsDir(
		const char	*			pszPath,
		FLMBOOL *				pbIsDir);
#endif

	FLMBOOL				m_bCanDoAsync;
};
	
FSTATIC FLMBOOL f_canReducePath(
	const char *	pszSource);

FSTATIC const char * f_findFileNameStart(
	const char * 	pszPath);

FSTATIC char * f_getPathComponent(
	char **			ppszPath,
	FLMUINT *		puiEndChar);

/****************************************************************************
Desc:	Returns TRUE if character is a "slash" separator
****************************************************************************/
FINLINE FLMBOOL f_isSlashSeparator(
	char	cChar)
{
#ifdef FLM_UNIX
	return( cChar == '/' ? TRUE : FALSE);
#else
	return( cChar == '/' || cChar == '\\' ? TRUE : FALSE);
#endif
}

/****************************************************************************
Desc:	Return a pointer to the next path component in ppszPath.
****************************************************************************/
FSTATIC char * f_getPathComponent(
	char **			ppszPath,
	FLMUINT *		puiEndChar)
{
	char *	pszComponent;
	char *	pszEnd;
	
	pszComponent = pszEnd = *ppszPath;
	if (f_isSlashSeparator( *pszEnd))
	{
		// handle the condition of sys:\system   the colon would have terminated
		// the previous token, to pComponent would now be pointing at the '\'.
		// We need to move past the '\' to find the next token.

		pszEnd++;
	}

	// Find the end of the path component

	while (*pszEnd)
	{
		if (f_isSlashSeparator( *pszEnd)
#ifndef FLM_UNIX
			|| *pszEnd == ':'
#endif
			)
		{
			break;
		}
		pszEnd++;
	}

	if (*pszEnd)
	{

		// A delimiter was found, assume that there is another path component
		// after this one.
		// Return a pointer to the beginning of the next path component

		*ppszPath = pszEnd + 1;

		*puiEndChar = *pszEnd;

		// NULL terminate the path component

		*pszEnd = 0;
	}
	else
	{

		// There is no "next path component" so return a pointer to the 
		// NULL-terminator

		*ppszPath = pszEnd;
		*puiEndChar = 0;
	}	
	
	// Return the path component

	return( pszComponent);
}

/****************************************************************************
Desc:	Will determine whether any format of (UNC, drive based, NetWare
		UNC) path can be reduced any further.
****************************************************************************/
FSTATIC FLMBOOL f_canReducePath(
	const char *	pszSource)
{
#if defined FLM_UNIX
	F_UNREFERENCED_PARM( pszSource);
	return( TRUE);
#else
	FLMBOOL			bCanReduce;
	const char *	pszTemp = pszSource;

	// Determine whether the passed path is UNC or not
	// (UNC format is: \\FileServer\Volume\Path).

	if (f_strncmp( "\\\\", pszSource, 2 ) == 0)
	{
		FLMUINT	uiSlashCount = 0;

		pszTemp += 2;

		// Search forward for at least two slash separators
		// If we find at least two, the path can be reduced.

		bCanReduce = FALSE;
		while (*pszTemp)
		{
			pszTemp++;
			if (f_isSlashSeparator( *pszTemp))
			{
				++uiSlashCount;
				if (uiSlashCount == 2)
				{
					bCanReduce = TRUE;
					break;
				}
			}
		}
	}
	else
	{
		bCanReduce = TRUE;

		// Search forward for the colon.

		while (*pszTemp)
		{
			if (*pszTemp == ':')
			{

				// If nothing comes after the colon,
				// we can't reduce any more.

				if (*(pszTemp + 1) == 0)
				{
					bCanReduce = FALSE;
				}
				break;
			}
			pszTemp++;
		}
	}

	return( bCanReduce);
#endif
}

/****************************************************************************
Desc:	Return pointer to start of filename part of path.
		Search for the last slash separator.
****************************************************************************/
FSTATIC const char * f_findFileNameStart(
	const char * 	pszPath)
{
	const char *	pszFileNameStart;

	pszFileNameStart = pszPath;
	while (*pszPath)
	{
		if (f_isSlashSeparator( *pszPath))
		{
			pszFileNameStart = pszPath + 1;
		}
		pszPath++;
	}
	return( pszFileNameStart);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_allocFileSystem(
	IF_FileSystem **		ppFileSystem)
{
	if( (*ppFileSystem = f_new F_FileSystem) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI FlmGetFileSystem(
	IF_FileSystem **		ppFileSystem)
{
	*ppFileSystem = f_getFileSysPtr();
	(*ppFileSystem)->AddRef();
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE f_allocFileHdl(
	F_FileHdl **		ppFileHdl)
{
	if( (*ppFileHdl = f_new F_FileHdl) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileSystem::setup( void)
{
#if defined( FLM_WIN)
	OSVERSIONINFO		versionInfo;

	versionInfo.dwOSVersionInfoSize = sizeof( OSVERSIONINFO);
	if( !GetVersionEx( &versionInfo))
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_FAILURE));
	}

	// Async writes are not supported on Win32s (3.1) or
	// Win95, 98, ME, etc.

	m_bCanDoAsync =
		(versionInfo.dwPlatformId != VER_PLATFORM_WIN32_WINDOWS &&
		 versionInfo.dwPlatformId != VER_PLATFORM_WIN32s)
		 ? TRUE
		 : FALSE;
#else
	m_bCanDoAsync = TRUE;
#endif

	return( NE_FLM_OK);
}
	
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
	
	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->create( pszFileName, uiIoFlags)))
	{
		goto Exit;
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

/****************************************************************************
Desc:    Create a block-oriented file, return a file handle to created file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createBlockFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	FLMUINT			uiBlockSize,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}

	pFileHdl->setBlockSize( uiBlockSize);
	
	if (RC_BAD( rc = pFileHdl->create( pszFileName, uiIoFlags)))
	{
		goto Exit;
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

/****************************************************************************
Desc:	Create a unique file, return a file handle to created file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createUniqueFile(
	char *			pszPath,
	const char *	pszFileExtension,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pFileHdl->createUnique( pszPath, 
			pszFileExtension,	uiIoFlags)))
	{
		goto Exit;
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

/****************************************************************************
Desc: Open a file, return a file handle to opened file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::openFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pFileHdl->open( pszFileName, uiIoFlags)))
	{
		goto Exit;
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

/****************************************************************************
Desc:	Open a block-oriented file, return a file handle to opened file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::openBlockFile(
	const char *	pszFileName,
	FLMUINT			uiIoFlags,
	FLMUINT			uiBlockSize,
	IF_FileHdl **	ppFileHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = f_allocFileHdl( &pFileHdl)))
	{
		goto Exit;
	}
	
	pFileHdl->setBlockSize( uiBlockSize);
	
	if (RC_BAD( rc = pFileHdl->open( pszFileName, uiIoFlags)))
	{
		goto Exit;
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

/****************************************************************************
Desc:    Open a directory, return a file handle to opened directory.
****************************************************************************/
RCODE FLMAPI F_FileSystem::openDir(
	const char *	pszDirName,
	const char *	pszPattern,
	IF_DirHdl **	ppDirHdl)
{
	RCODE				rc = NE_FLM_OK;
	F_DirHdl *		pDirHdl = NULL;
	
	if( RC_BAD( rc = f_allocDirHdl( &pDirHdl)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pDirHdl->openDir( pszDirName, pszPattern)))
	{
		goto Exit;
	}

	*ppDirHdl = (IF_DirHdl *)pDirHdl;
	pDirHdl = NULL;
	
Exit:

	if( pDirHdl)
	{
		pDirHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:    Create a directory.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createDir(
	const char *	pszDirName)
{
	RCODE				rc = NE_FLM_OK;
	F_DirHdl *		pDirHdl = NULL;
	
	if( RC_BAD( rc = f_allocDirHdl( &pDirHdl)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pDirHdl->createDir( pszDirName)))
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

		for( rc = pDirHdl->next(); RC_OK( rc) ; rc = pDirHdl->next())
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

	if( RC_BAD( rc = removeEmptyDir( pszDirName)))
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
Desc:	Remove an empty directory
****************************************************************************/
RCODE F_FileSystem::removeEmptyDir(
	const char *			pszDirPath)
{
#if defined( FLM_WIN)

	if( !RemoveDirectory( (LPTSTR)pszDirPath))
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	 if( rmdir( pszDirPath) == -1 )
	 {
		 return( f_mapPlatformError( errno, NE_FLM_IO_DELETING_FILE));
	 }

    return( NE_FLM_OK);
	 
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareRemoveDir( pszDirPath));

#endif
}

/****************************************************************************
Desc:	Determine if a file or directory exists.
****************************************************************************/
RCODE FLMAPI F_FileSystem::doesFileExist(
	const char *	pszPath)
{
#if defined( FLM_WIN)

	DWORD		dwFileAttr = GetFileAttributes( (LPTSTR)pszPath);

	if( dwFileAttr == (DWORD)-1)
		return RC_SET( NE_FLM_IO_PATH_NOT_FOUND);

	return NE_FLM_OK;

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

   if( access( pszPath, F_OK) == -1)
	{
		return( f_mapPlatformError( errno, NE_FLM_CHECKING_FILE_EXISTENCE));
	}

	return( NE_FLM_OK);
	
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareTestIfFileExists( pszPath));
	
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
#if defined( FLM_WIN)

	WIN32_FIND_DATA find_data;
	FILETIME			ftLocalFileTime;
	SYSTEMTIME		stLastFileWriteTime;
	HANDLE			hSearch = INVALID_HANDLE_VALUE;
	RCODE				rc = NE_FLM_OK;
	F_TMSTAMP		tmstamp;

	hSearch = FindFirstFile( (LPTSTR)pszPath, &find_data);
	if( hSearch == INVALID_HANDLE_VALUE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		switch( rc)
		{
	   	case NE_FLM_IO_NO_MORE_FILES:
				rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
				goto Exit;
			default:
				goto Exit;
		}
	}

	// Convert it to a local time, so we can adjust based on our own
	// GroupWise time zone.
		
	if( FileTimeToLocalFileTime( &(find_data.ftLastWriteTime),
											&ftLocalFileTime) == FALSE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

	// Convert the local time to a system time so we can map it into
	// a GroupWise Date\Time structure
		
	if( FileTimeToSystemTime( &ftLocalFileTime,
									   &stLastFileWriteTime) == FALSE)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
		goto Exit;
	}

   // Fill the time stamp structure
	
   f_memset( &tmstamp, 0, sizeof( F_TMSTAMP));
	
   tmstamp.hour = (FLMBYTE)stLastFileWriteTime.wHour;
   tmstamp.minute = (FLMBYTE)stLastFileWriteTime.wMinute;
   tmstamp.second = (FLMBYTE)stLastFileWriteTime.wSecond;
	tmstamp.hundredth = (FLMBYTE)stLastFileWriteTime.wMilliseconds;
   tmstamp.year = (FLMUINT16)stLastFileWriteTime.wYear;
	tmstamp.month = (FLMBYTE)(stLastFileWriteTime.wMonth - 1);
   tmstamp.day = (FLMBYTE)stLastFileWriteTime.wDay;

   // Convert and return the file time stamp as seconds since January 1, 1970
	
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

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	struct stat   	filestatus;

	if( stat( pszPath, &filestatus) == -1)
	{
       return( f_mapPlatformError( errno, NE_FLM_GETTING_FILE_INFO));
	}

	*puiTimeStamp = (FLMUINT)filestatus.st_mtime; // st_mtime is UTC
	return NE_FLM_OK;
	
#elif defined( FLM_RING_ZERO_NLM)

	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiTmp;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	LONG			lErrorCode;
	struct DirectoryStructure * pFileInfo = NULL;

	flmAssert( puiTimeStamp);
	*puiTimeStamp = 0;
	
	f_strcpy( (char *)&ucPseudoLNamePath[1], pszPath);
	ucPseudoLNamePath[ 0] = (FLMBYTE)f_strlen( pszPath );
	
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
		*puiTimeStamp = f_localTimeToUTC(*puiTimeStamp);
	}

Exit:

	if( lErrorCode != 0 )
	{
		rc = f_mapPlatformError( lErrorCode, NE_FLM_PARSING_FILE_NAME);
	}
	
	return( rc);

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

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	struct stat     filestatus;

	if( stat( (char *)pszDirName, &filestatus) == -1)
	{
		return FALSE;
	}

	return ( S_ISDIR( filestatus.st_mode)) ? TRUE : FALSE;
	
#elif defined( FLM_RING_ZERO_NLM)
	
	LONG			lIsFile;
	FLMBYTE		ucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		ucLNamePath[ F_PATH_MAX_SIZE];
	LONG			lVolumeID;
	LONG			lPathID;
	LONG			lLNamePathCount;
	LONG			lDirectoryID;
	FLMBOOL		bIsDir = FALSE;

	f_strcpy( (char *)&ucPseudoLNamePath[1], pszDirName);
	ucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pszDirName);
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
	
#endif
}

/****************************************************************************
Desc:    Delete a file or directory
****************************************************************************/
RCODE FLMAPI F_FileSystem::deleteFile(
	const char *		pszFileName)
{
#if defined( FLM_WIN)

	if( DeleteFile( (LPTSTR)pszFileName) == FALSE)
	{
		return( f_mapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}
	
	return( NE_FLM_OK);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	struct stat FileStat;

	if( stat( (char *)pszFileName, &FileStat) == -1)
	{
		return( f_mapPlatformError( errno, NE_FLM_GETTING_FILE_INFO));
	}

	// Ensure that the path does NOT designate a directory for deletion
	
	if( S_ISDIR(FileStat.st_mode))
	{
		return( RC_SET( NE_FLM_IO_ACCESS_DENIED));
	}

	// Delete the file
	
	if( unlink( (char *)pszFileName) == -1)
	{
       return( f_mapPlatformError( errno, NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);
	
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareDeleteFile( pszFileName));
	 
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

	if( RC_BAD( rc = copyPartialFile( pSrcFileHdl, 0, ui64SrcSize, 
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
Desc:	Do a partial copy from one file into another file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::copyPartialFile(
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
Desc: Rename a file.
****************************************************************************/
RCODE FLMAPI F_FileSystem::renameFile(
	const char *		pszFileName,
	const char *		pszNewFileName)
{
#if defined( FLM_WIN)

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
				rc = f_mapPlatformError( error, NE_FLM_RENAMING_FILE);
				break;
		}
	}

	return( rc);

#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)

	RCODE			rc;
	FLMBOOL		bSrcIsDir;
	FLMUINT64	ui64BytesCopied;

	if( RC_BAD( rc = targetIsDir( (char*)pszFileName, &bSrcIsDir)))
	{
		return( rc);
	}

	errno = 0;

	if( RC_BAD( renameSafe( pszFileName, pszNewFileName)))
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
					return( f_mapPlatformError( errno, NE_FLM_RENAMING_FILE));
				}
			}
		}
	}

	return( NE_FLM_OK);
	
#elif defined( FLM_RING_ZERO_NLM)

	return( f_netwareRenameFile( pszFileName, pszNewFileName));

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
	*puiSectorSize = FLM_NLM_SECTOR_SIZE;
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
		rc = f_mapPlatformError( GetLastError(), NE_FLM_INITIALIZING_IO_SYSTEM);
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
RCODE FLMAPI F_FileSystem::setReadOnly(
	const char *	pszFileName,
	FLMBOOL			bReadOnly)
{
	RCODE				rc = NE_FLM_OK;

#if defined( FLM_WIN)

	DWORD				dwAttr;

	dwAttr = GetFileAttributes( (LPTSTR)pszFileName);
	if( dwAttr == (DWORD)-1)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
	if( bReadOnly)
	{
		dwAttr |= FILE_ATTRIBUTE_READONLY;
	}
	else
	{
		dwAttr &= ~FILE_ATTRIBUTE_READONLY;
	}
	
	if( !SetFileAttributes( (LPTSTR)pszFileName, dwAttr))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
#elif defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
	struct stat		filestatus;

	if( stat( (char *)pszFileName, &filestatus))
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
	if( bReadOnly)
	{
		filestatus.st_mode &= ~S_IWUSR;
	}
	else
	{
		filestatus.st_mode |= S_IWUSR;
	}
	
	if( chmod( (char *)pszFileName, filestatus.st_mode))
	{
		rc = RC_SET( NE_FLM_FAILURE);
		goto Exit;
	}
#else
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( bReadOnly);

	rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
	goto Exit;
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FLMAPI F_FileSystem::canDoAsync( void)
{
	return( m_bCanDoAsync);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE F_FileSystem::targetIsDir(
	const char	*	pszPath,
	FLMBOOL *		pbIsDir)
{
	struct stat		sbuf;
	RCODE				rc = NE_FLM_OK;

	*pbIsDir = FALSE;
	if( stat( pszPath, &sbuf) < 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_IO_ACCESS_DENIED);
	}
	else if( (sbuf.st_mode & S_IFMT) == S_IFDIR)
	{
		*pbIsDir = TRUE;
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
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE F_FileSystem::renameSafe(
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
			
			rc = f_mapPlatformError( errno, NE_FLM_IO_RENAME_FAILURE);
			goto Exit;
		}
	}

	errno = 0;
	if( rename( pszSrcFile, pszDestFile) != 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_IO_RENAME_FAILURE);
	}

Exit:

	return( rc);
}
#endif

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
	FLMUINT64 			ui64FileSize = 0;
	FLMUINT 				uiBytesWritten = 0;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();

	if (RC_BAD( rc = pFileSystem->doesFileExist( pszSourceFile)))
	{
		if( rc == NE_FLM_IO_PATH_NOT_FOUND)
		{
			if( RC_BAD( rc = pFileSystem->createFile( 
				pszSourceFile, FLM_IO_RDWR, &pFileHdl)))
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
		if( RC_BAD( rc = pFileSystem->openFile( pszSourceFile,
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
		pFileHdl->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Split the path into its components
Output:
		pServer - pointer to a buffer to hold the server name
		pVolume - pointer to a buffer to hold the volume name
		pDirPath  - pointer to a buffer to hold the path
		pFileName pointer to a buffer to hold the filename

		All of the output parameters are optional.  If you do not want one
		of the components, simply give a NULL pointer.

Note: if the input path has no file name, d:\dir_1 for example, then
		pass a NULL pointer for pFileName.  Otherwise dir_1 will be returned
		as pFileName.

		The server name may be ommitted in the input path:
			sys:\system\autoexec.ncf

		UNC paths of the form:
			\\server-name\volume-name\dir_1\dir_2\file.ext
		are supported.

		DOS paths of the form:
			d:\dir_1\dir_2\file.ext
		are also supported.

Example:
		Given this input:  orm-prod48/sys:\system\autoexec.ncf
		The output would be:
			pServer = "orm-prod48"
			pVolume = "sys:"
			pDirPath  = "\system"
			pFileName "autoexec.ncf"
****************************************************************************/
void FLMAPI F_FileSystem::pathParse(
	const char *		pszInputPath,
	char *				pszServer,
	char *				pszVolume,
	char *				pszDirPath,
	char *				pszFileName)
{
	char					szInput[ F_PATH_MAX_SIZE];
	char *				pszNext;
	char *				pszColon;
	char *				pszComponent;
	FLMUINT				uiEndChar;
	FLMBOOL				bUNC = FALSE;
	IF_FileSystem *	pFileSystem = f_getFileSysPtr();
	
	// Initialize return buffers

	if (pszServer)
	{
		*pszServer = 0;
	}
	if (pszVolume)
	{
		*pszVolume = 0;
	}
	if (pszDirPath)
	{
		*pszDirPath = 0;
	}
	if (pszFileName)
	{

		// Get the file name

		*pszFileName = 0;
		pFileSystem->pathReduce( pszInputPath, szInput, pszFileName);
	}
	else
	{
		f_strcpy( szInput, pszInputPath);
	}
	
	// Split out the rest of the components

	pszComponent = &szInput [0];

	// Is this a UNC path?

	if (szInput[0] == '\\' && szInput[1] == '\\')
	{

		// Yes, assume a UNC path

		pszComponent += 2;
		bUNC = TRUE;
	}

	pszNext = pszColon = pszComponent;

	// Is there a ':' in the szInput path?

	while (*pszColon && *pszColon != ':')
	{
		pszColon++;
	}
	if (*pszColon || bUNC)
	{
		
		// Yes, assume there is a volume in the path
		
		pszComponent = f_getPathComponent( &pszNext, &uiEndChar);
		if (uiEndChar != ':')
		{
			// Assume that this component is the server

			if (pszServer)
			{
				f_strcpy( pszServer, pszComponent);
			}

			// Get the next component

			pszComponent = f_getPathComponent( &pszNext, &uiEndChar);
		}

		// Assume that this component is the volume

		if (pszVolume)
		{
			char *	pszSrc = pszComponent;
			char *	pszDst = pszVolume;

			while (*pszSrc)
			{
				*pszDst++ = *pszSrc++;
			}
			*pszDst++ = ':';
			*pszDst = 0;
		}

		// For UNC paths, the leading '\' of the path is set to 0 by
		// f_getPathComponent.  This code restores the leading '\'.

		if (f_isSlashSeparator( (char)uiEndChar))
		{
			*(--pszNext) = (char)uiEndChar;
		}
	}

	// Assume that all that is left of the input is the path

	if (pszDirPath)
	{
		f_strcpy( pszDirPath, pszNext);
	}
}	

/****************************************************************************
Desc:		This function will strip off the filename or trailing
			directory of a path.  The stripped component of the path will
			be placed into the area pointed at by string.  The source
			path will not be modified.  The dest path will contain the
			remainder of the stripped path.  A stripped path can be processed
			repeatedly by this function until there is no more path to reduce.
			If the string is set to NULL, the copying of the stripped portion of
			the path will be bypassed by the function.

Notes:	This function handles drive based, UNC, Netware, and UNIX type
			paths.
****************************************************************************/
RCODE FLMAPI F_FileSystem::pathReduce(
	const char *	pszPath,
	char *			pszDir,
	char * 			pszPathComponent)
{
	RCODE				rc = NE_FLM_OK;
	const char *	pszFileNameStart;
	char				szLocalPath[ F_PATH_MAX_SIZE];
	FLMUINT			uiLen;

	// Check for valid path pointers

	if( !pszPath || !pszDir)
	{
		rc = RC_SET( NE_FLM_INVALID_PARM);
		goto Exit;
	}

	if ((uiLen = f_strlen( pszPath)) == 0)
	{
		rc = RC_SET( NE_FLM_IO_CANNOT_REDUCE_PATH);
		goto Exit;
	}

	// Trim out any trailing slash separators
	
	if( f_isSlashSeparator( pszPath [uiLen - 1]))
	{
		f_strcpy( szLocalPath, pszPath);
		
		while( f_isSlashSeparator( szLocalPath[ uiLen - 1]))
		{
			szLocalPath[ --uiLen] = 0;
			if( !uiLen)
			{
				rc = RC_SET( NE_FLM_IO_CANNOT_REDUCE_PATH);
				goto Exit;
			}
		}
		
		pszPath = szLocalPath;
	}

	if( f_canReducePath( pszPath))
	{
		// Search for a slash or beginning of path

		pszFileNameStart = f_findFileNameStart( pszPath);

		// Copy the sliced portion of the path if requested by caller

		if( pszPathComponent)
		{
			f_strcpy( pszPathComponent, pszFileNameStart);
		}

		// Copy the reduced source path to the dir path

		if (pszFileNameStart > pszPath)
		{
			uiLen = (FLMUINT)(pszFileNameStart - pszPath);
			f_memcpy( pszDir, pszPath, uiLen);

			if (uiLen >= 2 && f_isSlashSeparator( pszDir [uiLen - 1])
#ifndef FLM_UNIX
				 && pszDir [uiLen - 2] != ':'
#endif
				 )
			{
				// Trim off the trailing path separator

				pszDir [uiLen - 1] = 0;
			}
			else
			{
				pszDir [uiLen] = 0;
			}
		}
		else
		{
			*pszDir = 0;
		}
	}
	else
	{
		// We've found the drive id or server\volume specifier.

		if (pszPathComponent)
		{
			f_strcpy( pszPathComponent, pszPath);
		}
		
		*pszDir = 0;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:   Internal function for WpioPathBuild() and WpioPathModify().
	     Appends string the path & adds a path delimiter if necessary.
In:     *path     = pointer to an IO_PATH
	     *string   = pointer to a NULL terminated string
	     *end_ptr  = pointer to the end of the IO_PATH which is being built.
****************************************************************************/
RCODE FLMAPI F_FileSystem::pathAppend(
	char *			pszPath,
	const char *	pszPathComponent)
{

	// Don't put a slash separator if pszPath is empty

	if (*pszPath)
	{
		FLMUINT		uiStrLen = f_strlen( pszPath);
		char *		pszEnd = pszPath + uiStrLen - 1;

		if (!f_isSlashSeparator( *pszEnd))
		{

		   // Check for maximum path size - 2 is for slash separator
			// and null byte.

			if (uiStrLen + 2 + f_strlen( pszPathComponent) > F_PATH_MAX_SIZE)
			{
				return RC_SET( NE_FLM_IO_PATH_TOO_LONG);
			}

			pszEnd++;
#if defined( FLM_UNIX)
			*pszEnd = '/';
#else
			*pszEnd = '\\';
#endif
		}
		else
		{

		   // Check for maximum path size +1 is for null byte.

			if (uiStrLen + 1 + f_strlen( pszPathComponent) > F_PATH_MAX_SIZE)
			{
				return RC_SET( NE_FLM_IO_PATH_TOO_LONG);
			}
		}

		f_strcpy( pszEnd + 1, pszPathComponent);
	}
	else
	{
		f_strcpy( pszPath, pszPathComponent);
	}

   return( NE_FLM_OK);
}

/****************************************************************************
Desc:	Convert an PATH into a fully qualified, storable C string
		reference to a file or directory.
In:	pszPath - the path to convert.
		pszStorageString - a pointer to a string that is atleast 
		F_PATH_MAX_SIZE in size
****************************************************************************/
RCODE FLMAPI F_FileSystem::pathToStorageString(
	const char *	pszPath,
	char *			pszStorageString)
{
#ifdef FLM_WIN

	char *	pszNamePart;

	if (GetFullPathName( (LPCSTR)pszPath,
							(DWORD)F_PATH_MAX_SIZE - 1,
							(LPSTR)pszStorageString,
							(LPSTR *)&pszNamePart) != 0)
	{

	}
	else
	{
		// Convert to upper case.

		while (*pszPath)
		{
			*pszStorageString++ = *pszPath;
			pszPath++;
		}
		
		*pszStorageString = 0;
	}
	
	return( NE_FLM_OK);

#elif defined( FLM_RING_ZERO_NLM)

	while (*pszPath)
	{
		*pszStorageString++ = f_toupper( *pszPath);
		pszPath++;
	}
	*pszStorageString = 0;
	return( NE_FLM_OK);
	
#else

	char			szFile[ F_PATH_MAX_SIZE];
	char			szDir[ F_PATH_MAX_SIZE];
	char *		pszRealPath = NULL;
	RCODE			rc = NE_FLM_OK;

	if (RC_BAD( rc = pathReduce( pszPath, szDir, szFile)))
	{
		goto Exit;
	}

	if (!szDir [0])
	{
		szDir [0] = '.';
		szDir [1] = '\0';
	}

	if (RC_BAD( rc = f_alloc( (FLMUINT)PATH_MAX, &pszRealPath)))
	{
		goto Exit;
	}

	if (!realpath( (char *)szDir, (char *)pszRealPath))
	{
		rc = f_mapPlatformError( errno, NE_FLM_PARSING_FILE_NAME);
		goto Exit;
	}

	if (f_strlen( pszRealPath) >= F_PATH_MAX_SIZE)
	{
		rc = RC_SET( NE_FLM_IO_PATH_TOO_LONG);
		goto Exit;
	}

	f_strcpy( pszStorageString, pszRealPath);

	if (RC_BAD( rc = pathAppend( pszStorageString, szFile)))
	{
		goto Exit;
	}

Exit:

	if (pszRealPath)
	{
		f_free( &pszRealPath);
	}

	return( rc);

#endif
}

/****************************************************************************
Desc:		Generates a file name given a seed and some modifiers, it is built
			to be called in a loop until the file can be sucessfully written or
			created with the increment being changed every time.
In:		bModext		-> if TRUE then we will use the extension for collisions.
In\Out:	puiTime		-> a modified time stamp which is used as the base
								filename.  To properly set up this value, make sure
								the puiTime points to a 0 the first time this routine
								is called and it will be set up for you.  Thereafter,
                        do not change it between calls.
			pHighChars->	these are the 8 bits that were shifted off the top of
								the time struct.  It will be set up for you the first
								time you call this routine if puiTime points to a 0
								the first time this routine is called.  Do not change
								this value between calls.
			pszFileName -> should be pointing to a null string on the way in.
								going out it will be the complete filename.
			pszFileExt	-> the last char of the ext will be used for collisions,
								depending on the bModext flag.  If null then
								the extension will be .00x where x is the collision
								counter.
Notes:	The counter on the collision is 0-9, a-z.
****************************************************************************/
void FLMAPI F_FileSystem::pathCreateUniqueName(
	FLMUINT *		puiTime,
	char *			pszFileName,
	const char *	pszFileExt,
	FLMBYTE *		pHighChars,
	FLMBOOL			bModext)
{
	FLMINT		iCount, iLength;
	FLMUINT		uiSdTmp = 0;
	FLMUINT		uiIncVal = 1;

	f_setupTime( puiTime, pHighChars);
	uiSdTmp = *puiTime;

	*(pszFileName + 8) = NATIVE_DOT;
	f_memset( (pszFileName + 9), NATIVE_ZERO, 3 );
	
	if ( ( pszFileExt != NULL ))
	{
		if ((iLength = f_strlen(pszFileExt)) > 3)
		{
			iLength = 3;
		}
      f_memmove( (pszFileName + 9), pszFileExt, iLength);
   }

	if( bModext == TRUE)
	{
		f_hexToNative((FLMBYTE)(uiSdTmp & 0x0000001F), pszFileName+(11));
   }
	else
	{
	 	uiIncVal = 32;
	}
	
	uiSdTmp = uiSdTmp >> 5;
	for( iCount = 0; iCount < 6; iCount++)
	{
		f_hexToNative((FLMBYTE)(uiSdTmp & 0x0000000F), pszFileName+(7-iCount));
		uiSdTmp = uiSdTmp >> 4;
	}

	for( iCount = 0; iCount < 2; iCount++)
	{
		f_hexToNative((FLMBYTE)(*pHighChars & 0x0000000F), pszFileName+(1-iCount));
		*pHighChars = *pHighChars >> 4;
	}

   *(pszFileName + 12) = '\0';
	*puiTime += uiIncVal;
}

/****************************************************************************
Desc:		Compares the current file against a pattern template
****************************************************************************/
FLMBOOL FLMAPI F_FileSystem::doesFileMatch(
	const char *	pszFileName,
	const char *	pszTemplate)
{
	FLMUINT		uiPattern;
	FLMUINT		uiChar;

	if( !*pszTemplate)
	{
		return( TRUE);
	}

	while( *pszTemplate)
	{
		uiPattern = *pszTemplate++;
		switch( uiPattern)
		{
			case NATIVE_WILDCARD:
			{
				if( *pszTemplate == 0)
				{
					return( TRUE);
				}

				while( *pszFileName)
				{
					if( doesFileMatch( pszFileName, pszTemplate))
					{
						return( TRUE);
					}
					pszFileName++;
				}
				
				return( FALSE);
			}
			
			case NATIVE_QUESTIONMARK:
			{
				if( *pszFileName++ == 0)
				{
					return( FALSE);
				}
				break;
			}
			
			default:
			{
				uiChar = *pszFileName++;
				if( f_toupper( uiPattern) != f_toupper( uiChar))
				{
					return( FALSE);
				}
				break;
			}
		}
	}

	return( (*pszFileName != 0) ? FALSE : TRUE);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FLMAPI F_FileSystem::deleteMultiFileStream(
	const char *			pszDirectory,
	const char *			pszBaseName)
{
	RCODE						rc = NE_FLM_OK;
	F_MultiFileOStream *	pMultiStream = NULL;
	
	if( (pMultiStream = f_new F_MultiFileOStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pMultiStream->processDirectory( 
		pszDirectory, pszBaseName, TRUE)))
	{
		goto Exit;
	}
	
Exit:

	if( pMultiStream)
	{
		pMultiStream->Release();
	}

	return( rc);
}

/****************************************************************************
Desc: This routine obtains exclusive access to a database by creating
		a .lck file.  FLAIM holds the .lck file open as long as the database
		is open.  When the database is finally closed, it deletes the .lck
		file.  This is only used for 3.x databases.
****************************************************************************/
RCODE FLMAPI F_FileSystem::createLockFile(
	const char *		pszPath,
	IF_FileHdl **		ppLockFileHdl)
{
	RCODE					rc = NE_FLM_OK;
	IF_FileHdl *		pLockFileHdl = NULL;

	// Attempt to create the lock file.  If that succeeds, we have 
	// the lock.  If it fails, the lock file may have been left because
	// of a crash.  Hence, we first try to delete the file.  If that succeeds,
	// we then attempt to create the file again.  If that, or the 2nd create
	// fail, we simply return an access denied error.

#ifdef FLM_UNIX
	if( RC_BAD( createFile( pszPath,
			FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW, 
			&pLockFileHdl)))
	{
		if( RC_BAD( openFile( pszPath, 
			FLM_IO_RDWR | FLM_IO_SH_DENYRW, &pLockFileHdl)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
	}

	if( RC_BAD( pLockFileHdl->lock()))
	{
		rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
		goto Exit;
	}
#else
	if( RC_BAD( createFile( pszPath,
		FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW | FLM_IO_DELETE_ON_RELEASE,
		&pLockFileHdl)))
	{
		if( RC_BAD( deleteFile( pszPath)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
		else if( RC_BAD( createFile( pszPath, 
			FLM_IO_RDWR | FLM_IO_EXCL | FLM_IO_SH_DENYRW | FLM_IO_DELETE_ON_RELEASE,
			&pLockFileHdl)))
		{
			rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
			goto Exit;
		}
	}
#endif

	*ppLockFileHdl = pLockFileHdl;
	pLockFileHdl = NULL;

Exit:

	if (pLockFileHdl)
	{
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	return( rc);
}
