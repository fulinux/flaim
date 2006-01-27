//-------------------------------------------------------------------------
// Desc:	Class for accessing file system directory information.
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
// $Id: ftkdir.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define ERR_NO_FILES_FOUND                      0xFF
#define ERR_INVALID_PATH                        0x9C

#if defined( FLM_WIN)

	FSTATIC FLMBOOL f_fileMeetsFindCriteria(
		F_IO_FIND_DATA *	pFindData);

#elif defined( FLM_UNIX)

	FSTATIC int Find1(
		char *				FindTemplate,
		F_IO_FIND_DATA *	DirInfo);


	FSTATIC int Find2(
		F_IO_FIND_DATA *	pFindData);

	FSTATIC FLMBYTE flmReturnFileAttributes(
		mode_t				uiFileMode,
		const char *		pszFileName);

	FSTATIC int flmRetrieveFileStat(
		const char *		pszFilePath,
		struct stat	*		pStatusRec);

#elif !defined( FLM_NLM)

	#error Platform not supported

#endif

/****************************************************************************
Desc:		Get the next item in a directory
****************************************************************************/
RCODE F_DirHdlImp::Next()
{
#if defined( FLM_WIN) || defined( FLM_UNIX)
	char				szFoundPath[ F_PATH_MAX_SIZE];
	char				szDummyPath[ F_PATH_MAX_SIZE];
	FLMUINT			uiSearchAttributes;
	FLMUINT			uiFoundAttrib;

	if( RC_BAD( m_rc))
	{
		goto Exit;
	}

	uiSearchAttributes =
		F_IO_FA_NORMAL | F_IO_FA_RDONLY | F_IO_FA_ARCHIVE | F_IO_FA_DIRECTORY;

	for( ;;)
	{
		if ( m_bFirstTime )
		{
			m_bFirstTime = FALSE;
			
			if( RC_BAD( m_rc = f_fileFindFirst( m_DirectoryPath, uiSearchAttributes, 
				&m_FindData, szFoundPath, &uiFoundAttrib)))
			{
				goto Exit;
			}
			
			m_bFindOpen = TRUE;
			m_uiAttrib = uiFoundAttrib;
		}
		else
		{
			if( RC_BAD( m_rc = f_fileFindNext( &m_FindData, szFoundPath,
				&uiFoundAttrib)))
			{
				goto Exit;
			}
			
			m_uiAttrib = uiFoundAttrib;
		}

		if( RC_BAD( m_rc = f_pathReduce( szFoundPath, szDummyPath, m_szFileName)))
		{
			goto Exit;
		}

		if( f_doesFileMatch( m_szFileName, m_ucPattern))
		{
			break;
		}
	}

#elif defined( FLM_NLM)

	LONG				lError = 0;
	
	if( RC_BAD( m_rc))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( (lError = DirectorySearch( 0, m_lVolumeNumber, m_lDirectoryNumber,
			LONGNameSpace, m_lCurrentEntryNumber, (BYTE *)"\x02\xFF*",
			-1, &m_pCurrentItem, &m_lCurrentEntryNumber)) != 0)
		{
			if( (lError == ERR_NO_FILES_FOUND) || (lError == ERR_INVALID_PATH))
			{
				m_rc = RC_SET( FERR_IO_NO_MORE_FILES);
			}
			else
			{
				m_rc = MapNWtoFlaimError(lError, FERR_READING_FILE);
			}
			
			break;
		}

		if( f_doesFileMatch( (const char *)m_pCurrentItem->DFileName, m_ucPattern))
		{
			break;
		}
	}
#endif

Exit:

	return( m_rc);
}
								
/****************************************************************************
Desc:		Open a directory
****************************************************************************/
RCODE F_DirHdlImp::OpenDir(
	const char *		pDirPath,
	const char *		pszPattern)
{
#if defined( FLM_WIN) || defined( FLM_UNIX)
	RCODE		rc = FERR_OK;

	m_rc = FERR_OK;
	m_bFirstTime = TRUE;
	m_bFindOpen = FALSE;
	m_uiAttrib = 0;
	
	f_strcpy( m_DirectoryPath, pDirPath);

	if( pszPattern)
	{
		if( f_strlen( pszPattern) >= sizeof( m_ucPattern))
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		f_strcpy( m_ucPattern, pszPattern);
	}

Exit:

	return( rc);

#elif defined( FLM_NLM)

	// Notes:
	// 	1. DOS file names, not long file names !  If we want to support long
	// 		file names, then increase the size of the filename buffer and change 
	// 		the name space.
	// 	2. '*.*' doesn't work as a pattern.  '*' seems to do the trick.
	// 	3. These Netware APIs are case sensitive.  If you want to specify a 
	// 		a pattern like "*.db"  make sure that the files you are looking for
	// 		were created with lowercase "db" extensions.
	// 		
	// 		Also, the path needs to match the case also.  For example, 
	// 		sys:\_netware won't work.  SYS:\_NETWARE will.
	// 	4. Server names are not supported by ConvertPathString
	// 		'Connecting to remote servers' is not supported by this code.

	LONG			unused;	
	FLMBYTE		pseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE		LNamePath[ F_PATH_MAX_SIZE];
	LONG			lLNamePathCount;
	LONG			lError = 0;

	m_rc = FERR_OK;
	m_lVolumeNumber = F_NW_DEFAULT_VOLUME_NUMBER;
	m_lCurrentEntryNumber = 0xFFFFFFFFL;
	f_memcpy( m_DirectoryPath, pDirPath, F_PATH_MAX_SIZE);

	LNamePath[0] = 0;
	lLNamePathCount = 0;

	f_strcpy( (char *)&pseudoLNamePath[1], pDirPath);
	pseudoLNamePath[0] = (FLMBYTE)f_strlen( (const char *)&pseudoLNamePath[1] );
	
	if( (lError = ConvertPathString( 0, 0, pseudoLNamePath, &m_lVolumeNumber,		
		&unused, (BYTE *)LNamePath, &lLNamePathCount)) != 0)
	{
		goto Exit;
	}
											
	// Remember the directory number (think of it as a NetWare directory ID)
	
	if( (lError = MapPathToDirectoryNumber( 0, m_lVolumeNumber, 0, 
		(BYTE *)LNamePath, lLNamePathCount, LONGNameSpace, &m_lDirectoryNumber, 
		&unused)) != 0)
	{
		goto Exit;
	}
		
	// Save the pattern for later
	
	if( pszPattern)
	{
		f_strncpy( m_ucPattern, pszPattern, sizeof( m_ucPattern) - 1);
		m_ucPattern[ sizeof( m_ucPattern) - 1] = 0;
	}
	else
	{
		m_ucPattern[ 0] = '\0';
	}
	
Exit:
	
	if( lError != 0)
	{
		m_rc = MapNWtoFlaimError(lError, FERR_OPENING_FILE);
	}

	return( m_rc);
#endif
}

/****************************************************************************
Desc:		Find the first file that matches the supplied criteria
****************************************************************************/
#if defined( FLM_WIN) || defined( FLM_UNIX)
RCODE f_fileFindFirst(
	const char *			pszSearchPath,
   FLMUINT					uiSearchAttrib,
	F_IO_FIND_DATA	*		pFindData,
   char *					pszFoundPath,
	FLMUINT *				puiFoundAttrib)
{
#ifdef FLM_WIN
	RCODE				rc = FERR_OK;
	char 				szTmpPath[ F_PATH_MAX_SIZE];
   const char *	pszWildCard = "*.*";

	f_memset( pFindData, 0, sizeof( F_IO_FIND_DATA));
	pFindData->findHandle = INVALID_HANDLE_VALUE;
	pFindData->uiSearchAttrib = uiSearchAttrib;

	if( !pszSearchPath)
	{
		rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
	f_strcpy( pFindData->szSearchPath, pszSearchPath);

	// As per MS-DOS specification.

	if( uiSearchAttrib & F_IO_FA_NORMAL )
	{
		uiSearchAttrib |= F_IO_FA_ARCHIVE;
	}	

	f_strcpy( szTmpPath, pszSearchPath);
	
	if( RC_BAD( rc = f_pathAppend( szTmpPath, pszWildCard)))
	{
		goto Exit;
	}

   if( (pFindData->findHandle = FindFirstFile( (LPTSTR)szTmpPath,
	      &(pFindData->findBuffer))) == INVALID_HANDLE_VALUE)
   {
		rc = MapWinErrorToFlaim( GetLastError(), FERR_OPENING_FILE);
		goto Exit;
	}

	// Loop until a file with correct attributes is found
	
	for( ;;)
	{
		if( f_fileMeetsFindCriteria( pFindData))
		{
			break;
		}

		if( FindNextFile( pFindData->findHandle, 
			&(pFindData->findBuffer)) == FALSE)
		{
			rc = MapWinErrorToFlaim( GetLastError(), FERR_READING_FILE);
			goto Exit;
		}
	}

	// Append the file name to the path name
   
	f_strcpy( pszFoundPath, pFindData->szSearchPath);
	if( RC_BAD( rc = f_pathAppend( pszFoundPath, 
		pFindData->findBuffer.cFileName)))
	{
		goto Exit;
	}
	
	// Return the found file attribute

   *puiFoundAttrib = pFindData->findBuffer.dwFileAttributes;

Exit:

	if( RC_BAD( rc) && pFindData && 
		pFindData->findHandle != INVALID_HANDLE_VALUE)
	{
		f_fileFindClose( pFindData);
	}

	return( rc);

#elif defined( FLM_UNIX)
	RCODE					rc = FERR_OK;
	char	 				szTmpPath[ F_PATH_MAX_SIZE];
	FSTATIC char		pszWildCard[] = {'*',0};
	int					iRetVal;

	if( !pszSearchPath)
	{
		rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	
	f_strcpy( szTmpPath, pszSearchPath);
	
	if( RC_BAD( rc = f_pathAppend( szTmpPath, pszWildCard)))
	{
		goto Exit;
	}

	f_memset( pFindData, 0, sizeof( F_IO_FIND_DATA));
	
	if( uiSearchAttrib & F_IO_FA_DIRECTORY)
	{
		pFindData->mode_flag |= S_IFDIR;
	}

	if( uiSearchAttrib & F_IO_FA_RDONLY)
	{
		pFindData->mode_flag |= S_IREAD;
	}

	iRetVal = Find1( (char*)szTmpPath, pFindData);

	if( iRetVal != 0)
	{
		// If there were no more files found then return no more files
		// instead of mapping to error path not found or io error.
		// To return no more files ret_val is ENOENT (set in Find2)
		// and errno is not set

		if( iRetVal == ENOENT && errno == 0)
		{
			rc = RC_SET( FERR_IO_NO_MORE_FILES);
		}
		else
		{
			rc = MapErrnoToFlaimErr( errno, FERR_READING_FILE);
		}
		
		goto Exit;
	}

	// filter out ".." (PARENT) and "." (CURRENT) directories
	
	if( uiSearchAttrib & F_IO_FA_DIRECTORY )
	{
		while( (f_strcmp( pFindData->name, "..") == 0) ||
			   (f_strcmp( pFindData->name, ".") == 0))
		{
			if( (iRetVal = Find2( pFindData)) != 0)
			{
				// If there were no more files found then return no more files
				// instead of mapping to error path not found or io error.
				// To return no more files ret_val is ENOENT (set in Find2)
				// and errno is not set
				
				if( iRetVal == ENOENT && errno == 0)
				{
					rc = RC_SET( FERR_IO_NO_MORE_FILES);
				}
				else
				{
					rc = MapErrnoToFlaimErr( errno, FERR_READING_FILE);
				}
				
				goto Exit;
			}
		}
	}

	// Append the file name to the path name
	
	f_strcpy( pszFoundPath, pszSearchPath);
	
	if( RC_BAD( rc = f_pathAppend( pszFoundPath, pFindData->name)))
	{
		goto Exit;
	}

	*puiFoundAttrib = flmReturnFileAttributes(
			pFindData->FileStat.st_mode, pszFoundPath);

	// Save the search path in the FERR_IO_FIND_DATA struct for a find next call
	
	f_strcpy( pFindData->search_path, pszSearchPath);

Exit:

	return( rc);

#endif
}
#endif

/****************************************************************************
Desc:		Find the next file that matches the supplied criteria
****************************************************************************/
#if defined( FLM_WIN) || defined( FLM_UNIX)
RCODE f_fileFindNext(
	F_IO_FIND_DATA *	pFindData,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
	RCODE			rc = FERR_OK;

#ifdef FLM_WIN

   if( FindNextFile( pFindData->findHandle, 
		&(pFindData->findBuffer)) == FALSE)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_READING_FILE);
		goto Exit;
	}

	// Loop until a file with correct attributes is found

	for( ;;)
	{
		if( f_fileMeetsFindCriteria( pFindData))
		{
			break;
		}

		if( FindNextFile( pFindData->findHandle, 
			&(pFindData->findBuffer)) == FALSE)
		{
			rc = MapWinErrorToFlaim( GetLastError(), FERR_READING_FILE);
			goto Exit;
		}
	}

	// Append the file name to the path name
   
	f_strcpy( pszFoundPath, pFindData->szSearchPath);
	if( RC_BAD( rc = f_pathAppend( pszFoundPath,
		pFindData->findBuffer.cFileName)))
	{
		goto Exit;
	}
	
	// Return the found file attribute

   *puiFoundAttrib = pFindData->findBuffer.dwFileAttributes;

#elif defined( FLM_UNIX)
	int	iRetVal;

	if( (iRetVal =  Find2( pFindData)) != 0)
	{
		// If there were no more files found then return no more files
		// instead of mapping to error path not found or io error.
		// To return no more files ret_val is ENOENT (set in Find2)
		// and errno is not set
		
		if( iRetVal == ENOENT && errno == 0)
		{
			return( RC_SET( FERR_IO_NO_MORE_FILES));
		}
		
		return( MapErrnoToFlaimErr( errno, FERR_READING_FILE));
	}

	// Append the file name to the path name
	
	f_strcpy( pszFoundPath, pFindData->search_path);
	
	if( RC_BAD( rc = f_pathAppend( pszFoundPath, pFindData->name)))
	{
		goto Exit;
	}

	*puiFoundAttrib = flmReturnFileAttributes(
			pFindData->FileStat.st_mode, pszFoundPath);
#else
	rc = RC_SET( FERR_NOT_IMPLEMENTED);
	goto Exit;
#endif

Exit:

   return( rc);
}
#endif

/****************************************************************************
Desc:		Releases any memory allocated to an F_IO_FIND_DATA structure
****************************************************************************/
#if defined( FLM_WIN) || defined( FLM_UNIX)
void f_fileFindClose(
	F_IO_FIND_DATA *	pFindData)
{
#ifdef FLM_WIN

	// Don't call it on an already closed or invalid handle.
	
	if( pFindData->findHandle != INVALID_HANDLE_VALUE)
	{
		FindClose( pFindData->findHandle );
		pFindData->findHandle = INVALID_HANDLE_VALUE;
	}
#elif defined( FLM_UNIX)
	if( pFindData->globbuf.gl_pathv)
	{
		pFindData->globbuf.gl_offs = 0;
		globfree( &pFindData->globbuf);
		pFindData->globbuf.gl_pathv = 0;
	}
#endif
}
#endif

/****************************************************************************
Desc:		Find the next file that matches the supplied criteria
****************************************************************************/
#ifdef FLM_WIN
FSTATIC FLMBOOL f_fileMeetsFindCriteria(
	F_IO_FIND_DATA *		pFindData)
{
	// Fail ".." (PARENT) and "." (CURRENT) directories.  Then,
	// if the file found possesses any of the search attributes, it's
	// a match.

	if( !((f_strcmp( pFindData->findBuffer.cFileName, "..") == 0) ||
    (f_strcmp( pFindData->findBuffer.cFileName, ".") == 0) ||
	 (!(pFindData->uiSearchAttrib & F_IO_FA_DIRECTORY) &&
		(pFindData->findBuffer.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))))
	{
		if( (pFindData->findBuffer.dwFileAttributes & 
				pFindData->uiSearchAttrib) || 
			((pFindData->uiSearchAttrib & F_IO_FA_NORMAL) &&
				(pFindData->findBuffer.dwFileAttributes == 0)))
		{
			return( TRUE);
		}
	}

	return( FALSE);
}
#endif

/****************************************************************************
Desc:		Search for file names matching FindTemplate (UNIX)
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC int Find1(
	char *				pszFindTemplate,
	F_IO_FIND_DATA *	DirInfo)
{
	char			szMask[ F_PATH_MAX_SIZE];
	char *		pszPathSeparator;
	FLMUINT		uiLen;
	
	// If supplied template is illegal, return immediately
	
	if( (pszFindTemplate == NULL) || ! f_strlen( pszFindTemplate))
	{
		return( EINVAL );
	}

	// Now separate the template into a PATH and a template MASK
	// If no separating slash character found, use current directory
	// as path!
	
	(void) f_strcpy( DirInfo->full_path, pszFindTemplate );
	
	if( (pszPathSeparator = strrchr( DirInfo->full_path, '/')) == NULL)
	{
		getcwd( DirInfo->full_path, F_PATH_MAX_SIZE);
		uiLen = f_strlen( DirInfo->full_path );
		DirInfo->full_path[ uiLen] = '/';
		DirInfo->full_path[ uiLen+1] = '\0';
		f_strcat( DirInfo->full_path, pszFindTemplate );
		pszPathSeparator = strrchr( DirInfo->full_path, '/');
	}

	// Copy the template MASK, and null terminate the PATH
	
	f_strcpy( szMask, pszPathSeparator + 1);
	
	if( !f_strlen( szMask))
	{
		f_strcpy( szMask, "*");
	}

	*pszPathSeparator = '\0';

	// Use ROOT directory if PATH is empty
	
	if( !f_strlen( DirInfo->full_path))
	{
		f_strcpy( DirInfo->full_path, "/");
	}

	f_strcpy( DirInfo->dirpath, DirInfo->full_path);

	// Open the specified directory.  Return immediately if error detected!
	
	errno = 0;
	DirInfo->globbuf.gl_pathv = 0;

	if( glob( pszFindTemplate, GLOB_NOSORT, 0, &DirInfo->globbuf) != 0 &&
		 !DirInfo->globbuf.gl_pathc)
	{
		globfree(&DirInfo->globbuf);
		DirInfo->globbuf.gl_pathv = 0;
		return ENOENT;
	}
	
	// Call Find2 to get the 1st matching file
	
	return( Find2( DirInfo));

}
#endif

/****************************************************************************
Desc:		Search for file names matching pszFindTemplate (UNIX)
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC int Find2(
	F_IO_FIND_DATA *		pFindData)
{
	int			stat;
	glob_t *		pglob = &pFindData->globbuf;
	char *		pszTmp;
	char *		pszLastSlash;

	errno = 0;

	for( ;;)
	{
		if( pglob->gl_offs == pglob->gl_pathc)
		{
			pglob->gl_offs = 0;
			globfree(pglob);
			pglob->gl_pathv = 0;
			return( ENOENT);
		}
		
		// Get status of file
		
		f_strcpy( pFindData->full_path, pglob->gl_pathv[pglob->gl_offs++]);
		
		if( (stat = flmRetrieveFileStat( 
				pFindData->full_path, &pFindData->FileStat)) != 0)  
		{
			// If file name just read from directory is NO longer there 
			// (deleted by another process) then just advance to the next file in
			// the directory
			
			if( stat == ENOENT)
			{
				continue;
			}
			else
			{
				break;
			}
		}

		// If we don't want directories, and current entry is a directory,
		// then skip it
		
		if( (!S_ISDIR( pFindData->mode_flag)) && 
			 S_ISDIR( pFindData->FileStat.st_mode))
		{
			continue;
		}
	
		// If we only want regular files and file is NOT regular, then skip it.
		// This means there is no way to retrieve named pipes, sockets, or links.
		
		if( (pFindData->mode_flag == F_IO_FA_NORMAL) &&
			 (!S_ISREG( pFindData->FileStat.st_mode)))
		{
			continue;
		}

		pszTmp = &pFindData->full_path[ 0];
		pszLastSlash = NULL;
		
		while( *pszTmp)
		{
			if( *pszTmp == '/')
			{
				pszLastSlash = pszTmp;
			}
			
			pszTmp++;
		}

		if( pszLastSlash)
		{
			f_strcpy( pFindData->name, &pszLastSlash[ 1]);
		}
		else
		{
			f_strcpy( pFindData->name, pFindData->full_path);
		}
		
		stat = 0;
		break;
	}
	
	return( stat);
}
#endif

/****************************************************************************
Desc: Return file's attributes (UNIX)
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC FLMBYTE flmReturnFileAttributes(
	mode_t			uiFileMode,
	const char *	pszFileName)
{
	FLMBYTE  	ucIOmode = 0;

	// Return the found file attribute
	
	if( S_ISDIR( uiFileMode ))
	{
		ucIOmode |= F_IO_FA_DIRECTORY;
	}
	else
	{
		if( access( (char *)pszFileName, (int)(R_OK | W_OK)) == 0)
		{
			ucIOmode |= F_IO_FA_NORMAL;
		}
		else if( access( (char *)pszFileName, (int)R_OK) == 0)
		{
			ucIOmode |= F_IO_FA_RDONLY;
		}
	}

	return( ucIOmode);
}
#endif

/****************************************************************************
Desc: Return file's attributes (UNIX)
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC int flmRetrieveFileStat(
	const char *	pszFilePath,
	struct stat	*	pStatusRec)
{
	// Get status of last file read from directory, using the standard
	// UNIX stat call
	
	errno = 0;
	
	if( stat( pszFilePath, pStatusRec) == -1)
	{
		if( errno == ENOENT || errno == ELOOP)
		{
			// Get status of symbolic link rather than referenced file
			
			errno = 0;
			
			if( lstat( pszFilePath, pStatusRec ) == -1)
			{
				return( errno);
			}
		}
		else
		{
			return( errno);
		}
	}

	return( 0);    
}
#endif

/****************************************************************************
Desc:		NetWare implementation to create a directory
Input:	pPathToDirectory = fully-qualified path to the directory to be created
			Examples:
				In this example, the directory 'myDir' would be created:
				sys:\system\myDir
		
				Note that ConvertPathString doesn't support server names.  So
				it returns an error on paths like:
					server-name/volume:\directory_1
****************************************************************************/
#ifdef FLM_NLM
RCODE F_DirHdlImp::_CreateDir(
	const char *	pszPathToDirectory)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			pucPseudoLNamePath[ F_PATH_MAX_SIZE + 1];
	FLMBYTE			pucLNamePath[ F_PATH_MAX_SIZE];
	LONG				lVolumeID;
	LONG				lPathID;
	LONG				lLNamePathCount;
	LONG				lNewDirectoryID;
	void				*pNotUsed;
	LONG				lErrorCode;
	
	f_strcpy( (char *)&pucPseudoLNamePath[1], pszPathToDirectory);
	pucPseudoLNamePath[0] = (FLMBYTE)f_strlen( pszPathToDirectory);
	
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
Desc:		Return the current item's name
****************************************************************************/
const char * F_DirHdlImp::CurrentItemName( void)
{
#if defined( FLM_WIN) || defined( FLM_UNIX)
	const char *	pszName = NULL;
	
	if( RC_OK( m_rc))
	{
		pszName = m_szFileName;
	}
	
	return( pszName);

#elif defined( FLM_NLM)

	FLMUINT		uiLength;
	char *		pszName = NULL;

	if( RC_BAD( m_rc))
	{
		goto Exit;
	}

	if( !m_pCurrentItem)
	{
		goto Exit;
	}
	
	uiLength = sizeof( m_ucTempBuffer) - 1;
	
	if( m_pCurrentItem->DFileNameLength < uiLength)
	{
		uiLength = m_pCurrentItem->DFileNameLength;
	}
	
	f_strncpy( (char *)m_ucTempBuffer, (const char *)m_pCurrentItem->DFileName, uiLength);
	pszName = m_ucTempBuffer;
	pszName[ uiLength] = 0;

Exit:

	return( pszName);
#endif
}

/****************************************************************************
Desc:		Return the current item's size
****************************************************************************/
FLMUINT F_DirHdlImp::CurrentItemSize()
{
	FLMUINT	uiSize = 0;

	if( RC_OK( m_rc))
	{
#if defined( FLM_WIN)
		uiSize = (FLMUINT)m_FindData.findBuffer.nFileSizeLow;
#elif defined( FLM_UNIX)
		uiSize = (FLMUINT)m_FindData.FileStat.st_size;
#elif defined( FLM_NLM)
		if( m_pCurrentItem != NULL )
		{
			uiSize = (FLMUINT)m_pCurrentItem->DFileSize;
		}
#endif
	}
	
	return( uiSize);
}

/****************************************************************************
Desc:		Returns whether or not current item is a directory.
****************************************************************************/
FLMBOOL F_DirHdlImp::CurrentItemIsDir()
{
#if defined( FLM_WIN) || defined( FLM_UNIX)
	return( ((m_uiAttrib & F_IO_FA_DIRECTORY)
						 ? TRUE
						 : FALSE));
#elif defined( FLM_NLM)
	if( !m_pCurrentItem)
	{
		return( FALSE);
	}

	return( (m_pCurrentItem->DFileAttributes & SUBDIRECTORY_BIT)
		? TRUE 
		: FALSE);
#endif
}
