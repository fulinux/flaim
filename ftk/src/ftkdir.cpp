//------------------------------------------------------------------------------
//	Desc:	Class for doing file directory operations.
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
// $Id: ftkdir.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

#define ERR_NO_FILES_FOUND                      0xFF
#define ERR_INVALID_PATH                        0x9C

#if defined( FLM_WIN)

	FSTATIC FLMBOOL f_fileMeetsFindCriteria(
		F_IO_FIND_DATA *		pFindData);

#elif defined( FLM_UNIX) || defined( FLM_NLM)

	FSTATIC int Find1(
		char *				FindTemplate,
		F_IO_FIND_DATA *	DirInfo);


	FSTATIC int Find2(
		F_IO_FIND_DATA *	DirStuff);

	FSTATIC FLMBYTE ReturnAttributes(
		mode_t		FileMode,
		char *		pszFileName);

	FSTATIC int RetrieveFileStat(
		char *			FilePath,
		struct stat	*	StatusRec);

#else

	#error Platform not supported

#endif

/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_DirHdl::F_DirHdl()
{
	m_rc = NE_FLM_OK;
	m_bFirstTime = TRUE;
	m_bFindOpen = FALSE;
	m_uiAttrib = 0;
	m_szPattern[ 0] = '\0';
}

/****************************************************************************
Desc:
****************************************************************************/
const char * FLMAPI F_DirHdl::currentItemName( void)
{
	const char *	pszName = NULL;

	if( RC_OK( m_rc))
	{
		pszName = m_szFileName;
	}
	
	return( pszName);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void FLMAPI F_DirHdl::currentItemPath(
	char *		pszPath)
{
	if( RC_OK( m_rc))
	{
		f_strcpy( pszPath, m_szDirectoryPath);
		gv_pFileSystem->pathAppend( pszPath, m_szFileName);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FLMAPI F_DirHdl::currentItemIsDir( void)
{
	return( ((m_uiAttrib & F_IO_FA_DIRECTORY)
						 ? TRUE
						 : FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT64 FLMAPI F_DirHdl::currentItemSize( void)
{
	FLMUINT64	ui64Size = 0;

	if( RC_OK( m_rc))
	{
#if defined( FLM_WIN)
		ui64Size = (((FLMUINT64)m_FindData.findBuffer.nFileSizeHigh) << 32) + 
						m_FindData.findBuffer.nFileSizeLow;
#elif defined( FLM_UNIX) || defined ( FLM_NLM)
		ui64Size = m_FindData.FileStat.st_size;
#endif
	}
	return( ui64Size);
}

/****************************************************************************
Desc:	Get the next item in a directory
****************************************************************************/
RCODE FLMAPI F_DirHdl::next( void)
{
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

			if( RC_BAD( m_rc = f_fileFindFirst( m_szDirectoryPath, uiSearchAttributes,
				&m_FindData, szFoundPath, &uiFoundAttrib)))
			{
				goto Exit;
			}
			
			m_bFindOpen = TRUE;
			m_uiAttrib = uiFoundAttrib;
		}
		else
		{
			if( RC_BAD( m_rc = f_fileFindNext( &m_FindData, 
				szFoundPath, &uiFoundAttrib)))
			{
				goto Exit;
			}
			
			m_uiAttrib = uiFoundAttrib;
		}

		if( RC_BAD( m_rc = gv_pFileSystem->pathReduce( szFoundPath, 
			szDummyPath, m_szFileName)))
		{
			goto Exit;
		}

		if( gv_pFileSystem->doesFileMatch( m_szFileName, m_szPattern))
		{
			break;
		}
	}

Exit:

	return( m_rc);
}

/****************************************************************************
Desc:	Open a directory
****************************************************************************/
RCODE F_DirHdl::openDir(
	const char *	pszDirName,
	const char *	pszPattern)
{
	RCODE		rc = NE_FLM_OK;

	m_rc = NE_FLM_OK;
	m_bFirstTime = TRUE;
	m_bFindOpen = FALSE;
	m_uiAttrib = 0;

	f_strcpy( m_szDirectoryPath, pszDirName);

	if( pszPattern)
	{
		if( f_strlen( pszPattern) >= sizeof( m_szPattern))
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}

		f_strcpy( m_szPattern, pszPattern);
	}

Exit:

	return( rc);

}

/****************************************************************************
Desc:	Create a directory (and parent directories if necessary).
****************************************************************************/
RCODE F_DirHdl::createDir(
	const char *	pszDirPath)
{
	char *			pszParentDir = NULL;
	RCODE				rc = NE_FLM_OK;

	if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &pszParentDir)))
	{
		goto Exit;
	}

	// Discover the parent directory of the given one

	if( RC_BAD( rc = gv_pFileSystem->pathReduce( pszDirPath, 
		pszParentDir, NULL)))
	{
		goto Exit;
	}

	// If pathReduce couldn't reduce the path at all, then an
	// invalid path was supplied.

	if( f_strcmp( pszDirPath, pszParentDir) == 0)
	{
		rc = RC_SET( NE_FLM_IO_INVALID_FILENAME);
		goto Exit;
	}

	// If a parent directory was found, and it doesn't already exist, create it

	if( *pszParentDir)
	{
		// If the "parent" is actually a regular file we need to return an error

		if( RC_OK( gv_pFileSystem->doesFileExist( pszParentDir)))
		{
			if( !gv_pFileSystem->isDir( pszParentDir))
			{
				rc = RC_SET( NE_FLM_IO_ACCESS_DENIED);
				goto Exit;
			}
		}
		else if( RC_BAD( rc = createDir( pszParentDir)))
		{
			goto Exit;
		}
	}

#if defined( FLM_WIN)

	if( !CreateDirectory((LPTSTR)pszDirPath, NULL))
	{
		rc = MapPlatformError( GetLastError(), NE_FLM_CREATING_FILE);
	}

#elif defined( FLM_UNIX) || defined( FLM_NLM)

	if( mkdir( (char *)pszDirPath, 0777) == -1)
	{
		rc = MapPlatformError( errno, NE_FLM_CREATING_FILE);
	}

#endif

Exit:

	if( pszParentDir)
	{
		f_free( &pszParentDir);
	}
	
	return( rc);
}

/****************************************************************************
Desc:		Remove a directory
Notes:	The directory must be empty.
****************************************************************************/
RCODE F_DirHdl::removeDir(
	const char *	pszDirName)
{
#if defined( FLM_WIN)

	if( !RemoveDirectory((LPTSTR)pszDirName))
	{
		return( MapPlatformError( GetLastError(), NE_FLM_IO_DELETING_FILE));
	}

	return( NE_FLM_OK);

#elif defined( FLM_UNIX) || defined( FLM_NLM)

	 if( rmdir( (char *)pszDirName) == -1)
	 {
		 return( MapPlatformError( errno, NE_FLM_IO_DELETING_FILE));
	 }

    return( NE_FLM_OK);

#endif
}

/****************************************************************************
Desc:		Find the first file that matches the supplied criteria
****************************************************************************/
RCODE f_fileFindFirst(
	char *				pszSearchPath,
   FLMUINT				uiSearchAttrib,
	F_IO_FIND_DATA	*	pFindData,
   char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
#ifdef FLM_WIN
	char 			szTmpPath[ F_PATH_MAX_SIZE];
   char *		pszWildCard = "*.*";
	RCODE			rc = NE_FLM_OK;

	f_memset( pFindData, 0, sizeof( F_IO_FIND_DATA));
	pFindData->findHandle = INVALID_HANDLE_VALUE;
	pFindData->uiSearchAttrib = uiSearchAttrib;

	if( !pszSearchPath)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}

	f_strcpy( pFindData->szSearchPath, pszSearchPath);

	if( uiSearchAttrib & F_IO_FA_NORMAL )
	{
		uiSearchAttrib |= F_IO_FA_ARCHIVE;
	}

	f_strcpy( szTmpPath, pszSearchPath);
	
	if( RC_BAD( rc = gv_pFileSystem->pathAppend( szTmpPath, pszWildCard)))
	{
		goto Exit;
	}

   if( (pFindData->findHandle = FindFirstFile( (LPTSTR)szTmpPath,
	      &(pFindData->findBuffer))) == INVALID_HANDLE_VALUE)
   {
		rc = MapPlatformError( GetLastError(), NE_FLM_OPENING_FILE);
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
			rc = MapPlatformError( GetLastError(), NE_FLM_READING_FILE);
			goto Exit;
		}
	}

	// Append the file name to the path name

	f_strcpy( pszFoundPath, pFindData->szSearchPath);
	
	if( RC_BAD( rc = gv_pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->findBuffer.cFileName)))
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

#else

	char 					szTmpPath[ F_PATH_MAX_SIZE];
	FSTATIC char		pszWildCard[] = {'*',0};
	int					iRetVal;
	RCODE					rc = NE_FLM_OK;

	if( !pszSearchPath)
	{
		rc = RC_SET( NE_FLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}

	f_strcpy( szTmpPath, pszSearchPath);
	if( RC_BAD( rc = gv_pFileSystem->pathAppend( szTmpPath, pszWildCard)))
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
			rc = RC_SET( NE_FLM_IO_NO_MORE_FILES);
		}
		else
		{
			rc = MapPlatformError( errno, NE_FLM_READING_FILE);
		}
		
		goto Exit;
	}

	// filter out ".." (PARENT) and "." (CURRENT) directories
	
	if( uiSearchAttrib & F_IO_FA_DIRECTORY )
	{
		while( (f_strcmp( (FLMBYTE *)pFindData->name, (FLMBYTE *)"..") == 0) ||
			   (f_strcmp( (FLMBYTE *)pFindData->name, (FLMBYTE *)".") == 0))
		{
			if( (iRetVal = Find2( pFindData)) != 0)
			{
				// If there were no more files found then return no more files
				// instead of mapping to error path not found or io error.
				// To return no more files ret_val is ENOENT (set in Find2)
				// and errno is not set
				
				if( iRetVal == ENOENT && errno == 0)
				{
					rc = RC_SET( NE_FLM_IO_NO_MORE_FILES);
				}
				else
				{
					rc = MapPlatformError( errno, NE_FLM_READING_FILE);
				}
				
				goto Exit;
			}
		}
	}

	// Append the file name to the path name
	
	f_strcpy( pszFoundPath, pszSearchPath);
	
	if( RC_BAD( rc = gv_pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->name)))
	{
		goto Exit;
	}

	*puiFoundAttrib = (FLMUINT)ReturnAttributes(
			pFindData->FileStat.st_mode, pszFoundPath);

	// Save the search path in the NE_FLM_IO_FIND_DATA struct
	// for a find next call

	f_strcpy( pFindData->search_path, pszSearchPath);

Exit:

	return( rc);
#endif
}

/****************************************************************************
Desc:		Find the next file that matches the supplied criteria
****************************************************************************/
RCODE f_fileFindNext(
	F_IO_FIND_DATA *	pFindData,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib)
{
	RCODE			rc = NE_FLM_OK;

#ifdef FLM_WIN

   if( FindNextFile( pFindData->findHandle,
		&(pFindData->findBuffer)) == FALSE)
	{
		rc = MapPlatformError( GetLastError(), NE_FLM_READING_FILE);
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
			rc = MapPlatformError( GetLastError(), NE_FLM_READING_FILE);
			goto Exit;
		}
	}

	// Append the file name to the path name

	f_strcpy( pszFoundPath, pFindData->szSearchPath);
	
	if( RC_BAD( rc = gv_pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->findBuffer.cFileName)))
	{
		goto Exit;
	}

	// Return the found file attribute

   *puiFoundAttrib = pFindData->findBuffer.dwFileAttributes;

#elif defined( FLM_UNIX) || defined( FLM_NLM)
	int	iRetVal;

	if( (iRetVal =  Find2( pFindData)) != 0)
	{
		// If there were no more files found then return no more files
		// instead of mapping to error path not found or io error.
		// To return no more files ret_val is ENOENT (set in Find2)
		// and errno is not set
		
		if( iRetVal == ENOENT && errno == 0)
		{
			return( RC_SET( NE_FLM_IO_NO_MORE_FILES));
		}
		
		return( MapPlatformError( errno, NE_FLM_READING_FILE));
	}

	// Append the file name to the path name
	
	f_strcpy( pszFoundPath, pFindData->search_path);
	
	if( RC_BAD( rc = gv_pFileSystem->pathAppend( pszFoundPath, 
		(char *)pFindData->name)))
	{
		goto Exit;
	}

	*puiFoundAttrib = (FLMUINT)ReturnAttributes(
			pFindData->FileStat.st_mode, pszFoundPath);
#else
	rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
	goto Exit;
#endif

Exit:

   return( rc);
}

/****************************************************************************
Desc:		Releases any memory allocated to an F_IO_FIND_DATA structure
****************************************************************************/
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
#elif defined( FLM_UNIX) || defined ( FLM_NLM)
	if( pFindData->globbuf.gl_pathv)
	{
		pFindData->globbuf.gl_offs = 0;
		globfree( &pFindData->globbuf);
		pFindData->globbuf.gl_pathv = 0;
	}
#endif
}

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
#if defined( FLM_UNIX) || defined( FLM_NLM)

FSTATIC int Find1(
	char *				FindTemplate,
	F_IO_FIND_DATA *	DirInfo)
{
	char  	MaskNam[ F_PATH_MAX_SIZE];
	char  	*PathSeparator;
	FLMINT	uiFindLen;
	FLMINT	uiLen;
#ifdef FLM_NLM
	char  	szPosixNam[ F_PATH_MAX_SIZE];
	FLMINT	uiCount;				
#endif

	// If supplied template is illegal, return immediately

	if( (FindTemplate == (char*)NULL) || !( uiFindLen = f_strlen( FindTemplate)))
	{
		return( EINVAL);
	}

	// Now separate the template into a PATH and a template MASK
	// If no separating slash character found, use current directory
	// as path!

	f_strcpy( DirInfo->full_path, FindTemplate);
	
#ifdef FLM_NLM
	if( (( PathSeparator = strrchr( DirInfo->full_path, '/')) == NULL) &&
		( PathSeparator = strrchr( DirInfo->full_path, '\\')) == NULL)
#else
	if( (PathSeparator = strrchr( DirInfo->full_path, '/')) == NULL)
#endif
	{
		(void) getcwd( DirInfo->full_path, F_PATH_MAX_SIZE);
		uiLen = f_strlen( DirInfo->full_path );
		DirInfo->full_path[uiLen] = '/';
		DirInfo->full_path[uiLen+1] = '\0';
		(void) f_strcat( DirInfo->full_path, FindTemplate );
		PathSeparator = strrchr( DirInfo->full_path, '/');
	}

	// Copy the template MASK, and null terminate the PATH
	
	f_strcpy( MaskNam, PathSeparator + 1);
	
	if( ! f_strlen(MaskNam))
	{
		(void) f_strcpy( MaskNam, "*");
	}

	*PathSeparator = '\0';

	// Use ROOT directory if PATH is empty
	
	if( ! f_strlen(DirInfo->full_path))
	{
		(void) f_strcpy( DirInfo->full_path, "/");
	}

	f_strcpy( DirInfo->dirpath, DirInfo->full_path );

	// Open the specified directory.  Return immediately
	// if error detected!

	errno = 0;
	DirInfo->globbuf.gl_pathv = 0;

#ifdef FLM_NLM
	// glob does not seem to be able to handle a non-posix path
	// on NetWare.
	for( uiCount = 0; uiCount <= uiFindLen; uiCount++)
	{
		if( FindTemplate[ uiCount] == '\\')
		{
			szPosixNam[ uiCount] = '/';
		}
		else
		{
			szPosixNam[ uiCount] = FindTemplate[ uiCount];
		}
	}
	if( glob( szPosixNam, GLOB_NOSORT, 0, &DirInfo->globbuf) != 0 &&
		 !DirInfo->globbuf.gl_pathc)
#else
	if( glob( FindTemplate, GLOB_NOSORT, 0, &DirInfo->globbuf) != 0 &&
		 !DirInfo->globbuf.gl_pathc)
#endif
	{
		globfree(&DirInfo->globbuf);
		DirInfo->globbuf.gl_pathv = 0;
		return ENOENT;
	}
	
	// Call Find2 to get the 1st matching file

	return( Find2(DirInfo) );
}
#endif


/****************************************************************************
Desc:		Search for file names matching FindTemplate (UNIX)
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
FSTATIC int Find2(
	F_IO_FIND_DATA *		DirStuff)
{
	int			stat;
	glob_t *		pglob = &DirStuff->globbuf;
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
			return ENOENT;
		}
		
		// Get status of file

		f_strcpy(DirStuff->full_path, pglob->gl_pathv[pglob->gl_offs++]);
		if( (stat = RetrieveFileStat( DirStuff->full_path,
											 &DirStuff->FileStat)) != 0 )
		{
			// If file name just read from directory is NO
			// longer there (deleted by another process)
			// then just advance to the next file in
			// directory!

			if( stat == ENOENT)
			{
				continue;
			}
			else
			{
				break;
			}
		}

		// If we don't want directories, and current entry
		// is a directory, then skip it!

		if( (! S_ISDIR(DirStuff->mode_flag)) &&
			  S_ISDIR(DirStuff->FileStat.st_mode))
		{
			continue;
		}

		// If we only want regular files and file is NOT
		// regular, then skip it!  This means there is no
		// way to retrieve named pipes, sockets, or links!

		if ( (DirStuff->mode_flag == F_IO_FA_NORMAL) &&
			  (! S_ISREG(DirStuff->FileStat.st_mode)) )
		{
			continue;
		}

		pszTmp = &DirStuff->full_path[ 0];
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
			f_strcpy( DirStuff->name, &pszLastSlash[ 1]);
		}
		else
		{
			f_strcpy( DirStuff->name, DirStuff->full_path);
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
#if defined( FLM_UNIX) || defined( FLM_NLM)
FSTATIC FLMBYTE ReturnAttributes(
	mode_t	FileMode,
	char *	fileName)
{
	FLMBYTE  IOmode = 0;

	// Return the found file attribute
	
	if( S_ISDIR( FileMode ) )
	{
		IOmode |= F_IO_FA_DIRECTORY;
	}
	else
	{
		if( access( (char *)fileName, (int)(R_OK | W_OK)) == 0)
		{
			IOmode |= F_IO_FA_NORMAL;
		}
		else if( access( (char *)fileName, (int)R_OK ) == 0)
		{
			IOmode |= F_IO_FA_RDONLY;
		}
	}

	return( IOmode);
}
#endif

/****************************************************************************
Desc: Return file's attributes (UNIX) || (NetWare)
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
FSTATIC int RetrieveFileStat(
	char *			FilePath,
	struct stat	*	StatusRec)
{
	// Get status of last file read from directory, using the standard
	// UNIX stat call

	errno = 0;
	if( stat( FilePath, StatusRec ) == -1)
	{
		if( errno == ENOENT || errno == ELOOP)
		{
			// Get status of symbolic link rather than referenced file!

			errno = 0;
			if( lstat( FilePath, StatusRec ) == -1)
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
