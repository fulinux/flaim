//------------------------------------------------------------------------------
// Desc:	Contains functions for file name/path manipulation
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
// $Id: ftkpath.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

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
	const char *	pszInputPath,
	char *			pszServer,
	char *			pszVolume,
	char *			pszDirPath,
	char *			pszFileName)
{
	char			szInput[ F_PATH_MAX_SIZE];
	char *		pszNext;
	char *		pszColon;
	char *		pszComponent;
	FLMUINT		uiEndChar;
	FLMBOOL		bUNC = FALSE;
	
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
		gv_pFileSystem->pathReduce( pszInputPath, szInput, pszFileName);
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
	return NE_FLM_OK;
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
		rc = MapErrnoToFlaimErr( errno, NE_FLM_PARSING_FILE_NAME);
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
Desc:
****************************************************************************/
FINLINE void HexToNative(
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
FINLINE void SetUpTime(
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

	SetUpTime( puiTime, pHighChars);
	uiSdTmp = *puiTime;

	/* Add on the filename extension if passed from the caller */
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
		HexToNative((FLMBYTE)(uiSdTmp & 0x0000001F), pszFileName+(11));
   }
	else
	{
	 	uiIncVal = 32;
	}
	uiSdTmp = uiSdTmp >> 5;
	for( iCount = 0; iCount < 6; iCount++)		/* set pos 2-7 of filename	*/
	{
		HexToNative((FLMBYTE)(uiSdTmp & 0x0000000F), pszFileName+(7-iCount));
		uiSdTmp = uiSdTmp >> 4;
	} /* End for() */

	for( iCount = 0; iCount < 2; iCount++)		/* set pos 0-1 of filename	*/
	{
		HexToNative((FLMBYTE)(*pHighChars & 0x0000000F), pszFileName+(1-iCount));
		*pHighChars = *pHighChars >> 4;
	} /* End for() */

   /* Append on a NULL terminator */
   *(pszFileName + 12) = '\0';
	*puiTime += uiIncVal;

   return;
}

/****************************************************************************
Desc:		Compares the current file against a pattern template
****************************************************************************/
FLMBOOL FLMAPI f_doesFileMatch(
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
				/* if the match_template ends in an asterisk, then we match the*/
				/* remaining string by default, return a match.						*/

				if( *pszTemplate == 0)
				{
					return( TRUE);
				}

				/* Found an asterisk somewhere in the match_template, now let's
					see if we match anywhere on the remaining input string.		*/

				while( *pszFileName)
				{
					if( f_doesFileMatch( pszFileName, pszTemplate))
					{
						return( TRUE);				/* found a match, return 		*/
					}
					pszFileName++;
				}
				return( FALSE);						/* did not find match, return	*/
			case NATIVE_QUESTIONMARK:
				if( *pszFileName++ == 0)			/* skip one character for '?'	*/
				{
					return( FALSE);
				}
				break;
			default:
				uiChar = *pszFileName++;
				if( f_toupper( uiPattern) != f_toupper( uiChar))
				{
					return( FALSE);
				}
				break;
		}
	}

	return( (*pszFileName != 0) ? FALSE : TRUE );
}
