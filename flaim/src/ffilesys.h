//-------------------------------------------------------------------------
// Desc:	File system class - definitions.
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
// $Id: ffilesys.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FFILESYS_H
#define FFILESYS_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

/****************************************************************************
Desc:		The F_FileSystem class provides a file system abstraction for
			interacting with the underlying O/S file system.
****************************************************************************/
class F_FileSystemImp : public F_FileSystem
{
public:

	F_FileSystemImp()
	{
	}

	virtual ~F_FileSystemImp()
	{
	}

	RCODE Create(									// Create a new file handle
		const char *		pszFilePath,		// Name of file to be created
		FLMUINT				uiIoFlags,			// Access amd Mode flags
		F_FileHdl **		ppFileHdl);			// Returns open file handle object.

	RCODE CreateBlockFile(						// Create a new block-oriented file handle
		const char *		pszFilePath,		// Name of file to be created
		FLMUINT				uiIoFlags,			// Access amd Mode flags
		FLMUINT				uiBlockSize,		// Block size
		F_FileHdlImp **	ppFileHdl);			// Returns open file handle object.

	RCODE CreateUnique(							// Create a new file (with a unique
														// file name).
		char *				pszDirPath,			// Directory where file is to be created.
		const char *		pszFileExtension,	// Extension to be used on file.
		FLMUINT				uiIoFlags,			// Access and Mode flags.
		F_FileHdl **		ppFileHdl);			// Returns open file handle object.

	RCODE Open(										// Open an existing file.
		const char *		pszFilePath,		// Name of file to be opened.
		FLMUINT				uiIoFlags,			// Access and Mode flags.
		F_FileHdl **		ppFileHdl);			// Returns open file handle object.

	RCODE OpenBlockFile(							// Open an existing block-oriented file.
		const char *		pszFilePath,		// Name of file to be opened.
		FLMUINT				uiIoFlags,			// Access and Mode flags.
		FLMUINT				uiBlockSize,		// Block size.
		F_FileHdlImp **	ppFileHdl);			// Returns open file handle object.

	RCODE OpenDir(									// Open a directory
		const char *		pszDirPath,			// Directory to be opened.
		const char *		pszPattern,			// File name pattern.
		F_DirHdl **			ppDirHdl);			// Returns open directory handle
														// object.

	RCODE CreateDir(								// Create a directory
		const char *		pszDirPath);		// Directory to be created.

	RCODE RemoveDir(								// Remove a directory
		const char *		pszDirPath,			// Directory to be removed.
		FLMBOOL				bClear = FALSE);	// OK to delete files if dir is not empty?

	RCODE Exists(									// See if a file or directory exists.
		const char *		pszPath);			// Name of file or directory to check.

	FLMBOOL IsDir(									// See if a path is a directory.
		const char *		pszPath);			// Name of path to check.

	RCODE GetTimeStamp(							// Get the date/time when the file
														// was last updated.
		const char *		pszPath,				// Path to file
		FLMUINT *			puiTimeStamp);		// Buffer in which time stamp is 
														// returned.

	RCODE Delete(									// Delete a file or directory
		const char *		pszPath);			// Name of file or directory to delete.

	RCODE Copy(										// Copy a file.
		const char *		pszSrcFilePath,	// Name of source file to be copied.
		const char *		pszDestFilePath,	// Name of destination file.
		FLMBOOL				bOverwrite,			// Overwrite destination file?
		FLMUINT *			puiBytesCopied);	// Number of bytes copied.

	RCODE Rename(									// Rename a file.
		const char *		pszFilePath,		// File to be renamed
		const char *		pszNewFilePath);	// New file name

	// Miscellaneous methos

	RCODE SetReadOnly(
		const char *	pszFileName,
		FLMBOOL			bReadOnly);
		
	RCODE GetSectorSize(							// Get the sector size of the volume for
		const char *		pFileName,			// this file.
		FLMUINT *			puiSectorSize);

private:

#ifdef FLM_NLM
	RCODE _CreateDir(
		const char *		pszPathToDirectory);
#endif

	FINLINE RCODE getFileHdl(
		F_FileHdlImp **	ppFileHdl)
	{
		if( (*ppFileHdl = f_new F_FileHdlImp) == NULL)
		{
			return( RC_SET( FERR_MEM));
		}

		return( FERR_OK);
	}

	RCODE RemoveEmptyDir(
		const char *		pszDirPath);

#if defined( FLM_UNIX)
	RCODE renameSafe(
		const char *		pszSrcFile,
		const char *		pszDestFile);

	RCODE targetIsDir(
		const char	*		pszPath,
		FLMBOOL *			pbIsDir);
#endif
};

#include "fpackoff.h"

#endif
