//------------------------------------------------------------------------------
// Desc:	This include file contains the class definitions for FLAIM's
//			file system class.
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
// $Id: ffilesys.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FFILESYS_H
#define FFILESYS_H

/*===========================================================================
Desc:		The F_FileSystem class provides a file system abstraction for
			interacting with the underlying O/S file system.
===========================================================================*/
class F_FileSystem : public IF_FileSystem, public XF_Base
{
public:

	F_FileSystem()
	{
	}

	virtual ~F_FileSystem() 				// Destructor
	{
	}

	RCODE XFLMAPI Create(					// Create a new file handle
		const char *	pszFileName,		// Name of file to be created
		FLMUINT			uiIoFlags,			// Access amd Mode flags
		IF_FileHdl **	ppFileHdl);			// Returns open file handle object.

	RCODE XFLMAPI CreateBlockFile(		// Create a new block-oriented file handle
		const char *	pszFileName,		// Name of file to be created
		FLMUINT			uiIoFlags,			// Access amd Mode flags
		FLMUINT			uiBlockSize,		// Block size
		IF_FileHdl **	ppFileHdl);			// Returns open file handle object.

	RCODE XFLMAPI CreateUnique(			// Create a new file (with a unique
													// file name).
		const char *	pszDirName,			// Directory where file is to be
													// created.
		const char *	pszFileExtension,	// Extension to be used on file.
		FLMUINT			uiIoFlags,			// Access and Mode flags.
		IF_FileHdl **	ppFileHdl);			// Returns open file handle object.

	RCODE XFLMAPI Open(						// Open an existing file.
		const char *	pszFileName,		// Name of file to be opened.
		FLMUINT			uiIoFlags,			// Access and Mode flags.
		IF_FileHdl **	ppFileHdl);			// Returns open file handle object.

	RCODE XFLMAPI OpenBlockFile(			// Open an existing block-oriented file.
		const char *	pszFileName,		// Name of file to be opened.
		FLMUINT			uiIoFlags,			// Access and Mode flags.
		FLMUINT			uiBlockSize,		// Block size.
		IF_FileHdl **	ppFileHdl);			// Returns open file handle object.

	RCODE XFLMAPI OpenDir(					// Open a directory
		const char *	pszDirName,			// Directory to be opened.
		const char *	pszPattern,			// File name pattern.
		IF_DirHdl **	ppDirHdl);			// Returns open directory handle
													// object.

	RCODE XFLMAPI CreateDir(				// Create a directory
		const char *	pszDirName);		// Directory to be created.

	RCODE XFLMAPI RemoveDir(				// Remove a directory
		const char *	pszDirName,			// Directory to be removed.
		FLMBOOL			bClear = FALSE);	// OK to delete files if dir is not empty?

	RCODE XFLMAPI Exists(					// See if a file or directory exists.
		const char *	pszFileName);		// Name of file or directory to check.

	FLMBOOL XFLMAPI IsDir(					// See if a path is a directory.
		const char *	pszFileName);		// Name of path to check.

	RCODE XFLMAPI GetTimeStamp(			// Get the date/time when the file
													// was last updated.
		const char *	pszFileName,		// Path to file
		FLMUINT *		puiTimeStamp);		// Buffer in which time stamp is
													// returned.

	RCODE XFLMAPI Delete(					// Delete a file or directory
		const char *	pszFileName);		// Name of file or directory to delete.

	RCODE XFLMAPI Copy(						// Copy a file.
		const char *	pszSrcFileName,	// Name of source file to be copied.
		const char *	pszDestFileName,	// Name of destination file.
		FLMBOOL			bOverwrite,			// Overwrite destination file?
		FLMUINT64 *		pui64BytesCopied);// Number of bytes copied.

	RCODE XFLMAPI Rename(					// Rename a file.
		const char *	pszFileName,		// File to be renamed
		const char *	pszNewFileName);	// New file name

	void XFLMAPI pathParse(					// ftkpath.cpp
		const char *	pszPath,
		char *			pszServer,
		char *			pszVolume,
		char *			pszDirPath,
		char *			pszFileName);

	RCODE XFLMAPI pathReduce(				// ftkpath.cpp
		const char *	pszSourcePath,
		char *			pszDestPath,
		char *			pszString);

	RCODE XFLMAPI pathAppend(				// ftkpath.cpp
		char *			pszPath,
		const char *	pszPathComponent);

	RCODE XFLMAPI pathToStorageString(	// ftkpath.cpp
		const char *	pszPath,
		char *			pszString);

	void XFLMAPI pathCreateUniqueName(	// ftkpath.cpp
		FLMUINT *		puiTime,
		char *			pszFileName,
		const char *	pszFileExt,
		FLMBYTE *		pHighChars,
		FLMBOOL			bModext);

	FLMBOOL XFLMAPI doesFileMatch(		// ftkpath.cpp
		const char *	pszFileName,
		const char *	pszTemplate);

	RCODE XFLMAPI GetSectorSize(			// Get the sector size of the volume for
		const char *	pszFileName,		// this file.
		FLMUINT *		puiSectorSize);

	RCODE SetReadOnly(
		const char *	pszFileName,
		FLMBOOL			bReadOnly);

private:

#if defined( FLM_UNIX)
	RCODE unix_RenameSafe(
		const char *	pszSrcFile,
		const char *	pszDestFile);

	RCODE unix_TargetIsDir(
		const char	*	tpath,
		FLMBOOL *		isdir);
#endif
};

#ifndef ALLOCATE_SYS_DATA
extern F_FileSystem *	gv_pFileSystem;
#else
F_FileSystem *				gv_pFileSystem;
#endif

#endif 		// #ifndef FFILESYS_H

