//------------------------------------------------------------------------------
//	Desc:	This interface encapsulates the concept of a file system directory.
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
// $Id: fdir.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FDIR_H
#define FDIR_H

class F_DirHdl;

#if defined( FLM_WIN)

	typedef struct
	{
		 HANDLE					findHandle;
		 WIN32_FIND_DATA		findBuffer;
		 char 	   			szSearchPath[ F_PATH_MAX_SIZE];
		 FLMUINT					uiSearchAttrib;
	} F_IO_FIND_DATA;

	#define XF_IO_FA_NORMAL			FILE_ATTRIBUTE_NORMAL		// Normal file
	#define XF_IO_FA_RDONLY			FILE_ATTRIBUTE_READONLY		// Read only attribute
	#define XF_IO_FA_HIDDEN			FILE_ATTRIBUTE_HIDDEN		// Hidden file
	#define XF_IO_FA_SYSTEM			FILE_ATTRIBUTE_SYSTEM		// System file
	#define XF_IO_FA_VOLUME			FILE_ATTRIBUTE_VOLUME		// Volume label
	#define XF_IO_FA_DIRECTORY		FILE_ATTRIBUTE_DIRECTORY	// Directory
	#define XF_IO_FA_ARCHIVE		FILE_ATTRIBUTE_ARCHIVE		// Archive

#elif defined( FLM_UNIX) || defined( FLM_NLM)

	typedef struct _DirInfo
	{
		mode_t		mode_flag;
		struct stat	FileStat;
		char 			name[ F_PATH_MAX_SIZE+1];
		char			search_path[ F_PATH_MAX_SIZE+1];
		char			full_path[ F_PATH_MAX_SIZE];
		char			pattern_str[ F_PATH_MAX_SIZE];
		char			dirpath[ F_PATH_MAX_SIZE];
		glob_t      globbuf;
	} F_IO_FIND_DATA;

	#define XF_IO_FA_NORMAL			0x01	// Normal file, no attributes
	#define XF_IO_FA_RDONLY			0x02	// Read only attribute
	#define XF_IO_FA_HIDDEN			0x04	// Hidden file
	#define XF_IO_FA_SYSTEM			0x08	// System file
	#define XF_IO_FA_VOLUME			0x10	// Volume label
	#define XF_IO_FA_DIRECTORY		0x20	// Directory
	#define XF_IO_FA_ARCHIVE		0x40	// Archive

#else 
	#error Platform not supported
#endif

RCODE f_fileFindFirst(
	char *				pszSearchPath,
   FLMUINT				uiSearchAttrib,
	F_IO_FIND_DATA	*	find_data,
   char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib);

RCODE f_fileFindNext(
	F_IO_FIND_DATA *	pFindData,
	char *				pszFoundPath,
	FLMUINT *			puiFoundAttrib);

void f_fileFindClose(
	F_IO_FIND_DATA *		pFindData);

/****************************************************************************
Class:   F_Dir
Desc:    Implementation of the F_Directory interface for Win32 and Unix
****************************************************************************/
class F_DirHdl : public IF_DirHdl, public XF_Base
{
public:

	F_DirHdl();

	virtual ~F_DirHdl()
	{

		if( m_bFindOpen)
		{
			f_fileFindClose( &m_FindData);
		}
	}

	/*-- Iteration Methods ---------------------------------------------------*/
	/* Methods to enumerate the contents of a directory. */

	RCODE XFLMAPI Next( void);          // Set the iteration cursor to the next
													// item in the directory

	/* --- Methods for accessing the current item (that the cursor is on) ----*/

	const char * XFLMAPI CurrentItemName( void);

	FINLINE void XFLMAPI CurrentItemPath(
		char *	pszPath)
	{
		if( RC_OK( m_rc))
		{
			f_strcpy( pszPath, m_szDirectoryPath);
			gv_pFileSystem->pathAppend( pszPath, m_szFileName);
		}
	}

	FLMUINT64 XFLMAPI CurrentItemSize( void);

	FLMBOOL XFLMAPI CurrentItemIsDir( void);

	RCODE XFLMAPI OpenDir(					// Open directory
		const char *	pszDirName,			// Directory to be opened.
		const char *	pszPattern);		// File name pattern.

	RCODE XFLMAPI CreateDir(				// Create a directory.
		const char *	pszDirName);		// Name of directory to be created.

	RCODE XFLMAPI RemoveDir(				// Remove a directory
		const char *	pszDirPath);		// Name of directory to be removed

private:

	char					m_szDirectoryPath[ F_PATH_MAX_SIZE];  // Path to directory
	char					m_szPattern[ F_PATH_MAX_SIZE]; // Pattern for matching
	FLMUINT32			m_ui32RefCount;
	RCODE					m_rc;
	FLMBOOL				m_bFirstTime;  // Indicates whether to use FindFirst or FindNext
	FLMBOOL				m_bFindOpen;   // Indicates if need to call f_fileFindClose
	FLMBOOL				m_EOF;         // Indicates if EndOfDirectory has been reached
	char					m_szFileName[ F_PATH_MAX_SIZE]; // Next item found in directory
	FLMUINT				m_uiAttrib;
	F_IO_FIND_DATA		m_FindData;
};

#endif 		// #ifndef FDIR_H
