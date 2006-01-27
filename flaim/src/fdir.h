//-------------------------------------------------------------------------
// Desc:	File system directory class - definitions.
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
// $Id: fdir.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FDIR_H
#define FDIR_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

class F_DirHdl;							// Forward Reference
typedef F_DirHdl *			F_DirHdl_p;

#if defined( FLM_WIN)

	typedef struct
	{
		 HANDLE					findHandle;
		 WIN32_FIND_DATA		findBuffer;
		 char  	   			szSearchPath[ F_PATH_MAX_SIZE];
		 FLMUINT					uiSearchAttrib;
	} F_IO_FIND_DATA;

	#define F_IO_FA_NORMAL		FILE_ATTRIBUTE_NORMAL		/* Normal file */
	#define F_IO_FA_RDONLY		FILE_ATTRIBUTE_READONLY		/* Read only attribute */
	#define F_IO_FA_HIDDEN		FILE_ATTRIBUTE_HIDDEN		/* Hidden file */
	#define F_IO_FA_SYSTEM		FILE_ATTRIBUTE_SYSTEM		/* System file */
	#define F_IO_FA_VOLUME		FILE_ATTRIBUTE_VOLUME		/* Volume label */
	#define F_IO_FA_DIRECTORY	FILE_ATTRIBUTE_DIRECTORY	/* Directory */
	#define F_IO_FA_ARCHIVE		FILE_ATTRIBUTE_ARCHIVE		/* Archive */

#elif defined( FLM_UNIX)

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

	#define F_IO_FA_NORMAL		0x01	/* Normal file, no attributes */
	#define F_IO_FA_RDONLY		0x02	/* Read only attribute */
	#define F_IO_FA_HIDDEN		0x04	/* Hidden file */
	#define F_IO_FA_SYSTEM		0x08	/* System file */
	#define F_IO_FA_VOLUME		0x10	/* Volume label */
	#define F_IO_FA_DIRECTORY	0x20	/* Directory */
	#define F_IO_FA_ARCHIVE		0x40	/* Archive */

#elif !defined( FLM_NLM)

	#error Platform not supported
	
#endif

#if defined( FLM_WIN) || defined( FLM_UNIX)
RCODE f_fileFindFirst(
	const char *			pszSearchPath,
   FLMUINT					uiSearchAttrib,
	F_IO_FIND_DATA	*		find_data,
   char *					pszFoundPath,
	FLMUINT *				puiFoundAttrib);

RCODE f_fileFindNext(
	F_IO_FIND_DATA *		pFindData,
	char *					pszFoundPath,
	FLMUINT *				puiFoundAttrib);

void f_fileFindClose(
	F_IO_FIND_DATA *		pFindData);
#endif

/****************************************************************************
Desc:    Implementation of the F_Directory interface for Win and Unix
****************************************************************************/
class F_DirHdlImp : public F_DirHdl
{
public:

	F_DirHdlImp()
	{
		m_rc = FERR_OK;
		m_bFirstTime = TRUE;
		m_bFindOpen = FALSE;
		m_uiAttrib = 0;
		m_ucPattern[ 0] = '\0';
#ifdef FLM_NLM
		m_lVolumeNumber = 0;
		m_lCurrentEntryNumber = 0xFFFFFFFFL;
#endif
	}

	virtual ~F_DirHdlImp()
	{
#ifndef FLM_NLM
		if( m_bFindOpen)
		{
			f_fileFindClose( &m_FindData);
		}
#endif
	}

	virtual RCODE Next( void);
												
	const char * CurrentItemName( void);

	FLMUINT CurrentItemSize( void);

	FLMBOOL CurrentItemIsDir( void);

	FINLINE void CurrentItemPath(
		char *				pszPath)
	{
		if( RC_OK( m_rc))
		{
			f_strcpy( pszPath, m_DirectoryPath);
#ifdef FLM_NLM
			f_pathAppend( pszPath, CurrentItemName());
#else
			f_pathAppend( pszPath, m_szFileName);
#endif
		}
	}

	RCODE OpenDir(
		const char *			pszPath,
		const char *			pszPattern);

	RCODE CreateDir(
		const char *			pszDirPath);

	RCODE RemoveDir(
		const char *			pszDirPath);

private:

	RCODE _CreateDir(
		const char *			pszDirPath);

	char									m_DirectoryPath[ F_PATH_MAX_SIZE];
	char									m_ucPattern[ F_PATH_MAX_SIZE];
	RCODE									m_rc;
	FLMBOOL								m_bFirstTime;
	FLMBOOL								m_bFindOpen;
	FLMBOOL								m_EOF;
	char									m_szFileName[ F_PATH_MAX_SIZE];
	FLMUINT								m_uiAttrib;
#ifndef FLM_NLM
	F_IO_FIND_DATA						m_FindData;
#else
	LONG									m_lVolumeNumber;
	LONG									m_lDirectoryNumber;
	LONG									m_lCurrentEntryNumber;
	struct DirectoryStructure *	m_pCurrentItem;
	char									m_ucTempBuffer[ F_FILENAME_SIZE];
#endif
};

#include "fpackoff.h"

#endif
