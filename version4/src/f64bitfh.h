//-------------------------------------------------------------------------
// Desc:	Abstraction class for 64 bit files - class definition.
// Tabs:	3
//
//		Copyright (c) 2001-2006 Novell, Inc. All Rights Reserved.
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
// $Id: f64bitfh.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef F64BITFH_H
#define F64BITFH_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define F_64BIT_FHDL_LIST_SIZE						8
#define F_64BIT_FHDL_DEFAULT_MAX_FILE_SIZE		((FLMUINT)0xFFFFFFFF)

typedef struct
{
	F_FileHdl *		pFileHdl;
	FLMUINT			uiFileNum;
	FLMBOOL			bDirty;
} FH_INFO;
																		
/****************************************************************************
Desc:	This object is used to simulate a 64-bit file system.
****************************************************************************/
class F_64BitFileHandle : public F_Base
{
public:

	F_64BitFileHandle(
		FLMUINT			uiMaxFileSize = F_64BIT_FHDL_DEFAULT_MAX_FILE_SIZE);
		
	virtual ~F_64BitFileHandle();

	void Close(
		FLMBOOL			bDelete = FALSE);

												
	RCODE Create(
		const char *	pIoPath);
	
	RCODE CreateUnique(
		char *			pIoPath,
		const char *	pszFileExtension);

	RCODE Delete(
		const char *	pIoPath);

	RCODE Open(
		const char *	pIoPath);

	RCODE Flush( void);

	RCODE Read(
		FLMUINT64		ui64Offset,
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE Write(
		FLMUINT64		ui64Offset,
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesWritten);

	RCODE GetPath(
		char *			pszFilePath);

	FINLINE RCODE Size(
		FLMUINT64 *		pui64FileSize)
	{
		*pui64FileSize = m_ui64EOF;
		return( FERR_OK);
	}

private:

	RCODE GetFileHdl(
		FLMUINT			uiFileNum,
		FLMBOOL			bGetForWrite,
		F_FileHdl **	ppFileHdl);	

	RCODE CreateLockFile(
		const char *	pszBasePath);

	void ReleaseLockFile(
		const char *	pszBasePath,
		FLMBOOL			bDelete);

	FINLINE void FormatFileNum(
		FLMUINT			uiFileNum,
		char *			pucStr)
	{
		f_sprintf( pucStr, "%08X.64", (unsigned)uiFileNum);
	}

	RCODE GetFileNum(
		const char *	pucFileName,
		FLMUINT *		puiFileNum);

	FINLINE void DataFilePath(
		FLMUINT			uiFileNum,
		char *			pszPath)
	{
		char		ucFileName[ F_FILENAME_SIZE];

		f_strcpy( pszPath, m_ucPath);
		FormatFileNum( uiFileNum, ucFileName);
		f_pathAppend( pszPath, ucFileName);
	}

	FINLINE FLMUINT GetFileNum(
		FLMUINT64		ui64Offset)
	{
		return( (FLMUINT)(ui64Offset / (FLMUINT64)m_uiMaxFileSize));
	}

	FINLINE FLMUINT GetFileOffset(
		FLMUINT64		ui64Offset)
	{
		return( (FLMUINT)(ui64Offset % (FLMUINT64)m_uiMaxFileSize));
	}

	FH_INFO				m_pFileHdlList[ F_64BIT_FHDL_LIST_SIZE];
	char					m_ucPath[ F_PATH_MAX_SIZE];
	FLMBOOL				m_bOpen;
	FLMUINT64			m_ui64EOF;
	FLMUINT				m_uiMaxFileSize;
	F_FileHdl *			m_pLockFileHdl;
};

#include "fpackoff.h"

#endif
