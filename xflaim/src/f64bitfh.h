//------------------------------------------------------------------------------
// Desc:	FLAIM's 64-bit file abstraction class
//
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
// $Id: f64bitfh.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef F64BITFH_H
#define F64BITFH_H

#define F_64BIT_FHDL_LIST_SIZE						8
#define F_64BIT_FHDL_DEFAULT_MAX_FILE_SIZE		((FLMUINT)0xFFFFFFFF)

typedef struct
{
	IF_FileHdl *	pFileHdl;
	FLMUINT			uiFileNum;
	FLMBOOL			bDirty;
} FH_INFO;

/*===========================================================================
Desc:		This object is used to simulate a 64-bit file system.
===========================================================================*/
class F_64BitFileHandle : public XF_RefCount, public XF_Base
{
public:

	F_64BitFileHandle(					// Constructor
		FLMUINT			uiMaxFileSize = F_64BIT_FHDL_DEFAULT_MAX_FILE_SIZE);

	~F_64BitFileHandle();				// Destructor

	void Close(								// Close a file - The destructor will call this
		FLMBOOL			bDelete = FALSE);


	RCODE Create(
		const char *	pszPath);	// File to be created (creates a directory for
												// the data files)

	RCODE CreateUnique(					// Create a new file (with a unique file name --
												// creates a directory for the data files).
		const char *	pszPath,			// Directory where the file is to be created
		const char *	pszFileExtension);	// Extension to be used on the new file.

	RCODE Delete(							// Delete a file
		const char *	pszPath);

	RCODE Open(								// Initiates access to an existing file.
		const char *	pszPath);		// File to be opened (specifies the data
												// file directory).

	RCODE Flush( void);					// Flushes a file's buffers to disk

	RCODE Read(								// Reads a buffer of data from a file
		FLMUINT64	ui64Offset,				// Offset to begin reading
		FLMUINT		uiLength,				// Number of bytes to read
		void *		pvBuffer,				// Buffer
		FLMUINT *	puiBytesRead);			// [out] Number of bytes read

	RCODE Write(							// Writes a buffer of data to a file.
		FLMUINT64	ui64Offset,				// Offset
		FLMUINT		uiLength,				// Number of bytes to write.
		void *		pvBuffer,				// Buffer that contains bytes to be written
		FLMUINT *	puiBytesWritten);		// Number of bytes written.

	RCODE GetPath(							// Returns the full path to the data file
												// directory
		char *	pszFilePath);

	FINLINE RCODE Size(
		FLMUINT64 *	pui64FileSize)
	{
		*pui64FileSize = m_ui64EOF;
		return( NE_XFLM_OK);
	}

	RCODE Truncate(
		FLMUINT64	ui64NewSize);

private:

	/*
	Methods
	*/

	RCODE GetFileHdl(						// Get / Open / Create data file
		FLMUINT				uiFileNum,
		FLMBOOL				bGetForWrite,
		IF_FileHdl **		ppFileHdl);

	RCODE CreateLockFile(				// Creates a lock file for exclusive
												// access to the 64-bit file.  This is
												// necessary since this object maintains
												// an in-memory EOF offset.
		const char *		pszBasePath);

	FINLINE void ReleaseLockFile(
		const char *		pszBasePath,
		FLMBOOL				bDelete)
	{
#ifndef FLM_UNIX
		F_UNREFERENCED_PARM( bDelete);
		F_UNREFERENCED_PARM( pszBasePath);
#endif

		if( m_pLockFileHdl)
		{

			// Release the lock file

			(void)m_pLockFileHdl->Close();
			m_pLockFileHdl->Release();
			m_pLockFileHdl = NULL;

#ifdef FLM_UNIX
			if( bDelete)
			{
				char		szTmpPath[ F_PATH_MAX_SIZE];

				// Delete the lock file

				f_strcpy( szTmpPath, pszBasePath);
				gv_pFileSystem->pathAppend( szTmpPath, "64.LCK");
				gv_pFileSystem->Delete( szTmpPath);
			}
#endif
		}
	}

	/*===========================================================================
	Private:	FormatFileNum
	Desc:		Formats the file number into a string using characters 0 - 9 and
				a - z
	===========================================================================*/
	FINLINE void FormatFileNum(
		FLMUINT	uiFileNum,
		char *	pszStr)
	{
		f_sprintf( pszStr, "%08X.64", (unsigned)uiFileNum);
	}

	RCODE GetFileNum(
		const char *	pszFileName,
		FLMUINT *		puiFileNum);

	/*===========================================================================
	Private:	DataFilePath
	Desc:		Returns the specified data file's path
	===========================================================================*/
	FINLINE void DataFilePath(
		FLMUINT		uiFileNum,
		char *		pszPath)
	{
		char	szFileName[ 13];

		f_strcpy( pszPath, m_szPath);
		FormatFileNum( uiFileNum, szFileName);
		gv_pFileSystem->pathAppend( pszPath, szFileName);
	}

	FINLINE FLMUINT GetFileNum(
		FLMUINT64		ui64Offset)
	{
		return( (FLMUINT)(ui64Offset / m_uiMaxFileSize));
	}

	FINLINE FLMUINT GetFileOffset(
		FLMUINT64		ui64Offset)
	{
		return( (FLMUINT)(ui64Offset % m_uiMaxFileSize));
	}

	/*
	Data
	*/

	FH_INFO				m_pFileHdlList[ F_64BIT_FHDL_LIST_SIZE];
	char					m_szPath[ F_PATH_MAX_SIZE]; // Data file directory path
	FLMBOOL				m_bOpen;
	FLMUINT64			m_ui64EOF;
	FLMUINT				m_uiMaxFileSize;
	IF_FileHdl *		m_pLockFileHdl;
};

#endif 		// #ifndef F64BITFH_H
