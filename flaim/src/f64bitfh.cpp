//-------------------------------------------------------------------------
// Desc:	Abstraction class for 64 bit files.
// Tabs:	3
//
//		Copyright (c) 2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: f64bitfh.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
****************************************************************************/
F_64BitFileHandle::F_64BitFileHandle(
	FLMUINT			uiMaxFileSize)
{
	m_bOpen = FALSE;
	m_ucPath[ 0] = 0;
	m_ui64EOF = 0;
	m_pLockFileHdl = NULL;
	f_memset( m_pFileHdlList, 0, sizeof( FH_INFO) * F_64BIT_FHDL_LIST_SIZE);
	m_uiMaxFileSize = uiMaxFileSize;
	if( !m_uiMaxFileSize)
	{
		m_uiMaxFileSize = F_64BIT_FHDL_DEFAULT_MAX_FILE_SIZE;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
F_64BitFileHandle::~F_64BitFileHandle()
{
	if( m_bOpen)
	{
		Close();
	}

	flmAssert( !m_pLockFileHdl);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_64BitFileHandle::ReleaseLockFile(
	const char *	pszBasePath,
	FLMBOOL			bDelete)
{
#ifndef FLM_UNIX
	F_UNREFERENCED_PARM( bDelete);
	F_UNREFERENCED_PARM( pszBasePath);
#endif
	
	if( m_pLockFileHdl)
	{
		/*
		Release the lock file
		*/

		(void)m_pLockFileHdl->Close();
		m_pLockFileHdl->Release();
		m_pLockFileHdl = NULL;

#ifdef FLM_UNIX
		if( bDelete)
		{
			char		szTmpPath[ F_PATH_MAX_SIZE];

			/*
			Delete the lock file
			*/

			f_strcpy( szTmpPath, pszBasePath);
			f_pathAppend( szTmpPath, "64.LCK");
			gv_FlmSysData.pFileSystem->Delete( szTmpPath);
		}
#endif
	}
}
	
/****************************************************************************
Desc:	Closes all data files associated with the object
****************************************************************************/
void F_64BitFileHandle::Close(
	FLMBOOL			bDelete)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLoop;
	F_DirHdl *		pDir = NULL;
	char				szTmpPath[ F_PATH_MAX_SIZE];

	if( !m_bOpen)
	{
		return;
	}

	for( uiLoop = 0; uiLoop < F_64BIT_FHDL_LIST_SIZE; uiLoop++)
	{
		if( m_pFileHdlList[ uiLoop].pFileHdl)
		{
			if( m_pFileHdlList[ uiLoop].bDirty)
			{
				(void)m_pFileHdlList[ uiLoop].pFileHdl->Flush();
			}
			m_pFileHdlList[ uiLoop].pFileHdl->Close();
			m_pFileHdlList[ uiLoop].pFileHdl->Release();
			f_memset( &m_pFileHdlList[ uiLoop], 0, sizeof( FH_INFO));
		}
	}

	m_ui64EOF = 0;
	m_bOpen = FALSE;

	if( bDelete)
	{
		if( RC_OK( gv_FlmSysData.pFileSystem->OpenDir(
			m_ucPath, "*.64", &pDir)))
		{
			/*
			Remove all data files
			*/

			for( rc = pDir->Next(); !RC_BAD( rc) ; rc = pDir->Next() )
			{
				pDir->CurrentItemPath( szTmpPath);
				flmAssert( f_strstr( szTmpPath, ".64") != 0);
				(void)gv_FlmSysData.pFileSystem->Delete( szTmpPath);
			}

			pDir->Release();
			pDir = NULL;
		}

		/*
		Release and delete the lock file
		*/

		(void)ReleaseLockFile( m_ucPath, TRUE);

		/*
		Remove the directory
		*/

		(void)gv_FlmSysData.pFileSystem->RemoveDir( m_ucPath);
	}
	else
	{
		(void)ReleaseLockFile( m_ucPath, FALSE);
	}
}
												
/****************************************************************************
Desc:	Removes a 64-bit file
****************************************************************************/
RCODE F_64BitFileHandle::Delete(
	const char *	pszPath)
{
	RCODE				rc = FERR_OK;
	F_DirHdl *		pDir = NULL;
	char				szTmpPath[ F_PATH_MAX_SIZE];

	// Can't use this handle to delete something if we already
	// have a file open.

	if( m_bOpen)
	{
		// Can't jump to exit, because it calls ReleaseLockFile

		return( RC_SET( FERR_FAILURE));
	}

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Exists( pszPath)))
	{
		goto Exit;
	}

	if( !gv_FlmSysData.pFileSystem->IsDir( pszPath))
	{
		/*
		If the path specifies a single file rather than a 
		64-bit directory, just go ahead and delete the file.
		*/
	
		rc = gv_FlmSysData.pFileSystem->Delete( pszPath);
		goto Exit;
	}

	if( RC_BAD( rc = CreateLockFile( pszPath)))
	{
		goto Exit;
	}

	if( RC_OK( gv_FlmSysData.pFileSystem->OpenDir(
		pszPath, "*.64", &pDir)))
	{
		/*
		Remove all data files
		*/

		for( rc = pDir->Next(); !RC_BAD( rc) ; rc = pDir->Next())
		{
			pDir->CurrentItemPath( szTmpPath);
			flmAssert( f_strstr( szTmpPath, ".64") != 0);
			(void)gv_FlmSysData.pFileSystem->Delete( szTmpPath);
		}

		pDir->Release();
		pDir = NULL;
		rc = FERR_OK;
	}

	/*
	Release and delete the lock file
	*/

	(void)ReleaseLockFile( pszPath, TRUE);

	/*
	Remove the directory
	*/

	(void)gv_FlmSysData.pFileSystem->RemoveDir( pszPath);

Exit:

	(void)ReleaseLockFile( pszPath, FALSE);

	return( rc);
}

/****************************************************************************
Desc:	Creates a new 64-bit "file"
****************************************************************************/
RCODE F_64BitFileHandle::Create(
	const char *	pszPath)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bCreatedDir = FALSE;

	if( m_bOpen)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->CreateDir( pszPath)))
	{
		goto Exit;
	}

	f_strcpy( m_ucPath, pszPath);
	bCreatedDir = TRUE;

	/*
	Create the lock file
	*/

	if( RC_BAD( rc = CreateLockFile( m_ucPath)))
	{
		goto Exit;
	}

	/*
	Initialize the EOF to 0 and set the state to open
	*/
	
	m_ui64EOF = 0;
	m_bOpen = TRUE;

Exit:

	/*
	Release the lock file
	*/

	if( RC_BAD( rc))
	{
		(void)ReleaseLockFile( m_ucPath, TRUE);
		if( bCreatedDir)
		{
			(void)gv_FlmSysData.pFileSystem->RemoveDir( m_ucPath);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	Creates a new 64-bit file with a unique, generated name
****************************************************************************/
RCODE F_64BitFileHandle::CreateUnique(
	char *			pszPath,
	const char *	pszFileExtension)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCount;
	FLMBOOL		bModext = TRUE;
	FLMBOOL		bCreatedDir = FALSE;
	FLMUINT		uiBaseTime = 0;
	char			ucHighByte = 0;
	char			szDirName[ F_FILENAME_SIZE];
	char			szTmpPath[ F_PATH_MAX_SIZE];
	char			szBasePath[ F_PATH_MAX_SIZE];

	if( m_bOpen)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( !pszPath || pszPath[ 0] == '\0')
	{
#if defined( FLM_UNIX)
		f_strcpy( szBasePath, "./");
#elif defined( FLM_NLM)
		f_strcpy( szBasePath, "SYS:_NETWARE");
#else
		szBasePath[ 0] = '\0';
#endif
	}
	else
	{
		f_strcpy( szBasePath, pszPath);
	}

	if ((pszFileExtension) && (f_strlen( pszFileExtension) >= 3))
	{
		bModext = FALSE;
	}

	uiCount = 0;
	szDirName[ 0] = '\0';
	do
	{
		f_pathCreateUniqueName( &uiBaseTime, szDirName, pszFileExtension,
										&ucHighByte, bModext);

		f_strcpy( szTmpPath, szBasePath);
		f_pathAppend( szTmpPath, szDirName);
		rc = gv_FlmSysData.pFileSystem->CreateDir( szTmpPath);
	} while ((rc != FERR_OK) && (uiCount++ < 20));

	if( RC_BAD( rc))
	{
		goto Exit;
	}

	f_strcpy( m_ucPath, szTmpPath);
	bCreatedDir = TRUE;

	/*
	Create the lock file
	*/

	if( RC_BAD( rc = CreateLockFile( m_ucPath)))
	{
		goto Exit;
	}

	/*
	Initialize the EOF to 0 and set the state to open
	*/
	
	m_ui64EOF = 0;
	m_bOpen = TRUE;

Exit:

	/*
	Release the lock file
	*/

	if( RC_BAD( rc))
	{
		ReleaseLockFile( m_ucPath, TRUE);

		if( bCreatedDir)
		{
			(void)gv_FlmSysData.pFileSystem->RemoveDir( m_ucPath);
		}
	}

	return( rc);
}
	
/****************************************************************************
Desc:	Opens an existing 64-bit file
****************************************************************************/
RCODE F_64BitFileHandle::Open(
	const char *	pszPath)
{
	RCODE				rc = FERR_OK;
	F_DirHdl *		pDir = NULL;
	FLMUINT			uiTmp;
	FLMUINT			uiHighFileNum = 0;
	FLMUINT			uiHighOffset = 0;

	if( m_bOpen)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( RC_BAD( gv_FlmSysData.pFileSystem->Exists( pszPath)) ||
		!gv_FlmSysData.pFileSystem->IsDir( pszPath))
	{
		rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
		goto Exit;
	}

	f_strcpy( m_ucPath, pszPath);

	/*
	Create the lock file
	*/

	if( RC_BAD( rc = CreateLockFile( m_ucPath)))
	{
		goto Exit;
	}

	/*
	Need to determine the current EOF
	*/

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->OpenDir(
		m_ucPath, "*.64", &pDir)))
	{
		goto Exit;
	}

	/*
	Find all data files to determine the EOF
	*/

	for( rc = pDir->Next(); !RC_BAD( rc) ; rc = pDir->Next() )
	{
		if( RC_OK( GetFileNum( pDir->CurrentItemName(), &uiTmp)))
		{
			if( uiTmp >= uiHighFileNum)
			{
				uiHighFileNum = uiTmp;
				uiHighOffset = pDir->CurrentItemSize();
			}
		}
	}
	rc = FERR_OK;

	m_ui64EOF = (((FLMUINT64)uiHighFileNum) * (FLMUINT64)m_uiMaxFileSize) +	
		(FLMUINT64)uiHighOffset;
	m_bOpen = TRUE;

Exit:

	if( pDir)
	{
		pDir->Release();
	}

	/*
	Release the lock file
	*/

	if( RC_BAD( rc))
	{
		ReleaseLockFile( m_ucPath, FALSE);
	}

	return( rc);
}

/****************************************************************************
Desc:	Flushes cached data to the data file(s)
****************************************************************************/
RCODE F_64BitFileHandle::Flush( void)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLoop;

	if( !m_bOpen)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < F_64BIT_FHDL_LIST_SIZE; uiLoop++)
	{
		if( m_pFileHdlList[ uiLoop].bDirty)
		{
			if( RC_BAD( rc = m_pFileHdlList[ uiLoop].pFileHdl->Flush()))
			{
				goto Exit;
			}
			m_pFileHdlList[ uiLoop].bDirty = FALSE;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads data from the file
****************************************************************************/
RCODE F_64BitFileHandle::Read(
	FLMUINT64	ui64Offset,				// Offset to begin reading
	FLMUINT		uiLength,				// Number of bytes to read
	void *		pvBuffer,				// Buffer
	FLMUINT *	puiBytesRead)			// [out] Number of bytes read
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiFileNum = GetFileNum( ui64Offset);
	FLMUINT			uiFileOffset = GetFileOffset( ui64Offset);
	FLMUINT			uiTmp;
	FLMUINT			uiTotalBytesRead = 0;
	FLMUINT			uiBytesToRead;
	FLMUINT			uiMaxReadLen;
	F_FileHdl *		pFileHdl;

	/*
	Handle the case of a 0-byte read
	*/

	if( !uiLength)
	{
		if( ui64Offset >= m_ui64EOF)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
		}
		goto Exit;
	}

	/*
	Read the data file(s), moving to new files as needed.
	*/

	for( ;;)
	{
		if( ui64Offset >= m_ui64EOF)
		{
			rc = RC_SET( FERR_IO_END_OF_FILE);
			goto Exit;
		}

		uiMaxReadLen = m_uiMaxFileSize - uiFileOffset;
		flmAssert( uiMaxReadLen != 0);
		uiTmp = (uiLength >= uiMaxReadLen ? uiMaxReadLen : uiLength);
		uiBytesToRead = (((FLMUINT64)uiTmp > (FLMUINT64)(m_ui64EOF - ui64Offset)) 
								? (FLMUINT)(m_ui64EOF - ui64Offset) 
								: uiTmp);

		if( RC_BAD( rc = GetFileHdl( uiFileNum, FALSE, &pFileHdl)))
		{
			if( rc == FERR_IO_PATH_NOT_FOUND)
			{
				/*
				Handle the case of a sparse file by filling the unread
				portion of the buffer with zeros.
				*/

				f_memset( pvBuffer, 0, uiBytesToRead);
				uiTmp = uiBytesToRead;
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pFileHdl->Read( uiFileOffset, uiBytesToRead, 
				pvBuffer, &uiTmp)))
			{
				if( rc == FERR_IO_END_OF_FILE)
				{
					/*
					Handle the case of a sparse file by filling the unread
					portion of the buffer with zeros.
					*/

					f_memset( &(((FLMBYTE *)(pvBuffer))[ uiTmp]), 
						0, (FLMUINT)(uiBytesToRead - uiTmp));
					uiTmp = uiBytesToRead;
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
		}

		uiTotalBytesRead += uiTmp;
		uiLength -= uiTmp;
		if( !uiLength)
		{
			break;
		}

		/*
		Set up for next read
		*/

		pvBuffer = ((FLMBYTE *)pvBuffer) + uiTmp;
		ui64Offset += uiTmp;
		uiFileNum = GetFileNum( ui64Offset);
		uiFileOffset = GetFileOffset( ui64Offset);
	}

Exit:

	*puiBytesRead = uiTotalBytesRead;
	return( rc);
}

/****************************************************************************
Desc:	Writes data to the file
****************************************************************************/
RCODE F_64BitFileHandle::Write(
	FLMUINT64	ui64Offset,				// Offset
	FLMUINT		uiLength,				// Number of bytes to write.
	void *		pvBuffer,				// Buffer that contains bytes to be written
	FLMUINT *	puiBytesWritten)		// Number of bytes written.
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiFileNum = GetFileNum( ui64Offset);
	FLMUINT			uiFileOffset = GetFileOffset( ui64Offset);
	FLMUINT			uiTmp;
	FLMUINT			uiTotalBytesWritten = 0;
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiMaxWriteLen;
	F_FileHdl *		pFileHdl;

	/*
	Don't allow zero-length writes
	*/

	flmAssert( uiLength);

	/*
	Write to the data file(s), moving to new files as needed.
	*/

	for( ;;)
	{
		if( RC_BAD( rc = GetFileHdl( uiFileNum, TRUE, &pFileHdl)))
		{
			goto Exit;
		}

		uiMaxWriteLen = m_uiMaxFileSize - uiFileOffset;
		flmAssert( uiMaxWriteLen != 0);
		uiBytesToWrite = uiLength >= uiMaxWriteLen ? uiMaxWriteLen : uiLength;

		uiTmp = 0;
		rc = pFileHdl->Write( uiFileOffset, uiBytesToWrite, pvBuffer, &uiTmp);

		uiTotalBytesWritten += uiTmp;
		uiLength -= uiTmp;
		ui64Offset += uiTmp;
		
		if( RC_BAD( rc))
		{
			goto Exit;
		}

		if( !uiLength)
		{
			break;
		}

		/*
		Set up for next write
		*/

		pvBuffer = ((FLMBYTE *)pvBuffer) + uiTmp;
		uiFileNum = GetFileNum( ui64Offset);
		uiFileOffset = GetFileOffset( ui64Offset);
	}

Exit:

	if( ui64Offset > m_ui64EOF)
	{
		m_ui64EOF = ui64Offset;
	}

	*puiBytesWritten = uiTotalBytesWritten;
	return( rc);
}

/****************************************************************************
Desc:	Returns the requested file handle
****************************************************************************/
RCODE F_64BitFileHandle::GetFileHdl(
	FLMUINT			uiFileNum,
	FLMBOOL			bGetForWrite,
	F_FileHdl **	ppFileHdl)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiSlot;
	F_FileHdl *		pTmpHdl;
	char				ucPath[ F_PATH_MAX_SIZE];

	flmAssert( m_bOpen);

	*ppFileHdl = NULL;

	uiSlot = uiFileNum % F_64BIT_FHDL_LIST_SIZE;
	pTmpHdl = m_pFileHdlList[ uiSlot].pFileHdl;

	if( pTmpHdl && m_pFileHdlList[ uiSlot].uiFileNum != uiFileNum)
	{
		if( RC_BAD( rc = pTmpHdl->Flush()))
		{
			goto Exit;
		}

		pTmpHdl->Close();
		pTmpHdl->Release();
		pTmpHdl = NULL;

		f_memset( &m_pFileHdlList[ uiSlot], 0, sizeof( FH_INFO));
	}
	
	if( !pTmpHdl)
	{
		DataFilePath( uiFileNum, ucPath);
		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Open( ucPath, F_IO_RDWR,
			&pTmpHdl)))
		{
			if( rc == FERR_IO_PATH_NOT_FOUND && bGetForWrite)
			{
				if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Create( ucPath,
 					F_IO_RDWR | F_IO_EXCL, &pTmpHdl)))
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		m_pFileHdlList[ uiSlot].pFileHdl = pTmpHdl;
		m_pFileHdlList[ uiSlot].uiFileNum = uiFileNum;
		flmAssert( !m_pFileHdlList[ uiSlot].bDirty);
	}

	*ppFileHdl = m_pFileHdlList[ uiSlot].pFileHdl;
	if( bGetForWrite)
	{
		m_pFileHdlList[ uiSlot].bDirty = TRUE;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Given a data file name, returns the file's number
****************************************************************************/
RCODE F_64BitFileHandle::GetFileNum(
	const char *		pucFileName,
	FLMUINT *			puiFileNum)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCnt = 0;
	FLMUINT		uiDigit;
	FLMUINT		uiFileNum = 0;

	if( f_strlen( pucFileName) != 11) // XXXXXXXX.64
	{
		rc = RC_SET( FERR_IO_INVALID_PATH);
		goto Exit;
	}

	if( f_strcmp( &pucFileName[ 8], ".64") != 0)
	{
		rc = RC_SET( FERR_IO_INVALID_PATH);
		goto Exit;
	}

	while( uiCnt < 8)
	{
		uiDigit = pucFileName[ uiCnt];
		if( uiDigit >= NATIVE_LOWER_A && uiDigit <= NATIVE_LOWER_F)
		{
			uiDigit = (FLMUINT)(uiDigit - NATIVE_LOWER_A) + 10;
		}
		else if( uiDigit >= NATIVE_UPPER_A && uiDigit <= NATIVE_UPPER_F)
		{
			uiDigit = (FLMUINT)(uiDigit - NATIVE_UPPER_A) + 10;
		}
		else if( uiDigit >= NATIVE_ZERO && uiDigit <= NATIVE_NINE)
		{
			uiDigit -= NATIVE_ZERO;
		}
		else
		{
			/*
			Invalid character found in the file name
			*/

			rc = RC_SET( FERR_IO_INVALID_PATH);
			goto Exit;
		}

		uiFileNum <<= 4;
		uiFileNum += uiDigit;
		uiCnt++;
	}

	*puiFileNum = uiFileNum;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	This routine obtains exclusive access to a 64-bit file by creating
		a .lck file.  The object holds the .lck file open as long as the
		64-bit file is open.
****************************************************************************/
RCODE F_64BitFileHandle::CreateLockFile(
	const char *		pszBasePath)
{
	RCODE					rc = FERR_OK;
	char					szLockPath [F_PATH_MAX_SIZE];
	F_FileHdlImp *		pLockFileHdl = NULL;

	f_strcpy( szLockPath, pszBasePath);
	f_pathAppend( szLockPath, "64.LCK");

	/*
	Attempt to create the lock file.  If it fails, the lock file
	may have been left because of a crash.  Hence, we first try
	to delete the file.  If that succeeds, we then attempt to
	create the file again.  If it, or the 2nd create fail, we simply 
	return an access denied error.
	*/

#ifndef FLM_UNIX
	if( RC_BAD( gv_FlmSysData.pFileSystem->Create( szLockPath,
		F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYRW | F_IO_DELETE_ON_CLOSE,
		(F_FileHdl **)&pLockFileHdl)))
	{
		if( RC_BAD( gv_FlmSysData.pFileSystem->Delete( szLockPath)))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
		else if (RC_BAD( gv_FlmSysData.pFileSystem->Create( szLockPath,
			F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYRW | F_IO_DELETE_ON_CLOSE, 
			(F_FileHdl **)&pLockFileHdl)))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
	}
#else
	if( RC_BAD( gv_FlmSysData.pFileSystem->Create( szLockPath,
		F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYRW,
		(F_FileHdl **)&pLockFileHdl)))
	{
		if( RC_BAD( gv_FlmSysData.pFileSystem->Open( szLockPath,
			F_IO_RDWR | F_IO_SH_DENYRW,
			(F_FileHdl **)&pLockFileHdl)))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
	}

	if( RC_BAD( pLockFileHdl->Lock()))
	{
		rc = RC_SET( FERR_IO_ACCESS_DENIED);
		goto Exit;
	}
#endif

	m_pLockFileHdl = pLockFileHdl;
	pLockFileHdl = NULL;

Exit:

	if (pLockFileHdl)
	{
		(void)pLockFileHdl->Close();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}
	return( rc);
}
