//------------------------------------------------------------------------------
// Desc:	Contains the methods for the F_FileHdl class for UNIX.
//
// Tabs:	3
//
//		Copyright (c) 1999-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fposix.cpp 3123 2006-01-24 17:19:50 -0700 (Tue, 24 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#include "ftksys.h"

#if defined( FLM_UNIX)

#ifdef FLM_AIX
	#ifndef _LARGE_FILES
		#define _LARGE_FILES
	#endif
	#include <stdio.h>
#endif

#include <sys/types.h>
#ifndef FLM_OSX
	#include <aio.h>
#endif

#include <fcntl.h>

extern FLMBOOL		gv_bOkToDoAsyncWrites;

#if defined( FLM_SOLARIS)
	#include <sys/statvfs.h>
#elif defined( FLM_LINUX)
	#include <sys/vfs.h>

	extern FLMUINT		gv_uiLinuxMajorVer;
	extern FLMUINT		gv_uiLinuxMinorVer;
	extern FLMUINT		gv_uiLinuxRevision;
#elif defined( FLM_OSF)

	// Tru64 4.0 does not have this declaration. Tru64 5.0 renames statfs
	// in vague ways, so we put these declarations before including
	// <sys/stat.h>

	// DSS NOTE: statfs declaration below conflicts with one found in
	// sys/mount.h header file, so I commented it out.  This was when I
	// compiled using the GNU compiler.

	struct statfs;
	#include <sys/mount.h>
#endif

/******************************************************************************
Desc:
*******************************************************************************/
F_FileHdl::F_FileHdl()
{
	m_bFileOpened = FALSE;
	m_bDeleteOnRelease = FALSE;
	m_bOpenedReadOnly = FALSE;
	m_pszFileName = NULL;
	
	m_fd = -1;
	m_bDoDirectIO = FALSE;
	m_uiExtendSize = 0;
//	m_uiMaxAutoExtendSize = gv_XFlmSysData.uiMaxFileSize;
	m_uiBytesPerSector = 0;
	m_ui64NotOnSectorBoundMask = 0;
	m_ui64GetSectorBoundMask = 0;
	m_uiExtendSize = 0;
	m_ui64CurrentPos = 0;
	m_bDoDirectIO = FALSE;
	m_bCanDoAsync = FALSE;
	m_pucAlignedBuff = NULL;
	m_uiAlignedBuffSize = 0;
}

/******************************************************************************
Desc:
******************************************************************************/
F_FileHdl::~F_FileHdl()
{
	if( m_bFileOpened)
	{
		(void)close();
	}
	
	if( m_pucAlignedBuff)
	{
		free( m_pucAlignedBuff);
	}
	
	if (m_pszFileName)
	{
		f_free( &m_pszFileName);
	}
}

/***************************************************************************
Desc:	Open or create a file.
***************************************************************************/
RCODE F_FileHdl::openOrCreate(
	const char *	pszFileName,
   FLMUINT			uiAccess,
	FLMBOOL			bCreateFlag)
{
	RCODE			rc = NE_FLM_OK;
	FLMBOOL		bDoDirectIO = FALSE;
	char			szSaveFileName[ F_PATH_MAX_SIZE];
	int         openFlags = O_RDONLY;

#if !defined( FLM_UNIX) || defined( FLM_LINUX) || defined( FLM_SOLARIS)
	bDoDirectIO = (uiAccess & FLM_IO_DIRECT) ? TRUE : FALSE;
#endif

// HPUX needs this defined to access files larger than 2 GB.  The Linux
// man pages *say* it's needed although as of Suse 9.1 it actually
// isn't.  Including this flag on Linux anyway just it case...
#if defined( FLM_HPUX) || defined( FLM_LINUX)
	openFlags |= O_LARGEFILE;
#endif

	// Save the file name in case we have to create the directory

	if( bCreateFlag && (uiAccess & FLM_IO_CREATE_DIR))
	{
		f_strcpy( szSaveFileName, pszFileName);
	}

   if( bCreateFlag)
   {
		openFlags |= O_CREAT;
		if( uiAccess & FLM_IO_EXCL)
		{
	  		openFlags |= O_EXCL;
		}
		else
		{
			openFlags |= O_TRUNC;
		}
	}

   if( !(uiAccess & FLM_IO_RDONLY))
	{
		openFlags |= O_RDWR;
	}
	
   if( !(uiAccess & FLM_IO_RDONLY))
	{
      openFlags |= O_RDWR;
	}
	
	// If doing direct IO, need to get the sector size.

	if( bDoDirectIO)
	{
		if( !m_uiBlockSize)
		{
			bDoDirectIO = FALSE;
		}
		else
		{
			if( RC_BAD( rc = gv_pFileSystem->getSectorSize( 
				pszFileName, &m_uiBytesPerSector)))
			{
				goto Exit;
			}
			
			m_ui64NotOnSectorBoundMask = m_uiBytesPerSector - 1;
			m_ui64GetSectorBoundMask = ~m_ui64NotOnSectorBoundMask;

			// Can't do direct IO if the block size isn't a multiple of
			// the sector size.

			if( m_uiBlockSize < m_uiBytesPerSector ||
				 m_uiBlockSize % m_uiBytesPerSector != 0)
			{
				bDoDirectIO = FALSE;
			}
			else
			{
#if defined( FLM_LINUX)
				FLMUINT		uiMajor = gv_uiLinuxMajorVer;
				FLMUINT		uiMinor = gv_uiLinuxMinorVer;
				FLMUINT		uiRevision = gv_uiLinuxRevision;
																																														
				if( uiMajor > 2 || (uiMajor == 2 && uiMinor > 6) ||
					(uiMajor == 2 && uiMinor == 6 && uiRevision >= 5))
				{
					openFlags |= O_DIRECT;
					
					if( gv_bOkToDoAsyncWrites)
					{
						m_bCanDoAsync = TRUE;
					}
				}
				else
				{
					bDoDirectIO = FALSE;
				}
#elif defined( FLM_SOLARIS)
				if( gv_bOkToDoAsyncWrites)
				{
					m_bCanDoAsync = TRUE;
				}
#endif
			}
		}
	}
	
Retry_Create:

	// Try to create or open the file

	if ((m_fd = ::open( pszFileName, openFlags, 0600)) == -1)
	{
		if ((errno == ENOENT) && (uiAccess & FLM_IO_CREATE_DIR))
		{
			char	szTemp[ F_PATH_MAX_SIZE];
			char	szIoDirPath[ F_PATH_MAX_SIZE];

			uiAccess &= ~FLM_IO_CREATE_DIR;

			// Remove the file name for which we are creating the directory

			if( RC_OK( gv_pFileSystem->pathReduce( szSaveFileName,
													szIoDirPath, szTemp)))
			{
				if( RC_OK( rc = gv_pFileSystem->createDir( szIoDirPath)))
				{
					goto Retry_Create;
				}
				else
				{
					goto Exit;
				}
			}
		}
#ifdef FLM_LINUX
		else if( errno == EINVAL && bDoDirectIO)
		{
			openFlags &= ~O_DIRECT;
			bDoDirectIO = FALSE;
			m_bCanDoAsync = FALSE;
			goto Retry_Create;
		}
#endif
		
		rc = MapPlatformError( errno, NE_FLM_OPENING_FILE);
		goto Exit;
	}

#if defined( FLM_SOLARIS)
	if( bDoDirectIO)
	{
		directio( m_fd, DIRECTIO_ON);
	}
#endif
	
	m_bDoDirectIO = bDoDirectIO;

Exit:

	if( RC_BAD( rc))
	{
		m_fd = -1;
		m_bDoDirectIO = FALSE;
		m_bCanDoAsync = FALSE;
	}
	
   return( rc);
}

/******************************************************************************
Desc:	Create a file 
******************************************************************************/
RCODE FLMAPI F_FileHdl::create(
	const char *	pszFileName,
	FLMUINT			uiIoFlags)
{
	RCODE			rc = NE_FLM_OK;

	flmAssert( m_bFileOpened == FALSE);

	if( m_bDeleteOnRelease)
	{
		// This file handle had better not been used for another file
		// before.  Otherwise, we will get a memory leak.

		flmAssert( m_pszFileName == NULL);

		// Note: 'OpenOrCreate' will set m_pszFileName
		
		if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
		{
			goto Exit;
		}

		f_strcpy( m_pszFileName, pszFileName);
	}

	if( RC_BAD( rc = openOrCreate( pszFileName, uiIoFlags, TRUE)))
	{
		goto Exit;
	}
	
	m_bFileOpened = TRUE;
	m_ui64CurrentPos = 0;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	if( RC_BAD( rc) && m_bDeleteOnRelease && m_pszFileName)
	{
		f_free( &m_pszFileName);
	}

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FLMAPI F_FileHdl::createUnique(
	char *			pszDirName,
	const char *	pszFileExtension,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = NE_FLM_OK;
	char *			pszTmp;
	FLMBOOL			bModext = TRUE;
	FLMUINT			uiBaseTime = 0;
	FLMBYTE			ucHighByte = 0;
	char				szFileName[ F_FILENAME_SIZE];
	char *			pszDirPath;
	char				szDirPath[ F_PATH_MAX_SIZE];
	char				szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT			uiCount;

	flmAssert( !m_bFileOpened);
	f_memset( szFileName, 0, sizeof( szFileName));

	if( m_bDeleteOnRelease)
	{
		// This file handle had better not been used for another file
		// before.  Otherwise, we will get a memory leak.

		flmAssert( !m_pszFileName);
	}

	if( !pszDirName || pszDirName[ 0] == '\0')
	{
		f_strcpy( szDirPath, "./");
	}
	else
	{
		f_strcpy( szDirPath, pszDirName);
	}
	pszDirPath = &szDirPath [0];

   // Search backwards replacing trailing spaces with NULLs.

	pszTmp = pszDirPath;
	pszTmp += (f_strlen( pszTmp) - 1);
	while (*pszTmp == ' ' && pszTmp >= pszDirPath)
	{
		*pszTmp = 0;
		pszTmp--;
	}

	// Append a slash if one isn't already there

	if (pszTmp >= pszDirPath && *pszTmp != '/')
	{
		pszTmp++;
		*pszTmp++ = '/';
	}
	else
	{
		pszTmp++;
	}
	*pszTmp = 0;

	if( pszFileExtension && f_strlen( pszFileExtension) >= 3)
	{
		bModext = FALSE;
	}

	uiCount = 0;
	do
	{
		gv_pFileSystem->pathCreateUniqueName( &uiBaseTime, szFileName,
										pszFileExtension,
										&ucHighByte, bModext);

		f_strcpy( szTmpPath, pszDirPath);
		gv_pFileSystem->pathAppend( szTmpPath, szFileName);
		if( m_pszFileName)
		{
			f_free( &m_pszFileName);
		}
		
		rc = create( szTmpPath, uiIoFlags | FLM_IO_EXCL);
		
		if (rc == NE_FLM_IO_DISK_FULL)
		{
			gv_pFileSystem->deleteFile( pszDirPath);
			goto Exit;
		}
		
		if( rc == NE_FLM_IO_PATH_NOT_FOUND || rc == NE_FLM_IO_INVALID_PASSWORD)
		{
			goto Exit;
		}
	} while ((rc != NE_FLM_OK) && (uiCount++ < 10));

   // Check if the path was created

   if( uiCount >= 10 && rc != NE_FLM_OK)
   {
		rc = RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE);
		goto Exit;
   }

	m_bFileOpened = TRUE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;

	// Created file name needs to be returned.

	f_strcpy( pszDirName, szTmpPath);

Exit:

	if( RC_BAD( rc) && m_pszFileName)
	{
		f_free( &m_pszFileName);
		m_pszFileName = NULL;
	}
	
	return( rc);
}

/******************************************************************************
Desc:	Open a file
******************************************************************************/
RCODE FLMAPI F_FileHdl::open(
	const char *	pszFileName,
	FLMUINT			uiIoFlags)
{
	RCODE			rc = NE_FLM_OK;

	flmAssert( !m_bFileOpened);

	if( m_bDeleteOnRelease)
	{
		// This file handle had better not been used for another file
		// before.  Otherwise, we will get a memory leak.

		flmAssert( !m_pszFileName);

		if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszFileName)))
		{
			goto Exit;
		}
	}

	// Loop on error open conditions.

	for( ;;)
	{
		if( RC_OK( rc = openOrCreate( pszFileName, uiIoFlags, FALSE)))
		{
			break;
		}

		if( rc != NE_FLM_IO_TOO_MANY_OPEN_FILES)
		{
			goto Exit;
		}

		// If for some reason we cannot open the file, then
		// try to close some other file handle in the list.

//		gv_XFlmSysData.pFileHdlMgr->releaseOneAvail( FALSE);
	}

	m_bFileOpened = TRUE;
	m_ui64CurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & FLM_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & FLM_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	if( RC_BAD( rc) && m_bDeleteOnRelease && m_pszFileName)
	{
		f_free( &m_pszFileName);
		m_pszFileName = NULL;
	}

	return( rc);
}

/******************************************************************************
Desc:	Close a file
******************************************************************************/
RCODE FLMAPI F_FileHdl::close( void)
{
	FLMBOOL	bDeleteAllowed = TRUE;
	RCODE		rc = NE_FLM_OK;

	if( !m_bFileOpened)
	{
		goto Exit;
	}

	::close( m_fd);
	
	m_fd = -1;
	m_bFileOpened = FALSE;
	m_bOpenedReadOnly = FALSE;
	m_bOpenedExclusive = FALSE;

	if( m_bDeleteOnRelease)
	{
		flmAssert( m_pszFileName);

		if( bDeleteAllowed)
		{
			gv_pFileSystem->deleteFile( m_pszFileName);
		}
		m_bDeleteOnRelease = FALSE;

		f_free( &m_pszFileName);
		m_pszFileName = NULL;
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Make sure all file data is safely on disk
******************************************************************************/
RCODE FLMAPI F_FileHdl::flush( void)
{
#ifdef FLM_SOLARIS
	// Direct I/O on Solaris is ADVISORY, meaning that the
	// operating system may or may not actually honor the
	// option for some or all operations on a given file.
	// Thus, the only way to guarantee that writes are on disk
	// is to call fdatasync.
	//
	// If a process is killed (with SIGKILL or SIGTERM), the
	// dirty cache buffers associated with open files will be discarded unless
	// the process intercepts the signal and properly closes the files.
	//
	// NOTES FROM THE UNIX MAN PAGES ON SIGNALS
	//
	// When killing a process or series of processes, it is common sense
	// to start trying with the least dangerous signal, SIGTERM. That way,
	// programs that care about an orderly shutdown get the chance to follow
	// the procedures that they have been designed to execute when getting
	// the SIGTERM signal, such as cleaning up and closing open files.  If you
	// send a SIGKILL to a process, you remove any chance for the process
	// to do a tidy cleanup and shutdown, which might have unfortunate
	// consequences.

	if( fdatasync( m_fd) != 0)
	{
		 return( MapPlatformError( errno, NE_FLM_FLUSHING_FILE));
	}
#else
	if( !m_bDoDirectIO)
	{
	#ifdef FLM_OSX
		if( fsync( m_fd) != 0)
	#else
		if( fdatasync( m_fd) != 0)
	#endif
		{
			 return( MapPlatformError( errno, NE_FLM_FLUSHING_FILE));
		}
	}
#endif
	
	return( NE_FLM_OK);
}

/******************************************************************************
Desc:	Read from a file
******************************************************************************/
RCODE F_FileHdl::directRead(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,	
   void *			pvBuffer,
	FLMBOOL			bBuffHasFullSectors,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesRead;
	FLMBYTE *		pucReadBuffer;
	FLMBYTE *		pucDestBuffer;
	FLMUINT			uiMaxBytesToRead;
	FLMINT			iTmp;
	FLMBOOL			bHitEOF;
	
	flmAssert( m_bFileOpened);
	flmAssert( m_bDoDirectIO);
	
	if( puiBytesRead)
	{
		*puiBytesRead = 0;
	}

	if( ui64ReadOffset == FLM_IO_CURRENT_POS)
	{
		ui64ReadOffset = m_ui64CurrentPos;
	}

	// This loop does multiple reads (if necessary) to get all of the
	// data.  It uses aligned buffers and reads at sector offsets.

	pucDestBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{
		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if( (ui64ReadOffset & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT64)(FLMUINT)pucDestBuffer) & m_ui64NotOnSectorBoundMask) ||
			 ((uiBytesToRead & m_ui64NotOnSectorBoundMask) &&
			  !bBuffHasFullSectors))
		{
			if( !m_pucAlignedBuff)
			{
				if( RC_BAD( rc = allocAlignBuffer()))
				{
					goto Exit;
				}
			}
			pucReadBuffer = m_pucAlignedBuff;

			// Must read enough bytes to cover all of the sectors that
			// contain the data we are trying to read.  The value of
			// (ui64ReadOffset & m_ui64NotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round that up to the next sector
			// to get the total number of bytes we are going to read.

			uiMaxBytesToRead = (FLMUINT)roundUpToSectorMultiple( 
					uiBytesToRead + (ui64ReadOffset & m_ui64NotOnSectorBoundMask));

			// Can't read more than the aligned buffer will hold.

			if( uiMaxBytesToRead > m_uiAlignedBuffSize)
			{
				uiMaxBytesToRead = m_uiAlignedBuffSize;
			}
		}
		else
		{
			uiMaxBytesToRead = (FLMUINT)roundUpToSectorMultiple( uiBytesToRead);
			flmAssert( uiMaxBytesToRead >= uiBytesToRead);
			pucReadBuffer = pucDestBuffer;
		}

		bHitEOF = FALSE;

		if( (iTmp = pread( m_fd, pucReadBuffer,
			uiMaxBytesToRead, getSectorStartOffset( ui64ReadOffset))) == -1)
		{
			rc = MapPlatformError( errno, NE_FLM_READING_FILE);
			goto Exit;
		}
		uiBytesRead = (FLMUINT)iTmp;

		if( uiBytesRead < uiMaxBytesToRead)
		{
			bHitEOF = TRUE;
		}

		// If the offset we want to read from is not on a sector
		// boundary, increment the read buffer pointer to the
		// offset where the data we need starts and decrement the
		// bytes read by the difference between the start of the
		// sector and the actual read offset.

		if( ui64ReadOffset & m_ui64NotOnSectorBoundMask)
		{
			pucReadBuffer += (ui64ReadOffset & m_ui64NotOnSectorBoundMask);
			flmAssert( uiBytesRead >= m_uiBytesPerSector);
			uiBytesRead -= (FLMUINT)(ui64ReadOffset & m_ui64NotOnSectorBoundMask);
		}

		// If bytes read is more than we actually need, truncate it back
		// so that we only copy what we actually need.

		if( uiBytesRead > uiBytesToRead)
		{
			uiBytesRead = uiBytesToRead;
		}

		uiBytesToRead -= uiBytesRead;

		if( puiBytesRead)
		{
			(*puiBytesRead) += uiBytesRead;
		}
		
		m_ui64CurrentPos = ui64ReadOffset + uiBytesRead;

		// If using a different buffer for reading, copy the
		// data read into the destination buffer.

		if( pucDestBuffer != pucReadBuffer)
		{
			f_memcpy( pucDestBuffer, pucReadBuffer, uiBytesRead);
		}

		if( !uiBytesToRead)
		{
			break;
		}

		// Still more to read - did we hit EOF above?

		if( bHitEOF)
		{
			rc = RC_SET( NE_FLM_IO_END_OF_FILE);
			break;
		}

		pucDestBuffer += uiBytesRead;
		ui64ReadOffset += uiBytesRead;
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Read from a file
******************************************************************************/
RCODE FLMAPI F_FileHdl::read(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,	
   void *			pvBuffer,
   FLMUINT *		puiBytesRead)
{
	RCODE				rc = NE_FLM_OK;
	FLMINT      	iBytesRead;
	
	flmAssert( m_bFileOpened);

	if( m_bDoDirectIO)
	{
		rc = directRead( ui64ReadOffset, uiBytesToRead, 
			pvBuffer, FALSE, puiBytesRead);
		goto Exit;
	}

	if( ui64ReadOffset == FLM_IO_CURRENT_POS)
	{
		ui64ReadOffset = m_ui64CurrentPos;
	}
	
	if( (iBytesRead = pread( m_fd, pvBuffer, 
		uiBytesToRead, ui64ReadOffset)) == -1)
	{
		rc = MapPlatformError(errno, NE_FLM_READING_FILE);
		goto Exit;
	}

	if( puiBytesRead)
	{
		*puiBytesRead = (FLMUINT)iBytesRead;
	}
	
	m_ui64CurrentPos = ui64ReadOffset + (FLMUINT)iBytesRead;
	
	if( (FLMUINT)iBytesRead < uiBytesToRead)
	{
		rc = RC_SET( NE_FLM_IO_END_OF_FILE);
		goto Exit;
	}

Exit:

	return( rc);
}

/******************************************************************************
Note:	This function assumes that the pvBuffer that is passed in is
		a multiple of a the sector size.
******************************************************************************/
RCODE FLMAPI F_FileHdl::sectorRead(
	FLMUINT64		ui64ReadOffset,
	FLMUINT			uiBytesToRead,
   void *			pvBuffer,
   FLMUINT *		puiBytesRead)
{
	if( m_bDoDirectIO)
	{
		return( directRead( ui64ReadOffset, uiBytesToRead, 
			pvBuffer, TRUE, puiBytesRead));
	}
	else
	{
		return( read( ui64ReadOffset, uiBytesToRead, pvBuffer, puiBytesRead));
	}
}

/******************************************************************************
Desc:	Sets current position of file.
******************************************************************************/
RCODE FLMAPI F_FileHdl::seek(
	FLMUINT64		ui64Offset,
	FLMINT			iWhence,
	FLMUINT64 *		pui64NewOffset)
{
	RCODE			rc = NE_FLM_OK;

	switch( iWhence)
	{
		case FLM_IO_SEEK_CUR:
		{
			m_ui64CurrentPos += ui64Offset;
			break;
		}
		
		case FLM_IO_SEEK_SET:
		{
			m_ui64CurrentPos = ui64Offset;
			break;
		}
		
		case FLM_IO_SEEK_END:
		{
			if( RC_BAD( rc = size( &m_ui64CurrentPos)))
			{
				goto Exit;
			}
			
			break;
		}
		
		default:
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
	if( pui64NewOffset)
	{
		*pui64NewOffset = m_ui64CurrentPos;
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Return the size of the file
******************************************************************************/
RCODE FLMAPI F_FileHdl::size(
	FLMUINT64 *		pui64Size)
{
	RCODE				rc = NE_FLM_OK;
   struct stat 	statBuf;

   if( fstat( m_fd, &statBuf) == -1)
   {
      rc = MapPlatformError( errno, NE_FLM_GETTING_FILE_SIZE);
		goto Exit;
   }
	
	*pui64Size = statBuf.st_size;
	
Exit:

	return( rc);
}

/******************************************************************************
Desc:
******************************************************************************/
RCODE FLMAPI F_FileHdl::tell(
	FLMUINT64 *		pui64Offset)
{
	*pui64Offset = m_ui64CurrentPos;
	return( NE_FLM_OK);
}

/******************************************************************************
Desc:	Truncate the file to the indicated size
******************************************************************************/
RCODE FLMAPI F_FileHdl::truncate(
	FLMUINT64		ui64Size)
{
	RCODE				rc = NE_FLM_OK;

	flmAssert( m_bFileOpened);

	if( ftruncate( m_fd, ui64Size) == -1)
	{
		rc = MapPlatformError( errno, NE_FLM_TRUNCATING_FILE);
		goto Exit;
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Write to a file
******************************************************************************/
RCODE FLMAPI F_FileHdl::write(
	FLMUINT64		ui64WriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWrittenRV)
{
	RCODE			rc = NE_FLM_OK;
	FLMINT      iBytesWritten = 0;
	
	flmAssert( m_bFileOpened);

	if( m_bDoDirectIO)
	{
		rc = directWrite( ui64WriteOffset, uiBytesToWrite, pvBuffer, 
			uiBytesToWrite, NULL, puiBytesWrittenRV, FALSE, TRUE);
		goto Exit;
	}
	
	if( ui64WriteOffset == FLM_IO_CURRENT_POS)
	{
		ui64WriteOffset = m_ui64CurrentPos;
	}

  	if( (iBytesWritten = pwrite( m_fd, pvBuffer,
		uiBytesToWrite, ui64WriteOffset)) == -1)
	{
		rc = MapPlatformError(errno, NE_FLM_WRITING_FILE);
		goto Exit;
	}

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = (FLMUINT)iBytesWritten;
	}
	
	m_ui64CurrentPos = ui64WriteOffset + (FLMUINT)iBytesWritten;
	
	if( (FLMUINT)iBytesWritten < uiBytesToWrite)
	{
		rc = RC_SET( NE_FLM_IO_DISK_FULL);
		goto Exit;
	}

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Allocate an aligned buffer.
******************************************************************************/
RCODE F_FileHdl::allocAlignBuffer( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( m_pucAlignedBuff)
	{
		goto Exit;
	}

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = (FLMUINT)roundUpToSectorMultiple( 64 * 1024);
	
#if defined( FLM_SOLARIS)
	if( (m_pucAlignedBuff = (FLMBYTE *)memalign( 
		sysconf( _SC_PAGESIZE), m_uiAlignedBuffSize)) == NULL)
#elif defined( FLM_OSX)
	if( (m_pucAlignedBuff = (FLMBYTE *)malloc( m_uiAlignedBuffSize)) == NULL)
#else
	if( posix_memalign( (void **)&m_pucAlignedBuff, 
		sysconf( _SC_PAGESIZE), m_uiAlignedBuffSize) == -1)
#endif
	{
		m_uiAlignedBuffSize = 0;
		rc = MapPlatformError( errno, NE_FLM_MEM);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/******************************************************************************
Note:	This routine assumes that the size of pvBuffer is a multiple of
		sector size and can be used to write out full sectors.  Even if
		uiBytesToWrite does not account for full sectors, data from the
		buffer will still be written out - a partial sector on disk will
		not be preserved.
******************************************************************************/
RCODE F_FileHdl::directWrite(
	FLMUINT64		ui64WriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT,
	F_IOBuffer *	pBufferObj,
	FLMUINT *		puiBytesWrittenRV,
	FLMBOOL			bBuffHasFullSectors,
	FLMBOOL			bZeroFill)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiBytesRead;
	FLMUINT			uiMaxBytesToWrite;
	FLMUINT			uiBytesBeingOutput;
	FLMBYTE *		pucWriteBuffer;
	FLMBYTE *		pucSrcBuffer;
#ifdef FLM_DEBUG
	FLMBOOL			bDoAsync = (pBufferObj != NULL) 
										? TRUE 
										: FALSE;
#endif
	FLMBOOL			bDidAsync = FALSE;
	FLMUINT			uiLastWriteOffset;
	FLMUINT			uiLastWriteSize;
	
	flmAssert( m_bFileOpened);

#ifdef FLM_DEBUG
	if( bDoAsync)
	{
		flmAssert( m_bCanDoAsync);
	}
#endif

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = 0;
	}

	if( ui64WriteOffset == FLM_IO_CURRENT_POS)
	{
		ui64WriteOffset = m_ui64CurrentPos;
	}

	// This loop is for direct IO - must make sure we use
	// aligned buffers.
	
	pucSrcBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{
		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if( (ui64WriteOffset & m_ui64NotOnSectorBoundMask) ||
			 (((FLMUINT64)(FLMUINT)pucSrcBuffer) & m_ui64NotOnSectorBoundMask) ||
			 ((uiBytesToWrite & m_ui64NotOnSectorBoundMask) && !bBuffHasFullSectors))
		{
			// Cannot be using a temporary write buffer if we are doing
			// asynchronous writes!

			flmAssert( !bDoAsync || !m_bCanDoAsync);
			
			if( !m_pucAlignedBuff)
			{
				if( RC_BAD( rc = allocAlignBuffer()))
				{
					goto Exit;
				}
			}
			pucWriteBuffer = m_pucAlignedBuff;

			// Must write enough bytes to cover all of the sectors that
			// contain the data we are trying to write out.  The value of
			// (ui64WriteOffset & m_ui64NotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round to the next sector to get the
			// total number of bytes we are going to write.

			uiMaxBytesToWrite = (FLMUINT)roundUpToSectorMultiple( 
				uiBytesToWrite + (ui64WriteOffset & m_ui64NotOnSectorBoundMask));

			// Can't write more than the aligned buffer will hold.

			if( uiMaxBytesToWrite > m_uiAlignedBuffSize)
			{
				uiMaxBytesToWrite = m_uiAlignedBuffSize;
				uiBytesBeingOutput = uiMaxBytesToWrite -
							(FLMUINT)(ui64WriteOffset & m_ui64NotOnSectorBoundMask);
			}
			else
			{
				uiBytesBeingOutput = uiBytesToWrite;
			}

			// If the write offset is not on a sector boundary, or if
			// we are writing a partial sector, we must read the
			// sector into the buffer.

			if( (ui64WriteOffset & m_ui64NotOnSectorBoundMask) ||
				(uiBytesBeingOutput < m_uiBytesPerSector && !bBuffHasFullSectors))
			{
				// Read the first sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if( RC_BAD( rc = read( getSectorStartOffset( ui64WriteOffset),
					m_uiBytesPerSector, pucWriteBuffer, &uiBytesRead)))
				{
					if( rc != NE_FLM_IO_END_OF_FILE)
					{
						goto Exit;
					}

					rc = NE_FLM_OK;
					f_memset( &pucWriteBuffer[ uiBytesRead], 0, 
						m_uiBytesPerSector - uiBytesRead);
				}
			}

			// Finally, copy the data from the source buffer into the
			// write buffer.

			f_memcpy( &pucWriteBuffer[ ui64WriteOffset & m_ui64NotOnSectorBoundMask],
								pucSrcBuffer, uiBytesBeingOutput);
		}
		else
		{
			uiMaxBytesToWrite = (FLMUINT)roundUpToSectorMultiple( uiBytesToWrite);
			uiBytesBeingOutput = uiBytesToWrite;
			pucWriteBuffer = pucSrcBuffer;
			
			if( bZeroFill && uiMaxBytesToWrite > uiBytesToWrite)
			{
				f_memset( &pucWriteBuffer[ uiBytesToWrite], 0,
							uiMaxBytesToWrite - uiBytesToWrite);
			}
		}

		// Position the file to the nearest sector below the write offset.

		uiLastWriteOffset = (FLMUINT)getSectorStartOffset( ui64WriteOffset);
		uiLastWriteSize = uiMaxBytesToWrite;
		
		if( !m_bCanDoAsync || !pBufferObj)
		{
			FLMINT		iBytesWritten;
			
			if( (iBytesWritten = pwrite( m_fd, 
				pucWriteBuffer, uiMaxBytesToWrite, uiLastWriteOffset)) == -1)
			{
				rc = MapPlatformError( errno, NE_FLM_WRITING_FILE);
				goto Exit;
			}

			if( (FLMUINT)iBytesWritten < uiMaxBytesToWrite)
			{
				rc = RC_SET( NE_FLM_IO_DISK_FULL);
				goto Exit;
			}
		}
		else
		{
			struct aiocb *		pAio = pBufferObj->getAIOStruct();
			
			f_memset( pAio, 0, sizeof( struct aiocb));
			pAio->aio_lio_opcode = LIO_WRITE;
			pAio->aio_sigevent.sigev_notify = SIGEV_NONE;
			pAio->aio_fildes = m_fd;
			pAio->aio_offset = uiLastWriteOffset;
			pAio->aio_nbytes = uiMaxBytesToWrite;
			pAio->aio_buf = pucWriteBuffer;
			
			if( aio_write( pAio) == -1)
			{
				rc = MapPlatformError( errno, NE_FLM_WRITING_FILE);
				goto Exit;
			}
			
			pBufferObj->makePending();
			bDidAsync = TRUE;
		}

		uiBytesToWrite -= uiBytesBeingOutput;
		if( puiBytesWrittenRV)
		{
			(*puiBytesWrittenRV) += uiBytesBeingOutput;
		}
		
		m_ui64CurrentPos = ui64WriteOffset + uiBytesBeingOutput;
		
		if( !uiBytesToWrite)
		{
			break;
		}

		flmAssert( !pBufferObj);

		pucSrcBuffer += uiBytesBeingOutput;
		ui64WriteOffset += uiBytesBeingOutput;
	}

Exit:

	if( !bDidAsync && pBufferObj)
	{
		pBufferObj->notifyComplete( rc);
	}

	return( rc);
}

/******************************************************************************
Desc:	Returns flag indicating whether or not we can do async writes.
******************************************************************************/
FLMBOOL FLMAPI F_FileHdl::canDoAsync()
{
	return( m_bCanDoAsync);
}

/******************************************************************************
Desc:	Attempts to lock byte 0 of the file.  This method is used to
		lock byte 0 of the .lck file to ensure that only one process
		has access to a database.
******************************************************************************/
RCODE FLMAPI F_FileHdl::lock( void)
{
	RCODE				rc = NE_FLM_OK;
	struct flock   LockStruct;

	// Lock first byte in file

	f_memset( &LockStruct, 0, sizeof( LockStruct));
	LockStruct.l_type   = F_WRLCK;
	LockStruct.l_whence = SEEK_SET;
	LockStruct.l_start  = 0;
	LockStruct.l_len    = 1;

	if( fcntl( m_fd, F_SETLK, &LockStruct) == -1)
	{
		rc = RC_SET( NE_FLM_IO_FILE_LOCK_ERR);
		goto Exit;
	} 

Exit:

	return( rc);
}

/******************************************************************************
Desc:	Attempts to unlock byte 0 of the file.
******************************************************************************/
RCODE FLMAPI F_FileHdl::unlock( void)
{
	RCODE				rc = NE_FLM_OK;
	struct flock   LockStruct;

	// Lock first byte in file

	f_memset( &LockStruct, 0, sizeof( LockStruct));
	LockStruct.l_type   = F_UNLCK;
	LockStruct.l_whence = SEEK_SET;
	LockStruct.l_start  = 0;
	LockStruct.l_len    = 1;

	if( fcntl( m_fd, F_SETLK, &LockStruct) == -1)
	{
		rc = RC_SET( NE_FLM_IO_FILE_UNLOCK_ERR);
		goto Exit;
	} 

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Determines the kernel version of the linux system we are running on
***************************************************************************/
#ifdef FLM_LINUX
void flmGetLinuxKernelVersion(
	FLMUINT *		puiMajor,
	FLMUINT *		puiMinor,
	FLMUINT *		puiRevision)
{
	int			fd = -1;
	int			iBytesRead;
	char			szBuffer [80];
	char *		pszVer;
	FLMUINT		uiMajorVer = 0;
	FLMUINT		uiMinorVer = 0;
	FLMUINT		uiRevision = 0;

	if( (fd = open( "/proc/version", O_RDONLY, 0600)) == -1)
	{
		goto Exit;
	}

	if( (iBytesRead = read( fd, szBuffer, sizeof( szBuffer))) == -1)
	{
		goto Exit;
	}
	if( (pszVer = f_strstr( szBuffer, "version ")) == NULL)
	{
		goto Exit;
	}
	pszVer += 8;

	while( *pszVer >= '0' && *pszVer <= '9')
	{
		uiMajorVer *= 10;
		uiMajorVer += (FLMUINT)(*pszVer - '0');
		pszVer++;
	}
	
	if( *pszVer == '.')
	{
		pszVer++;
		while (*pszVer >= '0' && *pszVer <= '9')
		{
			uiMinorVer *= 10;
			uiMinorVer += (FLMUINT)(*pszVer - '0');
			pszVer++;
		}
	}
	
	if( *pszVer == '.')
	{
		pszVer++;
		while (*pszVer >= '0' && *pszVer <= '9')
		{
			uiRevision *= 10;
			uiRevision += (FLMUINT)(*pszVer - '0');
			pszVer++;
		}
	}
	
Exit:

	if( fd != -1)
	{
		close( fd);
	}
	
	if( puiMajor)
	{
		*puiMajor = uiMajorVer;
	}
	
	if( puiMinor)
	{
		*puiMinor = uiMinorVer;
	}
	
	if( puiRevision)
	{
		*puiRevision = uiRevision;
	}
}
#endif

/***************************************************************************
Desc:	Determines if the linux system we are running on is 2.4 or greater.
***************************************************************************/
#ifdef FLM_LINUX
FLMUINT flmGetLinuxMaxFileSize( void)
{
#ifdef FLM_32BIT
	return( FLM_MAXIMUM_FILE_SIZE);
#else
	FLMUINT	uiMaxFileSize = 0x7FF00000;
	
//	flmAssert( gv_XFlmSysData.uiLinuxMajorVer);
	
	// Is version 2.4 or greater?

//	if( gv_XFlmSysData.uiLinuxMajorVer > 2 || 
//		 (gv_XFlmSysData.uiLinuxMajorVer == 2 && 
//		  gv_XFlmSysData.uiLinuxMinorVer >= 4))
//	{
//		uiMaxFileSize = FLM_MAXIMUM_FILE_SIZE;
//	}
	
	return( uiMaxFileSize);
#endif
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_LINUX
FINLINE FLMUINT64 flmGetLinuxMemInfoValue(
	char *			pszMemInfoBuffer,
	char *			pszTag)
{
	char *			pszTmp;
	FLMUINT64		ui64Bytes = 0;

	if( (pszTmp = f_strstr( pszMemInfoBuffer, pszTag)) == NULL)
	{
		return( 0);
	}
	
	pszTmp += f_strlen( pszTag);
	
	while( *pszTmp == ASCII_SPACE)
	{
		pszTmp++;
	}

	while( *pszTmp >= '0' && *pszTmp <= '9')
	{
		ui64Bytes *= 10;
		ui64Bytes += (FLMUINT)(*pszTmp - '0');
		pszTmp++;
	}
	
	return( ui64Bytes * 1024);
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_LINUX
void flmGetLinuxMemInfo(
	FLMUINT64 *		pui64TotalMem,
	FLMUINT64 *		pui64AvailMem)
{
	int				fd = -1;
	int				iBytesRead;
	int				iMemInfoBufSize = 4096;
	char *			pszMemInfoBuf = NULL;
	FLMUINT64		ui64TotalMem = 0;
	FLMUINT64		ui64AvailMem = 0;

	if( (pszMemInfoBuf = (char *)malloc( iMemInfoBufSize)) == NULL)
	{
		goto Exit;
	}
	
	if( (fd = open( "/proc/meminfo", O_RDONLY, 0600)) == -1)
	{
		goto Exit;
	}

	if( (iBytesRead = read( fd, pszMemInfoBuf, iMemInfoBufSize - 1)) == -1)
	{
		goto Exit;
	}
	
	pszMemInfoBuf[ iBytesRead] = 0;
	
	if( (ui64TotalMem = 
		flmGetLinuxMemInfoValue( pszMemInfoBuf, "MemTotal:")) != 0)
	{
		ui64AvailMem = 
				flmGetLinuxMemInfoValue( pszMemInfoBuf, "MemFree:") +
				flmGetLinuxMemInfoValue( pszMemInfoBuf, "Buffers:") +
				flmGetLinuxMemInfoValue( pszMemInfoBuf, "Cached:");
	}
	
Exit:

	if( pui64TotalMem)
	{
		*pui64TotalMem = ui64TotalMem;
	}
	
	if( pui64AvailMem)
	{
		*pui64AvailMem = ui64AvailMem;
	}

	if( pszMemInfoBuf)
	{
		free( pszMemInfoBuf);
	}
	
	if( fd != -1)
	{
		close( fd);
	}
}
#endif

/****************************************************************************
Desc: This routine gets the block size for the file system a file belongs to.
****************************************************************************/
FLMUINT flmGetFSBlockSize(
	FLMBYTE *	pszFileName)
{
	FLMUINT		uiFSBlkSize = 4096;
	FLMBYTE *	pszTmp = pszFileName + f_strlen( (const char *)pszFileName) - 1;
	FLMBYTE *	pszDir;
	FLMBYTE		ucRestoreByte = 0;

	while( pszTmp != pszFileName && *pszTmp != '/')
	{
		pszTmp--;
	}
	
	if( *pszTmp == '/')
	{
		if (pszTmp == pszFileName)
		{
			pszTmp++;
		}
		ucRestoreByte = *pszTmp;
		*pszTmp = 0;
		pszDir = pszFileName;
	}
	else
	{
		pszDir = (FLMBYTE *)".";
	}

#if defined( FLM_SOLARIS)
	struct statvfs statfsbuf;
	if (statvfs( (char *)pszDir, &statfsbuf) == 0)
	{
		uiFSBlkSize = (FLMUINT)statfsbuf.f_bsize;
	}
#elif defined( FLM_LINUX) || defined( FLM_OSF)
	struct statfs statfsbuf;
	if (statfs( (char *)pszDir, &statfsbuf) == 0)
	{
		uiFSBlkSize = (FLMUINT)statfsbuf.f_bsize;
	}
#endif

	if( ucRestoreByte)
	{
		*pszTmp = ucRestoreByte;
	}
	
	return( uiFSBlkSize);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
static void sparc_asm_code( void)
{
	asm( ".align 8");
	asm( ".global sparc_atomic_add_32");
	asm( ".type sparc_atomic_add_32, #function");
	asm( "sparc_atomic_add_32:");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "    ld [%o0], %l0");
	asm( "    add %l0, %o1, %l2");
	asm( "    cas [%o0], %l0, %l2");
	asm( "    cmp %l0, %l2");
	asm( "    bne sparc_atomic_add_32");
	asm( "    nop");
	asm( "    add %l2, %o1, %o0");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "retl");
	asm( "nop");
	
	asm( ".align 8");
	asm( ".global sparc_atomic_xchg_32");
	asm( ".type sparc_atomic_xchg_32, #function");
	asm( "sparc_atomic_xchg_32:");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "    ld [%o0], %l0");
	asm( "    mov %o1, %l1");
	asm( "    cas [%o0], %l0, %l1");
	asm( "    cmp %l0, %l1");
	asm( "    bne sparc_atomic_xchg_32");
	asm( "    nop");
	asm( "    mov %l0, %o0");
	asm( "    membar #LoadLoad | #LoadStore | #StoreStore | #StoreLoad");
	asm( "retl");
	asm( "nop");
}
#endif

#endif // FLM_UNIX

#if defined( FLM_WATCOM_NLM)
	int fposixDummy(void)
	{
		return( 0);
	}
#endif
