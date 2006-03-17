//-------------------------------------------------------------------------
// Desc:	Posix File I/O
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
// $Id: fposix.cpp 12331 2006-01-23 10:19:55 -0700 (Mon, 23 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_UNIX)

#ifdef FLM_AIX
	#ifndef _LARGE_FILES
		#define _LARGE_FILES
	#endif
	#include <stdio.h>
	#include <sys/atomic_op.h>
#endif

#include <sys/types.h>
#include <fcntl.h>

#if !defined( O_SYNC)
	#define O_SYNC 	0				
#endif

#if !defined( O_DSYNC)
	#define O_DSYNC 	O_SYNC
#endif

#define MAX_CREATION_TRIES		10

#if defined( FLM_SOLARIS)
	#include <sys/statvfs.h>
#elif defined( FLM_LINUX)
	#include <sys/vfs.h>
#endif

extern RCODE gv_CriticalFSError;

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdlImp::F_FileHdlImp()
{
	m_fd = INVALID_HANDLE_VALUE;
	m_uiMaxAutoExtendSize = gv_FlmSysData.uiMaxFileSize;
	m_uiBlockSize = 0;
	m_uiBytesPerSector = 0;
	m_uiNotOnSectorBoundMask = 0;
	m_uiGetSectorBoundMask = 0;
	m_uiExtendSize = 0;
	m_uiCurrentPos = 0;
	m_bDoDirectIO = FALSE;
	m_bCanDoAsync = FALSE;
	m_pucAlignedBuff = NULL;
	m_uiAlignedBuffSize = 0;
}

/****************************************************************************
Desc:
****************************************************************************/
F_FileHdlImp::~F_FileHdlImp()
{
	if( m_bFileOpened)
	{
		Close();
	}
	
	if( m_pucAlignedBuff)
	{
		free( m_pucAlignedBuff);
	}
}

/***************************************************************************
Desc:		Open or create a file.
***************************************************************************/
RCODE F_FileHdlImp::OpenOrCreate(
	const char *		pFileName,
   FLMUINT				uiAccess,
	FLMBOOL				bCreateFlag)
{
	RCODE					rc = FERR_OK;
	FLMBOOL				bDoDirectIO = FALSE;
	char					szSaveFileName[ F_PATH_MAX_SIZE];
	int         		openFlags = O_RDONLY;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

#if !defined( FLM_UNIX)
	bDoDirectIO = (uiAccess & F_IO_DIRECT) ? TRUE : FALSE;
#endif

// HPUX needs this defined to access files larger than 2 GB.  The Linux
// man pages *say* it's needed although as of Suse 9.1 it actually
// isn't.  Including this flag on Linux anyway just it case...

#if defined( FLM_HPUX) || defined( FLM_LINUX)
	openFlags |= O_LARGEFILE;
#endif
	if( uiAccess & F_IO_DELETE_ON_CLOSE)
	{
		if( !m_pszIoPath)
		{
			if( RC_BAD( rc = f_alloc( F_PATH_MAX_SIZE, &m_pszIoPath)))
			{
				goto Exit;
			}
		}
		
		f_strcpy( m_pszIoPath, pFileName);
		m_bDeleteOnClose = TRUE;
	}
	else
	{
		m_bDeleteOnClose = FALSE;
	}

	// Save the file name in case we have to create the directory

	if( bCreateFlag && (uiAccess & F_IO_CREATE_DIR))
	{
		f_strcpy( szSaveFileName, pFileName);
	}

   if( bCreateFlag)
   {
		openFlags |= O_CREAT;
		
		if( uiAccess & F_IO_EXCL)
		{
	  		openFlags |= O_EXCL;
		}
	}
	
	if( uiAccess & F_IO_TRUNC)
	{
		openFlags |= O_TRUNC;
	}

   if( !(uiAccess & F_IO_RDONLY))
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
			if( RC_BAD( rc = gv_FlmSysData.pFileSystem->GetSectorSize( 
				pFileName, &m_uiBytesPerSector)))
			{
				goto Exit;
			}
			
			m_uiNotOnSectorBoundMask = m_uiBytesPerSector - 1;
			m_uiGetSectorBoundMask = ~m_uiNotOnSectorBoundMask;

			// Can't do direct IO if the block size isn't a multiple of
			// the sector size.

			if( m_uiBlockSize < m_uiBytesPerSector ||
				 m_uiBlockSize % m_uiBytesPerSector != 0)
			{
				bDoDirectIO = FALSE;
			}
		}
	}
	
Retry_Create:

	// Try to create or open the file

	if( (m_fd = open( pFileName, openFlags, 0600)) == INVALID_HANDLE_VALUE)
	{
		if( (errno == ENOENT) && (uiAccess & F_IO_CREATE_DIR))
		{
			char		szTemp[ F_PATH_MAX_SIZE];
			char		ioDirPath[ F_PATH_MAX_SIZE];

			uiAccess &= ~F_IO_CREATE_DIR;

			// Remove the file name for which we are creating the directory

			if( RC_OK( f_pathReduce( szSaveFileName, ioDirPath, szTemp)))
			{
				F_FileSystemImp	FileSystem;

				if( RC_BAD( rc = FileSystem.CreateDir( ioDirPath)))
				{
					goto Exit;
				}
				
				goto Retry_Create;
			}
		}
		
		rc = MapErrnoToFlaimErr( errno, FERR_OPENING_FILE);
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
		m_fd = INVALID_HANDLE_VALUE;
		m_bDoDirectIO = FALSE;
		m_bCanDoAsync = FALSE;
	}
	
   return( rc);
}

/****************************************************************************
Desc:		Create a file 
****************************************************************************/
RCODE F_FileHdlImp::Create(
	const char *	pIoPath,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = FERR_OK;

	flmAssert( !m_bFileOpened);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = OpenOrCreate( pIoPath, uiIoFlags, TRUE)))
	{
		goto Exit;
	}
	
	m_bFileOpened = TRUE;
	m_uiCurrentPos = 0;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_FileHdlImp::CreateUnique(
	char *				pIoPath,
	const char *		pszFileExtension,
	FLMUINT				uiIoFlags)
{
	RCODE				rc = FERR_OK;
	char *			pszTmp;
	FLMBOOL			bModext = TRUE;
	FLMUINT			uiBaseTime = 0;
	char				ucHighByte = 0;
	char				ucFileName[ F_FILENAME_SIZE];
	char *			pDirPath;
	char				szDirPath[ F_PATH_MAX_SIZE];
	char				szTmpPath[ F_PATH_MAX_SIZE];
	FLMUINT			uiCount;

	flmAssert( !m_bFileOpened);
	f_memset( ucFileName, 0, sizeof( ucFileName));
	
	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( !pIoPath || pIoPath[ 0] == '\0')
	{
		f_strcpy( szDirPath, "./");
	}
	else
	{
		f_strcpy( szDirPath, pIoPath);
	}
	
	pDirPath = szDirPath;

   // Search backwards replacing trailing spaces with NULLs.

	pszTmp = pDirPath;
	pszTmp += (f_strlen( pszTmp) - 1);
	
	while ((*pszTmp == ' ') && pszTmp >= pDirPath)
	{
		*pszTmp = 0;
		pszTmp--;
	}

	// Append a slash if one isn't already there

	if( pszTmp >= pDirPath && *pszTmp != '/')
	{
		pszTmp++;
		*pszTmp++ = '/';
	}
	else
	{
		pszTmp++;
	}
	
	*pszTmp = 0;

	if( (pszFileExtension) && (f_strlen( pszFileExtension) >= 3))
	{
		bModext = FALSE;
	}

	uiCount = 0;
	do
	{
		f_pathCreateUniqueName( &uiBaseTime,  ucFileName, pszFileExtension,
										&ucHighByte, bModext);

		f_strcpy( szTmpPath, pDirPath);
		f_pathAppend( szTmpPath, ucFileName);

		rc = Create( szTmpPath, uiIoFlags | F_IO_EXCL);

		if( rc == FERR_IO_DISK_FULL)
		{
			F_FileSystemImp	FileSystem;

			FileSystem.Delete( pDirPath);
			goto Exit;
		}
		
		if( rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PASSWORD)
		{
			goto Exit;
		}
	} while( rc != FERR_OK && (uiCount++ < MAX_CREATION_TRIES));

   // Check if the path was created

   if( uiCount >= MAX_CREATION_TRIES && rc != FERR_OK)
   {
		rc = RC_SET( FERR_IO_PATH_CREATE_FAILURE);
		goto Exit;
   }
	
	m_bFileOpened = TRUE;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

	// Created file name needs to be returned.

	f_strcpy( pIoPath, szTmpPath);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Open a file
****************************************************************************/
RCODE F_FileHdlImp::Open(
	const char *	pIoPath,
	FLMUINT			uiIoFlags)
{
	RCODE				rc = FERR_OK;

	flmAssert( !m_bFileOpened);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	// Loop on error open conditions.
	
	for( ;;)
	{
		if( RC_BAD( rc = OpenOrCreate( pIoPath, uiIoFlags, FALSE)))
		{
			if( rc != FERR_IO_TOO_MANY_OPEN_FILES)
			{
				goto Exit;
			}
			
			// If for some reason we cannot open the file, then
			// try to close some other file handle in the list.
	
			if( RC_BAD( rc = gv_FlmSysData.pFileHdlMgr->ReleaseOneAvail()))
			{
				goto Exit;
			}
			
			continue;
		}
		
		break;
	}

	m_bFileOpened = TRUE;
	m_uiCurrentPos = 0;
	m_bOpenedReadOnly = (uiIoFlags & F_IO_RDONLY) ? TRUE : FALSE;
	m_bOpenedExclusive = (uiIoFlags & F_IO_SH_DENYRW) ? TRUE : FALSE;

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Close a file
****************************************************************************/
RCODE F_FileHdlImp::Close( void)
{
	FLMBOOL	bDeleteAllowed = TRUE;
	RCODE		rc = FERR_OK;

	if( !m_bFileOpened)
	{
		goto Exit;
	}

	flmAssert( m_fd != INVALID_HANDLE_VALUE);
	close( m_fd);
	
	m_fd = INVALID_HANDLE_VALUE;
	m_bFileOpened = m_bOpenedReadOnly = m_bOpenedExclusive = FALSE;

	if( m_bDeleteOnClose)
	{
		flmAssert( m_pszIoPath);

		if( bDeleteAllowed)
		{
			F_FileSystemImp	FileSystem;
			
			FileSystem.Delete( m_pszIoPath);
		}
		
		m_bDeleteOnClose = FALSE;
	}

	if( m_pszIoPath)
	{
		f_free( &m_pszIoPath);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Make sure all file data is safely on disk
****************************************************************************/
RCODE F_FileHdlImp::Flush( void)
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
		 return( MapErrnoToFlaimErr( errno, FERR_FLUSHING_FILE));
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
			 return( MapErrnoToFlaimErr( errno, FERR_FLUSHING_FILE));
		}
	}
	
#endif
	
	return( FERR_OK);
}

/****************************************************************************
Desc:		Read from a file
****************************************************************************/
RCODE F_FileHdlImp::DirectRead(
	FLMUINT			uiReadOffset,
	FLMUINT			uiBytesToRead,	
   void *			pvBuffer,
	FLMBOOL			bBuffHasFullSectors,
	FLMUINT *		puiBytesRead)
{
	RCODE				rc = FERR_OK;
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

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	} 

	if( uiReadOffset == F_IO_CURRENT_POS)
	{
		uiReadOffset = m_uiCurrentPos;
	}

	// This loop does multiple reads (if necessary) to get all of the
	// data.  It uses aligned buffers and reads at sector offsets.

	pucDestBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{
		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if( (uiReadOffset & m_uiNotOnSectorBoundMask) ||
			 (((FLMUINT)pucDestBuffer) & m_uiNotOnSectorBoundMask) ||
			 ((uiBytesToRead & m_uiNotOnSectorBoundMask) &&
			  (!bBuffHasFullSectors)))
		{
			if( !m_pucAlignedBuff)
			{
				if( RC_BAD( rc = AllocAlignBuffer()))
				{
					goto Exit;
				}
			}
			
			pucReadBuffer = m_pucAlignedBuff;

			// Must read enough bytes to cover all of the sectors that
			// contain the data we are trying to read.  The value of
			// (uiReadOffset & m_uiNotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round that up to the next sector
			// to get the total number of bytes we are going to read.

			uiMaxBytesToRead = RoundUpToSectorMultiple( uiBytesToRead +
									  (uiReadOffset & m_uiNotOnSectorBoundMask));

			// Can't read more than the aligned buffer will hold.

			if( uiMaxBytesToRead > m_uiAlignedBuffSize)
			{
				uiMaxBytesToRead = m_uiAlignedBuffSize;
			}
		}
		else
		{
			uiMaxBytesToRead = RoundUpToSectorMultiple( uiBytesToRead);
			flmAssert( uiMaxBytesToRead >= uiBytesToRead);
			pucReadBuffer = pucDestBuffer;
		}

		bHitEOF = FALSE;

#ifdef HAVE_PREAD
		if( (iTmp = pread( m_fd, pucReadBuffer,
			uiMaxBytesToRead, GetSectorStartOffset( uiReadOffset))) == -1)
		{
			rc = MapErrnoToFlaimErr( errno, FERR_READING_FILE);
			goto Exit;
		}
#else
		if( lseek( m_fd, GetSectorStartOffset( uiReadOffset), SEEK_SET) == -1)
		{
			rc = MapErrnoToFlaimErr( errno, FERR_POSITIONING_IN_FILE);
			goto Exit;
		}
	
		if( (iTmp = read( m_fd, pucReadBuffer, uiMaxBytesToRead)) == -1)
		{
			rc = MapErrnoToFlaimErr(errno, FERR_READING_FILE);
			goto Exit;
		}
#endif
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

		if( uiReadOffset & m_uiNotOnSectorBoundMask)
		{
			pucReadBuffer += (uiReadOffset & m_uiNotOnSectorBoundMask);
			flmAssert( uiBytesRead >= m_uiBytesPerSector);
			uiBytesRead -= (uiReadOffset & m_uiNotOnSectorBoundMask);
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
		
		m_uiCurrentPos = uiReadOffset + uiBytesRead;

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
			rc = RC_SET( FERR_IO_END_OF_FILE);
			break;
		}

		pucDestBuffer += uiBytesRead;
		uiReadOffset += uiBytesRead;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Read from a file
****************************************************************************/
RCODE F_FileHdlImp::Read(
	FLMUINT			uiReadOffset,
	FLMUINT			uiBytesToRead,	
   void *			pvBuffer,
   FLMUINT *		puiBytesRead)
{
	RCODE				rc = FERR_OK;
	FLMINT      	iBytesRead;
	
	flmAssert( m_bFileOpened);

	if( m_bDoDirectIO)
	{
		rc = DirectRead( uiReadOffset, uiBytesToRead, 
			pvBuffer, FALSE, puiBytesRead);
		goto Exit;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( uiReadOffset == F_IO_CURRENT_POS)
	{
		uiReadOffset = m_uiCurrentPos;
	}
	
#ifdef HAVE_PREAD
	if( (iBytesRead = pread( m_fd, pvBuffer, uiBytesToRead, uiReadOffset)) == -1)
	{
		rc = MapErrnoToFlaimErr(errno, FERR_READING_FILE);
		goto Exit;
	}
#else
	if( m_uiCurrentPos != uiReadOffset)
	{
		if( lseek( m_fd, uiReadOffset, SEEK_SET) == -1)
		{
			rc = MapErrnoToFlaimErr( errno, FERR_POSITIONING_IN_FILE);
			goto Exit;
		}
	}

	if( (iBytesRead = read( m_fd, pvBuffer, uiBytesToRead)) == -1)
	{
		rc = MapErrnoToFlaimErr(errno, FERR_READING_FILE);
		goto Exit;
	}
#endif

	if( puiBytesRead)
	{
		*puiBytesRead = (FLMUINT)iBytesRead;
	}
	
	m_uiCurrentPos = uiReadOffset + (FLMUINT)iBytesRead;
	
	if( (FLMUINT)iBytesRead < uiBytesToRead)
	{
		rc = RC_SET( FERR_IO_END_OF_FILE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
Note:	This function assumes that the pvBuffer that is passed in is
		a multiple of a the sector size.
****************************************************************************/
RCODE F_FileHdlImp::SectorRead(
	FLMUINT			uiReadOffset,
	FLMUINT			uiBytesToRead,
   void *			pvBuffer,
   FLMUINT *		puiBytesRead)
{
	if( m_bDoDirectIO)
	{
		return( DirectRead( uiReadOffset, uiBytesToRead, 
			pvBuffer, TRUE, puiBytesRead));
	}
	else
	{
		return( Read( uiReadOffset, uiBytesToRead, pvBuffer, puiBytesRead));
	}
}

/****************************************************************************
Desc:		Sets current position of file.
Note:		F_IO_SEEK_END is not supported.
****************************************************************************/
RCODE F_FileHdlImp::Seek(
	FLMUINT			uiOffset,
	FLMINT			iWhence,
	FLMUINT *		puiNewOffset)
{
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	switch( iWhence)
	{
		case F_IO_SEEK_CUR:
		{
			m_uiCurrentPos += uiOffset;
			break;
		}
		
		case F_IO_SEEK_SET:
		{
			m_uiCurrentPos = uiOffset;
			break;
		}
		
		case F_IO_SEEK_END:
		{
			if( RC_BAD( rc = Size( &m_uiCurrentPos)))
			{
				goto Exit;
			}
			break;
		}
		
		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	
#ifndef HAVE_PREAD
	if( lseek( m_fd, m_uiCurrentPos, SEEK_SET) == -1)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_POSITIONING_IN_FILE);
		goto Exit;
	}
#endif

	*puiNewOffset = m_uiCurrentPos;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Return the size of the file
****************************************************************************/
RCODE F_FileHdlImp::Size(
	FLMUINT *		puiSize)
{
	RCODE				rc = FERR_OK;
   struct stat 	statBuf;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

   if( fstat( m_fd, &statBuf) == -1)
   {
      rc = MapErrnoToFlaimErr( errno, FERR_GETTING_FILE_SIZE);
		goto Exit;
   }
	
	// VISIT: We need to change the file handle interface to return 64-bit
	// sizes and offsets.  When this is done, remove the FLMUINT typecast below.
	
	*puiSize = (FLMUINT)statBuf.st_size;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Returns m_uiCurrentPos
****************************************************************************/
RCODE F_FileHdlImp::Tell(
	FLMUINT *	puiOffset)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	*puiOffset = m_uiCurrentPos;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Truncate the file to the indicated size
WARNING: Direct IO methods are calling this method.  Make sure that all
			changes to this method work in direct IO mode.
****************************************************************************/
RCODE F_FileHdlImp::Truncate(
	FLMUINT		uiSize)
{
	RCODE			rc = FERR_OK;

	flmAssert( m_bFileOpened);

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( ftruncate( m_fd, uiSize) == -1)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_TRUNCATING_FILE);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Write to a file
****************************************************************************/
RCODE F_FileHdlImp::Write(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT *		puiBytesWrittenRV)
{
	RCODE				rc = FERR_OK;
	FLMINT      	iBytesWritten = 0;
	
	flmAssert( m_bFileOpened);

	if( m_bDoDirectIO)
	{
		rc = DirectWrite( uiWriteOffset, uiBytesToWrite, pvBuffer, 
			uiBytesToWrite, NULL, puiBytesWrittenRV, FALSE, TRUE);
			
		goto Exit;
	}
	
	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}
	
	if( uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}

#ifdef HAVE_PREAD
	if( (iBytesWritten = pwrite(m_fd, pvBuffer, uiBytesToWrite,
											uiWriteOffset)) == -1)
	{
		rc = MapErrnoToFlaimErr( errno, FERR_WRITING_FILE);
		goto Exit;
	}
#else
	if( m_uiCurrentPos != uiWriteOffset)
	{
		if( lseek(m_fd, uiWriteOffset, SEEK_SET) == -1)
		{
			rc = MapErrnoToFlaimErr( errno, FERR_POSITIONING_IN_FILE);
			goto Exit;
		}
	}
	
  	if( (iBytesWritten = write( m_fd, pvBuffer, uiBytesToWrite)) == -1)
	{
		rc = MapErrnoToFlaimErr(errno, FERR_WRITING_FILE);
		goto Exit;
	}
#endif

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = (FLMUINT)iBytesWritten;
	}
	
	m_uiCurrentPos = uiWriteOffset + (FLMUINT)iBytesWritten;
	
	if( (FLMUINT)iBytesWritten < uiBytesToWrite)
	{
		rc = RC_SET( FERR_IO_DISK_FULL);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Allocate an aligned buffer.
****************************************************************************/
RCODE F_FileHdlImp::AllocAlignBuffer( void)
{
#if !defined( FLM_LINUX) && !defined( FLM_SOLARIS)
	return( RC_SET( FERR_NOT_IMPLEMENTED));
#else
	RCODE	rc = FERR_OK;
	
	if( m_pucAlignedBuff)
	{
		goto Exit;
	}

	// Allocate at least 64K - this will handle most read and write
	// operations and will also be a multiple of the sector size most of
	// the time.  The calculation below rounds it up to the next sector
	// boundary if it is not already on one.

	m_uiAlignedBuffSize = RoundUpToSectorMultiple( 64 * 1024);
	
	if( (m_pucAlignedBuff = (FLMBYTE *)memalign( 
		sysconf(_SC_PAGESIZE), m_uiAlignedBuffSize)) == NULL) 
	{
		m_uiAlignedBuffSize = 0;
		rc = MapErrnoToFlaimErr( errno, FERR_MEM);
		goto Exit;
	}
	
Exit:

	return( rc);
#endif
}

/****************************************************************************
Desc:
Note:	This routine assumes that the size of pvBuffer is a multiple of
		sector size and can be used to write out full sectors.  Even if
		uiBytesToWrite does not account for full sectors, data from the
		buffer will still be written out - a partial sector on disk will
		not be preserved.
****************************************************************************/
RCODE F_FileHdlImp::DirectWrite(
	FLMUINT			uiWriteOffset,
	FLMUINT			uiBytesToWrite,
	const void *	pvBuffer,
	FLMUINT,
	F_IOBuffer *	pBufferObj,
	FLMUINT *		puiBytesWrittenRV,
	FLMBOOL			bBuffHasFullSectors,
	FLMBOOL			bZeroFill)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesRead;
	FLMUINT			uiMaxBytesToWrite;
	FLMUINT			uiBytesBeingOutput;
	FLMBYTE *		pucWriteBuffer;
	FLMBYTE *		pucSrcBuffer;
	FLMBOOL			bDoAsync = (pBufferObj != NULL) 
										? TRUE 
										: FALSE;
	FLMBOOL			bDidAsync = FALSE;
	FLMUINT			uiLastWriteOffset;
	FLMUINT			uiLastWriteSize;
	
	flmAssert( m_bFileOpened);

#ifdef FLM_DEBUG
	if( bDoAsync)
	{
		flmAssert( m_bCanDoAsync);
	}
#else
	(void)bDoAsync;
#endif

	if( puiBytesWrittenRV)
	{
		*puiBytesWrittenRV = 0;
	}

	if( RC_BAD( rc = GET_FS_ERROR()))
	{
		goto Exit;
	}

	if( uiWriteOffset == F_IO_CURRENT_POS)
	{
		uiWriteOffset = m_uiCurrentPos;
	}

	// This loop is for direct IO - must make sure we use
	// aligned buffers.
	
	pucSrcBuffer = (FLMBYTE *)pvBuffer;
	for (;;)
	{
		// See if we are using an aligned buffer.  If not, allocate
		// one (if not already allocated), and use it.

		if( (uiWriteOffset & m_uiNotOnSectorBoundMask) ||
			 (((FLMUINT)pucSrcBuffer) & m_uiNotOnSectorBoundMask) ||
			 ((uiBytesToWrite & m_uiNotOnSectorBoundMask) && !bBuffHasFullSectors))
		{
			// Cannot be using a temporary write buffer if we are doing
			// asynchronous writes!

			flmAssert( !bDoAsync || !m_bCanDoAsync);
			
			if( !m_pucAlignedBuff)
			{
				if( RC_BAD( rc = AllocAlignBuffer()))
				{
					goto Exit;
				}
			}
			pucWriteBuffer = m_pucAlignedBuff;

			// Must write enough bytes to cover all of the sectors that
			// contain the data we are trying to write out.  The value of
			// (uiWriteOffset & m_uiNotOnSectorBoundMask) will give us the
			// number of additional bytes that are in the sector prior to
			// the read offset.  We then round to the next sector to get the
			// total number of bytes we are going to write.

			uiMaxBytesToWrite = RoundUpToSectorMultiple( uiBytesToWrite +
									  (uiWriteOffset & m_uiNotOnSectorBoundMask));

			// Can't write more than the aligned buffer will hold.

			if( uiMaxBytesToWrite > m_uiAlignedBuffSize)
			{
				uiMaxBytesToWrite = m_uiAlignedBuffSize;
				uiBytesBeingOutput = uiMaxBytesToWrite -
										(uiWriteOffset & m_uiNotOnSectorBoundMask);
			}
			else
			{
				uiBytesBeingOutput = uiBytesToWrite;
			}

			// If the write offset is not on a sector boundary, or if
			// we are writing a partial sector, we must read the
			// sector into the buffer.

			if( (uiWriteOffset & m_uiNotOnSectorBoundMask) ||
				(uiBytesBeingOutput < m_uiBytesPerSector && !bBuffHasFullSectors))
			{
				// Read the first sector that is to be written out.
				// Read one sector's worth of data - so that we will
				// preserve what is already in the sector before
				// writing it back out again.

				if( RC_BAD( rc = Read( GetSectorStartOffset( uiWriteOffset),
					m_uiBytesPerSector, pucWriteBuffer, &uiBytesRead)))
				{
					if( rc != FERR_IO_END_OF_FILE)
					{
						goto Exit;
					}

					rc = FERR_OK;
					f_memset( &pucWriteBuffer[ uiBytesRead], 0, 
						m_uiBytesPerSector - uiBytesRead);
				}
			}

			// Finally, copy the data from the source buffer into the
			// write buffer.

			f_memcpy( &pucWriteBuffer[ uiWriteOffset & m_uiNotOnSectorBoundMask],
								pucSrcBuffer, uiBytesBeingOutput);
		}
		else
		{
			uiMaxBytesToWrite = RoundUpToSectorMultiple( uiBytesToWrite);
			uiBytesBeingOutput = uiBytesToWrite;
			pucWriteBuffer = pucSrcBuffer;
			
			if( bZeroFill && uiMaxBytesToWrite > uiBytesToWrite)
			{
				f_memset( &pucWriteBuffer[ uiBytesToWrite], 0,
							uiMaxBytesToWrite - uiBytesToWrite);
			}
		}

		// Position the file to the nearest sector below the write offset.

		uiLastWriteOffset = GetSectorStartOffset( uiWriteOffset);
		uiLastWriteSize = uiMaxBytesToWrite;
		
		if( !m_bCanDoAsync || !pBufferObj)
		{
			FLMINT		iBytesWritten;
			
#ifdef HAVE_PREAD
			if( (iBytesWritten = pwrite( m_fd, 
				pucWriteBuffer, uiMaxBytesToWrite, uiLastWriteOffset)) == -1)
			{
				rc = MapErrnoToFlaimErr( errno, FERR_WRITING_FILE);
				goto Exit;
			}
#else
			if( lseek( m_fd, uiLastWriteOffset, SEEK_SET) == -1)
			{
				rc = MapErrnoToFlaimErr( errno, FERR_POSITIONING_IN_FILE);
				goto Exit;
			}

			if( (iBytesWritten = write( m_fd, pucWriteBuffer, uiMaxBytesToWrite)) == -1)
			{
				rc = MapErrnoToFlaimErr( errno, FERR_WRITING_FILE);
				goto Exit;
			}
#endif

			if( (FLMUINT)iBytesWritten < uiMaxBytesToWrite)
			{
				rc = RC_SET( FERR_IO_DISK_FULL);
				goto Exit;
			}
		}
		else
		{
			flmAssert( 0);
		}

		uiBytesToWrite -= uiBytesBeingOutput;
		if( puiBytesWrittenRV)
		{
			(*puiBytesWrittenRV) += uiBytesBeingOutput;
		}
		
		m_uiCurrentPos = uiWriteOffset + uiBytesBeingOutput;
		
		if( !uiBytesToWrite)
		{
			break;
		}

		flmAssert( !pBufferObj);

		pucSrcBuffer += uiBytesBeingOutput;
		uiWriteOffset += uiBytesBeingOutput;
	}

Exit:

	if( !bDidAsync && pBufferObj)
	{
		pBufferObj->notifyComplete( rc);
	}

	return( rc);
}

/****************************************************************************
Desc:		Returns flag indicating whether or not we can do async writes.
****************************************************************************/
FLMBOOL F_FileHdlImp::CanDoAsync( void)
{
	return( m_bCanDoAsync);
}

/****************************************************************************
Desc:		Attempts to lock byte 0 of the file.  This method is used to
			lock byte 0 of the .lck file to ensure that only one process
			has access to a database.
****************************************************************************/
RCODE F_FileHdlImp::Lock( void)
{
	RCODE				rc = FERR_OK;
	struct flock   LockStruct;

	// Lock first byte in file

	f_memset( &LockStruct, 0, sizeof( LockStruct));
	LockStruct.l_type   = F_WRLCK;
	LockStruct.l_whence = SEEK_SET;
	LockStruct.l_start  = 0;
	LockStruct.l_len    = 1;

	if( fcntl( m_fd, F_SETLK, &LockStruct) == -1)
	{
		rc = RC_SET( FERR_IO_FILE_LOCK_ERR);
		goto Exit;
	} 

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Attempts to unlock byte 0 of the file.
****************************************************************************/
RCODE F_FileHdlImp::Unlock( void)
{
	struct flock   LockStruct;
	RCODE				rc = FERR_OK;

	// Lock first byte in file

	f_memset( &LockStruct, 0, sizeof( LockStruct));
	LockStruct.l_type   = F_UNLCK;
	LockStruct.l_whence = SEEK_SET;
	LockStruct.l_start  = 0;
	LockStruct.l_len    = 1;

	if( fcntl( m_fd, F_SETLK, &LockStruct) == -1)
	{
		rc = RC_SET( FERR_IO_FILE_UNLOCK_ERR);
		goto Exit;
	} 

Exit:

	return( rc);
}

/***************************************************************************
Desc:		Determines the kernel version of a linux system
***************************************************************************/
#ifdef FLM_LINUX
void flmGetLinuxKernelVersion(
	FLMUINT *		puiMajor,
	FLMUINT *		puiMinor,
	FLMUINT *		puiRevision)
{
	int			fd = INVALID_HANDLE_VALUE;
	int			iBytesRead;
	char			szBuffer [80];
	char *		pszVer;
	FLMUINT		uiMajorVer = 0;
	FLMUINT		uiMinorVer = 0;
	FLMUINT		uiRevision = 0;

	if( (fd = open( "/proc/version", O_RDONLY, 0600)) == INVALID_HANDLE_VALUE)
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

	if( fd != INVALID_HANDLE_VALUE)
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
FLMUINT flmGetLinuxMaxFileSize(
	FLMUINT		uiSizeofFLMUINT)
{
	FLMUINT		uiMaxFileSize = MAX_FILE_SIZE_VER40;

	// Determine if we are on 32 or 64 bit platform.  If on 64 bit, we can
	// support the larger file size.  Otherwise, we have to open up the
	// /proc/version file and see if we are linux version 2.4 or greater.

	// NOTE: The only reason we are passing uiSizeofFLMUINT in as a parameter is
	// to spoof the compiler so it won't give us a warning.  If we did
	// if (sizeof( FLMUINT) > 4) the compiler would give us a warning that the
	// condition is either always TRUE or FALSE (depending on whether we are
	// on a 32 bit or 64 bit platform.

	if( uiSizeofFLMUINT > 4)
	{
		uiMaxFileSize = F_MAXIMUM_FILE_SIZE;
		goto Exit;
	}
	
	flmAssert( gv_FlmSysData.uiLinuxMajorVer);
	
	// Is version 2.4 or greater?

	if( gv_FlmSysData.uiLinuxMajorVer > 2 || 
		 (gv_FlmSysData.uiLinuxMajorVer == 2 && 
		  gv_FlmSysData.uiLinuxMinorVer >= 4))
	{
		uiMaxFileSize = F_MAXIMUM_FILE_SIZE;
	}
	
Exit:

	return( uiMaxFileSize);
}
#endif

/****************************************************************************
Desc: This routine gets the block size for the file system a file belongs to.
****************************************************************************/
FLMUINT flmGetFSBlockSize(
	const char *	pszFileName)
{
#define DEFAULT_FS_BSIZE      1024
	FLMUINT			uiFSBlkSize = DEFAULT_FS_BSIZE;
	char				szTmpBuf[ F_PATH_MAX_SIZE];
	char *			pszTmp;
	const char *	pszDir;
	
	f_strcpy( szTmpBuf, pszFileName);
	pszTmp = szTmpBuf + f_strlen( szTmpBuf) - 1; 

	while( pszTmp != &szTmpBuf[ 0] && *pszTmp != '/')
	{
		pszTmp--;
	}
	
	if( *pszTmp == '/')
	{
		if( pszTmp == &szTmpBuf[ 0])
		{
			pszTmp++;
		}
		
		*pszTmp = 0;
		pszDir = szTmpBuf;
	}
	else
	{
		pszDir = ".";
	}

#if defined( FLM_SOLARIS)

	struct statvfs statfsbuf;
	
	if( statvfs( pszDir, &statfsbuf) == 0)
	{
		uiFSBlkSize = (FLMUINT)statfsbuf.f_bsize;
	}
	
#elif defined( FLM_LINUX)

	struct statfs statfsbuf;
	
	if( statfs( pszDir, &statfsbuf) == 0)
	{
		uiFSBlkSize = (FLMUINT)statfsbuf.f_bsize;
	}
	
#endif

	return( uiFSBlkSize);
}

#endif // FLM_UNIX

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WATCOM_NLM)
int fposixDummy(void)
{
	return( 0);
}
#endif
