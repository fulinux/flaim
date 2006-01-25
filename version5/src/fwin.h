//------------------------------------------------------------------------------
// Desc:	This include file contains the class definitions for FLAIM's WIN
//			FileHdl classes.
//
// Tabs:	3
//
//		Copyright (c) 1997-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fwin.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FWIN_H
#define FWIN_H

#ifdef FLM_WIN

RCODE MapWinErrorToFlaim(
	DWORD		udErrCode,
	RCODE		defaultRc);

// Forward references

class F_FileHdlPage;

/*===========================================================================
Class:	F_FileHdl
Desc:		The F_FileHdl class provides support for basic IO operations
			using the WIN I/O calls.
===========================================================================*/
class F_FileHdl : public IF_FileHdl, public XF_Base
{
public:
	F_FileHdl();								// F_FileHdl Constructor

	virtual ~F_FileHdl();					// F_FileHdl Destructor - free/close this
													// file handle

	// BEGINNING OF FUNCTIONS THAT MUST BE IMPLEMENTED ON ALL PLATFORMS

	RCODE XFLMAPI Close( void);			// Close a file - The destructor will call this
													// This is used to obtain an error code.

	RCODE XFLMAPI Create(					// Create a new file.
		const char *	pszFileName,		// File to be created
		FLMUINT			uiIoFlags);			// Access and Mode Flags

	RCODE XFLMAPI CreateUnique(			// Create a new file (with a unique file name).
		const char *	pszDirName,			// Directory where the file is to be created
		const char *	pszFileExtension,	// Extension to be used on the new file.
		FLMUINT			uiIoFlags);			// Access and Mode Flags

	RCODE XFLMAPI Open(						// Initiates access to an existing file.
		const char *	pszFileName,		// File to be opened
		FLMUINT			uiIoFlags);			// Access and Mode Flags

	RCODE XFLMAPI Flush( void);			// Flushes a file's buffers to disk

	RCODE XFLMAPI Read(						// Reads a buffer of data from a file
		FLMUINT64		ui64Offset,			// Offset to being reading at.
		FLMUINT			uiLength,			// Number of bytes to read
		void *			pvBuffer,			// Buffer to place read bytes into
		FLMUINT *		puiBytesRead);		// [out] number of bytes read

	RCODE XFLMAPI Seek(						// Moves the current position in the file
		FLMUINT64		ui64Offset,			// Offset to seek to
		FLMINT			iWhence,				// Location to apply sdwOffset to.
		FLMUINT64 *		pui64NewOffset);	// [out] new file offset

	RCODE XFLMAPI Size(						// Returns to size of the open file.
		FLMUINT64 *		pui64Size);			// [out] size of the file

	RCODE XFLMAPI Tell(						// Returns to current position of the file
													// pointer in the open file.
		FLMUINT64 *		pui64Offset);		// [out] current file position

	RCODE XFLMAPI Truncate(					// Decreases the size of a file.
		FLMUINT64		ui64Size);			// Size to truncate the file to.

	RCODE XFLMAPI Write(						// Writes a buffer of data to a file.
		FLMUINT64		ui64Offset,			// Offset to seek to.
		FLMUINT			uiLength,			// Number of bytes to write.
		const void *	pvBuffer,			// Buffer that contains bytes to be written
		FLMUINT *		puiBytesWritten);	// Number of bytes written.

	// Some I/O subsystems (such as direct IO) can only read and write sectors
	// (512 byte chunks).  If uiOffset is not on a sector boundary or
	// uiLength is not an exact multiple of a sector size, the I/O system
	// would have to try to read or write a partial sector - something that
	// requires extra overhead, particularly for write operations - because
	// in order to write a partial sector, the I/O subsystem first has to
	// read the sector in to memory before writing it out in order to
	// preserve the part of the sector that was not being written to.

	// The SectorRead and SectorWrite routines are provided to allow
	// the caller to tell the I/O subsystem that it is OK to do full
	// sector reads or writes if it needs to, because pvBuffer is
	// guaranteed to be a multiple of 512 bytes big.  If the I/O
	// subsystem can only do sector reads and writes, it can use the
	// extra buffer space in pvBuffer.  When a program calls SectorWrite
	// it is also telling the I/O subsystem that it does not need to
	// read a partially written sector from disk before writing it out.
	// It will be OK to write whatever data is in the pvBuffer to fill out
	// the sector.

	FINLINE RCODE XFLMAPI SectorRead(	// Allows sector reads to be done.
		FLMUINT64		ui64ReadOffset,	// Offset to being reading at.
		FLMUINT			uiBytesToRead,		// Number of bytes to read
		void *			pvBuffer,			// Buffer to place read bytes into
		FLMUINT *		puiBytesReadRV)	// [out] number of bytes read
	{
		if (m_bDoDirectIO)
		{
			return( DirectRead( ui64ReadOffset, uiBytesToRead,
					pvBuffer, TRUE, puiBytesReadRV));
		}
		else
		{
			return( Read( ui64ReadOffset, uiBytesToRead, pvBuffer, puiBytesReadRV));
		}
	}

	FINLINE RCODE XFLMAPI SectorWrite(		// Allows sector writes to be done.
		FLMUINT64		ui64WriteOffset,		// Offset to seek to.
		FLMUINT			uiBytesToWrite,		// Number of bytes to write.
		const void *	pvBuffer,				// Buffer that contains bytes to be written
		FLMUINT			uiBufferSize,			// Actual buffer size.
		void *			pvBufferObj,			// Buffer object for async write
		FLMUINT *		puiBytesWrittenRV,	// Number of bytes written.
		FLMBOOL			bZeroFill = TRUE)		// Zero fill the buffer?
	{
		uiBufferSize = uiBufferSize;	// Parameter is not used.
		if (m_bDoDirectIO)
		{
			return( DirectWrite( ui64WriteOffset, uiBytesToWrite,
					pvBuffer, (F_IOBuffer *)pvBufferObj, TRUE,
					bZeroFill, puiBytesWrittenRV));
		}
		else
		{
			flmAssert( pvBufferObj == NULL);
			return( Write( ui64WriteOffset, uiBytesToWrite, pvBuffer,
								puiBytesWrittenRV));
		}
	}

	FINLINE FLMBOOL XFLMAPI CanDoAsync(	// Return whether or not we can do async
		void)										// writes.
	{
		return( m_bCanDoAsync);
	}

	FINLINE void XFLMAPI setExtendSize(
		FLMUINT		uiExtendSize)
	{
		m_uiExtendSize = uiExtendSize;
	}

	FINLINE void XFLMAPI setMaxAutoExtendSize(
		FLMUINT		uiMaxAutoExtendSize)
	{
		m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
	}

	// METHODS NEEDED TO INTERACT WITH FILE HANDLE MANAGER

	FINLINE void setupFileHdl(
		FLMUINT			uiFileId,
		FLMBOOL			bDeleteOnRelease)
	{
		m_uiFileId = uiFileId;
		m_bDeleteOnRelease = bDeleteOnRelease;
	}

	FINLINE FLMUINT getFileId( void)
	{
		return m_uiFileId;
	}

	// END OF FUNCTIONS THAT MUST BE DEFINED FOR ALL PLATFORMS

	FINLINE FLMUINT GetSectorSize( void)
	{
		return( m_uiBytesPerSector);
	}

	FINLINE void SetBlockSize(
		FLMUINT	uiBlockSize)
	{
		m_uiBlockSize = uiBlockSize;
	}

	FINLINE HANDLE getFileHandle( void)
	{
		return m_FileHandle;
	}

private:

	RCODE OpenOrCreate(							// Open or create a file
		const char *	pszFileName,			// Name of file to open or create.
	   FLMUINT			uiAccess,				// Access flags
		FLMBOOL			bCreateFlag);			// Create flag

	RCODE AllocAlignBuffer( void);			// Allocate an aligned buffer.

	RCODE DoOneRead(
		FLMUINT64		ui64Offset,				// Offset being reading at.
		FLMUINT			uiLength,				// Number of bytes to read
		void *			pvBuffer,				// Buffer to place read bytes into
		FLMUINT *		puiBytesRead);			// [out] number of bytes read

	RCODE DirectRead(								// Reads a buffer of data from a file
		FLMUINT64		uiOffset,				// Offset being reading at.
		FLMUINT			uiLength,				// Number of bytes to read
		void *			pvBuffer,				// Buffer to place read bytes into
		FLMBOOL			bBuffHasFullSectors,	// Buffer is sector aligned.
		FLMUINT *		puiBytesRead);			// [out] number of bytes read

	RCODE DirectWrite(							// Writes a buffer of data from a file
		FLMUINT64		uiOffset,				// Offset being written to.
		FLMUINT			uiLength,				// Number of bytes to write
		const void *	pvBuffer,				// Buffer to write from.
		F_IOBuffer *	pBufferObj,				// Buffer object for async writes
		FLMBOOL			bBuffHasFullSectors,	// Buffer is sector aligned.
		FLMBOOL			bZeroFill,
		FLMUINT *		puiBytesWritten);		// [out] number of bytes written

	FINLINE FLMUINT64 RoundToNextSector(
		FLMUINT64		ui64Bytes)
	{
		return( (ui64Bytes + m_ui64NotOnSectorBoundMask) & 
						m_ui64GetSectorBoundMask);
	}

	FINLINE FLMUINT64 TruncateToPrevSector(
		FLMUINT64		ui64Offset)
	{
		return( ui64Offset & m_ui64GetSectorBoundMask);
	}

	RCODE extendFile(
		FLMUINT64		ui64EndOfLastWrite,
		FLMUINT			uiMaxBytesToExtend,
		FLMBOOL			bFlush);

	// The following are for every platform

	F_FileHdl *			m_pNext;					// Next file handle in list
	F_FileHdl *			m_pPrev;					// Prev file handle in list
	FLMBOOL				m_bInList;				// Is this file handle in a list?
	FLMBOOL				m_bFileOpened;			// Is the file currently opened/closed.
	FLMUINT				m_uiAvailTime;			// Time when placed in avail list.
	FLMUINT				m_uiFileId;				// FFILE Unique File Id
	FLMBOOL				m_bDeleteOnRelease;	// Delete this file when it is released.
	FLMBOOL				m_bOpenedReadOnly;	// Opened the file read only
	FLMBOOL				m_bOpenedExclusive;	// Opened the file in exclusive mode
	char *				m_pszFileName;			// File name for this FileHdl

	// The following are for windows platform

	HANDLE				m_FileHandle;			// WIN file handle
	FLMUINT				m_uiBlockSize;			// Block size, if known.
	FLMUINT				m_uiBytesPerSector;	// Bytes per sector for this volume.
	FLMUINT64			m_ui64NotOnSectorBoundMask;
	FLMUINT64			m_ui64GetSectorBoundMask;
	FLMBOOL				m_bDoDirectIO;			// TRUE = do direct file I/O
	FLMUINT				m_uiExtendSize;		// Size to extend by if in direct mode.
	FLMUINT				m_uiMaxAutoExtendSize;
														// Don't do additional extending
														// once file reaches this size.
	FLMBYTE *			m_pucAlignedBuff;
														// Buffer that is aligned for doing
														// direct IO.
	FLMUINT				m_uiAlignedBuffSize;
														// Size of aligned buffer.
	FLMUINT64			m_ui64CurrentPos;		// Current position in file
	FLMBOOL				m_bCanDoAsync;			// Is this handle set up to do ASYNC?
	OVERLAPPED			m_Overlapped;			// Used when NOT doing async.

friend class F_FileHdlPage;
friend class F_FileHdlMgr;

};

#endif		// #ifdef FLM_WIN

#endif 		// #ifndef FWIN_H
