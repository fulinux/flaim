//------------------------------------------------------------------------------
// Desc:	This include file contains the class definitions for FLAIM's POSIX
//			FileHdl classes.
//
// Tabs:	3
//
//		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fposix.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FPOSIX_H
#define FPOSIX_H

#ifdef FLM_UNIX

class F_FileHdl : public IF_FileHdl, public XF_Base
{
public:

	F_FileHdl();

	~F_FileHdl();

	RCODE XFLMAPI Setup(							
		FLMUINT				uiFileId);

	RCODE XFLMAPI Close( void);
												
	RCODE XFLMAPI Create(
		const char *		pszFileName,
		FLMUINT				uiIoFlags);

	RCODE XFLMAPI CreateUnique(
		const char *		pszDirName,
		const char *		pszFileExtension,
		FLMUINT				uiIoFlags);

	RCODE XFLMAPI Open(
		const char *		pszFileName,
		FLMUINT				uiIoFlags);

	RCODE XFLMAPI Flush( void);

	RCODE XFLMAPI Read(
		FLMUINT64			ui64Offset,
		FLMUINT				uiLength,
		void *				pvBuffer,
		FLMUINT *			puiBytesRead);

	RCODE XFLMAPI Seek(
		FLMUINT64			ui64Offset,
		FLMINT				iWhence,
		FLMUINT64 *			pui64NewOffset);

	RCODE XFLMAPI Size(
		FLMUINT64 *			pui64Size);

	RCODE XFLMAPI Tell(
		FLMUINT64 *			pui64Offset);

	RCODE XFLMAPI Truncate(
		FLMUINT64			ui64Size);

	RCODE XFLMAPI Write(
		FLMUINT64			ui64Offset,
		FLMUINT				uiLength,
		const void *		pvBuffer,
		FLMUINT *			puiBytesWritten);

	// Some I/O subsystems (such as direct IO) can only read and 
	// write sectors.  If uiOffset is not on a sector boundary or
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

	RCODE XFLMAPI SectorRead(
		FLMUINT64			ui64ReadOffset,
		FLMUINT				uiBytesToRead,
		void *				pvBuffer,
		FLMUINT *			puiBytesReadRV);

	RCODE XFLMAPI SectorWrite(
		FLMUINT64			ui64WriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT				uiBufferSize,
		void *				pvBufferObj,
		FLMUINT *			puiBytesWrittenRV,
		FLMBOOL				bZeroFill = TRUE)
	{
		if( m_bDoDirectIO)
		{
			return( DirectWrite( ui64WriteOffset, uiBytesToWrite, 
				pvBuffer, uiBufferSize, (F_IOBuffer *)pvBufferObj, 
				puiBytesWrittenRV, TRUE, bZeroFill));
		}
		else
		{
			return( Write( ui64WriteOffset, uiBytesToWrite, 
				pvBuffer, puiBytesWrittenRV));
		}
	}

	FLMBOOL XFLMAPI CanDoAsync( void);

	FINLINE FLMBOOL XFLMAPI UsingDirectIo( void)
	{
		return( m_bDoDirectIO);
	}

	FINLINE void XFLMAPI setExtendSize(
		FLMUINT				uiExtendSize)
	{
		m_uiExtendSize = uiExtendSize;
	}

	FINLINE void XFLMAPI setMaxAutoExtendSize(
		FLMUINT				uiMaxAutoExtendSize)
	{
		m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
	}

	RCODE XFLMAPI Lock( void);

	RCODE XFLMAPI Unlock( void);

	FINLINE void XFLMAPI SetBlockSize(
		FLMUINT				uiBlockSize)
	{
		m_uiBlockSize = uiBlockSize;
	}
	
	FINLINE FLMUINT XFLMAPI GetSectorSize( void)
	{
		return( m_uiBytesPerSector);
	}

	FINLINE void setupFileHdl(
		FLMUINT				uiFileId,
		FLMBOOL				bDeleteOnRelease)
	{
		m_uiFileId = uiFileId;
		m_bDeleteOnRelease = bDeleteOnRelease;
	}

	FINLINE FLMUINT getFileId( void)
	{
		return m_uiFileId;
	}

private:

	RCODE OpenOrCreate(
		const char *		pszFileName,
		FLMUINT				uiAccess,
		FLMBOOL				bCreateFlag);

	FINLINE FLMUINT64 RoundUpToSectorMultiple(
		FLMUINT64			ui64Bytes)
	{
		return( (ui64Bytes + m_ui64NotOnSectorBoundMask) &
					m_ui64GetSectorBoundMask);
	}

	FINLINE FLMUINT64 GetSectorStartOffset(
		FLMUINT64			ui64Offset)
	{
		return( ui64Offset & m_ui64GetSectorBoundMask);
	}

	RCODE DirectRead(
		FLMUINT64			ui64ReadOffset,
		FLMUINT				uiBytesToRead,	
   	void *				pvBuffer,
		FLMBOOL				bBuffHasFullSectors,
		FLMUINT *			puiBytesRead);

	RCODE DirectWrite(
		FLMUINT64			ui64WriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT				uiBufferSize,
		F_IOBuffer *		pBufferObj,
		FLMUINT *			puiBytesWrittenRV,
		FLMBOOL				bBuffHasFullSectors,
		FLMBOOL				bZeroFill);
	
	RCODE AllocAlignBuffer( void);
	
	// The following are for every platform.

	F_FileHdl *				m_pNext;					// Next file handle in list
	F_FileHdl *				m_pPrev;					// Prev file handle in list
	FLMBOOL					m_bInList;				// Is this file handle in a list?
	FLMBOOL					m_bFileOpened;			// Is the file currently opened/closed.
	FLMUINT					m_uiAvailTime;			// Time when placed in avail list.
	FLMUINT					m_uiFileId;				// FFILE Unique File Id
	FLMBOOL					m_bDeleteOnRelease;	// Delete this file when it is released.
	FLMBOOL					m_bOpenedReadOnly;	// Opened the file read only
	FLMBOOL					m_bOpenedExclusive;	// Opened the file in exclusive mode
	char *					m_pszFileName;			// File name for this FileHdl
	
	// Specific to this platform
	
	int				   	m_fd;
	FLMUINT					m_uiBlockSize;
	FLMUINT					m_uiBytesPerSector;
	FLMUINT64				m_ui64NotOnSectorBoundMask;
	FLMUINT64				m_ui64GetSectorBoundMask;
	FLMUINT64				m_ui64CurrentPos;
	FLMUINT					m_uiExtendSize;
	FLMUINT					m_uiMaxAutoExtendSize;
	FLMBOOL					m_bCanDoAsync;
	FLMBOOL					m_bDoDirectIO;
	FLMBYTE *				m_pucAlignedBuff;
	FLMUINT					m_uiAlignedBuffSize;

	friend class F_FileHdlPage;
	friend class F_FileHdlMgr;
};

FLMUINT flmGetFSBlockSize(
	FLMBYTE *			pszFileName);
	
#if defined( FLM_LINUX)
	void flmGetLinuxKernelVersion(
		FLMUINT *		puiMajor,
		FLMUINT *		puiMinor,
		FLMUINT *		puiRevision);
		
	FLMUINT flmGetLinuxMaxFileSize( void);
	
	void flmGetLinuxMemInfo(
		FLMUINT64 *		pui64TotalMem,
		FLMUINT64 *		pui64AvailMem);
#endif

#endif	// FLM_UNIX

#endif	// FPOSIX_H
