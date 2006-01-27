//-------------------------------------------------------------------------
// Desc:	Windows I/O - definitions.
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
// $Id: fwin.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FWIN_H
#define FWIN_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

class F_FileHdlImp : public F_FileHdlImpBase
{
public:

	F_FileHdlImp();

	virtual ~F_FileHdlImp();

	RCODE Setup(							
		FLMUINT				uiFileId);

	RCODE Close( void);
												
	RCODE Create(
		const char *		pszIoPath,
		FLMUINT				uiIoFlags);

	RCODE CreateUnique(
		char *				pszIoPath,
		const char *		pszFileExtension,
		FLMUINT				uiIoFlags);

	RCODE Open(
		const char *		pszIoPath,
		FLMUINT				uiIoFlags);

	RCODE Flush( void);

	RCODE Read(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		void *				pvBuffer,
		FLMUINT *			puiBytesRead);

	RCODE Seek(
		FLMUINT				uiOffset,
		FLMINT				iWhence,
		FLMUINT *			puiNewOffset);

	RCODE Size(
		FLMUINT *			puiSize);

	RCODE Tell(
		FLMUINT *			puiOffset);

	RCODE Truncate(
		FLMUINT				uiSize);

	RCODE Write(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		const void *		pvBuffer,
		FLMUINT *			puiBytesWritten);

	FINLINE RCODE SectorRead(
		FLMUINT			uiReadOffset,
		FLMUINT			uiBytesToRead,
		void *			pvBuffer,
		FLMUINT *		puiBytesReadRV)
	{
		if( m_bDoDirectIO)
		{
			return( DirectRead( uiReadOffset, uiBytesToRead,
					pvBuffer, TRUE, puiBytesReadRV));
		}
		else
		{
			return( Read( uiReadOffset, uiBytesToRead, pvBuffer, puiBytesReadRV));
		}
	}

	FINLINE RCODE SectorWrite(
		FLMUINT			uiWriteOffset,
		FLMUINT			uiBytesToWrite,
		const void *	pvBuffer,
		FLMUINT			uiBufferSize,
		F_IOBuffer *	pBufferObj,
		FLMUINT *		puiBytesWrittenRV,
		FLMBOOL			bZeroFill = TRUE)
	{
		uiBufferSize = uiBufferSize;
		if( m_bDoDirectIO)
		{
			return( DirectWrite( uiWriteOffset, uiBytesToWrite,
					pvBuffer, pBufferObj, TRUE, bZeroFill, puiBytesWrittenRV));
		}
		else
		{
			flmAssert( pBufferObj == NULL);
			return( Write( uiWriteOffset, uiBytesToWrite, pvBuffer,
								puiBytesWrittenRV));
		}
	}

	FINLINE FLMBOOL CanDoAsync( void)
	{
		return( m_bCanDoAsync);
	}

	FINLINE void setExtendSize(
		FLMUINT			uiExtendSize)
	{
		m_uiExtendSize = uiExtendSize;
	}

	FINLINE void setMaxAutoExtendSize(
		FLMUINT			uiMaxAutoExtendSize)
	{
		m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
	}

	FINLINE FLMUINT GetSectorSize( void)
	{
		return( m_uiBytesPerSector);
	}

	FINLINE void SetBlockSize(
		FLMUINT			uiBlockSize)
	{
		m_uiBlockSize = uiBlockSize;
	}

	FINLINE HANDLE getFileHandle( void)
	{
		return m_FileHandle;
	}

private:

	RCODE OpenOrCreate(
		const char *		pszFileName,
		FLMUINT				uiAccess,
		FLMBOOL				bCreateFlag);

	RCODE AllocAlignBuffer( void);

	RCODE DoOneRead(
		DWORD					udReadOffset,
		DWORD					udBytesToRead,
		LPVOID				pvReadBuffer,
		LPDWORD				pudBytesRead);

	RCODE DoOneRead(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		void *				pvBuffer,
		FLMBOOL				bBuffHasFullSectors,
		FLMUINT *			puiBytesRead);

	RCODE DirectRead(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		void *				pvBuffer,
		FLMBOOL				bBuffHasFullSectors,
		FLMUINT *			puiBytesRead);

	RCODE DirectWrite(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		const void *		pvBuffer,
		F_IOBuffer *		pBufferObj,
		FLMBOOL				bBuffHasFullSectors,
		FLMBOOL				bZeroFill,
		FLMUINT *			puiBytesWritten);

	FINLINE FLMUINT RoundToNextSector(
		FLMUINT				uiBytes)
	{
		return( ((uiBytes) + m_uiNotOnSectorBoundMask) &
					m_uiGetSectorBoundMask);
	}

	FINLINE FLMUINT TruncateToPrevSector(
		FLMUINT				uiOffset)
	{
		return( (uiOffset) & m_uiGetSectorBoundMask);
	}

	RCODE extendFile(
		FLMUINT				uiEndOfLastWrite,
		FLMUINT				uiMaxBytesToExtend,
		FLMBOOL				bFlush);

	HANDLE					m_FileHandle;
	FLMUINT					m_uiBlockSize;
	FLMUINT					m_uiBytesPerSector;
	FLMUINT					m_uiNotOnSectorBoundMask;
	FLMUINT					m_uiGetSectorBoundMask;
	FLMBOOL					m_bDoDirectIO;
	FLMUINT					m_uiExtendSize;
	FLMUINT					m_uiMaxAutoExtendSize;
	FLMBYTE *				m_pucAlignedBuff;
	FLMUINT					m_uiAlignedBuffSize;
	FLMUINT					m_uiCurrentPos;
	FLMBOOL					m_bCanDoAsync;
	OVERLAPPED				m_Overlapped;

	friend class F_FileHdlPage;
};

#include "fpackoff.h"

#endif
