//-------------------------------------------------------------------------
// Desc:	Posix File I/O - definitions
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
// $Id: fposix.h 12330 2006-01-23 10:07:04 -0700 (Mon, 23 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FPOSIX_H
#define FPOSIX_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h


#define INVALID_HANDLE_VALUE 		(-1)

/****************************************************************************
Desc:
****************************************************************************/
class F_FileHdlImp : public F_FileHdlImpBase
{
public:

	F_FileHdlImp();

	virtual ~F_FileHdlImp();

	RCODE Setup(							
		FLMUINT				uiFileId);

	RCODE Close( void);
												
	RCODE Create(
		const char *		pIoPath,
		FLMUINT				uiIoFlags);

	RCODE CreateUnique(
		char *				pIoPath,
		const char *		pszFileExtension,
		FLMUINT				uiIoFlags);

	RCODE Open(
		const char *		pIoPath,
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

	RCODE SectorRead(
		FLMUINT				uiReadOffset,
		FLMUINT				uiBytesToRead,
		void *				pvBuffer,
		FLMUINT *			puiBytesReadRV);

	RCODE SectorWrite(
		FLMUINT				uiWriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT				uiBufferSize,
		F_IOBuffer *		pBufferObj,
		FLMUINT *			puiBytesWrittenRV,
		FLMBOOL				bZeroFill = TRUE)
	{
		if( m_bDoDirectIO)
		{
			return( DirectWrite( uiWriteOffset, uiBytesToWrite, 
				pvBuffer, uiBufferSize, pBufferObj, puiBytesWrittenRV, TRUE, 
				bZeroFill));
		}
		else
		{
			return( Write( uiWriteOffset, uiBytesToWrite, 
				pvBuffer, puiBytesWrittenRV));
		}
	}

	FLMBOOL CanDoAsync( void);

	FINLINE FLMBOOL UsingDirectIo( void)
	{
		return( m_bDoDirectIO);
	}

	FINLINE void setExtendSize(
		FLMUINT				uiExtendSize)
	{
		m_uiExtendSize = uiExtendSize;
	}

	FINLINE void setMaxAutoExtendSize(
		FLMUINT				uiMaxAutoExtendSize)
	{
		m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
	}

	RCODE Lock( void);

	RCODE Unlock( void);

	FINLINE void SetBlockSize(
		FLMUINT				uiBlockSize)
	{
		m_uiBlockSize = uiBlockSize;
	}
	
	FINLINE FLMUINT GetSectorSize( void)
	{
		return( m_uiBytesPerSector);
	}

private:

	RCODE OpenOrCreate(
		const char *		pFileName,
		FLMUINT				uiAccess,
		FLMBOOL				bCreateFlag);

	FINLINE FLMUINT RoundUpToSectorMultiple(
		FLMUINT				uiBytes)
	{
		return( (uiBytes + m_uiNotOnSectorBoundMask) &
					m_uiGetSectorBoundMask);
	}

	FINLINE FLMUINT GetSectorStartOffset(
		FLMUINT				uiOffset)
	{
		return( uiOffset & m_uiGetSectorBoundMask);
	}

	RCODE DirectRead(
		FLMUINT				uiReadOffset,
		FLMUINT				uiBytesToRead,	
   	void *				pvBuffer,
		FLMBOOL				bBuffHasFullSectors,
		FLMUINT *			puiBytesRead);

	RCODE DirectWrite(
		FLMUINT				uiWriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT				uiBufferSize,
		F_IOBuffer *		pBufferObj,
		FLMUINT *			puiBytesWrittenRV,
		FLMBOOL				bBuffHasFullSectors,
		FLMBOOL				bZeroFill);
	
	RCODE AllocAlignBuffer( void);
	
	int				   	m_fd;
	FLMUINT					m_uiBlockSize;
	FLMUINT					m_uiBytesPerSector;
	FLMUINT					m_uiNotOnSectorBoundMask;
	FLMUINT					m_uiGetSectorBoundMask;
	FLMUINT					m_uiCurrentPos;
	FLMUINT					m_uiExtendSize;
	FLMUINT					m_uiMaxAutoExtendSize;
	FLMBOOL					m_bCanDoAsync;
	FLMBOOL					m_bDoDirectIO;
	FLMBYTE *				m_pucAlignedBuff;
	FLMUINT					m_uiAlignedBuffSize;

	friend class F_FileHdlPage;
};

FLMUINT flmGetFSBlockSize(
	const char *			pszFileName);
	
#ifdef FLM_LINUX
	void flmGetLinuxKernelVersion(
		FLMUINT *		puiMajor,
		FLMUINT *		puiMinor,
		FLMUINT *		puiRevision);
		
	FLMUINT flmGetLinuxMaxFileSize(
		FLMUINT			uiSizeofFLMUINT);
#endif
	
#include "fpackoff.h"

#endif
