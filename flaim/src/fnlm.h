//-------------------------------------------------------------------------
// Desc:	I/O for Netware OS - class definitions
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
// $Id: fnlm.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FNLM_H
#define FNLM_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// NOTE: We WANT this header file to come after fpackon.h so that it will
// be byte packed.

// #include "zomni.h"

#define NLM_SECTOR_SIZE 512

RCODE NWTestIfFileExists(
	const char *			pPath);

RCODE NWDeleteFile(
	const char *			pPath);

RCODE NWRenameFile(
	const char *      	pOldFilePath,
	const char *      	pNewFilePath);

class F_FileHdlImp : public F_FileHdlImpBase
{
public:

	F_FileHdlImp();						// F_FileHdlImp Constructor

	virtual ~F_FileHdlImp( void)
	{
		if( m_bFileOpened)
		{
			(void)Close();
		}
	}

	RCODE Setup(							
		FLMUINT				uiFileId);

	RCODE Close();
												
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

	FINLINE RCODE Flush( void)
	{
		return( FERR_OK);
	}

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
		FLMBOOL				bZeroFill = TRUE);

	FINLINE FLMBOOL CanDoAsync( void)
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

	FINLINE void setSuballocation(
		FLMBOOL				bDoSuballocation)
	{
		m_bDoSuballocation = bDoSuballocation;
	}

	FINLINE FLMUINT GetSectorSize( void)
	{
		return( NLM_SECTOR_SIZE);
	}

	FINLINE void SetBlockSize( FLMUINT)
	{
	}
	
private:

	RCODE OpenOrCreate(
		const char	*		pszFileName,
		FLMUINT				uiAccess,
		FLMBOOL				bCreateFlag);

	RCODE _Read(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		void *				pvBuffer,
		FLMUINT *			puiBytesRead);

	RCODE _DirectIORead(
		FLMUINT				uiOffset,
		FLMUINT				uiLength,
		void *				pvBuffer,
		FLMUINT *			puiBytesRead);

	RCODE _DirectIOSectorRead(
		FLMUINT				uiReadOffset,
		FLMUINT				uiBytesToRead,	
		void *				pvBuffer,
		FLMUINT *			puiBytesReadRV);

	RCODE _Write(
		FLMUINT				uiWriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT *			puiBytesWrittenRV);

	RCODE _DirectIOWrite(
		FLMUINT				uiWriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT *			puiBytesWrittenRV);

	RCODE Expand(
		LONG					lStartSector,
		LONG					lSectorsToAlloc);

	RCODE WriteSectors(
		void *				pvBuffer,
		LONG					lStartSector,
		LONG					lSectorCount,
		F_IOBuffer *		pBufferObj,
		FLMBOOL *			pbDidAsync = NULL);

	RCODE _DirectIOSectorWrite(
		FLMUINT				uiWriteOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		F_IOBuffer *		pBufferObj,
		FLMUINT *			puiBytesWrittenRV,
		FLMBOOL				bZeroFill);

	LONG						m_lFileHandle;
	LONG						m_lOpenAttr;
	LONG						m_lVolumeID;
	LONG						m_lLNamePathCount;
	FLMBOOL					m_bDoSuballocation;
	FLMUINT					m_uiExtendSize;
	FLMUINT					m_uiMaxAutoExtendSize;
	FLMBOOL					m_bDoDirectIO;
	LONG						m_lSectorsPerBlock;
	LONG						m_lMaxBlocks;
	FLMUINT					m_uiCurrentPos;
	FLMBOOL					m_bNSS;
	FLMINT64					m_NssKey;
	FLMBOOL					m_bNSSFileOpen;

	friend class F_FileHdlPage;
};

#include "fpackoff.h"

#endif
