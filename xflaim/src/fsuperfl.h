//------------------------------------------------------------------------------
// Desc:	This include file contains the class definitions for FLAIM's
//			super file class.
//
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsuperfl.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FSUPERFL_H
#define FSUPERFL_H

#define MAX_CHECKED_OUT_FILE_HDLS	8

void bldSuperFileExtension(
	FLMUINT			uiFileNum,
	char *			pszFileExtension);

typedef struct
{
	IF_FileHdl *	pFileHdl;
	FLMUINT			uiFileNumber;
	FLMBOOL			bDirty;
} CHECKED_OUT_FILE_HDL;

/****************************************************************************
Desc:		The F_SuperFileHdl object manages the control and block files
			associated with a FLAIM Super File.  This class also provides
			backward compatibility with prior file formats.
Note:
****************************************************************************/
class F_SuperFileHdl : public F_Object
{
public:
	F_SuperFileHdl();

	~F_SuperFileHdl();

	RCODE setup(
		const char *		pszDbFileName,
		const char *		pszDataDir);

	RCODE createFile(
		FLMUINT				uiFileNumber);

	RCODE readBlock(
		FLMUINT				uiBlkAddress,
		FLMUINT				uiBytesToRead,
		void *				pvBuffer,
		FLMUINT *			puiBytesRead);

	RCODE writeBlock(
		FLMUINT				uiBlkAddress,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT				uiBufferSize,
		IF_IOBuffer *		pIOBuffer,
		FLMUINT *			puiBytesWritten);

	RCODE readHeader(
		FLMUINT				uiOffset,
		FLMUINT				uiBytesToRead,
		void *				pvBuffer,
		FLMUINT *			puiBytesRead);

	RCODE writeHeader(
		FLMUINT				uiOffset,
		FLMUINT				uiBytesToWrite,
		const void *		pvBuffer,
		FLMUINT *			puiBytesWritten);

	RCODE getFilePath(
		FLMUINT				uiFileNumber,
		char *				pszPath);

	RCODE	getFileHdl(
		FLMUINT				uiFileNumber,
		FLMBOOL				bGetForUpdate,
		IF_FileHdl **		ppFileHdlRV);

	RCODE getFileSize(
		FLMUINT				uiFileNumber,
		FLMUINT64 *			pui64FileSize);

	RCODE releaseFile(
		FLMUINT				uiFileNum,
		FLMBOOL				bCloseFile);

	RCODE releaseFiles(
		FLMBOOL				bCloseFiles);

	RCODE truncateFile(
		FLMUINT				uiEOFBlkAddress);

	void truncateFiles(
		FLMUINT				uiStartFileNum,
		FLMUINT				uiEndFileNum);

	RCODE releaseFile(
		CHECKED_OUT_FILE_HDL *	pChkFileHdl,
		FLMBOOL						bCloseFile);

	FINLINE void enableFlushMinimize( void)
	{
		m_bMinimizeFlushes = TRUE;
	}

	void disableFlushMinimize( void);

	RCODE flush( void);

	FINLINE void setBlockSize(
		FLMUINT		uiBlockSize)
	{
		m_uiBlockSize = uiBlockSize;
	}

	FINLINE void setExtendSize(
		FLMUINT		uiExtendSize)
	{
		m_uiExtendSize = uiExtendSize;
	}

	FINLINE void setMaxAutoExtendSize(
		FLMUINT		uiMaxAutoExtendSize)
	{
		m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
	}

	FINLINE FLMBOOL canDoAsync( void)
	{
		if (m_pCheckedOutFileHdls[ 0].pFileHdl)
		{
			return( m_pCheckedOutFileHdls[ 0].pFileHdl->canDoAsync());
		}
		else
		{
			IF_FileHdl *		pFileHdl;

			if( RC_OK( getFileHdl( 0, FALSE, &pFileHdl)))
			{
				return( pFileHdl->canDoAsync());
			}
		}

		return( FALSE);
	}

private:

	FINLINE CHECKED_OUT_FILE_HDL * getCkoFileHdlPtr(
		FLMUINT		uiFileNum,
		FLMUINT *	puiSlot)
	{
		*puiSlot = (uiFileNum
			? (uiFileNum % (m_uiCkoArraySize - 1)) + 1
			: 0);

		return( &m_pCheckedOutFileHdls[ *puiSlot]);
	}

	FINLINE void clearCkoFileHdl(
		CHECKED_OUT_FILE_HDL *		pCkoFileHdl)
	{
		pCkoFileHdl->pFileHdl = NULL;
		pCkoFileHdl->uiFileNumber = 0;
		pCkoFileHdl->bDirty = FALSE;
	}

	void copyCkoFileHdls(
		CHECKED_OUT_FILE_HDL *	pSrcCkoArray,
		FLMUINT						uiSrcHighestUsedSlot);

	RCODE reallocCkoArray(
		FLMUINT	uiFileNum);

	char *						m_pszDbFileName;
	char *						m_pszDataFileNameBase;
	FLMUINT						m_uiExtOffset;
	FLMUINT						m_uiDataExtOffset;
	CHECKED_OUT_FILE_HDL		m_CheckedOutFileHdls[
										MAX_CHECKED_OUT_FILE_HDLS + 1];
	CHECKED_OUT_FILE_HDL *	m_pCheckedOutFileHdls;
	FLMUINT						m_uiCkoArraySize;
	FLMUINT						m_uiBlockSize;
	FLMUINT						m_uiExtendSize;
	FLMUINT						m_uiMaxAutoExtendSize;
	FLMUINT						m_uiLowestDirtySlot;
	FLMUINT						m_uiHighestDirtySlot;
	FLMUINT						m_uiHighestUsedSlot;
	FLMUINT						m_uiHighestFileNumber;
	FLMBOOL						m_bMinimizeFlushes;
	FLMBOOL						m_bSetupCalled;
};

#endif	// FSUPERFL_H
