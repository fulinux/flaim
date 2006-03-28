//-------------------------------------------------------------------------
// Desc:	Super-file class definitions.
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
// $Id: fsuperfl.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FSUPERFL_H
#define FSUPERFL_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define MAX_CHECKED_OUT_FILE_HDLS	8

class F_SuperFileHdl;
class F_FileIdList;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct CHECKED_OUT_FILE_HDL
{
	F_FileHdlImp *	pFileHdl;
	FLMUINT			uiFileNumber;
	FLMBOOL			bDirty;
} CHECKED_OUT_FILE_HDL;

void bldSuperFileExtension( 
	FLMUINT		uiDbVersion,
	FLMUINT		uiFileNum,
	char *		pszFileExtension);

/****************************************************************************
Desc:		This class keeps a list of file IDs for file numbers in a database.
			It does not know the use of the file IDs, and it is not limited in
			the number of file IDs it can keep track of, but it will generally
			be as follows:

			FileNumber
				0				This is the database file (xxx.db)
				1-4095		These are the data files (xxx_data.nnn)
				4096-8192	These are the rollback files (xxx_rb.nnn).  In this
								case, the caller will map the file number in and
								out of this range.
****************************************************************************/
class F_FileIdList : public F_Base
{
public:
	F_FileIdList();

	virtual ~F_FileIdList();

	RCODE setup( void);

	RCODE getFileId(
		FLMUINT				uiFileNumber,
		FLMUINT *			puiFileId);

private:

	F_MUTEX				m_hMutex;
	FLMUINT				m_uiFileIdTblSize;
	FLMUINT *			m_puiFileIdTbl;
};

/****************************************************************************
Desc:		The F_SuperFileHdl object manages the control and block files
			associated with a FLAIM Super File.  This class also provides
			backward compatibility with prior file formats.
****************************************************************************/
class F_SuperFileHdl : public F_Base
{
public:
	F_SuperFileHdl();

	virtual ~F_SuperFileHdl(); 

	RCODE Setup(							// Configures the object.  Should
												// be called exactly once.
		F_FileIdList *		pFileIdList,
		const char *		pszDbFileName,
		const char *		pszDataDir);

	FINLINE void setECacheMgr(			// Sets the ECacheMgr to be used by
												// the super file.
		FlmECache *			pECacheMgr)
	{
		m_pECacheMgr = pECacheMgr;
	}
	
	RCODE CreateFile(						// Create a block file (>= 3.0 only)
		FLMUINT			uiFileNumber);		// File number to create

	RCODE ReadBlock(						// Reads a block from a block file or
												// the log
		FLMUINT			uiBlkAddress,		// Block address
		FLMUINT			uiBytesToRead,		// Number of bytes to read from block
		void *			pvBuffer,			// Buffer to place read bytes into
		FLMUINT *		puiBytesRead);		// [out] number of bytes read

	RCODE WriteBlock(						// Writes a block to a block file or
												// the log
		FLMUINT			uiBlkAddress,		// Block address
		FLMUINT			uiBytesToWrite,	// Number of bytes to write
		void *			pvBuffer,			// Buffer to write bytes from
		FLMUINT			uiBufferSize,		// Actual size of buffer
		F_IOBuffer *	pIOBuffer,			// If non-NULL, contains info for
													// doing an async write.
		FLMUINT *		puiBytesWritten);	// [out] number of bytes written

	RCODE ReadHeader(						// Reads data from the DB header
		FLMUINT			uiOffset,
		FLMUINT			uiBytesToRead,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE WriteHeader(					// Writes data to the DB header
		FLMUINT			uiOffset,
		FLMUINT			uiBytesToWrite,
		void *			pvBuffer,
		FLMUINT *		puiBytesWritten);

	RCODE GetFilePath(					// Generates a block file's path
		FLMUINT			uiFileNumber,		// File number
		char *			pIoPath);			// Returned path

	RCODE	GetFileHdl(						// Returns a file's handle given
												// the file's number
		FLMUINT				uiFileNumber,
		FLMBOOL				bGetForUpdate,
		F_FileHdlImp **	ppFileHdlRV);

	RCODE GetFileSize(					// Returns the physical size of
												// a file
		FLMUINT			uiFileNumber,		// File number
		FLMUINT *		puiFileSize);		// File size return value

	RCODE ReleaseFile(					// Release a single file
		FLMUINT	uiFileNum,
		FLMBOOL	bCloseFile);

	RCODE ReleaseFiles(					// Releases all file handles and
												// returns them to the file handle
												// manager.  This is called at
												// the end of a transaction.
		FLMBOOL		bCloseFiles);			// Should files be closed?

	RCODE TruncateFile(					// Truncate the file(s) to given address.
		FLMUINT	uiEOFBlkAddress);			// End of file block address. 

	void TruncateFiles(					// Truncate the files specified by the
		FLMUINT		uiStartFileNum,	// start and end file numbers.
		FLMUINT		uiEndFileNum);

	RCODE ReleaseFile(					// Release a single file
		CHECKED_OUT_FILE_HDL *	pChkFileHdl,
		FLMBOOL						bCloseFile);

	FINLINE void enableFlushMinimize( void)
	{
		m_bMinimizeFlushes = TRUE;
	}

	void disableFlushMinimize( void);

	RCODE Flush( void);

	FINLINE void SetBlockSize(
		FLMUINT		uiBlockSize)
	{
		m_uiBlockSize = uiBlockSize;
	}

	FINLINE void SetDbVersion(
		FLMUINT		uiDbVersion)
	{
		m_uiDbVersion = uiDbVersion;
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

	FINLINE FLMBOOL CanDoAsync( void)
	{
		if( m_pCheckedOutFileHdls[ 0].pFileHdl)
		{
			return( m_pCheckedOutFileHdls[ 0].pFileHdl->CanDoAsync());
		}
		else
		{
			F_FileHdlImp *		pFileHdl;

			if( RC_OK( GetFileHdl( 0, FALSE, &pFileHdl)))
			{
				return( pFileHdl->CanDoAsync());
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
	F_FileIdList *				m_pFileIdList;
	CHECKED_OUT_FILE_HDL		m_CheckedOutFileHdls[
										MAX_CHECKED_OUT_FILE_HDLS + 1];
	CHECKED_OUT_FILE_HDL *	m_pCheckedOutFileHdls;
	FLMUINT						m_uiCkoArraySize;
	FLMUINT						m_uiBlockSize;
	FLMUINT						m_uiExtendSize;
	FLMUINT						m_uiMaxAutoExtendSize;
	FLMUINT						m_uiDbVersion;
	FLMUINT						m_uiLowestDirtySlot;
	FLMUINT						m_uiHighestDirtySlot;
	FLMUINT						m_uiHighestUsedSlot;
	FLMUINT						m_uiHighestFileNumber;
	FlmECache *					m_pECacheMgr;
	FLMBOOL						m_bMinimizeFlushes;
	FLMBOOL						m_bSetupCalled;
};

#include "fpackoff.h"

#endif
