//-------------------------------------------------------------------------
// Desc:	File handle class - definitions.
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
// $Id: ffilehdl.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FFILEHDL_H
#define FFILEHDL_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define DEFAULT_OPEN_THRESHOLD		100		// 100 file handles to cache
#define DEFAULT_MAX_AVAIL_TIME		900		// 15 minutes	

class F_FileHdlImp;
class F_FileHdlMgr;
class F_FileHdlMgrPage;
class F_FileHdlPage;
class F_ListMgr;
class F_MutexRef;

typedef F_FileHdlMgr *		F_FileHdlMgr_p;

#define	FHM_AVAIL_LIST		0
#define	FHM_USED_LIST		1
#define	FHM_LNODE_COUNT	2

RCODE flmCloseAllFiles();

RCODE DetermineLockMgr(
	FFILE_p			pFile,
	F_FileHdlImp *	pFileHdl);

RCODE flmCopyPartial(
	F_FileHdl *		pSrcFileHdl,
	FLMUINT			uiSrcOffset,
	FLMUINT			uiSrcSize,
	F_FileHdl *		pDestFileHdl,
	FLMUINT			uiDestOffset,
	FLMUINT *		puiBytesCopiedRV);

#ifdef FLM_DEBUG
	#define GET_FS_ERROR()	(gv_CriticalFSError)
#else
	#define GET_FS_ERROR()	(FERR_OK)
#endif

RCODE f_filecpy(
	const char *	pszSourceFile,
	const char *	pszData);

RCODE f_filecat(
	const char *	pszSourceFile,
	const char *	pszData);

/****************************************************************************
Desc:		The F_FileHdlMgr class manages F_FileHdlImp objects. 
			The F_FileHdlMgr maintains two lists: 
				1) Available F_FileHdlImp's - F_FileHdlImp's not currently 
				being used.
				2) Used F_FileHdlImp's - F_FileHdlImp's that have been 
				checked out.
****************************************************************************/
class F_FileHdlMgr : public F_Base
{
public:

	F_FileHdlMgr(
		F_MUTEX	* phMutex);

	FINLINE virtual ~F_FileHdlMgr()
	{
		// Free all file handles in the available and used lists

		(void) m_ListMgr.ClearList( FHM_USED_LIST);
		(void) m_ListMgr.ClearList( FHM_AVAIL_LIST);
	}

	RCODE Setup( 
		FLMUINT		uiOpenThreshold = DEFAULT_OPEN_THRESHOLD,	
		FLMUINT		uiMaxAvailTime = DEFAULT_MAX_AVAIL_TIME);	

	FINLINE void SetMutexPtr(
		F_MUTEX	* 	phMutex)
	{
		m_phMutex = phMutex;
	}

	FINLINE RCODE SetOpenThreshold(
		FLMUINT		uiOpenThreshold)
	{
		return Setup( uiOpenThreshold, m_uiMaxAvailTime);
	}

	FINLINE RCODE SetMaxAvailTime(
		FLMUINT	uiMaxAvailTime)
	{
		return Setup( m_uiOpenThreshold, uiMaxAvailTime);
	}

	FLMUINT GetUniqueId();				// Returns a unique id that can be assigned
												// to a F_FileHdlImp.

	/*
	Methods for Avail & Used List Management
	*/

	FINLINE RCODE FindAvail(
		FLMUINT				uiFileId,			// Desired FileHdr's ID
		FLMBOOL				bReadOnlyFlag,		// TRUE if file is read only
		F_FileHdlImp **	ppFileHdl)			// [out] returned FileHdl object.
	{
		F_MutexRef			MutexRef( m_phMutex );

		return FindAvail( &MutexRef, uiFileId, bReadOnlyFlag,
								ppFileHdl);
	}

	FINLINE RCODE InsertNew(
		F_FileHdlImp *		pFileHdl)			// FileHdl to add to this manager.
	{
		F_MutexRef			MutexRef( m_phMutex);
		return InsertNew( &MutexRef, pFileHdl);
	}

	FINLINE RCODE MakeAvailAndRelease(
		F_FileHdlImp *		pFileHdl)			// FileHdl to move to the avail list.
	{
		F_MutexRef			MutexRef( m_phMutex);
		return MakeAvailAndRelease( &MutexRef, pFileHdl);
	}

	FINLINE RCODE Remove(
 		FLMUINT	uiFileId)
	{
		F_MutexRef			MutexRef( m_phMutex);
		return Remove( &MutexRef, uiFileId, TRUE);
	}

	RCODE Remove(							// Remove specific file handle.
		F_FileHdlImp *		pFileHdl);

	FINLINE RCODE RemoveAvail(
 		FLMUINT		uiFileId)
	{
		F_MutexRef			MutexRef( m_phMutex);
		return Remove( &MutexRef, uiFileId, FALSE);
	}

	RCODE CheckAgedItems(				// Remove aged items older than 
		FLMUINT			uiMinSecondsOpened);// minimum seconds opened.
		
	FINLINE RCODE ReleaseOneAvail()
	{
 		F_MutexRef			MutexRef( m_phMutex);
		return ReleaseOneAvail( &MutexRef);
	}

	RCODE ReleaseUsedFiles();			// Release all used files.

	FINLINE FLMUINT GetOpenThreshold()
	{
		return m_uiOpenThreshold;
	}

	FLMUINT GetOpenedFiles();

	FINLINE FLMUINT GetMaxAvailTime()
	{
		return m_uiMaxAvailTime;
	}

	// Misc. Methods

	FINLINE F_ListMgr * GetListMgr()
	{
		return &m_ListMgr;
	}

private:

	F_MUTEX	*	m_phMutex;			// Points to owning mutex
	FLMUINT		m_uiOpenThreshold;	// FileHdl open threshold.
	FLMUINT		m_uiMaxAvailTime;		// Time to close any available files.

	F_ListMgr	m_ListMgr;				// List manager
	F_ListNode	m_LNodes[ FHM_LNODE_COUNT];
	FLMBOOL		m_bIsSetup;				// TRUE when list manager is set up.
	FLMUINT		m_uiFileIdCounter;	// Used for GetUniqueId() call.

	// Methods for Avail & Used List Management

	RCODE FindAvail(						// Determines if the F_FileHdlMgr has an 
												// available F_FileHdlImp for the specified FileId
		F_MutexRef *		pMutexRef,
		FLMUINT				uiFileId,		// Desired FileHdr's ID
		FLMBOOL				bReadOnlyFlag,	// TRUE if looking for read only file
		F_FileHdlImp **	ppFileHdl);		// [out] returned F_FileHdlImp object.

	RCODE InsertNew(
		F_MutexRef *		pMutexRef,
		F_FileHdlImp *		pFileHdl);		// FileHdl to add to this manager.

	RCODE MakeAvailAndRelease(			// Make the specified F_FileHdlImp available for
												// someone else to use.
		F_MutexRef *		pMutexRef,
		F_FileHdlImp *		pFileHdl);	// F_FileHdlImp to move to the avail list.

	RCODE Remove(							// Remove (close&free) all FileHdl's that 
												// have the specified FileId.
		F_MutexRef *	pMutexRef,
		FLMUINT			uiFileId,
		FLMBOOL			bFreeUsedFiles);// Should used files be freed

	RCODE ReleaseOneAvail(				// FERR_OK or FERR_NOT_FOUND.  Releases
												// a single file handle on the avail list.
		F_MutexRef *	pMutexRef);

	// Misc. Methods

	RCODE CheckAgedItems(				// Remove aged items older than 
		F_MutexRef *	pMutexRef);			// m_udMaxAvailTime.

	// Manager Statistics Methods

	FLMUINT GetOpenedFiles(				// Return number of opened file handles
		F_MutexRef *	pMutexRef);

	friend class F_FileHdlMgrPage;
};

/****************************************************************************
Desc:		Base class for file handle implementations
****************************************************************************/
class F_FileHdlImpBase : public F_FileHdl
{
public:

	F_FileHdlImpBase()
	{
		m_uiAvailTime = 0;
		m_bFileOpened = FALSE;
		m_bDeleteOnClose = FALSE;
		m_bOpenedReadOnly = FALSE;
		m_pszIoPath = NULL;
	}

	virtual ~F_FileHdlImpBase()
	{
		if( m_pszIoPath)
		{
			f_free( &m_pszIoPath);
		}
	}

	FINLINE FLMUINT GetFileId()
	{
		return m_uiFileId;
	}

	FINLINE FLMUINT GetAvailTime( void)
	{
		return m_uiAvailTime;
	}

	FINLINE void SetAvailTime( void)
	{
		m_uiAvailTime = (FLMUINT)FLM_GET_TIMER();
	}

	FINLINE FLMBOOL IsOpenedReadOnly( void)
	{
		return( m_bOpenedReadOnly);
	}

	FINLINE FLMBOOL IsOpenedExclusive( void)
	{
		return( m_bOpenedExclusive);
	}

	virtual RCODE SectorRead(						// Allows sector reads to be done.
		FLMUINT			uiReadOffset,				// Offset to being reading at.
		FLMUINT			uiBytesToRead,				// Number of bytes to read
		void *			pvBuffer,					// Buffer to place read bytes into
		FLMUINT *		puiBytesReadRV) = 0;		// [out] number of bytes read

	virtual RCODE SectorWrite(						// Allows sector writes to be done.
		FLMUINT			uiWriteOffset,				// Offset to seek to.
		FLMUINT			uiBytesToWrite,			// Number of bytes to write.
		const void *	pvBuffer,					// Buffer that contains bytes to be written
		FLMUINT			uiBufferSize,				// Actual buffer size.
		F_IOBuffer *	pBufferObj,					// Do asynchronous write if non-NULL.
		FLMUINT *		puiBytesWrittenRV,		// Number of bytes written.
		FLMBOOL			bZeroFill = TRUE) = 0;	// Zero fill the buffer?

protected:

	FLMBOOL			m_bOpenedReadOnly;	// Opened the file read only
	FLMBOOL			m_bOpenedExclusive;	// Opened the file in exclusive mode
	F_ListNode		m_LNode[2];				// List Item Node, used by FListItem object.
	FLMBOOL			m_bFileOpened;			// Is the file currently opened/closed.
	FLMUINT			m_uiAvailTime;			// Time when placed in avail list.
	FLMUINT			m_uiFileId;				// FFILE Unique File Id
	FLMBOOL			m_bDeleteOnClose;		// Delete this file when it is released.
	char *			m_pszIoPath;			// Path of this FileHdl

friend class F_FileHdlPage;
};

#include "fpackoff.h"

#ifdef FLM_NLM
	#include "fnlm.h"
#elif defined( FLM_WIN)
	#include "fwin.h"
#elif defined( FLM_UNIX)
	#include "fposix.h"
#endif

#endif // FFILEHDL_H
