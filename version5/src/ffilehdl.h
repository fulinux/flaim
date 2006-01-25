//------------------------------------------------------------------------------
//	Desc:	This include file contains the class definitions for FLAIM's
//			FileHdlMgr and FileHdl classes.
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
// $Id: ffilehdl.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FFILEHDL_H
#define FFILEHDL_H

class F_FileHdl;							// Forward Reference
class F_FileHdlMgr;						// Forward Reference
class F_FileHdlMgrPage;					// Source: imonfmgr.cpp
class F_FileHdlPage;						// Source: imonfhdl.cpp

#define	FHM_AVAIL_LIST		0
#define	FHM_USED_LIST		1
#define	FHM_LNODE_COUNT	2

RCODE DetermineLockMgr(
	F_Database *	pDatabase,
	F_FileHdl *		pFileHdl);

RCODE flmCopyPartial(
	IF_FileHdl *	pSrcFileHdl,
	FLMUINT64		ui64SrcOffset,
	FLMUINT64		ui64SrcSize,
	IF_FileHdl *	pDestFileHdl,
	FLMUINT64		ui64DestOffset,
	FLMUINT64 *		pui64BytesCopiedRV);

RCODE f_filecpy(
	const char *	pszSourceFile,
	const char *	pszData);

RCODE f_filecat(
	const char *	pszSourceFile,
	const char *	pszData);

#include "fwin.h"
#include "fposix.h"

/*===========================================================================
Class:	F_FileHdlMgr
Desc:		The F_FileHdlMgr class manages F_FileHdl objects.
			The F_FileHdlMgr maintains two lists:
				1) Available F_FileHdl's - F_FileHdl's not currently being used.
				2) Used F_FileHdl's - F_FileHdl's that have been checked out.
===========================================================================*/
class F_FileHdlMgr : public XF_RefCount, public XF_Base
{
public:

	F_FileHdlMgr();

	FINLINE ~F_FileHdlMgr()
	{
		if (m_hMutex != F_MUTEX_NULL)
		{
			lockMutex( FALSE);
			freeUsedList( TRUE);
			freeAvailList( TRUE);
			unlockMutex( FALSE);
			f_mutexDestroy( &m_hMutex);
		}
	}

	RCODE setupFileHdlMgr(				// Setup the F_FileHdlMgr object
		FLMUINT	uiOpenThreshold = XFLM_DEFAULT_OPEN_THRESHOLD,
												// High water mark for open handles.
		FLMUINT	uiMaxAvailTime = XFLM_DEFAULT_MAX_AVAIL_TIME);
												// Max avail time to wait to close files.

	FINLINE void setOpenThreshold(
		FLMUINT		uiOpenThreshold)
	{
		if (m_bIsSetup)
		{
			lockMutex( FALSE);
			m_uiOpenThreshold = uiOpenThreshold;
			unlockMutex( FALSE);
		}
	}

	FINLINE void setMaxAvailTime(
		FLMUINT	uiMaxAvailTime)
	{
		if (m_bIsSetup)
		{
			lockMutex( FALSE);
			m_uiMaxAvailTime = uiMaxAvailTime;
			unlockMutex( FALSE);
		}
	}

	FINLINE FLMUINT getUniqueId( void)
	{
		FLMUINT	uiTemp;

		lockMutex( FALSE);
		uiTemp = ++m_uiFileIdCounter;
		unlockMutex( FALSE);
		return( uiTemp);
	}

	// Methods for Avail & Used List Management

	void findAvail(						// Determines if the F_FileHdlMgr has an
												// available F_FileHdl for the specified FileId
		FLMUINT			uiFileId,		// Desired FileHdr's ID
		FLMBOOL			bReadOnlyFlag,	// TRUE if looking for read only file
		F_FileHdl **	ppFileHdl);		// [out] returned F_FileHdl object.

	void makeAvailAndRelease(			// Make the specified F_FileHdl available for
												// someone else to use.
		FLMBOOL			bMutexAlreadyLocked,
		F_FileHdl *		pFileHdl);		// F_FileHdl to move to the available list.

	void removeFileHdls(					// Remove (close&free) all FileHdl's that
												// have the specified FileId.
		FLMUINT			uiFileId);

	void checkAgedFileHdls(						// Remove aged items older than
		FLMUINT			uiMinSecondsOpened);	// minimum seconds opened.

	void FINLINE releaseOneAvail(
		FLMBOOL	bMutexAlreadyLocked
		)
	{
		lockMutex( bMutexAlreadyLocked);
		if (m_pFirstAvail)
		{
			removeFromList( TRUE,
				m_pFirstAvail, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
		}
		unlockMutex( bMutexAlreadyLocked);
	}

	// Manager Statistics Methods

	FINLINE FLMUINT getOpenThreshold( void)
	{
		return m_uiOpenThreshold;
	}

	FINLINE FLMUINT getOpenedFiles( void)
	{
		FLMUINT		uiTemp;

		lockMutex( FALSE);
		uiTemp = m_uiNumUsed + m_uiNumAvail;
		unlockMutex( FALSE);
		return( uiTemp);
	}

	FINLINE FLMUINT getMaxAvailTime( void)
	{
		return m_uiMaxAvailTime;
	}

	// Misc. Methods

	void freeAvailList(
		FLMBOOL	bMutexAlreadyLocked);

	void freeUsedList(
		FLMBOOL	bMutexAlreadyLocked);

	FINLINE void insertInUsedList(
		FLMBOOL		bMutexAlreadyLocked,
		F_FileHdl *	pFileHdl,
		FLMBOOL		bInsertAtEnd
		)
	{
		insertInList( bMutexAlreadyLocked,
			pFileHdl, bInsertAtEnd,
			&m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);
	}

private:

	void insertInList(
		FLMBOOL			bMutexAlreadyLocked,
		F_FileHdl *		pFileHdl,
		FLMBOOL			bInsertAtEnd,
		F_FileHdl **	ppFirst,
		F_FileHdl **	ppLast,
		FLMUINT *		puiCount);

	void removeFromList(
		FLMBOOL			bMutexAlreadyLocked,
		F_FileHdl *		pFileHdl,
		F_FileHdl **	ppFirst,
		F_FileHdl **	ppLast,
		FLMUINT *		puiCount);

	FINLINE void lockMutex(
		FLMBOOL	bMutexAlreadyLocked)
	{
		if (m_hMutex != F_MUTEX_NULL && !bMutexAlreadyLocked)
		{
			f_mutexLock( m_hMutex);
		}
	}

	FINLINE void unlockMutex(
		FLMBOOL	bMutexAlreadyLocked)
	{
		if (m_hMutex != F_MUTEX_NULL && !bMutexAlreadyLocked)
		{
			f_mutexUnlock( m_hMutex);
		}
	}

	// Private variables

	F_MUTEX				m_hMutex;
	FLMUINT				m_uiOpenThreshold;	// FileHdl open threshold.
	FLMUINT				m_uiMaxAvailTime;		// Time to close any available files.

	// Used list

	F_FileHdl *			m_pFirstUsed;
	F_FileHdl *			m_pLastUsed;
	FLMUINT				m_uiNumUsed;

	// Avail list

	F_FileHdl *			m_pFirstAvail;
	F_FileHdl *			m_pLastAvail;
	FLMUINT				m_uiNumAvail;

	FLMBOOL				m_bIsSetup;				// TRUE when list manager is set up.
	FLMUINT				m_uiFileIdCounter;	// Used for getUniqueId() call.

friend class F_FileHdlMgrPage;
friend class F_FileHdl;

};

#endif 		// #ifndef FFILEHDL_H
