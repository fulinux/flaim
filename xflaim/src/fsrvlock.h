//------------------------------------------------------------------------------
// Desc:	This include file contains the class definitions for FLAIM's
//			ServerLockManager and ServerLockObject classes.
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
// $Id: fsrvlock.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FSRVLOCK_H
#define FSRVLOCK_H

FLMUINT flmStrHashBucket(
	char *		pszStr,
	FBUCKET *	pHashTbl,
	FLMUINT		uiNumBuckets);

	/*
	*** Define the 'C++' classes for FLAIM's File Handle cache code.
	*/

class ServerLockManager;				// Forward Reference
class ServerLockObject;					// Forward Reference
class F_ItemId;							// Forward Reference
class FFileItemId;						// Forward Reference
class RFileItemId;						// Forward Reference

typedef ServerLockManager *	ServerLockManager_p;
typedef ServerLockObject *		ServerLockObject_p;
typedef F_ItemId *				F_ItemId_p;
typedef FFileItemId *			FFileItemId_p;
typedef RFileItemId *			RFileItemId_p;

/**************************************************************************
Struct:	LOCK_WAITER			(Lock Waiter)
Desc: 	This structure is used to keep track of threads waiting for a
			lock.
**************************************************************************/
typedef struct Lock_Waiter *	LOCK_WAITER_p;

typedef struct Lock_Waiter
{
	ServerLockObject_p	pLockObject;	// Pointer to lock object.
	FLMUINT					uiThreadId;		// Thread of waiter
	F_SEM						hESem;			// Semaphore to signal when lock is
													// granted (or denied).
	RCODE *					pRc;				// Pointer to return code that is to
													// be set when lock is granted or
													// denied.
	FLMUINT					uiWaitStartTime;
													// Time we started waiting.
	FLMUINT					uiWaitTime;		// Time pending lock request should
													// wait before being timed out.
													// Zero means should not be timed out.
	FLMBOOL					bExclReq;		// TRUE if exclusive lock request.
	FLMINT					iPriority;		// Priority of waiter.
	F_TMSTAMP				StartTime;		// Time we started waiting (for stats)
	XFLM_DB_STATS *		pDbStats;		// Statistics to update.
	LOCK_WAITER_p			pNext;			// Next lock waiter in list.
	LOCK_WAITER_p			pPrev;			// Previous lock waiter in list.
	LOCK_WAITER_p			pNextGlobal;	// Next lock waiter in global list
													// that is ordered according to
													// udWaitEndTime.
	LOCK_WAITER_p			pPrevGlobal;	// Previous lock waiter in global list
} LOCK_WAITER;

/*===========================================================================
Class:	ServerLockManager
Desc:		The ServerLockManager class manages ServerLockObject objects.
===========================================================================*/

class ServerLockManager : public F_Object
{
public:

	ServerLockManager()
	{
		m_hMutex = F_MUTEX_NULL;
		m_pFirstLockWaiter = NULL;
		m_pHashTbl = NULL;
		m_uiNumAvail = 0;
		m_pAvailLockList = NULL;
	}

	virtual ~ServerLockManager();		// ServerLockManager Destructor - free
												// ServerLockObjects owned by this
												// ServerLockManager.

	FINLINE RCODE setupLockMgr( void)
	{
		return( f_mutexCreate( &m_hMutex));
	}

	RCODE SetupHashTbl();				// Setup hash table for lock manager.

	void CheckLockTimeouts(				// See if any pending lock requests have
		FLMBOOL	bMutexAlreadyLocked,
		FLMBOOL	bTimeoutAll);			// timed out.

	void InsertWaiter(					// Insert waiter into global list.
		LOCK_WAITER *	pLockWaiter);

	FINLINE void RemoveWaiter(
		LOCK_WAITER *	pLockWaiter)
	{
		if (pLockWaiter->pNextGlobal)
			pLockWaiter->pNextGlobal->pPrevGlobal = pLockWaiter->pPrevGlobal;

		if (pLockWaiter->pPrevGlobal)
		{
			pLockWaiter->pPrevGlobal->pNextGlobal = pLockWaiter->pNextGlobal;
		}
		else
		{
			m_pFirstLockWaiter = pLockWaiter->pNextGlobal;
		}
	}

	ServerLockObject_p GetLockObject(// Return a lock object for the file.
		F_ItemId *	pItemId);

	void SignalLockWaiter(				// Unlink a lock object from lists
		FLMUINT					uiThreadId);

	void UnlinkLockObject(				// Unlink a lock object from lists
		ServerLockObject *	pLockObject,
		FLMBOOL					bPutInAvailList);

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

private:

	// Private variables

	F_MUTEX				m_hMutex;
	FBUCKET *			m_pHashTbl;				// Hash table.
	LOCK_WAITER *		m_pFirstLockWaiter;	// Pointer to first in list of global
														// lock waiters.
	FLMUINT				m_uiNumAvail;			// Number of lock objects in avail
														// list.
	ServerLockObject *
							m_pAvailLockList;		// List of available lock objects.

friend class F_ServerLockMgrPage;
friend class ServerLockObject;

};

/*===========================================================================
Class:	F_ItemId
Desc:		The item id that identifies a particular object.
===========================================================================*/
class F_ItemId : public F_Object
{
public:
	F_ItemId();								// F_ItemId Constructor

	virtual ~F_ItemId();					// F_ItemId Destructor

	virtual FLMBOOL IsEqual(			// Compare to another F_ItemId.
		F_ItemId *	pItemId) = 0;

	virtual FLMUINT GetHashBucket(	// Get hash bucket for lock item id
		FBUCKET *	pHashTbl,
		FLMUINT		uiHashTblSize) = 0;

	FLMUINT GetItemType()				// Returns the type of item.
	{	return m_uiItemType; }

protected:
	FLMUINT	m_uiItemType;				// Item type.
#define			FFILE_ITEM			1
#define			RFILE_ITEM			2
#define			FFILE_TRANS_ITEM	3
#define			RFILE_TRANS_ITEM	4
};


	/*
	Public:	constructor, destructor
	*/
	FINLINE F_ItemId::F_ItemId()
	{
		m_uiItemType = 0;
	}

	FINLINE F_ItemId::~F_ItemId()
	{
	}

/*===========================================================================
Class:	FFileItemId
Desc:		The item id that identifies an FFILE object.
===========================================================================*/
class FFileItemId : public F_ItemId
{
public:

	// Constructor

	FFileItemId(
		F_Database *	pDatabase,
		FLMBOOL			bTrans)
	{
		m_pDatabase = pDatabase;
		m_uiItemType = (FLMUINT)(bTrans
										? (FLMUINT)FFILE_TRANS_ITEM
										: (FLMUINT)FFILE_ITEM);
	}

	virtual ~FFileItemId()
	{
	}

	FLMBOOL IsEqual(						// Compare to another F_ItemId.
		F_ItemId *	pItemId);

	FINLINE FLMUINT GetHashBucket(
		FBUCKET *	pHashTbl,
		FLMUINT		uiHashTblSize)
	{
		char	szFileName[ F_PATH_MAX_SIZE];

		// Extract the file name

		this->GetFileName( szFileName);

		// Determine what hash bucket the file should be in - based on file name.

		return( flmStrHashBucket( szFileName, pHashTbl, uiHashTblSize));
	}

	FINLINE F_Database * getDatabase( void)	// was GetFilePtr
	{
		return m_pDatabase;
	}

	void GetFileName(						// Returns file name
		char *	pszFileNameRV);

private:
	F_Database *	m_pDatabase;
};

/*===========================================================================
Class:	RFileItemId
Desc:		The item id that identifies a file being used by rebuild.
===========================================================================*/
class RFileItemId : public F_ItemId
{
public:
	RFileItemId(							// RFileItemId Constructor
		char *		pszFileName,
		FLMBOOL		bTrans = FALSE);

	virtual ~RFileItemId()				// RFileItemId Destructor
	{
	}

	FLMBOOL IsEqual(						// Compare to another F_ItemId.
		F_ItemId *	pItemId);

	FINLINE FLMUINT GetHashBucket(		// Get hash bucket for lock item id
		FBUCKET *	pHashTbl,
		FLMUINT		uiHashTblSize)
	{
		// Determine what hash bucket the file should be in - based on file name.

		return( flmStrHashBucket( m_szFileName, pHashTbl, uiHashTblSize));
	}

	FINLINE void GetFileName(			// Returns file name
		char *	pszFileNameRV)
	{
		f_strcpy( pszFileNameRV, m_szFileName);
	}

private:
	char	m_szFileName [F_PATH_MAX_SIZE];		// File's name.
};

/*===========================================================================
Desc:		The ServerLockObject is used to lock and unlock a particular
			object.
===========================================================================*/
class ServerLockObject : public F_Object
{
public:
	ServerLockObject();					// ServerLockObject Constructor

	virtual ~ServerLockObject()
	{
		if( m_pItemId)
		{
			m_pItemId->Release();
		}
	}

	void Setup(
		ServerLockManager *	pServerLockMgr,
		F_ItemId *				pItemId,
		FLMUINT					uiBucket);

	RCODE Lock(
		F_Db *				pDb,
		F_SEM					hWaitSem,
		FLMBOOL				bLogEvent,
		FLMBOOL				bSendSuspendEvent,	// Send suspend event when waiting
															// for lock.
		FLMBOOL				bExclLock,				// Exclusive or shared lock?
		FLMUINT				uiMaxWaitSecs,			// Maximum wait time in seconds.
		FLMINT				iPriority,				// Lock priority
		XFLM_DB_STATS *	pDbStats = NULL);		// Place to gather DB stats.

	RCODE Unlock(
		FLMBOOL				bLogEvent,
		F_Db *				pDb,						// used for event callbacks.
		FLMBOOL				bRelease = FALSE,		// Release object if no one is waiting
		XFLM_DB_STATS *	pDbStats = NULL);		// Place to gather DB stats.

	FLMINT Release(
		FLMBOOL	bMutexAlreadyLocked);
												// Decrement ref count, when it gets
												// down to 1, put it in the avail list.

	FLMINT FLMAPI Release( void)
	{
		return( Release( FALSE));
	}

	void RemoveWaiter(
		LOCK_WAITER *	pLockWaiter);

	FLMBOOL ThreadWaitingLock( void)
	{
		return( ((m_pFirstLockWaiter) ? (FLMBOOL)TRUE : (FLMBOOL)FALSE) );
	}

	FLMUINT LockCnt( void)
	{
		return( m_uiLockCnt);
	}

	void GetLockInfo(						// Returns lock information
		FLMINT				iPriority,	// A count of all lock requests with
												// a priority >= to this will be returned
												// in pLockInfo.
		eDbLockType *		peCurrLockType,
		FLMUINT *			puiThreadId,
		FLMUINT *			puiNumExclQueued,
		FLMUINT *			puiNumSharedQueued,
		FLMUINT *			puiPriorityCount);

	RCODE GetLockInfo(
		IF_LockInfoClient *	pLockInfo);

	FLMBOOL haveHigherPriorityWaiter(
		FLMINT			iPriority);

private:

	ServerLockManager *	m_pServerLockMgr;	// Server Lock Manager pointer.
	F_ItemId *				m_pItemId;			// ID for object this lock
														// object represents
	FLMUINT					m_uiLockThreadId;	// Thread of thread that has this
														// object locked.  Zero if none.
	FLMUINT					m_uiLockTime;		// Time lock was granted, if
														// exclusive lock.
	FLMUINT					m_uiLockCnt;		// Number of locks that have been
														// granted thus far.
	LOCK_WAITER *			m_pFirstLockWaiter;
														// Pointer to first in list of
														// lock waiters.
	LOCK_WAITER *			m_pLastLockWaiter;// Pointer to last in list of
														// lock waiters.
	FLMUINT					m_uiNumWaiters;	// Number of threads waiting.
	ServerLockObject *	m_pNext;				// Next in hash bucket or avail list
	ServerLockObject *	m_pPrev;				// Prev in hash bucket or avail list
	FLMUINT					m_uiSharedLockCnt;// Number of shared locks that have
														// been granted.
	FLMBOOL					m_bExclLock;		// Is the granted lock exclusive?
	FLMUINT					m_uiBucket;			// Hash bucket this object is in.
														// 0xFFFF means it is in avail list.
	F_TMSTAMP				m_StartTime;		// Time exclusive lock was grabbed.
	FLMBOOL					m_bStartTimeSet;	// Was m_StartTime set?

friend class ServerLockManager;

};

#endif	// FSRVLOCK_H
