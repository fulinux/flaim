//-------------------------------------------------------------------------
// Desc:	Database locking class - definitions.
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
// $Id: fsrvlock.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FSRVLOCK_H
#define FSRVLOCK_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

class ServerLockManager;
class ServerLockObject;
class F_ItemId;
class FFileItemId;
class RFileItemId;

typedef ServerLockManager *	ServerLockManager_p;
typedef ServerLockObject *		ServerLockObject_p;
typedef F_ItemId *				F_ItemId_p;
typedef FFileItemId *			FFileItemId_p;
typedef RFileItemId *			RFileItemId_p;

/**************************************************************************
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
	DB_STATS *				pDbStats;		// Statistics to update.
	LOCK_WAITER_p			pNext;			// Next lock waiter in list.
	LOCK_WAITER_p			pPrev;			// Previous lock waiter in list.
	LOCK_WAITER_p			pNextGlobal;	// Next lock waiter in global list
													// that is ordered according to
													// udWaitEndTime.
	LOCK_WAITER_p			pPrevGlobal;	// Previous lock waiter in global list
} LOCK_WAITER;

/****************************************************************************
Desc:	The ServerLockManager class manages ServerLockObject objects. 
****************************************************************************/
class ServerLockManager : public F_Base
{
public:

	FINLINE ServerLockManager(
		F_MUTEX	* 				phMutex)		
	{
		m_phMutex = phMutex;
		m_pFirstLockWaiter = NULL;
		m_pHashTbl = NULL;
		m_uiNumAvail = 0;
		m_pAvailLockList = NULL;
	}

	virtual ~ServerLockManager();

	FINLINE void SetMutexPtr(
		F_MUTEX	*				phMutex)
	{
		m_phMutex = phMutex;
	}

	RCODE SetupHashTbl( void);

	FINLINE void CheckLockTimeouts(
		FLMBOOL					bTimeoutAll)
	{
		F_MutexRef		MutexRef( m_phMutex);

		CheckLockTimeouts( &MutexRef, bTimeoutAll);
	}

	void InsertWaiter(
		LOCK_WAITER *			pLockWaiter);

	FINLINE void RemoveWaiter(
		LOCK_WAITER_p			pLockWaiter)
	{
		if (pLockWaiter->pNextGlobal)
		{
			pLockWaiter->pNextGlobal->pPrevGlobal = pLockWaiter->pPrevGlobal;
		}

		if (pLockWaiter->pPrevGlobal)
		{
			pLockWaiter->pPrevGlobal->pNextGlobal = pLockWaiter->pNextGlobal;
		}
		else
		{
			m_pFirstLockWaiter = pLockWaiter->pNextGlobal;
		}
	}

	ServerLockObject_p GetLockObject(
		F_ItemId *				pItemId);

	void SignalLockWaiter(
		FLMUINT					uiThreadId);

	FINLINE void UnlinkLockObject(
		ServerLockObject *	pLockObject,
		FLMBOOL					bPutInAvailList)
	{
		F_MutexRef	MutexRef( m_phMutex);

		UnlinkLockObject( pLockObject, bPutInAvailList, &MutexRef);
	}

	void UnlinkLockObject(
		ServerLockObject_p	pLockObject,
		FLMBOOL					bPutInAvailList,
		F_MutexRef *			pMutexRef);

	F_MUTEX * GetSemPtr( void)
	{
		return( m_phMutex);
	}

private:

	F_MUTEX *					m_phMutex;
	FBUCKET_p					m_pHashTbl;
	LOCK_WAITER *				m_pFirstLockWaiter;
	FLMUINT						m_uiNumAvail;
	ServerLockObject *		m_pAvailLockList;

	void CheckLockTimeouts(
		F_MutexRef *			pMutexRef,
		FLMBOOL					bTimeoutAll);

friend class F_ServerLockMgrPage;
};

/****************************************************************************
Desc:	The item id that identifies a particular object.
****************************************************************************/
class F_ItemId : public F_Base
{
public:

	FINLINE F_ItemId()
	{
		m_uiItemType = 0;
	}

	virtual ~F_ItemId()
	{
	}

	virtual FLMBOOL IsEqual(
		F_ItemId *				pItemId) = 0;

	virtual FLMUINT GetHashBucket(
		FBUCKET_p				pHashTbl,
		FLMUINT					uiHashTblSize) = 0;

	FINLINE FLMUINT GetItemType( void)
	{
		return( m_uiItemType);
	}

protected:

	FLMUINT	m_uiItemType;
	
#define FFILE_ITEM			1
#define RFILE_ITEM			2
#define FFILE_TRANS_ITEM	3
#define RFILE_TRANS_ITEM	4
};

/****************************************************************************
Desc:	The item id that identifies an FFILE object.
****************************************************************************/
class FFileItemId : public F_ItemId
{
public:

	FINLINE FFileItemId(
		FFILE_p			pFile,
		FLMBOOL			bTrans)
	{
		m_pFile = pFile;
		m_uiItemType = (FLMUINT)((bTrans)
										? (FLMUINT)FFILE_TRANS_ITEM
										: (FLMUINT)FFILE_ITEM);
	}

	virtual ~FFileItemId()
	{
	}

	FLMBOOL IsEqual(
		F_ItemId *		pItemId);

	FINLINE FLMUINT GetHashBucket(
		FBUCKET_p		pHashTbl,
		FLMUINT			uiHashTblSize)
	{
		char		szFileName[ F_PATH_MAX_SIZE];

		// Extract the file name

		this->GetFileName( szFileName);

		// Determine what hash bucket the file should be in - based on file name.

		return( flmStrHashBucket( szFileName, pHashTbl, uiHashTblSize));
	}

	FINLINE FFILE * GetFilePtr( void)
	{
		return( m_pFile);
	}

	void GetFileName(
		char *			pszFileNameRV);

private:

	FFILE_p				m_pFile;	
};

/****************************************************************************
Desc:	The item id that identifies a file being used by rebuild.
****************************************************************************/
class RFileItemId : public F_ItemId
{
public:
	RFileItemId(
		FLMBYTE *		pszFileName,
		FLMBOOL			bTrans = FALSE);

	virtual ~RFileItemId()
	{
	}

	FLMBOOL IsEqual(
		F_ItemId *		pItemId);

	FINLINE FLMUINT GetHashBucket(
		FBUCKET_p		pHashTbl,
		FLMUINT			uiHashTblSize)
	{
		return( flmStrHashBucket( m_szFileName, pHashTbl, uiHashTblSize));
	}
	
	FINLINE void GetFileName(
		char *			pszFileNameRV)
	{
		f_strcpy( pszFileNameRV, m_szFileName);
	}

private:

	char				m_szFileName[ F_PATH_MAX_SIZE];
};

/****************************************************************************
Desc:	The ServerLockObject is used to lock and unlock a particular
		object.
****************************************************************************/
class ServerLockObject : public F_Base
{
public:

	ServerLockObject();

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
		FLMBOOL					bLogEvent,
		FDB_p						pDb,
		FLMBOOL					bSendSuspendEvent,
		FLMBOOL					bExclLock,
		FLMUINT					uiMaxWaitSecs,
		FLMINT					iPriority,
		DB_STATS *				pDbStats = NULL);

	RCODE Unlock(
		FLMBOOL					bLogEvent,
		FDB_p						pDb,
		FLMBOOL					bRelease = FALSE,
		DB_STATS *				pDbStats = NULL);
		
	FLMUINT Release(
		F_MutexRef *			pMutexRef);

	FINLINE FLMUINT Release( void)
	{
		return( Release( NULL));
	}

	FINLINE F_ItemId * GetItemIdPtr( void)
	{
		return( m_pItemId);
	}

	FINLINE FLMUINT GetHashBucket( void)
	{
		return m_uiBucket;
	}

	FINLINE ServerLockObject * GetNext( void)
	{
		return( m_pNext);
	}

	FINLINE void SetNext(
		ServerLockObject *	pNext)
	{
		m_pNext = pNext;
	}

	FINLINE ServerLockObject * GetPrev( void)
	{
		return( m_pPrev);
	}

	FINLINE void SetPrev(
		ServerLockObject *	pPrev)
	{
		m_pPrev = pPrev;
	}

	void RemoveWaiter(
		LOCK_WAITER *			pLockWaiter);

	FINLINE ServerLockManager * GetLockManager( void)
	{
		return( m_pServerLockMgr);
	}

	FINLINE FLMBOOL ThreadWaitingLock( void)
	{
		return( ((m_pFirstLockWaiter) ? (FLMBOOL)TRUE : (FLMBOOL)FALSE) );
	}

	FINLINE FLMUINT LockCnt( void)
	{
		return( m_uiLockCnt);
	}

	void GetLockInfo(
		FLMINT					iPriority,
		FLOCK_INFO *			pLockInfo);

	RCODE GetLockInfo(
		FLMBOOL					bGetWaiters,
		void *					pvLockUsers);

	RCODE GetLockInfo(
		FlmLockInfo *			pLockInfo);

	FLMBOOL haveHigherPriorityWaiter(
		FLMINT					iPriority);

private:

	ServerLockManager *		m_pServerLockMgr;
	F_ItemId *					m_pItemId;
	FLMUINT						m_uiLockThreadId;
	FLMUINT						m_uiLockTime;
	FLMUINT						m_uiLockCnt;
	LOCK_WAITER *				m_pFirstLockWaiter;
	LOCK_WAITER *				m_pLastLockWaiter;
	FLMUINT						m_uiNumWaiters;
	ServerLockObject *		m_pNext;
	ServerLockObject *		m_pPrev;
	FLMUINT						m_uiSharedLockCnt;
	FLMBOOL						m_bExclLock;
	FLMUINT						m_uiBucket;
	F_TMSTAMP					m_StartTime;
	FLMBOOL						m_bStartTimeSet;
};

#include "fpackoff.h"

#endif
