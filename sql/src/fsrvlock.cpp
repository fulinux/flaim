//------------------------------------------------------------------------------
// Desc:	Contains the methods for the ServerLockManager and
// 		ServerLockObject classes.
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
// $Id: fsrvlock.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

#define LOCK_HASH_ENTRIES	256

/****************************************************************************
Desc:
****************************************************************************/
ServerLockManager::~ServerLockManager()
{
	// Free everything in the avail lock list.  This is where all
	// of the lock objects should be at this point.

	if (m_hMutex != F_MUTEX_NULL)
	{
		lockMutex( FALSE);

		// Signal all pending lock waiters.

		CheckLockTimeouts( TRUE, TRUE);

		while (m_pAvailLockList)
		{
			ServerLockObject_p	pLockObject = m_pAvailLockList;

			UnlinkLockObject( pLockObject, FALSE);

			pLockObject->Release();
		}
		unlockMutex( FALSE);
		f_mutexDestroy( &m_hMutex);
	}

	// Free the hash table.

	f_free( &m_pHashTbl);
}

/****************************************************************************
Desc: Initializes the lock manager's hash table.
****************************************************************************/
RCODE ServerLockManager::SetupHashTbl()
{
	return( flmAllocHashTbl( LOCK_HASH_ENTRIES, &m_pHashTbl));
}

/****************************************************************************
Desc:	Finds the lock object for the passed in item identifier.
		If one is not found, one will be created.
****************************************************************************/
ServerLockObject_p ServerLockManager::GetLockObject(
	F_ItemId_p	pItemId)
{
	FLMUINT					uiBucket;
	ServerLockObject *	pLockObject;
	ServerLockObject *	pTmpLockObject;

	// Get the hash bucket.

	uiBucket = pItemId->GetHashBucket( m_pHashTbl, LOCK_HASH_ENTRIES);

	// See if the desired file is already in the hash bucket.

	lockMutex( FALSE);

	pLockObject = (ServerLockObject_p)m_pHashTbl [uiBucket].pFirstInBucket;

	// See if any of the objects match.

	while (pLockObject)
	{
		if (pItemId->IsEqual( pLockObject->m_pItemId))
		{
			goto Exit;
		}
		pLockObject = pLockObject->m_pNext;
	}

	// If we didn't find a matching object, allocate an object and link it into
	// the hash bucket.  Check to see if we have any in the avail list
	// first so that we don't have to allocate memory if we can avoid it.

	if ((pLockObject = m_pAvailLockList) != NULL)
	{
		UnlinkLockObject( pLockObject, FALSE);
	}
	else
	{
		if ((pLockObject = f_new ServerLockObject) == NULL)
		{
			goto Exit;
		}
	}

	// Setup the new object and put it into the hash bucket.

	pLockObject->Setup( this, pItemId, uiBucket);
	pTmpLockObject =
			(ServerLockObject_p)m_pHashTbl [uiBucket].pFirstInBucket;
	pLockObject->m_pPrev = NULL;
	pLockObject->m_pNext = pTmpLockObject;
	if (pTmpLockObject)
	{
		pTmpLockObject->m_pPrev = pLockObject;
	}
	m_pHashTbl [uiBucket].pFirstInBucket = pLockObject;

Exit:

	unlockMutex( FALSE);
	return( pLockObject);
}

/****************************************************************************
Desc:	Unlinks a lock object from whatever list it is in.  This routine
		assumes that the server lock manager's mutex is already locked.
****************************************************************************/
void ServerLockManager::UnlinkLockObject(
	ServerLockObject *	pLockObject,
	FLMBOOL					bPutInAvailList)
{
	ServerLockObject *	pTmpLockObject;
	FLMUINT					uiBucket;

	// If hash bucket 0xFFFF, unlink from the avail list.  Otherwise,
	// unlink from the hash bucket it is in.

	if ((uiBucket = pLockObject->m_uiBucket) == 0xFFFF)
	{
		if ((pTmpLockObject = pLockObject->m_pPrev) == NULL)
		{
			m_pAvailLockList = pLockObject->m_pNext;
		}
		else
		{
			pTmpLockObject->m_pNext = pLockObject->m_pNext;
		}
		m_uiNumAvail--;
	}
	else
	{
		if ((pTmpLockObject = pLockObject->m_pPrev) == NULL)
		{
			m_pHashTbl [uiBucket].pFirstInBucket = pLockObject->m_pNext;
		}
		else
		{
			pTmpLockObject->m_pNext = pLockObject->m_pNext;
		}
	}
	if ((pTmpLockObject = pLockObject->m_pNext) != NULL)
	{
		pTmpLockObject->m_pPrev = pLockObject->m_pPrev;
	}

	if (bPutInAvailList)
	{
		if (m_uiNumAvail >= 50)
		{
			flmAssert( getRefCount() == 1);
			pLockObject->Release();
		}
		else
		{
			pLockObject->Setup( this, NULL, 0xFFFF);
			if (m_pAvailLockList)
			{
				m_pAvailLockList->m_pPrev = pLockObject;
			}
			pLockObject->m_pPrev = NULL;
			pLockObject->m_pNext = m_pAvailLockList;
			m_pAvailLockList = pLockObject;
			m_uiNumAvail++;
		}
	}
}

/****************************************************************************
Desc: Checks for any pending lock requests that have timed out.
****************************************************************************/
void ServerLockManager::CheckLockTimeouts(
	FLMBOOL			bMutexAlreadyLocked,
	FLMBOOL			bTimeoutAll)
{
	FLMUINT			uiCurrTime;
	LOCK_WAITER_p	pLockWaiter;

	lockMutex( bMutexAlreadyLocked);
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();

	while ((m_pFirstLockWaiter) &&
			 ((bTimeoutAll) ||
			  ((m_pFirstLockWaiter->uiWaitTime) &&
			   (FLM_ELAPSED_TIME( uiCurrTime, m_pFirstLockWaiter->uiWaitStartTime) >=
						m_pFirstLockWaiter->uiWaitTime))))
	{
		// Sanity check

		flmAssert( m_pFirstLockWaiter->pPrevGlobal == NULL);

		// Lock waiter has timed out.

		pLockWaiter = m_pFirstLockWaiter;

		// Remove from global list and lock object's list

		RemoveWaiter( pLockWaiter);
		pLockWaiter->pLockObject->RemoveWaiter( pLockWaiter);

		// Tell the waiter that the lock request timed out.

		*(pLockWaiter->pRc) = RC_SET( NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT);
		f_semSignal( pLockWaiter->hESem);
	}

	unlockMutex( bMutexAlreadyLocked);
}


/****************************************************************************
Desc: Signal a lock waiter that has a matching thread id.
****************************************************************************/
void ServerLockManager::SignalLockWaiter(
	FLMUINT					uiThreadId)
{
	FLMUINT			uiCurrTime;
	LOCK_WAITER_p	pLockWaiter;
	LOCK_WAITER_p	pNextWaiter;

	lockMutex( FALSE);
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();

	for( pLockWaiter = m_pFirstLockWaiter;
		pLockWaiter;
		pLockWaiter = pNextWaiter)
	{
		pNextWaiter = pLockWaiter->pNextGlobal;

		if( pLockWaiter->uiThreadId == uiThreadId)
		{
			// Remove from global list and lock object's list

			RemoveWaiter( pLockWaiter);
			pLockWaiter->pLockObject->RemoveWaiter( pLockWaiter);

			// Tell the waiter that the lock request timed out.

			*(pLockWaiter->pRc) = RC_SET( NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT);
			f_semSignal( pLockWaiter->hESem);
			break;
		}
	}

	unlockMutex( FALSE);
	return;
}

/****************************************************************************
Desc:	Inserts a waiter into the global list of waiters, sorted by
		its end wait time.
		
		NOTE: This routine assumes that the lock manager's semaphore
		is already locked.
****************************************************************************/
void ServerLockManager::InsertWaiter(
	LOCK_WAITER_p	pLockWaiter)
{
	LOCK_WAITER_p	pPrevLockWaiter;

	// Determine where in the list this lock waiter should go.

	if ((pPrevLockWaiter = m_pFirstLockWaiter) != NULL)
	{
		FLMUINT	uiCurrTime = FLM_GET_TIMER();
		FLMUINT	uiElapTime;
		FLMUINT	uiTimeLeft;

		while (pPrevLockWaiter)
		{
			// Waiters with zero wait time go to end of list.
			// They never time out.

			if (!pPrevLockWaiter->uiWaitTime)
			{

				// Should go BEFORE the first zero waiter.

				pPrevLockWaiter = pPrevLockWaiter->pPrevGlobal;
				break;
			}
			else if (!pLockWaiter->uiWaitTime)
			{
				if (!pPrevLockWaiter->pNextGlobal)
				{
					break;
				}
				pPrevLockWaiter = pPrevLockWaiter->pNextGlobal;
			}
			else
			{
				// Determine how much time is left on the previous
				// lock waiter's timer.  If it is less than the
				// new lock waiter's wait time, the new lock waiter
				// should be inserted AFTER it.  Otherwise, the
				// new lock waiter should be inserted BEFORE it.

				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
										pPrevLockWaiter->uiWaitStartTime);
				if (uiElapTime >= pPrevLockWaiter->uiWaitTime)
				{
					uiTimeLeft = 0;
				}
				else
				{
					uiTimeLeft = pPrevLockWaiter->uiWaitTime - uiElapTime;
				}

				// New lock waiter will time out before previous lock
				// waiter - insert it BEFORE the previous lock waiter.

				if (pLockWaiter->uiWaitTime < uiTimeLeft)
				{
					pPrevLockWaiter = pPrevLockWaiter->pPrevGlobal;
					break;
				}
				else
				{
					if (!pPrevLockWaiter->pNextGlobal)
						break;
					pPrevLockWaiter = pPrevLockWaiter->pNextGlobal;
				}
			}
		}
	}

	// Insert into list AFTER pPrevLockWaiter.

	if ((pLockWaiter->pPrevGlobal = pPrevLockWaiter) != NULL)
	{
		if ((pLockWaiter->pNextGlobal = pPrevLockWaiter->pNextGlobal) != NULL)
		{
			pLockWaiter->pNextGlobal->pPrevGlobal = pLockWaiter;
		}
		pPrevLockWaiter->pNextGlobal = pLockWaiter;
	}
	else
	{
		if( (pLockWaiter->pNextGlobal = m_pFirstLockWaiter) != NULL)
		{
			m_pFirstLockWaiter->pPrevGlobal = pLockWaiter;
		}
		m_pFirstLockWaiter = pLockWaiter;
	}
}

/****************************************************************************
Desc: See if this item ID is equal to another F_ItemId.
****************************************************************************/
FLMBOOL FFileItemId::IsEqual(
	F_ItemId_p	pItemId)
{
	FFileItemId_p	pFFileItemId;
	RFileItemId_p	pRFileItemId;
	char				szName1 [F_FILENAME_SIZE];
	char				szName2 [F_FILENAME_SIZE];
	FLMUINT			uiItemType = pItemId->GetItemType();

	switch (uiItemType)
	{
		case FFILE_ITEM:
		case FFILE_TRANS_ITEM:
			if (m_uiItemType != uiItemType)
				return( FALSE);
			pFFileItemId = (FFileItemId_p)pItemId;

			/* First see if the FFILE pointers are the same. */

			if (pFFileItemId->getDatabase() ==
				this->getDatabase())
			{
				return( TRUE);
			}

			/* Next see if the file names are the same. */

			this->GetFileName( szName1);
			pFFileItemId->GetFileName( szName2);
#if !defined( FLM_UNIX)
			if (f_stricmp( szName1, szName2) == 0)
				return( TRUE);
#else
			if (f_strcmp( szName1, szName2) == 0)
				return( TRUE);
#endif
			break;
		case RFILE_ITEM:
		case RFILE_TRANS_ITEM:
			if ((uiItemType == RFILE_ITEM &&
					m_uiItemType != FFILE_ITEM) ||
				 (uiItemType == RFILE_TRANS_ITEM &&
					m_uiItemType != FFILE_TRANS_ITEM))
			{
				return( FALSE);
			}
			pRFileItemId = (RFileItemId_p)pItemId;

			/* See if the file names are the same. */

			this->GetFileName( szName1);
			pRFileItemId->GetFileName( szName2);
#if !defined( FLM_UNIX)
			if (f_stricmp( szName1, szName2) == 0)
				return( TRUE);
#else
			if (f_strcmp( szName1, szName2) == 0)
				return( TRUE);
#endif
			break;
		default:
			break;
	}
	return( FALSE);
}

/****************************************************************************
Desc: Get file name for this file item.
****************************************************************************/
void FFileItemId::GetFileName(
	char *	pszFileNameRV)
{
	// _ExtractFileName( m_pFile, pszFileNameRV);
	char		szTmpPath[ F_PATH_MAX_SIZE];

	(void)gv_SFlmSysData.pFileSystem->pathReduce( 
			m_pDatabase->m_pszDbPath, szTmpPath, pszFileNameRV);

	// Convert to upper case for consistency when hashing.

#if !defined( FLM_UNIX)
	while (*pszFileNameRV)
	{
		*pszFileNameRV = (char)f_toupper( *pszFileNameRV);
		pszFileNameRV++;
	}
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
RFileItemId::RFileItemId(
	char *		pszFileName,
	FLMBOOL		bTrans)
{
	char *	pszTmp = &m_szFileName [0];

	while (*pszFileName)
	{
		// Convert to uppercase for consistency when hashing.

#if !defined( FLM_UNIX)
		*pszFileName = (char)f_toupper( *pszFileName);
#else
		*pszTmp++ = *pszFileName;
#endif
		pszFileName++;
	}
	*pszTmp = 0;
	m_uiItemType = (FLMUINT)((bTrans)
									? (FLMUINT)RFILE_TRANS_ITEM
									: (FLMUINT)RFILE_ITEM);
}

/****************************************************************************
Desc: See if this item ID is equal to another F_ItemId.
****************************************************************************/
FLMBOOL RFileItemId::IsEqual(
	F_ItemId *	pItemId)
{
	FFileItemId *	pFFileItemId;
	RFileItemId *	pRFileItemId;
	char				szName1 [F_FILENAME_SIZE];
	char				szName2 [F_FILENAME_SIZE];
	FLMUINT			uiItemType = pItemId->GetItemType();

	switch (uiItemType)
	{
		case FFILE_ITEM:
		case FFILE_TRANS_ITEM:
			if ((uiItemType == FFILE_ITEM &&
					m_uiItemType != RFILE_ITEM) ||
				 (uiItemType == FFILE_TRANS_ITEM &&
					m_uiItemType != RFILE_TRANS_ITEM))
			{
				return( FALSE);
			}
			pFFileItemId = (FFileItemId_p)pItemId;

			/* See if the file names are the same. */

			this->GetFileName( szName1);
			pFFileItemId->GetFileName( szName2);
#if !defined( FLM_UNIX)
			if (f_stricmp( szName1, szName2) == 0)
				return( TRUE);
#else
			if (f_strcmp( szName1, szName2) == 0)
				return( TRUE);
#endif
			break;
		case RFILE_ITEM:
		case RFILE_TRANS_ITEM:
			if (m_uiItemType != uiItemType)
				return( FALSE);
			pRFileItemId = (RFileItemId_p)pItemId;

			/* See if the file names are the same. */

			this->GetFileName( szName1);
			pRFileItemId->GetFileName( szName2);
#if !defined( FLM_UNIX)
			if (f_stricmp( szName1, szName2) == 0)
				return( TRUE);
#else
			if (f_strcmp( szName1, szName2) == 0)
				return( TRUE);
#endif
			break;
		default:
			break;
	}
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
ServerLockObject::ServerLockObject()
{
	m_pServerLockMgr = NULL;
	m_pItemId = NULL;
	m_uiLockThreadId = 0;
	m_uiLockTime = 0;
	m_uiLockCnt = 0;
	m_pFirstLockWaiter =
	m_pLastLockWaiter = NULL;
	m_uiNumWaiters = 0;
	m_pNext = m_pPrev = NULL;
	m_uiSharedLockCnt = 0;
	m_bExclLock = FALSE;
	m_uiBucket = 0xFFFF;
	m_bStartTimeSet = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT ServerLockObject::Release(
	FLMBOOL	bMutexAlreadyLocked)
{
	FLMINT	iRefCnt = --m_refCnt;

	if( !iRefCnt)
	{
		delete this;
		goto Exit;
	}

	// When it is no longer pointed to from anything but the server lock
	// manager, put it into the avail list.

	if (iRefCnt == 1)
	{
		LOCK_WAITER *	pLockWaiter;

		// Signal all waiters that they cannot get the lock.

		m_pServerLockMgr->lockMutex( bMutexAlreadyLocked);
		while (m_pFirstLockWaiter)
		{
			pLockWaiter = m_pFirstLockWaiter;

			// Remove from global list and lock object's list

			RemoveWaiter( pLockWaiter);
			m_pServerLockMgr->RemoveWaiter( pLockWaiter);

			// Tell the waiter that the lock request timed out and signal
			// the thread to wake it up.

			*(pLockWaiter->pRc) = RC_SET( NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT);
			f_semSignal( pLockWaiter->hESem);
		}

		m_pServerLockMgr->UnlinkLockObject( this, TRUE);
		m_pServerLockMgr->unlockMutex( bMutexAlreadyLocked);
	}

Exit:

	return( iRefCnt);
}

/****************************************************************************
Desc: Initialize some data for the lock object.
****************************************************************************/
void ServerLockObject::Setup(
	ServerLockManager_p	pServerLockMgr,
	F_ItemId_p				pItemId,
	FLMUINT					uiBucket)
{
	m_pServerLockMgr = pServerLockMgr;
	if (m_pItemId)
	{
		m_pItemId->Release();
		m_pItemId = NULL;
	}
	if ((m_pItemId = pItemId) != NULL)
	{
		m_pItemId->AddRef();
	}
	m_uiBucket = uiBucket;
}

/****************************************************************************
Desc:	Removes a waiter from the list of waiters on this object.
		NOTE: This routine assumes that the lock manager's semaphore
		is already locked.
****************************************************************************/
void ServerLockObject::RemoveWaiter(
	LOCK_WAITER_p	pLockWaiter)
{
	if (pLockWaiter->pNext)
		pLockWaiter->pNext->pPrev = pLockWaiter->pPrev;
	else
		m_pLastLockWaiter = pLockWaiter->pPrev;

	if (pLockWaiter->pPrev)
		pLockWaiter->pPrev->pNext = pLockWaiter->pNext;
	else
		m_pFirstLockWaiter = pLockWaiter->pNext;
	flmAssert( m_uiNumWaiters > 0);
	m_uiNumWaiters--;
}

/****************************************************************************
Desc:	Lock this object.  If object is locked, wait the specified
		number of seconds.
****************************************************************************/
RCODE ServerLockObject::Lock(
	F_Db *				pDb,
	F_SEM					hWaitSem,
	FLMBOOL				bLogEvent,
	FLMBOOL				bSendSuspendEvent,	// Send suspend event, as opposed to
														// waiting event, when waiting.
	FLMBOOL				bExclReq,				// Exclusive or shared lock?
	FLMUINT				uiMaxWaitSecs,			// Maximum wait time in seconds.
	FLMINT				iPriority,				// Lock priority
	SFLM_DB_STATS *	pDbStats)				// Place to collect stats.
{
	RCODE				rc = NE_SFLM_OK;
	RCODE				TempRc;
	LOCK_WAITER		LockWait;
	FLMBOOL			bMutexLocked = FALSE;

	flmAssert( hWaitSem != F_SEM_NULL);
	
	m_pServerLockMgr->lockMutex( FALSE);
	bMutexLocked = TRUE;

	if ((m_pFirstLockWaiter) ||
		 (m_bExclLock) ||
		 ((bExclReq) && (m_uiSharedLockCnt)))
	{

		// Object is locked, wait to get lock.

		if (!uiMaxWaitSecs)
		{
			rc = RC_SET( NE_SFLM_DATABASE_LOCK_REQ_TIMEOUT);
			goto Exit;
		}

		// Set up to wait for the lock.

		f_memset( &LockWait, 0, sizeof( LockWait));
		LockWait.pLockObject = this;
		LockWait.hESem = hWaitSem;

		// Link into list of waiters on this object.

		if ((LockWait.pPrev = m_pLastLockWaiter) != NULL)
		{
			LockWait.pPrev->pNext = &LockWait;
		}
		else
		{
			m_pFirstLockWaiter = &LockWait;
		}
		m_pLastLockWaiter = &LockWait;
		m_uiNumWaiters++;

		LockWait.uiThreadId = f_threadId();
		LockWait.pRc = &rc;
		
		rc = RC_SET( NE_SFLM_FAILURE);
		
		LockWait.bExclReq = bExclReq;
		LockWait.iPriority = iPriority;
		LockWait.uiWaitStartTime = (FLMUINT)FLM_GET_TIMER();
		
		if (bExclReq && pDbStats)
		{
			f_timeGetTimeStamp( &LockWait.StartTime);
			LockWait.pDbStats = pDbStats;
		}
		
		if (uiMaxWaitSecs == SFLM_NO_TIMEOUT)
		{
			LockWait.uiWaitTime = 0;
		}
		else
		{
			LockWait.uiWaitTime = FLM_SECS_TO_TIMER_UNITS( uiMaxWaitSecs);
		}

		// Link to list of global waiters - ordered by end time.

		m_pServerLockMgr->InsertWaiter( &LockWait);

		m_pServerLockMgr->unlockMutex( FALSE);
		bMutexLocked = FALSE;

		// Do the event callback, if any registered.

		if (bLogEvent &&
			 gv_SFlmSysData.EventHdrs [SFLM_EVENT_LOCKS].pEventCBList)
		{
			flmDoEventCallback( SFLM_EVENT_LOCKS,
						(eEventType)((bSendSuspendEvent)
										 ? SFLM_EVENT_LOCK_SUSPENDED
										 : SFLM_EVENT_LOCK_WAITING),
						pDb, LockWait.uiThreadId,
						0, 0, 0, NE_SFLM_OK);
		}

		// Now just wait to be signaled.

		if (RC_BAD( TempRc = f_semWait( LockWait.hESem, F_SEM_WAITFOREVER)))
		{
			RC_UNEXPECTED_ASSERT( TempRc);
			rc = TempRc;
		}
		else
		{
			// Process that signaled us better set the rc to something
			// besides NE_SFLM_FAILURE.

			if (rc == NE_SFLM_FAILURE)
			{
				RC_UNEXPECTED_ASSERT( rc);
			}
		}

		// Do the event callback, if any registered.

		if (bLogEvent &&
			 gv_SFlmSysData.EventHdrs [SFLM_EVENT_LOCKS].pEventCBList)
		{
			if (RC_BAD( rc))
			{
				flmDoEventCallback( SFLM_EVENT_LOCKS,
							SFLM_EVENT_LOCK_TIMEOUT,
							pDb, LockWait.uiThreadId,
							0, 0, 0, NE_SFLM_OK);
			}
			else
			{
				flmDoEventCallback( SFLM_EVENT_LOCKS,
							(eEventType)((bSendSuspendEvent)
											 ? SFLM_EVENT_LOCK_RESUMED
											 : SFLM_EVENT_LOCK_GRANTED),
							pDb, LockWait.uiThreadId,
							0, 0, 0, NE_SFLM_OK);
			}
		}
	}
	else
	{

		// Object is NOT locked in a conflicting mode.  Grant the
		// lock immediately.

		m_uiLockThreadId = f_threadId();
		m_bExclLock = bExclReq;
		if (!bExclReq)
		{
			m_uiSharedLockCnt++;
		}
		else
		{
			m_uiLockTime = (FLMUINT)FLM_GET_TIMER();
			flmAssert( m_uiSharedLockCnt == 0);

			// Take care of statistics gathering.

			if (pDbStats)
			{

				// If m_bStartTimeSet is TRUE, we started the
				// clock the last time nobody had the exclusive
				// lock, so we need to sum up idle time now.

				if (m_bStartTimeSet)
				{
					flmAddElapTime( &m_StartTime, &pDbStats->NoLocks.ui64ElapMilli);
					pDbStats->NoLocks.ui64Count++;
				}

				// Restart the clock for this locker.

				f_timeGetTimeStamp( &m_StartTime);
				m_bStartTimeSet = TRUE;
			}
			else
			{
				m_bStartTimeSet = FALSE;
			}
		}

		// Do the event callback, if any registered.

		if (bLogEvent &&
			 !bSendSuspendEvent &&
			 gv_SFlmSysData.EventHdrs [SFLM_EVENT_LOCKS].pEventCBList)
		{
			m_pServerLockMgr->unlockMutex( FALSE);
			bMutexLocked = FALSE;
			flmDoEventCallback( SFLM_EVENT_LOCKS,
						SFLM_EVENT_LOCK_GRANTED,
						pDb, m_uiLockThreadId,
						0, 0, 0, NE_SFLM_OK);
		}
	}
	
Exit:

	if (RC_OK( rc))
	{
		m_uiLockCnt++;
	}
	
	if (bMutexLocked)
	{
		m_pServerLockMgr->unlockMutex( FALSE);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Unlock this object.  If there is a pending lock request, give
		the lock to the next waiter.
****************************************************************************/
RCODE ServerLockObject::Unlock(
	FLMBOOL				bLogEvent,
	F_Db *				pDb,
	FLMBOOL				bRelease,
	SFLM_DB_STATS *	pDbStats)
{
	RCODE				rc = NE_SFLM_OK;
	F_SEM				hESem;
	LOCK_WAITER_p	pLockWaiter;

	m_pServerLockMgr->lockMutex( FALSE);

	if (m_bExclLock)
	{
		flmAssert( m_uiSharedLockCnt == 0);
		m_bExclLock = FALSE;

		// Record how long the lock was held, if we were tracking
		// it.

		if (pDbStats && m_bStartTimeSet)
		{
			flmAddElapTime( &m_StartTime, &pDbStats->HeldLock.ui64ElapMilli);
			pDbStats->HeldLock.ui64Count++;
		}
		m_bStartTimeSet = FALSE;
	}
	else
	{
		flmAssert( m_uiSharedLockCnt > 0);
		m_uiSharedLockCnt--;
	}

	// Do the event callback, if any registered.
	// NOTE: flmDoEventCallback locks the event mutex.
	// Since we are inside the lock manager's mutex lock,
	// the callback should not do ANYTHING that would cause
	// us to end up in here again!

	if (bLogEvent &&
		 gv_SFlmSysData.EventHdrs [SFLM_EVENT_LOCKS].pEventCBList)
	{
		flmDoEventCallback( SFLM_EVENT_LOCKS,
					SFLM_EVENT_LOCK_RELEASED,
					pDb, m_uiLockThreadId,
					0, 0, 0, NE_SFLM_OK);
	}

	m_uiLockThreadId = 0;

	/* See if we need to signal the next set of waiters. */

	if (m_pFirstLockWaiter && !m_uiSharedLockCnt)
	{
		m_bExclLock = m_pFirstLockWaiter->bExclReq;
		while (m_pFirstLockWaiter)
		{
			if (!m_bExclLock)
			{
				m_uiSharedLockCnt++;
			}


			pLockWaiter = m_pFirstLockWaiter;
			hESem = pLockWaiter->hESem;

			// Unlink the waiter from the list of waiters on this lock
			// object and then from the global list of waiters.
			// IMPORTANT NOTE: Do NOT signal the semaphore until AFTER
			// doing this unlinking.  This is because LOCK_WAITER
			// structures exist only on the stack of the thread
			// being signaled.  If we tried to assign m_pFirstLockWaiter after
			// signaling the semaphore, the LOCK_WAITER structure could
			// disappear and m_pFirstLockWaiter would get garbage.

			RemoveWaiter( pLockWaiter);
			m_pServerLockMgr->RemoveWaiter( pLockWaiter);

			// Update statistics for the waiter.

			if (pLockWaiter->pDbStats)
			{
				flmAddElapTime( &pLockWaiter->StartTime,
								&pLockWaiter->pDbStats->WaitingForLock.ui64ElapMilli);
				pLockWaiter->pDbStats->WaitingForLock.ui64Count++;
			}

			// Grant the lock to this waiter and signal the thread
			// to wake it up.

			m_uiLockThreadId = pLockWaiter->uiThreadId;
			if (m_bExclLock)
			{
				m_uiLockTime = (FLMUINT)FLM_GET_TIMER();

				// Restart the stats timer

				if (pDbStats)
				{
					m_bStartTimeSet = TRUE;
					f_timeGetTimeStamp( &m_StartTime);
				}
			}

			*(pLockWaiter->pRc) = NE_SFLM_OK;

			f_semSignal( hESem);

			// If the next waiter is not a shared lock request or
			// the lock that was granted was exclusive, we stop
			// here.

			if (m_bExclLock ||
				 (m_pFirstLockWaiter && m_pFirstLockWaiter->bExclReq))
			{
				break;
			}
		}
	}
	else if (bRelease &&
				!m_pFirstLockWaiter &&	// No one is wating for the object
				!m_uiSharedLockCnt)		// No one has the object locked
	{
		// Release the object.  If the reference count drops to 1,
		// the object will be put in the avail list.  The caller
		// should have performed an AddRef() on the object at
		// some point prior to calling this method.  Once this routine
		// returns the caller should not attempt further access of the object.
		Release( TRUE);
		bRelease = FALSE;
	}

	// Start timer, if not already running.  If the timer is not set at
	// this point, it will be because nobody has been granted the exclusive
	// lock.  If someone was granted the exclusive lock, the timer would
	// have been started above.  We start it here so we can track idle
	// time.

	if (pDbStats && !m_bStartTimeSet && !bRelease)
	{
		flmAssert( !m_bExclLock);
		m_bStartTimeSet = TRUE;
		f_timeGetTimeStamp( &m_StartTime);
	}

	// If we get to this point and bRelease is still TRUE, someone was
	// waiting to acquire the lock or there is still a shared lock
	// count.

	if( bRelease)
	{
		// All lock waiters should have done an AddRef on the lock object.
		// At this point we should still have our reference to the object,
		// the lock manager's reference, and a reference from at least one
		// waiter (that may have been granted above).
		flmAssert( m_refCnt >= 3);
		m_refCnt--;
	}

	m_pServerLockMgr->unlockMutex( FALSE);
	return( rc);
}

/****************************************************************************
Desc: Returns information about the pending lock requests.
****************************************************************************/
void ServerLockObject::GetLockInfo(
	FLMINT			iPriority,
	eDbLockType *	peCurrLockType,
	FLMUINT *		puiThreadId,
	FLMUINT *		puiNumExclQueued,
	FLMUINT *		puiNumSharedQueued,
	FLMUINT *		puiPriorityCount)
{
	LOCK_WAITER_p	pLockWaiter;

	*puiNumExclQueued = 0;
	*puiNumSharedQueued = 0;
	*puiPriorityCount = 0;

	m_pServerLockMgr->lockMutex( FALSE);

	// Get the type of lock, if any.

	if (m_bExclLock)
	{
		*peCurrLockType = SFLM_LOCK_EXCLUSIVE;
		*puiThreadId = m_uiLockThreadId;
	}
	else if (m_uiSharedLockCnt)
	{
		*peCurrLockType = SFLM_LOCK_SHARED;
		*puiThreadId = 0;
	}
	else
	{
		*peCurrLockType = SFLM_LOCK_NONE;
		*puiThreadId = 0;
	}

	// Get information on pending lock requests.

	pLockWaiter = m_pFirstLockWaiter;
	for ( ; pLockWaiter; pLockWaiter = pLockWaiter->pNext )
	{

		// Count the number of exclusive and shared waiters.

		if (pLockWaiter->bExclReq)
		{
			(*puiNumExclQueued)++;
		}
		else
		{
			(*puiNumSharedQueued)++;
		}

		// Count the number of waiters at or above input priority.

		if (pLockWaiter->iPriority >= iPriority)
		{
			(*puiPriorityCount)++;
		}
	}

	m_pServerLockMgr->unlockMutex( FALSE);
}

/****************************************************************************
Desc:	Return the lock waiters for this object.
****************************************************************************/
RCODE ServerLockObject::GetLockInfo(
	IF_LockInfoClient *	pLockInfo)
{
	RCODE				rc = NE_SFLM_OK;
	LOCK_WAITER *	pLockWaiter;
	FLMUINT			uiCnt;
	FLMUINT			uiElapTime;
	FLMUINT			uiCurrTime;
	FLMUINT			uiMilli;

	m_pServerLockMgr->lockMutex( FALSE);
	uiCurrTime = (FLMUINT)FLM_GET_TIMER();

	if (!m_uiNumWaiters && !m_uiLockThreadId)
	{
		pLockInfo->setLockCount( 0);
		goto Exit;
	}
	uiCnt = m_uiNumWaiters + 1;	// Add one for lock holder.
	if( pLockInfo->setLockCount( uiCnt) == FALSE)
	{
		goto Exit;
	}

	// Output the lock holder first.

	uiElapTime = FLM_ELAPSED_TIME( uiCurrTime, m_uiLockTime);
	uiMilli = FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
	if( pLockInfo->addLockInfo( 0, m_uiLockThreadId, uiMilli) == FALSE)
	{
		goto Exit;
	}
	uiCnt--;

	// Output the lock waiters.

	pLockWaiter = m_pFirstLockWaiter;
	while( pLockWaiter && uiCnt)
	{
		uiElapTime = FLM_ELAPSED_TIME( uiCurrTime, pLockWaiter->uiWaitStartTime);
		uiMilli = FLM_TIMER_UNITS_TO_MILLI( uiElapTime);
		if( pLockInfo->addLockInfo( (m_uiNumWaiters - uiCnt) + 1,
			pLockWaiter->uiThreadId, uiMilli) == FALSE)
		{
			goto Exit;
		}
		pLockWaiter = pLockWaiter->pNext;
		uiCnt--;
	}
	flmAssert( pLockWaiter == NULL && uiCnt == 0);

Exit:

	m_pServerLockMgr->unlockMutex( FALSE);
	return( rc);
}

/****************************************************************************
Desc: Returns TRUE if there are lock waiters with a priority > iPriority
****************************************************************************/
FLMBOOL ServerLockObject::haveHigherPriorityWaiter(
	FLMINT			iPriority)
{
	LOCK_WAITER_p	pLockWaiter;
	FLMBOOL			bWaiters = FALSE;

	m_pServerLockMgr->lockMutex( FALSE);

	pLockWaiter = m_pFirstLockWaiter;
	for ( ; pLockWaiter; pLockWaiter = pLockWaiter->pNext )
	{
		// If we find a waiter with a priority > the specified
		// priority, we're done.

		if (pLockWaiter->iPriority > iPriority)
		{
			bWaiters = TRUE;
			break;
		}
	}

	m_pServerLockMgr->unlockMutex( FALSE);
	return( bWaiters);
}
