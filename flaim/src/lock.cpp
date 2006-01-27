//-------------------------------------------------------------------------
// Desc:	Database locking and unlocking.
// Tabs:	3
//
//		Copyright (c) 1991,1994-2006 Novell, Inc. All Rights Reserved.
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
// $Id: lock.cpp 12315 2006-01-19 15:16:37 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	This routine locks a database for exclusive access.
Ret:	FERR_OK		- Indicates that the database was successfully locked.
		FCERR_LOCK	-	Error locking database file.
****************************************************************************/
RCODE dbLock(
	FDB *		pDb,
	FLMUINT	uiMaxLockWait)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bGotFileLock = FALSE;
	FFILE *		pFile = pDb->pFile;

	// There must NOT be a shared lock on the file.

	if (pDb->uiFlags & FDB_FILE_LOCK_SHARED)
	{
		rc = RC_SET( FERR_PERMISSION);
		goto Exit;
	}

	// Must acquire an exclusive file lock first, if it hasn't been
	// acquired.

	if (!(pDb->uiFlags & FDB_HAS_FILE_LOCK))
	{
		if (RC_BAD( rc = pFile->pFileLockObj->Lock( TRUE, pDb, FALSE, TRUE,
									uiMaxLockWait, 0, pDb->pDbStats)))
		{
			goto Exit;
		}
		bGotFileLock = TRUE;
		pDb->uiFlags |= (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT);
	}

	if( RC_OK( rc = dbWriteLock( pFile, pDb->pDbStats)))
	{
		pDb->uiFlags |= FDB_HAS_WRITE_LOCK;
	}

Exit:

	if (rc == FERR_IO_FILE_LOCK_ERR)
	{
		if (bGotFileLock)
		{
			(void)pFile->pFileLockObj->Unlock( TRUE, pDb);
			pDb->uiFlags &= (~(FDB_HAS_FILE_LOCK | 
				FDB_FILE_LOCK_IMPLICIT | FDB_HAS_WRITE_LOCK));
		}

		if (pDb->uiTransType != FLM_NO_TRANS)
		{

			// Unlink the DB from the transaction.

			(void)flmUnlinkDbFromTrans( pDb, FALSE);
		}
	}
	else if (RC_BAD( rc))
	{
		if (bGotFileLock)
		{
			(void)pFile->pFileLockObj->Unlock( TRUE, pDb);
			pDb->uiFlags &= (~(FDB_HAS_FILE_LOCK | 
				FDB_FILE_LOCK_IMPLICIT | FDB_HAS_WRITE_LOCK));
		}
	}

	return( rc);
}

/****************************************************************************
Desc:	This routine unlocks a database that was previously locked
		using the dbLock routine.
Ret:	FERR_OK		- 	Indicates that the database was
							successfully unlocked.
		FERR_UNLOCK	-	Error unlocking file.
****************************************************************************/
RCODE dbUnlock(
	FDB *			pDb)
{
	RCODE	rc = FERR_OK;

	// If we have the write lock, unlock it first.

	flmAssert( pDb->uiFlags & FDB_HAS_WRITE_LOCK);

	dbWriteUnlock( pDb->pFile, pDb->pDbStats);
	pDb->uiFlags &= ~FDB_HAS_WRITE_LOCK;

	// Give up the file lock, if it was acquired implicitly.
	
	if (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT)
	{
		if (RC_OK( rc = pDb->pFile->pFileLockObj->Unlock( TRUE, pDb)))
		{
			pDb->uiFlags &=
				(~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_IMPLICIT));
		}
	}

// Exit:
	return( rc);
}
