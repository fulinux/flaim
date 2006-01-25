//-------------------------------------------------------------------------
// Desc:	Begin transaction
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
// $Id: fltrbeg.cpp 12266 2006-01-19 14:45:33 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE flmReadDictionary(
	FDB *			pDb);

/****************************************************************************
Desc:	This routine unlinks an FDB from a transaction's list of FDBs.
****************************************************************************/
void flmUnlinkDbFromTrans(
	FDB *			pDb,
	FLMBOOL		bCommitting)
{
	FFILE *	pFile = pDb->pFile;

	flmAssert( pDb->pIxdFixups == NULL);
	if( pDb->uiTransType != FLM_NO_TRANS)
	{
		if (pDb->uiFlags & FDB_HAS_WRITE_LOCK)
		{

			// If this is a commit operation and we have a commit callback,
			// call the callback function before unlocking the DIB.

			if( bCommitting && pDb->fnCommit)
			{
				FLMBOOL	bSavedInvisTrans;

				CB_ENTER( pDb, &bSavedInvisTrans);
				pDb->fnCommit( (HFDB)pDb, pDb->pvCommitData);
				CB_EXIT( pDb, bSavedInvisTrans);
			}

			dbUnlock( pDb);
		}

		f_mutexLock( gv_FlmSysData.hShareMutex);
		if (pDb->pDict)
		{
			flmUnlinkFdbFromDict( pDb);
		}

		/*
		Unlink the transaction from the FFILE if it is a read
		transaction.
		*/

		if( pDb->uiTransType == FLM_READ_TRANS)
		{
			if (pDb->pNextReadTrans)
			{
				pDb->pNextReadTrans->pPrevReadTrans = pDb->pPrevReadTrans;
			}
			else if (!pDb->uiKilledTime)
			{
				pFile->pLastReadTrans = pDb->pPrevReadTrans;
			}
			if (pDb->pPrevReadTrans)
			{
				pDb->pPrevReadTrans->pNextReadTrans = pDb->pNextReadTrans;
			}
			else if (pDb->uiKilledTime)
			{
				pFile->pFirstKilledTrans = pDb->pNextReadTrans;
			}
			else
			{
				pFile->pFirstReadTrans = pDb->pNextReadTrans;
			}

			// Zero out so it will be zero for next transaction begin.

			pDb->uiKilledTime = 0;
		}
		else
		{
			// Reset to NULL or zero for next update transaction.

			pDb->pBlobList = NULL;
			pDb->pIxStartList = pDb->pIxStopList = NULL;
			flmAssert( pDb->pIxdFixups == NULL);
		}

		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		pDb->uiTransType = FLM_NO_TRANS;
		pDb->uiFlags &= (~(FDB_UPDATED_DICTIONARY |
								FDB_INVISIBLE_TRANS |
								FDB_DONT_KILL_TRANS |
								FDB_DONT_POISON_CACHE));
	}
}

/****************************************************************************
Desc:	This routine reads a file's local dictionary.  This is called only
		when we did not have a dictionary off of the FFILE - which will be
		the first transaction after a database is opened.
****************************************************************************/
FSTATIC RCODE flmReadDictionary(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;

	// We better still be opening the database for the first time

	flmAssert( pDb->pFile->uiFlags & DBF_BEING_OPENED);

	if (RC_BAD( rc = fdictRebuild( pDb)))
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// At this point, we will not yet have opened the database for
	// general use, so there is no way that any other thread can have
	// created a dictionary yet.

	flmAssert( pDb->pFile->pDictList == NULL);

	// Link the new local dictionary to its file structure.

	flmLinkDictToFile( pDb->pFile, pDb->pDict);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit:
	if (RC_BAD( rc) && pDb->pDict)
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}
	return( rc);
}

/****************************************************************************
Desc:	This routine starts a transaction for the specified database.  The
		transaction may be part of an overall larger transaction.
Ret:	SUCCESS
			Indicates the transaction has been started for the desired
			database.
		FERR_NO_TRANS_ACTIVE
			No global transaction has begun.  This means that someone is
			trying to do some modifications outside of a transaction.
		other FLAIM error codes
****************************************************************************/
RCODE flmBeginDbTrans(
	FDB *			pDb,
	FLMUINT		uiTransType,		/* Type of transaction to start, if one
												has not been started. */
	FLMUINT		uiMaxLockWait,		/* Maximum number of seconds to wait for
												lock requests - if we have to create a
												transaction. */
	FLMUINT		uiFlags,				/* Transaction flags */
	FLMBYTE *	pucLogHdr
	)
{
	RCODE			rc = FERR_OK;
	FFILE *		pFile = pDb->pFile;
	FLMBOOL		bMutexLocked = FALSE;
	FLMBYTE *	pucLastCommittedLogHdr;
	DB_STATS *	pDbStats = pDb->pDbStats;

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	// Initialize a few things - as few as is necessary to avoid
	// unnecessary overhead.

	pDb->eAbortFuncId = FLM_UNKNOWN_FUNC;
	pDb->AbortRc = FERR_OK;
	pucLastCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	pDb->KrefCntrl.bKrefSetup = FALSE;
	pDb->uiTransType = uiTransType;
	pDb->uiThreadId = (FLMUINT)f_threadId();
	pDb->uiTransCount++;

	/*
	Link the FDB to the file's most current FDICT structure,
	if there is one.
	Also, if it is a read transaction, link the FDB
	into the list of read transactions off of
	the FFILE structure.
	*/

	// Lock the mutex.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;
	if (pFile->pDictList)
	{

		// Link the FDB to the FDICT.

		flmLinkFdbToDict( pDb, pFile->pDictList);
	}

	/*
	If it is a read transaction, link into the list of
	read transactions off of the FFILE structure.  Until we
	get the log header transaction ID below, we set uiCurrTransID
	to zero and link this transaction in at the beginning of the
	list.
	*/

	if (uiTransType == FLM_READ_TRANS)
	{
		flmGetLogHdrInfo( pucLastCommittedLogHdr, &pDb->LogHdr);

		// Link in at the end of the transaction list.

		pDb->pNextReadTrans = NULL;
		if ((pDb->pPrevReadTrans = pFile->pLastReadTrans) != NULL)
		{

			// Make sure transaction IDs are always in ascending order.  They
			// should be at this point.

			flmAssert( pFile->pLastReadTrans->LogHdr.uiCurrTransID <=
							pDb->LogHdr.uiCurrTransID);
			pFile->pLastReadTrans->pNextReadTrans = pDb;
		}
		else
		{
			pFile->pFirstReadTrans = pDb;
		}
		pFile->pLastReadTrans = pDb;
		pDb->uiInactiveTime = 0;

		if( uiFlags & FLM_DONT_KILL_TRANS)
		{
			pDb->uiFlags |= FDB_DONT_KILL_TRANS;
		}
		else
		{
			pDb->uiFlags &= ~FDB_DONT_KILL_TRANS;
		}
		if (pucLogHdr)
		{
			f_memcpy( pucLogHdr, &pDb->pFile->ucLastCommittedLogHdr[0],
						LOG_HEADER_SIZE);
		}
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;

	if( uiFlags & FLM_DONT_POISON_CACHE)
	{
		pDb->uiFlags |= FDB_DONT_POISON_CACHE;
	}
	else
	{
		pDb->uiFlags &= ~FDB_DONT_POISON_CACHE;
	}

	/*
	Put an exclusive lock on the database if we are not in a read
	transaction.  Read transactions require no lock.
	*/

	if (uiTransType != FLM_READ_TRANS)
	{
		flmAssert( pDb->pIxStats == NULL);

		// Set the bHadUpdOper to TRUE for all transactions to begin with.
		// Many calls to flmBeginDbTrans are internal, and we WANT the
		// normal behavior at the end of the transaction when it is
		// committed or aborted.  The only time this flag will be set
		// to FALSE is when the application starts the transaction as
		// opposed to an internal starting of the transaction.

		pDb->bHadUpdOper = TRUE;

		// Initialize the count of blocks changed to be 0

		pDb->uiBlkChangeCnt = 0;

		if (RC_BAD( rc = dbLock( pDb, uiMaxLockWait)))
		{
			goto Exit;
		}

		// If there was a problem with the RFL volume, we must wait
		// for a checkpoint to be completed before continuing.
		// The checkpoint thread looks at this same flag and forces
		// a checkpoint.  If it completes one successfully, it will
		// reset this flag.
		//
		// Also, if the last forced checkpoint had a problem
		// (pFile->CheckpointRc != FERR_OK), we don't want to
		// start up a new update transaction until it is resolved.

		if (!pFile->pRfl->seeIfRflVolumeOk() ||
			 RC_BAD( pFile->CheckpointRc))
		{
			rc = RC_SET( FERR_MUST_WAIT_CHECKPOINT);
			goto Exit;
		}

		// Set the first log block address to zero.

		pFile->uiFirstLogBlkAddress = 0;

		// Header must be read before opening roll forward log file to make
		// sure we have the most current log file and log options.

		f_memcpy( pFile->ucUncommittedLogHdr, pucLastCommittedLogHdr,
			LOG_HEADER_SIZE);
		flmGetLogHdrInfo( pucLastCommittedLogHdr, &pDb->LogHdr);

		/*
		Need to increment the current checkpoint for update transactions
		so that it will be correct when we go to mark cache blocks.
		*/

		if (pDb->uiFlags & FDB_REPLAYING_RFL)
		{
			// During recovery we need to set the transaction ID to the
			// transaction ID that was logged.

			pDb->LogHdr.uiCurrTransID = pFile->pRfl->getCurrTransID();
		}
		else
		{
			pDb->LogHdr.uiCurrTransID++;
		}
		f_mutexLock( gv_FlmSysData.hShareMutex);

		// Link FDB to the most current local dictionary, if there
		// is one.

		if (pFile->pDictList != pDb->pDict && pFile->pDictList)
		{
			flmLinkFdbToDict( pDb, pFile->pDictList);
		}
		pFile->uiUpdateTransID = pDb->LogHdr.uiCurrTransID;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);

		// Set the transaction EOF to the current file EOF

		pDb->uiTransEOF = pDb->LogHdr.uiLogicalEOF;

		// Put the transaction ID into the uncommitted log header.

		UD2FBA( (FLMUINT32)pDb->LogHdr.uiCurrTransID,
					&pFile->ucUncommittedLogHdr [LOG_CURR_TRANS_ID]);

		if (pucLogHdr)
		{
			f_memcpy( pucLogHdr, &pDb->pFile->ucUncommittedLogHdr [0],
							LOG_HEADER_SIZE);
		}
	}

	if (pDbStats)
	{
		f_timeGetTimeStamp( &pDb->TransStartTime);
	}

	// If we do not have a dictionary, read it in from disk.
	// NOTE: This should only happen when we are first opening
	// the database.

	if (!pDb->pDict)
	{
		if (RC_BAD( rc = flmReadDictionary( pDb)))
		{
			goto Exit;
		}
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if (uiTransType != FLM_READ_TRANS)
	{
		if (RC_OK( rc))
		{
			rc = pFile->pRfl->logBeginTransaction( pDb);
		}
#ifdef FLM_DBG_LOG
		flmDbgLogUpdate( pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
				0, 0, rc, "TBeg");
#endif
	}

	if( uiTransType == FLM_UPDATE_TRANS &&
		 gv_FlmSysData.EventHdrs [F_EVENT_UPDATES].pEventCBList)
	{
		flmTransEventCallback( F_EVENT_BEGIN_TRANS, (HFDB)pDb, rc,
					(FLMUINT)(RC_OK( rc)
								 ? pDb->LogHdr.uiCurrTransID
								 : (FLMUINT)0));
	}

	if (RC_BAD( rc))
	{
		// If there was an error, unlink the database from the transaction
		// structure as well as from the FDICT structure.

		flmUnlinkDbFromTrans( pDb, FALSE);

		if (pDb->pStats)
		{
			(void)flmStatUpdate( &gv_FlmSysData.Stats, &pDb->Stats);
		}
	}

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Starts a transaction.
*END************************************************************************/
RCODE 
	FlmDbTransBegin(
		HFDB				hDb,
			// [IN] Database handle.
		FLMUINT			uiTransType,
			// [IN] Specifies the type of transaction to begin.
			// Possible values are:
			//
			// FLM_READ_TRANS:  Begins a read transaction.
			// FLM_UPDATE_TRANS:  Begins an update transaction.
		FLMUINT			uiMaxLockWait,
			// [IN] Maximum lock wait time.  Specifies the amount of time
			// to wait for lock requests occuring during the transaction
			// to be granted.  Valid values are 0 through 255 seconds.  Zero
			// is used to specify no-wait locks.
		FLMBYTE *		pszHeader
			// [IN] 2K buffer
			// [OUT] Returns the first 2K of the file, including the current
			// version of the log header (from memory)
	)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bIgnore;
	FLMUINT		uiFlags = FLM_GET_TRANS_FLAGS( uiTransType);
	FDB *			pDb = (FDB *)hDb;

	uiTransType = FLM_GET_TRANS_TYPE( uiTransType);

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		FCL_WIRE		Wire( pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			if( RC_BAD( rc = Wire.doTransOp(
				FCS_OP_TRANSACTION_BEGIN, uiTransType, uiFlags,
				uiMaxLockWait, pszHeader)))
			{
				goto Exit;
			}
		}

		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// Verify the transaction type.

	if (( uiTransType != FLM_UPDATE_TRANS) &&
		 ( uiTransType != FLM_READ_TRANS))
	{
		rc = RC_SET( FERR_ILLEGAL_TRANS);
		goto Exit;
	}

	// Verify the transaction flags

	if( (uiFlags & FLM_DONT_KILL_TRANS) && uiTransType != FLM_READ_TRANS)
	{
		rc = RC_SET( FERR_ILLEGAL_TRANS);
		goto Exit;
	}

	// Can't start an update transaction on a database that
	// is locked in shared mode.

	if ((uiTransType == FLM_UPDATE_TRANS) &&
		 (pDb->uiFlags & FDB_FILE_LOCK_SHARED))
	{
		rc = RC_SET( FERR_PERMISSION);
		goto Exit;
	}

	// If the database has an invisible transaction going, abort it
	// before going any further - we don't want application transactions
	// to be nested under invisible transactions.  Application transactions
	// take precedence over invisible transactions.

	if ((pDb->uiTransType != FLM_NO_TRANS) &&
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		if (RC_BAD( rc = flmAbortDbTrans( pDb)))
		{
			goto Exit;
		}
	}

	// If the database is not running a transaction, start one.
	// Otherwise, start a nested transaction - first verifying that
	// the transation type matches.

	if (pDb->uiTransType == FLM_NO_TRANS)
	{
		FLMUINT		uiBytesRead;

		if( pszHeader)
		{
			if( RC_BAD( rc = pDb->pSFileHdl->ReadHeader( 
				0, 2048, pszHeader, &uiBytesRead)))
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = flmBeginDbTrans( pDb, uiTransType, 
			uiMaxLockWait, uiFlags,
			pszHeader ? &pszHeader [16] : NULL)))
		{
			goto Exit;
		}
		pDb->bHadUpdOper = FALSE;
	}
	else
	{
		// Cannot nest transactions.

		rc = RC_SET( FERR_TRANS_ACTIVE);
		goto Exit;
	}

Exit:

	flmExit( FLM_DB_TRANS_BEGIN, pDb, rc);
	return( rc);
}

/*API~***********************************************************************
Desc : Returns the type of the current database transaction.
*END************************************************************************/
RCODE FlmDbGetTransType(
	HFDB			hDb,
	FLMUINT *	puiTransTypeRV
	)
{
	RCODE		   rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT *	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pDb->pCSContext, pDb);

		// Send a request to get the transaction type.

		if( RC_BAD( rc = Wire.sendOp( 
			FCS_OPCLASS_TRANS, FCS_OP_TRANSACTION_GET_TYPE)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response.
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}
		*puiTransTypeRV = Wire.getTransType();
		rc = Wire.getRCode();
		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (!pDb)
	{
		rc = RC_SET( FERR_BAD_HDL);
		goto Exit;
	}

	fdbUseCheck( pDb);
	pDb->uiInitNestLevel++;
	(void)flmResetDiag( pDb);

	// If the transaction is an internal transaction that is invisible to
	// the application, return FLM_NO_TRANS.  Application is not supposed
	// see invisible transactions.

	*puiTransTypeRV = (FLMUINT)(((pDb->uiTransType == FLM_NO_TRANS) ||
										  (pDb->uiFlags & FDB_INVISIBLE_TRANS))
										 ? (FLMUINT)FLM_NO_TRANS
										 : pDb->uiTransType);

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

Exit:

	flmExit( FLM_DB_GET_TRANS_TYPE, pDb, rc);
	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Obtains a a lock on the database.
*END************************************************************************/
RCODE 
	FlmDbLock(
		HFDB				hDb,
			// [IN] Handle of database to be locked.
		FLOCK_TYPE		eLockType,
			// [IN] Type of lock request - must be FLM_LOCK_EXCLUSIVE or
			// FLM_LOCK_SHARED
		FLMINT			iPriority,
			// [IN] Priority to be assigned to lock.
		FLMUINT			uiTimeout
			// [IN] Seconds to wait for lock to be granted.  FLM_NO_TIMEOUT
			// means that it will wait forever for the lock to be granted.
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bIgnore;
	FDB *		pDb = (FDB *)hDb;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT_p		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_LOCK)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1, 
			(FLMUINT)eLockType)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_SIGNED_NUMBER, 
			0, iPriority)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiTimeout)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// eLockType better be exclusive or shared

	if ((eLockType != FLM_LOCK_EXCLUSIVE) && (eLockType != FLM_LOCK_SHARED))
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Nesting of locks is not allowed - this test also keeps this call from
	// being executed inside an update transaction that implicitly acquired
	// the lock.

	if (pDb->uiFlags &
			(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED | FDB_FILE_LOCK_IMPLICIT))
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Attempt to acquire the lock.

	if (RC_BAD( rc = pDb->pFile->pFileLockObj->Lock( TRUE, pDb, FALSE,
							(FLMBOOL)((eLockType == FLM_LOCK_EXCLUSIVE)
									  ? (FLMBOOL)TRUE
									  : (FLMBOOL)FALSE),
									  uiTimeout, iPriority,
									  pDb->pDbStats)))
	{
		goto Exit;
	}
	pDb->uiFlags |= FDB_HAS_FILE_LOCK;
	if (eLockType == FLM_LOCK_SHARED)
	{
		pDb->uiFlags |= FDB_FILE_LOCK_SHARED;
	}

Exit:

	flmExit( FLM_DB_LOCK, pDb, rc);
	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Releases a lock on the database
*END************************************************************************/
RCODE 
	FlmDbUnlock(
		HFDB				hDb
			// [IN] Handle of database to be unlocked.
	)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = (FDB *)hDb;
	FLMBOOL	bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT_p		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_UNLOCK)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK | FDB_CLOSING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If we don't have an explicit lock, can't do the unlock.  It is
	// also illegal to do the unlock during an update transaction.

	if (!(pDb->uiFlags & FDB_HAS_FILE_LOCK) ||
		 (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT) ||
		 (pDb->uiTransType == FLM_UPDATE_TRANS))
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	// Unlock the file.

	if (RC_BAD( rc = pDb->pFile->pFileLockObj->Unlock( TRUE, pDb)))
	{
		goto Exit;
	}

	// Unset the flags that indicated the file was explicitly locked.

	pDb->uiFlags &= (~(FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED));

Exit:

	if( RC_OK( rc))
	{
		rc = flmCheckDatabaseState( pDb);
	}

	flmExit( FLM_DB_UNLOCK, pDb, rc);
	return( rc);
}

/*API~***********************************************************************
Desc : Returns information about current and pending locks on the
		 database.
*END************************************************************************/
RCODE FlmDbGetLockInfo(
	HFDB				hDb,
	FLMINT			iPriority,
	FLOCK_INFO *	pLockInfo
	)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	FLMBOOL	bIgnore;

	if (IsInCSMode( hDb))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pDb = (FDB *)hDb;
	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	pDb->pFile->pFileLockObj->GetLockInfo( iPriority, pLockInfo);

Exit:

	flmExit( FLM_DB_GET_LOCK_INFO, pDb, rc);
	return( rc);
}

/*API~***********************************************************************
Desc : Returns information about the lock held by the specified database
		 handle.
*END************************************************************************/
RCODE FlmDbGetLockType(
	HFDB				hDb,
	FLOCK_TYPE *	peLockType,
	FLMBOOL *		pbImplicit)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = NULL;
	FLMBOOL	bIgnore;

	if( peLockType)
	{
		*peLockType = FLM_LOCK_NONE;
	}

	if( pbImplicit)
	{
		*pbImplicit = FALSE;
	}

	if (IsInCSMode( hDb))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pDb = (FDB *)hDb;
	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
									  FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	if( pDb->uiFlags & FDB_HAS_FILE_LOCK)
	{
		if( peLockType)
		{
			if( pDb->uiFlags & FDB_FILE_LOCK_SHARED)
			{
				*peLockType = FLM_LOCK_SHARED;
			}
			else
			{
				*peLockType = FLM_LOCK_EXCLUSIVE;
			}
		}
		
		if( pbImplicit)
		{
			*pbImplicit = (pDb->uiFlags & FDB_FILE_LOCK_IMPLICIT) 
								? TRUE 
								: FALSE;
		}
	}

Exit:

	flmExit( FLM_DB_GET_LOCK_TYPE, pDb, rc);
	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Forces a checkpoint on the database.
*END************************************************************************/
RCODE 
	FlmDbCheckpoint(
		HFDB		hDb,
			// [IN] Handle of database to perform the checkpoint on.
		FLMUINT	uiTimeout
			// [IN] Seconds to wait to obtain lock on the database.
			// FLM_NO_TIMEOUT means that it will wait forever for
			// the lock to be granted.
	)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = (FDB *)hDb;
	FLMBOOL	bStartedTrans;

	bStartedTrans = FALSE;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT_p		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_CHECKPOINT)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiTimeout)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	// Start an update transaction.  Must not already be one going.

	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
									  0, uiTimeout | FLM_AUTO_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// Commit the transaction, forcing it to be checkpointed.

	bStartedTrans = FALSE;
	pDb->bHadUpdOper = FALSE;
	if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, TRUE)))
	{
		goto Exit;
	}
Exit:
	if (bStartedTrans)
	{
		(void)flmAbortDbTrans( pDb);
	}

	flmExit( FLM_DB_CHECKPOINT, pDb, rc);
	return( rc);
}
