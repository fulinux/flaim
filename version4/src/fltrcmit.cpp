//-------------------------------------------------------------------------
// Desc:	Commit transaction
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fltrcmit.cpp 12266 2006-01-19 14:45:33 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	This routine commits an active transaction for a particular
		database.  If the database is open via a server, a message is
		sent to the server to commit the transaction.  Otherwise, the
		transaction is committed locally.
****************************************************************************/
RCODE flmCommitDbTrans(
	FDB *			pDb,
	FLMUINT		uiNewLogicalEOF,		// New logical end-of-file.  This is only
												// set by the FlmDbReduceSize function when
												// it is truncating the file.
	FLMBOOL		bForceCheckpoint,		// Force a checkpoint?
	FLMBOOL *	pbEmpty					// May be NULL
	)
{
	RCODE	  			rc = FERR_OK;
	FLMBYTE *		pucUncommittedLogHdr;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiCPFileNum = 0;
	FLMUINT			uiCPOffset = 0;
	FLMUINT			uiTransId = 0;
	FLMBOOL			bTransEndLogged;
	FLMBOOL			bForceCloseOnError = FALSE;
	FLMBOOL			bOkToLogAbort = TRUE;
	DB_STATS *		pDbStats = pDb->pDbStats;
	FLMUINT			uiTransType;
	FLMBOOL			bInvisibleTrans = FALSE;
	FLMBOOL			bIndexAfterCommit = FALSE;

	pDb->uiFlags |= FDB_COMMITTING_TRANS;

	// See if we even have a transaction going.

	if ((uiTransType = pDb->uiTransType) == FLM_NO_TRANS)
	{
		goto Exit;	// Will return FERR_OK.
	}

	// See if we have a transaction going which should be aborted.

	if (flmCheckBadTrans( pDb))
	{
		rc = RC_SET( FERR_ABORT_TRANS);
		goto Exit;
	}

	// If we are in a read transaction we can skip most of the stuff
	// below because no updates would have occurred.  This will help
	// improve performance.

	if (uiTransType == FLM_READ_TRANS)
	{

		if( pDb->KrefCntrl.bKrefSetup)
		{
			// KrefCntrlFree could be called w/o checking bKrefSetup because
			// it checks the flag, but it is more optimal to check the
			// flag before making the call because most of the time it will
			// be false.

			KrefCntrlFree( pDb);
		}
		goto Exit1;
	}

	// At this point, we know we have an update transaction.

	pFile->pRfl->clearLogHdrs();

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
			0, 0, FERR_OK, "TCmit");
#endif
	uiTransId = pDb->LogHdr.uiCurrTransID;

	// If the transaction had no update operations, restore it
	// to its pre-transaction state - make it appear that no
	// transaction ever happened.

	if (!pDb->bHadUpdOper)
	{
		bOkToLogAbort = FALSE;
		rc = pFile->pRfl->logEndTransaction( RFL_TRNS_COMMIT_PACKET, TRUE);

		// Even though we didn't have any update operations, there may have
		// been operations during the transaction (i.e., query operations)
		// that initialized the KREF in order to generate keys.

		KrefCntrlFree( pDb);

		// Restore everything as if the transaction never happened.

		f_mutexLock( gv_FlmSysData.hShareMutex);
		pFile->uiUpdateTransID = 0;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		if (pbEmpty)
		{
			*pbEmpty = TRUE;
		}
		goto Exit1;
	}

	// Log commit record to roll-forward log

	bOkToLogAbort = FALSE;
	if (RC_BAD( rc = pFile->pRfl->logEndTransaction(
								RFL_TRNS_COMMIT_PACKET, FALSE, &bTransEndLogged)))
	{
		goto Exit1;
	}
	bForceCloseOnError = TRUE;

	// Commit any keys in the KREF buffers.

	if (RC_BAD( rc = KYKeysCommit( pDb, TRUE)))
	{
		flmLogError( rc, "calling KYKeysCommit from flmCommitDbTrans");
		goto Exit1;
	}

	if (RC_BAD( rc = FSCommitIxCounts( pDb)))
	{
		flmLogError( rc, "calling FSCommitIxCounts from flmCommitDbTrans");
		goto Exit1;
	}

	// Reinitialize the log header.  If the local dictionary was updated
	// during the transaction, increment the local dictionary ID so that
	// other concurrent users will know that it has been modified and
	// that they need to re-read it into memory.

	// If we are in recovery mode, see if we need to force
	// a checkpoint with what we have so far.  We force a
	// checkpoint on one of two conditions:
	
	// 1. If it appears that we have a buildup of dirty cache
	//		blocks.  We force a checkpoint on this condition
	//		because it will be more efficient than replacing
	//		cache blocks one at a time.
	//		We check for this condition by looking to see if
	//		our LRU block is not used and it is dirty.  That is
	//		a pretty good indicator that we have a buildup
	//		of dirty cache blocks.
	// 2.	We are at the end of the roll-forward log.  We
	//		want to force a checkpoint here to complete the
	//		recovery phase.

	if ( pDb->uiFlags & FDB_REPLAYING_RFL)
	{
		// If we are in the middle of upgrading, and are forcing
		// a checkpoint, use the file number and offset that were
		// set in the FDB.

		if ((pDb->uiFlags & FDB_UPGRADING) && bForceCheckpoint)
		{
			uiCPFileNum = pDb->uiUpgradeCPFileNum;
			uiCPOffset = pDb->uiUpgradeCPOffset;
		}
		else
		{
			SCACHE *		pTmpSCache;
			F_Rfl *		pRfl = pFile->pRfl;

			f_mutexLock( gv_FlmSysData.hShareMutex);
			pTmpSCache = gv_FlmSysData.SCacheMgr.pLRUCache;

			// Test for buildup of dirty cache blocks.

			if (((pTmpSCache) &&
				  (!pTmpSCache->uiUseCount) &&
				  (pTmpSCache->ui16Flags &
					(CA_DIRTY | CA_LOG_FOR_CP | CA_WRITE_TO_LOG)))

			||	// Test for end of roll-forward log.

				pRfl->atEndOfLog()
			||
				bForceCheckpoint)
			{
				bForceCheckpoint = TRUE;
				uiCPFileNum = pRfl->getCurrFileNum();
				uiCPOffset = pRfl->getCurrReadOffset();
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
		}
	}

	// Move information collected in the pDb->LogHdr into the
	// uncommitted log header.  Other things that need to be
	// set have already been set in the uncommitted log header
	// at various places in the code.

	// Mutex does not have to be locked while we do this because
	// the update transaction is the only one that ever accesses
	// the uncommitted log header buffer.

	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];

	// Set the new logical EOF if passed in.

	if (uiNewLogicalEOF)
	{
		pDb->LogHdr.uiLogicalEOF = uiNewLogicalEOF;
	}
	UD2FBA( (FLMUINT32)pDb->LogHdr.uiLogicalEOF,
		&pucUncommittedLogHdr [LOG_LOGICAL_EOF]);

	// Increment the commit counter.

	flmIncrUint( &pucUncommittedLogHdr [LOG_COMMIT_COUNT], 1);

	// Set the last committed transaction ID

	if( (bTransEndLogged || (pDb->uiFlags & FDB_REPLAYING_COMMIT)) &&
		pDb->pFile->FileHdr.uiVersionNum >= FLM_VER_4_31)
	{
		UD2FBA( (FLMUINT32)uiTransId, 
			&pucUncommittedLogHdr [LOG_LAST_RFL_COMMIT_ID]);
	}

	// Write the header

	pFile->pRfl->commitLogHdrs( pucUncommittedLogHdr,
							pFile->ucCheckpointLogHdr);

	// Commit any record cache.

	flmRcaCommitTrans( pDb);

	// Push the IXD_FIXUP values back into the IXD

	if (pDb->pIxdFixups)
	{
		IXD_FIXUP *	pIxdFixup;
		IXD_FIXUP *	pDeleteIxdFixup;
		IXD *			pIxd;

		pIxdFixup = pDb->pIxdFixups;
		while (pIxdFixup)
		{
			if( RC_BAD( fdictGetIndex(
				pDb->pDict, pDb->pFile->bInLimitedMode,
				pIxdFixup->uiIndexNum, NULL, &pIxd, TRUE)))
			{
				flmAssert( 0);
				pIxd = NULL;
			}

			if( pIxd)
			{
				pIxd->uiLastContainerIndexed = pIxdFixup->uiLastContainerIndexed;
				pIxd->uiLastDrnIndexed = pIxdFixup->uiLastDrnIndexed;
			}
			pDeleteIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
		}
		pDb->pIxdFixups = NULL;
	}

	// Set the update transaction ID back to zero only 
	// AFTER we know the transaction has safely committed.

	f_mutexLock( gv_FlmSysData.hShareMutex);
	f_memcpy( pFile->ucLastCommittedLogHdr, pucUncommittedLogHdr,
					LOG_HEADER_SIZE);
	pFile->uiUpdateTransID = 0;
	ScaReleaseLogBlocks( pFile);
	if (pDb->uiFlags & FDB_UPDATED_DICTIONARY)
	{
		// Link the new local dictionary to its file.
		// Since the new local dictionary will be linked at the head
		// of the list of FDICT structures, see if the FDICT currently
		// at the head of the list is unused and can be unlinked.

		if ((pFile->pDictList) && (!pFile->pDictList->uiUseCount))
		{
			flmUnlinkDict( pFile->pDictList);
		}
		flmLinkDictToFile( pFile, pDb->pDict);
	}

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit1:

	// If the local dictionary was updated during this transaction,
	// link the new local dictionary structures to their file - or free
	// them if there was an error.

	if (pDb->uiFlags & FDB_UPDATED_DICTIONARY)
	{
		if( RC_BAD( rc) && pDb->pDict)
		{
			// Unlink the FDB from the FDICT. - Shouldn't have
			// to lock semaphore, because the DICT is NOT linked
			// to the FFILE.

			flmAssert( pDb->pDict->pFile == NULL);
			flmUnlinkFdbFromDict( pDb);
		}
	}

	if (RC_BAD( rc))
	{

		// Since we failed to commit, do an abort.  We are purposely not
		// checking the return code from flmAbortDbTrans because we already
		// have an error return code.  If we attempted to log the transaction
		// to the RFL and failed, we don't want to try to log an abort packet.
		// The RFL code has already reset the log back to the starting point 
		// of the transaction, thereby discarding all operations.

		pDb->uiFlags &= ~FDB_COMMITTING_TRANS;
		(void)flmAbortDbTrans( pDb, bOkToLogAbort);
		uiTransType = FLM_NO_TRANS;

		// Do we need to force all handles to close?

		if( bForceCloseOnError)
		{

			// Since the commit packet has already been logged to the RFL,
			// we must have failed when trying to write the log header.  The
			// database is in a bad state and must be closed.

			// Set the "must close" flag on all FDBs linked to the FFILE
			// and set the FFILE's "must close" flag.  This will cause any
			// subsequent operations on the database to fail until all
			// handles have been closed.

			flmSetMustCloseFlags( pFile, rc, FALSE);
		}
	}
	else
	{
		bInvisibleTrans = (pDb->uiFlags & FDB_INVISIBLE_TRANS) ? TRUE : FALSE;
		if (uiTransType == FLM_UPDATE_TRANS)
		{
			if (gv_FlmSysData.EventHdrs [F_EVENT_UPDATES].pEventCBList)
			{
				flmTransEventCallback( F_EVENT_COMMIT_TRANS, (HFDB)pDb, rc,
							uiTransId);
			}

			// Do the BLOB and indexing work before we unlock the db.

			FBListAfterCommit( pDb);
			
			if (pDb->pIxStopList || pDb->pIxStartList)
			{
				
				// Must not call flmIndexingAfterCommit until after
				// completeTransWrites.  Otherwise, there is a potential
				// deadlock condition where flmIndexingAfterCommit is
				// waiting on an indexing thread to quit, but that
				// thread is waiting to be signaled by this thread that
				// writes are completed.  However, flmIndexingAfterCommit
				// also must only be called while the database is still
				// locked.  If we were to leave the database locked for
				// every call to completeTransWrites, however, we would
				// lose the group commit capability.  Hence, we opt to
				// only lose it when there are actual indexing operations
				// to start or stop - which should be very few transactions.
				// That is what the bIndexAfterCommit flag is for.
				
				bIndexAfterCommit = TRUE;
			}
		}
	}

	// Unlock the database, if the update transaction is still going.
	// NOTE: We check uiTransType because it may have been reset
	// to FLM_NO_TRANS up above if flmAbortDbTrans was called.

	if (uiTransType == FLM_UPDATE_TRANS)
	{
		if (RC_BAD( rc))
		{

			// SHOULD NEVER HAPPEN - because it would have been taken
			// care of above - flmAbortDbTrans would have been called and
			// uiTransType would no longer be FLM_UPDATE_TRANS.

			flmAssert( 0);
			(void)pFile->pRfl->completeTransWrites( pDb, FALSE, TRUE);
		}
		else if (!bForceCheckpoint)
		{
			if (bIndexAfterCommit)
			{
				rc = pFile->pRfl->completeTransWrites( pDb, TRUE, FALSE);
				flmIndexingAfterCommit( pDb);
				flmUnlinkDbFromTrans( pDb, TRUE);
			}
			else
			{
				rc = pFile->pRfl->completeTransWrites( pDb, TRUE, TRUE);
			}
		}
		else
		{

			// Do checkpoint, if forcing.  Before doing the checkpoint
			// we have to make sure the roll-forward log writes
			// complete.  We don't want to unlock the DB while the
			// writes are happening in this case - thus, the FALSE
			// parameter to completeTransWrites.

			if (RC_OK( rc = pFile->pRfl->completeTransWrites( pDb, TRUE, FALSE)))
			{
				bForceCloseOnError = FALSE;
				rc = ScaDoCheckpoint( pDbStats, pDb->pSFileHdl, pFile,
						(pDb->uiFlags & FDB_DO_TRUNCATE) ? TRUE : FALSE,
						TRUE, CP_TIME_INTERVAL_REASON,
						uiCPFileNum, uiCPOffset);
			}
			if (bIndexAfterCommit)
			{
				flmIndexingAfterCommit( pDb);
			}
			flmUnlinkDbFromTrans( pDb, TRUE);
		}

		if (RC_BAD( rc) && bForceCloseOnError)
		{

			// Since the commit packet has already been logged to the RFL,
			// we must have failed when trying to write the log header.  The
			// database is in a bad state and must be closed.

			// Set the "must close" flag on all FDBs linked to the FFILE
			// and set the FFILE's "must close" flag.  This will cause any
			// subsequent operations on the database to fail until all
			// handles have been closed.

			flmSetMustCloseFlags( pFile, rc, FALSE);
		}
	}
	else
	{

		// Unlink the database from the transaction
		// structure as well as from the FDICT structure.

		flmUnlinkDbFromTrans( pDb, FALSE);
	}

	if (pDbStats && uiTransType != FLM_NO_TRANS)
	{
		FLMUINT64	ui64ElapMilli = 0;

		flmAddElapTime( &pDb->TransStartTime, &ui64ElapMilli);
		pDbStats->bHaveStats = TRUE;
		if (uiTransType == FLM_READ_TRANS)
		{
			pDbStats->ReadTransStats.CommittedTrans.ui64Count++;
			pDbStats->ReadTransStats.CommittedTrans.ui64ElapMilli +=
					ui64ElapMilli;
			if (bInvisibleTrans)
			{
				pDbStats->ReadTransStats.InvisibleTrans.ui64Count++;
				pDbStats->ReadTransStats.InvisibleTrans.ui64ElapMilli +=
					ui64ElapMilli;
			}
		}
		else
		{
			pDbStats->UpdateTransStats.CommittedTrans.ui64Count++;
			pDbStats->UpdateTransStats.CommittedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
	}

	// Update stats

	if (pDb->pStats)
	{
		(void)flmStatUpdate( &gv_FlmSysData.Stats, &pDb->Stats);
	}

Exit:

	pDb->uiFlags &= ~FDB_COMMITTING_TRANS;
	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Commits an active transaction.
*END************************************************************************/
RCODE 
		// FERR_ILLEGAL_TRANS_OP - Active child transactions must be committed
		// before the parent transaction can be committed.
		//
		// FERR_ABORT_TRANS - The transaction cannot be committed and must
		// be aborted.
		//
		// FERR_NO_TRANS_ACTIVE - No transaction is active
	FlmDbTransCommit(
		HFDB			hDb,
			// [IN] Database handle.
		FLMBOOL *	pbEmpty				// May be NULL
	)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

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
			rc = Wire.doTransOp(	FCS_OP_TRANSACTION_COMMIT, 0, 0, 0);
		}
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK | FDB_CLOSING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If there is an invisible transaction going, it should not be
	// commitable by an application.

	if ((pDb->uiTransType == FLM_NO_TRANS) ||
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if( RC_BAD( pDb->AbortRc))
	{
		rc = RC_SET( FERR_ABORT_TRANS);
		goto Exit;
	}

	if (pbEmpty)
	{
		*pbEmpty = FALSE;
	}
	rc = flmCommitDbTrans( pDb, 0, FALSE, pbEmpty);

Exit:

	if( RC_OK( rc))
	{
		rc = flmCheckDatabaseState( pDb);
	}

	flmExit( FLM_DB_TRANS_COMMIT, pDb, rc);
	return( rc);
}
