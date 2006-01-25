//-------------------------------------------------------------------------
// Desc:	Abort transaction
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
// $Id: fltrabrt.cpp 12266 2006-01-19 14:45:33 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	This routine aborts an active transaction for a particular
		database.  If the database is open via a server, a message is
		sent to the server to abort the transaction.  Otherwise, the
		transaction is rolled back locally.
****************************************************************************/
RCODE flmAbortDbTrans(
	FDB *				pDb,
	FLMBOOL			bOkToLogAbort)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiTransType;
	FLMBYTE *		pucLastCommittedLogHdr;
	FLMBYTE *		pucUncommittedLogHdr;
	FLMBOOL			bDumpedCache = FALSE;
	DB_STATS *		pDbStats = pDb->pDbStats;
	FLMBOOL			bKeepAbortedTrans;
	FLMUINT			uiTransId;
	FLMBOOL			bInvisibleTrans;

	// Get transaction type

	if ((uiTransType = pDb->uiTransType) == FLM_NO_TRANS)
	{
		goto Exit;	// Will return SUCCESS.
	}

	// No recovery required if it is a read transaction.

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

		goto Unlink_From_Trans;
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
			0, 0, FERR_OK, "TAbrt");
#endif

	pFile->pRfl->clearLogHdrs();

	// If the transaction had no update operations, restore it
	// to its pre-transaction state - make it appear that no
	// transaction ever happened.

	pucLastCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	pucUncommittedLogHdr = &pFile->ucUncommittedLogHdr [0];
	uiTransId = pDb->LogHdr.uiCurrTransID;

	// Free up all keys associated with this database.  This is done even
	// if we didn't have any update operations because the KREF may
	// have been initialized by key generation operations performed
	// by cursors, etc.

	KrefCntrlFree( pDb);

	// Free any index counts we may have allocated.

	FSFreeIxCounts( pDb);
	
	if (pDb->bHadUpdOper)
	{
		// Dump any BLOB structures that should be aborted.

		FBListAfterAbort( pDb);

		// Dump any start and stop indexing stubs that should be aborted.

		flmIndexingAfterAbort( pDb);

		// Log the abort record to the rfl file, or throw away the logged
		// records altogether, depending on the LOG_KEEP_ABORTED_TRANS_IN_RFL
		// flag.  If the RFL volume is bad, we will not attempt to keep this
		// transaction in the RFL.

		if (!pFile->pRfl->seeIfRflVolumeOk())
		{
			bKeepAbortedTrans = FALSE;
		}
		else
		{
			bKeepAbortedTrans =
				(pucUncommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL])
				? TRUE
				: FALSE;
		}
	}
	else
	{
		bKeepAbortedTrans = FALSE;
	}

	// Log an abort transaction record to the roll-forward log or
	// throw away the entire transaction, depending on the
	// bKeepAbortedTrans flag.

	// If the transaction is being "dumped" because of a failed commit,
	// don't log anything to the RFL.

	if( bOkToLogAbort)
	{
		flmAssert( pDb->LogHdr.uiCurrTransID == pFile->pRfl->getCurrTransID());
		if (RC_BAD( rc = pFile->pRfl->logEndTransaction(
									RFL_TRNS_ABORT_PACKET, !bKeepAbortedTrans)))
		{
			goto Exit1;
		}
	}
#ifdef FLM_DEBUG
	else
	{
		// If bOkToLogAbort is FALSE, this always means that either a
		// commit failed while trying to log an end transaction packet or a
		// commit packet was logged and the transaction commit subsequently
		// failed for some other reason.  In either case, the RFL should be
		// in a good state, with its current transaction ID reset to 0.  If
		// not, either bOkToLogAbort is being used incorrectly by the caller
		// or there is a bug in the RFL logic.

		flmAssert( pFile->pRfl->getCurrTransID() == 0);
	}
#endif

	// If there were no operations in the transaction, restore
	// everything as if the transaction never happened.

	if (!pDb->bHadUpdOper)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		pFile->uiUpdateTransID = 0;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);

		// Pretend we dumped cache - shouldn't be any to worry about at
		// this point.

		bDumpedCache = TRUE;
		goto Exit1;
	}

	// Dump ALL modified cache blocks associated with the DB.
	// NOTE: This needs to be done BEFORE the call to flmGetLogHdrInfo
	// below, because that call will change pDb->LogHdr.uiCurrTransID,
	// and that value is used by flmRcaAbortTrans.

	ScaFreeModifiedBlocks( pDb);
	flmRcaAbortTrans( pDb);
	bDumpedCache = TRUE;

	// Reset the LogHdr from the last committed log header in pFile.

	flmGetLogHdrInfo( pucLastCommittedLogHdr, &pDb->LogHdr);
	if (RC_BAD( rc = flmPhysRollback( pDb,
				 (FLMUINT)FB2UD( &pucUncommittedLogHdr [LOG_ROLLBACK_EOF]),
				 pFile->uiFirstLogBlkAddress, FALSE, 0)))
	{
		goto Exit1;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);

	// Put the new transaction ID into the log header even though
	// we are not committing.  We want to keep the transaction IDs
	// incrementing even though we aborted.

	UD2FBA( (FLMUINT32)uiTransId,
			&pucLastCommittedLogHdr [LOG_CURR_TRANS_ID]);

	// Preserve where we are at in the roll-forward log.  Even though
	// the transaction aborted, we may have kept it in the RFL instead of
	// throw it away.

	f_memcpy( &pucLastCommittedLogHdr [LOG_RFL_FILE_NUM],
				 &pucUncommittedLogHdr [LOG_RFL_FILE_NUM], 4);
	f_memcpy( &pucLastCommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET],
				 &pucUncommittedLogHdr [LOG_RFL_LAST_TRANS_OFFSET], 4);
	f_memcpy( &pucLastCommittedLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM],
				 &pucUncommittedLogHdr [LOG_LAST_TRANS_RFL_SERIAL_NUM],
				 F_SERIAL_NUM_SIZE);
	f_memcpy( &pucLastCommittedLogHdr [LOG_RFL_NEXT_SERIAL_NUM],
				 &pucUncommittedLogHdr [LOG_RFL_NEXT_SERIAL_NUM],
				 F_SERIAL_NUM_SIZE);

	// The following items tell us where we are at in the roll-back log.
	// During a transaction we may log blocks for the checkpoint or for
	// read transactions.  So, even though we are aborting this transaction,
	// there may be other things in the roll-back log that we don't want
	// to lose.  These items should not be reset until we do a checkpoint,
	// which is when we know it is safe to throw away the entire roll-back log.

	f_memcpy( &pucLastCommittedLogHdr [LOG_ROLLBACK_EOF],
				 &pucUncommittedLogHdr [LOG_ROLLBACK_EOF], 4);
	f_memcpy( &pucLastCommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR],
				 &pucUncommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR], 4);

	f_mutexUnlock( gv_FlmSysData.hShareMutex);

	pFile->pRfl->commitLogHdrs( pucLastCommittedLogHdr,
							pFile->ucCheckpointLogHdr);

Exit1:

	// Dump cache, if not done above.

	if (!bDumpedCache)
	{
		ScaFreeModifiedBlocks( pDb);
		flmRcaAbortTrans( pDb);
		bDumpedCache = TRUE;
	}

	// Throw away IXD_FIXUPs

	if (pDb->pIxdFixups)
	{
		IXD_FIXUP *	pIxdFixup;
		IXD_FIXUP *	pDeleteIxdFixup;

		pIxdFixup = pDb->pIxdFixups;
		while (pIxdFixup)
		{
			pDeleteIxdFixup = pIxdFixup;
			pIxdFixup = pIxdFixup->pNext;
			f_free( &pDeleteIxdFixup);
		}
		pDb->pIxdFixups = NULL;
	}

	if (uiTransType != FLM_READ_TRANS &&
		 gv_FlmSysData.EventHdrs[ F_EVENT_UPDATES].pEventCBList)
	{
		flmTransEventCallback( F_EVENT_ABORT_TRANS, (HFDB)pDb, rc,
						uiTransId);
	}

Unlink_From_Trans:

	bInvisibleTrans = (pDb->uiFlags & FDB_INVISIBLE_TRANS) ? TRUE : FALSE;
	if (pDb->uiFlags & FDB_HAS_WRITE_LOCK)
	{
		RCODE	tmpRc;

		if (RC_BAD( tmpRc = pFile->pRfl->completeTransWrites( pDb, FALSE, FALSE)))
		{
			if (RC_OK( rc))
			{
				rc = tmpRc;
			}
		}
	}

	// Unlink the database from the transaction
	// structure as well as from the FLDICT structure.

	flmUnlinkDbFromTrans( pDb, FALSE);

	if (pDbStats)
	{
		FLMUINT64	ui64ElapMilli = 0;

		flmAddElapTime( &pDb->TransStartTime, &ui64ElapMilli);
		pDbStats->bHaveStats = TRUE;
		if (uiTransType == FLM_READ_TRANS)
		{
			pDbStats->ReadTransStats.AbortedTrans.ui64Count++;
			pDbStats->ReadTransStats.AbortedTrans.ui64ElapMilli +=
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
			pDbStats->UpdateTransStats.AbortedTrans.ui64Count++;
			pDbStats->UpdateTransStats.AbortedTrans.ui64ElapMilli +=
					ui64ElapMilli;
		}
	}

	if (pDb->pStats)
	{
		(void)flmStatUpdate( &gv_FlmSysData.Stats, &pDb->Stats);
	}


Exit:

	return( rc);
}

/*API~***********************************************************************
Area : TRANSACTION
Desc : Aborts an active transaction.
*END************************************************************************/
RCODE 
		// FERR_ILLEGAL_TRANS_OP - Active child transactions must be committed
		// or aborted before the parent transaction can be aborted.
		//
		// FERR_NO_TRANS_ACTIVE - No transaction is active
	FlmDbTransAbort(
		HFDB			hDb
	)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		FCL_WIRE	Wire( pDb->pCSContext, pDb);
		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp( FCS_OP_TRANSACTION_ABORT, 0, 0, 0);
		}
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK | FDB_CLOSING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// If there is an invisible transaction going, it should not be
	// abortable by an application.

	if ((pDb->uiTransType == FLM_NO_TRANS) ||
		 (pDb->uiFlags & FDB_INVISIBLE_TRANS))
	{
		rc = RC_SET( FERR_NO_TRANS_ACTIVE);
		goto Exit;
	}
	rc = flmAbortDbTrans( pDb);

Exit:

	if( RC_OK( rc))
	{
		rc = flmCheckDatabaseState( pDb);
	}

	flmExit( FLM_DB_TRANS_ABORT, pDb, rc);
	return( rc);
}
