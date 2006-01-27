//-------------------------------------------------------------------------
// Desc:	Retrieve current transaction number and commit count.
// Tabs:	3
//
//		Copyright (c) 1991-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fltrnum.cpp 12266 2006-01-19 14:45:33 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~***********************************************************************
Desc : Retrieves the current transaction number of a database
Notes: This routine should only be called only from within an update
		 transaction since read transactions are not assigned a transaction
		 number.
*END************************************************************************/
RCODE FlmDbGetTransId(
	HFDB				hDb,
	FLMUINT *		puiTrNumRV
	)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		
		CS_CONTEXT_p	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		// Send a request to get the transaction ID.

		if (RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_GET_TRANS_ID)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		/* Read the response. */
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}
		*puiTrNumRV = Wire.getTransId();

		rc = Wire.getRCode();
		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
										FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	*puiTrNumRV = pDb->LogHdr.uiCurrTransID;

Exit:

	flmExit( FLM_DB_GET_TRANS_ID, pDb, rc);
	return( rc);
}


/*API~***********************************************************************
Desc : Retrieves the last commit sequence number of a database.
Notes: Whenever a transaction is committed, FLAIM increments the commit
		 sequence number to indicate that the database has been modified.
		 An application may use this routine to determine if the database
		 has been modified.
*END************************************************************************/
RCODE FlmDbGetCommitCnt(
	HFDB				hDb,
	FLMUINT *		puiCommitCount
	)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT_p	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		/* Send a request to get the commit count. */

		if (RC_BAD( rc = Wire.sendOp( 
			FCS_OPCLASS_DATABASE, FCS_OP_GET_COMMIT_CNT)))
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
		*puiCommitCount = (FLMUINT)Wire.getCount();

		rc = Wire.getRCode();
		goto ExitCS;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto ExitCS;
	}

	if (RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
										FDB_TRANS_GOING_OK, 0, &bIgnore)))
	{
		goto Exit;
	}

	// See if we have a transaction going which should be aborted.

	if (pDb->uiTransType != FLM_NO_TRANS)
	{
		if (flmCheckBadTrans( pDb))
		{
			rc = RC_SET( FERR_ABORT_TRANS);
			goto Exit;
		}
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	*puiCommitCount = (FLMUINT)FB2UD(
			&pDb->pFile->ucLastCommittedLogHdr [LOG_COMMIT_COUNT]);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);

Exit:

ExitCS:

	flmExit( FLM_DB_GET_COMMIT_CNT, pDb, rc);
	return( rc);
}
