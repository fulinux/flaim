//-------------------------------------------------------------------------
// Desc:	Reserve next DRN.
// Tabs:	3
//
//		Copyright (c) 1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flresdrn.cpp 12266 2006-01-19 14:45:33 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~***********************************************************************
Area : UPDATE
Desc : Returns the next DRN that record ADD would return.  The database/store
		 must be in an existing update transaction.
Notes: 
*END************************************************************************/
RCODE	
	// FERR_ILLEGAL_TRANS	- a READ transaction is going - cannot promote to
	//							     an update transaction
	// FERR_ABORT_TRANS		- Previous operation failed forcing the caller
	//								  to abort this transaction
	// FERR_NO_TRANS_ACTIVE - an update transaction must be active.
	// IO_FILE_LOCK_ERR 		- Could not get the lock on the database or store.
	FlmReserveNextDrn(
		HFDB			hDb,
			// [IN] Database handle.
		FLMUINT		uiContainer,
			// [IN] Container number.
		FLMUINT *	puiDrnRV
			// [OUT] Pointer to a FLMUINT variable which will return the next DRN 
			// value for the container as if FlmRecordAdd is called with *drnRV
			// equal to zero.
			//
			// NOTE:  When FLAIM is allowed to assign the DRN, it always selects
			// the highest available DRN.  If, for example, a user successfully
			// adds a record with DRN 5000 to a previously empty container and
			// then adds a second record (allowing FLAIM to automatically assign
			// the DRN) the second record will be assigned DRN 5001.
	)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	LFILE *		pLFile;
	FLMBOOL		bIgnore;
	FLMUINT		uiDrn = 0;

	if (IsInCSMode( hDb))
	{
		fdbInitCS( pDb);

		CS_CONTEXT_p	pCSContext = pDb->pCSContext;
		FCL_WIRE			Wire( pCSContext, pDb);

		// Send the request

		if( RC_BAD( rc = Wire.sendOp( 
			FCS_OPCLASS_RECORD, FCS_OP_RESERVE_NEXT_DRN)))
		{
			goto ExitCS;
		}

		if( uiContainer)
		{
			if (RC_BAD( rc = Wire.sendNumber(
				WIRE_VALUE_CONTAINER_ID, uiContainer)))
			{
				goto Transmission_Error;
			}
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		// Read the response

		if( RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto ExitCS;
		}

		*puiDrnRV = Wire.getDrn();
		goto ExitCS;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto ExitCS;
	}

	bIgnore = FALSE;					// Set to shut up compiler.

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
										FDB_TRANS_GOING_OK,	// byFlags
										0, 						// wAutoTrans
										&bIgnore)))				// bStartedAutoTrans
	{
		goto Exit;
	}

	if( pDb->uiFlags & FDB_COMMITTING_TRANS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}

	if( RC_BAD( fdictGetContainer( pDb->pDict, uiContainer, &pLFile)))
	{
#ifdef FLM_DBG_LOG
		uiDrn = 0;
#endif
		goto Exit;
	}
	uiDrn = (FLMUINT) 0;					// Must initialize before call.
	if( RC_BAD( rc = FSGetNextDrn( pDb, pLFile, TRUE, &uiDrn)))
	{
#ifdef FLM_DBG_LOG
		uiDrn = 0;
#endif
		goto Exit;
	}

	*puiDrnRV = uiDrn;						// Set return value.

Exit:

	if (RC_OK( rc))
	{
		rc = pDb->pFile->pRfl->logUpdatePacket( 
			RFL_RESERVE_DRN_PACKET, uiContainer, *puiDrnRV, 0);
	}

	if( gv_FlmSysData.EventHdrs[ F_EVENT_UPDATES].pEventCBList)
	{
		flmUpdEventCallback( pDb, F_EVENT_RESERVE_DRN, hDb, rc, *puiDrnRV,
								uiContainer, NULL, NULL);
	}

#ifdef FLM_DBG_LOG
	flmDbgLogUpdate( pDb->pFile->uiFFileId, pDb->LogHdr.uiCurrTransID,
			uiContainer, uiDrn, rc, "RDrn");
#endif

ExitCS:

	flmExit( FLM_RESERVE_NEXT_DRN, pDb, rc);

	return( rc);
}


/*API~***********************************************************************
Area : UPDATE
Desc : Searches for an available DRN in the dictionary container.  Differs
		 from FlmReserveNextDrn in that it will attempt to reuse dictionary
		 DRNS.  The database/store must be in an existing update transaction.
Notes: 
*END************************************************************************/
RCODE FlmFindUnusedDictDrn(
	HFDB					hDb,
	FLMUINT				uiStartDrn,
	FLMUINT				uiEndDrn,
	FLMUINT *			puiDrnRV)
{
	RCODE			rc;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bIgnore = FALSE;
	FDICT_p		pDict;
	FLMUINT		uiCurrDrn;
	FLMUINT		uiStopSearch;

	if( RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS, FDB_TRANS_GOING_OK,
		0, &bIgnore)))
	{
		*puiDrnRV = (FLMUINT)-1;
		goto Exit;
	}

	// Search through the ITT table looking for the first occurance
	// of ITT_EMPTY_SLOT
	
	pDict = pDb->pDict;
	uiCurrDrn = f_max( uiStartDrn, 1);
	uiStopSearch = f_min( uiEndDrn, pDict->uiIttCnt - 1);
	
	while (uiCurrDrn <= uiStopSearch)	
	{
		if (pDict->pIttTbl[ uiCurrDrn].uiType == ITT_EMPTY_SLOT)
		{
			break;
		}
		else
		{
			uiCurrDrn++;
		}	
	}

	if (uiCurrDrn > uiEndDrn)
	{
		rc = RC_SET( FERR_NO_MORE_DRNS);
		goto Exit;
	}

	*puiDrnRV = uiCurrDrn;

Exit:

	fdbExit( pDb);
	return( rc);
}
