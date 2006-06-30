//------------------------------------------------------------------------------
// Desc:	This module contains the routines for inserting a row into a table.
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

#include "flaimsys.h"

//------------------------------------------------------------------------------
// Desc:	Delete a row from the database.
//------------------------------------------------------------------------------
RCODE F_Db::deleteRow(
	FLMUINT				uiTableNum,
	FLMUINT64			ui64RowId,
	FLMBOOL				bLogDelete)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBYTE				ucKeyBuf[ FLM_MAX_NUM_BUF_SIZE];
	FLMUINT				uiKeyLen;
	F_Row *				pRow = NULL;
	F_Btree *			pBTree = NULL;
	FLMBOOL				bStartedTrans = FALSE;
	
	// Make sure we are in an update transaction.
	
	if (RC_BAD( rc = checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}
	
	// Cannot delete from internal system tables, unless it is an
	// internal delete.
	
	if (bLogDelete)
	{
		F_TABLE *	pTable = m_pDict->getTable( uiTableNum);
		if (pTable->bSystemTable)
		{
			rc = RC_SET( NE_SFLM_CANNOT_DELETE_IN_SYSTEM_TABLE);
			goto Exit;
		}
	}
	
	// First we need to retrieve the row and update any index keys.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->retrieveRow( this,
								uiTableNum, ui64RowId, &pRow)))
	{
		goto Exit;
	}
	
	// Delete any index keys for the row.
	
	if (RC_BAD( rc = updateIndexKeys( uiTableNum, pRow, NULL)))
	{
		goto Exit;
	}
	
	// Get a B-Tree object to delete the row from the b-tree.

	if( RC_BAD( rc = getCachedBTree( uiTableNum, &pBTree)))
	{
		goto Exit;
	}
	uiKeyLen = sizeof( ucKeyBuf);
	if( RC_BAD( rc = flmNumber64ToStorage( ui64RowId, &uiKeyLen, ucKeyBuf,
									FALSE, TRUE)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pBTree->btRemoveEntry( ucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}

	// Remove the row from row cache if it is there.
	
	gv_SFlmSysData.pRowCacheMgr->removeRow( this, pRow, TRUE, FALSE);
	pRow = NULL;
	
	if (bLogDelete)
	{
		if (RC_BAD( rc = m_pDatabase->m_pRfl->logDeleteRow( this, uiTableNum, ui64RowId)))
		{
			goto Exit;
		}
	}

	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if (bStartedTrans)
	{
		transAbort();
	}

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	if (pBTree)
	{
		pBTree->Release();
	}

	return( rc);
}

