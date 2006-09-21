//------------------------------------------------------------------------------
// Desc: Native C routines to support C# Db class
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

#include "xflaim.h"

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Db_Release(
	FLMUINT64	ui64This)
{
	IF_Db *	pDb = ((IF_Db *)(FLMUINT)ui64This);
	
	if (pDb)
	{
		pDb->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_backupBegin(
	FLMUINT64	ui64This,
	FLMBOOL		bFullBackup,
	FLMBOOL		bLockDb,
	FLMUINT32	ui32MaxLockWait,
	FLMUINT64 *	pui64BackupRef)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = ((IF_Db *)(FLMUINT)ui64This);
	IF_Backup *	pBackup = NULL;

	if (RC_BAD( rc = pDb->backupBegin(
								(eDbBackupType)(bFullBackup
														? XFLM_FULL_BACKUP
														: XFLM_INCREMENTAL_BACKUP),
								(eDbTransType)(bLockDb
													? XFLM_READ_TRANS
													: XFLM_UPDATE_TRANS),
								(FLMUINT)ui32MaxLockWait, &pBackup)))
	{
		goto Exit;
	}

Exit:

	*pui64BackupRef = (FLMUINT64)((FLMUINT)pBackup);
	return( rc);
}
