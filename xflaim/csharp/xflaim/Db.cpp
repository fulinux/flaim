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
	FLMUINT64		ui64This)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);
	
	if (pDb)
	{
		pDb->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transBegin(
	FLMUINT64		ui64This,
	FLMUINT32		ui32TransType,
	FLMUINT32		uiMaxLockWait,
	FLMUINT32		uiFlags)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);

	return( pDb->transBegin( (eDbTransType)ui32TransType, 
		uiMaxLockWait, uiFlags));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transBeginClone(
	FLMUINT64		ui64This,
	FLMUINT64		ui64DbToClone)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);
	IF_Db *			pDbToClone = ((IF_Db *)(FLMUINT)ui64DbToClone);

	return( pDb->transBegin( pDbToClone));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transCommit(
	FLMUINT64		ui64This)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);

	return( pDb->transCommit());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transAbort(
	FLMUINT64		ui64This)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);

	return( pDb->transAbort());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_Db_getTransType(
	FLMUINT64		ui64This)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);

	return( (FLMUINT32)pDb->getTransType());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_doCheckpoint(
	FLMUINT64		ui64This,
	FLMUINT32		ui32Timeout)
{
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);

	return( pDb->doCheckpoint( ui32Timeout));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_backupBegin(
	FLMUINT64		ui64This,
	FLMBOOL			bFullBackup,
	FLMBOOL			bLockDb,
	FLMUINT32		ui32MaxLockWait,
	FLMUINT64 *		pui64BackupRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = ((IF_Db *)(FLMUINT)ui64This);
	IF_Backup *		pBackup = NULL;

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
