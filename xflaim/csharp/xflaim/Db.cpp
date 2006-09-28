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
	IF_Db *	pDb)
{
	if (pDb)
	{
		pDb->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transBegin(
	IF_Db *		pDb,
	FLMUINT32	ui32TransType,
	FLMUINT32	ui32MaxLockWait,
	FLMUINT32	ui32Flags)
{
	return( pDb->transBegin( (eDbTransType)ui32TransType, 
		(FLMUINT)ui32MaxLockWait, (FLMUINT)ui32Flags));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transBeginClone(
	IF_Db *	pDb,
	IF_Db *	pDbToClone)
{
	return( pDb->transBegin( pDbToClone));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transCommit(
	IF_Db *	pDb)
{
	return( pDb->transCommit());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_transAbort(
	IF_Db *	pDb)
{
	return( pDb->transAbort());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_Db_getTransType(
	IF_Db *	pDb)
{
	return( (FLMUINT32)pDb->getTransType());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_doCheckpoint(
	IF_Db *		pDb,
	FLMUINT32	ui32Timeout)
{
	return( pDb->doCheckpoint( ui32Timeout));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Db_backupBegin(
	IF_Db *			pDb,
	FLMBOOL			bFullBackup,
	FLMBOOL			bLockDb,
	FLMUINT32		ui32MaxLockWait,
	IF_Backup **	ppBackup)
{
	return( pDb->backupBegin(
								(eDbBackupType)(bFullBackup
														? XFLM_FULL_BACKUP
														: XFLM_INCREMENTAL_BACKUP),
								(eDbTransType)(bLockDb
													? XFLM_READ_TRANS
													: XFLM_UPDATE_TRANS),
								(FLMUINT)ui32MaxLockWait, ppBackup));
}
