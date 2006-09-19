//------------------------------------------------------------------------------
// Desc:
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
FLMEXTC FLMEXP void FLMAPI xflaim_Backup_Release(
	FLMUINT64	ui64This)
{
	IF_Backup *	pBackup = ((IF_Backup *)(FLMUINT)ui64This);
	
	if (pBackup)
	{
		pBackup->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT64 FLMAPI xflaim_Backup_getBackupTransId(
	FLMUINT64	ui64This)
{
	IF_Backup *	pBackup = ((IF_Backup *)(FLMUINT)ui64This);
	
	return( pBackup->getBackupTransId());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT64 FLMAPI xflaim_Backup_getLastBackupTransId(
	FLMUINT64	ui64This)
{
	IF_Backup *	pBackup = ((IF_Backup *)(FLMUINT)ui64This);
	
	return( pBackup->getLastBackupTransId());
}

typedef RCODE (FLMAPI * BACKUP_CLIENT)(
	const void *	pvData,
	FLMUINT			uiDataLen);

/****************************************************************************
Desc:
****************************************************************************/
class CS_BackupClient : public IF_BackupClient
{
public:

	CS_BackupClient(
		BACKUP_CLIENT	fnBackupClient)
	{
		m_fnBackupClient = fnBackupClient;
	}

	virtual ~CS_BackupClient()
	{
	}

	RCODE FLMAPI WriteData(
		const void *			pvBuffer,
		FLMUINT					uiBytesToWrite)
	{
		return( m_fnBackupClient( pvBuffer, uiBytesToWrite));
	}

private:

	BACKUP_CLIENT	m_fnBackupClient;
};

typedef RCODE (FLMAPI * BACKUP_STATUS)(
	FLMUINT64	ui64BytesToDo,
	FLMUINT64	ui64BytesDone);

/****************************************************************************
Desc:
****************************************************************************/
class CS_BackupStatus : public IF_BackupStatus
{
public:

	CS_BackupStatus(
		BACKUP_STATUS	fnBackupStatus)
	{
		m_fnBackupStatus = fnBackupStatus;
	}

	virtual ~CS_BackupStatus()
	{
	}

	RCODE FLMAPI backupStatus(
		FLMUINT64	ui64BytesToDo,
		FLMUINT64	ui64BytesDone)
	{
		return( m_fnBackupStatus( ui64BytesToDo, ui64BytesDone));
	}

private:

	BACKUP_STATUS	m_fnBackupStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Backup_backup(
	FLMUINT64		ui64This,
	const char *	pszBackupPath,
	const char *	pszPassword,
	FLMUINT *		puiSeqNum,
	BACKUP_CLIENT	fnBackupClient,
	BACKUP_STATUS	fnBackupStatus)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Backup *			pBackup = (IF_Backup *)((FLMUINT)ui64This);
	IF_BackupClient *	pBackupClient = NULL;
	IF_BackupStatus *	pBackupStatus = NULL;

	if (fnBackupClient)
	{
		if ((pBackupClient = f_new CS_BackupClient( fnBackupClient)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	if (fnBackupStatus)
	{
		if ((pBackupStatus = f_new CS_BackupStatus( fnBackupStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	if (RC_BAD( rc = pBackup->backup( pszBackupPath, pszPassword, pBackupClient,
								pBackupStatus, puiSeqNum)))
	{
		goto Exit;
	}
	
Exit:

	if (pBackupClient)
	{
		pBackupClient->Release();
	}
	
	if (pBackupStatus)
	{
		pBackupStatus->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Backup_endBackup(
	FLMUINT64	ui64This)
{
	IF_Backup *	pBackup = ((IF_Backup *)(FLMUINT)ui64This);
	
	return( pBackup->endBackup());
}
