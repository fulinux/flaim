//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DbSystem class
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
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_createDbSystem(
	FLMUINT64 *	pui64System)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem * 		pDbSystem = NULL;
	
	if (RC_BAD( rc = FlmAllocDbSystem( &pDbSystem)))
	{
		*pui64System = 0;
	}
	else
	{
		*pui64System = (FLMUINT64)((FLMUINT)pDbSystem);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbSystem_Release(
	FLMUINT64	ui64This)
{
	IF_DbSystem *	pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	
	if (pDbSystem)
	{
		pDbSystem->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbCreate(
	FLMUINT64				ui64This,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	const char *			pszDictFileName,
	const char *			pszDictBuf,
	XFLM_CREATE_OPTS *	pCreateOpts,
	FLMUINT64 *				pui64Db)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = NULL;
	IF_DbSystem *	pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);

	if (RC_BAD( rc = pDbSystem->dbCreate( pszDbPath, pszDataDir, pszRflDir,
								pszDictFileName, pszDictBuf,
								pCreateOpts, (IF_Db **)&pDb)))
	{
		goto Exit;
	}

 Exit:

  	*pui64Db = (FLMUINT64)((FLMUINT)pDb);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbOpen(
	FLMUINT64				ui64This,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	const char *			pszPassword,
	FLMBOOL					bAllowLimited,
	FLMUINT64 *				pui64Db)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = NULL;
	IF_DbSystem *	pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	
	if (RC_BAD( rc = pDbSystem->dbOpen( pszDbPath, pszDataDir, pszRflDir,
								pszPassword, bAllowLimited, (IF_Db **)&pDb)))
	{
		goto Exit;
	}

 Exit:

  	*pui64Db = (FLMUINT64)((FLMUINT)pDb);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbRemove(
	FLMUINT64				ui64This,
	const char *			pszDbPath,
	const char *			pszDataDir,
	const char *			pszRflDir,
	FLMBOOL					bRemoveRflFiles)
{
	IF_DbSystem *	pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
 
	return( pDbSystem->dbRemove( pszDbPath, pszDataDir, pszRflDir, bRemoveRflFiles));
}

// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cs
typedef enum
{
	RESTORE_OPEN_BACKUP_SET		= 1,
	RESTORE_OPEN_RFL_FILE		= 2,
	RESTORE_OPEN_INC_FILE		= 3,
	RESTORE_READ					= 4,
	RESTORE_CLOSE					= 5,
	RESTORE_ABORT_FILE			= 6
} eRestoreClientAction;

typedef RCODE (FLMAPI * RESTORE_CLIENT)(
	eRestoreClientAction	eAction,
	FLMUINT					uiFileNum,
	FLMUINT					uiBytesRequested,
	void *					pvBuffer,
	FLMUINT *				puiBytesRead);

/****************************************************************************
Desc:
****************************************************************************/
class CS_RestoreClient : public IF_RestoreClient
{
public:

	CS_RestoreClient(
		RESTORE_CLIENT	fnRestoreClient)
	{
		m_fnRestoreClient = fnRestoreClient;
	}

	virtual ~CS_RestoreClient()
	{
	}

	RCODE FLMAPI openBackupSet( void)
	{
		return( m_fnRestoreClient( RESTORE_OPEN_BACKUP_SET, 0, 0, NULL, NULL));
	}

	RCODE FLMAPI openRflFile(
		FLMUINT	uiFileNum)
	{
		return( m_fnRestoreClient( RESTORE_OPEN_RFL_FILE, uiFileNum, 0, NULL, NULL));
	}

	RCODE FLMAPI openIncFile(
		FLMUINT	uiFileNum)
	{
		return( m_fnRestoreClient( RESTORE_OPEN_INC_FILE, uiFileNum, 0, NULL, NULL));
	}

	RCODE FLMAPI read(
		FLMUINT		uiBytesRequested,
		void *		pvBuffer,
		FLMUINT *	puiBytesRead)
	{
		return( m_fnRestoreClient( RESTORE_READ, 0, uiBytesRequested,
					pvBuffer, puiBytesRead));
	}

	
	RCODE FLMAPI close( void)
	{
		return( m_fnRestoreClient( RESTORE_CLOSE, 0, 0, NULL, NULL));
	}

	RCODE FLMAPI abortFile( void)
	{
		return( m_fnRestoreClient( RESTORE_ABORT_FILE, 0, 0, NULL, NULL));
	}

private:

	RESTORE_CLIENT	m_fnRestoreClient;
};

// WARNING NOTE: Any changes to this enum should also be reflected in DbSystem.cs
typedef enum
{
	REPORT_PROGRESS					= 1,
	REPORT_ERROR						= 2,
	REPORT_BEGIN_TRANS				= 3,
	REPORT_COMMIT_TRANS				= 4,
	REPORT_ABORT_TRANS				= 5,
	REPORT_BLOCK_CHAIN_FREE			= 6,
	REPORT_INDEX_SUSPEND				= 7,
	REPORT_INDEX_RESUME				= 8,
	REPORT_REDUCE						= 9,
	REPORT_UPGRADE						= 10,
	REPORT_OPEN_RFL_FILE				= 11,
	REPORT_RFL_READ					= 12,
	REPORT_ENABLE_ENCRYPTION		= 13,
	REPORT_WRAP_KEY					= 14,
	REPORT_SET_NEXT_NODE_ID			= 15,
	REPORT_NODE_SET_META_VALUE		= 16,
	REPORT_NODE_SET_PREFIX_ID		= 17,
	REPORT_NODE_FLAGS_UPDATE		= 18,
	REPORT_ATTRIBUTE_SET_VALUE		= 19,
	REPORT_NODE_SET_VALUE			= 20,
	REPORT_NODE_UPDATE				= 21,
	REPORT_INSERT_BEFORE				= 22,
	REPORT_NODE_CREATE				= 23,
	REPORT_NODE_CHILDREN_DELETE	= 24,
	REPORT_ATTRIBUTE_DELETE			= 25,
	REPORT_NODE_DELETE				= 26,
	REPORT_DOCUMENT_DONE				= 27,
	REPORT_ROLL_OVER_DB_KEY			= 28
} eRestoreStatusAction;

typedef RCODE (FLMAPI * RESTORE_STATUS)(
	eRestoreStatusAction	eAction,
	eRestoreAction *		peRestoreAction,
	FLMUINT64				ui64TransId,
	FLMUINT64				ui64LongNum1,
	FLMUINT64				ui64LongNum2,
	FLMUINT64				ui64LongNum3,
	FLMUINT					uiShortNum1,
	FLMUINT					uiShortNum2,
	FLMUINT					uiShortNum3,
	FLMUINT					uiShortNum4);

/****************************************************************************
Desc:
****************************************************************************/
class CS_RestoreStatus : public IF_RestoreStatus
{
public:

	CS_RestoreStatus(
		RESTORE_STATUS	fnRestoreStatus)
	{
		m_fnRestoreStatus = fnRestoreStatus;
	}

	virtual ~CS_RestoreStatus()
	{
	}

	RCODE FLMAPI reportProgress(
		eRestoreAction *		peAction,
		FLMUINT64				ui64BytesToDo,
		FLMUINT64				ui64BytesDone)
	{
		return( m_fnRestoreStatus( REPORT_PROGRESS, peAction, 0,
					ui64BytesToDo, ui64BytesDone, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportError(
		eRestoreAction *		peAction,
		RCODE						rcErr)
	{
		return( m_fnRestoreStatus( REPORT_ERROR, peAction, 0,
					0, 0, 0,
					(FLMUINT)rcErr, 0, 0, 0));
	}

	RCODE FLMAPI reportOpenRflFile(
		eRestoreAction *		peAction,
		FLMUINT					uiFileNum)
	{
		return( m_fnRestoreStatus( REPORT_OPEN_RFL_FILE, peAction, 0,
					0, 0, 0,
					uiFileNum, 0, 0, 0));
	}

	RCODE FLMAPI reportRflRead(
		eRestoreAction *		peAction,
		FLMUINT					uiFileNum,
		FLMUINT					uiBytesRead)
	{
		return( m_fnRestoreStatus( REPORT_RFL_READ, peAction, 0,
					0, 0, 0,
					uiFileNum, uiBytesRead, 0, 0));
	}

	RCODE FLMAPI reportBeginTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( REPORT_BEGIN_TRANS, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportCommitTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( REPORT_COMMIT_TRANS, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportAbortTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( REPORT_ABORT_TRANS, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportBlockChainFree(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT64				ui64MaintDocNum,
		FLMUINT					uiStartBlkAddr,
		FLMUINT					uiEndBlkAddr,
		FLMUINT					uiCount)
	{
		return( m_fnRestoreStatus( REPORT_BLOCK_CHAIN_FREE, peAction, ui64TransId,
					ui64MaintDocNum, 0, 0,
					uiStartBlkAddr, uiEndBlkAddr, uiCount, 0));
	}

	RCODE FLMAPI reportIndexSuspend(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiIndexNum)
	{
		return( m_fnRestoreStatus( REPORT_INDEX_SUSPEND, peAction, ui64TransId,
					0, 0, 0,
					uiIndexNum, 0, 0, 0));
	}

	RCODE FLMAPI reportIndexResume(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiIndexNum)
	{
		return( m_fnRestoreStatus( REPORT_INDEX_RESUME, peAction, ui64TransId,
					0, 0, 0,
					uiIndexNum, 0, 0, 0));
	}

	RCODE FLMAPI reportReduce(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCount)
	{
		return( m_fnRestoreStatus( REPORT_REDUCE, peAction, ui64TransId,
					0, 0, 0,
					uiCount, 0, 0, 0));
	}

	RCODE FLMAPI reportUpgrade(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiOldDbVersion,
		FLMUINT					uiNewDbVersion)
	{
		return( m_fnRestoreStatus( REPORT_UPGRADE, peAction, ui64TransId,
					0, 0, 0,
					uiOldDbVersion, uiNewDbVersion, 0, 0));
	}

	RCODE FLMAPI reportEnableEncryption(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( REPORT_ENABLE_ENCRYPTION, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportWrapKey(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( REPORT_WRAP_KEY, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}
		
	RCODE FLMAPI reportRollOverDbKey(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( REPORT_ROLL_OVER_DB_KEY, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}
		
	RCODE FLMAPI reportDocumentDone(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64DocumentId)
	{
		return( m_fnRestoreStatus( REPORT_DOCUMENT_DONE, peAction, ui64TransId,
					ui64DocumentId, 0, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportNodeDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		return( m_fnRestoreStatus( REPORT_NODE_DELETE, peAction, ui64TransId,
					ui64NodeId, 0, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportAttributeDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementId,
		FLMUINT					uiAttrNameId)
	{
		return( m_fnRestoreStatus( REPORT_ATTRIBUTE_DELETE, peAction, ui64TransId,
					ui64ElementId, 0, 0,
					uiCollection, uiAttrNameId, 0, 0));
	}
		
	RCODE FLMAPI reportNodeChildrenDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ParentNodeId,
		FLMUINT					uiNameId)
	{
		return( m_fnRestoreStatus( REPORT_NODE_CHILDREN_DELETE, peAction, ui64TransId,
					ui64ParentNodeId, 0, 0,
					uiCollection, uiNameId, 0, 0));
	}
		
	RCODE FLMAPI reportNodeCreate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64RefNodeId,
		eDomNodeType			eNodeType,
		FLMUINT					uiNameId,
		eNodeInsertLoc			eLocation)
	{
		return( m_fnRestoreStatus( REPORT_NODE_CREATE, peAction, ui64TransId,
					ui64RefNodeId, 0, 0,
					uiCollection, (FLMUINT)eNodeType, uiNameId, (FLMUINT)eLocation));
	}
		
	RCODE FLMAPI reportInsertBefore(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ParentNodeId,
		FLMUINT64				ui64NewChildNodeId,
		FLMUINT64				ui64RefChildNodeId)
	{
		return( m_fnRestoreStatus( REPORT_INSERT_BEFORE, peAction, ui64TransId,
					ui64ParentNodeId, ui64NewChildNodeId, ui64RefChildNodeId,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportNodeUpdate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		return( m_fnRestoreStatus( REPORT_NODE_UPDATE, peAction, ui64TransId,
					ui64NodeId, 0, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportNodeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		return( m_fnRestoreStatus( REPORT_NODE_SET_VALUE, peAction, ui64TransId,
					ui64NodeId, 0, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportAttributeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementNodeId,
		FLMUINT					uiAttrNameId)
	{
		return( m_fnRestoreStatus( REPORT_ATTRIBUTE_SET_VALUE, peAction, ui64TransId,
					ui64ElementNodeId, 0, 0,
					uiCollection, uiAttrNameId, 0, 0));
	}
		
	RCODE FLMAPI reportNodeFlagsUpdate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiFlags,
		FLMBOOL					bAdd)
	{
		return( m_fnRestoreStatus( REPORT_NODE_FLAGS_UPDATE, peAction, ui64TransId,
					ui64NodeId, 0, 0,
					uiCollection, uiFlags, (FLMUINT)bAdd, 0));
	}
		
	RCODE FLMAPI reportNodeSetPrefixId(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT					uiAttrNameId,
		FLMUINT					uiPrefixId)
	{
		return( m_fnRestoreStatus( REPORT_NODE_SET_PREFIX_ID, peAction, ui64TransId,
					ui64NodeId, 0, 0,
					uiCollection, uiAttrNameId, uiPrefixId, 0));
	}
		
	RCODE FLMAPI reportNodeSetMetaValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		FLMUINT64				ui64MetaValue)
	{
		return( m_fnRestoreStatus( REPORT_NODE_SET_META_VALUE, peAction, ui64TransId,
					ui64NodeId, ui64MetaValue, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportSetNextNodeId(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NextNodeId)
	{
		return( m_fnRestoreStatus( REPORT_SET_NEXT_NODE_ID, peAction, ui64TransId,
					ui64NextNodeId, 0, 0,
					uiCollection, 0, 0, 0));
	}


private:

	RESTORE_STATUS	m_fnRestoreStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbRestore(
	FLMUINT64			ui64This,
	const char *		pszDbFileName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszBackupPath,
	const char *		pszPassword,
	RESTORE_CLIENT		fnRestoreClient,
	RESTORE_STATUS		fnRestoreStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbSystem *			pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_RestoreClient *	pRestoreClient = NULL;
	IF_RestoreStatus *	pRestoreStatus = NULL;

	if (fnRestoreClient)
	{
		if ((pRestoreClient = f_new CS_RestoreClient( fnRestoreClient)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	if (fnRestoreStatus)
	{
		if ((pRestoreStatus = f_new CS_RestoreStatus( fnRestoreStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbRestore( pszDbFileName, pszDataDir, pszRflDir, pszBackupPath,
								pszPassword, pRestoreClient, pRestoreStatus)))
	{
		goto Exit;
	}

Exit:

	if (pRestoreClient)
	{
		pRestoreClient->Release();
	}
	if (pRestoreStatus)
	{
		pRestoreStatus->Release();
	}

	return( rc);
}

typedef RCODE (FLMAPI * DB_CHECK_STATUS)(
	FLMBOOL							bHaveProgressInfo,
	XFLM_PROGRESS_CHECK_INFO *	pProgressInfo,
	XFLM_CORRUPT_INFO *			pCorruptInfo);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbCheckStatus : public IF_DbCheckStatus
{
public:

	CS_DbCheckStatus(
		DB_CHECK_STATUS	fnDbCheckStatus)
	{
		m_fnDbCheckStatus = fnDbCheckStatus;
	}

	virtual ~CS_DbCheckStatus()
	{
	}

	RCODE FLMAPI reportProgress(
		XFLM_PROGRESS_CHECK_INFO *	pProgCheck)
	{
		return( m_fnDbCheckStatus( TRUE, pProgCheck, NULL));
	}
	
	RCODE FLMAPI reportCheckErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo,
		FLMBOOL *				pbFix)
	{
		if (pbFix)
		{
			*pbFix = TRUE;
		}
		return( m_fnDbCheckStatus( FALSE, NULL, pCorruptInfo));
	}

private:

	DB_CHECK_STATUS	m_fnDbCheckStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbCheck(
	FLMUINT64			ui64This,
	const char *		pszDbName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszPassword,
	FLMUINT				uiFlags,
	DB_CHECK_STATUS	fnCheckStatus,
	FLMUINT64 *			pui64DbInfo)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbSystem *			pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_DbCheckStatus *	pDbCheckStatus = NULL;
	IF_DbInfo *				pDbInfo = NULL;

	if (fnCheckStatus)
	{
		if ((pDbCheckStatus = f_new CS_DbCheckStatus( fnCheckStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbCheck( pszDbName, pszDataDir, pszRflDir, pszPassword,
								uiFlags, &pDbInfo, pDbCheckStatus)))
	{
		goto Exit;
	}

Exit:

	*pui64DbInfo = (FLMUINT64)((FLMUINT)pDbInfo);

	if (pDbCheckStatus)
	{
		pDbCheckStatus->Release();
	}

	return( rc);
}

typedef RCODE (FLMAPI * DB_COPY_STATUS)(
	FLMUINT64		ui64BytesToCopy,
	FLMUINT64		ui64BytesCopied,
	FLMBOOL			bNewSrcFile,
	const char *	pszSrcFileName,
	const char *	pszDestFileName);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbCopyStatus : public IF_DbCopyStatus
{
public:

	CS_DbCopyStatus(
		DB_COPY_STATUS	fnCopyStatus)
	{
		m_fnCopyStatus = fnCopyStatus;
	}

	virtual ~CS_DbCopyStatus()
	{
	}
	
	RCODE FLMAPI dbCopyStatus(
		FLMUINT64		ui64BytesToCopy,
		FLMUINT64		ui64BytesCopied,
		FLMBOOL			bNewSrcFile,
		const char *	pszSrcFileName,
		const char *	pszDestFileName)
	{
		return( m_fnCopyStatus( ui64BytesToCopy, ui64BytesCopied, bNewSrcFile,
									pszSrcFileName, pszDestFileName));
	}
		
private:

	DB_COPY_STATUS	m_fnCopyStatus;

};

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbCopy(
	FLMUINT64			ui64This,
	const char *		pszSrcDbName,
	const char *		pszSrcDataDir,
	const char *		pszSrcRflDir,
	const char *		pszDestDbName,
	const char *		pszDestDataDir,
	const char *		pszDestRflDir,
	DB_COPY_STATUS		fnCopyStatus)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_DbCopyStatus *	pDbCopyStatus = NULL;

	if (fnCopyStatus)
	{
		if ((pDbCopyStatus = f_new CS_DbCopyStatus( fnCopyStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbCopy( pszSrcDbName, pszSrcDataDir, pszSrcRflDir, pszDestDbName,
								pszDestDataDir, pszDestRflDir, pDbCopyStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbCopyStatus)
	{
		pDbCopyStatus->Release();
	}

	return( rc);
}

typedef RCODE (FLMAPI * DB_RENAME_STATUS)(
	const char *	pszSrcFileName,
	const char *	pszDestFileName);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbRenameStatus : public IF_DbRenameStatus
{
public:

	CS_DbRenameStatus(
		DB_RENAME_STATUS	fnRenameStatus)
	{
		m_fnRenameStatus = fnRenameStatus;
	}

	virtual ~CS_DbRenameStatus()
	{
	}
	
	RCODE FLMAPI dbRenameStatus(
		const char *	pszSrcFileName,
		const char *	pszDestFileName)
	{
		return( m_fnRenameStatus( pszSrcFileName, pszDestFileName));
	}
		
private:

	DB_RENAME_STATUS	m_fnRenameStatus;

};

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbRename(
	FLMUINT64			ui64This,
	const char *		pszSrcDbName,
	const char *		pszSrcDataDir,
	const char *		pszSrcRflDir,
	const char *		pszDestDbName,
	FLMBOOL				bOverwriteDestOk,
	DB_RENAME_STATUS	fnRenameStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbSystem *			pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_DbRenameStatus *	pDbRenameStatus = NULL;

	if (fnRenameStatus)
	{
		if ((pDbRenameStatus = f_new CS_DbRenameStatus( fnRenameStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbRename( pszSrcDbName, pszSrcDataDir, pszSrcRflDir, pszDestDbName,
								bOverwriteDestOk, pDbRenameStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbRenameStatus)
	{
		pDbRenameStatus->Release();
	}

	return( rc);
}

typedef RCODE (FLMAPI * DB_REBUILD_STATUS)(
	FLMBOOL					bHaveRebuildInfo,
	XFLM_REBUILD_INFO *	pRebuildInfo,
	XFLM_CORRUPT_INFO *	pCorruptInfo);

/****************************************************************************
Desc:
****************************************************************************/
class CS_DbRebuildStatus : public IF_DbRebuildStatus
{
public:

	CS_DbRebuildStatus(
		DB_REBUILD_STATUS	fnDbRebuildStatus)
	{
		m_fnDbRebuildStatus = fnDbRebuildStatus;
	}

	virtual ~CS_DbRebuildStatus()
	{
	}

	RCODE FLMAPI reportRebuild(
		XFLM_REBUILD_INFO *	pRebuildInfo)
	{
		return( m_fnDbRebuildStatus( TRUE, pRebuildInfo, NULL));
	}
	
	RCODE FLMAPI reportRebuildErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo)
	{
		return( m_fnDbRebuildStatus( FALSE, NULL, pCorruptInfo));
	}

private:

	DB_REBUILD_STATUS	m_fnDbRebuildStatus;
};

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_dbRebuild(
	FLMUINT64				ui64This,
	const char *			pszSourceDbPath,
	const char *			pszSourceDataDir,
	const char *			pszDestDbPath,
	const char *			pszDestDataDir,
	const char *			pszDestRflDir,
	const char *			pszDictPath,
	const char *			pszPassword,
	XFLM_CREATE_OPTS *	pCreateOpts,
	DB_REBUILD_STATUS		fnRebuildStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbSystem *			pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_DbRebuildStatus *	pDbRebuildStatus = NULL;
	FLMUINT64				ui64TotNodes;
	FLMUINT64				ui64NodesRecov;
	FLMUINT64				ui64DiscardedDocs;

	if (fnRebuildStatus)
	{
		if ((pDbRebuildStatus = f_new CS_DbRebuildStatus( fnRebuildStatus)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
 
	if (RC_BAD( rc = pDbSystem->dbRebuild( pszSourceDbPath, pszSourceDataDir,
								pszDestDbPath, pszDestDataDir, pszDestRflDir,
								pszDictPath, pszPassword, pCreateOpts,
								&ui64TotNodes, &ui64NodesRecov, &ui64DiscardedDocs,
								pDbRebuildStatus)))
	{
		goto Exit;
	}

Exit:

	if (pDbRebuildStatus)
	{
		pDbRebuildStatus->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openBufferIStream(
	const char *			pszBuffer,
	FLMUINT64 *				pui64IStream)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiStrCharCount = f_strlen( pszBuffer);
	F_BufferIStream *	pIStream = NULL;
	char *				pszAllocBuffer = NULL;
	
	// Create the buffer stream object.
	
	if ((pIStream = f_new F_BufferIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	// Call the openStream method so that it will allocate a buffer internally.
	
	if (RC_BAD( rc = pIStream->openStream( NULL, uiStrCharCount, &pszAllocBuffer)))
	{
		goto Exit;
	}
	
	// Copy the data from the passed in string into pucBuffer, excluding the null
	// terminating byte.
	
	f_memcpy( pszAllocBuffer, pszBuffer, uiStrCharCount);
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openFileIStream(
	FLMUINT64		ui64This,
	const char *	pszFileName,
	FLMUINT64 *		pui64IStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_PosIStream *	pIStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openFileIStream( pszFileName, &pIStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openMultiFileIStream(
	FLMUINT64		ui64This,
	const char *	pszDirectory,
	const char *	pszBaseName,
	FLMUINT64 *		pui64IStream)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DbSystem *	pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_IStream *	pIStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openMultiFileIStream( pszDirectory, pszBaseName, &pIStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openBufferedIStream(
	FLMUINT64		ui64This,
	FLMUINT64		ui64InputIStream,
	FLMUINT			uiBufferSize,
	FLMUINT64 *		pui64IStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)ui64InputIStream);
	IF_IStream *		pIStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openBufferedIStream( pInputStream, uiBufferSize, &pIStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openUncompressingIStream(
	FLMUINT64		ui64This,
	FLMUINT64		ui64InputIStream,
	FLMUINT64 *		pui64IStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)ui64InputIStream);
	IF_IStream *		pIStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openUncompressingIStream( pInputStream, &pIStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openBase64Encoder(
	FLMUINT64		ui64This,
	FLMUINT64		ui64InputIStream,
	FLMBOOL			bInsertLineBreaks,
	FLMUINT64 *		pui64IStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)ui64InputIStream);
	IF_IStream *		pIStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openBase64Encoder( pInputStream, bInsertLineBreaks, &pIStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openBase64Decoder(
	FLMUINT64		ui64This,
	FLMUINT64		ui64InputIStream,
	FLMUINT64 *		pui64IStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)ui64InputIStream);
	IF_IStream *		pIStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openBase64Decoder( pInputStream, &pIStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64IStream = (FLMUINT64)((FLMUINT)pIStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openFileOStream(
	FLMUINT64		ui64This,
	const char *	pszFileName,
	FLMBOOL			bTruncateIfExists,
	FLMUINT64 *		pui64OStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_OStream *		pOStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openFileOStream( pszFileName, bTruncateIfExists, &pOStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64OStream = (FLMUINT64)((FLMUINT)pOStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openMultiFileOStream(
	FLMUINT64		ui64This,
	const char *	pszDirectory,
	const char *	pszBaseName,
	FLMUINT			uiMaxFileSize,
	FLMBOOL			bOkToOverwrite,
	FLMUINT64 *		pui64OStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_OStream *		pOStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openMultiFileOStream( pszDirectory, pszBaseName,
								uiMaxFileSize, bOkToOverwrite, &pOStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64OStream = (FLMUINT64)((FLMUINT)pOStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_removeMultiFileStream(
	FLMUINT64		ui64This,
	const char *	pszDirectory,
	const char *	pszBaseName)
{
	IF_DbSystem *	pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	
	return( pDbSystem->removeMultiFileStream( pszDirectory, pszBaseName));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openBufferedOStream(
	FLMUINT64		ui64This,
	FLMUINT64		ui64InputOStream,
	FLMUINT			uiBufferSize,
	FLMUINT64 *		pui64OStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_OStream *		pInputOStream = (IF_OStream *)((FLMUINT)ui64InputOStream);
	IF_OStream *		pOStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openBufferedOStream( pInputOStream,
								uiBufferSize, &pOStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64OStream = (FLMUINT64)((FLMUINT)pOStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_openCompressingOStream(
	FLMUINT64		ui64This,
	FLMUINT64		ui64InputOStream,
	FLMUINT64 *		pui64OStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_OStream *		pInputOStream = (IF_OStream *)((FLMUINT)ui64InputOStream);
	IF_OStream *		pOStream = NULL;
	
	if (RC_BAD( rc = pDbSystem->openCompressingOStream( pInputOStream, &pOStream)))
	{
		goto Exit;
	}
	
Exit:

	*pui64OStream = (FLMUINT64)((FLMUINT)pOStream);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DbSystem_writeToOStream(
	FLMUINT64		ui64This,
	FLMUINT64		ui64IStream,
	FLMUINT64		ui64OStream)
{
	IF_DbSystem *		pDbSystem = (IF_DbSystem *)((FLMUINT)ui64This);
	IF_IStream *		pIStream = (IF_IStream *)((FLMUINT)ui64IStream);
	IF_OStream *		pOStream = (IF_OStream *)((FLMUINT)ui64OStream);
	
	return( pDbSystem->writeToOStream( pIStream, pOStream));
}
