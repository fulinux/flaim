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
	IF_DbSystem *	pDbSystem = ((IF_DbSystem *)(FLMUINT)ui64This);
	
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
	IF_DbSystem *	pDbSystem = ((IF_DbSystem *)(FLMUINT)ui64This);

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
	IF_DbSystem *	pDbSystem = ((IF_DbSystem *)(FLMUINT)ui64This);
	
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
	IF_DbSystem *	pDbSystem = ((IF_DbSystem *)(FLMUINT)ui64This);
 
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
	void *					pvObj,
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
		void *			pvObj,
		RESTORE_CLIENT	fnRestoreClient)
	{
		m_pvObj = pvObj;
		m_fnRestoreClient = fnRestoreClient;
	}

	virtual ~CS_RestoreClient()
	{
	}

	RCODE FLMAPI openBackupSet( void)
	{
		return( m_fnRestoreClient( m_pvObj, RESTORE_OPEN_BACKUP_SET, 0, 0, NULL, NULL));
	}

	RCODE FLMAPI openRflFile(
		FLMUINT	uiFileNum)
	{
		return( m_fnRestoreClient( m_pvObj, RESTORE_OPEN_RFL_FILE, uiFileNum, 0, NULL, NULL));
	}

	RCODE FLMAPI openIncFile(
		FLMUINT	uiFileNum)
	{
		return( m_fnRestoreClient( m_pvObj, RESTORE_OPEN_INC_FILE, uiFileNum, 0, NULL, NULL));
	}

	RCODE FLMAPI read(
		FLMUINT		uiBytesRequested,
		void *		pvBuffer,
		FLMUINT *	puiBytesRead)
	{
		return( m_fnRestoreClient( m_pvObj, RESTORE_READ, 0, uiBytesRequested,
					pvBuffer, puiBytesRead));
	}

	
	RCODE FLMAPI close( void)
	{
		return( m_fnRestoreClient( m_pvObj, RESTORE_CLOSE, 0, 0, NULL, NULL));
	}

	RCODE FLMAPI abortFile( void)
	{
		return( m_fnRestoreClient( m_pvObj, RESTORE_ABORT_FILE, 0, 0, NULL, NULL));
	}

private:

	void *			m_pvObj;
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
	void *					pvObj,
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
		void *			pvObj,
		RESTORE_STATUS	fnRestoreStatus)
	{
		m_pvObj = pvObj;
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_PROGRESS, peAction, 0,
					ui64BytesToDo, ui64BytesDone, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportError(
		eRestoreAction *		peAction,
		RCODE						rcErr)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_ERROR, peAction, 0,
					0, 0, 0,
					(FLMUINT)rcErr, 0, 0, 0));
	}

	RCODE FLMAPI reportOpenRflFile(
		eRestoreAction *		peAction,
		FLMUINT					uiFileNum)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_OPEN_RFL_FILE, peAction, 0,
					0, 0, 0,
					uiFileNum, 0, 0, 0));
	}

	RCODE FLMAPI reportRflRead(
		eRestoreAction *		peAction,
		FLMUINT					uiFileNum,
		FLMUINT					uiBytesRead)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_RFL_READ, peAction, 0,
					0, 0, 0,
					uiFileNum, uiBytesRead, 0, 0));
	}

	RCODE FLMAPI reportBeginTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_BEGIN_TRANS, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportCommitTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_COMMIT_TRANS, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportAbortTrans(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_ABORT_TRANS, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_BLOCK_CHAIN_FREE, peAction, ui64TransId,
					ui64MaintDocNum, 0, 0,
					uiStartBlkAddr, uiEndBlkAddr, uiCount, 0));
	}

	RCODE FLMAPI reportIndexSuspend(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiIndexNum)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_INDEX_SUSPEND, peAction, ui64TransId,
					0, 0, 0,
					uiIndexNum, 0, 0, 0));
	}

	RCODE FLMAPI reportIndexResume(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiIndexNum)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_INDEX_RESUME, peAction, ui64TransId,
					0, 0, 0,
					uiIndexNum, 0, 0, 0));
	}

	RCODE FLMAPI reportReduce(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCount)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_REDUCE, peAction, ui64TransId,
					0, 0, 0,
					uiCount, 0, 0, 0));
	}

	RCODE FLMAPI reportUpgrade(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiOldDbVersion,
		FLMUINT					uiNewDbVersion)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_UPGRADE, peAction, ui64TransId,
					0, 0, 0,
					uiOldDbVersion, uiNewDbVersion, 0, 0));
	}

	RCODE FLMAPI reportEnableEncryption(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_ENABLE_ENCRYPTION, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}

	RCODE FLMAPI reportWrapKey(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_WRAP_KEY, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}
		
	RCODE FLMAPI reportRollOverDbKey(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_ROLL_OVER_DB_KEY, peAction, ui64TransId,
					0, 0, 0,
					0, 0, 0, 0));
	}
		
	RCODE FLMAPI reportDocumentDone(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64DocumentId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_DOCUMENT_DONE, peAction, ui64TransId,
					ui64DocumentId, 0, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportNodeDelete(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_DELETE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_ATTRIBUTE_DELETE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_CHILDREN_DELETE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_CREATE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_INSERT_BEFORE, peAction, ui64TransId,
					ui64ParentNodeId, ui64NewChildNodeId, ui64RefChildNodeId,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportNodeUpdate(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_UPDATE, peAction, ui64TransId,
					ui64NodeId, 0, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportNodeSetValue(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_SET_VALUE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_ATTRIBUTE_SET_VALUE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_FLAGS_UPDATE, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_SET_PREFIX_ID, peAction, ui64TransId,
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
		return( m_fnRestoreStatus( m_pvObj, REPORT_NODE_SET_META_VALUE, peAction, ui64TransId,
					ui64NodeId, ui64MetaValue, 0,
					uiCollection, 0, 0, 0));
	}
		
	RCODE FLMAPI reportSetNextNodeId(
		eRestoreAction *		peAction,
		FLMUINT64				ui64TransId,
		FLMUINT					uiCollection,
		FLMUINT64				ui64NextNodeId)
	{
		return( m_fnRestoreStatus( m_pvObj, REPORT_SET_NEXT_NODE_ID, peAction, ui64TransId,
					ui64NextNodeId, 0, 0,
					uiCollection, 0, 0, 0));
	}


private:

	void *			m_pvObj;
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
	void *				pvRestoreClientObj,
	RESTORE_CLIENT		fnRestoreClient,
	void *				pvRestoreStatusObj,
	RESTORE_STATUS		fnRestoreStatus)
{
	RCODE						rc = NE_XFLM_OK;
	IF_DbSystem *			pDbSystem = ((IF_DbSystem *)(FLMUINT)ui64This);
	IF_RestoreClient *	pRestoreClient = NULL;
	IF_RestoreStatus *	pRestoreStatus = NULL;

	if (pvRestoreClientObj)
	{
		if ((pRestoreClient = f_new CS_RestoreClient( pvRestoreClientObj, fnRestoreClient)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
	}
	if (pvRestoreStatusObj)
	{
		if ((pRestoreStatus = f_new CS_RestoreStatus( pvRestoreStatusObj, fnRestoreStatus)) == NULL)
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

typedef RCODE (FLMAPI * DB_COPY_STATUS)(
	void *			pvObj,
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
		void *			pvObj,
		DB_COPY_STATUS	fnCopyStatus)
	{
		m_pvObj = pvObj;
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
		return( m_fnCopyStatus( m_pvObj, ui64BytesToCopy, ui64BytesCopied,
								bNewSrcFile, pszSrcFileName, pszDestFileName));
	}
		
private:

	void *			m_pvObj;
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
	void *				pvObj,
	DB_COPY_STATUS		fnCopyStatus)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DbSystem *		pDbSystem = ((IF_DbSystem *)(FLMUINT)ui64This);
	IF_DbCopyStatus *	pDbCopyStatus = NULL;

	if (pvObj)
	{
		if ((pDbCopyStatus = f_new CS_DbCopyStatus( pvObj, fnCopyStatus)) == NULL)
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
