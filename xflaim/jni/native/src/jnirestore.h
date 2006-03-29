//------------------------------------------------------------------------------
// Desc:
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: $
//------------------------------------------------------------------------------

/****************************************************************************
Desc:
****************************************************************************/
class JNIRestoreClient : public IF_RestoreClient, public XF_Base
{
public:

	JNIRestoreClient(
		jobject			jClient,
		JavaVM *			pJvm)
	{
		flmAssert( jClient);
		flmAssert( pJvm);
		m_jClient = jClient;
		m_pJvm = pJvm;
	}

	RCODE XFLMAPI openBackupSet( void);

	RCODE XFLMAPI openRflFile(
		FLMUINT			uiFileNum);

	RCODE XFLMAPI openIncFile(
		FLMUINT			uiFileNum);

	RCODE XFLMAPI read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE XFLMAPI close( void);

	RCODE XFLMAPI abortFile( void);
	
	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_RestoreClient::getRefCount());
	}

	virtual FINLINE FLMUINT32 XFLMAPI AddRef( void)
	{
		return( IF_RestoreClient::AddRef());
	}

	virtual FINLINE FLMUINT32 XFLMAPI Release( void)
	{
		return( IF_RestoreClient::Release());
	}

private:

	jobject		m_jClient;
	JavaVM *		m_pJvm;
};

/****************************************************************************
Desc:
****************************************************************************/
class JNIRestoreStatus : public IF_RestoreStatus, public XF_Base
{
public:

	JNIRestoreStatus(
		jobject				jStatus,
		JavaVM *				pJvm)
	{
		flmAssert( jStatus);
		flmAssert( pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}
	
	RCODE XFLMAPI reportProgress(
		eRestoreAction *	peAction,
		FLMUINT64			ui64BytesToDo,
		FLMUINT64			ui64BytesDone);

	RCODE XFLMAPI reportError(
		eRestoreAction *	peAction,
		RCODE					rcErr);

	RCODE XFLMAPI reportOpenRflFile(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum);

	RCODE XFLMAPI reportRflRead(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum,
		FLMUINT				uiBytesRead);

	RCODE XFLMAPI reportBeginTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLMAPI reportCommitTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLMAPI reportAbortTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLMAPI reportBlockChainFree(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT64			ui64MaintDocNum,
		FLMUINT				uiStartBlkAddr,
		FLMUINT				uiEndBlkAddr,
		FLMUINT				uiCount);

	RCODE XFLMAPI reportIndexSuspend(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiIndexNum);

	RCODE XFLMAPI reportIndexResume(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiIndexNum);

	RCODE XFLMAPI reportReduce(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCount);

	RCODE XFLMAPI reportUpgrade(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiOldDbVersion,
		FLMUINT				uiNewDbVersion);

	RCODE XFLMAPI reportEnableEncryption(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE XFLMAPI reportWrapKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);
		
	RCODE XFLMAPI reportRollOverDbKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);
		
	RCODE XFLMAPI reportDocumentDone(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLMAPI reportNodeDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLMAPI reportAttributeDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ElementId,
		FLMUINT				uiAttrNameId);
			
	RCODE XFLMAPI reportNodeChildrenDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiNameId);
		
	RCODE XFLMAPI reportNodeCreate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64RefNodeId,
		eDomNodeType		eNodeType,
		FLMUINT				uiNameId,
		eNodeInsertLoc		eLocation);
		
	RCODE XFLMAPI reportInsertBefore(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ParentId,
		FLMUINT64			ui64NewChildId,
		FLMUINT64			ui64RefChildId);
		
	RCODE XFLMAPI reportNodeUpdate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLMAPI reportNodeSetValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE XFLMAPI reportAttributeSetValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ElementNodeId,
		FLMUINT				uiAttrNameId);
		
	RCODE XFLMAPI reportNodeFlagsUpdate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiFlags,
		FLMBOOL				bAdd);
		
	RCODE XFLMAPI reportNodeSetPrefixId(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiAttrNameId,
		FLMUINT				uiPrefixId);
			
	RCODE XFLMAPI reportNodeSetMetaValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT64			ui64MetaValue);
		
	RCODE XFLMAPI reportSetNextNodeId(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NextNodeId);

	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_RestoreStatus::getRefCount());
	}

	virtual FINLINE FLMUINT32 XFLMAPI AddRef( void)
	{
		return( IF_RestoreStatus::AddRef());
	}

	virtual FINLINE FLMUINT32 XFLMAPI Release( void)
	{
		return( IF_RestoreStatus::Release());
	}

private:

	jobject			m_jStatus;
	JavaVM *			m_pJvm;
};
