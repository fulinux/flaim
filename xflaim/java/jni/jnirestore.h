//------------------------------------------------------------------------------
// Desc:
// Tabs:	3
//
// Copyright (c) 2003-2007 Novell, Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Library Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, contact Novell, Inc.
//
// To contact Novell about this file by physical or electronic mail, 
// you may find current contact information at www.novell.com.
//
// $Id$
//------------------------------------------------------------------------------

/****************************************************************************
Desc:
****************************************************************************/
class JNIRestoreClient : public IF_RestoreClient
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

	RCODE FLMAPI openBackupSet( void);

	RCODE FLMAPI openRflFile(
		FLMUINT			uiFileNum);

	RCODE FLMAPI openIncFile(
		FLMUINT			uiFileNum);

	RCODE FLMAPI read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE FLMAPI close( void);

	RCODE FLMAPI abortFile( void);
	
	FINLINE FLMINT FLMAPI getRefCount( void)
	{
		return( IF_RestoreClient::getRefCount());
	}

	virtual FINLINE FLMINT FLMAPI AddRef( void)
	{
		return( IF_RestoreClient::AddRef());
	}

	virtual FINLINE FLMINT FLMAPI Release( void)
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
class JNIRestoreStatus : public IF_RestoreStatus
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
	
	RCODE FLMAPI reportProgress(
		eRestoreAction *	peAction,
		FLMUINT64			ui64BytesToDo,
		FLMUINT64			ui64BytesDone);

	RCODE FLMAPI reportError(
		eRestoreAction *	peAction,
		RCODE					rcErr);

	RCODE FLMAPI reportOpenRflFile(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum);

	RCODE FLMAPI reportRflRead(
		eRestoreAction *	peAction,
		FLMUINT				uiFileNum,
		FLMUINT				uiBytesRead);

	RCODE FLMAPI reportBeginTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE FLMAPI reportCommitTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE FLMAPI reportAbortTrans(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE FLMAPI reportBlockChainFree(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT64			ui64MaintDocNum,
		FLMUINT				uiStartBlkAddr,
		FLMUINT				uiEndBlkAddr,
		FLMUINT				uiCount);

	RCODE FLMAPI reportIndexSuspend(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiIndexNum);

	RCODE FLMAPI reportIndexResume(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiIndexNum);

	RCODE FLMAPI reportReduce(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCount);

	RCODE FLMAPI reportUpgrade(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiOldDbVersion,
		FLMUINT				uiNewDbVersion);

	RCODE FLMAPI reportEnableEncryption(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);

	RCODE FLMAPI reportWrapKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);
		
	RCODE FLMAPI reportRollOverDbKey(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId);
		
	RCODE FLMAPI reportDocumentDone(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE FLMAPI reportNodeDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE FLMAPI reportAttributeDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ElementId,
		FLMUINT				uiAttrNameId);
			
	RCODE FLMAPI reportNodeChildrenDelete(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiNameId);
		
	RCODE FLMAPI reportNodeCreate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64RefNodeId,
		eDomNodeType		eNodeType,
		FLMUINT				uiNameId,
		eNodeInsertLoc		eLocation);
		
	RCODE FLMAPI reportInsertBefore(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ParentId,
		FLMUINT64			ui64NewChildId,
		FLMUINT64			ui64RefChildId);
		
	RCODE FLMAPI reportNodeUpdate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE FLMAPI reportNodeSetValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);
		
	RCODE FLMAPI reportAttributeSetValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64ElementNodeId,
		FLMUINT				uiAttrNameId);
		
	RCODE FLMAPI reportNodeFlagsUpdate(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiFlags,
		FLMBOOL				bAdd);
		
	RCODE FLMAPI reportNodeSetPrefixId(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiAttrNameId,
		FLMUINT				uiPrefixId);
			
	RCODE FLMAPI reportNodeSetMetaValue(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT64			ui64MetaValue);
		
	RCODE FLMAPI reportSetNextNodeId(
		eRestoreAction *	peAction,
		FLMUINT64			ui64TransId,
		FLMUINT				uiCollection,
		FLMUINT64			ui64NextNodeId);

	FINLINE FLMINT FLMAPI getRefCount( void)
	{
		return( IF_RestoreStatus::getRefCount());
	}

	virtual FINLINE FLMINT FLMAPI AddRef( void)
	{
		return( IF_RestoreStatus::AddRef());
	}

	virtual FINLINE FLMINT FLMAPI Release( void)
	{
		return( IF_RestoreStatus::Release());
	}

private:

	jobject			m_jStatus;
	JavaVM *			m_pJvm;
};
