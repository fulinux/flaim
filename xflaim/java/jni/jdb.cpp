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

#include "xflaim.h"
#include "xflaim_Db.h"
#include "jniftk.h"

#define THIS_FDB() \
	((IF_Db *)(FLMUINT)lThis)

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_FDB()->Release();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iTransactionType,
	jint				iMaxLockWait,
	jint				iFlags)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();

	if (RC_BAD( rc = pDb->transBegin( (eDbTransType)iTransactionType,
		(FLMUINT)iMaxLockWait, (FLMUINT)iFlags)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transCommit(
	JNIEnv *			pEnv,
	jobject			obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();

	(void)obj;
	
	if (RC_BAD( rc = pDb->transCommit()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transAbort(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->transAbort()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1import(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jobject			jIStream,
	jint				iCollection)
{
	RCODE						rc = NE_XFLM_OK;
	IF_PosIStream *		pIStream = NULL;
	IF_Db *					pDb = THIS_FDB();
	
	jclass class_JIStream = pEnv->FindClass( "xflaim/PosIStream");
	jfieldID fid_this = pEnv->GetFieldID( class_JIStream, "m_this", "J");
	pIStream = (IF_PosIStream *)((FLMUINT)pEnv->GetLongField( jIStream, fid_this));
	
	if (!pIStream)
	{
		ThrowError( NE_XFLM_FAILURE, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = pDb->import( pIStream, (FLMUINT)iCollection)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getFirstDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jobject			jNode)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = NULL;
	IF_Db *			pDb = THIS_FDB();
	
	// Get the real pNode if one was passed in so we can pass it to the
	// getFirstDocument method.
	
	if (jNode)
	{
		jclass class_JDOMNode = pEnv->FindClass( "xflaim.DOMNode");
		jfieldID fid_this = pEnv->GetFieldID( class_JDOMNode, "m_this", "Z");
		
		pNode = (IF_DOMNode *)(FLMUINT)pEnv->GetLongField( jNode, fid_this);
		
		// Clear the jNode's reference to this node as it is going to go away.
		
		pEnv->SetLongField( jNode, fid_this, (jlong)0);
	}
	
	if (RC_BAD( rc = pDb->getFirstDocument( (FLMUINT)iCollection, &pNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lNodeId,
	jlong				lpOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lpOldNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lpOldNodeRef;	
	}

	if (RC_BAD( rc = pDb->getNode( (FLMUINT)iCollection, (FLMUINT64)lNodeId,
								   &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;		
	}
	
Exit:

	return( (jlong)(FLMUINT)ifpNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1createDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (RC_BAD( rc = pDb->createDocument((FLMUINT)iCollection, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)(FLMUINT)ifpNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1createRootElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jint				iTag)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (RC_BAD( rc = pDb->createRootElement((FLMUINT)iCollection,
			(FLMUINT)iTag, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)(FLMUINT)ifpNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createElementDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName,
	jint				iDataType,
	jint				iRequestedNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiNameId = iRequestedNum;
	jchar *			pszNamespaceURI = NULL;
	jchar *			pszElementName;
	
	if (sNamespaceURI)
	{
		pszNamespaceURI = (jchar *)pEnv->GetStringCritical( sNamespaceURI, NULL);
	}
	
	flmAssert( sElementName);
	pszElementName = (jchar *)pEnv->GetStringCritical( sElementName, NULL);
	
	if (RC_BAD( rc = pDb->createElementDef( pszNamespaceURI, pszElementName,
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:
	
	if (pszNamespaceURI)
	{
		pEnv->ReleaseStringCritical( sNamespaceURI, pszNamespaceURI);
	}
	
	pEnv->ReleaseStringCritical( sElementName, pszElementName);
	return( (jlong)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1backupBegin(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				eBackupType,
	jint				eTransType,
	jint				iMaxLockWait,
	jlong				lReusedRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_Backup *		ifpBackup = NULL;
	
	if (lReusedRef)
	{
		ifpBackup = (IF_Backup *)(FLMUINT)lReusedRef;
	}
	
	if (RC_BAD( rc = pDb->backupBegin( (eDbBackupType)eBackupType, 
			(eDbTransType)eTransType, (FLMUINT)iMaxLockWait, &ifpBackup)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
		
Exit:

	return( (jlong)(FLMUINT)ifpBackup);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1keyRetrieve(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iIndex,
	jlong				lKey,
	jint				iFlags,
	jlong				lFoundKey)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pSearchKey = (IF_DataVector *)(FLMUINT)lKey;
	IF_DataVector *	pFoundKey = (IF_DataVector *)(FLMUINT)lFoundKey;
	IF_Db *				pDb = THIS_FDB();
	FLMUINT				uiIndex = (FLMUINT)iIndex;
	FLMUINT				uiFlags = (FLMUINT)iFlags;
	
	if (RC_BAD( rc = pDb->keyRetrieve( uiIndex, pSearchKey, uiFlags, pFoundKey)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}
