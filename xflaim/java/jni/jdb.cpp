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

FSTATIC RCODE getUniString(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf);
	
FSTATIC RCODE getDictName(
	IF_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	F_DynaBuf *		pDynaBuf);
	
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
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin__JIII(
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
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin__JJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lSrcDb)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_Db *			pSrcDb = ((IF_Db *)(FLMUINT)lSrcDb);

	if (RC_BAD( rc = pDb->transBegin( pSrcDb)))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1getTransType(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *			pDb = THIS_FDB();
	
	return( (jint)pDb->getTransType());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1doCheckpoint(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iTimeout)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->doCheckpoint( (FLMUINT)iTimeout)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1dbLock(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iLockType,
	jint				iPriority,
	jint				iTimeout)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->dbLock( (eLockType)iLockType,
									(FLMINT)iPriority, (FLMUINT)iTimeout)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1dbUnlock(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->dbUnlock()))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockType(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMBOOL			bImplicit;
	
	if (RC_BAD( rc = pDb->getLockType( &lockType, &bImplicit)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (int)lockType);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getLockImplicit(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMBOOL			bImplicit;
	
	if (RC_BAD( rc = pDb->getLockType( &lockType, &bImplicit)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( bImplicit ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockThreadId(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiThreadId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockNumExclQueued(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumExclQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockNumSharedQueued(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumSharedQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockPriorityCount(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumSharedQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1indexSuspend(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->indexSuspend( (FLMUINT)iIndexNum)))
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
JNIEXPORT void JNICALL Java_xflaim_Db__1indexResume(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->indexResume( (FLMUINT)iIndexNum)))
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
JNIEXPORT jint JNICALL Java_xflaim_Db__1reduceSize(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iCount)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiCount = 0;
	
	if (RC_BAD( rc = pDb->reduceSize( (FLMUINT)iCount, &uiCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiCount);
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
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getFirstDocument( (FLMUINT)iCollection, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getLastDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getLastDocument( (FLMUINT)iCollection, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jint				iFlags,
	jlong				lDocumentId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getDocument( (FLMUINT)iCollection,
									(FLMUINT)iFlags, (FLMUINT64)lDocumentId,
									&pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1documentDone__JIJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lDocumentId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->documentDone( (FLMUINT)iCollection,
									(FLMUINT64)lDocumentId)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1documentDone__JJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lNode)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)lNode);
	IF_Db *			pDb = THIS_FDB();

	if (RC_BAD( rc = pDb->documentDone( pNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
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
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);

	if (RC_BAD( rc = pDb->getNode( (FLMUINT)iCollection, (FLMUINT64)lNodeId,
								   &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;		
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lElementNodeId,
	jint				iAttrNameId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);

	if (RC_BAD( rc = pDb->getAttribute( (FLMUINT)iCollection,
									(FLMUINT64)lElementNodeId,
								   (FLMUINT)iAttrNameId, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;		
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
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
FSTATIC RCODE getUniString(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf)
{
	RCODE						rc = NE_XFLM_OK;
	const FLMUNICODE *	puzStr = NULL;
	FLMUINT					uiStrCharCount;
	
	if (sStr)
	{
		puzStr = (const FLMUNICODE *)pEnv->GetStringChars( sStr, NULL);
		uiStrCharCount = (FLMUINT)pEnv->GetStringLength( sStr);
		if (RC_BAD( rc = pDynaBuf->appendData( puzStr,
									sizeof( FLMUNICODE) * uiStrCharCount)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDynaBuf->appendUniChar( 0)))
		{
			goto Exit;
		}
	}
	else
	{
		pDynaBuf->truncateData( 0);
	}
	
Exit:

	if (puzStr)
	{
		pEnv->ReleaseStringChars( sStr, puzStr);
	}

	return( rc);
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
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createElementDef( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(),
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getElementNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getElementNameId( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createUniqueElmDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createUniqueElmDef( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(),
											&uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createAttributeDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sAttributeName,
	jint				iDataType,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucAttributeNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	attributeName( ucAttributeNameBuf, sizeof( ucAttributeNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sAttributeName);
	if (RC_BAD( rc = getUniString( pEnv, sAttributeName, &attributeName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createAttributeDef( namespaceURI.getUnicodePtr(),
											attributeName.getUnicodePtr(),
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getAttributeNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sAttributeName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucAttributeNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	attributeName( ucAttributeNameBuf, sizeof( ucAttributeNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sAttributeName);
	if (RC_BAD( rc = getUniString( pEnv, sAttributeName, &attributeName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getAttributeNameId( namespaceURI.getUnicodePtr(),
											attributeName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createPrefixDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPrefixName,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucPrefixNameBuf [200];
	F_DynaBuf	prefixName( ucPrefixNameBuf, sizeof( ucPrefixNameBuf));
	
	flmAssert( sPrefixName);
	if (RC_BAD( rc = getUniString( pEnv, sPrefixName, &prefixName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createPrefixDef( prefixName.getUnicodePtr(),
														&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getPrefixId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPrefixName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucPrefixNameBuf [200];
	F_DynaBuf	prefixName( ucPrefixNameBuf, sizeof( ucPrefixNameBuf));
	
	flmAssert( sPrefixName);
	if (RC_BAD( rc = getUniString( pEnv, sPrefixName, &prefixName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getPrefixId( prefixName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createEncDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sEncType,
	jstring			sEncName,
	jint				iKeySize,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucEncTypeBuf [200];
	FLMBYTE		ucEncNameBuf [200];
	F_DynaBuf	encType( ucEncTypeBuf, sizeof( ucEncTypeBuf));
	F_DynaBuf	encName( ucEncNameBuf, sizeof( ucEncNameBuf));
	
	flmAssert( sEncType);
	if (RC_BAD( rc = getUniString( pEnv, sEncType, &encType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sEncName);
	if (RC_BAD( rc = getUniString( pEnv, sEncName, &encName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createEncDef( encType.getUnicodePtr(),
														encName.getUnicodePtr(),
														(FLMUINT)iKeySize,
														&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getEncDefId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sEncName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucEncNameBuf [200];
	F_DynaBuf	encName( ucEncNameBuf, sizeof( ucEncNameBuf));
	
	flmAssert( sEncName);
	if (RC_BAD( rc = getUniString( pEnv, sEncName, &encName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getEncDefId( encName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createCollectionDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sCollectionName,
	jint				iEncNumber,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucCollectionNameBuf [200];
	F_DynaBuf	collectionName( ucCollectionNameBuf, sizeof( ucCollectionNameBuf));
	
	flmAssert( sCollectionName);
	if (RC_BAD( rc = getUniString( pEnv, sCollectionName, &collectionName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createCollectionDef( collectionName.getUnicodePtr(),
														&uiNameId, (FLMUINT)iEncNumber)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getCollectionNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sCollectionName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucCollectionNameBuf [200];
	F_DynaBuf	collectionName( ucCollectionNameBuf, sizeof( ucCollectionNameBuf));
	
	flmAssert( sCollectionName);
	if (RC_BAD( rc = getUniString( pEnv, sCollectionName, &collectionName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getCollectionNumber( collectionName.getUnicodePtr(),
										&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getIndexNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sIndexName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucIndexNameBuf [200];
	F_DynaBuf	indexName( ucIndexNameBuf, sizeof( ucIndexNameBuf));
	
	flmAssert( sIndexName);
	if (RC_BAD( rc = getUniString( pEnv, sIndexName, &indexName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getIndexNumber( indexName.getUnicodePtr(),
										&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDictionaryDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getDictionaryDef( (FLMUINT)iDictType,
									(FLMUINT)iDictNumber, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE getDictName(
	IF_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	F_DynaBuf *		pDynaBuf)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNameSize = 0;
	FLMUNICODE *	puzName = NULL;
	
	// Determine how much space is needed to get the name.

	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											(FLMUNICODE *)NULL, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
		{
			goto Exit;
		}
	}
		
	// uiNameSize comes back as number of characters, so to
	// get the buffer size needed, we need to add one for a null
	// terminator, and then multiply by the size of a unicode character.
	
	uiNameSize++;
	uiNameSize *= sizeof( FLMUNICODE);
	
	if (RC_BAD( rc = pDynaBuf->allocSpace( uiNameSize, (void **)&puzName)))
	{
		goto Exit;
	}
	
	// Now get the name.
	
	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											puzName, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											puzName, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getDictionaryName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, (FLMUINT)iDictType, (FLMUINT)iDictNumber,
									FALSE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getElementNamespace(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, ELM_ELEMENT_TAG, (FLMUINT)iDictNumber,
									TRUE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getAttributeNamespace(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, ELM_ATTRIBUTE_TAG, (FLMUINT)iDictNumber,
									TRUE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
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
