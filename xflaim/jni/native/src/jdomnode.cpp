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
#include "xflaim_DOMNode.h"
#include "jniftk.h"

#define THIS_NODE() \
	((IF_DOMNode *)(FLMUINT)lThis)

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iNodeType,
	jint				iNameId,
	jint				iInsertLoc,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if( RC_BAD( rc = pThisNode->createNode( ifpDb, (eDomNodeType)iNodeType, 
			(FLMUINT)iNameId, (eNodeInsertLoc)iInsertLoc, &ifpNewNode, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)ifpNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1deleteNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	
	if (RC_BAD( rc = pNode->deleteNode( ifpDb)))
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
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1deleteChildren(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	
	if (RC_BAD( rc = pNode->deleteChildren( ifpDb)))
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
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getNodeType(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_NODE()->getNodeType());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iNameId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if( RC_BAD( rc = pThisNode->createAttribute( ifpDb, 
		(FLMUINT)iNameId, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)ifpNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getFirstAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if ( RC_BAD( rc = pThisNode->getFirstAttribute( ifpDb, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)ifpNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLastAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if ( RC_BAD( rc = pThisNode->getLastAttribute( ifpDb, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)ifpNewNode));		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iAttributeId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if( RC_BAD( rc = pThisNode->getAttribute( ifpDb, (FLMUINT)iAttributeId,
											   &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)ifpNewNode));		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1deleteAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iAttributeId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	
	if (RC_BAD( rc = pThisNode->deleteAttribute( ifpDb, (FLMUINT)iAttributeId)))
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
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iAttributeId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	jboolean			bRv = false;
	
	rc = pThisNode->hasAttribute( ifpDb, (FLMUINT)iAttributeId, NULL);
	
	if (RC_OK( rc))
	{
		bRv = true;
	}
	else
	{
		bRv = false;
		if (rc != NE_XFLM_DOM_NODE_NOT_FOUND)
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
Exit:

	return( bRv);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasAttributes(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMBOOL			bHasAttr = FALSE;
	
	if( RC_BAD( rc = pThisNode->hasAttributes( ifpDb, &bHasAttr)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( bHasAttr ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasNextSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMBOOL			bHasNextSib = FALSE;
	
	if (RC_BAD( rc = pThisNode->hasNextSibling( ifpDb, &bHasNextSib)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bHasNextSib ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasPreviousSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMBOOL			bHasPreviousSib = FALSE;
	
	if (RC_BAD( rc = pThisNode->hasPreviousSibling( ifpDb, &bHasPreviousSib)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bHasPreviousSib ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasChildren(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMBOOL			bHasChild = FALSE;
	
	if (RC_BAD( rc = pThisNode->hasChildren( ifpDb, &bHasChild)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bHasChild ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1isNamespaceDecl(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMBOOL			bIsDecl = FALSE;
	
	if (RC_BAD( rc = pThisNode->isNamespaceDecl( ifpDb, &bIsDecl)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( bIsDecl ? true : false);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getParentNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getParentNode( ifpDb, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getFirstChild(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getFirstChild( ifpDb, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLastChild(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getLastChild( ifpDb, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getPreviousSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)(FLMUINT)lThis;
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getPreviousSibling( ifpDb, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNextSibling(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getNextSibling( ifpDb, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getPreviousDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getPreviousDocument( ifpDb, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNextDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getNextDocument( ifpDb, &ifpNewNode)))
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
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getPrefix(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUNICODE		uzPrefix[ 128];
	FLMUNICODE *	puzPrefix = uzPrefix;
	FLMUINT			uiBufSize = sizeof( uzPrefix);
	FLMUINT			uiNumChars;
	jstring			sPrefix = NULL;
	
	if (RC_BAD( rc = pThisNode->getPrefix( ifpDb, 
		(FLMUNICODE *)NULL, 0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof( FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzPrefix)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getPrefix( ifpDb, puzPrefix, uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sPrefix = pEnv->NewString( puzPrefix, uiNumChars);
	
Exit:

	if (puzPrefix != uzPrefix)
	{
		f_free( &puzPrefix);
	}	
	
	return( sPrefix);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getChildElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iNameId,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getChildElement( ifpDb, 
		(FLMUINT)iNameId, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getSiblingElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jint				iNameId,
	jboolean			bNext,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if (RC_BAD( rc = pThisNode->getSiblingElement( ifpDb, 
		(FLMUINT)iNameId, bNext ? TRUE : FALSE, &ifpNewNode)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getParentId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getParentId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNodeId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getNodeId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getDocumentId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getDocumentId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}
 
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getPrevSibId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getPrevSibId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getNextSibId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getNextSibId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getFirstChildId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getFirstChildId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLastChildId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT64		ui64Id;
	
	if (RC_BAD( rc = pThisNode->getLastChildId( ifpDb, &ui64Id)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)ui64Id);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiId;
	
	if (RC_BAD( rc = pThisNode->getNameId( ifpDb, &uiId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiId);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getNamespaceURI(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)	
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUNICODE		uzNamespaceURI[ 128];
	FLMUNICODE *	puzNamespaceURI = uzNamespaceURI;
	FLMUINT			uiBufSize = sizeof( uzNamespaceURI);
	FLMUINT			uiNumChars;
	jstring			sNamespaceURI = NULL;
	
	if (RC_BAD( rc = pThisNode->getNamespaceURI( ifpDb, 
		(FLMUNICODE *)NULL, 0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzNamespaceURI)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getNamespaceURI( ifpDb, puzNamespaceURI,
						uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sNamespaceURI = pEnv->NewString( puzNamespaceURI, uiNumChars);
	
Exit:

	if (puzNamespaceURI != uzNamespaceURI)
	{
		f_free( &puzNamespaceURI);
	}	
	
	return( sNamespaceURI);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getLocalName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)(FLMUINT)lThis;
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUNICODE		uzLocalName[ 128];
	FLMUNICODE *	puzLocalName = uzLocalName;
	FLMUINT			uiBufSize = sizeof(uzLocalName);
	FLMUINT			uiNumChars;
	jstring			sLocalName = NULL;
	
	if (RC_BAD( rc = pThisNode->getLocalName( ifpDb, (FLMUNICODE *)NULL, 
		0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}

	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzLocalName)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getLocalName( ifpDb, puzLocalName,
		uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sLocalName = pEnv->NewString( puzLocalName, uiNumChars);
	
Exit:

	if (puzLocalName != uzLocalName)
	{
		f_free( &puzLocalName);
	}	
	
	return( sLocalName);		
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getQualifiedName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)(FLMUINT)lThis;
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUNICODE		uzQualName[ 128];
	FLMUNICODE *	puzQualName = uzQualName;
	FLMUINT			uiNumChars;
	FLMUINT			uiBufSize = sizeof( uzQualName);
	jstring			sLocalName = NULL;
	
	if (RC_BAD( rc = pThisNode->getQualifiedName( ifpDb, (FLMUNICODE *)NULL,
		0, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize =  (uiNumChars + 1)* sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, puzQualName)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}
	
	if (RC_BAD( rc = pThisNode->getQualifiedName( ifpDb, puzQualName,
		uiBufSize, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sLocalName = pEnv->NewString( puzQualName, uiNumChars);
	
Exit:

	if (puzQualName != uzQualName)
	{
		f_free( &puzQualName);
	}	
	
	return( sLocalName);				
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getCollection(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiColNum;
	
	if (RC_BAD( rc = pThisNode->getCollection( ifpDb, &uiColNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiColNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DOMNode *		pThisNode = THIS_NODE();
	IF_Db *				ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMINT64				i64Val;
	
	if (RC_BAD( rc = pThisNode->getINT64( ifpDb, &i64Val)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)i64Val);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DOMNode__1getString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUNICODE		uzBuffer[ 128];
	FLMUNICODE *	puzBuf = uzBuffer;
	FLMUINT			uiBufSize = sizeof(uzBuffer);
	FLMUINT			uiNumChars;
	jstring			sBuf = NULL;
	
	if (RC_BAD( rc = pThisNode->getUnicodeChars( ifpDb, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}	
		
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof(FLMUNICODE);
		
		if (RC_BAD( rc = f_alloc( uiBufSize, &puzBuf)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}

	if (RC_BAD( rc = pThisNode->getUnicode( ifpDb, puzBuf, uiBufSize, 0,
											uiNumChars, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sBuf = pEnv->NewString( puzBuf, uiNumChars);	
	
Exit:

	if (puzBuf != uzBuffer)
	{
		f_free( &puzBuf);
	}
	
	return( sBuf);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getStringLen(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiVal;
	
	if (RC_BAD(rc = pThisNode->getUnicodeChars( ifpDb, &uiVal)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiVal);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DOMNode__1getDataType(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiType;
	
	if (RC_BAD( rc = pThisNode->getDataType( ifpDb, &uiType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( uiType);
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getDataLength(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiLength;
	
	if (RC_BAD( rc = pThisNode->getDataLength( ifpDb, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)uiLength);
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_DOMNode__1getBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiLength;
	jbyteArray		Data = NULL;
	void *			pvData = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	if (RC_BAD(rc = pThisNode->getDataLength( ifpDb, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}	

	Data = pEnv->NewByteArray( uiLength);
	
	if ( (pvData = pEnv->GetPrimitiveArrayCritical( Data, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = pThisNode->getBinary(ifpDb, pvData, 0, uiLength, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		if (RC_BAD( rc))
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, JNI_ABORT);
			pEnv->DeleteLocalRef( Data);
			Data = NULL;
		}
		else
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, 0);
		}
	}

	return( Data);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lValue)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;

	if (RC_BAD( rc = pThisNode->setINT64( ifpDb, (FLMINT64)lValue)))
	{
		ThrowError( rc, pEnv);
	}
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jstring			sValue,
	jboolean			bLast)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	jchar *			pszValue = NULL;
	FLMUINT			uiLength = 0;
	FLMBOOL			bMustRelease = FALSE;

	if (sValue)
	{
		pszValue = (jchar *)pEnv->GetStringCritical( sValue, NULL);
		bMustRelease = TRUE;
		uiLength = (FLMUINT)pEnv->GetStringLength( sValue);
	}
	
	if (RC_BAD( rc = pThisNode->setUnicode( ifpDb, pszValue, uiLength,
		bLast ? TRUE : FALSE)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		pEnv->ReleaseStringCritical( sValue, pszValue);
	}
}
  
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1setBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jbyteArray		Value)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMUINT			uiLength = pEnv->GetArrayLength( Value);
	void *			pvValue = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	if( (pvValue = pEnv->GetPrimitiveArrayCritical( Value, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if( RC_BAD( rc = pThisNode->setBinary(ifpDb, pvValue, uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		pEnv->ReleasePrimitiveArrayCritical( Value, pvValue, JNI_ABORT);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DOMNode__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_DOMNode *	pNode = THIS_NODE();
	
	if( pNode)
	{
		pNode->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1createAnnotation(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if( lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if( RC_BAD( rc = pThisNode->createAnnotation( ifpDb, &ifpNewNode, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)ifpNewNode));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DOMNode__1getAnnotation(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef,
	jlong				lReusedNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	IF_DOMNode *	ifpNewNode = NULL;
	
	if( lReusedNodeRef)
	{
		ifpNewNode = (IF_DOMNode *)(FLMUINT)lReusedNodeRef;
	}
	
	if( RC_BAD( rc = pThisNode->getAnnotation( ifpDb, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)((FLMUINT)ifpNewNode));	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DOMNode__1hasAnnotation(
 	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lpDbRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pThisNode = THIS_NODE();
	IF_Db *			ifpDb = (IF_Db *)(FLMUINT)lpDbRef;
	FLMBOOL			bHasAnnotation;
	
	if( RC_BAD( rc = pThisNode->hasAnnotation( ifpDb, &bHasAnnotation)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (bHasAnnotation ? true : false));
}
