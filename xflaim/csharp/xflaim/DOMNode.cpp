//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DOMNode class
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
FLMEXTC FLMEXP void FLMAPI xflaim_DOMNode_Release(
	FLMUINT64	ui64This)
{
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)ui64This);
	
	if (pNode)
	{
		pNode->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_createNode(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32NodeType,
	FLMUINT32	ui32NameId,
	FLMUINT32	ui32InsertLoc,
	FLMUINT64 *	pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->createNode( pDb, (eDomNodeType)ui32NodeType, (FLMUINT)ui32NameId,
								(eNodeInsertLoc)ui32InsertLoc, &pNode, NULL);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_createChildElement(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32ChildElementNameId,
	FLMBOOL		bFirstChild,
	FLMUINT64 *	pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->createChildElement( pDb, (FLMUINT)ui32ChildElementNameId,
								(eNodeInsertLoc)(bFirstChild ? XFLM_FIRST_CHILD : XFLM_LAST_CHILD),
								&pNode, NULL);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_deleteNode(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->deleteNode( pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_deleteChildren(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->deleteChildren( pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DOMNode_getNodeType(
	FLMUINT64	ui64This)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);

	return( (FLMUINT32)pThisNode->getNodeType());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_isDataLocalToNode(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMBOOL *	pbLocal)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->isDataLocalToNode( pDb, pbLocal));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_createAttribute(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMUINT64 *	pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->createAttribute( pDb, (FLMUINT)ui32AttrNameId, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getFirstAttribute(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getFirstAttribute( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getLastAttribute(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getLastAttribute( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttribute(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMUINT64 *	pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getAttribute( pDb, (FLMUINT)ui32AttrNameId, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_deleteAttribute(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->deleteAttribute( pDb, (FLMUINT)ui32AttrNameId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_hasAttribute(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMBOOL *	pbHasAttr)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	rc = pThisNode->hasAttribute( pDb, (FLMUINT)ui32AttrNameId, NULL);

	if (RC_OK( rc))
	{
		*pbHasAttr = TRUE;
	}
	else if (rc == NE_XFLM_DOM_NODE_NOT_FOUND)
	{
		*pbHasAttr = FALSE;
		rc = NE_XFLM_OK;
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_hasAttributes(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMBOOL *	pbHasAttrs)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->hasAttributes( pDb, pbHasAttrs));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_hasNextSibling(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMBOOL *	pbHasNextSibling)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->hasNextSibling( pDb, pbHasNextSibling));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_hasPreviousSibling(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMBOOL *	pbHasPreviousSibling)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->hasPreviousSibling( pDb, pbHasPreviousSibling));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_hasChildren(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMBOOL *	pbHasChildren)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->hasChildren( pDb, pbHasChildren));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_isNamespaceDecl(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMBOOL *	pbIsNamespaceDecl)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->isNamespaceDecl( pDb, pbIsNamespaceDecl));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getParentId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64ParentId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getParentId( pDb, pui64ParentId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNodeId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64NodeId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getNodeId( pDb, pui64NodeId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getDocumentId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64DocumentId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getDocumentId( pDb, pui64DocumentId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getPrevSibId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64PrevSibId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getPrevSibId( pDb, pui64PrevSibId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNextSibId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64NextSibId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getNextSibId( pDb, pui64NextSibId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getFirstChildId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64FirstChildId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getFirstChildId( pDb, pui64FirstChildId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getLastChildId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64LastChildId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getLastChildId( pDb, pui64LastChildId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNameId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32NameId)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNameId;

	rc = pThisNode->getNameId( pDb, &uiNameId);
	*pui32NameId = (FLMUINT32)uiNameId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setULong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64	ui64Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setUINT64( pDb, ui64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setAttributeValueULong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMUINT64	ui64Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setAttributeValueUINT64( pDb, (FLMUINT)ui32AttrNameId,
			ui64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setLong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMINT64		i64Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setINT64( pDb, i64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setAttributeValueLong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMINT64		i64Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setAttributeValueINT64( pDb, (FLMUINT)ui32AttrNameId,
			i64Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setUInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setUINT( pDb, (FLMUINT)ui32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setAttributeValueUInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMUINT32	ui32Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setAttributeValueUINT( pDb, (FLMUINT)ui32AttrNameId,
			(FLMUINT)ui32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMINT32		i32Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setINT( pDb, (FLMINT)i32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setAttributeValueInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMINT32		i32Value,
	FLMUINT32	ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setAttributeValueINT( pDb, (FLMUINT)ui32AttrNameId,
			(FLMINT)i32Value, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setString(
	FLMUINT64				ui64This,
	FLMUINT64				ui64Db,
	const FLMUNICODE *	puzValue,
	FLMBOOL					bLast,
	FLMUINT32				ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setUnicode( pDb, puzValue, 0, bLast,
							(FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setAttributeValueString(
	FLMUINT64				ui64This,
	FLMUINT64				ui64Db,
	FLMUINT32				ui32AttrNameId,
	const FLMUNICODE *	puzValue,
	FLMUINT32				ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setAttributeValueUnicode( pDb, (FLMUINT)ui32AttrNameId,
							puzValue, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setBinary(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	const void *	pvValue,
	FLMUINT32		ui32Len,
	FLMBOOL			bLast,
	FLMUINT32		ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setBinary( pDb, pvValue, (FLMUINT)ui32Len, bLast,
							(FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setAttributeValueBinary(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32AttrNameId,
	const void *	pvValue,
	FLMUINT32		ui32Len,
	FLMUINT32		ui32EncId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setAttributeValueBinary( pDb, (FLMUINT)ui32AttrNameId,
							pvValue, (FLMUINT)ui32Len, (FLMUINT)ui32EncId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getDataLength(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32 *		pui32DataLength)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiDataLength;

	rc = pThisNode->getDataLength( pDb, &uiDataLength);
	*pui32DataLength = (FLMUINT32)uiDataLength;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getDataType(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32 *		pui32DataType)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiDataType;

	rc = pThisNode->getDataType( pDb, &uiDataType);
	*pui32DataType = (FLMUINT32)uiDataType;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getULong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT64 *	pui64Value)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getUINT64( pDb, pui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueULong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMBOOL		bDefaultOk,
	FLMUINT64	ui64DefaultToUse,
	FLMUINT64 *	pui64Value)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueUINT64( pDb, (FLMUINT)ui32AttrNameId,
								pui64Value, ui64DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueUINT64( pDb, (FLMUINT)ui32AttrNameId,
								pui64Value);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getLong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMINT64 *	pi64Value)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getINT64( pDb, pi64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueLong(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMBOOL		bDefaultOk,
	FLMINT64		i64DefaultToUse,
	FLMINT64 *	pi64Value)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueINT64( pDb, (FLMUINT)ui32AttrNameId,
								pi64Value, i64DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueINT64( pDb, (FLMUINT)ui32AttrNameId,
								pi64Value);
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getUInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32Value)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiValue;

	rc = pThisNode->getUINT( pDb, &uiValue);
	*pui32Value = (FLMUINT32)uiValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueUInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMBOOL		bDefaultOk,
	FLMUINT32	ui32DefaultToUse,
	FLMUINT32 *	pui32Value)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiValue;

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueUINT( pDb, (FLMUINT)ui32AttrNameId,
								&uiValue, (FLMUINT)ui32DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueUINT( pDb, (FLMUINT)ui32AttrNameId,
								&uiValue);
	}
	*pui32Value = (FLMUINT32)uiValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMINT32 *	pi32Value)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMINT			iValue;

	rc = pThisNode->getINT( pDb, &iValue);
	*pi32Value = (FLMINT32)iValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueInt(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32AttrNameId,
	FLMBOOL		bDefaultOk,
	FLMINT32		i32DefaultToUse,
	FLMINT32 *	pi32Value)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMINT			iValue;

	if (bDefaultOk)
	{
		rc = pThisNode->getAttributeValueINT( pDb, (FLMUINT)ui32AttrNameId,
								&iValue, (FLMINT)i32DefaultToUse);
	}
	else
	{
		rc = pThisNode->getAttributeValueINT( pDb, (FLMUINT)ui32AttrNameId,
								&iValue);
	}
	*pi32Value = (FLMINT32)iValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getString(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32StartPos,
	FLMUINT32		ui32NumChars,
	FLMUNICODE **	ppuzValue)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;
	FLMUINT			uiBufSize;

	*ppuzValue = NULL;
	if (RC_BAD( rc = pThisNode->getUnicodeChars( pDb, &uiNumChars)))
	{
		goto Exit;
	}
	if ((FLMUINT)ui32StartPos >= uiNumChars)
	{
		if (RC_BAD( rc = f_alloc( sizeof( FLMUNICODE), ppuzValue)))
		{
			goto Exit;	
		}
		(*ppuzValue) [0] = 0;
		goto Exit;
	}
	uiNumChars -= (FLMUINT)ui32StartPos;
	if (ui32NumChars && (FLMUINT)ui32NumChars < uiNumChars)
	{
		uiNumChars = (FLMUINT)ui32NumChars;
	}

	uiBufSize = (uiNumChars + 1) * sizeof( FLMUNICODE);
	if (RC_BAD( rc = f_alloc( uiBufSize, ppuzValue)))
	{
		goto Exit;	
	}

	if (RC_BAD( rc = pThisNode->getUnicode( pDb, *ppuzValue, uiBufSize,
											(FLMUINT)ui32StartPos, uiNumChars, NULL)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueString(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32AttrNameId,
	FLMUNICODE **	ppuzValue)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getAttributeValueUnicode( pDb, (FLMUINT)ui32AttrNameId,
									ppuzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getStringLen(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32 *		pui32NumChars)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	rc = pThisNode->getUnicodeChars( pDb, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getBinary(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32StartPos,
	FLMUINT32		ui32NumBytes,
	void *			pvValue)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getBinary( pDb, pvValue, (FLMUINT)ui32StartPos,
											(FLMUINT)ui32NumBytes, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueDataLength(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32AttrNameId,
	FLMUINT32 *		pui32DataLen)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiDataLen;

	rc = pThisNode->getAttributeValueBinary( pDb, (FLMUINT)ui32AttrNameId,
												NULL, 0, &uiDataLen);
	*pui32DataLen = (FLMUINT32)uiDataLen;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAttributeValueBinary(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32AttrNameId,
	FLMUINT32		ui32Len,
	void *			pvValue)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getAttributeValueBinary( pDb, (FLMUINT)ui32AttrNameId, pvValue,
								(FLMUINT)ui32Len, NULL));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getDocumentNode(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getDocumentNode( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getParentNode(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getParentNode( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getFirstChild(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getFirstChild( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getLastChild(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getLastChild( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getChild(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32NodeType,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getChild( pDb, (eDomNodeType)ui32NodeType, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getChildElement(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32ElementNameId,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getChildElement( pDb, (FLMUINT)ui32ElementNameId, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getSiblingElement(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32ElementNameId,
	FLMBOOL			bNext,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getSiblingElement( pDb, (FLMUINT)ui32ElementNameId,
							bNext, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAncestorElement(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32ElementNameId,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getAncestorElement( pDb, (FLMUINT)ui32ElementNameId, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getDescendantElement(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32ElementNameId,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getDescendantElement( pDb, (FLMUINT)ui32ElementNameId, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getPreviousSibling(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getPreviousSibling( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNextSibling(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getNextSibling( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getPreviousDocument(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getPreviousDocument( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNextDocument(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getNextDocument( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getPrefixChars(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32NumChars)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	rc = pThisNode->getPrefix( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getPrefix(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzPrefix)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	return( pThisNode->getPrefix( pDb, puzPrefix, (FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE),
											&uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getPrefixId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32PrefixId)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiPrefixId;

	rc = pThisNode->getPrefixId( pDb, &uiPrefixId);
	*pui32PrefixId = (FLMUINT32)uiPrefixId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getEncDefId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32EncDefId)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiEncDefId;

	rc = pThisNode->getEncDefId( pDb, &uiEncDefId);
	*pui32EncDefId = (FLMUINT32)uiEncDefId;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setPrefix(
	FLMUINT64				ui64This,
	FLMUINT64				ui64Db,
	const FLMUNICODE *	puzPrefix)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setPrefix( pDb, puzPrefix));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setPrefixId(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32PrefixId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setPrefixId( pDb, (FLMUINT)ui32PrefixId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNamespaceURIChars(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32NumChars)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	rc = pThisNode->getNamespaceURI( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getNamespaceURI(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzNamespaceURI)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	return( pThisNode->getNamespaceURI( pDb, puzNamespaceURI,
		(FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE), &uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getLocalNameChars(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32NumChars)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	rc = pThisNode->getLocalName( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getLocalName(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzLocalName)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	return( pThisNode->getLocalName( pDb, puzLocalName,
		(FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE), &uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getQualifiedNameChars(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32NumChars)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	rc = pThisNode->getQualifiedName( pDb, (FLMUNICODE *)NULL, 0, &uiNumChars);
	*pui32NumChars = (FLMUINT32)uiNumChars;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getQualifiedName(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32NumChars,
	FLMUNICODE *	puzQualifiedName)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiNumChars;

	return( pThisNode->getQualifiedName( pDb, puzQualifiedName,
		(FLMUINT)(ui32NumChars + 1) * sizeof( FLMUNICODE), &uiNumChars));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getCollection(
	FLMUINT64	ui64This,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32Collection)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT			uiCollection;

	rc = pThisNode->getCollection( pDb, &uiCollection);
	*pui32Collection = (FLMUINT32)uiCollection;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_createAnnotation(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->createAnnotation( pDb, &pNode, NULL);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAnnotation(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pThisNode->getAnnotation( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getAnnotationId(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64AnnotationId)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getAnnotationId( pDb, pui64AnnotationId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_hasAnnotation(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMBOOL *		pbHasAnnotation)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->hasAnnotation( pDb, pbHasAnnotation));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_getMetaValue(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64 *		pui64Value)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->getMetaValue( pDb, pui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DOMNode_setMetaValue(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT64		ui64Value)
{
	IF_DOMNode *	pThisNode = (IF_DOMNode *)((FLMUINT)ui64This);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pThisNode->setMetaValue( pDb, ui64Value));
}
