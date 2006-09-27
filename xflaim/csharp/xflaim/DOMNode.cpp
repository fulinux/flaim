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
