//------------------------------------------------------------------------------
// Desc: Native C routines to support C# Query class
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
#include "flaimsys.h"

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_createQuery(
	FLMUINT32	ui32Collection,
	FLMUINT64 *	pui64Query)
{
	RCODE			rc = NE_XFLM_OK;
	F_Query *	pQuery;

	if ((pQuery = f_new F_Query) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	pQuery->setCollection( (FLMUINT)ui32Collection);

Exit:

	*pui64Query = (FLMUINT64)((FLMUINT)pQuery);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Query_Release(
	FLMUINT64	ui64Query)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	
	if (pQuery)
	{
		pQuery->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_setLanguage(
	FLMUINT64	ui64Query,
	FLMUINT32	ui32Language)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->setLanguage( (FLMUINT)ui32Language));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_setupQueryExpr(
	FLMUINT64				ui64Query,
	FLMUINT64				ui64Db,
	const FLMUNICODE *	puzQueryExpr)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *		pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pQuery->setupQueryExpr( pDb, puzQueryExpr));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_copyCriteria(
	FLMUINT64	ui64Query,
	FLMUINT64	ui64QueryToCopy)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Query *	pQueryToCopy = (IF_Query *)((FLMUINT)ui64QueryToCopy);

	return( pQuery->copyCriteria( pQueryToCopy));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addXPathComponent(
	FLMUINT64	ui64Query,
	FLMUINT32	ui32XPathAxis,
	FLMUINT32	ui32NodeType,
	FLMUINT32	ui32NameId)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addXPathComponent( (eXPathAxisTypes)ui32XPathAxis,
							(eDomNodeType)ui32NodeType, (FLMUINT)ui32NameId));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addOperator(
	FLMUINT64	ui64Query,
	FLMUINT32	ui32Operator,
	FLMUINT32	ui32CompareFlags)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addOperator( (eQueryOperators)ui32Operator,
							(FLMUINT)ui32CompareFlags));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addStringValue(
	FLMUINT64				ui64Query,
	const FLMUNICODE *	puzValue)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addUnicodeValue( puzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addBinaryValue(
	FLMUINT64		ui64Query,
	const void *	pvValue,
	FLMINT32			i32ValueLen)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addBinaryValue( pvValue, (FLMUINT)i32ValueLen));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addULongValue(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Value)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addUINT64Value( ui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addLongValue(
	FLMUINT64		ui64Query,
	FLMINT64			i64Value)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addINT64Value( i64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addUIntValue(
	FLMUINT64		ui64Query,
	FLMUINT32		ui32Value)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addUINTValue( (FLMUINT)ui32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addIntValue(
	FLMUINT64		ui64Query,
	FLMINT32			i32Value)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addINTValue( (FLMINT)i32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addBoolean(
	FLMUINT64		ui64Query,
	FLMBOOL			bValue)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addBoolean( bValue, FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addUnknown(
	FLMUINT64		ui64Query)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	return( pQuery->addBoolean( FALSE, TRUE));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getFirst(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT64		ui64OldNode,
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)ui64OldNode);

	rc = pQuery->getFirst( pDb, &pNode, (FLMUINT)ui32TimeLimit);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getLast(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT64		ui64OldNode,
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)ui64OldNode);

	rc = pQuery->getLast( pDb, &pNode, (FLMUINT)ui32TimeLimit);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getNext(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT64		ui64OldNode,
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)ui64OldNode);

	rc = pQuery->getNext( pDb, &pNode, (FLMUINT)ui32TimeLimit);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getPrev(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT64		ui64OldNode,
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)ui64OldNode);

	rc = pQuery->getPrev( pDb, &pNode, (FLMUINT)ui32TimeLimit);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getCurrent(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT64		ui64OldNode,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)ui64OldNode);

	rc = pQuery->getCurrent( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}
