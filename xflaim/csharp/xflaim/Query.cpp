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
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

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
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

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
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

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
	FLMUINT32		ui32TimeLimit,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

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
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pQuery->getCurrent( pDb, &pNode);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Query_resetQuery(
	FLMUINT64		ui64Query)
{
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	
	pQuery->resetQuery();
}

// IMPORTANT NOTE: This structure must be kept in sync with the
// corresponding structure in C# code.
typedef struct
{
	FLMUINT32	ui32OptType;
	FLMUINT32	ui32Cost;
	FLMUINT64	ui64NodeId;
	FLMUINT64	ui64EndNodeId;
	char			szIxName [80];
	FLMUINT32	ui32IxNum;
	FLMBOOL		bMustVerifyPath;
	FLMBOOL		bDoNodeMatch;
	FLMBOOL		bCanCompareOnKey;
	FLMUINT64	ui64KeysRead;
	FLMUINT64	ui64KeyHadDupDoc;
	FLMUINT64	ui64KeysPassed;
	FLMUINT64	ui64NodesRead;
	FLMUINT64	ui64NodesTested;
	FLMUINT64	ui64NodesPassed;
	FLMUINT64	ui64DocsRead;
	FLMUINT64	ui64DupDocsEliminated;
	FLMUINT64	ui64NodesFailedValidation;
	FLMUINT64	ui64DocsFailedValidation;
	FLMUINT64	ui64DocsPassed;
} CS_XFLM_OPT_INFO;

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getStatsAndOptInfo(
	FLMUINT64			ui64Query,
	XFLM_OPT_INFO **	ppOptInfoArray,
	FLMUINT32 *			pui32NumOptInfos)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Query *			pQuery = (IF_Query *)((FLMUINT)ui64Query);
	FLMUINT				uiNumOptInfos = 0;

	*ppOptInfoArray = NULL;
	if (RC_BAD( rc = pQuery->getStatsAndOptInfo( &uiNumOptInfos, ppOptInfoArray)))
	{
		goto Exit;
	}

Exit:

	*pui32NumOptInfos = (FLMUINT32)uiNumOptInfos;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Query_getOptInfo(
	XFLM_OPT_INFO *		pOptInfoArray,
	FLMUINT32				ui32InfoToGet,
	CS_XFLM_OPT_INFO *	pCSOptInfo)
{
	XFLM_OPT_INFO *	pOptInfo = &pOptInfoArray [ui32InfoToGet];

	pCSOptInfo->ui32OptType = (FLMUINT32)pOptInfo->eOptType;
	pCSOptInfo->ui32Cost = (FLMUINT32)pOptInfo->uiCost;
	pCSOptInfo->ui64NodeId = pOptInfo->ui64NodeId;
	pCSOptInfo->ui64EndNodeId = pOptInfo->ui64EndNodeId;
	f_memcpy( pCSOptInfo->szIxName, pOptInfo->szIxName, sizeof( pCSOptInfo->szIxName));
	pCSOptInfo->ui32IxNum = (FLMUINT32)pOptInfo->uiIxNum;
	pCSOptInfo->bMustVerifyPath = pOptInfo->bMustVerifyPath;
	pCSOptInfo->bDoNodeMatch = pOptInfo->bDoNodeMatch;
	pCSOptInfo->bCanCompareOnKey = pOptInfo->bCanCompareOnKey;
	pCSOptInfo->ui64KeysRead = pOptInfo->ui64KeysRead;
	pCSOptInfo->ui64KeyHadDupDoc = pOptInfo->ui64KeyHadDupDoc;
	pCSOptInfo->ui64KeysPassed = pOptInfo->ui64KeysPassed;
	pCSOptInfo->ui64NodesRead = pOptInfo->ui64NodesRead;
	pCSOptInfo->ui64NodesTested = pOptInfo->ui64NodesTested;
	pCSOptInfo->ui64NodesPassed = pOptInfo->ui64NodesPassed;
	pCSOptInfo->ui64DocsRead = pOptInfo->ui64DocsRead;
	pCSOptInfo->ui64DupDocsEliminated = pOptInfo->ui64DupDocsEliminated;
	pCSOptInfo->ui64NodesFailedValidation = pOptInfo->ui64NodesFailedValidation;
	pCSOptInfo->ui64DocsFailedValidation = pOptInfo->ui64DocsFailedValidation;
	pCSOptInfo->ui64DocsPassed = pOptInfo->ui64DocsPassed;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Query_setDupHandling(
	FLMUINT64	ui64Query,
	FLMBOOL		bRemoveDups)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	
	pQuery->setDupHandling( bRemoveDups);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_setIndex(
	FLMUINT64	ui64Query,
	FLMUINT32	ui32Index)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	
	return( pQuery->setIndex( (FLMUINT)ui32Index));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getIndex(
	FLMUINT64	ui64Query,
	FLMUINT64	ui64Db,
	FLMUINT32 *	pui32Index,
	FLMBOOL *	pbHaveMultiple)
{
	RCODE			rc;
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *		pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT		uiIndex;
	
	rc = pQuery->getIndex( pDb, &uiIndex, pbHaveMultiple);
	*pui32Index = (FLMUINT32)uiIndex;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_addSortKey(
	FLMUINT64	ui64Query,
	FLMUINT64	ui64SortKeyContext,
	FLMBOOL		bChildToContext,
	FLMBOOL		bElement,
	FLMUINT32	ui32NameId,
	FLMUINT32	ui32CompareFlags,
	FLMUINT32	ui32Limit,
	FLMUINT32	ui32KeyComponent,
	FLMBOOL		bSortDescending,
	FLMBOOL		bSortMissingHigh,
	FLMUINT64 *	pui64Context)
{
	RCODE			rc;
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	void *		pvContext = NULL;
	
	rc = pQuery->addSortKey( (void *)((FLMUINT)ui64SortKeyContext),
				bChildToContext, bElement, (FLMUINT)ui32NameId,
				(FLMUINT)ui32CompareFlags, (FLMUINT)ui32Limit,
				(FLMUINT)ui32KeyComponent, bSortDescending, bSortMissingHigh,
				&pvContext);
	*pui64Context = (FLMUINT64)((FLMUINT)pvContext);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_enablePositioning(
	FLMUINT64	ui64Query)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	
	return( pQuery->enablePositioning());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_positionTo(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32TimeLimit,
	FLMUINT32		ui32Position,
	FLMUINT64 *		pui64Node)
{
	RCODE				rc;
	IF_Query *		pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *			pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));

	rc = pQuery->positionTo( pDb, &pNode, (FLMUINT)ui32TimeLimit, (FLMUINT)ui32Position);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_positionToByKey(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32TimeLimit,
	FLMUINT64		ui64SearchKey,
	FLMUINT32		ui32RetrieveFlags,
	FLMUINT64 *		pui64Node)
{
	RCODE					rc;
	IF_Query *			pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *				pDb = (IF_Db *)((FLMUINT)ui64Db);
	IF_DOMNode *		pNode = (IF_DOMNode *)((FLMUINT)(*pui64Node));
	IF_DataVector *	pSearchKey = (IF_DataVector *)((FLMUINT)ui64SearchKey);

	rc = pQuery->positionTo( pDb, &pNode, (FLMUINT)ui32TimeLimit,
						pSearchKey, (FLMUINT)ui32RetrieveFlags);
	*pui64Node = (FLMUINT64)((FLMUINT)pNode);
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getPosition(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT32 *		pui32Position)
{
	RCODE			rc;
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *		pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT		uiPosition;

	rc = pQuery->getPosition( pDb, &uiPosition);
	*pui32Position = (FLMUINT32)uiPosition;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_buildResultSet(
	FLMUINT64	ui64Query,
	FLMUINT64	ui64Db,
	FLMUINT32	ui32TimeLimit)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *		pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pQuery->buildResultSet( pDb, (FLMUINT)ui32TimeLimit));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Query_stopBuildingResultSet(
	FLMUINT64	ui64Query)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	pQuery->stopBuildingResultSet();
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_Query_enableResultSetEncryption(
	FLMUINT64	ui64Query)
{
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);

	pQuery->enableResultSetEncryption();
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_Query_getCounts(
	FLMUINT64		ui64Query,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32TimeLimit,
	FLMBOOL			bPartialCountOk,
	FLMUINT32 *		pui32ReadCount,
	FLMUINT32 *		pui32PassedCount,
	FLMUINT32 *		pui32PositionableToCount,
	FLMBOOL *		pbDoneBuildingResultSet)
{
	RCODE			rc;
	IF_Query *	pQuery = (IF_Query *)((FLMUINT)ui64Query);
	IF_Db *		pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT		uiReadCount;
	FLMUINT		uiPassedCount;
	FLMUINT		uiPositionableToCount;

	rc = pQuery->getCounts( pDb, (FLMUINT)ui32TimeLimit, bPartialCountOk,
							&uiReadCount, &uiPassedCount, &uiPositionableToCount,
							pbDoneBuildingResultSet);
	*pui32ReadCount = (FLMUINT32)uiReadCount;
	*pui32PassedCount = (FLMUINT32)uiPassedCount;
	*pui32PositionableToCount = (FLMUINT32)uiPositionableToCount;
	return( rc);
}
