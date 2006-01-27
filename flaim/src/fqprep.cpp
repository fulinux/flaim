//-------------------------------------------------------------------------
// Desc:	Query preparation for execution
// Tabs:	3
//
//		Copyright (c) 1996-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fqprep.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

POOL_STATS	g_SubQueryOptPoolStats = {0,0};

FSTATIC void flmCurPruneNode(
	FQNODE_p 	pQNode);

FSTATIC RCODE flmCurAddSubQuery(
	CURSOR_p		pCursor,
	FQNODE_p		pQNode);

FSTATIC RCODE flmCurDivideQTree(
	CURSOR_p		pCursor);

FSTATIC RCODE flmCurCreateSQList(
	CURSOR_p		pCursor);

FSTATIC void flmCurClipNode(
	FQNODE_p 	pQNode);

FSTATIC void flmCurReplaceNode(
	FQNODE_p		pNodeToReplace,
	FQNODE_p		pReplacementNode);

FSTATIC RCODE flmCurDoDeMorgan(
	CURSOR_p		pCursor);

FSTATIC RCODE flmCurCopyQTree(
	FQNODE_p		pSrcTree,
	FQNODE_p  * ppDestTree,
	POOL *		pPool);

FSTATIC RCODE flmApplyAssociativity(
	FQNODE_p *	ppTree,
	FQNODE_p		pAndNode,
	FQNODE_p		pOrNode,
	POOL *		pPool);

FSTATIC RCODE flmCurStratify(
	CURSOR_p		pCursor,
	POOL *		pPool,
	FLMBOOL *	pbStratified,
	FQNODE_p *	ppTree);

/****************************************************************************
Desc:	Prunes an FQNODE, along with its children, from a query tree.
Ret:
****************************************************************************/
FSTATIC void flmCurPruneNode(
	FQNODE_p	pQNode
	)
{
	// If necessary, unlink the node from any parent or siblings

	if (pQNode->pParent)
	{
		if (!pQNode->pPrevSib)
		{
			pQNode->pParent->pChild = pQNode->pNextSib;
			if (pQNode->pNextSib)
			{
				pQNode->pNextSib->pPrevSib = NULL;
			}
		}
		else
		{
			pQNode->pPrevSib->pNextSib = pQNode->pNextSib;
			if (pQNode->pNextSib)
			{
				pQNode->pNextSib->pPrevSib = pQNode->pPrevSib;
			}
		}
		pQNode->pParent = pQNode->pPrevSib = pQNode->pNextSib = NULL;
	}
}

/****************************************************************************
Desc:	Allocates space for a subquery, initializes certain members, and adds
		it to the subquery list.
Ret:
****************************************************************************/
FSTATIC RCODE flmCurAddSubQuery(
	CURSOR_p	pCursor,
	FQNODE_p	pQNode )
{
	RCODE			rc = FERR_OK;
	SUBQUERY_p	pSubQuery;

	if ((pSubQuery = (SUBQUERY_p)GedPoolCalloc( &pCursor->SQPool,
											sizeof( SUBQUERY))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	GedSmartPoolInit( &pSubQuery->OptPool, &g_SubQueryOptPoolStats);

	pSubQuery->pTree = pQNode;

	if (!pCursor->pSubQueryList)
	{
		pCursor->pSubQueryList = pSubQuery;
	}
	else
	{
		SUBQUERY_p		pTmpSubQuery;
		
		for( pTmpSubQuery = pCursor->pSubQueryList;
				pTmpSubQuery->pNext;
				pTmpSubQuery = pTmpSubQuery->pNext)
			;
		pTmpSubQuery->pNext = pSubQuery;
		pSubQuery->pPrev = pTmpSubQuery;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Scans a query tree and breaks it into a set of
		conjunct subqueries.
****************************************************************************/
FSTATIC RCODE flmCurDivideQTree(
	CURSOR_p	pCursor
	)
{
	RCODE		rc = FERR_OK;
	FQNODE_p	pQNode;
	FQNODE_p	pParent;

	// Caller has already verified that pCursor->pTree is non-NULL.

	pQNode = pCursor->pTree;
	for (;;)
	{
		if (GET_QNODE_TYPE( pQNode) == FLM_OR_OP)
		{

			// If there are no children to the OR operator, they
			// have all been pruned off, so we prune off the OR
			// node and ascend back up the tree until we get
			// to a node that has a child.

			while (!pQNode->pChild)
			{
				if ((pParent = pQNode->pParent) == NULL)
				{
					goto Exit;
				}

				// Prune this OR node - we have processed both
				// of its operands.

				flmCurPruneNode( pQNode);
				pQNode = pParent;
			}
			pQNode = pQNode->pChild;
		}
		else
		{


			// When we reach a non-OR node, prune it out and make it
			// into its own sub-query.  Save the parent node so
			// we can get back to it.

			pParent = pQNode->pParent;
			flmCurPruneNode( pQNode);
			if (RC_BAD( rc = flmCurAddSubQuery( pCursor, pQNode)))
			{
				goto Exit;
			}

			// Go to parent node, if any

			if ((pQNode = pParent) == NULL)
			{
				break;
			}
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Scans a query tree and breaks it into a set of
		conjunct subqueries.
Ret:
****************************************************************************/
FSTATIC RCODE flmCurCreateSQList(
	CURSOR_p	pCursor)
{
	RCODE	rc = FERR_OK;

	if (pCursor->pSubQueryList)
	{
		flmCurFreeSQList( pCursor, TRUE);
	}

	// If there is no query criteria, still need to have
	// at least one sub-query.

	if (pCursor->pTree)
	{
		rc = flmCurDivideQTree( pCursor);
	}
	else
	{
		rc = flmCurAddSubQuery( pCursor, NULL);
	}

	if (RC_BAD( rc))
	{
		flmCurFreeSQList( pCursor, TRUE);
	}
	return( rc);
}

/****************************************************************************
Desc:	Clips an FQNODE from a query tree, grafting its children to its parent.
****************************************************************************/
FSTATIC void flmCurClipNode(
	FQNODE_p 	pQNode)
{
	FQNODE_p		pTmpQNode;

	// If necessary, unlink pQNode from its parent, children and siblings

	if (pQNode->pChild == NULL)
	{
		if (pQNode->pPrevSib)
		{
			pQNode->pPrevSib->pNextSib = pQNode->pNextSib;
		}
		if (pQNode->pNextSib)
		{
			pQNode->pNextSib->pPrevSib = pQNode->pPrevSib;
		}
		if (pQNode->pParent && pQNode->pParent->pChild == pQNode)
		{
			pQNode->pParent->pChild = pQNode->pNextSib;
		}
	}
	else
	{
		pQNode->pChild->pPrevSib = pQNode->pPrevSib;
		if (pQNode->pPrevSib)
		{
			pQNode->pPrevSib->pNextSib = pQNode->pChild;
		}
		for (pTmpQNode = pQNode->pChild;
				;
				pTmpQNode = pTmpQNode->pNextSib)
		{
			pTmpQNode->pParent = pQNode->pParent;
			if (!pTmpQNode->pNextSib)
			{
				break;
			}
		}
		if (pQNode->pParent && pQNode->pParent->pChild == pQNode)
		{
			pQNode->pParent->pChild = pQNode->pChild;
		}
		pTmpQNode->pNextSib = pQNode->pNextSib;
		if (pQNode->pNextSib)
		{
			pQNode->pNextSib->pPrevSib = pTmpQNode;
		}
	}
	pQNode->pParent =
	pQNode->pPrevSib =
	pQNode->pNextSib =
	pQNode->pChild = NULL;
}

/****************************************************************************
Desc:	Replace one node with another node in the tree.
****************************************************************************/
FSTATIC void flmCurReplaceNode(
	FQNODE_p		pNodeToReplace,
	FQNODE_p		pReplacementNode
	)
{
	FQNODE_p	pParentNode;
	FLMBOOL	bLinkAsFirst = (pNodeToReplace->pNextSib) ? TRUE : FALSE;

	pParentNode = pNodeToReplace->pParent;
	flmCurPruneNode( pReplacementNode);
	flmCurPruneNode( pNodeToReplace);
	if (pParentNode)
	{
		if (bLinkAsFirst)
		{
			flmCurLinkFirstChild( pParentNode, pReplacementNode);
		}
		else
		{
			flmCurLinkLastChild( pParentNode, pReplacementNode);
		}
	}
}

/****************************************************************************
Desc:	Applies DeMorgan's laws to get rid of NOT operators in a tree - this
		is necessary to do before we optimize the query.
****************************************************************************/
FSTATIC RCODE flmCurDoDeMorgan(
	CURSOR_p		pCursor
	)
{
	RCODE		rc = FERR_OK;
	FQNODE_p	pQNode;
	FLMBOOL	bNotted;
	QTYPES	eOp;

	if ((pQNode = pCursor->pTree) == NULL)
	{
		goto Exit;
	}

	// Traverse the tree.

	bNotted = FALSE;
	for (;;)
	{
		eOp = GET_QNODE_TYPE( pQNode);
		if (eOp == FLM_NOT_OP)
		{
			bNotted = !bNotted;

			// Go down to the child node.

			pQNode = pQNode->pChild;
		}
		else if (IS_LOG_OP( eOp))
		{
			if (bNotted)
			{
				pQNode->eOpType = (eOp == FLM_AND_OP)
										? FLM_OR_OP
										: FLM_AND_OP;
				eOp = pQNode->eOpType;
			}

			// Go down to child

			pQNode = pQNode->pChild;
		}
		else
		{
			if (IS_COMPARE_OP( eOp))
			{
				if (bNotted)
				{
					switch (eOp)
					{
						case FLM_EQ_OP:
							pQNode->eOpType = FLM_NE_OP;
							break;
						case FLM_NE_OP:
							pQNode->eOpType = FLM_EQ_OP;
							break;
						case FLM_LT_OP:
							pQNode->eOpType = FLM_GE_OP;
							break;
						case FLM_LE_OP:
							pQNode->eOpType = FLM_GT_OP;
							break;
						case FLM_GE_OP:
							pQNode->eOpType = FLM_LT_OP;
							break;
						case FLM_GT_OP:
							pQNode->eOpType = FLM_LE_OP;
							break;
						default:
							pQNode->uiStatus |= FLM_NOTTED;
							break;
					}
					pQNode->uiStatus |= FLM_FOR_EVERY;
				}
			}
			else if (eOp == FLM_BOOL_VAL)
			{
				if (bNotted)
				{
					pQNode->pQAtom->val.uiBool =
						(FLMUINT)((pQNode->pQAtom->val.uiBool == FLM_TRUE)
									 ? (FLMUINT)FLM_FALSE
									 : (FLMUINT)((pQNode->pQAtom->val.uiBool == FLM_FALSE)
													 ? (FLMUINT)FLM_TRUE
													 : (FLMUINT)FLM_UNK));
				}
			}
			else if (IS_FIELD( eOp))
			{

				// At this point, it had better be an exists test on
				// the field.

				flmAssert( !pQNode->pParent ||
					  pQNode->pParent->eOpType == FLM_AND_OP ||
					  pQNode->pParent->eOpType == FLM_OR_OP ||
					  pQNode->pParent->eOpType == FLM_NOT_OP);

				if (bNotted)
				{
					pQNode->uiStatus |= FLM_NOTTED;
				}
			}
			else if (eOp == FLM_USER_PREDICATE)
			{
				if (bNotted)
				{
					pQNode->uiStatus |= (FLM_NOTTED | FLM_FOR_EVERY);
				}
			}

			// Go back up the tree until we hit something that has
			// a sibling.

			while (!pQNode->pNextSib)
			{

				// If there are no more parents, we are done.

				if ((pQNode = pQNode->pParent) == NULL)
				{
					goto Exit;
				}

				// NOT nodes can be clipped out.  Reverse the
				// bNotted flag as we go back up through them.

				if (pQNode->eOpType == FLM_NOT_OP)
				{
					FQNODE_p	pKeepNode;

					bNotted = !bNotted;

					// If this NOT node has no parent, the root
					// of the tree needs to be set to its child.

					pKeepNode = pQNode->pChild;
					if (!pQNode->pParent)
					{
						pCursor->pTree = pKeepNode;
					}
					flmCurClipNode( pQNode);
					pQNode = pKeepNode;
				}

				// For the AND and OR operators, check the two children
				// node to see if they are a FLM_BOOL_VAL type. 
				// FLM_BOOL_VAL nodes can be weeded out of the criteria
				// as we go back up the tree.

				else if (pQNode->eOpType == FLM_OR_OP ||
							pQNode->eOpType == FLM_AND_OP)
				{
					FLMUINT	uiLeftBoolVal = 0;
					FLMUINT	uiRightBoolVal = 0;
					FQNODE_p	pLeftNode = pQNode->pChild;
					FQNODE_p	pRightNode = pLeftNode->pNextSib;
					FQNODE_p	pReplacementNode = NULL;

					if (pLeftNode->eOpType == FLM_BOOL_VAL)
					{
						uiLeftBoolVal = pLeftNode->pQAtom->val.uiBool;
					}
					if (pRightNode->eOpType == FLM_BOOL_VAL)
					{
						uiRightBoolVal = pRightNode->pQAtom->val.uiBool;
					}

					if (uiLeftBoolVal && uiRightBoolVal)
					{
						FLMUINT	uiNewBoolVal;

						if (pQNode->eOpType == FLM_AND_OP)
						{
							if (uiLeftBoolVal == FLM_FALSE ||
								 uiRightBoolVal == FLM_FALSE)
							{
								uiNewBoolVal = FLM_FALSE;
							}
							else if (uiLeftBoolVal == FLM_TRUE &&
										uiRightBoolVal == FLM_TRUE)
							{
								uiNewBoolVal = FLM_TRUE;
							}
							else
							{
								uiNewBoolVal = FLM_UNK;
							}
						}
						else // FLM_OR_OP
						{
							if (uiLeftBoolVal == FLM_TRUE ||
								 uiRightBoolVal == FLM_TRUE)
							{
								uiNewBoolVal = FLM_TRUE;
							}
							else if (uiLeftBoolVal == FLM_FALSE &&
										uiRightBoolVal == FLM_FALSE)
							{
								uiNewBoolVal = FLM_FALSE;
							}
							else
							{
								uiNewBoolVal = FLM_UNK;
							}
						}

						// Doesn't really matter which one we use to
						// replace the AND or OR node - we will use
						// the left one.

						pLeftNode->pQAtom->val.uiBool = uiNewBoolVal;
						pReplacementNode = pLeftNode;
						flmCurReplaceNode( pQNode, pReplacementNode);
						pQNode = pReplacementNode;
					}
					else if (uiLeftBoolVal || uiRightBoolVal)
					{
						if (pQNode->eOpType == FLM_OR_OP)
						{
							if (uiLeftBoolVal)
							{
								pReplacementNode = (uiLeftBoolVal == FLM_TRUE)
														 ? pLeftNode
														 : pRightNode;
							}
							else
							{
								pReplacementNode = (uiRightBoolVal == FLM_TRUE)
														 ? pRightNode
														 : pLeftNode;
							}
						}
						else	// pQNode->eOpType == FLM_AND_OP
						{
							if (uiLeftBoolVal)
							{
								pReplacementNode = (uiLeftBoolVal != FLM_TRUE)
														 ? pLeftNode
														 : pRightNode;
							}
							else
							{
								pReplacementNode = (uiRightBoolVal != FLM_TRUE)
														 ? pRightNode
														 : pLeftNode;
							}
						}
						flmCurReplaceNode( pQNode, pReplacementNode);
						pQNode = pReplacementNode;
						if (!pQNode->pParent)
						{
							pCursor->pTree = pQNode;
						}
					}
				}
			}

			// pQNode will NEVER be NULL if we get here, because we
			// will jump to Exit in those cases.

			pQNode = pQNode->pNextSib;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:	Copies a passed-in query node into a new node, using the passed-in
		memory pool.
****************************************************************************/
RCODE flmCurCopyQNode(
	FQNODE_p			pSrcNode,
	QTINFO_p			pDestQTInfo,
	FQNODE_p * 		ppDestNode,
	POOL *			pPool)
{
	RCODE				rc = FERR_OK;
	FLMUINT *		pTmpPath;
	FLMUINT *		pFldPath;
	void *			pVal = NULL;
	FLMUINT			uiPathCnt;
	QTYPES			eType;
	FLMUINT			uiLen;
	FLMUINT			uiFlags;
	FLMUINT			uiCnt;
	FQNODE_p			pDestNd;

	if( IS_OP( pSrcNode->eOpType))
	{
		eType = pSrcNode->eOpType;
		pVal = NULL;
		uiLen = 0;
		uiFlags = pSrcNode->uiStatus;
	}
	else
	{
		uiLen = pSrcNode->pQAtom->uiBufLen;
		uiFlags = pSrcNode->pQAtom->uiFlags;
		eType = pSrcNode->pQAtom->eType;
		
		switch( eType)
		{
			case FLM_BOOL_VAL:
			{
				pVal = (void *)&pSrcNode->pQAtom->val.uiBool;
				break;
			}
			
			case FLM_INT32_VAL:
			{
				pVal = (void *)&pSrcNode->pQAtom->val.iVal;
				break;
			}
			
			case FLM_REC_PTR_VAL:
			case FLM_UINT32_VAL:
			{
				pVal = (void *)&pSrcNode->pQAtom->val.uiVal;
				break;
			}
			
			case FLM_FLD_PATH:
			{
				// Count the number of fields in the field path.

				uiPathCnt = 0;
				pFldPath = pSrcNode->pQAtom->val.QueryFld.puiFldPath;
				
				while (*pFldPath)
				{
					uiPathCnt++;
					if( uiPathCnt > GED_MAXLVLNUM + 1)
					{
						rc = RC_SET( FERR_CURSOR_SYNTAX);
						goto Exit;
					}
					pFldPath++;
				}

				// Allocate a temporary array to hold the field path.  We need
				// to do this so we can reverse the field path.  The field path
				// needs to be reversed because it is reversed back in the call
				// to flmCurMakeQNode below.
				
				if ((pTmpPath = (FLMUINT *)GedPoolCalloc( pPool,
						((uiPathCnt + 1) *	sizeof( FLMUINT)))) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					break;
				}
				
				pFldPath = pSrcNode->pQAtom->val.QueryFld.puiFldPath;
				
				for (uiCnt = 0; uiCnt < uiPathCnt; uiCnt++)
				{
					pTmpPath[ uiPathCnt - uiCnt - 1] = pFldPath[ uiCnt];
				}
				pVal = (void *)pTmpPath;
				break;
			}
			
			case FLM_TEXT_VAL:
			{
				pVal = (void *)pSrcNode->pQAtom->val.pucBuf;
				break;
			}
			
			case FLM_BINARY_VAL:
			{
				pVal = (void *)pSrcNode->pQAtom->val.pucBuf;
				break;
			}
			
			case FLM_USER_PREDICATE:
			{
				pVal = NULL;
				uiLen = 0;
				break;
			}
			
			default:
			{
				rc = RC_SET( FERR_CURSOR_SYNTAX);
				break;
			}
		}
	}

	if( RC_BAD( rc = flmCurMakeQNode( pPool, eType, pVal, uiLen, 
		uiFlags, ppDestNode)))
	{
		goto Exit;
	}
	
	pDestNd = *ppDestNode;
	pDestNd->uiStatus = pSrcNode->uiStatus;

	// Need to save some additional things for nodes containing
	// callback data.

	if( eType == FLM_USER_PREDICATE)
	{
		if( pDestQTInfo)
		{
			if( (pDestNd->pQAtom->val.pPredicate =
						pSrcNode->pQAtom->val.pPredicate->copy()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
			
			if( RC_BAD( rc = flmCurAddRefPredicate( pDestQTInfo,
										pDestNd->pQAtom->val.pPredicate)))
			{
				goto Exit;
			}

			// Need to release to decrement counter - because call
			// to flmCurAddRefPredicate will do an addRef() and the
			// call to copy() returns one with a refcount of one.

			pDestNd->pQAtom->val.pPredicate->Release();
		}
		else
		{
			pDestNd->pQAtom->val.pPredicate = pSrcNode->pQAtom->val.pPredicate;

			// Don't do an addRef on pPredicate because there is really no place
			// to call a corresponding release.
		}
	}
	else if( (eType == FLM_FLD_PATH) &&
				(pSrcNode->pQAtom->val.QueryFld.fnGetField))
	{
		pDestNd->pQAtom->val.QueryFld.fnGetField =
			pSrcNode->pQAtom->val.QueryFld.fnGetField;
		pDestNd->pQAtom->val.QueryFld.bValidateOnly =
			pSrcNode->pQAtom->val.QueryFld.bValidateOnly;
			
		if ((pSrcNode->pQAtom->val.QueryFld.pvUserData) &&
			 (pSrcNode->pQAtom->val.QueryFld.uiUserDataLen))
		{
			if ((pDestNd->pQAtom->val.QueryFld.pvUserData =
				GedPoolAlloc( pPool,
					pSrcNode->pQAtom->val.QueryFld.uiUserDataLen)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
			
			f_memcpy( pDestNd->pQAtom->val.QueryFld.pvUserData,
						 pSrcNode->pQAtom->val.QueryFld.pvUserData,
						 pSrcNode->pQAtom->val.QueryFld.uiUserDataLen);
			pDestNd->pQAtom->val.QueryFld.uiUserDataLen = 
					pSrcNode->pQAtom->val.QueryFld.uiUserDataLen;
		}
		else
		{
			pDestNd->pQAtom->val.QueryFld.pvUserData = NULL;
			pDestNd->pQAtom->val.QueryFld.uiUserDataLen = 0;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Copies a passed-in query tree into a new tree, using the passed-in
		memory pool.
****************************************************************************/
FSTATIC RCODE flmCurCopyQTree(
	FQNODE_p			pSrcTree,
	FQNODE_p  * 	ppDestTree,
	POOL *			pPool)
{
	RCODE				rc = FERR_OK;
	FQNODE_p			pQNode;
	FQNODE_p			pDestNode;
	FQNODE_p			pParentNode;

	// Don't try to copy a NULL tree.
	
	if ((pQNode = pSrcTree) == NULL)
	{
		*ppDestTree = NULL;
		goto Exit;
	}

	pQNode = pSrcTree;
	pParentNode = NULL;
	for (;;)
	{
		if (RC_BAD( rc = flmCurCopyQNode( pQNode, NULL, &pDestNode, pPool)))
		{
			goto Exit;
		}

		// Link into destination tree.

		if (!pParentNode)
		{
			*ppDestTree = pDestNode;
		}
		else
		{
			flmCurLinkLastChild( pParentNode, pDestNode);
		}

		// Traverse to child.

		if (pQNode->pChild)
		{
			pParentNode = pDestNode;
			pQNode = pQNode->pChild;
		}
		else
		{

			// Traverse back up the tree until we find a node that
			// has a sibling.

			while (!pQNode->pNextSib)
			{
				if ((pQNode = pQNode->pParent) == NULL)
				{

					// We are done when we arrive back at the root of
					// the tree.

					goto Exit;
				}
				pParentNode = pParentNode->pParent;
			}

			// If we get to this point, pNextSib is guaranteed to
			// NOT be NULL.

			pQNode = pQNode->pNextSib;
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
class F_NodeList
{
public:
	F_NodeList()
	{
		m_pNodeList = &m_nodeList [0];
		m_uiListSize = sizeof( m_nodeList) / sizeof( FQNODE_p);
		m_uiNumNodes = 0;
	}

	~F_NodeList()
	{
		if (m_pNodeList != &m_nodeList [0])
		{
			f_free( &m_pNodeList);
		}
	}

	RCODE gatherNodeList(
		FQNODE_p	pStartNode,
		QTYPES	eOp,
		FQNODE_p	pExcludeNode);

	FINLINE FLMUINT getNumNodes( void)
	{
		return( m_uiNumNodes);
	}

	FINLINE FQNODE_p getNode(
		FLMUINT	uiNodeIndex)
	{
		return( m_pNodeList [uiNodeIndex]);
	}

private:
	FQNODE_p		m_nodeList [50];
	FQNODE_p *	m_pNodeList;
	FLMUINT		m_uiListSize;
	FLMUINT		m_uiNumNodes;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_NodeList::gatherNodeList(
	FQNODE_p	pStartNode,
	QTYPES	eOp,
	FQNODE_p	pExcludeNode)
{
	RCODE		rc = FERR_OK;
	FQNODE_p	pTmpNode;
	QTYPES	eTmpOp;

	pTmpNode = pStartNode->pChild;
	for (;;)
	{
		eTmpOp = GET_QNODE_TYPE( pTmpNode);
		if (eTmpOp == eOp)
		{
			pTmpNode = pTmpNode->pChild;
		}
		else
		{
			// Put the node into a list - only if it is not the exclude node that
			// was passed into the function.

			if (pTmpNode != pExcludeNode)
			{

				// See if we need to resize the list.  Increase list size by 200

				if (m_uiNumNodes == m_uiListSize)
				{
					FLMUINT	uiNewListSize = m_uiListSize + 200;

					if (m_pNodeList == &m_nodeList [0])
					{
						if (RC_BAD( rc = f_alloc( uiNewListSize * sizeof( FQNODE_p), &m_pNodeList)))
						{
							goto Exit;
						}
						f_memcpy( m_pNodeList, &m_nodeList [0], sizeof( FQNODE_p) * m_uiNumNodes);
					}
					else
					{
						if (RC_BAD( rc = f_realloc( uiNewListSize * sizeof( FQNODE_p), &m_pNodeList)))
						{
							goto Exit;
						}
					}
					m_uiListSize = uiNewListSize;
				}
				m_pNodeList [m_uiNumNodes] = pTmpNode;
				m_uiNumNodes++;
			}

			// Go to the node's sibling.  If already at a sibling node, travel back up
			// the tree until we hit the node we started from.

			for (;;)
			{
				if (pTmpNode->pNextSib)
				{
					pTmpNode = pTmpNode->pNextSib;
					break;
				}
				if ((pTmpNode = pTmpNode->pParent) == pStartNode)
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE flmApplyAssociativity(
	FQNODE_p *	ppTree,
	FQNODE_p		pAndNode,
	FQNODE_p		pOrNode,
	POOL *		pPool
	)
{
	RCODE			rc = FERR_OK;
	QTYPES		eOp;
	FQNODE_p		pAndParent;
	FQNODE_p		pTmpNode;
	FQNODE_p		pCopyNode;
	FQNODE_p		pNewAndNode;
	FQNODE_p		pNewOrNode;
	F_NodeList	andNodeList;
	F_NodeList	orNodeList;
	FLMUINT		uiNumAndNodes;
	FLMUINT		uiNumOrNodes;
	FLMUINT		uiOrLoop;
	FLMUINT		uiAndLoop;

	// Find highest AND node above the passed in AND node.  That will be the node we are going to
	// replace in the tree.

	while (pAndNode->pParent)
	{
		eOp = GET_QNODE_TYPE( pAndNode->pParent);
		if (eOp == FLM_AND_OP)
		{
			pAndNode = pAndNode->pParent;
		}
		else
		{
			break;
		}
	}

	// Make a list of all of the nodes below the highest AND node that are not AND nodes.
	// Exclude the OR node that was passed into this function.

	if (RC_BAD( rc = andNodeList.gatherNodeList( pAndNode, FLM_AND_OP, pOrNode)))
	{
		goto Exit;
	}

	// Make a list of all of the nodes below the OR node that are not OR nodes.

	if (RC_BAD( rc = orNodeList.gatherNodeList( pOrNode, FLM_OR_OP, pOrNode)))
	{
		goto Exit;
	}

	uiNumAndNodes = andNodeList.getNumNodes();
	uiNumOrNodes = orNodeList.getNumNodes();

	// Has to be at least two OR nodes and one AND node.

	flmAssert( uiNumOrNodes >= 2 && uiNumAndNodes >= 1);

	// AND all of the nodes in the OR list with each of the nodes in the AND list.
	// Each group of ANDed nodes is then ORed together.

	pNewOrNode = NULL;
	for (uiOrLoop = 0; uiOrLoop < uiNumOrNodes; uiOrLoop++)
	{
		pTmpNode = orNodeList.getNode( uiOrLoop);
		if (RC_BAD( rc = flmCurCopyQTree( pTmpNode, &pNewAndNode, pPool)))
		{
			goto Exit;
		}
		for (uiAndLoop = 0; uiAndLoop < uiNumAndNodes; uiAndLoop++)
		{
			pTmpNode = andNodeList.getNode( uiAndLoop);
			if (RC_BAD( rc = flmCurCopyQTree( pTmpNode, &pCopyNode, pPool)))
			{
				goto Exit;
			}

			// AND copied node with last AND node.  Last AND node becomes the newly
			// created AND operator.

			if (RC_BAD( rc = flmCurGraftNode( pPool, pCopyNode, FLM_AND_OP, &pNewAndNode)))
			{
				goto Exit;
			}
		}

		// OR the AND list with the last OR node, if any.  Last OR node becomes the newly
		// created OR operator.

		if (pNewOrNode)
		{
			if (RC_BAD( rc = flmCurGraftNode( pPool, pNewAndNode, FLM_OR_OP, &pNewOrNode)))
			{
				goto Exit;
			}
		}
		else
		{
			pNewOrNode = pNewAndNode;
		}
	}

	// Prune the AND operator from the tree and replace it with the newly created
	// OR operator.

	if ((pAndParent = pAndNode->pParent) == NULL)
	{
		*ppTree = pNewOrNode;
	}
	else
	{

		// Prune AND operator

		flmCurPruneNode( pAndNode);

		// Graft the OR operator in as child to pAndParent.

		pAndParent->pChild->pNextSib = pNewOrNode;
		pNewOrNode->pPrevSib = pAndParent->pChild;
		pNewOrNode->pParent = pAndParent;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Applies associativity to logical operators in a query tree to render
		it in a format that consists of OR operators at the top, followed by
		a layer of AND operators.
Ret:
****************************************************************************/
FSTATIC RCODE flmCurStratify(
	CURSOR_p		pCursor,
	POOL *		pPool,
	FLMBOOL *	pbStratified,
	FQNODE_p *	ppTree)
{
	RCODE			rc = FERR_OK;
	QTYPES		eOp;
	QTYPES		eLeftOp;
	QTYPES		eRightOp;
	FQNODE_p		pTree = *ppTree;
	FQNODE_p		pCurrNode;
	FLMBOOL		bStratified = TRUE;
	void *		pvMark = GedPoolMark( pPool);
	FLMUINT		uiCount = 0;
	FLMUINT		uiStartTime = FLM_GET_TIMER();
	FLMUINT		uiCheckTime = 0;
	FLMUINT		uiTimeOut;

	// If either uiMaxStratifyIterations or uiMaxStratifyTime is zero, we will
	// not limit the depth that we stratify to.

	if (gv_FlmSysData.uiMaxStratifyIterations && gv_FlmSysData.uiMaxStratifyTime)
	{
		FLM_SECS_TO_TIMER_UNITS( gv_FlmSysData.uiMaxStratifyTime, uiTimeOut);
	}
	else
	{
		uiTimeOut = 0;
	}

	if (!pTree)
	{
		goto Exit;
	}

	pCurrNode = pTree;
	while (pCurrNode)
	{
		eOp = GET_QNODE_TYPE( pCurrNode);
		if (eOp == FLM_AND_OP)
		{
			eLeftOp = GET_QNODE_TYPE( pCurrNode->pChild);
			eRightOp = GET_QNODE_TYPE( pCurrNode->pChild->pNextSib);

			// If there are OR operators under an AND, apply associativity to
			// move them above.

			if (eLeftOp == FLM_OR_OP || eRightOp == FLM_OR_OP)
			{
				if( ++uiCount == gv_FlmSysData.uiMaxStratifyIterations)
				{
					uiCheckTime = FLM_GET_TIMER();
					if( uiTimeOut && FLM_ELAPSED_TIME( uiCheckTime, uiStartTime) > uiTimeOut)
					{
						GedPoolReset( pPool, pvMark);
						rc = flmCurCopyQTree( pCursor->QTInfo.pSaveQuery,
									&pTree, pPool);

						bStratified = FALSE;
						goto Exit;
					}
					uiCount = 0;
				}
				if (eLeftOp == FLM_OR_OP)
				{
					if (RC_BAD( rc = flmApplyAssociativity( &pTree, pCurrNode, pCurrNode->pChild, pPool)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = flmApplyAssociativity( &pTree, pCurrNode, pCurrNode->pChild->pNextSib, pPool)))
					{
						goto Exit;
					}
				}

				// After doing associativity, need to start over at the top
				// of the tree.

				pCurrNode = pTree;
				continue;
			}
		}

		// Find the next node to process

		if (pCurrNode->pChild)
		{
			pCurrNode = pCurrNode->pChild;
		}
		else
		{

			// Travel back up the tree until we find a node
			// that has a sibling.

			for (;;)
			{
				if (pCurrNode->pNextSib)
				{
					pCurrNode = pCurrNode->pNextSib;
					break;
				}
				if ((pCurrNode = pCurrNode->pParent) == NULL)
				{
					break;
				}
			}
		}
	}

Exit:

	if (pbStratified)
	{
		*pbStratified = bStratified;
	}
	
	*ppTree = pTree;
	return( rc);
}

/****************************************************************************
Desc: Prepares a query for subsequent use by partitioning it and optimizing
		its subqueries.
****************************************************************************/
RCODE flmCurPrep(
	CURSOR_p	pCursor)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bStratified;

	// Make sure we are in a good state.  If pCursor->rc is bad,
	// we should not attempt to optimize, because we may crash.

	if (RC_BAD( pCursor->rc))
	{
		rc = pCursor->rc;
		goto Exit;
	}

	// Finish the query and partition it.
	// Return error if expecting an operand and part of a query was given.

	if (pCursor->QTInfo.uiNestLvl ||
		 ((pCursor->QTInfo.uiExpecting & FLM_Q_OPERAND) && pCursor->QTInfo.pTopNode))
	{
		rc = RC_SET( FERR_CURSOR_SYNTAX);
		goto Exit;
	}
	if (pCursor->QTInfo.pTopNode == NULL)
	{
		pCursor->QTInfo.pTopNode = pCursor->QTInfo.pCurAtomNode;
		pCursor->QTInfo.pCurAtomNode = NULL;
	}

	// Save a copy of the tree before it gets all cut up.
	// Save only after flattening it.  This is so we can clone the
	// query if FlmCursorClone is called.

	if (RC_BAD( rc = flmCurCopyQTree( pCursor->QTInfo.pTopNode,
								&pCursor->QTInfo.pSaveQuery,
								&pCursor->QueryPool)))
	{
		goto Exit;
	}

	pCursor->pTree = pCursor->QTInfo.pTopNode;

	// Apply DeMorgan's law to push NOT operators down below the other
	// logical operators.

	if (RC_BAD( rc = flmCurDoDeMorgan( pCursor)))
	{
		goto Exit;
	}

	// See if our root node is a FLM_BOOL_VAL.

	if (pCursor->pTree && pCursor->pTree->eOpType == FLM_BOOL_VAL)
	{
		if (pCursor->pTree->pQAtom->val.uiBool == FLM_TRUE)
		{
			pCursor->pTree = NULL;
		}
		else
		{
			pCursor->bEmpty = TRUE;
			goto Exit;
		}
	}

	// Use associativity to render the query tree in a form that
	// has all the OR operators at the top, followed by AND operators.

	if (RC_BAD( rc = flmCurStratify( pCursor, &pCursor->QueryPool,
								&bStratified, &pCursor->pTree)))

	{
		goto Exit;
	}

	// Based on the new query tree, break the query into a set of
	// subqueries, each of which has a tree that represents a conjunct
	// query.

	if (RC_BAD( rc = flmCurCreateSQList( pCursor)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmCurOptimize( pCursor, bStratified)))
	{
		if (rc == FERR_EMPTY_QUERY)
		{
			rc = FERR_OK;
			pCursor->bEmpty = TRUE;
		}
		goto Exit;
	}

Exit:
	if (RC_BAD( rc))
	{
		pCursor->rc = rc;
	}
	else
	{
		pCursor->bOptimized = TRUE;
	}
	return( rc);
}
