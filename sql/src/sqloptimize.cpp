//-------------------------------------------------------------------------
// Desc:	Optimize an SQL query
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
//-------------------------------------------------------------------------

#include "flaimsys.h"

// Local function prototypes

FSTATIC RCODE sqlGetRowIdValue(
	SQL_VALUE *	pValue);
	
FSTATIC RCODE setupPredicate(
	SQL_PRED *					pPred,
	SQL_TABLE *					pTable,
	FLMUINT						uiColumnNum,
	eSQLQueryOperators		eOperator,
	FLMUINT						uiCompareRules,
	FLMBOOL						bNotted,
	SQL_VALUE *					pValue);
	
FSTATIC RCODE sqlCompareValues(
	SQL_VALUE *			pValue1,
	FLMBOOL				bInclusive1,
	FLMBOOL				bNullIsLow1,
	SQL_VALUE *			pValue2,
	FLMBOOL				bInclusive2,
	FLMBOOL				bNullIsLow2,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMINT *				piCmp);
	
FSTATIC SQL_NODE * sqlEvalLogicalOperands(
	SQL_NODE *		pSQLNode);
	
FSTATIC SQL_NODE * sqlClipNotNode(
	SQL_NODE *	pNotNode,
	SQL_NODE **	ppExpr);
	
FSTATIC RCODE createDNFNode(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pParentDNFNode,
	SQL_DNF_NODE **	ppDNFNode,
	SQL_NODE *			pNode);
	
FSTATIC RCODE copyAndLinkSubTree(
	F_Pool *			pPool,
	SQL_DNF_NODE *	pSrcSubTree,
	SQL_DNF_NODE *	pParentNode);
	
FSTATIC RCODE distributeAndOverOr(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pOldOrNode,
	SQL_DNF_NODE **	ppDNFTree);
	
#if 0
FSTATIC FLMBOOL predIsForTable(
	SQL_NODE *	pPredRootNode,
	SQL_TABLE *	pTable);
	
FSTATIC void rankIndexes(
	SQL_TABLE *	pTable);
#endif
	
//-------------------------------------------------------------------------
// Desc:	Get the row ID constant from an SQL_VALUE node.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlGetRowIdValue(
	SQL_VALUE *	pValue)
{
	RCODE	rc = NE_SFLM_OK;

	switch (pValue->eValType)
	{
		case SQL_UINT_VAL:
			pValue->eValType = SQL_UINT64_VAL;
			pValue->val.ui64Val = (FLMUINT64)pValue->val.uiVal;
			break;
		case SQL_MISSING_VAL:
		case SQL_UINT64_VAL:
			break;
		case SQL_INT_VAL:
			pValue->eValType = SQL_UINT64_VAL;
			pValue->val.ui64Val = (FLMUINT64)((FLMINT64)(pValue->val.iVal));
			break;
		case SQL_INT64_VAL:
			pValue->eValType = SQL_UINT64_VAL;
			pValue->val.ui64Val = (FLMUINT64)(pValue->val.i64Val);
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_Q_INVALID_ROW_ID_VALUE);
			goto Exit;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Setup a predicate using the passed in parameters.
//-------------------------------------------------------------------------
FSTATIC RCODE setupPredicate(
	SQL_PRED *					pPred,
	SQL_TABLE *					pTable,
	FLMUINT						uiColumnNum,
	eSQLQueryOperators		eOperator,
	FLMUINT						uiCompareRules,
	FLMBOOL						bNotted,
	SQL_VALUE *					pValue)
{
	RCODE		rc = NE_SFLM_OK;

	pPred->pTable = pTable;
	pPred->uiColumnNum = uiColumnNum;
	if (!pValue || pValue->eValType != SQL_UTF8_VAL)
	{

		// Comparison rules don't matter for anything that is
		// not text, so we normalize them to zero, so the test
		// below to see if the comparison rule is the same as
		// the comparison rule of the operator will work.

		pPred->uiCompareRules = 0;
	}
	else
	{
		pPred->uiCompareRules = uiCompareRules;
	}
	pPred->bNotted = bNotted;
	switch (eOperator)
	{
		case SQL_EXISTS_OP:
		case SQL_NE_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pValue;
			break;
		case SQL_APPROX_EQ_OP:
			pPred->eOperator = eOperator;
			pPred->pFromValue = pValue;
			pPred->bInclFrom = TRUE;
			pPred->bInclUntil = TRUE;
			break;
		case SQL_EQ_OP:
			if ((pValue->uiFlags & SQL_VAL_IS_CONSTANT) &&
				 (pValue->uiFlags & SQL_VAL_HAS_WILDCARDS))
			{
				pPred->eOperator = SQL_MATCH_OP;
				pPred->pFromValue = pValue;
			}
			else
			{
				pPred->eOperator = SQL_RANGE_OP;
				pPred->pFromValue = pValue;
				pPred->pUntilValue = pValue;
				pPred->bInclFrom = TRUE;
				pPred->bInclUntil = TRUE;
			}
			break;
		case SQL_LE_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pValue;
			pPred->bInclUntil = TRUE;
			break;
		case SQL_LT_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = NULL;
			pPred->pUntilValue = pValue;
			pPred->bInclUntil = FALSE;
			break;
		case SQL_GE_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = pValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = TRUE;
			break;
		case SQL_GT_OP:
			pPred->eOperator = SQL_RANGE_OP;
			pPred->pFromValue = pValue;
			pPred->pUntilValue = NULL;
			pPred->bInclFrom = FALSE;
			break;
		default:
			rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
			goto Exit;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Compare two values
//-------------------------------------------------------------------------
FSTATIC RCODE sqlCompareValues(
	SQL_VALUE *			pValue1,
	FLMBOOL				bInclusive1,
	FLMBOOL				bNullIsLow1,
	SQL_VALUE *			pValue2,
	FLMBOOL				bInclusive2,
	FLMBOOL				bNullIsLow2,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMINT *				piCmp)
{
	RCODE	rc = NE_SFLM_OK;

	// We have already called sqlCanCompare, so no need to do it here

	if (!pValue1)
	{
		if (!pValue2)
		{
			if (bNullIsLow2)
			{
				*piCmp = (FLMINT)(bNullIsLow1 ? 0 : 1);
			}
			else
			{
				*piCmp = (FLMINT)(bNullIsLow1 ? -1 : 0);
			}
		}
		else
		{
			*piCmp = (FLMINT)(bNullIsLow1 ? -1 : 1);
		}
		goto Exit;
	}
	else if (!pValue2)
	{
		*piCmp = (FLMINT)(bNullIsLow2 ? 1 : -1);
		goto Exit;
	}

	if (RC_BAD( rc = sqlCompare( pValue1, pValue2, 
		uiCompareRules, uiLanguage, piCmp)))
	{
		goto Exit;
	}

	// If everything else is equal, the last distinguisher
	// is the inclusive flags and which side of the
	// value we are on if we are exclusive which is indicated
	// by the bNullIsLow flags

	if (*piCmp == 0)
	{
		if (bInclusive1 != bInclusive2)
		{
			if (bNullIsLow1)
			{
				if (bNullIsLow2)
				{
					//			*--> v1
					//			o--> v2		v1 < v2

					//			o--> v1
					//			*--> v2		v1 > v2

					*piCmp = bInclusive1 ? -1 : 1;
				}
				else
				{
					//			*--> v1
					// v2 <--o				v1 > v2

					//			o--> v1
					// v2	<--*				v1 > v2

					*piCmp = 1;
				}
			}
			else
			{
				if (bNullIsLow2)
				{
					// v1 <--*
					//			o--> v2		v1 < v2

					// v1 <--o
					//			*--> v2		v1 < v2

					*piCmp = -1;
				}
				else
				{
					// v1	<--*
					//	v2	<--o				v1 > v2

					// v1	<--o
					// v2	<--*				v1 < v2

					*piCmp = bInclusive1 ? 1 : -1;
				}
			}
		}
		else if (!bInclusive1)
		{

			// bInclusive2 is also FALSE

			if (bNullIsLow1)
			{
				if (!bNullIsLow2)
				{
					//			o--> v1
					// v2	<--o				v1 > v2
					*piCmp = 1;
				}
//				else
//				{
					// 		o--> v1
					// 		o--> v2		v1 == v2
					// *piCmp = 0;
//				}
			}
			else
			{
				if (bNullIsLow2)
				{

					// v1	<--o
					//			o--> v2		v1 < v2

					*piCmp = -1;
				}
//				else
//				{
					// v1	<--o
					// v2	<--o				v1 == v2
					// *piCmp = 0;
//				}
			}
		}
//		else
//		{
			// bInclusive1 == TRUE && bInclusive2 == TRUE
			// else case covers the cases where
			// both are inclusive, in which case it
			// doesn't matter which is low and which
			// is high

					// v1	<--*
					//			*--> v2		v1 == v2

					// v1	<--*
					// v2	<--*				v1 == v2

					//			*--> v1
					// v2	<--*				v1 == v2

					//			*--> v1
					//			*--> v2		v1 == v2

//		}
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Intersect a predicate into an existing predicate.
//-------------------------------------------------------------------------
RCODE SQLQuery::intersectPredicates(
	SQL_PRED *				pPred,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL					bNotted,
	SQL_VALUE *				pValue,
	FLMBOOL *				pbAlwaysFalse,
	FLMBOOL *				pbIntersected)
{
	RCODE		rc = NE_SFLM_OK;
	FLMINT	iCmp;
	FLMBOOL	bDoMatch;

	*pbIntersected = FALSE;
	if (!pValue || pValue->eValType != SQL_UTF8_VAL)
	{
		bDoMatch = FALSE;

		// Comparison rules don't matter for anything that is
		// not text, so we normalize them to zero, so the test
		// below to see if the comparison rule is the same as
		// the comparison rule of the operator will work.

		uiCompareRules = 0;
	}
	else
	{
		bDoMatch = (eOperator == SQL_EQ_OP &&
						(pValue->uiFlags & SQL_VAL_IS_CONSTANT) &&
						(pValue->uiFlags & SQL_VAL_HAS_WILDCARDS))
						? TRUE
						: FALSE;
	}

	if (eOperator == SQL_EXISTS_OP)
	{
		*pbIntersected = TRUE;

		// An exists operator will either merge with an existing predicate or
		// cancel the whole thing out as an empty result.

		// If this predicate is not-exists, another predicate ANDed with this
		// one can never return a result that will match, unless that predicate
		// is also a not-exists, in which case, we simply combine this one
		// with that one.

		if (bNotted)
		{
			if (pPred->eOperator != SQL_EXISTS_OP || !pPred->bNotted)
			{
				*pbAlwaysFalse = TRUE;
			}
		}
	}
	else if (pPred->eOperator == SQL_EXISTS_OP)
	{

		*pbIntersected = TRUE;
		
		// If the first predicate is an exists operator
		// it will be the only one, because otherwise
		// it will have been merged with another operator
		// in the code just above.

		flmAssert( !pPred->pNext);

		// If the predicate is notted, another predicate
		// ANDed with this one can never return a result.

		if (pPred->bNotted)
		{
			*pbAlwaysFalse = TRUE;
		}
		else
		{

			// Change the predicate to the current
			// operator.
			
			if (RC_BAD( rc = setupPredicate( pPred, pPred->pTable,
								pPred->uiColumnNum, eOperator, uiCompareRules,
								bNotted, pValue)))
			{
				goto Exit;
			}
		}
	}
	
	// See if the operator intersects a range operator

	else if (pPred->eOperator == SQL_RANGE_OP &&
				pPred->uiCompareRules == uiCompareRules &&
				!bDoMatch &&
				(eOperator == SQL_EQ_OP ||
				 eOperator == SQL_LE_OP ||
				 eOperator == SQL_LT_OP ||
				 eOperator == SQL_GE_OP ||
				 eOperator == SQL_GT_OP))
	{
		SQL_VALUE *	pFromValue;
		SQL_VALUE *	pUntilValue;
		FLMBOOL		bInclFrom;
		FLMBOOL		bInclUntil;
		
		*pbIntersected = TRUE;

		pFromValue = (eOperator == SQL_EQ_OP ||
						  eOperator == SQL_GE_OP ||
						  eOperator == SQL_GT_OP)
						  ? pValue
						  : NULL;
		pUntilValue = (eOperator == SQL_EQ_OP ||
							eOperator == SQL_LE_OP ||
							eOperator == SQL_LT_OP)
							? pValue
							: NULL;
		bInclFrom = (FLMBOOL)(eOperator == SQL_EQ_OP ||
									 eOperator == SQL_GE_OP
									 ? TRUE
									 : FALSE);
		bInclUntil = (FLMBOOL)(eOperator == SQL_EQ_OP ||
									  eOperator == SQL_LE_OP
									  ? TRUE
									  : FALSE);

		// If the value type is not compatible with the predicate's
		// value type, we cannot do the comparison, and there is
		// no intersection.

		if (!sqlCanCompare( pValue, pPred->pFromValue) ||
			 !sqlCanCompare( pValue, pPred->pUntilValue))
		{
			*pbAlwaysFalse = TRUE;
		}
		else if (RC_BAD( rc = sqlCompareValues( pFromValue,
							bInclFrom, TRUE,
							pPred->pFromValue, pPred->bInclFrom, TRUE,
							uiCompareRules, m_uiLanguage, &iCmp)))
		{
			goto Exit;
		}
		else if (iCmp > 0)
		{

			// From value is greater than predicate's from value.
			// If the from value is also greater than the predicate's
			// until value, we have no intersection.

			if (RC_BAD( rc = sqlCompareValues( pFromValue,
						bInclFrom, TRUE,
						pPred->pUntilValue, pPred->bInclUntil, FALSE,
						uiCompareRules, m_uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			if (iCmp > 0)
			{
				*pbAlwaysFalse = TRUE;
			}
			else
			{
				pPred->pFromValue = pFromValue;
				pPred->bInclFrom = bInclFrom;
			}
		}
		else if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pUntilValue, pPred->bInclUntil, FALSE,
							uiCompareRules, m_uiLanguage, &iCmp)))
		{
			goto Exit;
		}
		else if (iCmp < 0)
		{

			// Until value is less than predicate's until value.  If the
			// until value is also less than predicate's from value, we
			// have no intersection.

			if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pFromValue, pPred->bInclFrom, TRUE,
							uiCompareRules, m_uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			if (iCmp < 0)
			{
				*pbAlwaysFalse = TRUE;
			}
			else
			{
				pPred->pUntilValue = pUntilValue;
				pPred->bInclUntil = bInclUntil;
			}
		}
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	See if a predicate can be unioned with another one.
//-------------------------------------------------------------------------
RCODE SQLQuery::unionPredicates(
	SQL_PRED *				pPred,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL					bNotted,
	SQL_VALUE *				pValue,
	FLMBOOL *				pbUnioned)
{
	RCODE		rc = NE_SFLM_OK;
	FLMINT	iCmp;
	FLMBOOL	bDoMatch;

	*pbUnioned = FALSE;
	if (!pValue || pValue->eValType != SQL_UTF8_VAL)
	{
		bDoMatch = FALSE;

		// Comparison rules don't matter for anything that is
		// not text, so we normalize them to zero, so the test
		// below to see if the comparison rule is the same as
		// the comparison rule of the operator will work.

		uiCompareRules = 0;
	}
	else
	{
		bDoMatch = (eOperator == SQL_EQ_OP &&
						(pValue->uiFlags & SQL_VAL_IS_CONSTANT) &&
						(pValue->uiFlags & SQL_VAL_HAS_WILDCARDS))
						? TRUE
						: FALSE;
	}

	if (eOperator == SQL_EXISTS_OP || eOperator == SQL_NE_OP)
	{

		// See if there is another operator that is an exact
		// match of this one.

		if (pPred->eOperator == eOperator &&
			 pPred->bNotted == bNotted)
		{
			
			// Perfect match - no need to do any more.
			
			*pbUnioned = TRUE;
		}
	}
	
	// See if the operator overlaps with another range operator

	else if (pPred->eOperator == SQL_RANGE_OP &&
				pPred->uiCompareRules == uiCompareRules &&
				!bDoMatch &&
				(eOperator == SQL_EQ_OP ||
				 eOperator == SQL_LE_OP ||
				 eOperator == SQL_LT_OP ||
				 eOperator == SQL_GE_OP ||
				 eOperator == SQL_GT_OP))
	{
		SQL_VALUE *	pFromValue;
		SQL_VALUE *	pUntilValue;
		FLMBOOL		bInclFrom;
		FLMBOOL		bInclUntil;

		pFromValue = (eOperator == SQL_EQ_OP ||
						  eOperator == SQL_GE_OP ||
						  eOperator == SQL_GT_OP)
						  ? pValue
						  : NULL;
		pUntilValue = (eOperator == SQL_EQ_OP ||
							eOperator == SQL_LE_OP ||
							eOperator == SQL_LT_OP)
							? pValue
							: NULL;
		bInclFrom = (FLMBOOL)(eOperator == SQL_EQ_OP ||
									 eOperator == SQL_GE_OP
									 ? TRUE
									 : FALSE);
		bInclUntil = (FLMBOOL)(eOperator == SQL_EQ_OP ||
									  eOperator == SQL_LE_OP
									  ? TRUE
									  : FALSE);

		// If the value type is not compatible with the predicate's
		// value type, we cannot do the comparison, and there is
		// no overlap.

		if (!sqlCanCompare( pValue, pPred->pFromValue) ||
			 !sqlCanCompare( pValue, pPred->pUntilValue))
		{
			// Nothing to do here
		}
		else if (RC_BAD( rc = sqlCompareValues( pFromValue,
							bInclFrom, TRUE,
							pPred->pFromValue, pPred->bInclFrom, TRUE,
							uiCompareRules, m_uiLanguage, &iCmp)))
		{
			goto Exit;
		}
		else if (iCmp >= 0)
		{

			// From value is greater than or equal to the predicate's
			// from value.
			// If the from value is also less than or equal to the
			// predicate's until value, we have an overlap.

			if (RC_BAD( rc = sqlCompareValues( pFromValue,
						bInclFrom, TRUE,
						pPred->pUntilValue, pPred->bInclUntil, FALSE,
						uiCompareRules, m_uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			if (iCmp <= 0)
			{

				// If the until value is greater than the predicate's
				// until value, change the predicate's until value.

				if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pUntilValue, pPred->bInclUntil, FALSE,
							uiCompareRules, m_uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				if (iCmp > 0)
				{
					pPred->pUntilValue = pUntilValue;
					pPred->bInclUntil = bInclUntil;
				}
				*pbUnioned = TRUE;
				goto Exit;
			}
		}

		// At this point we already know that the from value is
		// less than the predicate's from value.
		// See if the until value is greater than or equal
		// to the from value.  If it is we have an overlap.

		else if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pFromValue, pPred->bInclFrom, TRUE,
							uiCompareRules, m_uiLanguage, &iCmp)))
		{
			goto Exit;
		}
		else if (iCmp >= 0)
		{

			// Until value is greater than or equal to the predicate's
			// from value, so we definitely have an overlap.  We
			// already know that the from value is less than the
			// predicate's from value, so we will change that for sure.

			pPred->pFromValue = pFromValue;
			pPred->bInclFrom = bInclFrom;

			// See if the until value is greater than the
			// predicate's until value, in which case we need to
			// change the predicate's until value.

			if (RC_BAD( rc = sqlCompareValues( pUntilValue,
							bInclUntil, FALSE,
							pPred->pUntilValue, pPred->bInclUntil, FALSE,
							uiCompareRules, m_uiLanguage, &iCmp)))
			{
				goto Exit;
			}
			if (iCmp > 0)
			{
				pPred->pUntilValue = pUntilValue;
				pPred->bInclUntil = bInclUntil;
			}
			*pbUnioned = TRUE;
		}
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Convert an operand to a predicate.  If it is merged with another
//			predicate, remove it and return the next node in the list of
//			operands.  If it is not merged, still return the next node in
//			the list of operands.
//-------------------------------------------------------------------------
RCODE SQLQuery::addPredicate(
	SQL_SUBQUERY *			pSubQuery,
	FLMUINT *				puiOperand,
	SQL_TABLE *				pTable,
	FLMUINT					uiColumnNum,
	eSQLQueryOperators	eOperator,
	FLMUINT					uiCompareRules,
	FLMBOOL					bNotted,
	SQL_VALUE *				pValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiOperand = *puiOperand;
	FLMUINT		uiLoop;
	SQL_NODE *	pCurrNode;
	SQL_NODE *	pOperandNode = pSubQuery->ppOperands [uiOperand];
	FLMBOOL		bAlwaysFalse;
	SQL_PRED *	pPred;
	
	// Convert the constant value in a node id predicate to
	// a 64 bit unsigned value.

	if (eOperator != SQL_EXISTS_OP && !uiColumnNum)
	{
		if (RC_BAD( rc = sqlGetRowIdValue( pValue)))
		{
			goto Exit;
		}
	}
	
	// Look at all of the operands up to the one we are processing to
	// see if this operand should be merged with a previous one.

	for (uiLoop = 0; uiLoop < uiOperand; uiLoop++)
	{
		pCurrNode = pSubQuery->ppOperands [uiLoop];
		if (pCurrNode->eNodeType != SQL_PRED_NODE)
		{
			pCurrNode = pCurrNode->pNextSib;
			continue;
		}
		pPred = &pCurrNode->nd.pred;
		if (pPred->pTable == pTable && pPred->uiColumnNum == uiColumnNum)
		{
			FLMBOOL bIntersected;
			
			if (RC_BAD( rc = intersectPredicates( pPred, eOperator,
										uiCompareRules, bNotted, pValue,
										&bAlwaysFalse, &bIntersected)))
			{
				goto Exit;
			}
			if (!bIntersected && !bAlwaysFalse)
			{
				continue;
			}
			
			// If we get a false result, then we know that the
			// intersection of predicates is creating a situation where
			// it can never be true, so this sub-query can never return
			// anything.  Therefore, we remove the sub-query.
	
			if (bAlwaysFalse)
			{
				
				// Remove the sub-query - it will never return anything.
				
				if (pSubQuery->pPrev)
				{
					pSubQuery->pPrev->pNext = pSubQuery->pNext;
				}
				else
				{
					m_pFirstSubQuery = pSubQuery->pNext;
				}
				if (pSubQuery->pNext)
				{
					pSubQuery->pNext->pPrev = pSubQuery->pPrev;
				}
				else
				{
					m_pLastSubQuery = pSubQuery->pPrev;
				}
				
				// Setup so that we will quit processing this sub-query's
				// operands - it is now unlinked.
				
				uiOperand = pSubQuery->uiOperandCount;
			}
			else
			{
				
				flmAssert( bIntersected);
				
				// We intersected, so we want to remove the current
				// operand node out of the list and set up so that
				// we will increment to the next one in the list.

				pSubQuery->uiOperandCount--;
				if (uiOperand < pSubQuery->uiOperandCount)
				{
					f_memmove( &pSubQuery->ppOperands [uiOperand],
						&pSubQuery->ppOperands [uiOperand + 1],
						sizeof( SQL_NODE *) * (pSubQuery->uiOperandCount - uiOperand));
				}
			}
			goto Exit;
		}
	}

	// If we didn't find one to intersect with or union with, we need to
	// create a new operand node of type SQL_PRED_NODE.  Can't just modify
	// this node, because other sub-queries may be pointing to it also, and
	// they would modify it in a different way.  Unlike other nodes, predicate
	// nodes are ALWAYS tied to one and only one sub-query.
	
	if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE),
										(void **)&pOperandNode)))
	{
		goto Exit;
	}

	// Set the stuff that needs to be set for this predicate.
	
	pOperandNode->eNodeType = SQL_PRED_NODE;
	if (RC_BAD( rc = setupPredicate( &pOperandNode->nd.pred,
								pTable, uiColumnNum,
								eOperator, uiCompareRules, bNotted, pValue)))
	{
		goto Exit;
	}
	
	// Go to the next operand
	
	uiOperand++;

Exit:

	*puiOperand = uiOperand;

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Convert all of the operands underneath an AND or OR operator to
//			predicates where possible, except for the operands which are AND
//			or OR nodes.
//-------------------------------------------------------------------------
RCODE SQLQuery::convertOperandsToPredicates( void)
{
	RCODE						rc = NE_SFLM_OK;
	SQL_NODE *				pSQLNode;
	SQL_VALUE *				pValue;
	SQL_TABLE *				pTable;
	FLMUINT					uiColumnNum;
	FLMUINT					uiOperand;
	SQL_SUBQUERY *			pSubQuery;
	SQL_SUBQUERY *			pNextSubQuery;
	
	pSubQuery = m_pFirstSubQuery;
	while (pSubQuery)
	{
		pNextSubQuery = pSubQuery->pNext;
		uiOperand = 0;
		while (uiOperand < pSubQuery->uiOperandCount)
		{
			pSQLNode = pSubQuery->ppOperands [uiOperand];
			if (pSQLNode->eNodeType == SQL_COLUMN_NODE)
			{
				if (RC_BAD( rc = addPredicate( pSubQuery, &uiOperand,
									pSQLNode->nd.column.pTable,
									pSQLNode->nd.column.uiColumnNum,
									SQL_EXISTS_OP, 0, pSQLNode->bNotted, NULL)))
				{
					goto Exit;
				}
			}
			else if (pSQLNode->eNodeType == SQL_OPERATOR_NODE &&
						isSQLCompareOp( pSQLNode->nd.op.eOperator) &&
						((pSQLNode->pFirstChild->eNodeType == SQL_COLUMN_NODE &&
						  pSQLNode->pLastChild->eNodeType == SQL_VALUE_NODE) ||
						 (pSQLNode->pFirstChild->eNodeType == SQL_VALUE_NODE &&
						  pSQLNode->pLastChild->eNodeType == SQL_COLUMN_NODE)))
			{
				eSQLQueryOperators	eOperator = pSQLNode->nd.op.eOperator;
				
				// Have a Column,Op,Value or Value,Op,Column.  Convert to a
				// predicate node and merge with other predicate nodes that
				// have already been created, if possible.
				
				if (pSQLNode->pFirstChild->eNodeType == SQL_COLUMN_NODE)
				{
					pTable = pSQLNode->pFirstChild->nd.column.pTable;
					uiColumnNum = pSQLNode->pFirstChild->nd.column.uiColumnNum;
					pValue = &pSQLNode->pLastChild->currVal;
				}
				else
				{
					pTable = pSQLNode->pLastChild->nd.column.pTable;
					uiColumnNum = pSQLNode->pLastChild->nd.column.uiColumnNum;
					pValue = &pSQLNode->pFirstChild->currVal;
					
					// Need to invert the operator in this case.
					
					switch (pSQLNode->nd.op.eOperator)
					{
						case SQL_EQ_OP:
						case SQL_NE_OP:
							// No change
							break;
						case SQL_LT_OP:
							eOperator = SQL_GE_OP;
							break;
						case SQL_LE_OP:
							eOperator = SQL_GT_OP;
							break;
						case SQL_GT_OP:
							eOperator = SQL_LE_OP;
							break;
						case SQL_GE_OP:
							eOperator = SQL_LT_OP;
							break;
						default:
							// Should never get here!
							flmAssert( 0);
							break;
					}
				}
				
				if (RC_BAD( rc = addPredicate( pSubQuery, &uiOperand,
									pTable, uiColumnNum,
									eOperator, pSQLNode->nd.op.uiCompareRules,
									pSQLNode->bNotted, pValue)))
				{
					goto Exit;
				}
			}
			else
			{
				
				// Can't do anything with this operand, leave it and go to the
				// next one.
				
				uiOperand++;
			}
		}
		pSubQuery = pNextSubQuery;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Evaluate operands of an AND or OR operator to see if we can
//			replace one.
//			TRUE && P1 will be replaced with P1
//			FALSE && P1 will be replaced with FALSE
//			UNKNOWN && P1 will be replaced with UNKNOWN
//			TRUE || P1 will be replaced with TRUE
//			UNKNOWN || P1 will be replaced with UNKNOWN
//			FALSE || P1 will be replaced with P1
//-------------------------------------------------------------------------
FSTATIC SQL_NODE * sqlEvalLogicalOperands(
	SQL_NODE *		pSQLNode)
{
	eSQLQueryOperators	eOperator = pSQLNode->nd.op.eOperator;
	SQL_NODE *				pChildNode;
	SQLBoolType				eChildBoolVal;
	SQLBoolType				eClipValue = (eOperator == SQL_AND_OP)
												 ? SQL_TRUE
												 : SQL_FALSE;
	SQL_NODE *				pReplacementNode = NULL;
	
	pChildNode = pSQLNode->pFirstChild;
	while (pChildNode)
	{
		if (isSQLNodeBool( pChildNode))
		{
			eChildBoolVal = pChildNode->currVal.val.eBool;
		}
		else
		{
			pChildNode = pChildNode->pNextSib;
			continue;
		}
	
		// For AND operators eClipValue will be SQL_TRUE.  For OR
		// operators, it will be SQL_FALSE.  Those nodes should all be
		// clipped out.  If, after clipping the value, there is only
		// one node left, whatever it is should be moved up to replace
		// the AND or the OR  node.
		
		if (eChildBoolVal == eClipValue)
		{
			if (pChildNode->pPrevSib)
			{
				pChildNode->pPrevSib->pNextSib = pChildNode->pNextSib;
			}
			else
			{
				pSQLNode->pFirstChild = pChildNode->pNextSib;
			}
			if (pChildNode->pNextSib)
			{
				pChildNode->pNextSib->pPrevSib = pChildNode->pPrevSib;
			}
			else
			{
				pSQLNode->pLastChild = pChildNode->pPrevSib;
			}
			if (pSQLNode->pFirstChild != pSQLNode->pLastChild)
			{
				pChildNode = pChildNode->pNextSib;
				continue;
			}
			else
			{
				pReplacementNode = pSQLNode->pFirstChild;
				break;
			}
		}
		else
		{
			
			// The child node is a a boolean value that should simply replace
			// the AND or OR operator node.  This handles the following cases:
			//	1. Value is SQL_UNKNOWN and operator is SQL_OR or SQL_AND
			// 2. Value is SQL_FALSE and operator is SQL_AND
			// 3. Value is SQL_TRUE and operator is SQL_OR.
			
			pReplacementNode = pChildNode;
			break;
		}
	}

	// If we got a replacement node, link it in where the AND or OR
	// node was.
	
	if (pReplacementNode)
	{
		SQL_NODE *	pParentNode;
		
		if ((pParentNode = pSQLNode->pParent) == NULL)
		{
			pReplacementNode->pParent = NULL;
			pReplacementNode->pPrevSib = NULL;
			pReplacementNode->pNextSib = NULL;
		}
		else
		{
			pReplacementNode->pParent = pParentNode;
			if ((pReplacementNode->pPrevSib = pSQLNode->pPrevSib) != NULL)
			{
				pReplacementNode->pPrevSib->pNextSib = pReplacementNode;
			}
			else
			{
				pParentNode->pFirstChild = pReplacementNode;
			}
			
			if ((pReplacementNode->pNextSib = pSQLNode->pNextSib) != NULL)
			{
				pReplacementNode->pNextSib->pPrevSib = pReplacementNode;
			}
			else
			{
				pParentNode->pLastChild = pReplacementNode;
			}
		}
		pSQLNode = pReplacementNode;
	}

	return( pSQLNode);
}

//-------------------------------------------------------------------------
// Desc:	Clip a NOT node out of the tree.
//-------------------------------------------------------------------------
FSTATIC SQL_NODE * sqlClipNotNode(
	SQL_NODE *	pNotNode,
	SQL_NODE **	ppExpr)
{
	SQL_NODE *	pKeepNode;

	// If this NOT node has no parent, the root
	// of the tree needs to be set to its child.

	pKeepNode = pNotNode->pFirstChild;

	// Child better not have any siblings - NOT nodes only have
	// one operand.

	flmAssert( !pKeepNode->pNextSib && !pKeepNode->pPrevSib);

	// Set child to point to the NOT node's parent.

	if ((pKeepNode->pParent = pNotNode->pParent) == NULL)
	{
		*ppExpr = pKeepNode;
	}
	else
	{

		// Link child in where the NOT node used to be.

		if ((pKeepNode->pPrevSib = pNotNode->pPrevSib) != NULL)
		{
			pKeepNode->pPrevSib->pNextSib = pKeepNode;
		}
		else
		{
			pKeepNode->pParent->pFirstChild = pKeepNode;
		}
		if ((pKeepNode->pNextSib = pNotNode->pNextSib) != NULL)
		{
			pKeepNode->pNextSib->pPrevSib = pKeepNode;
		}
		else
		{
			pKeepNode->pParent->pLastChild = pKeepNode;
		}
	}
	return( pKeepNode);
}

//-------------------------------------------------------------------------
// Desc:	Reduce the query tree.  This will strip out NOT nodes and
//			resolve constant expressions to a single node.  It also weeds
//			out all boolean constants that are operands of AND or OR operators.
//			Finally, if the bFlattenTree parameter is TRUE, it will coalesce
//			AND and OR nodes so that they can have multiple operands.
//-------------------------------------------------------------------------
RCODE SQLQuery::reduceTree(
	FLMBOOL	bFlattenTree)
{
	RCODE						rc = NE_SFLM_OK;
	SQL_NODE *				pSQLNode = m_pQuery;
	SQL_NODE *				pTmpNode;
	SQL_NODE *				pParentNode = NULL;
	eSQLNodeTypes			eNodeType;
	eSQLQueryOperators	eOperator;
	FLMBOOL					bNotted = FALSE;

	for (;;)
	{
		eNodeType = pSQLNode->eNodeType;

		// Need to save bNotted on each node so that when we traverse
		// back up the tree it can be reset properly.  If bNotted is
		// TRUE and pSQLNode is an operator, we may change the operator in
		// some cases.  Even if we change the operator, we still want to
		// set the bNotted flag because it also implies "for every" when set
		// to TRUE, and we need to remember that as well.

		pSQLNode->bNotted = bNotted;
		if (eNodeType == SQL_OPERATOR_NODE)
		{
			eOperator = pSQLNode->nd.op.eOperator;
			if (eOperator == SQL_AND_OP || eOperator == SQL_OR_OP)
			{
				// AND and OR nodes better have child nodes
				
				if (!pSQLNode->pFirstChild || !pSQLNode->pLastChild)
				{
					flmAssert( 0);
					rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
					goto Exit;
				}
				if (bNotted)
				{
					eOperator = (eOperator == SQL_AND_OP
									 ? SQL_OR_OP
									 : SQL_AND_OP);
					pSQLNode->nd.op.eOperator = eOperator;
				}
				if (pParentNode)
				{
					
					// Logical sub-expressions can only be operands of
					// AND, OR, or NOT operators.

					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
					if (bFlattenTree && pParentNode->nd.op.eOperator == eOperator)
					{
						
						// Move all of pSQLNode's children become the immediate
						// children of pParentNode.
						
						pTmpNode = pSQLNode->pFirstChild;
						while (pTmpNode)
						{
							pTmpNode->pParent = pParentNode;
							pTmpNode = pTmpNode->pNextSib;
						}
						
						if (pSQLNode->pPrevSib)
						{
							pSQLNode->pPrevSib->pNextSib = pSQLNode->pFirstChild;
							pSQLNode->pFirstChild->pPrevSib = pSQLNode->pPrevSib;
						}
						if (pSQLNode->pNextSib)
						{
							pSQLNode->pNextSib->pPrevSib = pSQLNode->pLastChild;
							pSQLNode->pLastChild->pNextSib = pSQLNode->pNextSib;
						}
						
						// Continue processing from pSQLNode's first child, which
						// is the beginning of the list of nodes we just replaced
						// pSQLNode with.
						
						pSQLNode = pSQLNode->pFirstChild;
						continue;
					}
				}
			}
			else if (eOperator == SQL_NOT_OP)
			{

				// Logical sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				bNotted = !bNotted;

				// Clip NOT nodes out of the tree.

				pSQLNode = sqlClipNotNode( pSQLNode, &m_pQuery);
				pParentNode = pSQLNode->pParent;
				continue;
			}
			else if (isSQLCompareOp( eOperator))
			{

				// Comparison sub-expressions can only be operands of
				// AND, OR, or NOT operators.

				if (pParentNode)
				{
					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				if (bNotted)
				{
					switch (eOperator)
					{
						case SQL_EQ_OP:
							eOperator = SQL_NE_OP;
							break;
						case SQL_NE_OP:
							eOperator = SQL_EQ_OP;
							break;
						case SQL_LT_OP:
							eOperator = SQL_GE_OP;
							break;
						case SQL_LE_OP:
							eOperator = SQL_GT_OP;
							break;
						case SQL_GT_OP:
							eOperator = SQL_LE_OP;
							break;
						case SQL_GE_OP:
							eOperator = SQL_LT_OP;
							break;
						default:

							// Don't change the other operators.
							// Will just use the bNotted flag when
							// evaluating.

							break;
					}
					pSQLNode->nd.op.eOperator = eOperator;
				}
			}
			else
			{

				// Better be an arithmetic operator we are dealing with
				// at this point.

				flmAssert( isSQLArithOp( eOperator));

				// Arithmetic sub-expressions can only be operands
				// of arithmetic or comparison operators

				if (pParentNode)
				{
					if (!isSQLCompareOp( pParentNode->nd.op.eOperator) &&
						 !isSQLArithOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
			}
		}
		else if (eNodeType == SQL_COLUMN_NODE)
		{
			flmAssert( !pSQLNode->pFirstChild);
		}
		else
		{
			flmAssert( eNodeType == SQL_VALUE_NODE);

			// If bNotted is TRUE and we have a boolean value, change
			// the value: FALSE ==> TRUE, TRUE ==> FALSE.

			if (bNotted && pSQLNode->currVal.eValType == SQL_BOOL_VAL)
			{
				if (pSQLNode->currVal.val.eBool == SQL_TRUE)
				{
					pSQLNode->currVal.val.eBool = SQL_FALSE;
				}
				else if (pSQLNode->currVal.val.eBool == SQL_FALSE)
				{
					pSQLNode->currVal.val.eBool = SQL_TRUE;
				}
			}

			// Values can only be operands of arithmetic or comparison operators,
			// unless they are boolean values, in which case they can only be
			// operands of logical operators.

			if (pParentNode)
			{
				if (pSQLNode->currVal.eValType == SQL_BOOL_VAL)
				{
					if (!isSQLLogicalOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
				else
				{
					if (!isSQLCompareOp( pParentNode->nd.op.eOperator) &&
						 !isSQLArithOp( pParentNode->nd.op.eOperator))
					{
						rc = RC_SET( NE_SFLM_Q_ILLEGAL_OPERAND);
						goto Exit;
					}
				}
			}

			// A value node should not have any children

			flmAssert( !pSQLNode->pFirstChild);
		}

		// Do traversal to child node, if any

		if (pSQLNode->pFirstChild)
		{
			pParentNode = pSQLNode;
			pSQLNode = pSQLNode->pFirstChild;
			continue;
		}

		// Go back up the tree until we hit something that has
		// a sibling.

		while (!pSQLNode->pNextSib)
		{

			// If there are no more parents, we are done.

			if ((pSQLNode = pSQLNode->pParent) == NULL)
			{
				goto Exit;
			}

			flmAssert( pSQLNode->eNodeType == SQL_OPERATOR_NODE);

			// Evaluate arithmetic expressions if both operands are
			// constants.

			if (isSQLArithOp( pSQLNode->nd.op.eOperator) &&
				 pSQLNode->pFirstChild->eNodeType == SQL_VALUE_NODE &&
				 pSQLNode->pLastChild->eNodeType == SQL_VALUE_NODE)
			{
				if (RC_BAD( rc = sqlEvalArithOperator(
											&pSQLNode->pFirstChild->currVal,
											&pSQLNode->pLastChild->currVal,
											pSQLNode->nd.op.eOperator,
											&pSQLNode->currVal)))
				{
					goto Exit;
				}
				pSQLNode->eNodeType = SQL_VALUE_NODE;
				pSQLNode->currVal.uiFlags = SQL_VAL_IS_CONSTANT;
				pSQLNode->pFirstChild = NULL;
				pSQLNode->pLastChild = NULL;
			}
			else
			{

				// For the AND and OR operators, check the operands to
				// see if they are boolean values.  Boolean values can
				// be weeded out of the criteria as we go back up the
				// tree.

				if (pSQLNode->nd.op.eOperator == SQL_OR_OP ||
					 pSQLNode->nd.op.eOperator == SQL_AND_OP)
				{
					pSQLNode = sqlEvalLogicalOperands( pSQLNode);
					if (!pSQLNode->pParent)
					{
						m_pQuery = pSQLNode;
					}
				}
			}

			pParentNode = pSQLNode->pParent;
		}

		// pSQLNode will NEVER be NULL if we get here, because we
		// will jump to Exit in those cases.

		pSQLNode = pSQLNode->pNextSib;

		// Need to reset the bNotted flag to what it would have
		// been as we traverse back up the tree.

		bNotted = pParentNode->bNotted;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Allocate and set up a DNF node.
//-------------------------------------------------------------------------
FSTATIC RCODE createDNFNode(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pParentDNFNode,
	SQL_DNF_NODE **	ppDNFNode,
	SQL_NODE *			pNode)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_DNF_NODE *	pDNFNode;
	
	if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
												(void **)&pDNFNode)))
	{
		goto Exit;
	}
	
	// pDNFNode->pNode will be NULL if it is an AND or OR operator.
	
	if (pNode->eNodeType == SQL_OPERATOR_NODE)
	{
		if (pNode->nd.op.eOperator == SQL_AND_OP)
		{
			pDNFNode->bAndOp = TRUE;
		}
		else if (pNode->nd.op.eOperator == SQL_OR_OP)
		{
			// No need to really set as it is already 0 from poolCalloc.
			// pDNFNode->bAndOp = FALSE;
		}
		else
		{
			pDNFNode->pNode = pNode;
		}
	}
	else
	{
		pDNFNode->pNode = pNode;
	}
	if ((pDNFNode->pParent = pParentDNFNode) != NULL)
	{
		if ((pDNFNode->pPrevSib = pParentDNFNode->pLastChild) != NULL)
		{
			pDNFNode->pPrevSib->pNextSib = pDNFNode;
		}
		else
		{
			pParentDNFNode->pFirstChild = pDNFNode;
		}
		pParentDNFNode->pLastChild = pDNFNode;
	}
	*ppDNFNode = pDNFNode;
	
Exit:

	return( rc);
}
	
//-------------------------------------------------------------------------
// Desc:	Copy the sub-tree pointed to by pSrcSubTree and then link the
//			new sub-tree as the last child of pParentNode.
//-------------------------------------------------------------------------
FSTATIC RCODE copyAndLinkSubTree(
	F_Pool *			pPool,
	SQL_DNF_NODE *	pSrcSubTree,
	SQL_DNF_NODE *	pParentNode)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_DNF_NODE *	pNewSubTree = NULL;
	SQL_DNF_NODE *	pCurrDestParentNode = NULL;
	SQL_DNF_NODE *	pCurrSrcNode = pSrcSubTree;
	SQL_DNF_NODE *	pNewDestNode = NULL;
	
	for (;;)
	{
		if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
													(void **)&pNewDestNode)))
		{
			goto Exit;
		}
		pNewDestNode->pNode = pCurrSrcNode->pNode;
		pNewDestNode->bAndOp = pCurrSrcNode->bAndOp;
		if (!pNewSubTree)
		{
			pNewSubTree = pNewDestNode;
		}
		else
		{
			pNewDestNode->pParent = pCurrDestParentNode;
			if ((pNewDestNode->pPrevSib = pCurrDestParentNode->pLastChild) != NULL)
			{
				pNewDestNode->pPrevSib->pNextSib = pNewDestNode;
			}
			else
			{
				pCurrDestParentNode->pFirstChild = pNewDestNode;
			}
			pCurrDestParentNode->pLastChild = pNewDestNode;
		}
		
		// Try to go down to a child node
		
		if (pCurrSrcNode->pFirstChild)
		{
			pCurrSrcNode = pCurrSrcNode->pFirstChild;
			pCurrDestParentNode = pNewDestNode;
			continue;
		}
		
		// No child nodes, go back up parent chain until we find one that
		// has a sibling.
		
		for (;;)
		{
			if (pCurrSrcNode == pSrcSubTree)
			{
				break;
			}
			if (pCurrSrcNode->pNextSib)
			{
				break;
			}
			pCurrSrcNode = pCurrSrcNode->pParent;
			pCurrDestParentNode = pCurrDestParentNode->pParent;
		}
		if (pCurrSrcNode == pSrcSubTree)
		{
			break;
		}
		pCurrSrcNode = pCurrSrcNode->pNextSib;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Distribute an AND operator over an OR operator.  The AND operator
//			is the parent node of the passed in pOldOrNode.  A new list of
//			AND nodes is created which will replace the original AND node in
//			the tree.
//-------------------------------------------------------------------------
FSTATIC RCODE distributeAndOverOr(
	F_Pool *				pPool,
	SQL_DNF_NODE *		pOldOrNode,
	SQL_DNF_NODE **	ppDNFTree)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_DNF_NODE *	pOldAndNode;
	SQL_DNF_NODE *	pOldAndParentNode;
	SQL_DNF_NODE *	pNewAndNode;
	SQL_DNF_NODE *	pFirstNewAndNode;
	SQL_DNF_NODE *	pLastNewAndNode;
	SQL_DNF_NODE *	pOrChildNode;
	SQL_DNF_NODE *	pAndChildNode;

	// Parent node to pOldOrNode better be an AND node.
	
	pOldAndNode = pOldOrNode->pParent;
	flmAssert( !pOldAndNode->pNode && pOldAndNode->bAndOp);
				
	// Distribute ALL of the AND node's children (except this OR node)
	// across ALL of the OR node's children
	
	pFirstNewAndNode = NULL;
	pLastNewAndNode = NULL;
	pOrChildNode = pOldOrNode->pFirstChild;
	while (pOrChildNode)
	{
		if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
													(void **)&pNewAndNode)))
		{
			goto Exit;
		}
		pNewAndNode->bAndOp = TRUE;
		if ((pNewAndNode->pPrevSib = pLastNewAndNode) != NULL)
		{
			pLastNewAndNode->pNextSib = pNewAndNode;
		}
		else
		{
			pFirstNewAndNode = pNewAndNode;
		}
		pLastNewAndNode = pNewAndNode;
		
		// Copy all of the old AND node's children, except for this
		// OR node as children of the new AND node.
		
		pAndChildNode = pOldAndNode->pFirstChild;
		while (pAndChildNode)
		{
			if (pAndChildNode != pOldOrNode)
			{
				
				if (RC_BAD( rc = copyAndLinkSubTree( pPool, pAndChildNode, pNewAndNode)))
				{
					goto Exit;
				}
			}
			pAndChildNode = pAndChildNode->pNextSib;
		}
		
		// Copy the entire sub-tree of pOrChildNode and link it as the last
		// child of the new AND node.
		
		if (RC_BAD( rc = copyAndLinkSubTree( pPool, pOrChildNode, pNewAndNode)))
		{
			goto Exit;
		}
		pOrChildNode = pOrChildNode->pNextSib;
	}
	
	// Link the newly created AND list in where the old
	// AND node was (pOldAndNode).  If it was at the root
	// of the tree, we will need to create a new OR root.
	
	if ((pOldAndParentNode = pOldAndNode->pParent) == NULL)
	{
		if (RC_BAD( rc = pPool->poolCalloc( sizeof( SQL_DNF_NODE),
													(void **)&pOldAndParentNode)))
		{
			goto Exit;
		}
		
		// NOTE: No need to set anything in this new node, we want it to be
		// an OR node, which means that bAndOp is FALSE and pNode is NULL - both
		// of which are set by the poolCalloc.
		
		*ppDNFTree = pOldAndParentNode;
	}
	
	// Point all of the new AND nodes to the parent of the old AND node.
	
	pAndChildNode = pFirstNewAndNode;
	while (pAndChildNode)
	{
		pAndChildNode->pParent = pOldAndParentNode;
		pAndChildNode = pAndChildNode->pNextSib;
	}
	
	// Link the new list of AND nodes where the old AND node was.
	// Although the old AND node is still allocated, it is no longer
	// pointed to from the tree.
	
	if ((pFirstNewAndNode->pPrevSib = pOldAndNode->pPrevSib) != NULL)
	{
		pFirstNewAndNode->pPrevSib->pNextSib = pFirstNewAndNode;
	}
	else
	{
		pOldAndParentNode->pFirstChild = pFirstNewAndNode;
	}
	if ((pLastNewAndNode->pNextSib = pOldAndNode->pNextSib) != NULL)
	{
		pLastNewAndNode->pNextSib->pPrevSib = pLastNewAndNode;
	}
	else
	{
		pOldAndParentNode->pLastChild = pLastNewAndNode;
	}
	
Exit:

	return( rc);
}
				
//-------------------------------------------------------------------------
// Desc:	Convert query tree to disjunctive normal form (DNF).  Result is
//			a list of sub-queries that are ORed together.
//-------------------------------------------------------------------------
RCODE SQLQuery::convertToDNF( void)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_NODE *		pCurrNode;
	SQL_DNF_NODE *	pParentDNFNode;
	SQL_DNF_NODE *	pCurrDNFNode;
	SQL_DNF_NODE *	pDNFTree;
	SQL_DNF_NODE *	pAndList;
	SQL_DNF_NODE *	pExprList;
	F_Pool			pool;
	SQL_SUBQUERY *	pSubQuery;
	FLMUINT			uiLoop;
	
	pool.poolInit( 1024); 
	
	// If the top node in the tree is not an AND or OR operator,
	// create a single subquery that has a single operand.
	
	if (m_pQuery->eNodeType != SQL_OPERATOR_NODE ||
		 (m_pQuery->nd.op.eOperator != SQL_AND_OP &&
		  m_pQuery->nd.op.eOperator != SQL_OR_OP))
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_SUBQUERY),
													(void **)&m_pFirstSubQuery)))
		{
			goto Exit;
		}
		m_pLastSubQuery = m_pFirstSubQuery;
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE *),
												(void **)&m_pFirstSubQuery->ppOperands)))
		{
			goto Exit;
		}
		m_pFirstSubQuery->uiOperandCount = 1;
		m_pFirstSubQuery->ppOperands [0] = m_pQuery;
		goto Exit;
	}
	
	// Create the tree of DNF nodes to point to all of the AND and OR nodes
	// in the tree and their immediate child nodes.

	pCurrNode = m_pQuery;
	pParentDNFNode = NULL;
	pDNFTree = NULL;
	for (;;)
	{
		if (RC_BAD( rc = createDNFNode( &pool, pParentDNFNode,
									&pCurrDNFNode, pCurrNode)))
		{
			goto Exit;
		}
		if (!pDNFTree)
		{
			pDNFTree = pCurrDNFNode;
		}
		
		// Don't traverse down to child nodes if it is not an AND or OR node.
		
		if (pCurrNode->eNodeType == SQL_OPERATOR_NODE &&
			 (pCurrNode->nd.op.eOperator == SQL_AND_OP ||
			  pCurrNode->nd.op.eOperator == SQL_OR_OP))
		{
			if (pCurrNode->pFirstChild)
			{
				pCurrNode = pCurrNode->pFirstChild;
				pParentDNFNode = pCurrDNFNode;
				continue;
			}
		}
		
		// Go back up to parent until we find one that has a sibling.
		
		while (!pCurrNode->pNextSib)
		{
			if ((pCurrNode = pCurrNode->pParent) == NULL)
			{
				break;
			}
			pParentDNFNode = pParentDNFNode->pParent;
		}
		if (!pCurrNode)
		{
			break;
		}
		pCurrNode = pCurrNode->pNextSib;
	}
	
	// Now traverse the DNF tree and move all OR operators to the top.
	// When we are done we should have a DNF tree with either a single AND
	// node and a list of subordinate expressions, or a single OR node with
	// a mix of AND child nodes or non-AND expressions.

	pCurrDNFNode = pDNFTree;	
	for (;;)
	{
		
		// If we hit an OR node that is not the root node, it's parent should be
		// an AND node.  Distribute the AND node's operands over all of the
		// OR node's operands.
			
		if (pCurrDNFNode->pNode->eNodeType == SQL_OPERATOR_NODE &&
			 pCurrDNFNode->pNode->nd.op.eOperator == SQL_OR_OP &&
			 pCurrDNFNode->pParent)
		{
			if (RC_BAD( rc = distributeAndOverOr( &pool, pCurrDNFNode,
											&pDNFTree)))
			{
				goto Exit;
			}
			
			// Start over at the top of the tree.
			
			pCurrDNFNode = pDNFTree;
			continue;
		}
		
		// Go to first child, if there is one.
		
		if (pCurrDNFNode->pFirstChild)
		{
			pCurrDNFNode = pCurrDNFNode->pFirstChild;
			continue;
		}
		
		// No child nodes, go to sibling nodes.  If no sibling nodes,
		// traverse back up parent chain until we find one.
		
		while (!pCurrDNFNode->pNextSib)
		{
			if ((pCurrDNFNode = pCurrDNFNode->pParent) == NULL)
			{
				break;
			}
		}
		if (!pCurrDNFNode)
		{
			break;
		}
		pCurrDNFNode = pCurrDNFNode->pNextSib;
	}
	
	// If we get to this point, we have created a DNF tree that either
	// as an OR at the top or an AND at the top.  If it is an OR at the
	// top, we have multiple sub-queries.  If it is an AND at the top, we
	// have a single sub-query.
	
	if (pDNFTree->bAndOp)
	{
		pAndList = pDNFTree;
	}
	else
	{
		pAndList = pDNFTree->pFirstChild;
		flmAssert( pAndList);
	}
	
	while (pAndList)
	{
		if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_SUBQUERY),
													(void **)&pSubQuery)))
		{
			goto Exit;
		}
		
		// Link the subquery as the last sub-query in our sub-query list
		
		if ((pSubQuery->pPrev = m_pLastSubQuery) != NULL)
		{
			pSubQuery->pPrev->pNext = pSubQuery;
		}
		else
		{
			m_pFirstSubQuery = pSubQuery;
		}
		m_pLastSubQuery = pSubQuery;

		// The child may be a simple expression, in which case it is its
		// own sub-query.

		if (pAndList->pNode)
		{
			pSubQuery->uiOperandCount = 1;
			
			// The expression should not have any child nodes.
			
			flmAssert( !pExprList->pFirstChild && !pExprList->pLastChild);
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE *),
													(void **)&pSubQuery->ppOperands)))
			{
				goto Exit;
			}
			pSubQuery->ppOperands [0] = pAndList->pNode;
			
			// NULL out the node's parent pointer and sibling pointers - just
			// to keep things tidy.
			
			pAndList->pNode->pParent = NULL;
			pAndList->pNode->pNextSib = NULL;
			pAndList->pNode->pPrevSib = NULL;
		}
		else
		{
			
			// Count the expressions in the list - should be at least one.
			
			pExprList = pAndList->pFirstChild;
			flmAssert( pExprList);
			while (pExprList)
			{
				
				// All of the expressions should point to nodes in the query
				// tree, and should  not be AND or OR nodes.  Furthermore,
				// they should not have child nodes
				
				flmAssert( pExprList->pNode && !pExprList->pFirstChild &&
								!pExprList->pLastChild);
				pSubQuery->uiOperandCount++;
				pExprList = pExprList->pNextSib;
			}
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_NODE *) * pSubQuery->uiOperandCount,
													(void **)&pSubQuery->ppOperands)))
			{
				goto Exit;
			}
			
			// Set the pointers in the operand list for the sub-query.
			
			for (uiLoop = 0, pExprList = pAndList->pFirstChild;
				  pExprList;
				  uiLoop++, pExprList = pExprList->pNextSib)
			{
				pSubQuery->ppOperands [uiLoop] = pExprList->pNode;
				
				// NULL out the node's parent pointer and sibling pointers - just
				// to keep things tidy.
				
				pExprList->pNode->pParent = NULL;
				pExprList->pNode->pNextSib = NULL;
				pExprList->pNode->pPrevSib = NULL;
			}
		}
		flmAssert( uiLoop == pSubQuery->uiOperandCount);
		pAndList = pAndList->pNextSib;
	}
	
Exit:

	return( rc);
}

#if 0
//-------------------------------------------------------------------------
// Desc: Determine if a particular predicate is associated with the
//			specified table.  Only return TRUE if the predicate is associated
//			with this table and only with this table.
//-------------------------------------------------------------------------
FSTATIC FLMBOOL predIsForTable(
	SQL_NODE *	pPredRootNode,
	SQL_TABLE *	pTable)
{
	FLMBOOL		bIsAssociated = FALSE;
	SQL_NODE *	pCurrNode = pPredRootNode;
	
	for (;;)
	{
		if (pCurrNode->eNodeType == SQL_COLUMN_NODE)
		{
			 if (pCurrNode->nd.column.pTable == pTable)
			 {
				 bIsAssociated = TRUE;
			 }
			 else
			 {
				 bIsAssociated = FALSE;
				 break;
			 }
		}
		
		if (pCurrNode->pFirstChild)
		{
			pCurrNode = pCurrNode->pFirstChild;
			continue;
		}
		
		// No child nodes, traverse to sibling - or sibling of first node
		// in the parent chain that has a next sibling.
		
		for (;;)
		{
			if (pCurrNode == pPredRootNode)
			{
				break;
			}
			if (pCurrNode->pNextSib)
			{
				break;
			}
			pCurrNode = pCurrNode->pParent;
		}
		if (pCurrNode == pPredRootNode)
		{
			break;
		}
		
		// If we get to here, there should be a next sibling.
		
		pCurrNode = pCurrNode->pNextSib;
		flmAssert( pCurrNode);
	}
	
	return( bIsAssociated);
}

//-------------------------------------------------------------------------
// Desc:	Associate a predicate with all of the indexes it pertains to
// 		with respect to a particular table.
//-------------------------------------------------------------------------
RCODE SQLQuery::getPredKeys(
	SQL_PRED *	pPred,
	SQL_TABLE *	pTable)
{
	RCODE				rc = NE_SFLM_OK;
	ICD *				pIcd;
	SQL_INDEX *		pIndex;
	SQL_KEY *		pKey;
	
//visit - notes for reference: column number should be unique for the table.
//visit - notes for reference: no required/non-required pieces for indexes on tables


	if (RC_BAD( rc = m_pDb->m_pDict->getAttribute( m_pDb, pPred->uiColumnNum,
								&defInfo)))
	{
		goto Exit;
	}
	
	// This ICD chain will only contain ICDs for this particular column on
	// the table the column belongs to - because column numbers are globally
	// unique.

	for (pIcd = defInfo.m_pFirstIcd; pIcd; pIcd = pIcd->pNextInChain)
	{
		
		// If the table has an index specified for it, skip this ICD if
		// it is not that index.
		
		if (pTable->bIndexSet && pTable->uiIndex != pIcd->pIxd->uiIndexNum)
		{
			continue;
		}

		// Cannot use the index if it is off-line.

		if (pIcd->pIxd->uiFlags & (IXD_OFFLINE | IXD_SUSPENDED))
		{
			continue;
		}
		
		// Find the index off of the table.  If not there, add it.
		
		pIndex = pTable->pFirstIndex;
		while (pIndex->uiIndexNum != pIcd->pIxd->uiIndexNum)
		{
			pIndex = pIndex->pNext;
		}
		if (!pIndex)
		{
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_INDEX),
												(void **)&pIndex)))
			{
				goto Exit;
			}
			pIndex->pTable = pTable;
			pIndex->uiIndexNum = pIcd->pIxd->uiIndexNum;
			pIndex->uiNumComponents = pIcd->pIxd->uiNumKeyComponents;
			if ((pIndex->pPrev = pTable->pLastIndex) != NULL)
			{
				pIndex->pPrev->pNext = pIndex;
			}
			else
			{
				pTable->pFirstIndex = pIndex;
			}
			pTable->pLastIndex = pIndex;
			
			// Allocate a single key for the index.
			
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_KEY),
												(void **)&pKey)))
			{
				goto Exit;
			}
			pIndex->pLastKey = pIndex->pFirstKey = pKey;
			pKey->pIndex = pIndex;
			
			// Allocate an array of key components for the key.
			
			if (RC_BAD( rc = m_pool.poolCalloc( sizeof( SQL_PRED *) * pIndex->uiNumComponents,
												(void **)&pKey->ppKeyComponents)))
			{
				goto Exit;
			}
		}
		else
		{
			pKey = pIndex->pFirstKey;
		}
		
		// There should not be multiple predicates in a sub-query that
		// have the same column, so this key component should NOT already
		// be populated.
		
		flmAssert( !pKey->ppKeyComponents [pIcd->uiKeyComponent - 1]);
		pKey->ppKeyComponents [pIcd->uiKeyComponent - 1] = pPred;
		
		// NOTE: Costs will be calculated later.
		
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Determine the order in which to evaluate indexes for a particular
//			table and sub-query.  Those that have a primary key will be
//			give preference over those that don't.
//-------------------------------------------------------------------------
FSTATIC void rankIndexes(
	SQL_TABLE *	pTable)
{
	SQL_INDEX *	pIndex;
	SQL_INDEX *	pPrevIndex;
	SQL_INDEX *	pNextIndex;
	SQL_KEY *	pKey;
	FLMUINT		uiComponentCount;
	FLMUINT		uiPrevComponentCount;
	
	pIndex = pTable->pFirstIndex;
	while (pIndex)
	{
		pNextIndex = pIndex->pNext;
		pPrevIndex = pIndex->pPrev;
		
		// There should only be one key off of the index right now.
		pKey = pIndex->pFirstKey;
		flmAssert( !pKey->pNext);
		
		// Determine how many of the key's components point to a
		// predicate.  This stops at the first NULL pointer.  There may
		// be pointers after that one, but we really don't care, because
		// we won't use those components to generate a key.
		
		pKey->uiComponentsUsed = 0;
		while (pKey->uiComponentsUsed < pIndex->uiNumComponents &&
				 pKey->ppKeyComponents [pKey->uiComponentsUsed])
		{
			pKey->uiComponentsUsed++;
		}
		
		// See if this key is using more components that the key for
		// prior indexes.
		
		while (pPrevIndex)
		{
			if (pKey->uiComponentsUsed > pPrevIndex->pFirstKey->uiComponentsUsed)
			{
				// Move our current key up in front of the previous key - meaning
				// it will be evaluated ahead of that key.
				
				// First, unlink the index from its current spot.  pIndex->pPrev
				// must be non-NULL - otherwise, we wouldn't have a pPrevIndex.
				
				flmAssert( pIndex->pPrev);
				pIndex->pPrev->pNext = pIndex->pNext;
				if (pIndex->pNext)
				{
					pIndex->pNext->pPrev = pIndex->pPrev;
				}
				else
				{
					pTable->pLastIndex = pIndex->pPrev;
				}
				
				// Now, link it in front of pPrevIndex
				
				pIndex->pNext = pPrevIndex;
				if ((pIndex->pPrev = pPrevIndex->pPrev) != NULL)
				{
					pIndex->pPrev->pNext = pIndex;
				}
				else
				{
					pTable->pFirstIndex = pIndex;
				}
				pPrevIndex->pPrev = pIndex;
				pPrevIndex = pIndex->pPrev;
			}
			else
			{
				pPrevIndex = pPrevIndex->pPrev;
			}
		}
		
		pIndex = pNextIndex;
	}
}

//-------------------------------------------------------------------------
// Desc:	Choose the best index for a table of the indexes for which we have
//			generated predicate keys.
//-------------------------------------------------------------------------
RCODE SQLQuery::chooseBestIndex(
	SQL_TABLE *	pTable,
	FLMUINT *	puiCost)
{
	RCODE			rc = NE_SFLM_OK
	SQL_INDEX *	pIndex = pTable->pFirstIndex;
	
	while (pIndex)
	{
		
		// Should only be one key on each index at this point.
		
		flmAssert( pIndex->pFirstKey && pIndex->pFirstKey == pIndex->pLastKey);
		
		pIndex = pIndex->pNext;
	}

	visit
Exit:

	return( rc);
}
		
//-------------------------------------------------------------------------
// Desc:	Calculate the cost of doing a table scan for a table.
//-------------------------------------------------------------------------
RCODE SQLQuery::calcTableScanCost(
	SQL_TABLE *			pTable,
	FLMUINT64 *			pui64Cost,
	SQLTableCursor **	ppSQLTableCursor)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT64	ui64LeafBlocksBetween;
	FLMUINT64	ui64TotalRefs;
	FLMUINT		bTotalsEstimated;
	
	if ((*ppSQLTableCursor = f_new SQLTableCursor) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = (*ppSQLTableCursor)->setupRange( m_pDb,
								pTable->uiTableNum, TRUE, 1, FLM_MAX_UINT64,
								&ui64LeafBlocksBetween, &ui64TotalRefs,
								&bTotalsEstimated)))
	{
		(*ppSQLTableCursor)->Release();
		*ppSQLTableCursor = NULL;
		goto Exit;
	}
	if (!ui64LeafBlocksBetween)
	{
		*pui64Cost = 1;
	}
	else
	{
		*pui64Cost = ui64LeafBlocksBetween;
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Merge keys from pSrcTable into pDestTable.
//-------------------------------------------------------------------------
RCODE SQLQuery::mergeKeys(
	SQL_TABLE *	pDestTable,
	SQL_TABLE *	pSrcTable)
{
	visit
}

//-------------------------------------------------------------------------
// Desc:	Optimize a particular table for a particular sub-query.
//-------------------------------------------------------------------------
RCODE SQLQuery::optimizeTable(
	SQL_SUBQUERY *	pSubQuery,
	SQL_TABLE *		pTable)
{
	RCODE						rc = NE_SFLM_OK;
	SQL_TABLE				tmpTable;
	FLMUINT					uiLoop;
	SQL_NODE *				pOperand;
	SQL_PRED *				pPred;
	void *					pvMark = m_pool.poolMark();
	SQLTableCursor *		pSQLTableCursor = NULL;
	
	// This routine should not be called if the table has already been
	// marked to do a table scan.
visit - the caller should handle this case
	
	flmAssert( !pTable->bScan);
	
	f_memset( &tmpTable, 0, sizeof( SQL_TABLE));
	tmpTable.uiTableNum = pTable->uiTableNum;
	tmpTable.uiIndex = pTable->uiIndex;
	tmpTable.bIndexSet = pTable->bIndexSet;
	
	// Traverse the predicates of the sub-query.  If any are found
	// that are not predicates, the table must be scanned.
	
	for (uiLoop = 0, pOperand = pSubQuery->ppOperands [0];
		  uiLoop < pSubQuery->uiOperandCount;
		  uiLoop++, pOperand = pSubQuery->ppOperands [uiLoop])
	{
		
		// If we hit a predicate that has not been turned into
		// an SQL_PRED_NODE, it is not optimizable.
		
		if (pOperand->eNodeType != SQL_PRED_NODE)
		{
			
			// See if the current table is involved in this predicate.  If so,
			// and it is the only table involved, the table should be scanned.
			// Setting pFirstIndex and pLastIndex to NULL will cause this to
			// happen below.
			
			if (predIsForTable( pOperand, pTable))
			{
				m_pool.poolReset( pvMark);
				tmpTable.pFirstIndex = NULL;
				tmpTable.pLastIndex = NULL;
				break;
			}
		}
		else if (pOperand->nd.pred.pTable == pTable)
		{
			SQL_PRED *	pPred = &pOperand->nd.pred;
			
			// We cannot use from and until keys for not/negative operators.
			// We set pFirstIndex and pLastIndex to NULL to indicate that a
			// table scan must occur.
			
			if ((pPred->bNotted && pPred->eOperator == SQL_MATCH_OP) ||
				  pPred->eOperator == SQL_NE_OP)
			{
				m_pool.poolReset( pvMark);
				tmpTable.pFirstIndex = NULL;
				tmpTable.pLastIndex = NULL;
				break;
			}
			
visit - before we collect keys for this predicate, we should check to see
if the predicate is a subset of any other predicate in previous sub-queries that
we have already optimized where the predicate in the previous sub-query was used
to optimize that previous sub-query.  If so, we should simply merge this sub-query with
that one - it will be a waste of time to get another set of keys for this
sub-query.  This, of course, implies that we need to keep track of the
predicates that were selected to optimize a particular sub-query.
			
			// See if there are any indexes for this predicate's column.
			// For now we are just collecting them.  We will calculate
			// the best one later.
			
			if (RC_BAD( rc = getPredKeys( pPred, &tmpTable)))
			{
				goto Exit;
			}
		}
	}
	
	// If we didn't find indexes for this table, set the bScan flag.

	if (!tmpTable.pFirstIndex)
	{
		tmpTable.bScan = TRUE;
		if (RC_BAD( rc = calcTableScanCost( &tmpTable, &tmpTable.uiCost,
									&pSQLTableCursor)))
		{
			goto Exit;
		}
	}
	else
	{
	
		// Rank the indexes to determine which ones to estimate cost for
		// first.
		
		rankIndexes( &tmpTable);
		
		// Find the index with the lowest cost, if any.
		// If the lowest cost index is still high, estimate the cost of doing
		// a table scan.
		
		if (RC_BAD( rc = chooseBestIndex( &tmpTable, &tmpTable.uiCost)))
		{
			goto Exit;
		}
		
		// Should be one index left after this.  If the cost is high, see if
		// a table scan would be cheaper.
	
		if (tmpTable.uiCost > 8)
		{
			FLMUINT		uiScanCost;
			
			if (RC_BAD( rc = calcTableScanCost( &tmpTable, &uiScanCost,
									&pSQLTableCursor)))
			{
				goto Exit;
			}
			if (uiScanCost < tmpTable.uiCost)
			{
				m_pool.poolReset( pvMark);
				tmpTable.uiCost = uiScanCost;
				tmpTable.bScan = TRUE;
				tmpTable.pFirstIndex = NULL;
				tmpTable.pLastIndex = NULL;
			}
			else
			{
				pSQLTableCursor->Release();
				pSQLTableCursor = NULL;
			}
		}
	}
	
	// If we determined that we must do a table scan, set the bScan flag
	// for the master table.  Otherwise, merge these keys
	
	if (tmpTable.bScan)
	{
		pTable->bScan = TRUE;
		pTable->uiCost = tmpTable.uiCost;
		
		// Better have calculated a cost and have a collection
		// cursor at this point.
		
		flmAssert( pSQLTableCursor);
		pTable->pSQLTableCursor = pSQLTableCursor;
		pSQLTableCursor = NULL;
		pTable->pFirstIndex = NULL;
		pTable->pLastIndex = NULL;
	}
	else
	{
		if (RC_BAD( rc = mergeKeys( pTable, &tmpTable)))
		{
			goto Exit;
		}
		pTable->uiCost += tmpTable.uiCost;
	}
	
Exit:

	if (pSQLTableCursor)
	{
		pSQLTableCursor->Release();
	}

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Optimize the sub-queries of an SQL query.
//-------------------------------------------------------------------------
RCODE SQLQuery::optimizeSubQueries( void)
{
	RCODE				rc = NE_SFLM_OK;
	SQL_SUBQUERY *	pSubQuery;
	SQL_TABLE *		pTable;
	
	// For each table in our expression, attempt to pick an index for each
	// subquery.
	
	for (pTable = m_pFirstTable; pTable; pTable = pTable->pNext)
	{
		pSubQuery = m_pFirstSubQuery;
		while (pSubQuery)
		{
			if (RC_BAD( rc = optimizeTable( pSubQuery, pTable)))
			{
				goto Exit;
			}
			
			// If the optimization decided we should scan the table, there
			// is no need to look at any more sub-queries for this table.
			
			if (pTable->bScan)
			{
				break;
			}
			pSubQuery = pSubQuery->pNext;
		}
		
		// See if a table scan is going to be cheaper.
		
		if (!pTable->bScan && pTable->uiCost > 8)
		{
			SQLTableCursor *	pSQLTableCursor = NULL;
			FLMUINT					uiScanCost;
			
			if (RC_BAD( rc = calcTableScanCost( pTable, &uiScanCost,
										&pSQLTableCursor)))
			{
				goto Exit;
			}
			if (uiScanCost < pTable->uiCost)
			{
				pTable->uiCost = uiScanCost;
				pTable->bScan = TRUE;
				pTable->pSQLTableCursor = pSQLTableCursor;
				pTable->pFirstIndex = NULL;
				pTable->pLastIndex = NULL;
			}
			else
			{
				pSQLTableCursor->Release();
			}
		}
	}
	
Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Optimize an SQL query.
//-------------------------------------------------------------------------
RCODE SQLQuery::optimize( void)
{
	RCODE	rc = NE_SFLM_OK;
	
	if (m_bOptimized)
	{
		goto Exit;
	}
	
	// We save the F_Database object so that we can always check and make
	// sure we are associated with this database on any query operations
	// that occur after optimization. -- Link it into the list of queries
	// off of the F_Database object.  NOTE: We may not always use the
	// same F_Db object, but it must always be the same F_Database object.

	m_pDatabase = m_pDb->m_pDatabase;
	m_pNext = NULL;
	m_pDatabase->lockMutex();
	if ((m_pPrev = m_pDatabase->m_pLastSQLQuery) != NULL)
	{
		m_pPrev->m_pNext = this;
	}
	else
	{
		m_pDatabase->m_pFirstSQLQuery = this;
	}
	m_pDatabase->m_pLastSQLQuery = this;
	m_pDatabase->unlockMutex();
	
	// Make sure we have a completed expression

	if (!criteriaIsComplete())
	{
		rc = RC_SET( NE_SFLM_Q_INCOMPLETE_QUERY_EXPR);
		goto Exit;
	}

	m_uiLanguage = m_pDb->getDefaultLanguage();

	// An empty expression should scan the database and return everything.

	if (!m_pQuery)
	{
		if (m_bIndexSet && m_uiIndex)
		{
			rc = setupIndexScan();
		}
		else
		{
			m_bScan = TRUE;
		}
		goto Exit;
	}

	// Handle the case of a value node or arithmetic expression at the root
	// These types of expressions do not return results from the database.

	if (m_pQuery->eNodeType == SQL_VALUE_NODE)
	{
		if (m_pQuery->currVal.eValType == SQL_BOOL_VAL &&
			 m_pQuery->currVal.val.eBool == SQL_TRUE)
		{
			m_bScan = TRUE;
		}
		else
		{
			m_bEmpty = TRUE;
		}
	}
	else if (m_pQuery->eNodeType == SQL_OPERATOR_NODE &&
		  		isSQLArithOp( pQNode->nd.op.eOperator))
	{
		m_bEmpty = TRUE;
		goto Exit;
	}

	// If the user explicitly said to NOT use an index, we will not

	if (m_bIndexSet && !m_uiIndex)
	{
		m_bScan = TRUE;
		goto Exit;
	}

	// Flatten the AND and OR operators in the query tree.  Strip out
	// NOT operators, resolve constant arithmetic expressions, and
	// weed out boolean constants.
	
	if (RC_BAD( rc = reduceTree( TRUE)))
	{
		goto Exit;
	}
	
	// Convert to DNF
	
	if (RC_BAD( rc = convertToDNF()))
	{
		goto Exit;
	}
	
	// Convert all operands of each sub-query to predicates where
	// possible.
	
	if (RC_BAD( rc = convertOperandsToPredicates()))
	{
		goto Exit;
	}
	
	// Optimize each sub-query.
	
	if (RC_BAD( rc = optimizeSubQueries()))
	{
		goto Exit;
	}
	
	m_bOptimized = TRUE;
	
Exit:

	return( rc);
}
#endif
