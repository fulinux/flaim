//-------------------------------------------------------------------------
// Desc:	Structures, classes, prototypes, and defines needed to support
//			ODBC in FLAIM.
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

#ifndef FLAIMODBC_H
#define FLAIMODBC_H

// Forward declarations

struct SQL_NODE;
struct SQL_VALUE;
struct SQL_TABLE;
struct SQL_INDEX;
struct SQL_KEY;
struct SQL_COLUMN;
class FSTableCursor;

//-------------------------------------------------------------------------
// Desc:	Types of nodes in SQL query.
//-------------------------------------------------------------------------
typedef enum
{
	SQL_NO_NODE = 0,
	SQL_OPERATOR_NODE,
	SQL_VALUE_NODE,
	SQL_COLUMN_NODE,
	SQL_PRED_NODE
} eSQLNodeTypes;

#define SQL_FIRST_ARITH_OP		SQL_BITAND_OP
#define SQL_LAST_ARITH_OP		SQL_NEG_OP

FINLINE FLMBOOL isLegalSQLOperator(
	eSQLQueryOperators	eOperator)
{
	return( (eOperator >= SQL_AND_OP && eOperator <= SQL_NEG_OP)
			  ? TRUE
			  : FALSE);
}

FINLINE FLMBOOL isSQLLogicalOp(
	eSQLQueryOperators	eOperator)
{
	return( (eOperator >= SQL_AND_OP && eOperator <= SQL_NOT_OP) ? TRUE : FALSE);
}

FINLINE FLMBOOL isSQLCompareOp(
	eSQLQueryOperators	eOperator)
{
	return( (eOperator >= SQL_EQ_OP && eOperator <= SQL_GE_OP) ? TRUE : FALSE);
}

FINLINE FLMBOOL isSQLArithOp(
	eSQLQueryOperators	eOperator)
{
	return( (eOperator >= SQL_FIRST_ARITH_OP && 
				eOperator <= SQL_LAST_ARITH_OP) ? TRUE : FALSE);
}

FINLINE FLMBOOL isSQLValUnsigned(
	eSQLValTypes	eValType)
{
	return( eValType == SQL_UINT_VAL || eValType == SQL_UINT64_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL isSQLValSigned(
	eSQLValTypes	eValType)
{
	return( eValType == SQL_INT_VAL || eValType == SQL_INT64_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL isSQLVal64Bit(
	eSQLValTypes	eValType)
{
	return( eValType == SQL_UINT64_VAL || eValType == SQL_INT64_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL isSQLValNativeNum(
	eSQLValTypes	eValType)
{
	return( eValType == SQL_UINT_VAL || eValType == SQL_INT_VAL
			 ? TRUE
			 : FALSE);
}

typedef struct SQL_PRED
{
	SQL_TABLE *				pTable;
	FLMUINT					uiColumnNum;
	eSQLQueryOperators	eOperator;		// Operator of the predicate
	FLMUINT					uiCompareRules;// Comparison rules
	FLMBOOL					bNotted;			// Has operator been notted?
	SQL_VALUE *				pFromValue;		// Points to SQL_VALUE that has the FROM value for
													// this predicate.  Will be NULL for unary
													// operators such as exists
	FLMBOOL					bInclFrom;		// Flag indicating if the from value is
													// inclusive.
	SQL_VALUE *				pUntilValue;	// Points to SQL_VALUE that has the UNTIL value
													// for this predicate.
	FLMBOOL					bInclUntil;		// Flag indicating if until value is
													// inclusive.
	SQL_PRED *				pNext;													
} SQL_PRED;

typedef struct SQL_OP
{
	eSQLQueryOperators		eOperator;
	FLMUINT						uiCompareRules;
} SQL_OP;

typedef struct SQL_STRING_VALUE
{
	FLMBYTE *		pszStr;			// Should always be null-terminated.
	FLMUINT			uiByteLen;		// Includes null-terminating character.
	FLMUINT			uiNumChars;		// Does not count null-terminating character.
} SQL_STRING_VALUE;

typedef struct SQL_BINARY_VALUE
{
	FLMBYTE *		pucValue;
	FLMUINT			uiByteLen;
} SQL_BINARY_VALUE;

typedef struct SQL_VALUE
{
	eSQLValTypes	eValType;
	FLMUINT			uiFlags;
#define SQL_VAL_IS_STREAM		0x0001
#define SQL_VAL_IS_CONSTANT	0x0002	// During query evaluation, this indicates
													// that this value is a constant.  If it
													// is a FLM_UTF8_VAL, then asterisks will
													// be treated as a wildcard, unless
													// escaped (\*).  If the value is NOT
													// a constant, the asterisk is NEVER
													// treated as a wildcard, and the
													// backslash is NEVER treated as an
													// escape character.
#define SQL_VAL_HAS_WILDCARDS	0x0004	// This is only set if the value is a
													// constant, FLM_UTF8_VAL, that has
													// wildcards.
	FLMUINT		uiDataLen;					// Length in bytes if the type is text
													// or binary
	union
	{
		SQLBoolType				eBool;
		FLMUINT					uiVal;
		FLMUINT64				ui64Val;
		FLMINT					iVal;
		FLMINT64					i64Val;
		SQL_STRING_VALUE		str;
		SQL_BINARY_VALUE		bin;
		IF_PosIStream *		pIStream;
	} val;									// Holds or points to the atom value.
} SQL_VALUE;

/***************************************************************************
Desc:	Can two values be compared?
***************************************************************************/
FINLINE FLMBOOL sqlCanCompare(
	SQL_VALUE *	pValue1,
	SQL_VALUE *	pValue2
	)
{
	if (!pValue1 || !pValue2 ||
		 pValue1->eValType == pValue2->eValType)
	{
		return( TRUE);
	}
	else
	{
		switch (pValue1->eValType)
		{
			case SQL_UINT_VAL:
			case SQL_UINT64_VAL:
			case SQL_INT_VAL:
			case SQL_INT64_VAL:
				return( (FLMBOOL)(pValue2->eValType == SQL_UINT_VAL ||
										pValue2->eValType == SQL_UINT64_VAL ||
										pValue2->eValType == SQL_INT_VAL ||
										pValue2->eValType == SQL_INT64_VAL
										? TRUE
										: FALSE));
			default:
				return( FALSE);
		}
	}
}

typedef struct SQL_KEY
{
	SQL_INDEX *				pIndex;
	FLMUINT					uiComponentsUsed;
	SQL_PRED **				ppKeyComponents;
	SQL_KEY *				pNext;
	SQL_KEY *				pPrev;
} SQL_KEY;

typedef struct SQL_INDEX
{
	FLMUINT		uiIndexNum;
	FLMUINT		uiNumComponents;
	SQL_TABLE *	pTable;
	SQL_KEY *	pFirstKey;
	SQL_KEY *	pLastKey;
	SQL_INDEX *	pNext;
	SQL_INDEX *	pPrev;
} SQL_INDEX;

typedef struct SQL_TABLE
{
	FLMUINT					uiTableNum;
	FSTableCursor *		pFSTableCursor;
	FLMUINT					uiCost;
	FLMBOOL					bScan;
	FLMBOOL					bScanIndex;
	FLMUINT					uiIndex;
	FLMBOOL					bIndexSet;
	SQL_INDEX *				pFirstIndex;
	SQL_INDEX *				pLastIndex;
	SQL_TABLE *				pNext;
	SQL_TABLE *				pPrev;
} SQL_TABLE;

typedef struct SQL_COLUMN
{
	SQL_TABLE *	pTable;
	FLMUINT		uiColumnNum;
} SQL_COLUMN;

typedef struct SQL_NODE
{
	eSQLNodeTypes	eNodeType;			// Type of node this is
	FLMUINT			uiNestLevel;		// Nesting level of node - only used when
												// setting up the query
	FLMBOOL			bUsedValue;			// Used during evaluation
	FLMBOOL			bLastValue;			// Used during evaluation
	FLMBOOL			bNotted;
	SQL_NODE *		pParent;				// Parent of this query node
	SQL_NODE *		pPrevSib;			// Previous sibling of this query node
	SQL_NODE *		pNextSib;			// Next sibling of this query node
	SQL_NODE *		pFirstChild;		// First child of this query node
	SQL_NODE *		pLastChild;			// Last child of this query node
	union
	{
		SQL_OP		op;
		SQL_COLUMN	column;
		SQL_VALUE	value;
		SQL_PRED		pred;
	} nd;
} SQL_NODE;

FINLINE FLMBOOL isSQLNodeBool(
	SQL_NODE *	pNode
	)
{
	return( (pNode->eNodeType == SQL_VALUE_NODE &&
				pNode->nd.value.eValType == SQL_BOOL_VAL) ? TRUE : FALSE);
}

typedef struct SQL_PARSE_STATE
{
	SQL_NODE *				pRootNode;
	SQL_NODE *				pCurOperatorNode;
	SQL_NODE *				pLastNode;
	FLMUINT					uiNestLevel;
	FLMBOOL					bExpectingOperator;
	FLMBOOL					bExpectingLParen;
	SQL_PARSE_STATE *		pPrev;
	SQL_PARSE_STATE *		pNext;
} SQL_PARSE_STATE;

typedef struct SQL_SUBQUERY
{
	FLMUINT			uiOperandCount;
	SQL_NODE **		ppOperands;
	SQL_SUBQUERY *	pNext;
	SQL_SUBQUERY *	pPrev;
} SQL_SUBQUERY;

typedef struct SQL_DNF_NODE
{
	SQL_DNF_NODE *	pParent;
	SQL_DNF_NODE *	pFirstChild;
	SQL_DNF_NODE *	pLastChild;
	SQL_DNF_NODE *	pNextSib;
	SQL_DNF_NODE *	pPrevSib;
	SQL_NODE *		pNode;		// If NULL, bAndOp is used to tell if it is an OR or AND operator
	FLMBOOL			bAndOp;		// Only set if pNode is NULL.
} SQL_DNF_NODE;

//-------------------------------------------------------------------------
// Desc: SQLQuery class - for building up an SQL query.
//-------------------------------------------------------------------------
class SQLQuery : public F_Object
{
public:

	SQLQuery();
	
	~SQLQuery();

	FINLINE FLMBOOL expectingOperand( void)
	{
		return( !m_pCurrParseState->bExpectingOperator);
	}

	FINLINE FLMBOOL expectingOperator( void)
	{
		return( m_pCurrParseState->bExpectingOperator);
	}
	
	RCODE addOperator(
		eSQLQueryOperators	eOperator,
		FLMUINT					uiCompareRules);
		
	RCODE allocOperandNode(
		eSQLNodeTypes	eNodeType,
		SQL_NODE **		ppSQLNode);
		
	RCODE addTable(
		FLMUINT			uiTableNum,
		SQL_TABLE **	ppTable);
		
	RCODE addColumn(
		FLMUINT	uiTableNum,
		FLMUINT	uiColumnNum);
		
	RCODE addUTF8String(
		const FLMBYTE *	pszUTF8Str,
		FLMUINT				uiStrLen,
		FLMUINT				uiNumChars);
		
	RCODE addBinary(
		const FLMBYTE *	pucValue,
		FLMUINT				uiValueLen);
		
	RCODE addUINT64(
		FLMUINT64		ui64Num);
		
	RCODE addINT64(
		FLMINT64			i64Num);
		
	RCODE addUINT(
		FLMUINT			uiNum);

	RCODE addINT(
		FLMINT			iNum);
		
	FINLINE RCODE addNumber(
		FLMUINT64		ui64Num,
		FLMBOOL			bNeg)
	{
		if (!bNeg)
		{
			if (ui64Num <= (FLMUINT64)(FLM_MAX_UINT))
			{
				return( addUINT( (FLMUINT)ui64Num));
			}
			else
			{
				return( addUINT64( ui64Num));
			}
		}
		else
		{
			if (ui64Num <= (FLMUINT64)(FLM_MAX_INT))
			{
				return( addINT( (FLMINT)(-((FLMINT64)ui64Num))));
			}
			else
			{
				return( addINT64( -((FLMINT64)ui64Num)));
			}
		}
	}
		
	RCODE addBoolean(
		FLMBOOL	bValue,
		FLMBOOL	bUnknown);
		
	FINLINE FLMBOOL criteriaIsComplete( void)
	{
		// Make sure we have a completed expression
	
		if (m_pCurrParseState)
		{
			if (m_pCurrParseState->pPrev ||
				 m_pCurrParseState->uiNestLevel ||
				 (m_pCurrParseState->pLastNode &&
				  m_pCurrParseState->pLastNode->eNodeType == SQL_OPERATOR_NODE))
			{
				return( FALSE);
			}
		}
		return( TRUE);
	}
	
	RCODE getNext(
		F_Row **	ppRow);
	
	RCODE getPrev(
		F_Row **	ppRow);
		
	RCODE getFirst(
		F_Row **	ppRow);
		
	RCODE getLast(
		F_Row **	ppRow);

	RCODE evalCriteria(
		SQL_VALUE *	pSqlValue,
		F_Pool *		pPool,
		F_Row *		pRow);
		
private:

	RCODE allocParseState( void);
	
	RCODE allocValueNode(
		FLMUINT			uiValLen,
		eSQLValTypes	eValType,
		SQL_NODE **		ppSQLNode);
		
	RCODE intersectPredicates(
		SQL_PRED *				pPred,
		eSQLQueryOperators	eOperator,
		FLMUINT					uiCompareRules,
		FLMBOOL					bNotted,
		SQL_VALUE *				pValue,
		FLMBOOL *				pbAlwaysFalse,
		FLMBOOL *				pbIntersected);
		
	RCODE unionPredicates(
		SQL_PRED *				pPred,
		eSQLQueryOperators	eOperator,
		FLMUINT					uiCompareRules,
		FLMBOOL					bNotted,
		SQL_VALUE *				pValue,
		FLMBOOL *				pbUnioned);
		
	RCODE addPredicate(
		SQL_SUBQUERY *			pSubQuery,
		FLMUINT *				puiOperand,
		SQL_TABLE *				pTable,
		FLMUINT					uiColumnNum,
		eSQLQueryOperators	eOperator,
		FLMUINT					uiCompareRules,
		FLMBOOL					bNotted,
		SQL_VALUE *				pValue);
		
	RCODE convertOperandsToPredicates( void);
		
	RCODE flattenTree( void);

	RCODE convertToDNF( void);

	RCODE getPredKeys(
		SQL_PRED *	pPred,
		SQL_TABLE *	pTable);
		
	RCODE chooseBestIndex(
		SQL_TABLE *	pTable,
		FLMUINT *	puiCost);
		
	RCODE calcTableScanCost(
		SQL_TABLE *			pTable,
		FLMUINT64 *			pui64Cost,
		FSTableCursor **	ppFSTableCursor);
		
	RCODE mergeKeys(
		SQL_TABLE *	pDestTable,
		SQL_TABLE *	pSrcTable);
		
	RCODE optimizeTable(
		SQL_SUBQUERY *	pSubQuery,
		SQL_TABLE *		pTable);
		
	RCODE optimizeSubQueries( void);
	
	RCODE optimize( void);
	
	F_Pool				m_pool;
	FLMUINT				m_uiLanguage;
	SQL_PARSE_STATE *	m_pCurrParseState;
	SQL_SUBQUERY *		m_pFirstSubQuery;
	SQL_SUBQUERY *		m_pLastSubQuery;
	SQL_TABLE *			m_pFirstTable;
	SQL_TABLE *			m_pLastTable;
	FLMBOOL				m_bOptimized;
	SQL_NODE *			m_pQuery;
	F_Database *		m_pDatabase;
	F_Db *				m_pDb;
	FLMBOOL				m_bScan;
	FLMBOOL				m_bScanIndex;
	FLMUINT				m_uiIndexNum;
	FLMBOOL				m_bIndexSet;
	FLMBOOL				m_bEmpty;
	SQLQuery *			m_pNext;		
	SQLQuery *			m_pPrev;
friend class F_Db;
friend class F_Database;
};

typedef struct SQL_KEYPOS
{
	FLMBYTE	ucKey [SFLM_MAX_KEY_SIZE];
	FLMUINT	uiKeyLen;
} SQL_KEYPOS;


RCODE sqlCompare(								// sqleval.cpp
	SQL_VALUE *		pValue1,
	SQL_VALUE *		pValue2,
	FLMUINT			uiCompareRules,
	FLMUINT			uiLanguage,
	FLMINT *			piCmp);

RCODE sqlEvalArithOperator(				// sqleval.cpp
	SQL_VALUE *				pLValue,
	SQL_VALUE *				pRValue,
	eSQLQueryOperators	eOperator,
	SQL_VALUE *				pResult);

#endif	// #ifndef FLAIMODBC_H
