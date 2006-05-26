//-------------------------------------------------------------------------
// Desc:	Parse SQL
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

FSTATIC RCODE sqlCompareText(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMUINT			uiCompareRules,
	FLMBOOL			bOpIsMatch,
	FLMUINT			uiLanguage,
	FLMINT *			piResult);
	
FSTATIC RCODE sqlCompareBinary(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMINT *			piResult);
	
FSTATIC void sqlArithOpUUBitAND(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUBitOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUBitXOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpUSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

FSTATIC void sqlArithOpSUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult);

typedef void SQL_ARITH_OP(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	SQL_VALUE *		pResult);

SQL_ARITH_OP * SQL_ArithOpTable[ 
	((SQL_LAST_ARITH_OP - SQL_FIRST_ARITH_OP) + 1) * 4 ] =
{
/*	U = Unsigned		S = Signed
					U + U					U + S
						S + U					S + S */
/* BITAND */	sqlArithOpUUBitAND,		sqlArithOpUUBitAND,
						sqlArithOpUUBitAND,		sqlArithOpUUBitAND,
/* BITOR  */	sqlArithOpUUBitOR,		sqlArithOpUUBitOR,
						sqlArithOpUUBitOR,		sqlArithOpUUBitOR,
/* BITXOR */	sqlArithOpUUBitXOR,		sqlArithOpUUBitXOR,
						sqlArithOpUUBitXOR,		sqlArithOpUUBitXOR,
/* MULT   */	sqlArithOpUUMult,			sqlArithOpUSMult,
						sqlArithOpSUMult,			sqlArithOpSSMult,
/* DIV    */	sqlArithOpUUDiv,			sqlArithOpUSDiv,
						sqlArithOpSUDiv,			sqlArithOpSSDiv,
/* MOD    */	sqlArithOpUUMod,			sqlArithOpUSMod,
						sqlArithOpSUMod,			sqlArithOpSSMod,
/* PLUS   */	sqlArithOpUUPlus,			sqlArithOpUSPlus,
						sqlArithOpSUPlus,			sqlArithOpSSPlus,
/* MINUS  */	sqlArithOpUUMinus,		sqlArithOpUSMinus,
						sqlArithOpSUMinus,		sqlArithOpSSMinus
};

//-------------------------------------------------------------------------
// Desc:	Compare two entire strings.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlCompareText(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMUINT			uiCompareRules,
	FLMBOOL			bOpIsMatch,
	FLMUINT			uiLanguage,
	FLMINT *			piResult)
{
	RCODE					rc = NE_SFLM_OK;
	F_BufferIStream	bufferLStream;
	IF_PosIStream *	pLStream;
	F_BufferIStream	bufferRStream;
	IF_PosIStream *	pRStream;

	// Types must be text

	if (pLValue->eValType != SQL_UTF8_VAL || pRValue->eValType != SQL_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
		goto Exit;
	}

	// Open the streams

	if( !(pLValue->uiFlags & SQL_VAL_IS_STREAM))
	{
		if (RC_BAD( rc = bufferLStream.open( (const char *)pLValue->val.pucBuf,
											pLValue->uiDataLen)))
		{
			goto Exit;
		}

		pLStream = &bufferLStream;
	}
	else
	{
		pLStream = pLValue->val.pIStream;
	}

	if( !(pRValue->uiFlags & SQL_VAL_IS_STREAM))
	{
		if( RC_BAD( rc = bufferRStream.open( (const char *)pRValue->val.pucBuf,
											pRValue->uiDataLen)))
		{
			goto Exit;
		}
		pRStream = &bufferRStream;
	}
	else
	{
		pRStream = pRValue->val.pIStream;
	}

	if( RC_BAD( rc = f_compareUTF8Streams( 
		pLStream, 
		(bOpIsMatch && (pLValue->uiFlags & SQL_VAL_IS_CONSTANT)) ? TRUE : FALSE,
		pRStream,
		(bOpIsMatch && (pRValue->uiFlags & SQL_VAL_IS_CONSTANT)) ? TRUE : FALSE,
		uiCompareRules, uiLanguage, piResult)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Performs binary comparison on two streams - may be text or binary,
//			it really doesn't matter.  Returns XFLM_TRUE or XFLM_FALSE.
//-------------------------------------------------------------------------
FSTATIC RCODE sqlCompareBinary(
	SQL_VALUE *		pLValue,
	SQL_VALUE *		pRValue,
	FLMINT *			piResult)
{
	RCODE					rc = NE_SFLM_OK;
	F_BufferIStream	bufferLStream;
	IF_PosIStream *	pLStream;
	F_BufferIStream	bufferRStream;
	IF_PosIStream *	pRStream;
	FLMBYTE				ucLByte;
	FLMBYTE				ucRByte;
	FLMUINT				uiOffset = 0;
	FLMBOOL				bLEmpty = FALSE;

	*piResult = 0;

	// Types must be binary

	if (pLValue->eValType != SQL_BINARY_VAL ||
		 pRValue->eValType != SQL_BINARY_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
		goto Exit;
	}

	// Open the streams

	if( !(pLValue->uiFlags & SQL_VAL_IS_STREAM))
	{
		if (RC_BAD( rc = bufferLStream.open( (const char *)pLValue->val.pucBuf,
											pLValue->uiDataLen)))
		{
			goto Exit;
		}

		pLStream = &bufferLStream;
	}
	else
	{
		pLStream = pLValue->val.pIStream;
	}

	if( !(pRValue->uiFlags & SQL_VAL_IS_STREAM))
	{
		if( RC_BAD( rc = bufferRStream.open( (const char *)pRValue->val.pucBuf,
											pRValue->uiDataLen)))
		{
			goto Exit;
		}
		pRStream = &bufferRStream;
	}
	else
	{
		pRStream = pRValue->val.pIStream;
	}

	for (;;)
	{
		if (RC_BAD( rc = flmReadStorageAsBinary( 
			pLStream, &ucLByte, 1, uiOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				bLEmpty = TRUE;
			}
			else
			{
				goto Exit;
			}
		}

		if (RC_BAD( rc = flmReadStorageAsBinary( 
			pRStream, &ucRByte, 1, uiOffset, NULL)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				if( bLEmpty)
				{
					*piResult = 0;
				}
				else
				{
					*piResult = 1;
				}
			}
			goto Exit;
		}
		else if( bLEmpty)
		{
			*piResult = -1;
			goto Exit;
		}

		if( ucLByte != ucRByte)
		{
			*piResult = ucLByte < ucRByte ? -1 : 1;
			goto Exit;
		}

		uiOffset++;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Compare two values.  This routine assumes that pValue1 and pValue2
//			are non-null.
//-------------------------------------------------------------------------
RCODE sqlCompare(
	SQL_VALUE *		pValue1,
	SQL_VALUE *		pValue2,
	FLMUINT			uiCompareRules,
	FLMUINT			uiLanguage,
	FLMINT *			piCmp)
{
	RCODE		rc = NE_SFLM_OK;

	// We have already called sqlCanCompare, so no need to do it here

	switch (pValue1->eValType)
	{
		case SQL_BOOL_VAL:
			*piCmp = pValue1->val.eBool > pValue2->val.eBool
					 ? 1
					 : pValue1->val.eBool < pValue2->val.eBool
						? -1
						: 0;
			break;
		case SQL_UINT_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.uiVal > pValue2->val.uiVal
							 ? 1
							 : pValue1->val.uiVal < pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = (FLMUINT64)pValue1->val.uiVal > pValue2->val.ui64Val
							 ? 1
							 : (FLMUINT64)pValue1->val.uiVal < pValue2->val.ui64Val
								? -1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.uiVal > (FLMUINT)pValue2->val.iVal
							 							 ? 1
							 : pValue1->val.uiVal < (FLMUINT)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = pValue2->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.uiVal >
							 (FLMUINT64)pValue2->val.i64Val
							 ? 1
							 : (FLMUINT64)pValue1->val.uiVal <
								(FLMUINT64)pValue2->val.i64Val
								? -1
								: 0;
					break;
				default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_UINT64_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.ui64Val > (FLMUINT64)pValue2->val.uiVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = pValue1->val.ui64Val > pValue2->val.ui64Val
							 ? 1
							 : pValue1->val.ui64Val < pValue2->val.ui64Val
							   ? -1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.iVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = pValue2->val.i64Val < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.i64Val
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.i64Val
							   ? -1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_INT_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT)pValue1->val.iVal < pValue2->val.uiVal
							 ? -1
							 : (FLMUINT)pValue1->val.iVal > pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT64)pValue1->val.iVal < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.iVal > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue1->val.iVal < pValue2->val.iVal
							 ? -1
							 : pValue1->val.iVal > pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = (FLMINT64)pValue1->val.iVal < pValue2->val.i64Val
							 ? -1
							 : (FLMINT64)pValue1->val.iVal > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_INT64_VAL:
			switch (pValue2->eValType)
			{
				case SQL_UINT_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val <
							 (FLMUINT64)pValue2->val.uiVal
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val >
							   (FLMUINT64)pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case SQL_UINT64_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case SQL_INT_VAL:
					*piCmp = pValue1->val.i64Val < (FLMINT64)pValue2->val.iVal
							 ? -1
							 : pValue1->val.i64Val > (FLMINT64)pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case SQL_INT64_VAL:
					*piCmp = pValue1->val.i64Val < pValue2->val.i64Val
							 ? -1
							 : pValue1->val.i64Val > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
				default:
					rc = RC_SET_AND_ASSERT( NE_SFLM_Q_COMPARE_OPERAND_TYPE_MISMATCH);
					goto Exit;
			}
			break;
		case SQL_BINARY_VAL:
			if (RC_BAD( rc = sqlCompareBinary( pValue1, pValue2, piCmp)))
			{
				goto Exit;
			}
			break;
		case SQL_UTF8_VAL:
			if (RC_BAD( rc = sqlCompareText( pValue1, pValue2,
				uiCompareRules, FALSE, uiLanguage, piCmp)))
			{
				goto Exit;
			}
			break;
		default:
			break;
	}

Exit:

	return( rc);
}

//-------------------------------------------------------------------------
// Desc:	Returns a 64-bit unsigned integer
//-------------------------------------------------------------------------
FINLINE FLMUINT64 sqlGetUInt64(
	SQL_VALUE *		pValue)
{
	if (pValue->eValType == SQL_UINT_VAL)
	{
		return( (FLMUINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == SQL_UINT64_VAL)
	{
		return( pValue->val.ui64Val);
	}
	else if( pValue->eValType == SQL_INT64_VAL)
	{
		if( pValue->val.i64Val >= 0)
		{
			return( (FLMUINT64)pValue->val.i64Val);
		}
	}
	else if( pValue->eValType == SQL_INT_VAL)
	{
		if( pValue->val.iVal >= 0)
		{
			return( (FLMUINT64)pValue->val.iVal);
		}
	}
	
	flmAssert( 0);
	return( 0);
}

//-------------------------------------------------------------------------
// Desc:	Returns a 64-bit signed integer
//-------------------------------------------------------------------------
FINLINE FLMINT64 sqlGetInt64(
	SQL_VALUE *		pValue)
{
	if (pValue->eValType == SQL_INT_VAL)
	{
		return( (FLMINT64)pValue->val.iVal);
	}
	else if( pValue->eValType == SQL_INT64_VAL)
	{
		return( pValue->val.i64Val);
	}
	else if( pValue->eValType == SQL_UINT_VAL)
	{
		return( (FLMINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == SQL_UINT64_VAL)
	{
		if( pValue->val.ui64Val <= (FLMUINT64)FLM_MAX_INT64)
		{
			return( (FLMINT64)pValue->val.ui64Val);
		}
	}
		
	flmAssert( 0);
	return( 0);
}

//-------------------------------------------------------------------------
// Desc:	Performs the bit and operation
//-------------------------------------------------------------------------
FSTATIC void sqlArithOpUUBitAND(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal & pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) & sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the bit or operation
***************************************************************************/
FSTATIC void sqlArithOpUUBitOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal | pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) | sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the bit xor operation
***************************************************************************/
FSTATIC void sqlArithOpUUBitXOR(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal ^ pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) ^ sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpUUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal * pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) * sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpUSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = (FLMINT)pLValue->val.uiVal * pRValue->val.iVal;
		pResult->eValType = SQL_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			sqlGetUInt64( pLValue) * sqlGetInt64( pRValue);
		pResult->eValType = SQL_INT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpSSMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL 
									: SQL_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)(sqlGetInt64( pLValue) *
										sqlGetInt64( pRValue));

		pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL 
									: SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void sqlArithOpSUMult(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * 
			(FLMINT)pRValue->val.uiVal;
		pResult->eValType = SQL_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			(sqlGetInt64( pLValue) * sqlGetUInt64( pRValue));
		pResult->eValType = SQL_INT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpUUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal / pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue / ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpUSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal / pRValue->val.iVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  / i64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpSSDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? SQL_INT_VAL : SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue  / i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void sqlArithOpSUDiv(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.uiVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  / ui64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpUUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal % pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue  % ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpUSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal % pRValue->val.iVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  % i64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpSSMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? SQL_INT_VAL : SQL_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue % i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void sqlArithOpSUMod(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.uiVal;
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  % ui64RValue;
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = SQL_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpUUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal + pRValue->val.uiVal;
		pResult->eValType = SQL_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			sqlGetUInt64( pLValue) + sqlGetUInt64( pRValue);
		pResult->eValType = SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpUSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( (pRValue->val.iVal >= 0) || 
			 (pLValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = pLValue->val.uiVal + (FLMUINT)pRValue->val.iVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal + pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64		ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64			i64RValue = sqlGetInt64( pRValue);

		if( (i64RValue >= 0) || (ui64LValue > gv_ui64MaxSignedIntVal))
		{			pResult->val.ui64Val = ui64LValue + (FLMUINT64)i64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue + i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpSSPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal + pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = 
			sqlGetInt64( pLValue) + sqlGetInt64( pRValue);
		pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void sqlArithOpSUPlus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( (pLValue->val.iVal >= 0) ||
			 (pRValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = (FLMUINT)pLValue->val.iVal + pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal + (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( (i64LValue >= 0) || (ui64RValue > gv_ui64MaxSignedIntVal))
		{
			pResult->val.ui64Val = (FLMUINT64)i64LValue + ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue + (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpUUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pLValue->val.uiVal >= pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.uiVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)(pLValue->val.uiVal - pRValue->val.uiVal);
			pResult->eValType = SQL_INT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64LValue >= ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue - ui64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)(ui64LValue - ui64RValue);
			pResult->eValType = SQL_INT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpUSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{	
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal < 0) 
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = sqlGetUInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( i64RValue < 0)
		{
			pResult->val.ui64Val = ui64LValue - i64RValue;
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpSSMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if(( pLValue->val.iVal > 0) && ( pRValue->val.iVal < 0))
		{
			pResult->val.uiVal = (FLMUINT)(pLValue->val.iVal - pRValue->val.iVal);
			pResult->eValType = SQL_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMINT64		i64RValue = sqlGetInt64( pRValue);

		if( (i64LValue > 0) && (i64RValue < 0))
		{
			pResult->val.ui64Val = (FLMUINT64)( i64LValue - i64RValue);
			pResult->eValType = SQL_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void sqlArithOpSUMinus(
	SQL_VALUE *	pLValue,
	SQL_VALUE *	pRValue,
	SQL_VALUE *	pResult)
{
	if (isSQLValNativeNum( pLValue->eValType) && 
		 isSQLValNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal > gv_uiMaxSignedIntVal)
		{
			pResult->val.iVal = (pLValue->val.iVal - gv_uiMaxSignedIntVal) - 
				(FLMINT)(pRValue->val.uiVal - gv_uiMaxSignedIntVal);
			pResult->eValType = SQL_INT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? SQL_INT_VAL : SQL_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = sqlGetInt64( pLValue);
		FLMUINT64	ui64RValue = sqlGetUInt64( pRValue);

		if( ui64RValue > gv_ui64MaxSignedIntVal)
		{
			pResult->val.i64Val = (i64LValue - gv_ui64MaxSignedIntVal) -
				(FLMINT64)(ui64RValue - gv_ui64MaxSignedIntVal);
			pResult->eValType = SQL_INT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? SQL_INT64_VAL : SQL_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:	Do an arithmetic operator.
***************************************************************************/
RCODE sqlEvalArithOperator(
	SQL_VALUE *				pLValue,
	SQL_VALUE *				pRValue,
	eSQLQueryOperators	eOperator,
	SQL_VALUE *				pResult)
{
	RCODE					rc = NE_SFLM_OK;
	SQL_ARITH_OP *		fnOp;
	FLMUINT				uiOffset = 0;

	if (!isSQLArithOp( eOperator))
	{
		rc = RC_SET( NE_SFLM_Q_INVALID_OPERATOR);
		goto Exit;
	}

	if (pLValue->eValType == SQL_MISSING_VAL ||
		 pRValue->eValType == SQL_MISSING_VAL)
	{
		pResult->eValType = SQL_MISSING_VAL;
		goto Exit;
	}

	if (isSQLValUnsigned( pLValue->eValType))
	{
		if (isSQLValUnsigned( pRValue->eValType))
		{
			uiOffset = 0;
		}
		else if (isSQLValSigned( pRValue->eValType))
		{
			uiOffset = 1;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	else if (isSQLValSigned( pLValue->eValType))
	{
		if (isSQLValUnsigned( pRValue->eValType))
		{
			uiOffset = 2;
		}
		else if (isSQLValSigned( pRValue->eValType))
		{
			uiOffset = 3;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_SFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	fnOp = SQL_ArithOpTable[ ((((FLMUINT)eOperator) - 
					SQL_FIRST_ARITH_OP) * 4) + uiOffset];
	fnOp( pLValue, pRValue, pResult);

Exit:

	return( rc);
}

