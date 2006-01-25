//------------------------------------------------------------------------------
// Desc:	Contains the methods for doing evaluation of query expressions.
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
// $Id: fqeval.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"
#include "fquery.h"

FSTATIC RCODE fqApproxCompare(
	FQVALUE *			pLValue,
	FQVALUE *			pRValue,
	FLMINT *				piResult);

FSTATIC RCODE fqCompareBinary(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMINT *					piResult);

FSTATIC RCODE fqCompareText(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMUINT					uiCompareRules,
	FLMBOOL					bOpIsMatch,
	FLMUINT					uiLanguage,
	FLMINT *					piResult);

FSTATIC void fqOpUUBitAND(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUBitOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUBitXOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpUSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

FSTATIC void fqOpSUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult);

typedef void FQ_OPERATION(
	FQVALUE *		pLValue,
	FQVALUE *		pRValue,
	FQVALUE *		pResult);

FQ_OPERATION * FQ_ArithOpTable[ 
	((XFLM_LAST_ARITH_OP - XFLM_FIRST_ARITH_OP) + 1) * 4 ] =
{
/*	U = Unsigned		S = Signed
					U + U					U + S
						S + U					S + S */
/* BITAND */	fqOpUUBitAND,		fqOpUUBitAND,
						fqOpUUBitAND,		fqOpUUBitAND,
/* BITOR  */	fqOpUUBitOR,		fqOpUUBitOR,
						fqOpUUBitOR,		fqOpUUBitOR,
/* BITXOR */	fqOpUUBitXOR,		fqOpUUBitXOR,
						fqOpUUBitXOR,		fqOpUUBitXOR,
/* MULT   */	fqOpUUMult,			fqOpUSMult,
						fqOpSUMult,			fqOpSSMult,
/* DIV    */	fqOpUUDiv,			fqOpUSDiv,
						fqOpSUDiv,			fqOpSSDiv,
/* MOD    */	fqOpUUMod,			fqOpUSMod,
						fqOpSUMod,			fqOpSSMod,
/* PLUS   */	fqOpUUPlus,			fqOpUSPlus,
						fqOpSUPlus,			fqOpSSPlus,
/* MINUS  */	fqOpUUMinus,		fqOpUSMinus,
						fqOpSUMinus,		fqOpSSMinus
};

/***************************************************************************
Desc:		Returns a 64-bit unsigned integer
***************************************************************************/
FINLINE FLMUINT64 fqGetUInt64(
	FQVALUE *		pValue)
{
	if (pValue->eValType == XFLM_UINT_VAL)
	{
		return( (FLMUINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == XFLM_UINT64_VAL)
	{
		return( pValue->val.ui64Val);
	}
	else if( pValue->eValType == XFLM_INT64_VAL)
	{
		if( pValue->val.i64Val >= 0)
		{
			return( (FLMUINT64)pValue->val.i64Val);
		}
	}
	else if( pValue->eValType == XFLM_INT_VAL)
	{
		if( pValue->val.iVal >= 0)
		{
			return( (FLMUINT64)pValue->val.iVal);
		}
	}
	
	flmAssert( 0);
	return( 0);
}

/***************************************************************************
Desc:		Returns a 64-bit signed integer
***************************************************************************/
FINLINE FLMINT64 fqGetInt64(
	FQVALUE *		pValue)
{
	if (pValue->eValType == XFLM_INT_VAL)
	{
		return( (FLMINT64)pValue->val.iVal);
	}
	else if( pValue->eValType == XFLM_INT64_VAL)
	{
		return( pValue->val.i64Val);
	}
	else if( pValue->eValType == XFLM_UINT_VAL)
	{
		return( (FLMINT64)pValue->val.uiVal);
	}
	else if( pValue->eValType == XFLM_UINT64_VAL)
	{
		if( pValue->val.ui64Val <= (FLMUINT64)FLM_MAX_INT64)
		{
			return( (FLMINT64)pValue->val.ui64Val);
		}
	}
		
	flmAssert( 0);
	return( 0);
}

/***************************************************************************
Desc:		Performs the bit and operation
***************************************************************************/
FSTATIC void fqOpUUBitAND(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal & pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) & fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the bit or operation
***************************************************************************/
FSTATIC void fqOpUUBitOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal | pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) | fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the bit xor operation
***************************************************************************/
FSTATIC void fqOpUUBitXOR(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal ^ pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) ^ fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpUUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal * pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) * fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpUSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = (FLMINT)pLValue->val.uiVal * pRValue->val.iVal;
		pResult->eValType = XFLM_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			fqGetUInt64( pLValue) * fqGetInt64( pRValue);
		pResult->eValType = XFLM_INT64_VAL;
	}
}
	
/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpSSMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL 
									: XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)(fqGetInt64( pLValue) *
										fqGetInt64( pRValue));

		pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL 
									: XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the multiply operation
***************************************************************************/
FSTATIC void fqOpSUMult(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal * 
			(FLMINT)pRValue->val.uiVal;
		pResult->eValType = XFLM_INT_VAL;
	}
	else
	{
		pResult->val.i64Val = (FLMINT64)
			(fqGetInt64( pLValue) * fqGetUInt64( pRValue));
		pResult->eValType = XFLM_INT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpUUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal / pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue / ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpUSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal / pRValue->val.iVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  / i64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpSSDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue  / i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the divide operation
***************************************************************************/
FSTATIC void fqOpSUDiv(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal / pRValue->val.uiVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  / ui64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// Divide by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpUUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal % pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue  % ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpUSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.uiVal % pRValue->val.iVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = ui64LValue  % i64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpSSMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
										? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue)
		{
			pResult->val.i64Val = i64LValue % i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
										? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs the modulo operation
***************************************************************************/
FSTATIC void fqOpSUMod(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal)
		{
			pResult->val.iVal = pLValue->val.iVal % pRValue->val.uiVal;
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue)
		{
			pResult->val.i64Val = i64LValue  % ui64RValue;
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.uiVal = 0;				// MOD by ZERO case.
			pResult->eValType = XFLM_MISSING_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpUUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.uiVal = pLValue->val.uiVal + pRValue->val.uiVal;
		pResult->eValType = XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.ui64Val = 
			fqGetUInt64( pLValue) + fqGetUInt64( pRValue);
		pResult->eValType = XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpUSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( (pRValue->val.iVal >= 0) || 
			 (pLValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = pLValue->val.uiVal + (FLMUINT)pRValue->val.iVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal + pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64		ui64LValue = fqGetUInt64( pLValue);
		FLMINT64			i64RValue = fqGetInt64( pRValue);

		if( (i64RValue >= 0) || (ui64LValue > gv_ui64MaxSignedIntVal))
		{
			pResult->val.ui64Val = ui64LValue + (FLMUINT64)i64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue + i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpSSPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		pResult->val.iVal = pLValue->val.iVal + pRValue->val.iVal;
		pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
	}
	else
	{
		pResult->val.i64Val = 
			fqGetInt64( pLValue) + fqGetInt64( pRValue);
		pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
	}
}

/***************************************************************************
Desc:		Performs an addition operation
***************************************************************************/
FSTATIC void fqOpSUPlus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( (pLValue->val.iVal >= 0) ||
			 (pRValue->val.uiVal > gv_uiMaxSignedIntVal))
		{
			pResult->val.uiVal = (FLMUINT)pLValue->val.iVal + pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal + (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( (i64LValue >= 0) || (ui64RValue > gv_ui64MaxSignedIntVal))
		{
			pResult->val.ui64Val = (FLMUINT64)i64LValue + ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue + (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpUUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pLValue->val.uiVal >= pRValue->val.uiVal)
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.uiVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)(pLValue->val.uiVal - pRValue->val.uiVal);
			pResult->eValType = XFLM_INT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64LValue >= ui64RValue)
		{
			pResult->val.ui64Val = ui64LValue - ui64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)(ui64LValue - ui64RValue);
			pResult->eValType = XFLM_INT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpUSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{	
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.iVal < 0) 
		{
			pResult->val.uiVal = pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = (FLMINT)pLValue->val.uiVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMUINT64	ui64LValue = fqGetUInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( i64RValue < 0)
		{
			pResult->val.ui64Val = ui64LValue - i64RValue;
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = (FLMINT64)ui64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}
	
/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpSSMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if(( pLValue->val.iVal > 0) && ( pRValue->val.iVal < 0))
		{
			pResult->val.uiVal = (FLMUINT)(pLValue->val.iVal - pRValue->val.iVal);
			pResult->eValType = XFLM_UINT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - pRValue->val.iVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMINT64		i64RValue = fqGetInt64( pRValue);

		if( (i64LValue > 0) && (i64RValue < 0))
		{
			pResult->val.ui64Val = (FLMUINT64)( i64LValue - i64RValue);
			pResult->eValType = XFLM_UINT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - i64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:		Performs a subtraction operation
***************************************************************************/
FSTATIC void fqOpSUMinus(
	FQVALUE *	pLValue,
	FQVALUE *	pRValue,
	FQVALUE *	pResult)
{
	if( isNativeNum( pLValue->eValType) && 
		isNativeNum( pRValue->eValType))
	{
		if( pRValue->val.uiVal > gv_uiMaxSignedIntVal)
		{
			pResult->val.iVal = (pLValue->val.iVal - gv_uiMaxSignedIntVal) - 
				(FLMINT)(pRValue->val.uiVal - gv_uiMaxSignedIntVal);
			pResult->eValType = XFLM_INT_VAL;
		}
		else
		{
			pResult->val.iVal = pLValue->val.iVal - (FLMINT)pRValue->val.uiVal;
			pResult->eValType = (pResult->val.iVal < 0) 
									? XFLM_INT_VAL : XFLM_UINT_VAL;
		}
	}
	else
	{
		FLMINT64		i64LValue = fqGetInt64( pLValue);
		FLMUINT64	ui64RValue = fqGetUInt64( pRValue);

		if( ui64RValue > gv_ui64MaxSignedIntVal)
		{
			pResult->val.i64Val = (i64LValue - gv_ui64MaxSignedIntVal) -
				(FLMINT64)(ui64RValue - gv_ui64MaxSignedIntVal);
			pResult->eValType = XFLM_INT64_VAL;
		}
		else
		{
			pResult->val.i64Val = i64LValue - (FLMINT64)ui64RValue;
			pResult->eValType = (pResult->val.i64Val < 0) 
									? XFLM_INT64_VAL : XFLM_UINT64_VAL;
		}
	}
}

/***************************************************************************
Desc:  	Compare two entire strings.
****************************************************************************/
RCODE fqCompareCollStreams(
	F_CollIStream *	pLStream,
	F_CollIStream *	pRStream,
	FLMBOOL				bOpIsMatch,
	FLMUINT				uiLanguage,
	FLMINT *				piResult)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT16			ui16RCol;
	FLMUINT16			ui16LCol;
	FLMUINT16			ui16RSubCol;
	FLMUINT16			ui16LSubCol;
	FLMBYTE				ucRCase;
	FLMBYTE				ucLCase;
	F_CollStreamPos	savedRPos;
	F_CollStreamPos	savedLPos;
	F_CollStreamPos	startLPos;
	FLMUNICODE			uLChar = 0;
	FLMBOOL				bLCharIsWild = FALSE;
	FLMUNICODE			uRChar = 0;
	FLMBOOL				bRCharIsWild = FALSE;
	FLMBOOL				bPrevLWasWild = FALSE;
	FLMBOOL				bPrevRWasWild = FALSE;
	FLMBOOL				bAllowTwoIntoOne;

	// If we are doing a "match" operation, we don't want two
	// character sequences like Ch, ae, etc. turned into a single
	// a single collation, because then matches that involve wildcards
	// like "aetna == a*" would not match properly.
	// When not doing a match operation, we WANT two character sequences
	// turned into a single collation value so that we can know if
	// something is > or <.  When doing match operations, all we care
	// about is if they are equal or not, so there is no need to look
	// at double character collation properties.

	bAllowTwoIntoOne = bOpIsMatch ? FALSE : TRUE;

	for( ;;)
	{
GetNextLChar:

		if( bLCharIsWild)
		{
			bPrevLWasWild = TRUE;
		}

		pLStream->getCurrPosition( &startLPos);
		if( RC_BAD( rc = pLStream->read( 
			bAllowTwoIntoOne,
			&uLChar, &bLCharIsWild, &ui16LCol, &ui16LSubCol, &ucLCase)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;

				// If the last character was a wildcard, we have a match!

				if( bPrevLWasWild)
				{
					*piResult = 0;
					goto Exit;
				}

				for( ;;)
				{
					if( RC_BAD( rc = pRStream->read( 
						bAllowTwoIntoOne,
						&uRChar, &bRCharIsWild, &ui16RCol, &ui16RSubCol, &ucRCase)))
					{
						if( rc == NE_XFLM_EOF_HIT)
						{
							rc = NE_XFLM_OK;
							*piResult = 0;
						}

						goto Exit;
					}

					// Break out when we hit a non-wild character

					if( !bRCharIsWild)
					{
						break;
					}
				}

				*piResult = -1;
			}

			goto Exit;
		}

		if( bLCharIsWild)
		{
			// Consume multiple wildcards

			if( bPrevLWasWild)
			{
				goto GetNextLChar;
			}

			// See if we match anywhere on the remaining right string

			for( ;;)
			{
				pRStream->getCurrPosition( &savedRPos);
				pLStream->getCurrPosition( &savedLPos);

				if( RC_BAD( rc = fqCompareCollStreams( pLStream, pRStream,
					bOpIsMatch, uiLanguage, piResult)))
				{
					goto Exit;
				}

				if( !(*piResult))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRStream->positionTo( &savedRPos)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRStream->read( 
					bAllowTwoIntoOne, 
					NULL, NULL, NULL, NULL, NULL)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						break;
					}
					goto Exit;
				}

				if( RC_BAD( rc = pLStream->positionTo( &savedLPos)))
				{
					goto Exit;
				}
			}

			*piResult = 1;
			goto Exit;
		}

GetNextRChar:

		if( bRCharIsWild)
		{
			bPrevRWasWild = TRUE;
		}

		if( RC_BAD( rc = pRStream->read( 
			bAllowTwoIntoOne, 
			&uRChar, &bRCharIsWild, &ui16RCol, &ui16RSubCol, &ucRCase)))
		{
			if( rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;

				// If the last character was a wildcard, we have a match!

				if( bPrevRWasWild)
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

		if( bRCharIsWild)
		{
			if( bPrevRWasWild)
			{
				goto GetNextRChar;
			}

			// See if we match anywhere on the remaining left string

			if( RC_BAD( rc = pLStream->positionTo( &startLPos)))
			{
				goto Exit;
			}

			for( ;;)
			{
				pLStream->getCurrPosition( &savedLPos);
				pRStream->getCurrPosition( &savedRPos);

				if( RC_BAD( rc = fqCompareCollStreams( pLStream, pRStream,
					bOpIsMatch, uiLanguage, piResult)))
				{
					goto Exit;
				}

				if( !(*piResult))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRStream->positionTo( &savedRPos)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pLStream->positionTo( &savedLPos)))
				{
					goto Exit;
				}

				// Skip the character we just processed

				if( RC_BAD( rc = pLStream->read( 
					bAllowTwoIntoOne,
					NULL, NULL, NULL, NULL, NULL)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						break;
					}
					goto Exit;
				}
			}

			*piResult = -1;
			goto Exit;
		}

		if( ui16LCol != ui16RCol)
		{
			*piResult = ui16LCol < ui16RCol ? -1 : 1;
			goto Exit;
		}
		else if( ui16LSubCol != ui16RSubCol)
		{
			*piResult = ui16LSubCol < ui16RSubCol ? -1 : 1;
			goto Exit;
		}
		else if( ucLCase != ucRCase) 
		{
			// NOTE: If we are doing a case insensitive comparison,
			// ucLCase and ucRCase should be equal (both will have been
			// set to zero
			
			*piResult = ucLCase < ucRCase ? -1 : 1;
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:  	Compare two entire strings.
****************************************************************************/
FSTATIC RCODE fqCompareText(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMUINT					uiCompareRules,
	FLMBOOL					bOpIsMatch,
	FLMUINT					uiLanguage,
	FLMINT *					piResult)
{
	RCODE								rc = NE_XFLM_OK;
	F_BufferIStream				bufferLStream;
	IF_PosIStream *				pLStream;
	F_BufferIStream				bufferRStream;
	IF_PosIStream *				pRStream;
	F_CollIStream					lStream;
	F_CollIStream					rStream;

	// Types must be text

	if (pLValue->eValType != XFLM_UTF8_VAL || 
		pRValue->eValType != XFLM_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if( !(pLValue->uiFlags & VAL_IS_STREAM))
	{
		if (RC_BAD( rc = bufferLStream.open( pLValue->val.pucBuf,
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

	if( !(pRValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = bufferRStream.open( pRValue->val.pucBuf,
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

	if (pOpComparer)
	{
		rc = pOpComparer->compare( pLStream, pRStream, piResult);
		goto Exit;
	}

	// Open up the collated streams

	if (RC_BAD( rc = lStream.open( pLStream, FALSE, uiLanguage, uiCompareRules,
		(bOpIsMatch && (pLValue->uiFlags & VAL_IS_CONSTANT)) ? TRUE : FALSE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = rStream.open( pRStream, FALSE, uiLanguage, uiCompareRules,
		(bOpIsMatch && (pRValue->uiFlags & VAL_IS_CONSTANT)) ? TRUE : FALSE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = fqCompareCollStreams(  &lStream, &rStream,
		bOpIsMatch, uiLanguage, piResult)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Approximate compare - only works for strings right now.
****************************************************************************/
FSTATIC RCODE fqApproxCompare(
	FQVALUE *			pLValue,
	FQVALUE *			pRValue,
	FLMINT *				piResult)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiLMeta;
	FLMUINT				uiRMeta;
	FLMUINT64			ui64StartPos;
	F_BufferIStream	bufferLStream;
	IF_PosIStream *	pLStream;
	F_BufferIStream	bufferRStream;
	IF_PosIStream *	pRStream;

	// Types must be text

	if (pLValue->eValType != XFLM_UTF8_VAL ||
		 pRValue->eValType != XFLM_UTF8_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if (!(pLValue->uiFlags & VAL_IS_STREAM))
	{
		if (RC_BAD( rc = bufferLStream.open( pLValue->val.pucBuf,
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

	if (!(pRValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = bufferRStream.open( pRValue->val.pucBuf,
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

	if ((pLValue->uiFlags & VAL_IS_CONSTANT) ||
		 !(pRValue->uiFlags & VAL_IS_CONSTANT))
	{
		for( ;;)
		{
			if( RC_BAD( rc = flmGetNextMetaphone( pLStream, &uiLMeta)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					*piResult = 0;
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}

			ui64StartPos = pRStream->getCurrPosition();

			for( ;;)
			{
				if( RC_BAD( rc = flmGetNextMetaphone( pRStream, &uiRMeta)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						*piResult = -1;
					}

					goto Exit;
				}

				if( uiLMeta == uiRMeta)
				{
					break;
				}

			}

			if( RC_BAD( rc = pRStream->positionTo( ui64StartPos)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		for( ;;)
		{
			if( RC_BAD( rc = flmGetNextMetaphone( pRStream, &uiRMeta)))
			{
				if( rc == NE_XFLM_EOF_HIT)
				{
					*piResult = 0;
					rc = NE_XFLM_OK;
				}
				goto Exit;
			}

			ui64StartPos = pLStream->getCurrPosition();

			for( ;;)
			{
				if( RC_BAD( rc = flmGetNextMetaphone( pLStream, &uiLMeta)))
				{
					if( rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						*piResult = 1;
					}

					goto Exit;
				}

				if( uiLMeta == uiRMeta)
				{
					break;
				}

			}

			if( RC_BAD( rc = pLStream->positionTo( ui64StartPos)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}
	
/***************************************************************************
Desc:	Performs binary comparison on two streams - may be text or binary,
		it really doesn't matter.  Returns XFLM_TRUE or XFLM_FALSE.
***************************************************************************/
FSTATIC RCODE fqCompareBinary(
	IF_OperandComparer *	pOpComparer,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	FLMINT *					piResult)
{
	RCODE					rc = NE_XFLM_OK;
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

	if ( pLValue->eValType != XFLM_BINARY_VAL ||
		  pRValue->eValType != XFLM_BINARY_VAL)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Open the streams

	if( !(pLValue->uiFlags & VAL_IS_STREAM))
	{
		if (RC_BAD( rc = bufferLStream.open( pLValue->val.pucBuf,
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

	if( !(pRValue->uiFlags & VAL_IS_STREAM))
	{
		if( RC_BAD( rc = bufferRStream.open( pRValue->val.pucBuf,
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

	if (pOpComparer)
	{
		rc = pOpComparer->compare( pLStream, pRStream, piResult);
		goto Exit;
	}

	for (;;)
	{
		if (RC_BAD( rc = flmReadStorageAsBinary( 
			pLStream, &ucLByte, 1, uiOffset, NULL)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
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
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
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

/***************************************************************************
Desc:	Compare two values.  This routine assumes that pValue1 and pValue2
		are non-null.
***************************************************************************/
RCODE fqCompare(
	FQVALUE *				pValue1,
	FQVALUE *				pValue2,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FLMUINT					uiLanguage,
	FLMINT *					piCmp)
{
	RCODE		rc = NE_XFLM_OK;

	// We have already called fqCanCompare, so no need to do it here

	switch (pValue1->eValType)
	{
		case XFLM_BOOL_VAL:
			*piCmp = pValue1->val.eBool > pValue2->val.eBool
					 ? 1
					 : pValue1->val.eBool < pValue2->val.eBool
						? -1
						: 0;
			break;
		case XFLM_UINT_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.uiVal > pValue2->val.uiVal
							 ? 1
							 : pValue1->val.uiVal < pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = (FLMUINT64)pValue1->val.uiVal > pValue2->val.ui64Val
							 ? 1
							 : (FLMUINT64)pValue1->val.uiVal < pValue2->val.ui64Val
								? -1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.uiVal > (FLMUINT)pValue2->val.iVal
							 							 ? 1
							 : pValue1->val.uiVal < (FLMUINT)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case XFLM_INT64_VAL:
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
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_UINT64_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.ui64Val > (FLMUINT64)pValue2->val.uiVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.uiVal
							   ? -1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = pValue1->val.ui64Val > pValue2->val.ui64Val
							 ? 1
							 : pValue1->val.ui64Val < pValue2->val.ui64Val
							   ? -1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue2->val.iVal < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.iVal
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.iVal
							   ? -1
								: 0;
					break;
				case XFLM_INT64_VAL:
					*piCmp = pValue2->val.i64Val < 0 ||
							 pValue1->val.ui64Val > (FLMUINT64)pValue2->val.i64Val
							 ? 1
							 : pValue1->val.ui64Val < (FLMUINT64)pValue2->val.i64Val
							   ? -1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_INT_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT)pValue1->val.iVal < pValue2->val.uiVal
							 ? -1
							 : (FLMUINT)pValue1->val.iVal > pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = pValue1->val.iVal < 0 ||
							 (FLMUINT64)pValue1->val.iVal < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.iVal > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue1->val.iVal < pValue2->val.iVal
							 ? -1
							 : pValue1->val.iVal > pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case XFLM_INT64_VAL:
					*piCmp = (FLMINT64)pValue1->val.iVal < pValue2->val.i64Val
							 ? -1
							 : (FLMINT64)pValue1->val.iVal > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
            default:
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_INT64_VAL:
			switch (pValue2->eValType)
			{
				case XFLM_UINT_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val <
							 (FLMUINT64)pValue2->val.uiVal
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val >
							   (FLMUINT64)pValue2->val.uiVal
							   ? 1
								: 0;
					break;
				case XFLM_UINT64_VAL:
					*piCmp = pValue1->val.i64Val < 0 ||
							 (FLMUINT64)pValue1->val.i64Val < pValue2->val.ui64Val
							 ? -1
							 : (FLMUINT64)pValue1->val.i64Val > pValue2->val.ui64Val
							   ? 1
								: 0;
					break;
				case XFLM_INT_VAL:
					*piCmp = pValue1->val.i64Val < (FLMINT64)pValue2->val.iVal
							 ? -1
							 : pValue1->val.i64Val > (FLMINT64)pValue2->val.iVal
							   ? 1
								: 0;
					break;
				case XFLM_INT64_VAL:
					*piCmp = pValue1->val.i64Val < pValue2->val.i64Val
							 ? -1
							 : pValue1->val.i64Val > pValue2->val.i64Val
							   ? 1
								: 0;
					break;
				default:
					rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
					goto Exit;
			}
			break;
		case XFLM_BINARY_VAL:
			if (RC_BAD( rc = fqCompareBinary( pOpComparer, pValue1,
												pValue2, piCmp)))
			{
				goto Exit;
			}
			break;
		case XFLM_UTF8_VAL:
			if (RC_BAD( rc = fqCompareText( pOpComparer,
				pValue1, pValue2,
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

/***************************************************************************
Desc:	Do a comparison operator.
***************************************************************************/
RCODE fqCompareOperands(
	FLMUINT					uiLanguage,
	FQVALUE *				pLValue,
	FQVALUE *				pRValue,
	eQueryOperators		eOperator,
	FLMUINT					uiCompareRules,
	IF_OperandComparer *	pOpComparer,
	FLMBOOL					bNotted,
	XFlmBoolType *			peBool)
{
	RCODE			rc = NE_XFLM_OK;
	FLMINT		iCmp;

	if (!pLValue || pLValue->eValType == XFLM_MISSING_VAL ||
		 !pRValue || pRValue->eValType == XFLM_MISSING_VAL ||
		 !fqCanCompare( pLValue, pRValue))
	{
		*peBool = (bNotted ? XFLM_TRUE : XFLM_FALSE);
	}

	// At this point, both operands are known to be present and are of
	// types that can be compared.  The comparison
	// will therefore be performed according to the
	// operator specified.
	
	else
	{
		switch (eOperator)
		{
			case XFLM_EQ_OP:
			case XFLM_NE_OP:
				if (pLValue->eValType == XFLM_UTF8_VAL ||
					 pRValue->eValType == XFLM_UTF8_VAL)
				{
					if (RC_BAD( rc = fqCompareText( pOpComparer, pLValue, pRValue,
						uiCompareRules, TRUE, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
						uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
					{
						goto Exit;
					}
				}
				if (eOperator == XFLM_EQ_OP)
				{
					*peBool = (iCmp == 0 ? XFLM_TRUE : XFLM_FALSE);
				}
				else
				{
					*peBool = (iCmp != 0 ? XFLM_TRUE : XFLM_FALSE);
				}
				break;

			case XFLM_APPROX_EQ_OP:
				if (RC_BAD( rc = fqApproxCompare( pLValue, pRValue, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp == 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_LT_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp < 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_LE_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp <= 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_GT_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp > 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			case XFLM_GE_OP:
				if (RC_BAD( rc = fqCompare( pLValue, pRValue, 
					uiCompareRules, pOpComparer, uiLanguage, &iCmp)))
				{
					goto Exit;
				}
				*peBool = (iCmp >= 0 ? XFLM_TRUE : XFLM_FALSE);
				break;

			default:
				*peBool = XFLM_UNKNOWN;
				rc = RC_SET_AND_ASSERT( NE_XFLM_QUERY_SYNTAX);
				goto Exit;
		}
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	Do an arithmetic operator.
***************************************************************************/
RCODE fqArithmeticOperator(
	FQVALUE *			pLValue,
	FQVALUE *			pRValue,
	eQueryOperators	eOperator,
	FQVALUE *			pResult)
{
	RCODE					rc = NE_XFLM_OK;
	FQ_OPERATION *		fnOp;
	FLMUINT				uiOffset = 0;

	if( !isArithOp( eOperator))
	{
		rc = RC_SET( NE_XFLM_SYNTAX);
		goto Exit;
	}

	if (pLValue->eValType == XFLM_MISSING_VAL ||
		 pRValue->eValType == XFLM_MISSING_VAL)
	{
		pResult->eValType = XFLM_MISSING_VAL;
		goto Exit;
	}

	if( isUnsigned( pLValue))
	{
		if( isUnsigned( pRValue))
		{
			uiOffset = 0;
		}
		else if( isSigned( pRValue))
		{
			uiOffset = 1;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}
	else if( isSigned( pLValue))
	{
		if( isUnsigned( pRValue))
		{
			uiOffset = 2;
		}
		else if( isSigned( pRValue))
		{
			uiOffset = 3;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	fnOp = FQ_ArithOpTable[ ((((FLMUINT)eOperator) - 
					XFLM_FIRST_ARITH_OP) * 4) + uiOffset];
	fnOp( pLValue, pRValue, pResult);

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_CollIStream::read(
	FLMBOOL			bAllowTwoIntoOne,
	FLMUNICODE *	puChar,
	FLMBOOL *		pbCharIsWild,
	FLMUINT16 *		pui16Col,
	FLMUINT16 *		pui16SubCol,
	FLMBYTE *		pucCase)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE			uChar;
	FLMUINT16			ui16WpChar;
	FLMUINT16			ui16NextWpChar;
	FLMUINT16			ui16Col;
	FLMUINT16			ui16SubCol;
	FLMBOOL				bTwoIntoOne;
	FLMBYTE				ucCase;
	FLMBOOL				bAsian;
	FLMBOOL				bLastCharWasSpace = FALSE;
	FLMUINT64			ui64AfterLastSpacePos = 0;
	FLMUINT64			ui64CurrCharPos = 0;

	if (pbCharIsWild)
	{
		*pbCharIsWild = FALSE;
	}

	// Is this a double-byte (Asian) character set?

	bAsian = (m_uiLanguage >= FIRST_DBCS_LANG && m_uiLanguage <= LAST_DBCS_LANG)
				? TRUE
				: FALSE;

	// Get the next character from the stream

GetNextChar:

	ui16WpChar = 0;
	ui16NextWpChar = 0;
	ui16Col = 0;
	ui16SubCol = 0;
	bTwoIntoOne = FALSE;
	ucCase = 0;

	if (m_uNextChar)
	{
		uChar = m_uNextChar;
		m_uNextChar = 0;
	}
	else
	{
		ui64CurrCharPos = m_pIStream->getCurrPosition();
		if( RC_BAD( rc = readCharFromStream( &uChar)))
		{
			if (rc != NE_XFLM_EOF_HIT)
			{
				goto Exit;
			}
			
			// If we were skipping spaces, we need to
			// process a single space character, unless we are
			// ignoring trailing white space.
			
			if (bLastCharWasSpace &&
				 !(m_uiCompareRules & XFLM_COMP_IGNORE_TRAILING_SPACE))
			{
				// bLastCharWasSpace flag can only be TRUE if either
				// XFLM_COMP_IGNORE_TRAILING_SPACE is set or
				// XFLM_COMP_COMPRESS_WHITESPACE is set.
				
				flmAssert( m_uiCompareRules & XFLM_COMP_COMPRESS_WHITESPACE);
				uChar = ASCII_SPACE;
				rc = NE_XFLM_OK;
				goto Process_Char;
			}
			goto Exit;
		}
	}

	if ((uChar = flmConvertChar( uChar, m_uiCompareRules)) == 0)
	{
		goto GetNextChar;
	}

	// Deal with spaces

	if (uChar == ASCII_SPACE)
	{
		if (m_uiCompareRules & XFLM_COMP_COMPRESS_WHITESPACE)
		{
			bLastCharWasSpace = TRUE;
			ui64AfterLastSpacePos = m_pIStream->getCurrPosition();
			goto GetNextChar;
		}
		else if (m_uiCompareRules & XFLM_COMP_IGNORE_TRAILING_SPACE)
		{
			if (!bLastCharWasSpace)
			{
				bLastCharWasSpace = TRUE;
				
				// Save where we are at so that if this doesn't turn out
				// to be trailing spaces, we can restore this position.
				
				ui64AfterLastSpacePos = m_pIStream->getCurrPosition();
			}
			goto GetNextChar;
		}
	}
	else
	{
		if (m_uiCompareRules & XFLM_COMP_IGNORE_LEADING_SPACE)
		{
			m_ui64EndOfLeadingSpacesPos = ui64CurrCharPos;
			m_uiCompareRules &= (~(XFLM_COMP_IGNORE_LEADING_SPACE));
		}
		
		// If the last character was a space, we need to process it.
		
		if (bLastCharWasSpace)
		{
			
			// Position back to after the last space, and process a space
			// character.
			
			if (RC_BAD( rc = m_pIStream->positionTo( ui64AfterLastSpacePos)))
			{
				goto Exit;
			}
			
			uChar = ASCII_SPACE;
			bLastCharWasSpace = FALSE;
		}
		else if (uChar == ASCII_BACKSLASH)
		{
			// If wildcards are allowed, the backslash should be treated
			// as an escape character, and the next character is the one
			// we want.  Otherwise, it should be treated as
			// the actual character we want returned.
			
			if (m_bMayHaveWildCards)
			{
			
				// Got a backslash.  Means the next character is to be taken
				// no matter what because it is escaped.
			
				if (RC_BAD( rc = readCharFromStream( &uChar)))
				{
					if (rc != NE_XFLM_EOF_HIT)
					{
						goto Exit;
					}
					rc = NE_XFLM_OK;
					uChar = ASCII_BACKSLASH;
				}
			}
		}
		else if (uChar == ASCII_WILDCARD)
		{
			if (m_bMayHaveWildCards && pbCharIsWild)
			{
				*pbCharIsWild = TRUE;
			}
		}
	}
	
Process_Char:

	if (!bAsian)
	{
		
		// Must check for double characters if non-US and non-Asian
		// character set

		if (m_uiLanguage != XFLM_US_LANG)
		{
			if (RC_BAD( rc = flmWPCheckDoubleCollation( 
				m_pIStream, m_bUnicodeStream, bAllowTwoIntoOne, 
				&uChar, &m_uNextChar, &bTwoIntoOne, m_uiLanguage)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		if (RC_BAD( rc = readCharFromStream( &m_uNextChar)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				m_uNextChar = 0;
			}
			else
			{
				RC_UNEXPECTED_ASSERT( rc);
				goto Exit;
			}
		}
	}

	// Convert each character to its WP equivalent

	if (!flmUnicodeToWP( uChar, &ui16WpChar))
	{
		ui16WpChar = 0;
	}

	if (!flmUnicodeToWP( m_uNextChar, &ui16NextWpChar))
	{
		ui16NextWpChar = 0;
	}

	// If we have an unconvertible UNICODE character, the collation
	// value for it will be COLS0

	if (!ui16WpChar)
	{
		if (!bAsian)
		{
			ui16Col = COLS0;
		}
		else
		{
			if (uChar < 0x20)
			{
				ui16Col = 0xFFFF;
				ui16SubCol = uChar;
			}
			else
			{
				ui16Col = uChar;
				ui16SubCol = 0;
			}
		}
	}
	else
	{
		if (!bAsian)
		{
			ui16Col = flmWPGetCollation( ui16WpChar, m_uiLanguage);
			if (bTwoIntoOne)
			{
				// Since two characters were merged into one, increment
				// the collation value by one.  In the case of something
				// like 'ch', there is a collation value between 'c' and
				// 'd'.  flmWPGetCollation would have returned the
				// collation value for 'c' ... incrementing by one gives
				// us the proper collation value for 'ch' (i.e., the
				// collation value between 'c' and 'd').

				ui16Col++;
			}
		}
		else
		{
			if (flmWPAsiaGetCollation( ui16WpChar, ui16NextWpChar, ui16Col,
					&ui16Col, &ui16SubCol, &ucCase, !m_bCaseSensitive) == 2)
			{
				
				// Next character was consumed by collation

				m_uNextChar = 0;
			}
		}
	}

	if (pui16Col)
	{
		*pui16Col = ui16Col;
	}

	// Consume m_uNextChar if two characters merged into one

	if (bTwoIntoOne)
	{
		m_uNextChar = 0;
	}
	
	// Subcollation

	if( pui16SubCol)
	{
		if( uChar > 127 && !bAsian)
		{
			ui16SubCol = ui16WpChar
							  ? flmWPGetSubCol( ui16WpChar, ui16Col, m_uiLanguage)
							  : uChar;

			if( !m_bCaseSensitive)
			{
				// If the sub-collation value is the original
				// character, it means that the collation could not
				// distinguish the characters and sub-collation is being
				// used to do it.  However, this creates a problem when the
				// characters are the same character except for case.  In that
				// scenario, we incorrectly return a not-equal when we are
				// doing a case-insensitive comparison.  So, at this point,
				// we need to use the sub-collation for the upper-case of the
				// character instead of the sub-collation for the character
				// itself.

				if( ui16WpChar && ui16SubCol == ui16WpChar)
				{
					ui16SubCol = flmWPGetSubCol(
												flmWPUpper( ui16WpChar),
												ui16Col, m_uiLanguage);
				}
			}
		}

		*pui16SubCol = ui16SubCol;
	}

	// Case

	if( pucCase)
	{
		if (!m_bCaseSensitive)
		{
			*pucCase = 0;
		}
		else
		{
			if (!bAsian && ui16WpChar)
			{
				// flmWPIsUpper() returns FALSE if the character is lower or
				// TRUE if the character is not lower case.
	
				if( flmWPIsUpper( ui16WpChar))
				{
					if( bTwoIntoOne)
					{
						if( flmWPIsUpper( ui16NextWpChar))
						{
							ucCase = 0x03;
						}
						else
						{
							ucCase = 0x10;
						}
					}
					else
					{
						ucCase = 0x01;
					}
				}
			}
			*pucCase = ucCase;
		}
	}

	if (puChar)
	{
		*puChar = uChar;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_DbSystem::compareUTF8Strings(
	const FLMBYTE *		pucLString,
	FLMUINT					uiLStrBytes,
	FLMBOOL					bLeftWild,
	const FLMBYTE *		pucRString,
	FLMUINT					uiRStrBytes,
	FLMBOOL					bRightWild,
	FLMUINT					uiCompareRules,
	FLMUINT					uiLanguage,
	FLMINT *					piResult)
{
	RCODE						rc = NE_XFLM_OK;
	F_BufferIStream		bufferLStream;
	F_BufferIStream		bufferRStream;
	F_CollIStream			lStream;
	F_CollIStream			rStream;

	if (RC_BAD( rc = bufferLStream.open( pucLString, uiLStrBytes)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = bufferRStream.open( pucRString, uiRStrBytes)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = lStream.open( &bufferLStream, FALSE, uiLanguage,
								uiCompareRules, bLeftWild)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = rStream.open( &bufferRStream, FALSE, uiLanguage,
								uiCompareRules, bRightWild)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fqCompareCollStreams( &lStream, &rStream,
		(bLeftWild || bRightWild) ? TRUE : FALSE,
		uiLanguage, piResult)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE F_DbSystem::compareUnicodeStrings(
	const FLMUNICODE *	puzLString,
	FLMUINT					uiLStrBytes,
	FLMBOOL					bLeftWild,
	const FLMUNICODE *	puzRString,
	FLMUINT					uiRStrBytes,
	FLMBOOL					bRightWild,
	FLMUINT					uiCompareRules,
	FLMUINT					uiLanguage,
	FLMINT *					piResult)
{
	RCODE						rc = NE_XFLM_OK;
	F_BufferIStream		bufferLStream;
	F_BufferIStream		bufferRStream;
	F_CollIStream			lStream;
	F_CollIStream			rStream;

	if( RC_BAD( rc = bufferLStream.open( (FLMBYTE *)puzLString, uiLStrBytes)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = bufferRStream.open( (FLMBYTE *)puzRString, uiRStrBytes)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = lStream.open( &bufferLStream, TRUE, uiLanguage,
		uiCompareRules, bLeftWild)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = rStream.open( &bufferRStream, TRUE, uiLanguage,
		uiCompareRules, bRightWild)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fqCompareCollStreams( &lStream, &rStream,
		(bLeftWild || bRightWild) ? TRUE : FALSE, uiLanguage, piResult)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE XFLMAPI F_DbSystem::utf8IsSubStr(
	const FLMBYTE *	pszString,
	const FLMBYTE *	pszSubString,
	FLMUINT				uiCompareRules,
	FLMUINT				uiLanguage,
	FLMBOOL *			pbExists)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iResult = 0;
	FLMBYTE *		pszSearch = NULL;
	FLMUINT			uiSubStringLen = f_strlen( pszSubString);
	
	if( RC_BAD( rc = f_alloc( uiSubStringLen + 3, &pszSearch)))
	{
		goto Exit;
	}
	
	pszSearch[0] = '*';
	f_memcpy( &pszSearch[ 1], pszSubString, uiSubStringLen);
	pszSearch[ uiSubStringLen + 1] = '*';
	pszSearch[ uiSubStringLen + 2] = '\0';

	if( RC_BAD( rc = compareUTF8Strings( 
		pszString, f_strlen( pszString), FALSE, pszSearch, 
		uiSubStringLen + 2, TRUE, uiCompareRules, uiLanguage, &iResult)))
	{
		goto Exit;
	}
	
	*pbExists = (iResult)?FALSE:TRUE;

Exit:

	if( pszSearch)
	{
		f_free( &pszSearch);
	}
	
	return( rc);
}
