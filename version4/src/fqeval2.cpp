//-------------------------------------------------------------------------
// Desc:	Query evaluation
// Tabs:	3
//
//		Copyright (c) 1993-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fqeval2.cpp 12271 2006-01-19 14:48:13 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE OpSyntaxError(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUBitAND(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUBitOR(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUBitXOR(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUSMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSSMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSUMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpURMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRUMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSRMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRSMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRRMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUSDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSSDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSUDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpURDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRUDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSRDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRSDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRRDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUSMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSSMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSUMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUSPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSSPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSUPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpURPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRUPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSRPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRSPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRRPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUUMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpUSMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSSMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSUMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpURMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRUMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpSRMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRSMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);

FSTATIC RCODE OpRRMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult);


/****************************************************************************
Desc:		Returns a syntax error on the operation.  The combined values are 
			not allowed in the operation.
Ret:		FERR_CURSOR_SYNTAX
****************************************************************************/
FSTATIC RCODE OpSyntaxError(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	F_UNREFERENCED_PARM( pLhs);
	F_UNREFERENCED_PARM( pRhs);
	F_UNREFERENCED_PARM( pResult );
	return( RC_SET( FERR_CURSOR_SYNTAX));
}

/****************************************************************************
Desc:		Performs the bit AND operation on signed and unsigned values.
			Used for UtoU, UtoSigned, SignedToU and SignedToSigned.
Notes:	The signed and unsigned values better take up the same location
			in the union.  I (Scott) am a little concerned that we are even
			working on signed values.  I think that a uniary cast to an
			unsigned in the query would be better.
****************************************************************************/

FSTATIC RCODE OpUUBitAND(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal & pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return( FERR_OK);
}
	
/****************************************************************************
Desc:		Performs the bit OR operation on signed and unsigned values.
Notes:	The signed and unsigned values better take up the same location
			in the union.  
****************************************************************************/

FSTATIC RCODE OpUUBitOR(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal | pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return( FERR_OK);
}

/****************************************************************************
Desc:		Performs the bit XOR operation on signed and unsigned values.
Notes:	The signed and unsigned values better take up the same location
			in the union.  
****************************************************************************/

FSTATIC RCODE OpUUBitXOR(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	pResult->val.uiVal = pLhs->val.uiVal ^ pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return( FERR_OK);
}

/****************************************************************************
Desc:		Performs the multiply operation on unsigned, signed and real values.
Notes:	Overflow conditions are not checked for.  For the most part,
			a signed value is usually a negative value - otherwise it would
			be an unsigned value.
****************************************************************************/

FSTATIC RCODE OpUUMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	pResult->val.uiVal = pLhs->val.uiVal * pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return( FERR_OK);
}
	
FSTATIC RCODE OpUSMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	

	pResult->val.iVal = (FLMINT)pLhs->val.uiVal * pRhs->val.iVal;
	pResult->eType = FLM_INT32_VAL;
	return( FERR_OK);
}
	
FSTATIC RCODE OpSSMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	pResult->val.iVal = pLhs->val.iVal * pRhs->val.iVal;
	pResult->eType = (pResult->val.iVal < 0) ? FLM_INT32_VAL : FLM_UINT32_VAL;
	return( FERR_OK);
}

FSTATIC RCODE OpSUMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	pResult->val.iVal = pLhs->val.iVal * (FLMINT)pRhs->val.uiVal;
	pResult->eType = FLM_INT32_VAL;
	return( FERR_OK);
}

FSTATIC RCODE OpURMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = (FLMLREAL)pLhs->val.uiVal * pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}
FSTATIC RCODE OpRUMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real * (FLMLREAL) pRhs->val.uiVal;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpSRMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = (FLMLREAL)pLhs->val.iVal * pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}
FSTATIC RCODE OpRSMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real * (FLMLREAL) pRhs->val.iVal;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}	
FSTATIC RCODE OpRRMult(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real * pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

/****************************************************************************
Desc:		Performs the divide operation on unsigned, signed and real values.
			Divide on signed and unsigned values will result in a signed or
			unsigned value that is floored down to an integer.
Notes:	Divide by zero is checked for, but the result is 0 unsigned.
			For the most part, a signed value is usually a negative value.
VISIT:	There are problems in casting large unsigned values into a signed
			value.  We should be treating INT32_VAL as a negative bit and
			always working with unsigned values.  I hate to change too much
			though.
VISIT:	Need to visit the divide/mod by zero cases.
****************************************************************************/

FSTATIC RCODE OpUUDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	if( pRhs->val.uiVal)
	{
		pResult->val.uiVal = pLhs->val.uiVal / pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.uiVal = 0;				// Divide by ZERO case.
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}
	
FSTATIC RCODE OpUSDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	if( pRhs->val.iVal)
	{
		pResult->val.iVal = (FLMINT)pLhs->val.uiVal / pRhs->val.iVal;
		pResult->eType = FLM_INT32_VAL;
	}
	else							// Divide by ZERO case - let's try not to crash.
	{
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}
	
FSTATIC RCODE OpSSDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if( pRhs->val.iVal)
	{
		pResult->val.iVal = pLhs->val.iVal / pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) 
									? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	else	
	{
		pResult->val.uiVal = 0;				// divide by ZERO case
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}

FSTATIC RCODE OpSUDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if( pRhs->val.uiVal)
	{
		pResult->val.iVal = pLhs->val.iVal / (FLMINT)pRhs->val.uiVal;
		pResult->eType = FLM_INT32_VAL;
	}
	else	// Divide by ZERO case - let's try not to crash.
	{
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}

FSTATIC RCODE OpURDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	if( pRhs->val.Real)
	{
		pResult->val.Real = (FLMLREAL)pLhs->val.uiVal / pRhs->val.Real;
		pResult->eType = FLM_REAL_VAL;
	}
	else 
	{
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
#endif
}

FSTATIC RCODE OpRUDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	if( pRhs->val.uiVal)
	{
		pResult->val.Real = pLhs->val.Real / (FLMLREAL)pRhs->val.uiVal;
		pResult->eType = FLM_REAL_VAL;
	}
	else
	{
 		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpSRDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	if( pRhs->val.Real)
	{
		pResult->val.Real = (FLMLREAL)pLhs->val.iVal / pRhs->val.Real;
		pResult->eType = FLM_REAL_VAL;
	}
	else
	{
 		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
#endif
}

FSTATIC RCODE OpRSDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	if( pRhs->val.iVal)
	{
		pResult->val.Real = pLhs->val.Real / (FLMLREAL)pRhs->val.iVal;
		pResult->eType = FLM_REAL_VAL; 
	}
	else
	{
 		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpRRDiv(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	if( pRhs->val.Real)
	{
		pResult->val.Real = pLhs->val.Real / pRhs->val.Real;
		pLhs->eType = FLM_REAL_VAL;
	}
	else
	{
 		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
#endif
}

/****************************************************************************
Desc:		Performs the modulo operation on unsigned, signed and real values.
Notes:	Mod by zero is checked for, but the result is 0 unsigned.
			For the most part, a signed value is usually a negative value.
****************************************************************************/
FSTATIC RCODE OpUUMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	if( pRhs->val.uiVal != 0)
	{
		pResult->val.uiVal = pLhs->val.uiVal % pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.uiVal = 0;				// MOD by ZERO case.
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}
	
FSTATIC RCODE OpUSMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	if( pRhs->val.iVal != 0)
	{
		pResult->val.iVal = (FLMINT)pLhs->val.uiVal / pRhs->val.iVal;
		pResult->eType = FLM_INT32_VAL;
	}
	else	
	{
		pResult->val.uiVal = 0;				// MOD by ZERO case
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}
	
FSTATIC RCODE OpSSMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if( pRhs->val.iVal != 0)
	{
		pResult->val.iVal = pLhs->val.iVal % pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0)
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	else	
	{
		pResult->val.uiVal = 0;				// MOD by ZERO case
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}

FSTATIC RCODE OpSUMod(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if( pRhs->val.uiVal != 0)
	{
		pResult->val.iVal = pLhs->val.iVal % ((FLMINT)pRhs->val.uiVal);
		pResult->eType = FLM_INT32_VAL;
	}
	else	// Divide by ZERO case - let's try not to crash.
	{
		pResult->val.uiVal = 0;
		pResult->eType = FLM_UNKNOWN;
	}
	return( FERR_OK);
}


/****************************************************************************
Desc:		Performs an addition operation on unsigned, signed and real values.
Notes:	Underflow and overflow conditions are not checked.  In addition,
			casting unsigned values to signed could lose data.  Casting
			ANYTHING to real will always lose some data.
****************************************************************************/

FSTATIC RCODE OpUUPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	pResult->val.uiVal = pLhs->val.uiVal + pRhs->val.uiVal;
	pResult->eType = FLM_UINT32_VAL;
	return( FERR_OK);
}
	
FSTATIC RCODE OpUSPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	if( ( pRhs->val.iVal >= 0) || 
		 ( pLhs->val.uiVal > MAX_SIGNED_VAL))
	{
		pResult->val.uiVal = pLhs->val.uiVal + (FLMUINT)pRhs->val.iVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = (FLMINT)pLhs->val.uiVal + pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) 
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	return( FERR_OK);
}
	
FSTATIC RCODE OpSSPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	pResult->val.iVal = pLhs->val.iVal + pRhs->val.iVal;
	pResult->eType = (pResult->val.iVal < 0) 
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	return( FERR_OK);
}

FSTATIC RCODE OpSUPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if( ( pLhs->val.iVal >= 0) ||
		 ( pRhs->val.uiVal > MAX_SIGNED_VAL))
	{
		pResult->val.uiVal = (FLMUINT)pLhs->val.iVal + pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = pLhs->val.iVal + (FLMINT)pRhs->val.uiVal;
		pResult->eType = (pResult->val.iVal < 0) 
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	return( FERR_OK);
}

FSTATIC RCODE OpURPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = (FLMLREAL)pLhs->val.uiVal + pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

FSTATIC RCODE OpRUPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real + (FLMLREAL)pRhs->val.uiVal;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpSRPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = (FLMLREAL)pLhs->val.iVal + pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

FSTATIC RCODE OpRSPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real + (FLMLREAL)pRhs->val.iVal;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpRRPlus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real + pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

/****************************************************************************
Desc:		Performs a subtraction operation on unsigned, signed and real values.
Notes:	Underflow and overflow conditions are not checked.  In addition,
			casting unsigned values to signed could lose data.  Casting
			ANYTHING to real will always lose some data.
****************************************************************************/
FSTATIC RCODE OpUUMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	if( pLhs->val.uiVal >= pRhs->val.uiVal)
	{
		pResult->val.uiVal = pLhs->val.uiVal - pRhs->val.uiVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = (FLMINT)(pLhs->val.uiVal - pRhs->val.uiVal);
		pResult->eType = FLM_INT32_VAL;
	}
		return( FERR_OK);
}
	
FSTATIC RCODE OpUSMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{	
	/* VISIT: The original code had...
	if(( pRhs->val.iVal < 0) || ( pRhs->val.uiVal < pLhs->val.uiVal))
	I don't know what the second part means.
	*/

	if( pRhs->val.iVal < 0) 
	{
		pResult->val.uiVal = pLhs->val.uiVal - pRhs->val.iVal;
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = (FLMINT)pLhs->val.uiVal - pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) 
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	return( FERR_OK);
}
	
FSTATIC RCODE OpSSMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if(( pLhs->val.iVal > 0) && ( pRhs->val.iVal < 0))
	{
		pResult->val.uiVal = (FLMUINT)( pLhs->val.iVal - pRhs->val.iVal);
		pResult->eType = FLM_UINT32_VAL;
	}
	else
	{
		pResult->val.iVal = pLhs->val.iVal - pRhs->val.iVal;
		pResult->eType = (pResult->val.iVal < 0) 
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	return( FERR_OK);
}

FSTATIC RCODE OpSUMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
	if( pRhs->val.uiVal > MAX_SIGNED_VAL)
	{
		pResult->val.iVal = ( pLhs->val.iVal - MAX_SIGNED_VAL)
			 - (FLMINT)(pRhs->val.uiVal - MAX_SIGNED_VAL);
		pResult->eType = FLM_INT32_VAL;
	}
	else
	{
		pResult->val.iVal = pLhs->val.iVal - (FLMINT)pRhs->val.uiVal;
		pResult->eType = (pResult->val.iVal < 0) 
								? FLM_INT32_VAL : FLM_UINT32_VAL;
	}
	return( FERR_OK);
}

FSTATIC RCODE OpURMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = (FLMLREAL)pLhs->val.uiVal - pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

FSTATIC RCODE OpRUMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real - (FLMLREAL)pRhs->val.uiVal;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpSRMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = (FLMLREAL)pLhs->val.iVal - pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

FSTATIC RCODE OpRSMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real - (FLMLREAL)pRhs->val.iVal;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}	

FSTATIC RCODE OpRRMinus(
	FQATOM_p	pLhs,		
	FQATOM_p	pRhs,	
	FQATOM_p	pResult)
{
#ifndef FLM_REAL
	return( OpSyntaxError( pLhs, pRhs, pResult));
#else
	pResult->val.Real = pLhs->val.Real - pRhs->val.Real;
	pResult->eType = FLM_REAL_VAL;
	return( FERR_OK);
#endif
}

FQ_OPERATION * FQ_DoOperation [ 
	((LAST_ARITH_OP - FIRST_ARITH_OP) + 1) * 9 ] =
{
/*	U = Unsigned		S = Signed		R = Real
					U + U					U + S						U + R
						S + U					S + S						S + R
							R + U					R + S						R + R */
/* BITAND */	OpUUBitAND,			OpUUBitAND,				OpSyntaxError,
						OpUUBitAND,			OpUUBitAND,				OpSyntaxError,
							OpSyntaxError,		OpSyntaxError,			OpSyntaxError,
/* BITOR  */	OpUUBitOR,			OpUUBitOR,				OpSyntaxError,
						OpUUBitOR,			OpUUBitOR,				OpSyntaxError,
							OpSyntaxError,		OpSyntaxError,			OpSyntaxError,
/* BITXOR */	OpUUBitXOR,			OpUUBitXOR,				OpSyntaxError,
						OpUUBitXOR,			OpUUBitXOR,				OpSyntaxError,
							OpSyntaxError,		OpSyntaxError,			OpSyntaxError,
/* MULT   */	OpUUMult,			OpUSMult,				OpURMult,
						OpSUMult,			OpSSMult,				OpSRMult,
							OpRUMult,			OpRSMult,				OpRRMult,
/* DIV    */	OpUUDiv,				OpUSDiv,					OpURDiv,
						OpSUDiv,				OpSSDiv,					OpSRDiv,
							OpRUDiv,				OpRSDiv,					OpRRDiv,
/* MOD    */	OpUUMod,				OpUSMod,					OpSyntaxError,
						OpSUMod,				OpSSMod,					OpSyntaxError,
							OpSyntaxError,		OpSyntaxError,			OpSyntaxError,
/* PLUS   */	OpUUPlus,			OpUSPlus,				OpURPlus,
						OpSUPlus,			OpSSPlus,				OpSRPlus,
							OpRUPlus,			OpRSPlus,				OpRRPlus,
/* MINUS  */	OpUUMinus,			OpUSMinus,				OpURMinus,
						OpSUMinus,			OpSSMinus,				OpSRMinus,
							OpRUMinus,			OpRSMinus,				OpRRMinus
};
