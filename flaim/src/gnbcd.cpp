//-------------------------------------------------------------------------
// Desc:	BCD numbers in GEDCOM.
// Tabs:	3
//
//		Copyright (c) 1992-1993,1995-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gnbcd.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

// External Data

extern FLMBYTE ucMaxBcdINT32[ ];

extern FLMBYTE ucMinBcdINT32[ ];

extern FLMBYTE ucMaxBcdUINT32[ ];


/****************************************************************************
Desc:	Given an unsigned number create the matching FLAIM-specific BCD number.
Note:	If terminating byte is half-full, low-nibble value is
		undefined.  Example: -125 creates B1-25-FX
Method:
		using a MOD algorithm, stack BCD values -- popping to
		destination reverses the order for correct final sequence
****************************************************************************/
RCODE GedPutUINT(
	POOL *		pPool,
	NODE *		pNode,
	FLMUINT		uiNum,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucPtr;
	FLMBYTE		ucNibStk[ F_MAX_NUM_BUF + 1];	/* spare byte for odd BCD counts */
	FLMBYTE *	pucNibStk;

	if( pNode == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	/* push spare (undefined) nibble for possible half-used terminating byte */

	pucNibStk = &ucNibStk[ 1];

	/* push terminator nibble -- popped last */

	*pucNibStk++ = 0x0F;

	/* push digits */
	/* do 32 bit division until we get down to 16 bits */

	while( uiNum >= 10)
	{
		*pucNibStk++ = (FLMBYTE)(uiNum % 10);	/* push BCD nibbles in reverse order */
		uiNum /= 10;
	}
	*pucNibStk++ = (FLMBYTE)uiNum;			/* push last nibble of number */

	/* Determine number of bytes required for BCD number & allocate space */

	if( (pucPtr = (FLMBYTE *)GedAllocSpace( pPool, pNode, FLM_NUMBER_TYPE,
					((pucNibStk - ucNibStk) >> 1),	/* count: nibbleCount/2 & truncate */
					uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/* Pop stack and pack nibbles into byte stream a pair at a time */

	do
	{
		*pucPtr++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);	/* spare stack byte stops seg wrap */

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Given an signed number create the matching FLAIM-specific BCD number.
Note:	If terminating byte is half-full, low-nibble value is
		undefined.  Example: -125 creates B1-25-FX
Method:
		using a MOD algorithm, stack BCD values -- popping to
		destination reverses the order for correct final sequence
WARNING:
		-2,147,483,648 may yield different results on different platforms
****************************************************************************/
RCODE GedPutINT(
	POOL *		pPool,
	NODE *		pNode,
	FLMINT		iNum,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiNum;
	FLMBYTE *	pucPtr;
	FLMBYTE		ucNibStk[ F_MAX_NUM_BUF + 1];	/* spare byte for odd BCD counts */
	FLMBYTE *	pucNibStk;
	FLMINT		iNegFlag;

	if( pNode == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	/* push spare (undefined) nibble for possible half-used terminating byte */

	pucNibStk = &ucNibStk[ 1];

	/* push terminator nibble -- popped last */

	*pucNibStk++ = 0x0F;

	/* separate sign from magnituted; (FLMUINT)un = +/- n & flag */

	uiNum = ((iNegFlag = iNum < 0) != 0)
		?	-iNum
		:	iNum;

	/* push digits */
	/* do 32 bit division until we get down to 16 bits */

	while( uiNum >= 10)
	{
		*pucNibStk++ = (FLMBYTE)(uiNum % 10);	/* push BCD nibbles in reverse order */
		uiNum /= 10;
	}
	*pucNibStk++ = (FLMBYTE)uiNum;			/* push last nibble of number */

	if( iNegFlag)
		*pucNibStk++ = 0x0B;					/* push sign nibble last */

	/* Determine number of bytes required for BCD number & allocate space */

	if( (pucPtr = (FLMBYTE *)GedAllocSpace( pPool, pNode, FLM_NUMBER_TYPE,
					((pucNibStk - ucNibStk) >> 1), 	/* count: nibbleCount/2 & truncate */
					uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/* Pop stack and pack nibbles into byte stream a pair at a time */

	do
	{
		*pucPtr++ = (FLMBYTE)((pucNibStk[ -1] << 4) | pucNibStk[ -2]);
	}
	while( (pucNibStk -= 2) > &ucNibStk[ 1]);	/* spare stack byte stops seg wrap */

	if (pNode->ui32EncId)
	{
		pNode->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:
	return( rc);
}

/*API~*********************************************************************
Name : GedGetINT
Area : GEDCOM
Desc : Returns a signed value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into WSDWORD.
		// FERR_CONV_NUM_UNDERFLOW - the value is too small to fit into WSDWORD.
	GedGetINT(
		NODE *		pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMINT  *	piNum
			// [OUT] Returns the 16-bit unsigned number value.
	)
{
	BCD_TYPE	bcd;
	RCODE		rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_BAD(rc = flmBcd2Num( GedValType( pNode), GedValLen( pNode),
								(const FLMBYTE *)GedValPtr( pNode), &bcd)))
	{
		goto Exit;
	}
	if( bcd.bNegFlag)
	{
		*piNum = -((FLMINT)bcd.uiNum);
		rc = (bcd.uiNibCnt < 11) ||
			  (bcd.uiNibCnt == 11 &&
			  (!bcd.pucPtr || (f_memcmp( bcd.pucPtr, ucMinBcdINT32, 6) <= 0)))
					?	FERR_OK
					:	RC_SET( FERR_CONV_NUM_UNDERFLOW);
		// goto Exit;
	}
	else
	{
		*piNum = (FLMINT)bcd.uiNum;
		rc = (bcd.uiNibCnt < 10) ||
			  (bcd.uiNibCnt == 10 &&
			  (!bcd.pucPtr || (f_memcmp( bcd.pucPtr, ucMaxBcdINT32, 5) <= 0)))
					?	FERR_OK
					:	RC_SET( FERR_CONV_NUM_OVERFLOW);
		// goto Exit;
	}

Exit:
	return( rc);
}

/*API~*********************************************************************
Name : GedGetINT32
Area : GEDCOM
Desc : Returns a 32-bit signed value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into WSDWORD.
		// FERR_CONV_NUM_UNDERFLOW - the value is too small to fit into WSDWORD.
	GedGetINT32(
		NODE *		pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMINT32  *	pi32Num
			// [OUT] Returns the 16-bit unsigned number value.
	)
{
	BCD_TYPE	bcd;
	RCODE		rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_OK(rc = flmBcd2Num( GedValType( pNode), GedValLen( pNode),
								(const FLMBYTE *)GedValPtr( pNode), &bcd)))
	{
		if( bcd.bNegFlag)
		{
			*pi32Num = -((FLMINT32)bcd.uiNum);
			rc = (bcd.uiNibCnt < 11) ||
				  (bcd.uiNibCnt == 11 &&
				  (!bcd.pucPtr || (f_memcmp( bcd.pucPtr, ucMinBcdINT32, 6) <= 0)))
						?	FERR_OK
						:	RC_SET( FERR_CONV_NUM_UNDERFLOW);
			// goto Exit;
		}
		else
		{
			*pi32Num = (FLMINT32)bcd.uiNum;
			rc = (bcd.uiNibCnt < 10) ||
				  (bcd.uiNibCnt == 10 &&
				  (!bcd.pucPtr || (f_memcmp( bcd.pucPtr, ucMaxBcdINT32, 5) <= 0)))
						?	FERR_OK
						:	RC_SET( FERR_CONV_NUM_OVERFLOW);
			// goto Exit;
		}
	}

Exit:
	return( rc);
}

/*API~*********************************************************************
Name : GedGetINT16
Area : GEDCOM
Desc : Returns a 16-bit signed value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into WSWORD.
		// FERR_CONV_NUM_UNDERFLOW - the value is too small to fit into WSWORD.
	GedGetINT16(
		NODE *			pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMINT16 	*	pi16Num
			// [OUT] Returns the 16-bit unsigned number value.
	)
{
	BCD_TYPE	bcd;
	RCODE		rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_OK( rc = flmBcd2Num( GedValType( pNode), GedValLen( pNode),
									(const FLMBYTE *)GedValPtr( pNode), &bcd)))
	{
		if( bcd.bNegFlag )
		{
			*pi16Num = -((FLMINT16)(bcd.uiNum));
			rc = (bcd.uiNibCnt < 6) ||
				  (bcd.uiNibCnt == 6 && bcd.uiNum <= FLM_MAX_INT16 )
						?	FERR_OK
						:	RC_SET( FERR_CONV_NUM_UNDERFLOW);
			// goto Exit;
		}
		else
		{
			*pi16Num = (FLMINT16)bcd.uiNum;
			rc = (bcd.uiNibCnt < 5) ||
				  (bcd.uiNibCnt == 5 && bcd.uiNum < FLM_MAX_INT16 )
						?	FERR_OK
						:	RC_SET( FERR_CONV_NUM_OVERFLOW);
			//goto Exit;
		}
	}

Exit:
	return( rc);
}

/*API~*********************************************************************
Name : GedGetUINT
Area : GEDCOM
Desc : Returns an unsigned value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into FLMUINT16
		// FERR_CONV_NUM_UNDERFLOW - the value is a negative number
	GedGetUINT(
		NODE *			pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMUINT *		puiNum
			// [OUT] Returns the 32-bit unsigned number value.
	)
{
	BCD_TYPE	bcd;
	RCODE		rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_OK(rc = flmBcd2Num( GedValType( pNode),
										GedValLen( pNode),
										(const FLMBYTE *)GedValPtr( pNode),
										&bcd)))
	{
		*puiNum = bcd.uiNum;

		if( bcd.bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
		else if( bcd.uiNibCnt < 10)
		{
			rc = FERR_OK;
		}
		else if( bcd.uiNibCnt == 10)
		{
			rc = (!bcd.pucPtr || (f_memcmp( bcd.pucPtr, ucMaxBcdUINT32, 5) <= 0))
					?	FERR_OK
					:	RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
	}

Exit:
	
	return( rc);
}

/*API~*********************************************************************
Name : GedGetUINT8
Area : GEDCOM
Desc : Returns an 8-bit unsigned value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into FLMUINT16
		// FERR_CONV_NUM_UNDERFLOW - the value is a negative number
	GedGetUINT8(
		NODE *			pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMUINT8 *		pui8Num
			// [OUT] Returns the 8-bit unsigned number value.
	)
{
	BCD_TYPE		bcd;
	RCODE			rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_OK(rc = flmBcd2Num( GedValType( pNode),
								 GedValLen( pNode),
								 (const FLMBYTE *)GedValPtr( pNode),
								 &bcd)))
	{
		*pui8Num = (FLMUINT8)bcd.uiNum;
		rc = bcd.bNegFlag
				?	RC_SET( FERR_CONV_NUM_UNDERFLOW)
				:	(bcd.uiNibCnt < 3) || (bcd.uiNibCnt == 3 && bcd.uiNum < FLM_MAX_UINT8)
					?	FERR_OK
					:	RC_SET( FERR_CONV_NUM_OVERFLOW);
	}
	
Exit:

	return( rc);
}

/*API~*********************************************************************
Name : GedGetUINT32
Area : GEDCOM
Desc : Returns a 32-bit unsigned value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into FLMUINT16
		// FERR_CONV_NUM_UNDERFLOW - the value is a negative number
	GedGetUINT32(
		NODE *			pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMUINT32 *		pui32Num
			// [OUT] Returns the 32-bit unsigned number value.
	)
{
	BCD_TYPE		bcd;
	RCODE			rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_OK(rc = flmBcd2Num( GedValType( pNode),
										GedValLen( pNode),
										(const FLMBYTE *)GedValPtr( pNode),
										&bcd)))
	{
		*pui32Num = (FLMUINT32)bcd.uiNum;

		if( bcd.bNegFlag)
		{
			rc = RC_SET( FERR_CONV_NUM_UNDERFLOW);
		}
		else if( bcd.uiNibCnt < 10)
		{
			rc = FERR_OK;
		}
		else if( bcd.uiNibCnt == 10)
		{
			rc = (!bcd.pucPtr || (f_memcmp( bcd.pucPtr, ucMaxBcdUINT32, 5) <= 0))
					?	FERR_OK
					:	RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
		else
		{
			rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
		}
	}

Exit:
	
	return( rc);
}

/*API~*********************************************************************
Name : GedGetUINT16
Area : GEDCOM
Desc : Returns a 16-bit unsigned value from a GEDCOM node field.
		 The data in the node may be a number type, text type or context type.
Notes:
*END************************************************************************/
RCODE // FERR_OK
		// FERR_CONV_ILLEGAL	- node was not a number type or had
		// non-converatble values if a context or text type.
		// FERR_CONV_NUM_OVERFLOW	- the value is too large to fit into FLMUINT16
		// FERR_CONV_NUM_UNDERFLOW - the value is a negative number
	GedGetUINT16(
		NODE *			pNode,
			// [IN] Pointer to a GEDCOM node.
		FLMUINT16	*	pui16Num
			// [OUT] Returns the 16-bit unsigned number value.
	)
{
	BCD_TYPE		bcd;
	RCODE			rc = FERR_OK;

	if (pNode->ui32EncId)
	{
		if (!(pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( RC_OK(rc = flmBcd2Num( GedValType( pNode),
										GedValLen( pNode),
										(const FLMBYTE *)GedValPtr( pNode),
										&bcd)) == FERR_OK)
	{
		*pui16Num = (FLMUINT16)bcd.uiNum;
		rc = bcd.bNegFlag
				?	RC_SET( FERR_CONV_NUM_UNDERFLOW)
				:	(bcd.uiNibCnt < 5) ||
					(bcd.uiNibCnt == 5 && bcd.uiNum < FLM_MAX_UINT16 )
						?	FERR_OK
						:	RC_SET( FERR_CONV_NUM_OVERFLOW);
	}

Exit:
	
	return( rc);
}

