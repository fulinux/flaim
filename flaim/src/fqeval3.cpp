//-------------------------------------------------------------------------
// Desc:	Query evaluation
// Tabs:	3
//
//		Copyright (c) 1993-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fqeval3.cpp 12271 2006-01-19 14:48:13 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:		Performs unary minus operation on list of query atoms.
****************************************************************************/
RCODE flmCurDoNeg(
	FQATOM_p pResult)
{
	RCODE		rc = FERR_OK;
	FQATOM_p	pTmpQAtom;

	/* Perform operation on list according to operand types */

	for( pTmpQAtom = pResult; pTmpQAtom; pTmpQAtom = pTmpQAtom->pNext)
	{
		if( IS_UNSIGNED( pTmpQAtom->eType))
		{
			if( pTmpQAtom->val.uiVal >= MAX_SIGNED_VAL)
				pTmpQAtom->eType = NO_TYPE;
			else
			{
				pTmpQAtom->val.iVal = -((FLMINT)(pTmpQAtom->val.uiVal));
				pTmpQAtom->eType = FLM_INT32_VAL;
			}
		}
		else if( IS_SIGNED( pTmpQAtom->eType))
		{
			pTmpQAtom->val.iVal *= -1;
		}
		else if( pTmpQAtom->eType != FLM_UNKNOWN)
		{
			rc = RC_SET( FERR_CURSOR_SYNTAX);
			break;
		}
	}
	return( rc);
}


/****************************************************************************
Desc:		Performs match begin operation on two stack elements of buffered type.
			Will do a MATCH or MATCH_BEGIN, depending on the bMatchEntire flag.
			When bMatchEntire is TRUE, we are doing the MATCH operation.
****************************************************************************/
FLMUINT flmCurDoMatchOp(
	FQATOM_p		pLhs,
	FQATOM_p		pRhs,
	FLMUINT		uiLang,
	FLMBOOL		bLeadingWildCard,
	FLMBOOL		bTrailingWildCard)
{
	FLMUINT		uiFlags = pLhs->uiFlags | pRhs->uiFlags;
	FLMUINT		uiTrueFalse = 0;

	/* Verify operand types - non-text and non-binary return false */
	if( !IS_BUF_TYPE( pLhs->eType) || !IS_BUF_TYPE( pRhs->eType))
		goto Exit;

	/* If one of the operands is binary, simply do a byte comparison of the two
		values without regard to case or wildcards. */
	if(( pLhs->eType == FLM_BINARY_VAL) || ( pRhs->eType == FLM_BINARY_VAL))
	{
		FLMUINT	uiLen1;
		FLMUINT	uiLen2;

		uiLen1 = pLhs->uiBufLen;
		uiLen2 = pRhs->uiBufLen;
		flmAssert( !bLeadingWildCard);
		if ((bTrailingWildCard) && (uiLen2 > uiLen1))
		{
			uiLen2 = uiLen1;
		}

		uiTrueFalse = (FLMUINT)(((uiLen1 == uiLen2) &&
										 (f_memcmp( pLhs->val.pucBuf,
														pRhs->val.pucBuf, uiLen1) == 0))
										? (FLMUINT)FLM_TRUE
										: (FLMUINT)FLM_FALSE);
		goto Exit;
	}

	/* If wildcards are set, do a string search, first making necessary
		adjustments for case sensitivity. */

	/*
	NOTE: THIS IS MATCH BEGIN CASE WITHOUT WILD CARD.

	The non-wild case for bMatchEntire (DO_MATCH) does NOT
	come through this section of code.  Rather, flmCurDoEQ is called
	instead of this routine in that case.
	*/
		
	if( pLhs->eType == FLM_TEXT_VAL && pRhs->eType == FLM_TEXT_VAL)	
	{
		// Always true if there is a wild card.
		uiTrueFalse = flmTextMatch( pLhs->val.pucBuf, pLhs->uiBufLen,
										pRhs->val.pucBuf, pRhs->uiBufLen,
										uiFlags, bLeadingWildCard, bTrailingWildCard,
										uiLang );
	}
	else
	{
		uiTrueFalse = FLM_FALSE;
	}

Exit:

	return uiTrueFalse;
}


/****************************************************************************
Desc:		Performs 'contains' operation on stack elements of buffered type.
Note:		Wildcards and case insensitive searches are not allowed yet on 
			binary values.
VISIT:	Changed to take the OR bits from the left and right wFlags instead
			of just taking the left if it had a value otherwise right.
VISIT:	Don't like ALWAYS setting spaces to space and converting to upper.
			We should have a flag so that if the ?hs is used again it doesn't
			parse through the data again.
****************************************************************************/
FLMUINT flmCurDoContainsOp(
	FQATOM_p		pLhs,
	FQATOM_p		pRhs,
	FLMUINT		uiLang
	)
{
	FLMBYTE *	pResult = NULL;
	FLMUINT		uiFlags = pLhs->uiFlags | pRhs->uiFlags;
	FLMUINT		uiTrueFalse = 0;			// set for invalid response.

	/* Verify operands -- both should be buffered types */
	if( !IS_BUF_TYPE( pLhs->eType) || !IS_BUF_TYPE( pRhs->eType))
		goto Exit;

	/* If one of the operands is binary, simply do a byte comparison of the two
		values without regard to case or wildcards. */
	if(( pLhs->eType == FLM_BINARY_VAL) || ( pRhs->eType == FLM_BINARY_VAL))
	{
		uiTrueFalse = FLM_FALSE;
		for( pResult = pLhs->val.pucBuf;
				(FLMUINT)(pResult - pLhs->val.pucBuf) < pLhs->uiBufLen;
				pResult++)
		{
			if(( *pResult == pRhs->val.pucBuf[0]) &&
				( f_memcmp( pLhs->val.pucBuf, pRhs->val.pucBuf, pRhs->uiBufLen) == 0))
			{
				uiTrueFalse = FLM_TRUE;
				goto Exit;
			}
		}
		goto Exit;
	}

	uiTrueFalse = flmTextMatch( pLhs->val.pucBuf, pLhs->uiBufLen,
										pRhs->val.pucBuf, pRhs->uiBufLen,
										uiFlags, TRUE,		// Leading wild card.
										TRUE,		// trailing WC
										uiLang );
Exit:

	return uiTrueFalse;
}


/****************************************************************************
Desc:		Performs a compare on two operands.  Strings are matched fully.
****************************************************************************/
FLMINT flmCurDoRelationalOp(
	FQATOM_p		pLhs,
	FQATOM_p		pRhs,
	FLMUINT		uiLang)				// Language for text
{
	FLMUINT		uiFlags = pLhs->uiFlags | pRhs->uiFlags;	// String flags.
	FLMINT		iCompVal = 0;

	/* Do operation according to operand types */
	switch( pLhs->eType)
	{
		case FLM_TEXT_VAL:
			flmAssert( pRhs->eType == FLM_TEXT_VAL);
			iCompVal = flmTextCompare( pLhs->val.pucBuf, pLhs->uiBufLen,
						 						  pRhs->val.pucBuf, pRhs->uiBufLen, uiFlags, uiLang);
			break;

		case FLM_UINT32_VAL:
			switch( pRhs->eType)
			{
				case FLM_UINT32_VAL:
					iCompVal = FQ_COMPARE( pLhs->val.uiVal, pRhs->val.uiVal);
					break;

				case FLM_INT32_VAL:
					if( pRhs->val.iVal < 0)
						iCompVal = 1;
					else
						iCompVal = FQ_COMPARE( pLhs->val.uiVal, (FLMUINT)pRhs->val.iVal);
					break;

				default:
					flmAssert( 0);					// Shouldn't happen
					break;
			}
			break;

		case FLM_INT32_VAL:
			switch( pRhs->eType)
			{
				case FLM_INT32_VAL:
					iCompVal = FQ_COMPARE( pLhs->val.iVal, pRhs->val.iVal);
					break;		

				case FLM_UINT32_VAL:
					if( pLhs->val.iVal < 0)
						iCompVal = -1;
					else
						iCompVal = FQ_COMPARE((FLMUINT)pLhs->val.iVal, pRhs->val.uiVal);
					break;

				default:
					flmAssert( 0);					// Shouldn't happen
					break;
			}
			break;

		case FLM_REC_PTR_VAL:
			flmAssert( pRhs->eType == FLM_REC_PTR_VAL || pRhs->eType == FLM_UINT32_VAL);
			iCompVal = FQ_COMPARE( pLhs->val.uiVal, pRhs->val.uiVal);
			break;

		case FLM_BINARY_VAL:
			flmAssert( (pRhs->eType == FLM_BINARY_VAL) || (pRhs->eType == FLM_TEXT_VAL));
			if ((iCompVal = f_memcmp( pLhs->val.pucBuf, pRhs->val.pucBuf,
							((pLhs->uiBufLen > pRhs->uiBufLen)
										? pRhs->uiBufLen
										: pLhs->uiBufLen))) == 0)
			{
				if (pLhs->uiBufLen < pRhs->uiBufLen)
				{
					iCompVal = -1;
				}
				else if (pLhs->uiBufLen > pRhs->uiBufLen)
				{
					iCompVal = 1;
				}
			}
			break;
		default:
			flmAssert( 0);		// Shouldn't happen
			break;
	}
	return iCompVal;
}
