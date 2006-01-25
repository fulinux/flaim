//-------------------------------------------------------------------------
// Desc:	Traverse to previous element in a b-tree.
// Tabs:	3
//
//		Copyright (c) 1991-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsprev.cpp 12286 2006-01-19 14:55:18 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/***************************************************************************
Desc:		Go to the previous element in the logical b-tree while building key
Out:		stack set up for next element
Return:	FERR_OK, FERR_BT_END_OF_DATA (0xFFFF) or error number
Note:		This could be called at any level of the b-tree.
*****************************************************************************/
RCODE FSBtPrevElm(
	FDB_p			pDb,
	LFILE *		pLFile,	/* Logical file definition */
	BTSK_p		pStack		/* Stack of variables for each level */
	)
{
	RCODE			rc = FERR_OK;						/* Return code */
	FLMUINT		uiBlkAddr;
	FLMUINT		uiTargetElm;							/* Target element to scan for */
	FLMUINT		uiPrevElm = 0;
	FLMUINT		uiPrevKeyCnt = 0;
	FLMUINT		uiElmKeyLen = 0;
	FLMUINT		uiKeyBufSize = pStack->uiKeyBufSize;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMBYTE *	pCurElm;							/* Points to the current element */
	FLMBYTE *	pBlk;

	/* Check if you are at or before the first element in the block */
	if( pStack->uiCurElm <= BH_OVHD)
	{
		pBlk = BLK_PTR( pStack );
		/* YES - read in the previous block & go to the last element */
		if( (uiBlkAddr = (FLMUINT) FB2UD( &pBlk[ BH_PREV_BLK ])) == BT_END)
		{
			/* Unless you are at the end */
			rc = FERR_BT_END_OF_DATA;
		}
		else
		{
			if( RC_OK(rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
			{
				/* Set blkEnd & curElm & adjust parent block to previous element */
				pBlk = pStack->pBlk;
				pStack->uiCurElm = pStack->uiBlkEnd = pStack->uiBlkEnd;
				if( pStack->uiFlags & FULL_STACK)
					rc = FSAdjustStack( pDb, pLFile, pStack, FALSE);
			}
		}
	}
	/* Move down 1 before the current element */
	if( RC_OK(rc))
	{
		if( pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			pStack->uiCurElm -= BNE_DATA_OVHD;
			pBlk = pStack->pBlk;
			pCurElm = &pBlk[ pStack->uiCurElm ];
			flmCopyDrnKey( pStack->pKeyBuf, pCurElm);
			goto Exit;
		}

		/* Set up pointing to first element in the block */
		uiTargetElm = pStack->uiCurElm;
		pStack->uiCurElm = BH_OVHD;				/* Start at first element */
		pBlk = pStack->pBlk;
		
		while( pStack->uiCurElm < uiTargetElm )	/* Loop till target is hit */
		{
			pCurElm = &pBlk[ pStack->uiCurElm ];
			uiPrevKeyCnt = (FLMUINT) (BBE_GET_PKC( pCurElm ));
			uiElmKeyLen  = (FLMUINT) (BBE_GET_KL( pCurElm ));

			if( uiElmKeyLen + uiPrevKeyCnt > uiKeyBufSize )
			{
				rc = RC_SET( FERR_CACHE_ERROR);
				goto Exit;
			}
			if( uiElmKeyLen)
			{
				f_memcpy( &pStack->pKeyBuf[ uiPrevKeyCnt ], 
							 &pCurElm[ uiElmOvhd ], uiElmKeyLen );
			}
			uiPrevElm = pStack->uiCurElm;
			if( RC_BAD(rc = FSBlkNextElm( pStack )))
			{
				rc = (rc == FERR_BT_END_OF_DATA) ? FERR_OK : rc;	// Hide element end
				break;		/* Break out if at the end of the block */
			}
		}
		pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen;
		pStack->uiCurElm = uiPrevElm;
	}
Exit:
	return( rc );
}


/***************************************************************************
Desc:	Return the last DIN in the current element's reference list
Out:		state information updated to refer to the last reference
Return:	DIN the din of the last reference
Notes:	The element must be the last continued element for a key.
			This algorithm will not work for bit-vector types.
*****************************************************************************/
FLMUINT FSRefLast(
	BTSK_p		pStack,		/* Small stack to hold btree variables*/
	DIN_STATE_p	pState,		/* Holds offset, one run number, etc.*/
	FLMUINT *	puiDomainRV)		/* Returns the elements domain */
{
	FLMBYTE *	pCurElm = CURRENT_ELM( pStack );
	FLMBYTE *	pCurRef;					/* Points to start of references 	 */
	FLMUINT		uiRefSize;						/* Size of the element's references  */

	/* Point past the domain, ignore return value */
	pCurRef = pCurElm;
	*puiDomainRV = FSGetDomain( &pCurRef, pStack->uiElmOvhd );
	uiRefSize = (FLMUINT)(BBE_GET_RL(pCurElm) - (pCurRef - BBE_REC_PTR(pCurElm)));

	return FSGetPrevRef( pCurRef, pState, uiRefSize);
}

/****************************************************************************
Desc:	Position and return the previous reference saving the state
Out:   	state updated from current position
Return:	previous value from target
***************************************************************************/
FLMUINT FSGetPrevRef(
	FLMBYTE *	pCurRef,			// Points to start of references 
	DIN_STATE_p	pState,				// Holds tate information to get next
	FLMUINT		uiTarget)			// Stop when the target is hit and back off
{
	FLMUINT		uiDin ;					/* Current din to compute and return */
	FLMUINT		uiOneRuns = 0;
	FLMUINT		uiDelta = 0;
	FLMUINT		uiLastOffset = 0;				/* Last offset - used to back up     */
	FLMBYTE		byValue;

	RESET_DINSTATE_p( pState );
	uiDin = DINNextVal( pCurRef, pState );

	while( pState->uiOffset < uiTarget)
	{
		/* Get the current byte to see what kind of item it is */
		byValue = (FLMBYTE) pCurRef[ uiLastOffset = pState->uiOffset ];
		if( DIN_IS_REAL_ONE_RUN(byValue))
		{
			uiDelta = 0;
			uiOneRuns = DINOneRunVal( pCurRef, pState);
			uiDin -= uiOneRuns;
		}
		else
		{
			uiDelta = DINNextVal( pCurRef, pState);
			uiDin -= uiDelta;
		}
	}
	/**
	***	Hit the end of the reference set for the current element.
	***	The current din is a correct return value.  The pState structure
	***	must be setup to refer to the last entry using uiLastOffset.
	**/
	if(( pState->uiOffset = uiLastOffset) != 0)
	{
		if( uiDelta == 0)
		{
			/**	One runs was the last entry, setup for one run state **/
			uiOneRuns--;						/* uiOneRuns state is zero based */
			pState->uiOnes = uiOneRuns;
		}
	}
	return( uiDin);
}

/****************************************************************************
Desc:  	Go to the previous reference given a valid cursor.
Out:   	cursor updated if there is a previous reference
Return:	RCODE FERR_OK | FERR_BT_END_OF_DATA (0xFFFF) or error
****************************************************************************/
RCODE FSRefPrev(
	FDB_p			pDb,
	LFILE *		pLFile,	/* Logical file definition */
	BTSK_p		pStack,	/* Small stack to hold btree variables*/
	DIN_STATE_p	pState,	/* Holds offset, one run number, etc.*/
	FLMUINT *	puiDinRV)		/* Last din used and returns din */
{
	RCODE			rc;			/* Return code */
	FLMBYTE *	pCurRef;	/* Points to current reference */
	FLMBYTE *	pCurElm;	/* Points to current element */
	FLMUINT		uiDin = *puiDinRV;
	FLMUINT		uiDummyDomain;
	FLMBYTE		byValue;

	/* Point to the start of the current reference */
	pCurRef = pCurElm = CURRENT_ELM( pStack );
	FSGetDomain( &pCurRef, pStack->uiElmOvhd );

	/* Was this the first reference */
	if( pState->uiOffset == 0)
	{
		/**
		***	Read in the previous element or return FERR_BT_END_OF_DATA if first
		**/

		if(  BBE_IS_FIRST( pCurElm))
			return( FERR_BT_END_OF_DATA );
			
		if( RC_BAD(rc = FSBtPrevElm( pDb, pLFile, pStack )))
			return( rc );

		uiDin = FSRefLast( pStack, pState, &uiDummyDomain );
	}

	/**
	***	Start reading until hit the current state values
	**/
	else
	{
		/* Get current byte - could be a 1 run */
		byValue = pCurRef[ pState->uiOffset ];

		if( DIN_IS_REAL_ONE_RUN(byValue) && pState->uiOnes)
				// 03/11/96 !(pState->uiOnes == 1))
		{
			/**
			***	One runs are easy if you are not on the first one run (above)
			**/
			uiDin++;										/* Previous din is one more */
			pState->uiOnes--;
		}
		else
		{
			uiDin = FSGetPrevRef( pCurRef, pState, pState->uiOffset);
		}
	}

	*puiDinRV = uiDin;
	return( FERR_OK );
}
