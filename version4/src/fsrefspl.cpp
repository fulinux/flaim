//-------------------------------------------------------------------------
// Desc:	Index reference splitting routines.
// Tabs:	3
//
//		Copyright (c) 1991-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsrefspl.cpp 12286 2006-01-19 14:55:18 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMUINT FSSplitRefSet(
	FLMBYTE *		leftBuf,
	FLMUINT *		leftLenRV,
	FLMBYTE *		rightBuf,
	FLMUINT *		rightLenRV,
	FLMBYTE *		refPtr,
	FLMUINT			uiRefLen,
	FLMUINT			uiSplitFactor);

/****************************************************************************
Desc:  	Try to split a reference set.  The size is over the first threshold.
			If you split this update the b-tree with the new element and position
			to the current element for insert of the din.
Out:   	The element may be split, or may not.  It is not the callers concern.
****************************************************************************/
RCODE FSRefSplit(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK_p *		pStackRV,					/* Stack area */
	FLMBYTE *	pElmBuf,					/* Setup with elements key */
	FLMUINT		din,						/* din to insert */
	FLMUINT		uiDeleteFlag,			/* Set if you are to delete din */
	FLMUINT		uiSplitFactor)			/* Set to SPLIT_90_10 | SPLIT_50_50*/
{
	RCODE			rc = FERR_OK;				/* Must be set */
	BTSK_p		pStack = *pStackRV;			/* Stack may change on you! */
	FLMBYTE *	pCurElm = CURRENT_ELM( pStack );/* Points to current element*/
	FLMINT		iElmLen;						/* Length of element in pElmBuf[] */

	FLMBYTE		leftBuf[ MAX_REC_ELM ];	/* Left buffer */
	FLMUINT		leftDomain;
	FLMUINT		leftLen;

	FLMBYTE		rightBuf[MAX_REC_ELM ];	/* Right (current) buffer */
	FLMUINT		rightDomain;
	FLMUINT		rightLen;

	FLMBYTE *	refPtr;						/* Points to the references */
	FLMBYTE *	recPtr;
	FLMUINT		uiRefLen;						/* Length of references */
	FLMUINT		firstFlag = 0;
	
	refPtr = pCurElm;
	recPtr = BBE_REC_PTR( pCurElm );
	rightDomain = FSGetDomain( &refPtr, (FLMBYTE)pStack->uiElmOvhd  );
	uiRefLen = (FLMUINT)(BBE_GET_RL( pCurElm ) - (FLMUINT)(refPtr - recPtr));
FSRS_try_again:
	leftDomain = FSSplitRefSet( leftBuf, &leftLen, rightBuf, &rightLen,
										 refPtr, uiRefLen, uiSplitFactor );

	if( leftDomain ==  0)			/* Split failed, setup to add */
	{
		/* Try again using a different split factor - OK to fail above */
		/* In the future, should just handle no splitting and go on */
		if(  uiSplitFactor == SPLIT_50_50)
		{
			uiSplitFactor = SPLIT_90_10;
			goto FSRS_try_again;
		}
#if 0
		return( RC_SET( FERR_BTREE_ERROR));			/* Can't handle this right now */
#else
		/* Setup for inserting the din into the right buffer and call replace */
		leftDomain = DIN_DOMAIN(din) + 1;
		f_memcpy( rightBuf, refPtr, rightLen = uiRefLen );
		leftLen = 0;
#endif
	}
	/**
	***	Write the right element's references.
	***	Write the right domain if non-zero and replace element
	**/

	iElmLen = (FLMINT)(BBE_REC_OFS( pElmBuf ));
	refPtr = recPtr = &pElmBuf[ iElmLen ];
	if( rightDomain)								/* Write the domain if present */
	{
		*refPtr++ = SEN_DOMAIN;
		SENPutNextVal( &refPtr, rightDomain );
	}

	if( DIN_DOMAIN(din) < leftDomain)		/* Remember references DESCEND */
	{
		/* Build element inserting the input din */
		if( uiDeleteFlag)
		{
			if( FSSetDeleteRef( refPtr, rightBuf, din, &rightLen))
			{
				/* rightLen should not have changed if error found */
				return ( RC_SET( FERR_KEY_NOT_FOUND) );
#if 0
				f_memcpy( refPtr, rightBuf, rightLen );
				rc = RC_SET( FERR_KEY_NOT_FOUND);	/* Return with this error */
#endif
			}
		}
		else if( FSSetInsertRef( refPtr, rightBuf, din, &rightLen ))
		{
			/* Reference there so give up and return success */
			goto Exit;			/* rc is set to FERR_OK */
		}
	}
	else
		f_memcpy( refPtr, rightBuf, rightLen );

	/* The other flags and lengths are been set by the caller */
	iElmLen += BBE_SET_RL( pElmBuf, rightLen + (FLMUINT) (refPtr - recPtr ));

	if(  BBE_IS_FIRST( pElmBuf ) && leftLen)
	{
		firstFlag++;
		BBE_CLR_FIRST( pElmBuf );				/* Element will no longer be first */
		
		/* Log the block before modifying it. */
		if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
			goto Exit;
		pCurElm = CURRENT_ELM( pStack );/* Points to current element*/

		BBE_CLR_FIRST( pCurElm );			/* Clear first flag */

		/* Call replace below because FIRST flag is now clear */
	}

	/* Can call replace because FIRST flag was NOT set */
	if( RC_BAD( rc = FSBtReplace( pDb, pLFile, &pStack, pElmBuf, iElmLen )))
		goto Exit;

	/**
	***	Write the left buffer
	***	Should be positioned to the right buffer
	***
	**/
	if( leftLen)
	{
		/**
		***	Adjust variables to build and point to the
		***	left buffers references.  Set the domain
		***	and insert into the b-tree.  Then go to the next element
		**/
		BBE_CLR_LAST( pElmBuf );							/* Element will no longer be last */
		if(  firstFlag)
			BBE_SET_FIRST( pElmBuf);
		iElmLen = (FLMINT)(BBE_REC_OFS( pElmBuf ));
		refPtr = recPtr = &pElmBuf[ iElmLen ];
		*refPtr++ = SEN_DOMAIN;
		SENPutNextVal( &refPtr, leftDomain );

		if( DIN_DOMAIN(din) >= leftDomain)
		{
			/* Build element inserting the input din */
			if( uiDeleteFlag)
			{
				if( FSSetDeleteRef( refPtr, leftBuf, din, &leftLen))
					return( RC_SET( FERR_KEY_NOT_FOUND ));
			}
			else
				/* If this returned with an error code, is alreay in set */
				if( FSSetInsertRef( refPtr, leftBuf, din, &leftLen ))
					f_memcpy( refPtr, leftBuf, leftLen );
		}
		else
			f_memcpy( refPtr, leftBuf, leftLen );

		iElmLen += BBE_SET_RL( pElmBuf, leftLen + (FLMUINT) (refPtr - recPtr));

		/* Setup the pStack and bsKeyBuf[] for the insert */

		if( RC_BAD(rc = FSBtScanTo( pStack, &pElmBuf[ BBE_KEY ], 
											(FLMUINT)(BBE_GET_KL(pElmBuf)),  0)))
			goto Exit;

		rc = FSBtInsert( pDb, pLFile, &pStack, pElmBuf, iElmLen );
	}
Exit:
	return( rc );
}

/****************************************************************************
Desc:  	Split a reference set within a domain value.  If buffer cannot be
			split then will return a leftDomain value of ZERO.
			Must have a minimum of 2 references in left and right buffers.
Out:   	References split to the left or right buffers.
Return:	0 if the split failed, else the leftDomain value
Notes: 	Two references are needed in each side because one reference may
			be deleted leaving the left buffer with a domain & maybe should not.
****************************************************************************/
FSTATIC FLMUINT FSSplitRefSet(
	FLMBYTE *		leftBuf,
	FLMUINT *		leftLenRV,
	FLMBYTE *		rightBuf,
	FLMUINT * 		rightLenRV,
	FLMBYTE *		refPtr,
	FLMUINT			uiRefLen,
	FLMUINT			uiSplitFactor)			/* Set to SPLIT_90_10 | SPLIT_50_50*/
{
	FLMUINT			leftDomain =  0;
	FLMUINT			din = 0;
	FLMUINT			oneRuns = 0;
	FLMUINT			delta;
	FLMUINT			rightLen;
	FLMUINT			offsetTarget = (uiSplitFactor == SPLIT_90_10) 
											? REF_SPLIT_90_10 : REF_SPLIT_50_50;
	DIN_STATE		leftState, rightState, refState;
	FLMBYTE			byValue;
	FLMUINT			uiLeftCnt;

	RESET_DINSTATE( leftState );
	RESET_DINSTATE( rightState);
	RESET_DINSTATE( refState  );
	
	/* Read the first din value */
	din =	DINNextVal( refPtr, &refState );
	DINPutNextVal( leftBuf, &leftState, din );
	uiLeftCnt = 1;

	// Must have at least 2 in the left buffer.

	while( refState.uiOffset < offsetTarget || uiLeftCnt < 2)
	{
		byValue = refPtr[ refState.uiOffset ];
		if( DIN_IS_REAL_ONE_RUN( byValue ))
		{
			oneRuns = DINOneRunVal( refPtr, &refState );
			DINPutOneRunVal( leftBuf, &leftState, oneRuns );
			din -= oneRuns;
		}
		else
		{
			delta = DINNextVal( refPtr, &refState );
			DINPutNextVal( leftBuf, &leftState, delta );
			din -= delta;
		}
		uiLeftCnt++;
	}

	/* Made it past the target point - find where domain changes */
	leftDomain = DIN_DOMAIN( din );

	/* Don't parse past the end */
	while( refState.uiOffset < uiRefLen)
	{
		byValue = refPtr[ refState.uiOffset ];
		if( DIN_IS_REAL_ONE_RUN( byValue ))
		{
			oneRuns = DINOneRunVal( refPtr, &refState );
			if( DIN_DOMAIN(din-oneRuns) != leftDomain)
			{
				/* This is tricky, write only correct number of one runs */
				delta = din & 0xFF;
				if( delta)
					DINPutOneRunVal( leftBuf, &leftState, delta );

				/* Increment delta because setting up for next element */
				delta++;
				oneRuns -= delta;
				/* Write din and one runs below */
				din -= delta;
				break;
			}
			DINPutOneRunVal( leftBuf, &leftState, oneRuns );
			din -= oneRuns;
		}
		else
		{
			delta = DINNextVal( refPtr, &refState );
			din -= delta;
			if( DIN_DOMAIN(din) != leftDomain)
			{
				oneRuns = 0;
				break;
			}
			DINPutNextVal( leftBuf, &leftState, delta );
		}
	}
	if( refState.uiOffset == uiRefLen)
		return(  0 );			/* Cannot split, caller take care of*/

	/* Start writing to the right side, compare /w uiRefLen proves > 2 refs */
	DINPutNextVal( rightBuf, &rightState, din );
	if( oneRuns)
		DINPutOneRunVal( rightBuf, &rightState, oneRuns );

	*leftLenRV = leftState.uiOffset;
	rightLen = (FLMUINT)(uiRefLen - refState.uiOffset);

	f_memcpy( &rightBuf[ rightState.uiOffset ],
				 &refPtr[ refState.uiOffset ],
				 rightLen );

	*rightLenRV = rightLen + rightState.uiOffset;

	return( leftDomain );
}
