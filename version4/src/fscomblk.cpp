//-------------------------------------------------------------------------
// Desc:	B-tree block combining
// Tabs:	3
//
//		Copyright (c) 1992-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fscomblk.cpp 12283 2006-01-19 14:53:15 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMINT FSBlkCompressPKC(
	BTSK_p			pStack,
	FLMBYTE *		pTempPKCBuf);


/*
Desc:  	Build a PKC buffer for a single element
Return:	Length used within the PKC buffer
*/
FINLINE FLMUINT FSElmBuildPKC(
	FLMBYTE *		pPkcBuf,
	FLMBYTE *		pElement,
	FLMBYTE *		pElmPkcBuf,
	FLMUINT			uiElmOvhd)
{
	FLMUINT			uiPkc;
	FLMUINT		   uiKeyLen;
	
	if( uiElmOvhd == BNE_DATA_OVHD)
		return 0;

	uiKeyLen = (FLMUINT)(BBE_GET_KL( pElement ));

	if( (uiPkc = (FLMUINT)(BBE_GET_PKC( pElement))) != 0)
	{
		f_memmove( pPkcBuf, pElmPkcBuf, uiPkc );
	}
	if( uiPkc + uiKeyLen > BBE_PKC_MAX)
		uiKeyLen = (FLMUINT)(BBE_PKC_MAX - uiPkc);

	f_memmove( &pPkcBuf[ uiPkc ], &pElement[ uiElmOvhd ], uiKeyLen );

	return( uiPkc + uiKeyLen);
}


/****************************************************************************
Desc:		Try to combine two blocks into a single block.  The algorithm will
			alternate tring the block to the right or uiLeft of the 'target' block.
			The bsCurElm MUST be positioned to the current element.  The bsCurElm
			may be at the very end of the block not pointing to an element.  If
			so then if blocks are combine bsCurElm should then be valid.
Notes: 	Remember that this can be called on any level of the btree.
****************************************************************************/
RCODE FSCombineBlks(
	FDB *				pDb,
	LFILE *			pLFile,
	BTSK_p *			pStackRV	/* Stack may change on you */
	)
{
	RCODE			   rc = FERR_OK;
	BTSK_p			pStack = *pStackRV;
	SCACHE *			pLeftCache;
	SCACHE *			pRightCache;
	FLMBYTE *		pLeftBlk;
	FLMBYTE *		pRightBlk;
	FLMBYTE *		pBlk = pStack->pBlk;
	FLMBOOL			bReleaseLeft = FALSE;
	FLMBOOL			bReleaseRight = FALSE;
	FLMUINT			uiLeftBlkAddr;
	FLMUINT			uiRightBlkAddr;
	FLMUINT			uiLeftBlkEnd;
	FLMUINT			uiRightBlkEnd;
	FLMUINT			uiBlkAddr = pStack->uiBlkAddr;
	FLMUINT			uiBlkEnd = pStack->uiBlkEnd;
	FLMUINT        uiCurElm = pStack->uiCurElm;
	FLMUINT			uiElmOvhd = pStack->uiElmOvhd;
	FLMUINT			uiBlkSize;
	FLMUINT			uiPosToElm = 0;
	FLMUINT			uiPosToBlk = 0;
	FLMUINT			uiSplitPoint;
	FLMUINT			uiTargetSplitPoint;
	FLMINT			iTemp;
	FLMINT			iDelta;
	BTSK			   tempStack;
	FLMBYTE			pPkcBuf[ BBE_PKC_MAX ];
	DB_STATS *		pDbStats;

	f_yieldCPU();				/* NLM - release cpu */

	/* Return if either block is leftmost or rightmost block */
	uiLeftBlkAddr  = (FLMUINT) FB2UD( &pBlk[ BH_PREV_BLK ]);
	uiRightBlkAddr = (FLMUINT) FB2UD( &pBlk[ BH_NEXT_BLK ]);
	if( ( uiLeftBlkAddr == BT_END) || ( uiRightBlkAddr == BT_END))
	{
		goto Exit;		// Should return SUCCESS
	}

	uiBlkSize = pDb->pFile->FileHdr.uiBlockSize - uiElmOvhd;

	/* Read in left and right blocks - make sure all cache ptrs are valid */

	if( RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF, uiLeftBlkAddr, NULL,
								&pLeftCache)))
	{
		goto Exit;
	}
	bReleaseLeft = TRUE;

	if( RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF, uiRightBlkAddr, NULL,
								&pRightCache)))
	{
		goto Exit;
	}
	bReleaseRight = TRUE;

	/* Determine if there is room without compressing first elm in current blk*/
	pLeftBlk  = pLeftCache->pucBlk;
	uiLeftBlkEnd  = (FLMUINT) FB2UW( &pLeftBlk[ BH_BLK_END ]);
	pRightBlk = pRightCache->pucBlk;
	uiRightBlkEnd = (FLMUINT) FB2UW( &pRightBlk[ BH_BLK_END ]);

	/* Don't want to fill too tight - so don't subtract BH_OVHD from sum */
	if( uiLeftBlkEnd + uiBlkEnd + uiRightBlkEnd > uiBlkSize + uiBlkSize)
	{
FSCB_Unpin:
		if( bReleaseRight)
		{
			ScaReleaseCache( pRightCache, FALSE);
			bReleaseRight = FALSE;
		}
		if( bReleaseLeft)
		{
			ScaReleaseCache( pLeftCache, FALSE);
			bReleaseLeft = FALSE;
		}

		if( RC_OK( rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
		{
			pStack->uiCurElm = uiCurElm;
			/* bsKeyBuf[] could have been wiped out in a scanTo call! */
			FSBlkBuildPKC( pStack, pStack->pKeyBuf, FSBBPKC_AT_CURELM);
		}
		goto Exit;
	}

	/**---------------------------------------------------------------------
	*** This is a very good yet extreamly tricky algorithm!
	***	If the delta (difference in size) of the left and right blocks
	*** is more than the size of the middle block the entire middle will
	***	will be moved to the left or right block that is not very full.
	***	Otherwise, uiTargetSplitPoint will be computed to be around the
	*** point that will place elements in both blocks to fill the left
	*** and right blocks to about the same point.
	*** NOTE: Read the comments if all is fuzzy.
	***--------------------------------------------------------------------*/

	iTemp = uiBlkEnd - BH_OVHD + BBE_PKC_MAX;	/* temp is max. bytes to move */
	uiTargetSplitPoint = 0;

	if( uiLeftBlkEnd < uiRightBlkEnd )			/* Put most in the left block */
	{
		iDelta = uiRightBlkEnd - uiLeftBlkEnd;
		if( iTemp <= iDelta)
			pStack->uiCurElm = uiBlkEnd;		/* Put all in the left block */
		else
			uiTargetSplitPoint = BH_OVHD + iDelta + ((iTemp - iDelta) >> 1);
	}
	else												/* Put most in the right block */
	{
		iDelta = uiLeftBlkEnd - uiRightBlkEnd;
		if( iTemp <= iDelta )
			pStack->uiCurElm = BH_OVHD;			/* Put all in right block */
		else
			uiTargetSplitPoint = BH_OVHD + ((iTemp - iDelta) >> 1);
	}

	if( uiTargetSplitPoint)						/* If try to divide into both blocks */
	{
		pStack->uiCurElm = uiTargetSplitPoint;

		/* Scan AFTER targetSplitPoint */
		if( RC_BAD(rc = FSBtScanTo( pStack, NULL, 0, 0)))
			goto Exit;
	
		/**-----------------------------------------------------------------
		*** Last check to see if elements will fit in both blocks.
		*** This is still a chance that all elements will go to one block
		***----------------------------------------------------------------*/
		if( ( uiLeftBlkEnd + (pStack->uiCurElm - BH_OVHD) > uiBlkSize ) ||
			 ( uiRightBlkEnd + (uiBlkEnd - pStack->uiCurElm) + BBE_PKC_MAX > uiBlkSize ))
			goto FSCB_Unpin;
	}

	/* We are now guarenteed to fit!!! - log blocks and start moving */

	if( (pDbStats = pDb->pDbStats) != NULL)
	{
		LFILE_STATS *		pLFileStats;

		if( (pLFileStats = fdbGetLFileStatPtr( pDb, pLFile)) != NULL)
		{
			pLFileStats->bHaveStats =
			pDbStats->bHaveStats = TRUE;
			pLFileStats->ui64BlockCombines++;
		}
	}

	if( RC_BAD( rc = ScaLogPhysBlk( pDb, &pLeftCache )))
		goto Exit;
	pLeftBlk  = pLeftCache->pucBlk;

	uiSplitPoint = pStack->uiCurElm;
	if( uiSplitPoint != BH_OVHD )				/* Elements to move into LEFT block? */
	{
		FLMUINT		uiCompressBytes;
		FLMUINT		uiBytesAdded = uiSplitPoint - BH_OVHD;
	
		tempStack.pSCache = pLeftCache;
		tempStack.pBlk = pLeftCache->pucBlk;
		FSBlkToStack( &tempStack );
		tempStack.uiKeyBufSize = MAX_KEY_SIZ;

		/* Algorithm could call the move routine if it wasn't for uiCompressBytes */
		f_memmove( &pLeftBlk[ uiLeftBlkEnd ], &pBlk[ BH_OVHD ], uiBytesAdded );
		tempStack.uiCurElm = uiLeftBlkEnd;

		uiLeftBlkEnd += uiBytesAdded;
		tempStack.uiBlkEnd = uiLeftBlkEnd;
		UW2FBA( uiLeftBlkEnd, &pLeftBlk[ BH_BLK_END ]);
	
		uiCompressBytes = FSBlkCompressPKC( &tempStack, pPkcBuf );
		if( uiCompressBytes == 0xFFFF)	/* Special case that shouldn't happen. */
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
		if( uiCurElm < uiSplitPoint )
		{
			uiPosToBlk = uiLeftBlkAddr;
			uiPosToElm = (FLMUINT)((uiLeftBlkEnd - uiBytesAdded) + uiCurElm - BH_OVHD);

			// NOTE: uiCompressBytes should be zero for fixed element blocks - see
			// FSBlkCompressPKC

			if( uiCurElm != BH_OVHD )
				uiPosToElm -= uiCompressBytes;
		}
	}
	UD2FBA( uiRightBlkAddr, &pLeftBlk[ BH_NEXT_BLK ] );
	ScaReleaseCache( pLeftCache, FALSE);
	bReleaseLeft = FALSE;

	/**-------------------------------------------------------------------
	***		WOW - done with the left block.  Move the rest of the data
	***		into the right block.  Be carefull to compress the first
	***		element of the right block and decompress the element that
	***		is at the split point in the middle (deleted) block.
	***------------------------------------------------------------------*/

	if( RC_BAD( rc = ScaLogPhysBlk( pDb, &pRightCache )))
		goto Exit;
	pRightBlk = pRightCache->pucBlk;

	if( uiSplitPoint != uiBlkEnd )					/* Elements to move into RIGHT block? */
	{
		FLMBYTE *		pSplitPoint = &pBlk[ uiSplitPoint ];
		tempStack.pSCache = pRightCache;
		tempStack.pBlk = pRightCache->pucBlk;
		FSBlkToStack( &tempStack );
		tempStack.uiKeyBufSize = MAX_KEY_SIZ;
		
		/* Setup to position to current block and element at end of routine */
		if( uiCurElm >= uiSplitPoint )
		{
			uiPosToBlk = uiRightBlkAddr;
			uiPosToElm = (FLMUINT)((uiCurElm - uiSplitPoint) + BH_OVHD);

			// No PKC in fixed element blocks.

			if( uiCurElm != uiSplitPoint && tempStack.uiElmOvhd != BNE_DATA_OVHD)
			{
				uiPosToElm += BBE_GET_PKC( pSplitPoint );
			}
		}
		/**---------------------------------------------------------------
		*** If ScanTo() doesn't match uiCurElm exact then current element
		*** does not have the proper stuff in the pkc buffer or bsKeyBuf
		***--------------------------------------------------------------*/
		/* pStack->uiCurElm is uiSplitPoint */
		FSBlkBuildPKC( pStack, pPkcBuf, FSBBPKC_AT_CURELM );
		if( RC_BAD( rc = FSBlkMoveElms( &tempStack, pSplitPoint,
												 (FLMUINT)(uiBlkEnd - uiSplitPoint), pPkcBuf )))
			goto Exit;
	}

	UD2FBA( uiLeftBlkAddr, &pRightBlk[ BH_PREV_BLK ] );
	ScaReleaseCache( pRightCache, FALSE);
	bReleaseRight = FALSE;

	/* Now we can free the current block - blkFixLinks may work instead */

	rc = FSBlockFree( pDb, pStack->pSCache);
	pStack->pSCache = NULL;
	pStack->pBlk = NULL;
	if( RC_BAD( rc))
		return( rc );

	/**----------------------------------------------------------------------
	***		Now the hard part. Go to the parent & delete the current element.
	***		Go to the previous element and modify to reflect the new last
	***		element in the left block.
	***---------------------------------------------------------------------*/

	if( RC_BAD( rc = FSDelParentElm( pDb, pLFile, &pStack )))
		return( rc );

	if( uiSplitPoint != BH_OVHD )				/* Is there is new stuff in left blk?*/
	{
		/* Position and fixup the parent elements */
		if( RC_OK(rc = FSGetBlock( pDb, pLFile, uiLeftBlkAddr, pStack)))
		{
			rc = FSNewLastBlkElm( pDb, pLFile, &pStack,
										 FSNLBE_GREATER | FSNLBE_POSITION );
		}
	}

	/**------------------------------------------------------
	*** Position the pStack to where you should be.
	*** The parent element is pointing to the right block.
	***-----------------------------------------------------*/
	if( RC_OK( rc ))
	{
		/* Read in position block and position to current element */
		if( RC_OK(rc = FSGetBlock( pDb, pLFile, uiPosToBlk, pStack)))
		{
			pStack->uiCurElm = uiPosToElm;
			if( uiPosToBlk == uiLeftBlkAddr )
			{
				rc = FSAdjustStack( pDb, pLFile, pStack, FALSE);
			}
			/**--------------------------------------------
			***  This line must	be explained!
			***  We should really be replacing the original
			***  PKC buffer into what was there before this
			***  routine was called.  This works because
			***  delete/insert pairs call scanTo before the 
			***  insert.  
			***--------------------------------------------*/
			FSBlkBuildPKC( pStack, pStack->pKeyBuf, FSBBPKC_AT_CURELM );
		}
	}
	*pStackRV = pStack;
Exit:
	if( bReleaseLeft)
	{
		ScaReleaseCache( pLeftCache, FALSE);
	}
	if( bReleaseRight)
	{
		ScaReleaseCache( pRightCache, FALSE);
	}
	return( rc );
}

/****************************************************************************
Desc:	Move 1 or more elements into the bsCurElm location within a block.
		Everything MUST fit.  Will compress the element coming in as well
		as the element at bsCurElm.  This will also work if you are moving
		data down within the same block.
****************************************************************************/
RCODE FSBlkMoveElms(
	BTSK_p    	pStack,			/* Stack containing block to accept data*/
	FLMBYTE *  	pInsertElm,		/* Element(s) to insert into block */
	FLMUINT   	uiInsElmLen,	/* Length of the Element(s) */
	FLMBYTE *	pElmPkcBuf		/* PKC buffer for element if elm has PKC*/
	)
{
	FLMBYTE *	pBlk = pStack->pBlk;
	FLMUINT    	uiCurElm = pStack->uiCurElm;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMUINT    	uiBytesInPkc;

	FLMBYTE    	pInsertElmPckBuf[ BBE_PKC_MAX ];
	FLMUINT    	uiMovedKeyLen,  uiMovedPkc;
	FLMUINT    	uiTemp;
	FLMUINT    	uiNewBlkEnd;
	FLMUINT    	uiInsertElmKeyLen;
	FLMINT    	iDistanceToShiftDown;
	FLMUINT    	uiAreaToShiftDown;
	FLMBYTE		pPkcBuf[ BBE_PKC_MAX ];
	FLMUINT    	uiInsertElmPkc;
	FLMUINT    	uiInsertElmPkcLen;

	if( uiElmOvhd == BNE_DATA_OVHD)
	{
		if( (uiAreaToShiftDown = (pStack->uiBlkEnd - uiCurElm)) > 0)
		{
			shiftN( &pBlk[ uiCurElm ], uiAreaToShiftDown, uiInsElmLen);
		}
		f_memmove( &pBlk[ uiCurElm], pInsertElm, uiInsElmLen);
		pStack->uiBlkEnd += uiInsElmLen;
		UW2FBA( pStack->uiBlkEnd, &pBlk[ BH_BLK_END ]);
		goto Exit;
	}

	// ELSE Normal complex move with previos key count (PKC)

	/* Puts up to BBE_PKC_MAX bytes in pPkcBuf[] only */
	uiBytesInPkc = FSBlkBuildPKC( pStack, pPkcBuf, FSBBPKC_BEFORE_CURELM );

	/* Compute real pkc for element */
	uiInsertElmPkcLen = FSElmBuildPKC( pInsertElmPckBuf, pInsertElm, pElmPkcBuf, uiElmOvhd );

	uiMovedPkc = FSElmComparePKC( pPkcBuf, uiBytesInPkc, pInsertElmPckBuf, uiInsertElmPkcLen );

	/**----------------------------------------------------------------
	*** Compute how much area pInsertElm[] will take when moved,
	*** compute uiBlkEnd and move most of the element except the key
	***---------------------------------------------------------------*/
	uiInsertElmKeyLen  = (FLMUINT)(BBE_GET_KL( pInsertElm ));
	uiInsertElmPkc     = (FLMUINT)(BBE_GET_PKC( pInsertElm ));
	uiMovedKeyLen = (FLMUINT)(uiInsertElmKeyLen + uiInsertElmPkc - uiMovedPkc);
	iDistanceToShiftDown = (FLMINT)(uiInsElmLen + uiMovedKeyLen - uiInsertElmKeyLen);
	if( (uiAreaToShiftDown = (FLMUINT)(pStack->uiBlkEnd - uiCurElm)) > 0)
	{
		shiftN( &pBlk[ uiCurElm ], uiAreaToShiftDown, iDistanceToShiftDown );
	}

	uiNewBlkEnd = (FLMUINT)(pStack->uiBlkEnd + iDistanceToShiftDown);
	UW2FBA( uiNewBlkEnd, &pBlk[ BH_BLK_END ]);
	pStack->uiBlkEnd = uiNewBlkEnd;

	/* Move the first pInsertElm[] overhead values and key to where to be inserted*/
	FSSetElmOvhd( &pBlk[uiCurElm], uiElmOvhd, uiMovedPkc, uiMovedKeyLen, pInsertElm);

	/**--------------------------------------------------------------
	*** The tricky part is to move the key!
	*** The key could move in 2 parts pInsertElmPckBuf and pInsertElm[]
	***-------------------------------------------------------------*/

	if( uiMovedKeyLen + uiMovedPkc > BBE_PKC_MAX )		/* Key not entirely in pPkcBuf[] */
	{
		/* Move all that is in the pPkcBuf[] */
		f_memcpy( &pBlk[ uiCurElm + uiElmOvhd ],
						&pInsertElmPckBuf[ uiMovedPkc ],
						uiTemp = (FLMUINT)(BBE_PKC_MAX - uiMovedPkc) );

		/* Move the rest that is in the element */
		f_memmove( &pBlk[ uiCurElm + uiElmOvhd + uiTemp ],
						&pInsertElm[ uiElmOvhd + uiInsertElmKeyLen - (uiMovedKeyLen - uiTemp) ],
						uiMovedKeyLen - uiTemp );
	}
	else if( uiMovedKeyLen)
	{
		/* Entire key fits within the pPkcBuf[] */
		f_memcpy( &pBlk[ uiCurElm + uiElmOvhd ], 
						&pInsertElmPckBuf[ uiMovedPkc ], uiMovedKeyLen );
	}
	/* Move the rest of the element(s) over to the block */
	uiTemp = uiElmOvhd + uiInsertElmKeyLen;
	f_memmove( &pBlk[ uiCurElm + uiElmOvhd + uiMovedKeyLen ],	/* Better move HIGH-LOW */
						&pInsertElm[ uiTemp ], uiInsElmLen - uiTemp );

	/**---------------------------------------------------------------
	*** Now - if uiAreaToShiftDown has a value then position to the 
	*** old uiCurElm and try to compress more out of the element
	***--------------------------------------------------------------*/

	if( uiAreaToShiftDown)
	{
		pStack->uiCurElm = uiCurElm + iDistanceToShiftDown;
		/* Could change pStack->wBlkEnd */
		FSBlkCompressPKC( pStack, pPkcBuf );
	}
	pStack->uiCurElm = uiCurElm;	/* Points to start of inserted element(s) */

Exit:
	return( FERR_OK );
}

/****************************************************************************
Desc:		Build the PKC portion scanning in a block to but not
			including pStack->uiCurElm
Notes:	General routine for split and combine code.
			More setup needed if you are calling FSBtInsert() or FSBlkInsElm().
			This routine is fastest known way to build PKC from any element.
****************************************************************************/
FLMUINT FSBlkBuildPKC(
	BTSK_p			pStack,
	FLMBYTE *		pPkcBuf,
	FLMUINT			uiFlags
	)
{
	FLMUINT			uiMoveArea;
	FLMUINT			uiPkc;
	FLMUINT			uiTargetCurElm;
	FLMUINT			uiElmOvhd		= pStack->uiElmOvhd;
	FLMUINT			uiCurElm;
	FLMUINT			uiElmKeyLen;
	FLMBYTE *		pCurElm;

	if( uiElmOvhd == BNE_DATA_OVHD)
	{
		return 0;
	}
	/* Code below is fastest way to position to bsCurElm */
	uiTargetCurElm = pStack->uiCurElm;
	uiMoveArea = uiPkc = 0;
	uiCurElm = BH_OVHD;
	while( uiCurElm < uiTargetCurElm)
	{
FSBB_one_more_time:

		pCurElm = &pStack->pBlk[ uiCurElm ];
		uiElmKeyLen = (FLMUINT)(BBE_GET_KL( pCurElm));
		if( uiElmKeyLen )
		{
			/* Move minimum data over to the pPkcBuf[] */
			uiPkc = (FLMUINT)(BBE_GET_PKC( pCurElm ));
			uiMoveArea = ((uiPkc + uiElmKeyLen) > BBE_PKC_MAX)
						  ? (BBE_PKC_MAX - uiPkc)
						  : uiElmKeyLen;

			/* Most common uiMoveArea value for data records & numeric keys */
			if( uiMoveArea == 1)
				pPkcBuf[ uiPkc ] = pCurElm[ uiElmOvhd ];
			else if( uiMoveArea )
			{
				f_memmove( &pPkcBuf[ uiPkc ], &pCurElm[ uiElmOvhd ], uiMoveArea);
			}
		}

		if( pStack->uiBlkType == BHT_LEAF)
		{
			/* Goto the next element in the block */
			uiCurElm += BBE_GET_RL( pCurElm);
		}
		else if( BNE_IS_DOMAIN( pCurElm))			/* Non-leaf block */
		{
			uiCurElm += BNE_DOMAIN_LEN;
		}
		uiCurElm += uiElmOvhd + uiElmKeyLen;
	}

	/* Hit the target current element */
	if( uiFlags == FSBBPKC_AT_CURELM)
	{
		/* Copy the current element into the pPkcBuf[] */
		uiFlags = FSBBPKC_BEFORE_CURELM;
		goto FSBB_one_more_time;
	}		
	
	return( uiPkc + uiMoveArea);
}

/****************************************************************************
Desc:  	Compress out (or in) the PKC bytes in the current element
Notes: 	pTempPkcBuf passed in only to save pStack space.
****************************************************************************/
FSTATIC FLMINT FSBlkCompressPKC(
	BTSK_p			pStack,
	FLMBYTE *		pTempPkcBuf
	)
{
	FLMUINT			uiTempPkcLen;
	FLMUINT			uiPkcBufLen;
	FLMUINT			uiCurPkc, uiTruePkc;
	FLMINT			iCompressBytes = 0;
	FLMBYTE *		pCurElm;
	FLMBYTE			pPkcBuf[ BBE_PKC_MAX ];

	if( pStack->uiElmOvhd == BNE_DATA_OVHD)
		goto Exit;

	/* Build the PKC buffer from the block */
	uiTempPkcLen = FSBlkBuildPKC( pStack, pTempPkcBuf, FSBBPKC_BEFORE_CURELM);

	/**-------------------------------------------------------
	*** Position to the current element and build its own
	*** pkc buffer.  Compare and see if equals the current
	*** element's pkc value.  If not compress/decompress
	***------------------------------------------------------*/

	pCurElm = &pStack->pBlk[ pStack->uiCurElm ];
	uiCurPkc    = (FLMUINT)(BBE_GET_PKC( pCurElm));
	uiPkcBufLen = FSElmBuildPKC( pPkcBuf, pCurElm, pTempPkcBuf, pStack->uiElmOvhd);

	uiTruePkc = FSElmComparePKC( pTempPkcBuf, uiTempPkcLen, pPkcBuf, uiPkcBufLen);

	if( uiTruePkc != uiCurPkc)
	{
		FLMBYTE *	pBlk = pStack->pBlk;
		FLMUINT		uiCurElm  = pStack->uiCurElm;
		FLMUINT		uiBlkEnd  = pStack->uiBlkEnd;
		FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
		FLMUINT		keyLen    = (FLMUINT)(BBE_GET_KL(pCurElm));
		FLMUINT		uiTemp;

		if( uiTruePkc > uiCurPkc)
		{
			/* Need to compress out some more bytes */
			iCompressBytes = uiTruePkc - uiCurPkc;
			uiTemp = uiCurElm + uiElmOvhd + iCompressBytes;
			shiftN( &pBlk[ uiTemp ], (FLMUINT)(uiBlkEnd - uiTemp),
				(FLMINT)(-iCompressBytes));

			/* Reassign the element overhead */
			FSSetElmOvhd( pCurElm, uiElmOvhd,
										(FLMUINT)(uiCurPkc + iCompressBytes),
										(FLMUINT)(keyLen - iCompressBytes),
										pCurElm);
			uiBlkEnd -= iCompressBytes;
		}
		else /* uiTruePkc < uiCurPkc */
		{
			return( 0xFFFF); /* FERR_BTREE_ERROR Cannot ever happen right now */
		}
		UW2FBA( uiBlkEnd, &pBlk[ BH_BLK_END ]);
		pStack->uiBlkEnd = uiBlkEnd;
	}
Exit:
	return( iCompressBytes);
}

