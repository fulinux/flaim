//-------------------------------------------------------------------------
// Desc:	B-tree block splitting.
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fssplblk.cpp 12289 2006-01-19 14:56:21 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE FSBtResetStack(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK_p *		ppStack,
	FLMBYTE *	pElement,
	FLMUINT		uiElmLen);

FSTATIC RCODE FSMoveToNextBlk(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK_p *		ppStack,
  	FLMUINT		nextBlkNum,
	FLMBYTE *	pElement,
	FLMUINT		elmLen,
	FLMUINT		uiBlockSize,
	FLMUINT	*	blkNumRV,
	FLMUINT	*	curElmRV);

/***************************************************************************
Desc:		Split a block using different algorithms depending on context
Out:		ppStack points to current pStack element
Return:	RCODE - 0=OK or error code.
Notes:	goto's are added to insure proper cleanup and to help maintain flow.
*****************************************************************************/
RCODE FSBlkSplit(
	FDB *			pDb,
	LFILE *		pLFile,
	BTSK_p *		ppStack,
	FLMBYTE *	pElement,
	FLMUINT		elmLen)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiBlockSize = pDb->pFile->FileHdr.uiBlockSize;
	BTSK_p		pStack = *ppStack;
	FLMUINT    	oldCurElm = pStack->uiCurElm;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMUINT    	tempWord;
	FLMUINT    	elmKeyLen;
	FLMUINT    	prevKeyCnt;
	FLMUINT    	curElm;
	FLMBYTE *  	curElmPtr;
	FLMBYTE *  	pBlk;
	FLMUINT   	blkNum = pStack->uiBlkAddr;
	FLMUINT    	uiBlkEnd;
	BTSK      	newBlkStk;
	BTSK			nextBlkStk;
	FLMBYTE *  	newBlkPtr;
	FLMBYTE *	nextBlkPtr;
	FLMUINT   	newBlkNum;
	FLMUINT		nextBlkNum;
	FLMUINT   	blkNumRestore = 0;
	FLMUINT    	curElmRestore = 0;
	FLMBOOL    	bNewRootFlag;
	FLMBOOL		bDoubleSplit;
	DB_STATS *	pDbStats;

	f_yieldCPU();
	bNewRootFlag = bDoubleSplit = FALSE;
	FSInitStackCache( &newBlkStk, 1);
	FSInitStackCache( &nextBlkStk, 1);

	if( (pDbStats = pDb->pDbStats) != NULL)
	{
		LFILE_STATS *		pLFileStats;

		if( (pLFileStats = fdbGetLFileStatPtr( pDb, pLFile)) != NULL)
		{
			pLFileStats->bHaveStats =
			pDbStats->bHaveStats = TRUE;
			pLFileStats->ui64BlockSplits++;
		}
	}

	/***************************************************************************
	***	Algorithm designed Dec 30, 1991 - Implemented July 21, 1992
	***	Algorithm too difficult to implement when designed.  Implemented to save
	*** disk space, improve speed and reduce code by using shared routines.
	***
	***		If there is room to move data to the next block then do it
	***		and update the parent block with the new last element. Otherwise...
	***		Divide data from current block and next block into the new block
	***		(2/3 split).  Delete parent element and update parent /w 2 elements.
	***
	***************************************************************************
	***
	***	SPLIT THE BLOCK - old code before July 21, 1992
	***
	***	Allocate a new block, find the split point the divide the data
	***	between the two blocks.  Build a new non-leaf element to insert into
	*** the parent block.  Check for root splits and then recursively call
	*** FSBtInsert() in insert the created non-leaf element.
	***
	***************************************************************************/

	// Reset the stack if not a FULL_STACK, fast data inserts use NO_STACK
	if( pStack->uiFlags & NO_STACK )
	{
		if( RC_BAD( rc = FSBtResetStack( pDb, pLFile, &pStack, pElement, elmLen)))
		{
			goto Exit;
		}
	}
	
	// Log the block before modifying it

	if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack )))
	{
		goto Exit;
	}

	pBlk = pStack->pBlk;
	bNewRootFlag = BH_IS_ROOT_BLK(pBlk);

	if( (nextBlkNum = FB2UD( &pBlk[ BH_NEXT_BLK ])) != BT_END)
	{
		/**-----------------------------------------------------------------
		*** Try to move the elements from the current block to
		*** the next block while inserting the element.  If this
		*** succeeds than we are better off than spliting blocks.
		*** If succeeds then all block operations have been taken care of.
		***----------------------------------------------------------------*/

		if( RC_OK( rc = FSMoveToNextBlk( pDb, pLFile, &pStack, nextBlkNum,
													 pElement, elmLen, uiBlockSize,
													 &blkNumRestore, &curElmRestore)))
		{
			goto FSBlkSplit_position;
		}

		if( rc != FERR_BLOCK_FULL)
		{
			goto Exit;
		}
	}
	/**-----------------------------------------------------------------
	*** Initialize variables, create a new block and setup header.
	*** This is common stuff that will be done if you are
	*** splitting a RIGHT-MOST block or a middle block.
	*** Remember we could be working with leaf or non-leaf blks.
	***----------------------------------------------------------------*/

	if( RC_BAD( rc = ScaCreateBlock( pDb, pLFile, &newBlkStk.pSCache )))
	{
		goto Exit;
	}

	newBlkStk.pBlk = newBlkStk.pSCache->pucBlk;

	newBlkPtr  = GET_CABLKPTR( &newBlkStk);
	newBlkNum  = GET_BH_ADDR( newBlkPtr );
	pBlk     = pStack->pBlk;
	uiBlkEnd     = pStack->uiBlkEnd;
	UD2FBA( nextBlkNum, &newBlkPtr[ BH_NEXT_BLK ]);
	UD2FBA( blkNum,     &newBlkPtr[ BH_PREV_BLK ]);
	UD2FBA( newBlkNum,  &pBlk[ BH_NEXT_BLK ]);

	/* 
	Write over the root bit if present as set type.
	VISIT: Converting from 2x to 3x this is a good time to 
	convert all elements to 3x non-leaf element format.
	*/
	newBlkPtr[ BH_TYPE ] = pBlk[ BH_TYPE ] = (FLMBYTE)(BH_GET_TYPE( pBlk ));
	newBlkPtr[ BH_LEVEL ] = pBlk[ BH_LEVEL ];
	tempWord   = FB2UW( &pBlk[ BH_LOG_FILE_NUM ]);
	UW2FBA( tempWord, &newBlkPtr[ BH_LOG_FILE_NUM ]);
	UW2FBA( BH_OVHD,  &newBlkPtr[ BH_BLK_END ]);
	/**----------------------------------------------------------------------
	*** In both cases (middle or end split) if you split at the wrong
	*** place you will still not be able to fit the element[] in the block.
	*** The while loop will insure at least a split into two blocks where
	*** there may not be room to move any elements from the next block.
	*** NOTE: the max element may reach 793 bytes (640=Key,150=references).
	*** Exception:
	***    For 1K blocks, you may insert in the middle where the next (is last)
	***    element + elmLen > uiBlockSize AND oldCurElm + elmLen > uiBlockSize.
	***    In this case you must do double block split by inserting the element
	***    at the bottom of this routine after all splitting has been done.
	***---------------------------------------------------------------------*/

	pStack->uiCurElm = (nextBlkNum == BT_END)
						  ? (FFILE_MAX_FILL * 
								pDb->pFile->FileHdr.uiBlockSize / 100)
						  : ((uiBlockSize / 20) * 13);		/* Leave at least 65% full */

 	if( RC_BAD(rc = FSBtScanTo( pStack, NULL, 0,  0)))
	{
 		goto Exit;
	}
 	
	if( pStack->uiCmpStatus == BT_GT_KEY )	/* scanTo didn't move current elm's key */
	{
		/* Save the key in the pKeyBuf */
		curElmPtr = &pBlk[ pStack->uiCurElm ];
		if( pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			flmCopyDrnKey( pStack->pKeyBuf, curElmPtr);
		}
		else
		{
			elmKeyLen  = (FLMUINT)(BBE_GET_KL( curElmPtr ));
			if( elmKeyLen)										/* Copy key into pKeyBuf */
			{
				prevKeyCnt = (FLMUINT)(BBE_GET_PKC( curElmPtr ));
				f_memcpy( &pStack->pKeyBuf[ prevKeyCnt ],
							 &curElmPtr[ uiElmOvhd ], elmKeyLen );
			}
		}
	}											/* pStack->wCurElm is always > BH_OVHD */

	curElm = pStack->uiCurElm;

	/**-------------------------------------------------------------------
	***  Check to see if the new element will fit whereever it goes.
	***  Don't try to optimally place it because the next block may move
	***  stuff over.  curElm may be at BLK_END value or BLK_END-uiElmOvhd
	***  08/12/96 - Bug found moving the LEM only to the new block.
	***  This happened on a large data element with PKC of 2 where the
	***  split moved ONLY the LEM (last element marker). 
	***  Real special case where we have not dupped with stopmer or TST.
	***  Here are the specs...
	
	In Insert Element the insert failed on...
		if( wBlkEnd + uiElmOvhd + insertLen > uiBlockSize)
		
	and so this block split routine is called.

	so the original code...
		while( (curElm > oldCurElm ) && (curElm + elmLen + uiElmOvhd > uiBlockSize ))
	
	works find except on a worst case split.

	Given that:
		   	curElm + uiElmOvhd == wBlkEnd
		and   elmLen = insertLen + PCKLen

	Substitue the variables for the worst case split and we get...
	 while( ...	                   (wBlkEnd + insertLen + PCKLen > uiBlockSize) 
	 
	This resulted in a FALSE condition for our worst case split.
	Solving the two equations for two false conditions results in:
			PCKLen < uiElmOvhd

	Original code before 08/12/96...
	while( (curElm > oldCurElm ) && (curElm + elmLen + uiElmOvhd > uiBlockSize ))
	***------------------------------------------------------------------*/

	while( (curElm > oldCurElm ) 
	    && (curElm + elmLen + uiElmOvhd + uiElmOvhd > uiBlockSize ))
	{
		FSBtPrevElm( pDb, pLFile, pStack );
		curElm = pStack->uiCurElm;
	}

	newBlkStk.uiBlkAddr = newBlkNum;
	newBlkStk.pKeyBuf = pStack->pKeyBuf;
	FSBlkToStack( &newBlkStk );			/* Setup block pointers & values */
	newBlkStk.uiKeyBufSize = pStack->uiKeyBufSize;

	curElmPtr = &pBlk[ curElm ];			/* Point to split point */

	if( curElm == oldCurElm )				/* Decide which block to put newElm */
	{
		/* Decide whether to place the new element in the current or new block */
		/* Give preference to placing with the current block - fills better */
		if( (curElm + elmLen + uiElmOvhd < uiBlockSize )
		&&  ((uiBlkEnd - curElm) > uiElmOvhd))
			goto Addto_Current_Blk;			/* Place with the current block */

		if( uiBlkEnd > curElm)				/* Move if not at end of block */
			FSBlkMoveElms( &newBlkStk, curElmPtr,
								(FLMUINT)(uiBlkEnd - curElm), pStack->pKeyBuf);

		/* Set the block end in the current blocks header & restore values */
		pStack->uiBlkEnd = curElm;
		UW2FBA( curElm, &pBlk[ BH_BLK_END ]);
		blkNumRestore = newBlkNum;
		curElmRestore = BH_OVHD;			/* Setup to insert element*/
		newBlkStk.uiCurElm = curElmRestore;

		if( newBlkStk.uiBlkEnd + elmLen + uiElmOvhd > uiBlockSize)
		{
			bDoubleSplit = 1;					/* Double split - move element later */
		}
		else
		{
			FSBlkMoveElms( &newBlkStk, pElement, elmLen, NULL );
		}						
	}

	else if( curElm > oldCurElm )			/* Place new element in current blk */
	{
Addto_Current_Blk:
		/* First move stuff over to the new blk */
		FSBlkMoveElms( &newBlkStk, curElmPtr,
							(FLMUINT)(uiBlkEnd - curElm), pStack->pKeyBuf);

		/* Set the block end in the current blocks header */
		pStack->uiBlkEnd = curElm;
		UW2FBA( curElm, &pBlk[ BH_BLK_END ]);
		blkNumRestore = blkNum;
		curElmRestore = oldCurElm;	/* Setup to insert element*/
		pStack->uiCurElm = curElmRestore;
		FSBlkMoveElms( pStack, pElement, elmLen, NULL );
	}
	else if( curElm < oldCurElm)				/* Place new element in new block */
	{
		blkNumRestore = newBlkNum;				/* Don't mess with the order! */
		FSBlkMoveElms( &newBlkStk, curElmPtr,
							(FLMUINT)(oldCurElm-curElm), pStack->pKeyBuf);
		newBlkStk.uiCurElm = newBlkStk.uiBlkEnd;
		curElmRestore = newBlkStk.uiCurElm;

		// May not fit with 1K blocks - check for double split

		if( curElmRestore + elmLen + (uiBlkEnd - oldCurElm) + uiElmOvhd > uiBlockSize)
		{
			bDoubleSplit = 1;
		}
		else
		{
			FSBlkMoveElms( &newBlkStk, pElement, elmLen, NULL );
		}

		newBlkStk.uiCurElm = newBlkStk.uiBlkEnd;
		pStack->uiCurElm = oldCurElm;			/* Position for scanTo */

	 	if( RC_BAD(rc = FSBtScanTo( pStack, NULL, 0,  0)))
		{
	 		goto Exit;
		}
	
		// GWBUG (8/98) Check if oldCurElm < uiBlkEnd
		
		if( oldCurElm < uiBlkEnd)
		{
			FSBlkMoveElms( &newBlkStk, &pBlk[oldCurElm],
											(FLMUINT)(uiBlkEnd - oldCurElm),
											pStack->pKeyBuf );
		}

		// Set the block end in the current blocks header

		pStack->uiBlkEnd = curElm;
		UW2FBA( curElm, &pBlk[ BH_BLK_END ]);
	}

	// All done with the current block - unpin and set to dirty

	/**-------------------------------------------------------------
	*** All done moving data from current block to new block.
	*** if created new right most block
	***    check if new to create new root block and init new root (easy)
	*** else
	***    try to move stuff from the next block into the new block
	***------------------------------------------------------------*/

	if( nextBlkNum == BT_END )
	{
		FLMUINT 		uiLfNum = pLFile->uiLfNum;

		// We are done with block

		FSReleaseBlock( &newBlkStk, FALSE);

		// At the root?

		if( bNewRootFlag)
		{
			FLMBYTE		byType;

			/**--------------------------------------------------------------------
			***	Create a new root block
			*** Hitting 6 levels at 11.5 keys per block would be 2,000,000 blocks
			***-------------------------------------------------------------------*/

			if( pStack->uiLevel + 1 >= BH_MAX_LEVELS)
			{
				rc = RC_SET( FERR_BTREE_FULL);
				goto Exit;
			}

			// Set the default block type from what the type of the current root
			// or use the database version to decide.

			if( pLFile->uiLfType == LF_INDEX)
			{
				if( pLFile->pIxd->uiFlags & IXD_POSITIONING)
					byType = BHT_NON_LEAF_COUNTS + BHT_ROOT_BLK;
				else
					byType = BHT_NON_LEAF + BHT_ROOT_BLK;
			}
			else
				byType = BHT_NON_LEAF_DATA + BHT_ROOT_BLK;

			// Move all pStack elements down by doing a shift

			shiftN( (FLMBYTE *) pStack,
					  (FLMUINT)(sizeof(BTSK) * (pStack->uiLevel+1)),
					  (FLMINT)sizeof(BTSK));

			// Create a new block

			if( RC_BAD( rc = ScaCreateBlock( pDb, pLFile, &pStack->pSCache)))
			{
				goto Exit;
			}

			pBlk = pStack->pBlk = pStack->pSCache->pucBlk;

			// Set prev/next block addresses to BT_END

			UD2FBA( BT_END,  &pBlk[ BH_PREV_BLK ]);
			UD2FBA( BT_END,  &pBlk[ BH_NEXT_BLK ]);
			UW2FBA( BH_OVHD, &pBlk[ BH_BLK_END ]);

			// Set logical file number in the block header and block type

			UW2FBA( uiLfNum, &pBlk[ BH_LOG_FILE_NUM ]);

			pBlk[ BH_TYPE ] = byType;
			pBlk[ BH_LEVEL ] = (FLMBYTE)(++(pStack->uiLevel));
			pStack->uiBlkAddr = GET_BH_ADDR( pBlk );
			FSBlkToStack( pStack);
			pLFile->uiRootBlk = pStack->uiBlkAddr;

			// Always update the pLFile because level is incremented

			rc = flmLFileWrite( pDb, pLFile);
			pStack++;
			*ppStack = pStack;

			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}
	}
	else										/* move stuff from the right block */
	{											/* Remember that newBlk is still pinned */
		FLMINT		iBytesToMove;		/* 03/20/96 - made a signed value. */
		
		nextBlkStk.pKeyBuf = pStack->pKeyBuf;
		if( RC_BAD( rc = FSGetBlock( pDb, pLFile, nextBlkNum, &nextBlkStk)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = FSLogPhysBlk( pDb, &nextBlkStk)))
		{
			goto Exit;
		}

		nextBlkStk.uiKeyBufSize = pStack->uiKeyBufSize;
		nextBlkPtr = nextBlkStk.pBlk;
		UD2FBA( newBlkNum, &nextBlkPtr[ BH_PREV_BLK ]);	/* UPDATE PREV_BLK */
	
		// Try to move so that the two blocks have about the same free space

		iBytesToMove = (FLMINT) ((nextBlkStk.uiBlkEnd - newBlkStk.uiBlkEnd) / 2);
																
		if( iBytesToMove > 100)				/* Don't even bother if neg or <100 */
		{
			// Log the block before modifying it. Moved 03/20/96 from 5 above.*/
		
			nextBlkStk.uiCurElm = iBytesToMove + BH_OVHD;
			if( RC_BAD( rc = FSBtScanTo( &nextBlkStk, NULL, 0,  0)))
			{
				goto Exit;
			}

			while( (nextBlkStk.uiCurElm > BH_OVHD) && 
				(nextBlkStk.uiCurElm + newBlkStk.uiBlkEnd + 
				uiElmOvhd - BH_OVHD >= uiBlockSize))
			{
				(void)FSBtPrevElm( pDb, pLFile, &nextBlkStk );
			}
			
			/* GWUG 23654:
			**	Never try to move elements from the next block to the
			**	new block if we are positioned on the first element.
			** Added 9/10/96.
			*/

			// Never try to move if on LEM or at the end

			if( (nextBlkStk.uiCurElm > BH_OVHD) && 
				(nextBlkStk.uiCurElm + uiElmOvhd < nextBlkStk.uiBlkEnd))
			{
				FLMUINT tempEnd;
				
				/* 03/20/96 removed..if( rc == BT_GT_KEY ) */
				/* Always make sure the key is in the buffer. */
				/* Save the key in the pKeyBuf */

				curElmPtr = &nextBlkPtr[ nextBlkStk.uiCurElm ];
				if( pStack->uiBlkType != BHT_NON_LEAF_DATA)
				{
					elmKeyLen  = (FLMUINT)(BBE_GET_KL(  curElmPtr ));
					if( elmKeyLen)										/* Copy key into pKeyBuf */
					{
						prevKeyCnt = (FLMUINT)(BBE_GET_PKC( curElmPtr ));
						f_memcpy( &(nextBlkStk.pKeyBuf)[ prevKeyCnt ],
									&curElmPtr[ uiElmOvhd ], elmKeyLen );
					}
				}

				// Out 9/10/96 - Check is above.
				// if( nextBlkStk.wCurElm + uiElmOvhd < nextBlkStk.wBlkEnd)
			
				tempWord = nextBlkStk.uiCurElm;
				newBlkStk.uiCurElm = newBlkStk.uiBlkEnd;	/* Be sure to position */
				
				FSBlkMoveElms( &newBlkStk, &nextBlkPtr[ BH_OVHD ],
									(FLMUINT)(tempWord - BH_OVHD), NULL );
				
				// Move the elements in the next block DOWN expanding PKC

				tempEnd = nextBlkStk.uiBlkEnd;

				// Make sure uiBlkEnd is reality or move will not work!

				nextBlkStk.uiBlkEnd = nextBlkStk.uiCurElm = BH_OVHD;
				UW2FBA( BH_OVHD, &nextBlkPtr[ BH_BLK_END]);	/* Not really needed*/
			
				FSBlkMoveElms( &nextBlkStk, &nextBlkPtr[ tempWord ],
									(FLMUINT)(tempEnd - tempWord),
									nextBlkStk.pKeyBuf);

				if( (pStack-1)->uiBlkType == BHT_NON_LEAF_COUNTS)
				{
					if( RC_BAD( rc = FSUpdateAdjacentBlkCounts( pDb, pLFile, 
													pStack, &nextBlkStk)))
					{
						goto Exit;
					}
				}
			}
		}

		FSReleaseBlock( &newBlkStk, FALSE);
	}

	/**--------------------------------------------------------------------
	*** Insert the new last element in the current block replacing what
	*** is there.  Insert the last element from the new block.
	***
	*** All blocks should be dirty and unpined!  This means we have to read
	*** them in again.
	***-------------------------------------------------------------------*/

	if( RC_BAD(rc = FSGetBlock( pDb, pLFile, blkNum, pStack)))
	{
		goto Exit;
	}

	if( pStack->uiCurElm >= pStack->uiBlkEnd)
	{
		pStack->uiCurElm = curElm;
	}

	// Passing 0 means insert only.

	if( RC_BAD( rc = FSNewLastBlkElm( pDb, pLFile, &pStack,
									(nextBlkNum == BT_END ) ? 0: FSNLBE_LESS )))
	{
		*ppStack = pStack;
		goto Exit;
	}

	if( RC_BAD( rc = FSAdjustStack( pDb, pLFile, pStack, TRUE)))
	{
		if( rc != FERR_BT_END_OF_DATA )
		{
			goto Exit;
		}
	}

	// Parent is positioned to the nextBlk element.

	if( RC_BAD( rc = FSGetBlock( pDb, pLFile, newBlkNum, pStack)))
	{
		goto Exit;
	}

	if( (nextBlkNum == BT_END) && !bNewRootFlag)
	{
		FLMUINT		uiNewRefCount;
		FLMUINT		uiOldRefCount;
		FLMBYTE *	pTmpElement;

		// Only update the counts if the inserting a key and not replacing.

		if( (pStack-1)->uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			if( RC_BAD( rc = FSBlockCounts( pStack, BH_OVHD,
					pStack->uiBlkEnd, NULL, NULL, &uiNewRefCount)))
			{
				goto Exit;
			}
		}

		pStack--;

		// Modify the parent last element marker (LEM) to point to
		// the new last block.  THEN delete the previous element that
		// also pointer to the new last block.

		// Read the parent block

		if( RC_BAD( rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
		{
			return( rc );
		}

		if( RC_BAD( rc = FSLogPhysBlk( pDb, pStack)))
		{
			return( rc );
		}

		// Change where element points to in the last element marker (LEM)

		pTmpElement = pStack->pBlk + pStack->uiCurElm;
		FSSetChildBlkAddr( pTmpElement, newBlkNum, pStack->uiElmOvhd);

		if( pStack->uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			uiOldRefCount = FB2UD( &pTmpElement[ BNE_CHILD_COUNT]);
			if( RC_BAD( rc = FSChangeBlkCounts( pDb, pStack,
				(FLMINT) (uiNewRefCount - uiOldRefCount))))
			{
				goto Exit;
			}
			UD2FBA( uiNewRefCount, &pTmpElement[ BNE_CHILD_COUNT]);
		}
		pStack++;
	}
	else
   {
		// Inserts the new element into the tree.
		rc = FSNewLastBlkElm( pDb, pLFile, &pStack, 0);
	}

FSBlkSplit_position:

	*ppStack = pStack;

	// Read in block - should be positioned to the current element inserted

	if( RC_OK(rc))
	{
		// Parent element is on the newBlock - see if you need to back up

		if( blkNumRestore == blkNum )
		{
			if( RC_BAD( rc = FSAdjustStack( pDb, pLFile, pStack, FALSE)))
			{
				if( rc != FERR_BT_END_OF_DATA)
				{
					goto Exit;
				}
			}
		}

		if( RC_OK( rc = FSGetBlock( pDb, pLFile, blkNumRestore, pStack)))
		{
			// Set up the key buffer (pKeyBuf) to be correct for future inserts

			pStack->uiCurElm = curElmRestore;
			if( RC_BAD(rc = FSBtScanTo( pStack, NULL, 0,  0)))
			{
				goto Exit;
			}

			if( bDoubleSplit)
			{
				// Now insert the element if flag set
				// This will cause another split.
				// RECURSIVE CALL

				rc = FSBlkSplit( pDb, pLFile, ppStack, pElement, elmLen);
			}
		}
   }

Exit:

	FSReleaseBlock( &newBlkStk, FALSE);
	FSReleaseBlock( &nextBlkStk, FALSE);

	return( rc);
}

/***************************************************************************
Desc:		Reset the pStack to set up for a block split
Return:	Offset into split point in block 1 (pCache)
Out:		Offset where to split the block
Notes:	Must guarantee at least 1 element in both blocks.
*****************************************************************************/
FSTATIC RCODE FSBtResetStack(
	FDB *		  	pDb,				/* Pointer to database DBC structure. */
	LFILE *	  	pLFile,			/* Logical file definition */
	BTSK_p *	  	ppStack,			/* Stack of variables for each level */
	FLMBYTE *	pElement,		/* The input element to insert */
	FLMUINT	  	elmLen)			/* Length of the element */
{
	RCODE			rc;
	BTSK_p		pStack = *ppStack;					/* Stack holding all state info */
	FLMUINT		oldPKC = pStack->uiPKC;/* Save old PKC value */
	FLMUINT		oldPvElmPKC = pStack->uiPrevElmPKC;/* Save old prev elm PKC value */
	FLMUINT		oldBlock = pStack->uiBlkAddr;	/* Save old block number */
	FLMUINT		oldCurElm = pStack->uiCurElm;/* Save old current element value */
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;

	if( RC_BAD(rc = FSBtSearch( pDb, pLFile,
										 &pStack,&pElement[uiElmOvhd],
										 (elmLen-uiElmOvhd),0)))
		return( rc );

	/* In case of continuation elements, parse to matching curElm */
	while( (oldBlock != pStack->uiBlkAddr ) &&
			 (oldCurElm != pStack->uiCurElm ))
	{
		if( (rc = FSBtNextElm( pDb, pLFile, pStack )) == FERR_BT_END_OF_DATA)
		{
			return( RC_SET( FERR_BTREE_ERROR) );
		}
		else if( RC_BAD( rc))
		{
			return( rc);
		}
	}

	// Reset original PKC values

	pStack->uiPKC = oldPKC;
	pStack->uiPrevElmPKC = oldPvElmPKC;
	*ppStack = pStack;
	pStack->uiFlags = FULL_STACK;

	return( FERR_OK);
}

/****************************************************************************
Desc:  	Try to move elements between two blocks while inserting an element
			Set up pStack to point to where element was inserted.
Out:		pStack updated to point to current element
Return:	RCODE, FERR_OK - OK, FAILURE - cannot move, else error number
Notes: 	Stack could point to leaf or non-leaf nodes.  Parent elements
			should point to the next block so caller can fixup parent element.
****************************************************************************/
FSTATIC RCODE FSMoveToNextBlk(
	FDB *		  	pDb,
	LFILE *	  	pLFile,
	BTSK_p *	  	ppStack,
	FLMUINT    	nextBlkNum,
	FLMBYTE *   pElement,
	FLMUINT     elmLen,
	FLMUINT     uiBlockSize,
	FLMUINT *	blkNumRV,
	FLMUINT  *	curElmRV)
{
	RCODE     	rc = FERR_OK;
	BTSK_p		pStack = *ppStack;
	FLMUINT  	uiBlkEnd = pStack->uiBlkEnd;
	FLMUINT   	oldCurElm = pStack->uiCurElm;	// Original current element value
	FLMBYTE *  	curElmPtr;
	BTSK      	nextBlkStk;							// Used for the move element routine
	FLMUINT    	nextBlkFreeBytes;
	FLMUINT    	elmKeyLen;
	FLMUINT    	prevKeyCnt;
	FLMUINT 	  	curElm;
	FLMUINT		uiElmOvhd = pStack->uiElmOvhd;
	FLMBOOL		bInsertInCurrentBlock;
	FLMBYTE *  	pBlk  = pStack->pBlk;

	FSInitStackCache( &nextBlkStk, 1);
	nextBlkStk.pKeyBuf = pStack->pKeyBuf;

	if( RC_BAD( rc = FSGetBlock( pDb, pLFile, nextBlkNum, &nextBlkStk)))
	{
		goto Exit;
	}

	nextBlkFreeBytes = (FLMUINT)(uiBlockSize - nextBlkStk.uiBlkEnd - uiElmOvhd);
	pStack->uiCurElm = (uiBlkEnd - (nextBlkFreeBytes / 2));

	if( RC_BAD(rc = FSBtScanTo( pStack, NULL, 0,  0)))
	{
		goto Exit;
	}

	rc = (pStack->uiCmpStatus == BT_END_OF_DATA) ? FERR_BT_END_OF_DATA : FERR_OK;

	for( ; RC_OK(rc); rc = FSBlkNextElm( pStack))
	{
		// The current element is positioned to mininum split point.
		// Keep testing till at end of block or split will fit
		// within both blocks while still adding the element to insert.

		// Save the key in the pKeyBuf so can move entire element

		curElmPtr  = CURRENT_ELM( pStack );

		if( pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			prevKeyCnt = elmKeyLen = 0;
		}
		else
		{
			prevKeyCnt = (FLMUINT)(BBE_GET_PKC( curElmPtr));
			elmKeyLen  = (FLMUINT)(BBE_GET_KL(  curElmPtr));
		}

		if( elmKeyLen)									// Copy key into pKeyBuf.
		{
			f_memcpy( &pStack->pKeyBuf[ prevKeyCnt ],
			          &curElmPtr[ uiElmOvhd ], elmKeyLen );
		}

		// Will element be inserted into the current block?

		if( oldCurElm <= (curElm = pStack->uiCurElm))
		{
			// Insert the element in the current block - could be at the end 

			if( curElm + elmLen + uiElmOvhd > uiBlockSize )
			{
				rc = FERR_BT_END_OF_DATA;	
				break;
			}

			// Left fits - try right block - could fail because of pkc value/

			if( prevKeyCnt + (uiBlkEnd - curElm) >= nextBlkFreeBytes)
			{
				continue;								// Doesn't fit - try again
			}

			bInsertInCurrentBlock = TRUE;
		}
		else									
		{
			// Moving elements from current block so no need to check it
			// Cannot remember why I put BBE_PKC_MAX in the line below 

			if( elmLen + prevKeyCnt + BBE_PKC_MAX +
					(uiBlkEnd - curElm) >= nextBlkFreeBytes)
			{
				continue;
			}

			bInsertInCurrentBlock = FALSE;
		}

		if( RC_BAD( rc = FSLogPhysBlk( pDb, &nextBlkStk)))
		{
			goto Exit;
		}

		if( bInsertInCurrentBlock)
		{
			FSBlkMoveElms( &nextBlkStk, curElmPtr,
								(FLMUINT)(uiBlkEnd - curElm), pStack->pKeyBuf);
			pStack->uiBlkEnd = curElm;
			UW2FBA( curElm, &pBlk[ BH_BLK_END ]);
			*curElmRV = oldCurElm;
			pStack->uiCurElm = (*curElmRV);
			FSBlkMoveElms( pStack, pElement, elmLen, NULL );
			*blkNumRV = pStack->uiBlkAddr;
		}
		else
		{
			FSBlkMoveElms( &nextBlkStk, curElmPtr,
							(FLMUINT)(uiBlkEnd - curElm), pStack->pKeyBuf);
			pStack->uiBlkEnd = curElm;
			UW2FBA( curElm, &pBlk[ BH_BLK_END ]);
			*curElmRV = (FLMUINT)(BH_OVHD + prevKeyCnt + (oldCurElm - curElm));
			nextBlkStk.uiCurElm = (*curElmRV);
			FSBlkMoveElms( &nextBlkStk, pElement, elmLen, NULL );
			*blkNumRV = nextBlkNum;
		}

		if( pLFile->pIxd && (pLFile->pIxd->uiFlags & IXD_POSITIONING))
		{
			if( RC_BAD( rc = FSUpdateAdjacentBlkCounts( pDb, pLFile, 
											pStack, &nextBlkStk)))
			{
				goto Exit;
			}
		}

		break;
	}

	// If cannot move then return

	if( RC_BAD(rc))
	{
		if( rc == FERR_BT_END_OF_DATA)
		{
			// Restore old current element
			pStack->uiCurElm = oldCurElm;
			rc = RC_SET( FERR_BLOCK_FULL);
		}

		goto Exit;
	}

	// Fix up the elements in the parent block pointing to the new
	// last element in the current block.  REMEMBER - pStack may change
	// on you!

	rc = FSNewLastBlkElm( pDb, pLFile, ppStack, FSNLBE_LESS | FSNLBE_POSITION);

Exit:

	FSReleaseBlock( &nextBlkStk, FALSE);
	return( rc);
}
