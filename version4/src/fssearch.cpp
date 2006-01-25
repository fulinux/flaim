//-------------------------------------------------------------------------
// Desc:	B-tree searching.
// Tabs:	3
//
//		Copyright (c) 1990-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fssearch.cpp 12321 2006-01-19 15:55:00 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMUINT FSKeyCmp( 
	BTSK_p		pStack,
	FLMBYTE *	key, 
	FLMUINT		uiKeyLen,
	FLMUINT 		drnDomain);

	
/***************************************************************************
Desc:		Search the b-tree for a matching key.  Set up stack for updates.
			If index search using dinDomain.  dinDomain MUST be 0 for data recs.
			dinDomain is ONE more than the expected target for index ref sets.
			This is because of the compare routine must have dinDomain contain
			a value in order to test the domain.

In:		stackRV - points to 'n' level stack, returns leaf stack level
			key - points to key requested to search for
			keylen - length of the key
			dinDomain - domain of index key requested

Out:		Stack set up for each level & stack[level].bsStatus set to...
			BT_EQ_KEY (0) if equal key was found
			BT_GT_KEY (1) if greater than key was found
			BT_END_OF_DATA (0xFFFF) if marker was hit before eq or gt key found

Return:	RCODE - FERR_OK,FERR_OLD_VIEW or other error
Notes:	All buffers for the stack (pKeyBuf and BlkBuf must have been
			allocated before the call.
Notes:	btSearch responsible for bsLevel & bsBlock in stack structure.
*****************************************************************************/
RCODE FSBtSearch(
	FDB_p			pDb,
	LFILE *		pLFile,				/* Logical file definition 					*/
	BTSK_p *		pStackRV,			/* Stack of variables for each level		*/
	FLMBYTE *	key,					/* The input key to search for 				*/
	FLMUINT		keyLen,				/* Length of the key (not null term)		*/
	FLMUINT		dinDomain			/* INDEXES ONLY - lower bounds of din		*/
	)
{
	RCODE			rc = FERR_OK;		// Technically, don't need to set, but we
	BTSK_p		pStack = *pStackRV;
	FLMBYTE *	pKeyBuf = pStack->pKeyBuf;// Used to set key buf on each btsk.
	FLMUINT		uiBlkAddr;
	FLMUINT		uiKeyBufSize;
	LFILE			TmpLFile;
											// don't want a maintenance problem.

	uiKeyBufSize = (pLFile->uiLfType == LF_INDEX) ? MAX_KEY_SIZ : DIN_KEY_SIZ;
	
	/* Get the correct root block specified in the LFILE. */

	if( RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}
		goto Exit;
	}

	/**----------------------------------------------
	***  MAIN LOOP
	***    Read each block going down the b-tree.
	***    Save state information in the pStack[].
	***---------------------------------------------*/
	
	for(;;)
	{
		pStack->uiFlags = FULL_STACK;
		pStack->uiKeyBufSize = uiKeyBufSize;
		if( pStack->uiBlkType != BHT_NON_LEAF_DATA)
		{
			rc = FSBtScan( pStack, key, keyLen, dinDomain);
		}
		else
		{
			rc = FSBtScanNonLeafData( pStack, keyLen == 1 
			? (FLMUINT) *key : (FLMUINT) byteToLong( key));
		}
		if( RC_BAD( rc))
		{
			goto Exit;
		}

		// VISIT: Verify byLevel for cyclic loops. 
		
		if( !pStack->uiLevel)							// Leaf level?
			break;										// Done

		uiBlkAddr = FSChildBlkAddr( pStack );

		pStack++;											// Next btree pStack level.
		pStack->pKeyBuf = pKeyBuf;					// need to set for each pStack.

		if( RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack )))
			goto Exit;
	}
	*pStackRV = pStack;									// Set the stack return value.

Exit:
	return( rc);
}
	
/***************************************************************************
Desc:		Search the right-most end of the b-tree.  Set up stack for updates.
Out:		Stack set up for each level & stack[level].bsStatus set to...
			BT_EQ_KEY (0) if equal key was found
			BT_GT_KEY (1) if greater than key was found
			BT_END_OF_DATA (0xFFFF) if marker was hit before eq or gt key found
*****************************************************************************/
RCODE FSBtSearchEnd(
	FDB_p			pDb,
	LFILE *		pLFile,				/* Logical file definition 					*/
	BTSK_p *		pStackRV,			/* Stack of variables for each level		*/
	FLMUINT		uiDrn)				/* Used to position and setup for update	*/
{
	RCODE			rc = FERR_OK;		// Technically, don't need to set, but we
	BTSK_p		pStack = *pStackRV;
	FLMBYTE *	pKeyBuf = pStack->pKeyBuf;// Used to set key buf on each btsk.
	FLMBYTE		key[ DIN_KEY_SIZ + 4 ];/* Key buffer pointed to by stack */
	FLMUINT		uiBlkAddr;
	LFILE			TmpLFile;

	/* Get the correct root block specified in the LFILE. */
	if( RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}
		goto Exit;
	}

	longToByte( uiDrn, key);
	for(;;)
	{
		pStack->uiFlags = FULL_STACK;
		pStack->uiKeyBufSize = DIN_KEY_SIZ;

		// Remove all scanning from non-leaf data blocks (both formats).
		if( pStack->uiLevel)
		{
			pStack->uiCurElm = pStack->uiBlkEnd;	// Position past last element 
			FSBtPrevElm( pDb, pLFile, pStack );	// Build full key in pKeyBuf[]
		}
		else
		{
			if( pStack->uiBlkType != BHT_NON_LEAF_DATA)
			{
				rc = FSBtScan( pStack, key, DIN_KEY_SIZ, 0);
			}
			else
			{
				rc = FSBtScanNonLeafData( pStack, uiDrn);
			}
			if( RC_BAD( rc))
				goto Exit;
		}
		if( !pStack->uiLevel)							// Leaf level?
			break;											// Done

		uiBlkAddr = FSChildBlkAddr( pStack );
		pStack++;											// Next btree pStack level.
		pStack->pKeyBuf = pKeyBuf;						// need to set for each pStack.

		if( RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack )))
			goto Exit;
	}
	*pStackRV = pStack;									// Set the stack return value.

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Returns the root block of a passed-in LFILE
****************************************************************************/
RCODE FSGetRootBlock(
	FDB_p				pDb,
	LFILE **			ppLFile,				/* Logical file definition 					*/
	LFILE *			pTmpLFile,
	BTSK_p			pStack)				/* Stack of variables for each level		*/
{
	RCODE				rc = FERR_OK;
	LFILE *			pLFile = *ppLFile;
	FLMUINT			uiBlkAddr;
	FLMBOOL			bRereadLFH = FALSE;

	/*
	Make Sure this is the correct root block in the LFILE area.
	If not then read in the LFH structure and try again.
	It would be nice to have a routine that reads only root blocks.
	DSS: Added check for uiBlkAddr >= pDb->Loghdr.uiLogicalEOF
	because the pLFile could have a root block address of an aborted
	update transaction where the root block has not yet been fixed up by
	the aborting transaction (in a shared environment).
	*/
	
	if( ((uiBlkAddr = pLFile->uiRootBlk) == BT_END) ||
		 (uiBlkAddr >= pDb->LogHdr.uiLogicalEOF))
	{
		bRereadLFH = TRUE;
	}
	else if( RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
	{
		if( rc == FERR_DATA_ERROR || (rc == FERR_OLD_VIEW && !pDb->uiKilledTime))
		{
			bRereadLFH = TRUE;
			pStack->uiBlkAddr = BT_END;
		}
		else
		{
			goto Exit;
		}
	}
	else	/* Check for valid root block - Root Flag and Logical file number */
	{
		FLMBYTE *		pBlk = pStack->pBlk;

		if( !(BH_IS_ROOT_BLK( pBlk)) 
		 ||  (pLFile->uiLfNum != FB2UW( &pBlk[ BH_LOG_FILE_NUM ])))
		{ 
			bRereadLFH = TRUE;
			FSReleaseBlock( pStack, FALSE);
			pStack->uiBlkAddr = BT_END;
		}
	}

	/* Reread the LFH from disk if we do not have the root block */
	if( bRereadLFH)
	{
		/*
		If we are in a read transaction and we are using an HFSHARE
		structure that is shared among multiple threads, copy the LFILE
		structure so that we don't mess up a thread that may be doing
		an update.  The only members of the LFILE that might be
		different after the FSLFileRead are as follows:

			uiRootBlk and VER11 elements are: byLevel, uiNextDrn, and uiLastBlk.
		
		None of these members are needed outside this routine during a
		read transaction, so it is OK to use a temporary LFILE inside
		this routine.
		*/
		
		if( flmGetDbTransType( pDb) == FLM_READ_TRANS)
		{
			f_memcpy( pTmpLFile, pLFile, sizeof( LFILE));
			pLFile = pTmpLFile;
		}

		if( RC_BAD( rc = flmLFileRead( pDb, pLFile)))
		{
			goto Exit;
		}

		/* If there is no root block, return right away */

		if( (uiBlkAddr = pLFile->uiRootBlk) == BT_END)
		{
			/*
			The caller of FSGetRootBlock is expected to check for and
			handle FERR_NO_ROOT_BLOCK.  It should NEVER be returned
			to the application.

			NOTE: Checking for BT_END_OF_DATA will not work in every
			case to check for no root block because it is not always
			initialized before calling FSGetRootBlock, so it could
			have garbage in it if we don't end up going through this
			code path.
			*/

			rc = RC_SET( FERR_NO_ROOT_BLOCK);
			pStack->uiCmpStatus = BT_END_OF_DATA;
			pStack->uiBlkAddr = BT_END;
			goto Exit;
		}
		if( RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlkAddr, pStack)))
			goto Exit;		/* Usually returns OLD_VIEW or BLOCK_MODIFIED */
	}

Exit:
	*ppLFile = pLFile;
	return( rc);
}

/***************************************************************************
Desc:		Scan a b-tree block for a matching key at any b-tree block level.
Notes:	This routine has been optimized for speed.  Routine calls
			have been taken out in order to improve the performance.
*****************************************************************************/
RCODE FSBtScan(
	BTSK_p			pStack,					// [in/out] Stack of variables for each level 
	FLMBYTE *		pSearchKey,				// The input key to search for 
	FLMUINT			uiSearchKeyLen,		// Length of the key (not null terminated) 
	FLMUINT			dinDomain)				// INDEXES ONLY - lower bounds of din
{
	RCODE				rc = FERR_OK;// MUST initialize.
	FLMBYTE *		pCurElm;					// Points to the current element.
	FLMBYTE *		pBlk;						// Points to the cache block.
	FLMBYTE *		pKeyBuf;					// Points to pStack->pKeyBuf (optimization).
	FLMBYTE *		pElmKey;					// Points to the key within the element.
	FLMUINT			uiRecLen = 0;			// Length of the record portion.
	FLMUINT			uiPrevKeyCnt;			// Number left end bytes compressed
	FLMUINT			uiElmKeyLen;			// Length of the current element's key portion
	FLMUINT			uiBlkType;				// B-tree block type - Leaf or non-leaf.
	FLMUINT			uiElmOvhd;				// Number bytes overhead for element.
	FLMUINT			uiBytesMatched;		// Number of bytes matched with pSearchKey 
													// and cur element. Related to uiPrevKeyCnt.
	uiBlkType = pStack->uiBlkType;
	flmAssert( uiBlkType != BHT_NON_LEAF_DATA);

	// Initialize stack variables for possibly better performance. 
	pKeyBuf = pStack->pKeyBuf;
	pBlk  = pStack->pBlk;
	uiElmOvhd = pStack->uiElmOvhd;
	pStack->uiCurElm = BH_OVHD;
	pStack->uiKeyLen = pStack->uiPKC = pStack->uiPrevElmPKC = 0;
	uiBytesMatched = 0;

	for( ;;)									// while bsCurElm < bsBlkEnd 
	{
		pCurElm = &pBlk[ pStack->uiCurElm];
		uiElmKeyLen = BBE_GETR_KL( pCurElm );

		// Read in RAW mode - doesn't do all bit checking 
		
		if( (uiPrevKeyCnt = (BBE_GETR_PKC( pCurElm ))) > BBE_PKC_MAX)
		{
			uiElmKeyLen += (uiPrevKeyCnt & BBE_KL_HBITS) << BBE_KL_SHIFT_BITS;
			uiPrevKeyCnt &= BBE_PKC_MAX;
		}

		// Should not have a non-zero PKC if we are on the first element
		// of a block

		if( uiPrevKeyCnt && pStack->uiCurElm == BH_OVHD)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}

		// Get the record portion length when on the leaf blocks.
		
		if( uiBlkType == BHT_LEAF)
		{
			uiRecLen = BBE_GET_RL( pCurElm );
		}

		pStack->uiPrevElmPKC = pStack->uiPKC;
		
		// The zero length key is the terminating element in a right-most block.

		if( (pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen) == 0)
		{
			pStack->uiPrevElmPKC = f_min( uiBytesMatched, BBE_PKC_MAX);
			pStack->uiPKC = 0;
			pStack->uiCmpStatus = BT_END_OF_DATA;
			goto Exit;
		}

		// Handle special case of left-end compression maxing out.
		if( uiPrevKeyCnt == BBE_PKC_MAX && BBE_PKC_MAX < uiBytesMatched)
		{
			uiBytesMatched = BBE_PKC_MAX;
		}

		// Check out this element to see if the key matches.
		if( uiPrevKeyCnt == uiBytesMatched)
		{
			pElmKey = &pCurElm[ uiElmOvhd ];
			for(;;)
			{
				// All bytes of the search key are matched?
				if( uiBytesMatched == uiSearchKeyLen)
				{
					pStack->uiPKC = f_min( uiBytesMatched, BBE_PKC_MAX);
					// Build pKeyBuf with the search key because it matches.
					// Current key is either equal or greater than search key.

					if( uiSearchKeyLen < pStack->uiKeyLen)
					{
						f_memcpy( &pKeyBuf[ uiSearchKeyLen], pElmKey, 
							pStack->uiKeyLen - uiSearchKeyLen);
						pStack->uiCmpStatus = BT_GT_KEY;
					}
					else
					{
						if( dinDomain)
						{
							FLMBYTE *	pCurRef = pCurElm;
							if( (dinDomain - 1) < 
									FSGetDomain( &pCurRef, (FLMBYTE)uiElmOvhd))
							{
								// Keep going...
								goto Next_Element;
							}
						}
						pStack->uiCmpStatus = BT_EQ_KEY;
					}
					f_memcpy( pKeyBuf, pSearchKey, uiSearchKeyLen);
					goto Exit;
				}
				// .. else matches all the bytes in the element key.
				if( uiBytesMatched == pStack->uiKeyLen)
				{
					pStack->uiPKC = f_min( uiBytesMatched, BBE_PKC_MAX);
					// Need an outer break call here - forced to do a goto.
					goto Next_Element;
				}

				// Compare the next byte in the search key and element
				if( pSearchKey[ uiBytesMatched] != *pElmKey)
					break;
				uiBytesMatched++;
				pElmKey++;
			}
			pStack->uiPKC = f_min( uiBytesMatched, BBE_PKC_MAX);

			// Check if we are done comparing, if so build pKeyBuf[].
			if( pSearchKey[ uiBytesMatched] < *pElmKey)
			{
				if( uiBytesMatched)
				{
					f_memcpy( pKeyBuf, pSearchKey, uiBytesMatched);
				}
				f_memcpy( &pKeyBuf[ uiBytesMatched], pElmKey, 
								pStack->uiKeyLen - uiBytesMatched);
				pStack->uiCmpStatus = BT_GT_KEY;
				goto Exit;
			}
		}
		else if( uiPrevKeyCnt < uiBytesMatched)
		{
			// Current key > search key.  Set pKeyBuf and break out.
			pStack->uiPKC = uiPrevKeyCnt;
			if( uiPrevKeyCnt)
			{
				// VISIT: Call a small memcpy here.
				f_memcpy( pKeyBuf, pSearchKey, uiPrevKeyCnt);
			}
			f_memcpy( &pKeyBuf[ uiPrevKeyCnt], &pCurElm[ uiElmOvhd], uiElmKeyLen);
			pStack->uiCmpStatus = BT_GT_KEY;
			goto Exit;
		}
		// else the key is less than the search key (uiPrevKeyCnt > uiBytesMatched).

Next_Element:

		/* Position to the next element */
		pStack->uiCurElm += uiElmKeyLen + ((uiBlkType == BHT_LEAF )
						? (BBE_KEY + uiRecLen)
						: (BNE_IS_DOMAIN(pCurElm) ? (BNE_DOMAIN_LEN + uiElmOvhd)
														  : uiElmOvhd));
		
		// Most common check first.
		if( pStack->uiCurElm < pStack->uiBlkEnd)
			continue;

		if( pStack->uiCurElm == pStack->uiBlkEnd)
		{
			// On the equals conditition it may be OK in some very special cases.
			pStack->uiCmpStatus = BT_END_OF_DATA;
			goto Exit;
		}
		// Marched off the end of the block - something is corrupt.
		rc = RC_SET( FERR_CACHE_ERROR);
		goto Exit;
	}

Exit:
	return( rc);
}

/***************************************************************************
Desc:		Binary search into a non-leaf data record block.
In:   	Stack, key, keyLen, dinDomain (value for indexes only else 0)
Out:		Stack set up - bsStatus set to following...
			BT_EQ_KEY (0) if equal key was found
			BT_GT_KEY (1) if greater than key was found
			BT_END_OF_DATA (0xFFFF) if marker was hit before eq or gt key found
Return:	RCODE - FERR_OK
*****************************************************************************/
RCODE FSBtScanNonLeafData(
	BTSK_p			pStack,
	FLMUINT			uiDrn)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pBlk = pStack->pBlk;					// Points to the cache block.
	FLMUINT		uiLow = 0,
					uiMid, 
					uiHigh = ((pStack->uiBlkEnd - BH_OVHD) >> 3) - 1;
	FLMUINT		uiTblSize = uiHigh;
	FLMUINT		uiCurDrn;

	pStack->uiCmpStatus = BT_GT_KEY;
	for(;;)
	{
		uiMid = (uiLow + uiHigh) >> 1;		// (uiLow + uiHigh) / 2
		
		uiCurDrn = byteToLong( &pBlk[ BH_OVHD + (uiMid << 3)]);
		if( uiCurDrn == 0)
		{
			// Special case - at the end of a rightmost block.
			pStack->uiCmpStatus = BT_EQ_KEY;	//BT_END_OF_DATA;
			break;
		}
		if( uiDrn == uiCurDrn)
		{
			// Remember a data record can span multiple blocks (same DRN).
			while( uiMid)
			{
				uiCurDrn = byteToLong( &pBlk[ BH_OVHD + ((uiMid - 1) << 3)]);
				if( uiDrn != uiCurDrn)
					break;
				uiMid--;
			}
			pStack->uiCmpStatus = BT_EQ_KEY;
			break;
		}

		// Down to one item if too high then position to next item.
		if( uiLow >= uiHigh)
		{
			if( (uiDrn > uiCurDrn) && uiMid < uiTblSize)
				uiMid++;
			break;
		}

		// If too high then try lower section
		if( uiDrn < uiCurDrn)
		{
			// First item too high?
			if( uiMid == 0)
				break;
			uiHigh = uiMid - 1;
		}
		else	// try upper section because mid value is too low.
		{
			if( uiMid == uiTblSize)
			{
				uiMid++;
				pStack->uiCmpStatus = BT_END_OF_DATA;
				break;	// Done - Hit the top
			}
			uiLow = uiMid + 1;					/* Too low */
		}
	}

	// Set curElm and the key buffer.
	pStack->uiCurElm = BH_OVHD + (uiMid << 3);
	longToByte( uiCurDrn, pStack->pKeyBuf);

//Exit:
	return( rc);
}

/****************************************************************************
Desc:  	Read the block information and initialize all needed pStack elements.
			Assumes that bsBlock is correct.  Normalizes BH_TYPE to[ 0,1] values.
			Note that byElmOvhd cannot be set for non-leaf blocks unless
			the version number is passed in.
Out:   	stack updated with needed block contents.  
Notes:	Same code is in FSGetBlock(). 
****************************************************************************/
void FSBlkToStack(
	BTSK_p			pStack)
{
	FLMBYTE *		pBlk = pStack->pBlk;
	FLMUINT			uiBlkType;

	pStack->uiBlkType = uiBlkType = (FLMUINT)(BH_GET_TYPE( pBlk ));

	/**
	***	The standard overhead is used in the pStack
	***	Compares are made to determine if the element is extended.
	**/
	if( uiBlkType == BHT_LEAF)
	{
		pStack->uiElmOvhd = BBE_KEY;
	}
	else if( uiBlkType == BHT_NON_LEAF_DATA)
	{
		pStack->uiElmOvhd = BNE_DATA_OVHD;
	}
	else if( uiBlkType == BHT_NON_LEAF)
	{
		pStack->uiElmOvhd = BNE_KEY_START;
	}
	else if( uiBlkType == BHT_NON_LEAF_COUNTS)
	{
		pStack->uiElmOvhd = BNE_KEY_COUNTS_START;
	}
	else
	{
		flmAssert(0);
		pStack->uiElmOvhd = BNE_KEY_START;
	}

	pStack->uiKeyLen = pStack->uiPKC = pStack->uiPrevElmPKC = 0;
	pStack->uiCurElm = BH_OVHD;
	pStack->uiBlkEnd = (FLMUINT)FB2UW( &pBlk[ BH_ELM_END ] );
	pStack->uiLevel  = (FLMUINT)pBlk[ BH_LEVEL ];
}

/***************************************************************************
Desc:		Scan to a specific element (pStack->uiCurElm) in a b-tree block.
			Builds the current key in pStack->pKeyBuf and sets up the stack
			for an insert of an element. 
			The block must exist and never contain a lone LEM (last elm marker). 
Notes:	This may be called at ANY b-tree level.
*****************************************************************************/
RCODE FSBtScanTo(
	BTSK_p			pStack,				// Stack of variables for each level
	FLMBYTE *		pSearchKey,			// The input key to search for 
	FLMUINT			uiSearchKeyLen,	// Length of the key (not null term) 
	FLMUINT			dinDomain			// INDEXES ONLY - lower bounds of din
	)
{
	FLMBYTE *		pCurElm;				// Points to the current element.
	FLMBYTE *		pBlk;					// Points to block - optimization
	FLMBYTE *		pKeyBuf = pStack->pKeyBuf;	// Key buffer to fill
	FLMBYTE *		pPrevElm;			// Points to previous element.
	RCODE				rc = FERR_OK;
	FLMUINT			uiPrevKeyCnt = 0;	// Current elements previous key count.
	FLMUINT			uiElmKeyLen = 0;	// Current element key length.
	FLMUINT			uiTargetCurElm = pStack->uiCurElm;// Target value to scan to.
	FLMUINT			uiElmOvhd;			// Number bytes overhead for element.
	FLMUINT			uiKeyBufLen;		// Length of key in pKeyBuf.

	// Initialize section 
	FSBlkToStack( pStack );				// Read information & put to stack. 
	pBlk  = pStack->pBlk;
	uiElmOvhd = pStack->uiElmOvhd;

	if( uiTargetCurElm > pStack->uiBlkEnd)
	{
		uiTargetCurElm = pStack->uiBlkEnd;
	}

	// The code is easy for non-leaf data blocks.
	if( pStack->uiBlkType == BHT_NON_LEAF_DATA)
	{
		// target may be any byte offset in the block.
		while( pStack->uiCurElm < uiTargetCurElm )
		{
			pStack->uiCurElm += BNE_DATA_OVHD;
		}
		if( uiTargetCurElm < pStack->uiBlkEnd)
		{
			flmCopyDrnKey( pKeyBuf, &pBlk[ pStack->uiCurElm ]);
			pStack->uiCmpStatus = BT_EQ_KEY;
		}
		else
		{
			pStack->uiCmpStatus = BT_END_OF_DATA;
		}
		goto Exit;
	}
	
	// Note: There is no way pPrevElm can be accessed and point to NULL
	// unless the block is corrupt and starts with a PKC value.
	pCurElm = NULL;
	uiKeyBufLen = 0;
	while( pStack->uiCurElm < uiTargetCurElm)
	{
		pPrevElm = pCurElm;
		pCurElm = &pBlk[ pStack->uiCurElm ];
		uiPrevKeyCnt = BBE_GET_PKC( pCurElm );
		uiElmKeyLen  = BBE_GET_KL(  pCurElm );
		if( (pStack->uiKeyLen = uiPrevKeyCnt + uiElmKeyLen) > pStack->uiKeyBufSize)
		{
			rc = RC_SET( FERR_CACHE_ERROR);
			goto Exit;
		}

		// Copy the minimum number of bytes from the previous element.
		// A memcpy may be slower because this shouldn't copy very many bytes.
		if( uiPrevKeyCnt > uiKeyBufLen)
		{
			FLMUINT		uiCopyLength = uiPrevKeyCnt - uiKeyBufLen;
			FLMBYTE *	pSrcPtr = &pPrevElm[ uiElmOvhd];

			flmAssert( pCurElm != NULL);
			// VISIT: Replace with an inline memcpy.
			while( uiCopyLength--)
			{
				pKeyBuf[ uiKeyBufLen++] = *pSrcPtr++;
			}
		}
		else
		{
			uiKeyBufLen = uiPrevKeyCnt;
		}
		
		// Position to the next element

		if( pStack->uiBlkType == BHT_LEAF)
		{	
			pStack->uiCurElm += (FLMUINT)(BBE_LEN( pCurElm ));
			if( pStack->uiCurElm + BBE_LEM_LEN >= pStack->uiBlkEnd)
			{
				f_memcpy( &pKeyBuf[ uiKeyBufLen], &pCurElm[ uiElmOvhd], uiElmKeyLen);

				if( uiSearchKeyLen && (pStack->uiCurElm < pStack->uiBlkEnd))
				{
					// This is a rare and unsure case where caller needs to have
					// pStack->uiPrevElmPKC set correctly.
					FSKeyCmp( pStack, pSearchKey, uiSearchKeyLen, dinDomain);
				}
				goto Hit_End;
			}
		}
		else
		{
			pStack->uiCurElm += (FLMUINT)(BNE_LEN( pStack, pCurElm));	/* Adds in DOMAIN */

			if( pStack->uiCurElm >= pStack->uiBlkEnd)
			{
				// Make sure that pKeyBuf has the last element's key.
				f_memcpy( &pKeyBuf[ uiKeyBufLen], &pCurElm[ uiElmOvhd], uiElmKeyLen);

Hit_End:
				pStack->uiKeyLen = 0;
				pStack->uiPrevElmPKC = pStack->uiPKC;	/* Change previous element */
				pStack->uiPKC = 0;
				pStack->uiCmpStatus = BT_END_OF_DATA;
				goto Exit;
			}
		}
	}
	// Check to see if the scan hit where you wanted, if so setup stack & pKeyBuf.

	if( pStack->uiCurElm == uiTargetCurElm)
	{
		// BE CAREFUL.  Names with "target" point to this element.  All other
		// references include pCurElm point to the previous element.
		FLMBYTE *	pTargetCurElm = CURRENT_ELM( pStack);
		FLMUINT		uiTargetPrevKeyCnt = BBE_GET_PKC( pTargetCurElm);
		FLMUINT		uiTargetElmKeyLen = BBE_GET_KL( pTargetCurElm);

		// Compare the current key so that prevPKC and PKC are set.
		pStack->uiCmpStatus = BT_EQ_KEY;
		if( pCurElm)
		{
			if( uiSearchKeyLen)
			{
				// Copy the entire key into keyBuf to compare - output is ->uiPKC
				f_memcpy( &pKeyBuf[ uiPrevKeyCnt], &pCurElm[ uiElmOvhd], uiElmKeyLen);
				pStack->uiCmpStatus = FSKeyCmp( pStack, pSearchKey, 
							uiSearchKeyLen, dinDomain);
			}
			else if( uiTargetPrevKeyCnt > uiKeyBufLen)

			{
				// Copy what is necessary.  uiPrevKeyCnt is equal to uiKeyBufLen.

				FLMUINT		uiCopyLength = uiTargetPrevKeyCnt - uiKeyBufLen;
				FLMBYTE *	pSrcPtr = &pCurElm[ uiElmOvhd];

				// VISIT: Replace with an inline memcpy.
				while( uiCopyLength--)
				{
					pKeyBuf[ uiKeyBufLen++] = *pSrcPtr++;
				}
			}
		}

		if( (pStack->uiKeyLen = 
				uiTargetPrevKeyCnt + uiTargetElmKeyLen) > pStack->uiKeyBufSize)
		{
			rc = RC_SET( FERR_CACHE_ERROR);
			goto Exit;
		}

		if( uiTargetElmKeyLen)
		{
			f_memcpy( &pKeyBuf[ uiTargetPrevKeyCnt], 
					&pTargetCurElm[ uiElmOvhd], uiTargetElmKeyLen);

			if( uiSearchKeyLen)
			{
				pStack->uiCmpStatus = FSKeyCmp( pStack, pSearchKey, uiSearchKeyLen, dinDomain );
			}
		}
	
		else
		{
			/*
			This will be hit on a condition where we want to insert "ABCD (10)" into
			ABCD (15)
			ABCD (5)
			between the two keys.  (10) is the DIN value.  Because the keys are equal
			we don't have to call compare again.  The uiPKC is the uiPrevKeyCnt value.
			*/
			pStack->uiPrevElmPKC = pStack->uiPKC;
			pStack->uiPKC = uiTargetPrevKeyCnt;
		}
	}
	else
	{
		// Copy the remaining bytes of the current key into the buffer.
		if( pCurElm)
		{
			f_memcpy( &pKeyBuf[ uiPrevKeyCnt], &pCurElm[ uiElmOvhd], uiElmKeyLen);
		}
		pStack->uiCmpStatus = BT_GT_KEY;
		// Not necessary to compare because we just wanted a rough position & keyBuf.
	}

Exit:
	return( rc );
}

/***************************************************************************
Desc:		Standard key compare routine for a key and a b-tree element
Return:	BT_EQ_KEY (0) if equal key was found
			BT_GT_KEY (1) if greater than key was found
			BT_LT_KEY (2) if less than (keep going)
*****************************************************************************/
FSTATIC FLMUINT FSKeyCmp(
	BTSK_p			pStack,				/* Stack of variables for each level */
	FLMBYTE *		key,					/* The input key to search for */
	FLMUINT			uiKeyLen,			/*	Length of the key (not null term) */
	FLMUINT			dinDomain)			/* INDEXES ONLY - lower bounds of din*/
{
	FLMBYTE *		pCurElm;
	FLMBYTE *		pKeyBuf;	  					/* Current element's key */
	FLMUINT			uiCmp;						/* Return value */
	FLMUINT			uiCompareLen;				/* Length to compare */
	FLMUINT			uiOrigCompareLen;			/* Original compare length */
	FLMUINT			uiCurElmKeyLen;			/* Current element's length */
	FLMUINT			uiPKCTemp;

	/* Get again the current element's key length & compute compare length */
	uiCurElmKeyLen = pStack->uiKeyLen;
	uiOrigCompareLen = uiCompareLen = f_min( uiKeyLen, uiCurElmKeyLen );
	pKeyBuf = pStack->pKeyBuf;				/* Point to the local key buffer */
	pStack->uiPrevElmPKC = pStack->uiPKC;	/* Change previous element */
	pStack->uiPKC = 0;

	while( uiCompareLen--)
	{
		if( *key++ == *pKeyBuf++) 			/* Just do a left-right compare */
			continue;

		uiPKCTemp = uiOrigCompareLen - (uiCompareLen + 1);
		pStack->uiPKC = (uiPKCTemp > BBE_PKC_MAX) ? BBE_PKC_MAX : uiPKCTemp;
		/* Not equal so return */
		return( (*(--key) < *(--pKeyBuf)) ? BT_GT_KEY : BT_LT_KEY );
	}

	/* Set the prev key count value */
	pStack->uiPKC = (uiOrigCompareLen <= BBE_PKC_MAX)
							? uiOrigCompareLen
							: BBE_PKC_MAX;
					  		

	/** Set return status,  If equal then compare the dinDomain if needed. **/
	uiCmp = uiKeyLen > uiCurElmKeyLen ? BT_LT_KEY :
		 (uiKeyLen < uiCurElmKeyLen ? BT_GT_KEY : BT_EQ_KEY) ;

	if( (uiCmp == BT_EQ_KEY) && dinDomain)
	{
		pCurElm = CURRENT_ELM( pStack );
		if( (dinDomain - 1) < FSGetDomain( &pCurElm, (FLMBYTE)pStack->uiElmOvhd ))
		{
			uiCmp = BT_LT_KEY;						/* Keep going */
		}
	}
	return( uiCmp );
}


/****************************************************************************
Desc:  	Goto the next element within the block
Notes: 	The key is NOT moved into the byKeyBuf[]
			LEAF: Returns FERR_BT_END_OF_DATA if positioned on the last-element-marker
			(LEM) or is NOW positioned on the LEM.
****************************************************************************/
RCODE FSBlkNextElm(
	BTSK_p			pStack)						/* Stack of variables for each level */
{
	FLMBYTE *		elmPtr;
	FLMUINT	 		uiElmSize;
	RCODE				rc = FERR_BT_END_OF_DATA;			/* Code assumes at the end */

	elmPtr = &pStack->pBlk[ pStack->uiCurElm ];

	if( pStack->uiBlkType == BHT_LEAF)
	{	
		uiElmSize = BBE_LEN( elmPtr );
		if( pStack->uiCurElm + BBE_LEM_LEN < pStack->uiBlkEnd )
		{
			if( (pStack->uiCurElm += uiElmSize) + BBE_LEM_LEN < pStack->uiBlkEnd)
				rc = FERR_OK;
		}
	}
	else
	{
		if( pStack->uiBlkType == BHT_NON_LEAF_DATA)	
			uiElmSize = BNE_DATA_OVHD;
		else
			uiElmSize = (FLMUINT) BNE_LEN( pStack, elmPtr);	/* Adds in DOMAIN if present */

		if( pStack->uiCurElm < pStack->uiBlkEnd)
		{
			/* Check if this is not the last element within the block */
			if( (pStack->uiCurElm += uiElmSize) < pStack->uiBlkEnd)
				rc = FERR_OK;
		}
	}
	return( rc );
}

/***************************************************************************
Desc:		Go to the next element in the logical b-tree while building the key
Notes:	You may be at any level of the b-tree!
*****************************************************************************/
RCODE FSBtNextElm(
	FDB_p			pDb,
	LFILE *		pLFile,			/* Logical file definition */
	BTSK_p		pStack)			/* Stack of variables for each level */
{
	RCODE			rc = FERR_OK;						/* Return code */

	if( pStack->uiCurElm < BH_OVHD )		/* Before first element in block? */
	{
		pStack->uiCurElm = BH_OVHD;
	}
	else if( (rc = FSBlkNextElm( pStack)) == FERR_BT_END_OF_DATA)
	{
		FLMBYTE *	pBlk = BLK_ELM_ADDR( pStack, BH_NEXT_BLK );
		FLMUINT	blkNum = FB2UD( pBlk );
		if( blkNum != BT_END )				/* If not end, read in the next block */
		{
			/* Current element was last element in the block - goto next block */
			
			if( RC_OK(rc = FSGetBlock( pDb, pLFile, blkNum, pStack )))
			{
				/* Set blk end and adjust parent block to next element */
				pBlk = pStack->pBlk;
				pStack->uiBlkEnd = (FLMUINT)FB2UW( &pBlk[ BH_ELM_END ]);
				pStack->uiCurElm = BH_OVHD;
				pStack->uiPKC = 0;
				pStack->uiPrevElmPKC = 0;

				if( pStack->uiFlags & FULL_STACK)		/* Adjust the stack if needed */
					rc = FSAdjustStack( pDb, pLFile, pStack, TRUE);
			}
		}
	} /* At this point if rc == FERR_BT_END_OF_DATA then is at last block, else OK */

	if( RC_OK(rc))	/* If there is a next element, setup stack and byKeyBuf[] */
	{
		FLMBYTE *	pCurElm = CURRENT_ELM( pStack );
		FLMUINT		uiKeyLen;

		if( pStack->uiBlkType == BHT_NON_LEAF_DATA)
		{
			flmCopyDrnKey( pStack->pKeyBuf, pCurElm);
			goto Exit;
		}

		/* Copy key to the stack->pKeyBuf & check for end key */
		if( (uiKeyLen = BBE_GET_KL( pCurElm)) != 0)
		{
			FLMUINT		uiPKC = (FLMUINT)(BBE_GET_PKC( pCurElm ));
			
			if( uiKeyLen + uiPKC <= pStack->uiKeyBufSize )
			{
				pStack->uiKeyLen = (uiKeyLen + uiPKC);
				f_memcpy( &pStack->pKeyBuf[ uiPKC], 
							 &pCurElm[ pStack->uiElmOvhd ], uiKeyLen);
			}
			else
			{
				rc = RC_SET( FERR_CACHE_ERROR);
				goto Exit;
			}
		}
	}
Exit:	
	return( rc );
}

/***************************************************************************
Desc:		Adjust a full stack if pStack->byFlags & FULL_STACK
			If the stack->byFlags is set to FULL_STACK then the parent must
			point to the next element when changing to the next block in order
			for FSBtInsert() or FSBtDelete() to work correctly during block
			splits.
*****************************************************************************/
RCODE FSAdjustStack(
	FDB_p			pDb,
	LFILE *		pLFile,		/* Logical file definition */
	BTSK_p		pStack,		/* Stack of variables for each level */
	FLMBOOL		bMovedNext)
{
	RCODE			rc = FERR_OK;

	pStack->uiFlags = FULL_STACK;

	/**---------------------------------------------------------------------
	***		Pop the stack and go to the next element
	***		This is a recursive call back to FSBtNextElm() or FSBtPrevElm()
	***		Watch out, this will not work if the concurrency model changes
	***		to a b-tree locking method like other products use.
	***--------------------------------------------------------------------*/

	/* Pop the pStack going to the parents block */
	pStack--;

	/* It is very rare that block will need to be read.  Maybe
		some sort of split case.  The block should have already have been
		read.
	*/
	
	if( RC_OK(rc = FSGetBlock( pDb, pLFile, pStack->uiBlkAddr, pStack)))
	{
		rc = bMovedNext
			  ? FSBtNextElm( pDb, pLFile, pStack)		/* Next element */
			  : FSBtPrevElm( pDb, pLFile, pStack);		/* Previous element */
	}

	/* Push the pStack and unpin the current block */		
	pStack++;

	return( rc );
}

/***************************************************************************
Desc:		Read in a block from the cache and set most stack elements.
*****************************************************************************/

RCODE FSGetBlock(
	FDB_p			pDb,
	LFILE *		pLFile,			// Logical file definition
	FLMUINT		uiBlkAddr,		// Block Address to read.
	BTSK_p		pStack)			// Stack of variables for each level
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pBlk;

	/*
	Release whatever block might be there first.  If no block is
	there (pStack->pSCache == NULL), FSReleaseBlock does nothing.  Stacks
	are ALWAYS initialized to set pSCache to NULL, so this is OK to call
	even if stack has never been used to read a block yet.
	*/

	if( pStack->pSCache)
	{
		/* If we already have the block we want, keep it! */

		if( pStack->pSCache->uiBlkAddress != uiBlkAddr)
		{
			FSReleaseBlock( pStack, FALSE);
		}
	}

	if( !pStack->pSCache)
	{
		flmAssert( !pStack->pBlk);

		if( RC_BAD( rc = ScaGetBlock( pDb, pLFile, BHT_LEAF,
								uiBlkAddr, NULL, &pStack->pSCache)))
		{
			goto Exit;
		}
	}

	pStack->pBlk = pBlk = pStack->pSCache->pucBlk;
	if( pStack->uiBlkAddr != uiBlkAddr)
	{
		FLMUINT	uiBlkType;

		pStack->uiBlkAddr = uiBlkAddr;

		// set other pStack elements.

		pStack->uiBlkType = uiBlkType = (FLMUINT)(BH_GET_TYPE( pBlk ));

		/**
		***	The standard overhead is used in the stack
		***	Compares are made to determine if the element is extended.
		**/

		if( uiBlkType == BHT_LEAF)
		{
			pStack->uiElmOvhd = BBE_KEY;
		}
		else if( uiBlkType == BHT_NON_LEAF_DATA)
		{
			pStack->uiElmOvhd = BNE_DATA_OVHD;
		}
		else if( uiBlkType == BHT_NON_LEAF)
		{
			pStack->uiElmOvhd = BNE_KEY_START;
		}
		else if( uiBlkType == BHT_NON_LEAF_COUNTS)
		{
			pStack->uiElmOvhd = BNE_KEY_COUNTS_START;
		}
		else
		{
			rc = RC_SET( FERR_DATA_ERROR);
			FSReleaseBlock( pStack, FALSE);
			goto Exit;
		}

		pStack->uiKeyLen = pStack->uiPKC = pStack->uiPrevElmPKC = 0;
		pStack->uiLevel = (FLMUINT)pBlk[ BH_LEVEL ];
		pStack->uiCurElm = BH_OVHD;
	}
	pStack->uiBlkEnd = (FLMUINT)FB2UW( &pBlk[ BH_ELM_END ] );

Exit:
	
	return( rc );
}

/***************************************************************************
Desc:	Release all of the cache associated with a stack.
***************************************************************************/
void FSReleaseStackCache(
	BTSK_p		pStack,
	FLMUINT		uiNumLevels,
	FLMBOOL		bMutexAlreadyLocked)
{
	FLMBOOL	bSemLocked = FALSE;

	while( uiNumLevels)
	{
		if( pStack->pSCache)
		{
			if( !bSemLocked && !bMutexAlreadyLocked)
			{
				f_mutexLock( gv_FlmSysData.hShareMutex);
				bSemLocked = TRUE;
			}
			ScaReleaseCache( pStack->pSCache, TRUE);
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;
		}
		uiNumLevels--;
		pStack++;
	}
	if( bSemLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}


