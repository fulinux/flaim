//-------------------------------------------------------------------------
// Desc:	Query positioning keys
// Tabs:	3
//
//		Copyright (c) 1997-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fqkeys.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define DOMAIN_TO_DRN(uiDomain)	(FLMUINT)(((uiDomain) + 1) * 256 + 1)
#define DRN_TO_DOMAIN(uiDrn)		(FLMUINT)(((uiDrn) - 1) / 256 - 1)

FSTATIC FLMINT flmPosKeyCompare(
	POS_KEY_p		pKey1,
	POS_KEY_p		pKey2);

FSTATIC RCODE flmLoadPosKeys(
	CURSOR_p			pCursor,
	POS_KEY_p		pKeys,
	FLMUINT			uiNumKeys,
	FLMBOOL			bLeafLevel);

FSTATIC RCODE flmKeyIsMatch(
	CURSOR_p				pCursor,
	IXD_p					pIxd,
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiDrn,
	POS_KEY_p *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMUINT				uiKeyArrayGrowSize);

FSTATIC RCODE flmExamineBlock(
	CURSOR_p				pCursor,
	IXD_p					pIxd,
	FLMBYTE *			pucBlk,
	FSIndexCursor *	pFSIndexCursor,
	FLMUINT **			ppuiChildBlockAddresses,
	FLMUINT *			puiNumChildBlocks,
	FLMUINT *			puiBlkAddressArrayAllocSize,
	POS_KEY_p *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMBOOL *			pbHighKeyInRange);

FSTATIC RCODE flmGetLastKey(
	FDB *					pDb,
	CURSOR_p				pCursor,
	IXD_p					pIxd,
	LFILE *				pLFile,
	FLMUINT				uiBlockAddress,
	POS_KEY_p *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize);

FSTATIC RCODE flmCurGetPosKeys(
	FDB *				pDb,
	CURSOR_p			pCursor);

/****************************************************************************
Desc: Compares the contents of the key buffers for two cursor positioning keys,
		returning one of the following values:
				<0		Indicates that the first key is less than the second.
				 0		Indicates that the two keys are equal.
				>0		Indicates that the first key is greater then the second.
****************************************************************************/
FSTATIC FLMINT flmPosKeyCompare(
	POS_KEY_p	pKey1,
	POS_KEY_p	pKey2
	)
{
	FLMINT	iCmp;
	
	if (pKey1->uiKeyLen > pKey2->uiKeyLen)
	{
		if ((iCmp = f_memcmp( pKey1->pucKey, pKey2->pucKey,
								pKey2->uiKeyLen)) == 0)
		{
			iCmp = 1;
		}
	}
	else if( pKey1->uiKeyLen < pKey2->uiKeyLen)
	{
		if ((iCmp = f_memcmp( pKey1->pucKey, pKey2->pucKey,
								pKey1->uiKeyLen)) == 0)
		{
			iCmp = -1;
		}
	}
	else
	{
		if ((iCmp = f_memcmp( pKey1->pucKey,
							pKey2->pucKey, pKey2->uiKeyLen)) == 0)
		{
			// Compare DRNs if everything else is the same.  NOTE: DRNs are in
			// reverse order in the positioning key array.

			if (pKey1->uiDrn && pKey2->uiDrn)
			{
				if (pKey1->uiDrn > pKey2->uiDrn)
				{
					iCmp = -1;
				}
				else if (pKey1->uiDrn < pKey2->uiDrn)
				{
					iCmp = 1;
				}
			}
		}
	}
	return iCmp;
}

/****************************************************************************
Desc: Loads a set of positioning keys into a subquery's array, allocating it
		if necessary.
****************************************************************************/
FSTATIC RCODE flmLoadPosKeys(
	CURSOR_p		pCursor,
	POS_KEY_p	pKeys,
	FLMUINT		uiNumKeys,
	FLMBOOL		bLeafLevel
	)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiKeyCnt;
	FLMUINT	uiRFactor;
	FLMUINT	uiTotCnt;

	// If the B-tree was empty, the key array will be left NULL.
	
	if (!pKeys || !uiNumKeys)
	{
		goto Exit;
	}
	
	// Allocate the array of positioning keys in the subquery.
		
	uiKeyCnt = (uiNumKeys > FLM_MAX_POS_KEYS + 1)
				? FLM_MAX_POS_KEYS + 1
				: uiNumKeys;
	if (RC_BAD( rc = f_calloc( uiKeyCnt * sizeof( POS_KEY),
										&pCursor->pPosKeyArray)))
	{
		goto Exit;
	}
	pCursor->uiNumPosKeys = uiKeyCnt;
	pCursor->bLeafLevel = bLeafLevel;

	// If there are less keys than the number of slots in the positioning
	// key array, each key must be put into multiple slots.  Calculate how
	// many slots correspond to each key (uiSlots), and then set the keys
	// into their corresponding slots.  NOTE: it will often be the case that
	// the number of keys does not divide evenly into the number of slots in
	// the array.  In these cases thare will be a remainder, uiRFactor.  If
	// uiRFactor = n, the first n keys will be set into (uiSlots + 1) slots.
	
	if (uiNumKeys <= FLM_MAX_POS_KEYS + 1)
	{
		for (uiTotCnt = 0; uiTotCnt < uiKeyCnt; uiTotCnt++)
		{
			f_memcpy( &pCursor->pPosKeyArray[ uiTotCnt],
						 &pKeys[ uiTotCnt],
						 sizeof( POS_KEY));

			// NOTE: we're keeping this memory for the positioning key to which
			// it is being copied.
			pKeys [uiTotCnt].pucKey = NULL;
		}
	}

	// If there are more keys than the number of slots in the positioning
	// key array, a certain number of keys must be skipped for each key that
	// is set in the array.  Calculate how many keys must be skipped for each
	// slot (uiIntervalSize), and then iterate through the passed-in set of
	// keys, setting the appropriate ones into their corresponding slots.
	// NOTE: it will often be the case that the number of slots in the array
	// does not divide evenly into the number of keys.  In these cases there
	// will be a remainder (uiRFactor).  Where uiRFactor = n,
	// (uiIntervalSize + 1) keys will be skipped before each of the first n
	// slots in the array are filled.
	
	else
	{
		FLMUINT		uiLoopCnt;
		FLMUINT		uiIntervalSize = (uiNumKeys - 2) / (FLM_MAX_POS_KEYS - 1) - 1;
		
		uiRFactor = (uiNumKeys - 2) % (FLM_MAX_POS_KEYS - 1);

		f_memcpy( &pCursor->pPosKeyArray[ 0], &pKeys[ 0], sizeof( POS_KEY));
		f_memcpy( &pCursor->pPosKeyArray[ 1], &pKeys[ 1], sizeof( POS_KEY));
		
		// NOTE: we're keeping this memory for the positioning key to which
		// it is being copied.
		pKeys [0].pucKey = NULL;
		pKeys [1].pucKey = NULL;

		uiKeyCnt = 2;
		for( uiTotCnt = 2; uiTotCnt < FLM_MAX_POS_KEYS; uiTotCnt++)
		{
			for( uiLoopCnt = 0; uiLoopCnt < uiIntervalSize; uiLoopCnt++)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
				uiKeyCnt++;
			}
			
			if( uiRFactor)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
				uiKeyCnt++;
				uiRFactor--;
			}
			
			f_memcpy( &pCursor->pPosKeyArray[ uiTotCnt],
						&pKeys[ uiKeyCnt], sizeof( POS_KEY));
						
			// NOTE: we're keeping this memory for the positioning key to which
			// it is being copied.
			
			pKeys [uiKeyCnt].pucKey = NULL;
			uiKeyCnt++;
		}

		// Make sure the last key in the positioning key array is the last
		// key in the result set, then free the memory used for the pKey array.
		
		f_memcpy( &pCursor->pPosKeyArray[ FLM_MAX_POS_KEYS],
						&pKeys[ uiNumKeys - 1], sizeof( POS_KEY));
		pKeys [uiNumKeys - 1].pucKey = NULL;
		while (uiKeyCnt < uiNumKeys - 1)
		{
			f_free( &pKeys[ uiKeyCnt].pucKey);
			uiKeyCnt++;
		}
	}
	
Exit:
	return( rc);
}

/****************************************************************************
Desc: Evaluates an index key against selection criteria, and adds it to the
		passed-in key array.
****************************************************************************/
FSTATIC RCODE flmKeyIsMatch(
	CURSOR_p				pCursor,
	IXD_p					pIxd,
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiDrn,
	POS_KEY_p *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMUINT				uiKeyArrayGrowSize
	)
{
	RCODE				rc = FERR_OK;
	SUBQUERY_p		pSubQuery = pCursor->pSubQueryList;
	FlmRecord *		pKey = NULL;
	FLMBOOL			bHaveMatch = FALSE;
	FLMUINT			uiResult;
	POS_KEY_p		pPosKey;
	
	// If pSubQuery->bDoKeyMatch is FALSE, the selection criteria for this
	// query are satisfied by a contiguous set of index keys.  Therefore,
	// there is no need to evaluate keys against the selection criteria.
	// We have already established that the passed-in key falls within
	// the range of keys that contains the result set of the query.
	// NOTE: bDoRecMatch cannot ever be set, otherwise, positioning is not
	// allowed.

	bHaveMatch = !pSubQuery->OptInfo.bDoKeyMatch;
	if (!bHaveMatch)
	{

		// Get the key in the form of a FlmRecord object.

		if (RC_BAD( rc = flmIxKeyOutput( pIxd, pucKey, uiKeyLen, &pKey, TRUE)))
		{
			goto Exit;
		}
		pKey->setID( uiDrn);

		// Evaluate the key against the subquery - there will only
		// be one at this point.

		if (RC_BAD( rc = flmCurEvalCriteria( pCursor, pSubQuery,
									pKey, TRUE, &uiResult)))
		{
			if (rc == FERR_TRUNCATED_KEY)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}
		bHaveMatch = (uiResult == FLM_TRUE) ? TRUE : FALSE;
	}
			
	if (bHaveMatch)
	{
		if (*puiNumKeys == *puiKeyArrayAllocSize)
		{
			if (RC_BAD( rc = f_recalloc(
				(*puiKeyArrayAllocSize + uiKeyArrayGrowSize) * sizeof( POS_KEY),
				ppKeys)))
			{
				goto Exit;
			}
			(*puiKeyArrayAllocSize) += uiKeyArrayGrowSize;
		}
		pPosKey = &((*ppKeys)[*puiNumKeys]);
		if (RC_BAD( rc = f_calloc( uiKeyLen, &pPosKey->pucKey)))
		{
			goto Exit;
		}
		f_memcpy( pPosKey->pucKey, pucKey, uiKeyLen);
		pPosKey->uiKeyLen = uiKeyLen;
		pPosKey->uiDrn = uiDrn;
		(*puiNumKeys)++;
	}
	
Exit:
	if (pKey)
	{
		pKey->Release();
	}
	return( rc);
}

/****************************************************************************
Desc: Examines an index B-tree block to find the keys in it that could be
		used to position within a cursor's result set.
Visit:This code NEEDS to use the b-tree routines and NOT use the low level
		format codes to go to the next element or key.  Other problems include
		doing the same work for each element even though you are at the same
		level of the b-tree.
****************************************************************************/
FSTATIC RCODE flmExamineBlock(
	CURSOR_p				pCursor,
	IXD_p					pIxd,
	FLMBYTE *			pucBlk,
	FSIndexCursor *	pFSIndexCursor,
	FLMUINT **			ppuiChildBlockAddresses,
	FLMUINT *			puiNumChildBlocks,
	FLMUINT *			puiBlkAddressArrayAllocSize,
	POS_KEY_p *			ppKeys,
	FLMUINT *			puiNumKeys,
	FLMUINT *			puiKeyArrayAllocSize,
	FLMBOOL *			pbHighKeyInRange
	)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			ucFromKey [MAX_KEY_SIZ];
	FLMUINT			uiFromKeyLen;
	FLMBYTE			ucUntilKey [MAX_KEY_SIZ];
	FLMUINT			uiUntilKeyLen;
	FLMUINT			uiUntilDrn = 0;
	FLMBOOL			bRangeOverlaps;
	FLMBOOL			bUntilKeyInSet;
	FLMBOOL			bUntilKeyPastEndOfKeys;
	FLMUINT			uiDomain;
	DIN_STATE		dinState;
	FLMBOOL			bFirstRef;
	FLMUINT			uiEndOfBlock = FB2UW( &pucBlk [BH_BLK_END]);
	FLMUINT			uiCurrElmOffset = BH_OVHD;
	FLMUINT			uiBlkType = (FLMUINT)BH_GET_TYPE( pucBlk);
	FLMUINT			uiElmLength;
	FLMBYTE *		pucElement;
	FLMBYTE *		pucElm;
	FLMBYTE *		pucElmKey;
	FLMBYTE *		pucElmRecord;
	FLMBYTE *		pucChildBlkAddr;
	FLMUINT			uiChildBlkAddr;
	FLMUINT			uiElmRecLen;
	FLMUINT			uiElmKeyLen;
	FLMUINT			uiElmPKCLen;
	FLMUINT			uiElmOvhd;

	// This loop moves across a database block from the leftmost element to the
	// rightmost.  Each contiguous pair of elements is viewed as a "key range",
	// where the first key in the pair is the start key and the second is the
	// end key.  In the loop, each key range is checked to see if it overlaps
	// with any part of the query's result set.  If it does, two things happen:
	// first, the down pointer from the end key is added to a passed-in list;
	// second, the end key is checked to see if it satisfies the query's
	// selection criteria.  If it does, it is added to a passed-in list of
	// positioning keys.
	// NOTE: until key is given a key length of 0 so that in the first iteration,
	// the key range will be from FO_FIRST to the leftmost key in the block.

	if( uiBlkType == BHT_LEAF)
	{
		uiElmOvhd = BBE_KEY;
	}
	else if( uiBlkType == BHT_NON_LEAF_DATA)
	{
		uiElmOvhd = BNE_DATA_OVHD;
	}
	else if( uiBlkType == BHT_NON_LEAF)
	{
		uiElmOvhd = BNE_KEY_START;
	}
	else
	{
		uiElmOvhd = BNE_KEY_COUNTS_START;
	}
	
	uiUntilKeyLen = 0;
	bUntilKeyPastEndOfKeys = FALSE;
	bFirstRef = TRUE;
	while (uiCurrElmOffset < uiEndOfBlock)
	{

		// Move the until key into the start key buffer.

		if (uiUntilKeyLen)
		{
			f_memcpy( ucFromKey, ucUntilKey, uiUntilKeyLen);
		}
		uiFromKeyLen = uiUntilKeyLen;

		pucElement = &pucBlk [uiCurrElmOffset];
		pucElm = pucElement;
		uiDomain = FSGetDomain( &pucElm, uiElmOvhd);

		if (uiBlkType == BHT_LEAF)
		{
			uiElmLength = (FLMUINT)(BBE_LEN( pucElement));
			pucElmKey = &pucElement [BBE_KEY];
			pucElmRecord = BBE_REC_PTR( pucElement);
			uiElmRecLen = BBE_GET_RL( pucElement);
			if (bFirstRef)
			{
				RESET_DINSTATE( dinState);
				uiUntilDrn = SENNextVal( &pucElm);
				bFirstRef = FALSE;
			}
			else
			{
				FLMUINT uiRefSize = uiElmRecLen -
											(FLMUINT)(pucElm - pucElmRecord);

				if (dinState.uiOffset < uiRefSize)
				{

					// Not at end, read current value.

					DINNextVal( pucElm, &dinState);
				}

				if (dinState.uiOffset >= uiRefSize)
				{
					uiCurrElmOffset += uiElmLength;
					bFirstRef = TRUE;

					// No need to go any further if we have run
					// off the end of the list of keys for the query.

					if (bUntilKeyPastEndOfKeys)
					{
						break;
					}
					else
					{
						continue;
					}
				}
				else
				{
					DIN_STATE	savedState;

					// Don't move the dinState, stay
					// put and get the next DIN value

					savedState.uiOffset = dinState.uiOffset;
					savedState.uiOnes   = dinState.uiOnes;
					uiUntilDrn -= DINNextVal( pucElm, &savedState);
				}
			}
		}
		else if (uiBlkType == BHT_NON_LEAF_DATA)
		{
			uiElmLength = uiElmOvhd;
			pucElmKey = pucElement;
			uiUntilDrn = DOMAIN_TO_DRN( uiDomain);
		}
		else
		{
			uiElmLength = BBE_GET_KL( pucElement ) + uiElmOvhd + 
							(BNE_IS_DOMAIN(pucElement) ? BNE_DOMAIN_LEN : 0);
			pucElmKey = &pucElement [uiElmOvhd];
			uiUntilDrn = DOMAIN_TO_DRN( uiDomain);
		}

		// See if we are on the last element.  If it is a leaf block,
		// it does NOT represent a key.  If it is a non-leaf block,
		// it represents the highest possible key, but there is no
		// data to extract fields from.

		if ((uiBlkType == BHT_LEAF) && (uiElmLength == uiElmOvhd))
		{
			goto Exit;		// Should return FERR_OK
		}

		if ((uiBlkType != BHT_LEAF) && (uiElmLength == uiElmOvhd))
		{
			uiElmKeyLen = uiElmPKCLen = uiUntilKeyLen = 0;
		}
		else
		{

			// Get the element key length and previous key count (PKC).

			uiElmKeyLen = (FLMUINT)(BBE_GET_KL( pucElement));
			uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( pucElement));

			// Now copy the current partial key into the EndKey key buffer.
			
			f_memcpy( &ucUntilKey [uiElmPKCLen], pucElmKey, uiElmKeyLen);
			uiUntilKeyLen = uiElmKeyLen + uiElmPKCLen;
		}

		// Test for Overlap of from key (exclusive) to until key (inclusive)
		// with search keys.

		bRangeOverlaps = pFSIndexCursor->compareKeyRange(
									ucFromKey, uiFromKeyLen,
									(FLMBOOL)((uiFromKeyLen)
												 ? TRUE
												 : FALSE),
									ucUntilKey, uiUntilKeyLen, FALSE,
									&bUntilKeyInSet, &bUntilKeyPastEndOfKeys);

		// Does this range overlap a range of keys?

		if (bRangeOverlaps)
		{

			// If we are not at the leaf level, get and save child block address.

			if (uiBlkType != BHT_LEAF)
			{
				// THIS CODE SHOULD BE USING A STACK!!!!

				if (uiElmOvhd == BNE_DATA_OVHD)
				{
					pucChildBlkAddr = &pucElement[ BNE_DATA_CHILD_BLOCK];
				}
				else
				{
					pucChildBlkAddr = &pucElement [BNE_CHILD_BLOCK];
				}
				uiChildBlkAddr = FB2UD( pucChildBlkAddr );

				// Save uiChildBlkAddr to array of child block addresses.

				if (*puiNumChildBlocks == *puiBlkAddressArrayAllocSize)
				{
					if (RC_BAD( rc = f_recalloc(
						(*puiBlkAddressArrayAllocSize + FLM_ADDR_GROW_SIZE)
						* sizeof( FLMUINT), ppuiChildBlockAddresses)))
					{
						goto Exit;
					}
					(*puiBlkAddressArrayAllocSize) += FLM_ADDR_GROW_SIZE;
				}
				(*ppuiChildBlockAddresses)[ *puiNumChildBlocks] = uiChildBlkAddr;
				(*puiNumChildBlocks)++;
			}

			// If the last element in the block has just been processed, the key
			// will have a length of 0.  If it is somewhere within the range of
			// keys that contains the query's result set, return TRUE in
			// pbHighKeyInRange.  At a higher level, if only one more key is
			// needed to fill the array of positioning keys, the B-Tree will
			// then be traversed to the leaf level to retrieve and test the
			// rightmost key.

			if (!uiUntilKeyLen && bUntilKeyInSet)
			{
				*pbHighKeyInRange = TRUE;
			}

			// If the key falls into one of the key ranges that contain the
			// query's result set, see if it satisfies the selection criteria.
			// If so, increment the counter for the positioning key array and
			// put the key into the array.
			
			else if (bUntilKeyInSet)
			{
				if (RC_BAD( rc = flmKeyIsMatch( pCursor, pIxd,
														  ucUntilKey, uiUntilKeyLen,
														  uiUntilDrn,
														  ppKeys, puiNumKeys,
														  puiKeyArrayAllocSize,
														  FLM_KEYS_GROW_SIZE)))
				{
					goto Exit;
				}
			}
		}

		// If this is not the first reference, stay inside the element and
		// get the next reference.

		if (!bFirstRef)
		{
			continue;
		}

		uiCurrElmOffset += uiElmLength;

		// No need to go any further if we have run off the end of the list
		// of keys for the query.

		if (bUntilKeyPastEndOfKeys)
		{
			break;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc: Finds the rightmost key in the leaf level of a B-tree, and evaluates
		it against the selection criteria of the given subquery.
Visit:This routine must be rewritten to get rid of the low level BTREE
		definitions.  The next() btree calls should have been used.
****************************************************************************/
FSTATIC RCODE flmGetLastKey(
	FDB *				pDb,
	CURSOR_p			pCursor,
	IXD_p				pIxd,
	LFILE *			pLFile,
	FLMUINT			uiBlockAddress,
	POS_KEY_p *		ppKeys,
	FLMUINT *		puiNumKeys,
	FLMUINT *		puiKeyArrayAllocSize
	)
{
	RCODE				rc = FERR_OK;
	FLMBYTE			ucEndKey [MAX_KEY_SIZ];
	FLMUINT			uiEndKeyLen = 0;
	FLMUINT			uiEndDrn = 0;
	BTSK				stack;
	FLMBYTE			ucKeyBuf [MAX_KEY_SIZ];
	BTSK_p			pStack = &stack;
	FLMUINT			uiEndOfBlock;
	FLMUINT			uiCurrElmOffset;
	FLMUINT			uiBlkType;
	FLMUINT			uiElmLength;
	FLMBYTE *		pucBlk;
	FLMBYTE *		pucElement = NULL;
	FLMBYTE *		pucElm;
	FLMBYTE *		pucElmKey;
	FLMBYTE *		pucElmRecord;
	FLMUINT			uiElmRecLen;
	FLMBYTE *		pucBlockAddress;
	FLMUINT			uiElmKeyLen;
	FLMUINT			uiElmPKCLen;
	FLMBOOL			bHaveLastKey = FALSE;
	FLMUINT			uiElmOvhd = 0;
	DIN_STATE		dinState;
	FLMUINT			uiRefSize;

	FSInitStackCache( pStack, 1);
	pStack->pKeyBuf = &ucKeyBuf [0];
	
	// uiBlockAddress contains the address of the rightmost B-Tree block at
	// some unspecified level of the B-Tree (usually not the leaf level).
	// This loop works down the right side of the B-Tree from the passed-in
	// block address until it reaches the rightmost block at the leaf level.
	// The rightmost key is then found in that block.
	
	for( ;;)
	{
		if (RC_BAD(rc = FSGetBlock( pDb, pLFile, uiBlockAddress, pStack)))
		{
			goto Exit;
		}
		pucBlk = pStack->pBlk;
		uiBlkType = (FLMUINT)(BH_GET_TYPE( pucBlk));
		uiEndOfBlock = (FLMUINT)pStack->uiBlkEnd;
		uiCurrElmOffset = BH_OVHD;
		
		// This loop works across a B-Tree block from the leftmost key to the
		// rightmost key.  At non-leaf levels of the B-Tree, the child block
		// address associated with the rightmost key is then used to progress
		// further down the right side of the B-Tree.

		while (uiCurrElmOffset < uiEndOfBlock)
		{

			pucElement = &pucBlk [uiCurrElmOffset];

			if (uiBlkType == BHT_LEAF)
			{
				uiElmOvhd = BBE_KEY;
				uiElmLength = (FLMUINT)(BBE_LEN( pucElement));
				pucElmKey = &pucElement [BBE_KEY];

				// See if we are on the last element.  If it is a leaf block,
				// it does NOT represent a key; the previous element that was
				// processed contained the last key, which means we're finished.
				
				if (uiElmLength == uiElmOvhd)
				{
					bHaveLastKey = TRUE;
					break;
				}

				// Get the last DRN in the element - in case this element is
				// the last one before the end.

				pucElmRecord = BBE_REC_PTR( pucElement);
				uiElmRecLen = BBE_GET_RL( pucElement);
				pucElm = pucElement;
				(void)FSGetDomain( &pucElm, uiElmOvhd);
				RESET_DINSTATE( dinState);
				uiEndDrn = SENNextVal( &pucElm);
				uiRefSize = uiElmRecLen -
											(FLMUINT)(pucElm - pucElmRecord);
				for (;;)
				{
					if (dinState.uiOffset < uiRefSize)
					{

						// Not at end, read current value.

						DINNextVal( pucElm, &dinState);
					}

					if (dinState.uiOffset >= uiRefSize)
					{
						break;
					}
					else
					{
						DIN_STATE	savedState;

						// Don't move the dinState, stay
						// put and get the next DIN value

						savedState.uiOffset = dinState.uiOffset;
						savedState.uiOnes   = dinState.uiOnes;
						uiEndDrn -= DINNextVal( pucElm, &savedState);
					}
				}
			}
			else if( uiBlkType == BHT_NON_LEAF_DATA)
			{
				uiElmOvhd = uiElmLength = BNE_DATA_OVHD;
				pucElmKey = pucElement;
			}
			else
			{
				uiElmOvhd = pStack->uiElmOvhd;

				uiElmLength = BBE_GET_KL( pucElement ) + uiElmOvhd + 
							(BNE_IS_DOMAIN(pucElement) ? BNE_DOMAIN_LEN : 0);	
				pucElmKey = &pucElement [uiElmOvhd];
			}

			if ((uiBlkType != BHT_LEAF) && (uiElmLength == uiElmOvhd))
			{
				uiElmKeyLen = uiElmPKCLen = uiEndKeyLen = 0;
			}
			else if (uiBlkType == BHT_NON_LEAF_DATA)
			{
				uiElmLength = BNE_DATA_OVHD;
				f_memcpy( ucEndKey, pucElmKey, DIN_KEY_SIZ);
			}
			else
			{

				/* Get the element key length and previous key count (PKC). */

				uiElmKeyLen = (FLMUINT)(BBE_GET_KL( pucElement));
				uiElmPKCLen = (FLMUINT)(BBE_GET_PKC( pucElement));

				f_memcpy( &ucEndKey [uiElmPKCLen], pucElmKey, uiElmKeyLen);
				uiEndKeyLen = (FLMUINT)(uiElmKeyLen + uiElmPKCLen);
			}
			uiCurrElmOffset += uiElmLength;
		}
		
		if (!bHaveLastKey)
		{

			// Get and save child block address.

			pucBlockAddress = (FLMBYTE *)((uiElmOvhd == BNE_DATA_OVHD)
													? &pucElement [BNE_DATA_CHILD_BLOCK]
													: &pucElement [BNE_CHILD_BLOCK]);
			uiBlockAddress = FB2UD( pucBlockAddress );
		}
		else
		{

			// We have reached the leaf level of the B-Tree, and we have the
			// rightmost key.  See if it satisfies the selection criteria for
			// the query. If so, put it into the passed-in array of positioning
			// keys.  Then break out of the loop; we're finished.

			if (RC_BAD( rc = flmKeyIsMatch( pCursor, pIxd,
											ucEndKey, uiEndKeyLen, uiEndDrn,
											ppKeys, puiNumKeys,
											puiKeyArrayAllocSize, 1)))
			{
				goto Exit;
			}

			break;
		}
	}
	
Exit:
	FSReleaseBlock( pStack, FALSE);
	return( rc);
}

/****************************************************************************
Desc: Frees the allocations associated with a subquery's array.
****************************************************************************/
void flmCurFreePosKeys(
	CURSOR_p			pCursor
	)
{
	FLMUINT	uiLoopCnt;
	
	if (pCursor->pPosKeyArray)
	{
		for (uiLoopCnt = 0; uiLoopCnt < pCursor->uiNumPosKeys; uiLoopCnt++)
		{
			f_free( &pCursor->pPosKeyArray[ uiLoopCnt].pucKey);
		}
		f_free( &pCursor->pPosKeyArray);
		pCursor->uiNumPosKeys = 0;
	}
	pCursor->uiLastPrcntPos = 0;
	pCursor->uiLastPrcntOffs = 0;
	pCursor->bUsePrcntPos = FALSE;
}

/****************************************************************************
Desc: Gets a set of positioning keys for a particular subquery.
****************************************************************************/
FSTATIC RCODE flmCurGetPosKeys(
	FDB *				pDb,
	CURSOR_p			pCursor
	)
{
	RCODE				rc = FERR_OK;
	BTSK				stack [BH_MAX_LEVELS];
	FLMBYTE			ucKeyBuf [MAX_KEY_SIZ];
	BTSK_p			pStack = stack;
	LFILE *			pLFile;
	LFILE				TmpLFile;
	IXD_p				pIxd;
	SUBQUERY_p		pSubQuery;
	FLMUINT *		puiChildBlockAddresses = NULL;
	FLMUINT *		puiTmpBlocks = NULL;
	FLMUINT 			uiNumChildBlocks = 0;
	FLMUINT			uiNumTmpBlks;
	FLMUINT			uiBlkAddressArrayAllocSize = 0;
	POS_KEY_p		pKeys = NULL;
	FLMUINT			uiNumKeys = 0;
	FLMUINT			uiKeyArrayAllocSize = 0;
	FLMBOOL			bHighKeyInRange = FALSE;

	FSInitStackCache( &stack[ 0], BH_MAX_LEVELS);

	// Check to verify that it is possible to set up an array of positioning keys
	// for this query.  The following conditions must be met:
	// 1) The query must use one and only one index
	// 2) The criteria must be solvable using only the index keys
	// 3)  The selection criteria cannot include DRNs.

	if (((pSubQuery = pCursor->pSubQueryList) == NULL) ||
		 pSubQuery->pNext ||
		 pSubQuery->OptInfo.eOptType != QOPT_USING_INDEX ||
		 pSubQuery->OptInfo.bDoRecMatch ||
		 pSubQuery->bHaveDrnFlds)
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Free the existing key array, if there is one
	
	if (pCursor->pPosKeyArray)
	{
		flmCurFreePosKeys( pCursor);
	}

	// Get the necessary LFILE and IXD information from the subquery index.

	if (RC_BAD( rc = fdictGetIndex(
		pDb->pDict, pDb->pFile->bInLimitedMode,
		pSubQuery->OptInfo.uiIxNum, &pLFile, &pIxd)))
	{
		goto Exit;
	}

	// Set up a B-tree stack structure and get the root block in the index
	// B-tree.

	pStack->pKeyBuf = &ucKeyBuf [0];
	
	// If no root block returned from FSGetRootBlock, the array will be
	// returned empty, with rc set to success.
	
	if (RC_BAD( rc = FSGetRootBlock( pDb, &pLFile, &TmpLFile, pStack)))
	{
		if (rc == FERR_NO_ROOT_BLOCK)
		{
			flmAssert( pLFile->uiRootBlk == BT_END);
			rc = FERR_OK;
		}
		goto Exit;
	}
	uiNumTmpBlks = 1;

	// Extract the array of positioning keys by working down the B-tree
	// from the root block.  This loop will terminate when all levels of
	// the B-Tree have been processed, or when enough keys have been
	// found to populate the array.
	// NOTE: pSubQuery->pPosKeyPool has been initialized at a higher level.
	
	for(;;)
	{
		FLMUINT	uiBlkCnt = 0;
		
		// Work across the present level of the B-Tree from right to left.
		
		for(;;)
		{
		
			// This function moves across a database block from the leftmost
			// element to the rightmost, checking each key to see if it is
			// found in the query's result set.  If it is, it is added to a
			// list of possible positioning keys, and its pointers to child
			// blocks in the B-Tree are also kept.  In the event that not
			// enough keys are found at a given level in the B-Tree, the list
			// of child block pointers is used to work through the next level
			// of the B-Tree.
			
			if (RC_BAD( rc = flmExamineBlock( pCursor, pIxd, pStack->pBlk,
											 pSubQuery->pFSIndexCursor,
											 &puiChildBlockAddresses,
											 &uiNumChildBlocks,
											 &uiBlkAddressArrayAllocSize,
											 &pKeys, &uiNumKeys, &uiKeyArrayAllocSize,
											 &bHighKeyInRange)))
			{
				goto Exit;
			}
			uiBlkCnt++;
			
			// uiNumTmpBlks has the number of blocks to be processed at the
			// current level of the B-Tree.  When those have been processed,
			// break out of this loop and go to the next level of the B-Tree.
			
			if (uiBlkCnt == uiNumTmpBlks)
			{
				break;
			}
			if (RC_BAD( rc = FSGetBlock( pDb, pLFile, puiTmpBlocks[ uiBlkCnt],
										pStack )))
			{
				goto Exit;
			}
		}

		// If we're not on the leaf level, and we have at least
		// FLM_MIN_POS_KEYS - 1 keys, we need to go out and evaluate
		// the last key at the leaf level.
		
		if (uiNumKeys >= FLM_MIN_POS_KEYS - 1 &&
			 bHighKeyInRange && uiNumChildBlocks)
		{
			if (RC_BAD( rc = flmGetLastKey( pDb, pCursor, pIxd, pLFile,
										puiChildBlockAddresses [uiNumChildBlocks - 1],
										&pKeys, &uiNumKeys, &uiKeyArrayAllocSize)))
			{
				goto Exit;
			}
		}
		
		// If we have enough keys, or if we have reached the last level of the
		// B-tree, load up the subquery key array and quit.
			
		if ((uiNumKeys >= FLM_MIN_POS_KEYS) || !uiNumChildBlocks)
		{
			rc = flmLoadPosKeys( pCursor, pKeys, uiNumKeys,
								(FLMBOOL)((uiNumChildBlocks == 0)
											 ? TRUE
											 : FALSE));
			goto Exit;
		}
		
		// If not enough keys, go to the next level of the B-tree and traverse
		// it to find keys.  This should be done down to the last level.
		
		else
		{
			FLMUINT		uiKeyCnt;

			f_free( &puiTmpBlocks);
			puiTmpBlocks = puiChildBlockAddresses;
			uiNumTmpBlks = uiNumChildBlocks;
			puiChildBlockAddresses = NULL;
			uiNumChildBlocks = uiBlkAddressArrayAllocSize = 0;
			for (uiKeyCnt = 0; uiKeyCnt < uiNumKeys; uiKeyCnt++)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
			}
			f_free( &pKeys);
			pKeys = NULL;
			uiNumKeys = 0;
			uiKeyArrayAllocSize = 0;
			if (RC_BAD( rc = FSGetBlock( pDb, pLFile,
										 puiTmpBlocks[ 0],
										 pStack )))
			{
				goto Exit;
			}
			bHighKeyInRange = FALSE;
		}
	}
	
Exit:
	if ( pKeys)
	{
		if (RC_BAD( rc))
		{
			for ( FLMUINT uiKeyCnt = 0; uiKeyCnt < uiNumKeys; uiKeyCnt++)
			{
				f_free( &pKeys[ uiKeyCnt].pucKey);
			}
		}
		f_free( &pKeys);
	}
	f_free( &puiChildBlockAddresses);
	f_free( &puiTmpBlocks);
	FSReleaseStackCache( stack, BH_MAX_LEVELS, FALSE);

	return( rc);
}

/****************************************************************************
Desc: Gets a set of positioning keys for a particular subquery.
****************************************************************************/
RCODE flmCurSetupPosKeyArray(
	CURSOR_p	pCursor
	)
{
	RCODE	rc = FERR_OK;
	FDB_p	pDb = NULL;
	
	// Optimize the subqueries as necessary

	if (!pCursor->bOptimized)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}
	
	// Set up the pDb

	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}
	
	// Set up array of positioning keys.

	if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
	{
		goto Exit;
	}
Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_CONFIG, pDb, rc);
	}
	return( rc);
}

/****************************************************************************
Desc: Gets the approximate percentage position of a passed-in key within a
		cursor's result set.
****************************************************************************/
RCODE flmCurGetPercentPos(
	CURSOR_p			pCursor,
	FLMUINT *		puiPrcntPos
	)
{
	RCODE				rc = FERR_OK;
	FDB_p				pDb = NULL;
	IXD_p				pIxd;
	POS_KEY_p		pPosKeyArray;
	POS_KEY			CompKey;
	FLMUINT			uiLowOffset;
	FLMUINT			uiMidOffset;
	FLMUINT			uiHighOffset;
	FLMUINT			uiIntervalSize;
	FLMUINT			uiRFactor;
	FLMINT			iCmp;
	FLMUINT			uiContainer;

	// Optimize the subqueries as necessary

	if (!pCursor->bOptimized)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}

	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}
	
	// If no array of positioning keys exists in the subquery, set one up.

	if (!pCursor->uiNumPosKeys)
	{
		if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
		{
			goto Exit;
		}
			
		// If no positioning keys exist, either the index or the result set
		// is empty.  Return NOT_FOUND.
		
		if (!pCursor->uiNumPosKeys)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}

	// If the number of positioning keys is 1, the position is 0%.

	if (pCursor->uiNumPosKeys == 1)
	{
		*puiPrcntPos = 0;
		goto Exit;
	}

	pPosKeyArray = pCursor->pPosKeyArray;

	if (pCursor->uiNumPosKeys == 2)
	{
		uiIntervalSize = FLM_MAX_POS_KEYS;
		uiRFactor = 0;
	}
	else
	{
		uiIntervalSize = FLM_MAX_POS_KEYS / (pCursor->uiNumPosKeys - 1);
		uiRFactor = FLM_MAX_POS_KEYS % (pCursor->uiNumPosKeys - 1);
	}

	// DEFECT 84741 -- only want to return a position of 1 for the second key
	// if the positioning key array is full.
	
	// Get an IXD, then convert the passed-in key from GEDCOM format to a
	// buffer containing the key in the FLAIM internal format.
	
	if (RC_BAD( rc = fdictGetIndex( pDb->pDict,
								pDb->pFile->bInLimitedMode,
								pCursor->pSubQueryList->OptInfo.uiIxNum,
								NULL, &pIxd)))
	{
		goto Exit;
	}

	if (pCursor->ReadRc == FERR_BOF_HIT)
	{
		*puiPrcntPos = 0;
		goto Exit;
	}
	if (pCursor->ReadRc == FERR_EOF_HIT)
	{
		*puiPrcntPos = FLM_MAX_POS_KEYS;
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}
	if (RC_BAD( rc = pCursor->pSubQueryList->pFSIndexCursor->currentKeyBuf(
								pDb, &pDb->TempPool, &CompKey.pucKey,
								&CompKey.uiKeyLen, &CompKey.uiDrn, &uiContainer)))
	{
		if (rc == FERR_EOF_HIT || rc == FERR_BOF_HIT || rc == FERR_NOT_FOUND)
		{
			rc = FERR_OK;
			*puiPrcntPos = 0;
		}
		goto Exit;
	}
	flmAssert( uiContainer == pCursor->uiContainer);

	// If a set position call has been performed, and no reposisioning has
	// been done since, check the passed-in key to see if it matches the
	// key returned from the set position call. If so, return the percent
	// passed in on the set position call. This is to create some symmetry
	// where the user calls set position, then takes the resulting key and
	// passes it back into a get position call.

	if (pCursor->bUsePrcntPos &&
		 pCursor->uiLastPrcntPos <= FLM_MAX_POS_KEYS)
	{
		if (flmPosKeyCompare( &pPosKeyArray[ pCursor->uiLastPrcntOffs],
						&CompKey) == 0)
		{
			*puiPrcntPos = pCursor->uiLastPrcntPos;
			goto Exit;
		}
		pCursor->bUsePrcntPos = FALSE;
	}

	// Do a binary search in the array of positioning keys for the passed-in
	// key. NOTE: the point of this search is to find the closest key <= to
	// the passed- in key. The range of values returned is
	// 0 to FLM_MAX_POS_KEYS (currently defined to be 1000), where 0 and
	// FLM_MAX_POS_KEYS represent	the first and last keys in the query's
	// result set, respectively. Numbers between these two endpoints represent
	// intervals between two keys that are adjacent in the array, but which
	// may have any number of intervening keys in the index.
	
	uiLowOffset = 0;
	uiHighOffset = pCursor->uiNumPosKeys - 1;
	for(;;)
	{
		if (uiLowOffset == uiHighOffset)
		{
			uiMidOffset = uiLowOffset;

			// Defect #84741 (fix after failing regression test -
			// zeroeth object was always returning position 1).
			// Must do final comparison to determine which side of
			// the positioning key our key falls on.  Remember,
			// the positioning key represents all keys that are
			// LESS THAN OR EQUAL to it.  Thus, if this key is
			// greater than it, we should use the next positioning
			// key.

			if ((flmPosKeyCompare( &pPosKeyArray[ uiMidOffset],
							&CompKey) < 0) &&
				 (uiMidOffset < pCursor->uiNumPosKeys - 1))
			{
				uiMidOffset++;
			}
			break;
		}
		
		uiMidOffset = (FLMUINT)((uiHighOffset + uiLowOffset) / 2);
	
		iCmp = flmPosKeyCompare( &pPosKeyArray[ uiMidOffset], &CompKey);

		if( iCmp < 0)
		{
			uiLowOffset = uiMidOffset + 1;
		}
		else if( iCmp > 0)
		{
			if( uiMidOffset == uiLowOffset)
			{
				break;
			}
			else
			{
				uiHighOffset = uiMidOffset - 1;
			}
		}
		else
		{
			break;
		}
	}

	// DEFECT 84741 -- the first object should only return a position of 1
	// if there are FLM_MAX_POS_KEYS positioning keys in the array.

	if (uiMidOffset == 0 ||
		 (uiMidOffset == 1 && 
		  pCursor->uiNumPosKeys == FLM_MAX_POS_KEYS + 1))
	{
		*puiPrcntPos = uiMidOffset;
	}
	else if (uiMidOffset == pCursor->uiNumPosKeys - 1)
	{
		*puiPrcntPos = FLM_MAX_POS_KEYS;
	}
	else if (uiMidOffset <= uiRFactor)
	{
		*puiPrcntPos = uiMidOffset * (uiIntervalSize + 1);
	}
	else if (uiRFactor)
	{
		*puiPrcntPos = uiRFactor * (uiIntervalSize + 1) +
							(uiMidOffset - uiRFactor) * uiIntervalSize;
	}
	else
	{
		*puiPrcntPos = uiMidOffset * uiIntervalSize;
	}
	
Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_GET_CONFIG, pDb, rc);
	}
	
	return( rc);
}

/****************************************************************************
Desc: Sets a query's position to a percentage represented by one of an array
		of positioning keys.
****************************************************************************/
RCODE flmCurSetPercentPos(
	CURSOR_p			pCursor,
	FLMUINT			uiPrcntPos
	)
{
	RCODE				rc = FERR_OK;
	FDB_p				pDb = NULL;
	FLMUINT			uiPrcntOffs;
	FLMUINT			uiIntervalSize;
	FLMUINT			uiRFactor;
	SUBQUERY_p		pSubQuery = NULL;
	POS_KEY_p		pPosKey;
	
	// Optimize the subqueries as necessary

	if (!pCursor->bOptimized)
	{
		if (RC_BAD( rc = flmCurPrep( pCursor)))
		{
			goto Exit;
		}
	}
	
	// Check the value for the percentage position.  Should be between
	// 0 and FLM_MAX_POS_KEYS.
	
	flmAssert( uiPrcntPos <= FLM_MAX_POS_KEYS);
	
	// Initialize some variables
	
	pCursor->uiLastRecID = 0;
	pDb = pCursor->pDb;
	if (RC_BAD(rc = flmCurDbInit( pCursor)))
	{
		goto Exit;
	}

	// If no array of positioning keys exists in the subquery, set one up.

	if (!pCursor->uiNumPosKeys)
	{
		if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
		{
			goto Exit;
		}
			
		// If no positioning keys exist, either the index or the result set
		// is empty.  Return BOF or EOF.
		
		if (!pCursor->uiNumPosKeys)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	}

	pSubQuery = pCursor->pSubQueryList;

Retry:

	// Calculate the percent position using the following rules:
	//	1) If the number of positioning keys is 1, the position is 0%.
	//	2) If the number of positioning keys is 2, the position is either 0% or
	//		FLM_MAX_POS_KEYS.
	// 3) If there are more than 2 positioning keys, calculate the interval into
	//		which the percentage position falls.

	if (pCursor->uiNumPosKeys == 1)
	{
		uiPrcntOffs = 0;
	}
	else
	{
		if (pCursor->uiNumPosKeys == 2)
		{
			uiIntervalSize = FLM_MAX_POS_KEYS;
			uiRFactor = 0;
		}
		else
		{
			uiIntervalSize = FLM_MAX_POS_KEYS / (pCursor->uiNumPosKeys - 1);
			uiRFactor = FLM_MAX_POS_KEYS % (pCursor->uiNumPosKeys - 1);
		}
		
		// Convert passed-in number to an array offset.

		if (uiPrcntPos)
		{
			if (uiPrcntPos == 0 || pCursor->uiNumPosKeys == FLM_MAX_POS_KEYS + 1)
			{
				uiPrcntOffs = uiPrcntPos;
			}
			else if( uiPrcntPos == FLM_MAX_POS_KEYS)
			{
				uiPrcntOffs = pCursor->uiNumPosKeys - 1;
			}
			else if( uiPrcntPos <= uiRFactor * (uiIntervalSize + 1))
			{
				uiPrcntOffs = uiPrcntPos / (uiIntervalSize + 1);
			}
			else
			{
				uiPrcntOffs = uiRFactor +
									(uiPrcntPos - (uiIntervalSize + 1) * uiRFactor) /
									uiIntervalSize;
			}
		}
		else
		{
			uiPrcntOffs = 0;
		}
	}
	pPosKey = &pCursor->pPosKeyArray [uiPrcntOffs];

	// If the keys were generated from the leaf level, we can
	// position directly to them.  If not, we must call the
	// positionToDomain routine.

	if (pCursor->bLeafLevel)
	{
		rc = pSubQuery->pFSIndexCursor->positionTo( pDb, pPosKey->pucKey,
					pPosKey->uiKeyLen, pPosKey->uiDrn);
	}
	else
	{
		rc = pSubQuery->pFSIndexCursor->positionToDomain( pDb,
					pPosKey->pucKey, pPosKey->uiKeyLen,
					DRN_TO_DOMAIN( pPosKey->uiDrn));
	}

	if (RC_BAD( rc))
	{
		RCODE	saveRc;

		if (rc != FERR_BOF_HIT && rc != FERR_EOF_HIT && rc != FERR_NOT_FOUND)
		{
			goto Exit;
		}
			
		// If the positioning key was not found, the database has undergone
		// significant change since the array of positioning keys was generated.
		// Try to regenerate the array and reposition.

		saveRc = rc;
		if (RC_BAD( rc = flmCurGetPosKeys( pDb, pCursor)))
		{
			goto Exit;
		}
		if (pCursor->pPosKeyArray [0].pucKey == NULL)
		{
			rc = saveRc;
			goto Exit;
		}
		goto Retry;
	}

	// Retrieve the current key and DRN from the index cursor.

	if (RC_BAD( rc = pSubQuery->pFSIndexCursor->currentKey( pDb,
							&pSubQuery->pRec, &pSubQuery->uiDrn)))
	{
		goto Exit;
	}
	pSubQuery->bFirstReference = FALSE;
	pSubQuery->uiCurrKeyMatch = FLM_TRUE;

	// These should have already been set by the call to currentKey.

	flmAssert( pSubQuery->pRec->getContainerID() == pCursor->uiContainer);
	flmAssert( pSubQuery->pRec->getID() == pSubQuery->uiDrn);

	pSubQuery->bRecIsAKey = TRUE;
	
	// If we got this far, the positioning operation was a success.  Set
	// the query return code to success so it doesn't mess up subsequent
	// read operations.

	pCursor->uiLastRecID = pSubQuery->uiDrn;
	pCursor->rc = FERR_OK;
	pCursor->uiLastPrcntPos = uiPrcntPos;
	pCursor->uiLastPrcntOffs = uiPrcntOffs;
	pCursor->bUsePrcntPos = TRUE;

Exit:
	if (pDb)
	{
		flmExit( FLM_CURSOR_CONFIG, pDb, rc);
	}
	
	return( rc);
}
