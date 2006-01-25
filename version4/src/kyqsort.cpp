//-------------------------------------------------------------------------
// Desc:	Key sorting - for indexing.
// Tabs:	3
//
//		Copyright (c) 1990-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kyqsort.cpp 12315 2006-01-19 15:16:37 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define	KY_SWAP( pKrefTbl, leftP, rightP)	\
	pTempKref = pKrefTbl [leftP]; \
	pKrefTbl [leftP] = pKrefTbl [rightP]; \
	pKrefTbl [rightP] = pTempKref
	
FSTATIC FLMINT _KrefCompare(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY_p	pKreftA,
	KREF_ENTRY_p	pKreftB);

FSTATIC RCODE KYAddUniqueKeys(
	FDB *				pDb);

FSTATIC RCODE _KrefQuickSort(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY_p *	pEntryTbl,
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds);

FSTATIC RCODE _KrefKillDups(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY_p *	pKrefTbl,
	FLMUINT *		puiKrefTotalRV);

/****************************************************************************
Desc:		Checks if the current database has any UNIQUE indexes that need 
			to checked. Also does duplicate processing for the record.
****************************************************************************/
RCODE	KYProcessDupKeys(
	FDB *				pDb,
	FLMBOOL			bHadUniqueKeys
	)
{
	RCODE				rc = FERR_OK;
	KREF_CNTRL_p	pKrefCntrl = &pDb->KrefCntrl;
	FLMUINT			uiCurRecKrefCnt;

	pKrefCntrl->uiTrnsSeqCntr++;	

	//  Sort and remove duplicates from the list of this record.

	uiCurRecKrefCnt = pKrefCntrl->uiCount - pKrefCntrl->uiLastRecEnd;

	if( uiCurRecKrefCnt > 1)
	{
		FLMUINT	uiSortFlags = KY_DUP_CHK_SRT;

		/* NLM - release cpu. - the QuickSort can take a while */

		f_yieldCPU();

		if( RC_BAD( rc = _KrefQuickSort( &uiSortFlags,
											&pKrefCntrl->pKrefTbl [pKrefCntrl->uiLastRecEnd],
											0, uiCurRecKrefCnt - 1)))
		{
			goto Exit;
		}

		/* Found any duplicates? */

		if( uiSortFlags & KY_DUPS_FOUND)
		{
			if( RC_BAD( rc = _KrefKillDups( &uiSortFlags,
										&pKrefCntrl->pKrefTbl [pKrefCntrl->uiLastRecEnd],
										&uiCurRecKrefCnt)))
			{
				goto Exit;
			}
			pKrefCntrl->uiCount = pKrefCntrl->uiLastRecEnd + uiCurRecKrefCnt;
		}
	}

	if( bHadUniqueKeys)
	{
		/* Now check the keys for uniquness in table, and database. */

		if( RC_BAD(rc = KYAddUniqueKeys( pDb)))
		{
			goto Exit;
		}
	}
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Remove anything that was put into the KREF table by the current
			record update operation.
****************************************************************************/
void KYAbortCurrentRecord(
	FDB *			pDb)
{
	flmAssert( pDb->KrefCntrl.bKrefSetup);

	// Reset the CDL and pIxHasCmpKeys tables

	if (pDb->pDict->uiIfdCnt)
	{
		f_memset( pDb->KrefCntrl.ppCdlTbl, 0,
					pDb->pDict->uiIfdCnt * sizeof( CDL_p));
	}
	if (pDb->pDict->uiIxdCnt)
	{
		f_memset( pDb->KrefCntrl.pIxHasCmpKeys, 0, pDb->pDict->uiIxdCnt);
	}
	pDb->KrefCntrl.uiCount = pDb->KrefCntrl.uiLastRecEnd;
	GedPoolReset( pDb->KrefCntrl.pPool, pDb->KrefCntrl.pReset);
}

/****************************************************************************
Desc:		Commit (write out) all reference lists from the CURRENT pDb.
			Will take care of optimially freeing or resetting memory.
Note:		Before 11/96 there was code to not write out references that did 
			not belong to the specific commited DB transaction.  
			This isn't saving us any time so just output everything so we don't
			have really hard bugs to debug.
****************************************************************************/
RCODE KYKeysCommit(
	FDB *			pDb,
	FLMBOOL		bCommittingTrans)
{
	RCODE				rc = FERR_OK;
	KREF_CNTRL_p	pKrefCntrl = &pDb->KrefCntrl;

	// If KrefCntrl has not been initialized, there is no
	// work to do.

	if( pKrefCntrl->bKrefSetup)
	{
		LFILE *			pLFile = NULL;
		FLMUINT			uiTotal = pKrefCntrl->uiLastRecEnd;
		KREF_ENTRY_p	pKref;
		KREF_ENTRY_p * pKrefTbl = pKrefCntrl->pKrefTbl;
		FLMUINT			uiKrefNum;
		FLMUINT			uiLastIxNum;

		// We should not have reached this point if bAbortTrans is TRUE

		flmAssert( RC_OK( pDb->AbortRc));

		// uiTotal and uiLastRecEnd must be the same at this point.
		// If not, we have a bug.

		flmAssert( uiTotal == pKrefCntrl->uiLastRecEnd);

		// Sort the KREF table, if it contains more than one record and key.
		// This will sort all keys from the same index the same.

		if ((uiTotal > 1) && (pKrefCntrl->uiTrnsSeqCntr > 1))
		{
			FLMUINT	uiQsortFlags = KY_FINAL_SRT;

			// NLM - release cpu - the quick sort can really pig out the CPU.

			f_yieldCPU();

			if (RC_BAD( rc = _KrefQuickSort( &uiQsortFlags, pKrefTbl, 0, uiTotal - 1 )))
				goto Exit;
		}

		// Initialization of FOR loop
		uiLastIxNum = 0;

		// Loop through the KREF table outputting all keys
		for( uiKrefNum = 0; uiKrefNum < uiTotal; uiKrefNum++)
		{
			pKref = pKrefTbl [uiKrefNum];

			// See if the LFILE changed

			flmAssert( pKref->ui16IxNum  > 0 && 
				pKref->ui16IxNum < FLM_UNREGISTERED_TAGS); // Sanity check

			if( pKref->ui16IxNum != uiLastIxNum)
			{
				uiLastIxNum = pKref->ui16IxNum;
				if( RC_BAD( rc = fdictGetIndex(
						pDb->pDict, pDb->pFile->bInLimitedMode,
						uiLastIxNum, &pLFile, NULL, TRUE)))
				{
					goto Exit;
				}
			}

			// Flush the key to the index

			if( RC_BAD(rc = FSRefUpdate( pDb, pLFile, pKref)))
			{
				goto Exit;
			}
		}
						
		if (bCommittingTrans)
		{
			KrefCntrlFree( pDb);
		}
		else
		{
			// Empty the table out so we can add more keys in this trans.

			GedPoolReset( pKrefCntrl->pPool, NULL);
			pKrefCntrl->uiCount =
			pKrefCntrl->uiTotalBytes =
			pKrefCntrl->uiLastRecEnd =
			pKrefCntrl->uiTrnsSeqCntr = 0;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Adds all unique key values.  Backs out on any unique error so that
			the transaction may continue.
Notes:	All duplicates have been removed as well as matching keys.
****************************************************************************/
FSTATIC RCODE KYAddUniqueKeys(
	FDB *		pDb)
{
	RCODE				rc = FERR_OK;
	KREF_CNTRL_p	pKrefCntrl = &pDb->KrefCntrl;
	KREF_ENTRY_p *	pKrefTbl = pKrefCntrl->pKrefTbl;
	KREF_ENTRY_p	pKref;
	FLMUINT			uiCurKrefNum, uiPrevKrefNum;
	FLMUINT			uiTargetCount;
	FLMUINT			uiLastIxNum;
	LFILE *			pLFile;
	FLMBOOL			bUniqueErrorHit = FALSE;

	// Unique indexes can't be built in the background

	flmAssert( !(pDb->uiFlags & FDB_BACKGROUND_INDEXING));

	// Start at the first key for this current record checking for
	// all keys that belong to a unique index.  We must keep all keys around
	// until the last key is added/delete so that we can back out all of the
	// changes on a unique error.

	for( uiCurKrefNum = pKrefCntrl->uiLastRecEnd, 
			uiLastIxNum = 0,
			uiTargetCount = pKrefCntrl->uiCount;
		  uiCurKrefNum < uiTargetCount;
		  // Increment uiCurKrefNum at bottom of loop
		  )
	{
		pKref = pKrefTbl [uiCurKrefNum];

		if( pKref->uiFlags & KREF_UNIQUE_KEY)
		{
			flmAssert( pKref->ui16IxNum  > 0 && 
				pKref->ui16IxNum < FLM_UNREGISTERED_TAGS); // Sanity check

			if( pKref->ui16IxNum != uiLastIxNum)
			{
				uiLastIxNum = pKref->ui16IxNum;
				if (RC_BAD( rc = fdictGetIndex(
						pDb->pDict, pDb->pFile->bInLimitedMode,
						uiLastIxNum, &pLFile, NULL)))
				{
					// Return the index offline error - should not happen
					flmAssert( rc != FERR_INDEX_OFFLINE);
					goto Exit;
				}
			}

			// Flush the key to the index.

			if( RC_BAD(rc = FSRefUpdate( pDb, pLFile, pKref)))
			{
				pDb->Diag.uiInfoFlags |= FLM_DIAG_INDEX_NUM;
				pDb->Diag.uiIndexNum = pKref->ui16IxNum;

				// Check only for FERR_NOT_UNIQUE

				if( rc != FERR_NOT_UNIQUE)
					goto Exit;

				bUniqueErrorHit = TRUE;

				// Cycle through again backing out all keys.

				uiTargetCount = uiCurKrefNum;
	  			uiCurKrefNum = pKrefCntrl->uiLastRecEnd;
				// Make sure uiCurKrefNum is NOT incremented at the top of loop.
				continue;		
			}
			// Toggle the delete flag so on unique error we can back out.
			// This sets the ADD to DELETE and the DELETE to ADD (0)

			pKref->uiFlags ^= KREF_DELETE_FLAG;	
		}	
		uiCurKrefNum++;
	}	

	if( bUniqueErrorHit)
	{
		rc = RC_SET( FERR_NOT_UNIQUE);
		pKrefCntrl->uiCount = pKrefCntrl->uiLastRecEnd;
	}
	else
	{
		// Scoot ever key down removing the processed keys.

		for( uiCurKrefNum = uiPrevKrefNum = pKrefCntrl->uiLastRecEnd, 
					uiTargetCount = pKrefCntrl->uiCount;
				uiCurKrefNum < uiTargetCount;
				uiCurKrefNum++)
		{
			pKref = pKrefTbl [uiCurKrefNum];
	
			if( !(pKref->uiFlags & KREF_UNIQUE_KEY ))
			{
				pKrefTbl [ uiPrevKrefNum++ ] = pKrefTbl [uiCurKrefNum ];
			}
		}
		pKrefCntrl->uiCount = uiPrevKrefNum;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Compare function used to compare two keys.  The compare is 
			different depending on the sort pass this is on.
Note:		We must compare each item in the structure because UNIX will
			place structure elements in any order it feels.

	(SORT1)	KY_DUP_CHK_SRT ==  pLfd | key | DELETE_FLAG  
	(SORT2)  KY_FINAL_SRT   ==  pLfd | key | DRN | TrnsSeq - set IGNORE on EQ
****************************************************************************/
FSTATIC FLMINT _KrefCompare(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY_p 	pKrefA,
	KREF_ENTRY_p	pKrefB )
{
	FLMUINT			uiMinLen;						/* Minimum key length of A or B */
	FLMINT			iCompare;

	/* Compare (SORT1) #1, (SORT2) #2 - Index Number. */

	if ((iCompare = ((FLMINT) pKrefA->ui16IxNum) - ((FLMINT) pKrefB->ui16IxNum)) != 0)
		return( iCompare);

	/* Compare (SORT1) #2, (SORT2) #3: KEY - including NULL character at end. */
	/* Comparing the NULL character advoids checking the key length. */
	/* VISIT: There could be a BUG where key length should be checked, but
		it has to do with not storing all compound key pieces in the key. */

	uiMinLen = f_min( pKrefA->ui16KeyLen, pKrefB->ui16KeyLen) + 1;
	if ((iCompare = f_memcmp( &pKrefA [1], &pKrefB [1], uiMinLen)) == 0)
	{
		if( *puiQsortFlags & KY_FINAL_SRT)
		{
			/* Compare (SORT2) The DRN so we load by low DRN to high DRN. */

			if( pKrefA->uiDrn < pKrefB->uiDrn)
				return -1;
			else if( pKrefA->uiDrn > pKrefB->uiDrn )
				return 1;

			/*
			Compare (SORT2) Sequence number, so operations occur in
			correct order. - this will ALWAYS set iCompare to -1 or 1.
			It is only possible to have different operations here like
			ADD - DELETE - ADD - DELETE when sorted by uiTrnsSeq.  This
			is why we will set KY_DUPS_FOUND to get rid of duplicates.
			*/

			iCompare = ((FLMINT)pKrefA->uiTrnsSeq) - ((FLMINT)pKrefB->uiTrnsSeq);

		}
		else // if( *puiQsortFlags & KY_DUP_CHK_SRT )
     	{

			/* Compare (SORT1) Operation Flag, Delete or Add. */

			*puiQsortFlags |= KY_DUPS_FOUND;

			/* Sort so the delete elements are first. */
			
			if ((iCompare = ((FLMINT)(pKrefB->uiFlags & KREF_DELETE_FLAG)) -
								 ((FLMINT)(pKrefA->uiFlags & KREF_DELETE_FLAG))) == 0)
			{
				/*  Exact duplicate - will remove later */

				pKrefA->uiFlags |= KREF_EQUAL_FLAG;
				pKrefB->uiFlags |= KREF_EQUAL_FLAG;
			}
			else
			{
				/* Data is same but different operation, (delete then an add). */

				pKrefA->uiFlags |= KREF_IGNORE_FLAG;
				pKrefB->uiFlags |= KREF_IGNORE_FLAG;
			}
		}
	}
	return( iCompare);
}

/***************************************************************************
Desc:		Quick sort an array of KREF_ENTRY_p values.
Notes:	Optimized the above quicksort algorithm.  This is the same code
			as the quick sort in FRSET.C which has lots of comments.  We
			didn't combine the code because a general quick sort would be
			slower for the KREF and I didn't want to change that much code.
****************************************************************************/

FSTATIC RCODE _KrefQuickSort(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY_p *	pEntryTbl,
	FLMUINT			uiLowerBounds,
	FLMUINT			uiUpperBounds)
{
	FLMUINT			uiLBPos, uiUBPos, uiMIDPos;
	FLMUINT			uiLeftItems, uiRightItems;
	KREF_ENTRY_p	pCurEntry, pTempKref;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	pCurEntry = pEntryTbl[ uiMIDPos ];
	for( ;;)
	{
		while( (uiLBPos == uiMIDPos)				// Don't compare with target
			||  ((iCompare = 
						_KrefCompare( puiQsortFlags, pEntryTbl[ uiLBPos], pCurEntry)) < 0))
		{
			if( uiLBPos >= uiUpperBounds) break;
			uiLBPos++;
		}

		while( (uiUBPos == uiMIDPos)				// Don't compare with target
			||  (((iCompare = 
						_KrefCompare( puiQsortFlags, pCurEntry, pEntryTbl[ uiUBPos])) < 0)))
		{
			if( !uiUBPos)	break;
			uiUBPos--;
		}
		
		if( uiLBPos < uiUBPos )			// Interchange and continue loop.
		{
			/* Interchange [uiLBPos] with [uiUBPos]. */

			KY_SWAP( pEntryTbl, uiLBPos, uiUBPos );
			uiLBPos++;						// Scan from left to right.
			uiUBPos--;						// Scan from right to left.
		}
		else									// Past each other - done
		{
			break;
		}
	}
	/* Check for swap( LB, MID ) - cases 3 and 4 */

	if( uiLBPos < uiMIDPos )
	{
		/* Interchange [uiLBPos] with [uiMIDPos] */

		KY_SWAP( pEntryTbl, uiMIDPos, uiLBPos );
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{
		/* Interchange [uUBPos] with [uiMIDPos] */

		KY_SWAP( pEntryTbl, uiMIDPos, uiUBPos );
		uiMIDPos = uiUBPos;
	}

	/* Check the left piece. */

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos )
							? uiMIDPos - uiLowerBounds		// 2 or more
							: 0;
	uiRightItems = (uiMIDPos + 1 < uiUpperBounds )
							? uiUpperBounds - uiMIDPos 		// 2 or more
							: 0;

	if( uiLeftItems < uiRightItems )
	{
		/* Recurse on the LEFT side and goto the top on the RIGHT side. */

		if( uiLeftItems )
		{
			(void) _KrefQuickSort( puiQsortFlags, pEntryTbl, 
				uiLowerBounds, uiMIDPos - 1 );
		}
		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems )	// Compute a truth table to figure out this check.
	{
		/* Recurse on the RIGHT side and goto the top for the LEFT side. */

		if( uiRightItems )
		{
			(void) _KrefQuickSort( puiQsortFlags, pEntryTbl, 
				uiMIDPos + 1, uiUpperBounds );
		}
		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
//Exit:	
	return FERR_OK;
}

/****************************************************************************
Desc:		Kill all duplicate references out of the kref list
Notes:	This will ONLY work if EVERY kref has been compared to its neighbor.
			We may have to compare every neighbor again if the new quick sort
			doesn't work.
****************************************************************************/
FSTATIC RCODE _KrefKillDups(
	FLMUINT *		puiQsortFlags,
	KREF_ENTRY_p *	pKrefTbl,		/* Portion of KREF table where duplicates
												are to be eliminated. */
	FLMUINT *		puiKrefTotalRV)/* Number of elements in portion of KREF table
												where duplicates are to be eliminated.
												It returns the number of elements that
												are left after duplicates are eliminated.*/
{
	FLMUINT			uiTotal = (*puiKrefTotalRV);
	FLMUINT			uiCurKrefNum;
	KREF_ENTRY_p 	pCurKref;
	FLMUINT			uiLastUniqueKrefNum = 0;

	for (uiCurKrefNum = 1; uiCurKrefNum < uiTotal; uiCurKrefNum++)
	{
		pCurKref = pKrefTbl [uiCurKrefNum];

		/*
		If the current KREF equals the last unique one, we can
		remove it from the list by skipping the current entry.
		To check if they are equal, first look at the KREF_EQUAL_FLAGs 
		on both of them.  If both KREFs have this flag set, we still 
		have to call the compare routine.  The flags could have been set for
		two pairs of different keys - such as A, A, B, B.  In this
		sequence of keys, all four KREFs would have the flag set, but
		the 2nd "A" is not equal to the 1st "B" - thus the need for the
		call to krefCompare to confirm that the keys are really equal.
		*/

		if ((pKrefTbl [uiLastUniqueKrefNum]->uiFlags & KREF_EQUAL_FLAG) &&
			 (pCurKref->uiFlags & KREF_EQUAL_FLAG) &&
			 (_KrefCompare( puiQsortFlags, pKrefTbl[uiLastUniqueKrefNum], pCurKref) == 0))
		{
			/* 
			If the current KREF had it's ignore flag set, propagate that
			to the last unique KREF also and remove the current key.
			This will remove all but the first duplicate key.
			This is possible because quick sort may not compare every item.
			*/

			if (pCurKref->uiFlags & KREF_IGNORE_FLAG)
				pKrefTbl [uiLastUniqueKrefNum]->uiFlags |= KREF_IGNORE_FLAG;
		}
		else
		{
			// Increment to the next slot if we like this kref. 
				
			if( !(pKrefTbl [uiLastUniqueKrefNum]->uiFlags & KREF_IGNORE_FLAG))
			{
				uiLastUniqueKrefNum++;
			}

			// Move the item to the current location.

			pKrefTbl [uiLastUniqueKrefNum] = pCurKref;

		}
	}
	if( !(pKrefTbl [uiLastUniqueKrefNum]->uiFlags & KREF_IGNORE_FLAG))
	{
		uiLastUniqueKrefNum++;
	}

	*puiKrefTotalRV = uiLastUniqueKrefNum;	// One based number
//Exit:
	return( FERR_OK);
}
