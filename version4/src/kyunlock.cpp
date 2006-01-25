//-------------------------------------------------------------------------
// Desc:	Unlock/free KREF structures.
// Tabs:	3
//
//		Copyright (c) 1992-2000,2002-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: kyunlock.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define KREF_TBL_SIZE					512			
#define KREF_TBL_THRESHOLD				400
#define KREF_POOL_BLOCK_SIZE			8192
#define KREF_TOTAL_BYTES_THRESHOLD	((KREF_POOL_BLOCK_SIZE * 3) - 250)

/****************************************************************************
Desc:		Setup routine for the KREF_CNTRL structure for record updates.
			Will check to see if all structures, buffers and memory pools 
			need to be allocated: Kref key buffer, CDL table, KrefTbl and pool.
			The goal is to have only one allocation for most small transactions.
			As of Nov 96, each DB will have its own KREF_CNTRL struture so the
			session temp pool does not have to be used.  This means that the
			CDL and cmpKeys arrays do not have to be allocated for each
			record operation (like we did in the session pool).
****************************************************************************/
RCODE KrefCntrlCheck(
	FDB_p				pDb)
{
	RCODE				rc = FERR_OK;			// Set for cleaner code.
	KREF_CNTRL_p	pKrefCntrl;

	pKrefCntrl = &pDb->KrefCntrl;

	/* Check if we need to flush between the records and not during
		the processing of a record.  This simplifies how we reuse the memory.
	*/

	if( pKrefCntrl->bKrefSetup)
	{
		if( (pKrefCntrl->uiCount >= KREF_TBL_THRESHOLD)
		 || (pKrefCntrl->uiTotalBytes >= KREF_TOTAL_BYTES_THRESHOLD))

		{
			if( RC_BAD( rc = KYKeysCommit( pDb, FALSE)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		FLMUINT		uiKrefTblSize = KREF_TBL_SIZE * sizeof(KREF_ENTRY_p);
		FLMUINT		uiCDLSize = pDb->pDict->uiIfdCnt * sizeof( CDL_p);
		FLMUINT		uiIxdSize = pDb->pDict->uiIxdCnt;
		FLMUINT		uiKeyBufSize = MAX_KEY_SIZ + 8;
	
		f_memset( pKrefCntrl, 0, sizeof( KREF_CNTRL));
		pKrefCntrl->bKrefSetup = TRUE;
		if (pDb->uiTransType == FLM_UPDATE_TRANS)
		{
			pKrefCntrl->pPool = &pDb->pFile->krefPool;
			pKrefCntrl->bReusePool = TRUE;
		}
		else
		{
			pKrefCntrl->pPool = &pDb->tmpKrefPool;
			pKrefCntrl->bReusePool = FALSE;
		}

		if (pKrefCntrl->bReusePool)
		{
			GedPoolReset( pKrefCntrl->pPool, NULL);
		}
		else
		{
			GedPoolInit( pKrefCntrl->pPool, KREF_POOL_BLOCK_SIZE);
		}

		if( RC_BAD( rc = f_alloc( uiKrefTblSize,
			&pKrefCntrl->pKrefTbl))
		 || (uiCDLSize && RC_BAD( rc = f_calloc( uiCDLSize,
				&pKrefCntrl->ppCdlTbl)))
		 || (uiIxdSize && RC_BAD( rc = f_calloc( uiIxdSize,
				&pKrefCntrl->pIxHasCmpKeys)))
		 || RC_BAD( rc = f_calloc( uiKeyBufSize,
				&pKrefCntrl->pKrefKeyBuf)))
		{
			KrefCntrlFree( pDb);
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		pKrefCntrl->uiKrefTblSize = KREF_TBL_SIZE;
	}

	pKrefCntrl->pReset = GedPoolMark( pKrefCntrl->pPool); 

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Resets or frees the memory associated with the KREF.
****************************************************************************/
void KrefCntrlFree(
	FDB_p	pDb)
{
	KREF_CNTRL_p	pKrefCntrl = &pDb->KrefCntrl;

	if( pKrefCntrl->bKrefSetup)
	{
		if (pKrefCntrl->bReusePool)
		{
			GedPoolReset( pKrefCntrl->pPool, NULL);
		}
		else
		{
			GedPoolFree( pKrefCntrl->pPool);
		}

		if( pKrefCntrl->pKrefTbl)
		{
			f_free( &pKrefCntrl->pKrefTbl);
		}

		if( pKrefCntrl->ppCdlTbl)
		{
			f_free( &pKrefCntrl->ppCdlTbl);
		}

		if( pKrefCntrl->pIxHasCmpKeys)
		{
			f_free( &pKrefCntrl->pIxHasCmpKeys);
		}

		if( pKrefCntrl->pKrefKeyBuf)
		{
			f_free( &pKrefCntrl->pKrefKeyBuf);
		}

		// Just set everyone back to zero.

		f_memset( pKrefCntrl, 0, sizeof(KREF_CNTRL));
	}
}
