//-------------------------------------------------------------------------
// Desc:	Check database for corruptions.
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
// $Id: flchkdb.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE chkGetDictInfo(
	DB_INFO *		pDbInfo);

FSTATIC RCODE chkVerifyBlkChain(
	DB_INFO *			pDbInfo,
	BLOCK_INFO *		pBlkInfo,
	eCorruptionLocale	eLocale,
	FLMUINT				uiFirstBlkAddr,
	FLMUINT				uiBlkType,
	FLMUINT *			puiBlkCount,
	FLMBOOL *			pbStartOverRV);

FSTATIC RCODE chkVerifyLFHBlocks(
	DB_INFO *		pDbInfo,
	FLMBOOL *		pbStartOverRV);

FSTATIC RCODE chkVerifyAvailList(
	DB_INFO *		pDbInfo,
	FLMBOOL *		pbStartOverRV);

/*API~***********************************************************************
Desc:		Checks for physical corruption in a FLAIM database.
Note:		The routine verifies the database by first reading through
		 	the database to count certain block types which are in linked lists.
			It then verifies the linked lists.  It also verifies the B-TREEs
			in the database.  The reason for the first pass is so that when we
			verify the linked lists, we can keep ourselves from getting into
			an infinite loop if there is a loop in the lists.
*END************************************************************************/
FLMEXP RCODE FLMAPI FlmDbCheck(
	HFDB						hDb,
	const char *			pDbFileName,
	const char *			pszDataDir,
	const char *			pRflDir,
	FLMUINT					uiCheckFlags,
	POOL *					pPool,
	DB_CHECK_PROGRESS *	pCheckProgress,
	STATUS_HOOK				fnStatusFunc,
	void *					AppArg)
{
	RCODE						rc = FERR_OK;
	F_SuperFileHdl *		pSFileHdl = NULL;
	FLMBYTE *				pBlk = NULL;
	FLMUINT					uiFileEnd;
	FLMUINT					uiBlockSize;
	DB_CHECK_PROGRESS		Progress;
	FLMBOOL					bOpenedDb = FALSE;
	FDB *						pDb = (FDB *)hDb;
	FLMBOOL					bIgnore;
	FLMUINT					uiLoop;
	FLMUINT					uiTmpSize;
	FLMBOOL					bStartOver;
	FLMBOOL					bOkToCloseTrans = FALSE;
	DB_INFO *				pDbInfo;
	POOL						localPool;
	void *					pvDbInfoMark;

	if( hDb != HFDB_NULL && IsInCSMode( hDb))
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto ExitCS;
	}
	
	GedPoolInit( &localPool, 512);

	if( !pPool)
	{
		pPool = &localPool;
	}

	pDbInfo = (DB_INFO *)GedPoolCalloc( pPool, sizeof( DB_INFO));
	
	pvDbInfoMark = GedPoolMark( pPool);

	if( hDb == HFDB_NULL)
	{
		if( RC_BAD( rc = FlmDbOpen( pDbFileName, pszDataDir, 
			pRflDir, 0, NULL, &hDb)))
		{
			goto Exit;
		}
		
		bOpenedDb = TRUE;
		pDb = (FDB *)hDb;
	}

	pDbInfo->fnStatusFunc = fnStatusFunc;
	pDbInfo->LastStatusRc = FERR_OK;
	pDbInfo->pDb = pDb;

	if( pCheckProgress)
	{
		pDbInfo->pProgress = pCheckProgress;
	}
	else
	{
		pDbInfo->pProgress = &Progress;
	}

	f_memset( pDbInfo->pProgress, 0, sizeof( DB_CHECK_PROGRESS));

	pDbInfo->bDbInitialized = TRUE;
	if( RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS, 0, 0, &bIgnore)))
	{
		goto Exit;
	}

	// Initialize the information block and Progress structure.

	// Because FlmDbCheck will start and stop read transactions during its
	// processing we can't allow any existing read transactions to exist.
	// However, it is OK for an update transaction to be in progress.  An update
	// transaction will NOT be stopped and restarted.  The only reason a read
	// transaction may be stopped and restarted is if we get an old view
	// error - something that cannot normally happen during an update
	// transaction.

	if( flmGetDbTransType( pDb) == FLM_READ_TRANS)
	{
		// If it is an invisible transaction, it may be aborted.

		if( pDb->uiFlags & FDB_INVISIBLE_TRANS)
		{
			if( RC_BAD( rc = flmAbortDbTrans( pDb)))
			{
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET( FERR_TRANS_ACTIVE);
			goto Exit;
		}
	}

	// Since we know that the check will start read transactions
	// during its processing, set the flag to indicate that the KRef table
	// should be cleaned up on exit if we are still in a read transaction.

	bOkToCloseTrans = TRUE;
	uiBlockSize = pDb->pFile->FileHdr.uiBlockSize;

	// Check does its own reads using the handle to the file.

	pSFileHdl = pDb->pSFileHdl;
	pDbInfo->pSFileHdl = pSFileHdl;

	// Allocate memory to use for reading through the data blocks.

	if( RC_BAD( rc = f_alloc( uiBlockSize, &pBlk)))
	{
		goto Exit;
	}

Begin_Check:

	// Initialize all statistics in the DB_INFO structure.

	rc = FERR_OK;
	bStartOver = FALSE;

	GedPoolReset( pPool, pvDbInfoMark);
	pDbInfo->pProgress->bPhysicalCorrupt = FALSE;
	pDbInfo->pProgress->bLogicalIndexCorrupt = FALSE;
	pDbInfo->pProgress->ui64DatabaseSize = 0;
	pDbInfo->pProgress->uiNumFields = 0;
	pDbInfo->pProgress->uiNumIndexes = 0;
	pDbInfo->pProgress->uiNumContainers = 0;
	pDbInfo->pProgress->uiNumLogicalFiles = 0;
	pDbInfo->pLogicalFiles = NULL;
	pDbInfo->pProgress->pLfStats = NULL;
	pDbInfo->uiFlags = uiCheckFlags;
	pDbInfo->uiMaxLockWait = 15;
	pDbInfo->bStartedUpdateTrans = FALSE;
	f_memset( &pDbInfo->pProgress->AvailBlocks, 0, sizeof( BLOCK_INFO));
	f_memset( &pDbInfo->pProgress->LFHBlocks, 0, sizeof( BLOCK_INFO));
	f_memset( &pDbInfo->FileHdr, 0, sizeof( FILE_HDR));
	f_memset( &Progress, 0, sizeof( DB_CHECK_PROGRESS));
	Progress.AppArg = AppArg;

	// Get the dictionary information for the file

	if (RC_BAD( rc = chkGetDictInfo( pDbInfo)))
	{
		goto Exit;
	}

	Progress.ui64BytesExamined = 0;

	for (uiLoop = 1;
		  uiLoop <= 
			  MAX_DATA_BLOCK_FILE_NUMBER(
					pDb->pFile->FileHdr.uiVersionNum);
		  uiLoop++)
	{
		if (RC_BAD( pSFileHdl->GetFileSize( uiLoop, &uiTmpSize)))
		{
			break;
		}
		Progress.ui64DatabaseSize += (FLMUINT64)uiTmpSize;
	}

	// See if we have a valid end of file

	uiFileEnd = pDbInfo->pDb->LogHdr.uiLogicalEOF;
	if( FSGetFileOffset( uiFileEnd) % uiBlockSize != 0)
	{
		if( RC_BAD( rc = chkReportError( pDbInfo, FLM_BAD_FILE_SIZE,
			LOCALE_NONE, 0, 0, 0xFF, uiFileEnd, 0, 0, 0, 0xFFFF, 0, NULL)))
		{
			goto Exit;
		}
	}
	else if (Progress.ui64DatabaseSize < FSGetSizeInBytes(
													pDbInfo->pDb->pFile->uiMaxFileSize,
													uiFileEnd))
	{
		Progress.ui64DatabaseSize = FSGetSizeInBytes(
													pDbInfo->pDb->pFile->uiMaxFileSize,
													uiFileEnd);
	}

	// Verify and count the LFH and PCODE blocks, B-Trees, and the
	// AVAIL list.

	if( RC_BAD( rc = chkVerifyLFHBlocks( pDbInfo, &bStartOver)))
	{
		goto Exit;
	}
	
	if( bStartOver)
	{
		goto Begin_Check;
	}

	// Check the b-trees.
	
	if (RC_BAD( rc = chkVerifyBTrees( pDbInfo, pPool, &bStartOver)))
	{
		goto Exit;
	}
	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Check the avail list.

	if (RC_BAD( rc = chkVerifyAvailList( pDbInfo, &bStartOver)))
	{
		goto Exit;
	}
	if (bStartOver)
	{
		goto Begin_Check;
	}

	// Signal that we are finished.

	pDbInfo->pProgress->iCheckPhase = CHECK_FINISHED;
	pDbInfo->pProgress->bStartFlag = TRUE;
	if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
	{
		goto Exit;
	}
	pDbInfo->pProgress->bStartFlag = FALSE;

Exit:

	// Pass out any error code returned by the callback.

	if ((RC_OK( rc)) && (RC_BAD( pDbInfo->LastStatusRc)))
	{
		rc = pDbInfo->LastStatusRc;
	}

	if (pDb && pDbInfo->bDbInitialized)
	{

		// Close down the transaction, if one is going

		if( bOkToCloseTrans &&
			flmGetDbTransType( pDb) == FLM_READ_TRANS)
		{
			KrefCntrlFree( pDb);
			(void)flmAbortDbTrans( pDb);
		}
		
		fdbExit( pDb);
	}
	
	// Free memory, if allocated

	if (pBlk)
	{
		f_free( &pBlk);
	}

	// Close the database we opened.

	if( bOpenedDb)
	{
		(void)FlmDbClose( &hDb);
	}
	
	GedPoolFree( &localPool);

ExitCS:

	return( rc);
}

/***************************************************************************
Desc:	This routine opens a file and reads its dictionary into memory.
*****************************************************************************/
FSTATIC RCODE chkGetDictInfo(
	DB_INFO *		pDbInfo)
{
	RCODE		rc = FERR_OK;
	FDB *		pDb = pDbInfo->pDb;

	// Close down the transaction, if one is going.

	if (flmGetDbTransType( pDb) != FLM_UPDATE_TRANS)
	{
		if (pDb->uiTransType == FLM_READ_TRANS)
		{
			(void)flmAbortDbTrans( pDb);
		}

		// Start a read transaction on the file to ensure we are connected
		// to the file's dictionary structures.

		if (RC_BAD( rc = flmBeginDbTrans( pDb, FLM_READ_TRANS,
			0, FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
		f_memcpy( &pDbInfo->FileHdr, &pDb->pFile->FileHdr, sizeof( FILE_HDR));
		pDbInfo->pProgress->uiVersionNum = pDbInfo->FileHdr.uiVersionNum;
		pDbInfo->pProgress->uiBlockSize = pDbInfo->FileHdr.uiBlockSize;
		pDbInfo->pProgress->uiDefaultLanguage = pDbInfo->FileHdr.uiDefaultLanguage;
	}

Exit:
	return( rc);
}

/********************************************************************
Desc: This routine follows all of the blocks in a chain, verifying
		that they are properly linked.  It also verifies each block's
		header.
*********************************************************************/
FSTATIC RCODE chkVerifyBlkChain(
	DB_INFO *			pDbInfo,
	BLOCK_INFO *		pBlkInfo,
	eCorruptionLocale	eLocale,
	FLMUINT				uiFirstBlkAddr,
	FLMUINT				uiBlkType,
	FLMUINT *			puiBlkCount,
	FLMBOOL *			pbStartOverRV)
{
	RCODE					rc = FERR_OK;
	eCorruptionType	eCorruption = FLM_NO_CORRUPTION;
	SCACHE *				pSCache = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMUINT				uiPrevBlkAddress;
	FLMUINT				uiBlkCount = 0;
	STATE_INFO			StateInfo;
	FLMBOOL				bStateInitialized = FALSE;
	FLMUINT64			ui64SaveBytesExamined;
	FDB *					pDb = pDbInfo->pDb;
	FILE_HDR *			pFileHdr = &pDb->pFile->FileHdr;
	FLMUINT				uiVersionNum = pFileHdr->uiVersionNum;
	FLMUINT				uiBlockSize = pFileHdr->uiBlockSize;
	FLMUINT				uiMaxBlocks =
									(FLMUINT)(FSGetSizeInBytes(
										pDb->pFile->uiMaxFileSize,
										pDb->LogHdr.uiLogicalEOF) /
										(FLMUINT64)uiBlockSize);

	uiPrevBlkAddress = BT_END;

	/* There must be at least ONE block if it is the LFH chain. */

	if ((uiBlkType == BHT_LFH_BLK) && (uiFirstBlkAddr == BT_END))
	{
		eCorruption = FLM_BAD_LFH_LIST_PTR;
		(void)chkReportError( pDbInfo, eCorruption, eLocale,
										0, 0, 0xFF, 0, 0, 0, 0, 0xFFFF, 0, NULL);
		goto Exit;
	}

	/* Read through all of the blocks, verifying them as we go. */

Restart_Chain:
	uiBlkCount = 0;
	flmInitReadState( &StateInfo, &bStateInitialized, uiVersionNum,
							pDb, NULL,
							(FLMUINT)((uiBlkType == BHT_FREE) ? (FLMUINT)0xFF : (FLMUINT)0),
							uiBlkType, NULL);
	ui64SaveBytesExamined = pDbInfo->pProgress->ui64BytesExamined;
	StateInfo.uiBlkAddress = uiFirstBlkAddr;
	while ((StateInfo.uiBlkAddress != BT_END) && (uiBlkCount < uiMaxBlocks))
	{
		StateInfo.pBlk = NULL;
		if (RC_BAD( rc = chkBlkRead( pDbInfo,
									StateInfo.uiBlkAddress,
									StateInfo.pLogicalFile ? StateInfo.pLogicalFile->pLFile : NULL,
									&pBlk, &pSCache, &eCorruption)))
		{
			if (rc == FERR_OLD_VIEW)
			{
				FLMUINT	uiSaveDictSeq = pDb->pDict->uiDictSeq;

				if (RC_BAD( rc = chkGetDictInfo( pDbInfo)))
					goto Exit;

				/* If the dictionary ID changed, start over. */

				if (pDb->pDict->uiDictSeq != uiSaveDictSeq)
				{
					*pbStartOverRV = TRUE;
					goto Exit;
				}

				pDbInfo->pProgress->ui64BytesExamined = ui64SaveBytesExamined;
				goto Restart_Chain;
			}
			pBlkInfo->eCorruption = eCorruption;
			pBlkInfo->uiNumErrors++;
			rc = chkReportError( pDbInfo, eCorruption,
											eLocale, 0, 0, 0xFF,
											StateInfo.uiBlkAddress,
											0, 0, 0, 0xFFFF, 0, pBlk);
		}
		StateInfo.pBlk = pBlk;
		uiBlkCount++;
		pDbInfo->pProgress->ui64BytesExamined += (FLMUINT64)uiBlockSize;
		if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
		{
			goto Exit;
		}

		f_yieldCPU();

		if ((eCorruption = flmVerifyBlockHeader( &StateInfo, pBlkInfo,
										 uiBlockSize, 0,
										 (uiBlkType == BHT_FREE)
											? 0L : uiPrevBlkAddress, TRUE, TRUE)) != FLM_NO_CORRUPTION)
		{
			pBlkInfo->eCorruption = eCorruption;
			pBlkInfo->uiNumErrors++;
			chkReportError( pDbInfo, eCorruption, eLocale, 0, 0, 0xFF,
											StateInfo.uiBlkAddress,
											0, 0, 0, 0xFFFF, 0, pBlk);
			goto Exit;
		}
		uiPrevBlkAddress = StateInfo.uiBlkAddress;
		StateInfo.uiBlkAddress = (FLMUINT)FB2UD( &pBlk [BH_NEXT_BLK]);
	}
	if ((StateInfo.uiBlkAddress != BT_END) &&
		 (RC_OK( pDbInfo->LastStatusRc)))
	{
		switch (uiBlkType)
		{
			case BHT_LFH_BLK:
				eCorruption = FLM_BAD_LFH_LIST_END;
				break;
			case BHT_PCODE_BLK:
				eCorruption = FLM_BAD_PCODE_LIST_END;
				break;
			case BHT_FREE:
				eCorruption = FLM_BAD_AVAIL_LIST_END;
				break;
		}
		pBlkInfo->eCorruption = eCorruption;
		pBlkInfo->uiNumErrors++;
		chkReportError( pDbInfo, eCorruption, eLocale, 0, 0, 0xFF,
								 uiPrevBlkAddress, 0, 0, 0, 0xFFFF, 0,
								 pBlk);
		goto Exit;
	}

Exit:
	if (puiBlkCount)
	{
		*puiBlkCount = uiBlkCount;
	}
	if (bStateInitialized && StateInfo.pRecord)
	{
		StateInfo.pRecord->Release();
	}

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if( pBlk)
	{
		f_free( &pBlk);
	}

	if (RC_OK(rc) && (eCorruption != FLM_NO_CORRUPTION))
	{
		rc = (uiBlkType == BHT_FREE)
			  ? RC_SET( FERR_DATA_ERROR)
			  : RC_SET( FERR_DD_ERROR);
	}

	return( rc);
}


/***************************************************************************
Desc:	This routine verifies the LFH blocks.
*****************************************************************************/
FSTATIC RCODE chkVerifyLFHBlocks(
	DB_INFO *	pDbInfo,
	FLMBOOL *	pbStartOverRV)
{
	RCODE	rc = FERR_OK;

	pDbInfo->pProgress->uiLfNumber = 0;
	pDbInfo->pProgress->uiLfType = 0;
	pDbInfo->pProgress->iCheckPhase = CHECK_LFH_BLOCKS;
	pDbInfo->pProgress->bStartFlag = TRUE;
	if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
	{
		goto Exit;
	}
	pDbInfo->pProgress->bStartFlag = FALSE;

	f_yieldCPU();

	// Go through the LFH blocks.

	if (RC_BAD( rc = chkVerifyBlkChain( pDbInfo, &pDbInfo->pProgress->LFHBlocks,
							LOCALE_LFH_LIST,
							pDbInfo->pDb->pFile->FileHdr.uiFirstLFHBlkAddr,
							BHT_LFH_BLK, NULL, pbStartOverRV)) ||
		 *pbStartOverRV)
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:	This routine reads through the blocks in the AVAIL list and verifies
		that we don't have a loop or some other corruption in the list.
*****************************************************************************/
FSTATIC RCODE chkVerifyAvailList(
	DB_INFO *	pDbInfo,				/* Pointer to structure where statistics
									 			and other information are returned. */
	FLMBOOL *	pbStartOverRV)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiBlkCount;

	pDbInfo->pProgress->uiLfNumber = 0;
	pDbInfo->pProgress->uiLfType = 0;
	pDbInfo->pProgress->iCheckPhase = CHECK_AVAIL_BLOCKS;
	pDbInfo->pProgress->bStartFlag = TRUE;
	if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
	{
		goto Exit;
	}
	pDbInfo->pProgress->bStartFlag = FALSE;

	f_yieldCPU();
 
	if (RC_BAD( rc = chkVerifyBlkChain( pDbInfo, 
								&pDbInfo->pProgress->AvailBlocks,
								LOCALE_AVAIL_LIST,
							  pDbInfo->pDb->LogHdr.uiFirstAvailBlkAddr,
							  BHT_FREE, &uiBlkCount, pbStartOverRV)) ||
		 *pbStartOverRV)
	{
		goto Exit;
	}

	// See if the block count matches the block count stored in the
	// log header.

	if (uiBlkCount != pDbInfo->pDb->LogHdr.uiAvailBlkCount)
	{
		(void)chkReportError( pDbInfo, FLM_BAD_AVAIL_BLOCK_COUNT,
										LOCALE_AVAIL_LIST,
										0, 0, 0xFF, 0, 0, 0, 0, 0xFFFF, 0, NULL);
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

Exit:
	return( rc);
}
