//-------------------------------------------------------------------------
// Desc:	Check database b-trees for physical corruptions.
// Tabs:	3
//
//		Copyright (c) 1991-1992,1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flchktr.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE chkReadBlkFromDisk(
	FILE_HDR *			pFileHdr,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiFilePos,
	FLMUINT				uiBlkAddress,
	LFILE *				pLFile,
	FFILE *				pFile,
	FLMBYTE *			pBlk);

FSTATIC RCODE chkVerifyElmFields(
	STATE_INFO *		pStateInfo,
	DB_INFO *			pDbInfo,
	IX_CHK_INFO *		pIxChkInfo,
	POOL *				pTmpPool,
	FLMUINT *			puiErrElmRecOffsetRV,
	eCorruptionType *	peElmErrCorruptCode);

FSTATIC RCODE chkVerifySubTree(
	DB_INFO *		pDbInfo,
	IX_CHK_INFO *	pIxChkInfo,
	STATE_INFO *	ParentState,
	STATE_INFO *	pStateInfo,
	FLMUINT			uiBlkAddress,
	POOL *			pTmpPool,
	FLMBYTE *		pucResetKey,
	FLMUINT			uiResetKeyLen,
	FLMUINT			uiResetDrn);

FSTATIC RCODE chkGetLfInfo(
	DB_INFO *		pDbInfo,
	POOL *			pPool,
	LF_STATS *		pLfStats,
	LFILE *			pLFile,
	LF_STATS *		pCurrLfStats,
	FLMBOOL *		pbCurrLfLevelChangedRV);

FSTATIC RCODE chkSetupLfTable(
	DB_INFO *		pDbInfo,
	POOL *			pPool);

FSTATIC RCODE chkSetupIxInfo(
	DB_INFO *		pDbInfo,
	IX_CHK_INFO *	pIxInfoRV);

FSTATIC RCODE chkOutputIndexKeys(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	IXD *				pIxd,
	REC_KEY *		pKeyList);

/****************************************************************************
Desc:	Frees memory allocated to a result set and deletes any temporary files
		that may have been created.
****************************************************************************/
FINLINE void chkRSFree(
	void **	ppRSetRV)
{
	FResultSet *	pRSet;

	pRSet = *((FResultSet **)ppRSetRV);
	pRSet->Release();
	*ppRSetRV = (void *)0;
}

/****************************************************************************
Desc:	Adds a variable-length entry to a result set.
****************************************************************************/
FINLINE RCODE chkRSAddEntry(
	void *		pRSet,
	FLMBYTE *	pEntry,
	FLMUINT		uiEntryLength)
{
	return( ((FResultSet *)pRSet)->AddEntry( pEntry, uiEntryLength));
}

/****************************************************************************
Desc:	This routine counts the number of fields in an object table.
****************************************************************************/
FINLINE void chkCountFields(
	FDICT *		pDict,
	FLMUINT *	puiNumFieldsRV)
{
	FLMUINT		uiTblSize = pDict->uiIttCnt;
	ITT *			pItt = pDict->pIttTbl;
	FLMUINT		uiCount = 0;
	FLMUINT		uiCurrObj;

	for (uiCurrObj = 0; uiCurrObj < uiTblSize; uiCurrObj++, pItt++)
	{
		if( ITT_IS_FIELD( pItt))
		{
			uiCount++;
		}
	}
	(*puiNumFieldsRV) += uiCount;
}

/****************************************************************************
Desc:	Frees memory allocated to an IX_CHK_INFO structure
****************************************************************************/
FINLINE RCODE chkFreeIxInfo(
	IX_CHK_INFO *	pIxInfoRV)
{
	GedPoolFree( &(pIxInfoRV->pool));
	chkRSFree( &(pIxInfoRV->pRSet));
	f_free( &(pIxInfoRV->puiIxArray));
	f_memset( pIxInfoRV, 0, sizeof( IX_CHK_INFO));

	return FERR_OK;
}

/********************************************************************
Desc:
*********************************************************************/
RCODE chkBlkRead(
	DB_INFO *			pDbInfo,
	FLMUINT				uiBlkAddress,
	LFILE *				pLFile,
	FLMBYTE **			ppBlk,
	SCACHE **			ppSCache,
	eCorruptionType *	peCorruption
	)
{
	FDB *			pDb = pDbInfo->pDb;
	FILE_HDR *	pFileHdr = &pDb->pFile->FileHdr;
	RCODE			rc = FERR_OK;

	if( *ppSCache)
	{
		ScaReleaseCache( *ppSCache, FALSE);
		*ppSCache = NULL;
		*ppBlk = NULL;
	}
	else if( *ppBlk)
	{
		f_free( ppBlk);
		*ppBlk = NULL;
	}

	if( pDb->uiKilledTime)
	{
		rc = RC_SET( FERR_OLD_VIEW);
		goto Exit;
	}

	/*
	Get the block from cache.
	*/

	if( RC_OK( rc = ScaGetBlock( pDb, pLFile, 0, 
		uiBlkAddress, NULL, ppSCache)))
	{
		*ppBlk = (*ppSCache)->pucBlk;
	}
	else
	{
		/*
		Try to read the block directly from disk.
		*/

		FLMUINT		uiBlkLen = pFileHdr->uiBlockSize;
		FLMUINT		uiTransID;
		FLMBYTE *	pucBlk;
		FLMUINT		uiLastReadTransID;
		FLMUINT		uiPrevBlkAddr;
		FLMUINT		uiFilePos;

		/*
		If we didn't get a corruption error, jump to exit.
		*/

		if( !FlmErrorIsFileCorrupt( rc))
		{
			goto Exit;
		}

		/*
		Allocate memory for the block.
		*/

		if( RC_BAD( rc = f_calloc( uiBlkLen, ppBlk)))
		{
			goto Exit;
		}
		pucBlk = *ppBlk;

		uiFilePos = uiBlkAddress;
		uiTransID = pDb->LogHdr.uiCurrTransID;
		uiLastReadTransID = 0xFFFFFFFF;

		// Follow version chain until we find version we need.

		for (;;)
		{

			if( RC_BAD( rc = chkReadBlkFromDisk( pFileHdr, pDbInfo->pSFileHdl,
				uiFilePos, uiBlkAddress, pLFile, pDb->pFile, pucBlk)))
			{
				goto Exit;
			}

			/*
			See if we can use the current version of the block, or if we
			must go get a previous version.
			*/

			if( (FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]) <= uiTransID)
			{
				break;
			}

			// If the transaction ID is greater than or equal to the last
			// one we read, we have a corruption.  This test will keep us
			// from looping around forever.

			if ((FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]) >= uiLastReadTransID)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				goto Exit;
			}
			uiLastReadTransID = (FLMUINT)FB2UD( &pucBlk [BH_TRANS_ID]);

			// Block is too new, go for next older version.

			// If previous block address is same as current file position or
			// zero, we have a problem.

			uiPrevBlkAddr = (FLMUINT)FB2UD( &pucBlk [BH_PREV_BLK_ADDR]);
			if ((uiPrevBlkAddr == uiFilePos) || (!uiPrevBlkAddr))
			{
				rc = (pDb->uiKilledTime)
					  ? RC_SET( FERR_OLD_VIEW)
					  : RC_SET( FERR_DATA_ERROR);
				goto Exit;
			}
			uiFilePos = uiPrevBlkAddr;
		}

		/* See if we even got the block we thought we wanted. */

		if (GET_BH_ADDR( pucBlk) != uiBlkAddress)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			goto Exit;
		}
	}

Exit:

	*peCorruption = FLM_NO_CORRUPTION;
	if (RC_BAD( rc))
	{
		switch (rc)
		{
			case FERR_DATA_ERROR:
				*peCorruption = FLM_COULD_NOT_SYNC_BLK;
				break;
			case FERR_BLOCK_CHECKSUM:
				*peCorruption = FLM_BAD_BLK_CHECKSUM;
				break;
			default:
				break;
		}
	}
	return( rc);
}

/************************************************************************
Desc:
*************************************************************************/
FSTATIC RCODE chkReadBlkFromDisk(
	FILE_HDR *			pFileHdr,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiFilePos,
	FLMUINT				uiBlkAddress,
	LFILE *				pLFile,
	FFILE *				pFile,
	FLMBYTE *			pBlk)
{
	RCODE		   rc = FERR_OK;
	FLMUINT		uiBytesRead;
	FLMUINT		uiBlkLen = pFileHdr->uiBlockSize;

	if (RC_BAD( rc = pSFileHdl->ReadBlock( uiFilePos,
												uiBlkLen, pBlk, &uiBytesRead)))
	{
		if (rc == FERR_IO_END_OF_FILE)
		{
			rc = RC_SET( FERR_DATA_ERROR);
		}
		goto Exit;
	}

	if( uiBytesRead < uiBlkLen)
	{
		rc = RC_SET( FERR_DATA_ERROR);
		goto Exit;
	}

	/* Verify the block checksum BEFORE decrypting or using any data. */

	if( RC_BAD( rc = BlkCheckSum( pBlk, CHECKSUM_CHECK,
							uiBlkAddress, uiBlkLen)))
	{
		goto Exit;
	}

	// If this is an index block it may be encrypted, we
	// need to decrypt it before we can use it.
	// The function ScaDecryptBlock will check if the index
	// is encrypted first.  If not, it will return.
	if (pLFile && pLFile->uiLfType == LF_INDEX)
	{
		if (RC_BAD( rc = ScaDecryptBlock( pFile, pBlk)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC RCODE chkVerifyElmFields(
	STATE_INFO *			pStateInfo,
	DB_INFO *				pDbInfo,
	IX_CHK_INFO *			pIxChkInfo,
	POOL *					pTmpPool,
	FLMUINT *				puiErrElmRecOffsetRV,
	eCorruptionType *		peElmCorruptCode)
{
	FLMBYTE *	pValue = pStateInfo->pValue;
	FLMBYTE *	pData = pStateInfo->pData;
	FLMBYTE *	pTempValue;
	FlmRecord *	pRecord = pStateInfo->pRecord;
	FLMBOOL		bKRefAbortRec = FALSE;
	FLMUINT		uiSaveElmRecOffset = 0;
	void *		pDbPoolMark = NULL;
	void *		pKeyMark = NULL;
	FLMBOOL		bResetDbPool = FALSE;
	RCODE			rc = FERR_OK;
	void *		pvField = pStateInfo->pvField;
	REC_KEY *	pKeyList = NULL;
	REC_KEY *	pTmpKey;

	*peElmCorruptCode = FLM_NO_CORRUPTION;
	
	pTempValue = (pValue)
					 ? (FLMBYTE *)&pValue [pStateInfo->uiFieldProcessedLen]
					 : (FLMBYTE *)NULL;

	while ((*peElmCorruptCode == FLM_NO_CORRUPTION) &&
			 (pStateInfo->uiElmRecOffset < pStateInfo->uiElmRecLen))
	{
		uiSaveElmRecOffset = pStateInfo->uiElmRecOffset;
		if ((*peElmCorruptCode = flmVerifyElmFOP( pStateInfo)) != FLM_NO_CORRUPTION)
		{
			break;
		}

		if (!pStateInfo->bElmRecOK)
		{
			pValue = pTempValue = NULL;
			if (pRecord)
			{
				pRecord->clear();
			}
			continue;
		}
		switch (pStateInfo->uiFOPType)
		{
			case FLM_FOP_CONT_DATA:
				if ((pTempValue != NULL) && (pStateInfo->uiFOPDataLen))
				{
					f_memcpy( pTempValue,
									pStateInfo->pFOPData, pStateInfo->uiFOPDataLen);
					pTempValue += pStateInfo->uiFOPDataLen;
				}
				break;
			case FLM_FOP_STANDARD:
			case FLM_FOP_OPEN:
			case FLM_FOP_TAGGED:
			case FLM_FOP_NO_VALUE:
				if( pValue)
				{
					pValue = pTempValue = NULL;
				}
				
				if (pvField)
				{
					pvField = NULL;
				}

				if (!pRecord)
				{
					if( (pRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}

				if (RC_BAD( rc = pRecord->insertLast( pStateInfo->uiFieldLevel,
											pStateInfo->uiFieldNum,
											pStateInfo->uiFieldType, &pvField)))
				{
					goto Exit;
				}


				if( pStateInfo->uiFieldLen)
				{
					if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
																				pStateInfo->uiFieldType,
																				pStateInfo->uiFieldLen,
																				0,
																				0,
																				0,
																				&pValue,
																				NULL)))
					{
						goto Exit;
					}
					pTempValue = pValue;
				}

				if ((pTempValue) && (pStateInfo->uiFOPDataLen))
				{
					f_memcpy( pTempValue, pStateInfo->pFOPData,
										pStateInfo->uiFOPDataLen);
					pTempValue += pStateInfo->uiFOPDataLen;
				}
				break;

			case FLM_FOP_ENCRYPTED:
			{
				if( pValue)
				{
					pValue = pTempValue = NULL;
				}

				if( pData)
				{
					pData = NULL;
				}
				
				if (pvField)
				{
					pvField = NULL;
				}

				if (!pRecord)
				{
					if( (pRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}

				if (RC_BAD( rc = pRecord->insertLast( pStateInfo->uiFieldLevel,
											pStateInfo->uiFieldNum,
											pStateInfo->uiFieldType, &pvField)))
				{
					goto Exit;
				}


				if( pStateInfo->uiFieldLen)
				{
					if (RC_BAD( rc = pRecord->allocStorageSpace( pvField,
																				pStateInfo->uiFieldType,
																				pStateInfo->uiFieldLen,
																				pStateInfo->uiEncFieldLen,
																				pStateInfo->uiEncId,
																				FLD_HAVE_ENCRYPTED_DATA,
																				&pData,
																				&pValue)))
					{
						goto Exit;
					}
					pTempValue = pValue;
				}

				if ((pTempValue) && (pStateInfo->uiFOPDataLen))
				{
					f_memcpy( pTempValue, pStateInfo->pFOPData,
										pStateInfo->uiFOPDataLen);
					pTempValue += pStateInfo->uiFOPDataLen;
				}
				break;
			}

			case FLM_FOP_JUMP_LEVEL:
				break;
		}

		if ((!pStateInfo->uiEncId && pStateInfo->uiFieldProcessedLen == pStateInfo->uiFieldLen) ||
			 (pStateInfo->uiEncId && pStateInfo->uiFieldProcessedLen == pStateInfo->uiEncFieldLen))
		{

			/*
			The whole field has been retrieved.  Verify the field and
			graft it into the record being built.
			*/

			if (pValue && (pDbInfo->uiFlags & FLM_CHK_FIELDS))
			{
				if (pStateInfo->uiFieldType == 0xFF)
				{
					// Hit Rec Info object - don't care what's in it - must not
					// assert, because this would kill our ability to check
					// older versions of the database which have REC_INFO
					// data in them.
					
					*peElmCorruptCode = FLM_NO_CORRUPTION;
				}
				else
				{
					if (!pStateInfo->uiEncId)
					{
						*peElmCorruptCode = flmVerifyField( pValue,
							pStateInfo->uiFieldLen, pStateInfo->uiFieldType);
					}
					else
					{
						// Decrypt the field and store the decrypted data.
						
						if (!pStateInfo->pDb->pFile->bInLimitedMode)
						{
							if (RC_BAD( rc = flmDecryptField( pStateInfo->pDb->pDict,
								pRecord, pvField, pStateInfo->uiEncId, pTmpPool)))
							{
								goto Exit;
							}
							*peElmCorruptCode = flmVerifyField( pData,
								pStateInfo->uiFieldLen, pStateInfo->uiFieldType);
						}
						else
						{
							// If we can't decrypt the field, then just pass
							// it for now.
							
							*peElmCorruptCode = FLM_NO_CORRUPTION;
						}
					}
				}
			}
			else
			{
				*peElmCorruptCode = FLM_NO_CORRUPTION;
			}

			pValue = pTempValue = NULL;
		}

		// If this is the last element of the record, verify the
		// record's keys.

		if( BBE_IS_LAST( pStateInfo->pElm) &&
			(pStateInfo->uiElmRecOffset == pStateInfo->uiElmRecLen))
		{
			pValue = pTempValue = NULL;

			if( !pDbInfo->pProgress->bPhysicalCorrupt && (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING))
			{
				FLMUINT				uiLoop;
				IXD *					pIxd;

				if( pStateInfo->pLogicalFile->pLFile->uiLfType == LF_CONTAINER)
				{
					// Need to mark the DB's temporary pool.  The index code
					// allocates memory for new CDL entries from the DB pool.  If
					// the pool is not reset, it grows during the check and
					// becomes VERY large.

					pDbPoolMark = GedPoolMark( &(pDbInfo->pDb->TempPool));
					bResetDbPool = TRUE;

					/*
					Set up the KRef table so that flmGetRecKeys
					will work correctly.
					*/

					if( RC_BAD( rc = KrefCntrlCheck( pStateInfo->pDb)))
					{
						goto Exit;
					}
					bKRefAbortRec = TRUE;

					for( uiLoop = 0; uiLoop < pIxChkInfo->uiIxCount; uiLoop++)
					{
						if( RC_BAD( rc = fdictGetIndex(
							pStateInfo->pDb->pDict,
							pStateInfo->pDb->pFile->bInLimitedMode,
							pIxChkInfo->puiIxArray[ uiLoop], NULL, &pIxd, TRUE)))
						{
							goto Exit;
						}

						if( pIxd->uiFlags & IXD_OFFLINE)
						{
							continue;
						}

						if( pIxd->uiContainerNum ==
							pStateInfo->pLogicalFile->pLFile->uiLfNum ||
							!pIxd->uiContainerNum)
						{
							/*
							Mark the field pool so that it can be reset
							after the record keys have been generated
							and output.
							*/

							pKeyMark = GedPoolMark( pTmpPool);

							/*
							Build the record keys for the current index.
							Do not remove duplicate keys.  The result set
							will remove any duplicates.
							*/

							if (RC_BAD( rc = flmGetRecKeys(
								pStateInfo->pDb, pIxd, pRecord,
								pStateInfo->pLogicalFile->pLFile->uiLfNum,
								FALSE, pTmpPool, &pKeyList)))
							{
								goto Exit;
							}

							/*
							If the record generated keys for the current
							index, output the keys to the result set.
							*/

							if( pKeyList)
							{
								if( RC_BAD( rc = chkOutputIndexKeys(
									pStateInfo, pIxChkInfo,	pIxd, pKeyList)))
								{
									goto Exit;
								}

								pTmpKey = pKeyList;
								while( pTmpKey)
								{
									pTmpKey->pKey->Release();
									pTmpKey->pKey = NULL;
									pTmpKey = pTmpKey->pNextKey;
								}
								pKeyList = NULL;
							}

							/* Reset the field pool */

							GedPoolReset( pTmpPool, pKeyMark);
						}
					}

					/*
					Clean up any keys that may have been added to the
					KRef table.
					*/

					KYAbortCurrentRecord( pStateInfo->pDb);
					bKRefAbortRec = FALSE;

					/* Reset the DB's temporary pool */

					(void)GedPoolReset( &(pDbInfo->pDb->TempPool), pDbPoolMark);
					bResetDbPool = FALSE;
				}
			}

			if (pRecord)
			{
				pRecord->clear();
			}
			pValue = pTempValue = NULL;
			GedPoolReset( pTmpPool, NULL);
		}

		if( *peElmCorruptCode != FLM_NO_CORRUPTION)
		{
			pStateInfo->bElmRecOK = FALSE;
		}
	}

Exit:

	/*
	Clean up any keys that may have been added to the
	KRef table.  This is a fail-safe case to clean up the
	KRef in case KYKeysCommit didn't get called above.
	*/

	if (bKRefAbortRec)
	{
		KYAbortCurrentRecord( pStateInfo->pDb);
	}

	/*
	Free any keys in the key list
	*/

	if( pKeyList)
	{
		pTmpKey = pKeyList;
		while( pTmpKey)
		{
			pTmpKey->pKey->Release();
			pTmpKey->pKey = NULL;
			pTmpKey = pTmpKey->pNextKey;
		}
	}

	/* Reset the DB's temporary pool */

	if( bResetDbPool)
	{
		(void)GedPoolReset( &(pDbInfo->pDb->TempPool), pDbPoolMark);
	}

	if( *peElmCorruptCode != FLM_NO_CORRUPTION || RC_BAD( rc))
	{
		pValue = pTempValue = NULL;
		GedPoolReset( pTmpPool, NULL);
		if (pRecord)
		{
			pRecord->clear();
		}
	}

	pStateInfo->pValue = pValue;
	pStateInfo->pData = pData;
	pStateInfo->pvField = pvField;
	pStateInfo->pRecord = pRecord;

	if( *peElmCorruptCode != FLM_NO_CORRUPTION)
	{
		*puiErrElmRecOffsetRV = uiSaveElmRecOffset;
	}

	return( rc);
}

/***************************************************************************
Desc:	This routine checks all of the blocks/links in a sub-tree of a
		B-TREE.  It calls itself recursively whenever it descends a level
		in the tree.
*****************************************************************************/
FSTATIC RCODE chkVerifySubTree(
	DB_INFO *	  	pDbInfo,
	IX_CHK_INFO *	pIxChkInfo,
	STATE_INFO *	pParentState,
	STATE_INFO *	pStateInfo,
	FLMUINT			uiBlkAddress,
	POOL *			pTmpPool,
	FLMBYTE *		pucResetKey,
	FLMUINT			uiResetKeyLen,
	FLMUINT			uiResetDrn)
{
	RCODE			  		rc = FERR_OK;
	SCACHE *				pSCache = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMUINT				uiLevel = pStateInfo->uiLevel;
	FLMUINT				uiBlkType = pStateInfo->uiBlkType;
	FLMUINT				uiLfType = pStateInfo->pLogicalFile->pLFile->uiLfType;
	FLMUINT				uiBlockSize =
								pDbInfo->pDb->pFile->FileHdr.uiBlockSize;
	FLMUINT				uiParentBlkAddress;
	FLMBYTE *			pChildBlkAddr;
	FLMUINT				uiChildBlkAddress;
	FLMUINT				uiPrevNextBlkAddress;
	eCorruptionType	eElmCorruptCode;
	eCorruptionType  	eBlkCorruptionCode = FLM_NO_CORRUPTION;
	eCorruptionType	eLastCorruptCode = FLM_NO_CORRUPTION;
	FLMUINT				uiNumErrors = 0;
	FLMUINT				uiErrElmRecOffset = 0;
	FLMUINT64		  	ui64SaveKeyCount = 0;
	FLMUINT64			ui64SaveKeyRefs = 0;
	BLOCK_INFO			SaveBlkInfo;
	BLOCK_INFO *		pBlkInfo;
	FLMBOOL			  	bProcessElm;
	FLMBOOL			  	bCountElm;
	FLMBOOL				bDescendToChildBlocks;
	FLMINT			  	iCompareStatus;
	eCorruptionType	eHdrCorruptCode;

	/* Setup the state information. */

	pStateInfo->pBlk = NULL;
	pStateInfo->uiBlkAddress = uiBlkAddress;
	uiPrevNextBlkAddress = pStateInfo->uiNextBlkAddr;
	uiParentBlkAddress = (pParentState)
								? pParentState->uiBlkAddress
								: BT_END;

	f_yieldCPU();

	/* Read the sub-tree root block into memory. */

	bDescendToChildBlocks = TRUE;
	if (RC_BAD( rc = chkBlkRead( pDbInfo,
								uiBlkAddress,
								pStateInfo->pLogicalFile ? pStateInfo->pLogicalFile->pLFile : NULL,
								&pBlk, &pSCache,&eBlkCorruptionCode)))
	{
		if (eBlkCorruptionCode != FLM_NO_CORRUPTION)
		{
			uiNumErrors++;
			eLastCorruptCode = eBlkCorruptionCode;
			chkReportError( pDbInfo, eBlkCorruptionCode, LOCALE_B_TREE,
								 pDbInfo->pProgress->uiLfNumber,
								 pDbInfo->pProgress->uiLfType,
								 uiLevel, uiBlkAddress,
								 uiParentBlkAddress, 0, 0, 0xFFFF, 0, pBlk);
			if (eBlkCorruptionCode == FLM_BAD_BLK_CHECKSUM)
			{
				bDescendToChildBlocks = FALSE;

				/*
				Allow to continue the check, but if this is a non-leaf block
				a non-zero eBlkCorruptionCode will prevent us from descending to
				child blocks.  Set rc to SUCCESS so we won't goto Exit below.
				*/

				rc = FERR_OK;
			}
			else if (eBlkCorruptionCode == FLM_COULD_NOT_SYNC_BLK)
			{
				eLastCorruptCode = eBlkCorruptionCode;

				/*
				Need the goto here, because rc is changed to SUCCESS,
				and the goto below would get skipped.
				*/

				rc = FERR_OK;
				goto fix_state;
			}
		}
		else if (rc == FERR_OLD_VIEW)
		{
			pDbInfo->bReposition = TRUE;
		}

		/* If rc was not changed to SUCCESS above, goto Exit. */

		if (RC_BAD( rc))
			goto Exit;
	}
	pStateInfo->pBlk = pBlk;

	/* Verify the block header */

	/* Don't recount the block if we are resetting. */

	if (!uiResetKeyLen)
	{
		pDbInfo->pProgress->ui64BytesExamined += (FLMUINT64)uiBlockSize;
		pBlkInfo = &pStateInfo->BlkInfo;
	}
	else
	{
		pBlkInfo = NULL;
	}
	chkCallProgFunc( pDbInfo);

	/* Check the block header. */

	if ((eHdrCorruptCode =
				flmVerifyBlockHeader( pStateInfo, pBlkInfo, uiBlockSize,
									(pParentState == NULL)
									? BT_END
									: 0,
									(pParentState == NULL)
									? BT_END
									: pParentState->uiLastChildAddr,
									TRUE, TRUE)) == FLM_NO_CORRUPTION)
	{

		/*
		Verify the previous block's next block address -- it should
		equal the current block's address.
		*/

		if ((uiPrevNextBlkAddress) &&
				(uiPrevNextBlkAddress != uiBlkAddress))
		{
			eHdrCorruptCode = FLM_BAD_PREV_BLK_NEXT;
		}
	}
	if (eHdrCorruptCode != FLM_NO_CORRUPTION)
	{
		eLastCorruptCode = eHdrCorruptCode;
		uiNumErrors++;
		chkReportError( pDbInfo, eHdrCorruptCode, LOCALE_B_TREE,
							 pDbInfo->pProgress->uiLfNumber,
							 pDbInfo->pProgress->uiLfType,
							uiLevel, uiBlkAddress,
							uiParentBlkAddress, 0, 0, 0xFFFF, 0,
							pBlk);
	}

	/* Go through the elements in the block. */

	pStateInfo->uiElmOffset = BH_OVHD;
	while ((pStateInfo->uiElmOffset < pStateInfo->uiEndOfBlock) &&
			 (RC_OK( pDbInfo->LastStatusRc)))
	{

		/*
		If we are resetting, save any statistical information so we
		can back it out if we need to.
		*/

		if (uiResetKeyLen)
		{
			ui64SaveKeyCount = pStateInfo->ui64KeyCount;
			ui64SaveKeyRefs = pStateInfo->ui64KeyRefs;
			f_memcpy( &SaveBlkInfo, &pStateInfo->BlkInfo, sizeof( BLOCK_INFO));
			bCountElm = FALSE;
			bProcessElm = FALSE;
		}
		else
		{
			bCountElm = TRUE;
			bProcessElm = TRUE;
		}

		pStateInfo->BlkInfo.ui64ElementCount++;

		if ((eElmCorruptCode = flmVerifyElement( pStateInfo, pDbInfo->uiFlags)) != FLM_NO_CORRUPTION)
		{

			/* Report any errors in the element. */

			eLastCorruptCode = eElmCorruptCode;
			uiNumErrors++;
			if (RC_BAD( rc = chkReportError( pDbInfo, eElmCorruptCode,
									LOCALE_B_TREE,
									pDbInfo->pProgress->uiLfNumber,
									pDbInfo->pProgress->uiLfType,
									uiLevel, uiBlkAddress,
									 uiParentBlkAddress,
									 pStateInfo->uiElmOffset, pStateInfo->uiElmDrn,
									 0xFFFF, 0, pBlk)))
			{
				break;
			}
		}

		/* Keep track of the number of continuation elements. */

		if ((uiBlkType == BHT_LEAF) &&
			 (!BBE_IS_FIRST( pStateInfo->pElm)) &&
			 (pStateInfo->uiElmLen != BBE_LEM_LEN))
		{
			pStateInfo->BlkInfo.ui64ContElementCount++;
			pStateInfo->BlkInfo.ui64ContElmBytes += pStateInfo->uiElmLen;
		}

		/* Do some further checking. */

		if (eElmCorruptCode != FLM_NO_CORRUPTION)
		{
			pStateInfo->bElmRecOK = FALSE;
		}
		else
		{
			/* See if we are resetting */

			iCompareStatus = 0;
			if ((uiResetKeyLen) &&
				 (pStateInfo->bValidKey) &&
				 ((!pStateInfo->uiCurKeyLen) ||
				 ((iCompareStatus = flmCompareKeys( pStateInfo->pCurKey,
													 pStateInfo->uiCurKeyLen,
													 pucResetKey, uiResetKeyLen)) >= 0)))
			{
				if (iCompareStatus > 0)
				{
					if (uiBlkType == BHT_LEAF)
					{
						uiResetKeyLen = 0;
						pucResetKey = NULL;
						uiResetDrn = 0;
						bCountElm = TRUE;
					}
					bProcessElm = TRUE;
				}
				else if( uiLfType == LF_INDEX)
				{
					FLMBYTE *	pTmpElm = pStateInfo->pElm;
					FLMUINT		uiLowestDrn = FSGetDomain( &pTmpElm, 
															pStateInfo->uiElmOvhd);

					if( uiResetDrn >= uiLowestDrn)
					{
						bProcessElm = TRUE;
						bCountElm = TRUE;
					}
				}
				else
				{
					/* Processing a container */
					bProcessElm = TRUE;
				}
			}

			if (uiBlkType == BHT_LEAF)
			{
				/* No need to parse LEM element. */

				if ((pStateInfo->uiCurKeyLen != 0) && (pStateInfo->bValidKey))
				{
					if (uiLfType == LF_CONTAINER)
					{
						if (pStateInfo->uiElmDrn != DRN_LAST_MARKER)
						{
							if( RC_BAD( rc = chkVerifyElmFields( pStateInfo,
								pDbInfo, pIxChkInfo, pTmpPool,
								&uiErrElmRecOffset, &eElmCorruptCode)))
							{
								goto Exit;
							}
						}
					}
					else if( bProcessElm)
					{
						uiErrElmRecOffset = 0xFFFF;
						if( !pDbInfo->pProgress->bPhysicalCorrupt && (pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING))
						{
							if(( RC_BAD( rc = flmVerifyIXRefs( pStateInfo, pIxChkInfo,
														uiResetDrn,	&eElmCorruptCode))) ||
								(pDbInfo->bReposition))
							{
								goto Exit;
							}
						}
						else
						{
							if(( RC_BAD( rc = flmVerifyIXRefs( pStateInfo, NULL,
														uiResetDrn, &eElmCorruptCode))) ||
								(pDbInfo->bReposition))
							{
								goto Exit;
							}
						}
					}
				}

				if( bProcessElm)
				{
					uiResetKeyLen = 0;
					pucResetKey = NULL;
					uiResetDrn = 0;

					if (eElmCorruptCode != FLM_NO_CORRUPTION)
					{

						/* Report any errors in the element. */

						eLastCorruptCode = eElmCorruptCode;
						uiNumErrors++;
						chkReportError( pDbInfo, eElmCorruptCode,
											LOCALE_B_TREE,
											pDbInfo->pProgress->uiLfNumber,
											pDbInfo->pProgress->uiLfType,
											uiLevel, uiBlkAddress,
											 	 uiParentBlkAddress,
												 pStateInfo->uiElmOffset,
											 	 pStateInfo->uiElmDrn,
												 uiErrElmRecOffset,
											 	 pStateInfo->uiFieldNum,
												 pBlk);
				
						if (RC_BAD( pDbInfo->LastStatusRc))
						{
							break;
						}
					}
				}
			}
			else
			{
				if (uiBlkType == BHT_NON_LEAF_DATA)
				{
					pChildBlkAddr = &pStateInfo->pElm [BNE_DATA_CHILD_BLOCK];
				}
				else
				{
					pChildBlkAddr = &pStateInfo->pElm [BNE_CHILD_BLOCK];
				}
				uiChildBlkAddress = (FLMUINT)FB2UD( pChildBlkAddr );

				/*
				Check the child sub-tree -- NOTE, this is a recursive call.
				First see if we have a pucResetKey that we want to position
				to.  If so, make sure we are positioned to it before
				descending to the child block.
				*/

				if (bProcessElm)
				{
					if (!bDescendToChildBlocks)
					{
						rc = FERR_OK;
					}
					else
					{
						rc = chkVerifySubTree( pDbInfo, pIxChkInfo, pStateInfo,
								(pStateInfo - 1), uiChildBlkAddress, pTmpPool,
								pucResetKey, uiResetKeyLen, uiResetDrn);
					}

					if ((RC_BAD( rc)) ||
						 (RC_BAD( pDbInfo->LastStatusRc)) ||
						 (pDbInfo->bReposition))
					{
						goto Exit;
					}

					/*
					Once we reach the key, set it to an empty to key so that
					we will always descend to the child block after this point.
					*/

					uiResetKeyLen = 0;
					pucResetKey = NULL;
					uiResetDrn = 0;
				}

				/* Save the child block address in the level information. */

				pStateInfo->uiLastChildAddr = uiChildBlkAddress;
			}
		}

		/*
		If we were resetting on this element, restore the statistics to what
		they were before.
		*/

		if (!bCountElm)
		{
			pStateInfo->ui64KeyCount = ui64SaveKeyCount;
			pStateInfo->ui64KeyRefs = ui64SaveKeyRefs;
			f_memcpy( &pStateInfo->BlkInfo, &SaveBlkInfo, sizeof( BLOCK_INFO));
		}

		/* Go to the next element. */

		pStateInfo->uiElmOffset += pStateInfo->uiElmLen;
	}

	/* Verify that we ended exactly on the end of the block. */

	if ((eLastCorruptCode == FLM_NO_CORRUPTION) &&
		 (pStateInfo->uiEndOfBlock >= BH_OVHD) &&
		 (pStateInfo->uiEndOfBlock <= uiBlockSize) &&
		 (pStateInfo->uiElmOffset > pStateInfo->uiEndOfBlock))
	{
		eLastCorruptCode = FLM_BAD_ELM_END;
		uiNumErrors++;
		chkReportError( pDbInfo, eLastCorruptCode, LOCALE_B_TREE,
								 pDbInfo->pProgress->uiLfNumber,
								 pDbInfo->pProgress->uiLfType,
								uiLevel, uiBlkAddress,
								 uiParentBlkAddress,
								 pStateInfo->uiElmOffset, 0, 0xFFFF, 0,
								 pBlk);
	}

	/* Verify that the last key in the block matches the parent's key. */

	if ((eLastCorruptCode == FLM_NO_CORRUPTION) && (pParentState) &&
			(RC_OK( pDbInfo->LastStatusRc)))
	{
		if ((pStateInfo->bValidKey) && (pParentState->bValidKey) &&
			 (flmCompareKeys( pStateInfo->pCurKey, pStateInfo->uiCurKeyLen,
			 						pParentState->pCurKey,
									pParentState->uiCurKeyLen) != 0))
		{
			eLastCorruptCode = FLM_BAD_PARENT_KEY;
			uiNumErrors++;
			chkReportError( pDbInfo, eLastCorruptCode, LOCALE_B_TREE,
								 pDbInfo->pProgress->uiLfNumber,
								 pDbInfo->pProgress->uiLfType,
								 uiLevel, uiBlkAddress,
									 uiParentBlkAddress,
									 0, 0, 0xFFFF, 0, pBlk);
		}
	}

fix_state:

	/*
	If the block could not be verified, set the level's next block
	address and last child address to zero to indicate that we really
	aren't sure we're at the right place in this level in the B-TREE.
	*/

	if (eLastCorruptCode != FLM_NO_CORRUPTION)
	{
		pStateInfo->BlkInfo.eCorruption = eLastCorruptCode;
		pStateInfo->BlkInfo.uiNumErrors += uiNumErrors;

		/*
		Reset all child block states.
		*/

		for( ;;)
		{
			pStateInfo->uiNextBlkAddr = 0;
			pStateInfo->uiLastChildAddr = 0;
			pStateInfo->bValidKey = FALSE;
			pStateInfo->uiElmLastFlag = 0xFF;

			/*
			Quit when the leaf level has been reached.
			*/
			
			if( pStateInfo->uiLevel == 0)
			{
				break;
			}
			pStateInfo--;
		}
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if( pBlk)
	{
		f_free( &pBlk);
	}

	pStateInfo->pBlk = NULL;

	return( rc);
}

/***************************************************************************
Desc:	This routine reads the LFH areas from disk to make sure they are up
		to date in memory.
*****************************************************************************/
FSTATIC RCODE chkGetLfInfo(
	DB_INFO *		pDbInfo,
	POOL *			pPool,
	LF_STATS *		pLfStats,
	LFILE *			pLFile,
	LF_STATS *		pCurrLfStats,
	FLMBOOL *		pbCurrLfLevelChangedRV)
{
	SCACHE *				pSCache = NULL;
	FLMBYTE *			pBlk = NULL;
	FLMUINT				uiSaveLevel;
	eCorruptionType	eBlkCorruptionCode;
	RCODE					rc = FERR_OK;

	/* Read in the block containing the logical file header. */

	if (RC_BAD( rc = chkBlkRead( pDbInfo,
		pLFile->uiBlkAddress, pLFile, &pBlk, &pSCache,
		&eBlkCorruptionCode)))
	{

		/* Log the error. */

		if (eBlkCorruptionCode != FLM_NO_CORRUPTION)
		{

			//Bug #22003
			chkReportError( pDbInfo, eBlkCorruptionCode, LOCALE_LFH_LIST,
								 0, 0, 0xFF,
								pLFile->uiBlkAddress, 0, 0, 0,
								0xFFFF, 0, pBlk);
		}
		goto Exit;
	}

	/* Copy the LFH from the block to the LFILE. */

	uiSaveLevel = pLfStats->uiNumLevels;
	if (RC_BAD( rc = flmBufferToLFile( 
							&pBlk[ pLFile->uiOffsetInBlk], pLFile,
							pLFile->uiBlkAddress,
							pLFile->uiOffsetInBlk)))
	{
		goto Exit;
	}

	/*
	Read root block to get the number of levels in the B-TREE
	*/

	if (pLFile->uiRootBlk == BT_END)
	{
		pLfStats->uiNumLevels = 0;
	}
	else
	{
		if (RC_BAD( rc = chkBlkRead( pDbInfo,
								pLFile->uiRootBlk, pLFile, &pBlk, &pSCache,
								&eBlkCorruptionCode)))
		{
			if (eBlkCorruptionCode != FLM_NO_CORRUPTION)
			{
				//Bug #22003
				chkReportError( pDbInfo, eBlkCorruptionCode, LOCALE_B_TREE,
										pLFile->uiLfNum, pLFile->uiLfType, 0xFF,
										pLFile->uiRootBlk, 0, 0, 0,
										0xFFFF, 0, pBlk);
			}
			goto Exit;
		}
		pLfStats->uiNumLevels = (FLMUINT)(pBlk [BH_LEVEL]) + 1;

		/*
		GW Bug 55264: Need to make sure that the level extracted from
		the block is valid.
		*/

		if( pBlk [BH_LEVEL] >= BH_MAX_LEVELS)
		{
			chkReportError( pDbInfo, FLM_BAD_BLK_HDR_LEVEL, LOCALE_B_TREE,
									pLFile->uiLfNum, pLFile->uiLfType,
									(FLMUINT)(pBlk [BH_LEVEL]),
									pLFile->uiRootBlk, 0, 0, 0,
									0xFFFF, 0, pBlk);
			/*
			Force pLfStats->uiNumLevels to 1 so that we don't crash
			*/

			pLfStats->uiNumLevels = 1;
		}
	}

	/*
	If the number of levels changed, reset the level information
	on this logical file.
	*/

	if (uiSaveLevel != pLfStats->uiNumLevels && pLfStats->uiNumLevels)
	{
		if (pLfStats->uiNumLevels > uiSaveLevel)
		{
			if ((pLfStats->pLevelInfo =
					(LEVEL_INFO *)GedPoolCalloc( pPool,
						(FLMUINT)(sizeof( LEVEL_INFO) * pLfStats->uiNumLevels))) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}

		if (pCurrLfStats == pLfStats)
		{
			*pbCurrLfLevelChangedRV = TRUE;
		}
	}

Exit:

	if( pSCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}
	else if( pBlk)
	{
		f_free( &pBlk);
	}

	return( rc);
}


/***************************************************************************
Desc:	This routine allocates and initializes the LF table (array of
		LF_HDR structures).
*****************************************************************************/
FSTATIC RCODE chkSetupLfTable(
	DB_INFO *	pDbInfo,
	POOL *		pPool)
{
	FLMUINT		uiCnt;
	FLMUINT		uiNumIndexes = 0;
	FLMUINT		uiIxStart;
	FLMUINT		uiNumDataCont = 0;
	FLMUINT		uiNumDictCont = 0;
	FLMUINT		uiIxOffset;
	FLMUINT		uiDataOffset;
	FLMUINT		uiDictOffset;
	FDB *			pDb = pDbInfo->pDb;
	LFILE *		pLFile;
	LFILE *		pTmpLFile;
	LF_HDR *		pLogicalFile;
	LF_STATS *	pLfStats;
	RCODE			rc = FERR_OK;

	/*
	Set up the table such that the dictionary is checked first,
	followed by data containers, and then indexes.  This is
	necessary for the logical (index) check to work.  The
	data records must be extracted before the indexes are
	checked so that the temporary result set, used during
	the logical check, can be built.
	*/
	
	pDbInfo->pProgress->uiNumFields =
	pDbInfo->pProgress->uiNumIndexes =
	pDbInfo->pProgress->uiNumContainers = 0;
	pDbInfo->pProgress->uiNumLogicalFiles =
		(FLMUINT)((pDb->pDict)
					 ? (FLMUINT)pDb->pDict->uiLFileCnt
					 : (FLMUINT)0);

	/* Determine the number of fields. */

	if (pDb->pDict)
	{
		chkCountFields( pDb->pDict, &pDbInfo->pProgress->uiNumFields);

		for (uiCnt = 0, pLFile = (LFILE *)pDb->pDict->pLFileTbl;
			  uiCnt < pDb->pDict->uiLFileCnt;
			  uiCnt++, pLFile++)
		{
			if (pLFile->uiLfType == LF_INDEX)
			{
				pDbInfo->pProgress->uiNumIndexes++;
				uiNumIndexes++;
			}
			else
			{
				pDbInfo->pProgress->uiNumContainers++;
				if( pLFile->uiLfNum == FLM_DICT_CONTAINER)
				{
					uiNumDictCont++;
				}
				else
				{
					uiNumDataCont++;
				}
			}
		}
	}

	/* Allocate memory for each LFILE, then set up each LFILE. */

	if (!pDbInfo->pProgress->uiNumLogicalFiles)
	{
		pDbInfo->pLogicalFiles = NULL;
		pDbInfo->pProgress->pLfStats = NULL;
	}
	else
	{
		if ((pDbInfo->pLogicalFiles =
									(LF_HDR *)GedPoolCalloc( pPool,
										(FLMUINT)(sizeof( LF_HDR) *
										pDbInfo->pProgress->uiNumLogicalFiles))) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		if ((pDbInfo->pProgress->pLfStats =
									(LF_STATS *)GedPoolCalloc( pPool,
										(FLMUINT)(sizeof( LF_STATS) *
										pDbInfo->pProgress->uiNumLogicalFiles))) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		uiDictOffset = 0;
		uiDataOffset = uiNumDictCont;
		uiIxOffset = uiDataOffset + uiNumDataCont;
		uiIxStart = uiIxOffset;
		
		for (uiCnt = 0, pTmpLFile = (LFILE *)pDb->pDict->pLFileTbl;
			  uiCnt < pDbInfo->pProgress->uiNumLogicalFiles;
			  uiCnt++, pTmpLFile++)
		{
			if( pTmpLFile->uiLfType == LF_INDEX)
			{
				FLMUINT	uiTmpIxOffset = uiIxOffset;

				// Indexes need to be in order from lowest to highest
				// because the result set is sorted that way.

				while (uiTmpIxOffset > uiIxStart)
				{
					if (pDbInfo->pLogicalFiles [uiTmpIxOffset - 1].pLFile->uiLfNum <
								pTmpLFile->uiLfNum)
					{
						break;
					}
					f_memcpy( &pDbInfo->pLogicalFiles [uiTmpIxOffset],
								 &pDbInfo->pLogicalFiles [uiTmpIxOffset - 1],
								 sizeof( LF_HDR));
					f_memcpy( &pDbInfo->pProgress->pLfStats [uiTmpIxOffset],
								 &pDbInfo->pProgress->pLfStats [uiTmpIxOffset - 1],
								 sizeof( LF_STATS));
					uiTmpIxOffset--;
				}

				pLogicalFile = &(pDbInfo->pLogicalFiles[ uiTmpIxOffset]);
				pLfStats = &(pDbInfo->pProgress->pLfStats[ uiTmpIxOffset]);
				uiIxOffset++;
			}
			else
			{
				if( pTmpLFile->uiLfNum == FLM_DICT_CONTAINER)
				{
					pLogicalFile = &(pDbInfo->pLogicalFiles[ uiDictOffset]);
					pLfStats = &(pDbInfo->pProgress->pLfStats[ uiDictOffset]);
					uiDictOffset++;
				}
				else
				{
					pLogicalFile = &(pDbInfo->pLogicalFiles[ uiDataOffset]);
					pLfStats = &(pDbInfo->pProgress->pLfStats[ uiDataOffset]);
					uiDataOffset++;
				}
			}
			pLogicalFile->pLfStats = pLfStats;

			/*
			Copy the LFILE information - so we can return the
			information even after the database has been closed.
			*/

			if ((pLogicalFile->pLFile = pLFile =
										(LFILE *)GedPoolAlloc( pPool,
											(FLMUINT)sizeof( LFILE))) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			/*
			Copy the LFILE structure so we can get enough information
			to read them from disk, then read them from disk so we have
			a read-consistent view of them.
			*/

			f_memcpy( pLFile, pTmpLFile, sizeof( LFILE));
			if (RC_BAD( rc = flmLFileRead( pDb, pLFile)))
			{
				goto Exit;
			}
			pLfStats->uiLfType = pLFile->uiLfType;
			if (pLFile->uiLfType == LF_INDEX)
			{
				pLfStats->uiIndexNum = pLFile->uiLfNum;
				pLfStats->uiContainerNum = 0;
			}
			else
			{
				pLfStats->uiIndexNum = 0;
				pLfStats->uiContainerNum = pLFile->uiLfNum;
			}

			/*
			If the logical file is an index, get pointers to the index
			definition and its field definitions.
			*/

			if (pLFile->uiLfType == LF_INDEX)
			{
				IXD *	pTmpIxd;
				IFD_p	pTmpIfd;

				if (RC_BAD( rc = fdictGetIndex(
							pDb->pDict,
							pDb->pFile->bInLimitedMode,
							pLFile->uiLfNum,
							NULL, &pTmpIxd, TRUE)))
				{
					if (rc == FERR_BAD_IX)
					{
						chkReportError( pDbInfo, FLM_BAD_PCODE_IXD_TBL,
											 LOCALE_IXD_TBL,
											 pLFile->uiLfNum, pLFile->uiLfType,
											 0xFF, 0,
											 0, 0, 0, 0xFFFF, 0,
											 NULL);
						rc = RC_SET( FERR_PCODE_ERROR);
					}

					goto Exit;
				}

				pTmpIfd = pTmpIxd->pFirstIfd;

				/*
				Copy the IXD and IFD information - so we can return the
				information even after the database has been closed.
				*/

				if ((pLogicalFile->pIxd = (IXD *)GedPoolAlloc( pPool,
								(FLMUINT)(sizeof( IXD) +
								sizeof( IFD) * pTmpIxd->uiNumFlds))) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
				pLogicalFile->pIfd = (IFD_p)(&pLogicalFile->pIxd [1]);
				f_memcpy( pLogicalFile->pIxd, pTmpIxd, sizeof( IXD));
				f_memcpy( pLogicalFile->pIfd, pTmpIfd,
								sizeof( IFD) * pTmpIxd->uiNumFlds);
				pLfStats->uiContainerNum = pLogicalFile->pIxd->uiContainerNum;
			}

			/*
			Get the current number of levels in the logical file and
			allocate an array of LEVEL_INFO structures for the levels.
			*/

			pLfStats->uiNumLevels = 0;
			if (RC_BAD( rc = chkGetLfInfo( pDbInfo, pPool, pLfStats, pLFile,
													 NULL, NULL)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}


/***************************************************************************
Desc:	This routine checks all of the B-TREES in the database -- all
		indexes and containers.
*****************************************************************************/
RCODE chkVerifyBTrees(
	DB_INFO *	pDbInfo,
	POOL *		pPool,
	FLMBOOL *	pbStartOverRV)
{
	RCODE		  					rc = FERR_OK;
	FDB *							pDb = pDbInfo->pDb;
	FLMUINT						uiCurrLf;
	FLMUINT						uiCurrLevel;
	FLMBYTE *					pKeyBuffer = NULL;
	FLMUINT						uiKeysAllocated = 0;
	STATE_INFO					State [BH_MAX_LEVELS];
	FLMBOOL						bStateInitialized [BH_MAX_LEVELS];
	FLMBYTE	  					ucResetKeyBuff [MAX_KEY_SIZ];
	FLMBYTE *					pucResetKey = NULL;
	FLMUINT						uiResetKeyLen = 0;
	FLMUINT						uiResetDrn = 0;
	LF_HDR *						pLogicalFile;
	LF_STATS *					pLfStats;
	LFILE *						pLFile;
	FLMUINT						uiSaveDictSeq;
	FLMUINT						uiTmpLf;
	LF_STATS *					pTmpLfStats;
	POOL							tmpPool;
	FLMBOOL						bRSFinalized = FALSE;
	DB_CHECK_PROGRESS *		pProgress = pDbInfo->pProgress;
	IX_CHK_INFO					IxChkInfo;
	IX_CHK_INFO *				pIxChkInfo = NULL;
	void *						pvPoolMark;
	FILE_HDR *					pFileHdr = &pDb->pFile->FileHdr;

	for (uiCurrLevel = 0; uiCurrLevel < BH_MAX_LEVELS; uiCurrLevel++)
	{
		bStateInitialized [uiCurrLevel] = FALSE;
	}
	
	if (*pbStartOverRV)
	{
		goto Exit;
	}
	
	pvPoolMark = GedPoolMark( pPool);
	uiSaveDictSeq = pDb->pDict->uiDictSeq;
	
	if( RC_BAD( rc = chkSetupLfTable( pDbInfo, pPool)))
	{
		goto Exit;
	}

	if( pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING)
	{
		if( RC_BAD( rc = chkSetupIxInfo( pDbInfo, &IxChkInfo)))
		{
			goto Exit;
		}
		pIxChkInfo = &IxChkInfo;
	}

	/*
	Loop through all of the logical files in the database
	and perform a structural and logical check.
	*/
	
	uiCurrLf = 0;
	while (uiCurrLf < pDbInfo->pProgress->uiNumLogicalFiles)
	{
		pProgress->uiCurrLF = uiCurrLf + 1;
		pLogicalFile = &pDbInfo->pLogicalFiles [uiCurrLf];
		pLfStats = &pDbInfo->pProgress->pLfStats [uiCurrLf];
		pLFile = pLogicalFile->pLFile;
		if (pLFile->uiRootBlk == BT_END)
		{
			rc = FERR_OK;
			uiCurrLf++;
			uiResetKeyLen = 0;
			pucResetKey = NULL;
			uiResetDrn = 0;
			continue;
		}

		/* Allocate space to hold the keys, if not already allocated. */

		if (uiKeysAllocated < pLfStats->uiNumLevels)
		{

			/* If there is already a key allocated, deallocate it */

			if (pKeyBuffer)
			{
				f_free( &pKeyBuffer);
				uiKeysAllocated = 0;
			}
		
			if( RC_BAD( rc = f_alloc( 
				pLfStats->uiNumLevels * MAX_KEY_SIZ, &pKeyBuffer)))
			{
				goto Exit;
			}
			uiKeysAllocated = pLfStats->uiNumLevels;
		}

		/* Setup PROGRESS_CHECK_INFO structure */

		pProgress->iCheckPhase = CHECK_B_TREE;
		pProgress->bStartFlag = TRUE;
		pProgress->uiLfNumber = pLFile->uiLfNum;
		pProgress->uiLfType = pLFile->uiLfType;

		if( pLFile->uiLfType == LF_INDEX)
		{
			pProgress->bUniqueIndex = (pLogicalFile->pIxd->uiFlags & IXD_UNIQUE) ? TRUE : FALSE;
		}

		if (RC_BAD( rc = chkCallProgFunc( pDbInfo)))
		{
			break;
		}

		pProgress->bStartFlag = FALSE;

		f_yieldCPU();


		/* Initialize the state information for each level of the B-TREE. */

		for (uiCurrLevel = 0; uiCurrLevel < pLfStats->uiNumLevels; uiCurrLevel++)
		{

			/*
			If we are resetting to a particular key, save the statistics
			which were gathered so far.
			*/

			if (uiResetKeyLen)
			{

				/* Save the statistics which were gathered. */

				pLfStats->pLevelInfo [uiCurrLevel].ui64KeyCount =
					State [uiCurrLevel].ui64KeyCount;
				f_memcpy( &pLfStats->pLevelInfo [uiCurrLevel].BlockInfo,
									&State [uiCurrLevel].BlkInfo, sizeof( BLOCK_INFO));
			}

			flmInitReadState( &State [uiCurrLevel], &bStateInitialized [uiCurrLevel],
									pFileHdr->uiVersionNum,
									pDb, pLogicalFile, uiCurrLevel,
									(FLMUINT)((!uiCurrLevel)
												 ? (FLMUINT)BHT_LEAF
												 : (FLMUINT)BHT_NON_LEAF),
									&pKeyBuffer [uiCurrLevel * MAX_KEY_SIZ]);

			if (!uiResetKeyLen)
			{
				State [uiCurrLevel].uiLastChildAddr = BT_END;
				State [uiCurrLevel].uiElmLastFlag = TRUE;
			}
			else
			{

				/* Restore the statistics which were gathered so far. */

				State [uiCurrLevel].ui64KeyCount =
					pLfStats->pLevelInfo [uiCurrLevel].ui64KeyCount;
				f_memcpy( &State [uiCurrLevel].BlkInfo,
							 &pLfStats->pLevelInfo [uiCurrLevel].BlockInfo,
							 sizeof( BLOCK_INFO));
			}
		}

		/*
		Need to finalize the result set used by the logical
		check.  If the current logical file is an index and the
		result set has not been finalized, call chkRSFinalize.
		*/

		if( !pDbInfo->pProgress->bPhysicalCorrupt &&
			(pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING) &&
			bRSFinalized == FALSE && pLFile->uiLfType == LF_INDEX)
		{
			FLMUINT64	ui64NumRSKeys = 0;

			/*
			Finalize the result set.
			*/

			if( RC_BAD( rc = chkRSFinalize( pIxChkInfo, &ui64NumRSKeys)))
			{
				goto Exit;
			}

			/*
			Reset uiNumKeys to reflect the number of keys
			in the result set now that all duplicates have
			been eliminated.
			*/

			if( pDbInfo->pProgress->ui64NumKeys > ui64NumRSKeys)
			{
				pDbInfo->pProgress->ui64NumDuplicateKeys =
					pDbInfo->pProgress->ui64NumKeys - ui64NumRSKeys;
			}
			pDbInfo->pProgress->ui64NumKeys = ui64NumRSKeys;

			/*
			Set bRSFinalized to TRUE so that subsequent passes will not
			attempt to finalize the result set again.
			*/

			bRSFinalized = TRUE;
		}

		/*
		Call chkVerifySubTree to check the B-TREE starting at the
		root block.
		*/

		GedPoolInit( &tmpPool, 512);
		pDbInfo->bReposition = FALSE;
		rc = chkVerifySubTree( pDbInfo, pIxChkInfo, NULL,
								 &State [pLfStats->uiNumLevels - 1],
								 pLFile->uiRootBlk, &tmpPool,
								 pucResetKey, uiResetKeyLen, uiResetDrn);
		GedPoolFree( &tmpPool);

		if (rc == FERR_OLD_VIEW)
		{

			// If it is a read transaction, reset.
			
			if( flmGetDbTransType( pDb) == FLM_READ_TRANS)
			{

				// Free the KrefCntrl

				KrefCntrlFree( pDb);

				// Abort the read transaction

				if( RC_BAD( rc = flmAbortDbTrans( pDb)))
				{
					goto Exit;
				}

				// Try to start a new read transaction
			
				if( RC_BAD( rc = flmBeginDbTrans( pDb,
					FLM_READ_TRANS, 0, FLM_DONT_POISON_CACHE)))
				{
					goto Exit;
				}
			}
			rc = FERR_OK;
			pDbInfo->bReposition = TRUE;
		}
		if (RC_BAD( rc))
		{
			goto Exit;
		}

		// We may get told to reposition if we had to repair
		// an index or we got an old view error.

		if (pDbInfo->bReposition)
		{

			/* If the dictionary has changed we must start all over. */

			if (pDb->pDict->uiDictSeq != uiSaveDictSeq)
			{
				*pbStartOverRV = TRUE;
				goto Exit;
			}

			/*
			Save the current key at the bottom level of the B-Tree.
			This is the point we want to try to reset to.  Don't change
			the reset key if the current key length is zero - this may
			have occurred because of some error - we want to keep moving
			forward in the file if at all possible.
			*/

			if (State [0].uiCurKeyLen)
			{
				uiResetKeyLen = State [0].uiCurKeyLen;
				pucResetKey = &ucResetKeyBuff [0];
				uiResetDrn = State [0].uiCurrIxRefDrn;
				f_memcpy( pucResetKey, State [0].pCurKey, uiResetKeyLen);
			}

			// Re-read each logical file's LFH information.

			pProgress->ui64DatabaseSize =
				FSGetSizeInBytes( pDb->pFile->uiMaxFileSize,
										pDb->LogHdr.uiLogicalEOF);

			/*
			Reread each of the LFH blocks and update the root block
			address and other pertinent information for each logical
			file.
			*/

			for (uiTmpLf = 0, pTmpLfStats = pDbInfo->pProgress->pLfStats;
					uiTmpLf < pDbInfo->pProgress->uiNumLogicalFiles;
					uiTmpLf++, pTmpLfStats++)
			{
				FLMBOOL	bCurrLfLevelChanged = FALSE;
				
				if (RC_BAD( rc = chkGetLfInfo( pDbInfo, pPool,
					pTmpLfStats, pDbInfo->pLogicalFiles [uiTmpLf].pLFile,
					pLfStats, &bCurrLfLevelChanged)))
				{
					goto Exit;
				}

				/*
				If the number of levels for the current logical file
				changed, reset things so we will recheck the entire logical
				file.
				*/
				
				if (bCurrLfLevelChanged)
				{
					pucResetKey = NULL;
					uiResetKeyLen = 0;
				}
			}
			continue;
		}

		/*
		Verify that all of the levels' next block address's
		are BT_END.
		*/

		if (RC_OK( pDbInfo->LastStatusRc))
		{
			for (uiCurrLevel = 0; uiCurrLevel < pLfStats->uiNumLevels; uiCurrLevel++)
			{

				/* Save the statistics which were gathered. */

				pLfStats->pLevelInfo [uiCurrLevel].ui64KeyCount =
					State [uiCurrLevel].ui64KeyCount;
				f_memcpy( &pLfStats->pLevelInfo [uiCurrLevel].BlockInfo,
							 &State [uiCurrLevel].BlkInfo, sizeof( BLOCK_INFO));

				/*
				Make sure the last block had a NEXT block address
				of BT_END.
				*/

				if ((State [uiCurrLevel].uiNextBlkAddr) &&
					 (State [uiCurrLevel].uiNextBlkAddr != BT_END))
				{
					chkReportError( pDbInfo, FLM_BAD_LAST_BLK_NEXT,
										LOCALE_B_TREE,
									 pDbInfo->pProgress->uiLfNumber,
									 pDbInfo->pProgress->uiLfType,
										 uiCurrLevel, 0, 0, 0, 0,
										 0xFFFF, 0, NULL);
				}
			}
		}

		if (RC_BAD( pDbInfo->LastStatusRc))
		{
			break;
		}

		uiCurrLf++;
		pucResetKey = NULL;
		uiResetKeyLen = 0;
		uiResetDrn = 0;

	}

	/*
	If index check was requested, no structural corruptions
	were detected, and this is the last logical file, need to make
	sure that the result set is empty.
	*/

	if( RC_OK( rc) && !pDbInfo->pProgress->bPhysicalCorrupt &&
		(pDbInfo->uiFlags & FLM_CHK_INDEX_REFERENCING) &&
		bRSFinalized == TRUE &&
		uiCurrLf == pDbInfo->pProgress->uiNumLogicalFiles)
	{
		for( ;;)
		{
			if( RC_BAD( rc = chkGetNextRSKey( pIxChkInfo)))
			{
				if( rc == FERR_EOF_HIT || rc == FERR_NOT_FOUND)
				{
					rc = FERR_OK;
					break;
				}
				goto Exit;
			}
			else
			{
				/* Updated statistics */
				
				pIxChkInfo->pDbInfo->pProgress->ui64NumKeysExamined++;

				if( RC_BAD( rc = chkResolveIXMissingKey(
											&(State[ 0]), pIxChkInfo)))
				{
					goto Exit;
				}
			}
		}
	}
	
	/* Clear the unique index flag */

	pProgress->bUniqueIndex = FALSE;

Exit:

	// Clear the pRecord for each level in the state array.

	for (uiCurrLevel = 0; uiCurrLevel < BH_MAX_LEVELS; uiCurrLevel++)
	{
		if (bStateInitialized [uiCurrLevel] && State [uiCurrLevel].pRecord)
		{
			State [uiCurrLevel].pRecord->Release();
			State [uiCurrLevel].pRecord = NULL;
		}
	}
	
	/*	Cleanup any temporary index check files */

	if( pIxChkInfo != NULL)
	{
		chkFreeIxInfo( pIxChkInfo);
	}

	if (pKeyBuffer)
	{
		f_free( &pKeyBuffer);
	}

	if (RC_OK( rc) && RC_BAD( pDbInfo->LastStatusRc))
	{
		rc = pDbInfo->LastStatusRc;
	}

	return( rc);
}


/********************************************************************
Desc:
*********************************************************************/
RCODE chkReportError(
	DB_INFO *			pDbInfo,
	eCorruptionType	eCorruption,
	eCorruptionLocale	eErrLocale,
	FLMUINT				uiErrLfNumber,
	FLMUINT				uiErrLfType,
	FLMUINT				uiErrBTreeLevel,
	FLMUINT				uiErrBlkAddress,
	FLMUINT				uiErrParentBlkAddress,
	FLMUINT				uiErrElmOffset,
	FLMUINT				uiErrDrn,
	FLMUINT				uiErrElmRecOffset,
	FLMUINT				uiErrFieldNum,
	FLMBYTE *			pBlk)
{
	CORRUPT_INFO	CorruptInfo;
	FLMBOOL			bFixErr;

	CorruptInfo.eCorruption = eCorruption;
	CorruptInfo.eErrLocale = eErrLocale;
	CorruptInfo.uiErrLfNumber = uiErrLfNumber;
	CorruptInfo.uiErrLfType = uiErrLfType;
	CorruptInfo.uiErrBTreeLevel = uiErrBTreeLevel;
	CorruptInfo.uiErrBlkAddress = uiErrBlkAddress;
	CorruptInfo.uiErrParentBlkAddress = uiErrParentBlkAddress;
	CorruptInfo.uiErrElmOffset = uiErrElmOffset;
	CorruptInfo.uiErrDrn = uiErrDrn;
	CorruptInfo.uiErrElmRecOffset = uiErrElmRecOffset;
	CorruptInfo.uiErrFieldNum = uiErrFieldNum;
	CorruptInfo.pBlk = pBlk;
	CorruptInfo.pErrIxKey = NULL;
	CorruptInfo.pErrRecord = NULL;
	CorruptInfo.pErrRecordKeyList = NULL;
	if ((pDbInfo->fnStatusFunc) && (RC_OK( pDbInfo->LastStatusRc)))
	{
		bFixErr = FALSE;
		pDbInfo->LastStatusRc = (*pDbInfo->fnStatusFunc)( FLM_PROBLEM_STATUS,
												(void *)&CorruptInfo,
												(void *)&bFixErr,
												pDbInfo->pProgress->AppArg);
	}
	if (eCorruption != FLM_OLD_VIEW)
	{
		pDbInfo->pProgress->bPhysicalCorrupt = TRUE;
		pDbInfo->uiFlags &= ~FLM_CHK_INDEX_REFERENCING;
	}

	return( pDbInfo->LastStatusRc);
}

/***************************************************************************
Desc:	Initializes an IX_CHK_INFO structure
*****************************************************************************/
FSTATIC RCODE chkSetupIxInfo(
	DB_INFO *		pDbInfo,
	IX_CHK_INFO *	pIxInfoRV
	)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiIxCount = 0;
	FLMUINT			uiIxNum = 0;
	IXD *				pIxd;
	LFILE *			pLFile;
	char				szTmpIoPath [F_PATH_MAX_SIZE];
	char				szBaseName [F_FILENAME_SIZE];
	FDB *				pDb = pDbInfo->pDb;

	f_memset( pIxInfoRV, 0, sizeof( IX_CHK_INFO));
	GedPoolInit( &(pIxInfoRV->pool), 512);
	pIxInfoRV->pDbInfo = pDbInfo;

	/* Set up the result set path */
	
	if( RC_BAD( rc = flmGetTmpDir( szTmpIoPath)))
	{
		if( rc == FERR_IO_PATH_NOT_FOUND ||
			 rc == FERR_IO_INVALID_PATH)
		{
			if( RC_BAD( rc = f_pathReduce( pDb->pFile->pszDbPath,
										szTmpIoPath, szBaseName)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}

	/*
	Initialize the result set.  The result set will be used
	to build an ordered list of keys for comparision to
	the database's indexes.
	*/

	if( RC_BAD( rc = chkRSInit( szTmpIoPath, &(pIxInfoRV->pRSet))))
	{
		goto Exit;
	}

	/* Build list of all indexes */

	uiIxCount = 0;
	if( pDb->pDict)
	{
		uiIxCount += pDb->pDict->uiIxdCnt;
	}

	/*
	Allocate memory to save each index number and its associated
	container number.
	*/

	if( RC_BAD( rc = f_alloc( 
				(FLMUINT)((sizeof( FLMUINT) * uiIxCount) + (sizeof( FLMBOOL) * uiIxCount)),
				&(pIxInfoRV->puiIxArray))))
	{
		goto Exit;
	}

	/* Save the index numbers into the array. */

	uiIxCount = 0;
	if( pDb->pDict)
	{
		for( uiIxNum = 0, pIxd = (IXD *)pDb->pDict->pIxdTbl;
			  uiIxNum < pDb->pDict->uiIxdCnt;
			  uiIxNum++, pIxd++)
		{
			pIxInfoRV->puiIxArray[ uiIxCount] = pIxd->uiIndexNum;
			uiIxCount++;
		}
	}

	if( RC_OK( fdictGetIndex( pDb->pDict,
						pDb->pFile->bInLimitedMode,
						FLM_DICT_INDEX, &pLFile, NULL)))
	{
		pIxInfoRV->puiIxArray[ uiIxCount] = FLM_DICT_INDEX;
		uiIxCount++;
	}

	pIxInfoRV->uiIxCount = uiIxCount;
	pIxInfoRV->bGetNextRSKey = TRUE;

Exit:

	/*
	Clean up any memory on error exit.
	*/

	if( RC_BAD( rc))
	{
		GedPoolFree( &(pIxInfoRV->pool));
		if( pIxInfoRV->puiIxArray)
		{
			f_free( &(pIxInfoRV->puiIxArray));
		}
	}

	return( rc);
}

/********************************************************************
Desc: Outputs keys to the temporary result set
*********************************************************************/
FSTATIC RCODE chkOutputIndexKeys(
	STATE_INFO *	pStateInfo,
	IX_CHK_INFO *	pIxChkInfo,
	IXD *				pIxd,
	REC_KEY *		pKeyList
	)
{
	FLMUINT		uiKeyLen;
	REC_KEY *   pKey;
	FLMBYTE		ucBuf[ MAX_KEY_SIZ + RS_KEY_OVERHEAD];
	RCODE			rc = FERR_OK;


	pKey = pKeyList;
	while( pKey)
	{
		/* Set the index and reference */

		UW2FBA( (FLMUINT16)pIxd->uiIndexNum, &(ucBuf[ RS_IX_OFFSET]));
		UD2FBA( (FLMUINT32)pStateInfo->uiElmDrn, &(ucBuf[ RS_REF_OFFSET]));

		/* Convert the key tree to a collation key */
		
		if( RC_BAD( rc = KYTreeToKey( pIxChkInfo->pDbInfo->pDb,
			pIxd, pKey->pKey, pKey->pKey->getContainerID(),
			&(ucBuf[ RS_KEY_OVERHEAD]), &uiKeyLen, 0)))
		{
			goto Exit;
		}

		/* Add the composite key (index, ref, key) to the result set */

		if( RC_BAD( rc = chkRSAddEntry( pIxChkInfo->pRSet, ucBuf,
			uiKeyLen + RS_KEY_OVERHEAD)))
		{
			goto Exit;
		}

		/*
		Update statistics.  Note that uiNumKeys will reflect the
		total number of keys generated by records, including any
		duplicate keys.  This value is updated to reflect the
		correct number of keys once the result set has been finalized.
		*/
		
		pIxChkInfo->pDbInfo->pProgress->ui64NumKeys++;
		pKey = pKey->pNextKey;
	}

Exit:

	return( rc);
}
