//-------------------------------------------------------------------------
// Desc:	Data dictionary creation routines.
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
// $Id: ddcreate.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE DDMakeDictIxKey(
	FDB *				pDb,
	FlmRecord *		pRecord,
	FLMBYTE *		pKeyBuf,
	FLMUINT  *		puiKeyLenRV);

FSTATIC RCODE DDCheckNameConflict(
	FDB *				pDb,
	LFILE *			pDictIxLFile,
	FlmRecord *		pNewRec,
	FLMUINT			uiDrn,
	FlmRecord *		pOldRec);
	
FSTATIC RCODE DDCheckIDConflict(
	FDB *				pDb,
	LFILE *			pDictContLFile,
	FLMUINT			uiDrn);

FSTATIC RCODE DDIxDictRecord(
	FDB *				pDb,
	LFILE *			pDictIxLFile,
	FLMUINT			uiDrn,
	FlmRecord *		pRecord,
	FLMUINT			uiFlags);

/**************************************************************************** 
Desc:		Allocate the LFILE and read in the LFile entries.  The default
			data container and the dictionary container will be at hard coded
			slots at the first of the table.  The LFiles do not need to be in
			any numeric order.
****************************************************************************/
RCODE fdictReadLFiles(
	FDB *			pDb,						/* (IN) (OUT) table pointers */
	FDICT *		pDict)
{
	RCODE			rc = FERR_OK;
	LFILE *		pLFiles = NULL;
	LFILE *		pLFile;
	SCACHE *		pSCache;
	FLMBOOL		bReleaseCache = FALSE;
	FLMBYTE *	pucBlk;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiPos;
	FLMUINT		uiEndPos;
	FLMUINT		uiEstCount;
	FLMUINT		uiLFileCnt;
	FLMUINT		uiLFHCnt;
	FFILE *		pFile = pDb->pFile;
	FLMUINT		uiBlkSize = pFile->FileHdr.uiBlockSize;
	LFILE			TmpLFile;

	f_memset( &TmpLFile, 0, sizeof( LFILE));

	for( uiEstCount = 0, uiLFileCnt = 4,
			uiBlkAddress = pDb->pFile->FileHdr.uiFirstLFHBlkAddr
		; uiBlkAddress != BT_END
		; )
	{
		if( RC_BAD( rc = ScaGetBlock( pDb, NULL, BHT_LFH_BLK,
										uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pucBlk = pSCache->pucBlk;
		uiPos = BH_OVHD;
		if( (uiEndPos = (FLMUINT)FB2UW( &pucBlk[ BH_ELM_END])) <= BH_OVHD)
		{
			uiEndPos = BH_OVHD;
			uiLFHCnt = 0;
		}
		else
		{
			if( uiEndPos > uiBlkSize)
				uiEndPos = uiBlkSize;
			uiLFHCnt = (FLMUINT)((uiEndPos - BH_OVHD) / LFH_SIZE);
			uiEndPos = (FLMUINT)(BH_OVHD + uiLFHCnt * LFH_SIZE);
		}

		// May allocate too many like the inactive ones but OK for now.
		// Allocate an additional 2 for the default data and dict containers.

		if( !uiEstCount)				/* First time */
		{
			uiEstCount = uiLFHCnt + uiLFileCnt;
			if( uiEstCount)
			{
				if( RC_BAD( rc = f_calloc( uiEstCount * sizeof( LFILE), &pLFiles)))
				{
					goto Exit;
				}
			}
		}
		else if( uiLFHCnt)
		{
			uiEstCount += uiLFHCnt;

			if( RC_BAD(rc = f_recalloc( uiEstCount * sizeof(LFILE), &pLFiles)))
			{
				goto Exit;
			}
		}

		/* Read through all of the logical file definitions in the block */

		for( ; uiPos < uiEndPos; uiPos += LFH_SIZE)
		{
			FLMUINT	uiLfNum;

			// Have to fix up the offsets later when they are read in

			if( RC_BAD( rc = flmBufferToLFile( &pucBlk[ uiPos], &TmpLFile,
								uiBlkAddress, uiPos)))
			{
				goto Exit;
			}

			if( TmpLFile.uiLfType == LF_INVALID)
			{
				continue;
			}

			uiLfNum = TmpLFile.uiLfNum;

			if( uiLfNum == FLM_DATA_CONTAINER)
			{
				pLFile = pLFiles + LFILE_DATA_CONTAINER_OFFSET;
			}
			else if( uiLfNum == FLM_DICT_CONTAINER)
			{
				pLFile = pLFiles + LFILE_DICT_CONTAINER_OFFSET;
			}
			else if( uiLfNum == FLM_DICT_INDEX)
			{
				pLFile = pLFiles + LFILE_DICT_INDEX_OFFSET;
			}
			else if( uiLfNum == FLM_TRACKER_CONTAINER)
			{
				pLFile = pLFiles + LFILE_TRACKER_CONTAINER_OFFSET;
			}
			else
			{
				pLFile = pLFiles + uiLFileCnt++;
			}

			f_memcpy( pLFile, &TmpLFile, sizeof(LFILE));
		}

		// Get the next block in the chain

		uiBlkAddress = (FLMUINT)FB2UD( &pucBlk[ BH_NEXT_BLK]);
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

	// This routine could be called to re-read in the dictionary.

	if( pDict->pLFileTbl)
	{
		f_free( &pDict->pLFileTbl);
	}

	pDict->pLFileTbl = pLFiles;
	pDict->uiLFileCnt = uiLFileCnt;

Exit:
	
	if( bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	if( RC_BAD(rc) && pLFiles)
	{
		f_free( &pLFiles);
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Add data dictionary records to the data dictionary.
****************************************************************************/
RCODE fdictCreate(
	FDB *					pDb,
	const char *		pszDictPath,
	const char *		pDictBuf)
{
	RCODE    			rc = FERR_OK;
	F_FileHdl *			pDictFileHdl = NULL;
	FlmRecord *			pDictRec = NULL;
	void *				pvField;
	const char *		pucGedBuf;
	LFILE *				pDictContLFile;
	LFILE *				pDictIxLFile;
	FLMUINT				uiDrn = 0;
	FLMUINT				uiCurrDictNum;
	FLMUINT				uiLFileCount;
	FLMBOOL				bFileOpen = FALSE;
	LFILE					DictContLFile;
	LFILE					DictIxLFile;
	LFILE					TempLFile;
	char					ucTempBuf[ 256];
	FLMUINT				uiBufLen = sizeof( ucTempBuf);
	F_NameTable 		nameTable;

	// Initialize the name table

	if( RC_BAD( rc = nameTable.setupFromDb( HFDB_NULL)))
	{
		goto Exit;
	}

	/* Create Dictionary and Default Data containers */

	if( RC_BAD(rc = flmLFileCreate( pDb, &DictContLFile, FLM_DICT_CONTAINER,
											 LF_CONTAINER)))
	{
		goto Exit;
	}
	uiCurrDictNum = FLM_DICT_CONTAINER;

	if( RC_BAD(rc = flmLFileCreate( pDb, &TempLFile, FLM_DATA_CONTAINER,
								LF_CONTAINER)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmLFileCreate( pDb, &DictIxLFile, FLM_DICT_INDEX,
							LF_INDEX)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = flmLFileCreate( pDb, &TempLFile, FLM_TRACKER_CONTAINER,
							LF_CONTAINER)))
	{
		goto Exit;
	}

	uiLFileCount = 4;

	// If we have a GEDCOM buffer, there is no need to open the file

	if( pDictBuf)
	{
		pucGedBuf = pDictBuf;
		uiBufLen = f_strlen( pDictBuf) + 1;
	}
	else if( pszDictPath)
	{
		pucGedBuf = ucTempBuf;
		if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Open( 
				pszDictPath, F_IO_RDONLY, &pDictFileHdl)))
		{
			goto Exit;
		}
		bFileOpen = TRUE;
	}
	else
	{
		/*
		Neither a dictionary buffer or file were specified.  Create will
		be done with an empty dictionary.
		*/

		goto Done_Getting_Dict;
	}

	/*
	Create a new FDICT so we can write the dictionary records.
	This FDICT is temporary and will be allocated again.
	*/

	if( RC_BAD( rc = fdictCreateNewDict( pDb)))
	{
		goto Exit;
	}

	if( (pDictRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetContainer( pDb->pDict, 
				FLM_DICT_CONTAINER, &pDictContLFile)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictGetIndex( pDb->pDict,
					pDb->pFile->bInLimitedMode,
					FLM_DICT_INDEX, &pDictIxLFile, NULL)))
	{
		goto Exit;
	}

	/*
	Read through the dictionary records, adding them or creating dictionaries
	as we go.
	*/

	for( ;;)
	{
		/* Get records from buffer or file. */

		rc = ( pDictFileHdl)
			  ? pDictRec->importRecord( pDictFileHdl, &nameTable)
			  : pDictRec->importRecord( &pucGedBuf, uiBufLen, &nameTable);

		if( RC_BAD( rc))
		{
			if( rc == FERR_END || rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else if( uiDrn)
			{
				// If an error occur then at least set the DRN of the 
				// previous record in the diagnostic information.

				pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
				pDb->Diag.uiDrn = uiDrn;
			}
			goto Exit;
		}

		// See if we are switching dictionaries.
		
		pvField = pDictRec->root();
		if( pDictRec->getFieldID( pvField) == FLM_DICT_TAG)
		{
			rc = RC_SET( FERR_INVALID_TAG);
			goto Exit;
		}

		// Assign all fields a DRN value - parse for completeness.
		// If there is no DRN in the record (zero), one will be assigned
		// by FDDDictRecUpdate.

		uiDrn = pDictRec->getID();

		/*
		Add the data dictionary record.  This also checks to see
		if the record is already defined.
		*/

		if( RC_BAD( rc = fdictRecUpdate( pDb, pDictContLFile,
									pDictIxLFile, &uiDrn, pDictRec, NULL)))
		{
			goto Exit;
		}

		// Don't need to do the processing below if it is not a record
		// being put into the dictionary.

		if( uiCurrDictNum != FLM_DICT_CONTAINER)
		{
			continue;
		}

		// Create an LFILE for each index and container.
	
		if( pDictRec->getFieldID( pvField) == FLM_INDEX_TAG ||
			 pDictRec->getFieldID( pvField) == FLM_CONTAINER_TAG)
		{
			pvField = pDictRec->root();
			if( RC_BAD( rc = flmLFileCreate( pDb, &TempLFile, uiDrn,
									((pDictRec->getFieldID( pvField) == FLM_INDEX_TAG)
										? (FLMUINT)LF_INDEX 
										: (FLMUINT)LF_CONTAINER))))
			{
				goto Exit;
			}
			uiLFileCount++;
		}
	}

Done_Getting_Dict:

	// Create the FDICT again, this time with the dictionary pcode. 

	if( RC_BAD( rc = fdictCreateNewDict( pDb)))
	{
		goto Exit;
	}

Exit:

	if( bFileOpen)
	{
		pDictFileHdl->Close();
	}

	if( pDictFileHdl)
	{
		pDictFileHdl->Release();
	}

	if( pDictRec)
	{
		pDictRec->Release();
	}

	return( rc);
}

/**************************************************************************** 
Desc:		Creates a new dictionary for a database.
			This occurs when on database create and on a dictionary change.
****************************************************************************/
RCODE fdictCreateNewDict(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;

	// Unlink the DB from the current FDICT, if any.

	if( pDb->pDict)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		flmUnlinkFdbFromDict( pDb);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	// Allocate a new FDICT structure for the new dictionary we
	// are going to create.

	if( RC_BAD( rc = fdictRebuild( pDb)))
	{
		goto Exit;
	}

	// Update the FDB structure to indicate that the dictionary
	// was updated.

	pDb->uiFlags |= FDB_UPDATED_DICTIONARY;
	
Exit:

	// If we allocated an FDICT and there was an error, free the FDICT.

	if( (RC_BAD( rc)) && (pDb->pDict))
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}
	
	return( rc);
}

/**************************************************************************** 
Desc:	Add a new field, container or index definition to the dictionary.
****************************************************************************/
RCODE flmAddRecordToDict(
	FDB *			pDb,
	FlmRecord *	pRecord,
	FLMUINT		uiDictId,
	FLMBOOL		bRereadLFiles)
{
	RCODE			rc = FERR_OK;
	TDICT			tDict;
	FLMBOOL		bTDictInitialized = FALSE;

	if( RC_BAD( rc = fdictCopySkeletonDict( pDb)))
	{
		goto Exit;
	}

	bTDictInitialized = TRUE;
	if( RC_BAD( rc = fdictInitTDict( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictProcessRec( &tDict, pRecord, uiDictId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictBuildTables( &tDict, bRereadLFiles, TRUE)))
	{
		goto Exit;
	}

	pDb->uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	if( bTDictInitialized)
	{
		GedPoolFree( &tDict.pool);
	}

	// If we allocated an FDICT and there was an error, free the FDICT.

	if( (RC_BAD( rc)) && (pDb->pDict))
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:		Add an index a dictionary record to the container LFILE and the
			index LFILE.
****************************************************************************/
RCODE fdictRecUpdate(
	FDB *				pDb,
	LFILE *			pDictContLFile,
	LFILE *			pDictIxLFile,
	FLMUINT *		puiDrnRV,
	FlmRecord *		pNewRec,
	FlmRecord *		pOldRec,
	FLMBOOL			bRebuildOp)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiDrn = *puiDrnRV;
	FLMBOOL			bAllocatedID;
	void *			pvField;
	FLMBYTE *		pucKeyField = NULL;
	FLMUINT32		ui32BufLen;
	FLMUINT			uiEncType;

	bAllocatedID = FALSE;

	// Make sure we are using a valid DRN

	if( (uiDrn >= FLM_RESERVED_TAG_NUMS) &&
		 (uiDrn != 0xFFFFFFFF))
	{
		pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
		pDb->Diag.uiDrn = uiDrn;
		rc = RC_SET( FERR_BAD_DICT_DRN);
		goto Exit;
	}

	// Allocate an unused DRN, if one has not been allocated.

	if( (pNewRec) && ((!uiDrn) || (uiDrn == 0xFFFFFFFF)))
	{
		FLMBOOL	bAllocAtEnd = (!uiDrn) ? TRUE : FALSE;

		bAllocatedID = TRUE;
		if( bAllocAtEnd)
		{
			if( RC_BAD( rc = FSGetNextDrn( pDb, pDictContLFile, FALSE, &uiDrn)))
			{
				goto Exit;
			}
		}
		else
		{
			// Scott 12/99: This must not be called any more.
			// The code merged ITT values into the table.
			
			flmAssert(0);
		}

		// Verify that we are not at our highest possible dictionary DRN.
		
		if( uiDrn >= FLM_RESERVED_TAG_NUMS)
		{
			rc = RC_SET( FERR_NO_MORE_DRNS);
			goto Exit;
		}
	}

	// The following code makes sure that the DRN and name have not already been
	// used, if adding.  It also makes sure that there is no conflict in
	// the type/name index.  It checks the entire shared dictionary
	// hierarchy if necessary - child and parent - to ensure no
	// conflicts.

	if( pNewRec)
	{
		// Check for ID conflicts in the dictionary being added to

		if( (!pOldRec) && (!bAllocatedID))
		{
			if( RC_BAD( rc = DDCheckIDConflict( pDb, pDictContLFile, uiDrn)))
			{
				if( (rc == FERR_ID_RESERVED) || (rc == FERR_DUPLICATE_DICT_REC))
				{
					pvField = pNewRec->root();
					if( (rc == FERR_DUPLICATE_DICT_REC) &&
						 (pNewRec->getFieldID( pvField) == FLM_RESERVED_TAG))
					{
						rc = RC_SET( FERR_CANNOT_RESERVE_ID);
					}
					pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
					pDb->Diag.uiDrn = uiDrn;
				}
				goto Exit;
			}
		}

		// Check for name conflicts in the dictionary being added to

		if (pNewRec)
		{
			if (RC_BAD( rc = DDCheckNameConflict( pDb, pDictIxLFile, pNewRec,
											uiDrn, pOldRec)))
				goto Exit;
		}
	}

	if (!pOldRec && pNewRec)
	{
		// If this is an encryption definition record, we need to generate
		// a new key.
		
		if (pNewRec->getFieldID( pNewRec->root()) == FLM_ENCDEF_TAG && 
				!bRebuildOp && !(pDb->uiFlags & FDB_REPLAYING_RFL))
		{
			F_CCS			Ccs;

			// If we are running in limited mode, we will not be able to complete
			// this operation.

			if( pDb->pFile->bInLimitedMode)
			{
				rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
				goto Exit;
			}

			// Should not have a key yet.
			
			if( pNewRec->find(pNewRec->root(),
									FLM_KEY_TAG) != NULL)
			{
				rc = RC_SET( FERR_CANNOT_SET_KEY);
				goto Exit;
			}

			if( (pvField = pNewRec->find( pNewRec->root(),
													FLM_TYPE_TAG)) == NULL)
			{
				rc = RC_SET( FERR_MISSING_ENC_TYPE);
				goto Exit;
			}

			if( RC_BAD( rc = DDGetEncType( pNewRec, pvField, &uiEncType)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = Ccs.init( FALSE, uiEncType)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = Ccs.generateEncryptionKey()))
			{
				goto Exit;
			}

			if( RC_BAD( rc = Ccs.getKeyToStore( &pucKeyField, &ui32BufLen,
				NULL, pDb->pFile->pDbWrappingKey)))
			{
				goto Exit;
			}

			// Create the key field
			
			if( RC_BAD( rc = pNewRec->insert( pNewRec->root(), INSERT_LAST_CHILD,
				FLM_KEY_TAG, FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			// Set the value of the new field
			
			if( RC_BAD( rc = pNewRec->setNative( pvField, 
				(const char *)pucKeyField)))
			{
				goto Exit;
			}
		}
	}

	// Delete the old record and its index entries, if any

	if( pOldRec)
	{
		// Delete the old record's index entries

		if( RC_BAD( rc = DDIxDictRecord( pDb, pDictIxLFile, uiDrn,
			pOldRec, KREF_DELETE_FLAG)))
		{
			goto Exit;
		}
		
		// Delete the old record - unless it is a modify

		if( !pNewRec)
		{
			if( RC_BAD( rc = FSRecUpdate( pDb, pDictContLFile, NULL, uiDrn,
				REC_UPD_DELETE)))
			{
				goto Exit;
			}
		}
	}

	// Add the new record, if any

	if( pNewRec)
	{
		// Add the record's index keys

		if( RC_BAD( rc = DDIxDictRecord( pDb, pDictIxLFile, uiDrn,
			pNewRec, 0)))
		{
			goto Exit;
		}

		// Add or modify the record itself

		if( RC_BAD( rc = FSRecUpdate( pDb, pDictContLFile, pNewRec,
						uiDrn, (FLMUINT)((pOldRec)
											  ? (FLMUINT)REC_UPD_MODIFY
											  : (FLMUINT)REC_UPD_ADD))))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_OK( rc))
	{
		*puiDrnRV = uiDrn;
	}

	if (pucKeyField)
	{
		f_free( &pucKeyField);
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Creates a collated type/name key for a dictionary record.
****************************************************************************/
FSTATIC RCODE DDMakeDictIxKey(
	FDB *				pDb,
	FlmRecord *		pRecord,
	FLMBYTE *		pKeyBuf,
	FLMUINT *		puiKeyLenRV)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiElmLen;
	FLMUINT				uiKeyLen = 0;
	void *				pvField = pRecord->root();
	const FLMBYTE *	pExportPtr;

	// Collate the name

	pExportPtr = pRecord->getDataPtr( pvField),
	uiElmLen = MAX_KEY_SIZ - uiKeyLen;

	if( RC_BAD( rc = KYCollateValue( &pKeyBuf [uiKeyLen], &uiElmLen,
									pExportPtr,
									pRecord->getDataLength( pvField), 
									FLM_TEXT_TYPE, uiElmLen,
									NULL, NULL,
									pDb->pFile->FileHdr.uiDefaultLanguage,
									FALSE, FALSE, FALSE, NULL)))
	{
		goto Exit;
	}

	uiKeyLen += uiElmLen;

Exit:

	*puiKeyLenRV = uiKeyLen;
	return( rc);
}

/**************************************************************************** 
Desc:	Checks to make sure a dictionary name has not already been used.
****************************************************************************/
FSTATIC RCODE DDCheckNameConflict(
	FDB *			pDb,
	LFILE *		pDictIxLFile,		/* Dictionary index to check in. */
	FlmRecord *	pNewRec,				/* Record whose name is to be checked. */
	FLMUINT		uiDrn,				/* DRN of new record. */
	FlmRecord *	pOldRec)				/* Old record, non-NULL indicates that this
											is a modifiy operation. */
{
	RCODE		rc = FERR_OK;
	BTSK		StackArray[ BH_MAX_LEVELS];
	BTSK_p	pStack;
	FLMBYTE	BtKeyBuf[ MAX_KEY_SIZ];
	FLMBYTE	IxKeyBuf[ MAX_KEY_SIZ];
	FLMUINT	uiKeyLen;
	void *	pvField;

	FSInitStackCache( &StackArray [0], BH_MAX_LEVELS);

	if (RC_BAD( rc = DDMakeDictIxKey( pDb, pNewRec, IxKeyBuf, &uiKeyLen)))
		goto Exit;
	StackArray[0].pKeyBuf = BtKeyBuf;
	pStack = StackArray;
	if (RC_BAD( rc = FSBtSearch( pDb, pDictIxLFile, &pStack,
						IxKeyBuf, uiKeyLen, 0L)))
		goto Exit;
	if (pStack->uiCmpStatus == BT_EQ_KEY)
	{
		FLMUINT		uiElmDoman;
		DIN_STATE	DinState;
		FLMUINT		uiFoundDrn;

		/*
		If this is an ADD (!pOldRec), or the record found
		is different than the one being updated, we have
		a problem.
		*/

		uiFoundDrn = FSRefFirst( pStack, &DinState, &uiElmDoman);
		if ((!pOldRec) || (uiFoundDrn != uiDrn))
		{
			pvField = pNewRec->root();
			pDb->Diag.uiInfoFlags |= FLM_DIAG_DRN;
			pDb->Diag.uiDrn = uiDrn;
			rc = (pNewRec->getFieldID( pvField) == FLM_RESERVED_TAG)
					? RC_SET( FERR_CANNOT_RESERVE_NAME)
					: RC_SET( FERR_DUPLICATE_DICT_NAME);
			goto Exit;
		}
	}
Exit:
	FSReleaseStackCache( StackArray, BH_MAX_LEVELS, FALSE);
	return( rc);
}


/**************************************************************************** 
Desc:	Checks to make sure a dictionary DRN has not already been used.
****************************************************************************/
FSTATIC RCODE DDCheckIDConflict(
	FDB *				pDb,
	LFILE *			pDictContLFile,	// Pointer to dictionary container LFILE.
	FLMUINT			uiDrn)				// DRN of record.
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pOldRec = NULL;

	// Read to see if there is an existing record.
	// NOTE: Deliberately not bringing into cache if not found.

	if( RC_BAD( rc = FSReadRecord( pDb, pDictContLFile, uiDrn,
							&pOldRec, NULL, NULL)))
	{
		if (rc == FERR_NOT_FOUND)
		{
			rc = FERR_OK;
		}
		else
		{
			goto Exit;
		}
	}

	if( pOldRec)
	{
		void * pvField = pOldRec->root();

		rc = ( pOldRec->getFieldID( pvField) == FLM_RESERVED_TAG)
			  ? RC_SET( FERR_ID_RESERVED)
			  : RC_SET( FERR_DUPLICATE_DICT_REC);
	}

Exit:
	if( pOldRec)
	{
		pOldRec->Release();
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Generate a key for an index record and add or delete it from
		the index.
****************************************************************************/
FSTATIC RCODE DDIxDictRecord(
	FDB *			pDb,
	LFILE *		pDictIxLFile,
	FLMUINT		uiDrn,
	FlmRecord *	pRecord,
	FLMUINT		uiFlags)
{
	RCODE		 	rc;
	union
	{
		FLMBYTE		KeyBuf [sizeof( KREF_ENTRY) + MAX_KEY_SIZ];
		KREF_ENTRY	KrefEntry;
	};
	FLMUINT			uiKeyLen;

	flmAssert( pDictIxLFile->uiLfNum > 0 && 
		pDictIxLFile->uiLfNum < FLM_UNREGISTERED_TAGS); // Sanity check
	KrefEntry.ui16IxNum = (FLMUINT16)pDictIxLFile->uiLfNum;
	KrefEntry.uiDrn = uiDrn;
	KrefEntry.uiFlags = uiFlags;
	KrefEntry.uiTrnsSeq = 1;

	/* Add or delete the key/reference. */

	if (RC_BAD( rc = DDMakeDictIxKey( pDb, pRecord,
												&KeyBuf [sizeof( KREF_ENTRY)], &uiKeyLen)))
	{
		goto Exit;
	}
	KrefEntry.ui16KeyLen = (FLMUINT16)uiKeyLen;

	rc = FSRefUpdate( pDb, pDictIxLFile, &KrefEntry);
Exit:
	return( rc);
}

