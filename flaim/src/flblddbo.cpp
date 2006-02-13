//-------------------------------------------------------------------------
// Desc:	Rebuild corrupted database.
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
// $Id: flblddbo.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE bldDetermineBlkSize(
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiMaxFileSize,
	FLMUINT *			puiBlkSizeRV,
	STATUS_HOOK			fnStatusFunc,
	REBUILD_INFO *		pCallbackData,
	void *				AppArg);

/****************************************************************************
Desc:	Extract create options from file header and log header pieces.
****************************************************************************/
void flmGetCreateOpts(
	FILE_HDR *		pFileHdr,
	FLMBYTE *		pucLogHdr,
	CREATE_OPTS *	pCreateOpts)
{
	f_memset( pCreateOpts, 0, sizeof( CREATE_OPTS));
	if (pFileHdr)
	{
		pCreateOpts->uiBlockSize = pFileHdr->uiBlockSize;
		pCreateOpts->uiVersionNum = pFileHdr->uiVersionNum;
		pCreateOpts->uiDefaultLanguage = pFileHdr->uiDefaultLanguage;
		pCreateOpts->uiAppMajorVer = pFileHdr->uiAppMajorVer;
		pCreateOpts->uiAppMinorVer = pFileHdr->uiAppMinorVer;
	}
	else
	{
		pCreateOpts->uiBlockSize = DEFAULT_BLKSIZ;
		pCreateOpts->uiVersionNum = FLM_CURRENT_VERSION_NUM;
		pCreateOpts->uiDefaultLanguage = DEFAULT_LANG;

		// uiAppMajorVer and uiAppMinorVer are already zero.
	}

	if (pucLogHdr)
	{
		pCreateOpts->uiMinRflFileSize =
			(FLMUINT)FB2UD( &pucLogHdr [LOG_RFL_MIN_FILE_SIZE]);
		pCreateOpts->uiMaxRflFileSize =
			(FLMUINT)FB2UD( &pucLogHdr [LOG_RFL_MAX_FILE_SIZE]);
		pCreateOpts->bKeepRflFiles =
			(FLMBOOL)((pucLogHdr [LOG_KEEP_RFL_FILES]) ? TRUE : FALSE);
		pCreateOpts->bLogAbortedTransToRfl =
			(FLMBOOL)((pucLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL]) ? TRUE : FALSE);
	}
	else
	{
		pCreateOpts->uiMinRflFileSize = DEFAULT_MIN_RFL_FILE_SIZE;
		pCreateOpts->uiMaxRflFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
		pCreateOpts->bKeepRflFiles = DEFAULT_KEEP_RFL_FILES_FLAG;
		pCreateOpts->bLogAbortedTransToRfl = DEFAULT_LOG_ABORTED_TRANS_FLAG;
	}
}

/*API~***********************************************************************
Desc: 	Rebuilds a damaged database.
Notes:	This routine performs the following actions:  1) A temporary database
		 	is created; 2) A copy of the source database is saved;  3) The source
			database is scanned.  Data records from all containers are extracted
			and stored in the temporary database.  4) When the rebuild is
			complete, the temporary database file is copied over the source
			database file.
*END************************************************************************/
FLMEXP RCODE FLMAPI FlmDbRebuild(
	const char *			pszSourceDbPath,
	const char *			pszSourceDataDir,
	const char *			pszDestDbPath,
	const char *			pszDestDataDir,
	const char *			pszDestRflDir,
	const char *			pszDictPath,
	CREATE_OPTS *			pCreateOpts,
	FLMUINT *				puiTotRecsRV,
	FLMUINT *				puiRecsRecovRV,
	STATUS_HOOK				fnStatusFunc,
	void *					pvStatusData)
{
	RCODE						rc = FERR_OK;
	FDB *						pDb = NULL;
	FFILE *					pFile;
	F_SuperFileHdl *		pSFileHdl = NULL;
	FLMBOOL					bFileLocked = FALSE;
	FLMBOOL					bWriteLocked = FALSE;
	REBUILD_STATE *		pRebuildState = NULL;
	HDR_INFO *				pHdrInfo;
	CREATE_OPTS *			pDefaultCreateOpts = NULL;
	FLMUINT					uiTransID;
	ServerLockObject *	pWriteLockObj = NULL;
	ServerLockObject *	pFileLockObj = NULL;
	FLMBOOL					bMutexLocked = FALSE;
	F_FileHdlImp *			pLockFileHdl = NULL;
	FLOCK_INFO				LockInfo;
	FLMUINT					uiFileNumber;
	FLMUINT					uiDbVersion = 0;
	FLMBOOL					bUsedFFile = FALSE;
	FLMBOOL					bBadHeader = FALSE;
	FlmECache *				pECacheMgr = NULL;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	// See if there is an FFILE structure for this file
	// May unlock and re-lock the global mutex.

	if( RC_BAD( rc = flmFindFile( pszSourceDbPath, pszSourceDataDir,
		&pFile)))
	{
		goto Exit;
	}

	// If we didn't find an FFILE structure, get an
	// exclusive lock on the file.

	if( !pFile)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Attempt to get an exclusive lock on the file.

		if( RC_BAD( rc = flmCreateLckFile( pszSourceDbPath, &pLockFileHdl)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = flmCheckFFileState( pFile)))
		{
			goto Exit;
		}

		// The call to flmVerifyFileUse will wait if the file is in
		// the process of being opened by another thread.

		if (RC_BAD( rc = flmVerifyFileUse( gv_FlmSysData.hShareMutex, &pFile)))
		{
			goto Exit;
		}

		// Increment the use count on the FFILE so it will not
		// disappear while we are copying the file.

		if (++pFile->uiUseCount == 1)
		{
			flmUnlinkFileFromNUList( pFile);
		}
		bUsedFFile = TRUE;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// See if the thread already has a file lock.  If so, there
		// is no need to obtain another.  However, we also want to
		// make sure there is no write lock.  If there is,
		// we cannot do the rebuild right now.

		pFile->pFileLockObj->GetLockInfo( (FLMINT)0, &LockInfo);
		if (LockInfo.eCurrLockType == FLM_LOCK_EXCLUSIVE &&
			 LockInfo.uiThreadId == f_threadId())
		{

			// See if there is already a transaction going.

			pFile->pWriteLockObj->GetLockInfo( (FLMINT)0, &LockInfo);
			if ((LockInfo.eCurrLockType == FLM_LOCK_EXCLUSIVE) &&
				 (LockInfo.uiThreadId == f_threadId()))
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}
		}
		else
		{
			pFileLockObj = pFile->pFileLockObj;
			pFileLockObj->AddRef();
			if (RC_BAD( rc = pFileLockObj->Lock( TRUE, NULL, FALSE, TRUE,
									FLM_NO_TIMEOUT, 0)))
			{
				goto Exit;
			}
			bFileLocked = TRUE;
		}

		// Lock the write object to eliminate contention with
		// the checkpoint thread.

		pWriteLockObj = pFile->pWriteLockObj;
		pWriteLockObj->AddRef();

		// Only contention here is with the checkpoint thread.
		// Wait forever for the checkpoint thread to give
		// up the lock.

		if (RC_BAD( rc = dbWriteLock( pFile)))
		{
			goto Exit;
		}
		bWriteLocked = TRUE;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( REBUILD_STATE), &pRebuildState)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( HDR_INFO), &pRebuildState->pHdrInfo)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( STATE_INFO), &pRebuildState->pStateInfo)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		sizeof( CREATE_OPTS), &pDefaultCreateOpts)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		LOG_HEADER_SIZE, &pRebuildState->pLogHdr)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc(
		MAX_KEY_SIZ, &pRebuildState->pKeyBuffer)))
	{
		goto Exit;
	}

	pRebuildState->AppArg = pvStatusData;
	pRebuildState->fnStatusFunc = fnStatusFunc;
	pHdrInfo = pRebuildState->pHdrInfo;

	/* Open the corrupted database. */

	if ((pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pSFileHdl->Setup( NULL, pszSourceDbPath,
							pszSourceDataDir)))
	{
		goto Exit;
	}

	pRebuildState->pSFileHdl = pSFileHdl;

	/*
	Check the header information to see if we were in the middle
	of a previous copy.
	*/

	if (RC_OK( rc = flmGetHdrInfo( pSFileHdl,
									  &pHdrInfo->FileHdr,
									  &pHdrInfo->LogHdr,
									  pRebuildState->pLogHdr)))
	{

		if (!pCreateOpts)
		{
			flmGetCreateOpts( &pHdrInfo->FileHdr,
										 pRebuildState->pLogHdr, pDefaultCreateOpts);
			pCreateOpts = pDefaultCreateOpts;
		}
		rc = FERR_OK;
		uiDbVersion = pHdrInfo->FileHdr.uiVersionNum;
		pRebuildState->uiMaxFileSize = flmGetMaxFileSize( uiDbVersion,
													pRebuildState->pLogHdr);
	}
	else if ((rc == FERR_BLOCK_CHECKSUM) ||
				(rc == FERR_INCOMPLETE_LOG) ||
				(rc == FERR_DATA_ERROR) ||
				((rc == FERR_UNSUPPORTED_VERSION) &&
				  (pHdrInfo->FileHdr.uiVersionNum == 0)))
	{
		if ((rc == FERR_BLOCK_CHECKSUM) ||
			 (rc == FERR_DATA_ERROR))
		{
			bBadHeader = TRUE;
		}
		rc = FERR_OK;
		if (!pCreateOpts)
		{
			flmGetCreateOpts( &pHdrInfo->FileHdr,
										 pRebuildState->pLogHdr, pDefaultCreateOpts);
			pCreateOpts = pDefaultCreateOpts;
		}
		uiDbVersion = pHdrInfo->FileHdr.uiVersionNum;
		pRebuildState->uiMaxFileSize = flmGetMaxFileSize( uiDbVersion,
													pRebuildState->pLogHdr);
	}
	else if (rc == FERR_UNSUPPORTED_VERSION || rc == FERR_NEWER_FLAIM)
	{
		goto Exit;
	}
	else if ((rc == FERR_NOT_FLAIM) ||
				(!VALID_BLOCK_SIZE( pHdrInfo->FileHdr.uiBlockSize)))
	{
		FLMUINT	uiSaveBlockSize;
		FLMUINT	uiCalcBlockSize = 0;
		FLMBYTE	ucFileHdrBuf[ FLM_FILE_HEADER_SIZE];

		uiDbVersion = (FLMUINT)((rc != FERR_NOT_FLAIM)
										? pHdrInfo->FileHdr.uiVersionNum
										: FLM_CURRENT_VERSION_NUM);
		pSFileHdl->SetDbVersion( uiDbVersion);
		pRebuildState->uiMaxFileSize = flmGetMaxFileSize( uiDbVersion,
													pRebuildState->pLogHdr);
		if (!pCreateOpts)
		{
			if (rc != FERR_NOT_FLAIM)
			{
				flmGetCreateOpts( &pHdrInfo->FileHdr,
										 pRebuildState->pLogHdr, pDefaultCreateOpts);
			}
			else
			{
				flmGetCreateOpts( NULL, NULL, pDefaultCreateOpts);
			}

			// Set block size to zero, so we will always take the calculated
			// block size below.

			pDefaultCreateOpts->uiBlockSize = 0;
			pCreateOpts = pDefaultCreateOpts;
		}

		/* Try to determine the correct block size. */

		if (RC_BAD( rc = bldDetermineBlkSize( pSFileHdl,
											pRebuildState->uiMaxFileSize,
											&uiCalcBlockSize,
											fnStatusFunc, &pRebuildState->CallbackData,
											pRebuildState->AppArg)))
		{
			goto Exit;
		}

		uiSaveBlockSize = pCreateOpts->uiBlockSize;
		pCreateOpts->uiBlockSize = uiCalcBlockSize;

		// Initialize pHdrInfo->FileHdr to useable values.

		flmInitFileHdrInfo( pCreateOpts, &pHdrInfo->FileHdr, ucFileHdrBuf);

		/*
		Only use the passed-in block size (uiSaveBlockSize) if it
		was non-zero.
		*/

		if (uiSaveBlockSize)
			pCreateOpts->uiBlockSize = uiSaveBlockSize;
	}
	else
	{
		goto Exit;
	}

	// Calculate the file size.

	pSFileHdl->SetDbVersion( uiDbVersion);
	pRebuildState->CallbackData.ui64DatabaseSize = 0;
	for (uiFileNumber = 1;;uiFileNumber++)
	{
		FLMUINT	uiTmpSize;

		if (RC_BAD( pSFileHdl->GetFileSize( uiFileNumber, &uiTmpSize)))
		{
			break;
		}
		pRebuildState->CallbackData.ui64DatabaseSize += (FLMUINT64)uiTmpSize;
	}

	// Delete the destination database in case it already exists.

	if (RC_BAD( rc = FlmDbRemove( pszDestDbPath, pszDestDataDir,
								pszDestRflDir, TRUE)))
	{
		if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
		{
			rc = FERR_OK;
		}
		else
		{
			goto Exit;
		}
	}

	/*
	If no block size has been specified or determined yet, use what we
	read from the file header.
	*/

	if (!pCreateOpts->uiBlockSize)
		pCreateOpts->uiBlockSize = pHdrInfo->FileHdr.uiBlockSize;

	pSFileHdl->SetDbVersion( pHdrInfo->FileHdr.uiVersionNum);
	pSFileHdl->SetBlockSize( pHdrInfo->FileHdr.uiBlockSize);

	/*
	Set the ECache manger into the super file handle
	*/

	if( pFile && pFile->pECacheMgr)
	{
		pSFileHdl->setECacheMgr( pFile->pECacheMgr);
	}
	else if( gv_FlmSysData.bOkToUseESM)
	{
		if( (pECacheMgr = f_new FlmECache) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( !pECacheMgr->setupECache( pHdrInfo->FileHdr.uiBlockSize,
			pRebuildState->uiMaxFileSize))
		{
			pECacheMgr->Release();
			pECacheMgr = NULL;
		}
		else
		{
			pSFileHdl->setECacheMgr( pECacheMgr);
		}
	}

	/*
	When creating the new file, set the transaction ID to one greater than it
	is in the corrupt file.  However, don't let it get greater than about
	2 billion - want to leave room for 2 billion transactions in case they
	were corrupted somehow in our old file.
	*/

	uiTransID =
		((FLMUINT)FB2UD( &pRebuildState->pLogHdr [LOG_CURR_TRANS_ID]) + 1) & 0x7FFFFFFF;

	if (RC_BAD( rc = flmCreateNewFile( pszDestDbPath, pszDestDataDir,
					pszDestRflDir,
					pszDictPath, NULL,
					pCreateOpts, uiTransID, (FDB_p *)&pRebuildState->hDb,
					pRebuildState)))
	{
		goto Exit;
	}
	pDb = (FDB *)pRebuildState->hDb;

	/* Rebuild the database */

	if (RC_BAD( rc = flmDbRebuildFile( pRebuildState, bBadHeader)))
	{
		goto Exit;
	}

Exit:

	/* Close the temporary database, if it is still open. */

	if (pDb)
	{
		FFILE *	pTmpFile;
		FFILE *	pTmpFile1;

		// Get the FFILE pointer for the temporary file before closing it.

		pTmpFile = pDb->pFile;

		(void)FlmDbClose( (HFDB *)&pDb);

		// Force temporary FFILE structure to be cleaned up, if it
		// isn't already gone.  The following code searches for the
		// temporary file in the not-used list.  If it finds it,
		// it will unlink it.

		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		pTmpFile1 = gv_FlmSysData.pLrnuFile;
		while (pTmpFile1)
		{
			if (pTmpFile1 == pTmpFile)
			{
				flmFreeFile( pTmpFile);
				break;
			}
			pTmpFile1 = pTmpFile1->pNextNUFile;
		}
	}

	if (bUsedFFile)
	{
		if (!bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}
		if (!(--pFile->uiUseCount))
		{
			flmLinkFileToNUList( pFile);
		}
	}
	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	/* Unlock the file, if it is locked. */

	if (bWriteLocked)
	{
		dbWriteUnlock( pFile);
		bWriteLocked = FALSE;
	}

	if (bFileLocked)
	{
		RCODE	rc3;

		if (RC_BAD( rc3 = pFileLockObj->Unlock( TRUE, NULL)))
		{
			if (RC_OK( rc))
			{
				rc = rc3;
			}
		}
		bFileLocked = FALSE;
	}

	if( pSFileHdl)
	{
		pSFileHdl->Release();
	}

	if( pECacheMgr)
	{
		pECacheMgr->Release();
		pECacheMgr = NULL;
	}

	if (pWriteLockObj)
	{
		pWriteLockObj->Release();
		pWriteLockObj = NULL;
	}
	if (pFileLockObj)
	{
		pFileLockObj->Release();
		pFileLockObj = NULL;
	}

	if (pLockFileHdl)
	{
		(void)pLockFileHdl->Close();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	if( pDefaultCreateOpts)
	{
		f_free( &pDefaultCreateOpts);
	}

	if (pRebuildState)
	{
		if( puiTotRecsRV)
		{
			*puiTotRecsRV = pRebuildState->CallbackData.uiTotRecs;
		}

		if( puiRecsRecovRV)
		{
			*puiRecsRecovRV = pRebuildState->CallbackData.uiRecsRecov;
		}

		if ((pRebuildState->pStateInfo) &&
			 (pRebuildState->pStateInfo->pRecord))
		{
			pRebuildState->pStateInfo->pRecord->Release();

		}

		if (pRebuildState->pRecord)
		{
			pRebuildState->pRecord->Release();
			pRebuildState->pRecord = NULL;
		}

		if( pRebuildState->pHdrInfo)
		{
			f_free( &pRebuildState->pHdrInfo);
		}

		if( pRebuildState->pStateInfo)
		{
			f_free( &pRebuildState->pStateInfo);
		}

		if( pRebuildState->pLogHdr)
		{
			f_free( &pRebuildState->pLogHdr);
		}

		if( pRebuildState->pKeyBuffer)
		{
			f_free( &pRebuildState->pKeyBuffer);
		}

		f_free( &pRebuildState);
	}
	else
	{
		if( puiTotRecsRV)
		{
			*puiTotRecsRV = 0;
		}

		if( puiRecsRecovRV)
		{
			*puiRecsRecovRV = 0;
		}
	}

	return( rc);
}


/***************************************************************************
Desc:	This routine reads through a database and makes a best guess as to 
		the true block size of the database.
*****************************************************************************/
FSTATIC RCODE bldDetermineBlkSize(
	F_SuperFileHdl *	pSFileHdl,		// Super file handle for database.
	FLMUINT				uiMaxFileSize,
	FLMUINT *			puiBlkSizeRV,	// Calculated block size is returned here.
	STATUS_HOOK			fnStatusFunc,	// Callback function.
	REBUILD_INFO *		pCallbackData,	// Callback structure.
	void *				AppArg)			// User data for callback.
{
	RCODE				rc = FERR_OK;
	FLMBYTE			ucBlkHeader [BH_OVHD];
	FLMUINT			uiBytesRead;
	FLMUINT			uiBlkAddress;
	FLMUINT			uiFileNumber = 0;
	FLMUINT			uiOffset = 0;
	FLMUINT			uiCount4K = 0;
	FLMUINT			uiCount8K = 0;
	FLMUINT64		ui64BytesDone = 0;
	F_FileHdlImp *	pFileHdl = NULL;

	/* Start from byte offset 0 in the first file. */

	pCallbackData->iDoingFlag = REBUILD_GET_BLK_SIZ;
	pCallbackData->bStartFlag = TRUE;
	for (;;)
	{
		if (uiOffset >= uiMaxFileSize || !uiFileNumber)
		{
			uiOffset = 0;
			uiFileNumber++;
			if (RC_BAD( rc = pSFileHdl->GetFileHdl( 
				uiFileNumber, FALSE, &pFileHdl)))
			{
				if (rc == FERR_IO_PATH_NOT_FOUND)
				{
					rc = RC_SET( FERR_IO_END_OF_FILE);
					break;
				}
				goto Exit;
			}
		}

		if ((RC_OK(rc = pFileHdl->Read( uiOffset, BH_OVHD, ucBlkHeader,
									  &uiBytesRead))) ||
			 (rc == FERR_IO_END_OF_FILE))
		{
			if (RC_OK( rc))
			{
				ui64BytesDone += (FLMUINT64)MIN_BLOCK_SIZE;
			}
			else
			{
				ui64BytesDone += (FLMUINT64)uiBytesRead;
			}
			uiBlkAddress = GET_BH_ADDR( ucBlkHeader);
			if ((uiBytesRead == BH_OVHD) &&
				 (FSGetFileOffset( uiBlkAddress) == uiOffset))
			{
				if (uiOffset % 4096 == 0)
					uiCount4K++;
				if (uiOffset % 8192 == 0)
					uiCount8K++;
			}
			if (rc != FERR_OK || uiBytesRead < BH_OVHD)
			{

				// Even if the file is not full size, set offset to
				// the maximum file offset so we will attempt to go
				// to the next file at the top of this loop.  If that
				// fails, we will assume we truly are at EOF.

				uiOffset = uiMaxFileSize;
			}
			else
			{
				uiOffset += MIN_BLOCK_SIZE;
			}

			/* Call the callback function to report copy progress. */

			if (fnStatusFunc != NULL)
			{
				pCallbackData->ui64BytesExamined = ui64BytesDone;
				if (RC_BAD( rc = (*fnStatusFunc)( FLM_REBUILD_STATUS,
											(void *)pCallbackData,
											(void *)0,
											AppArg)))
				{
					goto Exit;
				}
				pCallbackData->bStartFlag = FALSE;
			}

			f_yieldCPU();
		}
		else
		{
			goto Exit;
		}
	}
	if (rc == FERR_IO_END_OF_FILE)
		rc = FERR_OK;

	// If our count of 4K blocks is greater than 66% of the number
	// of 4K blocks that would fit in the database, we will use
	// a 4K block size.  Otherwise, we will use an 8K block size.

	if (uiCount4K > 
		(FLMUINT)(((ui64BytesDone / 
			(FLMUINT64)4096) * (FLMUINT64)66) / (FLMUINT64)100))
	{
		*puiBlkSizeRV = 4096;
	}
	else
	{
		*puiBlkSizeRV = 8192;
	}
Exit:
	return( rc);
}
