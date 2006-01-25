//-------------------------------------------------------------------------
// Desc:	Routines for database configuration.
// Tabs:	3
//
//		Copyright (c) 1996-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fdbcnfig.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE flmDbGetSizes(
	FDB *			pDb,
	FLMUINT64 *	pui64DbFileSize,
	FLMUINT64 *	pui64RollbackFileSize,
	FLMUINT64 *	pui64RflFileSize);

void flmGetCPInfo(
	void *					pFilePtr,
	CHECKPOINT_INFO *		pCheckpointInfo);

/*******************************************************************************
Desc:	 Sets indexing callback function
*******************************************************************************/
void FlmSetIndexingCallback(
	HFDB						hDb,
	IX_CALLBACK				fnIxCallback,
	void *					pvAppData)
{
	((FDB_p)hDb)->fnIxCallback = fnIxCallback;
	((FDB_p)hDb)->IxCallbackData = pvAppData;
}

/*******************************************************************************
Desc:	 Returns indexing callback function
*******************************************************************************/
void FlmGetIndexingCallback(
	HFDB				hDb,
	IX_CALLBACK *	pfnIxCallback,
	void **			ppvAppData)
{
	if (pfnIxCallback)
	{
		*pfnIxCallback = ((FDB_p)hDb)->fnIxCallback;
	}

	if (ppvAppData)
	{
		*ppvAppData = ((FDB_p)hDb)->IxCallbackData;
	}
}

/*******************************************************************************
Desc : Configures a callback function which allows validation of records
		 before they are returned to the user or committed to the
		 database.
Notes: This function stores a pointer to a callback function which is
		 called whenever a record is added, deleted, modified or
		 retrieved.  This allows an application to validate record operations
		 before they are committed to the database (update operations)
		 or before records are returned to the application (read operations).
		 By default, no record validation is performed by FLAIM.
*******************************************************************************/
void FlmSetRecValidatorHook(
	HFDB						hDb,
	REC_VALIDATOR_HOOK   fnRecValidatorHook,
	void *					pvAppData)
{
	((FDB_p)hDb)->fnRecValidator = fnRecValidatorHook;
	((FDB_p)hDb)->RecValData = pvAppData;
}

/*******************************************************************************
Desc : Returns to the user the sessions current Rec Validator Hook values.
*******************************************************************************/
void FlmGetRecValidatorHook(
	HFDB						hDb,
	REC_VALIDATOR_HOOK * pfnRecValidatorHook, // [out] RecValidator func pointer
	void **					ppvAppData)				// [out] application data
{
	if (pfnRecValidatorHook)
	{
		*pfnRecValidatorHook = ((FDB_p)hDb)->fnRecValidator;
	}

	if (ppvAppData)
	{
		*ppvAppData = ((FDB_p)hDb)->RecValData;
	}
}

/*******************************************************************************
Desc : Configures a callback function which is called to return general
		 purpose information.
*******************************************************************************/
void FlmSetStatusHook(
	HFDB				hDb,
	STATUS_HOOK    fnStatusHook,
	void *			pvAppData)
{
	((FDB_p)hDb)->fnStatus = fnStatusHook;
	((FDB_p)hDb)->StatusData = pvAppData;
}

/*******************************************************************************
Desc : Returns to the user the session's current status hook values.
*******************************************************************************/
void FlmGetStatusHook(
	HFDB				hDb,
	STATUS_HOOK *	pfnStatusHook,
	void **			ppvAppData)
{
	if (pfnStatusHook)
	{
		*pfnStatusHook = ((FDB_p)hDb)->fnStatus;
	}

	if (ppvAppData)
	{
		*ppvAppData = ((FDB_p)hDb)->StatusData;
	}
}

/*******************************************************************************
Desc:	Allows an application to configure various options for a database.
*******************************************************************************/
RCODE 
		// FERR_NOT_IMPLEMENTED - Invalid eConfigType value
	FlmDbConfig(
		HFDB				hDb,
			// [IN] Handle to a database.
		eDbConfigType	eConfigType,
			// [IN] Database option to configure.
		void *			Value1,
			// [IN] The type and domain of Value1 are determined by eConfigType.
		void *		   Value2
			// [IN] The type and domain of Value2 are determined by eConfigType.
	)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB_p)hDb;
	FFILE *		pFile = pDb->pFile;
	FLMBOOL		bDbInitialized = FALSE;
	FLMBOOL		bStartedTrans = FALSE;
	FLMBOOL		bDbLocked = FALSE;

	/*
	Handle client/server requests
	*/

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bDbInitialized = TRUE;

		CS_CONTEXT_p		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_CONFIG)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TYPE, (FLMUINT)eConfigType)))
		{
			goto Transmission_Error;
		}

		switch( eConfigType)
		{
			case FDB_SET_APP_VERSION:
			case FDB_RFL_KEEP_FILES:
			case FDB_KEEP_ABORTED_TRANS_IN_RFL:
			case FDB_AUTO_TURN_OFF_KEEP_RFL:
			case FDB_SET_APP_DATA:
				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER2,
					(FLMUINT)Value1)))
				{
					goto Transmission_Error;
				}
				break;

			case FDB_RFL_DIR:
			{
				FLMUNICODE *	puzRflDir;

				if( RC_BAD( rc = fcsConvertNativeToUnicode( 
					Wire.getPool(), (const char *)Value1, &puzRflDir)))
				{
					goto Transmission_Error;
				}

				if( RC_BAD( rc = Wire.sendString( 
					WIRE_VALUE_FILE_PATH, puzRflDir)))
				{
					goto Transmission_Error;
				}
				break;
			}

			case FDB_RFL_FILE_LIMITS:
				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1,
					(FLMUINT)Value1)))
				{
					goto Transmission_Error;
				}

				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER2,
					(FLMUINT)Value2)))
				{
					goto Transmission_Error;
				}
				break;

			case FDB_FILE_EXTEND_SIZE:
				if( RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_NUMBER1,
					(FLMUINT)Value1)))
				{
					goto Transmission_Error;
				}
				break;

			case FDB_RFL_ROLL_TO_NEXT_FILE:
				break;

			default:
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		/* Read the response. */
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	/*
	Process the local (non-C/S) request
	*/

	switch( eConfigType)
	{
		case FDB_RFL_KEEP_FILES:
		{
			FLMBOOL	bKeepFiles = (FLMBOOL)(Value1 ? TRUE : FALSE);

			// This operation is not legal for pre 4.3 databases.

			if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// Make sure we don't have a transaction going

			if( pDb->uiTransType != FLM_NO_TRANS)
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}

			// Make sure there is no active backup running

			f_mutexLock( gv_FlmSysData.hShareMutex);
			if( pDb->pFile->bBackupActive)
			{
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				rc = RC_SET( FERR_BACKUP_ACTIVE);
				goto Exit;
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			// Need to lock the database but not start a transaction yet.

			if( !(pDb->uiFlags & (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED)))
			{
				if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0,
											FLM_NO_TIMEOUT)))
				{
					goto Exit;
				}
				bDbLocked = TRUE;
			}

			// If we aren't changing the keep flag, jump to exit without doing
			// anything.

			if ((bKeepFiles &&
				  pDb->pFile->ucLastCommittedLogHdr [LOG_KEEP_RFL_FILES]) ||
				 (!bKeepFiles &&
				  !pDb->pFile->ucLastCommittedLogHdr [LOG_KEEP_RFL_FILES]))
			{
				goto Exit;	// Will return FERR_OK;
			}

			// Force a checkpoint and roll to the next RFL file numbers.
			// When changing from keep to no-keep or vice versa, we need to
			// go to a new RFL file so that the new RFL file gets new
			// serial numbers and a new keep or no-keep flag.

			if (RC_BAD( rc = FlmDbCheckpoint( hDb, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}

			f_memcpy( pDb->pFile->ucUncommittedLogHdr,
						 pDb->pFile->ucLastCommittedLogHdr,
						 LOG_HEADER_SIZE);
			pDb->pFile->ucUncommittedLogHdr [LOG_KEEP_RFL_FILES] =
				(FLMBYTE)((bKeepFiles) ? (FLMBYTE)1 : (FLMBYTE)0);

			// Force a new RFL file - this will also write out the entire
			// log header - including the changes we made above.

			if (RC_BAD( rc = pDb->pFile->pRfl->finishCurrFile( pDb, TRUE)))
			{
				goto Exit;
			}
			break;
		}

		case FDB_RFL_DIR:
		{
			const char *	pszNewRflDir = (const char *)Value1;

			// This operation is not legal for pre 4.3 databases.

			if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// Make sure we don't have a transaction going

			if( pDb->uiTransType != FLM_NO_TRANS)
			{
				rc = RC_SET( FERR_TRANS_ACTIVE);
				goto Exit;
			}

			// Make sure there is no active backup running

			f_mutexLock( gv_FlmSysData.hShareMutex);
			if( pDb->pFile->bBackupActive)
			{
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				rc = RC_SET( FERR_BACKUP_ACTIVE);
				goto Exit;
			}
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			// Make sure the path exists and that it is a directory
			// rather than a file.

			if (pszNewRflDir && *pszNewRflDir)
			{
				if( !gv_FlmSysData.pFileSystem->IsDir( pszNewRflDir))
				{
					rc = RC_SET( FERR_IO_INVALID_PATH);
					goto Exit;
				}
			}

			// Need to lock the database because we can't change the RFL
			// directory until after the checkpoint has completed.  The
			// checkpoint code will unlock the transaction, but not the
			// file if we have an explicit lock.  We need to do this to
			// prevent another transaction from beginning before we have
			// changed the RFL directory.

			if( !(pDb->uiFlags & (FDB_HAS_FILE_LOCK | FDB_FILE_LOCK_SHARED)))
			{
				if( RC_BAD( rc = FlmDbLock( hDb, FLM_LOCK_EXCLUSIVE, 0,
											FLM_NO_TIMEOUT)))
				{
					goto Exit;
				}
				bDbLocked = TRUE;
			}

			// Force a checkpoint and roll to the next RFL file numbers.  Both
			// of these steps are necessary to ensure that we won't have to do
			// any recovery using the current RFL file - because we do not
			// move the current RFL file to the new directory.  Forcing the
			// checkpoint ensures that we have no transactions that will need
			// to be recovered if we were to crash.  Rolling the RFL file number
			// ensures that no more transactions will be logged to the current
			// RFL file.

			if (RC_BAD( rc = FlmDbCheckpoint( hDb, FLM_NO_TIMEOUT)))
			{
				goto Exit;
			}

			// Force a new RFL file.

			if (RC_BAD( rc = pDb->pFile->pRfl->finishCurrFile( pDb, FALSE)))
			{
				goto Exit;
			}

			// Set the RFL directory to the new value now that we have
			// finished the checkpoint and rolled to the next RFL file.

			f_mutexLock( gv_FlmSysData.hShareMutex);
			rc = pFile->pRfl->setRflDir( pszNewRflDir);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
			break;
		}

		case FDB_RFL_FILE_LIMITS:
		{
			FLMUINT	uiMinRflSize = (FLMUINT)Value1;
			FLMUINT	uiMaxRflSize = (FLMUINT)Value2;

			// Make sure the limits are valid.

			if (pDb->pFile->FileHdr.uiVersionNum >= FLM_VER_4_3)
			{

				// Maximum must be enough to hold at least one packet plus
				// the RFL header.  Minimum must not be greater than the
				// maximum.  NOTE: Minimum and maximum are allowed to be
				// equal, but in all cases, maximum takes precedence over
				// minimum.  We will first NOT exceed the maximum.  Then,
				// if possible, we will go above the minimum.

				if (uiMaxRflSize < RFL_MAX_PACKET_SIZE + 512)
				{
					uiMaxRflSize = RFL_MAX_PACKET_SIZE + 512;
				}
				if (uiMaxRflSize > gv_FlmSysData.uiMaxFileSize)
				{
					uiMaxRflSize = gv_FlmSysData.uiMaxFileSize;
				}
				if (uiMinRflSize > uiMaxRflSize)
				{
					uiMinRflSize = uiMaxRflSize;
				}
			}

			// Start an update transaction.  Must not already be one going.

			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
											  0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS,
											  &bStartedTrans)))
			{
				goto Exit;
			}

			// Commit the transaction.

			UD2FBA( (FLMUINT32)uiMinRflSize,
				&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
			if (pDb->pFile->FileHdr.uiVersionNum >= FLM_VER_4_3)
			{
				UD2FBA( (FLMUINT32)uiMaxRflSize,
					&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MAX_FILE_SIZE]);
			}
			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
			{
				goto Exit;
			}
			bStartedTrans = FALSE;
			break;
		}

		case FDB_RFL_ROLL_TO_NEXT_FILE:

			// This operation is not legal for pre 4.3 databases.

			if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			/*
			NOTE: finishCurrFile will not roll to the next file if the current
			file has not been created.
			*/

			if (RC_BAD( rc = pDb->pFile->pRfl->finishCurrFile( pDb, FALSE)))
			{
				goto Exit;
			}
			break;

		case FDB_SET_APP_VERSION:
		{
			FLMUINT		uiOldMajorVer;
			FLMUINT		uiOldMinorVer;

			/*
			Start an update transaction.
			*/

			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
				0, 5 | FLM_AUTO_TRANS, &bStartedTrans)))
			{
				goto Exit;
			}

			/*
			Set the version.
			*/

			f_mutexLock( gv_FlmSysData.hShareMutex);
			uiOldMajorVer = pDb->pFile->FileHdr.uiAppMajorVer;
			pDb->pFile->FileHdr.uiAppMajorVer = (FLMUINT)Value1;
			uiOldMinorVer = pDb->pFile->FileHdr.uiAppMinorVer;
			pDb->pFile->FileHdr.uiAppMinorVer = (FLMUINT)Value2;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);

			/*
			Commit the transaction.  NOTE: This will always cause
			us to write out the application version numbers, because
			we always write out the prefix - first 512 bytes.
			*/

			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
			{
				/*
				Undo the changes made above
				*/

				f_mutexLock( gv_FlmSysData.hShareMutex);
				pDb->pFile->FileHdr.uiAppMajorVer = uiOldMajorVer;
				pDb->pFile->FileHdr.uiAppMinorVer = uiOldMinorVer;
				f_mutexUnlock( gv_FlmSysData.hShareMutex);
				goto Exit;
			}
			bStartedTrans = FALSE;
			break;
		}

		case FDB_KEEP_ABORTED_TRANS_IN_RFL:
		case FDB_AUTO_TURN_OFF_KEEP_RFL:

			// These operations are not legal for pre 4.3 databases.

			if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
			{
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			// Start an update transaction.  Must not already be one going.

			bDbInitialized = TRUE;
			if (RC_BAD( rc = fdbInit( pDb, FLM_UPDATE_TRANS,
											  0, FLM_NO_TIMEOUT | FLM_AUTO_TRANS,
											  &bStartedTrans)))
			{
				goto Exit;
			}

			// Change the uncommitted log header

			if (eConfigType == FDB_KEEP_ABORTED_TRANS_IN_RFL)
			{
				pDb->pFile->ucUncommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL] =
									(FLMBYTE)(Value1
												 ? (FLMBYTE)1
												 : (FLMBYTE)0);
			}
			else
			{
				pDb->pFile->ucUncommittedLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL] =
									(FLMBYTE)(Value1
												 ? (FLMBYTE)1
												 : (FLMBYTE)0);
			}

			// Commit the transaction.

			if (RC_BAD( rc = flmCommitDbTrans( pDb, 0, FALSE)))
			{
				goto Exit;
			}
			bStartedTrans = FALSE;
			break;

		case FDB_FILE_EXTEND_SIZE:
			pDb->pFile->uiFileExtendSize = (FLMUINT)Value1;
			break;

		case FDB_SET_APP_DATA:
			pDb->pvAppData = Value1;
			break;

		case FDB_SET_COMMIT_CALLBACK:
			pDb->fnCommit = (COMMIT_FUNC)((FLMUINT)Value1);
			pDb->pvCommitData = Value2;
			break;

		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		flmAbortDbTrans( pDb);
	}

	if( bDbLocked)
	{
		FlmDbUnlock( hDb);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns database, rollback, and rollforward sizes.  We are guaranteed
		to be inside an update transaction at this point.
****************************************************************************/
FSTATIC RCODE flmDbGetSizes(
	FDB *			pDb,
	FLMUINT64 *	pui64DbFileSize,
	FLMUINT64 *	pui64RollbackFileSize,
	FLMUINT64 *	pui64RflFileSize
	)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMUINT			uiDbVersion = pFile->FileHdr.uiVersionNum;
	FLMUINT			uiEndAddress;
	FLMUINT			uiLastFileNumber;
	FLMUINT			uiLastFileSize;
	char				szTmpName[ F_PATH_MAX_SIZE];
	char				szRflDir[ F_PATH_MAX_SIZE];
	char				szPrefix[ F_FILENAME_SIZE];
	F_FileHdlImp *	pFileHdl = NULL;
	F_DirHdl *		pDirHdl = NULL;

	// Better be inside an update transaction at this point.

	flmAssert( pDb->uiTransType == FLM_UPDATE_TRANS);

	// See if they want the database files sizes.

	if (pui64DbFileSize)
	{
		uiEndAddress = pDb->LogHdr.uiLogicalEOF;
		uiLastFileNumber = FSGetFileNumber( uiEndAddress);

		// Last file number better be in the proper range.

		flmAssert( uiLastFileNumber >= 1 &&
					  uiLastFileNumber <= MAX_DATA_BLOCK_FILE_NUMBER( uiDbVersion));

		// Get the actual size of the last file.

		if (RC_BAD( rc = pDb->pSFileHdl->GetFileSize( uiLastFileNumber,
										&uiLastFileSize)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND ||
				 rc == FERR_IO_INVALID_PATH)
			{
				if (uiLastFileNumber > 1)
				{
					rc = FERR_OK;
					uiLastFileSize = 0;
				}
				else
				{

					// Should always be a data file #1

					flmAssert( 0);
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// One of two situations exists with respect to the last
		// file: 1) it has not been fully written out yet (blocks
		// are still cached, or 2) it has been written out and
		// extended beyond what the logical EOF shows.  We want
		// the larger of these two possibilities.

		if (FSGetFileOffset( uiEndAddress) > uiLastFileSize)
		{
			uiLastFileSize = FSGetFileOffset( uiEndAddress);
		}

		if (uiLastFileNumber == 1)
		{

			// Only one file - use last file's size.

			*pui64DbFileSize = (FLMUINT64)uiLastFileSize;
		}
		else
		{

			// Size is the sum of full size for all files except the last one,
			// plus the calculated (or actual) size of the last one.

			(*pui64DbFileSize) = (FLMUINT64)(uiLastFileNumber - 1) *
											 (FLMUINT64)pFile->uiMaxFileSize +
											 (FLMUINT64)uiLastFileSize;
		}
	}

	// See if they want the rollback files sizes.

	if (pui64RollbackFileSize)
	{
		uiEndAddress = (FLMUINT)FB2UD(
								&pFile->ucUncommittedLogHdr [LOG_ROLLBACK_EOF]);
		uiLastFileNumber = FSGetFileNumber( uiEndAddress);

		// Last file number better be in the proper range.

		flmAssert( !uiLastFileNumber ||
					  (uiLastFileNumber >=
							FIRST_LOG_BLOCK_FILE_NUMBER( uiDbVersion) &&
					   uiLastFileNumber <=
							MAX_LOG_BLOCK_FILE_NUMBER( uiDbVersion)));

		// Get the size of the last file number.

		if (RC_BAD( rc = pDb->pSFileHdl->GetFileSize( uiLastFileNumber,
										&uiLastFileSize)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND ||
				 rc == FERR_IO_INVALID_PATH)
			{
				if (uiLastFileNumber)
				{
					rc = FERR_OK;
					uiLastFileSize = 0;
				}
				else
				{

					// Should always have rollback file #0

					flmAssert( 0);
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}

		// If the EOF offset for the last file is greater than the
		// actual file size, use it instead of the actual file size.

		if (FSGetFileOffset( uiEndAddress) > uiLastFileSize)
		{
			uiLastFileSize = FSGetFileOffset( uiEndAddress);
		}

		// Special case handling here because rollback file numbers start with
		// zero and then skip to a file number that is one beyond the
		// highest data file number - so the calculation for file size needs
		// to account for this.

		if (!uiLastFileNumber)
		{
			*pui64RollbackFileSize = (FLMUINT64)uiLastFileSize;
		}
		else
		{
			FLMUINT	uiFirstLogFileNum =
							FIRST_LOG_BLOCK_FILE_NUMBER( uiDbVersion);

			// Add full size of file zero plus a full size for every file
			// except the last one.

			(*pui64RollbackFileSize) = (FLMUINT64)(uiLastFileNumber -
																	uiFirstLogFileNum + 1) *
												 (FLMUINT64)pFile->uiMaxFileSize +
												 (FLMUINT64)uiLastFileSize;
		}
	}

	// See if they want the roll-forward log file sizes.

	if (pui64RflFileSize)
	{
		char *	pszDbFileName = pFile->pszDbPath;

		*pui64RflFileSize = 0;
		if (uiDbVersion < FLM_VER_4_3)
		{

			// For pre-4.3 versions, only need to get the size for one
			// RFL file.

			if (RC_BAD( rc = rflGetFileName( uiDbVersion, pszDbFileName,
														NULL, 1, szTmpName)))
			{
				goto Exit;
			}

			// Open the file and get its size.

			if (RC_BAD( rc = gv_FlmSysData.pFileSystem->OpenBlockFile( szTmpName,
			  						F_IO_RDWR | F_IO_SH_DENYNONE | F_IO_DIRECT,
									512, &pFileHdl)))
			{
				if (rc == FERR_IO_PATH_NOT_FOUND || rc == FERR_IO_INVALID_PATH)
				{
					rc = FERR_OK;
					uiLastFileSize = 0;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				if (RC_BAD( rc = pFileHdl->Size( &uiLastFileSize)))
				{
					goto Exit;
				}
			}
			if (pFileHdl)
			{
				pFileHdl->Release();
				pFileHdl = NULL;
			}
			*pui64RflFileSize = (FLMUINT64)uiLastFileSize;
		}
		else
		{

			// For 4.3 and greater, need to scan the RFL directory for
			// RFL files.  The call below to rflGetDirAndPrefix is done
			// to get the prefix.  It will not return the correct
			// RFL directory name, because we are passing in a NULL
			// RFL directory path (which may or may not be correct).
			// That's OK, because we get the RFL directory directly
			// from the F_Rfl object anyway.

			if (RC_BAD( rc = rflGetDirAndPrefix( uiDbVersion, pszDbFileName,
										NULL, szRflDir, szPrefix)))
			{
				goto Exit;
			}

			// We need to get the RFL directory from the F_Rfl object.

			f_strcpy( szRflDir, pFile->pRfl->getRflDirPtr());

			// See if the directory exists.  If not, we are done.

			if (gv_FlmSysData.pFileSystem->IsDir( szRflDir))
			{

				// Open the directory and scan for RFL files.

				if (RC_BAD( rc = gv_FlmSysData.pFileSystem->OpenDir(
					szRflDir, "*", &pDirHdl)))
				{
					goto Exit;
				}
				for (;;)
				{
					if (RC_BAD( rc = pDirHdl->Next()))
					{
						if (rc == FERR_IO_NO_MORE_FILES)
						{
							rc = FERR_OK;
							break;
						}
						else
						{
							goto Exit;
						}
					}
					pDirHdl->CurrentItemPath( szTmpName);

					// If the item looks like an RFL file name, get
					// its size.

					if (!pDirHdl->CurrentItemIsDir() &&
						  rflGetFileNum( uiDbVersion, szPrefix, szTmpName,
												&uiLastFileNumber))
					{

						// Open the file and get its size.

						if (RC_BAD( rc = gv_FlmSysData.pFileSystem->OpenBlockFile(
												szTmpName,
			  									F_IO_RDWR | F_IO_SH_DENYNONE | F_IO_DIRECT,
												512, &pFileHdl)))
						{
							if (rc == FERR_IO_PATH_NOT_FOUND ||
								 rc == FERR_IO_INVALID_PATH)
							{
								rc = FERR_OK;
								uiLastFileSize = 0;
							}
							else
							{
								goto Exit;
							}
						}
						else
						{
							if (RC_BAD( rc = pFileHdl->Size( &uiLastFileSize)))
							{
								goto Exit;
							}
						}
						if (pFileHdl)
						{
							pFileHdl->Release();
							pFileHdl = NULL;
						}
						(*pui64RflFileSize) += (FLMUINT64)uiLastFileSize;
					}
				}
			}
		}
	}

Exit:
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	if (pDirHdl)
	{
		pDirHdl->Release();
	}
	return( rc);
}

/*******************************************************************************
Desc:	Returns information about a particular database.
*******************************************************************************/
RCODE 
		// FERR_NOT_IMPLEMENTED - Invalid eGetConfigType value
		// FRC_NOT_FOUND - Requested information is not available
	FlmDbGetConfig(
		HFDB					hDb,
			// [IN] Handle to a database.
		eDbGetConfigType	eGetConfigType,
			// [IN] Information to retrieve.  Possible values of eGetConfigType:
			//
			//		PARAM		TYPE					MEANING / USE
			//
			//	FDB_GET_VERSION:  Retrieves the database's version number.
			//
			//		Value1	FLMUINT *			Returns version number.
			//		Value2	Not Used
			//		Value3	Not Used
			//
			//	FDB_GET_BLKSIZ:  Retrieves the database's block size.
			//
			//		Value1	FLMUINT *			Returns block size.
			//		Value2	Not Used
			//		Value3	Not Used
			//
			//	FDB_GET_DEFAULT_LANG:  Retrieves the database's default language.
			//
			//		Value1	FLMUINT *			Returns default language.
			//		Value2	Not Used
			//		Value3	Not Used
			//
			//	FDB_GET_PATH:  Retrieves the store's file path.  If the
			// path is not available, FRC_NOT_FOUND will be returned.
			//
			//		Value1	FLMBYTE *			Returns database file name.
			//		Value2	Not Used
			//		Value3	Not Used
			//
			//	FDB_GET_TRANS_ID:  Returns the stores' current transaction ID,
			// if any.  A value of zero is returned if there is no current
			// transaction.
			//
			//		Value1	FLMUINT *			Returns transaction ID.
			//		Value2	Not Used
			//		Value3	Not Used
			//
			//	FDB_GET_CHECKPOINT_INFO:  Returns checkpoint information for the
			// store.
			//
			//		Value1	CHECKPOINT_INFO *	Returns checkpoint info.
			//		Value2	Not Used
			//		Value3	Not Used
			//
			//	FDB_GET_LOCK_HOLDER:  Returns holder of lock.
			//
			//		Value1	LOCK_USER *			Returns current lock holder.
			//		Value2	Not Used
			//		Value3	Not Used
			//											
			//	FDB_GET_LOCK_WAITERS:  Returns waiters for the lock.
			//
			//		Value1	LOCK_USER **		Returns array of LOCK_USER
			//											structures.  Will return NULL if
			//											there are no waiters.  NOTE: Caller
			//											must delete[] the array!
			//		Value2	Not Used
			//		Value3	Not Used
			//											
			//	FDB_GET_LOCK_WAITERS_EX:  Calls methods of a user-supplied
			// object to return information about entries in the lock table
			//
			//		Value1	FlmLockInfo *		Returns lock information object.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_RFL_DIR:  Current RFL directory
			//
			//		Value1	FLMBYTE *			Returns RFL directory.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_RFL_FILE_NUM:  Current RFL file number
			//
			//		Value1	FLMUINT *			Returns current RFL file number.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_RFL_HIGHEST_NU:  Highest RFL file number that is
			// no longer needed for recovery after a server crash
			//
			//		Value1	FLMUINT *			Returns highest RFL file number.
			//						
			//	FDB_GET_RFL_FILE_SIZE_LIMITS:  Gets the minimum and
			// maximum RFL file sizes.
			//
			//		Value1	FLMUINT *			Returns minimum RFL file size.
			//		Value2	FLMUINT *			Returns maximum RFL file size.
			//		Value3	Not Used
			//						
			//	FDB_GET_RFL_KEEP_FLAG:  Returns a boolean to indicate whether
			// or not RFL files are being preserved
			//
			//		Value1	FLMBOOL *			Returns keep flag.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_LAST_BACKUP_TRANS_ID:  Transaction ID of the last backup.
			// The backup may have been a full backup or an incremental backup.
			//
			//		Value1	FLMUINT *			Returns transaction ID.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:  Gets the approx. number of
			// blocks that have changed since the last full or incremental backup.
			//
			//		Value1	FLMUINT *			Returns blocks changed.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_SERIAL_NUMBER:  Gets the database's serial number
			//
			//		Value1	FLMBYTE *			Returns serial number.  Buffer size
			//											should be at least F_SERIAL_NUM_SIZE.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:  Returns a boolean to
			// indicate whether or not keeping of roll-forward log files
			// will be automatically turned off when we run out of disk space.
			//
			//		Value1	FLMBOOL *			Returns flag.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:  Returns a boolean to
			// to indicate whether or not we are keeping aborted transactions
			// in the roll-forward log.
			//
			//		Value1	FLMBOOL *			Returns flag.
			//		Value2	Not Used
			//		Value3	Not Used
			//						
			//	FDB_GET_SIZES: Returns database size, rollback size,
			// and RFL file size.  NULL pointers may be passed if one
			// or more of the sizes is not requested.
			//
			//		Value1	FLMUINT64 *			Returns database size.
			//		Value2	FLMUINT64 *			Returns rollback size.
			//		Value3	FLMUINT64 *			Returns RFL size.
			//
			// FDB_GET_FILE_EXTEND_SIZE: Returns the amount by which the database
			// is extending files whenever it has to extend them.
			//		Value1	FLMUINT *			Returns extend size.
			//
			// FDB_GET_APP_DATA: Returns the application object for the DB.
			//		Value1	void **				Returns application object.
			//
			// FDB_GET_NEXT_INC_BACKUP_SEQ_NUM: Returns the sequence number
			// of the next incremental backup.
			//		Value1	FLMUINT *			Returns the sequence number
			//
			// FDB_GET_DICT_SEQ_NUM: Returns the sequence number of 
			// the dictionary
			//		Value1	FLMUINT *			Returns the sequence number	
			//
			// FDB_GET_FFILE_ID: Returns the ID of the FDB's FFILE
			//		Value1	FLMUINT *			Returns the ID
			//
			// FDB_GET_MUST_CLOSE_RC: Returns the error that caused the
			// "must close" flag to be set
			//		Value1	RCODE *				RCODE of "must close" error
			//
		void *				Value1,
			// [OUT] The type and domain of Value1 is determined by eGetConfigType.
		void *				Value2,
			// [OUT] The type and domain of Value2 is determined by eGetConfigType.
		void *				Value3
			// [OUT] The type and domain of Value3 is determined by eGetConfigType.
	)
{
	RCODE					rc = FERR_OK;
	FDB *					pDb = (FDB *)hDb;
	FFILE *				pFile = pDb->pFile;
	FLMBOOL				bDbInitialized = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiTransType = FLM_NO_TRANS;
	CHECKPOINT_INFO *	pCheckpointInfo;

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bDbInitialized = TRUE;

		CS_CONTEXT_p		pCSContext = pDb->pCSContext;
		FCL_WIRE				Wire( pCSContext, pDb);
		CREATE_OPTS			createOpts;

		if( !pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendOp(
			FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_GET_CONFIG)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TYPE, (FLMUINT)eGetConfigType)))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.sendTerminate()))
		{
			goto Transmission_Error;
		}

		/* Read the response. */
	
		if (RC_BAD( rc = Wire.read()))
		{
			goto Transmission_Error;
		}

		if( RC_BAD( rc = Wire.getRCode()))
		{
			goto Exit;
		}

		switch( eGetConfigType)
		{
			case FDB_GET_VERSION:
				Wire.copyCreateOpts( &createOpts);
				*((FLMUINT *)Value1) = createOpts.uiVersionNum;
				break;
			case FDB_GET_BLKSIZ:
				Wire.copyCreateOpts( &createOpts);
				*((FLMUINT *)Value1) = createOpts.uiBlockSize;
				break;
			case FDB_GET_DEFAULT_LANG:
				Wire.copyCreateOpts( &createOpts);
				*((FLMUINT *)Value1) = createOpts.uiDefaultLanguage;
				break;
			case FDB_GET_PATH:
			case FDB_GET_RFL_DIR:
			{
				char *		pszPath;
				POOL *		pPool = Wire.getPool();
				void *		pvMark = GedPoolMark( pPool);

				if( RC_BAD( rc = fcsConvertUnicodeToNative( pPool,
					(FLMUNICODE *)Wire.getFilePath(), &pszPath)))
				{
					goto Exit;
				}
				f_strcpy( (char *)Value1, pszPath);
				GedPoolReset( pPool, pvMark);
				break;
			}
			case FDB_GET_TRANS_ID:
			case FDB_GET_RFL_FILE_NUM:
			case FDB_GET_RFL_HIGHEST_NU:
			case FDB_GET_LAST_BACKUP_TRANS_ID:
			case FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:
			case FDB_GET_FILE_EXTEND_SIZE:
			case FDB_GET_APP_DATA:
			case FDB_GET_NEXT_INC_BACKUP_SEQ_NUM:
			case FDB_GET_DICT_SEQ_NUM:
			case FDB_GET_FFILE_ID:
			case FDB_GET_MUST_CLOSE_RC:
				*((FLMUINT *)Value1) = (FLMUINT)Wire.getNumber1();
				break;
			case FDB_GET_RFL_FILE_SIZE_LIMITS:
				if (Value1)
				{
					*((FLMUINT *)Value1) = (FLMUINT)Wire.getNumber1();
				}
				if (Value2)
				{
					*((FLMUINT *)Value2) = (FLMUINT)Wire.getNumber2();
				}
				break;
			case FDB_GET_RFL_KEEP_FLAG:
			case FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:
			case FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:
				*((FLMBOOL *)Value1) = Wire.getBoolean();
				break;
			case FDB_GET_CHECKPOINT_INFO:
				rc = fcsExtractCheckpointInfo( Wire.getHTD(), (CHECKPOINT_INFO *)Value1);
				break;
			case FDB_GET_LOCK_HOLDER:
				rc = fcsExtractLockUser( Wire.getHTD(), FALSE, ((LOCK_USER *)Value1));
				break;
			case FDB_GET_LOCK_WAITERS:
				rc = fcsExtractLockUser( Wire.getHTD(), TRUE, ((void *)Value1));
				break;
			case FDB_GET_SERIAL_NUMBER:
			{
				f_memcpy( (FLMBYTE *)Value1, 
					Wire.getSerialNum(), F_SERIAL_NUM_SIZE);
				break;
			}
			case FDB_GET_SIZES:
				if (Value1)
				{
					*((FLMUINT64 *)Value1) = (FLMUINT64)Wire.getNumber1();
				}
				if (Value2)
				{
					*((FLMUINT64 *)Value2) = (FLMUINT64)Wire.getNumber2();
				}
				if (Value3)
				{
					*((FLMUINT64 *)Value3) = (FLMUINT64)Wire.getNumber3();
				}
				break;

			default:
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				break;
		}

		goto Exit;

Transmission_Error:
		pCSContext->bConnectionGood = FALSE;
		goto Exit;
	}

	if (eGetConfigType == FDB_GET_RFL_FILE_NUM ||
				eGetConfigType == FDB_GET_RFL_HIGHEST_NU ||
				eGetConfigType == FDB_GET_RFL_FILE_SIZE_LIMITS ||
				eGetConfigType == FDB_GET_RFL_KEEP_FLAG ||
				eGetConfigType == FDB_GET_LAST_BACKUP_TRANS_ID ||
				eGetConfigType == FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP ||
				eGetConfigType == FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG ||
				eGetConfigType == FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG ||
				eGetConfigType == FDB_GET_SIZES ||
				eGetConfigType == FDB_GET_NEXT_INC_BACKUP_SEQ_NUM)
	{
		uiTransType = FLM_UPDATE_TRANS;
	}

	bDbInitialized = TRUE;
	if (RC_BAD( rc = fdbInit( pDb, uiTransType,
					FDB_TRANS_GOING_OK | FDB_DONT_RESET_DIAG, 
					FLM_NO_TIMEOUT | FLM_AUTO_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	switch( eGetConfigType)
	{
	case FDB_GET_VERSION:
		*((FLMUINT *)Value1) = pFile->FileHdr.uiVersionNum;
		break;
	case FDB_GET_BLKSIZ:
		*((FLMUINT *)Value1) = pFile->FileHdr.uiBlockSize;
		break;
	case FDB_GET_DEFAULT_LANG:
		*((FLMUINT *)Value1) = pFile->FileHdr.uiDefaultLanguage;
		break;
	case FDB_GET_PATH:
		if( RC_BAD( rc = flmGetFilePath( pFile, ((char *)Value1))))
		{
			goto Exit;
		}
		break;
	case FDB_GET_TRANS_ID:
		if (pDb->uiTransType != FLM_NO_TRANS)
		{
			*((FLMUINT *)Value1) = pDb->LogHdr.uiCurrTransID;
		}
		else if (pDb->uiFlags & FDB_HAS_FILE_LOCK)
		{

			// Get last committed value.

			*((FLMUINT *)Value1) = FB2UD( &pFile->ucLastCommittedLogHdr [LOG_CURR_TRANS_ID]);
		}
		else
		{
			*((FLMUINT *)Value1) = 0;
		}
		break;
	case FDB_GET_CHECKPOINT_INFO:
		pCheckpointInfo = (CHECKPOINT_INFO *)Value1;
		f_mutexLock( gv_FlmSysData.hShareMutex);
		flmGetCPInfo( pFile, pCheckpointInfo);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		break;
	case FDB_GET_LOCK_HOLDER:
		if (pFile->pFileLockObj)
		{
			rc = pFile->pFileLockObj->GetLockInfo( FALSE, (void *)Value1);
		}
		else
		{
			((LOCK_USER *)Value1)->uiThreadId = 0;
			((LOCK_USER *)Value1)->uiTime = 0;
		}
		break;
	case FDB_GET_LOCK_WAITERS:
		if (pFile->pFileLockObj)
		{
			rc = pFile->pFileLockObj->GetLockInfo( TRUE, (void *)Value1);
		}
		else
		{
			*((LOCK_USER **)Value1) = NULL;
		}
		break;

	case FDB_GET_LOCK_WAITERS_EX:
	{
		FlmLockInfo * pLockInfo = (FlmLockInfo *)Value1;

		if (pFile->pFileLockObj)
		{
			rc = pFile->pFileLockObj->GetLockInfo( pLockInfo);
		}
		else
		{
			pLockInfo->setLockCount( 0);
		}
		break;
	}

	case FDB_GET_RFL_DIR:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_strcpy( (char *)Value1, pDb->pFile->pRfl->getRflDirPtr());
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		break;

	case FDB_GET_RFL_FILE_NUM:
	{
		FLMUINT		uiLastCPFile;
		FLMUINT		uiLastTransFile;

		/*
		Get the CP and last trans RFL file numbers.  Need to
		return the higher of the two.  No need to lock the
		mutex because we are in an update transaction.
		*/

		uiLastCPFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
			LOG_RFL_LAST_CP_FILE_NUM]);

		uiLastTransFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
			LOG_RFL_FILE_NUM]);

		*((FLMUINT *)Value1) = uiLastCPFile > uiLastTransFile
									 ? uiLastCPFile
									 : uiLastTransFile;
		break;
	}

	case FDB_GET_RFL_HIGHEST_NU:
	{
		FLMUINT		uiLastCPFile;
		FLMUINT		uiLastTransFile;

		/*
		Get the CP and last trans RFL file numbers.  Need to
		return the lower of the two minus 1.
		*/

		uiLastCPFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
			LOG_RFL_LAST_CP_FILE_NUM]);

		uiLastTransFile = FB2UD( &pDb->pFile->ucUncommittedLogHdr[
			LOG_RFL_FILE_NUM]);

		*((FLMUINT *)Value1) =
			(FLMUINT)((uiLastCPFile < uiLastTransFile
						? uiLastCPFile
						: uiLastTransFile) - 1);
		break;
	}

	case FDB_GET_RFL_FILE_SIZE_LIMITS:
		if (Value1)
		{
			*((FLMUINT *)Value1) = (FLMUINT)FB2UD(
					&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
		}
		if (Value2)
		{
			if (pDb->pFile->FileHdr.uiVersionNum >= FLM_VER_4_3)
			{
				*((FLMUINT *)Value2) = (FLMUINT)FB2UD(
					&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MAX_FILE_SIZE]);
			}
			else
			{
				*((FLMUINT *)Value2) = (FLMUINT)FB2UD(
					&pDb->pFile->ucUncommittedLogHdr [LOG_RFL_MIN_FILE_SIZE]);
			}
		}
		break;

	case FDB_GET_RFL_KEEP_FLAG:
		*((FLMBOOL *)Value1) =
				pDb->pFile->ucUncommittedLogHdr [LOG_KEEP_RFL_FILES]
				? TRUE
				: FALSE;
		break;

	case FDB_GET_LAST_BACKUP_TRANS_ID:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
		*((FLMUINT *)Value1) = (FLMUINT)FB2UD(
				&pDb->pFile->ucUncommittedLogHdr [LOG_LAST_BACKUP_TRANS_ID]);
		break;

	case FDB_GET_BLOCKS_CHANGED_SINCE_BACKUP:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
		*((FLMUINT *)Value1) = (FLMUINT)FB2UD(
				&pDb->pFile->ucUncommittedLogHdr[ LOG_BLK_CHG_SINCE_BACKUP]);
		break;

	case FDB_GET_SERIAL_NUMBER:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
		f_mutexLock( gv_FlmSysData.hShareMutex);
		f_memcpy( (FLMBYTE *)Value1, 
			&pDb->pFile->ucLastCommittedLogHdr [LOG_DB_SERIAL_NUM],
			F_SERIAL_NUM_SIZE);
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		break;

	case FDB_GET_AUTO_TURN_OFF_KEEP_RFL_FLAG:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			*((FLMBOOL *)Value1) = FALSE;
		}
		else
		{
			*((FLMBOOL *)Value1) =
				pDb->pFile->ucUncommittedLogHdr [LOG_AUTO_TURN_OFF_KEEP_RFL]
				? TRUE
				: FALSE;
		}
		break;

	case FDB_GET_KEEP_ABORTED_TRANS_IN_RFL_FLAG:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			*((FLMBOOL *)Value1) = FALSE;
		}
		else
		{
			*((FLMBOOL *)Value1) =
				pDb->pFile->ucUncommittedLogHdr [LOG_KEEP_ABORTED_TRANS_IN_RFL]
				? TRUE
				: FALSE;
		}
		break;

	case FDB_GET_SIZES:
		rc = flmDbGetSizes( pDb, (FLMUINT64 *)Value1, (FLMUINT64 *)Value2,
										 (FLMUINT64 *)Value3);
		break;

	case FDB_GET_FILE_EXTEND_SIZE:
		*((FLMUINT *)Value1) = pDb->pFile->uiFileExtendSize;
		break;

	case FDB_GET_APP_DATA:
		*((void **)Value1) = pDb->pvAppData;
		break;

	case FDB_GET_NEXT_INC_BACKUP_SEQ_NUM:
		if (pDb->pFile->FileHdr.uiVersionNum < FLM_VER_4_3)
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
		*((FLMUINT *)Value1) = (FLMUINT)FB2UD(
				&pDb->pFile->ucUncommittedLogHdr[ LOG_INC_BACKUP_SEQ_NUM]);
		break;

	case FDB_GET_DICT_SEQ_NUM:
		if( pDb->pDict)
		{
			*((FLMUINT *)Value1) = pDb->pDict->uiDictSeq;
		}
		else
		{
			*((FLMUINT *)Value1) = pDb->pFile->pDictList->uiDictSeq;
		}
		break;
	case FDB_GET_FFILE_ID:
		*((FLMUINT *)Value1) = pDb->pFile->uiFFileId;
		break;
	case FDB_GET_MUST_CLOSE_RC:
		*((RCODE *)Value1) = pDb->pFile->rcMustClose;
		break;
	default:
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		break;
	}

Exit:

	if( bStartedTrans)
	{
		flmAbortDbTrans( pDb);
	}

	if( bDbInitialized)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc: Retrieves the Checkpoint info for the pFile passed in.  This assumes the
		hShareMutex has already been locked.
*****************************************************************************/
void flmGetCPInfo(
	void *					pFilePtr,
	CHECKPOINT_INFO *		pCheckpointInfo
	)
{
	FFILE *	pFile;
	FLMUINT	uiElapTime;
	FLMUINT	uiCurrTime;
				
	flmAssert( pFilePtr);
	flmAssert( pCheckpointInfo);

	pFile = (FFILE *)pFilePtr;

	f_memset( pCheckpointInfo, 0, sizeof( CHECKPOINT_INFO));
	if (pFile->pCPInfo)
	{
		pCheckpointInfo->bRunning = pFile->pCPInfo->bDoingCheckpoint;
		if (pCheckpointInfo->bRunning)
		{
			if (pFile->pCPInfo->uiStartTime)
			{
				uiCurrTime = FLM_GET_TIMER();

				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
							pFile->pCPInfo->uiStartTime);
				FLM_TIMER_UNITS_TO_MILLI( uiElapTime, pCheckpointInfo->uiRunningTime);
			}
			else
			{
				pCheckpointInfo->uiRunningTime = 0;
			}
			pCheckpointInfo->bForcingCheckpoint =
				pFile->pCPInfo->bForcingCheckpoint;
			if (pFile->pCPInfo->uiForceCheckpointStartTime)
			{
				uiCurrTime = FLM_GET_TIMER();
				uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
							pFile->pCPInfo->uiForceCheckpointStartTime);
				FLM_TIMER_UNITS_TO_MILLI( uiElapTime,
					pCheckpointInfo->uiForceCheckpointRunningTime);
			}
			else
			{
				pCheckpointInfo->uiForceCheckpointRunningTime = 0;
			}
			pCheckpointInfo->iForceCheckpointReason =
				pFile->pCPInfo->iForceCheckpointReason;
			pCheckpointInfo->bWritingDataBlocks =
				pFile->pCPInfo->bWritingDataBlocks;
			pCheckpointInfo->uiLogBlocksWritten =
				pFile->pCPInfo->uiLogBlocksWritten;
			pCheckpointInfo->uiDataBlocksWritten =
				pFile->pCPInfo->uiDataBlocksWritten;
		}
		pCheckpointInfo->uiBlockSize =
			(FLMUINT)pFile->FileHdr.uiBlockSize;
		pCheckpointInfo->uiDirtyCacheBytes = 
			pFile->uiDirtyCacheCount * pFile->FileHdr.uiBlockSize;
		if (pFile->pCPInfo->uiStartWaitTruncateTime)
		{
			uiCurrTime = FLM_GET_TIMER();

			uiElapTime = FLM_ELAPSED_TIME( uiCurrTime,
						pFile->pCPInfo->uiStartWaitTruncateTime);
			FLM_TIMER_UNITS_TO_MILLI( uiElapTime, 
				pCheckpointInfo->uiWaitTruncateTime);
		}
		else
		{
			pCheckpointInfo->uiWaitTruncateTime = 0;
		}
	}
}
