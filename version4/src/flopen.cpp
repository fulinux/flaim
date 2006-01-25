//-------------------------------------------------------------------------
// Desc:	Open database
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
// $Id: flopen.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define MAX_DIRTY_PERCENT	70

FSTATIC RCODE flmPhysFileOpen(
	FDB_p					pDb,
	const char *		pszFilePath,
	const char *		pszRflDir,
	FLMUINT				uiOpenFlags,
	FLMBOOL				bNewFile,
	F_Restore *			pRestoreObj);

FSTATIC RCODE flmReadFileHdr(
	FDB *				pDb,
	FLMBYTE *		tempBuf,
	LOG_HDR *		pLogHdr,
	FLMBOOL			bAllowLimitedMode);

FSTATIC void flmFreeCPInfo(
	CP_INFO **		ppCPInfoRV);

FSTATIC RCODE flmCPThread(
	F_Thread *		pThread);

FSTATIC RCODE flmDoRecover(
	FDB *				pDb,
	F_Restore *		pRestoreObj);

/*API~***********************************************************************
Desc : Opens an existing FLAIM database.
*END************************************************************************/
RCODE FlmDbOpen(
	const char *		pszDbFileName,
	const char *		pszDataDir,
	const char *		pszRflDir,
	FLMUINT				uiOpenFlags,
	const char *		pszPassword,
	HFDB *				phDbRV)
{
	RCODE				rc = FERR_OK;
	CS_CONTEXT *	pCSContext;

	rc = FERR_OK;
	*phDbRV = HFDB_NULL;

	if( !pszDbFileName || *pszDbFileName == 0)
	{
		rc = RC_SET( FERR_IO_INVALID_PATH);
		goto Exit;
	}

	if (RC_BAD( rc = flmGetCSConnection( pszDbFileName, &pCSContext)))
	{
		goto Exit;
	}

	if (pCSContext)
	{
		if (RC_BAD( rc = flmOpenOrCreateDbClientServer(
										pszDbFileName, pszDataDir, pszRflDir,
										uiOpenFlags, NULL,
										NULL, NULL, TRUE,
										pCSContext, (FDB_p *)phDbRV)))
		{
			(void)flmCloseCSConnection( &pCSContext);
		}
		goto Exit;
	}

	/* Open the file. */

	if (RC_BAD( rc = flmOpenFile( NULL,
		pszDbFileName, pszDataDir, pszRflDir,
		uiOpenFlags, FALSE,
		NULL, NULL, pszPassword, (FDB_p *)phDbRV)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/***************************************************************************
Desc:	Opens a database via the server.
****************************************************************************/
RCODE flmOpenOrCreateDbClientServer(
	const char *		pszDbPath,
	const char *		pszDataDir,
	const char *		pszRflDir,
	FLMUINT				uiOpenFlags,
	const char *		pszDictFileName,
	const char *		pszDictBuf,
	CREATE_OPTS *		pCreateOpts,
	FLMBOOL				bOpening,
	CS_CONTEXT *		pCSContext,
	FDB_p *				ppDb)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bAllocatedFdb = FALSE;
	FLMUNICODE *	puzDbPath;
	FLMUNICODE *	puzDataDir = NULL;
	FLMUNICODE *	puzRflDir = NULL;
	FLMUNICODE *	puzTmp;
	POOL				pool;
	FCL_WIRE			Wire( pCSContext);
	FDB_p				pDb;

	GedPoolInit( &pool, 128);

	/* Allocate and initialize an FDB structure. */

	if (RC_BAD( rc = flmAllocFdb( ppDb)))
	{
		goto Exit;
	}
	pDb = *ppDb;

	bAllocatedFdb = TRUE;

	/* Convert the paths to UNICODE. */

	if( RC_BAD( rc = fcsConvertNativeToUnicode( 
		&pool, pszDbPath, &puzDbPath)))
	{
		goto Exit;
	}

	if( pszDataDir)
	{
		if( RC_BAD( rc = fcsConvertNativeToUnicode( 
			&pool, pszDataDir, &puzDataDir)))
		{
			goto Exit;
		}
	}

	if( pszRflDir)
	{
		if( RC_BAD( rc = fcsConvertNativeToUnicode( 
			&pool, pszRflDir, &puzRflDir)))
		{
			goto Exit;
		}
	}

	/* Send a request to open or create the database. */

	if( RC_BAD( rc = Wire.sendOp( FCS_OPCLASS_DATABASE,
					(FLMUINT)((bOpening)
						? (FLMUINT)FCS_OP_DATABASE_OPEN
						: (FLMUINT)FCS_OP_DATABASE_CREATE))))
	{
		goto Exit;
	}

	if( RC_BAD( rc = Wire.sendString( WIRE_VALUE_FILE_PATH, puzDbPath)))
	{
		goto Transmission_Error;
	}

	if( puzRflDir)
	{
		if( RC_BAD( rc = Wire.sendString( WIRE_VALUE_FILE_PATH_2, puzRflDir)))
		{
			goto Transmission_Error;
		}
	}

	if( puzDataDir)
	{
		if( RC_BAD( rc = Wire.sendString( WIRE_VALUE_FILE_PATH_3, puzDataDir)))
		{
			goto Transmission_Error;
		}
	}

	if (uiOpenFlags)
	{
		if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiOpenFlags)))
		{
			goto Transmission_Error;
		}
	}

	if (!bOpening)
	{
		if (pszDictFileName)
		{

			/* Convert the path to UNICODE. */

			GedPoolReset( &pool, NULL);
			if (RC_BAD( rc = fcsConvertNativeToUnicode( &pool, pszDictFileName, &puzTmp)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = Wire.sendString( WIRE_VALUE_DICT_FILE_PATH, puzTmp)))
			{
				goto Transmission_Error;
			}
		}

		if (pszDictBuf)
		{
			/* Convert the path to UNICODE. */

			GedPoolReset( &pool, NULL);
			if (RC_BAD( rc = fcsConvertNativeToUnicode( &pool, pszDictBuf, &puzTmp)))
			{
				goto Exit;
			}

			if (RC_BAD( rc = Wire.sendString( WIRE_VALUE_DICT_BUFFER, puzTmp)))
			{
				goto Transmission_Error;
			}
		}

		if (pCreateOpts)
		{
			if (RC_BAD( rc = Wire.sendCreateOpts( WIRE_VALUE_CREATE_OPTS, pCreateOpts)))
			{
				goto Transmission_Error;
			}
		}
	}
	
	if (RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */
	
	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	if (bOpening)
	{
		if (pCreateOpts)
		{
			Wire.copyCreateOpts( pCreateOpts);
		}
	}

	pDb->pCSContext = pCSContext;
	*ppDb = pDb;

Exit:
	if (RC_BAD( rc))
	{
		(void)FlmDbClose( (HFDB *)ppDb);
	}
	GedPoolFree( &pool);
	return( rc);

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/***************************************************************************
Desc:    Allocates and initializes an FDB structure for a database which
			is to be opened or created.
****************************************************************************/
RCODE flmAllocFdb(
	FDB_p *		ppDb)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = NULL;

	// Allocate the FDB structure.

	*ppDb = NULL;
	if( RC_BAD( rc = f_calloc( (FLMUINT)sizeof( FDB), ppDb)))
	{
		goto Exit;
	}
	pDb = *ppDb;

	// Initialize pool for temporary allocations, making the block size
	// what we need for most read operations - several Key buffers

	GedPoolInit( &pDb->tmpKrefPool, 8192);
	GedPoolInit( &pDb->TempPool, (MAX_KEY_SIZ * 4));

#if defined( FLM_DEBUG) && (defined( FLM_WIN) || defined( FLM_NLM))
	/* Create the semaphore for controlling access to the structure. */

	pDb->hMutex = F_MUTEX_NULL;
	if (RC_BAD( rc = f_mutexCreate( &pDb->hMutex)))
	{
		goto Exit;
	}
#endif

	// Set up statistics.

	if (RC_BAD( rc = flmStatInit( &pDb->Stats, FALSE)))
	{
		goto Exit;
	}
	pDb->bStatsInitialized = TRUE;

Exit:

	if (RC_BAD( rc) && pDb)
	{
		flmDbClose( (HFDB *)ppDb, FALSE);
	}
	return( rc);
}

/****************************************************************************
Desc: This routine performs all of the necessary steps to complete
		a create or open of an FFILE, including notifying other threads
		waiting for the open or create to complete.
****************************************************************************/
RCODE flmCompleteOpenOrCreate(
	FDB_p *	ppDb,
	RCODE		rc,
	FLMBOOL	bNewFile,
	FLMBOOL	bAllocatedFdb)
{
	FDB *	pDb = *ppDb;

	if (RC_OK( rc))
	{

		/*
		If this is a newly created FFILE, we need to notify any
		threads waiting for the file to be created or opened that
		the create or open is now complete.
		*/

		if (bNewFile)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			rc = flmNewFileFinish( pDb->pFile, rc);
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
		}
	}
	else if (bAllocatedFdb)
	{
		FFILE *	pFile = pDb->pFile;

		/*
		Temporarily increment the use count on the FFILE structure
		so that it will NOT be put into the UNUSED list by the call
		to flmDbClose below.  If it is put into the UNUSED list,
		it can be freed by another thread.
		*/

		if (bNewFile)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			pFile->uiUseCount++;
			f_mutexUnlock( gv_FlmSysData.hShareMutex);
		}

		flmDbClose( (HFDB *)ppDb, FALSE);

		// If we allocated the FFILE, notify any
		// waiting threads.

		if (bNewFile)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);

			// Decrement the use count to compensate for the increment
			// that occurred above.

			pFile->uiUseCount--;

			// If this is a newly created FFILE, we need to notify any
			// threads waiting for the file to be created or opened that
			// the create or open is now complete.

			rc = flmNewFileFinish( pFile, rc);
			flmFreeFile( pFile);
			f_mutexUnlock ( gv_FlmSysData.hShareMutex);
		}
	}
	return( rc);
}

/****************************************************************************
Desc: This routine will open a file, returning a pointer to the FDB and
		FFILE structures.
****************************************************************************/
RCODE flmOpenFile(
	FFILE *				pFile,
	const char *		pszDbPath,
	const char *		pszDataDir,
	const char *		pszRflDir,
	FLMUINT				uiOpenFlags,
	FLMBOOL				bInternalOpen,
	F_Restore *			pRestoreObj,
	F_FileHdlImp *		pLockFileHdl,
	const char *		pszPassword,
	FDB_p *				ppDb)
{
	RCODE				rc;
	FLMBOOL			bNewFile = FALSE;
	FLMBOOL			bMutexLocked = FALSE;
	FLMBOOL			bAllocatedFdb = FALSE;
	FDB_p				pDb;
	FLMBOOL			bNeedToOpen = FALSE;

	/* Allocate and initialize an FDB structure. */

	if (RC_BAD( rc = flmAllocFdb( ppDb)))
	{
		goto Exit;
	}
	pDb = *ppDb;
	if (bInternalOpen)
	{
		pDb->uiFlags |= FDB_INTERNAL_OPEN;
	}
	bAllocatedFdb = TRUE;

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	/*
	Free any unused structures that have been unused for the maximum
	amount of time.  May unlock and re-lock the global mutex.
	*/

	flmCheckNUStructs( 0);

	/*
	Look up the file using flmFindFile to see if we already
	have the file open.
	*/

	if (!pFile)
	{
		bNeedToOpen = TRUE;

		// May unlock and re-lock the global mutex.
		if (RC_BAD( rc = flmFindFile( pszDbPath, pszDataDir, &pFile)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = flmCheckFFileState( pFile)))
	{
		goto Exit;
	}

	if (!pFile)
	{
		if (RC_BAD( rc = flmAllocFile( pszDbPath, pszDataDir, pszPassword, &pFile)))
		{
			goto Exit;
		}
		flmAssert( !pLockFileHdl);
		bNewFile = TRUE;
	}
	else if( pLockFileHdl)
	{
		flmAssert( pFile);
		flmAssert( !pFile->uiUseCount);
		flmAssert( pFile->uiFlags & DBF_BEING_OPENED);
		flmAssert( !(pFile->uiFlags & DBF_IN_NU_LIST));

		/*
		Put the FFILE back in the NU list.  FlmDbRestore removed the FFILE from
		the NU list after allocating it so that it would not disappear during
		the restore.  When the FFILE is linked to the FDB (below), the FFILE
		will be removed from the NU list.
		*/

		flmLinkFileToNUList( pFile);

		/*
		Assign the lock file handle
		*/

		pFile->pLockFileHdl = pLockFileHdl;

		/*
		Set to NULL to prevent lock file from being released below
		*/

		pLockFileHdl = NULL;

		bNewFile = TRUE;
		bNeedToOpen = TRUE;
	}
	else
	{
		flmAssert( !pLockFileHdl);
		if (RC_BAD( rc = flmVerifyFileUse( gv_FlmSysData.hShareMutex, &pFile)))
		{
			goto Exit;
		}
	}

	// If there is a password, verify that it matches the current password.

	if ( pszPassword && pszPassword[0])
	{
		if ( pFile->pszDbPassword)
		{
			if ( f_strcmp( pszPassword, pFile->pszDbPassword) != 0)
			{
				if (uiOpenFlags & FO_ALLOW_LIMITED)
				{
					pFile->bInLimitedMode = TRUE;
					pFile->rcLimitedCode = RC_SET( FERR_PASSWD_INVALID);
				}
				else
				{
					rc = RC_SET( FERR_PASSWD_INVALID);
					goto Exit;
				}
			}
		}
		else if (uiOpenFlags & FO_ALLOW_LIMITED)
		{
			pFile->bInLimitedMode = TRUE;
			pFile->rcLimitedCode = RC_SET( FERR_PASSWD_INVALID);
		}
		else
		{
			rc = RC_SET( FERR_PASSWD_INVALID);
			goto Exit;
		}
	}

	// If there was no password passed in, but there should have been, then oops.

	else if (pFile->pszDbPassword && pFile->pszDbPassword[0])
	{
		if (uiOpenFlags & FO_ALLOW_LIMITED)
		{
			pFile->bInLimitedMode = TRUE;
			pFile->rcLimitedCode = RC_SET( FERR_REQUIRE_PASSWD);
		}
		else
		{
			rc = RC_SET( FERR_REQUIRE_PASSWD);
			goto Exit;
		}
	}

	// Link the FDB to the file.

	rc = flmLinkFdbToFile( pDb, pFile);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	bMutexLocked = FALSE;
	if (RC_BAD(rc))
	{
		goto Exit;
	}
		
	(void)flmStatGetDb( &pDb->Stats, pFile,
							0, &pDb->pDbStats, NULL, NULL);

	if (bNeedToOpen)
	{
		if (RC_BAD( rc = flmPhysFileOpen( pDb, pszDbPath, pszRflDir,
										  uiOpenFlags, bNewFile,
										  pRestoreObj)))
		{
			goto Exit;
		}
	}

	// Start a checkpoint thread

	if (bNewFile && !(uiOpenFlags & FO_DONT_REDO_LOG))
	{
		flmAssert( pFile->pCPThrd == NULL);
		if (RC_BAD( rc = flmStartCPThread( pFile)))
		{
			goto Exit;
		}

		if( !(uiOpenFlags & FO_DONT_RESUME_BACKGROUND_THREADS))
		{
			if (RC_BAD( rc = flmStartBackgrndIxThrds( pDb)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = flmStartMaintThread( pFile)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	if( pLockFileHdl)
	{
		pLockFileHdl->Release();
	}

	rc = flmCompleteOpenOrCreate( ppDb, rc, bNewFile, bAllocatedFdb);
	return( rc);
}

/****************************************************************************
Desc: This routine checks to see if it is OK for another FDB to use a file.
		If so, it increments the file's use counter.  NOTE: This routine
		assumes that the calling routine has locked the semaphore.
****************************************************************************/
RCODE flmVerifyFileUse(
	F_MUTEX		hMutex,
	FFILE **		ppFileRV
	)
{
	RCODE    rc = FERR_OK;
	FFILE *	pFile = *ppFileRV;

	// Can't open the file if it is being closed by someone else.

	if (pFile->uiFlags & DBF_BEING_CLOSED)
	{
		rc = RC_SET( FERR_IO_ACCESS_DENIED);
		goto Exit;
	}

	// If the file is in the process of being opened by another
	// thread, wait for the open to complete.

	if (pFile->uiFlags & DBF_BEING_OPENED)
	{
		if (RC_BAD( rc = flmWaitNotifyReq( hMutex,
												&pFile->pOpenNotifies,
												(void *)0)))
		{
			// GW Bug #24307.  If flmWaitNotifyReq returns a bad RC, assume that
			// the other thread will unlock and free the pFile structure.  This
			// routine should only unlock the pFile if an error occurs at some
			// other point.  See flmVerifyFileUse.

			// *ppFileRV is set to NULL at Exit.
			goto Exit;
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		*ppFileRV = NULL;
	}
	return( rc);
}

/****************************************************************************
Desc:	Returns the length of the base part of a database name.  If the
		name ends with a '.' or ".db", this will not be included in the
		returned length.
****************************************************************************/
void flmGetDbBasePath(
	char *				pszBaseDbName,
	const char *		pszDbName,
	FLMUINT *			puiBaseDbNameLen)
{
	FLMUINT		uiBaseLen = f_strlen( pszDbName);

	if( uiBaseLen <= 3 || 
		f_stricmp( &pszDbName[ uiBaseLen - 3], ".db") != 0)
	{
		if( pszDbName[ uiBaseLen - 1] == '.')
		{
			uiBaseLen--;
		}
	}
	else
	{
		uiBaseLen -= 3;
	}

	f_memcpy( pszBaseDbName, pszDbName, uiBaseLen);
	pszBaseDbName[ uiBaseLen] = 0;

	if( puiBaseDbNameLen)
	{
		*puiBaseDbNameLen = uiBaseLen;
	}
}

/****************************************************************************
Desc: This routine obtains exclusive access to a database by creating
		a .lck file.  FLAIM holds the .lck file open as long as the database
		is open.  When the database is finally closed, it deletes the .lck
		file.  This is only used for 3.x databases.
****************************************************************************/
RCODE flmCreateLckFile(
	const char *		pszFilePath,
	F_FileHdlImp **	ppLockFileHdlRV)
{
	RCODE					rc = FERR_OK;
	char					szLockPath [F_PATH_MAX_SIZE];
	F_FileHdlImp *		pLockFileHdl = NULL;
	FLMUINT				uiBaseLen;

	/*
	Extract the base name and put a .lck extension on it to create
	the full path for the .lck file.
	*/

	flmGetDbBasePath( szLockPath, pszFilePath, &uiBaseLen);
	f_strcpy( &szLockPath[ uiBaseLen], ".lck");

	/*
	Attempt to create the lock file.  If that succeeds, we are
	OK to use the database.  If it fails, the lock file may have
	been left because of a crash if FLAIM was not shut down properly.
	Hence, we first try to delete the file.  If that succeeds, we
	then attempt to create the file again.  If it, or the 2nd create
	fail, we simply return an access denied error.
	*/

#ifndef FLM_UNIX
	if( RC_BAD( gv_FlmSysData.pFileSystem->Create( szLockPath,
			F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYRW | F_IO_DELETE_ON_CLOSE,
			(F_FileHdl **)&pLockFileHdl)))
	{
		if( RC_BAD( gv_FlmSysData.pFileSystem->Delete( szLockPath)))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
		else if( RC_BAD( gv_FlmSysData.pFileSystem->Create( szLockPath, 
			F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYRW | F_IO_DELETE_ON_CLOSE,
			(F_FileHdl **)&pLockFileHdl)))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
	}
#else
	if( RC_BAD( gv_FlmSysData.pFileSystem->Create( szLockPath,
			F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYRW, 
			(F_FileHdl **)&pLockFileHdl)))
	{
		if( RC_BAD( gv_FlmSysData.pFileSystem->Open( szLockPath, 
			F_IO_RDWR | F_IO_SH_DENYRW, (F_FileHdl **)&pLockFileHdl)))
		{
			rc = RC_SET( FERR_IO_ACCESS_DENIED);
			goto Exit;
		}
	}

	if( RC_BAD( pLockFileHdl->Lock()))
	{
		rc = RC_SET( FERR_IO_ACCESS_DENIED);
		goto Exit;
	}
#endif

	*ppLockFileHdlRV = pLockFileHdl;
	pLockFileHdl = NULL;

Exit:

	if (pLockFileHdl)
	{
		(void)pLockFileHdl->Close();
		pLockFileHdl->Release();
		pLockFileHdl = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc: This routine obtains exclusive access to a database by creating
		a .lck file.  FLAIM holds the .lck file open as long as the database
		is open.  When the database is finally closed, it deletes the .lck
		file.  This is only used for 3.x databases.
****************************************************************************/
RCODE flmGetExclAccess(
	const char *		pszFilePath,
	FDB *					pDb)
{
	RCODE					rc = FERR_OK;
	FFILE *				pFile = pDb->pFile;
	FLMBOOL				bNotifyWaiters = FALSE;
	FLMBOOL				bMutexLocked = FALSE;

	/*
	If pFile->pLockFileHdl is non-NULL, it means that we currently
	have the file locked with a lock file.  There is no need to make
	this test inside a mutex lock, because the lock file handle can only
	be set to NULL when the use count goes to zero, meaning that the thread
	that sets it to NULL will be the only thread accessing it.
	
	However, it is possible that two or more threads will simultaneously
	test pLockFileHdl and discover that it is NULL.  In that case,
	we allow one thread to proceed and attempt to get a lock on the file
	while the other threads wait to be notified of the results of the
	attempt to lock the file.
	*/

	if (pFile->pLockFileHdl)
	{
		goto Exit;
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	bMutexLocked = TRUE;

	if (pFile->bBeingLocked)
	{

		/*
		If the file is in the process of being locked by another
		thread, wait for the lock to complete.  NOTE: flmWaitNotifyReq will
		re-lock the mutex before returning.
		*/

		rc = flmWaitNotifyReq( gv_FlmSysData.hShareMutex, &pFile->pLockNotifies,
												(void *)0);
		goto Exit;
	}
	else
	{

		/*
		No other thread was attempting to lock the file, so
		set this thread up to make the attempt.  Other threads
		coming in at this point will be required to wait and
		be notified of the results.
		*/

		pFile->bBeingLocked = TRUE;
		bNotifyWaiters = TRUE;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if (RC_BAD( rc = flmCreateLckFile( pszFilePath, &pFile->pLockFileHdl)))
	{
		goto Exit;
	}

Exit:

	if (bNotifyWaiters)
	{
		FNOTIFY *		pNotify;
		F_SEM				hSem;

		/* Notify any thread waiting on the lock what its status is. */

		if( !bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.hShareMutex);
			bMutexLocked = TRUE;
		}

		pNotify = pFile->pLockNotifies;
		while (pNotify)
		{
			*(pNotify->pRc) = rc;
			hSem = pNotify->hSem;
			pNotify = pNotify->pNext;
			f_semSignal( hSem);
		}

		pFile->bBeingLocked = FALSE;
		pFile->pLockNotifies = NULL;
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
		bMutexLocked = FALSE;
	}

	if( bMutexLocked)
	{
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine checks to see if it is OK for another FDB to use a file.
		If so, it increments the file's use counter.  NOTE: This routine
		assumes that the global mutex is NOT locked.
****************************************************************************/
FSTATIC RCODE flmPhysFileOpen(
	FDB_p					pDb,
	const char *		pszFilePath,	// File name
	const char *		pszRflDir,		// RFL directory
	FLMUINT				uiOpenFlags,	// Flags for doing physical open
	FLMBOOL				bNewFile,		// Is this a new file structure?
	F_Restore *			pRestoreObj)	// Restore object
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMBOOL			bAllowLimitedMode = FALSE;

	// If this is the first open of the database, read the header block

	if( bNewFile)
	{
		LOG_HDR		LogHdr;

#ifdef FLM_USE_NICI
		if( uiOpenFlags & FO_ALLOW_LIMITED)
#endif
		{
			bAllowLimitedMode = TRUE;
		}
		if( RC_BAD( rc = flmReadFileHdr( pDb, pFile->pucLogHdrWriteBuf, &LogHdr,
													bAllowLimitedMode)))
		{
			goto Exit;
		}

		// Allocate the pRfl object.  Could not do this until this point
		// because we need to have the version number, block size, etc.
		// setup in the pFile->FileHdr.

		flmAssert( !pFile->pRfl);

		if( (pFile->pRfl = f_new F_Rfl) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pFile->pRfl->setup( pFile, pszRflDir)))
		{
			goto Exit;
		}

		// Setup the FFILE's ECache object

		flmAssert( pFile->pECacheMgr == NULL);
		if( gv_FlmSysData.bOkToUseESM)
		{
			if( (pFile->pECacheMgr = f_new FlmECache) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( !pFile->pECacheMgr->setupECache( pFile->FileHdr.uiBlockSize,
				pFile->uiMaxFileSize))
			{
				pFile->pECacheMgr->Release();
				pFile->pECacheMgr = NULL;
			}
			else
			{
				// Normally the ECacheMgr is set in flmLinkFdbToFile but
				// we have to handle this special case here.  When this
				// FDB was linked there was no ECacheMgr.

				flmAssert( pDb->pSFileHdl != NULL);
				pDb->pSFileHdl->setECacheMgr( pFile->pECacheMgr);
			}
		}
	}
	else
	{
		if (pFile->bInLimitedMode && !(uiOpenFlags & FO_ALLOW_LIMITED))
		{
			// Should we override the open parameter to allow limited mode?

			if (pFile->bHaveEncKey) 
			{
				rc = RC_SET( pFile->rcLimitedCode);
				goto Exit;
			}
		}
	}

	/*
	We must have exclusive access.  Create a lock file for that
	purpose, if there is not already a lock file.
	*/

	if (!pFile->pLockFileHdl)
	{
		if (RC_BAD( rc = flmGetExclAccess( pszFilePath, pDb)))
		{
			goto Exit;
		}
	}

	// Do a recovery to ensure a consistent database
	// state before going any further.  The FO_DONT_REDO_LOG
	// flag is used ONLY by the VIEW program.

	if (bNewFile && !(uiOpenFlags & FO_DONT_REDO_LOG))
	{
		if (RC_BAD( rc = flmDoRecover( pDb, pRestoreObj)))
		{
			goto Exit;
		}
	}

Exit:

	if (RC_BAD( rc))
	{
		(void)pDb->pSFileHdl->ReleaseFiles( TRUE);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine finishes up after creating a new FFILE structure.  It
		will notify any other threads waiting for the operation to complete
		of the status of the operation.
****************************************************************************/
RCODE flmNewFileFinish(
	FFILE *		pFile,	/* Pointer to FFILE structure. */
	RCODE			OpenRc)  /* Return code to send to other threads that are
									waiting for the open to complete. */
{
	FNOTIFY *	pNotify;
	F_SEM			hSem;

	if (!pFile)
		goto Exit;

	/* Notify anyone waiting on the operation what its status is. */

	pNotify = pFile->pOpenNotifies;
	while (pNotify)
	{
		*(pNotify->pRc) = OpenRc;
		hSem = pNotify->hSem;
		pNotify = pNotify->pNext;
		f_semSignal( hSem);
	}
	
	pFile->pOpenNotifies = NULL;
	pFile->uiFlags &= (~(DBF_BEING_OPENED));
Exit:
	return OpenRc;
}

/****************************************************************************
Desc:		This routine is used to see if a file is already in use somewhere.
			This is only called for files which are opened directly by the
			application.
Notes:	This routine assumes that the global mutex is locked, but it
			may unlock and re-lock the mutex if needed.
****************************************************************************/
RCODE flmFindFile(
	const char *		pszDbPath,
	const char *		pszDataDir,
	FFILE **				ppFileRV)
{
	RCODE       	rc = FERR_OK;
	FBUCKET *   	pBucket;
	FLMUINT			uiBucket;
	FLMBOOL			bMutexLocked = TRUE;
	FFILE *			pFile;
	char				szDbPathStr1 [F_PATH_MAX_SIZE];
	char				szDbPathStr2 [F_PATH_MAX_SIZE];

	*ppFileRV = NULL;
	
	// Normalize the path to a string before looking for it.
	// NOTE: On non-UNIX platforms, this will basically convert
	// the string to upper case.

	if (RC_BAD( rc = f_pathToStorageString( pszDbPath, szDbPathStr1)))
	{
		goto Exit;
	}

Retry:

	*ppFileRV = NULL;

	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
		bMutexLocked = TRUE;
	}

	pBucket = gv_FlmSysData.pFileHashTbl;
	uiBucket = flmStrHashBucket( szDbPathStr1, pBucket, FILE_HASH_ENTRIES);
	pFile = (FFILE *)pBucket [uiBucket].pFirstInBucket;
	while( pFile)
	{
		if (RC_BAD( rc = f_pathToStorageString( pFile->pszDbPath, szDbPathStr2)))
		{
			goto Exit;
		}

		// Compare the strings.  It is OK to use f_strcmp on all platforms
		// because on non-UNIX platforms the calls to f_pathToStorageString
		// has already converted the characters to upper case - hence, we
		// are doing a case-insensitive comparison.

		if( f_strcmp( szDbPathStr1, szDbPathStr2) == 0)
		{
			// Make sure data paths match.

			if( pszDataDir && *pszDataDir)
			{
				if( RC_BAD( rc = f_pathToStorageString( 
					pszDataDir, szDbPathStr2)))
				{
					goto Exit;
				}

				if( pFile->pszDataDir)
				{
					if( RC_BAD( rc = f_pathToStorageString( 
						pFile->pszDataDir, szDbPathStr1)))
					{
						goto Exit;
					}

					// f_strcmp is ok, because pathToStorageString converts to upper-case
					// on non-UNIX platforms.  On UNIX we want the comparison to be
					// case-sensitive.

					if( f_strcmp( szDbPathStr1, szDbPathStr2) != 0)
			{
						rc = RC_SET( FERR_DATA_PATH_MISMATCH);
						goto Exit;
					}
				}
				else
				{
					rc = RC_SET( FERR_DATA_PATH_MISMATCH);
					goto Exit;
				}
			}
			else if( pFile->pszDataDir)
			{
				rc = RC_SET( FERR_DATA_PATH_MISMATCH);
				goto Exit;
			}
			*ppFileRV = pFile;
			break;
		}
		pFile = pFile->pNext;
	}

	if( *ppFileRV && pFile->uiFlags & DBF_BEING_CLOSED)
	{
		// Put ourselves into the notify list and then re-try
		// the lookup when we wake up

		if (RC_BAD( rc = flmWaitNotifyReq( gv_FlmSysData.hShareMutex,
												&pFile->pCloseNotifies,
												(void *)0)))
		{
			goto Exit;
		}

		// The mutex will be locked at this point.  Re-try the lookup.
		// IMPORTANT NOTE: pFile will have been destroyed by this 
		// time.  DO NOT use it for anything!

		goto Retry;
	}

Exit:

	// Make sure the global mutex is re-locked before exiting

	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.hShareMutex);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine allocates a new FFILE structure and links it
		into its hash buckets.
		NOTE: This routine assumes that the global mutex has already
		been locked. It may unlock it temporarily if there is an error,
		but will always relock it before exiting.
****************************************************************************/
RCODE flmAllocFile(
	const char *		pszDbPath,
	const char *		pszDataDir,
	const char *		pszDbPassword,
	FFILE_p *			ppFile)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiAllocLen;
	FLMUINT				uiDbNameLen = f_strlen( pszDbPath) + 1;
	FLMUINT				uiDirNameLen;
	FFILE *				pFile = NULL;
	FFileItemId *		pFileItemId1 = NULL;
	FFileItemId *		pFileItemId2 = NULL;

	uiDirNameLen = (pszDataDir && *pszDataDir)
						? f_strlen( pszDataDir) + 1
						: 0;
	uiAllocLen = (FLMUINT)(sizeof( FFILE) + uiDbNameLen + uiDirNameLen);
	if (RC_BAD( rc = f_calloc( uiAllocLen, &pFile)))
	{
		goto Exit;
	}

	pFile->hMaintSem = F_SEM_NULL;
	pFile->uiFFileId = gv_FlmSysData.uiNextFFileId++;
	pFile->pCPInfo = NULL;
	pFile->uiFileExtendSize = DEFAULT_FILE_EXTEND_SIZE;
	GedPoolInit( &pFile->krefPool, 8192);

	// Allocate a buffer for writing the database header
	
#ifdef FLM_WIN
	if ((pFile->pucLogHdrWriteBuf = (FLMBYTE *)VirtualAlloc( NULL,
		(DWORD)MAX_BLOCK_SIZE, MEM_COMMIT, PAGE_READWRITE)) == NULL)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_MEM);
		goto Exit;
	}
	f_memset( pFile->pucLogHdrWriteBuf, 0, MAX_BLOCK_SIZE);
#elif defined( FLM_LINUX) || defined( FLM_SOLARIS)
	if( (pFile->pucLogHdrWriteBuf = (FLMBYTE *)memalign( 
		sysconf(_SC_PAGESIZE), MAX_BLOCK_SIZE)) == NULL) 
	{
		rc = MapErrnoToFlaimErr(errno, FERR_MEM);
		goto Exit;
	}
	f_memset( pFile->pucLogHdrWriteBuf, 0, MAX_BLOCK_SIZE);
#else
	if (RC_BAD( rc = f_calloc( MAX_BLOCK_SIZE, &pFile->pucLogHdrWriteBuf)))
	{
		goto Exit;
	}
#endif

	// If a password was passed in, allocate a buffer for it.
	
	if (pszDbPassword && pszDbPassword[0])
	{
		if (RC_BAD( rc = f_calloc( f_strlen( pszDbPassword) + 1, &pFile->pszDbPassword)))
		{
			goto Exit;
		}
		f_memcpy( pFile->pszDbPassword, pszDbPassword, f_strlen(pszDbPassword));
	}

	// Setup the write buffer managers.

	if ((pFile->pBufferMgr = f_new F_IOBufferMgr) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pFile->pBufferMgr->setMaxBuffers( MAX_PENDING_WRITES);
	pFile->pBufferMgr->setMaxBytes( MAX_WRITE_BUFFER_BYTES);

	// Initialize members of FFILE.

	pFile->uiBucket = 0xFFFF;
	pFile->uiFlags = DBF_BEING_OPENED;

	// Copy the file name immediately following the FFILE structure.
	// and the data directory immediately following that.
	// NOTE: uiDbNameLen includes the null terminating byte.
	// and uiDirNameLen includes the null terminating byte.

	pFile->pszDbPath = (char *)(&pFile[1]);
	f_memcpy( pFile->pszDbPath, pszDbPath, uiDbNameLen);
	if (uiDirNameLen)
	{
		pFile->pszDataDir = pFile->pszDbPath + uiDbNameLen;
		f_memcpy( pFile->pszDataDir, pszDataDir, uiDirNameLen);
	}

	// Link the file into the various lists it needs to be linked into.

	if (RC_BAD( rc = flmLinkFileToBucket( pFile)))
	{
		goto Exit;
	}
	flmLinkFileToNUList( pFile);

	// Allocate the lock objects - must be done AFTER setting up the
	// file name stuff up above.

	if( (pFileItemId1 = f_new FFileItemId( pFile, TRUE)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (pFileItemId2 = f_new FFileItemId( pFile, FALSE)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Allocate and initialize the file ID list

	if( (pFile->pFileIdList = f_new F_FileIdList) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pFile->pFileIdList->setup()))
	{
		goto Exit;
	}

	// Allocate a lock object for write locking.

	if( (pFile->pWriteLockObj =
			gv_FlmSysData.pServerLockMgr->GetLockObject(
												pFileItemId1)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pFile->pWriteLockObj->AddRef();

	// Allocate a lock object for file locking.

	if( (pFile->pFileLockObj =
			gv_FlmSysData.pServerLockMgr->GetLockObject(
												pFileItemId2)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	pFile->pFileLockObj->AddRef();

	// Initialize the maintenance thread's semaphore

	if( RC_BAD( rc = f_semCreate( &pFile->hMaintSem)))
	{
		goto Exit;
	}

	*ppFile = pFile;

Exit:

	if( pFileItemId1)
	{
		pFileItemId1->Release();
	}

	if( pFileItemId2)
	{
		pFileItemId2->Release();
	}
	if( RC_BAD( rc))
	{
		if( pFile)
		{
			flmFreeFile( pFile);
		}
	}
	return( rc);
}

/***************************************************************************
Desc: This routine reads the header information for an existing
		flaim database and makes sure we have a valid database.  It
		also reads the log file header record.
*****************************************************************************/
FSTATIC RCODE flmReadFileHdr(
	FDB *			pDb,						// Pointer to operation context
	FLMBYTE *	pBuf,						// Internal buffer to be used for reading
	LOG_HDR *	pLogHdr,					// Returns log header stuff
	FLMBOOL		bAllowLimitedMode)	// Are we allowed to open in limited mode?
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	DB_STATS *		pDbStats = pDb->pDbStats;
	F_FileHdlImp *	pCFileHdl;

	// Read and verify the file and log headers.

	if( RC_BAD( rc = pDb->pSFileHdl->GetFileHdl( 0, FALSE, &pCFileHdl)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmReadAndVerifyHdrInfo( pDbStats,
				pCFileHdl, pBuf, &pFile->FileHdr, pLogHdr, NULL)))
	{
		goto Exit;
	}

	// Shove stuff into the log header area of the pFile.
	// IMPORTANT NOTE! - This code assumes that DB_LOG_HEADER_START
	// is found in the first 2K of the file - i.e., it will be inside
	// the pBuf that we read above.

	f_memcpy( pFile->ucLastCommittedLogHdr,
				 &pBuf[ DB_LOG_HEADER_START], LOG_HEADER_SIZE);

	// Create the database wrapping key from the data in Log Header

#ifdef FLM_USE_NICI
	if( pFile->FileHdr.uiVersionNum >= FLM_VER_4_60)
	{
		FLMUINT32	ui32KeyLen;
	
		ui32KeyLen = FB2UW( &pFile->ucLastCommittedLogHdr[ LOG_DATABASE_KEY_LEN]);

		// VISIT:  Looks like the database was created by a version of flaim that did not have
		// encryption.  Now we are opening it with a version that does.  Do we want to upgrade
		// automatically?  Perhaps we need a different version, one that says we have encryption or don't???
		// This code will force limited mode and not upgrade.

		if (ui32KeyLen == 0)
		{
			pFile->rcLimitedCode = FERR_ENCRYPTION_UNAVAILABLE;
			pFile->bInLimitedMode = TRUE;
			pFile->bHaveEncKey = FALSE;
		}
		else
		{
			// We do have an encryption key.  If we end up in limited mode, it is an error
			// condition from NICI that got us there.

			pFile->bHaveEncKey = TRUE;

			if ((pFile->pDbWrappingKey = f_new F_CCS) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		
			if( RC_BAD( rc = pFile->pDbWrappingKey->init( TRUE, FLM_NICI_AES)))
			{
				goto Exit;
			}
	
			// If the key was encrypted in a password, then the pszDbPassword parameter better be the key used to encrypt it.
			// If the key was not encrypted in a password, the pszDbPassword parameter should be NULL.
	
			if( RC_BAD( rc = pFile->pDbWrappingKey->setKeyFromStore(
									&pFile->ucLastCommittedLogHdr[LOG_DATABASE_KEY],
									ui32KeyLen, pFile->pszDbPassword, NULL, FALSE)))
			{
				if (bAllowLimitedMode)
				{
					pFile->rcLimitedCode = rc;
					rc = FERR_OK;
					pFile->bInLimitedMode = TRUE;
				}
				else
				{
					goto Exit;
				}
			}
		}
	}
	else
	{
		pFile->rcLimitedCode = FERR_ENCRYPTION_UNAVAILABLE;
		pFile->bInLimitedMode = TRUE;
		pFile->bHaveEncKey = FALSE;
	}
#else
	F_UNREFERENCED_PARM( bAllowLimitedMode);
	pFile->rcLimitedCode = FERR_ENCRYPTION_UNAVAILABLE;
	pFile->bInLimitedMode = TRUE;
	pFile->bHaveEncKey = FALSE;
#endif

	// Get the maximum file size.  On 4.3 and greater it is stored
	// in the log header.  For now, we will only get this when we
	// first open a database.  If we ever implement code to allow
	// it to be changed during a transaction, we would need to
	// reset pFile->uiMaxFileSize.

	pFile->uiMaxFileSize = flmGetMaxFileSize(
									pFile->FileHdr.uiVersionNum,
									pFile->ucLastCommittedLogHdr);
Exit:

	// Need to close the .db file so that we can set the block size.
	// This will allow direct I/O to be used when accessing the file later.

	if( pCFileHdl)
	{
		(void)pDb->pSFileHdl->ReleaseFile( (FLMUINT)0, TRUE);
		pDb->pSFileHdl->SetBlockSize( pFile->FileHdr.uiBlockSize);
		pDb->pSFileHdl->SetDbVersion( pFile->FileHdr.uiVersionNum);
	}

	return( rc);
}

/***************************************************************************
Desc: This routine frees a CP_INFO structure and all associated data.
*****************************************************************************/
FSTATIC void flmFreeCPInfo(
	CP_INFO **	ppCPInfoRV)
{
	CP_INFO *	pCPInfo;

	if ((pCPInfo = *ppCPInfoRV) != NULL)
	{
		if (pCPInfo->pSFileHdl)
		{
			pCPInfo->pSFileHdl->Release();
		}
		if (pCPInfo->bStatsInitialized)
		{
			FlmFreeStats( &pCPInfo->Stats);
		}
		f_free( ppCPInfoRV);
	}
}

/***************************************************************************
Desc: This routine begins a thread that will do checkpoints for the
		passed in file.  It gives the thread its own FLAIM session and its
		own handle to the database.
*****************************************************************************/
RCODE flmStartCPThread(
	FFILE *		pFile)
{
	RCODE			rc = FERR_OK;
	CP_INFO *	pCPInfo = NULL;
	char			szThreadName[ F_PATH_MAX_SIZE];
	char			szBaseName[ F_FILENAME_SIZE];

	// Allocate a CP_INFO structure that will be passed into the
	// thread when it is created.

	if (RC_BAD( rc = f_calloc( (FLMUINT)(sizeof( CP_INFO)), &pCPInfo)))
	{
		goto Exit;
	}
	pCPInfo->pFile = pFile;

	// Allocate a super file handle.

	if ((pCPInfo->pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	// Set up the super file

	if (RC_BAD( rc = pCPInfo->pSFileHdl->Setup( pFile->pFileIdList, 
									pFile->pszDbPath,
									pFile->pszDataDir)))
	{
		goto Exit;
	}

	if (pFile->FileHdr.uiVersionNum)
	{
		pCPInfo->pSFileHdl->SetBlockSize( pFile->FileHdr.uiBlockSize);
		pCPInfo->pSFileHdl->SetDbVersion( pFile->FileHdr.uiVersionNum);
	}

	if (RC_BAD( rc = flmStatInit( &pCPInfo->Stats, FALSE)))
	{
		goto Exit;
	}
	pCPInfo->bStatsInitialized = TRUE;

	// Generate the thread name

	if (RC_BAD( rc = f_pathReduce( pFile->pszDbPath,
							szThreadName, szBaseName)))
	{
		goto Exit;
	}

	f_sprintf( (char *)szThreadName, 
		"Checkpoint (%s)", (char *)szBaseName);

	// Start the checkpoint thread.

	if (RC_BAD( rc = f_threadCreate( &pFile->pCPThrd,
		flmCPThread, szThreadName, 
		FLM_CHECKPOINT_THREAD_GROUP, 0, pCPInfo, NULL, 32000)))
	{
		goto Exit;
	}

	pFile->pCPInfo = pCPInfo;

Exit:
	if (RC_BAD( rc))
	{
		flmFreeCPInfo( &pCPInfo);
	}

	return( rc);
}

/****************************************************************************
Desc: This routine functions as a thread.  It monitors open files and
		frees up files which have been closed longer than the maximum
		close time.
****************************************************************************/
FSTATIC RCODE flmCPThread(
	F_Thread *		pThread)
{
	CP_INFO *			pCPInfo = (CP_INFO *)pThread->getParm1();
	FFILE *				pFile = pCPInfo->pFile;
	F_SuperFileHdl *	pSFileHdl = pCPInfo->pSFileHdl;
	FLMBOOL				bTerminate = FALSE;
	FLMBOOL				bForceCheckpoint;
	FLMINT				iForceReason;
	FLMUINT				uiCurrTime;
	DB_STATS *			pDbStats;

	pThread->setThreadStatus( FLM_THREAD_STATUS_SLEEPING);
	while (!bTerminate)
	{
		f_sleep( 1000);

		// See if we should terminate the thread - pointer to
		// super file handle should be NULL.

		if( pThread->getShutdownFlag())
		{
			// Set terminate flag to TRUE and then see if
			// we have been set up to do one final checkpoint
			// to flush dirty buffers to disk.

			bTerminate = TRUE;
		}

		// Determine if we need to force a checkpoint.

		bForceCheckpoint = FALSE;
		iForceReason = 0;
		uiCurrTime = (FLMUINT)FLM_GET_TIMER();
		if (bTerminate)
		{
			bForceCheckpoint = TRUE;
			iForceReason = CP_SHUTTING_DOWN_REASON;
		}
		else if (!pFile->pRfl->seeIfRflVolumeOk() ||
					RC_BAD( pFile->CheckpointRc))
		{
			bForceCheckpoint = TRUE;
			iForceReason = CP_RFL_VOLUME_PROBLEM;
		}
		else if ((FLM_ELAPSED_TIME( uiCurrTime, pFile->uiLastCheckpointTime) >=
					 gv_FlmSysData.uiMaxCPInterval) ||
					(!gv_FlmSysData.uiMaxCPInterval))
		{
			bForceCheckpoint = TRUE;
			iForceReason = CP_TIME_INTERVAL_REASON;
		}

		if (gv_FlmSysData.Stats.bCollectingStats)
		{

			// Statistics are being collected for the system.  Therefore,
			// if we are not currently collecting statistics in the
			// start.  If we were collecting statistics, but the
			// start time was earlier than the start time in the system
			// statistics structure, reset the statistics.

			if (!pCPInfo->Stats.bCollectingStats)
			{
				flmStatStart( &pCPInfo->Stats);
			}
			else if (pCPInfo->Stats.uiStartTime <
						gv_FlmSysData.Stats.uiStartTime)
			{
				flmStatReset( &pCPInfo->Stats, FALSE, FALSE);
			}
			(void)flmStatGetDb( &pCPInfo->Stats, pFile,
							0, &pDbStats, NULL, NULL);
		}
		else
		{
			pDbStats = NULL;
		}

		// Lock write object - If we are forcing a checkpoint
		// wait until we get the lock.  Otherwise, if we can't get
		// the lock without waiting, don't do anything.

		if (bForceCheckpoint ||
			 (gv_FlmSysData.SCacheMgr.uiMaxDirtyCache &&
			  (pFile->uiDirtyCacheCount + pFile->uiLogCacheCount) * pFile->FileHdr.uiBlockSize >
				gv_FlmSysData.SCacheMgr.uiMaxDirtyCache))
		{
			if (RC_BAD( dbWriteLock( pFile, pDbStats)))
			{

				// THIS SHOULD NEVER HAPPEN BECAUSE dbWriteLock will
				// wait forever for the lock!

				flmAssert( 0);
				continue;
			}
			pThread->setThreadStatus( "Forcing checkpoint");

			// Must wait for any RFL writes to complete.

			(void)pFile->pRfl->seeIfRflWritesDone( TRUE);
		}
		else
		{
			if( RC_BAD( dbWriteLock( pFile, pDbStats, 0)))
			{
				continue;
			}

			pThread->setThreadStatus( FLM_THREAD_STATUS_RUNNING);

			// See if we actually need to do the checkpoint.  If the
			// current transaction ID and the last checkpoint transaction
			// ID are the same, no updates have occurred that would require
			// a checkpoint to take place.

			if (FB2UD( &pFile->ucLastCommittedLogHdr [LOG_LAST_CP_TRANS_ID]) ==
				 FB2UD( &pFile->ucLastCommittedLogHdr [LOG_CURR_TRANS_ID]) ||
				 !pFile->pRfl->seeIfRflWritesDone( FALSE))
			{
				dbWriteUnlock( pFile, pDbStats);
				continue;
			}
		}

		// Do the checkpoint.

		(void)ScaDoCheckpoint( pDbStats, pSFileHdl, pFile, FALSE,
							bForceCheckpoint, iForceReason, 0, 0);
		if (pDbStats)
		{
			(void)flmStatUpdate( &gv_FlmSysData.Stats, &pCPInfo->Stats);
		}

		dbWriteUnlock( pFile, pDbStats);

		// Unlink FDB from the FFILE - will be relinked
		// by next thread that wakes us up.

		f_mutexLock( gv_FlmSysData.hShareMutex);

		// If we are terminating, we don't want the FFILE structure
		// to appear in the not-used list, even for a moment.
		// Therefore, we unlink it from that, if it got into it,
		// before we unlock the mutex.

		if (bTerminate)
		{
			flmUnlinkFileFromNUList( pFile);
		}

		// Unlock the mutex

		f_mutexUnlock( gv_FlmSysData.hShareMutex);

		// Set the thread's status

		pThread->setThreadStatus( FLM_THREAD_STATUS_SLEEPING);
	}

	pThread->setThreadStatus( FLM_THREAD_STATUS_TERMINATING);
	flmFreeCPInfo( &pCPInfo);
	return( FERR_OK);
}

/****************************************************************************
Desc: Recover a database on startup.
****************************************************************************/
FSTATIC RCODE flmDoRecover(
	FDB *				pDb,
	F_Restore *		pRestoreObj)
{
	RCODE				rc = FERR_OK;
	FFILE *			pFile = pDb->pFile;
	FLMBYTE *		pucLastCommittedLogHdr;

	// At this point, pFile->ucLastCommittedLogHdr contains the log header
	// that was read from disk, which will be the state of the
	// log header as of the last completed checkpoint.  Therefore,
	// we copy it into pFile->ucCheckpointLogHdr.

	pucLastCommittedLogHdr = &pFile->ucLastCommittedLogHdr [0];
	f_memcpy( pFile->ucCheckpointLogHdr, pucLastCommittedLogHdr,
				LOG_HEADER_SIZE);

	// Do a physical rollback on the database to restore the last
	// checkpoint.

	if (RC_BAD( rc = flmPhysRollback( pDb,
			(FLMUINT)FB2UD( &pucLastCommittedLogHdr [LOG_ROLLBACK_EOF]),
			(FLMUINT)FB2UD( &pucLastCommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]),
			TRUE,
			(FLMUINT)FB2UD( &pucLastCommittedLogHdr [LOG_LAST_CP_TRANS_ID]))))
	{
		goto Exit;
	}
	UD2FBA( 0, &pucLastCommittedLogHdr [LOG_PL_FIRST_CP_BLOCK_ADDR]);
	UD2FBA( (FLMUINT32)pFile->FileHdr.uiBlockSize,
				&pucLastCommittedLogHdr [LOG_ROLLBACK_EOF]);
	if (RC_BAD( rc = flmWriteLogHdr( pDb->pDbStats, pDb->pSFileHdl,
								pFile, pucLastCommittedLogHdr,
								pFile->ucCheckpointLogHdr, TRUE)))
	{
		goto Exit;
	}

	// Set uiFirstLogCPBlkAddress to zero to indicate that no
	// physical blocks have been logged for the current checkpoint.
	// The above call to flmPhysRollback will have set the log header
	// to the same thing.

	pFile->uiFirstLogCPBlkAddress = 0;

	// Set the ucCheckpointLogHdr to be the same as the log header

	f_memcpy( pFile->ucCheckpointLogHdr, pFile->ucLastCommittedLogHdr,
						LOG_HEADER_SIZE);

	// Open roll forward log and redo the transactions that
	// occurred since the last checkpoint, if any.

	if (RC_BAD( rc = pFile->pRfl->recover( pDb, pRestoreObj)))
	{
		goto Exit;
	}
Exit:
	return( rc);
}

/**************************************************************************
Desc: See if a given database URL name is a client/server connection.
		If so, return a pointer to the CS_SESSION for the URL.  Otherwise,
		return NULL.
**************************************************************************/
RCODE flmGetCSConnection(
	const char *		pszUrlName,
	CS_CONTEXT_p *		ppCSContextRV)
{
	RCODE					rc = FERR_OK;
	FCL_WIRE				Wire;
	const char *		pszHostName = NULL;
	FLMINT				iSubProtocol;
	CS_CONTEXT_p		pCSContext = NULL;
	FUrl_p				pUrl = NULL;
	FCS_IPIS *			pIpIStream = NULL;
	FCS_IPOS	*			pIpOStream = NULL;
	FCS_TCP_CLIENT *	pTcpClient = NULL;
	FLMINT				iIPPort = 0;
	FCS_BIOS *			pBIStream = NULL;
	FCS_BIOS *			pBOStream = NULL;
	FCS_DIS *			pIDataStream = NULL;
	FCS_DOS *			pODataStream = NULL;
	FLMUINT				uiAddrType;
	FLMUINT				uiClientVersion;

	*ppCSContextRV = NULL;

	/*
	Allocate a C/S context
	*/

	if (RC_BAD( rc = f_calloc( sizeof( CS_CONTEXT),	&pCSContext)))
	{
		goto Exit;
	}
	GedPoolInit( &pCSContext->pool, 8192);

	/* Create a URL out of the URL name. */

	if ((pUrl = f_new FUrl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = pUrl->SetUrl( pszUrlName)))
	{
		goto Exit;
	}

	if( pUrl->IsLocal())
	{
		goto Exit;		// Returns SUCCESS in rc and NULL in ppCSContextRV
	}

	/* Determine the sub-protocol to use. */

	iSubProtocol = pUrl->GetSubProtocol();
	if( iSubProtocol == NO_SUB_PROTOCOL)
	{
		goto Exit; // Returns SUCCESS in rc and NULL in ppCSContext
	}

	/*
	Get the address type.
	*/

	uiAddrType = pUrl->GetAddrType();

	if( iSubProtocol == TCP_SUB_PROTOCOL)
	{
		if( uiAddrType != FLM_CS_IP_ADDR)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		iIPPort = pUrl->GetIPPort();
		if( (pszHostName = pUrl->GetIPHost()) == NULL)
		{
			pszHostName = "localhost";
		}
	}
	else if( iSubProtocol == STREAM_SUB_PROTOCOL)
	{
		if( uiAddrType == FLM_CS_IP_ADDR)
		{
			iIPPort = pUrl->GetIPPort();
		}
	}
	else
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	if ((pIDataStream = f_new FCS_DIS) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if ((pODataStream = f_new FCS_DOS) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/* Configure the I/O streams. */

	if( iSubProtocol == TCP_SUB_PROTOCOL)
	{
		if ((pTcpClient = f_new FCS_TCP_CLIENT) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pTcpClient->openConnection( pszHostName, 
			iIPPort, 30, 1200)))
		{
			goto Exit;
		}

		if ((pIpIStream = f_new FCS_IPIS( pTcpClient)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if ((pIpOStream = f_new FCS_IPOS( pTcpClient)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (RC_BAD( rc = pODataStream->setup( pIpOStream)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pIDataStream->setup( pIpIStream)))
		{
			goto Exit;
		}
	}
	else if( iSubProtocol == STREAM_SUB_PROTOCOL)
	{
		if ((pBIStream = f_new FCS_BIOS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if ((pBOStream = f_new FCS_BIOS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		pBOStream->setEventHook( flmStreamEventDispatcher,
			(void *)pCSContext);

		if (RC_BAD( rc = pODataStream->setup( pBOStream)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = pIDataStream->setup( pBIStream)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	if (iSubProtocol == TCP_SUB_PROTOCOL)
	{
		pCSContext->pTcpClient = pTcpClient;
		pTcpClient = NULL;
		pCSContext->pIStream = pIpIStream;
		pIpIStream = NULL;
		pCSContext->pOStream = pIpOStream;
		pIpOStream = NULL;
	}
	else if (iSubProtocol == STREAM_SUB_PROTOCOL)
	{
		pCSContext->pIStream = pBIStream;
		pBIStream = NULL;
		pCSContext->pOStream = pBOStream;
		pBOStream = NULL;
	}
	else
	{
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto Exit;
	}

	pCSContext->pIDataStream = pIDataStream;
	pIDataStream = NULL;
	pCSContext->pODataStream = pODataStream;
	pODataStream = NULL;
	pCSContext->iSubProtocol = iSubProtocol;
	pCSContext->uiSessionId = FCS_INVALID_ID;
	f_memcpy( pCSContext->pucAddr, pUrl->GetAddress(), FLM_CS_MAX_ADDR_LEN);
	f_strncpy( pCSContext->pucUrl, pszUrlName, FLM_CS_MAX_ADDR_LEN - 1);
	pCSContext->pucUrl[ FLM_CS_MAX_ADDR_LEN - 1] = '\0'; // Truncate the string (if necessary)

	/* Configure the wire object */

	Wire.setContext( pCSContext);

	uiClientVersion = FCS_VERSION_1_1_1;

Retry_Connect:

	/* Send a request to open a session. */

	if (RC_BAD( rc = Wire.sendOpcode( FCS_OPCLASS_SESSION, FCS_OP_SESSION_OPEN)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( 
		WIRE_VALUE_CLIENT_VERSION, FCS_VERSION_1_1_1)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( 
		WIRE_VALUE_FLAGS, FCS_SESSION_GEDCOM_SUPPORT)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Exit;
	}

	/* Read the response. */
	
	if (RC_BAD( rc = Wire.read()))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.getRCode()))
	{

		// Try a lower version number.

		switch (uiClientVersion)
		{
			case FCS_VERSION_1_1_0:
				break;
			case FCS_VERSION_1_1_1:
				uiClientVersion = FCS_VERSION_1_1_0;
				goto Retry_Connect;
		}
		goto Exit;
	}

	pCSContext->bConnectionGood = TRUE;
	pCSContext->uiSessionId = Wire.getSessionId();
	pCSContext->uiSessionCookie = Wire.getSessionCookie();
	pCSContext->uiServerFlaimVer = Wire.getFlaimVersion();
	if( pCSContext->uiServerFlaimVer < FLM_VER_4_3)
	{
		/*
		Versions of FLAIM prior to 4.3 did not send the server's code
		version.  However, they all supported GEDCOM as a wire format.
		*/

		pCSContext->bGedcomSupport = TRUE;
	}
	else
	{
		pCSContext->bGedcomSupport = (Wire.getFlags() & FCS_SESSION_GEDCOM_SUPPORT)
													? TRUE 
													: FALSE;
	}
	*ppCSContextRV = pCSContext;
	pCSContext = NULL;

Exit:
	if (RC_BAD( rc) || pCSContext)
	{
		flmCloseCSConnection( &pCSContext);
	}
	if (pUrl)
	{
		pUrl->Release();
	}
	return( rc);
}

/**************************************************************************
Desc: Close a client/server connection.
**************************************************************************/
void flmCloseCSConnection(
	CS_CONTEXT_p *		ppCSContext)
{
	if( !(*ppCSContext))
	{
		return;
	}

	CS_CONTEXT_p	pCSContext = *ppCSContext;
	FCL_WIRE			Wire( pCSContext);

	// Send a message to the FLAIM server indicating we are closing
	// the session.

	if( (pCSContext->uiSessionId != FCS_INVALID_ID) &&
		 (pCSContext->bConnectionGood))
	{
		if( RC_BAD( Wire.sendOpcode(
			FCS_OPCLASS_SESSION, FCS_OP_SESSION_CLOSE)))
		{
			goto Clear_Session_ID;
		}

		if( RC_BAD( Wire.sendNumber(
			WIRE_VALUE_SESSION_ID, pCSContext->uiSessionId)))
		{
			goto Clear_Session_ID;
		}

		if( RC_BAD( Wire.sendNumber(
			WIRE_VALUE_SESSION_COOKIE, pCSContext->uiSessionCookie)))
		{
			goto Clear_Session_ID;
		}

		if( RC_BAD( Wire.sendTerminate()))
		{
			goto Clear_Session_ID;
		}

		/*
		Read the response.  Ignore the return code.
		*/

		(void)Wire.read();

Clear_Session_ID:
		pCSContext->uiSessionId = FCS_INVALID_ID;
	}

	/* Free all of the input and output streams and the URL. */

	if( pCSContext->pODataStream)
	{
		pCSContext->pODataStream->Release();
		pCSContext->pODataStream = NULL;
	}

	if( pCSContext->pIDataStream)
	{
		pCSContext->pIDataStream->Release();
		pCSContext->pIDataStream = NULL;
	}

	if( pCSContext->pOStream)
	{
		pCSContext->pOStream->Release();
		pCSContext->pOStream = NULL;
	}

	if( pCSContext->pIStream)
	{
		pCSContext->pIStream->Release();
		pCSContext->pIStream = NULL;
	}

	if( pCSContext->pTcpClient != NULL)
	{
		((FCS_TCP_CLIENT *)pCSContext->pTcpClient)->Release();
		pCSContext->pTcpClient = NULL;
	}

	GedPoolFree( &pCSContext->pool);
	f_free( ppCSContext);
}
