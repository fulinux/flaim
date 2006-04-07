//-------------------------------------------------------------------------
// Desc:	Debug logging routines.
// Tabs:	3
//
//		Copyright (c) 1999-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fldbglog.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_DBG_LOG

FSTATIC void _flmDbgLogFlush( void);

FSTATIC void _flmDbgOutputMsg(
	char *	pszMsg);

// Global data

F_MUTEX				g_hDbgLogMutex = F_MUTEX_NULL;
F_FileSystem *		g_pFileSystem = NULL;
F_FileHdl *			g_pLogFile = NULL;
char *				g_pszLogBuf = NULL;
FLMUINT				g_uiLogBufOffset = 0;
FLMUINT				g_uiLogFileOffset = 0;
FLMBOOL				g_bDbgLogEnabled = TRUE;

#define DBG_LOG_BUFFER_SIZE		((FLMUINT)512000)


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogInit( void)
{
	FLMBYTE			szLogPath[ 256];
	RCODE				rc	= FERR_OK;

	flmAssert( g_hDbgLogMutex == F_MUTEX_NULL);
	flmAssert( g_pFileSystem == NULL);

	// Allocate a buffer for the log

	if( RC_BAD( rc = f_alloc( 
		DBG_LOG_BUFFER_SIZE + 1024, &g_pszLogBuf)))
	{
		goto Exit;
	}

	// Create the mutex

	if( RC_BAD( f_mutexCreate( &g_hDbgLogMutex)))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Create a new file system object

	if( (g_pFileSystem = f_new F_FileSystem) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Build the file path

#ifdef FLM_NLM
	f_strcpy( szLogPath, "SYS:\\FLMDBG.LOG");
#else
	f_sprintf( szLogPath, "FLMDBG.LOG");
#endif

	// Create the file.

	if( RC_BAD( rc = g_pFileSystem->Create( szLogPath, 
		F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYNONE | F_IO_DIRECT, &g_pLogFile)))
	{

		// See if we can open the file and then truncate it.

		if( RC_OK( g_pFileSystem->Open( szLogPath,
							F_IO_RDWR | F_IO_SH_DENYNONE | F_IO_DIRECT, &g_pLogFile)))
		{
			if( RC_BAD( rc = g_pLogFile->Truncate( 0)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	flmAssert( RC_OK( rc));
}

/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogExit( void)
{
	if( g_bDbgLogEnabled)
	{
		// Output "Log End" message
		f_mutexLock( g_hDbgLogMutex);
		_flmDbgOutputMsg( "--- LOG END ---");
		f_mutexUnlock( g_hDbgLogMutex);
		
		// Flush the log
		flmDbgLogFlush();
	}

	// Free all resources

	if( g_hDbgLogMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &g_hDbgLogMutex);
	}

	if( g_pszLogBuf)
	{
		f_free( &g_pszLogBuf);
	}

	if( g_pLogFile)
	{
		g_pLogFile->Truncate( g_uiLogFileOffset + g_uiLogBufOffset);
		g_pLogFile->Close();
		g_pLogFile->Release();
		g_pLogFile = NULL;
	}

	if( g_pFileSystem)
	{
		g_pFileSystem->Release();
		g_pFileSystem = NULL;
	}
	g_bDbgLogEnabled = FALSE;
}


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogWrite(
	FLMUINT		uiFileId,
	FLMUINT		uiBlkAddress,
	FLMUINT		uiWriteAddress,
	FLMUINT		uiTransId,
	char *		pszEvent)
{
	char		pszTmpBuf[ 256];
	
	if( !g_bDbgLogEnabled)
		return;

	if( !uiWriteAddress)
	{
		f_sprintf( pszTmpBuf, "f%u b=%X t%u %s",
			(unsigned)uiFileId,
			(unsigned)uiBlkAddress, (unsigned)uiTransId, pszEvent);
	}
	else
	{
		f_sprintf( pszTmpBuf, "f%u b=%X a=%X t%u %s",
				(unsigned)uiFileId,
    			(unsigned)uiBlkAddress, (unsigned)uiWriteAddress,
				(unsigned)uiTransId, pszEvent);
	}
	f_mutexLock( g_hDbgLogMutex);
	_flmDbgOutputMsg( pszTmpBuf);
	f_mutexUnlock( g_hDbgLogMutex);
}


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogUpdate(
	FLMUINT		uiFileId,
	FLMUINT		uiTransId,
	FLMUINT		uiContainer,	// Zero if logging transaction begin, commit, abort
	FLMUINT		uiDrn,			// Zero if logging transaction begin, commit, abort
	RCODE			rc,
	char *		pszEvent)
{
	char		pszTmpBuf[ 256];
	char		szErr [12];
	
	if (!g_bDbgLogEnabled)
	{
		return;
	}
	if (RC_BAD( rc))
	{
		f_sprintf( szErr, " RC=%04X", (unsigned)rc);
	}
	else
	{
		szErr [0] = 0;
	}

	if( uiContainer)
	{
		f_sprintf( pszTmpBuf, "f%u t%u c%u d%u %s%s",
			(unsigned)uiFileId,
			(unsigned)uiTransId, (unsigned)uiContainer, 
			(unsigned)uiDrn, pszEvent, szErr);
	}
	else
	{
		f_sprintf( pszTmpBuf, "f%u t%u %s%s",
			(unsigned)uiFileId,
			(unsigned)uiTransId, pszEvent,
			szErr);
	}

	f_mutexLock( g_hDbgLogMutex);
	_flmDbgOutputMsg( pszTmpBuf);
	f_mutexUnlock( g_hDbgLogMutex);
}


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogMsg(
	char *		pszMsg)
{
	if (!g_bDbgLogEnabled)
		return;
	f_mutexLock( g_hDbgLogMutex);
	_flmDbgOutputMsg( pszMsg);
	f_mutexUnlock( g_hDbgLogMutex);
}


/****************************************************************************
Desc:
****************************************************************************/
void flmDbgLogFlush( void)
{
	f_mutexLock( g_hDbgLogMutex);
	_flmDbgLogFlush();
	f_mutexUnlock( g_hDbgLogMutex);
}


/****************************************************************************
Desc:
****************************************************************************/
FSTATIC void _flmDbgLogFlush( void)
{
	FLMUINT			uiBytesToWrite;
	FLMUINT			uiBytesWritten;
	char *			pszBufPtr = g_pszLogBuf;
	FLMUINT			uiTotalToWrite = g_uiLogBufOffset;
	RCODE				rc = FERR_OK;
	FLMUINT			uiBufferSize = DBG_LOG_BUFFER_SIZE + 1024;

	while( uiTotalToWrite)
	{
		if( uiTotalToWrite > 0xFE00)
		{
			uiBytesToWrite = 0xFE00;
		}
		else
		{
			uiBytesToWrite = uiTotalToWrite;
		}

		if( RC_BAD( rc = g_pLogFile->SectorWrite(
			g_uiLogFileOffset, uiBytesToWrite,
			pszBufPtr, uiBufferSize, NULL, &uiBytesWritten, FALSE)))
		{
			goto Exit;
		}

		flmAssert( uiBytesToWrite == uiBytesWritten);
		g_uiLogFileOffset += uiBytesWritten;
		pszBufPtr += uiBytesWritten;
		uiBufferSize -= uiBytesWritten;
		uiTotalToWrite -= uiBytesWritten;
	}

	if (g_uiLogBufOffset & 0x1FF)
	{
		if (g_uiLogBufOffset > 512)
		{
			f_memcpy( g_pszLogBuf,
				&g_pszLogBuf [g_uiLogBufOffset & 0xFFFFFE00],
					512);
			g_uiLogBufOffset &= 0x1FF;
		}
		g_uiLogFileOffset -= g_uiLogBufOffset;
	}
	else
	{
		g_uiLogBufOffset = 0;
	}

Exit:

	flmAssert( RC_OK( rc));
}


/****************************************************************************
Desc:
****************************************************************************/
void _flmDbgOutputMsg(
	char *		pszMsg)
{
	char *	pszBufPtr = &(g_pszLogBuf[ g_uiLogBufOffset]);

	f_sprintf( pszBufPtr, "%s\n", pszMsg);
	g_uiLogBufOffset += f_strlen( pszBufPtr);

	if( g_uiLogBufOffset >= DBG_LOG_BUFFER_SIZE)
	{
		_flmDbgLogFlush();
	}
}

#endif	// #ifdef FLM_DBG_LOG

/****************************************************************************
Desc:
****************************************************************************/
#if( (defined( FLM_NLM) && !defined( __MWERKS__)) || defined( FLM_OSX))
void gv_fldbglog()
{
}
#endif
