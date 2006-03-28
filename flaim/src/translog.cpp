//-------------------------------------------------------------------------
// Desc:	Rollback logging.
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
// $Id: translog.cpp 12315 2006-01-19 15:16:37 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC void lgWriteComplete(
	F_IOBuffer *	pIOBuffer);

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DBG_LOG
void scaLogWrite(
	FLMUINT		uiFFileId,
	FLMUINT		uiWriteAddress,
	FLMBYTE *	pucBlkBuf,
	FLMUINT		uiBufferLen,
	FLMUINT		uiBlockSize,
	char *		pszEvent)
{
	FLMUINT	uiOffset = 0;
	FLMUINT	uiBlkAddress;

	while (uiOffset < uiBufferLen)
	{
		uiBlkAddress = (FLMUINT)(GET_BH_ADDR( pucBlkBuf));

		// A uiWriteAddress of zero means we are writing exactly at the
		// block address - i.e., it is the data block, not the log block.

		flmDbgLogWrite( uiFFileId, uiBlkAddress,
							(FLMUINT)((uiWriteAddress)
										 ? uiWriteAddress + uiOffset
										 : uiBlkAddress),
			(FLMUINT)(FB2UD( &pucBlkBuf [BH_TRANS_ID])), pszEvent);
		uiOffset += uiBlockSize;
		pucBlkBuf += uiBlockSize;
	}
}
#endif

/****************************************************************************
Desc:	This is the callback routine that is called when a disk write is
		completed.
****************************************************************************/
FSTATIC void lgWriteComplete(
	F_IOBuffer *	pIOBuffer)
{
#ifdef FLM_DBG_LOG
	FFILE *		pFile = (FFILE *)pIOBuffer->getCompletionCallbackData( 0);
	FLMUINT		uiBlockSize = pFile->FileHdr.uiBlockSize;
	FLMUINT		uiLength = pIOBuffer->getBufferSize();
	char *		pszEvent;
#endif
	DB_STATS *	pDbStats = pIOBuffer->getDbStats();

#ifdef FLM_DBG_LOG
	pszEvent = (char *)(RC_OK( pIOBuffer->getCompletionCode())
							  ? (char *)"LGWRT"
							  : (char *)"LGWRT-FAIL");
	scaLogWrite( pFile->uiFFileId, 0, pIOBuffer->getBuffer(), uiLength,
							 uiBlockSize, pszEvent);
#endif

	if (pDbStats)
	{

		// Must lock mutex, because this may be called from async write
		// completion at any time.

		f_mutexLock( gv_FlmSysData.hShareMutex);
		pDbStats->LogBlockWrites.ui64ElapMilli += pIOBuffer->getElapTime();
		f_mutexUnlock( gv_FlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	This routine flushes a log buffer to the log file.
****************************************************************************/
RCODE lgFlushLogBuffer(
	DB_STATS *			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	FLMBOOL				bDoAsync)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesWritten;
	F_IOBuffer *	pAsyncBuffer;

	if (!bDoAsync)
	{
		pAsyncBuffer = NULL;
	}
	else
	{
		pAsyncBuffer = pFile->pCurrLogBuffer;
	}

	if (pDbStats)
	{
		pDbStats->bHaveStats = TRUE;
		pDbStats->LogBlockWrites.ui64Count++;
		pDbStats->LogBlockWrites.ui64TotalBytes += pFile->uiCurrLogWriteOffset;
	}

	pFile->pCurrLogBuffer->setCompletionCallback( lgWriteComplete);
	pFile->pCurrLogBuffer->setCompletionCallbackData( 0,
		(void *)pFile);
	pSFileHdl->setMaxAutoExtendSize( pFile->uiMaxFileSize);
	pSFileHdl->setExtendSize( pFile->uiFileExtendSize);
	pFile->pCurrLogBuffer->startTimer( pDbStats);

	// NOTE: No guarantee that pFile->pCurrLogBuffer will still be around
	// after the call to WriteBlock, unless we are doing
	// non-asynchronous write.

	rc = pSFileHdl->WriteBlock( pFile->uiCurrLogBlkAddr,
				pFile->uiCurrLogWriteOffset,
				pFile->pCurrLogBuffer->getBuffer(),
				pFile->pCurrLogBuffer->getBufferSize(),
				pAsyncBuffer, &uiBytesWritten);
				
	if (!pAsyncBuffer)
	{
		pFile->pCurrLogBuffer->notifyComplete( rc);
	}
	
	pFile->pCurrLogBuffer = NULL;

	if (RC_BAD( rc))
	{
		if (pDbStats)
		{
			pDbStats->uiWriteErrors++;
		}
		goto Exit;
	}
	
Exit:

	pFile->uiCurrLogWriteOffset = 0;
	pFile->pCurrLogBuffer = NULL;
	return( rc);
}

/****************************************************************************
Desc:	This routine writes a block to the log file.
****************************************************************************/
RCODE lgOutputBlock(
	DB_STATS	*			pDbStats,
	F_SuperFileHdl *	pSFileHdl,
	FFILE *				pFile,
	SCACHE *				pLogBlock,		// Cached log block.
	FLMBYTE *			pucBlk,			// Pointer to the corresponding modified
												// block in cache.  This block will be
												// modified to the logged version of
												// the block
	FLMBOOL				bDoAsync,		// Do asynchronous writes?
	FLMUINT *			puiLogEofRV)	// Returns log EOF
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiFilePos = *puiLogEofRV;
	FLMUINT		uiBlkSize = pFile->FileHdr.uiBlockSize;
	FLMBYTE *	pucLogBlk;
	FLMUINT		uiBlkAddress;
	FLMUINT		uiLogBufferSize;

	// Time for a new block file?
	
	if (FSGetFileOffset( uiFilePos) >= pFile->uiMaxFileSize)
	{
		FLMUINT	uiFileNumber;

		// Write out the current buffer, if it has anything in it.

		if (pFile->uiCurrLogWriteOffset)
		{
			if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl,
											pFile, bDoAsync)))
			{
				goto Exit;
			}
		}

		uiFileNumber = FSGetFileNumber( uiFilePos);

		if (!uiFileNumber)
		{
			uiFileNumber = FIRST_LOG_BLOCK_FILE_NUMBER(
									pFile->FileHdr.uiVersionNum);
		}
		else
		{
			uiFileNumber++;
		}

		if (uiFileNumber > MAX_LOG_BLOCK_FILE_NUMBER(
									pFile->FileHdr.uiVersionNum))
		{
			rc = RC_SET( FERR_DB_FULL);
			goto Exit;
		}

		if (RC_BAD( rc = pSFileHdl->CreateFile( uiFileNumber )))
		{
			goto Exit;
		}
		uiFilePos = FSBlkAddress( uiFileNumber, 0 );
	}

	// Copy the log block to the log buffer.

	if (!pFile->uiCurrLogWriteOffset)
	{
		pFile->uiCurrLogBlkAddr = uiFilePos;

		// Get a buffer for logging.
		//
		// NOTE: Buffers are not kept by the FFILE's buffer manager,
		// so once we are done with this buffer, it will be freed

		uiLogBufferSize = MAX_LOG_BUFFER_SIZE;

		for( ;;)
		{
			if (RC_BAD( rc = pFile->pBufferMgr->getBuffer( 
				&pFile->pCurrLogBuffer, uiLogBufferSize, uiLogBufferSize)))
			{
				// If we failed to get a buffer of the requested size,
				// reduce the buffer size by half and try again

				if( rc == FERR_MEM)
				{
					uiLogBufferSize /= 2;
					if( uiLogBufferSize < uiBlkSize)
					{
						goto Exit;
					}
					rc = FERR_OK;
					continue;
				}
				goto Exit;
			}
			break;
		}
	}

	// Copy data from log block to the log buffer

	pucLogBlk = pFile->pCurrLogBuffer->getBuffer() +
						pFile->uiCurrLogWriteOffset;
	f_memcpy( pucLogBlk, pLogBlock->pucBlk, uiBlkSize);

	// If we are logging this block for the current update
	// transaction, set the BEFORE IMAGE (BI) flag in the block header
	// so we will know that this block is a before image block that
	// needs to be restored when aborting the current update
	// transaction

	if (pLogBlock->ui16Flags & CA_WRITE_TO_LOG)
	{
		BH_SET_BI( pucLogBlk);
	}

	// If this is an index block, and it is encrypted, we need to encrypt
	// it before we calculate the checksum
	
	if (BH_GET_TYPE( pucLogBlk) != BHT_FREE && pucLogBlk[ BH_ENCRYPTED])
	{
		FLMUINT		uiBufLen = getEncryptSize( pucLogBlk);

		flmAssert( uiBufLen <= uiBlkSize);

		if (RC_BAD( rc = ScaEncryptBlock( pLogBlock->pFile,
													 pucLogBlk,
													 uiBufLen,
													 uiBlkSize)))
		{
			goto Exit;
		}
	}
	
	// Calculate the block checksum

	uiBlkAddress = GET_BH_ADDR( pucLogBlk);
	BlkCheckSum( pucLogBlk, CHECKSUM_SET, uiBlkAddress, uiBlkSize);

	// Set up for next log block write

	pFile->uiCurrLogWriteOffset += uiBlkSize;

	// If this log buffer is full, write it out

	if (pFile->uiCurrLogWriteOffset == 
		pFile->pCurrLogBuffer->getBufferSize())
	{
		if (RC_BAD( rc = lgFlushLogBuffer( pDbStats, pSFileHdl,
									pFile, bDoAsync)))
		{
			goto Exit;
		}
	}

	// Save the previous block address into the modified block's
	// block header area.  Also save the transaction id

	UD2FBA( (FLMUINT32)uiFilePos, &pucBlk [BH_PREV_BLK_ADDR]);
	f_memcpy( &pucBlk [BH_PREV_TRANS_ID], &pLogBlock->pucBlk [BH_TRANS_ID], 4);

	*puiLogEofRV = uiFilePos + uiBlkSize;

Exit:

	return( rc);
}
