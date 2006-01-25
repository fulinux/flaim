//------------------------------------------------------------------------------
// Desc:	Check b-trees for physical integrity.
//
// Tabs:	3
//
//		Copyright (c) 1991-1992, 1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flchktr.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC RCODE chkReadBlkFromDisk(
	F_Db *				pDb,
	F_Database *		pDatabase,
	XFLM_DB_HDR *		pDbHdr,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiFilePos,
	F_BLK_HDR *			pBlkHdr);

/********************************************************************
Desc:	Read a block - try cache first, then disk if not in cache.
*********************************************************************/
RCODE F_DbCheck::blkRead(
	FLMUINT				uiBlkAddress,
	F_BLK_HDR **		ppBlkHdr,
	F_CachedBlock **	ppSCache,
	FLMINT *				piBlkErrCodeRV)
{
	RCODE		rc = NE_XFLM_OK;

	if (*ppSCache)
	{
		ScaReleaseCache( *ppSCache, FALSE);
		*ppSCache = NULL;
		*ppBlkHdr = NULL;
	}
	else if (*ppBlkHdr)
	{
		f_free( ppBlkHdr);
		*ppBlkHdr = NULL;
	}

	if (m_pDb->m_uiKilledTime)
	{
		rc = RC_SET( NE_XFLM_OLD_VIEW);
		goto Exit;
	}

	// Get the block from cache.

	if (RC_OK( rc = m_pDb->m_pDatabase->getBlock( m_pDb, NULL, 
		uiBlkAddress, NULL, ppSCache)))
	{
		*ppBlkHdr = (*ppSCache)->getBlockPtr();
	}
	else
	{
		// Try to read the block directly from disk.

		FLMUINT			uiBlkSize = m_pDb->m_pDatabase->m_uiBlockSize;
		FLMUINT64		ui64TransID;
		F_BLK_HDR *		pBlkHdr;
		FLMUINT64		ui64LastReadTransID;
		FLMUINT			uiPrevBlkAddr;
		FLMUINT			uiFilePos;

		// If we didn't get a corruption error, jump to exit.

		if( !F_DbSystem::_errorIsFileCorrupt( rc))
		{
			goto Exit;
		}

		// Allocate memory for the block.

		if( RC_BAD( rc = f_calloc( uiBlkSize, ppBlkHdr)))
		{
			goto Exit;
		}
		
		pBlkHdr = *ppBlkHdr;
		uiFilePos = uiBlkAddress;
		ui64TransID = m_pDb->m_ui64CurrTransID;
		ui64LastReadTransID = ~((FLMUINT64)0);

		// Follow version chain until we find version we need.

		for (;;)
		{
			if (RC_BAD( rc = chkReadBlkFromDisk(
									m_pDb,
									m_pDb->m_pDatabase,
									&m_pDb->m_pDatabase->m_lastCommittedDbHdr,
									m_pDb->m_pSFileHdl,
									uiFilePos, pBlkHdr)))
			{
				goto Exit;
			}

			// See if we can use the current version of the block, or if we
			// must go get a previous version.

			if (pBlkHdr->ui64TransID <= ui64TransID)
			{
				break;
			}

			// If the transaction ID is greater than or equal to the last
			// one we read, we have a corruption.  This test will keep us
			// from looping around forever.

			if (pBlkHdr->ui64TransID >= ui64LastReadTransID)
			{
				rc = RC_SET( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			ui64LastReadTransID = pBlkHdr->ui64TransID;

			// Block is too new, go for next older version.

			// If previous block address is same as current file position or
			// zero, we have a problem.

			uiPrevBlkAddr = (FLMUINT)pBlkHdr->ui32PriorBlkImgAddr;
			if (uiPrevBlkAddr == uiFilePos || !uiPrevBlkAddr)
			{
				rc = (m_pDb->m_uiKilledTime)
					  ? RC_SET( NE_XFLM_OLD_VIEW)
					  : RC_SET( NE_XFLM_DATA_ERROR);
				goto Exit;
			}
			uiFilePos = uiPrevBlkAddr;
		}

		// See if we even got the block we thought we wanted.

		if ((FLMUINT)pBlkHdr->ui32BlkAddr != uiBlkAddress)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
			goto Exit;
		}
	}

Exit:

	*piBlkErrCodeRV = 0;
	if (RC_BAD( rc))
	{
		switch (rc)
		{
			case NE_XFLM_DATA_ERROR:
				*piBlkErrCodeRV = FLM_COULD_NOT_SYNC_BLK;
				break;
			case NE_XFLM_BLOCK_CRC:
				*piBlkErrCodeRV = FLM_BAD_BLK_CHECKSUM;
				break;
		}
	}
	return( rc);
}

/************************************************************************
Desc:	Read a block from disk
*************************************************************************/
FSTATIC RCODE chkReadBlkFromDisk(
	F_Db *				pDb,
	F_Database *		pDatabase,
	XFLM_DB_HDR *		pDbHdr,
	F_SuperFileHdl *	pSFileHdl,
	FLMUINT				uiFilePos,
	F_BLK_HDR *			pBlkHdr
	)
{
	RCODE		   rc = NE_XFLM_OK;
	FLMUINT		uiBytesRead;
	FLMUINT		uiBlkSize = (FLMUINT)pDbHdr->ui16BlockSize;
	F_Dict *		pDict;

	if (RC_BAD( rc = pSFileHdl->ReadBlock( uiFilePos, uiBlkSize,
								(FLMBYTE *)pBlkHdr, &uiBytesRead)))
	{
		if (rc == NE_XFLM_IO_END_OF_FILE)
		{
			rc = RC_SET( NE_XFLM_DATA_ERROR);
		}
		goto Exit;
	}

	if (uiBytesRead < uiBlkSize)
	{
		rc = RC_SET( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	if (RC_BAD( rc = flmPrepareBlockForUse( uiBlkSize, pBlkHdr)))
	{
		goto Exit;
	}
	
	// Decrypt the block if encrypted

	if (RC_BAD( rc = pDb->getDictionary( &pDict)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pDatabase->decryptBlock( pDict, (FLMBYTE *)pBlkHdr)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/********************************************************************
Desc:	Report an error
*********************************************************************/
RCODE F_DbCheck::chkReportError(
	FLMINT			iErrCode,
	FLMUINT			uiErrLocale,
	FLMUINT			uiErrLfNumber,
	FLMUINT			uiErrLfType,
	FLMUINT			uiErrBTreeLevel,
	FLMUINT			uiErrBlkAddress,
	FLMUINT			uiErrParentBlkAddress,
	FLMUINT			uiErrElmOffset,
	FLMUINT64		ui64ErrNodeId)
{
	XFLM_CORRUPT_INFO		CorruptInfo;
	FLMBOOL					bFixErr;

	CorruptInfo.iErrCode = iErrCode;
	CorruptInfo.uiErrLocale = uiErrLocale;
	CorruptInfo.uiErrLfNumber = uiErrLfNumber;
	CorruptInfo.uiErrLfType = uiErrLfType;
	CorruptInfo.uiErrBTreeLevel = uiErrBTreeLevel;
	CorruptInfo.uiErrBlkAddress = uiErrBlkAddress;
	CorruptInfo.uiErrParentBlkAddress = uiErrParentBlkAddress;
	CorruptInfo.uiErrElmOffset = uiErrElmOffset;
	CorruptInfo.ui64ErrNodeId = ui64ErrNodeId;
	CorruptInfo.ifpErrIxKey = NULL;

	if (m_pDbCheckStatus && RC_OK( m_LastStatusRc))
	{
		bFixErr = FALSE;
		m_LastStatusRc = m_pDbCheckStatus->reportCheckErr( &CorruptInfo, &bFixErr);
	}

	if (iErrCode != FLM_OLD_VIEW)
	{
		m_bPhysicalCorrupt = TRUE;
		m_uiFlags &= ~XFLM_DO_LOGICAL_CHECK;
	}

	flmAssert( 0);
	return( m_LastStatusRc);
}

