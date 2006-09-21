//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DbInfo class
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

#include "xflaim.h"

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_Release(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	
	if (pDbInfo)
	{
		pDbInfo->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DbInfo_getNumCollections(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( (FLMUINT32)pDbInfo->getNumCollections());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DbInfo_getNumIndexes(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( (FLMUINT32)pDbInfo->getNumIndexes());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DbInfo_getNumLogicalFiles(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( (FLMUINT32)pDbInfo->getNumLogicalFiles());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT64 FLMAPI xflaim_DbInfo_getDatabaseSize(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( pDbInfo->getFileSize());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getAvailBlockStats(
	FLMUINT64	ui64This,
	FLMUINT64 *	pui64BytesUsed,
	FLMUINT32 *	pui32BlockCount,
	FLMINT32 *	pi32LastError,
	FLMUINT32 *	pui32NumErrors)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	FLMUINT		uiBlockCount;
	FLMUINT		uiNumErrors;

	pDbInfo->getAvailBlockStats( pui64BytesUsed, &uiBlockCount,
		pi32LastError, &uiNumErrors);
	if (pui32BlockCount)
	{
		*pui32BlockCount = (FLMUINT32)uiBlockCount;
	}
	if (pui32NumErrors)
	{
		*pui32NumErrors = (FLMUINT32)uiNumErrors;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getLFHBlockStats(
	FLMUINT64	ui64This,
	FLMUINT64 *	pui64BytesUsed,
	FLMUINT32 *	pui32BlockCount,
	FLMINT32 *	pi32LastError,
	FLMUINT32 *	pui32NumErrors)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	FLMUINT		uiBlockCount;
	FLMUINT		uiNumErrors;

	pDbInfo->getLFHBlockStats( pui64BytesUsed, &uiBlockCount,
		pi32LastError, &uiNumErrors);
	if (pui32BlockCount)
	{
		*pui32BlockCount = (FLMUINT32)uiBlockCount;
	}
	if (pui32NumErrors)
	{
		*pui32NumErrors = (FLMUINT32)uiNumErrors;
	}
}


/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getBTreeInfo(
	FLMUINT64		ui64This,
	FLMUINT32		ui32NthLogicalFile,
	FLMUINT32 *		pui32LfNum,
	FLMINT32 *		pi32LfType,
	FLMUINT32 *		pui32RootBlkAddress,
	FLMUINT32 *		pui32NumLevels)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	FLMUINT		uiLfNum;
	eLFileType	eLfType;
	FLMUINT		uiRootBlkAddress;
	FLMUINT		uiNumLevels;

	pDbInfo->getBTreeInfo( (FLMUINT)ui32NthLogicalFile, &uiLfNum,
		&eLfType, &uiRootBlkAddress, &uiNumLevels);
	if (pui32LfNum)
	{
		*pui32LfNum = (FLMUINT32)uiLfNum;
	}
	if (pi32LfType)
	{
		*pi32LfType = (FLMINT32)eLfType;
	}
	if (pui32RootBlkAddress)
	{
		*pui32RootBlkAddress = (FLMUINT32)uiRootBlkAddress;
	}
	if (pui32NumLevels)
	{
		*pui32NumLevels = (FLMUINT32)uiNumLevels;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getBTreeBlockStats(
	FLMUINT64		ui64This,
	FLMUINT32		ui32NthLogicalFile,
	FLMUINT32		ui32Level,
	FLMUINT64 *		pui64KeyCount,
	FLMUINT64 *		pui64BytesUsed,
	FLMUINT64 *		pui64ElementCount,
	FLMUINT64 *		pui64ContElementCount,
	FLMUINT64 *		pui64ContElmBytes,
	FLMUINT32 *		pui32BlockCount,
	FLMINT32 *		pi32LastError,
	FLMUINT32 *		pui32NumErrors)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	FLMUINT		uiBlockCount;
	FLMUINT		uiNumErrors;

	pDbInfo->getBTreeBlockStats( (FLMUINT)ui32NthLogicalFile, (FLMUINT)ui32Level, pui64KeyCount,
		pui64BytesUsed, pui64ElementCount, pui64ContElementCount, pui64ContElmBytes,
		&uiBlockCount, pi32LastError, &uiNumErrors);
	if (pui32BlockCount)
	{
		*pui32BlockCount = (FLMUINT32)uiBlockCount;
	}
	if (pui32NumErrors)
	{
		*pui32NumErrors = (FLMUINT32)uiNumErrors;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getDbHdr(
	FLMUINT64		ui64This,
	XFLM_DB_HDR *	pDbHdr)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	f_memcpy( pDbHdr, pDbInfo->getDbHdr(), sizeof( XFLM_DB_HDR));
}
