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
FLMEXTC FLMEXP FLMUINT FLMAPI xflaim_DbInfo_getNumCollections(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( pDbInfo->getNumCollections());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT FLMAPI xflaim_DbInfo_getNumIndexes(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( pDbInfo->getNumIndexes());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT FLMAPI xflaim_DbInfo_getNumLogicalFiles(
	FLMUINT64	ui64This)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	return( pDbInfo->getNumLogicalFiles());
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
	FLMUINT *	puiBlockCount,
	FLMINT *		piLastError,
	FLMUINT *	puiNumErrors)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	pDbInfo->getAvailBlockStats( pui64BytesUsed, puiBlockCount,
		piLastError, puiNumErrors);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getLFHBlockStats(
	FLMUINT64	ui64This,
	FLMUINT64 *	pui64BytesUsed,
	FLMUINT *	puiBlockCount,
	FLMINT *		piLastError,
	FLMUINT *	puiNumErrors)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	pDbInfo->getLFHBlockStats( pui64BytesUsed, puiBlockCount,
		piLastError, puiNumErrors);
}


/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getBTreeInfo(
	FLMUINT64		ui64This,
	FLMUINT			uiNthLogicalFile,
	FLMUINT *		puiLfNum,
	eLFileType *	peLfType,
	FLMUINT *		puiRootBlkAddress,
	FLMUINT *		puiNumLevels)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	pDbInfo->getBTreeInfo( uiNthLogicalFile, puiLfNum,
		peLfType, puiRootBlkAddress, puiNumLevels);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DbInfo_getBTreeBlockStats(
	FLMUINT64		ui64This,
	FLMUINT			uiNthLogicalFile,
	FLMUINT			uiLevel,
	FLMUINT64 *		pui64KeyCount,
	FLMUINT64 *		pui64BytesUsed,
	FLMUINT64 *		pui64ElementCount,
	FLMUINT64 *		pui64ContElementCount,
	FLMUINT64 *		pui64ContElmBytes,
	FLMUINT *		puiBlockCount,
	FLMINT *			piLastError,
	FLMUINT *		puiNumErrors)
{
	IF_DbInfo *	pDbInfo = ((IF_DbInfo *)(FLMUINT)ui64This);
	pDbInfo->getBTreeBlockStats( uiNthLogicalFile, uiLevel, pui64KeyCount,
		pui64BytesUsed, pui64ElementCount, pui64ContElementCount, pui64ContElmBytes,
		puiBlockCount, piLastError, puiNumErrors);
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
