//------------------------------------------------------------------------------
// Desc: Native C routines to support C# DataVector class
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
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_Release(
	FLMUINT64	ui64This)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	if (pDataVector)
	{
		pDataVector->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_setDocumentID(
	FLMUINT64	ui64This,
	FLMUINT64	ui64DocumentID)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	pDataVector->setDocumentID( ui64DocumentID);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setID(
	FLMUINT64	ui64This,
	FLMUINT32	ui32ElementNumber,
	FLMUINT64	ui64ID)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setID( (FLMUINT)ui32ElementNumber, ui64ID));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setNameId(
	FLMUINT64	ui64This,
	FLMUINT32	ui32ElementNumber,
	FLMUINT32	ui32NameId,
	FLMBOOL		bIsAttr,
	FLMBOOL		bIsData)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setNameId( (FLMUINT)ui32ElementNumber,
					(FLMUINT)ui32NameId, bIsAttr, bIsData));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setULong(
	FLMUINT64	ui64This,
	FLMUINT32	ui32ElementNumber,
	FLMUINT64	ui64Value)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setUINT64( (FLMUINT)ui32ElementNumber, ui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setLong(
	FLMUINT64	ui64This,
	FLMUINT32	ui32ElementNumber,
	FLMINT64		i64Value)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setINT64( (FLMUINT)ui32ElementNumber, i64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setUInt(
	FLMUINT64	ui64This,
	FLMUINT32	ui32ElementNumber,
	FLMUINT32	ui32Value)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setUINT( (FLMUINT)ui32ElementNumber, (FLMUINT)ui32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setInt(
	FLMUINT64	ui64This,
	FLMUINT32	ui32ElementNumber,
	FLMINT32		i32Value)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setINT( (FLMUINT)ui32ElementNumber, (FLMINT)i32Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setString(
	FLMUINT64				ui64This,
	FLMUINT32				ui32ElementNumber,
	const FLMUNICODE *	puzValue)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setUnicode( (FLMUINT)ui32ElementNumber, puzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_setBinary(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	const void *	pvValue,
	FLMUINT32		ui32Len)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->setBinary( (FLMUINT)ui32ElementNumber, pvValue, (FLMUINT)ui32Len));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_setRightTruncated(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	pDataVector->setRightTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_setLeftTruncated(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	pDataVector->setLeftTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_clearRightTruncated(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	pDataVector->clearRightTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_clearLeftTruncated(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	pDataVector->clearLeftTruncated( (FLMUINT)ui32ElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMBOOL FLMAPI xflaim_DataVector_isRightTruncated(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->isRightTruncated( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMBOOL FLMAPI xflaim_DataVector_isLeftTruncated(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->isLeftTruncated( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT64 FLMAPI xflaim_DataVector_getDocumentID(
	FLMUINT64		ui64This)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->getDocumentID());
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT64 FLMAPI xflaim_DataVector_getID(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->getID( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DataVector_getNameId(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( (FLMUINT32)pDataVector->getNameId( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMBOOL FLMAPI xflaim_DataVector_isAttr(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->isAttr( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMBOOL FLMAPI xflaim_DataVector_isDataComponent(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->isDataComponent( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMBOOL FLMAPI xflaim_DataVector_isKeyComponent(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->isKeyComponent( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DataVector_getDataLength(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( (FLMUINT32)pDataVector->getDataLength( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP FLMUINT32 FLMAPI xflaim_DataVector_getDataType(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( (FLMUINT32)pDataVector->getDataType( (FLMUINT)ui32ElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_getULong(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	FLMUINT64 *		pui64Value)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->getUINT64( (FLMUINT)ui32ElementNumber, pui64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_getLong(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	FLMINT64 *		pi64Value)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->getINT64( (FLMUINT)ui32ElementNumber, pi64Value));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_getUInt(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	FLMUINT32 *		pui32Value)
{
	RCODE					rc;
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	FLMUINT				uiValue;
	
	rc = pDataVector->getUINT( (FLMUINT)ui32ElementNumber, &uiValue);
	*pui32Value = (FLMUINT32)uiValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_getInt(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	FLMINT32 *		pi32Value)
{
	RCODE					rc;
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	FLMINT				iValue;
	
	rc = pDataVector->getINT( (FLMUINT)ui32ElementNumber, &iValue);
	*pi32Value = (FLMINT32)iValue;
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_getString(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	FLMUNICODE **	ppuzValue)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	
	return( pDataVector->getUnicode( (FLMUINT)ui32ElementNumber, ppuzValue));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_getBinary(
	FLMUINT64		ui64This,
	FLMUINT32		ui32ElementNumber,
	FLMUINT32		ui32Len,
	void *			pvValue)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	FLMUINT				uiLength = (FLMUINT)ui32Len;
	
	if (RC_BAD( rc = pDataVector->getBinary( (FLMUINT)ui32ElementNumber,
		pvValue, &uiLength)))
	{
		goto Exit;
	}
	flmAssert( uiLength == (FLMUINT)ui32Len);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_outputKey(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32IndexNum,
	FLMBOOL			bOutputIds,
	FLMBYTE *		pucKey,
	FLMINT32 *		pi32Len)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	IF_Db *				pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT				uiLength;

	if (RC_BAD( rc = pDataVector->outputKey( pDb, (FLMUINT)ui32IndexNum,
			bOutputIds, pucKey, XFLM_MAX_KEY_SIZE, &uiLength)))
	{
		goto Exit;
	}
	*pi32Len = (FLMINT32)uiLength;
	
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_outputData(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32IndexNum,
	FLMBYTE *		pucData,
	FLMINT32			i32BufSize,
	FLMINT32 *		pi32Len)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	IF_Db *				pDb = (IF_Db *)((FLMUINT)ui64Db);
	FLMUINT				uiLength;

	if (RC_BAD( rc = pDataVector->outputData( pDb, (FLMUINT)ui32IndexNum,
			pucData, (FLMUINT)i32BufSize, &uiLength)))
	{
		goto Exit;
	}
	*pi32Len = (FLMINT32)uiLength;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_inputKey(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32IndexNum,
	FLMBYTE *		pucKey,
	FLMINT32			i32KeyLen)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	IF_Db *				pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pDataVector->inputKey( pDb, (FLMUINT)ui32IndexNum,
			pucKey, (FLMUINT)i32KeyLen));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP RCODE FLMAPI xflaim_DataVector_inputData(
	FLMUINT64		ui64This,
	FLMUINT64		ui64Db,
	FLMUINT32		ui32IndexNum,
	FLMBYTE *		pucData,
	FLMINT32			i32DataLen)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	IF_Db *				pDb = (IF_Db *)((FLMUINT)ui64Db);

	return( pDataVector->inputData( pDb, (FLMUINT)ui32IndexNum,
			pucData, (FLMUINT)i32DataLen));
}

/****************************************************************************
Desc:
****************************************************************************/
FLMEXTC FLMEXP void FLMAPI xflaim_DataVector_reset(
	FLMUINT64		ui64This)
{
	IF_DataVector *	pDataVector = ((IF_DataVector *)(FLMUINT)ui64This);
	pDataVector->reset();
}
