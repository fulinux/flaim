//-------------------------------------------------------------------------
// Desc:	Methods for the FlmRecord class
// Tabs:	3
//
//		Copyright (c) 1999-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: frec.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_NLM) && !defined( __MWERKS__)
	// Disable "Warning! W549: col(XX) 'sizeof' operand contains
	// compiler generated information"
	
	#pragma warning 549 9
#endif

FSTATIC RCODE importTree( 
	F_FileHdl *				pFileHdl,	
	char **					pBuf,
	FLMUINT					uiBufSize,
	F_NameTable *			pNameTable,
	FlmRecord *				pRec);

FSTATIC RCODE importField(
	FLMUINT					uiLevel,
	GED_STREAM *			pGedStream,
	F_NameTable *			pNameTable,	
	FlmRecord *				pRec);

FLMBYTE arr[ 16] = 
{ 
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xC0,
  0x7F, 0xFF, 0xFF, 0xE0, 0x7F, 0xFF, 0xFF, 0xE0
};

#define f_isalnum(c) \
	((c) < 128 ? (( ((FLMBYTE)(arr[(c) >> 3])) << ((c) & 0x07)) & 0x80) : 0)

#define f_isdigit(c) \
	((c) < 60 ?  (( ((FLMBYTE)(arr[(c) >> 3])) << ((c) & 0x07)) & 0x80) : 0)

#define f_isalpha(c) \
	((c) < 128 && (c) > 58 ? (( ((FLMBYTE)(arr[(c) >> 3])) << \
		((c) & 0x07)) & 0x80) : 0)

/*****************************************************************************
Desc:
*****************************************************************************/
FlmRecord::FlmRecord() 
{
	m_pucBuffer = NULL;
	m_uiBufferSize = 0;
	m_uiFldTblSize = 0;
	m_uiFlags = 0;
	clear();
}

/*****************************************************************************
Desc:
*****************************************************************************/
FlmRecord::~FlmRecord()
{
	flmAssert( m_uiFlags & RCA_OK_TO_DELETE);

	if( m_pucBuffer)
	{
		flmAssert( *((FlmRecord **)m_pucBuffer) == this);
		gv_FlmSysData.RCacheMgr.pRecBufAlloc->freeBuf( 
			m_uiBufferSize, &m_pucBuffer);
	}
}

/*****************************************************************************
Desc:		Makes a copy of self and returns pointer to the copy.
*****************************************************************************/
FlmRecord * FlmRecord::copy( void)
{
	RCODE				rc	= FERR_OK;
	FlmRecord *		pNewRec = NULL;
	FLMBOOL			bHeapAlloc = FALSE;
	
#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		goto Exit;
	}
#endif

	if( (pNewRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( m_uiBufferSize)
	{
		if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->allocBuf( 
			m_uiBufferSize, &pNewRec, sizeof( FlmRecord *), &pNewRec->m_pucBuffer,
			&bHeapAlloc)))
		{
			goto Exit;
		}
	
		f_memcpy( &pNewRec->m_pucBuffer[ sizeof( FlmRecord *)], 
			&m_pucBuffer[ sizeof( FlmRecord *)], 
			m_uiBufferSize - sizeof( FlmRecord *));
			
		if( bHeapAlloc)
		{
			pNewRec->m_uiFlags |= RCA_HEAP_BUFFER;
		}
	}

	pNewRec->m_uiBufferSize = m_uiBufferSize;
	pNewRec->m_uiContainerID = m_uiContainerID;
	pNewRec->m_uiRecordID = m_uiRecordID;
	pNewRec->m_uiFldTblSize = m_uiFldTblSize;
	pNewRec->m_uiFldTblOffset = m_uiFldTblOffset;
	pNewRec->m_uiDataBufOffset = m_uiDataBufOffset;
	pNewRec->m_uiFirstAvail = m_uiFirstAvail;
	pNewRec->m_uiAvailFields = m_uiAvailFields;
	pNewRec->m_bHolesInData = m_bHolesInData;

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = pNewRec->checkRecord()))
	{
		goto Exit;
	}
#endif

	pNewRec->compressMemory();

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = pNewRec->checkRecord()))
	{
		goto Exit;
	}
#endif

Exit:

	if( RC_BAD( rc) && pNewRec)
	{
		pNewRec->Release();
		pNewRec = NULL;
	}
	
	return( pNewRec);
}

/*****************************************************************************
Desc: 	Return a existing record to a new state (no fields)
*****************************************************************************/
RCODE FlmRecord::clear(
	FLMBOOL			bReleaseMemory)
{
	RCODE		rc = FERR_OK;

	if( isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( bReleaseMemory)
	{
		if( m_pucBuffer)
		{
			gv_FlmSysData.RCacheMgr.pRecBufAlloc->freeBuf( 
				m_uiBufferSize, &m_pucBuffer);
		}

		m_uiBufferSize = 0;
		m_uiFldTblSize = 0;
	}

	m_uiFlags = 0;
	m_uiContainerID = 0;
	m_uiRecordID = 0;
	m_uiFldTblOffset = 0;
	m_uiDataBufOffset = 0;
	m_uiFirstAvail = 0;
	m_uiAvailFields = 0;
	m_bHolesInData = FALSE;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Retrieve values from a context field.
*****************************************************************************/
RCODE FlmRecord::getRecPointer(
	void *		pvField,
	FLMUINT *	puiRecPointer)
{
	RCODE			rc = FERR_OK;
	FlmField *	pField = getFieldPointer( pvField);
	
	*puiRecPointer = 0xFFFFFFFF;

	if( !pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	
	if( isEncryptedField( pField) &&
			!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if( getFieldDataType( pField) != FLM_CONTEXT_TYPE)
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	if( getFieldDataLength( pField) == 4)
	{
		*puiRecPointer = (FLMUINT)(FB2UD( getDataPtr( pField)));
	}
	else
	{
		flmAssert( getFieldDataLength( pField) == 0);
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc: 	Create a FlmBlob object
*****************************************************************************/
RCODE FlmRecord::getBlob(
	void *			pvField,
	FlmBlob **		ppBlob)
{
	RCODE				rc = FERR_OK;
	FlmField *		pField = getFieldPointer( pvField);
	FLMBYTE *		pucData;
	FLMUINT			uiDataLen;
	FlmBlobImp *	pNewBlob = NULL;
	
	*ppBlob = NULL;
	
	if( !pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	
	if( isEncryptedField( pField) &&
			!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	if( (uiDataLen = getFieldDataLength( pField)) != 0 &&
		(getFieldDataType( pField) == FLM_BLOB_TYPE))
	{
		if( (pNewBlob = f_new FlmBlobImp) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		*ppBlob = pNewBlob;

		if( (pucData = pNewBlob->getImportDataPtr( uiDataLen)) != NULL)
		{
			f_memcpy( pucData, getDataPtr( pField), uiDataLen);
		}
		else
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setINT(
	void *		pvField,
	FLMINT		iNumber,
	FLMUINT		uiEncId)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucData;
	FLMBYTE			ucStorageBuf[ F_MAX_NUM_BUF + 1];
	FLMUINT			uiStorageLen;
	FLMUINT			uiEncDataLen = 0;
	FLMUINT			uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	uiStorageLen = sizeof( ucStorageBuf);
	
	if( RC_BAD( rc = FlmINT2Storage( iNumber, &uiStorageLen, ucStorageBuf)))
	{
		goto Exit;
	}
	
	if( uiEncId)
	{
		// For encrypted fields, we want to make sure we allocate
		// enough space for the encrypted data too.  The data does 
		// not get encrypted until the call to FlmRecordModify or
		// FlmRecordAdd.

		if( uiStorageLen % 16)
		{
			uiEncDataLen = uiStorageLen + (16 - (uiStorageLen % 16));
		}
		else
		{
			uiEncDataLen = uiStorageLen;
		}
		
		uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

	if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_NUMBER_TYPE,
		uiStorageLen, uiEncDataLen, uiEncId, uiEncFlags, &pucData, NULL)))
	{
		goto Exit;
	}

	f_memcpy( pucData, ucStorageBuf, uiStorageLen);

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setUINT(
	void *			pvField,
	FLMUINT			uiNumber,
	FLMUINT			uiEncId)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucData;
	FLMBYTE			ucStorageBuf[ F_MAX_NUM_BUF + 1];
	FLMUINT			uiStorageLen;
	FLMUINT			uiEncDataLen = 0;
	FLMUINT			uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	uiStorageLen = sizeof( ucStorageBuf);
	if( RC_BAD( rc = FlmUINT2Storage( uiNumber, &uiStorageLen, ucStorageBuf)))
	{
		goto Exit;
	}

	if( uiEncId)
	{
		// For encrypted fields, we want to make sure we allocate
		// enough space for the encrypted data too.  The data does 
		// not get encrypted until the call to FlmRecordModify or
		// FlmRecordAdd.

		if( uiStorageLen % 16)
		{
			uiEncDataLen = uiStorageLen + (16 - (uiStorageLen % 16));
		}
		else
		{
			uiEncDataLen = uiStorageLen;
		}
		
		uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

	if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_NUMBER_TYPE,
		uiStorageLen, uiEncDataLen, uiEncId, uiEncFlags, &pucData, NULL)))
	{
		goto Exit;
	}

	f_memcpy( pucData, ucStorageBuf, uiStorageLen);

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setRecPointer(
	void *		pvField,
	FLMUINT		uiRecPointer,
	FLMUINT		uiEncId)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucData;
	FLMUINT		uiEncDataLen = 0;
	FLMUINT		uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	if( uiEncId)
	{
		// For encrypted fields, we want to make sure we allocate
		// enough space for the encrypted data too.  The data does 
		// not get encrypted until the call to FlmRecordModify or
		// FlmRecordAdd.
		
		uiEncDataLen = 16;
		uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

	if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_CONTEXT_TYPE,
		4, uiEncDataLen, uiEncId, uiEncFlags, &pucData, NULL)))
	{
		goto Exit;
	}

	UD2FBA( (FLMUINT32)uiRecPointer, pucData);

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setUnicode(
	void *					pvField,
	const FLMUNICODE *	pUnicode,
	FLMUINT					uiEncId)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucData;
	FLMUINT		uiBufLen;
	FLMUINT		uiEncDataLen = 0;
	FLMUINT		uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	// A NULL or empty pUnicode string is allowed - on those 
	// cases just set the field type.

	if( !pUnicode || *pUnicode == 0)
	{
		// Field may have had a value pointer that now 
		// needs to be set to NULL

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_TEXT_TYPE,
			0, 0, 0, 0, &pucData, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		uiBufLen = FlmGetUnicodeStorageLength( pUnicode);

		if( uiEncId)
		{
			// For encrypted fields, we want to make sure we allocate
			// enough space for the encrypted data too.  The data does 
			// not get encrypted until the call to FlmRecordModify or
			// FlmRecordAdd.

			if( uiBufLen % 16)
			{
				uiEncDataLen = uiBufLen + (16 - (uiBufLen % 16));
			}
			else
			{
				uiEncDataLen = uiBufLen;
			}
			
			uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
		}

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_TEXT_TYPE,
			uiBufLen, uiEncDataLen, uiEncId, uiEncFlags, &pucData, NULL)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = FlmUnicode2Storage( pUnicode, &uiBufLen, pucData)))
		{
			goto Exit;
		}
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setNative(
	void *			pvField,
	const char *	pszString,
	FLMUINT			uiEncId)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucData;
	FLMUINT			uiBufLen;
	FLMUINT			uiEncDataLen = 0;
	FLMUINT			uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	if( !pszString || *pszString == 0)
	{
		// Field may have had a value pointer that now 
		// needs to be set to NULL

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_TEXT_TYPE,
			0, 0, 0, 0, &pucData, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		uiBufLen = FlmGetNativeStorageLength( pszString);

		if( uiEncId)
		{
			// For encrypted fields, we want to make sure we allocate
			// enough space for the encrypted data too.  The data does 
			// not get encrypted until the call to FlmRecordModify or
			// FlmRecordAdd.

			if( uiBufLen % 16)
			{
				uiEncDataLen = uiBufLen + (16 - (uiBufLen % 16));
			}
			else
			{
				uiEncDataLen = uiBufLen;
			}
			
			uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
		}

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_TEXT_TYPE,
			uiBufLen, uiEncDataLen, uiEncId, uiEncFlags, &pucData, NULL)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = FlmNative2Storage( pszString, &uiBufLen, pucData)))
		{
			goto Exit;
		}
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}
	 
/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setBinary(
	void *				pvField,
	const void *		pvBuf,
	FLMUINT				uiBufLen,
	FLMUINT				uiEncId)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucData;
	FLMUINT			uiEncDataLen = 0;
	FLMUINT			uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	if( !uiBufLen)
	{
		// Field may have had a value pointer that now 
		// needs to be set to NULL

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), 
			FLM_BINARY_TYPE, 0, 0, 0, 0, &pucData, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		if( uiEncId)
		{
			// For encrypted fields, we want to make sure we allocate
			// enough space for the encrypted data too.  The data does 
			// not get encrypted until the call to FlmRecordModify or
			// FlmRecordAdd.

			if( uiBufLen % 16)
			{
				uiEncDataLen = uiBufLen + (16 - (uiBufLen % 16));
			}
			else
			{
				uiEncDataLen = uiBufLen;
			}
			
			uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
		}

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField),
			FLM_BINARY_TYPE, uiBufLen, uiEncDataLen, uiEncId, uiEncFlags,
			&pucData, NULL)))
		{
			goto Exit;
		}
	
		f_memcpy( pucData, pvBuf, uiBufLen);
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Copies the BLOB control data into the record.
*****************************************************************************/
RCODE FlmRecord::setBlob( 
	void *			pvField,
	FlmBlob *		pBlob,
	FLMUINT			uiEncId)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucData;
	FLMUINT			uiDataLen = ((FlmBlobImp *)pBlob)->getDataLength();
	FLMUINT			uiEncDataLen = 0;
	FLMUINT			uiEncFlags = 0;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

	if( !uiDataLen)
	{
		// Field may have had a value pointer that now 
		// needs to be set to NULL

		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField), FLM_BLOB_TYPE,
			0, 0, 0, 0, &pucData, NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		if( uiEncId)
		{
			// For encrypted fields, we want to make sure we allocate
			// enough space for the encrypted data too.  The data does 
			// not get encrypted until the call to FlmRecordModify or
			// FlmRecordAdd.

			if( uiDataLen % 16)
			{
				uiEncDataLen = uiDataLen + (16 - (uiDataLen % 16));
			}
			else
			{
				uiEncDataLen = uiDataLen;
			}
			
			uiEncFlags = FLD_HAVE_DECRYPTED_DATA;
		}
	
		if( RC_BAD( rc = getNewDataPtr( getFieldPointer( pvField),
			FLM_BLOB_TYPE, uiDataLen, uiEncDataLen, uiEncId, uiEncFlags,
			&pucData, NULL)))
		{
			goto Exit;
		}
	
		f_memcpy( pucData, ((FlmBlobImp *)pBlob)->getDataPtr(), uiDataLen);
	}

#ifdef FLM_CHECK_RECORD
	if( RC_BAD( rc = checkRecord()))
	{
		flmAssert( 0);
		goto Exit;
	}
#endif

Exit:

	return( rc);
}

/*****************************************************************************
Desc: 	Create a new field at the specified position. Following the insert
			the current record position will be positioned on the new field.
*****************************************************************************/
RCODE FlmRecord::insert(
	void *		pvField,
	FLMUINT		uiInsertAt,
	FLMUINT		uiFieldID,
	FLMUINT		uiDataType,
	void **		ppvField)
{
	RCODE			rc = FERR_OK;
	FlmField *	pField;
	FlmField *	pNewField = NULL;
	FlmField *	pTmpField;
	FLMUINT		uiLevel;

	if( isReadOnly() || isCached() || !uiFieldID)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( !pvField)
	{
		pField = getFirstField();
	}
	else
	{
		pField = getFieldPointer( pvField);
	}

	uiLevel = getFieldLevel( pField);

	// User is adding first field (no fields in record)

	if( !pField)
	{
		if( RC_BAD( rc = createField( NULL, &pNewField)))
		{
			goto Exit;
		}

		goto Exit;
	}

	// Perform desired insert.

	switch( uiInsertAt)
	{
		case INSERT_PREV_SIB:
		{
			if( (pField = prevField( pField)) == NULL)
			{
				flmAssert( 0);
				rc = RC_SET( FERR_ILLEGAL_OP);
				goto Exit;
			}

			if( RC_BAD( rc = createField( pField, &pNewField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = setFieldLevel( pNewField, uiLevel)))
			{
				goto Exit;
			}

			break;
		}

		case INSERT_NEXT_SIB:
		{
			// If current field has children then we need to position to the end
			// of it's sub-tree before before inserting the new next sibling.

			FlmField * pSubTreeEnd = lastSubTreeField( pField);

			if( RC_BAD( rc = createField(
				pSubTreeEnd ? pSubTreeEnd : pField, &pNewField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = setFieldLevel( pNewField, uiLevel)))
			{
				goto Exit;
			}

			break;
		}

		case INSERT_FIRST_CHILD:
		{
First_Child:
			if( RC_BAD( rc = createField( pField, &pNewField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = setFieldLevel( pNewField, uiLevel + 1)))
			{
				goto Exit;
			}

			break;
		}

		case INSERT_LAST_CHILD:
		{
			if( (pTmpField = lastSubTreeField( pField)) != NULL)
			{
				if( RC_BAD( rc = createField( pTmpField, &pNewField)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = setFieldLevel( pNewField, uiLevel + 1)))
				{
					goto Exit;
				}
			}
			else
			{
				// There are no children, so add as first child

				goto First_Child;
			}

			break;
		}

		default:
		{
			flmAssert( 0);
			break;
		}
	}

Exit:	

	if( pNewField) 
	{
		pNewField->ui16FieldID = (FLMUINT16)uiFieldID;
		setFieldDataType( pNewField, uiDataType);
	}

	if( ppvField)
	{
		*ppvField = getFieldVoid( pNewField);
	}

	return( rc);
}

/*****************************************************************************
Desc:		Returns the last field within a subtree.
*****************************************************************************/
FlmField * FlmRecord::lastSubTreeField(
	FlmField *		pField)
{
	FlmField *	pTempField = lastChildField( pField);
	FlmField *	pLastChild = NULL;
	FLMUINT		uiStartLevel = getFieldLevel( pField);

	for( ; pTempField && getFieldLevel( pTempField) > uiStartLevel;
			pTempField = nextField( pTempField))
	{
		pLastChild = pTempField;
	}

#ifdef FLM_DEBUG
	if( pLastChild)
	{
		flmAssert( pLastChild->ui16FieldID);
	}
#endif

	return( pLastChild);
}

/*****************************************************************************
Desc:		Special level-based insert. 
Note:		The level must be a value between 0 - ((last field)->level + 1)
*****************************************************************************/
RCODE FlmRecord::insertLast(
	FLMUINT		uiLevel,
	FLMUINT		uiFieldID,
	FLMUINT		uiDataType,
	void **		ppvField)
{
	RCODE			rc = FERR_OK;
	FlmField *	pField = NULL;
	FlmField *	pLastField;
	
	if( isReadOnly() || isCached() || !uiFieldID)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Insert new field following current last field

	pLastField = getLastField();

#ifdef FLM_DEBUG
	if( pLastField && uiLevel > getFieldLevel( pLastField) + 1)
	{
		flmAssert( 0);
	}
#endif

	if( RC_BAD( rc = createField( pLastField, &pField)))
	{
		goto Exit;
	}

	// Set up the new field and set as the current field

	pField->ui16FieldID = (FLMUINT16)uiFieldID;
	setFieldDataType( pField, uiDataType);

	if( RC_BAD( rc = setFieldLevel( pField, uiLevel)))
	{
		goto Exit;
	}

	if( ppvField)
	{
		*ppvField = getFieldVoid( pField);
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc: 	Delete a field (and it's subtree).
*****************************************************************************/
RCODE FlmRecord::remove(
	FlmField *		pField)
{
	RCODE				rc = FERR_OK;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( !pField)
	{
		goto Exit;
	}

	if( RC_BAD( rc = removeFields( pField, lastSubTreeField( pField))))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc: 	Internal position to the next sibling
*****************************************************************************/
FlmField * FlmRecord::nextSiblingField(
	FlmField *		pField)
{
	FLMUINT		uiLevel = getFieldLevel( pField);

	while( (pField = nextField( pField)) != NULL &&
		getFieldLevel( pField) > uiLevel)
	{
		;
	}

	pField = (pField && getFieldLevel( pField) == uiLevel)
					? pField
					: NULL;

#ifdef FLM_DEBUG
	if( pField)
	{
		flmAssert( pField->ui16FieldID);
	}
#endif

	return( pField);
}

/*****************************************************************************
Desc: 	Position to the prev sibling
*****************************************************************************/
void * FlmRecord::prevSibling(
	void *		pvField)
{
	FlmField *		pField = getFieldPointer( pvField);
	FLMUINT			uiLevel;

	if( !pField)
	{
		return( NULL);
	}

	uiLevel = getFieldLevel( pField);
	while( (pField = prevField( pField)) != NULL &&
		getFieldLevel( pField) > uiLevel)
	{
		;
	}

	pField = (pField && getFieldLevel( pField) == uiLevel)
					? pField
					: NULL;

#ifdef FLM_DEBUG
	if( pField)
	{
		flmAssert( pField->ui16FieldID);
	}
#endif

	return( getFieldVoid( pField));
}

/*****************************************************************************
Desc: 	Return the last child field
*****************************************************************************/
FlmField * FlmRecord::lastChildField( 
	FlmField *		pField)
{
	FlmField *	pLastField = NULL;

	if( !pField)
	{
		return( NULL);
	}

	for( pField = firstChildField( pField); 
		pField; 
		pField = nextSiblingField( pField))
	{
		pLastField = pField;
	}

#ifdef FLM_DEBUG
	if( pLastField)
	{
		flmAssert( pLastField->ui16FieldID);
	}
#endif

	return( pLastField);
}

/*****************************************************************************
Desc: 	Return the parent field.
*****************************************************************************/
void * FlmRecord::parent(
	FlmField *		pField)
{
	FLMUINT		uiLevel;

	if( !pField)
	{
		return( NULL);
	}

	uiLevel = getFieldLevel( pField);
	while( (pField = prevField( pField)) != NULL && 
		getFieldLevel( pField) >= uiLevel)
	{
		;
	}

#ifdef FLM_DEBUG
	if( pField)
	{
		flmAssert( pField->ui16FieldID);
	}
#endif

	return( getFieldVoid( pField));
}

/*****************************************************************************
Desc: 	Set the right truncated flag.
*****************************************************************************/
void FlmRecord::setRightTruncated(
	FlmField *	pField,
	FLMBOOL		bTrueFalse)
{
	flmAssert( !isCached() && !isReadOnly());

	if( bTrueFalse)
	{
		pField->ui8TypeAndLevel |= FLD_DATA_RIGHT_TRUNCATED;
	}
	else
	{
		pField->ui8TypeAndLevel &= ~FLD_DATA_RIGHT_TRUNCATED;
	}
}

/*****************************************************************************
Desc: 	Set the left truncated flag.
*****************************************************************************/
void FlmRecord::setLeftTruncated(
	FlmField *	pField,
	FLMBOOL		bTrueFalse)
{
	flmAssert( !isCached() && !isReadOnly());

	if( bTrueFalse)
	{
		pField->ui8TypeAndLevel |= FLD_DATA_LEFT_TRUNCATED;
	}
	else
	{
		pField->ui8TypeAndLevel &= ~FLD_DATA_LEFT_TRUNCATED;
	}
}

/*****************************************************************************
Desc: 	Called from FLAIM's filesystem code to directly populate a record.
*****************************************************************************/
RCODE FlmRecord::allocStorageSpace(
	void *		pvField,
	FLMUINT		uiDataType,
	FLMUINT		uiLength,
	FLMUINT		uiEncLength,
	FLMUINT		uiEncId,
	FLMUINT		uiFlags,
	FLMBYTE **	ppucData,
	FLMBYTE **	ppucEncData)
{

	return( getNewDataPtr( getFieldPointer( pvField), uiDataType,
		uiLength, uiEncLength, uiEncId, uiFlags, ppucData, ppucEncData));
}

/*****************************************************************************
Desc:		Gives the implementor information about how many fields and how much
			data will be placed within this record.
*****************************************************************************/
RCODE FlmRecord::preallocSpace(
	FLMUINT		uiFieldCount,
	FLMUINT		uiDataSize)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiNewSize;
	FlmRecord *		pThis = this;
	FLMBOOL			bHeapAlloc = FALSE;

	if( isCached() || isReadOnly())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	flmAssert( uiFieldCount);
	flmAssert( !m_uiFldTblOffset);
	flmAssert( !m_uiDataBufOffset);

	uiNewSize = FLM_ALIGN_SIZE + 
					(uiFieldCount * sizeof( FlmField)) + uiDataSize;

	if( m_uiBufferSize < uiNewSize || (m_uiBufferSize - uiNewSize) >= 32)
	{
		if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->reallocBuf( 
			m_uiBufferSize, uiNewSize, &pThis, sizeof( FlmRecord *), &m_pucBuffer,
			&bHeapAlloc)))
		{
			goto Exit;
		}
		
		if( bHeapAlloc)
		{
			m_uiFlags |= RCA_HEAP_BUFFER;
		}
		else
		{
			m_uiFlags &= ~RCA_HEAP_BUFFER;
		}

		m_uiBufferSize = uiNewSize;
	}

	m_uiFldTblSize = uiFieldCount;

Exit:

	return( rc);
}

/*****************************************************************************
Desc: 	Copies the binary data into users buffer
*****************************************************************************/
RCODE FlmRecord::getBinary(
	void *		pvField,
	void *		pvBuf,
	FLMUINT *	puiBufLen)
{
	RCODE				rc = FERR_OK;
	FlmField *		pField = getFieldPointer( pvField);

	if( !pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	
	if (isEncryptedField( pField) &&
			!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}

	*puiBufLen = f_min( (*puiBufLen), getFieldDataLength( pField));
	f_memcpy( pvBuf, getDataPtr( pField), *puiBufLen);

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Return most (if not all) unused memory back to the system.  Doesn't
			compress avail slots out of the field list because the application
			could still have "pointers" into the list.
*****************************************************************************/
RCODE FlmRecord::compressMemory( void)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiLength;
	FLMUINT			uiTmp;
	FLMUINT			uiNewSize = 0;
	FLMUINT			uiNewDataSize = 0;
	FLMUINT			uiNewDataOffset;
	FlmField *		pFld;
	FLMBYTE *		pucNewBuf = NULL;
	FLMBYTE *		pucNewData;
	FLMUINT			uiPicketFenceSize = 0;
	FlmRecord *		pThis = this;
	FLMBOOL			bHeapAlloc = FALSE;

	if( isReadOnly() || isCached())
	{
		flmAssert( 0);
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

	if( !m_uiBufferSize || 
		(!m_bHolesInData && m_uiDataBufOffset == getDataBufSize()))
	{
		if( m_uiFldTblOffset == m_uiFldTblSize)
		{
			goto Exit;
		}
	}

	// Scan the record and determine the compressed size

	pFld = getFieldPointer( root());
	while( pFld)
	{
		uiLength = getFieldDataLength( pFld);

		if (!isEncryptedField( pFld))
		{
			if( uiLength > 4 && uiLength < 0xFF)
			{
				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiNewDataSize & FLM_ALLOC_ALIGN) != 0)
					{
						uiNewDataSize += (FLM_ALLOC_ALIGN + 1) -
													(uiNewDataSize & FLM_ALLOC_ALIGN);
					}
				}

				uiNewDataSize += uiLength;
			}
			else if( uiLength >= 0xFF)
			{
				// Add 1 to the length to allow for the flags byte which is only
				// present on long fields and encrypted fields.

				uiTmp = uiNewDataSize + sizeof( FLMUINT16) + 1;

				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
					{
						uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
					}
				}

				uiNewDataSize = uiTmp + uiLength;
			}
		}
		else  // Encrypted field
		{

#ifdef FLM_PICKET_FENCE
			uiPicketFenceSize = FLD_PICKET_FENCE_SIZE;
#endif

			// FLM_ENC_FLD_OVERHEAD includes 1 byte for the flags

			uiTmp = uiNewDataSize + FLM_ENC_FLD_OVERHEAD;

			if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
			{
				if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
				{
					uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
				}
			}

			uiNewDataSize = uiTmp + uiLength + getEncryptedDataLength(pFld) +
																				uiPicketFenceSize;

		}

		pFld = nextField( pFld);
	}

	uiNewSize = FLM_ALIGN_SIZE +
					uiNewDataSize + (m_uiFldTblOffset * sizeof( FlmField));

	// Re-allocate the buffer

	if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->allocBuf( 
		uiNewSize, &pThis, sizeof( FlmRecord *), &pucNewBuf, &bHeapAlloc)))
	{
		goto Exit;
	}

	uiNewDataOffset = 0;
	pucNewData = pucNewBuf + FLM_ALIGN_SIZE + (m_uiFldTblOffset * sizeof( FlmField));

	pFld = getFieldPointer( root());
	while( pFld)
	{
		uiLength = getFieldDataLength( pFld);

		if (!isEncryptedField( pFld))
		{
			if( uiLength > 4 && uiLength < 0xFF)
			{
				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiNewDataOffset & FLM_ALLOC_ALIGN) != 0)
					{
						uiNewDataOffset += (FLM_ALLOC_ALIGN + 1) -
													(uiNewDataOffset & FLM_ALLOC_ALIGN);
					}
				}

				flmAssert( uiNewDataOffset + uiLength <= uiNewDataSize);
				f_memcpy( &pucNewData[ uiNewDataOffset],
					getDataPtr( pFld), uiLength);
				pFld->ui32DataOffset = (FLMUINT32)uiNewDataOffset;
				uiNewDataOffset += uiLength;
			}
			else if( uiLength >= 0xFF)
			{
				uiTmp = uiNewDataOffset + sizeof( FLMUINT16) + 1;

				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
					{
						uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
					}
				}

				flmAssert( uiTmp + uiLength <= uiNewDataSize);
				pucNewData[ uiNewDataOffset] = 0;  // Set the flags byte.
				UW2FBA( (FLMUINT32)uiLength, &pucNewData[ uiNewDataOffset + FLD_ENC_ENCID]);
				f_memcpy( &pucNewData[ uiTmp], getDataPtr( pFld), uiLength);
				pFld->ui32DataOffset = (FLMUINT32)uiNewDataOffset;
				uiNewDataOffset = uiTmp + uiLength;
			}
		}
		else  // Encrypted field
		{
			FLMUINT		uiFlags;
			FLMUINT		uiEncId;
			FLMUINT		uiEncLength;
			FLMUINT		uiEncTmp;

#ifdef FLM_PICKET_FENCE
			uiPicketFenceSize = FLD_PICKET_FENCE_SIZE;
#endif

			uiFlags = getEncFlags( pFld);
			uiEncId = getEncryptionID( pFld);
			uiEncLength = getEncryptedDataLength( pFld);

			uiTmp = uiNewDataOffset + FLM_ENC_FLD_OVERHEAD;

			if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
			{
				if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
				{
					uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
				}
			}

			uiEncTmp = uiTmp + uiLength + (uiPicketFenceSize / 2);

			flmAssert( uiEncTmp + uiEncLength + (uiPicketFenceSize / 2) <= uiNewDataSize);
			pucNewData[ uiNewDataOffset + FLD_ENC_FLAGS] = ( FLMBYTE)uiFlags;
			UW2FBA( (FLMUINT32)uiEncId, &pucNewData[ uiNewDataOffset + FLD_ENC_ENCID]);
			UW2FBA( (FLMUINT32)uiLength, &pucNewData[ uiNewDataOffset + FLD_ENC_DATA_LEN]);
			UW2FBA( (FLMUINT32)uiEncLength, &pucNewData[ uiNewDataOffset + FLD_ENC_ENCRYPTED_DATA_LEN]);
#ifdef FLM_PICKET_FENCE
			// Set the picket fence
			f_sprintf( (char *)&pucNewData[ uiTmp], FLD_RAW_FENCE);
			uiTmp += (FLD_PICKET_FENCE_SIZE / 2);
#endif
			f_memcpy( &pucNewData[ uiTmp], getDataPtr( pFld), uiLength);
#ifdef FLM_PICKET_FENCE
			// Set the picket fence
			f_sprintf( (char *)&pucNewData[ uiEncTmp], FLD_ENC_FENCE);
			uiEncTmp += (FLD_PICKET_FENCE_SIZE / 2);
#endif
			f_memcpy( &pucNewData[ uiEncTmp], getEncryptionDataPtr( pFld), uiEncLength);
			pFld->ui32DataOffset = (FLMUINT32)uiNewDataOffset;
			uiNewDataOffset = uiEncTmp + uiEncLength;
		}

		pFld = nextField( pFld);
	}

	// The field list cannot be compressed because the application may
	// still have "pointers" (offsets) into the table

	f_memcpy( pucNewBuf, m_pucBuffer,
		FLM_ALIGN_SIZE + (m_uiFldTblOffset * sizeof( FlmField)));

#ifdef FLM_DEBUG

	uiTmp = (FLM_ALIGN_SIZE + uiNewDataOffset +
		(sizeof( FlmField) * m_uiFldTblOffset));

	flmAssert( uiTmp == uiNewSize);

#endif

	// Update the member variables

	gv_FlmSysData.RCacheMgr.pRecBufAlloc->freeBuf( 
		m_uiBufferSize, &m_pucBuffer);

	m_pucBuffer = pucNewBuf;
	pucNewBuf = NULL;
	
	if( bHeapAlloc)
	{
		m_uiFlags |= RCA_HEAP_BUFFER;
	}
	else
	{
		m_uiFlags &= ~RCA_HEAP_BUFFER;
	}

	m_uiBufferSize = uiNewSize;
	m_uiFldTblSize = m_uiFldTblOffset;
	m_uiDataBufOffset = uiNewDataSize;
	m_bHolesInData = FALSE;

Exit:

	if( pucNewBuf)
	{
		gv_FlmSysData.RCacheMgr.pRecBufAlloc->freeBuf( 
			uiNewSize, &pucNewBuf);
	}

	return( rc);
}

/*****************************************************************************
Desc:		Return most all unused memory back to the system.  This should
			be called by the Release method of the record object.
*****************************************************************************/
RCODE FlmRecord::compactMemory( void)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLength;
	FLMUINT		uiTmp;
	FLMUINT		uiNewSize = 0;
	FLMUINT		uiNewDataSize = 0;
	FLMUINT		uiNewDataOffset;
	FLMUINT		uiFields = 0;
	FLMUINT		uiSlot;
	FlmField *	pFld;
	FlmField *	pNewFld;
	FlmField *	pNewFldTbl;
	FLMBYTE *	pucNewBuf = NULL;
	FLMBYTE *	pucNewData;
	FLMUINT		uiPicketFenceSize = 0;
	FlmRecord *	pThis = this;
	FLMBOOL		bHeapAlloc = FALSE;

	flmAssert( isCached());
	flmAssert( m_ui32RefCnt == 1);

	// Temporarily increment the reference count so that we don't hit
	// debug asserts while processing

	m_ui32RefCnt++;

	if( !m_uiBufferSize ||
		(!m_bHolesInData && m_uiDataBufOffset == getDataBufSize()))
	{
		if( !m_uiFirstAvail && m_uiFldTblOffset == m_uiFldTblSize)
		{
			goto Exit;
		}
	}
	
	if( isOldVersion())
	{
		FLMUINT		uiTotalMemory = getTotalMemory();
		
		flmAssert( gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes >= uiTotalMemory);
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes -= uiTotalMemory;
	}
			
	// Scan the record and determine the compressed size

	pFld = getFieldPointer( root());
	while( pFld)
	{
		uiLength = getFieldDataLength( pFld);

		if (!isEncryptedField(pFld))
		{
			if( uiLength > 4 && uiLength < 0xFF)
			{
				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiNewDataSize & FLM_ALLOC_ALIGN) != 0)
					{
						uiNewDataSize += (FLM_ALLOC_ALIGN + 1) -
													(uiNewDataSize & FLM_ALLOC_ALIGN);
					}
				}

				uiNewDataSize += uiLength;
			}
			else if( uiLength >= 0xFF)
			{
				// Add one extra byte for the flags byte.

				uiTmp = uiNewDataSize + sizeof( FLMUINT16) + 1;

				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
					{
						uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
					}
				}

				uiNewDataSize = uiTmp + uiLength;
			}
		}
		else  // Encrypted field
		{

#ifdef FLM_PICKET_FENCE
			uiPicketFenceSize = FLD_PICKET_FENCE_SIZE;
#endif

			uiTmp = uiNewDataSize + FLM_ENC_FLD_OVERHEAD;

			if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
			{
				if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
				{
					uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
				}
			}

			uiNewDataSize = uiTmp + uiLength + getEncryptedDataLength(pFld) +
																			uiPicketFenceSize;

		}

		uiFields++;
		pFld = nextField( pFld);
	}

	uiNewSize = FLM_ALIGN_SIZE +
					uiNewDataSize + (uiFields * sizeof( FlmField));

	flmAssert( uiNewSize <= m_uiBufferSize);

	// Allocate a new buffer

	if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->allocBuf(
		uiNewSize, &pThis, sizeof( FlmRecord *), &pucNewBuf, &bHeapAlloc)))
	{
		goto Exit;
	}

	pNewFldTbl = (FlmField *)(pucNewBuf + FLM_ALIGN_SIZE);
	uiSlot = 0;
	uiNewDataOffset = 0;
	pucNewData = pucNewBuf + FLM_ALIGN_SIZE + (uiFields * sizeof( FlmField));

	pFld = getFieldPointer( root());
	while( pFld)
	{
		uiLength = getFieldDataLength( pFld);
		pNewFld = &pNewFldTbl[ uiSlot];
		f_memcpy( pNewFld, pFld, sizeof( FlmField));

		if (!isEncryptedField(pFld))
		{

			if( uiLength > 4 && uiLength < 0xFF)
			{
				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiNewDataOffset & FLM_ALLOC_ALIGN) != 0)
					{
						uiNewDataOffset += (FLM_ALLOC_ALIGN + 1) -
													(uiNewDataOffset & FLM_ALLOC_ALIGN);
					}
				}

				flmAssert( uiNewDataOffset + uiLength <= uiNewDataSize);
				f_memcpy( &pucNewData[ uiNewDataOffset], getDataPtr( pFld), uiLength);
				pNewFld->ui32DataOffset = (FLMUINT32)uiNewDataOffset;
				uiNewDataOffset += uiLength;
			}
			else if( uiLength >= 0xFF)
			{
				uiTmp = uiNewDataOffset + sizeof( FLMUINT16) + 1;

				if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
				{
					if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
					{
						uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
					}
				}

				flmAssert( uiTmp + uiLength <= uiNewDataSize);
				pucNewData[ uiNewDataOffset] = 0;	// Flags
				UW2FBA( (FLMUINT32)uiLength, &pucNewData[ uiNewDataOffset + FLD_ENC_ENCID]);
				f_memcpy( &pucNewData[ uiTmp], getDataPtr( pFld), uiLength);
				pNewFld->ui32DataOffset = (FLMUINT32)uiNewDataOffset;
				uiNewDataOffset = uiTmp + uiLength;
			}
		}
		else
		{
			FLMUINT		uiFlags;
			FLMUINT		uiEncId;
			FLMUINT		uiEncLength;
			FLMUINT		uiEncTmp;

#ifdef FLM_PICKET_FENCE
			uiPicketFenceSize = FLD_PICKET_FENCE_SIZE;
#endif

			uiFlags = getEncFlags( pFld);
			uiEncId = getEncryptionID( pFld);
			uiEncLength = getEncryptedDataLength( pFld);

			uiTmp = uiNewDataOffset + FLM_ENC_FLD_OVERHEAD;

			if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
			{
				if( (uiTmp & FLM_ALLOC_ALIGN) != 0)
				{
					uiTmp += (FLM_ALLOC_ALIGN + 1) - (uiTmp & FLM_ALLOC_ALIGN);
				}
			}

			uiEncTmp = uiTmp + uiLength + (uiPicketFenceSize / 2);

			flmAssert( uiEncTmp + uiEncLength + (uiPicketFenceSize / 2) <= uiNewDataSize);
			pucNewData[ uiNewDataOffset + FLD_ENC_FLAGS] = ( FLMBYTE)uiFlags;
			UW2FBA( (FLMUINT16)uiEncId, &pucNewData[ uiNewDataOffset + FLD_ENC_ENCID]);
			UW2FBA( (FLMUINT16)uiLength, &pucNewData[ uiNewDataOffset + FLD_ENC_DATA_LEN]);
			UW2FBA( (FLMUINT16)uiEncLength, &pucNewData[ uiNewDataOffset + FLD_ENC_ENCRYPTED_DATA_LEN]);

#ifdef FLM_PICKET_FENCE
			f_sprintf( (char *)&pucNewData[ uiTmp], FLD_RAW_FENCE);
			uiTmp += (FLD_PICKET_FENCE_SIZE / 2);
#endif

			f_memcpy( &pucNewData[ uiTmp], getDataPtr( pFld), uiLength);

#ifdef FLM_PICKET_FENCE
			f_sprintf( (char *)&pucNewData[ uiEncTmp], FLD_ENC_FENCE);
			uiEncTmp += (FLD_PICKET_FENCE_SIZE / 2);
#endif

			f_memcpy( &pucNewData[ uiEncTmp], getEncryptionDataPtr( pFld), uiEncLength);
			pNewFld->ui32DataOffset = (FLMUINT32)uiNewDataOffset;
			uiNewDataOffset = uiEncTmp + uiEncLength;
		}

		pFld = nextField( pFld);

		pNewFld->uiPrev = (FIELDLINK)uiSlot;
		if( pFld)
		{
			pNewFld->uiNext = (FIELDLINK)(uiSlot + 2);
		}
		else
		{
			pNewFld->uiNext = 0;
		}

		uiSlot++;
	}

	// Update the member variables

	gv_FlmSysData.RCacheMgr.pRecBufAlloc->freeBuf( 
		m_uiBufferSize, &m_pucBuffer);

	m_pucBuffer = pucNewBuf;
	pucNewBuf = NULL;

	if( bHeapAlloc)
	{
		m_uiFlags |= RCA_HEAP_BUFFER;
	}
	else
	{
		m_uiFlags &= ~RCA_HEAP_BUFFER;
	}
	
	m_uiBufferSize = uiNewSize;
	m_uiFldTblOffset = uiFields;
	m_uiFldTblSize = uiFields;
	m_uiDataBufOffset = uiNewDataSize;
	m_bHolesInData = FALSE;
	m_uiAvailFields = 0;
	m_uiFirstAvail = 0;

	if( isOldVersion())
	{
		FLMUINT		uiTotalMemory = getTotalMemory();
		
		gv_FlmSysData.RCacheMgr.Usage.uiOldVerBytes += uiTotalMemory;
	}

Exit:

	if( pucNewBuf)
	{
		gv_FlmSysData.RCacheMgr.pRecBufAlloc->freeBuf( 
			uiNewSize, &pucNewBuf);
	}

	// Remove the reference count added above

	m_ui32RefCnt--;
	return( rc);
}

/*****************************************************************************
Desc:		Create a new field after pCurField.
*****************************************************************************/
RCODE	FlmRecord::createField(
	FlmField *		pPrevField,
	FlmField **		ppNewField)
{
	RCODE				rc = FERR_OK;
	FlmField *		pNewField;
	FLMUINT			uiNewSize;
	FlmField *		pFldTbl = NULL;
	void *			pvPrevField;
	FlmRecord *		pThis = this;
	FLMBOOL			bHeapAlloc = FALSE;

	flmAssert( !isReadOnly());
	flmAssert( !isCached());

	pvPrevField = getFieldVoid( pPrevField);
	pPrevField = NULL;

	if( m_uiFirstAvail)
	{
		flmAssert( m_uiAvailFields);
		pNewField = &(getFieldTable()[ m_uiFirstAvail - 1]);
		m_uiFirstAvail = pNewField->uiNext;
		m_uiAvailFields--;
	}
	else
	{
		if( m_uiFldTblOffset == m_uiFldTblSize)
		{
			// Resize the buffer

			if( m_uiBufferSize)
			{
				uiNewSize = m_uiBufferSize + (sizeof( FlmField) * 8);
			}
			else
			{
				uiNewSize = FLM_ALIGN_SIZE + (sizeof( FlmField) * 8);
			}

			if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->reallocBuf(
				m_uiBufferSize, uiNewSize, &pThis, sizeof( FlmRecord *), 
				&m_pucBuffer, &bHeapAlloc)))
			{
				goto Exit;
			}

			m_uiBufferSize = uiNewSize;
			m_uiFldTblSize += 8;

			if( m_uiDataBufOffset)
			{
				f_memmove( &(getFieldTable()[ m_uiFldTblSize]),
					&(getFieldTable()[ m_uiFldTblSize - 8]),
					m_uiDataBufOffset);
			}
			
			if( bHeapAlloc)
			{
				m_uiFlags |= RCA_HEAP_BUFFER;
			}
			else
			{
				m_uiFlags &= ~RCA_HEAP_BUFFER;
			}
		}

		pFldTbl = getFieldTable();
		pNewField = &pFldTbl[ m_uiFldTblOffset++];
	}

	f_memset( pNewField, 0, sizeof( FlmField));
	pPrevField = getFieldPointer( pvPrevField);

	if( pPrevField)
	{
		pNewField->ui16FieldID = 0xFFFF;
		pNewField->uiPrev = (FIELDLINK)((FLMUINT)pvPrevField);
		pNewField->uiNext = pPrevField->uiNext;

		if( pPrevField->uiNext)
		{
			getFieldPointer( (void *)((FLMUINT)(pPrevField->uiNext)))->uiPrev =
				(FIELDLINK)((FLMUINT)getFieldVoid( pNewField));
		}

		pPrevField->uiNext = (FIELDLINK)((FLMUINT)getFieldVoid( pNewField));
	}

	*ppNewField = pNewField;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Remove a specific field, or a range of fields.
*****************************************************************************/
RCODE FlmRecord::removeFields(
	FlmField *		pFirstField,
	FlmField *		pLastField)
{
	RCODE				rc = FERR_OK;
	FlmField *		pCurField;
	FLMUINT			uiFieldsRemoved = 0;

	flmAssert( !isReadOnly());
	flmAssert( !isCached());

	// Setup the common case first of removing a single field.

	if( !pLastField)
	{
		pLastField = pFirstField;
	}

	// Fix up the prev and next pointers

	if( pFirstField->uiPrev)
	{
		getFieldPointer( (void *)((FLMUINT)(pFirstField->uiPrev)))->uiNext =
			pLastField->uiNext;
	}

	if( pLastField->uiNext)
	{
		getFieldPointer( (void *)((FLMUINT)(pLastField->uiNext)))->uiPrev =
			pFirstField->uiPrev;

		pLastField->uiNext = 0;
	}

	// Clear the field slots

	pCurField = pFirstField;
	while( pCurField)
	{
		pCurField->uiPrev = 0;
		pCurField->ui32DataOffset = 0;
		pCurField->ui16FieldID = 0;
		pCurField->ui8DataLen = 0;
		pCurField->ui8TypeAndLevel = 0;
		pCurField = nextField( pCurField);
		uiFieldsRemoved++;
	}

	pLastField->uiNext = m_uiFirstAvail;
	m_uiFirstAvail = (FIELDLINK)((FLMUINT)getFieldVoid( pFirstField));
	m_uiAvailFields += uiFieldsRemoved;

	return( rc);
}

/*****************************************************************************
Desc:		Returns a pointer that is of size 'uiNewLength' that the caller
			can then write a fields data to. If possible the fields current
			data pointer will be reused.
*****************************************************************************/
RCODE	FlmRecord::getNewDataPtr(
	FlmField *	pField,
	FLMUINT		uiDataType,
	FLMUINT		uiNewLength,
	FLMUINT		uiEncNewLength,
	FLMUINT		uiEncId,
	FLMUINT		uiFlags,
	FLMBYTE **	ppDataPtr,
	FLMBYTE **	ppEncDataPtr)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucDataPtr = NULL;
	FLMBYTE *	pucEncDataPtr = NULL;
	FLMUINT		uiOldLength;
	FLMUINT		uiAlignment = 0;
	FLMUINT		uiSlot = (FLMUINT)getFieldVoid( pField);
	FLMUINT		uiDataBufSize;
	FLMUINT		uiHeader = 0;
	FLMUINT		uiOldHeader = 0;
	FLMUINT		uiAllocStart = 0;
	FLMUINT		uiTmp;
	FLMBOOL		bNewEncrypted = (uiEncId ? TRUE : FALSE);
	FLMBOOL		bOldEncrypted;
	FLMBYTE *	pucTmp;
	FLMUINT		uiNewSize;
	FLMUINT		uiPicketFenceSize = 0;
	FlmRecord *		pThis = this;
	FLMBOOL			bHeapAlloc = FALSE;
	
	flmAssert( !isCached());
	flmAssert( !isReadOnly());

#ifdef FLM_PICKET_FENCE
	uiPicketFenceSize = (bNewEncrypted ? FLD_PICKET_FENCE_SIZE : 0);
#endif

	// Test for an invalid encryption Id.  This could be
	// indicative of an uninitialized variaqble.

	if (uiEncId > FLM_RESERVED_TAG_NUMS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_BAD_ENCDEF_ID);
		goto Exit;
	}

	// Make sure the requested size doesn't exceed the maximum
	// field value size

	if( uiNewLength > FLM_MAX_FIELD_VAL_SIZE ||
		 uiEncNewLength > FLM_MAX_FIELD_VAL_SIZE)
	{
		rc = RC_SET( FERR_VALUE_TOO_LARGE);
		goto Exit;
	}
	
	bOldEncrypted = isEncryptedField( pField);

	// If the new field is encrypted, we need to prefix the value
	// with the encryption overhead.
	// Otherwise, if the new length is >= 0xFF, we need to prefix the value
	// with a 2-byte length

	uiHeader = (bNewEncrypted ? FLM_ENC_FLD_OVERHEAD
									  : (uiNewLength >= 0xFF ? sizeof( FLMUINT16) + 1
									  								 : 0));
	// Determine the true original data length

	uiOldLength = getFieldDataLength( pField);

	if( uiOldLength >= 0xFF ||
		 bOldEncrypted)
	{
		flmAssert( pField->ui8DataLen == 0xFF);

		// Account for the header
		uiOldHeader = (bOldEncrypted ? FLM_ENC_FLD_OVERHEAD : sizeof( FLMUINT16) + 1);
		uiOldLength += uiOldHeader;

		// Special work if this is a binary field

		if( getFieldDataType( pField) == FLM_BINARY_TYPE)
		{
			// Since the value is binary, need to account for any
			// alignment bytes

			if( ((pField->ui32DataOffset + uiOldHeader) & FLM_ALLOC_ALIGN) != 0)
			{
				uiAlignment = (FLM_ALLOC_ALIGN + 1) -
					((pField->ui32DataOffset + uiOldHeader) & FLM_ALLOC_ALIGN);
				uiOldLength += uiAlignment;
			}

			if (bOldEncrypted)
			{
				// Add the length of the encrypted data.

				uiOldLength += getEncryptedDataLength( pField);
			}
		}
	}

#ifdef FLM_PICKET_FENCE
	if (bOldEncrypted)
	{
		uiOldLength += FLD_PICKET_FENCE_SIZE;  // Add the picket fence size
	}
#endif

	// If the old length is > 4 and the new length is <= 4 the value MUST
	// be stored in ui32DataOffset.  To be simple, this is coded for
	// the four cases.

	uiAlignment = 0;

	if( uiOldLength <= sizeof(FLMUINT32))
	{
		if( uiNewLength <= sizeof(FLMUINT32))
		{
			if (!bNewEncrypted)
			{
				pField->ui32DataOffset = 0;
				pucDataPtr = (FLMBYTE *)&(pField->ui32DataOffset);
				pField->ui8DataLen = (FLMUINT8)uiNewLength;
			}
			else
			{
				// The new field is encrypted.
				// If this is a binary field, it must start on an aligned byte.

				if( uiDataType == FLM_BINARY_TYPE &&
					((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN) != 0)
				{
					uiAlignment = (FLM_ALLOC_ALIGN + 1) -
						((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN);
				}

				uiAllocStart = m_uiDataBufOffset;
				uiDataBufSize = getDataBufSize();

				if( uiNewLength + uiEncNewLength + uiAlignment +
					 uiHeader + m_uiDataBufOffset + uiPicketFenceSize > uiDataBufSize)
				{
					// Re-allocate the buffer.

					uiNewSize = m_uiBufferSize +
									uiHeader +
									uiAlignment +
									uiNewLength +
									uiEncNewLength +
									uiPicketFenceSize +
									32;

					if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->reallocBuf( 
						m_uiBufferSize, uiNewSize,
						&pThis, sizeof( FlmRecord *), &m_pucBuffer, &bHeapAlloc)))
					{
						goto Exit;
					}

					m_uiBufferSize = uiNewSize;
					pField = &(getFieldTable()[ uiSlot - 1]);
					
					if( bHeapAlloc)
					{
						m_uiFlags |= RCA_HEAP_BUFFER;
					}
					else
					{
						m_uiFlags &= ~RCA_HEAP_BUFFER;
					}
				}

				pucDataPtr = getDataBufPtr() + uiAllocStart + uiHeader +
									uiAlignment + (uiPicketFenceSize / 2);
				pucEncDataPtr = pucDataPtr + uiNewLength + (uiPicketFenceSize / 2);

				setEncHeader( getDataBufPtr() + uiAllocStart, uiFlags, uiEncId,
								  uiNewLength, uiEncNewLength);

				pField->ui8DataLen = 0xFF;
				pField->ui32DataOffset = (FLMUINT32)uiAllocStart;

				m_uiDataBufOffset += uiHeader + uiAlignment + uiNewLength +
												uiEncNewLength + uiPicketFenceSize;
			}
		}
		else
		{
			// If this is a binary field, it must start on an aligned byte.

			if( uiDataType == FLM_BINARY_TYPE &&
				((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN) != 0)
			{
				uiAlignment = (FLM_ALLOC_ALIGN + 1) -
					((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN);
			}

			uiAllocStart = m_uiDataBufOffset;
			uiDataBufSize = getDataBufSize();

			if( uiNewLength + uiAlignment + uiHeader + uiEncNewLength +
				 m_uiDataBufOffset + uiPicketFenceSize > uiDataBufSize)
			{
				// Re-allocate the buffer.
				uiNewSize = m_uiBufferSize +
								uiHeader +
								uiAlignment +
								uiNewLength +
								uiEncNewLength +
								uiPicketFenceSize +
								32;

				if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->reallocBuf( 
					m_uiBufferSize, uiNewSize,
					&pThis, sizeof( FlmRecord *), &m_pucBuffer, &bHeapAlloc)))
				{
					goto Exit;
				}

				m_uiBufferSize = uiNewSize;
				pField = &(getFieldTable()[ uiSlot - 1]);
				
				if( bHeapAlloc)
				{
					m_uiFlags |= RCA_HEAP_BUFFER;
				}
				else
				{
					m_uiFlags &= ~RCA_HEAP_BUFFER;
				}
			}

			pucDataPtr = getDataBufPtr() + uiAllocStart + uiHeader +
											uiAlignment + (uiPicketFenceSize / 2);
			if (bNewEncrypted)
			{
				pucEncDataPtr = pucDataPtr + uiNewLength + (uiPicketFenceSize / 2);
			}

			// The presence of uiHeader indicates that the length of the new field
			// is greater than 255 bytes, or it is encrypted, therefore we will
			// store the actual length in the buffer.

			if( uiHeader)
			{
				if (!bNewEncrypted)
				{
					pucTmp = getDataBufPtr() + uiAllocStart;
					*pucTmp = 0;	// Flags
					pucTmp++;
					UW2FBA( (FLMUINT16)uiNewLength, pucTmp);
					pField->ui8DataLen = 0xFF;
					pField->ui32DataOffset = (FLMUINT32)uiAllocStart;
				}
				else // Encrypted
				{
					setEncHeader( getDataBufPtr() + uiAllocStart, uiFlags,
									 	uiEncId, uiNewLength, uiEncNewLength);

					pField->ui8DataLen = 0xFF;
					pField->ui32DataOffset = (FLMUINT32)uiAllocStart;
				}
			}
			else
			{
				flmAssert( !bNewEncrypted);
				pField->ui8DataLen = (FLMUINT8)uiNewLength;
				pField->ui32DataOffset = (FLMUINT32)(uiAllocStart + uiAlignment);
			}

			m_uiDataBufOffset += uiHeader + uiAlignment + uiNewLength +
										uiEncNewLength + uiPicketFenceSize;
		}
	}
	else // uiOldLength > sizeof(FLMUINT32)
	{
		if( uiNewLength > sizeof(FLMUINT32))
		{

			if( uiDataType == FLM_BINARY_TYPE &&
				((pField->ui32DataOffset + uiHeader) & FLM_ALLOC_ALIGN) != 0)
			{
				uiAlignment = (FLM_ALLOC_ALIGN + 1) -
					((pField->ui32DataOffset + uiHeader) & FLM_ALLOC_ALIGN);

			}

			// Smaller or same size?

			if( (uiTmp = (uiHeader + uiAlignment + uiNewLength +
									uiEncNewLength + uiPicketFenceSize)) <= uiOldLength)
			{
				if( uiTmp != uiOldLength)
				{
					m_bHolesInData = TRUE;
				}

				if( uiHeader)
				{
					if (!bNewEncrypted)
					{
						pucTmp = getDataBufPtr() + pField->ui32DataOffset;
						*pucTmp = 0;	// Flags
						pucTmp++;
						UW2FBA( (FLMUINT16)uiNewLength, pucTmp);
						pField->ui8DataLen = 0xFF;
					}
					else
					{
						// It's encrypted...

						setEncHeader( getDataBufPtr() + pField->ui32DataOffset,
										  uiFlags,
										  uiEncId,
										  uiNewLength,
										  uiEncNewLength);

						pField->ui8DataLen = 0xFF;
						// No need to set the offset because we are re-using the same
						// buffer space.
					}
				}
				else
				{
					flmAssert( uiNewLength < 0xFF);
					flmAssert( !bNewEncrypted);
					pField->ui8DataLen = (FLMUINT8)uiNewLength;
				}

				pucDataPtr = getDataBufPtr() +
					pField->ui32DataOffset + uiHeader + uiAlignment + (uiPicketFenceSize / 2);
				if (bNewEncrypted)
				{
					pucEncDataPtr = pucDataPtr + uiNewLength + (uiPicketFenceSize / 2);
				}
			}
			else
			{
				// The new value is larger than the original value.

				// If this is a binary field it must start on an aligned byte.

				uiAlignment = 0;

				if( uiDataType == FLM_BINARY_TYPE &&
					((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN) != 0)
				{
					uiAlignment = (FLM_ALLOC_ALIGN + 1) -
						((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN);
				}

				uiAllocStart = m_uiDataBufOffset;
				uiDataBufSize = getDataBufSize();

				if( (m_uiDataBufOffset + uiHeader +
							uiAlignment + uiNewLength +
							uiEncNewLength + uiPicketFenceSize) > uiDataBufSize)
				{
					// Re-allocate the buffer.

					uiNewSize = m_uiBufferSize +
									uiHeader +
									uiAlignment +
									uiNewLength +
									uiEncNewLength +
									uiPicketFenceSize +
									32;

					if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->reallocBuf( 
						m_uiBufferSize, uiNewSize,
						&pThis, sizeof( FlmRecord *), &m_pucBuffer, &bHeapAlloc)))
					{
						goto Exit;
					}

					m_uiBufferSize = uiNewSize;
					pField = &(getFieldTable()[ uiSlot - 1]);
					
					if( bHeapAlloc)
					{
						m_uiFlags |= RCA_HEAP_BUFFER;
					}
					else
					{
						m_uiFlags &= ~RCA_HEAP_BUFFER;
					}
				}

				pucDataPtr = getDataBufPtr() + m_uiDataBufOffset + uiHeader +
														uiAlignment + (uiPicketFenceSize / 2);
				if (bNewEncrypted)
				{
					pucEncDataPtr = pucDataPtr + uiNewLength + (uiPicketFenceSize / 2);
				}

				if( uiHeader)
				{
					if (!bNewEncrypted)
					{
						pucTmp = getDataBufPtr() + uiAllocStart;
						*pucTmp = 0;	// Flags
						pucTmp++;
						UW2FBA( (FLMUINT16)uiNewLength, pucTmp);
						pField->ui8DataLen = 0xFF;
						pField->ui32DataOffset = (FLMUINT32)uiAllocStart;
					}
					else
					{
						setEncHeader( getDataBufPtr() + uiAllocStart, uiFlags,
										  uiEncId, uiNewLength, uiEncNewLength);

						pField->ui8DataLen = 0xFF;
						pField->ui32DataOffset = (FLMUINT32)uiAllocStart;
					}
				}
				else
				{
					flmAssert( uiNewLength < 0xFF);
					flmAssert( !bNewEncrypted);
					pField->ui32DataOffset = (FLMUINT32)(uiAllocStart + uiAlignment);
					pField->ui8DataLen = (FLMUINT8)uiNewLength;
				}

				m_uiDataBufOffset += uiHeader + uiAlignment + uiNewLength +
											uiEncNewLength + uiPicketFenceSize;
				m_bHolesInData = TRUE;
			}
		}
		else // uiNewLength <= sizeof(FLMUINT32)
		{
			if (!bNewEncrypted)
			{
				pField->ui32DataOffset = 0;
				pucDataPtr = (FLMBYTE *)&(pField->ui32DataOffset);
				pField->ui8DataLen = (FLMUINT8)uiNewLength;
				m_bHolesInData = TRUE;
			}
			else
			{
				// The new field is encrypted.

				// If this is a binary field, it must start on an aligned byte.

				if( uiDataType == FLM_BINARY_TYPE &&
					((pField->ui32DataOffset + uiHeader) & FLM_ALLOC_ALIGN) != 0)
				{
					uiAlignment = (FLM_ALLOC_ALIGN + 1) -
						((pField->ui32DataOffset + uiHeader) & FLM_ALLOC_ALIGN);
				}

				// New field will be stored in a buffer.
				// See if we can re-use the old buffer.

				if( (uiTmp = (uiHeader + uiAlignment + uiNewLength +
									uiEncNewLength + uiPicketFenceSize)) <= uiOldLength)
				{
					if( uiTmp != uiOldLength)
					{
						m_bHolesInData = TRUE;
					}

					setEncHeader( getDataBufPtr() + pField->ui32DataOffset,
										uiFlags,
										uiEncId,
										uiNewLength,
										uiEncNewLength);

					pField->ui8DataLen = 0xFF;
					uiAllocStart = pField->ui32DataOffset;

				}
				else
				{
					// The field position is changing, so we need to recalculate the alignment
					// (if needed).

					// If this is a binary field, it must start on an aligned byte.

					if( uiDataType == FLM_BINARY_TYPE &&
						((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN) != 0)
					{
						uiAlignment = (FLM_ALLOC_ALIGN + 1) -
							((m_uiDataBufOffset + uiHeader) & FLM_ALLOC_ALIGN);
					}

					uiAllocStart = m_uiDataBufOffset;
					uiDataBufSize = getDataBufSize();
	
					if( uiNewLength + uiEncNewLength + uiAlignment +
						uiHeader + m_uiDataBufOffset + uiPicketFenceSize > uiDataBufSize)
					{
						// Re-allocate the buffer.
	
						uiNewSize = m_uiBufferSize +
										uiHeader +
										uiAlignment +
										uiNewLength +
										uiEncNewLength +
										uiPicketFenceSize +
										32;
	
						if( RC_BAD( rc = gv_FlmSysData.RCacheMgr.pRecBufAlloc->reallocBuf( 
							m_uiBufferSize, uiNewSize,
							&pThis, sizeof( FlmRecord *), &m_pucBuffer, &bHeapAlloc)))
						{
							goto Exit;
						}
	
						m_uiBufferSize = uiNewSize;
						pField = &(getFieldTable()[ uiSlot - 1]);
						
						if( bHeapAlloc)
						{
							m_uiFlags |= RCA_HEAP_BUFFER;
						}
						else
						{
							m_uiFlags &= ~RCA_HEAP_BUFFER;
						}
					}
	
					setEncHeader( getDataBufPtr() + uiAllocStart,
									  uiFlags,
									  uiEncId,
									  uiNewLength,
									  uiEncNewLength);
	
					pField->ui8DataLen = 0xFF;
					pField->ui32DataOffset = (FLMUINT32)uiAllocStart;
	
					m_uiDataBufOffset += uiHeader + uiAlignment + uiNewLength +
													uiEncNewLength + uiPicketFenceSize;
				}

				pucDataPtr = getDataBufPtr() + uiAllocStart + uiHeader +
												uiAlignment + (uiPicketFenceSize / 2);
				pucEncDataPtr = pucDataPtr + uiNewLength + (uiPicketFenceSize / 2);
			}
		}
	}

	setFieldDataType( pField, uiDataType);

#ifdef FLM_DEBUG
	f_memset( pucDataPtr, 0, uiNewLength);
	f_memset( pucEncDataPtr, 0, uiEncNewLength);
#endif

	if (ppDataPtr)
	{
		*ppDataPtr = pucDataPtr;
	}
	if (ppEncDataPtr)
	{
		*ppEncDataPtr = pucEncDataPtr;
	}

#ifdef FLM_PICKET_FENCE
	// Set the picket fences
	if (bNewEncrypted)
	{
		pucDataPtr -= 4;
		f_memcpy( pucDataPtr, (FLMBYTE *)FLD_RAW_FENCE,
											FLD_PICKET_FENCE_SIZE / 2);
		pucEncDataPtr -= 4;
		f_memcpy( pucEncDataPtr, (FLMBYTE *)FLD_ENC_FENCE,
											FLD_PICKET_FENCE_SIZE / 2);
	}
#endif

Exit:

	if( RC_BAD( rc))
	{
		if (ppDataPtr)
		{
			*ppDataPtr = NULL;
		}

		if (ppEncDataPtr)
		{
			*ppEncDataPtr = NULL;
		}
	}

	return( rc);
}

/*****************************************************************************
Desc:		Add a globally shared reference to this object.
*****************************************************************************/
FLMUINT FlmRecord::AddRef(
	FLMBOOL			bMutexLocked)
{
	FLMUINT		uiRefCnt = 0;
	FLMBOOL		bLockedMutex = FALSE;

#ifdef ATOMIC_INCDEC_SUPPORT
	(void)bMutexLocked;
	uiRefCnt = (FLMUINT)ftkAtomicIncrement( &m_ui32RefCnt);
#else

	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		bLockedMutex = TRUE;
		bMutexLocked = TRUE;
	}

	uiRefCnt = ++m_ui32RefCnt;
#endif

	if( bLockedMutex)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	}
	flmAssert( uiRefCnt > 1);
	return( uiRefCnt);
}

/*****************************************************************************
Desc:		Removes a globally shared reference to this object.
*****************************************************************************/
FLMUINT FlmRecord::Release(
	FLMBOOL			bMutexLocked)
{
	FLMUINT		uiRefCnt = 0;
	FLMBOOL		bUnlockMutex = FALSE;

#ifdef ATOMIC_INCDEC_SUPPORT
	if( isCached() && m_ui32RefCnt == 2)
	{
		if( !bMutexLocked)
		{
			f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
			bMutexLocked = TRUE;
			bUnlockMutex = TRUE;
		}
	}
	
	uiRefCnt = (FLMUINT)ftkAtomicDecrement( &m_ui32RefCnt);
#else
	if( !bMutexLocked)
	{
		f_mutexLock( gv_FlmSysData.RCacheMgr.hMutex);
		bMutexLocked = TRUE;
		bUnlockMutex = TRUE;
	}

	uiRefCnt = --m_ui32RefCnt;
#endif

	if( !uiRefCnt)
	{
		flmAssert( !isCached());

		m_uiFlags |= RCA_OK_TO_DELETE;
		delete this;
	}
	else if( bMutexLocked && uiRefCnt == 1 && isCached())
	{
		// If the record is still cached, and the reference count
		// is 1, it is safe to compact the record's buffer

		if( m_bHolesInData || getFreeMemory())
		{
#ifdef FLM_CHECK_RECORD
			if (RC_BAD( checkRecord()))
			{
				flmAssert( 0);
			}
#endif

			compactMemory();

#ifdef FLM_CHECK_RECORD
			if (RC_BAD( checkRecord()))
			{
				flmAssert( 0);
			}
#endif
			flmAssert( !getFreeMemory());
		}
	}

	if( bUnlockMutex)
	{
		f_mutexUnlock( gv_FlmSysData.RCacheMgr.hMutex);
	}

	return( uiRefCnt);
}

/*****************************************************************************
Desc: 	Return the nth occurance of the field with a matching field id.
*****************************************************************************/
void * FlmRecord::find(
	void *		pvField,
	FLMUINT		uiFieldID,
	FLMUINT		uiOccur,
	FLMUINT		uiFindOption)
{
	FLMUINT			uiStartLevel;
	FlmField *		pField = getFieldPointer( pvField);

	if( !uiOccur)
	{
		uiOccur = 1;
	}

	if( !pField)
  	{
		goto Exit;
	}

	uiStartLevel = getFieldLevel( pField);

	do
	{
		if( (uiFieldID == pField->ui16FieldID) && (--uiOccur < 1))
		{
			return( getFieldVoid( pField));
		}

	} while( (pField = nextField( pField)) != NULL &&
		((getFieldLevel( pField) > uiStartLevel) ||
			(uiFindOption == SEARCH_FOREST &&
				getFieldLevel( pField) == uiStartLevel)));

Exit:

	return( NULL);
}

/*****************************************************************************
Desc: 	Given a null terminated field path, return the matching occurance.
*****************************************************************************/
void * FlmRecord::find(
	void *		pvField,
	FLMUINT *	puiPathArray,
	FLMUINT		uiOccur,
	FLMUINT		uiFindOption)
{
	FlmField *	pSaveField;
	FLMUINT *	path;
	FLMUINT		uiLevel;
	FlmField *	pField = getFieldPointer( pvField);

	// Handle empty record

	if( !pvField)
	{
		return( NULL);
	}

	if( !uiOccur)
	{
		uiOccur = 1;
	}

	uiLevel = getFieldLevel( pField);

	for(;;)
	{
		path = puiPathArray + ( getFieldLevel( pField) - uiLevel);
		pSaveField = pField;

		if( *path == pField->ui16FieldID)
		{
			if( *(path + 1) == 0 && (--uiOccur < 1))
			{
				return( getFieldVoid( pField));
			}

			if( (pField = firstChildField( pField)) != NULL)
			{
				continue;
			}
			pField = pSaveField;
		}

		do
		{
			pField = nextField( pField);
		}
		while( pField != NULL && getFieldLevel( pField) > getFieldLevel( pSaveField));

		if( !pField || getFieldLevel( pField) < uiLevel ||
			(uiFindOption != SEARCH_FOREST && getFieldLevel( pField) == uiLevel))
		{
			break;
		}
	}

	return( NULL);
}

/*****************************************************************************
Desc:		Import a record from a file
*****************************************************************************/
RCODE FlmRecord::importRecord(
	F_FileHdl *		pFileHdl,
	F_NameTable *	pNameTable)
{
	char			ucBuffer[ 1024];
	char *		pucData = &ucBuffer[ 0];

	flmAssert( !isCached());
	flmAssert( !isReadOnly());
	
	return( importTree( pFileHdl, &pucData,
		sizeof( ucBuffer), pNameTable, this));
}

/*****************************************************************************
Desc: 	Import a record from a buffer
*****************************************************************************/
RCODE FlmRecord::importRecord(
	const char **		ppBuffer,
	FLMUINT				uiBufSize,
	F_NameTable *		pNameTable)
{
	flmAssert( !isCached());
	flmAssert( !isReadOnly());
	
	return( importTree( NULL, (char **)ppBuffer, uiBufSize, pNameTable, this));
}

/*****************************************************************************
Desc : This function parses and builds one complete GEDCOM tree from a GEDCOM
		 character buffer or file.  The beginning level number is used only to
		 delimit the tree (possiblly in forest); the generated tree's level
		 always start at zero.
*****************************************************************************/
FSTATIC RCODE importTree(
	F_FileHdl *				pFileHdl,
	char **					ppBuf,
	FLMUINT					uiBufSize,
	F_NameTable * 			pNameTable,
	FlmRecord *				pRec)
{
	GED_STREAM		gedStream;
	FLMUINT      	level;
	FLMUINT			levelBase = 0;
	FLMUINT			levelPrior = 0;
	FLMBYTE			nextChar;
	FLMBOOL			bFirstFieldProcessed = FALSE;
	RCODE				rc;
	FLMUINT			startPos;
	
	gedStream.pFileHdl = pFileHdl;
	gedStream.pThis = *ppBuf;
	gedStream.pBuf = *ppBuf;
	gedStream.uiBufSize = uiBufSize;

	pRec->clear();

	if( pFileHdl)
	{
		// Find 1st starting file position

		if( RC_OK( pFileHdl->Seek( 0L, F_IO_SEEK_CUR, &gedStream.uiFilePos)))
		{
			gedStream.pLast = gedStream.pBuf;
			gedReadChar( &gedStream, gedStream.uiFilePos);
		}
		else
			return( RC_SET( FERR_FILE_ER));
	}
	else
	{
		gedStream.errorIO = 0;
		gedStream.uiFilePos = 0;
		gedStream.pLast = gedStream.pBuf + (uiBufSize - 1);
		gedStream.thisC = f_toascii( *gedStream.pBuf);
	}

	for(;;)
	{
		gedSkipBlankLines( &gedStream);
		startPos = gedStream.uiFilePos;

		if( f_isdigit( gedStream.thisC))
		{
			level = 0;
			do
			{
				level = gedStream.thisC - ASCII_ZERO + (level * 10);
				nextChar = (FLMBYTE)(gedNextChar( &gedStream));
			} while( f_isdigit( nextChar));

			if( ! f_iswhitespace( gedStream.thisC))
			{
				rc = RC_SET( FERR_BAD_FIELD_LEVEL);
				break;
			}

			if( level > GED_MAXLVLNUM)
			{
				rc = RC_SET( FERR_GED_MAXLVLNUM);
				break;
			}

			if( bFirstFieldProcessed)
			{
				if( levelBase >= level)
				{
					goto successful;
				}
				else if( (levelPrior < level) &&	((levelPrior + 1) != level))
				{
					rc = RC_SET( FERR_GED_SKIP_LEVEL);
					break;
				}
			}
			else
			{
				levelBase = level;
			}
			levelPrior = level;

			// Process import tag and value

			rc = importField( (level - levelBase), &gedStream,
				pNameTable, pRec);

			if( RC_OK( rc))
			{
				bFirstFieldProcessed = TRUE;
				continue;
			}
		}
		else if( gedStream.thisC == '\0' || gedStream.thisC == ASCII_CTRLZ)
		{
			if( gedStream.errorIO)
			{
				rc = RC_SET( FERR_FILE_ER);
			}
			else if( bFirstFieldProcessed)
			{
successful:
				if( pFileHdl == NULL)
				{
					*ppBuf = gedStream.pThis + 
									(FLMINT32)(startPos - gedStream.uiFilePos);
				}
				gedStream.uiFilePos = startPos;
				rc = FERR_OK;
			}
			else
			{
				rc = RC_SET( FERR_END);
			}
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_LEVEL);
		}
		break;
	}

	if( rc != FERR_OK)
	{
		pRec->clear();
		if( pFileHdl == NULL)
		{
			*ppBuf = gedStream.pThis;
		}
	}

	if( pFileHdl)
	{
		pFileHdl->Seek( gedStream.uiFilePos, 
			F_IO_SEEK_SET, &gedStream.uiFilePos);
	}

	return( rc);
}

/*****************************************************************************
Desc:	Parse the tag, value, and length from a GEDCOM buffer, and insert it into
		a FlmRecord.  Continuation lines and embedded comments are also handled.
*****************************************************************************/
FSTATIC RCODE importField(
	FLMUINT			uiLevel,
	GED_STREAM *	pGedStream,
	F_NameTable * 	pNameTable,
	FlmRecord *		pRec)
{
	FLMUINT		startPos;
	RCODE			rc = FERR_OK;
	FLMUINT		drn = 0;
	FLMUINT		uiTagNum;
	char			tagBuf[ GED_MAXTAGLEN + 1];
	void *		pvField;

	gedSkipWhiteSpaces( pGedStream);

	// Process optional xref-id

	startPos = pGedStream->uiFilePos;
	if( pGedStream->thisC == ASCII_AT)
	{
		int badDRN;
		for( badDRN = 0, gedNextChar( pGedStream);
			  pGedStream->thisC != ASCII_AT;
			  gedNextChar( pGedStream))
		{
			FLMUINT	priorDrn = drn;

			if( ! badDRN)
			{
				if( f_isdigit( pGedStream->thisC))
				{
					drn = (drn * 10) + pGedStream->thisC - ASCII_ZERO;
					badDRN = priorDrn != (drn / 10);
				}
				else
				{
					badDRN = 1;
				}
			}
		}

		if( badDRN)
		{
			drn = 0;
		}

		gedNextChar( pGedStream);
		if( f_iswhitespace( pGedStream->thisC))
		{
			gedSkipWhiteSpaces( pGedStream);
		}
		else
		{
			rc = RC_SET( FERR_GED_BAD_RECID);
			goto Exit;
		}
	}

	if( drn)
	{
		// Record can only have one ID (DRN)

		flmAssert( pRec->getID() == 0);
		pRec->setID( drn);
	}

	// Determine the Tag Number and insert a new field

	startPos = pGedStream->uiFilePos;

	if( !gedCopyTag( pGedStream, tagBuf))
	{
		return( RC_SET( FERR_INVALID_TAG));
	}

	if( !pNameTable->getFromTagTypeAndName( NULL, tagBuf,
		FLM_FIELD_TAG, &uiTagNum))
	{
		// See if tag is the reserved tag with the number following

		if( tagBuf[0] == f_toascii( 'T') &&
			 tagBuf[1] == f_toascii( 'A') &&
			 tagBuf[2] == f_toascii( 'G') &&
			 tagBuf[3] == f_toascii( '_'))
		{
			uiTagNum = f_atoi( &tagBuf[ 4]);
		}
		else
		{
			return( RC_SET( FERR_NOT_FOUND));
		}
	}

	if( RC_BAD( rc = pRec->insertLast( uiLevel, uiTagNum,
		FLM_TEXT_TYPE, &pvField)))
	{
		goto Exit;
	}

	gedSkipWhiteSpaces( pGedStream);

	// Alternate xref_ptr used instead of "value"

	startPos = pGedStream->uiFilePos;
	if( pGedStream->thisC == ASCII_AT)
	{
		for( drn = 0; gedNextChar( pGedStream) != ASCII_AT;)
		{
			FLMUINT	priorDrn = drn;
			if( f_isdigit( pGedStream->thisC))
			{
				drn = (drn * 10) + pGedStream->thisC - ASCII_ZERO;
				if( priorDrn == (drn / 10))
				{
					continue;
				}
			}

			rc = RC_SET( FERR_GED_BAD_VALUE);
			goto Exit;
		}
		gedNextChar( pGedStream);

		if( RC_BAD( rc = pRec->setUINT( pvField, drn)))
		{
			goto Exit;
		}

		if( gedCopyValue( pGedStream, NULL))
		{
			rc = RC_SET( FERR_GED_BAD_VALUE);
			goto Exit;
		}
	}
	else
	{
		FLMINT	valLength;
		FLMUINT	tempPos = pGedStream->uiFilePos;

		if( (valLength = gedCopyValue( pGedStream, NULL)) > 0)
		{
			FLMBYTE * vp;

			 if (RC_BAD( rc = pRec->allocStorageSpace(
													pvField, FLM_TEXT_TYPE, valLength,
													0, 0, 0, &vp, NULL)))
			{
				goto Exit;
			}
			gedReadChar( pGedStream, tempPos);
			gedCopyValue( pGedStream, (char *)vp);
		}
	}

	startPos = pGedStream->uiFilePos;

Exit:

	gedReadChar( pGedStream, startPos);
	return( rc);
}

/*****************************************************************************
Desc:	Parse the tag and, if `dest' is not NULL, copy and terminating w/null
*****************************************************************************/
FLMUINT gedCopyTag(
	GED_STREAM *	pGedStream,
	char *			dest)
{
	if( f_isalpha( pGedStream->thisC) || pGedStream->thisC == ASCII_UNDERSCORE )
	{
		int	tagLen, maxTag;

		for( tagLen = 0, maxTag = GED_MAXTAGLEN;
			f_isalnum( pGedStream->thisC) || pGedStream->thisC == ASCII_UNDERSCORE;
			gedNextChar( pGedStream))
		{
			tagLen++;
			if( dest && maxTag)
			{
				*dest++ = (FLMBYTE)(pGedStream->thisC);
				maxTag--;
			}
		}

		if( dest)
		{
			*dest = '\0';
		}

		switch( pGedStream->thisC)
		{
			case '\0':
				if( pGedStream->errorIO)
				{
					break;
				}

			case ASCII_SPACE:
			case ASCII_NEWLINE:
			case ASCII_CR:
			case ASCII_TAB:
			case ASCII_CTRLZ:
			{
				return( (FLMUINT) tagLen);
			}
		}
	}

	return( 0);
}

/*****************************************************************************
Desc:	Parse the value and, if `dest' is not NULL, copy and terminating w/null
*****************************************************************************/
FLMINT gedCopyValue(
	GED_STREAM *	pGedStream,
	char *			dest)
{
	FLMUINT		OEMcount = 0;
	FLMUINT		valLength = 0;
	FLMUINT		wsCount;
	FLMUINT		lineLength = 0;
	FLMUINT		firstPos = 0;
	FLMBYTE		c;

	for(;;)
	{
		gedSkipWhiteSpaces( pGedStream);
		if( pGedStream->thisC == ASCII_DQUOTE)
		{
			gedNextChar( pGedStream);
			firstPos = pGedStream->uiFilePos;
		}
		else
		{
			firstPos = 0;
		}

		for( lineLength = 0;;)
		{
			switch( pGedStream->thisC)
			{
				case ASCII_DQUOTE:
				case ASCII_SPACE:
				case ASCII_TAB:
					for( wsCount = 0;;)
					{
						switch( gedNextChar( pGedStream))
						{
							case '\0':
								if( pGedStream->errorIO)
								{
									return( 0);
								}

							case ASCII_NEWLINE:
							case ASCII_CR:
							case ASCII_CTRLZ:
								goto eol;

							case ASCII_SPACE:
							case ASCII_TAB:
								wsCount++;
								continue;

							default:
								gedReadChar( pGedStream, pGedStream->uiFilePos - 1);
								lineLength += wsCount;
								goto valid;
						}
					}

				case '\0':
					if( pGedStream->errorIO)
					{
						return( 0);
					}
				case ASCII_NEWLINE:
				case ASCII_CR:
				case ASCII_CTRLZ:
eol:				if( lineLength)
					{
						valLength += lineLength;
						if( dest)
						{
							gedReadChar( pGedStream, firstPos);
							while( lineLength--)
							{
								c = (FLMBYTE)pGedStream->thisC;

								if ((c >= ASCII_SPACE) && (c < 127))
								{
									*dest++ = c;
								}
								else if (c == ASCII_TAB)
								{
									*dest++ = WHITE_SPACE_CODE | NATIVE_TAB;
								}
								else if (c == ASCII_NEWLINE)
								{
									*dest++ = WHITE_SPACE_CODE | NATIVE_LINEFEED;
								}
								else if (c == ASCII_CR)
								{
									*dest++ = WHITE_SPACE_CODE | HARD_RETURN;
								}
								else
								{
									*dest++ = (c >= 127)
														? OEM_CODE
														: UNK_EQ_1_CODE | NATIVE_TYPE;
									*dest++ = c;
								}
								gedNextChar( pGedStream);
							}

							if( pGedStream->thisC == ASCII_DQUOTE)
							{
								gedNextChar( pGedStream);
							}
						}
					}
					break;

				default:
valid:
					c = (FLMBYTE)pGedStream->thisC;
					if (((c < ASCII_SPACE) &&
						  (c != ASCII_TAB) &&
						  (c != ASCII_CR) &&
						  (c != ASCII_NEWLINE)) ||
							(c >= 127))
					{
						OEMcount += 1;
					}

					if( !firstPos)
					{
						firstPos = pGedStream->uiFilePos;
					}
					lineLength++;
					gedNextChar( pGedStream);
					continue;
			}
			break;
		}

		gedSkipBlankLines( pGedStream);
		if( pGedStream->thisC != ASCII_PLUS)
		{
			break;
		}
		gedNextChar( pGedStream);
	}

	return( valLength + OEMcount);
}

/*****************************************************************************
Desc:	strip blank lines from buffer by repositioning pointer
*****************************************************************************/
void  gedSkipBlankLines(
	GED_STREAM *		pGedStream)
{
	for(;;)
	{
		gedSkipWhiteSpaces( pGedStream);
		switch( pGedStream->thisC)
		{
			case ASCII_POUND:
				for(;;)
				{
					switch( gedNextChar( pGedStream))
					{
						case ASCII_NEWLINE:
						case ASCII_CR:
							goto outerFor;
						case '\0':
						case ASCII_CTRLZ:
							return;
					}
				}
			case ASCII_NEWLINE:
			case ASCII_CR:
outerFor:
				gedNextChar( pGedStream);
				continue;
		}
		return;
	}
}

/*****************************************************************************
Desc:	get current character then increment pointer.
		buffer may be refilled if file available
*****************************************************************************/
FLMINT gedNextChar(
	GED_STREAM *		pGedStream)
{
	pGedStream->errorIO = 0;
	if( pGedStream->pThis < pGedStream->pLast)
	{
		pGedStream->pThis++;
		goto returnC;
	}

	if( pGedStream->pFileHdl)
	{
		FLMUINT	bytesRead;
		RCODE		rc;

		rc = pGedStream->pFileHdl->Read( F_IO_CURRENT_POS,
			pGedStream->uiBufSize, pGedStream->pBuf, &bytesRead);

		if( rc == FERR_OK	|| (rc == FERR_IO_END_OF_FILE && bytesRead))
		{
			pGedStream->pThis = pGedStream->pBuf;
			pGedStream->pLast = pGedStream->pBuf + (bytesRead - 1);
returnC:
			pGedStream->uiFilePos++;
			return( pGedStream->thisC = f_toascii( *pGedStream->pThis));
		}
		pGedStream->errorIO = rc != FERR_IO_END_OF_FILE;
	}
	return( pGedStream->thisC = '\0');
}

/*****************************************************************************
Desc:	get an ASCII character at absolute file position and reset pointers
*****************************************************************************/
FLMINT gedReadChar(
	GED_STREAM *	pGedStream,
	FLMUINT			uiFilePos)
{
	RCODE				rc = FERR_OK;
	
	pGedStream->errorIO = 0;

	if( pGedStream->pFileHdl)
	{
		FLMUINT		bytesRead;
		char *		pszTemp;
		
		pszTemp = pGedStream->pThis + 
						(FLMINT32)(uiFilePos - pGedStream->uiFilePos);

		if( pGedStream->pBuf == pGedStream->pLast ||
			 pszTemp > pGedStream->pLast || pszTemp < pGedStream->pBuf)
		{
			if( RC_OK(rc = pGedStream->pFileHdl->Seek( uiFilePos,
					F_IO_SEEK_SET, &pGedStream->uiFilePos)) &&
				(RC_OK( rc = pGedStream->pFileHdl->Read( F_IO_CURRENT_POS, 
					pGedStream->uiBufSize, pGedStream->pBuf, &bytesRead)) ||
					(rc == FERR_IO_END_OF_FILE && bytesRead)))
			{
				pGedStream->pThis = pGedStream->pBuf;
				pGedStream->pLast = pGedStream->pBuf + (bytesRead - 1);
				return( pGedStream->thisC = f_toascii( *pGedStream->pThis));
			}
			else
			{
				pGedStream->errorIO = 1;
				return( pGedStream->thisC = '\0');
			}
		}
		else
		{
			pGedStream->uiFilePos = uiFilePos;
			pGedStream->pThis = pszTemp;
			return( pGedStream->thisC = f_toascii( *pszTemp));
		}
	}

	if( (pGedStream->pBuf + uiFilePos) > pGedStream->pLast)
	{
		return( pGedStream->thisC = '\0');
	}

	pGedStream->pThis = pGedStream->pBuf + uiFilePos;
	pGedStream->uiFilePos = uiFilePos;

	return( pGedStream->thisC = f_toascii( *pGedStream->pThis));
}

/*****************************************************************************
Desc:		Import a record from GEDCOM
*****************************************************************************/
RCODE FlmRecord::importRecord(
	NODE *		pNode)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucData;
	FLMBYTE *	pucEncData;
	HFDB			hDb;
	FLMUINT		uiContainerID = 0;
	FLMUINT		uiRecordID = 0;
	void *		pvField;
	FlmField *	pField;
	FLMUINT		uiDataType;
	FLMUINT		uiFlags;

	flmAssert( pNode);
	flmAssert( !isCached());
	flmAssert( !isReadOnly());
	
	clear();

	// Copy any source info that is within the root node.

	if( RC_OK( GedGetRecSource( pNode, &hDb, &uiContainerID, &uiRecordID)))
	{
		setContainerID( uiContainerID);
		setID( uiRecordID);
	}

	while( pNode)
	{
		uiDataType = GedValType( pNode);
		if( RC_BAD( rc = insertLast( GedNodeLevel( pNode), 
			GedTagNum( pNode), uiDataType, &pvField)))
		{
			goto Exit;
		}

		pField = getFieldPointer( pvField);

		if( GedValLen( pNode) || GedEncLen( pNode))
		{
			uiFlags = pNode->ui32EncFlags;

			if( RC_BAD( rc = getNewDataPtr( pField, uiDataType,
				pNode->ui32Length, pNode->ui32EncLength, pNode->ui32EncId, uiFlags,
				(!uiFlags || uiFlags & FLD_HAVE_DECRYPTED_DATA)
													? &pucData
													: NULL,
				(uiFlags & FLD_HAVE_ENCRYPTED_DATA)
													? &pucEncData
													: NULL)))
			{
				goto Exit;
			}

			if (!uiFlags || uiFlags & FLD_HAVE_DECRYPTED_DATA)
			{
				f_memcpy( pucData, GedValPtr( pNode), GedValLen( pNode));
			}
			
			if ( uiFlags & FLD_HAVE_ENCRYPTED_DATA)
			{
				f_memcpy( pucEncData, GedEncPtr( pNode), GedEncLen( pNode));
			}

		}
		// Copy any of the value flags that this node may have

		if( GedIsRightTruncated( pNode))
		{
			setRightTruncated( pField, TRUE);
		}

		if( GedIsLeftTruncated( pNode))
		{
			setLeftTruncated( pField, TRUE);
		}

		pNode = GedNodeNext( pNode);
		if( pNode && GedNodeLevel( pNode) == 0)
		{
			break;
		}
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:		Export the record to GEDCOM
*****************************************************************************/
RCODE FlmRecord::exportRecord(
	HFDB			hDb,
	POOL *		pPool,
	NODE **		ppRoot)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	pucData;
	FLMBYTE *	pucEncData = NULL;
	FLMBYTE *	exPtr;
	FlmField *	pField = getFieldPointer(root());
	NODE *		pPrevNode = NULL;
	NODE *		pNode = NULL;
	FLMUINT		uiLevelBase = getFieldLevel( pField);
	FLMUINT		uiEncLength = 0;

	flmAssert( ppRoot != NULL);
	flmAssert( pPool != NULL);

	*ppRoot = NULL;

	while( pField)
	{
		if( *ppRoot)
		{
			if( (pNode = GedNodeMake( pPool,
				pField->ui16FieldID, &rc)) == NULL)
			{
				goto Exit;
			}
		}
		else
		{
			// Create root source node

			if( RC_BAD( rc = gedCreateSourceNode( pPool, pField->ui16FieldID,
				hDb, getContainerID(), getID(), &pNode)))
			{
				goto Exit;
			}
		}

		// Link in new node

		if( pPrevNode)
		{
			pPrevNode->next = pNode;
		}
		else
		{
			*ppRoot = pNode;
		}

		pNode->prior = pPrevNode;
		GedNodeLevelSet( pNode, (getFieldLevel( pField) - uiLevelBase));
		pPrevNode = pNode;

		if (isEncryptedField(pField))
		{
			uiEncLength = getEncryptedDataLength( pField);
			pucData = (FLMBYTE *)GedAllocSpace( pPool,
															pNode,
															getFieldDataType( pField),
															getFieldDataLength( pField),
															getEncryptionID( pField),
															uiEncLength);
			pucEncData = (FLMBYTE *)GedEncPtr( pNode);

			if( !pucData && uiEncLength)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
		else
		{
			pucData = (FLMBYTE *)GedAllocSpace( pPool, pNode,
								getFieldDataType( pField), getFieldDataLength( pField));

			if( !pucData && getFieldDataLength( pField))
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}

		// Directly copy over the NODE value pucData.
		if( getFieldDataLength( pField))
		{
			if( (exPtr = getDataPtr( pField)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
			f_memcpy( pucData, exPtr, getFieldDataLength( pField));
			if (pNode->ui32EncId)
			{
				flmAssert( pNode->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA);
			}
		}

		if (pNode->ui32EncId)
		{
			if ((exPtr = getEncryptionDataPtr( pField)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			flmAssert( uiEncLength);
			flmAssert( pNode->ui32EncFlags & FLD_HAVE_ENCRYPTED_DATA);
			f_memcpy( pucEncData, exPtr, uiEncLength);
		}

		// Copy any of the value flags that this node may have

		if( isRightTruncated( pField))
		{
			GedSetRightTruncated( pNode);
		}

		if( isLeftTruncated( pField))
		{
			GedSetLeftTruncated( pNode);
		}

		pField = nextField( pField);
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
*****************************************************************************/
void * FlmRecord::getFieldVoid(
	FlmField *		pField)
{
	FLMUINT		uiSlot;

	if( !pField)
	{
		return( NULL);
	}

	if( !m_uiFldTblOffset ||
		pField > &(getFieldTable()[ m_uiFldTblOffset - 1]))
	{
		flmAssert( 0);
		return( NULL);
	}

	uiSlot = (pField - getFieldTable()) + 1;
	return( (void *)uiSlot);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FlmField * FlmRecord::getFieldPointer(
	void *			pvField)
{
	FLMUINT		uiSlot = (FLMUINT)pvField;
	FlmField *	pField = NULL;

	if( !uiSlot)
	{
		goto Exit;
	}

	if( uiSlot > m_uiFldTblOffset)
	{
		flmAssert( 0);
		goto Exit;
	}

	pField = &(getFieldTable()[ uiSlot - 1]);
	flmAssert( pField->ui16FieldID);

Exit:

	return( pField);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FLMBYTE * FlmRecord::getDataPtr(
	FlmField *		pField)
{
	if (!isEncryptedField(pField))
	{
		if( !pField->ui8DataLen)
		{
			return( NULL);
		}
		else if( pField->ui8DataLen <= 4)
		{
			return( (FLMBYTE *)&(pField->ui32DataOffset));
		}
		else if( pField->ui8DataLen < 0xFF)
		{
			return( getDataBufPtr() + pField->ui32DataOffset);
		}
		else
		{
			FLMUINT	uiOffset = pField->ui32DataOffset + sizeof( FLMUINT16) + 1;

			if( getFieldDataType( pField) == FLM_BINARY_TYPE)
			{
				if( (uiOffset & FLM_ALLOC_ALIGN) != 0)
				{
					uiOffset += (FLM_ALLOC_ALIGN + 1) - (uiOffset & FLM_ALLOC_ALIGN);
				}
			}

			return( getDataBufPtr() + uiOffset);
		}
	}
	else
	{
		FLMUINT	uiOffset = pField->ui32DataOffset + FLM_ENC_FLD_OVERHEAD;
		FLMUINT	uiPicketFenceSize = 0;

#ifdef FLM_PICKET_FENCE
		uiPicketFenceSize = FLD_PICKET_FENCE_SIZE / 2;
#endif

		if( getFieldDataType( pField) == FLM_BINARY_TYPE)
		{
			if( (uiOffset & FLM_ALLOC_ALIGN) != 0)
			{
				uiOffset += (FLM_ALLOC_ALIGN + 1) - (uiOffset & FLM_ALLOC_ALIGN);
			}
		}

		return ( getDataBufPtr() + uiOffset + uiPicketFenceSize);
	}
}

/*****************************************************************************
Desc:	Get the pointer to the encrypted data.
*****************************************************************************/
FLMBYTE * FlmRecord::getEncryptionDataPtr(
	FlmField *		pField)
{
	FLMBYTE *	pucTmp = NULL;
	FLMUINT		uiOffset;
	FLMUINT		uiPicketFenceSize = 0;

	// We assume that this field is encrypted.  If it isn't,
	// then we will return NULL.
	if (!isEncryptedField(pField))
	{
		flmAssert( 0);
		goto Exit;
	}

#ifdef FLM_PICKET_FENCE
	uiPicketFenceSize = FLD_PICKET_FENCE_SIZE;
#endif

	uiOffset = pField->ui32DataOffset + FLM_ENC_FLD_OVERHEAD;

	if( getFieldDataType( pField) == FLM_BINARY_TYPE)
	{
		if( (uiOffset & FLM_ALLOC_ALIGN) != 0)
		{
			uiOffset += (FLM_ALLOC_ALIGN + 1) - (uiOffset & FLM_ALLOC_ALIGN);
		}
	}

	pucTmp = getDataBufPtr() + uiOffset + getFieldDataLength( pField) +
																		uiPicketFenceSize;

Exit:

	return pucTmp;
}

/*****************************************************************************
Desc:
*****************************************************************************/
FlmField * FlmRecord::getFirstField( void)
{
	if( !m_uiFldTblOffset)
	{
		return( NULL);
	}

	flmAssert( (getFieldTable()[ 0]).ui16FieldID);
	return( &(getFieldTable()[ 0]));
}

/*****************************************************************************
Desc:
*****************************************************************************/
FlmField * FlmRecord::getLastField( void)
{
	FLMUINT		uiLoop;
	FlmField *	pField = NULL;

	if( !m_uiFldTblOffset)
	{
		goto Exit;
	}

	for( uiLoop = m_uiFldTblOffset; uiLoop; uiLoop--)
	{
		pField = &(getFieldTable()[ uiLoop - 1]);

		if( pField->ui16FieldID)
		{
			break;
		}
	}

Exit:

	return( pField);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FlmField * FlmRecord::prevField(
	FlmField *		pField)
{
	if( !pField)
	{
		return( NULL);
	}

	return( getFieldPointer( (void *)((FLMUINT)(pField->uiPrev))));
}

/*****************************************************************************
Desc:
*****************************************************************************/
FlmField * FlmRecord::nextField(
	FlmField *		pField)
{
	if( !pField)
	{
		return( NULL);
	}

	return( getFieldPointer( (void *)((FLMUINT)(pField->uiNext))));
}

/*****************************************************************************
Desc:
*****************************************************************************/
FLMUINT FlmRecord::getTotalMemory( void)
{
	return( gv_FlmSysData.RCacheMgr.pRecBufAlloc->getTrueSize(
		m_uiBufferSize, m_pucBuffer) + sizeof( FlmRecord));
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::setFieldLevel(
	FlmField *		pField,
	FLMUINT			uiLevel)
{
	flmAssert( !isCached());
	flmAssert( !isReadOnly());
	
	if( uiLevel > 7)
	{
		flmAssert( 0);
		return( RC_SET( FERR_BAD_FIELD_LEVEL));
	}

	pField->ui8TypeAndLevel &= 0x1F;
	pField->ui8TypeAndLevel |= (((FLMUINT8)uiLevel) << 5);

	return( FERR_OK);
}

/*****************************************************************************
Desc:
*****************************************************************************/
FLMUINT FlmRecord::getFieldDataLength(
	FlmField *			pField)
{
	FLMBYTE *		pucLength;
	FLMUINT			uiDataLen = 0;

	flmAssert( pField);

	if( pField->ui8DataLen != 0xFF)
	{
		uiDataLen = pField->ui8DataLen;
		goto Exit;
	}

	pucLength = getDataBufPtr() + pField->ui32DataOffset + 1;

	if (isEncryptedField( pField))
	{
		pucLength += 2;
	}

	uiDataLen = FB2UW( pucLength);

Exit:

	return uiDataLen;
}

/*****************************************************************************
Desc:	
*****************************************************************************/
void FlmRecord::setFieldDataType(
	FlmField *		pField,
	FLMUINT			uiDataType)
{
	flmAssert( !isCached());
	flmAssert( !isReadOnly());
	
	pField->ui8TypeAndLevel &= 0xF8;

	if( uiDataType <= 3)
	{
		pField->ui8TypeAndLevel |= (FLMUINT8)uiDataType;
	}
	else
	{
		flmAssert( uiDataType == FLM_BLOB_TYPE);
		pField->ui8TypeAndLevel |= (FLMUINT8)FLD_BLOB_TYPE;
	}
}

/*****************************************************************************
Desc:	Function to determine if the field is encrypted.
*****************************************************************************/
FLMBOOL FlmRecord::isEncryptedField(
	FlmField *			pField
	)
{
	FLMBOOL		bResult = FALSE;
	FLMUINT		uiFlags;
	FLMBYTE *	pucBuffer;

	if (!pField)
	{
		goto Exit;
	}

	if (pField->ui8DataLen == 0xFF)
	{
		pucBuffer = getDataBufPtr() + pField->ui32DataOffset;
		uiFlags = pucBuffer[ FLD_ENC_FLAGS];
		if (uiFlags && uiFlags <= 0x03)
			{
			bResult = TRUE;
		}
	}
Exit:
	return bResult;
}

/*****************************************************************************
Desc:	Function to get the length of the encrypted data.
*****************************************************************************/
FLMUINT FlmRecord::getEncryptedDataLength(
	FlmField *			pField
	)
{
	FLMBYTE *		pucBuffer = NULL;
	FLMUINT			uiEncDataLength = 0;

	// Make sure this is an encrypted field.
	if (!isEncryptedField(pField))
	{
		flmAssert( 0);
		goto Exit;
	}

	pucBuffer = getDataBufPtr() + pField->ui32DataOffset;
	uiEncDataLength = FB2UW( &pucBuffer[ FLD_ENC_ENCRYPTED_DATA_LEN]);

Exit:

	return uiEncDataLength;

}


/*****************************************************************************
Desc:	Function to get the DRN of the encryption definition record in the
		dictionary.
*****************************************************************************/
FLMUINT FlmRecord::getEncryptionID(
	FlmField *			pField
	)
{
	FLMBYTE *	pucBuffer = NULL;
	FLMUINT		uiEncId = 0;

	if (!isEncryptedField( pField))
	{
		flmAssert( 0);
		goto Exit;
	}

	pucBuffer = getDataBufPtr() + pField->ui32DataOffset;

	uiEncId = FB2UW( &pucBuffer[ FLD_ENC_ENCID]);

Exit:

	return uiEncId;

}

/*****************************************************************************
Desc:
*****************************************************************************/
FLMUINT FlmRecord::getEncFlags(
	FlmField *			pField
	)
{
	FLMBYTE *	pucBuffer;

	if (!isEncryptedField( pField))
	{
		flmAssert( 0);
		return( 0);
	}

	pucBuffer = getDataBufPtr() + pField->ui32DataOffset;

	return( (FLMUINT)pucBuffer[ FLD_ENC_FLAGS]);

}

/*****************************************************************************
Desc:
*****************************************************************************/
void FlmRecord::setEncFlags(
	FlmField *		pField,
	FLMUINT			uiFlags
	)
{
	FLMBYTE *	pucBuffer;

	if (!isEncryptedField( pField))
	{
		flmAssert( 0);
		return;
	}
	pucBuffer = getDataBufPtr() + pField->ui32DataOffset;
	pucBuffer[ FLD_ENC_FLAGS] = (FLMBYTE)uiFlags;
}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::getINT(
	void *			pvField,
	FLMINT *			piNumber)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bEncrypted;
	FlmField *	pField = getFieldPointer( pvField);

	if (!pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	bEncrypted = isEncryptedField( pField);

	if (!bEncrypted ||
		 (getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = FlmStorage2INT( getFieldDataType( pField),
									getFieldDataLength( pField),
									(const FLMBYTE *)getDataPtr( pField),
									piNumber);
	}
	else if (bEncrypted &&
		 		!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);

}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::getUINT(
	void *			pvField,
	FLMUINT *		puiNumber)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bEncrypted;
	FlmField *	pField = getFieldPointer( pvField);

	if (!pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	bEncrypted = isEncryptedField( pField);

	if (!bEncrypted ||
		 (getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = FlmStorage2UINT( getFieldDataType( pField),
									 getFieldDataLength( pField),
									 (const FLMBYTE *)getDataPtr( pField),
									 puiNumber);
	}
	else if (bEncrypted &&
		 		!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);

}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::getUINT32(
	void *			pvField,
	FLMUINT32 *		pui32Number)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bEncrypted;
	FlmField *	pField = getFieldPointer( pvField);

	if (!pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	bEncrypted = isEncryptedField( pField);

	if (!bEncrypted ||
	 	 (getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = FlmStorage2UINT32(	getFieldDataType( pField),
										getFieldDataLength( pField),
										(const FLMBYTE *)getDataPtr( pField),
										pui32Number);
	}
	else if (bEncrypted &&
		 		!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);

}


/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::getUnicode(
	void *			pvField,
	FLMUNICODE *	pUnicode,
	FLMUINT *		puiBufLen)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bEncrypted;
	FlmField *	pField = getFieldPointer( pvField);

	if (!pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	bEncrypted = isEncryptedField( pField);

	if (!bEncrypted ||
		 (getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = FlmStorage2Unicode( getFieldDataType( pField),
								 		 getFieldDataLength( pField),
								 		 (const FLMBYTE *)getDataPtr( pField),
								 		 puiBufLen,
								 		 pUnicode);
	}
	else if (bEncrypted &&
		 		!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);

}

/*****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmRecord::getNative(
	void *			pvField,
	char *			pszString,
	FLMUINT *		puiBufLen)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bEncrypted;
	FlmField *	pField = getFieldPointer( pvField);

	if (!pField)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	bEncrypted = isEncryptedField( pField);

	if (!bEncrypted ||
		 (getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = FlmStorage2Native( getFieldDataType( pField),
										getFieldDataLength( pField),
										(const FLMBYTE *)getDataPtr( pField),
										puiBufLen,
										pszString);
	}
	else if (bEncrypted &&
		 		!(getEncFlags( pField) & FLD_HAVE_DECRYPTED_DATA))
	{
		rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
		goto Exit;
	}
	else
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:

	return( rc);

}

/*****************************************************************************
Desc:	Set the header fields in an encrypted field.
*****************************************************************************/
void FlmRecord::setEncHeader(
	FLMBYTE *		pBuffer,
	FLMUINT			uiFlags,
	FLMUINT			uiEncId,
	FLMUINT			uiNewLength,
	FLMUINT			uiEncNewLength)
{
	*pBuffer = (FLMBYTE)uiFlags;
	pBuffer++;
	UW2FBA( uiEncId, pBuffer);
	pBuffer += 2;
	UW2FBA( uiNewLength, pBuffer);
	pBuffer += 2;
	UW2FBA( uiEncNewLength, pBuffer);

}

/*****************************************************************************
Desc:
*****************************************************************************/
void * FlmRecord::locateFieldByPosition(
	 FLMUINT			uiPosition)
{
	FlmField *		pField = getFieldPointer( root());
	FLMUINT			uiLoop;

	flmAssert( pField);
	
	for (uiLoop = 1; pField && uiLoop < uiPosition; uiLoop++)
	{
		pField = nextField( pField);
	}

	return (getFieldVoid( pField));
}

#undef new
#undef delete

/****************************************************************************
Desc:
****************************************************************************/
void * FlmRecord::operator new(
	FLMSIZET			uiSize)
#ifndef FLM_NLM	
	throw()
#endif
{
	F_UNREFERENCED_PARM( uiSize);
	flmAssert( gv_FlmSysData.RCacheMgr.pRecAlloc->getCellSize() >= uiSize);
	return( gv_FlmSysData.RCacheMgr.pRecAlloc->allocCell());
}

/****************************************************************************
Desc:
****************************************************************************/
void * FlmRecord::operator new[]( FLMSIZET)
#ifndef FLM_NLM	
	throw()
#endif
{
	flmAssert( 0);
	return( NULL);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
void * FlmRecord::operator new(
	FLMSIZET			uiSize,
	const char *,
	int)
#ifndef FLM_NLM	
	throw()
#endif
{
	flmAssert( gv_FlmSysData.RCacheMgr.pRecAlloc->getCellSize() >= uiSize);
	return( gv_FlmSysData.RCacheMgr.pRecAlloc->allocCell());
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_DEBUG
void * FlmRecord::operator new[](
	FLMSIZET,		// uiSize,
	const char *,	// pszFile,
	int)				// iLine)
#ifndef FLM_NLM	
	throw()
#endif
{
	flmAssert( 0);
	return( NULL);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
void FlmRecord::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}
	
	flmAssert( ((FlmRecord *)ptr)->m_ui32RefCnt == 0);
	gv_FlmSysData.RCacheMgr.pRecAlloc->freeCell( ptr);
}

/****************************************************************************
Desc:
****************************************************************************/
void FlmRecord::operator delete[](
	void *		// ptr)
	)
{
	flmAssert( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( __WATCOMC__)
void FlmRecord::operator delete( 
	void *			ptr,
	const char *,	// file
	int				// line
	)
{
	if( !ptr)
	{
		return;
	}

	flmAssert( ((FlmRecord *)ptr)->m_ui32RefCnt == 0);
	gv_FlmSysData.RCacheMgr.pRecAlloc->freeCell( ptr);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( __WATCOMC__)
void FlmRecord::operator delete[](
	void *,			// ptr,
	const char *,	// file
	int				// line
	)
{
	flmAssert( 0);
}

#endif

/*****************************************************************************
Desc:	Verify the structural integrity of the FlmRecord
*****************************************************************************/
RCODE FlmRecord::checkRecord()
{
	RCODE			rc = FERR_OK;
	FlmField *	pFld;

	// Check each field to make sure it is acceptable.
	pFld = getFieldPointer( root());
	while( pFld)
	{
		if (RC_BAD( rc = checkField( pFld)))
		{
			flmAssert( 0);
			goto Exit;
		}

		pFld = nextField( pFld);
	}

Exit:

	return( rc);
}


/******************************************************************************
Desc:	Verify the structure and content of the FlmField
******************************************************************************/
RCODE FlmRecord::checkField(
	 FlmField *			pFld)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pucFldBuffer;
	FLMUINT16		ui16DataLen;
	FLMUINT16		ui16EncDataLen;
	FLMUINT			uiAlignment = 0;
	FLMBYTE *		pucDataFence = NULL;
	FLMBYTE *		pucEncFence = NULL;
	FLMUNICODE *	pUnicode = NULL;
	FLMUINT			uiBufLen;
	FLMUINT			uiEncID;
	FLMUINT			uiSlot = (FLMUINT)getFieldVoid( pFld);

	// Determine if the field may be encrypted.  Fields >= 255 and
	// encrypted fields have their actual length stored in the buffer.

	if ( pFld->ui8DataLen < 0xFF)
	{
		goto Exit;
	}

	// Get the buffer pointer and test it to make sure the structure is okay.

	pucFldBuffer = getDataBufPtr() + pFld->ui32DataOffset;

	// Test for encryption.  If the field is not encrypted, we won't do any check
	// right now.

	if (*pucFldBuffer == 0)
	{
		goto Exit;
	}

	// If we have any other value now besides 1, 2 or 3, we have some corruption!

	flmAssert( *pucFldBuffer <= 3);

	// Make sure we have an encryption ID and that it appears to be within range
	// at least.

	uiEncID = FB2UW( &pucFldBuffer[ FLD_ENC_ENCID]);

	if (!uiEncID)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if (uiEncID > FLM_RESERVED_TAG_NUMS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_BAD_ENCDEF_ID);
		goto Exit;
	}

	ui16DataLen = FB2UW( &pucFldBuffer[ FLD_ENC_DATA_LEN]);
	if (!ui16DataLen)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	ui16EncDataLen = FB2UW( &pucFldBuffer[ FLD_ENC_ENCRYPTED_DATA_LEN]);
	if (!ui16EncDataLen)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}


	// Adjust for alignment issues of binary fields.
	if( getFieldDataType( pFld) == FLM_BINARY_TYPE)
	{
		// Since the value is binary, need to account for any
		// alignment bytes

		if( ((pFld->ui32DataOffset + FLM_ENC_FLD_OVERHEAD) & FLM_ALLOC_ALIGN) != 0)
		{
			uiAlignment = (FLM_ALLOC_ALIGN + 1) -
				((pFld->ui32DataOffset + FLM_ENC_FLD_OVERHEAD) & FLM_ALLOC_ALIGN);
		}
	}

#ifdef FLM_PICKET_FENCE
	// Verify the picket fences.

	pucDataFence = pucFldBuffer + FLM_ENC_FLD_OVERHEAD + uiAlignment;
	pucEncFence = pucDataFence + ui16DataLen + 4;

	if ( f_memcmp( pucDataFence, FLD_RAW_FENCE, FLD_PICKET_FENCE_SIZE / 2))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if ( f_memcmp( pucEncFence, FLD_ENC_FENCE, FLD_PICKET_FENCE_SIZE / 2))
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
#else
	F_UNREFERENCED_PARM( pucDataFence);
	F_UNREFERENCED_PARM( pucEncFence);
#endif

	// Verify the raw data if it is a text type.

	if( getFieldDataType( pFld) == FLM_TEXT_TYPE)
	{
		// Let's try to get this as unicode.
		// Allocate a temporary buffer.

		uiBufLen = (ui16DataLen * 2) + 2;
		if (RC_BAD( rc = f_alloc( uiBufLen, &pUnicode)))
		{
			goto Exit;
		}

		if (RC_BAD( rc = getUnicode( (void *)uiSlot,
											  pUnicode,
											  &uiBufLen)))
		{
			if (rc == FERR_ENCRYPTION_UNAVAILABLE)
			{
				rc = FERR_OK;
			}
			else
			{
				goto Exit;
			}
		}

		f_free( &pUnicode);
		pUnicode = NULL;
	}
	else if ( getFieldDataType( pFld) == FLM_NUMBER_TYPE)
	{
		FLMUINT	uiTemp;
		
		if ( RC_BAD( rc = getUINT( (void *)uiSlot, &uiTemp)))
		{
			if ( rc == FERR_ENCRYPTION_UNAVAILABLE)
			{
				rc = FERR_OK;
			}
			else
			{
				flmAssert(0);
				goto Exit;
			}
		}
	}

Exit:

	if (pUnicode)
	{
		f_free( &pUnicode);
	}

	return( rc);
}
