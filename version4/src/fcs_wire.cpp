//-------------------------------------------------------------------------
// Desc:	Wire class.  Routines to read and parse an entire client request or
//			server response.
// Tabs:	3
//
//		Copyright (c) 1998-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_wire.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
*****************************************************************************/
FCS_WIRE::FCS_WIRE( FCS_DIS * pDIStream, FCS_DOS * pDOStream)
{
	GedPoolInit( &m_pool, 2048);
	m_pPool = &m_pool;
	m_pDIStream = pDIStream;
	m_pDOStream = pDOStream;
	m_pRecord = NULL;
	m_pFromKey = NULL;
	m_pUntilKey = NULL;
	m_bSendGedcom = FALSE;
	(void)resetCommon();
}

/****************************************************************************
Desc:
*****************************************************************************/
FCS_WIRE::~FCS_WIRE( void)
{
	if( m_pRecord)
	{
		m_pRecord->Release();
		m_pRecord = NULL;
	}

	if( m_pFromKey)
	{
		m_pFromKey->Release();
		m_pFromKey = NULL;
	}

	if( m_pUntilKey)
	{
		m_pUntilKey->Release();
		m_pUntilKey = NULL;
	}

	GedPoolFree( &m_pool);
}

/****************************************************************************
Desc:	Resets all member variables to their default / initial values.
*****************************************************************************/
void FCS_WIRE::resetCommon( void)
{
	if( m_pRecord)
	{
		m_pRecord->Release();
		m_pRecord = NULL;
	}

	if( m_pFromKey)
	{
		m_pFromKey->Release();
		m_pFromKey = NULL;
	}

	if( m_pUntilKey)
	{
		m_pUntilKey->Release();
		m_pUntilKey = NULL;
	}

	m_uiClass = 0;
	m_uiOp = 0;
	m_uiRCode = 0;
	m_uiDrn = 0;
	m_uiTransType = FLM_READ_TRANS;
	m_ui64Count = 0;
	m_uiItemId = 0;
	m_uiIndexId = 0;
	m_puzItemName = NULL;
	m_pHTD = NULL;
	m_uiSessionId = FCS_INVALID_ID;
	m_uiSessionCookie = 0;
	m_uiContainer = FLM_DATA_CONTAINER;
	m_uiTransId = FCS_INVALID_ID;
	m_uiIteratorId = FCS_INVALID_ID;
	m_puzFilePath = NULL;
	m_puzFilePath2 = NULL;
	m_puzFilePath3 = NULL;
	m_pucBlock = NULL;
	m_pucSerialNum = NULL;
	m_uiBlockSize = 0;
	m_bIncludesAsync = FALSE;
	fcsInitCreateOpts( &m_CreateOpts);
	GedPoolReset( m_pPool, NULL);
	m_bFlag = FALSE;
	m_ui64Number1 = 0;
	m_ui64Number2 = 0;
	m_ui64Number3 = 0;
	m_uiAddress = 0;
	m_uiFlags = 0;
	m_uiFlaimVersion = 0;
	m_i64SignedValue = 0;
	m_pCSContext = NULL;
	m_pDb = NULL;
}

/****************************************************************************
Desc:	Reads the class and opcode for a client request or server response.
*****************************************************************************/
RCODE FCS_WIRE::readOpcode( void)
{
	FLMBYTE	ucClass;
	FLMBYTE	ucOp;
	RCODE		rc = FERR_OK;

	/*
	Read the opcode.
	*/

	if( RC_BAD( rc = m_pDIStream->read( &ucClass, 1, NULL)))
	{
		goto Exit;
	}
	m_uiClass = ucClass;

	if( RC_BAD( rc = m_pDIStream->read( &ucOp, 1, NULL)))
	{
		goto Exit;
	}
	m_uiOp = ucOp;

Exit:

	return( rc);
}

	
/****************************************************************************
Desc:	Reads a client request or server response and sets the appropriate
		member variable values.
*****************************************************************************/
RCODE FCS_WIRE::readCommon(
	FLMUINT *	puiTagRV,
	FLMBOOL *	pbEndRV)
{
	FLMUINT16	ui16Tmp;
	FLMUINT		uiTag = 0;
	RCODE			rc = FERR_OK;

	/*
	Initialize return variables.
	*/

	*pbEndRV = FALSE;

	/*
	Read the tag.
	*/
	
	if( RC_BAD( rc = m_pDIStream->readUShort( &ui16Tmp)))
	{
		goto Exit;
	}
	uiTag = ui16Tmp;

	/*
	Read the request / response values.
	*/
	
	switch( (uiTag & WIRE_VALUE_TAG_MASK))
	{
		case WIRE_VALUE_RCODE:
		{
			rc = readNumber( uiTag, &m_uiRCode);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SESSION_ID:
		{
			rc = readNumber( uiTag, &m_uiSessionId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SESSION_COOKIE:
		{
			rc = readNumber( uiTag, &m_uiSessionCookie);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_CONTAINER_ID:
		{
			rc = readNumber( uiTag, &m_uiContainer);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_COUNT:
		{
			rc = readNumber( uiTag, NULL, NULL, &m_ui64Count);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_DRN:
		{
			rc = readNumber( uiTag, &m_uiDrn);
			uiTag = 0;
			break;
		}
		
		case WIRE_VALUE_INDEX_ID:
		{
			rc = readNumber( uiTag,	&m_uiIndexId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_HTD:
		{
			rc = m_pDIStream->readHTD( m_pPool, 0, 0, &m_pHTD, NULL);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_RECORD:
		{
			FlmRecord *		pRecord = m_pRecord;
			if( RC_OK( rc = receiveRecord( &pRecord)))
			{
				if( m_pRecord != pRecord)
				{
					if( m_pRecord)
					{
						m_pRecord->Release();
					}
					m_pRecord = pRecord;
				}
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FROM_KEY:
		{
			FlmRecord *		pFromKey = m_pFromKey;
			if( RC_OK( rc = receiveRecord( &pFromKey)))
			{
				if( m_pFromKey != pFromKey)
				{
					if( m_pFromKey)
					{
						m_pFromKey->Release();
					}
					m_pFromKey = pFromKey;
				}
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_UNTIL_KEY:
		{
			FlmRecord *		pUntilKey = m_pUntilKey;
			if( RC_OK( rc = receiveRecord( &pUntilKey)))
			{
				if( m_pUntilKey != pUntilKey)
				{
					if( m_pUntilKey)
					{
						m_pUntilKey->Release();
					}
					m_pUntilKey = pUntilKey;
				}
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_CREATE_OPTS:
		{
			rc = receiveCreateOpts();
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ITERATOR_ID:
		{
			rc = readNumber( uiTag, &m_uiIteratorId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_TRANSACTION_TYPE:
		{
			rc = readNumber( uiTag, &m_uiTransType);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_TRANSACTION_ID:
		{
			rc = readNumber( uiTag, &m_uiTransId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ITEM_NAME:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzItemName);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ITEM_ID:
		{
			rc = readNumber( uiTag, &m_uiItemId);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_BOOLEAN:
		{
			FLMUINT		uiTmp;

			if( RC_OK( rc = readNumber( uiTag, &uiTmp)))
			{
				m_bFlag = (FLMBOOL)((uiTmp) ? (FLMBOOL)TRUE : (FLMBOOL)FALSE);
			}
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_NUMBER1:
		{
			/*
			General-purpose number value
			*/

			rc = readNumber( uiTag, NULL, NULL, &m_ui64Number1);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_NUMBER2:
		{
			/*
			General-purpose number value
			*/

			rc = readNumber( uiTag, NULL, NULL, &m_ui64Number2);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_NUMBER3:
		{
			/*
			General-purpose number value
			*/

			rc = readNumber( uiTag, NULL, NULL, &m_ui64Number3);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_ADDRESS:
		{
			/*
			Block address, etc.
			*/

			rc = readNumber( uiTag, &m_uiAddress);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SIGNED_NUMBER:
		{
			/*
			General-purpose signed number value
			*/

			rc = readNumber( uiTag, NULL, NULL, NULL, &m_i64SignedValue);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FILE_PATH:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzFilePath);
			uiTag = 0;
			break;
		}
				
		case WIRE_VALUE_FILE_PATH_2:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzFilePath2);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FILE_PATH_3:
		{
			rc = m_pDIStream->readUTF( m_pPool, &m_puzFilePath3);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_BLOCK:
		{
			rc = m_pDIStream->readLargeBinary( m_pPool, 
				&m_pucBlock, &m_uiBlockSize);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_SERIAL_NUM:
		{
			FLMUINT	uiSerialLen;

			if( RC_BAD( rc = m_pDIStream->readBinary( m_pPool,
				&m_pucSerialNum, &uiSerialLen)))
			{
				goto Exit;
			}
			
			if( uiSerialLen != F_SERIAL_NUM_SIZE)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			uiTag = 0;
			break;
		}

		case WIRE_VALUE_START_ASYNC:
		{
			/*
			Asynchronous data follows.
			*/

			m_bIncludesAsync = TRUE;
			*pbEndRV = TRUE;
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FLAGS:
		{
			rc = readNumber( uiTag, &m_uiFlags);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_FLAIM_VERSION:
		{
			/*
			FLAIM code version
			*/

			rc = readNumber( uiTag, &m_uiFlaimVersion);
			uiTag = 0;
			break;
		}

		case WIRE_VALUE_TERMINATE:
		{
			/*
			No more parameters.  End the incomming message.
			*/

			rc = m_pDIStream->endMessage();
			*pbEndRV = TRUE;
			uiTag = 0;
			break;
		}

		default:
		{
			break;
		}
	}

Exit:

	*puiTagRV = uiTag;
	return( rc);
}

/****************************************************************************
Desc:	Copies the internal CREATE_OPTS structure into a user-supplied location
*****************************************************************************/
void FCS_WIRE::copyCreateOpts(
	CREATE_OPTS *		pCreateOptsRV)
{
	f_memcpy( pCreateOptsRV, &m_CreateOpts, sizeof( CREATE_OPTS));
}

/****************************************************************************
Desc:	Reads a numeric value from the specified data input stream.
*****************************************************************************/
RCODE FCS_WIRE::readNumber(
	FLMUINT			uiTag,
	FLMUINT *		puiNumber,
	FLMINT *			piNumber,
	FLMUINT64 *		pui64Number,
	FLMINT64 *		pi64Number)
{

	RCODE			rc = FERR_OK;

	/*
	Sanity check.
	*/

	flmAssert( !(puiNumber && piNumber));
	
	/*
	Read the number of bytes specified by the
	value's tag.
	*/

	switch( ((uiTag & WIRE_VALUE_TYPE_MASK) >> 
		WIRE_VALUE_TYPE_START_BIT))
	{
		case WIRE_VALUE_TYPE_GEN_0:
		{
			if( puiNumber)
			{
				*puiNumber = 0;
			}
			else if( piNumber)
			{
				*piNumber = 0;
			}
			else if( pui64Number)
			{
				*pui64Number = 0;
			}
			else if( pi64Number)
			{
				*pi64Number = 0;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_1:
		{
			FLMBYTE	ucValue;

			if( RC_BAD( rc = m_pDIStream->read( &ucValue, 1, NULL)))
			{
				goto Exit;
			}

			if( puiNumber)
			{
				*puiNumber = (FLMUINT)ucValue;
			}
			else if( piNumber)
			{
				*piNumber = (FLMINT)*((FLMINT8 *)&ucValue);
			}
			else if( pui64Number)
			{
				*pui64Number = (FLMUINT64)ucValue;
			}
			else if( pi64Number)
			{
				*pi64Number = (FLMINT64)*((FLMINT8 *)&ucValue);
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_2:
		{
			if( puiNumber || pui64Number)
			{
				FLMUINT16	ui16Value;

				if( RC_BAD( rc = m_pDIStream->readUShort( &ui16Value)))
				{
					goto Exit;
				}

				if( puiNumber)
				{
					*puiNumber = (FLMUINT)ui16Value;
				}
				else if( pui64Number)
				{
					*pui64Number = (FLMUINT64)ui16Value;
				}
			}
			else if( piNumber || pi64Number)
			{
				FLMINT16		i16Value;

				if( RC_BAD( rc = m_pDIStream->readShort( &i16Value)))
				{
					goto Exit;
				}

				if( piNumber)
				{
					*piNumber = (FLMINT)i16Value;
				}
				else if( pi64Number)
				{
					*pi64Number = (FLMINT)i16Value;
				}
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_4:
		{
			if( puiNumber || pui64Number)
			{
				FLMUINT32	ui32Value;

				if( RC_BAD( rc = m_pDIStream->readUInt( &ui32Value)))
				{
					goto Exit;
				}

				if( puiNumber)
				{
					*puiNumber = (FLMUINT)ui32Value;
				}
				else if( pui64Number)
				{
					*pui64Number = (FLMUINT64)ui32Value;
				}
			}
			else if( piNumber || pi64Number)
			{
				FLMINT32		i32Value;

				if( RC_BAD( rc = m_pDIStream->readInt( &i32Value)))
				{
					goto Exit;
				}

				if( piNumber)
				{
					*piNumber = (FLMINT)i32Value;
				}
				else if( pi64Number)
				{
					*pi64Number = (FLMINT64)i32Value;
				}
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_8:
		{
			if( puiNumber || piNumber)
			{
				rc = RC_SET( FERR_CONV_NUM_OVERFLOW);
			}
			else
			{
				if( pui64Number)
				{
					if( RC_BAD( rc = m_pDIStream->readUInt64( pui64Number)))
					{
						goto Exit;
					}
				}
				else if( pi64Number)
				{
					if( RC_BAD( rc = m_pDIStream->readInt64( pi64Number)))
					{
						goto Exit;
					}
				}
				else
				{
					flmAssert( 0);
					rc = RC_SET( FERR_INVALID_PARM);
					goto Exit;
				}
			}
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes an unsigned number to the specified data output stream.	
*****************************************************************************/
RCODE FCS_WIRE::writeUnsignedNumber(
	FLMUINT		uiTag,
	FLMUINT64	ui64Number)
{
	RCODE			rc = FERR_OK;

	if( ui64Number <= (FLMUINT64)0x000000FF)
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_1 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeByte( (FLMBYTE)ui64Number)))
		{
			goto Exit;
		}
	}
	else if( ui64Number <= (FLMUINT64)0x0000FFFF)
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_2 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)ui64Number)))
		{
			goto Exit;
		}
	}
	else if( ui64Number <= (FLMUINT64)0xFFFFFFFF)
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_4 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeUInt32( (FLMUINT32)ui64Number)))
		{
			goto Exit;
		}
	}
	else
	{
		uiTag |= WIRE_VALUE_TYPE_GEN_8 <<
			WIRE_VALUE_TYPE_START_BIT;

		if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDOStream->writeUInt64( ui64Number)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Writes a signed number to the specified data output stream.	
*****************************************************************************/
RCODE FCS_WIRE::writeSignedNumber(
	FLMUINT		uiTag,
	FLMINT64		i64Number)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = writeUnsignedNumber( uiTag, (FLMUINT64)i64Number)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Skips a parameter or return value in the data stream
*****************************************************************************/
RCODE FCS_WIRE::skipValue( 
	FLMUINT		uiTag)
{
	RCODE			rc = FERR_OK;

	switch( ((uiTag & WIRE_VALUE_TYPE_MASK) >> 
		WIRE_VALUE_TYPE_START_BIT))
	{
		case WIRE_VALUE_TYPE_GEN_0:
		{
			break;
		}

		case WIRE_VALUE_TYPE_GEN_1:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 1)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_2:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 2)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_4:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 4)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_GEN_8:
		{
			if( RC_BAD( rc = m_pDIStream->skip( 8)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_BINARY:
		{
			if( RC_BAD( rc = m_pDIStream->readBinary( NULL, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_LARGE_BINARY:
		{
			if( RC_BAD( rc = m_pDIStream->readLargeBinary( NULL, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_HTD:
		{
			if( RC_BAD( rc = m_pDIStream->readHTD( NULL, 0, 0, NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_TYPE_RECORD:
		{
			if( RC_BAD( rc = receiveRecord( NULL)))
			{
				goto Exit;
			}
		}

		case WIRE_VALUE_TYPE_UTF:
		{
			if( RC_BAD( rc = m_pDIStream->readUTF( NULL, NULL)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Sends an opcode to the client
*****************************************************************************/
RCODE FCS_WIRE::sendOpcode(
	FLMUINT					uiClass,
	FLMUINT					uiOp)
{
	FLMBYTE		ucClass = (FLMBYTE)uiClass;
	FLMBYTE		ucOp = (FLMBYTE)uiOp;
	RCODE			rc = FERR_OK;
	
	if( RC_BAD( rc = m_pDOStream->write( &ucClass, 1)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDOStream->write( &ucOp, 1)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendTerminate( void)
{
	RCODE			rc = FERR_OK;
	
	if( RC_BAD( rc = m_pDOStream->writeUShort( 0)))
	{
		goto Exit;
	}

	/*
	Terminate the stream message.
	*/

	if( RC_BAD( rc = m_pDOStream->endMessage()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendNumber(
	FLMUINT			uiTag,
	FLMUINT64		ui64Value,
	FLMINT64			i64Value)
{
	RCODE			rc = FERR_OK;
	
	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_AREA_ID:
		case WIRE_VALUE_OP_SEQ_NUM:
		case WIRE_VALUE_FLAGS:
		case WIRE_VALUE_CLIENT_VERSION:
		case WIRE_VALUE_SESSION_ID:
		case WIRE_VALUE_SESSION_COOKIE:
		case WIRE_VALUE_CONTAINER_ID:
		case WIRE_VALUE_INDEX_ID:
		case WIRE_VALUE_ITEM_ID:
		case WIRE_VALUE_TRANSACTION_ID:
		case WIRE_VALUE_TRANSACTION_TYPE:
		case WIRE_VALUE_DRN:
		case WIRE_VALUE_COUNT:
		case WIRE_VALUE_AUTOTRANS:
		case WIRE_VALUE_MAX_LOCK_WAIT:
		case WIRE_VALUE_RECORD_COUNT:
		case WIRE_VALUE_BOOLEAN:
		case WIRE_VALUE_ITERATOR_ID:
		case WIRE_VALUE_SHARED_DICT_ID:
		case WIRE_VALUE_PARENT_DICT_ID:
		case WIRE_VALUE_TYPE:
		case WIRE_VALUE_NUMBER1:
		case WIRE_VALUE_NUMBER2:
		case WIRE_VALUE_NUMBER3:
		case WIRE_VALUE_ADDRESS:
		case WIRE_VALUE_FLAIM_VERSION:
		{
			if( RC_BAD( rc = writeUnsignedNumber( uiTag, ui64Value)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_SIGNED_NUMBER:
		{
			if( RC_BAD( rc = writeSignedNumber( uiTag, i64Value)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendBinary(
	FLMUINT			uiTag,
	FLMBYTE *		pData,
	FLMUINT			uiLength)
{
	RCODE			rc = FERR_OK;
	
	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_PASSWORD:
		case WIRE_VALUE_SERIAL_NUM:
		{
			uiTag |= WIRE_VALUE_TYPE_BINARY <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeBinary( pData, uiLength)))
			{
				goto Exit;
			}
			break;
		}

		case WIRE_VALUE_BLOCK:
		{
			uiTag |= WIRE_VALUE_TYPE_LARGE_BINARY <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeLargeBinary( pData, uiLength)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a record
*****************************************************************************/
RCODE FCS_WIRE::sendRecord(
	FLMUINT			uiTag,
	FlmRecord *		pRecord)
{
#define RECORD_OUTPUT_BUFFER_SIZE	64
	FLMBYTE		pucBuffer[ RECORD_OUTPUT_BUFFER_SIZE];
	FLMBYTE *	pucBufPos;
	FLMBYTE		ucDescriptor;
	RCODE			rc = FERR_OK;

	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_RECORD:
		case WIRE_VALUE_FROM_KEY:
		case WIRE_VALUE_UNTIL_KEY:
		{
			uiTag |= WIRE_VALUE_TYPE_RECORD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			/*
			The format of a record is (X = 1 bit):

			X				X					XXXXXX			0-64 bytes		HTD
			RESERVED		HTD_FOLLOWS		ID_LENGTH		ID_VALUE			TREE (optional)

			This sequence can repeat, terminating with a 0 byte.
			*/

			ucDescriptor = 0;
			pucBufPos = pucBuffer;
			ucDescriptor |= (FLMBYTE)RECORD_HAS_HTD_FLAG;

			/*
			Output the descriptor.
			*/

			ucDescriptor |= (FLMBYTE)RECORD_ID_SIZE;

			*pucBufPos = ucDescriptor;
			pucBufPos++;

			/*
			Output the ID.  Current format of a record ID is:

				4-byte container ID, 4-byte DRN
			*/

			longToByte( pRecord->getContainerID(), pucBufPos);
			pucBufPos += 4;

			longToByte( pRecord->getID(), pucBufPos);
			pucBufPos += 4;

			/*
			Send the descriptor and record source.
			*/

			if( RC_BAD( rc = m_pDOStream->write( pucBuffer, 
				pucBufPos - pucBuffer)))
			{
				goto Exit;
			}

			/*
			Send the record.
			*/

			if( RC_BAD( rc = m_pDOStream->writeHTD( NULL, pRecord, FALSE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendDrnList(
	FLMUINT			uiTag,
	FLMUINT *		puiList)
{
	FLMUINT		uiItemCount;
	FLMUINT		uiLoop;
	FLMUINT		uiBufSize = 0;
	FLMBYTE *	pucItemBuf = NULL;
	FLMBYTE *	pucItemPos;
	RCODE			rc = FERR_OK;
	
	/*
	If the list pointer is invalid, goto exit.
	*/

	if( !puiList)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_DRN_LIST:
		{
			uiTag |= WIRE_VALUE_TYPE_BINARY <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			/*
			Count the entries in the list.  For now, support only a list of
			2048 elements.
			*/

			for( uiItemCount = 0; uiItemCount < 2048; uiItemCount++)
			{
				if( !puiList[ uiItemCount])
				{
					/*
					End-Of-List.
					*/
					break;
				}
			}

			/*
			Allocate a buffer for the list.
			*/

			uiBufSize = (FLMUINT)(((FLMUINT)sizeof( FLMUINT) * uiItemCount) + (FLMUINT)4);
			if( RC_BAD( rc = f_calloc( uiBufSize, &pucItemBuf)))
			{
				goto Exit;
			}
			pucItemPos = pucItemBuf;

			/*
			Set the item count.
			*/

			UD2FBA( uiItemCount, pucItemPos);
			pucItemPos += 4;

			/*
			Put the items into the buffer.
			*/

			for( uiLoop = 0; uiLoop < uiItemCount; uiLoop++)
			{
				UD2FBA( puiList[ uiLoop], pucItemPos);
				pucItemPos += 4;
			}

			/*
			Send the list.
			*/

			if( RC_BAD( rc = m_pDOStream->writeBinary(
				pucItemBuf, uiBufSize)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	if( pucItemBuf)
	{
		f_free( (void **)&pucItemBuf);
	}

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendString(
	FLMUINT			uiTag,
	FLMUNICODE *	puzString)
{
	RCODE			rc = FERR_OK;
	
	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_FILE_NAME:
		case WIRE_VALUE_FILE_PATH:
		case WIRE_VALUE_FILE_PATH_2:
		case WIRE_VALUE_FILE_PATH_3:
		case WIRE_VALUE_DICT_FILE_PATH:
		case WIRE_VALUE_ITEM_NAME:
		case WIRE_VALUE_DICT_BUFFER:
		{
			uiTag |= WIRE_VALUE_TYPE_UTF <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeUTF( puzString)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendHTD(
	FLMUINT			uiTag,
	NODE *			pHTD)
{
	RCODE			rc = FERR_OK;
	
	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_HTD:
		case WIRE_VALUE_ITERATOR_SELECT:
		case WIRE_VALUE_ITERATOR_FROM:
		case WIRE_VALUE_ITERATOR_WHERE:
		case WIRE_VALUE_ITERATOR_CONFIG:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeHTD( pHTD, NULL, TRUE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendHTD(
	FLMUINT			uiTag,
	FlmRecord *		pRecord)
{
	RCODE			rc = FERR_OK;
	
	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_HTD:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = m_pDOStream->writeHTD( NULL, pRecord, FALSE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Copies the current HTD tree to the application's pool
*****************************************************************************/
RCODE FCS_WIRE::getHTD( 
	POOL *		pPool,
	NODE **		ppTreeRV)
{
	RCODE		rc = FERR_OK;

	if( !m_pHTD)
	{
		*ppTreeRV = NULL;
		goto Exit;
	}

	if( (*ppTreeRV = GedCopy( pPool, GED_FOREST, m_pHTD)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}

	
/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendCreateOpts(
	FLMUINT			uiTag,
	CREATE_OPTS *	pCreateOpts)
{
	NODE *			pRootNd = NULL;
	void *			pvMark = GedPoolMark( m_pPool);
	RCODE				rc = FERR_OK;
	FLMUINT			uiTmp;
	
	/*
	If no create options, goto exit.
	*/

	if( !pCreateOpts)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	/*
	Send the parameter tag and value.
	*/

	switch( uiTag)
	{
		case WIRE_VALUE_CREATE_OPTS:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD << WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}


			/*
			Build the root node of the CreateOpts tree.
			*/

			if( (pRootNd = GedNodeMake( m_pPool, FCS_COPT_CONTEXT, &rc)) == NULL)
			{
				goto Exit;
			}

			/*
			Add fields to the tree.
			*/

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_BLOCK_SIZE, (void *)&pCreateOpts->uiBlockSize,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_MIN_RFL_FILE_SIZE, (void *)&pCreateOpts->uiMinRflFileSize,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_MAX_RFL_FILE_SIZE, (void *)&pCreateOpts->uiMaxRflFileSize,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			uiTmp = pCreateOpts->bKeepRflFiles ? 1 : 0;
			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_KEEP_RFL_FILES, (void *)&uiTmp,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			uiTmp = pCreateOpts->bLogAbortedTransToRfl ? 1 : 0;
			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_LOG_ABORTED_TRANS, (void *)&uiTmp,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_DEFAULT_LANG, (void *)&pCreateOpts->uiDefaultLanguage,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_VERSION, (void *)&pCreateOpts->uiVersionNum,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_APP_MAJOR_VER, (void *)&pCreateOpts->uiAppMajorVer,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = gedAddField( m_pPool, pRootNd,
				FCS_COPT_APP_MINOR_VER, (void *)&pCreateOpts->uiAppMinorVer,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}

			/*
			Send the tree.
			*/

			if( RC_BAD( rc = m_pDOStream->writeHTD( pRootNd, NULL, TRUE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	GedPoolReset( m_pPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc:		Sends a value to the client
*****************************************************************************/
RCODE FCS_WIRE::sendNameTable(
	FLMUINT			uiTag,
	F_NameTable *	pNameTable)
{
	void *			pvMark = GedPoolMark( m_pPool);
	NODE *			pRootNd;
	NODE *			pNd;
	NODE *			pItemIdNd;
	FLMUINT			uiMaxNameChars = 1024;
	FLMUNICODE *	puzItemName = NULL;
	FLMUINT			uiId;
	FLMUINT			uiType;
	FLMUINT			uiSubType;
	FLMUINT			uiNextPos;
	RCODE				rc = FERR_OK;
	
	// If the name table pointer is invalid, goto exit.

	if( !pNameTable)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Allocate a temporary name buffer

	if( (puzItemName = (FLMUNICODE *)GedPoolAlloc( m_pPool, 
		uiMaxNameChars * sizeof( FLMUNICODE))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Send the parameter tag and value.

	switch( uiTag)
	{
		case WIRE_VALUE_NAME_TABLE:
		{
			uiTag |= WIRE_VALUE_TYPE_HTD <<
				WIRE_VALUE_TYPE_START_BIT;

			if( RC_BAD( rc = m_pDOStream->writeUShort( (FLMUINT16)uiTag)))
			{
				goto Exit;
			}


			// Build the root node of the name table tree.

			if( (pRootNd = GedNodeMake( m_pPool, 
				FCS_NAME_TABLE_CONTEXT, &rc)) == NULL)
			{
				goto Exit;
			}
				
			uiNextPos = 0;
			while( pNameTable->getNextTagNumOrder( &uiNextPos, puzItemName, 
				NULL, uiMaxNameChars * sizeof( FLMUNICODE), 
				&uiId, &uiType, &uiSubType))
			{
				if( (pItemIdNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_ID, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUINT( m_pPool, pItemIdNd, uiId)))
				{
					goto Exit;
				}

				if( (pNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_NAME, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUNICODE( m_pPool, pNd, puzItemName)))
				{
					goto Exit;
				}

				GedChildGraft( pItemIdNd, pNd, GED_LAST);

				if( (pNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_TYPE, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUINT( m_pPool, pNd, uiType)))
				{
					goto Exit;
				}

				GedChildGraft( pItemIdNd, pNd, GED_LAST);

				if( (pNd = GedNodeMake( m_pPool, 
					FCS_NAME_TABLE_ITEM_SUBTYPE, &rc)) == NULL)
				{
					goto Exit;
				}

				if( RC_BAD( rc = GedPutUINT( m_pPool, pNd, uiSubType)))
				{
					goto Exit;
				}

				GedChildGraft( pItemIdNd, pNd, GED_LAST);

				// Graft the item into the tree

				GedChildGraft( pRootNd, pItemIdNd, GED_LAST);

				// Release CPU to prevent CPU hog

				f_yieldCPU();
			}

			// Send the tree.

			if( RC_BAD( rc = m_pDOStream->writeHTD( pRootNd, 
				NULL, TRUE, m_bSendGedcom)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
#ifdef FLM_DEBUG
			flmAssert( 0);
#else
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
#endif
		}
	}

Exit:

	GedPoolReset( m_pPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Receives a record
*****************************************************************************/
RCODE FCS_WIRE::receiveRecord(
	FlmRecord **	ppRecord)
{
	FLMBYTE					ucDescriptor = 0;
	FLMUINT					uiIdLen = 0;
	FLMUINT32				ui32Container;
	FLMUINT32				ui32Drn;
	void *					pvMark = GedPoolMark( m_pPool);
	FLMBOOL					bHasId = FALSE;
	RCODE						rc = FERR_OK;

	/*
	Read the record.
	*/

	if( RC_BAD( rc = m_pDIStream->read( &ucDescriptor, 1, NULL)))
	{
		goto Exit;
	}

	uiIdLen = (FLMUINT)(ucDescriptor & RECORD_ID_SIZE_MASK);

	if( uiIdLen != RECORD_ID_SIZE)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	else if( uiIdLen)
	{
		bHasId = TRUE;
	}

	/*
	Read the record ID.
	*/

	if( bHasId)
	{
		if( RC_BAD( rc = m_pDIStream->readUInt( &ui32Container)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = m_pDIStream->readUInt( &ui32Drn)))
		{
			goto Exit;
		}
	}

	/*
	Read the record.
	*/

	if( (ucDescriptor & RECORD_HAS_HTD_FLAG))
	{
		if( RC_BAD( rc = m_pDIStream->readHTD( m_pPool,
			ui32Container, ui32Drn, NULL, ppRecord)))
		{
			goto Exit;
		}
	}

Exit:

	if( RC_BAD( rc) && ppRecord && *ppRecord)
	{
		(*ppRecord)->Release();
		*ppRecord = NULL;
	}

	GedPoolReset( m_pPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc:	Receives a CREATE_OPTS structure as an HTD tree.
*****************************************************************************/
RCODE FCS_WIRE::receiveCreateOpts( void)
{
	NODE *		pRootNd;
	NODE *		pTmpNd;
	void *		pPoolMark;
	FLMUINT		fieldPath[ 8];
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	pPoolMark = GedPoolMark( m_pPool);
  
	/*
	Initialize the CREATE_OPTS structure to its default values.
	*/

	fcsInitCreateOpts( &m_CreateOpts);

	/*
	Receive the tree.
	*/

	if( RC_BAD( rc = m_pDIStream->readHTD( m_pPool,
		0, 0, &pRootNd, NULL)))
	{
		goto Exit;
	}

	/*
	Parse the tree and extract the values.
	*/

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_BLOCK_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiBlockSize);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_MIN_RFL_FILE_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiMinRflFileSize);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_MAX_RFL_FILE_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiMaxRflFileSize);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_KEEP_RFL_FILES;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		m_CreateOpts.bKeepRflFiles = (FLMBOOL)((uiTmp)
															? (FLMBOOL)TRUE
															: (FLMBOOL)FALSE);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_LOG_ABORTED_TRANS;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		m_CreateOpts.bLogAbortedTransToRfl = (FLMBOOL)((uiTmp)
																	  ? (FLMBOOL)TRUE
																	  : (FLMBOOL)FALSE);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_DEFAULT_LANG;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiDefaultLanguage);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_VERSION;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiVersionNum);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_APP_MAJOR_VER;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiAppMajorVer);
	}

	fieldPath[ 0] = FCS_COPT_CONTEXT;
	fieldPath[ 1] = FCS_COPT_APP_MINOR_VER;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pRootNd, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &m_CreateOpts.uiAppMinorVer);
	}

Exit:

	GedPoolReset( m_pPool, pPoolMark);
	return( rc);
}

/****************************************************************************
Desc:	Receives a name table.
*****************************************************************************/
RCODE FCS_WIRE::receiveNameTable(
	F_NameTable **		ppNameTable)
{
	NODE *			pRootNd;
	NODE *			pItemIdNd;
	NODE *			pNd = NULL;
	void *			pvMark = GedPoolMark( m_pPool);
	FLMUINT			uiMaxNameChars = 1024;
	FLMUNICODE *	puzItemName;
	FLMUINT			uiItemId;
	FLMUINT			uiItemType;
	FLMUINT			uiItemSubType;
	F_NameTable *	pNameTable = NULL;
	FLMBOOL			bCreatedTable = FALSE;
	RCODE				rc = FERR_OK;

	// Allocate a temporary name buffer

	if( (puzItemName = (FLMUNICODE *)GedPoolAlloc( m_pPool, 
		uiMaxNameChars * sizeof( FLMUNICODE))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Initialize the name table.

	if( (pNameTable = *ppNameTable) == NULL)
	{
		if( (pNameTable = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		bCreatedTable = TRUE;
	}
	else
	{
		pNameTable->clearTable();
	}

	// Receive the tree.

	if( RC_BAD( rc = m_pDIStream->readHTD( m_pPool,
		0, 0, &pRootNd, NULL)))
	{
		goto Exit;
	}

	// Parse the tree and extract the values.

	pItemIdNd = GedChild( pRootNd);
	while( pItemIdNd)
	{
		if( GedTagNum( pItemIdNd) == FCS_NAME_TABLE_ITEM_ID)
		{
			if( RC_BAD( rc = GedGetUINT( pItemIdNd, &uiItemId)))
			{
				goto Exit;
			}

			uiItemType = 0;
			uiItemSubType = 0;
			pNd = GedChild( pItemIdNd);
			while( pNd)
			{
				switch( GedTagNum( pNd))
				{
					case FCS_NAME_TABLE_ITEM_NAME:
					{
						FLMUINT		uiStrLen = uiMaxNameChars * sizeof( FLMUNICODE);

						if( RC_BAD( rc = GedGetUNICODE( pNd, puzItemName,
							&uiStrLen)))
						{
							goto Exit;
						}

						break;
					}

					case FCS_NAME_TABLE_ITEM_TYPE:
					{
						if( RC_BAD( rc = GedGetUINT( pNd, &uiItemType)))
						{
							goto Exit;
						}

						break;
					}

					case FCS_NAME_TABLE_ITEM_SUBTYPE:
					{
						if( RC_BAD( rc = GedGetUINT( pNd, &uiItemSubType)))
						{
							goto Exit;
						}

						break;
					}
				}

				pNd = GedSibNext( pNd);
			}

			if( puzItemName[ 0])
			{
				if( RC_BAD( rc = pNameTable->addTag( puzItemName, NULL, 
					uiItemId, uiItemType, uiItemSubType, FALSE)))
				{
					goto Exit;
				}
			}
		}

		pItemIdNd = GedSibNext( pItemIdNd);

		// Release CPU to prevent CPU hog

		f_yieldCPU();
	}

	pNameTable->sortTags();
	*ppNameTable = pNameTable;
	pNameTable = NULL;

Exit:

	if( pNameTable && bCreatedTable)
	{
		pNameTable->Release();
	}

	GedPoolReset( m_pPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FCL_WIRE::FCL_WIRE( CS_CONTEXT_p pCSContext, FDB_p pDb) :
	FCS_WIRE( pCSContext != NULL ? pCSContext->pIDataStream : NULL,
			  pCSContext != NULL ? pCSContext->pODataStream : NULL)
{
	m_pCSContext = pCSContext;
	m_pDb = pDb;

	if( m_pCSContext)
	{
		m_bSendGedcom = m_pCSContext->bGedcomSupport;
	}
}

/****************************************************************************
Desc:	Sets the CS CONTEXT in FCL_WIRE and the I/O streams in FCS_WIRE
*****************************************************************************/
void FCL_WIRE::setContext(
	CS_CONTEXT_p		pCSContext)
{
	m_pCSContext = pCSContext;
	m_bSendGedcom = pCSContext->bGedcomSupport;
	FCS_WIRE::setDIStream( pCSContext->pIDataStream);
	FCS_WIRE::setDOStream( pCSContext->pODataStream);
}

/****************************************************************************
Desc:	Send a client/server opcode with session id, and optionally the
		database id
****************************************************************************/
RCODE FCL_WIRE::sendOp(
	FLMUINT			uiClass,
	FLMUINT			uiOp)
{
	RCODE				rc = FERR_OK;

	if (!m_pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Exit;
	}

	/* Send the class and opcode. */

	if (RC_BAD( rc = sendOpcode( (FLMBYTE)uiClass, (FLMBYTE)uiOp)))
	{
		goto Transmission_Error;
	}

	/* Send session ID. */

	if (RC_BAD( rc = sendNumber(
		WIRE_VALUE_SESSION_ID, m_pCSContext->uiSessionId)))
	{
		goto Transmission_Error;
	}

	/* Send session cookie. */

	if (RC_BAD( rc = sendNumber(
		WIRE_VALUE_SESSION_COOKIE, m_pCSContext->uiSessionCookie)))
	{
		goto Transmission_Error;
	}

	/* Send operation sequence number. */

	m_pCSContext->uiOpSeqNum++;
	if (RC_BAD( rc = sendNumber( 
		WIRE_VALUE_OP_SEQ_NUM, m_pCSContext->uiOpSeqNum)))
	{
		goto Transmission_Error;
	}

Exit:

	return( rc);

Transmission_Error:
	m_pCSContext->bConnectionGood = FALSE;
	goto Exit;
}


/****************************************************************************
Desc:	This routine instructs the server to start or end a transaction
****************************************************************************/
RCODE FCL_WIRE::doTransOp(
	FLMUINT			uiOp,
	FLMUINT			uiTransType,
	FLMUINT			uiFlags,
	FLMUINT			uiMaxLockWait,
	FLMBYTE *		pszHeader,
	FLMBOOL			bForceCheckpoint)
{
	FLMUINT			uiTransFlags = 0;
	RCODE				rc = FERR_OK;

	/* Send request to server. */

	if( RC_BAD( rc = sendOp( FCS_OPCLASS_TRANS, uiOp)))
	{
		goto Exit;
	}

	if( uiOp == FCS_OP_TRANSACTION_BEGIN)
	{
		if (RC_BAD( rc = sendNumber(
			WIRE_VALUE_TRANSACTION_TYPE, uiTransType)))
		{
			goto Transmission_Error;
		}

		if (RC_BAD( rc = sendNumber(
			WIRE_VALUE_MAX_LOCK_WAIT, uiMaxLockWait)))
		{
			goto Transmission_Error;
		}

		if( pszHeader)
		{
			uiTransFlags |= FCS_TRANS_FLAG_GET_HEADER;
		}

		if( uiFlags & FLM_DONT_KILL_TRANS)
		{
			uiTransFlags |= FCS_TRANS_FLAG_DONT_KILL;
		}

		if( uiFlags & FLM_DONT_POISON_CACHE)
		{
			uiTransFlags |= FCS_TRANS_FLAG_DONT_POISON;
		}
	}
	else if( uiOp == FCS_OP_TRANSACTION_COMMIT_EX)
	{
		if( pszHeader)
		{
			if( RC_BAD( rc = sendBinary(
				WIRE_VALUE_BLOCK, pszHeader, F_TRANS_HEADER_SIZE)))
			{
				goto Exit;
			}
		}

		if( bForceCheckpoint)
		{
			uiTransFlags |= FCS_TRANS_FORCE_CHECKPOINT;
		}
	}

	if( uiTransFlags)
	{
		if (RC_BAD( rc = sendNumber(
			WIRE_VALUE_FLAGS, uiTransFlags)))
		{
			goto Transmission_Error;
		}
	}

	if( RC_BAD( rc = sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */

	if( RC_BAD( rc = read()))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = getRCode()))
	{
		goto Exit;
	}

	if( pszHeader)
	{
		if( getBlockSize())
		{
			f_memcpy( pszHeader, getBlock(), getBlockSize());
		}
		else
		{
			f_memset( pszHeader, 0, 2048);
		}
	}

	if (!m_pDb)
	{
		m_pCSContext->bTransActive = (FLMBOOL)((uiOp == FCS_OP_TRANSACTION_BEGIN)
													  ? (FLMBOOL)TRUE
													  : (FLMBOOL)FALSE);
	}

Exit:

	return( rc);
Transmission_Error:
	m_pCSContext->bConnectionGood = FALSE;
	goto Exit;
}

/****************************************************************************
Desc:	Reads a server response for the client.
*****************************************************************************/
RCODE	FCL_WIRE::read( void)
{
	FLMUINT	uiTag;
	FLMUINT	uiCount = 0;
	FLMBOOL	bDone = FALSE;
	RCODE		rc = FERR_OK;

	/*
	Read the opcode.
	*/

	if( RC_BAD( rc = readOpcode()))
	{
		goto Exit;
	}
	
	/*
	Read the request / response values.
	*/
	
	for( ;;)
	{
		if (RC_BAD( rc = readCommon( &uiTag, &bDone)))
		{
			if( rc == FERR_EOF_HIT && !uiCount)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}

		if( bDone)
		{
			goto Exit;
		}

		/*
		uiTag will be non-zero if readCommon did not understand it.
		*/

		uiCount++;
		if( uiTag)
		{
			switch( (uiTag & WIRE_VALUE_TAG_MASK))
			{
				case WIRE_VALUE_NAME_TABLE:
				{
					if( RC_BAD( rc = receiveNameTable( &m_pNameTable)))
					{
						goto Exit;
					}
					break;
				}

				default:
				{
					if( RC_BAD( rc = skipValue( uiTag)))
					{
						goto Exit;
					}
					break;
				}
			}
		}
	}

Exit:

	if( rc == FERR_EOF_HIT)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	return( rc);
}
