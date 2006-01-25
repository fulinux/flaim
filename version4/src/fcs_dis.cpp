//-------------------------------------------------------------------------
// Desc:	Data input stream class.
// Tabs:	3
//
//		Copyright (c) 1998-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_dis.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc: 
****************************************************************************/
FCS_DIS::FCS_DIS( void)
{
	m_pIStream = NULL;
	m_uiBOffset = m_uiBDataSize = 0;
	m_bSetupCalled = FALSE;
}

/****************************************************************************
Desc: 
****************************************************************************/
FCS_DIS::~FCS_DIS( void)
{
	if( m_bSetupCalled)
	{
		(void)close();
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::setup( 
	FCS_ISTM *		pIStream)
{
	m_pIStream = pIStream;
	m_bSetupCalled = TRUE;

	return( FERR_OK);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readByte( 
	FLMBYTE *		pValue)
{
	return( read( pValue, 1, NULL));
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readShort( 
	FLMINT16 *		pValue)
{
	FLMUINT16	ui16Value;
	RCODE			rc;
	
	// Read the data.

	if( RC_OK( rc = read( (FLMBYTE *)pValue, 2, NULL)))
	{
		ui16Value = byteToInt( (FLMBYTE *)pValue);
		*pValue = *((FLMINT16 *)&ui16Value);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readUShort( 
	FLMUINT16 *		pValue)
{
	RCODE			rc;
	
	// Read the data.

	if( RC_OK( rc = read( (FLMBYTE *)pValue, 2, NULL)))
	{
		*pValue = byteToInt( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readInt( 
	FLMINT32 *		pValue)
{
	FLMUINT32	ui32Value;
	RCODE			rc;
	
	// Read the data.

	if( RC_OK( rc = read( (FLMBYTE *)pValue, 4, NULL)))
	{
		ui32Value = byteToLong( (FLMBYTE *)pValue);
		*pValue = *((FLMINT32 *)&ui32Value);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readUInt( 
	FLMUINT32 *		pValue)
{
	RCODE			rc;
	
	// Read the data.

	if( RC_OK( rc = read( (FLMBYTE *)pValue, 4, NULL)))
	{
		*pValue = byteToLong( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readInt64( 
	FLMINT64 *		pValue)
{
	FLMUINT64	ui64Value;
	RCODE			rc;
	
	// Read the data.

	if( RC_OK( rc = read( (FLMBYTE *)pValue, 8, NULL)))
	{
		ui64Value = byteToLong64( (FLMBYTE *)pValue);
		*pValue = *((FLMINT64 *)&ui64Value);
	}
	
	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::readUInt64( 
	FLMUINT64 *		pValue)
{
	RCODE			rc;
	
	// Read the data.

	if( RC_OK( rc = read( (FLMBYTE *)pValue, 8, NULL)))
	{
		*pValue = byteToLong64( (FLMBYTE *)pValue);
	}

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE FCS_DIS::skip( 
	FLMUINT			uiBytesToSkip)
{
	return( read( NULL, uiBytesToSkip, NULL));
}

/****************************************************************************
Desc:	Flushes any pending data and closes the DIS
****************************************************************************/
RCODE FCS_DIS::close( void)
{
	RCODE		rc = FERR_OK;

	/*
	Verify that Setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Terminate and flush.
	*/
	
	if( RC_BAD( rc = endMessage()))
	{
		goto Exit;
	}

	/*
	Reset the member variables.
	*/

	m_pIStream = NULL;

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Returns the state of the stream (open == TRUE, closed == FALSE)
****************************************************************************/
FLMBOOL FCS_DIS::isOpen( void)
{
	/*
	Verify that Setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	if( m_pIStream && m_pIStream->isOpen())
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Flushes and terminates the current parent stream message
****************************************************************************/
RCODE FCS_DIS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	/*
	Verify that Setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pIStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	/*
	Flush any pending data.
	*/

	if( RC_BAD( rc = flush()))
	{
		goto Exit;
	}

	/*
	Terminate the message.
	*/

	if( RC_BAD( rc = m_pIStream->endMessage()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Flushes any pending data
****************************************************************************/
RCODE FCS_DIS::flush( void)
{
	RCODE		rc = FERR_OK;

	/*
	Verify that Setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pIStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	
	/*
	Flush the passed-in input stream.
	*/

	if( RC_BAD( rc = m_pIStream->flush()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Reads the specified number of bytes.
****************************************************************************/
RCODE FCS_DIS::read(
	FLMBYTE *	pucData,
	FLMUINT		uiLength,
	FLMUINT *	puiBytesRead)
{
	FLMUINT		uiCopySize;
	FLMUINT		uiReadLen;
	FLMBYTE *	pucPos = NULL;
	RCODE			rc = FERR_OK;

	/*
	Verify that Setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pIStream)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( puiBytesRead)
	{
		*puiBytesRead = uiLength;
	}

	pucPos = pucData;
	while( uiLength)
	{
		if( m_uiBOffset == m_uiBDataSize)
		{
			m_uiBOffset = m_uiBDataSize = 0;

			if( RC_BAD( rc = m_pIStream->read( m_pucBuffer,
				FCS_DIS_BUFFER_SIZE, &uiReadLen)))
			{
				if( uiReadLen)
				{
					rc = FERR_OK;
				}
				else
				{
					goto Exit;
				}
			}
			m_uiBDataSize = uiReadLen;
		}

		uiCopySize = m_uiBDataSize - m_uiBOffset;
		if( uiLength < uiCopySize)
		{
			uiCopySize = uiLength;
		}

		if( pucPos)
		{
#if defined( FLM_NLM) || defined( FLM_WIN)
			if( uiCopySize == 1)
			{
				*pucPos = m_pucBuffer[ m_uiBOffset];
			}
			else if( uiLength == 2)
			{
				*(FLMUINT16 *)pucPos = *((FLMUINT16 *)&m_pucBuffer[ m_uiBOffset]);
			}
			else if( uiLength == 4)
			{
				*(FLMUINT32 *)pucPos = *((FLMUINT32 *)&m_pucBuffer[ m_uiBOffset]);
			}
			else
			{
				f_memcpy( pucPos, &(m_pucBuffer[ m_uiBOffset]), uiCopySize);
			}
#else
				f_memcpy( pucPos, &(m_pucBuffer[ m_uiBOffset]), uiCopySize);
#endif
			pucPos += uiCopySize;
		}
		m_uiBOffset += uiCopySize;
		uiLength -= uiCopySize;
	}
	
Exit:

	if( RC_OK( rc) && uiLength)
	{
		/*
		Unable to satisfy the read request.
		*/

		rc = RC_SET( FERR_EOF_HIT);
	}

	if( puiBytesRead)
	{
		(*puiBytesRead) -= uiLength;
	}

	return( rc);
}


/****************************************************************************
Desc:	Reads a binary token from the stream.  The token is tagged with a
		length.
****************************************************************************/
RCODE FCS_DIS::readBinary(
	POOL *		pPool,
	FLMBYTE **	ppValue,
	FLMUINT *	puiDataSize)
{
	FLMUINT16	ui16DataSize;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = readUShort( &ui16DataSize)))
	{
		goto Exit;
	}

	if( pPool)
	{
		/*
		If the data size is non-zero, allocate a buffer and
		read the entire binary value.
		*/

		if( ui16DataSize)
		{
			if( (*ppValue = (FLMBYTE *)GedPoolAlloc( pPool, ui16DataSize)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( RC_BAD( rc = read( *ppValue, ui16DataSize, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			*ppValue = NULL;
		}
	}
	else
	{
		/*
		The application is not interested in the value.  Just skip the
		to the end of the value.
		*/

		if( RC_BAD( rc = skip( ui16DataSize)))
		{
			goto Exit;
		}
	}

Exit:

	if( puiDataSize)
	{
		*puiDataSize = ui16DataSize;
	}

	return( rc);
}


/****************************************************************************
Desc:	Reads a large binary token from the stream.  The token is tagged with a
		length.
****************************************************************************/
RCODE FCS_DIS::readLargeBinary(
	POOL *		pPool,
	FLMBYTE **	ppValue,
	FLMUINT *	puiDataSize)
{
	FLMUINT32	ui32DataSize;
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = readUInt( &ui32DataSize)))
	{
		goto Exit;
	}

	if( pPool)
	{
		/*
		If the data size is non-zero, allocate a buffer and
		read the entire binary value.
		*/

		if( ui32DataSize)
		{
			if( (*ppValue = (FLMBYTE *)GedPoolAlloc( pPool, ui32DataSize)) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			if( RC_BAD( rc = read( *ppValue, ui32DataSize, NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			*ppValue = NULL;
		}
	}
	else
	{
		/*
		The application is not interested in the value.  Just skip the
		to the end of the value.
		*/

		if( RC_BAD( rc = skip( ui32DataSize)))
		{
			goto Exit;
		}
	}

Exit:

	if( puiDataSize)
	{
		*puiDataSize = (FLMUINT)ui32DataSize;
	}

	return( rc);
}


/****************************************************************************
Desc:	Reads a UTF-8 string from the stream.
****************************************************************************/
RCODE	FCS_DIS::readUTF(
	POOL *			pPool,
	FLMUNICODE **	ppValue)
{
	FLMBYTE		ucByte1;
	FLMBYTE		ucByte2;
	FLMBYTE		ucByte3;
	FLMBYTE		ucLoByte;
	FLMBYTE		ucHiByte;
	FLMUINT16	ui16UTFLen;
	FLMUINT		uiOffset = 0;
	RCODE			rc = FERR_OK;
	
	/*
	Read the data.
	*/

	if( RC_BAD( rc = readUShort( &ui16UTFLen)))
	{
		goto Exit;
	}

	/*
	Check the size of the UTF string.  FLAIM does not support
	strings that are larger than 32K characters.
	*/

	if( ui16UTFLen >= 32767)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	/*
	Allocate space for the string.
	*/

	if( pPool)
	{
		*ppValue = (FLMUNICODE *)GedPoolAlloc( pPool, 
			(FLMUINT)((FLMUINT)sizeof( FLMUNICODE) * (FLMUINT)(ui16UTFLen + 1)));
	}
	else if( ppValue)
	{
		*ppValue = NULL;
	}

	while( ui16UTFLen)
	{
		/*
		Read and decode the bytes.
		*/

		if( RC_BAD( rc = read( &ucByte1, 1, NULL)))
		{
			goto Exit;
		}

		if( (ucByte1 & 0xC0) != 0xC0)
		{
			ucHiByte = 0;
			ucLoByte = ucByte1;
		}
		else
		{
			if( RC_BAD( rc = read( &ucByte2, 1, NULL)))
			{
				goto Exit;
			}

			if( (ucByte1 & 0xE0) == 0xE0)
			{
				if( RC_BAD( rc = read( &ucByte3, 1, NULL)))
				{
					goto Exit;
				}

				ucHiByte =
					(FLMBYTE)(((ucByte1 & 0x0F) << 4) | ((ucByte2 & 0x3C) >> 2));
				ucLoByte = (FLMBYTE)(((ucByte2 & 0x03) << 6) | (ucByte3 & 0x3F));
			}
			else
			{
				ucHiByte = (FLMBYTE)(((ucByte1 & 0x1C) >> 2));
				ucLoByte = (FLMBYTE)(((ucByte1 & 0x03) << 6) | (ucByte2 & 0x3F));
			}
		}

		if( pPool)
		{
			(*ppValue)[ uiOffset] = 
				(FLMUNICODE)(((((FLMUNICODE)(ucHiByte)) << 8) | 
					((FLMUNICODE)(ucLoByte))));
		}

		uiOffset++;
		ui16UTFLen--;
	}

	if( pPool)
	{
		(*ppValue)[ uiOffset] = 0;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: Reads an Hierarchical Tagged Data record from the stream.
****************************************************************************/
RCODE FCS_DIS::readHTD(
	POOL *			pPool,
	FLMUINT			uiContainer,
	FLMUINT			uiDrn,
	NODE **			ppNode,
	FlmRecord **	ppRecord)
{

	FLMBYTE		ucType;
	FLMBYTE		ucLevel = 0;
	FLMBYTE		ucPrevLevel = 0;
	FLMBYTE		ucDescriptor;
	FLMBYTE		ucFlags;
	FLMUINT16	ui16Tag;
	FLMBOOL		bHasValue;
	FLMBOOL		bChild;
	FLMBOOL		bSibling;
	FLMBOOL		bLeftTruncated;
	FLMBOOL		bRightTruncated;
	NODE *		pRoot = NULL;
	NODE *		pNode = NULL;
	NODE *		pPrevNode = NULL;
	void *		pField = NULL;
	void *		pvMark = NULL;
	RCODE			rc = FERR_OK;

	if( pPool)
	{
		pvMark = GedPoolMark( pPool);
	}

	for( ;;)
	{
		/*
		Reset variables.
		*/

		bChild = FALSE;
		bSibling = FALSE;

		/*
		Read the attribute's tag number.
		*/

		if( RC_BAD( rc = readUShort( &ui16Tag)))
		{
			goto Exit;
		}

		/*
		A tag number of 0 indicates that the end of the HTD data
		stream has been reached.
		*/

		if( !ui16Tag)
		{
			break;
		}

		/*
		Read the attribute's descriptor.
		*/

		if( RC_BAD(rc = read( &ucDescriptor, 1, NULL)))
		{
			goto Exit;
		}

		/*
		Set the flag indicating whether or not the
		attribute has a value.
		*/

		bHasValue = (FLMBOOL)((ucDescriptor & HTD_HAS_VALUE_FLAG)
							? (FLMBOOL)TRUE
							: (FLMBOOL)FALSE);

		/*
		Set the value type.
		*/

		ucType = (FLMBYTE)((ucDescriptor & HTD_VALUE_TYPE_MASK));

		/*
		Get the attribute's level.
		*/

		switch( (ucDescriptor & HTD_LEVEL_MASK) >> HTD_LEVEL_POS)
		{
			case HTD_LEVEL_SIBLING:
			{
				bSibling = TRUE;
				ucLevel = ucPrevLevel;
				break;
			}
			
			case HTD_LEVEL_CHILD:
			{
				if( ucLevel < 0xFF)
				{
					bChild = TRUE;
					ucLevel = (FLMBYTE)(ucPrevLevel + 1);
				}
				else
				{
					rc = RC_SET( FERR_BAD_FIELD_LEVEL);
					goto Exit;
				}
				break;
			}
			
			case HTD_LEVEL_BACK:
			{
				if( ucLevel > 0)
				{
					ucLevel = (FLMBYTE)(ucPrevLevel - 1);
				}
				else
				{
					rc = RC_SET( FERR_BAD_FIELD_LEVEL);
					goto Exit;
				}
				break;
			}

			case HTD_LEVEL_BACK_X:
			{
				FLMBYTE ucLevelsBack;

				if( RC_BAD(rc = read( &ucLevelsBack, 1, NULL)))
				{
					goto Exit;
				}

				if( ucPrevLevel >= ucLevelsBack)
				{
					ucLevel = (FLMBYTE)(ucPrevLevel - ucLevelsBack);
				}
				else
				{
					rc = RC_SET( FERR_BAD_FIELD_LEVEL);
					goto Exit;
				}
				break;
			}
		}

		/*
		Allocate the record object
		*/

		if( ppRecord && ucLevel == 0)
		{
			if( *ppRecord)
			{
				if( (*ppRecord)->isReadOnly() || 
					(*ppRecord)->getRefCount() > 1)
				{
					(*ppRecord)->Release();

					if( (*ppRecord = f_new FlmRecord) == NULL)
					{
						rc = RC_SET( FERR_MEM);
						goto Exit;
					}
				}
				else
				{
					// Reuse the existing FlmRecord object.
					(*ppRecord)->clear();
				}
			}
			else
			{
				if( (*ppRecord = f_new FlmRecord) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}
			(*ppRecord)->setContainerID( uiContainer);
			(*ppRecord)->setID( uiDrn);
		}

		/*
		Allocate the attribute.
		*/

		if( pPool && ppNode)
		{
			pNode = GedNodeMake( pPool, ui16Tag, &rc);
			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}

		bLeftTruncated = FALSE;
		bRightTruncated = FALSE;

		/*
		Read the attribute's value.
		*/
		
		switch( ucType)
		{
			case HTD_TYPE_UNICODE:
			{
				FLMUNICODE * pUTF;
	
				if( pNode)
				{
					GedValTypeSet( pNode, FLM_TEXT_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_TEXT_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				/*
				Read UNICODE text in UTF-8 format.
				*/

				if( pPool)
				{
					if( RC_BAD( rc = readUTF( pPool, &pUTF)))
					{
						goto Exit;
					}

					if( pNode)
					{
						if( RC_BAD( rc = GedPutUNICODE( pPool, pNode, pUTF)))
						{
							goto Exit;
						}
					}

					if( ppRecord)
					{
						if( RC_BAD( rc = (*ppRecord)->setUnicode( pField, pUTF)))
						{
							goto Exit;
						}
					}
				}
				else
				{
					if( RC_BAD( rc = readUTF( NULL, NULL)))
					{
						goto Exit;
					}
				}
				break;
			}

			case HTD_TYPE_UINT:
			{
				FLMUINT32		ui32Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_NUMBER_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_NUMBER_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				/*
				Read an unsigned 32-bit integer.
				*/

				if( RC_BAD( rc = readUInt( &ui32Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutUINT( pPool, pNode, ui32Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setUINT( pField, ui32Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_INT:
			{
				FLMINT32		i32Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_NUMBER_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_NUMBER_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				/*
				Read a signed 32-bit integer.
				*/

				if( RC_BAD( rc = readInt( &i32Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutINT( pPool, pNode, i32Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setINT( pField, i32Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_CONTEXT:
			{
				FLMUINT32		ui32Value;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_CONTEXT_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_CONTEXT_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				/*
				Read an unsigned 32-bit integer.
				*/

				if( RC_BAD( rc = readUInt( &ui32Value)))
				{
					goto Exit;
				}

				if( pNode)
				{
					if( RC_BAD( rc = GedPutRecPtr( pPool, pNode, ui32Value)))
					{
						goto Exit;
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->setRecPointer( pField, ui32Value)))
					{
						goto Exit;
					}
				}

				break;
			}

			case HTD_TYPE_BINARY:
			{
				FLMUINT16	ui16DataSize;
				FLMBYTE *	pucData = NULL;

				if( pNode)
				{
					GedValTypeSet( pNode, FLM_BINARY_TYPE);
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, FLM_BINARY_TYPE, &pField)))
					{
						goto Exit;
					}
				}

				if( !bHasValue)
				{
					break;
				}

				/*
				Read a binary data stream.
				*/

				if( RC_BAD( rc = readUShort( &ui16DataSize)))
				{
					goto Exit;
				}

				if( pPool)
				{
					if( pNode)
					{
						if( (pucData = (FLMBYTE *)GedAllocSpace( pPool, pNode,
							FLM_BINARY_TYPE, ui16DataSize)) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
					else if( ppRecord)
					{
						if( RC_BAD(rc = (*ppRecord)->allocStorageSpace( pField,
							FLM_BINARY_TYPE, ui16DataSize, 0, 0, 0, &pucData, NULL)))
						{
							goto Exit;
						}
					}

					if( RC_BAD( rc = read( pucData, ui16DataSize, NULL)))
					{
						goto Exit;
					}

					if( pNode)
					{
						if( ppRecord)
						{
							if( RC_BAD( rc = (*ppRecord)->setBinary( pField, pucData, ui16DataSize)))
							{
								goto Exit;
							}
						}
					}
				}
				else
				{
					if( RC_BAD( rc = skip( ui16DataSize)))
					{
						goto Exit;
					}
				}
				break;
			}

			case HTD_TYPE_DATE:
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}

			case HTD_TYPE_GEDCOM:
			{
				FLMBYTE		ucGedType;
				FLMUINT16	ui16DataSize;
				FLMBYTE *	pucData = NULL;

				/*
				Read the GEDCOM data type and flags
				*/

				if( RC_BAD( rc = read( &ucGedType, 1, NULL)))
				{
					goto Exit;
				}
				ucFlags = ucGedType & 0xF0;
				ucGedType &= 0x0F;

				if( ucFlags & 0x10)
				{
					bLeftTruncated = TRUE;
				}

				if( ucFlags & 0x20)
				{
					bRightTruncated = TRUE;
				}

				if( ucGedType != FLM_TEXT_TYPE &&
					ucGedType != FLM_NUMBER_TYPE &&
					ucGedType != FLM_BINARY_TYPE &&
					ucGedType != FLM_BLOB_TYPE &&
					ucGedType != FLM_CONTEXT_TYPE)
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}

				if( pNode)
				{
					GedValTypeSet( pNode, ucGedType);
					if( bLeftTruncated)
					{
						GedSetLeftTruncated( pNode);
					}

					if( bRightTruncated)
					{
						GedSetRightTruncated( pNode);
					}
				}

				if( ppRecord)
				{
					if( RC_BAD( rc = (*ppRecord)->insertLast( ucLevel, 
						ui16Tag, ucGedType, &pField)))
					{
						goto Exit;
					}

					if( bLeftTruncated)
					{
						(*ppRecord)->setLeftTruncated( pField, TRUE);
					}

					if( bRightTruncated)
					{
						(*ppRecord)->setRightTruncated( pField, TRUE);
					}
				}

				if( !bHasValue)
				{
					break;
				}

				/*
				Read the data size.
				*/

				if( RC_BAD( rc = readUShort( &ui16DataSize)))
				{
					goto Exit;
				}

				/*
				Read the data value.
				*/

				if( pPool)
				{
					if( pNode)
					{
						if( (pucData = (FLMBYTE *)GedAllocSpace( pPool, pNode,
							ucGedType, ui16DataSize)) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
					}
					else if( ppRecord)
					{
						if (RC_BAD( rc = (*ppRecord)->allocStorageSpace( pField,
							ucGedType, ui16DataSize, 0, 0, 0, &pucData, NULL)))
						{
							goto Exit;
						}
					}

					if( RC_BAD( rc = read( pucData, ui16DataSize, NULL)))
					{
						goto Exit;
					}

					if( pNode)
					{
						if( ppRecord)
						{
							if( RC_BAD( rc = (*ppRecord)->setBinary( pField, pucData, ui16DataSize)))
							{
								goto Exit;
							}
						}
					}
				}
				else
				{
					if( RC_BAD( rc = skip( ui16DataSize)))
					{
						goto Exit;
					}
				}
				break;
			}

			default:
			{
				rc = RC_SET( FERR_NOT_IMPLEMENTED);
				goto Exit;
			}
		}

		/*
		Set the truncation flags
		*/

		if( ucType != HTD_TYPE_GEDCOM)
		{
			if( pNode)
			{
				if( bLeftTruncated)
				{
					GedSetLeftTruncated( pNode);
				}

				if( bRightTruncated)
				{
					GedSetRightTruncated( pNode);
				}
			}
			else if( pField)
			{
				if( bLeftTruncated)
				{
					(*ppRecord)->setLeftTruncated( pField, TRUE);
				}

				if( bRightTruncated)
				{
					(*ppRecord)->setRightTruncated( pField, TRUE);
				}
			}
		}
		
		/*
		Graft the attribute into the tree.
		*/
		
		if( pNode)
		{
			if( pRoot == NULL)
			{
				pRoot = pNode;
			}
			else
			{
				if( bSibling)
				{
					pPrevNode->next = pNode;
					pNode->prior = pPrevNode;
					GedNodeLevelSet( pNode, GedNodeLevel( pPrevNode));
				}
				else if( bChild)
				{
					pPrevNode->next = pNode;
					pNode->prior = pPrevNode;
					GedNodeLevelSet( pNode, GedNodeLevel( pPrevNode) + 1);
				}
				else
				{
					pPrevNode->next = pNode;
					pNode->prior = pPrevNode;
					GedNodeLevelSet( pNode, ucLevel);
				}
			}
		}

		ucPrevLevel = ucLevel;
		pPrevNode = pNode;

		/*
		Reset the pool if a GEDCOM record is not
		going to be returned.
		*/

		if( pPool && !ppNode)
		{
			GedPoolReset( pPool, pvMark);
		}
	}

Exit:

	if( RC_OK( rc))
	{
		if( ppNode)
		{
			*ppNode = pRoot;
		}
	}
	else
	{
		if( ppRecord && *ppRecord)
		{
			(*ppRecord)->Release();
		}
	}

	if( pPool && !ppNode)
	{
		GedPoolReset( pPool, pvMark);
	}

	return( rc);		
}
