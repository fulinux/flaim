//-------------------------------------------------------------------------
// Desc:	Data output stream class.
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_dos.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:
****************************************************************************/
FCS_DOS::FCS_DOS( void)
{
	m_pOStream = NULL;
	m_uiBOffset = 0;
	GedPoolInit( &m_tmpPool, 512);
	m_bSetupCalled = FALSE;
}


/****************************************************************************
Desc:
****************************************************************************/
FCS_DOS::~FCS_DOS( void)
{
	if( m_bSetupCalled)
	{
		(void)close();
	}
	GedPoolFree( &m_tmpPool);
}

/****************************************************************************
Desc:	Writes a specified number of bytes from a buffer to the output
		stream.
****************************************************************************/
RCODE FCS_DOS::write(
	FLMBYTE *		pucData,
	FLMUINT			uiLength)
{
	RCODE		rc = FERR_OK;

	/*
	Verify that setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Write the data.
	*/

Retry_Write:

	if( FCS_DOS_BUFFER_SIZE - m_uiBOffset >= uiLength)
	{
#if defined( FLM_NLM) || defined( FLM_WIN)
		if( uiLength == 1)
		{
			m_pucBuffer[ m_uiBOffset] = *pucData;
			m_uiBOffset++;
		}
		else if( uiLength == 2)
		{
			*(FLMUINT16 *)&(m_pucBuffer[ m_uiBOffset]) = *((FLMUINT16 *)pucData);
			m_uiBOffset += 2;
		}
		else if( uiLength == 4)
		{
			*(FLMUINT32 *)&(m_pucBuffer[ m_uiBOffset]) = *((FLMUINT32 *)pucData);
			m_uiBOffset += 4;
		}
		else
		{
			f_memcpy( &(m_pucBuffer[ m_uiBOffset]), pucData, uiLength);
			m_uiBOffset += uiLength;
		}
#else
		f_memcpy( &(m_pucBuffer[ m_uiBOffset]), pucData, uiLength);
		m_uiBOffset += uiLength;
#endif
	}
	else
	{
		if( m_uiBOffset > 0)
		{
			if( RC_BAD( rc = flush()))
			{
				goto Exit;
			}
		}

		if( uiLength <= FCS_DOS_BUFFER_SIZE)
		{
			goto Retry_Write;
		}

		if( RC_BAD( rc = m_pOStream->write( pucData, uiLength)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Writes a UNICODE string to the stream in the UTF-8 format.
****************************************************************************/
RCODE	FCS_DOS::writeUTF(
	FLMUNICODE *	puzValue)
{
	FLMUINT			uiUTFLen;
	FLMUNICODE *	puzTmp;
	RCODE				rc = FERR_OK;
	
	/*
	Verify that setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Verify pValue is valid.
	*/
	
	flmAssert( puzValue != NULL);

	/*
	Determine the size of the string.
	*/
	
	uiUTFLen = 0;
	puzTmp = puzValue;
	while( *puzTmp)
	{
		uiUTFLen++;
		puzTmp++;
	}

	if( RC_BAD( rc = writeUShort( (FLMUINT16)uiUTFLen)))
	{
		goto Exit;
	}

	puzTmp = puzValue;
	while( *puzTmp)
	{
		if( *puzTmp <= 0x007F)
		{
			if( RC_BAD( rc = writeByte( (FLMBYTE)(*puzTmp))))
			{
				goto Exit;
			}
		}
		else if( *puzTmp >= 0x0080 && *puzTmp <= 0x07FF)
		{
			if( RC_BAD( rc = writeUShort((FLMUINT16)
				((((FLMUINT16)(0xC0 | (FLMBYTE)((*puzTmp & 0x07C0) >> 6))) << 8) |
				(FLMUINT16)(0x80 | (FLMBYTE)(*puzTmp & 0x003F))))))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = writeUShort((FLMUINT16)
				((((FLMUINT16)(0xE0 | (FLMBYTE)((*puzTmp & 0xF000) >> 12))) << 8) |
				(FLMUINT16)(0x80 | (FLMBYTE)((*puzTmp & 0x0FC0) >> 6))))))
			{
				goto Exit;
			}

			if( RC_BAD( rc = writeByte( (0x80 | (FLMBYTE)(*puzTmp & 0x003F)))))
			{
				goto Exit;
			}
		}

		puzTmp++;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Writes a binary token (including length) to the stream.
****************************************************************************/
RCODE FCS_DOS::writeBinary(
	FLMBYTE *	pucValue,
	FLMUINT		uiSize)
{
	RCODE			rc = FERR_OK;

	flmAssert( uiSize <= 0x0000FFFF);

	if( RC_BAD( rc = writeUShort( (FLMUINT16)uiSize)))
	{
		goto Exit;
	}

	if( uiSize)
	{
		if( RC_BAD( rc = write( pucValue, uiSize)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Writes a large binary token (including length) to the stream.
****************************************************************************/
RCODE FCS_DOS::writeLargeBinary(
	FLMBYTE *	pucValue,
	FLMUINT		uiSize)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = writeUInt32( (FLMUINT32)uiSize)))
	{
		goto Exit;
	}

	if( uiSize)
	{
		if( RC_BAD( rc = write( pucValue, uiSize)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Writes a Hierarchical Tagged Data record to the stream.
****************************************************************************/
RCODE FCS_DOS::writeHTD(
	NODE *		pHTD,
	FlmRecord *	pRecord,
	FLMBOOL		bSendForest,
	FLMBOOL		bSendAsGedcom)
{
	FLMUINT		uiPrevLevel = 0;
	FLMUINT		uiLevelsBack = 0;
	FLMUINT		uiDescriptor = 0;
	FLMUINT		uiCurLevel = 0;
	FLMUINT		uiCurValType = 0;
	FLMUINT		uiCurDataLen = 0;
	FLMBOOL		bLeftTruncated;
	FLMBOOL		bRightTruncated;
	FLMBYTE *	pucCurData = NULL;
	FLMBYTE		pucTmpBuf[ 32];
	void *		pvMark = GedPoolMark( &m_tmpPool);
	NODE *		pCurNode = NULL;
	void *		pCurField = NULL;
	RCODE			rc = FERR_OK;

	/*
	Verify that setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Set the current node or field
	*/

	if( pHTD)
	{
		pCurNode = pHTD;
	}
	else
	{
		pCurField = pRecord->root();
	}

	while( pCurNode || pCurField)
	{
		/*
		See if we are done sending the tree/forest.
		*/

		if( pCurNode)
		{
			if( !bSendForest && (pCurNode != pHTD) &&
				(GedNodeLevel( pCurNode) == GedNodeLevel( pHTD)))
			{
				break;
			}
		}

		/*
		Output the attribute's tag number.
		*/

		if( pCurNode)
		{
			flmUINT16ToBigEndian( (FLMUINT16)GedTagNum( pCurNode), pucTmpBuf);
		}
		else if( pCurField)
		{
			flmUINT16ToBigEndian( (FLMUINT16)pRecord->getFieldID( pCurField), pucTmpBuf);
		}

		if( RC_BAD( rc = write( pucTmpBuf, 2)))
		{
			goto Exit;
		}

		/*
		Setup the attribute's descriptor.
		*/

		uiDescriptor = 0;
		uiLevelsBack = 0;

		if( pCurNode)
		{
			uiCurLevel = GedNodeLevel( pCurNode);
		}
		else
		{
			uiCurLevel = pRecord->getLevel( pCurField);
		}

		if( uiCurLevel == uiPrevLevel)
		{
			(void)(uiDescriptor |= (HTD_LEVEL_SIBLING << HTD_LEVEL_POS));
		}
		else if( uiCurLevel == uiPrevLevel + 1)
		{
			uiDescriptor |= (HTD_LEVEL_CHILD << HTD_LEVEL_POS);
		}
		else if( uiCurLevel == uiPrevLevel - 1)
		{
			uiDescriptor |= (HTD_LEVEL_BACK << HTD_LEVEL_POS);
		}
		else if( uiCurLevel < uiPrevLevel)
		{
			uiDescriptor |= (HTD_LEVEL_BACK_X << HTD_LEVEL_POS);
			uiLevelsBack = uiPrevLevel - uiCurLevel;
		}
		else
		{
			flmAssert( 0);
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		if( pCurNode)
		{
			uiCurDataLen = GedValLen( pCurNode);
			uiCurValType = GedValType( pCurNode) & 0x0F;
			bLeftTruncated = GedIsLeftTruncated( pCurNode);
			bRightTruncated = GedIsRightTruncated( pCurNode);
			pucCurData = (FLMBYTE *)GedValPtr( pCurNode);
		}
		else
		{
			uiCurDataLen = pRecord->getDataLength( pCurField);
			uiCurValType = (FLMUINT)pRecord->getDataType( pCurField);
			bLeftTruncated = pRecord->isLeftTruncated( pCurField);
			bRightTruncated = pRecord->isRightTruncated( pCurField);
			pucCurData = (FLMBYTE *)(pRecord->getDataPtr( pCurField));
		}

		if( uiCurDataLen)
		{
			uiDescriptor |= HTD_HAS_VALUE_FLAG;
		}

		if( bSendAsGedcom)
		{
			uiDescriptor |= HTD_TYPE_GEDCOM;
		}
		else
		{
			switch( uiCurValType)
			{
				case FLM_TEXT_TYPE:
				{
					uiDescriptor |= HTD_TYPE_UNICODE;
					break;
				}

				case FLM_NUMBER_TYPE:
				{
					/*
					To save conversion time, cheat to determine if
					the number is negative.
					*/

					if( ((*pucCurData & 0xF0) == 0xB0))
					{
						uiDescriptor |= HTD_TYPE_INT;
					}
					else
					{
						uiDescriptor |= HTD_TYPE_UINT;
					}
					break;
				}

				case FLM_CONTEXT_TYPE:
				{
					uiDescriptor |= HTD_TYPE_CONTEXT;
					break;
				}

				case FLM_BINARY_TYPE:
				{
					uiDescriptor |= HTD_TYPE_BINARY;
					break;
				}

				default:
				{
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}
			}
		}

		/*
		Output the attribute's descriptor.
		*/

		pucTmpBuf[ 0] = (FLMBYTE)uiDescriptor;
		if( RC_BAD( rc = write( pucTmpBuf, 1)))
		{
			goto Exit;
		}

		/*
		Output the "levels back" value (if available).
		*/

		if( uiLevelsBack)
		{
			flmAssert( uiLevelsBack <= 0xFF);
			pucTmpBuf[ 0] = (FLMBYTE)uiLevelsBack;
			if( RC_BAD( rc = write( pucTmpBuf, 1)))
			{
				goto Exit;
			}
		}

		/*
		Output the attribute's value.
		*/

		if( bSendAsGedcom)
		{
			/*
			Output the GEDCOM data type and flags
			*/

			pucTmpBuf[ 0] = (FLMBYTE)uiCurValType;
			if( bLeftTruncated)
			{
				pucTmpBuf[ 0] |= 0x10;
			}

			if( bRightTruncated)
			{
				pucTmpBuf[ 0] |= 0x20;
			}

			if( RC_BAD( rc = write( pucTmpBuf, 1)))
			{
				goto Exit;
			}

			if( uiCurDataLen)
			{
				/*
				Output the data size.
				*/
				flmAssert( uiCurDataLen <= 0x0000FFFF);

				flmUINT16ToBigEndian( (FLMUINT16)uiCurDataLen, pucTmpBuf);
				if( RC_BAD( rc = write( pucTmpBuf, 2)))
				{
					goto Exit;
				}

				/*
				Send the data.
				*/

				if( RC_BAD( rc = write( pucCurData, uiCurDataLen)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			/*
			Send the value.
			*/

			switch( uiCurValType)
			{
				case FLM_TEXT_TYPE:
				{
					/*
					Extract the value.
					*/

					if( uiCurDataLen)
					{
						FLMUINT			uiBufSize;
						FLMUNICODE *	puzValue;

						/*
						Reset the temporary pool.
						*/

						GedPoolReset( &m_tmpPool, pvMark);
						if( uiCurDataLen <= 32751)
						{
							/*
							Allocate a buffer that is twice the size of the
							attribute's value length.  This is necessary because the
							UNICODE conversion will may double the size of the
							attribute's value.  A "safety" zone of 32 bytes is added
							to the buffer size to allow for strings that may require
							more than 2x the attribute's size and to account for
							null-termination bytes.
							*/

							uiBufSize = (2 * uiCurDataLen) + 32;
						}
						else
						{
							/*
							Allocate a full 64K.
							*/

							uiBufSize = 65535;
						}
		
						if( (puzValue = (FLMUNICODE *)GedPoolAlloc( &m_tmpPool,
							uiBufSize)) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}

						/*
						Extract UNICODE from the attribute.
						*/
						
						if( (pCurNode && RC_BAD( rc = GedGetUNICODE( pCurNode, puzValue, &uiBufSize))) ||
							(pCurField && RC_BAD( rc = pRecord->getUnicode( pCurField, puzValue, &uiBufSize))))
						{
							if( rc == FERR_CONV_DEST_OVERFLOW)
							{
								/*
								Since we did not correctly guess the buffer size,
								try again.  This time, take the slow (but safe)
								approach of calculating the size of the UNICODE string.
								*/

								if( (pCurNode && RC_BAD( rc = GedGetUNICODE( pCurNode, NULL, &uiBufSize))) ||
									(pCurField && RC_BAD( rc = pRecord->getUnicodeLength( pCurField, &uiBufSize))))
								{
									goto Exit;
								}

								/*
								Add two bytes to account for null-termination.
								*/

								uiBufSize += 2;

								/*
								Reset the pool to clear the prior allocation.
								*/

								GedPoolReset( &m_tmpPool, pvMark);

								/*
								Allocate the new buffer.
								*/
								
								if( (puzValue = (FLMUNICODE *)GedPoolAlloc( &m_tmpPool,
									uiBufSize)) == NULL)
								{
									rc = RC_SET( FERR_MEM);
									goto Exit;
								}

								/*
								Extract the UNICODE string.
								*/
								
								if( (pCurNode && RC_BAD( rc = GedGetUNICODE( 
											pCurNode, puzValue, &uiBufSize))) ||
									 (pCurField && RC_BAD( rc = pRecord->getUnicode( 
									 		pCurField, puzValue, &uiBufSize))))
								{
									goto Exit;
								}
							}
							else
							{
								goto Exit;
							}
						}

						/*
						Write the attribute's value.
						*/

						if( RC_BAD( rc = writeUTF( puzValue)))
						{
							goto Exit;
						}
					}
						
					break;
				}

				case FLM_NUMBER_TYPE:
				{
					if( uiCurDataLen)
					{	
						if( uiDescriptor & HTD_TYPE_INT)
						{
							/*
							Since the number is negative, extract and send it
							as a signed 32-bit value.
							*/

							FLMINT		iValue;

							if( (pCurNode && RC_BAD( rc = GedGetINT( pCurNode, &iValue))) ||
								(pCurField && RC_BAD( rc = pRecord->getINT( pCurField, &iValue))))
							{
								goto Exit;
							}

							/*
							Write the value.
							*/

							if( RC_BAD( rc = writeInt32( (FLMINT32)iValue)))
							{
								goto Exit;
							}
						}
						else
						{
							/*
							The number is non-negative
							*/

							FLMUINT		uiValue;

							if( (pCurNode && RC_BAD( rc = GedGetUINT( pCurNode, &uiValue))) ||
								(pCurField && RC_BAD( rc = pRecord->getUINT( pCurField, &uiValue))))
							{
								goto Exit;
							}

							/*
							Write the value.
							*/

							if( RC_BAD( rc = writeUInt32( (FLMUINT32)uiValue)))
							{
								goto Exit;
							}
						}
					}
					break;
				}

				case FLM_CONTEXT_TYPE:
				{
					/*
					Extract the value.
					*/

					if( uiCurDataLen)
					{
						/*
						The context node has a DRN value associated with
						it.  Send the value as an unsigned 32-bit number.
						*/

						FLMUINT		uiDrn;

						if( (pCurNode && RC_BAD( rc = GedGetRecPtr( pCurNode, &uiDrn))) ||
							(pCurField && RC_BAD( rc = pRecord->getUINT( pCurField, &uiDrn))))
						{
							goto Exit;
						}

						if( RC_BAD( rc = writeUInt32( (FLMUINT32)uiDrn)))
						{
							goto Exit;
						}
					}
					break;
				}

				case FLM_BINARY_TYPE:
				{
					/*
					Extract the value.
					*/

					if( uiCurDataLen)
					{
						if( RC_BAD( rc = writeUShort( (FLMUINT16)uiCurDataLen)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = write( pucCurData, uiCurDataLen)))
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
		}

		uiPrevLevel = uiCurLevel;
		if( pCurNode)
		{
			pCurNode = GedNodeNext( pCurNode);
		}
		else
		{
			pCurField = pRecord->next( pCurField);
		}
	}

	/*
	Write a zero tag to indicate the end of the transmission.
	*/

	if( RC_BAD( rc = writeUShort( 0)))
	{
		goto Exit;
	}

Exit:

	GedPoolReset( &m_tmpPool, pvMark);
	return( rc);		
}


/****************************************************************************
Desc:	Flushes any pending data and closes the stream.
****************************************************************************/
RCODE FCS_DOS::close( void)
{
	RCODE		rc = FERR_OK;

	/*
	Verify that setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Flush and terminate any pending message.
	*/

	if( RC_BAD( rc = endMessage()))
	{
		goto Exit;
	}


	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Flushes pending data.
****************************************************************************/
RCODE	FCS_DOS::flush( void)
{
	/*
	Verify that setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	/*
	Flush the output buffer.
	*/

	if( m_uiBOffset > 0)
	{
		m_pOStream->write( m_pucBuffer, m_uiBOffset);
		m_uiBOffset = 0;
	}

	/*
	Flush the parent stream.
	*/

	return( m_pOStream->flush());
}


/****************************************************************************
Desc:	Flushes and terminates the current parent stream message
****************************************************************************/
RCODE	FCS_DOS::endMessage( void)
{
	RCODE		rc = FERR_OK;

	/*
	Verify that Setup has been called.
	*/

	flmAssert( m_bSetupCalled == TRUE);

	if( !m_pOStream)
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

	if( RC_BAD( rc = m_pOStream->endMessage()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}
