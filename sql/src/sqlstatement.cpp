// Desc:	This module contains routines for doing database updates
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

#include "flaimsys.h"

/****************************************************************************
Desc:		Constructor
****************************************************************************/
SQLStatement::SQLStatement()
{
	m_fnStatus = NULL;
	m_pvCallbackData = NULL;
	m_tmpPool.poolInit( 4096);
	m_pucCurrLineBuf = NULL;
	m_uiCurrLineBufMaxBytes = 0;
	m_pXml = NULL;
	m_pColumnValues = &m_columnValues [0];
	m_uiColumnValueArraySize = SQL_DEFAULT_COLUMNS;
	resetStatement();
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
SQLStatement::~SQLStatement()
{
	resetStatement();

	if( m_pucCurrLineBuf)
	{
		f_free( &m_pucCurrLineBuf);
	}

	if (m_pColumnValues != &m_columnValues [0])
	{
		f_free( &m_pColumnValues);
	}
	m_tmpPool.poolFree();
}

/****************************************************************************
Desc:		Resets member variables so the object can be reused
****************************************************************************/
void SQLStatement::resetStatement( void)
{
	m_uiCurrLineNum = 0;
	m_uiCurrLineOffset = 0;
	m_ucUngetByte = 0;
	m_uiCurrLineFilePos = 0;
	m_uiCurrLineBytes = 0;
	m_pStream = NULL;
	m_uiFlags = 0;
	m_pDb = NULL;
	m_pTable = NULL;
	m_uiNumColumnValues = 0;
	if (m_pXml)
	{
		m_pXml->Release();
		m_pXml = NULL;
	}
	f_memset( &m_sqlStats, 0, sizeof( SQL_STATS));

	m_tmpPool.poolReset( NULL);
}

/****************************************************************************
Desc:	Initializes the SQL statement object (allocates buffers, etc.)
****************************************************************************/
RCODE SQLStatement::setupStatement( void)
{
	RCODE			rc = NE_SFLM_OK;

	resetStatement();

	if (RC_BAD( rc = FlmGetXMLObject( &m_pXml)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	See if the current line has the specified token in it starting
		from the current offset.
****************************************************************************/
FLMBOOL SQLStatement::lineHasToken(
	const char *	pszToken)
{
	FLMUINT			uiOffset;
	
	uiOffset = m_uiCurrLineOffset;
	while (uiOffset < m_uiCurrLineBytes)
	{
		if (m_pucCurrLineBuf [uiOffset] != (char)(*pszToken))
		{
			
			// Do NOT change m_uiCurrLineOffset if we return FALSE.
			
			return( FALSE);
		}
		pszToken++;
		uiOffset++;
		if (*pszToken == 0)
		{
			m_uiCurrLineOffset = uiOffset;
			return( TRUE);
		}
	}
	return( FALSE);
}

/****************************************************************************
Desc:	Get next byte from input stream.
****************************************************************************/
RCODE SQLStatement::getByte(
	FLMBYTE *	pucByte)
{
	RCODE	rc = NE_SFLM_OK;
	
	if (m_ucUngetByte)
	{
		*pucByte = m_ucUngetByte;
		m_ucUngetByte = 0;
	}
	else
	{
		if( RC_BAD( rc = m_pStream->read( (char *)pucByte, 1, NULL)))
		{
			goto Exit;
		}
	}
	m_sqlStats.uiChars++;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads next line from the input stream.
****************************************************************************/
RCODE SQLStatement::getLine( void)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucBytes [4];
	FLMUINT		uiNumBytes;
	FLMUINT		uiLoop;
	
	m_uiCurrLineBytes = 0;
	m_uiCurrLineOffset = 0;
	m_uiCurrLineFilePos = m_sqlStats.uiChars;	

	for (;;)
	{
		if( RC_BAD( rc = getByte( &ucBytes [0])))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				if (m_uiCurrLineBytes)
				{
					rc = NE_SFLM_OK;
				}
			}
			goto Exit;
		}
		
		// Keep count of the characters.
		
		if( m_fnStatus && (m_sqlStats.uiChars % 1024) == 0)
		{
			m_fnStatus( SQL_PARSE_STATS,
				(void *)&m_sqlStats, NULL, NULL, m_pvCallbackData);
		}
		
		// Convert CRLF->CR

		if( ucBytes [0] == ASCII_CR)
		{
			if( RC_BAD( rc = getByte( &ucBytes [0])))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = NE_SFLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}

			if( ucBytes [0] != ASCII_NEWLINE)
			{
				ungetByte( ucBytes [0]);
			}
			
			// End of the line
			
			break;
		}
		else if (ucBytes [0] == ASCII_NEWLINE)
		{
			
			// End of the line
			
			break;
		}

		if( ucBytes [0] <= 0x7F)
		{
			uiNumBytes = 1;
		}
		else
		{

			if( RC_BAD( rc = getByte( &ucBytes [1])))
			{
				if (rc == NE_SFLM_EOF_HIT)
				{
					rc = RC_SET( NE_SFLM_BAD_UTF8);
				}
				goto Exit;
			}
	
			if( (ucBytes [1] >> 6) != 0x02)
			{
				rc = RC_SET( NE_SFLM_BAD_UTF8);
				goto Exit;
			}
	
			if( (ucBytes [0] >> 5) == 0x06)
			{
				uiNumBytes = 2;
			}
			else
			{
				if( RC_BAD( rc = getByte( &ucBytes [2])))
				{
					if (rc == NE_SFLM_EOF_HIT)
					{
						rc = RC_SET( NE_SFLM_BAD_UTF8);
					}
					goto Exit;
				}
		
				if( (ucBytes [2] >> 6) != 0x02 || (ucBytes [0] >> 4) != 0x0E)
				{
					rc = RC_SET( NE_SFLM_BAD_UTF8);
					goto Exit;
				}
				uiNumBytes = 3;
			}
		}

		// We have a character, add it to the current line.
		
		if (m_uiCurrLineBytes + uiNumBytes > m_uiCurrLineBufMaxBytes)
		{
			// Allocate more space for the line buffer
			
			if (RC_BAD( rc = f_realloc( m_uiCurrLineBufMaxBytes + 512,
						&m_pucCurrLineBuf)))
			{
				goto Exit;
			}
			m_uiCurrLineBufMaxBytes += 512;
		}
		for (uiLoop = 0; uiLoop < uiNumBytes; uiLoop++)
		{
			m_pucCurrLineBuf [m_uiCurrLineBytes++] = ucBytes [uiLoop];
		}
	}

	// Increment the line count

	m_uiCurrLineNum++;			
	m_sqlStats.uiLines++;
	if( m_fnStatus && (m_sqlStats.uiLines % 100) == 0)
	{
		m_fnStatus( SQL_PARSE_STATS,
			(void *)&m_sqlStats, NULL, NULL, m_pvCallbackData);
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Skips any whitespace characters in the input stream
****************************************************************************/
RCODE SQLStatement::skipWhitespace(
	FLMBOOL 			bRequired)
{
	FLMBYTE		ucChar;
	FLMUINT		uiCount = 0;
	RCODE			rc = NE_SFLM_OK;

	for( ;;)
	{
		if ((ucChar = getChar()) == 0)
		{
			uiCount++;
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			continue;
		}

		if (ucChar != ASCII_SPACE && ucChar != ASCII_TAB)
		{
			ungetChar();
			break;
		}
		uiCount++;
	}

	if( !uiCount && bRequired)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_EXPECTING_WHITESPACE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a table, column, or index name.
//------------------------------------------------------------------------------
RCODE SQLStatement::getName(
	char *		pszName,
	FLMUINT		uiNameBufSize,
	FLMUINT *	puiNameLen)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiCharCount = 0;
	FLMBYTE		ucChar;
	
	// Always leave room for a null terminating character.
	
	uiNameBufSize--;

	// Get the first character - must be between A and Z

	ucChar = getChar();

	if ((ucChar >= 'a' && ucChar <= 'z') ||
		 (ucChar >= 'A' && ucChar <= 'Z'))
	{
		*pszName = (char)ucChar;
		uiCharCount++;
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				SQL_ERR_ILLEGAL_TABLE_NAME_CHAR,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}

	// Cannot go off of the current line
	
	for (;;)
	{
		if ((ucChar = getChar()) == 0)
		{
			break;
		}
		if ((ucChar >= 'a' && ucChar <= 'z') ||
			 (ucChar >= 'A' && ucChar <= 'Z') ||
			 (ucChar >= '0' && ucChar <= '9') ||
			 (ucChar == '_'))
		{
			if (uiCharCount >= uiNameBufSize)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						SQL_ERR_TABLE_NAME_TOO_LONG,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			pszName [uiCharCount++] = (char)ucChar;
		}
		else
		{
			ungetChar();
			break;
		}
	}

	pszName [uiCharCount] = 0;

Exit:

	if (puiNameLen)
	{
		*puiNameLen = uiCharCount;
	}
	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse the table name for the current statement.  Make sure it is valid.
//------------------------------------------------------------------------------
RCODE SQLStatement::getTableName(
	FLMBOOL	bMustExist)
{
	RCODE		rc = NE_SFLM_OK;
	char		szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT	uiTableNameLen;

	if (RC_BAD( rc = getName( szTableName, sizeof( szTableName), &uiTableNameLen)))
	{
		goto Exit;
	}
	
	// See if the table name is defined
	
	if (RC_BAD( rc = m_pDb->m_pDict->getTable( szTableName, &m_pTable, TRUE)))
	{
		if (rc != NE_SFLM_BAD_TABLE)
		{
			goto Exit;
		}
		if (bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					SQL_ERR_UNDEFINED_TABLE,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		else
		{
			rc = NE_SFLM_OK;
		}
	}
	else
	{
		if (!bMustExist)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					SQL_ERR_TABLE_ALREADY_DEFINED,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Reallocate the column value array if needed.
//------------------------------------------------------------------------------
RCODE SQLStatement::allocColumnValueArray(
	FLMUINT	uiNumColumnsNeeded)
{
	RCODE	rc = NE_SFLM_OK;
	
	if (uiNumColumnsNeeded > m_uiColumnValueArraySize)
	{
		F_COLUMN_VALUE *	pNewArray;
		
		// Increase the array size by at least 20.
		
		uiNumColumnsNeeded += 20;
		if (m_pColumnValues == &m_columnValues [0])
		{
			if (RC_BAD( rc = f_alloc( sizeof( F_COLUMN_VALUE) * uiNumColumnsNeeded,
										&pNewArray)))
			{
				goto Exit;
			}
			if (m_uiNumColumnValues)
			{
				f_memcpy( pNewArray, m_pColumnValues,
					m_uiNumColumnValues * sizeof( F_COLUMN_VALUE));
			}
			m_pColumnValues = pNewArray;
		}
		else
		{
			if (RC_BAD( rc = f_realloc( sizeof( F_COLUMN_VALUE) * uiNumColumnsNeeded,
										&m_pColumnValues)))
			{
				goto Exit;
			}
		}
		m_uiColumnValueArraySize = uiNumColumnsNeeded;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a string value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getStringValue(
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMBYTE		ucQuoteChar = 0;
	FLMBOOL		bEscaped = FALSE;
	FLMBYTE		szTmpBuf [300];
	F_DynaBuf	dynaBuf( szTmpBuf, sizeof( szTmpBuf));
	FLMUINT		uiNumChars = 0;
	FLMBYTE *	pucValue;
	FLMUINT		uiSenLen;
	
	// Leading white space has already been skipped.
	
	// See if we have a quote character.
	
	ucChar = getChar();
	if (ucChar == '"' || ucChar == '\'')
	{
		ucQuoteChar = ucChar;
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_EXPECTING_QUOTE_CHAR,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	for (;;)
	{
		// Should not hit the end of the line if quoted.
		
		if ((ucChar = getChar()) == 0)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_MISSING_QUOTE,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		if (bEscaped)
		{
			// Can only escape backslash (the escape character), quotes, and
			// a few other characters.
			
			if (ucChar == 'n')
			{
				ucChar = ASCII_NEWLINE;
			}
			else if (ucChar == 't')
			{
				ucChar = ASCII_TAB;
			}
			else if (ucChar == 'r')
			{
				ucChar = ASCII_CR;
			}
			else if (ucChar == '\'' || ucChar == '"')
			{
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_INVALID_ESCAPED_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			
			// Save the escaped character to the buffer.

			if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
			{
				goto Exit;
			}
			uiNumChars++;
		}
		else if (ucChar == '\\')
		{
			bEscaped = TRUE;
		}
		else if (ucChar == ucQuoteChar)
		{
			break;
		}
		else
		{
			
			// Save the character to our buffer.
			
			if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
			{
				goto Exit;
			}
			
			// Handle multi-byte UTF8 characters.  The getLine() method has
			// already checked for valid UTF8, so that is all we should be
			// seeing here - thus the asserts.
			
			if (ucChar > 0x7F)
			{
				
				// It is at least two bytes.
				
				ucChar = getChar();
				flmAssert( (ucChar >> 6) == 0x02);
				if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
				{
					goto Exit;
				}
				
				// See if it is three bytes.
				
				if ((ucChar >> 5) != 0x06)
				{
					ucChar = getChar();
					flmAssert( (ucChar >> 6) == 0x02);
					if (RC_BAD( rc = dynaBuf.appendByte( ucChar)))
					{
						goto Exit;
					}
				}
			}
			uiNumChars++;
		}
	}
	
	// Add a null terminating byte
	
	if (RC_BAD( rc = dynaBuf.appendByte( 0)))
	{
		goto Exit;
	}
	
	// Allocate space for the UTF8 string.
	
	uiSenLen = f_getSENByteCount( uiNumChars);
	pColumnValue->uiValueLen = dynaBuf.getDataLength() + uiSenLen;
	if (RC_BAD( rc = m_tmpPool.poolAlloc( pColumnValue->uiValueLen,
											(void **)&pucValue)))
	{
		goto Exit;
	}
	pColumnValue->pucColumnValue = pucValue;
	f_encodeSEN( uiNumChars, &pucValue);
	
	// Copy the string from the dynaBuf to the column.
	
	f_memcpy( pucValue, dynaBuf.getBufferPtr(), dynaBuf.getDataLength());
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a numeric value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getNumberValue(
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMUINT64	ui64Value = 0;
	FLMBOOL		bNeg = FALSE;
	FLMBOOL		bHex = FALSE;
	FLMUINT		uiDigitCount = 0;
	FLMUINT		uiDigitValue = 0;
	FLMUINT		uiSavedLineNum = m_uiCurrLineNum;
	FLMUINT		uiSavedOffset = m_uiCurrLineOffset;
	FLMUINT		uiSavedFilePos = m_uiCurrLineFilePos;
	FLMUINT		uiSavedLineBytes = m_uiCurrLineBytes;
	FLMBYTE *	pucValue;
	
	// Leading white space has already been skipped.
	
	// Go until we hit a character that is not a number.
	
	for (;;)
	{
		
		// If we hit the end of the line, we are done.
		
		if ((ucChar = getChar()) == 0)
		{
			break;
		}
		
		// Ignore white space
		
		{
			continue;
		}
		if (ucChar >= '0' && ucChar <= '9')
		{
			uiDigitValue = (FLMUINT)(ucChar - '0');
			uiDigitCount++;
		}
		else if (ucChar >= 'a' && ucChar <= 'f')
		{
			if (!bHex)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_ILLEGAL_HEX_DIGIT,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			uiDigitValue = (FLMUINT)(ucChar - 'a' + 10);
			uiDigitCount++;
		}
		else if (ucChar >= 'A' && ucChar <= 'F')
		{
			if (!bHex)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_ILLEGAL_HEX_DIGIT,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			uiDigitValue = (FLMUINT)(ucChar - 'A' + 10);
			uiDigitCount++;
		}
		else if (ucChar == ',' || ucChar == ')' ||
					ucChar == ASCII_SPACE || ucChar == ASCII_TAB)
		{
			
			// terminate when we hit a comma or right paren or white
			// space.  Need to unget the character so the caller can handle it.
			
			ungetChar();
			break;
		}
		else if (ucChar == 'X' || ucChar == 'x')
		{
			if (bHex || uiDigitCount != 1 || ui64Value)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NON_NUMERIC_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			else
			{
				bHex = TRUE;
				uiDigitCount = 0;
				continue;
			}
		}
		else if (ucChar == '-')
		{
			if (bHex || uiDigitCount)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NON_NUMERIC_CHARACTER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			else
			{
				bNeg = TRUE;
				continue;
			}
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_NON_NUMERIC_CHARACTER,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		if (bHex)
		{
			if (ui64Value > (FLM_MAX_UINT64 >> 4))
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NUMBER_OVERFLOW,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			ui64Value <<= 4;
			ui64Value += (FLMUINT64)uiDigitValue;
		}
		else
		{
			if (ui64Value > (FLM_MAX_UINT64 / 10))
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_NUMBER_OVERFLOW,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
			ui64Value *= 10;
			ui64Value += (FLMUINT64)uiDigitValue;
		}
		
		// If it is a negative number, make sure we have not
		// exceeded the maximum negative value.
		
		if (bNeg && ui64Value > ((FLMUINT64)1 << 63))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_NUMBER_OVERFLOW,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
	}
	
	// If we didn't hit any digits, we have an invalid number.
	
	if (!uiDigitCount)
	{
		setErrInfo( uiSavedLineNum,
				uiSavedOffset,
				SQL_ERR_NUMBER_VALUE_EMPTY,
				uiSavedFilePos,
				uiSavedLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// Allocate space for ui64Value SEN plus one byte for the sign.
	
	pColumnValue->uiValueLen = f_getSENByteCount( ui64Value) + 1;
	if (RC_BAD( rc = m_tmpPool.poolAlloc( pColumnValue->uiValueLen,
											(void **)&pucValue)))
	{
		goto Exit;
	}
	pColumnValue->pucColumnValue = pucValue;
	
	*pucValue++ = (FLMBYTE)(bNeg ? (FLMBYTE)1 : (FLMBYTE)0);
	
	// Set the number into the data.  uiNumChars will hold bNeg.
	
	f_encodeSEN( ui64Value, &pucValue);
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a binary value from the input stream.
//------------------------------------------------------------------------------
RCODE SQLStatement::getBinaryValue(
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE			rc = NE_SFLM_OK;
	FLMBYTE		ucChar;
	FLMBYTE		szTmpBuf [300];
	F_DynaBuf	dynaBuf( szTmpBuf, sizeof( szTmpBuf));
	FLMBYTE		ucCurrByte;
	FLMBOOL		bGetHighNibble;
	FLMUINT		uiSavedLineNum = m_uiCurrLineNum;
	FLMUINT		uiSavedOffset = m_uiCurrLineOffset;
	FLMUINT		uiSavedFilePos = m_uiCurrLineFilePos;
	FLMUINT		uiSavedLineBytes = m_uiCurrLineBytes;
	
	// Leading white space has already been skipped.
	
	// Go until we hit a character that is not a hex digit.
	
	ucCurrByte = 0;
	bGetHighNibble = TRUE;
	for (;;)
	{
		
		// It is OK for white space to be in the middle of a binary
		// piece of data.  It is also allowed to span multiple lines.
		
		if ((ucChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			continue;
		}
		
		// Ignore white space
		
		if (ucChar == ASCII_SPACE || ucChar == ASCII_TAB)
		{
			continue;
		}
		if (ucChar >= '0' && ucChar <= '9')
		{
			if (bGetHighNibble)
			{
				ucCurrByte = (ucChar - '0') << 4;
			}
			else
			{
				ucCurrByte |= (ucChar - '0');
			}
		}
		else if (ucChar >= 'a' && ucChar <= 'f')
		{
			if (bGetHighNibble)
			{
				ucCurrByte = (ucChar - 'a' + 10) << 4;
			}
			else
			{
				ucCurrByte |= (ucChar - 'a' + 10);
			}
		}
		else if (ucChar >= 'A' && ucChar <= 'F')
		{
			if (bGetHighNibble)
			{
				ucCurrByte = (ucChar - 'A' + 10) << 4;
			}
			else
			{
				ucCurrByte |= (ucChar - 'A' + 10);
			}
		}
		else if (ucChar == ',' || ucChar == ')')
		{
			
			// terminate when we hit a comma or right paren.  Need to
			// unget the character because the caller will be expecting
			// to see it.
			
			ungetChar();
			break;
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_NON_HEX_CHARACTER,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		if (bGetHighNibble)
		{
			bGetHighNibble = FALSE;
		}
		else
		{
			if (RC_BAD( rc = dynaBuf.appendByte( ucCurrByte)))
			{
				goto Exit;
			}
			bGetHighNibble = TRUE;
			ucCurrByte = 0;
		}
	}
	
	// Add last byte if bGetHighNibble is FALSE - means we got the high nibble
	// into the high four bits of ucCurrByte
	
	if (!bGetHighNibble)
	{
		if (RC_BAD( rc = dynaBuf.appendByte( ucCurrByte)))
		{
			goto Exit;
		}
	}
	
	// An empty binary value is invalid.
	
	if ((pColumnValue->uiValueLen = dynaBuf.getDataLength()) == 0)
	{
		setErrInfo( uiSavedLineNum,
				uiSavedOffset,
				SQL_ERR_BINARY_VALUE_EMPTY,
				uiSavedFilePos,
				uiSavedLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// Allocate space for the binary data.
	
	if (RC_BAD( rc = m_tmpPool.poolAlloc( pColumnValue->uiValueLen,
											(void **)&pColumnValue->pucColumnValue)))
	{
		goto Exit;
	}
	
	// Copy the binary data from the dynaBuf to the column.
	
	f_memcpy( pColumnValue->pucColumnValue, dynaBuf.getBufferPtr(),
				 pColumnValue->uiValueLen);
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse a value from the input stream.  Value must be of the specified
//			type.
//------------------------------------------------------------------------------
RCODE SQLStatement::getValue(
	F_COLUMN_VALUE *	pColumnValue)
{
	RCODE	rc = NE_SFLM_OK;
	
	switch (pColumnValue->eColumnDataType)
	{
		case SFLM_STRING_TYPE:
			if (RC_BAD( rc = getStringValue( pColumnValue)))
			{
				goto Exit;
			}
			break;
		case SFLM_NUMBER_TYPE:
			if (RC_BAD( rc = getNumberValue( pColumnValue)))
			{
				goto Exit;
			}
			break;
		case SFLM_BINARY_TYPE:
			if (RC_BAD( rc = getBinaryValue( pColumnValue)))
			{
				goto Exit;
			}
			break;
		default:
			flmAssert( 0);
			break;
	}
	
Exit:

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Insert a row into the database.
//------------------------------------------------------------------------------
RCODE F_Db::insertRow(
	FLMUINT				uiTableNum,
	F_COLUMN_VALUE *	pColumnValues,
	FLMUINT				uiNumColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	const FLMBYTE *	pucValue;
	const FLMBYTE *	pucEnd;
	FLMUINT64			ui64Num;
	FLMUINT				uiNumChars;
	FLMBOOL				bNeg;

	// Create a row object.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
								uiTableNum, &pRow)))
	{
		goto Exit;
	}
	
	// Set the column values into the row.
	
	for (; uiNumColumnValues; uiNumColumnValues--, pColumnValues++)
	{
		if (!pColumnValues->uiValueLen)
		{
			continue;
		}
		switch (pColumnValues->eColumnDataType)
		{
			case SFLM_STRING_TYPE:
				pucValue = (const FLMBYTE *)pColumnValues->pucColumnValue;
				pucEnd = pucValue + pColumnValues->uiValueLen;
				if (RC_BAD( rc = f_decodeSEN( &pucValue, pucEnd, &uiNumChars)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pRow->setUTF8( this,
												pColumnValues->uiColumnNum,
												(const char *)pucValue,
												(FLMUINT)(pucEnd - pucValue),
												uiNumChars)))
				{
					goto Exit;
				}
				break;
			case SFLM_NUMBER_TYPE:
				pucValue = (const FLMBYTE *)pColumnValues->pucColumnValue;
				pucEnd = pucValue + pColumnValues->uiValueLen;
				
				bNeg = (FLMBOOL)(*pucValue ? (FLMBOOL)TRUE : (FLMBOOL)FALSE);
				pucValue++;
				
				if (RC_BAD( rc = f_decodeSEN64( &pucValue, pucEnd, &ui64Num)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pRow->setNumber64( this,
										pColumnValues->uiColumnNum, ui64Num, bNeg)))
				{
					goto Exit;
				}
				break;
			case SFLM_BINARY_TYPE:
				if (RC_BAD( rc = pRow->setBinary( this,
												pColumnValues->uiColumnNum,
												(const void *)(pColumnValues->pucColumnValue),
												pColumnValues->uiValueLen)))
				{
					goto Exit;
				}
				break;
			default:
				flmAssert( 0);
				break;
		}
	}
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logInsertRow( this, uiTableNum, pColumnValues,
										uiNumColumnValues)))
	{
		goto Exit;
	}

Exit:

	if (pRow)
	{
		pRow->ReleaseRow();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Process the insert statement.  The "INSERT" keyword has already been
//			parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processInsertRow( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	F_COLUMN_VALUE *	pColumnValue;
	F_COLUMN *			pColumn;
	FLMUINT				uiLoop;
	char					szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiColumnNameLen;

	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: INSERT INTO table_name (column1,column2,...) VALUES (value1,value2,...)
	// OR:     INSERT INTO table_name VALUES (value1,value2,...)

	// Whitespace must follow the "INSERT"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// INTO must follow the INSERT.

	if (!lineHasToken( "into"))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_EXPECTING_INTO,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}

	// Whitespace must follow the "INTO"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// Get the table name.

	if (RC_BAD( rc = getTableName( TRUE)))
	{
		goto Exit;
	}
	
	if (m_pTable->bSystemTable)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_CANNOT_UPDATE_SYSTEM_TABLE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
	// Whitespace does not have to follow the table name

	if (RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	// If left paren follows table name, then columns are being listed.

	m_uiNumColumnValues = 0;
	if (lineHasToken( "("))
	{

		// Get the list of columns for which there will be values.
		
		for (;;)
		{
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			
			// Get the column name
			
			if (RC_BAD( rc = getName( szColumnName, sizeof( szColumnName),
											&uiColumnNameLen)))
			{
				goto Exit;
			}
			
			// See if the column is defined in the table.
			
			if (uiColumnNameLen)
			{
				if ((pColumn = m_pDb->m_pDict->findColumn( m_pTable, szColumnName)) == NULL)
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset,
							SQL_ERR_UNDEFINED_COLUMN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				
				// Make room in the column value table for this column
				
				if (m_uiNumColumnValues == m_uiColumnValueArraySize)
				{
					if (RC_BAD( rc = allocColumnValueArray( m_uiNumColumnValues + 1)))
					{
						goto Exit;
					}
				}
				pColumnValue = &m_pColumnValues [m_uiNumColumnValues];
				pColumnValue->uiColumnNum = pColumn->uiColumnNum;
				pColumnValue->eColumnDataType = pColumn->eDataTyp;
				pColumnValue->uiValueLen = 0;
				m_uiNumColumnValues++;
			}

			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			
			// See if we are at the end of the list of columns
			
			if (lineHasToken( ")"))
			{
				break;
			}
			else if (!lineHasToken( ","))
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_EXPECTING_COMMA,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
	}
	else
	{
		
		// Allocate the column value array

		if (RC_BAD( rc = allocColumnValueArray( m_pTable->uiNumColumns)))
		{
			goto Exit;
		}
		for (uiLoop = 0, pColumn = m_pTable->pColumns, pColumnValue = m_pColumnValues;
			  uiLoop < m_pTable->uiNumColumns;
			  uiLoop++, pColumn++, pColumnValue++)
		{
			if (pColumn->uiColumnNum)
			{
				pColumnValue->uiColumnNum = pColumn->uiColumnNum;
				pColumnValue->eColumnDataType = pColumn->eDataTyp;
				pColumnValue->uiValueLen = 0;
				m_uiNumColumnValues++;
			}
		}
	}

	// Allow for no values to be specified if no columns were.
	
	if (m_uiNumColumnValues)
	{
		if (!lineHasToken( "values"))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_EXPECTING_INTO,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
			
		// Should be a left paren
		
		if (!lineHasToken( "("))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					SQL_ERR_EXPECTING_LPAREN,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_SFLM_INVALID_SQL);
			goto Exit;
		}
		
		for (uiLoop = 0, pColumnValue = m_pColumnValues;
			  uiLoop < m_uiNumColumnValues;
			  uiLoop++, pColumnValue++)
		{
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			
			// Get the column value
			
			if (RC_BAD( rc = getValue( pColumnValue)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			if (uiLoop == m_uiNumColumnValues - 1)
			{
				if (!lineHasToken( ")"))
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset,
							SQL_ERR_EXPECTING_RPAREN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				else
				{
					break;
				}
			}
			else if (!lineHasToken( ","))
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						SQL_ERR_EXPECTING_COMMA,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_SFLM_INVALID_SQL);
				goto Exit;
			}
		}
	}
	
	// Insert the row.
	
	if (RC_BAD( rc = m_pDb->insertRow( m_pTable->uiTableNum,
										m_pColumnValues, m_uiNumColumnValues)))
	{
		goto Exit;
	}
	
	// Commit the transaction if we started it
	
	if (bStartedTrans)
	{
		bStartedTrans = FALSE;
		if (RC_BAD( rc = m_pDb->transCommit()))
		{
			goto Exit;
		}
	}

Exit:

	if (bStartedTrans)
	{
		m_pDb->transAbort();
	}

	return( rc);
}

//------------------------------------------------------------------------------
// Desc:	Parse and execute an SQL statement.
//------------------------------------------------------------------------------
RCODE SQLStatement::executeSQL(
	IF_IStream *	pStream,
	F_Db *			pDb,
	SQL_STATS *		pSQLStats)
{
	RCODE		rc = NE_SFLM_OK;
	FLMBOOL	bWhitespaceRequired = FALSE;

	// Reset the state of the parser

	if (RC_BAD( rc = setupStatement()))
	{
		goto Exit;
	}

	m_pStream = pStream;
	m_pDb = pDb;

	// Process all of the statements.

	for (;;)
	{
		if (RC_BAD( rc = skipWhitespace( bWhitespaceRequired)))
		{
			if (rc == NE_SFLM_EOF_HIT)
			{
				rc = NE_SFLM_OK;
				break;
			}
			goto Exit;
		}

		if (lineHasToken( "insert"))
		{
			if( RC_BAD( rc = processInsertRow()))
			{
				goto Exit;
			}
		}
		bWhitespaceRequired = TRUE;
	}

	// Call the status hook one last time

	if (m_fnStatus)
	{
		m_fnStatus( SQL_PARSE_STATS,
			(void *)&m_sqlStats, NULL, NULL, m_pvCallbackData);
	}

	// Tally and return the SQL statistics

	if( pSQLStats)
	{
		pSQLStats->uiLines += m_sqlStats.uiLines;
		pSQLStats->uiChars += m_sqlStats.uiChars;
	}

Exit:

	if( RC_BAD( rc) && pSQLStats)
	{
		pSQLStats->uiErrLineNum = m_sqlStats.uiErrLineNum
			? m_sqlStats.uiErrLineNum
			: m_uiCurrLineNum;

		pSQLStats->uiErrLineOffset = m_sqlStats.uiErrLineOffset
			? m_sqlStats.uiErrLineOffset
			: m_uiCurrLineOffset;

		pSQLStats->eErrorType = m_sqlStats.eErrorType;
		
		pSQLStats->uiErrLineFilePos = m_sqlStats.uiErrLineFilePos;
		pSQLStats->uiErrLineBytes = m_sqlStats.uiErrLineBytes;
	}

	m_pDb = NULL;

	return( rc);
}
