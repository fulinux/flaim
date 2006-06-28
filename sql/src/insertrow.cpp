//------------------------------------------------------------------------------
// Desc:	This module contains the routines for inserting a row into a table.
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

//------------------------------------------------------------------------------
// Desc:	Insert a row into the database.
//------------------------------------------------------------------------------
RCODE F_Db::insertRow(
	FLMUINT				uiTableNum,
	F_COLUMN_VALUE *	pColumnValues)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	const FLMBYTE *	pucValue;
	const FLMBYTE *	pucEnd;
	FLMUINT64			ui64Num;
	FLMUINT				uiNumChars;
	FLMBOOL				bNeg;
	F_COLUMN_VALUE *	pColumnValue;
	F_TABLE *			pTable = m_pDict->getTable( uiTableNum);
	F_COLUMN *			pColumn;

	// Create a row object.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
								uiTableNum, &pRow)))
	{
		goto Exit;
	}
	
	// Set the column values into the row.
	
	for (pColumnValue = pColumnValues;
		  pColumnValue;
		  pColumnValue = pColumnValue->pNext)
	{
		if (!pColumnValue->uiValueLen)
		{
			continue;
		}
		pColumn = m_pDict->getColumn( pTable, pColumnValue->uiColumnNum);
		switch (pColumn->eDataTyp)
		{
			case SFLM_STRING_TYPE:
				pucValue = (const FLMBYTE *)pColumnValue->pucColumnValue;
				pucEnd = pucValue + pColumnValue->uiValueLen;
				if (RC_BAD( rc = f_decodeSEN( &pucValue, pucEnd, &uiNumChars)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pRow->setUTF8( this,
												pColumnValue->uiColumnNum,
												(const char *)pucValue,
												(FLMUINT)(pucEnd - pucValue),
												uiNumChars)))
				{
					goto Exit;
				}
				break;
			case SFLM_NUMBER_TYPE:
				pucValue = (const FLMBYTE *)pColumnValue->pucColumnValue;
				pucEnd = pucValue + pColumnValue->uiValueLen;
				
				bNeg = (FLMBOOL)(*pucValue ? (FLMBOOL)TRUE : (FLMBOOL)FALSE);
				pucValue++;
				
				if (RC_BAD( rc = f_decodeSEN64( &pucValue, pucEnd, &ui64Num)))
				{
					goto Exit;
				}
				if (RC_BAD( rc = pRow->setNumber64( this,
										pColumnValue->uiColumnNum, ui64Num, bNeg)))
				{
					goto Exit;
				}
				break;
			case SFLM_BINARY_TYPE:
				if (RC_BAD( rc = pRow->setBinary( this,
												pColumnValue->uiColumnNum,
												(const void *)(pColumnValue->pucColumnValue),
												pColumnValue->uiValueLen)))
				{
					goto Exit;
				}
				break;
			default:
				flmAssert( 0);
				break;
		}
	}
	
	// Do whatever indexing needs to be done.
	
	if (RC_BAD( rc = updateIndexKeys( uiTableNum, NULL, pRow)))
	{
		goto Exit;
	}
	
	// Log the insert row.
	
	if (RC_BAD( rc = m_pDatabase->m_pRfl->logInsertRow( this, uiTableNum,
							pColumnValues)))
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
	F_COLUMN_VALUE *	pFirstColValue;
	F_COLUMN_VALUE *	pLastColValue;
	F_COLUMN_VALUE *	pColumnValue;
	F_COLUMN *			pColumn;
	FLMUINT				uiLoop;
	char					szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiColumnNameLen;
	char					szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiTableNameLen;
	F_TABLE *			pTable;

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

	if (RC_BAD( rc = getTableName( TRUE, szTableName, sizeof( szTableName),
								&uiTableNameLen, &pTable)))
	{
		goto Exit;
	}
	
	if (pTable->bSystemTable)
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

	pFirstColValue = NULL;
	pLastColValue = NULL;
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
				if ((pColumn = m_pDb->m_pDict->findColumn( pTable, szColumnName)) == NULL)
				{
					setErrInfo( m_uiCurrLineNum,
							m_uiCurrLineOffset,
							SQL_ERR_UNDEFINED_COLUMN,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_SFLM_INVALID_SQL);
					goto Exit;
				}
				
				// Allocate a column value.
				
				if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_VALUE),
											(void **)&pColumnValue)))
				{
					goto Exit;
				}
				
				pColumnValue->uiColumnNum = pColumn->uiColumnNum;
				pColumnValue->uiValueLen = 0;
				pColumnValue->pNext = NULL;
				if (pLastColValue)
				{
					pLastColValue->pNext = pColumnValue;
				}
				else
				{
					pFirstColValue = pColumnValue;
				}
				pLastColValue = pColumnValue;
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
		
		for (uiLoop = 0, pColumn = pTable->pColumns;
			  uiLoop < pTable->uiNumColumns;
			  uiLoop++, pColumn++)
		{
			if (pColumn->uiColumnNum)
			{
				// Allocate a column value.
				
				if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_VALUE),
											(void **)&pColumnValue)))
				{
					goto Exit;
				}
				pColumnValue->uiColumnNum = pColumn->uiColumnNum;
				pColumnValue->uiValueLen = 0;
				pColumnValue->pNext = NULL;
				if (pLastColValue)
				{
					pLastColValue->pNext = pColumnValue;
				}
				else
				{
					pFirstColValue = pColumnValue;
				}
				pLastColValue = pColumnValue;
			}
		}
	}

	// Allow for no values to be specified if no columns were.
	
	if (pFirstColValue)
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
		
		pColumnValue = pFirstColValue;
		for (;;)
		{
			pColumn = m_pDb->m_pDict->getColumn( pTable, pColumnValue->uiColumnNum);
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			
			// Get the column value
			
			if (RC_BAD( rc = getValue( pColumn, pColumnValue)))
			{
				goto Exit;
			}
			
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			if ((pColumnValue = pColumnValue->pNext) == NULL)
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
	
	if (RC_BAD( rc = m_pDb->insertRow( pTable->uiTableNum, pFirstColValue)))
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


