//------------------------------------------------------------------------------
// Desc:	This module contains the routines for creating a table.
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
// Desc:	Create a new table in the database.  Caller should have already
//			verified that the table name is unique.  This routine will verify
//			that column names are unique in the table, and it will assign
//			the table number as well as the column numbers.
//------------------------------------------------------------------------------
RCODE F_Db::createTable(
	FLMUINT				uiTableNum,
	const char *		pszTableName,
	FLMUINT				uiTableNameLen,
	FLMUINT				uiEncDefNum,
	F_COLUMN_DEF *		pColumnDefs,
	FLMUINT				uiNumColumnDefs)
{
	RCODE					rc = NE_SFLM_OK;
	F_Row *				pRow = NULL;
	F_TABLE *			pTable;
	F_COLUMN *			pColumn;
	const char *		pszTmp;
	FLMUINT				uiLen;
	F_COLUMN_DEF *		pColumnDef;
	FLMUINT				uiColumnNum;
	
	// Create a new dictionary, if we don't already have one.
	
	if (!(m_uiFlags & FDB_UPDATED_DICTIONARY))
	{
		if (RC_BAD( rc = dictClone()))
		{
			goto Exit;
		}
	}
	
	// Create a row for the table in the table definition table.
	
	if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
								SFLM_TBLNUM_TABLES, &pRow)))
	{
		goto Exit;
	}
	
	// Determine the table number to use - find lowest non-used table
	// number.
	
	if (!uiTableNum)
	{
		uiTableNum = 1;
		while (uiTableNum <= m_pDict->m_uiHighestTableNum)
		{
			if (!m_pDict->m_pTableTbl [uiTableNum - 1].uiTableNum)
			{
				break;
			}
			uiTableNum++;
		}
	}
	
	// The call to addTable will initialize either the empty slot we found, or
	// the next slot at the end of the table table.  It will reallocate the
	// table array if necessary.
	
	if (RC_BAD( rc = m_pDict->addTable( uiTableNum, pRow->m_ui64RowId,
										pszTableName, FALSE, uiNumColumnDefs, uiEncDefNum)))
	{
		goto Exit;
	}
	pTable = m_pDict->getTable( uiTableNum);
	
	// Populate the columns for the row in the table definition table.
	
	if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_TABLES_TABLE_NAME,
								pszTableName, uiTableNameLen, uiTableNameLen)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_TABLES_TABLE_NUM,
								uiTableNum)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_TABLES_ENCDEF_NUM,
								uiEncDefNum)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_TABLES_NUM_COLUMNS,
								uiNumColumnDefs)))
	{
		goto Exit;
	}
	
	// Add all of the columns to the column definition table.
	
	for (uiColumnNum = 0, pColumnDef = pColumnDefs;
		  pColumnDef;
		  pColumnDef = pColumnDef->pNext)
	{
		
		// Create a row for the column in the column definition table.
		
		if (pRow)
		{
			pRow->ReleaseRow();
			pRow = NULL;
		}
		if (RC_BAD( rc = gv_SFlmSysData.pRowCacheMgr->createRow( this,
									SFLM_TBLNUM_COLUMNS, &pRow)))
		{
			goto Exit;
		}
		
		// Verify that the column name is unique.
		
		if ((pColumn = m_pDict->findColumn( pTable,
								pColumnDef->pszColumnName)) != NULL)
		{
			rc = RC_SET( NE_SFLM_COLUMN_NAME_ALREADY_DEFINED);
			goto Exit;
		}
		
		// Add the column definition to the in-memory dictionary structures.
		
		uiColumnNum++;
		if (RC_BAD( rc = m_pDict->addColumn( uiTableNum, pRow->m_ui64RowId,
										uiColumnNum, pColumnDef->pszColumnName,
										pColumnDef->uiFlags, pColumnDef->eColumnDataType,
										pColumnDef->uiMaxLen, pColumnDef->uiEncDefNum)))
		{
			goto Exit;
		}
		
		// Populate the columns for the row in the column definition table.
		
		if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_COLUMNS_TABLE_NUM,
											uiTableNum)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_COLUMNS_COLUMN_NAME,
											pColumnDef->pszColumnName,
											pColumnDef->uiColumnNameLen,
											pColumnDef->uiColumnNameLen)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_COLUMNS_COLUMN_NUM,
											uiColumnNum)))
		{
			goto Exit;
		}
		switch (pColumnDef->eColumnDataType)
		{
			case SFLM_STRING_TYPE:
				pszTmp = SFLM_STRING_OPTION_STR;
				uiLen = SFLM_STRING_OPTION_STR_LEN;
				break;
			case SFLM_NUMBER_TYPE:
				pszTmp = SFLM_INTEGER_OPTION_STR;
				uiLen = SFLM_INTEGER_OPTION_STR_LEN;
				break;
			case SFLM_BINARY_TYPE:
				pszTmp = SFLM_BINARY_OPTION_STR;
				uiLen = SFLM_BINARY_OPTION_STR_LEN;
				break;
			default:
				flmAssert( 0);
				pszTmp = "Bad";
				uiLen = 3;
				break;
		}
		if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_COLUMNS_DATA_TYPE,
											pszTmp, uiLen, uiLen)))
		{
			goto Exit;
		}
		if (pColumnDef->uiMaxLen)
		{
			if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_COLUMNS_MAX_LEN,
												pColumnDef->uiMaxLen)))
			{
				goto Exit;
			}
		}
		if (pColumnDef->uiEncDefNum)
		{
			if (RC_BAD( rc = pRow->setUINT( this, SFLM_COLNUM_COLUMNS_ENCDEF_NUM,
												pColumnDef->uiEncDefNum)))
			{
				goto Exit;
			}
		}
		if (pColumnDef->uiFlags & COL_READ_ONLY)
		{
			if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_COLUMNS_READ_ONLY,
												"yes", 3, 3)))
			{
				goto Exit;
			}
		}
		if (pColumnDef->uiFlags & COL_NULL_ALLOWED)
		{
			if (RC_BAD( rc = pRow->setUTF8( this, SFLM_COLNUM_COLUMNS_NULL_ALLOWED,
												"yes", 3, 3)))
			{
				goto Exit;
			}
		}
	}
	
	flmAssert( uiColumnNum == uiNumColumnDefs);

	if (RC_BAD( rc = m_pDatabase->m_pRfl->logCreateTable( this, uiTableNum,
										pszTableName, uiTableNameLen,
										uiEncDefNum, pColumnDefs)))
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
// Desc:	Parse the data type for a column.
//------------------------------------------------------------------------------
RCODE SQLStatement::getDataType(
	eDataType *	peDataType,
	FLMUINT *	puiMax)
{
	RCODE		rc = NE_SFLM_OK;
	
	// Leading whitespace has already been skipped
	
	// Valid data types are:
	//		char(n)
	//		varchar(n)
	//		long varchar
	//		varwchar(n)	- unicode
	//		longwvarchar - unicode
	//		smallint - 16 bit signed integer
	//		integer - 32 bit signed integer
	//		tinyint - 8 bit signed integer
	//		bigint - 64 bit signed integer
	//		binary(n)
	//		varbinary(n)
	//		long varbinary
	
	if (lineHasToken( "char") ||
		 lineHasToken( "varchar") ||
		 lineHasToken( "varwchar"))
	{
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
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
		if (RC_BAD( rc = getUINT( FALSE, puiMax)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
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
		*peDataType = SFLM_STRING_TYPE;
	}
	else if (lineHasToken( "longvarchar") ||
				lineHasToken( "longwvarchar"))
	{
		*peDataType = SFLM_STRING_TYPE;
		*puiMax = 0;
	}
	else if (lineHasToken( "longvarbinary"))
	{
		*peDataType = SFLM_BINARY_TYPE;
		*puiMax = 0;
	}
	else if (lineHasToken( "long"))
	{
		if (RC_BAD( rc = skipWhitespace( TRUE)))
		{
			goto Exit;
		}
		if (lineHasToken( "varchar") ||
			 lineHasToken( "varwchar"))
		{
			*peDataType = SFLM_STRING_TYPE;
			*puiMax = 0;
		}
		else if (lineHasToken( "varbinary"))
		{
			*peDataType = SFLM_BINARY_TYPE;
			*puiMax = 0;
		}
		else
		{
			goto Invalid_Data_Type;
		}
	}
	else if (lineHasToken( "smallint") ||
				lineHasToken( "integer") ||
				lineHasToken( "tinyint") ||
				lineHasToken( "bigint"))
	{
		*peDataType = SFLM_NUMBER_TYPE;
	}
	else if (lineHasToken( "binary") ||
			   lineHasToken( "varbinary"))
	{
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
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
		if (RC_BAD( rc = getUINT( FALSE, puiMax)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
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
		*peDataType = SFLM_BINARY_TYPE;
	}
	else
	{
Invalid_Data_Type:
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				SQL_ERR_INVALID_DATA_TYPE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_SFLM_INVALID_SQL);
		goto Exit;
	}
	
Exit:
	return( rc);
}
		
//------------------------------------------------------------------------------
// Desc:	Process the create table statement.  The "CREATE TABLE" keywords
//			have already been parsed.
//------------------------------------------------------------------------------
RCODE SQLStatement::processCreateTable( void)
{
	RCODE					rc = NE_SFLM_OK;
	FLMBOOL				bStartedTrans = FALSE;
	char					szTableName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiTableNameLen;
	char					szColumnName [MAX_SQL_NAME_LEN + 1];
	FLMUINT				uiColumnNameLen;
	char *				pszTmp;
	F_COLUMN_DEF *		pColumnDef;
	F_COLUMN_DEF *		pFirstColDef;
	F_COLUMN_DEF *		pLastColDef;
	FLMUINT				uiNumColumnDefs;
	F_TABLE *			pTable;
	
	// If we are in a read transaction, we cannot do this operation
	
	if (RC_BAD( rc = m_pDb->checkTransaction( SFLM_UPDATE_TRANS, &bStartedTrans)))
	{
		goto Exit;
	}

	// SYNTAX: CREATE TABLE <tablename> (<column_name> <data_type,...)

	// Whitespace must follow the "CREATE TABLE"

	if (RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	// Get the table name - must not already be defined.

	if (RC_BAD( rc = getTableName( FALSE, szTableName, sizeof( szTableName),
								&uiTableNameLen, &pTable)))
	{
		goto Exit;
	}
	
	// Skip any whitespace after the table name.

	if (RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	// Left paren must follow table name
	
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
	
	// Get the table's columns
	
	pFirstColDef = NULL;
	pLastColDef = NULL;
	uiNumColumnDefs = 0;
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
		flmAssert( uiColumnNameLen);
		
		// Allocate a column definition
		
		if (RC_BAD( rc = m_tmpPool.poolAlloc( sizeof( F_COLUMN_DEF),
									(void **)&pColumnDef)))
		{
			goto Exit;
		}
		uiNumColumnDefs++;
		if (RC_BAD( rc = m_tmpPool.poolAlloc( uiColumnNameLen + 1,
												(void **)&pszTmp)))
		{
			goto Exit;
		}
		f_memcpy( pszTmp, szColumnName, uiColumnNameLen + 1);
		pColumnDef->pszColumnName = pszTmp;
		pColumnDef->uiColumnNameLen = uiColumnNameLen;
		pColumnDef->eColumnDataType = SFLM_UNKNOWN_TYPE;
		pColumnDef->uiFlags = COL_NULL_ALLOWED;
		pColumnDef->uiEncDefNum = 0;
		pColumnDef->uiMaxLen = 0;
		pColumnDef->pNext = NULL;
		if (pLastColDef)
		{
			pLastColDef->pNext = pColumnDef;
		}
		else
		{
			pFirstColDef = pColumnDef;
		}
		pLastColDef = pColumnDef;
		
		// Must be whitespace after the column name.

		if (RC_BAD( rc = skipWhitespace( TRUE)))
		{
			goto Exit;
		}
		
		// Data type must follow
		
		if (RC_BAD( rc = getDataType( &pColumnDef->eColumnDataType,
												&pColumnDef->uiMaxLen)))
		{
			goto Exit;
		}
		
		// Skip any white space
		
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
	
	// Create the table
	
	if (RC_BAD( rc = m_pDb->createTable( 0, szTableName, uiTableNameLen, 0,
										pFirstColDef, uiNumColumnDefs)))
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

