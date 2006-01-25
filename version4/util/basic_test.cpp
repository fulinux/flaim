//-------------------------------------------------------------------------
// Desc:	Basic unit test.
// Tabs:	3
//
//		Copyright (c) 2004-2006 Novell, Inc. All Rights Reserved.
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
// $Id: basic_test.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flmunittest.h"

FSTATIC const char * gv_pszSampleDictionary =	
	"0 @1@ field Person\n"
	" 1 type text\n"
	"0 @2@ field LastName\n"
	" 1 type text\n"
	"0 @3@ field FirstName\n"
	" 1 type text\n"
	"0 @4@ field Age\n"
	" 1 type number\n"
	"0 @5@ index LastFirst_IX\n"
	" 1 language US\n"
	" 1 key\n"
	"  2 field 2\n"
	"   3 required\n"
	"  2 field 3\n"
	"   3 required\n";

#define PERSON_TAG					1
#define LAST_NAME_TAG				2
#define FIRST_NAME_TAG				3
#define AGE_TAG						4

#ifdef FLM_NLM
	#define DB_NAME_STR					"SYS:\\SAMPLE.DB"
#else
	#define DB_NAME_STR					"sample.db"
#endif

/***************************************************************************
Desc:
****************************************************************************/
class IFlmTestImpl : public TestBase
{
public:

	inline const char * getName( void)
	{
		return( "Basic Test");
	}
	
	RCODE execute( void);
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE getTest( 
	IFlmTest **		ppTest)
{
	RCODE		rc = FERR_OK;

	if( (*ppTest = new IFlmTestImpl) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

Exit:

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::execute( void)
{
	RCODE					rc = FERR_OK;
	HFDB					hDb = HFDB_NULL;
	HFCURSOR				hCursor = HFCURSOR_NULL;
	FLMBOOL				bTransActive = FALSE;
	FLMUINT				uiDrn;
	FlmRecord *			pDefRec = NULL;
	FlmRecord *			pRec = NULL;
	void *				pvField;
	FLMBYTE				ucTmpBuf[ 64];

	// Initialize the FLAIM database engine.  This call
	// must be made once by the application prior to making any
	// other FLAIM calls

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}

	// Create or open a database.

	beginTest( 
		"Create Database Test",
		"Create a new database",
		"Self-explanatory",
		"");

Retry_Create:

	if( RC_BAD( rc = FlmDbCreate( DB_NAME_STR, NULL, 
		NULL, NULL, gv_pszSampleDictionary, NULL, &hDb)))
	{
		if( rc == FERR_FILE_EXISTS)
		{
			// Since the database already exists, we'll make a call
			// to FlmDbOpen to get a handle to it.

			if( RC_BAD( rc = FlmDbRemove( DB_NAME_STR, 
				NULL, NULL, TRUE)))
			{
				MAKE_ERROR_STRING( "FlmDbRemove failed", m_szDetails, rc);
				goto Exit;
			}
			
			goto Retry_Create;
		}
		else
		{
			MAKE_ERROR_STRING( "FlmDbCreate failed", m_szDetails, rc);
			goto Exit;
		}
	}

	endTest( "PASS");

	beginTest( 
		"Create/Populate Record Test",
		"Create a new record and populate it with data",
		"Self-explanatory",
		"");

	// Create a record object

	if( (pDefRec = new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "Could not allocate FlmRecord", m_szDetails, rc);
		goto Exit;
	}

	// Populate the record object with fields and values
	// The first field of a record will be inserted at
	// level zero (the first parameter of insertLast()
	// specifies the level number).  Subsequent fields
	// will be inserted at a non-zero level.

	if( RC_BAD( rc = pDefRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "insertLast failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->insertLast( 1, FIRST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "insertLast failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->setNative( pvField, "Foo")))
	{
		MAKE_ERROR_STRING( "setNative failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->insertLast( 1, LAST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "insertLast failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->setNative( pvField, "Bar")))
	{
		MAKE_ERROR_STRING( "setNative failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->insertLast( 1, AGE_TAG,
		FLM_NUMBER_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "insertLast failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = pDefRec->setUINT( pvField, 32)))
	{
		MAKE_ERROR_STRING( "setUINT failed", m_szDetails, rc);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "FlmDbTransBegin failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.  Initialize uiDrn to 0 so that FLAIM
	// will automatically assign a unique ID to the new record.  We could
	// also have specified a specific 32-bit ID to use for the record by
	// setting uiDrn to the desired ID value.

	uiDrn = 0;
	if( RC_BAD( rc = FlmRecordAdd( hDb, FLM_DATA_CONTAINER, 
		&uiDrn, pDefRec, 0)))
	{
		MAKE_ERROR_STRING( "FlmRecordAdd failed", m_szDetails, rc);
		goto Exit;
	}

	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( hDb)))
	{
		MAKE_ERROR_STRING( "FlmDbTransCommit failed", m_szDetails, rc);
		goto Exit;
	}
	bTransActive = FALSE;

	endTest("PASS");

	// Retrieve the record from the database by ID

	beginTest( 
		"Retrieve Record by ID Test",
		"Retrieve the record we just created by its ID",
		"Self-explanatory",
		"");

	if( RC_BAD( rc = FlmRecordRetrieve( hDb, FLM_DATA_CONTAINER, 
		uiDrn, FO_EXACT, &pRec, NULL)))
	{
		MAKE_ERROR_STRING( "FlmRecordRetrieve failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");

	// Now, build a query that retrieves the sample record.
	// First we need to initialize a cursor handle.

	beginTest( 
		"Retrieve Record by query Test",
		"Retrieve the record we just created using a query",
		"Self-explanatory",
		"");

	if( RC_BAD( rc = FlmCursorInit( hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "FlmCursorInit failed", m_szDetails, rc);
		goto Exit;
	}

	// We will search by first name and last name.  This will use the
	// LastFirst_IX defined in the sample dictionary for optimization.

	if( RC_BAD( rc = FlmCursorAddField( hCursor, LAST_NAME_TAG, 0)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddField failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddOp failed", m_szDetails, rc);
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "Bar");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddValue failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_AND_OP)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddOp failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddField( hCursor, FIRST_NAME_TAG, 0)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddField failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddOp failed", m_szDetails, rc);
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "Foo");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		MAKE_ERROR_STRING( "FlmCursorAddValue failed", m_szDetails, rc);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorFirst( hCursor, &pRec)))
	{
		MAKE_ERROR_STRING( "FlmCursorFirst failed", m_szDetails, rc);
		goto Exit;
	}

	// Free the cursor handle

	FlmCursorFree( &hCursor);
	endTest("PASS");

	// Close the database

	FlmDbClose( &hDb);

	// FlmDbRemove will delete the database and all of its files

	beginTest( 
		"Remove Database Test",
		"Remove the database",
		"Self-explanatory",
		"");

	if( RC_BAD( FlmDbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
	{
		MAKE_ERROR_STRING( "FlmDbRemove failed", m_szDetails, rc);
		goto Exit;
	}

	endTest("PASS");
	
Exit:

	if( RC_BAD( rc))
	{
		endTest("FAIL");
	}

	if( pDefRec)
	{
		pDefRec->Release();
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	if( bTransActive)
	{
		(void)FlmDbTransAbort( hDb);
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	FlmShutdown();

	if( RC_BAD( rc))
	{
		f_sprintf( (char *)ucTmpBuf, "Error %04X -- %s", (unsigned)rc, 
			(char *)FlmErrorString( rc));
		displayLine( (char *)ucTmpBuf);
	}
	
	return( rc);
}
