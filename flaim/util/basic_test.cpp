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
#define LAST_NAME_FIRST_NAME_IX	5

#ifdef FLM_NLM
	#define DB_NAME_STR					"SYS:\\SAMPLE.DB"
	#define DB_COPY_NAME_STR			"SYS:\\SAMPLECOPY.DB"
	#define DB_RENAME_NAME_STR			"SYS:\\SAMPLERENAME.DB"
	#define DB_RESTORE_NAME_STR		"SYS:\\SAMPLERESTORE.DB"
	#define BACKUP_PATH					"SYS:\\SAMPLEBACKUP"
#else
	#define DB_NAME_STR					"sample.db"
	#define DB_COPY_NAME_STR			"samplecopy.db"
	#define DB_RENAME_NAME_STR			"samplerename.db"
	#define DB_RESTORE_NAME_STR		"samplerestore.db"
	#define BACKUP_PATH					"samplebackup"
#endif

/***************************************************************************
Desc:
****************************************************************************/
class IFlmTestImpl : public TestBase
{
public:

	IFlmTestImpl()
	{
		m_hDb = HFDB_NULL;
	}
	
	virtual ~IFlmTestImpl()
	{
		if (m_hDb != HFDB_NULL)
		{
			(void)FlmDbClose( &m_hDb);
		}
	}

	inline const char * getName( void)
	{
		return( "Basic Test");
	}
	
	RCODE createDbTest( void);
	
	RCODE addRecordTest(
		FLMUINT *	puiDrn);
	
	RCODE modifyRecordTest(
		FLMUINT	uiDrn);
	
	RCODE deleteRecordTest(
		FLMUINT	uiDrn);
		
	RCODE queryRecordTest( void);
		
	RCODE keyRetrieveTest(
		FLMUINT	uiIndex,
		FLMBOOL	bLastNameFirstNameIx);
	
	RCODE addIndexTest(
		FLMUINT *	puiIndex);
		
	RCODE deleteIndexTest(
		FLMUINT	uiIndex);
		
	RCODE deleteFieldTest(
		FLMUINT	uiFieldNum);
		
	RCODE suspendIndexTest(
		FLMUINT	uiIndex);
		
	RCODE resumeIndexTest(
		FLMUINT	uiIndex);
		
	RCODE backupRestoreDbTest( void);
	
	RCODE compareRecords(
		const char *	pszDb1,
		const char *	pszDb2,
		const char *	pszWhat,
		FlmRecord *		pRecord1,
		FlmRecord *		pRecord2);
		
	RCODE compareIndexes(
		const char *	pszDb1,
		const char *	pszDb2,
		HFDB				hDb1,
		HFDB				hDb2,
		FLMUINT			uiIndexNum);
		
	RCODE compareContainers(
		const char *	pszDb1,
		const char *	pszDb2,
		HFDB				hDb1,
		HFDB				hDb2,
		FLMUINT			uiContainerNum);
		
	RCODE compareDbTest(
		const char *	pszDb1,
		const char *	pszDb2);
		
	RCODE copyDbTest(
		const char *	pszDestDbName,
		const char *	pszSrcDbName);
		
	RCODE renameDbTest(
		const char *	pszDestDbName,
		const char *	pszSrcDbName);
		
	RCODE removeDbTest(
		const char *	pszDbName);
		
	RCODE execute( void);
	
private:

	HFDB	m_hDb;
};

const char * gv_pszFamilyNames[] =
{
	"Walton",
	"Abernathy",
	"Stillwell",
	"Anderson",
	"Armstrong",
	"Adamson",
	"Bagwell",
	"Ballard",
	"Bennett",
	"Blackman",
	"Bottoms",
	"Bradley",
	"Butterfield",
	"Cavanagh",
	"Chadwick",
	"Clark",
	"Crabtree",
	"Cunningham",
	"Darnell",
	"McClintock",
	"Davidson",
	"Dingman",
	"Doyle",
	"Eastman",
	"Ballantine",
	"Edmunds",
	"Neil",
	"Erickson",
	"Fetterman",
	"Finn",
	"Flanagan",
	"Gerber",
	"Thedford",
	"Thorman",
	"Gibson",
	"Gruszczynski",
	"Haaksman",
	"Hathaway",
	"Pernell",
	"Phillips",
	"Highsmith",
	"Hollingworth",
	"Frankenberger",
	"Hutchison",
	"Irving",
	"Weatherspoon",
	"Itaya",
	"Janiszewski",
	"Jenkins",
	"Jung",
	"Keller",
	"Jackson",
	"Kingsbury",
	"Klostermann",
	"Langley",
	"Liddle",
	"Lockhart",
	"Ludwig",
	"Kristjanson",
	"MacCormack",
	"Richards",
	"Robbins",
	"McAuliffe",
	"Merryweather",
	"Moynihan",
	"Muller",
	"Newland",
	"OCarroll",
	"Okuzawa",
	"Ortiz",
	"Pachulski",
	"Parmaksezian",
	"Peacocke",
	"Poole",
	"Prewitt",
	"Quigley",
	"Qureshi",
	"Ratcliffe",
	"Rundle",
	"Ryder",
	"Sampson",
	"Satterfield",
	"Sharkey",
	"Silverman",
	"Snedeker",
	"Goodman",
	"Spitzer",
	"Szypulski",
	"Talbott",
	"Trisko",
	"Turrubiarte",
	"Upchurch",
	"Valdez",
	"Vandenheede",
	"Volker",
	"Wilke",
	"Wojciechowski",
	"Wyndham",
	"Yamashita",
	"York",
	"Zahn",
	"Zimmermann",
	NULL
};

const char * gv_pszGivenNames[] =
{
	"Robby",
	"Agatha",
	"Anatoli",
	"Zsazsa",
	"Arlen",
	"Augusta",
	"Bambi",
	"Bee",
	"Bennie",
	"Bonni",
	"Brennan",
	"Bryon",
	"Cal",
	"Caroline",
	"Charlotte",
	"Cristine",
	"Danny",
	"Dean",
	"Desdemona",
	"Dixie",
	"Doug",
	"Ellie",
	"Zelma",
	"Elsie",
	"Ursula",
	"Ernest",
	"Fanny",
	"Francis",
	"Gailya",
	"Gertrude",
	"Gloria",
	"Greg",
	"Harriot",
	"Hennrietta",
	"Howard",
	"Ian",
	"Sherwood",
	"Xavier",
	"Ira",
	"Jacklyn",
	"Jeff",
	"Philippe",
	"Vivianne",
	"Jeremy",
	"Wendie",
	"Abbie",
	"Johnny",
	"Kerrie",
	"Lacey",
	"Lilly",
	"Lucas",
	"Magdalena",
	"Maryanne",
	"Matt",
	"Dorelle",
	"Myron",
	"Netty",
	"Nicolette",
	"Octavio",
	"Oliver",
	"Paige",
	"Parker",
	"Patti",
	"Merv",
	"Preston",
	"Quinn",
	"Randall",
	"Jean",
	"Rebekah",
	"Ricardo",
	"Rose",
	"Russell",
	"Scarlet",
	"Shannon",
	"Larry",
	"Sophie",
	"Stephen",
	"Susette",
	"Christina",
	"Ted",
	"Enrico",
	"Theresa",
	"Timothy",
	"Tony",
	"Vanna",
	"Kalli",
	"Vern",
	"Alicia",
	"Wallace",
	"Yogi",
	"Aaron",
	"Yuji",
	"Zack",
	NULL
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
RCODE IFlmTestImpl::createDbTest( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	beginTest( "Create Database Test");

	for (;;)
	{
		if( RC_BAD( rc = FlmDbCreate( DB_NAME_STR, NULL, 
			NULL, NULL, gv_pszSampleDictionary, NULL, &m_hDb)))
		{
			if( rc == FERR_FILE_EXISTS)
			{
				// Since the database already exists, we'll make a call
				// to FlmDbOpen to get a handle to it.
	
				if( RC_BAD( rc = FlmDbRemove( DB_NAME_STR, 
					NULL, NULL, TRUE)))
				{
					MAKE_ERROR_STRING( "calling FlmDbRemove", rc, m_szFailInfo);
					goto Exit;
				}
			}
			else
			{
				MAKE_ERROR_STRING( "calling FlmDbCreate", rc, m_szFailInfo);
				goto Exit;
			}
		}
		else
		{
			break;
		}
	}

	bPassed = TRUE;
	
Exit:

	endTest( bPassed);

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addRecordTest(
	FLMUINT *	puiDrn
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	FlmRecord *		pCopyRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;
	FLMUINT			uiLoop;
	FLMUINT			uiLoop2;
	FLMUINT			uiDrn2;

	beginTest( "FlmRecordAdd Test");

	// Create a record object

	if( (pRec = new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// Populate the record object with fields and values
	// The first field of a record will be inserted at
	// level zero (the first parameter of insertLast()
	// specifies the level number).  Subsequent fields
	// will be inserted at a non-zero level.

	if( RC_BAD( rc = pRec->insertLast( 0, PERSON_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 1, FIRST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setNative( pvField, "Foo")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 1, LAST_NAME_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setNative( pvField, "Bar")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->insertLast( 1, AGE_TAG,
		FLM_NUMBER_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setUINT( pvField, 32)))
	{
		MAKE_ERROR_STRING( "calling setUINT", rc, m_szFailInfo);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	*puiDrn = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
		puiDrn, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	for (uiLoop = 0; gv_pszFamilyNames [uiLoop]; uiLoop++)
	{
		for (uiLoop2 = 0; gv_pszGivenNames [uiLoop2]; uiLoop2++)
		{
			if ((pCopyRec = pRec->copy()) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
				goto Exit;
			}
			if ((pvField = pCopyRec->find( pCopyRec->root(), FIRST_NAME_TAG)) == NULL)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				MAKE_ERROR_STRING( "corruption calling FlmRecord->copy()", rc, m_szFailInfo);
				goto Exit;
			}
			if( RC_BAD( rc = pCopyRec->setNative( pvField, gv_pszGivenNames [uiLoop2])))
			{
				MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
				goto Exit;
			}
			if ((pvField = pCopyRec->find( pCopyRec->root(), LAST_NAME_TAG)) == NULL)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				MAKE_ERROR_STRING( "corruption calling FlmRecord->copy()", rc, m_szFailInfo);
				goto Exit;
			}
			if( RC_BAD( rc = pCopyRec->setNative( pvField, gv_pszFamilyNames [uiLoop])))
			{
				MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
				goto Exit;
			}
			uiDrn2 = 0;
			if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DATA_CONTAINER, 
				&uiDrn2, pCopyRec, 0)))
			{
				MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
				goto Exit;
			}
			pCopyRec->Release();
			pCopyRec = NULL;
		}
	}

	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( pCopyRec)
	{
		pCopyRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::modifyRecordTest(
	FLMUINT	uiDrn
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	FlmRecord *		pModRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;

	// Retrieve the record from the database by ID

	beginTest( "FlmRecordRetrieve Test");
	if( RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DATA_CONTAINER, 
		uiDrn, FO_EXACT, &pRec, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	endTest( TRUE);

	
	beginTest( "FlmRecordModify Test");

	// Copy the record so we can modify it

	if( (pModRec = pRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
		goto Exit;
	}

	// Find the first name field and change it.

	pvField = pModRec->find( pModRec->root(), FIRST_NAME_TAG);
	if( RC_BAD( rc = pModRec->setNative( pvField, "FooFoo")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DATA_CONTAINER, 
		uiDrn, pModRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}

	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	if( pModRec)
	{
		pModRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::deleteRecordTest(
	FLMUINT	uiDrn
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	// Delete a record from the database

	beginTest( "FlmRecordDelete Test");
	if( RC_BAD( rc = FlmRecordDelete( m_hDb, FLM_DATA_CONTAINER, 
		uiDrn, FLM_AUTO_TRANS | 15)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;

Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::queryRecordTest( void)
{
	RCODE			rc = FERR_OK;
	FlmRecord *	pRec = NULL;
	HFCURSOR		hCursor = HFCURSOR_NULL;
	FLMBYTE		ucTmpBuf[ 64];
	FLMBOOL		bPassed = FALSE;
	
	// Now, build a query that retrieves the sample record.
	// First we need to initialize a cursor handle.

	beginTest( "Retrieve Record by query Test");

	if( RC_BAD( rc = FlmCursorInit( m_hDb, FLM_DATA_CONTAINER, &hCursor)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorInit", rc, m_szFailInfo);
		goto Exit;
	}

	// We will search by first name and last name.  This will use the
	// LastFirst_IX defined in the sample dictionary for optimization.

	if( RC_BAD( rc = FlmCursorAddField( hCursor, LAST_NAME_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "Bar");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_AND_OP)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp failed", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddField( hCursor, FIRST_NAME_TAG, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddField", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorAddOp( hCursor, FLM_EQ_OP)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddOp", rc, m_szFailInfo);
		goto Exit;
	}

	f_sprintf( (char *)ucTmpBuf, "FooFoo");
	if( RC_BAD( rc = FlmCursorAddValue( hCursor, FLM_STRING_VAL, 
		ucTmpBuf, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorAddValue", rc, m_szFailInfo);
		goto Exit;
	}

	if( RC_BAD( rc = FlmCursorFirst( hCursor, &pRec)))
	{
		MAKE_ERROR_STRING( "calling FlmCursorFirst", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	if (hCursor != HFCURSOR_NULL)
	{
		FlmCursorFree( &hCursor);
	}

	if( pRec)
	{
		pRec->Release();
	}

	endTest( bPassed);

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::keyRetrieveTest(
	FLMUINT	uiIndex,
	FLMBOOL	bLastNameFirstNameIx)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bPassed = FALSE;
	FLMUINT		uiFlags = FO_FIRST;
	FlmRecord *	pSearchKey = NULL;
	FLMUINT		uiSearchDrn = 0;
	FlmRecord *	pFoundKey = NULL;
	FLMUINT		uiFoundDrn = 0;
	char			szLastFirstName [100];
	char			szLastLastName [100];
	char			szCurrFirstName [100];
	char			szCurrLastName [100];
	void *		pvField;
	FLMUINT		uiLen;
	FLMINT		iLastCmp;
	FLMINT		iFirstCmp;

	if (bLastNameFirstNameIx)
	{
		beginTest( "FlmKeyRetrieve Test (Last+FirstIx)");
	}
	else
	{
		beginTest( "FlmKeyRetrieve Test (First+LastIx)");
	}
	szLastFirstName [0] = 0;
	szLastLastName [0] = 0;
	for (;;)
	{
		if (RC_BAD( rc = FlmKeyRetrieve( m_hDb, uiIndex,
								0, pSearchKey, uiSearchDrn, uiFlags,
								&pFoundKey, &uiFoundDrn)))
		{
			if (rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
				break;
			}
			else
			{
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Make sure this key is greater than the last key.
		
		if ((pvField = pFoundKey->find( pFoundKey->root(), LAST_NAME_TAG)) == NULL)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			MAKE_ERROR_STRING( "corruption calling FlmRecord->find()", rc, m_szFailInfo);
			goto Exit;
		}
		uiLen = sizeof( szCurrLastName);
		if (RC_BAD( rc = pFoundKey->getNative( pvField, szCurrLastName, &uiLen)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getNative()", rc, m_szFailInfo);
			goto Exit;
		}
		if ((pvField = pFoundKey->find( pFoundKey->root(), FIRST_NAME_TAG)) == NULL)
		{
			rc = RC_SET( FERR_DATA_ERROR);
			MAKE_ERROR_STRING( "corruption calling FlmRecord->find()", rc, m_szFailInfo);
			goto Exit;
		}
		uiLen = sizeof( szCurrFirstName);
		if (RC_BAD( rc = pFoundKey->getNative( pvField, szCurrFirstName, &uiLen)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getNative()", rc, m_szFailInfo);
			goto Exit;
		}

		iLastCmp = f_strcmp( szCurrLastName, szLastLastName);
		iFirstCmp = f_strcmp( szCurrFirstName, szLastFirstName);
		
		if (bLastNameFirstNameIx)
		{
			if (iLastCmp < 0)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				f_sprintf( m_szFailInfo, "Invalid last name order in index: "
					" %s before %s", szLastLastName, szCurrLastName);
				goto Exit;
			}
			else if (iLastCmp == 0)
			{
				if (iFirstCmp < 0)
				{
					rc = RC_SET( FERR_DATA_ERROR);
					f_sprintf( m_szFailInfo, "Invalid first name order in index: "
						" %s before %s", szLastFirstName, szCurrFirstName);
					goto Exit;
				}
			}
		}
		else
		{
			if (iFirstCmp < 0)
			{
				rc = RC_SET( FERR_DATA_ERROR);
				f_sprintf( m_szFailInfo, "Invalid first name order in index: "
					" %s before %s", szLastFirstName, szCurrFirstName);
				goto Exit;
			}
			else if (iFirstCmp == 0)
			{
				if (iLastCmp < 0)
				{
					rc = RC_SET( FERR_DATA_ERROR);
					f_sprintf( m_szFailInfo, "Invalid last name order in index: "
						" %s before %s", szLastLastName, szCurrLastName);
					goto Exit;
				}
			}
		}
		
		// Setup to get the next key.
		
		uiFlags = FO_EXCL;
		uiSearchDrn = uiFoundDrn;
		if (pSearchKey)
		{
			pSearchKey->Release();
		}
		pSearchKey = pFoundKey;
		pFoundKey = NULL;
		uiFoundDrn = 0;
		f_strcpy( szLastLastName, szCurrLastName);
		f_strcpy( szLastFirstName, szCurrFirstName);
	}
	bPassed = TRUE;

Exit:

	if (pSearchKey)
	{
		pSearchKey->Release();
	}
	if (pFoundKey)
	{
		pFoundKey->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::addIndexTest(
	FLMUINT *	puiIndex
	)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRec = NULL;
	void *			pvField;
	FLMBOOL			bTransActive = FALSE;
	FLMBOOL			bPassed = FALSE;
	char				szFieldNum [20];

	beginTest( "Add FirstName+LastName Index Test");

	// Create a record object

	if( (pRec = new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "allocating FlmRecord", rc, m_szFailInfo);
		goto Exit;
	}

	// 0 index FirstLast_IX
	
	if( RC_BAD( rc = pRec->insertLast( 0, FLM_INDEX_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "FirstLast_IX")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 language US

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_LANGUAGE_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = pRec->setNative( pvField, "US")))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}

	// 1 key

	if( RC_BAD( rc = pRec->insertLast( 1, FLM_KEY_TAG,
		FLM_CONTEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <FIRST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", FIRST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required

	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}

	// 2 field <LAST_NAME_TAG>

	if( RC_BAD( rc = pRec->insertLast( 2, FLM_FIELD_TAG,
		FLM_TEXT_TYPE, &pvField)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	f_sprintf( szFieldNum, "%u", LAST_NAME_TAG);
	if( RC_BAD( rc = pRec->setNative( pvField, szFieldNum)))
	{
		MAKE_ERROR_STRING( "calling setNative", rc, m_szFailInfo);
		goto Exit;
	}
	
	// 3 required
	
	if( RC_BAD( rc = pRec->insertLast( 3, FLM_REQUIRED_TAG,
		FLM_TEXT_TYPE, NULL)))
	{
		MAKE_ERROR_STRING( "calling insertLast", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Start an update transaction

	if( RC_BAD( rc = FlmDbTransBegin( m_hDb, FLM_UPDATE_TRANS, 15)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransBegin", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = TRUE;

	// Add the record to the database.

	*puiIndex = 0;
	if( RC_BAD( rc = FlmRecordAdd( m_hDb, FLM_DICT_CONTAINER, 
		puiIndex, pRec, 0)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordAdd", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Commit the transaction
	// If FlmDbTransCommit returns without an error, the changes made
	// above will be durable even if the system crashes.

	if( RC_BAD( rc = FlmDbTransCommit( m_hDb)))
	{
		MAKE_ERROR_STRING( "calling FlmDbTransCommit", rc, m_szFailInfo);
		goto Exit;
	}
	bTransActive = FALSE;

	bPassed = TRUE;
	
Exit:

	if( bTransActive)
	{
		(void)FlmDbTransAbort( m_hDb);
	}

	if( pRec)
	{
		pRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::deleteIndexTest(
	FLMUINT	uiIndex
	)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	beginTest( "Delete Index Test");

	// Delete the record from the dictionary.

	if( RC_BAD( rc = FlmRecordDelete( m_hDb, FLM_DICT_CONTAINER, uiIndex,
								 FLM_AUTO_TRANS | 15))) 
	{
		MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::deleteFieldTest(
	FLMUINT	uiFieldNum
	)
{
	RCODE			rc = FERR_OK;
	FlmRecord *	pDictRec = NULL;
	FlmRecord *	pNewRec = NULL;
	void *		pvField;
	FLMUINT		uiDrn;
	FLMBOOL		bPassed = FALSE;

	beginTest( "Delete Field Definition Test");

	// Delete the record from the dictionary.  This attempt should fail
	// because it is not properly marked.

	if( RC_BAD( rc = FlmRecordDelete( m_hDb, FLM_DICT_CONTAINER, uiFieldNum,
								 FLM_AUTO_TRANS | 15))) 
	{
		if (rc != FERR_CANNOT_DEL_ITEM)
		{
			MAKE_ERROR_STRING( "calling FlmRecordDelete", rc, m_szFailInfo);
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Should not be able to delete field %u!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Retrieve the field definition record.
	
	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DICT_CONTAINER,
								uiFieldNum, FO_EXACT, &pDictRec, &uiDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	
	// If it is not a field definition, we have the wrong definition record.
	
	if (pDictRec->getFieldID( pDictRec->root()) != FLM_FIELD_TAG)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Dictionary record %u, is not a field definition!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Make a copy of the dictionary record
	
	if ((pNewRec = pDictRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
		goto Exit;
	}
	
	// See if there is a state field.  If not add it.
	
	if ((pvField = pNewRec->find( pNewRec->root(), FLM_STATE_TAG)) == NULL)
	{
		if (RC_BAD( rc = pNewRec->insert( pNewRec->root(), INSERT_LAST_CHILD,
									FLM_STATE_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->insert()", rc, m_szFailInfo);
			goto Exit;
		}
	}
	
	// Attempt to set the state field on the record to "unused", this should
	// fail.
	
	if (RC_BAD( rc = pNewRec->setNative( pvField, "unused")))
	{
		MAKE_ERROR_STRING( "calling FlmRecord->setNative()", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DICT_CONTAINER, 
		uiFieldNum, pNewRec, FLM_AUTO_TRANS | 15)))
	{
		if (rc != FERR_CANNOT_MOD_FIELD_STATE)
		{
			MAKE_ERROR_STRING( "calling FlmRecordModify", rc, m_szFailInfo);
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Should not be able to set field %'s state to unused!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Set the state field on the record to "check", then run
	// FlmDbSweep.  The sweep should not set the field state
	// to unused.
	
	if (RC_BAD( rc = pNewRec->setNative( pvField, "check")))
	{
		MAKE_ERROR_STRING( "calling FlmRecord->setNative()", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DICT_CONTAINER, 
		uiFieldNum, pNewRec, FLM_AUTO_TRANS | 15)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordModify", rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = FlmDbSweep( m_hDb, SWEEP_CHECKING_FLDS, EACH_CHANGE, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbSweep", rc, m_szFailInfo);
		goto Exit;
	}
	if (pNewRec)
	{
		pNewRec->Release();
		pNewRec = NULL;
	}
	if (pDictRec)
	{
		pDictRec->Release();
		pDictRec = NULL;
	}
	
	// Retrieve the record again and make sure the state flag is not set
	// to unused.

	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DICT_CONTAINER,
								uiFieldNum, FO_EXACT, &pDictRec, &uiDrn)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
		goto Exit;
	}
	
	// If it is not a field definition, we have the wrong definition record.
	
	if (pDictRec->getFieldID( pDictRec->root()) != FLM_FIELD_TAG)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Dictionary record %u, is not a field definition!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	// Make a copy of the dictionary record
	
	if ((pNewRec = pDictRec->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		MAKE_ERROR_STRING( "calling FlmRecord->copy()", rc, m_szFailInfo);
		goto Exit;
	}
	
	// See if there is a state field.  If not add it.
	
	if ((pvField = pNewRec->find( pNewRec->root(), FLM_STATE_TAG)) == NULL)
	{
		if (RC_BAD( rc = pNewRec->insert( pNewRec->root(), INSERT_LAST_CHILD,
									FLM_STATE_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->insert()", rc, m_szFailInfo);
			goto Exit;
		}
	}
	else
	{
		char		szState [20];
		FLMUINT	uiLen = sizeof( szState);
		
		// State should be active if it is present.
		
		if (RC_BAD( rc = pNewRec->getNative( pvField, szState, &uiLen)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getNative()", rc, m_szFailInfo);
			goto Exit;
		}
		if (f_strnicmp( szState, "acti", 4) != 0)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Dictionary record %u's state should be active!",
					(unsigned)uiFieldNum);
			goto Exit;
		}
	}
	
	// Attempt to set the state field on the record to "purge", this should
	// succeed, and FlmDbSweep should get rid of the definition.
	
	if (RC_BAD( rc = pNewRec->setNative( pvField, "purge")))
	{
		MAKE_ERROR_STRING( "calling FlmRecord->setNative()", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = FlmRecordModify( m_hDb, FLM_DICT_CONTAINER, 
		uiFieldNum, pNewRec, FLM_AUTO_TRANS | 15)))
	{
		MAKE_ERROR_STRING( "calling FlmRecordModify", rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = FlmDbSweep( m_hDb, SWEEP_PURGED_FLDS, EACH_CHANGE, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbSweep", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Make sure the dictionary definition is gone now.

	if (pDictRec)
	{
		pDictRec->Release();
		pDictRec = NULL;
	}
	if (RC_BAD( rc = FlmRecordRetrieve( m_hDb, FLM_DICT_CONTAINER,
								uiFieldNum, FO_EXACT, &pDictRec, &uiDrn)))
	{
		if (rc != FERR_NOT_FOUND)
		{
			MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc, m_szFailInfo);
			goto Exit;
		}
		else
		{
			rc = FERR_OK;
		}
	}
	else
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "Dictionary record %u should have been purged by FlmDbSweep!",
				(unsigned)uiFieldNum);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	if (pDictRec)
	{
		pDictRec->Release();
	}
	if (pNewRec)
	{
		pNewRec->Release();
	}

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::suspendIndexTest(
	FLMUINT	uiIndex
	)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bPassed = FALSE;
	FINDEX_STATUS	indexStatus;

	beginTest( "Suspend Index Test");

	// Delete the record from the dictionary.

	if( RC_BAD( rc = FlmIndexSuspend( m_hDb, uiIndex)))
	{
		MAKE_ERROR_STRING( "calling FlmIndexSuspend", rc, m_szFailInfo);
		goto Exit;
	}
	
	// See if the index is actually suspended.
	
	if( RC_BAD( rc = FlmIndexStatus( m_hDb, uiIndex, &indexStatus)))
	{
		MAKE_ERROR_STRING( "calling FlmIndexStatus", rc, m_szFailInfo);
		goto Exit;
	}
	
	if (!indexStatus.bSuspended)
	{
		rc = RC_SET( FERR_FAILURE);
		f_sprintf( m_szFailInfo, "FlmIndexSuspend failed to suspend index %u",
			(unsigned)uiIndex);
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::resumeIndexTest(
	FLMUINT	uiIndex)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bPassed = FALSE;
	FINDEX_STATUS	indexStatus;

	beginTest( "Resume Index Test");

	// Delete the record from the dictionary.

	if (RC_BAD( rc = FlmIndexResume( m_hDb, uiIndex)))
	{
		MAKE_ERROR_STRING( "calling FlmIndexResume", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Wait for the index to come on-line
	
	for (;;)
	{
	
		// See if the index is actually resumed.
		
		if( RC_BAD( rc = FlmIndexStatus( m_hDb, uiIndex, &indexStatus)))
		{
			MAKE_ERROR_STRING( "calling FlmIndexStatus", rc, m_szFailInfo);
			goto Exit;
		}
		
		if (indexStatus.bSuspended)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "FlmIndexResume failed to resume index %u",
				(unsigned)uiIndex);
			goto Exit;
		}
		
		if (indexStatus.uiLastRecordIdIndexed == RECID_UNDEFINED)
		{
			break;
		}
	}
	
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::backupRestoreDbTest( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	HFBACKUP	hBackup = HFBACKUP_NULL;
	char		szTest [200];
	
	// Do the backup
	
	beginTest( "Backup Test");
	if (RC_BAD( rc = FlmDbBackupBegin( m_hDb, FLM_FULL_BACKUP, TRUE, &hBackup)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackupBegin", rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = FlmDbBackup( hBackup, BACKUP_PATH, NULL, NULL, NULL, NULL,
								NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackup", rc, m_szFailInfo);
		goto Exit;
	}
	if (RC_BAD( rc = FlmDbBackupEnd( &hBackup)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackupEnd", rc, m_szFailInfo);
		goto Exit;
	}
	endTest( TRUE);
	
	// Do the restore
	
	f_sprintf( szTest, "Restore Backup To %s Test", DB_RESTORE_NAME_STR);
	beginTest( szTest);
	if (RC_BAD( rc = FlmDbRestore( DB_RESTORE_NAME_STR, NULL, BACKUP_PATH,
								NULL, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbBackupEnd", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	if (hBackup != HFBACKUP_NULL)
	{
		(void)FlmDbBackupEnd( &hBackup);
	}

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareRecords(
	const char *	pszDb1,
	const char *	pszDb2,
	const char *	pszWhat,
	FlmRecord *		pRecord1,
	FlmRecord *		pRecord2)
{
	RCODE		rc = FERR_OK;
	void *	pvField1 = pRecord1->root();
	void *	pvField2 = pRecord2->root();
	FLMUINT	uiFieldNum1;
	FLMUINT	uiFieldNum2;
	FLMUINT	uiLevel1;
	FLMUINT	uiLevel2;
	FLMUINT	uiDataType1;
	FLMUINT	uiDataType2;
	FLMUINT	uiDataLength1;
	FLMUINT	uiDataLength2;
	FLMUINT	uiEncLength1;
	FLMUINT	uiEncLength2;
	FLMUINT	uiEncId1;
	FLMUINT	uiEncId2;
	
	for (;;)
	{
		if (!pvField1)
		{
			if (pvField2)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "%s in %s has more fields than in %s",
					pszWhat, pszDb2, pszDb1);
				goto Exit;
			}
			else
			{
				break;
			}
		}
		else if (!pvField2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "%s in %s has more fields than in %s",
				pszWhat, pszDb1, pszDb2);
			goto Exit;
		}
		
		// Compare the field number, data type, etc.
		
		if (RC_BAD( rc = pRecord1->getFieldInfo( pvField1, &uiFieldNum1,
												&uiLevel1, &uiDataType1, &uiDataLength1,
												&uiEncLength1, &uiEncId1)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getFieldInfo", rc, m_szFailInfo);
			goto Exit;
		}
		if (RC_BAD( rc = pRecord2->getFieldInfo( pvField2, &uiFieldNum2,
												&uiLevel2, &uiDataType2, &uiDataLength2,
												&uiEncLength2, &uiEncId2)))
		{
			MAKE_ERROR_STRING( "calling FlmRecord->getFieldInfo", rc, m_szFailInfo);
			goto Exit;
		}
		
		if (uiFieldNum1 != uiFieldNum2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Num mismatch in %s, %s: %u, %s: %u",
				pszWhat,
				pszDb1, (unsigned)uiFieldNum1,
				pszDb2, (unsigned)uiFieldNum2);
			goto Exit;
		}
		if (uiLevel1 != uiLevel2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Level mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiLevel1,
				pszDb2, (unsigned)uiLevel2);
			goto Exit;
		}
		if (uiDataType1 != uiDataType2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Type mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiDataType1,
				pszDb2, (unsigned)uiDataType2);
			goto Exit;
		}
		if (uiDataLength1 != uiDataLength2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Length mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiDataLength1,
				pszDb2, (unsigned)uiDataLength2);
			goto Exit;
		}
		if (uiEncLength1 != uiEncLength2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Enc. Length mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiEncLength1,
				pszDb2, (unsigned)uiEncLength2);
			goto Exit;
		}
		if (uiEncId1 != uiEncId2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Field Enc. Id mismatch in %s, Fld: %u, %s: %u, %s: %u",
				pszWhat, (unsigned)uiFieldNum1,
				pszDb1, (unsigned)uiEncId1,
				pszDb2, (unsigned)uiEncId2);
			goto Exit;
		}
		
		// Compare the data
		
		if (uiDataLength1)
		{
			const FLMBYTE *	pucData1 = pRecord1->getDataPtr( pvField1);
			const FLMBYTE *	pucData2 = pRecord2->getDataPtr( pvField2);
			
			if (f_memcmp( pucData1, pucData2, uiDataLength1) != 0)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "Field Value mismatch in %s, Fld: %u",
					pszWhat, (unsigned)uiFieldNum1);
				goto Exit;
			}
		}
		
		// Go to the next field in each key.
		
		pvField1 = pRecord1->next( pvField1);
		pvField2 = pRecord2->next( pvField2);
	}
	
Exit:

	return( rc);
}
		
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareIndexes(
	const char *	pszDb1,
	const char *	pszDb2,
	HFDB				hDb1,
	HFDB				hDb2,
	FLMUINT			uiIndexNum)
{
	RCODE			rc = FERR_OK;
	RCODE			rc1;
	RCODE			rc2;
	FLMUINT		uiFlags1;
	FLMUINT		uiFlags2;
	FlmRecord *	pSearchKey1 = NULL;
	FLMUINT		uiSearchDrn1 = 0;
	FlmRecord *	pSearchKey2 = NULL;
	FLMUINT		uiSearchDrn2 = 0;
	FlmRecord *	pFoundKey1 = NULL;
	FLMUINT		uiFoundDrn1 = 0;
	FlmRecord *	pFoundKey2 = NULL;
	FLMUINT		uiFoundDrn2 = 0;
	char			szWhat [40];
	FLMUINT		uiCount = 0;
	
	// Read through all keys and references in the index.  Make sure they
	// are identical.

	uiFlags1 = FO_FIRST;
	uiFlags2 = FO_FIRST;
	for (;;)
	{
		rc1 = FlmKeyRetrieve( hDb1, uiIndexNum,
								0, pSearchKey1, uiSearchDrn1, uiFlags1,
								&pFoundKey1, &uiFoundDrn1);
		rc2 = FlmKeyRetrieve( hDb2, uiIndexNum,
								0, pSearchKey2, uiSearchDrn2, uiFlags2,
								&pFoundKey2, &uiFoundDrn2);
		if (RC_BAD( rc1))
		{
			if (rc1 == FERR_EOF_HIT)
			{
				if (RC_OK( rc2))
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, "%s has more keys/refs in index %u than %2",
							pszDb2, (unsigned)uiIndexNum, pszDb1);
					goto Exit;
				}
				else if (rc2 == FERR_EOF_HIT)
				{
					break;
				}
				else
				{
					rc = rc2;
					MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc2, m_szFailInfo);
					goto Exit;
				}
			}
			else
			{
				rc = rc1;
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc1, m_szFailInfo);
				goto Exit;
			}
		}
		else
		{
			if (rc2 == FERR_EOF_HIT)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "%s has more keys/refs in index %u than %2",
						pszDb1, (unsigned)uiIndexNum, pszDb2);
				goto Exit;
			}
			else if (RC_BAD( rc2))
			{
				rc = rc2;
				MAKE_ERROR_STRING( "calling FlmKeyRetrieve", rc2, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Compare the two keys.
		
		uiCount++;
		f_sprintf( szWhat, "Ix #%u, Key #%u", (unsigned)uiIndexNum, (unsigned)uiCount);
		if (RC_BAD( rc = compareRecords( pszDb1, pszDb2, szWhat,
									pFoundKey1, pFoundKey2)))
		{
			goto Exit;
		}
		
		// Compare the references.
		
		if (uiFoundDrn1 != uiFoundDrn2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "Ref DRN mismatch in %s, %s: %u, %s: %u",
					szWhat, (unsigned)uiIndexNum,
					pszDb1, (unsigned)uiFoundDrn1, pszDb2, (unsigned)uiFoundDrn2);
			goto Exit;
		}
	
		// Setup to get the next key.
		
		uiFlags1 = FO_EXCL;
		uiSearchDrn1 = uiFoundDrn1;
		if (pSearchKey1)
		{
			pSearchKey1->Release();
		}
		pSearchKey1 = pFoundKey1;
		pFoundKey1 = NULL;
		uiFoundDrn1 = 0;
		
		uiFlags2 = FO_EXCL;
		uiSearchDrn2 = uiFoundDrn2;
		if (pSearchKey2)
		{
			pSearchKey2->Release();
		}
		pSearchKey2 = pFoundKey2;
		pFoundKey2 = NULL;
		uiFoundDrn2 = 0;
	}
	
Exit:

	if (pSearchKey1)
	{
		pSearchKey1->Release();
	}
	if (pSearchKey2)
	{
		pSearchKey2->Release();
	}
	if (pFoundKey1)
	{
		pFoundKey1->Release();
	}
	if (pFoundKey2)
	{
		pFoundKey2->Release();
	}

	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareContainers(
	const char *	pszDb1,
	const char *	pszDb2,
	HFDB				hDb1,
	HFDB				hDb2,
	FLMUINT			uiContainerNum)
{
	RCODE			rc = FERR_OK;
	RCODE			rc1;
	RCODE			rc2;
	FlmRecord *	pRecord1 = NULL;
	FLMUINT		uiDrn1;
	FlmRecord *	pRecord2 = NULL;
	FLMUINT		uiDrn2;
	char			szWhat [40];
	
	// Read through all records in the container.  Make sure they
	// are identical.

	uiDrn1 = 1;
	uiDrn2 = 1;
	for (;;)
	{
		rc1 = FlmRecordRetrieve( hDb1, uiContainerNum, uiDrn1,
						FO_INCL, &pRecord1, &uiDrn1);
		rc2 = FlmRecordRetrieve( hDb2, uiContainerNum, uiDrn2,
						FO_INCL, &pRecord2, &uiDrn2);
		if (RC_BAD( rc1))
		{
			if (rc1 == FERR_EOF_HIT)
			{
				if (RC_OK( rc2))
				{
					rc = RC_SET( FERR_FAILURE);
					f_sprintf( m_szFailInfo, "%s has more records in container %u than %s",
							pszDb2, (unsigned)uiContainerNum, pszDb1);
					goto Exit;
				}
				else if (rc2 == FERR_EOF_HIT)
				{
					break;
				}
				else
				{
					rc = rc2;
					MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc2, m_szFailInfo);
					goto Exit;
				}
			}
			else
			{
				rc = rc1;
				MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc1, m_szFailInfo);
				goto Exit;
			}
		}
		else
		{
			if (rc2 == FERR_EOF_HIT)
			{
				rc = RC_SET( FERR_FAILURE);
				f_sprintf( m_szFailInfo, "%s has more records in container %u than %s",
						pszDb1, (unsigned)uiContainerNum, pszDb2);
				goto Exit;
			}
			else if (RC_BAD( rc2))
			{
				rc = rc2;
				MAKE_ERROR_STRING( "calling FlmRecordRetrieve", rc2, m_szFailInfo);
				goto Exit;
			}
		}
		
		// Make sure these records have the same DRN
		
		if (uiDrn1 != uiDrn2)
		{
			rc = RC_SET( FERR_FAILURE);
			f_sprintf( m_szFailInfo, "DRN mismatch in container %u, %s: %u, %s: %u",
					(unsigned)uiContainerNum,
					pszDb1, (unsigned)uiDrn1, pszDb2, (unsigned)uiDrn2);
			goto Exit;
		}
		
		// Compare the two records.
		
		f_sprintf( szWhat, "Cont #%u, Rec #%u", (unsigned)uiContainerNum,
				(unsigned)uiDrn1);
		if (RC_BAD( rc = compareRecords( pszDb1, pszDb2, szWhat,
									pRecord1, pRecord2)))
		{
			goto Exit;
		}
		
		// If we are doing the dictionary container, we will jump out
		// to check any containers or indexes that it defines.

		if (uiContainerNum == FLM_DICT_CONTAINER)
		{
			if (pRecord1->getFieldID( pRecord1->root()) == FLM_CONTAINER_TAG)
			{
				if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, uiDrn1)))
				{
					goto Exit;
				}
			}
			else if (pRecord1->getFieldID( pRecord1->root()) == FLM_INDEX_TAG)
			{
				if (RC_BAD( rc = compareIndexes( pszDb1, pszDb2, hDb1, hDb2, uiDrn1)))
				{
					goto Exit;
				}
			}
		}

		uiDrn1++;
		uiDrn2++;
	}
	
Exit:

	if (pRecord1)
	{
		pRecord1->Release();
	}
	if (pRecord2)
	{
		pRecord2->Release();
	}
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::compareDbTest(
	const char *	pszDb1,
	const char *	pszDb2)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bPassed = FALSE;
	char			szTest [200];
	HFDB			hDb1 = HFDB_NULL;
	HFDB			hDb2 = HFDB_NULL;
	
	f_sprintf( szTest, "Compare Database Test (%s,%s)",
		pszDb1, pszDb2);
	beginTest( szTest);
	
	// Open each database.

	if( RC_BAD( rc = FlmDbOpen( pszDb1, NULL, NULL,
							FO_DONT_RESUME_BACKGROUND_THREADS, NULL, &hDb1)))
	{
		MAKE_ERROR_STRING( "calling FlmDbOpen", rc, m_szFailInfo);
		goto Exit;
	}
	if( RC_BAD( rc = FlmDbOpen( pszDb2, NULL, NULL,
							FO_DONT_RESUME_BACKGROUND_THREADS, NULL, &hDb2)))
	{
		MAKE_ERROR_STRING( "calling FlmDbOpen", rc, m_szFailInfo);
		goto Exit;
	}
	
	// Need to compare all of the records in the default data
	// container and the tracker container
	
	if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, FLM_DATA_CONTAINER)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, FLM_TRACKER_CONTAINER)))
	{
		goto Exit;
	}
	
	// Compare all of the keys in the dictionary index
	
	if (RC_BAD( rc = compareIndexes( pszDb1, pszDb2, hDb1, hDb2, FLM_DICT_INDEX)))
	{
		goto Exit;
	}

	// Compare the records in the dictionary container.
	// This will cause recursive calls to compareContainers for any
	// containers defined in the dictionary, as well as calls to
	// compareIndexes for an indexes defined in the dictionary.

	if (RC_BAD( rc = compareContainers( pszDb1, pszDb2, hDb1, hDb2, FLM_DICT_CONTAINER)))
	{
		goto Exit;
	}
	
	bPassed = TRUE;
	
Exit:

	if (hDb1 != HFDB_NULL)
	{
		(void)FlmDbClose( &hDb1);
	}
	if (hDb2 != HFDB_NULL)
	{
		(void)FlmDbClose( &hDb2);
	}

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::copyDbTest(
	const char *	pszDestDbName,
	const char *	pszSrcDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	
	// FlmDbCopy will copy a database and all of its files

	f_sprintf( szTest, "Copy Database Test (%s --> %s)",
		pszSrcDbName, pszDestDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbCopy( pszSrcDbName, NULL, NULL,
										 pszDestDbName, NULL, NULL, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbCopy", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::renameDbTest(
	const char *	pszDestDbName,
	const char *	pszSrcDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	
	// FlmDbRename will rename a database and all of its files

	f_sprintf( szTest, "Rename Database Test (%s --> %s)",
		pszSrcDbName, pszDestDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbRename( pszSrcDbName, NULL, NULL,
										 pszDestDbName, TRUE, NULL, NULL)))
	{
		MAKE_ERROR_STRING( "calling FlmDbRename", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::removeDbTest(
	const char *	pszDbName)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;
	char		szTest [200];
	
	// FlmDbRemove will delete the database and all of its files

	f_sprintf( szTest, "Remove Database Test (%s)", pszDbName);
	beginTest( szTest);

	if( RC_BAD( rc = FlmDbRemove( pszDbName, NULL, NULL, TRUE)))
	{
		MAKE_ERROR_STRING( "calling FlmDbRemove", rc, m_szFailInfo);
		goto Exit;
	}
	bPassed = TRUE;
	
Exit:

	endTest( bPassed);
	return( rc);
}
	
/***************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestImpl::execute( void)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiDrn;
	FLMUINT	uiIndex;

	// Initialize the FLAIM database engine.  This call
	// must be made once by the application prior to making any
	// other FLAIM calls

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}

	// Create database test
	
	if (RC_BAD( rc = createDbTest()))
	{
		goto Exit;
	}
	
	// FlmRecordAdd test
	
	if (RC_BAD( rc = addRecordTest( &uiDrn)))
	{
		goto Exit;
	}
	
	// FlmRecordModify test
	
	if (RC_BAD( rc = modifyRecordTest( uiDrn)))
	{
		goto Exit;
	}
	
	// Retrieve record and query tests
	
	if (RC_BAD( rc = queryRecordTest()))
	{
		goto Exit;
	}

	// FlmRecordDelete test
	
	if (RC_BAD( rc = deleteRecordTest( uiDrn)))
	{
		goto Exit;
	}
	
	// FlmKeyRetrieve test
	
	if (RC_BAD( rc = keyRetrieveTest( LAST_NAME_FIRST_NAME_IX, TRUE)))
	{
		goto Exit;
	}
	
	// Add index test
	
	if (RC_BAD( rc = addIndexTest( &uiIndex)))
	{
		goto Exit;
	}

	// FlmKeyRetrieve test
	
	if (RC_BAD( rc = keyRetrieveTest( uiIndex, FALSE)))
	{
		goto Exit;
	}
	
	// Suspend index test
	
	if (RC_BAD( rc = suspendIndexTest( uiIndex)))
	{
		goto Exit;
	}

	// Resume index test
	
	if (RC_BAD( rc = resumeIndexTest( uiIndex)))
	{
		goto Exit;
	}

	// Delete index test
	
	if (RC_BAD( rc = deleteIndexTest( uiIndex)))
	{
		goto Exit;
	}
	
	// Delete field test
	
	if (RC_BAD( rc = deleteFieldTest( AGE_TAG)))
	{
		goto Exit;
	}

	// Hot Backup/Restore test
	
	if (RC_BAD( rc = backupRestoreDbTest()))
	{
		goto Exit;
	}
	
	// Compare the restored database to the current database
	
	if (RC_BAD( rc = compareDbTest( DB_NAME_STR, DB_RESTORE_NAME_STR)))
	{
		goto Exit;
	}
	
	// Close the database

	FlmDbClose( &m_hDb);
	
	// Copy database test
	
	if (RC_BAD( rc = copyDbTest( DB_COPY_NAME_STR, DB_NAME_STR)))
	{
		goto Exit;
	}
	
	// Compare the restored database to the copied database
	
	if (RC_BAD( rc = compareDbTest( DB_COPY_NAME_STR, DB_RESTORE_NAME_STR)))
	{
		goto Exit;
	}
	
	// Rename database test
	
	if (RC_BAD( rc = renameDbTest( DB_RENAME_NAME_STR, DB_COPY_NAME_STR)))
	{
		goto Exit;
	}
	
	// Remove database test
	
	if (RC_BAD( rc = removeDbTest( DB_RENAME_NAME_STR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = removeDbTest( DB_NAME_STR)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = removeDbTest( DB_RESTORE_NAME_STR)))
	{
		goto Exit;
	}


Exit:

	FlmShutdown();

	return( rc);
}

