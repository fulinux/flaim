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

#define PERSON_TAG					1
#define LAST_NAME_TAG				2
#define FIRST_NAME_TAG				3
#define AGE_TAG						4
#define LAST_NAME_FIRST_NAME_IX	5

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
		
	RCODE removeDbTest( void);
		
	RCODE execute( void);
	
private:

	HFDB	m_hDb;
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
RCODE IFlmTestImpl::removeDbTest( void)
{
	RCODE		rc = FERR_OK;
	FLMBOOL	bPassed = FALSE;

	// FlmDbRemove will delete the database and all of its files

	beginTest( "Remove Database Test");

	if( RC_BAD( rc = FlmDbRemove( DB_NAME_STR, NULL, NULL, TRUE)))
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
	
	// Close the database

	FlmDbClose( &m_hDb);
	
	if (RC_BAD( rc = removeDbTest()))
	{
		goto Exit;
	}

Exit:

	FlmShutdown();

	return( rc);
}

