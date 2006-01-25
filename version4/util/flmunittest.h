//------------------------------------------------------------------------------
// Desc: Interface definition that all unit tests must implement
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: testdef.h 2961 2005-11-14 20:18:15Z ahodgkinson $
//------------------------------------------------------------------------------

#ifndef FLMUNITTEST_H
#define FLMUNITTEST_H

#include "flaimsys.h"
#ifndef FLM_NLM
	#include <stdio.h>
#endif

// Status codes passed to recordUnitTestResults

#define PASS								"PASS"
#define FAIL								"FAIL"

#define MAX_SMALL_BUFFER_SIZE			255
#define MAX_BUFFER_SIZE					2500

#define DATA_ORDER						"Testcase Name,Owner,Description,Steps,Build,Status,Environment," \
														"ResDetails,""Attributes,Folder\n"

#ifndef ELEMCOUNT
	#define ELEMCOUNT(a) \
		sizeof(a) / sizeof(a[0])
#endif

#define MAKE_ERR_STRING( str, buf) \
	f_sprintf( buf, str" file: %s. line: %u.", __FILE__, __LINE__); \
	flmAssert( 0)

#define MAKE_ERROR_STRING( str, buf, rcode) \
	f_sprintf( buf, str" "#rcode" == %X. file: %s. line: %u.", \
		(unsigned)rcode, __FILE__, __LINE__); \
	flmAssert( 0)

#define MAKE_FLM_ERROR_STRING( str, buf, rcode) \
	f_sprintf( buf, str" "#rcode" == %X : %s. file: %s. line: %u.", \
		(unsigned)rcode, FlmErrorString( rcode), __FILE__, __LINE__); \
	flmAssert( 0);
	
#define MAKE_GENERIC_ERROR_STRING( str, buf, num) \
	f_sprintf( buf, str": %lX  file: %s. line: %u.", \
		(unsigned long)num, __FILE__, __LINE__); \
	flmAssert( 0);
	
#define MAKE_GENERIC_ERROR_STRING64( str, buf, num) \
	f_sprintf( buf, str": %llX  file: %s. line: %u.", \
		(unsigned long long)num, __FILE__, __LINE__); \
	flmAssert( 0);

#define MAKE_SMI_ERR_STRING( str, buf, err) \
	f_sprintf( buf, "%s. ERROR: %d (%x). File: %s, Line %u.", \
		str, err, err, __FILE__, __LINE__); \
	flmAssert( 0);

// Error Codes

#define UNITTEST_INVALID_CSVFILE				-101
#define UNITTEST_INVALID_CONFIGFILE			-102
#define UNITTEST_CONFIGPATH_READ_FAILED	-103
#define UNITTEST_BUFFER_TOO_SMALL			-104
#define UNITTEST_INVALID_PASSWORD			-105
#define UNITTEST_INVALID_USER_NAME			-106
#define UNITTEST_INVALID_PARAM				-107
#define UNITTEST_INVALID_CONFIGPATH			-108

typedef struct KeyCompInfo
{
	void *	pvComp;
	FLMUINT	uiDataType;
	FLMUINT	uiDataSize;
} KEY_COMP_INFO;

typedef struct ElementNodeInfo
{
	void *	pvData;
	FLMUINT	uiDataType;
	FLMUINT	uiDataSize;
	FLMUINT	uiDictNum;
} ELEMENT_NODE_INFO;

typedef struct unitTestData_t
{
	char userName[ MAX_SMALL_BUFFER_SIZE];
	char buildNumber[ MAX_SMALL_BUFFER_SIZE];
	char environment[ MAX_SMALL_BUFFER_SIZE];
	char folder[ MAX_SMALL_BUFFER_SIZE];
	char attrs[ MAX_SMALL_BUFFER_SIZE];
	char csvFilename[ MAX_SMALL_BUFFER_SIZE];
} unitTestData;

RCODE createUnitTest( 
	const char *		configPath,
	const char *		buildNum, 
	const char *		user, 
	const char *		environment,
	unitTestData *		uTD);

RCODE recordUnitTestResults(
	unitTestData *		uTD,
	const char *		testName,
	const char *		testDescr, 
	const char *		steps, 
	const char *		status, 
	const char *		resultDetails);

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTest : public F_Base
{
public:

	virtual RCODE init(
		FLMBOOL			bLog,
		const char *	pszLogfile,
		FLMBOOL			bDisplay,
		FLMBOOL			bVerboseDisplay,
		const char *	pszConfigFile,
		const char *	pszEnvironment,
		const char *	pszBuild,
		const char *	pszUser) = 0;

	virtual const char * getName( void) = 0;
	
	virtual RCODE execute( void) = 0;
};

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTestDisplayer : public F_Base
{
public:

	IFlmTestDisplayer();

	RCODE init( void);
	
	void appendString( 
		const char * 		pszString);
};

/****************************************************************************
Desc:
****************************************************************************/
class ITestReporter : public F_Base
{
	unitTestData 	m_uTD;
	FLMBOOL 			m_bInitialized;

public:

	ITestReporter()
	{
		m_bInitialized = false;
		f_memset( &m_uTD, 0, sizeof( unitTestData));
	}

	virtual ~ITestReporter();
	
	RCODE init(
		const char * 		configFile,
		const char * 		buildNum,
		const char * 		environment,
		const char * 		userName);

	RCODE recordUnitTestResults(
		const char *		testName,
		const char *		testDescr,
		const char *		steps,
		const char *		status,
		const char *		resultDetails);
};

/****************************************************************************
Desc:
****************************************************************************/
class IFlmTestLogger : public F_Base
{
private:

	FLMBOOL			m_bInitialized;
	char				m_szFilename[ 128];

public:

	IFlmTestLogger()
	{
		m_bInitialized = FALSE;
		m_szFilename[ 0] = '\0';
	}

	RCODE init( 
		const char *		pszFilename);
	
	RCODE appendString( 
		const char *		pszString);
};

/******************************************************************************
Desc:	Quick, dirty, and wholly inadequate subsititute for the STL BitSet.
		This only works with strings.
******************************************************************************/
class FlagSet
{
public:

	FlagSet()
	{
		m_pbFlagArray = NULL;
		m_ppucElemArray = NULL;
		m_uiNumElems = 0;
	}

	FlagSet( const FlagSet& fs);

	~FlagSet()
	{
		reset();
	}

	FlagSet crossProduct( FlagSet& fs2);
	
	FlagSet& operator=( const FlagSet& fs);

	FLMBOOL removeElem( 
		FLMBYTE *			pElem);
	
	FLMBOOL removeElemContaining(
		FLMBYTE *			pszSubString);
		
	FLMBOOL setElemFlag(
		FLMBYTE *			pElem);
		
	FINLINE FLMBOOL allElemFlagsSet()
	{
		FLMBOOL 		bAllSet = TRUE;
		
		for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
		{
			if( m_pbFlagArray[uiLoop] == FALSE)
			{
				bAllSet = FALSE;
			}
		}
		
		return( bAllSet);
	}
	
	FINLINE FLMBOOL noElemFlagsSet( void)
	{
		FLMBOOL		bNoneSet = TRUE;
		
		for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
		{
			if( m_pbFlagArray[uiLoop] == TRUE)
			{
				bNoneSet = FALSE;
			}
		}
		
		return( bNoneSet);
	}

	FINLINE void unsetAllFlags( void)
	{
		f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * m_uiNumElems);
	}

	FINLINE FLMUINT getNumElements( void)
	{
		return m_uiNumElems;
	}
	
	void reset( void);
	
	void clearFlags( void)
	{
		f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * m_uiNumElems);
	}
	
	void init( 
		FLMBYTE **	ppucElemArray, 
		FLMUINT 		uiNumElems);
		
private:

	FLMBOOL containsSubstring(
		FLMBYTE *		pszString,
		FLMBYTE *		pszSub)
	{
		FLMBYTE *	pszStringTemp = NULL;
		FLMBYTE *	pszSubTemp  = NULL;
		FLMBOOL		bContainsSub = FALSE;
 
    // First scan quickly through the two strings looking for a
    // single-character match.  When it's found, then compare the
    // rest of the substring.

		pszSubTemp = pszSub;
		if( *pszSub == 0)
		{
			goto Exit;
		}
		
		for( ; *pszString != 0; pszString++) 
		{
			if( *pszString != *pszSubTemp) 
			{
				 continue;
			}
			
			pszStringTemp = pszString;
			for( ;;) 
			{
				if( *pszSubTemp == 0) 
				{
					bContainsSub = TRUE;
					goto Exit;
				}
				 
				if( *pszStringTemp++ != *pszSubTemp++) 
				{
					break;
				}
			}
			
			pszSubTemp = pszSub;
			
		 }
		 
Exit:

		 return( bContainsSub);
	}

	FLMBOOL *	m_pbFlagArray;
	FLMBYTE **	m_ppucElemArray;
	FLMUINT		m_uiNumElems;
};

/******************************************************************************
Desc: Makes iterating through the keys of an index more convenient
******************************************************************************/
FINLINE FLMBOOL isPlaceHolder( const ELEMENT_NODE_INFO & elm)
{
	if( elm.pvData == NULL &&
		 elm.uiDataSize == 0 &&
		 elm.uiDictNum == 0)
	{
		return( TRUE);
	}
	
	return( FALSE);
}

/******************************************************************************
Desc: Makes iterating through the keys of an index more convenient
******************************************************************************/
class TestBase : public IFlmTest
{

public:

	TestBase()
	{
		m_bLog = FALSE;
		m_bDisplay = FALSE;
		m_bDisplayVerbose = FALSE;
		m_pLogger = NULL;
		m_pDisplayer = NULL;
		m_pReporter = NULL;
		m_hDb = HFDB_NULL;
		m_pszTestName = NULL;
		m_pszTestDesc = NULL;
		m_pszSteps = NULL;
	}

	virtual ~TestBase();

	RCODE init(
		FLMBOOL			bLog,
		const char *	pszLogfile,
		FLMBOOL			bDisplay,
		FLMBOOL			bVerboseDisplay,
		const char *	pszConfigFile,
		const char *	pszEnvironment,
		const char *	pszBuild,
		const char *	pszUser);

protected:

	FLMBOOL						m_bLog;
	FLMBOOL						m_bDisplay;
	FLMBOOL						m_bDisplayVerbose;
	IFlmTestLogger *			m_pLogger;
	IFlmTestDisplayer *		m_pDisplayer;
	ITestReporter *			m_pReporter;
	HFDB 							m_hDb;
#define DETAILS_BUF_SIZ	1024
	char							m_szDetails[ DETAILS_BUF_SIZ];
	const char *				m_pszTestName;
	const char *				m_pszTestDesc;
	const char *				m_pszSteps;

	void beginTest( 
		const char *			pszTestName, 
		const char *			pszTestDesc,
		const char *			pszTestSteps,
		const char *			pszDetails);

	void endTest( 
		const char *			pszTestResult);

	void log( 
		const char *			pszString);
		
	void display( 
		const char *			pszString);
		
	void displayLine(
		const char *			pszString);
		
	RCODE logTestResults(
		const char * 			pszTestResult);

	RCODE displayTestResults(
		const char * 			pszTestResult);

	RCODE outputAll( 
		const char * 			pszTestResult);

	RCODE initCleanTestState( 
		const char  *			pszDibName);

	RCODE openTestState(
		const char *			pszDibName);

	RCODE shutdownTestState(
		const char *			pszDibName,
		FLMBOOL					bRemoveDib);

	RCODE createCompoundDoc(
		ELEMENT_NODE_INFO *	pElementNodes,
		FLMUINT					uiNumElementNodes,
		FLMUINT64 *				pui64DocId);
};

/****************************************************************************
Desc:
****************************************************************************/
class ArgList
{
public:

	ArgList();
	
	~ArgList();

	const char * operator[](
		FLMUINT 			uiIndex);

	const char * getArg(
		FLMUINT 			uiIndex);

	RCODE addArg(
		const char * 	pszArg);

	FLMUINT getNumEntries( void);

	RCODE expandArgs(
		char ** 			ppszArgs,
		FLMUINT			uiNumArgs);

	RCODE expandFileArgs(
		const char * 	pszFilename);
		
private:

	enum
	{
		INIT_SIZE = 10,
		GROW_FACTOR = 2
	};

	char **		m_ppszArgs;
	FLMUINT		m_uiCapacity;
	FLMUINT		m_uiNumEntries;

	RCODE resize( void);
	
	RCODE getTokenFromFile( 
		char *		pszToken,
		F_FileHdl * pFileHdl);

	FLMBOOL isWhitespace( 
		char 			c)
	{
		return (c == ' ' || c == '\t' || c == 0xd || c == 0xa)
			? TRUE
			: FALSE;
	}
};

#define FILENAME_ITERATOR_MAX_PATH_LENGTH			30
#define FILENAME_ITERATOR_MAX_PREFIX_LENGTH		8
#define FILENAME_ITERATOR_MAX_EXTENSION_LENGTH	3
#define FILENAME_ITERATOR_NULL 						-1

/****************************************************************************
Desc:	Simple class for generating sequential filenames
****************************************************************************/
class FilenameIterator : public F_Base
{
public:
	FilenameIterator(
		const char *	pszPath, 
		const char *	pszPrefix);

	void getNext(
		char *			pszPath);

	void getCurrent(
		char *			pszPath);

	void reset();

private:
	char			m_pszFullPrefix[					//stores everything but the .ext
		FILENAME_ITERATOR_MAX_PATH_LENGTH +		//store the path
		1 +												//store the path separator 
		FILENAME_ITERATOR_MAX_PREFIX_LENGTH +	//store the file prefix
		1];												//store the '\0'
	FLMUINT		m_uiExtension;

	void produceFilename(
		char *			pszPath);
};

#endif
