//------------------------------------------------------------------------------
// Desc:	Unit test driver
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
// $Id: flmunittest.cpp 3105 2006-01-11 11:14:10 -0700 (Wed, 11 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#include "flmunittest.h"

#if defined( FLM_WIN)
	#if defined( FLM_64BIT)
		#define PLATPROC_STR		"w64ia64"
	#else
		#define PLATPROC_STR		"w32x86"
	#endif
#elif defined( FLM_NLM)
	#define PLATPROC_STR			"nwx86"
#elif defined( FLM_LINUX)
	#define PLATPROC_STR			"lxx86"
#elif defined( FLM_OSX)
	#define PLATPROC_STR			"osx"
#elif defined( FLM_SOLARIS)
	#define PLATPROC_STR			"solaris"
#endif

extern RCODE getTest( 
	IFlmTest **		ppTest);

#ifdef FLM_NLM
	extern "C"
	{
		void flmUnittestFatalRuntimeError( void);
	
		int flmUnittestLoad( void);
		
		int flmUnittestUnload( void);
	}
#endif

FLMBOOL gv_bShutdown = FALSE;

/****************************************************************************
Desc:
****************************************************************************/
struct TEST_INFO
{
	bool 				bLog;
	char 				pszLogfile[ 256];
	bool 				bDisplay;
	bool 				bVerboseDisplay;
	char 				pszEnvironment[ 32];
	char 				pszBuild[ 32];
	char 				pszUser[ 32];
	char 				pszConfig[ 256];
	TEST_INFO * 	pNext;

	TEST_INFO()
	{
		bLog = FALSE;
		bDisplay = FALSE;
		bVerboseDisplay = FALSE;
		pNext = NULL;
		pszLogfile[ 0] = 0;
		pszConfig[ 0] = 0;
		
		f_strcpy( pszEnvironment, PLATPROC_STR);
		f_strcpy( pszBuild, __DATE__);
		f_strcpy( pszUser, "defaultUser");
	}
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE ITestReporter::init(
	const char * 	configFile,
	const char * 	buildNum,
	const char * 	environment,
	const char * 	userName)
{
	RCODE 			rc = FERR_OK;

	if( (rc = createUnitTest( configFile, buildNum, environment, userName,
		&(this->m_uTD))) != 0)
	{
		goto Exit;
	}

	m_bInitialized = true;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
ITestReporter::~ITestReporter()
{
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ITestReporter::recordUnitTestResults(
	const char *		testName,
	const char *		testDescr,
	const char *		steps,
	const char *		status,
	const char *		resultDetails)
{
	return ::recordUnitTestResults( &(this->m_uTD), testName, testDescr,
		steps, status, resultDetails);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestLogger::init( 
	const char *		pszFilename)
{
	f_strcpy( m_szFilename, pszFilename);
	m_bInitialized = TRUE;
	return( FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestLogger::appendString( 
	const char *		pszString)
{
	RCODE				rc = FERR_OK;
	char *			pszTemp = NULL;

	if ( RC_BAD( rc = f_alloc( f_strlen( pszString) + 3, &pszTemp)))
	{
		goto Exit;
	}

	f_sprintf( pszTemp, "%s\n", pszString);

	if( RC_BAD( rc = f_filecat( m_szFilename, pszString)))
	{
		goto Exit;
	}

Exit:

	if ( pszTemp)
	{
		f_free( &pszTemp);
	}

	return rc;
}

/****************************************************************************
Desc:
****************************************************************************/
IFlmTestDisplayer::IFlmTestDisplayer()
{
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE IFlmTestDisplayer::init( void)
{
	return( FERR_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
void IFlmTestDisplayer::appendString( 
	const char * 	pszString)
{
#ifdef FLM_NLM
	F_UNREFERENCED_PARM( pszString);
#else
	printf( pszString);
#endif
}

/****************************************************************************
Desc:
****************************************************************************/
void printHelp()
{
#ifndef FLM_NLM
	printf( "\nCommand-line usage:");
	printf( "\n\n[-l<log file>] [-d]");
	printf( "[-c<config>] [-b<build>] [-u<user>]");
	printf( "\n-l - Specifies a log file to print to");
	printf( "\n-d - Display output");
	printf( "\n-t - Specifies configuration file for reporting");
	printf( "\n-b - Specifies the build number");
	printf( "\n-u - Specifies the user running the unit test");
	printf( "\n-i - Interactive mode (pause before exit)");
	printf( "\n-h - Shows this screen");
#endif
}

#if defined( FLM_NLM)
	#define main		nlm_main
	
	extern "C"
	{
		int nlm_main( 
			int				argc,
			char ** 			argv);
	}
#endif

/****************************************************************************
Desc:
****************************************************************************/
int main( 
	int				argc,
	char ** 			argv)
{
	RCODE				rc = FERR_OK;
	IFlmTest *		pTest = NULL;
	unsigned int	i = 1;
	ArgList			args;
	TEST_INFO		testInfo;
	FLMBOOL			bPauseBeforeExit = FALSE;

	if( argc > 1)
	{
		if( (f_strcmp( argv[1], "--help") == 0) || 
			 (f_strcmp( argv[1], "-h") == 0))
		{
			printHelp();
			goto Exit;
		}
	}

	args.expandArgs( argv, argc);

	while( i < args.getNumEntries())
	{
		if ( args[i][0] != '-')
		{
			goto Exit;
		}
				
		if( (args[i][1] == 'l') || (args[i][1] == 'L'))
		{
			testInfo.bLog = true;
			f_strcpy( testInfo.pszLogfile, &args[i][2]);
		}
		else if( (args[i][1] == 'd') || (args[i][1] == 'D'))
		{
			testInfo.bDisplay = true;
		}
		else if( (args[i][1] == 'c') || (args[i][1] == 'C'))
		{
			f_strcpy( testInfo.pszConfig, &args[i][2]);
		}
		else if( (args[i][1] == 'b') || (args[i][1] == 'B'))
		{
			f_strcpy( testInfo.pszBuild, &args[i][2]);
		}
		else if( (args[i][1] == 'u') || (args[i][1] == 'U'))
		{
			f_strcpy( testInfo.pszUser, &args[i][2]);
		}
		else if( (args[i][1] == 'v') || (args[i][1] == 'V'))
		{
			testInfo.bVerboseDisplay = TRUE;
		}
		else if( (args[i][1] == 'i') || (args[i][1] == 'I'))
		{
			bPauseBeforeExit = TRUE;
		}		
		else
		{
#ifndef FLM_NLM
			printf( "\nInvalid parameter");
			printHelp();
#endif
			goto Exit;
		}
		
		i++;
	}

#ifndef FLM_NLM
	printf("Running %s\n", argv[0]);
#endif

	if( RC_BAD( rc = getTest( &pTest)))
	{
#ifndef FLM_NLM
		printf( "ERROR: Unable to create test instance\n");
#endif
		goto Exit;
	}

	if( pTest->init( testInfo.bLog, testInfo.pszLogfile, testInfo.bDisplay,
		testInfo.bVerboseDisplay, testInfo.pszConfig, testInfo.pszEnvironment,
		testInfo.pszBuild, testInfo.pszUser) != 0)
	{
#ifndef FLM_NLM
		printf( "\nTest initialization failed");
#endif
		goto Exit;
	}

	if( RC_BAD( rc = pTest->execute()))
	{
		goto Exit;
	}

Exit:

	if( pTest)
	{
		pTest->Release();
	}

	if( bPauseBeforeExit)
	{
#ifndef FLM_NLM
		printf( "Press any key to exit.\n");
		getchar();
#endif
	}

	return( (int)rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FlagSet::removeElem(
	FLMBYTE *	pElem)
{
	FLMBOOL 		bElemExisted = FALSE;

	for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
	{
		if( f_strcmp( (char *)pElem, (char *)m_ppucElemArray[ uiLoop]) == 0)
		{
			bElemExisted = TRUE;
			if( uiLoop < m_uiNumElems - 1)
			{
				f_free( &m_ppucElemArray[ uiLoop]);
				
				f_memmove( &m_ppucElemArray[ uiLoop], 
					&m_ppucElemArray[ uiLoop + 1], 
					(m_uiNumElems - ( uiLoop + 1)) * sizeof( FLMBYTE *));
					
				f_memmove( &m_pbFlagArray[ uiLoop], 
					&m_pbFlagArray[ uiLoop + 1], 
					(m_uiNumElems - ( uiLoop + 1)) * sizeof( FLMBYTE *));
			}
			
			m_uiNumElems--;
		}
	}
	
	return( bElemExisted);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FlagSet::removeElemContaining(
	FLMBYTE *	pszSubString)
{
	FLMBOOL 		bElemExisted = FALSE;

	for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; )
	{
		if( containsSubstring( m_ppucElemArray[ uiLoop], pszSubString))
		{
			bElemExisted = TRUE;
			
			if( uiLoop < m_uiNumElems - 1)
			{
				f_memmove( &m_ppucElemArray[ uiLoop], 
					&m_ppucElemArray[ uiLoop + 1], 
					(m_uiNumElems - (uiLoop + 1)) * sizeof( FLMBYTE *));
					
				f_memmove( &m_pbFlagArray[ uiLoop], 
					&m_pbFlagArray[ uiLoop + 1], 
					(m_uiNumElems - (uiLoop + 1)) * sizeof( FLMBYTE *));
			}
			
			m_uiNumElems--;
		}
		else
		{
			uiLoop++;
		}
	}
	
	return( bElemExisted);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FlagSet::setElemFlag(
	FLMBYTE *	pElem)
{
	FLMBOOL 		bIsInSet = FALSE;

	for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
	{
		if( f_strcmp( (char *)pElem, (char *)m_ppucElemArray[ uiLoop]) == 0 &&
			  !m_pbFlagArray [uiLoop])
		{
			m_pbFlagArray[ uiLoop] = TRUE;
			bIsInSet = TRUE;
			break;
		}
	}
	
	return( bIsInSet);
}

/****************************************************************************
Desc:
****************************************************************************/
FlagSet FlagSet::crossProduct( 
	FlagSet&		fs2)
{
	FlagSet		fsCross;
	FLMUINT		uiLoop1;
	FLMUINT		uiCrossProductElems = this->getNumElements() * fs2.getNumElements();
	FLMBYTE **	ppszCross = NULL;
	
	if( RC_BAD( f_alloc( sizeof( FLMBYTE *) * uiCrossProductElems,
		&ppszCross)))
	{
		flmAssert( 0);
		goto Exit;
	}

	for( uiLoop1 = 0; uiLoop1 < this->getNumElements(); uiLoop1++)
	{
		for( FLMUINT uiLoop2 = 0; uiLoop2 < fs2.getNumElements(); uiLoop2++)
		{
			FLMUINT	uiIndex = uiLoop1 * fs2.getNumElements() + uiLoop2;
			
			if( RC_BAD( f_alloc( f_strlen((char *)this->m_ppucElemArray[ uiLoop1]) + 
						f_strlen((char *)fs2.m_ppucElemArray[ uiLoop2]) + 1,
					&ppszCross[ uiIndex])))
			{
				flmAssert( 0);
			}
			
			f_strcpy( (char *)ppszCross[ uiIndex], 
				(char *)this->m_ppucElemArray[ uiLoop1]);
				
			f_strcat( (char *)ppszCross[ uiIndex], 
				(char *)fs2.m_ppucElemArray[ uiLoop2]);
		}
	}
	
	fsCross.init( ppszCross, uiCrossProductElems);

	for( uiLoop1 = 0; uiLoop1 < uiCrossProductElems; uiLoop1++)
	{
		f_free( &ppszCross[ uiLoop1]);
	}
	
	f_free( &ppszCross);
	
Exit:

	return( fsCross);
}

/****************************************************************************
Desc:
****************************************************************************/
FlagSet& FlagSet::operator=( 
	const FlagSet&		fs)
{
	if( this != &fs)
	{
		if( m_ppucElemArray || m_pbFlagArray)
		{
			this->reset();
		}
		
		this->init( fs.m_ppucElemArray, fs.m_uiNumElems);
	}
	
	return( *this);
}

/****************************************************************************
Desc:
****************************************************************************/
FlagSet::FlagSet( 
	const FlagSet&		fs)
{
	if( RC_BAD( f_alloc( sizeof( FLMBYTE *) * fs.m_uiNumElems,
										&m_ppucElemArray)))
	{
		flmAssert( 0);
	}
	
	if( RC_BAD( f_alloc( sizeof( FLMBOOL) * fs.m_uiNumElems,
										&m_pbFlagArray)))
	{
		flmAssert( 0);
	}
	
	f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * fs.m_uiNumElems);
	
	for( FLMUINT uiLoop = 0; uiLoop < fs.m_uiNumElems; uiLoop++)
	{
		if( RC_BAD( f_alloc( f_strlen( (char *)fs.m_ppucElemArray[uiLoop]) + 1,
				&m_ppucElemArray[ uiLoop])))
		{
			flmAssert( 0);
		}
		
		f_strcpy( (char *)m_ppucElemArray[uiLoop], 
					 (char *)fs.m_ppucElemArray[uiLoop]);
	}
	
	m_uiNumElems = fs.m_uiNumElems;
}

/****************************************************************************
Desc:
****************************************************************************/
void FlagSet::reset( void)
{
	for( FLMUINT uiLoop = 0; uiLoop < m_uiNumElems; uiLoop++)
	{
		f_free( &m_ppucElemArray[ uiLoop]);
	}

	f_free( &m_ppucElemArray);
	f_free( &m_pbFlagArray);	

	m_uiNumElems = 0;
	m_ppucElemArray = NULL;
	m_pbFlagArray = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
void FlagSet::init( 
	FLMBYTE **		ppucElemArray,
	FLMUINT			uiNumElems)
{
	reset();
	
	if( RC_BAD( f_alloc( sizeof( FLMBYTE *) * uiNumElems,
										&m_ppucElemArray)))
	{
		flmAssert( 0);
	}
	
	if( RC_BAD( f_alloc( sizeof( FLMBOOL) * uiNumElems,
										&m_pbFlagArray)))
	{
		flmAssert( 0);
	}
	
	f_memset( m_pbFlagArray, 0, sizeof( FLMBOOL) * uiNumElems);
	
	for( FLMUINT uiLoop = 0; uiLoop < uiNumElems; uiLoop++)
	{
		if( RC_BAD( f_alloc( f_strlen( (char *)ppucElemArray[ uiLoop]) + 1,
				&m_ppucElemArray[ uiLoop])))
		{
			flmAssert( 0);
		}
		
		f_strcpy( (char *)m_ppucElemArray[uiLoop], (char *)ppucElemArray[uiLoop]);
	}
	
	m_uiNumElems = uiNumElems;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE createUnitTest(
	const char *	configPath, 
	const char *	buildNum, 
	const char *	environment,  
	const char *	user, 
	unitTestData *	uTD)
{
	RCODE				rc = FERR_OK;
	F_FileHdl *		pConfigFileHdl = NULL;
	F_FileHdl *		pCSVFileHdl = NULL;
	FLMBYTE			buffer[ MAX_BUFFER_SIZE] = "";
	FLMUINT			size = MAX_BUFFER_SIZE;
	char *			strPos1 = NULL;
	char *			strPos2 = NULL;

	if( !configPath || !buildNum || !environment || !uTD || !user)
	{
		flmAssert(0);
	}

	if( f_strlen(user) > MAX_SMALL_BUFFER_SIZE)
	{
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	else
	{
		f_strcpy( uTD->userName, user);
	}

	if( f_strlen(environment) > MAX_SMALL_BUFFER_SIZE)
	{
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	else
	{
		f_strcpy( uTD->environment, environment);
	}

	if( f_strlen( buildNum) > MAX_SMALL_BUFFER_SIZE)
	{
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	else
	{
		f_strcpy( uTD->buildNumber, buildNum);
	}
	
	if( configPath[ 0])
	{
		if( RC_BAD( rc = FlmAllocFileHandle( &pConfigFileHdl)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pConfigFileHdl->Open(
			configPath, F_IO_RDONLY | F_IO_SH_DENYNONE)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = pConfigFileHdl->Size( &size)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pConfigFileHdl->Read( 0, size, buffer, &size)))
		{
			goto Exit;
		}
	
		#ifdef FLM_WIN
		{
			char		szTemp[ MAX_BUFFER_SIZE];
			char *	pszTemp = szTemp;
			size_t	newsize = size;
		
			for( unsigned int i = 0; i < size; i++)
			{
				if( ((i + 1) < size) 
					&& (buffer[i] == 0x0D && buffer[ i + 1] == 0x0A))
				{
					*pszTemp++ = 0x0A;
					i++;
					newsize--;
				}
				else
				{
					*pszTemp++ = buffer[ i];
				}
			}
				
			f_memcpy( buffer, szTemp, (FLMSIZET)newsize);
			size = newsize;
		}
		#endif
		
		// Get the FOLDER
		
		strPos1 = f_strchr( (const char *)buffer, ':');
		strPos2 = f_strchr( (const char *)strPos1, '\n');
		
		if( !strPos1 || !strPos2)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		for( strPos1++; *strPos1 == ' ' || *strPos1 == '\t'; strPos1++);
		
		if( strPos2-strPos1 > MAX_SMALL_BUFFER_SIZE)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		
		f_strncpy( uTD->folder, strPos1, strPos2-strPos1);
		uTD->folder[ strPos2 - strPos1] = '\0';

		// Get the ATTRIBUTES
		
		strPos1 = f_strchr( (const char *)strPos1, ':');
		strPos2 = f_strchr( (const char *)strPos1, '\n');
		
		if( !strPos1 || !strPos2)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		for( strPos1++;*strPos1 == ' ' || *strPos1 == '\t';strPos1++);
		
		if( strPos2-strPos1 > MAX_SMALL_BUFFER_SIZE)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}
		
		f_strncpy( uTD->attrs, strPos1, strPos2-strPos1);
		uTD->attrs[strPos2-strPos1] = '\0';

		// Get the CSVFILE
		
		strPos1 = f_strchr( (const char *)strPos1, ':');
		strPos2 = f_strchr( (const char *)strPos1, '\n');
		
		// Allow for possible \r
		
		if( *( --strPos2) != '\r')
		{
			strPos2++;
		}

		if( !strPos1 || !strPos2)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		for( strPos1++;*strPos1 == ' ' || *strPos1 == '\t';strPos1++);

		if( strPos2-strPos1 > MAX_SMALL_BUFFER_SIZE)
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		f_strncpy( uTD->csvFilename, strPos1, strPos2-strPos1);
		uTD->csvFilename[ strPos2 - strPos1] = '\0';

		if( RC_BAD( rc = FlmAllocFileHandle( &pCSVFileHdl)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = pCSVFileHdl->Open( uTD->csvFilename,
			F_IO_RDWR | F_IO_SH_DENYNONE)))
		{
			if ( rc == FERR_IO_PATH_NOT_FOUND)
			{
				// Create the file and write the header
				
				if( RC_BAD( rc = f_filecat( uTD->csvFilename, DATA_ORDER)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			goto Exit;
		}
	}

Exit:

	if( pConfigFileHdl)
	{
		pConfigFileHdl->Release();
	}
	
	if( pCSVFileHdl)
	{
		pCSVFileHdl->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	After each unit test call this to record the unit test status
		to a CSV file.
			uTD - contains the configuration information
			testName - a Unique for this module unite test name.
			testDescr - A description of the unit test.
			steps - The steps the unit test performs.
			status - Maybe be PASS or FAIL
			resultDetails - details that explain the result status.
****************************************************************************/
RCODE recordUnitTestResults(
	unitTestData *		uTD, 
	const char *		testName, 
	const char *		testDescr, 
	const char *		steps, 
	const char *		status, 
	const char *		resultDetails) 
{
	RCODE		rc = FERR_OK;
	char		buffer[ MAX_BUFFER_SIZE];
	
	if( !testName || !testDescr || !steps || !status || !resultDetails || !uTD )
	{
		flmAssert(0);
	}

	if( uTD->csvFilename[ 0])
	{
		f_sprintf( buffer, "%s,%s,%s,%s,%s,%s,%s,%s,"/*%s,*/"%s,%s\n", 
			testName, uTD->userName, testDescr, steps, uTD->buildNumber, status, 
			uTD->environment, resultDetails, uTD->attrs, uTD->folder);

		if( RC_BAD( rc = f_filecat( uTD->csvFilename, buffer)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::init(
	FLMBOOL			bLog,
	const char *	pszLogfile,
	FLMBOOL			bDisplay,
	FLMBOOL			bVerboseDisplay,
	const char *	pszConfigFile,
	const char *	pszEnvironment,
	const char *	pszBuild,
	const char *	pszUser)
{
	RCODE		rc = FERR_MEM;

	if( RC_BAD( rc = FlmStartup()))
	{
		goto Exit;
	}
	
	// VISIT: here -- disable asserts on FLAIM errors via a config call!!!

	m_bLog = bLog;
	m_bDisplay = bDisplay;

	// Set up logger and displayer if true
	
	if( m_bLog)
	{
		if( ( m_pLogger = f_new IFlmTestLogger) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pLogger->init( pszLogfile)))
		{
			goto Exit;
		}
	}
	
	if( m_bDisplay)
	{
		if( (m_pDisplayer = f_new IFlmTestDisplayer) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		
		if( RC_BAD( rc = m_pDisplayer->init()))
		{
			goto Exit;
		}
	}

	if( (m_pReporter = f_new ITestReporter) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pReporter->init( pszConfigFile, pszBuild,
		pszEnvironment, pszUser)))
	{
		goto Exit;
	}
	m_bDisplayVerbose = bVerboseDisplay;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::beginTest( 
	const char * 	pszTestName, 
	const char * 	pszTestDesc,
	const char * 	pszTestSteps,
	const char * 	pszDetails)
{
	char		szTemp[ 256];

	m_pszTestName = pszTestName;
	m_pszTestDesc = pszTestDesc;
	m_pszSteps = pszTestSteps;

	if( m_bDisplayVerbose)
	{
		displayLine(
			"========================================"
			"=======================================");
			
		f_sprintf( szTemp, "Test Name: %s", m_pszTestName);
		displayLine( szTemp);
		
		f_sprintf( szTemp, "Test Description: %s", m_pszTestDesc);
		displayLine( szTemp);
		
		f_sprintf( szTemp, "Steps: %s", m_pszSteps);
		displayLine( szTemp);
	}
	else
	{
		f_sprintf( szTemp, "Test Name: %s ... ", m_pszTestName);
		display( szTemp);
	}

	f_strcpy( m_szDetails, pszDetails);
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::endTest( 
	const char * 	pszTestResult)
{
	outputAll( pszTestResult);
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::log(
	const char *	pszString)
{
	if( m_bLog)
	{
		m_pLogger->appendString( pszString);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::display(
	const char * 	pszString)
{
	if( m_bDisplay)
	{
		m_pDisplayer->appendString( pszString);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void TestBase::displayLine(
	const char * 	pszString)
{
	if( m_bDisplay)
	{
		m_pDisplayer->appendString( pszString);
		m_pDisplayer->appendString( "\n");
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::initCleanTestState( 
	const char * 	pszDibName)
{
	RCODE				rc = FERR_OK;
	CREATE_OPTS		createOpts;

	// Create the database
	
	f_memset( &createOpts, 0, sizeof( CREATE_OPTS));
	
	if ( RC_BAD( rc = FlmDbCreate( pszDibName, 
		NULL, NULL, NULL, NULL, &createOpts, &m_hDb)))
	{
		if( rc == FERR_FILE_EXISTS)
		{
			if( RC_BAD( rc = FlmDbRemove( pszDibName,
				NULL, NULL, TRUE)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = FlmDbCreate( pszDibName, 
			NULL, NULL, NULL, NULL, &createOpts, &m_hDb)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::openTestState(
	const char *		pszDibName)
{
	RCODE					rc = FERR_OK;
	CREATE_OPTS			createOpts;

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Exists( pszDibName)))
	{
		// Create the database

		f_memset( &createOpts, 0, sizeof( CREATE_OPTS));
		
		if( RC_BAD( rc = FlmDbCreate( pszDibName, 
			NULL, NULL, NULL, NULL, &createOpts, &m_hDb)))
		{
			goto Exit;
		}
	}
	else
	{
		// Open the existing database

		if( RC_BAD( rc = FlmDbOpen( pszDibName, NULL, NULL, 
			0, NULL, &m_hDb)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::shutdownTestState(
	const char *	pszDibName,
	FLMBOOL			bRemoveDib)
{
	RCODE			rc = FERR_OK;

	if( bRemoveDib)
	{
		if( m_hDb != HFDB_NULL)
		{
			FlmDbClose( &m_hDb);
		}
		
		if( RC_BAD( rc = FlmDbRemove( pszDibName, NULL, NULL, TRUE)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::logTestResults(
	const char * 	pszTestResult)
{
	RCODE			rc = FERR_OK;
	char *		pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( DETAILS_BUF_SIZ + 64, &pszTemp)))
	{
		goto Exit;
	}

	log(
			"========================================"
			"=======================================");
			
	f_sprintf( pszTemp, "Test Name: %s", m_pszTestName);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Test Description: %s", m_pszTestDesc);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Steps: %s", m_pszSteps);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Test Result: %s", pszTestResult);
	log( pszTemp);
	
	f_sprintf( pszTemp, "Details: %s", m_szDetails);
	log( pszTemp);
	
	log(
			"========================================"
			"=======================================");

Exit:

	if( pszTemp)
	{
		f_free( &pszTemp);
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::displayTestResults(
	const char *	pszTestResult)
{
	RCODE			rc = FERR_OK;
	char *		pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( DETAILS_BUF_SIZ + 64, &pszTemp)))
	{
		goto Exit;
	}

	if( m_bDisplayVerbose)
	{
		f_sprintf( pszTemp, "Test Result: %s", pszTestResult);
		displayLine( pszTemp);
		
		f_sprintf( pszTemp, "Details: %s", m_szDetails);
		displayLine( pszTemp);
		
		displayLine(
			"========================================"
			"=======================================");
	}
	else
	{
		f_sprintf( pszTemp, "Result: %s", pszTestResult);
		displayLine( pszTestResult);
	}

Exit:

	if( pszTemp)
	{
		f_free( &pszTemp);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE TestBase::outputAll( 
	const char * 	pszTestResult)
{
	RCODE			rc = FERR_OK;

	if( RC_BAD( rc = displayTestResults( pszTestResult)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = logTestResults( pszTestResult)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pReporter->recordUnitTestResults(
		m_pszTestName, m_pszTestDesc, m_pszSteps, pszTestResult, m_szDetails)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
TestBase::~TestBase()
{
	if( m_pLogger)
	{
		m_pLogger->Release();
	}
	
	if( m_pDisplayer)
	{
		m_pDisplayer->Release();
	}
	
	if( m_pReporter)
	{
		m_pReporter->Release();
	}
	
	if( m_hDb != HFDB_NULL)
	{
		FlmDbClose( &m_hDb);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
ArgList::ArgList()
{
	m_uiCapacity = INIT_SIZE;
	m_uiNumEntries = 0;
	
	f_alloc( m_uiCapacity * sizeof( char *), &m_ppszArgs);
}

/****************************************************************************
Desc:
****************************************************************************/
ArgList::~ArgList()
{
	FLMUINT	uiLoop;

	if( m_ppszArgs)
	{
		for( uiLoop = 0; uiLoop < m_uiNumEntries; uiLoop++)
		{
			f_free( &( m_ppszArgs[uiLoop]));
		}
		
		f_free( &m_ppszArgs);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::resize( void)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLoop;
	char **		ppszTemp = NULL;

	if( RC_BAD( rc = f_alloc( m_uiCapacity * GROW_FACTOR * sizeof(char*), 
		&ppszTemp)))
	{
		goto Exit;
	}

	m_uiCapacity *= GROW_FACTOR;
	
	for( uiLoop = 0; uiLoop < m_uiNumEntries; uiLoop++)
	{
		if( RC_BAD( rc = f_alloc( f_strlen( m_ppszArgs[uiLoop]) + 1, 
			&ppszTemp[uiLoop])))
		{
			f_free( &ppszTemp);
			goto Exit;
		}
		
		f_strcpy( ppszTemp[uiLoop], m_ppszArgs[uiLoop]);
	}

	f_free( &m_ppszArgs);
	m_ppszArgs = ppszTemp;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT ArgList::getNumEntries( void)
{
	return( m_uiNumEntries);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::addArg( 
	const char *	pszArg)
{
	RCODE		rc = FERR_OK;

	if( m_uiNumEntries >= m_uiCapacity)
	{
		if( RC_BAD( rc = resize()))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = f_alloc( f_strlen( pszArg) + 1,
		&m_ppszArgs[m_uiNumEntries])))
	{
		goto Exit;
	}

	f_strcpy( m_ppszArgs[ m_uiNumEntries++], pszArg);

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * ArgList::getArg(
	FLMUINT		uiIndex)
{
	return( m_ppszArgs[ uiIndex]);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::expandFileArgs( 
	const char *		pszFilename)
{
	RCODE				rc = FERR_OK;
	char				token[64];
	F_FileHdl *		pFileHdl = NULL;

	if( RC_BAD( rc = gv_FlmSysData.pFileSystem->Open(
		pszFilename, F_IO_RDWR, &pFileHdl)))
	{
		goto Exit;
	}

	while( RC_OK( rc = getTokenFromFile(token, pFileHdl)))
	{
		if( token[0] == '@')
		{
			if( RC_BAD( rc = expandFileArgs( &token[1])))
			{
				goto Exit;
			}
		}
		else
		{
			flmAssert(*token);
			if( RC_BAD( rc = addArg( token)))
			{
				goto Exit;
			}
		}
	}
	
	if( rc == FERR_IO_END_OF_FILE)
	{
		rc = FERR_OK;
	}

Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::expandArgs(
	char ** 		ppszArgs,
	FLMUINT 		uiNumArgs)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLoop;

	for( uiLoop = 0; uiLoop < uiNumArgs; uiLoop++)
	{
		if( ppszArgs[uiLoop][0] == '@')
		{
			if( RC_BAD( rc = expandFileArgs( &ppszArgs[uiLoop][ 1])))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = addArg( ppszArgs[ uiLoop])))
			{
				goto Exit;
			}
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
const char * ArgList::operator []( FLMUINT uiIndex)
{
	return( m_ppszArgs[ uiIndex]);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ArgList::getTokenFromFile(
	char * 			pszToken,
	F_FileHdl * 	pFileHdl)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiSize = 1;
	FLMUINT		uiOffset = 0;
	FLMUINT		uiTrueOffset = 0;
	char			c;

	for(;;)
	{
Skip_WS_And_Comments:

		if ( RC_BAD( rc = pFileHdl->Read( 0, 1, &c, &uiSize)))
		{
			goto Exit;
		}

		// Skip whitespace
		
		while( isWhitespace(c))
		{
			if( RC_BAD( rc = pFileHdl->Read( 0, 1, &c, &uiSize)))
			{
				goto Exit;
			}
		}

		if( c == '#')
		{
			// Skip comment

			for (;;)
			{
				if( RC_BAD( rc = pFileHdl->Read( 0, 1, &c, &uiSize)))
				{
					goto Exit;
				}

#ifdef FLM_UNIX
				// On unix platforms, an EOL is indicated by an LF
				
				if( c == 0x0A)
				{
					break;
				}
#else
				// On Windows and NetWare we need to look for CR/LF
				
				if( c == 0x0D)
				{

					if( RC_BAD( rc = pFileHdl->Read( 0, 1, &c, &uiSize)))
					{
						goto Exit;
					}

					// Newline found
					
					if( c == 0x0A)
					{
						break;
					}
					else
					{
						// Rewind
						
						uiOffset = 0;
						uiTrueOffset = 0;

						if( RC_BAD( rc = pFileHdl->Tell( &uiOffset)))
						{
							goto Exit;
						}
						
						uiOffset--;

						if( RC_BAD( rc = pFileHdl->Seek( uiOffset, 
							F_IO_SEEK_SET, &uiTrueOffset)))
						{
							goto Exit;
						}
					}
				}
#endif
			}
			
			goto Skip_WS_And_Comments;
		}

		while( !isWhitespace( c))
		{
			if( c == '#')
			{
				break;
			}

			*pszToken++ = c;
			if( RC_BAD( rc = pFileHdl->Read( 0, 1, &c, &uiSize)))
			{
				goto Exit;
			}
		}

		// Put the char back
		
		if( RC_BAD( rc = pFileHdl->Tell( &uiOffset)))
		{
			goto Exit;
		}
		uiOffset--;

		if( RC_BAD( rc = pFileHdl->Seek( uiOffset, 
			F_IO_SEEK_SET, &uiTrueOffset)))
		{
			goto Exit;
		}
		break;
	}

Exit:

	*pszToken = '\0';
	return( rc);
}

/***************************************************************************
Desc:
****************************************************************************/
FilenameIterator::FilenameIterator(
	const char * 		pszPath, 
	const char * 		pszPrefix)
{

	f_strcpy( m_pszFullPrefix, pszPath);

#if defined( FLM_UNIX)
	f_strcat( m_pszFullPrefix, "/");
#elif defined( FLM_WIN) || defined( FLM_NLM)
	f_strcat( m_pszFullPrefix, "\\");
#else
	#error Platform not supported.
#endif

	f_strcat( m_pszFullPrefix, pszPrefix);
	this->reset();
}

/***************************************************************************
Desc:
****************************************************************************/
void FilenameIterator::getNext(
	char *			pszBuffer)
{
	// Note:  the maximum number of filenames in the sequence is
	// 16 ^ FILENAME_ITERATOR_MAX_EXTENSION_LENGTH.  We could check if
	// we've blown that here
	
	// Increment the extension, then produce the file
	
	m_uiExtension++;
	produceFilename( pszBuffer);
}

/***************************************************************************
Desc:
****************************************************************************/
void FilenameIterator::getCurrent(
	char *			pszBuffer)
{
	// Since the meaning of calling getCurrent before calling getNext is
	// not defined, give back a null
	
	if (m_uiExtension == (FLMUINT)FILENAME_ITERATOR_NULL)
	{
		f_strcpy( pszBuffer, "null");
	}
	else
	{
		produceFilename( pszBuffer);
	}
}

/***************************************************************************
Desc:
****************************************************************************/
void FilenameIterator::reset( void)
{
	// -1 is used as the initial, or null value.  The m_uiExtension is
	// incremented before it is used.
	
	m_uiExtension = (FLMUINT)FILENAME_ITERATOR_NULL;
}

/***************************************************************************
Desc:
****************************************************************************/
void FilenameIterator::produceFilename(
	char *			pszBuffer)
{
	char 		pszTemp[ FILENAME_ITERATOR_MAX_EXTENSION_LENGTH + 1];
	
	f_strcpy( pszBuffer, m_pszFullPrefix);
	f_strcat( pszBuffer, ".");
	f_sprintf( pszTemp, "%03x", (unsigned)m_uiExtension);
	f_strcat( pszBuffer, pszTemp);
}

/***************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_NLM
void flmUnittestFatalRuntimeError( void)
{
}
#endif

/***************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_NLM
int flmUnittestLoad( void)
{
	return (int)FlmStartup();
}
#endif

/***************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_NLM
int flmUnittestUnload( void)
{
	FlmShutdown();
	return 0;
}
#endif
