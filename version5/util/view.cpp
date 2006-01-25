//------------------------------------------------------------------------------
// Desc: This file is the main for the database view utility
//
// Tabs:	3
//
//		Copyright (c) 1992-1995, 1997-2006 Novell, Inc. All Rights Reserved.
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
// $Id: view.cpp 3119 2006-01-19 13:39:12 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#define MAIN_MODULE

#include "view.h"

#define UTIL_ID	"VIEW"

// Main Menu options

#define MAIN_MENU_DB_HEADER		1
#define MAIN_MENU_LOGICAL_FILES  2

// Local function prototypes

FSTATIC void ViewShowHelp(
	FLMBOOL			bShowFullUsage);

FSTATIC FLMUINT ViewGetChar(
	const char *	pszMessage1,
	const char *	pszMessage2,
	FLMUINT			uiDefaultChar);

FSTATIC FLMBOOL ViewGetFileName(
	FLMUINT			uiCol,
	FLMUINT			uiRow,
	FLMBOOL			bDispOnly);

FSTATIC FLMBOOL ViewOpenFile( void);

FSTATIC void ViewDoMainMenu( void);

FSTATIC FLMBOOL ViewSetupMainMenu( void);

FSTATIC FLMBOOL ViewOpenFileDirect( void);

#ifdef FLM_NLM
	FLMBOOL						gv_bSynchronized = FALSE;
	extern "C"
	{
	int nlm_main(
		int		iArgC,
		char **	ppszArgV);
	}

	FSTATIC void viewCleanup( void);
#endif

static FLMBOOL		bPauseBeforeExiting = FALSE;
FLMUINT				gv_uiTopLine = 0;
FLMUINT				gv_uiBottomLine = 0;

/********************************************************************
Desc: ?
*********************************************************************/
#if defined (FLM_UNIX)
int main(
	int			iArgC,
	char **		ppszArgV)
#elif defined( FLM_NLM)
int nlm_main(
	int			iArgC,
	char **		ppszArgV)
#else
int __cdecl main(
	int			iArgC,
	char **		ppszArgV)
#endif
{
#define MAX_ARGS     30
	FLMUINT		uiArg;
	FLMINT		iArgCnt = (FLMINT)iArgC;
	F_DbSystem	dbSystem;
	char *		ppszArgs [MAX_ARGS];
	char			szCommandBuffer [300];
	RCODE			rc = NE_XFLM_OK;

	if( RC_BAD( rc = dbSystem.init()))
	{
		ViewShowRCError( "calling dbsystem.init()", rc);
		goto Exit;
	}

	// Setup defaults for fixing the file header if necessary

	gv_ViewFixOptions.uiBlockSize = XFLM_DEFAULT_BLKSIZ;
	gv_ViewFixOptions.uiVersionNum = XFLM_CURRENT_VERSION_NUM;
	gv_ViewFixOptions.uiMinRflFileSize = XFLM_DEFAULT_MIN_RFL_FILE_SIZE;
	gv_ViewFixOptions.uiMaxRflFileSize = XFLM_DEFAULT_MAX_RFL_FILE_SIZE;
	gv_ViewFixOptions.bKeepRflFiles = XFLM_DEFAULT_KEEP_RFL_FILES_FLAG;
	gv_ViewFixOptions.bLogAbortedTransToRfl = XFLM_DEFAULT_LOG_ABORTED_TRANS_FLAG;
	gv_ViewFixOptions.uiDefaultLanguage = XFLM_DEFAULT_LANG;

	// See if a file name was passed in

	gv_szViewFileName [0] = '\0';
	gv_szDataDir [0] = 0;
	gv_szRflDir [0] = 0;
	gv_szPassword [0] = 0;
	gv_bViewExclusive = FALSE;
	gv_bViewFileOpened = FALSE;
	gv_bViewHdrRead = FALSE;
	gv_bViewHaveDictInfo = FALSE;
	gv_bShutdown = FALSE;
	gv_bRunning = TRUE;
	gv_pSFileHdl = NULL;

#ifdef FLM_NLM

	// Setup the routines to be called when the NLM exits itself

	atexit( viewCleanup);

#endif

#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif
	WpsInit( 0xFFFF, 0xFFFF, "FLAIM Database Viewer");
	WpsOptimize();

	WpsScrSize( NULL, &gv_uiBottomLine);
	gv_uiTopLine = 2;
	gv_uiBottomLine -= 3;

	// Ask the user to enter parameters if none were entered on the command
	// line.

	if (iArgCnt < 2)
	{
		for (;;)
		{
			WpsStrOut( "\nView Params (enter ? for help): ");
			szCommandBuffer [0] = 0;
			WpsLineEd( szCommandBuffer,
				sizeof( szCommandBuffer) - 1, &gv_bShutdown);
			if (f_stricmp( szCommandBuffer, "?") == 0)
			{
				ViewShowHelp( FALSE);
			}
			else
			{
				break;
			}
			if (gv_bShutdown)
			{
				goto Exit;
			}
		}
		flmUtilParseParams( szCommandBuffer, MAX_ARGS, &iArgCnt, &ppszArgs [1]);
		ppszArgs [0] = ppszArgV [0];
		iArgCnt++;
		ppszArgV = &ppszArgs [0];
	}

	uiArg = 1;
	while (uiArg < (FLMUINT)iArgCnt)
	{
#ifdef FLM_UNIX
		if (ppszArgV [uiArg][0] == '-')
#else
		if ((ppszArgV [uiArg][0] == '/') || (ppszArgV [uiArg][0] == '-'))
#endif
		{
			switch( ppszArgV [uiArg][1])
			{
				case 'x':
				case 'X':
					gv_bViewExclusive = TRUE;
					break;
				case 'b':
				case 'B':
					gv_ViewFixOptions.uiBlockSize =
						f_atoi( &ppszArgV [uiArg][2]);
					break;
				case 'd':
				case 'D':
					switch (ppszArgV [uiArg][2])
					{
						case 'r':
						case 'R':
							f_strcpy( gv_szRflDir, &ppszArgV [uiArg][3]);
							break;
						case 'd':
						case 'D':
							f_strcpy( gv_szDataDir, &ppszArgV [uiArg][3]);
							break;
						default:
							break;
					}
					break;
				case 'l':
				case 'L':
					gv_ViewFixOptions.uiMaxRflFileSize =
						f_atol( &ppszArgV [uiArg][2]);
					break;
				case 'm':
				case 'M':
					gv_ViewFixOptions.uiMinRflFileSize =
						f_atol( &ppszArgV [uiArg][2]);
					break;
				case 'p':
				case 'P':
					switch( ppszArgV [uiArg][2])
					{
						case 0:
							bPauseBeforeExiting = TRUE;
							break;
						case 'w':
						case 'W':
							if ( ppszArgV [uiArg][3])
							{
								f_strcpy( gv_szPassword, &ppszArgV [uiArg][3]);
							}
							else
							{
								ViewShowHelp( TRUE);
								bPauseBeforeExiting = TRUE;
								goto Exit;
							}
							break;
						default:
							break;
					}
					break;
				case '?':
					ViewShowHelp( TRUE);
					bPauseBeforeExiting = TRUE;
					goto Exit;
				default:
					break;
			}
		}
		else if (f_stricmp( ppszArgV [uiArg], "?") == 0)
		{
			ViewShowHelp( TRUE);
			bPauseBeforeExiting = TRUE;
			goto Exit;
		}
		else if (!gv_szViewFileName [0])
		{
			f_strcpy( gv_szViewFileName, ppszArgV [uiArg]);
		}
		uiArg++;
	}

	gv_ViewPool.poolInit(2048);
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, 0);

	// Open the file

	if (ViewOpenFile())
	{

		// Execute the main menu

		ViewDoMainMenu();
		ViewFreeMenuMemory();

		// Close the file

		if (gv_bViewDbInitialized)
		{
			if (RC_BAD ( rc = gv_hViewDb->transAbort( )))
			{
				ViewShowRCError( "calling transAbort()", rc);
				goto Exit;
			}

			gv_hViewDb->Release();
		}
	}

Exit:
	if (gv_pSFileHdl)
	{
		gv_pSFileHdl->Release();
		gv_pSFileHdl = NULL;
	}

	if ((bPauseBeforeExiting) && (!gv_bShutdown))
	{
		WpsStrOut( "\nPress any character to exit VIEW: ");
		for (;;)
		{
			if (gv_bShutdown)
			{
				break;
			}
			if (WpkTestKB())
			{
				(void)WpkIncar();
				break;
			}
			viewGiveUpCPU();
		}
	}

	WpsExit();
	dbSystem.exit();

#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif
	gv_bRunning = FALSE;
	return 0;
}

/********************************************************************
Desc:	Display a help screen.
*********************************************************************/
FSTATIC void ViewShowHelp(
	FLMBOOL	bShowFullUsage
	)
{
#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
#endif
	WpsStrOut( "\n");
	if (bShowFullUsage)
	{
		WpsStrOut( "Usage: view <DbName> [Options]\n");
	}
	else
	{
		WpsStrOut( "Parameters: <DbName> [Options]\n\n");
	}
	WpsStrOut( 
"   DbName   = Name of database to view.\n");
	WpsStrOut( 
"   Options  =\n");
	WpsStrOut( 
"        -dr<Dir>     = RFL directory.\n");
	WpsStrOut( 
"        -dd<Dir>     = Data directory.\n");
	WpsStrOut( 
"        -x           = Open database in exclusive mode.\n");
	WpsStrOut( 
"        -f           = Fix database header.  If the options below are not set,\n");
	WpsStrOut( 
"                       defaults will be used.\n");
	WpsStrOut( 
"        -b<Size>     = Set block size to Size (only used if -f is specified).\n");
	WpsStrOut( 
"        -m<Size>     = Set minimum RFL file size to Size (only used if -f\n");
	WpsStrOut( 
"                       option is used).\n");
	WpsStrOut( 
"        -l<Size>     = Set maximum RFL file size to Size (only used if -f\n");
	WpsStrOut( 
"                       option is used).\n");
	WpsStrOut( 
"                       used).\n");
	WpsStrOut( 
"        -p           = Pause before exiting.\n");
	WpsStrOut( 
"        -pw<Passwd>  = Database password.\n");
	WpsStrOut( 
"        -?           = A '?' anywhere in the command line will cause this\n");
	WpsStrOut( 
"                       screen to be displayed, with or without the leading '-'.\n");
	WpsStrOut( 
"Options may be specified anywhere in the command line.\n");
}

/***************************************************************************
Desc: Prompt user for a single character response and get the response.
*****************************************************************************/
FSTATIC FLMUINT ViewGetChar(
	const char *	pszMessage1,
	const char *	pszMessage2,
	FLMUINT			uiDefaultChar)
{
	FLMUINT	uiChar;
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;

	WpsScrSize( &uiNumCols, &uiNumRows);
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, uiNumRows - 2);
	WpsScrBackFor( WPS_RED, WPS_WHITE);
	if (pszMessage1)
	{
		WpsStrOutXY( pszMessage1, 0, uiNumRows - 2);
	}
	WpsStrOutXY( pszMessage2, 0, 23);
	for (;;)
	{
		if (gv_bShutdown)
		{
			uiChar = WPK_ESCAPE;
			break;
		}
		else if (WpkTestKB())
		{
			uiChar = (FLMUINT)WpkIncar();
			break;
		}
		viewGiveUpCPU();
	}
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, uiNumRows - 2);
	if (uiChar == WPK_ENTER)
	{
		uiChar = uiDefaultChar;
	}
	if (uiChar >= 'a' && uiChar <= 'z')
	{
		uiChar = uiChar - 'a' + 'A';
	}
	return( uiChar);
}

/***************************************************************************
Desc:	This routine reads and verifies the information contained in the
		file header and log header of a FLAIM database.
*****************************************************************************/
FINLINE RCODE ViewReadAndVerifyHdrInfo(
	FLMUINT32 *			pui32CalcCRC
	)

{
	return( flmGetHdrInfo( gv_pSFileHdl, &gv_ViewDbHdr, pui32CalcCRC));
}

/***************************************************************************
Desc: Read the header information from the database -- this includes
		the file header and the log header.
*****************************************************************************/
void ViewReadHdr(
	FLMUINT32 *	pui32CalcCRC
	)
{
	RCODE		rc;
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;

	WpsScrSize( &uiNumCols, &uiNumRows);

	gv_bViewHdrRead = TRUE;
	if (RC_OK( rc = ViewReadAndVerifyHdrInfo( pui32CalcCRC)))
	{
		return;
	}

	// Had some sort of error

	ViewShowRCError( "reading header information", rc);

	// Make sure we have a valid block size

	if (!F_DbSystem::validBlockSize((FLMUINT)gv_ViewDbHdr.ui16BlockSize))
	{
		gv_ViewDbHdr.ui16BlockSize = XFLM_DEFAULT_BLKSIZ;
	}
}

/********************************************************************
Desc:	Ask for input from the user
*********************************************************************/
void ViewAskInput(
	const char *	pszPrompt,
	const char *	pszBuffer,
	FLMUINT			uiBufLen)
{
	char	szTempBuf [80];

	WpsStrOut( pszPrompt);
	if (uiBufLen > sizeof( szTempBuf))
	{
		uiBufLen = sizeof( szTempBuf);
	}
	szTempBuf [0] = 0;
	WpsLineEd( szTempBuf, uiBufLen, &gv_bShutdown);
	f_strcpy( pszBuffer, szTempBuf);
}

/***************************************************************************
Desc:	This routine asks the user for the file name to be viewed.
*****************************************************************************/
FSTATIC FLMBOOL ViewGetFileName(
	FLMUINT		uiCol,
	FLMUINT		uiRow,
	FLMBOOL		bDispOnly)
{
	const char *	pszPrompt = "Enter database file name: ";

	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( uiCol, uiRow);
	if (bDispOnly)
	{
		WpsStrOutXY( pszPrompt, uiCol, uiRow);
		WpsStrOutXY( gv_szViewFileName,
							uiCol + f_strlen( pszPrompt), uiRow);
	}
	else
	{
		WpsScrPos( uiCol, uiRow);
		ViewAskInput( pszPrompt, gv_szViewFileName, 40);
		if (!gv_szViewFileName [0] ||
			 f_strcmp( gv_szViewFileName, "\\") == 0)
		{
			return( FALSE);
		}
	}
	return( TRUE);
}

/****************************************************************************
Desc: This routine opens a database file in DIRECT mode - because we couldn't
		get it open by calling the normal FLAIM functions.
****************************************************************************/
FSTATIC FLMBOOL ViewOpenFileDirect( void)
{
	RCODE				rc;
	IF_FileHdl *	pCFileHdl;

	if (RC_BAD( rc = gv_pSFileHdl->GetFileHdl( 0, FALSE, &pCFileHdl)))
	{
		ViewShowRCError( "opening file in direct mode", rc);
		return( FALSE);
	}

	gv_bViewFileOpened = TRUE;
	return( TRUE);
}

/***************************************************************************
Desc:	This routine opens the database file which is to be viewed.
*****************************************************************************/
FSTATIC FLMBOOL ViewOpenFile( void)
{
	RCODE			rc;
	FLMBOOL		bOk = FALSE;
	F_DbSystem	dbSystem;

Get_File_Name:

	// Prompt for file name if necessary

 	WpsScrClr( 0, 1);
	if (!gv_szViewFileName [0])
	{
		if (!ViewGetFileName( 5, 5, FALSE))
		{
			goto Exit;
		}
	}
	else
	{
		if (!ViewGetFileName( 5, 5, TRUE))
		{
			goto Exit;
		}
	}
	if (gv_pSFileHdl)
	{
		gv_pSFileHdl->Release();
		gv_pSFileHdl = NULL;
	}

	if ((gv_pSFileHdl = f_new F_SuperFileHdl) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ViewShowRCError( "creating super file handle", rc);
		goto Exit;
	}
	if (RC_BAD( rc = gv_pSFileHdl->Setup( NULL, gv_szViewFileName, gv_szDataDir)))
	{
		ViewShowRCError( "setting up super file handle", rc);
		goto Exit;
	}

	rc = ViewReadAndVerifyHdrInfo( NULL);

	gv_pSFileHdl->ReleaseFiles(TRUE);
	gv_pSFileHdl->SetBlockSize( (FLMUINT)gv_ViewDbHdr.ui16BlockSize);
	if (RC_BAD( rc))
	{
		if (rc == NE_XFLM_IO_PATH_NOT_FOUND)
		{
			goto Path_Not_Found;
		}
		else
		{
			goto Other_Error;
		}
	}

	if (RC_BAD( rc = dbSystem.openDb( gv_szViewFileName, gv_szDataDir,
										 gv_szRflDir, gv_szPassword, XFLM_DONT_REDO_LOG,
										 (IF_Db **)&gv_hViewDb)))
	{
		char		szTBuf[ 100];

		if (rc == NE_XFLM_IO_PATH_NOT_FOUND)
		{
Path_Not_Found:
			if (ViewGetChar( NULL,
				"File not found, try another file name? (Y/N, Default=Y): ",
				'Y') != 'Y')
			{
				goto Exit;
			}
			gv_szViewFileName [0] = 0;
			goto Get_File_Name;
		}
		else
		{
Other_Error:
			f_strcpy( szTBuf, "Error opening file: ");
			f_strcpy( &szTBuf [f_strlen( szTBuf)], F_DbSystem::_errorString( rc));
			if (ViewGetChar( szTBuf,
				"Open file in DIRECT MODE anyway? (Y/N, Default=Y): ",
				'Y') == 'Y')
			{
				if (!ViewOpenFileDirect())
				{
					goto Exit;
				}
			}
			else
			{
				goto Exit;
			}
		}
	}
	else
	{
		gv_bViewDbInitialized = TRUE;
//		if (RC_BAD ( rc = gv_hViewDb->transBegin( XFLM_READ_TRANS)))
//		{
//			ViewShowRCError( "calling transBegin()", rc);
//			goto Exit;
//		}
		if (!ViewOpenFileDirect())
		{
			goto Exit;
		}
	}

	bOk = TRUE;

Exit:

	if (!bOk)
	{
		if (gv_pSFileHdl)
		{
			gv_pSFileHdl->Release();
			gv_pSFileHdl = NULL;
		}
		gv_bViewFileOpened = FALSE;
	}
	return( bOk);
}

/***************************************************************************
Desc:	This routine gets the dictionary information for a database and
		locks it into memory.
*****************************************************************************/
RCODE ViewGetDictInfo( void)
{
	RCODE			rc = NE_XFLM_OK;
	F_Db *		pDb = gv_hViewDb;
//	FLMUINT		uiSaveFlags;
	FLMUINT		uiFlags;

	if (gv_bViewDbInitialized)
	{

		// If we have a transaction going, abort it and start another one.

		if (pDb->getTransType() != XFLM_NO_TRANS)
		{
			pDb->transAbort();
		}

		// Need to fake out flmBeginDbTrans to avoid an assert.
		// This may be the first time we read in a dictionary due to the
		// fact that we did not do recovery and rollback.  flmBeginDbTrans
		// expects the DBF_BEING_OPENED flag to be set the first time
		// a dictionary is read in.  Otherwise, it will assert.

//		uiSaveFlags = pDb->pFile->uiFlags;
//		pDb->pFile->uiFlags |= DBF_BEING_OPENED;
		uiFlags = DBF_BEING_OPENED;  // VISIT:  This needs the other flags...

		// Start a read transaction.
		if (RC_BAD( rc = pDb->transBegin( XFLM_READ_TRANS, XFLM_NO_TIMEOUT,
			uiFlags, NULL)))
		{
			gv_bViewHaveDictInfo = FALSE;
			goto Exit;
		}
		gv_bViewHaveDictInfo = TRUE;
//		pDb->pFile->uiFlags = uiSaveFlags;
	}

Exit:

	return rc;
}

/***************************************************************************
Desc:	This routine sets up the main menu for the VIEW program.
*****************************************************************************/
FSTATIC FLMBOOL ViewSetupMainMenu( void)
{
	FLMBOOL	bOk = FALSE;
	FLMUINT	uiRow;
	FLMUINT	uiCol;

	// Initialize the menu structures

	ViewMenuInit( "Main Menu");
	uiRow = 3;
	uiCol = 20;

	// Add each menu item to the menu

	if (!ViewAddMenuItem( LBL_DB_HEADER, 0,
												VAL_IS_EMPTY, 0, 0,
												0, 0xFFFFFFFF, 0, MOD_DISABLED,
												uiCol, uiRow++, MAIN_MENU_DB_HEADER,
												WPS_BLACK, WPS_WHITE,
												WPS_BLUE, WPS_WHITE))
	{
		goto Exit;
	}

	if (gv_ViewDbHdr.ui32FirstLFBlkAddr == 0)
	{
		if (!ViewAddMenuItem( LBL_LOGICAL_FILES, 0,
									VAL_IS_LABEL_INDEX, (FLMUINT)LBL_NONE, 0,
									0, 0xFFFFFFFF, 0, MOD_DISABLED,
									uiCol, uiRow++, 0,
									WPS_BLACK, WPS_LIGHTGRAY,
									WPS_BLUE, WPS_LIGHTGRAY))
		{
			goto Exit;
		}
	}
	else
	{
		if (!ViewAddMenuItem( LBL_LOGICAL_FILES, 0,
									VAL_IS_EMPTY, 0, 0,
									0, 0xFFFFFFFF, 0, MOD_DISABLED,
									uiCol, uiRow++, MAIN_MENU_LOGICAL_FILES,
									WPS_BLACK, WPS_WHITE,
									WPS_BLUE, WPS_WHITE))
		{
			goto Exit;
		}
	}
	bOk = TRUE;

Exit:

	return( bOk);
}

/***************************************************************************
Desc: This routine executes the main menu of the VIEW program.  From here
		the user may view various parts of the database until he presses
		the ESC key.
*****************************************************************************/
FSTATIC void ViewDoMainMenu( void)
{
	FLMUINT		uiOption;
	VIEW_INFO	SaveView;
	FLMBOOL		bRepaint = TRUE;
	FLMUINT		uiBlkAddress;
	BLK_EXP		BlkExp;

	// Loop getting commands until the ESC key is pressed

	ViewReset( &SaveView);
	for( ;;)
	{

		// Redisplay the main menu each time, because the other options will
		// have destroyed the menu.

		if (gv_bViewPoppingStack)
		{
			if (!gv_bViewHdrRead)
			{
				ViewReadHdr();
			}
			ViewSearch();
		}
		if (bRepaint)
		{
			if (!ViewSetupMainMenu())
			{
				return;
			}
		}
		bRepaint = TRUE;
		uiOption = ViewGetMenuOption();
		switch (uiOption)
		{
			case ESCAPE_OPTION:
				return;
			case MAIN_MENU_DB_HEADER:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				ViewDbHeader();
				break;
			case MAIN_MENU_LOGICAL_FILES:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				ViewLogicalFiles();
				break;
			case SEARCH_OPTION:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				gv_uiViewSearchLfNum = XFLM_DATA_COLLECTION;
				gv_uiViewSearchLfType = XFLM_LF_COLLECTION;
				if (ViewGetKey())
				{
					ViewSearch();
				}
				break;
			case GOTO_BLOCK_OPTION:
				if (!gv_bViewHdrRead)
				{
					ViewReadHdr();
				}
				if (GetBlockAddrType( &uiBlkAddress))
				{
					BlkExp.uiType = 0xFF;
					BlkExp.uiLevel = 0xFF;
					BlkExp.uiNextAddr = 0xFFFFFFFF;
					BlkExp.uiPrevAddr = 0xFFFFFFFF;
					BlkExp.uiLfNum = 0;
					ViewBlocks( uiBlkAddress, uiBlkAddress, &BlkExp);
				}
				else
				{
					bRepaint = FALSE;
				}
				break;
			case EDIT_OPTION:
			default:
				bRepaint = FALSE;
				break;
		}
	}
}

#ifdef FLM_NLM
/****************************************************************************
Desc: This routine shuts down all threads in the VIEW NLM.
****************************************************************************/
FSTATIC void viewCleanup(
	void
	)
{
	gv_bShutdown = TRUE;
	while (gv_bRunning)
	{
		viewGiveUpCPU();
	}
}
#endif

