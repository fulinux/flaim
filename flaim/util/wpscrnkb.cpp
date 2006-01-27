//-------------------------------------------------------------------------
// Desc:	Screen display routines for all platforms.
// Tabs:	3
//
//		Copyright (c) 1992,1994-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: wpscrnkb.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "ftx.h"
#include "wpscreen.h"
#include "sharutil.h"

FSTATIC FTX_WINDOW_p
	wpsGetThrdWin( void);
	
FSTATIC void
	wpsLock(
		F_MUTEX	*	phMutex
	);

FSTATIC void
	wpsUnlock(
		F_MUTEX	*	phMutex
	);

static FTX_INFO_p					gv_pFtxInfo = NULL;
static FLMBOOL						gv_bPrivateFTX = TRUE;
static FLMBOOL						gv_bShutdown = FALSE;
static FLMBOOL						gv_bInitialized = FALSE;
static FLMBOOL						gv_bOptimize = FALSE;
static WPSSCREEN_p				gv_pScreenList = NULL;
static F_MUTEX						gv_hDispMutex = F_MUTEX_NULL;

/****************************************************************************
Desc: Initialize and set the title
Ret:
****************************************************************************/
void 
	WpsInit(
		FLMUINT			rows,
		FLMUINT			cols,
		const char *	screenTitle)
{
	char	szTitleAndVer[ 100];

	if( gv_bInitialized)
	{
		return;
	}

	// Setup utilities title which includes the software version.
#ifdef SECURE_UTIL
	f_sprintf( (char *)szTitleAndVer, "%s - %s (%u)", 
						screenTitle, SRC_VER_STR, (unsigned)UTIL_VER); 
#else
	f_sprintf( (char *)szTitleAndVer, "%s - %s (UNSECURE:%u)", 
						screenTitle, SRC_VER_STR, (unsigned)UTIL_VER); 
#endif

	FTXInit( szTitleAndVer,
		(FLMBYTE)cols, (FLMBYTE)rows, WPS_BLACK, WPS_LIGHTGRAY, 
		NULL, NULL, &gv_pFtxInfo);

	if( RC_BAD( f_mutexCreate( &gv_hDispMutex)))
	{
		flmAssert( 0);
	}

	WpsThrdInit( szTitleAndVer);
	gv_bPrivateFTX = TRUE;
	gv_bInitialized = TRUE;
	return;
}


/****************************************************************************
Desc: Initialize WPS using an existing FTX environment
Ret:
****************************************************************************/
void 
	WpsInitFTX(
		FTX_INFO_p	pFtxInfo
	)
{
	if( gv_bInitialized)
	{
		return;
	}

	if( RC_BAD( f_mutexCreate( &gv_hDispMutex)))
	{
		flmAssert( 0);
	}

	gv_pFtxInfo = pFtxInfo;
	gv_bPrivateFTX = FALSE;
	gv_bInitialized = TRUE;
	return;
}


/****************************************************************************
Desc: Restores the screen to an initial state
Ret:
****************************************************************************/
void  WpsExit( void)
{
	if( !gv_bInitialized)
	{
		return;
	}

	gv_bShutdown = TRUE;
	WpsThrdExit();
	f_mutexDestroy( &gv_hDispMutex);
	if( gv_bPrivateFTX == TRUE)
	{
		FTXFree( &gv_pFtxInfo);
	}
	gv_bInitialized = FALSE;
	return;
}


/****************************************************************************
Desc: Initialize and set the title of a thread's screen
Ret:
****************************************************************************/
void 
	WpsThrdInitUsingScreen(
		FTX_SCREEN_p	pFtxScreen,
		const char *	screenTitle)
{
	FLMUINT			uiRows;
	FLMUINT			uiThrdId;
	WPSSCREEN_p		pCurScreen = NULL;


	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();
	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen == NULL)
	{
		if( RC_BAD( f_calloc( sizeof( WPSSCREEN), &pCurScreen)))
		{
			flmAssert( 0);
		}
		pCurScreen->uiScreenId = uiThrdId;
		pCurScreen->pNext = gv_pScreenList;
		gv_pScreenList = pCurScreen;

		if( pFtxScreen != NULL)
		{
			pCurScreen->pScreen = pFtxScreen;
			pCurScreen->bPrivate = FALSE;
		}
		else
		{
			if( FTXScreenInit( gv_pFtxInfo, screenTitle,
				&(pCurScreen->pScreen)) != FTXRC_SUCCESS)
			{
				flmAssert( 0);
			}
			pCurScreen->bPrivate = TRUE;
		}

		if( FTXScreenGetSize( pCurScreen->pScreen, NULL,
			&uiRows) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}
	
		if( FTXWinInit( pCurScreen->pScreen, 0,
			(FLMBYTE)(uiRows - 1), &(pCurScreen->pWin)) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}

		if( FTXWinMove( pCurScreen->pWin, 0, 1) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}

		if( FTXWinInit( pCurScreen->pScreen, 0,
			1, &(pCurScreen->pTitleWin)) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}

		if( FTXWinPaintBackground( pCurScreen->pTitleWin,
			WPS_RED) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}

		if( FTXWinPrintStr( pCurScreen->pTitleWin,
			screenTitle) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}

		if( FTXWinOpen( pCurScreen->pTitleWin) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}

		if( FTXWinOpen( pCurScreen->pWin) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
		}
	}

	wpsUnlock( &gv_hDispMutex);

	return;
}


/****************************************************************************
Desc: Frees all screen resources allocated to a thread
Ret:
****************************************************************************/
void 
	WpsThrdExit( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN_p		pPrevScreen = NULL;
	WPSSCREEN_p		pCurScreen = NULL;


	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();
	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pPrevScreen = pCurScreen;
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen != NULL)
	{
		if( pCurScreen == gv_pScreenList)
		{
			gv_pScreenList = pCurScreen->pNext;
		}
		
		if( pPrevScreen != NULL)
		{
			pPrevScreen->pNext = pCurScreen->pNext;
		}

		if( pCurScreen->bPrivate == TRUE)
		{
			if( FTXScreenFree( &(pCurScreen->pScreen)) != FTXRC_SUCCESS)
			{
				flmAssert( 0);
			}
		}
		else
		{
			if( FTXWinFree( &(pCurScreen->pTitleWin)) != FTXRC_SUCCESS)
			{
				flmAssert( 0);
			}

			if( FTXWinFree( &(pCurScreen->pWin)) != FTXRC_SUCCESS)
			{
				flmAssert( 0);
			}
		}

		f_free( &pCurScreen);
	}

	wpsUnlock( &gv_hDispMutex);

	return;
}


/****************************************************************************
Desc: Returns the size of the screen in columns and rows.
****************************************************************************/
void 
	WpsScrSize(
	FLMUINT *	puiNumColsRV,
	FLMUINT *	puiNumRowsRV
	)
{
	FTXWinGetCanvasSize( wpsGetThrdWin(), puiNumColsRV, puiNumRowsRV);
}


/****************************************************************************
Desc:    Output a string at present cursor location.
Ret:
Notes:   The old windows way is
			TextOut( hdc, 0, 0, string, strlen( string));
****************************************************************************/
FLMINT 
	WpsStrOut(
		const char *		string
	)
{
	FTXWinPrintStr( wpsGetThrdWin(), string);
	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return( (FLMINT)0);
}


/****************************************************************************
Desc:    Output a formatted string at present cursor location.
Ret:
****************************************************************************/
FLMINT 
	WpsPrintf(
		const char *	pucFormat, ...)
{
	char			pucBuffer[ 512];
	f_va_list	args;

	f_va_start( args, pucFormat);
	f_vsprintf( (char *)pucBuffer, pucFormat, &args);
	f_va_end( args);
	FTXWinPrintStr( wpsGetThrdWin(), pucBuffer);

	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return( (FLMINT)0);
}


/****************************************************************************
Desc:    Output a formatted string at present cursor location with color
Ret:
****************************************************************************/
FLMINT 
	WpsCPrintf(
		FLMUINT			uiBack,
		FLMUINT			uiFore,
		const char *	pucFormat, ...)
{
	char			pucBuffer[ 512];
	f_va_list	args;
	FLMUINT		uiOldBack;
	FLMUINT		uiOldFore;

	f_va_start( args, pucFormat);
	f_vsprintf( (char *)pucBuffer, pucFormat, &args);
	f_va_end( args);

	FTXWinGetBackFore( wpsGetThrdWin(), &uiOldBack, &uiOldFore);
	FTXWinSetBackFore( wpsGetThrdWin(), uiBack, uiFore);
	FTXWinPrintStr( wpsGetThrdWin(), pucBuffer);
	FTXWinSetBackFore( wpsGetThrdWin(), uiOldBack, uiOldFore);

	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return( (FLMINT)0);
}


/****************************************************************************
Desc: Output a character to the screen at the current location. If char is
		a LineFeed then a CarriageReturn will be inserted before the LineFeed.
Ret:  none
Notes:On NLM becomes a blocking function if the char is the newline character.
****************************************************************************/
void 
	WpsChrOut(
		FLMBYTE		chr
	)
{
	FTXWinPrintChar( wpsGetThrdWin(), (FLMBYTE)chr);
	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return;
}


/****************************************************************************
Desc:    Clear the screen from the col/row down
Ret:
Notes:   If col==row==0 then clear entire screen
****************************************************************************/
void 
	WpsScrClr(
		FLMUINT		col,
		FLMUINT		row
	)
{
	FTX_WINDOW_p	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	
	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( col == 255)
	{
		col = uiCurrCol;
	}

	if( row == 255)
	{
		row = uiCurrRow;
	}

	FTXWinClearXY( pThrdWin, col, row);
	FTXWinSetCursorPos( pThrdWin, col, row);
	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return;
}


/****************************************************************************
Desc:    Position to the column and row specified.
Notes:   The NLM could call GetPositionOfOutputCursor(&r,&c);
****************************************************************************/
void 
	WpsScrPos(              /* Position to col/row on screen */
		FLMUINT		col,
		FLMUINT		row
	)
{
	FTX_WINDOW_p	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;
	
	
	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( col == 255)
	{
		col = uiCurrCol;
	}

	if( row == 255)
	{
		row = uiCurrRow;
	}

	FTXWinSetCursorPos( pThrdWin, col, row);
	
	return;
}


/****************************************************************************
Desc:    Clear from input cursor to end of line
****************************************************************************/
void 
	WpsLineClr(
		FLMUINT		col,
		FLMUINT		row
	)
{
	FTX_WINDOW_p	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	
	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( col == 255)
	{
		col = uiCurrCol;
	}

	if( row == 255)
	{
		row = uiCurrRow;
	}

	FTXWinClearLine( pThrdWin, col, row);
	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return;
}


/****************************************************************************
Desc:    Edit a line of data like gets(s).  Newline replaced by NULL character.
Ret:     WPK Character
Notes:   Does not support WP extended character input - but could easily!
****************************************************************************/
FLMUINT 
	WpsLineEd(
		char *		string,
		FLMUINT		maxLen,
		FLMBOOL *	pbShutdown
	)
{
	FLMUINT		uiCharCount;
	FLMUINT		uiCursorType;
	

	FTXWinGetCursorType( wpsGetThrdWin(), &uiCursorType);
	FTXWinSetCursorType( wpsGetThrdWin(), WPS_CURSOR_UNDERLINE);
	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
	uiCharCount = FTXLineEd( wpsGetThrdWin(), string, (FLMUINT)maxLen);
	FTXSetShutdownFlag( gv_pFtxInfo, NULL);
	FTXWinSetCursorType( wpsGetThrdWin(), uiCursorType);

	return( uiCharCount);
}


/****************************************************************************
Desc:    Sets the FTX shutdown flag pointer
Ret:
****************************************************************************/
void
	WpsSetShutdown(
		FLMBOOL *    pbShutdown
			// [IN] Pointer to the global shutdown variable
	)
{
	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
}


/****************************************************************************
Desc:    Edit a line of data with advanced features.
Ret:     Number of characters input.
****************************************************************************/
FLMUINT
	WpsLineEditExt(
		char *		pbyBuffer,
		FLMUINT      uiBufSize,
		FLMUINT		 uiMaxWidth,
		FLMBOOL *    pbShutdown,
		FLMUINT *    puiTermChar)
{
	FLMUINT		uiCharCount = 0;
	FLMUINT		uiCursorType;

	FTXWinGetCursorType( wpsGetThrdWin(), &uiCursorType);
	FTXWinSetCursorType( wpsGetThrdWin(), WPS_CURSOR_UNDERLINE);
	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
	FTXLineEdit( wpsGetThrdWin(), pbyBuffer, uiBufSize, uiMaxWidth,
		&uiCharCount, puiTermChar);
	FTXSetShutdownFlag( gv_pFtxInfo, NULL);
	FTXWinSetCursorType( wpsGetThrdWin(), uiCursorType);

	return( (FLMINT)uiCharCount);
}


/****************************************************************************
Desc:    Get the current X coordinate of the cursor
Ret:     FLMINT value of the current cursor column
****************************************************************************/
FLMUINT 
	WpsCurrCol( void)
{
	FLMUINT		uiCol;


	FTXWinGetCursorPos( wpsGetThrdWin(), &uiCol, NULL);
	return( uiCol);
}


/****************************************************************************
Desc:    Get the current Y coordinate of the cursor
Ret:     FLMINT value of the current cursor row
****************************************************************************/
FLMUINT 
	WpsCurrRow( void)
{
	FLMUINT		uiRow;


	FTXWinGetCursorPos( wpsGetThrdWin(), NULL, &uiRow);
	return( uiRow);
}


/****************************************************************************
Desc:    Set the background and foreground colors
Ret:     None
****************************************************************************/
void 
	WpsScrBackFor(
		FLMUINT		background,
		FLMUINT		forground
	)
{
	FTXWinSetBackFore( wpsGetThrdWin(), background, forground);
	return;
}


/****************************************************************************
Desc : Sets the cursor attributes.
****************************************************************************/
FLMBOOL 
	WpsCursorSetType(
		FLMUINT		uiType
	)
{
	FTXWinSetCursorType( wpsGetThrdWin(), uiType);
	FTXRefresh( gv_pFtxInfo);
	return( TRUE);
}


/****************************************************************************
Desc:    Specifies that display performance (throughput) should be
			optimal.
****************************************************************************/
void 
	WpsOptimize( void)
{
	gv_bOptimize = TRUE;
}


/****************************************************************************
Desc: Draws a border around the current thread's screen
Ret:  none
****************************************************************************/
void 
	WpsDrawBorder( void)
{
	FTXWinDrawBorder( wpsGetThrdWin());
	if( gv_bOptimize == FALSE)
	{
		FTXRefresh( gv_pFtxInfo);
	}

	return;
}


/****************************************************************************
Desc:    Convert keyboard sequences/scan codes to WPK key strokes.
Ret:     WPK Character
Notes:   Does not support WP extended character input - but could easily!
****************************************************************************/
FLMUINT 
	WpkIncar( void)
{
	FLMUINT		uiChar;
	

	FTXWinInputChar( wpsGetThrdWin(), &uiChar);
	return( uiChar);
}


/****************************************************************************
Desc:    Convert keyboard sequences/scan codes to WPK key strokes.  This
			routine accepts a pointer to a shutdown flag.
Ret:     WPK Character
****************************************************************************/
FLMUINT 
	WpkGetChar(
		FLMBOOL *		pbShutdown
	)
{
	FLMUINT		uiChar;
	

	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
	FTXWinInputChar( wpsGetThrdWin(), &uiChar);
	FTXSetShutdownFlag( gv_pFtxInfo, NULL);

	return( uiChar);
}


/****************************************************************************
Desc:    Tests the keyboard for a pending character
Ret:		1 if key available, 0 if no key available
****************************************************************************/
FLMUINT 
	WpkTestKB( void)
{
	FLMUINT		uiCharAvail;
	

	uiCharAvail = (FLMUINT)(FTXWinTestKB( wpsGetThrdWin()) ==
		FTXRC_SUCCESS ? 1 : 0);
	return( uiCharAvail);
}


/****************************************************************************
Desc:		Returns a pointer to a thread's screen
****************************************************************************/
FTX_SCREEN_p
	WpsGetThrdScreen( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN_p		pCurScreen = NULL;

	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();

	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen == NULL)
	{
		flmAssert( 0);
	}

	wpsUnlock( &gv_hDispMutex);
	return( pCurScreen->pScreen);
}

/****************************************************************************
Desc:		Locks the specified semaphore
****************************************************************************/
FSTATIC void
	wpsLock(
		F_MUTEX	*	phMutex
	)
{
	f_mutexLock( *phMutex);
}


/****************************************************************************
Desc:		Unlocks the specified semaphore
****************************************************************************/
FSTATIC void
	wpsUnlock(
		F_MUTEX	*	phMutex
	)
{
	f_mutexUnlock( *phMutex);
}


/****************************************************************************
Desc:		Returns a pointer to a thread's screen
****************************************************************************/
FSTATIC FTX_WINDOW_p
	wpsGetThrdWin( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN_p		pCurScreen = NULL;


	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();
	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen == NULL)
	{
		flmAssert( 0);
	}

	wpsUnlock( &gv_hDispMutex);
	return( pCurScreen->pWin);
}

