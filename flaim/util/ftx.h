//-------------------------------------------------------------------------
// Desc:	Cross-platform text user interface APIs - windowing - definitions.
// Tabs:	3
//
//		Copyright (c) 1996-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftx.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FTX_H
#define FTX_H

#include "flaimsys.h"

typedef FLMUINT					FTXRCODE;

#define FTXRC_SUCCESS			0
#define FTXRC_INVALID_WIN		1
#define FTXRC_MEM					2
#define FTXRC_INVALID_POS		3
#define FTXRC_NULL_POINTER		4
#define FTXRC_SHUTDOWN			5
#define FTXRC_EMPTY_LIST		6
#define FTXRC_BUF_OVERRUN		7
#define FTXRC_NO_INPUT			8
#define FTXRC_INVALID_SCREEN	9
#define FTXRC_ILLEGAL_OP		10

#define CV_BOFFSET				1
#define CV_KEYBUF_SIZE			64
#define CV_MAX_WINNAME_LEN		32


/* Colors */

enum WPS_COLORS {
		WPS_BLACK,
		WPS_BLUE,
		WPS_GREEN,
		WPS_CYAN,
		WPS_RED,
		WPS_MAGENTA,
		WPS_BROWN,
		WPS_LIGHTGRAY,
		WPS_DARKGRAY,
		WPS_LIGHTBLUE,
		WPS_LIGHTGREEN,
		WPS_LIGHTCYAN,
		WPS_LIGHTRED,
		WPS_LIGHTMAGENTA,
		WPS_YELLOW,
		WPS_WHITE
};

#define  WPS_DIM        0x1c           /* Turn bold OFF */
#define  WPS_HI         0x1d           /* Turn bold ON  */


/* Cursor Flags */

#define	WPS_CURSOR_BLOCK			0x01
#define	WPS_CURSOR_UNDERLINE		0x02
#define	WPS_CURSOR_INVISIBLE		0x04
#define	WPS_CURSOR_VISIBLE		0x08


/**---------------------------
***  WP Keyboard Definitions
***--------------------------*/

/* (WpkIncar & 0xFF) values - try to move to WPK values for consistency */

#define WPK_ESCAPE      0xE01B            /* Escape (ESC) */
#define WPK_ESC         WPK_ESCAPE
#define WPK_SPACE       0x20

/* Num Pad Keys */
#define WPK_HOME        0xE008            /* HOME key */
#define WPK_UP          0xE017            /* Up arrow */
#define WPK_PGUP        0xE059            /* Page Up */
#define WPK_LEFT        0xE019            /* Left arrow */
#define WPK_RIGHT       0xE018            /* Right arrow */
#define WPK_END         0xE055            /* END key */
#define WPK_DOWN        0xE01A            /* Down arrow */
#define WPK_PGDN        0xE05A            /* Page Down */

#define WPK_INSERT      0xE05D            /* Insert key */
#define WPK_DELETE      0xE051            /* Delete key */
#define WPK_BACKSPACE   0xE050            /* Backspace */
#define WPK_TAB         0xE009            /* TAB */

#define WPK_ENTER       0xE00a            /* Enter */
#define WPK_F1          0xE020            /* F1 */
#define WPK_F2          0xE021            /* F2 */
#define WPK_F3          0xE022            /* F3 */
#define WPK_F4          0xE023            /* F4 */
#define WPK_F5          0xE024            /* F5 */
#define WPK_F6          0xE025            /* F6 */
#define WPK_F7          0xE026            /* F7 */
#define WPK_F8          0xE027            /* F8 */
#define WPK_F9          0xE028            /* F9 */
#define WPK_F10         0xE029            /* F10 */
#define WPK_F11         0xE030            /* F11 */
#define WPK_F12         0xE031            /* F12 */

/*------------*/
/* Shift Keys */
/*------------*/

#define WPK_STAB        0xE05E            /* Shift TAB */

#define WPK_SF1         0xE02C            /* F1 */
#define WPK_SF2         0xE02D            /* F2 */
#define WPK_SF3         0xE02E            /* F3 */
#define WPK_SF4         0xE02F            /* F4 */
#define WPK_SF5         0xE030            /* F5 */
#define WPK_SF6         0xE031            /* F6 */
#define WPK_SF7         0xE032            /* F7 */
#define WPK_SF8         0xE033            /* F8 */
#define WPK_SF9         0xE034            /* F9 */
#define WPK_SF10        0xE035            /* F10 */
#define WPK_SF11        0xE036            /* F11 */
#define WPK_SF12        0xE037            /* F12 */

/*------------*/
/* Alt Keys   */
/*------------*/

#define WPK_ALT_A       0xFDDC
#define WPK_ALT_B       0xFDDD
#define WPK_ALT_C       0xFDDE
#define WPK_ALT_D       0xFDDF
#define WPK_ALT_E       0xFDE0
#define WPK_ALT_F       0xFDE1
#define WPK_ALT_G       0xFDE2
#define WPK_ALT_H       0xFDE3
#define WPK_ALT_I       0xFDE4
#define WPK_ALT_J       0xFDE5
#define WPK_ALT_K       0xFDE6
#define WPK_ALT_L       0xFDE7
#define WPK_ALT_M       0xFDE8
#define WPK_ALT_N       0xFDE9
#define WPK_ALT_O       0xFDEA
#define WPK_ALT_P       0xFDEB
#define WPK_ALT_Q       0xFDEC
#define WPK_ALT_R       0xFDED
#define WPK_ALT_S       0xFDEE
#define WPK_ALT_T       0xFDEF
#define WPK_ALT_U       0xFDF0
#define WPK_ALT_V       0xFDF1
#define WPK_ALT_W       0xFDF2
#define WPK_ALT_X       0xFDF3
#define WPK_ALT_Y       0xFDF4
#define WPK_ALT_Z       0xFDF5

#define WPK_ALT_1       0xFDF7            /* ALT 1 */
#define WPK_ALT_2       0xFDF8            /* ALT 2 */
#define WPK_ALT_3       0xFDF9            /* ALT 3 */
#define WPK_ALT_4       0xFDFA            /* ALT 4 */
#define WPK_ALT_5       0xFDFB            /* ALT 5 */
#define WPK_ALT_6       0xFDFC            /* ALT 6 */
#define WPK_ALT_7       0xFDFD            /* ALT 7 */
#define WPK_ALT_8       0xFDFE            /* ALT 8 */
#define WPK_ALT_9       0xFDFF            /* ALT 9 */
#define WPK_ALT_0       0xFDF6            /* ALT 0 */

#define WPK_ALT_MINUS   0xE061            /* ALT MINUS */
#define WPK_ALT_EQUAL   0xE06B            /* ALT EQUAL */

#define WPK_ALT_F1      0xE038            /* ALT F1 */
#define WPK_ALT_F2      0xE039            /* ALT F2 */
#define WPK_ALT_F3      0xE03A            /* ALT F3 */
#define WPK_ALT_F4      0xE03B            /* ALT F4 */
#define WPK_ALT_F5      0xE03C            /* ALT F5 */
#define WPK_ALT_F6      0xE03D            /* ALT F6 */
#define WPK_ALT_F7      0xE03E            /* ALT F7 */
#define WPK_ALT_F8      0xE03F            /* ALT F8 */
#define WPK_ALT_F9      0xE040            /* ALT F9 */
#define WPK_ALT_F10     0xE041            /* ALT F10 -F11,F12 NOT SUPPORTED*/

/*------------*/
/* CTRL Keys  */
/*------------*/

#define WPK_GOTO        0xE058            /* GOTO cntl-home */
#define WPK_CTRL_HOME   0xE058            /* CTRL Home */
#define WPK_CTRL_UP     0xE063            /* CTRL Up arrow */
#define WPK_CTRL_PGUP   0xE057            /* CTRL Page Up */

#define WPK_CTRL_LEFT   0xE054            /* CTRL Left arrow */
#define WPK_CTRL_RIGHT  0xE053            /* CTRL Right arrow */

#define WPK_CTRL_END    0xE00B            /* CTRL END */
#define WPK_CTRL_DOWN   0xE064            /* CTRL Down arrow */
#define WPK_CTRL_PGDN   0xE00C            /* CTRL Page Down */
#define WPK_CTRL_INSERT 0xE06E            /* CTRL Insert */
#define WPK_CTRL_DELETE 0xE06D            /* CTRL Delete */

#define WPK_CTRL_ENTER  0xE05F            /* CTRL Enter */

#define WPK_CTRL_A      0xE07C
#define WPK_CTRL_B      0xE07D
#define WPK_CTRL_C      0xE07E
#define WPK_CTRL_D      0xE07F
#define WPK_CTRL_E      0xE080
#define WPK_CTRL_F      0xE081
#define WPK_CTRL_G      0xE082
#define WPK_CTRL_H      0xE083
#define WPK_CTRL_I      0xE084
#define WPK_CTRL_J      0xE085
#define WPK_CTRL_K      0xE086
#define WPK_CTRL_L      0xE087
#define WPK_CTRL_M      0xE088
#define WPK_CTRL_N      0xE089
#define WPK_CTRL_O      0xE08A
#define WPK_CTRL_P      0xE08B
#define WPK_CTRL_Q      0xE08C
#define WPK_CTRL_R      0xE08D
#define WPK_CTRL_S      0xE08E
#define WPK_CTRL_T      0xE08F
#define WPK_CTRL_U      0xE090
#define WPK_CTRL_V      0xE091
#define WPK_CTRL_W      0xE092
#define WPK_CTRL_X      0xE093
#define WPK_CTRL_Y      0xE094
#define WPK_CTRL_Z      0xE095

#define WPK_CTRL_1      0xE06B            /* F1 - NOT SUPPORTED IN WP TO F10*/
#define WPK_CTRL_2      0xE06C            /* F2 */
#define WPK_CTRL_3      0xE06D            /* F3 */
#define WPK_CTRL_4      0xE06E            /* F4 */
#define WPK_CTRL_5      0xE06F            /* F5 */
#define WPK_CTRL_6      0xE070            /* F6 */
#define WPK_CTRL_7      0xE071            /* F7 */
#define WPK_CTRL_8      0xE072            /* F8 */
#define WPK_CTRL_9      0xE073            /* F9 */
#define WPK_CTRL_0      0xE074            /* F10 */

#define WPK_CTRL_MINUS  0xE060            /* MINUS */
#define WPK_CTRL_EQUAL  0xE061            /* EQUAL - NOT SUPPORTED IN WP */

#define WPK_CTRL_F1     0xE038            /* F1 */
#define WPK_CTRL_F2     0xE039            /* F2 */
#define WPK_CTRL_F3     0xE03A            /* F3 */
#define WPK_CTRL_F4     0xE03B            /* F4 */
#define WPK_CTRL_F5     0xE03C            /* F5 */
#define WPK_CTRL_F6     0xE03D            /* F6 */
#define WPK_CTRL_F7     0xE03E            /* F7 */
#define WPK_CTRL_F8     0xE03F            /* F8 */
#define WPK_CTRL_F9     0xE040            /* F9 */
#define WPK_CTRL_F10    0xE041            /* F10 */

// Forward declarations

struct FTX_INFO;
struct FTX_WINDOW;
struct FTX_SCREEN;

/* Type Definitions */

typedef struct nlm_char_info
{

	char				charValue;
	char				attribute;

} NLM_CHAR_INFO;


typedef FLMBOOL (* KEY_HANDLER_p)(
				FTX_INFO *					pFtxInfo,
				FLMUINT						uiKeyIn,
				FLMUINT *					puiKeyOut,
				void *						pvAppData);

typedef struct FTX_WINDOW
{
	FLMBYTE *		pucBuffer;
	FLMBYTE *		pucForeAttrib;
	FLMBYTE *		pucBackAttrib;
	FLMUINT			uiBackground;
	FLMUINT			uiForeground;
	FLMUINT			uiUlx;
	FLMUINT			uiUly;
	FLMUINT			uiCols;
	FLMUINT			uiRows;
	FLMUINT			uiCurX;
	FLMUINT			uiCurY;
	FLMUINT			uiOffset;
	FLMUINT			uiCursorType;
	char				pucName[ CV_MAX_WINNAME_LEN + 4];
	FLMBOOL			bScroll;
	FLMBOOL			bOpen;
	FLMBOOL			bInitialized;
	FLMBOOL			bForceOutput;
	FLMBOOL			bNoLineWrap;
	FLMUINT			uiId;
	FTX_WINDOW *	pWinPrev;
	FTX_WINDOW *	pWinNext;
	FTX_SCREEN *	pScreen;
} FTX_WINDOW;


typedef struct FTX_SCREEN
{
	F_MUTEX				hScreenMutex;
	F_SEM					hKeySem; /* Semaphore that will be signaled when
											keystrokes are available */
	FLMUINT				uiRows;
	FLMUINT				uiCols;
	FLMUINT				uiBackground;
	FLMUINT				uiForeground;
	FLMUINT				uiCursorType;
	char					pucName[ CV_MAX_WINNAME_LEN + 4];
	FLMBOOL				bInitialized;
	FLMBOOL				bChanged;
	FLMBOOL				bActive;
	FLMBOOL				bUpdateCursor;
	FLMUINT				uiSequence;
	FLMUINT				uiId;
	FLMBOOL *			pbShutdown;
	FTX_WINDOW *		pWinImage;
	FTX_WINDOW *		pWinScreen;
	FTX_WINDOW *		pWinCur;
	FTX_SCREEN *		pScreenPrev;
	FTX_SCREEN *		pScreenNext;
	FTX_INFO *			pFtxInfo;

} FTX_SCREEN;

typedef struct FTX_INFO
{
	F_MUTEX				hFtxMutex;
	F_Thread *			pBackgroundThrd;
	F_Thread *			pKeyboardThrd;
	F_Thread *			pDisplayThrd;
	KEY_HANDLER_p		pKeyHandler;
	void *				pvKeyHandlerData;
	FLMUINT				uiRows;
	FLMUINT				uiCols;
	FLMUINT				uiBackground;
	FLMUINT				uiForeground;
	FLMUINT				uiCursorType;
	FLMUINT				puiKeyBuffer[ CV_KEYBUF_SIZE];
	FLMUINT				uiCurKey;
	FLMUINT				uiSequence;
	FLMBOOL				bExiting;
	FLMBOOL				bScreenSwitch;
	FLMBOOL				bRefreshDisabled;
	FLMBOOL				bEnablePingChar;
	FLMBOOL *			pbShutdown;
	FTX_SCREEN *		pScreenCur;
#if defined( FLM_WIN) || defined( VC32) || defined( VC4)
	PCHAR_INFO   		pCells;
#elif defined( FLM_NLM)
	void *				pvScreenHandle;
	NLM_CHAR_INFO *	pCells;
#endif

} FTX_INFO;

FTXRCODE
	FTXWinPrintChar(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiChar);

FTXRCODE
	FTXWinPrintStr(
		FTX_WINDOW *	pWindow,
		const char *	pucString);

int FTXVSprintf(
	int				iMaxLen,
	char *			szDestStr,
	const char *	szFormatStr,
	f_va_list *		args);

FTXRCODE
	FTXWinPrintf(
		FTX_WINDOW *	pWindow,
		const char *	pucFormat, ...);

FTXRCODE
	FTXWinCPrintf(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiBackground,
		FLMUINT			uiForeground,
		const char *	pucFormat, ...);

FTXRCODE
	FTXWinPrintStrR(
		FTX_WINDOW *	pWindow,
		const char *	pucString);

FTXRCODE
	FTXWinPrintStrXY(
		FTX_WINDOW *	pWindow,
		const char *	pucString,
		FLMUINT			uiCol,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinPrintStrXYR(
		FTX_WINDOW *	pWindow,
		const char *	pucString,
		FLMUINT			uiCol,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinMove(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiCol,
		FLMUINT			uiRow);

FTXRCODE
	FTXInit(
		const char *	pucAppName,
		FLMUINT			uiCols,
		FLMUINT			uiRows,
		FLMUINT			uiBackground,
		FLMUINT			uiForeground,
		KEY_HANDLER_p	pKeyHandler,
		void *			pvKeyHandlerData,
		FTX_INFO **		ppFtxInfo);

FTXRCODE
	FTXFree(
		FTX_INFO **		ppFtxInfo);

FTXRCODE
	FTXCycleScreensNext(
		FTX_INFO *		pFtxInfo);

FTXRCODE
	FTXCycleScreensPrev(
		FTX_INFO *		pFtxInfo);
	
FTXRCODE
	FTXRefreshCursor(
		FTX_INFO *		pFtxInfo);

FTXRCODE
	FTXInvalidate(
		FTX_INFO *		pFtxInfo);

FTXRCODE
	FTXScreenInit(
		FTX_INFO *		pFtxInfo,
		const char *	pucName,
		FTX_SCREEN **	ppScreen);

FTXRCODE
	FTXScreenFree(
		FTX_SCREEN **	ppScreen);

FTXRCODE
	FTXScreenInitStandardWindows(
		FTX_SCREEN *	pScreen,
		FLMUINT			uiTitleBackColor,
		FLMUINT			uiTitleForeColor,
		FLMUINT			uiMainBackColor,
		FLMUINT			uiMainForeColor,
		FLMBOOL			bBorder,
		FLMBOOL			bBackFill,
		const char *	pucTitle,
		FTX_WINDOW **	ppTitleWin,
		FTX_WINDOW **	ppMainWin);

FTXRCODE
	FTXScreenSetShutdownFlag(
		FTX_SCREEN *	pScreen,
		FLMBOOL *		pbShutdownFlag);

FTXRCODE
	FTXCaptureScreen(
		FTX_INFO *			pFtxInfo,
		FLMBYTE *			pText,
		FLMBYTE *			pForeAttrib,
		FLMBYTE *			pBackAttrib);

FTXRCODE
	FTXRefresh(
		FTX_INFO *		pFtxInfo);

FTXRCODE
	FTXSetRefreshState(
		FTX_INFO *		pFtxInfo,
		FLMBOOL			bDisable);

FTXRCODE
	FTXAddKey(
		FTX_INFO *		pFtxInfo,
		FLMUINT			uiKey);

FTXRCODE
	FTXWinInit(
		FTX_SCREEN *	pScreen,
		FLMUINT 			uiCols,
		FLMUINT			uiRows,
		FTX_WINDOW **	ppWindow);

FTXRCODE
	FTXWinFree(
		FTX_WINDOW **	ppWindow);

FTXRCODE
	FTXWinOpen(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXWinSetName(
		FTX_WINDOW *	pWindow,
		const char *	pucName);

FTXRCODE
	FTXWinClose(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXWinSetFocus(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXScreenDisplay(
		FTX_SCREEN *	pScreen);

FTXRCODE
	FTXWinPaintBackground(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiBackground);

FTXRCODE
	FTXWinPaintForeground(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiForeground);

FTXRCODE
	FTXWinPaintRow(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiBackground,
		FLMUINT *		puiForeground,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinSetChar(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiChar);

FTXRCODE
	FTXWinPaintRowForeground(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiForeground,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinPaintRowBackground(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiBackground,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinSetBackFore(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiBackground,
		FLMUINT			uiForeground);

FTXRCODE
	FTXWinGetCanvasSize(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiNumCols,
		FLMUINT *		puiNumRows);

FTXRCODE
	FTXWinGetSize(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiNumCols,
		FLMUINT *		puiNumRows);

FTXRCODE
	FTXWinGetCurrRow(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiRow);

FTXRCODE
	FTXWinGetCurrCol(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiCol);

FTXRCODE
	FTXWinGetBackFore(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiBackground,
		FLMUINT *		puiForeground);

FTXRCODE
	FTXWinDrawBorder(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXWinSetTitle(
		FTX_WINDOW *	pWindow,
		const char *	pucTitle,
		FLMUINT			uiBackground,
		FLMUINT			uiForeground);

FTXRCODE
	FTXWinSetHelp(
		FTX_WINDOW *	pWindow,
		const char *	pszHelp,
		FLMUINT			uiBackground,
		FLMUINT			uiForeground);

FTXRCODE
	FTXLineEdit(
		FTX_WINDOW *	pWindow,
		char *   		pucBuffer,
		FLMUINT      	uiBufSize,
		FLMUINT      	uiMaxWidth,
		FLMUINT *		puiCharCount,
		FLMUINT *   	puiTermChar);

FLMUINT
	FTXLineEd(
		FTX_WINDOW *	pWindow,
		char *			pucBuffer,
		FLMUINT			uiBufSize);

FTXRCODE
	FTXWinSetCursorPos(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiCol,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinGetCursorPos(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiCol,
		FLMUINT *		puiRow);

FTXRCODE
	FTXWinClear(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXWinClearXY(
		FTX_WINDOW *	pWindow,
		FLMUINT 			uiCol,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinClearLine(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiCol,
		FLMUINT			uiRow);

FTXRCODE
	FTXWinClearToEOL(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXWinSetCursorType(
		FTX_WINDOW *	pWindow,
		FLMUINT			uiType);

FTXRCODE
	FTXWinGetCursorType(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiType);

FTXRCODE
	FTXScreenGetSize(
		FTX_SCREEN *	pScreen,
		FLMUINT *		puiNumCols,
		FLMUINT *		puiNumRows);

FTXRCODE
	FTXWinInputChar(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiChar);

FTXRCODE
	FTXWinTestKB(
		FTX_WINDOW *	pWindow);

FTXRCODE
	FTXWinSetScroll(
		FTX_WINDOW *	pWindow,
		FLMBOOL			bScroll);

FTXRCODE
	FTXWinSetLineWrap(
		FTX_WINDOW *	pWindow,
		FLMBOOL			bLineWrap);

FTXRCODE
	FTXWinGetScroll(
		FTX_WINDOW *	pWindow,
		FLMBOOL *		pbScroll);

FTXRCODE
	FTXWinGetScreen(
		FTX_WINDOW *	pWindow,
		FTX_SCREEN **	ppScreen);

FTXRCODE
	FTXWinGetPosition(
		FTX_WINDOW *	pWindow,
		FLMUINT *		puiCol,
		FLMUINT *		puiRow);

FTXRCODE
	FTXSetShutdownFlag(
		FTX_INFO *		pFtxInfo,
		FLMBOOL *		pbShutdownFlag);

FTXRCODE
	FTXMessageWindow(
		FTX_SCREEN *		pScreen,
		FLMUINT				uiBack,
		FLMUINT				uiFore,
		const char *		pucMessage1,
		const char *		pucMessage2,
		FTX_WINDOW **		ppWindow);

FTXRCODE
	FTXDisplayMessage(
		FTX_SCREEN *		pScreen,
		FLMUINT				uiBack,
		FLMUINT				uiFore,
		const char *		pucMessage1,
		const char *		pucMessage2,
		FLMUINT *			puiTermChar);

FTXRCODE
	FTXGetInput(
		FTX_SCREEN *	pScreen,
		const char *	pszMessage,
		char *			pszResponse,
		FLMUINT			uiMaxRespLen,
		FLMUINT *		puiTermChar);

FTXRCODE
	FTXEnablePingChar(
		FTX_INFO *			pFtxInfo);

/* Routines Necessary to Support Keyboard Manager Thread */

FLMBOOL
	ftxKBTest( void);
	
FLMUINT
	ftxKBGetChar( void);

/* Support for Audio */

FLMBOOL
	ftxBeep( void);

/* Specialized version of sprintf for handling unicode, etc. */

int FTXSprintf(
	int				iMaxLen,
	char *			szDestStr,
	const char *	szFormatStr, ...);
	
/*
Default handlers
*/

RCODE _ftxDefaultDisplayHandler(
	F_Thread *		pThread);

RCODE _ftxDefaultKeyboardHandler(
	F_Thread *		pThread);

/*
Misc.
*/

FTX_INFO * _getGlobalFtxInfo( void);

void debugFTXCycleScreens( void);

#endif
