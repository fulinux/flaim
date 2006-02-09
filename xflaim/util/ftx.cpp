//------------------------------------------------------------------------------
// Desc:	This file contains functions for supporting a cross-platform
//			text-based user interface.
//
// Tabs:	3
//
//		Copyright (c) 1996-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftx.cpp 3140 2006-01-25 13:52:51 -0700 (Wed, 25 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"
#include "ftx.h"

#if defined( FLM_WIN)

	typedef struct
	{
		unsigned char LeadChar;
		unsigned char SecondChar;
	} ftxWin32CharPair;

	typedef struct
	{
		unsigned short ScanCode;
		ftxWin32CharPair RegChars;
		ftxWin32CharPair ShiftChars;
		ftxWin32CharPair CtrlChars;
		ftxWin32CharPair AltChars;
	} ftxWin32EnhKeyVals;

	typedef struct
	{
		ftxWin32CharPair RegChars;
		ftxWin32CharPair ShiftChars;
		ftxWin32CharPair CtrlChars;
		ftxWin32CharPair AltChars;
	} ftxWin32NormKeyVals;

	/*
	 * Table of key values for enhanced keys
	 */
	static ftxWin32EnhKeyVals ftxWin32EnhancedKeys[] = {
        { 28, {  13,   0 }, {  13,   0 }, {  10,   0 }, {   0, 166 } },
        { 53, {  47,   0 }, {  63,   0 }, {   0, 149 }, {   0, 164 } },
        { 71, { 224,  71 }, { 224,  71 }, { 224, 119 }, {   0, 151 } },
        { 72, { 224,  72 }, { 224,  72 }, { 224, 141 }, {   0, 152 } },
        { 73, { 224,  73 }, { 224,  73 }, { 224, 134 }, {   0, 153 } },
        { 75, { 224,  75 }, { 224,  75 }, { 224, 115 }, {   0, 155 } },
        { 77, { 224,  77 }, { 224,  77 }, { 224, 116 }, {   0, 157 } },
        { 79, { 224,  79 }, { 224,  79 }, { 224, 117 }, {   0, 159 } },
        { 80, { 224,  80 }, { 224,  80 }, { 224, 145 }, {   0, 160 } },
        { 81, { 224,  81 }, { 224,  81 }, { 224, 118 }, {   0, 161 } },
        { 82, { 224,  82 }, { 224,  82 }, { 224, 146 }, {   0, 162 } },
        { 83, { 224,  83 }, { 224,  83 }, { 224, 147 }, {   0, 163 } }
        };

	/*
	 * macro for the number of elements of in EnhancedKeys[]
	 */
	#define FTX_WIN32_NUM_EKA_ELTS    (sizeof( ftxWin32EnhancedKeys) / sizeof( ftxWin32EnhKeyVals))

	/*
	 * Table of key values for normal keys. Note that the table is padded so
	 * that the key scan code serves as an index into the table.
	 */

	static ftxWin32NormKeyVals ftxWin32NormalKeys[] = {

        /* padding */
        { /*  0 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /*  1 */ {  27,   0 }, {  27,   0 }, {  27,   0 }, {   0,   1 } },
        { /*  2 */ {  49,   0 }, {  33,   0 }, {   0,   0 }, {   0, 120 } },
        { /*  3 */ {  50,   0 }, {  64,   0 }, {   0,   3 }, {   0, 121 } },
        { /*  4 */ {  51,   0 }, {  35,   0 }, {   0,   0 }, {   0, 122 } },
        { /*  5 */ {  52,   0 }, {  36,   0 }, {   0,   0 }, {   0, 123 } },
        { /*  6 */ {  53,   0 }, {  37,   0 }, {   0,   0 }, {   0, 124 } },
        { /*  7 */ {  54,   0 }, {  94,   0 }, {  30,   0 }, {   0, 125 } },
        { /*  8 */ {  55,   0 }, {  38,   0 }, {   0,   0 }, {   0, 126 } },
        { /*  9 */ {  56,   0 }, {  42,   0 }, {   0,   0 }, {   0, 127 } },
        { /* 10 */ {  57,   0 }, {  40,   0 }, {   0,   0 }, {   0, 128 } },
        { /* 11 */ {  48,   0 }, {  41,   0 }, {   0,   0 }, {   0, 129 } },
        { /* 12 */ {  45,   0 }, {  95,   0 }, {  31,   0 }, {   0, 130 } },
        { /* 13 */ {  61,   0 }, {  43,   0 }, {   0,   0 }, {   0, 131 } },
        { /* 14 */ {   8,   0 }, {   8,   0 }, { 127,   0 }, {   0,  14 } },
        { /* 15 */ {   9,   0 }, {   0,  15 }, {   0, 148 }, {   0,  15 } },
        { /* 16 */ { 113,   0 }, {  81,   0 }, {  17,   0 }, {   0,  16 } },
        { /* 17 */ { 119,   0 }, {  87,   0 }, {  23,   0 }, {   0,  17 } },
        { /* 18 */ { 101,   0 }, {  69,   0 }, {   5,   0 }, {   0,  18 } },
        { /* 19 */ { 114,   0 }, {  82,   0 }, {  18,   0 }, {   0,  19 } },
        { /* 20 */ { 116,   0 }, {  84,   0 }, {  20,   0 }, {   0,  20 } },
        { /* 21 */ { 121,   0 }, {  89,   0 }, {  25,   0 }, {   0,  21 } },
        { /* 22 */ { 117,   0 }, {  85,   0 }, {  21,   0 }, {   0,  22 } },
        { /* 23 */ { 105,   0 }, {  73,   0 }, {   9,   0 }, {   0,  23 } },
        { /* 24 */ { 111,   0 }, {  79,   0 }, {  15,   0 }, {   0,  24 } },
        { /* 25 */ { 112,   0 }, {  80,   0 }, {  16,   0 }, {   0,  25 } },
        { /* 26 */ {  91,   0 }, { 123,   0 }, {  27,   0 }, {   0,  26 } },
        { /* 27 */ {  93,   0 }, { 125,   0 }, {  29,   0 }, {   0,  27 } },
        { /* 28 */ {  13,   0 }, {  13,   0 }, {  10,   0 }, {   0,  28 } },

        /* padding */
        { /* 29 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 30 */ {  97,   0 }, {  65,   0 }, {   1,   0 }, {   0,  30 } },
        { /* 31 */ { 115,   0 }, {  83,   0 }, {  19,   0 }, {   0,  31 } },
        { /* 32 */ { 100,   0 }, {  68,   0 }, {   4,   0 }, {   0,  32 } },
        { /* 33 */ { 102,   0 }, {  70,   0 }, {   6,   0 }, {   0,  33 } },
        { /* 34 */ { 103,   0 }, {  71,   0 }, {   7,   0 }, {   0,  34 } },
        { /* 35 */ { 104,   0 }, {  72,   0 }, {   8,   0 }, {   0,  35 } },
        { /* 36 */ { 106,   0 }, {  74,   0 }, {  10,   0 }, {   0,  36 } },
        { /* 37 */ { 107,   0 }, {  75,   0 }, {  11,   0 }, {   0,  37 } },
        { /* 38 */ { 108,   0 }, {  76,   0 }, {  12,   0 }, {   0,  38 } },
        { /* 39 */ {  59,   0 }, {  58,   0 }, {   0,   0 }, {   0,  39 } },
        { /* 40 */ {  39,   0 }, {  34,   0 }, {   0,   0 }, {   0,  40 } },
        { /* 41 */ {  96,   0 }, { 126,   0 }, {   0,   0 }, {   0,  41 } },

        /* padding */
        { /* 42 */ {    0,  0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 43 */ {  92,   0 }, { 124,   0 }, {  28,   0 }, {   0,   0 } },
        { /* 44 */ { 122,   0 }, {  90,   0 }, {  26,   0 }, {   0,  44 } },
        { /* 45 */ { 120,   0 }, {  88,   0 }, {  24,   0 }, {   0,  45 } },
        { /* 46 */ {  99,   0 }, {  67,   0 }, {   3,   0 }, {   0,  46 } },
        { /* 47 */ { 118,   0 }, {  86,   0 }, {  22,   0 }, {   0,  47 } },
        { /* 48 */ {  98,   0 }, {  66,   0 }, {   2,   0 }, {   0,  48 } },
        { /* 49 */ { 110,   0 }, {  78,   0 }, {  14,   0 }, {   0,  49 } },
        { /* 50 */ { 109,   0 }, {  77,   0 }, {  13,   0 }, {   0,  50 } },
        { /* 51 */ {  44,   0 }, {  60,   0 }, {   0,   0 }, {   0,  51 } },
        { /* 52 */ {  46,   0 }, {  62,   0 }, {   0,   0 }, {   0,  52 } },
        { /* 53 */ {  47,   0 }, {  63,   0 }, {   0,   0 }, {   0,  53 } },

        /* padding */
        { /* 54 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 55 */ {  42,   0 }, {   0,   0 }, { 114,   0 }, {   0,   0 } },

        /* padding */
        { /* 56 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 57 */ {  32,   0 }, {  32,   0 }, {  32,   0 }, {  32,   0 } },

        /* padding */
        { /* 58 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 59 */ {   0,  59 }, {   0,  84 }, {   0,  94 }, {   0, 104 } },
        { /* 60 */ {   0,  60 }, {   0,  85 }, {   0,  95 }, {   0, 105 } },
        { /* 61 */ {   0,  61 }, {   0,  86 }, {   0,  96 }, {   0, 106 } },
        { /* 62 */ {   0,  62 }, {   0,  87 }, {   0,  97 }, {   0, 107 } },
        { /* 63 */ {   0,  63 }, {   0,  88 }, {   0,  98 }, {   0, 108 } },
        { /* 64 */ {   0,  64 }, {   0,  89 }, {   0,  99 }, {   0, 109 } },
        { /* 65 */ {   0,  65 }, {   0,  90 }, {   0, 100 }, {   0, 110 } },
        { /* 66 */ {   0,  66 }, {   0,  91 }, {   0, 101 }, {   0, 111 } },
        { /* 67 */ {   0,  67 }, {   0,  92 }, {   0, 102 }, {   0, 112 } },
        { /* 68 */ {   0,  68 }, {   0,  93 }, {   0, 103 }, {   0, 113 } },

        /* padding */
        { /* 69 */ {    0,  0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },
        { /* 70 */ {    0,  0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 71 */ {   0,  71 }, {  55,   0 }, {   0, 119 }, {   0,   0 } },
        { /* 72 */ {   0,  72 }, {  56,   0 }, {   0, 141 }, {   0,   0 } },
        { /* 73 */ {   0,  73 }, {  57,   0 }, {   0, 132 }, {   0,   0 } },
        { /* 74 */ {   0,   0 }, {  45,   0 }, {   0,   0 }, {   0,   0 } },
        { /* 75 */ {   0,  75 }, {  52,   0 }, {   0, 115 }, {   0,   0 } },
        { /* 76 */ {   0,   0 }, {  53,   0 }, {   0,   0 }, {   0,   0 } },
        { /* 77 */ {   0,  77 }, {  54,   0 }, {   0, 116 }, {   0,   0 } },
        { /* 78 */ {   0,   0 }, {  43,   0 }, {   0,   0 }, {   0,   0 } },
        { /* 79 */ {   0,  79 }, {  49,   0 }, {   0, 117 }, {   0,   0 } },
        { /* 80 */ {   0,  80 }, {  50,   0 }, {   0, 145 }, {   0,   0 } },
        { /* 81 */ {   0,  81 }, {  51,   0 }, {   0, 118 }, {   0,   0 } },
        { /* 82 */ {   0,  82 }, {  48,   0 }, {   0, 146 }, {   0,   0 } },
        { /* 83 */ {   0,  83 }, {  46,   0 }, {   0, 147 }, {   0,   0 } },

        /* padding */
        { /* 84 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },
        { /* 85 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },
        { /* 86 */ {   0,   0 }, {   0,   0 }, {   0,   0 }, {   0,   0 } },

        { /* 87 */ { 224, 133 }, { 224, 135 }, { 224, 137 }, { 224, 139 } },
        { /* 88 */ { 224, 134 }, { 224, 136 }, { 224, 138 }, { 224, 140 } }

	};

	static HANDLE								gv_hStdOut;
	static HANDLE								gv_hStdIn;
	static FLMBOOL								gv_bAllocatedConsole = FALSE;
	static CONSOLE_SCREEN_BUFFER_INFO	gv_ConsoleScreenBufferInfo;

	FSTATIC FLMUINT ftxWin32KBGetChar( void);

	FSTATIC ftxWin32CharPair * ftxWin32GetExtendedKeycode(
		KEY_EVENT_RECORD *	pKE);

	static int chbuf = -1;

#elif defined( FLM_UNIX)

	#if defined( FLM_HPUX) || defined( FLM_OSF)
		#define _XOPEN_CURSES
		#define _XOPEN_SOURCE_EXTENDED 1
	#endif
	
	// curses.h pollutes name spaces like crazy; these definitions
	// are required to get the code to compile cleanly on all platforms.
	
	#if defined( bool)
		#undef bool
	#endif
	
	#if defined( EO)
		#undef EO
	#endif
	
	#if defined( ERR)
		#undef ERR
	#endif
	
	#if defined( FLM_SOLARIS)
		#define _WIDEC_H
	#endif
	
	#include <curses.h>
	
	#ifdef FLM_AIX
		#ifdef wgetch
			#undef wgetch
		#endif
		
		extern "C"
		{
			extern int wgetch( WINDOW *);
			extern int clear( void);
		}
	#endif
	
	static int ungetChar;
	
	// Curses gives us only a limited number of color pairs.  We use this
	// static color_pairs array for only the colors we need. flm2curses is
	// used to convert from flaim colors to curses colors and last_pair
	// is the last_pair is the last pair that we used.
	
	static short flm2curses[8];
	static short color_pairs[8][8];
	short last_pair = 0;

	FSTATIC void ftxUnixDisplayChar(
		FLMUINT			uiChar,
		FLMUINT			uiAttr);
		
	FSTATIC void ftxUnixDisplayRefresh( void);
	
	FSTATIC void ftxUnixDisplayReset( void);

	FSTATIC void ftxUnixDisplayInit( void);
	
	FSTATIC void ftxUnixDisplayFree( void);
	
	FSTATIC void ftxUnixDisplayGetSize(
		FLMUINT *		puiNumColsRV,
		FLMUINT * 		puiNumRowsRV);
		
	FSTATIC void ftxUnixDisplaySetCursorPos(
		FLMUINT			uiCol,
		FLMUINT			uiRow);
		
	FSTATIC FLMUINT ftxUnixKBGetChar( void);
		
	FSTATIC FLMBOOL ftxUnixKBTest( void);
	
#endif

static FLMBOOL			gv_bInitialized = FALSE;
static FLMBOOL			gv_bDisplayInitialized = FALSE;
static FLMUINT			gv_uiInitCount = 0;
static FTX_INFO *		gv_pFtxInfo = NULL;
static FLMBOOL			gv_bPrivateFTX = TRUE;
static FLMBOOL			gv_bShutdown = FALSE;
static FLMBOOL			gv_bOptimize = FALSE;
static WPSSCREEN *	gv_pScreenList = NULL;
static F_MUTEX			gv_hDispMutex = F_MUTEX_NULL;

FSTATIC FTX_WINDOW * wpsGetThrdWin( void);

#if defined( FLM_WIN)

FSTATIC FTXRCODE
	ftxWin32Refresh(
		FTX_INFO *		pFtxInfo);

#elif defined( FLM_NLM)

	extern "C"
	{
		LONG AllocateResourceTag(
			LONG		pvLoadRecord,
			BYTE *	pvResourceDescriptionString,
			LONG		ResourceSignature);
		
		#ifndef ScreenSignature
			#define ScreenSignature		0x4E524353			/* 'NRCS' */
		#endif
		
		int OpenScreen(
			void *	pvScreenName,
			void *	pvResourceTag,
			void **	pvScreenHandle);
		
		void CloseScreen(
			void *	pvScreenHandle);
		
		void ActivateScreen(
			void *	pvScreenHandle);
		
		void ClearScreen(
			void *	pvScreenHandle);
		
		void GetScreenSize(
			WORD *	swScreenHeight,
			WORD *	swScreenWidth);
		
		void PositionInputCursor(
			void *	pvScreenHandle,
			WORD		swRow,
			WORD		swColumn);
		
		void EnableInputCursor(
			void *	pvScreenHandle);
		
		void DisableInputCursor(
			void *	pvScreenHandle);
		
		LONG PositionOutputCursor(
			void *	pvScreenHandle,
			WORD		swRow,
			WORD		swColumn);
		
		void GetOutputCursorPosition(
			void *	pvScreenHandle,
			WORD *	row,
			WORD *	column);
		
		void SetInputToOutputCursorPosition(
			void *	pvScreenHandle);
		
		void GetKey(
			void *	pvScreenHandle,
			BYTE *	pucKeyType,
			BYTE *	pucKeyValue,
			BYTE *	pucKeyStatus,
			BYTE *	pucScanCode,
			LONG		sdLinesToProtect);
		
		LONG UngetKey(
				struct ScreenStruct *screenID,
				BYTE keyType,
				BYTE keyValue,
				BYTE keyStatus,
				BYTE scanCode);
		
		LONG CheckKeyStatus(
			void *	pvScreenHandle);
		
		LONG DisplayScreenTextWithAttribute(
			void *	pvScreenHandleD,
			LONG		sdLine,
			LONG		sdColumn,
			LONG		sdLength,
			BYTE		ucLineAttribute,
			BYTE *	pszText);
		
		LONG WriteScreenCharacterAttribute(
			void *	pvScreenHandle,
			LONG		sdLine,
			LONG		sdColumn,
			BYTE		ucCharacter,
			BYTE		ucAttribute);
		
		void SetCursorStyle(
			void *	pvScreenHandle,
			WORD		swNewCursorStyle);
	
	}	// extern "C"

	FSTATIC FTXRCODE ftxNLMRefresh(
		FTX_INFO *		pFtxInfo);

#else

	FSTATIC FTXRCODE ftxRefresh(
		FTX_INFO *		pFtxInfo);

#endif

FSTATIC FTXRCODE ftxSyncImage(
	FTX_INFO *		pFtxInfo);

FSTATIC FTXRCODE ftxWinReset(
	FTX_WINDOW *	pWindow);

FSTATIC FTXRCODE ftxCursorUpdate(
	FTX_INFO *		pFtxInfo);

FSTATIC FTXRCODE ftxWinPrintChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar);

FINLINE void ftxKeyboardFlush(
	FTX_INFO *		pFtxInfo
	)
{
	pFtxInfo->uiCurKey = 0;
	f_memset( pFtxInfo->puiKeyBuffer, (FLMBYTE)0,
		sizeof( FLMUINT) * CV_KEYBUF_SIZE);
}

FSTATIC FTXRCODE ftxWinClearLine(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow);

FSTATIC FTXRCODE ftxWinClose(
	FTX_WINDOW *	pWindow);

FSTATIC FTXRCODE ftxWinFree(
	FTX_WINDOW *	pWindow);

FSTATIC FTXRCODE ftxWinOpen(
	FTX_WINDOW *	pWindow);

FSTATIC FTXRCODE ftxScreenFree(
	FTX_SCREEN *	pScreen);

FSTATIC FTXRCODE ftxWinSetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow);

FSTATIC FTXRCODE ftxDisplayInit(
	FTX_INFO *		pFtxInfo,
	FLMUINT			uiRows,
	FLMUINT			uiCols,
	const char *	pszTitle);

FSTATIC void ftxDisplayReset(
	FTX_INFO *		pFtxInfo);

FSTATIC void ftxDisplayGetSize(
	FLMUINT *		puiNumColsRV,
	FLMUINT *		puiNumRowsRV);

FSTATIC FLMBOOL ftxDisplaySetCursorType(
	FTX_INFO *		pFtxInfo,
	FLMUINT			uiType);

FSTATIC void ftxDisplayExit( void);

FSTATIC void ftxDisplaySetCursorPos(
	FTX_INFO *		pFtxInfo,
	FLMUINT			uiCol,
	FLMUINT			uiRow);

FSTATIC void ftxDisplaySetBackFore(
	FLMUINT			uiBackground,
	FLMUINT			uiForeground);

RCODE _ftxBackgroundThread(
	F_Thread *		pThread);

#if defined( FLM_UNIX)
FSTATIC FLMUINT ftxDisplayStrOut(
	const char *	pszString,
	FLMUINT			uiAttribute);
#endif

/* Scan Code Conversion Tables */

#if defined( FLM_WIN) || defined( FLM_NLM)
static FLMUINT ScanCodeToWPK[] = {
	0,             0,             0,             0,             /* 00..03 */
	0,             0,             0,             0,             /* 04 */
	0,             0,             0,             0,             /* 08 */
	0,             0,             0,             WPK_STAB,      /* 0C */
	WPK_ALT_Q,     WPK_ALT_W,     WPK_ALT_E,     WPK_ALT_R,     /* 10 */
	WPK_ALT_T,     WPK_ALT_Y,     WPK_ALT_U,     WPK_ALT_I,     /* 14 */
	WPK_ALT_O,     WPK_ALT_P,     0,             0,             /* 18 */
	0,             0,             WPK_ALT_A,     WPK_ALT_S,     /* 1C */
	WPK_ALT_D,     WPK_ALT_F,     WPK_ALT_G,     WPK_ALT_H,     /* 20 */
	WPK_ALT_J,     WPK_ALT_K,     WPK_ALT_L,     0,             /* 24 */
	0,             0,             0,             0,             /* 28 */
	WPK_ALT_Z,     WPK_ALT_X,     WPK_ALT_C,     WPK_ALT_V,     /* 2C */
	WPK_ALT_B,     WPK_ALT_N,     WPK_ALT_M,     0,             /* 30 */
	0,             0,             0,             0,             /* 34 */
	0,             0,             0,             WPK_F1,        /* 38 */
	WPK_F2,        WPK_F3,        WPK_F4,        WPK_F5,        /* 3C */
	WPK_F6,        WPK_F7,        WPK_F8,        WPK_F9,        /* 40 */
										/* F8 MAY BE BAD*/
	WPK_F10,       WPK_F11,       WPK_F12,       WPK_HOME,      /* 44 */
	WPK_UP,        WPK_PGUP,      0,             WPK_LEFT,      /* 48 */
	0,             WPK_RIGHT,     0,             WPK_END,       /* 4C */
	WPK_DOWN,      WPK_PGDN,      WPK_INSERT,    WPK_DELETE,    /* 50 */

	WPK_SF1,       WPK_SF2,       WPK_SF3,       WPK_SF4,       /* 54 */
	WPK_SF5,       WPK_SF6,       WPK_SF7,       WPK_SF8,       /* 58 */
	WPK_SF9,       WPK_SF10,      WPK_CTRL_F1,   WPK_CTRL_F2,   /* 5C */
	WPK_CTRL_F3,   WPK_CTRL_F4,   WPK_CTRL_F5,   WPK_CTRL_F6,   /* 60 */
	WPK_CTRL_F7,   WPK_CTRL_F8,   WPK_CTRL_F9,   WPK_CTRL_F10,  /* 64 */

	WPK_ALT_F1,    WPK_ALT_F2,    WPK_ALT_F3,    WPK_ALT_F4,    /* 68 */
	WPK_ALT_F5,    WPK_ALT_F6,    WPK_ALT_F7,    WPK_ALT_F8,    /* 6C */
	WPK_ALT_F9,    WPK_ALT_F10,   0,             WPK_CTRL_LEFT, /* 70 */
	WPK_CTRL_RIGHT,WPK_CTRL_END,  WPK_CTRL_PGDN, WPK_CTRL_HOME, /* 74 */

	WPK_CTRL_1,    WPK_CTRL_2,    WPK_CTRL_3,    WPK_CTRL_4,    /* 78 */
	WPK_CTRL_5,    WPK_CTRL_6,    WPK_CTRL_7,    WPK_CTRL_8,    /* 7C */
	WPK_CTRL_9,    WPK_CTRL_0,    WPK_CTRL_MINUS,WPK_CTRL_EQUAL,/* 80 */
	WPK_CTRL_PGUP, 0,             0,             0,             /* 84 */
	0,					0,					0,					0,					/* 88 */
	0,					WPK_CTRL_UP,	0,					0,					/* 8C */
	0,					WPK_CTRL_DOWN,	0,					0					/* 90 */
};
#endif

#ifdef FLM_NLM
	void *	g_pvScreenTag;
#endif

/****************************************************************************
Desc:		Initializes the FTX environment.
****************************************************************************/
FTXRCODE FTXInit(
	const char *	pszAppName,
	FLMUINT			uiCols,
	FLMUINT			uiRows,
	FLMUINT			uiBackground,
	FLMUINT			uiForeground,
	KEY_HANDLER 	pKeyHandler,
	void *			pvKeyHandlerData,
	FTX_INFO **		ppFtxInfo)
{
	FTX_INFO *		pFtxInfo;
	FTXRCODE			rc = FTXRC_SUCCESS;

	*ppFtxInfo = NULL;

	if( gv_bInitialized)
	{
		gv_uiInitCount++;
		*ppFtxInfo = gv_pFtxInfo;
		goto Exit;
	}

	if( RC_BAD( f_calloc( sizeof( FTX_INFO), &pFtxInfo)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}
	gv_pFtxInfo = pFtxInfo;

	if( RC_BAD( f_mutexCreate( &(pFtxInfo->hFtxMutex))))
	{
		rc = FTXRC_MEM;
	}

#ifdef FLM_NLM

		g_pvScreenTag = (void *)AllocateResourceTag(
				(LONG)f_getNLMHandle(),
				(BYTE *)"Screen", (LONG)ScreenSignature);

		(void)OpenScreen( pszAppName,
			g_pvScreenTag, &pFtxInfo->pvScreenHandle);
		ActivateScreen( pFtxInfo->pvScreenHandle);

#endif

	if( (rc = ftxDisplayInit( pFtxInfo, uiRows, uiCols,
		pszAppName)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	ftxDisplayReset( pFtxInfo);
	ftxDisplayGetSize( &(pFtxInfo->uiCols), &(pFtxInfo->uiRows));

	if( pFtxInfo->uiCols > uiCols)
	{
		pFtxInfo->uiCols = uiCols;
	}

	if( pFtxInfo->uiRows > uiRows)
	{
		pFtxInfo->uiRows = uiRows;
	}

	pFtxInfo->uiCursorType = WPS_CURSOR_INVISIBLE;
	ftxDisplaySetCursorType( pFtxInfo, pFtxInfo->uiCursorType);

#if defined( FLM_WIN)

	if( RC_BAD( f_calloc( (FLMUINT)(sizeof( CHAR_INFO) * (pFtxInfo->uiCols *
		pFtxInfo->uiRows)), &pFtxInfo->pCells)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

#elif !defined( FLM_NLM) || !defined( FLM_UNIX)

	pFtxInfo->uiRows--;

#endif

	if( RC_BAD( f_threadCreate( &pFtxInfo->pBackgroundThrd,
		_ftxBackgroundThread, "ftx_background")))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	pFtxInfo->uiBackground = uiBackground;
	pFtxInfo->uiForeground = uiForeground;

	if( RC_BAD( f_threadCreate( &pFtxInfo->pDisplayThrd,
		_ftxDefaultDisplayHandler, "ftx_display")))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	// Start the keyboard handler

	pFtxInfo->uiCurKey = 0;
	f_memset( pFtxInfo->puiKeyBuffer, 0, sizeof( FLMUINT) * CV_KEYBUF_SIZE);
	pFtxInfo->pKeyHandler = pKeyHandler;
	pFtxInfo->pvKeyHandlerData = pvKeyHandlerData;

	if( RC_BAD( f_threadCreate( &pFtxInfo->pKeyboardThrd,
		_ftxDefaultKeyboardHandler, "ftx_keyboard")))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	gv_bInitialized = TRUE;
	gv_uiInitCount++;
	*ppFtxInfo = pFtxInfo;

Exit:

	return( rc);

}

/****************************************************************************
Desc:		Frees all resources allocated to the FTX environment
Notes:	All screens and windows are freed automatically
****************************************************************************/
FTXRCODE FTXFree(
	FTX_INFO **	ppFtxInfo)
{
	FTX_INFO *			pFtxInfo;
	FTX_SCREEN *		pScreen;
	FTXRCODE				rc = FTXRC_SUCCESS;

	if( !gv_bInitialized)
	{
		rc = FTXRC_ILLEGAL_OP;
		goto Exit;
	}

	if( --gv_uiInitCount > 0)
	{
		*ppFtxInfo = NULL;
		goto Exit;
	}

	if( !ppFtxInfo)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	pFtxInfo = *ppFtxInfo;

	// Shut down the display, keyboard, and backgroudn threads

	f_threadDestroy( &pFtxInfo->pKeyboardThrd);
	f_threadDestroy( &pFtxInfo->pDisplayThrd);
	f_threadDestroy( &pFtxInfo->pBackgroundThrd);

	ftxLock( &(pFtxInfo->hFtxMutex));

	gv_bInitialized = FALSE;
	pFtxInfo->bExiting = TRUE;

	while( (pScreen = pFtxInfo->pScreenCur) != NULL)
	{
		ftxScreenFree( pScreen);
	}

	ftxDisplayReset( pFtxInfo);
	ftxDisplayExit();

#if defined( FLM_WIN)

	f_free( &pFtxInfo->pCells);

#elif defined( FLM_NLM)

	CloseScreen( pFtxInfo->pvScreenHandle);

#endif

	ftxUnlock( &(pFtxInfo->hFtxMutex));

	f_mutexDestroy( &(pFtxInfo->hFtxMutex));

	f_free( &pFtxInfo);
	*ppFtxInfo = NULL;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Refreshes the current screen
****************************************************************************/
FTXRCODE FTXRefresh(
	FTX_INFO *		pFtxInfo)
{
	FTX_WINDOW *	pWinScreen;
	FTXRCODE			rc = FTXRC_SUCCESS;

	ftxLock( &(pFtxInfo->hFtxMutex));

	if( !pFtxInfo->bRefreshDisabled && pFtxInfo->pScreenCur)
	{
		ftxLock( &(pFtxInfo->pScreenCur->hScreenMutex));
		if( pFtxInfo->pScreenCur->bChanged || pFtxInfo->bScreenSwitch)
		{
			if( pFtxInfo->bScreenSwitch)
			{
				pWinScreen = pFtxInfo->pScreenCur->pWinScreen;
				f_memset( pWinScreen->pszBuffer, 0,
					pWinScreen->uiRows * pWinScreen->uiCols);
				#ifdef FLM_UNIX
					ftxUnixDisplayReset();
				#endif
			}

#if defined( FLM_WIN)

			rc = ftxWin32Refresh( pFtxInfo);

#elif defined( FLM_NLM)

			rc = ftxNLMRefresh( pFtxInfo);

#else
			rc = ftxRefresh( pFtxInfo);

#endif
			pFtxInfo->pScreenCur->bChanged = FALSE;
			pFtxInfo->bScreenSwitch = FALSE;
			pFtxInfo->pScreenCur->bUpdateCursor = TRUE;
		}

		if( pFtxInfo->pScreenCur->bUpdateCursor)
		{
			ftxCursorUpdate( pFtxInfo);
		}
		ftxUnlock( &(pFtxInfo->pScreenCur->hScreenMutex));
	}

	ftxUnlock( &(pFtxInfo->hFtxMutex));

	return( rc);
}


/****************************************************************************
Desc:		Enables or disables refresh
****************************************************************************/
FTXRCODE FTXSetRefreshState(
	FTX_INFO *		pFtxInfo,
	FLMBOOL			bDisable)
{
	FTXRCODE			rc = FTXRC_SUCCESS;

	ftxLock( &(pFtxInfo->hFtxMutex));
	pFtxInfo->bRefreshDisabled = bDisable;
	ftxUnlock( &(pFtxInfo->hFtxMutex));

	return( rc);
}


/****************************************************************************
Desc:		Allows a keyboard handler to add a key to the FTX key buffer
****************************************************************************/
FTXRCODE FTXAddKey(
	FTX_INFO *		pFtxInfo,
	FLMUINT			uiKey)
{

	FLMBOOL			bSet = FALSE;
	FLMUINT			uiLoop;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pFtxInfo->hFtxMutex));

	uiLoop = pFtxInfo->uiCurKey;
	while( uiLoop < CV_KEYBUF_SIZE)
	{
		if( pFtxInfo->puiKeyBuffer[ uiLoop] == 0)
		{
			pFtxInfo->puiKeyBuffer[ uiLoop] = uiKey;
			bSet = TRUE;
			goto Exit;
		}
		uiLoop++;
	}

	if( !bSet)
	{
		uiLoop = 0;
		while( uiLoop < pFtxInfo->uiCurKey)
		{
			if( pFtxInfo->puiKeyBuffer[ uiLoop] == 0)
			{
				pFtxInfo->puiKeyBuffer[ uiLoop] = uiKey;
				bSet = TRUE;
				goto Exit;
			}
			uiLoop++;
		}
	}

Exit:

	ftxUnlock( &(pFtxInfo->hFtxMutex));

	if( !bSet)
	{
		rc = FTXRC_BUF_OVERRUN;
	}
	else
	{
		if( pFtxInfo->pScreenCur != NULL)
		{
			f_semSignal( pFtxInfo->pScreenCur->hKeySem);
		}
	}

	return( rc);

}


/****************************************************************************
Desc:		Cycles to the next screen in the FTX environment
****************************************************************************/
FTXRCODE FTXCycleScreensNext(
	FTX_INFO *		pFtxInfo)
{
	FTX_SCREEN *		pScreenTmp;
	FTX_SCREEN *		pScreenLast;
	FTXRCODE				rc = FTXRC_SUCCESS;


	ftxLock( &(pFtxInfo->hFtxMutex));

	if( pFtxInfo->pScreenCur && pFtxInfo->pScreenCur->pScreenNext)
	{
		pScreenTmp = pFtxInfo->pScreenCur;
		pFtxInfo->pScreenCur = pFtxInfo->pScreenCur->pScreenNext;

		pScreenLast = pFtxInfo->pScreenCur;
		while( pScreenLast->pScreenNext)
		{
			pScreenLast = pScreenLast->pScreenNext;
		}

		pScreenLast->pScreenNext = pScreenTmp;
		pScreenTmp->pScreenPrev = pScreenLast;
		pScreenTmp->pScreenNext = NULL;
		pFtxInfo->pScreenCur->pScreenPrev = NULL;

		pFtxInfo->bScreenSwitch = TRUE;
		ftxKeyboardFlush( pFtxInfo);
	}

	ftxUnlock( &(pFtxInfo->hFtxMutex));
	return( rc);
}

/****************************************************************************
Desc:		Cycles to the previous screen in the FTX environment
****************************************************************************/
FTXRCODE FTXCycleScreensPrev(
	FTX_INFO *		pFtxInfo)
{
	FTX_SCREEN *		pScreenPreviousFront;
	FTX_SCREEN *		pScreenLast;
	FTXRCODE				rc = FTXRC_SUCCESS;


	ftxLock( &(pFtxInfo->hFtxMutex));

	if( pFtxInfo->pScreenCur && pFtxInfo->pScreenCur->pScreenNext)
	{
		pScreenPreviousFront = pFtxInfo->pScreenCur;
		pScreenLast = pScreenPreviousFront;

		while ( pScreenLast->pScreenNext)
		{
			pScreenLast = pScreenLast->pScreenNext;
		}
		pScreenLast->pScreenPrev->pScreenNext = NULL;
		pScreenLast->pScreenPrev = NULL;
		pScreenLast->pScreenNext = pScreenPreviousFront;
		pScreenPreviousFront->pScreenPrev = pScreenLast;
		pFtxInfo->pScreenCur = pScreenLast;

		pFtxInfo->bScreenSwitch = TRUE;
		ftxKeyboardFlush( pFtxInfo);
	}

	ftxUnlock( &(pFtxInfo->hFtxMutex));
	return( rc);
}

/****************************************************************************
Desc:		Cycles to the next screen in the FTX environment (used when
			debugging on NetWare).
****************************************************************************/
void debugFTXCycleScreens( void)
{
	if( gv_pFtxInfo)
	{
		FTXCycleScreensNext( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:		Force cursor refresh
****************************************************************************/
FTXRCODE FTXRefreshCursor(
	FTX_INFO *		pFtxInfo)
{
	FTXRCODE				rc = FTXRC_SUCCESS;

	ftxLock( &(pFtxInfo->hFtxMutex));

	if( pFtxInfo->pScreenCur)
	{
		pFtxInfo->pScreenCur->bUpdateCursor = TRUE;
	}

	ftxUnlock( &(pFtxInfo->hFtxMutex));
	return( rc);
}


/****************************************************************************
Desc:		Invalidates the current screen so that it will be completly redrawn
****************************************************************************/
FTXRCODE FTXInvalidate(
	FTX_INFO *		pFtxInfo)
{
	FTX_WINDOW *	pWinScreen;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pFtxInfo->hFtxMutex));

	if( pFtxInfo->pScreenCur)
	{
		ftxLock( &(pFtxInfo->pScreenCur->hScreenMutex));
		pWinScreen = pFtxInfo->pScreenCur->pWinScreen;
		f_memset( pWinScreen->pszBuffer, 0,
			pWinScreen->uiRows * pWinScreen->uiCols);
		pFtxInfo->pScreenCur->bChanged = TRUE;
		ftxUnlock( &(pFtxInfo->pScreenCur->hScreenMutex));
	}

#ifdef FLM_UNIX
	ftxUnixDisplayReset();
#endif
	ftxUnlock( &(pFtxInfo->hFtxMutex));
	return( rc);
}


/****************************************************************************
Desc:		Allocates and initializes a new screen object
****************************************************************************/
FTXRCODE FTXScreenInit(
	FTX_INFO *		pFtxInfo,
	const char *	pszName,
	FTX_SCREEN **	ppScreen)
{
	FTX_SCREEN *	pScreen;
	FTX_SCREEN *	pScreenTmp;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pFtxInfo->hFtxMutex));

	*ppScreen = NULL;
	if( RC_BAD( f_calloc( sizeof( FTX_SCREEN), &pScreen)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	if( RC_BAD( f_mutexCreate( &(pScreen->hScreenMutex))))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	if( RC_BAD( f_semCreate( &(pScreen->hKeySem))))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	pScreen->uiRows = pFtxInfo->uiRows;
	pScreen->uiCols = pFtxInfo->uiCols;
	pScreen->uiBackground = pFtxInfo->uiBackground;
	pScreen->uiForeground = pFtxInfo->uiForeground;
	pScreen->uiCursorType = WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE;

	pScreen->pFtxInfo = pFtxInfo;

	if( f_strlen( pszName) <= CV_MAX_WINNAME_LEN)
	{
		f_strcpy( pScreen->szName, pszName);
	}
	else
	{
		f_strcpy( pScreen->szName, "?");
	}

	pScreen->bInitialized = TRUE;

	if( (rc = FTXWinInit( pScreen, pScreen->uiCols, pScreen->uiRows,
		&(pScreen->pWinScreen))) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	pScreen->pWinScreen->uiBackground = pScreen->uiBackground;
	pScreen->pWinScreen->uiForeground = pScreen->uiForeground;

	if( (rc = FTXWinInit( pScreen, pScreen->uiCols, pScreen->uiRows,
		&(pScreen->pWinImage))) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	f_memset( pScreen->pWinScreen->pszBuffer, 0,
		pScreen->pWinScreen->uiRows *
			pScreen->pWinScreen->uiCols);

Exit:

	if( rc != FTXRC_SUCCESS)
	{
		pScreen->bInitialized = FALSE;
	}
	else
	{
		if( pFtxInfo->pScreenCur)
		{
			pScreenTmp = pFtxInfo->pScreenCur;
			while( pScreenTmp->pScreenNext)
			{
				pScreenTmp = pScreenTmp->pScreenNext;
			}
			pScreenTmp->pScreenNext = pScreen;
			pScreen->pScreenPrev = pScreenTmp;
		}
		else
		{
			pFtxInfo->pScreenCur = pScreen;
			pFtxInfo->bScreenSwitch = TRUE;
		}

		pScreen->uiId = pFtxInfo->uiSequence++;
		*ppScreen = pScreen;
	}

	ftxUnlock( &(pFtxInfo->hFtxMutex));
	return( rc);

}


/****************************************************************************
Desc:		Frees all resources allocated to a screen, including all window
			objects
****************************************************************************/
FTXRCODE FTXScreenFree(
	FTX_SCREEN **	ppScreen)
{

	FTX_SCREEN *	pScreen;
	FTX_INFO *		pFtxInfo;
	FTXRCODE			rc = FTXRC_SUCCESS;


	if( !ppScreen)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	pScreen = *ppScreen;
	if( !pScreen)
	{
		goto Exit;
	}

	if( pScreen->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_SCREEN;
		goto Exit;
	}

	pFtxInfo = pScreen->pFtxInfo;
	ftxLock( &(pFtxInfo->hFtxMutex));
	rc = ftxScreenFree( pScreen);
	pFtxInfo->bScreenSwitch = TRUE;
	ftxUnlock( &(pFtxInfo->hFtxMutex));

Exit:

	if( rc == FTXRC_SUCCESS)
	{
		*ppScreen = NULL;
	}

	return( rc);

}


/****************************************************************************
Desc:		Makes the passed-in screen the visible screen
****************************************************************************/
FTXRCODE FTXScreenDisplay(
	FTX_SCREEN *	pScreen)
{
	FLMBOOL				bScreenValid = FALSE;
	FTX_SCREEN *		pTmpScreen;
	FTXRCODE				rc = FTXRC_SUCCESS;

	ftxLock( &(gv_pFtxInfo->hFtxMutex));

	// Make sure the screen is still in the list.  If it isn't, the thread
	// that owned the screen may have terminated.

	pTmpScreen = gv_pFtxInfo->pScreenCur;
	while( pTmpScreen)
	{
		if( pTmpScreen == pScreen)
		{
			bScreenValid = TRUE;
			break;
		}

		pTmpScreen = pTmpScreen->pScreenPrev;
	}

	pTmpScreen = gv_pFtxInfo->pScreenCur;
	while( pTmpScreen)
	{
		if( pTmpScreen == pScreen)
		{
			bScreenValid = TRUE;
			break;
		}

		pTmpScreen = pTmpScreen->pScreenNext;
	}

	if( !bScreenValid)
	{
		rc = FTXRC_INVALID_SCREEN;
		goto Exit;
	}

	if( pScreen != pScreen->pFtxInfo->pScreenCur)
	{
		if( pScreen->pScreenNext != NULL)
		{
			pScreen->pScreenNext->pScreenPrev = pScreen->pScreenPrev;
		}

		if( pScreen->pScreenPrev != NULL)
		{
			pScreen->pScreenPrev->pScreenNext = pScreen->pScreenNext;
		}

		pScreen->pScreenPrev = NULL;
		pScreen->pScreenNext = pScreen->pFtxInfo->pScreenCur;
		pScreen->pFtxInfo->pScreenCur->pScreenPrev = pScreen;
		pScreen->pFtxInfo->pScreenCur = pScreen;
		pScreen->pFtxInfo->bScreenSwitch = TRUE;
		ftxKeyboardFlush( pScreen->pFtxInfo);
	}

Exit:

	ftxUnlock( &(gv_pFtxInfo->hFtxMutex));
	return( rc);
}


/****************************************************************************
Desc:		Retrieves the size of the passed-in screen
****************************************************************************/
FTXRCODE FTXScreenGetSize(
	FTX_SCREEN *	pScreen,
	FLMUINT *		puiNumCols,
	FLMUINT *		puiNumRows)
{

	FTXRCODE			rc = FTXRC_SUCCESS;


	if( pScreen == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	if( puiNumCols)
	{
		*puiNumCols = pScreen->uiCols;
	}

	if( puiNumRows)
	{
		*puiNumRows = pScreen->uiRows;
	}

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Sets the screen's shutdown flag
****************************************************************************/
FTXRCODE FTXScreenSetShutdownFlag(
	FTX_SCREEN *	pScreen,
	FLMBOOL *		pbShutdownFlag)
{

	FTXRCODE			rc = FTXRC_SUCCESS;


	if( pScreen == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	pScreen->pbShutdown = pbShutdownFlag;

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Creates a title window and main window (with border)
****************************************************************************/
FTXRCODE FTXScreenInitStandardWindows(
	FTX_SCREEN *	pScreen,
	FLMUINT			uiTitleBackColor,
	FLMUINT			uiTitleForeColor,
	FLMUINT			uiMainBackColor,
	FLMUINT			uiMainForeColor,
	FLMBOOL			bBorder,
	FLMBOOL			bBackFill,
	const char *	pszTitle,
	FTX_WINDOW **	ppTitleWin,
	FTX_WINDOW **	ppMainWin)
{
	FLMUINT			uiScreenCols;
	FLMUINT			uiScreenRows;
	FTX_WINDOW *	pTitleWin;
	FTX_WINDOW *	pMainWin;
	FTXRCODE			rc;

	if( (rc = FTXScreenGetSize( pScreen,
		&uiScreenCols, &uiScreenRows)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinInit( pScreen, 0, 1, &pTitleWin)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinSetBackFore( pTitleWin,
		uiTitleBackColor, uiTitleForeColor)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinClear( pTitleWin)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	FTXWinSetCursorType( pTitleWin, WPS_CURSOR_INVISIBLE);

	if( pszTitle)
	{
		FTXWinPrintf( pTitleWin, "%s", pszTitle);
	}

	if( (rc = FTXWinOpen( pTitleWin)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinInit( pScreen, uiScreenCols,
		uiScreenRows - 1, &pMainWin)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinMove( pMainWin, 0, 1)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinSetBackFore( pMainWin,
		uiMainBackColor, uiMainForeColor)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinClear( pMainWin)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( bBorder)
	{
		if( (rc = FTXWinDrawBorder( pMainWin)) != FTXRC_SUCCESS)
		{
			goto Exit;
		}
	}

#if defined( FLM_WIN) || defined( FLM_NLM)
	if( bBackFill)
	{
		FTXWinSetChar( pMainWin, 176);
	}
#else
	F_UNREFERENCED_PARM( bBackFill);
#endif

	if( (rc = FTXWinOpen( pMainWin)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( ppTitleWin)
	{
		*ppTitleWin = pTitleWin;
	}

	if( ppMainWin)
	{
		*ppMainWin = pMainWin;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:		Allocates and initializes a window object
****************************************************************************/
FTXRCODE FTXWinInit(
	FTX_SCREEN *	pScreen,
	FLMUINT			uiCols,
	FLMUINT			uiRows,
	FTX_WINDOW **	ppWindow)
{

	FLMUINT			uiSize;
	FTX_WINDOW *	pWindow;
	FTX_WINDOW *	pWinTmp;
	FTXRCODE			rc = FTXRC_SUCCESS;

	*ppWindow = NULL;

	if( !pScreen->bInitialized)
	{
		rc = FTXRC_INVALID_SCREEN;
		goto Exit;
	}

	ftxLock( &(pScreen->hScreenMutex));

	if( uiRows > pScreen->uiRows || uiCols > pScreen->uiCols)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	if( uiRows == 0)
	{
		uiRows = pScreen->uiRows;
	}

	if( uiCols == 0)
	{
		uiCols = pScreen->uiCols;
	}

	if( RC_BAD( f_calloc( sizeof( FTX_WINDOW), &pWindow)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	uiSize = (FLMUINT)((uiRows * uiCols) + 1);

	if( RC_BAD( f_calloc( (FLMUINT)(sizeof( FLMBYTE) * uiSize),
		&pWindow->pszBuffer)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	if( RC_BAD( f_calloc( (FLMUINT)(sizeof( FLMBYTE) * uiSize),
		&pWindow->pucForeAttrib)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	if( RC_BAD( f_calloc( (FLMUINT)(sizeof( FLMBYTE) * uiSize),
		&pWindow->pucBackAttrib)))
	{
		rc = FTXRC_MEM;
		goto Exit;
	}

	f_memset( pWindow->pucForeAttrib, (FLMBYTE)pScreen->uiForeground, uiSize);
	f_memset( pWindow->pucBackAttrib, (FLMBYTE)pScreen->uiBackground, uiSize);

	pWindow->uiRows = uiRows;
	pWindow->uiCols = uiCols;

	pWindow->uiCursorType = WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE;
	pWindow->bScroll = TRUE;
	pWindow->bOpen = FALSE;
	pWindow->bInitialized = TRUE;
	pWindow->bForceOutput = FALSE;

	pWindow->pScreen = pScreen;
	pWindow->uiId = pScreen->uiSequence++;

	ftxWinReset( pWindow);

	if( pScreen->pWinCur)
	{
		pWinTmp = pScreen->pWinCur;
		while( pWinTmp->pWinNext)
		{
			pWinTmp = pWinTmp->pWinNext;
		}

		pWindow->pWinPrev = pWinTmp;
		pWinTmp->pWinNext = pWindow;
	}
	else
	{
		pScreen->pWinCur = pWindow;
	}

Exit:

	if( rc == FTXRC_SUCCESS)
	{
		*ppWindow = pWindow;
	}

	ftxUnlock( &(pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Frees all resources associated with the passed-in window object
****************************************************************************/
FTXRCODE FTXWinFree(
	FTX_WINDOW **	ppWindow)
{

	FTX_WINDOW *	pWindow;
	FTX_SCREEN *	pScreen;
	FTXRCODE			rc = FTXRC_SUCCESS;


	if( !ppWindow)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	pWindow = *ppWindow;

	if( pWindow->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	pScreen = pWindow->pScreen;
	ftxLock( &(pScreen->hScreenMutex));
	rc = ftxWinFree( pWindow);
	ftxUnlock( &(pScreen->hScreenMutex));

Exit:

	if( rc == FTXRC_SUCCESS)
	{
		*ppWindow = NULL;
	}

	return( rc);

}


/****************************************************************************
Desc:		Opens the specified window and makes it visible
****************************************************************************/
FTXRCODE FTXWinOpen(
	FTX_WINDOW *	pWindow)
{

	FTXRCODE				rc = FTXRC_SUCCESS;


	if( pWindow == NULL || !pWindow->bInitialized)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	ftxLock( &(pWindow->pScreen->hScreenMutex));
	rc = ftxWinOpen( pWindow);
  	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Closes (or hides) the specified window
****************************************************************************/
FTXRCODE FTXWinClose(
	FTX_WINDOW *	pWindow)
{

	FTXRCODE					rc = FTXRC_SUCCESS;


	if( pWindow == NULL)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	if( !pWindow->bInitialized || !pWindow->bOpen)
	{
		goto Exit;
	}

	ftxLock( &(pWindow->pScreen->hScreenMutex));
	rc = ftxWinClose( pWindow);
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Sets the specified window's name
****************************************************************************/
FTXRCODE FTXWinSetName(
	FTX_WINDOW *	pWindow,
	const char *	pszName)
{
	FTXRCODE				rc = FTXRC_SUCCESS;

	ftxLock( &(pWindow->pScreen->hScreenMutex));

	if( f_strlen( pszName) > CV_MAX_WINNAME_LEN)
	{
		rc = FTXRC_BUF_OVERRUN;
		goto Exit;
	}
	f_strcpy( pWindow->szName, pszName);

Exit:

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Moves the specified window to a new location on the screen
****************************************************************************/
FTXRCODE FTXWinMove(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	if( (FLMUINT)uiCol + (FLMUINT)pWindow->uiCols >
		(FLMUINT)pWindow->pScreen->uiCols)
	{
		rc = FTXRC_INVALID_POS;
		goto Exit;
	}

	if( uiRow + pWindow->uiRows > pWindow->pScreen->uiRows)
	{
		rc = FTXRC_INVALID_POS;
		goto Exit;
	}

	if( pWindow->uiUlx != uiCol || pWindow->uiUly != uiRow)
	{
		pWindow->uiUlx = uiCol;
		pWindow->uiUly = uiRow;
		if( pWindow->bOpen)
		{
			pWindow->pScreen->bChanged = TRUE;
		}
	}

Exit:

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Sets the input focus to the specified window
****************************************************************************/
FTXRCODE FTXWinSetFocus(
	FTX_WINDOW *	pWindow)
{

	ftxLock( &(pWindow->pScreen->hScreenMutex));

	if( pWindow->bOpen && pWindow->pScreen->pWinCur != pWindow)
	{
		ftxWinClose( pWindow);
		ftxWinOpen( pWindow);
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( FTXRC_SUCCESS);

}


/****************************************************************************
Desc:		Sets the background color of all characters in the specified window
			to the same color
****************************************************************************/
FTXRCODE FTXWinPaintBackground(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiBackground)
{

	FLMUINT			uiSize;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	uiSize = (FLMUINT)(pWindow->uiRows * pWindow->uiCols);
	f_memset( pWindow->pucBackAttrib, (FLMBYTE)uiBackground, uiSize);
	pWindow->uiBackground = uiBackground;

	if( pWindow->bOpen)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);

}


/****************************************************************************
Desc:		Sets the background and/or foreground color of a row in the
			specified window
****************************************************************************/
FTXRCODE FTXWinPaintRow(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiBackground,
	FLMUINT *		puiForeground,
	FLMUINT			uiRow)
{

	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	if( uiRow < (pWindow->uiRows - (2 * pWindow->uiOffset)))
	{
		if( puiBackground != NULL)
		{
			f_memset( pWindow->pucBackAttrib +
				(pWindow->uiCols * (uiRow + pWindow->uiOffset)) + pWindow->uiOffset,
				(FLMBYTE)*puiBackground, pWindow->uiCols - (2 * pWindow->uiOffset));
		}
		if( puiForeground != NULL)
		{
			f_memset( pWindow->pucForeAttrib +
				(pWindow->uiCols * (uiRow + pWindow->uiOffset)) + pWindow->uiOffset,
				(FLMBYTE)*puiForeground, pWindow->uiCols - (2 * pWindow->uiOffset));
		}

		if( pWindow->bOpen)
		{
			pWindow->pScreen->bChanged = TRUE;
		}
	}
	else
	{
		rc = FTXRC_INVALID_POS;
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Sets all of the characters in the window to the specified character
****************************************************************************/
FTXRCODE FTXWinSetChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar)
{
	FLMUINT			uiSize;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));


	uiSize = (FLMUINT)(pWindow->uiCols - pWindow->uiOffset) *
		(FLMUINT)(pWindow->uiRows - pWindow->uiOffset);

	f_memset( pWindow->pszBuffer, (FLMBYTE)uiChar, uiSize);
	if( pWindow->bOpen)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Sets the background color of a row in the specified window.
****************************************************************************/
FTXRCODE FTXWinPaintRowBackground(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiBackground,
	FLMUINT			uiRow)
{

	return( FTXWinPaintRow( pWindow, &uiBackground, NULL, uiRow));

}


/****************************************************************************
Desc:		Sets the foreground color of all characters in the specified window
****************************************************************************/
FTXRCODE FTXWinPaintForeground(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiForeground)
{

	FLMUINT			uiSize;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	uiSize = (FLMUINT)(pWindow->uiRows * pWindow->uiCols);
	f_memset( pWindow->pucForeAttrib, (FLMBYTE)uiForeground, uiSize);
	pWindow->uiForeground = uiForeground;

	if( pWindow->bOpen)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Sets the foreground color of a row in the specified window.
****************************************************************************/
FTXRCODE FTXWinPaintRowForeground(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiForeground,
	FLMUINT			uiRow)
{
	return( FTXWinPaintRow( pWindow, NULL, &uiForeground, uiRow));
}


/****************************************************************************
Desc:		Sets the background and foreground color of the pen associated
			with the current window
****************************************************************************/
FTXRCODE FTXWinSetBackFore(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiBackground,
	FLMUINT			uiForeground)
{

	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	pWindow->uiBackground = uiBackground;
	pWindow->uiForeground = uiForeground;

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Retrieves the current background and/or foreground color of
			the pen associated with the specified window
****************************************************************************/
FTXRCODE FTXWinGetBackFore(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiBackground,
	FLMUINT *		puiForeground)
{

	ftxLock( &(pWindow->pScreen->hScreenMutex));

	if( puiBackground != NULL)
	{
		*puiBackground = pWindow->uiBackground;
	}

	if( puiForeground != NULL)
	{
		*puiForeground = pWindow->uiForeground;
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( FTXRC_SUCCESS);

}


/****************************************************************************
Desc:		Prints a character at the current cursor location in the
			specified window.
****************************************************************************/
FTXRCODE FTXWinPrintChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar)
{
	FTXRCODE			rc;


	ftxLock( &(pWindow->pScreen->hScreenMutex));
	rc = ftxWinPrintChar( pWindow, uiChar);
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);

}


/****************************************************************************
Desc:		Prints a string starting at the current cursor location in the
			specified window.
****************************************************************************/
FTXRCODE FTXWinPrintStr(
	FTX_WINDOW *	pWindow,
	const char *	pszString)
{
	FLMBOOL			bSemLocked = FALSE;
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( pszString == NULL)
	{
		goto Exit;
	}

	ftxLock( &(pWindow->pScreen->hScreenMutex));
	bSemLocked = TRUE;

	while( *pszString != '\0')
	{
		if( (rc = ftxWinPrintChar( pWindow,
							(FLMUINT)*pszString)) != FTXRC_SUCCESS)
		{
			goto Exit;
		}
		pszString++;
	}

Exit:

	if( bSemLocked)
	{
		ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	}

	return( rc);

}

/****************************************************************************
Desc:    Output a formatted string at present cursor location.
****************************************************************************/
FTXRCODE FTXWinPrintf(
	FTX_WINDOW *	pWindow,
	const char *	pszFormat, ...)
{
	char			pszBuffer[ 512];
	f_va_list	args;

	f_va_start( args, pszFormat);
	f_vsprintf( pszBuffer, pszFormat, &args);
	f_va_end( args);
	return( FTXWinPrintStr( pWindow, pszBuffer));
}

/****************************************************************************
Desc:    Output a formatted string (with color) at present cursor location.
****************************************************************************/
FTXRCODE FTXWinCPrintf(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiBackground,
	FLMUINT			uiForeground,
	const char *	pszFormat, ...)
{
	char			szBuffer[ 512];
	FLMUINT		uiOldBackground;
	FLMUINT		uiOldForeground;
	FTXRCODE		rc;
	f_va_list	args;

	uiOldBackground = pWindow->uiBackground;
	uiOldForeground = pWindow->uiForeground;
	pWindow->uiBackground = uiBackground;
	pWindow->uiForeground = uiForeground;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	rc = FTXWinPrintStr( pWindow, szBuffer);

	pWindow->uiBackground = uiOldBackground;
	pWindow->uiForeground = uiOldForeground;

	return( rc);
}


/****************************************************************************
Desc:		Prints a string at a specific offset in the specified window.
****************************************************************************/
FTXRCODE FTXWinPrintStrXY(
	FTX_WINDOW *	pWindow,
	const char *	pszString,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FTXRCODE			rc;

	if( (rc = FTXWinSetCursorPos( pWindow, uiCol, uiRow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinPrintStr( pWindow, pszString)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

Exit:

	return( rc);

}

/****************************************************************************
Desc:		Sets the cursor position in the specified window
****************************************************************************/
FTXRCODE FTXWinSetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{

	FTXRCODE			rc;


	ftxLock( &(pWindow->pScreen->hScreenMutex));
	rc = ftxWinSetCursorPos( pWindow, uiCol, uiRow);
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);

}

/****************************************************************************
Desc:		Enables or disables scrolling in the specified window
****************************************************************************/
FTXRCODE FTXWinSetScroll(
	FTX_WINDOW *	pWindow,
	FLMBOOL			bScroll)
{

	FTXRCODE			rc = FTXRC_SUCCESS;


	if( pWindow == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	pWindow->bScroll = bScroll;

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Enables or disables line wrap
****************************************************************************/
FTXRCODE FTXWinSetLineWrap(
	FTX_WINDOW *	pWindow,
	FLMBOOL			bLineWrap)
{

	FTXRCODE			rc = FTXRC_SUCCESS;

	if( pWindow == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	pWindow->bNoLineWrap = !bLineWrap;

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Retrieves the scroll flag for the specified window
****************************************************************************/
FTXRCODE FTXWinGetScroll(
	FTX_WINDOW *	pWindow,
	FLMBOOL *		pbScroll)
{

	FTXRCODE			rc = FTXRC_SUCCESS;

	if( pWindow == NULL || pbScroll == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	*pbScroll = pWindow->bScroll;

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Retrieves the screen of the current window
****************************************************************************/
FTXRCODE FTXWinGetScreen(
	FTX_WINDOW *	pWindow,
	FTX_SCREEN **	ppScreen)
{
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( pWindow == NULL || ppScreen == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	*ppScreen = NULL;

	if( pWindow->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	*ppScreen = pWindow->pScreen;

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Retrieves the windows position on the screen
****************************************************************************/
FTXRCODE FTXWinGetPosition(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiCol,
	FLMUINT *		puiRow)
{
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( !pWindow)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	if( !pWindow->bInitialized)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	if( puiCol)
	{
		*puiCol = pWindow->uiUlx;
	}

	if( puiRow)
	{
		*puiRow = pWindow->uiUly;
	}

Exit:

	return( rc);

}


/****************************************************************************
Desc:		Clears from the specified column and row to the end of the row in
			the specified window
****************************************************************************/
FTXRCODE FTXWinClearLine(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{

	FTXRCODE		rc;


	ftxLock( &(pWindow->pScreen->hScreenMutex));
	rc = ftxWinClearLine( pWindow, uiCol, uiRow);
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);

}


/****************************************************************************
Desc:		Clears from the current cursor position to the end of the current
			line
****************************************************************************/
FTXRCODE FTXWinClearToEOL(
	FTX_WINDOW *	pWindow)
{
	FTXRCODE		rc;

	ftxLock( &(pWindow->pScreen->hScreenMutex));
	rc = ftxWinClearLine( pWindow,
		(FLMUINT)(pWindow->uiCurX - pWindow->uiOffset),
		(FLMUINT)(pWindow->uiCurY - pWindow->uiOffset));
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);
}


/****************************************************************************
Desc:		Clears the canvas of the specified window starting at the requested
			row and column offset
****************************************************************************/
FTXRCODE FTXWinClearXY(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{

	FLMUINT			uiSaveCol;
	FLMUINT			uiSaveRow;
	FLMUINT			uiLoop;
	FTXRCODE			rc;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	uiSaveCol = pWindow->uiCurX;
	uiSaveRow = pWindow->uiCurY;

	if( (rc = ftxWinClearLine( pWindow, uiCol, uiRow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}
	uiLoop = (FLMUINT)(uiRow + 1);

	while( uiLoop < pWindow->uiRows - pWindow->uiOffset)
	{
		if( (rc = ftxWinClearLine( pWindow, 0, uiLoop)) != FTXRC_SUCCESS)
		{
			goto Exit;
		}
		uiLoop++;
	}

	pWindow->uiCurY = uiSaveRow;
	pWindow->uiCurX = uiSaveCol;

Exit:

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}


/****************************************************************************
Desc:		Clears the canvas area of the specified window
****************************************************************************/
FTXRCODE FTXWinClear(
	FTX_WINDOW *	pWindow)
{
	FTXRCODE		rc;

	if( (rc = FTXWinClearXY( pWindow, 0, 0)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinSetCursorPos( pWindow, 0, 0)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

Exit:

	return( rc);

}

/****************************************************************************
Desc:		Draws a border around the canvas area of the specified window
****************************************************************************/
FTXRCODE FTXWinDrawBorder(
	FTX_WINDOW *	pWindow)
{

	FLMUINT			uiLoop;
	FLMBOOL			bScroll;
	FLMUINT			uiCols;
	FLMUINT			uiRows;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxLock( &(pWindow->pScreen->hScreenMutex));

	uiCols = pWindow->uiCols;
	uiRows = pWindow->uiRows;

	if( (uiRows > 2 && uiCols > 2))
	{
		pWindow->bForceOutput = TRUE;

		pWindow->uiOffset = 0;
		bScroll = pWindow->bScroll;

		pWindow->uiOffset = 0;
		pWindow->bScroll = FALSE;

		ftxWinSetCursorPos( pWindow, 0, 0);
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)201);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		ftxWinSetCursorPos( pWindow, (FLMUINT)(uiCols - 1), 0);
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)187);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		ftxWinSetCursorPos( pWindow, 0, (FLMUINT)(uiRows - 1));
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)200);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		ftxWinSetCursorPos( pWindow, (FLMUINT)(uiCols - 1),
			(FLMUINT)(uiRows - 1));
#if defined( FLM_WIN) || defined( FLM_NLM)
		ftxWinPrintChar( pWindow, (FLMUINT)188);
#else
		ftxWinPrintChar( pWindow, (FLMUINT)'+');
#endif

		for( uiLoop = 1; uiLoop < uiCols - 1; uiLoop++)
		{
			ftxWinSetCursorPos( pWindow, uiLoop, 0);
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)205);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'-');
#endif

			ftxWinSetCursorPos( pWindow, uiLoop,
				(FLMUINT)(uiRows - 1));
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)205);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'-');
#endif
		}

		for( uiLoop = 1; uiLoop < uiRows - 1; uiLoop++)
		{
			ftxWinSetCursorPos( pWindow, 0, uiLoop);
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)186);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'|');
#endif

			ftxWinSetCursorPos( pWindow, (FLMUINT)(uiCols - 1),
				uiLoop);
#if defined( FLM_WIN) || defined( FLM_NLM)
			ftxWinPrintChar( pWindow, (FLMUINT)186);
#else
			ftxWinPrintChar( pWindow, (FLMUINT)'|');
#endif
		}

		pWindow->uiOffset = 1;
		pWindow->bScroll = bScroll;
		pWindow->bForceOutput = FALSE;

		ftxWinSetCursorPos( pWindow, 0, 0);
	}

	ftxUnlock( &(pWindow->pScreen->hScreenMutex));
	return( rc);

}

/****************************************************************************
Desc:		Draws a border around the canvas area of the specified window
****************************************************************************/
FTXRCODE FTXWinSetTitle(
	FTX_WINDOW *	pWindow,
	const char *	pszTitle,
	FLMUINT			uiBackground,
	FLMUINT			uiForeground)
{
	FLMBOOL			bScroll = FALSE;
	FLMUINT			uiCols;
	FLMUINT			uiRows;
	FLMUINT			uiStrLen;
	FTXRCODE			rc = FTXRC_SUCCESS;
	FLMUINT			uiSaveForeground;
	FLMUINT			uiSaveBackground;

	ftxLock( &(pWindow->pScreen->hScreenMutex));

	uiCols = pWindow->uiCols;
	uiRows = pWindow->uiRows;

	if( (uiRows > 2 && uiCols > 2))
	{
		pWindow->bForceOutput = TRUE;

		pWindow->uiOffset = 0;
		bScroll = pWindow->bScroll;

		pWindow->uiOffset = 0;
		pWindow->bScroll = FALSE;
		uiSaveBackground = pWindow->uiBackground;
		pWindow->uiBackground = uiBackground;
		uiSaveForeground = pWindow->uiForeground;
		pWindow->uiForeground = uiForeground;
		uiStrLen = f_strlen( pszTitle);
		if( uiStrLen < uiCols)
		{
			ftxWinSetCursorPos( pWindow, (FLMUINT)((uiCols - uiStrLen) / 2), 0);
		}
		else
		{
			ftxWinSetCursorPos( pWindow, 0, 0);
		}

		while( *pszTitle != '\0')
		{
			if( (rc = ftxWinPrintChar( pWindow, (FLMUINT)*pszTitle)) != FTXRC_SUCCESS)
			{
				goto Exit;
			}
			pszTitle++;
		}
		pWindow->uiBackground = uiSaveBackground;
		pWindow->uiForeground = uiSaveForeground;
	}

Exit:

	pWindow->uiOffset = 1;
	pWindow->bScroll = bScroll;
	pWindow->bForceOutput = FALSE;
	ftxWinSetCursorPos( pWindow, 0, 0);
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);

}


/****************************************************************************
Desc:		Draws a border around the canvas area of the specified window
****************************************************************************/
FTXRCODE FTXWinSetHelp(
	FTX_WINDOW *	pWindow,
	const char *	pszHelp,
	FLMUINT			uiBackground,
	FLMUINT			uiForeground)
{
	FLMBOOL			bScroll = FALSE;
	FLMUINT			uiCols;
	FLMUINT			uiRows;
	FLMUINT			uiStrLen;
	FTXRCODE			rc = FTXRC_SUCCESS;
	FLMUINT			uiSaveForeground;
	FLMUINT			uiSaveBackground;

	ftxLock( &(pWindow->pScreen->hScreenMutex));

	uiCols = pWindow->uiCols;
	uiRows = pWindow->uiRows;

	if( (uiRows > 2 && uiCols > 2))
	{
		pWindow->bForceOutput = TRUE;

		pWindow->uiOffset = 0;
		bScroll = pWindow->bScroll;

		pWindow->uiOffset = 0;
		pWindow->bScroll = FALSE;
		uiSaveBackground = pWindow->uiBackground;
		pWindow->uiBackground = uiBackground;
		uiSaveForeground = pWindow->uiForeground;
		pWindow->uiForeground = uiForeground;

		uiStrLen = f_strlen( pszHelp);
		if( uiStrLen < uiCols)
		{
			ftxWinSetCursorPos( pWindow, (FLMUINT)((uiCols - uiStrLen) / 2), uiRows-1);
		}
		else
		{
			ftxWinSetCursorPos( pWindow, 0, uiRows-1);
		}

		while( *pszHelp != '\0')
		{
			if( (rc = ftxWinPrintChar( pWindow, (FLMUINT)*pszHelp)) != FTXRC_SUCCESS)
			{
				goto Exit;
			}
			pszHelp++;
		}
		pWindow->uiBackground = uiSaveBackground;
		pWindow->uiForeground = uiSaveForeground;
	}

Exit:

	pWindow->uiOffset = 1;
	pWindow->bScroll = bScroll;
	pWindow->bForceOutput = FALSE;
	ftxWinSetCursorPos( pWindow, 0, 0);
	ftxUnlock( &(pWindow->pScreen->hScreenMutex));

	return( rc);

}

/****************************************************************************
Desc:		Tests the key buffer for an available key
****************************************************************************/
FTXRCODE FTXWinTestKB(
	FTX_WINDOW *		pWindow)
{

	FTX_INFO *		pFtxInfo;
	FTXRCODE			rc = FTXRC_SUCCESS;

	pFtxInfo = pWindow->pScreen->pFtxInfo;
	ftxLock( &(pFtxInfo->hFtxMutex));

	if( !pWindow->bOpen || pWindow->pScreen->pWinCur != pWindow ||
		pFtxInfo->pScreenCur != pWindow->pScreen)
	{
		rc = FTXRC_NO_INPUT;
		goto Exit;
	}

	if( pFtxInfo->puiKeyBuffer[ pFtxInfo->uiCurKey] == 0)
	{
		rc = FTXRC_NO_INPUT;
	}

Exit:

	ftxUnlock( &(pFtxInfo->hFtxMutex));
	return( rc);

}


/****************************************************************************
Desc:		Gets a character from the keyboard
****************************************************************************/
FTXRCODE FTXWinInputChar(
	FTX_WINDOW *	pWindow,
	FLMUINT *		puiChar)
{
	FTX_INFO *		pFtxInfo = NULL;
	FTXRCODE			rc = FTXRC_SUCCESS;
	FLMBOOL			bLocked = FALSE;


	if( puiChar)
	{
		*puiChar = 0;
	}

	if( pWindow == NULL)
	{
		rc = FTXRC_NULL_POINTER;
		goto Exit;
	}

	if( pWindow->bInitialized == FALSE)
	{
		rc = FTXRC_INVALID_WIN;
		goto Exit;
	}

	if( !pWindow->bOpen || pWindow->pScreen->pWinCur != pWindow)
	{
		rc = FTXRC_NO_INPUT;
		goto Exit;
	}

	pFtxInfo = pWindow->pScreen->pFtxInfo;

	for( ;;)
	{
		ftxLock( &(pFtxInfo->hFtxMutex));
		bLocked = TRUE;

		if( (pWindow->pScreen->pFtxInfo->pbShutdown != NULL &&
			*(pWindow->pScreen->pFtxInfo->pbShutdown) == TRUE) ||
			(pWindow->pScreen->pbShutdown != NULL &&
			*(pWindow->pScreen->pbShutdown) == TRUE))
		{
			rc = FTXRC_SHUTDOWN;
			goto Exit;
		}

		if( pFtxInfo->pScreenCur == pWindow->pScreen)
		{
			if( pFtxInfo->puiKeyBuffer[ pFtxInfo->uiCurKey])
			{
				if( puiChar)
				{
					*puiChar = pFtxInfo->puiKeyBuffer[ pFtxInfo->uiCurKey];
				}
				pFtxInfo->puiKeyBuffer[ pFtxInfo->uiCurKey] = 0;
				pFtxInfo->uiCurKey++;
				if( pFtxInfo->uiCurKey >= CV_KEYBUF_SIZE)
				{
					pFtxInfo->uiCurKey = 0;
				}
				break;
			}
		}
		ftxUnlock( &(pFtxInfo->hFtxMutex));
		bLocked = FALSE;
		(void)f_semWait( pWindow->pScreen->hKeySem, 1000);
	}

Exit:

	if( bLocked)
	{
		ftxUnlock( &(pFtxInfo->hFtxMutex));
	}

	return( rc);

}


/****************************************************************************
Desc:		Line editor routine
****************************************************************************/
FTXRCODE FTXLineEdit(
	FTX_WINDOW *	pWindow,
		// [IN] Pointer to input window.
	char *			pszBuffer,
		// [IN/OUT] Initialized buffer to be edited.
	FLMUINT			uiBufSize,
		// [IN] Size of pszBuffer (bytes).
	FLMUINT			uiMaxWidth,
		// [IN] Maximum size of the region.
	FLMUINT *		puiCharCount,
		// [OUT]		Number of characters input.
	FLMUINT *		puiTermChar
		// [OUT]		Character which caused the routine to return.
	)
{

	char				szLineBuf[ 256];
	char				szSnapBuf[ 256];
	FLMUINT			uiCharCount;
	FLMUINT			uiBufPos;
	FLMUINT			uiStartCol;
	FLMUINT			uiStartRow;
	FLMUINT			uiChar;
	FLMUINT			uiCursorOutputPos = 0;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumCols;
	FLMUINT			uiSaveCursor;
	FLMUINT			uiLoop;
	FLMUINT			uiCharsLn;
	FLMUINT			uiOutputStart = 0;
	FLMUINT			uiCursorPos;
	FLMUINT			uiOutputEnd = 0;
	FLMBOOL			bDone;
	FLMBOOL			bInsert;
	FLMBOOL			bRefresh;
	FLMBOOL			bGotChar = FALSE;
	FLMBOOL			bSaveScroll = FALSE;
	FTXRCODE			rc = FTXRC_SUCCESS;


	if( puiCharCount)
	{
		*puiCharCount = 0;
	}

	uiSaveCursor = FTXWinGetCursorType( pWindow);
	FTXWinGetCanvasSize( pWindow, &uiNumCols, &uiNumRows);
	uiStartCol = FTXWinGetCurrCol( pWindow);
	uiStartRow = FTXWinGetCurrRow( pWindow);
	FTXWinGetScroll( pWindow, &bSaveScroll);

	if( uiBufSize < 2 || uiMaxWidth < 2 || (uiNumCols - uiStartCol) < 3)
	{
		return( 0);
	}

	FTXWinSetScroll( pWindow, FALSE);
	FTXWinSetFocus( pWindow);
	FTXRefresh( pWindow->pScreen->pFtxInfo);

	uiCharsLn = (FLMUINT)(uiNumCols - uiStartCol);
	if( uiCharsLn > uiMaxWidth)
	{
		uiCharsLn = uiMaxWidth;
	}

	f_memset( szLineBuf, (FLMBYTE)32, uiCharsLn);

	szLineBuf[ uiCharsLn] = '\0';
	pszBuffer[ uiBufSize - 1] = '\0';
	uiCharCount = f_strlen( pszBuffer);
	if( uiCharCount > 0)
	{
		bGotChar = TRUE;
		uiBufPos = uiCharCount;
		uiCursorPos = (uiBufPos < uiCharsLn) ? uiBufPos : (uiCharsLn - 1);
	}
	else
	{
		uiBufPos = 0;
		uiCursorPos = 0;
	}

	bDone = FALSE;
	bInsert = TRUE;
	bRefresh = FALSE;
	uiChar = 0;

	while( !bDone)
	{
		if( (pWindow->pScreen->pFtxInfo->pbShutdown != NULL &&
			*(pWindow->pScreen->pFtxInfo->pbShutdown) == TRUE) ||
			(pWindow->pScreen->pbShutdown != NULL &&
			*(pWindow->pScreen->pbShutdown) == TRUE))
		{
			pszBuffer[ 0] = '\0';
			uiCharCount = 0;
			rc = FTXRC_SHUTDOWN;
			break;
		}

		if( !bGotChar)
		{
			if( (rc = FTXWinInputChar( pWindow, &uiChar)) != FTXRC_SUCCESS)
			{
				goto Exit;
			}
			bGotChar = TRUE;

			switch( uiChar)
			{
				case WPK_HOME:
				{
					uiBufPos = 0;
					uiCursorPos = 0;
					break;
				}
				case WPK_LEFT:
				{
					if( uiBufPos > 0)
					{
						uiBufPos--;
						if( uiCursorPos > 0)
						{
							uiCursorPos--;
						}
					}
					break;
				}
				case WPK_RIGHT:
				{
					if( uiBufPos < uiCharCount)
					{
						uiBufPos++;
						if( uiCursorPos < (uiCharsLn - 1))
						{
							uiCursorPos++;
						}
					}
					break;
				}
				case WPK_END:
				{
					if( uiBufPos != uiCharCount)
					{
						if( uiCharCount < (uiCharsLn - 1))
						{
							uiCursorPos = uiCharCount;
						}
						else
						{
							uiCursorPos = (FLMUINT)(uiCharsLn - 1);
						}
						uiBufPos = uiCharCount;
					}
					break;
				}
				case WPK_CTRL_LEFT:
				{
					if( uiBufPos > 0)
					{
						if( pszBuffer[ uiBufPos - 1] == ' ')
						{
							if( uiCursorPos > 0)
							{
								uiCursorPos--;
							}
							uiBufPos--;
						}
						while(
							uiBufPos > 0 && pszBuffer[ uiBufPos] == ' ')
						{
							uiBufPos--;
							if( uiCursorPos > 0)
							{
								uiCursorPos--;
							}
						}
						while( uiBufPos > 0 && pszBuffer[ uiBufPos] != ' ')
						{
							uiBufPos--;
							if( uiCursorPos > 0)
							{
								uiCursorPos--;
							}
						}
					}
					if( uiBufPos > 0 && pszBuffer[ uiBufPos] == ' ' &&
						uiBufPos < uiCharCount)
					{
						uiBufPos++;
						if( uiCursorPos < (uiCharsLn - 1))
						{
							uiCursorPos++;
						}
					}
					break;
				}
				case WPK_CTRL_RIGHT:
				{
					if( uiBufPos < uiCharCount)
					{
						while( uiBufPos < uiCharCount && pszBuffer[ uiBufPos] != ' ')
						{
							uiBufPos++;
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
						}
						while( uiBufPos < uiCharCount && pszBuffer[ uiBufPos] == ' ')
						{
							uiBufPos++;
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
						}
					}
					break;
				}
				case WPK_INSERT:
				{
					if( bInsert == TRUE)
					{
						bInsert = FALSE;
						FTXWinSetCursorType( pWindow,
							WPS_CURSOR_VISIBLE | WPS_CURSOR_BLOCK);
					}
					else
					{
						bInsert = TRUE;
						FTXWinSetCursorType( pWindow,
							WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE);
					}
					ftxCursorUpdate( pWindow->pScreen->pFtxInfo);
					break;
				}
				case WPK_DELETE:
				{
					if( uiBufPos < uiCharCount)
					{
						f_memmove( &(pszBuffer[ uiBufPos]),
							&(pszBuffer[ uiBufPos + 1]), uiCharCount - uiBufPos);
						uiCharCount--;
					}
					break;
				}
				case WPK_BACKSPACE:
				{
					if( uiBufPos > 0)
					{
						if( uiCursorPos > 0)
						{
							uiCursorPos--;
						}
						uiBufPos--;
						f_memmove( &(pszBuffer[ uiBufPos]),
							&(pszBuffer[ uiBufPos + 1]), uiCharCount - uiBufPos);
						uiCharCount--;
					}
					break;
				}
				case WPK_CTRL_B:
				{
					if( uiBufPos > 0)
					{
						uiCharCount -= uiBufPos;
						f_memmove( pszBuffer,
							&(pszBuffer[ uiBufPos]), uiCharCount + 1);
						uiBufPos = 0;
						uiCursorPos = 0;
					}
					break;
				}
				case WPK_CTRL_D:
				{
					if( uiBufPos < uiCharCount)
					{
						uiCharCount = uiBufPos;
						pszBuffer[ uiCharCount] = '\0';
					}
					break;
				}
				default:
				{
					if( (uiChar & 0xFF00) == 0)
					{
						if( bInsert && uiBufPos < uiCharCount &&
							uiCharCount < (uiBufSize - 1))
						{
							for( uiLoop = 0; uiLoop < uiCharCount - uiBufPos; uiLoop++)
							{
								pszBuffer[ uiCharCount - uiLoop] =
									pszBuffer[ uiCharCount - uiLoop - 1];
							}

							pszBuffer[ uiBufPos] = (char)uiChar;
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
							pszBuffer[ ++uiCharCount] = '\0';
							uiBufPos++;
						}
						else if( (uiBufPos < uiCharCount && !bInsert) ||
							uiCharCount < (uiBufSize - 1))
						{
							pszBuffer[ uiBufPos] = (char)uiChar;
							if( uiBufPos == uiCharCount)
							{
								pszBuffer[ ++uiCharCount] = '\0';
							}
							if( uiCursorPos < (uiCharsLn - 1))
							{
								uiCursorPos++;
							}
							uiBufPos++;
						}
					}
					else if( uiChar & 0xFF00)
					{
						bDone = TRUE;
						bGotChar = FALSE;
					}
				}
			}
		}

		if( bGotChar)
		{
			uiOutputStart = (FLMUINT)(uiBufPos - uiCursorPos);
			uiOutputEnd = (FLMUINT)(uiOutputStart + uiCharsLn);
			if( uiOutputEnd > uiCharCount)
			{
				uiOutputEnd = uiCharCount;
			}

			f_memset( szSnapBuf, (FLMBYTE)32, uiCharsLn);
			szSnapBuf[ uiCharsLn] = '\0';
			f_memmove( szSnapBuf, &(pszBuffer[ uiOutputStart]),
				(FLMUINT)(uiOutputEnd - uiOutputStart));

			uiCursorOutputPos = 0;
			uiLoop = 0;
			while( uiLoop < uiCharsLn)
			{
				if( szSnapBuf[ uiLoop] != szLineBuf[ uiLoop])
				{
					bRefresh = TRUE;
					uiCursorOutputPos = uiLoop;
					break;
				}
				uiLoop++;
			}

			uiLoop = uiCharsLn;
			while( uiLoop > uiCursorOutputPos)
			{
				if( szSnapBuf[ uiLoop - 1] != szLineBuf[ uiLoop - 1])
				{
					bRefresh = TRUE;
					break;
				}
				uiLoop--;
			}
			szSnapBuf[ uiLoop] = '\0';
			bGotChar = FALSE;
		}

		if( bRefresh)
		{
			f_memset( szLineBuf, (FLMBYTE)32, uiCharsLn);
			szLineBuf[ uiCharsLn] = '\0';
			f_memmove( szLineBuf, &(pszBuffer[ uiOutputStart]),
				(FLMUINT)(uiOutputEnd - uiOutputStart));

			FTXWinSetCursorPos( pWindow,
				(FLMUINT)(uiStartCol + uiCursorOutputPos), uiStartRow);
			FTXWinPrintStr( pWindow, &(szSnapBuf[ uiCursorOutputPos]));
			FTXWinSetCursorPos( pWindow,
				(FLMUINT)(uiStartCol + uiCursorPos), uiStartRow);

			FTXRefresh( pWindow->pScreen->pFtxInfo);
			bRefresh = FALSE;
		}
		else
		{
			FLMUINT	uiTmpCol = FTXWinGetCurrCol( pWindow);
			FLMUINT	uiTmpRow = FTXWinGetCurrRow( pWindow);

			if( uiTmpCol != uiStartCol + uiCursorPos || uiTmpRow != uiStartRow)
			{
				FTXWinSetCursorPos( pWindow, (FLMUINT)(uiStartCol + uiCursorPos),
					uiStartRow);
				ftxCursorUpdate( pWindow->pScreen->pFtxInfo);
			}
		}
	}

	if( puiTermChar)
	{
		*puiTermChar = uiChar;
	}

	if( puiCharCount)
	{
		*puiCharCount = uiCharCount;
	}

Exit:

	FTXWinSetCursorType( pWindow, uiSaveCursor);
	FTXWinSetScroll( pWindow, bSaveScroll);

	return( rc);

}


/****************************************************************************
Desc:		Line editor routine which assumes some defaults
****************************************************************************/
FLMUINT FTXLineEd(
	FTX_WINDOW *	pWindow,
	char *			pszBuffer,
	FLMUINT			uiBufSize)
{

	FLMUINT		uiTermChar;
	FLMUINT		uiStartCol;
	FLMUINT		uiStartRow;
	FLMUINT		uiCharsInput;
	FLMBOOL		bDone = FALSE;


	uiStartCol = FTXWinGetCurrCol( pWindow);
	uiStartRow = FTXWinGetCurrRow( pWindow);

	while( !bDone)
	{
		if( (pWindow->pScreen->pFtxInfo->pbShutdown != NULL &&
			*(pWindow->pScreen->pFtxInfo->pbShutdown) == TRUE) ||
			(pWindow->pScreen->pbShutdown != NULL &&
			*(pWindow->pScreen->pbShutdown) == TRUE))
		{
			pszBuffer[ 0] = '\0';
			uiCharsInput = 0;
			break;
		}

		pszBuffer[ 0] = '\0';
		if( FTXLineEdit( pWindow, pszBuffer, uiBufSize, 255, &uiCharsInput,
			&uiTermChar) != FTXRC_SUCCESS)
		{
			uiCharsInput = 0;
			*pszBuffer = '\0';
			goto Exit;
		}

		switch( uiTermChar)
		{
			case WPK_ENTER:
			{
				FTXWinPrintChar( pWindow, '\n');
				bDone = TRUE;
				break;
			}
			case WPK_ESC:
			{
				pszBuffer[ 0] = '\0';
				bDone = TRUE;
				break;
			}
			default:
			{
				FTXWinClearLine( pWindow, uiStartCol, uiStartRow);
				FTXWinSetCursorPos( pWindow, uiStartCol, uiStartRow);
				break;
			}
		}
	}

Exit:

	return( uiCharsInput);
}


/****************************************************************************
Desc: Displays a message window
*****************************************************************************/
FTXRCODE FTXMessageWindow(
	FTX_SCREEN *		pScreen,
	FLMUINT				uiBack,
	FLMUINT				uiFore,
	const char *		pszMessage1,
	const char *		pszMessage2,
	FTX_WINDOW **		ppWindow)
{
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows = 10;
	FLMUINT			uiNumWinCols;
	FLMUINT			uiNumCanvCols;
	FLMBYTE			pucTmpBuf[ 128];
	FLMUINT			uiMessageLen;
	FTX_WINDOW *	pWindow = NULL;
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( (rc = FTXScreenGetSize( pScreen,
		&uiNumCols, &uiNumRows)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	uiNumWinCols = uiNumCols - 8;

	if( (rc = FTXWinInit( pScreen, uiNumWinCols,
		uiNumWinRows, &pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinSetScroll( pWindow, FALSE)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	FTXWinSetCursorType( pWindow, WPS_CURSOR_INVISIBLE);

	if( (rc = FTXWinSetBackFore( pWindow,
		uiBack, uiFore)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinClear( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinDrawBorder( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( (rc = FTXWinMove( pWindow, (FLMUINT)((uiNumCols - uiNumWinCols) / 2),
		(FLMUINT)((uiNumRows - uiNumWinRows) / 2))) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	FTXWinGetCanvasSize( pWindow, &uiNumCanvCols, NULL);

	if( (rc = FTXWinOpen( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if( pszMessage1)
	{
		f_strncpy( pucTmpBuf, pszMessage1, uiNumCanvCols);
		pucTmpBuf[ uiNumCanvCols] = '\0';
		uiMessageLen = f_strlen( pucTmpBuf);

		FTXWinSetCursorPos( pWindow,
			(FLMUINT)((uiNumCanvCols - uiMessageLen) / 2), 3);
		FTXWinPrintf( pWindow, "%s", pucTmpBuf);
	}

	if( pszMessage2)
	{
		f_strncpy( pucTmpBuf, pszMessage2, uiNumCanvCols);
		pucTmpBuf[ uiNumCanvCols] = '\0';
		uiMessageLen = f_strlen( pucTmpBuf);

		FTXWinSetCursorPos( pWindow,
			(FLMUINT)((uiNumCanvCols - uiMessageLen) / 2), 4);
		FTXWinPrintf( pWindow, "%s", pucTmpBuf);
	}

	FTXRefresh( pScreen->pFtxInfo);
Exit:

	if( rc != FTXRC_SUCCESS && pWindow)
	{
		*ppWindow = NULL;
		FTXWinFree( &pWindow);
	}
	else
	{
		*ppWindow = pWindow;
	}

	return( rc);
}


/****************************************************************************
Desc: Displays a dialog-style message box
*****************************************************************************/
FTXRCODE FTXDisplayMessage(
	FTX_SCREEN *		pScreen,
	FLMUINT				uiBack,
	FLMUINT				uiFore,
	const char *		pszMessage1,
	const char *		pszMessage2,
	FLMUINT *			puiTermChar)
{
	FTX_WINDOW *	pWindow = NULL;
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( puiTermChar)
	{
		*puiTermChar = 0;
	}

	if ((rc = FTXMessageWindow( pScreen, uiBack, uiFore,
						pszMessage1, pszMessage2,
						&pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	for( ;;)
	{
		if( (pWindow->pScreen->pFtxInfo->pbShutdown != NULL &&
			*(pWindow->pScreen->pFtxInfo->pbShutdown) == TRUE) ||
			(pWindow->pScreen->pbShutdown != NULL &&
			*(pWindow->pScreen->pbShutdown) == TRUE))
		{
			rc = FTXRC_SHUTDOWN;
			goto Exit;
		}

		if( FTXWinTestKB( pWindow) == FTXRC_SUCCESS)
		{
			FLMUINT		uiChar;

			FTXWinInputChar( pWindow, &uiChar);

			if( uiChar == WPK_ESCAPE || uiChar == WPK_ENTER)
			{
				if( puiTermChar)
				{
					*puiTermChar = uiChar;
				}
				break;
			}

		}
		else
		{
			f_sleep( 10);
		}
	}

Exit:

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
FTXRCODE FTXGetInput(
	FTX_SCREEN *	pScreen,
	const char *	pszMessage,
	char *			pszResponse,
	FLMUINT			uiMaxRespLen,
	FLMUINT *		puiTermChar)
{
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumWinRows = 3;
	FLMUINT			uiNumWinCols;
	FTX_WINDOW *	pWindow = NULL;
	FTXRCODE			rc = FTXRC_SUCCESS;

	if ( (rc =
		FTXScreenGetSize( pScreen, &uiNumCols, &uiNumRows)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	uiNumWinCols = uiNumCols - 8;

	if ( (rc = FTXWinInit( pScreen, uiNumWinCols,
		uiNumWinRows, &pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ( (rc = FTXWinSetScroll( pWindow, FALSE)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	FTXWinSetCursorType( pWindow, WPS_CURSOR_UNDERLINE);

	if ( (rc =
		FTXWinSetBackFore( pWindow, WPS_CYAN, WPS_WHITE)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ( (rc = FTXWinClear( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ( (rc = FTXWinDrawBorder( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ( (rc = FTXWinMove( pWindow, (uiNumCols - uiNumWinCols) / 2,
		(uiNumRows - uiNumWinRows) / 2)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ( (rc = FTXWinOpen( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ( (rc = FTXWinClear( pWindow)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	FTXWinPrintf( pWindow, "%s: ", pszMessage);

	if ((rc = FTXLineEdit( pWindow, pszResponse, uiMaxRespLen, uiMaxRespLen,
		NULL, puiTermChar)) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

Exit:

	if( pWindow)
	{
		FTXWinFree( &pWindow);
	}

	return( rc);
}

/****************************************************************************
Desc:		Synchronizes the "camera-ready" display image with the "in-memory"
			image
****************************************************************************/
FSTATIC FTXRCODE ftxSyncImage(
	FTX_INFO *		pFtxInfo)
{

	FTX_WINDOW *	pWin;
	FTX_SCREEN *	pScreenCur;
	char *			pszWTBuf;
	char *			pszSBuf;
	FLMBYTE *		pucWTBackAttrib;
	FLMBYTE *		pucWTForeAttrib;
	FLMBYTE *		pucSBackAttrib;
	FLMBYTE *		pucSForeAttrib;
	FLMUINT			uiLoop;
	FLMUINT			uiOffset;
	FTXRCODE			rc = FTXRC_SUCCESS;

	pScreenCur = pFtxInfo->pScreenCur;

	ftxWinReset( pScreenCur->pWinImage);
	pWin = pScreenCur->pWinCur;

	if( pWin)
	{
		while( pWin->pWinNext)
		{
			pWin = pWin->pWinNext;
		}
	}

	while( pWin != NULL)
	{
		if( pWin->bOpen)
		{
			pszSBuf = pWin->pszBuffer;
			pucSBackAttrib = pWin->pucBackAttrib;
			pucSForeAttrib = pWin->pucForeAttrib;

			uiOffset = (FLMUINT)(((FLMUINT)pScreenCur->pWinImage->uiCols *
				(FLMUINT)pWin->uiUly) + (FLMUINT)pWin->uiUlx);

			pszWTBuf = pScreenCur->pWinImage->pszBuffer + uiOffset;
			pucWTBackAttrib = pScreenCur->pWinImage->pucBackAttrib + uiOffset;
			pucWTForeAttrib = pScreenCur->pWinImage->pucForeAttrib + uiOffset;

			for( uiLoop = 0; uiLoop < pWin->uiRows; uiLoop++)
			{
				f_memmove( pszWTBuf, pszSBuf, pWin->uiCols);
				f_memmove( pucWTBackAttrib, pucSBackAttrib, pWin->uiCols);
				f_memmove( pucWTForeAttrib, pucSForeAttrib, pWin->uiCols);

				pszSBuf += pWin->uiCols;
				pucSBackAttrib += pWin->uiCols;
				pucSForeAttrib += pWin->uiCols;

				pszWTBuf += pScreenCur->pWinImage->uiCols;
				pucWTBackAttrib += pScreenCur->pWinImage->uiCols;
				pucWTForeAttrib += pScreenCur->pWinImage->uiCols;
			}
		}
		pWin = pWin->pWinPrev;
	}

	return( rc);

}


/****************************************************************************
Desc:		Win32 display update
****************************************************************************/
#if defined( FLM_WIN)
FSTATIC FTXRCODE ftxWin32Refresh(
	FTX_INFO *			pFtxInfo)
{
	PCHAR_INFO			paCell;
	COORD					size;
	COORD					coord;
	SMALL_RECT			region;
	HANDLE				hStdOut;
	FLMUINT				uiLoop;
	FLMUINT				uiSubloop;
	FLMUINT				uiOffset;
	FLMUINT				uiLeft = 0;
	FLMUINT				uiRight = 0;
	FLMUINT				uiTop = 0;
	FLMUINT				uiBottom = 0;
	FLMBOOL				bTopSet = FALSE;
	FLMBOOL				bLeftSet = FALSE;
	FLMBOOL				bChanged = FALSE;
	FTX_WINDOW *		pWinImage;
	FTX_WINDOW *		pWinScreen;


	ftxSyncImage( pFtxInfo);
	pWinImage = pFtxInfo->pScreenCur->pWinImage;
	pWinScreen = pFtxInfo->pScreenCur->pWinScreen;

	for( uiLoop = 0; uiLoop < pWinImage->uiRows; uiLoop++)
	{
		for( uiSubloop = 0; uiSubloop < pWinImage->uiCols; uiSubloop++)
		{
			uiOffset = (FLMUINT)((uiLoop * (FLMUINT)(pWinImage->uiCols)) + uiSubloop);

			if( pWinImage->pszBuffer[ uiOffset] !=
					pWinScreen->pszBuffer[ uiOffset] ||
				pWinImage->pucForeAttrib[ uiOffset] !=
					pWinScreen->pucForeAttrib[ uiOffset] ||
				pWinImage->pucBackAttrib[ uiOffset] !=
					pWinScreen->pucBackAttrib[ uiOffset])
			{
				pWinScreen->pszBuffer[ uiOffset] =
					pWinImage->pszBuffer[ uiOffset];
				pWinScreen->pucForeAttrib[ uiOffset] =
					pWinImage->pucForeAttrib[ uiOffset];
				pWinScreen->pucBackAttrib[ uiOffset] =
					pWinImage->pucBackAttrib[ uiOffset];

				if( uiSubloop > uiRight)
				{
					uiRight = uiSubloop;
				}
				if( uiLoop > uiBottom)
				{
					uiBottom = uiLoop;
				}
				if( !bTopSet)
				{
					uiTop = uiLoop;
					bTopSet = TRUE;
				}
				if( !bLeftSet || uiLeft > uiSubloop)
				{
					uiLeft = uiSubloop;
					bLeftSet = TRUE;
				}
				if( !bChanged)
				{
					bChanged = TRUE;
				}
			}

			paCell = &(pFtxInfo->pCells[ ((uiLoop + pWinImage->uiUly) *
				pWinScreen->uiCols) + (uiSubloop + pWinImage->uiUlx)]);
			paCell->Char.AsciiChar = pWinImage->pszBuffer[ uiOffset];
			paCell->Attributes =
				(WORD)(((pWinImage->pucForeAttrib[ uiOffset] & 0x8F) |
				((pWinImage->pucBackAttrib[ uiOffset] << 4) & 0x7F)));
		}
	}

	if( bChanged)
	{
		if( (hStdOut = GetStdHandle( STD_OUTPUT_HANDLE)) ==
			INVALID_HANDLE_VALUE)
		{
			goto Exit;
		}

		size.X = (SHORT)pWinScreen->uiCols;
		size.Y = (SHORT)pWinScreen->uiRows;
		coord.X = (SHORT)uiLeft;
		coord.Y = (SHORT)uiTop;
		region.Left = (SHORT)uiLeft;
		region.Right = (SHORT)uiRight;
		region.Top = (SHORT)uiTop;
		region.Bottom = (SHORT)uiBottom;
		WriteConsoleOutput( hStdOut, pFtxInfo->pCells, size, coord, &region);
	}

Exit:

	return( FTXRC_SUCCESS);
}

#elif defined( FLM_NLM)

/****************************************************************************
Desc:		NLM display update
****************************************************************************/
FSTATIC FTXRCODE ftxNLMRefresh(
	FTX_INFO *			pFtxInfo)
{
	FLMUINT				uiLoop;
	FLMUINT				uiSubLoop;
	FLMUINT				uiOffset;
	FTX_WINDOW *		pWinImage;
	FTX_WINDOW *		pWinScreen;
	FLMBOOL				bModified;
	LONG					udCnt;
	LONG					udStartColumn;
	FLMUINT				uiStartOffset;
	BYTE *				pucStartValue;
	BYTE					attribute;
	BYTE					ucStartAttr;

	ftxSyncImage( pFtxInfo);
	pWinImage = pFtxInfo->pScreenCur->pWinImage;
	pWinScreen = pFtxInfo->pScreenCur->pWinScreen;

	for( uiLoop = 0; uiLoop < (FLMUINT)pWinImage->uiRows; uiLoop++)
	{
		bModified = FALSE;
		for( uiSubLoop = 0; uiSubLoop < pWinImage->uiCols; uiSubLoop++)
		{
			uiOffset = (FLMUINT)((uiLoop * (FLMUINT)(pWinImage->uiCols)) + uiSubLoop);
			if((pWinImage->pszBuffer[ uiOffset] !=
					pWinScreen->pszBuffer[ uiOffset] ||
				pWinImage->pucForeAttrib[ uiOffset] !=
					pWinScreen->pucForeAttrib[ uiOffset] ||
				pWinImage->pucBackAttrib[ uiOffset] !=
					pWinScreen->pucBackAttrib[ uiOffset]))
			{
				attribute = (pWinImage->pucBackAttrib[ uiOffset] << 4) +
						pWinImage->pucForeAttrib[ uiOffset];
				if (!bModified || attribute != ucStartAttr)
				{
					if (bModified)
					{
						(void)DisplayScreenTextWithAttribute( pFtxInfo->pvScreenHandle,
							(LONG)uiLoop, (LONG)udStartColumn, udCnt, ucStartAttr,
							pucStartValue);
					}
					ucStartAttr = attribute;
					udCnt = 0;
					uiStartOffset = uiOffset;
					udStartColumn = (LONG)uiSubLoop;
					bModified = TRUE;
					pucStartValue = (BYTE *)&pWinImage->pszBuffer[ uiOffset];
				}
				udCnt++;
			}
			else
			{
				if (bModified)
				{
					bModified = FALSE;
					(void)DisplayScreenTextWithAttribute( pFtxInfo->pvScreenHandle,
						(LONG)uiLoop, (LONG)udStartColumn, udCnt, ucStartAttr,
						pucStartValue);
				}
			}
		}
		if (bModified)
		{
			bModified = FALSE;
			(void)DisplayScreenTextWithAttribute( pFtxInfo->pvScreenHandle,
				(LONG)uiLoop, (LONG)udStartColumn, udCnt, ucStartAttr,
				pucStartValue);
		}
	}

	return( FTXRC_SUCCESS);
}

#else

/****************************************************************************
Desc:		Win16/Other display update
****************************************************************************/
FSTATIC FTXRCODE ftxRefresh(
	FTX_INFO *			pFtxInfo)
{

	char *			pszWTBuf;
	char *			pszSBuf;
	FLMBYTE *		pucWTBackAttrib;
	FLMBYTE *		pucWTForeAttrib;
	FLMBYTE *		pucSBackAttrib;
	FLMBYTE *		pucSForeAttrib;
	FLMUINT			uiChangeStart = 0;
	FLMUINT			uiChangeEnd = 0;
	FLMUINT			uiSaveChar;
	FLMBOOL			bChange;
	FLMUINT			uiLoop;
	FLMUINT			uiSubloop;
	FLMUINT			uiTempAttrib;
	FTX_WINDOW *	pWinImage;
	FTX_WINDOW *	pWinScreen;
	FTXRCODE			rc = FTXRC_SUCCESS;


	ftxSyncImage( pFtxInfo);
	pWinImage = pFtxInfo->pScreenCur->pWinImage;
	pWinScreen = pFtxInfo->pScreenCur->pWinScreen;

	pFtxInfo->uiCursorType = WPS_CURSOR_INVISIBLE;
	ftxDisplaySetCursorType( pFtxInfo, pFtxInfo->uiCursorType);

	pszSBuf = pWinScreen->pszBuffer;
	pucSBackAttrib = pWinScreen->pucBackAttrib;
	pucSForeAttrib = pWinScreen->pucForeAttrib;

	pszWTBuf = pWinImage->pszBuffer;
	pucWTBackAttrib = pWinImage->pucBackAttrib;
	pucWTForeAttrib = pWinImage->pucForeAttrib;

	for( uiLoop = 0; uiLoop < pWinScreen->uiRows; uiLoop++)
	{
		uiSubloop = 0;
		while( uiSubloop < pWinScreen->uiCols)
		{
			bChange = FALSE;
			if( pszSBuf[ uiSubloop] != pszWTBuf[ uiSubloop] ||
				pucSBackAttrib[ uiSubloop] != pucWTBackAttrib[ uiSubloop] ||
				pucSForeAttrib[ uiSubloop] != pucWTForeAttrib[ uiSubloop])
			{
				bChange = TRUE;
				uiChangeStart = uiSubloop;
				uiChangeEnd = uiSubloop;

				while( pucWTBackAttrib[ uiChangeStart] ==
					pucWTBackAttrib[ uiSubloop] &&
					pucWTForeAttrib[ uiChangeStart] == pucWTForeAttrib[ uiSubloop] &&
					uiSubloop < pWinScreen->uiCols)
				{
					if( pszSBuf[ uiSubloop] != pszWTBuf[ uiSubloop] ||
						pucSBackAttrib[ uiSubloop] != pucWTBackAttrib[ uiSubloop] ||
						pucSForeAttrib[ uiSubloop] != pucWTForeAttrib[ uiSubloop])
					{
						uiChangeEnd = uiSubloop;
					}
					pszSBuf[ uiSubloop] = pszWTBuf[ uiSubloop];
					pucSBackAttrib[ uiSubloop] = pucWTBackAttrib[ uiSubloop];
					pucSForeAttrib[ uiSubloop] = pucWTForeAttrib[ uiSubloop];
					uiSubloop++;
				}
				uiSubloop--;
			}

			if( bChange)
			{
				ftxDisplaySetCursorPos( pFtxInfo, uiChangeStart, uiLoop);
				uiSaveChar = pszSBuf[ uiChangeEnd + 1];
				pszSBuf[ uiChangeEnd + 1] = '\0';
				uiTempAttrib = (pucSBackAttrib [uiChangeStart] << 4) +
						pucSForeAttrib [uiChangeStart];
				ftxDisplaySetBackFore( pucSBackAttrib[ uiChangeStart],
					pucSForeAttrib[ uiChangeStart]);
				ftxDisplayStrOut( &(pszSBuf[ uiChangeStart]), uiTempAttrib);
				pszSBuf[ uiChangeEnd + 1] = uiSaveChar;
			}

			uiSubloop++;
		}

		pszSBuf += pWinScreen->uiCols;
		pucSBackAttrib += pWinScreen->uiCols;
		pucSForeAttrib += pWinScreen->uiCols;

		pszWTBuf += pWinImage->uiCols;
		pucWTBackAttrib += pWinImage->uiCols;
		pucWTForeAttrib += pWinImage->uiCols;

	}

	return( rc);
}
#endif


/****************************************************************************
Desc:		Initializes / resets a window object
****************************************************************************/
FSTATIC FTXRCODE ftxWinReset(
	FTX_WINDOW *		pWindow)
{

	FLMUINT			uiSize;
	FTXRCODE			rc = FTXRC_SUCCESS;


	uiSize = (FLMUINT)(pWindow->uiRows * pWindow->uiCols);

	f_memset( pWindow->pszBuffer, (FLMBYTE)' ', uiSize);
	f_memset( pWindow->pucBackAttrib, (FLMBYTE)pWindow->pScreen->uiBackground, uiSize);
	f_memset( pWindow->pucForeAttrib, (FLMBYTE)pWindow->pScreen->uiForeground, uiSize);

	pWindow->uiBackground = pWindow->pScreen->uiBackground;
	pWindow->uiForeground = pWindow->pScreen->uiForeground;

	pWindow->uiCurX = pWindow->uiOffset;
	pWindow->uiCurY = pWindow->uiOffset;

	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for freeing a screen object
****************************************************************************/
FSTATIC FTXRCODE ftxScreenFree(
	FTX_SCREEN *	pScreen)
{
	FTX_WINDOW *	pWin;
	FTXRCODE			rc = FTXRC_SUCCESS;


	while( (pWin = pScreen->pWinCur) != NULL)
	{
		ftxWinFree( pWin);
	}

	if( pScreen == pScreen->pFtxInfo->pScreenCur)
	{
		pScreen->pFtxInfo->pScreenCur = pScreen->pScreenNext;
		if( pScreen->pFtxInfo->pScreenCur)
		{
			pScreen->pFtxInfo->pScreenCur->pScreenPrev = NULL;
		}
	}
	else
	{
		if( pScreen->pScreenNext)
		{
			pScreen->pScreenNext->pScreenPrev = pScreen->pScreenPrev;
		}

		if( pScreen->pScreenPrev)
		{
			pScreen->pScreenPrev->pScreenNext = pScreen->pScreenNext;
		}
	}

	f_mutexDestroy( &(pScreen->hScreenMutex));
	f_semDestroy( &(pScreen->hKeySem));
	f_free( &pScreen);

	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for freeing a window object
****************************************************************************/
FSTATIC FTXRCODE ftxWinFree(
	FTX_WINDOW *	pWindow)
{
	FTX_WINDOW *	pWin;
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( pWindow->bOpen)
	{
		ftxWinClose( pWindow);
	}

	pWin = pWindow->pScreen->pWinCur;
	while( pWin != pWindow)
	{
		pWin = pWin->pWinNext;
		if( pWin == NULL)
		{
			break;
		}
	}

	if( pWin)
	{
		if( pWin == pWindow->pScreen->pWinCur)
		{
			pWindow->pScreen->pWinCur = pWin->pWinNext;
			if( pWindow->pScreen->pWinCur)
			{
				pWindow->pScreen->pWinCur->pWinPrev = NULL;
			}
		}
		else
		{
			if( pWin->pWinNext)
			{
				pWin->pWinNext->pWinPrev = pWin->pWinPrev;
			}

			if( pWin->pWinPrev)
			{
				pWin->pWinPrev->pWinNext = pWin->pWinNext;
			}
		}
	}

	f_free( &(pWindow->pszBuffer));
	f_free( &(pWindow->pucForeAttrib));
	f_free( &(pWindow->pucBackAttrib));
	f_free( &pWindow);

	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for opening a window
****************************************************************************/
FSTATIC FTXRCODE ftxWinOpen(
	FTX_WINDOW *	pWindow)
{

	FTXRCODE				rc = FTXRC_SUCCESS;


	if( pWindow->bOpen)
	{
		goto Exit;
	}

	if( pWindow != pWindow->pScreen->pWinCur)
	{
		if( pWindow->pWinNext != NULL)
		{
			pWindow->pWinNext->pWinPrev = pWindow->pWinPrev;
		}

		if( pWindow->pWinPrev != NULL)
		{
			pWindow->pWinPrev->pWinNext = pWindow->pWinNext;
		}

		pWindow->pWinPrev = NULL;
		pWindow->pWinNext = pWindow->pScreen->pWinCur;
		if( pWindow->pWinNext)
		{
			pWindow->pWinNext->pWinPrev = pWindow;
		}
		pWindow->pScreen->pWinCur = pWindow;
	}
	pWindow->bOpen = TRUE;

Exit:

	pWindow->pScreen->bChanged = TRUE;
	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for closing a window
****************************************************************************/
FSTATIC FTXRCODE ftxWinClose(
	FTX_WINDOW *	pWindow)
{

	FTX_WINDOW *			pWinTmp;
	FTXRCODE					rc = FTXRC_SUCCESS;


	if( pWindow->pScreen->pWinCur == pWindow &&
		pWindow->pWinNext != NULL)
	{
		pWindow->pScreen->pWinCur = pWindow->pWinNext;
	}

	if( pWindow->pWinNext != NULL)
	{
		pWindow->pWinNext->pWinPrev = pWindow->pWinPrev;
	}

	if( pWindow->pWinPrev != NULL)
	{
		pWindow->pWinPrev->pWinNext = pWindow->pWinNext;
	}

	pWinTmp = pWindow->pScreen->pWinCur;
	while( pWinTmp->pWinNext)
	{
		pWinTmp = pWinTmp->pWinNext;
	}

	pWindow->pWinPrev = pWinTmp;
	pWinTmp->pWinNext = pWindow;
	pWindow->pWinNext = NULL;
	pWindow->bOpen = FALSE;
	pWindow->pScreen->bChanged = TRUE;

	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for printing a character
****************************************************************************/
FSTATIC FTXRCODE ftxWinPrintChar(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiChar)
{
	FLMBOOL			bChanged = FALSE;
	FLMUINT			uiOffset;
	FLMUINT			uiRow;
	FTXRCODE			rc = FTXRC_SUCCESS;


	uiOffset = (FLMUINT)((FLMUINT)(pWindow->uiCurY * pWindow->uiCols) +
					pWindow->uiCurX);

	if( uiOffset >= ((FLMUINT)(pWindow->uiCols) * pWindow->uiRows))
	{
		goto Exit;
	}

	if( (uiChar > 31 && uiChar <= 126) || pWindow->bForceOutput)
	{
		if( (FLMUINT)pWindow->pszBuffer[ uiOffset] != uiChar ||
			pWindow->pucForeAttrib[ uiOffset] != pWindow->uiForeground ||
			pWindow->pucBackAttrib[ uiOffset] != pWindow->uiBackground)
		{
			pWindow->pszBuffer[ uiOffset] = (FLMBYTE)uiChar;
			pWindow->pucForeAttrib[ uiOffset] = (FLMBYTE)pWindow->uiForeground;
			pWindow->pucBackAttrib[ uiOffset] = (FLMBYTE)pWindow->uiBackground;
			bChanged = TRUE;
		}
		pWindow->uiCurX++;
	}
	else
	{
		switch( uiChar)
		{
			case 9: /* TAB */
			{
				pWindow->uiCurX += (FLMUINT)(8 - (pWindow->uiCurX % 8));

				if( pWindow->uiCurX > pWindow->uiCols)
				{
					pWindow->uiCurX = pWindow->uiOffset;
					pWindow->uiCurY++;
				}
				break;
			}
			case 10: /* LF */
			{
				pWindow->uiCurX = pWindow->uiOffset;
				pWindow->uiCurY++;
				break;
			}
			case 13: /* CR */
			{
				pWindow->uiCurX = pWindow->uiOffset;
				break;
			}
		}
	}

	if( pWindow->uiCurX + pWindow->uiOffset >= pWindow->uiCols)
	{
		if( pWindow->bNoLineWrap)
		{
			pWindow->uiCurX = (pWindow->uiCols - pWindow->uiOffset) - 1;
		}
		else
		{
			pWindow->uiCurY++;
			pWindow->uiCurX = pWindow->uiOffset;
		}
	}

	if( pWindow->uiCurY + pWindow->uiOffset >= pWindow->uiRows)
	{
		pWindow->uiCurY = (FLMUINT)(pWindow->uiRows - pWindow->uiOffset - 1);
		if( pWindow->bScroll)
		{
			if( pWindow->uiRows - pWindow->uiOffset > 1)
			{
				if( pWindow->uiOffset)
				{
					for( uiRow = pWindow->uiOffset;
						uiRow < pWindow->uiRows - (2 * pWindow->uiOffset); uiRow++)
					{
						uiOffset = (FLMUINT)((FLMUINT)(uiRow * pWindow->uiCols) +
							pWindow->uiOffset);
						f_memmove( pWindow->pszBuffer + uiOffset,
							pWindow->pszBuffer + uiOffset + pWindow->uiCols,
							pWindow->uiCols - (2 * pWindow->uiOffset));

						f_memmove( pWindow->pucForeAttrib + uiOffset,
							pWindow->pucForeAttrib + uiOffset + pWindow->uiCols,
							pWindow->uiCols - (2 * pWindow->uiOffset));

						f_memmove( pWindow->pucBackAttrib + uiOffset,
							pWindow->pucBackAttrib + uiOffset + pWindow->uiCols,
							pWindow->uiCols - (2 * pWindow->uiOffset));
					}
				}
				else
				{
					f_memmove( pWindow->pszBuffer,
						pWindow->pszBuffer + pWindow->uiCols,
						(pWindow->uiRows - 1) * pWindow->uiCols);

					f_memmove( pWindow->pucForeAttrib,
						pWindow->pucForeAttrib + pWindow->uiCols,
						(pWindow->uiRows - 1) * pWindow->uiCols);

					f_memmove( pWindow->pucBackAttrib,
						pWindow->pucBackAttrib + pWindow->uiCols,
						(pWindow->uiRows - 1) * pWindow->uiCols);
				}
			}

			uiOffset = (FLMUINT)(((FLMUINT)(pWindow->uiRows - pWindow->uiOffset - 1) *
				pWindow->uiCols) + pWindow->uiOffset);

			f_memset( pWindow->pszBuffer + uiOffset, (FLMBYTE)' ',
				pWindow->uiCols - (2 * pWindow->uiOffset));
			f_memset( pWindow->pucForeAttrib + uiOffset, (FLMBYTE)pWindow->uiForeground,
				pWindow->uiCols - (2 * pWindow->uiOffset));
			f_memset( pWindow->pucBackAttrib + uiOffset, (FLMBYTE)pWindow->uiBackground,
				pWindow->uiCols - (2 * pWindow->uiOffset));
			bChanged = TRUE;
		}
	}

Exit:

	if( pWindow->bOpen && bChanged)
	{
		pWindow->pScreen->bChanged = TRUE;
	}

	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for updating the cursor
****************************************************************************/
FSTATIC FTXRCODE ftxCursorUpdate(
	FTX_INFO *		pFtxInfo)
{
	FLMUINT			uiCurX;
	FLMUINT			uiCurY;
	FTX_WINDOW *	pWinCur;
	FTXRCODE			rc = FTXRC_SUCCESS;


	if( pFtxInfo->pScreenCur && pFtxInfo->pScreenCur->bUpdateCursor)
	{
		pWinCur = pFtxInfo->pScreenCur->pWinCur;
		if( pWinCur && pWinCur->bOpen)
		{
			uiCurX = (FLMUINT)(pWinCur->uiUlx + pWinCur->uiCurX);
			uiCurY = (FLMUINT)(pWinCur->uiUly + pWinCur->uiCurY);

			ftxDisplaySetCursorPos( pFtxInfo, uiCurX, uiCurY);
			ftxDisplaySetCursorType( pFtxInfo, pWinCur->uiCursorType);
			pFtxInfo->uiCursorType = pWinCur->uiCursorType;
		}
		else
		{
			ftxDisplaySetCursorType( pFtxInfo, WPS_CURSOR_INVISIBLE);
			pFtxInfo->uiCursorType = WPS_CURSOR_INVISIBLE;
		}
		pFtxInfo->pScreenCur->bUpdateCursor = FALSE;
	}

	return( rc);

}

/****************************************************************************
Desc:		Low-level routine for clearing a line
****************************************************************************/
FSTATIC FTXRCODE ftxWinClearLine(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{

	FLMUINT		uiOffset;
	FLMUINT		uiSize;
	FTXRCODE		rc = FTXRC_SUCCESS;


	if( (pWindow->uiRows - (2 * pWindow->uiOffset)) > uiRow &&
		(pWindow->uiCols - (2 * pWindow->uiOffset)) > uiCol)
	{
		pWindow->uiCurY = (FLMUINT)(uiRow + pWindow->uiOffset);
		pWindow->uiCurX = (FLMUINT)(uiCol + pWindow->uiOffset);

		uiOffset = ((FLMUINT)(pWindow->uiCurY) * pWindow->uiCols) +
			pWindow->uiCurX;

		uiSize = (FLMUINT)(pWindow->uiCols - pWindow->uiOffset) - pWindow->uiCurX;

		f_memset( pWindow->pszBuffer + uiOffset, (FLMBYTE)' ', uiSize);
		f_memset( pWindow->pucForeAttrib + uiOffset, (FLMBYTE)pWindow->uiForeground, uiSize);
		f_memset( pWindow->pucBackAttrib + uiOffset, (FLMBYTE)pWindow->uiBackground, uiSize);

		pWindow->uiCurY = (FLMUINT)(uiRow + pWindow->uiOffset);
		pWindow->uiCurX = (FLMUINT)(uiCol + pWindow->uiOffset);
		if( pWindow->bOpen)
		{
			pWindow->pScreen->bChanged = TRUE;
		}
	}

	return( rc);

}


/****************************************************************************
Desc:		Low-level routine for setting the cursor's position
****************************************************************************/
FSTATIC FTXRCODE ftxWinSetCursorPos(
	FTX_WINDOW *	pWindow,
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FTXRCODE			rc = FTXRC_SUCCESS;

	if( (pWindow->uiRows - (2 * pWindow->uiOffset)) > uiRow &&
		(pWindow->uiCols - (2 * pWindow->uiOffset)) > uiCol)
	{
		pWindow->uiCurY = (FLMUINT)(uiRow + pWindow->uiOffset);
		pWindow->uiCurX = (FLMUINT)(uiCol + pWindow->uiOffset);
		pWindow->pScreen->bUpdateCursor = TRUE;
	}

	return( rc);
}


/****************************************************************************
Desc:		Initializes the "physical" screen
****************************************************************************/
FSTATIC FTXRCODE ftxDisplayInit(
	FTX_INFO *		pFtxInfo,
	FLMUINT			uiRows,		// 0xFF means use current screen height.
	FLMUINT			uiCols,		// 0xFF means use current screen width.
	const char *	pszTitle)
{
#if defined( FLM_WIN)

	// Allocate a console if the application does not already have
	// one.

	if( AllocConsole())
	{
		gv_bAllocatedConsole = TRUE;
	}

	// Set up the console.

	if( (gv_hStdOut = GetStdHandle( STD_OUTPUT_HANDLE)) ==
		INVALID_HANDLE_VALUE)
	{
		return( FTXRC_MEM);
	}

	// If FTX allocated the console, re-size the console to match
	// the requested size

	if( gv_bAllocatedConsole)
	{
		COORD			conSize;

		conSize.X = (SHORT)uiCols;
		conSize.Y = (SHORT)uiRows;
		
		SetConsoleScreenBufferSize( gv_hStdOut, conSize);
	}

	SMALL_RECT	conRec;

	conRec.Left = 0;
	conRec.Top = 0;
	conRec.Right = (SHORT)(uiCols - 1);
	conRec.Bottom = (SHORT)(uiRows - 1);
	SetConsoleWindowInfo( gv_hStdOut, TRUE, &conRec);

	if( (gv_hStdIn = GetStdHandle( STD_INPUT_HANDLE)) ==
		INVALID_HANDLE_VALUE)
	{
		return( FTXRC_MEM);
	}

	/* Save information about the screen attributes */

	if( !GetConsoleScreenBufferInfo( gv_hStdOut, &gv_ConsoleScreenBufferInfo))
	{
		return( FTXRC_MEM);
	}

	FlushConsoleInputBuffer( gv_hStdIn);
	SetConsoleMode( gv_hStdIn, 0);
	SetConsoleTitle( (LPCTSTR)pszTitle);

#elif defined( FLM_UNIX)

	F_UNREFERENCED_PARM( uiRows);
	F_UNREFERENCED_PARM( uiCols);
	F_UNREFERENCED_PARM( pszTitle);
	
	ftxUnixDisplayInit();

#else

	F_UNREFERENCED_PARM( uiRows);
	F_UNREFERENCED_PARM( uiCols);
	F_UNREFERENCED_PARM( pszTitle);

#endif

	/* Set default cursor type */

	ftxDisplaySetCursorType( pFtxInfo, WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE);

	/* Set default foreground/background colors */

	ftxDisplaySetBackFore( WPS_BLACK, WPS_LIGHTGRAY);

	gv_bDisplayInitialized = TRUE;
	return( FTXRC_SUCCESS);
}


/****************************************************************************
Desc:		Restores the "physical" screen to an initial state
****************************************************************************/
FSTATIC void ftxDisplayExit( void)
{
	if( gv_bDisplayInitialized)
	{
#if defined( FLM_UNIX)

		ftxUnixDisplayFree();

#elif defined( FLM_WIN)

		/*
		Reset the console cursor
		*/

		CONSOLE_CURSOR_INFO     CursorInfo;

		CursorInfo.bVisible = TRUE;
		CursorInfo.dwSize = (DWORD)25;
		SetConsoleCursorInfo( gv_hStdOut, &CursorInfo);

		/*
		Reset the screen attributes
		*/

		SetConsoleTextAttribute( gv_hStdOut,
			gv_ConsoleScreenBufferInfo.wAttributes);

		/*
		Free the console if the application allocated one.
		*/

		if( gv_bAllocatedConsole)
		{
			(void)FreeConsole();
			gv_bAllocatedConsole = FALSE;
		}

#endif
	}

	gv_bDisplayInitialized = FALSE;
	return;
}


/****************************************************************************
Desc:    Resets (clears) the "physical" screen and positions the cursor
			at the origin
****************************************************************************/
FSTATIC void ftxDisplayReset(
	FTX_INFO *		pFtxInfo)
{
#if defined( FLM_WIN)
	{
		COORD									coord;
		DWORD									dCharWritten;
		DWORD									dCharsToWrite;
		CONSOLE_SCREEN_BUFFER_INFO		Console;

		F_UNREFERENCED_PARM( pFtxInfo);

		if( GetConsoleScreenBufferInfo( gv_hStdOut, &Console) == FALSE)
			return;

		dCharsToWrite = Console.dwMaximumWindowSize.X *
			Console.dwMaximumWindowSize.Y;

		coord.X = 0;
		coord.Y = 0;

		// Fill the screen with spaces
		FillConsoleOutputCharacter( gv_hStdOut, ' ',
				dCharsToWrite, coord, &dCharWritten);

		// Set the screen colors back to default colors.
		FillConsoleOutputAttribute( gv_hStdOut, FOREGROUND_INTENSITY,
				dCharsToWrite, coord, &dCharWritten);
	}

#elif defined( FLM_UNIX)
	F_UNREFERENCED_PARM( pFtxInfo);

	ftxUnixDisplayReset();
#elif defined( FLM_NLM)
	ClearScreen( pFtxInfo->pvScreenHandle);
#else
	F_UNREFERENCED_PARM( pFtxInfo);
	clrscr();	/* Clear entire screen */

#endif
	return;
}


/****************************************************************************
Desc: Returns the size of the "physical" screen in columns and rows
****************************************************************************/
FSTATIC void ftxDisplayGetSize(
	FLMUINT *	puiNumColsRV,
	FLMUINT *	puiNumRowsRV
	)
{
#if defined( FLM_WIN)
	CONSOLE_SCREEN_BUFFER_INFO Console;

	if( GetConsoleScreenBufferInfo( gv_hStdOut, &Console) == FALSE)
		return;

	*puiNumColsRV = (FLMUINT)Console.dwSize.X;
	*puiNumRowsRV = (FLMUINT)Console.dwSize.Y;

#elif defined( FLM_UNIX)
	ftxUnixDisplayGetSize( puiNumColsRV, puiNumRowsRV);
#else

	WORD	screenHeight;
	WORD	screenWidth;

	GetScreenSize( &screenHeight, &screenWidth);

	*puiNumColsRV = (FLMUINT)screenWidth;
	*puiNumRowsRV = (FLMUINT)screenHeight;

	/* NLM may call GetScreenInfo() - but the screenID must be passed in */
#endif

}

/****************************************************************************
Desc : Sets the "physical" cursor attributes
****************************************************************************/
FSTATIC FLMBOOL ftxDisplaySetCursorType(
	FTX_INFO *	pFtxInfo,
	FLMUINT		uiType)
{
#if defined( FLM_WIN)
	{
		CONSOLE_CURSOR_INFO     CursorInfo;


		F_UNREFERENCED_PARM( pFtxInfo);
		if( uiType & WPS_CURSOR_INVISIBLE)
		{
			CursorInfo.dwSize = (DWORD)99;
			CursorInfo.bVisible = FALSE;
		}
		else
		{
			CursorInfo.bVisible = TRUE;
			if( uiType & WPS_CURSOR_BLOCK)

			{
				CursorInfo.dwSize = (DWORD)99;
			}
			else
			{
				CursorInfo.dwSize = (DWORD)25;
			}
		}

		return( (FLMBOOL)SetConsoleCursorInfo( gv_hStdOut, &CursorInfo));
	}

#elif defined( FLM_NLM)

	if (uiType & WPS_CURSOR_INVISIBLE)
	{
		DisableInputCursor( pFtxInfo->pvScreenHandle);
	}
	else if (uiType & WPS_CURSOR_BLOCK)
	{
		EnableInputCursor( pFtxInfo->pvScreenHandle);
		SetCursorStyle( pFtxInfo->pvScreenHandle,
			0x0c00);	// CURSOR_BLOCK
	}
	else
	{
		EnableInputCursor( pFtxInfo->pvScreenHandle);
		SetCursorStyle( pFtxInfo->pvScreenHandle,
			0x0c0B);	// CURSOR_NORMAL
	}

	return( TRUE);
#else

	F_UNREFERENCED_PARM( pFtxInfo);
	F_UNREFERENCED_PARM( uiType);
	return( FALSE);

#endif
}


/****************************************************************************
Desc:    Sets the "physical" cursor to the column and row specified
****************************************************************************/
FSTATIC void ftxDisplaySetCursorPos(
	FTX_INFO *	pFtxInfo,
	FLMUINT		uiCol,
	FLMUINT		uiRow)
{
	if( uiCol == (FLMUINT)255 ||
		uiRow == (FLMUINT)255)
	{
		return;
	}

#if defined( FLM_NLM)
	PositionOutputCursor( pFtxInfo->pvScreenHandle,
			(WORD)uiRow, (WORD)uiCol);

	// Wake up the input thread and send it a special code to
	// cause the cursor to be re-positioned.
	UngetKey( (struct ScreenStruct *)pFtxInfo->pvScreenHandle,
		0xFE,
		(BYTE)uiRow, (BYTE)uiCol, 0);
#elif defined( FLM_WIN)

	{
		COORD    coord;

		F_UNREFERENCED_PARM( pFtxInfo);

		coord.X = (SHORT)uiCol;
		coord.Y = (SHORT)uiRow;
		SetConsoleCursorPosition( gv_hStdOut, coord);
	}

#elif defined( FLM_UNIX)

	F_UNREFERENCED_PARM( pFtxInfo);
	ftxUnixDisplaySetCursorPos( uiCol, uiRow);
	ftxUnixDisplayRefresh();

#else
	F_UNREFERENCED_PARM( pFtxInfo);

	/* gotoxy() uses 1 based numbers for borland/OS2- "Neither arg can be 0" */

	gotoxy( (FLMUINT)(uiCol + 1), (FLMUINT)(uiRow + 1));

#endif
}


/****************************************************************************
Desc:    Outputs a string to the "physical" screen
****************************************************************************/
#if defined( FLM_UNIX)
FSTATIC FLMUINT ftxDisplayStrOut(
	const char *	pszString,
	FLMUINT			uiAttr)
{
	while( *pszString)
	{
		ftxUnixDisplayChar( (FLMUINT)*pszString, uiAttr);
		pszString++;
	}

	return( (FLMUINT)0);
}
#endif

/****************************************************************************
Desc:    Set the background and foreground colors of the "physical" screen
****************************************************************************/
FSTATIC void ftxDisplaySetBackFore(
	FLMUINT	uiBackground,
	FLMUINT	uiForeground
	)
{

#if defined( FLM_WIN)

	FLMUINT	uiAttrib = 0;

	uiAttrib = (uiForeground & 0x8F) | (( uiBackground << 4) & 0x7F);
	SetConsoleTextAttribute( gv_hStdOut, (WORD)uiAttrib);

#else

	F_UNREFERENCED_PARM( uiBackground);
	F_UNREFERENCED_PARM( uiForeground);

#endif

}


/****************************************************************************
Desc:    Gets a character from the "physical" keyboard and converts keyboard
			sequences/scan codes to WPK key strokes.
Ret:     WPK Character
Notes:   Does not support WP extended character input
****************************************************************************/
FLMUINT ftxKBGetChar( void)
{
	FLMUINT	uiChar = 0;
#ifdef FLM_NLM
	BYTE		scanCode;
	BYTE		keyType;
	BYTE		keyValue;
	BYTE		keyStatus;
#endif

#if defined( FLM_NLM)

get_key:

	// Are we exiting?

	if( gv_pFtxInfo->pbShutdown != NULL)
	{
		if( *(gv_pFtxInfo->pbShutdown) == TRUE)
		{
			return( uiChar);
		}
	}

	// Get a key

	GetKey( gv_pFtxInfo->pvScreenHandle,
		&keyType, &keyValue,
		&keyStatus, &scanCode, 0);

	switch (keyType)
	{
		case 0:	// NORMAL_KEY
			if (keyStatus & 4)		// CTRL key pressed
			{
				uiChar = WPK_CTRL_A + keyValue - 1;
			}
			else if (keyStatus & 8)	// ALT key pressed
			{
				uiChar = ScanCodeToWPK [scanCode];
			}
			else	// Handles SHIFT key case.
			{
				uiChar = (FLMUINT)keyValue;
			}
			break;
		case 1:	// FUNCTION_KEY
			uiChar = ScanCodeToWPK [scanCode];
			if (keyStatus & 4)		// CTRL key pressed
			{
				uiChar = WPK_CTRL_F1 + (uiChar - WPK_F1);
			}
			else if (keyStatus & 8)	// ALT key pressed
			{
				uiChar = WPK_ALT_F1 + (uiChar - WPK_F1);
			}
			else if (keyStatus & 2)	// SHIFT key pressed
			{
				uiChar = WPK_SF1 + (uiChar - WPK_F1);
			}
			break;
		case 2:	// ENTER_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_ENTER;
			}
			else
			{
				uiChar = WPK_ENTER;
			}
			break;
		case 3:	// ESCAPE_KEY
			uiChar = WPK_ESCAPE;
			break;
		case 4:	// BACKSPACE_KEY
			uiChar = WPK_BACKSPACE;
			break;
		case 5:	// DELETE_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_DELETE;
			}
			else
			{
				uiChar = WPK_DELETE;
			}
			break;
		case 6:	// INSERT_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_INSERT;
			}
			else
			{
				uiChar = WPK_INSERT;
			}
			break;
		case 7:	// CURSOR_UP_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_UP;
			}
			else
			{
				uiChar = WPK_UP;
			}
			break;
		case 8:	// CURSOR_DOWN_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_DOWN;
			}
			else
			{
				uiChar = WPK_DOWN;
			}
			break;
		case 9:	// CURSOR_RIGHT_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_RIGHT;
			}
			else
			{
				uiChar = WPK_RIGHT;
			}
			break;
		case 10:	// CURSOR_LEFT_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_LEFT;
			}
			else
			{
				uiChar = WPK_LEFT;
			}
			break;
		case 11:	// CURSOR_HOME_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_HOME;
			}
			else
			{
				uiChar = WPK_HOME;
			}
			break;
		case 12:	// CURSOR_END_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_END;
			}
			else
			{
				uiChar = WPK_END;
			}
			break;
		case 13:	// CURSOR_PUP_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_PGUP;
			}
			else
			{
				uiChar = WPK_PGUP;
			}
			break;
		case 14:	// CURSOR_PDOWN_KEY
			if (keyStatus & 4)
			{
				uiChar = WPK_CTRL_PGDN;
			}
			else
			{
				uiChar = WPK_PGDN;
			}
			break;
		case 0xFE:
			// Re-position the input cursor
			PositionInputCursor( gv_pFtxInfo->pvScreenHandle,
				(WORD)keyValue, (WORD)keyStatus);
			goto get_key;
		case 0xFF:
			// Ping
			uiChar = (FLMUINT)0xFFFF;
			break;
		default:
			uiChar = (FLMUINT)keyValue;
			break;
	}
#elif defined( FLM_WIN)
	uiChar = (FLMUINT) ftxWin32KBGetChar();
#elif defined( FLM_UNIX)
	uiChar = ftxUnixKBGetChar();
#else
	uiChar = (FLMUINT) getch();
#endif

#if defined( FLM_WIN)
	if( uiChar == 0 || uiChar == 0x00E0)
	{
		FLMUINT	scanCode = (FLMUINT)ftxWin32KBGetChar();
		if( scanCode < (sizeof( ScanCodeToWPK) / sizeof( FLMUINT)))
		{
			uiChar = ScanCodeToWPK[ scanCode ];
		}
	}
	else if( (uiChar > 0) && (uiChar < 0x20))
	{
		switch( uiChar)
		{
			case  0x0D:
				uiChar = WPK_ENTER;            break;
			case  0x1B:
				uiChar = WPK_ESCAPE;           break;
			case  0x08:
				uiChar = WPK_BACKSPACE;        break;
			case  0x09:
				uiChar = WPK_TAB;              break;
			case  0x0A:
				uiChar = WPK_CTRL_ENTER;       break;

			/* Default is a ctrl-letter code */
			default:
				uiChar = (FLMUINT)((WPK_CTRL_A - 1) + uiChar);
				break;
		}
	}
#endif
	return( uiChar);
}

/****************************************************************************
Desc:    Returns TRUE if a key is available from the "physical" keyboard
****************************************************************************/
FLMBOOL ftxKBTest( void)
{
#if defined( FLM_UNIX)

	return( ftxUnixKBTest());

#elif defined( FLM_WIN)

	DWORD				lRecordsRead;
	INPUT_RECORD	inputRecord;
	FLMBOOL			bKeyHit = FALSE;

	// VISIT: If a keyboard handler has not been started, need
	// to protect this code with a critical section?

	for( ;;)
	{
		if( PeekConsoleInput( gv_hStdIn, &inputRecord, 1, &lRecordsRead))
		{
			if( !lRecordsRead)
			{
				break;
			}

			if( inputRecord.EventType == KEY_EVENT)
			{
				if( inputRecord.Event.KeyEvent.bKeyDown &&
					(inputRecord.Event.KeyEvent.uChar.AsciiChar ||
					ftxWin32GetExtendedKeycode( &(inputRecord.Event.KeyEvent))))
				{
					bKeyHit = TRUE;
					break;
				}
			}

			if( !ReadConsoleInput( gv_hStdIn, &inputRecord, 1, &lRecordsRead))
			{
				goto Exit;
			}
		}
		else
		{
			break;
		}
	}

Exit:

	return( bKeyHit);
#elif defined( FLM_NLM)

	return( (FLMBOOL)CheckKeyStatus( gv_pFtxInfo->pvScreenHandle));

#else

	return( kbhit());

#endif
}

/****************************************************************************
Desc:    Causes the console to "beep"
Ret:		If the console does not support this feature, FALSE is returned.
****************************************************************************/
FLMBOOL ftxBeep( void)
{
#if defined( FLM_WIN)

	Beep( (DWORD)2000, (DWORD)250);
	return( TRUE);

#else

	return( FALSE);

#endif
}

/****************************************************************************
Desc:    Gets a character from the Win32 console
Ret:		Retrieved character
****************************************************************************/
#if defined( FLM_WIN)
FSTATIC FLMUINT ftxWin32KBGetChar( void)
{
	INPUT_RECORD				ConInpRec;
	DWORD							NumRead;
	ftxWin32CharPair *		pCP;
	int							uiChar = 0;		/* single character buffer */


	/*
	 * check pushback buffer (chbuf) a for character
	 */
	if( chbuf != -1 )
	{
		/*
		 * something there, clear buffer and return the character.
		 */
		uiChar = (unsigned char)(chbuf & 0xFF);
		chbuf = -1;
		return uiChar;
	}

	for( ;;)
	{
		if( gv_pFtxInfo->pbShutdown != NULL)
		{
			if( *(gv_pFtxInfo->pbShutdown) == TRUE)
			{
				return( 0);
			}
		}

		/*
		 * Get a console input event.
		 */
		if( !ReadConsoleInput( gv_hStdIn,
			&ConInpRec, 1L, &NumRead) || (NumRead == 0L))
      {
			uiChar = -1;
			break;
		}

		/*
		 * Look for, and decipher, key events.
		 */
		if ( ConInpRec.EventType == KEY_EVENT)
		{
			if( ConInpRec.Event.KeyEvent.bKeyDown)
			{
				if ( !(ConInpRec.Event.KeyEvent.dwControlKeyState &
					(LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED | SHIFT_PRESSED | CAPSLOCK_ON)))
				{
					if( (uiChar = (FLMUINT)ConInpRec.Event.KeyEvent.uChar.AsciiChar) != 0)
					{
					  break;
					}
				}

				/*
				 * Hard case: either an extended code or an event which should
				 * not be recognized. let _getextendedkeycode() do the work...
				 */

				if( (pCP = ftxWin32GetExtendedKeycode(
					&(ConInpRec.Event.KeyEvent))) != NULL)
				{
					uiChar = pCP->LeadChar;
					if( pCP->SecondChar)
					{
						chbuf = pCP->SecondChar;
					}
					break;
				}
			}
			else
			{
				if( ConInpRec.Event.KeyEvent.uChar.AsciiChar == (unsigned char)0xFF &&
					ConInpRec.Event.KeyEvent.wRepeatCount == 0)
				{
					// Ping
					uiChar = (FLMUINT)0xFFFF;
					break;
				}
			}
		}
	}

	return( uiChar);
}
#endif

#if defined( FLM_WIN)
FSTATIC ftxWin32CharPair * ftxWin32GetExtendedKeycode(
	KEY_EVENT_RECORD *	pKE)
{
	DWORD						CKS;		/* hold dwControlKeyState value */
	ftxWin32CharPair *	pCP;		/* pointer to CharPair containing extended
												code */
	int						iLoop;

	if( (CKS = pKE->dwControlKeyState) & ENHANCED_KEY )
	{
		/*
		 * Find the appropriate entry in EnhancedKeys[]
		 */

		for( pCP = NULL, iLoop = 0; iLoop < FTX_WIN32_NUM_EKA_ELTS; iLoop++)
		{
			if( ftxWin32EnhancedKeys[ iLoop].ScanCode == pKE->wVirtualScanCode)
			{
				if( CKS & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED))
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].AltChars);
				}
				else if( CKS & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].CtrlChars);
				}
				else if( CKS & SHIFT_PRESSED)
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].ShiftChars);
				}
				else
				{
					pCP = &(ftxWin32EnhancedKeys[ iLoop].RegChars);
				}
				break;
			}
		}
	}
	else
	{
		/*
		 * Regular key or a keyboard event which shouldn't be recognized.
		 * Determine which by getting the proper field of the proper
		 * entry in NormalKeys[], and examining the extended code.
		 */

		if( CKS & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED))
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].AltChars);
		}
		else if( CKS & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].CtrlChars);
		}
		else if( CKS & SHIFT_PRESSED)
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].ShiftChars);
		}
      else
		{
			pCP = &(ftxWin32NormalKeys[pKE->wVirtualScanCode].RegChars);
			if( (CKS & CAPSLOCK_ON) && pCP->SecondChar == 0)
			{
				if( pCP->LeadChar >= 'a' && pCP->LeadChar <= 'z')
				{
					pCP->LeadChar = pCP->LeadChar - 'a' + 'A';
				}
			}
		}

		if( pCP->LeadChar == 0 && pCP->SecondChar == 0)
		{
			pCP = NULL;
		}
  }

  return( pCP);
}
#endif


/****************************************************************************
Desc:
****************************************************************************/
RCODE _ftxDefaultDisplayHandler(
	F_Thread *		pThread)
{
#if defined( FLM_WIN)
	FLMUINT		uiRefreshCount = 0;
#endif

	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			break;
		}

		if( gv_pFtxInfo->pbShutdown != NULL)
		{
			if( *(gv_pFtxInfo->pbShutdown) == TRUE)
			{
				break;
			}
		}

#if defined( FLM_WIN)
		if( ++uiRefreshCount > 60)
		{
			uiRefreshCount = 0;

			/*
			Update the cursor to work around a bug in NT where the
			cursor is set to visible when the console is made
			full-screen.
			*/

			FTXRefreshCursor( gv_pFtxInfo);
		}
#endif

		FTXRefresh( gv_pFtxInfo);
		f_sleep( 50); // Refresh 20 times a second
	}

	return( NE_XFLM_OK);
}


/****************************************************************************
Desc:
****************************************************************************/
RCODE _ftxDefaultKeyboardHandler(
	F_Thread *		pThread)
{
	FLMUINT		uiChar;

	FTXEnablePingChar( gv_pFtxInfo);

	for( ;;)
	{
		if( pThread->getShutdownFlag())
		{
			break;
		}

		if( gv_pFtxInfo->pbShutdown != NULL)
		{
			if( *(gv_pFtxInfo->pbShutdown) == TRUE)
			{
				break;
			}
		}

		uiChar = 0;

#if !defined( FLM_NLM) && !defined( FLM_WIN)		// NetWare and Windows will wake up periodically
																// to check for shutdown and therefore do not
																// need to poll the keyboard.
		if( ftxKBTest())
#endif
		{
			uiChar = ftxKBGetChar();
			if( gv_pFtxInfo->pKeyHandler && uiChar != 0xFFFF)
			{
				gv_pFtxInfo->pKeyHandler( gv_pFtxInfo, uiChar,
					&uiChar, gv_pFtxInfo->pvKeyHandlerData);
			}

			switch( uiChar)
			{
				case 0:
				{
					// Ignore the keystroke
					break;
				}

				case WPK_CTRL_A:
				{
					FTXCycleScreensNext( gv_pFtxInfo);
					FTXRefresh( gv_pFtxInfo);
					break;
				}

				case WPK_CTRL_B:
				{
					// Enter the debugger
					flmAssert( 0);
					break;
				}

				case WPK_CTRL_L:
				{
					FTXInvalidate( gv_pFtxInfo);
					FTXRefresh( gv_pFtxInfo);
					break;
				}

				case WPK_CTRL_S:
				{
					FTXCycleScreensPrev( gv_pFtxInfo);
					FTXRefresh( gv_pFtxInfo);
					break;
				}

				case 0xFFFF:
				{
					// Ping
					break;
				}

				default:
				{
					FTXAddKey( gv_pFtxInfo, uiChar);
					break;
				}
			}
		}

#if !defined( FLM_NLM) && !defined( FLM_WIN)
		f_sleep( 0);
#endif
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FTX_INFO * _getGlobalFtxInfo( void)
{
	return( gv_pFtxInfo);
}


/****************************************************************************
Desc:
****************************************************************************/
RCODE _ftxBackgroundThread(
	F_Thread *		pThread)
{
	for( ;;)
	{
		// Ping the keyboard handler to cause it to wake up
		// periodically to check for the shutdown flag
		if( gv_pFtxInfo->bEnablePingChar)
		{
#if defined( FLM_NLM)
			UngetKey( (struct ScreenStruct *)gv_pFtxInfo->pvScreenHandle,
					0xFF, 0, 0, 0);
#elif defined( FLM_WIN)
			{
				INPUT_RECORD		inputRec;
				DWORD					numWritten;

				f_memset( &inputRec, (FLMBYTE)0, sizeof( INPUT_RECORD));
				inputRec.EventType = KEY_EVENT;
				inputRec.Event.KeyEvent.bKeyDown = FALSE;
				inputRec.Event.KeyEvent.wRepeatCount = 0;
				inputRec.Event.KeyEvent.wVirtualKeyCode = 0;
				inputRec.Event.KeyEvent.wVirtualScanCode = 0;
				inputRec.Event.KeyEvent.uChar.AsciiChar = (unsigned char)0xFF;
				inputRec.Event.KeyEvent.dwControlKeyState = 0;

				WriteConsoleInput( gv_hStdIn, &inputRec, 1, &numWritten);
			}
#endif
		}

		if( pThread->getShutdownFlag())
		{
			break;
		}

		f_sleep( 250);
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FTXRCODE FTXDisplayScrollWindow(
	FTX_INFO *		pFtxInfo,
	FTX_SCREEN *	pScreen,
	const char *	pszTitle,
	const char *	pszMessage,
	FLMUINT			uiCols,
	FLMUINT			uiRows)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiScreenCols = 0;
	FLMUINT			uiScreenRows = 0;
	FLMUINT			uiCanvasCols = 0;
	FLMUINT			uiCanvasRows = 0;
	FLMUINT			uiMessageLen = f_strlen( pszMessage);
	FLMUINT			uiNumLines  = 0;
	FLMUINT			uiLineLen = 0;
	FLMUINT			uiLoop = 0;
	char **			ppszRows = NULL;
	FLMUINT			uiCurrentLine = 0;
	const char *	pszBeginLine = NULL;
	FLMUINT			uiLastSpace;
	FTX_WINDOW *	pWin = NULL;
	FLMUINT			uiChar = 0;

	if( RC_BAD( rc = FTXScreenGetSize( pScreen, &uiScreenCols, &uiScreenRows)))
	{
		goto Exit;
	}

	// Make sure the window will fit on the screen
	
	if( ( uiCols > uiScreenCols) || ( uiRows > uiScreenRows))
	{
		rc = RC_SET( FTXRC_ILLEGAL_OP);
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinInit( pScreen, uiCols, uiRows,	&pWin)))
	{
		goto Exit;
	}

	// Center the window
	
	if( RC_BAD( rc = FTXWinMove( pWin, (FLMUINT)((uiScreenCols - uiCols) / 2),
		(FLMUINT)((uiScreenRows - uiRows) / 2))))
	{
		goto Exit;
	}

	if( RC_BAD( rc = FTXWinClear( pWin)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXWinSetFocus( pWin)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXWinDrawBorder( pWin)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXWinSetTitle( pWin, pszTitle, WPS_BLUE, WPS_WHITE)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXWinOpen( pWin)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FTXWinSetScroll( pWin, FALSE)))
	{
		goto Exit;
	}
	
	FTXWinSetCursorType( pWin, WPS_CURSOR_INVISIBLE);
	FTXWinGetCanvasSize( pWin, &uiCanvasCols, &uiCanvasRows);

	uiMessageLen = f_strlen( pszMessage);
	
	// Count the number of lines
	
	for ( uiLoop = 0, uiLastSpace = (FLMUINT)~0;
			uiLoop < (uiMessageLen + 1);
			uiLoop++)
	{
		uiLineLen++;
		
		if( (pszMessage[ uiLoop] == '\n') || 
			(uiLineLen >= uiCanvasCols) ||
			(pszMessage[ uiLoop] == '\0'))
		{
			uiNumLines++;
			if( uiLastSpace != (FLMUINT)~0 && uiLineLen >= uiCanvasCols)
			{
				uiLoop = uiLastSpace;
				uiLastSpace = (FLMUINT)~0;
			}
			uiLineLen = 0;
		}
		else
		{
			if( pszMessage[ uiLoop] <= ' ')
			{
				uiLastSpace = uiLoop;
			}
		}
	}

	if( RC_BAD( rc = f_alloc( sizeof( FLMBYTE *) * uiNumLines, &ppszRows)))
	{
		goto Exit;
	}

	uiLineLen = 0;
	pszBeginLine = pszMessage;
	
	for( uiLoop = 0, uiLastSpace = (FLMUINT)~(0);
		  uiLoop < (uiMessageLen + 1);
		  uiLoop++)
	{
		if ( uiLineLen >= uiCanvasCols ||
			  pszMessage[ uiLoop] == '\n' ||
			  pszMessage[ uiLoop] == '\0')
		{

			if (uiLastSpace != (FLMUINT)~(0) && uiLineLen >= uiCanvasCols)
			{
				uiLineLen = (FLMUINT)(&pszMessage[ uiLastSpace] - pszBeginLine + 1);
				uiLoop = uiLastSpace;
				uiLastSpace = (FLMUINT)~(0);
			}

			if( RC_BAD( rc = f_alloc( uiLineLen + 1, &ppszRows[ uiCurrentLine])))
			{
				goto Exit;
			}
			f_memcpy( ppszRows[ uiCurrentLine], pszBeginLine, uiLineLen);
			ppszRows[ uiCurrentLine++][uiLineLen] = '\0';
			pszBeginLine += uiLineLen;
			uiLineLen = 0;
			continue;
		}
		else
		{
			if (uiLastSpace != (FLMUINT)~0)
			{
				flmAssert( (FLMUINT)(&pszMessage[ uiLastSpace] - pszBeginLine) <= uiCanvasCols);
			}
			if (pszMessage[ uiLoop] <= ' ')
			{
				uiLastSpace = uiLoop;
			}
		}
		uiLineLen++;
	}

	uiCurrentLine = 0;

	for ( uiLoop = 0; uiLoop < uiNumLines && uiLoop < uiRows - 2; uiLoop++)
	{
		if( RC_BAD( rc = FTXWinSetCursorPos( pWin, 0, uiLoop)))
		{
			goto Exit;
		}
		
		FTXWinPrintStr( pWin, ppszRows[ uiLoop]);
	}
	
	if( RC_BAD( rc = FTXRefresh( pFtxInfo)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if ( FTXWinTestKB( pWin))
		{
			FTXWinInputChar( pWin, &uiChar);
			switch( uiChar)
			{
				case WPK_DOWN:
				{
					if( uiCurrentLine < ( uiNumLines - 1))
					{
						uiCurrentLine++;
						if( RC_BAD( rc = FTXWinClear( pWin)))
						{
							goto Exit;
						}
						
						for ( uiLoop = 0;
								uiLoop < (uiNumLines - uiCurrentLine) &&
										uiLoop < uiRows - 2;
								uiLoop++)
						{
							if( RC_BAD( rc = FTXWinSetCursorPos( pWin, 0, uiLoop)))
							{
								goto Exit;
							}
							
							if( RC_BAD( rc = FTXWinClearToEOL( pWin)))
							{
								goto Exit;
							}
							
							FTXWinPrintStr( pWin, ppszRows[ uiLoop + uiCurrentLine]);
						}
						
						if( RC_BAD( rc = FTXRefresh( pFtxInfo)))
						{
							goto Exit;
						}
					}
					
					break;
				}
				
				case WPK_UP:
				{
					if ( uiCurrentLine > 0)
					{
						uiCurrentLine--;
						if( RC_BAD( rc = FTXWinClear( pWin)))
						{
							goto Exit;
						}
						
						for ( uiLoop = 0;
								uiLoop < (uiNumLines - uiCurrentLine) &&
										uiLoop < uiRows - 2;
								uiLoop++)
						{
							if( RC_BAD( rc = FTXWinSetCursorPos( pWin, 0, uiLoop)))
							{
								goto Exit;
							}
							
							if( RC_BAD( rc = FTXWinClearToEOL( pWin)))
							{
								goto Exit;
							}
							
							FTXWinPrintStr( pWin, ppszRows[ uiLoop + uiCurrentLine]);
						}
						
						if( RC_BAD( rc = FTXRefresh( pFtxInfo)))
						{
							goto Exit;
						}
					}
					
					break;
				}
				
				case WPK_ENTER:
				case WPK_ESC:
				{
					goto Exit;
				}
			}
		}
		
		f_yieldCPU();
	}

Exit:

	if( pWin)
	{
		FTXWinFree( &pWin);
		FTXRefresh( pFtxInfo);
	}
	
	if( ppszRows)
	{
		for( uiLoop = 0; uiLoop < uiNumLines; uiLoop++)
		{
			f_free( &ppszRows[ uiLoop]);
		}
		
		f_free( &ppszRows);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayInit( void)
{
	initscr();
	noecho();
	cbreak();
	halfdelay(4);
	meta(stdscr, TRUE);
	keypad(stdscr, TRUE);
	scrollok(stdscr, FALSE);
	move(0, 0);
	refresh();
	
	ungetChar = (chtype)ERR;
	
	if( has_colors())
	{
		start_color();

		flm2curses[ WPS_BLACK] = COLOR_BLACK;
		flm2curses[ WPS_BLUE] = COLOR_BLUE;
		flm2curses[ WPS_GREEN] = COLOR_GREEN;
		flm2curses[ WPS_CYAN] = COLOR_CYAN;
		flm2curses[ WPS_RED] = COLOR_RED;
		flm2curses[ WPS_MAGENTA] = COLOR_MAGENTA;
		flm2curses[ WPS_BROWN] = COLOR_YELLOW;
		flm2curses[ WPS_LIGHTGRAY] = COLOR_WHITE;

		int defaultbg = A_NORMAL | ' ';
		bkgd(defaultbg);
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayFree( void)
{
	endwin();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayGetSize(
	FLMUINT *		puiNumColsRV,
	FLMUINT * 		puiNumRowsRV)
{
	*puiNumColsRV = (FLMUINT)COLS;
	*puiNumRowsRV = (FLMUINT)LINES;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
static int flm_to_curses_attr(
	int				attr)
{
	int 	fg;
	int	bg;
	int	curses_attr = 0;

	fg = attr & 0x0f;
	bg = (attr >> 4) & 0x07;

	curses_attr = (fg > WPS_LIGHTGRAY) ? A_BOLD : 0;
	fg &= 0x07;
	
	if (has_colors())
	{
		if (color_pairs[bg][fg] == 0)
		{
			if (last_pair >= COLOR_PAIRS)
			{
				color_pairs[bg][fg] = 1;
			}
			else
			{
				color_pairs[bg][fg] = ++last_pair;
				init_pair(last_pair, flm2curses[fg], flm2curses[bg]);
			}
		}
		
		curses_attr |= COLOR_PAIR(color_pairs[bg][fg]);
	}
	else
	{
		curses_attr |= (bg != WPS_BLUE) ? A_REVERSE : 0;
	}

	return( curses_attr);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayChar(
	FLMUINT			uiChar,
	FLMUINT			uiAttr)
{
	addch( (chtype)uiChar | flm_to_curses_attr(uiAttr));
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayRefresh( void)
{
	refresh();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplayReset( void)
{
	clearok( stdscr, TRUE);
	refresh();
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC void ftxUnixDisplaySetCursorPos(
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	move( uiRow, uiCol);
	refresh();
}
#endif

/****************************************************************************
Desc:		Simulate Ctrl + Shift + character keys
****************************************************************************/
#ifdef FLM_UNIX
static FLMUINT ftxUnixSimulatedKey(
	FLMUINT			c)
{
	// We simulate Insert, Delete, Home, End, PgUp and PgDn. We can
	// also simulate the CTRL- combinations of these if needed

	chtype ch = (chtype) c + 'a' - 1;
	
	switch( ch)
	{
		case 'i':
		{
			c = WPK_INSERT;
			break;
		}
	
		case 'd':
		{
			c = WPK_DELETE;
			break;
		}
	
		case 'b':
		{
			c = WPK_PGUP;
			break;
		}
	
		case 'f':
		{
			c = WPK_PGDN;
			break;
		}
	
		case 'h':
		{
			c = WPK_HOME;
			break;
		}
	
		case 'e':
		{
			c = WPK_END;
			break;
		}
	
		default:
		{
			c = (FLMUINT)ERR;
			break;
		}
	}
	
	return( c);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
static FLMUINT ftxUnixHandleEsc( void)
{
	// On unix ESC is the prefix for many function keys. It's a bad
	// idea to use ESC as a character by itself because it can result
	// in a delay of as much as a second.  If we don't handle all
	// escape sequences, the first escape character can cause FLAIM to
	// exit!  So, we discard unrecognized escape sequences here.

	int c = WPK_ESCAPE;
	int c2;

	if( (c2 = getch()) == ERR)
	{
		goto Exit;
	}

	switch( c2)
	{
		case '1':
		{
			c = WPK_F1;
			break;
		}

		case '2':
		{
			c = WPK_F2;
			break;
		}

		case '3':
		{
			c = WPK_F3;
			break;
		}

		case '4':
		{
			c = WPK_F4;
			break;
		}

		case '5':
		{
			c = WPK_F5;
			break;
		}

		case '6':
		{
			c = WPK_F6;
			break;
		}

		case '7':
		{
			c = WPK_F7;
			break;
		}

		case '8':
		{
			c = WPK_F8;
			break;
		}

		case '9':
		{
			c = WPK_F9;
			break;
		}

		case '0':
		{
			c = WPK_F10;
			break;
		}

		case 'i':
		{
			c = WPK_INSERT;
			break;
		}

		case 'd':
		{
			c = WPK_DELETE;
			break;
		}

		case KEY_F( 0):
		{
			c = WPK_ALT_F10;
			break;
		}

		case '\t':
		{
			c = WPK_STAB;
			break;
		}

		case KEY_FIND:
		case KEY_HOME:
		{
			c = WPK_CTRL_HOME;
			break;
		}

		case KEY_END:
		case KEY_SELECT:
		case KEY_LL:
		{
			c = WPK_CTRL_END;
			break;
		}

		case KEY_LEFT:
		{
			c = WPK_CTRL_LEFT;
			break;
		}

		case KEY_RIGHT:
		{
			c = WPK_CTRL_RIGHT;
			break;
		}

		case KEY_DOWN:
		{
			c = WPK_CTRL_DOWN;
			break;
		}

		case KEY_UP:
		{
			c = WPK_CTRL_UP;
			break;
		}

		case 0x000A:
		case 0x000D:
		case KEY_ENTER:
		{
			c = WPK_CTRL_ENTER;
			break;
		}

		case KEY_NPAGE:
		{
			c = WPK_CTRL_PGDN;
			break;
		}

		case KEY_PPAGE:
		{
			c = WPK_CTRL_PGUP;
			break;
		}

		case KEY_IC:
		{
			c = WPK_CTRL_INSERT;
			break;
		}

		default:
		{
			if( c2 >= '0' && c2 <= '9')
			{
				c = WPK_ALT_0 + c2 - '0';
			}
			else if( c2 >= 'a' && c2 <= 'z')
			{
				c = WPK_ALT_A + c2 - 'a';
			}
			else if( (c2 >= 1) && (c <= 032))
			{
				c = ftxUnixSimulatedKey(c2);
			}
			else if( c2 >= KEY_F(1) && c2 <= KEY_F( 10))
			{
				c = WPK_ALT_F1 + c - KEY_F(1);
			}
			else if( c2 == erasechar() || c2 == '' || c2 == 0127)
			{
				c = WPK_ESCAPE;
			}
			else if( c2 == 033)
			{
				c = WPK_ESCAPE;
				break;
			}
			else
			{
				c = ERR;
				while( getch() != ERR);
			}
			
			break;
		}
	}
	
Exit:

	return( c);
}
#endif

/****************************************************************************
Desc:
Notes:	On Unix some terminal types (notably Solaris xterm) do not generate
			proper key codes for Insert, Home, PgUp, PgDn etc. Use a different
			terminal emulator (rxvt for eg). They can also be simulated by the
			key combination Meta-Shift-I, Meta-Shift-U, Meta-Shift-D etc.
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC FLMUINT ftxUnixKBGetChar( void)
{
	int 	c;

Again:

	if( ungetChar != ERR)
	{
		c = ungetChar;
		ungetChar = ERR;
	}
	else
	{
		while( (c = getch()) == ERR);
	}

	if (c == killchar())
	{
		c = WPK_DELETE;
	}
	else if (c == erasechar())
	{
		c = WPK_BACKSPACE;
	}
	else if (c == '\t')
	{
		c = WPK_TAB;
	}
	else if( c >= 1 && c <= 032 && c != 10 && c != 13)
	{
		c = WPK_CTRL_A + (c - 1);
	}
	else if ((c >= (128 + '0')) && (c <= (128 + '9')))
	{
		c = WPK_ALT_0 + (c - 128 - '0');
	}
	else if ((c >= (128 + 'a')) && (c <= (128 + 'z')))
	{
		c = WPK_ALT_A + (c - 128 - 'a');
	}
	else if ((c >= 128) && (c <= (128 + 032)))
	{
		c = ftxUnixSimulatedKey(c - 128);
	}
	else if (c >= KEY_F(1) && c <= KEY_F(10))
	{
		c = WPK_F1 + c - KEY_F(1);
	}
	else if (c >= KEY_F(11) && c <= KEY_F(20))
	{
		c = WPK_SF1 + c - KEY_F(11);
	}
	else
	{
		switch( c)
		{
			case KEY_F( 0):
			{
				c = WPK_ALT_F10;
				break;
			}
	
			case KEY_BACKSPACE:
			{
				c = WPK_BACKSPACE;
				break;
			}
	
			case 033:
			{
				c = ftxUnixHandleEsc();
				break;
			}
			
			case 0127:
			case KEY_DC:
			{
				c = WPK_DELETE;
				break;
			}
	
			case KEY_FIND:
			case KEY_HOME:
			{
				c = WPK_HOME;
				break;
			}
	
			case KEY_END:
			case KEY_SELECT:
			case KEY_LL:
			{
				c = WPK_END;
				break;
			}
	
			case KEY_LEFT:
			{
				c = WPK_LEFT;
				break;
			}
	
			case KEY_RIGHT:
			{
				c = WPK_RIGHT;
				break;
			}
	
			case KEY_DOWN:
			{
				c = WPK_DOWN;
				break;
			}
	
			case KEY_UP:
			{
				c = WPK_UP;
				break;
			}
	
			case 0x000A:
			case 0x000D:
			case KEY_ENTER:
			{
				c = WPK_ENTER;
				break;
			}
	
			case KEY_NPAGE:
			{
				c = WPK_PGDN;
				break;
			}
	
			case KEY_PPAGE:
			{
				c = WPK_PGUP;
				break;
			}
	
			case KEY_IC:
			{
				c = WPK_INSERT;
				break;
			}
		}
	}

	if( c == ERR)
	{
		goto Again;
	}
	
	return( (FLMUINT)c);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_UNIX
FSTATIC FLMBOOL ftxUnixKBTest( void)
{
	int 	c;

	if( ungetChar != ERR)
	{
		c = ungetChar;
	}
	else
	{
		if( (c = getch()) != ERR)
		{
			ungetChar = c;
		}
	}
	
	return( (c == ERR) ? FALSE : TRUE);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void wpsLock(
	F_MUTEX	*	phMutex)
{
	f_mutexLock( *phMutex);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void wpsUnlock(
	F_MUTEX	*	phMutex)
{
	f_mutexUnlock( *phMutex);
}

/****************************************************************************
Desc: Initialize and set the title
****************************************************************************/
void WpsInit(
	FLMUINT			uiRows,				// 0xFFFF means use current screen height.
	FLMUINT			uiCols,				// 0xFFFF means use current screen width.
	const char *	pszScreenTitle)
{
	char	szTitleAndVer[ 100];

	if( gv_bInitialized)
	{
		return;
	}

	FTXInit( pszScreenTitle, uiCols, uiRows, WPS_BLACK, WPS_LIGHTGRAY,
		NULL, NULL, &gv_pFtxInfo);

	if( RC_BAD( f_mutexCreate( &gv_hDispMutex)))
	{
		flmAssert( 0);
	}

	WpsThrdInit( szTitleAndVer);
	gv_bPrivateFTX = TRUE;
	gv_bInitialized = TRUE;
}


/****************************************************************************
Desc: Initialize WPS using an existing FTX environment
****************************************************************************/
void WpsInitFTX(
	FTX_INFO *	pFtxInfo
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
}


/****************************************************************************
Desc: Restores the screen to an initial state
****************************************************************************/
void WpsExit( void)
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
}

/****************************************************************************
Desc: Initialize and set the title of a thread's screen
***************************************************************************/
void WpsThrdInitUsingScreen(
	FTX_SCREEN *	pFtxScreen,
	char *			pszScreenTitle
	)
{
	FLMUINT			uiRows = 0;
	FLMUINT			uiThrdId;
	WPSSCREEN *		pCurScreen = NULL;

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
			if( FTXScreenInit( gv_pFtxInfo, pszScreenTitle,
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
			pszScreenTitle) != FTXRC_SUCCESS)
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
}


/****************************************************************************
Desc: Frees all screen resources allocated to a thread
Ret:
****************************************************************************/
void WpsThrdExit( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN *		pPrevScreen = NULL;
	WPSSCREEN *		pCurScreen = NULL;


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
}

/****************************************************************************
Desc: Returns the size of the screen in columns and rows.
****************************************************************************/
void WpsScrSize(
	FLMUINT *	puiNumColsRV,
	FLMUINT *	puiNumRowsRV)
{
	FTXWinGetCanvasSize( wpsGetThrdWin(), puiNumColsRV, puiNumRowsRV);
}

/****************************************************************************
Desc:	Output a string at present cursor location.
****************************************************************************/
void WpsStrOut(
	const char *	pszString)
{
	FTXWinPrintStr( wpsGetThrdWin(), pszString);
	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:	Output a formatted string at present cursor location.
****************************************************************************/
void WpsPrintf(
	const char *	pszFormat, ...)
{
	char			szBuffer[ 512];
	f_va_list	args;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);
	FTXWinPrintStr( wpsGetThrdWin(), szBuffer);

	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:	Output a formatted string at present cursor location with color
****************************************************************************/
void WpsCPrintf(
	FLMUINT			uiBack,
	FLMUINT			uiFore,
	const char *	pszFormat, ...)
{
	char			szBuffer[ 512];
	f_va_list	args;
	FLMUINT		uiOldBack;
	FLMUINT		uiOldFore;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	FTXWinGetBackFore( wpsGetThrdWin(), &uiOldBack, &uiOldFore);
	FTXWinSetBackFore( wpsGetThrdWin(), uiBack, uiFore);
	FTXWinPrintStr( wpsGetThrdWin(), szBuffer);
	FTXWinSetBackFore( wpsGetThrdWin(), uiOldBack, uiOldFore);

	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}


/****************************************************************************
Desc: 	Output a character to the screen at the current location. If char is
			a LineFeed then a CarriageReturn will be inserted before the LineFeed.
Notes:	On NetWare becomes a blocking function if the char 
			is the newline character.
****************************************************************************/
void WpsChrOut(
	char			chr)
{
	FTXWinPrintChar( wpsGetThrdWin(), (FLMUINT)chr);
	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:    Clear the screen from the col/row down
Notes:   If col==row==0 then clear entire screen
****************************************************************************/
void WpsScrClr(
	FLMUINT			uiCol,
	FLMUINT			uiRow)
{
	FTX_WINDOW *	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinClearXY( pThrdWin, uiCol, uiRow);
	FTXWinSetCursorPos( pThrdWin, uiCol, uiRow);
	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:    Position to the column and row specified.
Notes:   The NLM could call GetPositionOfOutputCursor(&r,&c);
****************************************************************************/
void WpsScrPos(
	FLMUINT	uiCol,
	FLMUINT	uiRow
	)
{
	FTX_WINDOW *	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;


	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinSetCursorPos( pThrdWin, uiCol, uiRow);
}

/****************************************************************************
Desc:    Clear from input cursor to end of line
****************************************************************************/
void WpsLineClr(
	FLMUINT	uiCol,
	FLMUINT	uiRow
	)
{
	FTX_WINDOW *	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;

	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinClearLine( pThrdWin, uiCol, uiRow);
	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:    Edit a line of data like gets(s).  Newline replaced by NULL character.
Notes:   Does not support WP extended character input - but could easily!
****************************************************************************/
FLMUINT WpsLineEd(
	char *		pszString,
	FLMUINT		uiMaxLen,
	FLMBOOL *	pbShutdown)
{
	FLMUINT		uiCharCount;
	FLMUINT		uiCursorType;

	uiCursorType = FTXWinGetCursorType( wpsGetThrdWin());
	FTXWinSetCursorType( wpsGetThrdWin(), WPS_CURSOR_UNDERLINE);
	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
	uiCharCount = FTXLineEd( wpsGetThrdWin(), pszString, uiMaxLen);
	FTXSetShutdownFlag( gv_pFtxInfo, NULL);
	FTXWinSetCursorType( wpsGetThrdWin(), uiCursorType);

	return( uiCharCount);
}


/****************************************************************************
Desc:    Sets the FTX shutdown flag pointer
Ret:
****************************************************************************/
void WpsSetShutdown(
	FLMBOOL *    pbShutdown)
{
	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
}

/****************************************************************************
Desc:    Edit a line of data with advanced features.
Ret:     Number of characters input.
****************************************************************************/
FLMUINT WpsLineEditExt(
	char *		pszBuffer,
	FLMUINT		uiBufSize,
	FLMUINT		uiMaxWidth,
	FLMBOOL *	pbShutdown,
	FLMUINT *	puiTermChar)
{
	FLMUINT		uiCharCount = 0;
	FLMUINT		uiCursorType;


	uiCursorType = FTXWinGetCursorType( wpsGetThrdWin());
	FTXWinSetCursorType( wpsGetThrdWin(), WPS_CURSOR_UNDERLINE);
	FTXSetShutdownFlag( gv_pFtxInfo, pbShutdown);
	FTXLineEdit( wpsGetThrdWin(), pszBuffer, uiBufSize, uiMaxWidth,
		&uiCharCount, puiTermChar);
	FTXSetShutdownFlag( gv_pFtxInfo, NULL);
	FTXWinSetCursorType( wpsGetThrdWin(), uiCursorType);

	return( (FLMINT)uiCharCount);
}

/****************************************************************************
Desc:	Get the current X coordinate of the cursor
****************************************************************************/
FLMUINT WpsCurrCol( void)
{
	FLMUINT		uiCol;

	FTXWinGetCursorPos( wpsGetThrdWin(), &uiCol, NULL);
	return( uiCol);
}


/****************************************************************************
Desc:	Get the current Y coordinate of the cursor
****************************************************************************/
FLMUINT WpsCurrRow( void)
{
	FLMUINT		uiRow;

	FTXWinGetCursorPos( wpsGetThrdWin(), NULL, &uiRow);
	return( uiRow);
}

/****************************************************************************
Desc:    Set the background and foreground colors
Ret:     None
****************************************************************************/
void WpsScrBackFor(
	FLMUINT	uiBackColor,
	FLMUINT	uiForeColor
	)
{
	FTXWinSetBackFore( wpsGetThrdWin(), uiBackColor, uiForeColor);
}

/****************************************************************************
Desc : Sets the cursor attributes.
****************************************************************************/
void WpsCursorSetType(
	FLMUINT		uiType
	)
{
	FTXWinSetCursorType( wpsGetThrdWin(), uiType);
	FTXRefresh( gv_pFtxInfo);
}

/****************************************************************************
Desc:    Specifies that display performance (throughput) should be
			optimal.
****************************************************************************/
void WpsOptimize( void)
{
	gv_bOptimize = TRUE;
}

/****************************************************************************
Desc: Draws a border around the current thread's screen
Ret:  none
****************************************************************************/
void WpsDrawBorder( void)
{
	FTXWinDrawBorder( wpsGetThrdWin());
	if( !gv_bOptimize)
	{
		FTXRefresh( gv_pFtxInfo);
	}
}

/****************************************************************************
Desc:    Convert keyboard sequences/scan codes to WPK key strokes.
Notes:   Does not support WP extended character input - but could easily!
****************************************************************************/
FLMUINT WpkIncar( void)
{
	FLMUINT		uiChar;

	FTXWinInputChar( wpsGetThrdWin(), &uiChar);
	return( uiChar);
}

/****************************************************************************
Desc:    Convert keyboard sequences/scan codes to WPK key strokes.  This
			routine accepts a pointer to a shutdown flag.
****************************************************************************/
FLMUINT WpkGetChar(
	FLMBOOL *		pbShutdown)
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
FLMUINT WpkTestKB( void)
{
	FLMUINT		uiCharAvail;

	uiCharAvail = (FLMUINT)(FTXWinTestKB( wpsGetThrdWin()) ==
		FTXRC_SUCCESS ? 1 : 0);
	return( uiCharAvail);
}

/****************************************************************************
Desc:		Returns a pointer to a thread's screen
****************************************************************************/
FTX_SCREEN * WpsGetThrdScreen( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN *		pCurScreen = NULL;

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
Desc:		Returns a pointer to a thread's screen
****************************************************************************/
FSTATIC FTX_WINDOW * wpsGetThrdWin( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN *		pCurScreen = NULL;

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
