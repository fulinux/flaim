//-------------------------------------------------------------------------
// Desc:	Screen display routines for all platforms - definitions.
// Tabs:	3
//
//		Copyright (c) 1992-2006 Novell, Inc. All Rights Reserved.
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
// $Id: wpscreen.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef  WPSCREEN_H
#define  WPSCREEN_H

#include "ftx.h"

typedef struct wps_screen
{
	FLMBOOL					bPrivate;
	FLMUINT					uiScreenId;
	FTX_SCREEN *			pScreen;
	FTX_WINDOW *			pTitleWin;
	FTX_WINDOW *			pWin;
	struct wps_screen *	pNext;
	void *					hThis;
} WPSSCREEN, * WPSSCREEN_p;

#ifdef __cplusplus
	extern "C" {
#endif

void WpsInit(
	FLMUINT			rows,
	FLMUINT			cols,
	const char *	title);

void WpsInitFTX(
	FTX_INFO *		pFtxInfo);

void WpsExit( void);

void WpsThrdInitUsingScreen(
	FTX_SCREEN *	pScreen,
	const char *	screenTitle);

#define WpsThrdInit(a) \
	WpsThrdInitUsingScreen( NULL, (a))

void WpsThrdExit( void);

void WpsWPOut(
	FLMINT			WPChr);

FLMINT WpsStrOut(
	const char *	string);

FLMINT WpsPrintf(
	const char * 	pucFormat, ...);

FLMINT WpsCPrintf(
	FLMUINT			uiBack,
	FLMUINT			uiFore,
	const char *	pucFormat, ...);

void WpsOptimize( void);

void WpsScrReset( void);

#define WpsScrReset() \
	(WpsScrClr(0,0))

void WpsScrClr(
	FLMUINT			col,
	FLMUINT			row);

void WpsScrPos(
	FLMUINT			col,
	FLMUINT			row);

void WpsScrBackFor(
	FLMUINT		  background,
	FLMUINT		  forground);

void WpsLineClr(
	FLMUINT			col,
	FLMUINT			row);

FLMUINT WpsLineEd(
	char *   		string,
	FLMUINT     	maxLen,
	FLMBOOL *		pbShutdown);

FLMUINT WpsLineEditExt(
	char *			pbyBuffer,
	FLMUINT			wBufSize,
	FLMUINT			wMaxWidth,
	FLMBOOL *		pbShutdown,
	FLMUINT *		pwTermChar);

FLMINT WpsStrOutXY(
	const char *   string,
	FLMUINT			col,
	FLMUINT			row);

#define WpsStrOutXY( string, col, row)    (WpsScrPos( col, row), WpsStrOut( string))

void WpsDrawBorder( void);

FLMBOOL WpsCursorSetType(
	FLMUINT			uiType);

void WpsCurOff( void);

void WpsCurOn( void);

void WpsScrSize(
	FLMUINT *		puiNumColsRV,
	FLMUINT *		puiNumRowsRV);

FLMUINT WpsCurrRow( void);

FLMUINT WpsCurrCol( void);

FLMUINT WpkIncar( void);

FLMUINT WpkGetChar(
	FLMBOOL *		pbShutdown);

FLMUINT WpkTestKB( void);

FTX_SCREEN * WpsGetThrdScreen( void);

void WpsSetShutdown(
	FLMBOOL *    	pbShutdown);

#ifdef __cplusplus
}
#endif

#endif
