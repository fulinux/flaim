//-------------------------------------------------------------------------
// Desc:	Unix text user interface APIs - windowing - definitions.
// Tabs:	3
//
//		Copyright (c) 1997,1999-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftxunix.h 12235 2006-01-19 14:23:41 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef FTXUNIX_H
#define FTXUNIX_H

void ftxUnixDisplayInit( void);

void ftxUnixDisplayFree( void);

void ftxUnixDisplayGetSize(
		FLMUINT *	puiNumColsRV,
		FLMUINT *	puiNumRowsRV);
		
void ftxUnixDisplayChar(
		FLMUINT 		uiChar,
		FLMUINT		uiAttr);


void ftxUnixDisplayRefresh( void);


void ftxUnixDisplayReset( void);


void ftxUnixDisplaySetCursorPos( 
		FLMUINT		uiCol,
		FLMUINT		uiRow);


FLMUINT ftxUnixKBGetChar( void);

FLMBOOL ftxUnixKBTest( void);

#endif
