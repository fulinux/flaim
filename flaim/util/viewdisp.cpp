//-------------------------------------------------------------------------
// Desc:	Display routines for the database viewer utility.
// Tabs:	3
//
//		Copyright (c) 1992-1995,1998-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: viewdisp.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "view.h"

/***************************************************************************
Name:    ViewShowError
Desc:    This routine displays an error message to the screen.
*****************************************************************************/
void ViewShowError(
	const char *	Message)
{
	FLMUINT		uiChar;
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	WpsScrSize( &uiNumCols, &uiNumRows);
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, uiNumRows - 2);
	WpsScrBackFor( WPS_RED, WPS_WHITE);
	WpsStrOutXY( Message, 0, uiNumRows - 2);
	WpsStrOutXY( "Press ENTER to continue: ", 0, 23);
	for( ;;)
	{
		uiChar = (FLMUINT)WpkIncar();
		if( (uiChar == WPK_ENTER) || (uiChar == WPK_ESCAPE))
			break;
	}
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, uiNumRows - 2);
	ViewEscPrompt();
}

/***************************************************************************
Name:    ViewShowRCError
Desc:    This routine displays a FLAIM error message to the screen.  It
				formats the RCODE into a message and calls ViewShowError
				to display the error.
*****************************************************************************/
void ViewShowRCError(
	const char *	szWhat,
	RCODE       	rc)
{
	char	TBuf[ 100];

	f_strcpy( TBuf, "Error ");
	f_strcpy( &TBuf [f_strlen( TBuf)], szWhat);
	f_strcpy( &TBuf [f_strlen( TBuf)], ": ");
	f_strcpy( &TBuf [f_strlen( TBuf)], FlmErrorString( rc));
	ViewShowError( TBuf);
}
