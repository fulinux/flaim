//------------------------------------------------------------------------------
// Desc:	This file contains display routines for the VIEW program.
//
// Tabs:	3
//
//		Copyright (c) 1992-1995, 1998-2000, 2002-2003,2005-2006 Novell, Inc.
//		All Rights Reserved.
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
// $Id: viewdisp.cpp 3119 2006-01-19 13:39:12 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "view.h"

/***************************************************************************
Desc:	This routine displays an error message to the screen.
*****************************************************************************/
void ViewShowError(
	const char *	pszMessage)
{
	FLMUINT	uiChar;
	FLMUINT	uiNumCols;
	FLMUINT	uiNumRows;

	f_conGetScreenSize( &uiNumCols, &uiNumRows);
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	f_conSetBackFore( FLM_RED, FLM_WHITE);
	f_conStrOutXY( pszMessage, 0, uiNumRows - 2);
	f_conStrOutXY( "Press ENTER to continue: ", 0, uiNumRows - 1);
	for (;;)
	{
		uiChar = f_conGetKey();
		if (uiChar == FKB_ENTER || uiChar == FKB_ESCAPE)
		{
			break;
		}
	}
	f_conSetBackFore( FLM_BLACK, FLM_WHITE);
	f_conClearScreen( 0, uiNumRows - 2);
	ViewEscPrompt();
}

/***************************************************************************
Desc:	This routine displays a FLAIM error message to the screen.  It
		formats the RCODE into a message and calls ViewShowError
		to display the error.
*****************************************************************************/
void ViewShowRCError(
	const char *	pszWhat,
	RCODE				rc)
{
	char	szTBuf [100];

	f_sprintf( szTBuf, "Error %s: 0x%04X", pszWhat, (unsigned)rc); 
	ViewShowError( szTBuf);
}
