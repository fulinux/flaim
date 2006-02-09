//-------------------------------------------------------------------------
// Desc: Menuing system for database viewer utility.
// Tabs: 3
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
// $Id: viewmenu.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "view.h"

FSTATIC char * Months[] = {
	"Jan",
	"Feb",
	"Mar",
	"Apr",
	"May",
	"Jun",
	"Jul",
	"Aug",
	"Sep",
	"Oct",
	"Nov",
	"Dec"
};

FSTATIC void ViewDispMenuItem(
	VIEW_MENU_ITEM_p	ViewMenuPtr
	);

FSTATIC void ViewRefreshMenu(
	VIEW_MENU_ITEM_p	PrevItem
	);

FSTATIC void UpdateHorizCursor(
	FLMUINT	OnFlag
	);

FSTATIC void DoUpArrow(
	void
	);

FSTATIC void DoDownArrow(
	void
	);

FSTATIC void DoPageDown(
	void
	);

FSTATIC void DoPageUp(
	void
	);

FSTATIC void DoHome(
	void
	);

FSTATIC void DoEnd(
	void
	);

FSTATIC void DoRightArrow(
	void
	);

FSTATIC void DoLeftArrow(
	void
	);

FSTATIC void ByteToHex(
	FLMBYTE		 c,
	FLMBYTE *	 DestBuf,
	FLMUINT		 UpperCaseFlag
	);

FSTATIC void ViewHelpScreen( void);

extern FLMUINT gv_uiTopLine;
extern FLMUINT gv_uiBottomLine;


/***************************************************************************
Name:		ViewFreeMenuMemory
Desc:		This routine frees the memory used to hold menu items.
*****************************************************************************/
void ViewFreeMenuMemory(
	void
	)
{
	if (gv_pViewMenuFirstItem != NULL)
	{
		GedPoolFree( &gv_ViewPool);
		gv_pViewMenuFirstItem =
		gv_pViewMenuLastItem =
		gv_pViewMenuCurrItem = NULL;
	}
}

/***************************************************************************
Name:		ViewMenuInit
Desc:		This routine initializes variables to start a new menu.
*****************************************************************************/
FLMINT ViewMenuInit(
	const char *		Title)
{
	FLMUINT		Col;

	/* Clear the screen and display the menu title */

	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, 0);

	/* Display the title in the middle of the top line of the screen */

	Col = (80 - f_strlen( Title)) / 2;

	WpsStrOutXY( Title, Col, 0);
	ViewUpdateDate( TRUE, &gv_ViewLastTime);

	/* Deallocate any old memory, if any */

	ViewFreeMenuMemory();

	return( 1);
}

/***************************************************************************
Name:		ByteToHex
Desc:		This routine converts a FLMBYTE value to two ASCII Hex characters.
*****************************************************************************/
FSTATIC void ByteToHex(
	FLMBYTE		c,
	FLMBYTE *	DestBuf,
	FLMUINT		UpperCaseFlag
	)
{
	FLMBYTE		TempC;

	/* Convert the upper four bits */

	if ((TempC = ((c & 0xF0) >> 4)) <= 9)
		*DestBuf = '0' + TempC;
	else if (UpperCaseFlag)
		*DestBuf = 'A' + TempC - 10;
	else
		*DestBuf = 'a' + TempC - 10;
	DestBuf++;

	/* Convert the lower four bits */

	if ((TempC = (c & 0x0F)) <= 9)
		*DestBuf = '0' + TempC;
	else if (UpperCaseFlag)
		*DestBuf = 'A' + TempC - 10;
	else
		*DestBuf = 'a' + TempC - 10;
	DestBuf++;

	/* Terminate with a NULL character */

	*DestBuf = 0;
}

/***************************************************************************
Name:		ViewDispMenuItem
Desc:		This routine displays a menu item on the screen.
*****************************************************************************/
FSTATIC void ViewDispMenuItem(
	VIEW_MENU_ITEM_p	ViewMenuPtr
	)
{
	FLMUINT		Row;
	FLMUINT		Col = ViewMenuPtr->Col;
	FLMUINT		uiLoop;
	char			TempBuf[ 80];

	if (!gv_bViewEnabled)
		return;

	/* Calculate row and column where the item is to be displayed */

	Row = gv_uiTopLine + (ViewMenuPtr->Row - gv_uiViewTopRow);

	/* If it is a HEX display, output the address first */

	if ((ViewMenuPtr->ValueType & 0x0F) == VAL_IS_BINARY_HEX)
	{
		f_sprintf( (char *)TempBuf, "%03u:%08X", 
				(unsigned)ViewMenuPtr->ModFileNumber, (unsigned)ViewMenuPtr->ModFileOffset);
		WpsScrBackFor( WPS_WHITE, WPS_GREEN);
		WpsStrOutXY( TempBuf, 0, Row);
	}

	/* If the item is the current item, display it using the */
	/* select colors.	 Otherwise, display using the unselect colors */

	if (ViewMenuPtr->ItemNum == gv_uiViewMenuCurrItemNum)
	{
		if (gv_pViewMenuCurrItem == NULL)
			gv_pViewMenuCurrItem = ViewMenuPtr;
		if (ViewMenuPtr->Option)
			WpsScrBackFor( ViewMenuPtr->SelectBackColor,
											 ViewMenuPtr->SelectForeColor);
		else
			WpsScrBackFor( ViewMenuPtr->UnselectForeColor,
											 ViewMenuPtr->UnselectBackColor);
	}
	else
		WpsScrBackFor( ViewMenuPtr->UnselectBackColor,
										 ViewMenuPtr->UnselectForeColor);

	if (ViewMenuPtr->iLabelIndex < 0)
	{
		Col += (ViewMenuPtr->LabelWidth + 1);
		TempBuf[ 0] = 0;
	}
	else if (!ViewMenuPtr->LabelWidth)
	{
		f_strcpy( TempBuf, Labels[ ViewMenuPtr->iLabelIndex]);
		f_strcpy( &TempBuf[ f_strlen( TempBuf)], " ");
	}
	else
	{
		for( uiLoop = 0; uiLoop < ViewMenuPtr->LabelWidth; uiLoop++)
		{
			TempBuf[ uiLoop] = '.';
		}
		
		TempBuf[ ViewMenuPtr->LabelWidth] = ' ';
		TempBuf[ ViewMenuPtr->LabelWidth + 1] = 0;
		f_memcpy( TempBuf, Labels[ ViewMenuPtr->iLabelIndex],
							f_strlen( Labels[ ViewMenuPtr->iLabelIndex]));
	}
	if (ViewMenuPtr->Option)
	{
		if (ViewMenuPtr->ItemNum == gv_uiViewMenuCurrItemNum)
			WpsStrOutXY( "*>", (Col - 2), Row);
		else
			WpsStrOutXY( "* ", (Col - 2), Row);
	}
	else
	{
		if (ViewMenuPtr->ItemNum == gv_uiViewMenuCurrItemNum)
			WpsStrOutXY( " >", (Col - 2), Row);
		else
			WpsStrOutXY( "  ", (Col - 2), Row);
	}
	if (TempBuf[ 0])
		WpsStrOutXY( TempBuf, Col, Row);

	/* Now output the value */

	Col += f_strlen( TempBuf);
	switch( ViewMenuPtr->ValueType & 0x0F)
	{
		case VAL_IS_LABEL_INDEX:
			WpsStrOutXY( (Labels[ ViewMenuPtr->Value]), Col, Row);
			break;
		case VAL_IS_ERR_INDEX:
			{
			eCorruptionType	eCorruption = (eCorruptionType)ViewMenuPtr->Value;
			WpsStrOutXY( FlmVerifyErrToStr( eCorruption), Col, Row);
			break;
			}
		case VAL_IS_TEXT_PTR:
			WpsStrOutXY( (const char *)ViewMenuPtr->Value, Col, Row);
			break;
		case VAL_IS_BINARY_HEX:
		case VAL_IS_BINARY_PTR:
		{
			FLMUINT		BytesPerLine = MAX_HORIZ_SIZE( Col);
			FLMUINT		BytesProcessed = 0;
			FLMUINT		i;
			FLMUINT		j;
			FLMUINT		k;
			FLMUINT		NumBytes;
			FLMBYTE *	ValPtr = (FLMBYTE *)ViewMenuPtr->Value;
			FLMUINT		ValLen = ViewMenuPtr->ValueLen;

			/* Process each character in the value */

			i = 0;
			j = 0;
			k = BytesPerLine * 3 + 5;
			NumBytes = 0;

			/* Fill up a single line with whatever will fit on the line in */
			/* hex format. */

			f_memset( TempBuf, ' ', 80);
			TempBuf[ k - 3] = '|';
			while( (BytesProcessed < ValLen) && (i < BytesPerLine))
			{
				ByteToHex( ValPtr[ BytesProcessed], (FLMBYTE *)&TempBuf[ j], TRUE);
				if ((ValPtr[ BytesProcessed] > ' ') &&
						(ValPtr[ BytesProcessed] <= 127))
					TempBuf[ k] = ValPtr[ BytesProcessed];
				k++;
				NumBytes++;
				BytesProcessed++;
				i++;
				j += 2;
				TempBuf[ j] = ' ';
				j++;
			}
			TempBuf[ k] = 0;

			/* Output the line */

			WpsStrOutXY( TempBuf, Col, Row);
			if (ViewMenuPtr->ItemNum == gv_uiViewMenuCurrItemNum)
				UpdateHorizCursor( TRUE);
			break;
		}
		case VAL_IS_NUMBER:
			switch( ViewMenuPtr->ValueType & 0xF0)
			{
				case DISP_DECIMAL:
					f_sprintf( (char *)TempBuf, "%u", (unsigned)ViewMenuPtr->Value);
					break;
				case DISP_HEX:
					if (ViewMenuPtr->Value == 0xFFFFFFFF)
						f_strcpy( TempBuf, "None");
					else if (ViewMenuPtr->Value == 0)
						f_strcpy( TempBuf, "0");
					else
						f_sprintf( (char *)TempBuf, "0x%X", (unsigned)ViewMenuPtr->Value);
					break;
				case DISP_DECIMAL_HEX:
					f_sprintf( (char *)TempBuf, "%u (0x%X)",
									 (unsigned)ViewMenuPtr->Value, (unsigned)ViewMenuPtr->Value);
					break;
				case DISP_HEX_DECIMAL:
				default:
					if (ViewMenuPtr->Value == 0xFFFFFFFF)
						f_strcpy( TempBuf, "None");
					else if (ViewMenuPtr->Value == 0)
						f_strcpy( TempBuf, "0");
					else
						f_sprintf( (char *)TempBuf, "0x%X (%u)",
										 (unsigned)ViewMenuPtr->Value, (unsigned)ViewMenuPtr->Value);
					break;
			}
			WpsStrOutXY( TempBuf, Col, Row);
			break;
		case VAL_IS_EMPTY:
		default:
			break;
	}
}

/***************************************************************************
Name:		ViewAddMenuItem
Desc:		This routine adds a menu item to the item list.
*****************************************************************************/
FLMINT ViewAddMenuItem(
	FLMINT		LabelIndex,
	FLMUINT		LabelWidth,
	FLMUINT		ValueType,
	FLMUINT		Value,
	FLMUINT		ValueLen,
	FLMUINT		ModFileNumber,
	FLMUINT		ModFileOffset,
	FLMUINT		ModBufLen,
	FLMUINT		ModType,
	FLMUINT		Col,
	FLMUINT		Row,
	FLMUINT		Option,
	FLMUINT		UnselectBackColor,
	FLMUINT		UnselectForeColor,
	FLMUINT		SelectBackColor,
	FLMUINT		SelectForeColor
	)
{
	VIEW_MENU_ITEM_p	ViewMenuPtr;
	FLMUINT				Size = sizeof( VIEW_MENU_ITEM);

	/* Allocate the memory for the item */


	if ((ValueType & 0x0F) == VAL_IS_TEXT_PTR)
		Size += (f_strlen( (const char *)Value) + 1);
	else if ((ValueType & 0x0F) == VAL_IS_BINARY)
		Size += ValueLen;
	if ((ViewMenuPtr = 
			(VIEW_MENU_ITEM_p)GedPoolAlloc( &gv_ViewPool, Size)) == NULL)
	{
		ViewShowError( "Could not allocate memory for menu value");
		return( 0);
	}

	/* Link the item into the array and set some values in the item */

	ViewMenuPtr->NextItem = NULL;
	ViewMenuPtr->PrevItem = gv_pViewMenuLastItem;
	if (gv_pViewMenuLastItem != NULL)
	{
		gv_pViewMenuLastItem->NextItem = ViewMenuPtr;
		ViewMenuPtr->ItemNum = gv_pViewMenuLastItem->ItemNum + 1;
	}
	else
	{
		gv_pViewMenuFirstItem = ViewMenuPtr;
		ViewMenuPtr->ItemNum = 0;
	}
	gv_pViewMenuLastItem = ViewMenuPtr;
	ViewMenuPtr->iLabelIndex = LabelIndex;
	ViewMenuPtr->LabelWidth = LabelWidth;
	ViewMenuPtr->ValueType = ValueType;
	if ((ValueType & 0x0F) == VAL_IS_TEXT_PTR)
	{
		ViewMenuPtr->Value = (FLMUINT)((FLMBYTE *)&ViewMenuPtr[ 1]);
		f_strcpy( (char *)ViewMenuPtr->Value, (const char *)Value);
	}
	else if ((ValueType & 0x0F) == VAL_IS_BINARY)
	{
		ViewMenuPtr->ValueType = VAL_IS_BINARY_PTR;
		ViewMenuPtr->Value = (FLMUINT)((FLMBYTE *)&ViewMenuPtr[ 1]);
		f_memcpy( (void *)ViewMenuPtr->Value, (void *)Value, ValueLen);
	}
	else
		ViewMenuPtr->Value = Value;
	ViewMenuPtr->ValueLen = ValueLen;
	ViewMenuPtr->ModFileOffset = ModFileOffset;
	ViewMenuPtr->ModFileNumber = ModFileNumber;
	ViewMenuPtr->ModBufLen = ModBufLen;
	ViewMenuPtr->ModType = ModType;
	ViewMenuPtr->Col = Col;
	ViewMenuPtr->Row = Row;
	ViewMenuPtr->Option = Option;
	ViewMenuPtr->UnselectBackColor = UnselectBackColor;
	ViewMenuPtr->UnselectForeColor = UnselectForeColor;
	ViewMenuPtr->SelectBackColor = SelectBackColor;
	ViewMenuPtr->SelectForeColor = SelectForeColor;
	ViewMenuPtr->HorizCurPos = 0;
	if ((ViewMenuPtr->Row >= gv_uiViewTopRow) &&
			(ViewMenuPtr->Row <= gv_uiViewBottomRow))
		ViewDispMenuItem( ViewMenuPtr);
	return( 1);
}

/***************************************************************************
Name:		ViewEscPrompt
Desc:		This routine displays the prompt to press ESCAPE.	This prompt
			appears at the bottom of every screen.
*****************************************************************************/
void ViewEscPrompt(
	void
	)
{
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	WpsScrSize( &uiNumCols, &uiNumRows);
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, uiNumRows - 1);
	WpsScrBackFor( WPS_RED, WPS_WHITE);
	WpsStrOutXY( "ESC=Exit, ?=Help", 0, uiNumRows - 1);
	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsStrOutXY( "File: ", 20, uiNumRows - 1);
	WpsStrOutXY( gv_szViewFileName, 26, uiNumRows - 1);
	gv_uiViewLastFileOffset = VIEW_INVALID_FILE_OFFSET;
}

/***************************************************************************
Name:		ViewRefreshMenu
Desc:		This routine refreshes the menu display.	If NULL is passed in
				the entire screen is refreshed.	Otherwise, the item passed as
				well as the current item are refreshed.
*****************************************************************************/
FSTATIC void ViewRefreshMenu(
	VIEW_MENU_ITEM_p	PrevItem
	)
{
	VIEW_MENU_ITEM_p	ViewMenuPtr;

	gv_uiViewMenuCurrItemNum = gv_pViewMenuCurrItem->ItemNum;
	if (PrevItem != NULL)
	{
		ViewDispMenuItem( PrevItem);
		ViewDispMenuItem( gv_pViewMenuCurrItem);
	}
	else
	{

		/* Refresh the entire screen */

		WpsScrBackFor( WPS_BLACK, WPS_WHITE);
		WpsScrClr( 0, 1);

		ViewMenuPtr = gv_pViewMenuFirstItem;
		while( ViewMenuPtr)
		{
			if ((ViewMenuPtr->Row >= gv_uiViewTopRow) &&
					(ViewMenuPtr->Row <= gv_uiViewBottomRow))
				ViewDispMenuItem( ViewMenuPtr);
			ViewMenuPtr = ViewMenuPtr->NextItem;
		}
		ViewEscPrompt();
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void UpdateHorizCursor(
	FLMUINT	OnFlag
	)
{
	char			TempBuf[ 4];
	FLMUINT		i;
	FLMUINT		Row;
	FLMUINT		Col = gv_pViewMenuCurrItem->Col + 1;
	FLMBYTE *	ValPtr = (FLMBYTE *)gv_pViewMenuCurrItem->Value;

	if (gv_pViewMenuCurrItem->HorizCurPos > HORIZ_SIZE( gv_pViewMenuCurrItem) - 1)
		gv_pViewMenuCurrItem->HorizCurPos = HORIZ_SIZE( gv_pViewMenuCurrItem) - 1;
	i = gv_pViewMenuCurrItem->HorizCurPos;
	ByteToHex( *(ValPtr + i), (FLMBYTE *)TempBuf, TRUE);
	TempBuf[ 2] = 0;
	if (OnFlag)
		WpsScrBackFor( WPS_RED, WPS_WHITE);
	else
		WpsScrBackFor( gv_pViewMenuCurrItem->UnselectForeColor,
									 gv_pViewMenuCurrItem->UnselectBackColor);

	/* Calculate row and column where the item is to be displayed */

	Row = gv_uiTopLine + (gv_pViewMenuCurrItem->Row - gv_uiViewTopRow);
	WpsStrOutXY( TempBuf, (Col + i * 3), Row);
	if (((TempBuf[ 0] = ValPtr[ i]) < ' ') ||
			((FLMBYTE)TempBuf[ 0] > 127))
		TempBuf[ 0] = ' ';
#if defined( FLM_WIN)
	if (OnFlag)
		TempBuf [0] = 128;
#endif
	TempBuf[ 1] = 0;
	WpsStrOutXY( TempBuf,
			(Col + MAX_HORIZ_SIZE( Col) * 3 + 5 + i), Row);
	WpsScrBackFor( gv_pViewMenuCurrItem->UnselectForeColor,
								 gv_pViewMenuCurrItem->UnselectBackColor);
	if (OnFlag)
		WpsStrOutXY( ">", (Col + i * 3 - 1), Row);
	else
		WpsStrOutXY( " ", (Col + i * 3 - 1), Row);
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoUpArrow(
	void
	)
{
	VIEW_MENU_ITEM_p	ViewMenuPrevItem;

	if ((ViewMenuPrevItem = gv_pViewMenuCurrItem->PrevItem) != NULL)
	{
		if (ViewMenuPrevItem->HorizCurPos != gv_pViewMenuCurrItem->HorizCurPos)
			ViewMenuPrevItem->HorizCurPos = gv_pViewMenuCurrItem->HorizCurPos;
		gv_pViewMenuCurrItem = ViewMenuPrevItem;
		if (gv_pViewMenuCurrItem->Row < gv_uiViewTopRow)
		{
			gv_uiViewTopRow--;
			gv_uiViewBottomRow--;
			if (gv_pViewMenuCurrItem->Row < gv_uiViewTopRow)
				gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->NextItem;
			ViewRefreshMenu( NULL);
		}
		else
			ViewRefreshMenu( gv_pViewMenuCurrItem->NextItem);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoDownArrow(
	void
	)
{
	VIEW_MENU_ITEM_p	ViewMenuNextItem;

	if ((ViewMenuNextItem = gv_pViewMenuCurrItem->NextItem) != NULL)
	{
		if (ViewMenuNextItem->HorizCurPos != gv_pViewMenuCurrItem->HorizCurPos)
			ViewMenuNextItem->HorizCurPos = gv_pViewMenuCurrItem->HorizCurPos;
		gv_pViewMenuCurrItem = ViewMenuNextItem;
		if (gv_pViewMenuCurrItem->Row > gv_uiViewBottomRow)
		{
			gv_uiViewTopRow++;
			gv_uiViewBottomRow++;
			if (gv_pViewMenuCurrItem->Row > gv_uiViewBottomRow)
				gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->PrevItem;
			ViewRefreshMenu( NULL);
		}
		else
			ViewRefreshMenu( gv_pViewMenuCurrItem->PrevItem);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoPageDown(
	void
	)
{
	FLMUINT					TargRow;
	VIEW_MENU_ITEM_p	ViewSaveItem;

	if (gv_uiViewBottomRow < gv_pViewMenuLastItem->Row)
	{
		gv_uiViewBottomRow += LINES_PER_PAGE;
		if (gv_uiViewBottomRow > gv_pViewMenuLastItem->Row)
			gv_uiViewBottomRow = gv_pViewMenuLastItem->Row;
		gv_uiViewTopRow = gv_uiViewBottomRow - LINES_PER_PAGE + 1;
		TargRow = gv_pViewMenuCurrItem->Row + LINES_PER_PAGE;
		if (TargRow > gv_uiViewBottomRow)
			TargRow = gv_uiViewBottomRow;
		while( (gv_pViewMenuCurrItem->NextItem != NULL) &&
					 (gv_pViewMenuCurrItem->Row < TargRow))
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->NextItem;
		if (gv_pViewMenuCurrItem->Row > gv_uiViewBottomRow)
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->PrevItem;
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->NextItem != NULL)
	{
		ViewSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuLastItem;
		ViewRefreshMenu( ViewSaveItem);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoPageUp(
	void
	)
{
	FLMUINT				TargRow;
	VIEW_MENU_ITEM_p	ViewSaveItem;

	if (gv_uiViewTopRow > 0)
	{
		if (gv_uiViewTopRow < LINES_PER_PAGE)
			gv_uiViewTopRow = 0;
		else
			gv_uiViewTopRow -= LINES_PER_PAGE;
		gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
		TargRow = gv_pViewMenuCurrItem->Row - LINES_PER_PAGE;
		if (TargRow < gv_uiViewTopRow)
			TargRow = gv_uiViewTopRow;
		while( (gv_pViewMenuCurrItem->PrevItem != NULL) &&
					 (gv_pViewMenuCurrItem->Row > TargRow))
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->PrevItem;
		if (gv_pViewMenuCurrItem->Row < gv_uiViewTopRow)
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->NextItem;
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->PrevItem != NULL)
	{
		ViewSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		ViewRefreshMenu( ViewSaveItem);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoHome(
	void
	)
{
	VIEW_MENU_ITEM_p	ViewSaveItem;

	if (gv_uiViewTopRow != 0)
	{
		gv_uiViewTopRow = 0;
		gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->PrevItem != NULL)
	{
		ViewSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		ViewRefreshMenu( ViewSaveItem);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoEnd(
	void
	)
{
	VIEW_MENU_ITEM_p	ViewSaveItem;

	if (gv_uiViewBottomRow < gv_pViewMenuLastItem->Row)
	{
		gv_uiViewBottomRow = gv_pViewMenuLastItem->Row;
		gv_uiViewTopRow = gv_uiViewBottomRow - LINES_PER_PAGE + 1;
		gv_pViewMenuCurrItem = gv_pViewMenuLastItem;
		ViewRefreshMenu( NULL);
	}
	else if (gv_pViewMenuCurrItem->NextItem != NULL)
	{
		ViewSaveItem = gv_pViewMenuCurrItem;
		gv_pViewMenuCurrItem = gv_pViewMenuLastItem;
		ViewRefreshMenu( ViewSaveItem);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoRightArrow(
	void
	)
{
	if ((!HAVE_HORIZ_CUR( gv_pViewMenuCurrItem)) ||
			(gv_pViewMenuCurrItem->HorizCurPos == HORIZ_SIZE( gv_pViewMenuCurrItem) - 1))
	{
		if (gv_pViewMenuCurrItem->ItemNum != gv_pViewMenuLastItem->ItemNum)
		{
			gv_pViewMenuCurrItem->HorizCurPos = 0;
			DoDownArrow();
		}
	}
	else if (gv_pViewMenuCurrItem->HorizCurPos <
					 HORIZ_SIZE( gv_pViewMenuCurrItem) - 1)
	{
		UpdateHorizCursor( FALSE);
		gv_pViewMenuCurrItem->HorizCurPos++;
		UpdateHorizCursor( TRUE);
	}
}

/********************************************************************
Desc: ?
*********************************************************************/
FSTATIC void DoLeftArrow(
	void
	)
{
	if ((!HAVE_HORIZ_CUR( gv_pViewMenuCurrItem)) ||
			(gv_pViewMenuCurrItem->HorizCurPos == 0))
	{
		if (gv_pViewMenuCurrItem->PrevItem != NULL)
			gv_pViewMenuCurrItem->HorizCurPos =
				HORIZ_SIZE( gv_pViewMenuCurrItem->PrevItem) - 1;
		DoUpArrow();
	}
	else if (gv_pViewMenuCurrItem->HorizCurPos > 0)
	{
		UpdateHorizCursor( FALSE);
		gv_pViewMenuCurrItem->HorizCurPos--;
		UpdateHorizCursor( TRUE);
	}
}

/***************************************************************************
Name:		ViewHelpScreen
Desc:		This routine displays a help screen showing available commands.
*****************************************************************************/
FSTATIC void ViewHelpScreen(
	void
	)
{
	FLMUINT	 wChar;

	/* Clear the screen and display the menu title */

	WpsScrBackFor( WPS_BLACK, WPS_WHITE);
	WpsScrClr( 0, 1);
	WpsScrPos( 0, 3);
	WpsStrOut( "     RECOGNIZED KEYBOARD CHARACTERS\n");
	WpsStrOut( "\n");
	WpsStrOut( "     ESCAPE               - Exit Screen\n");
	WpsStrOut( "     U,u,8                - Up Arrow\n");
	WpsStrOut( "     D,d,2                - Down Arrow\n");
	WpsStrOut( "     +,3                  - Page Down\n");
	WpsStrOut( "     R,r,6                - Right Arrow\n");
	WpsStrOut( "     L,l,5                - Left Arrow\n");
	WpsStrOut( "     -,9                  - Page Up\n");
	WpsStrOut( "     H,h,7                - Home\n");
	WpsStrOut( "     Z,z,1                - End\n");
	WpsStrOut( "     E,e                  - Edit Data\n");
	WpsStrOut( "     A,a                  - Edit Data in RAW Mode (no checksum)\n");
	WpsStrOut( "     G,g                  - Goto Block\n");
	WpsStrOut( "     X,x                  - Display Hex\n");
	WpsStrOut( "     Y,y                  - Display Decrypted\n");
	WpsStrOut( "     S,s                  - Search\n");
	WpsStrOut( "     ?                    - Show this help screen\n");
	WpsStrOut( "\n");
	WpsStrOut( "	  PRESS ANY CHARACTER TO EXIT HELP SCREEN\n");

	for (;;)
	{

		/* Update date and time */

		ViewUpdateDate( FALSE, &gv_ViewLastTime);

		/* See what character was pressed */

		wChar = (!WpkTestKB()) ? 0 : (WpkIncar());
		if (gv_bShutdown)
			return;
		if (wChar)
			break;
		viewGiveUpCPU();
	}
	ViewRefreshMenu( NULL);
}

/***************************************************************************
Name:		ViewGetMenuOption
Desc:		This routine allows the user to press keys while in a menu.
			Keys for navigating through the menu are handled inside this
			routine.	 Other keys are passed to the calling routine or are
			ignored altogether.
*****************************************************************************/
FLMINT ViewGetMenuOption(
	void
	)
{
	FLMUINT c;

	/* Make sure we have a pointer to the current item */

	if (gv_pViewMenuCurrItem == NULL)
	{
		gv_pViewMenuCurrItem = gv_pViewMenuFirstItem;
		while( (gv_pViewMenuCurrItem->NextItem != NULL) &&
					 (gv_pViewMenuCurrItem->ItemNum < gv_uiViewMenuCurrItemNum))
			gv_pViewMenuCurrItem = gv_pViewMenuCurrItem->NextItem;
		if (gv_pViewMenuCurrItem->ItemNum != gv_uiViewMenuCurrItemNum)
		{
			gv_uiViewMenuCurrItemNum = gv_pViewMenuCurrItem->ItemNum;
			ViewDispMenuItem( gv_pViewMenuCurrItem);
		}
	}

	/* Loop getting user input */

	ViewEscPrompt();
	for( ;;)
	{

		/* Set the file position of the current object */

		if (gv_pViewMenuCurrItem->ModFileOffset != VIEW_INVALID_FILE_OFFSET)
		{
			gv_uiViewCurrFileOffset = gv_pViewMenuCurrItem->ModFileOffset;
			gv_uiViewCurrFileNumber = gv_pViewMenuCurrItem->ModFileNumber;
			if (HAVE_HORIZ_CUR( gv_pViewMenuCurrItem))
				gv_uiViewCurrFileOffset += gv_pViewMenuCurrItem->HorizCurPos;
		}

		/* Update date and time */

		ViewUpdateDate( FALSE, &gv_ViewLastTime);

		/* See what character was pressed */

		viewGiveUpCPU();
		c = (!WpkTestKB()) ? 0 : (WpkIncar());
		if (gv_bShutdown)
			return( ESCAPE_OPTION);
		switch( c)
		{
			case WPK_ESCAPE:
				return( ESCAPE_OPTION);
			case WPK_UP:
			case 'U':
			case 'u':
			case '8':
				DoUpArrow();
				break;
			case WPK_DOWN:
			case 'D':
			case 'd':
			case '2':
				DoDownArrow();
				break;
			case WPK_PGDN:
			case '+':
			case '3':
				DoPageDown();
				break;
			case WPK_PGUP:
			case '-':
			case '9':
				DoPageUp();
				break;
			case WPK_HOME:
			case 'H':
			case 'h':
			case '7':
				DoHome();
				break;
			case WPK_END:
			case 'Z':
			case 'z':
			case '1':
				DoEnd();
				break;
			case '\n':
			case '\r':
			case WPK_ENTER:
				if (gv_pViewMenuCurrItem->Option)
					return( gv_pViewMenuCurrItem->Option);
				break;
			case 'G':
			case 'g':
			case 7:	/* Control-G */
				return( GOTO_BLOCK_OPTION);
			case WPK_RIGHT:
			case 'R':
			case 'r':
			case '6':
				DoRightArrow();
				break;
			case WPK_LEFT:
			case 'L':
			case 'l':
			case '4':
				DoLeftArrow();
				break;
			case 'E':
			case 'e':
				return( EDIT_OPTION);
			case 'A':
			case 'a':
				return( EDIT_RAW_OPTION);
			case 'x':
			case 'X':
				return( HEX_OPTION);
			case 'Y':
			case 'y':
				return( DECRYPT_OPTION);
			case 'S':
			case 's':
				return( SEARCH_OPTION);
			case '?':
				ViewHelpScreen();
				break;
			default:
				break;
		}
	}
}

/***************************************************************************
Name:		ViewUpdateDate
Desc:		This routine updates the date and time on the screen.
*****************************************************************************/
void ViewUpdateDate(
	FLMUINT			UpdateFlag,
	F_TMSTAMP  *	LastTime
	)
{
	F_TMSTAMP	CurrTime;
	char			TempBuf[ 40];
	FLMUINT		Hour;
	FLMBYTE		AmPm[ 4];
	FLMUINT		uiNumCols;
	FLMUINT		uiNumRows;

	WpsScrSize( &uiNumCols, &uiNumRows);
	f_timeGetTimeStamp( &CurrTime);

	/* Update the date, if it has changed or the UpdateFlag is set */

	if ((UpdateFlag) ||
			(LastTime->year != CurrTime.year) ||
			(LastTime->month != CurrTime.month) ||
			(LastTime->day != CurrTime.day))
	{
		f_sprintf( (char *)TempBuf, "%s %u, %u",
						 Months[ CurrTime.month],
						 (unsigned)CurrTime.day,
						 (unsigned)CurrTime.year);
		WpsScrBackFor( WPS_BLACK, WPS_WHITE);
		WpsStrOutXY( TempBuf, 0, 0);
	}

	/* Update the time, if it has changed or the UpdateFlag is set */

	if ((UpdateFlag) ||
			(LastTime->hour != CurrTime.hour) ||
			(LastTime->minute != CurrTime.minute) ||
			(LastTime->second != CurrTime.second))
	{
		if (CurrTime.hour == 0)
			Hour = 12;
		else if (CurrTime.hour > 12)
			Hour = (FLMUINT)CurrTime.hour - 12;
		else
			Hour = (FLMUINT)CurrTime.hour;
		if (CurrTime.hour >= 12)
			f_strcpy( (char *)AmPm, "pm");
		else
			f_strcpy( (char *)AmPm, "am");
		f_sprintf( (char *)TempBuf, "%2u:%02u:%02u %s", 
						(unsigned)Hour,
						(unsigned)CurrTime.minute,
						(unsigned)CurrTime.second, AmPm);
		WpsScrBackFor( WPS_BLACK, WPS_WHITE);
		WpsStrOutXY( TempBuf, 66, 0);
	}

	if ((UpdateFlag) ||
		  (gv_uiViewLastFileOffset != gv_uiViewCurrFileOffset) ||
		  ((gv_pViewMenuCurrItem->ModFileOffset == VIEW_INVALID_FILE_OFFSET) &&
			 (gv_uiViewLastFileOffset != VIEW_INVALID_FILE_OFFSET)))
	{
		if (gv_pViewMenuCurrItem == NULL)
		{
			gv_uiViewLastFileOffset = VIEW_INVALID_FILE_OFFSET;
		}
		else if (gv_pViewMenuCurrItem->ModFileOffset == VIEW_INVALID_FILE_OFFSET)
		{
			gv_uiViewLastFileNumber = gv_pViewMenuCurrItem->ModFileNumber;
			gv_uiViewLastFileOffset = gv_pViewMenuCurrItem->ModFileOffset;
		}
		else
		{
			gv_uiViewLastFileNumber = gv_uiViewCurrFileNumber;
			gv_uiViewLastFileOffset = gv_uiViewCurrFileOffset;
		}

		if (gv_uiViewLastFileOffset == VIEW_INVALID_FILE_OFFSET)
			f_strcpy( TempBuf, "File: N/A  File Pos: N/A       ");
		else
			f_sprintf( (char *)TempBuf, "File: %03u  File Pos: 0x%08X", 
								(unsigned)gv_uiViewLastFileNumber, (unsigned)gv_uiViewLastFileOffset);
		WpsScrBackFor( WPS_BLACK, WPS_WHITE);
		WpsStrOutXY( TempBuf, 47, uiNumRows - 1);
	}

	/* Save the date and time */

	f_memcpy( LastTime, &CurrTime, sizeof( F_TMSTAMP));
}

/***************************************************************************
Name: ViewReset
Desc: This routine resets the view parameters for a menu and saves
		the parameters for the current menu.  This is done whenever a new
		menu is being entered.	It allows the previous menu to be restored
		to its original state upon returning.
*****************************************************************************/
void ViewReset(
	VIEW_INFO_p SaveView
	)
{
	SaveView->CurrItem = gv_uiViewMenuCurrItemNum;
	SaveView->TopRow = gv_uiViewTopRow;
	SaveView->BottomRow = gv_uiViewBottomRow;
	SaveView->CurrFileOffset = gv_uiViewCurrFileOffset;
	SaveView->CurrFileNumber = gv_uiViewCurrFileNumber;
	gv_pViewMenuCurrItem = NULL;
	gv_uiViewMenuCurrItemNum = 0;
	gv_uiViewTopRow = 0;
	gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
	gv_uiViewCurrFileOffset = 0;
	gv_uiViewCurrFileNumber = 0;
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewDisable(
	void
	)
{
	gv_bViewEnabled = FALSE;
}

/********************************************************************
Desc: ?
*********************************************************************/
void ViewEnable(
	void
	)
{
	VIEW_MENU_ITEM_p	vp;
	FLMUINT				Distance = 0xFFFFFFFF;
	VIEW_MENU_ITEM_p	Closest = NULL;
	FLMUINT				StartOffset;
	FLMUINT				EndOffset;

	if (!gv_bViewEnabled)
	{
		if (gv_uiViewCurrFileOffset != VIEW_INVALID_FILE_OFFSET)
		{
			vp = gv_pViewMenuFirstItem;
			while( vp != NULL)
			{
				if (vp->ModFileOffset != VIEW_INVALID_FILE_OFFSET)
				{
					StartOffset = vp->ModFileOffset;
					switch( vp->ModType & 0x0F)
					{
						case MOD_FLMUINT:
						case MOD_KEY_LEN:
							EndOffset = StartOffset + 3;
							break;
						case MOD_FLMUINT16:
							EndOffset = StartOffset + 1;
							break;
						case MOD_BINARY:
						case MOD_TEXT:
							EndOffset = StartOffset + vp->ModBufLen - 1;
							break;
						case MOD_CHILD_BLK:
							EndOffset = StartOffset + 2;
							break;
						default:
							EndOffset = StartOffset;
							break;
					}
					if ((gv_uiViewCurrFileOffset >= StartOffset) &&
						 (gv_uiViewCurrFileOffset <= EndOffset))
					{
						if ((vp->ModType & 0xF0) == MOD_DISABLED)
						{
							Closest = vp;
							Distance = 0;
						}
						else
						{
							Closest = vp;
							break;
						}
					}
					else if (gv_uiViewCurrFileOffset < StartOffset)
					{
						if (StartOffset - gv_uiViewCurrFileOffset < Distance)
						{
							Closest = vp;
							Distance = StartOffset - gv_uiViewCurrFileOffset;
						}
					}
					else
					{
						if (gv_uiViewCurrFileOffset - StartOffset < Distance)
						{
							Closest = vp;
							Distance = gv_uiViewCurrFileOffset - StartOffset;
						}
					}
				}
				vp = vp->NextItem;
			}
		}
		if (Closest != NULL)
		{
			gv_pViewMenuCurrItem = vp = Closest;
			gv_uiViewMenuCurrItemNum = vp->ItemNum;
			if (vp->Row < LINES_PER_PAGE)
				gv_uiViewTopRow = 0;
			else
				gv_uiViewTopRow = vp->Row - LINES_PER_PAGE / 2 + 1;
			gv_uiViewBottomRow = gv_uiViewTopRow + LINES_PER_PAGE - 1;
			if (gv_uiViewBottomRow > gv_pViewMenuLastItem->Row)
				gv_uiViewBottomRow = gv_pViewMenuLastItem->Row;
			if (gv_uiViewBottomRow - gv_uiViewTopRow + 1 < LINES_PER_PAGE)
			{
				if (gv_uiViewBottomRow < LINES_PER_PAGE + 1)
					gv_uiViewTopRow = 0;
				else
					gv_uiViewTopRow = gv_uiViewBottomRow + 1 - LINES_PER_PAGE;
			}
			if ((HAVE_HORIZ_CUR( vp)) &&
					(gv_uiViewCurrFileOffset - vp->ModFileOffset <=
						(FLMUINT)vp->ModBufLen))
				vp->HorizCurPos =
					(FLMUINT)(gv_uiViewCurrFileOffset - vp->ModFileOffset);
		}
		gv_bViewEnabled = TRUE;
		ViewRefreshMenu( NULL);
	}
}

/***************************************************************************
Name: ViewRestore
Desc: This routine restores the view parameters for a menu which were
		previously saved by the ViewReset routine.
*****************************************************************************/
void ViewRestore(
	VIEW_INFO_p SaveView
	)
{
	gv_uiViewMenuCurrItemNum = SaveView->CurrItem;
	gv_pViewMenuCurrItem = NULL;
	gv_uiViewTopRow = SaveView->TopRow;
	gv_uiViewBottomRow = SaveView->BottomRow;
	gv_uiViewCurrFileOffset = SaveView->CurrFileOffset;
	gv_uiViewCurrFileNumber = SaveView->CurrFileNumber;
}
