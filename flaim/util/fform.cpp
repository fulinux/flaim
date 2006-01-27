//-------------------------------------------------------------------------
// Desc:	FlmForm class
// Tabs:	3
//
//		Copyright (c) 1999-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fform.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "fform.h"
#include "flaimsys.h"

#define COLUMN_DIFF(uiCol1,uiCol2) \
	(FLMUINT)(((uiCol1) <= (uiCol2)) \
				 ? (FLMUINT)((uiCol2) - (uiCol1)) \
				 : (FLMUINT)((uiCol1) - (uiCol2)))

FSTATIC FLMBOOL flmIsUnsigned(
	char *				pszEditBuf,
	FLMUINT *			puiValue,
	char *				pszErrMsg);

FSTATIC RCODE flintRecEditKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

FSTATIC FLMBOOL pulldownKeyFunc(
	FlmPulldownList *	pPulldown,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvAppData);

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmForm::FlmForm()
{
	m_pScreen = NULL;
	m_pThread = NULL;
	m_bMonochrome = FALSE;		// Future support
	m_pDisplayWindow = NULL;
	m_pStatusWindow = NULL;
	m_uiRows = 0;
	m_uiCols = 0;
	m_uiUpperLeftRow = 0;
	m_uiUpperLeftColumn = 0;
	m_uiBackColor = WPS_BLUE;
	m_uiForeColor = WPS_WHITE;
	m_uiTopFormRow = 0;
	m_pCurObject = NULL;
	m_pFirstObject = NULL;
	m_pLastObject = NULL;
	m_pEventCB = NULL;
	m_pvAppData = NULL;
	m_bGetKeyStrokes = FALSE;
	m_bShowingHelpStatus = FALSE;
	m_uiLastTimeBeeped = 0;
	m_bValuesChanged = FALSE;

	m_bInteracting = FALSE;
	m_pszEditBuf = NULL;
	m_uiEditBufSize = 0;
	m_uiMaxCharsToEnter = 0;
	m_uiNumCharsEntered = 0;
	m_uiEditColumn = 0;
	m_uiEditRow = 0;
	m_uiEditWidth = 0;
	m_uiEditBufPos = 0;
	m_uiEditBufLeftColPos = 0;
}

FlmForm::~FlmForm()
{
	FlmFormObject *	pFormObject;
	FlmFormObject *	pNextObject;

	// Delete all form objects

	pFormObject = m_pFirstObject;
	while (pFormObject)
	{
		pNextObject = pFormObject->getNextObject();
		pFormObject->Release();
		pFormObject = pNextObject;
		if (pFormObject == m_pFirstObject)
		{
			break;
		}
	}

	// Release the windows

	if (m_pDisplayWindow)
	{
		(void)FTXWinFree( &m_pDisplayWindow);
	}
	if (m_pStatusWindow)
	{
		(void)FTXWinFree( &m_pStatusWindow);
	}
	if (m_pszEditBuf)
	{
		f_free( &m_pszEditBuf);
	}
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
RCODE FlmForm::init(
	FTX_SCREEN_p			pScreen,
	FlmThreadContext *	pThread,
	const char *			pszTitle,
	FLMUINT					uiTitleBackColor,
	FLMUINT					uiTitleForeColor,
	const char *			pszHelp,
	FLMUINT					uiHelpBackColor,
	FLMUINT					uiHelpForeColor,
	FLMUINT					uiULX,
	FLMUINT					uiULY,
	FLMUINT					uiLRX,
	FLMUINT					uiLRY,
	FLMBOOL					bCreateStatusBar,
	FLMBOOL					bCreateBorder,
	FLMUINT					uiBackColor,
	FLMUINT					uiForeColor)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiNumCols;
	FLMUINT			uiNumRows;
	FLMUINT			uiStartCol;

	// Form must not already have been initialized.

	if (m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_pScreen = pScreen;
	m_pThread = pThread;

	if( !uiLRX && !uiLRY)
	{
		if( FTXScreenGetSize( pScreen,
			&uiNumCols, &uiNumRows) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		uiNumRows -= uiULY;
		uiNumCols -= uiULX;

	}
	else
	{
		uiNumRows = (uiLRY - uiULY) + 1;
		uiNumCols = (uiLRX - uiULX) + 1;
	}

	uiStartCol = uiULX;

	if (bCreateStatusBar)
	{
		uiNumRows -= 2; // Add 2 to account for the status bar
	}

	// Create the display window of the appropriate size.

	if (FTXWinInit( pScreen, uiNumCols,
		uiNumRows, &m_pDisplayWindow) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Position the window on the screen.

	m_uiUpperLeftRow = uiULY;
	m_uiUpperLeftColumn = uiStartCol;
	if (FTXWinMove( m_pDisplayWindow, uiStartCol, uiULY) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Prevent window from scrolling.

	if (FTXWinSetScroll( m_pDisplayWindow, FALSE) != FTXRC_SUCCESS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Don't wrap lines.

	if (FTXWinSetLineWrap( m_pDisplayWindow, FALSE) != FTXRC_SUCCESS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Input cursor should not be present initially.

	if (FTXWinSetCursorType( m_pDisplayWindow,
		WPS_CURSOR_INVISIBLE) != FTXRC_SUCCESS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Set foreground and background color for window.

	if (FTXWinSetBackFore( m_pDisplayWindow,
		m_bMonochrome ? WPS_BLACK : uiBackColor,
		m_bMonochrome ? WPS_WHITE : uiForeColor) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	m_uiBackColor = uiBackColor;
	m_uiForeColor = uiForeColor;

	// Clear the window

	if (FTXWinClear( m_pDisplayWindow) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (bCreateBorder)
	{
		if (FTXWinDrawBorder( m_pDisplayWindow) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		m_uiUpperLeftRow++;
		m_uiUpperLeftColumn++;
	}

	if (pszTitle && *pszTitle)
	{
		if (FTXWinSetTitle( m_pDisplayWindow, pszTitle,
				uiTitleBackColor, uiTitleForeColor) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	if (pszHelp && *pszHelp)
	{
		if (FTXWinSetHelp( m_pDisplayWindow, pszHelp,
								uiHelpBackColor, uiHelpForeColor) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	if (bCreateStatusBar)
	{
		if (FTXWinInit( pScreen, uiNumCols, 2,
			&m_pStatusWindow) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (FTXWinMove( m_pStatusWindow, uiStartCol,
			uiULY + uiNumRows) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (FTXWinSetScroll( m_pStatusWindow, FALSE) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		if (FTXWinSetCursorType( m_pStatusWindow,
			WPS_CURSOR_INVISIBLE) != FTXRC_SUCCESS)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}

		if (FTXWinSetBackFore( m_pStatusWindow,
			m_bMonochrome ? WPS_BLACK : WPS_GREEN,
			WPS_WHITE) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (FTXWinClear( m_pStatusWindow) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if (FTXWinOpen( m_pStatusWindow) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}

	// Open the display window AFTER the status
	// window so it will get the input focus.
	
	if (FTXWinOpen( m_pDisplayWindow) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Get the actual number of rows and columns that are
	// available in the window - must be done after
	// setting the border, etc.

	if (FTXWinGetCanvasSize( m_pDisplayWindow, &m_uiCols,
		&m_uiRows) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	m_uiTopFormRow = 0;
	refresh();

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Redisplay the form.
===========================================================================*/
void FlmForm::refresh( void)
{
	FlmFormObject *	pObject;

	// Set foreground and background color for window.

	FTXWinSetBackFore( m_pDisplayWindow,
		m_bMonochrome ? WPS_BLACK : m_uiBackColor,
		m_bMonochrome ? WPS_WHITE : m_uiForeColor);

	// Clear the screen

	(void)FTXWinClear( m_pDisplayWindow);

	// Redisplay all of the items on the screen.

	pObject = m_pFirstObject;
	while (pObject)
	{
		displayObject( pObject);
		if ((pObject = pObject->getNextObject()) == m_pFirstObject)
			break;
	}
}

/*===========================================================================
Desc:		Sets foreground and background color for an object in the form.
===========================================================================*/
void FlmForm::setObjectColor(
	FLMUINT			uiObjectId,
	FLMUINT			uiBackColor,
	FLMUINT			uiForeColor
	)
{
	FlmFormObject *	pObject;

	// Form must be initialized

	flmAssert( m_pDisplayWindow != NULL);

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		flmAssert( 0);
	}
	else
	{

		// Set the foreground and background color of the object.

		pObject->setBackColor( uiBackColor);
		pObject->setForeColor( uiForeColor);

		// Redisplay the object.

		displayObject( pObject);
	}
}

/*===========================================================================
Desc:		Gets an object's foreground and background colors.
===========================================================================*/
void FlmForm::getObjectColor(
	FLMUINT		uiObjectId,
	FLMUINT *	puiBackColor,
	FLMUINT *	puiForeColor
	)
{
	FlmFormObject *	pObject;

	// Form must be initialized

	flmAssert( m_pDisplayWindow != NULL);

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		flmAssert( 0);
	}
	else
	{

		// Get the foreground and background color of the object.

		*puiBackColor = pObject->getBackColor();
		*puiForeColor = pObject->getForeColor();
	}
}

/*===========================================================================
Desc:		Sets display only flag for an object in the form.
===========================================================================*/
void FlmForm::setObjectDisplayOnlyFlag(
	FLMUINT		uiObjectId,
	FLMBOOL		bDisplayOnly
	)
{
	FlmFormObject *	pObject;

	// Form must be initialized

	flmAssert( m_pDisplayWindow != NULL);

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		flmAssert( 0);
	}
	else
	{

		// Set the display only flag for the object.

		pObject->setDisplayOnly( bDisplayOnly);
	}
}

/*===========================================================================
Desc:		Sets display only flag for an object in the form.
===========================================================================*/
void FlmForm::getObjectDisplayOnlyFlag(
	FLMUINT		uiObjectId,
	FLMBOOL *	pbDisplayOnly
	)
{
	FlmFormObject *	pObject;

	// Form must be initialized

	flmAssert( m_pDisplayWindow != NULL);

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		flmAssert( 0);
	}
	else
	{

		// Get the display only flag for the object.

		*pbDisplayOnly = pObject->isDisplayOnly();
	}
}

/*===========================================================================
Desc:		Sets value for an object in the form.
===========================================================================*/
RCODE FlmForm::setObjectValue(
	FLMUINT			uiObjectId,
	void *			pvValue,
	FLMUINT			uiLen
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	F_UNREFERENCED_PARM( uiLen);

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Set the object's value, depending on object type.

	switch (pObject->getObjectType())
	{
		case FORM_TEXT_OBJECT:
		{
			FlmFormTextObject *	pTextObject = (FlmFormTextObject *)pObject;
			rc = pTextObject->setValue( (const char *)pvValue);
			break;
		}
		case FORM_UNSIGNED_OBJECT:
		{
			FlmFormUnsignedObject *	pUnsignedObject =
				(FlmFormUnsignedObject *)pObject;
			rc = pUnsignedObject->setValue( (FLMUINT)pvValue);
			break;
		}
		case FORM_SIGNED_OBJECT:
		{
			FlmFormSignedObject *	pSignedObject =
				(FlmFormSignedObject *)pObject;
			rc = pSignedObject->setValue( (FLMINT)pvValue);
			break;
		}
		case FORM_PULLDOWN_OBJECT:
		{
			FlmFormPulldownObject *	pPulldownObject = (FlmFormPulldownObject *)pObject;
			rc = pPulldownObject->setCurrentItem( (FLMUINT)pvValue);
			break;
		}
		case FORM_RECORD_OBJECT:
		{
			FlmFormRecordObject *	pRecordObject = (FlmFormRecordObject *)pObject;
			rc = pRecordObject->setValue( (NODE *)pvValue);
			break;
		}
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}
	if (RC_BAD( rc))
	{
		goto Exit;
	}
	m_bValuesChanged = TRUE;

	// Redisplay the object.

	if (m_bInteracting && pObject == m_pCurObject)
	{

		// Call changeFocus to reset edit buffer.

		(void)changeFocus( pObject, FALSE, TRUE);
	}
	else
	{
		displayObject( pObject);
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets a return address to populate when the form has been
			filled in - or when the object changes its value.
===========================================================================*/
RCODE FlmForm::setObjectReturnAddress(
	FLMUINT		uiObjectId,
	void *		pvReturnAddress,
	FLMUINT *	puiReturnLen
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	pObject->setReturnAddress( pvReturnAddress, puiReturnLen);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets a return path in a GEDCOM tree to populate when the
			form has been filled in.
===========================================================================*/
RCODE FlmForm::setObjectReturnPath(
	FLMUINT		uiObjectId,
	FLMUINT *	puiReturnPath
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	pObject->setReturnPath( puiReturnPath);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets the help lines for an object in the form.
===========================================================================*/
RCODE FlmForm::setObjectHelp(
	FLMUINT			uiObjectId,
	const char *	pszHelpLine1,
	const char *	pszHelpLine2)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// See if we can find the object.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	rc = pObject->setHelp( pszHelpLine1, pszHelpLine2);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets value for a object in the form.
===========================================================================*/
RCODE FlmForm::setFormEventCB(
	FORM_EVENT_CB_p	pEventCB,
	void *				pvAppData,
	FLMBOOL				bGetKeyStrokes
	)
{
	RCODE	rc = FERR_OK;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_pEventCB = pEventCB;
	m_pvAppData = pvAppData;
	m_bGetKeyStrokes = bGetKeyStrokes;
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Add a text object to the form.
===========================================================================*/
RCODE FlmForm::addTextObject(
	FLMUINT			uiObjectId,
	const char *	pszDefaultVal,
	FLMUINT			uiMaxChars,
	FLMUINT			uiWidth,
	FLMUINT			uiFormat,
	FLMBOOL			bDisplayOnly,
	FLMUINT			uiBackColor,
	FLMUINT			uiForeColor,
	FLMUINT			uiRow,
	FLMUINT			uiColumn)
{
	RCODE						rc = FERR_OK;
	FlmFormObject *		pPrevObject;
	FlmFormTextObject *	pTextObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object ID is not already used.

	if ((pPrevObject = findObject( uiObjectId)) != NULL)
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Find the objects this object should be linked to.  This routine
	// also determines if there is an overlap.  If so, it will return
	// an error.

	if (RC_BAD( rc = getObjectLocation( uiWidth, uiRow, uiColumn,
							&pPrevObject)))
	{
		goto Exit;
	}

	// Create a text object and link into the form.

	if ((pTextObject = new FlmFormTextObject) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = pTextObject->setup( this, uiObjectId, uiMaxChars,
								uiWidth, uiFormat, bDisplayOnly,
								uiBackColor, uiForeColor,
								uiRow, uiColumn)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pTextObject->setValue( pszDefaultVal)))
	{
		goto Exit;
	}

	// Link the object into the form in its proper place.

	linkObjectInForm( pTextObject, pPrevObject);

	// Display the object

	displayObject( pTextObject);

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Add an unsigned numeric object to the form.
===========================================================================*/
RCODE FlmForm::addUnsignedObject(
	FLMUINT		uiObjectId,
	FLMUINT		uiDefaultVal,
	FLMUINT		uiMinVal,
	FLMUINT		uiMaxVal,
	FLMUINT		uiWidth,
	FLMUINT		uiFormat,
	FLMBOOL		bDisplayOnly,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn
	)
{
	RCODE							rc = FERR_OK;
	FlmFormObject *				pPrevObject;
	FlmFormUnsignedObject *	pUnsignedObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object ID is not already used.

	if ((pPrevObject = findObject( uiObjectId)) != NULL)
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Find the objects this object should be linked to.  This routine
	// also determines if there is an overlap.  If so, it will return
	// an error.

	if (RC_BAD( rc = getObjectLocation( uiWidth, uiRow, uiColumn,
							&pPrevObject)))
	{
		goto Exit;
	}

	// Create an unsigned number object and link into the form.

	if ((pUnsignedObject = new FlmFormUnsignedObject) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pUnsignedObject->setup( this, uiObjectId, uiMinVal, uiMaxVal,
								uiWidth, uiFormat, bDisplayOnly,
								uiBackColor, uiForeColor,
								uiRow, uiColumn)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pUnsignedObject->setValue( uiDefaultVal)))
	{
		goto Exit;
	}

	// Link the object into the form in its proper place.

	linkObjectInForm( pUnsignedObject, pPrevObject);

	// Display the object

	displayObject( pUnsignedObject);

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Add a signed numeric object to the form.
===========================================================================*/
RCODE FlmForm::addSignedObject(
	FLMUINT		uiObjectId,
	FLMINT		iDefaultVal,
	FLMINT		iMinVal,
	FLMINT		iMaxVal,
	FLMUINT		uiWidth,
	FLMUINT		uiFormat,
	FLMBOOL		bDisplayOnly,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn
	)
{
	RCODE						rc = FERR_OK;
	FlmFormObject *			pPrevObject;
	FlmFormSignedObject *	pSignedObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object ID is not already used.

	if ((pPrevObject = findObject( uiObjectId)) != NULL)
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Find the objects this object should be linked to.  This routine
	// also determines if there is an overlap.  If so, it will return
	// an error.

	if (RC_BAD( rc = getObjectLocation( uiWidth, uiRow, uiColumn,
							&pPrevObject)))
	{
		goto Exit;
	}

	// Create a signed number object and link into the form.

	if ((pSignedObject = new FlmFormSignedObject) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pSignedObject->setup( this, uiObjectId, iMinVal, iMaxVal,
								uiWidth, uiFormat, bDisplayOnly,
								uiBackColor, uiForeColor,
								uiRow, uiColumn)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pSignedObject->setValue( iDefaultVal)))
	{
		goto Exit;
	}

	// Link the object into the form in its proper place.

	linkObjectInForm( pSignedObject, pPrevObject);

	// Display the object

	displayObject( pSignedObject);

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Add a pulldown object to the form.
===========================================================================*/
RCODE FlmForm::addPulldownObject(
	FLMUINT		uiObjectId,
	FLMUINT		uiWidth,
	FLMUINT		uiHeight,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn,
	FLMBOOL		bAutoEnter
	)
{
	RCODE							rc = FERR_OK;
	FlmFormObject *			pPrevObject;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object ID is not already used.

	if ((pPrevObject = findObject( uiObjectId)) != NULL)
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Find the objects this object should be linked to.  This routine
	// also determines if there is an overlap.  If so, it will return
	// an error.

	if (RC_BAD( rc = getObjectLocation( uiWidth, uiRow, uiColumn,
							&pPrevObject)))
	{
		goto Exit;
	}

	// Create a signed number object and link into the form.

	if ((pPulldownObject = new FlmFormPulldownObject) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pPulldownObject->setup( this, uiObjectId,
								uiWidth, uiHeight,
								uiBackColor, uiForeColor,
								uiRow, uiColumn)))
	{
		goto Exit;
	}
	pPulldownObject->setEnterMode( bAutoEnter);

	// Link the object into the form in its proper place.

	linkObjectInForm( pPulldownObject, pPrevObject);

	// Display the object

	displayObject( pPulldownObject);

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Add a pulldown item to a pulldown object in the form.
===========================================================================*/
RCODE FlmForm::addPulldownItem(
	FLMUINT			uiObjectId,
	FLMUINT			uiItemId,
	const char *	pszDisplayValue,
	FLMUINT			uiShortcutKey)
{
	RCODE							rc = FERR_OK;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pPulldownObject =
			(FlmFormPulldownObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Add the item to the pulldown

	if (RC_BAD( rc = pPulldownObject->addItem( uiItemId, pszDisplayValue,
								uiShortcutKey)))
	{
		goto Exit;
	}

	// Display the object

	displayObject( pPulldownObject);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Remove a pulldown item from a pulldown object in the form.
===========================================================================*/
RCODE FlmForm::removePulldownItem(
	FLMUINT		uiObjectId,
	FLMUINT		uiItemId
	)
{
	RCODE							rc = FERR_OK;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pPulldownObject =
			(FlmFormPulldownObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Remove the item to the pulldown

	if (RC_BAD( rc = pPulldownObject->removeItem( uiItemId)))
	{
		goto Exit;
	}

	// Display the object

	displayObject( pPulldownObject);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Removes all pulldown items from a pulldown object in the form.
===========================================================================*/
RCODE FlmForm::clearPulldownItems(
	FLMUINT		uiObjectId
	)
{
	RCODE							rc = FERR_OK;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pPulldownObject =
			(FlmFormPulldownObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Clear all items from the pulldown

	if (RC_BAD( rc = pPulldownObject->clearItems()))
	{
		goto Exit;
	}

	// Display the object

	displayObject( pPulldownObject);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets a callback function for inserting new items into a pulldown
			object inside a form.
===========================================================================*/
RCODE FlmForm::setPulldownInsertCallback(
	FLMUINT			uiObjectId,
	INSERT_FUNC_p	pCallback,
	void *			pvAppData
	)
{
	RCODE							rc = FERR_OK;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pPulldownObject =
			(FlmFormPulldownObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Set the callback for the pulldown object.

	pPulldownObject->setPulldownInsertCallback( pCallback, pvAppData);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets flag indicating that we are to return all of the items
			in the list for this object on the screen, not just the current one.
===========================================================================*/
RCODE FlmForm::setPulldownReturnAll(
	FLMUINT	uiObjectId,
	FLMUINT	uiItemIdTag,
	FLMUINT	uiItemNameTag
	)
{
	RCODE							rc = FERR_OK;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pPulldownObject =
			(FlmFormPulldownObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	// Set the callback for the pulldown object.

	pPulldownObject->setPulldownReturnAll( uiItemIdTag, uiItemNameTag);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the pulldown object for a specific object on the form.
===========================================================================*/
FlmPulldownList * FlmForm::getPulldownObject(
	FLMUINT			uiObjectId
	)
{
	FlmPulldownList *			pPulldown = NULL;
	FlmFormPulldownObject *	pPulldownObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pPulldownObject =
			(FlmFormPulldownObject *)findObject( uiObjectId)) == NULL)
	{
		goto Exit;
	}

	// Get the callback for the pulldown object.

	pPulldown = pPulldownObject->getPulldownObject();
Exit:
	return( pPulldown);
}

/*===========================================================================
Desc:		Sets the mode for entering an object - does it happen
			automatically?  Used only for pulldown and record objects.
===========================================================================*/
RCODE FlmForm::setEnterMode(
	FLMUINT		uiObjectId,
	FLMBOOL		bAutoEnter
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	if (pObject->getObjectType() == FORM_PULLDOWN_OBJECT)
	{
		((FlmFormPulldownObject *)pObject)->setEnterMode( bAutoEnter);
	}
	else if (pObject->getObjectType() == FORM_RECORD_OBJECT)
	{
		((FlmFormRecordObject *)pObject)->setEnterMode( bAutoEnter);
	}
	else
	{
		flmAssert( 0);
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Adds a GEDCOM record(s) object - for editing GEDCOM.
===========================================================================*/
RCODE FlmForm::addRecordObject(
	FLMUINT			uiObjectId,
	const char *	pszTitle,
	NODE *			pDefaultRecords,
	FLMUINT			uiWidth,
	FLMUINT			uiBackColor,
	FLMUINT			uiForeColor,
	FLMUINT			uiRow,
	FLMUINT			uiColumn,
	FLMBOOL			bAutoEnter)
{
	RCODE							rc = FERR_OK;
	FlmFormObject *			pPrevObject;
	FlmFormRecordObject *	pRecordObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object ID is not already used.

	if ((pPrevObject = findObject( uiObjectId)) != NULL)
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Find the objects this object should be linked to.  This routine
	// also determines if there is an overlap.  If so, it will return
	// an error.

	if (RC_BAD( rc = getObjectLocation( uiWidth, uiRow, uiColumn,
							&pPrevObject)))
	{
		goto Exit;
	}

	// Create a signed number object and link into the form.

	if ((pRecordObject = new FlmFormRecordObject) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pRecordObject->setup( this, uiObjectId, pszTitle,
								uiWidth,
								uiBackColor, uiForeColor,
								uiRow, uiColumn)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pRecordObject->setValue( pDefaultRecords)))
	{
		goto Exit;
	}
	pRecordObject->setEnterMode( bAutoEnter);

	// Link the object into the form in its proper place.

	linkObjectInForm( pRecordObject, pPrevObject);

	// Display the object

	displayObject( pRecordObject);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets the database handle and container for a GEDCOM record(s) object.
===========================================================================*/
RCODE FlmForm::setRecordObjectDbAndCont(
	FLMUINT		uiObjectId,
	HFDB			hDb,
	FLMUINT		uiContainer
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object ID is not already used.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	((FlmFormRecordObject *)pObject)->setDb( hDb);
	((FlmFormRecordObject *)pObject)->setContainer( uiContainer);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Moves the form cursor to the first object on the form.
===========================================================================*/
FLMUINT FlmForm::firstObject(
	FLMBOOL	bSkipDisplayOnly,
	FLMBOOL	bSkipEditable,
	FLMBOOL	bRaiseExitEvent,
	FLMBOOL	bRaiseEnterEvent
	)
{
	FLMUINT				uiChar = 0;
	FlmFormObject *	pFirstObject;

	if (m_pFirstObject)
	{
		flmAssert( !bSkipDisplayOnly || !bSkipEditable);

		// Verify the current object.

		if (bRaiseExitEvent && m_pCurObject)
		{
			if (!verifyObject( bRaiseExitEvent))
				goto Exit;
		}
		pFirstObject = m_pFirstObject;

		// Skip any objects that are display only if the
		// bSkipDisplayOnly flag is set

		if (bSkipDisplayOnly)
		{
			while (pFirstObject->isDisplayOnly())
			{
				pFirstObject = pFirstObject->getNextObject();
				if (pFirstObject == m_pFirstObject)
				{
					break;
				}
			}
		}

		// Skip any objects that are NOT display only if the
		// bSkipEditable flag is set

		else if (bSkipEditable)
		{
			while (!pFirstObject->isDisplayOnly())
			{
				pFirstObject = pFirstObject->getNextObject();
				if (pFirstObject == m_pFirstObject)
				{
					break;
				}
			}
		}

		uiChar = changeFocus( pFirstObject, bRaiseEnterEvent, FALSE);
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Moves the form cursor to the last object on the form.
===========================================================================*/
FLMUINT FlmForm::lastObject(
	FLMBOOL	bSkipDisplayOnly,
	FLMBOOL	bSkipEditable,
	FLMBOOL	bRaiseExitEvent,
	FLMBOOL	bRaiseEnterEvent
	)
{
	FLMUINT				uiChar = 0;
	FlmFormObject *	pLastObject;

	if (m_pLastObject)
	{
		flmAssert( !bSkipDisplayOnly || !bSkipEditable);

		// Verify the current object.

		if (!verifyObject( bRaiseExitEvent))
			goto Exit;
		pLastObject = m_pLastObject;

		// Skip any objects that are display only if the
		// bSkipDisplayOnly flag is set

		if (bSkipDisplayOnly)
		{
			while (pLastObject->isDisplayOnly())
			{
				pLastObject = pLastObject->getPrevObject();
				if (pLastObject == m_pLastObject)
				{
					break;
				}
			}
		}

		// Skip any objects that are NOT display only if the
		// bSkipEditable flag is set

		else if (bSkipEditable)
		{
			while (!pLastObject->isDisplayOnly())
			{
				pLastObject = pLastObject->getPrevObject();
				if (pLastObject == m_pLastObject)
				{
					break;
				}
			}
		}

		uiChar = changeFocus( pLastObject, bRaiseEnterEvent, FALSE);
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Moves the form cursor to the next object on the form.
===========================================================================*/
FLMUINT FlmForm::nextObject(
	FLMBOOL	bSkipDisplayOnly,
	FLMBOOL	bSkipEditable,
	FLMBOOL	bRaiseExitEvent,
	FLMBOOL	bRaiseEnterEvent
	)
{
	FLMUINT				uiChar = 0;
	FlmFormObject *	pNextObject;

	if (m_pCurObject)
	{
		flmAssert( !bSkipDisplayOnly || !bSkipEditable);

		// Verify the current object.

		if (!verifyObject( bRaiseExitEvent))
			goto Exit;

		// Skip any objects that are display only if the
		// bSkipDisplayOnly flag is set

		pNextObject = m_pCurObject->getNextObject();
		if (bSkipDisplayOnly)
		{
			while (pNextObject != m_pCurObject &&
					 pNextObject->isDisplayOnly())
			{
				pNextObject = pNextObject->getNextObject();
			}
		}

		// Skip any objects that are NOT display only if the
		// bSkipEditable flag is set

		else if (bSkipEditable)
		{
			while (pNextObject != m_pCurObject &&
					 !pNextObject->isDisplayOnly())
			{
				pNextObject = pNextObject->getNextObject();
			}
		}

		uiChar = changeFocus( pNextObject, bRaiseEnterEvent, FALSE);
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Moves the form cursor to the previous object on the form.
===========================================================================*/
FLMUINT FlmForm::prevObject(
	FLMBOOL	bSkipDisplayOnly,
	FLMBOOL	bSkipEditable,
	FLMBOOL	bRaiseExitEvent,
	FLMBOOL	bRaiseEnterEvent
	)
{
	FLMUINT				uiChar = 0;
	FlmFormObject *	pPrevObject;

	if (m_pCurObject)
	{
		flmAssert( !bSkipDisplayOnly || !bSkipEditable);

		// Verify the current object.

		if (!verifyObject( bRaiseExitEvent))
			goto Exit;
		pPrevObject = m_pCurObject->getPrevObject();

		// Skip any objects that are display only if the
		// bSkipDisplayOnly flag is set

		if (bSkipDisplayOnly)
		{
			while (pPrevObject != m_pCurObject &&
					 pPrevObject->isDisplayOnly())
			{
				pPrevObject = pPrevObject->getPrevObject();
			}
		}

		// Skip any objects that are NOT display only if the
		// bSkipEditable flag is set

		else if (bSkipEditable)
		{
			while (pPrevObject != m_pCurObject &&
					 !pPrevObject->isDisplayOnly())
			{
				pPrevObject = pPrevObject->getPrevObject();
			}
		}

		uiChar = changeFocus( pPrevObject, bRaiseEnterEvent, FALSE);
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Moves the form cursor to the object on the form that is above the
			current object (if any).
===========================================================================*/
FLMUINT FlmForm::upObject(
	FLMBOOL	bSkipDisplayOnly,
	FLMBOOL	bSkipEditable,
	FLMBOOL	bRaiseExitEvent,
	FLMBOOL	bRaiseEnterEvent
	)
{
	FLMUINT				uiChar = 0;
	FlmFormObject *	pUpObject;
	FlmFormObject *	pPrevObject;
	FLMUINT				uiRow;
	FLMUINT				uiColumn;
	FLMUINT				uiClosestDistance;
	FLMUINT				uiDistance;

	if (m_pCurObject)
	{
		flmAssert( !bSkipDisplayOnly || !bSkipEditable);

		// Verify the current object.

		if (!verifyObject( bRaiseExitEvent))
			goto Exit;
		uiRow = m_pCurObject->getRow();
		uiColumn = m_pCurObject->getColumn();

		// Skip past all of the objects on the current row.

		pPrevObject = m_pCurObject->getPrevObject();
		while (pPrevObject != m_pCurObject &&
				 pPrevObject->getRow() == uiRow)
		{
			pPrevObject = pPrevObject->getPrevObject();
		}

		// Skip any objects that are display only if the
		// bSkipDisplayOnly flag is set

		if (bSkipDisplayOnly)
		{
			while (pPrevObject != m_pCurObject &&
					 pPrevObject->isDisplayOnly())
			{
				pPrevObject = pPrevObject->getPrevObject();
			}
		}

		// Skip any objects that are NOT display only if the
		// bSkipEditable flag is set

		else if (bSkipEditable)
		{
			while (pPrevObject != m_pCurObject &&
					 !pPrevObject->isDisplayOnly())
			{
				pPrevObject = pPrevObject->getPrevObject();
			}
		}

		// If there are no rows above this one, there is no up object.

		if (pPrevObject->getRow() < uiRow)
		{
			uiRow = pPrevObject->getRow();

			// Find the object on this current row whose column is closest to
			// our current object.

			uiClosestDistance = COLUMN_DIFF( uiColumn, pPrevObject->getColumn());
			pUpObject = pPrevObject;
			pPrevObject = pPrevObject->getPrevObject();
			while (pPrevObject->getRow() == uiRow)
			{
				if ((pPrevObject->isDisplayOnly() && !bSkipDisplayOnly) ||
					 (!pPrevObject->isDisplayOnly() && !bSkipEditable))
				{
					uiDistance = COLUMN_DIFF( uiColumn, pPrevObject->getColumn());
					if (uiDistance >= uiClosestDistance)
					{
						break;
					}
					pUpObject = pPrevObject;
					uiClosestDistance = uiDistance;
				}
				pPrevObject = pPrevObject->getPrevObject();
			}
			uiChar = changeFocus( pUpObject, bRaiseEnterEvent, FALSE);
		}
		else if (m_uiTopFormRow)
		{

			// Scroll up one line of the form is not displaying the top line
			// on the screen.

			m_uiTopFormRow--;
			m_uiEditRow++;
			refresh();
		}
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Moves the form cursor to the object on the form that is below the
			current object (if any).
===========================================================================*/
FLMUINT FlmForm::downObject(
	FLMBOOL	bSkipDisplayOnly,
	FLMBOOL	bSkipEditable,
	FLMBOOL	bRaiseExitEvent,
	FLMBOOL	bRaiseEnterEvent
	)
{
	FLMUINT				uiChar = 0;
	FlmFormObject *	pDownObject;
	FlmFormObject *	pNextObject;
	FLMUINT				uiRow;
	FLMUINT				uiColumn;
	FLMUINT				uiClosestDistance;
	FLMUINT				uiDistance;

	if (m_pCurObject)
	{
		flmAssert( !bSkipDisplayOnly || !bSkipEditable);

		// Verify the current object.

		if (!verifyObject( bRaiseExitEvent))
			goto Exit;

		uiRow = m_pCurObject->getRow();
		uiColumn = m_pCurObject->getColumn();

		// Skip past all of the objects on the current row.

		pNextObject = m_pCurObject->getNextObject();
		while (pNextObject != m_pCurObject &&
				 pNextObject->getRow() == uiRow)
		{
			pNextObject = pNextObject->getNextObject();
		}

		// Skip any objects that are display only if the
		// bSkipDisplayOnly flag is set

		if (bSkipDisplayOnly)
		{
			while (pNextObject != m_pCurObject &&
					 pNextObject->isDisplayOnly())
			{
				pNextObject = pNextObject->getNextObject();
			}
		}

		// Skip any objects that are NOT display only if the
		// bSkipEditable flag is set

		else if (bSkipEditable)
		{
			while (pNextObject != m_pCurObject &&
					 !pNextObject->isDisplayOnly())
			{
				pNextObject = pNextObject->getNextObject();
			}
		}

		// If there are no rows below this one, there is no down object.

		if (pNextObject->getRow() > uiRow)
		{
			uiRow = pNextObject->getRow();

			// Find the object on this current row whose column is closest to
			// our current object.

			uiClosestDistance = COLUMN_DIFF( uiColumn, pNextObject->getColumn());
			pDownObject = pNextObject;
			pNextObject = pNextObject->getNextObject();
			while (pNextObject->getRow() == uiRow)
			{
				if ((pNextObject->isDisplayOnly() && !bSkipDisplayOnly) ||
					 (!pNextObject->isDisplayOnly() && !bSkipEditable))
				{
					uiDistance = COLUMN_DIFF( uiColumn, pNextObject->getColumn());
					if (uiDistance >= uiClosestDistance)
					{
						break;
					}
					pDownObject = pNextObject;
					uiClosestDistance = uiDistance;
				}
				pNextObject = pNextObject->getNextObject();
			}
			uiChar = changeFocus( pDownObject, bRaiseEnterEvent, FALSE);
		}
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Populates any registered return addresses with data from the form.
===========================================================================*/
RCODE FlmForm::getAllReturnData( void)
{
	RCODE					rc = FERR_OK;
	RCODE					tmpRc;
	FlmFormObject *	pObject = m_pFirstObject;

	for (;;)
	{

		// Don't quit on first bad one. Save RC and continue.

		if (RC_BAD( tmpRc = pObject->populateReturnAddress()))
		{
			rc = tmpRc;
		}
		pObject = pObject->getNextObject();
		if (pObject == m_pFirstObject)
		{
			break;
		}
	}
	return( rc);
}

/*===========================================================================
Desc:		Populates any registered return paths in a GEDCOM tree with data
			from the form.
===========================================================================*/
RCODE FlmForm::getAllReturnDataToTree(
	POOL *		pPool,
	NODE *		pTree)
{
	RCODE					rc = FERR_OK;
	RCODE					tmpRc;
	FlmFormObject *	pObject = m_pFirstObject;

	for (;;)
	{

		// Don't quit on first bad one. Save RC and continue.

		if (RC_BAD( tmpRc = pObject->populateReturnPath( pPool, pTree)))
		{
			rc = tmpRc;
		}
		pObject = pObject->getNextObject();
		if (pObject == m_pFirstObject)
		{
			break;
		}
	}
	return( rc);
}

/*===========================================================================
Desc:		Executes a form - handles all user input in the form.  Return value
			is the last character that was pressed.
===========================================================================*/
FLMUINT FlmForm::interact(
	FLMBOOL *	pbValuesChanged,
	FLMUINT *	puiCurrObjectId)
{
	FLMUINT	uiChar;

	m_bValuesChanged = FALSE;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		uiChar = 0;
		goto Exit;
	}

	// Set focus to the first object on the form.

	m_bInteracting = TRUE;
	m_pCurObject = NULL;
	firstObject( TRUE, FALSE, FALSE, TRUE);
	beep( NULL, (const char *)m_pCurObject);

	// Loop forever getting input.

	for (;;)
	{

		// See if we have been told to exit.

		if ((m_pThread) &&
			 (m_pThread->getShutdownFlag()))
		{
			uiChar = 0;
			goto Exit;
		}

		// Need to refresh the input cursor, in case it was changed by
		// a callback function.

		displayInputCursor();
		if (FTXWinTestKB( m_pDisplayWindow) == FTXRC_SUCCESS)
		{
			FTXWinInputChar( m_pDisplayWindow, &uiChar);

			// Clear error message out of status line.

			beep( NULL, (const char *)m_pCurObject);
			if (m_pEventCB && m_bGetKeyStrokes)
			{
				if (!m_pEventCB( FORM_EVENT_KEY_STROKE, this, m_pCurObject, uiChar,
								&uiChar, m_pvAppData))
				{

					// Must verify current object, unless callback changes the
					// character to ESCAPE.

					if (uiChar == WPK_ESCAPE || verifyObject( TRUE))
					{
						goto Exit;
					}
					else
					{
						continue;
					}
				}
			}
Handle_Char:
			switch( uiChar)
			{
				case WPK_DOWN:
				case WPK_CTRL_N:
					if ((uiChar = downObject( TRUE, FALSE, TRUE, TRUE)) != 0)
						goto Handle_Char;
					break;
				case WPK_UP:
				case WPK_CTRL_P:
					if ((uiChar = upObject( TRUE, FALSE, TRUE, TRUE)) != 0)
						goto Handle_Char;
					break;
				case WPK_PGUP:
				case WPK_CTRL_DOWN:
				case WPK_CTRL_UP:
				case WPK_PGDN:
					break;
				case WPK_CTRL_HOME:
					if ((uiChar = firstObject( TRUE, FALSE, TRUE, TRUE)) != 0)
						goto Handle_Char;
					break;
				case WPK_CTRL_END:
					if ((uiChar = lastObject( TRUE, FALSE, TRUE, TRUE)) != 0)
						goto Handle_Char;
					break;
				case WPK_CTRL_D:
					if (m_pCurObject->getObjectType() == FORM_PULLDOWN_OBJECT)
					{
Edit_Pulldown_Error:
						beep( "Cannot edit: press ENTER to select from list");
					}
					else if (m_pCurObject->getObjectType() == FORM_RECORD_OBJECT)
					{
Edit_Record_Error:
						beep( "Press ENTER to edit record");
					}
					else
					{
						clearLine();
					}
					break;
				case WPK_DELETE:
					if (m_pCurObject->getObjectType() == FORM_PULLDOWN_OBJECT)
					{
						goto Edit_Pulldown_Error;
					}
					else if (m_pCurObject->getObjectType() == FORM_RECORD_OBJECT)
					{
						goto Edit_Record_Error;
					}
					else
					{
						deleteChar();
					}
					break;
				case WPK_HOME:
					cursorHome();
					break;
				case WPK_END:
					cursorEnd();
					break;
				case WPK_INSERT:
					if (m_bInsertMode)
					{
						m_bInsertMode = FALSE;
						FTXWinSetCursorType( m_pDisplayWindow,
							WPS_CURSOR_VISIBLE | WPS_CURSOR_BLOCK);
					}
					else
					{
						m_bInsertMode = TRUE;
						FTXWinSetCursorType( m_pDisplayWindow,
							WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE);
					}
					break;
				case WPK_RIGHT:
					if ((uiChar = cursorRight()) != 0)
						goto Handle_Char;
					break;
				case WPK_LEFT:
					if ((uiChar = cursorLeft()) != 0)
						goto Handle_Char;
					break;
				case WPK_CTRL_RIGHT:
				case WPK_CTRL_LEFT:
				case WPK_CTRL_C:
				case WPK_CTRL_X:
				case WPK_CTRL_V:
					break;
				case WPK_BACKSPACE:
					if (m_pCurObject->getObjectType() == FORM_PULLDOWN_OBJECT)
					{
						goto Edit_Pulldown_Error;
					}
					else if (m_pCurObject->getObjectType() == FORM_RECORD_OBJECT)
					{
						goto Edit_Record_Error;
					}
					else
					{
						backspaceChar();
					}
					break;
				case WPK_ENTER:
					if (m_pCurObject->getObjectType() == FORM_PULLDOWN_OBJECT)
					{
						FlmFormPulldownObject *	pPulldownObject =
														(FlmFormPulldownObject *)m_pCurObject;
						FLMUINT						uiOldItemId;
						FLMUINT						uiNewItemId;

						pPulldownObject->getCurrentItem( &uiOldItemId);
						uiChar = pPulldownObject->select();
						pPulldownObject->getCurrentItem( &uiNewItemId);
						if (uiOldItemId != uiNewItemId)
						{
							m_bValuesChanged = TRUE;
						}
						if (uiChar)
							goto Handle_Char;
					}
					else if (m_pCurObject->getObjectType() == FORM_RECORD_OBJECT)
					{
						FLMBOOL	bChanged;

						uiChar = ((FlmFormRecordObject *)m_pCurObject)->edit( &bChanged);
						if (bChanged)
						{
							m_bValuesChanged = TRUE;
						}
						if (uiChar)
							goto Handle_Char;
					}
					else
					{
						if ((uiChar = nextObject( TRUE, FALSE, TRUE, TRUE)) != 0)
							goto Handle_Char;
					}
					break;
				case WPK_TAB:
					if ((uiChar = nextObject( TRUE, FALSE, TRUE, TRUE)) != 0)
						goto Handle_Char;
					break;
				case WPK_STAB:
					if ((uiChar = prevObject( TRUE, FALSE, TRUE, TRUE)) != 0)
						goto Handle_Char;
					break;
				case WPK_ALT_H:	// Help on form
				case WPK_CTRL_H:	// Help on field in form.
					break;
				case WPK_ESCAPE:
				case WPK_ALT_Q:
				case WPK_F1:		// Done with form - submit.

					// Verify the current object if not ESCAPE character

					if (uiChar == WPK_ESCAPE || verifyObject( TRUE))
					{
						if (m_pEventCB && m_bGetKeyStrokes)
						{
							if (m_pEventCB( FORM_EVENT_EXIT_FORM, this,
											m_pCurObject, uiChar,
											&uiChar, m_pvAppData))
							{
								goto Exit;
							}
						}
						else
						{
							goto Exit;
						}
					}
					break;
				case 0:
					break;
				default:
					if ((uiChar >= ' ') && (uiChar <= 0x7F))
					{
						if (m_pCurObject->getObjectType() == FORM_PULLDOWN_OBJECT)
						{
							goto Edit_Pulldown_Error;
						}
						else if (m_pCurObject->getObjectType() == FORM_RECORD_OBJECT)
						{
							goto Edit_Record_Error;
						}
						else
						{
							insertChar( uiChar);
						}
					}
					else
					{
						beep( "Keystroke not recognized");
					}
					break;
			}
		}
		else
		{
			f_sleep( 1);
		}
	}

Exit:
	if (puiCurrObjectId && m_pCurObject)
	{
		*puiCurrObjectId = m_pCurObject->getObjectId();
	}
	*pbValuesChanged = m_bValuesChanged;
	m_bInteracting = FALSE;
	return( uiChar);
}

/*===========================================================================
Desc:		Retrieves various kinds of information about an object on a form.
===========================================================================*/
RCODE FlmForm::getObjectInfo(
	FLMUINT				uiObjectId,
	FormObjectType *	peObjectType,
	FLMUINT *			puiBackColor,
	FLMUINT *			puiForeColor,
	FLMUINT *			puiRow,
	FLMUINT *			puiCol,
	FLMUINT *			puiWidth,
	void *				pvMin,
	void *				pvMax,
	FLMBOOL *			pbDisplayOnly
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if (peObjectType)
	{
		*peObjectType = pObject->getObjectType();
	}
	if (puiBackColor)
	{
		*puiBackColor = pObject->getBackColor();
	}
	if (puiForeColor)
	{
		*puiForeColor = pObject->getForeColor();
	}
	if (puiRow)
	{
		*puiRow = pObject->getRow();
	}
	if (puiCol)
	{
		*puiCol = pObject->getColumn();
	}
	if (puiWidth)
	{
		*puiWidth = pObject->getWidth();
	}
	if (pvMin)
	{
		switch (pObject->getObjectType())
		{
			case FORM_UNSIGNED_OBJECT:
				*((FLMUINT *)pvMin) = ((FlmFormUnsignedObject *)pObject)->getMin();
				break;
			case FORM_SIGNED_OBJECT:
				*((FLMINT *)pvMin) = ((FlmFormSignedObject *)pObject)->getMin();
				break;
			default:
				*((FLMUINT *)pvMin) = 0;
				break;
		}
	}
	if (pvMax)
	{
		switch (pObject->getObjectType())
		{
			case FORM_UNSIGNED_OBJECT:
				*((FLMUINT *)pvMax) = ((FlmFormUnsignedObject *)pObject)->getMax();
				break;
			case FORM_SIGNED_OBJECT:
				*((FLMINT *)pvMax) = ((FlmFormSignedObject *)pObject)->getMax();
				break;
			default:
				*((FLMUINT *)pvMax) = pObject->getMaxEditChars();
				break;
		}
	}
	if (pbDisplayOnly)
	{
		*pbDisplayOnly = pObject->isDisplayOnly();
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Retrieve a text value from the form
===========================================================================*/
RCODE FlmForm::getTextVal(
	FLMUINT		uiObjectId,
	FLMUINT *	puiLen,
	char *		pszValRV)
{
	RCODE						rc = FERR_OK;
	FlmFormTextObject *	pTextObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pTextObject =
			(FlmFormTextObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if (pTextObject->getObjectType() != FORM_TEXT_OBJECT)
	{
		flmAssert( 0);
	}
	if (RC_BAD( rc = pTextObject->getValue( puiLen, pszValRV)))
	{
		goto Exit;
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Retrieve an unsigned numeric value from the form
===========================================================================*/
RCODE FlmForm::getUnsignedVal(
	FLMUINT		uiObjectId,
	FLMUINT *	puiValRV
	)
{
	RCODE					rc = FERR_OK;
	FlmFormObject *	pObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pObject = findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if (pObject->getObjectType() == FORM_UNSIGNED_OBJECT)
	{
		if (RC_BAD( rc = ((FlmFormUnsignedObject *)pObject)->getValue( puiValRV)))
		{
			goto Exit;
		}
	}
	else if (pObject->getObjectType() == FORM_PULLDOWN_OBJECT)
	{
		if (RC_BAD( rc = ((FlmFormPulldownObject *)pObject)->getCurrentItem( puiValRV)))
		{
			goto Exit;
		}
	}
	else
	{
		flmAssert( 0);
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Retrieve a signed numeric value from the form
===========================================================================*/
RCODE FlmForm::getSignedVal(
	FLMUINT		uiObjectId,
	FLMINT *		piValRV
	)
{
	RCODE						rc = FERR_OK;
	FlmFormSignedObject *	pSignedObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pSignedObject =
			(FlmFormSignedObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if (pSignedObject->getObjectType() != FORM_SIGNED_OBJECT)
	{
		flmAssert( 0);
	}
	if (RC_BAD( rc = pSignedObject->getValue( piValRV)))
	{
		goto Exit;
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Retrieve a GEDCOM record value from the form
===========================================================================*/
RCODE FlmForm::getRecordVal(
	FLMUINT		uiObjectId,
	NODE * *		ppRecord,
	POOL *		pPool)
{
	RCODE							rc = FERR_OK;
	FlmFormRecordObject *	pRecordObject;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Make sure the object has been defined.

	if ((pRecordObject =
			(FlmFormRecordObject *)findObject( uiObjectId)) == NULL)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	if (pRecordObject->getObjectType() != FORM_RECORD_OBJECT)
	{
		flmAssert( 0);
	}
	if (RC_BAD( rc = pRecordObject->getValue( ppRecord, pPool)))
	{
		goto Exit;
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Verifies the current object and sets its value if it is verified.
			Otherwise, returns FALSE.
===========================================================================*/
FLMBOOL FlmForm::verifyObject(
	FLMBOOL	bRaiseExitEvent
	)
{
	FLMBOOL	bOk = TRUE;
	FLMUINT	uiVal;
	FLMINT	iVal;

	// Verify the input - make sure it is null terminated first.

	m_pszEditBuf [m_uiNumCharsEntered] = 0;
	switch (m_pCurObject->getObjectType())
	{
		case FORM_TEXT_OBJECT:
			((FlmFormTextObject *)m_pCurObject)->setValue( m_pszEditBuf);
			break;
		case FORM_UNSIGNED_OBJECT:
			if (!((FlmFormUnsignedObject *)m_pCurObject)->verifyValue( m_pszEditBuf,
																			&uiVal))
			{
				bOk = FALSE;
				goto Exit;
			}
			((FlmFormUnsignedObject *)m_pCurObject)->setValue( uiVal);
			break;
		case FORM_SIGNED_OBJECT:
			if (!((FlmFormSignedObject *)m_pCurObject)->verifyValue( m_pszEditBuf,
																			&iVal))
			{
				bOk = FALSE;
				goto Exit;
			}
			((FlmFormSignedObject *)m_pCurObject)->setValue( iVal);
			break;
		case FORM_PULLDOWN_OBJECT:
		case FORM_RECORD_OBJECT:
			break;
	}

	// Do exit callback

	if (m_pEventCB && bRaiseExitEvent)
	{
		bOk = m_pEventCB( FORM_EVENT_EXIT_OBJECT, this, m_pCurObject, 0,
						NULL, m_pvAppData);
	}
Exit:
	return( bOk);
}

/*===========================================================================
Desc:		Moves the form cursor to the first object on the form.
===========================================================================*/
FLMUINT FlmForm::changeFocus(
	FlmFormObject *	pNewObject,
	FLMBOOL				bRaiseEnterEvent,
	FLMBOOL				bForceChange)
{
	FLMUINT		uiChar = 0;
	char *		pszTmp;
	FLMUINT		uiMaxEditChars;

	// Form must be initialized

	if (!m_pDisplayWindow)
	{
		goto Exit;
	}

	// Go to the new object on the form.

	if ((pNewObject) && ((m_pCurObject != pNewObject) || (bForceChange)))
	{

		// Redisplay the current object.

		if (m_pCurObject)
		{
			displayObject( m_pCurObject);
		}

		// Determine if the object is visible.

		m_pCurObject = pNewObject;
		if (!isVisible( pNewObject))
		{

			// Object not visible, reposition the form and redisplay the entire
			// form.

			// If the object will fit in the first page of the form, set the
			// top form of the row to zero.

			if (pNewObject->getRow() < m_uiTopFormRow)
			{
				m_uiTopFormRow = pNewObject->getRow();
			}
			else
			{
				if (pNewObject->getRow() > m_uiRows - 1)
				{
					m_uiTopFormRow = pNewObject->getRow() - m_uiRows + 1;
				}
				else
				{
					flmAssert( 0);				// Should never be able to get here.
					m_uiTopFormRow = 0;
				}
			}
			refresh();
		}

		// Do the entry callback

		if (m_pEventCB && bRaiseEnterEvent)
		{
			(void)m_pEventCB( FORM_EVENT_ENTER_OBJECT, this, m_pCurObject, 0,
							NULL, m_pvAppData);
		}

		// Set up to edit the new object.

		uiMaxEditChars = pNewObject->getMaxEditChars();
		if (m_uiEditBufSize < uiMaxEditChars + 1)
		{
			// Allocate a new buffer.

			if( RC_BAD( f_alloc( uiMaxEditChars + 1, &pszTmp)))
			{
				goto Exit;
			}

			if (m_pszEditBuf)
			{
				f_free( &m_pszEditBuf);
				m_pszEditBuf = NULL;
				m_uiEditBufSize = 0;
			}
			m_pszEditBuf = pszTmp;
			m_uiEditBufSize = uiMaxEditChars + 1;
		}
		m_uiMaxCharsToEnter = uiMaxEditChars;
		pNewObject->formatEditBuffer( m_pszEditBuf);
		m_uiNumCharsEntered = f_strlen( m_pszEditBuf);
		m_uiEditWidth = pNewObject->getWidth();
		m_uiEditBufPos = 0;
		m_uiEditBufLeftColPos = 0;
		m_uiEditColumn = pNewObject->getColumn();
		m_uiEditRow = pNewObject->getRow() - m_uiTopFormRow;
		m_bInsertMode = TRUE;
		m_pCurObject = pNewObject;
		m_bShowingHelpStatus = FALSE;
		beep( NULL, (const char *)m_pCurObject);
		FTXWinSetCursorType( m_pDisplayWindow,
							WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE);

		// Set the background and foreground color

		refreshLine( m_uiEditBufPos, m_uiEditWidth);
		if ((m_pCurObject->getObjectType() == FORM_PULLDOWN_OBJECT) &&
			 (((FlmFormPulldownObject *)m_pCurObject)->isAutoEnterMode()))
		{
			uiChar = WPK_ENTER;
		}
		else if ((m_pCurObject->getObjectType() == FORM_RECORD_OBJECT) &&
			 (((FlmFormRecordObject *)m_pCurObject)->isAutoEnterMode()))
		{
			uiChar = WPK_ENTER;
		}
	}
Exit:
	return( uiChar);
}

/*===========================================================================
Desc:		Finds an object by tag number.
===========================================================================*/
FlmFormObject * FlmForm::findObject(
	FLMUINT	uiObjectId
	)
{
	FlmFormObject *	pObject;

	// Follow the linked list of objects until we find the one that matches
	// the tag we are looking for.

	if ((pObject = m_pFirstObject) != NULL)
	{
		for (;;)
		{
			if (pObject->getObjectId() == uiObjectId)
				break;
			pObject = pObject->getNextObject();

			// break when we have circled all the way around.

			if (pObject == m_pFirstObject)
			{
				pObject = NULL;
				break;
			}
		}
	}

	return( pObject);
}

/*===========================================================================
Desc:		Determines if an object is currently visible.
===========================================================================*/
FLMBOOL FlmForm::isVisible(
	FlmFormObject *	pObject
	)
{
	return( (FLMBOOL)((pObject->getRow() >= m_uiTopFormRow &&
							 pObject->getRow() <=
								m_uiTopFormRow + m_uiRows - 1)
							? (FLMBOOL)TRUE
							: (FLMBOOL)FALSE));
}

/*===========================================================================
Desc:		Displays an object - if it is visible.
===========================================================================*/
void FlmForm::displayObject(
	FlmFormObject *	pObject
	)
{
	// If the object is visible, tell it to display itself.

	if (isVisible( pObject))
	{
		pObject->display( pObject->getRow() - m_uiTopFormRow,
								pObject->getColumn());
	}
}

/*===========================================================================
Desc:		Determines where an object should be linked into the form.
			Also determines if the object overlaps any other objects.  If so,
			an error will be returned.
===========================================================================*/
RCODE FlmForm::getObjectLocation(
	FLMUINT				uiWidth,
	FLMUINT				uiRow,
	FLMUINT				uiColumn,
	FlmFormObject **	ppPrevObject
	)
{
	RCODE				rc = FERR_OK;
	FlmFormObject *	pPrevObject;
	FlmFormObject *	pNextObject;

	// Follow linked list of objects until we find the one that is just
	// before this one.

	pPrevObject = NULL;
	pNextObject = m_pFirstObject;
	while (pNextObject && pNextObject->getRow() < uiRow)
	{
		pPrevObject = pNextObject;
		if ((pNextObject = pNextObject->getNextObject()) ==
							m_pFirstObject)
		{
			pNextObject = NULL;
			break;
		}
	}

	// If the next object is on the same row as the new
	// object, search down the row until we hit an
	// object whose column is > the new object's column.

	while (pNextObject &&
			 pNextObject->getRow() == uiRow &&
			 pNextObject->getColumn() < uiColumn)
	{
		pPrevObject = pNextObject;
		if ((pNextObject = pNextObject->getNextObject()) ==
									m_pFirstObject)
		{
			pNextObject = NULL;
			break;
		}
	}

	// If the previous object is on the same row, see if there
	// is any overlap between it, the new object, and the next
	// object.

	if (uiWidth)
	{
		if (pPrevObject &&
			 pPrevObject->getWidth() &&
			 pPrevObject->getRow() == uiRow)
		{
			if (pPrevObject->getColumn() + pPrevObject->getWidth() - 1 >=
						uiColumn)
			{
				flmAssert( 0);
				rc = RC_SET( FERR_FAILURE);
				goto Exit;
			}
		}
		if (pNextObject &&
			 pNextObject->getWidth() &&
			 pNextObject->getRow() == uiRow)
		{
			if (uiColumn + uiWidth - 1 >= pNextObject->getColumn())
			{
				flmAssert( 0);
				rc = RC_SET( FERR_FAILURE);
				goto Exit;
			}
		}
	}

	// Make sure the column plus its width will fit in the form window.

	if (uiColumn + uiWidth > m_uiCols)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	*ppPrevObject = pPrevObject;

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Links an object into a form - after the passed in prev object.
===========================================================================*/
void FlmForm::linkObjectInForm(
	FlmFormObject *	pNewObject,
	FlmFormObject *	pPrevObject
	)
{
	FlmFormObject *	pNextObject;

	if (!m_pFirstObject)
	{
		pNextObject = pPrevObject =
		m_pFirstObject = m_pLastObject = pNewObject;
	}
	else if (!pPrevObject)
	{
		pPrevObject = m_pLastObject;
		pNextObject = m_pFirstObject;
		m_pFirstObject = pNewObject;
	}
	else
	{
		if ((pNextObject = pPrevObject->getNextObject()) ==
				m_pFirstObject)
		{
			m_pLastObject = pNewObject;
		}
	}
	pNewObject->setNextObject( pNextObject);
	pNewObject->setPrevObject( pPrevObject);
	pNextObject->setPrevObject( pNewObject);
	pPrevObject->setNextObject( pNewObject);
}

/*===========================================================================
Desc:		Refresh the portion of the line specified.
===========================================================================*/
void FlmForm::refreshLine(
	FLMUINT	uiPos,
	FLMUINT	uiChars
	)
{
	FLMBYTE	ucSaveByte;

	// Use foreground color as background color, and vice versa when
	// editing.

	FTXWinSetBackFore( m_pDisplayWindow,
		m_bMonochrome ? WPS_WHITE : m_pCurObject->getForeColor(),
		m_bMonochrome ? WPS_BLACK : m_pCurObject->getBackColor());

	// If there are fewer characters in the edit buffer than
	// what we are being asked to display, we need to blank
	// out whatever portion comes after the end of line.

	if (uiChars > m_uiNumCharsEntered - uiPos)
	{
		if (uiPos < m_uiNumCharsEntered)
		{
			m_pszEditBuf [m_uiNumCharsEntered] = 0;
			FTXWinPrintStrXY( m_pDisplayWindow, &m_pszEditBuf [uiPos],
				m_uiEditColumn + uiPos - m_uiEditBufLeftColPos, m_uiEditRow);
			uiChars -= (m_uiNumCharsEntered - uiPos);
		}
		else
		{
			FTXWinSetCursorPos( m_pDisplayWindow,
				m_uiEditColumn + uiPos - m_uiEditBufLeftColPos,
				m_uiEditRow);
		}

		// Blank out the rest of the box up to the number of
		// characters specified.

		while (uiChars)
		{
			FTXWinPrintChar( m_pDisplayWindow, ' ');
			uiChars--;
		}
	}
	else
	{
		ucSaveByte = m_pszEditBuf [uiPos + uiChars];
		m_pszEditBuf [uiPos + uiChars] = 0;
		FTXWinPrintStrXY( m_pDisplayWindow, &m_pszEditBuf [uiPos],
			m_uiEditColumn + uiPos - m_uiEditBufLeftColPos, m_uiEditRow);
		m_pszEditBuf [uiPos + uiChars] = ucSaveByte;
	}
}

/*===========================================================================
Desc:		Move cursor to the beginning of the line
===========================================================================*/
void FlmForm::cursorHome( void)
{
	if (m_uiEditBufPos)
	{
		m_uiEditBufPos = 0;
		if (m_uiEditBufLeftColPos)
		{
			m_uiEditBufLeftColPos = 0;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
	}
	else
	{
		beep( "Already at beginning of line");
	}
}

/*===========================================================================
Desc:		Move cursor to the end of the line
===========================================================================*/
void FlmForm::cursorEnd( void)
{
	if (m_uiEditBufPos < m_uiNumCharsEntered)
	{
		m_uiEditBufPos = m_uiNumCharsEntered;
		if (m_uiEditBufLeftColPos + m_uiEditWidth <= m_uiNumCharsEntered)
		{
			m_uiEditBufLeftColPos = m_uiNumCharsEntered - m_uiEditWidth + 1;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
	}
	else
	{
		beep( "Already at end of line");
	}
}

/*===========================================================================
Desc:		Move cursor right one character.
===========================================================================*/
FLMUINT FlmForm::cursorRight( void)
{
	FLMUINT	uiChar = 0;

	if ((m_uiEditBufPos < m_uiNumCharsEntered) &&
		 (m_pCurObject->getObjectType() != FORM_PULLDOWN_OBJECT) &&
		 (m_pCurObject->getObjectType() != FORM_RECORD_OBJECT))
	{
		m_uiEditBufPos++;
		if (m_uiEditBufLeftColPos + m_uiEditWidth - 1 < m_uiEditBufPos)
		{
			m_uiEditBufLeftColPos++;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
	}
	else
	{
		uiChar = nextObject();
	}
	return( uiChar);
}

/*===========================================================================
Desc:		Move cursor left one character.
===========================================================================*/
FLMUINT FlmForm::cursorLeft( void)
{
	FLMUINT	uiChar = 0;

	if ((m_uiEditBufPos) &&
		 (m_pCurObject->getObjectType() != FORM_PULLDOWN_OBJECT) &&
		 (m_pCurObject->getObjectType() != FORM_RECORD_OBJECT))
	{
		m_uiEditBufPos--;
		if (m_uiEditBufLeftColPos > m_uiEditBufPos)
		{
			m_uiEditBufLeftColPos--;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
	}
	else
	{
		uiChar = prevObject();
	}
	return( uiChar);
}

/*===========================================================================
Desc:		Delete the character the cursor is positioned on.
===========================================================================*/
void FlmForm::deleteChar( void)
{
	if (m_uiEditBufPos < m_uiNumCharsEntered)
	{
		if (m_uiEditBufPos < m_uiNumCharsEntered - 1)
		{
			f_memmove( &m_pszEditBuf [m_uiEditBufPos],
				&m_pszEditBuf [m_uiEditBufPos + 1],
				m_uiNumCharsEntered - m_uiEditBufPos - 1);
		}
		m_uiNumCharsEntered--;
		if (m_uiEditWidth <= m_uiNumCharsEntered - m_uiEditBufLeftColPos + 1)
		{
			refreshLine( m_uiEditBufPos,
				m_uiEditWidth - (m_uiEditBufPos - m_uiEditBufLeftColPos));
		}
		else
		{
			refreshLine( m_uiEditBufPos,
				m_uiNumCharsEntered - m_uiEditBufPos + 1);
		}
		m_bValuesChanged = TRUE;
	}
	else
	{
		beep( "At end of line - nothing to delete");
	}
}

/*===========================================================================
Desc:		Insert a character where the cursor is positioned.
===========================================================================*/
void FlmForm::insertChar(
	FLMUINT	uiChar
	)
{

	// See if the buffer is full.

	if ((m_uiNumCharsEntered == m_uiMaxCharsToEnter) &&
		 ((m_bInsertMode) || (m_uiEditBufPos == m_uiNumCharsEntered)))
	{
		beep( "Maximum number of characters have been entered");
		return;
	}

	if (m_uiEditBufPos == m_uiNumCharsEntered)
	{
		m_pszEditBuf [m_uiNumCharsEntered++] = (FLMBYTE)uiChar;
		m_uiEditBufPos++;
		if (m_uiNumCharsEntered == m_uiMaxCharsToEnter)
		{
			refreshLine( m_uiEditBufPos - 1, 1);
		}
		else if (m_uiEditBufPos - m_uiEditBufLeftColPos >= m_uiEditWidth)
		{
			m_uiEditBufLeftColPos++;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
		else
		{
			refreshLine( m_uiEditBufPos - 1, 1);
		}
	}
	else if (m_bInsertMode)
	{
		f_memmove( &m_pszEditBuf [m_uiEditBufPos + 1],
						&m_pszEditBuf [m_uiEditBufPos],
						m_uiNumCharsEntered - m_uiEditBufPos);
		m_uiNumCharsEntered++;
		m_pszEditBuf [m_uiEditBufPos++] = (FLMBYTE)uiChar;
		if (m_uiEditBufPos - m_uiEditBufLeftColPos >= m_uiEditWidth)
		{
			m_uiEditBufLeftColPos++;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
		else if (m_uiEditWidth <= m_uiNumCharsEntered - m_uiEditBufLeftColPos + 1)
		{
			refreshLine( m_uiEditBufPos - 1,
				m_uiEditWidth - (m_uiEditBufPos - 1 - m_uiEditBufLeftColPos));
		}
		else
		{
			refreshLine( m_uiEditBufPos - 1,
				m_uiNumCharsEntered - (m_uiEditBufPos - 1));
		}
	}
	else
	{
		m_pszEditBuf [m_uiEditBufPos++] = (FLMBYTE)uiChar;
		if (m_uiEditBufPos - m_uiEditBufLeftColPos >= m_uiEditWidth)
		{
			m_uiEditBufLeftColPos++;
			refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
		}
		else
		{
			refreshLine( m_uiEditBufPos - 1, 1);
		}
	}
	m_bValuesChanged = TRUE;
}

/*===========================================================================
Desc:		Delete the character to the left of the cursor.
===========================================================================*/
void FlmForm::backspaceChar( void)
{
	if (m_uiEditBufPos)
	{
		m_uiEditBufPos--;
		if (m_uiEditBufPos < m_uiNumCharsEntered - 1)
		{
			f_memmove( &m_pszEditBuf [m_uiEditBufPos],
				&m_pszEditBuf [m_uiEditBufPos + 1],
				m_uiNumCharsEntered - m_uiEditBufPos - 1);
		}
		m_uiNumCharsEntered--;
		if (m_uiEditBufPos <= m_uiEditBufLeftColPos &&
			 m_uiEditBufLeftColPos)
		{
			if (m_uiEditBufPos)
			{
				m_uiEditBufLeftColPos = m_uiEditBufPos - 1;
			}
			else
			{
				m_uiEditBufLeftColPos = 0;
			}
			if (m_uiEditWidth > m_uiNumCharsEntered - m_uiEditBufLeftColPos + 1)
			{
				refreshLine( m_uiEditBufLeftColPos,
						m_uiNumCharsEntered - m_uiEditBufLeftColPos + 1);
			}
			else
			{
				refreshLine( m_uiEditBufLeftColPos, m_uiEditWidth);
			}
		}
		else if (m_uiEditWidth > m_uiNumCharsEntered - m_uiEditBufLeftColPos + 1)
		{
			refreshLine( m_uiEditBufPos,
				m_uiNumCharsEntered - m_uiEditBufPos + 2);
		}
		else
		{
			refreshLine( m_uiEditBufPos,
				m_uiEditWidth - (m_uiEditBufPos - m_uiEditBufLeftColPos));
		}
		m_bValuesChanged = TRUE;
	}
	else
	{
		beep( "Cannot backspace - at beginning of line");
	}
}

/*===========================================================================
Desc:		Delete all characters from current cursor position to end of line,
			including the character the cursor is currently positioned on.
===========================================================================*/
void FlmForm::clearLine( void)
{
	if (m_uiEditBufPos < m_uiNumCharsEntered)
	{
		m_uiNumCharsEntered = m_uiEditBufPos;
		refreshLine( m_uiEditBufPos,
			m_uiEditWidth - (m_uiEditBufPos - m_uiEditBufLeftColPos));
		m_bValuesChanged = TRUE;
	}
	else
	{
		beep( "Nothing to clear - at end of line");
	}
}

/*===========================================================================
Desc:		Redisplay the input cursor.
===========================================================================*/
void FlmForm::displayInputCursor( void)
{
	FLMUINT	uiCol;
	FLMUINT	uiRow;

	FTXWinGetCursorPos( m_pDisplayWindow, &uiCol, &uiRow);
	if ((uiCol != m_uiEditColumn + (m_uiEditBufPos - m_uiEditBufLeftColPos)) ||
		 (uiRow != m_uiEditRow))
	{
		FTXWinSetCursorPos( m_pDisplayWindow,
			m_uiEditColumn + (m_uiEditBufPos - m_uiEditBufLeftColPos), m_uiEditRow);
	}
}

/*===========================================================================
Desc:		Beep and display an error message in the status box.
===========================================================================*/
void FlmForm::beep(
	const char *	pszErrMsg1,
	const char *	pszErrMsg2)
{
	FLMUINT	uiElapTime;
	FLMUINT	uiMilli;
	
	if (m_pStatusWindow)
	{
		if (pszErrMsg1)
		{
			FTXWinSetBackFore( m_pStatusWindow,
					m_bMonochrome ? WPS_BLACK : WPS_RED,
					WPS_WHITE);
			FTXWinClearLine( m_pStatusWindow, 0, 0);
			FTXWinClearLine( m_pStatusWindow, 0, 1);
			FTXWinPrintStrXY( m_pStatusWindow, pszErrMsg1, 0, 0);
			if (pszErrMsg2)
			{
				FTXWinPrintStrXY( m_pStatusWindow, pszErrMsg2, 0, 1);
			}
			FTXWinSetBackFore( m_pStatusWindow,
					m_bMonochrome ? WPS_BLACK : WPS_GREEN,
					WPS_WHITE);
			uiElapTime = FLM_GET_TIMER() - m_uiLastTimeBeeped;
			FLM_TIMER_UNITS_TO_MILLI( uiElapTime, uiMilli);
			if (uiMilli >= 100)
			{
				ftxBeep();
				m_uiLastTimeBeeped = FLM_GET_TIMER();
			}
			m_bShowingHelpStatus = FALSE;
		}
		else if (!m_bShowingHelpStatus)
		{
			FlmFormObject *	pObject = (FlmFormObject *)pszErrMsg2;
			char					szHelp1 [100];
			char					szHelp2 [100];

			FTXWinClearLine( m_pStatusWindow, 0, 0);
			FTXWinClearLine( m_pStatusWindow, 0, 1);
			pObject->getHelp( szHelp1, sizeof( szHelp1), szHelp2, sizeof( szHelp2));
			if (szHelp1 [0])
			{
				FTXWinPrintStrXY( m_pStatusWindow, szHelp1, 0, 0);
			}
			if (szHelp2 [0])
			{
				FTXWinPrintStrXY( m_pStatusWindow, szHelp2, 0, 1);
			}

			m_bShowingHelpStatus = TRUE;
		}
	}
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmFormObject::FlmFormObject()
{
	m_uiObjectId = 0;
	m_uiRow = 0;
	m_uiColumn = 0;
	m_uiWidth = 0;
	m_uiFormat = 0;
	m_bDisplayOnly = FALSE;
	m_uiHelpBuffSize = 0;
	m_pszHelpLine1 = NULL;
	m_pszHelpLine2 = NULL;

	m_pForm = NULL;
	m_uiMaxEditChars = 0;
	m_pvReturnAddress = NULL;
	m_puiReturnLen = NULL;
	m_uiReturnPath [0] = 0;
	m_uiBackColor = WPS_WHITE;
	m_uiForeColor = WPS_BLUE;
	m_pNextObject = NULL;
	m_pPrevObject = NULL;
}

FlmFormObject::~FlmFormObject()
{
	if (m_pszHelpLine1)
	{
		f_free( &m_pszHelpLine1);
	}
}

/*===========================================================================
Desc:		Output text to the field - justify to left, center or right.
===========================================================================*/
void FlmFormObject::outputText(
	char *			pszText,
	FLMUINT			uiRow,
	FLMUINT			uiColumn)
{
	FLMUINT			uiChars = f_strlen( pszText);
	FTX_WINDOW_p	pWindow = m_pForm->getWindow();
	FLMUINT			uiPreFill = 0;
	FLMUINT			uiPostFill = 0;
	char *			pszDispText;
	char *			pszRestoreChar = NULL;
	char				ucSave;

	FTXWinSetBackFore( pWindow,
			m_pForm->getMonochrome() ? WPS_BLACK : m_uiBackColor,
			m_pForm->getMonochrome() ? WPS_WHITE : m_uiForeColor);

	if (m_uiFormat & FORMAT_LEFT_JUSTIFY)
	{
Left_Justify:
		pszDispText = pszText;
		if (uiChars <= m_uiWidth)
		{
			uiPostFill = m_uiWidth - uiChars;
		}
		else
		{
			pszRestoreChar = &pszText [m_uiWidth];
			ucSave = *pszRestoreChar;
			*pszRestoreChar = 0;
		}
	}
	else if (m_uiFormat & FORMAT_RIGHT_JUSTIFY)
	{
		if (uiChars > m_uiWidth)
		{
			pszDispText = &pszText [uiChars - m_uiWidth];
		}
		else
		{
			pszDispText = pszText;
			uiPreFill = m_uiWidth - uiChars;
		}
	}
	else if (m_uiFormat & FORMAT_CENTER_JUSTIFY)
	{
		if (uiChars > m_uiWidth)
		{
			pszDispText = &pszText [(uiChars - m_uiWidth) / 2];
			pszRestoreChar = &pszDispText [m_uiWidth];
			ucSave = *pszRestoreChar;
			*pszRestoreChar = 0;
		}
		else
		{
			pszDispText = pszText;
			uiPreFill = (m_uiWidth - uiChars) / 2;
			uiPostFill = m_uiWidth - uiChars - uiPreFill;
		}
	}
	else
	{
		goto Left_Justify;
	}

	FTXWinSetCursorPos( pWindow, uiColumn, uiRow);

	// Do the blank prefill.

	while (uiPreFill)
	{
		FTXWinPrintChar( pWindow, ' ');
		uiPreFill--;
	}

	// Output the text - will be right justified.

	if (*pszDispText)
	{
		FTXWinPrintStr( pWindow, pszDispText);
	}

	// Blank fill the rest

	while (uiPostFill)
	{
		FTXWinPrintChar( pWindow, ' ');
		uiPostFill--;
	}

	// Restore any overwritten character.

	if (pszRestoreChar)
	{
		*pszRestoreChar = ucSave;
	}
}

/*===========================================================================
Desc:		Set help for an object on the form.
===========================================================================*/
RCODE FlmFormObject::setHelp(
	const char *	pszHelpLine1,
	const char *	pszHelpLine2)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLen1;
	FLMUINT		uiLen2;
	char *		pszTmp;

	uiLen1 = (FLMUINT)((pszHelpLine1)
							 ? (FLMUINT)f_strlen( pszHelpLine1)
							 : (FLMUINT)0);
	uiLen2 = (FLMUINT)((pszHelpLine2)
							 ? (FLMUINT)f_strlen( pszHelpLine2)
							 : (FLMUINT)0);

	// Allocate one buffer for both strings.

	if (uiLen1 + uiLen2 + 2 > m_uiHelpBuffSize)
	{
		if( RC_BAD( rc = f_alloc( uiLen1 + uiLen2 + 2, &pszTmp)))
		{
			goto Exit;
		}
		
		if (m_pszHelpLine1)
		{
			f_free( &m_pszHelpLine1);
			m_pszHelpLine1 = NULL;
			m_uiHelpBuffSize = 0;
		}
		m_pszHelpLine1 = pszTmp;
		m_uiHelpBuffSize = uiLen1 + uiLen2 + 2;
	}

	// Copy help line one into the buffer.

	if (pszHelpLine1 && uiLen1)
	{
		f_memcpy( m_pszHelpLine1, pszHelpLine1, uiLen1);
	}
	m_pszHelpLine1 [uiLen1] = 0;

	// Copy help line 2 into the buffer.

	m_pszHelpLine2 = m_pszHelpLine1 + uiLen1 + 1;
	if (pszHelpLine2 && uiLen2)
	{
		f_memcpy( m_pszHelpLine2, pszHelpLine2, uiLen2);
	}
	m_pszHelpLine2 [uiLen2] = 0;
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Get the help lines for an object on the form.
===========================================================================*/
void FlmFormObject::getHelp(
	char *		pszHelpLine1,
	FLMUINT		uiHelpSize1,
	char *		pszHelpLine2,
	FLMUINT		uiHelpSize2)
{
	FLMUINT		uiLen;

	if (!m_pszHelpLine1)
	{
		*pszHelpLine1 = 0;
	}
	else
	{
		uiLen = f_strlen( m_pszHelpLine1);
		if (uiLen >= uiHelpSize1)
		{
			uiLen = uiHelpSize1 - 1;
		}
		if (uiLen)
		{
			f_memcpy( pszHelpLine1, m_pszHelpLine1, uiLen);
		}
		pszHelpLine1 [uiLen] = 0;

	}

	if (!m_pszHelpLine2)
	{
		*pszHelpLine2 = 0;
	}
	else
	{
		uiLen = f_strlen( m_pszHelpLine2);
		if (uiLen >= uiHelpSize2)
		{
			uiLen = uiHelpSize2 - 1;
		}
		if (uiLen)
		{
			f_memcpy( pszHelpLine2, m_pszHelpLine2, uiLen);
		}
		pszHelpLine2 [uiLen] = 0;
	}
}

/*===========================================================================
Desc:		Finds the registered return path in the tree - will create a node,
			unless the root node of the path does not match the root node of
			the tree.
===========================================================================*/
NODE * FlmFormObject::findPath(
	POOL *		pPool,
	NODE *		pTree)
{
	NODE *	pNode = NULL;
	NODE *	pParentNode;
	FLMUINT	uiLevel;
	FLMUINT	uiTagNum;
	RCODE		rc;

	if (m_uiReturnPath [0] == GedTagNum( pTree))
	{
		pNode = pTree;
		uiLevel = 1;
		for (;;)
		{
			uiTagNum = m_uiReturnPath [uiLevel];
			if (!uiTagNum)
				break;
			pParentNode = pNode;
			pNode = GedChild( pParentNode);
			while ((pNode) && (GedTagNum( pNode) != uiTagNum))
			{
				pNode = GedSibNext( pNode);
			}
			if (!pNode)
			{
				if ((pNode = GedNodeMake( pPool, uiTagNum, &rc)) == NULL)
				{
					goto Exit;
				}
				GedChildGraft( pParentNode, pNode, GED_LAST);
			}
			uiLevel++;
		}
	}
Exit:
	return( pNode);
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmFormTextObject::FlmFormTextObject()
{
	m_pszValue = NULL;
}

FlmFormTextObject::~FlmFormTextObject()
{
	if (m_pszValue)
	{
		f_free( &m_pszValue);
	}
}

/*===========================================================================
Desc:		Sets up a text object on a form.
===========================================================================*/
RCODE FlmFormTextObject::setup(
	FlmForm *	pForm,
	FLMUINT		uiObjectId,
	FLMUINT		uiMaxChars,
	FLMUINT		uiWidth,
	FLMUINT		uiFormat,
	FLMBOOL		bDisplayOnly,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn
	)
{
	RCODE	rc = FERR_OK;

	if (m_pForm)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( uiMaxChars + 1, &m_pszValue)))
	{
		goto Exit;
	}

	m_uiMaxEditChars = uiMaxChars;
	m_pForm = pForm;
	m_eObjectType = FORM_TEXT_OBJECT;
	m_uiObjectId =  uiObjectId;
	m_uiRow = uiRow;
	m_uiColumn = uiColumn;
	m_uiWidth = uiWidth;
	m_uiFormat = uiFormat;
	m_bDisplayOnly = bDisplayOnly;
	setBackColor( uiBackColor);
	setForeColor( uiForeColor);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Set the current value for a text object in a form.
===========================================================================*/
RCODE FlmFormTextObject::setValue(
	const char *	pszValue)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiLen = f_strlen( pszValue);

	if (uiLen > m_uiMaxEditChars)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
		uiLen = m_uiMaxEditChars;
	}
	f_memcpy( m_pszValue, pszValue, uiLen);
	m_pszValue [uiLen] = 0;
//Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the current value for a text object in a form.
===========================================================================*/
RCODE FlmFormTextObject::getValue(
	FLMUINT *	puiLen,
	char *		pszValue)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiLen;

	if (!pszValue)
	{
		*puiLen = (FLMUINT)((m_pszValue)
									? (FLMUINT)f_strlen( m_pszValue) + 1
									: (FLMUINT)1);
	}
	else
	{
		uiLen = f_strlen( m_pszValue);
		if (uiLen > *puiLen - 1)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			uiLen = *puiLen - 1;
		}
		f_memcpy( pszValue, m_pszValue, uiLen);
		pszValue [uiLen] = 0;
		*puiLen = uiLen + 1;
	}
	return( rc);
}

/*===========================================================================
Desc:		Displays the current value for a text object in a form.
===========================================================================*/
void FlmFormTextObject::display(
	FLMUINT		uiDisplayRow,
	FLMUINT		uiDisplayColumn
	)
{
	outputText( m_pszValue, uiDisplayRow, uiDisplayColumn);
}

/*===========================================================================
Desc:		Formats the current value into an edit buffer that will be used to
			edit the object.
===========================================================================*/
void FlmFormTextObject::formatEditBuffer(
	char *	pszEditBuf)
{
	f_strcpy( pszEditBuf, m_pszValue);
}

/*===========================================================================
Desc:		Formats the current value into an the return address, if any.
===========================================================================*/
RCODE FlmFormTextObject::populateReturnAddress( void)
{
	RCODE			rc = FERR_OK;

	if (m_pvReturnAddress)
	{
		rc = getValue( m_puiReturnLen, (char *)m_pvReturnAddress);
	}
	return( rc);
}

/*===========================================================================
Desc:		Formats the current value into an node in the tree, if any.
===========================================================================*/
RCODE FlmFormTextObject::populateReturnPath(
	POOL *		pPool,
	NODE *		pTree)
{
	RCODE			rc = FERR_OK;
	char *		pszTmp;
	FLMUINT		uiLen;
	NODE *		pNode;

	if (m_uiReturnPath [0])
	{
		if ((pNode = findPath( pPool, pTree)) != NULL)
		{
			if( RC_BAD( rc = f_alloc( m_uiMaxEditChars + 1, &pszTmp)))
			{
				goto Exit;
			}

			uiLen = m_uiMaxEditChars + 1;
			if (RC_OK( rc = getValue( &uiLen, pszTmp)))
			{
				rc = GedPutNATIVE( pPool, (NODE *)pNode, pszTmp);
			}
			f_free( &pszTmp);
		}
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmFormUnsignedObject::FlmFormUnsignedObject()
{
	m_uiValue = 0;			// Current value in the object
	m_uiMin = 0;
	m_uiMax = 0;
}

FlmFormUnsignedObject::~FlmFormUnsignedObject()
{
}

/*===========================================================================
Desc:		Sets up an unsigned object on a form.
===========================================================================*/
RCODE FlmFormUnsignedObject::setup(
	FlmForm *	pForm,
	FLMUINT		uiObjectId,
	FLMUINT		uiMin,
	FLMUINT		uiMax,
	FLMUINT		uiWidth,
	FLMUINT		uiFormat,
	FLMBOOL		bDisplayOnly,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn
	)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiHexDigits;

	if (m_pForm)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_pForm = pForm;
	m_eObjectType = FORM_UNSIGNED_OBJECT;
	m_uiObjectId =  uiObjectId;
	if (uiMin > uiMax)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_uiMin = uiMin;
	m_uiMax = uiMax;

	// Determine the maximum number of edit characters - based on maximum
	// value.  Calculate the maximum decimal digits and the maximum
	// hex digits - take the greater of these two.

	m_uiMaxEditChars = 1;
	uiMax /= 10;
	while (uiMax)
	{
		uiMax /= 10;
		m_uiMaxEditChars++;
	}
	uiHexDigits = 3;			// Accounts for at least "0x" and one digit
	uiMax = m_uiMax >> 4;
	while (uiMax)
	{
		uiMax >>= 4;
		uiHexDigits++;
	}
	if (uiHexDigits > m_uiMaxEditChars)
	{
		m_uiMaxEditChars = uiHexDigits;
	}
	if (uiWidth > m_uiMaxEditChars)
	{
		m_uiMaxEditChars = uiWidth;
	}
	m_uiRow = uiRow;
	m_uiColumn = uiColumn;
	m_uiWidth = uiWidth;
	m_uiFormat = uiFormat;
	m_bDisplayOnly = bDisplayOnly;
	setBackColor( uiBackColor);
	setForeColor( uiForeColor);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Set the current value for an unsigned object in a form.
===========================================================================*/
RCODE FlmFormUnsignedObject::setValue(
	FLMUINT	uiValue
	)
{
	RCODE		rc = FERR_OK;

	if (uiValue < m_uiMin || uiValue > m_uiMax)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
	}
	m_uiValue = uiValue;
//Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the current value for an unsigned object in a form.
===========================================================================*/
RCODE FlmFormUnsignedObject::getValue(
	FLMUINT *	puiValue
	)
{
	*puiValue = m_uiValue;
	return( FERR_OK);
}

/*===========================================================================
Desc:		Converts the text and verifies that it is a proper unsigned value.
===========================================================================*/
FSTATIC FLMBOOL flmIsUnsigned(
	char *		pszEditBuf,
	FLMUINT *	puiValue,
	char *		pszErrMsg)
{
	FLMBOOL	bOk = TRUE;
	FLMUINT	uiValue = 0;

	// Convert the value - using HEX or decimal.

	if ((*pszEditBuf == '0') &&
		 ((*(pszEditBuf + 1) == 'x') ||
		  (*(pszEditBuf + 1) == 'X')))
	{
		pszEditBuf += 2;

		if (!(*pszEditBuf))
		{
			bOk = FALSE;
			f_strcpy( pszErrMsg,
				"Invalid HEX number - must have at least one hex digit");
			goto Exit;
		}

		// Hex conversion

		while (*pszEditBuf)
		{
			if (((*pszEditBuf >= '0') && (*pszEditBuf <= '9')) ||
				 ((*pszEditBuf >= 'A') && (*pszEditBuf <= 'F')) ||
				 ((*pszEditBuf >= 'a') && (*pszEditBuf <= 'f')))
			{
				if (uiValue > 0x0FFFFFFF)
				{
					bOk = FALSE;
					f_strcpy( pszErrMsg, "HEX number is too large");
					goto Exit;
				}
				uiValue <<= 4;
				if ((*pszEditBuf >= '0') && (*pszEditBuf <= '9'))
				{
					uiValue += (FLMUINT)(*pszEditBuf - '0');
				}
				else if ((*pszEditBuf >= 'A') && (*pszEditBuf <= 'F'))
				{
					uiValue += (FLMUINT)(*pszEditBuf - 'A' + 10);
				}
				else
				{
					uiValue += (FLMUINT)(*pszEditBuf - 'a' + 10);
				}
			}
			else
			{
				f_strcpy( pszErrMsg, "Invalid digit in HEX number");
				bOk = FALSE;
				goto Exit;
			}
			pszEditBuf++;
		}
	}
	else
	{
		if (!(*pszEditBuf))
		{
			bOk = FALSE;
			f_strcpy( pszErrMsg,
				"Invalid number - must have at least one decimal digit");
			goto Exit;
		}


		// Decimal conversion

		while (*pszEditBuf)
		{
			if ((*pszEditBuf >= '0') && (*pszEditBuf <= '9'))
			{
				if (uiValue > (0xFFFFFFFF / 10))
				{
Too_Large_Error:
					bOk = FALSE;
					f_strcpy( pszErrMsg, "Decimal number is too large");
					goto Exit;
				}
				uiValue *= 10;
				if (uiValue > 0xFFFFFFFF - (FLMUINT)(*pszEditBuf - '0'))
				{
					goto Too_Large_Error;
				}
				uiValue += (FLMUINT)(*pszEditBuf - '0');
			}
			else
			{
				f_strcpy( pszErrMsg, "Invalid digit in number");
				bOk = FALSE;
				goto Exit;
			}
			pszEditBuf++;
		}
	}
	*puiValue = uiValue;
Exit:
	return( bOk);
}

/*===========================================================================
Desc:		Converts the text and verifies that it is a proper unsigned value.
===========================================================================*/
FLMBOOL FlmFormUnsignedObject::verifyValue(
	char *		pszEditBuf,
	FLMUINT *	puiValue)
{
	FLMBOOL		bOk = TRUE;
	char			szErrMsg [100];
	FLMUINT		uiValue;

	bOk = flmIsUnsigned( pszEditBuf, &uiValue, szErrMsg);
	if (!bOk)
	{
		m_pForm->beep( szErrMsg);
		goto Exit;
	}
	if ((uiValue < m_uiMin) || (uiValue > m_uiMax))
	{
		f_sprintf( (char *)szErrMsg, "Value must be >= %u and <= %u", 
					(unsigned)m_uiMin, (unsigned)m_uiMax);
		m_pForm->beep( szErrMsg);
		bOk = FALSE;
		goto Exit;
	}
	*puiValue = uiValue;
Exit:
	return( bOk);
}

/*===========================================================================
Desc:		Displays the current value for an unsigned object in a form.
===========================================================================*/
void FlmFormUnsignedObject::display(
	FLMUINT		uiDisplayRow,
	FLMUINT		uiDisplayColumn
	)
{
	char	szValue [20];

	formatEditBuffer( szValue);
	outputText( szValue, uiDisplayRow, uiDisplayColumn);
}

/*===========================================================================
Desc:		Formats the current value into an edit buffer that will be used to
			edit the object.
===========================================================================*/
void FlmFormUnsignedObject::formatEditBuffer(
	char *	pszEditBuf)
{
	char	szFormat [10];

	if (m_uiFormat & (FORMAT_UPPER_HEX | FORMAT_LOWER_HEX))
	{
		if (m_uiFormat & FORMAT_ZERO_LEAD)
		{
			if (m_uiFormat & FORMAT_UPPER_HEX)
			{
				f_sprintf( (char *)szFormat, "0x%%0%uX", (unsigned)m_uiWidth);
			}
			else
			{
				f_sprintf( (char *)szFormat, "0x%%0%ux", (unsigned)m_uiWidth);
			}
			f_sprintf( (char *)pszEditBuf, "%s%u", (char *)szFormat, (unsigned)m_uiValue);
		}
		else if (m_uiFormat & FORMAT_UPPER_HEX)
		{
			f_sprintf( (char *)pszEditBuf, "0x%X", (unsigned)m_uiValue);
		}
		else
		{
			f_sprintf( (char *)pszEditBuf, "0x%x", (unsigned)m_uiValue);
		}
	}
	else
	{
		if (m_uiFormat & FORMAT_ZERO_LEAD)
		{
			f_sprintf( (char *)szFormat, "%%0%uu", (unsigned)m_uiWidth);
			f_sprintf( (char *)pszEditBuf, "%s%u", (char *)szFormat, (unsigned)m_uiValue);
		}
		else
		{
			f_sprintf( (char *)pszEditBuf, "%u", (unsigned)m_uiValue);
		}
	}
}

/*===========================================================================
Desc:		Formats the current value into an the return address, if any.
===========================================================================*/
RCODE FlmFormUnsignedObject::populateReturnAddress( void)
{
	RCODE		rc = FERR_OK;

	if (m_pvReturnAddress)
	{
		rc = getValue( (FLMUINT *)m_pvReturnAddress);
	}
	return( rc);
}

/*===========================================================================
Desc:		Formats the current value into an node in the tree, if any.
===========================================================================*/
RCODE FlmFormUnsignedObject::populateReturnPath(
	POOL *		pPool,
	NODE *		pTree)
{
	RCODE			rc = FERR_OK;
	NODE *		pNode;
	FLMUINT		uiValue;

	if (m_uiReturnPath [0])
	{
		if ((pNode = findPath( pPool, pTree)) != NULL)
		{
			if (RC_OK( rc = getValue( &uiValue)))
			{
				rc = GedPutUINT( pPool, pNode, uiValue);
			}
		}
	}
	return( rc);
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmFormSignedObject::FlmFormSignedObject()
{
	m_iValue = 0;			// Current value in the object
	m_iMin = 0;
	m_iMax = 0;
}

FlmFormSignedObject::~FlmFormSignedObject()
{
}

/*===========================================================================
Desc:		Sets up a signed object on a form.
===========================================================================*/
RCODE FlmFormSignedObject::setup(
	FlmForm *	pForm,
	FLMUINT		uiObjectId,
	FLMINT		iMin,
	FLMINT		iMax,
	FLMUINT		uiWidth,
	FLMUINT		uiFormat,
	FLMBOOL		bDisplayOnly,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn
	)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiTmpDigits;
	FLMUINT	uiTmp;

	if (m_pForm)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_pForm = pForm;
	m_eObjectType = FORM_SIGNED_OBJECT;
	m_uiObjectId =  uiObjectId;
	if (iMin > iMax)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_iMin = iMin;
	m_iMax = iMax;

	// First look at the maximum width required for the maximum number.
	// Look at the number of digits required for both a HEX and a
	// non-hex representation.

	m_uiMaxEditChars = 1;
	if (iMax < 0)
	{
		m_uiMaxEditChars++;
		iMax = -iMax;
	}
	iMax /= 10;
	while (iMax)
	{
		iMax /= 10;
		m_uiMaxEditChars++;
	}

	// Calculate the digits required to enter the number in hex.

	uiTmpDigits = 3;			// Accounts for at least "0x" and one digit
	uiTmp = ((FLMUINT)m_iMax) >> 4;
	while (uiTmp)
	{
		uiTmp >>= 4;
		uiTmpDigits++;
	}
	if (uiTmpDigits > m_uiMaxEditChars)
	{
		m_uiMaxEditChars = uiTmpDigits;
	}

	// Now look at the minimum number.

	uiTmpDigits = 1;
	if (iMin < 0)
	{
		uiTmpDigits++;
		iMin = -iMin;
	}
	iMin /= 10;
	while (iMin)
	{
		iMin /= 10;
		uiTmpDigits++;
	}
	if (uiTmpDigits > m_uiMaxEditChars)
	{
		m_uiMaxEditChars = uiTmpDigits;
	}

	// Calculate the hex digits for the minimum number.

	uiTmpDigits = 3;			// Accounts for at least "0x" and one digit
	uiTmp = ((FLMUINT)m_iMin) >> 4;
	while (uiTmp)
	{
		uiTmp >>= 4;
		uiTmpDigits++;
	}
	if (uiTmpDigits > m_uiMaxEditChars)
	{
		m_uiMaxEditChars = uiTmpDigits;
	}

	// Finally, if the width is greater, set it as the maximum
	// edit characters.

	if (uiWidth > m_uiMaxEditChars)
	{
		m_uiMaxEditChars = uiWidth;
	}

	m_uiRow = uiRow;
	m_uiColumn = uiColumn;
	m_uiWidth = uiWidth;
	m_uiFormat = uiFormat;
	m_bDisplayOnly = bDisplayOnly;
	setBackColor( uiBackColor);
	setForeColor( uiForeColor);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Set the current value for a signed object in a form.
===========================================================================*/
RCODE FlmFormSignedObject::setValue(
	FLMINT	iValue
	)
{
	RCODE		rc = FERR_OK;

	if (iValue < m_iMin || iValue > m_iMax)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
	}
	m_iValue = iValue;
//Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the current value for a signed object in a form.
===========================================================================*/
RCODE FlmFormSignedObject::getValue(
	FLMINT *	piValue
	)
{
	*piValue = m_iValue;
	return( FERR_OK);
}

/*===========================================================================
Desc:		Converts the text and verifies that it is a proper signed value.
===========================================================================*/
FLMBOOL FlmFormSignedObject::verifyValue(
	char *		pszEditBuf,
	FLMINT *		piValue)
{
	FLMBOOL	bOk = TRUE;
	FLMUINT	uiValue = 0;
	FLMINT	iValue;
	char		szErrMsg [100];
	FLMBOOL	bNegative;

	// Convert the value - using HEX or decimal.

	if ((*pszEditBuf == '0') &&
		 ((*(pszEditBuf + 1) == 'x') ||
		  (*(pszEditBuf + 1) == 'X')))
	{
		pszEditBuf += 2;

		if (!(*pszEditBuf))
		{
			bOk = FALSE;
			m_pForm->beep( "Invalid HEX number");
			goto Exit;
		}

		// Hex conversion

		while (*pszEditBuf)
		{
			if (((*pszEditBuf >= '0') && (*pszEditBuf <= '9')) ||
				 ((*pszEditBuf >= 'A') && (*pszEditBuf <= 'F')) ||
				 ((*pszEditBuf >= 'a') && (*pszEditBuf <= 'f')))
			{
				if (uiValue > 0x0FFFFFFF)
				{
					bOk = FALSE;
					m_pForm->beep( "HEX number is too large");
					goto Exit;
				}
				uiValue <<= 4;
				if ((*pszEditBuf >= '0') && (*pszEditBuf <= '9'))
				{
					uiValue += (FLMUINT)(*pszEditBuf - '0');
				}
				else if ((*pszEditBuf >= 'A') && (*pszEditBuf <= 'F'))
				{
					uiValue += (FLMUINT)(*pszEditBuf - 'A' + 10);
				}
				else
				{
					uiValue += (FLMUINT)(*pszEditBuf - 'a' + 10);
				}
			}
			else
			{
				m_pForm->beep( "Invalid digit in HEX number");
				bOk = FALSE;
				goto Exit;
			}
			pszEditBuf++;
		}
		iValue = (FLMINT)uiValue;
	}
	else
	{
		if (*pszEditBuf == '-')
		{
			bNegative = TRUE;
			pszEditBuf++;
		}
		else
		{
			bNegative = FALSE;
		}

		// Decimal conversion

		while (*pszEditBuf)
		{
			if ((*pszEditBuf >= '0') && (*pszEditBuf <= '9'))
			{
				if (((!bNegative) && (uiValue > (0x7FFFFFFF / 10))) ||
					 ((bNegative) && (uiValue > (0x80000000 / 10))))
				{
Too_Large_Error:
					bOk = FALSE;
					if (!bNegative)
					{
						m_pForm->beep( "Decimal number is too large");
					}
					else
					{
						m_pForm->beep( "Negative decimal number is too small");
					}
					goto Exit;
				}
				uiValue *= 10;
				if (((!bNegative) &&
					  (uiValue > 0x7FFFFFFF - (FLMUINT)(*pszEditBuf - '0'))) ||
					 ((bNegative) &&
					  (uiValue > 0x80000000 - (FLMUINT)(*pszEditBuf - '0'))))
				{
					goto Too_Large_Error;
				}
				uiValue += (FLMUINT)(*pszEditBuf - '0');
			}
			else
			{
				m_pForm->beep( "Invalid digit in number");
				bOk = FALSE;
				goto Exit;
			}
			pszEditBuf++;
		}
		iValue = (FLMINT)uiValue;
		if ((bNegative) && (uiValue != 0x80000000))
		{
			iValue = -iValue;
		}
	}
	if ((iValue < m_iMin) || (iValue > m_iMax))
	{
		f_sprintf( (char *)szErrMsg, "Value must be >= %u and <= %u", 
					(unsigned)m_iMin, (unsigned)m_iMax);
		m_pForm->beep( szErrMsg);
		bOk = FALSE;
		goto Exit;
	}
	*piValue = iValue;
Exit:
	return( bOk);
}

/*===========================================================================
Desc:		Displays the current value for a signed object in a form.
===========================================================================*/
void FlmFormSignedObject::display(
	FLMUINT		uiDisplayRow,
	FLMUINT		uiDisplayColumn
	)
{
	char	szValue [20];

	formatEditBuffer( szValue);
	outputText( szValue, uiDisplayRow, uiDisplayColumn);
}

/*===========================================================================
Desc:		Formats the current value into an edit buffer that will be used to
			edit the object.
===========================================================================*/
void FlmFormSignedObject::formatEditBuffer(
	char *	pszEditBuf)
{
	char	szFormat [10];

	if (m_uiFormat & (FORMAT_UPPER_HEX | FORMAT_LOWER_HEX))
	{
		if (m_uiFormat & FORMAT_ZERO_LEAD)
		{
			if (m_uiFormat & FORMAT_UPPER_HEX)
			{
				f_sprintf( (char *)szFormat, "0x%%0%uX", (unsigned)m_uiWidth);
			}
			else
			{
				f_sprintf( (char *)szFormat, "0x%%0%ux", (unsigned)m_uiWidth);
			}
			f_sprintf( (char *)pszEditBuf, "%s%d", (char *)szFormat, (int)m_iValue);
		}
		else if (m_uiFormat & FORMAT_UPPER_HEX)
		{
			f_sprintf( (char *)pszEditBuf, "0x%X", (unsigned)m_iValue);
		}
		else
		{
			f_sprintf( (char *)pszEditBuf, "0x%x", (unsigned)m_iValue);
		}
	}
	else
	{
		if (m_uiFormat & FORMAT_ZERO_LEAD)
		{
			f_sprintf( (char *)szFormat, "%%0%ud", (unsigned)m_uiWidth);
			f_sprintf( (char *)pszEditBuf, "%s%d", (char *)szFormat, (int)m_iValue);
		}
		else
		{
			f_sprintf( (char *)pszEditBuf, "%u", (unsigned)m_iValue);
		}
	}
}

/*===========================================================================
Desc:		Formats the current value into an the return address, if any.
===========================================================================*/
RCODE FlmFormSignedObject::populateReturnAddress( void)
{
	RCODE		rc = FERR_OK;

	if (m_pvReturnAddress)
	{
		rc = getValue( (FLMINT *)m_pvReturnAddress);
	}
	return( rc);
}

/*===========================================================================
Desc:		Formats the current value into an node in the tree, if any.
===========================================================================*/
RCODE FlmFormSignedObject::populateReturnPath(
	POOL *		pPool,
	NODE *		pTree)
{
	RCODE			rc = FERR_OK;
	NODE *		pNode;
	FLMINT		iValue;

	if (m_uiReturnPath [0])
	{
		if ((pNode = findPath( pPool, pTree)) != NULL)
		{
			if (RC_OK( rc = getValue( &iValue)))
			{
				rc = GedPutINT( pPool, pNode, iValue);
			}
		}
	}
	return( rc);
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmFormRecordObject::FlmFormRecordObject()
{
	m_pszTitle = NULL;
	m_bAutoEnter = TRUE;
	m_hDb = HFDB_NULL;
	m_uiContainer = FLM_DATA_CONTAINER;
	GedPoolInit( &m_pool, 512);
	m_pRecord = NULL;
	m_uiFirstNode = 0;
	m_uiCurrNode = 0;
}

FlmFormRecordObject::~FlmFormRecordObject()
{
	GedPoolFree( &m_pool);
	if (m_pszTitle)
	{
		f_free( &m_pszTitle);
	}
}

/*===========================================================================
Desc:		Sets up a GEDCOM record object on a form.
===========================================================================*/
RCODE FlmFormRecordObject::setup(
	FlmForm *		pForm,
	FLMUINT			uiObjectId,
	const char *	pszTitle,
	FLMUINT			uiWidth,
	FLMUINT			uiBackColor,
	FLMUINT			uiForeColor,
	FLMUINT			uiRow,
	FLMUINT			uiColumn)
{
	RCODE		rc = FERR_OK;
	FLMUINT	uiLen;

	if (m_pForm)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_pForm = pForm;
	m_eObjectType = FORM_RECORD_OBJECT;
	m_uiObjectId =  uiObjectId;
	m_uiMaxEditChars = 16;
	if (m_uiMaxEditChars < uiWidth)
	{
		m_uiMaxEditChars = uiWidth;
	}
	m_uiRow = uiRow;
	m_uiColumn = uiColumn;
	m_uiWidth = uiWidth;
	m_bDisplayOnly = FALSE;
	setBackColor( uiBackColor);
	setForeColor( uiForeColor);
	if (pszTitle && *pszTitle)
	{
		uiLen = f_strlen( pszTitle);
		if( RC_BAD( rc = f_alloc( uiLen + 1, &m_pszTitle)))
		{
			goto Exit;
		}
		f_memcpy( m_pszTitle, pszTitle, uiLen + 1);
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Set the current value for a GEDCOM record object in a form.
===========================================================================*/
RCODE FlmFormRecordObject::setValue(
	NODE *	pRecord
	)
{
	RCODE	rc = FERR_OK;

	GedPoolReset( &m_pool, NULL);
	if (!pRecord)
	{
		m_pRecord = NULL;
	}
	else
	{
		if ((m_pRecord = GedCopy( &m_pool, GED_FOREST, pRecord)) == NULL)
		{
			GedPoolReset( &m_pool, NULL);
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the current value for a GEDCOM record object in a form.
===========================================================================*/
RCODE FlmFormRecordObject::getValue(
	NODE * *		ppRecordValue,
	POOL *		pPool)
{
	RCODE			rc = FERR_OK;

	if (!m_pRecord)
	{
		*ppRecordValue = NULL;
	}
	else
	{
		if ((*ppRecordValue = GedCopy( pPool, GED_FOREST, m_pRecord)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Displays the current value for a GEDCOM record object in a form.
===========================================================================*/
void FlmFormRecordObject::display(
	FLMUINT		uiDisplayRow,
	FLMUINT		uiDisplayColumn
	)
{
	char	szValue [200];

	formatEditBuffer( szValue);
	outputText( szValue, uiDisplayRow, uiDisplayColumn);
}

/*===========================================================================
Desc:		Formats the current value into an edit buffer that will be used to
			edit the object.
===========================================================================*/
void FlmFormRecordObject::formatEditBuffer(
	char *	pszEditBuf)
{
	if (m_pszTitle && *m_pszTitle)
	{
		f_strcpy( pszEditBuf, m_pszTitle);
	}
	else
	{
		*pszEditBuf = 0;
	}
}

/*===========================================================================
Desc:		Formats the current value into an the return address, if any.
===========================================================================*/
RCODE FlmFormRecordObject::populateReturnAddress( void)
{
	RCODE		rc = FERR_OK;

	if (m_pvReturnAddress)
	{
		rc = getValue( (NODE * *)m_pvReturnAddress, (POOL *)m_puiReturnLen);
	}
	return( rc);
}

/*===========================================================================
Desc:		Formats the current value into an node in the tree, if any.
===========================================================================*/
RCODE FlmFormRecordObject::populateReturnPath(
	POOL *		pPool,
	NODE *		pTree)
{
	RCODE			rc = FERR_OK;
	NODE *		pNode;
	NODE *		pTmpNode;
	NODE *		pTmpNode2;
	NODE *		pLastSib;

	if (m_uiReturnPath [0])
	{
		if ((pNode = findPath( pPool, pTree)) != NULL)
		{
			// Clip out any previous children.

			if (GedChild( pNode))
			{
				GedClip( GED_FOREST, GedChild( pNode));
			}

			// Graft in the new nodes as a forest of children.

			pTmpNode = m_pRecord;
			pLastSib = NULL;
			while (pTmpNode)
			{
				if ((pTmpNode2 = GedCopy( pPool, GED_TREE, pTmpNode)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
				if (pLastSib)
				{
					GedSibGraft( pLastSib, pTmpNode2, GED_LAST);
				}
				else
				{
					GedChildGraft( pNode, pTmpNode2, GED_LAST);
				}
				pLastSib = pTmpNode2;
				pTmpNode = GedSibNext( pTmpNode);
			}
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Name:	flintRecEditKeyHook
Desc:	Keyboard callback for edit records within a form.
*****************************************************************************/
FSTATIC RCODE flintRecEditKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut
	)
{
	F_UNREFERENCED_PARM( pCurNd);
	F_UNREFERENCED_PARM( pRecEditor);
	*((FLMUINT *)UserData) = uiKeyIn;
	switch (uiKeyIn)
	{
		case WPK_TAB:
		case WPK_STAB:
		case WPK_LEFT:
		case WPK_RIGHT:
			if (puiKeyOut)
			{
				*puiKeyOut = WPK_ESCAPE;
			}
			break;
		default:
			if (puiKeyOut)
			{
				*puiKeyOut = uiKeyIn;
			}
			break;
	}
	return( FERR_OK);
}


/*===========================================================================
Desc:		Pops into the editor for the GEDCOM record.
===========================================================================*/
FLMUINT FlmFormRecordObject::edit(
	FLMBOOL *	pbChanged
	)
{
	FLMUINT			uiExitChar = 0;
	FTX_SCREEN_p	pScreen = m_pForm->getScreen();
	F_RecEditor	*	pRecEditor = NULL;
	POOL				tmpPool;
	NODE *			pFirstRecord;
	NODE *			pTmpRecord;
	NODE *			pCopyRecord;
	NODE *			pLastCopiedRecord;
	NODE *			pCurrentNode;
	NODE *			pFirstNode;
	FLMUINT			uiCurrNode;
	FLMUINT			uiFirstNode;
	FLMUINT			uiNumRows;
	FLMUINT			uiNumCols;
	FLMBOOL			bHaveFirstNode;
	FLMBOOL			bHaveCurrNode;

	*pbChanged = FALSE;
	GedPoolInit( &tmpPool, 512);
	
	if (RC_BAD( getValue( &pTmpRecord, &tmpPool)))
	{
		goto Exit;
	}

	if (FTXScreenGetSize( pScreen, &uiNumCols, &uiNumRows) != FTXRC_SUCCESS)
	{
		goto Exit;
	}

	if ((pRecEditor = new F_RecEditor) == NULL)
	{
		goto Exit;
	}

	if (RC_BAD( pRecEditor->Setup( pScreen)))
	{
		goto Exit;
	}

	pRecEditor->setTitle( m_pszTitle);
	pRecEditor->setKeyHook( flintRecEditKeyHook, (void *)&uiExitChar);
	pRecEditor->setTree( pTmpRecord);
	pRecEditor->setDefaultSource( m_hDb, m_uiContainer);
	pRecEditor->setShutdown( m_pForm->getThread()->getShutdownFlagAddr());

	// Get a pointer to the current node - using the ordinal position
	// that was previously saved.

	pCurrentNode = pFirstNode = pTmpRecord;
	uiCurrNode = m_uiCurrNode;
	uiFirstNode = m_uiFirstNode;
	flmAssert( uiCurrNode >= uiFirstNode);
	while (pCurrentNode && uiCurrNode)
	{
		uiCurrNode--;
		pCurrentNode = pCurrentNode->next;
		if (uiFirstNode)
		{
			uiFirstNode--;
			pFirstNode = pCurrentNode;
		}
	}
	if (pCurrentNode)
	{
		pRecEditor->setCurrentNode( pCurrentNode);
	}
	if (pFirstNode)
	{
		pRecEditor->setFirstNode( pFirstNode);
	}
	pRecEditor->interactiveEdit( 3, 3, uiNumCols - 6, uiNumRows - 6, TRUE, TRUE);
	if (uiExitChar == WPK_ESCAPE)
	{
		uiExitChar = 0;
	}
	else
	{

		// Locate the first record - in case some were inserted.

		pCurrentNode = pRecEditor->getCurrentNode();
		pFirstNode = pRecEditor->getFirstNode();
		pFirstRecord = pTmpRecord = pRecEditor->getRootNode( pCurrentNode);
		for (;;)
		{
			if ((pTmpRecord = GedSibPrev( pTmpRecord)) == NULL)
			{
				break;
			}
			pFirstRecord = pTmpRecord;
		}

		// Determine the ordinal position of the first and current node.

		m_uiCurrNode = 0;
		m_uiFirstNode = 0;
		uiCurrNode = 0;
		bHaveFirstNode = bHaveCurrNode = FALSE;
		pTmpRecord = pFirstRecord;
		while (pTmpRecord)
		{
			if (pTmpRecord == pCurrentNode)
			{
				m_uiCurrNode = uiCurrNode;
				bHaveCurrNode = TRUE;
			}
			else if (pTmpRecord == pFirstNode)
			{
				m_uiFirstNode = uiCurrNode;
				bHaveFirstNode = TRUE;
			}
			if (bHaveFirstNode && bHaveCurrNode)
			{
				break;
			}
			pTmpRecord = pRecEditor->getNextNode( pTmpRecord, FALSE);
			uiCurrNode++;
		}

		// Copy the records back into the form record object.  While
		// copying them back, see if any of them were changed.

		GedPoolReset( &m_pool, NULL);
		m_pRecord = NULL;
		pLastCopiedRecord = NULL;
		pTmpRecord = pFirstRecord;
		while (pTmpRecord)
		{
			if (pRecEditor->isRecordModified( pTmpRecord))
			{
				*pbChanged = TRUE;
			}

			// Copy the record into the form object.

			pRecEditor->copyCleanRecord( &m_pool, pTmpRecord, &pCopyRecord);
			if (pLastCopiedRecord)
			{
				GedSibGraft( pLastCopiedRecord, pCopyRecord, GED_LAST);
			}
			else
			{
				m_pRecord = pCopyRecord;
			}
			pLastCopiedRecord = pCopyRecord;

			// Go to the next record.

			pTmpRecord = GedSibNext( pTmpRecord);
		}
	}

Exit:
	if (pRecEditor)
	{
		pRecEditor->Release();
	}
	GedPoolFree( &tmpPool);
	return( uiExitChar);
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmFormPulldownObject::FlmFormPulldownObject()
{
	m_pPulldown = NULL;
	m_bAutoEnter = TRUE;
	m_bReturnAll = FALSE;
	m_uiItemIdTag = 0;
	m_uiItemNameTag = 0;
}

FlmFormPulldownObject::~FlmFormPulldownObject()
{
	if (m_pPulldown)
	{
		m_pPulldown->Release();
	}
}

/*===========================================================================
Desc:		Sets up a pulldown object on a form.
===========================================================================*/
RCODE FlmFormPulldownObject::setup(
	FlmForm *	pForm,
	FLMUINT		uiObjectId,
	FLMUINT		uiWidth,
	FLMUINT		uiHeight,
	FLMUINT		uiBackColor,
	FLMUINT		uiForeColor,
	FLMUINT		uiRow,
	FLMUINT		uiColumn
	)
{
	RCODE		rc = FERR_OK;

	F_UNREFERENCED_PARM( uiHeight);

	if (m_pForm)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	m_pForm = pForm;
	m_eObjectType = FORM_PULLDOWN_OBJECT;
	m_uiObjectId =  uiObjectId;
	m_uiMaxEditChars = 16;
	if (m_uiMaxEditChars < uiWidth)
	{
		m_uiMaxEditChars = uiWidth;
	}

	if ((m_pPulldown = new FlmPulldownList) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	m_pPulldown->setBackColor( uiBackColor);
	m_pPulldown->setForeColor( uiForeColor);
	m_uiRow = uiRow;
	m_uiColumn = uiColumn;
	m_uiWidth = uiWidth;
	setBackColor( uiBackColor);
	setForeColor( uiForeColor);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets the current item for a pulldown object on a form.
===========================================================================*/
RCODE FlmFormPulldownObject::setCurrentItem(
	FLMUINT		uiItemId
	)
{
	RCODE	rc = FERR_OK;

	if (!m_pPulldown)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	rc = m_pPulldown->setCurrentItem( uiItemId);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the current item for a pulldown object on a form.
===========================================================================*/
RCODE FlmFormPulldownObject::getCurrentItem(
	FLMUINT *	puiItemId
	)
{
	RCODE	rc = FERR_OK;

	if (!m_pPulldown)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	rc = m_pPulldown->getCurrentItem( puiItemId);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Adds an item to a pulldown object on a form.
===========================================================================*/
RCODE FlmFormPulldownObject::addItem(
	FLMUINT			uiItemId,
	const char *	pszDisplayValue,
	FLMUINT			uiShortcutKey)
{
	RCODE	rc = FERR_OK;

	if (!m_pPulldown)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	rc = m_pPulldown->addItem( uiItemId, pszDisplayValue, uiShortcutKey,
										NULL, NULL, NULL);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Adds an item from a pulldown object on a form.
===========================================================================*/
RCODE FlmFormPulldownObject::removeItem(
	FLMUINT		uiItemId
	)
{
	RCODE	rc = FERR_OK;

	if (!m_pPulldown)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	rc = m_pPulldown->removeItem( uiItemId);
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Removes all items from a pulldown object on a form.
===========================================================================*/
RCODE FlmFormPulldownObject::clearItems( void)
{
	RCODE	rc = FERR_OK;

	if (!m_pPulldown)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	rc = m_pPulldown->clearItems();
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Displays the current value for a pulldown object in a form.
===========================================================================*/
void FlmFormPulldownObject::display(
	FLMUINT		uiDisplayRow,
	FLMUINT		uiDisplayColumn
	)
{
	char	szValue [80];

	formatEditBuffer( szValue);
	outputText( szValue, uiDisplayRow, uiDisplayColumn);
}

/*===========================================================================
Desc:		Formats the current value into an edit buffer that will be used to
			edit the object.
===========================================================================*/
void FlmFormPulldownObject::formatEditBuffer(
	char *	pszEditBuf)
{
	FLMUINT	uiItemId;
	FLMUINT	uiLen;

	if (RC_BAD( m_pPulldown->getCurrentItem( &uiItemId)))
	{
		*pszEditBuf = 0;
	}
	else
	{
		uiLen = m_uiMaxEditChars + 1;
		m_pPulldown->getItemDispValue( uiItemId, &uiLen, pszEditBuf);
	}
}

/*===========================================================================
Desc:		Formats the current value into an the return address, if any.
===========================================================================*/
RCODE FlmFormPulldownObject::populateReturnAddress( void)
{
	RCODE		rc = FERR_OK;

	if (m_pvReturnAddress)
	{
		rc = getCurrentItem( (FLMUINT *)m_pvReturnAddress);
	}
	return( rc);
}

/*===========================================================================
Desc:		Formats the current value into an node in the tree, if any.
===========================================================================*/
RCODE FlmFormPulldownObject::populateReturnPath(
	POOL *		pPool,
	NODE *		pTree)
{
	RCODE			rc = FERR_OK;
	NODE *		pNode;
	FLMUINT		uiValue;
	FLMUINT		uiItemId;

	if (m_uiReturnPath [0])
	{
		if ((pNode = findPath( pPool, pTree)) != NULL)
		{
			if (m_bReturnAll)
			{

				// Clip out any previous children.

				if (GedChild( pNode))
				{
					GedClip( GED_FOREST, GedChild( pNode));
				}

				// Graft in the new nodes as a forest of children
				// list of IDs are the immediate children.  Underneat
				// each ID node is the name node.

				if (RC_OK( m_pPulldown->getFirstItem( &uiItemId)))
				{
					for (;;)
					{
						FLMUINT		uiLen;
						char *		pszName;
						NODE *		pIdNode;
						NODE *		pNameNode;

						// Create the ID node.

						if ((pIdNode = GedNodeMake( pPool, m_uiItemIdTag,
												&rc)) == NULL)
						{
							break;
						}
						if (RC_BAD( rc = GedPutUINT( pPool, pIdNode, uiItemId)))
						{
							break;
						}

						// Create the name node.

						if ((pNameNode = GedNodeMake( pPool, m_uiItemNameTag,
													&rc)) == NULL)
						{
							break;
						}
						GedChildGraft( pIdNode, pNameNode, GED_LAST);

						m_pPulldown->getItemDispValue( uiItemId, &uiLen, NULL);
						uiLen++;
						if ((pszName = (char *)GedAllocSpace( pPool,
																		pNameNode, FLM_TEXT_TYPE,
																		uiLen)) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							break;
						}
						m_pPulldown->getItemDispValue( uiItemId, &uiLen, pszName);
						GedChildGraft( pNode, pIdNode, GED_LAST);

						// Go to the next item in the list.

						if (RC_BAD( m_pPulldown->getNextItem( uiItemId, &uiItemId)))
						{
							break;
						}
					}
				}
			}
			else
			{
				if (RC_OK( rc = getCurrentItem( &uiValue)))
				{
					rc = GedPutRecPtr( pPool, pNode, uiValue);
				}
			}
		}
	}
	return( rc);
}

/*===========================================================================
Desc:		Handles keystrokes for a pulldown box on a form.
===========================================================================*/
FSTATIC FLMBOOL pulldownKeyFunc(
	FlmPulldownList *	pPulldown,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvAppData
	)
{
	FLMBOOL	bContinue = TRUE;

	F_UNREFERENCED_PARM( pPulldown);
	F_UNREFERENCED_PARM( pvAppData);

	*puiKeyOut = uiKeyIn;

	switch (uiKeyIn)
	{
		case WPK_TAB:
		case WPK_STAB:
		case WPK_LEFT:
		case WPK_RIGHT:
			bContinue = FALSE;
			break;
		default:
			break;
	}

	return( bContinue);
}

/*===========================================================================
Desc:		Enters the editor for the pulldown list and allows the user to
			select one of the items from the list.
===========================================================================*/
FLMUINT FlmFormPulldownObject::select( void)
{
	FLMUINT			uiBoxUpperLeftCol;
	FLMUINT			uiBoxUpperLeftRow;
	FLMUINT			uiBoxHeight;
	FLMUINT			uiBoxWidth;
	FLMUINT			uiExitChar;
	FLMUINT			uiExitValue;
	FLMUINT			uiFormColumns = m_pForm->getColumns();
	FLMUINT			uiFormRows = m_pForm->getRows();
	FTX_SCREEN_p	pScreen = m_pForm->getScreen();
	FLMBOOL			bRedisplay;

	for (;;)
	{
		m_pPulldown->calcEditLocation(
						uiFormRows, uiFormColumns,
						m_pForm->getEditRow(), m_uiColumn,
						m_uiWidth,
						&uiBoxWidth, &uiBoxHeight,
						&uiBoxUpperLeftCol, &uiBoxUpperLeftRow);

		m_pPulldown->interact( pScreen, m_pForm->getThread(),
						uiBoxWidth, uiBoxHeight, TRUE,
						m_pForm->getUpperLeftColumn() + uiBoxUpperLeftCol,
						m_pForm->getUpperLeftRow() + uiBoxUpperLeftRow,
						FALSE, 0, &uiExitChar, &uiExitValue, 1,
						&bRedisplay, pulldownKeyFunc, NULL);
		if (!bRedisplay)
		{
			break;
		}
	}
	display( m_pForm->getEditRow(), m_uiColumn);
	if (uiExitChar == WPK_ENTER)
	{
		uiExitChar = WPK_TAB;
	}
	else if (uiExitChar == WPK_ESCAPE)
	{
		uiExitChar = 0;
	}
	return( uiExitChar);
}

/*===========================================================================
Desc:		Set insert key callback for pulldown object on a form.
===========================================================================*/
void FlmFormPulldownObject::setPulldownInsertCallback(
	INSERT_FUNC_p	pCallback,
	void *			pvAppData)
{
	if (m_pPulldown)
	{
		m_pPulldown->setPulldownInsertCallback( pCallback, pvAppData);
	}
}

/*===========================================================================
Desc:		Initializes variables
===========================================================================*/
FlmPulldownList::FlmPulldownList()
{
	m_bMonochrome = FALSE;
	m_bInteracting = FALSE;
	m_uiBackColor = WPS_BLUE;
	m_uiForeColor = WPS_WHITE;
	m_pFirstItem = NULL;
	m_pLastItem = NULL;
	m_pCurrentItem = NULL;
	m_pTopItem = NULL;
	m_pWindow = NULL;
	m_pShortcutKeys = NULL;
	m_uiShortcutKeyArraySize = 0;
	m_uiNumShortcutKeys = 0;
	m_pszTypedownBuf = NULL;
	m_uiTypedownBufSize = 0;
	m_uiNumTypedownChars = 0;
	m_uiMaxTypedownChars = 0;
	m_uiDispOffset = 0;
	m_pThread = NULL;
	m_pszListTitle = NULL;
	m_uiListTitleBufSize = 0;
	m_uiListTitleBackColor = WPS_BLACK;
	m_uiListTitleForeColor = WPS_WHITE;
	m_pszListHelp = NULL;
	m_uiListHelpBufSize = 0;
	m_uiListHelpBackColor = WPS_BLACK;
	m_uiListHelpForeColor = WPS_WHITE;
	m_pInsertFunc = NULL;
	m_pvAppData = NULL;
}

FlmPulldownList::~FlmPulldownList()
{
	clearItems();
	if (m_pShortcutKeys)
	{
		f_free( &m_pShortcutKeys);
	}
	if (m_pszTypedownBuf)
	{
		f_free( &m_pszTypedownBuf);
	}
	if (m_pWindow)
	{
		(void)FTXWinFree( &m_pWindow);
	}
	if (m_pszListTitle)
	{
		f_free( &m_pszListTitle);
	}
	if (m_pszListHelp)
	{
		f_free( &m_pszListHelp);
	}
}

/*===========================================================================
Desc:		Finds an item by the item ID.
===========================================================================*/
FlmPulldownItem * FlmPulldownList::findItem(
	FLMUINT		uiItemId
	)
{
	FlmPulldownItem *	pItem;

	pItem = m_pFirstItem;
	while (pItem)
	{
		if (pItem->uiItemId == uiItemId)
		{
			break;
		}
		pItem = pItem->pNext;
	}
	return( pItem);
}

/*===========================================================================
Desc:		Sets the title for the pulldown list.
===========================================================================*/
RCODE FlmPulldownList::setTitle(
	const char *	pszListTitle,
	FLMUINT			uiBackColor,
	FLMUINT			uiForeColor)
{
	RCODE			rc = FERR_OK;
	char *		pszTmp;
	FLMUINT		uiLen;

	if (!pszListTitle || !(*pszListTitle))
	{
		if (m_pszListTitle)
		{
			f_free( &m_pszListTitle);
		}
		m_pszListTitle = NULL;
		m_uiListTitleBufSize = 0;
	}
	else
	{
		uiLen = f_strlen( pszListTitle);
		if (m_uiListTitleBufSize < uiLen + 1)
		{
			if( RC_BAD( rc = f_alloc( uiLen + 1, &pszTmp)))
			{
				goto Exit;
			}
			if (m_pszListTitle)
			{
				f_free( &m_pszListTitle);
				m_pszListTitle = NULL;
				m_uiListTitleBufSize = 0;
			}
			m_pszListTitle = pszTmp;
			m_uiListTitleBufSize = uiLen + 1;
		}
		f_strcpy( m_pszListTitle, pszListTitle);
	}
	m_uiListTitleBackColor = uiBackColor;
	m_uiListTitleForeColor = uiForeColor;
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets the help for the pulldown list.
===========================================================================*/
RCODE FlmPulldownList::setHelp(
	const char *	pszListHelp,
	FLMUINT			uiBackColor,
	FLMUINT			uiForeColor)
{
	RCODE			rc = FERR_OK;
	char *		pszTmp;
	FLMUINT		uiLen;

	if (!pszListHelp || !(*pszListHelp))
	{
		if (m_pszListHelp)
		{
			f_free( &m_pszListHelp);
		}
		m_pszListHelp = NULL;
		m_uiListHelpBufSize = 0;
	}
	else
	{
		uiLen = f_strlen( pszListHelp);
		if (m_uiListHelpBufSize < uiLen + 1)
		{
			if( RC_BAD( rc = f_alloc( uiLen + 1, &pszTmp)))
			{
				goto Exit;
			}

			if (m_pszListHelp)
			{
				f_free( &m_pszListHelp);
				m_pszListHelp = NULL;
				m_uiListHelpBufSize = 0;
			}
			m_pszListHelp = pszTmp;
			m_uiListHelpBufSize = uiLen + 1;
		}
		f_strcpy( m_pszListHelp, pszListHelp);
	}
	m_uiListHelpBackColor = uiBackColor;
	m_uiListHelpForeColor = uiForeColor;
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets the current item for a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::setCurrentItem(
	FLMUINT		uiItemId
	)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;

	if (!uiItemId)
	{
		m_pCurrentItem = m_pFirstItem;
	}
	else
	{

		// Find the item

		if ((pItem = findItem( uiItemId)) == NULL)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
		m_pCurrentItem = pItem;
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the current item for a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::getCurrentItem(
	FLMUINT *	puiItemId
	)
{
	RCODE	rc = FERR_OK;

	if (!m_pCurrentItem)
	{
		if (!m_pFirstItem)
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
		m_pCurrentItem = m_pFirstItem;
	}
	*puiItemId = m_pCurrentItem->uiItemId;
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Gets the first item in a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::getFirstItem(
	FLMUINT *	puiItemId
	)
{
	RCODE	rc = FERR_OK;

	if (!m_pFirstItem)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	*puiItemId = m_pFirstItem->uiItemId;

Exit:

	return( rc);
}

/*===========================================================================
Desc:		Gets the last item in a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::getLastItem(
	FLMUINT *	puiItemId
	)
{
	RCODE	rc = FERR_OK;

	if (!m_pLastItem)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	*puiItemId = m_pLastItem->uiItemId;

Exit:

	return( rc);
}

/*===========================================================================
Desc:		Gets the next item in a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::getNextItem(
	FLMUINT		uiItemId,
	FLMUINT *	puiNextItemId
	)
{
	FlmPulldownItem *	pItem;
	RCODE					rc = FERR_OK;

	pItem = m_pFirstItem;
	while( pItem)
	{
		if( pItem->uiItemId == uiItemId)
		{
			pItem = pItem->pNext;
			break;
		}
		pItem = pItem->pNext;
	}

	if( !pItem)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	*puiNextItemId = pItem->uiItemId;

Exit:

	return( rc);
}

/*===========================================================================
Desc:		Gets the previous item in a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::getPrevItem(
	FLMUINT		uiItemId,
	FLMUINT *	puiPrevItemId
	)
{
	FlmPulldownItem *	pItem;
	RCODE					rc = FERR_OK;

	pItem = m_pLastItem;
	while( pItem)
	{
		if( pItem->uiItemId == uiItemId)
		{
			pItem = pItem->pPrev;
			break;
		}
		pItem = pItem->pPrev;
	}

	if( !pItem)
	{
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	*puiPrevItemId = pItem->uiItemId;

Exit:

	return( rc);
}

/*===========================================================================
Desc:		Gets the display value for an item in the pulldown list.
===========================================================================*/
RCODE FlmPulldownList::getItemDispValue(
	FLMUINT		uiItemId,
	FLMUINT *	puiDispBufLen,
	char *		pszDisplayValue)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;

	// Find the item

	if ((pItem = findItem( uiItemId)) == NULL)
	{
		flmAssert( 0);
		*pszDisplayValue = 0;
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	if (!pszDisplayValue)
	{
		*puiDispBufLen = pItem->uiDisplayValueLen;
	}
	else
	{
		(*puiDispBufLen)--;
		if (*puiDispBufLen > pItem->uiDisplayValueLen)
		{
			*puiDispBufLen = pItem->uiDisplayValueLen;
		}
		f_memcpy( pszDisplayValue, pItem->pszDisplayValue, *puiDispBufLen);
		pszDisplayValue [*puiDispBufLen] = 0;
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Sets the display value for an item in the pulldown list.
===========================================================================*/
RCODE FlmPulldownList::setDisplayValue(
	FLMUINT			uiItemId,
	const char *	pszDisplayValue)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;
	FLMUINT				uiLen;
	char *				pszTmp;

	// Find the item

	if ((pItem = findItem( uiItemId)) == NULL)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	uiLen = f_strlen( pszDisplayValue);
	if (uiLen <= pItem->uiDisplayValueLen)
	{
		f_strcpy( pItem->pszDisplayValue, pszDisplayValue);

		// See if we need to recalculate maximum typedown
		// characters.

		if (pItem->uiDisplayValueLen >= m_uiMaxTypedownChars)
		{
			recalcMaxWidth();
		}
	}
	else
	{
		if( RC_BAD( rc = f_alloc( uiLen + 1, &pszTmp)))
		{
			goto Exit;
		}

		f_free( &pItem->pszDisplayValue);
		pItem->pszDisplayValue = pszTmp;
		f_strcpy( pszTmp, pszDisplayValue);

		// See if we need to increase the size of the typedown
		// buffer.

		if (uiLen + 1 > m_uiTypedownBufSize)
		{
			if (m_pszTypedownBuf)
			{
				f_free( &m_pszTypedownBuf);
				m_pszTypedownBuf = NULL;
				m_uiTypedownBufSize = 0;
			}

			if( RC_BAD( rc = f_alloc( uiLen + 6, &m_pszTypedownBuf)))
			{
				f_free( &pItem);
				goto Exit;
			}
			m_uiTypedownBufSize = uiLen + 6;
		}
		if (uiLen > m_uiMaxTypedownChars)
		{
			m_uiMaxTypedownChars = uiLen;
		}
	}
	pItem->uiDisplayValueLen = uiLen;
	refresh();
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Recalculates the maximum width for the pulldown list.
===========================================================================*/
void FlmPulldownList::recalcMaxWidth( void)
{
	FlmPulldownItem *	pItem;

	m_uiMaxTypedownChars = 0;
	pItem = m_pFirstItem;
	while (pItem)
	{
		if (pItem->uiDisplayValueLen > m_uiMaxTypedownChars)
		{
			m_uiMaxTypedownChars = pItem->uiDisplayValueLen;
		}
		pItem = pItem->pNext;
	}
}

/*===========================================================================
Desc:		Finds the entry for a particular shortcut key.
===========================================================================*/
FlmShortcutKey * FlmPulldownList::findShortcutKey(
	FLMUINT	uiShortcutKey
	)
{
	FlmShortcutKey *	pShortcutKey;
	FLMUINT				uiLow;
	FLMUINT				uiMid;
	FLMUINT				uiHigh;
	FLMUINT				uiNumShortcutKeys;

	uiShortcutKey = (FLMUINT)f_toupper( uiShortcutKey);

	if (m_uiNumShortcutKeys <= 1)
	{
		pShortcutKey =
			(FlmShortcutKey *)(((m_pShortcutKeys) &&
									  (m_pShortcutKeys [0].uiShortcutKey == uiShortcutKey))
									 ? m_pShortcutKeys
									 : (FlmShortcutKey *)NULL);
		goto Exit;
	}

	pShortcutKey = NULL;
	uiNumShortcutKeys = m_uiNumShortcutKeys - 1;
	for (uiHigh = uiNumShortcutKeys, uiLow = 0 ; ; )
	{
		uiMid = (uiLow + uiHigh) >> 1;		// (uiLow + uiHigh) / 2

		if (uiShortcutKey == m_pShortcutKeys [uiMid ].uiShortcutKey)
		{

			// Found a match

			pShortcutKey = &m_pShortcutKeys [uiMid];
			break;
		}

		// Check if we are done - where low equals high.

		if (uiLow >= uiHigh)
		{
			break;			// Done - item not found.
		}

		if (uiShortcutKey < m_pShortcutKeys [uiMid ].uiShortcutKey)
		{
			if (uiMid == 0)
			{
				break;					// Way too high?
			}
			uiHigh = uiMid - 1;		// Too high
		}
		else
		{
			if (uiMid == uiNumShortcutKeys)
			{
				break;	// Done - Hit the top
			}
			uiLow = uiMid + 1;	// Too low
		}
	}
Exit:
	return( pShortcutKey);
}

/*===========================================================================
Desc:		Records a shortcut key and the item it is pointing to.
===========================================================================*/
RCODE FlmPulldownList::addShortcutKey(
	FLMUINT				uiShortcutKey,
	FlmPulldownItem *	pItem
	)
{
	RCODE					rc = FERR_OK;
	FlmShortcutKey *	pShortcutKey;
	FLMUINT				uiPos;
	FlmShortcutKey		TmpShortCut;

	uiShortcutKey = (FLMUINT)f_toupper( uiShortcutKey);

	// Make sure the shortcut key is not already defined.

	if ((pShortcutKey = findShortcutKey( uiShortcutKey)) != NULL)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Make sure we have space to insert the shortcut key

	if (m_uiNumShortcutKeys == m_uiShortcutKeyArraySize)
	{
		if( RC_BAD( rc = f_alloc( sizeof( FlmShortcutKey) * 
			(m_uiShortcutKeyArraySize + 5), &pShortcutKey)))
		{
			goto Exit;
		}

		// Make sure the shortcut is not defined.

		if (m_pShortcutKeys)
		{
			f_memcpy( pShortcutKey, m_pShortcutKeys,
					sizeof( FlmShortcutKey) * m_uiShortcutKeyArraySize);
			f_free( &m_pShortcutKeys);
		}
		m_pShortcutKeys = pShortcutKey;
		m_uiShortcutKeyArraySize += 5;
	}

	// Insert the shortcut key at end of list.

	pShortcutKey = &m_pShortcutKeys [m_uiNumShortcutKeys];
	m_uiNumShortcutKeys++;
	pShortcutKey->uiShortcutKey = uiShortcutKey;
	pShortcutKey->pItem = pItem;
	pItem->uiShortcutKey = uiShortcutKey;

	// Bubble key down to where it belongs in the list.

	if (m_uiNumShortcutKeys > 1)
	{
		uiPos = m_uiNumShortcutKeys - 1;
		while (uiPos &&
				 m_pShortcutKeys [m_uiNumShortcutKeys - 1].uiShortcutKey <
				 m_pShortcutKeys [uiPos - 1].uiShortcutKey)
		{
			uiPos--;
		}

		// No need to do anything if the new shortcut is
		// already in its place.

		if (uiPos < m_uiNumShortcutKeys - 1)
		{

			// Save new shortcut into a temporary buffer.

			f_memcpy( &TmpShortCut, &m_pShortcutKeys [m_uiNumShortcutKeys - 1],
				sizeof( FlmShortcutKey));

			// Move everything from the insert position to the end of the array
			// up one slot.

			f_memmove( &m_pShortcutKeys [uiPos + 1], &m_pShortcutKeys [uiPos],
				sizeof( FlmShortcutKey) * (m_uiNumShortcutKeys - uiPos - 1));

			// Put the new shortcut into its slot in the array.

			f_memcpy( &m_pShortcutKeys [uiPos], &TmpShortCut,
				sizeof( FlmShortcutKey));
		}
	}

Exit:
	return( rc);
}

/*===========================================================================
Desc:		Removes a shortcut key that is pointing to an item.
===========================================================================*/
void FlmPulldownList::removeShortcutKey(
	FLMUINT				uiShortcutKey
	)
{
	FlmShortcutKey *	pShortcutKey;
	FLMUINT				uiPos;

	uiShortcutKey = (FLMUINT)f_toupper( uiShortcutKey);

	// Make sure the shortcut key is defined.

	if ((pShortcutKey = findShortcutKey( uiShortcutKey)) == NULL)
	{
		flmAssert( 0);
	}
	else
	{

		// Move everything above that position down by one.

		uiPos = (FLMUINT)(pShortcutKey - m_pShortcutKeys);
		f_memmove( pShortcutKey, &pShortcutKey [1],
				sizeof( FlmShortcutKey) * 
				(m_uiNumShortcutKeys - uiPos - 1));
		m_uiNumShortcutKeys--;
	}
}

/*===========================================================================
Desc:		Adds an item to a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::addItem(
	FLMUINT				uiItemId,
	const char *		pszDisplayValue,
	FLMUINT				uiShortcutKey,
	FlmPulldownList *	pSubList,
	ITEM_FUNC_p			pFunc,
	void *				pvAppData)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;
	FLMUINT				uiLen;

	// Make sure the item is not already defined.

	if ((pItem = findItem( uiItemId)) != NULL)
	{
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Allocate memory for the new item.

	if( RC_BAD( rc = f_alloc( sizeof( FlmPulldownItem), &pItem)))
	{
		goto Exit;
	}

	f_memset( pItem, 0, sizeof( FlmPulldownItem));
	uiLen = f_strlen( pszDisplayValue);

	if( RC_BAD( rc = f_alloc( uiLen + 1, &pItem->pszDisplayValue)))
	{
		f_free( &pItem);
		goto Exit;
	}

	// See if we need to increase the size of the typedown
	// buffer.

	if (uiLen + 1 > m_uiTypedownBufSize)
	{
		if (m_pszTypedownBuf)
		{
			f_free( &m_pszTypedownBuf);
			m_pszTypedownBuf = NULL;
			m_uiTypedownBufSize = 0;
		}

		if( RC_BAD( rc = f_alloc( uiLen + 6, &m_pszTypedownBuf)))
		{
			f_free( &pItem);
			goto Exit;
		}

		m_uiTypedownBufSize = uiLen + 6;
	}
	if (uiLen > m_uiMaxTypedownChars)
	{
		m_uiMaxTypedownChars = uiLen;
	}

	pItem->uiItemId = uiItemId;
	f_memcpy( pItem->pszDisplayValue, pszDisplayValue, uiLen + 1);
	pItem->uiDisplayValueLen = uiLen;
	pItem->uiShortcutKey = 0;
	pItem->pSubList = pSubList;
	pItem->pFunc = pFunc;
	pItem->pvAppData = pvAppData;
	if ((pItem->pPrev = m_pLastItem) == NULL)
	{
		pItem->uiItemNumber = 1;
		m_pFirstItem = pItem;
	}
	else
	{
		m_pLastItem->pNext = pItem;
		pItem->uiItemNumber = m_pLastItem->uiItemNumber + 1;
	}
	m_pLastItem = pItem;

	if (!m_pCurrentItem)
	{
		m_pCurrentItem = pItem;
	}
	if (uiShortcutKey)
	{
		if (RC_BAD( rc = addShortcutKey( uiShortcutKey, pItem)))
		{
			removeItem( uiItemId);
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Adds an item to a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::insertItem(
	FLMUINT				uiPositionItemId,
	FLMBOOL				bInsertBefore,
	FLMUINT				uiItemId,
	const char *		pszDisplayValue,
	FLMUINT				uiShortcutKey,
	FlmPulldownList *	pSubList,
	ITEM_FUNC_p			pFunc,
	void *				pvAppData)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;
	FlmPulldownItem *	pPositionItem;
	FlmPulldownItem *	pBeforeItem;
	FlmPulldownItem *	pAfterItem;
	FLMUINT				uiLen;
	FLMUINT				uiItemNumber;

	// Find the position item, if any.

	pPositionItem = (FlmPulldownItem *)((uiPositionItemId)
													? findItem( uiPositionItemId)
													: (FlmPulldownItem *)NULL);

	// Make sure the item is not already defined.

	if ((pItem = findItem( uiItemId)) != NULL)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_EXISTS);
		goto Exit;
	}

	// Allocate memory for the new item.

	if( RC_BAD( rc = f_alloc( sizeof( FlmPulldownItem), &pItem)))
	{
		goto Exit;
	}

	f_memset( pItem, 0, sizeof( FlmPulldownItem));
	uiLen = f_strlen( pszDisplayValue);

	if( RC_BAD( rc = f_alloc( uiLen + 1, &pItem->pszDisplayValue)))
	{
		f_free( &pItem);
		goto Exit;
	}

	// See if we need to increase the size of the typedown
	// buffer.

	if (uiLen + 1 > m_uiTypedownBufSize)
	{
		if (m_pszTypedownBuf)
		{
			f_free( &m_pszTypedownBuf);
			m_pszTypedownBuf = NULL;
			m_uiTypedownBufSize = 0;
		}

		if( RC_BAD( rc = f_alloc( uiLen + 6, &m_pszTypedownBuf)))
		{
			f_free( &pItem);
			goto Exit;
		}

		m_uiTypedownBufSize = uiLen + 6;
	}
	if (uiLen > m_uiMaxTypedownChars)
	{
		m_uiMaxTypedownChars = uiLen;
	}

	pItem->uiItemId = uiItemId;
	f_memcpy( pItem->pszDisplayValue, pszDisplayValue, uiLen + 1);
	pItem->uiDisplayValueLen = uiLen;
	pItem->uiShortcutKey = 0;
	pItem->pSubList = pSubList;
	pItem->pFunc = pFunc;
	pItem->pvAppData = pvAppData;

	if (!pPositionItem)
	{
		pPositionItem = (FlmPulldownItem *)((bInsertBefore)
														? m_pFirstItem
														: m_pLastItem);
	}

	// Determine the before and after items.

	if (bInsertBefore)
	{
		pBeforeItem = (FlmPulldownItem *)((pPositionItem)
													 ? pPositionItem->pPrev
													 : (FlmPulldownItem *)NULL);

		pAfterItem = pPositionItem;
	}
	else
	{
		pBeforeItem = pPositionItem;
		pAfterItem = (FlmPulldownItem *)((pPositionItem)
													 ? pPositionItem->pNext
													 : (FlmPulldownItem *)NULL);

	}

	// Link new item between the before and after items.

	if ((pItem->pPrev = pBeforeItem) == NULL)
	{
		pItem->uiItemNumber = 1;
		m_pFirstItem = pItem;
	}
	else
	{
		pBeforeItem->pNext = pItem;
		pItem->uiItemNumber = pBeforeItem->uiItemNumber + 1;
	}
	if ((pItem->pNext = pAfterItem) == NULL)
	{
		m_pLastItem = pItem;
	}
	else
	{
		pAfterItem->pPrev = pItem;
	}
	if (!m_pCurrentItem)
	{
		m_pCurrentItem = pItem;
	}
	if (!m_pTopItem)
	{
		m_pTopItem = pItem;
	}

	// Renumber everything after the new item.

	uiItemNumber = pItem->uiItemNumber + 1;
	while (pAfterItem)
	{
		pAfterItem->uiItemNumber = uiItemNumber++;
		pAfterItem = pAfterItem->pNext;
	}

	// Add the shortcut key, if any.

	if (uiShortcutKey)
	{
		if (RC_BAD( rc = addShortcutKey( uiShortcutKey, pItem)))
		{
			removeItem( uiItemId);
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Adds shortcut key for a particular item in the list.
===========================================================================*/
RCODE FlmPulldownList::addShortcutKey(
	FLMUINT				uiItemId,
	FLMUINT				uiShortcutKey
	)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;

	// Make sure the item is not already defined.

	if ((pItem = findItem( uiItemId)) == NULL)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

	flmAssert( uiShortcutKey);
	if (RC_BAD( rc = addShortcutKey( uiShortcutKey, pItem)))
	{
		goto Exit;
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Removes an item from a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::removeItem(
	FLMUINT		uiItemId
	)
{
	RCODE					rc = FERR_OK;
	FlmPulldownItem *	pItem;
	FlmPulldownItem *	pTmpItem;
	FLMUINT				uiSaveLen;

	if ((pItem = findItem( uiItemId)) == NULL)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}
	uiSaveLen = pItem->uiDisplayValueLen;

	// If the one we are removing is the current item, adjust
	// the current item pointer.

	if (m_pCurrentItem == pItem)
	{
		if ((m_pCurrentItem = pItem->pNext) == NULL)
		{
			m_pCurrentItem = pItem->pPrev;
		}
	}

	// If the one we are removing is the top item, adjust
	// the top item pointer.

	if (m_pTopItem == pItem)
	{
		if ((m_pTopItem = pItem->pNext) == NULL)
		{
			m_pTopItem = pItem->pPrev;
		}
	}

	// Decrement all of the item numbers of items that come after this
	// one.

	pTmpItem = pItem->pNext;
	while (pTmpItem)
	{
		pTmpItem->uiItemNumber--;
		pTmpItem = pTmpItem->pNext;
	}

	// Unlink the item from the list.

	if (pItem->pNext)
	{
		pItem->pNext->pPrev = pItem->pPrev;
	}
	else
	{
		m_pLastItem = pItem->pPrev;
	}
	if (pItem->pPrev)
	{
		pItem->pPrev->pNext = pItem->pNext;
	}
	else
	{
		m_pFirstItem = pItem->pNext;
	}
	if (pItem->pszDisplayValue)
	{
		f_free( &pItem->pszDisplayValue);
	}
	if (pItem->uiShortcutKey)
	{
		removeShortcutKey( pItem->uiShortcutKey);
	}
	f_free( &pItem);

	// See if we need to recalculate maximum typedown
	// characters.

	if (uiSaveLen >= m_uiMaxTypedownChars)
	{
		recalcMaxWidth();
	}
Exit:
	return( rc);
}

/*===========================================================================
Desc:		Clears all items from a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::clearItems( void)
{
	FlmPulldownItem *	pItem;
	FlmPulldownItem *	pNextItem;

	pItem = m_pFirstItem;
	while (pItem)
	{
		pNextItem = pItem->pNext;
		if (pItem->pszDisplayValue)
		{
			f_free( &pItem->pszDisplayValue);
		}
		f_free( &pItem);
		pItem = pNextItem;
	}
	m_pFirstItem = NULL;
	m_pLastItem = NULL;
	m_pCurrentItem = NULL;

	if (m_pShortcutKeys)
	{
		f_free( &m_pShortcutKeys);
		m_uiShortcutKeyArraySize = 0;
		m_uiNumShortcutKeys = 0;
		m_pShortcutKeys = NULL;
	}
	return( FERR_OK);
}

/*===========================================================================
Desc:		Display one item in the pulldown list.
===========================================================================*/
void FlmPulldownList::displayItem(
	FlmPulldownItem *	pItem,
	FLMBOOL				bIsCurrentItem)
{
	FLMUINT			uiRow = pItem->uiItemNumber - m_pTopItem->uiItemNumber;
	FLMUINT			uiLen;
	char *			pszDisp;
	char				ucSave;

	if (bIsCurrentItem)
	{
		FTXWinSetBackFore( m_pWindow,
				m_bMonochrome ? WPS_WHITE : m_uiForeColor,
				m_bMonochrome ? WPS_BLACK : m_uiBackColor);
	}
	else
	{
		FTXWinSetBackFore( m_pWindow,
				m_bMonochrome ? WPS_BLACK : m_uiBackColor,
				m_bMonochrome ? WPS_WHITE : m_uiForeColor);
	}

	FTXWinSetCursorPos( m_pWindow, 0, uiRow);

	if (m_uiDispOffset < pItem->uiDisplayValueLen)
	{
		uiLen = pItem->uiDisplayValueLen - m_uiDispOffset;
		pszDisp = &pItem->pszDisplayValue [m_uiDispOffset];
		if (uiLen > m_uiCols)
		{
			uiLen = m_uiCols;
			ucSave = pszDisp [uiLen];
			pszDisp [uiLen] = 0;
			FTXWinPrintStr( m_pWindow, pszDisp);
			pszDisp [uiLen] = ucSave;
		}
		else
		{
			FTXWinPrintStr( m_pWindow, pszDisp);
		}
	}
	else
	{
		uiLen = 0;
	}

	// Clear the rest of the line

	if (uiLen < m_uiCols)
	{
		FTXWinClearLine( m_pWindow, uiLen, uiRow);
	}
	if (bIsCurrentItem)
	{
		if (!m_uiNumTypedownChars || m_uiNumTypedownChars <= m_uiDispOffset)
		{
			FTXWinSetCursorPos( m_pWindow, 0, uiRow);
		}
		else if (m_uiNumTypedownChars - m_uiDispOffset <= m_uiCols)
		{
			FTXWinSetCursorPos( m_pWindow, m_uiNumTypedownChars - m_uiDispOffset, uiRow);
		}
		else
		{
			FTXWinSetCursorPos( m_pWindow, m_uiCols, uiRow);
		}
	}
}

/*===========================================================================
Desc:		Refresh the display
===========================================================================*/
void FlmPulldownList::refresh( void)
{
	FLMUINT				uiCnt;
	FlmPulldownItem *	pItem;
	FLMBOOL				bDispCurrent;

	if (m_bInteracting)
	{
		FTXWinSetBackFore( m_pWindow,
			m_bMonochrome ? WPS_BLACK : m_uiBackColor,
			m_bMonochrome ? WPS_WHITE : m_uiForeColor);

		// Clear the screen

		(void)FTXWinClear( m_pWindow);

		bDispCurrent = FALSE;
		for (uiCnt = 0, pItem = m_pTopItem;
			  uiCnt < m_uiRows && pItem;
			  uiCnt++, pItem = pItem->pNext)
		{
			if (pItem == m_pCurrentItem)
			{
				bDispCurrent = TRUE;
			}
			else
			{
				displayItem( pItem, FALSE);
			}
		}

		// Always display the current item last, so that the cursor will
		// be positioned on it.

		if (bDispCurrent)
		{
			displayItem( m_pCurrentItem, TRUE);
		}
	}
}

/*===========================================================================
Desc:		Move cursor to the specified item - refresh if necessary.
===========================================================================*/
void FlmPulldownList::positionTo(
	FLMUINT	uiItemId
	)
{
	FlmPulldownItem *	pItem;

	if (m_bInteracting)
	{
		if ((pItem = findItem( uiItemId)) != NULL)
		{
			m_uiNumTypedownChars = 0;
			positionTo( pItem, FALSE);
		}
	}
	else
	{
		setCurrentItem( uiItemId);
	}
}

/*===========================================================================
Desc:		Move cursor to the specified item - refresh if necessary.
===========================================================================*/
void FlmPulldownList::positionTo(
	FlmPulldownItem *	pItem,
	FLMBOOL				bForceRefresh
	)
{
	FLMUINT	uiCnt;
	FLMUINT	uiCounter;

	if ((bForceRefresh) ||
		 ((pItem->uiItemNumber < m_pTopItem->uiItemNumber) ||
		  (pItem->uiItemNumber > m_pTopItem->uiItemNumber + m_uiRows - 1)))
	{
		m_pCurrentItem = pItem;
		if (pItem->uiItemNumber <= m_uiRows)
		{
			m_pTopItem = m_pFirstItem;
		}
		else
		{
			m_pTopItem = m_pCurrentItem;
			for (uiCnt = (m_uiRows - 1) / 2, uiCounter = 0;
					uiCounter < uiCnt;
					uiCounter++)
			{
				m_pTopItem = m_pTopItem->pPrev;
			}
		}
		refresh();
	}
	else
	{
		displayItem( m_pCurrentItem, FALSE);
		m_pCurrentItem = pItem;
		displayItem( m_pCurrentItem, TRUE);
	}
}

/*===========================================================================
Desc:		Move cursor down
===========================================================================*/
void FlmPulldownList::cursorDown( void)
{
	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	if (m_pCurrentItem->pNext)
	{
		if (m_pCurrentItem->pNext->uiItemNumber - m_pTopItem->uiItemNumber + 1 >
			 m_uiRows)
		{
			m_pCurrentItem = m_pCurrentItem->pNext;
			m_pTopItem = m_pTopItem->pNext;
			refresh();
		}
		else
		{
			displayItem( m_pCurrentItem, FALSE);
			m_pCurrentItem = m_pCurrentItem->pNext;
			displayItem( m_pCurrentItem, TRUE);
		}
	}
	else
	{
		displayItem( m_pCurrentItem, TRUE);
	}
}

/*===========================================================================
Desc:		Move cursor up
===========================================================================*/
void FlmPulldownList::cursorUp( void)
{
	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	if (m_pCurrentItem->pPrev)
	{
		if (m_pCurrentItem == m_pTopItem)
		{
			m_pCurrentItem = m_pCurrentItem->pPrev;
			m_pTopItem = m_pTopItem->pPrev;
			refresh();
		}
		else
		{
			displayItem( m_pCurrentItem, FALSE);
			m_pCurrentItem = m_pCurrentItem->pPrev;
			displayItem( m_pCurrentItem, TRUE);
		}
	}
	else
	{
		displayItem( m_pCurrentItem, TRUE);
	}
}

/*===========================================================================
Desc:		Scroll display left
===========================================================================*/
void FlmPulldownList::scrollLeft( void)
{
	if (m_uiDispOffset)
	{
		m_uiDispOffset--;
		refresh();
	}
}

/*===========================================================================
Desc:		Scroll display right
===========================================================================*/
void FlmPulldownList::scrollRight( void)
{
	if (m_uiMaxTypedownChars - m_uiDispOffset > m_uiCols)
	{
		m_uiDispOffset++;
		refresh();
	}
}

/*===========================================================================
Desc:		Move cursor down a page
===========================================================================*/
void FlmPulldownList::pageDown( void)
{
	FLMUINT	uiCnt;

	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	if (m_pCurrentItem->uiItemNumber + m_uiRows > m_pLastItem->uiItemNumber)
	{
		cursorEnd();
	}
	else
	{
		uiCnt = m_uiRows;
		while (uiCnt)
		{
			m_pCurrentItem = m_pCurrentItem->pNext;
			m_pTopItem = m_pTopItem->pNext;
			uiCnt--;
		}
		refresh();
	}
}

/*===========================================================================
Desc:		Move cursor up a page
===========================================================================*/
void FlmPulldownList::pageUp( void)
{
	FLMUINT	uiCnt;

	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	if (m_pTopItem->uiItemNumber < m_uiRows)
	{
		cursorHome();
	}
	else
	{
		uiCnt = m_uiRows;
		while (uiCnt)
		{
			m_pCurrentItem = m_pCurrentItem->pPrev;
			m_pTopItem = m_pTopItem->pPrev;
			uiCnt--;
		}
		refresh();
	}
}

/*===========================================================================
Desc:		Move cursor to first item
===========================================================================*/
void FlmPulldownList::cursorHome( void)
{
	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	if (m_pTopItem != m_pFirstItem)
	{
		m_pCurrentItem = m_pFirstItem;
		m_pTopItem = m_pFirstItem;
		refresh();
	}
	else if (m_pCurrentItem != m_pFirstItem)
	{
		displayItem( m_pCurrentItem, FALSE);
		m_pCurrentItem = m_pFirstItem;
		displayItem( m_pCurrentItem, TRUE);
	}
	else
	{
		displayItem( m_pCurrentItem, TRUE);
	}
}

/*===========================================================================
Desc:		Move cursor to last item
===========================================================================*/
void FlmPulldownList::cursorEnd( void)
{
	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	if (m_pLastItem->uiItemNumber - m_pTopItem->uiItemNumber + 1 > m_uiRows)
	{
		m_pCurrentItem = m_pLastItem;
		m_pTopItem = m_pLastItem;
		while (m_pLastItem->uiItemNumber - m_pTopItem->uiItemNumber + 1 < m_uiRows)
		{
			m_pTopItem = m_pTopItem->pPrev;
		}
		refresh();
	}
	else if (m_pCurrentItem != m_pLastItem)
	{
		displayItem( m_pCurrentItem, FALSE);
		m_pCurrentItem = m_pLastItem;
		displayItem( m_pCurrentItem, TRUE);
	}
	else
	{
		displayItem( m_pCurrentItem, TRUE);
	}
}

/*===========================================================================
Desc:		Move cursor to last item
===========================================================================*/
void FlmPulldownList::backspaceChar( void)
{
	FlmPulldownItem *	pItem;

	if (m_uiNumTypedownChars)
	{
		m_uiNumTypedownChars--;
		m_pszTypedownBuf [m_uiNumTypedownChars] = 0;

		if (!m_uiNumTypedownChars)
		{
			cursorHome();
		}
		else
		{

			// See if any of the items match this one

			pItem = m_pFirstItem;
			while ((pItem) &&
					 (f_strnicmp( pItem->pszDisplayValue, m_pszTypedownBuf,
										m_uiNumTypedownChars) != 0))
			{
				pItem = pItem->pNext;
			}

			// If we found something, move to it.

			if (pItem)
			{
				positionTo( pItem, FALSE);
			}
			else
			{
				flmAssert( 0);
			}
		}
	}
}

/*===========================================================================
Desc:		Perform typedown
===========================================================================*/
void FlmPulldownList::typedown(
	FLMUINT	uiChar
	)
{
	FlmPulldownItem *	pItem;

	// Not a shortcut key, see about typedown.

	if (m_uiNumTypedownChars < m_uiMaxTypedownChars)
	{
		m_pszTypedownBuf [m_uiNumTypedownChars] = (FLMBYTE)uiChar;
		m_pszTypedownBuf [m_uiNumTypedownChars + 1] = 0;

		// See if any of the items match this one

		pItem = m_pFirstItem;
		while ((pItem) &&
				 (f_strnicmp( pItem->pszDisplayValue, m_pszTypedownBuf,
									m_uiNumTypedownChars + 1) != 0))
		{
			pItem = pItem->pNext;
		}

		// If we found something, move to it.

		if (pItem)
		{
			m_uiNumTypedownChars++;
			positionTo( pItem, FALSE);
		}
	}
}

/*===========================================================================
Desc:		See if the key typed is a shortcut.  If so, move to that
			item.
===========================================================================*/
FLMBOOL FlmPulldownList::shortcutKey(
	FLMUINT	uiChar
	)
{
	FlmShortcutKey *	pShortcutKey = findShortcutKey( uiChar);

	if (!pShortcutKey)
	{
		return( FALSE);
	}
	m_uiNumTypedownChars = 0;
	*m_pszTypedownBuf = 0;
	positionTo( pShortcutKey->pItem, FALSE);
	return( TRUE);
}

/*===========================================================================
Desc:		Calculates where a pulldown list should be displayed for
			editing.
===========================================================================*/
void FlmPulldownList::calcEditLocation(
	FLMUINT		uiScreenRows,
	FLMUINT		uiScreenCols,
	FLMUINT		uiAnchorRow,
	FLMUINT		uiLeftAnchorCol,
	FLMUINT		uiAnchorWidth,
	FLMUINT *	puiBoxWidth,
	FLMUINT *	puiBoxHeight,
	FLMUINT *	puiUpperLeftCol,
	FLMUINT *	puiUpperLeftRow
	)
{
	FLMUINT	uiTotalItems = (FLMUINT)((m_pLastItem)
												 ? m_pLastItem->uiItemNumber
												 : (FLMUINT)1);
	FLMUINT	uiMinBoxHeight;
	FLMUINT	uiMaxBoxHeight;
	FLMUINT	uiMinBoxWidth;
	FLMUINT	uiMaxBoxWidth;
	FLMUINT	uiRightAnchorCol = uiLeftAnchorCol + uiAnchorWidth;
	FLMUINT	uiTestLeftCol;
	FLMUINT	uiTestLeftRow;
	FLMUINT	uiTestWidth;
	FLMUINT	uiTestHeight;
	FLMUINT	uiQuadrant;
	FLMUINT	uiMinWidth;

	// Decrement left anchor column by one if it is not zero already.
	
	if (uiLeftAnchorCol)
	{
		uiLeftAnchorCol--;
	}
	
	uiMinWidth = (FLMUINT)((m_pszListTitle)
								  ? (FLMUINT)f_strlen( m_pszListTitle)
								  : (FLMUINT)0);
	if (m_pszListHelp)
	{
		if (uiMinWidth < f_strlen( m_pszListHelp))
		{
			uiMinWidth = f_strlen( m_pszListHelp);
		}
	}
	uiMinWidth += 4;
	if (!m_pLastItem)
	{
		uiMinBoxHeight = uiMaxBoxHeight = 5;
		uiMinBoxWidth = uiMaxBoxWidth = 15;
	}
	else
	{
		uiMinBoxHeight = uiMaxBoxHeight = uiTotalItems + 2;
		if (uiMinBoxHeight > 7)
		{
			uiMinBoxHeight = 7;
		}
		uiMinBoxWidth = uiMaxBoxWidth = m_uiMaxTypedownChars + 2;
		if (uiMinBoxWidth > 15)
		{
			uiMinBoxWidth = 15;
		}
	}
	if (uiMaxBoxWidth < uiMinWidth)
		uiMaxBoxWidth = uiMinWidth;

	// Test quadrant to right/down from right side anchor.

	uiQuadrant = 1;
	for (;; uiQuadrant++)
	{
		switch (uiQuadrant)
		{
			case 1:	// quadrant to right and down from right side of anchor
				uiTestLeftCol = uiRightAnchorCol;
				uiTestLeftRow = uiAnchorRow;
				uiTestWidth = uiScreenCols - uiRightAnchorCol;
				uiTestHeight = uiScreenRows - uiAnchorRow;
				break;
			case 2:	// quadrant to right and up from right side of anchor
				uiTestLeftCol = uiRightAnchorCol;
				uiTestLeftRow = ((FLMUINT)(uiMaxBoxHeight > uiAnchorRow)
												  ? (FLMUINT)0
												  : uiAnchorRow + 1 - uiMaxBoxHeight);
				uiTestWidth = uiScreenCols - uiRightAnchorCol;
				uiTestHeight = uiAnchorRow + 1;
				break;
			case 3:	// quadrant to left and down from left side of anchor
				uiTestLeftCol = (FLMUINT)((uiMaxBoxWidth >= uiLeftAnchorCol)
												  ? (FLMUINT)0
												  : uiLeftAnchorCol + 1 - uiMaxBoxWidth);
				uiTestLeftRow = uiAnchorRow;
				uiTestWidth = uiLeftAnchorCol + 1;
				uiTestHeight = uiScreenRows - uiAnchorRow;
				break;
			case 4:	// quadrant to left and up from left side of anchor
				uiTestLeftCol = (FLMUINT)((uiMaxBoxWidth >= uiLeftAnchorCol)
												  ? (FLMUINT)0
												  : uiLeftAnchorCol + 1 - uiMaxBoxWidth);
				uiTestLeftRow = ((FLMUINT)(uiMaxBoxHeight > uiAnchorRow)
												  ? (FLMUINT)0
												  : uiAnchorRow + 1 - uiMaxBoxHeight);
				uiTestWidth = uiLeftAnchorCol + 1;
				uiTestHeight = uiAnchorRow + 1;
				break;
			case 5:
			default:

				// Try moving anchor column to the left and/or up one column
				// at a time and testing the four options above again.

				if (uiRightAnchorCol)
				{
					if ((uiRightAnchorCol > uiLeftAnchorCol + 2) ||
						 (!uiAnchorRow))
					{
						uiRightAnchorCol--;
						uiQuadrant = 0;
						continue;
					}
					else
					{
						uiAnchorRow--;
						uiQuadrant = 0;
						continue;
					}
				}
				else if (uiAnchorRow)
				{
					uiAnchorRow--;
					uiQuadrant = 0;
					continue;
				}

				flmAssert( 0);	// We have too small of a screen if we get to here.
				break;
		}

		if ((uiTestHeight >= uiMinBoxHeight) &&
			 (uiTestWidth >= uiMinBoxWidth))
		{
			*puiBoxHeight = (FLMUINT)((uiTestHeight >= uiMaxBoxHeight)
											  ? uiMaxBoxHeight
											  : uiTestHeight);
			*puiBoxWidth = (FLMUINT)((uiTestWidth >= uiMaxBoxWidth)
											 ? uiMaxBoxWidth
											 : uiTestWidth);
			*puiUpperLeftCol = uiTestLeftCol;
			*puiUpperLeftRow = uiTestLeftRow;
			goto Exit;
		}
	}

Exit:
	return;
}

/*===========================================================================
Desc:		User interaction with a pulldown list.
===========================================================================*/
RCODE FlmPulldownList::interact(
	FTX_SCREEN_p			pScreen,
	FlmThreadContext *	pThread,
	FLMUINT					uiWidth,
	FLMUINT					uiHeight,
	FLMBOOL					bDoBorder,
	FLMUINT					uiULX,
	FLMUINT					uiULY,
	FLMBOOL					bReturnOnShortcut,
	FLMUINT					uiResponseTimeout,
	FLMUINT *				puiExitChar,
	FLMUINT *				puiExitValue,
	FLMUINT					uiExitValueDepth,
	FLMBOOL *				pbRedisplay,
	LIST_KEY_FUNC_p		pKeyFunc,
	void *					pvAppData
	)
{
	RCODE					rc = FERR_OK;
	FLMUINT				uiChar;
	FLMUINT				uiScreenCols;
	FLMUINT				uiScreenRows;
	FLMUINT				uiStartTime;
	FlmPulldownItem *	pSaveCurrentItem;
	FLMUINT				uiMinWidth;
	FLMBOOL				bBoxFixed = TRUE;
	FLMUINT				uiSaveWidth = uiHeight;
	FLMUINT				uiSaveHeight = uiWidth;

	if (pbRedisplay)
	{
		*pbRedisplay = FALSE;
	}
Start_Over:
	m_pThread = pThread;
	m_pWindow = NULL;
	*puiExitChar = 0;
	*puiExitValue = 0;
	m_bInteracting = TRUE;
	m_uiNumTypedownChars = 0;

	flmAssert( uiExitValueDepth >= 1);

	f_memset( puiExitValue, 0, sizeof( FLMUINT) * uiExitValueDepth);

	FTXScreenGetSize( pScreen, &uiScreenCols, &uiScreenRows);

	// If either width or height is passed in as zero, we
	// need to calculate the box position - put in the center
	// of the screen.

	if (!uiWidth || !uiHeight)
	{
		bBoxFixed = FALSE;
		uiMinWidth = (FLMUINT)((m_pszListTitle)
									  ? (FLMUINT)f_strlen( m_pszListTitle)
									  : (FLMUINT)0);
		if (m_pszListHelp)
		{
			if (uiMinWidth < f_strlen( m_pszListHelp))
			{
				uiMinWidth = f_strlen( m_pszListHelp);
			}
		}
		uiMinWidth += 4;
		if (!itemCount())
		{
			uiWidth = uiMinWidth;
			uiHeight = 3;
		}
		else
		{
			uiWidth = maxWidth() + 2;
			if (uiWidth < uiMinWidth)
			{
				uiWidth = uiMinWidth;
			}
			uiHeight = itemCount() + 2;
		}
		if (uiWidth > uiScreenCols)
		{
			uiWidth = uiScreenCols;
		}
		if (uiHeight > uiScreenRows)
		{
			uiHeight = uiScreenRows;
		}
		uiULX = (uiScreenCols - uiWidth) / 2;
		uiULY = (uiScreenRows - uiHeight) / 2;
		if (uiULY < 3)
		{
			uiULY = 3;
		}
		if (uiULX < 3)
		{
			uiULX = 3;
		}
		if (uiHeight > uiScreenRows - (uiULY * 2))
		{
			uiHeight = uiScreenRows - (uiULY * 2);
		}
		if (uiWidth > uiScreenCols - (uiULX * 2))
		{
			uiWidth = uiScreenCols - (uiULX * 2);
		}
	}

	// Create the display window of the appropriate size.

	if (FTXWinInit( pScreen, uiWidth, uiHeight,
						&m_pWindow) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Position the window on the screen.

	if (FTXWinMove( m_pWindow, uiULX, uiULY) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Prevent window from scrolling.

	if (FTXWinSetScroll( m_pWindow, FALSE) != FTXRC_SUCCESS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Don't wrap lines.

	if (FTXWinSetLineWrap( m_pWindow, FALSE) != FTXRC_SUCCESS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Input cursor should not be present initially.

	if (FTXWinSetCursorType( m_pWindow,
		WPS_CURSOR_INVISIBLE) != FTXRC_SUCCESS)
	{
		flmAssert( 0);
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	// Set foreground and background color for window. 

	if (FTXWinSetBackFore( m_pWindow,
		m_bMonochrome ? WPS_BLACK : m_uiBackColor,
		m_bMonochrome ? WPS_WHITE : m_uiForeColor) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Clear the window

	if (FTXWinClear( m_pWindow) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (bDoBorder)
	{
		if (FTXWinDrawBorder( m_pWindow) != FTXRC_SUCCESS)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		if (m_pszListTitle && *m_pszListTitle)
		{
			FTXWinSetTitle( m_pWindow, m_pszListTitle,
									m_uiListTitleBackColor,
									m_uiListTitleForeColor);
		}
		if (m_pszListHelp && *m_pszListHelp)
		{
			FTXWinSetHelp( m_pWindow, m_pszListHelp,
									m_uiListHelpBackColor,
									m_uiListHelpForeColor);
		}
	}

	if (FTXWinOpen( m_pWindow) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (FTXWinGetCanvasSize( m_pWindow, &m_uiCols,
			&m_uiRows) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Position on the current value

	if (!m_pCurrentItem)
	{
		m_pCurrentItem = m_pFirstItem;
	}
	pSaveCurrentItem = m_pCurrentItem;
	if (m_pCurrentItem)
	{
		positionTo( m_pCurrentItem, TRUE);
	}
	FTXWinSetCursorType( m_pWindow,
							WPS_CURSOR_VISIBLE | WPS_CURSOR_UNDERLINE);

	// Loop forever getting input.

	uiStartTime = FLM_GET_TIMER();
	FLM_SECS_TO_TIMER_UNITS( uiResponseTimeout, uiResponseTimeout);
	for (;;)
	{

		// See if we have been told to exit.

		if ((pThread) &&
			 (pThread->getShutdownFlag()))
		{
			uiChar = 0;
			*puiExitValue = 0;
			goto Exit;
		}

		// See if the response timeout period has elapsed

		if( uiResponseTimeout)
		{
			if( (FLM_GET_TIMER() - uiStartTime) >= uiResponseTimeout)
			{
				uiChar = 0;
				*puiExitValue = 0;
				goto Exit;
			}
		}

		// Need to refresh the input cursor, in case it was changed by
		// a callback function.

		if (FTXWinTestKB( m_pWindow) == FTXRC_SUCCESS)
		{
			FTXWinInputChar( m_pWindow, &uiChar);
			uiStartTime = FLM_GET_TIMER();

			if (pKeyFunc)
			{
				if (!((*pKeyFunc)( this, uiChar, &uiChar, pvAppData)))
				{
					if (m_pCurrentItem)
					{
						*puiExitValue = m_pCurrentItem->uiItemId;
					}
					goto Exit;
				}
			}

			switch( uiChar)
			{
				case WPK_INSERT:
					if (m_pInsertFunc)
					{
						char		szItemName [100];
						FLMUINT	uiNewItem;
						FLMUINT	uiNewShortcutKey;
						FLMUINT	uiBeforeId;
						FLMUINT	uiAfterId;

						uiBeforeId = (FLMUINT)((m_pCurrentItem &&
														m_pCurrentItem->pPrev)
													  ? m_pCurrentItem->pPrev->uiItemId
													  : (FLMUINT)0);
						uiAfterId = (FLMUINT)((m_pCurrentItem &&
														m_pCurrentItem->pNext)
													  ? m_pCurrentItem->pNext->uiItemId
													  : (FLMUINT)0);

						if ((*m_pInsertFunc)( this, uiBeforeId, uiAfterId,
												&uiNewItem, &uiNewShortcutKey,
												szItemName, sizeof( szItemName),
												m_pvAppData))
						{
							if (RC_OK( insertItem( uiBeforeId, FALSE,
											uiNewItem, szItemName,
											uiNewShortcutKey, NULL,
											NULL, NULL)))
							{
								m_bInteracting = FALSE;
								setCurrentItem( uiNewItem);
								uiWidth = uiSaveWidth;
								uiHeight = uiSaveHeight;
								(void)FTXWinFree( &m_pWindow);
								m_pWindow = NULL;
								if (pbRedisplay)
								{
									*pbRedisplay = TRUE;
									goto Exit;
								}
								goto Start_Over;
							}
						}
					}
					break;
				case WPK_DELETE:
					if (m_pInsertFunc && m_pCurrentItem)
					{
						FLMUINT				uiItemId = m_pCurrentItem->uiItemId;
						FlmPulldownItem *	pItem;

						if (m_pCurrentItem->pNext)
						{
							pItem = m_pCurrentItem->pNext;
						}
						else
						{
							pItem = m_pCurrentItem->pPrev;
						}
						removeItem( uiItemId);
						m_bInteracting = FALSE;
						if (pItem)
						{
							setCurrentItem( pItem->uiItemId);
						}
						uiWidth = uiSaveWidth;
						uiHeight = uiSaveHeight;
						(void)FTXWinFree( &m_pWindow);
						m_pWindow = NULL;
						if (pbRedisplay)
						{
							*pbRedisplay = TRUE;
							goto Exit;
						}
						goto Start_Over;
					}
					break;
				case WPK_CTRL_N:
				case WPK_DOWN:
				case WPK_TAB:
					if (m_pCurrentItem)
					{
						cursorDown();
					}
					break;
				case WPK_CTRL_P:
				case WPK_UP:
				case WPK_STAB:
					if (m_pCurrentItem)
					{
						cursorUp();
					}
					break;
				case WPK_CTRL_LEFT:
					if (m_pCurrentItem)
					{
						scrollLeft();
					}
					break;
				case WPK_CTRL_RIGHT:
					if (m_pCurrentItem)
					{
						scrollRight();
					}
					break;
				case WPK_PGUP:
					if (m_pCurrentItem)
					{
						pageUp();
					}
					break;
				case WPK_CTRL_DOWN:
				case WPK_CTRL_UP:
					break;
				case WPK_PGDN:
					if (m_pCurrentItem)
					{
						pageDown();
					}
					break;
				case WPK_CTRL_D:
					break;
				case WPK_CTRL_HOME:
				case WPK_HOME:
					if (m_pCurrentItem)
					{
						cursorHome();
					}
					break;
				case WPK_CTRL_END:
				case WPK_END:
					if (m_pCurrentItem)
					{
						cursorEnd();
					}
					break;
				case WPK_BACKSPACE:
					if (m_pCurrentItem)
					{
						backspaceChar();
					}
					break;
				case WPK_ENTER:
Process_Key:
					if (m_pCurrentItem)
					{
						*puiExitValue = m_pCurrentItem->uiItemId;
						if (m_pCurrentItem->pSubList)
						{
							FLMUINT	uiBoxUpperLeftCol;
							FLMUINT	uiBoxUpperLeftRow;
							FLMUINT	uiBoxHeight;
							FLMUINT	uiBoxWidth;
							FLMUINT	uiExitChar;

							if (bBoxFixed)
							{
								m_pCurrentItem->pSubList->calcEditLocation(
												uiScreenRows, uiScreenCols,
												uiULY + 1 + (m_pCurrentItem->uiItemNumber -
																 m_pTopItem->uiItemNumber),
												uiULX, uiWidth,
												&uiBoxWidth, &uiBoxHeight,
												&uiBoxUpperLeftCol, &uiBoxUpperLeftRow);
							}
							else
							{
								uiBoxWidth = 0;
								uiBoxHeight = 0;
								uiBoxUpperLeftCol = 0;
								uiBoxUpperLeftRow = 0;
							}

							m_pCurrentItem->pSubList->setCurrentItem( 0);
										
							flmAssert( uiExitValueDepth > 1);
							if (uiExitValueDepth > 1)
							{
								m_pCurrentItem->pSubList->interact( pScreen, pThread,
											uiBoxWidth, uiBoxHeight, TRUE,
											uiBoxUpperLeftCol, uiBoxUpperLeftRow,
											bReturnOnShortcut, uiResponseTimeout,
											&uiExitChar, &puiExitValue [1],
											uiExitValueDepth - 1,
											NULL, pKeyFunc, pvAppData);
								if (!uiExitChar)
								{
									uiChar = 0;
									goto Exit;
								}
								if (uiExitChar != WPK_ESCAPE)
								{
									uiChar = uiExitChar;
									goto Exit;
								}
							}
							else
							{
								goto Exit;
							}
						}
						else if (m_pCurrentItem->pFunc)
						{
							if (!m_pCurrentItem->pFunc( this,
									m_pCurrentItem->uiItemId,
									m_pCurrentItem->pvAppData))
							{
								goto Exit;
							}
						}
						else
						{
							goto Exit;
						}
					}
					break;
				case WPK_ALT_H:
				case WPK_CTRL_H:
					break;			// Help
				case WPK_ESCAPE:
					m_pCurrentItem = pSaveCurrentItem;
					*puiExitValue = 0;
					goto Exit;
				case 0:				// Callback can change key to zero to do nothing.
					break;
				default:
					if (m_pCurrentItem)
					{

						// First, see if it is a shortcut key.

						if (shortcutKey( uiChar))
						{
							if (bReturnOnShortcut)
							{
								if (m_pCurrentItem->pSubList ||
									 m_pCurrentItem->pFunc)
								{
									goto Process_Key;
								}
								else
								{
									*puiExitValue = m_pCurrentItem->uiItemId;
									goto Exit;
								}
							}
						}
						else
						{
							typedown( uiChar);
						}
					}
					break;
			}
		}
		else
		{
			f_sleep( 1);
		}
	}

Exit:
	*puiExitChar = uiChar;
	m_pThread = NULL;
	if (m_pWindow)
	{
		(void)FTXWinFree( &m_pWindow);
		m_pWindow = NULL;
	}
	m_bInteracting = FALSE;
	return( rc);
}
