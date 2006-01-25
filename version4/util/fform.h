//-------------------------------------------------------------------------
// Desc:	FlmForm class definitions.
// Tabs:	3
//
//		Copyright (c) 1999,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fform.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FFORM_HPP
#define FFORM_HPP

#include "flaim.h"
#include "fshell.h"
#include "flm_edit.h"

// Forward declarations

class FlmForm;
class FlmFormObject;
class FlmFormTextObject;
class FlmFormUnsignedObject;
class FlmFormSignedObject;
class FlmFormRecordObject;
class FlmPulldownList;

// Formats for display

#define FORMAT_LEFT_JUSTIFY			0x00000001
#define FORMAT_RIGHT_JUSTIFY			0x00000002
#define FORMAT_CENTER_JUSTIFY			0x00000004
#define FORMAT_UPPER_HEX				0x00000008
#define FORMAT_LOWER_HEX				0x00000010
#define FORMAT_ZERO_LEAD				0x00000020
#define FORMAT_DATE_MASK				0x00000700
#define FORMAT_DATE_YYYYMMDD_DASH	0x00000100
#define FORMAT_DATE_YYYYMMDD_SLASH	0x00000200
#define FORMAT_DATE_MMYYYYDD_DASH	0x00000300
#define FORMAT_DATE_MMYYYYDD_SLASH	0x00000400
#define FORMAT_DATE_MONDDYYYY			0x00000500
#define FORMAT_DATE_MONTHDDYYYY		0x00000600
#define FORMAT_DATE_DDMONYYYY			0x00000700
#define FORMAT_TIME_24_HOUR			0x00000800
#define FORMAT_TIME_SHOW_HUNDREDTH	0x00001000

// Types of objects within a form.

enum eFormObjectType
{
	FORM_TEXT_OBJECT,
	FORM_UNSIGNED_OBJECT,
	FORM_SIGNED_OBJECT,
	FORM_PULLDOWN_OBJECT,
	FORM_RECORD_OBJECT
};

typedef enum eFormObjectType FormObjectType;

// Event types for form validation callback

enum eFormEventType
{
	FORM_EVENT_KEY_STROKE,		// Key stroke.
	FORM_EVENT_EXIT_FORM,		// Exiting the form.
	FORM_EVENT_EXIT_OBJECT,		// Exiting object,
	FORM_EVENT_ENTER_OBJECT,	// Entering a new object
	FORM_MAX_EVENTS
};

typedef enum eFormEventType FormEventType;

// Form event callback

typedef FLMBOOL (* FORM_EVENT_CB_p)(
				FormEventType		eFormEvent,
				FlmForm *			pForm,
				FlmFormObject *	pFormObject,
				FLMUINT				uiKeyIn,
				FLMUINT *			puiKeyOut,
				void *				pvAppData);

// List key handler

typedef FLMBOOL (* LIST_KEY_FUNC_p)(
				FlmPulldownList *	pPulldown,
				FLMUINT				uiKeyIn,
				FLMUINT *			puiKeyOut,
				void *				pvAppData);

// Function to execute when a menu item is selected.

typedef FLMBOOL (* ITEM_FUNC_p)(
				FlmPulldownList *	pPulldown,
				FLMUINT				uiItemId,
				void *				pvAppData);

// Function to execute when insert key is pressed on a pulldown.

typedef FLMBOOL (* INSERT_FUNC_p)(
				FlmPulldownList *	pPulldown,
				FLMUINT				uiBeforeId,
				FLMUINT				uiAfterItemId,
				FLMUINT *			puiNewItemId,
				FLMUINT *			puiNewItemShortcutKey,
				char *				pszNewItemName,
				FLMUINT				uiNewItemNameSize,
				void *				pvAppData);

/*===========================================================================
Class:	FlmForm
Desc:		This class manages a form on the screen.
===========================================================================*/

class FlmForm : public F_Base
{
public:

	FlmForm( void);
	virtual ~FlmForm( void);

	RCODE init(
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
		FLMUINT					uiForeColor);

	void refresh( void);

	void setObjectColor(
		FLMUINT		uiObjectId,
		FLMUINT		uiBackColor,
		FLMUINT		uiForeColor);

	void getObjectColor(
		FLMUINT		uiObjectId,
		FLMUINT *	puiBackColor,
		FLMUINT *	puiForeColor);

	void setObjectDisplayOnlyFlag(
		FLMUINT		uiObjectId,
		FLMBOOL		bDisplayOnly);

	void getObjectDisplayOnlyFlag(
		FLMUINT		uiObjectId,
		FLMBOOL *	pbDisplayOnly);

	RCODE setObjectValue(
		FLMUINT		uiObjectId,
		void *		pvValue,
		FLMUINT		uiLen);

	RCODE setObjectReturnAddress(
		FLMUINT		uiObjectId,
		void *		pvReturnAddress,
		FLMUINT *	puiReturnLen);

	RCODE setObjectReturnPath(
		FLMUINT		uiObjectId,
		FLMUINT *	puiPath);

	RCODE setObjectHelp(
		FLMUINT			uiObjectId,
		const char *	pszHelpLine1,
		const char *	pszHelpLine2);

	RCODE setFormEventCB(
		FORM_EVENT_CB_p	pEventCB,
		void *				pvAppData,
		FLMBOOL				bGetKeyStrokes);

	// Methods for adding objects to the form.

	RCODE addTextObject(
		FLMUINT			uiObjectId,
		const char *	pszDefaultVal,
		FLMUINT			uiMaxChars,
		FLMUINT			uiWidth,
		FLMUINT			uiFormat,
		FLMBOOL			bDisplayOnly,
		FLMUINT			uiBackColor,
		FLMUINT			uiForeColor,
		FLMUINT			uiRow,
		FLMUINT			uiColumn);

	RCODE addUnsignedObject(
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
		FLMUINT		uiColumn);

	RCODE addSignedObject(
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
		FLMUINT		uiColumn);

	RCODE addPulldownObject(
		FLMUINT		uiObjectId,
		FLMUINT		uiWidth,				// Box width
		FLMUINT		uiHeight,			// Box height
		FLMUINT		uiBackColor,
		FLMUINT		uiForeColor,
		FLMUINT		uiRow,
		FLMUINT		uiColumn,
		FLMBOOL		bAutoEnter = TRUE);

	RCODE addPulldownItem(
		FLMUINT			uiObjectId,
		FLMUINT			uiItemId,
		const char *	pszDisplayValue,
		FLMUINT			uiShortcutKey);

	RCODE removePulldownItem(
		FLMUINT		uiObjectId,
		FLMUINT		uiItemId);

	RCODE clearPulldownItems(
		FLMUINT		uiObjectId);

	RCODE setPulldownInsertCallback(
		FLMUINT			uiObjectId,
		INSERT_FUNC_p	pCallback,
		void *			pvAppData);

	RCODE setPulldownReturnAll(
		FLMUINT		uiObjectId,
		FLMUINT		uiItemIdTag,
		FLMUINT		uiItemNameTag);

	FlmPulldownList * getPulldownObject(
		FLMUINT			uiObjectid);

	RCODE setEnterMode(
		FLMUINT		uiObjectId,
		FLMBOOL		bAutoEnter);

	RCODE addRecordObject(
		FLMUINT			uiObjectId,
		const char *	pszTitle,
		NODE *			pDefaultRecords,
		FLMUINT			uiWidth,
		FLMUINT			uiBackColor,
		FLMUINT			uiForeColor,
		FLMUINT			uiRow,
		FLMUINT			uiColumn,
		FLMBOOL			bAutoEnter = TRUE);

	RCODE setRecordObjectDbAndCont(
		FLMUINT		uiObjectId,
		HFDB			hDb,
		FLMUINT		uiContainer);

	FLMUINT firstObject(			// Move to first object on form.
		FLMBOOL	bSkipDisplayOnly = TRUE,
		FLMBOOL	bSkipEditable = FALSE,
		FLMBOOL	bRaiseExitEvent = TRUE,
		FLMBOOL	bRaiseEnterEvent = TRUE);

	FLMUINT lastObject(			// Move to last object on form.
		FLMBOOL	bSkipDisplayOnly = TRUE,
		FLMBOOL	bSkipEditable = FALSE,
		FLMBOOL	bRaiseExitEvent = TRUE,
		FLMBOOL	bRaiseEnterEvent = TRUE);

	FLMUINT nextObject(			// Move to next object on form.
		FLMBOOL	bSkipDisplayOnly = TRUE,
		FLMBOOL	bSkipEditable = FALSE,
		FLMBOOL	bRaiseExitEvent = TRUE,
		FLMBOOL	bRaiseEnterEvent = TRUE);

	FLMUINT prevObject(			// Move to previous object on form.
		FLMBOOL	bSkipDisplayOnly = TRUE,
		FLMBOOL	bSkipEditable = FALSE,
		FLMBOOL	bRaiseExitEvent = TRUE,
		FLMBOOL	bRaiseEnterEvent = TRUE);

	FLMUINT upObject(				// Move to UP object on form.
		FLMBOOL	bSkipDisplayOnly = TRUE,
		FLMBOOL	bSkipEditable = FALSE,
		FLMBOOL	bRaiseExitEvent = TRUE,
		FLMBOOL	bRaiseEnterEvent = TRUE);

	FLMUINT downObject(			// Move to DOWN object on form.
		FLMBOOL	bSkipDisplayOnly = TRUE,
		FLMBOOL	bSkipEditable = FALSE,
		FLMBOOL	bRaiseExitEvent = TRUE,
		FLMBOOL	bRaiseEnterEvent = TRUE);

	RCODE getAllReturnData(				// Populates return addresses
		void);

	RCODE getAllReturnDataToTree(		// Populates return paths in GEDCOM tree
		POOL *	pPool,
		NODE *	pTree);

	FLMUINT interact(						// Executes the form
		FLMBOOL *	pbValuesChanged,
		FLMUINT *	puiCurrObjectId = NULL);

	// Methods for retrieving values from a form after
	// interact() has been called.

	RCODE getObjectInfo(
		FLMUINT				uiObjectId,
		FormObjectType *	peObjectType,
		FLMUINT *			puiBackColor,
		FLMUINT *			puiForeColor,
		FLMUINT *			puiRow,
		FLMUINT *			puiCol,
		FLMUINT *			puiWidth,
		void *				pvMin,
		void *				pvMax,
		FLMBOOL *			pbDisplayOnly);

	RCODE getTextVal(
		FLMUINT		uiObjectId,
		FLMUINT *	puiLen,
		char *		pszValRV);

	RCODE getUnsignedVal(
		FLMUINT		uiObjectId,
		FLMUINT *	puiValRV);

	RCODE getSignedVal(
		FLMUINT	uiObjectId,
		FLMINT *	piValRV);

	RCODE getRecordVal(
		FLMUINT		uiObjectId,
		NODE * *		ppRecord,
		POOL *		pPool);

	void beep(
		const char *	pszErrMsg1,
		const char *	pszErrMsg2 = NULL);

	// Other methods.

	inline FTX_WINDOW_p getWindow( void){ return m_pDisplayWindow;}
	inline FLMBOOL getMonochrome( void){ return m_bMonochrome;}
	inline FTX_SCREEN_p getScreen( void){ return m_pScreen;}
	inline FLMUINT getRows( void){ return m_uiRows;}
	inline FLMUINT getColumns( void){ return m_uiCols;}
	inline FLMUINT getUpperLeftColumn( void){ return m_uiUpperLeftColumn;}
	inline FLMUINT getUpperLeftRow( void){ return m_uiUpperLeftRow;}
	inline FLMUINT getEditRow( void){ return m_uiEditRow;}
	inline FlmThreadContext * getThread( void){ return m_pThread;}
	inline FLMBOOL valuesChanged( void){ return m_bValuesChanged;}

private:

	FLMBOOL verifyObject(				// Verify the current object
		FLMBOOL	bRaiseExitEvent);

	FLMUINT changeFocus(					// Change focus to a new object
		FlmFormObject *	pNewObject,
		FLMBOOL				bRaiseEnterEvent,
		FLMBOOL				bForceChange);

	FlmFormObject * findObject(		// Find an object by object ID number
		FLMUINT	uiObjectId);

	FLMBOOL isVisible(
		FlmFormObject * pObject);

	void displayObject(
		FlmFormObject *	pObject);

	RCODE getObjectLocation(
		FLMUINT				uiWidth,
		FLMUINT				uiRow,
		FLMUINT				uiColumn,
		FlmFormObject **	ppPrevObject);

	void linkObjectInForm(
		FlmFormObject *	pNewObject,
		FlmFormObject *	pPrevObject);

	// Line editing functions

	void refreshLine(
		FLMUINT	uiCol,
		FLMUINT	uiChars);
	void cursorHome( void);
	void cursorEnd( void);
	FLMUINT cursorRight( void);
	FLMUINT cursorLeft( void);
	void deleteChar( void);
	void insertChar(
		FLMUINT	uiChar);
	void backspaceChar( void);
	void clearLine( void);
	void displayInputCursor( void);

	FTX_SCREEN_p			m_pScreen;
	FlmThreadContext *	m_pThread;				// Thread this form is associated with - may be NULL.
	FLMBOOL					m_bMonochrome;
	FTX_WINDOW_p			m_pDisplayWindow;
	FTX_WINDOW_p			m_pStatusWindow;
	FLMUINT					m_uiRows;
	FLMUINT					m_uiCols;
	FLMUINT					m_uiUpperLeftRow;
	FLMUINT					m_uiUpperLeftColumn;
	FLMUINT					m_uiBackColor;
	FLMUINT					m_uiForeColor;
	FLMUINT					m_uiTopFormRow;		// Row of form currently displayed at top of window
	FlmFormObject *		m_pCurObject;			// Object that has the focus
	FlmFormObject *		m_pFirstObject;		// First object on the form.
	FlmFormObject *		m_pLastObject;			// Last object on the form.
	FORM_EVENT_CB_p		m_pEventCB;				// Event callback
	void *					m_pvAppData;			// App data for event callback
	FLMBOOL					m_bGetKeyStrokes;		// Give keystrokes to event callback?
	FLMBOOL					m_bShowingHelpStatus;// Are we currently displaying a help status message?
	FLMUINT					m_uiLastTimeBeeped;	// Last time we output a beep sound.
	FLMBOOL					m_bValuesChanged;		// Did values change as a result of the last call to
															// interact()?

	// Fields used for editing the current object

	FLMBOOL					m_bInteracting;			// Are we inside interact?
	char *					m_pszEditBuf;				// Edit buffer.
	FLMUINT					m_uiEditBufSize;			// Size of edit buffer.
	FLMUINT					m_uiMaxCharsToEnter;		// Maximum characters to enter.
	FLMUINT					m_uiNumCharsEntered;		// Characters entered so far.
	FLMUINT					m_uiEditColumn;			// Edit column in display window
	FLMUINT					m_uiEditRow;				// Edit row in display window.
	FLMUINT					m_uiEditWidth;				// Width of edit field within the window.
	FLMUINT					m_uiEditBufPos;			// Position in edit buffer where we are
																// currently inputting data.
	FLMUINT					m_uiEditBufLeftColPos;	// Position in edit buffer that is currently
																// display at leftmost position of input area.
	FLMBOOL					m_bInsertMode;				// In insert mode?
};

/*===========================================================================
Class:	FlmFormObject
Desc:		This class is the virtual base class for objects within a form.
===========================================================================*/
class FlmFormObject : public F_Base
{
public:

	FlmFormObject( void);
	virtual ~FlmFormObject( void);

	// Pure virtual methods that must be implemented by classes that
	// implement this base class.

	virtual void display(
		FLMUINT		uiDisplayRow,				// Row in window where object is currently positioned
		FLMUINT		uiDisplayColumn) = 0;	// Column in window where object is currently positioned

	virtual void formatEditBuffer(			// Formats the current value into a buffer that will be
		char *		pszEditBuf) = 0;			// used for editing.

	virtual RCODE populateReturnAddress(	// Populates a return address, if any - when the form
		void) = 0;									// is exited.

	virtual RCODE populateReturnPath(		// Populates a return path in a GEDCOM tree, if any
		POOL *	pPool,
		NODE *	pTree) = 0;

	// Other methods

	void outputText(								// Used by display functions to output text after
		char *			pszText,					// they have formatted it.
		FLMUINT			uiRow,
		FLMUINT			uiColumn);

	RCODE setHelp(									// Set help lines for an object.  These will be
		const char *	pszHelpLine1,			// displayed in the status bar when there is not
		const char *	pszHelpLine2);			// an error.

	void getHelp(									// Get help lines for an object.
		char *			pszHelpLine1,
		FLMUINT			uiHelpSize1,
		char *			pszHelpLine2,
		FLMUINT			uiHelpSize2);

	inline void setReturnAddress(
		void *		pvReturnAddress,
		FLMUINT *	puiReturnLen)
	{
		m_pvReturnAddress = pvReturnAddress;
		m_puiReturnLen = puiReturnLen;
	}

	inline void setReturnPath(
		FLMUINT *	puiPath
		)
	{
		FLMUINT	uiCnt = 0;
		while (*puiPath)
		{
			m_uiReturnPath [uiCnt++] = *puiPath++;
		}
		m_uiReturnPath [uiCnt] = 0;
	}

	inline FormObjectType getObjectType( void) {return m_eObjectType;}
	inline FLMUINT getObjectId( void) {return m_uiObjectId;}
	inline FLMUINT getBackColor( void) {return m_uiBackColor;}
	inline FLMUINT getForeColor( void) {return m_uiForeColor;}
	inline FLMUINT getRow( void) {return m_uiRow;}
	inline FLMUINT getColumn( void) {return m_uiColumn;}
	inline FLMUINT getWidth( void) {return m_uiWidth;}
	inline FLMBOOL isDisplayOnly( void) {return m_bDisplayOnly;}
	inline FlmFormObject * getNextObject( void) {return m_pNextObject;}
	inline FlmFormObject * getPrevObject( void) {return m_pPrevObject;}

	inline void setBackColor( FLMUINT uiBackColor) {m_uiBackColor = uiBackColor;}
	inline void setForeColor( FLMUINT uiForeColor) {m_uiForeColor = uiForeColor;}
	inline void setNextObject( FlmFormObject * pObject) {m_pNextObject = pObject;}
	inline void setPrevObject( FlmFormObject * pObject) {m_pPrevObject = pObject;}
	inline void setDisplayOnly( FLMBOOL bDisplayOnly) {m_bDisplayOnly = bDisplayOnly;}
	inline FLMUINT getMaxEditChars( void) {return m_uiMaxEditChars;}

protected:
	NODE * findPath(
		POOL *	pPool,
		NODE *	pTree);

	FormObjectType		m_eObjectType;		// Type of object
	FLMUINT				m_uiObjectId;		// Object identifier
	FLMUINT				m_uiRow;				// Row of the object within the form
	FLMUINT				m_uiColumn;			// Column of the object within the form
	FLMUINT				m_uiWidth;			// Display width of the object
	FLMUINT				m_uiFormat;			// Display format
	FLMBOOL				m_bDisplayOnly;	// Is object editable?
	FlmForm *			m_pForm;				// Form that holds this object
	FLMUINT				m_uiMaxEditChars;	// Maximum number of characters to edit.
	void *				m_pvReturnAddress;// Return address to populate when form is exited.
	FLMUINT *			m_puiReturnLen;	// Return length to populate when form is exited.
	FLMUINT				m_uiReturnPath [8];

private:

	FLMUINT				m_uiBackColor;
	FLMUINT				m_uiForeColor;
	FlmFormObject *	m_pNextObject;		// Next object in the form.
	FlmFormObject *	m_pPrevObject;		// Previous object in the form.
	FLMUINT				m_uiHelpBuffSize;
	char *				m_pszHelpLine1;	
	char *				m_pszHelpLine2;
};

/*===========================================================================
Class:	FlmFormTextObject
Desc:		This is the class for text objects within a form.
===========================================================================*/
class FlmFormTextObject : public FlmFormObject
{
public:
	FlmFormTextObject( void);
	~FlmFormTextObject( void);

	RCODE setup(
		FlmForm *	pForm,
		FLMUINT		uiObjectId,
		FLMUINT		uiMaxChars,
		FLMUINT		uiWidth,
		FLMUINT		uiFormat,
		FLMBOOL		bDisplayOnly,
		FLMUINT		uiBackColor,
		FLMUINT		uiForeColor,
		FLMUINT		uiRow,
		FLMUINT		uiColumn);

	RCODE setValue(
		const char *	pszValue);

	RCODE getValue(
		FLMUINT *	puiLen,
		char *		pszValue);

	virtual void display(
		FLMUINT		uiDisplayRow,
		FLMUINT		uiDisplayColumn);

	virtual void formatEditBuffer(
		char *		pszEditBuf);

	virtual RCODE populateReturnAddress( void);

	virtual RCODE populateReturnPath(
		POOL *	pPool,
		NODE *	pTree);

private:

	char *		m_pszValue;
};

/*===========================================================================
Class:	FlmFormUnsignedObject
Desc:		This is the class for unsigned number objects within a form.
===========================================================================*/
class FlmFormUnsignedObject : public FlmFormObject
{
public:
	FlmFormUnsignedObject( void);
	~FlmFormUnsignedObject( void);

	RCODE setup(
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
		FLMUINT		uiColumn);

	RCODE setValue(
		FLMUINT		uiValue);

	RCODE getValue(
		FLMUINT *	puiValue);

	FLMBOOL verifyValue(
		char *		pszEditBuf,
		FLMUINT *	puiValue);

	virtual void display(
		FLMUINT		uiDisplayRow,
		FLMUINT		uiDisplayColumn);

	virtual void formatEditBuffer(
		char *		pszEditBuf);

	virtual RCODE populateReturnAddress(
		void);

	virtual RCODE populateReturnPath(
		POOL *	pPool,
		NODE *	pTree);

	inline FLMUINT getMin( void) {return m_uiMin;}
	inline FLMUINT getMax( void) {return m_uiMax;}
private:
	FLMUINT			m_uiValue;			// Current value in the object
	FLMUINT			m_uiMin;
	FLMUINT			m_uiMax;
};

/*===========================================================================
Class:	FlmFormSignedObject
Desc:		This is the class for signed number objects within a form.
===========================================================================*/
class FlmFormSignedObject : public FlmFormObject
{
public:
	FlmFormSignedObject( void);
	~FlmFormSignedObject( void);

	RCODE setup(
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
		FLMUINT		uiColumn);

	RCODE setValue(
		FLMINT		iValue);

	RCODE getValue(
		FLMINT *	piValue);

	FLMBOOL verifyValue(
		char *		pszEditBuf,
		FLMINT *		piValue);

	virtual void display(
		FLMUINT		uiDisplayRow,
		FLMUINT		uiDisplayColumn);

	virtual void formatEditBuffer(
		char *		pszEditBuf);

	virtual RCODE populateReturnAddress(
		void);

	virtual RCODE populateReturnPath(
		POOL *	pPool,
		NODE *	pTree);

	inline FLMINT getMin( void) {return m_iMin;}
	inline FLMINT getMax( void) {return m_iMax;}
private:
	FLMINT			m_iValue;			// Current value in the object
	FLMINT			m_iMin;
	FLMINT			m_iMax;
};

/*===========================================================================
Class:	FlmFormRecordObject
Desc:		This is the class for GEDCOM record(s) objects within a form.
===========================================================================*/
class FlmFormRecordObject : public FlmFormObject
{
public:
	FlmFormRecordObject( void);
	~FlmFormRecordObject( void);

	RCODE setup(
		FlmForm *		pForm,
		FLMUINT			uiObjectId,
		const char *	pszTitle,
		FLMUINT			uiWidth,
		FLMUINT			uiBackColor,
		FLMUINT			uiForeColor,
		FLMUINT			uiRow,
		FLMUINT			uiColumn);

	inline void setDb(HFDB	hDb){ m_hDb = hDb;}
	inline HFDB getDb( void){ return m_hDb;}
	inline void setContainer(FLMUINT uiContainer){ m_uiContainer = uiContainer;}
	inline FLMUINT getContainer( void){ return m_uiContainer;}

	RCODE setValue(
		NODE *		pRecordValue);

	RCODE getValue(
		NODE * *		ppRecordValue,
		POOL *		pPool);

	virtual void display(
		FLMUINT		uiDisplayRow,
		FLMUINT		uiDisplayColumn);

	FLMBOOL verifyValue(
		char *		pszEditBuf,
		F_TIME *		pTimeValue);

	virtual void formatEditBuffer(
		char *		pszEditBuf);

	virtual RCODE populateReturnAddress( void);

	virtual RCODE populateReturnPath(
		POOL *	pPool,
		NODE *	pTree);

	FLMUINT edit(
		FLMBOOL *	pbChanged);

	inline void setEnterMode(
		FLMBOOL	bAutoEnter)
	{
		m_bAutoEnter = bAutoEnter;
	}

	inline FLMBOOL isAutoEnterMode( void)
	{
		return m_bAutoEnter;
	}

private:

	char *		m_pszTitle;
	FLMBOOL		m_bAutoEnter;
	HFDB			m_hDb;
	FLMUINT		m_uiContainer;
	POOL			m_pool;
	NODE *		m_pRecord;
	FLMUINT		m_uiFirstNode;
	FLMUINT		m_uiCurrNode;
};

/*===========================================================================
Class:	FlmFormPulldownObject
Desc:		This class is the class for pulldown objects within a form.
===========================================================================*/
class FlmFormPulldownObject : public FlmFormObject
{
public:
	FlmFormPulldownObject( void);
	~FlmFormPulldownObject( void);

	RCODE setup(
		FlmForm *	pForm,
		FLMUINT		uiObjectId,
		FLMUINT		uiWidth,				// Box width
		FLMUINT		uiHeight,			// Box height
		FLMUINT		uiBackColor,
		FLMUINT		uiForeColor,
		FLMUINT		uiRow,
		FLMUINT		uiColumn);

	RCODE setCurrentItem(
		FLMUINT		uiItemId);

	RCODE getCurrentItem(
		FLMUINT *	puiItemId);

	RCODE addItem(
		FLMUINT			uiItemId,
		const char *	pszDisplayValue,
		FLMUINT			uiShortcutKey);

	RCODE removeItem(
		FLMUINT		uiItemId);

	RCODE clearItems( void);

	virtual void display(
		FLMUINT		uiDisplayRow,
		FLMUINT		uiDisplayColumn);

	virtual void formatEditBuffer(
		char *		pszEditBuf);

	virtual RCODE populateReturnAddress(
		void);

	virtual RCODE populateReturnPath(
		POOL *		pPool,
		NODE *		pTree);

	FLMUINT select( void);

	inline void setEnterMode(
		FLMBOOL	bAutoEnter)
	{
		m_bAutoEnter = bAutoEnter;
	}

	inline FLMBOOL isAutoEnterMode( void)
	{
		return m_bAutoEnter;
	}

	void setPulldownInsertCallback(
		INSERT_FUNC_p	pCallback,
		void *			pvAppData);

	inline void setPulldownReturnAll(
		FLMUINT	uiItemIdTag,
		FLMUINT	uiItemNameTag
		)
	{
		m_bReturnAll = TRUE;
		m_uiItemIdTag = uiItemIdTag;
		m_uiItemNameTag = uiItemNameTag;
	}

	inline FlmPulldownList * getPulldownObject( void)
	{
		return( m_pPulldown);
	}

private:
	FlmPulldownList *	m_pPulldown;
	FLMBOOL				m_bAutoEnter;
	FLMBOOL				m_bReturnAll;
	FLMUINT				m_uiItemIdTag;
	FLMUINT				m_uiItemNameTag;
};

/*===========================================================================
Struct:	FlmPulldownItem
Desc:		This structure is for the individual items in a pulldown list.
===========================================================================*/
typedef struct FlmPulldownItemTag *	FlmPulldownItem_p;

typedef struct FlmPulldownItemTag
{
	FLMUINT				uiItemId;
	FLMUINT				uiItemNumber;
	char *				pszDisplayValue;
	FLMUINT				uiDisplayValueLen;
	FLMUINT				uiShortcutKey;
	FlmPulldownList *	pSubList;
	ITEM_FUNC_p			pFunc;
	void *				pvAppData;
	FlmPulldownItem_p	pNext;
	FlmPulldownItem_p	pPrev;
} FlmPulldownItem;

/*===========================================================================
Struct:	FlmShortcutKey
Desc:		This structure is to remember shortcut keys.
===========================================================================*/
typedef struct FlmShortcutKeyTag
{
	FLMUINT				uiShortcutKey;
	FlmPulldownItem_p	pItem;
} FlmShortcutKey;

/*===========================================================================
Class:	FlmPulldownList
Desc:		This class is the class for pulldown objects.
===========================================================================*/
class FlmPulldownList : public F_Base
{
public:
	FlmPulldownList( void);
	~FlmPulldownList( void);

	RCODE setTitle(
		const char *	pszListTitle,
		FLMUINT			uiBackColor,
		FLMUINT			uiForeColor);

	RCODE setHelp(
		const char *	pszListHelp,
		FLMUINT			uiBackColor,
		FLMUINT			uiForeColor);

	RCODE setCurrentItem(
		FLMUINT		uiItemId);

	RCODE getCurrentItem(
		FLMUINT *	puiItemId);

	RCODE getFirstItem(
		FLMUINT *	puiItemId);

	RCODE getLastItem(
		FLMUINT *	puiItemId);

	RCODE getNextItem(
		FLMUINT		uiItemId,
		FLMUINT *	puiNextItemId);

	RCODE getPrevItem(
		FLMUINT		uiItemId,
		FLMUINT *	puiPrevItemId);

	RCODE getItemDispValue(
		FLMUINT		uiItemId,
		FLMUINT *	puiDispBufLen,
		char *		pszDisplayValue);

	RCODE setDisplayValue(
		FLMUINT			uiItemId,
		const char *	pszDisplayValue);

	RCODE addItem(
		FLMUINT				uiItemId,
		const char *		pszDisplayValue,
		FLMUINT				uiShortcutKey,
		FlmPulldownList *	pSubList = NULL,
		ITEM_FUNC_p			pFunc = NULL,
		void *				pvAppData = NULL);

	RCODE insertItem(
		FLMUINT				uiPositionItemId,
		FLMBOOL				bInsertBefore,
		FLMUINT				uiItemId,
		const char *		pszDisplayValue,
		FLMUINT				uiShortcutKey,
		FlmPulldownList *	pSubList = NULL,
		ITEM_FUNC_p			pFunc = NULL,
		void *				pvAppData = NULL);

	RCODE addShortcutKey(
		FLMUINT				uiItemId,
		FLMUINT				uiShortcutKey);

	void removeShortcutKey(
		FLMUINT				uiShortcutKey);

	RCODE removeItem(
		FLMUINT		uiItemId);

	RCODE clearItems( void);

	void positionTo(
		FLMUINT	uiItemId);

	void refresh( void);

	void calcEditLocation(
		FLMUINT		uiScreenRows,				// Number of rows on the screen
		FLMUINT		uiScreenCols,				// Number of coluns on the screen
		FLMUINT		uiAnchorRow,				// Anchor row.
		FLMUINT		uiLeftAnchorCol,			// Left anchor column.
		FLMUINT		uiAnchorWidth,				// Anchor width.
		FLMUINT *	puiBoxWidth,				// Returns box width
		FLMUINT *	puiBoxHeight,				// Returns box height
		FLMUINT *	puiUpperLeftCol,			// Returns upper left column
		FLMUINT *	puiUpperLeftRow);			// Returns upper left row

	RCODE interact(
		FTX_SCREEN_p			pScreen,
		FlmThreadContext *	pThread,
		FLMUINT					uiWidth,				// Width of box - counting border
		FLMUINT					uiHeight,			// Height of box - counting border
		FLMBOOL					bDoBorder,			// Do a border on box?
		FLMUINT					uiULX,				// Upper Left corner X position
		FLMUINT					uiULY,				// Upper Left corner Y position
		FLMBOOL					bReturnOnShortcut,// Return when shortcut key is pressed?
		FLMUINT					uiResponseTimeout,// Number of seconds to wait before exiting
		FLMUINT *				puiExitChar,		// Exit character.
		FLMUINT *				puiExitValue,		// Value positioned on when exiting
		FLMUINT					uiExitValueDepth,	// Depth that list can reach.
		FLMBOOL *				pbRedisplay,		// Recalculate box location?
		LIST_KEY_FUNC_p		pKeyFunc,			// Key handler.
		void *					pvAppData);			// App data to key handler.

	inline void setBackColor( FLMUINT uiBackColor) {m_uiBackColor = uiBackColor;}
	inline void setForeColor( FLMUINT uiForeColor) {m_uiForeColor = uiForeColor;}
	inline FLMUINT getBackColor( void) {return m_uiBackColor;}
	inline FLMUINT getForeColor( void) {return m_uiForeColor;}
	inline FlmThreadContext * getThread( void){ return m_pThread;}
	inline FTX_WINDOW_p getWindow( void) { return( m_pWindow); }

	inline FLMUINT itemCount( void)
	{
		return( (FLMUINT)((m_pLastItem)
								? m_pLastItem->uiItemNumber
								: (FLMUINT)0));
	}

	inline FLMUINT maxWidth( void)
	{
		return( m_uiMaxTypedownChars);
	}

	inline void setPulldownInsertCallback(
		INSERT_FUNC_p	pCallback,
		void *			pvAppData)
	{
		m_pInsertFunc = pCallback;
		m_pvAppData = pvAppData;
	}

private:

	FlmPulldownItem *	findItem(
		FLMUINT	uiItemId);

	void recalcMaxWidth( void);

	FlmShortcutKey * findShortcutKey(
		FLMUINT				uiShortcutKey);

	RCODE addShortcutKey(
		FLMUINT				uiShortcutKey,
		FlmPulldownItem *	pItem);

	void displayItem(
		FlmPulldownItem *	pItem,
		FLMBOOL				bIsCurrentItem);

	void positionTo(
		FlmPulldownItem *	pItem,
		FLMBOOL				bForceRefresh);

	void cursorDown( void);
	void cursorUp( void);
	void scrollLeft( void);
	void scrollRight( void);
	void pageDown( void);
	void pageUp( void);
	void cursorHome( void);
	void cursorEnd( void);
	void backspaceChar( void);

	void typedown(
		FLMUINT	uiChar);

	FLMBOOL shortcutKey(
		FLMUINT	uiChar);

	FLMBOOL					m_bMonochrome;
	FLMBOOL					m_bInteracting;
	FLMUINT					m_uiBackColor;
	FLMUINT					m_uiForeColor;
	FlmPulldownItem *		m_pFirstItem;
	FlmPulldownItem *		m_pLastItem;
	FlmPulldownItem *		m_pCurrentItem;
	FlmPulldownItem *		m_pTopItem;
	FlmShortcutKey *		m_pShortcutKeys;
	FlmThreadContext *	m_pThread;
	FLMUINT					m_uiShortcutKeyArraySize;
	FLMUINT					m_uiNumShortcutKeys;
	FTX_WINDOW_p			m_pWindow;
	FLMUINT					m_uiRows;
	FLMUINT					m_uiCols;
	char *					m_pszTypedownBuf;
	FLMUINT					m_uiTypedownBufSize;
	FLMUINT					m_uiMaxTypedownChars;
	FLMUINT					m_uiNumTypedownChars;
	FLMUINT					m_uiDispOffset;
	char *					m_pszListTitle;
	FLMUINT					m_uiListTitleBufSize;
	FLMUINT					m_uiListTitleBackColor;
	FLMUINT					m_uiListTitleForeColor;
	char *					m_pszListHelp;
	FLMUINT					m_uiListHelpBufSize;
	FLMUINT					m_uiListHelpBackColor;
	FLMUINT					m_uiListHelpForeColor;
	INSERT_FUNC_p			m_pInsertFunc;
	void *					m_pvAppData;
};

#endif
