//-------------------------------------------------------------------------
// Desc:	View the roll-forward log.
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: viewrfl.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "ftx.h"
#include "sharutil.h"
#include "flm_edit.h"
#include "flmarg.h"
#include "fform.h"

#ifdef FLM_NLM
	extern "C"
	{
		FLMBOOL	gv_bSynchronized = FALSE;

		void SynchronizeStart();

		int nlm_main(
			int		ArgC,
			char **	ArgV);

		int atexit( void (*)( void ) );
	}

	FSTATIC void viewRflCleanup( void);
#endif

#define MAIN_MODULE
#include "rflread.h"

#define SRCH_LABEL_COLUMN	5
#define SRCH_ENTER_COLUMN	38

#define SRCH_PACKET_TYPE_TAG	1
#define SRCH_TRANS_ID_TAG		2
#define SRCH_CONTAINER_TAG		3
#define SRCH_INDEX_TAG			4
#define SRCH_DRN_TAG				5
#define SRCH_END_DRN_TAG		6
#define SRCH_MULTI_FILE_TAG	7

FSTATIC void viewRflFormatSerialNum(
	FLMBYTE *	pszBuf,
	FLMBYTE *	pucSerialNum);

FSTATIC RCODE viewRflShowHeader(
	F_RecEditor *		pParentEditor);

FSTATIC RCODE viewRflHeaderDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals);

FSTATIC RCODE viewRflGetEOF( void);

FSTATIC RCODE rflOpenNewFile(
	F_RecEditor *		pRecEditor,
	const char *		pszFileName,
	FLMBOOL				bPosAtBOF,
	POOL *				pTmpPool,
	NODE **				ppNd);

/*
NetWare hooks
*/

// Data

typedef struct RflTagName
{
	const char *	pszTagName;
	FLMUINT			uiTagNum;
} RFL_TAG_NAME;

RFL_TAG_NAME RflDictTags[] =
{
	{ "TransBegin", RFL_TRNS_BEGIN_FIELD},
	{ "TransCommit", RFL_TRNS_COMMIT_FIELD},
	{ "TransAbort", RFL_TRNS_ABORT_FIELD},
	{ "RecAdd", RFL_RECORD_ADD_FIELD},
	{ "RecModify", RFL_RECORD_MODIFY_FIELD},
	{ "RecDelete", RFL_RECORD_DELETE_FIELD},
	{ "ReserveDRN", RFL_RESERVE_DRN_FIELD},
	{ "ChangeFields", RFL_CHANGE_FIELDS_FIELD},
	{ "DataRecord", RFL_DATA_RECORD_FIELD},
	{ "IndexNum", RFL_INDEX_NUM_FIELD},
	{ "StartDRN", RFL_START_DRN_FIELD},
	{ "EndDRN", RFL_END_DRN_FIELD},
	{ "UnknownPacket", RFL_UNKNOWN_PACKET_FIELD},
	{ "NumBytesValid", RFL_NUM_BYTES_VALID_FIELD},
	{ "PacketAddress", RFL_PACKET_ADDRESS_FIELD},
	{ "PacketChecksum", RFL_PACKET_CHECKSUM_FIELD},
	{ "PacketChecksumValid", RFL_PACKET_CHECKSUM_VALID_FIELD},
	{ "PacketBodyLength", RFL_PACKET_BODY_LENGTH_FIELD},
	{ "NextPacketAddress", RFL_NEXT_PACKET_ADDRESS_FIELD},
	{ "PrevPacketAddress", RFL_PREV_PACKET_ADDRESS_FIELD},
	{ "TransID", RFL_TRANS_ID_FIELD},
	{ "StartSeconds", RFL_START_SECONDS_FIELD},
	{ "StartMillisec", RFL_START_MSEC_FIELD},
	{ "EndSeconds", RFL_END_SECONDS_FIELD},
	{ "EndMillisec", RFL_END_MSEC_FIELD},
	{ "StartTransAddr", RFL_START_TRNS_ADDR_FIELD},
	{ "ContainerNum", RFL_CONTAINER_FIELD},
	{ "RecordID", RFL_DRN_FIELD},
	{ "TagNum", RFL_TAG_NUM_FIELD},
	{ "FieldType", RFL_TYPE_FIELD},
	{ "Level", RFL_LEVEL_FIELD},
	{ "DataLen", RFL_DATA_LEN_FIELD},
	{ "Data", RFL_DATA_FIELD},
	{ "MoreData", RFL_MORE_DATA_FIELD},
	{ "InsertFld", RFL_INSERT_FLD_FIELD},
	{ "ModifyFld", RFL_MODIFY_FLD_FIELD},
	{ "DeleteFld", RFL_DELETE_FLD_FIELD},
	{ "EndChanges", RFL_END_CHANGES_FIELD},
	{ "UnknownChangeType", RFL_UNKNOWN_CHANGE_TYPE_FIELD},
	{ "Position", RFL_POSITION_FIELD},
	{ "ReplaceBytes", RFL_REPLACE_BYTES_FIELD},
	{ "UnknownChangeBytes", RFL_UNKNOWN_CHANGE_BYTES_FIELD},
	{ "IndexSet", RFL_INDEX_SET_FIELD},
	{ "IndexSet2", RFL_INDEX_SET2_FIELD},
	{ "StartUnknown", RFL_START_UNKNOWN_FIELD},
	{ "UserUnknown", RFL_UNKNOWN_USER_PACKET_FIELD},
	{ "RFLName", RFL_HDR_NAME_FIELD},
	{ "RFLVersion", RFL_HDR_VERSION_FIELD},
	{ "FileNumber", RFL_HDR_FILE_NUMBER_FIELD},
	{ "FileEOF", RFL_HDR_EOF_FIELD},
	{ "DBSerialNum", RFL_HDR_DB_SERIAL_NUM_FIELD},
	{ "FileSerialNum", RFL_HDR_FILE_SERIAL_NUM_FIELD},
	{ "NextFileSerialNum", RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD},
	{ "KeepSignature", RFL_HDR_KEEP_SIGNATURE_FIELD},
	{ "TransBeginEx", RFL_TRNS_BEGIN_EX_FIELD},
	{ "UpgradeDB", RFL_UPGRADE_PACKET_FIELD},
	{ "OldDbVersion", RFL_OLD_DB_VERSION_FIELD},
	{ "NewDbVersion", RFL_NEW_DB_VERSION_FIELD},
	{ "ReduceDb", RFL_REDUCE_PACKET_FIELD},
	{ "BlockCount", RFL_BLOCK_COUNT_FIELD},
	{ "LastCommitTransID", RFL_LAST_COMMITTED_TRANS_ID_FIELD},
	{ "IndexSuspend", RFL_INDEX_SUSPEND_FIELD},
	{ "IndexResume", RFL_INDEX_RESUME_FIELD},
	{ "BlockChainFree", RFL_BLK_CHAIN_FREE_FIELD},
	{ "TrackerRecDRN", RFL_TRACKER_REC_FIELD},
	{ "EndBlockAddr", RFL_END_BLK_ADDR_FIELD},
	{ "Flags", RFL_FLAGS_FIELD},
	{ "InsertEncrypted",  RFL_INSERT_ENC_FLD_FIELD},
	{ "ModifyEncrypted",  RFL_MODIFY_ENC_FLD_FIELD},
	{ "Encrypted", RFL_ENC_FIELD},
	{ "EncryptedDefId", RFL_ENC_DEF_ID_FIELD},
	{ "EncryptedDataLen", RFL_ENC_DATA_LEN_FIELD},
	{ "DataBaseKeyLen", RFL_DB_KEY_LEN_FIELD},
	{ "DataBaseKey", RFL_DB_KEY_FIELD},
	{ "WrapKey", RFL_WRAP_KEY_FIELD},
	{ "EnableEncryption", RFL_ENABLE_ENCRYPTION_FIELD},
	{ NULL, 0}
};

// Local Prototypes

void UIMain(
	int			ArgC,
	char **		ArgV);

RCODE viewRflMainKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE viewRflMainHelpHook(
	F_RecEditor *		pRecEditor,
	F_RecEditor *		pHelpEditor,
	POOL *				pPool,
	void *				UserData,
	NODE **				ppRootNd);

RCODE viewRflMainEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData);

RCODE viewRflInspectEntry(
	F_RecEditor *		pParentEditor);

RCODE viewRflInspectEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData);

RCODE viewRflInspectDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals);

RCODE viewRflInspectKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut);

RCODE viewRflNameTableInit(
	F_NameTable **		ppNameTable);

FSTATIC RCODE addLabel(
	FlmForm *			pForm,
	FLMUINT				uiObjectId,
	const char *		pszLabel,
	FLMUINT				uiRow);

FSTATIC FLMBOOL editSearchFormCB(
	FormEventType		eFormEvent,
	FlmForm *			pForm,
	FlmFormObject *	pFormObject,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvAppData);

FSTATIC RCODE getSearchCriteria(
	F_RecEditor *	pRecEditor,
	RFL_PACKET *	pSrchCriteria,
	FLMBOOL *		pbForward);

FSTATIC FLMBOOL rflPassesCriteria(
	RFL_PACKET *	pPacket,
	RFL_PACKET *	pSrchPacket);

/*--------------------------------------------------------
** Local (to this file only) global variables.
**-------------------------------------------------------*/
RFL_PACKET						gv_SrchCriteria;
FLMBOOL							gv_bSrchForward;
FLMBOOL							gv_bDoRefresh = TRUE;
FLMBOOL							gv_bShutdown = FALSE;
FTX_INFO *						gv_pFtxInfo = NULL;
const char *					gv_pucTitle = "FLAIM RFL Viewer v1.00";
char								gv_szRflPath [F_PATH_MAX_SIZE];
static F_NameTable *			gv_pNameTable = NULL;
#ifdef FLM_NLM
	static FLMBOOL				gv_bRunning = TRUE;
#endif


/****************************************************************************
Name: main
****************************************************************************/
#if defined( FLM_UNIX)
int main(
	int			ArgC,
	char **		ArgV
	)
#elif defined( FLM_NLM)
int nlm_main(
	int			ArgC,
	char **		ArgV
	)
#else
int __cdecl main(
	int			ArgC,
	char **		ArgV
	)
#endif   
{
	int	iResCode = 0;

	if( RC_BAD( FlmStartup()))
	{
		iResCode = -1;
		goto Exit;
	}

#ifdef FLM_NLM

	/* Setup the routines to be called when the NLM exits itself */
	
	atexit( viewRflCleanup);

#endif

	UIMain( ArgC, ArgV);

Exit:

	FlmShutdown();

#ifdef FLM_NLM
	if (!gv_bSynchronized)
	{
		SynchronizeStart();
		gv_bSynchronized = TRUE;
	}
	gv_bRunning = FALSE;
#endif

	return( iResCode);
}


/****************************************************************************
Name: UIMain
****************************************************************************/
void UIMain(
	int			iArgC,
	char **		ppszArgV
	)
{
	FTX_SCREEN *		pScreen = NULL;
	FTX_WINDOW *		pTitleWin = NULL;
	F_RecEditor	*		pRecEditor = NULL;
	FLMUINT				uiTermChar;
	RCODE					rc = FERR_OK;

	gv_pRflFileHdl = NULL;
	gv_uiRflEof = 0;
	f_memset( &gv_SrchCriteria, 0, sizeof( gv_SrchCriteria));
	gv_bSrchForward = TRUE;
	gv_SrchCriteria.uiPacketType = 0xFFFFFFFF;
	gv_SrchCriteria.uiMultiFileSearch = 1;

	if( FTXInit( gv_pucTitle, (FLMUINT)80, (FLMUINT)50,
		WPS_BLUE, WPS_WHITE, NULL, NULL,
		&gv_pFtxInfo) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	FTXSetShutdownFlag( gv_pFtxInfo, &gv_bShutdown);

	if( FTXScreenInit( gv_pFtxInfo, gv_pucTitle, &pScreen) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( FTXWinInit( pScreen, 0, 1, &pTitleWin) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( FTXWinPaintBackground( pTitleWin, WPS_RED) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( FTXWinPrintStr( pTitleWin, gv_pucTitle) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( FTXWinSetCursorType( pTitleWin,
		WPS_CURSOR_INVISIBLE) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( FTXWinOpen( pTitleWin) != FTXRC_SUCCESS)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (pRecEditor = new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( pRecEditor->Setup( pScreen)))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRecEditor->setTree( NULL);
	pRecEditor->setShutdown( &gv_bShutdown);
	pRecEditor->setKeyHook( viewRflMainKeyHook, 0);
	pRecEditor->setHelpHook( viewRflMainHelpHook, 0);
	pRecEditor->setEventHook( viewRflMainEventHook, (void *)0);

	/*
	Fire up the editor
	*/

	gv_szRflPath [0] = 0;
	if (iArgC > 1)
	{
		f_strcpy( gv_szRflPath, ppszArgV [1]);
	}

	if (!gv_szRflPath [0])
	{
		pRecEditor->requestInput(
				"Log File Name", gv_szRflPath,
				sizeof( gv_szRflPath), &uiTermChar);

		if( uiTermChar == WPK_ESCAPE)
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = pRecEditor->getFileSystem()->Open( gv_szRflPath,
			  			F_IO_RDWR | F_IO_SH_DENYNONE, &gv_pRflFileHdl)))
	{
		pRecEditor->displayMessage( "Unable to open file", rc,
			NULL, WPS_RED, WPS_WHITE);
		rc = FERR_OK;
	}
	else
	{
		viewRflNameTableInit( &gv_pNameTable);
		pRecEditor->setTitle( gv_szRflPath);
		pRecEditor->interactiveEdit( 0, 1);
		pRecEditor->setTree( NULL);
		if( gv_pNameTable)
		{
			gv_pNameTable->Release();
			gv_pNameTable = NULL;
		}
		gv_pRflFileHdl->Release();
		gv_pRflFileHdl = NULL;
	}

Exit:

	gv_bShutdown = TRUE;

	if( pRecEditor)
	{
		pRecEditor->Release();
		pRecEditor = NULL;
	}

	if( gv_pRflFileHdl)
	{
		gv_pRflFileHdl->Release();
	}

	FTXFree( &gv_pFtxInfo);
}


#ifdef FLM_NLM
/****************************************************************************
Desc: This routine shuts down all threads in the NLM.
****************************************************************************/
void viewRflCleanup( void)
{
	gv_bShutdown = TRUE;
	while( gv_bRunning)
	{
		f_sleep( 10);
		f_yieldCPU();
	}
}
#endif


/********************************************************************
Desc: Add a label to a form.
*********************************************************************/
FSTATIC RCODE addLabel(
	FlmForm *		pForm,
	FLMUINT			uiObjectId,
	const char *	pszLabel,
	FLMUINT			uiRow)
{
	FLMUINT	uiLen = f_strlen( pszLabel);

	return( pForm->addTextObject( uiObjectId, pszLabel,
		uiLen, uiLen,
		0, TRUE, WPS_BLUE, WPS_WHITE,
		uiRow, SRCH_LABEL_COLUMN));
}

/****************************************************************************
Desc:	Callback function for search form.
*****************************************************************************/
FSTATIC FLMBOOL editSearchFormCB(
	FormEventType		eFormEvent,
	FlmForm *			pForm,
	FlmFormObject *	pFormObject,
	FLMUINT				uiKeyIn,
	FLMUINT *			puiKeyOut,
	void *				pvAppData
	)
{
	F_UNREFERENCED_PARM( pForm);
	F_UNREFERENCED_PARM( pFormObject);
	F_UNREFERENCED_PARM( puiKeyOut);
	F_UNREFERENCED_PARM( pvAppData);

	if (eFormEvent == FORM_EVENT_KEY_STROKE)
	{
		switch (uiKeyIn)
		{
			case WPK_F1:
			case WPK_F2:
			case WPK_F3:
			case WPK_F4:
			case WPK_F5:
			case WPK_F6:
			case WPK_F7:
			case WPK_F8:
			case WPK_F9:
			case WPK_F10:
			case WPK_F11:
			case WPK_F12:
				return( FALSE);
			default:
				return( TRUE);
		}
	}
	return( TRUE);
}

/********************************************************************
Desc: Add a label to a form.
*********************************************************************/
FSTATIC RCODE getSearchCriteria(
	F_RecEditor *	pRecEditor,
	RFL_PACKET *	pSrchCriteria,
	FLMBOOL *		pbForward
	)
{
	RCODE				rc = FERR_OK;
	FTX_SCREEN *	pScreen = pRecEditor->getScreen();
	FlmForm *		pForm = NULL;
	FLMUINT			uiRow = 1;
	FLMUINT			uiScreenCols;
	FLMUINT			uiScreenRows;
	FLMUINT			uiChar = 0;
	FLMBOOL			bValuesChanged;
	FLMUINT			uiCurrObjectId;
	const char *	pszWhat = NULL;

	if (FTXScreenGetSize( pScreen,
			&uiScreenCols, &uiScreenRows) != FTXRC_SUCCESS)
	{
		pszWhat = "getting screen size";
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if ((pForm = new FlmForm) == NULL)
	{
		pszWhat = "allocating form";
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pForm->init( pScreen, NULL,
						"Search Criteria",
						WPS_BLUE, WPS_WHITE,
						"ESC=Quit, F1=search forward, other=search backward",
						WPS_BLUE, WPS_WHITE,
						0, 0,
						uiScreenCols - 1, uiScreenRows - 1, TRUE, TRUE,
						WPS_BLUE, WPS_LIGHTGRAY)))
	{
		pszWhat = "initializing form";
		goto Exit;
	}

	// Add the packet type selection field.

	pszWhat = "adding packet type";
	if (RC_BAD( rc = addLabel( pForm, SRCH_PACKET_TYPE_TAG + 100,
									"Packet Type", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownObject( SRCH_PACKET_TYPE_TAG,
									20, 10,
									WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_TRNS_BEGIN_PACKET,
								"B=Transaction Begin", (FLMUINT)'B')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_TRNS_BEGIN_EX_PACKET,
								"X=Transaction Begin (Extended)", (FLMUINT)'X')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_TRNS_COMMIT_PACKET,
								"C=Transaction Commit", (FLMUINT)'C')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_TRNS_ABORT_PACKET,
								"A=Transaction Abort", (FLMUINT)'A')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_ADD_RECORD_PACKET,
								"E=Add Record", (FLMUINT)'E')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_MODIFY_RECORD_PACKET,
								"M=Modify Record", (FLMUINT)'M')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_DELETE_RECORD_PACKET,
								"D=Delete Record", (FLMUINT)'D')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_RESERVE_DRN_PACKET,
								"R=Reserve DRN", (FLMUINT)'R')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_CHANGE_FIELDS_PACKET,
								"F=Change Fields", (FLMUINT)'F')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_DATA_RECORD_PACKET,
								"T=Data Record", (FLMUINT)'T')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_INDEX_SET_PACKET,
								"I=Index Set", (FLMUINT)'I')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_INDEX_SET_PACKET_VER_2,
								"2=Index Set2", (FLMUINT)'2')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_BLK_CHAIN_FREE_PACKET,
								"L=Block Chain Free", (FLMUINT)'L')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_START_UNKNOWN_PACKET,
								"S=Start Unknown", (FLMUINT)'S')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_UNKNOWN_PACKET,
								"U=User Unknown", (FLMUINT)'U')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_REDUCE_PACKET,
								"K=Reduce", (FLMUINT)'K')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_UPGRADE_PACKET,
								"G=Upgrade", (FLMUINT)'G')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_INDEX_SUSPEND_PACKET,
								"P=Index Suspend", (FLMUINT)'P')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								RFL_INDEX_RESUME_PACKET,
								"J=Index Resume", (FLMUINT)'J')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_PACKET_TYPE_TAG,
								0xFFFFFFFF,
								"*=All packet types", (FLMUINT)'*')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectValue( SRCH_PACKET_TYPE_TAG,
										(void *)pSrchCriteria->uiPacketType, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_PACKET_TYPE_TAG,
									&pSrchCriteria->uiPacketType, NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the transaction ID field

	pszWhat = "adding transaction ID";
	if (RC_BAD( rc = addLabel( pForm, SRCH_TRANS_ID_TAG + 100,
									"Transaction ID", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_TRANS_ID_TAG,
					pSrchCriteria->uiTransID,
					0, 0xFFFFFFFF, 10,
					0, FALSE, WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_TRANS_ID_TAG,
									&pSrchCriteria->uiTransID, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_TRANS_ID_TAG,
				"0=Match any trans ID, other=Specific trans ID to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the Container field

	pszWhat = "adding container";
	if (RC_BAD( rc = addLabel( pForm, SRCH_CONTAINER_TAG + 100,
									"Container", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_CONTAINER_TAG,
					pSrchCriteria->uiContainer,
					0, 0xFFFF, 5,
					0, FALSE, WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_CONTAINER_TAG,
									&pSrchCriteria->uiContainer, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_CONTAINER_TAG,
				"0=Match any container, other=Specific container to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the index field

	pszWhat = "adding index";
	if (RC_BAD( rc = addLabel( pForm, SRCH_INDEX_TAG + 100,
									"Index", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_INDEX_TAG,
					pSrchCriteria->uiIndex,
					0, 0xFFFF, 5,
					0, FALSE, WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_INDEX_TAG,
									&pSrchCriteria->uiIndex, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_INDEX_TAG,
				"0=Match any index, other=Specific index to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the DRN field

	pszWhat = "adding DRN";
	if (RC_BAD( rc = addLabel( pForm, SRCH_DRN_TAG + 100,
									"DRN", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_DRN_TAG,
					pSrchCriteria->uiDrn,
					0, 0xFFFFFFFF, 10,
					0, FALSE, WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_DRN_TAG,
									&pSrchCriteria->uiDrn, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_DRN_TAG,
				"0=Match any DRN, other=Specific DRN to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the End DRN field

	pszWhat = "adding end DRN";
	if (RC_BAD( rc = addLabel( pForm, SRCH_END_DRN_TAG + 100,
									"End DRN", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addUnsignedObject( SRCH_END_DRN_TAG,
					pSrchCriteria->uiEndDrn,
					0, 0xFFFFFFFF, 10,
					0, FALSE, WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_END_DRN_TAG,
									&pSrchCriteria->uiEndDrn, NULL)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectHelp( SRCH_END_DRN_TAG,
				"0=Match any End DRN, other=Specific End DRN to find",
				NULL)))
	{
		goto Exit;
	}
	uiRow += 2;

	// Add the packet type selection field.

	pszWhat = "adding multi-file flag";
	if (RC_BAD( rc = addLabel( pForm, SRCH_MULTI_FILE_TAG + 100,
									"Search Multiple Files", uiRow)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownObject( SRCH_MULTI_FILE_TAG,
									20, 10,
									WPS_LIGHTGRAY, WPS_RED, uiRow, SRCH_ENTER_COLUMN)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_MULTI_FILE_TAG, 1,
								"Y=Yes", (FLMUINT)'Y')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->addPulldownItem( SRCH_MULTI_FILE_TAG, 2,
								"N=No", (FLMUINT)'N')))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectValue( SRCH_MULTI_FILE_TAG,
									(void *)pSrchCriteria->uiMultiFileSearch, 0)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pForm->setObjectReturnAddress( SRCH_MULTI_FILE_TAG,
									&pSrchCriteria->uiMultiFileSearch, NULL)))
	{
		goto Exit;
	}
	uiRow += 2;


	pForm->setFormEventCB( editSearchFormCB, NULL, TRUE);
	uiChar = pForm->interact( &bValuesChanged, &uiCurrObjectId);

	if (uiChar == WPK_ESC)
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}
	*pbForward = (FLMBOOL)((uiChar == WPK_F1)
								  ? TRUE
								  : FALSE);

	if (RC_BAD( rc = pForm->getAllReturnData()))
	{
		pszWhat = "getting return data";
		goto Exit;
	}

Exit:
	if (RC_BAD( rc) && uiChar != WPK_ESC)
	{
		char	szErrMsg [100];

		f_sprintf( (char *)szErrMsg, "Error %s", pszWhat);
		pRecEditor->displayMessage( szErrMsg, rc,
						NULL, WPS_RED, WPS_WHITE);
	}
	if (pForm)
	{
		pForm->Release();
	}
	return( rc);
}

/****************************************************************************
Desc:	See if a packet passes the search criteria.
*****************************************************************************/
FSTATIC FLMBOOL rflPassesCriteria(
	RFL_PACKET *	pPacket,
	RFL_PACKET *	pSrchPacket
	)
{
	FLMBOOL	bPasses = FALSE;

	if (pSrchPacket->uiPacketType != pPacket->uiPacketType &&
		 pSrchPacket->uiPacketType != 0xFFFFFFFF)
	{
		goto Exit;
	}
	if (pSrchPacket->uiTransID != pPacket->uiTransID &&
		 pSrchPacket->uiTransID != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiIndex != pPacket->uiIndex &&
		 pSrchPacket->uiIndex != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiContainer != pPacket->uiContainer &&
		 pSrchPacket->uiContainer != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiDrn != pPacket->uiDrn &&
		 pSrchPacket->uiDrn != 0)
	{
		goto Exit;
	}
	if (pSrchPacket->uiEndDrn != pPacket->uiEndDrn &&
		 pSrchPacket->uiEndDrn != 0)
	{
		goto Exit;
	}
	bPasses = TRUE;
Exit:
	return( bPasses);
}

/***************************************************************************
Desc: Format a serial number for display.
*****************************************************************************/
FSTATIC void viewRflFormatSerialNum(
	FLMBYTE *	pszBuf,
	FLMBYTE *	pucSerialNum
	)
{
	f_sprintf( (char *)pszBuf,
			"%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
			(unsigned)pucSerialNum[ 0],
			(unsigned)pucSerialNum[ 1],
			(unsigned)pucSerialNum[ 2],
			(unsigned)pucSerialNum[ 3],
			(unsigned)pucSerialNum[ 4],
			(unsigned)pucSerialNum[ 5],
			(unsigned)pucSerialNum[ 6],
			(unsigned)pucSerialNum[ 7],
			(unsigned)pucSerialNum[ 8],
			(unsigned)pucSerialNum[ 9],
			(unsigned)pucSerialNum[ 10],
			(unsigned)pucSerialNum[ 11],
			(unsigned)pucSerialNum[ 12],
			(unsigned)pucSerialNum[ 13],
			(unsigned)pucSerialNum[ 14],
			(unsigned)pucSerialNum[ 15]);
}

/****************************************************************************
Desc:
*****************************************************************************/
FSTATIC RCODE viewRflHeaderDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals)
{
#define LABEL_WIDTH	32
	FLMUINT		uiCol = 0;
	FLMUINT		uiTag = 0;
	FLMUINT		uiLen;
	char *		pszTmp;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	if (!pNd)
	{
		goto Exit;
	}

	uiTag = GedTagNum( pNd);
	if (!pRecEditor->isSystemNode( pNd))
	{

		// Output the tag number.

		pszTmp = (char *)pDispVals [*puiNumVals].pucString;
		switch (uiTag)
		{
			case RFL_HDR_NAME_FIELD:
				f_strcpy( pszTmp, "RFL Name");
				break;
			case RFL_HDR_VERSION_FIELD:
				f_strcpy( pszTmp, "RFL Version");
				break;
			case RFL_HDR_FILE_NUMBER_FIELD:
				f_strcpy( pszTmp, "RFL File Number");
				break;
			case RFL_HDR_EOF_FIELD:
				f_strcpy( pszTmp, "File EOF");
				break;
			case RFL_HDR_DB_SERIAL_NUM_FIELD:
				f_strcpy( pszTmp, "Database Serial Number");
				break;
			case RFL_HDR_FILE_SERIAL_NUM_FIELD:
				f_strcpy( pszTmp, "RFL File Serial Number");
				break;
			case RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD:
				f_strcpy( pszTmp, "Next RFL File Serial Number");
				break;
			case RFL_HDR_KEEP_SIGNATURE_FIELD:
				f_strcpy( pszTmp, "Keep RFL Files Signature");
				break;
			default:
				f_sprintf( pszTmp, "TAG_%u", (unsigned)uiTag);
				break;
		}
		uiLen = f_strlen( pszTmp);
		if (uiLen < LABEL_WIDTH)
		{
			f_memset( &pszTmp [uiLen], '.', LABEL_WIDTH - uiLen);
		}
		pszTmp [LABEL_WIDTH] = ' ';
		pszTmp [LABEL_WIDTH + 1] = 0;
		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = WPS_WHITE;
		pDispVals[ *puiNumVals].uiBackground = WPS_BLUE;
		(*puiNumVals)++;
		uiCol += (LABEL_WIDTH + 1);

		// Output the value.

		pDispVals[ *puiNumVals].uiCol = uiCol;
		pDispVals[ *puiNumVals].uiForeground = WPS_YELLOW;
		pDispVals[ *puiNumVals].uiBackground = WPS_BLUE;

		(void)pRecEditor->getDisplayValue( pNd,
								F_RECEDIT_DEFAULT_TYPE,
								pDispVals[ *puiNumVals].pucString,
								sizeof( pDispVals[ *puiNumVals].pucString));
		(*puiNumVals)++;
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:	Shows the header of an RFL file.
*****************************************************************************/
FSTATIC RCODE viewRflShowHeader(
	F_RecEditor *		pParentEditor)
{
	F_RecEditor *		pRecEditor;
	NODE *				pHeaderNode;
	NODE *				pNode;
	FLMBYTE				ucHdrBuf [512];
	FLMUINT				uiBytesRead;
	FLMBYTE				szTmp [100];
	FLMUINT				uiTmp;
	POOL					tmpPool;
	RCODE					rc = FERR_OK;

	GedPoolInit( &tmpPool, 1024);

	if( (pRecEditor = new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( pRecEditor->Setup( pParentEditor->getScreen())))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRecEditor->setTree( NULL);
	pRecEditor->setShutdown( &gv_bShutdown);
	pRecEditor->setDisplayHook( viewRflHeaderDispHook, 0);
	pRecEditor->setEventHook( viewRflInspectEventHook, (void *)0);
//	pRecEditor->setKeyHook( viewRflInspectKeyHook, 0);
	pRecEditor->setTitle( "RFL Header");

	// Read the header from the file.

	if (RC_BAD( rc = gv_pRflFileHdl->Read( 0, 512, ucHdrBuf, &uiBytesRead)))
	{
		goto Exit;
	}

	// Create the name field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_NAME_FIELD, 0, &rc)) == NULL)
	{
		goto Exit;
	}
	f_memcpy( szTmp, &ucHdrBuf [RFL_NAME_POS], RFL_NAME_LEN);
	szTmp [RFL_NAME_LEN] = 0;
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	pHeaderNode = pNode;

	// Create the version field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_VERSION_FIELD, 0, &rc)) == NULL)
	{
		goto Exit;
	}
	f_memcpy( szTmp, &ucHdrBuf [RFL_VERSION_POS], RFL_VERSION_LEN);
	szTmp [RFL_VERSION_LEN] = 0;
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the file number field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_FILE_NUMBER_FIELD,
							0, &rc)) == NULL)
	{
		goto Exit;
	}
	uiTmp = (FLMUINT)FB2UD( &ucHdrBuf [RFL_FILE_NUMBER_POS]);
	if (RC_BAD( rc = GedPutUINT( &tmpPool, pNode, uiTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the EOF field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_EOF_FIELD,
							0, &rc)) == NULL)
	{
		goto Exit;
	}
	uiTmp = (FLMUINT)FB2UD( &ucHdrBuf [RFL_EOF_POS]);
	if (RC_BAD( rc = GedPutUINT( &tmpPool, pNode, uiTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the database serial number field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_DB_SERIAL_NUM_FIELD,
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	viewRflFormatSerialNum( szTmp, &ucHdrBuf [RFL_DB_SERIAL_NUM_POS]);
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the file serial number field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_FILE_SERIAL_NUM_FIELD,
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	viewRflFormatSerialNum( szTmp, &ucHdrBuf [RFL_SERIAL_NUM_POS]);
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the next file serial number field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD,
								0, &rc)) == NULL)
	{
		goto Exit;
	}
	viewRflFormatSerialNum( szTmp, &ucHdrBuf [RFL_NEXT_FILE_SERIAL_NUM_POS]);
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode, (char *)szTmp)))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);

	// Create the next file serial number field

	if ((pNode = GedNodeCreate( &tmpPool, RFL_HDR_KEEP_SIGNATURE_FIELD,
								0, &rc)) == NULL)
	{
		goto Exit;
	}

	// Null terminate just in case there is garbage in there.

	ucHdrBuf [RFL_KEEP_SIGNATURE_POS+50] = 0;
	if (RC_BAD( rc = GedPutNATIVE( &tmpPool, pNode,
								(char *)&ucHdrBuf [RFL_KEEP_SIGNATURE_POS])))
	{
		goto Exit;
	}
	GedSibGraft( pHeaderNode, pNode, GED_LAST);


	pRecEditor->setTree( pHeaderNode);
	pRecEditor->interactiveEdit( 0, 1);

Exit:

	if( pRecEditor)
	{
		pRecEditor->Release();
	}

	GedPoolFree( &tmpPool);
	return( rc);
}

/****************************************************************************
Desc:	Determine the RFL file EOF.
*****************************************************************************/
FSTATIC RCODE viewRflGetEOF( void)
{
	RCODE			rc = FERR_OK;
	NODE *		pTmpNd;
	POOL			tmpPool;
	FLMBYTE		ucHdrBuf [512];
	FLMUINT		uiBytesRead;
	FLMUINT		uiEof;

	GedPoolInit( &tmpPool, 4096);

	// First try to get the EOF from the file's header.

	if (RC_BAD( rc = gv_pRflFileHdl->Read( 0, 512, ucHdrBuf, &uiBytesRead)))
	{
		goto Exit;
	}
	uiEof = (FLMUINT)FB2UD( &ucHdrBuf [RFL_EOF_POS]);
	if (uiEof)
	{
		gv_uiRflEof = uiEof;
	}
	else
	{

		// File's header had a zero for the EOF, so try to position to
		// the last node in the file - this should cause us to set
		// the EOF value.

		if (RC_BAD( rc = RflGetPrevNode( NULL, FALSE, &tmpPool, &pTmpNd)))
		{
			goto Exit;
		}

		// If we still didn't get an EOF value, set it to the file size.

		if (!gv_uiRflEof)
		{
			if (RC_BAD( rc = gv_pRflFileHdl->Size( &gv_uiRflEof)))
			{
				goto Exit;
			}
		}
	}
Exit:
	GedPoolFree( &tmpPool);
	return( rc);
}

/****************************************************************************
Desc:	Opens a new RFL file.
*****************************************************************************/
FSTATIC RCODE rflOpenNewFile(
	F_RecEditor *		pRecEditor,
	const char *		pszFileName,
	FLMBOOL				bPosAtBOF,
	POOL *				pTmpPool,
	NODE **				ppNd)
{
	RCODE			rc = FERR_OK;
	F_FileHdl *	pFileHdl = NULL;
	F_FileHdl *	pSaveFileHdl = NULL;
	char			szPath [F_PATH_MAX_SIZE];
	char			szBaseName [F_FILENAME_SIZE];
	char			szPrefix [F_FILENAME_SIZE];
	FLMUINT		uiDbVersion = FLM_VER_4_3;
	FLMUINT		uiFileNum;

	// If no file name was specified, go to the next or previous file from
	// the current file.

	if (!pszFileName || !(*pszFileName))
	{
		if (RC_BAD( rc = f_pathReduce( gv_szRflPath, szPath, szPrefix)))
		{
			goto Exit;
		}

		// See if it is version 4.3 or greater first.

		uiDbVersion = FLM_VER_4_3;
		if (!rflGetFileNum( uiDbVersion, szPrefix, gv_szRflPath, &uiFileNum))
		{
			szPrefix [3] = 0;
			uiDbVersion = FLM_VER_4_0;
			if (!rflGetFileNum( uiDbVersion, szPrefix, gv_szRflPath, &uiFileNum))
			{
				rc = RC_SET( FERR_IO_PATH_NOT_FOUND);
				goto Exit;
			}
		}
		if (bPosAtBOF)
		{
			uiFileNum++;
		}
		else
		{
			uiFileNum--;
		}
		rflGetBaseFileName( uiDbVersion, szPrefix, uiFileNum, szBaseName);
		f_pathAppend( szPath, szBaseName);
		pszFileName = &szPath [0];
	}

	// See if we can open the next file.

	if( RC_BAD( rc = pRecEditor->getFileSystem()->Open( pszFileName,
			  			F_IO_RDWR | F_IO_SH_DENYNONE, &pFileHdl)))
	{
		goto Exit;
	}
	pSaveFileHdl = gv_pRflFileHdl;
	gv_pRflFileHdl = pFileHdl;
	pFileHdl = NULL;
	if (RC_BAD( rc = viewRflGetEOF()))
	{
		goto Exit;
	}
	
	pRecEditor->setTree( NULL);
	if (bPosAtBOF)
	{
		if( RC_BAD( rc = RflGetNextNode( NULL, FALSE, pTmpPool, ppNd)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = RflGetPrevNode( NULL, FALSE, pTmpPool, ppNd)))
		{
			goto Exit;
		}
	}

	pRecEditor->setTree( *ppNd, ppNd);
	pRecEditor->setControlFlags( *ppNd,
			(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
			F_RECEDIT_FLAG_READ_ONLY));
	pSaveFileHdl->Release();
	pSaveFileHdl = NULL;
	f_strcpy( gv_szRflPath, pszFileName);
	pRecEditor->setTitle( gv_szRflPath);
Exit:
	if (pFileHdl)
	{
		pFileHdl->Release();
	}
	if (pSaveFileHdl)
	{
		if (gv_pRflFileHdl)
		{
			gv_pRflFileHdl->Release();
		}
		gv_pRflFileHdl = pSaveFileHdl;
	}
	return( rc);
}

/****************************************************************************
Name:	viewRflMainKeyHook
Desc:	
*****************************************************************************/
RCODE viewRflMainKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	NODE *			pRootNd = NULL;
	NODE *			pTmpNd = NULL;
	NODE *			pNewNd;
	POOL				tmpPool;
	POOL				tmp2Pool;
	FTX_WINDOW *	pWindow = NULL;
	NODE *			pLastNd;
	NODE *			pFirstNd;
	RFL_PACKET *	pPacket;
	FLMBOOL			bSkipCurrent;
	RCODE				rc = FERR_OK;
	char				szResponse[ 80];
	FLMUINT			uiTermChar;
	FLMUINT			uiSrcLen;
	FLMUINT			uiOffset;

	F_UNREFERENCED_PARM( UserData);
	GedPoolInit( &tmpPool, 4096);
	GedPoolInit( &tmp2Pool, 4096);

	if( puiKeyOut)
	{
		*puiKeyOut = 0;
	}

	pRootNd = pRecEditor->getRootNode( pCurNd);
	switch( uiKeyIn)
	{
		case WPK_DOWN:
		case WPK_UP:
		case WPK_PGDN:
		case WPK_PGUP:
		case '?':
		{
			*puiKeyOut = uiKeyIn;
			break;
		}

		case WPK_END:
		{
			FLMUINT		uiLoop;

			pCurNd = NULL;
			pRecEditor->setTree( NULL);
			for( uiLoop = 0; uiLoop < 10; uiLoop++)
			{
				if( RC_BAD( rc = RflGetPrevNode( pCurNd, FALSE,
					&tmpPool, &pNewNd)))
				{
					goto Exit;
				}

				if( pNewNd)
				{
					if( !pCurNd)
					{
						pRecEditor->setTree( pNewNd, &pCurNd);
					}
					else
					{
						pRecEditor->insertRecord( pNewNd, &pCurNd);
					}
					pRecEditor->setControlFlags( pCurNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
				else
				{
					break;
				}
				GedPoolReset( &tmpPool, NULL);
			}
			pRecEditor->setCurrentAtBottom();
			break;
		}

		case WPK_HOME:
		{
			pRecEditor->setTree( NULL);
			if( RC_BAD( rc = RflGetNextNode( NULL, FALSE, &tmpPool, 
				&pTmpNd)))
			{
				goto Exit;
			}

			pRecEditor->setTree( pTmpNd, &pNewNd);
			pRecEditor->setControlFlags( pNewNd,
				(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
				F_RECEDIT_FLAG_READ_ONLY));
			break;
		}

		/*
		View a specific entry
		*/

		case WPK_ENTER:
		{
			viewRflInspectEntry( pRecEditor);
			break;
		}

		case 'h':
		case 'H':
			viewRflShowHeader( pRecEditor);
			break;

		case '0':
		case 'o':
			f_strcpy( szResponse, gv_szRflPath);
			pRecEditor->requestInput(
				"Log File Name",
				szResponse, sizeof( szResponse), &uiTermChar);
			if( uiTermChar == WPK_ESCAPE || !szResponse [0])
			{
				break;
			}
			
			if (RC_BAD( rc = rflOpenNewFile( pRecEditor, szResponse, TRUE,
									&tmpPool, &pTmpNd)))
			{
				pRecEditor->displayMessage( "Unable to open file", rc,
					NULL, WPS_RED, WPS_WHITE);
			}
			break;

		case 'N':
		case 'n':
			if (RC_BAD( rc = rflOpenNewFile( pRecEditor, NULL, TRUE,
									&tmpPool, &pTmpNd)))
			{
				pRecEditor->displayMessage( "Unable to open file", rc,
					NULL, WPS_RED, WPS_WHITE);
			}
			break;

		case 'P':
		case 'p':
			if (RC_BAD( rc = rflOpenNewFile( pRecEditor, NULL, FALSE,
									&tmpPool, &pTmpNd)))
			{
				pRecEditor->displayMessage( "Unable to open file", rc,
					NULL, WPS_RED, WPS_WHITE);
			}
			break;


		/*
		Goto a specific offset
		*/

		case 'G':
		case 'g':
		{
			szResponse [0] = '\0';
			pRecEditor->requestInput(
				"Offset",
				szResponse, sizeof( szResponse), &uiTermChar);

			if( uiTermChar == WPK_ESCAPE)
			{
				break;
			}
			
			if( (uiSrcLen = (FLMUINT)f_strlen( szResponse)) == 0)
			{
				uiOffset = 0;
			}
			else
			{
				if( RC_BAD( rc = pRecEditor->getNumber( szResponse, &uiOffset, NULL)))
				{
					pRecEditor->displayMessage( "Invalid offset", rc,
						NULL, WPS_RED, WPS_WHITE);
					break;
				}

				RflPositionToNode( uiOffset, FALSE, &tmpPool, &pTmpNd);
				if( pTmpNd)
				{
					pRecEditor->setTree( pTmpNd, &pNewNd);
					pRecEditor->setControlFlags( pNewNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
			}
			break;
		}

		/*
		Find something in the RFL log.
		*/

		case WPK_F1:
		case WPK_F3:
			gv_bSrchForward = TRUE;
			bSkipCurrent = TRUE;
			goto Do_Search;
		case WPK_F2:
			gv_bSrchForward = FALSE;
			bSkipCurrent = TRUE;
			goto Do_Search;
		case 'F':
		case 'f':
		case 's':
		case 'S':
		{
			if (RC_BAD( rc = getSearchCriteria( pRecEditor,
										&gv_SrchCriteria, &gv_bSrchForward)))
			{
				break;
			}
			bSkipCurrent = FALSE;
Do_Search:
			if (RC_BAD( rc = pRecEditor->createStatusWindow(
				" Searching ... (press ESC to interrupt) ",
				WPS_GREEN, WPS_WHITE, NULL, NULL, &pWindow)))
			{
				goto Exit;
			}

			FTXWinOpen( pWindow);
			pLastNd = NULL;
			pCurNd = pFirstNd = pRecEditor->getCurrentNode();

			// See if we have a match in our current tree.

			for (;;)
			{
				if (!pCurNd)
				{
					break;
				}
				pPacket = (RFL_PACKET *)GedValPtr( pCurNd);
				if (rflPassesCriteria( pPacket, &gv_SrchCriteria))
				{
					if (!bSkipCurrent || pCurNd != pFirstNd)
					{
						pRecEditor->setCurrentNode( pCurNd);
						gv_bDoRefresh = FALSE;
						break;
					}
				}
				if (pWindow)
				{
					FTXWinSetCursorPos( pWindow, 0, 1);
					FTXWinPrintf( pWindow,
						"File Offset : %08X", (unsigned)pPacket->uiFileOffset);
					FTXWinClearToEOL( pWindow);
					FTXWinSetCursorPos( pWindow, 0, 2);
					FTXWinPrintf( pWindow,
						"Trans ID    : %u", (unsigned)pPacket->uiTransID);
					FTXWinClearToEOL( pWindow);

					// Test for the escape key

					if (FTXWinTestKB( pWindow) == FTXRC_SUCCESS)
					{
						FLMUINT	uiChar;
						FTXWinInputChar( pWindow, &uiChar);
						if( uiChar == WPK_ESCAPE)
						{
							goto Exit;
						}
					}
				}

				pLastNd = pCurNd;
				if (gv_bSrchForward)
				{
					pCurNd = pRecEditor->getNextNode( pCurNd, FALSE);
				}
				else
				{
					pCurNd = pRecEditor->getPrevNode( pCurNd, FALSE);
				}
			}

			// If no match in the current tree, continue searching
			// until we find one.

			if (pCurNd)
			{
				break;
			}
			pCurNd = pLastNd;

			// If we do not have an EOF, determine one.  We don't
			// want to continue our search past this point.

			if (!gv_uiRflEof)
			{
				if (RC_BAD( rc = viewRflGetEOF()))
				{
					goto Exit;
				}
			}

			for (;;)
			{
				GedPoolReset( &tmpPool, NULL);
				if (gv_bSrchForward)
				{
					if (RC_BAD( rc = RflGetNextNode( pLastNd, FALSE,
												&tmpPool, &pCurNd, TRUE)))
					{
						goto Exit;
					}
				}
				else
				{
					if (RC_BAD( rc = RflGetPrevNode( pLastNd, FALSE,
												&tmpPool, &pCurNd)))
					{
						goto Exit;
					}
				}
				if (!pCurNd)
				{

					// See if we can go to the next or previous file.

					if (gv_SrchCriteria.uiMultiFileSearch == 1)
					{
						if (RC_BAD( rc = rflOpenNewFile( pRecEditor, NULL,
													gv_bSrchForward,
													&tmpPool, &pCurNd)))
						{
							if (rc == FERR_IO_PATH_NOT_FOUND)
							{
								rc = FERR_OK;
								break;
							}
							goto Exit;
						}
						if (pWindow)
						{
							FTXWinSetCursorPos( pWindow, 0, 3);
							FTXWinPrintf( pWindow,
								"File Name   : %s", gv_szRflPath);
							FTXWinClearToEOL( pWindow);
						}
					}
					else
					{
						break;
					}
				}
				pPacket = (RFL_PACKET *)GedValPtr( pCurNd);
				if (rflPassesCriteria( pPacket, &gv_SrchCriteria))
				{
					pRecEditor->setTree( NULL);
					pRecEditor->setTree( pCurNd, &pNewNd);
					pRecEditor->setControlFlags( pNewNd,
							(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
							F_RECEDIT_FLAG_READ_ONLY));
					break;
				}
				if (pWindow)
				{
					FTXWinSetCursorPos( pWindow, 0, 1);
					FTXWinPrintf( pWindow,
						"File Offset : %08X", (unsigned)pPacket->uiFileOffset);
					FTXWinClearToEOL( pWindow);
					FTXWinSetCursorPos( pWindow, 0, 2);
					FTXWinPrintf( pWindow,
						"Trans ID    : %u", (unsigned)pPacket->uiTransID);
					FTXWinClearToEOL( pWindow);

					// Test for the escape key

					if (FTXWinTestKB( pWindow) == FTXRC_SUCCESS)
					{
						FLMUINT	uiChar;
						FTXWinInputChar( pWindow, &uiChar);
						if( uiChar == WPK_ESCAPE)
						{
							goto Exit;
						}
					}
				}
				GedPoolReset( &tmp2Pool, NULL);
				if ((pLastNd = GedCopy( &tmp2Pool, 1, pCurNd)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}
			if (pWindow)
			{
				FTXWinFree( &pWindow);
			}
			break;
		}

		case WPK_ALT_Q:
		case WPK_ESCAPE:
		{
			*puiKeyOut = WPK_ESCAPE;
			break;
		}
	}

Exit:
	if (pWindow)
	{
		FTXWinFree( &pWindow);
	}
	GedPoolFree( &tmpPool);
	GedPoolFree( &tmp2Pool);
	return( rc);
}


/****************************************************************************
Name:	viewRflHelpHook
Desc:	
*****************************************************************************/
RCODE viewRflMainHelpHook(
	F_RecEditor *		pRecEditor,
	F_RecEditor *		pHelpEditor,
	POOL *				pPool,
	void *				UserData,
	NODE **				ppRootNd)
{
	NODE *	pNewTree = NULL;
	RCODE		rc = FERR_OK;

	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pHelpEditor);
	F_UNREFERENCED_PARM( UserData);

	if( (pNewTree = GedNodeMake( pPool, 1, &rc)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = GedPutNATIVE( pPool, pNewTree,
		"RFL Viewer Keyboard Commands")))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		(FLMUINT)'?', (void *)"?               Help (this screen)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_UP, (void *)"UP              Move cursor up",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_DOWN, (void *)"DOWN            Move cursor down",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_PGUP, (void *)"PG UP           Page up",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_PGDN, (void *)"PG DOWN         Page down",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_HOME, (void *)"HOME            Position to beginning of file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_END, (void *)"END             Position to end of file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'S', (void *)"S or F          Search for (find) a packet",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'O', (void *)"O               Open a new log file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'N', (void *)"N               Go to next log file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'P', (void *)"P               Go to previous log file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_F1, (void *)"F1 or F3        Search forward (using last criteria entered)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_F2, (void *)"F2              Search backward (using last criteria entered)",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'G', (void *)"G               Goto an offset in the file",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		'H', (void *)"H               Show RFL Header",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = gedAddField( pPool, pNewTree,
		WPK_ESCAPE, (void *)"ESC, ALT-Q      Exit",
		0, FLM_TEXT_TYPE)))
	{
		goto Exit;
	}

	*ppRootNd = pNewTree;

Exit:

	return( rc);
}


/****************************************************************************
Name:	viewRflMainEventHook
Desc:	
*****************************************************************************/
RCODE viewRflMainEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData)
{
	POOL					tmpPool;
	NODE *				pTmpNd;
	RCODE					rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	GedPoolInit( &tmpPool, 4096);

	switch( eEventType)
	{
		case F_RECEDIT_EVENT_IEDIT:
		{
			NODE *		pNewNd;

			if( RC_BAD( rc = RflGetNextNode( NULL, FALSE, &tmpPool, 
										&pTmpNd)))
			{
				goto Exit;
			}

			pRecEditor->setTree( pTmpNd, &pNewNd);
			pRecEditor->setControlFlags( pNewNd,
				(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
				F_RECEDIT_FLAG_READ_ONLY));
			break;
		}

		case F_RECEDIT_EVENT_REFRESH:
		{
			NODE *		pCurrentNd;
			NODE *		pNewTree;
			NODE *		pTopNd;
			NODE *		pBottomNd;
			FLMUINT		uiPriorCount;
			FLMUINT		uiNextCount;
			FLMUINT		uiCursorRow;

			if (!gv_bDoRefresh)
			{
				gv_bDoRefresh = TRUE;
				break;
			}

			/*
			Re-size the tree
			*/

			pCurrentNd = pRecEditor->getCurrentNode();
			pBottomNd = pTopNd = pCurrentNd;

			uiPriorCount = 0;
			pTmpNd = pTopNd;
			while( pTmpNd && uiPriorCount < pRecEditor->getNumRows())
			{
				pTmpNd = pRecEditor->getPrevNode( pTmpNd, FALSE);
				if( pTmpNd)
				{
					pTopNd = pTmpNd;
					uiPriorCount++;
				}
			}

			uiNextCount = 0;
			pTmpNd = pBottomNd;
			while( pTmpNd && uiNextCount < pRecEditor->getNumRows())
			{
				pBottomNd = pTmpNd;
				pTmpNd = pRecEditor->getNextNode( pTmpNd, FALSE);
				if( pTmpNd)
				{
					uiNextCount++;
				}
			}

			/*
			Clip the rest of the forest
			*/

			pTmpNd = GedSibNext( pBottomNd);
			if( pTmpNd)
			{
				pTmpNd->prior->next = NULL;
			}

			/*
			Reset the tree to the new "pruned" version
			*/

			if (pTopNd)
			{
				if( RC_BAD( rc = pRecEditor->copyBuffer( &tmpPool,
					pTopNd, &pNewTree)))
				{
					goto Exit;
				}
			}
			else
			{
				pNewTree = NULL;
			}

			/*
			Re-position the cursor
			*/

			uiCursorRow = pRecEditor->getCursorRow();
			pRecEditor->setTree( pNewTree, &pTmpNd);
			pNewTree = pTmpNd;

			if( uiPriorCount > uiCursorRow)
			{
				uiPriorCount -= uiCursorRow;
				while( uiPriorCount)
				{
					pTmpNd = pRecEditor->getNextNode( pTmpNd);
					if( pTmpNd)
					{
						pNewTree = pTmpNd;
					}
					uiPriorCount--;
				}
				pRecEditor->setCurrentNode( pNewTree);
				pRecEditor->setCurrentAtTop();
			}
			
			pTmpNd = pNewTree;
			while( uiCursorRow)
			{
				pTmpNd = pRecEditor->getNextNode( pTmpNd);
				if( pTmpNd)
				{
					pNewTree = pTmpNd;
				}
				else
				{
					break;
				}
				uiCursorRow--;
			}

			pRecEditor->setCurrentNode( pNewTree);
			break;
		}

		case F_RECEDIT_EVENT_GETDISPVAL:
		{
			DBE_VAL_INFO *		pValInfo = (DBE_VAL_INFO *)EventData;
			NODE *				pNd = pValInfo->pNd;

			RflFormatPacket( GedValPtr( pNd), (char *)pValInfo->pucBuf);
			break;
		}

		case F_RECEDIT_EVENT_GETNEXTNODE:
		{
			DBE_NODE_INFO *		pNodeInfo = (DBE_NODE_INFO *)EventData;

			pNodeInfo->pNd = pRecEditor->getNextNode( pNodeInfo->pCurNd, FALSE);
			if( !pNodeInfo->pNd)
			{
				if( RC_BAD( rc = RflGetNextNode( pNodeInfo->pCurNd,
					FALSE, &tmpPool, &pTmpNd)))
				{
					goto Exit;
				}

				if( pTmpNd)
				{
					pRecEditor->appendTree( pTmpNd, &pNodeInfo->pNd);
					pRecEditor->setControlFlags( pNodeInfo->pNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
			}
			pNodeInfo->bUseNd = TRUE;
			break;
		}

		case F_RECEDIT_EVENT_GETPREVNODE:
		{
			DBE_NODE_INFO *		pNodeInfo = (DBE_NODE_INFO *)EventData;

			pNodeInfo->pNd = pRecEditor->getPrevNode( pNodeInfo->pCurNd, FALSE);
			if( !pNodeInfo->pNd)
			{
				if( RC_BAD( rc = RflGetPrevNode( pNodeInfo->pCurNd, FALSE,
					&tmpPool, &pTmpNd)))
				{
					goto Exit;
				}

				if( pTmpNd)
				{
					pRecEditor->insertRecord( pTmpNd, &pNodeInfo->pNd);
					pRecEditor->setControlFlags( pNodeInfo->pNd,
						(F_RECEDIT_FLAG_HIDE_LEVEL | F_RECEDIT_FLAG_HIDE_TAG |
						F_RECEDIT_FLAG_READ_ONLY));
				}
			}
			pNodeInfo->bUseNd = TRUE;
			break;
		}

		default:
		{
			break;
		}
	}

Exit:

	GedPoolFree( &tmpPool);
	return( rc);
}


/****************************************************************************
Name:	viewRflInspectEntry
Desc:	
*****************************************************************************/
RCODE viewRflInspectEntry(
	F_RecEditor *		pParentEditor)
{
	F_RecEditor *		pRecEditor;
	NODE *				pExpandNd;
	POOL					tmpPool;
	RCODE					rc = FERR_OK;

	GedPoolInit( &tmpPool, 1024);

	if( (pRecEditor = new F_RecEditor) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( pRecEditor->Setup( pParentEditor->getScreen())))
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pRecEditor->setTree( NULL);
	pRecEditor->setShutdown( &gv_bShutdown);
	pRecEditor->setDisplayHook( viewRflInspectDispHook, 0);
	pRecEditor->setEventHook( viewRflInspectEventHook, (void *)0);
	pRecEditor->setKeyHook( viewRflInspectKeyHook, 0);
	pRecEditor->setTitle( "Log Entry");

	if( RC_BAD( rc = RflExpandPacket( pParentEditor->getCurrentNode(), &tmpPool,
								&pExpandNd)))
	{
		goto Exit;
	}

	pRecEditor->setTree( pExpandNd);
	pRecEditor->interactiveEdit( 0, 1);

Exit:

	if( pRecEditor)
	{
		pRecEditor->Release();
	}

	GedPoolFree( &tmpPool);
	return( rc);
}


/****************************************************************************
Name:	viewRflInspectDispHook
Desc:
*****************************************************************************/
RCODE viewRflInspectDispHook(
	F_RecEditor *			pRecEditor,
	NODE *					pNd,
	void *					UserData,
	DBE_DISP_COLUMN *		pDispVals,
	FLMUINT *				puiNumVals)
{
	FLMUINT		uiFlags;
	FLMUINT		uiCol = 0;
	FLMUINT		uiOffset;
	FLMUINT		uiTag = 0;
	FLMUINT		uiTmp;
	FLMBOOL		bBadField = FALSE;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);

	if( !pNd)
	{
		goto Exit;
	}

	uiTag = GedTagNum( pNd);
	pRecEditor->getControlFlags( pNd, &uiFlags);
	if( !pRecEditor->isSystemNode( pNd))
	{
		/*
		Output the record source
		*/

		uiOffset = 0;
		GedGetRecSource( pNd, NULL, NULL, &uiOffset);

		if( uiOffset)
		{
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
				"%8.8X", (unsigned)uiOffset);
			pDispVals[ *puiNumVals].uiCol = uiCol;
			pDispVals[ *puiNumVals].uiForeground = WPS_WHITE;
			pDispVals[ *puiNumVals].uiBackground = WPS_BLUE;
			(*puiNumVals)++;
		}
		uiCol += 10;

		/*
		Output the level
		*/

		f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
			"%u", (unsigned)GedNodeLevel( pNd));
		pDispVals[ *puiNumVals].uiCol = uiCol + (GedNodeLevel( pNd) * 2);
		pDispVals[ *puiNumVals].uiForeground = WPS_WHITE;
		pDispVals[ *puiNumVals].uiBackground = WPS_BLUE;
		uiCol += (FLMUINT)(f_strlen( pDispVals[ *puiNumVals].pucString) +
			(GedNodeLevel( pNd) * 2) + 1);
		(*puiNumVals)++;

		/*
		Output the tag
		*/

		if( RC_BAD( pRecEditor->getDictionaryName(
			uiTag, pDispVals[ *puiNumVals].pucString)))
		{
			f_sprintf( (char *)pDispVals[ *puiNumVals].pucString,
				"TAG_%u", (unsigned)uiTag);
		}

		/*
		Determine if the field is bad
		*/

		switch( uiTag)
		{
			case RFL_TAG_NUM_FIELD:
			case RFL_TYPE_FIELD:
			case RFL_LEVEL_FIELD:
			case RFL_DATA_LEN_FIELD:
			case RFL_DATA_FIELD:
			{
				NODE *		pParentNd = GedParent( pNd);
				FLMUINT		uiParentTag;

				if( pParentNd)
				{
					uiParentTag = GedTagNum( pParentNd);
					if( uiParentTag == RFL_INSERT_FLD_FIELD ||
						uiParentTag == RFL_MODIFY_FLD_FIELD ||
						uiParentTag == RFL_DELETE_FLD_FIELD)
					{
						break;
					}
				}
					
				bBadField = TRUE;
				break;
			}

			case RFL_PACKET_CHECKSUM_VALID_FIELD:
			{
				if( RC_OK( GedGetUINT( pNd, &uiTmp)))
				{
					if( !uiTmp)
					{
						bBadField = TRUE;
					}
				}
				break;
			}
		}

		if( bBadField)
		{
			pDispVals[ *puiNumVals].uiForeground = WPS_RED;
			pDispVals[ *puiNumVals].uiBackground = WPS_WHITE;
		}
		else
		{
#ifdef FLM_WIN
			pDispVals[ *puiNumVals].uiForeground = WPS_LIGHTGREEN;
#else
			pDispVals[ *puiNumVals].uiForeground = WPS_GREEN;
#endif
			pDispVals[ *puiNumVals].uiBackground = WPS_BLUE;
		}

		pDispVals[ *puiNumVals].uiCol = uiCol;
		uiCol += (FLMUINT)(f_strlen( pDispVals[ *puiNumVals].pucString) + 1);
		(*puiNumVals)++;

		/*
		Output the display value
		*/
		
		switch( uiTag)
		{
			case RFL_INSERT_FLD_FIELD:
			case RFL_MODIFY_FLD_FIELD:
			case RFL_DELETE_FLD_FIELD:
			{
				/*
				Don't output the value
				*/

				break;
			}
			default:
			{
				if( RC_BAD( rc = pRecEditor->getDisplayValue( pNd,
					F_RECEDIT_DEFAULT_TYPE, pDispVals[ *puiNumVals].pucString,
					sizeof( pDispVals[ *puiNumVals].pucString))))
				{
					goto Exit;
				}

				pDispVals[ *puiNumVals].uiCol = uiCol;
				pDispVals[ *puiNumVals].uiForeground = WPS_YELLOW;
				pDispVals[ *puiNumVals].uiBackground = WPS_BLUE;
				uiCol += (FLMUINT)(f_strlen( pDispVals[ *puiNumVals].pucString) + 1);
				(*puiNumVals)++;
			}
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Name:	viewRflInspectEventHook
Desc:	
*****************************************************************************/
RCODE viewRflInspectEventHook(
	F_RecEditor *		pRecEditor,
	eEventType			eEventType,
	void *				EventData,
	void *				UserData)
{
	RCODE					rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);
	F_UNREFERENCED_PARM( pRecEditor);

	switch( eEventType)
	{
		case F_RECEDIT_EVENT_NAME_TABLE:
		{
			DBE_NAME_TABLE_INFO *		pNameTableInfo = (DBE_NAME_TABLE_INFO *)EventData;

			pNameTableInfo->pNameTable = gv_pNameTable;
			pNameTableInfo->bInitialized = TRUE;
			break;
		}

		default:
		{
			break;
		}
	}

	return( rc);
}


/****************************************************************************
Name:	viewRflInspectKeyHook
Desc:	
*****************************************************************************/
RCODE viewRflInspectKeyHook(
	F_RecEditor *		pRecEditor,
	NODE *				pCurNd,
	FLMUINT				uiKeyIn,
	void *				UserData,
	FLMUINT *			puiKeyOut)
{
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( UserData);
	F_UNREFERENCED_PARM( pRecEditor);
	F_UNREFERENCED_PARM( pCurNd);

	if( puiKeyOut)
	{
		*puiKeyOut = 0;
	}

	switch( uiKeyIn)
	{
		case WPK_DOWN:
		case WPK_UP:
		case WPK_PGDN:
		case WPK_PGUP:
		case WPK_ESCAPE:
		case WPK_ENTER:
		case WPK_END:
		case WPK_HOME:
		case '?':
		{
			*puiKeyOut = uiKeyIn;
			break;
		}
	}

	return( rc);
}


/****************************************************************************
Name:	viewRflNameTableInit
Desc:	
*****************************************************************************/
RCODE viewRflNameTableInit(
	F_NameTable **		ppNameTable)
{
	FLMBOOL				bOpenDb = FALSE;
	char *				pucTmp;
	char					szIoDbPath [F_PATH_MAX_SIZE];
	char					szFileName[ F_PATH_MAX_SIZE];
	HFDB					hDb = HFDB_NULL;
	F_NameTable *		pNameTable = NULL;
	RFL_TAG_NAME *		pTag;
	RCODE					rc = FERR_OK;

	/*
	Try to open the database
	*/

	if( RC_BAD( f_pathReduce( gv_szRflPath, szIoDbPath, szFileName)))
	{
		goto Exit;
	}

	pucTmp = f_strchr( (const char *)szFileName, '.');
	if( f_stricmp( pucTmp, ".log") == 0)
	{
		*pucTmp = 0;
		if( f_strlen( szFileName) > 5)
		{
			pucTmp = &szFileName[ f_strlen( szFileName) - 5];
			pucTmp[ 0] = '.';
			pucTmp[ 1] = 'd';
			pucTmp[ 2] = 'b';
			pucTmp[ 3] = '\0';

			if (RC_BAD( rc = f_pathAppend( szIoDbPath, szFileName)))
			{
				goto Exit;
			}
			bOpenDb = TRUE;
		}
	}

	if( bOpenDb)
	{
		if( RC_OK( FlmConfig( FLM_MAX_UNUSED_TIME, (void *)0, (void *)0)))
		{
			FlmDbOpen( szIoDbPath, NULL, NULL, // VISIT
				FO_DONT_REDO_LOG, NULL, &hDb);
		}
	}

	if( (pNameTable = new F_NameTable) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pNameTable->setupFromDb( hDb)))
	{
		goto Exit;
	}

	// Build the name table

	pTag = &RflDictTags[ 0];
	while( pTag->pszTagName)
	{
		if( RC_BAD( rc = pNameTable->addTag( NULL, pTag->pszTagName,
			pTag->uiTagNum, FLM_FIELD_TAG, 0)))
		{
			flmAssert( 0);
			goto Exit;
		}
		pTag++;
	}

	*ppNameTable = pNameTable;
	pNameTable = NULL;

Exit:

	if( pNameTable)
	{
		pNameTable->Release();
	}

	if( hDb != HFDB_NULL)
	{
		FlmDbClose( &hDb);
	}

	return( rc);
}
