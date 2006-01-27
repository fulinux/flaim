//-------------------------------------------------------------------------
// Desc:	Class for displaying an F_FileHdl class structure in HTML on a web page.
// Tabs:	3
//
//		Copyright (c) 2001-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: imonfhdl.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc: Implements the display function of the F_FileHdlPage
*****************************************************************************/
RCODE F_FileHdlPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE				rc = FERR_OK;
	char				szFrom[20];
	char				szFileId[20];
	FLMINT			uiFileId;
	char				szList[10];
	FLMUINT			uiList;
	F_FileHdl *		pFileHdl = NULL;
	FLMBOOL			bRefresh;
	FLMUINT			uiSize;
	FLMUINT			uiOffset;
	FLMUINT			uiListCount;
	FLMBYTE			szTemp[150];
	FLMBOOL			bHadError = FALSE;
	FLMBOOL			bHighlight = FALSE;
	FLMBYTE *		pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( 250, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}
	
	//Let's first find out where we  came from, then get the appropriate parameters.


	szFrom[0] = '\0';
	szList[0] = '\0';


	//Get the "From" parameter
	if (RC_BAD( rc = ExtractParameter( uiNumParams,
												  ppszParams,
												  "From",
												  sizeof( szFrom),
												  szFrom)))
	{
		goto Exit;
	}
	

	//If the source of this req	uest is from the FileHdlMgr, then we will begin our search from there.
	if (f_stricmp( szFrom, "FileHdlMgr") == 0)
	{

		//Get the file id (index) and the list type.  The parameters must be set!

		if (RC_BAD( rc = ExtractParameter( uiNumParams,
													  ppszParams,
													  "FileId",
													  sizeof( szFileId),
													  szFileId)))
		{
			goto Exit;
		}

		uiFileId = f_atoud( szFileId);
		if (RC_BAD( rc = ExtractParameter( uiNumParams,
													  ppszParams,
													  "List",
													  sizeof( szList),
													  szList)))
		{
			goto Exit;
		}


		if (f_stricmp( szList, "Used") == 0)
		{
			uiList = FHM_USED_LIST;
		}
		else if (f_stricmp( szList, "Avail") == 0)
		{
			uiList = FHM_AVAIL_LIST;
		}
		else  //Invalid List option
		{
			rc = RC_SET( FERR_FAILURE);
			goto Exit;
		}


		// Now get the file handle from the appropriate list, either the Used list or the Available list
		f_mutexLock( gv_FlmSysData.hFileHdlMutex);

		uiListCount = gv_FlmSysData.pFileHdlMgr->GetListMgr()->GetCount(uiList);
		if (uiListCount > 0)
		{
			pFileHdl = (F_FileHdl *)gv_FlmSysData.pFileHdlMgr->GetListMgr()->GetItem(uiList, uiFileId);
			pFileHdl->AddRef();
		}
		else
		{
			pFileHdl = NULL;
		}
		f_mutexUnlock( gv_FlmSysData.hFileHdlMutex);


	}  //FileHdlMgr
	else
	{
		// Generate an error for now. (nothing else implemented)
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	//Before we actually create the rest of the page, let's verify that the file handle
	//we are about to display is still valid.

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");


	if (pFileHdl)
	{


		//Check to see if we are to refresh this page automatically.
		if ((bRefresh = DetectParameter( uiNumParams,
													ppszParams,
													"Refresh")) == TRUE)
		{
			//Send back the page with a refresh command in the header
			
			f_sprintf((char *)szTemp, 
						"%s/FileHdl?Refresh&From=%s&List=%s&FileId=%s",
						m_pszURLString,
						szFrom, szList, szFileId);

			fnPrintf( m_pHRequest, 
				"<HEAD>"
				"<META http-equiv=\"refresh\" content=\"5; url=%s\">"
				"<TITLE>File Handle Structure</TITLE>\n",
				szTemp);

		}
		else
		{
			fnPrintf( m_pHRequest, 
				"<HEAD><TITLE>File Handle Structure</TITLE>\n");
		}
		printStyle();
		fnPrintf( m_pHRequest, "</HEAD>\n");

		
		fnPrintf( m_pHRequest, "<body>\n");
		
		// If we are not to refresh this page, then don't include the refresh meta command
		if (!bRefresh)
		{
			f_sprintf( (char *)pszTemp,
				"<A HREF=%s/FileHdl?Refresh&From=%s&List=%s&FileId=%s>Start Auto-refresh (5 sec.)</A>",
				m_pszURLString, szFrom, szList, szFileId);
		}
		else
		{
			f_sprintf( (char *)pszTemp,
				"<A HREF=%s/FileHdl?From=%s&List=%s&FileId=%s>Stop Auto-refresh</A>",
				m_pszURLString, szFrom, szList, szFileId);
		}
		// Prepare the refresh link.
		f_sprintf( (char *)szTemp,
			"<A HREF=%s/FileHdl?From=%s&List=%s&FileId=%s>Refresh</A>",
			m_pszURLString, szFrom, szList, szFileId);

		
		
		//Insert a new table into the page to display the FileHdl fields
		printTableStart( "File Handle", 1, 100);

		printTableRowStart();
		printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 1, 1, FALSE);
		fnPrintf( m_pHRequest, "%s, ", szTemp);
		fnPrintf( m_pHRequest, "%s\n", pszTemp);
		printColumnHeadingClose();
		printTableRowEnd();

		printTableEnd();
		
		
		printTableStart( "File Handle - Methods", 2);

		// Write out the table headings.
		printTableRowStart();
		printColumnHeading( "Method Name");
		printColumnHeading( "Value");
		printTableRowEnd();


		//File Size
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "Size");
		if (RC_BAD( rc = pFileHdl->Size(&uiSize)))
		{
			bHadError = TRUE;
			goto Exit;
		}
		fnPrintf( m_pHRequest, TD_ui, uiSize);
		printTableRowEnd();



		//Current position - Tell
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "Tell (Current position)");
		if (RC_BAD( rc = pFileHdl->Tell(&uiOffset)))
		{
			bHadError = TRUE;
			goto Exit;
		}
		fnPrintf( m_pHRequest, TD_ui, uiOffset);
		printTableRowEnd();

		// Write out a table showing the private data
		write_data( (F_FileHdlImp *)pFileHdl);

	}
	else
	{

		fnPrintf( m_pHRequest, "<body>\n");

		fnPrintf( m_pHRequest, "Error - the File Header structure you are seeking is no longer valid.  "
			                 "This page will automatically redirect you to the File Handle Manager page after five seconds\n");

	}



	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();

Exit:

	if (pFileHdl)
	{
		f_mutexLock( gv_FlmSysData.hFileHdlMutex);
		pFileHdl->Release();
		f_mutexUnlock( gv_FlmSysData.hFileHdlMutex);
		pFileHdl = NULL;
	}


	if (bHadError)
	{
		printTableRowEnd();
		printTableEnd();
		fnPrintf( m_pHRequest, "Error - An error has occured during processing."
									  " File handle could not be retrieved.\n");
		fnPrintf( m_pHRequest, "</body></html>\n");
		fnEmit();
		bHadError = FALSE;
	}

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	return( rc);

}

/***************************************************************************
Desc:	Function to display the private data on a WIN32 platform
***************************************************************************/
#ifdef FLM_WIN
void F_FileHdlPage::write_data(
	F_FileHdlImp *			pFileHdl)
{
	F_Base *					pBase;
	F_ListItem *			pListItem;
	F_FileHdlImpBase *	pFileHdlBase;
	char						szAddress[20];
	FLMBOOL					bHighlight = FALSE;

	if (!pFileHdl)
	{
		return;
	}

	// Start the table
	printTableStart( "File Handle - Fields", 4);

	
	// Write out the table headings.
	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();

	// m_FileHandle (HANDLE)
	printAddress( &pFileHdl->m_FileHandle, szAddress);
	printHTMLString(
		"m_FileHandle",
		"HANDLE",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_FileHandle,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// m_uiBlockSize
	printHTMLUint(
		"m_uiBlockSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiBlockSize,
		pFileHdl->m_uiBlockSize,
		(bHighlight = ~bHighlight));

	// m_uiBytesPerSector
	printHTMLUint(
		"m_uiBytesPerSector",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiBytesPerSector,
		pFileHdl->m_uiBytesPerSector,
		(bHighlight = ~bHighlight));

	// m_uiNotOnSectorBoundMask
	printHTMLUint(
		"m_uiNotOnSectorBoundMask",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiNotOnSectorBoundMask,
		pFileHdl->m_uiNotOnSectorBoundMask,
		(bHighlight = ~bHighlight));

	// m_uiGetSectorBoundMask
	printHTMLUint(
		"m_uiGetSectorBoundMask",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiGetSectorBoundMask,
		pFileHdl->m_uiGetSectorBoundMask,
		(bHighlight = ~bHighlight));

	// m_bDoDirectIO
	printHTMLString(
		"m_bDoDirectIO",
		"FLMBOOL",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bDoDirectIO,
		(char *)(pFileHdl->m_bDoDirectIO ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_uiExtendSize
	printHTMLUint(
		"m_uiExtendSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiExtendSize,
		pFileHdl->m_uiExtendSize,
		(bHighlight = ~bHighlight));

	// m_uiMaxAutoExtendSize
	printHTMLUint(
		"m_uiMaxAutoExtendSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiMaxAutoExtendSize,
		pFileHdl->m_uiMaxAutoExtendSize,
		(bHighlight = ~bHighlight));

	// m_pucAlignedBuff
	printAddress( pFileHdl->m_pucAlignedBuff, szAddress);
	printHTMLString(
		"m_pucAlignedBuff",
		"FLMBOOL",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_pucAlignedBuff,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// m_uiAlignedBuffSize
	printHTMLUint(
		"m_uiAlignedBuffSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiAlignedBuffSize,
		pFileHdl->m_uiAlignedBuffSize,
		(bHighlight = ~bHighlight));

	// m_uiCurrentPos
	printHTMLUint(
		"m_uiCurrentPos",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiCurrentPos,
		pFileHdl->m_uiCurrentPos,
		(bHighlight = ~bHighlight));

	// m_bCanDoAsync
	printHTMLString(
		"m_bCanDoAsync",
		"FLMBOOL",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bCanDoAsync,
		(char *)(pFileHdl->m_bCanDoAsync ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_Overlapped (OVERLAPPED)
	printAddress( &pFileHdl->m_Overlapped, szAddress);
	printHTMLString(
		"m_Overlapped",
		"OVERLAPPED",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_Overlapped,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// Now we show private members of the base class F_FileHdlBase

	pFileHdlBase = (F_FileHdlImpBase *)pFileHdl;

	// m_LNode
	printAddress( &pFileHdlBase->m_LNode[0], szAddress);
	printHTMLString(
		"F_FileHdlBase.m_LNode",
		"LNODE",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_LNode,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// m_bFileOpened
	printHTMLString(
		"F_FileHdlBase.m_bFileOpened",
		"FLMBOOL",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_bFileOpened,
		(char *)(pFileHdlBase->m_bFileOpened ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_uiAvailTime
	printHTMLUint(
		"F_FileHdlBase.m_uiAvailTime",
		"FLMUINT",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_uiAvailTime,
		pFileHdlBase->m_uiAvailTime,
		(bHighlight = ~bHighlight));

	// m_uiFileId
	printHTMLUint(
		"F_FileHdlBase.m_uiFileId",
		"FLMUINT",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_uiFileId,
		pFileHdlBase->m_uiFileId,
		(bHighlight = ~bHighlight));

	// m_bDeleteOnClose
	printHTMLString(
		"F_FileHdlBase.m_bDeleteOnClose",
		"FLMBOOL",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_bDeleteOnClose,
		(char *)(pFileHdlBase->m_bDeleteOnClose ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_bOpenedReadOnly
	printHTMLString(
		"F_FileHdlBase.m_bOpenedReadOnly",
		"FLMBOOL",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_bOpenedReadOnly,
		(char *)(pFileHdlBase->m_bOpenedReadOnly ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_bOpenedExclusive
	printHTMLString(
		"F_FileHdlBase.m_bOpenedExclusive",
		"FLMBOOL",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_bOpenedExclusive,
		(char *)(pFileHdlBase->m_bOpenedExclusive ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_pszIoPath
	printHTMLString(
		"F_FileHdlBase.m_pszIoPath",
		"FLMBYTE *",
		(void *)pFileHdlBase,
		(void *)&pFileHdlBase->m_pszIoPath,
		(char *)pFileHdlBase->m_pszIoPath,
		(bHighlight = ~bHighlight));

	// Now show the private members of the F_ListItem class
	pListItem = (F_ListItem *)pFileHdlBase;

	// m_pListMgr
	printAddress( pListItem->m_pListMgr, szAddress);
	printHTMLString(
		"F_ListItem.m_pListMgr",
		"F_ListMgr *",
		(void *)pListItem,
		(void *)&pListItem->m_pListMgr,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// m_uiLNodeCnt
	printHTMLUint(
		"F_ListItem.m_uiLNodeCnt",
		"FLMUINT",
		(void *)pListItem,
		(void *)&pListItem->m_uiLNodeCnt,
		pListItem->m_uiLNodeCnt,
		(bHighlight = ~bHighlight));

	// m_pLNodes
	printAddress( pListItem->m_pLNodes, szAddress);
	printHTMLString(
		"F_ListItem.m_pLNodes",
		"LNODE *",
		(void *)pListItem,
		(void *)&pListItem->m_pLNodes,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// m_bInList
	printHTMLString(
		"F_ListItem.m_bInList",
		"FLMBOOL",
		(void *)pListItem,
		(void *)&pListItem->m_bInList,
		(char *)(pListItem->m_bInList ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// Now for the final base class - F_Base
	pBase = (F_Base *)pListItem;

	// m_ui32RefCnt
	printHTMLUint(
		"F_Base.m_ui32RefCnt",
		"FLMUINT",
		(void *)pBase,
		(void *)&pBase->m_ui32RefCnt,
		pBase->m_ui32RefCnt,
		(bHighlight = ~bHighlight));


	printTableEnd();

}

#endif

#ifdef FLM_UNIX
/***************************************************************************
Desc:	Function to display the private data on a UNIX platform
***************************************************************************/
void F_FileHdlPage::write_data(
	F_FileHdlImp *				pFileHdl)
{
	FLMBOOL			bHighlight = FALSE;

	if (!pFileHdl)
	{
		return;
	}

	// Start the table
	printTableStart( "File Handle Structure - Fields", 4);

	
	// Write out the table headings.
	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();


	// m_fd (int)
	printHTMLInt(
		"m_fd",
		"int",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_fd,
		pFileHdl->m_fd,
		(bHighlight = ~bHighlight));

	// m_uiCurrentPos
	printHTMLUint(
		"m_uiCurrentPos",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiCurrentPos,
		pFileHdl->m_uiCurrentPos,
		(bHighlight = ~bHighlight));

	// m_bDoDirectIO
	printHTMLString(
		"m_bDoDirectIO",
		"FLMBOOL",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bDoDirectIO,
		(char *)(pFileHdl->m_bDoDirectIO ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_uiMaxAutoExtendSize
	printHTMLUint(
		"m_uiMaxAutoExtendSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiMaxAutoExtendSize,
		pFileHdl->m_uiMaxAutoExtendSize,
		(bHighlight = ~bHighlight));

	// m_bCanDoAsync
	printHTMLString(
		"m_bCanDoAsync",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bCanDoAsync,
		(char *)(pFileHdl->m_bCanDoAsync ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	printTableEnd();
}
#endif

/***************************************************************************
Desc:	Function to display the private data on a Netware platform
***************************************************************************/
#ifdef FLM_NLM
void F_FileHdlPage::write_data(
	F_FileHdlImp *				pFileHdl)
{
	char				szAddress[20];
	FLMBOOL			bHighlight = FALSE;

	if (!pFileHdl)
	{
		return;
	}

	// Start the table
	printTableStart( "File Handle Structure - Fields", 4);

	
	// Write out the table headings.
	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();

	// m_lFileHandle (LONG)
	printHTMLUlong(
		"m_lFileHandle",
		"LONG",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_lFileHandle,
		pFileHdl->m_lFileHandle,
		(bHighlight = ~bHighlight));

	// m_lOpenAttr
	printHTMLUlong(
		"m_lOpenAttr",
		"LONG",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_lOpenAttr,
		pFileHdl->m_lOpenAttr,
		(bHighlight = ~bHighlight));

	// m_lVolumeID
	printHTMLUlong(
		"m_lVolumeID",
		"LONG",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_lVolumeID,
		pFileHdl->m_lVolumeID,
		(bHighlight = ~bHighlight));

	// m_lLNamePathCount
	printHTMLUlong(
		"m_lLNamePathCount",
		"LONG",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_lLNamePathCount,
		pFileHdl->m_lLNamePathCount,
		(bHighlight = ~bHighlight));

	// m_bDoSuballocation
	printHTMLString(
		"m_bDoSuballocation",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bDoSuballocation,
		(char *)(pFileHdl->m_bDoSuballocation ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_uiExtendSize
	printHTMLUint(
		"m_uiExtendSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiExtendSize,
		pFileHdl->m_uiExtendSize,
		(bHighlight = ~bHighlight));

	// m_uiMaxAutoExtendSize
	printHTMLUint(
		"m_uiMaxAutoExtendSize",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiMaxAutoExtendSize,
		pFileHdl->m_uiMaxAutoExtendSize,
		(bHighlight = ~bHighlight));

	// m_bDoDirectIO
	printHTMLString(
		"m_bDoDirectIO",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bDoDirectIO,
		(char *)(pFileHdl->m_bDoDirectIO ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_lSectorsPerBlock
	printHTMLUlong(
		"m_lSectorsPerBlock",
		"LONG",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_lSectorsPerBlock,
		pFileHdl->m_lSectorsPerBlock,
		(bHighlight = ~bHighlight));

	// m_lMaxBlocks
	printHTMLUlong(
		"m_lMaxBlocks",
		"LONG",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_lMaxBlocks,
		pFileHdl->m_lMaxBlocks);

	// m_uiCurrentPos
	printHTMLUint(
		"m_uiCurrentPos",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_uiCurrentPos,
		pFileHdl->m_uiCurrentPos,
		(bHighlight = ~bHighlight));

	// m_bNSS
	printHTMLString(
		"m_bNSS",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bNSS,
		(char *)(pFileHdl->m_bNSS ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	// m_NssKey (Key_t)
	printAddress( &pFileHdl->m_NssKey, szAddress);
	printHTMLString(
		"m_NssKey",
		"Key_t",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_NssKey,
		(char *)szAddress,
		(bHighlight = ~bHighlight));

	// m_bNSSFileOpen
	printHTMLString(
		"m_bNSSFileOpen",
		"FLMUINT",
		(void *)pFileHdl,
		(void *)&pFileHdl->m_bNSSFileOpen,
		(char *)(pFileHdl->m_bNSSFileOpen ? "Yes" : "No"),
		(bHighlight = ~bHighlight));

	printTableEnd();

}

#endif  // FLM_NLM
