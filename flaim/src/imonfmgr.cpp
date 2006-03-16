//-------------------------------------------------------------------------
// Desc:	Class for displaying a file handle manager in HTML on a web page.
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
// $Id: imonfmgr.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	This function implementd the display method of the FileHdlMgrPage web
		page.
*****************************************************************************/
RCODE F_FileHdlMgrPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE					rc = FERR_OK;
	F_FileHdlMgr_p		pFileHdlMgr;
	F_ListMgr *			pListMgr;
	FLMUINT				uiNumAvailItems;
	FLMUINT				uiNumUsedItems;
	char					szTemp[20];
	FLMUINT				uiLoop;
	FLMBOOL				bRefresh = FALSE;
	F_Base *				pBase;
	char					szAddress[20];
	FLMBOOL				bHighlight = FALSE;
	FLMBYTE *			pszTemp = NULL;
	FLMBYTE *			pszTemp1 = NULL;

	if( RC_BAD( rc = f_alloc( 250, &pszTemp)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( 250, &pszTemp1)))
	{
		printErrorPage( rc, TRUE, (char *)"Failed to allocate temporary buffer");
		goto Exit;
	}
	
	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	// Determine if we are being requested to refresh this page or  not.

	if ((bRefresh = DetectParameter(
							uiNumParams,
							ppszParams,
							"Refresh")) == TRUE)	
	{
		// Send back the page with a refresh command in the header
		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s/FileHdlMgr?Refresh\">"
			"<TITLE>gv_FlmSysData.pFileHdlMgr</TITLE>\n",
			m_pszURLString);
	}
	else
	{
		fnPrintf( m_pHRequest, 
			"<HEAD><TITLE>gv_FlmSysData.pFileHdlMgr</TITLE>\n");
	}
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");


	fnPrintf( m_pHRequest, "<body>\n");

	// If we are not to refresh this page, then don't include the
	// refresh meta command
	if (!bRefresh)
	{
		f_sprintf( (char *)pszTemp,
					"<A HREF=%s/FileHdlMgr?Refresh>Start Auto-refresh (5 sec.)</A>",
					 m_pszURLString);
	}
	else
	{
		f_sprintf( (char *)pszTemp,
					"<A HREF=%s/FileHdlMgr>Stop Auto-refresh</A>",
					 m_pszURLString);
	}
	// Prepare the refresh link.
	f_sprintf( (char *)pszTemp1, "<A HREF=%s/FileHdlMgr>Refresh</A>",
					 m_pszURLString);



	if (gv_FlmSysData.pFileHdlMgr == NULL)
	{
		fnPrintf( m_pHRequest,
					 "<CENTER>No File Handle Manager exists.  "
					 "Please ensure that a database has been opened."
					 "</CENTER>\n");
	}
	else
	{
		// Lock the file handle manager Mutex and add a reference
		// to it to hold it until we are done.
		f_mutexLock( gv_FlmSysData.hFileHdlMutex);
		pFileHdlMgr = gv_FlmSysData.pFileHdlMgr;
		pFileHdlMgr->AddRef();
		f_mutexUnlock( gv_FlmSysData.hFileHdlMutex);

		printTableStart( "File Handle Manager", 1, 100);

		printTableRowStart();
		printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
		fnPrintf( m_pHRequest, "%s, ", pszTemp1);
		fnPrintf( m_pHRequest, "%s\n", pszTemp);
		printColumnHeadingClose();
		printTableRowEnd();

		printTableEnd();


		printTableStart( "File Handle Manager - Methods", 2, 100);

		// Write out the table headings.
		printTableRowStart();
		printColumnHeading( "Method Name");
		printColumnHeading( "Value");
		printTableRowEnd();

		// GetOpenThreshold - method to return the file open threshold
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "GetOpenThreshold");
		fnPrintf( m_pHRequest, TD_ld, pFileHdlMgr->GetOpenThreshold());
		printTableRowEnd();

		// GetOpenedFiles - Returns the number of opened files
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "GetOpenedFiles");
		fnPrintf( m_pHRequest, TD_ld, pFileHdlMgr->GetOpenedFiles());
		printTableRowEnd();

		// GetMaxAvailTime
		printTableRowStart( bHighlight = ~bHighlight);
		fnPrintf( m_pHRequest, TD_s, "GetMaxAvailTime");
		FormatTime( pFileHdlMgr->GetMaxAvailTime(), szTemp);
		fnPrintf( m_pHRequest, TD_s, szTemp);
		fnPrintf( m_pHRequest, TR_END);
		printTableRowEnd();


		// Introduce another table to show the private members

		printTableStart( "File Handle Manager - Fields", 4, 100);

		// Write out the table headings.
		printTableRowStart();
		printColumnHeading( "Byte Offset (hex)");
		printColumnHeading( "Field Name");
		printColumnHeading( "Field Type");
		printColumnHeading( "Value");
		printTableRowEnd();

		// m_phMutex (show the address)
		printAddress( &pFileHdlMgr->m_phMutex, szAddress);
		printHTMLString(
			"m_phMutex",
			"F_MUTEX",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_phMutex,
			(char *)szAddress,
			(bHighlight = ~bHighlight));

		// m_uiOpenThreshold
		printHTMLUint(
			"m_uiOpenThreshold",
			"FLMUINT",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_uiOpenThreshold,
			pFileHdlMgr->m_uiOpenThreshold,
			(bHighlight = ~bHighlight));

		// m_uiMaxAvailTime
		FormatTime( pFileHdlMgr->m_uiMaxAvailTime, szTemp);
		printHTMLString(
			"m_uiMaxAvailTime",
			"FLMUINT",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_uiMaxAvailTime,
			(char *)szTemp,
			(bHighlight = ~bHighlight));

		// m_ListMgr (show address)
		printAddress( &pFileHdlMgr->m_ListMgr, szAddress);
		printHTMLString(
			"m_ListMgr",
			"F_ListMgr",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_ListMgr,
			(char *)szAddress,
			(bHighlight = ~bHighlight));

		// m_LNodes (show address)
		printAddress( &pFileHdlMgr->m_LNodes[0], szAddress);
		printHTMLString(
			"m_LNodes",
			"LNODE[]",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_LNodes,
			(char *)szAddress,
			(bHighlight = ~bHighlight));

		// m_bIsSetup
		printHTMLString(
			"m_bIsSetup",
			"FLMBOOL",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_bIsSetup,
			(char *)(pFileHdlMgr->m_bIsSetup ? "Yes" : "No"),
			(bHighlight = ~bHighlight));

		// m_uiFileIdCounter
		printHTMLUint(
			"m_uiFileIdCounter",
			"FLMUINT",
			(void *)pFileHdlMgr,
			(void *)&pFileHdlMgr->m_uiFileIdCounter,
			pFileHdlMgr->m_uiFileIdCounter,
			(bHighlight = ~bHighlight));

		// Now show the private member(s) of the F_Base class

		pBase = (F_Base *)pFileHdlMgr;

		// m_i32RefCnt
		printHTMLInt(
			"F_Base.m_i32RefCnt",
			"FLMINT32",
			(void *)pBase,
			(void *)&pBase->m_i32RefCnt,
			pBase->m_i32RefCnt,
			(bHighlight = ~bHighlight));


		printTableEnd();

		
		fnPrintf( m_pHRequest, "<br>\n");
		fnPrintf( m_pHRequest, "<center>\n");
		fnPrintf( m_pHRequest, "Shown below are the AVAILABLE and the USED File "
								  "Handle Lists.<BR>To select a File Handle to view, "
								  "you may click the appropriate \"NEXT FILE HANDLE\" "
								  "button or choose from the drop down selection list."
								  "\n");
		fnPrintf( m_pHRequest, "</center>\n");
		fnPrintf( m_pHRequest, "<br>\n");
		
		// Define the form to present the File Handle Lists
		// Begin with the Available List of File Handles


		// Get a reference to the list manager, then find out
		// how many items are in the Available List
		pListMgr = pFileHdlMgr->GetListMgr();
		uiNumAvailItems = pListMgr->GetCount( FHM_AVAIL_LIST);

		if (uiNumAvailItems > 0)
		{
			fnPrintf( m_pHRequest, "<form name=\"AvailSelection\" type=\"submit\" "
									  "method=\"get\" action=\"%s/FileHdl?Avail=Yes\">\n",
									  m_pszURLString);
			fnPrintf( m_pHRequest, "<center><H2>Available List</H2></center>\n");

			fnPrintf( m_pHRequest, "<center>\n");
		
			printButton( "Next File Handle", BT_Button, NULL, NULL,
				"ONCLICK='nextAvailHdl(document.AvailSelection.AvailOption)'");
			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");
			fnPrintf( m_pHRequest, "or select a specific file handle to view\n");
			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");

			fnPrintf( m_pHRequest, "<SELECT NAME=\"AvailOption\""
							"onChange=\"this.form.FileId.value = this.form."
							"AvailOption.options[this.form.AvailOption."
							"selectedIndex].text\">\n");

			for (uiLoop = 0; uiLoop < uiNumAvailItems; uiLoop++)
			{
				fnPrintf( m_pHRequest, "<OPTION> %ld\n", uiLoop);
			}

			fnPrintf( m_pHRequest, "</SELECT>\n");
		
			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");

			printButton( "Submit", BT_Submit);
			fnPrintf( m_pHRequest, "</center>\n");

			fnPrintf( m_pHRequest, "<INPUT name=\"From\" type=hidden "
									  "value=\"FileHdlMgr\"></INPUT>\n");
			fnPrintf( m_pHRequest, "<INPUT name=\"FileId\" type=hidden "
									  "value=0></INPUT>\n");
			fnPrintf( m_pHRequest, "<INPUT name=\"List\" type=hidden "
									  "value=\"Avail\"></INPUT>\n");

			fnPrintf( m_pHRequest, "</form>\n");
		}
		else
		{
			fnPrintf( m_pHRequest, "<center><H2>Available List - "
									  "No Entries</H2></center>\n");
		}


		uiNumUsedItems = pListMgr->GetCount( FHM_USED_LIST);


		if (uiNumUsedItems > 0)
		{
			fnPrintf( m_pHRequest, "<form name=\"UsedSelection\" type=\"submit\" "
									  "method=\"get\" action=\"%s/FileHdl\">\n",
									  m_pszURLString);
			fnPrintf( m_pHRequest, "<CENTER><H2>Used List</H2></CENTER>\n");

			fnPrintf( m_pHRequest, "<center>\n");
			printButton( "Next File Handle", BT_Button, NULL, NULL,
				"ONCLICK='nextUsedHdl(document.UsedSelection.UsedOption)'");
			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");
			fnPrintf( m_pHRequest, "or select a specific file handle to view\n");
			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");

			fnPrintf( m_pHRequest, "<SELECT NAME=\"UsedOption\"\n"
									  "onChange=\"this.form.FileId.value = "
									  "this.form.UsedOption.options[this.form."
									  "UsedOption.selectedIndex].text\">\n");
			for (uiLoop=0; uiLoop < uiNumUsedItems; uiLoop++)
			{
				fnPrintf( m_pHRequest, "<OPTION> %ld\n", uiLoop);
			}
			fnPrintf( m_pHRequest, "</SELECT>\n");

			fnPrintf( m_pHRequest, "&nbsp&nbsp\n");

			printButton( "Submit", BT_Submit);
			fnPrintf( m_pHRequest, "</center>\n");

			fnPrintf( m_pHRequest, "<INPUT name=\"From\" type=hidden "
									  "value=\"FileHdlMgr\"></INPUT>\n");
			fnPrintf( m_pHRequest, "<INPUT name=\"FileId\" type=hidden "
									  "value=0></INPUT>\n");
			fnPrintf( m_pHRequest, "<INPUT name=\"List\" type=hidden "
									  "value=\"Used\"></INPUT>\n");

			fnPrintf( m_pHRequest, "</form>\n");

		}
		else
		{
			fnPrintf( m_pHRequest, "<center><H2>Used List - "
									  "No Entries</H2></center>\n");
		}


		fnPrintf( m_pHRequest, "<SCRIPT>\n");

		// nextAvailHdl function - increments the selection option to the next
		// available option.  If at the end, it will wrap to the beginning.
		
		fnPrintf( m_pHRequest, "function nextAvailHdl(selectObj)\n");
		fnPrintf( m_pHRequest, "{\nvar FileId\n");

		fnPrintf( m_pHRequest, "switch (selectObj.selectedIndex)\n{\n");

		for (uiLoop = 0; uiLoop < uiNumAvailItems; uiLoop++)
		{
			fnPrintf( m_pHRequest, "case %ld:\n", uiLoop);
			fnPrintf( m_pHRequest, "\tFileId=%ld\n", (uiLoop + 1 < uiNumAvailItems ? uiLoop + 1 : 0));
			fnPrintf( m_pHRequest, "\tselectObj.selectedIndex=%ld\n",
									  (uiLoop + 1 < uiNumAvailItems ? uiLoop + 1 : 0));
			fnPrintf( m_pHRequest, "\tbreak\n");
		}
		fnPrintf( m_pHRequest, "default:\n\tbreak\n");
		fnPrintf( m_pHRequest, "}\n");
		fnPrintf( m_pHRequest, "document.AvailSelection.FileId.value = "
								  "FileId\n}\n");


		// nextUsedHdl function - increments the selection option to the next
		// available option.  If at the end, it will wrap to the beginning.
		
		fnPrintf( m_pHRequest, "function nextUsedHdl(selectObj)\n");
		fnPrintf( m_pHRequest, "{\nvar FileId\n");

		fnPrintf( m_pHRequest, "switch (selectObj.selectedIndex)\n{\n");

		for (uiLoop = 0; uiLoop < uiNumUsedItems; uiLoop++)
		{
			fnPrintf( m_pHRequest, "case %ld:\n", uiLoop);
			fnPrintf( m_pHRequest, "\tFileId=%ld\n",
									  (uiLoop + 1 < uiNumUsedItems ? uiLoop + 1 : 0));
			fnPrintf( m_pHRequest, "\tselectObj.selectedIndex=%ld\n", 
									  (uiLoop + 1 < uiNumUsedItems ? uiLoop + 1 : 0));
			fnPrintf( m_pHRequest, "\tbreak\n");
		}
		fnPrintf( m_pHRequest, "default:\n\tbreak\n");
		fnPrintf( m_pHRequest, "}\n");
		fnPrintf( m_pHRequest, "document.UsedSelection.FileId.value = FileId\n}\n");

		fnPrintf( m_pHRequest, "</SCRIPT>\n");

		// Now release the FileHdlMgr...
		f_mutexLock( gv_FlmSysData.hFileHdlMutex);
		pFileHdlMgr->Release();
		pFileHdlMgr = NULL;
		f_mutexUnlock( gv_FlmSysData.hFileHdlMutex);


	} // else

	fnPrintf( m_pHRequest, "</body></html>\n");

	fnEmit();
	
Exit:

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	if (pszTemp1)
	{
		f_free( &pszTemp1);
	}

	return( rc);
}
