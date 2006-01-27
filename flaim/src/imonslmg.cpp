//-------------------------------------------------------------------------
// Desc:	Class for displaying lock manager information in HTML on a web page.
// Tabs:	3
//
//		Copyright (c) 2002-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: imonslmg.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc: Implements the display function of the F_ServerLockMgrPage
*****************************************************************************/
RCODE F_ServerLockMgrPage::display(
	FLMUINT			uiNumParams,
	const char ** 	ppszParams)
{
	RCODE							rc = FERR_OK;
	ServerLockManager_p		pServerLockMgr;
	FLMBOOL						bHighlight = FALSE;
	FLMBOOL						bRefresh;
	char							szAddress[20];
	char *						pszTemp = NULL;

	if( RC_BAD( rc = f_alloc( 150, &pszTemp)))
	{
		printErrorPage( rc, TRUE, "Failed to allocate temporary buffer");
		goto Exit;
	}

	stdHdr();

	fnPrintf( m_pHRequest, HTML_DOCTYPE);
	fnPrintf( m_pHRequest, "<html>\n");

	// Get the pointer to the ServerLockManager
	pServerLockMgr = gv_FlmSysData.pServerLockMgr;

	if (!pServerLockMgr)
	{
		rc = RC_SET(FERR_NOT_FOUND);
		printErrorPage( rc, TRUE, "No ServerLockManager exists");
		goto Exit;
	}

	//Check to see if we are to refresh this page automatically.
	if ((bRefresh = DetectParameter( uiNumParams,
												ppszParams,
												"Refresh")) == TRUE)
	{
		//Send back the page with a refresh command in the header
		
		char	 szTemp[100];
		
		f_sprintf(szTemp, "%s/ServerLockManager?Refresh",
					m_pszURLString);

		fnPrintf( m_pHRequest, 
			"<HEAD>"
			"<META http-equiv=\"refresh\" content=\"5; url=%s\">"
			"<TITLE>Server Lock Manager</TITLE>\n",
			szTemp);

	}
	else
	{
		fnPrintf( m_pHRequest, 
			"<HEAD><TITLE>Server Lock Manager</TITLE>\n");
	}
	printStyle();
	fnPrintf( m_pHRequest, "</HEAD>\n");


	fnPrintf( m_pHRequest, "<body>\n");

	printTableStart( "ServerLockManager", 4, 100);

	// If we are not to refresh this page, then don't include the refresh meta command
	if (!bRefresh)
	{
		f_sprintf( pszTemp, 
			"<A HREF=%s/ServerLockManager?Refresh>Start Auto-refresh (5 sec.)</A>",
			m_pszURLString);
	}
	else
	{
		f_sprintf( pszTemp,
			"<A HREF=%s/ServerLockManager>Stop Auto-refresh</A>",
			m_pszURLString);
	}

	printTableRowStart();
	printColumnHeading( "", JUSTIFY_LEFT, FLM_IMON_COLOR_PUTTY_1, 4, 1, FALSE);
	fnPrintf( m_pHRequest, "<A HREF=%s/ServerLockManager>Refresh</A>, ",
								  m_pszURLString);
	fnPrintf( m_pHRequest, "%s\n", pszTemp);
	printColumnHeadingClose();

	printTableRowStart();
	printColumnHeading( "Byte Offset (hex)");
	printColumnHeading( "Field Name");
	printColumnHeading( "Field Type");
	printColumnHeading( "Value");
	printTableRowEnd();



	// m_phMutex
	printAddress( pServerLockMgr->m_phMutex, szAddress);
	printHTMLString(
		"m_phMutex",
		"F_MUTEX",
		(void *)pServerLockMgr,
		(void *)&pServerLockMgr->m_phMutex,
		szAddress,
		(bHighlight = ~bHighlight));

	// m_pHashTbl
	printAddress( pServerLockMgr->m_pHashTbl, szAddress);
	printHTMLString(
		"m_pHashTbl",
		"FBUCKET_p",
		(void *)pServerLockMgr,
		(void *)&pServerLockMgr->m_pHashTbl,
		szAddress,
		(bHighlight = ~bHighlight));

	// m_pFirstLockWaiter
	printAddress( pServerLockMgr->m_pFirstLockWaiter, szAddress);
	printHTMLString(
		"m_pFirstLockWaiter",
		"LOCK_WAITER",
		(void *)pServerLockMgr,
		(void *)&pServerLockMgr->m_pFirstLockWaiter,
		szAddress,
		(bHighlight = ~bHighlight));

	// m_uiNumAvail
	printHTMLUint(
		"m_uiNumAvail",
		"FLMUINT",
		(void *)pServerLockMgr,
		(void *)&pServerLockMgr->m_uiNumAvail,
		pServerLockMgr->m_uiNumAvail,
		(bHighlight = ~bHighlight));

	// m_pAvailLockList
	printAddress( pServerLockMgr->m_pAvailLockList, szAddress);
	printHTMLString(
		"m_pAvailLockList",
		"ServerLockObject *",
		(void *)pServerLockMgr,
		(void *)&pServerLockMgr->m_pAvailLockList,
		szAddress,
		(bHighlight = ~bHighlight));

	printTableEnd();


	printDocEnd();

	fnEmit();

Exit:

	if (pszTemp)
	{
		f_free( &pszTemp);
	}

	return( rc);
}


