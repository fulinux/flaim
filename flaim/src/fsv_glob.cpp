//-------------------------------------------------------------------------
// Desc:	Server global context.
// Tabs:	3
//
//		Copyright (c) 1998-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_glob.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSV_SCTX *		gv_pGlobalContext = NULL;

/****************************************************************************
Desc:	Initializes the server's global context.
*****************************************************************************/
RCODE fsvInitGlobalContext(
	FLMUINT			uiMaxSessions,
	const char *	pszServerBasePath,
	FSV_LOG_FUNC	pLogFunc)
{
	RCODE				rc = FERR_OK;
	FSV_SCTX	*		pTmpContext = NULL;

	if( gv_pGlobalContext)
	{
		// Context already initialized
		goto Exit;
	}
	
	if( (pTmpContext = f_new FSV_SCTX) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pTmpContext->Setup( uiMaxSessions,
		pszServerBasePath, pLogFunc)))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		if( pTmpContext)
		{
			pTmpContext->Release();
		}
	}
	else if( pTmpContext)
	{
		gv_pGlobalContext = pTmpContext;
	}

	return( rc);

}


/****************************************************************************
Desc:	Frees any resources allocated to the global context.	
*****************************************************************************/
void fsvFreeGlobalContext( void)
{
	if( gv_pGlobalContext)
	{
		gv_pGlobalContext->Release();
		gv_pGlobalContext = NULL;
	}
}


/****************************************************************************
Desc:	Sets the server's base (relative) path
*****************************************************************************/
RCODE fsvSetBasePath(
	const char *	pszServerBasePath)
{
	RCODE				rc = FERR_OK;

	if( !gv_pGlobalContext)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_pGlobalContext->SetBasePath( pszServerBasePath)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Sets the server's temporary directory
*****************************************************************************/
RCODE fsvSetTempDir(
	const char *	pszTempDir)
{
	RCODE				rc = FERR_OK;

	if( !gv_pGlobalContext)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	if( RC_BAD( rc = gv_pGlobalContext->SetTempDir( pszTempDir)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns a pointer to the server's global context object.
*****************************************************************************/
RCODE	fsvGetGlobalContext(
	FSV_SCTX **		ppGlobalContext)
{
	*ppGlobalContext = gv_pGlobalContext;
	return( FERR_OK);
}
