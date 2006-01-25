//-------------------------------------------------------------------------
// Desc:	Client/Server stream dispatcher and handler.
// Tabs:	3
//
//		Copyright (c) 1999-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsv_sev.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/***************************************************************************
Desc: Receives stream events and dispatches them to the appropriate handlers
*****************************************************************************/
RCODE flmStreamEventDispatcher(
	FCS_BIOS_p		pStream,
	FLMUINT			uiEvent,
	void *			UserData)
{
	CS_CONTEXT_p	pCSContext = (CS_CONTEXT_p)UserData;
	FLMUINT			uiStreamHandlerId = FSEV_HANDLER_UNKNOWN;
	RCODE				rc	= FERR_OK;

	/*
	Determine the handler
	*/

	if( pCSContext->uiStreamHandlerId == FSEV_HANDLER_UNKNOWN)
	{
		if( f_stricmp( pCSContext->pucAddr, "DS") == 0)
		{
			uiStreamHandlerId = FSEV_HANDLER_DS;
		}
		else if( f_stricmp( pCSContext->pucAddr, "LOOPBACK") == 0)
		{
			uiStreamHandlerId = FSEV_HANDLER_LOOPBACK;
		}

		pCSContext->uiStreamHandlerId = uiStreamHandlerId;
	}
	else
	{
		uiStreamHandlerId = pCSContext->uiStreamHandlerId;
	}

	/*
	Invoke the handler
	*/

	switch( uiStreamHandlerId)
	{
		case FSEV_HANDLER_LOOPBACK:
		{
			if( RC_BAD( rc = fsvStreamLoopback( pStream, uiEvent, UserData)))
			{
				goto Exit;
			}
			break;
		}

		default:
		{
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

	/*
	Release CPU to prevent CPU hog
	*/

	f_yieldCPU();

Exit:

	if( RC_BAD( rc))
	{
		/*
		Clear the saved handler ID in case a new handler is tried
		*/

		pCSContext->uiStreamHandlerId = FSEV_HANDLER_UNKNOWN;
	}

	return( rc);
}

/****************************************************************************
Desc:    Provides loopback support for C/S testing
*****************************************************************************/
RCODE fsvStreamLoopback(
	FCS_BIOS_p	pStream,
	FLMUINT		uiEvent,
	void *	UserData)
{
	CS_CONTEXT_p			pCSContext = (CS_CONTEXT_p)UserData;
	FCS_DIS					dataIStream;
	FCS_DOS					dataOStream;
	RCODE						rc = FERR_OK;

	F_UNREFERENCED_PARM( pStream);

	if( uiEvent == FCS_BIOS_EOM_EVENT)
	{
		if( RC_BAD( rc = dataIStream.setup(
			(FCS_BIOS_p)(pCSContext->pOStream))))
		{
			goto Exit;
		}

		if( RC_BAD( rc = dataOStream.setup(
			(FCS_BIOS_p)(pCSContext->pIStream))))
		{
			goto Exit;
		}

		if( RC_BAD( rc = fsvProcessRequest( &dataIStream, &dataOStream,
			&(pCSContext->pool), NULL)))
		{
			goto Exit;
		}
	}

Exit:
	
	return( rc);
}
