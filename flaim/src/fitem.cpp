//-------------------------------------------------------------------------
// Desc:	Functions for doing id-to-name and name-to-id mapping.
// Tabs:	3
//
//		Copyright (c) 1995-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fitem.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*******************************************************************************
Desc:		Retrieves a dictionary item name.
Notes:	Given an item ID, this routine will search a specified shared or
		 	local dictionary for the item.  If it is found, the name
			of the item will be returned.  This routine supports version 2.0 and
			higher databases only.
*******************************************************************************/
RCODE FlmGetItemName(
	HFDB			hDb,
	FLMUINT		uiItemId,
	FLMUINT		uiNameBufSize,
	char *		pszNameBuf)
{
	RCODE				rc = FERR_OK;
	FlmRecord *		pRecord = NULL;

	*pszNameBuf = 0;
	if( RC_BAD( rc = FlmRecordRetrieve( hDb,
		FLM_DICT_CONTAINER, uiItemId, FO_EXACT, &pRecord, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRecord->getNative( pRecord->root(), 
		pszNameBuf, &uiNameBufSize)))
	{
		goto Exit;
	}

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	return (rc == FERR_EOF_HIT) ? RC_SET( FERR_NOT_FOUND) : rc;
}
