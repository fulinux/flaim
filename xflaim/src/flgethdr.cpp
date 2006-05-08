//------------------------------------------------------------------------------
// Desc:	This file contains the flmGetHdrInfo routine -- a routine which
// 		reads the header information from a FLAIM database, verifies it
// 		and returns the header information in a structure.
//
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flgethdr.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/***************************************************************************
Desc:	This routine reads the header information in a FLAIM database,
		verifies the password, and returns the file header and log
		header information.
*****************************************************************************/
RCODE flmGetHdrInfo(
	F_SuperFileHdl *	pSFileHdl,
	XFLM_DB_HDR *		pDbHdr,
	FLMUINT32 *			pui32CalcCRC)
{
	RCODE				rc = NE_XFLM_OK;
	IF_FileHdl *	pCFileHdl;

	if( RC_BAD( rc = pSFileHdl->getFileHdl( 0, FALSE, &pCFileHdl)))
	{
		goto Exit;
	}

	rc = flmReadAndVerifyHdrInfo( NULL, pCFileHdl, pDbHdr, pui32CalcCRC);

Exit:

	return( rc);
}
