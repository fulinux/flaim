//-------------------------------------------------------------------------
// Desc:	Read and parse database header into a structure.
// Tabs:	3
//
//		Copyright (c) 1991-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flgethdr.cpp 12262 2006-01-19 14:42:10 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/***************************************************************************
Desc:	This routine reads the header information in a FLAIM database,
		verifies the password, and returns the file header and log
		header information.
*****************************************************************************/
RCODE flmGetHdrInfo(
	F_SuperFileHdl *	pSFileHdl,		/* Pointer to file handle. */
	FILE_HDR *			pFileHdrRV,		/* Returns file header information. */
	LOG_HDR *			pLogHdrRV,		/* Returns log header information. */
	FLMBYTE *			pLogHdr
	)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		pBuf = NULL;
	F_FileHdlImp *	pCFileHdl;

	if (RC_BAD( rc = f_alloc( 2048, &pBuf)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pSFileHdl->GetFileHdl( 0, FALSE, &pCFileHdl)))
	{
		goto Exit;
	}

	rc = flmReadAndVerifyHdrInfo( NULL, pCFileHdl,
											pBuf, pFileHdrRV, pLogHdrRV, pLogHdr);

Exit:

	if( pBuf)
	{
		f_free( &pBuf);
	}

	return( rc);
}
