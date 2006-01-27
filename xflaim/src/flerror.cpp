//------------------------------------------------------------------------------
// Desc:	This file contains error routines that are used throughout FLAIM.
//
// Tabs:	3
//
//		Copyright (c) 1997-2000, 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flerror.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:	The primary purpose of this function is to provide a way to easily
		trap errors when they occur.  Just put a breakpoint in this function
		to catch them.
Note:	Some of the most common errors will be coded so the use can set a
		break point.
****************************************************************************/
#ifdef FLM_DEBUG
RCODE flmMakeErr(
	RCODE				rc,
	const char *	pszFile,
	int				iLine,
	FLMBOOL			bAssert)
{
	if( rc == NE_XFLM_OK)
	{
		return NE_XFLM_OK;
	}

	// Switch on warning type return codes
	if( rc <= NE_XFLM_NOT_FOUND)
	{
		switch(rc)
		{
			case NE_XFLM_BOF_HIT:
				break;
			case NE_XFLM_EOF_HIT:
				break;
			case NE_XFLM_END:
				break;
			case NE_XFLM_EXISTS:
				break;
			case NE_XFLM_NOT_FOUND:
				break;
		}

		goto Exit;
	}

	switch(rc)
	{
		case NE_XFLM_IO_BAD_FILE_HANDLE:
			break;
		case NE_XFLM_DATA_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_BTREE_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_MEM:
			break;
		case NE_XFLM_OLD_VIEW:
			break;
		case NE_XFLM_SYNTAX:
			break;
		case NE_XFLM_BLOCK_CRC:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_CACHE_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_NOT_IMPLEMENTED:
			break;
		case NE_XFLM_CONV_DEST_OVERFLOW:
			break;
		case NE_XFLM_KEY_OVERFLOW:
			break;
		case NE_XFLM_FAILURE:
			break;
		case NE_XFLM_ILLEGAL_OP:
			break;
		case NE_XFLM_BAD_COLLECTION:
			break;
		default:
			rc = rc;
			break;
	}

Exit:
	
#if defined( FLM_DEBUG)
	if( bAssert)
	{
		flmAssert( 0);
	}
#else
	F_UNREFERENCED_PARM( bAssert);
#endif

	return( rc);
}
#endif

#if defined( FLM_WATCOM_NLM)
	int gv_iFlerrorDummy(void)
	{
		return( 0);
	}
#endif
