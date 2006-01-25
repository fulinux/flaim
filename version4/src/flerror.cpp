//-------------------------------------------------------------------------
// Desc:	Error routines.
// Tabs:	3
//
//		Copyright (c) 1997-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flerror.cpp 12262 2006-01-19 14:42:10 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

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
	int				iLine)
{
	if( rc == FERR_OK)
		return FERR_OK;
		
	// Switch on warning type return codes
	if( rc <= FERR_NOT_FOUND)
	{
		switch(rc)
		{
			case FERR_BOF_HIT:
				break;
			case FERR_EOF_HIT:
				break;
			case FERR_END:
				break;
			case FERR_EXISTS:
				break;
			case FERR_FAILURE:
				break;
			case FERR_NOT_FOUND:
				break;
			default:
				break;
		}
		return( rc);
	}

	switch(rc)
	{
		case FERR_DATA_ERROR:
		case FERR_BAD_RFL_PACKET:
			flmAssert( 0);
			flmLogError( rc, "", pszFile, iLine);
			break;
		case FERR_BTREE_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case FERR_MEM:
			break;
		case FERR_OLD_VIEW:
			break;
		case FERR_SYNTAX:
			break;
		case FERR_BLOCK_CHECKSUM:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case FERR_CACHE_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case FERR_BLOB_MISSING_FILE:
			break;
		case FERR_CONV_BAD_DIGIT:
			break;
		case FERR_NOT_IMPLEMENTED:
			break;
		case FERR_BAD_REFERENCE:
			break;
		case FERR_IO_ACCESS_DENIED:
			break;
		case FERR_IO_PATH_NOT_FOUND:
			break;
		case FERR_UNSUPPORTED_FEATURE:
			break;
		case FERR_ENCRYPTION_UNAVAILABLE:
			break;
		default:
			rc = rc;
			break;
	}

	return( rc);
}
#endif

#if defined( FLM_NLM) && !defined( __MWERKS__)
	int gv_iFlerrorDummy(void)
	{
		return( 0);
	}
#endif
