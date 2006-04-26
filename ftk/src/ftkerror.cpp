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

#include "ftksys.h"

/****************************************************************************
Desc:	The primary purpose of this function is to provide a way to easily
		trap errors when they occur.  Just put a breakpoint in this function
		to catch them.
****************************************************************************/
#ifdef FLM_DEBUG
RCODE f_makeErr(
	RCODE				rc,
	const char *,	// pszFile,
	int,				// iLine,
	FLMBOOL			bAssert)
{
	if( rc == NE_FLM_OK)
	{
		return( NE_FLM_OK);
	}

	// Switch on warning type return codes
	
	if( rc <= NE_FLM_NOT_FOUND)
	{
		switch(rc)
		{
			case NE_FLM_BOF_HIT:
				break;
			case NE_FLM_EOF_HIT:
				break;
			case NE_FLM_END:
				break;
			case NE_FLM_EXISTS:
				break;
			case NE_FLM_NOT_FOUND:
				break;
		}

		goto Exit;
	}
	
	// Switch on errors

	switch( rc)
	{
		case NE_FLM_IO_BAD_FILE_HANDLE:
			break;
		case NE_FLM_MEM:
			break;
		case NE_FLM_SYNTAX:
			break;
		case NE_FLM_NOT_IMPLEMENTED:
			break;
		case NE_FLM_CONV_DEST_OVERFLOW:
			break;
		case NE_FLM_FAILURE:
			break;
		case NE_FLM_ILLEGAL_OP:
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

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WATCOM_NLM)
int gv_ftkerrorDummy(void)
{
	return( 0);
}
#endif
