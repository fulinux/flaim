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
RCODE FLMAPI f_makeErr(
	RCODE				rc,
	const char *,	// pszFile,
	int,				// iLine,
	FLMBOOL			bAssert)
{
	if( rc == NE_FLM_OK)
	{
		return( NE_FLM_OK);
	}

#if defined( FLM_DEBUG)
	if( bAssert)
	{
		f_assert( 0);
	}
#else
	F_UNREFERENCED_PARM( bAssert);
#endif

	return( rc);
}
#endif

/***************************************************************************
Desc:   Map POSIX errno to Flaim IO errors.
***************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_LIBC_NLM)
RCODE FLMAPI f_mapPlatformError(
	FLMINT	iError,
	RCODE		defaultRc)
{
	switch (iError)
	{
		case 0:
		{
			return( NE_FLM_OK);
		}

		case ENOENT:
		{
			return( RC_SET( NE_FLM_IO_PATH_NOT_FOUND));
		}

		case EACCES:
		case EEXIST:
		{
			return( RC_SET( NE_FLM_IO_ACCESS_DENIED));
		}

		case EINVAL:
		{
			return( RC_SET( NE_FLM_IO_PATH_TOO_LONG));
		}

		case EIO:
		{
			return( RC_SET( NE_FLM_IO_DISK_FULL));
		}

		case ENOTDIR:
		{
			return( RC_SET( NE_FLM_IO_DIRECTORY_ERR));
		}

#ifdef EBADFD
		case EBADFD:
		{
			return( RC_SET( NE_FLM_IO_BAD_FILE_HANDLE));
		}
#endif

#ifdef EOF
		case EOF:
		{
			return( RC_SET( NE_FLM_IO_END_OF_FILE));
		}
#endif
			
		case EMFILE:
		{
			return( RC_SET( NE_FLM_IO_NO_MORE_FILES));
		}

		default:
		{
			return( RC_SET( defaultRc));
		}
	}
}
#endif

/***************************************************************************
Desc:
***************************************************************************/
#ifdef FLM_WIN
RCODE FLMAPI f_mapPlatformError(
	FLMINT	iErrCode,
	RCODE		defaultRc)
{
	switch( iErrCode)
	{
		case ERROR_NOT_ENOUGH_MEMORY:
		case ERROR_OUTOFMEMORY:
			return( RC_SET( NE_FLM_MEM));
			
		case ERROR_BAD_NETPATH:
		case ERROR_BAD_PATHNAME:
		case ERROR_DIRECTORY:
		case ERROR_FILE_NOT_FOUND:
		case ERROR_INVALID_DRIVE:
		case ERROR_INVALID_NAME:
		case ERROR_NO_NET_OR_BAD_PATH:
		case ERROR_PATH_NOT_FOUND:
			return( RC_SET( NE_FLM_IO_PATH_NOT_FOUND));

		case ERROR_ACCESS_DENIED:
		case ERROR_SHARING_VIOLATION:
		case ERROR_FILE_EXISTS:
		case ERROR_ALREADY_EXISTS:
			return( RC_SET( NE_FLM_IO_ACCESS_DENIED));

		case ERROR_BUFFER_OVERFLOW:
		case ERROR_FILENAME_EXCED_RANGE:
			return( RC_SET( NE_FLM_IO_PATH_TOO_LONG));

		case ERROR_DISK_FULL:
		case ERROR_HANDLE_DISK_FULL:
			return( RC_SET( NE_FLM_IO_DISK_FULL));

		case ERROR_CURRENT_DIRECTORY:
		case ERROR_DIR_NOT_EMPTY:
			return( RC_SET( NE_FLM_IO_DIRECTORY_ERR));

		case ERROR_DIRECT_ACCESS_HANDLE:
		case ERROR_INVALID_HANDLE:
		case ERROR_INVALID_TARGET_HANDLE:
			return( RC_SET( NE_FLM_IO_BAD_FILE_HANDLE));

		case ERROR_HANDLE_EOF:
			return( RC_SET( NE_FLM_IO_END_OF_FILE));

		case ERROR_OPEN_FAILED:
			return( RC_SET( NE_FLM_IO_OPEN_ERR));

		case ERROR_CANNOT_MAKE:
			return( RC_SET( NE_FLM_IO_PATH_CREATE_FAILURE));

		case ERROR_LOCK_FAILED:
		case ERROR_LOCK_VIOLATION:
			return( RC_SET( NE_FLM_IO_FILE_LOCK_ERR));

		case ERROR_NEGATIVE_SEEK:
		case ERROR_SEEK:
		case ERROR_SEEK_ON_DEVICE:
			return( RC_SET( NE_FLM_IO_SEEK_ERR));

		case ERROR_NO_MORE_FILES:
		case ERROR_NO_MORE_SEARCH_HANDLES:
			return( RC_SET( NE_FLM_IO_NO_MORE_FILES));

		case ERROR_TOO_MANY_OPEN_FILES:
			return( RC_SET( NE_FLM_IO_TOO_MANY_OPEN_FILES));

		case NO_ERROR:
			return( NE_FLM_OK);

		case ERROR_DISK_CORRUPT:
		case ERROR_DISK_OPERATION_FAILED:
		case ERROR_FILE_CORRUPT:
		case ERROR_FILE_INVALID:
		case ERROR_NOT_SAME_DEVICE:
		case ERROR_IO_DEVICE:
		default:
			return( RC_SET( defaultRc));

   }
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI f_enterDebugger(
	const char *	pszFile,
	int				iLine)
{
#ifdef FLM_WIN
	fprintf( stderr, "Assertion failed in %s on line %d\n", pszFile, iLine);
	fflush( stderr);
	DebugBreak();
#elif defined( FLM_NLM)
	(void)pszFile;
	(void)iLine;
	EnterDebugger();
#else
	fprintf( stderr, "Assertion failed in %s on line %d\n", pszFile, iLine);
	fflush( stderr);
	assert( 0);
#endif

	return( 0);
}

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_WATCOM_NLM)
int gv_ftkerrorDummy(void)
{
	return( 0);
}
#endif
