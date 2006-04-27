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

#define flmErrorCodeEntry(c)		{ c, #c }

typedef struct
{
	RCODE				rc;
	const char *	pszErrorStr;
} F_ERROR_CODE_MAP;

/****************************************************************************
Desc:
****************************************************************************/
F_ERROR_CODE_MAP gv_FlmGeneralErrors[
	NE_FLM_LAST_GENERAL_ERROR - NE_FLM_FIRST_GENERAL_ERROR - 1] =
{
	flmErrorCodeEntry( NE_FLM_NOT_IMPLEMENTED),
	flmErrorCodeEntry( NE_FLM_MEM),
	flmErrorCodeEntry( NE_FLM_INVALID_PARM),
	flmErrorCodeEntry( NE_FLM_TIMEOUT),
	flmErrorCodeEntry( NE_FLM_NOT_FOUND),
	flmErrorCodeEntry( NE_FLM_EXISTS),
	flmErrorCodeEntry( NE_FLM_USER_ABORT),
	flmErrorCodeEntry( NE_FLM_FAILURE),
	flmErrorCodeEntry( NE_FLM_BOF_HIT),
	flmErrorCodeEntry( NE_FLM_EOF_HIT),
	flmErrorCodeEntry( NE_FLM_END),
	flmErrorCodeEntry( NE_FLM_CONV_BAD_DIGIT),
	flmErrorCodeEntry( NE_FLM_CONV_DEST_OVERFLOW),
	flmErrorCodeEntry( NE_FLM_CONV_ILLEGAL),
	flmErrorCodeEntry( NE_FLM_CONV_NULL_SRC),
	flmErrorCodeEntry( NE_FLM_CONV_NUM_OVERFLOW),
	flmErrorCodeEntry( NE_FLM_CONV_NUM_UNDERFLOW),
	flmErrorCodeEntry( NE_FLM_SYNTAX),
	flmErrorCodeEntry( NE_FLM_UNSUPPORTED_FEATURE),
	flmErrorCodeEntry( NE_FLM_FILE_EXISTS),
	flmErrorCodeEntry( NE_FLM_COULD_NOT_CREATE_SEMAPHORE),
	flmErrorCodeEntry( NE_FLM_BAD_UTF8),
	flmErrorCodeEntry( NE_FLM_ERROR_WAITING_ON_SEMPAHORE),
	flmErrorCodeEntry( NE_FLM_BAD_PLATFORM_FORMAT),
	flmErrorCodeEntry( NE_FLM_BAD_SEN),
	flmErrorCodeEntry( NE_FLM_UNSUPPORTED_INTERFACE),
	flmErrorCodeEntry( NE_FLM_BAD_RCODE_TABLE),
	flmErrorCodeEntry( NE_FLM_BUFFER_OVERFLOW),
	flmErrorCodeEntry( NE_FLM_INVALID_XML),
	flmErrorCodeEntry( NE_FLM_ILLEGAL_FLAG),
	flmErrorCodeEntry( NE_FLM_ILLEGAL_OP),
	flmErrorCodeEntry( NE_FLM_COULD_NOT_START_THREAD),
	flmErrorCodeEntry( NE_FLM_BAD_BASE64_ENCODING),
	flmErrorCodeEntry( NE_FLM_STREAM_EXISTS),
	flmErrorCodeEntry( NE_FLM_MULTIPLE_MATCHES),
	flmErrorCodeEntry( NE_FLM_NOT_UNIQUE),
	flmErrorCodeEntry( NE_FLM_BTREE_ERROR),
	flmErrorCodeEntry( NE_FLM_BTREE_KEY_SIZE),
	flmErrorCodeEntry( NE_FLM_BTREE_FULL),
	flmErrorCodeEntry( NE_FLM_BTREE_BAD_STATE)
};

/****************************************************************************
Desc:
****************************************************************************/
F_ERROR_CODE_MAP gv_FlmIoErrors[
	NE_FLM_LAST_IO_ERROR - NE_FLM_FIRST_IO_ERROR - 1] =
{
	flmErrorCodeEntry( NE_FLM_IO_ACCESS_DENIED),
	flmErrorCodeEntry( NE_FLM_IO_BAD_FILE_HANDLE),
	flmErrorCodeEntry( NE_FLM_IO_COPY_ERR),
	flmErrorCodeEntry( NE_FLM_IO_DISK_FULL),
	flmErrorCodeEntry( NE_FLM_IO_END_OF_FILE),
	flmErrorCodeEntry( NE_FLM_IO_OPEN_ERR),
	flmErrorCodeEntry( NE_FLM_IO_SEEK_ERR),
	flmErrorCodeEntry( NE_FLM_IO_DIRECTORY_ERR),
	flmErrorCodeEntry( NE_FLM_IO_PATH_NOT_FOUND),
	flmErrorCodeEntry( NE_FLM_IO_TOO_MANY_OPEN_FILES),
	flmErrorCodeEntry( NE_FLM_IO_PATH_TOO_LONG),
	flmErrorCodeEntry( NE_FLM_IO_NO_MORE_FILES),
	flmErrorCodeEntry( NE_FLM_IO_DELETING_FILE),
	flmErrorCodeEntry( NE_FLM_IO_FILE_LOCK_ERR),
	flmErrorCodeEntry( NE_FLM_IO_FILE_UNLOCK_ERR),
	flmErrorCodeEntry( NE_FLM_IO_PATH_CREATE_FAILURE),
	flmErrorCodeEntry( NE_FLM_IO_RENAME_FAILURE),
	flmErrorCodeEntry( NE_FLM_IO_INVALID_PASSWORD),
	flmErrorCodeEntry( NE_FLM_SETTING_UP_FOR_READ),
	flmErrorCodeEntry( NE_FLM_SETTING_UP_FOR_WRITE),
	flmErrorCodeEntry( NE_FLM_IO_CANNOT_REDUCE_PATH),
	flmErrorCodeEntry( NE_FLM_INITIALIZING_IO_SYSTEM),
	flmErrorCodeEntry( NE_FLM_FLUSHING_FILE),
	flmErrorCodeEntry( NE_FLM_IO_INVALID_FILENAME),
	flmErrorCodeEntry( NE_FLM_IO_CONNECT_ERROR),
	flmErrorCodeEntry( NE_FLM_OPENING_FILE),
	flmErrorCodeEntry( NE_FLM_DIRECT_OPENING_FILE),
	flmErrorCodeEntry( NE_FLM_CREATING_FILE),
	flmErrorCodeEntry( NE_FLM_DIRECT_CREATING_FILE),
	flmErrorCodeEntry( NE_FLM_READING_FILE),
	flmErrorCodeEntry( NE_FLM_DIRECT_READING_FILE),
	flmErrorCodeEntry( NE_FLM_WRITING_FILE),
	flmErrorCodeEntry( NE_FLM_DIRECT_WRITING_FILE),
	flmErrorCodeEntry( NE_FLM_POSITIONING_IN_FILE),
	flmErrorCodeEntry( NE_FLM_GETTING_FILE_SIZE),
	flmErrorCodeEntry( NE_FLM_TRUNCATING_FILE),
	flmErrorCodeEntry( NE_FLM_PARSING_FILE_NAME),
	flmErrorCodeEntry( NE_FLM_CLOSING_FILE),
	flmErrorCodeEntry( NE_FLM_GETTING_FILE_INFO),
	flmErrorCodeEntry( NE_FLM_EXPANDING_FILE),
	flmErrorCodeEntry( NE_FLM_CHECKING_FILE_EXISTENCE),
	flmErrorCodeEntry( NE_FLM_RENAMING_FILE),
	flmErrorCodeEntry( NE_FLM_SETTING_FILE_INFO)
};

/****************************************************************************
Desc:
****************************************************************************/
F_ERROR_CODE_MAP gv_FlmNetErrors[
	NE_FLM_LAST_NET_ERROR - NE_FLM_FIRST_NET_ERROR - 1] =
{
	flmErrorCodeEntry( NE_FLM_NOIP_ADDR),
	flmErrorCodeEntry( NE_FLM_SOCKET_FAIL),
	flmErrorCodeEntry( NE_FLM_CONNECT_FAIL),
	flmErrorCodeEntry( NE_FLM_BIND_FAIL),
	flmErrorCodeEntry( NE_FLM_LISTEN_FAIL),
	flmErrorCodeEntry( NE_FLM_ACCEPT_FAIL),
	flmErrorCodeEntry( NE_FLM_SELECT_ERR),
	flmErrorCodeEntry( NE_FLM_SOCKET_SET_OPT_FAIL),
	flmErrorCodeEntry( NE_FLM_SOCKET_DISCONNECT),
	flmErrorCodeEntry( NE_FLM_SOCKET_READ_FAIL),
	flmErrorCodeEntry( NE_FLM_SOCKET_WRITE_FAIL),
	flmErrorCodeEntry( NE_FLM_SOCKET_READ_TIMEOUT),
	flmErrorCodeEntry( NE_FLM_SOCKET_WRITE_TIMEOUT),
	flmErrorCodeEntry( NE_FLM_SOCKET_ALREADY_CLOSED)
};

/****************************************************************************
Desc:
****************************************************************************/
F_ERROR_CODE_MAP gv_FlmStreamErrors[
	NE_FLM_LAST_STREAM_ERROR - NE_FLM_FIRST_STREAM_ERROR - 1] =
{
	flmErrorCodeEntry( NE_FLM_STREAM_DECOMPRESS_ERROR),
	flmErrorCodeEntry( NE_FLM_STREAM_NOT_COMPRESSED),
	flmErrorCodeEntry( NE_FLM_STREAM_TOO_MANY_FILES)
};

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
Desc:	Returns a pointer to the ASCII string representation
		of a return code.
****************************************************************************/
const char * FLMAPI f_errorString(
	RCODE			rc)
{
	const char *		pszErrorStr;

	if( rc == NE_FLM_OK)
	{
		pszErrorStr = "NE_FLM_OK";
	}
	else if( rc > NE_FLM_FIRST_GENERAL_ERROR &&
		rc < NE_FLM_LAST_GENERAL_ERROR)
	{
		pszErrorStr = gv_FlmGeneralErrors[
			rc - NE_FLM_FIRST_GENERAL_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_FLM_FIRST_IO_ERROR &&
		rc < NE_FLM_LAST_IO_ERROR)
	{
		pszErrorStr = gv_FlmIoErrors[
			rc - NE_FLM_FIRST_IO_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_FLM_FIRST_NET_ERROR &&
		rc < NE_FLM_LAST_NET_ERROR)
	{
		pszErrorStr = gv_FlmNetErrors[
			rc - NE_FLM_FIRST_NET_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_FLM_FIRST_STREAM_ERROR &&
		rc < NE_FLM_LAST_STREAM_ERROR)
	{
		pszErrorStr = gv_FlmStreamErrors[
			rc - NE_FLM_FIRST_STREAM_ERROR - 1].pszErrorStr;
	}
	else
	{
		pszErrorStr = "Unknown error";
	}

	return( pszErrorStr);
}

/***************************************************************************
Desc:   Map POSIX errno to Flaim IO errors.
***************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_NLM)
RCODE MapPlatformError(
	FLMINT	iError,
	RCODE		defaultRc)
{
	switch (err)
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
RCODE MapPlatformError(
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
Desc:	Checks the error code mapping tables on startup
****************************************************************************/
RCODE f_checkErrorCodeTables( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLoop;

	for( uiLoop = 0;
		uiLoop < (NE_FLM_LAST_GENERAL_ERROR - NE_FLM_FIRST_GENERAL_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmGeneralErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_FLM_FIRST_GENERAL_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_FLM_LAST_IO_ERROR - NE_FLM_FIRST_IO_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmIoErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_FLM_FIRST_IO_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_FLM_LAST_NET_ERROR - NE_FLM_FIRST_NET_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmNetErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_FLM_FIRST_NET_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_FLM_LAST_STREAM_ERROR - NE_FLM_FIRST_STREAM_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmStreamErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_FLM_FIRST_STREAM_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

Exit:

	return( rc);
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
