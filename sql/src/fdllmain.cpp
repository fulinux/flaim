//------------------------------------------------------------------------------
// Desc: This is the standard functionality that all com servers must export
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
// $Id: fdllmain.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

static F_DbSystem *			gv_pDbSystem = NULL;
static FLMATOMIC				gv_lockCount = 0;

FLMEXTC RCODE DllCanUnloadNow( void);
FLMEXTC RCODE DllStart( void);
FLMEXTC RCODE DllStop( void);

#if defined( FLM_UNIX)

#elif defined( FLM_WIN)

	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	
	#pragma comment(linker, "/export:DllCanUnloadNow=_DllCanUnloadNow@0,PRIVATE")
	#pragma comment(linker, "/export:DllStart=_DllStart@0,PRIVATE")
	#pragma comment(linker, "/export:DllStop=_DllStop@0,PRIVATE")

#elif !defined( FLM_NLM)
	#error platform not supported.
#endif

/******************************************************************************
Desc:
******************************************************************************/
void LockModule(void)
{
	f_atomicInc( &gv_lockCount);
}

/******************************************************************************
Desc:
******************************************************************************/
void UnlockModule(void)
{
	f_atomicDec( &gv_lockCount);
}

/******************************************************************************
Desc:	Returns 0 if it's okay to unload, or a non-zero status
		code if not.
******************************************************************************/
FLMEXTC RCODE DllCanUnloadNow( void)
{
	RCODE		rc = NE_SFLM_OK;

	flmAssert( gv_pDbSystem);

	if( gv_lockCount > 1)
	{
		rc = RC_SET( NE_SFLM_FAILURE);
	}
	else
	{
		flmAssert( gv_lockCount == 1);

		f_mutexLock( gv_SFlmSysData.hShareMutex);

		if (gv_SFlmSysData.pDatabaseHashTbl)
		{
			FBUCKET *   pDatabaseHashTbl;
			FLMUINT		uiCnt;

			for (uiCnt = 0, pDatabaseHashTbl = gv_SFlmSysData.pDatabaseHashTbl;
				uiCnt < FILE_HASH_ENTRIES;
				uiCnt++, pDatabaseHashTbl++)
			{
				if (pDatabaseHashTbl->pFirstInBucket != NULL)
				{
					rc = RC_SET( NE_SFLM_FAILURE);
					break;
				}
			}
		}

		f_mutexUnlock( gv_SFlmSysData.hShareMutex);
	}

	return( rc);
}

/******************************************************************************
Desc:	Called by PSA when it loads the library.  Must return 0 for
		success, or a non-zero error code.
******************************************************************************/
FLMEXTC RCODE DllStart( void)
{
	RCODE		rc = NE_SFLM_OK;
	
	if( (gv_pDbSystem = f_new F_DbSystem) == NULL)
	{
		rc = NE_SFLM_MEM;
		goto Exit;
	}

	if( RC_BAD( rc = gv_pDbSystem->init()))
	{
		goto Exit;
	}

Exit:

	if( RC_BAD( rc))
	{
		if( gv_pDbSystem)
		{
			gv_pDbSystem->Release();
			gv_pDbSystem = NULL;
		}
	}

	return( rc);
}

/******************************************************************************
Desc:	Called by PSA when it unloads the library.  The return value
		is ignored.
******************************************************************************/
FLMEXTC RCODE DllStop( void)
{
	if( gv_pDbSystem)
	{
		flmAssert( gv_lockCount == 1);

		gv_pDbSystem->exit();
		gv_pDbSystem->Release();
		gv_pDbSystem = NULL;
	}

	return( NE_SFLM_OK);
}

/******************************************************************************
Desc:
******************************************************************************/
FLMEXTC RCODE DllRegisterServer(
	const char *)
{
	return( NE_SFLM_OK);
}

/******************************************************************************
Desc:
******************************************************************************/
FLMEXTC RCODE DllUnregisterServer( void) 
{
	return( NE_SFLM_OK);
}
