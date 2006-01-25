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
#include "fcomfact.h"

static F_DbSystem *			gv_pDbSystem = NULL;
static FLMUINT32				gv_ui32LockCount = 0;

XFLMEXTC RCODE XFLMAPI DllCanUnloadNow( void);

XFLMEXTC RCODE XFLMAPI DllGetClassObject(
	RXFLMCLSID		rclsid,
	RXFLMIID			riid,
	void **			ppv);

XFLMEXTC RCODE XFLMAPI DllStart( void);

XFLMEXTC RCODE XFLMAPI DllStop( void);

#if defined( FLM_UNIX)

#ifdef __GNUC__
	void __attribute__ ((constructor)) flaim_init( void) {}
	void __attribute__ ((destructor)) flaim_fini( void) {}
#elif !defined( FLM_SOLARIS)
	extern "C" void _init(void) {}
	extern "C" void _fini(void) {}
#endif

#elif defined( FLM_WIN)

	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	
	static HMODULE s_module;
	
	#pragma comment(linker, "/export:DllCanUnloadNow=_DllCanUnloadNow@0,PRIVATE")
	#pragma comment(linker, "/export:DllGetClassObject=_DllGetClassObject@12,PRIVATE")
	#pragma comment(linker, "/export:DllStart=_DllStart@0,PRIVATE")
	#pragma comment(linker, "/export:DllStop=_DllStop@0,PRIVATE")
	#pragma comment(linker, "/export:_XTCOM_Table,DATA")

#elif !defined( FLM_NLM)
	#error platform not supported.
#endif

/******************************************************************************
Desc:
******************************************************************************/
void LockModule(void)
{
	ftkAtomicIncrement( &gv_ui32LockCount);
//	flmAssert( gv_ui32LockCount < 20);
}

/******************************************************************************
Desc:
******************************************************************************/
void UnlockModule(void)
{
	ftkAtomicDecrement( &gv_ui32LockCount);
}

/******************************************************************************
Desc:	Returns 0 if it's okay to unload, or a non-zero status
		code if not.
******************************************************************************/
XFLMEXTC RCODE XFLMAPI DllCanUnloadNow( void)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( gv_pDbSystem);

	if( gv_ui32LockCount > 1)
	{
		rc = RC_SET( NE_XFLM_FAILURE);
	}
	else
	{
		// gv_ui32LockCount should be 1 because gv_pDbSystem is non-null.

		flmAssert( gv_ui32LockCount == 1);

		// Check for open databases

		f_mutexLock( gv_XFlmSysData.hShareMutex);

		if (gv_XFlmSysData.pDatabaseHashTbl)
		{
			FBUCKET *   pDatabaseHashTbl;
			FLMUINT		uiCnt;

			for (uiCnt = 0, pDatabaseHashTbl = gv_XFlmSysData.pDatabaseHashTbl;
				uiCnt < FILE_HASH_ENTRIES;
				uiCnt++, pDatabaseHashTbl++)
			{
				if (pDatabaseHashTbl->pFirstInBucket != NULL)
				{
					rc = RC_SET( NE_XFLM_FAILURE);
					break;
				}
			}
		}

		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
	}

	return( rc);
}

/******************************************************************************
Desc:	Returns the desired interface to the class object for
		the specified service class.
******************************************************************************/
XFLMEXTC RCODE XFLMAPI DllGetClassObject(
	RXFLMCLSID		rclsid,
	RXFLMIID			riid,
	void **			ppv)
{
	static F_DbSystemFactory	gv_DbSysFactory;

	if( f_memcmp( &rclsid, &Internal_CLSID_F_DbSystemFactory,
		sizeof( Internal_CLSID_F_DbSystemFactory)) == 0)
	{
		return( gv_DbSysFactory.QueryInterface( riid, ppv));
	}

	*ppv = NULL;
	return( RC_SET( NE_XFLM_CLASS_NOT_AVAILABLE));
}

/******************************************************************************
Desc:	Called by PSA when it loads the library.  Must return 0 for
		success, or a non-zero error code.
******************************************************************************/
XFLMEXTC RCODE XFLMAPI DllStart( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( (gv_pDbSystem = f_new F_DbSystem) == NULL)
	{
		rc = NE_XFLM_MEM;
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
XFLMEXTC RCODE XFLMAPI DllStop( void)
{
	if( gv_pDbSystem)
	{
		flmAssert( gv_ui32LockCount == 1);

		gv_pDbSystem->exit();
		gv_pDbSystem->Release();
		gv_pDbSystem = NULL;
	}

	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:
******************************************************************************/
XFLMEXTC RCODE XFLMAPI DllRegisterServer(
	const char *)
{
	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:
******************************************************************************/
XFLMEXTC RCODE XFLMAPI DllUnregisterServer( void) 
{
	return( NE_XFLM_OK);
}

/******************************************************************************
Desc:	This is an array of all the CLSID's that XFlaim implements.
******************************************************************************/
extern "C" const XFLMCLSID * XTCOM_Table[] =
{
	&Internal_CLSID_F_DbSystemFactory,
	0
};
