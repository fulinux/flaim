//-------------------------------------------------------------------------
// Desc:	Entry point for Netware utilities
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; version 2.1
// of the License.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Library Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, contact Novell, Inc.
//
// To contact Novell about this file by physical or electronic mail, 
// you may find current contact information at www.novell.com.
//
// $Id: $
//-------------------------------------------------------------------------

#include "ftksys.h"
#include "ftknlm.h"

#if defined( FLM_NLM)

#if defined( FLM_RING_ZERO_NLM)

static SEMAPHORE						gv_lFlmSyncSem = 0;
static FLMBOOL							gv_bUnloadCalled = FALSE;
static FLMBOOL							gv_bMainRunning = FALSE;
static F_EXIT_FUNC					gv_fnExit = NULL;

// Variables defined in ftknlm.cpp

extern FLMATOMIC						gv_NetWareStartupCount;
extern void *							gv_MyModuleHandle;
extern rtag_t							gv_lAllocRTag;

RCODE f_nssInitialize( void);

void f_nssUninitialize( void);

extern "C" int nlm_main(
	int					iArgC,
	char **				ppszArgV);

#endif

/********************************************************************
Desc: Startup routine for the NLM - that gets the main going in
		its own thread.
*********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void * f_nlmMainStub(
	void *		hThread,
	void *		pData)
{
	ARG_DATA *									pArgData = (ARG_DATA *)pData;
	struct LoadDefinitionStructure *		moduleHandle = pArgData->moduleHandle;

	(void)hThread;

	(void)kSetThreadName( (void *)kCurrentThread(),
		(BYTE *)pArgData->pszThreadName);

	nlm_main( pArgData->iArgC, pArgData->ppszArgV);

	Free( pArgData->ppszArgV);
	Free( pArgData->pszArgs);
	Free( pArgData->pszThreadName);
	Free( pArgData);

	gv_bMainRunning = FALSE;

	if( !gv_bUnloadCalled)
	{
		KillMe( moduleHandle);
	}
	
	kExitThread( NULL);
	return( NULL);
}
#endif
	
/********************************************************************
Desc: Signals the f_nlmEntryPoint thread to release the console.
*********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
void SynchronizeStart( void)
{
	if (gv_lFlmSyncSem)
	{
		(void)kSemaphoreSignal( gv_lFlmSyncSem);
	}
}
#endif

/********************************************************************
Desc: Startup routine for the NLM.
*********************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" LONG f_nlmEntryPoint(
	struct LoadDefinitionStructure *		moduleHandle,
	struct ScreenStruct *					initScreen,
	char *										commandLine,
	char *										loadDirectoryPath,
	LONG											uninitializedDataLength,
	LONG											fileHandle,
	LONG											(*ReadRoutine)
														(LONG		handle,
														 LONG		offset,
														 char *	buffer,
														 LONG		length),
	LONG											customDataOffset,
	LONG											customDataSize)
{
	char *		pszTmp;
	char *		pszArgStart;
	int			iArgC;
	int			iTotalArgChars;
	int			iArgSize;
	char **		ppszArgV = NULL;
	char *		pszArgs = NULL;
	char *		pszDestArg;
	bool			bFirstPass = true;
	char			cEnd;
	ARG_DATA *	pArgData = NULL;
	LONG			sdRet = 0;
	char *		pszThreadName;
	char *		pszModuleName;
	int			iModuleNameLen;
	int			iThreadNameLen;
	int			iLoadDirPathSize;
	void *		hThread = NULL;
	
	(void)initScreen;
	(void)uninitializedDataLength;
	(void)fileHandle;
	(void)ReadRoutine;
	(void)customDataOffset;
	(void)customDataSize;

	if( f_atomicInc( &gv_NetWareStartupCount) != 1)
	{
		goto Exit;
	}
	
	gv_MyModuleHandle = moduleHandle;
	gv_bUnloadCalled = FALSE;

	// Allocate the needed resource tags
	
	if( (gv_lAllocRTag = AllocateResourceTag( gv_MyModuleHandle,
		"FLAIM Memory", AllocSignature)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}

	// Syncronized start

	if (moduleHandle->LDFlags & 4)
	{
		gv_lFlmSyncSem = kSemaphoreAlloc( (BYTE *)"FLAIM_SYNC", 0);
	}

	// Initialize NSS
	
	if( RC_BAD( f_nssInitialize()))
	{
		sdRet = 1;
		goto Exit;
	}
	
	pszModuleName = (char *)(&moduleHandle->LDFileName[ 1]);
	iModuleNameLen = (int)(moduleHandle->LDFileName[ 0]);
	
	// First pass: Count the arguments in the command line
	// and determine how big of a buffer we will need.
	// Second pass: Put argments into allocated buffer.

Parse_Args:

	iTotalArgChars = 0;
	iArgC = 0;
	
	iLoadDirPathSize = f_strlen( (const char *)loadDirectoryPath); 
	iArgSize =  iLoadDirPathSize + iModuleNameLen;
	
	if( !bFirstPass)
	{
		ppszArgV[ iArgC] = pszDestArg;
		f_memcpy( pszDestArg, loadDirectoryPath, iLoadDirPathSize);
		f_memcpy( &pszDestArg[ iLoadDirPathSize], pszModuleName, iModuleNameLen);
		pszDestArg[ iArgSize] = 0;
		pszDestArg += (iArgSize + 1);
	}

	iArgC++;
	iTotalArgChars += iArgSize;
	pszTmp = commandLine;

	for (;;)
	{
		// Skip leading blanks.

		while( *pszTmp && *pszTmp == ' ')
		{
			pszTmp++;
		}

		if( *pszTmp == 0)
		{
			break;
		}

		if( *pszTmp == '"' || *pszTmp == '\'')
		{
			cEnd = *pszTmp;
			pszTmp++;
		}
		else
		{
			cEnd = ' ';
		}
		
		pszArgStart = pszTmp;
		iArgSize = 0;

		// Count the characters in the parameter.

		while( *pszTmp && *pszTmp != cEnd)
		{
			iArgSize++;
			pszTmp++;
		}

		if( !iArgSize && cEnd == ' ')
		{
			break;
		}

		// If 2nd pass, save the argument.

		if( !bFirstPass)
		{
			ppszArgV[ iArgC] = pszDestArg;
			
			if( iArgSize)
			{
				f_memcpy( pszDestArg, pszArgStart, iArgSize);
			}
			
			pszDestArg[ iArgSize] = 0;
			pszDestArg += (iArgSize + 1);
		}

		iArgC++;
		iTotalArgChars += iArgSize;

		// Skip trailing quote or blank.

		if( *pszTmp)
		{
			pszTmp++;
		}
	}

	if( bFirstPass)
	{
		if ((ppszArgV = (char **)Alloc(  sizeof( char *) * iArgC, 
			gv_lAllocRTag)) == NULL)
		{
			sdRet = 1;
			goto Exit;
		}

		if( (pszArgs = (char *)Alloc( iTotalArgChars + iArgC, 
			gv_lAllocRTag)) == NULL)
		{
			sdRet = 1;
			goto Exit;
		}
		
		pszDestArg = pszArgs;
		bFirstPass = false;
		goto Parse_Args;
	}

	iThreadNameLen = (int)(moduleHandle->LDName[ 0]);

	if( (pszThreadName = (char *)Alloc( iThreadNameLen + 1, gv_lAllocRTag)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}
	
	f_memcpy( pszThreadName, (char *)(&moduleHandle->LDName[ 1]), iThreadNameLen);
	pszThreadName[ iThreadNameLen] = 0;

	if( (pArgData = (ARG_DATA *)Alloc( sizeof( ARG_DATA), 
		gv_lAllocRTag)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}
	
	pArgData->ppszArgV = ppszArgV;
	pArgData->pszArgs = pszArgs;
	pArgData->iArgC = iArgC;
	pArgData->moduleHandle = moduleHandle;
	pArgData->pszThreadName = pszThreadName;

	gv_bMainRunning = TRUE;

	if( (hThread = kCreateThread( (BYTE *)"FTK main",
			f_nlmMainStub, NULL, 32768, (void *)pArgData)) == NULL)
	{
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}

	if( kSetThreadLoadHandle( hThread, (LONG)moduleHandle) != 0)
	{
		(void)kDestroyThread( hThread);
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}
			
	if( kScheduleThread( hThread) != 0)
	{
		(void)kDestroyThread( hThread);
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}
	
	// Synchronized start

	if( moduleHandle->LDFlags & 4)
	{
		(void)kSemaphoreWait( gv_lFlmSyncSem);
	}

Exit:

	if( sdRet != 0)
	{
		f_atomicDec( &gv_NetWareStartupCount);
		
		if( ppszArgV)
		{
			Free( ppszArgV);
		}

		if( pszArgs)
		{
			Free( pszArgs);
		}

		if( pszThreadName)
		{
			Free( pszThreadName);
		}

		if( pArgData)
		{
			Free( pArgData);
		}

		if( gv_lFlmSyncSem)
		{
			kSemaphoreFree( gv_lFlmSyncSem);
			gv_lFlmSyncSem = 0;
		}
		
		if( !gv_bUnloadCalled)
		{
			KillMe( moduleHandle);
		}
	}

	return( sdRet);
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void f_nlmExitPoint(void)
{
	if( f_atomicDec( &gv_NetWareStartupCount) > 0)
	{
		return;
	}
	
	gv_bUnloadCalled = TRUE;

	if( gv_fnExit)
	{
		(*gv_fnExit)();
		gv_fnExit = NULL;
	}

	while( gv_bMainRunning)
	{
		kYieldThread();
	}

	f_nssUninitialize();
	
	if( gv_lFlmSyncSem)
	{
		kSemaphoreFree( gv_lFlmSyncSem);
		gv_lFlmSyncSem = 0;
	}

	if( gv_lAllocRTag)
	{
		ReturnResourceTag( gv_lAllocRTag, 1);
		gv_lAllocRTag = 0;
	}
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" void exit(
	int		exitCode)
{
	(void)exitCode;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_RING_ZERO_NLM)
extern "C" int atexit(
	F_EXIT_FUNC		fnExit)
{
	gv_fnExit = fnExit;
	return( 0);
}
#endif
	
#endif // FLM_NLM

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_OSX)
void gv_fnlm2()
{
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if !defined( FLM_NLM)
int ftknlmDummy2( void)
{
	return( 0);
}
#endif
