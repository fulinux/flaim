//-------------------------------------------------------------------------
// Desc:	Startup/Exit module for FLAIM utilities on Netware.
// Tabs:	3
//
//		Copyright (c) 1999-2000,2002-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: nlmload.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaim.h"
#include "ftk.h"

#ifdef FLM_WATCOM_NLM

extern "C" void  * nlmStart(
	void *			hThread,
	void *			pData);

extern "C" void exit(
	int				exitCode);

extern "C" int nlm_main(
	int				iArgC,
	char **			ppszArgV);

extern "C" LONG FlmLoad(
	LoadDefStruct *				moduleHandle,
	struct ScreenStruct *		initScreen,
	char *							commandLine,
	char *							loadDirectoryPath,
	LONG								uninitializedDataLength,
	LONG								fileHandle,
	LONG								(*ReadRoutine)
											(LONG		handle,
											 LONG		offset,
											 char *	buffer,
											 LONG		length),
	LONG								customDataOffset,
	LONG								customDataSize);

extern "C" void FlmUnload( void);

static SEMAPHORE					gv_lFlmSyncSem = 0;
static FLMBOOL						gv_bUnloadCalled = FALSE;
static FLMBOOL						gv_bMainRunning = FALSE;
static LONG							gv_lAllocRTag = 0;
static LONG							gv_lMyModuleHandle = 0;
static FLM_EXIT_FUNC				gv_fnExit;

extern "C" void __wcpp_4_fatal_runtime_error_(
	char *		msg,
	unsigned		retcode)
{
	(void)msg;
	(void)retcode;
}

/********************************************************************
Desc: Signals the FlmLoad thread to release the console.
*********************************************************************/
void SynchronizeStart( void)
{
	if (gv_lFlmSyncSem)
	{
		(void)kSemaphoreSignal( gv_lFlmSyncSem);
	}
}

/********************************************************************
Desc: Startup routine for the NLM - that gets the main going in
		its own thread.
*********************************************************************/
void  * nlmStart(
	void *		hThread,
	void *		pData)
{
	ARG_DATA *				pArgData = (ARG_DATA *)pData;
	LoadDefStruct *		moduleHandle = pArgData->moduleHandle;

	(void)hThread;

	(void)kSetThreadName( (void *)kCurrentThread(),
		(BYTE *)pArgData->pszThreadName);

	(void)nlm_main( pArgData->iArgC, pArgData->ppszArgV);

	Free( pArgData->ppszArgV);
	Free( pArgData->pszArgs);
	Free( pArgData->pszThreadName);
	Free( pArgData);

	gv_bMainRunning = FALSE;
	
	if (!gv_bUnloadCalled)
	{
		KillMe( moduleHandle);
	}
	
	kExitThread( NULL);
	return( NULL);
}

/********************************************************************
Desc: Startup routine for the NLM.
*********************************************************************/
LONG FlmLoad(
	LoadDefStruct *			moduleHandle,
	struct ScreenStruct *	initScreen,
	char *						commandLine,
	char *						loadDirectoryPath,
	LONG							uninitializedDataLength,
	LONG							fileHandle,
	LONG							(*ReadRoutine)
										(LONG		handle,
										 LONG		offset,
										 char *	buffer,
										 LONG		length),
	LONG							customDataOffset,
	LONG							customDataSize)
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
	int			iTmpLen;
	void *		hThread = NULL;

	gv_bUnloadCalled = FALSE;

	(void)initScreen;
	(void)uninitializedDataLength;
	(void)fileHandle;
	(void)ReadRoutine;
	(void)customDataOffset;
	(void)customDataSize;

	// Allocate the needed resource tags

	gv_lMyModuleHandle = (LONG)moduleHandle;

	if( (gv_lAllocRTag = AllocateResourceTag(
						(LONG)moduleHandle,
						(BYTE *)"FLAIM Memory", AllocSignature)) == NULL)
	{
			sdRet = 1;
		goto Exit;
	}

	if (moduleHandle->LDFlags & 4)	// Synchronized start
	{
		gv_lFlmSyncSem = kSemaphoreAlloc( (BYTE *)"FLAIM", 0);
	}

	// First pass: Count the arguments in the command line
	// and determine how big of a buffer we will need.
	// Second pass: Put argments into allocated buffer.

Parse_Args:

	iTotalArgChars = 0;
	iArgC = 0;

	iArgSize = f_strlen( loadDirectoryPath);
	
	if (!bFirstPass)
	{
		ppszArgV [iArgC] = pszDestArg;
		f_memcpy( pszDestArg, loadDirectoryPath, iArgSize);
		pszDestArg [iArgSize] = 0;
		pszDestArg += (iArgSize + 1);
	}

	iArgC++;
	iTotalArgChars += iArgSize;
	pszTmp = commandLine;
	
	for (;;)
	{

		// Skip leading blanks.

		while ((*pszTmp) && (*pszTmp == ' '))
		{
			pszTmp++;
		}

		if (!(*pszTmp))
		{
			break;
		}

		if ((*pszTmp == '"') || (*pszTmp == '\''))
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

		while ((*pszTmp) && (*pszTmp != cEnd))
		{
			iArgSize++;
			pszTmp++;
		}

		if ((!iArgSize) && (cEnd == ' '))
		{
			break;
		}

		// If 2nd pass, save the argument.

		if (!bFirstPass)
		{
			ppszArgV [iArgC] = pszDestArg;
			if (iArgSize)
			{
				f_memcpy( pszDestArg, pszArgStart, iArgSize);
			}
			pszDestArg [iArgSize] = 0;
			pszDestArg += (iArgSize + 1);
		}

		iArgC++;
		iTotalArgChars += iArgSize;

		// Skip trailing quote or blank.

		if (*pszTmp)
		{
			pszTmp++;
		}
	}

	if (bFirstPass)
	{
			if ((ppszArgV = (char **)Alloc(  
				sizeof( char *) * iArgC, gv_lAllocRTag)) == NULL)
		{
			sdRet = 1;
			goto Exit;
		}

			if ((pszArgs = (char *)Alloc( 
				iTotalArgChars + iArgC, gv_lAllocRTag)) == NULL)
		{
			sdRet = 1;
			goto Exit;
		}
		
		pszDestArg = pszArgs;
		bFirstPass = false;
		goto Parse_Args;
	}

	pszTmp = (char *)(&moduleHandle->LDName [1]);
	iTmpLen = (int)(moduleHandle->LDName [0]);
	
	if ((pszThreadName = (char *)Alloc( 
		iTmpLen + 1, gv_lAllocRTag)) == NULL)
	{
		sdRet = 1;
		goto Exit;
	}
	f_memcpy( pszThreadName, pszTmp, iTmpLen);
	pszThreadName [iTmpLen] = 0;

	if ((pArgData = (ARG_DATA *)Alloc( 
		sizeof( ARG_DATA), gv_lAllocRTag)) == NULL)
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

	if ((hThread = kCreateThread( (BYTE *)"FLAIM", nlmStart, NULL, 32768,
		(void *)pArgData)) == NULL)
	{
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}

	if (kSetThreadLoadHandle( hThread, (LONG)moduleHandle) != 0)
	{
		(void)kDestroyThread( hThread);
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}
			
	if (kScheduleThread( hThread) != 0)
	{
		(void)kDestroyThread( hThread);
		gv_bMainRunning = FALSE;
		sdRet = 2;
		goto Exit;
	}

	if (moduleHandle->LDFlags & 4)
	{
		// Synchronized start

		(void)kSemaphoreWait( gv_lFlmSyncSem);
	}

Exit:

	if (sdRet != 0)
	{
		if (ppszArgV)
		{
				Free( ppszArgV);
		}

		if (pszArgs)
		{
				Free( pszArgs);
		}

		if (pszThreadName)
		{
				Free( pszThreadName);
		}

		if (pArgData)
		{
				Free( pArgData);
		}

		if (gv_lFlmSyncSem)
		{
			kSemaphoreFree( gv_lFlmSyncSem);
			gv_lFlmSyncSem = 0;
		}
		if (!gv_bUnloadCalled)
		{
			KillMe( moduleHandle);
		}
	}
	
	return( sdRet);
}

/****************************************************************************
Desc:
****************************************************************************/
void FlmUnload(void)
{
	gv_bUnloadCalled = TRUE;

	if( gv_fnExit)
	{
		(*gv_fnExit)();
		gv_fnExit = NULL;
	}

	while (gv_bMainRunning)
	{
		kYieldThread();
	}

	if (gv_lFlmSyncSem)
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

/****************************************************************************
Desc:
****************************************************************************/
void exit(
	int		exitCode)
{
	(void)exitCode;
}

/****************************************************************************
Desc:
****************************************************************************/
int atexit(
	FLM_EXIT_FUNC	fnExit)
{
	gv_fnExit = fnExit;
	return( 0);
}

#endif // FLM_WATCOM_NLM
