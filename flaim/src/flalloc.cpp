//-------------------------------------------------------------------------
// Desc:	Memory allocation routines.
// Tabs:	3
//
//		Copyright (c) 1991,1993,1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flalloc.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#undef f_free
#undef f_alloc
#undef f_calloc
#undef f_realloc
#undef f_recalloc

#ifdef FLM_NLM
	extern "C"
	{
		extern LONG	gv_lAllocRTag;
	}
#endif

#ifdef FLM_UNIX
	#ifdef HAVE_CONFIG_H
		#include "config.h"
	#endif	

	#ifdef HAVE_DLADDR
		 #include <dlfcn.h>
	#endif
	#include <stdlib.h>

#endif

#define F_GET_ALLOC_PTR( pDataPtr) \
	(FLMBYTE *)((FLMBYTE *)(pDataPtr) - sizeof( F_MEM_HDR))

#define F_GET_DATA_PTR( pAllocPtr) \
	(FLMBYTE *)((FLMBYTE *)(pAllocPtr) + sizeof( F_MEM_HDR))

#define F_GET_MEM_DATA_SIZE( pDataPtr) \
	(((F_MEM_HDR *)(F_GET_ALLOC_PTR( pDataPtr)))->uiDataSize)

// Picket fence

#define F_PICKET_FENCE			"FFFFFFFF"

#if defined( FLM_DEBUG)
	#define F_PICKET_FENCE_SIZE	8
#else
	#define F_PICKET_FENCE_SIZE	0
#endif

#define MEM_PTR_INIT_ARRAY_SIZE	512
#define MEM_MAX_STACK_WALK_DEPTH	32

// If stack tracking is on, leak checking also
// needs to be on.

#ifndef FLM_DEBUG
	#ifdef DEBUG_SIM_OUT_OF_MEM
		#undef DEBUG_SIM_OUT_OF_MEM
	#endif
#endif

#ifdef FLM_DEBUG

// Local function prototypes

FSTATIC FLMBOOL initMemTracking(
	void);

FSTATIC void saveMemTrackingInfo(
	F_MEM_HDR *	pHdr);

FSTATIC void updateMemTrackingInfo(
	F_MEM_HDR *	pHdr);

FSTATIC void freeMemTrackingInfo(
	FLMBOOL		bMutexAlreadyLocked,
	FLMUINT		uiId,
	FLMUINT *	puiStack);

#ifdef DEBUG_SIM_OUT_OF_MEM

//one of every OUT_OF_MEM_FREQUENCY calls will fail

#define OUT_OF_MEM_FREQUENCY				40000

//OUT_OF_MEM_SEQUENCE_LENGTH calls in a row will fail 

#define OUT_OF_MEM_SEQUENCE_LENGTH		10

FLMBOOL SimulateOutOfMemory()
{
	if (
		//is the flag turned on
		(gv_FlmSysData.uiOutOfMemSimEnabledFlag ==
			(FLMUINT)OUT_OF_MEM_SIM_ENABLED_FLAG) &&
		
		//continuing a sequence of failures
		((gv_FlmSysData.uiSimOutOfMemFailSequence > 0) ||

		//failing randomly for the first time, and starting a new sequence 
		(f_randomChoice( &gv_FlmSysData.memSimRandomGen, 0, OUT_OF_MEM_FREQUENCY) == 0)))
	{
		gv_FlmSysData.uiSimOutOfMemFailTotal++;
		gv_FlmSysData.uiSimOutOfMemFailSequence++;
		//if reached the end of failure sequence, reset back to 0 so the
		//sequence will cease
		if ( gv_FlmSysData.uiSimOutOfMemFailSequence >= OUT_OF_MEM_SEQUENCE_LENGTH)
		{
			gv_FlmSysData.uiSimOutOfMemFailSequence = 0;
		}
		return TRUE;
	}
	else
	{
		return FALSE;
	}
}

#endif //#ifdef DEBUG_SIM_OUT_OF_MEM

#if defined( FLM_NLM)

	void * memGetEBP(void);

#ifdef __MWERKS__

	void * memGetEBP(void)
	{
		__asm
		{
			mov	eax,[ebp]
		}
	}

#else

	#pragma aux memGetEBP = "mov eax,ebp";

#endif

void * memValueAtStackOffset(
	void *		pos,
	int			offset);

#ifdef __MWERKS__

	void * memValueAtStackOffset( void *, int)
	{
		__asm
		{
			mov	eax,[ebp+0x8]
			mov	ebx,[ebp+0xC]
			mov	eax,ss:[eax+ebx]
		}
	}

#else

	#pragma aux memValueAtStackOffset = "mov eax,ss:[eax+ebx]" parm [eax] [ebx];

#endif

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT * memWalkStack( void)
{
	FLMUINT		uiLoop;
 	FLMUINT		uiRtnAddr;
	FLMUINT		uiEbp = (FLMUINT) memGetEBP();
	FLMUINT		uiAddresses [MEM_MAX_STACK_WALK_DEPTH + 1];
	FLMUINT *	puiAddresses;

	uiEbp = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 0);
	uiRtnAddr = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 4);

	for( uiLoop = 0; uiLoop < MEM_MAX_STACK_WALK_DEPTH; uiLoop++)
	{
		FLMUINT	uiOldEbp;
		
		uiAddresses [uiLoop] = uiRtnAddr;
		
		if( !uiEbp)
		{
			break;
		}
		
		uiOldEbp = uiEbp;
		uiEbp = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 0);

		if (!uiEbp || uiEbp <= uiOldEbp || uiEbp > uiOldEbp + 5000)
		{
			break;
		}

		uiRtnAddr = (FLMUINT) memValueAtStackOffset( (void *) uiEbp, 4);
	}
	
	uiAddresses[ uiLoop] = 0;
	
	if( (puiAddresses = (FLMUINT *)os_malloc( 
		sizeof( FLMUINT) * (uiLoop+1))) != NULL)
	{
		f_memcpy( puiAddresses, &uiAddresses [0], 
			sizeof( FLMUINT) * (uiLoop + 1));
	}
	
	return( puiAddresses);
}

#elif defined( FLM_WIN)

/********************************************************************
Desc: Reads NSIZE bytes of memory from LPBASEADDRESS
*********************************************************************/
static BOOL CALLBACK ReadProcMemory(
	HANDLE,
	DWORD64	lpBaseAddress,
	PVOID		lpBuffer, 
	DWORD		nSize,
	LPDWORD	lpNumberOfBytesRead)
{
	static HANDLE hRealProcess = GetCurrentProcess();
	SIZE_T bytesRead = 0;
	BOOL rv = ReadProcessMemory(hRealProcess, 
			(const void *)((FLMUINT)lpBaseAddress & 0xFFFFFFFF),
			lpBuffer, SIZE_T(nSize), &bytesRead);

	if (lpNumberOfBytesRead)
	{
		*lpNumberOfBytesRead = DWORD(bytesRead & 0xFFFFFFFF);
	}

	return( rv);
}

/********************************************************************
Desc: Walk the call stack.
*********************************************************************/
FLMUINT * memWalkStack()
{
	STACKFRAME64	stackFrame;
	DWORD				machineType;
	FLMUINT			uiLoop;
	FLMUINT			uiAddresses [MEM_MAX_STACK_WALK_DEPTH + 1];
	FLMUINT *		puiAddresses;
	HANDLE			hProcess = GetCurrentProcess();
	FLMUINT			uiAddrCount;
	const void *	pc;
	const void *	fp;
	
	__asm
	{
		call $ + 5
		pop eax
		mov [pc], eax
		mov [fp], ebp
	}
	
#ifdef FLM_64BIT
	machineType = IMAGE_FILE_MACHINE_IA64;
#else
	machineType = IMAGE_FILE_MACHINE_I386;
#endif

	f_memset( &stackFrame, 0, sizeof( stackFrame));

	stackFrame.AddrPC.Offset = DWORD((FLMUINT)pc);
	stackFrame.AddrPC.Mode = AddrModeFlat;
	stackFrame.AddrFrame.Offset = DWORD((FLMUINT)fp);
	stackFrame.AddrFrame.Mode = AddrModeFlat;

	f_mutexLock( gv_FlmSysData.hMemTrackingMutex);

	// We have already processed the address inside memWalkStack

	uiAddrCount = 1;
	uiLoop = 0;
	for (;;)
	{
		if( !StackWalk64( machineType, hProcess, 0, &stackFrame,
			0, ReadProcMemory, SymFunctionTableAccess64, SymGetModuleBase64, 0))
		{
			break;
		}

		if( !stackFrame.AddrFrame.Offset)
		{
			break;
		}

		// Skip the first two addresses.  These represent the following:
		// 1) memWalkStack
		// 2) saveMemTrackingInfo or updateMemTrackingInfo
		// We don't need to see them in the stack trace.

		uiAddrCount++;
		if (uiAddrCount > 2)
		{
			uiAddresses [uiLoop] = (FLMUINT)stackFrame.AddrReturn.Offset;
			uiLoop++;
			if (uiLoop == MEM_MAX_STACK_WALK_DEPTH)
			{
				break;
			}
		}
	}

	f_mutexUnlock( gv_FlmSysData.hMemTrackingMutex);

	uiAddresses [uiLoop] = 0;
	if ((puiAddresses = (FLMUINT *)os_malloc( 
		sizeof( FLMUINT) * (uiLoop+1))) != NULL)
	{
		f_memcpy( puiAddresses, &uiAddresses [0], sizeof( FLMUINT) * (uiLoop + 1));
	}
	return( puiAddresses);
}
#else
FLMUINT * memWalkStack()
{
	return( NULL);
}
#endif

/********************************************************************
Desc: Initialize memory tracking
*********************************************************************/
FSTATIC FLMBOOL initMemTracking( void)
{
	RCODE		rc;
	F_MUTEX	memMutex;

	if (!gv_FlmSysData.bMemTrackingInitialized && !gv_FlmSysData.uiInitThreadId)
	{
		gv_FlmSysData.uiInitThreadId = f_threadId();
		rc = f_mutexCreate( &memMutex);
		f_sleep( 50);

		// Only set to initialized if we were the last thread
		// to set gv_FlmSysData.uiInitThreadId

		if (f_threadId() == gv_FlmSysData.uiInitThreadId)
		{
			if (RC_OK( rc))
			{
				gv_FlmSysData.hMemTrackingMutex = memMutex;
			}
			else
			{
				gv_FlmSysData.hMemTrackingMutex = F_MUTEX_NULL;
			}
#ifdef FLM_WIN
			SymSetOptions( SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
			gv_FlmSysData.hMemProcess = GetCurrentProcess();
			SymInitialize( gv_FlmSysData.hMemProcess, NULL, TRUE);
#endif
			gv_FlmSysData.bMemTrackingInitialized = TRUE;
		}
		else
		{
			if (RC_OK( rc))
			{
				f_mutexDestroy( &memMutex);
			}
		}
	}

	// Go into a loop until we see initialized flag set to TRUE
	// Could be another thread that is doing it.

	while (!gv_FlmSysData.bMemTrackingInitialized)
	{
		f_sleep( 10);
	}
	return( (gv_FlmSysData.hMemTrackingMutex != F_MUTEX_NULL) ? TRUE : FALSE);
}

/********************************************************************
Desc: Save memory tracking information - called on alloc or realloc.
*********************************************************************/
FSTATIC void saveMemTrackingInfo(
	F_MEM_HDR *	pHdr)
{
	FLMUINT	uiNewCnt;
	FLMUINT	uiId;
	void **	pNew;

	if (gv_FlmSysData.bTrackLeaks && initMemTracking())
	{
		f_mutexLock( gv_FlmSysData.hMemTrackingMutex);

		// See if there is enough room in the array

		if (gv_FlmSysData.uiMemNumPtrs == gv_FlmSysData.uiMemTrackingPtrArraySize)
		{

			// If array is not initialized, use initial count.  Otherwise
			// double the size.

			uiNewCnt = (FLMUINT)((!gv_FlmSysData.uiMemTrackingPtrArraySize)
										? MEM_PTR_INIT_ARRAY_SIZE
										: gv_FlmSysData.uiMemTrackingPtrArraySize * 2);
			if ((pNew = (void **)os_malloc( sizeof( void *) * uiNewCnt)) != NULL)
			{

				// Copy the pointers from the old array, if any,
				// into the newly allocated array.

				if (gv_FlmSysData.uiMemTrackingPtrArraySize)
				{
					f_memcpy( pNew, gv_FlmSysData.ppvMemTrackingPtrs,
							sizeof( void *) * gv_FlmSysData.uiMemTrackingPtrArraySize);
					os_free( gv_FlmSysData.ppvMemTrackingPtrs);
				}
				f_memset( &pNew [gv_FlmSysData.uiMemTrackingPtrArraySize], 0,
						sizeof( void *) * (uiNewCnt - gv_FlmSysData.uiMemTrackingPtrArraySize));
				gv_FlmSysData.ppvMemTrackingPtrs = pNew;
				gv_FlmSysData.uiMemTrackingPtrArraySize = uiNewCnt;
			}
		}

		// If we are still full, we were not able to reallocate memory, so we
		// do nothing.

		if (gv_FlmSysData.uiMemNumPtrs == gv_FlmSysData.uiMemTrackingPtrArraySize)
		{
			pHdr->uiAllocationId = 0;
		}
		else
		{
			// Find an empty slot - there has to be one!

			uiId = gv_FlmSysData.uiMemNextPtrSlotToUse;
			while (gv_FlmSysData.ppvMemTrackingPtrs [uiId])
			{
				if (++uiId == gv_FlmSysData.uiMemTrackingPtrArraySize)
				{
					uiId = 0;
				}
			}

			// Allocation ID in the header is offset by one to avoid
			// using a value of zero.

			pHdr->uiAllocationId = uiId + 1;
			gv_FlmSysData.ppvMemTrackingPtrs [uiId] = pHdr;
			gv_FlmSysData.uiMemNumPtrs++;
			if ((gv_FlmSysData.uiMemNextPtrSlotToUse = uiId + 1) ==
					gv_FlmSysData.uiMemTrackingPtrArraySize)
			{
				gv_FlmSysData.uiMemNextPtrSlotToUse = 0;
			}
		}
		pHdr->uiAllocCnt = ++gv_FlmSysData.uiAllocCnt;
		f_mutexUnlock( gv_FlmSysData.hMemTrackingMutex);
	}
	else
	{
		pHdr->uiAllocationId = 0;
		pHdr->uiAllocCnt = 0;
	}

	// Follow the stack.

	if (gv_FlmSysData.bTrackLeaks && gv_FlmSysData.bStackWalk)
	{
		pHdr->puiStack = memWalkStack();
	}
	else
	{
		pHdr->puiStack = NULL;
	}
}

/********************************************************************
Desc: Update memory tracking information - called after realloc
*********************************************************************/
FSTATIC void updateMemTrackingInfo(
	F_MEM_HDR *	pHdr)
{
	if (pHdr->puiStack)
	{
		os_free( pHdr->puiStack);
		pHdr->puiStack = NULL;
	}
	if (gv_FlmSysData.bTrackLeaks && gv_FlmSysData.bStackWalk)
	{
		pHdr->puiStack = memWalkStack();
	}
}

/********************************************************************
Desc: Free memory tracking information - called on free.
*********************************************************************/
FSTATIC void freeMemTrackingInfo(
	FLMBOOL		bMutexAlreadyLocked,
	FLMUINT		uiId,
	FLMUINT *	puiStack
	)
{
	if (uiId)
	{
		// NOTE: If uiId is non-zero, it means we had to have
		// successfully initialized, so we are guaranteed to
		// have a mutex.

		if ( !bMutexAlreadyLocked)
		{
			f_mutexLock( gv_FlmSysData.hMemTrackingMutex);
		}

		// Allocation ID in the header is offset by one so that it
		// is never zero - a value of zero means that the allocation
		// does not have a slot for tracking it in the array.

		gv_FlmSysData.ppvMemTrackingPtrs [uiId - 1] = NULL;
		flmAssert( gv_FlmSysData.uiMemNumPtrs);
		gv_FlmSysData.uiMemNumPtrs--;

		if ( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( gv_FlmSysData.hMemTrackingMutex);
		}
	}

	// Free the stack information, if any.

	if (puiStack)
	{
		os_free( puiStack);
	}
}

/********************************************************************
Desc: Log memory leaks.
*********************************************************************/
void logMemLeak(
	F_MEM_HDR *		pHdr)
{
	char			szMessageBuffer [1024];
	char *		pszTmp = &szMessageBuffer [0];
	F_FileHdl *	pFileHdl = NULL;
	FLMBOOL		bOldTrackLeaks = gv_FlmSysData.bTrackLeaks;

	gv_FlmSysData.bTrackLeaks = FALSE;	// This ensures that any future
													// allocations (for instance,
													// allocating the file handle for the
													// memtest.ert file) will not try to
													// lock the mem tracking mutex.



	// Format message to be logged.

	f_strcpy( pszTmp, "Abort=Debug, Retry=Continue, Ignore=Don't Show\r\n");
	while (*pszTmp)
	{
		pszTmp++;
	}
#if defined( FLM_64BIT)
	f_sprintf(	pszTmp, "Unfreed Pointer: 0x%016I64x\r\n", (FLMUINT)(&pHdr [1]));
#else
	f_sprintf(	pszTmp, "Unfreed Pointer: 0x%08x\r\n",
		(unsigned)((FLMUINT)(&pHdr [1])));
#endif
	while (*pszTmp)
	{
		pszTmp++;
	}

	if (pHdr->pszFileName)
	{
		f_sprintf( pszTmp, "Source: %s, Line#: %u\r\n", pHdr->pszFileName,
								(unsigned)pHdr->iLineNumber);
		while (*pszTmp)
		{
			pszTmp++;
		}
	}

	if (pHdr->uiAllocCnt)
	{
		f_sprintf( pszTmp, "Malloc #: %u\r\n", (unsigned)pHdr->uiAllocCnt);
		while (*pszTmp)
		{
			pszTmp++;
		}
  	}
	f_sprintf( pszTmp, "Size: %u bytes\r\n", (unsigned)pHdr->uiDataSize);
	while (*pszTmp)
	{
		pszTmp++;
	}

	if (pHdr->puiStack)
	{
		FLMUINT *			puiStack = pHdr->puiStack;
		FLMUINT				uiLen = pszTmp - szMessageBuffer;
		char					szFuncName [200];
		char *				pszFuncName;
#ifdef FLM_WIN
		IMAGEHLP_SYMBOL *	pImgHlpSymbol;

		pImgHlpSymbol = (IMAGEHLP_SYMBOL *)os_malloc(
									sizeof( IMAGEHLP_SYMBOL) + 100);
#endif

		while (*puiStack)
		{
			szFuncName [0] = 0;
#if defined( FLM_WIN)
			if (pImgHlpSymbol)
			{
#ifdef FLM_64BIT
				DWORD64		udDisplacement;
#else
				DWORD			udDisplacement;
#endif

				pImgHlpSymbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
				pImgHlpSymbol->Address = *puiStack;
				pImgHlpSymbol->MaxNameLength = 100;

				if (SymGetSymFromAddr( gv_FlmSysData.hMemProcess, *puiStack,
												&udDisplacement, pImgHlpSymbol))
				{
					f_sprintf( szFuncName, "\t%s + %X\r\n",
									(&pImgHlpSymbol->Name [0]),
									udDisplacement);
				}
			}
#elif defined( FLM_NLM)
			{
				szFuncName [0] = '\t';
				GetClosestSymbol( (BYTE *)(&szFuncName[1]), (LONG)(*puiStack));
			}
#else

#ifdef HAVE_DLADDR
			{
				Dl_info	dlip;

				if (dladdr( (void *)(*puiStack), &dlip) != 0 && dlip.dli_sname)
				{
					const char *	pszFileName;
					if (dlip.dli_saddr != (void *)(*puiStack))
					{
						pszFileName = strrchr(dlip.dli_fname, '/');
						if (!pszFileName)
						{
							pszFileName = dlip.dli_fname;
						}
						else
						{
							pszFileName++;		// skip over slash
						}
						f_sprintf( szFuncName, "\t0x%08x (%s)\r\n",
									(unsigned)(*puiStack), pszFileName); 
					}
					else
					{
						f_sprintf( szFuncName, "\t%s\r\n", dlip.dli_sname);
					}
				}
			}
#endif

#endif

			// If szFuncName [0] is zero, we didn't find a name, so we
			// just output the address in HEX.

			if (!szFuncName [0])
			{
				f_sprintf( szFuncName, "\t0x%08X\r\n", (unsigned)*puiStack );
			}

			// Output whatever portion of the name will fit into the
			// message buffer.

			pszFuncName = &szFuncName [0];
			while (*pszFuncName && uiLen < sizeof( szMessageBuffer) - 1)
			{
				*pszTmp++ = *pszFuncName++;
				uiLen++;
			}

			// Process next address in the stack.

			puiStack++;
		}
		*pszTmp = 0;
#ifdef FLM_WIN
		if (pImgHlpSymbol)
		{
			os_free( pImgHlpSymbol);
		}
#endif
	}

#ifdef FLM_WIN
	FLMINT	iRet;

	iRet =  MessageBox( NULL, (LPCTSTR)szMessageBuffer, "WIN Memory Testing",
					MB_ABORTRETRYIGNORE | MB_ICONINFORMATION | MB_TASKMODAL
					| MB_SETFOREGROUND | MB_DEFBUTTON2);
	if (iRet == IDIGNORE)
	{
		gv_FlmSysData.bLogLeaks = TRUE;
	}
	else if (iRet == IDABORT)
	{
		flmAssert( 0);
	}
#else
	gv_FlmSysData.bLogLeaks = TRUE;
#endif

	if (gv_FlmSysData.bLogLeaks)
	{
		F_FileSystemImp	FileSystem;
		RCODE					rc;
		FLMUINT				uiDummy;
#ifdef FLM_NLM
		const char *	pszErrPath = "sys:\\memtest.ert";
#else
		const char *	pszErrPath = "memtest.ert";
#endif

		if (RC_BAD( rc = FileSystem.Open( pszErrPath,
								F_IO_RDWR | F_IO_SH_DENYNONE, &pFileHdl)))
		{
			if (rc == FERR_IO_PATH_NOT_FOUND)
			{
				rc = FileSystem.Create( pszErrPath,
									F_IO_RDWR | F_IO_SH_DENYNONE, &pFileHdl);
			}
		}
		else
		{
			FLMUINT	uiOffset;

			// Position to append to file.

			rc = pFileHdl->Seek( 0, F_IO_SEEK_END, &uiOffset);
		}

		// If we successfully opened the file, write to it.

		if (RC_OK( rc))
		{
			if (RC_OK( pFileHdl->Write( F_IO_CURRENT_POS,
						(FLMUINT)(pszTmp - &szMessageBuffer [0]),
						szMessageBuffer, &uiDummy)))
			{
				(void)pFileHdl->Flush();
			}
			pFileHdl->Close();
		}
	}
//Exit:

	gv_FlmSysData.bTrackLeaks = bOldTrackLeaks;

	if (pFileHdl)
	{
		pFileHdl->Release();
	}
}
#endif

/********************************************************************
Desc: Initialize memory - if not already done.
*********************************************************************/
void f_memoryInit( void)
{
#ifdef FLM_DEBUG
	(void)initMemTracking();
#endif
}

/********************************************************************
Desc: Clean up memory and check for unfreed memory.
*********************************************************************/
void f_memoryCleanup( void)
{

#ifdef FLM_DEBUG
	if (initMemTracking())
	{
		FLMUINT		uiId;
		F_MEM_HDR *	pHdr;

		f_mutexLock( gv_FlmSysData.hMemTrackingMutex);
		for (uiId = 0; uiId < gv_FlmSysData.uiMemTrackingPtrArraySize; uiId++)
		{
			if ((pHdr = (F_MEM_HDR *)gv_FlmSysData.ppvMemTrackingPtrs [uiId]) != NULL)
			{
				logMemLeak( pHdr);
				freeMemTrackingInfo( TRUE, uiId + 1, pHdr->puiStack);
			}
		}

		// Free the memory pointer array.

		os_free( gv_FlmSysData.ppvMemTrackingPtrs);
		gv_FlmSysData.ppvMemTrackingPtrs = NULL;
		gv_FlmSysData.uiMemTrackingPtrArraySize = 0;
		gv_FlmSysData.uiMemNumPtrs = 0;

		f_mutexUnlock( gv_FlmSysData.hMemTrackingMutex);

		// Free up the mutex.

		f_mutexDestroy( &gv_FlmSysData.hMemTrackingMutex);

		// Reset to unitialized state.

		gv_FlmSysData.uiInitThreadId = 0;
		gv_FlmSysData.hMemTrackingMutex = F_MUTEX_NULL;
		gv_FlmSysData.bMemTrackingInitialized = FALSE;
#ifdef FLM_WIN
		SymCleanup( gv_FlmSysData.hMemProcess);
#endif
	}
#endif
}

/********************************************************************
Desc: Allocate Memory.
*********************************************************************/
RCODE f_alloc(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = FERR_OK;
	F_MEM_HDR *	pHdr;

#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( iLineNumber);
#endif

#ifdef DEBUG_SIM_OUT_OF_MEM
	if ( SimulateOutOfMemory())
	{
		*ppvPtr = NULL;
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
#endif	

	if ((pHdr = (F_MEM_HDR *)os_malloc( uiSize +
												sizeof( F_MEM_HDR) +
												F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pHdr [1]);

#ifdef FLM_DEBUG
	pHdr->iLineNumber = iLineNumber;
	pHdr->pszFileName = pszFileName;
	saveMemTrackingInfo( pHdr);

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif
#endif

Exit:

	return( rc);
}

/********************************************************************
Desc: Allocate and initialize memory.
*********************************************************************/
RCODE f_calloc(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = FERR_OK;
	F_MEM_HDR *	pHdr;

#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( pszFileName);
	F_UNREFERENCED_PARM( iLineNumber);
#endif

#ifdef DEBUG_SIM_OUT_OF_MEM
	if ( SimulateOutOfMemory())
	{
		*ppvPtr = NULL;
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
#endif	

	if ((pHdr = (F_MEM_HDR *)os_malloc( uiSize +
											sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pHdr [1]);
	f_memset( *ppvPtr, 0, uiSize);
#ifdef FLM_DEBUG
	pHdr->iLineNumber = iLineNumber;
	pHdr->pszFileName = pszFileName;
	saveMemTrackingInfo( pHdr);

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif

#endif
Exit:
	return( rc);
}

/********************************************************************
Desc: Reallocate memory.
*********************************************************************/
RCODE f_realloc(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = FERR_OK;
	F_MEM_HDR *	pNewHdr;
#ifdef FLM_DEBUG
	F_MEM_HDR *	pOldHdr;
	FLMUINT		uiOldAllocationId;
	FLMUINT *	puiOldStack;
#endif

#ifdef DEBUG_SIM_OUT_OF_MEM
	if ( SimulateOutOfMemory())
	{
		*ppvPtr = NULL;
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
#endif		

	if (!(*ppvPtr))
	{
		rc = f_alloc( uiSize, ppvPtr, pszFileName, iLineNumber);
		goto Exit;
	}

#ifdef FLM_DEBUG
	pOldHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

	#if F_PICKET_FENCE_SIZE

	// Verify the old picket fence

	if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pOldHdr->uiDataSize,
						F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
	{
		flmAssert( 0);
	}

	#endif

	uiOldAllocationId = pOldHdr->uiAllocationId;
	puiOldStack = pOldHdr->puiStack;
#endif

	if ((pNewHdr = (F_MEM_HDR *)os_realloc( F_GET_ALLOC_PTR( *ppvPtr),
											uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pNewHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pNewHdr [1]);
#ifdef FLM_DEBUG
	pNewHdr->iLineNumber = iLineNumber;
	pNewHdr->pszFileName = pszFileName;
	if (pNewHdr != pOldHdr)
	{
		freeMemTrackingInfo( FALSE, uiOldAllocationId, puiOldStack);
		saveMemTrackingInfo( pNewHdr);
	}
	else
	{
		updateMemTrackingInfo( pNewHdr);
	}

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif

#endif

Exit:
	return( rc);
}

/********************************************************************
Desc: Reallocate memory, and initialize the new part.
*********************************************************************/
RCODE f_recalloc(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	int				iLineNumber)
{
	RCODE			rc = FERR_OK;
	F_MEM_HDR *	pNewHdr;
	FLMUINT		uiOldSize;
#ifdef FLM_DEBUG
	F_MEM_HDR *	pOldHdr;
	FLMUINT		uiOldAllocationId;
	FLMUINT *	puiOldStack;
#endif

#ifdef DEBUG_SIM_OUT_OF_MEM
	if ( SimulateOutOfMemory())
	{
		*ppvPtr = NULL;
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
#endif		

	if (!(*ppvPtr))
	{
		rc = f_calloc( uiSize, ppvPtr, pszFileName, iLineNumber);
		goto Exit;
	}

#ifdef FLM_DEBUG
	pOldHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

	#if F_PICKET_FENCE_SIZE

	// Verify the old picket fence

	if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pOldHdr->uiDataSize,
						F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
	{
		flmAssert( 0);
	}

	#endif

	uiOldAllocationId = pOldHdr->uiAllocationId;
	puiOldStack = pOldHdr->puiStack;

#endif

	uiOldSize = F_GET_MEM_DATA_SIZE( *ppvPtr);

	if ((pNewHdr = (F_MEM_HDR *)os_realloc( F_GET_ALLOC_PTR( *ppvPtr),
											uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pNewHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pNewHdr [1]);
	if (uiOldSize < uiSize)
	{
		f_memset( ((FLMBYTE *)(*ppvPtr)) + uiOldSize, 0,
					 uiSize - uiOldSize);
	}
#ifdef FLM_DEBUG
	pNewHdr->iLineNumber = iLineNumber;
	pNewHdr->pszFileName = pszFileName;
	if (pNewHdr != pOldHdr)
	{
		freeMemTrackingInfo( FALSE, uiOldAllocationId, puiOldStack);
		saveMemTrackingInfo( pNewHdr);
	}
	else
	{
		updateMemTrackingInfo( pNewHdr);
	}

	#if F_PICKET_FENCE_SIZE

	f_memcpy( ((FLMBYTE *)(*ppvPtr)) + uiSize,
				F_PICKET_FENCE, F_PICKET_FENCE_SIZE);

	#endif

#endif
Exit:
	return( rc);
}

/********************************************************************
Desc: Free previously allocated memory.
*********************************************************************/
void f_free(
	void **	ppvPtr)
{
	if (*ppvPtr)
	{

#ifdef FLM_DEBUG
		F_MEM_HDR *	pHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

		#if F_PICKET_FENCE_SIZE

		// Check the picket fence

		if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pHdr->uiDataSize,
							F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
		{
			flmAssert( 0);
		}

		#endif

		freeMemTrackingInfo( FALSE, pHdr->uiAllocationId, pHdr->puiStack);
#endif

		os_free( F_GET_ALLOC_PTR( *ppvPtr));
		*ppvPtr = NULL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_NLM
void * nlm_realloc(
	void	*	pMemory,
	size_t	newSize)
{
	void *		pNewMemory;
	LONG			lSize;

 	if( !pMemory)
 	{
 		pNewMemory = Alloc( newSize, gv_lAllocRTag);
		goto Exit;
	}

	lSize = SizeOfAllocBlock( pMemory);

	pNewMemory = os_malloc( newSize);
	if( !pNewMemory)
	{
		goto Exit;
	}

	if( lSize > newSize)
	{
		lSize = newSize;
	}
	
	f_memcpy( pNewMemory, pMemory, lSize);

	if( pMemory)
	{
		Free( pMemory);
	}

Exit:

	return( pNewMemory);
}
#endif

#undef new
#undef delete
#define FLM_NEW_MEMORY_SIGNATURE		0xABCDABCD

/****************************************************************************
Desc:	
****************************************************************************/
void * F_Base::operator new(
	FLMSIZET			uiSize) 
#ifndef FLM_NLM	
	throw()
#endif
{
	void *	pvReturnPtr = NULL;

	uiSize += FLM_ALIGN_SIZE;
	f_alloc( uiSize, &pvReturnPtr, "unknown", 0);

	if( pvReturnPtr)
	{
		*((FLMUINT *)pvReturnPtr) = FLM_NEW_MEMORY_SIGNATURE;
		pvReturnPtr = (void *)(((FLMBYTE *)pvReturnPtr) + FLM_ALIGN_SIZE);
	}

	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void * F_Base::operator new[](
	FLMSIZET			uiSize)
#ifndef FLM_NLM	
	throw()
#endif
{
	void *	pvReturnPtr = NULL;

	uiSize += FLM_ALIGN_SIZE;
	f_alloc( uiSize, &pvReturnPtr, "unknown", 0);

	if( pvReturnPtr)
	{
		*((FLMUINT *)pvReturnPtr) = FLM_NEW_MEMORY_SIGNATURE;
		pvReturnPtr = (void *)(((FLMBYTE *)pvReturnPtr) + FLM_ALIGN_SIZE);
	}

	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
#ifdef FLM_DEBUG
void * F_Base::operator new(
	FLMSIZET			uiSize,
	const char *	pszFile,
	int				iLine)
#ifndef FLM_NLM	
	throw()
#endif
{
	void *	pvReturnPtr = NULL;

	uiSize += FLM_ALIGN_SIZE;
	f_alloc( uiSize, &pvReturnPtr, pszFile, iLine);

	if( pvReturnPtr)
	{
		*((FLMUINT *)pvReturnPtr) = FLM_NEW_MEMORY_SIGNATURE;
		pvReturnPtr = (void *)(((FLMBYTE *)pvReturnPtr) + FLM_ALIGN_SIZE);
	}

	return( pvReturnPtr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#ifdef FLM_DEBUG
void * F_Base::operator new[](
	FLMSIZET			uiSize,
	const char *	pszFile,
	int				iLine)
#ifndef FLM_NLM	
	throw()
#endif
{
	void *	pvReturnPtr = NULL;

	uiSize += FLM_ALIGN_SIZE;
	f_alloc( uiSize, &pvReturnPtr, pszFile, iLine);

	if( pvReturnPtr)
	{
		*((FLMUINT *)pvReturnPtr) = FLM_NEW_MEMORY_SIGNATURE;
		pvReturnPtr = (void *)(((FLMBYTE *)pvReturnPtr) + FLM_ALIGN_SIZE);
	}

	return( pvReturnPtr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
void F_Base::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	ptr = (void *)(((FLMBYTE *)ptr) - FLM_ALIGN_SIZE);
	if( *((FLMUINT *)ptr) != FLM_NEW_MEMORY_SIGNATURE)
	{
		// Something is wrong with the allocation ... don't
		// try to free it.

		flmAssert( 0);
		return;
	}

	f_free( &ptr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_Base::operator delete[](
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}

	ptr = (void *)(((FLMBYTE *)ptr) - FLM_ALIGN_SIZE);
	if( *((FLMUINT *)ptr) != FLM_NEW_MEMORY_SIGNATURE)
	{
		// Something is wrong with the allocation ... don't
		// try to free it.

		flmAssert( 0);
		return;
	}

	f_free( &ptr);
}

/****************************************************************************
Desc:	
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( __WATCOMC__)
void F_Base::operator delete(
	void *				ptr,
	const char *,		// file
	int)					// line
{
	if( !ptr)
	{
		return;
	}

	ptr = (void *)(((FLMBYTE *)ptr) - FLM_ALIGN_SIZE);
	if( *((FLMUINT *)ptr) != FLM_NEW_MEMORY_SIGNATURE)
	{
		// Something is wrong with the allocation ... don't
		// try to free it.

		flmAssert( 0);
		return;
	}

	f_free( &ptr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( __WATCOMC__)
void F_Base::operator delete[](
	void *			ptr,
	const char *,	// file
	int				// line
	)
{
	if( !ptr)
	{
		return;
	}

	ptr = (void *)(((FLMBYTE *)ptr) - FLM_ALIGN_SIZE);
	if( *((FLMUINT *)ptr) != FLM_NEW_MEMORY_SIGNATURE)
	{
		// Something is wrong with the allocation ... don't
		// try to free it.

		flmAssert( 0);
		return;
	}

	f_free( &ptr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
FLMINT F_Base::Release( void)
{
	FLMINT		iRefCnt = --m_i32RefCnt;

	if( !iRefCnt)
	{
		delete this;
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:	
****************************************************************************/
FLMEXP void FLMAPI FlmFreeMem(
	void *		pMem)
{
	f_free( &pMem);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT f_msize(
	void *			pvPtr)
{
	#if defined( FLM_UNIX)
		return( pvPtr ? F_GET_MEM_DATA_SIZE( (pvPtr)) : 0);
	#elif defined( FLM_NLM)
		return( pvPtr ? (unsigned)SizeOfAllocBlock(
			(F_GET_ALLOC_PTR( (pvPtr)))) : 0);
	#else
		return( pvPtr ? _msize( (F_GET_ALLOC_PTR( (pvPtr)))) : 0);
	#endif
}
