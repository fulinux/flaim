//------------------------------------------------------------------------------
// Desc:	This file contains the f_alloc, f_calloc, f_realloc, f_recalloc,
//			and f_free routines.
//
// Tabs:	3
//
//		Copyright (c) 1991, 1993, 1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flalloc.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_WIN

	// This pragma is needed because FLAIM may be built with a
	// packing other than 8-bytes for Windows (such as 1-byte packing).
	// Code in FLAIM that uses windows structures and system calls
	// MUST use 8-byte packing (the packing used by the O/S).
	// See Microsoft technical article Q117388.

	#pragma pack( push, enter_windows, 8)
		#include <imagehlp.h>
	#pragma pack( pop, enter_windows)
#endif

#ifdef FLM_UNIX
	#ifdef HAVE_DLADDR
		 #include <dlfcn.h>
	#endif
#endif

#if defined( FLM_UNIX) || defined( FLM_NLM) || defined( FLM_WIN)
	#define PTR_IN_MBLK(p,bp,offs)	(((FLMBYTE *)(p) > (FLMBYTE *)(bp)) && \
												 ((FLMBYTE *)(p) <= (FLMBYTE *)(bp) + (offs)))
#else
	#error Platform not supported
#endif

#ifdef USE_ALT_MEM_MANAGER

	// If USE_ALT_MEM_MANAGER is defined, memory allocations will be based on
	// a user-supplied module that implements f_alt_malloc, f_alt_free, and
	// f_alt_realloc.

	// Prototypes

	extern "C"
	{
		void * f_alt_malloc( unsigned size);
		void f_alt_free( void * memblock);
		void * f_alt_realloc( void * memblock, unsigned size);
	}

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

// Platform specific stuff needed for tracking leaks

#ifdef FLM_NLM
	extern "C"
	{
		void GetClosestSymbol(
			BYTE *	szBuffer,
			LONG		udAddress);
	}
#endif

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
		(gv_XFlmSysData.uiOutOfMemSimEnabledFlag ==
			(FLMUINT)OUT_OF_MEM_SIM_ENABLED_FLAG) &&
		
		//continuing a sequence of failures
		((gv_XFlmSysData.uiSimOutOfMemFailSequence > 0) ||

		//failing randomly for the first time, and starting a new sequence 
		(f_randomChoice( &gv_XFlmSysData.memSimRandomGen, 0, OUT_OF_MEM_FREQUENCY) == 0)
		 ))
	{
		gv_XFlmSysData.uiSimOutOfMemFailTotal++;
		gv_XFlmSysData.uiSimOutOfMemFailSequence++;
		//if reached the end of failure sequence, reset back to 0 so the
		//sequence will cease
		if ( gv_XFlmSysData.uiSimOutOfMemFailSequence >= OUT_OF_MEM_SEQUENCE_LENGTH)
		{
			gv_XFlmSysData.uiSimOutOfMemFailSequence = 0;
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

/************************************************************************
Desc:	Returns the current value of EBP--the value of the caller's stack 
		frame pointer.
*************************************************************************/
void * memGetEBP(void);

#ifdef FLM_MWERKS_NLM

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

/************************************************************************
Desc:	Returns the value at SS:[POS+OFFSET].
*************************************************************************/
void * memValueAtStackOffset(void *pos, int offset);

#ifdef FLM_MWERKS_NLM

	void *memValueAtStackOffset(void *, int)
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

/************************************************************************
Desc:
*************************************************************************/
FLMUINT * memWalkStack()
{
	FLMUINT		uiLoop;
 	FLMUINT		uiRtnAddr;
	FLMUINT		uiEbp = (FLMUINT) memGetEBP();
	FLMUINT		uiAddresses [MEM_MAX_STACK_WALK_DEPTH + 1];
	FLMUINT *	puiAddresses;

	uiEbp = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 0); // skip addrs we already know
	uiRtnAddr = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 4); // Caller's return addr

	for (uiLoop = 0; uiLoop < MEM_MAX_STACK_WALK_DEPTH; uiLoop++)
	{
		FLMUINT	uiOldEbp;
		
		uiAddresses [uiLoop] = uiRtnAddr;
		if (!uiEbp)
		{
			break;
		}
		
		uiOldEbp = uiEbp;
		uiEbp = (FLMUINT) memValueAtStackOffset( (void *)uiEbp, 0);		// Caller's frame ptr

		if (!uiEbp || uiEbp <= uiOldEbp || uiEbp > uiOldEbp + 5000)
		{
			break;
		}

		uiRtnAddr = (FLMUINT) memValueAtStackOffset( (void *) uiEbp, 4);			// Caller's return addr
	}
	uiAddresses [uiLoop] = 0;
	if ((puiAddresses = (FLMUINT *)os_malloc( 
		sizeof( FLMUINT) * (uiLoop+1))) != NULL)
	{
		f_memcpy( puiAddresses, &uiAddresses [0], sizeof( FLMUINT) * (uiLoop + 1));
	}
	return( puiAddresses);
}

#elif defined( FLM_WIN)

/********************************************************************
Desc: Walk the call stack.
*********************************************************************/
FLMUINT * memWalkStack()
{
	STACKFRAME64	stackFrame;
	CONTEXT			context;
	DWORD				machineType;
	FLMUINT			uiLoop;
	FLMUINT			uiAddresses [MEM_MAX_STACK_WALK_DEPTH + 1];
	FLMUINT *		puiAddresses;
	HANDLE			hThread;
	HANDLE			hProcess;
	FLMUINT			uiAddrCount;

	f_memset( &stackFrame, 0, sizeof( stackFrame));
	f_memset( &context, 0, sizeof( context));

#ifdef FLM_64BIT
	machineType = IMAGE_FILE_MACHINE_IA64;
#else
	machineType = IMAGE_FILE_MACHINE_I386;
#endif

	// While you can continue walking the stack...

	unsigned vEBP, vEIP;
	__asm mov vEBP, ebp
	__asm call near nextinstr
nextinstr:
	__asm pop vEIP;

	context.Ebp = vEBP;
	context.Eip = vEIP;
	stackFrame.AddrPC.Offset = vEIP;
	stackFrame.AddrFrame.Offset = vEBP;
	stackFrame.AddrPC.Mode = AddrModeFlat;
	stackFrame.AddrFrame.Mode = AddrModeFlat;

	// Must lock the mutex because StackWalk is not thread safe.

	f_mutexLock( gv_XFlmSysData.hMemTrackingMutex);
	hProcess = OpenProcess( PROCESS_VM_READ, FALSE, GetCurrentProcessId());
	hThread = OpenThread( THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME, 
			FALSE, GetCurrentThreadId());

	// We have already processed the address inside memWalkStack

	uiAddrCount = 1;
	uiLoop = 0;
	for (;;)
	{
		if (!StackWalk64( machineType, hProcess, hThread, &stackFrame,
							&context, NULL,
							SymFunctionTableAccess64, SymGetModuleBase64, NULL))
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

	f_mutexUnlock( gv_XFlmSysData.hMemTrackingMutex);

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

	if (!gv_XFlmSysData.bMemTrackingInitialized && !gv_XFlmSysData.uiInitThreadId)
	{
		gv_XFlmSysData.uiInitThreadId = f_threadId();
		rc = f_mutexCreate( &memMutex);
		f_sleep( 50);

		// Only set to initialized if we were the last thread
		// to set gv_XFlmSysData.uiInitThreadId

		if (f_threadId() == gv_XFlmSysData.uiInitThreadId)
		{
			if (RC_OK( rc))
			{
				gv_XFlmSysData.hMemTrackingMutex = memMutex;
			}
			else
			{
				gv_XFlmSysData.hMemTrackingMutex = F_MUTEX_NULL;
			}
#ifdef FLM_WIN
			SymSetOptions( SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
			gv_XFlmSysData.hMemProcess = GetCurrentProcess();
			SymInitialize( gv_XFlmSysData.hMemProcess, NULL, TRUE);
#endif
			gv_XFlmSysData.bMemTrackingInitialized = TRUE;
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

	while (!gv_XFlmSysData.bMemTrackingInitialized)
	{
		f_sleep( 10);
	}
	return( (gv_XFlmSysData.hMemTrackingMutex != F_MUTEX_NULL) ? TRUE : FALSE);
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

	if (gv_XFlmSysData.bTrackLeaks && initMemTracking())
	{
		f_mutexLock( gv_XFlmSysData.hMemTrackingMutex);

		// See if there is enough room in the array

		if (gv_XFlmSysData.uiMemNumPtrs == gv_XFlmSysData.uiMemTrackingPtrArraySize)
		{

			// If array is not initialized, use initial count.  Otherwise
			// double the size.

			uiNewCnt = (FLMUINT)((!gv_XFlmSysData.uiMemTrackingPtrArraySize)
										? MEM_PTR_INIT_ARRAY_SIZE
										: gv_XFlmSysData.uiMemTrackingPtrArraySize * 2);
			if ((pNew = (void **)os_malloc( sizeof( void *) * uiNewCnt)) != NULL)
			{

				// Copy the pointers from the old array, if any,
				// into the newly allocated array.

				if (gv_XFlmSysData.uiMemTrackingPtrArraySize)
				{
					f_memcpy( pNew, gv_XFlmSysData.ppvMemTrackingPtrs,
							sizeof( void *) * gv_XFlmSysData.uiMemTrackingPtrArraySize);
					os_free( gv_XFlmSysData.ppvMemTrackingPtrs);
					gv_XFlmSysData.ppvMemTrackingPtrs = NULL;
				}
				f_memset( &pNew [gv_XFlmSysData.uiMemTrackingPtrArraySize], 0,
						sizeof( void *) * (uiNewCnt - gv_XFlmSysData.uiMemTrackingPtrArraySize));
				gv_XFlmSysData.ppvMemTrackingPtrs = pNew;
				gv_XFlmSysData.uiMemTrackingPtrArraySize = uiNewCnt;
			}
		}

		// If we are still full, we were not able to reallocate memory, so we
		// do nothing.

		if (gv_XFlmSysData.uiMemNumPtrs == gv_XFlmSysData.uiMemTrackingPtrArraySize)
		{
			pHdr->uiAllocationId = 0;
		}
		else
		{
			// Find an empty slot - there has to be one!

			uiId = gv_XFlmSysData.uiMemNextPtrSlotToUse;
			while (gv_XFlmSysData.ppvMemTrackingPtrs [uiId])
			{
				if (++uiId == gv_XFlmSysData.uiMemTrackingPtrArraySize)
				{
					uiId = 0;
				}
			}

			// Allocation ID in the header is offset by one to avoid
			// using a value of zero.

			pHdr->uiAllocationId = uiId + 1;
			gv_XFlmSysData.ppvMemTrackingPtrs [uiId] = pHdr;
			gv_XFlmSysData.uiMemNumPtrs++;
			if ((gv_XFlmSysData.uiMemNextPtrSlotToUse = uiId + 1) ==
					gv_XFlmSysData.uiMemTrackingPtrArraySize)
			{
				gv_XFlmSysData.uiMemNextPtrSlotToUse = 0;
			}
		}
		pHdr->uiAllocCnt = ++gv_XFlmSysData.uiAllocCnt;
		f_mutexUnlock( gv_XFlmSysData.hMemTrackingMutex);
	}
	else
	{
		pHdr->uiAllocationId = 0;
		pHdr->uiAllocCnt = 0;
	}

	// Follow the stack.

	if (gv_XFlmSysData.bTrackLeaks && gv_XFlmSysData.bStackWalk)
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
	if (gv_XFlmSysData.bTrackLeaks && gv_XFlmSysData.bStackWalk)
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
			f_mutexLock( gv_XFlmSysData.hMemTrackingMutex);
		}

		// Allocation ID in the header is offset by one so that it
		// is never zero - a value of zero means that the allocation
		// does not have a slot for tracking it in the array.

		gv_XFlmSysData.ppvMemTrackingPtrs [uiId - 1] = NULL;
		flmAssert( gv_XFlmSysData.uiMemNumPtrs);
		gv_XFlmSysData.uiMemNumPtrs--;

		if ( !bMutexAlreadyLocked)
		{
			f_mutexUnlock( gv_XFlmSysData.hMemTrackingMutex);
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
	char				szTmpBuffer [1024];
	FLMUINT			uiMsgBufSize;
	char *			pszMessageBuffer;
	char *			pszTmp;
	IF_FileHdl *	pFileHdl = NULL;
	F_FileSystem	fileSys;
	FLMBOOL			bClearFileSys = FALSE;
	FLMBOOL			bSaveTrackLeaks = gv_XFlmSysData.bTrackLeaks;

	gv_XFlmSysData.bTrackLeaks = FALSE;

	// Need a big buffer to show an entire stack.

	uiMsgBufSize = 20480;
	if ((pszMessageBuffer = (char *)os_malloc( uiMsgBufSize)) == NULL)
	{
		pszMessageBuffer = &szTmpBuffer [0];
		uiMsgBufSize = sizeof( szTmpBuffer);
	}
	pszTmp = pszMessageBuffer;

	if( !gv_pFileSystem)
	{
		gv_pFileSystem = &fileSys;
		bClearFileSys = TRUE;
	}

	// Format message to be logged.

	f_strcpy( pszTmp, "Abort=Debug, Retry=Continue, Ignore=Don't Show\r\n");
	while (*pszTmp)
	{
		pszTmp++;
	}
#if defined( FLM_WIN) && defined( FLM_64BIT)
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
	f_sprintf( (char *)pszTmp, "Size: %u bytes\r\n", (unsigned)pHdr->uiDataSize);
	while (*pszTmp)
	{
		pszTmp++;
	}

	if (pHdr->puiStack)
	{
		FLMUINT *			puiStack = pHdr->puiStack;
		FLMUINT				uiLen = pszTmp - pszMessageBuffer;
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
#ifdef FLM_WIN
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

				if (SymGetSymFromAddr( gv_XFlmSysData.hMemProcess, *puiStack,
												&udDisplacement, pImgHlpSymbol))
				{
					f_sprintf( szFuncName, "\t%s + %X\r\n",
									(char *)(&pImgHlpSymbol->Name [0]),
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
			while (*pszFuncName && uiLen < uiMsgBufSize - 1)
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

	iRet =  MessageBox( NULL, (LPCTSTR)pszMessageBuffer, "WIN32 Memory Testing",
					MB_ABORTRETRYIGNORE | MB_ICONINFORMATION | MB_TASKMODAL 
					| MB_SETFOREGROUND | MB_DEFBUTTON2);
	if (iRet == IDIGNORE)
	{
		gv_XFlmSysData.bLogLeaks = TRUE;
	}
	else if (iRet == IDABORT)
	{
		flmAssert( 0);
	}
#else
	gv_XFlmSysData.bLogLeaks = TRUE;
#endif

	if (gv_XFlmSysData.bLogLeaks)
	{
		F_FileSystem	FileSystem;
		RCODE				rc;
		FLMUINT			uiDummy;
#ifdef FLM_NLM
		const char *	pszErrPath = "sys:\\memtest.ert";
#else
		const char *	pszErrPath = "memtest.ert";
#endif

	if (RC_BAD( rc = FileSystem.Open( pszErrPath, XFLM_IO_RDWR | XFLM_IO_SH_DENYNONE,
								&pFileHdl)))
		{
			if (rc == NE_XFLM_IO_PATH_NOT_FOUND)
			{
				rc = FileSystem.Create( pszErrPath, XFLM_IO_RDWR | XFLM_IO_SH_DENYNONE,
									&pFileHdl);
			}
		}
		else
		{
			// Position to append to file.

			rc = pFileHdl->Seek( 0, XFLM_IO_SEEK_END, NULL);
		}

		// If we successfully opened the file, write to it.

		if (RC_OK( rc))
		{
			if (RC_OK( pFileHdl->Write( XFLM_IO_CURRENT_POS,
						(FLMUINT)(pszTmp - pszMessageBuffer),
						pszMessageBuffer, &uiDummy)))
			{
				(void)pFileHdl->Flush();
			}
			pFileHdl->Close();
		}
	}
//Exit:
	if (pFileHdl)
	{
		pFileHdl->Release();
	}

	if( bClearFileSys)
	{
		gv_pFileSystem = NULL;
	}
	if (pszMessageBuffer != &szTmpBuffer [0])
	{
		os_free( pszMessageBuffer);
	}

	gv_XFlmSysData.bTrackLeaks = bSaveTrackLeaks;
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

		f_mutexLock( gv_XFlmSysData.hMemTrackingMutex);
		for (uiId = 0; uiId < gv_XFlmSysData.uiMemTrackingPtrArraySize; uiId++)
		{
			if ((pHdr = (F_MEM_HDR *)gv_XFlmSysData.ppvMemTrackingPtrs [uiId]) != NULL)
			{
				logMemLeak( pHdr);
				freeMemTrackingInfo( TRUE, uiId + 1, pHdr->puiStack);
			}
		}

		// Free the memory pointer array.

		os_free( gv_XFlmSysData.ppvMemTrackingPtrs);
		gv_XFlmSysData.ppvMemTrackingPtrs = NULL;
		gv_XFlmSysData.uiMemTrackingPtrArraySize = 0;
		gv_XFlmSysData.uiMemNumPtrs = 0;

		f_mutexUnlock( gv_XFlmSysData.hMemTrackingMutex);

		// Free up the mutex.

		f_mutexDestroy( &gv_XFlmSysData.hMemTrackingMutex);

		// Reset to unitialized state.

		gv_XFlmSysData.uiInitThreadId = 0;
		gv_XFlmSysData.hMemTrackingMutex = F_MUTEX_NULL;
		gv_XFlmSysData.bMemTrackingInitialized = FALSE;
#ifdef FLM_WIN
		SymCleanup( gv_XFlmSysData.hMemProcess);
#endif
	}
#endif
}

/********************************************************************
Desc: Allocate Memory.
*********************************************************************/
#ifdef FLM_DEBUG
RCODE f_allocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	FLMBOOL			bAllocFromNewOp,
	const char *	pszFileName,
	FLMINT			iLineNumber
	)
#else
RCODE f_allocImp(
	FLMUINT			uiSize,
	void **			ppvPtr
	)
#endif
{
	RCODE			rc = NE_XFLM_OK;
	F_MEM_HDR *	pHdr;

#ifdef DEBUG_SIM_OUT_OF_MEM
	if ( SimulateOutOfMemory())
	{
		*ppvPtr = NULL;
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
#endif	

	if ((pHdr = (F_MEM_HDR *)os_malloc( uiSize +
												sizeof( F_MEM_HDR) +
												F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	pHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pHdr [1]);
#ifdef FLM_DEBUG
	pHdr->bAllocFromNewOp = bAllocFromNewOp;
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
#ifdef FLM_DEBUG
RCODE f_callocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber
	)
#else
RCODE f_callocImp(
	FLMUINT			uiSize,
	void **			ppvPtr
	)
#endif
{
	RCODE			rc = NE_XFLM_OK;
	F_MEM_HDR *	pHdr;

#ifdef DEBUG_SIM_OUT_OF_MEM
	if ( SimulateOutOfMemory())
	{
		*ppvPtr = NULL;
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
#endif	

	if ((pHdr = (F_MEM_HDR *)os_malloc( uiSize +
											sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	pHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pHdr [1]);
	f_memset( *ppvPtr, 0, uiSize);
#ifdef FLM_DEBUG
	pHdr->bAllocFromNewOp = FALSE;
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
#ifdef FLM_DEBUG
RCODE f_reallocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber
	)
#else
RCODE f_reallocImp(
	FLMUINT			uiSize,
	void **			ppvPtr
	)
#endif
{
	RCODE			rc = NE_XFLM_OK;
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
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
#endif		

	if (!(*ppvPtr))
	{
#ifdef FLM_DEBUG
		rc = f_allocImp( uiSize, ppvPtr, FALSE, pszFileName, iLineNumber);
#else
		rc = f_allocImp( uiSize, ppvPtr);
#endif
		goto Exit;
	}

#ifdef FLM_DEBUG
	pOldHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

	#if F_PICKET_FENCE_SIZE

		// Verify the old picket fence
	
		if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pOldHdr->uiDataSize,
							F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_MEM);
			goto Exit;
		}

	#endif

	// Cannot realloc memory that was allocated via a new operator

	if (pOldHdr->bAllocFromNewOp)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_MEM);
		goto Exit;
	}

	uiOldAllocationId = pOldHdr->uiAllocationId;
	puiOldStack = pOldHdr->puiStack;
#endif

	if ((pNewHdr = (F_MEM_HDR *)os_realloc( F_GET_ALLOC_PTR( *ppvPtr),
											uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	pNewHdr->uiDataSize = uiSize;
	*ppvPtr = (void *)(&pNewHdr [1]);
#ifdef FLM_DEBUG
	pNewHdr->bAllocFromNewOp = FALSE;
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
#ifdef FLM_DEBUG
RCODE f_recallocImp(
	FLMUINT			uiSize,
	void **			ppvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber
	)
#else
RCODE f_recallocImp(
	FLMUINT			uiSize,
	void **			ppvPtr
	)
#endif
{
	RCODE			rc = NE_XFLM_OK;
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
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
#endif		

	if (!(*ppvPtr))
	{
#ifdef FLM_DEBUG
		rc = f_callocImp( uiSize, ppvPtr, pszFileName, iLineNumber);
#else
		rc = f_callocImp( uiSize, ppvPtr);
#endif
		goto Exit;
	}

#ifdef FLM_DEBUG
	pOldHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

	#if F_PICKET_FENCE_SIZE

		// Verify the old picket fence
	
		if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pOldHdr->uiDataSize,
							F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_MEM);
			goto Exit;
		}

	#endif

	// Cannot realloc memory that was allocated via a new operator

	if (pOldHdr->bAllocFromNewOp)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_MEM);
		goto Exit;
	}

	uiOldAllocationId = pOldHdr->uiAllocationId;
	puiOldStack = pOldHdr->puiStack;

#endif

	uiOldSize = F_GET_MEM_DATA_SIZE( *ppvPtr);
	if ((pNewHdr = (F_MEM_HDR *)os_realloc( F_GET_ALLOC_PTR( *ppvPtr),
											uiSize + sizeof( F_MEM_HDR) +
											F_PICKET_FENCE_SIZE)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
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
	pNewHdr->bAllocFromNewOp = FALSE;
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
void f_freeImp(
	void **	ppvPtr,
	FLMBOOL	bFreeFromDeleteOp)
{
#ifndef FLM_DEBUG
	F_UNREFERENCED_PARM( bFreeFromDeleteOp);
#endif

	if (*ppvPtr)
	{
#ifdef FLM_DEBUG

		F_MEM_HDR *	pHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( *ppvPtr);

		if (pHdr->bAllocFromNewOp && !bFreeFromDeleteOp ||
			 !pHdr->bAllocFromNewOp && bFreeFromDeleteOp)
		{

			// Either trying to free memory using f_free when
			// allocated from new, or trying to free memory
			// using delete when allocated from f_alloc,
			// f_calloc, f_realloc, or f_recalloc.

			RC_UNEXPECTED_ASSERT( NE_XFLM_MEM);
			return;
		}

		#if F_PICKET_FENCE_SIZE

			// Check the picket fence
	
			if (f_memcmp( ((FLMBYTE *)(*ppvPtr)) + pHdr->uiDataSize,
								F_PICKET_FENCE, F_PICKET_FENCE_SIZE) != 0)
			{
				RC_UNEXPECTED_ASSERT( NE_XFLM_MEM);
			}

		#endif

		freeMemTrackingInfo( FALSE, pHdr->uiAllocationId, pHdr->puiStack);
#endif
		os_free( F_GET_ALLOC_PTR( *ppvPtr));
		*ppvPtr = NULL;
	}
}

#ifdef FLM_DEBUG
/********************************************************************
Desc: Reset the stack information for an allocation.
*********************************************************************/
void f_resetStackInfoImp(
	void *			pvPtr,
	const char *	pszFileName,
	FLMINT			iLineNumber
	)
{
	if (pvPtr)
	{

		F_MEM_HDR *	pHdr = (F_MEM_HDR *)F_GET_ALLOC_PTR( pvPtr);

		pHdr->iLineNumber = iLineNumber;
		pHdr->pszFileName = pszFileName;
		f_mutexLock( gv_XFlmSysData.hMemTrackingMutex);
		pHdr->uiAllocCnt = ++gv_XFlmSysData.uiAllocCnt;
		f_mutexUnlock( gv_XFlmSysData.hMemTrackingMutex);
		updateMemTrackingInfo( pHdr);
	}
}
#endif


/************************************************************************
Desc:	Destructor
*************************************************************************/
F_Pool::~F_Pool()
{
	poolFree();
}

/************************************************************************
Desc:	Initialize a smart pool memory structure. A smart pool is one that
		will adjust it's block allocation size based on statistics it
		gathers within the POOL_STATS structure. For each pool that user
		wants to use smart memory management a global POOL_STATS structure
		should be declared. The POOL_STATS structure is used to track the
		total bytes allocated and determine what the correct pool block
		size should be.
*************************************************************************/
void F_Pool::smartPoolInit(
	POOL_STATS *	pPoolStats)
{
	m_pPoolStats = pPoolStats;
	if (m_pPoolStats && m_pPoolStats->uiCount)
	{
		setInitialSmartPoolBlkSize();
	}
	else
	{
		m_uiBlockSize = 2048;
	}
}

/****************************************************************************
Desc:	Allocates a block of memory from a memory pool.
Note:	If the number of bytes is more than the what is left in the
		current block then a new block will be allocated and the lbkl element
		of the PMS will be updated.
****************************************************************************/
RCODE F_Pool::poolAlloc(
	FLMUINT	uiSize,
	void **	ppvPtr
	)
{
	RCODE			rc = NE_XFLM_OK;
	MBLK *		pBlock = m_pLastBlock;
	MBLK *		pOldLastBlock = pBlock;
	FLMBYTE *	pucFreePtr;
	FLMUINT		uiBlockSize;

	// Adjust the size to a machine word boundary
	// NOTE: ORed and ANDed 0x800.. & 0x7FFF to prevent partial
	// stalls on Netware

	if (uiSize & (FLM_ALLOC_ALIGN | 0x80000000))
	{
		uiSize = ((uiSize + FLM_ALLOC_ALIGN) & (~(FLM_ALLOC_ALIGN) & 0x7FFFFFFF));
	}

	// Check if room in block

	if (!pBlock || uiSize > pBlock->uiFreeSize)
	{

		// Check if previous block has space for allocation

		if (pBlock &&
			 pBlock->pPrevBlock &&
			 uiSize <= pBlock->pPrevBlock->uiFreeSize)
		{
			pBlock = pBlock->pPrevBlock;
			goto Exit;
		}

		// Not enough memory in block - allocate new block

		// Determine the block size:
		// 1) start with max of last block size, initial pool size, or alloc size
		// 2) if this is an extra block alloc then increase the size by 1/2
		// 3) adjust size to include block header

		uiBlockSize = (pBlock) ? pBlock->uiBlockSize : m_uiBlockSize;
		uiBlockSize = f_max( uiSize, uiBlockSize);

		if (pBlock &&
			 uiBlockSize == pBlock->uiBlockSize &&
			 uiBlockSize <= 32769)
		{
			uiBlockSize += uiBlockSize / 2;
		}

		// Add in extra bytes for block overhead

		uiBlockSize += sizeof( MBLK);

		if (RC_BAD( rc = f_alloc( uiBlockSize, &pBlock)))
		{
			goto Exit;
		}

		// Initialize the block elements

		pBlock->uiBlockSize = uiBlockSize;
		pBlock->uiFreeOffset = sizeof( MBLK);
		pBlock->uiFreeSize = uiBlockSize - sizeof( MBLK);

		// Link in newly allocated block

		m_pLastBlock = pBlock;
		pBlock->pPrevBlock = pOldLastBlock;
	}

Exit:

	if (RC_OK( rc))
	{
		pucFreePtr = (FLMBYTE *)pBlock;
		pucFreePtr += pBlock->uiFreeOffset;		// Point to free space
		pBlock->uiFreeOffset += uiSize;			// Modify free offset
		pBlock->uiFreeSize -= uiSize;				// Modify free size

		m_uiBytesAllocated += uiSize;
		*ppvPtr = (void *)pucFreePtr;
	}
	else
	{
		*ppvPtr = NULL;
	}
	return( rc);
}

/****************************************************************************
Desc:	Allocates a block of memory from a memory pool.
****************************************************************************/
RCODE F_Pool::poolCalloc(
  	FLMUINT		uiSize,
	void **		ppvPtr)
{
	RCODE	rc;

	if (RC_OK( rc = poolAlloc( uiSize, ppvPtr)))
	{
		f_memset( *ppvPtr, 0, uiSize);
	}
	return( rc);
}

/****************************************************************************
Desc : Releases all memory allocated to a pool.
Note : All memory allocated to the pool is returned to the operating system.
*****************************************************************************/
void F_Pool::poolFree( void)
{
	MBLK *	pBlock = m_pLastBlock;
	MBLK *	pPrevBlock;

	// Free all blocks in chain

	while (pBlock)
	{
		pPrevBlock = pBlock->pPrevBlock;
		f_free( &pBlock);
		pBlock = pPrevBlock;
	}

	m_pLastBlock = NULL;

	// For Smart Pools update pool statictics

	if (m_pPoolStats)
	{
		updateSmartPoolStats();
	}
}

/****************************************************************************
Desc:		Resets memory blocks allocated to a pool.
Note:		Will reset the free space in the first memory block, and if
			any extra blocks exist they will be freed (destroyed).
*****************************************************************************/
void F_Pool::poolReset(
	void *		pvMark,
			// [IN] If pvMark is NULL, the first pool block is emptied and all
         // other blocks in the list are released to the operating system.
         // If pvMark is non-NULL, the pool list is searched for the block
         // containing pvMark.  All pool blocks following pvMark are
         // released and the block containing pvMark is is reset to
         // the byte referenced by pvMark.
	FLMBOOL		bReduceFirstBlock
   )
{
	MBLK *	pBlock = m_pLastBlock;
	MBLK *	pPrevBlock;

	if (!pBlock)
	{
		return;
	}

	// For Smart Pools update pool statictics

	if (m_pPoolStats)
	{
		updateSmartPoolStats();
	}

	if (pvMark)
	{
		freeToMark( pvMark);
		return;
	}

	// Free all blocks except last one in chain -- which is really
	// the first block allocated.  This will help us keep memory from
	// getting fragmented.

	while (pBlock->pPrevBlock)
	{
		pPrevBlock = pBlock->pPrevBlock;
		f_free( &pBlock);
		pBlock = pPrevBlock;
	}

	if (pBlock->uiBlockSize - sizeof(MBLK) > m_uiBlockSize && bReduceFirstBlock)
	{

		// The first block was not the default size, so free it

		f_free( &pBlock);
		m_pLastBlock = NULL;
	}
	else
	{

		// Reset the allocation pointers in the first block

		pBlock->uiFreeOffset  = sizeof( MBLK);
		pBlock->uiFreeSize = pBlock->uiBlockSize - sizeof( MBLK);
		m_pLastBlock = pBlock;

#ifdef FLM_MEM_CHK
		{
		/*	memset the reset memory so someone pointing to it will get a error. */
		FLMBYTE *	pucPtr = (FLMBYTE *) pBlock;
		pucPtr += pBlock->uiFreeOffset;
		f_memset( pucPtr, 'r', pBlock->uiFreeSize);	// Set memory to 'r' for Reset
		}
#endif
	}

	// on smart pools adjust the initial block size on pool resets

	if (m_pPoolStats)
	{
		setInitialSmartPoolBlkSize();
	}
}

/****************************************************************************
Desc:	Frees memory until the pvMark is found.
****************************************************************************/
void F_Pool::freeToMark(
	void *		pvMark)					// free until pvMark found
{
	MBLK *	pBlock = m_pLastBlock;
	MBLK *	pPrevBlock;

	// Initialize pool to no blocks

	m_pLastBlock = NULL;
	while (pBlock)
	{
		pPrevBlock = pBlock->pPrevBlock;

		// Check for mark point

		if (PTR_IN_MBLK( pvMark, pBlock, pBlock->uiBlockSize))
		{
			FLMUINT  uiOldFreeOffset = pBlock->uiFreeOffset;

			// Reset uiFreeOffset and uiFreeSize variables

			pBlock->uiFreeOffset = (FLMUINT)((FLMBYTE *)pvMark -
														(FLMBYTE *)pBlock);
			pBlock->uiFreeSize = pBlock->uiBlockSize - pBlock->uiFreeOffset;

#if defined( FLM_MEM_CHK) || defined( MEM_TEST)
			{
			// memset the memory so someone pointing to it will get a error.
			FLMBYTE *	pucPtr = (FLMBYTE *)pBlock;
			pucPtr += pBlock->uiFreeOffset;
			f_memset( pucPtr, 'r', pBlock->uiFreeSize);	// Set memory to 'r' for Reset
			}
#endif

			// For Smart Pools deduct the bytes allocated since pool mark

			if (m_pPoolStats)
			{
				flmAssert( uiOldFreeOffset >= pBlock->uiFreeOffset);
				m_uiBytesAllocated -= (uiOldFreeOffset - pBlock->uiFreeOffset);
			}

			break;
		}

		if (m_pPoolStats)
		{
			m_uiBytesAllocated -= (pBlock->uiFreeOffset - sizeof( MBLK));
		}

		f_free( &pBlock);
		pBlock = pPrevBlock;
	}

	if (pBlock)
	{
		m_pLastBlock = pBlock;
	}
}

#undef	new
#undef	delete
/****************************************************************************
Desc:	
****************************************************************************/
void * XF_Base::operator new(
	FLMSIZET			uiSize)
{
	void *	pvReturnPtr = NULL;

	// NOTICE: You should be using f_new so that we can track memory leaks

	RC_UNEXPECTED_ASSERT( NE_XFLM_MEM);

#ifdef FLM_DEBUG
	f_allocImp( uiSize, &pvReturnPtr, TRUE, "unknown", 0);
#else
	f_allocImp( uiSize, &pvReturnPtr);
#endif

	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void * XF_Base::operator new[](
	FLMSIZET			uiSize)
{
	void *	pvReturnPtr = NULL;

	// NOTICE: You should be using f_new[] so that we can track memory leaks

#ifdef FLM_DEBUG
	f_allocImp( uiSize, &pvReturnPtr, TRUE, "unknown", 0);
#else
	f_allocImp( uiSize, &pvReturnPtr);
#endif

	return( pvReturnPtr);
}

/****************************************************************************
Desc:	
****************************************************************************/
#ifdef FLM_DEBUG
void * XF_Base::operator new(
	FLMSIZET			uiSize,
	const char *	pszFile,
	int				iLine)
{
	void *	pvReturnPtr = NULL;

	f_allocImp( uiSize, &pvReturnPtr, TRUE, pszFile, iLine);

	return( pvReturnPtr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#ifdef FLM_DEBUG
void * XF_Base::operator new[](
	FLMSIZET			uiSize,
	const char *	pszFile,
	int				iLine)
{
	void *	pvReturnPtr = NULL;

	f_allocImp( uiSize, &pvReturnPtr, TRUE, pszFile, iLine);

	return( pvReturnPtr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
void XF_Base::operator delete(
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}
	f_freeImp( &ptr, TRUE);
}

/****************************************************************************
Desc:	
****************************************************************************/
void XF_Base::operator delete[](
	void *			ptr)
{
	if( !ptr)
	{
		return;
	}
	f_freeImp( &ptr, TRUE);
}

/****************************************************************************
Desc:	
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
void XF_Base::operator delete(
	void *			ptr,
	const char *,	// file
	int)			// line
{
	if( !ptr)
	{
		return;
	}

	f_freeImp( &ptr, TRUE);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
void XF_Base::operator delete[](
	void *			ptr,
	const char *,	// file
	int)			// line
{
	if( !ptr)
	{
		return;
	}

	f_freeImp( &ptr, TRUE);
}
#endif

/************************************************************************
Desc:	
*************************************************************************/
void * F_OSBase::operator new(
	FLMSIZET			uiSize)
{
	return( os_malloc( uiSize));
}

/****************************************************************************
Desc:	
****************************************************************************/
#ifdef FLM_DEBUG
void * F_OSBase::operator new(
	FLMSIZET			uiSize,
	const char *,	// pszFile,
	int)				// iLine)
{
	return( os_malloc( uiSize));
}
#endif

/************************************************************************
Desc:	
*************************************************************************/
void F_OSBase::operator delete(
	void *			ptr)
{
	os_free( ptr);
}

/****************************************************************************
Desc:	
****************************************************************************/
void F_OSBase::operator delete[](
	void *			ptr)
{
	os_free( &ptr);
}

/****************************************************************************
Desc:	
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
void F_OSBase::operator delete(
	void *			ptr,
	const char *,	// file
	int)				// line
{
	os_free( &ptr);
}
#endif

/****************************************************************************
Desc:	
****************************************************************************/
#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
void F_OSBase::operator delete[](
	void *			ptr,
	const char *,	// file
	int)				// line
{
	os_free( &ptr);
}
#endif
