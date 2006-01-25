//-------------------------------------------------------------------------
// Desc:	Object reference tracker
// Tabs:	3
//
//		Copyright (c) 1999-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fobjtrck.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#if defined( FLM_UNIX) && !defined( FLM_OSX)
	#include <dlfcn.h>
#endif

/****************************************************************************
The NetWare Internal Debugger encrypts all symbols by XORing each character 
in the symbol name with the character at the corresponding position in the 
following mask. To see how the crypt mask is defined in NetWare, see 
SYMDEB.386 in the NetWare source. We have three options:
	
	1. We can just emulate the internal debugger and decrypt the symbol, 
		print it, and reencrypt it each time we want to display it (less than 
		efficient). 
	2. We can make a decrypted copy of the symbols we are interested in on 
		module init, then use our own copy and free it on module exit.
	3. We can use the symbol list, but decrypt character by character into
		an internal string buffer, then print the buffer.
****************************************************************************/

#ifdef FLM_NLM
	extern "C"
	{
		void GetClosestSymbol(
			BYTE *	szBuffer,
			LONG		udAddress);
	}
#endif

/****************************************************************************
Desc:
****************************************************************************/
F_ObjRefTracker::F_ObjRefTracker(void)
{
	m_hRefListMutex = F_MUTEX_NULL;
	m_pListManager = NULL;;
	m_pFileSystem = NULL;
	m_pszObjName[ 0] = '\0';
	m_bLocalFS = FALSE;
	m_pAddrFmtHook = NULL;
	m_pUserData = NULL;
	m_pModHandle = NULL;
}

/****************************************************************************
Desc: Allocates required memory and initializes a local file system
****************************************************************************/
RCODE F_ObjRefTracker::setup(
	const char *	pszObjName,
	FLMBOOL			bLogToFile)
{
	RCODE				rc = FERR_OK;
	char				pszTmpBuf[ FORTRACK_MAX_OBJ_NAME_LEN + 5];
	char *			pucTmp;

	// Allocate a mutex

	if( RC_BAD( rc = f_mutexCreate( &m_hRefListMutex)))
	{
		goto Exit;
	}

	// Allocate the list

	if( (m_pListManager = f_new F_ListMgr) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( m_pListManager->Setup( &m_lnode, 1)))
	{
		goto Exit;
	}

	// Create a local file system object

	if( bLogToFile)
	{
		if( (m_pFileSystem = f_new F_FileSystemImp) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		m_bLocalFS = TRUE;
	}

	if( f_strlen( pszObjName) <= FORTRACK_MAX_OBJ_NAME_LEN)
	{
		f_strcpy( m_pszObjName, pszObjName);
	}
	else
	{
		f_sprintf( m_pszObjName, "OBJTRCK");
	}

	// Set the log path

	f_strcpy( pszTmpBuf, m_pszObjName);
	pucTmp = pszTmpBuf;
	while( *pucTmp)
	{
		if( *pucTmp >= 'a' && *pucTmp <= 'z')
		{
			*pucTmp = (*pucTmp - 'a') + 'A';
		}
		pucTmp++;
	}
	f_strcat( pszTmpBuf, ".OTL");

#ifdef FLM_NLM
	f_strcpy( m_pLogPath, "SYS:SYSTEM");
	f_pathAppend( m_pLogPath, pszTmpBuf);
#else
	f_strcpy( m_pLogPath, pszTmpBuf);
#endif

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_ObjRefTracker::~F_ObjRefTracker(void)
{
	if( m_pListManager)
	{
		m_pListManager->Release();
		m_pListManager = NULL;
	}

	if( m_hRefListMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hRefListMutex);
	}

	if( m_pFileSystem && m_bLocalFS)
	{
		m_pFileSystem->Release();
	}
}

/****************************************************************************
Desc: Track the given reference.
****************************************************************************/
void F_ObjRefTracker::trackRef(
	void *		pReferenceID,
	void *		pSubrefID)
{
	TrackingRecord *	pTrackingRec = NULL;
	void **				pStack;
	
	if( m_hRefListMutex == F_MUTEX_NULL)
	{
		// Reference Tracking has not been initialized, just exit.
		
		goto Exit;
	}

	if( !pReferenceID)
	{
		// Do not track NULL references
		
		goto Exit;
	}

	// If there is insufficient memory to allocate a tracking record, 
	// then we will never know if this reference is properly released.
	
	if( (pTrackingRec = f_new TrackingRecord( pReferenceID, pSubrefID)) == NULL)
	{
		char	pucMessage[ 100];
		
		logError( "trackRef: Insufficient memory to allocate tracking record");
		f_sprintf( pucMessage, "\treference %x.%x will not be tracked",
			(unsigned)((FLMUINT)pReferenceID), (unsigned)((FLMUINT)pSubrefID));
		logError( pucMessage);
		goto Exit;
	}

	if( RC_BAD( pTrackingRec->Setup( m_pListManager, &m_lnode, 1)))
	{
		goto Exit;
	}

	//Lock the list
	f_mutexLock( m_hRefListMutex);
	
	// Add the tracking record to the list
	m_pListManager->InsertAtEnd( 0, pTrackingRec);

	//Unlock the list
	f_mutexUnlock( m_hRefListMutex);

	pStack = (void **)pTrackingRec->getStack();
	getCallStack( pStack, CTRC_STACK_SIZE, /*skip = */ 1);

Exit:
	return;
}

/****************************************************************************
Desc:	This reference has been released, don't track it any more.
****************************************************************************/
void F_ObjRefTracker::untrackRef(
	void *		pReferenceID,
	void *		pSubrefID)
{
	TrackingRecord *	pTrackingRec = NULL;
	FLMBOOL				bListLocked = FALSE;

	if( m_hRefListMutex == F_MUTEX_NULL)
	{
		// Reference Tracking has not been initialized, just exit.
		
		goto Exit;
	}

	if( !pReferenceID)
	{
		goto Exit;
	}
	
	// Lock the list
	
	f_mutexLock( m_hRefListMutex);
	bListLocked	= TRUE;
	
	// Try to find the reference in the list
	
	pTrackingRec = (TrackingRecord *) m_pListManager->GetItem( 0, 0);
	while( pTrackingRec)
	{
		if( pTrackingRec->getReferenceID() == pReferenceID
		 && pTrackingRec->getSubrefID() == pSubrefID)
		{
			// The reference has been found.
			
			pTrackingRec->RemoveFromList();
			pTrackingRec->Release();
			break;
		}
		
		pTrackingRec = (TrackingRecord *) pTrackingRec->GetNextListItem();
	}
	
	if( !pTrackingRec)
	{
		// The reference was never tracked.  This isn't supposed to happen.
		
		char	pucMessage[100];
		
		f_sprintf( pucMessage, 
			"untrackRef: Reference %x.%x was not tracked", 
			(unsigned)((FLMUINT)pReferenceID), (unsigned)((FLMUINT)pSubrefID));
		logError( pucMessage);
		logError( "\tModify code to track this reference");
		goto Exit;
	}
	
Exit:

	if( bListLocked)
	{
		f_mutexUnlock( m_hRefListMutex);
	}
}

/****************************************************************************
Desc: Check the list for references that were never released.
****************************************************************************/
void F_ObjRefTracker::checkForUnreleasedRefs(
	FLMUINT *		puiCount)
{
	RCODE					rc = FERR_OK;
	TrackingRecord *	pTrackingRec;
	FLMUINT				uiFileCursor;
	FLMUINT				uiLoop;
	char					pucSymbol[ 125];
	char					pucBuffer[ 150];
	FLMBOOL				bHeaderDisplayed;
	F_FileHdl *			pFileHdl = NULL;
	FLMBOOL				bListLocked = FALSE;
	FLMUINT				uiCount = 0;

	if( m_hRefListMutex == F_MUTEX_NULL)
	{
		logError( "checkForUnreleasedReferences: Reference tracking "
					 "was not initialized");
		goto Exit;
	}

	if( m_pFileSystem)
	{
		if( RC_BAD( rc = m_pFileSystem->Open( m_pLogPath,
			F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYNONE,
			&pFileHdl)))
		{
			if( RC_BAD( rc = m_pFileSystem->Create( m_pLogPath,
				F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYNONE,
				&pFileHdl)))
			{
				goto Exit;
			}
		}
	}

	// Find EOF so text can be appended to the trace file.
	
	if( pFileHdl)
	{
		if( RC_BAD( rc = pFileHdl->Size( &uiFileCursor)))
		{
			goto Exit;
		}
	}

	// Lock the list
	
	f_mutexLock( m_hRefListMutex);
	bListLocked	= TRUE;
	bHeaderDisplayed = FALSE;
	
	// Process all unreleased references
	
	for( pTrackingRec = (TrackingRecord *)	m_pListManager->GetItem( 0, 0);
   		pTrackingRec;
			pTrackingRec = (TrackingRecord *) m_pListManager->GetItem( 0, 0))
	{
		void **		pStack;

		uiCount++;
		if( !bHeaderDisplayed)
		{
			f_sprintf( pucBuffer, "Unreleased references of type [%s]\n",
				m_pszObjName);

			if( RC_BAD( rc = logMessage( pucBuffer, pFileHdl, uiFileCursor)))
			{
				goto Exit;
			}
			
			bHeaderDisplayed = TRUE;
		}
		
		if( RC_BAD( rc = logMessage( " ", pFileHdl, uiFileCursor)))
		{
			goto Exit;
		}

		f_sprintf( pucBuffer, "    Unreleased reference (%X.%X) from thread: %X\n",
			(unsigned)((FLMUINT)pTrackingRec->getReferenceID()),
			(unsigned)((FLMUINT)pTrackingRec->getSubrefID()),
			(unsigned) pTrackingRec->getThreadID());
		
		if( RC_BAD( rc = logMessage( pucBuffer, pFileHdl, uiFileCursor)))
		{
			goto Exit;
		}

		pStack = (void **) pTrackingRec->getStack();
		for( uiLoop = 0; pStack[ uiLoop]; uiLoop++ )
		{
			formatAddress( pucSymbol, sizeof( pucSymbol), pStack[ uiLoop]);
			f_sprintf( pucBuffer, "        %-45.45s [addr = %8.8x]\n", pucSymbol,
				(unsigned)((FLMUINT)pStack[ uiLoop]));

			if( RC_BAD( rc = logMessage( pucBuffer, pFileHdl, uiFileCursor)))
			{
				goto Exit;
			}
		}
		
		m_pListManager->RemoveItem( 0, pTrackingRec);
	}

Exit:

	if( bListLocked)
	{
		f_mutexUnlock( m_hRefListMutex);
	}

	if( pFileHdl)
	{
		pFileHdl->Close();
		pFileHdl->Release();
	}

	if( puiCount)
	{
		*puiCount = uiCount;
	}
}

/****************************************************************************
Desc:	Sets the address coonversion callback function
****************************************************************************/
void F_ObjRefTracker::setAddressFormatter(
	ADDR_FMT_HOOK 		pFunc,
	void *				pvUserData)
{
	m_pAddrFmtHook = pFunc;
	m_pUserData = pvUserData;
}

/****************************************************************************
Desc:	Converts a return address to displayable format
****************************************************************************/
void F_ObjRefTracker::formatAddress(
	char *		pucBuffer,
	FLMUINT		uiSize,
	void *		pAddress)
{
#ifdef FLM_WIN
	PIMAGEHLP_SYMBOL			pihs = NULL;
#ifdef FLM_64BIT
	DWORD64						displacement;
#else
	DWORD							displacement;
#endif
	RCODE							rc = FERR_OK;
#endif

	if( m_pAddrFmtHook)
	{
		pucBuffer[ 0] = '\0';
		m_pAddrFmtHook( this, pAddress, (FLMBYTE *)pucBuffer, uiSize, m_pUserData);
		return;
	}

#if defined( FLM_NLM)

	if( uiSize == 0)
	{
		return;
	}

	GetClosestSymbol( (BYTE *)pucBuffer, (LONG)pAddress);

	return;

#elif defined( FLM_WIN)

	if( RC_OK( rc = f_alloc( sizeof( IMAGEHLP_SYMBOL) + 100, &pihs)))
	{
		pihs->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
		pihs->Address = (FLMUINT)pAddress; //stackFrame.AddrPC.Offset;
		pihs->MaxNameLength = (FLMINT32)uiSize;

		if ( SymGetSymFromAddr( GetCurrentProcess(),	(FLMUINT)pAddress,
										&displacement, pihs ) )
		{
			wsprintf( pucBuffer, "%s + %X",
				(char *)pihs->Name, (unsigned)displacement);
		}
		else
		{
			wsprintf( pucBuffer, "0x%08X", (unsigned)((FLMUINT)pAddress));
		}
	}
	else
	{
		wsprintf( pucBuffer, "0x%08X", (unsigned)((FLMUINT)pAddress));
	}

	f_free( &pihs);

#else
#ifdef HAVE_DLADDR
	Dl_info dlip;
	if (dladdr(pAddress, &dlip) != 0 && dlip.dli_sname)
	{
		const char *filename;
		if (dlip.dli_saddr != pAddress)
		{
			filename = strrchr(dlip.dli_fname, '/');
			if (!filename)
				filename = dlip.dli_fname;
			else
				filename++;		// skip over slash
			f_sprintf( pucBuffer, "0x%08x (%s)", (unsigned)((FLMUINT)pAddress),
						filename); 
		}
		else
			f_sprintf( pucBuffer, "%s", dlip.dli_sname);
		return;
	}
#endif
	f_sprintf( pucBuffer, "0x%08x", (unsigned)((FLMUINT)pAddress));
#endif
}

/****************************************************************************
Desc:	Walk the BP chain down the call stack, gathering return addresses.
****************************************************************************/
#if defined( FLM_WIN) && !defined( FLM_64BIT)
void F_ObjRefTracker::getCallStack(
	void *						stack[],
	FLMUINT						uiCount,
	FLMUINT						uiSkip)
{
	STACKFRAME					stackFrame;
	FLMINT32						ui32LastBP;
	FLMUINT						uiLoop;

	F_UNREFERENCED_PARM( uiSkip);
	ZeroMemory( (PVOID)&stackFrame, sizeof(STACKFRAME) );

	// TDOMAN: do this in assembly since we aren't sure we can rely on the
	//			  GetThreadContext and StackWalk API's
	_asm
	{
		mov	ui32LastBP, ebp						// save off next bp
	}

	// while you can continue walking the stack...
	for ( uiLoop = 0; uiLoop < uiCount; uiLoop++ )
	{
		// TDOMAN: we have to walk the stack ourselves since the VC4x API's
		//			  don't appear to be consistently reliable.
		__try
		{
			// TDOMAN: don't crash if the last bp wasn't want we expected
			_asm
			{
				push	esi
				push	edi
				mov	edi, ui32LastBP
				mov	esi, [edi]
				cmp	esi, edi
				 jbe	Done
				mov	ui32LastBP, esi				// save off next bp
				mov	esi, [edi + 4]
				mov	stackFrame, esi			// setup AddrPC
				pop	edi
				pop	esi
			}
		}
		__except (EXCEPTION_EXECUTE_HANDLER)
		{
			// If you do want to get these exceptions, you can turn off stack
			//	walking by setting fStackWalk to FALSE.
			goto Done;
		}
		stack[ uiLoop] = (void *)(FLMUINT)stackFrame.AddrPC.Offset;
	}

Done:
	stack[ uiLoop] = (void *)0;
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_UNIX) || defined( FLM_64BIT)
void F_ObjRefTracker::getCallStack(
	void *						stack[],
	FLMUINT,						//uiCount,
	FLMUINT)						//uiSkip)
{
	// Not supported on this platform
	stack[0] = (void *)0;
}
#endif

/****************************************************************************
****************************************************************************/
#if defined( FLM_NLM)

	void *DMGetEBP(void);

#if defined( __MWERKS__)


	void *DMGetEBP(void)
	{
		__asm
		{
			mov	eax,[ebp]
		}
	}  // end of assembly

#else

	#pragma aux DMGetEBP = "mov eax,ebp";

#endif

/****************************************************************************
Desc:
****************************************************************************/
void * DMValueAtStackOffset(void *pos, int offset);

#if defined( __MWERKS__)

	void *DMValueAtStackOffset(void *, int )
	{
		__asm
		{
			mov	eax,[ebp+0x8]
			mov	ebx,[ebp+0xC]
			mov	eax,ss:[eax+ebx]
		}
	}

#else

	#pragma aux DMValueAtStackOffset = "mov eax,ss:[eax+ebx]" parm [eax] [ebx];

#endif

/****************************************************************************
Desc: Traces back COUNT entries on the call stack, storing
		them in STACK. Note that this code requires that DS be build using the 
		Watcom /of+ option, (or equivalent) to generate traceable stack frames by
		emitting prelude code for every function that looks something like that
		of the SubRoutine below:
		
		Caller:
			push Parms						; caller pushes parameters
			call SubRoutine				; caller pushes his own return address
			add esp, parmSize				; caller clears parameters from stack
			...
		SubRoutine:		
			push ebp							; pushes caller's frame pointer
			mov ebp, esp					; creates SubRoutine's frame pointer
			...
			pop ebp							; restores caller's frame pointer
			ret								; returns to caller
		
		In this scheme, the MOV instruction in the prelude code always sets EBP 
		pointing to the PUSHed value of the previous (caller's) frame pointer 
		(see the first instruction in SubRoutine above). We discard the first
		'skipCount' + 1 return addresses because we aren't interested in the fact
		that the caller called DMAlloc, which called DMAllocFromTag,
		which called getCallStack. 
		The stack trace loop terminates when it detects a return address that is 
		outside of NDS code space (start and limit are stored in the load def 
		struct - module handle). Some inline assembly is used to access the stack 
		like data (see DMGetEBP and DMValueAtStackOffset above).
****************************************************************************/
void F_ObjRefTracker::getCallStack(
	void *		stack[],
	FLMUINT		uiCount,
	FLMUINT		uiSkipCount)
{
	FLMUINT		uiLoop;
 	void *		rtnAddr;
	void *		ebp = DMGetEBP();

	while( uiSkipCount--)
	{
		ebp = DMValueAtStackOffset( ebp, 0);
	}
	rtnAddr = DMValueAtStackOffset( ebp, 4);

	for( uiLoop = 0; --uiCount; )
	{
		void *oldebp;
		
		stack[ uiLoop++] = rtnAddr;
		if( !ebp)
		{
			break;
		}
		
		oldebp = ebp;
		ebp = DMValueAtStackOffset( ebp, 0);				// Caller's frame ptr

		if ( !ebp || ebp <= oldebp || ebp > (void *)((char *)oldebp+3000))
		{
			break;
		}

		rtnAddr = DMValueAtStackOffset( ebp, 4);			// Caller's return addr
	}
	stack[ uiLoop] = 0;
	return;
}
#endif // defined( FLM_NLM)

/****************************************************************************
Desc: Log an error message
****************************************************************************/
void F_ObjRefTracker::logError(
	const char *	pucMessage)
{
	char				pucBuffer[ 120];
	FLMUINT			uiDummy = 0;

	f_sprintf( pucBuffer, "Error: %s", pucMessage);
	logMessage( pucBuffer, NULL, uiDummy);

	flmAssert(0);
}

/****************************************************************************
Desc: Log a message to the trace file and to the DS trace screen
****************************************************************************/
RCODE F_ObjRefTracker::logMessage(
	const char *	message,
	F_FileHdl *		pFileHdl,
	FLMUINT &		fileCursor)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiBytesWritten;
	FLMBOOL			bFileOpened = FALSE;
	const char *	pCarriageReturn = "\n";
	
	if( !pFileHdl && m_pFileSystem)
	{
		if( RC_BAD( rc = m_pFileSystem->Open(
			m_pLogPath, F_IO_RDWR | F_IO_SH_DENYNONE, &pFileHdl)))
		{
			if( RC_BAD( rc = m_pFileSystem->Create(
				m_pLogPath, F_IO_RDWR | F_IO_EXCL | F_IO_SH_DENYNONE,
				&pFileHdl)))
			{
				goto Exit;
			}
		}

		bFileOpened = TRUE;
		flmAssert( pFileHdl);

		//Find EOF so text can be appended to the trace file.
		if( RC_BAD( rc = pFileHdl->Size( &fileCursor)))
		{
			goto Exit;
		}
	}

	if( pFileHdl)
	{
		if( RC_BAD( rc = pFileHdl->Write(
			fileCursor, (FLMUINT)f_strlen(message), 
			(void *)message, &uiBytesWritten)))
		{
			goto Exit;
		}
		fileCursor += uiBytesWritten;

		if( RC_BAD( rc = pFileHdl->Write(
			fileCursor, (FLMUINT)f_strlen(pCarriageReturn), 
			(void *)pCarriageReturn, &uiBytesWritten)))
		{
			fileCursor += uiBytesWritten;
		}
	}

Exit:

	if( bFileOpened)
	{
		pFileHdl->Close();
		pFileHdl->Release();
	}

	return( rc);
}	
