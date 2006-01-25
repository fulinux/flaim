//-------------------------------------------------------------------------
// Desc:	Buffer management for asynchronous I/O.
// Tabs:	3
//
//		Copyright (c) 2001-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fbuff.cpp 12246 2006-01-19 14:30:28 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc: 
****************************************************************************/
F_IOBufferMgr::F_IOBufferMgr()
{
	m_pFirstPending = NULL;
	m_pFirstAvail = NULL;
	m_pFirstUsed = NULL;
	m_uiMaxBuffers = 0;
	m_uiMaxBufferBytesToUse = 0;
	m_uiBufferBytesInUse = 0;
	m_uiBuffersInUse = 0;
	m_completionRc = FERR_OK;
	m_bKeepBuffers = FALSE;
}

/****************************************************************************
Desc: 
****************************************************************************/
F_IOBufferMgr::~F_IOBufferMgr()
{
	flmAssert( !m_pFirstPending && !m_pFirstUsed);
	while (m_pFirstPending)
	{
		m_pFirstPending->Release();
	}
	while (m_pFirstAvail)
	{
		m_pFirstAvail->Release();
	}
	while (m_pFirstUsed)
	{
		m_pFirstUsed->Release();
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE F_IOBufferMgr::waitForAllPendingIO( void)
{
	RCODE					rc = FERR_OK;
	F_IOBuffer *		pBuf;

	while( (pBuf = m_pFirstPending) != NULL)
	{
		(void)pBuf->waitToComplete();
	}

	rc = m_completionRc;
	m_completionRc = FERR_OK;

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
void F_IOBufferMgr::linkToList(
	F_IOBuffer **	ppListHead,
	F_IOBuffer *	pIOBuffer)
{
	flmAssert( pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_NONE);

	pIOBuffer->m_pPrev = NULL;
	if ((pIOBuffer->m_pNext = *ppListHead) != NULL)
	{
		(*ppListHead)->m_pPrev = pIOBuffer;
	}

	*ppListHead = pIOBuffer;
	if (ppListHead == &m_pFirstPending ||
		 ppListHead == &m_pFirstUsed)
	{
		pIOBuffer->m_eList = (ppListHead == &m_pFirstPending
									? F_IOBuffer::MGR_LIST_PENDING
									: F_IOBuffer::MGR_LIST_USED);
		pIOBuffer->m_bDeleteOnNotify = (m_bKeepBuffers
												  ? FALSE
												  : TRUE);
		m_uiBuffersInUse++;
		m_uiBufferBytesInUse += pIOBuffer->m_uiBufferSize;
	}
	else
	{
		pIOBuffer->m_eList = F_IOBuffer::MGR_LIST_AVAIL;
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
void F_IOBufferMgr::unlinkFromList(
	F_IOBuffer *	pIOBuffer)
{
	if (pIOBuffer->m_pNext)
	{
		pIOBuffer->m_pNext->m_pPrev = pIOBuffer->m_pPrev;
	}
	if (pIOBuffer->m_pPrev)
	{
		pIOBuffer->m_pPrev->m_pNext = pIOBuffer->m_pNext;
	}
	else if (pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_AVAIL)
	{
		m_pFirstAvail = pIOBuffer->m_pNext;
	}
	else if (pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_PENDING)
	{
		m_pFirstPending = pIOBuffer->m_pNext;
	}
	else if (pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_USED)
	{
		m_pFirstUsed = pIOBuffer->m_pNext;
	}
	else
	{
		flmAssert( 0);
	}
	if (pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_PENDING ||
		 pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_USED)
	{
		m_uiBuffersInUse--;
		flmAssert( m_uiBufferBytesInUse >= pIOBuffer->m_uiBufferSize);
		m_uiBufferBytesInUse -= pIOBuffer->m_uiBufferSize;
	}
	pIOBuffer->m_eList = F_IOBuffer::MGR_LIST_NONE;
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE F_IOBufferMgr::getBuffer(
	F_IOBuffer **		ppIOBuffer,
	FLMUINT				uiBufferSize,
	FLMUINT				uiBlockSize)
{
	RCODE					rc = FERR_OK;
	F_IOBuffer *		pIOBuffer = NULL;
	F_IOBuffer *		pBuf;

	if( RC_BAD( m_completionRc))
	{
		rc = m_completionRc;
		goto Exit;
	}

	if ((m_uiBufferBytesInUse + uiBufferSize > m_uiMaxBufferBytesToUse &&
		  m_pFirstPending) ||
		  m_uiBuffersInUse == m_uiMaxBuffers)
	{
		pBuf = m_pFirstPending;
		for (;;)
		{
			if( pBuf->isIOComplete())
			{
				if( RC_BAD( rc = pBuf->waitToComplete()))
				{
					goto Exit;
				}
				pBuf = m_pFirstPending;
				if (m_uiBufferBytesInUse + uiBufferSize > m_uiMaxBufferBytesToUse &&
					 m_pFirstPending)
				{
					continue;
				}
				else
				{
					flmAssert( m_uiBuffersInUse < m_uiMaxBuffers);
					break;
				}
			}
			if ((pBuf = pBuf->m_pNext) == NULL)
			{
#ifdef FLM_WIN
				f_sleep( 0);
#else
				f_yieldCPU();
#endif
				pBuf = m_pFirstPending;
			}
		}
	}

	// If we are set up to keep buffers, caller better always ask
	// for the same size.

	if (m_pFirstAvail)
	{
		pIOBuffer = m_pFirstAvail;
		unlinkFromList( pIOBuffer);
		flmAssert( pIOBuffer->getBufferSize() == uiBufferSize);
	}
	else
	{
		if ((pIOBuffer = f_new F_IOBuffer) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
		pIOBuffer->m_pIOBufferMgr = this;
		if (RC_BAD( rc = pIOBuffer->setupBuffer( uiBufferSize,
									uiBlockSize)))
		{
			goto Exit;
		}
	}

	// An F_IOBuffer object, once created must ALWAYS be linked
	// into the buffer manager's used list.

	linkToList( &m_pFirstUsed, pIOBuffer);

Exit:

	if (RC_BAD( rc) && pIOBuffer)
	{
		pIOBuffer->Release();
		pIOBuffer = NULL;
	}
	*ppIOBuffer = pIOBuffer;
	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
F_IOBuffer::F_IOBuffer()
{
	m_pIOBufferMgr = NULL;
	m_pucBuffer = NULL;
#ifdef FLM_DEBUG
	f_memset( m_UserData, 0, sizeof( m_UserData));
#endif
	m_uiBufferSize = 0;
	m_uiBlockSize = 0;
	m_eList = MGR_LIST_NONE;
	m_bDeleteOnNotify = TRUE;
	m_pNext = NULL;
	m_pPrev = NULL;
	m_fnCompletion = NULL;
	m_completionRc = FERR_OK;
#if defined( FLM_WIN)
	m_FileHandle = INVALID_HANDLE_VALUE;
	m_Overlapped.hEvent = 0;
#endif
#ifdef FLM_NLM
	m_hSem = F_SEM_NULL;
#endif
	m_pDbStats = NULL;
}

/****************************************************************************
Desc: 
****************************************************************************/
F_IOBuffer::~F_IOBuffer()
{
	// Unlink from list object is in, if any.

	if (m_eList != MGR_LIST_NONE)
	{
		flmAssert( m_pIOBufferMgr);
		m_pIOBufferMgr->unlinkFromList( this);
	}

#if defined( FLM_WIN)
	if( m_Overlapped.hEvent)
	{
		CloseHandle( m_Overlapped.hEvent);
	}
#endif

#ifdef FLM_NLM
	if (m_hSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hSem);
	}
#endif

	if (m_pucBuffer)
	{
#ifdef FLM_WIN
		(void)VirtualFree( m_pucBuffer, 0, MEM_RELEASE);
		m_pucBuffer = NULL;
#elif defined( FLM_LINUX) || defined( FLM_SOLARIS)
		free( m_pucBuffer);
		m_pucBuffer = NULL;
#else
		f_free( &m_pucBuffer);
#endif
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
void F_IOBuffer::makePending( void)
{
	flmAssert( m_eList == MGR_LIST_USED);

	// Unlink from used list

	m_pIOBufferMgr->unlinkFromList( this);

	// Link into pending list.

	m_pIOBufferMgr->linkToList( &m_pIOBufferMgr->m_pFirstPending, this);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE F_IOBuffer::setupBuffer(
	FLMUINT	uiBufferSize,
	FLMUINT	uiBlockSize)
{
	RCODE			rc = FERR_OK;

#if defined( FLM_WIN)
	if( (m_Overlapped.hEvent = CreateEvent( NULL, TRUE,
											FALSE, NULL)) == NULL)
	{
		rc = MapWinErrorToFlaim( GetLastError(),
					FERR_SETTING_UP_FOR_WRITE);
		goto Exit;
	}
#endif

#ifdef FLM_NLM
	if (RC_BAD( rc = f_semCreate( &m_hSem)))
	{
		goto Exit;
	}
#endif

	// Allocate a buffer

#ifdef FLM_WIN
	if ((m_pucBuffer = (FLMBYTE *)VirtualAlloc( NULL,
								(DWORD)uiBufferSize,
								MEM_COMMIT, PAGE_READWRITE)) == NULL)
	{
		rc = MapWinErrorToFlaim( GetLastError(), FERR_MEM);
		goto Exit;
	}
#elif defined( FLM_LINUX) || defined( FLM_SOLARIS)
	if( (m_pucBuffer = (FLMBYTE *)memalign( 
		sysconf(_SC_PAGESIZE), uiBufferSize)) == NULL) 
	{
		rc = MapErrnoToFlaimErr(errno, FERR_MEM);
		goto Exit;
	}
#else
	if (RC_BAD( rc = f_alloc( uiBufferSize, &m_pucBuffer)))
	{
		goto Exit;
	}
#endif

	m_uiBufferSize = uiBufferSize;
	m_uiBlockSize = uiBlockSize;

Exit:

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
void F_IOBuffer::notifyComplete(
	RCODE			rc)
{
	flmAssert( m_eList == MGR_LIST_PENDING ||
				  m_eList == MGR_LIST_USED);

	m_completionRc = rc;
	endTimer();
	if( m_fnCompletion)
	{
		m_fnCompletion( this);

		// Fix so completion callback won't be called twice.

		m_fnCompletion = NULL;
	}

	if (RC_BAD( rc) && RC_OK( m_pIOBufferMgr->m_completionRc))
	{
		m_pIOBufferMgr->m_completionRc = rc;
	}

	if (m_bDeleteOnNotify)
	{
		Release();
	}
	else
	{
		m_pIOBufferMgr->unlinkFromList( this);
		m_pIOBufferMgr->linkToList( &m_pIOBufferMgr->m_pFirstAvail, this);
	}
}

/****************************************************************************
Desc: 
****************************************************************************/
FLMBOOL F_IOBuffer::isIOComplete( void)
{
	FLMBOOL	bComplete = FALSE;
#ifdef FLM_NLM
	FLMUINT	uiSemCount;
#endif

	if( m_eList != MGR_LIST_PENDING)
	{
		bComplete = TRUE;
		goto Exit;
	}

	#ifdef FLM_WIN
	if (m_FileHandle == INVALID_HANDLE_VALUE ||
		 HasOverlappedIoCompleted( &m_Overlapped))
	{
		bComplete = TRUE;
	}
	#endif

#ifdef FLM_NLM
	if( (uiSemCount = (FLMUINT)kSemaphoreExamineCount( (SEMAPHORE)m_hSem)) != 0)
	{
		flmAssert( uiSemCount == 1);
		bComplete = TRUE;
	}
#endif

Exit:

	return( bComplete);
}

/****************************************************************************
Desc: 
****************************************************************************/
RCODE F_IOBuffer::waitToComplete( void)
{
	RCODE	rc = FERR_OK;

	// IMPORTANT NOTE! The call to notifyComplete will destroy this
	// object, so nothing in the object can be accessed after notifyComplete
	// is called.

#ifdef FLM_WIN
	if (m_FileHandle != INVALID_HANDLE_VALUE)
	{
		DWORD	udBytesWritten;

		if (!GetOverlappedResult( m_FileHandle, &m_Overlapped,
											&udBytesWritten, TRUE))
		{
			rc = MapWinErrorToFlaim( GetLastError(),
						FERR_WRITING_FILE);
		}

		notifyComplete( rc);
	}
#endif

#ifdef FLM_NLM
	if( kSemaphoreWait( (SEMAPHORE)m_hSem) != 0)
	{
		flmAssert( 0);
	}
	flmAssert( kSemaphoreExamineCount( (SEMAPHORE)m_hSem) == 0);
	rc = m_completionRc;
	notifyComplete( m_completionRc);
#endif

	return( rc);
}

/****************************************************************************
Desc: 
****************************************************************************/
#ifdef FLM_NLM
void F_IOBuffer::signalComplete(
	RCODE	rc)
{
	m_completionRc = rc;
	flmAssert( kSemaphoreExamineCount( (SEMAPHORE)m_hSem) == 0);
	kSemaphoreSignal( (SEMAPHORE)m_hSem);
}
#endif
