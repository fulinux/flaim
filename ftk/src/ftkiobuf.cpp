//------------------------------------------------------------------------------
// Desc:	This file contains the F_IOBuffer and F_IOBufferMgr classes.
//
// Tabs:	3
//
//		Copyright (c) 2001-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fbuff.cpp 3111 2006-01-19 13:10:50 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

/****************************************************************************
Desc:
****************************************************************************/
class F_IOBufferMgr : public IF_IOBufferMgr
{
public:

	F_IOBufferMgr();

	virtual ~F_IOBufferMgr();

	RCODE FLMAPI waitForAllPendingIO( void);

	FINLINE void FLMAPI setMaxBuffers(
		FLMUINT			uiMaxBuffers)
	{
		m_uiMaxBuffers = uiMaxBuffers;
	}

	FINLINE void FLMAPI setMaxBytes(
		FLMUINT			uiMaxBytes)
	{
		m_uiMaxBufferBytesToUse = uiMaxBytes;
	}

	FINLINE void FLMAPI enableKeepBuffer( void)
	{
		m_bKeepBuffers = TRUE;
	}

	RCODE FLMAPI getBuffer(
		IF_IOBuffer **		ppIOBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT				uiBlockSize);

	FINLINE FLMBOOL FLMAPI havePendingIO( void)
	{
		return( m_pFirstPending ? TRUE : FALSE);
	}

	FINLINE FLMBOOL FLMAPI haveUsed( void)
	{
		return( m_pFirstUsed ? TRUE : FALSE);
	}

private:

	// Private methods and variables

	F_IOBuffer *		m_pFirstPending;
	F_IOBuffer *		m_pFirstAvail;
	F_IOBuffer *		m_pFirstUsed;
	FLMUINT				m_uiMaxBuffers;
	FLMUINT				m_uiMaxBufferBytesToUse;
	FLMUINT				m_uiBufferBytesInUse;
	FLMUINT				m_uiBuffersInUse;
	RCODE					m_completionRc;
	FLMBOOL				m_bKeepBuffers;

	void linkToList(
		F_IOBuffer **	ppListHead,
		F_IOBuffer *	pIOBuffer);

	void unlinkFromList(
		F_IOBuffer *	pIOBuffer);

friend class F_IOBuffer;

};

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FLMAPI FlmAllocIOBufferMgr(
	IF_IOBufferMgr **			ppIOBufferMgr)
{
	if( (*ppIOBufferMgr = f_new F_IOBufferMgr) == NULL)
	{
		return( RC_SET( NE_FLM_MEM));
	}
	
	return( NE_FLM_OK);
}

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
	m_completionRc = NE_FLM_OK;
	m_bKeepBuffers = FALSE;
}

/****************************************************************************
Desc:
****************************************************************************/
F_IOBufferMgr::~F_IOBufferMgr()
{
	f_assert( !m_pFirstPending && !m_pFirstUsed);
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
RCODE FLMAPI F_IOBufferMgr::waitForAllPendingIO( void)
{
	RCODE				rc = NE_FLM_OK;
	F_IOBuffer *	pBuf;

	while( (pBuf = m_pFirstPending) != NULL)
	{
		(void)pBuf->waitToComplete();
	}

	rc = m_completionRc;
	m_completionRc = NE_FLM_OK;

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBufferMgr::linkToList(
	F_IOBuffer **	ppListHead,
	F_IOBuffer *	pIOBuffer)
{
	f_assert( pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_NONE);
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
		f_assert( 0);
	}
	if (pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_PENDING ||
		 pIOBuffer->m_eList == F_IOBuffer::MGR_LIST_USED)
	{
		m_uiBuffersInUse--;
		f_assert( m_uiBufferBytesInUse >= pIOBuffer->m_uiBufferSize);
		m_uiBufferBytesInUse -= pIOBuffer->m_uiBufferSize;
	}
	pIOBuffer->m_eList = F_IOBuffer::MGR_LIST_NONE;
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_IOBufferMgr::getBuffer(
	IF_IOBuffer **		ppIOBuffer,
	FLMUINT				uiBufferSize,
	FLMUINT				uiBlockSize)
{
	RCODE				rc = NE_FLM_OK;
	F_IOBuffer *	pIOBuffer = NULL;
	F_IOBuffer *	pBuf;

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
					f_assert( m_uiBuffersInUse < m_uiMaxBuffers);
					break;
				}
			}
			
			if ((pBuf = pBuf->m_pNext) == NULL)
			{
				f_yieldCPU();
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
		f_assert( pIOBuffer->getBufferSize() == uiBufferSize);
	}
	else
	{
		if ((pIOBuffer = f_new F_IOBuffer) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
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
	
#ifdef FLM_RING_ZERO_NLM
	f_assert( kSemaphoreExamineCount( pIOBuffer->m_hSem) == 0);
#endif

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
	m_completionRc = NE_FLM_OK;
#if defined( FLM_WIN)
	m_FileHandle = INVALID_HANDLE_VALUE;
	m_Overlapped.hEvent = 0;
#elif defined( FLM_LINUX) || defined( FLM_SOLARIS)
	m_aio.aio_fildes = -1;
#endif
#ifdef FLM_RING_ZERO_NLM
	m_hSem = NULL;
#endif
	m_pStats = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_IOBuffer::~F_IOBuffer()
{
	// Unlink from list object is in, if any.

	if (m_eList != MGR_LIST_NONE)
	{
		f_assert( m_pIOBufferMgr);
		m_pIOBufferMgr->unlinkFromList( this);
	}

#if defined( FLM_WIN)
	if( m_Overlapped.hEvent)
	{
		CloseHandle( m_Overlapped.hEvent);
	}
#endif

#ifdef FLM_RING_ZERO_NLM
	if( m_hSem)
	{
		(void)kSemaphoreFree( m_hSem);
		m_hSem = NULL;
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
void FLMAPI F_IOBuffer::makePending( void)
{
	f_assert( m_eList == MGR_LIST_USED);

	// Unlink from used list

	m_pIOBufferMgr->unlinkFromList( this);

	// Link into pending list.

	m_pIOBufferMgr->linkToList( &m_pIOBufferMgr->m_pFirstPending, this);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMBOOL FLMAPI F_IOBuffer::isPending( void)
{
	if( m_eList == MGR_LIST_PENDING)
	{
		return( TRUE);
	}
	
	return( FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_IOBuffer::setupBuffer(
	FLMUINT		uiBufferSize,
	FLMUINT		uiBlockSize)
{
	RCODE			rc = NE_FLM_OK;

#if defined( FLM_WIN)
	if( (m_Overlapped.hEvent = CreateEvent( NULL, TRUE,
											FALSE, NULL)) == NULL)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_SETTING_UP_FOR_WRITE);
		goto Exit;
	}
#endif

#ifdef FLM_RING_ZERO_NLM
	if( (m_hSem = kSemaphoreAlloc( (BYTE *)"FTK_SEM", 0)) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
#endif

	// Allocate a buffer

#ifdef FLM_WIN
	if ((m_pucBuffer = (FLMBYTE *)VirtualAlloc( NULL,
								(DWORD)uiBufferSize,
								MEM_COMMIT, PAGE_READWRITE)) == NULL)
	{
		rc = f_mapPlatformError( GetLastError(), NE_FLM_MEM);
		goto Exit;
	}
#elif defined( FLM_LINUX)
	if( posix_memalign( (void **)&m_pucBuffer, 
		sysconf( _SC_PAGESIZE), uiBufferSize) != 0)
	{
		rc = f_mapPlatformError( errno, NE_FLM_MEM);
		goto Exit;
	}
#elif defined( FLM_SOLARIS)
	if( (m_pucBuffer = (FLMBYTE *)memalign( sysconf( _SC_PAGESIZE),
		uiBufferSize)) == NULL)
	{
		rc = f_mapPlatformError( errno, NE_FLM_MEM);
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
void FLMAPI F_IOBuffer::notifyComplete(
	RCODE			rc)
{
	f_assert( m_eList == MGR_LIST_PENDING ||
				  m_eList == MGR_LIST_USED);

	m_completionRc = rc;
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
	FLMBOOL		bComplete = FALSE;
#ifdef FLM_RING_ZERO_NLM
	FLMUINT		uiSemCount;
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

#if defined( FLM_LINUX) || defined( FLM_SOLARIS)
	if( m_aio.aio_fildes == -1 || aio_error( &m_aio) != EINPROGRESS)
	{
		bComplete = TRUE;
	}
#endif

#ifdef FLM_RING_ZERO_NLM
	if( (uiSemCount = (FLMUINT)kSemaphoreExamineCount( m_hSem)) != 0)
	{
		f_assert( uiSemCount == 1);
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
	RCODE	rc = NE_FLM_OK;

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
			rc = f_mapPlatformError( GetLastError(), NE_FLM_WRITING_FILE);
		}

		notifyComplete( rc);
	}
#endif

#if defined( FLM_LINUX) || defined( FLM_SOLARIS)
	if( m_aio.aio_fildes != -1)
	{
		FLMINT						iAsyncResult;
		const struct aiocb *		ppAio[ 1];
		
		ppAio[ 0] = &m_aio;

		for( ;;)
		{
			aio_suspend( ppAio, 1, NULL);
			iAsyncResult = aio_error( &m_aio);
	
			if( !iAsyncResult)
			{
				if( (iAsyncResult = aio_return( &m_aio)) < 0)
				{
					f_assert( 0);
					rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
					goto WriteComplete;
				}
					
				break;
			}
				
			if( iAsyncResult == EINTR || iAsyncResult == EINPROGRESS)
			{
				continue;
			}
					
			f_assert( 0);
			rc = f_mapPlatformError( errno, NE_FLM_WRITING_FILE);
			goto WriteComplete;
		}
		
WriteComplete:
		
		notifyComplete( rc);
	}
#endif

#ifdef FLM_RING_ZERO_NLM
	if( kSemaphoreWait( m_hSem) != 0)
	{
		f_assert( 0);
	}
	
	f_assert( kSemaphoreExamineCount( m_hSem) == 0);
	rc = m_completionRc;
	notifyComplete( m_completionRc);
#endif

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_IOBuffer::startTimer(
	void *		pStats)
{
	if ((m_pStats = pStats) != NULL)
	{
		m_ui64ElapMilli = 0;
		f_timeGetTimeStamp( &m_StartTime);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT64 FLMAPI F_IOBuffer::getElapTime( void)
{
	return( m_ui64ElapMilli);
}

/****************************************************************************
Desc:
****************************************************************************/
void * FLMAPI F_IOBuffer::getStats( void)
{
	return( m_pStats);
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_IOBuffer::signalComplete(
	RCODE				rc)
{
#ifdef FLM_RING_ZERO_NLM
	m_completionRc = rc;
	f_assert( kSemaphoreExamineCount( m_hSem) == 0);
	kSemaphoreSignal( m_hSem);
#else
	F_UNREFERENCED_PARM( rc);
#endif
}
