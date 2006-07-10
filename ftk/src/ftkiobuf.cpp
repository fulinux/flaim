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
	
	RCODE setupBufferMgr(
		FLMUINT				uiMaxBuffers,
		FLMUINT				uiMaxBytes,
		FLMBOOL				bReuseBuffers);

	RCODE FLMAPI getBuffer(
		FLMUINT				uiBufferSize,
		IF_IOBuffer **		ppIOBuffer);

	RCODE FLMAPI waitForAllPendingIO( void);

	FINLINE FLMBOOL FLMAPI isIOPending( void)
	{
		return( m_pFirstPending ? TRUE : FALSE);
	}

	void linkToList(
		F_IOBuffer **		ppListHead,
		F_IOBuffer *		pIOBuffer);

	void unlinkFromList(
		F_IOBuffer *		pIOBuffer);
		
private:

	FLMUINT					m_uiMaxBuffers;
	FLMUINT					m_uiMaxBufferBytes;
	FLMUINT					m_uiTotalBuffers;
	FLMUINT					m_uiTotalBufferBytes;
	F_IOBuffer *			m_pFirstPending;
	F_IOBuffer *			m_pFirstAvail;
	F_IOBuffer *			m_pFirstUsed;
	FLMBOOL					m_bReuseBuffers;
	RCODE						m_completionRc;

	friend class F_IOBuffer;
};

/****************************************************************************
Desc:	
****************************************************************************/
RCODE FLMAPI FlmAllocIOBufferMgr(
	FLMUINT					uiMaxBuffers,
	FLMUINT					uiMaxBytes,
	FLMBOOL					bReuseBuffers,
	IF_IOBufferMgr **		ppIOBufferMgr)
{
	RCODE						rc = NE_FLM_OK;
	F_IOBufferMgr *		pBufferMgr = NULL;
	
	if( (pBufferMgr = f_new F_IOBufferMgr) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}
	
	if( RC_BAD( pBufferMgr->setupBufferMgr( uiMaxBuffers, 
		uiMaxBytes, bReuseBuffers)))
	{
		goto Exit;
	}
	
	*ppIOBufferMgr = pBufferMgr;
	pBufferMgr = NULL;
	
Exit:
	
	if( pBufferMgr)
	{
		pBufferMgr->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
F_IOBufferMgr::F_IOBufferMgr()
{
	m_uiMaxBuffers = 0;
	m_uiMaxBufferBytes = 0;
	
	m_uiTotalBuffers = 0;
	m_uiTotalBufferBytes = 0;
	
	m_pFirstPending = NULL;
	m_pFirstAvail = NULL;
	m_pFirstUsed = NULL;
	
	m_bReuseBuffers = FALSE;
	m_completionRc = NE_FLM_OK;
}

/****************************************************************************
Desc:
****************************************************************************/
F_IOBufferMgr::~F_IOBufferMgr()
{
	f_assert( !m_pFirstPending);
	f_assert( !m_pFirstUsed);
	
	while( m_pFirstAvail)
	{
		m_pFirstAvail->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_IOBufferMgr::setupBufferMgr(
	FLMUINT			uiMaxBuffers,
	FLMUINT			uiMaxBytes,
	FLMBOOL			bReuseBuffers)
{
	f_assert( uiMaxBuffers);
	f_assert( uiMaxBytes);
	
	m_uiMaxBuffers = uiMaxBuffers;
	m_uiMaxBufferBytes = uiMaxBytes;
	m_bReuseBuffers = bReuseBuffers;
	
	return( NE_FLM_OK);
}
		
/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_IOBufferMgr::getBuffer(
	FLMUINT				uiBufferSize,
	IF_IOBuffer **		ppIOBuffer)
{
	RCODE					rc = NE_FLM_OK;
	F_IOBuffer *		pIOBuffer = NULL;
	
	f_assert( *ppIOBuffer == NULL);
	
	if( RC_BAD( m_completionRc))
	{
		rc = m_completionRc;
		goto Exit;
	}
	
Retry:

	if( m_pFirstAvail)
	{
		pIOBuffer = m_pFirstAvail;
		unlinkFromList( pIOBuffer);
		pIOBuffer->resetBuffer();
		f_assert( pIOBuffer->getBufferSize() == uiBufferSize);
	}
	else if( !m_uiTotalBuffers ||
		(m_uiTotalBufferBytes + uiBufferSize <= m_uiMaxBufferBytes &&
		m_uiTotalBuffers < m_uiMaxBuffers))
	{
		if( m_uiTotalBufferBytes + uiBufferSize > m_uiMaxBufferBytes)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
			goto Exit;
		}

		if( (pIOBuffer = f_new F_IOBuffer) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
		
		if (RC_BAD( rc = pIOBuffer->setupBuffer( uiBufferSize, this)))
		{
			goto Exit;
		}
		
		m_uiTotalBufferBytes += uiBufferSize;
		m_uiTotalBuffers++;
	}
	else if( m_pFirstPending)
	{
		if( RC_BAD( rc = m_pFirstPending->waitToComplete()))
		{
			goto Exit;
		}
				
		goto Retry;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_MEM);
		goto Exit;
	}
	
	pIOBuffer->AddRef();
	linkToList( &m_pFirstUsed, pIOBuffer);
	*ppIOBuffer = pIOBuffer;
	pIOBuffer = NULL;
	
Exit:

	if( pIOBuffer)
	{
		pIOBuffer->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_IOBufferMgr::waitForAllPendingIO( void)
{
	RCODE				rc;
	RCODE				tmpRc;
	F_IOBuffer *	pBuf;

	while( (pBuf = m_pFirstPending) != NULL)
	{
		if( RC_BAD( tmpRc = pBuf->waitToComplete()))
		{
			if( RC_OK( m_completionRc))
			{
				m_completionRc = tmpRc;
			}
		}
	}
	
	rc = m_completionRc;
	m_completionRc = NE_FLM_OK;
	
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBufferMgr::linkToList(
	F_IOBuffer **		ppListHead,
	F_IOBuffer *		pIOBuffer)
{
	f_assert( pIOBuffer->m_eList == MGR_LIST_NONE);
	
	pIOBuffer->m_pPrev = NULL;
	
	if( (pIOBuffer->m_pNext = *ppListHead) != NULL)
	{
		(*ppListHead)->m_pPrev = pIOBuffer;
	}
	
	*ppListHead = pIOBuffer;
	
	if( ppListHead == &m_pFirstPending || ppListHead == &m_pFirstUsed)
	{
		pIOBuffer->m_eList = (ppListHead == &m_pFirstPending
									? MGR_LIST_PENDING
									: MGR_LIST_USED);
	}
	else
	{
		pIOBuffer->m_eList = MGR_LIST_AVAIL;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBufferMgr::unlinkFromList(
	F_IOBuffer *	pIOBuffer)
{
	if( pIOBuffer->m_pNext)
	{
		pIOBuffer->m_pNext->m_pPrev = pIOBuffer->m_pPrev;
	}
	
	if( pIOBuffer->m_pPrev)
	{
		pIOBuffer->m_pPrev->m_pNext = pIOBuffer->m_pNext;
	}
	else if( pIOBuffer->m_eList == MGR_LIST_AVAIL)
	{
		m_pFirstAvail = pIOBuffer->m_pNext;
	}
	else if( pIOBuffer->m_eList == MGR_LIST_PENDING)
	{
		m_pFirstPending = pIOBuffer->m_pNext;
	}
	else if( pIOBuffer->m_eList == MGR_LIST_USED)
	{
		m_pFirstUsed = pIOBuffer->m_pNext;
	}
	
	pIOBuffer->m_eList = MGR_LIST_NONE;
}

/****************************************************************************
Desc:
****************************************************************************/
FLMINT FLMAPI F_IOBuffer::Release( void)
{
	FLMINT		iRefCnt;

	if( m_refCnt <= 2)
	{
		if( m_pBufferMgr && m_eList != MGR_LIST_NONE)
		{
			m_pBufferMgr->unlinkFromList( this);
		}
	}
	
	if( m_refCnt == 2)
	{
		if( m_pBufferMgr)
		{
			if( m_pBufferMgr->m_bReuseBuffers)
			{
				m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstAvail, this);
			}
			else
			{
				f_assert( m_pBufferMgr->m_uiTotalBuffers);
				f_assert( m_pBufferMgr->m_uiTotalBufferBytes >= m_uiBufferSize);

				m_refCnt--;
				m_pBufferMgr->m_uiTotalBuffers--;
				m_pBufferMgr->m_uiTotalBufferBytes -= m_uiBufferSize;
				m_pBufferMgr = NULL;
			}
		}
	}
	
	iRefCnt = --m_refCnt;
	
	if( !iRefCnt)
	{
		delete this;
	}
	
	return( iRefCnt);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_IOBuffer::setupBuffer(
	FLMUINT				uiBufferSize,
	F_IOBufferMgr *	pBufferMgr)
{
	RCODE					rc = NE_FLM_OK;

	if( RC_BAD( rc = f_allocAlignedBuffer( uiBufferSize, 
		(void **)&m_pucBuffer)))
	{
		goto Exit;
	}
	
	m_uiBufferSize = uiBufferSize;
	m_pBufferMgr = pBufferMgr;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_IOBuffer::setPending( void)
{
	f_assert( !m_bPending);
	
	m_bPending = TRUE;
	m_uiStartTime = FLM_GET_TIMER();
	
	if( m_pBufferMgr)
	{
		f_assert( m_eList == MGR_LIST_USED);
		
		m_pBufferMgr->unlinkFromList( this);
		m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstPending, this);
	}
}
		
/****************************************************************************
Desc:
****************************************************************************/
void FLMAPI F_IOBuffer::clearPending( void)
{
	f_assert( m_bPending);
	
	m_bPending = FALSE;
	m_uiStartTime = 0;
	
	if( m_pBufferMgr)
	{
		f_assert( m_eList == MGR_LIST_PENDING);
		
		m_pBufferMgr->unlinkFromList( this);
		m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstUsed, this);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
void F_IOBuffer::notifyComplete(
	RCODE					completionRc)
{
	f_assert( m_bPending);
	
	m_bPending = FALSE;
	m_bCompleted = TRUE;
	m_completionRc = completionRc;
	m_uiEndTime = FLM_GET_TIMER();
	m_uiElapsedTime = FLM_TIMER_UNITS_TO_MILLI( 
		FLM_ELAPSED_TIME( m_uiEndTime, m_uiStartTime));

	if( m_fnCompletion)
	{
		m_fnCompletion( this, m_pvData);
		m_fnCompletion = NULL;
		m_pvData = NULL;
	}
	
	if( m_pBufferMgr)
	{
		f_assert( m_eList == MGR_LIST_PENDING);
		
		m_pBufferMgr->unlinkFromList( this);
		m_pBufferMgr->linkToList( &m_pBufferMgr->m_pFirstUsed, this);
		
		if( RC_OK( m_pBufferMgr->m_completionRc) && RC_BAD( completionRc))
		{
			m_pBufferMgr->m_completionRc = completionRc;
		}
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE FLMAPI F_IOBuffer::addCallbackData(
	void *							pvData)
{
	RCODE			rc = NE_FLM_OK;
	
	if( m_uiCallbackDataCount >= m_uiMaxCallbackData)
	{
		if( m_ppCallbackData == m_callbackData)
		{
			void **	pNewTable;
			
			if( RC_BAD( rc = f_alloc( 
				(m_uiCallbackDataCount + 1) * sizeof( void *), &pNewTable)))
			{
				goto Exit;
			}
			
			f_memcpy( pNewTable, m_ppCallbackData, 
				m_uiMaxCallbackData * sizeof( void *));
			m_ppCallbackData = pNewTable;
		}
		else
		{
			if( RC_BAD( rc = f_realloc( 
				(m_uiCallbackDataCount + 1) * sizeof( void *), &m_ppCallbackData)))
			{
				goto Exit;
			}
		}
		
		m_uiMaxCallbackData = m_uiCallbackDataCount + 1;
	}
	
	m_ppCallbackData[ m_uiCallbackDataCount] = pvData;
	m_uiCallbackDataCount++;
	
Exit:

	return( rc);
}
			
/****************************************************************************
Desc:
****************************************************************************/
void * FLMAPI F_IOBuffer::getCallbackData(
	FLMUINT							uiSlot)
{
	if( uiSlot < m_uiCallbackDataCount)
	{
		return( m_ppCallbackData[ uiSlot]);
	}
	
	return( NULL);
}
