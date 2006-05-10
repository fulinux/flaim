//------------------------------------------------------------------------------
//	Desc:	Shared utility routines
//
// Tabs:	3
//
//		Copyright (c) 1997-2006 Novell, Inc. All Rights Reserved.
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
// $Id: sharutil.cpp 3129 2006-01-25 11:46:17 -0700 (Wed, 25 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#include "flaimsys.h"
#include "sharutil.h"

FSTATIC FTX_WINDOW * wpsGetThrdWin( void);

FINLINE void wpsLock(
	F_MUTEX	*	phMutex)
{
	f_mutexLock( *phMutex);
}

FINLINE void wpsUnlock(
	F_MUTEX	*	phMutex)
{
	f_mutexUnlock( *phMutex);
}

static FLMBOOL						gv_bShutdown = FALSE;
static FLMBOOL						gv_bInitialized = FALSE;
static FLMBOOL						gv_bOptimize = FALSE;
static WPSSCREEN *				gv_pScreenList = NULL;
static F_MUTEX						gv_hDispMutex = F_MUTEX_NULL;

FSTATIC RCODE propertyExists(
	char *	pszProperty,
	char *	pszBuffer,
	char **	ppszValue);

FSTATIC RCODE _flmWrapperFunc(
	IF_Thread *		pThread);

/********************************************************************
Desc: Parses command-line parameters
*********************************************************************/
void flmUtilParseParams(
	char *		pszCommandBuffer,
	FLMINT		iMaxArgs,
	FLMINT *		iArgcRV,
	char **		ppArgvRV)
{
	FLMINT	iArgC = 0;

	for (;;)
	{
		/* Strip off leading white space. */

		while ((*pszCommandBuffer == ' ') || (*pszCommandBuffer == '\t'))
			pszCommandBuffer++;
		if (!(*pszCommandBuffer))
			break;

		if ((*pszCommandBuffer == '"') || (*pszCommandBuffer == '\''))
		{
			char cQuoteChar = *pszCommandBuffer;

			pszCommandBuffer++;
			ppArgvRV [iArgC] = pszCommandBuffer;
			iArgC++;
			while ((*pszCommandBuffer) && (*pszCommandBuffer != cQuoteChar))
				pszCommandBuffer++;
			if (*pszCommandBuffer)
				*pszCommandBuffer++ = 0;
		}
		else
		{
			ppArgvRV [iArgC] = pszCommandBuffer;
			iArgC++;
			while ((*pszCommandBuffer) &&
					 (*pszCommandBuffer != ' ') &&
					 (*pszCommandBuffer != '\t'))
				pszCommandBuffer++;
			if (*pszCommandBuffer)
				*pszCommandBuffer++ = 0;
		}

		/* Quit if we have reached the maximum allowable number of arguments. */

		if (iArgC == iMaxArgs)
			break;
	}
	*iArgcRV = iArgC;
}

/****************************************************************************
Name:	FlmVector::setElementAt
Desc:	a vector set item operation.  
****************************************************************************/
#define FLMVECTOR_START_AMOUNT 16
#define FLMVECTOR_GROW_AMOUNT 2
RCODE FlmVector::setElementAt( void * pData, FLMUINT uiIndex)
{
	RCODE rc = NE_XFLM_OK;
	if ( !m_pElementArray)
	{		
		TEST_RC( rc = f_calloc( sizeof( void*) * FLMVECTOR_START_AMOUNT,
			&m_pElementArray));
		m_uiArraySize = FLMVECTOR_START_AMOUNT;
	}

	if ( uiIndex >= m_uiArraySize)
	{		
		TEST_RC( rc = f_recalloc(
			sizeof( void*) * m_uiArraySize * FLMVECTOR_GROW_AMOUNT,
			&m_pElementArray));
		m_uiArraySize *= FLMVECTOR_GROW_AMOUNT;
	}

	m_pElementArray[ uiIndex] = pData;
Exit:
	return rc;
}

/****************************************************************************
Name:	FlmVector::getElementAt
Desc:	a vector get item operation
****************************************************************************/
void * FlmVector::getElementAt( FLMUINT uiIndex)
{
	//if you hit this you are indexing into the vector out of bounds.
	//unlike a real array, we can catch this here!  oh joy!
	flmAssert ( uiIndex < m_uiArraySize);	
	return m_pElementArray[ uiIndex];
}

/****************************************************************************
Name:	FlmStringAcc::appendCHAR
Desc:	append a char (or the same char many times) to the string
****************************************************************************/
RCODE FlmStringAcc::appendCHAR( char ucChar, FLMUINT uiHowMany)
{
	RCODE rc = NE_XFLM_OK;
	if ( uiHowMany == 1)
	{
		FLMBYTE szStr[ 2];
		szStr[ 0] = ucChar;
		szStr[ 1] = 0;
		rc = this->appendTEXT( (const FLMBYTE*)szStr);
	}
	else
	{
		FLMBYTE * pszStr;
		
		if( RC_BAD( rc = f_alloc( uiHowMany + 1, &pszStr)))
		{
			goto Exit;
		}
		f_memset( pszStr, ucChar, uiHowMany);
		pszStr[ uiHowMany] = 0;
		rc = this->appendTEXT( pszStr);
		f_free( &pszStr);
	}
Exit:
	return rc;
}

/****************************************************************************
Name:	FlmStringAcc::appendTEXT
Desc:	appending text to the accumulator safely.  all other methods in
		the class funnel through this one, as this one contains the logic
		for making sure storage requirements are met.
****************************************************************************/
RCODE FlmStringAcc::appendTEXT( const FLMBYTE * pszVal)
{	
	RCODE 			rc = NE_XFLM_OK;
	FLMUINT 			uiIncomingStrLen;
	FLMUINT 			uiStrLen;

	//be forgiving if they pass in a NULL
	if ( !pszVal)
	{
		goto Exit;
	}
	//also be forgiving if they pass a 0-length string
	else if( (uiIncomingStrLen = f_strlen( (const char *)pszVal)) == 0)
	{
		goto Exit;
	}
	
	//compute total size we need to store the new total
	if ( m_bQuickBufActive || m_pszVal)
	{
		uiStrLen = uiIncomingStrLen + m_uiValStrLen;
	}
	else
	{
		uiStrLen = uiIncomingStrLen;
	}

	//just use small buffer if it's small enough
	if ( uiStrLen < FSA_QUICKBUF_BUFFER_SIZE)
	{
		f_strcat( m_szQuickBuf, (const char *)pszVal);
		m_bQuickBufActive = TRUE;
	}
	//we are exceeding the quickbuf size, so get the bytes from the heap
	else
	{
		//ensure storage requirements are met (and then some)
		if ( m_pszVal == NULL)
		{
			FLMUINT uiNewBytes = (uiStrLen+1) * 4;
			if ( RC_OK ( rc = f_alloc(
				(FLMUINT)(sizeof( FLMBYTE) * uiNewBytes),
				&m_pszVal)))
			{
				m_uiBytesAllocatedForPszVal = uiNewBytes;
				m_pszVal[ 0] = 0;
			}
			else
			{
				goto Exit;
			}
		}
		else if ( (m_uiBytesAllocatedForPszVal-1) < uiStrLen)
		{
			FLMUINT uiNewBytes = (uiStrLen+1) * 4;
			if ( RC_OK( rc = f_realloc(
				(FLMUINT)(sizeof( FLMBYTE) * uiNewBytes),
				&m_pszVal)))
			{
				m_uiBytesAllocatedForPszVal = uiNewBytes;
			}
			else
			{
				goto Exit;
			}
		}

		//if transitioning from quick buf to heap buf, we need to
		//transfer over the quick buf contents and unset the flag
		if ( m_bQuickBufActive)
		{
			m_bQuickBufActive = FALSE;
			f_strcpy( m_pszVal, m_szQuickBuf);
			//no need to zero out m_szQuickBuf because it will never
			//be used again, unless a clear() is issued, in which
			//case it will be zeroed out then.
		}		

		//copy over the string
		f_strcat( m_pszVal, (const char *)pszVal);
	}
	m_uiValStrLen = uiStrLen;
Exit:
	return rc;
}

/****************************************************************************
Desc:	printf into the FlmStringAcc
****************************************************************************/
RCODE FlmStringAcc::printf(
	const char * pszFormatString,
	...)
{
	f_va_list		args;
	char *			pDestStr = NULL;
	const size_t	iSize = 4096;
	RCODE				rc = NE_XFLM_OK;

	if( RC_BAD( rc = f_alloc( iSize, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	this->clear();
	TEST_RC( rc = this->appendTEXT( (FLMBYTE *)pDestStr));

Exit:
	if ( pDestStr)
	{
		f_free( &pDestStr);
	}
	return rc;
}

/****************************************************************************
Desc:	formatted appender like sprintf
****************************************************************************/
RCODE FlmStringAcc::appendf(
	const char * pszFormatString,
	...)
{
	f_va_list		args;
	char *			pDestStr = NULL;
	const size_t	iSize = 4096;
	RCODE				rc = NE_XFLM_OK;

	if( RC_BAD( rc = f_alloc( iSize, &pDestStr)))
	{
		goto Exit;
	}

	f_va_start( args, pszFormatString);
	f_vsprintf( pDestStr, pszFormatString, &args);
	f_va_end( args);

	TEST_RC( rc = this->appendTEXT( (FLMBYTE *)pDestStr));

Exit:

	if ( pDestStr)
	{
		f_free( &pDestStr);
	}
	return rc;
}
	
/****************************************************************************
Desc:	Constructor
*****************************************************************************/
FlmContext::FlmContext()
{
	m_szCurrDir[ 0] = '\0';
	m_hMutex = F_MUTEX_NULL;
	m_bIsSetup = FALSE;
}

/****************************************************************************
Desc:	Destructor
*****************************************************************************/
FlmContext::~FlmContext( void)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
		m_hMutex = F_MUTEX_NULL;
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmContext::setup(
	FLMBOOL		bShared)
{
	RCODE		rc = NE_XFLM_OK;

	if( bShared)
	{
		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}

	m_bIsSetup = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmContext::setCurrDir(
	FLMBYTE *	pszCurrDir)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bIsSetup);

	lock();
	f_strcpy( (char *)m_szCurrDir, (const char *)pszCurrDir);
	unlock();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmContext::getCurrDir(
	FLMBYTE *	pszCurrDir)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( m_bIsSetup);

	lock();
	f_strcpy( (char *)pszCurrDir, (const char *)m_szCurrDir);
	unlock();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmContext::lock( void)
{
	flmAssert( m_bIsSetup);

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmContext::unlock( void)
{
	flmAssert( m_bIsSetup);

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmThreadContext::FlmThreadContext( void)
{
	m_pScreen = NULL;
	m_pWindow = NULL;
	m_bShutdown = FALSE;
	m_pLocalContext = NULL;
	m_pSharedContext = NULL;
	m_pNext = NULL;
	m_pPrev = NULL;
	m_uiID = 0;
	m_hMutex = F_MUTEX_NULL;
	m_pThrdFunc = NULL;
	m_pvAppData = NULL;
	m_pThread = NULL;
	m_bFuncExited = FALSE;
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmThreadContext::~FlmThreadContext( void)
{
	// Free the local context
	if( m_pLocalContext)
	{
		m_pLocalContext->Release();
	}

	// Destroy the semaphore
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::lock( void)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::unlock( void)
{
	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexUnlock( m_hMutex);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmThreadContext::setup(
	FlmSharedContext *	pSharedContext,
	char *					pszThreadName,
	THREAD_FUNC_p			pFunc,
	void *					pvAppData)
{
	RCODE		rc = NE_XFLM_OK;

	flmAssert( pSharedContext != NULL);

	m_pSharedContext = pSharedContext;
	m_pThrdFunc = pFunc;
	m_pvAppData = pvAppData;

	if( (m_pLocalContext = f_new FlmContext) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}

	if( pszThreadName &&
		f_strlen( pszThreadName) <= MAX_THREAD_NAME_LEN)
	{
		f_strcpy( m_szName, pszThreadName);
	}
	else
	{
		f_sprintf( m_szName, "flmGenericThread");
	}

	if( RC_BAD( rc = m_pLocalContext->setup( FALSE)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::getName(
	char *		pszName,
	FLMBOOL		bLocked)
{
	if( !bLocked)
	{
		lock();
	}

	f_strcpy( pszName, m_szName);

	if( !bLocked)
	{
		unlock();
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmThreadContext::execute( void)
{
	flmAssert( m_pThrdFunc != NULL);
	m_FuncRC = m_pThrdFunc( this, m_pvAppData);
	return m_FuncRC;
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmThreadContext::shutdown()
{
	m_bShutdown = TRUE;
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmThreadContext::exec( void)
{
	flmAssert( m_pThrdFunc != NULL);
	return( (RCODE)(m_pThrdFunc( this, m_pvAppData)));
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmSharedContext::FlmSharedContext( void)
{
	m_pParentContext = NULL;
	m_pThreadList = NULL;
	m_bLocalShutdownFlag = FALSE;
	m_pbShutdownFlag = &m_bLocalShutdownFlag;
	m_hSem = F_SEM_NULL;
	m_uiNextProcID = 1;
	m_bPrivateShare = FALSE;
}

/****************************************************************************
Desc:
*****************************************************************************/
FlmSharedContext::~FlmSharedContext( void)
{
	// Clean up the thread list
	shutdown();

	// Free the ESem
	if( m_hSem != F_SEM_NULL)
	{
		f_semDestroy( &m_hSem);
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::init(
	FlmSharedContext *	pSharedContext)
{
	RCODE		rc = NE_XFLM_OK;

	// Initialize the base class
	if( RC_BAD( rc = FlmContext::setup( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_semCreate( &m_hSem)))
	{
		goto Exit;
	}

	m_pParentContext = pSharedContext;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmSharedContext::shutdown( void)
{
	FLMBOOL	bLocked = FALSE;

	*m_pbShutdownFlag = TRUE;

	for( ;;)
	{
		lock();
		bLocked = TRUE;
		if( m_pThreadList)
		{
			m_pThreadList->shutdown();
		}
		else
		{
			break;
		}
		unlock();
		bLocked = FALSE;
		(void)f_semWait( m_hSem, 1000);
	}

	if( bLocked)
	{
		unlock();
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
void FlmSharedContext::wait( void)
{
	FLMBOOL	bLocked = FALSE;

	for( ;;)
	{
		lock();
		bLocked = TRUE;
		if( !m_pThreadList)
		{
			break;
		}
		unlock();
		bLocked = FALSE;
		(void)f_semWait( m_hSem, 1000);
	}

	if( bLocked)
	{
		unlock();
	}
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::spawn(
	FlmThreadContext *	pThread,
	FLMUINT *				puiThreadID)
{
	RCODE						rc = NE_XFLM_OK;
	char						szName[ MAX_THREAD_NAME_LEN + 1];
	IF_ThreadMgr *			pThreadMgr = NULL;

	registerThread( pThread);
	pThread->getName( szName);
	
	if( RC_BAD( rc = FlmGetThreadMgr( &pThreadMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pThreadMgr->createThread( NULL,
		_flmWrapperFunc, szName, 0, 0, pThread)))
	{
		goto Exit;
	}

	if( puiThreadID)
	{
		*puiThreadID = pThread->getID();
	}

Exit:

	if( pThreadMgr)
	{
		pThreadMgr->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::spawn(
	char *					pszThreadName,
	THREAD_FUNC_p			pFunc,
	void *					pvUserData,
	FLMUINT *				puiThreadID)
{
	FlmThreadContext *	pThread;
	RCODE						rc = NE_XFLM_OK;

	if( (pThread = f_new FlmThreadContext) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pThread->setup( this, pszThreadName, pFunc, pvUserData)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = spawn( pThread, puiThreadID)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::registerThread(
	FlmThreadContext *		pThread)
{
	RCODE		rc = NE_XFLM_OK;

	lock();
	pThread->setNext( m_pThreadList);
	if( m_pThreadList)
	{
		m_pThreadList->setPrev( pThread);
	}
	m_pThreadList = pThread;
	pThread->setID( m_uiNextProcID++);
	unlock();

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::deregisterThread(
	FlmThreadContext *		pThread)
{
	FlmThreadContext *	pTmpThrd;
	RCODE						rc = NE_XFLM_OK;

	lock();
	pTmpThrd = m_pThreadList;
	while( pTmpThrd)
	{
		if( pTmpThrd == pThread)
		{
			if( pTmpThrd->getPrev())
			{
				pTmpThrd->getPrev()->setNext( pTmpThrd->getNext());
			}

			if( pTmpThrd->getNext())
			{
				pTmpThrd->getNext()->setPrev( pTmpThrd->getPrev());
			}

			if( pTmpThrd == m_pThreadList)
			{
				m_pThreadList = pTmpThrd->getNext();
			}

			pTmpThrd->Release();
			break;
		}

		pTmpThrd = pTmpThrd->getNext();
	}

	f_semSignal( m_hSem);
	unlock();
	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::killThread(
	FLMUINT		uiThreadID,
	FLMUINT		uiMaxWait)
{
	FlmThreadContext *	pThread;
	FLMUINT					uiStartTime;
	RCODE						rc = NE_XFLM_OK;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			pThread->shutdown();
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	// Wait for the thread to exit
	uiStartTime = FLM_GET_TIMER();
	uiMaxWait = FLM_SECS_TO_TIMER_UNITS( uiMaxWait);
	for( ;;)
	{
		(void)f_semWait( m_hSem, 200);
		lock();
		pThread = m_pThreadList;
		while( pThread)
		{
			if( pThread->getID() == uiThreadID)
			{
				break;
			}
			pThread = pThread->getNext();
		}
		unlock();

		if( !pThread)
		{
			break;
		}

		if( uiMaxWait)
		{
			if( FLM_GET_TIMER() - uiStartTime >= uiMaxWait)
			{
				rc = RC_SET( NE_XFLM_FAILURE);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::setFocus(
	FLMUINT		uiThreadID)
{
	FlmThreadContext *	pThread;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			if( pThread->getScreen())
			{
				FTXScreenDisplay( pThread->getScreen());
			}
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
*****************************************************************************/
FLMBOOL FlmSharedContext::isThreadTerminating(
	FLMUINT		uiThreadID)
{
	FLMBOOL					bTerminating = FALSE;
	FlmThreadContext *	pThread;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			if( pThread->getShutdownFlag())
			{
				bTerminating = TRUE;
			}
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	return( bTerminating);
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE FlmSharedContext::getThread(
	FLMUINT					uiThreadID,
	FlmThreadContext **	ppThread)
{
	FlmThreadContext *	pThread;

	lock();
	pThread = m_pThreadList;
	while( pThread)
	{
		if( pThread->getID() == uiThreadID)
		{
			if( ppThread)
			{
				*ppThread = pThread;
			}
			break;
		}
		pThread = pThread->getNext();
	}
	unlock();

	return( ((pThread != NULL)
			? NE_XFLM_OK
			: RC_SET( NE_XFLM_NOT_FOUND)));
}

/****************************************************************************
Desc:
*****************************************************************************/
RCODE _flmWrapperFunc(
	IF_Thread *		pFlmThread)
{
	FlmThreadContext *	pThread = (FlmThreadContext *)pFlmThread->getParm1();
	FlmSharedContext *	pSharedContext = pThread->getSharedContext();

	pThread->setFlmThread( pFlmThread);
	if( RC_BAD( pThread->execute()))
	{
		goto Exit;
	}

Exit:

	pThread->setFuncExited();
	pThread->setFlmThread( NULL);

	// Unlink the thread from the shared context
	pSharedContext->deregisterThread( pThread);
	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:	callback to use to output a line
****************************************************************************/
void utilOutputLine( 
	char *				pszData, 
	void * 				pvUserData)
{
	FTX_WINDOW * 		pMainWindow = (FTX_WINDOW*)pvUserData;
	eColorType			uiBack, uiFore;
		
	FTXWinGetBackFore( pMainWindow, &uiBack, &uiFore);
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, "%s\n", pszData);
}

/****************************************************************************
Name:	utilPressAnyKey
Desc:	callback to serve as a 'pager' function when the Usage: help
		is too long to fit on one screen.
****************************************************************************/ 
void utilPressAnyKey( char * pszMessage, void * pvUserData)
{
	FTX_WINDOW *		pMainWindow = (FTX_WINDOW*)pvUserData;
	FLMUINT 				uiChar;
	eColorType			uiBack, uiFore;
	
	FTXWinGetBackFore( pMainWindow, &uiBack, &uiFore);
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, pszMessage);
	while( RC_BAD( FTXWinTestKB( pMainWindow)))
	{
		f_sleep( 100);
	}
	FTXWinCPrintf( pMainWindow, uiBack, uiFore,
		"\r                                                                  ");
	FTXWinCPrintf( pMainWindow, uiBack, uiFore, "\r");
	FTXWinInputChar( pMainWindow, &uiChar);
}

/****************************************************************************
Name:	utilInitWindow
Desc:	routine to startup the TUI
****************************************************************************/
RCODE utilInitWindow(
	char *			pszTitle,
	FLMUINT *		puiScreenRows,
	FTX_WINDOW **	ppMainWindow,
	FLMBOOL *		pbShutdown)
{
	FTX_SCREEN *	pScreen = NULL;
	FTX_WINDOW *	pTitleWin = NULL;
	FLMUINT			uiCols;
	int				iResCode = 0;

	if( RC_BAD( FTXInit( pszTitle, (FLMBYTE)80, (FLMBYTE)50,
		FLM_BLUE, FLM_WHITE, NULL, NULL)))
	{
		iResCode = 1;
		goto Exit;
	}

	FTXSetShutdownFlag( pbShutdown);

	if( RC_BAD( FTXScreenInit( pszTitle, &pScreen)))
	{
		iResCode = 1;
		goto Exit;
	}
	
	if( RC_BAD( FTXScreenGetSize( pScreen, &uiCols, puiScreenRows)))
	{
		iResCode = 1;
		goto Exit;
	}

	if( RC_BAD( FTXScreenInitStandardWindows( pScreen, FLM_RED, FLM_WHITE,
		FLM_BLUE, FLM_WHITE, FALSE, FALSE, pszTitle,
		&pTitleWin, ppMainWindow)))
	{
		iResCode = 1;
		goto Exit;
	}
	
Exit:
	return (RCODE)iResCode;
}

/****************************************************************************
Name:	utilShutdownWindow
Desc:	routine to shutdown the TUI
****************************************************************************/
void utilShutdownWindow()
{
	FTXExit();
}
	
/****************************************************************************
Desc:	read the contents of the argument file into the ppszReturnString buffer
****************************************************************************/
RCODE fileToString(
	char *	pszFile,
	char **	ppszReturnString)
{
	RCODE					rc = NE_XFLM_OK;
	char *				pszBuffer = NULL;
	IF_FileHdl *		pFileHdl = NULL;
	FLMUINT64			ui64FileSize = 0;
	FLMUINT				uiBytesRead = 0;
	IF_FileSystem *	pFileSystem = NULL;
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	if (RC_BAD(rc = pFileSystem->openFile( pszFile, FLM_IO_RDONLY, &pFileHdl)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pFileHdl->size( &ui64FileSize)))
	{
		goto Exit;
	}
	
	if( ui64FileSize == 0)
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_alloc( (FLMUINT)(ui64FileSize + 1), &pszBuffer)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pFileHdl->read( 0, (FLMUINT)ui64FileSize,
		pszBuffer, &uiBytesRead)))
	{
		goto Exit;
	}
	
	flmAssert( ui64FileSize == uiBytesRead);
	pszBuffer[ ui64FileSize] = 0;
	
Exit:

	if( pFileHdl)
	{
		pFileHdl->Release();
	}
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}
	
	if ( RC_BAD( rc) && pszBuffer)
	{
		f_free( &pszBuffer);
	}
	else if ( RC_OK( rc))
	{
		*ppszReturnString = pszBuffer;
	}
	
	return( rc);
}

/****************************************************************************
Desc:	allocate a copy of the arg string and return it out
****************************************************************************/
char * getStringClone(
	char * pszSrcStr)
{
	char * pszReturnVal = NULL;

	if( RC_BAD( f_alloc( f_strlen( pszSrcStr) + 1, &pszReturnVal)))
	{
		goto Exit;
	}
	f_strcpy( pszReturnVal, pszSrcStr);

Exit:

	return pszReturnVal;
}

/****************************************************************************
Desc:	fill a buffer with the current (or given) time
****************************************************************************/
FLMUINT utilGetTimeString(
	char *		pszOutString,
	FLMUINT		uiBufferSize,
	FLMUINT		uiInSeconds) //default param to use user-supplied time
{
	F_TMSTAMP timeStamp;
	FLMUINT uiSeconds;

	if ( uiInSeconds != 0)
	{
		f_timeSecondsToDate( uiInSeconds, &timeStamp);
	}
	else
	{
		f_timeGetTimeStamp( &timeStamp);
	}
	f_timeDateToSeconds( &timeStamp, &uiSeconds);
	char szTemp[ 256];
	f_sprintf( szTemp,
		"%4u-%02u-%02u %02u:%02u:%02u",
		(unsigned)timeStamp.year,
		(unsigned)(timeStamp.month + 1),
		(unsigned)timeStamp.day,
		(unsigned)timeStamp.hour,
		(unsigned)timeStamp.minute,
		(unsigned)timeStamp.second);
	f_strncpy( pszOutString, szTemp, uiBufferSize - 1);
	pszOutString[ uiBufferSize-1] = 0;
	return uiSeconds;
}

#define UTIL_PROP_DELIMITER '!'
FSTATIC RCODE propertyExists(
	char *	pszProperty,
	char *	pszBuffer,
	char **	ppszValue)
{
	flmAssert( pszProperty);
	FlmStringAcc acc;
	RCODE rc = NE_XFLM_OK; //returns only memory errors

	*ppszValue = NULL;

	if ( !pszBuffer)
	{
		goto Exit;
	}
	else
	{
		char *	pszValue;

		acc.appendf( "%s%c", pszProperty, UTIL_PROP_DELIMITER);
		if ( (pszValue = (char *)f_strstr( pszBuffer, pszProperty)) != NULL)
		{
			pszValue = (char *)(1 + f_strchr( pszValue, UTIL_PROP_DELIMITER));
			*ppszValue = getStringClone( pszValue);
			if ( !*ppszValue)
			{
				rc = RC_SET( NE_XFLM_MEM);
				goto Exit;
			}
			((FLMBYTE*)(f_strchr( *ppszValue, '\n')))[ 0] = 0; 
		}
		else
		{
			goto Exit;
		}
	}
Exit:
	return rc;
}

RCODE utilWriteProperty(
	char *		pszFile,
	char *		pszProp,
	char *		pszValue)
{
	RCODE					rc = NE_XFLM_OK;
	char *				pszContents = NULL;
	char *				pszExistingProperty;
	FlmStringAcc		newContents;
	IF_FileSystem *	pFileSystem = NULL;
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	//can't have newlines in the props or values
	flmAssert( !f_strchr( pszProp, '\n'));
	flmAssert( !f_strchr( pszValue, '\n'));

	if ( RC_BAD( pFileSystem->doesFileExist( pszFile)))
	{
		//add trailing newline
		TEST_RC( rc = f_filecpy( pszFile, "")); 
	}
	if ( RC_BAD( fileToString( pszFile, &pszContents)))
	{
		goto Exit;
	}
	//propertyExists returns out a new
	TEST_RC( rc = propertyExists( pszProp, pszContents, &pszExistingProperty));
	if ( !pszExistingProperty)
	{
		newContents.appendf( "%s%c%s\n", pszProp, UTIL_PROP_DELIMITER, pszValue);
		newContents.appendTEXT( (FLMBYTE *)pszContents);
	}
	else
	{
		f_free( &pszExistingProperty);
		pszExistingProperty = NULL;
		FLMUINT uiProps = 0;
		
		//write out nulls in place of the "\n"'s throughout the contents
		char *	pszNuller = pszContents;
		for( ;;)
		{
			pszNuller = (char *)f_strchr( pszNuller, '\n');
			if ( pszNuller)
			{
				pszNuller[ 0] = 0;
				pszNuller++;
				uiProps++;
			}
			else
			{
				break;
			}
		}
		char *	pszNextLine = pszContents;
		char *	pszNextProp;
		char *	pszNextVal;
		char *	pszBang;

		while ( uiProps--)
		{
			pszBang = (char *)f_strchr( pszNextLine, UTIL_PROP_DELIMITER);
			flmAssert( pszBang); //better have a UTIL_PROP_DELIMITER in it
			pszBang[ 0] = 0;
			pszNextProp = pszNextLine;
			pszNextVal = pszBang + 1;
			if ( !(STREQ( pszNextProp, pszProp)))
			{
				pszBang[ 0] = UTIL_PROP_DELIMITER;
				newContents.appendTEXT( (FLMBYTE *)pszNextLine);
				newContents.appendCHAR( '\n');
			}
			else
			{				
				newContents.appendf( "%s%c%s\n",
					pszNextProp, UTIL_PROP_DELIMITER, pszValue);
				pszBang[ 0] = UTIL_PROP_DELIMITER;
			}
			
			pszNextLine = pszNextLine + f_strlen( pszNextLine) + 1;
		}
	}
	rc = f_filecpy( pszFile, newContents.getTEXT());
	
Exit:

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	if ( pszContents)
	{
		f_free( &pszContents);
	}
	return rc; 
}

RCODE utilReadProperty(
	char *			pszFile,
	char *			pszProp,
	FlmStringAcc *	pAcc)
{
	RCODE					rc = NE_XFLM_OK;
	char *				pszContents = NULL;
	char *				pszValue = NULL;
	IF_FileSystem *	pFileSystem = NULL;
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}
	
	if ( RC_BAD( pFileSystem->doesFileExist( pszFile)))
	{
		//be nice here.  simply don't append anything into FlmStringAcc
		goto Exit;
	}
	if ( RC_BAD( fileToString( pszFile, &pszContents)))
	{
		goto Exit;
	}
	TEST_RC( rc = propertyExists( pszProp, pszContents, &pszValue));
	TEST_RC( rc = pAcc->appendTEXT( (FLMBYTE *)pszValue));

Exit:

	if( pFileSystem)
	{
		pFileSystem->Release();
	}

	if( pszValue)
	{
		f_free( &pszValue);
	}

	if( pszContents)
	{
		f_free( &pszContents);
	}
	
	return( rc); 
}

void scramble( 
	IF_RandomGenerator * pRandGen, 
	FLMUINT *				puiArray, 
	FLMUINT					uiNumElems)
{
	FLMUINT		uiLoop;
	FLMUINT		uiTmp;
	FLMUINT		uiIndex;
	
	for( uiLoop = 0; uiLoop < uiNumElems; uiLoop++)
	{
		uiIndex = pRandGen->getUINT32( 0, uiNumElems - 1);
		f_swap( 
			puiArray[uiLoop], 
			puiArray[uiIndex],
			uiTmp); 
	}
}

/****************************************************************************
Desc: Initialize and set the title
****************************************************************************/
void WpsInit(
	FLMUINT			uiRows,				// 0xFFFF means use current screen height.
	FLMUINT			uiCols,				// 0xFFFF means use current screen width.
	const char *	pszScreenTitle)
{
	char	szTitleAndVer[ 100];

	if( gv_bInitialized)
	{
		return;
	}

	// Setup utilities title which includes the software version.
#ifdef SECURE_UTIL
	f_sprintf( (char *)szTitleAndVer, "%s - %s (%u)",
						pszScreenTitle, SRC_VER_STR, (unsigned)UTIL_VER);
#else
	f_sprintf( (char *)szTitleAndVer, "%s - %s (UNSECURE:%u)",
						pszScreenTitle, SRC_VER_STR, (unsigned)UTIL_VER);
#endif

	FTXInit( szTitleAndVer, uiCols, uiRows, FLM_BLACK, FLM_LIGHTGRAY,
		NULL, NULL);

	if( RC_BAD( f_mutexCreate( &gv_hDispMutex)))
	{
		flmAssert( 0);
	}

	WpsThrdInit( szTitleAndVer);
	gv_bInitialized = TRUE;
}


/****************************************************************************
Desc: Initialize WPS using an existing FTX environment
****************************************************************************/
void WpsInitFTX( void)
{
	if( gv_bInitialized)
	{
		return;
	}

	if( RC_BAD( f_mutexCreate( &gv_hDispMutex)))
	{
		flmAssert( 0);
	}

	gv_bInitialized = TRUE;
}


/****************************************************************************
Desc: Restores the screen to an initial state
****************************************************************************/
void WpsExit( void)
{
	if( !gv_bInitialized)
	{
		return;
	}

	gv_bShutdown = TRUE;
	WpsThrdExit();
	f_mutexDestroy( &gv_hDispMutex);
	FTXExit();
	gv_bInitialized = FALSE;
}

/****************************************************************************
Desc: Initialize and set the title of a thread's screen
***************************************************************************/
void WpsThrdInitUsingScreen(
	FTX_SCREEN *	pFtxScreen,
	const char *	pszScreenTitle)
{
	FLMUINT			uiRows;
	FLMUINT			uiThrdId;
	WPSSCREEN *		pCurScreen = NULL;

	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();
	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen == NULL)
	{
		if( RC_BAD( f_calloc( sizeof( WPSSCREEN), &pCurScreen)))
		{
			flmAssert( 0);
		}
		pCurScreen->uiScreenId = uiThrdId;
		pCurScreen->pNext = gv_pScreenList;
		gv_pScreenList = pCurScreen;

		if( pFtxScreen != NULL)
		{
			pCurScreen->pScreen = pFtxScreen;
			pCurScreen->bPrivate = FALSE;
		}
		else
		{
			if( RC_BAD( FTXScreenInit( pszScreenTitle,
				&(pCurScreen->pScreen))))
			{
				flmAssert( 0);
			}
			pCurScreen->bPrivate = TRUE;
		}

		if( RC_BAD( FTXScreenGetSize( pCurScreen->pScreen, NULL, &uiRows)))
		{
			flmAssert( 0);
		}

		if( RC_BAD( FTXWinInit( pCurScreen->pScreen, 0,
			(FLMBYTE)(uiRows - 1), &(pCurScreen->pWin))))
		{
			flmAssert( 0);
		}

		FTXWinMove( pCurScreen->pWin, 0, 1);

		if( RC_BAD( FTXWinInit( pCurScreen->pScreen, 0,
			1, &(pCurScreen->pTitleWin))))
		{
			flmAssert( 0);
		}

		FTXWinPaintBackground( pCurScreen->pTitleWin, FLM_RED);
		FTXWinPrintStr( pCurScreen->pTitleWin, pszScreenTitle);
		FTXWinOpen( pCurScreen->pTitleWin);
		FTXWinOpen( pCurScreen->pWin);
	}

	wpsUnlock( &gv_hDispMutex);
}


/****************************************************************************
Desc: Frees all screen resources allocated to a thread
Ret:
****************************************************************************/
void WpsThrdExit( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN *		pPrevScreen = NULL;
	WPSSCREEN *		pCurScreen = NULL;


	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();
	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pPrevScreen = pCurScreen;
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen != NULL)
	{
		if( pCurScreen == gv_pScreenList)
		{
			gv_pScreenList = pCurScreen->pNext;
		}

		if( pPrevScreen != NULL)
		{
			pPrevScreen->pNext = pCurScreen->pNext;
		}

		if( pCurScreen->bPrivate == TRUE)
		{
			 FTXScreenFree( &(pCurScreen->pScreen));
		}
		else
		{
			FTXWinFree( &(pCurScreen->pTitleWin));
			FTXWinFree( &(pCurScreen->pWin));
		}

		f_free( &pCurScreen);
	}

	wpsUnlock( &gv_hDispMutex);
}


/****************************************************************************
Desc: Returns the size of the screen in columns and rows.
****************************************************************************/
void WpsScrSize(
	FLMUINT *	puiNumColsRV,
	FLMUINT *	puiNumRowsRV
	)
{
	FTXWinGetCanvasSize( wpsGetThrdWin(), puiNumColsRV, puiNumRowsRV);
}


/****************************************************************************
Desc:	Output a string at present cursor location.
****************************************************************************/
void WpsStrOut(
	const char *	pszString)
{
	FTXWinPrintStr( wpsGetThrdWin(), pszString);
	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}

/****************************************************************************
Desc:	Output a formatted string at present cursor location.
****************************************************************************/
void WpsPrintf(
	const char *	pszFormat, ...)
{
	char			szBuffer[ 512];
	f_va_list	args;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);
	FTXWinPrintStr( wpsGetThrdWin(), szBuffer);

	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}


/****************************************************************************
Desc:	Output a formatted string at present cursor location with color
****************************************************************************/
void WpsCPrintf(
	eColorType			uiBack,
	eColorType			uiFore,
	const char *		pszFormat, ...)
{
	char				szBuffer[ 512];
	f_va_list		args;
	eColorType		uiOldBack;
	eColorType		uiOldFore;

	f_va_start( args, pszFormat);
	f_vsprintf( szBuffer, pszFormat, &args);
	f_va_end( args);

	FTXWinGetBackFore( wpsGetThrdWin(), &uiOldBack, &uiOldFore);
	FTXWinSetBackFore( wpsGetThrdWin(), uiBack, uiFore);
	FTXWinPrintStr( wpsGetThrdWin(), szBuffer);
	FTXWinSetBackFore( wpsGetThrdWin(), uiOldBack, uiOldFore);

	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}


/****************************************************************************
Desc: Output a character to the screen at the current location. If char is
		a LineFeed then a CarriageReturn will be inserted before the LineFeed.
Notes:On NLM becomes a blocking function if the char is the newline character.
****************************************************************************/
void WpsChrOut(
	char	chr
	)
{
	FTXWinPrintChar( wpsGetThrdWin(), (FLMUINT)chr);
	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}


/****************************************************************************
Desc:    Clear the screen from the col/row down
Notes:   If col==row==0 then clear entire screen
****************************************************************************/
void WpsScrClr(
	FLMUINT	uiCol,
	FLMUINT	uiRow
	)
{
	FTX_WINDOW *	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;


	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinClearXY( pThrdWin, uiCol, uiRow);
	FTXWinSetCursorPos( pThrdWin, uiCol, uiRow);
	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}

/****************************************************************************
Desc:    Position to the column and row specified.
Notes:   The NLM could call GetPositionOfOutputCursor(&r,&c);
****************************************************************************/
void WpsScrPos(
	FLMUINT	uiCol,
	FLMUINT	uiRow
	)
{
	FTX_WINDOW *	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;


	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinSetCursorPos( pThrdWin, uiCol, uiRow);
}


/****************************************************************************
Desc:    Clear from input cursor to end of line
****************************************************************************/
void WpsLineClr(
	FLMUINT	uiCol,
	FLMUINT	uiRow
	)
{
	FTX_WINDOW *	pThrdWin;
	FLMUINT			uiCurrCol;
	FLMUINT			uiCurrRow;


	pThrdWin = wpsGetThrdWin();
	FTXWinGetCursorPos( pThrdWin, &uiCurrCol, &uiCurrRow);

	if( uiCol == 255)
	{
		uiCol = uiCurrCol;
	}

	if( uiRow == 255)
	{
		uiRow = uiCurrRow;
	}

	FTXWinClearLine( pThrdWin, uiCol, uiRow);
	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}

/****************************************************************************
Desc:    Edit a line of data like gets(s).  Newline replaced by NULL character.
Ret:     WPK Character
Notes:   Does not support WP extended character input - but could easily!
****************************************************************************/
FLMUINT WpsLineEd(
	char *		pszString,
	FLMUINT		uiMaxLen,
	FLMBOOL *	pbShutdown
	)
{
	FLMUINT		uiCharCount;
	FLMUINT		uiCursorType;


	uiCursorType = FTXWinGetCursorType( wpsGetThrdWin());
	FTXWinSetCursorType( wpsGetThrdWin(), FLM_CURSOR_UNDERLINE);
	FTXSetShutdownFlag( pbShutdown);
	uiCharCount = FTXLineEd( wpsGetThrdWin(), pszString, uiMaxLen);
	FTXSetShutdownFlag( NULL);
	FTXWinSetCursorType( wpsGetThrdWin(), uiCursorType);

	return( uiCharCount);
}


/****************************************************************************
Desc:    Sets the FTX shutdown flag pointer
Ret:
****************************************************************************/
void WpsSetShutdown(
	FLMBOOL *    pbShutdown
	)
{
	FTXSetShutdownFlag( pbShutdown);
}


/****************************************************************************
Desc:    Edit a line of data with advanced features.
Ret:     Number of characters input.
****************************************************************************/
FLMUINT WpsLineEditExt(
	char *		pszBuffer,
	FLMUINT		uiBufSize,
	FLMUINT		uiMaxWidth,
	FLMBOOL *	pbShutdown,
	FLMUINT *	puiTermChar
	)
{
	FLMUINT		uiCharCount = 0;
	FLMUINT		uiCursorType;


	uiCursorType = FTXWinGetCursorType( wpsGetThrdWin());
	FTXWinSetCursorType( wpsGetThrdWin(), FLM_CURSOR_UNDERLINE);
	FTXSetShutdownFlag( pbShutdown);
	FTXLineEdit( wpsGetThrdWin(), pszBuffer, uiBufSize, uiMaxWidth,
		&uiCharCount, puiTermChar);
	FTXSetShutdownFlag( NULL);
	FTXWinSetCursorType( wpsGetThrdWin(), uiCursorType);

	return( (FLMINT)uiCharCount);
}


/****************************************************************************
Desc:	Get the current X coordinate of the cursor
****************************************************************************/
FLMUINT WpsCurrCol( void)
{
	FLMUINT		uiCol;

	FTXWinGetCursorPos( wpsGetThrdWin(), &uiCol, NULL);
	return( uiCol);
}


/****************************************************************************
Desc:	Get the current Y coordinate of the cursor
****************************************************************************/
FLMUINT WpsCurrRow( void)
{
	FLMUINT		uiRow;

	FTXWinGetCursorPos( wpsGetThrdWin(), NULL, &uiRow);
	return( uiRow);
}

/****************************************************************************
Desc:    Set the background and foreground colors
Ret:     None
****************************************************************************/
void WpsScrBackFor(
	eColorType	backColor,
	eColorType	foreColor)
{
	FTXWinSetBackFore( wpsGetThrdWin(), backColor, foreColor);
}


/****************************************************************************
Desc : Sets the cursor attributes.
****************************************************************************/
void WpsCursorSetType(
	FLMUINT		uiType)
{
	FTXWinSetCursorType( wpsGetThrdWin(), uiType);
	FTXRefresh();
}

/****************************************************************************
Desc:    Specifies that display performance (throughput) should be
			optimal.
****************************************************************************/
void WpsOptimize( void)
{
	gv_bOptimize = TRUE;
}


/****************************************************************************
Desc: Draws a border around the current thread's screen
Ret:  none
****************************************************************************/
void WpsDrawBorder( void)
{
	FTXWinDrawBorder( wpsGetThrdWin());
	if( !gv_bOptimize)
	{
		FTXRefresh();
	}
}


/****************************************************************************
Desc:    Convert keyboard sequences/scan codes to WPK key strokes.
Notes:   Does not support WP extended character input - but could easily!
****************************************************************************/
FLMUINT WpkIncar( void)
{
	FLMUINT		uiChar;

	FTXWinInputChar( wpsGetThrdWin(), &uiChar);
	return( uiChar);
}


/****************************************************************************
Desc:    Convert keyboard sequences/scan codes to WPK key strokes.  This
			routine accepts a pointer to a shutdown flag.
****************************************************************************/
FLMUINT WpkGetChar(
	FLMBOOL *		pbShutdown
	)
{
	FLMUINT		uiChar;

	FTXSetShutdownFlag( pbShutdown);
	FTXWinInputChar( wpsGetThrdWin(), &uiChar);
	FTXSetShutdownFlag( NULL);

	return( uiChar);
}


/****************************************************************************
Desc:    Tests the keyboard for a pending character
Ret:		1 if key available, 0 if no key available
****************************************************************************/
FLMUINT WpkTestKB( void)
{
	FLMUINT		uiCharAvail;

	uiCharAvail = (FLMUINT)(FTXWinTestKB( wpsGetThrdWin()) ==
		NE_FLM_OK ? 1 : 0);
	return( uiCharAvail);
}


/****************************************************************************
Desc:		Returns a pointer to a thread's screen
****************************************************************************/
FTX_SCREEN * WpsGetThrdScreen( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN *		pCurScreen = NULL;

	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();

	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen == NULL)
	{
		flmAssert( 0);
	}

	wpsUnlock( &gv_hDispMutex);
	return( pCurScreen->pScreen);
}

/****************************************************************************
Desc:		Returns a pointer to a thread's screen
****************************************************************************/
FSTATIC FTX_WINDOW * wpsGetThrdWin( void)
{
	FLMUINT			uiThrdId;
	WPSSCREEN *		pCurScreen = NULL;


	wpsLock( &gv_hDispMutex);

	uiThrdId = f_threadId();
	pCurScreen = gv_pScreenList;
	while( pCurScreen != NULL)
	{
		if( pCurScreen->uiScreenId == uiThrdId)
		{
			break;
		}
		pCurScreen = pCurScreen->pNext;
	}

	if( pCurScreen == NULL)
	{
		flmAssert( 0);
	}

	wpsUnlock( &gv_hDispMutex);
	return( pCurScreen->pWin);
}
