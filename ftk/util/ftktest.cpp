//-------------------------------------------------------------------------
// Desc:	Basic unit test.
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id: $
//-------------------------------------------------------------------------

#include "ftk.h"

#define F_ATOM_TEST_THREADS		64
#define F_ATOM_TEST_ITERATIONS	100000

FSTATIC RCODE ftkTestAtomics( void);

FSTATIC RCODE FLMAPI ftkAtomicIncThread(
	IF_Thread *		pThread);
	
FSTATIC RCODE FLMAPI ftkAtomicDecThread(
	IF_Thread *		pThread);
	
FSTATIC RCODE FLMAPI ftkAtomicIncDecThread(
	IF_Thread *		pThread);
	
FSTATIC RCODE FLMAPI ftkAtomicExchangeThread(
	IF_Thread *		pThread);
	
FSTATIC RCODE ftkFastChecksumTest( void);

FSTATIC RCODE ftkPacketChecksumTest( void);

FSTATIC FLMBYTE ftkSlowPacketChecksum(
	const FLMBYTE *	pucPacket,
	FLMUINT				uiBytesToChecksum);
	
FSTATIC FLMATOMIC						gv_refCount;
FSTATIC FLMATOMIC						gv_spinLock;
	
/****************************************************************************
Desc:
****************************************************************************/
#ifdef FLM_RING_ZERO_NLM
extern "C" int nlm_main( void)
#else
int main( void)
#endif
{
	RCODE					rc = NE_FLM_OK;
	IF_DirHdl *			pDirHdl = NULL;
	IF_FileSystem *	pFileSystem = NULL;
	IF_BlockMgr *		pBlockMgr = NULL;
	IF_BTree *			pBTree = NULL;
	FLMUINT32			ui32RootBlkId;
	char					szTmpBuf[ 128];
	
	if( RC_BAD( rc = ftkStartup()))
	{
		goto Exit;
	}
	
	// Run some simple tests
	
	if( RC_BAD( rc = FlmGetFileSystem( &pFileSystem)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pFileSystem->openDir( ".", "*.*", &pDirHdl)))
	{
		goto Exit;
	}
	
	while( RC_OK( pDirHdl->next()))
	{
		f_printf( "%s\n", pDirHdl->currentItemName());
	}
	
	pDirHdl->Release();
	pDirHdl = NULL;
	
	if( RC_BAD( rc = FlmAllocBlockMgr( 4096, &pBlockMgr)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocBTree( pBlockMgr, &pBTree)))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = pBTree->btCreate( 1, FALSE, TRUE, &ui32RootBlkId)))
	{
		goto Exit;
	}
	
	pBTree->btDeleteTree();
	pBTree->Release();

	f_printf( "Running sprintf test: ");
	f_sprintf( szTmpBuf, "Hello, World! (You're number %u)\n", 1);
	f_printf( szTmpBuf);
	
	// Run a multi-threaded test to verify the proper operation of
	// the atomic operations
	
	if( RC_BAD( rc = ftkTestAtomics()))
	{
		goto Exit;
	}
	
	// Test the checksum routines
	
	if( RC_BAD( rc = ftkFastChecksumTest()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = ftkPacketChecksumTest()))
	{
		goto Exit;
	}
	
Exit:

	if( pDirHdl)
	{
		pDirHdl->Release();
	}
	
	if( pFileSystem)
	{
		pFileSystem->Release();
	}
	
	if( pBlockMgr)
	{
		pBlockMgr->Release();
	}

	ftkShutdown();
	return( (int)rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE ftkTestAtomics( void)
{
	RCODE					rc = NE_FLM_OK;
	IF_Thread *			pThreadList[ F_ATOM_TEST_THREADS];
	FLMUINT				uiLoop;
	
	gv_refCount = 0;
	f_memset( pThreadList, 0, sizeof( IF_Thread *) * F_ATOM_TEST_THREADS);

	f_printf( "Creating atomic increment threads: ");
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		if( RC_BAD( rc = 	f_threadCreate( &pThreadList[ uiLoop], 
			ftkAtomicIncThread)))
		{
			goto Exit;
		}
	}
	
	f_printf( "%u\n", uiLoop);	
		
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		pThreadList[ uiLoop]->waitToComplete();
		f_threadDestroy( &pThreadList[ uiLoop]);
	}
	
	if( gv_refCount != F_ATOM_TEST_THREADS * F_ATOM_TEST_ITERATIONS)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}

	f_printf( "Creating atomic decrement threads: ");
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		if( RC_BAD( rc = 	f_threadCreate( &pThreadList[ uiLoop], 
			ftkAtomicDecThread)))
		{
			goto Exit;
		}
	}
	
	f_printf( "%u\n", uiLoop);	
		
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		pThreadList[ uiLoop]->waitToComplete();
		f_threadDestroy( &pThreadList[ uiLoop]);
	}
	
	if( gv_refCount != 0)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
	f_printf( "Creating atomic inc/dec threads: ");
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		if( RC_BAD( rc = 	f_threadCreate( &pThreadList[ uiLoop], 
			ftkAtomicIncDecThread)))
		{
			goto Exit;
		}
	}
	
	f_printf( "%u\n", uiLoop);	
		
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		pThreadList[ uiLoop]->waitToComplete();
		f_threadDestroy( &pThreadList[ uiLoop]);
	}
	
	if( gv_refCount != 0)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
	gv_spinLock = 0;
	
	f_printf( "Creating atomic exchange threads: ");
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		if( RC_BAD( rc = 	f_threadCreate( &pThreadList[ uiLoop], 
			ftkAtomicExchangeThread)))
		{
			goto Exit;
		}
	}
	
	f_printf( "%u\n", uiLoop);	
		
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		pThreadList[ uiLoop]->waitToComplete();
		f_threadDestroy( &pThreadList[ uiLoop]);
	}
	
	if( gv_refCount != F_ATOM_TEST_THREADS * F_ATOM_TEST_ITERATIONS)
	{
		rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
		goto Exit;
	}
	
Exit:

	for( uiLoop = 0; uiLoop < F_ATOM_TEST_THREADS; uiLoop++)
	{
		if( pThreadList[ uiLoop])
		{
			f_threadDestroy( &pThreadList[ uiLoop]);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE FLMAPI ftkAtomicIncThread(
	IF_Thread *		pThread)
{
	FLMUINT		uiLoop;
	
	F_UNREFERENCED_PARM( pThread);
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_ITERATIONS; uiLoop++)
	{
		f_atomicInc( &gv_refCount);
		if( (uiLoop % 128) == 0)
		{
			f_yieldCPU();
		}
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE FLMAPI ftkAtomicDecThread(
	IF_Thread *		pThread)
{
	FLMUINT		uiLoop;
	
	F_UNREFERENCED_PARM( pThread);
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_ITERATIONS; uiLoop++)
	{
		f_atomicDec( &gv_refCount);
		if( (uiLoop % 128) == 0)
		{
			f_yieldCPU();
		}
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE FLMAPI ftkAtomicIncDecThread(
	IF_Thread *		pThread)
{
	FLMUINT		uiLoop;
	
	F_UNREFERENCED_PARM( pThread);
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_ITERATIONS; uiLoop++)
	{
		f_atomicInc( &gv_refCount);
		
		if( (uiLoop % 128) == 0)
		{
			f_yieldCPU();
		}
		
		f_atomicDec( &gv_refCount);
	}
	
	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE FLMAPI ftkAtomicExchangeThread(
	IF_Thread *		pThread)
{
	FLMUINT		uiLoop;
	FLMATOMIC	uiTmp;
	
	F_UNREFERENCED_PARM( pThread);
	
	for( uiLoop = 0; uiLoop < F_ATOM_TEST_ITERATIONS; uiLoop++)
	{
		while( f_atomicExchange( &gv_spinLock, 1) == 1)
		{
			f_yieldCPU();
		}
		
		uiTmp = gv_refCount + 1;
		
		if( (uiLoop % 128) == 0)
		{
			f_yieldCPU();
		}
		
		gv_refCount = uiTmp;
		
		f_atomicExchange( &gv_spinLock, 0);
	}
	
	return( NE_FLM_OK);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC RCODE ftkFastChecksumTest( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiSlowAdds = 0;
	FLMUINT			uiSlowXORs = 0;
	FLMUINT			uiFastAdds = 0;
	FLMUINT			uiFastXORs = 0;
	FLMUINT			uiDataLength;
	FLMBYTE *		pucData = NULL;
	FLMBYTE *		pucCur;
	FLMBYTE *		pucEnd;
	FLMUINT			uiSlowChecksum = 0;
	FLMUINT			uiFastChecksum = 0;
	FLMUINT			uiLoop;
	FLMUINT			uiIter;
	FLMUINT			uiPass;
	FLMUINT			uiStartTime;
	FLMUINT			uiSlowTime = 0;
	FLMUINT			uiFastTime = 0;
	
	f_printf( "Running checksum tests ... ");
	
	uiDataLength = 8192;
	if( RC_BAD( rc = f_alloc( uiDataLength, &pucData)))
	{
		goto Exit;
	}
	
	for( uiIter = 0; uiIter < 1000; uiIter++)
	{
		for( uiLoop = 0; uiLoop < uiDataLength; uiLoop++)
		{
			pucData[ uiLoop] = f_getRandomByte();
		}
		
		uiStartTime = FLM_GET_TIMER();
		
		for( uiPass = 0; uiPass < 100; uiPass++)
		{
			uiSlowAdds = 0;
			uiSlowXORs = 0;
	
			pucCur = pucData;
			pucEnd = pucData + uiDataLength;
		
			while( pucCur < pucEnd)	
			{
				uiSlowAdds += *pucCur;
				uiSlowXORs ^= *pucCur++;
			}
		
			uiSlowAdds &= 0xFF;
			uiSlowChecksum = (FLMUINT32)((uiSlowAdds << 16) + uiSlowXORs);
		}
		
		uiSlowTime += FLM_ELAPSED_TIME( FLM_GET_TIMER(), uiStartTime); 
		
		uiStartTime = FLM_GET_TIMER();
		
		for( uiPass = 0; uiPass < 100; uiPass++)
		{
			uiFastAdds = 0;
			uiFastXORs = 0;
	
			uiFastChecksum = f_calcFastChecksum( pucData, 
										uiDataLength, &uiFastAdds, &uiFastXORs);
		}
		
		uiFastTime += FLM_ELAPSED_TIME( FLM_GET_TIMER(), uiStartTime); 
	
		if( (uiSlowAdds != uiFastAdds) || 
			 (uiSlowXORs != uiFastXORs) || 
			 (uiSlowChecksum != uiFastChecksum))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
			goto Exit;
		}
	}
	
	f_printf( "Slow time = %u ms, FastTime = %u ms. ", 
		(unsigned)FLM_TIMER_UNITS_TO_MILLI( uiSlowTime), 
		(unsigned)FLM_TIMER_UNITS_TO_MILLI( uiFastTime));
	
Exit:

	f_printf( "done.\n");
	
	if( pucData)
	{
		f_free( &pucData);
	}
	
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC RCODE ftkPacketChecksumTest( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiDataLength;
	FLMBYTE *		pucData = NULL;
	FLMUINT			uiSlowChecksum = 0;
	FLMUINT			uiFastChecksum = 0;
	FLMUINT			uiLoop;
	FLMUINT			uiIter;
	FLMUINT			uiPass;
	FLMUINT			uiStartTime;
	FLMUINT			uiSlowTime = 0;
	FLMUINT			uiFastTime = 0;
	
	f_printf( "Running checksum tests ... ");
	
	uiDataLength = 8192;
	if( RC_BAD( rc = f_alloc( uiDataLength, &pucData)))
	{
		goto Exit;
	}
	
	for( uiIter = 0; uiIter < 1000; uiIter++)
	{
		for( uiLoop = 0; uiLoop < uiDataLength; uiLoop++)
		{
			pucData[ uiLoop] = f_getRandomByte();
		}
		
		uiStartTime = FLM_GET_TIMER();
		for( uiPass = 0; uiPass < 100; uiPass++)
		{
			uiSlowChecksum = ftkSlowPacketChecksum( pucData, uiDataLength);
		}
		uiSlowTime += FLM_ELAPSED_TIME( FLM_GET_TIMER(), uiStartTime); 
		
		uiStartTime = FLM_GET_TIMER();
		for( uiPass = 0; uiPass < 100; uiPass++)
		{
			uiFastChecksum = f_calcPacketChecksum( pucData, uiDataLength); 
		}
		uiFastTime += FLM_ELAPSED_TIME( FLM_GET_TIMER(), uiStartTime); 
	
		if( uiSlowChecksum != uiFastChecksum)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_FAILURE);
			goto Exit;
		}
	}
	
	f_printf( "Slow time = %u ms, FastTime = %u ms. ", 
		(unsigned)FLM_TIMER_UNITS_TO_MILLI( uiSlowTime), 
		(unsigned)FLM_TIMER_UNITS_TO_MILLI( uiFastTime));
	
Exit:

	f_printf( "done.\n");
	
	if( pucData)
	{
		f_free( &pucData);
	}
	
	return( rc);
}

/********************************************************************
Desc:
*********************************************************************/
FSTATIC FLMBYTE ftkSlowPacketChecksum(
	const FLMBYTE *	pucPacket,
	FLMUINT				uiBytesToChecksum)
{
	FLMUINT				uiChecksum = 0;
	FLMBYTE				ucTmp;
	const FLMBYTE *	pucEnd;
	const FLMBYTE *	pucSectionEnd;
	const FLMBYTE *	pucCur;

	// Checksum is calculated for every byte in the packet that comes
	// after the checksum byte.

	pucCur = pucPacket;
	pucEnd = pucPacket + uiBytesToChecksum;

#ifdef FLM_64BIT
	pucSectionEnd = pucPacket + (sizeof( FLMUINT) - ((FLMUINT)pucPacket & 0x7));
#else
	pucSectionEnd = pucPacket + (sizeof( FLMUINT) - ((FLMUINT)pucPacket & 0x3));
#endif

	flmAssert( pucSectionEnd >= pucPacket);

	if (pucSectionEnd > pucEnd)
	{
		pucSectionEnd = pucEnd;
	}

	while (pucCur < pucSectionEnd)
	{
		uiChecksum = (uiChecksum << 8) + *pucCur++;
	}

#ifdef FLM_64BIT
	pucSectionEnd = (FLMBYTE *)((FLMUINT)pucEnd & 0xFFFFFFFFFFFFFFF8); 
#else
	pucSectionEnd = (FLMBYTE *)((FLMUINT)pucEnd & 0xFFFFFFFC); 
#endif

	while (pucCur < pucSectionEnd)
	{
		uiChecksum ^= *((FLMUINT *) pucCur);
		pucCur += sizeof(FLMUINT);
	}

	while (pucCur < pucEnd)
	{
		uiChecksum ^= *pucCur++;
	}

	ucTmp = (FLMBYTE) uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE) uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE) uiChecksum;

#ifdef FLM_64BIT
	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;
#endif

	ucTmp ^= (FLMBYTE) (uiChecksum >> 8);
	uiChecksum = ucTmp;

	if ((uiChecksum = ucTmp) == 0)
	{
		uiChecksum = 1;
	}

	return ((FLMBYTE) uiChecksum);
}
