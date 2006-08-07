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
#define F_ATOM_TEST_ITERATIONS	1000000

FSTATIC RCODE ftkTestAtomics( void);

FSTATIC RCODE FLMAPI ftkAtomicIncThread(
	IF_Thread *		pThread);
	
FSTATIC RCODE FLMAPI ftkAtomicDecThread(
	IF_Thread *		pThread);
	
FSTATIC FLMATOMIC						gv_refCount;
	
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

	f_sprintf( szTmpBuf, "Hello, World! (You're number %u)\n", 1);
	f_printf( szTmpBuf);
	
	// Run a multi-threaded test to verify the proper operation of
	// the atomic operations
	
	if( RC_BAD( rc = ftkTestAtomics()))
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
FSTATIC RCODE ftkTestAtomics( void)
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
		if( (uiLoop % 64) == 0)
		{
			f_sleep( 0);
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
		if( (uiLoop % 8) == 0)
		{
			f_sleep( 0);
		}
	}
	
	return( NE_FLM_OK);
}
