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
