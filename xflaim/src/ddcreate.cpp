//------------------------------------------------------------------------------
// Desc:	Routines to service creation of a database dictionary.
//
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ddcreate.cpp 3111 2006-01-19 13:10:50 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

// Internal Static Routines

/****************************************************************************
Desc:	Read in LFH headers.
****************************************************************************/
RCODE F_Db::dictReadLFH( void)
{
	RCODE					rc = NE_XFLM_OK;
	LFILE *				pLFile;
	F_COLLECTION *		pCollection;
	F_CachedBlock *	pSCache;
	FLMBOOL				bReleaseCache = FALSE;
	F_BLK_HDR *			pBlkHdr;
	FLMUINT				uiBlkAddress;
	FLMUINT				uiPos;
	FLMUINT				uiEndPos;
	FLMUINT				uiBlkSize = m_pDatabase->m_uiBlockSize;
	LFILE					TmpLFile;
	F_COLLECTION		TmpCollection;

	f_memset( &TmpLFile, 0, sizeof( LFILE));
	f_memset( &TmpCollection, 0, sizeof( F_COLLECTION));

	uiBlkAddress =
			(FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32FirstLFBlkAddr;
	while (uiBlkAddress)
	{
		if (RC_BAD( rc = m_pDatabase->getBlock( this, NULL, 
			uiBlkAddress, NULL, &pSCache)))
		{
			goto Exit;
		}
		bReleaseCache = TRUE;

		pBlkHdr = pSCache->m_pBlkHdr;
		uiPos = SIZEOF_STD_BLK_HDR;
		uiEndPos = blkGetEnd( uiBlkSize, SIZEOF_STD_BLK_HDR, pBlkHdr);

		// Read through all of the logical file definitions in the block

		for( ; uiPos + sizeof( F_LF_HDR) <= uiEndPos; uiPos += sizeof( F_LF_HDR))
		{
			F_LF_HDR *	pLfHdr = (F_LF_HDR *)((FLMBYTE *)(pBlkHdr) + uiPos);
			eLFileType	eLfType = (eLFileType)pLfHdr->ui32LfType;

			// Have to fix up the offsets later when they are read in

			if (eLfType == XFLM_LF_INVALID)
			{
				continue;
			}

			// Populate the LFILE in the dictionary, if one has been set up.

			if (eLfType == XFLM_LF_INDEX)
			{
				FSLFileIn( (FLMBYTE *)pLfHdr,
					&TmpLFile, NULL, uiBlkAddress, uiPos);

				if (RC_OK( m_pDict->getIndex( TmpLFile.uiLfNum, &pLFile,
											NULL, TRUE)))
				{
					f_memcpy( pLFile, &TmpLFile, sizeof( LFILE));
				}

				// LFILE better have a non-zero root block.

				if (!TmpLFile.uiRootBlk)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
			else
			{

				// Better be a container

				flmAssert( eLfType == XFLM_LF_COLLECTION);

				FSLFileIn( (FLMBYTE *)pLfHdr,
					&TmpCollection.lfInfo, &TmpCollection, uiBlkAddress, uiPos);

				if (RC_OK( m_pDict->getCollection( TmpCollection.lfInfo.uiLfNum,
												&pCollection, TRUE)))
				{
					f_memcpy( pCollection, &TmpCollection, sizeof( F_COLLECTION));
				}

				// LFILE better have a non-zero root block.

				if (!TmpCollection.lfInfo.uiRootBlk)
				{
					rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
					goto Exit;
				}
			}
		}

		// Get the next block in the chain

		uiBlkAddress = (FLMUINT)pBlkHdr->ui32NextBlkInChain;
		ScaReleaseCache( pSCache, FALSE);
		bReleaseCache = FALSE;
	}

Exit:

	if (bReleaseCache)
	{
		ScaReleaseCache( pSCache, FALSE);
	}

	return( rc );
}

/****************************************************************************
Desc:	Read in all element, attribute, index, or collection definitions - as
		specified in uiDictType.
****************************************************************************/
RCODE F_Db::dictReadDefs(
	FLMUINT	uiDictType)
{
	RCODE				rc = NE_XFLM_OK;
	F_DataVector	key;
	LFILE *			pLFile;
	IXD *				pIxd;
	F_Btree *		pbTree = NULL;
	FLMBYTE			ucKeyBuf [MAX_KEY_SIZ];
	FLMUINT			uiKeyLen;
	FLMUINT			uiFoundDictType;
	FLMUINT			uiLowest;
	FLMUINT			uiHighest;
	FLMUINT			uiDictNum;
	IXKeyCompare	compareObject;

	if (RC_BAD( rc = m_pDict->getIndex( XFLM_DICT_NUMBER_INDEX, &pLFile, &pIxd)))
	{
		RC_UNEXPECTED_ASSERT( rc);
		goto Exit;
	}

	// First determine the low and high field numbers.

	// If the LFILE is not yet set up, the index has not yet been
	// created, so there will be no definitions to read.  This will
	// be the case when we are first creating the dictionary.  We have
	// started a transaction, and it is trying to read in the definitions
	// but there are none.

	flmAssert( pLFile->uiRootBlk);

	// Get a btree

	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpReserveBtree( &pbTree)))
	{
		goto Exit;
	}

	// Open the B-Tree

	compareObject.setIxInfo( this, pIxd);
	compareObject.setCompareNodeIds( FALSE);
	compareObject.setCompareDocId( FALSE);
	compareObject.setSearchKey( &key);
	
	if (RC_BAD( rc = pbTree->btOpen( this, pLFile, FALSE, FALSE,
										&compareObject)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = key.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.outputKey( pIxd, 0,
									ucKeyBuf, sizeof( ucKeyBuf), &uiKeyLen, SEARCH_KEY_FLAG)))
	{
		goto Exit;
	}

	// Position to the first key, if any

	if (RC_BAD( rc = pbTree->btLocateEntry( ucKeyBuf, sizeof( ucKeyBuf),
										&uiKeyLen, XFLM_INCL, NULL)))
	{

		// May not have found anything.

		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	key.reset();

	if (RC_BAD( rc = key.inputKey( pIxd, ucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}

	// See if we went past the last key of this type.

	if (RC_BAD( rc = key.getUINT( 0, &uiFoundDictType)))
	{
		goto Exit;
	}

	if (uiFoundDictType != uiDictType)
	{
		goto Exit;	// Will return NE_XFLM_OK
	}

	if (RC_BAD( rc = key.getUINT( 1, &uiLowest)))
	{
		goto Exit;
	}
	uiHighest = uiLowest;

	// Position to the end of keys of this type

	key.reset();
	if (RC_BAD( rc = key.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.setUINT( 1, 0xFFFFFFFF)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.outputKey( pIxd, 0,
								ucKeyBuf, sizeof( ucKeyBuf), &uiKeyLen, SEARCH_KEY_FLAG)))
	{
		goto Exit;
	}

	// Position to just past the specified key.

	if (RC_BAD( rc = pbTree->btLocateEntry( ucKeyBuf, sizeof( ucKeyBuf),
										&uiKeyLen, XFLM_EXCL, NULL)))
	{

		// May not have found anything, in which case we need to
		// position to the last key in the index.

		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			if (RC_BAD( rc = pbTree->btLastEntry( ucKeyBuf, sizeof( ucKeyBuf),
													&uiKeyLen)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Exit;
		}
	}
	else
	{

		// Backup one key - since we will have gone just beyond
		// keys of this type.

		if (RC_BAD( rc = pbTree->btPrevEntry( ucKeyBuf, sizeof( ucKeyBuf),
											&uiKeyLen)))
		{
			goto Exit;
		}
	}

	// At this point we better be positioned on the last key of this type

	key.reset();

	if (RC_BAD( rc = key.inputKey( pIxd, ucKeyBuf, uiKeyLen)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.getUINT( 0, &uiFoundDictType)))
	{
		goto Exit;
	}

	// See if we went past the last key of this type - should not
	// be possible, unless there is a corruption.

	if (uiFoundDictType != uiDictType)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BTREE_ERROR);
		goto Exit;
	}
	
	if (RC_BAD( rc = key.getUINT( 1, &uiHighest)))
	{
		goto Exit;
	}

	// uiHighest better be >= uiLowest or we have
	// b-tree corruption.

	if (uiHighest < uiLowest)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BTREE_ERROR);
		goto Exit;
	}

	// Pre-allocate the tables

	if (uiDictType == ELM_ELEMENT_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocElementTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_ATTRIBUTE_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocAttributeTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_INDEX_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocIndexTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_PREFIX_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocPrefixTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else if (uiDictType == ELM_ENCDEF_TAG)
	{
		if (RC_BAD( rc = m_pDict->allocEncDefTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}
	else	// (uiDictType == ELM_COLLECTION_TAG)
	{
		flmAssert( uiDictType == ELM_COLLECTION_TAG);

		if (RC_BAD( rc = m_pDict->allocCollectionTable( uiLowest, uiHighest)))
		{
			goto Exit;
		}
	}

	// Position back to the first key for this type

	key.reset();
	if (RC_BAD( rc = key.setUINT( 0, uiDictType)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = key.outputKey( pIxd, 0,
											ucKeyBuf, sizeof( ucKeyBuf),
											&uiKeyLen, SEARCH_KEY_FLAG)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = pbTree->btLocateEntry( ucKeyBuf, sizeof( ucKeyBuf),
										&uiKeyLen, XFLM_INCL, NULL)))
	{

		// May not have found anything.

		if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
		{
			rc = NE_XFLM_OK;
		}
		goto Exit;
	}

	// Loop through all of the keys of this dictionary type

	for (;;)
	{
		key.reset();

		if (RC_BAD( rc = key.inputKey( pIxd, ucKeyBuf, uiKeyLen)))
		{
			goto Exit;
		}

		// See if we went past the last key of this type.

		if (RC_BAD( rc = key.getUINT( 0, &uiFoundDictType)))
		{
			goto Exit;
		}

		if (uiFoundDictType != uiDictType)
		{
			break;
		}

		// Get the dictionary number

		if (RC_BAD( rc = key.getUINT( 1, &uiDictNum)))
		{
			goto Exit;
		}

		// No need to process any more elements or attributes if the
		// dictionary number is in the extended range.

		if ((uiDictType == ELM_ELEMENT_TAG &&
			  uiDictNum >= FLM_LOW_EXT_ELEMENT_NUM) ||
			 (uiDictType == ELM_ATTRIBUTE_TAG &&
			  uiDictNum >= FLM_LOW_EXT_ATTRIBUTE_NUM))
		{
			if (uiDictType == ELM_ELEMENT_TAG)
			{
				m_pDict->m_pNameTable->m_bLoadedAllElements = FALSE;
			}
			else
			{
				m_pDict->m_pNameTable->m_bLoadedAllAttributes = FALSE;
			}
			break;
		}

		if (RC_BAD( rc = m_pDict->updateDict( this,
									uiDictType, key.getDocumentID(), 0,
									TRUE, FALSE)))
		{
			goto Exit;
		}

		// Go to the next key

		if (RC_BAD( rc = pbTree->btNextEntry( ucKeyBuf,
											sizeof( ucKeyBuf),
											&uiKeyLen)))
		{

			// May not have found anything.

			if (rc == NE_XFLM_EOF_HIT || rc == NE_XFLM_NOT_FOUND)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}
	}

Exit:

	if (pbTree)
	{
		gv_XFlmSysData.pBtPool->btpReturnBtree( &pbTree);
	}

	return( rc);
}

/****************************************************************************
Desc:	Open a dictionary by reading in all of the dictionary tables
		from the dictionaries.
****************************************************************************/
RCODE F_Db::dictOpen( void)
{
	RCODE	rc = NE_XFLM_OK;

	// At this point, better not be pointing to a dictionary.

	flmAssert( !m_pDict);

	// Should never get here for a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Allocate a new F_Dict object for reading the dictionary
	// into memory.

	if ((m_pDict = f_new F_Dict) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	// Allocate the name table

	if (RC_BAD( rc = m_pDict->allocNameTable()))
	{
		goto Exit;
	}

	// Add in all of the reserved dictionary tags to the name table.

	if (RC_BAD( rc = m_pDict->getNameTable()->addReservedDictTags()))
	{
		goto Exit;
	}

	// Allocate the fixed collections and indexes and set them up

	if (RC_BAD( rc = m_pDict->setupPredefined(
										m_pDatabase->m_uiDefaultLanguage)))
	{
		goto Exit;
	}

	// Read in the LFH's for the predefined stuff.

	if (RC_BAD( rc = dictReadLFH()))
	{
		goto Exit;
	}

	// If dictionary collection is not yet set up, do nothing.

	if (m_pDict->m_pDictCollection->lfInfo.uiBlkAddress &&
		 m_pDict->m_pDictCollection->lfInfo.uiOffsetInBlk)
	{

		// Read in definitions in the following order:
		// 1) attribute definitions
		// 2) element definitions
		// 3) collection definitions
		// 4) index definitions
		// This guarantees that things will be defined by the
		// time they are referenced.

		if (RC_BAD( rc = dictReadDefs( ELM_ATTRIBUTE_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_ELEMENT_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_COLLECTION_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_INDEX_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_PREFIX_TAG)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = dictReadDefs( ELM_ENCDEF_TAG)))
		{
			goto Exit;
		}

		// Must read LFHs to get the LFILE information for the
		// collections and indexes we have just added.

		if (RC_BAD( rc = dictReadLFH()))
		{
			goto Exit;
		}
	}

	m_pDict->getNameTable()->sortTags();
	
	if (m_pDatabase)
	{
		m_pDict->m_bInLimitedMode = m_pDatabase->inLimitedMode();
	}
	// VISIT:  Should we assume limited mode if we don't have a database file ?

Exit:

	if (RC_BAD( rc) && m_pDict)
	{
		m_pDict->Release();
		m_pDict = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc:		Creates a new dictionary for a database.
			This occurs on database create and on a dictionary change.
****************************************************************************/
RCODE F_Db::createNewDict( void)
{
	RCODE	rc = NE_XFLM_OK;

	// Unlink the DB from the current F_Dict object, if any.

	if (m_pDict)
	{
		m_pDatabase->lockMutex();
		unlinkFromDict();
		m_pDatabase->unlockMutex();
	}

	// Allocate a new F_Dict object for the new dictionary we
	// are going to create.

	if (RC_BAD( rc = dictOpen()))
	{
		goto Exit;
	}

	// Update the F_Db flags to indicate that the dictionary
	// was updated.

	m_uiFlags |= FDB_UPDATED_DICTIONARY;

	// Create a special document in the dictionary to hold
	// the next element, next attribute, next index, and next
	// collection numbers.

	if (RC_BAD( rc = m_pDict->createNextDictNums( this)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add data dictionary records to the data dictionary.
****************************************************************************/
RCODE F_Db::dictCreate(
	const char *	pszDictPath,	// Name of dictionary file.  This is only
											// used if dictBuf is NULL.  If both
											// dictPath and dictBuf are NULL, the
											// database will be created with an empty
											// dictionary
	const char *	pszDictBuf)		// Buffer containing dictionary in ASCII
											// GEDCOM If NULL pszDictPath will be used
{
	RCODE    				rc = NE_XFLM_OK;
	IF_FileHdl *			pDictFileHdl = NULL;
	FLMBOOL					bFileOpen = FALSE;
	LFILE						TempLFile;
	F_COLLECTION			TempCollection;
	char *					pszXMLBuffer = NULL;
	FLMUINT64				ui64FileSize;
	FLMUINT					uiBytesRead;
	F_BufferIStream		stream;

	// This should never be called for a temporary database.

	flmAssert( !m_pDatabase->m_bTempDb);

	// Create the default data collection

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempCollection.lfInfo,
		&TempCollection, XFLM_DATA_COLLECTION, XFLM_LF_COLLECTION, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Create the dictionary collection and indexes

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempCollection.lfInfo,
		&TempCollection,
		XFLM_DICT_COLLECTION, XFLM_LF_COLLECTION, FALSE, TRUE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDatabase->lFileCreate( this,
		&TempLFile, NULL, XFLM_DICT_NUMBER_INDEX, XFLM_LF_INDEX, FALSE, FALSE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = m_pDatabase->lFileCreate( this,
		&TempLFile, NULL, XFLM_DICT_NAME_INDEX, XFLM_LF_INDEX, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Create the maintenance collection

	if (RC_BAD(rc = m_pDatabase->lFileCreate( this, &TempCollection.lfInfo,
		&TempCollection,
		XFLM_MAINT_COLLECTION, XFLM_LF_COLLECTION, FALSE, TRUE)))
	{
		goto Exit;
	}

	// Create a new dictionary we can work with.

	if (RC_BAD( rc = createNewDict()))
	{
		goto Exit;
	}

	// If we have an XML buffer, there is no need to open the file.

	if (!pszDictBuf && pszDictPath)
	{
		if (RC_BAD( rc = gv_pFileSystem->Open(
				pszDictPath, XFLM_IO_RDONLY, &pDictFileHdl)))
		{
			goto Exit;
		}
		bFileOpen = TRUE;

		// Get the file size and allocate a buffer to hold the entire thing.

		if (RC_BAD( rc = pDictFileHdl->Size( &ui64FileSize)))
		{
			goto Exit;
		}

		// Add 1 to size so we can NULL terminate the string we read.

		if (RC_BAD( rc = f_alloc( (FLMUINT)(ui64FileSize + 1), &pszXMLBuffer)))
		{
			goto Exit;
		}

		// Read the entire file into the buffer

		if (RC_BAD( rc = pDictFileHdl->Read( 0, (FLMUINT)ui64FileSize, 
			pszXMLBuffer, &uiBytesRead)))
		{
			goto Exit;
		}
		pszXMLBuffer [uiBytesRead] = 0;
		pszDictBuf = pszXMLBuffer;
	}
	if (!pszDictBuf || !(*pszDictBuf))
	{

		// Neither a dictionary buffer or file were specified.

		goto Exit;
	}

	// Parse through the buffer, extracting each XML document,
	// add to the dictionary and F_Dict object.  The import method
	// reads stuff from the stream, parses it into XML documents,
	// and calls documentDone when the document is complete.
	// The documentDone method checks the dictionary syntax,
	// adds to the dictionary, etc.

	if (RC_BAD( rc = stream.open( (FLMBYTE *)pszDictBuf, 0)))
	{
		goto Exit;
	}

	if (RC_BAD( import( &stream, XFLM_DICT_COLLECTION)))
	{
		goto Exit;
	}

	m_pDict->getNameTable()->sortTags();

Exit:

	if (bFileOpen)
	{
		pDictFileHdl->Close();
	}

	if (pDictFileHdl)
	{
		pDictFileHdl->Release();
	}

	if (pszXMLBuffer)
	{
		f_free( pszXMLBuffer);
	}

	return( rc);
}
