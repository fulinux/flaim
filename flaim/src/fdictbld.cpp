//-------------------------------------------------------------------------
// Desc:	Build dicitionary tables.
// Tabs:	3
//
//		Copyright (c) 1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fdictbld.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC POOL_STATS g_TDictPoolStats = {0,0};


FSTATIC void fdictFixupPointers(
	FDICT *	pNewDict,
	FDICT *	pOldDict);

FSTATIC RCODE fdictReallocAllTables(
	TDICT *			pTDict);

FSTATIC RCODE fdictReallocTbl(
	FLMUINT			uiElementSize,
	FLMUINT			uiTblSize,
	FLMUINT			uiAddElements,
	void **			ppvTblRV);

FSTATIC void fdictAddItem(
	TDICT *			pTDict,
	FLMUINT			uiFieldNum,
	FLMUINT			uiFieldType);

FSTATIC RCODE	fdictAddIndex(
	TDICT *			pTDict,
	DDENTRY *		pEntry);

FSTATIC RCODE fdictFixupIfdPointers(
	FDICT *			pDict,
	FLMUINT			uiIfdStartOffset);

FSTATIC RCODE fdictAddNewCCS(
	TDICT *			pTDict,
	TENCDEF *		pTEncDef,
	FLMUINT			uiRecNum);
/****************************************************************************
Desc:		Rebuild the dictionary tables reading in all dictionary
			records.
****************************************************************************/
RCODE fdictRebuild(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	TDICT			tDict;
	FLMUINT		uiCount;
	IXD_p			pIxd;
 	FLMBOOL		bTDictInitialized = FALSE;
	FLMBOOL		bSuspended;
	FLMUINT		uiOnlineTransId;

	// Allocate a new FDICT structure for reading the local dictionary
	// into memory.
	// At this point, pDb better not be pointing to a dictionary.

	flmAssert( pDb->pDict == NULL);
	if( RC_BAD( rc = flmAllocDict( &pDb->pDict)))
	{
		goto Exit;
	}

	if( !pDb->pDict->pLFileTbl)
	{
		// Read the local dictionary into memory.

		if( RC_BAD(rc = fdictReadLFiles( pDb, pDb->pDict)))
		{
			goto Exit;
		}

		// For a database create the LFiles still are not created.

		if( pDb->pDict->pLFileTbl->uiLfNum == 0)
		{
			goto Exit;
		}
	}

	bTDictInitialized = TRUE;
	if( RC_BAD( rc = fdictInitTDict( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictProcessAllDictRecs( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictBuildTables( &tDict, FALSE, FALSE)))
	{
		goto Exit;
	}

	// Loop through the IXD and set the uiLastDrnIndexed value.

	uiCount = pDb->pDict->uiIxdCnt;
	for( pIxd = pDb->pDict->pIxdTbl; uiCount--; pIxd++)
	{
		// Ignore any errors in case we are rebuilding.

		if( RC_BAD( flmGetIxTrackerInfo( pDb, pIxd->uiIndexNum,
					&pIxd->uiLastContainerIndexed,
					&pIxd->uiLastDrnIndexed, &uiOnlineTransId, &bSuspended)))
		{
			goto Exit;
		}

		if( bSuspended)
		{
			pIxd->uiFlags |= (IXD_SUSPENDED | IXD_OFFLINE);
		}
		else if( uiOnlineTransId == TRANS_ID_OFFLINE)
		{
			pIxd->uiFlags |= IXD_OFFLINE;
		}
	}

Exit:

	if( bTDictInitialized)
	{
		GedPoolFree( &tDict.pool);
	}

	return( rc );
}

/****************************************************************************
Desc:		Initializes and sets up a TDICT structure.
****************************************************************************/
RCODE fdictInitTDict(
	FDB *			pDb,
	TDICT *		pTDict)
{
	RCODE	rc = FERR_OK;

	f_memset( pTDict, 0, sizeof( TDICT));		// Set elements to zeros.
	GedSmartPoolInit( &pTDict->pool, &g_TDictPoolStats);		

	pTDict->pDb = pDb;
	pTDict->uiVersionNum = pDb->pFile->FileHdr.uiVersionNum;
	pTDict->uiDefaultLanguage =
		pDb->pFile->FileHdr.uiDefaultLanguage;
	pTDict->pDict = pDb->pDict;


	if( RC_BAD(rc = fdictGetContainer( pDb->pDict, FLM_DICT_CONTAINER,
											  	&pTDict->pLFile )))
		goto Exit;
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Build all of the dictionary tables given the temporary dictionary
			(pTDict) that was built in ddprep.
Note:		There are two ways this will be called.  The first is when
			we are building a dictionary from scratch.  The second is ONLY
			when a new field definition or container is added, or an index's
			state is changed.
****************************************************************************/
RCODE fdictBuildTables(
	TDICT *			pTDict,
	FLMBOOL			bRereadLFiles,
	FLMBOOL			bNewDict)
{
	RCODE				rc = FERR_OK;
	DDENTRY *		pEntry;
	TFIELD *			pTField;
	FLMUINT			uiEntryNum;
	TENCDEF *		pTEncDef;

	if( RC_BAD( rc = fdictReallocAllTables( pTDict)))
	{
		goto Exit;
	}

	// Go through and add each new item to the dictionary.

	for( pEntry = pTDict->pFirstEntry
		; pEntry 
		; pEntry = pEntry->pNextEntry )
	{
		uiEntryNum = pEntry->uiEntryNum;
		
		switch( pEntry->uiType)
		{
			case 0:	// Field
			{
				pTField = (TFIELD *) pEntry->vpDef;
				fdictAddItem( pTDict, uiEntryNum, pTField->uiFldInfo);
				break;
			}

			case ITT_INDEX_TYPE:
			{
				fdictAddItem( pTDict, uiEntryNum, ITT_INDEX_TYPE);
				if( RC_BAD( rc = fdictAddIndex( pTDict, pEntry )))
				{
					goto Exit;
				}
				break;
			}

			case ITT_CONTAINER_TYPE:
			{
				fdictAddItem( pTDict, uiEntryNum, ITT_CONTAINER_TYPE);
				// rc = fdictAddLFile( pTDict, pEntry ); Already done.
				break;
			}

			case ITT_ENCDEF_TYPE:
			{
				if (!pTDict->pDb->pFile)
				{
					flmAssert( 0);
					rc = RC_SET( FERR_ENCRYPTION_UNAVAILABLE);
					goto Exit;
				}
				else
				{
					fdictAddItem( pTDict, uiEntryNum, ITT_ENCDEF_TYPE);

					pTEncDef = (TENCDEF *) pEntry->vpDef;
					// Need to add a new CCS.
					if (RC_BAD( rc = fdictAddNewCCS(pTDict, pTEncDef, uiEntryNum)))
					{
						goto Exit;
					}
				}
				break;
			}

			default:
			{
				break;
			}
		}
	}

	if( pTDict->uiNewIfds || bNewDict)
	{
		if( RC_BAD( rc = fdictFixupIfdPointers( pTDict->pDict,
			bNewDict ? 0 : (pTDict->uiTotalIfds - pTDict->uiNewIfds))))
		{
			goto Exit;
		}
	}

	if( bRereadLFiles)
	{
		if( RC_BAD( rc = fdictReadLFiles( pTDict->pDb, pTDict->pDict)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = fdictFixupLFileTbl( pTDict->pDict)))
	{
		goto Exit;
	}

Exit:

	return( rc );
}

/****************************************************************************
Desc:	Fixup pointers in tables of copied dictionary.  This is called after
		a copy of one dictionary to another, or after a dictionary's tables
		have been reallocated.
****************************************************************************/
FSTATIC void fdictFixupPointers(
	FDICT *	pNewDict,
	FDICT *	pOldDict
	)
{
	FLMUINT	uiPos;
	FLMUINT	uiOffset;
	LFILE *	pOldLFile;
	LFILE *	pNewLFile;
	IFD *		pOldIfd;
	IFD *		pNewIfd;
	IXD *		pOldIxd;
	IXD *		pNewIxd;
	ITT *		pOldItt;
	ITT *		pNewItt;

	// Fixup anything that points to LFILE entries.

	if (pNewDict->pLFileTbl && pNewDict->pLFileTbl != pOldDict->pLFileTbl)
	{

		// Fixup pItt->pvItem pointers for indexes and containers

		for (uiPos = 0, pOldItt = pOldDict->pIttTbl,
					pNewItt = pNewDict->pIttTbl;
			  uiPos < pOldDict->uiIttCnt;
			  uiPos++, pOldItt++, pNewItt++)
		{
			if (ITT_IS_CONTAINER( pOldItt) || ITT_IS_INDEX( pOldItt))
			{
				if (pOldItt->pvItem)
				{
					LFILE *	pTmpLFile;

					pTmpLFile = (LFILE *)(pOldItt->pvItem);
					uiOffset = (FLMUINT)(pTmpLFile - pOldDict->pLFileTbl);
					pTmpLFile = pNewDict->pLFileTbl + uiOffset;
					pNewItt->pvItem = (void *)pTmpLFile;
				}
				else
				{
					flmAssert( pNewItt->pvItem == NULL);
				}
			}
			else if (ITT_IS_ENCDEF( pOldItt))
			{
				if (pOldItt->pvItem)
				{
					pNewItt->pvItem = pOldItt->pvItem;
					((F_CCS *)pNewItt->pvItem)->AddRef();
				}
				else
				{
					flmAssert( pNewItt->pvItem == NULL);
				}
			}
		}
	}

	// Fixup anything that points to IXD entries

	if (pNewDict->pIxdTbl && pNewDict->pIxdTbl != pOldDict->pIxdTbl)
	{

		// Fixup pLFile->pIxd pointers

		for (uiPos = 0, pOldLFile = pOldDict->pLFileTbl,
					pNewLFile = pNewDict->pLFileTbl;
			  uiPos < pOldDict->uiLFileCnt;
			  uiPos++, pOldLFile++, pNewLFile++)
		{
			if (pOldLFile->pIxd)
			{
				uiOffset = (FLMUINT)(pOldLFile->pIxd - pOldDict->pIxdTbl);
				pNewLFile->pIxd = pNewDict->pIxdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewLFile->pIxd == NULL);
			}
		}

		// Fixup pIfd->pIxd pointers

		for (uiPos = 0, pOldIfd = pOldDict->pIfdTbl,
					pNewIfd = pNewDict->pIfdTbl;
			  uiPos < pOldDict->uiIfdCnt;
			  uiPos++, pOldIfd++, pNewIfd++)
		{
			if (pOldIfd->pIxd)
			{
				uiOffset = (FLMUINT)(pOldIfd->pIxd - pOldDict->pIxdTbl);
				pNewIfd->pIxd = pNewDict->pIxdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pIxd == NULL);
			}
		}
	}

	// Fixup anything that points to IFD entries

	if (pNewDict->pIfdTbl && pNewDict->pIfdTbl != pOldDict->pIfdTbl)
	{

		// Fixup pIfd->pNextInChain pointers

		for (uiPos = 0, pOldIfd = pOldDict->pIfdTbl,
					pNewIfd = pNewDict->pIfdTbl;
			  uiPos < pOldDict->uiIfdCnt;
			  uiPos++, pOldIfd++, pNewIfd++)
		{
			if (pOldIfd->pNextInChain)
			{
				uiOffset = (FLMUINT)(pOldIfd->pNextInChain - pOldDict->pIfdTbl);
				pNewIfd->pNextInChain = pNewDict->pIfdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pNextInChain == NULL);
			}
		}

		// Fixup pIxd->pFirstIfd pointers

		for (uiPos = 0, pOldIxd = pOldDict->pIxdTbl,
					pNewIxd = pNewDict->pIxdTbl;
			  uiPos < pOldDict->uiIxdCnt;
			  uiPos++, pOldIxd++, pNewIxd++)
		{
			if (pOldIxd->pFirstIfd)
			{
				uiOffset = (FLMUINT)(pOldIxd->pFirstIfd - pOldDict->pIfdTbl);
				pNewIxd->pFirstIfd = pNewDict->pIfdTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIxd->pFirstIfd == NULL);
			}
		}

		// Fixup pItt->pvItem pointers

		for (uiPos = 0, pOldItt = pOldDict->pIttTbl,
					pNewItt = pNewDict->pIttTbl;
			  uiPos < pOldDict->uiIttCnt;
			  uiPos++, pOldItt++, pNewItt++)
		{
			if (ITT_IS_FIELD( pOldItt))
			{
				if (pOldItt->pvItem)
				{
					IFD *	pTmpIfd;

					pTmpIfd = (IFD *)(pOldItt->pvItem);
					uiOffset = (FLMUINT)(pTmpIfd - pOldDict->pIfdTbl);
					pTmpIfd = pNewDict->pIfdTbl + uiOffset;
					pNewItt->pvItem = (void *)pTmpIfd;
				}
				else
				{
					flmAssert( pNewItt->pvItem == NULL);
				}
			}
		}
	}

	// Fixup anything that points to field path entries

	if (pNewDict->pFldPathsTbl && pNewDict->pFldPathsTbl != pOldDict->pFldPathsTbl)
	{

		// Fixup pIfd->pFieldPathCToP and pIfd->pFieldPathPToC pointers

		for (uiPos = 0, pOldIfd = pOldDict->pIfdTbl,
					pNewIfd = pNewDict->pIfdTbl;
			  uiPos < pOldDict->uiIfdCnt;
			  uiPos++, pOldIfd++, pNewIfd++)
		{
			if (pOldIfd->pFieldPathCToP)
			{
				uiOffset = (FLMUINT)(pOldIfd->pFieldPathCToP - pOldDict->pFldPathsTbl);
				pNewIfd->pFieldPathCToP = pNewDict->pFldPathsTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pFieldPathCToP == NULL);
			}
			if (pOldIfd->pFieldPathPToC)
			{
				uiOffset = (FLMUINT)(pOldIfd->pFieldPathPToC - pOldDict->pFldPathsTbl);
				pNewIfd->pFieldPathPToC = pNewDict->pFldPathsTbl + uiOffset;
			}
			else
			{
				flmAssert( pNewIfd->pFieldPathPToC == NULL);
			}
		}

	}
}

/****************************************************************************
Desc:		Allocate all of the dictionary tables based on the counts that
			were incremented in pTDict.  Coded to add new fields, indexes or
			container, but not to modify or delete anything!
****************************************************************************/
FSTATIC RCODE fdictReallocAllTables(
	TDICT *			pTDict)
{
	RCODE				rc = FERR_OK;
	FDICT				OldDict;
	FDICT *			pDict = pTDict->pDict;

	// Save a copy of the old dictionary's pointers and counters
	// Easiest way to do this is to simply copy the structure.

	f_memcpy( &OldDict, pDict, sizeof( FDICT));

	if( pTDict->pLastEntry
	&&  pTDict->pLastEntry->uiEntryNum >= pDict->uiIttCnt
	&&  pTDict->pLastEntry->uiEntryNum < FLM_RESERVED_TAG_NUMS)
	{
		ITT *				pItt;
		FLMUINT			uiNewCount;

		uiNewCount = pTDict->pLastEntry->uiEntryNum + 1 - pDict->uiIttCnt;
		if (uiNewCount)
		{

			// Must fake out so that we don't lose the old table.

			pDict->pIttTbl = NULL;
			if( RC_BAD( rc = fdictReallocTbl( sizeof( ITT), pDict->uiIttCnt,
						uiNewCount, (void **) &pDict->pIttTbl)))
			{
				goto Exit;
			}
			pTDict->uiTotalItts = pDict->uiIttCnt + uiNewCount;

			// Copy the table to the new location (because of fake out above)

			if( OldDict.uiIttCnt)
			{
				f_memcpy( pDict->pIttTbl, OldDict.pIttTbl, 
					sizeof( ITT) * OldDict.uiIttCnt);
			}

			// Initialize the new items to empty.

			pItt = pDict->pIttTbl + pDict->uiIttCnt;
			for( ;uiNewCount--; pItt++)
			{
				pItt->uiType = ITT_EMPTY_SLOT;
				pItt->pvItem = NULL;
			}
		}
	}

	if (pTDict->uiNewIxds)
	{

		// Must fake out so that we don't lose the old table.

		pDict->pIxdTbl = NULL;
		if( RC_BAD( rc = fdictReallocTbl( sizeof( IXD), pDict->uiIxdCnt,
					pTDict->uiNewIxds, (void **)&pDict->pIxdTbl)))
		{
			goto Exit;
		}
		pTDict->uiTotalIxds = pDict->uiIxdCnt + pTDict->uiNewIxds;

		// Copy the table to the new location (because of fake out above)

		if( OldDict.uiIxdCnt)
		{
			f_memcpy( pDict->pIxdTbl, OldDict.pIxdTbl, 
				sizeof( IXD) * OldDict.uiIxdCnt);
		}
	}

	if (pTDict->uiNewIfds)
	{

		// Must fake out so that we don't lose the old table.

		pDict->pIfdTbl = NULL;
  		if( RC_BAD( rc = fdictReallocTbl( sizeof( IFD), pDict->uiIfdCnt,
					pTDict->uiNewIfds, (void **)&pDict->pIfdTbl)))
		{
			goto Exit;
		}
		pTDict->uiTotalIfds = pDict->uiIfdCnt + pTDict->uiNewIfds;

		// Copy the table to the new location (because of fake out above)

		if( OldDict.uiIfdCnt)
		{
			f_memcpy( pDict->pIfdTbl, OldDict.pIfdTbl, 
				sizeof( IFD) * OldDict.uiIfdCnt);
		}
	}

	if (pTDict->uiNewFldPaths)
	{

		// Must fake out so that we don't lose the old table.

		pDict->pFldPathsTbl = NULL;
		if( RC_BAD( rc = fdictReallocTbl( sizeof( FLMUINT), pDict->uiFldPathsCnt,
					pTDict->uiNewFldPaths, (void **)&pDict->pFldPathsTbl)))
		{
			goto Exit;
		}
		pTDict->uiTotalFldPaths = pDict->uiFldPathsCnt + pTDict->uiNewFldPaths;

		// Copy the table to the new location (because of fake out above)

		if( OldDict.uiFldPathsCnt)
		{
			f_memcpy( pDict->pFldPathsTbl, OldDict.pFldPathsTbl, 
				sizeof( FLMUINT) * OldDict.uiFldPathsCnt);
		}
	}

	fdictFixupPointers( pDict, &OldDict);

Exit:

	// Free any old tables where a new table was allocated.

	if (OldDict.pLFileTbl != pDict->pLFileTbl)
	{
		f_free( &OldDict.pLFileTbl);
	}
	if (OldDict.pIttTbl != pDict->pIttTbl)
	{
		f_free( &OldDict.pIttTbl);
	}
	if (OldDict.pIxdTbl != pDict->pIxdTbl)
	{
		f_free( &OldDict.pIxdTbl);
	}
	if (OldDict.pIfdTbl != pDict->pIfdTbl)
	{
		f_free( &OldDict.pIfdTbl);
	}
	if (OldDict.pFldPathsTbl != pDict->pFldPathsTbl)
	{
		f_free( &OldDict.pFldPathsTbl);
	}

	return( rc );
}


/****************************************************************************
Desc:		Allocate or reallocate a table.
****************************************************************************/
FSTATIC RCODE fdictReallocTbl(
	FLMUINT			uiElementSize,
	FLMUINT			uiTblSize,
	FLMUINT			uiAddElements,
	void **			ppvTblRV)
{	
	RCODE				rc = FERR_OK;

	// Does the table need to grow?

	if( uiAddElements)
	{
		if( *ppvTblRV)
		{
			if( RC_BAD( rc = f_recalloc( 
					uiElementSize * (uiTblSize + uiAddElements),
					ppvTblRV)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = f_calloc(	
					uiElementSize * (uiTblSize + uiAddElements),
					ppvTblRV)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc );
}

/****************************************************************************
Desc:		Add a new item to the item type table.
****************************************************************************/
FSTATIC void fdictAddItem(
	TDICT *			pTDict,
	FLMUINT			uiFieldNum,
	FLMUINT			uiFieldType)
{
	FDICT *			pDict = pTDict->pDict;
	ITT *				pItt;

	if( uiFieldNum < FLM_RESERVED_TAG_NUMS)
	{
		pItt = pDict->pIttTbl + uiFieldNum;
		pItt->uiType = uiFieldType;
		pItt->pvItem = NULL;

		if( uiFieldNum >= pDict->uiIttCnt)
		{
			pDict->uiIttCnt = uiFieldNum + 1;
		}
	}
}

/****************************************************************************
Desc:	Add the new IXD, IFD, field paths and LFILE for the index.
****************************************************************************/
FSTATIC RCODE fdictAddIndex(
	TDICT *			pTDict,
	DDENTRY *		pEntry)
{
	RCODE				rc = FERR_OK;
	FDICT *			pDict = pTDict->pDict;
	FLMUINT			uiIndexNum = pEntry->uiEntryNum;
	IXD *				pIxd;
	IFD *				pIfd;
	FLMUINT *		pFirstPToCFld;
	FLMUINT *		pFirstCToPFld;
	FLMUINT *		pCurFld;
	FLMUINT *		pTempFld;
	TIXD *			pTIxd;
	TIFD *			pTIfd;
	TIFP *			pTIfp;

	// The index numbers in the IXD array do not need to be in any order.
	// Just add all of the index information to the end of the table.

	pIxd = pDict->pIxdTbl + pDict->uiIxdCnt++;
	pIxd->uiIndexNum = uiIndexNum;

	pTIxd = (TIXD *) pEntry->vpDef;
	pIxd->uiContainerNum = pTIxd->uiContainerNum;
	pIxd->uiNumFlds = pTIxd->uiNumFlds;
	pIxd->uiFlags = pTIxd->uiFlags;
	pIxd->uiLanguage = pTIxd->uiLanguage;
	pIxd->uiLastContainerIndexed = 0xFFFFFFFF;
	pIxd->uiLastDrnIndexed = DRN_LAST_MARKER;
	pIxd->uiEncId = pTIxd->uiEncId;

	// Setup the IFD elements and the field paths.

	pIxd->pFirstIfd = pIfd = pDict->pIfdTbl + pDict->uiIfdCnt;
	pDict->uiIfdCnt += pIxd->uiNumFlds;

	for( pTIfd = pTIxd->pNextTIfd; pTIfd; pIfd++, pTIfd = pTIfd->pNextTIfd)
	{
		// This is a good place to set the IFD_LAST flag.
		// Could/Should be done in ddprep.c

		if( pTIfd->pNextTIfd == NULL)
			pTIfd->uiFlags |= IFD_LAST;

		pIfd->uiIndexNum = uiIndexNum;
		pIfd->pIxd = pIxd;
		pIfd->uiFlags = pTIfd->uiFlags;
		pIfd->uiLimit = pTIfd->uiLimit;
		pIfd->uiCompoundPos = pTIfd->uiCompoundPos;

		// The pTIfp->pNextTIfp are linked from parent to child.
		pTIfp = pTIfd->pTIfp;
		pCurFld = pDict->pFldPathsTbl + pDict->uiFldPathsCnt;
		pFirstPToCFld = pFirstCToPFld = pCurFld;

		pIfd->pFieldPathPToC = pFirstPToCFld;

		do
		{
			*pCurFld++ = pTIfp->uiFldNum;
			pTIfp = pTIfp->pNextTIfp;

		} while( pTIfp);

		pIfd->uiFldNum = *(pCurFld-1);
		pTempFld = pCurFld - 1;

		// Null Terminate
		*pCurFld++ = 0;

		pTIfp = pTIfd->pTIfp;
		if( pTIfp->pNextTIfp)		// If more than one field make the CToP path.
		{
			pFirstCToPFld = pCurFld;
			while( pTempFld != pFirstPToCFld)
			{
				*pCurFld++ = *pTempFld--;
			}
			*pCurFld++ = *pTempFld;
			*pCurFld++ = 0;
		}
		pIfd->pFieldPathCToP = pFirstCToPFld;
		pDict->uiFldPathsCnt += pCurFld - pFirstPToCFld;
	}

	return( rc );
}

/****************************************************************************
Desc:		Fixup the IFD chain and the pIfd->pIxd pointers.
****************************************************************************/
FSTATIC RCODE fdictFixupIfdPointers(
	FDICT *			pDict,
	FLMUINT			uiIfdStartOffset)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCount;
	IFD *				pIfd;
	ITT *				pItt;
	ITT *				pIttTbl = pDict->pIttTbl;

	// Go through the IFD list and setup the pNextInChain pointers
	// making sure that the required fields are first.

	for( uiCount = pDict->uiIfdCnt - uiIfdStartOffset,
		pIfd = pDict->pIfdTbl + uiIfdStartOffset;
		uiCount; uiCount--, pIfd++)
	{
		IFD *				pPrevInChain;
		IFD *				pTempIfd;
		
		if( pIfd->uiFldNum >= pDict->uiIttCnt)
		{
			if( pIfd->uiFldNum < FLM_RESERVED_TAG_NUMS)
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			continue;
		}
		else
		{
			pItt = pIttTbl + pIfd->uiFldNum;
			if( !ITT_IS_FIELD( pItt))
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
		}

		// Move the field type to the pIfd->uiFlags
		IFD_SET_FIELD_TYPE( pIfd, ITT_FLD_GET_TYPE( pItt));
		
		// Visit: We could verify all of the fields in the field path.
		// Need to include 'any', 'use', 'parent' tags as valid tags.

		if( !pItt->pvItem)
		{
			pItt->pvItem = (void *) pIfd;
		}
		else
		{
			// Follow the chain and index at the front or rear depending on 
			// if the field is required within the set.

			pTempIfd = (IFD *) pItt->pvItem;
			if( (pIfd->uiFlags & IFD_REQUIRED_IN_SET)
			 || !(pTempIfd->uiFlags & IFD_REQUIRED_IN_SET))
			{
				pIfd->pNextInChain = pTempIfd;
				pItt->pvItem = (void *) pIfd;
			}
			else
			{
				// Not required in set and first IFD is required in set.
				// Look for first not required IFD in the chain.

				pPrevInChain = pTempIfd;
				pTempIfd = pTempIfd->pNextInChain;
				
				for( ; pTempIfd; pTempIfd = pTempIfd->pNextInChain)
				{
					if( !(pTempIfd->uiFlags & IFD_REQUIRED_IN_SET))
						break;
					pPrevInChain = pTempIfd;
				}
				pIfd->pNextInChain = pPrevInChain->pNextInChain;
				pPrevInChain->pNextInChain = pIfd;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Fixup the ITT pointers into the LFILE elements and all of 
			the IXD pointers in the LDICT.
****************************************************************************/
RCODE fdictFixupLFileTbl(
	FDICT *			pDict)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiCount;
	LFILE *			pLFile;
	IXD *				pIxd;
	ITT *				pItt;
	ITT *				pIttTbl = pDict->pIttTbl;
	FLMUINT			uiIttCnt = pDict->uiIttCnt;

	for( uiCount = pDict->uiLFileCnt, pLFile = pDict->pLFileTbl
		; uiCount; uiCount--, pLFile++)
	{

		if( pLFile->uiLfNum != FLM_DATA_CONTAINER
		 && pLFile->uiLfNum != FLM_DICT_CONTAINER
		 && pLFile->uiLfNum != FLM_DICT_INDEX
		 && pLFile->uiLfNum != FLM_TRACKER_CONTAINER)
		{
			pItt = pIttTbl + pLFile->uiLfNum;
			
			if( uiIttCnt <= pLFile->uiLfNum || 
				(pLFile->uiLfType == LF_CONTAINER && !ITT_IS_CONTAINER( pItt)))
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			if( pLFile->uiLfType == LF_INDEX && !ITT_IS_INDEX( pItt))
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			
			pItt->pvItem = pLFile;
		}
		else if( pLFile->uiLfNum == FLM_DICT_INDEX)
		{
			// The first IXD should be the dictionary index.

			if( pDict->pIxdTbl && pDict->pIxdTbl->uiIndexNum == FLM_DICT_INDEX)
			{
				pLFile->pIxd = pDict->pIxdTbl;
			}
		}
	}

	// Now that all of the indexes/containers in the ITT table point
	// to the LFILE entries, fixup the LFILE to point to the IXD entries.

	for( uiCount = pDict->uiIxdCnt, pIxd = pDict->pIxdTbl;
		  uiCount; uiCount--, pIxd++)
	{
		if( uiIttCnt <= pIxd->uiIndexNum)
		{
			if( pIxd->uiIndexNum != FLM_DICT_INDEX)
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
		}
		else
		{
			pItt = pIttTbl + pIxd->uiIndexNum;
			pLFile = (LFILE *) pItt->pvItem;

			if( !pLFile)
			{
				rc = RC_SET( FERR_BAD_REFERENCE);
				goto Exit;
			}
			pLFile->pIxd = pIxd;
		}

		// Verify that the pIxd->uiContainerNum is actually a container.
		// A value of 0 means that the index is on ALL containers.

		if (pIxd->uiContainerNum)
		{
			if( uiIttCnt <= pIxd->uiContainerNum)
			{
				if( pIxd->uiContainerNum != FLM_DATA_CONTAINER
				 && pIxd->uiContainerNum != FLM_DICT_CONTAINER
				 && pIxd->uiContainerNum != FLM_TRACKER_CONTAINER)
				{
					rc = RC_SET( FERR_BAD_REFERENCE);
					goto Exit;
				}
			}
			else
			{
				pItt = pIttTbl + pIxd->uiContainerNum;
				if( !ITT_IS_CONTAINER( pItt))
				{
					rc = RC_SET( FERR_BAD_REFERENCE);
					goto Exit;
				}
			}
		}
	}

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Add a new CCS reference to the item type table.  If a key is included
			we can use it, otherwise we will have to generate one.
****************************************************************************/
FSTATIC RCODE fdictAddNewCCS(
	TDICT *			pTDict,
	TENCDEF *		pTEncDef,
	FLMUINT			uiRecNum)
{
	RCODE				rc = FERR_OK;
	FDICT *			pDict = pTDict->pDict;
	ITT *				pItt;
	F_CCS *			pCcs = NULL;
	FDB *				pDb = pTDict->pDb;
	F_CCS *			pDbWrappingKey;

	if( uiRecNum >= FLM_RESERVED_TAG_NUMS)
	{
		goto Exit;
	}
	
	if (!pDb->pFile->bInLimitedMode)
	{
		
		pDbWrappingKey = pDb->pFile->pDbWrappingKey;
	
		flmAssert( pTEncDef);
	
		if ((pCcs = f_new F_CCS) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}
	
		// Setup the F_CCS.
		if (RC_BAD( rc = pCcs->init( FALSE, pTEncDef->uiAlgType )))
		{
			goto Exit;
		}
	
		if (!pTEncDef->uiLength)
		{
			flmAssert( 0);
			rc = RC_SET( FERR_MISSING_ENC_KEY);
			goto Exit;
		}
	
		// We need to set the key information.  This also unwraps the key and stores the
		// handle.
		
		if( RC_BAD( rc = pCcs->setKeyFromStore( pTEncDef->pucKeyInfo,
			(FLMUINT32)pTEncDef->uiLength, NULL, pDbWrappingKey)))
		{
			goto Exit;
		}
	}

	// Save the CCS object in the ITT table.
	
	pItt = pDict->pIttTbl + uiRecNum;
	pItt->pvItem = (void *)pCcs;
	pCcs = NULL;

	if( uiRecNum >= pDict->uiIttCnt)
	{
		pDict->uiIttCnt = uiRecNum + 1;
	}

Exit:

	if (pCcs)
	{
		delete pCcs;
	}

	return( rc);

}

/****************************************************************************
Desc:		Copies an existing dictionary to a new dictionary.  This does not
			fix up all of the ITT's pvItem pointers (including the
			pFirstIfd pointer of fields in the ITT table).  To clone the
			dictionary, call fdictCloneDict.
****************************************************************************/
RCODE fdictCopySkeletonDict(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	FDICT *		pNewDict = NULL;
	FDICT *		pOldDict = pDb->pDict;
	FLMUINT		uiTblSize;
	FLMUINT		uiPos;
	LFILE *		pLFile;
	IXD *			pIxd;
	ITT *			pItt;
	ITT *			pNewIttTbl = NULL;
	FLMUINT		uiNewIttTblLen = 0;
	LFILE *		pNewDictIndexLFile = NULL;
	FLMUINT *	pOldFieldPathsTbl = NULL;
	FLMUINT *	pNewFieldPathsTbl = NULL;

	if( RC_BAD( rc = f_calloc( (FLMUINT)sizeof( FDICT), &pNewDict)))
	{
		goto Exit;
	}

	pNewDict->pNext = pNewDict->pPrev = NULL;
	pNewDict->pFile = NULL;
	pNewDict->uiUseCount = 1;

	// Nothing to do is not a legal state.
	if( !pOldDict)
	{
		flmAssert( pOldDict != NULL);
		pDb->pDict = pNewDict;
		goto Exit;
	}

	// ITT Table

	if( (uiTblSize = pNewDict->uiIttCnt = pOldDict->uiIttCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( ITT), &pNewDict->pIttTbl)))
		{
			goto Exit;
		}
		pNewIttTbl = pNewDict->pIttTbl;
		uiNewIttTblLen = uiTblSize;
		f_memcpy( pNewDict->pIttTbl, pOldDict->pIttTbl,
			uiTblSize * sizeof( ITT));

		// Clear out all of the pointer values.
		pItt = pNewDict->pIttTbl;
		for( uiPos = 0; uiPos < uiTblSize; uiPos++, pItt++)
		{
			if ( pItt->uiType == ITT_ENCDEF_TYPE && !pDb->pFile->bInLimitedMode)
			{
				flmAssert( pItt->pvItem);
				((F_CCS *)pItt->pvItem)->AddRef();
			}
			else
			{
				pItt->pvItem = NULL;
			}
		}
	}

	// LFILE Table

	if( (uiTblSize = pNewDict->uiLFileCnt = pOldDict->uiLFileCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( LFILE),
			&pNewDict->pLFileTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pLFileTbl, pOldDict->pLFileTbl,
			uiTblSize * sizeof( LFILE));

		for( pLFile = pNewDict->pLFileTbl; uiTblSize--; pLFile++)
		{
			if( pLFile->uiLfNum < FLM_RESERVED_TAG_NUMS)
			{
				// WARNING: The code must make a new LFILE
				// before the dictionary is aware of it.

				if( pLFile->uiLfNum < uiNewIttTblLen)
				{
					pItt = pNewIttTbl + pLFile->uiLfNum;
					pItt->pvItem = (void *) pLFile;
				}
			}
			else if( pLFile->uiLfNum == FLM_DICT_INDEX)
			{
				pNewDictIndexLFile = pLFile;
			}
		}
	}

	// IXD Table

	if( (uiTblSize = pNewDict->uiIxdCnt = pOldDict->uiIxdCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc(
			uiTblSize * sizeof( IXD), &pNewDict->pIxdTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pIxdTbl, pOldDict->pIxdTbl,
			uiTblSize * sizeof( IXD));

		// Fixup all of the pointers to the IXD.

		for( pIxd = pNewDict->pIxdTbl; uiTblSize--; pIxd++)
		{
			if( pIxd->uiIndexNum != FLM_DICT_INDEX)
			{
				pItt = pNewIttTbl + pIxd->uiIndexNum;
				pLFile = (LFILE *) pItt->pvItem;
				pLFile->pIxd = pIxd;
			}
			else if( pNewDictIndexLFile)
			{
				pNewDictIndexLFile->pIxd = pIxd;
			}
		}
	}

	// Field Paths Table

	if( (uiTblSize = pNewDict->uiFldPathsCnt = pOldDict->uiFldPathsCnt) != 0)
	{
		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( FLMUINT),
					&pNewDict->pFldPathsTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pFldPathsTbl, pOldDict->pFldPathsTbl,
			uiTblSize * sizeof( FLMUINT));

		pOldFieldPathsTbl = pOldDict->pFldPathsTbl;
		pNewFieldPathsTbl = pNewDict->pFldPathsTbl;
	}

	// IFD Table

	if( (uiTblSize = pNewDict->uiIfdCnt = pOldDict->uiIfdCnt) != 0)
	{
		IFD *			pIfd;
		FLMUINT		uiLastIndexNum;
		FLMUINT		uiOffset;

		if( RC_BAD( rc = f_alloc( uiTblSize * sizeof( IFD),
					&pNewDict->pIfdTbl)))
		{
			goto Exit;
		}
		f_memcpy( pNewDict->pIfdTbl, pOldDict->pIfdTbl,
			uiTblSize * sizeof( IFD));

		// Fixup all pFirstIfd pointers, backlinks to the pIxd and fldPathTbls.
		// Set all of the IfdChain values to NULL to be fixed up later.
		pIfd = pNewDict->pIfdTbl;
		uiLastIndexNum = 0;

		for( uiPos = 0; uiPos < uiTblSize; uiPos++, pIfd++)
		{
			pIfd->pNextInChain = NULL;

			if( pIfd->uiIndexNum != FLM_DICT_INDEX)
			{
				pItt = pNewIttTbl + pIfd->uiIndexNum;
				pLFile = (LFILE *) pItt->pvItem;
				pIxd = pLFile->pIxd;
			}
			else
			{
				pIxd = pNewDictIndexLFile->pIxd;
			}

			pIfd->pIxd = pIxd;
			if( uiLastIndexNum != pIfd->uiIndexNum)
			{
				pIxd->pFirstIfd = pIfd;
				uiLastIndexNum = pIfd->uiIndexNum;
			}

			// Fixup the field paths.

			flmAssert( pNewFieldPathsTbl != NULL);
			uiOffset = pIfd->pFieldPathCToP - pOldFieldPathsTbl;
			pIfd->pFieldPathCToP = pNewFieldPathsTbl + uiOffset;

			uiOffset = pIfd->pFieldPathPToC - pOldFieldPathsTbl;
			pIfd->pFieldPathPToC = pNewFieldPathsTbl + uiOffset;
		}
	}

	f_mutexLock( gv_FlmSysData.hShareMutex);
	flmUnlinkFdbFromDict( pDb);
	f_mutexUnlock( gv_FlmSysData.hShareMutex);
	pDb->pDict = pNewDict;
	pNewDict = NULL;

Exit:

	if( RC_BAD( rc) && pNewDict)
	{
		// Undo all of the allocations on the new table.
		if( pNewDict->pLFileTbl)
		{
			f_free( &pNewDict->pLFileTbl);
		}
		if( pNewDict->pIttTbl)
		{
			f_free( &pNewDict->pIttTbl);
		}
		if( pNewDict->pIxdTbl)
		{
			f_free( &pNewDict->pIxdTbl);
		}
		if( pNewDict->pIfdTbl)
		{
			f_free( &pNewDict->pIfdTbl);
		}
		if( pNewDict->pFldPathsTbl)
		{
			f_free( &pNewDict->pFldPathsTbl);
		}
		f_free( &pNewDict);
	}

	return( rc);
}

/**************************************************************************** 
Desc:	Creates a new version of the current dictionary and fixes up all
		pointers
****************************************************************************/
RCODE fdictCloneDict(
	FDB *			pDb)
{
	RCODE			rc = FERR_OK;
	TDICT			tDict;
	FLMBOOL		bTDictInitialized = FALSE;

	if( RC_BAD( rc = fdictCopySkeletonDict( pDb)))
	{
		goto Exit;
	}

	bTDictInitialized = TRUE;
	if( RC_BAD( rc = fdictInitTDict( pDb, &tDict)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = fdictBuildTables( &tDict, FALSE, TRUE)))
	{
		goto Exit;
	}

	pDb->uiFlags |= FDB_UPDATED_DICTIONARY;

Exit:

	if( bTDictInitialized)
	{
		GedPoolFree( &tDict.pool);
	}

	// If we allocated an FDICT and there was an error, free the FDICT.

	if( (RC_BAD( rc)) && (pDb->pDict))
	{
		flmFreeDict( pDb->pDict);
		pDb->pDict = NULL;
	}

	return( rc);
}
