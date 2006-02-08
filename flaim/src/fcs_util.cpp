//-------------------------------------------------------------------------
// Desc:	Server utility routines.
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcs_util.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FSTATIC FLMBOOL flmGetNextHexPacketSlot( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	f_randomGenerator *	pRandGen,
	FLMUINT *				puiSlot);

FSTATIC RCODE flmGetNextHexPacketBytes( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	FLMBYTE *				pucPacket,
	f_randomGenerator *	pRandGen,
	FLMBYTE *				pucBuf,
	FLMUINT					uiCount);

/****************************************************************************
Desc:	Converts a UNICODE string consisting of 7-bit ASCII characters to
		a native string.
*****************************************************************************/
RCODE fcsConvertUnicodeToNative(
	POOL *					pPool,
	const FLMUNICODE *	puzUnicode,
	char **					ppucNative)
{
	RCODE			rc = FERR_OK;
	char *		pucDest = NULL;
	FLMUINT		uiCount;

	uiCount = 0;
	while( puzUnicode[ uiCount])
	{
		if( puzUnicode[ uiCount] > 0x007F)
		{
			rc = RC_SET( FERR_CONV_ILLEGAL);
			goto Exit;
		}
		uiCount++;
	}

	if( (pucDest = (char *)GedPoolAlloc( pPool,
											(FLMUINT)(uiCount + 1))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	uiCount = 0;
	while( puzUnicode[ uiCount])
	{
		pucDest[ uiCount] = f_tonative( (FLMBYTE)puzUnicode[ uiCount]);
		uiCount++;
	}

	pucDest[ uiCount] = '\0';

Exit:

	*ppucNative = pucDest;
	return( rc);
}


/****************************************************************************
Desc:	Converts a native string to a double-byte UNICODE string.
*****************************************************************************/
RCODE fcsConvertNativeToUnicode(
	POOL *				pPool,
	const char *		pszNative,
	FLMUNICODE **		ppuzUnicode)
{
	RCODE				rc = FERR_OK;
	FLMUNICODE *	puzDest;
	FLMUINT			uiCount;
	
	uiCount = f_strlen( pszNative);
	
	if( (puzDest = (FLMUNICODE *)GedPoolAlloc( pPool,
		(FLMUINT)((FLMUINT)sizeof( FLMUNICODE) * 
			(FLMUINT)(uiCount + 1)))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	uiCount = 0;
	while( pszNative[ uiCount])
	{
		puzDest[ uiCount] = (FLMUNICODE)f_toascii( pszNative[ uiCount]);
		uiCount++;
	}

	puzDest[ uiCount] = 0;

Exit:

	*ppuzUnicode = puzDest;
	return( rc);
}


/****************************************************************************
Desc:	Initializes members of a CREATE_OPTS structure to their default values
*****************************************************************************/
void fcsInitCreateOpts(
	CREATE_OPTS *	pCreateOptsRV)
{
	/*
	Initialize the CREATE_OPTS structure to its default values.
	*/

	f_memset( pCreateOptsRV, 0, sizeof( CREATE_OPTS));

	pCreateOptsRV->uiBlockSize = DEFAULT_BLKSIZ;
	pCreateOptsRV->uiMinRflFileSize = DEFAULT_MIN_RFL_FILE_SIZE;
	pCreateOptsRV->uiMaxRflFileSize = DEFAULT_MAX_RFL_FILE_SIZE;
	pCreateOptsRV->bKeepRflFiles = DEFAULT_KEEP_RFL_FILES_FLAG;
	pCreateOptsRV->bLogAbortedTransToRfl = DEFAULT_LOG_ABORTED_TRANS_FLAG;
	pCreateOptsRV->uiDefaultLanguage = DEFAULT_LANG;
	pCreateOptsRV->uiVersionNum = FLM_CURRENT_VERSION_NUM;
}

/****************************************************************************
Desc:	Converts a CHECKPOINT_INFO structure to an HTD tree
*****************************************************************************/
RCODE fcsBuildCheckpointInfo(
	CHECKPOINT_INFO *		pChkptInfo,
	POOL *					pPool,
	NODE **					ppTree)
{
	NODE *		pRootNd = NULL;
	void *		pMark = GedPoolMark( pPool);
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	*ppTree = NULL;

	/*
	Build the root node of the tree.
	*/

	if( (pRootNd = GedNodeMake( pPool, FCS_CPI_CONTEXT, &rc)) == NULL)
	{
		goto Exit;
	}

	/*
	Add fields to the tree.
	*/

	if( pChkptInfo->bRunning)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_RUNNING, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiRunningTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_START_TIME, (void *)&pChkptInfo->uiRunningTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->bForcingCheckpoint)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_FORCING_CP, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiForceCheckpointRunningTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_FORCE_CP_START_TIME,
			(void *)&pChkptInfo->uiForceCheckpointRunningTime,
			4, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->iForceCheckpointReason)
	{
		uiTmp = pChkptInfo->iForceCheckpointReason;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_FORCE_CP_REASON, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->bWritingDataBlocks)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_WRITING_DATA_BLOCKS, (void *)&uiTmp,
			4, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiLogBlocksWritten)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_LOG_BLOCKS_WRITTEN, 
			(void *)&pChkptInfo->uiLogBlocksWritten,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiDataBlocksWritten)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_DATA_BLOCKS_WRITTEN,
			(void *)&pChkptInfo->uiDataBlocksWritten,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiDirtyCacheBytes)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_DIRTY_CACHE_BYTES,
			(void *)&pChkptInfo->uiDirtyCacheBytes,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiBlockSize)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_BLOCK_SIZE, (void *)&pChkptInfo->uiBlockSize,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pChkptInfo->uiWaitTruncateTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pRootNd,
			FCS_CPI_WAIT_TRUNC_TIME, (void *)&pChkptInfo->uiWaitTruncateTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	*ppTree = pRootNd;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pMark);
	}

	return( rc);
}


/****************************************************************************
Desc:	Converts a LOCK_USER structure (or list of structures) to an HTD tree
*****************************************************************************/
RCODE fcsBuildLockUser(
	LOCK_USER *		pLockUser,
	FLMBOOL			bList,
	POOL *			pPool,
	NODE **			ppTree)
{
	NODE *		pRootNd = NULL;
	NODE *		pContextNd = NULL;
	void *		pMark = GedPoolMark( pPool);
	RCODE			rc = FERR_OK;

	*ppTree = NULL;

	if( !pLockUser)
	{
		goto Exit;
	}

	/*
	Add fields to the tree.
	*/

	for( ;;)
	{
		if( (pContextNd = GedNodeMake( pPool, FCS_LUSR_CONTEXT, &rc)) == NULL)
		{
			goto Exit;
		}

		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_LUSR_THREAD_ID, (void *)&pLockUser->uiThreadId,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_LUSR_TIME, (void *)&pLockUser->uiTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		if( pRootNd == NULL)
		{
			pRootNd = pContextNd;
		}
		else
		{
			GedSibGraft( pRootNd, pContextNd, GED_LAST);
		}

		if( !bList)
		{
			break;
		}

		pLockUser++;
		if( !pLockUser->uiTime)
		{
			// Hit the last item in the list
			break;
		}
	}

	*ppTree = pRootNd;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pMark);
	}

	return( rc);
}


/****************************************************************************
Desc:	Converts an HTD tree to a LOCK_USER structure (or list of structures)
*****************************************************************************/
RCODE fcsExtractLockUser(
	NODE *			pTree,
	FLMBOOL			bExtractAsList,
	void *			pvLockUser)
{
	NODE *			pTmpNd;
	FLMUINT			uiItemCount = 0;
	FLMUINT			fieldPath[ 8];
	LOCK_USER *		pLockUser = NULL;
	FLMUINT			uiLoop;
	RCODE				rc = FERR_OK;

	if( !pTree)
	{
		if( bExtractAsList)
		{
			*((LOCK_USER **)pvLockUser) = NULL;
		}
		else
		{
			f_memset( (LOCK_USER *)pvLockUser, 0, sizeof( LOCK_USER));
		}
		goto Exit;
	}

	if( bExtractAsList)
	{
		pTmpNd = pTree;
		while( pTmpNd != NULL)
		{
			if( GedTagNum( pTmpNd) == FCS_LUSR_CONTEXT)
			{
				uiItemCount++;
			}
			pTmpNd = pTmpNd->next;
		}

		if( RC_BAD( rc = f_alloc( 
			sizeof( LOCK_USER) * (uiItemCount + 1), &pLockUser)))
		{
			goto Exit;
		}

		*((LOCK_USER **)pvLockUser) = pLockUser;
	}
	else
	{
		pLockUser = (LOCK_USER *)pvLockUser;
		f_memset( pLockUser, 0, sizeof( LOCK_USER));
		uiItemCount = 1;
	}
	
	/*
	Parse the tree and extract the values.
	*/

	for( uiLoop = 0; uiLoop < uiItemCount; uiLoop++)
	{
		fieldPath[ 0] = FCS_LUSR_CONTEXT;
		fieldPath[ 1] = FCS_LUSR_THREAD_ID;
		fieldPath[ 2] = 0;

		if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pLockUser[ uiLoop].uiThreadId);
		}

		fieldPath[ 0] = FCS_LUSR_CONTEXT;
		fieldPath[ 1] = FCS_LUSR_TIME;
		fieldPath[ 2] = 0;

		if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pLockUser[ uiLoop].uiTime);
		}

		pTree = GedSibNext( pTree);
	}

	if( bExtractAsList)
	{
		f_memset( &(pLockUser[ uiItemCount]), 0, sizeof( LOCK_USER));
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:	Extracts a CHECKPOINT_INFO structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractCheckpointInfo(
	NODE *					pTree,
	CHECKPOINT_INFO *		pChkptInfo)
{
	NODE *		pTmpNd;
	FLMUINT		fieldPath[ 8];
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	/*
	Initialize the structure
	*/

	f_memset( pChkptInfo, 0, sizeof( CHECKPOINT_INFO));

	/*
	Parse the tree and extract the values.
	*/

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_RUNNING;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pChkptInfo->bRunning = uiTmp ? TRUE : FALSE;
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_START_TIME;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiRunningTime);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_FORCING_CP;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pChkptInfo->bForcingCheckpoint = uiTmp ? TRUE : FALSE;
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_FORCE_CP_START_TIME;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiForceCheckpointRunningTime);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_FORCE_CP_REASON;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetINT( pTmpNd, &pChkptInfo->iForceCheckpointReason);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_WRITING_DATA_BLOCKS;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pChkptInfo->bWritingDataBlocks = uiTmp ? TRUE : FALSE;
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_LOG_BLOCKS_WRITTEN;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiLogBlocksWritten);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_DATA_BLOCKS_WRITTEN;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiDataBlocksWritten);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_DIRTY_CACHE_BYTES;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiDirtyCacheBytes);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_BLOCK_SIZE;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiBlockSize);
	}

	fieldPath[ 0] = FCS_CPI_CONTEXT;
	fieldPath[ 1] = FCS_CPI_WAIT_TRUNC_TIME;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pChkptInfo->uiWaitTruncateTime);
	}

	return( rc);
}

/****************************************************************************
Desc:	Translates a FLAIM query operator to a c/s query operator
*****************************************************************************/
RCODE fcsTranslateQFlmToQCSOp(
	QTYPES			eFlmOp,
	FLMUINT *		puiCSOp)
{
	RCODE		rc = FERR_OK;

	switch( eFlmOp)
	{
		case FLM_AND_OP:
			*puiCSOp = FCS_ITERATOR_AND_OP;
			break;
		case FLM_OR_OP:
			*puiCSOp = FCS_ITERATOR_OR_OP;
			break;
		case FLM_NOT_OP:
			*puiCSOp = FCS_ITERATOR_NOT_OP;
			break;
		case FLM_EQ_OP:
			*puiCSOp = FCS_ITERATOR_EQ_OP;
			break;
		case FLM_MATCH_OP:
			*puiCSOp = FCS_ITERATOR_MATCH_OP;
			break;
		case FLM_MATCH_BEGIN_OP:
			*puiCSOp = FCS_ITERATOR_MATCH_BEGIN_OP;
			break;
		case FLM_CONTAINS_OP:
			*puiCSOp = FCS_ITERATOR_CONTAINS_OP;
			break;
		case FLM_NE_OP:
			*puiCSOp = FCS_ITERATOR_NE_OP;
			break;
		case FLM_LT_OP:
			*puiCSOp = FCS_ITERATOR_LT_OP;
			break;
		case FLM_LE_OP:
			*puiCSOp = FCS_ITERATOR_LE_OP;
			break;
		case FLM_GT_OP:
			*puiCSOp = FCS_ITERATOR_GT_OP;
			break;
		case FLM_GE_OP:
			*puiCSOp = FCS_ITERATOR_GE_OP;
			break;
		case FLM_BITAND_OP:
			*puiCSOp = FCS_ITERATOR_BITAND_OP;
			break;
		case FLM_BITOR_OP:
			*puiCSOp = FCS_ITERATOR_BITOR_OP;
			break;
		case FLM_BITXOR_OP:
			*puiCSOp = FCS_ITERATOR_BITXOR_OP;
			break;
		case FLM_MULT_OP:
			*puiCSOp = FCS_ITERATOR_MULT_OP;
			break;
		case FLM_DIV_OP:
			*puiCSOp = FCS_ITERATOR_DIV_OP;
			break;
		case FLM_MOD_OP:
			*puiCSOp = FCS_ITERATOR_MOD_OP;
			break;
		case FLM_PLUS_OP:
			*puiCSOp = FCS_ITERATOR_PLUS_OP;
			break;
		case FLM_MINUS_OP:
			*puiCSOp = FCS_ITERATOR_MINUS_OP;
			break;
		case FLM_NEG_OP:
			*puiCSOp = FCS_ITERATOR_NEG_OP;
			break;
		case FLM_LPAREN_OP:
			*puiCSOp = FCS_ITERATOR_LPAREN_OP;
			break;
		case FLM_RPAREN_OP:
			*puiCSOp = FCS_ITERATOR_RPAREN_OP;
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

	return( rc);
}

/****************************************************************************
Desc:	Translates a FLAIM query operator to a c/s query operator
*****************************************************************************/
RCODE fcsTranslateQCSToQFlmOp(
	FLMUINT 		uiCSOp,
	QTYPES *		peFlmOp)
{
	RCODE		rc = FERR_OK;

	switch( uiCSOp)
	{
		case FCS_ITERATOR_AND_OP:
			*peFlmOp = FLM_AND_OP;
			break;
		case FCS_ITERATOR_OR_OP:
			*peFlmOp = FLM_OR_OP;
			break;
		case FCS_ITERATOR_NOT_OP:
			*peFlmOp = FLM_NOT_OP;
			break;
		case FCS_ITERATOR_EQ_OP:
			*peFlmOp = FLM_EQ_OP;
			break;
		case FCS_ITERATOR_MATCH_OP:
			*peFlmOp = FLM_MATCH_OP;
			break;
		case FCS_ITERATOR_MATCH_BEGIN_OP:
			*peFlmOp = FLM_MATCH_BEGIN_OP;
			break;
		case FCS_ITERATOR_CONTAINS_OP:
			*peFlmOp = FLM_CONTAINS_OP;
			break;
		case FCS_ITERATOR_NE_OP:
			*peFlmOp = FLM_NE_OP;
			break;
		case FCS_ITERATOR_LT_OP:
			*peFlmOp = FLM_LT_OP;
			break;
		case FCS_ITERATOR_LE_OP:
			*peFlmOp = FLM_LE_OP;
			break;
		case FCS_ITERATOR_GT_OP:
			*peFlmOp = FLM_GT_OP;
			break;
		case FCS_ITERATOR_GE_OP:
			*peFlmOp = FLM_GE_OP;
			break;
		case FCS_ITERATOR_BITAND_OP:
			*peFlmOp = FLM_BITAND_OP;
			break;
		case FCS_ITERATOR_BITOR_OP:
			*peFlmOp = FLM_BITOR_OP;
			break;
		case FCS_ITERATOR_BITXOR_OP:
			*peFlmOp = FLM_BITXOR_OP;
			break;
		case FCS_ITERATOR_MULT_OP:
			*peFlmOp = FLM_MULT_OP;
			break;
		case FCS_ITERATOR_DIV_OP:
			*peFlmOp = FLM_DIV_OP;
			break;
		case FCS_ITERATOR_MOD_OP:
			*peFlmOp = FLM_MOD_OP;
			break;
		case FCS_ITERATOR_PLUS_OP:
			*peFlmOp = FLM_PLUS_OP;
			break;
		case FCS_ITERATOR_MINUS_OP:
			*peFlmOp = FLM_MINUS_OP;
			break;
		case FCS_ITERATOR_NEG_OP:
			*peFlmOp = FLM_NEG_OP;
			break;
		case FCS_ITERATOR_LPAREN_OP:
			*peFlmOp = FLM_LPAREN_OP;
			break;
		case FCS_ITERATOR_RPAREN_OP:
			*peFlmOp = FLM_RPAREN_OP;
			break;
		default:
			rc = RC_SET( FERR_NOT_IMPLEMENTED);
			break;
	}

	return( rc);
}

/****************************************************************************
Desc:	Converts an FINDEX_STATUS structure to an HTD tree
*****************************************************************************/
RCODE fcsBuildIndexStatus(
	FINDEX_STATUS *	pIndexStatus,
	POOL *				pPool,
	NODE **				ppTree)
{
	NODE *		pContextNd = NULL;
	void *		pMark = GedPoolMark( pPool);
	FLMUINT		uiTmp;
	RCODE			rc = FERR_OK;

	*ppTree = NULL;

	if( !pIndexStatus)
	{
		goto Exit;
	}

	/*
	Add fields to the tree.
	*/

	if( (pContextNd = GedNodeMake( pPool, FCS_IXSTAT_CONTEXT, &rc)) == NULL)
	{
		goto Exit;
	}

	if( pIndexStatus->uiIndexNum)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_INDEX_NUM, (void *)&pIndexStatus->uiIndexNum,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiStartTime)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_START_TIME, (void *)&pIndexStatus->uiStartTime,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		// Send the "auto-online" flag for backwards compatibility

		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_AUTO_ONLINE, 
			(void *)&uiTmp, 0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}

		// Send the priority (high) for backwards compatibility

		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_PRIORITY,
			(void *)&uiTmp, 0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	// Set the suspended time field (for backwards compatibility)
	// if the index is suspended

	if( pIndexStatus->bSuspended)
	{
		f_timeGetSeconds( &uiTmp);
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_SUSPEND_TIME, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiLastRecordIdIndexed)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_LAST_REC_INDEXED, 
			(void *)&pIndexStatus->uiLastRecordIdIndexed,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiKeysProcessed)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_KEYS_PROCESSED, 
			(void *)&pIndexStatus->uiKeysProcessed,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->uiRecordsProcessed)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_RECS_PROCESSED, 
			(void *)&pIndexStatus->uiRecordsProcessed,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pIndexStatus->bSuspended)
	{
		uiTmp = (FLMUINT)pIndexStatus->bSuspended;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_IXSTAT_STATE, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	*ppTree = pContextNd;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Extracts an FINDEX_STATUS structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractIndexStatus(
	NODE *					pTree,
	FINDEX_STATUS *		pIndexStatus)
{
	NODE *		pTmpNd;
	FLMUINT		fieldPath[ 8];
	RCODE			rc = FERR_OK;

	/*
	Initialize the structure
	*/

	f_memset( pIndexStatus, 0, sizeof( FINDEX_STATUS));

	/*
	Make sure pTree is non-null
	*/

	if( !pTree)
	{
		goto Exit;
	}

	/*
	Parse the tree and extract the values.
	*/

	fieldPath[ 0] = FCS_IXSTAT_CONTEXT;
	fieldPath[ 1] = FCS_IXSTAT_INDEX_NUM;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiIndexNum);
	}

	fieldPath[ 1] = FCS_IXSTAT_START_TIME;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiStartTime);
	}

	fieldPath[ 1] = FCS_IXSTAT_LAST_REC_INDEXED;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiLastRecordIdIndexed);
	}

	fieldPath[ 1] = FCS_IXSTAT_KEYS_PROCESSED;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiKeysProcessed);
	}

	fieldPath[ 1] = FCS_IXSTAT_RECS_PROCESSED;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pIndexStatus->uiRecordsProcessed);
	}

	fieldPath[ 1] = FCS_IXSTAT_STATE;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		FLMUINT		uiTmp;
		(void)GedGetUINT( pTmpNd, &uiTmp);
		pIndexStatus->bSuspended = uiTmp ? TRUE : FALSE;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Converts an FLM_MEM_INFO structure to an HTD tree
*****************************************************************************/
RCODE fcsBuildMemInfo(
	FLM_MEM_INFO *		pMemInfo,
	POOL *				pPool,
	NODE **				ppTree)
{
	FLMUINT				uiTmp;
	NODE *				pContextNd = NULL;
	NODE *				pSubContext = NULL;
	void *				pMark = GedPoolMark( pPool);
	FLM_CACHE_USAGE *	pUsage;
	RCODE					rc = FERR_OK;

	*ppTree = NULL;

	if( !pMemInfo)
	{
		goto Exit;
	}

	/*
	Add fields to the tree.
	*/

	if( (pContextNd = GedNodeMake( pPool, 
		FCS_MEMINFO_CONTEXT, &rc)) == NULL)
	{
		goto Exit;
	}

	if( pMemInfo->bDynamicCacheAdjust)
	{
		uiTmp = 1;
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_DYNA_CACHE_ADJ, (void *)&uiTmp,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustPercent)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_PERCENT, 
			(void *)&pMemInfo->uiCacheAdjustPercent,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustMin)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_MIN,
			(void *)&pMemInfo->uiCacheAdjustMin,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustMax)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_MAX,
			(void *)&pMemInfo->uiCacheAdjustMax,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pMemInfo->uiCacheAdjustMinToLeave)
	{
		if( RC_BAD( rc = gedAddField( pPool, pContextNd,
			FCS_MEMINFO_CACHE_ADJ_MIN_LEAVE,
			(void *)&pMemInfo->uiCacheAdjustMinToLeave,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	pUsage = &pMemInfo->RecordCache;
	if( (pSubContext = GedNodeMake( pPool, 
		FCS_MEMINFO_RECORD_CACHE, &rc)) == NULL)
	{
		goto Exit;
	}

add_usage:

	if( pUsage->uiMaxBytes)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_MAX_BYTES,
			(void *)&pUsage->uiMaxBytes,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCount)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_COUNT,
			(void *)&pUsage->uiCount,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiOldVerCount)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_OLD_VER_COUNT,
			(void *)&pUsage->uiOldVerCount,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiTotalBytesAllocated)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_TOTAL_BYTES_ALLOC,
			(void *)&pUsage->uiTotalBytesAllocated,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiOldVerBytes)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_OLD_VER_BYTES,
			(void *)&pUsage->uiOldVerBytes,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheHits)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_HITS,
			(void *)&pUsage->uiCacheHits,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheHitLooks)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_HIT_LOOKS,
			(void *)&pUsage->uiCacheHitLooks,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheFaults)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_FAULTS,
			(void *)&pUsage->uiCacheFaults,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( pUsage->uiCacheFaultLooks)
	{
		if( RC_BAD( rc = gedAddField( pPool, pSubContext,
			FCS_MEMINFO_CACHE_FAULT_LOOKS,
			(void *)&pUsage->uiCacheFaultLooks,
			0, FLM_NUMBER_TYPE)))
		{
			goto Exit;
		}
	}

	if( GedChild( pSubContext))
	{
		GedChildGraft( pContextNd, pSubContext, GED_LAST);
	}

	if( pUsage != &pMemInfo->BlockCache)
	{
		pUsage = &pMemInfo->BlockCache;
		if( (pSubContext = GedNodeMake( pPool, 
			FCS_MEMINFO_BLOCK_CACHE, &rc)) == NULL)
		{
			goto Exit;
		}
		goto add_usage;
	}

	*ppTree = pContextNd;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Extracts a FLM_MEM_INFO structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractMemInfo(
	NODE *				pTree,
	FLM_MEM_INFO *		pMemInfo)
{
	NODE *					pTmpNd;
	FLMUINT					fieldPath[ 8];
	FLMUINT					uiTmp;
	FLM_CACHE_USAGE *		pUsage;
	FLMUINT					uiUsageTag;
	RCODE						rc = FERR_OK;

	/*
	Initialize the structure
	*/

	f_memset( pMemInfo, 0, sizeof( FLM_MEM_INFO));

	/*
	Make sure pTree is non-null
	*/

	if( !pTree)
	{
		goto Exit;
	}

	/*
	Parse the tree and extract the values.
	*/

	fieldPath[ 0] = FCS_MEMINFO_CONTEXT;
	fieldPath[ 1] = FCS_MEMINFO_DYNA_CACHE_ADJ;
	fieldPath[ 2] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &uiTmp);
		pMemInfo->bDynamicCacheAdjust = (FLMBOOL)(uiTmp ? TRUE : FALSE);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_PERCENT;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustPercent);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_MIN;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustMin);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_MAX;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustMax);
	}

	fieldPath[ 1] = FCS_MEMINFO_CACHE_ADJ_MIN_LEAVE;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pMemInfo->uiCacheAdjustMinToLeave);
	}

	pUsage = &pMemInfo->RecordCache;
	uiUsageTag = FCS_MEMINFO_RECORD_CACHE;

get_usage:

	fieldPath[ 0] = FCS_MEMINFO_CONTEXT;
	fieldPath[ 1] = uiUsageTag;
	fieldPath[ 2] = FCS_MEMINFO_MAX_BYTES;
	fieldPath[ 3] = 0;

	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiMaxBytes);
	}

	fieldPath[ 2] = FCS_MEMINFO_COUNT;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCount);
	}

	fieldPath[ 2] = FCS_MEMINFO_OLD_VER_COUNT;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiOldVerCount);
	}

	fieldPath[ 2] = FCS_MEMINFO_TOTAL_BYTES_ALLOC;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiTotalBytesAllocated);
	}

	fieldPath[ 2] = FCS_MEMINFO_OLD_VER_BYTES;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiOldVerBytes);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_HITS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheHits);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_HIT_LOOKS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheHitLooks);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_FAULTS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheFaults);
	}

	fieldPath[ 2] = FCS_MEMINFO_CACHE_FAULT_LOOKS;
	if( (pTmpNd = GedPathFind( GED_TREE, pTree, fieldPath, 1)) != NULL)
	{
		(void) GedGetUINT( pTmpNd, &pUsage->uiCacheFaultLooks);
	}

	if( pUsage != &pMemInfo->BlockCache)
	{
		pUsage = &pMemInfo->BlockCache;
		uiUsageTag = FCS_MEMINFO_BLOCK_CACHE;
		goto get_usage;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Builds a GEDCOM tree containing information on all FLAIM threads
*****************************************************************************/
RCODE fcsBuildThreadInfo(
	POOL *				pPool,
	NODE **				ppTree)
{
	NODE *				pContextNd = NULL;
	NODE *				pRootNd = NULL;
	void *				pMark = GedPoolMark( pPool);
	F_THREAD_INFO *	pThreadInfo = NULL;
	FLMUINT				uiNumThreads;
	FLMUINT				uiLoop;
	RCODE					rc = FERR_OK;

	*ppTree = NULL;

	// Query FLAIM for available threads

	if( RC_BAD( rc = FlmGetThreadInfo( pPool, &pThreadInfo, &uiNumThreads)))
	{
		goto Exit;
	}

	if( (pRootNd = GedNodeMake( pPool, 
		FCS_THREAD_INFO_ROOT, &rc)) == NULL)
	{
		goto Exit;
	}

	if( RC_BAD( rc = GedPutRecPtr( pPool, pRootNd, uiNumThreads)))
	{
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiNumThreads; uiLoop++)
	{
		// Add fields to the tree.

		if( (pContextNd = GedNodeMake( pPool, 
			FCS_THREAD_INFO_CONTEXT, &rc)) == NULL)
		{
			goto Exit;
		}

		GedChildGraft( pRootNd, pContextNd, GED_LAST);

		if( pThreadInfo->uiThreadId)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_ID, (void *)&pThreadInfo->uiThreadId,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->uiThreadGroup)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_GROUP, (void *)&pThreadInfo->uiThreadGroup,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->uiAppId)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_APP_ID, (void *)&pThreadInfo->uiAppId,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->uiStartTime)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_START_TIME, (void *)&pThreadInfo->uiStartTime,
				0, FLM_NUMBER_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->pszThreadName)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_NAME, (void *)pThreadInfo->pszThreadName,
				0, FLM_TEXT_TYPE)))
			{
				goto Exit;
			}
		}

		if( pThreadInfo->pszThreadStatus)
		{
			if( RC_BAD( rc = gedAddField( pPool, pContextNd,
				FCS_THREADINFO_THREAD_STATUS, (void *)pThreadInfo->pszThreadStatus,
				0, FLM_TEXT_TYPE)))
			{
				goto Exit;
			}
		}

		pThreadInfo++;
	}

	*ppTree = pRootNd;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Extracts a list of F_THREAD_INFO structure from an HTD tree.
*****************************************************************************/
RCODE fcsExtractThreadInfo(
	NODE *				pTree,
	POOL *				pPool,
	F_THREAD_INFO **	ppThreadInfo,
	FLMUINT *			puiNumThreads)
{
	NODE *				pTmpNd;
	NODE *				pContextNd;
	void *				pMark = GedPoolMark( pPool);
	FLMUINT				uiTmp;
	F_THREAD_INFO *	pThreadInfo;
	F_THREAD_INFO *	pCurThread;
	FLMUINT				uiNumThreads;
	FLMUINT				uiLoop;
	RCODE					rc = FERR_OK;

	*ppThreadInfo = NULL;
	*puiNumThreads = 0;

	if( GedTagNum( pTree) != FCS_THREAD_INFO_ROOT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = GedGetUINT( pTree, &uiNumThreads)))
	{
		goto Exit;
	}

	if( !uiNumThreads)
	{
		goto Exit;
	}

	if( (pThreadInfo = (F_THREAD_INFO *)GedPoolCalloc( pPool, 
		uiNumThreads * sizeof( F_THREAD_INFO))) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( (pContextNd = GedFind( 1, pTree, 
		FCS_THREAD_INFO_CONTEXT, 1)) == NULL)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( uiLoop = 0, pCurThread = pThreadInfo; 
		uiLoop < uiNumThreads; 
		uiLoop++, pCurThread++)
	{

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_ID, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiThreadId);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_GROUP, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiThreadGroup);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_APP_ID, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiAppId);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_START_TIME, 1)) != NULL)
		{
			(void) GedGetUINT( pTmpNd, &pCurThread->uiStartTime);
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_NAME, 1)) != NULL)
		{
			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, NULL, &uiTmp)))
			{
				goto Exit;
			}

			if( uiTmp)
			{
				uiTmp++;
				if( (pCurThread->pszThreadName = (char *)GedPoolAlloc( 
					pPool, uiTmp)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}

			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, 
				pCurThread->pszThreadName, &uiTmp)))
			{
				goto Exit;
			}
		}

		if( (pTmpNd = GedFind( 1, pContextNd, 
			FCS_THREADINFO_THREAD_STATUS, 1)) != NULL)
		{
			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, NULL, &uiTmp)))
			{
				goto Exit;
			}

			if( uiTmp)
			{
				uiTmp++;
				if( (pCurThread->pszThreadStatus = (char *)GedPoolAlloc( 
					pPool, uiTmp)) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
			}

			if( RC_BAD( rc = GedGetNATIVE( pTmpNd, 
				pCurThread->pszThreadStatus, &uiTmp)))
			{
				goto Exit;
			}
		}

		if( (pContextNd = GedSibNext( pContextNd)) != NULL)
		{
			if( GedTagNum( pContextNd) != FCS_THREAD_INFO_CONTEXT)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
	}

	*ppThreadInfo = pThreadInfo;
	*puiNumThreads = uiNumThreads;

Exit:

	if( RC_BAD( rc))
	{
		GedPoolReset( pPool, pMark);
	}

	return( rc);
}

/****************************************************************************
Desc:	Reads a block from a remote database
*****************************************************************************/
RCODE fcsGetBlock(
	HFDB			hDb,
	FLMUINT		uiAddress,
	FLMUINT		uiMinTransId,
	FLMUINT *	puiCount,
	FLMUINT *	puiBlocksExamined,
	FLMUINT *	puiNextBlkAddr,
	FLMUINT		uiFlags,
	FLMBYTE *	pucBlock)
{
	FDB *				pDb = (FDB *)hDb;
	RCODE				rc = FERR_OK;

	flmAssert( IsInCSMode( hDb));

	fdbInitCS( pDb);
	CS_CONTEXT_p		pCSContext = pDb->pCSContext;
	FCL_WIRE				Wire( pCSContext, pDb);

	if( !pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_DATABASE, FCS_OP_DATABASE_GET_BLOCK)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_ADDRESS, uiAddress)))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_TRANSACTION_ID,
		uiMinTransId)))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_COUNT, *puiCount)))
	{
		goto Transmission_Error;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_FLAGS, uiFlags)))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		if( rc != FERR_IO_END_OF_FILE)
		{
			goto Exit;
		}
	}

	*puiBlocksExamined = (FLMUINT)Wire.getNumber2();
	*puiCount = (FLMUINT)Wire.getCount();
	*puiNextBlkAddr = Wire.getAddress();
	if( *puiCount)
	{
		f_memcpy( pucBlock, Wire.getBlock(), Wire.getBlockSize());
	}

	goto Exit;

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;

Exit:

	fdbExit( pDb);
	return( rc);
}

/****************************************************************************
Desc:	Instructs the server to generate a serial number
*****************************************************************************/
RCODE fcsCreateSerialNumber(
	void *			pvCSContext,
	FLMBYTE *		pucSerialNum)
{
	RCODE				rc = FERR_OK;
	CS_CONTEXT *	pCSContext = (CS_CONTEXT *)pvCSContext;
	FCL_WIRE			Wire( pCSContext);

	if( !pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_MISC, FCS_OP_CREATE_SERIAL_NUM)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	if( !Wire.getSerialNum())
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	f_memcpy( pucSerialNum, Wire.getSerialNum(), F_SERIAL_NUM_SIZE);
	goto Exit;

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Sets or clears the backup active flag for the database
Note:	This should only be called internally from the backup routines.
*****************************************************************************/
RCODE fcsSetBackupActiveFlag(
	HFDB			hDb,
	FLMBOOL		bBackupActive)
{
	FDB *				pDb = (FDB *)hDb;
	RCODE				rc = FERR_OK;

	flmAssert( IsInCSMode( hDb));

	fdbInitCS( pDb);
	CS_CONTEXT_p		pCSContext = pDb->pCSContext;
	FCL_WIRE				Wire( pCSContext, pDb);

	if( !pCSContext->bConnectionGood)
	{
		rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendOp(
		FCS_OPCLASS_DATABASE, FCS_OP_DB_SET_BACKUP_FLAG)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = Wire.sendNumber( WIRE_VALUE_BOOLEAN, bBackupActive)))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.sendTerminate()))
	{
		goto Transmission_Error;
	}

	/* Read the response. */

	if (RC_BAD( rc = Wire.read()))
	{
		goto Transmission_Error;
	}

	if( RC_BAD( rc = Wire.getRCode()))
	{
		goto Exit;
	}

	goto Exit;

Transmission_Error:
	pCSContext->bConnectionGood = FALSE;
	goto Exit;

Exit:

	fdbExit( pDb);
	return( rc);
}

/****************************************************************************
Desc:	Commits an update transaction and updates the log header.
Note:	This should only be called internally from the backup routines.
*****************************************************************************/
RCODE fcsDbTransCommitEx(
	HFDB			hDb,
	FLMBOOL		bForceCheckpoint,
	FLMBYTE *	pucLogHdr)
{
	RCODE			rc = FERR_OK;
	FDB *			pDb = (FDB *)hDb;
	FLMBOOL		bInitializedFdb = FALSE;

	if( IsInCSMode( hDb))
	{
		fdbInitCS( pDb);
		bInitializedFdb = TRUE;
		FCL_WIRE Wire( pDb->pCSContext, pDb);

		if (!pDb->pCSContext->bConnectionGood)
		{
			rc = RC_SET( FERR_BAD_SERVER_CONNECTION);
		}
		else
		{
			rc = Wire.doTransOp(
				FCS_OP_TRANSACTION_COMMIT_EX, 0, 0, 0,
				pucLogHdr, bForceCheckpoint);
		}
	}
	else
	{
		rc = RC_SET( FERR_ILLEGAL_OP);
		goto Exit;
	}

Exit:

	if( bInitializedFdb)
	{
		fdbExit( pDb);
	}

	return( rc);
}

/****************************************************************************
Desc: Generates a hex-encoded, obfuscated string consisting of characters
		0-9, A-F from the passed-in data buffer.
*****************************************************************************/
RCODE flmGenerateHexPacket(
	FLMBYTE *		pucData,
	FLMUINT			uiDataSize,
	FLMBYTE **		ppucPacket)
{
	FLMUINT32 *				pui32CRCTbl = NULL;
	FLMBYTE *				pucBinPacket = NULL;
	FLMBYTE *				pucHexPacket = NULL;
	FLMBYTE *				pucUsedMap = NULL;
	FLMUINT32				ui32Tmp;
	FLMUINT					uiLoop;
	FLMUINT					uiSlot = 0;
	FLMBYTE					ucTmp[ 32];
	FLMUINT					uiBinPacketSize;
	FLMBOOL					bTmp;
	f_randomGenerator		randGen;
	RCODE						rc = FERR_OK;

	// Determine the packet size.  Make the minimum packet size 128 bytes
	// to account for the 64-byte "header" and for the overhead of the
	// CRC bytes, etc.  Round the packet size up to the nearest 64-byte
	// boundary after adding on the data size.

	uiBinPacketSize = 128 + uiDataSize;
	if( (uiBinPacketSize % 64) != 0)
	{
		uiBinPacketSize += (64 - (uiBinPacketSize % 64));
	}

	// Allocate buffers for building the packet

	if( RC_BAD( rc = f_alloc( uiBinPacketSize, &pucBinPacket)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = f_calloc( uiBinPacketSize, &pucUsedMap)))
	{
		goto Exit;
	}

	// First 64-bytes of the packet are reserved as a header

	f_memset( pucUsedMap, 0xFF, 64);

	// Initialize the CRC table.

	if( RC_BAD( rc = f_initCRCTable( &pui32CRCTbl)))
	{
		goto Exit;
	}

	// Initialize the random number generator and seed with the current
	// time.

	f_randomize( &randGen);

	// Fill the packet with random "noise"

	for( uiLoop = 0; uiLoop < uiBinPacketSize; uiLoop += 4)
	{
		ui32Tmp = f_randomLong( &randGen);
		UD2FBA( ui32Tmp, &pucBinPacket[ uiLoop]);
	}

	for( uiLoop = 0; uiLoop < 512; uiLoop++)
	{
		ui32Tmp = f_randomLong( &randGen);
		UD2FBA( ui32Tmp, &pucBinPacket[ f_randomChoice( 
			&randGen, 1, (int)(uiBinPacketSize / 4)) - 1]);
	}

	// Determine a new random seed based on bytes in the
	// packet header

	if( (ui32Tmp = (FLMUINT32)FB2UD( &pucBinPacket[ 
		f_randomChoice( &randGen, 1, 61) - 1])) == 0)
	{
		ui32Tmp = 1;
	}

	f_randomSetSeed( &randGen, ui32Tmp);

	// Use the CRC of the header and the also first four bytes
	// of the header as an 8-byte validation signature.  This will
	// be needed to decode the packet.

	// Initialize the CRC to 0xFFFFFFFF and then compute the 1's
	// complement of the returned CRC.  This implements the 
	// "standard" CRC used by PKZIP, etc.

	ui32Tmp = 0xFFFFFFFF;
	f_updateCRC( pui32CRCTbl, pucBinPacket, 64, &ui32Tmp);
	ui32Tmp = ~ui32Tmp;
	UD2FBA( ui32Tmp, &ucTmp[ 0]);
	f_memcpy( &ucTmp[ 4], pucBinPacket, 4);

	for( uiLoop = 0; uiLoop < 8; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			&randGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = ucTmp[ uiLoop];
	}

	// Encode the data size

	UD2FBA( uiDataSize, &ucTmp[ 0]);
	for( uiLoop = 0; uiLoop < 4; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			&randGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = ucTmp[ uiLoop];
	}

	// Randomly dispurse the data throughout the buffer.  Obfuscate the
	// data using the first 64-bytes of the buffer.

	for( uiLoop = 0; uiLoop < uiDataSize; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			&randGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = pucData[ uiLoop] ^ pucBinPacket[ uiLoop % 64];
	}

	// Calculate and encode the data CRC

	ui32Tmp = 0xFFFFFFFF;
	f_updateCRC( pui32CRCTbl, pucData, uiDataSize, &ui32Tmp);
	ui32Tmp = ~ui32Tmp;
	UD2FBA( ui32Tmp, &ucTmp[ 0]);

	for( uiLoop = 0; uiLoop < 4; uiLoop++)
	{
		bTmp = flmGetNextHexPacketSlot( pucUsedMap, uiBinPacketSize,
			&randGen, &uiSlot);

		flmAssert( bTmp);
		pucBinPacket[ uiSlot] = ucTmp[ uiLoop];
	}

	// Hex encode the binary packet

	if( RC_BAD( rc = f_alloc(
		(uiBinPacketSize * 2) + 1, &pucHexPacket)))
	{
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < uiBinPacketSize; uiLoop++)
	{
		FLMBYTE		ucLowNibble = pucBinPacket[ uiLoop] & 0x0F;
		FLMBYTE		ucHighNibble = (pucBinPacket[ uiLoop] & 0xF0) >> 4;

		pucHexPacket[ uiLoop << 1] = (ucHighNibble <= 9 
													? (ucHighNibble + '0') 
													: ((ucHighNibble - 0xA) + 'A'));

		pucHexPacket[ (uiLoop << 1) + 1] = (ucLowNibble <= 9 
													? (ucLowNibble + '0') 
													: ((ucLowNibble - 0xA) + 'A'));
	}

	pucHexPacket[ uiBinPacketSize * 2] = 0;
	*ppucPacket = pucHexPacket;
	pucHexPacket = NULL;

Exit:

	if( pui32CRCTbl)
	{
		f_freeCRCTable( &pui32CRCTbl);
	}

	if( pucUsedMap)
	{
		f_free( &pucUsedMap);
	}

	if( pucBinPacket)
	{
		f_free( &pucBinPacket);
	}

	if( pucHexPacket)
	{
		f_free( &pucHexPacket);
	}

	return( rc);
}

/****************************************************************************
Desc: Extracts a data buffer from the passed-in hex-encoded, obfuscated
		string.
*****************************************************************************/
RCODE flmExtractHexPacketData(
	FLMBYTE *		pucPacket,
	FLMBYTE **		ppucData,
	FLMUINT *		puiDataSize)
{
	FLMUINT32 *				pui32CRCTbl = NULL;
	FLMBYTE *				pucUsedMap = NULL;
	FLMBYTE *				pucData = NULL;
	FLMBYTE *				pucBinPacket = NULL;
	FLMBYTE *				pucTmp;
	FLMUINT32				ui32Tmp;
	FLMUINT32				ui32FirstCRC;
	FLMUINT32				ui32Seed;
	FLMUINT					uiPacketSize;
	FLMUINT					uiLoop;
	FLMUINT					uiDataSize;
	FLMBYTE					ucTmp[ 32];
	FLMBYTE					ucVal = 0;
	FLMBOOL					bValid;
	f_randomGenerator		randGen;
	RCODE						rc = FERR_OK;

	// Determine the packet size, ignoring all characters except 0-9, A-F

	uiPacketSize = 0;
	pucTmp = pucPacket;
	while( *pucTmp)
	{
		if( (*pucTmp >= '0' && *pucTmp <= '9') ||
			(*pucTmp >= 'A' && *pucTmp <= 'F'))
		{
			uiPacketSize++;
		}
		pucTmp++;
	}

	if( uiPacketSize & 0x00000001 || 
		(uiPacketSize % 4) != 0 || uiPacketSize < 128)
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	// Get the actual size of the decoded binary data by dividing
	// the packet size by 2

	uiPacketSize >>= 1;

	// Allocate a buffer and convert the data from hex ASCII to binary

	if( RC_BAD( rc = f_calloc( 
		uiPacketSize, &pucBinPacket)))
	{
		goto Exit;
	}

	uiLoop = 0;
	pucTmp = pucPacket;
	while( *pucTmp)
	{
		bValid = FALSE;
		if( *pucTmp >= '0' && *pucTmp <= '9')
		{
			ucVal = *pucTmp - '0';
			bValid = TRUE;
		}
		else if( *pucTmp >= 'A' && *pucTmp <= 'F')
		{
			ucVal = (*pucTmp - 'A') + 0x0A;
			bValid = TRUE;
		}

		if( bValid)
		{
			if( (uiLoop & 0x00000001) == 0)
			{
				ucVal <<= 4;
			}
			pucBinPacket[ uiLoop >> 1] |= ucVal;
			uiLoop++;
		}

		pucTmp++;
	}

	// Allocate the data map

	if( RC_BAD( rc = f_calloc( uiPacketSize, &pucUsedMap)))
	{
		goto Exit;
	}

	// First 64-bytes of the packet are reserved

	f_memset( pucUsedMap, 0xFF, 64);

	// Initialize the CRC table

	if( RC_BAD( rc = f_initCRCTable( &pui32CRCTbl)))
	{
		goto Exit;
	}

	// Determine the CRC of the 1st 64-bytes

	ui32FirstCRC = 0xFFFFFFFF;
	f_updateCRC( pui32CRCTbl, pucBinPacket, 64, &ui32FirstCRC);
	ui32FirstCRC = ~ui32FirstCRC;

	// Search for the random seed within the first 64 bytes

	ui32Seed = 0;
	for( uiLoop = 0; uiLoop < 61; uiLoop++)
	{
		ui32Tmp = FB2UD( &pucBinPacket[ uiLoop]);
		f_randomSetSeed( &randGen, ui32Tmp);

		if( RC_BAD( rc = flmGetNextHexPacketBytes( pucUsedMap, uiPacketSize, 
			pucBinPacket, &randGen, ucTmp, 8)))
		{
			goto Exit;
		}

		if( FB2UD( &ucTmp[ 0]) == ui32FirstCRC && 
			f_memcmp( &ucTmp[ 4], &pucBinPacket[ 0], 4) == 0)
		{
			ui32Seed = ui32Tmp;
			break;
		}

		// Reset the "used" map
		f_memset( pucUsedMap, 0, uiPacketSize);
		f_memset( pucUsedMap, 0xFF, 64);
	}

	if( !ui32Seed)
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	// Get the data size

	if( RC_BAD( rc = flmGetNextHexPacketBytes( pucUsedMap, uiPacketSize, 
		pucBinPacket, &randGen, ucTmp, 4)))
	{
		goto Exit;
	}

	uiDataSize = (FLMUINT)FB2UD( &ucTmp[ 0]);
	if( uiDataSize > uiPacketSize)
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	// Allocate space for the data

	if( RC_BAD( rc = f_alloc( uiDataSize, &pucData)))
	{
		goto Exit;
	}

	// Get the data

	if( RC_BAD( rc = flmGetNextHexPacketBytes( 
		pucUsedMap, uiPacketSize, 
		pucBinPacket, &randGen, pucData, uiDataSize)))
	{
		goto Exit;
	}

	// Un-obfuscate the data

	for( uiLoop = 0; uiLoop < uiDataSize; uiLoop++)
	{
		pucData[ uiLoop] = pucData[ uiLoop] ^ pucBinPacket[ uiLoop % 64];
	}

	// Get the data CRC

	if( RC_BAD( rc = flmGetNextHexPacketBytes( 
		pucUsedMap, uiPacketSize, 
		pucBinPacket, &randGen, ucTmp, 4)))
	{
		goto Exit;
	}

	// Verify the data CRC

	ui32Tmp = 0xFFFFFFFF;
	f_updateCRC( pui32CRCTbl, pucData, uiDataSize, &ui32Tmp);
	ui32Tmp = ~ui32Tmp;

	if( ui32Tmp != FB2UD( &ucTmp[ 0]))
	{
		rc = RC_SET( FERR_INVALID_CRC);
		goto Exit;
	}

	*ppucData = pucData;
	pucData = NULL;
	*puiDataSize = uiDataSize;

Exit:

	if( pui32CRCTbl)
	{
		f_freeCRCTable( &pui32CRCTbl);
	}

	if( pucUsedMap)
	{
		f_free( &pucUsedMap);
	}

	if( pucData)
	{
		f_free( &pucData);
	}

	if( pucBinPacket)
	{
		f_free( &pucBinPacket);
	}

	return( rc);
}

/****************************************************************************
Desc: Used by flmGenerateHexPacket to find an unused byte in the packet
*****************************************************************************/
FSTATIC FLMBOOL flmGetNextHexPacketSlot( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	f_randomGenerator *	pRandGen,
	FLMUINT *				puiSlot)
{
	FLMUINT		uiLoop;
	FLMUINT		uiSlot = 0;
	FLMBOOL		bFound = FALSE;

	for( uiLoop = 0; uiLoop < 100; uiLoop++)
	{
		uiSlot = ((FLMUINT)f_randomLong( pRandGen)) % uiMapSize;
		if( !pucUsedMap[ uiSlot])
		{
			bFound = TRUE;
			goto Exit;
		}
	}

	// Scan the table from the top to find an empty slot

	for( uiSlot = 0; uiSlot < uiMapSize; uiSlot++)
	{
		if( !pucUsedMap[ uiSlot])
		{
			bFound = TRUE;
			goto Exit;
		}
	}

Exit:

	if( bFound)
	{
		flmAssert( uiSlot < uiMapSize);
		*puiSlot = uiSlot;
		pucUsedMap[ uiSlot] = 0xFF;
	}

	return( bFound);
}

/****************************************************************************
Desc: Used by flmExtractHexPacket to get the next N bytes of data from the
		packet.
*****************************************************************************/
FSTATIC RCODE flmGetNextHexPacketBytes( 
	FLMBYTE *				pucUsedMap,
	FLMUINT					uiMapSize,
	FLMBYTE *				pucPacket,
	f_randomGenerator *	pRandGen,
	FLMBYTE *				pucBuf,
	FLMUINT					uiCount)
{
	FLMUINT		uiSlot;
	FLMUINT		uiLoop;
	RCODE			rc = FERR_OK;

	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		if( !flmGetNextHexPacketSlot( pucUsedMap, uiMapSize,
			pRandGen, &uiSlot))
		{
			rc = RC_SET( FERR_INVALID_CRC);
			goto Exit;
		}

		pucBuf[ uiLoop] = pucPacket[ uiSlot];
	}

Exit:

	return( rc);
}
	
/****************************************************************************
Desc: Decodes a string containing %XX sequences and does it in place.
		Typically, this data comes from an HTML form.
****************************************************************************/
void fcsDecodeHttpString(
	char *		pszSrc)
{
	char *	pszDest;

	pszDest = pszSrc;
	while( *pszSrc)
	{
		if( *pszSrc == '%')
		{
			pszSrc++;
			if( f_isHexChar( pszSrc[ 0]) && f_isHexChar( pszSrc[ 1]))
			{
				*pszDest = (f_getHexVal( pszSrc[ 0]) << 4) |
					f_getHexVal( pszSrc[ 1]);

				pszSrc += 2;
				pszDest++;
				continue;
			}
		}
		else if( *pszSrc == '+')
		{
			*pszDest = ' ';
			pszSrc++;
			pszDest++;
			continue;
		}

		if( pszSrc != pszDest)
		{
			*pszDest = *pszSrc;
		}
		pszSrc++;
		pszDest++;
	}

	*pszDest = 0;
}
