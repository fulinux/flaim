//------------------------------------------------------------------------------
// Desc:
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
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
//------------------------------------------------------------------------------

#include "xflaim.h"
#include "xflaim_Db.h"
#include "xflaim_IndexStatus.h"
#include "xflaim_ImportStats.h"
#include "xflaim_CheckpointInfo.h"
#include "jniftk.h"

// Field IDs for the IndexStatus class in JAVA.

static jfieldID	fidIndexNum = NULL;
static jfieldID	fidState = NULL;
static jfieldID	fidStartTime = NULL;
static jfieldID	fidLastDocumentIndexed = NULL;
static jfieldID	fidKeysProcessed = NULL;
static jfieldID	fidDocumentsProcessed = NULL;
static jfieldID	fidTransactions = NULL;

// Field IDs for the ImportStats class in JAVA.

static jfieldID	fidLines = NULL;
static jfieldID	fidChars = NULL;
static jfieldID	fidAttributes = NULL;
static jfieldID	fidElements = NULL;
static jfieldID	fidText = NULL;
static jfieldID	fidDocuments = NULL;
static jfieldID	fidErrLineNum = NULL;
static jfieldID	fidErrLineOffset = NULL;
static jfieldID	fidErrorType = NULL;
static jfieldID	fidErrLineFilePos = NULL;
static jfieldID	fidErrLineBytes = NULL;
static jfieldID	fidUTF8Encoding = NULL;

// Field IDs for the CheckpointInfo class in JAVA
	
static jfieldID	fidRunning = NULL;
static jfieldID	fidRunningTime = NULL;
static jfieldID	fidForcingCheckpoint = NULL;
static jfieldID	fidForceCheckpointRunningTime = NULL;
static jfieldID	fidForceCheckpointReason = NULL;
static jfieldID	fidWritingDataBlocks = NULL;
static jfieldID	fidLogBlocksWritten = NULL;
static jfieldID	fidDataBlocksWritten = NULL;
static jfieldID	fidDirtyCacheBytes = NULL;
static jfieldID	fidBlockSize = NULL;
static jfieldID	fidWaitTruncateTime = NULL;
	
#define THIS_FDB() \
	((IF_Db *)(FLMUINT)lThis)

FSTATIC RCODE getUniString(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf);
	
FSTATIC RCODE getUTF8String(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf);
	
FSTATIC RCODE getDictName(
	IF_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	F_DynaBuf *		pDynaBuf);
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_FDB()->Release();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin__JIII(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iTransactionType,
	jint				iMaxLockWait,
	jint				iFlags)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();

	if (RC_BAD( rc = pDb->transBegin( (eDbTransType)iTransactionType,
		(FLMUINT)iMaxLockWait, (FLMUINT)iFlags)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transBegin__JJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lSrcDb)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_Db *			pSrcDb = ((IF_Db *)(FLMUINT)lSrcDb);

	if (RC_BAD( rc = pDb->transBegin( pSrcDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transCommit(
	JNIEnv *			pEnv,
	jobject			obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();

	(void)obj;
	
	if (RC_BAD( rc = pDb->transCommit()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1transAbort(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->transAbort()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getTransType(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *			pDb = THIS_FDB();
	
	return( (jint)pDb->getTransType());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1doCheckpoint(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iTimeout)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->doCheckpoint( (FLMUINT)iTimeout)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1dbLock(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iLockType,
	jint				iPriority,
	jint				iTimeout)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->dbLock( (eLockType)iLockType,
									(FLMINT)iPriority, (FLMUINT)iTimeout)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1dbUnlock(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->dbUnlock()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockType(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMBOOL			bImplicit;
	
	if (RC_BAD( rc = pDb->getLockType( &lockType, &bImplicit)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (int)lockType);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getLockImplicit(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMBOOL			bImplicit;
	
	if (RC_BAD( rc = pDb->getLockType( &lockType, &bImplicit)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( bImplicit ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockThreadId(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiThreadId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockNumExclQueued(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumExclQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockNumSharedQueued(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumSharedQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getLockPriorityCount(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iPriority)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	eLockType		lockType;
	FLMUINT			uiThreadId;
	FLMUINT			uiNumExclQueued;
	FLMUINT			uiNumSharedQueued;
	FLMUINT			uiPriorityCount;
	
	if (RC_BAD( rc = pDb->getLockInfo( (FLMINT)iPriority, &lockType,
										&uiThreadId, &uiNumExclQueued,
										&uiNumSharedQueued, &uiPriorityCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiNumSharedQueued);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1indexSuspend(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->indexSuspend( (FLMUINT)iIndexNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1indexResume(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->indexResume( (FLMUINT)iIndexNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1indexGetNext(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iCurrIndex)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiCurrIndex = (FLMUINT)iCurrIndex;
	
	if (RC_BAD( rc = pDb->indexGetNext( &uiCurrIndex)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiCurrIndex);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CheckpointInfo_initIDs(
	JNIEnv *	pEnv,
	jclass	jCheckpointInfoClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fidRunning = pEnv->GetFieldID( jCheckpointInfoClass,
								"bRunning", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fidRunningTime = pEnv->GetFieldID( jCheckpointInfoClass,
								"iRunningTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidForcingCheckpoint = pEnv->GetFieldID( jCheckpointInfoClass,
								"bForcingCheckpoint", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fidForceCheckpointRunningTime = pEnv->GetFieldID( jCheckpointInfoClass,
								"iForceCheckpointRunningTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidForceCheckpointReason = pEnv->GetFieldID( jCheckpointInfoClass,
								"iForceCheckpointReason", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidWritingDataBlocks = pEnv->GetFieldID( jCheckpointInfoClass,
							"bWritingDataBlocks", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((fidLogBlocksWritten = pEnv->GetFieldID( jCheckpointInfoClass,
								"iLogBlocksWritten", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidDataBlocksWritten = pEnv->GetFieldID( jCheckpointInfoClass,
								"iDataBlocksWritten", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidDirtyCacheBytes = pEnv->GetFieldID( jCheckpointInfoClass,
								"iDirtyCacheBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidBlockSize = pEnv->GetFieldID( jCheckpointInfoClass,
								"iBlockSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidWaitTruncateTime = pEnv->GetFieldID( jCheckpointInfoClass,
								"iWaitTruncateTime", "I")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_IndexStatus_initIDs(
	JNIEnv *	pEnv,
	jclass	jIndexStatusClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fidIndexNum = pEnv->GetFieldID( jIndexStatusClass,
								"iIndexNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidState = pEnv->GetFieldID( jIndexStatusClass,
								"iState", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidStartTime = pEnv->GetFieldID( jIndexStatusClass,
								"iStartTime", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidLastDocumentIndexed = pEnv->GetFieldID( jIndexStatusClass,
								"lLastDocumentIndexed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fidKeysProcessed = pEnv->GetFieldID( jIndexStatusClass,
								"lKeysProcessed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fidDocumentsProcessed = pEnv->GetFieldID( jIndexStatusClass,
							"lDocumentsProcessed", "J")) == NULL)
	{
		goto Exit;
	}
	if ((fidTransactions = pEnv->GetFieldID( jIndexStatusClass,
								"lTransactions", "J")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_ImportStats_initIDs(
	JNIEnv *	pEnv,
	jclass	jImportStatsClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((fidIndexNum = pEnv->GetFieldID( jImportStatsClass,
								"iIndexNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidLines = pEnv->GetFieldID( jImportStatsClass,
								"iLines", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidChars = pEnv->GetFieldID( jImportStatsClass,
								"iChars", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidAttributes = pEnv->GetFieldID( jImportStatsClass,
								"iAttributes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidElements = pEnv->GetFieldID( jImportStatsClass,
								"iElements", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidText = pEnv->GetFieldID( jImportStatsClass,
								"iText", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidDocuments = pEnv->GetFieldID( jImportStatsClass,
							"iDocuments", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidErrLineNum = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidErrLineOffset = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineOffset", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidErrorType = pEnv->GetFieldID( jImportStatsClass,
								"iErrorType", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidErrLineFilePos = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineFilePos", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidErrLineBytes = pEnv->GetFieldID( jImportStatsClass,
								"iErrLineBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((fidUTF8Encoding = pEnv->GetFieldID( jImportStatsClass,
								"bUTF8Encoding", "Z")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Db__1indexStatus(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iIndex)
{
	RCODE					rc = NE_XFLM_OK;
	IF_Db *				pDb = THIS_FDB();
	XFLM_INDEX_STATUS	ixStatus;
	jclass				jIndexStatusClass = NULL;
	jobject				jIndexStatus = NULL;
	
	if (RC_BAD( rc = pDb->indexStatus( (FLMUINT)iIndex, &ixStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Find the IndexStatus class

	if ((jIndexStatusClass = pEnv->FindClass( "xflaim/IndexStatus")) == NULL)
	{
		goto Exit;
	}

	// Allocate an index status class.
	
	if ((jIndexStatus = pEnv->AllocObject( jIndexStatusClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the fields in the object
	
	pEnv->SetIntField( jIndexStatus, fidIndexNum, iIndex);
	pEnv->SetIntField( jIndexStatus, fidState, (jint)ixStatus.eState);
	pEnv->SetIntField( jIndexStatus, fidStartTime, (jint)ixStatus.uiStartTime);
	pEnv->SetLongField( jIndexStatus, fidLastDocumentIndexed,
						(jlong)ixStatus.ui64LastDocumentIndexed);
	pEnv->SetLongField( jIndexStatus, fidKeysProcessed,
						(jlong)ixStatus.ui64KeysProcessed);
	pEnv->SetLongField( jIndexStatus, fidDocumentsProcessed,
						(jlong)ixStatus.ui64DocumentsProcessed);
	pEnv->SetLongField( jIndexStatus, fidTransactions,
						(jlong)ixStatus.ui64Transactions);
	
Exit:

	return( jIndexStatus);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1reduceSize(
	JNIEnv *			pEnv,
	jobject,			// jobject
	jlong				lThis,
	jint				iCount)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMUINT			uiCount = 0;
	
	if (RC_BAD( rc = pDb->reduceSize( (FLMUINT)iCount, &uiCount)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)uiCount);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getDataType(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iNameId)
{
	RCODE		rc = NE_XFLM_OK;
	IF_Db *	pDb = THIS_FDB();
	FLMUINT	uiDataType;
	
	if (RC_BAD( rc = pDb->getDataType( (FLMUINT)iDictType, (FLMUINT)iNameId,
									&uiDataType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiDataType);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Db__1import(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jint				iCollection,
	jlong				lNodeToLinkTo,
	jint				iInsertLoc)
{
	RCODE						rc = NE_XFLM_OK;
	IF_PosIStream *		pIStream = (IF_PosIStream *)((FLMUINT)lIStream);
	IF_DOMNode *			pNodeToLinkTo = (IF_DOMNode *)((FLMUINT)lNodeToLinkTo);
	IF_Db *					pDb = THIS_FDB();
	XFLM_IMPORT_STATS		importStats;
	jclass					jImportStatsClass = NULL;
	jobject					jImportStats = NULL;
	
	if (!pIStream)
	{
		ThrowError( NE_XFLM_FAILURE, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = pDb->import( pIStream, (FLMUINT)iCollection,
									pNodeToLinkTo, (eNodeInsertLoc)iInsertLoc,
									&importStats)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	// Find the ImportStats class

	if ((jImportStatsClass = pEnv->FindClass( "xflaim/ImportStats")) == NULL)
	{
		goto Exit;
	}

	// Allocate an import stats class.
	
	if ((jImportStats = pEnv->AllocObject( jImportStatsClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the fields in the object
	
	pEnv->SetIntField( jImportStats, fidLines, (jint)importStats.uiLines);
	pEnv->SetIntField( jImportStats, fidChars, (jint)importStats.uiChars);
	pEnv->SetIntField( jImportStats, fidAttributes, (jint)importStats.uiAttributes);
	pEnv->SetIntField( jImportStats, fidElements, (jint)importStats.uiElements);
	pEnv->SetIntField( jImportStats, fidText, (jint)importStats.uiText);
	pEnv->SetIntField( jImportStats, fidDocuments, (jint)importStats.uiDocuments);
	pEnv->SetIntField( jImportStats, fidErrLineNum, (jint)importStats.uiErrLineNum);
	pEnv->SetIntField( jImportStats, fidErrLineOffset, (jint)importStats.uiErrLineOffset);
	pEnv->SetIntField( jImportStats, fidErrorType, (jint)importStats.eErrorType);
	pEnv->SetIntField( jImportStats, fidErrLineFilePos, (jint)importStats.uiErrLineFilePos);
	pEnv->SetIntField( jImportStats, fidErrLineBytes, (jint)importStats.uiErrLineBytes);
	pEnv->SetBooleanField( jImportStats, fidUTF8Encoding,
		(jboolean)(importStats.eXMLEncoding == XFLM_XML_UTF8_ENCODING
					  ? JNI_TRUE
					  : JNI_FALSE));
Exit:

	return( jImportStats);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getFirstDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getFirstDocument( (FLMUINT)iCollection, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getLastDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getLastDocument( (FLMUINT)iCollection, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jint				iFlags,
	jlong				lDocumentId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getDocument( (FLMUINT)iCollection,
									(FLMUINT)iFlags, (FLMUINT64)lDocumentId,
									&pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1documentDone__JIJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lDocumentId)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->documentDone( (FLMUINT)iCollection,
									(FLMUINT64)lDocumentId)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1documentDone__JJ(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lNode)
{
	RCODE				rc = NE_XFLM_OK;
	IF_DOMNode *	pNode = (IF_DOMNode *)((FLMUINT)lNode);
	IF_Db *			pDb = THIS_FDB();

	if (RC_BAD( rc = pDb->documentDone( pNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getNode(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lNodeId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);

	if (RC_BAD( rc = pDb->getNode( (FLMUINT)iCollection, (FLMUINT64)lNodeId,
								   &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;		
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getAttribute(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lElementNodeId,
	jint				iAttrNameId,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);

	if (RC_BAD( rc = pDb->getAttribute( (FLMUINT)iCollection,
									(FLMUINT64)lElementNodeId,
								   (FLMUINT)iAttrNameId, &pNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;		
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1createDocument(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (RC_BAD( rc = pDb->createDocument((FLMUINT)iCollection, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)(FLMUINT)ifpNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1createRootElement(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jint				iTag)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	ifpNewNode = NULL;
	
	if (RC_BAD( rc = pDb->createRootElement((FLMUINT)iCollection,
			(FLMUINT)iTag, &ifpNewNode)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)(FLMUINT)ifpNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE getUniString(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf)
{
	RCODE						rc = NE_XFLM_OK;
	const FLMUNICODE *	puzStr = NULL;
	FLMUINT					uiStrCharCount;
	
	if (sStr)
	{
		puzStr = (const FLMUNICODE *)pEnv->GetStringChars( sStr, NULL);
		uiStrCharCount = (FLMUINT)pEnv->GetStringLength( sStr);
		if (RC_BAD( rc = pDynaBuf->appendData( puzStr,
									sizeof( FLMUNICODE) * uiStrCharCount)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pDynaBuf->appendUniChar( 0)))
		{
			goto Exit;
		}
	}
	else
	{
		pDynaBuf->truncateData( 0);
	}
	
Exit:

	if (puzStr)
	{
		pEnv->ReleaseStringChars( sStr, puzStr);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE getUTF8String(
	JNIEnv *		pEnv,
	jstring		sStr,
	F_DynaBuf *	pDynaBuf)
{
	RCODE				rc = NE_XFLM_OK;
	const char *	pszStr = NULL;
	FLMUINT			uiStrCharCount;
	
	if (sStr)
	{
		pszStr = pEnv->GetStringUTFChars( sStr, NULL);
		uiStrCharCount = (FLMUINT)pEnv->GetStringUTFLength( sStr);
		if (RC_BAD( rc = pDynaBuf->appendData( pszStr, uiStrCharCount)))
		{
			goto Exit;
		}
	}
	else
	{
		pDynaBuf->truncateData( 0);
	}
	if (RC_BAD( rc = pDynaBuf->appendByte( 0)))
	{
		goto Exit;
	}
	
Exit:

	if (pszStr)
	{
		pEnv->ReleaseStringUTFChars( sStr, pszStr);
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createElementDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName,
	jint				iDataType,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createElementDef( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(),
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getElementNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getElementNameId( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createUniqueElmDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sElementName,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucElementNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	elementName( ucElementNameBuf, sizeof( ucElementNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sElementName);
	if (RC_BAD( rc = getUniString( pEnv, sElementName, &elementName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createUniqueElmDef( namespaceURI.getUnicodePtr(),
											elementName.getUnicodePtr(),
											&uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createAttributeDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sAttributeName,
	jint				iDataType,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucAttributeNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	attributeName( ucAttributeNameBuf, sizeof( ucAttributeNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sAttributeName);
	if (RC_BAD( rc = getUniString( pEnv, sAttributeName, &attributeName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createAttributeDef( namespaceURI.getUnicodePtr(),
											attributeName.getUnicodePtr(),
											(FLMUINT)iDataType, &uiNameId, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getAttributeNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sNamespaceURI,
	jstring			sAttributeName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucNamespaceBuf [200];
	FLMBYTE		ucAttributeNameBuf [200];
	F_DynaBuf	namespaceURI( ucNamespaceBuf, sizeof( ucNamespaceBuf));
	F_DynaBuf	attributeName( ucAttributeNameBuf, sizeof( ucAttributeNameBuf));
	
	if (sNamespaceURI)
	{
		if (RC_BAD( rc = getUniString( pEnv, sNamespaceURI, &namespaceURI)))
		{
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	flmAssert( sAttributeName);
	if (RC_BAD( rc = getUniString( pEnv, sAttributeName, &attributeName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getAttributeNameId( namespaceURI.getUnicodePtr(),
											attributeName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createPrefixDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPrefixName,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucPrefixNameBuf [200];
	F_DynaBuf	prefixName( ucPrefixNameBuf, sizeof( ucPrefixNameBuf));
	
	flmAssert( sPrefixName);
	if (RC_BAD( rc = getUniString( pEnv, sPrefixName, &prefixName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createPrefixDef( prefixName.getUnicodePtr(),
														&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getPrefixId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPrefixName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucPrefixNameBuf [200];
	F_DynaBuf	prefixName( ucPrefixNameBuf, sizeof( ucPrefixNameBuf));
	
	flmAssert( sPrefixName);
	if (RC_BAD( rc = getUniString( pEnv, sPrefixName, &prefixName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getPrefixId( prefixName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createEncDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sEncType,
	jstring			sEncName,
	jint				iKeySize,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucEncTypeBuf [200];
	FLMBYTE		ucEncNameBuf [200];
	F_DynaBuf	encType( ucEncTypeBuf, sizeof( ucEncTypeBuf));
	F_DynaBuf	encName( ucEncNameBuf, sizeof( ucEncNameBuf));
	
	flmAssert( sEncType);
	if (RC_BAD( rc = getUniString( pEnv, sEncType, &encType)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sEncName);
	if (RC_BAD( rc = getUniString( pEnv, sEncName, &encName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createEncDef( encType.getUnicodePtr(),
														encName.getUnicodePtr(),
														(FLMUINT)iKeySize,
														&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getEncDefId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sEncName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucEncNameBuf [200];
	F_DynaBuf	encName( ucEncNameBuf, sizeof( ucEncNameBuf));
	
	flmAssert( sEncName);
	if (RC_BAD( rc = getUniString( pEnv, sEncName, &encName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getEncDefId( encName.getUnicodePtr(), &uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1createCollectionDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sCollectionName,
	jint				iEncNumber,
	jint				iRequestedNum)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId = iRequestedNum;
	FLMBYTE		ucCollectionNameBuf [200];
	F_DynaBuf	collectionName( ucCollectionNameBuf, sizeof( ucCollectionNameBuf));
	
	flmAssert( sCollectionName);
	if (RC_BAD( rc = getUniString( pEnv, sCollectionName, &collectionName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->createCollectionDef( collectionName.getUnicodePtr(),
														&uiNameId, (FLMUINT)iEncNumber)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getCollectionNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sCollectionName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucCollectionNameBuf [200];
	F_DynaBuf	collectionName( ucCollectionNameBuf, sizeof( ucCollectionNameBuf));
	
	flmAssert( sCollectionName);
	if (RC_BAD( rc = getUniString( pEnv, sCollectionName, &collectionName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getCollectionNumber( collectionName.getUnicodePtr(),
										&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getIndexNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sIndexName)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNameId;
	FLMBYTE		ucIndexNameBuf [200];
	F_DynaBuf	indexName( ucIndexNameBuf, sizeof( ucIndexNameBuf));
	
	flmAssert( sIndexName);
	if (RC_BAD( rc = getUniString( pEnv, sIndexName, &indexName)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->getIndexNumber( indexName.getUnicodePtr(),
										&uiNameId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	return( (jint)uiNameId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDictionaryDef(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber,
	jlong				lOldNodeRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_DOMNode *	pNewNode = (IF_DOMNode *)(lOldNodeRef
														  ? (IF_DOMNode *)(FLMUINT)lOldNodeRef
														  : NULL);
	
	if (RC_BAD( rc = pDb->getDictionaryDef( (FLMUINT)iDictType,
									(FLMUINT)iDictNumber, &pNewNode)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pNewNode);
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE getDictName(
	IF_Db *			pDb,
	FLMUINT			uiDictType,
	FLMUINT			uiDictNumber,
	FLMBOOL			bGetNamespace,
	F_DynaBuf *		pDynaBuf)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiNameSize = 0;
	FLMUNICODE *	puzName = NULL;
	
	// Determine how much space is needed to get the name.

	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											(FLMUNICODE *)NULL, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
		{
			goto Exit;
		}
	}
		
	// uiNameSize comes back as number of characters, so to
	// get the buffer size needed, we need to add one for a null
	// terminator, and then multiply by the size of a unicode character.
	
	uiNameSize++;
	uiNameSize *= sizeof( FLMUNICODE);
	
	if (RC_BAD( rc = pDynaBuf->allocSpace( uiNameSize, (void **)&puzName)))
	{
		goto Exit;
	}
	
	// Now get the name.
	
	if (bGetNamespace)
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											(FLMUNICODE *)NULL, NULL,
											puzName, &uiNameSize)))
		{
			goto Exit;
		}
	}
	else
	{
		if (RC_BAD( rc = pDb->getDictionaryName( uiDictType, uiDictNumber,
											puzName, &uiNameSize,
											(FLMUNICODE *)NULL, NULL)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getDictionaryName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, (FLMUINT)iDictType, (FLMUINT)iDictNumber,
									FALSE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getElementNamespace(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, ELM_ELEMENT_TAG, (FLMUINT)iDictNumber,
									TRUE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getAttributeNamespace(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictNumber)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	FLMBYTE			ucNameBuf [200];
	F_DynaBuf		nameBuf( ucNameBuf, sizeof( ucNameBuf));
	jstring			jName = NULL;
	
	if (RC_BAD( rc = getDictName( pDb, ELM_ATTRIBUTE_TAG, (FLMUINT)iDictNumber,
									TRUE, &nameBuf)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
		
	// Create a string and return it.

	jName = pEnv->NewString( (const jchar *)nameBuf.getUnicodePtr(),
									(jsize)nameBuf.getUnicodeLength());

Exit:

	return( jName);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1backupBegin(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				eBackupType,
	jint				eTransType,
	jint				iMaxLockWait,
	jlong				lReusedRef)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = THIS_FDB();
	IF_Backup *		ifpBackup = NULL;
	
	if (lReusedRef)
	{
		ifpBackup = (IF_Backup *)(FLMUINT)lReusedRef;
	}
	
	if (RC_BAD( rc = pDb->backupBegin( (eDbBackupType)eBackupType, 
			(eDbTransType)eTransType, (FLMUINT)iMaxLockWait, &ifpBackup)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
		
Exit:

	return( (jlong)(FLMUINT)ifpBackup);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1keyRetrieve(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iIndex,
	jlong				lKey,
	jint				iFlags,
	jlong				lFoundKey)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	pSearchKey = (IF_DataVector *)(FLMUINT)lKey;
	IF_DataVector *	pFoundKey = (IF_DataVector *)(FLMUINT)lFoundKey;
	IF_Db *				pDb = THIS_FDB();
	FLMUINT				uiIndex = (FLMUINT)iIndex;
	FLMUINT				uiFlags = (FLMUINT)iFlags;
	
	if (RC_BAD( rc = pDb->keyRetrieve( uiIndex, pSearchKey, uiFlags, pFoundKey)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1changeItemState(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNum,
	jstring			sState)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBYTE		ucState [80];
	F_DynaBuf	stateBuf( ucState, sizeof( ucState));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sState, &stateBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->changeItemState( (FLMUINT)iDictType,
									(FLMUINT)iDictNum,
									(const char *)stateBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getRflFileName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iFileNum,
	jboolean			bBaseOnly)
{
	IF_Db *		pDb = THIS_FDB();
	char			szRflFileName [F_PATH_MAX_SIZE];
	FLMUINT		uiFileNameBufSize = sizeof( szRflFileName);
	
	pDb->getRflFileName( (FLMUINT)iFileNum,
									(FLMBOOL)(bBaseOnly ? TRUE : FALSE),
									szRflFileName, &uiFileNameBufSize, NULL);
	return( pEnv->NewStringUTF( szRflFileName));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setNextNodeId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCollection,
	jlong				lNextNodeId)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setNextNodeId( (FLMUINT)iCollection,
									(FLMUINT64)lNextNodeId)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setNextDictNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iDictType,
	jint				iDictNumber)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setNextDictNum( (FLMUINT)iDictType,
									(FLMUINT)iDictNumber)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setRflKeepFilesFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bKeep)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setRflKeepFilesFlag( (FLMBOOL)(bKeep ? TRUE : FALSE))))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getRflKeepFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBOOL		bKeep = FALSE;
	
	if (RC_BAD( rc = pDb->getRflKeepFlag( &bKeep)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( bKeep ? JNI_TRUE : JNI_FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setRflDir(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sRflDir)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBYTE		ucDirBuf [200];
	F_DynaBuf	dirBuf( ucDirBuf, sizeof( ucDirBuf));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &dirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->setRflDir( (const char *)dirBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getRflDir(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	char			szRflDir [F_PATH_MAX_SIZE];
	
	pDb->getRflDir( szRflDir);
	return( pEnv->NewStringUTF( szRflDir));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getRflFileNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiFileNum = 0;
	
	if (RC_BAD( rc = pDb->getRflFileNum( &uiFileNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiFileNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getHighestNotUsedRflFileNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiFileNum = 0;
	
	if (RC_BAD( rc = pDb->getHighestNotUsedRflFileNum( &uiFileNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiFileNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setRflFileSizeLimits(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iMinRflSize,
	jint				iMaxRflSize)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setRflFileSizeLimits( (FLMUINT)iMinRflSize,
									(FLMUINT)iMaxRflSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getMinRflFileSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiMinFileSize = 0;
	FLMUINT		uiMaxFileSize = 0;
	
	if (RC_BAD( rc = pDb->getRflFileSizeLimits( &uiMinFileSize, &uiMaxFileSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiMinFileSize);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getMaxRflFileSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiMinFileSize = 0;
	FLMUINT		uiMaxFileSize = 0;
	
	if (RC_BAD( rc = pDb->getRflFileSizeLimits( &uiMinFileSize, &uiMaxFileSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiMaxFileSize);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1rflRollToNextFile(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->rflRollToNextFile()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setKeepAbortedTransInRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bKeep)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setKeepAbortedTransInRflFlag( (FLMBOOL)(bKeep ? TRUE : FALSE))))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getKeepAbortedTransInRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBOOL		bKeep = FALSE;
	
	if (RC_BAD( rc = pDb->getKeepAbortedTransInRflFlag( &bKeep)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jboolean)(bKeep ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setAutoTurnOffKeepRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jboolean			bAutoTurnOff)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->setAutoTurnOffKeepRflFlag( (FLMBOOL)(bAutoTurnOff ? TRUE : FALSE))))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_Db__1getAutoTurnOffKeepRflFlag(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBOOL		bAutoTurnOff = FALSE;
	
	if (RC_BAD( rc = pDb->getAutoTurnOffKeepRflFlag( &bAutoTurnOff)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jboolean)(bAutoTurnOff ? JNI_TRUE : JNI_FALSE));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setFileExtendSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iFileExtendSize)
{
	IF_Db *		pDb = THIS_FDB();
	
	pDb->setFileExtendSize( (FLMUINT)iFileExtendSize);
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getFileExtendSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getFileExtendSize());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getDbVersion(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getDbVersion());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getBlockSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getBlockSize());
}
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getDefaultLanguage(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jint)pDb->getDefaultLanguage());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getTransID(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	
	return( (jlong)pDb->getTransID());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_Db__1getDbControlFileName(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	char			szFileName [F_PATH_MAX_SIZE];
	
	szFileName [0] = 0;
	if (RC_BAD( rc = pDb->getDbControlFileName( szFileName, sizeof( szFileName))))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( pEnv->NewStringUTF( szFileName));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getLastBackupTransID(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64LastBackupTransID = 0;
	
	if (RC_BAD( rc = pDb->getLastBackupTransID( &ui64LastBackupTransID)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64LastBackupTransID);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getBlocksChangedSinceBackup(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiBlocksChangedSinceBackup = 0;
	
	if (RC_BAD( rc = pDb->getBlocksChangedSinceBackup( &uiBlocksChangedSinceBackup)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiBlocksChangedSinceBackup);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getNextIncBackupSequenceNum(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT		uiNextIncBackupSequenceNum = 0;
	
	if (RC_BAD( rc = pDb->getNextIncBackupSequenceNum( &uiNextIncBackupSequenceNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jint)uiNextIncBackupSequenceNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceDataSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64Size = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( &ui64Size, NULL, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64Size);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceRollbackSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64Size = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( NULL, &ui64Size, NULL)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64Size);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceRflSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64Size = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( NULL, NULL, &ui64Size)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)ui64Size);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Db__1getDiskSpaceTotalSize(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMUINT64	ui64DataSize = 0;
	FLMUINT64	ui64RollbackSize = 0;
	FLMUINT64	ui64RflSize = 0;
	
	if (RC_BAD( rc = pDb->getDiskSpaceUsage( &ui64DataSize, &ui64RollbackSize,
										&ui64RflSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(ui64DataSize + ui64RollbackSize + ui64RflSize));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getMustCloseRC(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	return( (jint)pDb->getMustCloseRC());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_Db__1getAbortRC(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	return( (jint)pDb->getAbortRC());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1setMustAbortTrans(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iRc)
{
	IF_Db *		pDb = THIS_FDB();
	pDb->setMustAbortTrans( (RCODE)iRc);
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1enableEncryption(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->enableEncryption()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1wrapKey(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPassword)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	FLMBYTE		ucPassword [200];
	F_DynaBuf	passwordBuf( ucPassword, sizeof( ucPassword));
	
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pDb->wrapKey( (const char *)passwordBuf.getBufferPtr())))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}
		
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Db__1rollOverDbKey(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Db *		pDb = THIS_FDB();
	
	if (RC_BAD( rc = pDb->rollOverDbKey()))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_Db__1getSerialNumber(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *		pDb = THIS_FDB();
	char			ucSerialNumber [XFLM_SERIAL_NUM_SIZE];
	jbyteArray	jSerialNumber;
	
	pDb->getSerialNumber( ucSerialNumber);
	
	if ((jSerialNumber = pEnv->NewByteArray( (jsize)XFLM_SERIAL_NUM_SIZE)) == NULL)
	{
		goto Exit;
	}
	
   pEnv->SetByteArrayRegion( jSerialNumber, (jsize)0, (jsize)XFLM_SERIAL_NUM_SIZE,
					(const jbyte *)ucSerialNumber);

Exit:

	return( jSerialNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jobject JNICALL Java_xflaim_Db__1getCheckpointInfo(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Db *					pDb = THIS_FDB();
	XFLM_CHECKPOINT_INFO	checkpointInfo;
	jclass					jCheckpointInfoClass = NULL;
	jobject					jCheckpointInfo = NULL;
	
	pDb->getCheckpointInfo( &checkpointInfo);

	// Find the IndexStatus class

	if ((jCheckpointInfoClass = pEnv->FindClass( "xflaim/CheckpointInfo")) == NULL)
	{
		goto Exit;
	}

	// Allocate an index status class.
	
	if ((jCheckpointInfo = pEnv->AllocObject( jCheckpointInfoClass)) == NULL)
	{
		goto Exit;
	}
	
	// Set the fields in the object
	
	pEnv->SetBooleanField( jCheckpointInfo, fidRunning,
		(jboolean)(checkpointInfo.bRunning ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCheckpointInfo, fidRunningTime,
		(jint)checkpointInfo.uiRunningTime);
	pEnv->SetBooleanField( jCheckpointInfo, fidForcingCheckpoint,
		(jboolean)(checkpointInfo.bForcingCheckpoint ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCheckpointInfo, fidForceCheckpointRunningTime,
		(jint)checkpointInfo.uiForceCheckpointRunningTime);
	pEnv->SetIntField( jCheckpointInfo, fidForceCheckpointReason,
		(jint)checkpointInfo.iForceCheckpointReason);
	pEnv->SetBooleanField( jCheckpointInfo, fidWritingDataBlocks,
		(jboolean)(checkpointInfo.bWritingDataBlocks ? JNI_TRUE : JNI_FALSE));
	pEnv->SetIntField( jCheckpointInfo, fidLogBlocksWritten,
		(jint)checkpointInfo.uiLogBlocksWritten);
	pEnv->SetIntField( jCheckpointInfo, fidDataBlocksWritten,
		(jint)checkpointInfo.uiDataBlocksWritten);
	pEnv->SetIntField( jCheckpointInfo, fidDirtyCacheBytes,
		(jint)checkpointInfo.uiDirtyCacheBytes);
	pEnv->SetIntField( jCheckpointInfo, fidBlockSize,
		(jint)checkpointInfo.uiBlockSize);
	pEnv->SetIntField( jCheckpointInfo, fidWaitTruncateTime,
		(jint)checkpointInfo.uiWaitTruncateTime);
	
Exit:

	return( jCheckpointInfo);
}

