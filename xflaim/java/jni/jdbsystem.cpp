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

#include "xflaim_DbSystem.h"
#include "xflaim_CREATEOPTS.h"
#include "xflaim_SlabUsage.h"
#include "xflaim_CacheUsage.h"
#include "xflaim_CacheInfo.h"
#include "flaimsys.h"
#include "jniftk.h"
#include "jnirestore.h"
#include "jnistatus.h"

#define THIS_DBSYS() \
	((F_DbSystem *)(FLMUINT)lThis)
	
// Field IDs for the CREATEOPTS class.

static jfieldID	CREATEOPTS_fidBlockSize = NULL;
static jfieldID	CREATEOPTS_fidVersionNum = NULL;
static jfieldID	CREATEOPTS_fidMinRflFileSize = NULL;
static jfieldID	CREATEOPTS_fidMaxRflFileSize = NULL;
static jfieldID	CREATEOPTS_fidKeepRflFiles = NULL;
static jfieldID	CREATEOPTS_fidLogAbortedTransToRfl = NULL;
static jfieldID	CREATEOPTS_fidDefaultLanguage = NULL;

// field IDs for the SlabUsage class.

static jfieldID	SlabUsage_fidSlabs;
static jfieldID	SlabUsage_fidSlabBytes;
static jfieldID	SlabUsage_fidAllocatedCells;
static jfieldID	SlabUsage_fidFreeCells;

// field IDs for the CacheUsage class.

static jfieldID	CacheUsage_fidByteCount = NULL;
static jfieldID	CacheUsage_fidCount = NULL;
static jfieldID	CacheUsage_fidOldVerCount = NULL;
static jfieldID	CacheUsage_fidOldVerBytes = NULL;
static jfieldID	CacheUsage_fidCacheHits = NULL;
static jfieldID	CacheUsage_fidCacheHitLooks = NULL;
static jfieldID	CacheUsage_fidCacheFaults = NULL;
static jfieldID	CacheUsage_fidCacheFaultLooks = NULL;
static jfieldID	CacheUsage_fidSlabUsage = NULL;

// field IDs for the CacheInfo class.

static jfieldID	CacheInfo_fidMaxBytes = NULL;
static jfieldID	CacheInfo_fidTotalBytesAllocated = NULL;
static jfieldID	CacheInfo_fidDynamicCacheAdjust = NULL;
static jfieldID	CacheInfo_fidCacheAdjustPercent = NULL;
static jfieldID	CacheInfo_fidCacheAdjustMin = NULL;
static jfieldID	CacheInfo_fidCacheAdjustMax = NULL;
static jfieldID	CacheInfo_fidCacheAdjustMinToLeave = NULL;
static jfieldID	CacheInfo_fidDirtyCount = NULL;
static jfieldID	CacheInfo_fidDirtyBytes = NULL;
static jfieldID	CacheInfo_fidNewCount = NULL;
static jfieldID	CacheInfo_fidNewBytes = NULL;
static jfieldID	CacheInfo_fidLogCount = NULL;
static jfieldID	CacheInfo_fidLogBytes = NULL;
static jfieldID	CacheInfo_fidFreeCount = NULL;
static jfieldID	CacheInfo_fidFreeBytes = NULL;
static jfieldID	CacheInfo_fidReplaceableCount = NULL;
static jfieldID	CacheInfo_fidReplaceableBytes = NULL;
static jfieldID	CacheInfo_fidPreallocatedCache = NULL;
static jfieldID	CacheInfo_fidBlockCache = NULL;
static jfieldID	CacheInfo_fidNodeCache = NULL;
	
FSTATIC void getCreateOpts(
	JNIEnv *					pEnv,
	jobject					createOpts,
	XFLM_CREATE_OPTS *	pCreateOpts);
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1createDbSystem(
	JNIEnv *				pEnv,
	jobject)				// obj)
{
	IF_DbSystem * 		pDbSystem;
	
	if( RC_BAD( FlmAllocDbSystem( &pDbSystem)))
	{
		ThrowError( NE_XFLM_MEM, pEnv);
	}
	
	return( (jlong)(FLMUINT)pDbSystem);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_SlabUsage_initIDs(
	JNIEnv *	pEnv,
	jclass	jSlabUsageClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((SlabUsage_fidSlabs = pEnv->GetFieldID( jSlabUsageClass,
								"lSlabs", "J")) == NULL)
	{
		goto Exit;
	}
	if ((SlabUsage_fidSlabBytes = pEnv->GetFieldID( jSlabUsageClass,
								"lSlabBytes", "J")) == NULL)
	{
		goto Exit;
	}
	if ((SlabUsage_fidAllocatedCells = pEnv->GetFieldID( jSlabUsageClass,
								"lAllocatedCells", "J")) == NULL)
	{
		goto Exit;
	}
	if ((SlabUsage_fidFreeCells = pEnv->GetFieldID( jSlabUsageClass,
								"lFreeCells", "J")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CacheUsage_initIDs(
	JNIEnv *	pEnv,
	jclass	jCacheUsageClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((CacheUsage_fidByteCount = pEnv->GetFieldID( jCacheUsageClass,
								"iByteCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidCount = pEnv->GetFieldID( jCacheUsageClass,
								"iCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidOldVerCount = pEnv->GetFieldID( jCacheUsageClass,
								"iOldVerCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidOldVerBytes = pEnv->GetFieldID( jCacheUsageClass,
								"iOldVerBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidCacheHits = pEnv->GetFieldID( jCacheUsageClass,
								"iCacheHits", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidCacheHitLooks = pEnv->GetFieldID( jCacheUsageClass,
								"iCacheHitLooks", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidCacheFaults = pEnv->GetFieldID( jCacheUsageClass,
							"iCacheFaults", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidCacheFaultLooks = pEnv->GetFieldID( jCacheUsageClass,
								"iCacheFaultLooks", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheUsage_fidSlabUsage = pEnv->GetFieldID( jCacheUsageClass,
								"slabUsage", "Lxflaim/SlabUsage;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CacheInfo_initIDs(
	JNIEnv *	pEnv,
	jclass	jCacheInfoClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((CacheInfo_fidMaxBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iMaxBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidTotalBytesAllocated = pEnv->GetFieldID( jCacheInfoClass,
								"iTotalBytesAllocated", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidDynamicCacheAdjust = pEnv->GetFieldID( jCacheInfoClass,
								"bDynamicCacheAdjust", "V")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidCacheAdjustPercent = pEnv->GetFieldID( jCacheInfoClass,
								"iCacheAdjustPercent", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidCacheAdjustMin = pEnv->GetFieldID( jCacheInfoClass,
								"iCacheAdjustMin", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidCacheAdjustMax = pEnv->GetFieldID( jCacheInfoClass,
								"iCacheAdjustMax", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidCacheAdjustMinToLeave = pEnv->GetFieldID( jCacheInfoClass,
							"iCacheAdjustMinToLeave", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidDirtyCount = pEnv->GetFieldID( jCacheInfoClass,
								"iDirtyCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidDirtyBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iDirtyBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidNewCount = pEnv->GetFieldID( jCacheInfoClass,
								"iNewCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidNewBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iNewBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidLogCount = pEnv->GetFieldID( jCacheInfoClass,
								"iLogCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidLogBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iLogBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidFreeCount = pEnv->GetFieldID( jCacheInfoClass,
								"iFreeCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidFreeBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iFreeBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidReplaceableCount = pEnv->GetFieldID( jCacheInfoClass,
								"iReplaceableCount", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidReplaceableBytes = pEnv->GetFieldID( jCacheInfoClass,
								"iReplaceableBytes", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidPreallocatedCache = pEnv->GetFieldID( jCacheInfoClass,
								"bPreallocatedCache", "V")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidBlockCache = pEnv->GetFieldID( jCacheInfoClass,
								"BlockCache", "Lxflaim/CacheUsage;")) == NULL)
	{
		goto Exit;
	}
	if ((CacheInfo_fidNodeCache = pEnv->GetFieldID( jCacheInfoClass,
								"NodeCache", "Lxflaim/CacheUsage;")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_CREATEOPTS_initIDs(
	JNIEnv *	pEnv,
	jclass	jCREATEOPTSClass)
{
	
	// Get the field IDs for the fields in the class.
	
	if ((CREATEOPTS_fidBlockSize = pEnv->GetFieldID( jCREATEOPTSClass,
								"iBlockSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CREATEOPTS_fidVersionNum = pEnv->GetFieldID( jCREATEOPTSClass,
								"iVersionNum", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CREATEOPTS_fidMinRflFileSize = pEnv->GetFieldID( jCREATEOPTSClass,
								"iMinRflFileSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CREATEOPTS_fidMaxRflFileSize = pEnv->GetFieldID( jCREATEOPTSClass,
								"iMaxRflFileSize", "I")) == NULL)
	{
		goto Exit;
	}
	if ((CREATEOPTS_fidKeepRflFiles = pEnv->GetFieldID( jCREATEOPTSClass,
								"bKeepRflFiles", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((CREATEOPTS_fidLogAbortedTransToRfl = pEnv->GetFieldID( jCREATEOPTSClass,
							"bLogAbortedTransToRfl", "Z")) == NULL)
	{
		goto Exit;
	}
	if ((CREATEOPTS_fidDefaultLanguage = pEnv->GetFieldID( jCREATEOPTSClass,
								"iDefaultLanguage", "I")) == NULL)
	{
		goto Exit;
	}
	
Exit:

	return;
}

/****************************************************************************
Desc:	Get create options from the CREATEOPTS Java object.
****************************************************************************/
FSTATIC void getCreateOpts(
	JNIEnv *					pEnv,
	jobject					createOpts,
	XFLM_CREATE_OPTS *	pCreateOpts)
{
	pCreateOpts->uiBlockSize = (FLMUINT)pEnv->GetIntField( createOpts,
			CREATEOPTS_fidBlockSize); 
	pCreateOpts->uiVersionNum = (FLMUINT)pEnv->GetIntField( createOpts,
			CREATEOPTS_fidVersionNum);
	pCreateOpts->uiMinRflFileSize = (FLMUINT)pEnv->GetIntField( createOpts,
			CREATEOPTS_fidMinRflFileSize); 
	pCreateOpts->uiMaxRflFileSize = (FLMUINT)pEnv->GetIntField( createOpts,
			CREATEOPTS_fidMaxRflFileSize); 
	pCreateOpts->bKeepRflFiles = (FLMBOOL)(pEnv->GetBooleanField( createOpts,
			CREATEOPTS_fidKeepRflFiles) ? TRUE : FALSE); 
	pCreateOpts->bLogAbortedTransToRfl = (FLMBOOL)(pEnv->GetBooleanField( createOpts,
			CREATEOPTS_fidLogAbortedTransToRfl) ? TRUE : FALSE); 
	pCreateOpts->uiDefaultLanguage = (FLMUINT)pEnv->GetIntField( createOpts,
			CREATEOPTS_fidDefaultLanguage);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbCreate(
	JNIEnv *					pEnv,
	jobject,					// obj,
	jlong						lThis,
	jstring					sDbPath,
	jstring					sDataDir,
	jstring					sRflDir,
	jstring					sDictFileName,
	jstring					sDictBuf,
	jobject					CreateOpts)
{
	RCODE						rc = NE_XFLM_OK;
	F_Db *					pDb = NULL;
	XFLM_CREATE_OPTS		Opts;
	XFLM_CREATE_OPTS *	pOpts;
	FLMBYTE					ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf				dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE					ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf				dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE					ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf				rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE					ucDictFileName [F_PATH_MAX_SIZE];
	F_DynaBuf				dictFileNameBuf( ucDictFileName, sizeof( ucDictFileName));
	FLMBYTE					ucDictBuf [100];
	F_DynaBuf				dictBufBuf( ucDictBuf, sizeof( ucDictBuf));
	
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDictFileName, &dictFileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDictBuf, &dictBufBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (!CreateOpts)
	{
		pOpts = NULL;
	}
	else
	{
		getCreateOpts( pEnv, CreateOpts, &Opts);
		pOpts = &Opts;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCreate(
							(const char *)dbPathBuf.getBufferPtr(),
							dataDirBuf.getDataLength() > 1
							? (const char *)dataDirBuf.getBufferPtr()
							: (const char *)NULL,
							rflDirBuf.getDataLength() > 1
							? (const char *)rflDirBuf.getBufferPtr()
							: (const char *)NULL,
							dictFileNameBuf.getDataLength() > 1
							? (const char *)dictFileNameBuf.getBufferPtr()
							: (const char *)NULL,
							dictBufBuf.getDataLength() > 1
							? (const char *)dictBufBuf.getBufferPtr()
							: (const char *)NULL,
							pOpts, (IF_Db **)&pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

 Exit:

  	return( (jlong)((FLMUINT)pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbOpen(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sPassword,
	jboolean			bAllowLimited)
{
	RCODE 			rc = NE_XFLM_OK;
	F_Db * 			pDb = NULL;
	FLMBYTE			ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf		dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE			ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf		dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE			ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf		rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE			ucPassword [100];
	F_DynaBuf		passwordBuf( ucPassword, sizeof( ucPassword));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
 	if (RC_BAD( rc = THIS_DBSYS()->dbOpen(
							(const char *)dbPathBuf.getBufferPtr(),
							dataDirBuf.getDataLength() > 1
							? (const char *)dataDirBuf.getBufferPtr()
							: (const char *)NULL,
							rflDirBuf.getDataLength() > 1
							? (const char *)rflDirBuf.getBufferPtr()
							: (const char *)NULL,
							passwordBuf.getDataLength() > 1
							? (const char *)passwordBuf.getBufferPtr()
							: (const char *)NULL,
							bAllowLimited ? TRUE : FALSE,
							(IF_Db **)&pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
 	
Exit:

	return( (jlong)(FLMUINT)pDb);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRemove(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jboolean			bRemove)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE			ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf		dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE			ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf		dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE			ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf		rflDirBuf( ucRflDir, sizeof( ucRflDir));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbRemove(
							(const char *)dbPathBuf.getBufferPtr(),
							dataDirBuf.getDataLength() > 1
							? (const char *)dataDirBuf.getBufferPtr()
							: (const char *)NULL,
							rflDirBuf.getDataLength() > 1
							? (const char *)rflDirBuf.getBufferPtr()
							: (const char *)NULL,
							bRemove ? TRUE : FALSE)))
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
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRestore(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sBackupPath,
	jstring			sPassword,
	jobject			RestoreClient,
	jobject			RestoreStatus)
{
	RCODE						rc = NE_XFLM_OK;
	JavaVM *					pJvm = NULL;
	JNIRestoreClient *	pRestoreClient = NULL;
	JNIRestoreStatus *	pRestoreStatus = NULL;
	FLMBYTE					ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf				dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE					ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf				dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE					ucBackupPath [F_PATH_MAX_SIZE];
	F_DynaBuf				backupPathBuf( ucBackupPath, sizeof( ucBackupPath));
	FLMBYTE					ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf				rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE					ucPassword [100];
	F_DynaBuf				passwordBuf( ucPassword, sizeof( ucPassword));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBackupPath, &backupPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	pEnv->GetJavaVM( &pJvm);

	flmAssert( RestoreClient);
	if ((pRestoreClient = f_new JNIRestoreClient( RestoreClient, pJvm)) == NULL)
	{
		ThrowError( NE_XFLM_MEM, pEnv);
		goto Exit;
	}
	
	if (RestoreStatus != NULL)
	{
		if ((pRestoreStatus = f_new JNIRestoreStatus( RestoreStatus, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;
		}		
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbRestore(
		(const char *)dbPathBuf.getBufferPtr(),
		dataDirBuf.getDataLength() > 1
		? (const char *)dataDirBuf.getBufferPtr()
		: (const char *)NULL,
		backupPathBuf.getDataLength() > 1
		? (const char *)backupPathBuf.getBufferPtr()
		: (const char *)NULL,
		rflDirBuf.getDataLength() > 1
		? (const char *)rflDirBuf.getBufferPtr()
		: (const char *)NULL,
		passwordBuf.getDataLength() > 1
		? (const char *)passwordBuf.getBufferPtr()
		: (const char *)NULL,
		pRestoreClient, pRestoreStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pRestoreClient)
	{
		pRestoreClient->Release();
	}
	
	if (pRestoreStatus)
	{
		pRestoreStatus->Release();
	}
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRename(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sNewDbName,
	jboolean			bOverwriteDestOk,
	jobject			Status)
{
	RCODE						rc = NE_XFLM_OK;
	JavaVM *					pJvm;
	JNIRenameStatus *		pStatus = NULL;
	FLMBYTE					ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf				dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE					ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf				dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE					ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf				rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE					ucNewDbName [F_PATH_MAX_SIZE];
	F_DynaBuf				newDbNameBuf( ucNewDbName, sizeof( ucNewDbName));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sNewDbName);
	if (RC_BAD( rc = getUTF8String( pEnv, sNewDbName, &newDbNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (Status != NULL)
	{
		pEnv->GetJavaVM( &pJvm);
		if ((pStatus = f_new JNIRenameStatus( Status, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;	
		}
	}

	if (RC_BAD(rc = THIS_DBSYS()->dbRename(
		(const char *)dbPathBuf.getBufferPtr(),
		dataDirBuf.getDataLength() > 1
		? (const char *)dataDirBuf.getBufferPtr()
		: (const char *)NULL,
		rflDirBuf.getDataLength() > 1
		? (const char *)rflDirBuf.getBufferPtr()
		: (const char *)NULL,
		(const char *)newDbNameBuf.getBufferPtr(),
		bOverwriteDestOk ? TRUE : FALSE, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbCopy(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sSrcDbName,
	jstring			sSrcDataDir,
	jstring			sSrcRflDir,
	jstring			sDestDbName,
	jstring			sDestDataDir,
	jstring			sDestRflDir,
	jobject			Status)
{
	RCODE					rc = NE_XFLM_OK;
	JavaVM *				pJvm;
	JNICopyStatus *	pStatus = NULL;
	FLMBYTE				ucSrcDbName [F_PATH_MAX_SIZE];
	F_DynaBuf			srcDbNameBuf( ucSrcDbName, sizeof( ucSrcDbName));
	FLMBYTE				ucSrcDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf			srcDataDirBuf( ucSrcDataDir, sizeof( ucSrcDataDir));
	FLMBYTE				ucSrcRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf			srcRflDirBuf( ucSrcRflDir, sizeof( ucSrcRflDir));
	FLMBYTE				ucDestDbName [F_PATH_MAX_SIZE];
	F_DynaBuf			destDbNameBuf( ucDestDbName, sizeof( ucDestDbName));
	FLMBYTE				ucDestDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf			destDataDirBuf( ucDestDataDir, sizeof( ucDestDataDir));
	FLMBYTE				ucDestRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf			destRflDirBuf( ucDestRflDir, sizeof( ucDestRflDir));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sSrcDbName);
	if (RC_BAD( rc = getUTF8String( pEnv, sSrcDbName, &srcDbNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSrcDataDir, &srcDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSrcRflDir, &srcRflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	flmAssert( sDestDbName);
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDbName, &destDbNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDataDir, &destDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestRflDir, &destRflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (Status)
	{
		pEnv->GetJavaVM( &pJvm);
		if ( (pStatus = f_new JNICopyStatus( Status, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCopy(
		(const char *)srcDbNameBuf.getBufferPtr(),
		srcDataDirBuf.getDataLength() > 1
		? (const char *)srcDataDirBuf.getBufferPtr()
		: (const char *)NULL,
		srcRflDirBuf.getDataLength() > 1
		? (const char *)srcRflDirBuf.getBufferPtr()
		: (const char *)NULL,
		(const char *)destDbNameBuf.getBufferPtr(),
		destDataDirBuf.getDataLength() > 1
		? (const char *)destDataDirBuf.getBufferPtr()
		: (const char *)NULL,
		destRflDirBuf.getDataLength() > 1
		? (const char *)destRflDirBuf.getBufferPtr()
		: (const char *)NULL,
		pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}

Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbCheck(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbPath,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sPassword,
	jint				iFlags,
	jobject			Status)
{
	RCODE					rc = NE_XFLM_OK;
	JNICheckStatus *	pStatus = NULL;
	F_DbInfo *			pDbInfo = NULL;
	FLMBYTE				ucDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf			dbPathBuf( ucDbPath, sizeof( ucDbPath));
	FLMBYTE				ucDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf			dataDirBuf( ucDataDir, sizeof( ucDataDir));
	FLMBYTE				ucRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf			rflDirBuf( ucRflDir, sizeof( ucRflDir));
	FLMBYTE				ucPassword [100];
	F_DynaBuf			passwordBuf( ucPassword, sizeof( ucPassword));
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sDbPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sDbPath, &dbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDataDir, &dataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sRflDir, &rflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (Status != NULL)
	{
		JavaVM *		pJvm = NULL;
		
		pEnv->GetJavaVM( &pJvm);
		
		if ((pStatus = f_new JNICheckStatus( Status, pJvm)) == NULL)
		{
			ThrowError( NE_XFLM_MEM, pEnv);
			goto Exit;
		}		
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCheck(
		(const char *)dbPathBuf.getBufferPtr(),
		dataDirBuf.getDataLength() > 1
		? (const char *)dataDirBuf.getBufferPtr()
		: (const char *)NULL,
		rflDirBuf.getDataLength() > 1
		? (const char *)rflDirBuf.getBufferPtr()
		: (const char *)NULL,
		passwordBuf.getDataLength() > 1
		? (const char *)passwordBuf.getBufferPtr()
		: (const char *)NULL,
		(FLMUINT)iFlags, (IF_DbInfo **)&pDbInfo, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	
	return (jlong)(FLMUINT)pDbInfo;	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBufferIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sBuffer)
{
	RCODE					rc = NE_XFLM_OK;
	const char *		pszBuffer = NULL;
	FLMUINT				uiStrCharCount;
	F_BufferIStream *	pIStream = NULL;
	char *				pszAllocBuffer = NULL;
	
	// Get a pointer to the characters in the string.
	
	flmAssert( sBuffer);
	pszBuffer = pEnv->GetStringUTFChars( sBuffer, NULL);
	uiStrCharCount = (FLMUINT)pEnv->GetStringUTFLength( sBuffer);
	flmAssert( uiStrCharCount);
	
	// Create the buffer stream object.
	
	if ((pIStream = f_new F_BufferIStream) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Call the openStream method so that it will allocate a buffer
	// internally.  Add one to the size so that we allocate space for
	// a null terminating byte - because uiStrCharCount does NOT include
	// the null terminating byte.  Buffer pointer is returned in pucBuffer.
	
	if( RC_BAD( rc = pIStream->openStream( NULL, uiStrCharCount + 1, &pszAllocBuffer)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Copy the data from the passed in string into pucBuffer, including the NULL.
	
	f_memcpy( pszAllocBuffer, pszBuffer, uiStrCharCount);
	
	// NULL terminate the allocated buffer.
	
	pszAllocBuffer [uiStrCharCount] = 0;
	
Exit:

	if (pszBuffer)
	{
		pEnv->ReleaseStringUTFChars( sBuffer, pszBuffer);
	}

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openFileIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPath)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE				ucPath [F_PATH_MAX_SIZE];
	F_DynaBuf			pathBuf( ucPath, sizeof( ucPath));
	IF_PosIStream *	pIStream = NULL;
 
	// Get all of the string parameters into buffers.
	
	flmAssert( sPath);
	if (RC_BAD( rc = getUTF8String( pEnv, sPath, &pathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openFileIStream(
								(const char *)pathBuf.getBufferPtr(), &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)pIStream);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1createJDataVector(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis)
{
	RCODE					rc = NE_XFLM_OK;
	IF_DataVector *	ifpDataVector = NULL;
	
	if (RC_BAD( rc = THIS_DBSYS()->createIFDataVector( &ifpDataVector)))
	{
		ThrowError(rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)(FLMUINT)ifpDataVector);
}

/****************************************************************************
Desc: Rebuild status callback
****************************************************************************/
class JavaDbRebuildStatus : public IF_DbRebuildStatus
{
public:

	JavaDbRebuildStatus(
		JNIEnv *		pEnv,
		jobject		jDbRebuildStatusObject)
	{
		m_pEnv = pEnv;
		
		// Get a global reference to keep the object from being garbage
		// collected, and to allow it to be called across invocations into
		// the native interface.  Otherwise, the reference will be lost and
		// cannot be used by the callback function.
		
		m_jDbRebuildStatusObject = pEnv->NewGlobalRef( jDbRebuildStatusObject);
		m_jReportRebuildMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jDbRebuildStatusObject),
													"reportRebuild",
													"(IZJJJJJ)I");
		m_jReportRebuildErrMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jDbRebuildStatusObject),
													"reportRebuildErr",
													"(IIIIIIIIJ)I");
	}
	
	virtual ~JavaDbRebuildStatus()
	{
		if (m_jDbRebuildStatusObject)
		{
			m_pEnv->DeleteGlobalRef( m_jDbRebuildStatusObject);
		}
	}
			
	RCODE FLMAPI reportRebuild(
		XFLM_REBUILD_INFO *	pRebuild)
	{
		
		// VERY IMPORTANT NOTE!  m_pEnv points to the environment that was
		// passed in when this object was set up.  It is thread-specific, so
		// it is important that the callback happen inside the same thread
		// where the setIndexingStatusObject method was called.  It will not
		// work to set the index status object in one thread, but then do
		// the index operation in another thread.
		
		return( (RCODE)m_pEnv->CallIntMethod( m_jDbRebuildStatusObject,
									m_jReportRebuildMethodId,
									(jint)pRebuild->iDoingFlag,
									(jboolean)(pRebuild->bStartFlag ? JNI_TRUE : JNI_FALSE),
									(jlong)pRebuild->ui64FileSize,
									(jlong)pRebuild->ui64BytesExamined,
									(jlong)pRebuild->ui64TotNodes,
									(jlong)pRebuild->ui64NodesRecov,
									(jlong)pRebuild->ui64DiscardedDocs));
	}
	
	RCODE FLMAPI reportRebuildErr(
		XFLM_CORRUPT_INFO *	pCorruptInfo)
	{
		return( (RCODE)m_pEnv->CallIntMethod( m_jDbRebuildStatusObject,
									m_jReportRebuildErrMethodId,
									(jint)pCorruptInfo->iErrCode,
									(jint)pCorruptInfo->uiErrLocale,
									(jint)pCorruptInfo->uiErrLfNumber,
									(jint)pCorruptInfo->uiErrLfType,
									(jint)pCorruptInfo->uiErrBTreeLevel,
									(jint)pCorruptInfo->uiErrBlkAddress,
									(jint)pCorruptInfo->uiErrParentBlkAddress,
									(jint)pCorruptInfo->uiErrElmOffset,
									(jlong)pCorruptInfo->ui64ErrNodeId));
	}
	
private:

	JNIEnv *		m_pEnv;
	jobject		m_jDbRebuildStatusObject;
	jmethodID	m_jReportRebuildMethodId;
	jmethodID	m_jReportRebuildErrMethodId;
};

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRebuild(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis,
	jstring			sSourceDbPath,
	jstring			sSourceDataDir,
	jstring			sDestDbPath,
	jstring			sDestDataDir,
	jstring			sDestRflDir,
	jstring			sDictPath,
	jstring			sPassword,
	jobject			createOpts,
	jobject			jDbRebuildStatusObj)
{
	RCODE							rc = NE_XFLM_OK;
	JavaDbRebuildStatus *	pDbRebuildStatusObj = NULL;
	F_DbSystem *				pDbSystem = THIS_DBSYS();
	XFLM_CREATE_OPTS			createOptions;
	XFLM_CREATE_OPTS *		pCreateOptions;
	FLMUINT64					ui64TotNodes;
	FLMUINT64					ui64NodesRecov;
	FLMUINT64					ui64DiscardedDocs;
	FLMBYTE						ucSourceDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf					sourceDbPathBuf( ucSourceDbPath, sizeof( ucSourceDbPath));
	FLMBYTE						ucSourceDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf					sourceDataDirBuf( ucSourceDataDir, sizeof( ucSourceDataDir));
	FLMBYTE						ucDestDbPath [F_PATH_MAX_SIZE];
	F_DynaBuf					destDbPathBuf( ucDestDbPath, sizeof( ucDestDbPath));
	FLMBYTE						ucDestDataDir [F_PATH_MAX_SIZE];
	F_DynaBuf					destDataDirBuf( ucDestDataDir, sizeof( ucDestDataDir));
	FLMBYTE						ucDestRflDir [F_PATH_MAX_SIZE];
	F_DynaBuf					destRflDirBuf( ucDestRflDir, sizeof( ucDestRflDir));
	FLMBYTE						ucDictPath [F_PATH_MAX_SIZE];
	F_DynaBuf					dictPathBuf( ucDictPath, sizeof( ucDictPath));
	FLMBYTE						ucPassword [100];
	F_DynaBuf					passwordBuf( ucPassword, sizeof( ucPassword));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sSourceDbPath, &sourceDbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sSourceDataDir, &sourceDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDbPath, &destDbPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestDataDir, &destDataDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDestRflDir, &destRflDirBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sDictPath, &dictPathBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sPassword, &passwordBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Setup callback object, if one was passed in
	
	if (jDbRebuildStatusObj)
	{
		if ((pDbRebuildStatusObj = f_new JavaDbRebuildStatus( pEnv,
													jDbRebuildStatusObj)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	// Set up the create options.
	
	if (!createOpts)
	{
		pCreateOptions = NULL;
	}
	else
	{
		getCreateOpts( pEnv, createOpts, &createOptions);
		pCreateOptions = &createOptions;
	}
	
	// Call the rebuild function.
	
	if (RC_BAD( rc = pDbSystem->dbRebuild(
				(const char *)sourceDbPathBuf.getBufferPtr(),
				sourceDataDirBuf.getDataLength() > 1
				? (const char *)sourceDataDirBuf.getBufferPtr()
				: (const char *)NULL,
				(const char *)destDbPathBuf.getBufferPtr(),
				destDataDirBuf.getDataLength() > 1
				? (const char *)destDataDirBuf.getBufferPtr()
				: (const char *)NULL,
				destRflDirBuf.getDataLength() > 1
				? (const char *)destRflDirBuf.getBufferPtr()
				: (const char *)NULL,
				dictPathBuf.getDataLength() > 1
				? (const char *)dictPathBuf.getBufferPtr()
				: (const char *)NULL,
				passwordBuf.getDataLength() > 1
				? (const char *)passwordBuf.getBufferPtr()
				: (const char *)NULL,
				pCreateOptions,
				&ui64TotNodes,
				&ui64NodesRecov,
				&ui64DiscardedDocs,
				pDbRebuildStatusObj)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pDbRebuildStatusObj)
	{
		pDbRebuildStatusObj->Release();
	}

	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1updateIniFile(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis,
	jstring			sParamName,
	jstring			sValue)
{
	RCODE							rc = NE_XFLM_OK;
	F_DbSystem *				pDbSystem = THIS_DBSYS();
	FLMBYTE						ucParamName [80];
	F_DynaBuf					paramNameBuf( ucParamName, sizeof( ucParamName));
	FLMBYTE						ucValue [80];
	F_DynaBuf					valueBuf( ucValue, sizeof( ucValue));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sParamName, &paramNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sValue, &valueBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	// Call the rebuild function.
	
	if (RC_BAD( rc = pDbSystem->updateIniFile(
				(const char *)paramNameBuf.getBufferPtr(),
				(const char *)valueBuf.getBufferPtr())))
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
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbDup(
	JNIEnv *			pEnv,
  	jobject,			// obj,
  	jlong				lThis,
	jlong				lDbToDup)
{
	RCODE				rc = NE_XFLM_OK;
	F_DbSystem *	pDbSystem = THIS_DBSYS();
	IF_Db *			pDbToDup = (IF_Db *)((FLMUINT)lDbToDup);
	IF_Db *			pDb = NULL;

	if (!pDbToDup)
	{
		rc = RC_SET( NE_XFLM_INVALID_PARM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = pDbSystem->dbDup( pDbToDup, &pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openMultiFileIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDirectory,
	jstring			sBaseName)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	FLMBYTE				ucDirectory [F_PATH_MAX_SIZE];
	F_DynaBuf			directoryBuf( ucDirectory, sizeof( ucDirectory));
	FLMBYTE				ucBaseName [F_PATH_MAX_SIZE];
	F_DynaBuf			baseNameBuf( ucBaseName, sizeof( ucBaseName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDirectory, &directoryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBaseName, &baseNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openMultiFileIStream(
											(const char *)directoryBuf.getBufferPtr(),
											(const char *)baseNameBuf.getBufferPtr(),
											&pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBufferedIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jint				iBufferSize)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBufferedIStream( pInputStream,
												(FLMUINT)iBufferSize, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openUncompressingIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openUncompressingIStream( pInputStream, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openFileOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sFileName,
	jboolean			bTruncateIfFileExists)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	FLMBYTE				ucFileName [F_PATH_MAX_SIZE];
	F_DynaBuf			fileNameBuf( ucFileName, sizeof( ucFileName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sFileName, &fileNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openFileOStream(
								(const char *)fileNameBuf.getBufferPtr(),
								bTruncateIfFileExists ? TRUE : FALSE, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openMultiFileOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDirectory,
	jstring			sBaseName,
	jint				iMaxFileSize,
	jboolean			bOkToOverwrite)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	FLMBYTE				ucDirectory [F_PATH_MAX_SIZE];
	F_DynaBuf			directoryBuf( ucDirectory, sizeof( ucDirectory));
	FLMBYTE				ucBaseName [F_PATH_MAX_SIZE];
	F_DynaBuf			baseNameBuf( ucBaseName, sizeof( ucBaseName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDirectory, &directoryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBaseName, &baseNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->openMultiFileOStream(
								(const char *)directoryBuf.getBufferPtr(),
								(const char *)baseNameBuf.getBufferPtr(),
								(FLMUINT)iMaxFileSize,
								bOkToOverwrite ? TRUE : FALSE, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1removeMultiFileStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDirectory,
	jstring			sBaseName)
{
	RCODE					rc = NE_XFLM_OK;
	FLMBYTE				ucDirectory [F_PATH_MAX_SIZE];
	F_DynaBuf			directoryBuf( ucDirectory, sizeof( ucDirectory));
	FLMBYTE				ucBaseName [F_PATH_MAX_SIZE];
	F_DynaBuf			baseNameBuf( ucBaseName, sizeof( ucBaseName));
	
	// Get all of the string parameters into buffers.
	
	if (RC_BAD( rc = getUTF8String( pEnv, sDirectory, &directoryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	if (RC_BAD( rc = getUTF8String( pEnv, sBaseName, &baseNameBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = THIS_DBSYS()->removeMultiFileStream(
								(const char *)directoryBuf.getBufferPtr(),
								(const char *)baseNameBuf.getBufferPtr())))
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
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBufferedOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lInputOStream,
	jint				iBufferSize)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	IF_OStream *		pInputOStream = (IF_OStream *)((FLMUINT)lInputOStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBufferedOStream(
								pInputOStream, (FLMUINT)iBufferSize, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openCompressingOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lInputOStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_OStream *		pOStream = NULL;
	IF_OStream *		pInputOStream = (IF_OStream *)((FLMUINT)lInputOStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openCompressingOStream( pInputOStream, &pOStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pOStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1writeToOStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jlong				lOStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = (IF_IStream *)((FLMUINT)lIStream);
	IF_OStream *		pOStream = (IF_OStream *)((FLMUINT)lOStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->writeToOStream( pIStream, pOStream)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBase64Encoder(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream,
	jboolean			bInsertLineBreaks)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBase64Encoder( pInputStream,
								bInsertLineBreaks ? TRUE : FALSE, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openBase64Decoder(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lIStream)
{
	RCODE					rc = NE_XFLM_OK;
	IF_IStream *		pIStream = NULL;
	IF_IStream *		pInputStream = (IF_IStream *)((FLMUINT)lIStream);
	
	if (RC_BAD( rc = THIS_DBSYS()->openBase64Decoder( pInputStream, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return( (jlong)((FLMUINT)pIStream));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setDynamicMemoryLimit(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iCacheAdjustPercent,
	jint				iCacheAdjustMin,
	jint				iCacheAdjustMax,
	jint				iCacheAdjustMinToLeave)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (RC_BAD( rc = THIS_DBSYS()->setDynamicMemoryLimit(
								(FLMUINT)iCacheAdjustPercent,
								(FLMUINT)iCacheAdjustMin,
								(FLMUINT)iCacheAdjustMax,
								(FLMUINT)iCacheAdjustMinToLeave)))
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
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1setHardMemoryLimit(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iPercent,
	jboolean			bPercentOfAvail,
	jint				iMin,
	jint				iMax,
	jint				iMinToLeave,
	jboolean			bPreallocate)
{
	RCODE	rc = NE_XFLM_OK;
	
	if (RC_BAD( rc = THIS_DBSYS()->setHardMemoryLimit(
								(FLMUINT)iPercent,
								bPercentOfAvail? TRUE : FALSE,
								(FLMUINT)iMin,
								(FLMUINT)iMax,
								(FLMUINT)iMinToLeave,
								bPreallocate ? TRUE : FALSE)))
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
JNIEXPORT jboolean JNICALL Java_xflaim_DbSystem__1getDynamicCacheSupported(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_DBSYS()->getDynamicCacheSupported() ? JNI_TRUE : JNI_FALSE);
}

