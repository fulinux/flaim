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
	jstring					sDbFileName,
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
	char *					pszFilePath = NULL;
	char *					pszDataDir = NULL;
	char *					pszRflDir = NULL;
	char *					pszDictFileName = NULL;
	char *					pszDictBuf = NULL;

	flmAssert( sDbFileName);
	pszFilePath = (char *)pEnv->GetStringUTFChars( sDbFileName, NULL);

	if (sDataDir)
	{
		pszDataDir =  (char *)pEnv->GetStringUTFChars( sDataDir, NULL);
	}
	
	if (sRflDir)
	{
		pszRflDir =  (char *)pEnv->GetStringUTFChars( sRflDir, NULL);
	}
	
	if (sDictFileName)
	{
		pszDictFileName =  (char *)pEnv->GetStringUTFChars( sDictFileName, NULL);
	}
	
	if (sDictBuf)
	{
		pszDictBuf =  (char *)pEnv->GetStringUTFChars( sDictBuf, NULL);
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
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCreate( pszFilePath, pszDataDir,
		pszRflDir, pszDictFileName, pszDictBuf, pOpts, (IF_Db **)&pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

 Exit:

	pEnv->ReleaseStringUTFChars( sDbFileName, pszFilePath);

	if (pszDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDataDir, pszDataDir);
	}

	if (pszRflDir)
	{
		pEnv->ReleaseStringUTFChars( sRflDir, pszRflDir);
	}

	if (pszDictFileName)
	{
		pEnv->ReleaseStringUTFChars( sDictFileName, pszDictFileName);
	}

	if (pszDictBuf)
	{
		pEnv->ReleaseStringUTFChars( sDictBuf, pszDictBuf);
	}

  	return( (jlong)((FLMUINT)pDb));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbOpen(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbFileName,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sPassword,
	jboolean			bAllowLimited)
{
	RCODE 			rc = NE_XFLM_OK;
	F_Db * 			pDb = NULL;
	char * 			pszFilePath;
	char *			pszDataDir = NULL;
	char *			pszRflDir = NULL;
	char *			pszPassword = NULL;
 
 	flmAssert( sDbFileName);
	
	pszFilePath = (char *)pEnv->GetStringUTFChars( sDbFileName, NULL);
	
	if (sDataDir)
	{
		pszDataDir = (char *)pEnv->GetStringUTFChars( sDataDir, NULL);
	}
	
	if (sRflDir)
	{
		pszRflDir = (char *)pEnv->GetStringUTFChars( sRflDir, NULL);
	}
	
	if (sPassword)
	{
		pszPassword = (char *)pEnv->GetStringUTFChars( sPassword, NULL);
	}
  
 	if (RC_BAD( rc = THIS_DBSYS()->dbOpen( pszFilePath, pszDataDir,
 						pszRflDir, pszPassword, bAllowLimited ? TRUE : FALSE,
						(IF_Db **)&pDb)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
 	
Exit:

	pEnv->ReleaseStringUTFChars( sDbFileName, pszFilePath);
	
	if (pszDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDataDir, pszDataDir);
	}
	
	if (pszRflDir)
	{
		pEnv->ReleaseStringUTFChars( sRflDir, pszRflDir);
	}
	
	if( pszPassword)
	{
		pEnv->ReleaseStringUTFChars( sPassword, pszPassword);
	}
	
	return( (jlong)(FLMUINT)pDb);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRemove(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbName,
	jstring			sDataDir,
	jstring			sRflDir,
	jboolean			bRemove)
{
	char * 			pszName;
	char *			pszDataDir = NULL;
	char *			pszRflDir = NULL;

	flmAssert( sDbName);
	pszName = (char *)pEnv->GetStringUTFChars( sDbName, NULL);
	
	if (sDataDir)
	{
		pszDataDir = (char *)pEnv->GetStringUTFChars( sDataDir, NULL);
	}
	
	if (sRflDir)
	{
		pszRflDir = (char *)pEnv->GetStringUTFChars( sRflDir, NULL);
	}

	THIS_DBSYS()->dbRemove( pszName, pszDataDir, 
		pszRflDir, bRemove ? TRUE : FALSE);

	pEnv->ReleaseStringUTFChars( sDbName, pszName);
	
	if (pszDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDataDir, pszDataDir);
	}
	
	if (pszRflDir)
	{
		pEnv->ReleaseStringUTFChars( sRflDir, pszRflDir);
	}
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
	char * 					pszName;
	char *					pszDataDir = NULL;
	char *					pszRflDir = NULL;
	char *					pszBackupPath = NULL;
	char *					pszPassword = NULL;
	JavaVM *					pJvm = NULL;
	JNIRestoreClient *	pRestoreClient = NULL;
	JNIRestoreStatus *	pRestoreStatus = NULL;

	pEnv->GetJavaVM( &pJvm);

	flmAssert( sDbPath);
	pszName = (char *)pEnv->GetStringUTFChars( sDbPath, NULL);
	
	if (sDataDir)
	{
		pszDataDir =  (char *)pEnv->GetStringUTFChars( sDataDir, NULL);
	}
	
	if (sRflDir)
	{
		pszRflDir =  (char *)pEnv->GetStringUTFChars( sRflDir, NULL);
	}
	
	if (sBackupPath)
	{
		pszBackupPath =  (char *)pEnv->GetStringUTFChars( sBackupPath, NULL);
	}
	
	if (sPassword)
	{
		pszPassword =  (char *)pEnv->GetStringUTFChars( sPassword, NULL);
	}
	
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
	
	if (RC_BAD( rc = THIS_DBSYS()->dbRestore( pszName, pszDataDir, pszBackupPath,
		pszRflDir, pszPassword, pRestoreClient, pRestoreStatus)))
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
	
	pEnv->ReleaseStringUTFChars( sDbPath, pszName);
	
	if (pszDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDataDir, pszDataDir);
	}
	
	if (pszRflDir)
	{
		pEnv->ReleaseStringUTFChars( sRflDir, pszRflDir);
	}

	if (pszBackupPath)
	{
		pEnv->ReleaseStringUTFChars( sBackupPath, pszBackupPath);
	}
	
	if (pszPassword)
	{
		pEnv->ReleaseStringUTFChars( sPassword, pszPassword);
	}	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1dbRename(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbName,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sNewDbName,
	jboolean			bOverwriteDestOk,
	jobject			Status)
{
	RCODE					rc = NE_XFLM_OK;
	char *				pszDbName = NULL;
	char *				pszDataDir = NULL;
	char *				pszRflDir = NULL;
	char *				pszNewDbName = NULL;
	JavaVM *				pJvm;
	JNIRenameStatus *	pStatus = NULL;

	flmAssert( sDbName);
	flmAssert( sNewDbName);
	pszDbName = (char *)pEnv->GetStringUTFChars( sDbName, NULL);
	pszNewDbName = (char *)pEnv->GetStringUTFChars( sNewDbName, NULL);
	
	if (sDataDir)
	{
		pszDataDir =  (char *)pEnv->GetStringUTFChars( sDataDir, NULL);
	}
	
	if (sRflDir)
	{
		pszRflDir =  (char *)pEnv->GetStringUTFChars( sRflDir, NULL);
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

	if (RC_BAD(rc = THIS_DBSYS()->dbRename( pszDbName, pszDataDir, pszRflDir,
			pszNewDbName, bOverwriteDestOk ? TRUE : FALSE, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}
	
Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	
	pEnv->ReleaseStringUTFChars( sDbName, pszDbName);
	pEnv->ReleaseStringUTFChars( sNewDbName, pszNewDbName);
	
	if (pszDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDataDir, pszDataDir);
	}
	
	if (pszRflDir)
	{
		pEnv->ReleaseStringUTFChars( sRflDir, pszRflDir);
	}	
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
	char *				pszSrcDbName = NULL;
	char *				pszSrcDataDir = NULL;
	char *				pszSrcRflDir = NULL;
	char *				pszDestDbName = NULL;
	char *				pszDestDataDir = NULL;
	char *				pszDestRflDir = NULL;
	JavaVM *				pJvm;
	JNICopyStatus *	pStatus = NULL;

	flmAssert( sSrcDbName);
	pszSrcDbName = (char *)pEnv->GetStringUTFChars( sSrcDbName, NULL);
	
	if (sSrcDataDir)
	{
		pszSrcDataDir = (char *)pEnv->GetStringUTFChars( sSrcDataDir, NULL);
	}
	
	if (sSrcRflDir)
	{
		pszSrcRflDir = (char *)pEnv->GetStringUTFChars( sSrcRflDir, NULL);
	}

	flmAssert( sSrcDbName);
	pszDestDbName = (char *)pEnv->GetStringUTFChars( sDestDbName, NULL);

	if (sDestDataDir)
	{
		pszDestDataDir = (char *)pEnv->GetStringUTFChars( sDestDataDir, NULL);
	}

	if (sDestRflDir)
	{
		pszDestRflDir = (char *)pEnv->GetStringUTFChars( sDestRflDir, NULL);
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
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCopy( pszSrcDbName, pszSrcDataDir,
		pszSrcRflDir, pszDestDbName, pszDestDataDir, pszDestRflDir, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;	
	}

Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	
	pEnv->ReleaseStringUTFChars( sSrcDbName, pszSrcDbName);
	
	if (pszSrcDataDir)
	{
		pEnv->ReleaseStringUTFChars( sSrcDataDir, pszSrcDataDir);
	}
	
	if (pszSrcRflDir)
	{
		pEnv->ReleaseStringUTFChars( sSrcRflDir, pszSrcRflDir);
	}
	
	pEnv->ReleaseStringUTFChars( sDestDbName, pszDestDbName);
	
	if (pszDestDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDestDataDir, pszDestDataDir);
	}
	
	if (pszDestRflDir)
	{
		pEnv->ReleaseStringUTFChars( sDestRflDir, pszDestRflDir);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1dbCheck(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sDbName,
	jstring			sDataDir,
	jstring			sRflDir,
	jstring			sPassword,
	jint				iFlags,
	jobject			Status)
{
	RCODE					rc = NE_XFLM_OK;
	char *				pszDbName = NULL;
	char * 				pszDataDir = NULL;
	char *				pszRflDir = NULL;
	char *				pszPassword = NULL;
	JNICheckStatus *	pStatus = NULL;
	F_DbInfo *			pDbInfo = NULL;
	
	flmAssert( sDbName);
	pszDbName = (char *)pEnv->GetStringUTFChars( sDbName, NULL);
	
	if (sDataDir)
	{
		pszDataDir = (char *)pEnv->GetStringUTFChars( sDataDir, NULL);
	}
	
	if (sRflDir)
	{
		pszDataDir = (char *)pEnv->GetStringUTFChars( sRflDir, NULL);
	}

	if (sPassword)
	{
		pszPassword = (char *)pEnv->GetStringUTFChars( sPassword, NULL);
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
	
	if (RC_BAD( rc = THIS_DBSYS()->dbCheck( pszDbName, pszDataDir, pszRflDir,
		pszPassword, (FLMUINT)iFlags, (IF_DbInfo **)&pDbInfo, pStatus)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pStatus)
	{
		pStatus->Release();
	}
	
	pEnv->ReleaseStringUTFChars( sDbName, pszDbName);
	
	if (pszDataDir)
	{
		pEnv->ReleaseStringUTFChars( sDataDir, pszDataDir);
	}
	
	if (pszRflDir)
	{
		pEnv->ReleaseStringUTFChars( sRflDir, pszRflDir);
	}

	if (pszPassword)
	{
		pEnv->ReleaseStringUTFChars( sPassword, pszPassword);
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
	IF_PosIStream *	pIStream = NULL;

	char * pcBuffer = (char *)pEnv->GetStringUTFChars(sBuffer, NULL);
	FLMUINT uiBufLen = pEnv->GetStringUTFLength( sBuffer);

	if (RC_BAD( rc = THIS_DBSYS()->openBufferIStream( pcBuffer,
		uiBufLen, &pIStream)))
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
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1openFileIStream(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sPath)
{
	RCODE					rc = NE_XFLM_OK;
	char * 				pszPath = (char *)pEnv->GetStringUTFChars( sPath, NULL);
	IF_PosIStream *	pIStream = NULL;
	
	if (RC_BAD( rc = THIS_DBSYS()->openFileIStream( pszPath, &pIStream)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	pEnv->ReleaseStringUTFChars( sPath, pszPath);
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
													"(IVLLLLL)I");
		m_jReportRebuildErrMethodId = pEnv->GetMethodID( pEnv->GetObjectClass( jDbRebuildStatusObject),
													"reportRebuildErr",
													"(IIIIIIIIL)I");
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

