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

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DbSystem__1createDbSystem(
	JNIEnv *				pEnv,
	jobject)				// obj)
{
	F_DbSystem * pDbSystem;
	
	if( (pDbSystem = new F_DbSystem()) == NULL)
	{
		ThrowError( NE_XFLM_MEM, pEnv);
	}
	
	return( (jlong)(FLMUINT)pDbSystem);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1init(
	JNIEnv * 			pEnv,
	jobject,				// obj,
	jlong					lThis)
{
	RCODE					rc = NE_XFLM_OK;

	if (RC_BAD( rc = THIS_DBSYS()->init()))
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
JNIEXPORT void JNICALL Java_xflaim_DbSystem__1exit(
	JNIEnv *,			// pEnv,
	jobject,				// obj,
	jlong					lThis)
{
	THIS_DBSYS()->exit();
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
	XFLM_CREATE_OPTS *	pOpts = &Opts;
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
		jclass class_CREATEOPTS = pEnv->FindClass( "xflaim/CREATEOPTS");
												
		Opts.uiBlockSize = pEnv->GetIntField( CreateOpts, 
			pEnv->GetFieldID( class_CREATEOPTS, "iBlockSize", "I"));
			
		Opts.uiVersionNum = pEnv->GetIntField( CreateOpts, 
			pEnv->GetFieldID( class_CREATEOPTS, "iVersionNum", "I"));
			
		Opts.uiMinRflFileSize = pEnv->GetIntField( CreateOpts, 
			pEnv->GetFieldID( class_CREATEOPTS, "iMinRflFileSize", "I"));
			
		Opts.uiMaxRflFileSize = pEnv->GetIntField( CreateOpts, 
			pEnv->GetFieldID( class_CREATEOPTS, "iMaxRflFileSize", "I"));
			
		Opts.bKeepRflFiles = pEnv->GetBooleanField( CreateOpts, 
			pEnv->GetFieldID( class_CREATEOPTS, "bKeepRflFiles", "Z"))
					? TRUE
					: FALSE;
		  
		Opts.bLogAbortedTransToRfl = pEnv->GetBooleanField( CreateOpts, 
			pEnv->GetFieldID( class_CREATEOPTS, "bLogAbortedTransToRfl", "Z"))
					? TRUE
					: FALSE;
					
		Opts.uiDefaultLanguage = pEnv->GetIntField( CreateOpts,
			pEnv->GetFieldID( class_CREATEOPTS, "iDefaultLanguage", "I"));
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

  	return( (jlong)pDb);
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
	if ((pRestoreClient = new JNIRestoreClient( RestoreClient, pJvm)) == NULL)
	{
		ThrowError( NE_XFLM_MEM, pEnv);
		goto Exit;
	}
	
	if (RestoreStatus != NULL)
	{
		if ((pRestoreStatus = new JNIRestoreStatus( RestoreStatus, pJvm)) == NULL)
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
		if ((pStatus = new JNIRenameStatus( Status, pJvm)) == NULL)
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
		if ( (pStatus = new JNICopyStatus( Status, pJvm)) == NULL)
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
		
		if ((pStatus = new JNICheckStatus( Status, pJvm)) == NULL)
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
