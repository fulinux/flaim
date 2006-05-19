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

#include "jniftk.h"
#include "xflaim_Backup.h"

#define THIS_BACKUP() \
	((IF_Backup *)(FLMUINT)lThis)

/****************************************************************************
Desc:
****************************************************************************/
class JNIBackupClient : public IF_BackupClient
{
public:

	JNIBackupClient(
		jobject		jClient,
		JavaVM *		pJvm)
	{
		flmAssert( jClient);
		flmAssert( pJvm);
		m_jClient = jClient;
		m_pJvm = pJvm;
	}
	
	RCODE FLMAPI WriteData(
		const void *	pvBuffer,
		FLMUINT			uiBytesToWrite);
		
	FINLINE FLMINT FLMAPI getRefCount( void)
	{
		return( IF_BackupClient::getRefCount());
	}

	virtual FINLINE FLMINT FLMAPI AddRef( void)
	{
		return( IF_BackupClient::AddRef());
	}

	virtual FINLINE FLMINT FLMAPI Release( void)
	{
		return( IF_BackupClient::Release());
	}

private:

	jobject		m_jClient;
	JavaVM *		m_pJvm;
};

/****************************************************************************
Desc:
****************************************************************************/
class JNIBackupStatus : public IF_BackupStatus
{
public:

	JNIBackupStatus(
		jobject		jStatus,
		JavaVM *		pJvm)
	{
		flmAssert(jStatus);
		flmAssert(pJvm);
		m_jStatus = jStatus;
		m_pJvm = pJvm;
	}
	
	RCODE FLMAPI backupStatus(
		FLMUINT64	ui64BytesToDo,
		FLMUINT64	ui64BytesDone);
	
	FINLINE FLMINT FLMAPI getRefCount( void)
	{
		return( IF_BackupStatus::getRefCount());
	}

	virtual FINLINE FLMINT FLMAPI AddRef( void)
	{
		return( IF_BackupStatus::AddRef());
	}

	virtual FINLINE FLMINT FLMAPI Release( void)
	{
		return( IF_BackupStatus::Release());
	}

private:

	jobject			m_jStatus;
	JavaVM *			m_pJvm;
};

/****************************************************************************
Desc:
****************************************************************************/
RCODE JNIBackupClient::WriteData(
	const void *	pvBuffer,
	FLMUINT			uiBytesToWrite)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	jbyteArray		jBuff;
	void *			pvBuff;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}
	
	Cls = pEnv->GetObjectClass( m_jClient);
	MId = pEnv->GetMethodID( Cls, "writeData", "([B)I");
	
	flmAssert( MId);
	
	jBuff = pEnv->NewByteArray( (jsize)uiBytesToWrite);
	pvBuff = pEnv->GetPrimitiveArrayCritical(jBuff, NULL);
	f_memcpy(pvBuff, pvBuffer, uiBytesToWrite);
	pEnv->ReleasePrimitiveArrayCritical( jBuff, pvBuff, 0);
	
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jClient, MId, jBuff)))
	{
		goto Exit;
	}
		
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE JNIBackupStatus::backupStatus(
	FLMUINT64		ui64BytesToDo,
	FLMUINT64		ui64BytesDone)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	FLMBOOL			bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET(NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "backupStatus", "(JJ)I");
	flmAssert( MId);
		
	rc = (RCODE)pEnv->CallIntMethod( m_jStatus, MId, (jlong)ui64BytesToDo,
									 (jlong)ui64BytesDone);
									  
Exit:

	if (bMustDetach)
	{
		if (m_pJvm->DetachCurrentThread() != 0)
		{
			flmAssert( 0);
			rc = RC_SET( NE_XFLM_FAILURE);
		}
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Backup__1backup(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jstring			sBackupPath,
	jstring			sPassword,
	jobject			Client,
	jobject			Status)
{
	RCODE						rc = NE_XFLM_OK;
	IF_Backup *				pBackup = THIS_BACKUP();
	FLMUINT					uiSeqNum = 0;
	JavaVM *					pJvm;
	char *					pszBackupPath = NULL;
	char *					pszPassword = NULL;
	JNIBackupClient *		pClient;
	JNIBackupStatus *		pStatus = NULL;
 
	
	flmAssert( Client);
	
	pEnv->GetJavaVM( &pJvm);
	if( (pClient = f_new JNIBackupClient( Client, pJvm)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (sBackupPath)
	{
		pszBackupPath = (char *)pEnv->GetStringUTFChars( sBackupPath, NULL);
	}
	
	if (sPassword)
	{
		pszPassword = (char *)pEnv->GetStringUTFChars( sPassword, NULL);
	}
	
	if (Status)
	{
		if( (pStatus = f_new JNIBackupStatus( Status, pJvm)) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			ThrowError( rc, pEnv);
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = pBackup->backup( pszBackupPath, pszPassword, pClient,
		pStatus, &uiSeqNum)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if( pszBackupPath)
	{
		pEnv->ReleaseStringUTFChars( sBackupPath, pszBackupPath);
	}
	
	if( pszPassword)
	{
		pEnv->ReleaseStringUTFChars( sPassword, pszPassword);
	}

	if (pClient)
	{
		pClient->Release();
	}
	
	if (pStatus)
	{
		pStatus->Release();
	}
	
	return( uiSeqNum);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Backup__1endBackup(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Backup *		pThisBackup = THIS_BACKUP();

	if (RC_BAD( rc = pThisBackup->endBackup()))
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
JNIEXPORT jlong JNICALL Java_xflaim_Backup__1getBackupTransId(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_BACKUP()->getBackupTransId());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Backup__1getLastBackupTransId(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( THIS_BACKUP()->getLastBackupTransId());
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Backup__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_BACKUP()->Release();	
}
