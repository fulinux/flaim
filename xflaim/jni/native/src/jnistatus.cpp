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

#include "jnistatus.h"

/****************************************************************************
Desc:
****************************************************************************/
RCODE XFLMAPI JNIRenameStatus::dbRenameStatus(
	const char *		pszSrcFileName,
	const char *		pszDstFileName)
{
	RCODE					rc = NE_XFLM_OK;
	JNIEnv *				pEnv;
	jclass				Cls;
	jmethodID			MId;
	jstring				sSrcName;
	jstring				sDstName;
	FLMBOOL				bMustDetach = FALSE;
	
	if (m_pJvm->GetEnv( (void **)&pEnv, JNI_VERSION_1_2) != JNI_OK)
	{
		if (m_pJvm->AttachCurrentThread( (void **)&pEnv, NULL) != 0)
		{
			rc = RC_SET( NE_XFLM_FAILURE);	
			goto Exit;
		}
		
		bMustDetach = TRUE;
	}

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "dbRenameStatus",
							 "(Ljava/lang/String;Ljava/lang/String)I");
	flmAssert( MId);
	
	sSrcName = pEnv->NewStringUTF( pszSrcFileName);
	sDstName = pEnv->NewStringUTF( pszDstFileName);
	
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jStatus, 
		MId, sSrcName, sDstName)))
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
RCODE XFLMAPI JNICopyStatus::dbCopyStatus(
	FLMUINT64		ui64BytesToCopy,
	FLMUINT64		ui64BytesCopied,
	FLMBOOL			bNewSrcFile,
	const char *	pszSrcFileName,
	const char *	pszDestFileName)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	jstring			sSrcName;
	jstring			sDstName;
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

	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "dbCopyStatus",
							 "(JJZLjava/lang/String;Ljava/lang/String)I");
	flmAssert( MId);
	
	sSrcName = pEnv->NewStringUTF( pszSrcFileName);
	sDstName = pEnv->NewStringUTF( pszDestFileName);
	
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jStatus, MId, 
		(jlong)ui64BytesToCopy, (jlong)ui64BytesCopied, 
		(bNewSrcFile) ? true : false, sSrcName, sDstName)))
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
RCODE XFLMAPI JNICheckStatus::reportProgress(
	XFLM_PROGRESS_CHECK_INFO *)		// pProgCheck)
{
	RCODE				rc = NE_XFLM_OK;
	JNIEnv *			pEnv;
	jclass			Cls;
	jmethodID		MId;
	jobject			JProgCheck;
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

	// Have to create a new XFLM_PROGRESS_CHECK_INFO java class
	// and copy everything from pProgCheck into it.
	
	Cls = pEnv->FindClass( "xflaim/Structures/PROGRESS_CHECK");
	MId = pEnv->GetMethodID( Cls, "<init>", "()V");
	flmAssert( MId);

	JProgCheck = pEnv->NewObject( Cls, MId);
	Cls = pEnv->GetObjectClass( m_jStatus);
	MId = pEnv->GetMethodID( Cls, "reportProgress",
							 "(Lxflaim/Structures/PROGRESS_CHECK)I");
	flmAssert( MId);
	
	if( RC_BAD( rc = (RCODE)pEnv->CallIntMethod( m_jStatus, MId, JProgCheck)))
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
RCODE XFLMAPI JNICheckStatus::reportCheckErr(
	XFLM_CORRUPT_INFO *,		// pCorruptInfo,
	FLMBOOL *)					// pbFix)
{
	return( NE_XFLM_NOT_IMPLEMENTED);
}
