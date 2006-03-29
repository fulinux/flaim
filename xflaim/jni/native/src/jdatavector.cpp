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
#include "xflaim_DataVector.h"
#include "jniftk.h"

#define THIS_VECTOR() \
	((IF_DataVector *)(FLMUINT)lThis)

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_VECTOR()->Release();
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setDocumentId(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				lDocumentId)
{
	THIS_VECTOR()->setDocumentID( (FLMUINT64)lDocumentId);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setID(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementId,
	jlong				lID)
{
	RCODE				rc = NE_XFLM_OK;

	if (RC_BAD( rc = THIS_VECTOR()->setID( 
		(FLMUINT)iElementId, (FLMUINT64)lID)))
	{
		ThrowError(rc, pEnv);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setNameId(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber,
	jint				iNameId,
	jboolean			bIsAttr,
	jboolean			bIsData)
{
	RCODE					rc = NE_XFLM_OK;

	if (RC_BAD( rc = THIS_VECTOR()->setNameId( (FLMUINT)iElementNumber,
			(FLMUINT)iNameId, (FLMBOOL)(bIsAttr ? TRUE : FALSE),
			(FLMBOOL)(bIsData ? TRUE : FALSE))))
	{
		ThrowError( rc, pEnv);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setINT(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber,
	jint				iNum)
{
	RCODE					rc = NE_XFLM_OK;

	if (iNum  > 0x7FFFFFFF)
	{
		ThrowError( NE_XFLM_CONV_DEST_OVERFLOW, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = THIS_VECTOR()->setINT( 
		(FLMUINT)iElementNumber, (FLMINT)iNum)))
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
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setUINT(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber,
	jint				iUNum)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiNum = (FLMUINT)iUNum;
	
	if (uiNum  > 0xFFFFFFFF)
	{
		ThrowError( NE_XFLM_CONV_DEST_OVERFLOW, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = THIS_VECTOR()->setUINT( (FLMUINT)iElementNumber, uiNum)))
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
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber,
	jlong				lNum)
{
	RCODE					rc = NE_XFLM_OK;
	
	if (RC_BAD( rc = THIS_VECTOR()->setINT64( 
			(FLMUINT)iElementNumber, (FLMUINT64)lNum)))
	{
		ThrowError( rc, pEnv);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber,
	jstring			sValue)
{
	RCODE					rc = NE_XFLM_OK;
	jchar *				puzValue = NULL;
	FLMBOOL				bMustRelease = FALSE;
	
	if (sValue)
	{
		puzValue = (jchar *)pEnv->GetStringCritical( sValue, NULL);
		bMustRelease = TRUE;
	}
	
	if (RC_BAD( rc = THIS_VECTOR()->setUnicode( 
		(FLMUINT)iElementNumber, puzValue)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	if (bMustRelease)
	{
		pEnv->ReleaseStringCritical( sValue, puzValue);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber,
	jbyteArray		Value)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiLength = pEnv->GetArrayLength( Value);
	void *				pvValue = NULL;
	jboolean				bIsCopy = false;
	FLMBOOL				bMustRelease = false;
	
	if ( (pvValue = pEnv->GetPrimitiveArrayCritical( Value, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = THIS_VECTOR()->setBinary( (FLMUINT)iElementNumber, 
			pvValue, uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		pEnv->ReleasePrimitiveArrayCritical( Value, pvValue, JNI_ABORT);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setRightTruncated(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	THIS_VECTOR()->setRightTruncated( (FLMUINT)iElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1setLefttTruncated(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	THIS_VECTOR()->setLeftTruncated( (FLMUINT)iElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1clearRightTruncated(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	THIS_VECTOR()->clearRightTruncated( (FLMUINT)iElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1clearLefttTruncated(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	THIS_VECTOR()->clearLeftTruncated( (FLMUINT)iElementNumber);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DataVector__1getDocumentID(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	return( (jlong)(THIS_VECTOR()->getDocumentID()));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DataVector__1getID(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( (jlong)THIS_VECTOR()->getID( (FLMUINT)iElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DataVector__1getNameId(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( (jint)THIS_VECTOR()->getNameId( (FLMUINT)iElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DataVector__1isAttr(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( THIS_VECTOR()->isAttr( (FLMUINT)iElementNumber) ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DataVector__1isDataComponent(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( THIS_VECTOR()->isDataComponent(
				 (FLMUINT)iElementNumber) ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jboolean JNICALL Java_xflaim_DataVector__1isKeyComponent(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( THIS_VECTOR()->isKeyComponent(
				(FLMUINT)iElementNumber) ? true : false);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DataVector__1getDataLength(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( (jint)THIS_VECTOR()->getDataLength( (FLMUINT)iElementNumber));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DataVector__1getDataType(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	return( (jint)(THIS_VECTOR()->getDataType( (FLMUINT)iElementNumber)));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DataVector__1getINT(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iINT;
	
	if (RC_BAD( rc = THIS_VECTOR()->getINT( (FLMUINT)iElementNumber, &iINT)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)iINT);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jint JNICALL Java_xflaim_DataVector__1getUINT(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	RCODE					rc = NE_XFLM_OK;
	FLMINT				iINT;
	
	if (RC_BAD( rc = THIS_VECTOR()->getUINT( (FLMUINT)iElementNumber, 
		(FLMUINT *)&iINT)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jint)iINT);	
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_DataVector__1getLong(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT64			i64INT;
	
	if (RC_BAD( rc = THIS_VECTOR()->getINT64( (FLMUINT)iElementNumber, &i64INT)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

Exit:

	return( (jlong)i64INT);	

}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jstring JNICALL Java_xflaim_DataVector__1getString(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUNICODE			uzBuffer[ 128];
	FLMUNICODE *		puzBuf = uzBuffer;
	FLMUINT				uiBufSize = sizeof( uzBuffer);
	FLMUINT				uiNumChars;
	jstring				sBuf = NULL;
	
	if (RC_BAD( rc = THIS_VECTOR()->getUnicode( (FLMUINT)iElementNumber,
		NULL, &uiNumChars)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}	
		
	if (uiNumChars * sizeof( FLMUNICODE) >= uiBufSize)
	{
		uiBufSize = (uiNumChars + 1) * sizeof( FLMUNICODE);
		if (RC_BAD( rc = f_alloc( uiBufSize, &puzBuf)))
		{
			ThrowError( rc,  pEnv);
			goto Exit;	
		}
	}

	if (RC_BAD( rc = THIS_VECTOR()->getUnicode( (FLMUINT)iElementNumber,
		puzBuf, &uiBufSize)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	sBuf = pEnv->NewString( puzBuf, uiBufSize / sizeof( FLMUNICODE));
	
Exit:

	if (puzBuf != uzBuffer)
	{
		f_free( &puzBuf);
	}
	
	return( sBuf);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_DataVector__1getBinary(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jint				iElementNumber)
{
	RCODE				rc = NE_XFLM_OK;
	FLMUINT			uiLength;
	jbyteArray		Data;
	void *			pvData = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	uiLength = THIS_VECTOR()->getDataLength( (FLMUINT)iElementNumber);
	Data = pEnv->NewByteArray( uiLength);
	
	if ( (pvData = pEnv->GetPrimitiveArrayCritical( Data, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = THIS_VECTOR()->getBinary( (FLMUINT)iElementNumber,
		pvData, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		if (RC_BAD( rc))
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, JNI_ABORT);
			pEnv->DeleteLocalRef( Data);
			Data = NULL;
		}
		else
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, 0);
		}
	}

	return( Data);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_DataVector__1outputKey(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				ljDbRef,
	jint				iIndexNum,
	jboolean			bOutputIds)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = (IF_Db *)(FLMUINT)ljDbRef;
	FLMUINT			uiLength;
	jbyteArray		Key;
	void *			pvKey = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	uiLength = XFLM_MAX_KEY_SIZE;

	Key = pEnv->NewByteArray( uiLength);
	
	if( (pvKey = pEnv->GetPrimitiveArrayCritical( Key, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = THIS_VECTOR()->outputKey( pDb, (FLMUINT)iIndexNum,
			(bOutputIds ? TRUE : FALSE), (FLMBYTE *)pvKey,
			uiLength, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		if (RC_BAD( rc))
		{
			pEnv->ReleasePrimitiveArrayCritical( Key, pvKey, JNI_ABORT);
			pEnv->DeleteLocalRef( Key);
			Key = NULL;
		}
		else
		{
			pEnv->ReleasePrimitiveArrayCritical( Key, pvKey, 0);
		}
	}

	return( Key);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jbyteArray JNICALL Java_xflaim_DataVector__1outputData(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				ljDbRef,
	jint				iIndexNum)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = (IF_Db *)(FLMUINT)ljDbRef;
	FLMUINT			uiLength;
	jbyteArray		Data;
	void *			pvData = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	uiLength = XFLM_MAX_KEY_SIZE;

	Data = pEnv->NewByteArray( uiLength);
	
	if ( (pvData = pEnv->GetPrimitiveArrayCritical( Data, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = THIS_VECTOR()->outputData( pDb, (FLMUINT)iIndexNum,
		(FLMBYTE *)pvData, uiLength, &uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		if (RC_BAD( rc))
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, JNI_ABORT);
			pEnv->DeleteLocalRef( Data);
			Data = NULL;
		}
		else
		{
			pEnv->ReleasePrimitiveArrayCritical( Data, pvData, 0);
		}
	}

	return( Data);
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1inputKey(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				ljDbRef,
	jint				iIndexNum,
	jbyteArray		Key,
	jint				iKeyLen)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = (IF_Db *)(FLMUINT)ljDbRef;
	void *			pvKey = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	if ( (pvKey = pEnv->GetPrimitiveArrayCritical( Key, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = THIS_VECTOR()->inputKey( pDb, (FLMUINT)iIndexNum,
		(FLMBYTE *)pvKey, (FLMUINT)iKeyLen)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		pEnv->ReleasePrimitiveArrayCritical( Key, pvKey, JNI_ABORT);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1inputData(
	JNIEnv *			pEnv,
	jobject,			// obj,
	jlong				lThis,
	jlong				ljDbRef,
	jint				iIndexNum,
	jbyteArray		Data,
	jint				iDataLen)
{
	RCODE				rc = NE_XFLM_OK;
	IF_Db *			pDb = (IF_Db *)(FLMUINT)ljDbRef;
	void *			pvData = NULL;
	jboolean			bIsCopy = false;
	FLMBOOL			bMustRelease = false;
	
	if( (pvData = pEnv->GetPrimitiveArrayCritical( Data, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	bMustRelease = true;
	
	if (RC_BAD( rc = THIS_VECTOR()->inputKey( pDb, (FLMUINT)iIndexNum,
		(FLMBYTE *)pvData, (FLMUINT)iDataLen)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (bMustRelease)
	{
		pEnv->ReleasePrimitiveArrayCritical( Data, pvData, JNI_ABORT);
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_DataVector__1reset(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	THIS_VECTOR()->reset();
}
