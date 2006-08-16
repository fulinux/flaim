//------------------------------------------------------------------------------
// Desc:
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

#include "xflaim_Query.h"
#include "flaimsys.h"
#include "jniftk.h"

#define THIS_QUERY() ((F_Query *)((FLMUINT)lThis))
	
/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1release(
	JNIEnv *,		// pEnv,
	jobject,			// obj,
	jlong				lThis)
{
	IF_Query *	pQuery = THIS_QUERY();
	
	if (pQuery)
	{
		pQuery->Release();
	}
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT jlong JNICALL Java_xflaim_Query__1createQuery(
	JNIEnv *				pEnv,
	jobject,				// obj
	jint					iCollection)
{
	RCODE			rc = NE_XFLM_OK;
	F_Query *	pQuery;

	if ((pQuery = f_new F_Query) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	pQuery->setCollection( (FLMUINT)iCollection);
	
Exit:
	
	return( (jlong)((FLMUINT)pQuery));
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1setLanguage(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iLanguage)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();

	if (RC_BAD( rc = pQuery->setLanguage( (FLMUINT)iLanguage)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1setupQueryExpr(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lDbRef,
	jstring				sQuery)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	IF_Db *		pDb = (IF_Db *)((FLMUINT)lDbRef);
	FLMBYTE		ucQuery [512];
	F_DynaBuf	queryBuf( ucQuery, sizeof( ucQuery));
	
	if (RC_BAD( rc = getUniString( pEnv, sQuery, &queryBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}

	if (RC_BAD( rc = pQuery->setupQueryExpr( pDb, queryBuf.getUnicodePtr())))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1copyCriteria(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lQueryToCopy)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	IF_Query *	pQueryToCopy = (IF_Query *)((FLMUINT)lQueryToCopy);
	
	if (RC_BAD( rc = pQuery->copyCriteria( pQueryToCopy)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addXPathComponent(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iXPathAxis,
	jint					iNodeType,
	jint					iNameId)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addXPathComponent( (eXPathAxisTypes)iXPathAxis,
										(eDomNodeType)iNodeType, (FLMUINT)iNameId,
										NULL)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addOperator(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iOperator)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addOperator( (eQueryOperators)iOperator, 0, NULL)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addStringOperator(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jint					iOperator,
	jboolean				bCaseInsensitive,
	jboolean				bCompressWhitespace,
	jboolean				bNoWhitespace,
	jboolean				bNoUnderscores,
	jboolean				bNoDashes,
	jboolean				bWhitespaceAsSpace,
	jboolean				bIgnoreLeadingSpace,
	jboolean				bIgnoreTrailingSpace)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	FLMUINT		uiCompareRules = 0;
	
	if (bCaseInsensitive)
	{
		uiCompareRules |= XFLM_COMP_CASE_INSENSITIVE;
	}
	if (bCompressWhitespace)
	{
		uiCompareRules |= XFLM_COMP_COMPRESS_WHITESPACE;
	}
	if (bNoWhitespace)
	{
		uiCompareRules |= XFLM_COMP_NO_WHITESPACE;
	}
	if (bNoUnderscores)
	{
		uiCompareRules |= XFLM_COMP_NO_UNDERSCORES;
	}
	if (bNoDashes)
	{
		uiCompareRules |= XFLM_COMP_NO_DASHES;
	}
	if (bWhitespaceAsSpace)
	{
		uiCompareRules |= XFLM_COMP_WHITESPACE_AS_SPACE;
	}
	if (bIgnoreLeadingSpace)
	{
		uiCompareRules |= XFLM_COMP_IGNORE_LEADING_SPACE;
	}
	if (bIgnoreTrailingSpace)
	{
		uiCompareRules |= XFLM_COMP_IGNORE_TRAILING_SPACE;
	}
	if (RC_BAD( rc = pQuery->addOperator( (eQueryOperators)iOperator,
										uiCompareRules, NULL)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addStringValue(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jstring				sValue)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	FLMBYTE		ucValue [256];
	F_DynaBuf	valueBuf( ucValue, sizeof( ucValue));
	
	if (RC_BAD( rc = getUniString( pEnv, sValue, &valueBuf)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pQuery->addUnicodeValue( valueBuf.getUnicodePtr())))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addBinaryValue(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jbyteArray			Value)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	FLMUINT		uiLength = pEnv->GetArrayLength( Value);
	void *		pvValue = NULL;
	jboolean		bIsCopy = JNI_FALSE;
	
	if ((pvValue = pEnv->GetPrimitiveArrayCritical( Value, &bIsCopy)) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
	if (RC_BAD( rc = pQuery->addBinaryValue( pvValue, uiLength)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	if (pvValue)
	{
		pEnv->ReleasePrimitiveArrayCritical( Value, pvValue, JNI_ABORT);
	}
	
	return;
}

/****************************************************************************
Desc:
****************************************************************************/
JNIEXPORT void JNICALL Java_xflaim_Query__1addLongValue(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jlong					lValue)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addINT64Value( (FLMINT64)lValue)))
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
JNIEXPORT void JNICALL Java_xflaim_Query__1addBoolean(
	JNIEnv *				pEnv,
	jobject,				// obj
	jlong					lThis,
	jboolean				bValue,
	jboolean				bUnknown)
{
	RCODE			rc = NE_XFLM_OK;
	IF_Query *	pQuery = THIS_QUERY();
	
	if (RC_BAD( rc = pQuery->addBoolean( bValue ? TRUE : FALSE,
													 bUnknown ? TRUE : FALSE)))
	{
		ThrowError( rc, pEnv);
		goto Exit;
	}
	
Exit:

	return;
}

