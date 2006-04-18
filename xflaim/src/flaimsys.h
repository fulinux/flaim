//------------------------------------------------------------------------------
// Desc:	This is the master header file for FLAIM.  It is included by nearly
//			all of the source files.
//
// Tabs:	3
//
//		Copyright (c) 1991-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flaimsys.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef  FLAIMSYS_H
#define  FLAIMSYS_H

// Public includes

#include "xflaim.h"

#ifdef HAVE_CONFIG_H
	#include "../config.h"
#endif

// Put all forward references here

class F_Database;
class F_Dict;
class F_Db;
class F_NameTable;
class F_IOBuffer;
class F_Rfl;
class F_Btree;
class F_DOMNode;
class F_NodeList;
class F_Query;
class F_DbRebuild;
class F_DbCheck;
class F_DbInfo;
class F_KeyCollector;
class F_Thread;
class FSIndexCursor;
class FSCollectionCursor;
class FlmRecord;
class FlmDefaultRec;
class FDynSearchSet;
class F_CachedBlock;
class F_CachedNode;
class F_GlobalCacheMgr;
class F_BlockCacheMgr;
class F_NodeCacheMgr;
class F_NodeBufferIStream;
class F_BTreeIStream;
class F_IniFile;
class F_QueryResultSet;
class F_BTreeInfo;
class F_AttrItem;
class F_NodeVerifier;
class F_RebuildNodeIStream;

// Internal includes

#include "ftk.h"
#include "fbuff.h"
#include "fcollate.h"
#include "fdict.h"
#include "fxml.h"
#include "fstructs.h"
#include "fcache.h"
#include "flmstat.h"
#include "ffilehdl.h"
#include "ffilesys.h"
#include "f64bitfh.h"
#include "frset.h"
#include "fxpath.h"
#include "fbtrset.h"
#include "frecread.h"
#include "fsrvlock.h"
#include "fquery.h"
#include "ftkthrd.h"
#include "fcollate.h"
#include "fsrvlock.h"
#include "fdir.h"
#include "f_btree.h"
#include "f_btpool.h"
#include "rfl.h"
#include "fsuperfl.h"
#include "filesys.h"
#include "flog.h"
#include "flfixed.h"
#include "f_nici.h"

RCODE MapErrnoToFlaimErr(
	int		err,
	RCODE		defaultRc);

// Misc. global constants

#ifdef DEFINE_NUMBER_MAXIMUMS
	#define GV_EXTERN
#else
	#define GV_EXTERN		extern
#endif

GV_EXTERN FLMBOOL	gv_b32BitPlatform
#ifdef DEFINE_NUMBER_MAXIMUMS
	= (FLMBOOL)(sizeof( FLMUINT) == 4 ? TRUE : FALSE)
#endif
	;

GV_EXTERN FLMUINT	gv_uiMaxUInt32Val
#ifdef DEFINE_NUMBER_MAXIMUMS
		= (FLMUINT)0xFFFFFFFF
#endif
	;

GV_EXTERN FLMUINT	gv_uiMaxUIntVal
#ifdef DEFINE_NUMBER_MAXIMUMS
		= (FLMUINT)(~(FLMUINT)0)
#endif
		;

GV_EXTERN FLMUINT gv_uiMaxSignedIntVal
#ifdef DEFINE_NUMBER_MAXIMUMS
		= (FLMUINT)((((~(FLMUINT)0) << 1) >> 1))
#endif
	;

GV_EXTERN FLMUINT64 gv_ui64MaxSignedIntVal
#ifdef DEFINE_NUMBER_MAXIMUMS
	= (FLMUINT64)((((~(FLMUINT64)0) << 1) >> 1))
#endif
	;

#if defined( FLM_WIN) || defined( FLM_NLM)
	#define FLM_CAN_GET_PHYS_MEM
#elif defined( FLM_UNIX)
   #if defined( _SC_AVPHYS_PAGES)
		#define FLM_CAN_GET_PHYS_MEM
   #endif
#endif

// A global module lock allows us to properly implement DllCanUnloadNow
// This is only used in a COM environment.  The functions are actually
// defined in fdllmain.cpp

extern void LockModule(void);
extern void UnlockModule(void);

#define MAX_DIRTY_NODES_THRESHOLD		1024
#define MAX_DOM_HEADER_SIZE				118
#define FIXED_DOM_HEADER_SIZE				94
#define XFLM_FIXED_SIZE_HEADER_TOKEN	0xFF

// NOTE: ENCRYPT_MIN_CHUNK_SIZE should always be a power of 2
// getEncLen supposes that it is.

#define ENCRYPT_MIN_CHUNK_SIZE			16
#define ENCRYPT_BOUNDARY_MASK				(~(ENCRYPT_MIN_CHUNK_SIZE - 1))

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToUINT(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMUINT *		puiNum)
{
	if( bNeg)
	{
		return( RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW));
	}

	if( gv_b32BitPlatform && ui64Num > 0xFFFFFFFF)
	{
		return( RC_SET( NE_XFLM_CONV_NUM_OVERFLOW));
	}

	*puiNum = (FLMUINT)ui64Num;
	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToUINT32(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMUINT32 *		pui32Num)
{
	if( bNeg)
	{
		return( RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW));
	}

	if( ui64Num > 0xFFFFFFFF)
	{
		return( RC_SET( NE_XFLM_CONV_NUM_OVERFLOW));
	}

	*pui32Num = (FLMUINT32)ui64Num;
	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToUINT64(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMUINT64 *		pui64Num)
{
	if( bNeg)
	{
		return( RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW));
	}

	*pui64Num = ui64Num;
	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToINT(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMINT *			piNum)
{
	if( bNeg)
	{
		if (ui64Num == (FLMUINT64)(FLM_MAX_INT) + 1)
		{
			*piNum = FLM_MIN_INT;
		}
		else if( ui64Num > (FLMUINT64)(FLM_MAX_INT) + 1)
		{
			return( RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW));
		}
		else
		{
			*piNum = -((FLMINT)ui64Num);
		}
	}
	else
	{
		if( ui64Num > (FLMUINT64)FLM_MAX_INT)
		{
			return( RC_SET( NE_XFLM_CONV_NUM_OVERFLOW));
		}

		*piNum = (FLMINT)ui64Num;
	}

	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToINT32(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMINT32 *		pi32Num)
{
	if( bNeg)
	{
		if (ui64Num == (FLMUINT64)(FLM_MAX_INT32) + 1)
		{
			*pi32Num = FLM_MIN_INT32;
		}
		else if( ui64Num > (FLMUINT64)(FLM_MAX_INT32) + 1)
		{
			return( RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW));
		}
		else
		{
			*pi32Num = -((FLMINT32)ui64Num);
		}
	}
	else
	{
		if( ui64Num > (FLMUINT64)FLM_MAX_INT32)
		{
			return( RC_SET( NE_XFLM_CONV_NUM_OVERFLOW));
		}

		*pi32Num = (FLMINT32)ui64Num;
	}

	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc:
******************************************************************************/
FINLINE RCODE convertToINT64(
	FLMUINT64		ui64Num,
	FLMBOOL			bNeg,
	FLMINT64 *		pi64Num)
{
	if( bNeg)
	{
		if( ui64Num > gv_ui64MaxSignedIntVal + 1)
		{
			return( RC_SET( NE_XFLM_CONV_NUM_UNDERFLOW));
		}

		*pi64Num = -(FLMINT64)ui64Num;
	}
	else
	{
		if( ui64Num > gv_ui64MaxSignedIntVal)
		{
			return( RC_SET( NE_XFLM_CONV_NUM_OVERFLOW));
		}

		*pi64Num = (FLMINT64)ui64Num;
	}

	return( NE_XFLM_OK);
}

/*****************************************************************************
Desc: Calculate the number of bytes extra there are beyond the closest
		encryption boundary.
******************************************************************************/
FINLINE FLMUINT extraEncBytes(
	FLMUINT	uiDataLen)
{
	// This works if ENCRYPT_MIN_CHUNK_SIZE is a power of 2
	// Otherwise, it needs to be changed to uiDataLen % ENCRYPT_MIN_CHUNK_SIZE
	
	return( uiDataLen & (ENCRYPT_MIN_CHUNK_SIZE - 1));
}

/*****************************************************************************
Desc: Calculate the encryption length for a piece of data.
******************************************************************************/
FINLINE FLMUINT getEncLen(
	FLMUINT		uiDataLen)
{
	return( extraEncBytes( uiDataLen)
				? ((uiDataLen + ENCRYPT_MIN_CHUNK_SIZE) & ENCRYPT_BOUNDARY_MASK)
				: uiDataLen);
}

/*============================================================================
Some simple inline functions for dealing with tag numbers
============================================================================*/

FINLINE FLMBOOL elementIsUserDefined(
	FLMUINT	uiNum
	)
{
	return( uiNum <= XFLM_MAX_ELEMENT_NUM ? TRUE : FALSE);
}

FINLINE FLMBOOL attributeIsUserDefined(
	FLMUINT	uiNum
	)
{
	return( uiNum <= XFLM_MAX_ATTRIBUTE_NUM ? TRUE : FALSE);
}

FINLINE FLMBOOL elementIsReservedTag(
	FLMUINT	uiNum
	)
{
	return( uiNum >= XFLM_FIRST_RESERVED_ELEMENT_TAG &&
			  uiNum <= XFLM_LAST_RESERVED_ELEMENT_TAG ? TRUE : FALSE);
}

FINLINE FLMBOOL attributeIsReservedTag(
	FLMUINT	uiNum
	)
{
	return( uiNum >= XFLM_FIRST_RESERVED_ATTRIBUTE_TAG &&
			  uiNum <= XFLM_LAST_RESERVED_ATTRIBUTE_TAG ? TRUE : FALSE);
}

/****************************************************************************
Stuff for F_NameTable class
****************************************************************************/

typedef struct FlmTagInfoTag
{
	FLMUINT				uiType;
	FLMUNICODE *		puzTagName;
	FLMUINT				uiTagNum;
	FLMUINT				uiDataType;
	FLMUNICODE *		puzNamespace;
} FLM_TAG_INFO;

/****************************************************************************
Desc:	Class for name/number lookup.
****************************************************************************/
class F_NameTable : public XF_RefCount, public XF_Base
{
public:

	F_NameTable();

	~F_NameTable();

	RCODE setupNameTable( void);

	void clearTable(
		FLMUINT					uiPoolBlkSize);

	RCODE addReservedDictTags( void);

	RCODE getNextTagTypeAndNumOrder(
		FLMUINT					uiType,
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName = NULL,
		char *					pszTagName = NULL,
		FLMUINT					uiNameBufSize = 0,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiDataType = NULL,
		FLMUNICODE *			puzNamespace = NULL,
		FLMUINT					uiNamespaceBufSize = 0,
		FLMBOOL					bTruncatedNamesOk = TRUE);

	RCODE getNextTagTypeAndNameOrder(
		FLMUINT					uiType,
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName = NULL,
		char *					pszTagName = NULL,
		FLMUINT					uiNameBufSize = 0,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiDataType = NULL,
		FLMUNICODE *			puzNamespace = NULL,
		FLMUINT					uiNamespaceBufSize = 0,
		FLMBOOL					bTruncatedNamesOk = TRUE);

	RCODE getFromTagTypeAndName(
		F_Db *					pDb,
		FLMUINT					uiType,
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMBOOL					bMatchNamespace,
		const FLMUNICODE *	puzNamespace = NULL,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiDataType = NULL);

	RCODE getFromTagTypeAndNum(
		F_Db *					pDb,
		FLMUINT					uiType,
		FLMUINT					uiTagNum,
		FLMUNICODE *			puzTagName = NULL,
		char *					pszTagName = NULL,
		FLMUINT *				puiNameBufSize = NULL,
		FLMUINT *				puiDataType = NULL,
		FLMUNICODE *			puzNamespace = NULL,
		char *					pszNamespace = NULL,
		FLMUINT *				puiNamespaceBufSize = NULL,
		FLMBOOL					bTruncatedNamesOk = TRUE);

	RCODE addTag(
		FLMUINT					uiType,
		FLMUNICODE *			puzTagName,
		const char *			pszTagName,
		FLMUINT					uiTagNum,
		FLMUINT					uiDataType = 0,
		FLMUNICODE *			puzNamespace = NULL,
		FLMBOOL					bCheckDuplicates = TRUE,
		FLMBOOL					bLimitNumToLoad = TRUE);

	void sortTags( void);

	void removeTag(
		FLMUINT	uiType,
		FLMUINT	uiTagNum);

	RCODE cloneNameTable(
		F_NameTable *			pSrcNameTable);

	RCODE importFromNameTable(
		F_NameTable *			pSrcNameTable);

	FINLINE FLMBOOL haveAllElements( void)
	{
		return m_bLoadedAllElements;
	}

	FINLINE FLMBOOL haveAllAttributes( void)
	{
		return m_bLoadedAllAttributes;
	}

	FLMINT XFLMAPI AddRef( void);

	FLMINT XFLMAPI Release( void);
	
private:

	RCODE allocTag(
		FLMUINT					uiType,
		FLMUNICODE *			puzTagName,
		const char *			pszTagName,
		FLMUINT					uiTagNum,
		FLMUINT					uiDataType,
		FLMUNICODE *			puzNamespace,
		FLM_TAG_INFO **		ppTagInfo);

	RCODE reallocSortTables(
		FLMUINT					uiNewTblSize);

	RCODE copyTagName(
		FLMUNICODE *			puzDestTagName,
		char *					pszDestTagName,
		FLMUINT *				puiDestBufSize,
		FLMUNICODE *			puzSrcTagName,
		FLMBOOL					bTruncatedNamesOk);

	FLM_TAG_INFO * findTagByTypeAndNum(
		FLMUINT					uiType,
		FLMUINT					uiTagNum,
		FLMUINT *				puiInsertPos = NULL);

	FLM_TAG_INFO * findTagByTypeAndName(
		FLMUINT					uiType,
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMBOOL					bMatchNamespace,
		const FLMUNICODE *	puzNamespace,
		FLMBOOL *				pbAmbiguous,
		FLMUINT *				puiInsertPos = NULL);

	RCODE insertTagInTables(
		FLM_TAG_INFO *			pTagInfo,
		FLMUINT					uiTagTypeAndNameTblInsertPos,
		FLMUINT					uiTagTypeAndNumTblInsertPos);

	FLMUNICODE * findNamespace(
		FLMUNICODE *			puzNamespace,
		FLMUINT *				puiInsertPos);

	RCODE insertNamespace(
		FLMUNICODE *			puzNamespace,
		FLMUINT					uiInsertPos);

	F_Pool						m_pool;
	FLMUINT						m_uiMemoryAllocated;
	FLM_TAG_INFO **			m_ppSortedByTagTypeAndName;
	FLM_TAG_INFO **			m_ppSortedByTagTypeAndNum;
	FLMUINT						m_uiTblSize;
	FLMUINT						m_uiNumTags;
	FLMBOOL						m_bTablesSorted;
	FLMBOOL						m_bLoadedAllElements;
	FLMBOOL						m_bLoadedAllAttributes;
	FLMUINT						m_uiNumElementsLoaded;
	FLMUINT						m_uiNumAttributesLoaded;
	FLMUNICODE **				m_ppuzNamespaces;
	FLMUINT						m_uiNamespaceTblSize;
	FLMUINT						m_uiNumNamespaces;
	F_MUTEX						m_hRefMutex;
	
friend class F_Db;
};

/****************************************************************************
Storage conversion functions.
****************************************************************************/

#define FLM_MAX_NUM_BUF_SIZE		9

RCODE flmStorage2Number(
	FLMUINT					uiType,
	FLMUINT					uiBufLen,
	const FLMBYTE *		pucBuf,
	FLMUINT *				puiNum,
	FLMINT *					piNum);

RCODE flmStorage2Number64(
	FLMUINT					uiType,
	FLMUINT					uiBufLen,
	const FLMBYTE *		pucBuf,
	FLMUINT64 *				pui64Num,
	FLMINT64 *				pi64Num);

RCODE flmNumber64ToStorage(
	FLMUINT64				ui64Num,
	FLMUINT *				puiBufLen,
	FLMBYTE *				pucBuf,
	FLMBOOL					bNegative,
	FLMBOOL					bCollation);

RCODE FlmUINT2Storage(
	FLMUINT					uiNum,
	FLMUINT *				puiBufLen,
	FLMBYTE *				pucBuf);

RCODE FlmINT2Storage(
	FLMINT					iNum,
	FLMUINT *				puiBufLen,
	FLMBYTE *				pucBuf);

RCODE	flmUTF8ToStorage(
	const FLMBYTE *		pucUTF8,
	FLMUINT					uiBytesInBuffer,
	FLMBYTE *				pucBuf,
	FLMUINT *				puiBufLength);

RCODE flmGetCharCountFromStorageBuf(
	const FLMBYTE **		ppucBuf, 
	FLMUINT					uiBufSize,
	FLMUINT *				puiNumChars,
	FLMUINT *				puiSenLen = NULL);

RCODE flmStorage2UTF8(
	FLMUINT					uiType,
	FLMUINT					uiBufLength,
	const FLMBYTE *		pucBuffer,
	FLMUINT *				puiOutBufLen,
	FLMBYTE *				pucOutBuf);

RCODE flmStorage2Unicode(
	FLMUINT					uiType,
	FLMUINT					uiBufLength,
	const FLMBYTE *		pucBuffer,
	FLMUINT *				puiOutBufLen,
	void *					pOutBuf);

RCODE flmStorage2Unicode(
	FLMUINT					uiType,
	FLMUINT					uiStorageLength,
	const FLMBYTE *		pucStorageBuffer,
	IF_DynaBuf *			pBuffer);

RCODE	flmUnicode2Storage(
	const FLMUNICODE *	puzStr,
	FLMUINT					uiStrLen,
	FLMBYTE *				pucBuf,
	FLMUINT *				puiBufLength,
	FLMUINT *				puiCharCount);

RCODE flmStorageNum2StorageText(
	const FLMBYTE *		pucNum,
	FLMUINT					uiNumLen,
	FLMBYTE *				pucBuffer,
	FLMUINT *				puiBufLen);

/****************************************************************************
Desc: 	Returns the size of buffer needed to hold the unicode string in 
			FLAIM's storage format.
****************************************************************************/
FINLINE RCODE FlmGetUnicodeStorageLength(
	FLMUNICODE *	puzStr,
	FLMUINT *		puiByteCount)
{
	FLMUINT	uiByteCount;
	RCODE		rc;

	if( RC_BAD( rc = flmUnicode2Storage( puzStr, 0, NULL, 
		&uiByteCount, NULL)))
	{
		return( rc);
	}

	*puiByteCount = uiByteCount + sizeof( FLMUNICODE);
	return( NE_XFLM_OK);
}

/****************************************************************************
Desc: 	Copies and formats a Unicode string into FLAIM's storage format.
			The Unicode string must be in little-endian byte order.
			Unicode values that are not represented as WordPerfect 6.x characters
			are preserved as non-WP characters.
****************************************************************************/
FINLINE RCODE FlmUnicode2Storage(
	FLMUNICODE *	puzStr,
	FLMUINT *		puiBufLength,	// [IN] size of pBuf,
											// [OUT] amount of pBuf used to hold puzStr
	FLMBYTE *		pBuf)
{
	return( flmUnicode2Storage( puzStr, 0, pBuf, puiBufLength, NULL));
}

/****************************************************************************
Desc: 	Converts from FLAIM's internal storage format to a Unicode string
****************************************************************************/
FINLINE RCODE FlmStorage2Unicode(
	FLMUINT			uiType,
	FLMUINT			uiBufLength,
	FLMBYTE *		pBuffer,
	FLMUINT *		puiOutBufLen,
	FLMUNICODE *	puzOutBuf)
{
	return( flmStorage2Unicode( uiType, uiBufLength, pBuffer,
		puiOutBufLen, puzOutBuf));
}

/****************************************************************************
Desc: 	Convert storage text into a null-terminated UTF-8 string
****************************************************************************/
FINLINE RCODE FlmStorage2UTF8(
	FLMUINT		uiType,
	FLMUINT		uiBufLength,
	FLMBYTE *	pBuffer, 
	FLMUINT *	puiOutBufLenRV,
			// [IN] Specified the number of bytes available in buffer.
			// [OUT] Returns the number of bytes of UTF-8 text.  This value
			// does not include a terminating NULL byte in the buffer.
	FLMBYTE *	pOutBuffer)
			// [OUT] The buffer that will hold the output UTF-8 text.
			// If this value is NULL then only bufLenRV will computed so that
			// the caller may know how many bytes to allocate for a buffer.
{
	return( flmStorage2UTF8( uiType, uiBufLength, 
		pBuffer, puiOutBufLenRV, pOutBuffer));
}

typedef struct FlmVectorElementTag
{
	FLMUINT64	ui64ID;
	FLMUINT		uiNameId;
	FLMUINT		uiFlags;
#define				VECT_SLOT_HAS_DATA			0x01
#define				VECT_SLOT_HAS_ID				0x02
#define				VECT_SLOT_RIGHT_TRUNCATED	0x04
#define				VECT_SLOT_LEFT_TRUNCATED	0x08
#define				VECT_SLOT_HAS_NAME_ID		0x10
#define				VECT_SLOT_IS_ATTR				0x20
#define				VECT_SLOT_IS_DATA				0x40
	FLMUINT		uiDataType;
	FLMUINT		uiDataLength;
	FLMUINT		uiDataOffset;
} F_VECTOR_ELEMENT;

/*****************************************************************************
Desc:	Used to build keys and data components
*****************************************************************************/
class F_DataVector : public IF_DataVector, public XF_Base
{
public:

	// Constructor/Destructor

	F_DataVector();
	virtual ~F_DataVector();

	// Setter methods

	FINLINE void XFLMAPI setDocumentID(
		FLMUINT64	ui64DocumentID)
	{
		m_ui64DocumentID = ui64DocumentID;
	}

	RCODE XFLMAPI setID(
		FLMUINT		uiElementNumber,
		FLMUINT64	ui64ID);

	RCODE XFLMAPI setNameId(
		FLMUINT		uiElementNumber,
		FLMUINT		uiNameId,
		FLMBOOL		bIsAttr,
		FLMBOOL		bIsData);

	RCODE XFLMAPI setINT(
		FLMUINT	uiElementNumber,
		FLMINT	iNum);

	RCODE XFLMAPI setINT64(
		FLMUINT		uiElementNumber,
		FLMINT64		i64Num);

	RCODE XFLMAPI setUINT(
		FLMUINT	uiElementNumber,
		FLMUINT	uiNum);

	RCODE XFLMAPI setUINT64(
		FLMUINT		uiElementNumber,
		FLMUINT64	ui64Num);

	RCODE XFLMAPI setUnicode(
		FLMUINT					uiElementNumber,
		const FLMUNICODE *	puzUnicode);

	RCODE XFLMAPI setUTF8(
		FLMUINT				uiElementNumber,
		const FLMBYTE *	pszUtf8,
		FLMUINT				uiBytesInBuffer = 0);

	FINLINE RCODE XFLMAPI setBinary(
		FLMUINT				uiElementNumber,
		const void *		pvBinary,
		FLMUINT				uiBinaryLen)
	{
		return( storeValue( uiElementNumber,
							XFLM_BINARY_TYPE, (FLMBYTE *)pvBinary, uiBinaryLen));
	}

	FINLINE void XFLMAPI setRightTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			setRightTruncated( pVector);
		}
		else
		{
			// Need to set some data value before setting
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE void XFLMAPI setLeftTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			setLeftTruncated( pVector);
		}
		else
		{
			// Need to set some data value before setting
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE void XFLMAPI clearRightTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			clearRightTruncated( pVector);
		}
		else
		{
			// Need to set some data value before clearing
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE void XFLMAPI clearLeftTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			clearLeftTruncated( pVector);
		}
		else
		{
			// Need to set some data value before clearing
			// truncated.
			flmAssert( 0);
		}
	}

	FINLINE FLMBOOL XFLMAPI isRightTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			return( isRightTruncated( pVector));
		}

		return( FALSE);
	}

	FINLINE FLMBOOL XFLMAPI isLeftTruncated(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			return( isLeftTruncated( pVector));
		}

		return( FALSE);
	}

	// Getter methods

	FINLINE FLMUINT64 XFLMAPI getDocumentID( void)
	{
		return( m_ui64DocumentID);
	}

	FINLINE FLMUINT64 XFLMAPI getID(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_ID)) == NULL)
		{
			return( 0);
		}
		else
		{
			return( pVector->ui64ID);
		}
	}

	FINLINE FLMUINT XFLMAPI getNameId(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_NAME_ID)) == NULL)
		{
			return( 0);
		}
		else
		{
			return( pVector->uiNameId);
		}
	}

	FINLINE FLMBOOL XFLMAPI isAttr(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_NAME_ID)) == NULL)
		{
			return( FALSE);
		}
		else
		{
			return( (pVector->uiFlags & VECT_SLOT_IS_ATTR) ? TRUE : FALSE);
		}
	}

	FINLINE FLMBOOL XFLMAPI isDataComponent(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_NAME_ID)) == NULL)
		{
			return( FALSE);
		}
		else
		{
			return( (pVector->uiFlags & VECT_SLOT_IS_DATA) ? TRUE : FALSE);
		}
	}

	FINLINE FLMBOOL XFLMAPI isKeyComponent(
		FLMUINT		uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_NAME_ID)) == NULL)
		{
			return( FALSE);
		}
		else
		{
			return( (pVector->uiFlags & VECT_SLOT_IS_DATA) ? FALSE : TRUE);
		}
	}

	FLMUINT XFLMAPI getDataLength(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) == NULL)
		{
			return( 0);
		}
		else
		{
			return( pVector->uiDataLength);
		}
	}

	FLMUINT XFLMAPI getDataType(
		FLMUINT	uiElementNumber)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) == NULL)
		{
			return( XFLM_UNKNOWN_TYPE);
		}
		else
		{
			return( pVector->uiDataType);
		}
	}

	RCODE XFLMAPI getUTF8Ptr(
		FLMUINT				uiElementNumber,
		const FLMBYTE **	ppszUTF8,
		FLMUINT *			puiBufLen);

	FINLINE RCODE XFLMAPI getINT(
		FLMUINT	uiElementNumber,
		FLMINT *	piNum)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									NULL, piNum)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}

	FINLINE RCODE XFLMAPI getINT64(
		FLMUINT		uiElementNumber,
		FLMINT64 *	pi64Num)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number64( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									NULL, pi64Num)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}

	FINLINE RCODE XFLMAPI getUINT(
		FLMUINT		uiElementNumber,
		FLMUINT *	puiNum)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									puiNum, NULL)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}

	FINLINE RCODE XFLMAPI getUINT64(
		FLMUINT		uiElementNumber,
		FLMUINT64 *	pui64Num)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Number64( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									pui64Num, NULL)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}

	RCODE XFLMAPI getUnicode(
		FLMUINT			uiElementNumber,
		FLMUNICODE **	ppuzUnicode);

	FINLINE RCODE XFLMAPI getUnicode(
		FLMUINT			uiElementNumber,
		IF_DynaBuf *	pBuffer)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Unicode( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									pBuffer)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}
		
	FINLINE RCODE XFLMAPI getUnicode(
		FLMUINT			uiElementNumber,
		FLMUNICODE *	puzUnicode,
		FLMUINT *		puiBufLen)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2Unicode( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									puiBufLen, puzUnicode)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}

	FINLINE RCODE XFLMAPI getUTF8(
		FLMUINT			uiElementNumber,
		FLMBYTE *		pszUTF8,
		FLMUINT *		puiBufLen)
	{
		F_VECTOR_ELEMENT *	pVector;

		return( (RCODE)((pVector = getVector( uiElementNumber,
												VECT_SLOT_HAS_DATA)) != NULL
							 ? flmStorage2UTF8( pVector->uiDataType,
									pVector->uiDataLength,
									(FLMBYTE *)getDataPtr( pVector),
									puiBufLen, pszUTF8)
							 : RC_SET( NE_XFLM_NOT_FOUND)));
	}

	FINLINE RCODE XFLMAPI getBinary(
		FLMUINT			uiElementNumber,
		void *			pvBuffer,
		FLMUINT *		puiLength)
	{
		F_VECTOR_ELEMENT *	pVector;

		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			*puiLength = f_min( (*puiLength), pVector->uiDataLength);
			if (pvBuffer && *puiLength)
			{
				f_memcpy( pvBuffer, getDataPtr( pVector), *puiLength);
			}
			
			return( NE_XFLM_OK);
		}
		else
		{
			*puiLength = 0;
		}

		return( RC_SET( NE_XFLM_NOT_FOUND));
	}

	FINLINE RCODE XFLMAPI getBinary(
		FLMUINT				uiElementNumber,
		IF_DynaBuf *		pBuffer)
	{
		F_VECTOR_ELEMENT *	pVector;

		pBuffer->truncateData( 0);
		if ((pVector = getVector( uiElementNumber, VECT_SLOT_HAS_DATA)) != NULL)
		{
			return( pBuffer->appendData( getDataPtr( pVector),
													pVector->uiDataLength));
		}

		return( RC_SET( NE_XFLM_NOT_FOUND));
	}

	RCODE XFLMAPI outputKey(
		IF_Db *				pDb,
		FLMUINT				uiIndexNum,
		FLMUINT				uiMatchFlags,
		FLMBYTE *			pucKeyBuf,
		FLMUINT				uiKeyBufSize,
		FLMUINT *			puiKeyLen);

	RCODE XFLMAPI outputData(
		IF_Db *				pDb,
		FLMUINT				uiIndexNum,
		FLMBYTE *			pucDataBuf,
		FLMUINT				uiDataBufSize,
		FLMUINT *			puiDataLen);

	RCODE XFLMAPI inputKey(
		IF_Db *				pDb,
		FLMUINT				uiIndexNum,
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen);

	RCODE XFLMAPI inputData(
		IF_Db *				pDb,
		FLMUINT				uiIndexNum,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen);

	// Miscellaneous methods

	void XFLMAPI reset( void);

	FINLINE const void * XFLMAPI getDataPtr(
		FLMUINT	uiElementNumber)
	{
		return( getDataPtr( getVector( uiElementNumber, VECT_SLOT_HAS_DATA)));
	}

private:

	RCODE allocVectorArray(
		FLMUINT	uiElementNumber);

	RCODE storeValue(
		FLMINT				uiElementNumber,
		FLMUINT				uiDataType,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen,
		FLMBYTE **			ppucDataPtr = NULL);

	FINLINE F_VECTOR_ELEMENT * getVector(
		FLMUINT	uiElementNumber,
		FLMUINT	uiTestFlags)
	{
		F_VECTOR_ELEMENT *	pVector;

		if (uiElementNumber >= m_uiNumElements)
		{
			return( NULL);
		}
		pVector = &m_pVectorElements [uiElementNumber];
		if (!(pVector->uiFlags & uiTestFlags))
		{
			return( NULL);
		}
		else
		{
			return( pVector);
		}
	}

	FINLINE FLMBOOL isRightTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		return( (pVector->uiFlags & VECT_SLOT_RIGHT_TRUNCATED)
					? TRUE
					: FALSE);
	}

	FINLINE void setRightTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags |= VECT_SLOT_RIGHT_TRUNCATED;
	}

	FINLINE void clearRightTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags &= (~(VECT_SLOT_RIGHT_TRUNCATED));
	}

	FINLINE FLMBOOL isLeftTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		return( (pVector->uiFlags & VECT_SLOT_LEFT_TRUNCATED)
					? TRUE
					: FALSE);
	}

	FINLINE void setLeftTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags |= VECT_SLOT_LEFT_TRUNCATED;
	}

	FINLINE void clearLeftTruncated(
		F_VECTOR_ELEMENT *	pVector)
	{
		pVector->uiFlags &= (~(VECT_SLOT_LEFT_TRUNCATED));
	}

	FINLINE void * getDataPtr(
		F_VECTOR_ELEMENT *	pVector)
	{
		if (!pVector || !pVector->uiDataLength)
		{
			return( NULL);
		}
		else if (pVector->uiDataLength <= sizeof( FLMUINT))
		{
			return( (void *)&pVector->uiDataOffset);
		}
		else
		{
			return( (void *)(m_pucDataBuf + pVector->uiDataOffset));
		}
	}

	RCODE outputKey(
		IXD *					pIxd,
		FLMUINT				uiMatchFlags,
		FLMBYTE *			pucKeyBuf,
		FLMUINT				uiKeyBufSize,
		FLMUINT *			puiKeyLen,
		FLMUINT				uiSearchKeyFlag);

	RCODE outputData(
		IXD *					pIxd,
		FLMBYTE *			pucDataBuf,
		FLMUINT				uiDataBufSize,
		FLMUINT *			puiDataLen);

	RCODE inputKey(
		IXD *					pIxd,
		const FLMBYTE *	pucKey,
		FLMUINT				uiKeyLen);

	RCODE inputData(
		IXD *					pIxd,
		const FLMBYTE *	pucData,
		FLMUINT				uiDataLen);

#define MIN_VECTOR_ELEMENTS	6
	F_VECTOR_ELEMENT		m_VectorArray [MIN_VECTOR_ELEMENTS];
	F_VECTOR_ELEMENT *	m_pVectorElements;	// Pointer to vector elements
	FLMUINT					m_uiVectorArraySize;	// Size of vector array
	FLMUINT					m_uiNumElements;		// Number of elements actually
															// populated in the array.

	FLMBYTE					m_ucIntDataBuf[ 32];	// Internal data buffer
	FLMBYTE *				m_pucDataBuf;			// Values stored here
	FLMUINT					m_uiDataBufLength;	// Bytes of data allocated
	FLMUINT					m_uiDataBufOffset;	// Current offset into allocated
															// data buffer.
	FLMUINT64				m_ui64DocumentID;		// Document ID;

friend class F_Db;
friend class FSIndexCursor;
friend class FSCollectionCursor;
friend class F_QueryResultSet;
};

// Flags for the m_uiFlags member of the F_Db object

#define FDB_UPDATED_DICTIONARY	0x0001
												// Flag indicating whether the file's
												// local dictionary was updated
												// during the transaction.
#define FDB_DO_TRUNCATE				0x0002
												// Truncate log extents at the end
												// of a transaction.
#define FDB_HAS_FILE_LOCK			0x0004
												//	FDB has a file lock.
#define FDB_FILE_LOCK_SHARED		0x0008
												// File lock is shared.  Update
												// transactions are not allowed when
												// the lock is shared.
#define FDB_FILE_LOCK_IMPLICIT	0x0010
												// File lock is implicit - means file
												// lock was obtained when the update
												// transaction began and cannot be
												// released by a call to FlmDbUnlock.
#define FDB_DONT_KILL_TRANS		0x0020
												// Do not attempt to kill an active
												// read transaction on this database
												// handle.  This is used by FlmDbBackup.
#define FDB_INTERNAL_OPEN			0x0040
												// FDB is an internal one used by a
												// background thread.
#define FDB_DONT_POISON_CACHE		0x0080
												// If blocks are read from disk during
												// a transaction, release them at the LRU
												// end of the cache chain.
#define FDB_UPGRADING				0x0100
												// Database is being upgraded.
#define FDB_REPLAYING_RFL			0x0200
												// Database is being recovered
#define FDB_REPLAYING_COMMIT		0x0400
												// During replay of the RFL, this
												// is an actual call to FlmDbTransCommit.
#define FDB_BACKGROUND_INDEXING	0x0800
												// FDB is being used by a background indexing
												// thread
#define FDB_HAS_WRITE_LOCK			0x1000
												// FDB has the write lock
#define FDB_REBUILDING_DATABASE	0x2000
												// Database is being rebuilt
#define FDB_SWEEP_SCHEDULED		0x4000
												// Sweep operation scheduled due to a
												// dictionary change during the transaction

class F_ThreadInfo : public IF_ThreadInfo, public XF_Base
{
public:

	F_ThreadInfo()
	{
		m_Pool.poolInit( 1024);
		m_uiNumThreads = 0;
		m_pThreadInfoArray = NULL;
	}

	virtual ~F_ThreadInfo()
	{
		m_Pool.poolFree();
	}

	FLMUINT XFLMAPI getNumThreads( void)
	{
		return( m_uiNumThreads);
	}

	FINLINE void XFLMAPI getThreadInfo(
		FLMUINT				uiThreadNum,
		FLMUINT *			puiThreadId,
		FLMUINT *			puiThreadGroup,
		FLMUINT *			puiAppId,
		FLMUINT *			puiStartTime,
		const char **		ppszThreadName,
		const char **		ppszThreadStatus)
	{
		if (uiThreadNum < m_uiNumThreads)
		{
			F_THREAD_INFO *	pThrdInfo = &m_pThreadInfoArray [uiThreadNum];

			*puiThreadId = pThrdInfo->uiThreadId;
			*puiThreadGroup = pThrdInfo->uiThreadGroup;
			*puiAppId = pThrdInfo->uiAppId;
			*puiStartTime = pThrdInfo->uiStartTime;
			*ppszThreadName = pThrdInfo->pszThreadName;
			*ppszThreadStatus = pThrdInfo->pszThreadStatus;
		}
		else
		{
			*puiThreadId = 0;
			*puiThreadGroup = 0;
			*puiAppId = 0;
			*puiStartTime = 0;
			*ppszThreadName = NULL;
			*ppszThreadStatus = NULL;
		}
	}

private:

	F_Pool				m_Pool;
	F_THREAD_INFO *	m_pThreadInfoArray;
	FLMUINT				m_uiNumThreads;

friend class F_DbSystem;

};

/*****************************************************************************
Desc:	Class for performing database backup.
*****************************************************************************/
class F_Backup : public IF_Backup, public XF_Base
{
public:

	F_Backup();
	virtual ~F_Backup();

	FINLINE FLMUINT64 XFLMAPI getBackupTransId( void)
	{
		return( m_ui64TransId);
	}

	FINLINE FLMUINT64 XFLMAPI getLastBackupTransId( void)
	{
		return( m_ui64LastBackupTransId);
	}

	RCODE XFLMAPI backup(
		const char *			pszBackupPath,
		const char *			pszPassword,
		IF_BackupClient *		pClient,
		IF_BackupStatus *		pStatus,
		FLMUINT *				puiIncSeqNum);

	RCODE XFLMAPI endBackup( void);

private:

	void reset( void);

	F_Db *			m_pDb;
	eDbTransType	m_eTransType;
	FLMUINT64		m_ui64TransId;
	FLMUINT64		m_ui64LastBackupTransId;
	FLMUINT			m_uiDbVersion;
	FLMUINT			m_uiBlkChgSinceLastBackup;
	FLMBOOL			m_bTransStarted;
	FLMUINT			m_uiBlockSize;
	FLMUINT			m_uiLogicalEOF;
	FLMUINT			m_uiFirstReqRfl;
	FLMUINT			m_uiIncSeqNum;
	FLMBOOL			m_bCompletedBackup;
	eDbBackupType	m_eBackupType;
	RCODE				m_backupRc;
	FLMBYTE			m_ucNextIncSerialNum[ XFLM_SERIAL_NUM_SIZE];
	char				m_szDbPath[ F_PATH_MAX_SIZE];
	XFLM_DB_HDR		m_dbHdr;

friend class F_Db;
};


/*****************************************************************************
Desc:		An implementation of IF_Backup_Client that backs up to the
			local hard disk.
*****************************************************************************/
class F_DefaultBackupClient : public IF_BackupClient, public XF_Base
{
public:

	F_DefaultBackupClient(
		const char *	pszBackupPath);

	virtual ~F_DefaultBackupClient();

	RCODE XFLMAPI WriteData(
		const void *	pvBuffer,
		FLMUINT			uiBytesToWrite);

	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_BackupClient::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_BackupClient::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_BackupClient::Release());
	}

private:

	char						m_szPath[ F_PATH_MAX_SIZE];
	F_64BitFileHandle *	m_pFileHdl64;
	FLMUINT64				m_ui64Offset;
	RCODE						m_rc;
};

/*****************************************************************************
Desc:		The F_FSRestore class is used to read backup and RFL files from 
			a disk file system.
*****************************************************************************/
class F_FSRestore : public IF_RestoreClient, public XF_Base
{
public:

	virtual ~F_FSRestore();
	F_FSRestore();

	RCODE setup(
		const char *	pszDbPath,
		const char *	pszBackupSetPath,
		const char *	pszRflDir);

	RCODE XFLMAPI openBackupSet( void);

	RCODE XFLMAPI openIncFile(
		FLMUINT			uiFileNum);

	RCODE XFLMAPI openRflFile(
		FLMUINT			uiFileNum);

	RCODE XFLMAPI read(
		FLMUINT			uiLength,
		void *			pvBuffer,
		FLMUINT *		puiBytesRead);

	RCODE XFLMAPI close( void);

	RCODE XFLMAPI abortFile( void);

	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_RestoreClient::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_RestoreClient::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_RestoreClient::Release());
	}

protected:

	IF_FileHdl *			m_pFileHdl;
	F_64BitFileHandle *	m_pFileHdl64;
	FLMUINT64				m_ui64Offset;
	FLMUINT					m_uiDbVersion;
	char						m_szDbPath[ F_PATH_MAX_SIZE];
	char						m_szBackupSetPath[ F_PATH_MAX_SIZE];
	char						m_szRflDir[ F_PATH_MAX_SIZE];
	FLMBOOL					m_bSetupCalled;
	FLMBOOL					m_bOpen;
};

/*****************************************************************************
Desc:		Default implementation of a restore status object than can
			be inherited by a user implementation.
*****************************************************************************/
class F_DefaultRestoreStatus : public IF_RestoreStatus, public XF_Base
{
public:

	F_DefaultRestoreStatus()
	{
	}

	RCODE XFLMAPI reportProgress(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64BytesToDo,
		FLMUINT64)			// ui64BytesDone
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportError(
		eRestoreAction *	peAction,
		RCODE)				// rcErr
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportBeginTrans(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportCommitTrans(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportAbortTrans(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportRemoveData(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiKeyLen,
		FLMBYTE *)			// pucKey
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportInsertData(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiKeyLen,
		FLMBYTE *)			// pucKey
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportReplaceData(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiKeyLen,
		FLMBYTE *)			// pucKey
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportLFileCreate(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiLfNum
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportLFileUpdate(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiLfNum,
		FLMUINT,				// uiRootBlk,
		FLMUINT64,			// ui64NextNodeId,
		FLMUINT64,			// ui64FirstDocId,
		FLMUINT64			// ui64LastDocId
		)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportUpdateDict(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiDictType,
		FLMUINT,				// uiDictNum,
		FLMBOOL)				// bDeleting
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportIndexSuspend(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiIndexNum
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportIndexResume(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiIndexNum
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportReduce(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT)				// uiCount
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportUpgrade(
		eRestoreAction *	peAction,
		FLMUINT64,			// ui64TransId,
		FLMUINT,				// uiOldDbVersion,
		FLMUINT)				// uiNewDbVersion
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportEnableEncryption(
		eRestoreAction *	peAction,
		FLMUINT64			// ui64TransId
		)
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportWrapKey(
		eRestoreAction *	peAction,
		FLMUINT64)			// ui64TransId
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportOpenRflFile(
		eRestoreAction *	peAction,
		FLMUINT)				// uiFileNum
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI reportRflRead(
		eRestoreAction *	peAction,
		FLMUINT,				// uiFileNum,
		FLMUINT)				// uiBytesRead
	{
		*peAction = XFLM_RESTORE_ACTION_CONTINUE;
		return( NE_XFLM_OK);
	}

	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_RestoreStatus::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_RestoreStatus::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_RestoreStatus::Release());
	}
};

// Indexing actions

typedef enum IxActionTag
{
	IX_UNLINK_NODE = 0,
	IX_LINK_NODE,
	IX_DEL_NODE_VALUE,
	IX_ADD_NODE_VALUE,
	IX_LINK_AND_ADD_NODE
} IxAction;

typedef struct ElmAttrStateInfo
{
	FLMUINT		uiDictType;
	FLMUINT		uiDictNum;
	FLMUINT		uiState;
	FLMUINT64	ui64StateChangeCount;
} ELM_ATTR_STATE_INFO;

typedef struct KEY_GEN_INFO
{
	FLMUINT64		ui64DocumentID;
	IXD *				pIxd;
	FLMBOOL			bIsAsia;
	FLMBOOL			bIsCompound;
	CDL_HDR *		pCdlTbl;
	FLMBOOL			bUseSubtreeNodes;
	FLMBOOL			bAddKeys;
	FLMBYTE *		pucKeyBuf;
	FLMBYTE *		pucData;
	FLMUINT			uiDataBufSize;
	FLMBOOL			bDataBufAllocated;
} KEY_GEN_INFO;

typedef struct OLD_NODE_DATA
{
	eDomNodeType	eNodeType;
	FLMUINT			uiCollection;
	FLMUINT64		ui64NodeId;
	FLMUINT			uiNameId;
	FLMBYTE *		pucData;
	FLMUINT			uiDataLen;
} OLD_NODE_DATA;

/*****************************************************************************
Desc: Thread's database object - returned by dbOpen, dbCreate in F_DbSystem class
*****************************************************************************/
class F_OldNodeList : public XF_RefCount, public XF_Base
{
public:

	F_OldNodeList()
	{
		m_pNodeList = NULL;
		m_uiListSize = 0;
		m_uiNodeCount = 0;
		m_pool.poolInit( 512);
	}
	
	~F_OldNodeList();
	
	FLMBOOL findNodeInList(
		eDomNodeType	eNodeType,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		FLMUINT			uiNameId,
		FLMBYTE **		ppucData,
		FLMUINT *		puiDataLen,
		FLMUINT *		puiInsertPos);
		
	RCODE addNodeToList(
		F_Db *			pDb,
		F_DOMNode *		pNode);
		
	void resetList( void);
	
	FINLINE FLMUINT getNodeCount( void)
	{
		return( m_uiNodeCount);
	}
	
private:

	OLD_NODE_DATA *	m_pNodeList;
	F_Pool				m_pool;
	FLMUINT				m_uiListSize;
	FLMUINT				m_uiNodeCount;
};

/*****************************************************************************
Desc: Thread's database object - returned by dbOpen, dbCreate in F_DbSystem class
*****************************************************************************/
class F_Db : public IF_Db, public XF_Base
{
public:

	F_Db(
		FLMBOOL	bInternalOpen);
		
	virtual ~F_Db();

	RCODE XFLMAPI transBegin(
		eDbTransType			eTransType,
		FLMUINT					uiMaxLockWait = XFLM_NO_TIMEOUT,
		FLMUINT					uiFlags = 0,
		XFLM_DB_HDR *			pDbHeader = NULL);

	RCODE XFLMAPI transBegin(
		IF_Db *					pDb);

	RCODE XFLMAPI transCommit(
		FLMBOOL *				pbEmpty = NULL);

	RCODE XFLMAPI transAbort( void);

	FINLINE eDbTransType XFLMAPI getTransType( void)
	{
		return( m_eTransType);
	}

	RCODE XFLMAPI doCheckpoint(
		FLMUINT					uiTimeout);

	RCODE XFLMAPI dbLock(
		eDbLockType				eLockType,
		FLMINT					iPriority,
		FLMUINT					uiTimeout);

	RCODE XFLMAPI dbUnlock( void);

	RCODE XFLMAPI getLockType(
		eDbLockType *			peLockType,
		FLMBOOL *				pbImplicit);

	RCODE XFLMAPI getLockInfo(
		FLMINT					iPriority,
		eDbLockType *			peCurrLockType,
		FLMUINT *				puiThreadId,
		FLMUINT *				puiNumExclQueued,
		FLMUINT *				puiNumSharedQueued,
		FLMUINT *				puiPriorityCount);

	RCODE dupTrans(
		FLMUINT64	ui64TransId);

	RCODE demoteTrans( void);

	RCODE cancelTrans(
		FLMUINT64				ui64TransId);

	RCODE getCommitCnt(
		FLMUINT64 *				pui64CommitCount);

	// Index methods

	RCODE XFLMAPI indexStatus(
		FLMUINT					uiIndexNum,
		XFLM_INDEX_STATUS *	pIndexStatus);

	RCODE XFLMAPI indexGetNext(
		FLMUINT *				puiIndexNum);

	RCODE XFLMAPI indexSuspend(
		FLMUINT					uiIndexNum);

	RCODE XFLMAPI indexResume(
		FLMUINT					uiIndexNum);

	// Retrieval Functions

	RCODE	XFLMAPI keyRetrieve(
		FLMUINT					uiIndex,
		IF_DataVector *		ifpSearchKey,
		FLMUINT					uiFlags,
		IF_DataVector *		ifpFoundKey);

	RCODE XFLMAPI enableEncryption( void);
		
	RCODE XFLMAPI wrapKey(
		const char *	pszPassword = NULL);

	RCODE XFLMAPI rollOverDbKey( void);
			
	RCODE XFLMAPI changeItemState(
		FLMUINT					uiDictType,
		FLMUINT					uiDictNum,
		const char *			pszState);

	RCODE XFLMAPI reduceSize(
		FLMUINT     			uiCount,
		FLMUINT *				puiCountRV);

	RCODE XFLMAPI upgrade(
		IF_UpgradeClient *	pUpgradeClient);

	RCODE XFLMAPI createRootElement(
		FLMUINT					uiCollection,
		FLMUINT					uiNameId,
		IF_DOMNode **			ppElementNode,
		FLMUINT64 *				pui64NodeId = NULL);

	RCODE XFLMAPI createDocument(
		FLMUINT					uiCollection,
		IF_DOMNode **			ppDocumentNode,
		FLMUINT64 *				pui64NodeId = NULL);

	RCODE XFLMAPI getFirstDocument(
		FLMUINT					uiCollection,
		IF_DOMNode **			ppDocumentNode);

	RCODE XFLMAPI getLastDocument(
		FLMUINT					uiCollection,
		IF_DOMNode **			ppDocumentNode);

	RCODE XFLMAPI getDocument(
		FLMUINT					uiCollection,
		FLMUINT					uiFlags,
		FLMUINT64				ui64DocumentId,
		IF_DOMNode **			ppDocumentNode);

	RCODE XFLMAPI documentDone(
		FLMUINT					uiCollection,
		FLMUINT64				ui64RootId);

	RCODE XFLMAPI documentDone(
		IF_DOMNode *			pDocNode);

	FINLINE RCODE XFLMAPI createElementDef(
		const char *			pszNamespaceURI,
		const char *			pszElementName,
		FLMUINT					uiDataType,
		FLMUINT * 				puiElementNameId = NULL,
		IF_DOMNode **			ppDocumentNode = NULL)
	{
		return( createElemOrAttrDef( TRUE, FALSE, pszNamespaceURI,
			pszElementName, uiDataType, FALSE,
			puiElementNameId, (F_DOMNode **)ppDocumentNode));
	}

	FINLINE RCODE XFLMAPI createElementDef(
		const FLMUNICODE *	puzNamespaceURI,
		const FLMUNICODE *	puzElementName,
		FLMUINT					uiDataType,
		FLMUINT * 				puiElementNameId = NULL,
		IF_DOMNode **			ppDocumentNode = NULL)
	{
		return( createElemOrAttrDef( TRUE, TRUE, puzNamespaceURI, 
			puzElementName, uiDataType, FALSE,
			puiElementNameId, (F_DOMNode **)ppDocumentNode));
	}

	FINLINE RCODE XFLMAPI createUniqueElmDef(
		const char *			pszNamespaceURI,
		const char *			pszElementName,
		FLMUINT * 				puiElementNameId = NULL,
		IF_DOMNode **			ppDocumentNode = NULL)
	{
		return( createElemOrAttrDef( TRUE, FALSE, pszNamespaceURI,
			pszElementName, XFLM_NODATA_TYPE, TRUE,
			puiElementNameId, (F_DOMNode **)ppDocumentNode));
	}

	FINLINE RCODE XFLMAPI createUniqueElmDef(
		const FLMUNICODE *	puzNamespaceURI,
		const FLMUNICODE *	puzElementName,
		FLMUINT * 				puiElementNameId = NULL,
		IF_DOMNode **			ppDocumentNode = NULL)
	{
		return( createElemOrAttrDef( TRUE, TRUE, puzNamespaceURI, 
			puzElementName, XFLM_NODATA_TYPE, TRUE,
			puiElementNameId, (F_DOMNode **)ppDocumentNode));
	}

	RCODE XFLMAPI getElementNameId(
		const char *			pszNamespaceURI,
		const char *			pszElementName,
		FLMUINT *				puiElementNameId);

	RCODE XFLMAPI getElementNameId(
		const FLMUNICODE *	puzNamespaceURI,
		const FLMUNICODE *	puzElementName,
		FLMUINT *				puiElementNameId);

	FINLINE RCODE XFLMAPI createAttributeDef(
		const char *			pszNamespaceURI,
		const char *			pszAttributeName,
		FLMUINT					uiDataType,
		FLMUINT * 				puiAttributeNameId,
		IF_DOMNode **			ppDocumentNode = NULL)
	{
		return( createElemOrAttrDef( FALSE, FALSE, pszNamespaceURI, 
			pszAttributeName, uiDataType, FALSE, puiAttributeNameId,
			(F_DOMNode **)ppDocumentNode));
	}

	FINLINE RCODE XFLMAPI createAttributeDef(
		const FLMUNICODE *	puzNamespaceURI,
		const FLMUNICODE *	puzAttributeName,
		FLMUINT					uiDataType,
		FLMUINT * 				puiAttributeNameId,
		IF_DOMNode **			ppDocumentNode = NULL)
	{
		return( createElemOrAttrDef( FALSE, TRUE, puzNamespaceURI, 
			puzAttributeName, uiDataType, FALSE, puiAttributeNameId,
			(F_DOMNode **)ppDocumentNode));
	}

	RCODE XFLMAPI getAttributeNameId(
		const char *			pszNamespaceURI,
		const char *			pszAttributeName,
		FLMUINT *				puiAttributeNameId);

	RCODE XFLMAPI getAttributeNameId(
		const FLMUNICODE *	puzNamespaceURI,
		const FLMUNICODE *	puzAttributeName,
		FLMUINT *				puiAttributeNameId);

	FINLINE RCODE XFLMAPI createPrefixDef(
		const char *			pszPrefixName,
		FLMUINT *				puiPrefixNumber)
	{
		return( createPrefixDef( FALSE, pszPrefixName, puiPrefixNumber));
	}

	FINLINE RCODE XFLMAPI createPrefixDef(
		const FLMUNICODE *	puzPrefixName,
		FLMUINT * 				puiPrefixNumber)
	{
		return( createPrefixDef( TRUE, puzPrefixName, puiPrefixNumber));
	}

	RCODE XFLMAPI getPrefixId(
		const char *			pszPrefixName,
		FLMUINT *				puiPrefixNumber);

	RCODE XFLMAPI getPrefixId(
		const FLMUNICODE *	puzPrefixName,
		FLMUINT *				puiPrefixNumber);

	FINLINE RCODE XFLMAPI createEncDef(
		const char *			pszEncType,
		const char *			pszEncName,
		FLMUINT					uiKeySize = 0,
		FLMUINT *				puiEncDefNumber = NULL)
	{
		return( createEncDef( FALSE, pszEncType, pszEncName, 
			uiKeySize, puiEncDefNumber));
	}

	FINLINE RCODE XFLMAPI createEncDef(
		const FLMUNICODE *	puzEncType,
		const FLMUNICODE *	puzEncName,
		FLMUINT					uiKeySize = 0,
		FLMUINT *				puiEncDefNumber = NULL)
	{
		return( createEncDef( TRUE, puzEncType, puzEncName, 
			uiKeySize, puiEncDefNumber));
	}

	RCODE XFLMAPI getEncDefId(
		const char *			pszEncDefName,
		FLMUINT *				puiEncDefNumber);

	RCODE XFLMAPI getEncDefId(
		const FLMUNICODE *	puzEncDefName,
		FLMUINT *				puiEncDefNumber);

	FINLINE RCODE XFLMAPI createCollectionDef(
		const char *			pszCollectionName,
		FLMUINT * 				puiCollectionNumber,
		FLMUINT					uiEncNumber = 0)
	{
		return( createCollectionDef( FALSE, pszCollectionName,
			puiCollectionNumber, uiEncNumber));
	}

	FINLINE RCODE XFLMAPI createCollectionDef(
		const FLMUNICODE *	puzCollectionName,
		FLMUINT * 				puiCollectionNumber,
		FLMUINT					uiEncNumber = 0)
	{
		return( createCollectionDef( TRUE, puzCollectionName,
			puiCollectionNumber, uiEncNumber));
	}

	RCODE XFLMAPI getCollectionNumber(
		const char *			pszCollectionName,
		FLMUINT *				puiCollectionNumber);

	RCODE XFLMAPI getCollectionNumber(
		const FLMUNICODE *	puzCollectionName,
		FLMUINT *				puiCollectionNumber);

	RCODE XFLMAPI getIndexNumber(
		const char *			pszIndexName,
		FLMUINT *				puiIndexNumber);

	RCODE XFLMAPI getIndexNumber(
		const FLMUNICODE *	puzIndexName,
		FLMUINT *				puiIndexNumber);

	RCODE XFLMAPI getDictionaryDef(
		FLMUINT					uiDictType,
		FLMUINT					uiDictNumber,
		IF_DOMNode **			ppDocumentNode);

	RCODE XFLMAPI getDictionaryName(
		FLMUINT					uiDictType,
		FLMUINT					uiDictNumber,
		char *					pszName,
		FLMUINT *				puiNameBufSize,
		char *					pszNamespace = NULL,
		FLMUINT *				puiNamespaceBufSize = NULL);

	RCODE XFLMAPI getDictionaryName(
		FLMUINT					uiDictType,
		FLMUINT					uiDictNumber,
		FLMUNICODE *			puzName,
		FLMUINT *				puiNameBufSize,
		FLMUNICODE *			puzNamespace = NULL,
		FLMUINT *				puiNamespaceBufSize = NULL);

	RCODE XFLMAPI getNode(
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		IF_DOMNode **			ifppNode)
	{
		return( getNode( uiCollection, 
			ui64NodeId, XFLM_EXACT, (F_DOMNode **)ifppNode));
	}

	FINLINE RCODE getNode(
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		F_DOMNode **			ppNode)
	{
		return( getNode( uiCollection, ui64NodeId, XFLM_EXACT, ppNode));
	}

	FINLINE RCODE getNextNode(
		FLMUINT					uiCollection,
		FLMUINT64				ui64NodeId,
		F_DOMNode **			ppNode)
	{
		return( getNode( uiCollection, ui64NodeId, XFLM_EXCL, ppNode));
	}

	RCODE XFLMAPI getAttribute(
		FLMUINT					uiCollection,
		FLMUINT64				ui64ElementId,
		FLMUINT					uiAttrName,
		IF_DOMNode **			ppNode);

	RCODE XFLMAPI getDataType(
		FLMUINT					uiDictType,
		FLMUINT					uiNameId,
		FLMUINT *				puiDataType);

	RCODE XFLMAPI backupBegin(
		eDbBackupType			eBackupType,
		eDbTransType			eTransType,
		FLMUINT					uiMaxLockWait,
		IF_Backup **			ppBackup);

	void XFLMAPI getRflFileName(
		FLMUINT					uiFileNum,
		FLMBOOL					bBaseOnly,
		char *					pszFileName,
		FLMUINT *				puiFileNameBufSize,
		FLMBOOL *				pbNameTruncated = NULL);

	RCODE XFLMAPI import(
		IF_IStream *			pIStream,
		FLMUINT					uiCollection,
		IF_DOMNode *			pNodeToLinkTo = NULL,
		eNodeInsertLoc			eInsertLoc = XFLM_LAST_CHILD,
		XFLM_IMPORT_STATS *	pImportStats = NULL);

	RCODE XFLMAPI importDocument(
		IF_IStream *			ifpStream,
		FLMUINT					uiCollection,
		IF_DOMNode **			ppDocumentNode = NULL,
		XFLM_IMPORT_STATS *	pImportStats = NULL);

	RCODE XFLMAPI exportXML(
		IF_DOMNode *			pStartNode,
		IF_OStream *			pOStream,
		eExportFormatType		eFormat = XFLM_EXPORT_INDENT);
		
	RCODE XFLMAPI setNextNodeId(
		FLMUINT					uiCollection,
		FLMUINT64				ui64NextNodeId);

	RCODE XFLMAPI setNextDictNum(
		FLMUINT					uiDictType,
		FLMUINT					uiDictNumber);

	// Configuration methods

	RCODE XFLMAPI setRflKeepFilesFlag(
		FLMBOOL					bKeep);

	RCODE XFLMAPI getRflKeepFlag(
		FLMBOOL *				pbKeep);

	RCODE XFLMAPI setRflDir(
		const char *			pszNewRflDir);

	void XFLMAPI getRflDir(
		char *					pszRflDir);

	RCODE XFLMAPI getRflFileNum(
		FLMUINT *				puiRflFileNum);

	RCODE XFLMAPI getHighestNotUsedRflFileNum(
		FLMUINT *				puiHighestNotUsedRflFileNum);

	RCODE XFLMAPI setRflFileSizeLimits(
		FLMUINT					uiMinRflSize,
		FLMUINT					uiMaxRflSize);

	RCODE XFLMAPI getRflFileSizeLimits(
		FLMUINT *				puiRflMinFileSize,
		FLMUINT *				puiRflMaxFileSize);

	RCODE XFLMAPI rflRollToNextFile( void);

	RCODE XFLMAPI setKeepAbortedTransInRflFlag(
		FLMBOOL					bKeep);

	RCODE XFLMAPI getKeepAbortedTransInRflFlag(
		FLMBOOL *				pbKeep);

	RCODE XFLMAPI setAutoTurnOffKeepRflFlag(
		FLMBOOL					bAutoTurnOff);

	RCODE XFLMAPI getAutoTurnOffKeepRflFlag(
		FLMBOOL *				pbAutoTurnOff);

	FINLINE void XFLMAPI setFileExtendSize(
		FLMUINT					uiFileExtendSize)
	{
		m_pDatabase->m_uiFileExtendSize = uiFileExtendSize;
	}

	FINLINE FLMUINT XFLMAPI getFileExtendSize( void)
	{
		return( m_pDatabase->m_uiFileExtendSize);
	}

	FINLINE void XFLMAPI setAppData(
		void *			pvAppData)
	{
		m_pvAppData = pvAppData;
	}

	FINLINE void * XFLMAPI getAppData( void)
	{
		return( m_pvAppData);
	}

	FINLINE void XFLMAPI setDeleteStatusObject(
		IF_DeleteStatus *		pDeleteStatus)
	{
		if (m_pDeleteStatus)
		{
			m_pDeleteStatus->Release();
		}
		if ((m_pDeleteStatus = pDeleteStatus) != NULL)
		{
			m_pDeleteStatus->AddRef();
		}
	}

	FINLINE void XFLMAPI setCommitClientObject(
		IF_CommitClient *		pCommitClient)
	{
		if (m_pCommitClient)
		{
			m_pCommitClient->Release();
		}
		
		m_pCommitClient = pCommitClient;
		
		if (m_pCommitClient)
		{
			m_pCommitClient->AddRef();
		}
	}

	FINLINE void XFLMAPI setIndexingClientObject(
		IF_IxClient *	pIxClient)
	{
		if (m_pIxClient)
		{
			m_pIxClient->Release();
		}
		m_pIxClient = pIxClient;
		if (m_pIxClient)
		{
			m_pIxClient->AddRef();
		}
	}

	FINLINE void XFLMAPI setIndexingStatusObject(
		IF_IxStatus *			ifpIxStatus)
	{
		if (m_pIxStatus)
		{
			m_pIxStatus->Release();
		}
		m_pIxStatus = ifpIxStatus;
		if (m_pIxStatus)
		{
			m_pIxStatus->AddRef();
		}
	}

	// Configuration information getting methods

	FINLINE FLMUINT XFLMAPI getDbVersion( void)
	{
		return( (FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion);
	}

	FINLINE FLMUINT XFLMAPI getBlockSize( void)
	{
		return( m_pDatabase->m_uiBlockSize);
	}

	FINLINE FLMUINT XFLMAPI getDefaultLanguage( void)
	{
		return( m_pDatabase->m_uiDefaultLanguage);
	}

	FINLINE FLMUINT64 XFLMAPI getTransID( void)
	{
		if (m_eTransType != XFLM_NO_TRANS)
		{
			return( m_ui64CurrTransID);
		}
		else if (m_uiFlags & FDB_HAS_FILE_LOCK)
		{
			return( m_pDatabase->m_lastCommittedDbHdr.ui64CurrTransID);
		}

		return( 0);
	}

	void XFLMAPI getCheckpointInfo(
		XFLM_CHECKPOINT_INFO *	pCheckpointInfo);

	RCODE XFLMAPI getDbControlFileName(
		char *					pszControlFileName,
		FLMUINT					uiControlFileBufSize)
	{
		RCODE		rc = NE_XFLM_OK;
		FLMUINT	uiLen = f_strlen( m_pDatabase->m_pszDbPath);

		if (uiLen + 1 > uiControlFileBufSize)
		{
			uiLen = uiControlFileBufSize - 1;
			rc = RC_SET( NE_XFLM_BUFFER_OVERFLOW);
		}
		f_memcpy( pszControlFileName, m_pDatabase->m_pszDbPath, uiLen);
		pszControlFileName [uiLen] = 0;
		return( rc);
	}

	FINLINE FLMBOOL threadWaitingLock( void)
	{
		return( m_pDatabase->m_pDatabaseLockObj->ThreadWaitingLock());
	}

	RCODE XFLMAPI getLockWaiters(
		IF_LockInfoClient *	pLockInfo);

	RCODE XFLMAPI getLastBackupTransID(
		FLMUINT64 *				pui64LastBackupTransID);

	RCODE XFLMAPI getBlocksChangedSinceBackup(
		FLMUINT *				puiBlocksChangedSinceBackup);

	RCODE XFLMAPI getNextIncBackupSequenceNum(
		FLMUINT *				puiNextIncBackupSequenceNum);

	void XFLMAPI getSerialNumber(
		char *					pucSerialNumber);

	RCODE XFLMAPI getDiskSpaceUsage(
		FLMUINT64 *				pui64DataSize,
		FLMUINT64 *				pui64RollbackSize,
		FLMUINT64 *				pui64RflSize);

	FINLINE RCODE XFLMAPI getMustCloseRC( void)
	{
		return( m_pDatabase->m_rcMustClose);
	}

	FINLINE RCODE XFLMAPI getAbortRC( void)
	{
		return( m_AbortRc);
	}

	FINLINE RCODE startTransaction(
		eDbTransType	eReqTransType,
		FLMBOOL *		pbStartedTrans)
	{
		RCODE		rc;

		if( m_eTransType != XFLM_NO_TRANS)
		{
			return( RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_TRANS_OP));
		}

		if( !pbStartedTrans)
		{
			return( RC_SET( NE_XFLM_NO_TRANS_ACTIVE));
		}

		if( RC_BAD( rc = transBegin( eReqTransType)))
		{
			return( rc);
		}

		*pbStartedTrans = TRUE;
		return( NE_XFLM_OK);
	}

	FINLINE RCODE checkTransaction(
		eDbTransType	eReqTransType,
		FLMBOOL *		pbStartedTrans)
	{
		if( m_AbortRc)
		{
			return( m_AbortRc);
		}
		else if( m_eTransType >= eReqTransType)
		{
			return( NE_XFLM_OK);
		}

		return( startTransaction( eReqTransType, pbStartedTrans));
	}

	FINLINE void XFLMAPI setMustAbortTrans(
		RCODE		rc)
	{
		if( RC_BAD( rc) && RC_OK( m_AbortRc))
		{
			m_AbortRc = rc;
		}
	}

	RCODE getDictionary(
		F_Dict ** 				ppDict);

	FINLINE RCODE checkState(
		const char *			pszFileName,
		FLMINT					iLineNumber)
	{
		RCODE	rc = NE_XFLM_OK;

		if (m_bMustClose)
		{
			m_pDatabase->logMustCloseReason( pszFileName, iLineNumber);
			rc = RC_SET( NE_XFLM_MUST_CLOSE_DATABASE);
		}
		return( rc);
	}

	RCODE getElmAttrInfo(
		FLMUINT					uiType,
		FLMUINT64				ui64DocumentID,
		F_AttrElmInfo *		pDefInfo,
		FLMBOOL					bOpeningDict,
		FLMBOOL					bDeleting);

	RCODE getCollectionDef(
		FLMUINT64				ui64DocumentID,
		FLMUNICODE **			ppuzCollectionName,
		FLMUINT *				puiCollectionNumber,
		FLMUINT *				puiEncId);

	RCODE getPrefixDef(
		F_Dict *					pDict,
		FLMUINT64				ui64DocumentID,
		FLMUNICODE **			ppuzPrefixName,
		FLMUINT *				puiPrefixNumber);

	RCODE getEncDefDef(
		F_Dict *					pDict,
		FLMUINT64				ui64DocumentID,
		FLMUNICODE **			ppuzEncDefName,
		FLMUINT *				puiEncDefNumber,
		FLMUINT *				puiEncDefKeySize,
		F_CCS **					ppCcs);

	RCODE getIndexDef(
		FLMUINT64				ui64DocumentID,
		FLMUNICODE **			ppuzIndexName,
		FLMUINT *				puiIndexNumber,
		FLMUINT *				puiCollectionNumber,
		FLMUINT *				puiLanguage,
		FLMUINT *				puiFlags,
		FLMUINT64 *				pui64LastDocIndexed,
		FLMUINT *				puiEncId,
		F_DOMNode **			ppNode,
		FLMBOOL					bOpeningDict,
		FLMBOOL					bDeleting);

	RCODE getIndexComponentDef(
		F_Dict *					pDict,
		F_DOMNode *				pElementNode,
		FLMUINT					uiElementId,
		IXD *						pIxd,
		ICD *						pIcd);

	RCODE getNameTable(
		F_NameTable **			ppNameTable);

	FINLINE F_Database * getDatabase( void)
	{
		return m_pDatabase;
	}

	RCODE backgroundIndexBuild(
		F_Thread *				pThread,
		FLMBOOL *				pbShutdown,
		FLMINT *					piErrorLine);

	FINLINE FLMUINT getLogicalEOF( void)
	{
		return( m_uiLogicalEOF);
	}

	// Key Collector object, used when checking indexes

	FINLINE void setKeyCollector(
		F_KeyCollector *		pKeyColl)
	{
		m_pKeyColl = pKeyColl;
	}

	FINLINE F_KeyCollector * getKeyCollector( void)
	{
		return m_pKeyColl;
	}
	
	RCODE waitForMaintenanceToComplete( void);

	FINLINE F_Dict * getDict( void)
	{
		return( m_pDict);
	}
	
	FINLINE F_OldNodeList * getOldNodeList( void)
	{
		return( m_pOldNodeList);
	}

	void removeCollectionNodes(
		FLMUINT		uiCollection,
		FLMUINT64	ui64TransId);

private:

	RCODE createElemOrAttrDef(
		FLMBOOL					bElement,
		FLMBOOL					bUnicode,
		const void *			pvNamespaceURI,
		const void *			pvLocalName,
		FLMUINT					uiDataType,
		FLMBOOL					bUniqueChildElms,
		FLMUINT * 				puiNameId,
		F_DOMNode **			ppRootNode);

	RCODE createPrefixDef(
		FLMBOOL					bUnicode,
		const void *			pvName,
		FLMUINT * 				puiPrefixNumber);

	RCODE createCollectionDef(
		FLMBOOL					bUnicode,
		const void *			pvName,
		FLMUINT * 				puiCollectionNumber,
		FLMUINT					uiEncDefId);

	RCODE createEncDef(
		FLMBOOL			bUnicode,
		const void *	pvEncType,
		const void *	pvEncName,
		FLMUINT			uiKeySize,
		FLMUINT * 		puiEncDefId);

	// This routine assumes that the database mutex is locked
	FINLINE void linkToDict(
		F_Dict *					pDict)
	{
		if (pDict != m_pDict)
		{
			if (m_pDict)
			{
				unlinkFromDict();
			}
			if ((m_pDict = pDict) != NULL)
			{
				pDict->incrUseCount();
			}
		}
	}

	// This routine assumes the database mutex is locked.
	FINLINE void unlinkFromDict( void)
	{
		if (m_pDict)
		{

			// If the use count goes to zero and the F_Dict is not the first one
			// in the file's list or it is not linked to a file, unlink the F_Dict
			// object from its database and delete it.

			if (!m_pDict->decrUseCount() &&
		 		(m_pDict->getPrev() || !m_pDict->getDatabase()))
			{
				m_pDict->unlinkFromDatabase();
			}
			m_pDict = NULL;
		}
	}

	RCODE linkToDatabase(
		F_Database *		pDatabase);

	void unlinkFromDatabase();

	RCODE initDbFiles(
		const char *			pszRflDir,
		const char *			pszDictFileName,
		const char *			pszDictBuf,
		XFLM_CREATE_OPTS *	pCreateOpts);

	RCODE beginBackgroundTrans(
		F_Thread *			pThread);

	RCODE beginTrans(
		eDbTransType		eTransType,
		FLMUINT				uiMaxLockWait = XFLM_NO_TIMEOUT,
		FLMUINT				uiFlags = 0,
		XFLM_DB_HDR *		pDbHdr = NULL);

	RCODE beginTrans(
		F_Db *	pDb);

	RCODE	commitTrans(
		FLMUINT				uiNewLogicalEOF,
		FLMBOOL				bForceCheckpoint,
		FLMBOOL *			pbEmpty = NULL);

	RCODE	abortTrans(
		FLMBOOL				bOkToLogAbort = TRUE);

	RCODE readRollbackLog(
		FLMUINT				uiLogEOF,
		FLMUINT *			puiCurrAddr,
		F_BLK_HDR *			pBlkHdr,
		FLMBOOL *			pbIsBeforeImageBlk);

	RCODE processBeforeImage(
		FLMUINT				uiLogEOF,
		FLMUINT *			puiCurrAddrRV,
		F_BLK_HDR *			pBlkHdr,
		FLMBOOL				bDoingRecovery,
		FLMUINT64			ui64MaxTransID);

	RCODE physRollback(
		FLMUINT				uiLogEOF,
		FLMUINT				uiFirstLogBlkAddr,
		FLMBOOL				bDoingRecovery,
		FLMUINT64			ui64MaxTransID);

	void completeOpenOrCreate(
		RCODE					rc,
		FLMBOOL				bNewDatabase);

	RCODE startBackgroundIndexing( void);

	void unlinkFromTransList(
		FLMBOOL				bCommitting);

	RCODE lockExclusive(
		FLMUINT				uiMaxLockWait);

	void unlockExclusive( void);

	RCODE readDictionary( void);

	RCODE dictCreate(
		const char *		pszDictPath,
		const char *		pszDictBuf);

	RCODE dictOpen( void);

	RCODE dictReadLFH( void);

	RCODE dictReadDefs(
		FLMUINT				uiDictType);

	RCODE dictClone( void);

	RCODE createNewDict( void);

	FINLINE void getDbHdrInfo(
		XFLM_DB_HDR *		pDbHdr)
	{
		// IMPORTANT NOTE: Any changes to this method should also be
		// mirrored with changes to the other getDbHdrInfo call - see below.
		
		m_ui64CurrTransID = pDbHdr->ui64CurrTransID;
		m_uiLogicalEOF = (FLMUINT)pDbHdr->ui32LogicalEOF;

		// If we are doing a read transaction, this is only needed
		// if we are checking the database.

		m_uiFirstAvailBlkAddr = (FLMUINT)pDbHdr->ui32FirstAvailBlkAddr;
	}
	
	FINLINE void getDbHdrInfo(
		F_Db *	pDb)
	{
		m_ui64CurrTransID = pDb->m_ui64CurrTransID;
		m_uiLogicalEOF = pDb->m_uiLogicalEOF;

		// If we are doing a read transaction, this is only needed
		// if we are checking the database.

		m_uiFirstAvailBlkAddr = pDb->m_uiFirstAvailBlkAddr;
	}
	
	FINLINE FLMBOOL okToCommitTrans( void)
	{
		return( m_eTransType == XFLM_READ_TRANS ||
				  m_AbortRc == NE_XFLM_OK
				  ? TRUE
				  : FALSE);
	}

	RCODE processDupKeys(
		IXD *	pIxd);

	RCODE keysCommit(
		FLMBOOL				bCommittingTrans,
		FLMBOOL				bSortKeys = TRUE);

	RCODE refUpdate(
		LFILE *				pLFile,
		IXD *					pIxd,
		KREF_ENTRY *		pKrefEntry,
		FLMBOOL				bNormalUpdate);

	FINLINE RCODE flushKeys( void)
	{
		RCODE	rc = NE_XFLM_OK;

		if( m_bKrefSetup)
		{
			if( m_uiKrefCount)
			{
				if (RC_BAD( rc = keysCommit( FALSE)))
				{
					goto Exit;
				}
			}
			
			m_pKrefReset = m_pKrefPool->poolMark();
		}

	Exit:

		return( rc);
	}

	RCODE krefCntrlCheck( void);

	void krefCntrlFree( void);

	FINLINE FLMBOOL isKrefOverThreshold( void)
	{
		if( (((m_pKrefPool->getBlockSize() * 3) - 250) <= m_uiTotalKrefBytes) ||
			m_uiKrefCount > (m_uiKrefTblSize - 128))
		{
			return( TRUE);
		}
	
		return( FALSE);
	}

	RCODE addToKrefTbl(
		FLMUINT				uiKeyLen,
		FLMUINT				uiDataLen);

	RCODE verifyKeyContext(
		FLMBOOL *			pbVerified);

	RCODE buildContext(
		ICD *					pIcd,
		FLMUINT				uiKeyLen,
		FLMUINT				uiDataLen);

	RCODE buildData(
		ICD *					pIcd,
		FLMUINT				uiKeyLen,
		FLMUINT				uiDataLen);

	RCODE finishKeyComponent(
		ICD *		pIcd,
		FLMUINT	uiKeyLen);
		
	RCODE genTextKeyComponents(
		F_DOMNode *	pNode,
		ICD *			pIcd,
		FLMUINT		uiKeyLen,
		FLMBYTE **	ppucTmpBuf,
		FLMUINT *	puiTmpBufSize,
		void **		ppvMark);
		
	RCODE genOtherKeyComponent(
		F_DOMNode *	pNode,
		ICD *			pIcd,
		FLMUINT		uiKeyLen);
		
	RCODE buildKeys(
		ICD *			pIcd,
		FLMUINT		uiKeyLen);

	RCODE buildKeys(
		FLMUINT64	ui64DocumentID,
		IXD *			pIxd,
		CDL_HDR *	pCdlTbl,
		FLMBOOL		bUseSubtreeNodes,
		FLMBOOL		bAddKeys);

	RCODE genIndexKeys(
		FLMUINT64			ui64DocumentID,
		F_DOMNode *			pNode,
		IXD *					pIxd,
		ICD *					pIcd,
		IxAction				eAction);

	RCODE updateIndexKeys(
		FLMUINT				uiCollectionNum,
		F_DOMNode *			pNode,
		IxAction				eAction,
		FLMBOOL				bStartOfUpdate,
		FLMBOOL *			pbIsIndexed = NULL);

	RCODE attrIsInIndexDef(
		FLMUINT				uiAttrNameId,
		FLMBOOL *			pbIsInIndexDef);

	void indexingAfterCommit( void);

	void indexingAfterAbort( void);

	RCODE	addToStopList(
		FLMUINT				uiIndexNum);

	RCODE	addToStartList(
		FLMUINT				uiIndexNum);

	void stopBackgroundIndexThread(
		FLMUINT				uiIndexNum,
		FLMBOOL				bWait,
		FLMBOOL *			pbStopped);

	RCODE startIndexBuild(
		FLMUINT				uiIndexNum);

	RCODE checkDictDefInfo(
		FLMUINT64			ui64DocumentID,
		FLMBOOL				bDeleting,
		FLMUINT *			puiDictType,
		FLMUINT *			puiDictNumber);

	RCODE dictDocumentDone(
		FLMUINT64			ui64DocumentID,
		FLMBOOL				bDeleting,
		FLMUINT *			puiDictDefType);

	RCODE outputContextKeys(
		FLMUINT64			ui64DocumentId,
		IXD *					pIxd,
		IX_CONTEXT *		pIxContext,
		IX_CONTEXT **		ppIxContextList);

	RCODE removeCdls(
		FLMUINT64			ui64DocumentId,
		IXD *					pIxd,
		IX_CONTEXT *		pIxContext,
		ICD *					pRefIcd);

	RCODE indexDocument(
		IXD *					pIxd,
		F_DOMNode *			pDocNode);

	RCODE indexSetOfDocuments(
		FLMUINT					uiIndexNum,
		FLMUINT64				ui64StartDocumentId,
		FLMUINT64				ui64EndDocumentId,
		IF_IxStatus *			ifpIxStatus,
		IF_IxClient *			ifpIxClient,
		XFLM_INDEX_STATUS *	pIndexStatus,
		FLMBOOL *				pbHitEnd,
		F_Thread *				pThread = NULL);

	RCODE setIxStateInfo(
		FLMUINT				uiIndexNum,
		FLMUINT64			ui64LastDocumentIndexed,
		FLMUINT				uiState);

	RCODE buildIndex(
		FLMUINT				uiIndexNum,
		FLMUINT				uiState);

	RCODE readBlkHdr(
		FLMUINT				uiBlkAddress,
		F_BLK_HDR *			pBlkHdr,
		FLMINT *				piType);

	XFLM_LFILE_STATS * getLFileStatPtr(
		LFILE *				pLFile);

	RCODE checkAndUpdateState(
		eDomNodeType		eNodeType,
		FLMUINT				uiNameId);

#define FLM_UPD_ADD					0x00001
#define FLM_UPD_INTERNAL_CHANGE	0x00004

	RCODE findNode(
		FLMUINT				uiCollection,
		FLMUINT64 *			pui64NodeId,
		FLMUINT				uiFlags);
		
	RCODE getNode(
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId,
		FLMUINT				uiFlags,
		F_DOMNode **		ppNode);

	RCODE _updateNode(
		F_CachedNode *		pNode,
		FLMUINT				uiFlags);

	FINLINE RCODE updateNode(
		F_CachedNode *		pCachedNode,
		FLMUINT				uiFlags)
	{
		flmAssert( !m_pDatabase->m_pRfl->isLoggingEnabled());

		if( uiFlags || pCachedNode->getCollection() == XFLM_DICT_COLLECTION)
		{
			return( _updateNode( pCachedNode, uiFlags));
		}

		if( !pCachedNode->nodeIsDirty())
		{
			pCachedNode->setNodeDirty( this, FALSE);
		}

		return( NE_XFLM_OK);
	}

	RCODE getCachedBTree(
		FLMUINT				uiCollection,
		F_Btree **			ppBTree);

	RCODE flushNode(
		F_Btree *			pBTree,
		F_CachedNode *		pNode);

	RCODE purgeNode(
		FLMUINT				uiCollection,
		FLMUINT64			ui64NodeId);

	RCODE allocNode(
		eDomNodeType		eNodeType,
		F_DOMNode **		ppNode);

	RCODE createRootNode(
		FLMUINT				uiCollection,
		FLMUINT				uiElementNameId,
		eDomNodeType		eNodeType,
		F_DOMNode **		ppNewNode,
		FLMUINT64 *			pui64NodeId = NULL);

	RCODE sweep(
		F_Thread *			pThread);
	
	RCODE sweepGatherList(
		ELM_ATTR_STATE_INFO **	ppStateTbl,
		FLMUINT *					puiNumItems);

	RCODE sweepCheckElementState(
		F_DOMNode *					pElementNode,
		ELM_ATTR_STATE_INFO *	pStateTbl,
		FLMUINT *					puiNumItems,
		FLMBOOL *					pbStartedTrans);

	RCODE sweepCheckAttributeStates(
		F_DOMNode *					pElementNode,
		ELM_ATTR_STATE_INFO *	pStateTbl,
		FLMUINT *					puiNumItems,
		FLMBOOL *					pbStartedTrans);

	RCODE sweepFinalizeStates(
		ELM_ATTR_STATE_INFO *	pStateTbl,
		FLMUINT						uiNumItems,
		FLMBOOL *					pbStartedTrans);

	RCODE flushDirtyNodes( void);

	RCODE flushDirtyNode(
		F_CachedNode *				pNode);

	RCODE maintBlockChainFree(
		FLMUINT64			ui64MaintDocID,
		FLMUINT 				uiBlocksToDelete,
		FLMUINT				uiExpectedEndAddr,
		FLMUINT *			puiBlocksFreed);

	RCODE encryptData(
		FLMUINT				uiEncDefId,
		FLMBYTE *			pucIV,
		FLMBYTE *			pucBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT				uiDataLen,
		FLMUINT *			puiEncryptedLength);
		
	RCODE decryptData(
		FLMUINT				uiEncDefId,
		FLMBYTE *			pucIV,
		void *				pvInBuf,
		FLMUINT				uiInLen,
		void *				pvOutBuf,
		FLMUINT				uiOutBufLen);
	
	RCODE createDbKey();

	// Private data members.

	F_Database *			m_pDatabase;		// Pointer to F_Database object
	F_Dict *					m_pDict;				// Pointer to dictionary object
	F_Db *					m_pNextForDatabase;	// Next F_Db associated with F_Database
														// NOTE: gv_XFlmSysData.hShareMutex
														// must be locked to set this
	F_Db *					m_pPrevForDatabase;	// Prev F_Db associated with F_Database
														// NOTE: gv_XFlmSysData.hShareMutex
														// must be locked to set this
	void *					m_pvAppData;		// Application data that is used
														// to associate this F_Db with
														// an object in the application
														// space.
	FLMUINT					m_uiThreadId;		// Thread that started the current
														// transaction, if any.  NOTE:
														// Only set on transaction begin.
														// Hence, if operations are performed
														// by multiple threads, within the
														// transaction, it will not necessarily
														// reflect the thread that is currently
														// using the F_Db.
	FLMBOOL					m_bMustClose;		// An error has occurred that requires
														// the application to stop using (close)
														// this FDB
	F_SuperFileHdl *		m_pSFileHdl;		// Pointer to the super file handle
	FLMUINT					m_uiFlags;			// Flags for this F_Db.

	// TRANSACTION STATE STUFF

	FLMUINT					m_uiTransCount;	// Transaction counter for the F_Db.
														// Incremented whenever a transaction
														// is started on this F_Db.  Used so
														// that FLAIM can tell if an implicit
														// transaction it started is still in
														// effect.  This should NOT be
														// confused with update transaction
														// IDs.
	eDbTransType			m_eTransType;		// Type of transaction
	RCODE						m_AbortRc;			// If not NE_XFLM_OK, transaction must be
														// aborted.
	FLMUINT64  				m_ui64CurrTransID;// Current transaction ID.
	FLMUINT					m_uiFirstAvailBlkAddr;	// Address of first block in avail list
	FLMUINT					m_uiLogicalEOF;			// Current logical end of file.  New
																// blocks are allocated at this address.
	FLMUINT					m_uiUpgradeCPFileNum;
	FLMUINT					m_uiUpgradeCPOffset;
														// RFL file number and offset to set
														// RFL to during an upgrade operation
														// that happens during a restore or
														// recovery.
	FLMUINT					m_uiTransEOF;			// Address of logical end of file
														// when the last transaction
														// committed. A block beyond this
														// point in the file is going to be
														// a new block and will not need to
														// be logged.
	KEY_GEN_INFO			m_keyGenInfo;		// Information for generating index
														// keys.
	F_TMSTAMP				m_TransStartTime;	// Transaction start time, for stats

	// KREF STUFF
														
	KREF_ENTRY **			m_pKrefTbl;			// Pointer to KREF table, which is an array
														// of KREF_ENTRY * pointers.
	FLMUINT					m_uiKrefTblSize;	// KREF table size.
	FLMUINT					m_uiKrefCount;		// Number of entries in KREF table that
														// are currently used.
	FLMUINT					m_uiTotalKrefBytes;	// Total number of entries allocated
															// in the pool.
	FLMBYTE *				m_pucKrefKeyBuf;	// Pointer to temporary key buffer.
	FLMBOOL					m_bKrefSetup;		// True if the KRef table has been initialized.
	F_Pool *					m_pKrefPool;		// Memory pool to use
	FLMBOOL					m_bReuseKrefPool;	// Reuse pool instead of free it?
	FLMBOOL					m_bKrefCompoundKey;	// True if a compound key has been processed.
	void *					m_pKrefReset;			// Used to reset the Kref pool on
														// indexing failures
	F_Pool					m_tmpKrefPool;		// KREF pool to be used during
														// read transactions - only used when
														// checking indexes.

	// UPDATE TRANSACTION STUFF

	FLMBOOL					m_bHadUpdOper;		// Did this transaction have any
														// updates?
	FLMUINT					m_uiBlkChangeCnt;	// Number of times ScaLogPhysBlk has
														// been called during this transaction.
														// This is used by the cursor code to
														// know when it is necessary to
														// re-position in the B-Tree.0
	IXD_FIXUP *				m_pIxdFixups;		// List of indexes whose IXD needs
														// to be restored to its prior
														// state if the transaction aborts
	// READ TRANSACTION STUFF

	F_Db *					m_pNextReadTrans;	// Next active read transaction for
														// this database.
														// NOTE: If uiKilledTime (see below)
														// is non-zero, then transaction is
														// in killed list.
	F_Db *					m_pPrevReadTrans;	// Previous active read transaction
														// for this database.
														// NOTE: If m_uiKilledTime (see below)
														// is non-zero, then transaction is
														// in killed list.
	FLMUINT					m_uiInactiveTime;	// If non-zero, this is the last time
														// the checkpoint thread marked this
														// transaction as inactive.  If zero,
														// it means that the transaction is
														// active, or it has not been marked
														// by the checkpoint thread as
														// inactive.  If it stays non-zero for
														// five or more minutes, it will be
														// killed.
	FLMUINT					m_uiKilledTime;	// Time transaction was killed, if
														// non-zero.
	// Misc. DB Info.

	FLMBOOL					m_bItemStateUpdOk;//	This variable is used to ensure
														// that FlmDbSweep / recovery are the
														// only ways that:
														// 1) an element or attribute's state
														//		can be changed to 'unused'
														// 2) a 'purge' element or attribute
														//		can be deleted

	F_Pool					m_TempPool;			// Temporary memory pool.  It
														// is only used for the duration of
														// a FLAIM operation and then reset.
														// The first block in the pool is
														// retained between operations to
														// help performance.

	// Callback functions.
	IF_DeleteStatus *		m_pDeleteStatus;	// Handles status info coming back
															// from deleting a BTree
	IF_IxClient *			m_pIxClient;		// Indexing callback
	IF_IxStatus *			m_pIxStatus;		// Indexing status callback
	IF_CommitClient *		m_pCommitClient;	// Commit callback

	XFLM_STATS *			m_pStats;
	XFLM_DB_STATS *		m_pDbStats;			// DB statistics pointer.
	XFLM_LFILE_STATS *	m_pLFileStats;		// LFILE statistics pointer.
	FLMUINT					m_uiLFileAllocSeq;// Allocation sequence number for
														// LFILE statistics array so we
														// can tell if the array has been
														// reallocated and we need to reset
														// our pLFileStats pointer.
	XFLM_STATS				m_Stats;				// Statistics kept here until end
														// of transaction.
	FLMBOOL					m_bStatsInitialized;// Has statistics structure been
														// initialized?
	F_BKGND_IX *			m_pIxStartList;	// Indexing threads to start at 
														// the conclusion of the transaction.
	F_BKGND_IX *			m_pIxStopList;		// Indexing threads to stop at 
														// the conclusion of the transaction.
	F_Btree *				m_pCachedBTree;	// BTree object used for node operations
	F_KeyCollector *		m_pKeyColl;			// Special purpose object used when checking
														// indexes in the F_DbCheck class.
	F_OldNodeList *		m_pOldNodeList;	// List of old truncated nodes to use
														// updating indexes.
	FLMUINT					m_uiDirtyNodeCount;
	F_SEM						m_hWaitSem;			// Semaphore that is used when
														// waiting for reads to complete

friend class F_Database;
friend class F_Dict;
friend class F_DbSystem;
friend class F_Rfl;
friend class F_Btree;
friend class F_Backup;
friend class F_DOMNode;
friend class F_BTreeIStream;
friend class F_DataVector;
friend class F_NodeList;
friend class F_XMLImport;
friend class F_DbRebuild;
friend class F_DbCheck;
friend class F_Query;
friend class FSIndexCursor;
friend class FSCollectionCursor;
friend class F_BtRSFactory;
friend class F_BtResultSet;
friend class F_CachedBlock;
friend class F_CachedNode;
friend class F_BlockCacheMgr;
friend class F_NodeCacheMgr;
friend class F_GlobalCacheMgr;
friend class F_QueryResultSet;
friend class ServerLockObject;
friend class F_BTreeInfo;
friend class F_AttrItem;
};

/****************************************************************************
Stuff for F_Query class
****************************************************************************/

#define FLM_FALSE			1
#define FLM_TRUE			2
#define FLM_UNK			4

FINLINE FLMBOOL isLegalOperator(
	eQueryOperators	eOperator)
{
	return( (eOperator >= XFLM_AND_OP && eOperator <= XFLM_RBRACKET_OP)
			  ? TRUE
			  : FALSE);
}

FINLINE FLMBOOL isLogicalOp(
	eQueryOperators	eOperator)
{
	return( (eOperator >= XFLM_AND_OP && eOperator <= XFLM_NOT_OP) ? TRUE : FALSE);
}

FINLINE FLMBOOL isCompareOp(
	eQueryOperators	eOperator)
{
	return( (eOperator >= XFLM_EQ_OP && eOperator <= XFLM_GE_OP) ? TRUE : FALSE);
}

FINLINE FLMBOOL isArithOp(
	eQueryOperators	eOperator)
{
	return( (eOperator >= XFLM_FIRST_ARITH_OP && 
		eOperator <= XFLM_LAST_ARITH_OP) ? TRUE : FALSE);
}

FINLINE FLMBOOL isUnsigned(
	eValTypes	eValType)
{
	return( eValType == XFLM_UINT_VAL || eValType == XFLM_UINT64_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL isSigned(
	eValTypes	eValType)
{
	return( eValType == XFLM_INT_VAL || eValType == XFLM_INT64_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL is64BitVal(
	eValTypes	eValType)
{
	return( eValType == XFLM_UINT64_VAL || eValType == XFLM_INT64_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL isNativeNum(
	eValTypes	eValType)
{
	return( eValType == XFLM_UINT_VAL || eValType == XFLM_INT_VAL
			 ? TRUE
			 : FALSE);
}

FINLINE FLMBOOL isUniversal(
	FQNODE *	pQNode)
{
	return( pQNode->bNotted);
}

FINLINE FLMBOOL isExistential(
	FQNODE *	pQNode)
{
	return( !pQNode->bNotted);
}

FINLINE FLMBOOL isBoolNode(
	FQNODE *	pQNode
	)
{
	return( (pQNode->eNodeType == FLM_VALUE_NODE &&
				pQNode->currVal.eValType == XFLM_BOOL_VAL) ? TRUE : FALSE);
}

FINLINE FLMBOOL isSigned(
	FQVALUE *		pValue)
{
	if( pValue->eValType == XFLM_INT_VAL || pValue->eValType == XFLM_INT64_VAL)
	{
		return( TRUE);
	}

	return( FALSE);
}

FINLINE FLMBOOL isUnsigned(
	FQVALUE *		pValue)
{
	if( pValue->eValType == XFLM_UINT_VAL || pValue->eValType == XFLM_UINT64_VAL)
	{
		return( TRUE);
	}

	return( FALSE);
}

typedef struct ExprState *	EXPR_STATE_p;

typedef struct ExprState
{
	FQNODE *				pExpr;
	FQNODE *				pCurOperatorNode;
	FQNODE *				pLastNode;
	FLMUINT				uiNestLevel;
	FLMBOOL				bExpectingOperator;
	FLMBOOL				bExpectingLParen;
	FQFUNCTION *		pQFunction;
	XPATH_COMPONENT *	pXPathComponent;
	FLMUINT				uiNumExprNeeded;
	FLMUINT				uiNumExpressions;
	EXPR_STATE_p		pPrev;
	EXPR_STATE_p		pNext;
} EXPR_STATE;

/*****************************************************************************
Desc:	Object for using a buffer on the stack until we outgrow it.
*****************************************************************************/
class F_DynaBuf : public IF_DynaBuf
{
public:

	F_DynaBuf(
		FLMBYTE *		pucBuffer,
		FLMUINT			uiBufferSize)
	{
		m_pucBuffer = pucBuffer;
		m_uiBufferSize = uiBufferSize;
		m_uiOffset = 0;
		m_bAllocatedBuffer = FALSE;
	}
	
	virtual ~F_DynaBuf()
	{
		if( m_bAllocatedBuffer)
		{
			f_free( &m_pucBuffer);
		}
	}
	
	FINLINE void truncateData(
		FLMUINT			uiSize)
	{
		if( uiSize < m_uiOffset)
		{
			m_uiOffset = uiSize;
		}
	}
	
	FINLINE RCODE allocSpace(
		FLMUINT		uiSize,
		void **		ppvPtr)
	{
		RCODE		rc = NE_XFLM_OK;
		
		if( m_uiOffset + uiSize >= m_uiBufferSize)
		{
			if( RC_BAD( rc = resizeBuffer( m_uiOffset + uiSize + 512)))
			{
				goto Exit;
			}
		}
		
		*ppvPtr = &m_pucBuffer[ m_uiOffset];
		m_uiOffset += uiSize;
		
	Exit:
	
		return( rc);
	}
	
	FINLINE RCODE appendData(
		const void *		pvData,
		FLMUINT				uiSize)
	{
		RCODE		rc = NE_XFLM_OK;
		void *	pvTmp;
		
		if( RC_BAD( rc = allocSpace( uiSize, &pvTmp)))
		{
			goto Exit;
		}

		if( uiSize == 1)
		{
			*((FLMBYTE *)pvTmp) = *((FLMBYTE *)pvData);
		}
		else
		{
			f_memcpy( pvTmp, pvData, uiSize);
		}
		
	Exit:
	
		return( rc);
	}
		
	FINLINE RCODE appendByte(
		FLMBYTE		ucChar)
	{
		RCODE			rc = NE_XFLM_OK;
		FLMBYTE *	pucTmp;
		
		if( RC_BAD( rc = allocSpace( 1, (void **)&pucTmp)))
		{
			goto Exit;
		}
		
		*pucTmp = ucChar;
		
	Exit:
	
		return( rc);
	}
	
	FINLINE RCODE appendUniChar(
		FLMUNICODE	uChar)
	{
		RCODE				rc = NE_XFLM_OK;
		FLMUNICODE *	puTmp;
		
		if( RC_BAD( rc = allocSpace( sizeof( FLMUNICODE), (void **)&puTmp)))
		{
			goto Exit;
		}
		
		*puTmp = uChar;
		
	Exit:
	
		return( rc);
	}
	
	FINLINE FLMBYTE * getBufferPtr( void)
	{
		return( m_pucBuffer);
	}
	
	FINLINE FLMUNICODE * getUnicodePtr( void)
	{
		if( m_uiOffset >= sizeof( FLMUNICODE))
		{
			return( (FLMUNICODE *)m_pucBuffer);
		}
		
		return( NULL);
	}
	
	FINLINE FLMUINT getUnicodeLength( void)
	{
		if( m_uiOffset <= sizeof( FLMUNICODE))
		{
			return( 0);
		}
		
		return( (m_uiOffset >> 1) - 1);
	}
	
	FINLINE FLMUINT getDataLength( void)
	{
		return( m_uiOffset);
	}
	
	FINLINE RCODE copyFromBuffer(
		F_DynaBuf *		pSource)
	{
		RCODE		rc = NE_XFLM_OK;
		
		if( RC_BAD( rc = resizeBuffer( pSource->m_uiBufferSize)))
		{
			goto Exit;
		}
		
		if( (m_uiOffset = pSource->m_uiOffset) != 0)
		{
			f_memcpy( m_pucBuffer, pSource->m_pucBuffer, pSource->m_uiOffset);
		}
		
	Exit:
		
		return( rc);
	}		
	
private:

	RCODE resizeBuffer(
		FLMUINT		uiNewSize)
	{
		RCODE	rc = NE_XFLM_OK;
		
		if( !m_bAllocatedBuffer)
		{
			if( uiNewSize > m_uiBufferSize)
			{
				FLMBYTE *		pucOriginalBuf = m_pucBuffer;
				
				if( RC_BAD( rc = f_alloc( uiNewSize, &m_pucBuffer)))
				{
					m_pucBuffer = pucOriginalBuf;
					goto Exit;
				}
				
				m_bAllocatedBuffer = TRUE;
				
				if( m_uiOffset)
				{
					f_memcpy( m_pucBuffer, pucOriginalBuf, m_uiOffset);
				}
			}
		}
		else
		{
			if( RC_BAD( rc = f_realloc( uiNewSize, &m_pucBuffer)))
			{
				goto Exit;
			}
			
			if( uiNewSize < m_uiOffset)
			{
				m_uiOffset = uiNewSize;
			}
		}
		
		m_uiBufferSize = uiNewSize;
		
	Exit:
	
		return( rc);
	}

	FLMBOOL		m_bAllocatedBuffer;
	FLMBYTE *	m_pucBuffer;
	FLMUINT		m_uiBufferSize;
	FLMUINT		m_uiOffset;
};

/*****************************************************************************
Desc:	Object for gathering node information.
*****************************************************************************/
class F_NodeInfo : public IF_NodeInfo, public XF_Base
{
public:

	F_NodeInfo()
	{
		clearNodeInfo();
	}
	
	virtual ~F_NodeInfo()
	{
	}
	
	FINLINE void XFLMAPI clearNodeInfo( void)
	{
		f_memset( &m_nodeInfo, 0, sizeof( m_nodeInfo));
		m_ui64TotalNodes = 0;
	}
	
	RCODE XFLMAPI addNodeInfo(
		IF_Db *			pDb,
		IF_DOMNode *	pNode,
		FLMBOOL			bDoSubTree,
		FLMBOOL			bDoSelf = TRUE);
		
	FINLINE FLMUINT64 XFLMAPI getTotalNodeCount( void)
	{
		return( m_ui64TotalNodes);
	}
	
	FINLINE void XFLMAPI getNodeInfo(
		XFLM_NODE_INFO *	pNodeInfo)
	{
		f_memcpy( pNodeInfo, &m_nodeInfo, sizeof( m_nodeInfo));
	}
	
private:

	XFLM_NODE_INFO	m_nodeInfo;
	FLMUINT64		m_ui64TotalNodes;
};

typedef struct BTREE_INFO
{
	FLMUINT						uiLfNum;
	char *						pszLfName;
	FLMUINT						uiNumLevels;
	XFLM_BTREE_LEVEL_INFO	levelInfo [MAX_LEVELS];
} BTREE_INFO;

/*****************************************************************************
Desc:	Object for gathering B-Tree information.
*****************************************************************************/
class F_BTreeInfo : public IF_BTreeInfo, public XF_Base
{
public:
	F_BTreeInfo()
	{
		m_pIndexArray = NULL;
		m_uiIndexArraySize = 0;
		m_uiNumIndexes = 0;
		m_pCollectionArray = NULL;
		m_uiCollectionArraySize = 0;
		m_uiNumCollections = 0;
		m_pool.poolInit( 512);
	}
	
	virtual ~F_BTreeInfo()
	{
		if (m_pIndexArray)
		{
			f_free( &m_pIndexArray);
		}
		if (m_pCollectionArray)
		{
			f_free( &m_pCollectionArray);
		}
		m_pool.poolFree();
	}
	
	FINLINE void XFLMAPI clearBTreeInfo( void)
	{
		m_uiNumIndexes = 0;
		m_uiNumCollections = 0;
	}
	
	RCODE XFLMAPI collectIndexInfo(
		IF_Db *					pDb,
		FLMUINT					uiIndexNum,
		IF_BTreeInfoStatus *	pInfoStatus);
		
	RCODE XFLMAPI collectCollectionInfo(
		IF_Db *					pDb,
		FLMUINT					uiCollectionNum,
		IF_BTreeInfoStatus *	pInfoStatus);
			
	FINLINE FLMUINT XFLMAPI getNumIndexes( void)
	{
		return( m_uiNumIndexes);
	}
		
	FINLINE FLMUINT XFLMAPI getNumCollections( void)
	{
		return( m_uiNumCollections);
	}
		
	FINLINE FLMBOOL XFLMAPI getIndexInfo(
		FLMUINT		uiNthIndex,
		FLMUINT *	puiIndexNum,
		char **		ppszIndexName,
		FLMUINT *	puiNumLevels)
	{
		if (uiNthIndex < m_uiNumIndexes)
		{
			*puiIndexNum = m_pIndexArray [uiNthIndex].uiLfNum;
			*puiNumLevels = m_pIndexArray [uiNthIndex].uiNumLevels;
			*ppszIndexName = m_pIndexArray [uiNthIndex].pszLfName;
			return( TRUE);
		}
		else
		{
			*puiIndexNum = 0;
			*ppszIndexName = NULL;
			*puiNumLevels = 0;
			return( FALSE);
		}
	}
		
	FINLINE FLMBOOL XFLMAPI getCollectionInfo(
		FLMUINT		uiNthCollection,
		FLMUINT *	puiCollectionNum,
		char **		ppszCollectionName,
		FLMUINT *	puiNumLevels)
	{
		if (uiNthCollection < m_uiNumCollections)
		{
			*puiCollectionNum = m_pCollectionArray [uiNthCollection].uiLfNum;
			*puiNumLevels = m_pCollectionArray [uiNthCollection].uiNumLevels;
			*ppszCollectionName = m_pCollectionArray [uiNthCollection].pszLfName;
			return( TRUE);
		}
		else
		{
			*puiCollectionNum = 0;
			*puiNumLevels = 0;
			*ppszCollectionName = NULL;
			return( FALSE);
		}
	}
		
	FINLINE FLMBOOL XFLMAPI getIndexLevelInfo(
		FLMUINT						uiNthIndex,
		FLMUINT						uiBTreeLevel,
		XFLM_BTREE_LEVEL_INFO *	pLevelInfo)
	{
		if (uiNthIndex < m_uiNumIndexes &&
			 uiBTreeLevel < m_pIndexArray [uiNthIndex].uiNumLevels)
		{
			f_memcpy( pLevelInfo,
				&(m_pIndexArray [uiNthIndex].levelInfo [uiBTreeLevel]),
				sizeof( XFLM_BTREE_LEVEL_INFO));
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}

	FINLINE FLMBOOL XFLMAPI getCollectionLevelInfo(
		FLMUINT						uiNthCollection,
		FLMUINT						uiBTreeLevel,
		XFLM_BTREE_LEVEL_INFO *	pLevelInfo)
	{
		if (uiNthCollection < m_uiNumCollections &&
			 uiBTreeLevel < m_pCollectionArray [uiNthCollection].uiNumLevels)
		{
			f_memcpy( pLevelInfo,
				&(m_pCollectionArray [uiNthCollection].levelInfo [uiBTreeLevel]),
				sizeof( XFLM_BTREE_LEVEL_INFO));
			return( TRUE);
		}
		else
		{
			return( FALSE);
		}
	}

private:

	RCODE collectBlockInfo(
		F_Db *					pDb,
		LFILE *					pLFile,
		BTREE_INFO *			pBTreeInfo,
		F_BTREE_BLK_HDR *		pBlkHdr,
		IXD *						pIxd);
		
	RCODE collectBTreeInfo(
		F_Db *					pDb,
		LFILE *					pLFile,
		BTREE_INFO *			pBTreeInfo,
		IXD *						pIxd);

	FINLINE RCODE doCallback( void)
	{
		if (m_pInfoStatus)
		{
			return( m_pInfoStatus->infoStatus( m_uiCurrLfNum, m_bIsCollection,
						m_pszCurrLfName, m_uiCurrLevel,
						m_ui64CurrLfBlockCount, m_ui64CurrLevelBlockCount,
						m_ui64TotalBlockCount));
		}
		else
		{
			return( NE_XFLM_OK);
		}
	}
	
	BTREE_INFO *			m_pIndexArray;
	FLMUINT					m_uiIndexArraySize;
	FLMUINT					m_uiNumIndexes;
	BTREE_INFO *			m_pCollectionArray;
	FLMUINT					m_uiCollectionArraySize;
	FLMUINT					m_uiNumCollections;
	F_Pool					m_pool;
	
	// Items for the callback function.
	
	IF_BTreeInfoStatus *	m_pInfoStatus;
	FLMUINT					m_uiBlockSize;
	FLMUINT					m_uiCurrLfNum;
	FLMBOOL					m_bIsCollection;
	char *					m_pszCurrLfName;
	FLMUINT					m_uiCurrLevel;
	FLMUINT64				m_ui64CurrLfBlockCount;
	FLMUINT64				m_ui64CurrLevelBlockCount;
	FLMUINT64				m_ui64TotalBlockCount;
};

RCODE ixKeyCompare(
	F_Db *				pDb,
	IXD *					pIxd,
	F_DataVector *		pSearchKey,
	F_OldNodeList *	pOldNodeList1,
	F_OldNodeList *	pOldNodeList2,
	FLMBOOL				bCompareDocId,
	FLMBOOL				bCompareNodeIds,
	const void *		pvKey1,
	FLMUINT				uiKeyLen1,
	const void *		pvKey2,
	FLMUINT				uiKeyLen2,
	FLMINT *				piCompare);
	
/********************************************************************
Desc:	Class for comparing two keys in an index.
********************************************************************/
class IXKeyCompare : public IF_ResultSetCompare, public XF_Base
{
public:

	IXKeyCompare()
	{
		
		// m_pDb is used to sort truncated keys if necessary.
		// m_pIxd is used for comparison
		
		m_pDb = NULL;
		m_pIxd = NULL;
		m_pSearchKey = NULL;
		m_pOldNodeList = NULL;
		m_bCompareDocId = TRUE;
		m_bCompareNodeIds = TRUE;
	}

	virtual ~IXKeyCompare()
	{
		if (m_pOldNodeList)
		{
			m_pOldNodeList->Release();
		}
	}

	FINLINE RCODE XFLMAPI compare(
		const void *	pvKey1,
		FLMUINT			uiKeyLen1,
		const void *	pvKey2,
		FLMUINT			uiKeyLen2,
		FLMINT *			piCompare)
	{
		return( ixKeyCompare( m_pDb, m_pIxd, m_pSearchKey, m_pOldNodeList,
						m_pOldNodeList,
						m_bCompareDocId, m_bCompareNodeIds,
						pvKey1, uiKeyLen1, pvKey2, uiKeyLen2, piCompare));
	}
	
	FINLINE void setOldNodeList(
		F_OldNodeList *	pOldNodeList)
	{
		flmAssert( !m_pOldNodeList);
		if ((m_pOldNodeList = pOldNodeList) != NULL)
		{
			m_pOldNodeList->AddRef();
		}
	}
		
	FINLINE void setIxInfo(
		F_Db *	pDb,
		IXD *		pIxd)
	{
		m_pDb = pDb;
		m_pIxd = pIxd;
	}
	
	FINLINE void setSearchKey(
		F_DataVector *	pSearchKey)
	{
		m_pSearchKey = pSearchKey;
	}

	FINLINE void setCompareNodeIds(
		FLMBOOL	bCompareNodeIds)
	{
		m_bCompareNodeIds = bCompareNodeIds;
	}

	FINLINE void setCompareDocId(
		FLMBOOL	bCompareDocId)
	{
		m_bCompareDocId = bCompareDocId;
	}

	FINLINE FLMUINT getRefCount( void)
	{
		return( IF_ResultSetCompare::getRefCount());
	}

	virtual FINLINE FLMINT XFLMAPI AddRef( void)
	{
		return( IF_ResultSetCompare::AddRef());
	}

	virtual FINLINE FLMINT XFLMAPI Release( void)
	{
		return( IF_ResultSetCompare::Release());
	}
	
private:

	F_Db *				m_pDb;
	IXD *					m_pIxd;
	F_DataVector *		m_pSearchKey;
	F_OldNodeList *	m_pOldNodeList;
	FLMBOOL				m_bCompareDocId;
	FLMBOOL				m_bCompareNodeIds;
};

/*=============================================================================
Desc:	Result set class for queries that do sorting.
=============================================================================*/
class F_QueryResultSet : public XF_RefCount, public XF_Base
{
public:

	F_QueryResultSet()
	{
		m_pBTree = NULL;
		m_pResultSetDb = NULL;
		m_pSrcDb = NULL;
		m_pIxd = NULL;
		m_uiCurrPos = FLM_MAX_UINT;
		m_uiCount = 0;
		m_bPositioned = FALSE;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_QueryResultSet();
	
	// Initialize the result set
	
	RCODE initResultSet(
		FLMBOOL	bUseIxCompareObj,
		FLMBOOL	bEnableEncryption);
	
	FINLINE void setIxInfo(
		F_Db *	pSrcDb,
		IXD *		pIxd)
	{
		m_pSrcDb = pSrcDb;
		m_pIxd = pIxd;
		m_compareObj.setIxInfo( pSrcDb, pIxd);
	}

	// Entry Add and Sort Methods

	RCODE addEntry(						// Variable or fixed length entry coming in
		FLMBYTE *	pucKey,				// key for sorting.
		FLMUINT		uiKeyLength,
		FLMBOOL		bLockMutex);

	// Methods to read entries.

	RCODE getFirst(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getLast(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getNext(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getPrev(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE getCurrent(
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	RCODE positionToEntry(
		FLMBYTE *		pucKey,
		FLMUINT			uiKeyBufSize,
		FLMUINT *		puiKeyLen,
		F_DataVector *	pSearchKey,
		FLMUINT			uiFlags,
		FLMBOOL			bLockMutex);

	RCODE positionToEntry(
		FLMUINT		uiPosition,
		FLMBYTE *	pucKey,
		FLMUINT		uiKeyBufSize,
		FLMUINT *	puiKeyLen,
		FLMBOOL		bLockMutex);

	FINLINE FLMUINT getCount( void)
	{
		return( m_uiCount);
	}
	
	FINLINE FLMUINT getCurrPos( void)
	{
		return( m_uiCurrPos);
	}
	
	FINLINE void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
		
	FINLINE void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}
		
private:

	char				m_szResultSetDibName [F_PATH_MAX_SIZE];
	F_Db *			m_pResultSetDb;
	F_Btree *		m_pBTree;
	LFILE				m_LFile;
	F_Db *			m_pSrcDb;
	IXD *				m_pIxd;
	IXKeyCompare	m_compareObj;
	FLMUINT			m_uiCurrPos;
	FLMUINT			m_uiCount;
	FLMBOOL			m_bPositioned;
	F_MUTEX			m_hMutex;
};

typedef struct RS_WAITER
{
	FLMUINT					uiThreadId;		// Thread of waiter
	F_SEM						hESem;			// Semaphore to signal to wake up thread.
	RCODE *					pRc;				// Pointer to return code that is to
													// be set.
	FLMUINT					uiWaitStartTime;
													// Time we started waiting.
	FLMUINT					uiTimeLimit;	// Maximum time (milliseconds) to wait
													// before timing out.
	FLMUINT					uiNumToWaitFor;// Wait until we get at least this many
													// in the result set - or until the
													// result set is complete.
	RS_WAITER *				pNext;			// Next lock waiter in list.
	RS_WAITER *				pPrev;			// Previous lock waiter in list.
} RS_WAITER;

/****************************************************************************
Desc:	Class for setting up query criteria
****************************************************************************/
class F_Query : public IF_Query, public XF_Base
{
public:

	// Constructor and Destructor

	F_Query();
	virtual ~F_Query();

	// Methods for constructing a query

	FINLINE RCODE XFLMAPI setLanguage(
		FLMUINT	uiLanguage)
	{

		// Cannot change language after optimization

		if (m_bOptimized)
		{
			return( RC_SET( NE_XFLM_Q_ALREADY_OPTIMIZED));
		}
		m_uiLanguage = uiLanguage;
		return( NE_XFLM_OK);
	}

	FINLINE RCODE XFLMAPI setCollection(
		FLMUINT	uiCollection
		)
	{

		// Cannot change collection after optimization

		if (m_bOptimized)
		{
			return( RC_SET( NE_XFLM_Q_ALREADY_OPTIMIZED));
		}
		m_uiCollection = uiCollection;
		return( NE_XFLM_OK);
	}

	FINLINE RCODE XFLMAPI setupQueryExpr(
		IF_Db *					pDb,
		const FLMUNICODE *	puzQuery)
	{
		return( setupQueryExpr( TRUE, pDb, (void *)puzQuery));
	}
	
	FINLINE RCODE XFLMAPI setupQueryExpr(
		IF_Db *				pDb,
		const char *		pszQuery)
	{
		return( setupQueryExpr( FALSE, pDb, (void *)pszQuery));
	}

	RCODE XFLMAPI copyCriteria(
		IF_Query *	pSrcQuery);

	RCODE XFLMAPI addXPathComponent(
		eXPathAxisTypes		eXPathAxis,
		eDomNodeType			eNodeType,
		FLMUINT					uiNameId,
		IF_QueryNodeSource *	pNodeSource);

	RCODE XFLMAPI addOperator(
		eQueryOperators		eOperator,
		FLMUINT					uiCompareRules = 0,
		IF_OperandComparer *	pOpComparer = NULL);

	RCODE XFLMAPI addUnicodeValue(
		const FLMUNICODE *	puzVal);

	RCODE XFLMAPI addUTF8Value(
		const char *			pszVal,
		FLMUINT					uiUTF8Len = 0);

	RCODE XFLMAPI addBinaryValue(
		const void *			pvVal,
		FLMUINT					uiValLen);

	RCODE XFLMAPI addUINTValue(
		FLMUINT					uiVal);

	RCODE XFLMAPI addINTValue(
		FLMINT					iVal);

	RCODE XFLMAPI addUINT64Value(
		FLMUINT64				ui64Val);

	RCODE XFLMAPI addINT64Value(
		FLMINT64					i64Val);

	RCODE XFLMAPI addBoolean(
		FLMBOOL					bVal,
		FLMBOOL					bUnknown = FALSE);

	FINLINE RCODE XFLMAPI addFunction(
		eQueryFunctions		eFunction)
	{
		return( addFunction( eFunction, NULL, FALSE));
	}

	FINLINE RCODE XFLMAPI addFunction(
		IF_QueryValFunc *		pFuncObj,
		FLMBOOL					bHasXPathExpr)
	{
		// Pass XFLM_FUNC_xxx to private addFunction method - it really
		// doesn't matter, because it will be ignored.
		
		flmAssert( pFuncObj);
		return( addFunction( XFLM_FUNC_xxx, pFuncObj, bHasXPathExpr));
	}

	RCODE XFLMAPI getFirst(
		IF_Db *					pDb,
		IF_DOMNode **			ppNode,
		FLMUINT					uiTimeLimit = 0);

	RCODE XFLMAPI getLast(
		IF_Db *					pDb,
		IF_DOMNode **			ppNode,
		FLMUINT					uiTimeLimit = 0);

	RCODE XFLMAPI getNext(
		IF_Db *					pDb,
		IF_DOMNode **			ppNode,
		FLMUINT					uiTimeLimit = 0,
		FLMUINT					uiNumToSkip = 0,
		FLMUINT *				puiNumSkipped = NULL);

	RCODE XFLMAPI getPrev(
		IF_Db *					pDb,
		IF_DOMNode **			ppNode,
		FLMUINT					uiTimeLimit = 0,
		FLMUINT					uiNumToSkip = 0,
		FLMUINT *				puiNumSkipped = NULL);

	RCODE XFLMAPI getCurrent(
		IF_Db *				pDb,
		IF_DOMNode **		ppNode);

	void XFLMAPI resetQuery( void);

	RCODE XFLMAPI getStatsAndOptInfo(
		FLMUINT *			puiNumOptInfos,
		XFLM_OPT_INFO **	ppOptInfo);

	void XFLMAPI freeStatsAndOptInfo(
		XFLM_OPT_INFO **	ppOptInfo);

	void XFLMAPI setDupHandling(
		FLMBOOL	bRemoveDups);

	RCODE XFLMAPI setIndex(
		FLMUINT	uiIndex);

	RCODE XFLMAPI getIndex(
		IF_Db *		pDb,
		FLMUINT *	puiIndex,
		FLMBOOL *	pbHaveMultiple);

	RCODE XFLMAPI addSortKey(
		void *			pvSortKeyContext,
		FLMBOOL			bChildToContext,
		FLMBOOL			bElement,
		FLMUINT			uiNameId,
		FLMUINT			uiCompareRules,
		FLMUINT			uiLimit,
		FLMUINT			uiKeyComponent,
		FLMBOOL			bSortDescending,
		FLMBOOL			bSortMissingHigh,
		void **			ppvContext);

	FINLINE RCODE XFLMAPI enablePositioning( void)
	{
		if (m_bOptimized)
		{
			return( RC_SET( NE_XFLM_ILLEGAL_OP));
		}
		else
		{
			m_bPositioningEnabled = TRUE;
		}
		return( NE_XFLM_OK);
	}
	
	RCODE XFLMAPI positionTo(
		IF_Db *				pDb,
		IF_DOMNode **		ppNode,
		FLMUINT				uiTimeLimit,
		FLMUINT				uiPosition);
			
	RCODE XFLMAPI positionTo(
		IF_Db *				pDb,
		IF_DOMNode **		ppNode,
		FLMUINT				uiTimeLimit,
		IF_DataVector *	pSearchKey,
		FLMUINT				uiFlags);
		
	RCODE XFLMAPI getPosition(
		IF_Db *				pDb,
		FLMUINT *			puiPosition);
		
	RCODE XFLMAPI buildResultSet(
		IF_Db *				pDb,
		FLMUINT				uiTimeLimit);
		
	void XFLMAPI stopBuildingResultSet( void);
		
	RCODE XFLMAPI getCounts(
		IF_Db *				pDb,
		FLMUINT				uiTimeLimit,
		FLMBOOL				bPartialCountOk,
		FLMUINT *			puiReadCount,
		FLMUINT *			puiPassedCount,
		FLMUINT *			puiPositionableToCount,
		FLMBOOL *			pbDoneBuildingResultSet = NULL);
		
	FINLINE void XFLMAPI enableResultSetEncryption( void)
	{
		m_bEncryptResultSet = TRUE;
	}
	
	FINLINE void XFLMAPI setQueryStatusObject(
		IF_QueryStatus *		pQueryStatus)
	{
		if (m_pQueryStatus)
		{
			m_pQueryStatus->Release();
		}
		if ((m_pQueryStatus = pQueryStatus) != NULL)
		{
			m_pQueryStatus->AddRef();
		}
	}

	FINLINE void XFLMAPI setQueryValidatorObject(
		IF_QueryValidator *		pQueryValidator)
	{
		if (m_pQueryValidator)
		{
			m_pQueryValidator->Release();
		}
		if ((m_pQueryValidator = pQueryValidator) != NULL)
		{
			m_pQueryValidator->AddRef();
		}
	}
	
private:

	RCODE XFLMAPI addFunction(
		eQueryFunctions		eFunction,
		IF_QueryValFunc *		pFuncObj,
		FLMBOOL					bHasXPathExpr);

	FINLINE FLMBOOL timedOut( void)
	{
		if (m_uiTimeLimit)
		{
			FLMUINT	uiCurrTime;

			uiCurrTime = FLM_GET_TIMER();
			if (FLM_ELAPSED_TIME( uiCurrTime, m_uiStartTime) > m_uiTimeLimit)
			{
				return( TRUE);
			}
		}
		return( FALSE);
	}

	FINLINE RCODE queryStatus( void)
	{
		if (timedOut())
		{
			return( RC_SET( NE_XFLM_TIMEOUT));
		}
		if (m_uiBuildThreadId && m_bStopBuildingResultSet)
		{
			return( RC_SET( NE_XFLM_USER_ABORT));
		}
		return( (RCODE)(m_pQueryStatus
							 ? m_pQueryStatus->queryStatus( m_pCurrOpt)
							 : (RCODE)NE_XFLM_OK));
	}

	FINLINE RCODE newSource( void)
	{
		if (timedOut())
		{
			return( RC_SET( NE_XFLM_TIMEOUT));
		}
		if (m_uiBuildThreadId && m_bStopBuildingResultSet)
		{
			return( RC_SET( NE_XFLM_USER_ABORT));
		}
		return( (RCODE)(m_pQueryStatus
							 ? m_pQueryStatus->newSource( m_pCurrOpt)
							 : (RCODE)NE_XFLM_OK));
	}

	FINLINE RCODE incrNodesRead( void)
	{
		m_pCurrOpt->ui64NodesRead++;
		return( queryStatus());
	}

	FINLINE FLMBOOL expectingOperand( void)
	{
		return( !m_pCurExprState->bExpectingOperator);
	}

	FINLINE FLMBOOL expectingOperator( void)
	{
		return( m_pCurExprState->bExpectingOperator);
	}

	FINLINE FLMBOOL parsingFunction( void)
	{
		return( (FLMBOOL)(m_pCurExprState->pPrev &&
								m_pCurExprState->pQFunction
								? TRUE
								: FALSE));
	}

	FINLINE FLMBOOL parsingXPathExpr( void)
	{
		return( (FLMBOOL)(m_pCurExprState->pPrev &&
								m_pCurExprState->pXPathComponent &&
								!m_pCurExprState->pQFunction
								? TRUE
								: FALSE));
	}

	RCODE allocExprState( void);

	RCODE allocValueNode(
		FLMUINT		uiValLen,
		eValTypes	eValType,
		FQNODE **	ppQNode);

	RCODE intersectPredicates(
		CONTEXT_PATH *			pContextPath,
		FQNODE *					pXPathNode,
		eQueryOperators		eOperator,
		FLMUINT					uiCompareRules,
		IF_OperandComparer *	pOpComparer,
		FQNODE *					pContextNode,
		FLMBOOL					bNotted,
		FQVALUE *				pQValue,
		FLMBOOL *				pbClipContext);

	RCODE unionPredicates(
		CONTEXT_PATH *			pContextPath,
		FQNODE *					pXPathNode,
		eQueryOperators		eOperator,
		FLMUINT					uiCompareRules,
		IF_OperandComparer *	pOpComparer,
		FQNODE *					pContextNode,
		FLMBOOL					bNotted,
		FQVALUE *				pQValue);

	RCODE addPredicateToContext(
		OP_CONTEXT *			pContext,
		XPATH_COMPONENT *		pXPathComponent,
		XPATH_COMPONENT *		pXPathComp,
		eQueryOperators		eOperator,
		FLMUINT					uiCompareRules,
		IF_OperandComparer *	pOpComparer,
		FQNODE *					pContextNode,
		FLMBOOL					bNotted,
		FQVALUE *				pQValue,
		FLMBOOL *				pbClipContext,
		FQNODE **				ppQNode);

	RCODE createOpContext(
		OP_CONTEXT *	pParentContext,
		FLMBOOL			bIntersect,
		FQNODE *			pQRootNode);

	RCODE getPathPredicates(
		FQNODE *				pParentNode,
		FQNODE **			ppQNode,
		XPATH_COMPONENT *	pXPathContext);

	RCODE getPredicates(
		FQNODE **				ppExpr,
		FQNODE *					pStartNode,
		XPATH_COMPONENT *		pXPathComponent);

	RCODE optimizePredicate(
		XPATH_COMPONENT *	pXPathComponent,
		PATH_PRED *			pPred);

	RCODE optimizePath(
		CONTEXT_PATH *	pContextPath,
		PATH_PRED *		pSingleNodeIdPred,
		FLMBOOL			bIntersect);

	RCODE optimizeContext(
		OP_CONTEXT *	pContext,
		CONTEXT_PATH *	pSingleNodeIdPath,
		PATH_PRED *		pSingleNodeIdPred);

	RCODE setupIndexScan( void);

	RCODE checkSortIndex(
		FLMUINT			uiOptIndex);
		
	RCODE optimize( void);

	RCODE setupCurrPredicate(
		FLMBOOL			bForward);

	RCODE testPassed(
		IF_DOMNode **	ppNode,
		FLMBOOL *		pbPassed,
		FLMBOOL *		pbEliminatedDup);
		
	RCODE nextFromIndex(
		FLMBOOL			bEvalCurrDoc,
		FLMUINT			uiMaxToSkip,
		FLMUINT *		puiNumSkipped,
		IF_DOMNode **	ppNode);

	RCODE prevFromIndex(
		FLMBOOL			bEvalCurrDoc,
		FLMUINT			uiMaxToSkip,
		FLMUINT *		puiNumSkipped,
		IF_DOMNode **	ppNode);

	RCODE getDocFromIndexScan(
		FLMBOOL	bFirstLast,
		FLMBOOL	bForward);

	RCODE nextFromScan(
		FLMBOOL			bFirstDoc,
		FLMUINT			uiMaxToSkip,
		FLMUINT *		puiNumSkipped,
		IF_DOMNode **	ppNode);

	RCODE prevFromScan(
		FLMBOOL			bLastDoc,
		FLMUINT			uiMaxToSkip,
		FLMUINT *		puiNumSkipped,
		IF_DOMNode **	ppNode);

	void useLeafContext(
		FLMBOOL	bGetFirst);

	FLMBOOL useNextPredicate( void);

	FLMBOOL usePrevPredicate( void);

	RCODE getNodeSourceNode(
		FLMBOOL					bForward,
		IF_QueryNodeSource *	pNodeSource,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);
		
	RCODE getRootAxisNode(
		IF_DOMNode **	ppCurrNode);

	RCODE walkDocument(
		FLMBOOL					bForward,
		FLMBOOL					bWalkAttributes,
		FLMUINT					uiAttrNameId,
		IF_DOMNode **			ppCurrNode);

	RCODE getChildAxisNode(
		FLMBOOL					bForward,
		IF_DOMNode *			pContextNode,
		FLMUINT					uiChildNameId,
		IF_DOMNode **			ppCurrNode);

	RCODE getParentAxisNode(
		FLMBOOL					bForward,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);

	RCODE getAncestorAxisNode(
		FLMBOOL					bForward,
		FLMBOOL					bIncludeSelf,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);

	RCODE getDescendantAxisNode(
		FLMBOOL					bForward,
		FLMBOOL					bIncludeSelf,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);

	RCODE getSibAxisNode(
		FLMBOOL					bForward,
		FLMBOOL					bPrevSibAxis,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);

	RCODE getPrevOrAfterAxisNode(
		FLMBOOL					bForward,
		FLMBOOL					bPrevAxis,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);

	RCODE getAttrAxisNode(
		FLMBOOL					bForward,
		FLMBOOL					bAttrAxis,
		FLMUINT					uiAttrNameId,
		IF_DOMNode *			pContextNode,
		IF_DOMNode **			ppCurrNode);

	RCODE verifyOccurrence(
		FLMBOOL					bUseKeyNodes,
		XPATH_COMPONENT *		pXPathComponent,
		IF_DOMNode *			pCurrNode,
		FLMBOOL *				pbPassed);

	RCODE getXPathComponentFromAxis(
		IF_DOMNode *			pContextNode,
		FLMBOOL					bForward,
		FLMBOOL					bUseKeyNodes,
		XPATH_COMPONENT *		pXPathComponent,
		IF_DOMNode **			ppCurrNode,
		eXPathAxisTypes		eAxis,
		FLMBOOL					bAxisInverted,
		FLMBOOL					bCountNodes);

	RCODE getNextXPathValue(
		IF_DOMNode *			pContextNode,
		FLMBOOL					bForward,
		FLMBOOL					bUseKeyNodes,
		FLMBOOL					bXPathIsEntireExpr,
		FQNODE *					pQNode);

	RCODE getNextFunctionValue(
		IF_DOMNode *			pContextNode,
		FLMBOOL					bForward,
		FQNODE *					pCurrNode,
		F_DynaBuf *				pDynaBuf);
		
	RCODE getFuncValue(
		IF_DOMNode *			pContextNode,
		FLMBOOL					bForward,
		FQNODE **				ppCurrNode,
		FLMBOOL *				pbGetNodeValue,
		F_DynaBuf *				pDynaBuf);
		
	RCODE getXPathValue(
		IF_DOMNode *			pContextNode,
		FLMBOOL					bForward,
		FQNODE **				ppCurrNode,
		FLMBOOL *				pbGetNodeValue,
		FLMBOOL					bUseKeyNodes,
		FLMBOOL					bXPathIsEntireExpr);
		
	RCODE setExprReturnValue(
		FLMBOOL					bUseKeyNodes,
		FQNODE *					pQueryExpr,
		FLMBOOL *				pbPassed,
		IF_DOMNode **			ppNode);
		
	RCODE evalExpr(
		IF_DOMNode *			pContextNode,
		FLMBOOL					bForward,
		FLMBOOL					bUseKeyNodes,
		FQNODE *					pQueryExpr,
		FLMBOOL *				pbPassed,
		IF_DOMNode **			ppNode);

	RCODE getAppNode(
		FLMBOOL *				pbFirstLast,
		FLMBOOL					bForward,
		XPATH_COMPONENT *		pXPathComp);
		
	RCODE testKey(
		F_DataVector *			pKey,
		PATH_PRED *				pPred,
		FLMBOOL *				pbPasses,
		IF_DOMNode **			ppPassedNode);

	RCODE getKey(
		FLMBOOL *				pbFirstLast,
		FLMBOOL					bForward,
		XPATH_COMPONENT *		pXPathComponent);

	RCODE testMetaData(
		IF_DOMNode *			pNode,
		FLMUINT					uiMetaDataType,
		PATH_PRED *				pPred,
		FLMBOOL *				pbPasses);

	RCODE getANode(
		FLMBOOL *				pbFirstLast,
		FLMBOOL					bForward,
		XPATH_COMPONENT *		pXPathComponent);

	RCODE getContextNode(
		FLMBOOL					bForward,
		XPATH_COMPONENT *		pXPathComponent);

	RCODE getNextIndexNode(
		FLMBOOL *				pbFirstLast,
		FLMBOOL					bForward,
		FQNODE *					pExprXPathSource,
		FLMBOOL					bSkipCurrKey);

	RCODE objectAddRef(
		XF_RefCount *			pObject);

	RCODE setupQueryExpr(
		FLMBOOL					bUnicode,
		IF_Db *					pDb,
		const void *			pvQuery);

	RCODE allocDupCheckSet( void);

	RCODE checkIfDup(
		IF_DOMNode **			ppNode,
		FLMBOOL *				pbPassed);

	RCODE copyValue(
		FQVALUE *				pDestValue,
		FQVALUE *				pSrcValue);

	RCODE copyXPath(
		XPATH_COMPONENT *		pXPathContext,
		FQNODE *					pDestNode,
		FXPATH **				ppDestXPath,
		FXPATH *					pSrcXPath);

	RCODE copyFunction(
		XPATH_COMPONENT *		pXPathContext,
		FQFUNCTION **			ppDestFunc,
		FQFUNCTION *			pSrcFunc);

	RCODE copyNode(
		XPATH_COMPONENT *		pXPathContext,
		FQNODE **				ppDestNode,
		FQNODE *					pSrcNode);

	RCODE copyExpr(
		XPATH_COMPONENT *		pXPathContext,
		FQNODE **				ppDestExpr,
		FQNODE *					pSrcExpr);

	void clearQuery( void);

	void initVars( void);

	FINLINE RCODE validateNode(
		IF_DOMNode *			pNode,
		FLMBOOL *				pbPassed)
	{
		RCODE	rc = NE_XFLM_OK;
		
		if (*pbPassed && m_pQueryValidator)
		{
			if (RC_BAD( rc = m_pQueryValidator->validateNode(
										(IF_Db *)m_pDb, pNode, pbPassed)))
			{
				goto Exit;
			}
			if (!(*pbPassed))
			{
				if (!m_pQuery || m_pQuery->eNodeType != FLM_XPATH_NODE || m_bRemoveDups)
				{
					m_pCurrOpt->ui64DocsFailedValidation++;
				}
				m_pCurrOpt->ui64NodesFailedValidation++;
				if (RC_BAD( rc = queryStatus()))
				{
					goto Exit;
				}
			}
		}
	Exit:
		return( rc);
	}
	
	RCODE createResultSet( void);
	
	RCODE buildResultSet(
		IF_Db *	pDb,
		FLMUINT	uiTimeLimit,
		FLMUINT	uiNumToWaitFor);
		
	void checkResultSetWaiters(
		RCODE	rc);
	
	RCODE waitResultSetBuild(
		IF_Db *	pDb,
		FLMUINT	uiTimeLimit,
		FLMUINT	uiNumToWaitFor);
		
	RCODE getFirstFromResultSet(
		IF_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT			uiTimeLimit);
		
	RCODE getLastFromResultSet(
		IF_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT			uiTimeLimit);
		
	RCODE getNextFromResultSet(
		IF_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT			uiTimeLimit,
		FLMUINT			uiNumToSkip,
		FLMUINT *		puiNumSkipped);
		
	RCODE getPrevFromResultSet(
		IF_Db *			pDb,
		IF_DOMNode **	ppNode,
		FLMUINT			uiTimeLimit,
		FLMUINT			uiNumToSkip,
		FLMUINT *		puiNumSkipped);
		
	RCODE getCurrentFromResultSet(
		IF_Db *				pDb,
		IF_DOMNode **		ppNode);
		
	RCODE verifySortKeys( void);
	
	RCODE addToResultSet( void);
	
	RCODE							m_rc;
	FQNODE *						m_pQuery;
	FLMBOOL						m_bScan;
	FLMBOOL						m_bScanIndex;
	FLMBOOL						m_bResetAllXPaths;
	FSIndexCursor *			m_pFSIndexCursor;
	XFLM_OPT_INFO				m_scanOptInfo;
	XFLM_OPT_INFO *			m_pCurrOpt;
	FLMBOOL						m_bEmpty;
	IXD *							m_pSortIxd;
	F_QueryResultSet *		m_pSortResultSet;
	RS_WAITER *					m_pFirstWaiter;
	FLMBOOL						m_bStopBuildingResultSet;
	FLMUINT						m_uiBuildThreadId;
	FLMBOOL						m_bPositioningEnabled;
	FLMBOOL						m_bResultSetPopulated;
	FLMBOOL						m_bEntriesAlreadyInOrder;
	FLMBOOL						m_bEncryptResultSet;
	FLMUINT64					m_ui64RSDocsRead;
	FLMUINT64					m_ui64RSDocsPassed;
	EXPR_STATE *				m_pCurExprState;
	F_Pool						m_Pool;
	FLMBOOL						m_bOptimized;
	FLMUINT						m_uiLanguage;
	FLMUINT						m_uiCollection;
	IF_DOMNode *				m_pCurrDoc;
	IF_DOMNode *				m_pCurrNode;
	OP_CONTEXT *				m_pCurrContext;
	CONTEXT_PATH *				m_pCurrContextPath;
	PATH_PRED *					m_pCurrPred;
	FQNODE *						m_pExprXPathSource;
	eQueryStates				m_eState;
	IF_QueryStatus *			m_pQueryStatus;
	IF_QueryValidator *		m_pQueryValidator;
	F_Database *				m_pDatabase;
	F_Db *						m_pDb;
	F_Query *					m_pNext;				// Next query off of database
	F_Query *					m_pPrev;				// Prev query off of database
	XF_RefCount **				m_ppObjectList;
	FLMUINT						m_uiObjectListSize;
	FLMUINT						m_uiObjectCount;
	FLMBOOL						m_bRemoveDups;
	FDynSearchSet *			m_pDocIdSet;
	FLMUINT						m_uiIndex;
	FLMBOOL						m_bIndexSet;
	FLMUINT						m_uiTimeLimit;
	FLMUINT						m_uiStartTime;

friend class F_Db;
friend class F_Database;
friend class F_Dict;
friend class F_IStream;
};

/*****************************************************************************
Desc:		FLAIM database system object
******************************************************************************/
class F_DbSystem : public IF_DbSystem, public F_OSBase
{
public:

	F_DbSystem() 
	{
		m_refCnt = 1;
		LockModule();
	}

	virtual ~F_DbSystem()
	{
		UnlockModule();
	}

	FLMINT XFLMAPI AddRef( void);

	FLMINT XFLMAPI Release( void);

	RCODE XFLMAPI QueryInterface(
		RXFLMIID					riid,
		void **					ppv);

	RCODE XFLMAPI init( void);

	RCODE XFLMAPI updateIniFile(
		const char *			pszParamName,
		const char *			pszValue);

	void XFLMAPI exit();

	FINLINE void XFLMAPI getFileSystem(
		IF_FileSystem **		ppFileSystem)
	{
		*ppFileSystem = (IF_FileSystem *)gv_pFileSystem;
		(*ppFileSystem)->AddRef();
	}

	RCODE XFLMAPI dbCreate(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszDictFileName,
		const char *			pszDictBuf,
		XFLM_CREATE_OPTS *	pCreateOpts,
		FLMBOOL					bTempDb,
		IF_Db **					ppDb);

	FINLINE RCODE XFLMAPI dbCreate(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszDictFileName,
		const char *			pszDictBuf,
		XFLM_CREATE_OPTS *	pCreateOpts,
		IF_Db **					ppDb)
	{
		return( dbCreate( pszDbFileName, pszDataDir, pszRflDir, pszDictFileName,
								pszDictBuf, pCreateOpts, FALSE, ppDb));
	}

	FINLINE RCODE XFLMAPI dbOpen(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszPassword,
		FLMBOOL					bAllowLimited,
		IF_Db **					ppDb)
	{
		FLMUINT		uiOpenFlags = bAllowLimited ? XFLM_ALLOW_LIMITED_MODE : 0;
		
		return( openDb( pszDbFileName, pszDataDir, pszRflDir,
							 pszPassword, uiOpenFlags, ppDb));
	}

	RCODE XFLMAPI dbRebuild(						
		const char *				pszSourceDbPath,
		const char *				pszSourceDataDir,
		const char *				pszDestDbPath,
		const char *				pszDestDataDir,
		const char *				pszDestRflDir,
		const char *				pszDictPath,
		const char *				pszPassword,
		XFLM_CREATE_OPTS *		pCreateOpts,
		FLMUINT64 *					pui64TotNodes,
		FLMUINT64 *					pui64NodesRecov,
		FLMUINT64 *					pui64QuarantinedNodes,
		IF_DbRebuildStatus *		pRebuildStatus);

	RCODE XFLMAPI dbCheck(
		const char *				pszDbFileName,
		const char *				pszDataDir,
		const char *				pszRflDir,
		const char *				pszPassword,
		FLMUINT						uiFlags,
		IF_DbInfo **				ppDbInfo,
		IF_DbCheckStatus *		pDbCheck);

	FINLINE RCODE XFLMAPI dbDup(
		IF_Db *			ifpDb,
		IF_Db **			ppDb)
	{
		F_Db *	pDb = (F_Db *)ifpDb;

		return( openDatabase( pDb->m_pDatabase, NULL, NULL, NULL, NULL, 0,
									FALSE, NULL, NULL, NULL, ppDb));
	}

	FINLINE RCODE XFLMAPI setDynamicMemoryLimit(
		FLMUINT					uiCacheAdjustPercent,
		FLMUINT					uiCacheAdjustMin,
		FLMUINT					uiCacheAdjustMax,
		FLMUINT					uiCacheAdjustMinToLeave)
	{
		return( gv_XFlmSysData.pGlobalCacheMgr->setDynamicMemoryLimit(
						uiCacheAdjustPercent, uiCacheAdjustMin,
						uiCacheAdjustMax, uiCacheAdjustMinToLeave));
	}

	FINLINE RCODE XFLMAPI setHardMemoryLimit(
		FLMUINT					uiPercent,
		FLMBOOL					bPercentOfAvail,
		FLMUINT					uiMin,
		FLMUINT					uiMax,
		FLMUINT					uiMinToLeave,
		FLMBOOL					bPreallocate)
	{
		return( gv_XFlmSysData.pGlobalCacheMgr->setHardMemoryLimit( uiPercent,
						bPercentOfAvail, uiMin, uiMax, uiMinToLeave, bPreallocate));
	}

	// Determine if dyamic cache adjusting is supported.

	FINLINE FLMBOOL XFLMAPI getDynamicCacheSupported( void)
	{
#ifdef FLM_CAN_GET_PHYS_MEM
		return( TRUE);
#else
		return( FALSE);
#endif
	}
			
	FINLINE void XFLMAPI getCacheInfo(
		XFLM_CACHE_INFO *		pCacheInfo)
	{
		gv_XFlmSysData.pGlobalCacheMgr->getCacheInfo( pCacheInfo);
	}

	// Enable/disable cache debugging mode

	void XFLMAPI enableCacheDebug(
		FLMBOOL		bDebug);

	FLMBOOL XFLMAPI cacheDebugEnabled( void);

	// Clear cache

	FINLINE RCODE XFLMAPI clearCache(
		IF_Db *		pDb)
	{
		return( gv_XFlmSysData.pGlobalCacheMgr->clearCache( pDb));
	}

	// Close all files that have not been used for the specified number of
	// seconds.

	RCODE XFLMAPI closeUnusedFiles(
		FLMUINT		uiSeconds);

	// Maximum number of file handles available.

	void XFLMAPI setOpenThreshold(
		FLMUINT		uiThreshold);

	FLMUINT XFLMAPI getOpenThreshold( void);

	// Get the number of open files

	FLMUINT XFLMAPI getOpenFileCount( void);

	// Start gathering statistics.
	
	void XFLMAPI startStats( void);

	// Stop gathering statistics.
	
	void XFLMAPI stopStats( void);

	// Reset statistics.
	
	void XFLMAPI resetStats( void);

	RCODE XFLMAPI getStats(
		XFLM_STATS *			pFlmStats);

	void XFLMAPI freeStats(
		XFLM_STATS *			pFlmStats);

	// Set the maximum number of queries to save.
	
	void XFLMAPI setQuerySaveMax(
		FLMUINT					uiMaxToSave);

	FLMUINT XFLMAPI getQuerySaveMax( void);

	// Set temporary directory.
	
	RCODE XFLMAPI setTempDir(
		const char *			pszPath);

	RCODE XFLMAPI getTempDir(
		char *					pszPath);

	// Maximum seconds between checkpoints.	

	void XFLMAPI setCheckpointInterval(
		FLMUINT					uiSeconds);

	FLMUINT XFLMAPI getCheckpointInterval( void);

	// Set interval for dynamically adjusting cache limit.

	void XFLMAPI setCacheAdjustInterval(
		FLMUINT					uiSeconds);

	FLMUINT XFLMAPI getCacheAdjustInterval( void);

	// Set interval for dynamically cleaning out old cache blocks and records.
	
	void XFLMAPI setCacheCleanupInterval(
		FLMUINT					uiSeconds);

	FLMUINT XFLMAPI getCacheCleanupInterval( void);

	// Set interval for cleaning up unused structures.

	void XFLMAPI setUnusedCleanupInterval(
		FLMUINT					uiSeconds);

	FLMUINT XFLMAPI getUnusedCleanupInterval( void);

	// Set maximum time for an item to be unused.
	
	void XFLMAPI setMaxUnusedTime(
		FLMUINT					uiSeconds);

	FLMUINT XFLMAPI getMaxUnusedTime( void);
	
	// Specify the logger object

	void XFLMAPI setLogger(
		IF_LoggerClient *		pLogger);
		
	// Enable or disable use of ESM
	
	void XFLMAPI enableExtendedServerMemory(
		FLMBOOL					bEnable);

	FLMBOOL XFLMAPI extendedServerMemoryEnabled( void);

	void XFLMAPI deactivateOpenDb(
		const char *			pszDbFileName,
		const char *			pszDataDir);

	// Maximum dirty cache.
	
	void XFLMAPI setDirtyCacheLimits(
		FLMUINT					uiMaxDirty,
		FLMUINT					uiLowDirty);

	void XFLMAPI getDirtyCacheLimits(
		FLMUINT *			puiMaxDirty,
		FLMUINT *			puiLowDirty);

	RCODE XFLMAPI getThreadInfo(
		IF_ThreadInfo **	ppThreadInfo);

	RCODE XFLMAPI registerForEvent(
		eEventCategory		eCategory,
		IF_EventClient *	pEventClient);

	void XFLMAPI deregisterForEvent(
		eEventCategory		eCategory,
		IF_EventClient *	pEventClient);

	RCODE XFLMAPI getNextMetaphone(
		IF_IStream *		pIStream,
		FLMUINT *			puiMetaphone,
		FLMUINT *			puiAltMetaphone = NULL);

	RCODE XFLMAPI dbCopy(
		const char *		pszSrcDbName,
		const char *		pszSrcDataDir,
		const char *		pszSrcRflDir,
		const char *		pszDestDbName,
		const char *		pszDestDataDir,
		const char *		pszDestRflDir,
		IF_DbCopyStatus *	ifpStatus);

	RCODE XFLMAPI dbRemove(
		const char *		pszDbName,
		const char *		pszDataDir,
		const char *		pszRflDir,
		FLMBOOL				bRemoveRflFiles);

	RCODE XFLMAPI dbRename(
		const char *			pszDbName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszNewDbName,
		FLMBOOL					bOverwriteDestOk,
		IF_DbRenameStatus *	ifpStatus);

	RCODE XFLMAPI dbRestore(
		const char *			pszDbPath,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszBackupPath,
		const char *			pszPassword,
		IF_RestoreClient *	pRestoreObj,
		IF_RestoreStatus *	pRestoreStatus);

	RCODE XFLMAPI strCmp(
		FLMUINT					uiCompFlags,
		FLMUINT					uiLanguage,
		FLMUNICODE *			uzStr1,
		FLMUNICODE *			uzStr2,
		FLMINT *					piCmp);

	FLMBOOL XFLMAPI errorIsFileCorrupt(
		RCODE						rc);

	static FLMBOOL _errorIsFileCorrupt(
		RCODE						rc)
	{
		F_DbSystem		dbSystem;

		return( dbSystem.errorIsFileCorrupt( rc));
	}

	const char * XFLMAPI errorString(
		RCODE						rc);

	const char * XFLMAPI checkErrorToStr(
		FLMINT					iCheckErrorCode);

	static const char * _errorString(
		RCODE						rc)
	{
		F_DbSystem		dbSystem;

		return( dbSystem.errorString( rc));
	}

	RCODE XFLMAPI openBufferIStream(
		const char *			pucBuffer,
		FLMUINT					uiLength,
		IF_PosIStream **		ppIStream);

	RCODE XFLMAPI openFileIStream(
		const char *			pszPath,
		IF_PosIStream **		ppIStream);

	RCODE XFLMAPI openMultiFileIStream(
		const char *			pszDirectory,
		const char *			pszBaseName,
		IF_IStream **			ppIStream);
		
	RCODE XFLMAPI openBufferedIStream(
		IF_IStream *			pIStream,
		FLMUINT					uiBufferSize,
		IF_IStream **			ppIStream);

	RCODE XFLMAPI openUncompressingIStream(
		IF_IStream *			pIStream,
		IF_IStream **			ppIStream);
		
	RCODE XFLMAPI openFileOStream(
		const char *			pszFileName,
		FLMBOOL					bTruncateIfExists,
		IF_OStream **			ppOStream);
		
	RCODE XFLMAPI openMultiFileOStream(
		const char *			pszDirectory,
		const char *			pszBaseName,
		FLMUINT					uiMaxFileSize,
		FLMBOOL					bOverwrite,
		IF_OStream **			ppStream);
		
	RCODE XFLMAPI removeMultiFileStream(
		const char *			pszDirectory,
		const char *			pszBaseName);
		
	RCODE XFLMAPI openBufferedOStream(
		IF_OStream *			pOStream,
		FLMUINT					uiBufferSize,
		IF_OStream **			ppOStream);
		
	RCODE XFLMAPI openCompressingOStream(
		IF_OStream *			pOStream,
		IF_OStream **			ppOStream);
		
	RCODE XFLMAPI writeToOStream(
		IF_IStream *			pIStream,
		IF_OStream *			pOStream);

	RCODE XFLMAPI openBase64Encoder(
		IF_IStream *			pInputStream,
		FLMBOOL					bInsertLineBreaks,
		IF_IStream **			ppEncodedStream);

	RCODE XFLMAPI openBase64Decoder(
		IF_IStream *			pInputStream,
		IF_IStream **			ppDecodedStream);

	FINLINE RCODE XFLMAPI createMemoryPool(
		IF_Pool **				ppPool)
	{
		if( (*ppPool = f_new F_Pool) == NULL)
		{
			return( RC_SET( NE_XFLM_MEM));
		}
		
		return( NE_XFLM_OK);
	}		
		
	RCODE XFLMAPI createIFDataVector(
		IF_DataVector **		ifppDV);

	RCODE XFLMAPI createIFResultSet(
		IF_ResultSet **		ppResultSet);

	RCODE XFLMAPI createIFQuery(
		IF_Query **				ppQuery);
	
	FINLINE void XFLMAPI freeMem(
		void **					ppMem)
	{
		f_free( ppMem);
	}

	FINLINE RCODE internalDbOpen(
		F_Database *			pDatabase,
		F_Db **					ppDb)
	{
		RCODE		rc = NE_XFLM_OK;
		IF_Db *	pDb;

		if (RC_OK( rc = openDatabase( pDatabase, NULL, NULL, NULL,
				NULL, 0, TRUE, NULL, NULL, NULL, &pDb)))
		{
			*ppDb = (F_Db *)pDb;
		}
		return( rc);
	}

	RCODE openDb(
		const char *	pszDbFileName,
		const char *	pszDataDir,
		const char *	pszRflDir,
		const char *	pszPassword,
		FLMUINT			uiOpenFlags,
		IF_Db **			ppDb);
	
	void enableOutOfMemorySimulation(
		FLMBOOL			bEnable);

	FLMBOOL outOfMemorySimulationEnabled( void);

	static FINLINE FLMBOOL validBlockSize(
		FLMUINT			uiBlockSize)
	{
		if( uiBlockSize == 4096 || uiBlockSize == 8192)
		{
			return( TRUE);
		}

		return( FALSE);
	}

	static FLMUINT languageToNum(
		const char *			pszLanguage);

	static void languageToStr(
		FLMINT					iLangNum,
		char *					pszLanguage);

	static void getDbBasePath(
		char *					pszBaseDbName,
		const char *			pszDbName,
		FLMUINT *				puiBaseDbNameLen);

	RCODE XFLMAPI compareUTF8Strings(
		const FLMBYTE *		pucLString,
		FLMUINT					uiLStrBytes,
		FLMBOOL					bLeftWild,
		const FLMBYTE *		pucRString,
		FLMUINT					uiRStrBytes,
		FLMBOOL					bRightWild,
		FLMUINT					uiCompareRules,
		FLMUINT					uiLanguage,
		FLMINT *					piResult);
			
	RCODE XFLMAPI compareUnicodeStrings(
		const FLMUNICODE *	puzLString,
		FLMUINT					uiLStrBytes,
		FLMBOOL					bLeftWild,
		const FLMUNICODE *	puzRString,
		FLMUINT					uiRStrBytes,
		FLMBOOL					bRightWild,
		FLMUINT					uiCompareRules,
		FLMUINT					uiLanguage,
		FLMINT *					piResult);

	RCODE XFLMAPI utf8IsSubStr(
		const FLMBYTE *		pszString,
		const FLMBYTE *		pszSubString,
		FLMUINT					uiCompareRules,
		FLMUINT					uiLanguage,
		FLMBOOL *				pbExists);
	
	FLMBOOL XFLMAPI uniIsUpper(
		FLMUNICODE				uzChar);

	FLMBOOL XFLMAPI uniIsLower(
		FLMUNICODE				uzChar);

	FLMBOOL XFLMAPI uniIsAlpha(
		FLMUNICODE				uzChar);

	FLMBOOL XFLMAPI uniIsDecimalDigit(
		FLMUNICODE				uzChar);

	FLMUNICODE XFLMAPI uniToLower(
		FLMUNICODE				uzChar);

	RCODE	XFLMAPI nextUCS2Char(
		const FLMBYTE **		ppszUTF8,
		const FLMBYTE *		pszEndOfUTF8String,
		FLMUNICODE *			puzChar);
		
	RCODE XFLMAPI numUCS2Chars(
		const FLMBYTE *		pszUTF8,
		FLMUINT *				puiNumChars);

	RCODE XFLMAPI waitToClose(
		const char *	pszDbPath);
	
	RCODE XFLMAPI createIFNodeInfo(
		IF_NodeInfo **			ifppNodeInfo);
		
	RCODE XFLMAPI createIFBTreeInfo(
		IF_BTreeInfo **		ifppBTreeInfo);
		
private:

	// Methods

	RCODE readIniFile( void);

	RCODE setCacheParams(
		F_IniFile *			pIniFile);

	void cleanup( void);
	
	FINLINE RCODE internalDbDup(
		F_Db *	pDb,
		F_Db **	ppDb
		)
	{
		RCODE		rc = NE_XFLM_OK;
		IF_Db *	ifpDb;

		if (RC_OK( rc = openDatabase( pDb->m_pDatabase, NULL, NULL,
				NULL, NULL, 0, TRUE, NULL, NULL, NULL, &ifpDb)))
		{
			*ppDb = (F_Db *)ifpDb;
		}

		return( rc);
	}

	void lockSysData( void);
	
	void unlockSysData( void);
	
	RCODE initCharMappingTables( void);
	
	void initFastBlockCheckSum( void);

	void freeCharMappingTables( void);

	RCODE checkErrorCodeTables( void);

	RCODE allocDb(
		F_Db **				ppDb,
		FLMBOOL				bInternalOpen);

	RCODE findDatabase(
		const char *		pszDbPath,
		const char *		pszDataDir,
		F_Database **		ppDatabase);

	RCODE checkDatabaseClosed(
		const char *		pszDbName,
		const char *		pszDataDir);

	RCODE allocDatabase(
		const char *		pszDbPath,
		const char *		pszDataDir,
		FLMBOOL				bTempDb,
		F_Database **		ppDatabase);

	RCODE openDatabase(
		F_Database *			pDatabase,
		const char *			pszDbPath,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszPassword,
		FLMUINT					uiOpenFlags,
		FLMBOOL					bInternalOpen,
		IF_RestoreClient *	pRestoreObj,
		IF_RestoreStatus *	pRestoreStatus,
		IF_FileHdl *			pLockFileHdl,
		IF_Db **					ppDb);

	RCODE copyDb(
		const char *			pszSrcDbName,
		const char *			pszSrcDataDir,
		const char *			pszSrcRflDir,
		const char *			pszDestDbName,
		const char *			pszDestDataDir,
		const char *			pszDestRflDir,
		IF_DbCopyStatus *		ifpStatus);

	static RCODE monitorThrd(
		F_Thread *		pThread);
		
	static RCODE cacheCleanupThrd(
		F_Thread *		pThread);

	static void checkNotUsedObjects( void);

	static FLMATOMIC			m_flmSysSpinLock;
	static FLMUINT				m_uiFlmSysStartupCount;

friend class F_Db;
friend class F_Database;
friend class F_DbRebuild;
friend class F_DbCheck;
};

// Supported text types

typedef enum
{
	XFLM_UNICODE_TEXT = 1,
	XFLM_UTF8_TEXT
} eXFlmTextType;

/*------------------------------------------------------
	FLAIM Processing Hooks (call-backs)
-------------------------------------------------------*/

#define FLM_DATA_LEFT_TRUNCATED	0x10	// Data is left truncated
#define FLM_DATA_RIGHT_TRUNCATED	0x20	// Data is right truncated

RCODE flmReadSEN(
	IF_IStream *		pIStream,
	FLMUINT *			puiValue,
	FLMUINT *			puiLength = NULL);

RCODE flmReadSEN64(
	IF_IStream *		pIStream,
	FLMUINT64 *			pui64Value,
	FLMUINT *			puiLength = NULL);

RCODE flmReadUTF8CharAsUnicode(
	IF_IStream *		pStream,
	FLMUNICODE *		puChar);

RCODE flmReadUTF8CharAsUTF8(
	IF_IStream *		pIStream,
	FLMBYTE *			pucBuf,
	FLMUINT *			puiLen);

RCODE flmReadStorageAsText(
	IF_IStream *		pIStream,
	FLMBYTE *			pucStorageData,
	FLMUINT				uiDataLen,
	FLMUINT				uiDataType,
	void *				pvBuffer,
	FLMUINT 				uiBufLen,
	eXFlmTextType		eTextType,
	FLMUINT				uiMaxCharsToRead,
	FLMUINT				uiCharOffset,
	FLMUINT *			puiCharsRead,
	FLMUINT *			puiBufferBytesUsed);

RCODE flmReadStorageAsBinary(
	IF_IStream *		pIStream,
	void *				pvBuffer,
	FLMUINT 				uiBufLen,
	FLMUINT				uiByteOffset,
	FLMUINT *			puiBytesRead);

RCODE flmReadStorageAsNumber(
	IF_IStream *		pIStream,
	FLMUINT				uiDataType,
	FLMUINT64 *			pui64Number,
	FLMBOOL *			pbNeg);

RCODE flmReadLine(
	IF_IStream *		pIStream,
	FLMBYTE *			pucBuffer,
	FLMUINT *			puiSize);

typedef struct F_CollStreamPos
{
	FLMUINT64		ui64Position;
	FLMUNICODE		uNextChar;
} F_CollStreamPos;

/*****************************************************************************
Desc:
******************************************************************************/
class F_CollIStream : public F_PosIStream
{
public:

	F_CollIStream()
	{
		m_pIStream = NULL;
		m_uiLanguage = 0;
		m_bMayHaveWildCards = FALSE;
		m_bUnicodeStream = FALSE;
		m_uNextChar = 0;
	}

	virtual ~F_CollIStream()
	{
		if( m_pIStream)
		{
			m_pIStream->Release();
		}
	}

	RCODE open(
		IF_PosIStream *	pIStream,
		FLMBOOL				bUnicodeStream,
		FLMUINT				uiLanguage,
		FLMUINT				uiCompareRules,
		FLMBOOL				bMayHaveWildCards)
	{
		if( m_pIStream)
		{
			m_pIStream->Release();
		}

		m_pIStream = pIStream;
		m_pIStream->AddRef();
		m_uiLanguage = uiLanguage;
		m_uiCompareRules = uiCompareRules;
		m_bCaseSensitive = (uiCompareRules & XFLM_COMP_CASE_INSENSITIVE)
								  ? FALSE
								  : TRUE;
		m_bMayHaveWildCards = bMayHaveWildCards;
		m_bUnicodeStream = bUnicodeStream;		
		m_ui64EndOfLeadingSpacesPos = 0;
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI close( void)
	{
		if( m_pIStream)
		{
			m_pIStream->Release();
			m_pIStream = NULL;
		}
		
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI read(
		void *			pvBuffer,
		FLMUINT			uiBytesToRead,
		FLMUINT *		puiBytesRead)
	{
		RCODE		rc = NE_XFLM_OK;

		if( RC_BAD( rc = m_pIStream->read( pvBuffer, 
			uiBytesToRead, puiBytesRead)))
		{
			goto Exit;
		}

	Exit:

		return( rc);
	}

	RCODE read(
		FLMBOOL			bAllowTwoIntoOne,
		FLMUNICODE *	puChar,
		FLMBOOL *		pbCharIsWild,
		FLMUINT16 *		pui16Col,
		FLMUINT16 *		pui16SubCol,
		FLMBYTE *		pucCase);
		
	FINLINE FLMUINT64 XFLMAPI totalSize( void)
	{
		if( m_pIStream)
		{
			return( m_pIStream->totalSize());
		}

		return( 0);
	}

	FINLINE FLMUINT64 XFLMAPI remainingSize( void)
	{
		if( m_pIStream)
		{
			return( m_pIStream->remainingSize());
		}

		return( 0);
	}

	FINLINE RCODE XFLMAPI positionTo(
		FLMUINT64)
	{
		return( RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED));
	}

	FINLINE RCODE XFLMAPI positionTo(
		F_CollStreamPos *	pPos)
	{
		
		// Should never be able to position back to before the
		// leading spaces.
		
		m_uNextChar = pPos->uNextChar;
		flmAssert( pPos->ui64Position >= m_ui64EndOfLeadingSpacesPos);
		return( m_pIStream->positionTo( pPos->ui64Position));
	}

	FINLINE FLMUINT64 XFLMAPI getCurrPosition( void)
	{
		flmAssert( 0);
		return( 0);
	}

	FINLINE void XFLMAPI getCurrPosition(
		F_CollStreamPos *		pPos)
	{
		pPos->uNextChar = m_uNextChar;
		pPos->ui64Position = m_pIStream->getCurrPosition();
	}

private:

	FINLINE RCODE readCharFromStream(
		FLMUNICODE *		puChar)
	{
		RCODE		rc = NE_XFLM_OK;
		
		if( m_bUnicodeStream)
		{
			if( RC_BAD( rc = m_pIStream->read( puChar, sizeof( FLMUNICODE), NULL)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = flmReadUTF8CharAsUnicode( 
				m_pIStream, puChar)))
			{
				goto Exit;
			}
		}
		
	Exit:
	
		return( rc);
	}
		
	IF_PosIStream *	m_pIStream;
	FLMUINT				m_uiLanguage;
	FLMBOOL				m_bCaseSensitive;
	FLMUINT				m_uiCompareRules;
	FLMUINT64			m_ui64EndOfLeadingSpacesPos;
	FLMBOOL				m_bMayHaveWildCards;
	FLMBOOL				m_bUnicodeStream;
	FLMUNICODE			m_uNextChar;
};

#define FLM_ENCRYPT_CHUNK_SIZE 512

/*****************************************************************************
Desc:
******************************************************************************/
class F_BTreeIStream : public F_PosIStream
{
public:

	F_BTreeIStream()
	{
		m_pucBuffer = NULL;
		m_pBTree = NULL;
		m_bReleaseBTree = FALSE;
		reset();
	}

	virtual ~F_BTreeIStream()
	{
		reset();
	}

	FINLINE void reset( void)
	{
		m_pNextInPool = NULL;
		if( m_pBTree && m_bReleaseBTree)
		{
			m_pBTree->btClose();
			gv_XFlmSysData.pBtPool->btpReturnBtree( &m_pBTree);
			m_pBTree = NULL;
		}

		if( m_pucBuffer != &m_ucBuffer [0])
		{
			f_free( &m_pucBuffer);
		}

		m_pDb = NULL;
		m_uiCollection = 0;
		m_ui64NodeId = 0;
		m_pBTree = NULL;
		m_bReleaseBTree = FALSE;
		m_uiKeyLength = 0;
		m_uiStreamSize = 0;
		m_uiBufferBytes = 0;
		m_uiBufferOffset = 0;
		m_uiBufferStartOffset = 0;
		m_uiBufferSize = sizeof( m_ucBuffer);
		m_pucBuffer = &m_ucBuffer [0];
		m_ui32BlkAddr = 0;
		m_uiOffsetIndex = 0;
		m_bDataEncrypted = FALSE;
		m_bBufferDecrypted = FALSE;
		m_uiDataLength = 0;
		m_uiEncDefId = 0;
	}

	RCODE open(
		F_Db *			pDb,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		FLMUINT32		ui32BlkAddr = 0,
		FLMUINT			uiOffsetIndex = 0);

	RCODE open(
		F_Db *			pDb,
		F_Btree *		pBTree,
		FLMUINT			uiFlags,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		FLMUINT32		ui32BlkAddr = 0,
		FLMUINT			uiOffsetIndex = 0);

	FINLINE FLMUINT64 XFLMAPI totalSize( void)
	{
		return( m_uiStreamSize);
	}

	FINLINE FLMUINT64 XFLMAPI remainingSize( void)
	{
		return( m_uiStreamSize - (m_uiBufferStartOffset + m_uiBufferOffset));
	}

	FINLINE RCODE XFLMAPI close( void)
	{
		reset();
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI positionTo(
		FLMUINT64		ui64Position);

	FINLINE FLMUINT64 XFLMAPI getCurrPosition( void)
	{
		return( m_uiBufferStartOffset + m_uiBufferOffset);
	}

	RCODE XFLMAPI read(
		void *			pvBuffer,
		FLMUINT			uiBytesToRead,
		FLMUINT *		puiBytesRead);

	FLMINT XFLMAPI Release( void);
	
	FINLINE FLMUINT32 getBlkAddr( void)
	{
		return( m_ui32BlkAddr);
	}

	FINLINE FLMUINT getOffsetIndex( void)
	{
		return( m_uiOffsetIndex);
	}
	
private:

	F_BTreeIStream *	m_pNextInPool;
	F_Db *				m_pDb;
	F_Btree *			m_pBTree;
	FLMUINT				m_uiCollection;
	FLMUINT64			m_ui64NodeId;
	FLMUINT				m_uiStreamSize;
	FLMUINT				m_uiKeyLength;
	FLMUINT				m_uiBufferBytes;
	FLMUINT				m_uiBufferSize;
	FLMUINT				m_uiBufferOffset;
	FLMUINT				m_uiBufferStartOffset;
	FLMUINT				m_uiDataLength;
	FLMUINT				m_uiEncDefId;
	FLMBYTE				m_ucBuffer[ FLM_ENCRYPT_CHUNK_SIZE];
	FLMBYTE *			m_pucBuffer;
	FLMUINT				m_uiOffsetIndex;
	FLMUINT32			m_ui32BlkAddr;
	FLMBOOL				m_bReleaseBTree;
	FLMBOOL				m_bDataEncrypted;
	FLMBOOL				m_bBufferDecrypted;
	FLMBYTE				m_ucKey[ FLM_MAX_NUM_BUF_SIZE];
	FLMBYTE				m_ucIV [16];
friend class F_DOMNode;
friend class F_CachedNode;
friend class F_NodePool;
friend class F_Db;
};

/*****************************************************************************
Desc:
******************************************************************************/
class F_NodeBufferIStream : public F_BufferIStream
{
public:

	F_NodeBufferIStream()
	{
		m_pCachedNode = NULL;
		reset();
	}

	virtual ~F_NodeBufferIStream()
	{
		reset();
	}

	FINLINE void reset( void)
	{
		if (m_pCachedNode)
		{
			f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
			m_pCachedNode->decrNodeUseCount();
			m_pCachedNode->decrStreamUseCount();
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
			m_pCachedNode = NULL;
		}
	}

	F_CachedNode *				m_pCachedNode;
friend class F_CachedNode;
};

/*****************************************************************************
Desc:
******************************************************************************/
class F_DOMNode : public IF_DOMNode, public XF_Base
{
public:

	F_DOMNode()
	{
		m_pCachedNode = NULL;
		resetDOMNode( FALSE);
	}

	virtual ~F_DOMNode()
	{
		resetDOMNode( FALSE);
	}

	void resetDOMNode( 
		FLMBOOL bMutexAlreadyLocked)
	{
		m_pNextInPool = NULL;
		m_uiAttrNameId = 0;

		if (m_pCachedNode)
		{
			if( !bMutexAlreadyLocked)
			{
				f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
			}
			m_pCachedNode->decrNodeUseCount();
			if( !bMutexAlreadyLocked)
			{
				f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
			}
			m_pCachedNode = NULL;
		}
	}

	FLMINT XFLMAPI Release( void);

	RCODE XFLMAPI createNode(
		IF_Db *				pDb,
		eDomNodeType		eNodeType,
		FLMUINT				uiNameId,
		eNodeInsertLoc		eLocation,
		IF_DOMNode **		ppNewNode,
		FLMUINT64 *			pui64NodeId = NULL);

	RCODE XFLMAPI createChildElement(
		IF_Db *				pDb,
		FLMUINT				uiChildElementNameId,
		eNodeInsertLoc		eLocation,
		IF_DOMNode **		ppNewChildElementNode,
		FLMUINT64 *			pui64NodeId = NULL);
		
	RCODE XFLMAPI deleteNode(
		IF_Db *				pDb);

	RCODE XFLMAPI deleteChildren(
		IF_Db *				pDb,
		FLMUINT				uiNameId = 0);

	RCODE XFLMAPI createAttribute(
		IF_Db *				pDb,
		FLMUINT				uiAttrNameId,
		IF_DOMNode **		ppAttrNode);

	RCODE XFLMAPI getFirstAttribute(
		IF_Db *				pDb,
		IF_DOMNode **		ppAttrNode);

	RCODE XFLMAPI getLastAttribute(
		IF_Db *				pDb,
		IF_DOMNode **		ppAttrNode);

	FINLINE RCODE XFLMAPI getAttribute(
		IF_Db *				pDb,
		FLMUINT				uiAttrNameId,
		IF_DOMNode **		ppAttrNode)
	{
		return( hasAttribute( pDb, uiAttrNameId, ppAttrNode));
	}

	RCODE XFLMAPI deleteAttribute(
		IF_Db *				pDb,
		FLMUINT				uiAttrNameId);

	RCODE XFLMAPI hasAttribute(
		IF_Db *				pDb,
		FLMUINT				uiAttrNameId,
		IF_DOMNode **		ppAttrNode = NULL);

	RCODE XFLMAPI hasAttributes(
		IF_Db *				pDb,
		FLMBOOL *			pbHasAttrs);

	RCODE XFLMAPI hasNextSibling(
		IF_Db *				pDb,
		FLMBOOL *			pbHasNextSibling);

	RCODE XFLMAPI hasPreviousSibling(
		IF_Db *				pDb,
		FLMBOOL *			pbHasPreviousSibling);

	RCODE XFLMAPI hasChildren(
		IF_Db *				pDb,
		FLMBOOL *			pbHasChildren);

	RCODE XFLMAPI isNamespaceDecl(
		IF_Db *				pDb,
		FLMBOOL *			pbIsNamespaceDecl);

	FINLINE eDomNodeType XFLMAPI getNodeType( void)
	{
		if( m_uiAttrNameId)
		{
			return( ATTRIBUTE_NODE);
		}
		else if( m_pCachedNode)
		{
			return( m_pCachedNode->getNodeType());
		}
		
		flmAssert( 0);
		return( INVALID_NODE);
	}
	
	RCODE XFLMAPI getNodeId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64NodeId);

	RCODE XFLMAPI getParentId(
		IF_Db *			pDb,
		FLMUINT64 *		pui64ParentId);

	RCODE XFLMAPI getDocumentId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64DocumentId);
		
	RCODE XFLMAPI getPrevSibId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64PrevSibId);

	RCODE XFLMAPI getNextSibId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64NextSibId);

	RCODE XFLMAPI getFirstChildId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64FirstChildId);

	RCODE XFLMAPI getLastChildId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64LastChildId);

	RCODE XFLMAPI getNameId(
		IF_Db *				pDb,
		FLMUINT *			puiNameId);

	virtual RCODE XFLMAPI getEncDefId(
		IF_Db *					pDb,
		FLMUINT *				puiEncDefId);

	RCODE XFLMAPI getDataType(
		IF_Db *				pDb,
		FLMUINT *			puiDataType);

	RCODE XFLMAPI getDataLength(
		IF_Db *			pDb,
		FLMUINT *		puiLength);

	FINLINE RCODE XFLMAPI getUINT32(
		IF_Db *			pDb,
		FLMUINT32 *		pui32Value)
	{
		RCODE			rc;
		FLMUINT64	ui64Value;

		if( RC_BAD( rc = getNumber64( (F_Db *)pDb, &ui64Value, NULL)))
		{
			return( rc);
		}
		
		return( convertToUINT32( ui64Value, FALSE, pui32Value));
	}

	FINLINE RCODE XFLMAPI getUINT(
		IF_Db *			pDb,
		FLMUINT *		puiValue)
	{
		RCODE			rc;
		FLMUINT64	ui64Value;

		if( RC_BAD( rc = getNumber64( (F_Db *)pDb, &ui64Value, NULL)))
		{
			return( rc);
		}
		
		return( convertToUINT( ui64Value, FALSE, puiValue));
	}

	FINLINE RCODE XFLMAPI getUINT64(
		IF_Db *			pDb,
		FLMUINT64 *		pui64Value)
	{
		return( getNumber64( (F_Db *)pDb, pui64Value, NULL));
	}

	FINLINE RCODE XFLMAPI getINT32(
		IF_Db *			pDb,
		FLMINT32 *		pi32Value)
	{
		RCODE				rc;
		FLMUINT64		ui64Value;
		FLMBOOL			bNeg;

		if( RC_BAD( rc = getNumber64( (F_Db *)pDb, &ui64Value, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToINT32( ui64Value, bNeg, pi32Value));
	}

	FINLINE RCODE XFLMAPI getINT(
		IF_Db *				pDb,
		FLMINT *				piValue)
	{
		RCODE				rc;
		FLMUINT64		ui64Value;
		FLMBOOL			bNeg;

		if( RC_BAD( rc = getNumber64( (F_Db *)pDb, &ui64Value, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToINT( ui64Value, bNeg, piValue));
	}

	FINLINE RCODE XFLMAPI getINT64(
		IF_Db *				pDb,
		FLMINT64 *			pi64Value)
	{
		RCODE				rc;
		FLMUINT64		ui64Value;
		FLMBOOL			bNeg;

		if( RC_BAD( rc = getNumber64( (F_Db *)pDb, &ui64Value, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToINT64( ui64Value, bNeg, pi64Value));
	}

	RCODE XFLMAPI getMetaValue(
		IF_Db *					pDb,
		FLMUINT64 *				pui64Value);

	FINLINE RCODE XFLMAPI getUnicodeChars(
		IF_Db *				pDb,
		FLMUINT *			puiNumChars)
	{
		return( getUnicode( pDb, NULL, 0, 0, FLM_MAX_UINT, puiNumChars));
	}

	RCODE XFLMAPI getUnicode(
		IF_Db *				pDb,
		FLMUNICODE *		puzValueBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT				uiCharOffset,
		FLMUINT				uiMaxCharsRequested,
		FLMUINT *			puiCharsReturned = NULL,
		FLMUINT *			puiBufferBytesUsed = NULL);

	RCODE XFLMAPI getUnicode(
		IF_Db *				pDb,
		FLMUNICODE **		ppuzUnicodeValue);

	RCODE XFLMAPI getUnicode(
		IF_Db *				pDb,
		IF_DynaBuf *		pDynaBuf);

	RCODE XFLMAPI getUTF8(
		IF_Db *				pDb,
		FLMBYTE *			pszValueBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT				uiCharOffset,
		FLMUINT				uiMaxCharsRequested,
		FLMUINT *			puiCharsReturned = NULL,
		FLMUINT *			puiBufferBytesUsed = NULL);

	RCODE XFLMAPI getUTF8(
		IF_Db *				pDb,
		FLMBYTE **			ppszUTF8Value);
		
	RCODE XFLMAPI getUTF8(
		IF_Db *				pDb,
		IF_DynaBuf *		pDynaBuf);

	RCODE XFLMAPI getBinary(
		IF_Db *				pDb,
		void *				pvValue,
		FLMUINT				uiByteOffset,
		FLMUINT				uiBytesRequested,
		FLMUINT *			puiBytesReturned);

	RCODE XFLMAPI getBinary(
		IF_Db *				pDb,
		IF_DynaBuf *		pBuffer);

	FINLINE RCODE XFLMAPI getAttributeValueUINT32(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT32 *				pui32Num)
	{
		RCODE				rc = NE_XFLM_OK;
		FLMUINT64		ui64Num;
		FLMBOOL			bNeg;
		
		if( RC_BAD( rc = getAttributeValueNumber( (F_Db *)pDb, 
			uiAttrName, &ui64Num, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToUINT32( ui64Num, bNeg, pui32Num));
	}

	FINLINE RCODE XFLMAPI getAttributeValueUINT32(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT32 *				pui32Num,
		FLMUINT32				ui32NotFoundDefault)
	{
		RCODE		rc = NE_XFLM_OK;
		
		if( RC_BAD( rc = getAttributeValueUINT32( 
			pDb, uiAttrName, pui32Num)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				return( rc);
			}
			
			*pui32Num = ui32NotFoundDefault;
			rc = NE_XFLM_OK;
		}
		
		return( rc);
	}
			
	FINLINE RCODE XFLMAPI getAttributeValueUINT(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT *				puiNum)
	{
		RCODE				rc = NE_XFLM_OK;
		FLMUINT64		ui64Num;
		FLMBOOL			bNeg;
		
		if( RC_BAD( rc = getAttributeValueNumber( (F_Db *)pDb, 
			uiAttrName, &ui64Num, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToUINT( ui64Num, bNeg, puiNum));
	}

	FINLINE RCODE XFLMAPI getAttributeValueUINT(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT *				puiNum,
		FLMUINT					uiNotFoundDefault)
	{
		RCODE			rc = NE_XFLM_OK;
		
		if( RC_BAD( rc = getAttributeValueUINT( 
			pDb, uiAttrName, puiNum)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				return( rc);
			}
			
			*puiNum = uiNotFoundDefault;
			rc = NE_XFLM_OK;
		}
		
		return( rc);
	}

	FINLINE RCODE XFLMAPI getAttributeValueUINT64(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT64 *				pui64Num)
	{
		RCODE				rc = NE_XFLM_OK;
		FLMUINT64		ui64Num;
		FLMBOOL			bNeg;
		
		if( RC_BAD( rc = getAttributeValueNumber( (F_Db *)pDb, 
			uiAttrName, &ui64Num, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToUINT64( ui64Num, bNeg, pui64Num));
	}

	FINLINE RCODE XFLMAPI getAttributeValueUINT64(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT64 *				pui64Num,
		FLMUINT64				ui64NotFoundDefault)
	{
		RCODE			rc = NE_XFLM_OK;
		
		if( RC_BAD( rc = getAttributeValueUINT64( 
			pDb, uiAttrName, pui64Num)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				return( rc);
			}
			
			*pui64Num = ui64NotFoundDefault;
			rc = NE_XFLM_OK;
		}
		
		return( rc);
	}

	FINLINE RCODE XFLMAPI getAttributeValueINT(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT *					piNum)
	{
		RCODE				rc = NE_XFLM_OK;
		FLMUINT64		ui64Num;
		FLMBOOL			bNeg;
		
		if( RC_BAD( rc = getAttributeValueNumber( (F_Db *)pDb, 
			uiAttrName, &ui64Num, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToINT( ui64Num, bNeg, piNum));
	}

	FINLINE RCODE XFLMAPI getAttributeValueINT(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT *					piNum,
		FLMINT					iNotFoundDefault)
	{
		RCODE			rc = NE_XFLM_OK;
		
		if( RC_BAD( rc = getAttributeValueINT( 
			pDb, uiAttrName, piNum)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				return( rc);
			}
			
			*piNum = iNotFoundDefault;
			rc = NE_XFLM_OK;
		}
		
		return( rc);
	}
	
	FINLINE RCODE XFLMAPI getAttributeValueINT64(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT64 *				pi64Num)
	{
		RCODE				rc = NE_XFLM_OK;
		FLMUINT64		ui64Num;
		FLMBOOL			bNeg;
		
		if( RC_BAD( rc = getAttributeValueNumber( (F_Db *)pDb, 
			uiAttrName, &ui64Num, &bNeg)))
		{
			return( rc);
		}
		
		return( convertToINT64( ui64Num, bNeg, pi64Num));
	}

	FINLINE RCODE XFLMAPI getAttributeValueINT64(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT64 *				pi64Num,
		FLMINT64					i64NotFoundDefault)
	{
		RCODE			rc = NE_XFLM_OK;
		
		if( RC_BAD( rc = getAttributeValueINT64( 
			pDb, uiAttrName, pi64Num)))
		{
			if( rc != NE_XFLM_DOM_NODE_NOT_FOUND)
			{
				return( rc);
			}
			
			*pi64Num = i64NotFoundDefault;
			rc = NE_XFLM_OK;
		}
		
		return( rc);
	}
	
	FINLINE RCODE XFLMAPI getAttributeValueUnicode(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUNICODE *			puzValueBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiCharsReturned = NULL,
		FLMUINT *				puiBufferBytesUsed = NULL)
	{
		return( getAttributeValueText( pDb, uiAttrName, XFLM_UNICODE_TEXT,
			puzValueBuffer, uiBufferSize, puiCharsReturned, puiBufferBytesUsed));
	}

	RCODE XFLMAPI getAttributeValueUnicode(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUNICODE **			ppuzValueBuffer);

	RCODE XFLMAPI getAttributeValueUnicode(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		IF_DynaBuf *			pDynaBuf);
		
	RCODE XFLMAPI getAttributeValueUTF8(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMBYTE *				pucValueBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiCharsReturned = NULL,
		FLMUINT *				puiBufferBytesUsed = NULL)
	{
		return( getAttributeValueText( pDb, uiAttrName, XFLM_UTF8_TEXT,
			pucValueBuffer, uiBufferSize, puiCharsReturned, puiBufferBytesUsed));
	}

	RCODE XFLMAPI getAttributeValueUTF8(
		IF_Db *					pDb,
		FLMUINT					uiAttrNameId,
		FLMBYTE **				ppszValueBuffer);
		
	RCODE XFLMAPI getAttributeValueUTF8(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		IF_DynaBuf *			pDynaBuf);
		
	RCODE XFLMAPI getAttributeValueBinary(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		void *					pvValueBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiValueLength);

	RCODE XFLMAPI getAttributeValueBinary(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		IF_DynaBuf *			pDynaBuf);
		
	FINLINE RCODE XFLMAPI setUINT(
		IF_Db *				pDb,
		FLMUINT				uiValue,
		FLMUINT				uiEncDefId = 0)
	{
		return( setNumber64( pDb, 0, uiValue, uiEncDefId));
	}

	FINLINE RCODE XFLMAPI setUINT64(
		IF_Db *				pDb,
		FLMUINT64			ui64Value,
		FLMUINT				uiEncDefId = 0)
	{
		return( setNumber64( pDb, 0, ui64Value, uiEncDefId));
	}

	FINLINE RCODE XFLMAPI setINT(
		IF_Db *				pDb,
		FLMINT				iValue,
		FLMUINT				uiEncDefId = 0)
	{
		return( setNumber64( pDb, iValue, 0, uiEncDefId));
	}

	FINLINE RCODE XFLMAPI setINT64(
		IF_Db *				pDb,
		FLMINT64				i64Value,
		FLMUINT				uiEncDefId = 0)
	{
		return( setNumber64( pDb, i64Value, 0, uiEncDefId));
	}

	RCODE XFLMAPI setMetaValue(
		IF_Db *					pDb,
		FLMUINT64				ui64Value);

	FINLINE RCODE XFLMAPI setUnicode(
		IF_Db *					pDb,
		const FLMUNICODE *	puzValue,
		FLMUINT					uiValueLength = 0,
		FLMBOOL					bLast = TRUE,
		FLMUINT					uiEncDefId = 0)
	{
		F_Database *		pDatabase = ((F_Db *)pDb)->m_pDatabase;
		
		if( bLast && !pDatabase->m_pPendingInput)
		{
			return( setTextFastPath( (F_Db *)pDb, puzValue, uiValueLength,
				XFLM_UNICODE_TEXT, uiEncDefId));
		}
		else
		{
			return( setTextStreaming( (F_Db *)pDb, puzValue,
				uiValueLength, XFLM_UNICODE_TEXT, bLast, uiEncDefId));
		}
	}

	FINLINE RCODE XFLMAPI setUTF8(
		IF_Db *				pDb,
		const FLMBYTE *	pszValue,
		FLMUINT				uiValueLength = 0,
		FLMBOOL				bLast = TRUE,
		FLMUINT				uiEncDefId = 0)
	{
		F_Database *		pDatabase = ((F_Db *)pDb)->m_pDatabase;
		
		if( bLast && !pDatabase->m_pPendingInput)
		{
			return( setTextFastPath( (F_Db *)pDb, pszValue, uiValueLength,
				XFLM_UTF8_TEXT, uiEncDefId));
		}
		else
		{
			return( setTextStreaming( (F_Db *)pDb, pszValue,
				uiValueLength, XFLM_UTF8_TEXT, bLast, uiEncDefId));
		}
	}

	FINLINE RCODE XFLMAPI setBinary(
		IF_Db *			pDb,
		const void *	pvValue,
		FLMUINT			uiValueLength,
		FLMBOOL			bLast = TRUE,
		FLMUINT			uiEncDefId = 0)
	{
		F_Database *		pDatabase = ((F_Db *)pDb)->m_pDatabase;
		
		if( bLast && !pDatabase->m_pPendingInput)
		{
			return( setBinaryFastPath( (F_Db *)pDb, pvValue, 
				uiValueLength, uiEncDefId));
		}
		else
		{
			return( setBinaryStreaming( (F_Db *)pDb, pvValue, uiValueLength, bLast,
				uiEncDefId));
		}
	}

	FINLINE RCODE XFLMAPI setAttributeValueUINT(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT					uiValue,
		FLMUINT					uiEncDefId = 0)
	{
		return( setAttributeValueNumber( pDb, uiAttrName, 
			0, uiValue, uiEncDefId));
	}

	FINLINE RCODE XFLMAPI setAttributeValueUINT64(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT64				ui64Value,
		FLMUINT					uiEncDefId = 0)
	{
		return( setAttributeValueNumber( pDb, uiAttrName, 
			0, ui64Value, uiEncDefId));
	}

	FINLINE RCODE XFLMAPI setAttributeValueINT(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT					iValue,
		FLMUINT					uiEncDefId = 0)
	{
		return( setAttributeValueNumber( pDb, uiAttrName, 
			iValue, 0, uiEncDefId));
	}

	FINLINE RCODE XFLMAPI setAttributeValueINT64(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT64					i64Value,
		FLMUINT					uiEncDefId = 0)
	{
		return( setAttributeValueNumber( pDb, uiAttrName, 
			i64Value, 0, uiEncDefId));
	}

	RCODE XFLMAPI setAttributeValueUnicode(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		const FLMUNICODE *	puzValue,
		FLMUINT					uiEncDefId = 0);

	RCODE XFLMAPI setAttributeValueUTF8(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		const FLMBYTE *		pszValue,
		FLMUINT					uiLength,
		FLMUINT					uiEncDefId = 0);

	RCODE XFLMAPI setAttributeValueBinary(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		const void *			pvValue,
		FLMUINT					uiLength,
		FLMUINT					uiEncDefId = 0);

	RCODE XFLMAPI getDocumentNode(
		IF_Db *				pDb,
		IF_DOMNode **		ppDocument);
		
	RCODE XFLMAPI getNextDocument(
		IF_Db *				pDb,
		IF_DOMNode **		ppNextDocument);

	RCODE XFLMAPI getPreviousDocument(
		IF_Db *				pDb,
		IF_DOMNode **		ppPrevDocument);

	RCODE XFLMAPI getParentNode(
		IF_Db *				pDb,
		IF_DOMNode **		ppParent);

	RCODE XFLMAPI getFirstChild(
		IF_Db *				pDb,
		IF_DOMNode **		ppFirstChild);

	RCODE XFLMAPI getLastChild(
		IF_Db *				pDb,
		IF_DOMNode **		ppLastChild);

	RCODE XFLMAPI getNextSibling(
		IF_Db *				pDb,
		IF_DOMNode **		ppNextSibling);

	RCODE XFLMAPI getPreviousSibling(
		IF_Db *				pDb,
		IF_DOMNode **		ppPrevSibling);

	RCODE XFLMAPI getChild(
		IF_Db *				pDb,
		eDomNodeType		eNodeType,
		IF_DOMNode **		ppChild);

	RCODE XFLMAPI getChildElement(
		IF_Db *				pDb,
		FLMUINT				uiElementNameId,
		IF_DOMNode **		ppChild,
		FLMUINT				uiFlags = 0);

	RCODE XFLMAPI getSiblingElement(
		IF_Db *				pDb,
		FLMUINT				uiElementNameId,
		FLMBOOL				bNext,
		IF_DOMNode **		ppSibling);

	RCODE XFLMAPI getAncestorElement(
		IF_Db *				pDb,
		FLMUINT				uiElementNameId,
		IF_DOMNode **		ppAncestor);
		
	RCODE XFLMAPI getDescendantElement(
		IF_Db *				pDb,
		FLMUINT				uiElementNameId,
		IF_DOMNode **		ppDescendant);

	RCODE XFLMAPI insertBefore(
		IF_Db *				pDb,
		IF_DOMNode *		pNewChild,
		IF_DOMNode *		pRefChild);

	FINLINE RCODE XFLMAPI getPrefix(
		IF_Db *					pDb,
		FLMUNICODE *			puzPrefixBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiCharsReturned)
	{
		return( getPrefix( TRUE, pDb, (void *)puzPrefixBuffer, uiBufferSize,
			puiCharsReturned));
	}

	FINLINE RCODE XFLMAPI getPrefix(
		IF_Db *					pDb,
		char *					pszPrefixBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiCharsReturned)
	{
		return( getPrefix( FALSE, pDb, (void *)pszPrefixBuffer, uiBufferSize,
			puiCharsReturned));
	}

	RCODE XFLMAPI getPrefixId(
		IF_Db *					pDb,
		FLMUINT *				puiPrefixId);

	FINLINE RCODE XFLMAPI setPrefix(
		IF_Db *					pDb,
		const FLMUNICODE *	puzPrefix)
	{
		return setPrefix( TRUE, pDb, (void *)puzPrefix);
	}

	FINLINE RCODE XFLMAPI setPrefix(
		IF_Db *					pDb,
		const char *			pszPrefix)
	{
		return setPrefix( FALSE, pDb, (void *)pszPrefix);
	}

	RCODE XFLMAPI setPrefixId(
		IF_Db *					pDb,
		FLMUINT					uiPrefixId);

	FINLINE RCODE XFLMAPI getNamespaceURI(
		IF_Db *					pDb,
		FLMUNICODE *			puzNamespaceURIBuffer,
		FLMUINT					uiBufferSize,
		FLMUINT *				puiCharsReturned)
	{
		return getNamespaceURI( TRUE, pDb,
			(void *)puzNamespaceURIBuffer, uiBufferSize, puiCharsReturned);
	}

	FINLINE RCODE XFLMAPI getNamespaceURI(
		IF_Db *				pDb,
		char *				pszNamespaceURIBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT *			puiCharsReturned)
	{
		return getNamespaceURI( FALSE, pDb,
			(void *)pszNamespaceURIBuffer, uiBufferSize, puiCharsReturned);
	}

	FINLINE RCODE XFLMAPI getLocalName(
		IF_Db *				pDb,
		FLMUNICODE *		puzLocalNameBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT *			puiCharsReturned)
	{
		return( getLocalName( TRUE, pDb, (void *)puzLocalNameBuffer,
			uiBufferSize, puiCharsReturned));
	}

	FINLINE RCODE XFLMAPI getLocalName(
		IF_Db *				pDb,
		char *				pszLocalNameBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT *			puiCharsReturned)
	{
		return( getLocalName( FALSE, pDb, (void *)pszLocalNameBuffer, uiBufferSize,
			puiCharsReturned));
	}

	RCODE XFLMAPI getQualifiedName(
		IF_Db *				pDb,
		FLMUNICODE *		puzQualifiedNameBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT *			puiCharsReturned);

	RCODE XFLMAPI getQualifiedName(
		IF_Db *				pDb,
		char *				pszQualifiedNameBuffer,
		FLMUINT				uiBufferSize,
		FLMUINT *			puiCharsReturned);

	FINLINE RCODE XFLMAPI getCollection(
		IF_Db *,				// pDb,
		FLMUINT *			puiCollection)
	{
		*puiCollection = m_pCachedNode->getCollection();
		return( NE_XFLM_OK);
	}

	RCODE XFLMAPI createAnnotation(
		IF_Db *				pDb,
		IF_DOMNode **		ppAnnotation,
		FLMUINT64 *			pui64NodeId = NULL);

	RCODE XFLMAPI getAnnotation(
		IF_Db *				pDb,
		IF_DOMNode **		ppAnnotation);

	RCODE XFLMAPI getAnnotationId(
		IF_Db *				pDb,
		FLMUINT64 *			pui64AnnotationId);
		
	RCODE XFLMAPI hasAnnotation(
		IF_Db *				pDb,
		FLMBOOL *			pbHasAnnotation);
		
	FINLINE RCODE XFLMAPI getIStream(
		IF_Db *				pDb,
		IF_PosIStream **	ppIStream,
		FLMUINT *			puiDataType = NULL,
		FLMUINT *			puiDataLength = NULL)
	{
		return( getIStream( (F_Db *)pDb, NULL, ppIStream, 
			puiDataType, puiDataLength));
	}

	FINLINE RCODE XFLMAPI getTextIStream(
		IF_Db *				pDb,
		IF_PosIStream **	ppIStream,
		FLMUINT *			puiNumChars = NULL)
	{
		return( getTextIStream( (F_Db *)pDb, NULL, ppIStream, puiNumChars));
	}

	FLMUINT XFLMAPI compareNode(
		IF_DOMNode *			pNode,
		IF_Db *					pDb1,
		IF_Db *					pDb2,
		char *					pszErrBuff,
		FLMUINT					uiErrBuffLen);
		
	RCODE XFLMAPI isDataLocalToNode(
		IF_Db *					pDb,
		FLMBOOL *				pbDataIsLocal);
		
	// Public methods that are not part of the exposed public API in the IF_DOMNode interface

	FINLINE FLMBOOL isNamespaceDecl( void)
	{
		if( getModeFlags() & FDOM_NAMESPACE_DECL)
		{
			return( TRUE);
		}

		return( FALSE);
	}
	
	RCODE XFLMAPI setTextFastPath(
		F_Db *				pDb,
		const void *		pvValue,
		FLMUINT				uiNumBytesInBuffer,
		eXFlmTextType		eTextType,
		FLMUINT				uiEncDefId);

	RCODE getNodeId(
		F_Db *				pDb,
		FLMUINT64 *			pui64NodeId,
		FLMUINT *			puiAttrNameId);
		
	FINLINE FLMUINT64 getNodeId( void)
	{
		if( m_uiAttrNameId)
		{
			flmAssert( 0);
			return( 0);
		}
		
		if( m_pCachedNode)
		{
			return( m_pCachedNode->getNodeId());
		}

		return( 0);
	}

	FINLINE FLMUINT64 getIxNodeId( void)
	{
		if( m_pCachedNode)
		{
			return( m_pCachedNode->getNodeId());
		}

		return( 0);
	}

	FINLINE FLMUINT getCollection( void)
	{
		if( m_pCachedNode)
		{
			return( m_pCachedNode->getCollection());
		}

		return( 0);
	}

	RCODE getIStream(
		F_Db *						pDb,
		F_NodeBufferIStream *	pStackStream,
		IF_PosIStream **			ppIStream,
		FLMUINT *					puiDataType = NULL,
		FLMUINT *					puiDataLength = NULL);
	
	RCODE getTextIStream(
		F_Db *						pDb,
		F_NodeBufferIStream *	pStackStream,
		IF_PosIStream **			ppIStream,
		FLMUINT *					puiNumChars = NULL);
		
	FINLINE FLMBOOL isQuarantined( void)
	{
		return( (getModeFlags() & FDOM_QUARANTINED)
				  ? TRUE
				  : FALSE);
	}

	RCODE getAttributeValueNumber(
		F_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMUINT64 *				pui64Num,
		FLMBOOL *				pbNeg);
	
	RCODE getAttributeValueText(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		eXFlmTextType			eTextType,
		void *					pvBuffer,
		FLMUINT					uiBufSize,
		FLMUINT *				puiCharsReturned,
		FLMUINT *				puiBufferBytesUsed);
	
	RCODE setAttributeValueNumber(
		IF_Db *					pDb,
		FLMUINT					uiAttrName,
		FLMINT64					i64Value,
		FLMUINT64				ui64Value,
		FLMUINT					uiEncDefId);
	
	RCODE deleteAttributes(
		F_Db *					pDb,
		FLMUINT					uiAttrToDelete,
		FLMUINT					uiFlags);

private:

	// Methods

	RCODE setTextStreaming(
		F_Db *			pDb,
		const void *	pvValue,
		FLMUINT			uiLength,
		eXFlmTextType	eTextType,
		FLMBOOL			bLast,
		FLMUINT			uiEncDefId = 0);

	RCODE setBinaryStreaming(
		IF_Db *			pDb,
		const void *	pvValue,
		FLMUINT			uiLength,
		FLMBOOL			bLast,
		FLMUINT			uiEncDefId);
		
	RCODE setBinaryFastPath(
		IF_Db *			pDb,
		const void *	pvValue,
		FLMUINT			uiLength,
		FLMUINT			uiEncDefId);
		
	RCODE clearNodeValue(
		F_Db *			pDb);
	
	FINLINE RCODE makeWriteCopy(
		F_Db *	pDb)
	{
		return( gv_XFlmSysData.pNodeCacheMgr->makeWriteCopy( 
			pDb, &m_pCachedNode));
	}

	RCODE canSetValue(
		F_Db *			pDb,
		FLMUINT			uiDataType);

	RCODE isChildTypeValid(
		eDomNodeType	eChildNodeType);

	RCODE isDescendantOf(
		F_Db *			pDb,
		F_DOMNode *		pAncestor,
		FLMBOOL *		pbDescendant);

	RCODE getNumber64(
		F_Db *			pDb,
		FLMUINT64 *		pui64Num,
		FLMBOOL *		pbNeg);

	RCODE storeTextAsNumber(
		F_Db *			pDb,
		void *			pvValue,
		FLMUINT			uiLength,
		FLMUINT			uiEncDefId = 0);
		
	RCODE storeTextAsBinary(
		F_Db *			pDb,
		const void *	pvValue,
		FLMUINT			uiLength,
		FLMUINT			uiEncDefId = 0);
		
	RCODE storeBinaryAsText(
		F_Db *			pDb,
		const void *	pvValue,
		FLMUINT			uiLength,
		FLMUINT			uiEncDefId = 0);
		
	FINLINE RCODE syncFromDb(
		F_Db *			pDb)
	{
		F_CachedNode *		pCachedNode = m_pCachedNode;
	
		if( !pCachedNode)
		{
			return( RC_SET( NE_XFLM_DOM_NODE_DELETED));
		}
		else if( !pCachedNode->nodeLinkedToDatabase())
		{
			return( _syncFromDb( pDb));
		}
		else if( pDb->m_pDatabase != pCachedNode->getDatabase())
		{
			return( RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP));
		}
		else if( pCachedNode->isRightVersion( pDb->m_ui64CurrTransID) &&
			!pCachedNode->nodePurged())
		{
			if( m_uiAttrNameId)
			{
				if( !pCachedNode->m_uiAttrCount || 
					 !pCachedNode->getAttribute( m_uiAttrNameId, NULL))
				{
					return( RC_SET( NE_XFLM_DOM_NODE_DELETED));
				}
			}
			
			return( NE_XFLM_OK);
		}
	
		return( _syncFromDb( pDb));
	}

	RCODE _syncFromDb(
		F_Db *			pDb);

	RCODE unlinkNode(
		F_Db *			pDb,
		FLMUINT			uiFlags);

	RCODE addModeFlags(
		F_Db *			pDb,
		FLMUINT			uiFlags);

	RCODE removeModeFlags(
		F_Db *			pDb,
		FLMUINT			uiFlags);

	FINLINE FLMBOOL canHaveChildren( void)
	{

		if( m_pCachedNode)
		{
			eDomNodeType	eNodeType = getNodeType();

			return( (eNodeType == DOCUMENT_NODE ||
						eNodeType == ELEMENT_NODE)
					? TRUE
					: FALSE);
		}

		return( FALSE);
	}

	RCODE getData(
		F_Db *			pDb,
		FLMBYTE *		pucBuffer,
		FLMUINT *		puiLength);

	void addNodeToMRUList(
		F_DOMNode *		pNode);

	void removeNodeFromMRUList(
		F_DOMNode *		pNode);

	RCODE getNodeFromMRUList(
		F_Db *				pDb,
		eDomNodeType	eNodeType,
		FLMUINT				uiNameId,
		F_DOMNode **		ppNode);

	RCODE getNodeFromMRUListById(
		F_Db *				pDb,
		FLMUINT64			ui64NodeId,
		F_DOMNode **		ppNode);

	RCODE getLocalName(
		FLMBOOL			bUnicode,
		IF_Db *			pDb,
		void *			pvLocalName,
		FLMUINT			uiBufSize,
		FLMUINT *		puiCharsReturned);

	RCODE getPrefix(
		FLMBOOL			bUnicode,
		IF_Db *			pDb,
		void *			pvPrefix,
		FLMUINT			uiBufSize,
		FLMUINT *		puiCharsReturned);

	RCODE setPrefix(
		FLMBOOL			bUnicode,
		IF_Db *			pDb,
		void *			pvPrefix);
	
	FINLINE FLMUINT64 getDocumentId( void)
	{
		if( m_pCachedNode)
		{
			return( m_pCachedNode->getDocumentId());
		}
		
		flmAssert( 0);
		return( 0);
	}
	
	FINLINE FLMBOOL isRootNode( void)
	{
		eDomNodeType	eNodeType = getNodeType();

		if( eNodeType == DOCUMENT_NODE ||
			 eNodeType == ELEMENT_NODE)
		{
			return( m_pCachedNode->isRootNode());
		}

		return( FALSE);
	}

	FINLINE FLMUINT64 getParentId( void)
	{
		if( !m_pCachedNode)
		{
			return( 0);
		}

		if( m_uiAttrNameId)
		{
			return( m_pCachedNode->getNodeId());
		}

		return( m_pCachedNode->getParentId());
	}

	FINLINE void setParentId(
		FLMUINT64		ui64ParentId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setParentId( ui64ParentId);
	}
	
	FINLINE FLMUINT64 getFirstChildId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		if( m_pCachedNode)
		{
			return( m_pCachedNode->getFirstChildId());
		}

		return( 0);
	}

	FINLINE FLMUINT64 getLastChildId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		if( m_pCachedNode)
		{
			return( m_pCachedNode->getLastChildId());
		}

		return( 0);
	}
	
	FINLINE FLMUINT64 getPrevSibId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		if( m_pCachedNode)
		{
			return( m_pCachedNode->getPrevSibId());
		}
		
		return( 0);
	}
	
	FINLINE void setPrevSibId(
		FLMUINT64		ui64PrevSibId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setPrevSibId( ui64PrevSibId);
	}
	
	FINLINE FLMUINT64 getNextSibId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		if( m_pCachedNode)
		{
			return( m_pCachedNode->getNextSibId());
		}
		
		return( 0);
	}
	
	FINLINE void setNextSibId(
		FLMUINT64		ui64NextSibId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setNextSibId( ui64NextSibId);
	}
	
	FINLINE FLMUINT getDataChildCount( void)
	{
		flmAssert( getNodeType() == ELEMENT_NODE);

		if( m_pCachedNode)
		{
			return( m_pCachedNode->getDataChildCount());
		}
		
		return( 0);
	}
	
	FINLINE void setDataChildCount(
		FLMUINT			uiDataChildCount)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() == ELEMENT_NODE);

		m_pCachedNode->setDataChildCount( uiDataChildCount);
	}
	
	FINLINE FLMUINT getModeFlags( void)
	{
		if( m_uiAttrNameId)
		{
			return( m_pCachedNode->getModeFlags( m_uiAttrNameId));
		}
		else if( m_pCachedNode)
		{
			return( m_pCachedNode->getModeFlags());
		}
		
		return( 0);
	}

	RCODE getNamespaceURI(
		FLMBOOL			bUnicode,
		IF_Db *			pDb,
		void *			pvNamespaceURI,
		FLMUINT			uiBufSize,
		FLMUINT *		puiCharsReturned);

	RCODE setNumber64(
		IF_Db *			pDb,
		FLMINT64			i64Value,
		FLMUINT64		ui64Value,
		FLMUINT			uiEncDefId = 0);
		
	RCODE setStorageValue(
		F_Db *			pDb,
		void *			pvValue,
		FLMUINT			uiValueLen,
		FLMUINT			uiEncDefId,
		FLMBOOL			bLast);
		
	FINLINE void setBlkAddr(
		FLMUINT32		ui32BlkAddr)
	{
		m_pCachedNode->setBlkAddr( ui32BlkAddr);
	}
		
	FINLINE void setOffsetIndex(
		FLMUINT		uiOffsetIndex)
	{
		m_pCachedNode->setOffsetIndex( uiOffsetIndex);
	}
	
	FINLINE FLMUINT getOffsetIndex( void)
	{
		return( m_pCachedNode->getOffsetIndex());
	}
	
	FINLINE FLMUINT32 getBlkAddr( void)
	{
		return( m_pCachedNode->getBlkAddr());
	}
	
	FINLINE void unsetNodeDirtyAndNew(
		F_Db *		pDb)
	{
		m_pCachedNode->unsetNodeDirtyAndNew( pDb);
	}
	
	FINLINE FLMUINT getPrefixId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getPrefixId());
	}
	
	FINLINE void setPrefixId(
		FLMUINT			uiPrefixId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		
		m_pCachedNode->setPrefixId( uiPrefixId);
	}
	
	FINLINE FLMUINT getNameId( void)
	{
		if( m_uiAttrNameId)
		{
			return( m_uiAttrNameId);
		}
		
		return( m_pCachedNode->getNameId());
	}
	
	FINLINE void setNameId(
		FLMUINT			uiNameId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		
		m_pCachedNode->setNameId( uiNameId);
	}
	
	FINLINE FLMUINT getDataType( void)
	{
		if( m_uiAttrNameId)
		{
			F_AttrItem *	pAttrItem;

			if( (pAttrItem = m_pCachedNode->getAttribute( 
				m_uiAttrNameId, NULL)) == NULL)
			{
				flmAssert( 0);
				return( XFLM_UNKNOWN_TYPE);
			}
		
			return( pAttrItem->m_uiDataType);
		}

		return( m_pCachedNode->getDataType());
	}
	
	FINLINE void setDataType(
		FLMUINT		uiDataType)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setDataType( uiDataType);
	}
	
	FINLINE FLMUINT getDataLength( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getDataLength());
	}
	
	FINLINE void setDataLength(
		FLMUINT		uiDataLength)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setDataLength( uiDataLength);
	}
	
	FINLINE FLMBYTE * getDataPtr( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getDataPtr());
	}
	
	FINLINE FLMBOOL getQuickNumber64(
		FLMUINT64 *		pui64Num,
		FLMBOOL *		pbNeg)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getQuickNumber64( pui64Num, pbNeg));
	}
	
	FINLINE FLMBOOL nodeIsDirty( void)
	{
		return( m_pCachedNode->nodeIsDirty());
	}

	FINLINE FLMBOOL nodeUncommitted( void)
	{
		return( m_pCachedNode->nodeUncommitted());
	}
	
	FINLINE FLMUINT getStreamUseCount( void)
	{
		return( m_pCachedNode->getStreamUseCount());
	}
	
	FINLINE FLMUINT64 getAnnotationId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getAnnotationId());
	}
	
	FINLINE void setAnnotationId(
		FLMUINT64		ui64AnnotationId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setAnnotationId( ui64AnnotationId);
	}
	
	FINLINE void setLastChildId(
		FLMUINT64		ui64LastChildId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setLastChildId( ui64LastChildId);
	}
	
	FINLINE void setFirstChildId(
		FLMUINT64		ui64FirstChildId)
	{
		flmAssert( nodeUncommitted());
		flmAssert( getNodeType() != ATTRIBUTE_NODE);

		m_pCachedNode->setFirstChildId( ui64FirstChildId);
	}
	
	FINLINE FLMUINT getEncDefId( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getEncDefId());
	}
	
	FINLINE F_Database * getDatabase( void)
	{
		if( m_pCachedNode)
		{
			return( m_pCachedNode->getDatabase());
		}
		
		return( NULL);
	}
	
	FINLINE RCODE openPendingInput(
		F_Db *			pDb,
		FLMUINT			uiNewDataType)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->openPendingInput( pDb, uiNewDataType));
	}
	
	FINLINE FLMBOOL findChildElm(
		FLMUINT		uiChildElmNameId,
		FLMUINT *	puiInsertPos)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->findChildElm( uiChildElmNameId, puiInsertPos));
	}
	
	FINLINE RCODE removeChildElm(
		FLMUINT	uiChildElmOffset)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->removeChildElm( uiChildElmOffset));
	}
	
	FINLINE FLMUINT getChildElmCount( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getChildElmCount());
	}
		
	FINLINE FLMUINT64 getChildElmNodeId(
		FLMUINT	uiChildElmOffset)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getChildElmNodeId( uiChildElmOffset));
	}
	
	FINLINE FLMUINT getChildElmNameId(
		FLMUINT	uiChildElmOffset)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getChildElmNameId( uiChildElmOffset));
	}
	
	FINLINE FLMBOOL hasAttributes( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->hasAttributes());
	}
	
	FINLINE void setFlags(
		FLMUINT	uiFlags)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		m_pCachedNode->setFlags( uiFlags);
	}
	
	FINLINE void unsetFlags(
		FLMUINT	uiFlags)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		m_pCachedNode->unsetFlags( uiFlags);
	}
	
	FINLINE void setEncDefId(
		FLMUINT	uiEncDefId)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		m_pCachedNode->setEncDefId( uiEncDefId);
	}
	
	FINLINE RCODE flushPendingInput(
		F_Db *			pDb,
		FLMBOOL			bLast)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->flushPendingInput( pDb, bLast));
	}
	
	FINLINE FLMUINT getDataBufSize( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getDataBufSize());
	}

	FINLINE RCODE resizeDataBuffer(
		FLMUINT	uiSize,
		FLMBOOL	bMutexAlreadyLocked)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->resizeDataBuffer( uiSize, bMutexAlreadyLocked));
	}
		
	FINLINE RCODE headerToBuf(
		FLMBOOL				bFixedSizeHeader,
		FLMBYTE *			pucBuf,
		FLMUINT *			puiHeaderSize,
		XFLM_NODE_INFO *	pNodeInfo,
		F_Db *				pDb)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->headerToBuf( bFixedSizeHeader, pucBuf,
					puiHeaderSize, pNodeInfo, pDb));
	}
	
	FINLINE FLMINT64 getQuickINT64( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getQuickINT64());
	}
	
	FINLINE FLMUINT64 getQuickUINT64( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getQuickUINT64());
	}
	
	FINLINE void setUINT64(
		FLMUINT64	ui64Value)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		m_pCachedNode->setUINT64( ui64Value);
	}
	
	FINLINE void setINT64(
		FLMINT64		i64Value)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		m_pCachedNode->setINT64( i64Value);
	}
	
	FINLINE FLMUINT64 getMetaValue( void)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		return( m_pCachedNode->getMetaValue());
	}
	
	FINLINE void setMetaValue(
		FLMUINT64		ui64Value)
	{
		flmAssert( getNodeType() != ATTRIBUTE_NODE);
		m_pCachedNode->setMetaValue( ui64Value);
	}
	
	FINLINE RCODE checkAttrList( void)
	{
		if( !m_pCachedNode)
		{
			return( RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND));
		}
		else if( m_pCachedNode->getNodeType() != ELEMENT_NODE)
		{
			return( RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP));
		}
		else if( !m_pCachedNode->m_uiAttrCount)
		{
			return( RC_SET( NE_XFLM_DOM_NODE_NOT_FOUND));
		}
		return( NE_XFLM_OK);
	}

	// Data
	
	F_CachedNode *			m_pCachedNode;
	F_DOMNode *				m_pNextInPool;
	
	// Valid only if this is an attribute node
	
	FLMUINT					m_uiAttrNameId;
	
friend class F_Db;
friend class F_Database;
friend class F_BTreeIStream;
friend class F_NodeBufferIStream;
friend class F_Dict;
friend class F_XMLImport;
friend class F_NodePool;
friend class F_DbRebuild;
friend class F_Query;
friend class FSCollectionCursor;
friend class F_NodeCacheMgr;
friend class F_Rfl;
friend class F_CachedNode;
friend class F_OldNodeList;
friend class F_DbCheck;
friend class F_NodeInfo;
};

/*===========================================================================
Desc: Pool manager for DOM nodes
===========================================================================*/
class F_NodePool : public XF_RefCount, public XF_Base
{
public:

	F_NodePool()
	{
		m_pFirstBTreeIStream = NULL;
		m_hMutex = F_MUTEX_NULL;
	}

	~F_NodePool();

	RCODE setup( void);

	RCODE allocBTreeIStream(
		F_BTreeIStream **	ppBTreeIStream);
		
	void insertBTreeIStream(
		F_BTreeIStream *	pBTreeIStream);
		
private:

	F_BTreeIStream *			m_pFirstBTreeIStream;
	F_MUTEX						m_hMutex;

friend class F_DOMNode;
friend class F_BTreeIStream;
friend class F_NodeBufferIStream;
};

typedef struct
{
	FLMUINT		uiCollectionNum;
	FLMUINT64	ui64HighestNodeIdFound;
	FLMUINT64	ui64HighestNextNodeIdFound;
	FLMUINT		uiNumNextNodeIdsFound;
	FLMUINT		uiEncId;
} COLLECTION_INFO, * COLLECTION_INFO_p;

typedef struct Recov_Dict_Node *	RECOV_DICT_NODE_p;
typedef struct Recov_Dict_Info *	RECOV_DICT_INFO_p;

typedef struct Recov_Dict_Node
{
	RECOV_DICT_NODE_p	pNext;
	F_DOMNode *			pNode;
	FLMUINT64			ui64NodeId;
	FLMUINT				uiElmOffset;
	FLMBOOL				bAdded;
	FLMBOOL				bGotFromDataCollection;
	FLMUINT32			ui32BlkAddress;
} RECOV_DICT_NODE;

typedef struct Recov_Dict_Info
{
	RECOV_DICT_NODE *	pRecovNodes;
	F_Pool				pool;
} RECOV_DICT_INFO;

typedef struct RSIxKeyTag
{
	FLMBYTE			pucRSKeyBuf[ XFLM_MAX_KEY_SIZE];
	FLMUINT			uiRSKeyLen;
	FLMBYTE			pucRSDataBuf[ XFLM_MAX_KEY_SIZE];
	FLMUINT			uiRSDataLen;
} RS_IX_KEY;

/******************************************************************************
Desc:
******************************************************************************/
class F_KeyCollector : public XF_RefCount, public XF_Base
{

public:

	F_KeyCollector(
		F_DbCheck *	pDbCheck)
	{
		m_pDbCheck = pDbCheck;
		m_ui64TotalKeys = 0;
	}

	~F_KeyCollector(){}

	RCODE addKey(
		F_Db *				pDb,
		IXD *					pIxd,
		KREF_ENTRY *		pKref);

	FLMUINT64 getTotalKeys()
	{
		return m_ui64TotalKeys;
	}

private:

	F_DbCheck *				m_pDbCheck;
	FLMUINT64				m_ui64TotalKeys;
	
friend class F_DbCheck;
};


typedef struct
{
	F_DOMNode *		pNode;			// DOM Node to examine to see if it contains
											// dictionary information.
	FLMUINT			uiCollection;	// Collection the node came from.
	FLMUINT64		ui64NodeId;		// NodeId of node.
} CHK_RECORD, * CHK_RECORD_p;

typedef struct
{
	FLMUINT64	ui64BytesUsed;
	FLMUINT64	ui64ElementCount;
	FLMUINT64 	ui64ContElementCount;
	FLMUINT64 	ui64ContElmBytes;
	FLMUINT		uiBlockCount;
	FLMINT		iErrCode;
	FLMUINT		uiNumErrors;
} BLOCK_INFO;

typedef struct
{
	FLMUINT64		ui64KeyCount;
	BLOCK_INFO		BlockInfo;
} LEVEL_INFO;

typedef struct
{
	FLMUINT			uiLfNum;
	eLFileType		eLfType;
	FLMUINT			uiRootBlk;
	FLMUINT			uiNumLevels;
	LEVEL_INFO *	pLevelInfo;
} LF_HDR;

/******************************************************************************
Desc:
******************************************************************************/
typedef struct State_Info
{
	F_Db *				pDb;
	FLMUINT32			ui32BlkAddress;
	FLMUINT32			ui32NextBlkAddr;
	FLMUINT32			ui32PrevBlkAddr;
	FLMUINT32			ui32LastChildAddr;
	FLMUINT				uiElmLastFlag;
	FLMUINT64			ui64KeyCount;
	FLMUINT64			ui64KeyRefs;
	FLMUINT				uiBlkType;
	FLMUINT				uiLevel;
	FLMUINT				uiRootLevel;
	F_COLLECTION *		pCollection;
	FLMUINT				uiElmOffset;
	FLMBYTE *			pucElm;
	FLMUINT				uiElmLen;
	FLMUINT				uiElmKeyLenOffset;	// New
	FLMUINT				uiElmKeyOffset;		// New
	FLMBYTE *			pucElmKey;
	FLMUINT				uiElmKeyLen;
	FLMUINT				uiCurKeyLen;			// Used in Rebuild...
	FLMUINT				uiElmDataLenOffset;	// New
	FLMUINT				uiElmDataOffset;		// uiElmRecOffset;
	FLMUINT				uiElmDataLen;			// uiElmRecLen;
	FLMBYTE *			pucElmData;				// pucElmRec;
	FLMUINT				uiElmCounts;			// New
	FLMUINT				uiElmCountsOffset;	// New
	FLMUINT				uiElmOADataLenOffset;
	FLMUINT				uiElmOADataLen;
	FLMUINT64			ui64ElmNodeId;
	FLMBOOL				bValidKey;
	F_BLK_HDR *			pBlkHdr;
	F_NodeVerifier *	pNodeVerifier;
	BLOCK_INFO			BlkInfo;
	F_BtResultSet *	pNodeRS;
	F_BtResultSet *	pXRefRS;
	FLMUINT				uiCurrLf;
} STATE_INFO;

/******************************************************************************
Desc:
******************************************************************************/
class F_NodeVerifier : public XF_RefCount, public XF_Base
{
public:

	F_NodeVerifier();
	
	~F_NodeVerifier();

	void Reset(
		STATE_INFO *	pStateInfo);

	RCODE AddData(
		FLMUINT64		ui64NodeId,
		void *			pucData,
		FLMUINT			uiDataLen);

	RCODE finalize(
		F_Db *			pDb,
		F_Dict *			pDict,
		FLMUINT			uiCollection,
		FLMUINT64		ui64NodeId,
		FLMBOOL			bSkipDOMLinkCheck,
		FLMINT *			piElmErrCodeRV);

	FINLINE void setupNodeRS(
		F_BtResultSet *	pRS)
	{
		m_pRS = pRS;
	}

	FINLINE void setupXRefRS(
		F_BtResultSet *	pXRefRS)

	{
		m_pXRefRS = pXRefRS;
	}
	
private:

	RCODE verifyNameId(
		F_Db *				pDb,
		eDomNodeType		eNodeType,
		FLMUINT				uiNameId,
		F_NameTable *		pNameTable,
		FLMINT *				piErrCode);

	RCODE verifyPrefixId(
		F_Db *				pDb,
		FLMUINT				uiPrefixId,
		F_NameTable *		pNameTable,
		FLMINT *				piErrCode);

	RCODE checkForIndexes( 
		F_Db *				pDb,
		F_Dict *				pDict,
		FLMUINT				uiCollection);

	FLMBOOL				m_bFinalizeCalled;
	FLMBYTE *			m_pucBuf;
	FLMBYTE				m_ucBuf[ MAX_DOM_HEADER_SIZE];

	FLMUINT				m_uiBufSize;
	FLMUINT				m_uiBytesInBuf;
	FLMUINT				m_uiOverallLength;
	
	F_NODE_INFO			m_nodeInfo;

	F_BtResultSet *	m_pRS;
	F_BtResultSet *	m_pXRefRS;
};

/******************************************************************************
Desc:
******************************************************************************/
typedef struct
{
	F_NODE_INFO			nodeInfo;
	FLMUINT				uiStorageFlags;
	FLMUINT				uiDataBufSize;
	FLMBYTE *			pucData;
	FLMBOOL				bUseFilename;
	FLMBOOL				bGotFromDataCollection;
	FLMBOOL				bProcessed;
	FLMBYTE				ucSortKey;
} REBUILD_NODE_INFO;

/******************************************************************************
Desc:
******************************************************************************/
class RebuildNodeInfoStack : public XF_RefCount, public XF_Base
{
public:

	RebuildNodeInfoStack();
	
	~RebuildNodeInfoStack();

	RCODE setup( void);

	RCODE push(
		REBUILD_NODE_INFO *		pRebuildNodeInfo);

	RCODE pop(
		REBUILD_NODE_INFO *		pRebuildNodeInfo);
	
	void reset( void);

private:

	REBUILD_NODE_INFO *	m_pStack;
	FLMUINT					m_uiStackSize;
	FLMUINT					m_uiNextPos;
	FLMBOOL					m_bSetupCalled;
};

/******************************************************************************
Desc:
******************************************************************************/
typedef struct 
{
	FLMUINT		uiStartOfEntry;
	FLMUINT		uiEndOfEntry;
} BlkStruct;

/******************************************************************************
Desc:	DbCheck object for verifying the condition of the database.
******************************************************************************/
class F_DbCheck : public XF_RefCount, public XF_Base
{

public:
	F_DbCheck( void)
	{
		m_bSkipDOMLinkCheck = FALSE;
		m_pXRefRS = NULL;
		m_pDb = NULL;
		m_pIxd = NULL;
		m_pCollection = NULL;
		m_pLFile = NULL;
		m_pDbInfo = NULL;
		m_pBtPool = NULL;
		m_pResultSetDb = NULL;
		m_pRandGen = NULL;
		m_pDbCheckStatus = NULL;		
		f_memset( m_szResultSetDibName, 0, sizeof( m_szResultSetDibName));
		m_bPhysicalCorrupt = FALSE;
		m_bIndexCorrupt = FALSE;
		m_LastStatusRc = NE_XFLM_OK;
		m_uiFlags = 0;
		m_bStartedUpdateTrans = FALSE;
		m_puiIxArray = NULL;
		m_pIxRSet = NULL;
		m_bGetNextRSKey = FALSE;
		f_memset( &m_IxKey1, 0, sizeof( m_IxKey1));
		f_memset( &m_IxKey2, 0, sizeof( m_IxKey2));
		m_pCurrRSKey = NULL;
		m_pPrevRSKey = NULL;
		m_pBlkEntries = NULL;
		m_uiBlkEntryArraySize = 0;
	}

	~F_DbCheck();

	RCODE dbCheck(
		const char *			pszDbFileName,
		const char *			pszDataDir,
		const char *			pszRflDir,
		const char *			pszPassword,
		FLMUINT					uiFlags,
		IF_DbInfo **			ppDbInfo,
		IF_DbCheckStatus *	pDbCheck);

private:

	FINLINE RCODE chkCallProgFunc( void)
	{
		if (m_pDbCheckStatus && RC_OK( m_LastStatusRc))
		{
			m_LastStatusRc = m_pDbCheckStatus->reportProgress( &m_Progress);
		}
		return( m_LastStatusRc);
	}

	RCODE chkReportError(
		FLMINT			iErrCode,
		FLMUINT			uiErrLocale,
		FLMUINT			uiErrLfNumber,
		FLMUINT			uiErrLfType,
		FLMUINT			uiErrBTreeLevel,
		FLMUINT			uiErrBlkAddress,
		FLMUINT			uiErrParentBlkAddress,
		FLMUINT			uiErrElmOffset,
		FLMUINT64		ui64ErrNodeId);

	FINLINE XFLM_PROGRESS_CHECK_INFO * getProgress( void)
	{
		return( &m_Progress);
	}

	RCODE getBtResultSet(
		F_BtResultSet **	ppBtRSet);
		
	RCODE createAndOpenResultSetDb( void);

	RCODE closeAndDeleteResultSetDb( void);

	RCODE getDictInfo( void);

	RCODE verifyBlkChain(
		BLOCK_INFO *		pBlkInfo,
		FLMUINT				uiLocale,
		FLMUINT				uiFirstBlkAddr,
		FLMUINT				uiBlkType,
		FLMBOOL *			pbStartOverRV);

	RCODE verifyLFHBlocks(
		FLMBOOL *			pbStartOverRV);

	RCODE verifyAvailList(
		FLMBOOL *			pbStartOverRV);

	RCODE blkRead(
		FLMUINT				uiBlkAddress,
		F_BLK_HDR **		ppBlkHdr,
		F_CachedBlock **	ppSCache,
		FLMINT *				piBlkErrCodeRV);

	RCODE verifySubTree(
		STATE_INFO *		pParentState,
		STATE_INFO *		pStateInfo,
		FLMUINT				uiBlkAddress,
		FLMBYTE **			ppucResetKey,
		FLMUINT				uiResetKeyLen,
		FLMUINT64			ui64ResetNodeId);

	RCODE buildIndexKeyList(
		FLMUINT64 *			pui64TotalKeys);

	RCODE verifyBTrees(
		FLMBOOL *			pbStartOverRV);

	RCODE setupLfTable();

	RCODE setupIxInfo( void);

	RCODE getLfInfo(
		LF_HDR *				pLogicalFile,
		LFILE *				pLFile);
		
	RCODE verifyNodePointers(
		STATE_INFO *		pStateInfo,
		FLMINT *				piErrCode);

	RCODE verifyDOChain(
		STATE_INFO *		pParentState,
		FLMUINT				uiBlkAddr,
		FLMINT *				piElmErrCode);

	RCODE chkGetNextRSKey( void);
		
	RCODE verifyIXRSet(
		STATE_INFO *		pStateInfo);

	RCODE resolveIXMissingKey(
		STATE_INFO *		pStateInfo);

	RCODE verifyComponentInDoc(
		ICD *					pIcd,
		FLMUINT				uiComponent,
		F_DataVector *		pKey,
		FLMBOOL *			pbInDoc);
		
	RCODE getKeySource(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL *			pbKeyInDoc,
		FLMBOOL *			pbKeyInIndex);

	RCODE resolveRSetMissingKey(
		STATE_INFO *		pStateInfo);

	RCODE chkVerifyKeyExists(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL *			pbFoundRV);
		
	RCODE addDelKeyRef(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		FLMBOOL				bDelete);

	RCODE reportIxError(
		STATE_INFO *		pStateInfo,
		FLMINT				iErrCode,
		FLMBYTE *			pucErrKey,
		FLMUINT				uiErrKeyLen,
		FLMBOOL *			pbFixErrRV);

	RCODE startUpdate( void);

	RCODE keyToVector(
		FLMBYTE *			pucKey,
		FLMUINT				uiKeyLen,
		IF_DataVector **	ppKeyRV);

	RCODE verifyIXRefs(
		STATE_INFO *	pStateInfo,
		FLMUINT64		ui64ResetNodeId);

	RCODE verifyBlockStructure(
		FLMUINT					uiBlockSize,
		F_BTREE_BLK_HDR *		pBlkHdr);

	RCODE chkEndUpdate( void);
		
	F_Db *							m_pDb;
	F_COLLECTION *					m_pCollection;
	IXD *								m_pIxd;
	LFILE *							m_pLFile;
	F_DbInfo *						m_pDbInfo;
	FLMBOOL							m_bSkipDOMLinkCheck;
	F_BtResultSet *				m_pXRefRS;
	F_BtPool *						m_pBtPool;
	F_RandomGenerator *			m_pRandGen;
	char								m_szResultSetDibName [F_PATH_MAX_SIZE];
	F_Db *							m_pResultSetDb;
	IF_DbCheckStatus *			m_pDbCheckStatus;
	FLMBOOL							m_bPhysicalCorrupt;
	FLMBOOL							m_bIndexCorrupt;
	XFLM_PROGRESS_CHECK_INFO	m_Progress;
	RCODE								m_LastStatusRc;
	FLMUINT							m_uiFlags;
	FLMBOOL							m_bStartedUpdateTrans;
	FLMUINT *						m_puiIxArray;
	F_BtResultSet *				m_pIxRSet;
	FLMBOOL							m_bGetNextRSKey;
	RS_IX_KEY						m_IxKey1;
	RS_IX_KEY						m_IxKey2;
	RS_IX_KEY *						m_pCurrRSKey;
	RS_IX_KEY *						m_pPrevRSKey;
	BlkStruct *						m_pBlkEntries;
	FLMUINT							m_uiBlkEntryArraySize;
friend class F_DbInfo;
friend class F_KeyCollector;
};

// Check Error Codes

/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/
#define FLM_BAD_CHAR							1
#define FLM_BAD_ASIAN_CHAR					2
#define FLM_BAD_CHAR_SET					3
#define FLM_BAD_TEXT_FIELD					4
#define FLM_BAD_NUMBER_FIELD				5
#define FLM_BAD_FIELD_TYPE					6
#define FLM_BAD_IX_DEF						7
#define FLM_MISSING_REQ_KEY_FIELD		8
#define FLM_BAD_TEXT_KEY_COLL_CHAR		9
#define FLM_BAD_TEXT_KEY_CASE_MARKER	10
#define FLM_BAD_NUMBER_KEY					11
#define FLM_BAD_BINARY_KEY					12
#define FLM_BAD_CONTEXT_KEY				13
#define FLM_BAD_KEY_FIELD_TYPE			14
//#define Not_Used_15						15
//#define Not_Used_16						16
//#define Not_Used_17						17
#define FLM_BAD_KEY_LEN						18
#define FLM_BAD_LFH_LIST_PTR				19
#define FLM_BAD_LFH_LIST_END				20
#define FLM_INCOMPLETE_NODE				21	// Not really an error.  Part of a field has been split across entries
#define FLM_BAD_BLK_END						22
#define FLM_KEY_COUNT_MISMATCH			23
#define FLM_REF_COUNT_MISMATCH			24
#define FLM_BAD_CONTAINER_IN_KEY			25
#define FLM_BAD_BLK_HDR_ADDR				26
#define FLM_BAD_BLK_HDR_LEVEL				27
#define FLM_BAD_BLK_HDR_PREV				28
/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/
#define FLM_BAD_BLK_HDR_NEXT				29
#define FLM_BAD_BLK_HDR_TYPE				30
#define FLM_BAD_BLK_HDR_ROOT_BIT			31
#define FLM_BAD_BLK_HDR_BLK_END			32
#define FLM_BAD_BLK_HDR_LF_NUM			33
#define FLM_BAD_AVAIL_LIST_END			34
#define FLM_BAD_PREV_BLK_NEXT				35
#define FLM_BAD_FIRST_ELM_FLAG			36	// NOTE: This is only needed during rebuild
#define FLM_BAD_LAST_ELM_FLAG				37	// NOTE: This is only needed during rebuild
#define FLM_BAD_LEM							38
#define FLM_BAD_ELM_LEN						39
#define FLM_BAD_ELM_KEY_SIZE				40
#define FLM_BAD_ELM_KEY						41
#define FLM_BAD_ELM_KEY_ORDER				42
#define FLM_BAD_ELM_KEY_COMPRESS			43	// NOTE: 5.x keys are not compressed
#define FLM_BAD_CONT_ELM_KEY				44
#define FLM_NON_UNIQUE_FIRST_ELM_KEY	45
#define FLM_BAD_ELM_OFFSET					46
#define FLM_BAD_ELM_INVALID_LEVEL		47
#define FLM_BAD_ELM_FLD_NUM				48
#define FLM_BAD_ELM_FLD_LEN				49
#define FLM_BAD_ELM_FLD_TYPE				50
#define FLM_BAD_ELM_END						51
#define FLM_BAD_PARENT_KEY					52
#define FLM_BAD_ELM_DOMAIN_SEN			53
#define FLM_BAD_ELM_BASE_SEN				54
#define FLM_BAD_ELM_IX_REF					55
#define FLM_BAD_ELM_ONE_RUN_SEN			56
#define FLM_BAD_ELM_DELTA_SEN				57
#define FLM_BAD_ELM_DOMAIN					58
/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/
#define FLM_BAD_LAST_BLK_NEXT				59
#define FLM_BAD_FIELD_PTR					60
#define FLM_REBUILD_REC_EXISTS			61
#define FLM_REBUILD_KEY_NOT_UNIQUE		62
#define FLM_NON_UNIQUE_ELM_KEY_REF		63
#define FLM_OLD_VIEW							64
#define FLM_COULD_NOT_SYNC_BLK			65
#define FLM_IX_REF_REC_NOT_FOUND			66
#define FLM_IX_KEY_NOT_FOUND_IN_REC		67
#define FLM_KEY_NOT_IN_KEY_REFSET		68
#define FLM_BAD_BLK_CHECKSUM				69
#define FLM_BAD_LAST_DRN					70
#define FLM_BAD_FILE_SIZE					71
#define FLM_BAD_FIRST_LAST_ELM_FLAG		72
#define FLM_BAD_DATE_FIELD					73
#define FLM_BAD_TIME_FIELD					74
#define FLM_BAD_TMSTAMP_FIELD				75
#define FLM_BAD_DATE_KEY					76
#define FLM_BAD_TIME_KEY					77
#define FLM_BAD_TMSTAMP_KEY				78
#define FLM_BAD_BLOB_FIELD					79

/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/

#define FLM_BAD_PCODE_IXD_TBL				80
#define FLM_NODE_QUARANTINED				81
#define FLM_BAD_BLK_TYPE					82
#define FLM_BAD_ELEMENT_CHAIN				83
#define FLM_BAD_ELM_EXTRA_DATA			84
#define FLM_BAD_BLOCK_STRUCTURE			85
#define FLM_BAD_ROOT_PARENT				86
#define FLM_BAD_ROOT_LINK					87
#define FLM_BAD_PARENT_LINK				88
#define FLM_BAD_INVALID_ROOT				89
#define FLM_BAD_FIRST_CHILD_LINK			90
#define FLM_BAD_LAST_CHILD_LINK			91
#define FLM_BAD_PREV_SIBLING_LINK		92
#define FLM_BAD_NEXT_SIBLING_LINK		93
#define FLM_BAD_ANNOTATION_LINK			94
#define FLM_UNSUPPORTED_NODE_TYPE		95
#define FLM_BAD_INVALID_NAME_ID			96
#define FLM_BAD_INVALID_PREFIX_ID		97
#define FLM_BAD_DATA_BLOCK_COUNT			98
#define FLM_BAD_AVAIL_SIZE					99
#define FLM_BAD_NODE_TYPE					100
#define FLM_BAD_CHILD_ELM_COUNT			101
#define FLM_NUM_CORRUPT_ERRORS			101

/*
**  WARNING:	ANY CHANGES MADE TO THE CHECK ERROR CODE DEFINES MUST BE
**					REFLECTED IN THE FlmCorruptStrings TABLE FOUND IN FLERRSTR.CPP
*/

class F_DbInfo : public IF_DbInfo, public XF_Base
{
public:

	F_DbInfo()
	{
		m_uiLogicalCorruptions = 0;
		m_uiLogicalRepairs = 0;
		m_ui64FileSize = 0;
		m_uiNumIndexes = 0;
		m_uiNumCollections = 0;
		m_uiNumLogicalFiles = 0;
		m_pLogicalFiles = NULL;
		f_memset( &m_dbHdr, 0, sizeof( m_dbHdr));
		f_memset( &m_AvailBlocks, 0, sizeof( m_AvailBlocks));
		f_memset( &m_LFHBlocks, 0, sizeof( m_LFHBlocks));
	}

	virtual ~F_DbInfo()
	{
		freeLogicalFiles();
	}

	FINLINE void freeLogicalFiles( void)
	{
		FLMUINT	uiLoop;
		
		if (m_pLogicalFiles)
		{
			for (uiLoop = 0; uiLoop < m_uiNumLogicalFiles; uiLoop++)
			{
				if (m_pLogicalFiles [uiLoop].pLevelInfo)
				{
					f_free( &m_pLogicalFiles [uiLoop].pLevelInfo);
				}
			}
			f_free( &m_pLogicalFiles);
		}
		m_uiNumLogicalFiles = 0;
		m_uiNumIndexes = 0;
		m_uiNumCollections = 0;
	}
	
	FINLINE FLMUINT XFLMAPI getNumCollections( void)
	{
		return( m_uiNumCollections);
	}

	FINLINE FLMUINT XFLMAPI getNumIndexes( void)
	{
		return( m_uiNumIndexes);
	}

	FINLINE FLMUINT XFLMAPI getNumLogicalFiles( void)
	{
		return( m_uiNumLogicalFiles);
	}

	FINLINE FLMUINT64 XFLMAPI getFileSize( void)
	{
		return( m_ui64FileSize);
	}

	FINLINE XFLM_DB_HDR * XFLMAPI getDbHdr( void)
	{
		return( &m_dbHdr);
	}

	FINLINE void XFLMAPI getAvailBlockStats(
		FLMUINT64 *		pui64BytesUsed,
		FLMUINT *		puiBlockCount,
		FLMINT *			piLastError,
		FLMUINT *	puiNumErrors)
	{
		*pui64BytesUsed = m_LFHBlocks.ui64BytesUsed;
		*puiBlockCount = m_AvailBlocks.uiBlockCount;
		*piLastError = m_AvailBlocks.iErrCode;
		*puiNumErrors = m_AvailBlocks.uiNumErrors;
	}

	FINLINE void XFLMAPI getLFHBlockStats(
		FLMUINT64 *		pui64BytesUsed,
		FLMUINT *		puiBlockCount,
		FLMINT *			piLastError,
		FLMUINT *	puiNumErrors)
	{
		*pui64BytesUsed = m_LFHBlocks.ui64BytesUsed;
		*puiBlockCount = m_LFHBlocks.uiBlockCount;
		*piLastError = m_LFHBlocks.iErrCode;
		*puiNumErrors = m_LFHBlocks.uiNumErrors;
	}

	void XFLMAPI getBTreeInfo(
		FLMUINT			uiNthLogicalFile,
		FLMUINT *		puiLfNum,
		eLFileType *	peLfType,
		FLMUINT *		puiRootBlkAddress,
		FLMUINT *		puiNumLevels);

	void XFLMAPI getBTreeBlockStats(
		FLMUINT		uiNthLogicalFile,
		FLMUINT			uiLevel,
		FLMUINT64 *		pui64KeyCount,
		FLMUINT64 *		pui64BytesUsed,
		FLMUINT64 *		pui64ElementCount,
		FLMUINT64 *		pui64ContElementCount,
		FLMUINT64 *		pui64ContElmBytes,
		FLMUINT *		puiBlockCount,
		FLMINT *			piLastError,
		FLMUINT *	puiNumErrors);

private:

	FLMUINT							m_uiLogicalCorruptions;
	FLMUINT							m_uiLogicalRepairs;
	FLMUINT64						m_ui64FileSize;
	FLMUINT							m_uiNumIndexes;
	FLMUINT							m_uiNumCollections;
	FLMUINT							m_uiNumLogicalFiles;
	LF_HDR *							m_pLogicalFiles;
	XFLM_DB_HDR						m_dbHdr;
	BLOCK_INFO						m_AvailBlocks;
	BLOCK_INFO						m_LFHBlocks;
friend class F_DbCheck;
};

typedef struct
{
	FLMUINT			uiIndexNum;
	FLMUINT			uiCollection;
	FLMUINT64		ui64DocId;
} DOC_IXD_XREF;

// Verifier Flags

#define CHK_NEXT_SIBLING_VERIFIED		0x0001
#define CHK_PREV_SIBLING_VERIFIED		0x0002
#define CHK_FIRST_CHILD_VERIFIED			0x0004
#define CHK_LAST_CHILD_VERIFIED			0x0008
#define CHK_PARENT_VERIFIED				0x0010
#define CHK_ANNOTATION_VERIFIED			0x0020
#define CHK_ROOT_VERIFIED					0x0040

// Bitmap field order

#define CHK_BM_ROOT_ID						0x0001
#define CHK_BM_PARENT_ID					0x0002
#define CHK_BM_PREV_SIBLING				0x0004
#define CHK_BM_NEXT_SIBLING				0x0008
#define CHK_BM_FIRST_CHILD					0x0010
#define CHK_BM_LAST_CHILD					0x0020
#define CHK_BM_ANNOTATION					0x0040
#define CHK_MAX_BM_ENTRIES					8

typedef struct
{
	FLMUINT16			ui16Flags;				// Verify Flags to record visits/verifies
	FLMUINT16			ui16BitMap;				// Holds a map of present fields
	FLMUINT64			ui64NodeId;
} NODE_RS_HDR;

typedef struct
{
	NODE_RS_HDR			hdr;
	FLMUINT64			ui64FieldArray[ CHK_MAX_BM_ENTRIES];
} NODE_RS_ENTRY;

#define REBUILD_BLK_SIZE				(1024 * 50)
#define REBUILD_RSET_ENTRY_SIZE		21

/*=============================================================================
Desc: Class to rebuild a broken database.  This class is used by
		F_DbSystem::dbRebuild()
=============================================================================*/
class F_DbRebuild : public XF_RefCount, public XF_Base
{
public:

	// Constructor and Destructor

	F_DbRebuild( void)
	{
		m_pDb = NULL;
		m_pSFileHdl = NULL;
	}

	~F_DbRebuild()
	{
	}

	RCODE dbRebuild(
		const char *				pszSourceDbPath,
		const char *				pszSourceDataDir,
		const char *				pszDestDbPath,
		const char *				pszDestDataDir,
		const char *				pszDestRflDir,
		const char *				pDictPath,
		const char *				pszPassword,
		XFLM_CREATE_OPTS *		pCreateOpts,
		FLMUINT64 *					pui64TotNodes,
		FLMUINT64 *					pui64NodesRecov,
		FLMUINT64 *					pui64QuarantinedNodes,
		IF_DbRebuildStatus *		pRebuildStatus);

	FINLINE FLMUINT getBlockSize( void)
	{
		return( m_dbHdr.ui16BlockSize);
	}

	FINLINE RCODE reportStatus(
		FLMBOOL			bForce = FALSE)
	{
		RCODE		rc = NE_XFLM_OK;

		if( m_pRebuildStatus)
		{
			FLMUINT		uiCurrentTime = FLM_GET_TIMER();
			FLMUINT		uiElapTime = FLM_ELAPSED_TIME( uiCurrentTime, m_uiLastStatusTime);

			FLM_TIMER_UNITS_TO_SECS( uiElapTime, uiElapTime);

			if( bForce || uiElapTime >= 1)
			{
				m_uiLastStatusTime = uiCurrentTime;
				m_callbackData.bStartFlag = FALSE;
				if( RC_BAD( rc = m_pRebuildStatus->reportRebuild( &m_callbackData)))
				{
					m_cbrc = rc;
					goto Exit;
				}
			}
		}

	Exit:

		return( rc);
	}

private:

	RCODE rebuildDatabase( void);

	RCODE recoverNodes(
		FLMBOOL				bRecoverDictionary);

	FINLINE FLMBYTE getRSetPrefix(
		FLMUINT		uiNameId)
	{
		if( uiNameId == ELM_PREFIX_TAG)
		{
			return( 1);
		}

		if( uiNameId == ELM_ATTRIBUTE_TAG)
		{
			return( 2);
		}

		if( uiNameId == ELM_ELEMENT_TAG)
		{
			return( 3);
		}

		if( uiNameId == ELM_COLLECTION_TAG)
		{
			return( 4);
		}

		return( 5);
	}

	FINLINE void buildRSetEntry(
		FLMBYTE		ucPrefix,
		FLMUINT		uiCollection,
		FLMUINT64	ui64NodeId,
		FLMUINT		uiBlockAddr,
		FLMUINT		uiElmNumber,
		FLMBYTE *	pucBuffer)
	{
		pucBuffer[ 0] = ucPrefix;
		longToByte( (FLMUINT32)uiCollection, &pucBuffer[ 1]);
		long64ToByte( ui64NodeId, &pucBuffer[ 5]);
		longToByte( (FLMUINT32)uiBlockAddr, &pucBuffer[ 13]);
		longToByte( (FLMUINT32)uiElmNumber, &pucBuffer[ 17]);
	}

	FINLINE void extractRSetEntry(
		FLMBYTE *	pucBuffer,
		FLMUINT *	puiCollection,
		FLMUINT64 *	pui64NodeId,
		FLMUINT *	puiBlockAddr,
		FLMUINT *	puiElmNumber)
	{
		if( puiCollection)
		{
			*puiCollection = byteToLong( &pucBuffer[ 1]);
		}

		if( pui64NodeId)
		{
			*pui64NodeId = byteToLong64( &pucBuffer[ 5]);
		}

		if( puiBlockAddr)
		{
			*puiBlockAddr = byteToLong( &pucBuffer[ 13]);
		}

		if( puiElmNumber)
		{
			*puiElmNumber = byteToLong( &pucBuffer[ 17]);
		}
	}

	RCODE recoverTree(
		F_RebuildNodeIStream *	pIStream,
		FResultSet *				pNonRootRSet,
		F_DOMNode *					pParentNode,
		F_CachedNode *				pRecovCachedNode,
		FLMBYTE *					pucNodeIV);

	FINLINE RCODE reportCorruption(
		FLMINT				iErrCode,
		FLMUINT				uiErrBlkAddress,
		FLMUINT				uiErrElmOffset,
		FLMUINT64			ui64ErrNodeId)
	{
		RCODE		rc;

		if( m_pRebuildStatus)
		{
			m_corruptInfo.iErrCode = iErrCode;
			m_corruptInfo.uiErrBlkAddress = uiErrBlkAddress;
			m_corruptInfo.uiErrElmOffset = uiErrElmOffset;
			m_corruptInfo.ui64ErrNodeId = ui64ErrNodeId;
			rc = m_pRebuildStatus->reportRebuildErr( &m_corruptInfo);
			m_corruptInfo.iErrCode = 0;
			return( rc);
		}
		
		return( NE_XFLM_OK);
	}

	RCODE determineBlkSize(
		FLMUINT *			puiBlkSizeRV);

	F_Db *						m_pDb;
	F_SuperFileHdl *			m_pSFileHdl;
	IF_DbRebuildStatus *		m_pRebuildStatus;
	FLMBOOL						m_bBadHeader;
	FLMUINT						m_uiLastStatusTime;
	XFLM_DB_HDR					m_dbHdr;
	XFLM_CREATE_OPTS			m_createOpts;
	XFLM_REBUILD_INFO			m_callbackData;
	XFLM_CORRUPT_INFO			m_corruptInfo;
	RCODE							m_cbrc;

friend class F_RebuildNodeIStream;
};

RCODE chkBlkRead(
	F_DbInfo *			pDbInfo,
	FLMUINT				uiBlkAddress,
	F_BLK_HDR **		ppBlkHdr,
	F_CachedBlock **	ppSCache,
	FLMINT *				piBlkErrCodeRV);

FLMINT flmCompareKeys(
	FLMBYTE *	pBuf1,
	FLMUINT		uiBuf1Len,
	FLMBYTE *	pBuf2,
	FLMUINT		uiBuf2Len);

void flmInitReadState(
	STATE_INFO *	pStateInfo,
	FLMBOOL *		pbStateInitialized,
	FLMUINT			uiVersionNum,
	F_Db *			pDb,
	LF_HDR *			pLogicalFile,
	FLMUINT			uiLevel,
	FLMUINT			uiBlkType,
	FLMBYTE *		pucKeyBuffer);

FLMINT flmVerifyBlockHeader(
	STATE_INFO *	pStateInfo,
	BLOCK_INFO *	pBlockInfoRV,
	FLMUINT			uiBlockSize,
	FLMUINT			uiExpNextBlkAddr,
	FLMUINT			uiExpPrevBlkAddr,
	FLMBOOL			bCheckEOF);

RCODE flmVerifyElement(
	STATE_INFO *	pStateInfo,
	LFILE *			pLFile,
	IXD *				pIxd,
	FLMINT *			piErrCode);

void getEntryInfo(
	F_BTREE_BLK_HDR *		pBlkHdr,
	FLMUINT					uiOffset,
	FLMBYTE **				ppucElm,
	FLMUINT *				puiElmLen,
	FLMUINT *				puiElmKeyLen,
	FLMUINT *				puiElmDataLen,
	FLMBYTE **				ppucElmKey,
	FLMBYTE **				ppucElmData);
	
FINLINE RCODE F_NodeCacheMgr::allocDOMNode(
	F_DOMNode **		ppDOMNode)
{
	flmAssert( *ppDOMNode == NULL);
	
	if( m_pFirstNode)
	{
		f_resetStackInfo( m_pFirstNode);
		*ppDOMNode = m_pFirstNode;
		m_pFirstNode = m_pFirstNode->m_pNextInPool;
		(*ppDOMNode)->m_pNextInPool = NULL;
	}
	else 
	{
		if( (*ppDOMNode = f_new F_DOMNode) == NULL)
		{
			return( RC_SET( NE_XFLM_MEM));
		}
	}
	
	return( NE_XFLM_OK);
}
		
FINLINE RCODE F_NodeCacheMgr::makeWriteCopy(
	F_Db *				pDb,
	F_CachedNode **	ppCachedNode)
{
	F_CachedNode *		pCachedNode = *ppCachedNode;
	
	if( pCachedNode->getLowTransId() < pDb->m_ui64CurrTransID)
	{
		return( gv_XFlmSysData.pNodeCacheMgr->_makeWriteCopy( 
			pDb, ppCachedNode));
	}
	else if( pCachedNode->getStreamUseCount())
	{
		// The only thread that would be using this version of
		// the node is the updater thread.  It would be illegal
		// for the updater thread to have an input stream going
		// while trying to update this.
		
		return( RC_SET_AND_ASSERT( NE_XFLM_ILLEGAL_OP));
	}
	
	return( NE_XFLM_OK);
}
	
#endif // FLAIMSYS_H
