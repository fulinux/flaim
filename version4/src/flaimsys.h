//-------------------------------------------------------------------------
// Desc:	Internal header file that includes most other header files needed
//			by FLAIM itself.
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
// $Id: flaimsys.h 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef FLAIMSYS_H
#define FLAIMSYS_H

#include "flaim.h"

#if defined( FLM_DEBUG) && !defined( FLM_HPUX)
	#define f_new			new( __FILE__, __LINE__)
#else
	#define f_new			new
#endif

#ifdef FLM_DEBUG
	#define RC_SET( rc) \
		flmMakeErr(rc, __FILE__, __LINE__)

	RCODE	flmMakeErr(
		RCODE				rc,
		const char *	pszFile,
		int				iLine);
#else
	#define RC_SET(rc)		(rc)
#endif

class FResultSet;
class F_Thread;
class HRequest;
class F_Session;
class F_SessionMgr;
class F_XMLImport;
class F_XMLExport;
class F_HashTable;
class F_FileHdlMgr;
class F_FileSystemImp;
class ServerLockManager;
class F_ThreadMgr;
class FCS_ISTM;
class FCS_OSTM;
class FCS_DIS;
class FCS_DOS;
class F_Rfl;
class F_FileIdList;
class F_IOBufferMgr;
class F_IOBuffer;
class ServerLockObject;
class F_SuperFileHdl;
class FlmECache;
class F_FileHdlImp;
class FlmBlobImp;
class F_SlabManager;
class F_FixedAlloc;
class F_BufferAlloc;
class F_CCS;

typedef struct Kref_Entry *			KREF_ENTRY_p;
typedef struct Kref_Cntrl *			KREF_CNTRL_p;
typedef struct F_Event *				FEVENT_p;
typedef struct FBucket *				FBUCKET_p;
typedef struct Itt *						ITT_p;
typedef struct Ixd *						IXD_p;
typedef struct Ifd *						IFD_p;
typedef struct Cdl *						CDL_p;
typedef struct FDict *					FDICT_p;
typedef struct FNotify *				FNOTIFY_p;
typedef struct File_Hdr *				FILE_HDR_p;
typedef struct Log_Hdr *				LOG_HDR_p;
typedef struct FFile *					FFILE_p;
typedef struct FDb *						FDB_p;
typedef struct FlmBlob_Tag *			FBLOB_p;
typedef struct FlmBlobInfo *			FBLOBINFO_p;
typedef struct Din_State *				DIN_STATE_p;
typedef struct Btsk *					BTSK_p;
typedef struct SCache_Mgr *			SCACHE_MGR_p;
typedef struct CP_Info *				CP_INFO_p;
typedef struct RCache_Mgr_Tag *		RCACHE_MGR_p;
typedef struct CS_Context *			CS_CONTEXT_p;
typedef struct F_BkgndIx *				F_BKGND_IX_p;
typedef struct QueryHdrTag *			QUERY_HDR_p;

FLMUINT flmStrHashBucket(
	const char *		pszStr,
	FBUCKET_p			pHashTbl,
	FLMUINT				uiNumBuckets);

#include "ftk.h"
#include "ftkmem.h"
#include "ftknsem.h"
#include "ftksem.h"
#include "ftkwptxt.h"
#include "ftkthrd.h"
#include "fstructs.h"
#include "fddpcode.h"
#include "flist.h"
#include "fmutxref.h"
#include "ffilehdl.h"
#include "flmstat.h"
#include "fbuff.h"
#include "rfl.h"
#include "fsrvlock.h"
#include "filesys.h"
#include "fquery.h"
#include "fscursor.h"
#include "flog.h"
#include "frset.h"
#include "flmimon.h"
#include "fdir.h"
#include "ffilesys.h"
#include "flmstat.h"
#include "fcs.h"
#include "fsv.h"
#include "fxml.h"
#include "furl.h"
#include "ecache.h"
#include "fsuperfl.h"
#include "f64bitfh.h"
#include "fdynsset.h"
#include "frestore.h"
#include "fobjtrck.h"
#include "ftrace.h"
#include "gddiff.h"
#include "flfixed.h"
#include "f_nici.h"

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#if defined( FLM_NLM) && !defined( __MWERKS__)
	// Turn off  "Warning! W007: col(80) "&array" may not produce intended result
	#pragma warning 007 9
	#pragma warning 731 9
#endif

#if defined( FLM_NLM)

	#define os_malloc(size) \
		Alloc( (size), gv_lAllocRTag)

	void * nlm_realloc(
		void	*	pMemory,
		size_t	newSize);

	#define os_realloc		nlm_realloc

	#define os_free			Free

#else

	#define os_malloc			malloc
	#define os_realloc		realloc
	#define os_free			free

#endif

/****************************************************************************
										General FLAIM

This first section contains general FLAIM defines and macros
****************************************************************************/

RCODE f_netwareStartup( void);

void f_netwareShutdown( void);

void f_memoryInit( void);

void f_memoryCleanup( void);

RCODE nssInitialize( void);

void nssUninitialize( void);

#define MIN_BLOCK_SIZE		4096
#define MAX_BLOCK_SIZE		8192

// What are the valid block sizes for a DB. 

#define VALID_BLOCK_SIZE(s) ((s) == 4096 || (s) == 8192)

#define DRN_KEY_SIZ		4 					// size of key buffer for domains

#define MAX_READ_ATTEMPTS	15				// Used with read retries

#define FLM_IS_RIGHT_TRUNCATED_DATA		0x10000
#define FLM_IS_LEFT_TRUNCATED_DATA		0x20000
#define FLM_COMPARE_COLLATED_VALUES		0x40000

// Diagnostic flags and error codes they are set on.

#define FLM_DIAG_INDEX_NUM       0x0001
	// FERR_NOT_UNIQUE - returns index number of non-unique index
#define FLM_DIAG_DRN             0x0002
	// FERR_SYNTAX - dictionary syntax error info, returns dict rec DRN
	// FERR_INVALID_TAG - returns drn of last valid dict record processed
	// FERR_DUPLICATE_DICT_REC - returns drn of record with duplicate ID.
	// FERR_DUPLICATE_DICT_NAME - returns drn of record with duplicate name.
	// FERR_ID_RESERVED - returns drn of reserved ID.
	// FERR_CANNOT_RESERVE_ID - returns drn of ID that cannot be reserved.
	// FERR_CANNOT_RESERVE_NAME - returns drn of name that cannot be reserved.
	// FERR_BAD_DICT_DRN - returns dictionary DRN that was bad.
#define FLM_DIAG_FIELD_NUM       0x0004
	// FERR_SYNTAX - more dict syntax error info, returns field dict field num
	// FERR_BAD_FIELD_NUM - returns invalid field number in record
#define FLM_DIAG_FIELD_TYPE      0x0008
	// Field type for field that was not defined
	// FERR_BAD_FIELD_NUM
#define FLM_DIAG_ENC_ID				0x0010
	// Encryption Id that was not defined
	// FERR_PURGED_ENCDEF_FOUND - Purged encryption definition used

#define FLM_GET_TRANS_FLAGS(uiTransType) (((uiTransType) & 0xF0))

#define FLM_GET_TRANS_TYPE(uiTransType) (((uiTransType) & 0x0F))
		
/****************************************************************************
										GEDCOM Functions

This section contains prototypes, defines, ... that are used within FLAIM's
GEDCOM code.
****************************************************************************/

void GedSmartPoolInit(
	POOL *			pPool,
	POOL_STATS *	pPoolStats); 

// Defines for 'nth' parmeter, GedSibGraft(), GedChildGraft()
#define GED_LAST			32767          // last sib/child
#define GED_FIRST       (-GED_LAST)    // first sib/child

// Defines for treeCnt parameter in GedWalk()
#define GED_TREE        1              // do only this one tree
#define GED_FOREST      0              // do all sibling trees in forest
#define GED_MAXLVLNUM   31             // maximum level number for trees
#define GED_MAXTAGLEN   127            // maximum significant tag length

RCODE GedToTree(
	POOL *			pPool,
	F_FileHdl *		pFileHdl,
	char **			pBuf,
	FLMUINT			uiBufSize,
	NODE **			root,
	F_NameTable *	pNameTable);

void * GedAllocSpace(
	POOL *			pPool,
	NODE *			nd,
	FLMUINT			uiValType,
	FLMUINT			uiLen,
	FLMUINT			uiEncId = 0,
	FLMUINT			uiEncSize = 0);

RCODE GedBinToText(
	FLMBYTE *		bin,
	FLMUINT			uiLen,
	FLMBYTE *		txt,
	FLMUINT 	*		lenRV);

NODE * GedNodeCopy(
	POOL *			pPool,
	NODE *			node,
	NODE *			childList,
	NODE *			sibList );

RCODE GedNodeToBuf(
	FLMUINT			uiLevel,
	NODE *			node,
	void *			arg);

RCODE GedNumToText(
	const FLMBYTE *	num,
	FLMBYTE *			txt,
	FLMUINT 	*			txtLenRV);

RCODE GedTextToBin(
	const FLMBYTE *	txt,
	FLMUINT				len,
	FLMBYTE *			bin,
	FLMUINT  *			lenRV);

RCODE GedTextToNum(
	const FLMBYTE *	txt,
	FLMUINT				len,
	FLMBYTE *			num,
	FLMUINT 	*			lenRV);

void gedSetRecSource(
	NODE *				pNode,
	HFDB					hDb,
	FLMUINT				uiContainer,
	FLMUINT				uiDrn);

RCODE gedCreateSourceNode(
	POOL *				pPool,
	FLMUINT  			uiFieldNum,
	HFDB 					hDb,
	FLMUINT				uiContainer,
	FLMUINT				uiRecId,
	NODE **				ppNode);

NODE * GedChild(
	NODE *      		self);

NODE * GedChildGraft(
	NODE *      		self,
	NODE *      		child,
	FLMINT      		nth);

NODE * GedClip(
	FLMUINT     		uiTreeCount,
	NODE *      		self);

NODE * GedCopy(
	POOL *       		pPool,
	FLMUINT     		uiTreeCount,
	NODE *      		oldTree);

NODE * GedFind(
	FLMUINT     		uiTreeCnt,
	NODE *      		self,
	FLMUINT     		uiTagNum,
	FLMINT      		nth);

NODE * GedPathFind(
	FLMUINT     		treeCnt,
	NODE *      		self,
	FLMUINT *   		tnumPath,
	FLMINT      		nth);

RCODE GedGetBINARY(
	NODE *      		node,
	void *      		binaryRV,
	FLMUINT *			puiBinaryLength);

RCODE GedGetRecPtr(
	NODE *      		node,
	FLMUINT *			puiDrn);

RCODE GedGetNATIVE(
	NODE *      		node,
	char *   			pszBufferRV,
	FLMUINT *			puiLength);

FLMUINT GedGetRecId(
	NODE *      		node);

RCODE GedGetINT(
	NODE *      		node,
	FLMINT *				piNumberRV);

RCODE GedGetINT32(
	NODE *      		node,
	FLMINT32 *			pi32NumberRV);

RCODE GedGetINT16(
	NODE *      		node,
	FLMINT16 *  		pi16NumberRV);

RCODE GedGetUINT(
	NODE *      		node,
	FLMUINT *			puiNumberRV);

RCODE GedGetUINT32(
	NODE *      		node,
	FLMUINT32 *			pui32NumberRV);

RCODE GedGetUINT16(
	NODE *      		node,
	FLMUINT16 *			pui16NumberRV);

RCODE GedGetUINT8(
	NODE *      		node,
	FLMUINT8	*			pui8NumberRV);

NODE * GedParent(
	NODE *      		self);

RCODE GedPutBINARY(
	POOL *      		pPool,
	NODE *      		node,
	const void *      pvData,
	FLMUINT     		uiLength,
	FLMUINT				uiEncId = 0,
	FLMUINT				uiEncSize = 0);

RCODE GedPutRecId(
	POOL *       		pPool,
	NODE **				ppNd,
	FLMUINT     		uiRecId);

RCODE GedPutRecPtr(
	POOL *       		pPool,
	NODE *      		node,
	FLMUINT     		uiDRN,
	FLMUINT				uiEncId = 0,
	FLMUINT				uiEncSize = 0);

RCODE GedPutNATIVE(
	POOL *       		pPool,
	NODE *      		node,
	const char *   	pszNativeStr,
	FLMUINT				uiEncId = 0,
	FLMUINT				uiEncSize = 0);

RCODE GedPutINT(
	POOL *       		pPool,
	NODE *      		nd,
	FLMINT	   		iNumber,
	FLMUINT				uiEncId = 0,
	FLMUINT				uiEncSize = 0);

RCODE GedPutUINT(
	POOL *       		pPool,
	NODE *      		nd,
	FLMUINT				uiNumber,
	FLMUINT				uiEncId = 0,
	FLMUINT				uiEncSize = 0);

RCODE GedGetUNICODE(
	NODE *      		node,
	FLMUNICODE *		pUnicode,
	FLMUINT *			puiBufferLen);

RCODE GedPutUNICODE(
	POOL *       			pPool,
	NODE *      			node,
	const FLMUNICODE *	pUnicode,
	FLMUINT					uiEncId = 0,
	FLMUINT					uiEncSize = 0);

NODE * GedSibGraft(
	NODE *      		self,
	NODE *      		sib,
	FLMINT				nth);

NODE * GedSibNext(
	NODE *      		self);

NODE * GedSibPrev(
	NODE *      		self);

RCODE GedTreeToBuf(
	NODE *				nd,
	char *				pPtr,
	FLMUINT				uiBufferSize,
	F_NameTable *  	pNameTable);

typedef RCODE ( * GEDWALK_FUNC_p )(
	FLMUINT				uiLevel,
	NODE *				pNode,
	void *				pvAppArg);

RCODE GedWalk(
	FLMUINT     		uiTreeCnt,
	NODE *      		self,
	GEDWALK_FUNC_p		pWalkFunc,
	void *      		xyz);

NODE * GedNodeCreate(
	POOL *       		pPool,
	FLMUINT     		uiTagNum,
	FLMUINT     		uiId,
	RCODE *				rcRV);

void * GedValPtr(
	NODE *      		self);

void * GedEncPtr(
	NODE *      		self);

#define GedNodeMake( p, t, r)		GedNodeCreate( p, t, 0, r)
#define GedNodeType( n)				((FLMUINT) ((n)->ui8Type))
#define GedValType( n)           ((FLMUINT) ((n)->ui8Type & 0x0F))
#define GedValLen( n)            ((n)->ui32Length)
#define GedEncLen( n)            ((n)->ui32EncLength)
#define GedSetValLen( n, l)		((n)->ui32Length = (l))
#define GedTagNum( n)            ((FLMUINT) ((n)->ui16TagNum))
#define GedNodeLevel( n)         ((FLMUINT) ((n)->ui8Level))
#define GedNodeNext( n)          ((n)->next)
#define GedNodePrior( n)         ((n)->prior)
#define GedTagNumSet( n, t)      ((n)->ui16TagNum = (FLMUINT16) (t))
#define GedNodeLevelSet( n, l)	((n)->ui8Level = (FLMUINT8) (l))
#define GedNodeLevelAdd( n, al)	((n)->ui8Level += (FLMUINT8) (al))
#define GedNodeLevelSub( n, sl)	((n)->ui8Level -= (FLMUINT8) (sl))
#define GedValTypeSet( n, t)		((n)->ui8Type = (FLMUINT8) (t))
#define GedValTypeSetFlag( n, f)	((n)->ui8Type |= (FLMUINT8) (f))
#define GedNodeTypeSet( n, t)		((n)->ui8Type = (FLMUINT8) (t))
#define GedGetBinaryPtr( n)      (((n) && GedValType((n)) == FLM_BINARY_TYPE) \
												? GedValPtr((n)) : NULL)
#define GedGetBLOBHdrPtr( n)     (((n) && GedValType((n)) == FLM_BLOB_TYPE) \
												? GedValPtr((n)) : NULL)
#define GedGetBINARYPtr( n)      (GedGetBinaryPtr( n))
#define GedIsRightTruncated(n)	((n)->ui8Type & FLM_DATA_RIGHT_TRUNCATED)
#define GedSetRightTruncated(n)	((n)->ui8Type |= FLM_DATA_RIGHT_TRUNCATED)
#define GedIsLeftTruncated(n)		((n)->ui8Type & FLM_DATA_LEFT_TRUNCATED)
#define GedSetLeftTruncated(n)	((n)->ui8Type |= FLM_DATA_LEFT_TRUNCATED)
#define GedIdPtr( n)             ((n) && ((n)->ui8Type & HAS_REC_ID) \
											 ? (void *) (((FLMBYTE *)(n) + sizeof( NODE))) \
											 : (void *) NULL )
#define  GedGetRecId( n)         ((n) && ((n)->ui8Type & HAS_REC_ID)   \
											 ? (*((FLMUINT *)((FLMBYTE *)(n) + sizeof( NODE)))) \
											 : (FLMUINT)0 )
#define GedSetType( n, vType)		((n)->ui8Type = (((n)->ui8Type & \
											(HAS_REC_ID | HAS_REC_SOURCE)) | (FLMUINT8)(vType)))

RCODE GedGetRecSource(
	NODE *      		pNode,      
	HFDB *      		phDb,
	FLMUINT *   		puiContainer,
	FLMUINT *   		puiRecId);

#define FLM_MAX_NIB_CNT		11

RCODE flmBcd2Num(
	FLMUINT				uiValueType,
	FLMUINT				uiValueLength,
	const FLMBYTE *	pucValue,				
	BCD_TYPE  *			bcd);

/****************************************************************************
Desc:		These functions are used to parse GEDCOM buffers/files. They 
			are used by both the OLD NODE GEDCOM and the newer Compact GEDCOM.
****************************************************************************/

#define f_iswhitespace( c)	\
	((c) == ASCII_SPACE || (c) == ASCII_TAB)

void gedSkipBlankLines( 
	GED_STREAM *		pStream);

FLMINT gedNextChar( 
	 GED_STREAM *		pStream);

FLMINT gedReadChar( 
	 GED_STREAM *		pStream, 
	 FLMUINT				uiFilePos);

FLMUINT gedCopyTag( 
	 GED_STREAM *		pStream, 
	 char *				pszDest);

FLMINT gedCopyValue( 
	GED_STREAM *		pStream, 
	char *				pszDest);

#define gedSkipWhiteSpaces( x) \
	for(; f_iswhitespace( x->thisC); gedNextChar( x))

#define KY_CONTEXT_PREFIX		0x1E		// Preceeds each context key
#define KY_CONTEXT_LEN			3			// Length of each context index

FINLINE RCODE KYFlushKeys(
	FDB_p					pDb);

RCODE KrefCntrlCheck(
	FDB *					pDb);

void KrefCntrlFree(
	FDB_p					pDb);

RCODE flmProcessRecFlds(
	FDB *		  			pDb,
	IXD_p					pIxd,
	FLMUINT				uiContainerNum,
	FLMUINT  			uiDrn,
	FlmRecord *			pRecord,
	FLMUINT				uiAction,
	FLMBOOL				bPurgedFldsOk,
	FLMBOOL *			pbHadUniqueKeys);

RCODE flmEncryptField(
	FDICT *				pDict,
	FlmRecord *			pRecord,
	void *				pvField,
	FLMUINT				uiEncId,
	POOL *				pPool);

RCODE flmDecryptField(
	FDICT *				pDict,
	FlmRecord *			pRecord,
	void *				pvField,
	FLMUINT				uiEncId,
	POOL *				pPool);

FLMBOOL flmCheckIfdPath(
	IFD *					pIfd,
	FlmRecord *			pRecord,
	void **				ppPathFlds,
	FLMUINT				uiLeafFieldLevel,
	void *				pvLeafField,
	void **				ppvContextField);

RCODE KYAddToKrefTbl(
	FDB *					pDb,
	IXD_p					pIxd,
	FLMUINT				uiContainer,
	IFD_p					pIfd,
	FLMUINT				uiAction,
	FLMUINT				uiDrn,
	FLMBOOL *			pbHadUniqueKeys,
	const FLMBYTE *	pKey,
	FLMUINT				uiKeyLen,
	FLMBOOL				bAlreadyCollated,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bFldIsEncrypted);

RCODE KYProcessDupKeys(
	FDB *					pDb,
	FLMBOOL				bHadUniqueKeys);

void KYAbortCurrentRecord(
	FDB *					pDb);

RCODE KYKeysCommit(
	FDB *					pDb,
	FLMBOOL				bCommittingTrans);

void KYGetIxAndCdlEntries(
	FDB *					pDb,
	IXD_p					pIxd,
	IFD_p					pIfd,
	FLMUINT * 			puiIxEntryRV,
	FLMUINT * 			puiCdlEntryRV);

RCODE KYCmpKeyAdd2Lst(
	FDB *					pDb,
	IXD_p					pIxd,
	IFD_p					pIfd,
	void *				pFld,
	void *				pRootContext);

RCODE KYBuildCmpKeys(
	FDB *					pDb,
	FLMUINT				uiAction,
	FLMUINT				uiContainerNum,
	FLMUINT				uiDrn,
	FLMBOOL *			pbHadUniqueKeys,
	FlmRecord *			pRecord);

FLMUINT KYCombPostParts(
	FLMBYTE *			pKeyBuf,
	FLMUINT				uiKeyLen,
	FLMBYTE *			pLowUpBuf,
	FLMUINT  			wLuLen,
	FLMUINT				uiIxLang,
	FLMUINT				uiIfdAttr);

RCODE KYValidatePathRelation(
	FlmRecord *			pRecord,
	void *				pCurContext,
	void *				pCurField,
	FLD_CONTEXT *		pFldContext,
	FLMUINT				uiCompoundPos);

RCODE KYVerifyMatchingPaths(
	FlmRecord *			pRecord,
	void *				pCurContext,	
	void *				pCurFld,
	void *				pMatchFld);

RCODE KYTreeToKey(
	FDB *					pDb,
	IXD_p 				pIxd,
	FlmRecord *			pRecord,
	FLMUINT				uiContainerNum,
	FLMBYTE *			pKeyBuf,
	FLMUINT *			pwKeyLenRV,
	FLMUINT				uiFlags);

	// 'uiFlags' valid parameters

	#define KY_HIGH_FLAG       0x01
	#define KY_PATH_CHK_FLAG   0x02

RCODE KYCollateValue(
	FLMBYTE * 			pDest,
	FLMUINT *			puiDestLen,
	const FLMBYTE * 	pSrc,
	FLMUINT 				uiSrcLen,
	FLMUINT				uiIfdFlags,
	FLMUINT				uiLimit,
	FLMUINT *			puiCollationLen,
	FLMUINT *			puiLuLen,
	FLMUINT				uiLang,
	FLMBOOL				bCompoundPiece,
	FLMBOOL				bFirstSubstring,
	FLMBOOL				bInputTruncated,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbOriginalCharsLost = NULL,
	FLMBOOL				bFldIsEncrypted = FALSE);

FLMBOOL KYSubstringParse(
	const FLMBYTE **	pTxt,
	FLMUINT *			puiTxtLen,
	FLMUINT				uiFlags,
	FLMUINT				uiLimit,
	FLMBYTE *			pKeyBuf,
	FLMUINT *			puiKeyLen);

FLMBOOL KYEachWordParse(
	const FLMBYTE **	pTxt,
	FLMUINT *			puiTxtLenRV,
	FLMUINT				uiLimit,
	FLMBYTE *			pKeyBuf,
	FLMUINT *			puiKeyLenRV);

#define UNDF_CHR		0x0000			// Undefined char - ignore for now
#define IGNR_CHR		0x0001			// Ignore this char
#define SDWD_CHR		0x0002			// Space delimited word chr
#define DELI_CHR		0x0040			// Delimiter
#define WDJN_CHR		0x0080			// Word Joining chr ".,/-_"

// Implement later.
#define KATA_CHR		0x0004			// Katakana word chr
#define HANG_CHR		0x0008			// Hangul word chr
#define CJK_CHR		0x0010			// CJK word chr

RCODE flmBuildFromAndUntilKeys(
	IXD_p       		pIxd,
	QPREDICATE ** 		ppQPredicate,
	FLMBYTE *			pFromKey,	
	FLMUINT *			puiFromKeyLen,
	FLMBYTE *			pUntilKey,	
	FLMUINT *			puiUntilKeyLen,
	FLMBOOL *			pbDoRecMatch,	
	FLMBOOL *			pbDoKeyMatch,	
	FLMBOOL *			pbExclusiveUntilKey);

FLMUINT	flmTextGetCharType(
	const FLMBYTE *	pText,
	FLMUINT 				uiLen,
	FLMUINT16 *			pui16WPValue,
	FLMUNICODE *		puzUniValue,
	FLMUINT *			puiType);

FLMUINT	flmTextGetValue(
	const FLMBYTE *	pText,
	FLMUINT 				uiLen,
	FLMUINT *			puiWpChar2,
	FLMUINT				uiFlags,
	FLMUINT16 *			pui16WPValue,	
	FLMUNICODE * 		puzUniValue);

RCODE	krefFree(
	FDB *					pDb,
	FLMBOOL				bFree);

RCODE flmBeginDbTrans(
	FDB *					pDb,
	FLMUINT				uiTransType,
	FLMUINT				uiMaxLockWait,
	FLMUINT				uiFlags = 0,
	FLMBYTE *			pucLogHdr = NULL);

void flmUnlinkDbFromTrans(
	FDB *					pDb,
	FLMBOOL				bCommitting);

void flmLockDbMutex(
	FDB *					pDb);

#define flmLockDbMutex(pDb) \
	f_mutexLock( (pDb)->hShareMutex);

void flmUnlockDbMutex(
	FDB *					pDb);

#define flmUnlockDbMutex(pDb) \
	f_mutexUnlock( (pDb)->hShareMutex);

FLMUINT flmGetDbTransType(
	FDB_p					pDb);

#define flmGetDbTransType( pDb) \
	((pDb)->uiTransType)

RCODE	flmCommitDbTrans(
	FDB *					pDb,
	FLMUINT				uiNewLogicalEOF,
	FLMBOOL				bForceCheckpoint,
	FLMBOOL *			pbEmpty = NULL);

RCODE	flmAbortDbTrans(
	FDB *					pDb,
	FLMBOOL				bOkToLogAbort = TRUE);

FLMBOOL flmCheckBadTrans(
	FDB *					pDb);

#define flmCheckBadTrans(pDb)  \
	(((pDb)->uiTransType != FLM_READ_TRANS && RC_BAD( (pDb)->AbortRc)) \
			? TRUE \
			: FALSE)

void fdbInitCS(
	FDB *					pDb);

RCODE	fdbInit(
	FDB_p					pDb,
	FLMUINT				uiTransType,
	FLMUINT				uiFlags,
	FLMUINT				uiAutoTrans,
	FLMBOOL *			pbStartedTransRV);

// Defines for byFlags parameter in fdbInit

#define FDB_TRANS_GOING_OK				0x01
#define FDB_DONT_RESET_DIAG			0x02
#define FDB_INVISIBLE_TRANS_OK		0x04
#define FDB_CLOSING_OK					0x08

void fdbExit(
	FDB *					pDb);

#if defined( FLM_DEBUG) && (defined( FLM_WIN) || defined( FLM_NLM))
	void fdbUseCheck(
		FDB *				pDb);

	void fdbUnuse(
		FDB *				pDb);
#else
	#define fdbUseCheck( pDb)
	#define fdbUnuse( pDb)
#endif

void flmExit(
	eFlmFuncs			eFlmFuncId,
	FDB *					pDb,
	RCODE					rc);

RCODE flmCheckDatabaseStateImp(
	FDB *					pDb,
	const char *		pszFileName,
	FLMINT				iLineNumber);

RCODE flmCheckFFileStateImp(
	FFILE *				pFile,
	const char *		pszFileName,
	FLMINT				iLineNumber);

#define flmCheckDatabaseState( pDb) \
	flmCheckDatabaseStateImp( pDb, __FILE__, __LINE__)
	
#define flmCheckFFileState( pFile) \
	flmCheckFFileStateImp( pFile, __FILE__, __LINE__)

void flmLogError(
	RCODE					rc,
	const char *		pszDoing,
	const char *		pszFileName = NULL,
	FLMINT				iLineNumber = 0);

void flmLogMessage(
	FlmLogMessageSeverity 	eMsgSeverity,
	FlmColorType  				eForground,
	FlmColorType  				eBackground,
	const char *				pszFormat,
	...);

RCODE fdictGetField(
	FDICT *				pDict,
	FLMUINT				uiFldNum,
	FLMUINT *			puiFieldTypeRV,
	IFD **				ppFirstIfd,
   FLMUINT *			puiFieldStateRV);

RCODE fdictGetEncInfo(
	FDB *					pDb,
	FLMUINT				uiEncId,
	FLMUINT *			puiFieldType,
	FLMUINT *			puiFieldState);

RCODE fdictGetContainer(
	FDICT *				pDict,
	FLMUINT				uiContNum,
	LFILE **				ppLFile);

RCODE fdictGetIndex(
	FDICT *				pDict,
	FLMBOOL				bInLimitedMode,
	FLMUINT				uiIxdId,
	LFILE **				ppLFileRV,
	IXD_p * 				ppIxdRV,
	FLMBOOL				bOfflineOk = FALSE);

RCODE fdictGetNextIXD(
	FDICT *				pDict,
	FLMUINT				uiIxdId,
	IXD_p  *				ppIxdRV);

RCODE  fdictReadLFiles( 
	FDB *					pDb,
	FDICT *				pDict);

RCODE fdictRecUpdate(
	FDB *					pDb,
	LFILE *				pDictContLFile,
	LFILE *				pDictIxLFile,
	FLMUINT *			puiDrnRV,
	FlmRecord *			pNewRecord,
	FlmRecord *			pOldRecord,
	FLMBOOL				bRebuildOp = FALSE);

#define MAX_HOOK_RETRIES			10

void flmResetDiag(
	FDB *					pDb);						

#define flmResetDiag(pDb) \
	((pDb)->Diag.uiInfoFlags = 0)

RCODE flmSendCSOp(
	CS_CONTEXT *		pCSContext,
	FLMUINT				uiClass,
	FLMUINT				uiOp,
	FDB_p					pDb);

RCODE flmUnicodeToAscii(
	FLMUNICODE *		puzString);

LFILE_STATS * fdbGetLFileStatPtr(
	FDB *					pDb,
	LFILE *				pLFile);

RCODE flmIxKeyOutput(
	IXD_p					pIxd,
	FLMBYTE *			pucFromKey,
	FLMUINT				uiKeyLen,
	FlmRecord **		ppKeyRV,
	FLMBOOL				bFullFldPaths);

RCODE flmBuildKeyPaths(
	IFD_p					pIfd,
	FLMUINT				uiFldNum,
	FLMUINT				uiDataType,
	FLMBOOL				bFullFldPaths,
	FlmRecord *			pKey,
	void **				pvField);

void flmUpdEventCallback(
	FDB *					pDb,
	FEventType			eEventType,
	HFDB					hDb,
	RCODE					rc,
	FLMUINT				uiDrn,
	FLMUINT				uiContainer,
	FlmRecord *			pNewRecord,
	FlmRecord *			pOldRecord);

RCODE flmAddRecord(
	FDB *					pDb,
	LFILE *				pLFile,
	FLMUINT *			puiDrnRV,
	FlmRecord *			pRecord,
	FLMBOOL				bBatchProcessing,
	FLMBOOL				bDoInBackground,
	FLMBOOL				bCreateSuspended,
	FLMBOOL				bKeepInCache,
	FLMBOOL *			pbLogCompleteIndexSet);

RCODE	flmDeleteRecord(
	FDB *					pDb,
	LFILE *				pLFile,
	FLMUINT	 			uiDrn,
	FlmRecord **		ppOldRecord,
	FLMBOOL				bMissingKeysOK);

RCODE BlkCheckSum(
	FLMBYTE *			pucBlkPtr,
	FLMINT				Compare,
	FLMUINT				uiBlkAddress,
	FLMUINT				uiBlkSize);

#if defined( FLM_NLM) || defined( FLM_WIN)
	void FastBlockCheckSum(
		void *			pBlk,
		FLMUINT *		puiChecksum,	
		FLMUINT *		puiXORData,
		FLMUINT			uiNumberOfBytes);
	
	void InitFastBlockCheckSum( void);
#endif

#ifdef FLM_WIN
	RCODE MapWinErrorToFlaim(
		DWORD				udErrCode,
		RCODE				defaultRc);
#endif

#define	CHECKSUM_SET				0
#define	CHECKSUM_CHECK				1

#define IsInCSMode(pDb) \
	(FLMBOOL)((((FDB_p)pDb)->pCSContext != NULL) \
											  ? (FLMBOOL)TRUE \
											  : (FLMBOOL)FALSE)

RCODE flmGetCSConnection(
	const char *		pszUrlName,
	CS_CONTEXT_p *		ppCSContextRV);

void flmCloseCSConnection(
	CS_CONTEXT_p *		ppCSContext);

RCODE flmAllocHashTbl(
	FLMUINT        	uiHashTblSize,
	FBUCKET_p  *		ppHashTblRV);

RCODE flmGetTmpDir(
	char *				pszOutputTmpDir);

F_BKGND_IX * flmBackgroundIndexGet(
	FFILE *				pFile,
	FLMUINT				uiValue,
	FLMBOOL				bMutexLocked,
	FLMUINT *			puiThreadId = NULL);

FLMUINT flmBinHashBucket(
	void *				pBuf,
	FLMUINT				uiBufLen,
	FBUCKET_p			pHashTbl,
	FLMUINT				uiNumBuckets);

RCODE flmWaitNotifyReq(
	F_MUTEX				hMutex,
	FNOTIFY_p  *		ppNotifyListRV,
	void *				UserData);

RCODE flmLinkFileToBucket(
	FFILE_p				pFile);
	
void flmLinkFileToNUList(
	FFILE_p				pFile,
	FLMBOOL				bQuickTimeout = FALSE);
	
void flmUnlinkFileFromNUList(
	FFILE_p				pFile);

void flmCheckNUStructs(
	FLMUINT				uiCurrTime);

void flmUnlinkDict(
	FDICT_p				pDict);

void flmReleaseDict(
	FDB *					pDb,
	FDICT *				pDict);

RCODE flmLinkFdbToFile(
	FDB_p					pDb,
	FFILE_p				pFile);
	
void flmUnlinkFdbFromFile(
	FDB_p					pDb);
	
void flmDoEventCallback(
	FEventCategory		eCategory,
	FEventType			eEventType,
	void *				pvEventData1,
	void *				pvEventData2);

void flmSetMustCloseFlags(
	FFILE *				pFile,
	RCODE					rcMustClose,
	FLMBOOL				bMutexLocked);

void flmFreeFile(
	FFILE *  			pFile);

RCODE flmRcaInit(
	FLMUINT				uiMaxRecordCacheBytes);

RCODE flmRcaConfig(
	FLMUINT				uiType,
	void *				Value1,
	void *				Value2);

void flmRcaExit( void);

void flmRcaFindRec(
	FLMUINT				uiContainer,
	FLMUINT				uiDrn,
	FFILE_p				pFile,
	FLMUINT				uiVersionNeeded,
	FLMBOOL				bDontPoisonCache,
	FLMUINT *			puiNumLooks,
	RCACHE **			ppRCache,
	RCACHE **			ppNewerRCache,
	RCACHE **			ppOlderRCache);

RCODE flmRcaRetrieveRec(
	FDB *					pDb,
	FLMBOOL *			pbStartedTrans,
	FLMUINT				uiContainer,
	FLMUINT				uiDrn,
	FLMBOOL				bOkToGetFromDisk,
	BTSK_p				pStack,
	LFILE *				pLFile,
	FlmRecord **		ppRecord);

void flmRcaCleanupCache(
	FLMUINT				uiMaxLockTime,
	FLMBOOL				bMutexesLocked);

void flmRcaReduceCache(
	FLMBOOL				bMutexAlreadyLocked);

RCODE flmRcaInsertRec(
	FDB *					pDb,
	FLMUINT				uiContainer,
	FLMUINT				uiDrn,
	FlmRecord *			pRecord);

RCODE flmRcaRemoveRec(
	FDB *					pDb,
	FLMUINT				uiContainer,
	FLMUINT				uiDrn);

void flmRcaFreeFileRecs(
	FFILE_p				pFile);

void flmRcaAbortTrans(
	FDB *					pDb);

void flmRcaCommitTrans(
	FDB *					pDb);

void flmRcaRemoveContainerRecs(
	FDB *					pDb,
	FLMUINT				uiContainer);

RCODE flmAddField(
	FlmRecord *			pRecord,
	FLMUINT				uiTagNum,
	const void *		pvData,
	FLMUINT				uiDataLen,
	FLMUINT				uiDataType);

RCODE flmModField(
	FlmRecord *			pRecord,
	FLMUINT				uiTagNum,
	const void *		pvData,
	FLMUINT				uiDataLen,
	FLMUINT				uiDataType);

RCODE flmDelField(
	FlmRecord *			pRecord,
	FLMUINT				uiTagNum,
	FLMUINT				uiValue);
	
RCODE flmIncrField(
	FlmRecord *			pRecord,
	FLMUINT				uiTagNum);

RCODE flmDecrField(
	FlmRecord *			pRecord,
	FLMUINT				uiTagNum);

RCODE gedAddField(
	POOL *				pPool,
	NODE *				pRecord,
	FLMUINT				uiTagNum,
	const void *		pvData,
	FLMUINT				uiDataLen,
	FLMUINT				uiDataType);

void flmGetDbBasePath(
	char *				pszBaseDbName,
	const char *		pszDbName,
	FLMUINT *			puiBaseDbNameLen);

RCODE flmOpenFile(
	FFILE *				pFile,
	const char *		pszDbPath,
	const char *		pszDataDir,
	const char  *		pszRflDir,
	FLMUINT				uiOpenFlags,
	FLMBOOL				bInternalOpen,
	F_Restore *			pRestoreObj,
	F_FileHdlImp *		pLockFileHdl,
	const char *		pszPassword,
	FDB_p *				ppDb);

RCODE flmCompleteOpenOrCreate(
	FDB_p *				ppDb,
	RCODE					rc,
	FLMBOOL				bNewFile,
	FLMBOOL				bAllocatedFdb);

RCODE flmOpenOrCreateDbClientServer(
	const char *		pszDbPath,
	const char *		pszDataDir,
	const char *		pszRflDir,
	FLMUINT				uiOpenFlags,
	const char *		pszDictFileName,
	const char *		pszDictBuf,
	CREATE_OPTS *		pCreateOpts,
	FLMBOOL				bOpening,
	CS_CONTEXT *		pCSContext,
	FDB_p *				ppDb);

RCODE flmAllocFdb(
	FDB_p *				ppDb);

RCODE flmNewFileFinish(
	FFILE *				pFile,
	RCODE					OpenRc);

RCODE flmUnregisterFile(
	FFILE_p				pFile	);

RCODE flmVerifyFileUse(
	F_MUTEX				hMutex,
	FFILE_p *			ppFileRV);

RCODE flmCreateLckFile(
	const char *		pszFilePath,
	F_FileHdlImp **	ppLockFileHdlRV);

RCODE flmGetExclAccess(
	const char *		pszFilePath,
	FDB *					pDb);

RCODE flmFindFile(
	const char *		pDbPath,
	const char *		pszDataDir,
	FFILE_p  *			ppFileRV);

RCODE flmAllocFile(
	const char *		pDbPath,
	const char *		pszDataDir,
	const char *		pszDbPassword,
	FFILE_p *			ppFile);

RCODE flmStartCPThread(
	FFILE *				pFile);

RCODE flmStartMaintThread(
	FFILE *				pFile);

RCODE flmDbClose(
	HFDB *				phDbRV,
	FLMBOOL				bMutexLocked);

RCODE flmMaintFreeBlockChain(
	FDB *					pDb,
	FLMUINT				uiTrackerDrn,
	FLMUINT 				uiBlocksToDelete,
	FLMUINT				uiExpectedEndAddr,
	FLMUINT64 *			pui64BlocksFreed);

RCODE flmGetHdrInfo(
	F_SuperFileHdl_p 	pSFileHdl,
	FILE_HDR_p			pFileHdrRV,
	LOG_HDR_p			pLogHdrRV,
	FLMBYTE *			pLogHdr);

void flmGetCreateOpts(
	FILE_HDR *			pFileHdr,
	FLMBYTE *			pucLogHdr,
	CREATE_OPTS *		pCreateOpts);

void flmSetFilePrefix(
	FLMBYTE *			pPrefix,
	FLMUINT				uiMajorVer,
	FLMUINT				uiMinorVer);

FLMUINT flmAdjustBlkSize(
	FLMUINT				uiBlkSize);

void flmInitFileHdrInfo(
	CREATE_OPTS *		pCreateOpts,
	FILE_HDR_p			pFileHdr,
	FLMBYTE *			pFileHdrBuf);

RCODE flmWriteVersionNum(
	F_SuperFileHdl_p 	pSFileHdl,
	FLMUINT				uiVersionNum);

RCODE flmGetFileHdrInfo(
	FLMBYTE *			pPrefixBuf,
	FLMBYTE *			pFileHdrBuf,
	FILE_HDR_p			pFileHdrRV);

RCODE flmReadAndVerifyHdrInfo(
	DB_STATS *			pDbStats,
	F_FileHdl *			pFileHdl,
	FLMBYTE *			pReadBuf,
	FILE_HDR_p			pFileHdrRV,
	LOG_HDR_p			pLogHdrRV,
	FLMBYTE *			pLogHdr);

RCODE flmStartIndexBuild(
	FDB *					pDb,
	FLMUINT				uiIndexNum);

RCODE	flmAddToStopList(
	FDB *					pDb,
	FLMUINT				uiIndexNum);

RCODE	flmAddToStartList(
	FDB *					pDb,
	FLMUINT				uiIndexNum);

void flmIndexingAfterCommit(
	FDB *					pDb);

void flmIndexingAfterAbort(
	FDB *					pDb);

RCODE flmLFileIndexBuild(
	FDB *					pDb,
	LFILE *				pIxLFile,
	IXD_p					pIxd,
	FLMBOOL				bDoInBackground,
	FLMBOOL				bCreateSuspended,
	FLMBOOL *			pbLogCompleteIndexSet);

RCODE flmDbIndexSetOfRecords(
	HFDB					hDb,
	FLMUINT				uiIxNum,
	FLMUINT				uiContainerNum,
	FLMUINT				uiStartDrn,
	FLMUINT 				uiEndDrn);

RCODE flmSetIxTrackerInfo(
	FDB *					pDb, 
	FLMUINT				uiIndexNum,
	FLMUINT				uiLastContainerIndexed,
	FLMUINT				uiLastDrnIndexed,
	FLMUINT				uiOnlineTransId,
	FLMBOOL 				bSuspended);

RCODE flmGetIxTrackerInfo(
	FDB *					pDb, 
	FLMUINT				uiIndexNum,
	FLMUINT *			puiLastContainerIndexed,
	FLMUINT *			puiLastDrnIndexed,
	FLMUINT *			puiOnlineTransId,
	FLMBOOL *			pbSuspended);

RCODE flmStartBackgrndIxThrds(
	FDB *					pDb);

#define FLM_BACKGROUND_LOCK_PRIORITY			-100

void flmLogIndexingProgress(
	FLMUINT				uiIndexNum,
	FLMUINT				uiLastDrn);

RCODE flmIndexSetOfRecords(
	FDB *					pDb,
	FLMUINT				uiIxNum,
	FLMUINT				uiContainerNum,
	FLMUINT				uiStartDrn,
	FLMUINT				uiEndDrn,
	STATUS_HOOK			fnStatus,
	void *				StatusData,
	IX_CALLBACK			fnIxCallback,
	void *				IxCallbackData,
	FINDEX_STATUS *	pIndexStatus,
	FLMBOOL *			pbHitEnd,
	F_Thread *			pThread = NULL,
	FlmRecord *			pReusableRec = NULL);

RCODE flmRemoveContainerKeys(
	FDB *					pDb,
	FLMUINT				uiIndexNum,
	FLMUINT				uiContainerNum);

RCODE flmCheckDictFldRefs(
	FDICT *				pDict,
	FLMUINT				uiFieldNum);

RCODE flmCheckDictEncDefRefs(
	FDICT *				pDict,
	FLMUINT				uiEncId);

RCODE flmChangeItemState(
	FDB *					pDb,
	FLMUINT				uiItemId,
	FLMUINT				uiNewState);

RCODE flmBlobPlaceInTransactionList(
	FDB *					pDb,
	FLMUINT				uiAction,
	FlmRecord * 		pRecord,	
	void *				pvBlobField);

RCODE FB_OperationEnd(
	FDB *       		pDb,
	RCODE					rcOfOperation);

void FBListAfterCommit(
	FDB *       		pDb);

void FBListAfterAbort(
	FDB *       		pDb);

const char * flmErrorString(
	RCODE					rc);

RCODE tokenGetUnicode(
	const char *		pszToken,
	void **				ppvVal,
	FLMUINT *			puiValLen,
	FLMUINT *			puiValBufSize);

RCODE expImpInit(
	F_FileHdl *			pFileHdl,
	FLMUINT				uiFlag,
	EXP_IMP_INFO_p		pExpImpInfoRV);

	#define	EXPIMP_IMPORT_DICTIONARY		1
	#define	EXPIMP_EXPORT_DICTIONARY		2
	#define	EXPIMP_IMPORT_EXPORT_GEDCOM	3

void expImpFree(
	EXP_IMP_INFO_p		pExpImpInfo);

RCODE expFlush(
	EXP_IMP_INFO_p		pExpImpInfo);

RCODE expImpSeek(
	EXP_IMP_INFO_p		pExpImpInfo,
	FLMUINT				uiSeekPos);

RCODE expWriteRec(
	EXP_IMP_INFO_p		pExpImpInfo,
	FlmRecord *			pRecord,
	FLMUINT				uiDrn);

RCODE impReadRec(
	EXP_IMP_INFO_p		pExpImpInfo,
	FlmRecord **		ppRecordRV);

RCODE impFileIsExpImp(
	F_FileHdl *			pFileHdl,
	FLMBOOL *			pbFileIsExpImpRV);

#ifdef FLM_DBG_LOG

	void scaLogWrite(
		FLMUINT			uiFFileId,
		FLMUINT			uiWriteAddress,
		FLMBYTE *		pucBlkBuf,
		FLMUINT			uiBufferLen,
		FLMUINT			uiBlockSize,
		char *			pszEvent);
	
	void flmDbgLogWrite(
		FLMUINT			uiFileId,
		FLMUINT			uiBlkAddress,
		FLMUINT			uiWriteAddress,
		FLMUINT			uiTransId,
		char *			pszEvent);
	
	void flmDbgLogUpdate(
		FLMUINT			uiFileId,
		FLMUINT			uiTransId,
		FLMUINT			uiContainer,
		FLMUINT			uiDrn,
		RCODE				rc,
		char *			pszEvent);
	
	void flmDbgLogMsg(
		char *			pszMsg);
	
	void flmDbgLogFlush( void);
	
	void flmDbgLogInit( void);
	
	void flmDbgLogExit( void);

#endif	// #ifdef FLM_DBG_LOG

void flmDeleteCCSRefs(
	FDICT *		pDict);

/****************************************************************************
Desc:	Extracts the information from a log header.
****************************************************************************/
FINLINE void flmGetLogHdrInfo(
	FLMBYTE *			pucLogHdr,
	LOG_HDR_p			pLogHdr)
{
	pLogHdr->uiCurrTransID =
		(FLMUINT)FB2UD( &pucLogHdr [LOG_CURR_TRANS_ID]);
	pLogHdr->uiLogicalEOF =
		(FLMUINT)FB2UD( &pucLogHdr [LOG_LOGICAL_EOF]);

	// If we are doing a read transaction, these two are only
	// needed when checking the database.

	pLogHdr->uiFirstAvailBlkAddr =
		(FLMUINT)FB2UD( &pucLogHdr [LOG_PF_AVAIL_BLKS]);
	pLogHdr->uiAvailBlkCount =
		(FLMUINT)FB2UD( &pucLogHdr [LOG_PF_NUM_AVAIL_BLKS]);
}

/****************************************************************************
Desc: Get the length (in bytes) of the container part that will be
		tacked onto a collated key for storing container number.
****************************************************************************/
FINLINE FLMUINT getIxContainerPartLen(
	IXD *					pIxd)
{
	if (pIxd->uiFlags & IXD_HAS_POST)
	{
		return( 2);
	}
	else
	{
		IFD *	pLastIfd = &pIxd->pFirstIfd [pIxd->uiNumFlds];

		return( (pIxd->uiLanguage < FIRST_DBCS_LANG ||
					pIxd->uiLanguage > LAST_DBCS_LANG ||
					IFD_GET_FIELD_TYPE( pLastIfd) != FLM_TEXT_TYPE ||
					(pLastIfd->uiFlags & IFD_CONTEXT))
					? 3
					: 4);
	}
}

/****************************************************************************
Desc: Append the container number onto the key with appropriate
		separator.
****************************************************************************/
FINLINE void appendContainerToKey(
	IXD *					pIxd,
	FLMUINT				uiContainerNum,
	FLMBYTE *			pucKey,
	FLMUINT *			puiKeyLen)
{
	FLMUINT		uiContainerPartLen = getIxContainerPartLen( pIxd);

	// If container part length is 2, we are storing container
	// immediately after the last thing in the key with no
	// separator.  This is the case where the index is a post
	// index.  In this case, the post marker (0x01) will precede the
	// container somewhere in the key.  If container part length
	// is 3, we are storing a single byte zero separator after the
	// last key component before the container number.  This is for
	// indexes that are not post indexes and where the last field in
	// the index is either not text or the index is not asian.
	// If container part length is 4, we are storing a two byte zero
	// separator after the last key component before the container
	// number.  This is for indexes that are not post indexes and where
	// the last field in the index is text and the index is Asian.

	if (uiContainerPartLen > 2)
	{
		pucKey [(*puiKeyLen)++] = 0;
		if (uiContainerPartLen > 3)
		{
			pucKey[(*puiKeyLen)++] = 0;
		}
	}

	// Output in big-endian order - for sorting purposes only.
	// It isn't really important, except that people probably
	// expect the lower container numbers to show up first
	// in the index.

	pucKey [(*puiKeyLen)++] = (FLMBYTE)(uiContainerNum >> 8);
	pucKey [(*puiKeyLen)++] = (FLMBYTE)(uiContainerNum);
}

/****************************************************************************
Desc: Append the container number onto the key with appropriate
		separator.
****************************************************************************/
FINLINE FLMUINT getContainerFromKey(
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen)
{
	// Container number is always the last two bytes of the
	// key - in big-endian order - see above.

	return( (FLMUINT)(pucKey [uiKeyLen - 1]) +
			  ((FLMUINT)(pucKey [uiKeyLen - 2]) << 8));
}

/****************************************************************************
Desc: Increment a FLMUINT inside a portable buffer.
****************************************************************************/
FINLINE void flmIncrUint(
	FLMBYTE *			pucUint,
	FLMUINT				uiIncrVal)
{
	FLMUINT	uiTmp = (FLMUINT)FB2UD( pucUint) + uiIncrVal;
	UD2FBA( (FLMUINT32)uiTmp, pucUint);
}

/****************************************************************************
Desc: Decrement a FLMUINT inside a portable buffer.
****************************************************************************/
FINLINE void flmDecrUint(
	FLMBYTE *			pucUint,
	FLMUINT				uiDecrVal)
{
	FLMUINT	uiTmp = (FLMUINT)FB2UD( pucUint) - uiDecrVal;
	UD2FBA( (FLMUINT32)uiTmp, pucUint);
}

/****************************************************************************
Desc: This routine links an FDICT structure to its FFILE structure.
NOTE: This routine assumes that the global mutex is locked.
****************************************************************************/
FINLINE void flmLinkDictToFile(
	FFILE *     		pFile,
	FDICT *				pDict)
{
	if( (pDict->pNext = pFile->pDictList) != NULL)
	{
		pDict->uiDictSeq = pDict->pNext->uiDictSeq + 1;
		pDict->pNext->pPrev = pDict;
	}
	else
	{
		pDict->uiDictSeq = 1;
	}
	pFile->pDictList = pDict;
	pDict->pFile = pFile;
}

/****************************************************************************
Desc: This routine unlinks an FDB structure from its FDICT structure.
NOTE: This routine assumes that the global semaphore has been
		locked.
****************************************************************************/
FINLINE void flmUnlinkFdbFromDict(
	FDB *    			pDb)
{
	FDICT * pDict;

	if( (pDict = pDb->pDict) != NULL)
	{
		// If the use count goes to zero and the FDICT is not the first one
		// in the file's list or it is not linked to a file, unlink the FDICT
		// structure from its file and delete it.
		
		if ((!(--pDict->uiUseCount)) &&
		 	((pDict->pPrev) || (!pDict->pFile)))
		{
			flmUnlinkDict( pDict);
		}
		
		pDb->pDict = NULL;
	}
}

/****************************************************************************
Desc: This routine links an FDB structure to an FDICT structure.
NOTE: This routine assumes that the global semaphore has been
		locked if necessary.
****************************************************************************/
FINLINE void flmLinkFdbToDict(
	FDB *       		pDb,
	FDICT *				pDict)
{
	if( pDict != pDb->pDict)
	{
		if( pDb->pDict)
		{
			flmUnlinkFdbFromDict( pDb);
		}
		
		if( (pDb->pDict = pDict) != NULL)
		{
			pDict->uiUseCount++;
		}
	}
}

/****************************************************************************
Desc:	This routine compares two key values to see if they are equal.
****************************************************************************/
FINLINE FLMUINT KYKeyCompare(
	FLMBYTE *			pKey1,
	FLMUINT				wLen1,
	FLMBYTE *			pKey2,
	FLMUINT				wLen2)
{
	FLMUINT	wCmpLen = (wLen2 <= wLen1) ? wLen2 : wLen1;

	while( wCmpLen)
	{
		if (*pKey1 == *pKey2)
		{
			pKey1++;
			pKey2++;
			wCmpLen--;
		}
		else
			return( (*pKey1 < *pKey2) ? BT_LT_KEY : BT_GT_KEY);
	}

	return( (wLen1 < wLen2)
			  ? (BT_LT_KEY)
			  : ((wLen1 > wLen2)
				  ? (BT_GT_KEY)
				  : (BT_EQ_KEY)));
}

/****************************************************************************
Desc:	Flush all keys in KREF table, if any.  This is called by routines
		that are reading an index to ensure that all keys have been
		flushed to the index.
****************************************************************************/
FINLINE RCODE KYFlushKeys(
	FDB_p					pDb)
{
	RCODE	rc = FERR_OK;

	if (pDb->KrefCntrl.bKrefSetup )
	{
		if (RC_OK( rc = KYKeysCommit( pDb, FALSE)))
		{
			pDb->KrefCntrl.pReset = GedPoolMark( pDb->KrefCntrl.pPool);
		}
	}
	return( rc);
}

/****************************************************************************
Desc: This routine gets the path for a FFILE structure.
****************************************************************************/
FINLINE RCODE flmGetFilePath(
	FFILE *				pFile,
	char *				pszPathRV)
{
	f_strcpy( pszPathRV, pFile->pszDbPath);
	return( FERR_OK);
}

/****************************************************************************
Desc: This routine frees a dictionary and its associated tables.
****************************************************************************/
FINLINE void flmFreeDict(
	FDICT *    			pDict)
{
	f_free( (void **)&pDict->pLFileTbl);
	
	// Delete any F_CCS referenced objects from the ITT table before
	// we get rid of the ITT table
	
	flmDeleteCCSRefs( pDict);
	f_free( (void **)&pDict->pIttTbl);
	f_free( (void **)&pDict->pIxdTbl);
	f_free( (void **)&pDict->pIfdTbl);
	f_free( (void **)&pDict->pFldPathsTbl);
	f_free( (void **)&pDict);
}

/****************************************************************************
Desc:	This routine allocates a new FDICT structure and initializes it.
****************************************************************************/
FINLINE RCODE flmAllocDict(
	FDICT **				ppDictRV)
{
	RCODE					rc;
	FDICT *				pDict;

	if( RC_OK( rc = f_calloc( (FLMUINT)sizeof( FDICT), &pDict)))
	{
		pDict->uiUseCount++;
	}

	*ppDictRV = pDict;
	return( rc);
}

/****************************************************************************
Desc:	Outputs an update event callback.
****************************************************************************/
FINLINE void flmTransEventCallback(
	FEventType			eEventType,
	HFDB					hDb,
	RCODE					rc,
	FLMUINT				uiTransId)
{
	FLM_TRANS_EVENT	TransEvent;

	TransEvent.uiThreadId = f_threadId();
	TransEvent.hDb = hDb;
	TransEvent.uiTransID = uiTransId;
	TransEvent.rc = rc;
	flmDoEventCallback( F_EVENT_UPDATES, eEventType, &TransEvent, NULL);
}

/****************************************************************************
Desc:	Commit or abort an auto transaction.
		An abort will be performed if the rc passed in is bad.
****************************************************************************/
FINLINE RCODE flmEndAutoTrans(
	FDB *					pDb,
	RCODE					rc)
{
	if( RC_OK( rc))
	{
		rc = flmCommitDbTrans( pDb, 0, FALSE);
	}
	else
	{
		// Don't change the current bad return code.
		
		(void) flmAbortDbTrans( pDb);
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Determines if the passed-in version is supported by this
		code base.
****************************************************************************/
FINLINE RCODE flmCheckVersionNum(
	FLMUINT				uiVersionNum)
{
	RCODE			rc = FERR_OK;

	switch( uiVersionNum)
	{
		case FLM_VER_3_0:
		case FLM_VER_3_02:
		case FLM_VER_3_10:
		case FLM_VER_4_0:
		case FLM_VER_4_3:
		case FLM_VER_4_31:
		case FLM_VER_4_50:
		case FLM_VER_4_51:
		case FLM_VER_4_52:
		case FLM_VER_4_60:
			break;
		default:
			if( uiVersionNum > FLM_CURRENT_VERSION_NUM)
			{
				rc = RC_SET( FERR_NEWER_FLAIM);
			}
			else
			{
				rc = RC_SET( FERR_UNSUPPORTED_VERSION);
			}
			break;
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns a pointer to the correct entry in the SCache hash table for
		the given block address
****************************************************************************/
FINLINE SCACHE ** ScaHash(
	FLMUINT				uiSigBitsInBlkSize,
	FLMUINT				uiBlkAddress)
{
	return (SCACHE **)&gv_FlmSysData.SCacheMgr.ppHashTbl[
		(((uiBlkAddress) >> uiSigBitsInBlkSize) & 
			gv_FlmSysData.SCacheMgr.uiHashTblBits)];
}

/****************************************************************************
Desc:	Gets the transaction ID from the block header.  NOTE: This function
		assumes that the global mutex is locked.
****************************************************************************/
FINLINE FLMUINT scaGetLowTransID(
	SCACHE *				pSCache)
{
	return( FB2UD( &pSCache->pucBlk [BH_TRANS_ID]));
}

/****************************************************************************
Desc: This function should be called before EACH user defined callback is made.
		CB_ENTER & CB_EXIT protect FLAIM from other FLAIM calls that maybe made
		within a user's callback.
****************************************************************************/
FINLINE void CB_ENTER(	
	FDB *					pDb,
	FLMBOOL *			pbSavedInvisTrans)
{
	// Increment the In Flaim Function variable. 
	// This variable is used to prevent the early release of FLAIM temp pool

 	pDb->uiInFlmFunc++;
	*pbSavedInvisTrans = FALSE;

	// Check for the existences of a Invisible transaction

	if( pDb->uiFlags & FDB_INVISIBLE_TRANS)
	{
		// This action is taken to prevent a the pDb's transaction from being
		// closed by the user calling a trans commit/abort within their callback.
		
		pDb->uiFlags &= ~FDB_INVISIBLE_TRANS;
		*pbSavedInvisTrans = TRUE;
	}
}

/****************************************************************************
Desc: This function should be called after EACH user defined callback is made.
****************************************************************************/
FINLINE void CB_EXIT(	
	FDB *					pDb,
	FLMBOOL				bSavedInvisTrans)
{
	pDb->uiInFlmFunc--;

	if( bSavedInvisTrans)
	{
		pDb->uiFlags |= FDB_INVISIBLE_TRANS;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void KYFinishCurrentRecord(
	FDB *					pDb)
{
	pDb->KrefCntrl.uiLastRecEnd = pDb->KrefCntrl.uiCount;
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE FlmGetTransStatus(
	HFDB				hDb)
{
	RCODE				rc = FERR_OK;
	FDB *				pDb = (FDB *)hDb;

	// See if the database is being forced to close

	if( RC_BAD( rc = flmCheckDatabaseState( pDb)))
	{
		goto Exit;
	}

	rc = ((FDB *)hDb)->AbortRc;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
typedef struct RSIxKeyTag
{
	FLMUINT			uiRSIxNum;
		#define RS_KEY_OVERHEAD				6
		#define RS_IX_OFFSET					0
		#define RS_REF_OFFSET				2
		#define RS_KEY_OFFSET				6
	FLMBYTE			pucRSKeyBuf[ MAX_KEY_SIZ + RS_KEY_OVERHEAD + 2];
	FLMUINT			uiRSKeyLen;
	FLMUINT			uiRSRefDrn;
} RS_IX_KEY;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	LFILE *			pLFile;
	IXD *				pIxd;
	IFD *				pIfd;
	LF_STATS *		pLfStats;
} LF_HDR;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct Db_Info
{
	FILE_HDR						FileHdr;
	LF_HDR *						pLogicalFiles;
	DB_CHECK_PROGRESS *		pProgress;
	STATUS_HOOK					fnStatusFunc;
	RCODE							LastStatusRc;
	FLMBOOL						bReposition;
	FDB *							pDb;
	FLMBOOL						bDbInitialized;
	F_SuperFileHdl *			pSFileHdl;
	FLMUINT						uiFlags;
	FLMUINT						uiMaxLockWait;
	FLMBOOL						bStartedUpdateTrans;
} DB_INFO;

/****************************************************************************
Desc:
****************************************************************************/
typedef struct
{
	POOL					pool;
	FLMUINT				uiIxCount;
	FLMUINT *			puiIxArray;
	void *				pRSet;
	FLMBOOL				bGetNextRSKey;
	RS_IX_KEY			IxKey1;
	RS_IX_KEY			IxKey2;
	RS_IX_KEY *			pCurrRSKey;
	RS_IX_KEY *			pPrevRSKey;
	FLMBOOL				bCheckCounts;
	FLMUINT				uiRSIxKeyCount;
	FLMUINT				uiRSIxRefCount;
	FLMUINT				uiFlags;
	DB_INFO *			pDbInfo;

} IX_CHK_INFO;

typedef struct State_Info
{
	FLMUINT				uiVersionNum;
	FDB *					pDb;
	LF_HDR *				pLogicalFile;
	FLMUINT				uiLevel;
	FLMUINT				uiBlkType;
	FLMUINT				uiLastChildAddr;
	BLOCK_INFO			BlkInfo;
	FLMUINT				uiNextBlkAddr;
	FLMBYTE *			pCurKey;
	FLMUINT				uiCurKeyLen;
	FLMBOOL				bValidKey;
	FLMUINT64			ui64KeyCount;
	FLMUINT64			ui64KeyRefs;
	FLMUINT				uiBlkAddress;
	FLMBYTE *			pBlk;
	FLMUINT				uiEndOfBlock;
	FLMUINT				uiElmOffset;
	FLMBYTE *			pElm;
	FLMUINT				uiElmLen;
	FLMUINT				uiElmLastFlag;
	FLMBYTE *			pElmKey;
	FLMUINT				uiElmKeyLen;
	FLMUINT				uiElmPKCLen;
	FLMUINT				uiElmDrn;
	FLMUINT				uiLastElmDrn;
	FLMBOOL				bElmRecOK;
	FLMUINT				uiElmRecLen;
	FLMBYTE *			pElmRec;
	FLMUINT				uiElmRecOffset;
	FLMUINT				uiFOPType;
#define FLM_FOP_CONT_DATA		1
#define FLM_FOP_STANDARD		2
#define FLM_FOP_OPEN				3
#define FLM_FOP_TAGGED			4
#define FLM_FOP_NO_VALUE		5
#define FLM_FOP_JUMP_LEVEL		6
#define FLM_FOP_NEXT_DRN		7
#define FLM_FOP_REC_INFO		8
#define FLM_FOP_ENCRYPTED		9
#define FLM_FOP_BAD				0xFF
	FLMUINT				uiFieldLen;
	FLMUINT				uiFieldProcessedLen;
	FLMUINT				uiFieldType;
	FLMUINT				uiFieldNum;
	FLMUINT				uiFieldLevel;
	FLMUINT				uiJumpLevel;
	FLMBYTE *			pFOPData;
	FLMUINT				uiFOPDataLen;
	FLMBYTE *			pValue;
	FLMBYTE *			pData;
	void *				pvField;
	FlmRecord *			pRecord;
	FLMUINT				uiCurrIxRefDrn;
	FLMUINT				uiElmOvhd;
	FLMUINT				uiChildCount;
	FLMUINT				uiEncId;
	FLMUINT				uiEncFieldLen;
} STATE_INFO;

typedef struct
{
	FILE_HDR				FileHdr;
	LOG_HDR				LogHdr;
} HDR_INFO;

typedef struct Rebuild_State
{
	STATUS_HOOK			fnStatusFunc;
	void *				AppArg;
	REBUILD_INFO		CallbackData;
	CORRUPT_INFO		CorruptInfo;
	HDR_INFO *			pHdrInfo;
	FLMUINT				uiMaxFileSize;
	FLMBYTE *			pKeyBuffer;
	STATE_INFO *		pStateInfo;
	F_SuperFileHdl *	pSFileHdl;
	HFDB					hDb;
	FlmRecord *			pRecord;
	FLMBYTE *			pBlk;
	FLMBYTE *			pLogHdr;
} REBUILD_STATE;

RCODE flmCreateNewFile(
	const char *		pszFilePath,
	const char *		pszDataDir,
	const char *		pszRflDir,
	const char *		pszDictFileName,
	const char *		pszDictBuf,
	CREATE_OPTS *		pCreateOpts,
	FLMUINT				uiStartCheckpoint,
	FDB_p *				ppDb,
	REBUILD_STATE *	pvRebuildState = NULL);

RCODE flmDbRebuildFile(
	REBUILD_STATE *	pRebuildState,
	FLMBOOL				bBadHeader);

eCorruptionType flmVerifyWPChar(
	FLMUINT				uiCharSet,
	FLMUINT				uiChar);

eCorruptionType flmVerifyTextField(
	FLMBYTE *			pText,
	FLMUINT				uiTextLen);

eCorruptionType flmVerifyNumberField(
	FLMBYTE *			pNumber,
	FLMUINT				uiNumberLen);

eCorruptionType flmVerifyField(
	FLMBYTE *			pField,
	FLMUINT				uiFieldLen,
	FLMUINT				uiFieldType);

eCorruptionType flmVerifyKey(
	FLMBYTE *			pKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiIxLang,
	IFD *					pIfdArray,
	FLMUINT				uiNumIxFields);

eCorruptionType flmVerifyBlockHeader(
	STATE_INFO *		pStateInfo,
	BLOCK_INFO *		pBlockInfo,
	FLMUINT				uiBlockSize,
	FLMUINT				uiNextBlkAddress,
	FLMUINT				uiPrevBlkAddress,
	FLMBOOL				bCheckEOF,
	FLMBOOL				bCheckFullBlkAddr);

FLMINT flmCompareKeys(
	FLMBYTE *			pBuf1,
	FLMUINT				uiBuf1Len,
	FLMBYTE *			pBuf2,
	FLMUINT				uiBuf2Len);

eCorruptionType flmVerifyElement(
	STATE_INFO *		pStateInfo,
	FLMUINT				uiFlags);

eCorruptionType flmVerifyElmFOP(
	STATE_INFO *		pStateInfo);

RCODE flmVerifyIXRefs(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT				uiResetDrn,
	eCorruptionType *	piElmCorruptionCode);

void flmInitReadState(
	STATE_INFO *		pStateInfo,
	FLMBOOL *			pbStateInitialized,
	FLMUINT				uiVersionNum,
	FDB *					pDb,
	LF_HDR *				pLogicalFile,
	FLMUINT				uiLevel,
	FLMUINT				uiBlkType,
	FLMBYTE *			pKeyBuffer);

RCODE flmGetRecKeys(
	FDB *					pDb,
	IXD *					pIxd,		
	FlmRecord *			pRecord,
	FLMUINT				uiContainerNum,
	FLMBOOL				bRemoveDups,
	POOL *				pPool,
	REC_KEY **			ppKeysRV);

RCODE chkVerifyIXRSet(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT				uiIxRefDrn);

RCODE chkGetNextRSKey(
	IX_CHK_INFO *		pIxChkInfo);

RCODE	chkResolveIXMissingKey(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo);

RCODE	chkResolveNonUniqueKey(
	STATE_INFO *		pStateInfo,
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT				uiIndex,
	FLMBYTE *			pucKey,
	FLMUINT				uiKeyLen,
	FLMUINT				uiDrn);

RCODE chkRSInit(
	const char *		pszIoPath,
	void **				pRSetRV);

RCODE chkRSFinalize(
	IX_CHK_INFO *		pIxChkInfo,
	FLMUINT64 *			pui64TotalEntries);

FLMINT chkCompareKeySet(
	FLMUINT				uiIxNum1,
	FLMBYTE *			pData1,
	FLMUINT				uiLength1,
	FLMUINT				uiDrn1,
	FLMUINT				uiIxNum2,
	FLMBYTE *			pData2,
	FLMUINT				uiLength2,
	FLMUINT				uiDrn2);

RCODE chkBlkRead(
	DB_INFO *			pDbInfo,
	FLMUINT				uiBlkAddress,
	LFILE *				pLFile,
	FLMBYTE **			ppBlk,
	SCACHE **			ppSCache,
	eCorruptionType *	peCorruption);

RCODE chkVerifyBTrees(
	DB_INFO *			pDbInfo,
	POOL *				pPool,
	FLMBOOL *			pbStartOverRV);

RCODE chkReportError(
	DB_INFO *			pDbInfo,
	eCorruptionType	eCorruption,
	eCorruptionLocale	eErrLocale,
	FLMUINT				uiErrLfNumber,
	FLMUINT				uiErrLfType,
	FLMUINT				uiErrBTreeLevel,
	FLMUINT				uiErrBlkAddress,
	FLMUINT				uiErrParentBlkAddress,
	FLMUINT				uiErrElmOffset,
	FLMUINT				uiErrDrn,
	FLMUINT				uiErrElmRecOffset,
	FLMUINT				uiErrFieldNum,
	FLMBYTE *			pBlk);

#ifdef FLM_UNIX
	RCODE MapErrnoToFlaimErr(
		int				err,
		RCODE				defaultRc);
#endif

void flmGetCPInfo(
	void *				pFile,
	CHECKPOINT_INFO *	pCheckpointInfo);

/****************************************************************************
Desc:
****************************************************************************/
FINLINE RCODE chkCallProgFunc(
	DB_INFO *			pDbInfo)
{
	if( (pDbInfo->fnStatusFunc) && (RC_OK( pDbInfo->LastStatusRc)))
	{
		pDbInfo->LastStatusRc = (*pDbInfo->fnStatusFunc)( FLM_CHECK_STATUS,
												(void *)pDbInfo->pProgress,
												(void *)0,
												pDbInfo->pProgress->AppArg);
	}
	return( pDbInfo->LastStatusRc);
}

/****************************************************************************
Desc: 	The FlmBlobImp class provides support for database Binary Large 
	`		Objects (BLOB). This class replaces the old 'C' FlmBlobXxx functions.
NOTE:		Initially the only minimal BLOB support is being provided. 
****************************************************************************/
class FlmBlobImp : public FlmBlob
{
public:

	FlmBlobImp( void)
	{
		m_pHeaderBuf = NULL;
		m_uiHeaderLen = 0;
		m_hDb = HFDB_NULL;
		m_pFileHdl = NULL;
		m_bInDbList = FALSE;
		m_pPrevBlob = m_pNextBlob = NULL;
	}

	virtual ~FlmBlobImp( void)
	{
		(void) close();
	}

	RCODE referenceFile( 
		HFDB					hDb, 
		const char *		pszFilePath, 
		FLMBOOL				bOwned = FALSE);

	RCODE setupBlobFromField( 
		FDB *					pDb,
		const FLMBYTE *	pBlobData,
		FLMUINT				uiBlobDataLength);

	RCODE close( void);

	FLMBYTE * getImportDataPtr(
		FLMUINT				uiLength);
	
	FINLINE FLMUINT getDataLength( void)
	{	
		return( m_uiHeaderLen);
	}
	
	FINLINE const FLMBYTE * getDataPtr( void)
	{
		return( m_pHeaderBuf);
	}
	
	FINLINE FlmBlobImp * getNext( void)
	{
		return m_pNextBlob;
	}

	FINLINE void setNext( 
		FlmBlobImp *		pNext)
	{
		m_pNextBlob = pNext;
	}

	FINLINE FlmBlobImp * getPrev( void)
	{
		return( m_pPrevBlob);
	}

	FINLINE void setPrev(
		FlmBlobImp *		pPrev)
	{
		m_pPrevBlob = pPrev;
	}

	FLMINT compareFileName(
		const char *		pszFileName);

	RCODE buildFileName(
		char *				pszFileName);

	// Action states - not to be used by the application.

	void transitionAction(
		FLMBOOL				bDoTransition);

	void setCurrentAction(
		FLMUINT				uiCurrentAction)
	{
		m_uiCurrentAction = uiCurrentAction;
	}

	FLMUINT getAction( void)
	{
		return( m_uiAction);
	}

	void setInDbList(
		FLMBOOL			bInDbList);

private:

	FLMBYTE *				m_pHeaderBuf;
	FLMUINT					m_uiHeaderLen;
	HFDB						m_hDb;
	F_FileHdlImp *			m_pFileHdl;
	FLMUINT					m_uiStorageType;
#define BLOB_REFERENCE_TYPE		0x04
#define BLOB_OWNED_TYPE				0x10
	FLMUINT					m_uiFlags;
	FLMUINT					m_uiAction;
#define BLOB_CREATE_ACTION			1
#define BLOB_OPEN_ACTION			2
	FLMBOOL					m_bReadWriteAccess;
	FLMBOOL					m_bFileAccessed;
	FLMUINT					m_uiCurrentAction;
#define BLOB_DELETE_ACTION			3
#define BLOB_ADD_ACTION				4
#define BLOB_PURGE_STATUS			5
#define BLOB_NO_ACTION				0

	FLMBOOL					m_bInDbList;
	FlmBlobImp *			m_pPrevBlob;
	FlmBlobImp *			m_pNextBlob;

	RCODE buildBlobHeader( 
		const char *		pszUnportablePath);
		
	RCODE closeFile( void);
	
	RCODE openFile( void);
};

/**************************************************************************
Desc:		Determines the number of significant bits in the block size
**************************************************************************/
FINLINE FLMUINT flmGetSigBits(
	FLMUINT				uiBlockSize)
{
	FLMUINT		uiSigBits = 0;

	while( !(uiBlockSize & 0x0001))
	{
		uiSigBits++;
		uiBlockSize >>= 1;
	}

	return( uiSigBits);
}

#include "fpackoff.h"

#endif
