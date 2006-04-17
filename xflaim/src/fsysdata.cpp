//------------------------------------------------------------------------------
// Desc:	This file contains the routines that initialize and shut down FLAIM,
//			as well as routines for configuring FLAIM.
//
// Tabs:	3
//
//		Copyright (c) 1995-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fsysdata.cpp 3123 2006-01-24 17:19:50 -0700 (Tue, 24 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#define ALLOCATE_SYS_DATA
#define ALLOC_ERROR_TABLES

#include <stddef.h>
#include "flaimsys.h"
#include "inifile.h"

#define HIGH_FLMUINT				(~((FLMUINT)0))
#define FLM_MIN_FREE_BYTES		(2 * 1024 * 1024)

#ifdef FLM_32BIT
	#if defined( FLM_LINUX)
	
	// With mmap'd memory on Linux, you're effectively limited to about ~2 GB.
	// Userspace only gets ~3GB of usable address space anyway, and then you
	// have all of the thread stacks too, which you can't have 
	// overlapping the heap.
	
		#define FLM_MAX_CACHE_SIZE		(1500 * 1024 * 1024)
	#else
		#define FLM_MAX_CACHE_SIZE		(2000 * 1024 * 1024)
	#endif
#else
	#define FLM_MAX_CACHE_SIZE			(~((FLMUINT)0))
#endif

FLMATOMIC			F_DbSystem::m_flmSysSpinLock = 0;
FLMUINT				F_DbSystem::m_uiFlmSysStartupCount = 0;

static FLMBYTE		ucSENPrefixArray[] =
								{0, 0, 0x80, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC, 0xFE, 0xFF};

// Local Function Prototypes

#ifdef FLM_CAN_GET_PHYS_MEM
FSTATIC FLMUINT flmGetCacheBytes(
	FLMUINT		uiPercent,
	FLMUINT		uiMin,
	FLMUINT		uiMax,
	FLMUINT		uiMinToLeave,
	FLMBOOL		bCalcOnAvailMem,
	FLMUINT		uiBytesCurrentlyInUse);
#endif

FSTATIC RCODE flmVerifyDiskStructOffsets( void);

FSTATIC void flmFreeEvent(
	FEVENT *			pEvent,
	F_MUTEX			hMutex,
	FEVENT **		ppEventListRV);

FSTATIC void flmGetStringParam(
	const char *	pszParamName,
	char **			ppszValue,
	F_IniFile *		pIniFile);

FSTATIC void flmGetNumParam(
	char **		ppszParam,
	FLMUINT *	puiNum);

FSTATIC void flmGetUintParam(
	const char *	pszParamName,
	FLMUINT			uiDefaultValue,
	FLMUINT *		puiUint,
	F_IniFile *		pIniFile);

void flmGetBoolParam(
	const char *	pszParamName,
	FLMBOOL			uiDefaultValue,
	FLMBOOL *		pbBool,
	F_IniFile *		pIniFile);

FSTATIC RCODE flmGetIniFileName(
	FLMBYTE *		pszIniFileName,
	FLMUINT			uiBufferSz);

/****************************************************************************
Desc:		Error code to string mapping tables
****************************************************************************/
typedef struct
{
	RCODE				rc;
	const char *	pszErrorStr;
} F_ERROR_CODE_MAP;

#define flmErrorCodeEntry(c)		{ c, #c }

#ifdef FLM_NLM
	FLMBOOL gv_bNetWareStartupCalled = FALSE;
#endif

F_ERROR_CODE_MAP gv_FlmCommonErrors[
	NE_XFLM_LAST_COMMON_ERROR - NE_XFLM_FIRST_COMMON_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_NOT_IMPLEMENTED),
	flmErrorCodeEntry( NE_XFLM_MEM),
	{ XFLM_ERROR_BASE( 0x0003), "NotUsed"},
	{ XFLM_ERROR_BASE( 0x0004), "NotUsed"},
	flmErrorCodeEntry( NE_XFLM_INVALID_PARM),
	{ XFLM_ERROR_BASE( 0x0006), "NotUsed"},
	{ XFLM_ERROR_BASE( 0x0007), "NotUsed"},
	{ XFLM_ERROR_BASE( 0x0008), "NotUsed"},
	flmErrorCodeEntry( NE_XFLM_TIMEOUT),
	flmErrorCodeEntry( NE_XFLM_NOT_FOUND),
	{ XFLM_ERROR_BASE( 0x000B), "NotUsed"},
	flmErrorCodeEntry( NE_XFLM_EXISTS),
	{ XFLM_ERROR_BASE( 0x000D), "NotUsed"},
	{ XFLM_ERROR_BASE( 0x000E), "NotUsed"},
	{ XFLM_ERROR_BASE( 0x000F), "NotUsed"},
	flmErrorCodeEntry( NE_XFLM_USER_ABORT),
	flmErrorCodeEntry( NE_XFLM_FAILURE)
};

F_ERROR_CODE_MAP gv_FlmGeneralErrors[
	NE_XFLM_LAST_GENERAL_ERROR - NE_XFLM_FIRST_GENERAL_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_BOF_HIT),
	flmErrorCodeEntry( NE_XFLM_EOF_HIT),
	flmErrorCodeEntry( NE_XFLM_END),
	flmErrorCodeEntry( NE_XFLM_BAD_PREFIX),
	flmErrorCodeEntry( NE_XFLM_ATTRIBUTE_PURGED),
	flmErrorCodeEntry( NE_XFLM_BAD_COLLECTION),
	flmErrorCodeEntry( NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_DATA_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_BAD_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_MUST_INDEX_ON_PRESENCE),
	flmErrorCodeEntry( NE_XFLM_BAD_IX),
	flmErrorCodeEntry( NE_XFLM_BACKUP_ACTIVE),
	flmErrorCodeEntry( NE_XFLM_SERIAL_NUM_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_DB_SERIAL_NUM),
	flmErrorCodeEntry( NE_XFLM_BTREE_ERROR),
	flmErrorCodeEntry( NE_XFLM_BTREE_FULL),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_FILE_NUMBER),
	flmErrorCodeEntry( NE_XFLM_CANNOT_DEL_ELEMENT),
	flmErrorCodeEntry( NE_XFLM_CANNOT_MOD_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_CANNOT_INDEX_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_CONV_BAD_DIGIT),
	flmErrorCodeEntry( NE_XFLM_CONV_DEST_OVERFLOW),
	flmErrorCodeEntry( NE_XFLM_CONV_ILLEGAL),
	flmErrorCodeEntry( NE_XFLM_CONV_NULL_SRC),
	flmErrorCodeEntry( NE_XFLM_CONV_NUM_OVERFLOW),
	flmErrorCodeEntry( NE_XFLM_CONV_NUM_UNDERFLOW),
	flmErrorCodeEntry( NE_XFLM_BAD_ELEMENT_NUM),
	flmErrorCodeEntry( NE_XFLM_BAD_ATTRIBUTE_NUM),
	flmErrorCodeEntry( NE_XFLM_BAD_ENCDEF_NUM),
	flmErrorCodeEntry( NE_XFLM_DATA_ERROR),
	flmErrorCodeEntry( NE_XFLM_INVALID_FILE_SEQUENCE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_OP),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_ELEMENT_NUM),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_TRANS_TYPE),
	flmErrorCodeEntry( NE_XFLM_UNSUPPORTED_VERSION),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_TRANS_OP),
	flmErrorCodeEntry( NE_XFLM_INCOMPLETE_LOG),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_INDEX_DEF),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_INDEX_ON),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_STATE_CHANGE),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_SERIAL_NUM),
	flmErrorCodeEntry( NE_XFLM_NEWER_FLAIM),
	flmErrorCodeEntry( NE_XFLM_CANNOT_MOD_ELEMENT_STATE),
	flmErrorCodeEntry( NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_ELEMENT_NUMS),
	flmErrorCodeEntry( NE_XFLM_NO_TRANS_ACTIVE),
	flmErrorCodeEntry( NE_XFLM_NOT_UNIQUE),
	flmErrorCodeEntry( NE_XFLM_NOT_FLAIM),
	flmErrorCodeEntry( NE_XFLM_OLD_VIEW),
	flmErrorCodeEntry( NE_XFLM_SHARED_LOCK),
	flmErrorCodeEntry( NE_XFLM_SYNTAX),
	flmErrorCodeEntry( NE_XFLM_TRANS_ACTIVE),
	flmErrorCodeEntry( NE_XFLM_RFL_TRANS_GAP),
	flmErrorCodeEntry( NE_XFLM_BAD_COLLATED_KEY),
	flmErrorCodeEntry( NE_XFLM_UNSUPPORTED_FEATURE),
	flmErrorCodeEntry( NE_XFLM_MUST_DELETE_INDEXES),
	flmErrorCodeEntry( NE_XFLM_RFL_INCOMPLETE),
	flmErrorCodeEntry( NE_XFLM_CANNOT_RESTORE_RFL_FILES),
	flmErrorCodeEntry( NE_XFLM_INCONSISTENT_BACKUP),
	flmErrorCodeEntry( NE_XFLM_BLOCK_CRC),
	flmErrorCodeEntry( NE_XFLM_ABORT_TRANS),
	flmErrorCodeEntry( NE_XFLM_NOT_RFL),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_PACKET),
	flmErrorCodeEntry( NE_XFLM_DATA_PATH_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_STREAM_EXISTS),
	flmErrorCodeEntry( NE_XFLM_FILE_EXISTS),
	flmErrorCodeEntry( NE_XFLM_COULD_NOT_CREATE_SEMAPHORE),
	flmErrorCodeEntry( NE_XFLM_MUST_CLOSE_DATABASE),
	flmErrorCodeEntry( NE_XFLM_INVALID_ENCKEY_CRC),
	flmErrorCodeEntry( NE_XFLM_BAD_UTF8),
	flmErrorCodeEntry( NE_XFLM_COULD_NOT_CREATE_MUTEX),
	flmErrorCodeEntry( NE_XFLM_ERROR_WAITING_ON_SEMPAHORE),
	flmErrorCodeEntry( NE_XFLM_BAD_PLATFORM_FORMAT),
	flmErrorCodeEntry( NE_XFLM_HDR_CRC),
	flmErrorCodeEntry( NE_XFLM_NO_NAME_TABLE),
	flmErrorCodeEntry( NE_XFLM_MULTIPLE_MATCHES),
	flmErrorCodeEntry( NE_XFLM_UNALLOWED_UPGRADE),
	flmErrorCodeEntry( NE_XFLM_BTREE_BAD_STATE),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_ATTRIBUTE_NUM),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_INDEX_NUM),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_COLLECTION_NUM),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_ELEMENT_NAME),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_ATTRIBUTE_NAME),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_INDEX_NAME),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_COLLECTION_NAME),
	flmErrorCodeEntry( NE_XFLM_ELEMENT_PURGED),
	flmErrorCodeEntry( NE_XFLM_TOO_MANY_OPEN_DATABASES),
	flmErrorCodeEntry( NE_XFLM_DATABASE_OPEN),
	flmErrorCodeEntry( NE_XFLM_CACHE_ERROR),
	flmErrorCodeEntry( NE_XFLM_BTREE_KEY_SIZE),
	flmErrorCodeEntry( NE_XFLM_DB_FULL),
	flmErrorCodeEntry( NE_XFLM_QUERY_SYNTAX),
	flmErrorCodeEntry( NE_XFLM_COULD_NOT_START_THREAD),
	flmErrorCodeEntry( NE_XFLM_INDEX_OFFLINE),
	flmErrorCodeEntry( NE_XFLM_RFL_DISK_FULL),
	flmErrorCodeEntry( NE_XFLM_MUST_WAIT_CHECKPOINT),
	flmErrorCodeEntry( NE_XFLM_MISSING_ENC_ALGORITHM),
	flmErrorCodeEntry( NE_XFLM_INVALID_ENC_ALGORITHM),
	flmErrorCodeEntry( NE_XFLM_INVALID_ENC_KEY_SIZE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_STATE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_ELEMENT_NAME),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_ATTRIBUTE_NAME),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_COLLECTION_NAME),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_INDEX_NAME),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_ELEMENT_NUMBER),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_COLLECTION_NUMBER),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_INDEX_NUMBER),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_ENCDEF_NUMBER),
	flmErrorCodeEntry( NE_XFLM_COLLECTION_NAME_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_ELEMENT_NAME_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_ATTRIBUTE_NAME_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_INVALID_COMPARE_RULE),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_KEY_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_DATA_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_MISSING_KEY_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_MISSING_DATA_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_INVALID_INDEX_OPTION),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_ATTRIBUTE_NUMS),
	flmErrorCodeEntry( NE_XFLM_MISSING_ELEMENT_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_ATTRIBUTE_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_ELEMENT_NUMBER),
	flmErrorCodeEntry( NE_XFLM_MISSING_ATTRIBUTE_NUMBER),
	flmErrorCodeEntry( NE_XFLM_MISSING_INDEX_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_INDEX_NUMBER),
	flmErrorCodeEntry( NE_XFLM_MISSING_COLLECTION_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_COLLECTION_NUMBER),
	flmErrorCodeEntry( NE_XFLM_BAD_SEN),
	flmErrorCodeEntry( NE_XFLM_MISSING_ENCDEF_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_ENCDEF_NUMBER),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_INDEX_NUMS),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_COLLECTION_NUMS),
	flmErrorCodeEntry( NE_XFLM_CANNOT_DEL_ATTRIBUTE),
	flmErrorCodeEntry( NE_XFLM_TOO_MANY_PENDING_NODES),
	flmErrorCodeEntry( NE_XFLM_UNSUPPORTED_INTERFACE),
	flmErrorCodeEntry( NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG),
	flmErrorCodeEntry( NE_XFLM_DUP_SIBLING_IX_COMPONENTS),
	flmErrorCodeEntry( NE_XFLM_RFL_FILE_NOT_FOUND),
	flmErrorCodeEntry( NE_XFLM_BAD_RCODE_TABLE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM),
	flmErrorCodeEntry( NE_XFLM_CLASS_NOT_AVAILABLE),
	flmErrorCodeEntry( NE_XFLM_BUFFER_OVERFLOW),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_PREFIX_NUMBER),
	flmErrorCodeEntry( NE_XFLM_MISSING_PREFIX_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_PREFIX_NUMBER),
	flmErrorCodeEntry( NE_XFLM_UNDEFINED_ELEMENT_NAME),
	flmErrorCodeEntry( NE_XFLM_UNDEFINED_ATTRIBUTE_NAME),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_PREFIX_NAME),
	flmErrorCodeEntry( NE_XFLM_KEY_OVERFLOW),
	flmErrorCodeEntry( NE_XFLM_UNESCAPED_METACHAR),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_QUANTIFIER),
	flmErrorCodeEntry( NE_XFLM_UNEXPECTED_END_OF_EXPR),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_MIN_COUNT),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_MAX_COUNT),
	flmErrorCodeEntry( NE_XFLM_EMPTY_BRANCH_IN_EXPR),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_RPAREN_IN_EXPR),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_CLASS_SUBTRACTION),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_CHAR_RANGE_IN_EXPR),
	flmErrorCodeEntry( NE_XFLM_BAD_BASE64_ENCODING),
	flmErrorCodeEntry( NE_XFLM_NAMESPACE_NOT_ALLOWED),
	flmErrorCodeEntry( NE_XFLM_INVALID_NAMESPACE_DECL),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE),
	flmErrorCodeEntry( NE_XFLM_UNEXPECTED_END_OF_INPUT),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_PREFIX_NUMS),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_ENCDEF_NUMS),
	flmErrorCodeEntry( NE_XFLM_COLLECTION_OFFLINE),
	flmErrorCodeEntry( NE_XFLM_INVALID_XML),
	flmErrorCodeEntry( NE_XFLM_READ_ONLY),
	flmErrorCodeEntry( NE_XFLM_DELETE_NOT_ALLOWED),
	flmErrorCodeEntry( NE_XFLM_RESET_NEEDED),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_REQUIRED_VALUE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_INDEX_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE),
	flmErrorCodeEntry( NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_FLAG),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_REQUIRED),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_LIMIT),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_INDEX_ON),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_COMPARE_RULES),
	flmErrorCodeEntry( NE_XFLM_INPUT_PENDING),
	flmErrorCodeEntry( NE_XFLM_INVALID_NODE_TYPE),
	flmErrorCodeEntry( NE_XFLM_INVALID_CHILD_ELM_NODE_ID)
};

F_ERROR_CODE_MAP gv_FlmDomErrors[
	NE_XFLM_LAST_DOM_ERROR - NE_XFLM_FIRST_DOM_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_DOM_HIERARCHY_REQUEST_ERR),
	flmErrorCodeEntry( NE_XFLM_DOM_WRONG_DOCUMENT_ERR),
	flmErrorCodeEntry( NE_XFLM_DOM_DATA_ERROR),
	flmErrorCodeEntry( NE_XFLM_DOM_NODE_NOT_FOUND),
	flmErrorCodeEntry( NE_XFLM_DOM_INVALID_CHILD_TYPE),
	flmErrorCodeEntry( NE_XFLM_DOM_NODE_DELETED),
	flmErrorCodeEntry( NE_XFLM_DOM_DUPLICATE_ELEMENT)
};

F_ERROR_CODE_MAP gv_FlmIoErrors[
	NE_XFLM_LAST_IO_ERROR - NE_XFLM_FIRST_IO_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_IO_ACCESS_DENIED),
	flmErrorCodeEntry( NE_XFLM_IO_BAD_FILE_HANDLE),
	flmErrorCodeEntry( NE_XFLM_IO_COPY_ERR),
	flmErrorCodeEntry( NE_XFLM_IO_DISK_FULL),
	flmErrorCodeEntry( NE_XFLM_IO_END_OF_FILE),
	flmErrorCodeEntry( NE_XFLM_IO_OPEN_ERR),
	flmErrorCodeEntry( NE_XFLM_IO_SEEK_ERR),
	flmErrorCodeEntry( NE_XFLM_IO_DIRECTORY_ERR),
	flmErrorCodeEntry( NE_XFLM_IO_PATH_NOT_FOUND),
	flmErrorCodeEntry( NE_XFLM_IO_TOO_MANY_OPEN_FILES),
	flmErrorCodeEntry( NE_XFLM_IO_PATH_TOO_LONG),
	flmErrorCodeEntry( NE_XFLM_IO_NO_MORE_FILES),
	flmErrorCodeEntry( NE_XFLM_IO_DELETING_FILE),
	flmErrorCodeEntry( NE_XFLM_IO_FILE_LOCK_ERR),
	flmErrorCodeEntry( NE_XFLM_IO_FILE_UNLOCK_ERR),
	flmErrorCodeEntry( NE_XFLM_IO_PATH_CREATE_FAILURE),
	flmErrorCodeEntry( NE_XFLM_IO_RENAME_FAILURE),
	flmErrorCodeEntry( NE_XFLM_IO_INVALID_PASSWORD),
	flmErrorCodeEntry( NE_XFLM_SETTING_UP_FOR_READ),
	flmErrorCodeEntry( NE_XFLM_SETTING_UP_FOR_WRITE),
	flmErrorCodeEntry( NE_XFLM_IO_CANNOT_REDUCE_PATH),
	flmErrorCodeEntry( NE_XFLM_INITIALIZING_IO_SYSTEM),
	flmErrorCodeEntry( NE_XFLM_FLUSHING_FILE),
	flmErrorCodeEntry( NE_XFLM_IO_INVALID_FILENAME),
	flmErrorCodeEntry( NE_XFLM_IO_CONNECT_ERROR),
	flmErrorCodeEntry( NE_XFLM_OPENING_FILE),
	flmErrorCodeEntry( NE_XFLM_DIRECT_OPENING_FILE),
	flmErrorCodeEntry( NE_XFLM_CREATING_FILE),
	flmErrorCodeEntry( NE_XFLM_DIRECT_CREATING_FILE),
	flmErrorCodeEntry( NE_XFLM_READING_FILE),
	flmErrorCodeEntry( NE_XFLM_DIRECT_READING_FILE),
	flmErrorCodeEntry( NE_XFLM_WRITING_FILE),
	flmErrorCodeEntry( NE_XFLM_DIRECT_WRITING_FILE),
	flmErrorCodeEntry( NE_XFLM_POSITIONING_IN_FILE),
	flmErrorCodeEntry( NE_XFLM_GETTING_FILE_SIZE),
	flmErrorCodeEntry( NE_XFLM_TRUNCATING_FILE),
	flmErrorCodeEntry( NE_XFLM_PARSING_FILE_NAME),
	flmErrorCodeEntry( NE_XFLM_CLOSING_FILE),
	flmErrorCodeEntry( NE_XFLM_GETTING_FILE_INFO),
	flmErrorCodeEntry( NE_XFLM_EXPANDING_FILE),
	flmErrorCodeEntry( NE_XFLM_CHECKING_FILE_EXISTENCE),
	flmErrorCodeEntry( NE_XFLM_RENAMING_FILE),
	flmErrorCodeEntry( NE_XFLM_SETTING_FILE_INFO)
};

F_ERROR_CODE_MAP gv_FlmNetErrors[
	NE_XFLM_LAST_NET_ERROR - NE_XFLM_FIRST_NET_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_NOIP_ADDR),
	flmErrorCodeEntry( NE_XFLM_SOCKET_FAIL),
	flmErrorCodeEntry( NE_XFLM_CONNECT_FAIL),
	flmErrorCodeEntry( NE_XFLM_BIND_FAIL),
	flmErrorCodeEntry( NE_XFLM_LISTEN_FAIL),
	flmErrorCodeEntry( NE_XFLM_ACCEPT_FAIL),
	flmErrorCodeEntry( NE_XFLM_SELECT_ERR),
	flmErrorCodeEntry( NE_XFLM_SOCKET_SET_OPT_FAIL),
	flmErrorCodeEntry( NE_XFLM_SOCKET_DISCONNECT),
	flmErrorCodeEntry( NE_XFLM_SOCKET_READ_FAIL),
	flmErrorCodeEntry( NE_XFLM_SOCKET_WRITE_FAIL),
	flmErrorCodeEntry( NE_XFLM_SOCKET_READ_TIMEOUT),
	flmErrorCodeEntry( NE_XFLM_SOCKET_WRITE_TIMEOUT),
	flmErrorCodeEntry( NE_XFLM_SOCKET_ALREADY_CLOSED)
};

F_ERROR_CODE_MAP gv_FlmQueryErrors[
	NE_XFLM_LAST_QUERY_ERROR - NE_XFLM_FIRST_QUERY_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_Q_UNMATCHED_RPAREN),
	flmErrorCodeEntry( NE_XFLM_Q_UNEXPECTED_LPAREN),
	flmErrorCodeEntry( NE_XFLM_Q_UNEXPECTED_RPAREN),
	flmErrorCodeEntry( NE_XFLM_Q_EXPECTING_OPERAND),
	flmErrorCodeEntry( NE_XFLM_Q_EXPECTING_OPERATOR),
	flmErrorCodeEntry( NE_XFLM_Q_UNEXPECTED_COMMA),
	flmErrorCodeEntry( NE_XFLM_Q_EXPECTING_LPAREN),
	flmErrorCodeEntry( NE_XFLM_Q_UNEXPECTED_VALUE),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_NUM_FUNC_ARGS),
	flmErrorCodeEntry( NE_XFLM_Q_UNEXPECTED_XPATH_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_Q_ILLEGAL_LBRACKET),
	flmErrorCodeEntry( NE_XFLM_Q_ILLEGAL_RBRACKET),
	flmErrorCodeEntry( NE_XFLM_Q_ILLEGAL_OPERAND),
	flmErrorCodeEntry( NE_XFLM_Q_ALREADY_OPTIMIZED),
	flmErrorCodeEntry( NE_XFLM_Q_MISMATCHED_DB),
	flmErrorCodeEntry( NE_XFLM_Q_ILLEGAL_OPERATOR),
	flmErrorCodeEntry( NE_XFLM_Q_ILLEGAL_COMPARE_RULES),
	flmErrorCodeEntry( NE_XFLM_Q_INCOMPLETE_QUERY_EXPR),
	flmErrorCodeEntry( NE_XFLM_Q_NOT_POSITIONED),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_NODE_ID_VALUE),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_META_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_Q_NEW_EXPR_NOT_ALLOWED),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_CONTEXT_POS),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_FUNC_ARG),
	flmErrorCodeEntry( NE_XFLM_Q_EXPECTING_RPAREN),
	flmErrorCodeEntry( NE_XFLM_Q_TOO_LATE_TO_ADD_SORT_KEYS),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_SORT_KEY_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_Q_DUPLICATE_SORT_KEY_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_Q_MISSING_SORT_KEY_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_Q_NO_SORT_KEY_COMPONENTS_SPECIFIED),
	flmErrorCodeEntry( NE_XFLM_Q_SORT_KEY_CONTEXT_MUST_BE_ELEMENT),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_ELEMENT_NUM_IN_SORT_KEYS),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_ATTR_NUM_IN_SORT_KEYS),
	flmErrorCodeEntry( NE_XFLM_Q_NON_POSITIONABLE_QUERY),
	flmErrorCodeEntry( NE_XFLM_Q_INVALID_POSITION)
};

F_ERROR_CODE_MAP gv_FlmStreamErrors[
	NE_XFLM_LAST_STREAM_ERROR - NE_XFLM_FIRST_STREAM_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_STREAM_DECOMPRESS_ERROR),
	flmErrorCodeEntry( NE_XFLM_STREAM_NOT_COMPRESSED),
	flmErrorCodeEntry( NE_XFLM_STREAM_TOO_MANY_FILES)
};

F_ERROR_CODE_MAP gv_FlmNiciErrors[
	NE_XFLM_LAST_NICI_ERROR - NE_XFLM_FIRST_NICI_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_NICI_CONTEXT),
	flmErrorCodeEntry( NE_XFLM_NICI_ATTRIBUTE_VALUE),
	flmErrorCodeEntry( NE_XFLM_NICI_BAD_ATTRIBUTE),
	flmErrorCodeEntry( NE_XFLM_NICI_WRAPKEY_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_UNWRAPKEY_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_INVALID_ALGORITHM),
	flmErrorCodeEntry( NE_XFLM_NICI_GENKEY_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_BAD_RANDOM),
	flmErrorCodeEntry( NE_XFLM_PBE_ENCRYPT_FAILED),
	flmErrorCodeEntry( NE_XFLM_PBE_DECRYPT_FAILED),
	flmErrorCodeEntry( NE_XFLM_DIGEST_INIT_FAILED),
	flmErrorCodeEntry( NE_XFLM_DIGEST_FAILED),
	flmErrorCodeEntry( NE_XFLM_INJECT_KEY_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_FIND_INIT),
	flmErrorCodeEntry( NE_XFLM_NICI_FIND_OBJECT),
	flmErrorCodeEntry( NE_XFLM_NICI_KEY_NOT_FOUND),
	flmErrorCodeEntry( NE_XFLM_NICI_ENC_INIT_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_ENCRYPT_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_DECRYPT_INIT_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_DECRYPT_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_WRAPKEY_NOT_FOUND),
	flmErrorCodeEntry( NE_XFLM_NOT_EXPECTING_PASSWORD),
	flmErrorCodeEntry( NE_XFLM_EXPECTING_PASSWORD),
	flmErrorCodeEntry( NE_XFLM_EXTRACT_KEY_FAILED),
	flmErrorCodeEntry( NE_XFLM_NICI_INIT_FAILED),
	flmErrorCodeEntry( NE_XFLM_BAD_ENCKEY_SIZE),
	flmErrorCodeEntry( NE_XFLM_ENCRYPTION_UNAVAILABLE)
};

/****************************************************************************
Desc: This routine allocates and initializes a hash table.
****************************************************************************/
RCODE flmAllocHashTbl(
	FLMUINT		uiHashTblSize,
	FBUCKET **	ppHashTblRV)
{
	RCODE					rc = NE_XFLM_OK;
	FBUCKET *			pHashTbl = NULL;
	F_RandomGenerator	RandGen;
	FLMUINT				uiCnt;
	FLMUINT				uiRandVal;
	FLMUINT				uiTempVal;

	// Allocate memory for the hash table

	if (RC_BAD( rc = f_calloc(
		(FLMUINT)(sizeof( FBUCKET)) * uiHashTblSize, &pHashTbl)))
	{
		goto Exit;
	}

	// Initialize the hash table

	RandGen.randomSetSeed( 1);

	for (uiCnt = 0; uiCnt < uiHashTblSize; uiCnt++)
	{
		pHashTbl [uiCnt].uiHashValue = (FLMBYTE)uiCnt;
		pHashTbl [uiCnt].pFirstInBucket = NULL;
	}

	if( uiHashTblSize <= 256)
	{
		for( uiCnt = 0; uiCnt < uiHashTblSize - 1; uiCnt++)
		{
			uiRandVal = (FLMBYTE) RandGen.randomChoice( (FLMINT32)uiCnt,
										(FLMINT32)(uiHashTblSize - 1));
			if( uiRandVal != uiCnt)
			{
				uiTempVal = (FLMBYTE)pHashTbl [uiCnt].uiHashValue;
				pHashTbl [uiCnt].uiHashValue = pHashTbl [uiRandVal].uiHashValue;
				pHashTbl [uiRandVal].uiHashValue = uiTempVal;
			}
		}
	}

Exit:

	*ppHashTblRV = pHashTbl;
	return( rc);
}

#ifdef FLM_CAN_GET_PHYS_MEM
/****************************************************************************
Desc: This routine determines the number of cache bytes to use for caching
		based on a percentage of available physical memory or a percentage
		of physical memory (depending on bCalcOnAvailMem flag).
		uiBytesCurrentlyInUse indicates how many bytes are currently allocated
		by FLAIM - so it can factor that in if the calculation is to be based
		on the available memory.
		Lower limit is 1 megabyte.
****************************************************************************/
FSTATIC FLMUINT flmGetCacheBytes(
	FLMUINT		uiPercent,
	FLMUINT		uiMin,
	FLMUINT		uiMax,
	FLMUINT		uiMinToLeave,
	FLMBOOL		bCalcOnAvailMem,
	FLMUINT		uiBytesCurrentlyInUse
	)
{
	FLMUINT			uiMem;
#ifdef FLM_WIN
	MEMORYSTATUS	MemStatus;
#elif defined( FLM_UNIX)
	FLMUINT			uiProcMemLimit = HIGH_FLMUINT;
	FLMUINT			uiProcVMemLimit = HIGH_FLMUINT;
#endif

#ifdef FLM_WIN
	GlobalMemoryStatus( &MemStatus);
	uiMem = (FLMUINT)((bCalcOnAvailMem)
							? (FLMUINT)MemStatus.dwAvailPhys
							: (FLMUINT)MemStatus.dwTotalPhys);
#elif defined( FLM_UNIX)
	{
		#if defined( RLIMIT_VMEM)
			{
				struct rlimit	rlim;
	
				// Bump the process soft virtual limit up to the hard limit
				
				if( getrlimit( RLIMIT_VMEM, &rlim) == 0)
				{
					if( rlim.rlim_cur < rlim.rlim_max)
					{
						rlim.rlim_cur = rlim.rlim_max;
						(void)setrlimit( RLIMIT_VMEM, &rlim);
						if( getrlimit( RLIMIT_VMEM, &rlim) != 0)
						{
							rlim.rlim_cur = RLIM_INFINITY;
							rlim.rlim_max = RLIM_INFINITY;
						}
					}
	
					if( rlim.rlim_cur != RLIM_INFINITY)
					{
						uiProcVMemLimit = (FLMUINT)rlim.rlim_cur;
					}
				}
			}
		#endif

		#if defined( RLIMIT_DATA)
			{
				struct rlimit	rlim;
	
				// Bump the process soft heap limit up to the hard limit
				
				if( getrlimit( RLIMIT_DATA, &rlim) == 0)
				{
					if( rlim.rlim_cur < rlim.rlim_max)
					{
						rlim.rlim_cur = rlim.rlim_max;
						(void)setrlimit( RLIMIT_DATA, &rlim);
						if( getrlimit( RLIMIT_DATA, &rlim) != 0)
						{
							rlim.rlim_cur = RLIM_INFINITY;
							rlim.rlim_max = RLIM_INFINITY;
						}
					}
	
					if( rlim.rlim_cur != RLIM_INFINITY)
					{
						uiProcMemLimit = (FLMUINT)rlim.rlim_cur;
					}
				}
			}
		#endif
	
#ifdef FLM_AIX
		struct vminfo		tmpvminfo;
	#ifdef _SC_PAGESIZE
		long		iPageSize = sysconf(_SC_PAGESIZE);
	#else
		long		iPageSize = 4096;
	#endif

		if( iPageSize == -1)
		{
			// If sysconf returned an error, resort to using the default
			// page size for the Power architecture.

			iPageSize = 4096;
		}

		uiMem = HIGH_FLMUINT;
		
		if( vmgetinfo( &tmpvminfo, VMINFO, sizeof( tmpvminfo)) != -1)
		{
			if( bCalcOnAvailMem)
			{
				if( tmpvminfo.numfrb < HIGH_FLMUINT)
				{
					uiMem = (FLMUINT)tmpvminfo.numfrb;
				}
			}
			else
			{
				if( tmpvminfo.memsizepgs < HIGH_FLMUINT)
				{
					uiMem = (FLMUINT)tmpvminfo.memsizepgs;
				}
			}
		}
		
		if (HIGH_FLMUINT / (FLMUINT)iPageSize >= uiMem)
		{
			uiMem *= (FLMUINT)iPageSize;
		}
		else
		{
			uiMem = HIGH_FLMUINT;
		}
#elif defined( FLM_LINUX)

		FLMUINT64		ui64TotalMem;
		FLMUINT64		ui64AvailMem;
		FLMUINT64		ui64Mem;
		
		flmGetLinuxMemInfo( &ui64TotalMem, &ui64AvailMem);
		
		ui64Mem = bCalcOnAvailMem
								? ui64AvailMem
								: ui64TotalMem;
								
		if( HIGH_FLMUINT >= ui64Mem)
		{
			uiMem = (FLMUINT)ui64Mem;
		}
		else
		{
			uiMem = HIGH_FLMUINT;
		}

#else

		long		iPageSize = sysconf( _SC_PAGESIZE);

		// Get the amount of memory available to the system

		uiMem = (FLMUINT)((bCalcOnAvailMem)
								? (FLMUINT)sysconf(_SC_AVPHYS_PAGES)
								: (FLMUINT)sysconf(_SC_PHYS_PAGES));

		if (HIGH_FLMUINT / (FLMUINT)iPageSize >= uiMem)
		{
			uiMem *= (FLMUINT)iPageSize;
		}
		else
		{
			uiMem = HIGH_FLMUINT;
		}
#endif
	}
#elif defined( FLM_NLM)
	{

		#ifndef _SC_PHYS_PAGES
			#define _SC_PHYS_PAGES        56
		#endif
		#ifndef _SCAVPHYS_PAGES
			#define _SC_AVPHYS_PAGES		57
		#endif

		long				iPageSize = sysconf(_SC_PAGESIZE);

		// Get the amount of memory available to the system

		uiMem = (FLMUINT)((bCalcOnAvailMem)
								? (FLMUINT)sysconf(_SC_AVPHYS_PAGES)
								: (FLMUINT)sysconf(_SC_PHYS_PAGES));


		if (HIGH_FLMUINT / (FLMUINT)iPageSize >= uiMem)
		{
			uiMem *= (FLMUINT)iPageSize;
		}
		else
		{
			uiMem = HIGH_FLMUINT;
		}
	}
#else
	#error Getting physical memory is not supported by this platform.
#endif

	// If we are basing the calculation on available physical memory,
	// take in to account what has already been allocated.

	if (bCalcOnAvailMem)
	{
		if (uiMem > HIGH_FLMUINT - uiBytesCurrentlyInUse)
		{
			uiMem = HIGH_FLMUINT;
		}
		else
		{
			uiMem += uiBytesCurrentlyInUse;
		}
	}

	// Determine if there are limits on the amount of memory the
	// process can access and reset uiMem accordingly.  There may
	// be more available memory than the process is able to access.

#ifdef FLM_WIN

	// There could be more physical memory in the system than we could
	// actually allocate in our virtual address space.  Thus, we need to
	// make sure that we never exceed our total virtual address space.

	if (uiMem > (FLMUINT)MemStatus.dwTotalVirtual)
	{
		uiMem = (FLMUINT)MemStatus.dwTotalVirtual;
	}

#elif defined( FLM_UNIX)

	// The process might be limited in the amount of memory it
	// can access.

	if ( uiMem > uiProcMemLimit)
	{
		uiMem = uiProcMemLimit;
	}

	if( uiMem > uiProcVMemLimit)
	{
		uiMem = uiProcVMemLimit;
	}

#endif

	// If uiMax is zero, use uiMinToLeave to calculate the maximum.

	if (!uiMax)
	{
		if (!uiMinToLeave)
		{
			uiMax = uiMem;
		}
		else if (uiMinToLeave < uiMem)
		{
			uiMax = uiMem - uiMinToLeave;
		}
		else
		{
			uiMax = 0;
		}
	}

	// Calculate memory as a percentage of memory.

	uiMem = (FLMUINT)((uiMem > HIGH_FLMUINT / 100)
							? (FLMUINT)(uiMem / 100) * uiPercent
							: (FLMUINT)(uiMem * uiPercent) / 100);

	// Don't go above the maximum.

	if (uiMem > uiMax)
	{
		uiMem = uiMax;
	}

	// Don't go below the minimum.

	if (uiMem < uiMin)
	{
		uiMem = uiMin;
	}
	return( uiMem);
}
#endif

/***************************************************************************
Desc:	Verify that the distance (in bytes) between pvStart and pvEnd is
		what was specified in uiOffset.
****************************************************************************/
FINLINE void flmVerifyOffset(
	FLMUINT		uiCompilerOffset,
	FLMUINT		uiOffset,
	RCODE *		pRc
	)
{
	if (RC_OK( *pRc))
	{
		if ( uiCompilerOffset != uiOffset)
		{
			*pRc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PLATFORM_FORMAT);
		}
	}
}

/***************************************************************************
Desc:	Verify the offsets of each member of every on-disk structure.  This
		is a safety check to ensure that things work correctly on every
		platform.
****************************************************************************/
FSTATIC RCODE flmVerifyDiskStructOffsets( void)
{
	RCODE						rc = NE_XFLM_OK;
	FLMUINT					uiSizeOf;

	// Verify the XFLM_DB_HDR offsets.

	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, szSignature[0]),
						  XFLM_DB_HDR_szSignature_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui8IsLittleEndian),
						  XFLM_DB_HDR_ui8IsLittleEndian_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui8DefaultLanguage),
						  XFLM_DB_HDR_ui8DefaultLanguage_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui16BlockSize),
						  XFLM_DB_HDR_ui16BlockSize_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32DbVersion),
						  XFLM_DB_HDR_ui32DbVersion_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui8BlkChkSummingEnabled),
						  XFLM_DB_HDR_ui8BlkChkSummingEnabled_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui8RflKeepFiles),
						  XFLM_DB_HDR_ui8RflKeepFiles_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui8RflAutoTurnOffKeep),
						  XFLM_DB_HDR_ui8RflAutoTurnOffKeep_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui8RflKeepAbortedTrans),
						  XFLM_DB_HDR_ui8RflKeepAbortedTrans_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflCurrFileNum),
						  XFLM_DB_HDR_ui32RflCurrFileNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui64LastRflCommitID),
						  XFLM_DB_HDR_ui64LastRflCommitID_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflLastFileNumDeleted),
						  XFLM_DB_HDR_ui32RflLastFileNumDeleted_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflLastTransOffset),
						  XFLM_DB_HDR_ui32RflLastTransOffset_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflLastCPFileNum),
						  XFLM_DB_HDR_ui32RflLastCPFileNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflLastCPOffset),
						  XFLM_DB_HDR_ui32RflLastCPOffset_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui64RflLastCPTransID),
						  XFLM_DB_HDR_ui64RflLastCPTransID_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflMinFileSize),
						  XFLM_DB_HDR_ui32RflMinFileSize_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RflMaxFileSize),
						  XFLM_DB_HDR_ui32RflMaxFileSize_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui64CurrTransID),
						  XFLM_DB_HDR_ui64CurrTransID_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui64TransCommitCnt),
						  XFLM_DB_HDR_ui64TransCommitCnt_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RblEOF),
						  XFLM_DB_HDR_ui32RblEOF_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32RblFirstCPBlkAddr),
						  XFLM_DB_HDR_ui32RblFirstCPBlkAddr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32FirstAvailBlkAddr),
						  XFLM_DB_HDR_ui32FirstAvailBlkAddr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32FirstLFBlkAddr),
						  XFLM_DB_HDR_ui32FirstLFBlkAddr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32LogicalEOF),
						  XFLM_DB_HDR_ui32LogicalEOF_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32MaxFileSize),
						  XFLM_DB_HDR_ui32MaxFileSize_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui64LastBackupTransID),
						  XFLM_DB_HDR_ui64LastBackupTransID_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32IncBackupSeqNum),
						  XFLM_DB_HDR_ui32IncBackupSeqNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32BlksChangedSinceBackup),
						  XFLM_DB_HDR_ui32BlksChangedSinceBackup_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ucDbSerialNum[0]),
						  XFLM_DB_HDR_ucDbSerialNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ucLastTransRflSerialNum[0]),
						  XFLM_DB_HDR_ucLastTransRflSerialNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ucNextRflSerialNum[0]),
						  XFLM_DB_HDR_ucNextRflSerialNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ucIncBackupSerialNum[0]),
						  XFLM_DB_HDR_ucIncBackupSerialNum_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32DbKeyLen),
						  XFLM_DB_HDR_ui32DbKeyLen, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ucReserved[0]),
						  XFLM_DB_HDR_ucReserved_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, ui32HdrCRC),
						  XFLM_DB_HDR_ui32HdrCRC_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(XFLM_DB_HDR, DbKey[0]),
						  XFLM_DB_HDR_DbKey, &rc);

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = XFLM_DB_HDR_DbKey + XFLM_MAX_ENC_KEY_SIZE;
	if( sizeof( XFLM_DB_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PLATFORM_FORMAT);
	}

	// Verify the offsets in the F_BLK_HDR structure.

	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui32BlkAddr),
						  F_BLK_HDR_ui32BlkAddr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui32PrevBlkInChain),
						  F_BLK_HDR_ui32PrevBlkInChain_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui32NextBlkInChain),
						  F_BLK_HDR_ui32NextBlkInChain_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui32PriorBlkImgAddr),
						  F_BLK_HDR_ui32PriorBlkImgAddr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui64TransID),
						  F_BLK_HDR_ui64TransID_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui32BlkCRC),
						  F_BLK_HDR_ui32BlkCRC_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui16BlkBytesAvail),
						  F_BLK_HDR_ui16BlkBytesAvail_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui8BlkFlags),
						  F_BLK_HDR_ui8BlkFlags_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.stdBlkHdr.ui8BlkType),
						  F_BLK_HDR_ui8BlkType_OFFSET, &rc);

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = SIZEOF_STD_BLK_HDR;
	if (sizeof( F_BLK_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PLATFORM_FORMAT);
	}

	// Verify the offsets in the F_BTREE_BLK_HDR structure

	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.BTreeBlkHdr.stdBlkHdr),
						  F_BTREE_BLK_HDR_stdBlkHdr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.BTreeBlkHdr.ui16LogicalFile),
						  F_BTREE_BLK_HDR_ui16LogicalFile_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.BTreeBlkHdr.ui16NumKeys),
						  F_BTREE_BLK_HDR_ui16NumKeys_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.BTreeBlkHdr.ui8BlkLevel),
						  F_BTREE_BLK_HDR_ui8BlkLevel_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.BTreeBlkHdr.ui8BTreeFlags),
						  F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LARGEST_BLK_HDR, all.BTreeBlkHdr.ui16HeapSize),
						  F_BTREE_BLK_HDR_ui16HeapSize_OFFSET, &rc);

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = 40;
	if (sizeof( F_BTREE_BLK_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PLATFORM_FORMAT);
	}

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = 40;
	if (sizeof( F_LARGEST_BLK_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PLATFORM_FORMAT);
	}

	// Verify the offsets in the F_LF_HDR structure

	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui32LfNumber),
						  F_LF_HDR_ui32LfNumber_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui32LfType),
						  F_LF_HDR_ui32LfType_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui32RootBlkAddr),
						  F_LF_HDR_ui32RootBlkAddr_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui32EncId),
						  F_LF_HDR_ui32EncId_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui64NextNodeId),
						  F_LF_HDR_ui64NextNodeId_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui64FirstDocId),
						  F_LF_HDR_ui64FirstDocId_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ui64LastDocId),
						  F_LF_HDR_ui64LastDocId_OFFSET, &rc);
	flmVerifyOffset( (FLMUINT)offsetof(F_LF_HDR, ucZeroes[0]),
						  F_LF_HDR_ucZeroes_OFFSET, &rc);

	// Have to use a variable for sizeof.  If we don't, compiler barfs
	// because we are comparing two constants.

	uiSizeOf = 64;
	if (sizeof( F_LF_HDR) != uiSizeOf)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_PLATFORM_FORMAT);
	}

	return( rc);
}

/****************************************************************************
Desc:		Logs the reason for the "must close" flag being set
****************************************************************************/
void F_Database::logMustCloseReason(
	const char *	pszFileName,
	FLMINT			iLineNumber)
{
	char *						pszMsgBuf = NULL;
	IF_LogMessageClient *	pLogMsg = NULL;

	// Log a message indicating why the "must close" flag was set

	if( (pLogMsg = flmBeginLogMessage( XFLM_GENERAL_MESSAGE)) != NULL)
	{
		if( RC_OK( f_alloc( F_PATH_MAX_SIZE + 512, &pszMsgBuf)))
		{
			f_sprintf( (char *)pszMsgBuf,
				"Database (%s) must be closed because of a 0x%04X error, "
				"File=%s, Line=%d.",
				(char *)(m_pszDbPath ? (char *)m_pszDbPath : (char *)""),
				(unsigned)m_rcMustClose,
				pszFileName, (int)iLineNumber);

			pLogMsg->changeColor( XFLM_YELLOW, XFLM_BLACK);
			pLogMsg->appendString( pszMsgBuf);
		}
		flmEndLogMessage( &pLogMsg);
	}

	if( pszMsgBuf)
	{
		f_free( &pszMsgBuf);
	}
}

/****************************************************************************
Desc:	Acquires a write lock on the database
****************************************************************************/
RCODE F_Database::dbWriteLock(
	F_SEM					hWaitSem,
	XFLM_DB_STATS *	pDbStats,
	FLMUINT				uiTimeout)
{
	RCODE					rc = NE_XFLM_OK;

	if (RC_BAD( rc = m_pWriteLockObj->Lock( NULL, hWaitSem, FALSE,
		(FLMBOOL)(uiTimeout ? TRUE : FALSE),
		TRUE, uiTimeout, 0, pDbStats)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: This routine unlocks the write lock on a database.
****************************************************************************/
void F_Database::dbWriteUnlock(
	XFLM_DB_STATS *		pDbStats)
{
	(void)m_pWriteLockObj->Unlock( FALSE, NULL, FALSE, pDbStats);
}

/****************************************************************************
Desc: This shuts down the background threads
Note:	This routine assumes that the global mutex is locked.  The mutex will
		be unlocked internally, but will always be locked on exit.
****************************************************************************/
void F_Database::shutdownDatabaseThreads( void)
{
	RCODE					rc = NE_XFLM_OK;
	F_BKGND_IX	*		pBackgroundIx;
	F_Db *				pDb;
	F_Thread *			pThread;
	FLMUINT				uiThreadId;
	FLMUINT				uiThreadCount;
	FLMBOOL				bMutexLocked = TRUE;

	// Signal all background indexing threads to shutdown and all
	// threads in the FLM_DB_THREAD_GROUP that are associated with
	// this F_Database.

	for( ;;)
	{
		uiThreadCount = 0;

		// Shut down all background threads.

		uiThreadId = 0;
		for( ;;)
		{
			if( RC_BAD( rc = gv_XFlmSysData.pThreadMgr->getNextGroupThread(
				&pThread, FLM_BACKGROUND_INDEXING_THREAD_GROUP, &uiThreadId)))
			{
				if( rc == NE_XFLM_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					RC_UNEXPECTED_ASSERT( rc);
				}
			}
			else
			{
				pBackgroundIx = (F_BKGND_IX *)pThread->getParm1();
				if( pBackgroundIx && pBackgroundIx->pDatabase == this)
				{
					// Set the thread's terminate flag.

					uiThreadCount++;
					pThread->setShutdownFlag();
				}

				pThread->Release();
				pThread = NULL;
			}
		}

		// Shut down all threads in the FLM_DB_THREAD_GROUP.

		uiThreadId = 0;
		for( ;;)
		{
			if( RC_BAD( rc = gv_XFlmSysData.pThreadMgr->getNextGroupThread(
				&pThread, FLM_DB_THREAD_GROUP, &uiThreadId)))
			{
				if( rc == NE_XFLM_NOT_FOUND)
				{
					rc = NE_XFLM_OK;
					break;
				}
				else
				{
					RC_UNEXPECTED_ASSERT( rc);
				}
			}
			else
			{
				pDb = (F_Db *)pThread->getParm2();
				if (pDb && pDb->m_pDatabase == this)
				{

					// Set the thread's terminate flag.

					uiThreadCount++;
					pThread->setShutdownFlag();
				}

				pThread->Release();
				pThread = NULL;
			}
		}

		if( !uiThreadCount)
		{
			break;
		}

		// Unlock the global mutex

		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
		bMutexLocked = FALSE;

		// Give the threads a chance to terminate

		f_sleep( 50);

		// Re-lock the mutex and see if any threads are still active

		f_mutexLock( gv_XFlmSysData.hShareMutex);
		bMutexLocked = TRUE;

	}
	
	// Shut down the maintenance thread

	if( m_pMaintThrd)
	{
		flmAssert( bMutexLocked);
		
		m_pMaintThrd->setShutdownFlag();
		f_semSignal( m_hMaintSem);
		
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
		m_pMaintThrd->stopThread();
		f_mutexLock( gv_XFlmSysData.hShareMutex);
		
		m_pMaintThrd->Release();
		m_pMaintThrd = NULL;
		f_semDestroy( &m_hMaintSem);
	}

	// Re-lock the mutex

	if( !bMutexLocked)
	{
		f_mutexLock( gv_XFlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc: This routine frees a registered event.
****************************************************************************/
FSTATIC void flmFreeEvent(
	FEVENT *			pEvent,
	F_MUTEX			hMutex,
	FEVENT **		ppEventListRV)
{
	pEvent->pEventClient->Release();
	f_mutexLock( hMutex);
	if (pEvent->pPrev)
	{
		pEvent->pPrev->pNext = pEvent->pNext;
	}
	else
	{
		*ppEventListRV = pEvent->pNext;
	}
	if (pEvent->pNext)
	{
		pEvent->pNext->pPrev = pEvent->pPrev;
	}
	f_mutexUnlock( hMutex);
	f_free( &pEvent);
}

/****************************************************************************
Desc: This routine determines the hash bucket for a string.
****************************************************************************/
FLMUINT flmStrHashBucket(
	char *		pszStr,			// Pointer to string to be hashed
	FBUCKET *	pHashTbl,		// Hash table to use
	FLMUINT		uiNumBuckets)	// Number of hash buckets
{
	FLMUINT	uiHashIndex;

	if ((uiHashIndex = (FLMUINT)*pszStr) >= uiNumBuckets)
	{
		uiHashIndex -= uiNumBuckets;
	}

	while (*pszStr)
	{
		if ((uiHashIndex =
			(FLMUINT)((pHashTbl [uiHashIndex].uiHashValue) ^ (FLMUINT)(f_toupper( *pszStr)))) >=
				uiNumBuckets)
		{
			uiHashIndex -= uiNumBuckets;
		}
		pszStr++;
	}

	return( uiHashIndex);
}

/****************************************************************************
Desc:		This routine links a notify request into a notification list and
			then waits to be notified that the event has occurred.
Notes:	This routine assumes that the shared mutex is locked and that
			it is supposed to unlock it.
****************************************************************************/
RCODE flmWaitNotifyReq(
	F_MUTEX			hMutex,
	F_SEM				hSem,
	FNOTIFY **		ppNotifyListRV,
	void *			pvUserData)
{
	RCODE          rc = NE_XFLM_OK;
	RCODE          TempRc;
	FNOTIFY      	notifyItem;

	notifyItem.hSem = hSem;
	notifyItem.uiThreadId = f_threadId();
	notifyItem.pRc = &rc;
	notifyItem.pvUserData = pvUserData;
	notifyItem.pNext = *ppNotifyListRV;
	*ppNotifyListRV = &notifyItem;

	f_mutexUnlock( hMutex);
	
	if( RC_BAD( TempRc = f_semWait( notifyItem.hSem, F_SEM_WAITFOREVER)))
	{
		rc = TempRc;
	}

	f_mutexLock( hMutex);
	return( rc);
}

/****************************************************************************
Desc: This routine links an F_Database structure to its name hash bucket.
		NOTE: This function assumes that the global mutex has been
		locked.
****************************************************************************/
RCODE F_Database::linkToBucket( void)
{
	RCODE				rc = NE_XFLM_OK;
	F_Database *	pTmpDatabase;
	FBUCKET *		pBucket;
	FLMUINT			uiBucket;

	pBucket = gv_XFlmSysData.pDatabaseHashTbl;
	uiBucket = flmStrHashBucket( m_pszDbPath, pBucket, FILE_HASH_ENTRIES);
	pBucket = &pBucket [uiBucket];
	if (pBucket->pFirstInBucket)
	{
		pTmpDatabase = (F_Database *)pBucket->pFirstInBucket;
		pTmpDatabase->m_pPrev = this;
	}

	m_uiBucket = uiBucket;
	m_pPrev = NULL;
	m_pNext = (F_Database *)pBucket->pFirstInBucket;
	pBucket->pFirstInBucket = this;

	return( rc);
}

/****************************************************************************
Desc: This routine checks unused structures to see if any have been unused
		longer than the maximum unused time.  If so, it frees them up.
Note: This routine assumes that the calling routine has locked the global
		mutex prior to calling this routine.  The mutex may be unlocked and
		re-locked by one of the called routines.
****************************************************************************/
void F_DbSystem::checkNotUsedObjects( void)
{

	// Look for unused file handles

	if( gv_XFlmSysData.pFileHdlMgr)
	{
		gv_XFlmSysData.pFileHdlMgr->checkAgedFileHdls(
			gv_XFlmSysData.pFileHdlMgr->getMaxAvailTime());
	}
}

/****************************************************************************
Desc: This routine links an FDB structure to an F_Database structure.
		NOTE: This routine assumes that the global mutex has been
		locked.
****************************************************************************/
RCODE F_Db::linkToDatabase(
	F_Database *	pDatabase
	)
{
	RCODE			rc = NE_XFLM_OK;

	// If the use count on the file used to be zero, unlink it from the
	// unused list.

	flmAssert( !m_pDatabase);
	m_pPrevForDatabase = NULL;
	if ((m_pNextForDatabase = pDatabase->m_pFirstDb) != NULL)
	{
		pDatabase->m_pFirstDb->m_pPrevForDatabase = this;
	}

	pDatabase->m_pFirstDb = this;
	m_pDatabase = pDatabase;

	if (!(m_uiFlags & FDB_INTERNAL_OPEN))
	{
		pDatabase->incrOpenCount();
	}

	// Allocate the super file object

	if (!m_pSFileHdl)
	{
		if ((m_pSFileHdl = f_new F_SuperFileHdl) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}

		// Set up the super file

		if( RC_BAD( rc = m_pSFileHdl->Setup( pDatabase->m_pFileIdList,
			pDatabase->m_pszDbPath, pDatabase->m_pszDataDir)))
		{
			goto Exit;
		}

		if( pDatabase->m_lastCommittedDbHdr.ui32DbVersion)
		{
			m_pSFileHdl->SetBlockSize( pDatabase->m_uiBlockSize);
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc: This routine unlinks F_Db object from its F_Database structure.
		NOTE: This routine assumes that the global mutex has been
		locked.
****************************************************************************/
void F_Db::unlinkFromDatabase( void)
{
	if (!m_pDatabase)
	{
		return;
	}

	// Unlink the F_Db from the F_Database.

	if (m_pNextForDatabase)
	{
		m_pNextForDatabase->m_pPrevForDatabase = m_pPrevForDatabase;
	}

	if (m_pPrevForDatabase)
	{
		m_pPrevForDatabase->m_pNextForDatabase = m_pNextForDatabase;
	}
	else
	{
		m_pDatabase->m_pFirstDb = m_pNextForDatabase;
	}
	m_pNextForDatabase = m_pPrevForDatabase = NULL;

	// Decrement use counts in the F_Database, unless this was
	// an internal open.

	if (!(m_uiFlags & FDB_INTERNAL_OPEN))
	{
		m_pDatabase->decrOpenCount();
	}
	m_pDatabase = NULL;
}

/****************************************************************************
Desc: This routine dynamically adjusts the cache limit if that is the mode
		we are in.
****************************************************************************/
RCODE F_GlobalCacheMgr::adjustCache(
	FLMUINT *	puiCurrTime,
	FLMUINT *	puiLastCacheAdjustTime)
{
#ifndef FLM_CAN_GET_PHYS_MEM
	F_UNREFERENCED_PARM( puiCurrTime);
	F_UNREFERENCED_PARM( puiLastCacheAdjustTime);
	return( RC_SET( NE_XFLM_NOT_IMPLEMENTED));
#else
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCurrTime = *puiCurrTime;
	FLMUINT	uiLastCacheAdjustTime = *puiLastCacheAdjustTime;
	
	if (m_bDynamicCacheAdjust &&
		 FLM_ELAPSED_TIME( uiCurrTime, uiLastCacheAdjustTime) >=
			m_uiCacheAdjustInterval)
	{
		FLMUINT	uiCacheBytes;
		
		lockMutex();

		// Make sure the dynamic adjust flag is still set.

		if (m_bDynamicCacheAdjust &&
			 FLM_ELAPSED_TIME( uiCurrTime, uiLastCacheAdjustTime) >=
				m_uiCacheAdjustInterval)
		{
			uiCacheBytes = flmGetCacheBytes( m_uiCacheAdjustPercent,
									m_uiCacheAdjustMin, m_uiCacheAdjustMax,
									m_uiCacheAdjustMinToLeave, TRUE, totalBytes());
			if (RC_BAD( rc = setCacheLimit( uiCacheBytes, FALSE)))
			{
				unlockMutex();
				goto Exit;
			}
		}
		unlockMutex();
		*puiCurrTime = *puiLastCacheAdjustTime = FLM_GET_TIMER();
	}
	
Exit:

	return( rc);
#endif
}

/****************************************************************************
Desc: This routine functions as a thread.  It monitors open files and
		frees up files which have been closed longer than the maximum
		close time.
****************************************************************************/
RCODE F_DbSystem::monitorThrd(
	F_Thread *		pThread)
{
	FLMUINT		uiLastUnusedCleanupTime = 0;
	FLMUINT		uiCurrTime;
#ifdef FLM_CAN_GET_PHYS_MEM
	FLMUINT		uiLastCacheAdjustTime = 0;
#endif

	for (;;)
	{
		// See if we should shut down

		if( pThread->getShutdownFlag())
		{
			break;
		}

		uiCurrTime = FLM_GET_TIMER();

		// Check the not used stuff and lock timeouts.

		if ( FLM_ELAPSED_TIME( uiCurrTime, uiLastUnusedCleanupTime) >=
					gv_XFlmSysData.pGlobalCacheMgr->m_uiUnusedCleanupInterval)
		{
			// See if any unused structures have bee unused longer than the
			// maximum unused time.  Free them if they have.
			// May unlock and re-lock the global mutex.

			f_mutexLock( gv_XFlmSysData.hShareMutex);
			F_DbSystem::checkNotUsedObjects();
			f_mutexUnlock( gv_XFlmSysData.hShareMutex);

			// Reset the timer

			uiCurrTime = uiLastUnusedCleanupTime = FLM_GET_TIMER();
		}

		// Call the lock manager to check timeouts.  It is critial
		// that this routine be called on a regular interval to
		// timeout lock waiters that have expired.

		gv_XFlmSysData.pServerLockMgr->CheckLockTimeouts( FALSE, FALSE);

		// Check the adjusting cache limit

#ifdef FLM_CAN_GET_PHYS_MEM
		(void)gv_XFlmSysData.pGlobalCacheMgr->adjustCache( &uiCurrTime,
								&uiLastCacheAdjustTime);
#endif

		f_sleep( 1000);
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_DbSystem::cacheCleanupThrd(
	F_Thread *		pThread)
{
	FLMUINT		uiCurrTime;
	FLMUINT		uiLastDefragTime = 0;
	FLMUINT		uiLastCleanupTime = 0;
	FLMUINT		uiDefragInterval;
	FLMUINT		uiCleanupInterval = gv_XFlmSysData.pGlobalCacheMgr->m_uiCacheCleanupInterval;
	FLMBOOL		bDoNodeCacheFirst = TRUE;
	
	FLM_SECS_TO_TIMER_UNITS( 120, uiDefragInterval);
	
	for (;;)
	{
		if( pThread->getShutdownFlag())
		{
			break;
		}
		
		uiCurrTime = FLM_GET_TIMER();
		
		// Alternate between reducing node cache and block cache first.
		
		if (gv_XFlmSysData.pGlobalCacheMgr->cacheOverLimit() ||
			FLM_ELAPSED_TIME( uiCurrTime, uiLastCleanupTime) >= uiCleanupInterval)
		{
			if (bDoNodeCacheFirst)
			{
				f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
				gv_XFlmSysData.pNodeCacheMgr->reduceCache();
				f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
				
				f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
				(void)gv_XFlmSysData.pBlockCacheMgr->reduceCache( NULL);
				f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);
				
				bDoNodeCacheFirst = FALSE;
			}
			else
			{
				f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
				(void)gv_XFlmSysData.pBlockCacheMgr->reduceCache( NULL);
				f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);

				f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
				gv_XFlmSysData.pNodeCacheMgr->reduceCache();
				f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
				
				bDoNodeCacheFirst = TRUE;
			}

			uiLastCleanupTime = FLM_GET_TIMER();
		}
		
		if( FLM_ELAPSED_TIME( uiCurrTime, uiLastDefragTime) >= uiDefragInterval)
		{
			gv_XFlmSysData.pBlockCacheMgr->defragmentMemory();
			gv_XFlmSysData.pNodeCacheMgr->defragmentMemory();
			uiLastDefragTime = FLM_GET_TIMER();
		}

		f_sleep( 500);
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc: This routine does an event callback.  Note that the mutex is
		locked during the callback.
****************************************************************************/
void flmDoEventCallback(
	eEventCategory	eCategory,
	eEventType			eEvent,
	IF_Db *				pDb,
	FLMUINT				uiThreadId,
	FLMUINT64			ui64TransID,
	FLMUINT				uiIndexOrCollection,
	FLMUINT64			ui64NodeId,
	RCODE					rc)
{
	FEVENT *	pEvent;

	f_mutexLock( gv_XFlmSysData.EventHdrs [eCategory].hMutex);
	pEvent = gv_XFlmSysData.EventHdrs [eCategory].pEventCBList;
	while (pEvent)
	{
		pEvent->pEventClient->catchEvent( eEvent, pDb, uiThreadId, ui64TransID,
										uiIndexOrCollection, ui64NodeId, rc);
		pEvent = pEvent->pNext;
	}
	f_mutexUnlock( gv_XFlmSysData.EventHdrs [eCategory].hMutex);
}

/****************************************************************************
Desc:		This routine sets the "must close" flags on the
			F_Database and its FDBs
****************************************************************************/
void F_Database::setMustCloseFlags(
	RCODE				rcMustClose,
	FLMBOOL			bMutexLocked)
{
	F_Db *				pTmpDb;

	if( !bMutexLocked)
	{
		f_mutexLock( gv_XFlmSysData.hShareMutex);
	}

	if( !m_bMustClose)
	{
		m_bMustClose = TRUE;
		m_rcMustClose = rcMustClose;
		pTmpDb = m_pFirstDb;
		while( pTmpDb)
		{
			pTmpDb->m_bMustClose = TRUE;
			pTmpDb = pTmpDb->m_pNextForDatabase;
		}

		// Log a message indicating why the "must close" flag has been
		// set.  Calling checkState with the bMustClose flag
		// already set to TRUE will cause a message to be logged.

		(void)checkState( __FILE__, __LINE__);
	}

	if( !bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);
	}
}

/****************************************************************************
Desc:	Constructor
****************************************************************************/
F_HashTable::F_HashTable()
{
	m_hMutex = F_MUTEX_NULL;
	m_pGlobalList = NULL;
	m_ppHashTable = NULL;
	m_uiBuckets = 0;
	m_pCRCTable = NULL;
	m_bOwnCRCTable = FALSE;
}

/****************************************************************************
Desc:	Destructor
****************************************************************************/
F_HashTable::~F_HashTable()
{
	F_HashObject *		pCur;
	F_HashObject *		pNext;

	pCur = m_pGlobalList;
	while( pCur)
	{
		pNext = pCur->m_pNextInGlobal;
		unlinkObject( pCur);
		pCur->Release();
		pCur = pNext;
	}

	if( m_ppHashTable)
	{
		f_free( &m_ppHashTable);
	}

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}

	if( m_pCRCTable && m_bOwnCRCTable)
	{
		f_freeCRCTable( &m_pCRCTable);
	}
}

/****************************************************************************
Desc:	Configures the hash table prior to first use
****************************************************************************/
RCODE F_HashTable::setupHashTable(
	FLMBOOL			bMultithreaded,
	FLMUINT			uiNumBuckets,
	FLMUINT32 *		pCRCTable)
{
	RCODE			rc = NE_XFLM_OK;

	flmAssert( uiNumBuckets);

	// Create the hash table

	if( RC_BAD( rc = f_alloc( 
		sizeof( F_HashObject *) * uiNumBuckets, &m_ppHashTable)))
	{
		goto Exit;
	}

	m_uiBuckets = uiNumBuckets;
	f_memset( m_ppHashTable, 0, sizeof( F_HashObject *) * uiNumBuckets);

	if( bMultithreaded)
	{
		// Initialize the mutex

		if( RC_BAD( rc = f_mutexCreate( &m_hMutex)))
		{
			goto Exit;
		}
	}

	if( !pCRCTable)
	{
		// Initialize the CRC table

		if( RC_BAD( rc = f_initCRCTable( &m_pCRCTable)))
		{
			goto Exit;
		}
		m_bOwnCRCTable = TRUE;
	}
	else
	{
		m_pCRCTable = pCRCTable;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Retrieves an object from the hash table with the specified key.
		This routine assumes the table's mutex has already been locked.
		A reference IS NOT added to the object for the caller.
****************************************************************************/
RCODE F_HashTable::findObject(
	void *				pvKey,
	FLMUINT				uiKeyLen,
	F_HashObject **	ppObject)
{
	F_HashObject *		pObject = NULL;
	FLMUINT				uiBucket;
	FLMUINT32			ui32CRC = 0;
	RCODE					rc = NE_XFLM_OK;

	*ppObject = NULL;

	// Calculate the hash bucket and mutex offset

	uiBucket = getHashBucket( pvKey, uiKeyLen, &ui32CRC);

	// Search the bucket for an object with a matching
	// key.

	pObject = m_ppHashTable[ uiBucket];
	while( pObject)
	{
		if( pObject->getKeyCRC() == ui32CRC)
		{
			void *		pvTmpKey;
			FLMUINT		uiTmpKeyLen;

			pvTmpKey = pObject->getKey( &uiTmpKeyLen);
			if( uiTmpKeyLen == uiKeyLen &&
				f_memcmp( pvTmpKey, pvKey, uiKeyLen) == 0)
			{
				break;
			}
		}
		pObject = pObject->m_pNextInBucket;
	}

	if( !pObject)
	{
		rc = RC_SET( NE_XFLM_NOT_FOUND);
		goto Exit;
	}

	*ppObject = pObject;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Adds an object to the hash table
****************************************************************************/
RCODE F_HashTable::addObject(
	F_HashObject *		pObject)
{
	FLMUINT				uiBucket;
	F_HashObject *		pTmp;
	void *				pvKey;
	FLMUINT				uiKeyLen;
	FLMUINT32			ui32CRC;
	FLMBOOL				bMutexLocked = FALSE;
	RCODE					rc = NE_XFLM_OK;

	// Calculate and set the objects hash bucket

	flmAssert( pObject->getHashBucket() == F_INVALID_HASH_BUCKET);

	pvKey = pObject->getKey( &uiKeyLen);
	flmAssert( uiKeyLen);

	uiBucket = getHashBucket( pvKey, uiKeyLen, &ui32CRC);
	pObject->setKeyCRC( ui32CRC);

	// Lock the mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	// Make sure the object doesn't already exist

	if( RC_BAD( rc = findObject( pvKey, uiKeyLen, &pTmp)))
	{
		if( rc != NE_XFLM_NOT_FOUND)
		{
			goto Exit;
		}
		rc = NE_XFLM_OK;
	}
	else
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_EXISTS);
		goto Exit;
	}

	// Add a reference to the object

	pObject->AddRef();

	// Link the object into the appropriate lists

	linkObject( pObject, uiBucket);

Exit:

	// Unlock the mutex

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the next object in the linked list of objects in the hash
		table.  If *ppObject == NULL, the first object will be returned.
****************************************************************************/
RCODE F_HashTable::getNextObjectInGlobal(
	F_HashObject **	ppObject)
{
	FLMBOOL				bMutexLocked = FALSE;
	F_HashObject *		pOldObj;
	RCODE					rc = NE_XFLM_OK;

	// Lock the mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	if( !(*ppObject))
	{
		*ppObject = m_pGlobalList;
	}
	else
	{
		pOldObj = *ppObject;
		*ppObject = (*ppObject)->m_pNextInGlobal;
		pOldObj->Release();
	}

	if( *ppObject == NULL)
	{
		rc = RC_SET( NE_XFLM_EOF_HIT);
		goto Exit;
	}

	(*ppObject)->AddRef();

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Retrieves an object from the hash table with the specified key
****************************************************************************/
RCODE F_HashTable::getObject(
	void *				pvKey,
	FLMUINT				uiKeyLen,
	F_HashObject **	ppObject,
	FLMBOOL				bRemove)
{
	F_HashObject *	pObject;
	FLMBOOL			bMutexLocked = FALSE;
	RCODE				rc = NE_XFLM_OK;

	// Lock the mutex

	if( m_hMutex != F_MUTEX_NULL)
	{
		f_mutexLock( m_hMutex);
		bMutexLocked = TRUE;
	}

	// Search for an object with a matching key.

	if( RC_BAD( rc = findObject( pvKey, uiKeyLen, &pObject)))
	{
		goto Exit;
	}

	if( pObject && bRemove)
	{
		unlinkObject( pObject);
		if( !ppObject)
		{
			pObject->Release();
			pObject = NULL;
		}
	}

	if( ppObject)
	{
		if( !bRemove)
		{
			pObject->AddRef();
		}
		*ppObject = pObject;
		pObject = NULL;
	}

Exit:

	if( bMutexLocked)
	{
		f_mutexUnlock( m_hMutex);
	}

	return( rc);
}

/****************************************************************************
Desc:	Removes an object from the hash table by key
****************************************************************************/
RCODE F_HashTable::removeObject(
	void *			pvKey,
	FLMUINT			uiKeyLen)
{
	return( getObject( pvKey, uiKeyLen, NULL, TRUE));
}

/****************************************************************************
Desc:	Removes an object from the hash table by object pointer
****************************************************************************/
RCODE F_HashTable::removeObject(
	F_HashObject *		pObject)
{
	FLMUINT		uiKeyLen;
	void *		pvKey = pObject->getKey( &uiKeyLen);

	return( getObject( pvKey, uiKeyLen, NULL, TRUE));
}

/****************************************************************************
Desc:	Calculates the hash bucket of a key and optionally returns the key's
		CRC.
****************************************************************************/
FLMUINT F_HashTable::getHashBucket(
	void *		pvKey,
	FLMUINT		uiLen,
	FLMUINT32 *	pui32KeyCRC)
{
	FLMUINT32	ui32CRC = 0;

	f_updateCRC( m_pCRCTable, (FLMBYTE *)pvKey, uiLen, &ui32CRC);
	if( pui32KeyCRC)
	{
		*pui32KeyCRC = ui32CRC;
	}
	return( ui32CRC % m_uiBuckets);
}

/****************************************************************************
Desc:		Links an object to the global list and also to its bucket
Notes:	This routine assumes that the bucket's mutex is already locked
			if the hash table is multi-threaded.
****************************************************************************/
void F_HashTable::linkObject(
	F_HashObject *		pObject,
	FLMUINT				uiBucket)
{
	flmAssert( uiBucket < m_uiBuckets);
	flmAssert( pObject->getHashBucket() == F_INVALID_HASH_BUCKET);

	// Set the object's bucket

	pObject->setHashBucket( uiBucket);

	// Link the object to its hash bucket

	pObject->m_pNextInBucket = m_ppHashTable[ uiBucket];
	if( m_ppHashTable[ uiBucket])
	{
		m_ppHashTable[ uiBucket]->m_pPrevInBucket = pObject;
	}
	m_ppHashTable[ uiBucket] = pObject;

	// Link to the global list

	pObject->m_pNextInGlobal = m_pGlobalList;
	if( m_pGlobalList)
	{
		m_pGlobalList->m_pPrevInGlobal = pObject;
	}
	m_pGlobalList = pObject;
}

/****************************************************************************
Desc:		Unlinks an object from its bucket and the global list.
Notes:	This routine assumes that the bucket's mutex is already locked
			if the hash table is multi-threaded.
****************************************************************************/
void F_HashTable::unlinkObject(
	F_HashObject *		pObject)
{
	FLMUINT		uiBucket = pObject->getHashBucket();

	// Is the bucket valid?

	flmAssert( uiBucket < m_uiBuckets);

	// Unlink from the hash bucket

	if( pObject->m_pNextInBucket)
	{
		pObject->m_pNextInBucket->m_pPrevInBucket = pObject->m_pPrevInBucket;
	}

	if( pObject->m_pPrevInBucket)
	{
		pObject->m_pPrevInBucket->m_pNextInBucket = pObject->m_pNextInBucket;
	}
	else
	{
		m_ppHashTable[ uiBucket] = pObject->m_pNextInBucket;
	}

	pObject->m_pPrevInBucket = NULL;
	pObject->m_pNextInBucket = NULL;
	pObject->setHashBucket( F_INVALID_HASH_BUCKET);

	// Unlink from the global list

	if( pObject->m_pNextInGlobal)
	{
		pObject->m_pNextInGlobal->m_pPrevInGlobal = pObject->m_pPrevInGlobal;
	}

	if( pObject->m_pPrevInGlobal)
	{
		pObject->m_pPrevInGlobal->m_pNextInGlobal = pObject->m_pNextInGlobal;
	}
	else
	{
		m_pGlobalList = pObject->m_pNextInGlobal;
	}

	pObject->m_pPrevInGlobal = NULL;
	pObject->m_pNextInGlobal = NULL;
}

/***************************************************************************
Desc:	Lock the system data structure for access - called only by startup
		and shutdown.  NOTE: On platforms that do not support atomic exchange
		this is less than perfect - won't handle tight race conditions.
***************************************************************************/
void F_DbSystem::lockSysData( void)
{
	// Obtain the spin lock

	while( flmAtomicExchange( &m_flmSysSpinLock, 1) == 1)
	{
		f_sleep( 10);
	}
}

/***************************************************************************
Desc:	Unlock the system data structure for access - called only by startup
		and shutdown.
***************************************************************************/
void F_DbSystem::unlockSysData( void)
{
	(void)flmAtomicExchange( &m_flmSysSpinLock, 0);
}

/****************************************************************************
Desc :	Constructor for global cache manager.
****************************************************************************/
F_GlobalCacheMgr::F_GlobalCacheMgr()
{
	m_pSlabManager = NULL;
	m_bCachePreallocated = FALSE;
	
#ifdef FLM_CAN_GET_PHYS_MEM
	m_bDynamicCacheAdjust = TRUE;
	m_uiCacheAdjustPercent = XFLM_DEFAULT_CACHE_ADJUST_PERCENT;
	m_uiCacheAdjustMin = XFLM_DEFAULT_CACHE_ADJUST_MIN;
	m_uiCacheAdjustMax = XFLM_DEFAULT_CACHE_ADJUST_MAX;
	m_uiCacheAdjustMinToLeave = XFLM_DEFAULT_CACHE_ADJUST_MIN_TO_LEAVE;
	FLM_SECS_TO_TIMER_UNITS( XFLM_DEFAULT_CACHE_ADJUST_INTERVAL,
		m_uiCacheAdjustInterval);
	m_uiMaxBytes = flmGetCacheBytes( m_uiCacheAdjustPercent,
												m_uiCacheAdjustMin,
												m_uiCacheAdjustMax,
												m_uiCacheAdjustMinToLeave, TRUE, 0);
#else
	m_bDynamicCacheAdjust = FALSE;
	m_uiCacheAdjustPercent = 0;
	m_uiCacheAdjustMin = 0;
	m_uiCacheAdjustMax = 0;
	m_uiCacheAdjustMinToLeave = 0;
	m_uiCacheAdjustInterval = 0;
	m_uiMaxBytes = XFLM_DEFAULT_CACHE_ADJUST_MIN;
#endif
	FLM_SECS_TO_TIMER_UNITS( XFLM_DEFAULT_CACHE_CLEANUP_INTERVAL,
		m_uiCacheCleanupInterval);
	FLM_SECS_TO_TIMER_UNITS( XFLM_DEFAULT_UNUSED_CLEANUP_INTERVAL,
		m_uiUnusedCleanupInterval);
	m_hMutex = F_MUTEX_NULL;
	m_uiMaxSlabs = 0;
}
	
/****************************************************************************
Desc:
****************************************************************************/
F_GlobalCacheMgr::~F_GlobalCacheMgr()
{
	if (m_pSlabManager)
	{
		m_pSlabManager->Release();
	}
	if (m_hMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &m_hMutex);
	}
}
	
/****************************************************************************
Desc:
****************************************************************************/
RCODE F_GlobalCacheMgr::setup( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBYTE		szTmpBuffer[ 64];
	
	flmAssert( !m_pSlabManager);
	
	if (RC_BAD( rc = f_mutexCreate( &m_hMutex)))
	{
		goto Exit;
	}
	
	if( (m_pSlabManager = f_new F_SlabManager) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	f_getenv( "XFLM_PREALLOC_CACHE_SIZE", szTmpBuffer, sizeof( szTmpBuffer));
	
	if( RC_BAD( rc = m_pSlabManager->setup( f_atoi( (char *)szTmpBuffer)))) 
	{
		m_pSlabManager->Release();
		m_pSlabManager = NULL;
		goto Exit;
	}
	
	// Need to set max slabs here because we didn't know the slab size
	// in the constructor.
	
	m_uiMaxSlabs = m_uiMaxBytes / m_pSlabManager->getSlabSize();

Exit:

	return( rc);
}

/****************************************************************************
Desc: This routine sets the limits for record cache and block cache - dividing
		the total cache between the two caches.  It uses the same ratio
		currently in force.
****************************************************************************/
RCODE F_GlobalCacheMgr::setCacheLimit(
	FLMUINT		uiNewTotalCacheSize,
	FLMBOOL		bPreallocateCache)
{
	RCODE			rc = NE_XFLM_OK;
	
	if( uiNewTotalCacheSize > FLM_MAX_CACHE_SIZE)
	{
		uiNewTotalCacheSize = FLM_MAX_CACHE_SIZE;
	}
	
	if( m_bDynamicCacheAdjust || !bPreallocateCache)
	{
DONT_PREALLOCATE:

		if( uiNewTotalCacheSize < m_uiMaxBytes)
		{
			m_uiMaxBytes = uiNewTotalCacheSize;
			m_uiMaxSlabs = m_uiMaxBytes / m_pSlabManager->getSlabSize();
			f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
			gv_XFlmSysData.pNodeCacheMgr->reduceCache();
			f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
			
			f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
			rc = gv_XFlmSysData.pBlockCacheMgr->reduceCache( NULL);
			f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);
			if( RC_BAD( rc))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = m_pSlabManager->resize( 0)))
			{
				goto Exit;
			}
		}
		
		m_bCachePreallocated = FALSE;
	}
	else
	{
		if( RC_BAD( m_pSlabManager->resize( 
			uiNewTotalCacheSize, &uiNewTotalCacheSize)))
		{
			goto DONT_PREALLOCATE;				
		}
		
		m_bCachePreallocated = TRUE;
	}

	m_uiMaxBytes = uiNewTotalCacheSize;
	m_uiMaxSlabs = m_uiMaxBytes / m_pSlabManager->getSlabSize();
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_GlobalCacheMgr::clearCache(
	IF_Db *		pDb)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiSavedMaxBytes;
	FLMUINT		uiSavedMaxSlabs;

	lockMutex();

	uiSavedMaxBytes = m_uiMaxBytes;
	uiSavedMaxSlabs = m_uiMaxSlabs;
	
	m_uiMaxBytes = 0;
	m_uiMaxSlabs = 0;

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	gv_XFlmSysData.pNodeCacheMgr->reduceCache();
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	
	f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
	rc = gv_XFlmSysData.pBlockCacheMgr->reduceCache( (F_Db *)pDb);
	f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);

	if( RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	m_uiMaxBytes = uiSavedMaxBytes;
	m_uiMaxSlabs = uiSavedMaxSlabs;
	unlockMutex();
	return( rc);
}

/****************************************************************************
Desc : Startup the database engine.
Notes: This routine may be called multiple times.  However, if that is done
		 exit() should be called for each time this is called successfully.
		 This routine does not handle race conditions on platforms that do
		 not support atomic increment.
****************************************************************************/
RCODE F_DbSystem::init( void)
{
	RCODE				rc = NE_XFLM_OK;
	FLMINT			iEventCategory;
#ifdef FLM_USE_NICI
	int				iHandle;
#endif

	lockSysData();

	// See if FLAIM has already been started.  If so,
	// we are done.

	if (++m_uiFlmSysStartupCount > 1)
	{
		goto Exit;
	}

#ifdef FLM_NLM
	gv_bNetWareStartupCalled = TRUE;
	if( RC_BAD( rc = f_netwareStartup()))
	{
		goto Exit;
	}
#endif

	// Sanity check -- make sure we are using the correct
	// byte-swap macros for this platform

	flmAssert( FB2UD( (FLMBYTE *)"\x0A\x0B\x0C\x0D") == 0x0D0C0B0A);
	flmAssert( FB2UW( (FLMBYTE *)"\x0A\x0B") == 0x0B0A);

	// The memset needs to be first.

	f_memset( &gv_XFlmSysData, 0, sizeof( FLMSYSDATA));
	
#if defined( FLM_LINUX)
	flmGetLinuxKernelVersion( &gv_XFlmSysData.uiLinuxMajorVer,
									  &gv_XFlmSysData.uiLinuxMinorVer, 
									  &gv_XFlmSysData.uiLinuxRevision);
	gv_XFlmSysData.uiMaxFileSize = flmGetLinuxMaxFileSize();
#elif defined( FLM_AIX)
	// Call set setrlimit to increase the max allowed file size.
	// We don't have a good way to deal with any errors returned by
	// setrlimit(), so we just hope that there aren't any...
	struct rlimit rlim;
	rlim.rlim_cur = RLIM_INFINITY;
	rlim.rlim_max = RLIM_INFINITY;
	setrlimit( RLIMIT_FSIZE, &rlim);
	gv_XFlmSysData.uiMaxFileSize = XFLM_MAXIMUM_FILE_SIZE;
#else
	gv_XFlmSysData.uiMaxFileSize = XFLM_MAXIMUM_FILE_SIZE;
#endif

	// Initialize memory tracking variables - should be done before
	// call to f_memoryInit().

#ifdef FLM_DEBUG

	// Variables for memory allocation tracking.

	gv_XFlmSysData.bTrackLeaks = TRUE;
	gv_XFlmSysData.hMemTrackingMutex = F_MUTEX_NULL;
#ifdef DEBUG_SIM_OUT_OF_MEM
	gv_XFlmSysData.memSimRandomGen.randomSetSeed( 1);
#endif
#endif

	gv_XFlmSysData.hNodeCacheMutex = F_MUTEX_NULL;
	gv_XFlmSysData.hBlockCacheMutex = F_MUTEX_NULL;
	gv_XFlmSysData.hShareMutex = F_MUTEX_NULL;
	gv_XFlmSysData.hStatsMutex = F_MUTEX_NULL;
	gv_XFlmSysData.hLoggerMutex = F_MUTEX_NULL;
	gv_XFlmSysData.hIniMutex = F_MUTEX_NULL;
	
	// Initialize the event categories to have no mutex.

	for (iEventCategory = 0;
		  iEventCategory < XFLM_MAX_EVENT_CATEGORIES;
		  iEventCategory++)
	{
		gv_XFlmSysData.EventHdrs [iEventCategory].hMutex = F_MUTEX_NULL;
	}

	// Memory initialization should be first.

	f_memoryInit();

#if defined( FLM_NLM) || (defined( FLM_WIN) && !defined( FLM_64BIT))
	initFastBlockCheckSum();
#endif

	if (RC_BAD( rc = flmVerifyDiskStructOffsets()))
	{
		goto Exit;
	}

	// Check the error code tables

	if( RC_BAD( rc = checkErrorCodeTables()))
	{
		goto Exit;
	}

	// Initialize all of the fields

	FLM_SECS_TO_TIMER_UNITS( XFLM_DEFAULT_MAX_UNUSED_TIME,
		gv_XFlmSysData.uiMaxUnusedTime);
	FLM_SECS_TO_TIMER_UNITS( XFLM_DEFAULT_MAX_CP_INTERVAL,
		gv_XFlmSysData.uiMaxCPInterval);
		
	FLM_SECS_TO_TIMER_UNITS( XFLM_DEFAULT_REHASH_BACKOFF_INTERVAL,
		gv_XFlmSysData.uiRehashAfterFailureBackoffTime);
		
	if ((gv_XFlmSysData.pGlobalCacheMgr = f_new F_GlobalCacheMgr) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = gv_XFlmSysData.pGlobalCacheMgr->setup()))
	{
		goto Exit;
	}

	// Initialize the thread manager

	if( (gv_XFlmSysData.pThreadMgr = f_new F_ThreadMgr) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = gv_XFlmSysData.pThreadMgr->setupThreadMgr()))
	{
		goto Exit;
	}

	// Initialize the serial number generator

	if( RC_BAD( rc = f_initSerialNumberGenerator()))
	{
		goto Exit;
	}

	// Initialize the Unicode<->WP mapping tables

	if( RC_BAD( rc = initCharMappingTables()))
	{
		goto Exit;
	}

	// Create the mutexes for controlling access to cache and global structures.

	if (RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hNodeCacheMutex)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hBlockCacheMutex)))
	{
		goto Exit;
	}
	if (RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hShareMutex)))
	{
		goto Exit;
	}

	if ((gv_XFlmSysData.pBlockCacheMgr = f_new F_BlockCacheMgr) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = gv_XFlmSysData.pBlockCacheMgr->initCache()))
	{
		goto Exit;
	}
	if ((gv_XFlmSysData.pNodeCacheMgr = f_new F_NodeCacheMgr) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = gv_XFlmSysData.pNodeCacheMgr->initCache()))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hQueryMutex)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hIniMutex)))
	{
		goto Exit;
	}
	
	// Initialize a statistics structure.

	if (RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hStatsMutex)))
	{
		goto Exit;
	}
	
	f_memset( &gv_XFlmSysData.Stats, 0, sizeof( XFLM_STATS));
	gv_XFlmSysData.bStatsInitialized = TRUE;
	
	// Initialize the logging mutex
	
	if( RC_BAD( rc = f_mutexCreate( &gv_XFlmSysData.hLoggerMutex)))
	{
		goto Exit;
	}

	// Allocate memory for the file name hash table.

	if (RC_BAD(rc = flmAllocHashTbl( FILE_HASH_ENTRIES,
								&gv_XFlmSysData.pDatabaseHashTbl)))
	{
		goto Exit;
	}

	// Allocate and Initialize FLAIM Shared File Handle Manager

	if ((gv_XFlmSysData.pFileHdlMgr = f_new F_FileHdlMgr) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = gv_XFlmSysData.pFileHdlMgr->setupFileHdlMgr(
										XFLM_DEFAULT_OPEN_THRESHOLD,
										XFLM_DEFAULT_MAX_UNUSED_TIME)))
	{
		goto Exit;
	}

	// Allocate and Initialize FLAIM Shared File System object

	if ((gv_pFileSystem = f_new F_FileSystem) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

#ifdef FLM_DBG_LOG
	flmDbgLogInit();
#endif

#if defined( FLM_WIN)
	OSVERSIONINFO		versionInfo;

	versionInfo.dwOSVersionInfoSize = sizeof( OSVERSIONINFO);
	if( !GetVersionEx( &versionInfo))
	{
		return( MapWinErrorToFlaim( GetLastError(), NE_XFLM_FAILURE));
	}

	// Async writes are not supported on Win32s (3.1) or
	// Win95, 98, ME, etc.

	gv_XFlmSysData.bOkToDoAsyncWrites =
		(versionInfo.dwPlatformId != VER_PLATFORM_WIN32_WINDOWS &&
		 versionInfo.dwPlatformId != VER_PLATFORM_WIN32s)
		 ? TRUE
		 : FALSE;
#else
	gv_XFlmSysData.bOkToDoAsyncWrites = TRUE;
#endif

	// Allocate and Initialize FLAIM Server Lock Manager.

	if ((gv_XFlmSysData.pServerLockMgr = f_new ServerLockManager) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	if (RC_BAD( rc = gv_XFlmSysData.pServerLockMgr->setupLockMgr()))
	{
		goto Exit;
	}

	// Set up hash table for lock manager.

	if (RC_BAD( rc = gv_XFlmSysData.pServerLockMgr->SetupHashTbl()))
	{
		goto Exit;
	}

	// Set up mutexes for the event table.

	for (iEventCategory = 0;
		  iEventCategory < XFLM_MAX_EVENT_CATEGORIES;
		  iEventCategory++)
	{
		if (RC_BAD( rc = f_mutexCreate(
								&gv_XFlmSysData.EventHdrs [iEventCategory].hMutex)))
		{
			goto Exit;
		}
	}

	// Start the monitor thread

	if (RC_BAD( rc = f_threadCreate( &gv_XFlmSysData.pMonitorThrd,
		F_DbSystem::monitorThrd, "DB Monitor")))
	{
		goto Exit;
	}

	// Start the cache cleanup thread

	if (RC_BAD( rc = f_threadCreate( &gv_XFlmSysData.pCacheCleanupThrd,
		F_DbSystem::cacheCleanupThrd, "Cache Cleanup Thread")))
	{
		goto Exit;
	}
	
	if ((gv_XFlmSysData.pBtPool = f_new F_BtPool()) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = gv_XFlmSysData.pBtPool->btpInit()))
	{
		goto Exit;
	}

	if ((gv_XFlmSysData.pNodePool = f_new F_NodePool) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}
	
	if (RC_BAD( rc = gv_XFlmSysData.pNodePool->setup()))
	{
		goto Exit;
	}

	if( (gv_XFlmSysData.pXml = f_new F_XML) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = gv_XFlmSysData.pXml->buildCharTable()))
	{
		goto Exit;
	}

	// Check the metaphone routines

#ifdef FLM_DEBUG
	if( RC_BAD( rc = flmVerifyMetaphoneRoutines()))
	{
		goto Exit;
	}
#endif

#ifdef FLM_USE_NICI
	iHandle  = f_getpid();

	// Initialize NICI

	if( RC_BAD( rc = CCS_Init( &iHandle)))
	{
		rc = RC_SET( NE_XFLM_NICI_INIT_FAILED);
		goto Exit;
	}
#endif

	// Initialize the XFlaim cache settings from the .ini file (if present)

	readIniFile();

#ifdef FLM_NLM
	// Detect if we are running in RING0.  If so, use NSS File Handles
#define MAX_ADDRSPACE_NAMELEN 63
	char			szAddrSpaceName[ MAX_ADDRSPACE_NAMELEN + 1];
	if( getaddressspacename( getaddressspace(), szAddrSpaceName) != NULL)
	{
		if( f_strcmp( szAddrSpaceName, "OS") == 0)
		{
			gv_XFlmSysData.bUseNSSFileHdls = TRUE;
		}
	}
#endif

Exit:

	// If not successful, free up any resources that were allocated.

	if (RC_BAD( rc))
	{
		cleanup();
	}

	unlockSysData();
	return( rc);
}

/****************************************************************************
Desc:		Shuts the database engine down.
Notes:	This routine allows itself to be called multiple times, even before
			init() is called or if  init() fails.  May not handle race
			conditions very well on platforms that do not support atomic
			exchange.
****************************************************************************/
void F_DbSystem::exit( void)
{
	lockSysData();
	cleanup();
	unlockSysData();
}

/************************************************************************
Desc : Cleans up - assumes that the spin lock has already been
		 obtained.  This allows it to be called directly from
		 init() on error conditions.
************************************************************************/
void F_DbSystem::cleanup( void)
{
	FLMUINT		uiCnt;
	FLMINT		iEventCategory;

	// NOTE: We are checking and decrementing a global variable here.
	// However, on platforms that properly support atomic exchange,
	// we are OK, because the caller has obtained a spin lock before
	// calling this routine, so we are guaranteed to be the only thread
	// executing this code at this point.  On platforms that don't
	// support atomic exchange, our spin lock will be less reliable for
	// really tight race conditions.  But in reality, nobody should be
	// calling FlmStartup and FlmShutdown in race conditions like that
	// anyway.  We are only doing the spin lock stuff to try and be
	// nice about it if they are.

	// This check allows FlmShutdown to be called before calling
	// FlmStartup, or even if FlmStartup fails.

	if (!m_uiFlmSysStartupCount)
	{
		return;
	}

	// If we decrement the count and it doesn't go to zero, we are not
	// ready to do cleanup yet.

	if (--m_uiFlmSysStartupCount > 0)
	{
		return;
	}

	// Free any queries that have been saved in the query list.

	if (gv_XFlmSysData.hQueryMutex != F_MUTEX_NULL)
	{

		// Setting uiMaxQueries to zero will cause flmFreeSavedQueries
		// to free the entire list.  Also, embedded queries will not be
		// added back into the list when uiMaxQueries is zero.

		gv_XFlmSysData.uiMaxQueries = 0;
		flmFreeSavedQueries( FALSE);
	}

	// Shut down the monitor thread, if there is one.

	f_threadDestroy( &gv_XFlmSysData.pMonitorThrd);
	
	// Shut down the cache reduce thread

	f_threadDestroy( &gv_XFlmSysData.pCacheCleanupThrd);
	
	// Free all of the files and associated structures

	if (gv_XFlmSysData.pDatabaseHashTbl)
	{
		FBUCKET *   pDatabaseHashTbl;

		// F_Database destructor expects the global mutex to be locked
		// IMPORTANT NOTE: pDatabaseHashTbl is ALWAYS allocated
		// AFTER the mutex is allocated, so we are guaranteed
		// to have a mutex if pDatabaseHashTbl is non-NULL.

		f_mutexLock( gv_XFlmSysData.hShareMutex);
		for (uiCnt = 0, pDatabaseHashTbl = gv_XFlmSysData.pDatabaseHashTbl;
			  uiCnt < FILE_HASH_ENTRIES;
			  uiCnt++, pDatabaseHashTbl++)
		{
			F_Database *	pDatabase = (F_Database *)pDatabaseHashTbl->pFirstInBucket;
			F_Database *	pTmpDatabase;

			while (pDatabase)
			{
				pTmpDatabase = pDatabase;
				pDatabase = pDatabase->m_pNext;
				pTmpDatabase->freeDatabase();
			}
			pDatabaseHashTbl->pFirstInBucket = NULL;
		}

		// Unlock the global mutex
		f_mutexUnlock( gv_XFlmSysData.hShareMutex);

		// Free the hash table
		f_free( &gv_XFlmSysData.pDatabaseHashTbl);
	}

	// Free the statistics.

	if (gv_XFlmSysData.bStatsInitialized)
	{
		f_mutexLock( gv_XFlmSysData.hStatsMutex);
		flmStatFree( &gv_XFlmSysData.Stats);
		f_mutexUnlock( gv_XFlmSysData.hStatsMutex);
		gv_XFlmSysData.bStatsInitialized = FALSE;
	}
	
	if (gv_XFlmSysData.hStatsMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hStatsMutex);
	}

	// Free (release) FLAIM's File Shared File Handles.

	if (gv_XFlmSysData.pFileHdlMgr)
	{
		FLMUINT	uiRefCnt = gv_XFlmSysData.pFileHdlMgr->Release();

		// No one else should have a reference to the file handle manager
		// after this point.

#ifdef FLM_DEBUG
		flmAssert( 0 == uiRefCnt);
#else
		// Quiet the compiler about the unused variable
		(void)uiRefCnt;
#endif
		gv_XFlmSysData.pFileHdlMgr = NULL;
	}

	// Free (release) FLAIM's Server Lock Manager.

	if (gv_XFlmSysData.pServerLockMgr)
	{
		FLMUINT	uiRefCnt;

		// Release all locks.

		gv_XFlmSysData.pServerLockMgr->CheckLockTimeouts( FALSE, TRUE);

		// Release the lock manager.

		uiRefCnt = gv_XFlmSysData.pServerLockMgr->Release();

		// No one else should have a reference to the server lock manager
		// at this point, so lets trip a flmAssert if the object was really
		// not deleted.

#ifdef FLM_DEBUG
		flmAssert( 0 == uiRefCnt);
#else
		// Quiet the compiler about the unused variable
		(void)uiRefCnt;
#endif
		gv_XFlmSysData.pServerLockMgr = NULL;
	}
	
	// Make sure the purge list is empty

	if( gv_XFlmSysData.pNodeCacheMgr->m_pPurgeList)
	{
		f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
		gv_XFlmSysData.pNodeCacheMgr->cleanupPurgedCache();
		f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
		flmAssert( !gv_XFlmSysData.pNodeCacheMgr->m_pPurgeList);
	}

	// Free the node cache manager

	if( gv_XFlmSysData.pNodeCacheMgr)
	{
		gv_XFlmSysData.pNodeCacheMgr->Release();
		gv_XFlmSysData.pNodeCacheMgr = NULL;
	}
	
	// Free the block cache manager

	if( gv_XFlmSysData.pBlockCacheMgr)
	{
		gv_XFlmSysData.pBlockCacheMgr->Release();
		gv_XFlmSysData.pBlockCacheMgr = NULL;
	}

	// Free up callbacks that have been registered for events.

	for (iEventCategory = 0;
		  iEventCategory < XFLM_MAX_EVENT_CATEGORIES;
		  iEventCategory++)
	{
		if (gv_XFlmSysData.EventHdrs [iEventCategory].hMutex != F_MUTEX_NULL)
		{
			while (gv_XFlmSysData.EventHdrs [iEventCategory].pEventCBList)
			{
				flmFreeEvent(
					gv_XFlmSysData.EventHdrs [iEventCategory].pEventCBList,
					gv_XFlmSysData.EventHdrs [iEventCategory].hMutex,
					&gv_XFlmSysData.EventHdrs [iEventCategory].pEventCBList);
			}
			f_mutexDestroy( &gv_XFlmSysData.EventHdrs [iEventCategory].hMutex);
		}
	}

	// Free (release) FLAIM's File Shared File System Object.

	if (gv_pFileSystem)
	{
		FLMUINT	uiRefCnt = gv_pFileSystem->Release();

		// No one else should have a reference to the file system
		// after this point.

#ifdef FLM_DEBUG
		flmAssert( 0 == uiRefCnt);
#else
		// Quiet the compiler about the unused variable
		(void)uiRefCnt;
#endif
		gv_pFileSystem = NULL;
	}
#ifdef FLM_DBG_LOG
	flmDbgLogExit();
#endif

	// Free the serial number generator

	f_freeSerialNumberGenerator();

	// Release the logger
	
	if( gv_XFlmSysData.pLogger)
	{
		flmAssert( !gv_XFlmSysData.uiPendingLogMessages);
		gv_XFlmSysData.pLogger->Release();
		gv_XFlmSysData.pLogger = NULL;
	}

	if( gv_XFlmSysData.hLoggerMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hLoggerMutex);
	}
	
	// Release the thread manager

	if( gv_XFlmSysData.pThreadMgr)
	{
		gv_XFlmSysData.pThreadMgr->Release();
		gv_XFlmSysData.pThreadMgr = NULL;
	}

	// Free the btree pool

	if( gv_XFlmSysData.pBtPool)
	{
		gv_XFlmSysData.pBtPool->Release();
		gv_XFlmSysData.pBtPool = NULL;
	}

	// Free the node pool

	if( gv_XFlmSysData.pNodePool)
	{
		gv_XFlmSysData.pNodePool->Release();
		gv_XFlmSysData.pNodePool = NULL;
	}

	// Free the XML object

	if( gv_XFlmSysData.pXml)
	{
		gv_XFlmSysData.pXml->Release();
		gv_XFlmSysData.pXml = NULL;
	}

	// Release the global cache manager.  NOTE: This must happen
	// only AFTER releasing the node cache manager and block cache
	// manager.

	if( gv_XFlmSysData.pGlobalCacheMgr)
	{
		gv_XFlmSysData.pGlobalCacheMgr->Release();
		gv_XFlmSysData.pGlobalCacheMgr = NULL;
	}

	// Free the Unicode<->WP tables

	freeCharMappingTables();

#ifdef FLM_USE_NICI
	CCS_Shutdown();
#endif

	// Free the mutexes last of all.

	if (gv_XFlmSysData.hQueryMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hQueryMutex);
	}

	if (gv_XFlmSysData.hNodeCacheMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hNodeCacheMutex);
	}
	if (gv_XFlmSysData.hBlockCacheMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hBlockCacheMutex);
	}
	if (gv_XFlmSysData.hShareMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hShareMutex);
	}

	if (gv_XFlmSysData.hIniMutex != F_MUTEX_NULL)
	{
		f_mutexDestroy( &gv_XFlmSysData.hIniMutex);
	}

	// Memory cleanup

	f_memoryCleanup();

	// Shutdown the NetWare interfaces

#ifdef FLM_NLM
	if( gv_bNetWareStartupCalled)
	{
		f_netwareShutdown();
		gv_bNetWareStartupCalled = FALSE;
	}
#endif
}

/****************************************************************************
Desc:	Configures how memory will be dynamically regulated.
****************************************************************************/
RCODE F_GlobalCacheMgr::setDynamicMemoryLimit(
	FLMUINT	uiCacheAdjustPercent,
	FLMUINT	uiCacheAdjustMin,
	FLMUINT	uiCacheAdjustMax,
	FLMUINT	uiCacheAdjustMinToLeave)
{
#ifndef FLM_CAN_GET_PHYS_MEM
	F_UNREFERENCED_PARM( uiCacheAdjustPercent);
	F_UNREFERENCED_PARM( uiCacheAdjustMin);
	F_UNREFERENCED_PARM( uiCacheAdjustMax);
	F_UNREFERENCED_PARM( uiCacheAdjustMinToLeave);
	return( RC_SET( NE_XFLM_NOT_IMPLEMENTED));
#else
	RCODE		rc = NE_XFLM_OK;
	FLMUINT	uiCacheBytes;

	lockMutex();
	m_bDynamicCacheAdjust = TRUE;
	flmAssert( uiCacheAdjustPercent > 0 && uiCacheAdjustPercent <= 100);
	m_uiCacheAdjustPercent = uiCacheAdjustPercent;
	m_uiCacheAdjustMin = uiCacheAdjustMin;
	m_uiCacheAdjustMax = uiCacheAdjustMax;
	m_uiCacheAdjustMinToLeave = uiCacheAdjustMinToLeave;
	uiCacheBytes = flmGetCacheBytes( m_uiCacheAdjustPercent,
							m_uiCacheAdjustMin, m_uiCacheAdjustMax,
							m_uiCacheAdjustMinToLeave, TRUE, totalBytes());

	rc = setCacheLimit( uiCacheBytes, FALSE);
	unlockMutex();
	return( rc);
#endif
}

/****************************************************************************
Desc:		Sets a hard memory limit for cache.
****************************************************************************/
RCODE F_GlobalCacheMgr::setHardMemoryLimit(
	FLMUINT	uiPercent,
	FLMBOOL	bPercentOfAvail,
	FLMUINT	uiMin,
	FLMUINT	uiMax,
	FLMUINT	uiMinToLeave,
	FLMBOOL	bPreallocate)
{
	RCODE		rc = NE_XFLM_OK;

	lockMutex();
	m_bDynamicCacheAdjust = FALSE;
	if (uiPercent)
	{
#ifndef FLM_CAN_GET_PHYS_MEM
		F_UNREFERENCED_PARM( bPercentOfAvail);
		F_UNREFERENCED_PARM( uiMin);
		F_UNREFERENCED_PARM( uiMinToLeave);
		rc = RC_SET( NE_XFLM_NOT_IMPLEMENTED);
#else
		FLMUINT	uiCacheBytes;

		uiCacheBytes = flmGetCacheBytes( uiPercent, uiMin, uiMax, uiMinToLeave,
										bPercentOfAvail, totalBytes());
		rc = setCacheLimit( uiCacheBytes, bPreallocate);
#endif
	}
	else
	{
		rc = setCacheLimit( uiMax, bPreallocate);
	}

	unlockMutex();

	return( rc);
}

/****************************************************************************
Desc:		Returns information about memory usage
****************************************************************************/
void F_GlobalCacheMgr::getCacheInfo(
	XFLM_CACHE_INFO *			pCacheInfo)
{
	f_memset( pCacheInfo, 0, sizeof( XFLM_CACHE_INFO));
	
	lockMutex();
	pCacheInfo->uiMaxBytes = m_uiMaxBytes;
	pCacheInfo->uiTotalBytesAllocated = totalBytes();
	pCacheInfo->bDynamicCacheAdjust = m_bDynamicCacheAdjust;
	pCacheInfo->uiCacheAdjustPercent = m_uiCacheAdjustPercent;
	pCacheInfo->uiCacheAdjustMin = m_uiCacheAdjustMin;
	pCacheInfo->uiCacheAdjustMax = m_uiCacheAdjustMax;
	pCacheInfo->uiCacheAdjustMinToLeave = m_uiCacheAdjustMinToLeave;
	pCacheInfo->bPreallocatedCache = m_bCachePreallocated;
	unlockMutex();

	// Return block cache information.

	f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
	f_memcpy( &pCacheInfo->BlockCache, &gv_XFlmSysData.pBlockCacheMgr->m_Usage,
					sizeof( XFLM_CACHE_USAGE));

	pCacheInfo->uiFreeBytes = gv_XFlmSysData.pBlockCacheMgr->m_uiFreeBytes;
	pCacheInfo->uiFreeCount = gv_XFlmSysData.pBlockCacheMgr->m_uiFreeCount;
	pCacheInfo->uiReplaceableCount = gv_XFlmSysData.pBlockCacheMgr->m_uiReplaceableCount;
	pCacheInfo->uiReplaceableBytes = gv_XFlmSysData.pBlockCacheMgr->m_uiReplaceableBytes;

	f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);
		
	// Return node cache information.

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	f_memcpy( &pCacheInfo->NodeCache, &gv_XFlmSysData.pNodeCacheMgr->m_Usage,
					sizeof( XFLM_CACHE_USAGE));
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);
	
	// Gather per-database statistics

	f_mutexLock( gv_XFlmSysData.hShareMutex);
	if( gv_XFlmSysData.pDatabaseHashTbl)
	{
		FLMUINT			uiLoop;
		F_Database *	pDatabase;

		for( uiLoop = 0; uiLoop < FILE_HASH_ENTRIES; uiLoop++)
		{
			if( (pDatabase = (F_Database *)gv_XFlmSysData.pDatabaseHashTbl[
				uiLoop].pFirstInBucket) != NULL)
			{
				while( pDatabase)
				{
					if( pDatabase->m_uiDirtyCacheCount)
					{
						pCacheInfo->uiDirtyBytes +=
							pDatabase->m_uiDirtyCacheCount * pDatabase->m_uiBlockSize;
						pCacheInfo->uiDirtyCount += pDatabase->m_uiDirtyCacheCount;
					}

					if( pDatabase->m_uiNewCount)
					{
						pCacheInfo->uiNewBytes +=
							pDatabase->m_uiNewCount * pDatabase->m_uiBlockSize;
						pCacheInfo->uiNewCount += pDatabase->m_uiNewCount;
					}

					if( pDatabase->m_uiLogCacheCount)
					{
						pCacheInfo->uiLogBytes +=
							pDatabase->m_uiLogCacheCount * pDatabase->m_uiBlockSize;
						pCacheInfo->uiLogCount += pDatabase->m_uiLogCacheCount;
					}

					pDatabase = pDatabase->m_pNext;
				}
			}
		}
	}

	f_mutexUnlock( gv_XFlmSysData.hShareMutex);
}

/****************************************************************************
Desc:		Returns the number of open databases
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getOpenFileCount( void)
{
	return( gv_XFlmSysData.pFileHdlMgr->getOpenedFiles());
}

/****************************************************************************
Desc:		Close all files in the file handle manager that have not been
			used for the specified number of seconds.
****************************************************************************/
RCODE XFLMAPI F_DbSystem::closeUnusedFiles(
	FLMUINT		uiSeconds)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiValue;
	FLMUINT		uiCurrTime;
	FLMUINT		uiSave;

	// Convert seconds to timer units

	FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiValue);
	gv_XFlmSysData.pFileHdlMgr->checkAgedFileHdls( uiValue);

	// Free any other unused structures that have not been used for the
	// specified amount of time.

	uiCurrTime = (FLMUINT)FLM_GET_TIMER();
	f_mutexLock( gv_XFlmSysData.hShareMutex);

	// Temporarily set the maximum unused seconds in the FLMSYSDATA structure
	// to the value that was passed in to Value1.  Restore it after
	// calling checkNotUsedObject.

	uiSave = gv_XFlmSysData.uiMaxUnusedTime;
	gv_XFlmSysData.uiMaxUnusedTime = uiValue;

	// May unlock and re-lock the global mutex.

	checkNotUsedObjects();
	gv_XFlmSysData.uiMaxUnusedTime = uiSave;
	f_mutexUnlock( gv_XFlmSysData.hShareMutex);

	return( rc);
}

/****************************************************************************
Desc:		Set threshold for number of file handles that are opened by
			the file handle manager
****************************************************************************/
void XFLMAPI F_DbSystem::setOpenThreshold(
	FLMUINT		uiThreshold)
{
	gv_XFlmSysData.pFileHdlMgr->setOpenThreshold( uiThreshold);
}

/****************************************************************************
Desc:		Get the maximum number of file handles available.
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getOpenThreshold( void)
{
	return( gv_XFlmSysData.pFileHdlMgr->getOpenThreshold());
}

/****************************************************************************
Desc:		Enable/disable cache debugging mode
****************************************************************************/
void XFLMAPI F_DbSystem::enableCacheDebug(
	FLMBOOL		bDebug)
{
#ifdef FLM_DEBUG
	gv_XFlmSysData.pBlockCacheMgr->m_bDebug = bDebug;
	gv_XFlmSysData.pNodeCacheMgr->m_bDebug = bDebug;
#else
	F_UNREFERENCED_PARM( bDebug);
#endif
}

/****************************************************************************
Desc:		Returns cache debugging mode
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::cacheDebugEnabled( void)
{
#ifdef FLM_DEBUG
		return( gv_XFlmSysData.pBlockCacheMgr->m_bDebug ||
				  gv_XFlmSysData.pNodeCacheMgr->m_bDebug);
#else
		return( FALSE);
#endif
}

/****************************************************************************
Desc:		Start gathering statistics.
****************************************************************************/
void XFLMAPI F_DbSystem::startStats( void)
{
	f_mutexLock( gv_XFlmSysData.hStatsMutex);
	flmStatStart( &gv_XFlmSysData.Stats);
	f_mutexUnlock( gv_XFlmSysData.hStatsMutex);

	// Start query statistics, if they have not
	// already been started.

	f_mutexLock( gv_XFlmSysData.hQueryMutex);
	if (!gv_XFlmSysData.uiMaxQueries)
	{
		gv_XFlmSysData.uiMaxQueries = 20;
		gv_XFlmSysData.bNeedToUnsetMaxQueries = TRUE;
	}
	f_mutexUnlock( gv_XFlmSysData.hQueryMutex);
}

/****************************************************************************
Desc:		Stop gathering statistics.
****************************************************************************/
void XFLMAPI F_DbSystem::stopStats( void)
{
	f_mutexLock( gv_XFlmSysData.hStatsMutex);
	flmStatStop( &gv_XFlmSysData.Stats);
	f_mutexUnlock( gv_XFlmSysData.hStatsMutex);

	// Stop query statistics, if they were
	// started by a call to FLM_START_STATS.

	f_mutexLock( gv_XFlmSysData.hQueryMutex);
	if (gv_XFlmSysData.bNeedToUnsetMaxQueries)
	{
		gv_XFlmSysData.uiMaxQueries = 0;
		flmFreeSavedQueries( TRUE);
		// NOTE: flmFreeSavedQueries unlocks the mutex.
	}
	else
	{
		f_mutexUnlock( gv_XFlmSysData.hQueryMutex);
	}
}

/****************************************************************************
Desc:		Reset statistics.
****************************************************************************/
void XFLMAPI F_DbSystem::resetStats( void)
{
	FLMUINT		uiSaveMax;

	// Reset the block cache statistics.

	f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
	gv_XFlmSysData.pBlockCacheMgr->m_uiIoWaits = 0;
	gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiCacheHits = 0;
	gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiCacheHitLooks = 0;
	gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiCacheFaults = 0;
	gv_XFlmSysData.pBlockCacheMgr->m_Usage.uiCacheFaultLooks = 0;
	f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);

	// Reset the node cache statistics.

	f_mutexLock( gv_XFlmSysData.hNodeCacheMutex);
	gv_XFlmSysData.pNodeCacheMgr->m_uiIoWaits = 0;
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCacheHits = 0;
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCacheHitLooks = 0;
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCacheFaults = 0;
	gv_XFlmSysData.pNodeCacheMgr->m_Usage.uiCacheFaultLooks = 0;
	f_mutexUnlock( gv_XFlmSysData.hNodeCacheMutex);

	f_mutexLock( gv_XFlmSysData.hStatsMutex);
	flmStatReset( &gv_XFlmSysData.Stats, TRUE);
	f_mutexUnlock( gv_XFlmSysData.hStatsMutex);

	f_mutexLock( gv_XFlmSysData.hQueryMutex);
	uiSaveMax = gv_XFlmSysData.uiMaxQueries;
	gv_XFlmSysData.uiMaxQueries = 0;
	flmFreeSavedQueries( TRUE);
	// NOTE: flmFreeSavedQueries unlocks the mutex.

	// Restore the old maximum

	if(  uiSaveMax)
	{
		// flmFreeSavedQueries unlocks the mutex, so we
		// must relock it to restore the old maximum.

		f_mutexLock( gv_XFlmSysData.hQueryMutex);
		gv_XFlmSysData.uiMaxQueries = uiSaveMax;
		f_mutexUnlock( gv_XFlmSysData.hQueryMutex);
	}
}

/****************************************************************************
Desc:		Returns statistics that have been collected for a share.
Notes:	The statistics returned will be the statistics for ALL databases
****************************************************************************/
RCODE XFLMAPI F_DbSystem::getStats(
	XFLM_STATS *				pFlmStats)
{
	RCODE			rc = NE_XFLM_OK;

	// Get the statistics

	f_mutexLock( gv_XFlmSysData.hStatsMutex);
	rc = flmStatCopy( pFlmStats, &gv_XFlmSysData.Stats);
	f_mutexUnlock( gv_XFlmSysData.hStatsMutex);
	if (RC_BAD( rc))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Frees memory allocated to a FLM_STATS structure
****************************************************************************/
void XFLMAPI F_DbSystem::freeStats(
	XFLM_STATS *			pFlmStats)
{
	flmStatFree( pFlmStats);
}

/****************************************************************************
Desc:		Sets the path for all temporary files that come into use within a
			FLAIM share structure.  The share mutex should be locked when
			settting when called from FlmConfig().
****************************************************************************/
RCODE XFLMAPI F_DbSystem::setTempDir(
	const char *		pszPath)
{
	RCODE		rc = NE_XFLM_OK;

	f_mutexLock( gv_XFlmSysData.hShareMutex);

	// First, test the path

	if( RC_BAD( rc = gv_pFileSystem->Exists( pszPath)))
	{
		goto Exit;
	}

	f_strcpy( gv_XFlmSysData.szTempDir, pszPath);
	gv_XFlmSysData.bTempDirSet = TRUE;

Exit:

	f_mutexUnlock( gv_XFlmSysData.hShareMutex);
	return( rc);
}

/****************************************************************************
Desc:		Get the temporary directory.
****************************************************************************/
RCODE XFLMAPI F_DbSystem::getTempDir(
	char *		pszPath)
{
	RCODE		rc = NE_XFLM_OK;

	f_mutexLock( gv_XFlmSysData.hShareMutex);

	if( !gv_XFlmSysData.bTempDirSet )
	{
		*pszPath = 0;
		rc = RC_SET( NE_XFLM_IO_PATH_NOT_FOUND);
		goto Exit;
	}
	else
	{
		f_strcpy( pszPath, gv_XFlmSysData.szTempDir);
	}

Exit:

	f_mutexUnlock( gv_XFlmSysData.hShareMutex);
	return( rc);
}

/****************************************************************************
Desc:		Sets the maximum seconds between checkpoints
****************************************************************************/
void XFLMAPI F_DbSystem::setCheckpointInterval(
	FLMUINT						uiSeconds)
{
	FLM_SECS_TO_TIMER_UNITS( uiSeconds, gv_XFlmSysData.uiMaxCPInterval);
}

/****************************************************************************
Desc:		Gets the maximum seconds between checkpoints
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getCheckpointInterval( void)
{
	FLMUINT		uiTmp;

	FLM_TIMER_UNITS_TO_SECS( gv_XFlmSysData.uiMaxCPInterval, uiTmp);
	return( uiTmp);
}

/****************************************************************************
Desc:		Sets the interval for dynamically adjusting the cache limit.
****************************************************************************/
void XFLMAPI F_DbSystem::setCacheAdjustInterval(
	FLMUINT						uiSeconds)
{
	FLM_SECS_TO_TIMER_UNITS( uiSeconds,
			gv_XFlmSysData.pGlobalCacheMgr->m_uiCacheAdjustInterval);
}

/****************************************************************************
Desc:		Sets the interval for dynamically adjusting the cache limit.
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getCacheAdjustInterval( void)
{
	FLMUINT		uiTmp;

	FLM_TIMER_UNITS_TO_SECS(
		gv_XFlmSysData.pGlobalCacheMgr->m_uiCacheAdjustInterval, uiTmp);
	return( uiTmp);
}

/****************************************************************************
Desc:		Sets the interval for dynamically cleaning out old
			cache blocks and records
****************************************************************************/
void XFLMAPI F_DbSystem::setCacheCleanupInterval(
	FLMUINT						uiSeconds)
{
	FLM_SECS_TO_TIMER_UNITS( uiSeconds,
		gv_XFlmSysData.pGlobalCacheMgr->m_uiCacheCleanupInterval);
}

/****************************************************************************
Desc:		Gets the interval for dynamically cleaning out old
			cache blocks and records
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getCacheCleanupInterval( void)
{
	FLMUINT		uiTmp;

	FLM_TIMER_UNITS_TO_SECS(
		gv_XFlmSysData.pGlobalCacheMgr->m_uiCacheCleanupInterval, uiTmp);
	return( uiTmp);
}

/****************************************************************************
Desc:		Set interval for cleaning up unused structures
****************************************************************************/
void XFLMAPI F_DbSystem::setUnusedCleanupInterval(
	FLMUINT						uiSeconds)
{
	FLM_SECS_TO_TIMER_UNITS( uiSeconds,
		gv_XFlmSysData.pGlobalCacheMgr->m_uiUnusedCleanupInterval);
}

/****************************************************************************
Desc:		Gets the interval for cleaning up unused structures
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getUnusedCleanupInterval( void)
{
	FLMUINT		uiTmp;

	FLM_TIMER_UNITS_TO_SECS(
		gv_XFlmSysData.pGlobalCacheMgr->m_uiUnusedCleanupInterval, uiTmp);
	return( uiTmp);
}

/****************************************************************************
Desc:		Set the maximum time for an item to be unused before it is
			cleaned up
****************************************************************************/
void XFLMAPI F_DbSystem::setMaxUnusedTime(
	FLMUINT						uiSeconds)
{
	FLM_SECS_TO_TIMER_UNITS( uiSeconds, gv_XFlmSysData.uiMaxUnusedTime);
}

/****************************************************************************
Desc:		Gets the maximum time for an item to be unused
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getMaxUnusedTime( void)
{
	FLMUINT		uiTmp;

	FLM_TIMER_UNITS_TO_SECS( gv_XFlmSysData.uiMaxUnusedTime, uiTmp);
	return( uiTmp);
}

/****************************************************************************
Desc:		Enables or disables out-of-memory simulation
****************************************************************************/
void F_DbSystem::enableOutOfMemorySimulation(
	FLMBOOL						bEnable)
{
#ifdef DEBUG_SIM_OUT_OF_MEM
	gv_XFlmSysData.uiOutOfMemSimEnabledFlag = bEnable
												? OUT_OF_MEM_SIM_ENABLED_FLAG
												: 0;
#else
	F_UNREFERENCED_PARM( bEnable);
#endif
}

/****************************************************************************
Desc:		Returns the state of out-of-memory simulation
****************************************************************************/
FLMBOOL F_DbSystem::outOfMemorySimulationEnabled( void)
{
#ifdef DEBUG_SIM_OUT_OF_MEM
	if( gv_XFlmSysData.uiOutOfMemSimEnabledFlag == OUT_OF_MEM_SIM_ENABLED_FLAG)
	{
		return( TRUE);
	}
#endif

	return( FALSE);
}

/****************************************************************************
Desc:		Sets the logging object to be used for internal status messages
****************************************************************************/
void XFLMAPI F_DbSystem::setLogger(
	IF_LoggerClient *		pLogger)
{
	IF_LoggerClient *		pOldLogger = NULL;
	
	f_mutexLock( gv_XFlmSysData.hLoggerMutex);
	
	for( ;;)	
	{
		// While waiting for the pending message count to go to zero, other
		// threads may have come in to set different logger objects.  To handle
		// this case, we need to check gv_XFlmSysData.pLogger and release it
		// if non-NULL.
	
		if( gv_XFlmSysData.pLogger)
		{
			if( pOldLogger)
			{
				pOldLogger->Release();
			}
			
			pOldLogger = gv_XFlmSysData.pLogger;
			gv_XFlmSysData.pLogger = NULL;
		}
		
		if( !gv_XFlmSysData.uiPendingLogMessages)
		{
			break;
		}
		
		f_mutexUnlock( gv_XFlmSysData.hLoggerMutex);
		f_sleep( 100);
		f_mutexLock( gv_XFlmSysData.hLoggerMutex);
	}
	
	if( pOldLogger)
	{
		pOldLogger->Release();
	}
	
	if( (gv_XFlmSysData.pLogger = pLogger) != NULL)
	{
		gv_XFlmSysData.pLogger->AddRef();
	}
	
	f_mutexUnlock( gv_XFlmSysData.hLoggerMutex);
}

/****************************************************************************
Desc:		Enables or disables the use of ESM (if available)
****************************************************************************/
void XFLMAPI F_DbSystem::enableExtendedServerMemory(
	FLMBOOL						bEnable)
{
	gv_XFlmSysData.bOkToUseESM = bEnable;
}

/****************************************************************************
Desc:		Returns the state of ESM
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::extendedServerMemoryEnabled( void)
{
	return( gv_XFlmSysData.bOkToUseESM);
}

/****************************************************************************
Desc:		Deactivates open database handles, forcing the database to be
			(eventually) closed
Notes:	Passing NULL for the path values will cause all active database
			handles to be deactivated
****************************************************************************/
void XFLMAPI F_DbSystem::deactivateOpenDb(
	const char *	pszDbFileName,
	const char *	pszDataDir)
{
	F_Database *	pTmpDatabase;

	f_mutexLock( gv_XFlmSysData.hShareMutex);
	if( pszDbFileName)
	{
		// Look up the file using findDatabase to see if we have the
		// file open.  May unlock and re-lock the global mutex.

		if( RC_OK( findDatabase( pszDbFileName,
			pszDataDir, &pTmpDatabase)) && pTmpDatabase)
		{
			pTmpDatabase->setMustCloseFlags( NE_XFLM_OK, TRUE);
		}
	}
	else
	{
		if( gv_XFlmSysData.pDatabaseHashTbl)
		{
			FLMUINT		uiLoop;

			for( uiLoop = 0; uiLoop < FILE_HASH_ENTRIES; uiLoop++)
			{
				pTmpDatabase =
					(F_Database *)gv_XFlmSysData.pDatabaseHashTbl[ uiLoop].pFirstInBucket;

				while( pTmpDatabase)
				{
					pTmpDatabase->setMustCloseFlags( NE_XFLM_OK, TRUE);
					pTmpDatabase = pTmpDatabase->m_pNext;
				}
			}
		}
	}

	f_mutexUnlock( gv_XFlmSysData.hShareMutex);
}

/****************************************************************************
Desc:		Sets the maximum number of queries to save
****************************************************************************/
void XFLMAPI F_DbSystem::setQuerySaveMax(
	FLMUINT						uiMaxToSave)
{
	f_mutexLock( gv_XFlmSysData.hQueryMutex);
	gv_XFlmSysData.uiMaxQueries = uiMaxToSave;
	gv_XFlmSysData.bNeedToUnsetMaxQueries = FALSE;
	flmFreeSavedQueries( TRUE);
}

/****************************************************************************
Desc:		Gets the maximum number of queries to save
****************************************************************************/
FLMUINT XFLMAPI F_DbSystem::getQuerySaveMax( void)
{
	return( gv_XFlmSysData.uiMaxQueries);
}

/****************************************************************************
Desc:		Sets the maximum amount of dirty cache allowed
****************************************************************************/
void XFLMAPI F_DbSystem::setDirtyCacheLimits(
	FLMUINT	uiMaxDirty,
	FLMUINT	uiLowDirty)
{
	f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
	if( !uiMaxDirty)
	{
		gv_XFlmSysData.pBlockCacheMgr->m_bAutoCalcMaxDirty = TRUE;
		gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache = 0;
		gv_XFlmSysData.pBlockCacheMgr->m_uiLowDirtyCache = 0;
	}
	else
	{
		gv_XFlmSysData.pBlockCacheMgr->m_bAutoCalcMaxDirty = FALSE;
		gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache = uiMaxDirty;

		// Low threshhold must be no higher than maximum!

		if ((gv_XFlmSysData.pBlockCacheMgr->m_uiLowDirtyCache = uiLowDirty) >
			gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache)
		{
			gv_XFlmSysData.pBlockCacheMgr->m_uiLowDirtyCache =
				gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache;
		}
	}
	f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);
}

/****************************************************************************
Desc:		Gets the maximum amount of dirty cache allowed
****************************************************************************/
void XFLMAPI F_DbSystem::getDirtyCacheLimits(
	FLMUINT *	puiMaxDirty,
	FLMUINT *	puiLowDirty)
{
	f_mutexLock( gv_XFlmSysData.hBlockCacheMutex);
	if( puiMaxDirty)
	{
		*puiMaxDirty = gv_XFlmSysData.pBlockCacheMgr->m_uiMaxDirtyCache;
	}

	if( puiLowDirty)
	{
		*puiLowDirty = gv_XFlmSysData.pBlockCacheMgr->m_uiLowDirtyCache;
	}

	f_mutexUnlock( gv_XFlmSysData.hBlockCacheMutex);
}

/****************************************************************************
Desc:		Returns information about threads owned by the database engine
****************************************************************************/
RCODE XFLMAPI F_DbSystem::getThreadInfo(
	IF_ThreadInfo **	ppThreadInfo
	)
{
	RCODE				rc = NE_XFLM_OK;
	F_ThreadInfo *	pThreadInfo = NULL;

	if ((pThreadInfo = f_new F_ThreadInfo) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = gv_XFlmSysData.pThreadMgr->getThreadInfo(
								&pThreadInfo->m_Pool,
								&pThreadInfo->m_pThreadInfoArray,
								&pThreadInfo->m_uiNumThreads)))
	{
		goto Exit;
	}

Exit:

	if (RC_BAD( rc) && pThreadInfo)
	{
		pThreadInfo->Release();
		pThreadInfo = NULL;
	}

	*ppThreadInfo = (IF_ThreadInfo *)pThreadInfo;
	return( rc);
}

/****************************************************************************
Desc:		Registers for an event
****************************************************************************/
RCODE XFLMAPI F_DbSystem::registerForEvent(
	eEventCategory		eCategory,
	IF_EventClient *		pEventClient)
{
	RCODE		rc = NE_XFLM_OK;
	FEVENT *	pEvent;

	// Make sure it is a legal event category to register for.

	if (eCategory >= XFLM_MAX_EVENT_CATEGORIES)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_NOT_IMPLEMENTED);
		goto Exit;
	}

	// Allocate an event structure

	if (RC_BAD( rc = f_calloc(
		(FLMUINT)(sizeof( FEVENT)), &pEvent)))
	{
		goto Exit;
	}

	// Initialize the structure members and linkt to the
	// list of events off of the event category.

	pEvent->pEventClient = pEventClient;
	pEvent->pEventClient->AddRef();
	// pEvent->pPrev = NULL;		// done by f_calloc above.

	// Mutex should be locked to link into list.

	f_mutexLock( gv_XFlmSysData.EventHdrs [eCategory].hMutex);
	if ((pEvent->pNext =
			gv_XFlmSysData.EventHdrs [eCategory].pEventCBList) != NULL)
	{
		pEvent->pNext->pPrev = pEvent;
	}
	gv_XFlmSysData.EventHdrs [eCategory].pEventCBList = pEvent;
	f_mutexUnlock( gv_XFlmSysData.EventHdrs [eCategory].hMutex);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		De-registers for an event
****************************************************************************/
void XFLMAPI F_DbSystem::deregisterForEvent(
	eEventCategory		eCategory,
	IF_EventClient *		pEventClient)
{
	FEVENT *	pEvent = gv_XFlmSysData.EventHdrs [eCategory].pEventCBList;

	// Find the event and remove from the list.  If it is not
	// there, don't worry about it.

	while (pEvent)
	{
		if (pEventClient == pEvent->pEventClient)
		{
			flmFreeEvent( pEvent,
				gv_XFlmSysData.EventHdrs[ eCategory].hMutex,
				&gv_XFlmSysData.EventHdrs[ eCategory].pEventCBList);
			break;
		}
		pEvent = pEvent->pNext;
	}
}

/****************************************************************************
Desc:		Returns TRUE if the specified error indicates that the database
			is corrupt
****************************************************************************/
RCODE XFLMAPI F_DbSystem::getNextMetaphone(
	IF_IStream *	pIStream,
	FLMUINT *		puiMetaphone,
	FLMUINT *		puiAltMetaphone)
{
	return( flmGetNextMetaphone( 
		pIStream, puiMetaphone, puiAltMetaphone));
}

/****************************************************************************
Desc:		Returns TRUE if the specified error indicates that the database
			is corrupt
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::errorIsFileCorrupt(
	RCODE						rc)
{
	FLMBOOL		bIsCorrupt = FALSE;

	switch( rc)
	{
		case NE_XFLM_BTREE_ERROR :
		case NE_XFLM_DATA_ERROR :
		case NE_XFLM_NOT_FLAIM :
		case NE_XFLM_BLOCK_CRC :
		case NE_XFLM_HDR_CRC :
		case NE_XFLM_INCOMPLETE_LOG :
			bIsCorrupt = TRUE;
			break;
		default :
			break;
	}

	return( bIsCorrupt);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBYTE flmShiftRightRetByte(
	FLMUINT64	ui64Num,
	FLMBYTE		ucBits)
{
	return( ucBits < 64 ? (FLMBYTE)(ui64Num >> ucBits) : 0);
}

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT flmGetSENByteCount(
	FLMUINT64	ui64Num)
{
	FLMUINT		uiCount = 0;

	if( ui64Num < 0x80)
	{
		return( 1);
	}

	while( ui64Num)
	{
		uiCount++;
		ui64Num >>= 7;
	}

	// If the high bit is set, the counter will be incremented 1 beyond
	// the actual number of bytes need to represent the SEN.  We will need
	// to re-visit this if we ever go beyond 64-bits.

	return( uiCount < FLM_MAX_SEN_LEN ? uiCount : FLM_MAX_SEN_LEN);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
FLMUINT flmEncodeSEN(
	FLMUINT64		ui64Value,
	FLMBYTE **		ppucBuffer,
	FLMUINT			uiSizeWanted)
{
	FLMBYTE *			pucBuffer = *ppucBuffer;
	FLMUINT				uiSenLen = flmGetSENByteCount( ui64Value);

	flmAssert( uiSizeWanted <= FLM_MAX_SEN_LEN && 
				  (!uiSizeWanted || uiSizeWanted >= uiSenLen));

	uiSenLen = uiSizeWanted > uiSenLen ? uiSizeWanted : uiSenLen;

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							flmShiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = flmShiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	return( uiSenLen);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
RCODE flmEncodeSEN(
	FLMUINT64		ui64Value,
	FLMBYTE **		ppucBuffer,
	FLMBYTE *		pucEnd)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBYTE *		pucBuffer = *ppucBuffer;
	FLMUINT			uiSenLen = flmGetSENByteCount( ui64Value);
	
	if( *ppucBuffer + uiSenLen > pucEnd)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							flmShiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = flmShiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Encodes a number as a SEN
****************************************************************************/
FLMUINT flmEncodeSENKnownLength(
	FLMUINT64		ui64Value,
	FLMUINT			uiSenLen,
	FLMBYTE **		ppucBuffer)
{
	FLMBYTE *			pucBuffer = *ppucBuffer;

	if( uiSenLen == 1)
	{
		*pucBuffer++ = (FLMBYTE)ui64Value;
	}
	else
	{
		FLMUINT			uiTmp = (uiSenLen - 1) << 3;

		*pucBuffer++ = ucSENPrefixArray[ uiSenLen] + 
							flmShiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		while( uiTmp)
		{
			uiTmp -= 8;
			*pucBuffer++ = flmShiftRightRetByte( ui64Value, (FLMBYTE)uiTmp);
		}
	}

	*ppucBuffer = pucBuffer;
	return( uiSenLen);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE flmDecodeSEN64(
	const FLMBYTE **		ppucBuffer,
	const FLMBYTE *		pucEnd,
	FLMUINT64 *				pui64Value)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiSENLength;
	const FLMBYTE *	pucBuffer = *ppucBuffer;

	uiSENLength = gv_ucSENLengthArray[ *pucBuffer];
	if( pucBuffer + uiSENLength > pucEnd)
	{
		if (pui64Value)
		{
			*pui64Value = 0;
		}
		rc = RC_SET( NE_XFLM_BAD_SEN);
		goto Exit;
	}

	if (pui64Value)
	{
		switch( uiSENLength)
		{
			case 1:
				*pui64Value = *pucBuffer;
				break;
	
			case 2:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x3F)) << 8) + pucBuffer[ 1];
				break;
	
			case 3:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x1F)) << 16) +
					(((FLMUINT64)pucBuffer[ 1]) << 8) + pucBuffer[ 2];
				break;
	
			case 4:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x0F)) << 24) +
					(((FLMUINT64)pucBuffer[ 1]) << 16) +
					(((FLMUINT64)pucBuffer[ 2]) << 8) + pucBuffer[ 3];
				break;
	
			case 5:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x07)) << 32) +
					(((FLMUINT64)pucBuffer[ 1]) << 24) +
					(((FLMUINT64)pucBuffer[ 2]) << 16) +
					(((FLMUINT64)pucBuffer[ 3]) << 8) + pucBuffer[ 4];
				break;
	
			case 6:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x03)) << 40) +
					(((FLMUINT64)pucBuffer[ 1]) << 32) +
					(((FLMUINT64)pucBuffer[ 2]) << 24) +
					(((FLMUINT64)pucBuffer[ 3]) << 16) +
					(((FLMUINT64)pucBuffer[ 4]) << 8) + pucBuffer[ 5];
				break;
	
			case 7:
				*pui64Value = (((FLMUINT64)(*pucBuffer & 0x01)) << 48) +
					(((FLMUINT64)pucBuffer[ 1]) << 40) +
					(((FLMUINT64)pucBuffer[ 2]) << 32) +
					(((FLMUINT64)pucBuffer[ 3]) << 24) +
					(((FLMUINT64)pucBuffer[ 4]) << 16) +
					(((FLMUINT64)pucBuffer[ 5]) << 8) + pucBuffer[ 6];
				break;
	
			case 8:
				*pui64Value = (((FLMUINT64)pucBuffer[ 1]) << 48) +
					(((FLMUINT64)pucBuffer[ 2]) << 40) +
					(((FLMUINT64)pucBuffer[ 3]) << 32) +
					(((FLMUINT64)pucBuffer[ 4]) << 24) +
					(((FLMUINT64)pucBuffer[ 5]) << 16) +
					(((FLMUINT64)pucBuffer[ 6]) << 8) + pucBuffer[ 7];
				break;
	
			case 9:
				*pui64Value = (((FLMUINT64)pucBuffer[ 1]) << 56) +
					(((FLMUINT64)pucBuffer[ 2]) << 48) +
					(((FLMUINT64)pucBuffer[ 3]) << 40) +
					(((FLMUINT64)pucBuffer[ 4]) << 32) +
					(((FLMUINT64)pucBuffer[ 5]) << 24) +
					(((FLMUINT64)pucBuffer[ 6]) << 16) +
					(((FLMUINT64)pucBuffer[ 7]) << 8) + pucBuffer[ 8];
				break;
	
			default:
				*pui64Value = 0;
				flmAssert( 0);
				break;
		}
	}

Exit:

	*ppucBuffer = pucBuffer + uiSENLength;

	return( rc);
}

/****************************************************************************
Desc:		COM query interface
****************************************************************************/
RCODE XFLMAPI F_DbSystem::QueryInterface(
	RXFLMIID	riid,
	void **		ppvInt)
{
	RCODE		rc = NE_XFLM_OK;
	
	if( (f_memcmp(&riid, &Internal_IID_IF_DbSystem,
				   sizeof( Internal_IID_IF_DbSystem)) == 0)  ||
		 (f_memcmp(&riid, &Internal_IID_XFLMIUnknown,
		 		   sizeof( Internal_IID_XFLMIUnknown)) == 0) )
	{
		*ppvInt = this;
		AddRef();
	}
	else
	{
		rc = RC_SET( NE_XFLM_UNSUPPORTED_INTERFACE);
	}

	return( rc);
}

/****************************************************************************
Desc:		Increment the database system use count
****************************************************************************/
FLMINT XFLMAPI F_DbSystem::AddRef(void)
{
	FLMINT		iRefCnt = flmAtomicInc( &m_refCnt);

	// Note: We don't bother with a call to LockModule() here because
	// it's done in the constructor.  In fact, it's unlikely that
	// this AddRef() will ever be called, because the constructor
	// already sets the ref count to 1 and it's unlikely that there
	// will be to references to the same DbSystem object...

	return( iRefCnt);
}

/****************************************************************************
Desc:		Decrement the database system use count
****************************************************************************/
FLMINT XFLMAPI F_DbSystem::Release(void)
{
	FLMINT	iRefCnt = flmAtomicDec( &m_refCnt);

	if (iRefCnt == 0)
	{
		delete this;	
	}

	return( iRefCnt);
}

/****************************************************************************
Desc:		Returns a pointer to the ASCII string representation
			of a return code.
****************************************************************************/
const char * F_DbSystem::errorString(
	RCODE			rc)
{
	const char *		pszErrorStr;

	if( rc == NE_XFLM_OK)
	{
		pszErrorStr = "NE_XFLM_OK";
	}
	else if( rc > NE_XFLM_FIRST_COMMON_ERROR &&
		rc < NE_XFLM_LAST_COMMON_ERROR)
	{
		pszErrorStr = gv_FlmCommonErrors[
			rc - NE_XFLM_FIRST_COMMON_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_GENERAL_ERROR &&
		rc < NE_XFLM_LAST_GENERAL_ERROR)
	{
		pszErrorStr = gv_FlmGeneralErrors[
			rc - NE_XFLM_FIRST_GENERAL_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_DOM_ERROR &&
		rc < NE_XFLM_LAST_DOM_ERROR)
	{
		pszErrorStr = gv_FlmDomErrors[
			rc - NE_XFLM_FIRST_DOM_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_IO_ERROR &&
		rc < NE_XFLM_LAST_IO_ERROR)
	{
		pszErrorStr = gv_FlmIoErrors[
			rc - NE_XFLM_FIRST_IO_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_NET_ERROR &&
		rc < NE_XFLM_LAST_NET_ERROR)
	{
		pszErrorStr = gv_FlmNetErrors[
			rc - NE_XFLM_FIRST_NET_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_QUERY_ERROR &&
		rc < NE_XFLM_LAST_QUERY_ERROR)
	{
		pszErrorStr = gv_FlmQueryErrors[
			rc - NE_XFLM_FIRST_QUERY_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_STREAM_ERROR &&
		rc < NE_XFLM_LAST_STREAM_ERROR)
	{
		pszErrorStr = gv_FlmStreamErrors[
			rc - NE_XFLM_FIRST_STREAM_ERROR - 1].pszErrorStr;
	}
	else if( rc > NE_XFLM_FIRST_NICI_ERROR &&
		rc < NE_XFLM_LAST_NICI_ERROR)
	{
		pszErrorStr = gv_FlmNiciErrors[
			rc - NE_XFLM_FIRST_NICI_ERROR - 1].pszErrorStr;
	}
	else
	{
		pszErrorStr = "Unknown error";
	}

	return( pszErrorStr);
}

/****************************************************************************
Desc:		Checks the error code mapping tables on startup
****************************************************************************/
RCODE F_DbSystem::checkErrorCodeTables( void)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiLoop;

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_GENERAL_ERROR - NE_XFLM_FIRST_GENERAL_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmGeneralErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_GENERAL_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_DOM_ERROR - NE_XFLM_FIRST_DOM_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmDomErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_DOM_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_IO_ERROR - NE_XFLM_FIRST_IO_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmIoErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_IO_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_NET_ERROR - NE_XFLM_FIRST_NET_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmNetErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_NET_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_QUERY_ERROR - NE_XFLM_FIRST_QUERY_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmQueryErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_QUERY_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_STREAM_ERROR - NE_XFLM_FIRST_STREAM_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmStreamErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_STREAM_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

	for( uiLoop = 0;
		uiLoop < (NE_XFLM_LAST_NICI_ERROR - NE_XFLM_FIRST_NICI_ERROR - 1);
		uiLoop++)
	{
		if( gv_FlmNiciErrors[ uiLoop].rc !=
			(RCODE)(uiLoop + NE_XFLM_FIRST_NICI_ERROR + 1))
		{
			rc = RC_SET_AND_ASSERT( NE_XFLM_BAD_RCODE_TABLE);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Allocates an F_DbSystem object for non-COM applications
****************************************************************************/
XFLMEXP RCODE XFLMAPI FlmAllocDbSystem(
	IF_DbSystem **			ppDbSystem)
{
	flmAssert( ppDbSystem && *ppDbSystem == NULL);

	if( (*ppDbSystem = (IF_DbSystem *)f_new F_DbSystem) == NULL)
	{
		return( RC_SET( NE_XFLM_MEM));
	}

	return( NE_XFLM_OK);
}

/****************************************************************************
Desc:	Check for the existence of FLAIM performance tuning file. If
		found this routine will parse the contents and set the global 
		performance tuning variables.

		Looks for the following strings within the .ini file
			cache=<cacheBytes>				# Set a hard memory limit
			cache=<cache options>			# Set a hard memory limit or dynamically
													# adjusting limit.
			<cache options>					# Multiple options may be specified, in
													# any order, separated by commas.  All
													# are optional.
				%:<percentage>					# percentage of available or physical memory.
				AVAIL or TOTAL					# percentage is for available physical
													# memory or total physical memory.  NOTE:
													# these are ignored for a dynamic limit.
				MIN:<bytes>						# minimum number of bytes
				MAX:<bytes>						# maximum number of bytes
				LEAVE:<bytes>					# minimum nmber of bytes to leave
				DYN or HARD						# dynamic or hard limit
				PRE								# preallocate cache - valid only with HARD setting.
				
		Examples:
	
			cache=DYN,%:75,MIN:16000000	# Dynamic limit of 75% available memory,
													# minimum of 16 megabytes.
			cache=HARD,%:75,MIN:16000000	# Hard limit of 75% total physical memory,
													# minimum of 16 megabytes.
			cache=HARD,%:75,MIN:16000000,PRE
													# Hard limit of 75% total physical memory,
													# minimum of 16 megabytes, preallocated.
			cache=8000000						# Old style - hard limit of 8 megabytes.

			cacheadjustinterval=<seconds>
			cachecleanupinterval=<seconds>
****************************************************************************/

#define XFLM_INI_FILE_NAME "_xflm.ini"

RCODE F_DbSystem::readIniFile()
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bMutexLocked = FALSE;
	F_IniFile *		pIniFile = NULL;
	char				szIniFileName [F_PATH_MAX_SIZE];
	FLMUINT			uiParamValue;
	FLMUINT			uiMaxDirtyCache;
	FLMUINT			uiLowDirtyCache;

	// Initialize the ini file object...

	if ((pIniFile = f_new F_IniFile) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pIniFile->Init()))
	{
		goto Exit;
	}

	// Lock the ini file mutex

	f_mutexLock( gv_XFlmSysData.hIniMutex);
	bMutexLocked = TRUE;

	if (RC_BAD( rc = flmGetIniFileName( (FLMBYTE *)szIniFileName,
													F_PATH_MAX_SIZE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pIniFile->Read( szIniFileName)))
	{
		goto Exit;
	}

	// Set FLAIM parameters from the buffer read from the .ini file.

	setCacheParams( pIniFile);
	
	flmGetUintParam( XFLM_INI_CACHE_ADJUST_INTERVAL, 
							XFLM_DEFAULT_CACHE_CLEANUP_INTERVAL,
							&uiParamValue, pIniFile);
	setCacheAdjustInterval( uiParamValue);
	
	flmGetUintParam( XFLM_INI_CACHE_CLEANUP_INTERVAL, 
							XFLM_DEFAULT_CACHE_CLEANUP_INTERVAL,
							&uiParamValue, pIniFile);
	setCacheCleanupInterval( uiParamValue);

	flmGetUintParam( XFLM_INI_MAX_DIRTY_CACHE, 0, &uiMaxDirtyCache, pIniFile);

	flmGetUintParam( XFLM_INI_LOW_DIRTY_CACHE, 0, &uiLowDirtyCache, pIniFile);

	// Use FLAIM's default behavior if uiMaxDirtyCache is zero.  FLAIM's
	// default behavior is to use a maximum value for dirty cache.

	if (uiMaxDirtyCache)
	{
		setDirtyCacheLimits( uiMaxDirtyCache, uiLowDirtyCache);
	}

Exit:

	if (bMutexLocked)
	{
		f_mutexUnlock( gv_XFlmSysData.hIniMutex);
	}

	pIniFile->Release();
	return( rc);
}

/****************************************************************************
Desc: 	Given a tag and a value this function will open/create the 
			ini file and replace or insert the tag with it's value.
			NOTE: This function expects gv_XFlmSysData.hIniMutex to already be
			locked!
****************************************************************************/
RCODE XFLMAPI F_DbSystem::updateIniFile(
	const char *	pszParamName,
	const char *	pszValue)
{
	RCODE				rc = NE_XFLM_OK;
	F_IniFile *		pIniFile = NULL;
	char				szIniFileName [F_PATH_MAX_SIZE];

	if ((pIniFile = f_new F_IniFile) == NULL)
	{
		rc = RC_SET( NE_XFLM_MEM);
		goto Exit;
	}

	if (RC_BAD( rc = pIniFile->Init()))
	{
		goto Exit;
	}

	if (RC_BAD( rc = flmGetIniFileName( (FLMBYTE *)szIniFileName,
													F_PATH_MAX_SIZE)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pIniFile->Read( szIniFileName)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pIniFile->SetParam( pszParamName, pszValue)))
	{
		goto Exit;
	}

	if (RC_BAD( rc = pIniFile->Write()))
	{
		goto Exit;
	}

Exit:

	pIniFile->Release();
	return( rc);
}

/****************************************************************************
Desc:	Sets the parameters for controlling cache.
****************************************************************************/
RCODE F_DbSystem::setCacheParams(
	F_IniFile *			pIniFile)
{
	RCODE				rc = NE_XFLM_OK;
	char *			pszCacheParam = NULL;
	char				cChar;
	char *			pszSave;
	char *			pszValueName;
	FLMBOOL			bDynamicCacheAdjust;
	FLMBOOL			bCalcLimitOnAvail = FALSE;
	FLMUINT			uiCachePercent;
	FLMUINT			uiCacheMin;
	FLMUINT			uiCacheMax;
	FLMUINT			uiCacheMinToLeave;
	FLMBOOL			bPreallocated = FALSE;

	// Set up appropriate defaults.

	bDynamicCacheAdjust = getDynamicCacheSupported();

	if( bDynamicCacheAdjust)
	{
		uiCachePercent = XFLM_DEFAULT_CACHE_ADJUST_PERCENT;
		bCalcLimitOnAvail = TRUE;
		uiCacheMin = XFLM_DEFAULT_CACHE_ADJUST_MIN;
		uiCacheMax = XFLM_DEFAULT_CACHE_ADJUST_MAX;
		uiCacheMinToLeave = 0;
	}
	else
	{
		uiCachePercent = 0;
		uiCacheMin =
		uiCacheMax = XFLM_DEFAULT_CACHE_ADJUST_MIN;
		uiCacheMinToLeave = 0;
	}

	// Extract values from the cache parameter
	flmGetStringParam( XFLM_INI_CACHE, &pszCacheParam, pIniFile);
	
	if ((pszCacheParam != NULL) && (pszCacheParam[0] != '\0'))
	{
		// Parse until we hit white space.

		while (*pszCacheParam &&
				 *pszCacheParam != ' ' &&
				 *pszCacheParam != '\n' &&
				 *pszCacheParam != '\r' &&
				 *pszCacheParam != '\t')
		{

			// Get the value name.

			pszValueName = pszCacheParam;
			while (*pszCacheParam &&
					 *pszCacheParam != ' ' &&
					 *pszCacheParam != '\n' &&
					 *pszCacheParam != '\r' &&
					 *pszCacheParam != '\t' &&
					 *pszCacheParam != ':' &&
					 *pszCacheParam != ',' &&
					 *pszCacheParam != ';')
			{
				pszCacheParam++;
			}
			pszSave = pszCacheParam;
			cChar = *pszSave;
			*pszSave = 0;

			// New options only supported on platforms that have ways
			// of getting physical memory size and available physical
			// memory size.

			if ((f_stricmp( pszValueName, "MAX") == 0) ||
				 (f_stricmp( pszValueName, "MAXIMUM") == 0))
			{
				if (cChar == ':')
				{
					pszCacheParam++;
					flmGetNumParam( &pszCacheParam,
						&uiCacheMax);
				}
			}
			else if ((f_stricmp( pszValueName, "MIN") == 0) ||
						(f_stricmp( pszValueName, "MINIMUM") == 0))
			{
				if (cChar == ':')
				{
					pszCacheParam++;
					flmGetNumParam( &pszCacheParam,
						&uiCacheMin);
				}
			}
			else if ((f_stricmp( pszValueName, "%") == 0) ||
						(f_stricmp( pszValueName, "PERCENT") == 0))
			{
				if (cChar == ':')
				{
					pszCacheParam++;
					flmGetNumParam( &pszCacheParam,
						&uiCachePercent);
				}
			}
			else if ((f_stricmp( pszValueName, "DYNAMIC") == 0) ||
						(f_stricmp( pszValueName, "DYN") == 0))
			{
				if (cChar == ':' || cChar == ',' || cChar == ';')
				{
					pszCacheParam++;
				}
				bDynamicCacheAdjust = TRUE;
			}
			else if (f_stricmp( pszValueName, "AVAIL") == 0)
			{
				if (cChar == ':' || cChar == ',' || cChar == ';')
				{
					pszCacheParam++;
				}
				bCalcLimitOnAvail = TRUE;
			}
			else if (f_stricmp( pszValueName, "TOTAL") == 0)
			{
				if (cChar == ':' || cChar == ',' || cChar == ';')
				{
					pszCacheParam++;
				}
				bCalcLimitOnAvail = FALSE;
			}
			else if (f_stricmp( pszValueName, "LEAVE") == 0)
			{
				if (cChar == ':')
				{
					pszCacheParam++;
					flmGetNumParam( &pszCacheParam,
										 &uiCacheMinToLeave);
				}
			}
			else if (f_stricmp( pszValueName, "HARD") == 0)
			{
				if (cChar == ':' || cChar == ',' || cChar == ';')
				{
					pszCacheParam++;
				}
				bDynamicCacheAdjust = FALSE;
			}
			else if (f_stricmp( pszValueName, "PRE") == 0)
			{
				if (cChar == ':' || cChar == ',' || cChar == ';')
				{
					pszCacheParam++;
				}
				bPreallocated = TRUE;
			}
			else
			{
				*pszSave = cChar;

				// This will handle the old way where they just put in a
				// number for the byte maximum.

				bDynamicCacheAdjust = FALSE;
				uiCachePercent = 0;
				uiCacheMax = f_atol( pszValueName);

				// Don't really have to set these, but do it just to
				// be clean.

				uiCacheMin = 0;
				uiCacheMinToLeave = 0;
				bCalcLimitOnAvail = FALSE;
				break;
			}
			*pszSave = cChar;
		}
	}

	// Set up the stuff for controlling cache - hard limit or dynamically
	// adjusting limit.

	if( bDynamicCacheAdjust)
	{
		if (RC_BAD( rc = setDynamicMemoryLimit(
								uiCachePercent,
								uiCacheMin, uiCacheMax,
								uiCacheMinToLeave)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = setHardMemoryLimit( uiCachePercent,
									bCalcLimitOnAvail, uiCacheMin,
									uiCacheMax, uiCacheMinToLeave,
									bPreallocated)))
		{
			goto Exit;
		}
	}

Exit:

	return rc;
}

/****************************************************************************
Desc:	Gets a string parameter from the .ini file.
****************************************************************************/
FSTATIC void flmGetStringParam(
	const char *	pszParamName,
	char **			ppszValue,
	F_IniFile *		pIniFile)
{
	flmAssert( pIniFile);
	(void)pIniFile->GetParam( pszParamName, ppszValue);
}

/****************************************************************************
Desc: Gets one of the number parameters for setting cache stuff.
****************************************************************************/
FSTATIC void flmGetNumParam(
	char **		ppszParam,
	FLMUINT *	puiNum)
{
	char *		pszTmp = *ppszParam;
	FLMUINT		uiNum = 0;

	while ((*pszTmp >= '0') && (*pszTmp <= '9'))
	{
		uiNum *= 10;
		uiNum += (FLMUINT)(*pszTmp - '0');
		pszTmp++;
	}

	// Run past any other junk up to allowed terminators.

	while (*pszTmp &&
			 *pszTmp != ' ' &&
			 *pszTmp != '\n' &&
			 *pszTmp != '\r' &&
			 *pszTmp != '\t' &&
			 *pszTmp != ':' &&
			 *pszTmp != ',' &&
			 *pszTmp != ';')
	{
		pszTmp++;
	}

	// Skip past trailing colon, comma, or semicolon.

	if (*pszTmp == ':' || *pszTmp == ',' || *pszTmp == ';')
	{
		pszTmp++;
	}

	*puiNum = uiNum;
	*ppszParam = pszTmp;
}

/****************************************************************************
Desc:	Gets a FLMUINT parameter from the .ini file.
****************************************************************************/
FSTATIC void flmGetUintParam(
	const char *	pszParamName,
	FLMUINT			uiDefaultValue,
	FLMUINT *		puiUint,
	F_IniFile *		pIniFile)
{
	flmAssert( pIniFile);
	if (!pIniFile->GetParam( pszParamName, puiUint))
	{
		*puiUint = uiDefaultValue;
	}
}

/****************************************************************************
Desc:	Gets a FLMBOOL parameter from the .ini file.
****************************************************************************/
void flmGetBoolParam(
	const char *	pszParamName,
	FLMBOOL			bDefaultValue,
	FLMBOOL *		pbBool,
	F_IniFile *		pIniFile)
{
	flmAssert( pIniFile);
	if (!pIniFile->GetParam( pszParamName, pbBool))
	{
		*pbBool = bDefaultValue;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
FSTATIC RCODE flmGetIniFileName(
	FLMBYTE *		pszIniFileName,
	FLMUINT			uiBufferSz
	)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiFileNameLen = 0;
	F_DirHdl *	pDirHdl = NULL;

	if (!uiBufferSz)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_FAILURE);
		goto Exit;
	}

	// See if we have an environment variable to tell us wher the ini file should be.

	f_getenv( "XFLM_INI_PATH",
				 pszIniFileName,
				 uiBufferSz,
				 &uiFileNameLen);

	if( !uiFileNameLen)
	{
		// Perhaps we can find a data directory.  If there is one, we will
		// look in there.

		if( (pDirHdl = f_new F_DirHdl) == NULL)
		{
			rc = RC_SET( NE_XFLM_MEM);
			goto Exit;
		}
		
		if( RC_OK( rc = pDirHdl->OpenDir( (char *)".", (char *)"data")))
		{
			if (RC_OK( rc = pDirHdl->Next()))
			{
				if (pDirHdl->CurrentItemIsDir())
				{
					// Directory exists.  We will look there.
					f_strcpy( pszIniFileName, "data");
				}
				else
				{
					goto NoDataDir;
				}
			}
			else
			{
				rc = NE_XFLM_OK;
				goto NoDataDir;
			}
		}
		else
		{
NoDataDir:
			// Data directory does not exist.  We will look in the
			// current (default) dir.

			f_strcpy( pszIniFileName, ".");
		}
	}

	gv_pFileSystem->pathAppend( (char *)pszIniFileName, XFLM_INI_FILE_NAME);

Exit:

	if (pDirHdl)
	{
		pDirHdl->Release();
	}
	
	return( rc);
}
