//------------------------------------------------------------------------------
// Desc:	This file contains error routines that are used throughout FLAIM.
//
// Tabs:	3
//
//		Copyright (c) 1997-2000, 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flerror.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
Desc:		Error code to string mapping tables
****************************************************************************/
typedef struct
{
	RCODE				rc;
	const char *	pszErrorStr;
} F_ERROR_CODE_MAP;

#define flmErrorCodeEntry(c)		{ c, #c }

/****************************************************************************
Desc:
****************************************************************************/
F_ERROR_CODE_MAP gv_FlmGeneralErrors[
	NE_XFLM_LAST_GENERAL_ERROR - NE_XFLM_FIRST_GENERAL_ERROR - 1] =
{
	flmErrorCodeEntry( NE_XFLM_BAD_PREFIX),
	flmErrorCodeEntry( NE_XFLM_ATTRIBUTE_PURGED),
	flmErrorCodeEntry( NE_XFLM_BAD_COLLECTION),
	flmErrorCodeEntry( NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_DATA_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_MUST_INDEX_ON_PRESENCE),
	flmErrorCodeEntry( NE_XFLM_BAD_IX),
	flmErrorCodeEntry( NE_XFLM_BACKUP_ACTIVE),
	flmErrorCodeEntry( NE_XFLM_SERIAL_NUM_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_DB_SERIAL_NUM),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_FILE_NUMBER),
	flmErrorCodeEntry( NE_XFLM_CANNOT_DEL_ELEMENT),
	flmErrorCodeEntry( NE_XFLM_CANNOT_MOD_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_CANNOT_INDEX_DATA_TYPE),
	flmErrorCodeEntry( NE_XFLM_BAD_ELEMENT_NUM),
	flmErrorCodeEntry( NE_XFLM_BAD_ATTRIBUTE_NUM),
	flmErrorCodeEntry( NE_XFLM_BAD_ENCDEF_NUM),
	flmErrorCodeEntry( NE_XFLM_INVALID_FILE_SEQUENCE),
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
	flmErrorCodeEntry( NE_XFLM_NOT_FLAIM),
	flmErrorCodeEntry( NE_XFLM_OLD_VIEW),
	flmErrorCodeEntry( NE_XFLM_SHARED_LOCK),
	flmErrorCodeEntry( NE_XFLM_TRANS_ACTIVE),
	flmErrorCodeEntry( NE_XFLM_RFL_TRANS_GAP),
	flmErrorCodeEntry( NE_XFLM_BAD_COLLATED_KEY),
	flmErrorCodeEntry( NE_XFLM_MUST_DELETE_INDEXES),
	flmErrorCodeEntry( NE_XFLM_RFL_INCOMPLETE),
	flmErrorCodeEntry( NE_XFLM_CANNOT_RESTORE_RFL_FILES),
	flmErrorCodeEntry( NE_XFLM_INCONSISTENT_BACKUP),
	flmErrorCodeEntry( NE_XFLM_BLOCK_CRC),
	flmErrorCodeEntry( NE_XFLM_ABORT_TRANS),
	flmErrorCodeEntry( NE_XFLM_NOT_RFL),
	flmErrorCodeEntry( NE_XFLM_BAD_RFL_PACKET),
	flmErrorCodeEntry( NE_XFLM_DATA_PATH_MISMATCH),
	flmErrorCodeEntry( NE_XFLM_MUST_CLOSE_DATABASE),
	flmErrorCodeEntry( NE_XFLM_INVALID_ENCKEY_CRC),
	flmErrorCodeEntry( NE_XFLM_HDR_CRC),
	flmErrorCodeEntry( NE_XFLM_NO_NAME_TABLE),
	flmErrorCodeEntry( NE_XFLM_UNALLOWED_UPGRADE),
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
	flmErrorCodeEntry( NE_XFLM_DB_FULL),
	flmErrorCodeEntry( NE_XFLM_QUERY_SYNTAX),
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
	flmErrorCodeEntry( NE_XFLM_MISSING_ENCDEF_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_ENCDEF_NUMBER),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_INDEX_NUMS),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_COLLECTION_NUMS),
	flmErrorCodeEntry( NE_XFLM_CANNOT_DEL_ATTRIBUTE),
	flmErrorCodeEntry( NE_XFLM_TOO_MANY_PENDING_NODES),
	flmErrorCodeEntry( NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG),
	flmErrorCodeEntry( NE_XFLM_DUP_SIBLING_IX_COMPONENTS),
	flmErrorCodeEntry( NE_XFLM_RFL_FILE_NOT_FOUND),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_PREFIX_NUMBER),
	flmErrorCodeEntry( NE_XFLM_MISSING_PREFIX_NAME),
	flmErrorCodeEntry( NE_XFLM_MISSING_PREFIX_NUMBER),
	flmErrorCodeEntry( NE_XFLM_UNDEFINED_ELEMENT_NAME),
	flmErrorCodeEntry( NE_XFLM_UNDEFINED_ATTRIBUTE_NAME),
	flmErrorCodeEntry( NE_XFLM_DUPLICATE_PREFIX_NAME),
	flmErrorCodeEntry( NE_XFLM_NAMESPACE_NOT_ALLOWED),
	flmErrorCodeEntry( NE_XFLM_INVALID_NAMESPACE_DECL),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_PREFIX_NUMS),
	flmErrorCodeEntry( NE_XFLM_NO_MORE_ENCDEF_NUMS),
	flmErrorCodeEntry( NE_XFLM_COLLECTION_OFFLINE),
	flmErrorCodeEntry( NE_XFLM_DELETE_NOT_ALLOWED),
	flmErrorCodeEntry( NE_XFLM_RESET_NEEDED),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_REQUIRED_VALUE),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_INDEX_COMPONENT),
	flmErrorCodeEntry( NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE),
	flmErrorCodeEntry( NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_REQUIRED),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_LIMIT),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_INDEX_ON),
	flmErrorCodeEntry( NE_XFLM_CANNOT_SET_COMPARE_RULES),
	flmErrorCodeEntry( NE_XFLM_INPUT_PENDING),
	flmErrorCodeEntry( NE_XFLM_INVALID_NODE_TYPE),
	flmErrorCodeEntry( NE_XFLM_INVALID_CHILD_ELM_NODE_ID),
	flmErrorCodeEntry( NE_XFLM_RFL_END)
};

/****************************************************************************
Desc:
****************************************************************************/
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

/****************************************************************************
Desc:
****************************************************************************/
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
Desc:
****************************************************************************/
char * FlmCorruptStrings[ FLM_NUM_CORRUPT_ERRORS] =
{
	"BAD_CHAR",								/*1*/
	"BAD_ASIAN_CHAR",						/*2*/
	"BAD_CHAR_SET",						/*3*/
	"BAD_TEXT_FIELD",						/*4*/
	"BAD_NUMBER_FIELD",					/*5*/
	"BAD_FIELD_TYPE",						/*6*/
	"BAD_IX_DEF",							/*7*/
	"MISSING_REQ_KEY_FIELD",			/*8*/
	"BAD_TEXT_KEY_COLL_CHAR",			/*9*/
	"BAD_TEXT_KEY_CASE_MARKER",		/*10*/
	"BAD_NUMBER_KEY",						/*11*/
	"BAD_BINARY_KEY",						/*12*/
	"BAD_CONTEXT_KEY",					/*13*/
	"BAD_KEY_FIELD_TYPE",				/*14*/
	"Not_Used_15",							/*15*/
	"Not_Used_16",							/*16*/
	"Not_Used_17",							/*17*/
	"BAD_KEY_LEN",							/*18*/
	"BAD_LFH_LIST_PTR",					/*19*/
	"BAD_LFH_LIST_END",					/*20*/
	"INCOMPLETE_NODE",					/*21*/
	"BAD_BLK_END",							/*22*/
	"KEY_COUNT_MISMATCH",				/*23*/
	"REF_COUNT_MISMATCH",				/*24*/
	"BAD_CONTAINER_IN_KEY",				/*25*/
	"BAD_BLK_HDR_ADDR",					/*26*/
	"BAD_BLK_HDR_LEVEL",					/*27*/
	"BAD_BLK_HDR_PREV",					/*28*/

// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

	"BAD_BLK_HDR_NEXT",					/*29*/
	"BAD_BLK_HDR_TYPE",					/*30*/
	"BAD_BLK_HDR_ROOT_BIT",				/*31*/
	"BAD_BLK_HDR_BLK_END",				/*32*/
	"BAD_BLK_HDR_LF_NUM",				/*33*/
	"BAD_AVAIL_LIST_END",				/*34*/
	"BAD_PREV_BLK_NEXT",					/*35*/
	"BAD_FIRST_LAST_ELM_FLAG",			/*36*/
	"nu",										/*37*/
	"BAD_LEM",								/*38*/
	"BAD_ELM_LEN",							/*39*/
	"BAD_ELM_KEY_SIZE",					/*40*/
	"BAD_ELM_KEY",							/*41*/
	"BAD_ELM_KEY_ORDER",					/*42*/
	"nu",										/*43*/
	"BAD_CONT_ELM_KEY",					/*44*/
	"NON_UNIQUE_FIRST_ELM_KEY",		/*45*/
	"BAD_ELM_OFFSET",						/*46*/
	"BAD_ELM_INVALID_LEVEL",			/*47*/
	"BAD_ELM_FLD_NUM",					/*48*/
	"BAD_ELM_FLD_LEN",					/*49*/
	"BAD_ELM_FLD_TYPE",					/*50*/
	"BAD_ELM_END",							/*51*/
	"BAD_PARENT_KEY",						/*52*/
	"BAD_ELM_DOMAIN_SEN",				/*53*/
	"BAD_ELM_BASE_SEN",					/*54*/
	"BAD_ELM_IX_REF",						/*55*/
	"BAD_ELM_ONE_RUN_SEN",				/*56*/
	"BAD_ELM_DELTA_SEN",					/*57*/
	"BAD_ELM_DOMAIN",						/*58*/

// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

	"BAD_LAST_BLK_NEXT",					/*59*/
	"BAD_FIELD_PTR",						/*60*/
	"REBUILD_REC_EXISTS",				/*61*/
	"REBUILD_KEY_NOT_UNIQUE",			/*62*/
	"NON_UNIQUE_ELM_KEY_REF",			/*63*/
	"OLD_VIEW",								/*64*/
	"COULD_NOT_SYNC_BLK",				/*65*/
	"IX_REF_REC_NOT_FOUND",				/*66*/
	"IX_KEY_NOT_FOUND_IN_REC",			/*67*/
	"KEY_NOT_IN_KEY_REFSET",			/*68*/
	"BAD_BLK_CHECKSUM",					/*69*/
	"BAD_LAST_DRN",						/*70*/
	"BAD_FILE_SIZE",						/*71*/
	"nu",										/*72*/
	"BAD_DATE_FIELD",						/*73*/
	"BAD_TIME_FIELD",						/*74*/
	"BAD_TMSTAMP_FIELD",					/*75*/
	"BAD_DATE_KEY",    					/*76*/
	"BAD_TIME_KEY",  						/*77*/
	"BAD_TMSTAMP_KEY", 					/*78*/
	"BAD_BLOB_FIELD",						/*79*/

// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

	"BAD_PCODE_IXD_TBL",					/*80*/
	"NODE_QUARANTINED",					/*81*/
	"BAD_BLK_TYPE",						/*82*/
	"BAD_ELEMENT_CHAIN",					/*83*/
	"BAD_ELM_EXTR_DATA",					/*84*/
	"BAD_BLOCK_STRUCTURE",				/*85*/
	"BAD_ROOT_PARENT",					/*86*/
	"BAD_ROOT_LINK",						/*87*/
	"BAD_PARENT_LINK",					/*88*/
	"BAD_INVALID_ROOT",					/*89*/
	"BAD_FIRST_CHILD_LINK",				/*90*/
	"BAD_LAST_CHILD_LINK",				/*91*/
	"BAD_PREV_SIBLING_LINK",			/*92*/
	"BAD_NEXT_SIBLING_LINK",			/*93*/
	"BAD_ANNOTATION_LINK",				/*95*/
	"UNSUPPORTED_NODE_TYPE",			/*96*/
	"BAD_INVALID_NAME_ID",				/*97*/
	"BAD_INVALID_PREFIX_ID",			/*98*/
	"BAD_DATA_BLOCK_COUNT",				/*99*/
	"FLM_BAD_AVAIL_SIZE",				/*100*/
	"BAD_NODE_TYPE",						/*101*/
	"BAD_CHILD_ELM_COUNT",				/*102*/
};

/****************************************************************************
Desc:	The primary purpose of this function is to provide a way to easily
		trap errors when they occur.  Just put a breakpoint in this function
		to catch them.
Note:	Some of the most common errors will be coded so the use can set a
		break point.
****************************************************************************/
#ifdef FLM_DEBUG
RCODE flmMakeErr(
	RCODE				rc,
	const char *	pszFile,
	int				iLine,
	FLMBOOL			bAssert)
{
	if( rc == NE_XFLM_OK)
	{
		return NE_XFLM_OK;
	}

	// Switch on warning type return codes
	if( rc <= NE_XFLM_NOT_FOUND)
	{
		switch(rc)
		{
			case NE_XFLM_BOF_HIT:
				break;
			case NE_XFLM_EOF_HIT:
				break;
			case NE_XFLM_RFL_END:
				break;
			case NE_XFLM_EXISTS:
				break;
			case NE_XFLM_NOT_FOUND:
				break;
		}

		goto Exit;
	}

	switch(rc)
	{
		case NE_FLM_IO_BAD_FILE_HANDLE:
			break;
		case NE_XFLM_DATA_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_BTREE_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_MEM:
			break;
		case NE_XFLM_OLD_VIEW:
			break;
		case NE_XFLM_SYNTAX:
			break;
		case NE_XFLM_BLOCK_CRC:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_CACHE_ERROR:
			flmLogError( rc, "", pszFile, iLine);
			break;
		case NE_XFLM_NOT_IMPLEMENTED:
			break;
		case NE_XFLM_CONV_DEST_OVERFLOW:
			break;
		case NE_XFLM_KEY_OVERFLOW:
			break;
		case NE_XFLM_FAILURE:
			break;
		case NE_XFLM_ILLEGAL_OP:
			break;
		case NE_XFLM_BAD_COLLECTION:
			break;
		default:
			rc = rc;
			break;
	}

Exit:
	
#if defined( FLM_DEBUG)
	if( bAssert)
	{
		flmAssert( 0);
	}
#else
	F_UNREFERENCED_PARM( bAssert);
#endif

	return( rc);
}
#endif

#if defined( FLM_WATCOM_NLM)
	int gv_iFlerrorDummy(void)
	{
		return( 0);
	}
#endif

/****************************************************************************
Desc:	Returns a pointer to the string representation of a corruption
		error code.
****************************************************************************/
const char * FLMAPI F_DbSystem::checkErrorToStr(
	FLMINT	iCheckErrorCode)
{
	if( (iCheckErrorCode >= 1) && (iCheckErrorCode <= FLM_NUM_CORRUPT_ERRORS))
	{
		return( FlmCorruptStrings [iCheckErrorCode - 1]);
	}
	else if( iCheckErrorCode == 0)
	{
		return( "OK");
	}
	else
	{
		return( "Unknown Error");
	}
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
	else if( f_isToolkitError( rc))
	{
		pszErrorStr = f_errorString( rc);
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
	else if( rc > NE_XFLM_FIRST_QUERY_ERROR &&
		rc < NE_XFLM_LAST_QUERY_ERROR)
	{
		pszErrorStr = gv_FlmQueryErrors[
			rc - NE_XFLM_FIRST_QUERY_ERROR - 1].pszErrorStr;
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
