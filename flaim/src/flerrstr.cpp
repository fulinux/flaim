//-------------------------------------------------------------------------
// Desc:	Convert check error codes into strings.
// Tabs:	3
//
//		Copyright (c) 1992-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flerrstr.cpp 12262 2006-01-19 14:42:10 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*
**  WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
**					REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaim.h
*/

char *	FlmCorruptStrings[ FLM_LAST_CORRUPT_ERROR]
=	{
	"OK",										// 0
	"BAD_CHAR",								// 1
	"BAD_ASIAN_CHAR",						// 2
	"BAD_CHAR_SET",						// 3
	"BAD_TEXT_FIELD",						// 4
	"BAD_NUMBER_FIELD",					// 5
	"BAD_CONTEXT_FIELD",					// 6
	"BAD_FIELD_TYPE",						// 7
	"BAD_IX_DEF",							// 8
	"MISSING_REQ_KEY_FIELD",			// 9
	"BAD_TEXT_KEY_COLL_CHAR",			// 10
	"BAD_TEXT_KEY_CASE_MARKER",		// 11
	"BAD_NUMBER_KEY",						// 12
	"BAD_CONTEXT_KEY",					// 13
	"BAD_BINARY_KEY",						// 14
	"BAD_DRN_KEY",							// 15
	"BAD_KEY_FIELD_TYPE",				// 16
	"BAD_KEY_COMPOUND_MARKER",			// 17
	"BAD_KEY_POST_MARKER",				// 18
	"BAD_KEY_POST_BYTE_COUNT",			// 19
	"BAD_KEY_LEN",							// 20
	"BAD_LFH_LIST_PTR",					// 21
	"BAD_LFH_LIST_END",					// 22
	"BAD_PCODE_LIST_END",				// 23
	"BAD_BLK_END",							// 24
	"KEY_COUNT_MISMATCH",				// 25
	"REF_COUNT_MISMATCH",				// 26
	"BAD_CONTAINER_IN_KEY",				// 27
	"BAD_BLK_HDR_ADDR",					// 28
	"BAD_BLK_HDR_LEVEL",					// 29
	"BAD_BLK_HDR_PREV",					// 30
	"BAD_BLK_HDR_NEXT",					// 31
	"BAD_BLK_HDR_TYPE",					// 32
	"BAD_BLK_HDR_ROOT_BIT",				// 33
	"BAD_BLK_HDR_BLK_END",				// 34
	"BAD_BLK_HDR_LF_NUM",				// 35
	"BAD_AVAIL_LIST_END",				// 36
	"BAD_PREV_BLK_NEXT",					// 37
	"BAD_FIRST_ELM_FLAG",				// 38
	"BAD_LAST_ELM_FLAG",					// 39
	"BAD_LEM",								// 40
	"BAD_ELM_LEN",							// 41
	"BAD_ELM_KEY_SIZE",					// 42
	"BAD_ELM_PKC_LEN",					// 43
	"BAD_ELM_KEY_ORDER",					// 44
	"BAD_ELM_KEY_COMPRESS",				// 45
	"BAD_CONT_ELM_KEY",					// 46
	"NON_UNIQUE_FIRST_ELM_KEY",		// 47
	"BAD_ELM_FLD_OVERHEAD",				// 48
	"BAD_ELM_FLD_LEVEL_JUMP",			// 49
	"BAD_ELM_FLD_NUM",					// 50
	"BAD_ELM_FLD_LEN",					// 51
	"BAD_ELM_FLD_TYPE",					// 52
	"BAD_ELM_END",							// 53
	"BAD_PARENT_KEY",						// 54
	"BAD_ELM_DOMAIN_SEN",				// 55
	"BAD_ELM_BASE_SEN",					// 56
	"BAD_ELM_IX_REF",						// 57
	"BAD_ELM_ONE_RUN_SEN",				// 58
	"BAD_ELM_DELTA_SEN",					// 59
	"BAD_ELM_DOMAIN",						// 60
	"BAD_LAST_BLK_NEXT",					// 61
	"BAD_FIELD_PTR",						// 62
	"REBUILD_REC_EXISTS",				// 63
	"REBUILD_KEY_NOT_UNIQUE",			// 64
	"NON_UNIQUE_ELM_KEY_REF",			// 65
	"OLD_VIEW",								// 66
	"COULD_NOT_SYNC_BLK",				// 67
	"IX_REF_REC_NOT_FOUND",				// 68
	"IX_KEY_NOT_FOUND_IN_REC",			// 69
	"DRN_NOT_IN_KEY_REFSET",			// 70
	"BAD_BLK_CHECKSUM",					// 71
	"BAD_LAST_DRN",						// 72
	"BAD_FILE_SIZE",						// 73
	"BAD_AVAIL_BLOCK_COUNT",			// 74
	"BAD_DATE_FIELD",						// 75
	"BAD_TIME_FIELD",						// 76
	"BAD_TMSTAMP_FIELD",					// 77
	"BAD_DATE_KEY",    					// 78
	"BAD_TIME_KEY",  						// 79
	"BAD_TMSTAMP_KEY", 					// 80
	"BAD_BLOB_FIELD",						// 81
	"BAD_PCODE_IXD_TBL",					// 82
	"DICT_REC_ADD_ERR",					// 83
	"FLM_BAD_FIELD_FLAG",				// 84
	};

/*API~***********************************************************************
Desc : Returns a pointer to the string representation of a corruption
		 error code.
*END************************************************************************/
char * FlmVerifyErrToStr(
	eCorruptionType	eCorruption
	)
{
	return( FlmCorruptStrings [eCorruption]);
}
