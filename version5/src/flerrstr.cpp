//------------------------------------------------------------------------------
// Desc:	This module contains the checkErrorToStr method - which returns
// 		a pointer to a string which corresponds to a corruption code.
//
// Tabs:	3
//
//		Copyright (c) 1992-1997,2000-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flerrstr.cpp 3113 2006-01-19 13:20:35 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h

char * FlmCorruptStrings[ FLM_NUM_CORRUPT_ERRORS]
=	{
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

// WARNING:	ANY CHANGES MADE TO THE FlmCorruptStrings TABLE MUST BE
// REFLECTED IN THE CHECK CODE DEFINES FOUND IN flaimsys.h



/*API~***********************************************************************
Area : INFORMATION
Desc : Returns a pointer to the string representation of a corruption
		 error code.
Notes: Linking in this routine will bring in static string arrays.
*END************************************************************************/
const char * XFLMAPI F_DbSystem::checkErrorToStr(
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

