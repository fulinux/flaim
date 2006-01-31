//------------------------------------------------------------------------------
// Desc:	Contains interfaces for reading FLAIM 4.x databases.
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
// $Id: frecread.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FRECREAD_H
#define FRECREAD_H

class F_Block;
class F_4xNameTable;

#define FLM_4x_RESERVED_TAG_NUMS						32100
#define FLM_4x_UNREGISTERED_TAGS						32769
#define FLM_4x_DICT_CONTAINER							32000
#define FLM_4x_DATA_CONTAINER							32001
#define FLM_4x_TRACKER_CONTAINER						32002
#define FLM_4x_DICT_INDEX								32003
#define FLM_4x_DICT_FIELD_NUMS						32100
#define FLM_4x_LAST_DICT_FIELD_NUM					32175

#define FLM_4x_BH_MAX_LEVELS							8
#define FLM_4x_BH_CHECKSUM_LOW						0
#define FLM_4x_BH_ADDR									0
#define FLM_4x_BH_PREV_BLK								4
#define FLM_4x_BH_NEXT_BLK								8
#define FLM_4x_BH_TYPE									12
#define FLM_4x_BH_LEVEL									13
#define FLM_4x_BH_ELM_END								14
#define FLM_4x_BH_TRANS_ID								16
#define FLM_4x_BH_PREV_TRANS_ID						20
#define FLM_4x_BH_PREV_BLK_ADDR						24
#define FLM_4x_BH_LOG_FILE_NUM						28
#define FLM_4x_BH_CHECKSUM_HIGH						31
#define FLM_4x_BH_OVHD									32

#define FLM_4x_BT_END		  							0xFFFFFFFF

#define FLM_4x_BHT_FREE 								0		// Free block - avail list
#define FLM_4x_BHT_LEAF 								1		// Leaf block
#define FLM_4x_BHT_LFH_BLK								4		// LFH Header block
#define FLM_4x_BHT_NON_LEAF							6		// Non-leaf block - variable key size
#define FLM_4x_BHT_NON_LEAF_DATA						7		// Non-leaf block data block - fixed key size
#define FLM_4x_BHT_NON_LEAF_COUNTS					8		// Non-leaf index with counts

#define FLM_4x_BH_GET_TYPE(blk)						(((blk)[ FLM_4x_BH_TYPE]) & 0x0F)
#define FLM_4x_BH_IS_ROOT_BLK(blk)					(((blk)[ FLM_4x_BH_TYPE]) & 0x80)

#define FLM_4x_FOP_RECORD_INFO						0xB0
#define FLM_4x_FOP_IS_RECORD_INFO(p)				((*(p) & 0xFE) == FLM_4x_FOP_RECORD_INFO)

#define FLM_4x_TEXT_TYPE								0
#define FLM_4x_NUMBER_TYPE								1
#define FLM_4x_BINARY_TYPE								2
#define FLM_4x_CONTEXT_TYPE							3
#define FLM_4x_BLOB_TYPE								8
#define FLM_4x_UNKNOWN_TYPE							0xFFFFFFFF

#define FLM_4x_FIELD_TAG								(FLM_4x_RESERVED_TAG_NUMS +  0)
#define FLM_4x_INDEX_TAG								(FLM_4x_RESERVED_TAG_NUMS +  1)
#define FLM_4x_TYPE_TAG									(FLM_4x_RESERVED_TAG_NUMS +  2)
#define FLM_4x_CONTAINER_TAG							(FLM_4x_RESERVED_TAG_NUMS +  4)
#define FLM_4x_LANGUAGE_TAG							(FLM_4x_RESERVED_TAG_NUMS +  5)
#define FLM_4x_OPTIONAL_TAG							(FLM_4x_RESERVED_TAG_NUMS +  6)
#define FLM_4x_UNIQUE_TAG								(FLM_4x_RESERVED_TAG_NUMS +  7)
#define FLM_4x_KEY_TAG									(FLM_4x_RESERVED_TAG_NUMS +  8)
#define FLM_4x_REFS_TAG									(FLM_4x_RESERVED_TAG_NUMS +  9)
#define FLM_4x_AREA_TAG									(FLM_4x_RESERVED_TAG_NUMS + 17)
#define FLM_4x_STATE_TAG								(FLM_4x_RESERVED_TAG_NUMS + 25)
#define FLM_4x_BLOB_TAG									(FLM_4x_RESERVED_TAG_NUMS + 26)
#define FLM_4x_THRESHOLD_TAG							(FLM_4x_RESERVED_TAG_NUMS + 27)
#define FLM_4x_SUFFIX_TAG								(FLM_4x_RESERVED_TAG_NUMS + 29)
#define FLM_4x_SUBDIRECTORY_TAG						(FLM_4x_RESERVED_TAG_NUMS + 30)
#define FLM_4x_RESERVED_TAG							(FLM_4x_RESERVED_TAG_NUMS + 31)
#define FLM_4x_SUBNAME_TAG								(FLM_4x_RESERVED_TAG_NUMS + 32)
#define FLM_4x_NAME_TAG									(FLM_4x_RESERVED_TAG_NUMS + 33)
#define FLM_4x_BASE_TAG									(FLM_4x_RESERVED_TAG_NUMS + 36)
#define FLM_4x_CASE_TAG									(FLM_4x_RESERVED_TAG_NUMS + 38)
#define FLM_4x_COMBINATIONS_TAG						(FLM_4x_RESERVED_TAG_NUMS + 40)
#define FLM_4x_COUNT_TAG								(FLM_4x_RESERVED_TAG_NUMS + 41)
#define FLM_4x_POSITIONING_TAG						(FLM_4x_RESERVED_TAG_NUMS + 42)
#define FLM_4x_PAIRED_TAG								(FLM_4x_RESERVED_TAG_NUMS + 44)
#define FLM_4x_PARENT_TAG								(FLM_4x_RESERVED_TAG_NUMS + 45)
#define FLM_4x_POST_TAG									(FLM_4x_RESERVED_TAG_NUMS + 46)
#define FLM_4x_REQUIRED_TAG							(FLM_4x_RESERVED_TAG_NUMS + 47)
#define FLM_4x_USE_TAG									(FLM_4x_RESERVED_TAG_NUMS + 48)
#define FLM_4x_FILTER_TAG								(FLM_4x_RESERVED_TAG_NUMS + 49)
#define FLM_4x_LIMIT_TAG								(FLM_4x_RESERVED_TAG_NUMS + 50)
#define FLM_4x_DICT_TAG									(FLM_4x_RESERVED_TAG_NUMS + 54)
#define FLM_4x_RECINFO_TAG								(FLM_4x_RESERVED_TAG_NUMS + 70)
#define FLM_4x_DRN_TAG									(FLM_4x_RESERVED_TAG_NUMS + 71)
#define FLM_4x_DICT_SEQ_TAG							(FLM_4x_RESERVED_TAG_NUMS + 72)
#define FLM_4x_LAST_CONTAINER_INDEXED_TAG			(FLM_4x_RESERVED_TAG_NUMS + 73)
#define FLM_4x_LAST_DRN_INDEXED_TAG					(FLM_4x_RESERVED_TAG_NUMS + 74)
#define FLM_4x_ONLINE_TRANS_ID_TAG					(FLM_4x_RESERVED_TAG_NUMS + 75)

#define FLM_4x_FIELD_TAG_NAME							"Field"
#define FLM_4x_INDEX_TAG_NAME							"Index"
#define FLM_4x_TYPE_TAG_NAME							"Type"
#define FLM_4x_CONTAINER_TAG_NAME					"Container"
#define FLM_4x_LANGUAGE_TAG_NAME						"Language"
#define FLM_4x_OPTIONAL_TAG_NAME						"Optional"
#define FLM_4x_UNIQUE_TAG_NAME						"Unique"
#define FLM_4x_KEY_TAG_NAME							"Key"
#define FLM_4x_REFS_TAG_NAME							"Refs"
#define FLM_4x_AREA_TAG_NAME							"Area"
#define FLM_4x_STATE_TAG_NAME							"State"
#define FLM_4x_BLOB_TAG_NAME							"Blob"
#define FLM_4x_THRESHOLD_TAG_NAME					"Threshold"
#define FLM_4x_SUFFIX_TAG_NAME						"Suffix"
#define FLM_4x_SUBDIRECTORY_TAG_NAME				"Subdirectory"
#define FLM_4x_RESERVED_TAG_NAME						"Reserved"
#define FLM_4x_SUBNAME_TAG_NAME						"Subname"
#define FLM_4x_NAME_TAG_NAME							"Name"
#define FLM_4x_BASE_TAG_NAME							"Base"
#define FLM_4x_CASE_TAG_NAME							"Case"
#define FLM_4x_COMBINATIONS_TAG_NAME				"Combinations"
#define FLM_4x_COUNT_TAG_NAME							"Count"
#define FLM_4x_POSITIONING_TAG_NAME					"Positioning"
#define FLM_4x_PAIRED_TAG_NAME						"Paired"
#define FLM_4x_PARENT_TAG_NAME						"Parent"
#define FLM_4x_POST_TAG_NAME							"Post"
#define FLM_4x_REQUIRED_TAG_NAME						"Required"
#define FLM_4x_USE_TAG_NAME							"Use"
#define FLM_4x_FILTER_TAG_NAME						"Filter"
#define FLM_4x_LIMIT_TAG_NAME							"Limit"
#define FLM_4x_DICT_TAG_NAME							"Dict"
#define FLM_4x_RECINFO_TAG_NAME						"RecInfo"
#define FLM_4x_DRN_TAG_NAME							"Drn"
#define FLM_4x_DICT_SEQ_TAG_NAME						"DictSeq"
#define FLM_4x_LAST_CONTAINER_INDEXED_TAG_NAME	"LastContainerIndexed"
#define FLM_4x_LAST_DRN_INDEXED_TAG_NAME			"LastDrnIndexed"
#define FLM_4x_ONLINE_TRANS_ID_TAG_NAME			"OnlineTransId"

#define FLM_4x_ASCII_CHAR_CODE						0x00
#define FLM_4x_ASCII_CHAR_MASK						0x80
#define FLM_4x_CHAR_SET_CODE	 						0x80
#define FLM_4x_CHAR_SET_MASK	 						0xC0
#define FLM_4x_WHITE_SPACE_CODE						0xC0
#define FLM_4x_WHITE_SPACE_MASK						0xE0
#define FLM_4x_UNK_GT_255_CODE						0xE0
#define FLM_4x_UNK_GT_255_MASK						0xF8
#define FLM_4x_UNK_EQ_1_CODE	 						0xF0
#define FLM_4x_UNK_EQ_1_MASK	 						0xF8
#define FLM_4x_UNK_LE_255_CODE						0xF8
#define FLM_4x_UNK_LE_255_MASK						0xF8
#define FLM_4x_EXT_CHAR_CODE	 						0xE8
#define FLM_4x_OEM_CODE			 						0xE9
#define FLM_4x_UNICODE_CODE							0xEA

#define FLM_4x_FOP_IS_STANDARD(p)					(!(*(p) & 0x80) )
#define FLM_4x_FOP_OPEN									0x90
#define FLM_4x_FOP_IS_OPEN(p)							((*(p) & 0xF0) == FLM_4x_FOP_OPEN)
#define FLM_4x_FOP_GET_FLD_FLAGS(p)					((*p) & 0x07)
#define FLM_4x_FOP_2BYTE_FLDNUM(bv)					((bv) & 0x02)
#define FLM_4x_FOP_2BYTE_FLDLEN(bv)					((bv) & 0x01)
#define FLM_4x_FOP_TAGGED								0x80
#define FLM_4x_FOP_IS_TAGGED(p)						((*(p) & 0xF0) == FLM_4x_FOP_TAGGED)
#define FLM_4x_FOP_NO_VALUE							0xA8
#define FLM_4x_FOP_IS_NO_VALUE(p)					((*(p) & 0xF8) == FLM_4x_FOP_NO_VALUE)
#define FLM_4x_FOP_SET_LEVEL							0xA0
#define FLM_4x_FOP_IS_SET_LEVEL(p)					((*(p) & 0xF8) == FLM_4x_FOP_SET_LEVEL)
#define FLM_4x_FOP_LEVEL_MAX							0x07
#define FLM_4x_FSLEV_GET(p)							(*(p) & FLM_4x_FOP_LEVEL_MAX)
#define FLM_4x_FTAG_LEVEL(p)							((*p) & 0x08)
#define FLM_4x_FTAG_FLD_TYPE(p)						((*(p+1)) & 0x0F)
#define FLM_4x_FOPE_LEVEL(p)							((*p) & 0x08)
#define FLM_4x_FSTA_FLD_LEN(p)						((*p) & 0x3F)
#define FLM_4x_FSTA_LEVEL(p)							((*p) & 0x40)
#define FLM_4x_FSTA_FLD_NUM(p)						(*(p+1))
#define FLM_4x_FSTA_OVHD								2
#define FLM_4x_FTAG_OVHD								2
#define FLM_4x_FNOV_LEVEL(p)							((*p) & 0x04)

#define FLM_4x_BBE_FIRST_FLAG							0x80
#define FLM_4x_BBE_LAST_FLAG							0x40
#define FLM_4x_BBE_IS_FIRST(elm)						((*(elm)) & FLM_4x_BBE_FIRST_FLAG )
#define FLM_4x_BBE_IS_LAST(elm)						((*(elm)) & FLM_4x_BBE_LAST_FLAG )
#define FLM_4x_BBE_KL_HBITS							0x30
#define FLM_4x_BBE_PKC									0
#define FLM_4x_BBE_PKC_MAX								0x0F
#define FLM_4x_BBE_GET_PKC(elm)						((*(elm)) & FLM_4x_BBE_PKC_MAX)
#define FLM_4x_BBE_KL									1
#define FLM_4x_BBE_KL_SHIFT_BITS						4
#define FLM_4x_BBE_GET_KL(elm)						(((*(elm) & FLM_4x_BBE_KL_HBITS) << FLM_4x_BBE_KL_SHIFT_BITS) + \
																(elm)[ FLM_4x_BBE_KL])
#define FLM_4x_BBE_GETR_PKC(elm)						(*(elm) & 0x3F)
#define FLM_4x_BBE_GETR_KL(elm)						((elm)[FLM_4x_BBE_KL])
#define FLM_4x_BBE_RL									2
#define FLM_4x_BBE_GET_RL(elm)						((elm)[ FLM_4x_BBE_RL])
#define FLM_4x_BBE_KEY									3
#define FLM_4x_BBE_LEN(elm)							(FLM_4x_BBE_GET_RL(elm) + FLM_4x_BBE_GET_KL(elm) + FLM_4x_BBE_KEY)
#define FLM_4x_BBE_LEM_LEN								3
#define FLM_4x_BBE_REC_OFS(elm)						(FLM_4x_BBE_GET_KL(elm) + FLM_4x_BBE_KEY)
#define FLM_4x_BBE_REC_PTR(elm)						(&(elm)[ FLM_4x_BBE_REC_OFS(elm)])

#define FLM_4x_BNE_DATA_CHILD_BLOCK					4
#define FLM_4x_BNE_DATA_OVHD							8
#define FLM_4x_BNE_DOMAIN								0x80
#define FLM_4x_BNE_IS_DOMAIN(elm)					((*(elm)) & FLM_4x_BNE_DOMAIN)
#define FLM_4x_BNE_DOMAIN_LEN							3
#define FLM_4x_BNE_KEY_START							6
#define FLM_4x_BNE_KEY_COUNTS_START					10

#define FLM_4x_DIN_KEY_SIZ								4
#define FLM_4x_MAX_KEY_SIZ								640
#define FLM_4x_IXD_UNIQUE								0x00001

#define FLM_4x_DRN_LAST_MARKER						((FLMUINT) 0xFFFFFFFF)

#define FLM_4x_GET_BH_ADDR( pBlk)	 				(FB2UD(&(pBlk)[ FLM_4x_BH_ADDR]))
#define FLM_4x_BLK_ELM( stack,elm)					((stack)->pBlk[ (elm)])
#define FLM_4x_BLK_ELM_ADDR(stack,elm)				(&((stack)->pBlk->m_pucBlk[ (elm)]))
#define FLM_4x_CURRENT_ELM(stack)					(&((stack)->pBlk->m_pucBlk[ stack->uiCurElm]))
#define FLM_4x_FOP_IS_OPEN(p)							((*(p) & 0xF0) == FLM_4x_FOP_OPEN)

#define FLM_4x_ASCII_CHAR_CODE						0x00	// 0nnnnnnn
#define FLM_4x_CHAR_SET_CODE	 						0x80	// 10nnnnnn
#define FLM_4x_WHITE_SPACE_CODE						0xC0	// 110nnnnn
#define FLM_4x_WHITE_SPACE_MASK						0xE0	// 11100000
#define FLM_4x_UNK_EQ_1_CODE	 						0xF0	// 11110nnn
#define FLM_4x_EXT_CHAR_CODE	 						0xE8	// 11101000
#define FLM_4x_OEM_CODE			 						0xE9	// 11101001
#define FLM_4x_UNICODE_CODE							0xEA	// 11101010

#define FLM_4x_FLM_FILE_HEADER_SIZE					172
#define FLM_4x_FLAIM_HEADER_START					(2048 - FLM_4x_FLM_FILE_HEADER_SIZE)
#define FLM_4x_DB_LOG_HEADER_START					16
#define FLM_4x_FLAIM_NAME_POS							0
#define FLM_4x_VER_POS									5
#define FLM_4x_MINOR_VER_POS 							(FLM_4x_VER_POS + 2)
#define FLM_4x_SMINOR_VER_POS   						(FLM_4x_VER_POS + 3)
#define FLM_4x_DB_DEFAULT_LANGUAGE					13
#define FLM_4x_DB_BLOCK_SIZE		  					14
#define FLM_4x_DB_1ST_LFH_ADDR						32

#define FLM_4x_LOG_RFL_FILE_NUM						0
#define FLM_4x_LOG_RFL_LAST_TRANS_OFFSET			4
#define FLM_4x_LOG_RFL_LAST_CP_FILE_NUM			8
#define FLM_4x_LOG_RFL_LAST_CP_OFFSET				12
#define FLM_4x_LOG_ROLLBACK_EOF						16
#define FLM_4x_LOG_INC_BACKUP_SEQ_NUM				20
#define FLM_4x_LOG_CURR_TRANS_ID						24
#define FLM_4x_LOG_COMMIT_COUNT						28
#define FLM_4x_LOG_PL_FIRST_CP_BLOCK_ADDR			32
#define FLM_4x_LOG_LAST_RFL_FILE_DELETED			36
#define FLM_4x_LOG_RFL_MIN_FILE_SIZE				40
#define FLM_4x_LOG_HDR_CHECKSUM		 				44
#define FLM_4x_LOG_FLAIM_VERSION						46
#define FLM_4x_LOG_LAST_BACKUP_TRANS_ID			48
#define FLM_4x_LOG_BLK_CHG_SINCE_BACKUP			52
#define FLM_4x_LOG_LAST_CP_TRANS_ID					56
#define FLM_4x_LOG_PF_FIRST_BACKCHAIN				60
#define FLM_4x_LOG_PF_AVAIL_BLKS		 				64
#define FLM_4x_LOG_LOGICAL_EOF    	 				68
#define FLM_4x_LOG_LAST_RFL_COMMIT_ID				72
#define FLM_4x_LOG_KEEP_ABORTED_TRANS_IN_RFL		76
#define FLM_4x_LOG_PF_FIRST_BC_CNT	 				77
#define FLM_4x_LOG_KEEP_RFL_FILES					78
#define FLM_4x_LOG_AUTO_TURN_OFF_KEEP_RFL			79
#define FLM_4x_LOG_PF_NUM_AVAIL_BLKS 				80
#define FLM_4x_LOG_RFL_MAX_FILE_SIZE				84
#define FLM_4x_LOG_DB_SERIAL_NUM						88
#define FLM_4x_LOG_LAST_TRANS_RFL_SERIAL_NUM		104
#define FLM_4x_LOG_RFL_NEXT_SERIAL_NUM				120
#define FLM_4x_LOG_INC_BACKUP_SERIAL_NUM			136
#define FLM_4x_LOG_NU_152_153							152
#define FLM_4x_LOG_MAX_FILE_SIZE						154

#define FLM_4x_LFH_LF_NUMBER_OFFSET					0
#define FLM_4x_LFH_TYPE_OFFSET						2
#define FLM_4x_LFH_STATUS_OFFSET						3
#define FLM_4x_LFH_ROOT_BLK_OFFSET					4
#define FLM_4x_LFH_NEXT_DRN_OFFSET					12
#define FLM_4x_LFH_MAX_FILL_OFFSET					16
#define FLM_4x_LFH_MIN_FILL_OFFSET					17
#define FLM_4x_LFH_SIZE									32

#define FLM_4x_LFILE_DATA_CONTAINER_OFFSET		0
#define FLM_4x_LFILE_DICT_CONTAINER_OFFSET		1
#define FLM_4x_LFILE_DICT_INDEX_OFFSET				2
#define FLM_4x_LFILE_TRACKER_CONTAINER_OFFSET	3

#define FLM_VER_4_0										400
#define FLM_VER_4_3										430
#define FLM_VER_4_31										431
#define FLM_VER_4_50										450
#define FLM_VER_4_51										451

/****************************************************************************
Desc:		
****************************************************************************/
typedef struct
{
	F_Block *	pBlk;
	FLMBYTE		ucKeyBuf[ FLM_4x_DIN_KEY_SIZ];
	FLMUINT		uiKeyLen;
	FLMUINT		uiPKC;
	FLMUINT		uiPrevElmPKC;
	FLMUINT		uiBlkAddr;
	FLMUINT		uiCurElm;
	FLMUINT		uiBlkEnd;
	FLMUINT		uiElmOvhd;
	FLMUINT		uiBlkType;
	FLMUINT		uiLevel;
} BTSK;

typedef struct
{
	FLMBYTE *	pElement;
	FLMUINT		uiFieldType;
	FLMUINT		uiFieldLen;
	FLMUINT		uiPosInElm;
	FLMUINT		uiTagNum;
	FLMUINT		uiLevel;
} FSTATE;						

typedef struct Data_Piece
{
	FLMBYTE *				pData;
	FLMUINT					uiLength;
	struct Data_Piece *	pNext;
} DATAPIECE;

typedef struct
{
	FLMUINT		uiLevel;
	FLMUINT		uiFieldID;
	FLMUINT		uiFieldType;
	FLMUINT		uiFieldLen;
	DATAPIECE	DataPiece;
} TFIELD;

typedef struct Field_Group
{
#define NUM_FIELDS_IN_ARRAY			16
	TFIELD			pFields[ NUM_FIELDS_IN_ARRAY];	// Allocated array of fields
	struct Field_Group * pNext;							// Next temporary field group
} FLDGROUP;

typedef struct Locked_Block
{
	F_Block *					pBlock;
	struct Locked_Block *	pNext;
} LOCKED_BLOCK;

typedef struct FlmField
{
	FLMUINT16	ui16FieldID;
	FLMUINT8		ui8Level;
	FLMUINT8		ui8Type;
	FLMUINT		uiDataLength;		
	FLMUINT		uiDataOffset;
	FlmField *	pNext;
	FlmField *	pPrev;
} FIELD;

typedef struct
{
	FLMUNICODE *	puzTagName;
	FLMUINT			uiTagNum;
	FLMUINT			uiType;
	FLMUINT			uiSubType;
} FLM_4x_TAG_INFO;

typedef struct
{
	FLMBYTE *	pucPtr;
	FLMUINT		uiNibCnt;
	FLMUINT		uiNum;
	FLMBOOL		bNegFlag;
	FLMBYTE		ucNumBuf[ 12];
} BCD_TYPE;

typedef struct
{
	FLMUINT		uiFirstLFHBlkAddr;	// Address of first LFH block.
	FLMUINT		uiVersionNum;			// Database version		
	FLMUINT		uiBlockSize;			// Block size
	FLMUINT		uiDefaultLanguage;	// Default language
	FLMUINT		uiAppMajorVer;			// Application major version number
	FLMUINT		uiAppMinorVer;			// Application minor version number
	FLMUINT		uiSigBitsInBlkSize;	// Number of significant bits in block
												// size. 1K = 10, 2K = 11, 4K = 12 ...
} F_4x_FILE_HDR;

typedef struct
{
	FLMUINT  	uiCurrTransID;			// Current transaction ID.
	FLMUINT		uiFirstAvailBlkAddr;	// Address of first block in avail list
	FLMUINT		uiAvailBlkCount;		// Avail block count
	FLMUINT		uiLogicalEOF;			// Current logical end of file.  New
												// blocks are allocated at this address.
} F_4x_LOG_HDR;

typedef FLMINT (*	FLM_4x_TAG_COMPARE_FUNC)(
	FLM_4x_TAG_INFO *		pTagInfo1,
	FLM_4x_TAG_INFO *		pTagInfo2);

/****************************************************************************
Struct:	LFILE		(Logical File)
Desc:		This keeps track of the logical file information for an index or
			a container.
****************************************************************************/
typedef struct
{
	FLMUINT	   uiRootBlk;				// Address of root block.
	FLMUINT		uiNextDrn;				// Next DRN - only use when root is null
	FLMUINT		uiBlkAddress;			// Block address of LFile entry.
	FLMUINT		uiOffsetInBlk;			// Offset within block of entry.
	FLMUINT		uiLfNum;					// Index number or container number.
	FLMUINT		uiLfType; 				// Type of logical file
#define				FLM_4x_LF_CONTAINER		1
#define				FLM_4x_LF_INDEX			3
#define				FLM_4x_LF_INVALID			15
} F_4x_LFILE;

/**************************************************************************
Desc:		
**************************************************************************/
class F_Record : public XF_RefCount, public XF_Base
{
public:

	F_Record();
	~F_Record();

	FINLINE FLMUINT getID( void)
	{
		return( m_uiRecordID);
	}

	FINLINE void setID( 
		FLMUINT		uiRecordID)
	{
		m_uiRecordID = uiRecordID;
	}

	FINLINE FLMUINT getContainerID( void)
	{
		return( m_uiContainerID);
	}

	FINLINE void setContainerID( 
		FLMUINT		uiContainerID)
	{
		m_uiContainerID = uiContainerID;
	}

	F_Record * copy( void);

	void clear();

	RCODE getINT( 
		void *			pvField,
		FLMINT *			piNumber);

	RCODE getUINT( 
		void *			pvField,
		FLMUINT *		puiNumber);

	RCODE getUINT32(
		void *			pvField,
		FLMUINT32 *		pui32Number);

	RCODE getUnicode( 
		void *			pvField,
		FLMUNICODE *	pUnicode,
		FLMUINT *		puiBufLen);

	RCODE getNative( 
		void *			pvField,
		char *			pszString,
		FLMUINT *		puiBufLen);

	RCODE getBinary( 
		void *			pvField,
		void *			pvBuf,
		FLMUINT *		puiBufLen);

	FINLINE FLMUINT getLevel(
		void *		pvField)
	{
		return( ((FIELD *)pvField)->ui8Level);
	}

	FINLINE FLMUINT getFieldID( 
		void *		pvField)
	{
		return( ((FIELD *)pvField)->ui16FieldID);
	}

	FINLINE FLMUINT getDataType( 
		void *		pvField)
	{
		return( ((FIELD *)pvField)->ui8Type & 0x0F);
	}

	FINLINE FLMUINT getDataLength(
		void *		pvField)
	{
		return( ((FIELD *)pvField)->uiDataLength);
	}

	FINLINE FLMBYTE * getExportDataPtr( 
		void *		pvField)
	{
		return( getDataPtr( (FIELD *)pvField));
	}

	FINLINE FLMBOOL hasChild(
		void *		pvField)
	{
		return( (firstChildField( (FIELD *)pvField) != NULL) ? TRUE : FALSE);
	}

	FINLINE FLMBOOL isLast(
		void *		pvField)
	{
		return( (nextField( (FIELD *)pvField) == NULL) ? TRUE : FALSE);
	}

	FINLINE void * root( void)
	{
		return( m_pFirstFld);
	}

	FINLINE void * nextSibling( 
		void *		pvField)
	{
		return( pvField ? nextSiblingField( (FIELD *)pvField) : NULL);
	}

	FINLINE void * firstChild(
		void *		pvField)
	{
		return( pvField ? firstChildField( (FIELD *)pvField) : NULL);
	}

	FINLINE void * next( 
		void *		pvField)
	{
		return( pvField ? nextField( (FIELD * )pvField) : NULL);
	}

	FINLINE void * prev(
		void *		pvField)
	{
		return( pvField ? prevField( (FIELD *)pvField) : NULL);
	}

	void * prevSibling( 
		void *			pvField);

	void * lastChild( 
		void *			pvField);

	void * parent( 
		void *			pvField);

	void * find(
		void *		pvStartField,
		FLMUINT		uiFieldID,
		FLMUINT		uiOccur = 1,
		FLMBOOL		bSearchForest = FALSE);

	void * find( 
		void *		pvStartField,
		FLMUINT *	puiFieldPath,
		FLMUINT		uiOccur = 1, 
		FLMBOOL		bSearchForest = FALSE);

	RCODE preallocSpace(
		FLMUINT		uiDataSize);

	FLMBYTE * getImportDataPtr( 
		void *		pvField,
		FLMUINT		uiDataType, 
		FLMUINT		uiLength);

	RCODE insertLast(
		FLMUINT		uiLevel,
		FLMUINT		uiFieldID,
		FLMUINT		uiDataType,
		void **		ppvField);

private:

	void resetFieldList( void);

	FINLINE FIELD * getFirstField( void)
	{
		return( m_pFirstFld);
	}

	FINLINE FIELD * prevField(
		FIELD *		pField)
	{
		return( (FIELD *)(pField ? pField->pPrev : NULL));
	}

	FINLINE FIELD * nextField(
		FIELD *		pField)
	{
		return( (FIELD *)(pField ? pField->pNext : NULL));
	}

	FIELD * nextSiblingField(
		FIELD *		pField);

	FINLINE FIELD * firstChildField(
		FIELD *		pField)
	{
		FLMUINT8 ui8Level = pField->ui8Level;

		return( ((pField = nextField( pField)) != NULL && 
						pField->ui8Level > ui8Level)
					? pField
					: NULL);
	}

	FIELD * lastSubTreeField( 
		FIELD *		pField);

	RCODE createField( 
		FIELD *		pCurField,
		FIELD **		ppNewField);

	FLMBYTE * getDataPtr( 
		FIELD *		pField);

	RCODE	getNewDataPtr(
		FIELD *		pField, 
		FLMUINT		uiDataType,
		FLMUINT		uiNewLength, 
		FLMBYTE **	ppDataPtr);

	RCODE storage2INT(
		FLMUINT		uiType,
		FLMUINT		uiBufLength,
		FLMBYTE *	pBuf,
		FLMINT *		piNum);

	RCODE storage2UINT(
		FLMUINT		uiType,
		FLMUINT		uiBufLength,
		FLMBYTE *	pBuf,
		FLMUINT *	puiNum);

	RCODE storage2UINT32(
		FLMUINT		uiType,
		FLMUINT		uiBufLength,
		FLMBYTE *	pBuf,
		FLMUINT32 *	pui32Num);

	RCODE bcd2Num(
		FLMUINT		uiType,
		FLMUINT		uiBufLength,
		FLMBYTE *	pBuf,
		BCD_TYPE  *	bcd);

	RCODE getUnicode(
		FLMUINT			uiType,
		FLMUINT			uiBufLength,
		FLMBYTE *		pBuffer,
		FLMUINT *		puiStrBufLen,
		FLMUNICODE *	puzStrBuf);

	RCODE storage2Native(
		FLMUINT			uiType,
		FLMUINT			uiBufLength,
		FLMBYTE *		pBuffer, 
		FLMUINT *		puiOutBufLenRV,
		char *			pszOutBuffer);

	RCODE numToText(
		FLMBYTE *		pucNum,
		FLMBYTE *		pucOutBuffer,
		FLMUINT *		puiBufLen);

	RCODE contextToText(
		FLMBYTE *		pucValue,
		FLMBYTE *		pucOutBuffer,
		FLMUINT *		puiBufLen);

	FLMUINT textObjType(
		FLMBYTE			ucObj)
	{
		if( (ucObj & FLM_4x_ASCII_CHAR_MASK) == 
			FLM_4x_ASCII_CHAR_CODE)
		{
			return( FLM_4x_ASCII_CHAR_CODE);
		}
		else if( (ucObj & FLM_4x_WHITE_SPACE_MASK) == 
			FLM_4x_WHITE_SPACE_CODE)
		{
			return( FLM_4x_WHITE_SPACE_CODE);
		}
		else if( (ucObj & FLM_4x_UNK_EQ_1_MASK) == 
			FLM_4x_UNK_EQ_1_CODE)
		{
			return( FLM_4x_UNK_EQ_1_CODE);
		}
		else if( (ucObj & FLM_4x_CHAR_SET_MASK) == 
			FLM_4x_CHAR_SET_CODE)
		{
			return( FLM_4x_CHAR_SET_CODE);
		}

		return( ucObj);
	}

	FLMUINT		m_uiContainerID;
	FLMUINT		m_uiRecordID;
	F_Pool		m_pool;
#define FLM_4x_FIELD_LIST_SIZE		100
	FIELD			m_fieldList[ FLM_4x_FIELD_LIST_SIZE];
	FIELD *		m_pFirstFld;
	FIELD *		m_pLastFld;
	FIELD *		m_pAvailFld;
	FLMBYTE *	m_pDataBuf;
	FLMUINT		m_uiDataBufOffset;
	FLMUINT		m_uiDataBufLength;

friend class F_4xReader;
};

/**************************************************************************
Desc:		
**************************************************************************/
class F_4xReader : public XF_RefCount, public XF_Base
{
public:

	F_4xReader();

	~F_4xReader();

	RCODE openDatabase(
		char *				pszPath);

	void closeDatabase( void);

	RCODE retrieveRec(
		FLMUINT				uiContainer,
		FLMUINT				uiDrn,
		FLMUINT				uiFlags,
		F_Record **			ppRecord);

	RCODE retrieveNextRec(
		F_Record **			ppRecord);

	RCODE getNextDrn(
		FLMUINT				uiContainer,
		FLMUINT *			puiDrn);

	RCODE getNameTable(
		F_4xNameTable **	ppNameTable);

	FINLINE void setDefaultContainer(
		FLMUINT				uiContainer)
	{
		m_uiDefaultContainer = uiContainer;
	}

private:

	FINLINE void initStack(
		BTSK *		pStack)
	{
		FLMUINT	uiNumLevels = FLM_4x_BH_MAX_LEVELS;

		while( uiNumLevels--)
		{
			pStack->pBlk = NULL;
			pStack->uiBlkAddr = 0;
			pStack++;
		}
	}

	void releaseStack(
		BTSK *		pStack);

	RCODE readRecElements(
		BTSK *			pStack,
		F_4x_LFILE *	pLFile,
		F_Record **		ppRecord);

	RCODE getFldOverhead(
		FSTATE *			pState);

	RCODE btSearch(
		F_4x_LFILE *	pLFile,
		FLMUINT			uiDrn,
		BTSK **			ppStack);

	RCODE blkNextElm(
		BTSK *			pStack);

	RCODE btNextElm(
		BTSK *			pStack,
		F_4x_LFILE *	pLFile);

	RCODE btPrevElm(
		BTSK *			pStack,
		F_4x_LFILE *	pLFile);

	RCODE btScan(
		BTSK *			pStack,
		FLMBYTE *		pucSearchKey);

	RCODE btScanNonLeafData(
		BTSK *			pStack,
		FLMUINT			uiDrn);

	RCODE btSearchEnd(
		F_4x_LFILE *	pLFile,
		FLMUINT			uiDrn,
		BTSK **			ppStack);

	RCODE btAdjustStack(
		BTSK *			pStack,
		F_4x_LFILE *	pLFile,
		FLMBOOL			bMovedNext);

	FLMUINT childBlkAddr(
		BTSK *			pStack);

	FLMUINT lgHdrCheckSum(
		FLMBYTE *		pucLogHdr,
		FLMBOOL			bCompare);

	RCODE createLckFile(
		char *		pszFilePath);

	RCODE readBlock(
		FLMUINT			uiBlkAddr,
		F_Block **		ppBlock);

	FINLINE FLMUINT getEncryptSize(
		FLMBYTE *		pucBlk)
	{
		FLMUINT	uiLen = (FLMUINT)FB2UW( &pucBlk[ FLM_4x_BH_ELM_END]);

		if (uiLen % sizeof( FLMUINT32) != 0)
		{
			uiLen += (FLMUINT)(sizeof( FLMUINT32) - 
				(uiLen % sizeof( FLMUINT32)));
		}
		return( uiLen);
	}

	RCODE blkCheckSum(
		FLMBYTE *		pucBlkPtr,
		FLMUINT			uiBlkAddress,
		FLMUINT			uiBlkSize);

	RCODE readLFiles( void);

	RCODE getLFile(
		FLMUINT			uiLFile,
		F_4x_LFILE **	ppLFile);

	RCODE getRootBlock(
		F_4x_LFILE *	pLFile,
		BTSK *			pStack);

	RCODE getBlock(
		F_4x_LFILE *	pLFile,
		FLMUINT			uiBlkAddr,
		BTSK *			pStack);

	void blkToStack(
		F_Block **		ppBlock,
		BTSK *			pStack);

	RCODE getFieldType(
		FLMUINT			uiFieldNum,
		FLMUINT *		puiType);

	RCODE readDictionary( void);

	RCODE getTypeTag(
		F_Record *		pRec,
		void *			pvField,
		FLMUINT *		puiType);

	FINLINE F_Block ** getHashBucket(
		FLMUINT			uiBlkAddress)
	{
		return( &m_ppBlockTbl[
			(((uiBlkAddress) >> m_fileHdr.uiSigBitsInBlkSize) &
			(m_uiBlockTblSize - 1))]);
	}

	// Data

	F_Pool				m_tmpPool;
	F_SuperFileHdl *	m_pSuperHdl;
	F_4x_FILE_HDR		m_fileHdr;
	F_4x_LOG_HDR		m_logHdr;
	F_4x_LFILE *		m_pLFileTbl;
	FLMUINT				m_uiLFileCnt;
	FLMUINT *			m_puiFieldTbl;
	FLMUINT				m_uiFieldTblSize;
	FLMUINT				m_uiDefaultContainer;
	F_FileHdl *			m_pLckFile;
	FLMUINT				m_uiMaxFileSize;
	F_Block **			m_ppBlockTbl;
	FLMUINT				m_uiBlockTblSize;
	F_4xNameTable *	m_pNameTable;

friend class F_4xNameTable;
};

/**************************************************************************
Desc:		
**************************************************************************/
class F_Block : public XF_RefCount, public XF_Base
{
public:

	F_Block()
	{
		m_pucBlk = NULL;
		m_uiBlockSize = 0;
	}

	~F_Block()
	{
		if( m_pucBlk)
		{
			f_free( &m_pucBlk);
		}
	}

	FINLINE RCODE allocBlockBuf(
		FLMUINT			uiBlockSize)
	{
		RCODE			rc = NE_XFLM_OK;

		flmAssert( getRefCount() == 1);

		if( !m_uiBlockSize)
		{
			if( RC_BAD( rc = f_alloc( uiBlockSize, &m_pucBlk)))
			{
				goto Exit;
			}
		}
		else if( m_uiBlockSize != uiBlockSize)
		{
			if( RC_BAD( rc = f_realloc( uiBlockSize, &m_pucBlk)))
			{
				goto Exit;
			}
		}

		m_uiBlockSize = uiBlockSize;

	Exit:

		return( rc);
	}

private:

	FLMUINT			m_uiBlockSize;
	FLMBYTE *		m_pucBlk;

friend class F_4xReader;
};

/**************************************************************************
Desc:		
**************************************************************************/
class F_4xNameTable : public XF_RefCount, public XF_Base
{
public:

	F_4xNameTable();

	~F_4xNameTable();

	void clearTable( void);

	FLMBOOL getNextTagNumOrder(
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName,
		char *					pszTagName,
		FLMUINT					uiNameBufSize,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiType = NULL,
		FLMUINT *				puiSubType = NULL);

	FLMBOOL getNextTagNameOrder(
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName,
		char *					pszTagName,
		FLMUINT					uiNameBufSize,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiType = NULL,
		FLMUINT *				puiSubType = NULL);

	FLMBOOL getFromTagType(
		FLMUINT					uiType,
		FLMUINT *				puiNextPos,
		FLMUNICODE *			puzTagName,
		char *					pszTagName,
		FLMUINT					uiNameBufSize,
		FLMUINT *				puiTagNum = NULL,
		FLMUINT *				puiSubType = NULL);

	FLMBOOL getFromTagNum(
		FLMUINT					uiTagNum,
		FLMUNICODE *			puzTagName,
		char *					pszTagName,
		FLMUINT					uiNameBufSize,
		FLMUINT *				puiType = NULL,
		FLMUINT *				puiSubType = NULL);

	FLMBOOL getFromTagName(
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMUINT *				puiTagNum,
		FLMUINT *				puiType = NULL,
		FLMUINT *				puiSubType = NULL);

	FLMBOOL getFromTagTypeAndName(
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMUINT					uiType,
		FLMUINT *				puiTagNum,
		FLMUINT *				puiSubType = NULL);

	RCODE addTag(
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMUINT					uiTagNum,
		FLMUINT					uiType,
		FLMUINT					uiSubType,
		FLMBOOL					bCheckDuplicates = TRUE);

	RCODE setupNameTable(
		F_4xReader *	pDb);

	void sortTags( void);

private:

	RCODE allocTag(
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMUINT					uiTagNum,
		FLMUINT					uiType,
		FLMUINT					uiSubType,
		FLM_4x_TAG_INFO **	ppTagInfo);

	RCODE reallocSortTables(
		FLMUINT					uiNewTblSize);

	void copyTagName(
		FLMUNICODE *			puzDestTagName,
		char *					pszDestTagName,
		FLMUINT					uiDestBufSize,
		FLMUNICODE *			puzSrcTagName);

	FLM_4x_TAG_INFO * findTagByName(
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMUINT *				puiInsertPos = NULL);

	FLM_4x_TAG_INFO * findTagByNum(
		FLMUINT					uiTagNum,
		FLMUINT *				puiInsertPos = NULL);

	FLM_4x_TAG_INFO * findTagByTypeAndName(
		const FLMUNICODE *	puzTagName,
		const char *			pszTagName,
		FLMUINT					uiType,
		FLMUINT *				puiInsertPos = NULL);

	RCODE insertTagInTables(
		FLM_4x_TAG_INFO *		pTagInfo,
		FLMUINT					uiTagNameTblInsertPos,
		FLMUINT					uiTagTypeAndNameTblInsertPos,
		FLMUINT					uiTagNumTblInsertPos);

	FINLINE void tagInfoSwap(
		FLM_4x_TAG_INFO **	ppTagInfoTbl,
		FLMUINT					uiPos1,
		FLMUINT					uiPos2)
	{
		FLM_4x_TAG_INFO *		pTmpTagInfo = ppTagInfoTbl [uiPos1];

		ppTagInfoTbl [uiPos1] = ppTagInfoTbl [uiPos2];
		ppTagInfoTbl [uiPos2] = pTmpTagInfo;
	}

	static FLMINT tagNameCompare(
		const FLMUNICODE *	puzName1,
		const char *			pszName1,
		const FLMUNICODE *	puzName2);

	static FINLINE FLMINT compareTagNumOnly(
		FLM_4x_TAG_INFO *		pTagInfo1,
		FLM_4x_TAG_INFO *		pTagInfo2)
	{
		if (pTagInfo1->uiTagNum < pTagInfo2->uiTagNum)
		{
			return( -1);
		}
		else if (pTagInfo1->uiTagNum > pTagInfo2->uiTagNum)
		{
			return( 1);
		}
		else
		{
			return( 0);
		}
	}

	static FINLINE FLMINT compareTagNameOnly(
		FLM_4x_TAG_INFO *		pTagInfo1,
		FLM_4x_TAG_INFO *		pTagInfo2)
	{
		return (tagNameCompare( pTagInfo1->puzTagName, NULL,
					pTagInfo2->puzTagName));
	}

	static FINLINE FLMINT compareTagTypeAndName(
		FLM_4x_TAG_INFO *		pTagInfo1,
		FLM_4x_TAG_INFO *		pTagInfo2)
	{
		if (pTagInfo1->uiType < pTagInfo2->uiType)
		{
			return( -1);
		}
		else if (pTagInfo1->uiType > pTagInfo2->uiType)
		{
			return( 1);
		}
		else
		{
			return (tagNameCompare( pTagInfo1->puzTagName, NULL,
							pTagInfo2->puzTagName));
		}
	}

	void sortTagTbl(
		FLM_4x_TAG_INFO **			ppTagInfoTbl,
		FLMUINT							uiLowerBounds,
		FLMUINT							uiUpperBounds,
		FLM_4x_TAG_COMPARE_FUNC		fnTagCompare);

	F_Pool					m_pool;
	FLM_4x_TAG_INFO **	m_ppSortedByTagName;
	FLM_4x_TAG_INFO **	m_ppSortedByTagNum;
	FLM_4x_TAG_INFO **	m_ppSortedByTagTypeAndName;
	FLMUINT					m_uiTblSize;
	FLMUINT					m_uiNumTags;
	FLMBOOL					m_bTablesSorted;
};

#endif
