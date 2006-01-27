//-------------------------------------------------------------------------
// Desc:	RFL viewer utility - definitions.
// Tabs:	3
//
//		Copyright (c) 1998-2006 Novell, Inc. All Rights Reserved.
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
// $Id: rflread.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaim.h"
#include "flaimsys.h"

#ifndef RFLREAD_HPP
#define RFLREAD_HPP

extern "C"
{

#ifdef MAIN_MODULE
	#define REXTERN
#else
	#define REXTERN	extern
#endif

#ifndef RFL_BUFFER_SIZE
#define	RFL_BUFFER_SIZE						(65536 * 4)
#endif

REXTERN	F_FileHdl *	gv_pRflFileHdl;
REXTERN	FLMBYTE		gv_rflBuffer [RFL_BUFFER_SIZE];
REXTERN	FLMUINT		gv_uiRflEof;

// Tag numbers for internal fields.

#define RFL_TRNS_BEGIN_FIELD					32769
#define RFL_TRNS_COMMIT_FIELD					32770
#define RFL_TRNS_ABORT_FIELD					32771
#define RFL_RECORD_ADD_FIELD					32772
#define RFL_RECORD_MODIFY_FIELD				32773
#define RFL_RECORD_DELETE_FIELD				32774
#define RFL_RESERVE_DRN_FIELD					32775
#define RFL_CHANGE_FIELDS_FIELD				32776
#define RFL_DATA_RECORD_FIELD					32777
#define RFL_UNKNOWN_PACKET_FIELD				32778
#define RFL_NUM_BYTES_VALID_FIELD			32779
#define RFL_PACKET_ADDRESS_FIELD				32780
#define RFL_PACKET_CHECKSUM_FIELD			32781
#define RFL_PACKET_CHECKSUM_VALID_FIELD	32782
#define RFL_PACKET_BODY_LENGTH_FIELD		32783
#define RFL_NEXT_PACKET_ADDRESS_FIELD		32784
#define RFL_PREV_PACKET_ADDRESS_FIELD		32785
#define RFL_TRANS_ID_FIELD						32786
#define RFL_START_SECONDS_FIELD				32787
#define RFL_START_MSEC_FIELD					32788
#define RFL_END_SECONDS_FIELD					32789
#define RFL_END_MSEC_FIELD						32790
#define RFL_START_TRNS_ADDR_FIELD			32791
#define RFL_CONTAINER_FIELD					32792
#define RFL_DRN_FIELD							32793
#define RFL_TAG_NUM_FIELD						32794
#define RFL_TYPE_FIELD							32795
#define RFL_LEVEL_FIELD							32796
#define RFL_DATA_LEN_FIELD						32797
#define RFL_DATA_FIELD							32798
#define RFL_MORE_DATA_FIELD					32799
#define RFL_INSERT_FLD_FIELD					32800
#define RFL_MODIFY_FLD_FIELD					32801
#define RFL_DELETE_FLD_FIELD					32802
#define RFL_END_CHANGES_FIELD					32803
#define RFL_UNKNOWN_CHANGE_TYPE_FIELD		32804
#define RFL_POSITION_FIELD						32805
#define RFL_REPLACE_BYTES_FIELD				32806
#define RFL_UNKNOWN_CHANGE_BYTES_FIELD		32807
#define RFL_INDEX_SET_FIELD					32808
#define RFL_INDEX_NUM_FIELD					32809
#define RFL_START_DRN_FIELD					32810
#define RFL_END_DRN_FIELD						32811
#define RFL_START_UNKNOWN_FIELD				32812
#define RFL_UNKNOWN_USER_PACKET_FIELD		32813
#define RFL_HDR_NAME_FIELD						32814
#define RFL_HDR_VERSION_FIELD					32815
#define RFL_HDR_FILE_NUMBER_FIELD			32816
#define RFL_HDR_EOF_FIELD						32817
#define RFL_HDR_DB_SERIAL_NUM_FIELD			32818
#define RFL_HDR_FILE_SERIAL_NUM_FIELD		32819
#define RFL_HDR_NEXT_FILE_SERIAL_NUM_FIELD	32820
#define RFL_HDR_KEEP_SIGNATURE_FIELD		32821
#define RFL_TRNS_BEGIN_EX_FIELD				32822
#define RFL_UPGRADE_PACKET_FIELD				32823
#define RFL_OLD_DB_VERSION_FIELD				32824
#define RFL_NEW_DB_VERSION_FIELD				32825
#define RFL_REDUCE_PACKET_FIELD				32826
#define RFL_BLOCK_COUNT_FIELD					32827
#define RFL_LAST_COMMITTED_TRANS_ID_FIELD	32828
#define RFL_INDEX_SET2_FIELD					32829
#define RFL_INDEX_SUSPEND_FIELD				32830
#define RFL_INDEX_RESUME_FIELD				32831
#define RFL_BLK_CHAIN_FREE_FIELD				32832
#define RFL_TRACKER_REC_FIELD					32833
#define RFL_END_BLK_ADDR_FIELD				32834
#define RFL_FLAGS_FIELD							32835
#define RFL_INSERT_ENC_FLD_FIELD				32836
#define RFL_MODIFY_ENC_FLD_FIELD				32837
#define RFL_ENC_FIELD							32838
#define RFL_ENC_DEF_ID_FIELD					32839
#define RFL_ENC_DATA_LEN_FIELD				32840
#define RFL_DB_KEY_LEN_FIELD					32841
#define RFL_DB_KEY_FIELD						32842
#define RFL_WRAP_KEY_FIELD						32843
#define RFL_ENABLE_ENCRYPTION_FIELD			32844

typedef struct Rfl_Packet
{
	FLMUINT	uiFileOffset;					// File offset this packet was read from.
	FLMUINT	uiPacketAddress;				// Packet address that was read
	FLMUINT	uiPacketAddressBytes;		// Bytes that were actually in packet addr.
	FLMUINT	uiPacketChecksum;				// Packet checksum
	FLMBOOL	bHavePacketChecksum;			// Did we actually have a packet checksum?
	FLMBOOL	bValidChecksum;				// Is the checksum valid?
	FLMUINT	uiPacketType;					// Packet type
	FLMBOOL	bHavePacketType;				// Did we actually have a packet type?
	FLMBOOL	bValidPacketType;				// Is the packet type valid?
	FLMBOOL	bHaveTimes;						// Was the time bit set on the packet type?
	FLMUINT	uiPacketBodyLength;			// Packet body length
	FLMUINT	uiPacketBodyLengthBytes;	// Bytes that were in packet body length
	FLMUINT	uiNextPacketAddress;			// Next packet address - zero if no more
	FLMUINT	uiPrevPacketAddress;			// Prev packet address - zero if unknown
	FLMUINT	uiTransID;						// Transaction ID
	FLMUINT	uiTransIDBytes;				// Bytes that were actually in transID
	FLMUINT	uiTransStartAddr;				// Transaction start address
	FLMUINT	uiTransStartAddrBytes;		// Transaction start address bytes
	FLMUINT	uiContainer;					// container
	FLMUINT	uiContainerBytes;				// Bytes that were in container.
	FLMUINT	uiIndex;							// index
	FLMUINT	uiIndexBytes;					// Bytes that were in index.
	FLMUINT	uiDrn;							// DRN
	FLMUINT	uiDrnBytes;						// Bytes that were in DRN.
	FLMUINT	uiEndDrn;						// End DRN
	FLMUINT	uiEndDrnBytes;					// Bytes that were in End DRN.
	FLMUINT	uiStartSeconds;				// Start seconds
	FLMUINT	uiStartSecondsBytes;			// Bytes that were in start seconds
	FLMUINT	uiStartMicro;					// Start micro seconds
	FLMUINT	uiStartMicroBytes;			// Bytes that were in start micro seconds
	FLMUINT	uiEndSeconds;					// End seconds
	FLMUINT	uiEndSecondsBytes;			// Bytes that were in end seconds
	FLMUINT	uiEndMicro;						// End micro seconds
	FLMUINT	uiEndMicroBytes;				// Bytes that were in end micro seconds
	FLMUINT	uiLastCommittedTransID;		// Last committed transaction ID
	FLMUINT	uiLastCommittedTransIDBytes; // Bytes that were in the last committed trans ID
	FLMUINT	uiFlags;							// Operation flags
	FLMUINT	uiFlagsBytes;					// Bytes that were in flags
	FLMUINT	uiCount;							// Count (number of blocks, etc.)
	FLMUINT	uiCountBytes;					// Bytes that were in count
	FLMUINT	uiMultiFileSearch;			// Is search to span multiple files?
} RFL_PACKET, * RFL_PACKET_p;

RCODE RflPositionToNode(
	FLMUINT		uiFileOffset,
	FLMBOOL		bOperationsOnly,
	POOL *		pPool,
	NODE **		ppNodeRV);

RCODE RflGetNextNode(
	NODE *		pCurrOpNode,
	FLMBOOL		bOperationsOnly,
	POOL *		pPool,
	NODE **		ppNextNodeRV,
	FLMBOOL		bStopAtEOF = FALSE);

RCODE RflGetPrevNode(
	NODE *		pCurrOpNode,
	FLMBOOL		bOperationsOnly,
	POOL *		pPool,
	NODE **		ppPrevNodeRV);

void RflFormatPacket(
	void *			pPacket,
	char *			pszDispBuffer);

RCODE RflExpandPacket(
	NODE *		pPacketNode,
	POOL *		pPool,
	NODE **		ppForest);

}	// extern "C"

#endif
