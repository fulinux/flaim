//------------------------------------------------------------------------------
// Desc:	Various structures and classes used internally by FLAIM.
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
// $Id: fstructs.h 3109 2006-01-19 13:07:07 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FSTRUCTS_H
#define FSTRUCTS_H

#if defined( FLM_WIN) || defined( FLM_NLM) || defined( FLM_LINUX)
	#pragma pack( push, 1)
#else
	#pragma pack( 1)
#endif

class HRequest;
class F_BtPool;
class F_XMLImport;
class F_XMLExport;
class F_FileIdList;
class F_Rfl;
class F_SuperFileHdl;
class F_Btree;
class F_DbRebuild;
class F_DbCheck;
class F_XML;
class ServerLockManager;
class ServerLockObject;
class F_FileHdlMgr;
class F_ThreadMgr;
class F_Cursor;
class F_MultiAlloc;
class F_CachedNode;
class F_CachedBlock;
class F_BlockCacheMgr;
class F_NodeCacheMgr;
class F_GlobalCacheMgr;
class F_NodePool;
class F_BTreeInfo;
class F_NodeRelocator;
class F_NodeDataRelocator;
class F_NodeListRelocator;	
class F_AttrItemRelocator;	
class F_AttrBufferRelocator;	

#if defined( FLM_NLM)

	extern "C"
	{
		void ConvertTicksToSeconds(
			LONG		ticks,
			LONG *	seconds,
			LONG *	tenthsOfSeconds);
	
		void ConvertSecondsToTicks(
			LONG		seconds,
			LONG		tenthsOfSeconds,
			LONG *	ticks);
	}

	#define FLM_GET_TIMER()	(FLMUINT)GetCurrentTime()

	#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU)	\
		ConvertSecondsToTicks( (LONG)(uiSeconds), 0, (LONG *)(&(uiTU)))

	#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds)	\
	{ \
		LONG	udDummy; \
		ConvertTicksToSeconds( (LONG)(uiTU), (LONG *)(&(uiSeconds)), &udDummy); \
	}

	#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli)	\
	{ \
		LONG	udTenths; \
		LONG	udSeconds; \
		ConvertTicksToSeconds( (LONG)(uiTU), (LONG *)(&(udSeconds)), &udTenths); \
		uiMilli = (FLMUINT)(udSeconds) * 1000 + (FLMUINT)udTenths * 100; \
	}
	#define FLM_MILLI_TO_TIMER_UNITS( uiMilliSeconds, uiTU)	\
	{ \
		LONG udTenths, udSeconds; \
		udSeconds = ((LONG) uiMilliSeconds) / 1000; \
		udTenths = (((LONG) uiMilliSeconds) % 1000) / 100; \
		ConvertSecondsToTicks( udSeconds, udTenths, (LONG *)(&(uiTU))); \
	}

#elif defined( FLM_UNIX)

	// gettimeofday() is actually 4 times faster than time() on
	// Solaris. gethrtime() is even faster. On Linux time() is the
	// fastest; gettimeofday() is 50% slower. clock() is the
	// slowest on both Solaris and Linux. We use a new function for
	// millisec resolution. The implementation is OS dependent.

	#define FLM_GET_TIMER() (FLMUINT) f_timeGetMilliTime()
	#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU)  \
       ((uiTU) = ((uiSeconds) * 1000))
	#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds)  \
       ((uiSeconds) = ((uiTU) / 1000))
	#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli)   \
		 ((uiMilli) = (uiTU))
	#define FLM_MILLI_TO_TIMER_UNITS( uiMilli, uiTU)	\
		 ((uiTU) = (uiMilli))
#else /* FLM_WIN */

	#define FLM_GET_TIMER() \
		(FLMUINT)GetTickCount()

	#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU) \
		((uiTU) = (uiSeconds) * 1000)

	#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds) \
		((uiSeconds) = (uiTU) / 1000)

	#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli) \
		(uiMilli = (uiTU))

	#define FLM_MILLI_TO_TIMER_UNITS( uiMilliSeconds, uiTU) \
		(uiTU = (uiMilliSeconds))

#endif

// This macro for calculating elapsed time accounts for the
// possibility of the time wrapping - which it will for some
// of our counters (FLM_WIN is milliseconds and wraps in 49.7 days).

#define FLM_ELAPSED_TIME(uiLaterTime,uiEarlierTime) \
	(FLMUINT)(((uiLaterTime) >= (uiEarlierTime)) \
				 ? (FLMUINT)((uiLaterTime) - (uiEarlierTime)) \
				 : (FLMUINT)((0xFFFFFFFF - (uiEarlierTime)) + (uiLaterTime)))

/****************************************************************************
Desc:	Tests to see if database is NOT in native platform format.
****************************************************************************/
FINLINE FLMBOOL hdrIsNonNativeFormat(
	XFLM_DB_HDR *	pDbHdr)
{
	return( (FLMBOOL)(pDbHdr->ui8IsLittleEndian ==
							XFLM_NATIVE_IS_LITTLE_ENDIAN
							? (FLMBOOL)FALSE
							: (FLMBOOL)TRUE));
}

/****************************************************************************
Desc:	 	Block header - on-disk format.
****************************************************************************/
typedef struct FlmBlockHdrTag
{
	FLMUINT32	ui32BlkAddr;						// BH_ADDR
	FLMUINT32	ui32PrevBlkInChain;				// BH_PREV_BLK
	FLMUINT32	ui32NextBlkInChain;				// BH_NEXT_BLK
	FLMUINT32	ui32PriorBlkImgAddr;				// BH_PREV_BLK_ADDR
	FLMUINT64	ui64TransID;						// BH_TRANS_ID
	FLMUINT32	ui32BlkCRC;							// Block CRC
	FLMUINT16	ui16BlkBytesAvail;				// BH_BLK_END, BH_ELM_END
	FLMUINT8		ui8BlkFlags;						// Flags for the block
		#define BLK_FORMAT_IS_LITTLE_ENDIAN	0x01
		#define BLK_IS_BEFORE_IMAGE			0x02
											// This bit gets ORed into type if the
											// block is a Before Image block that
											// should be restored on transaction
											// abort.  This is only set when a block
											// is written to the log, so it only
											// needs to be unset when the block is
											// read back from the log.
		#define BLK_IS_ENCRYPTED	0x04


	FLMUINT8		ui8BlkType;							// BH_TYPE
		#define BT_FREE 					0		// Free block - avail list
		#define BT_LFH_BLK				1		// LFH Header block
		#define BT_LEAF					2		// New B-Tree Leaf block
		#define BT_NON_LEAF				3		// New B-Tree Non-leaf block block - fixed key size
		#define BT_NON_LEAF_COUNTS		4		// New B-Tree Non-leaf index with counts
		#define BT_LEAF_DATA				5		// New B-Tree Leaf block with Data
		#define BT_DATA_ONLY				6		// Data-only block
	// NOTE: IF adding more types, may need to modify the blkIsNewBTree function
	// below.

	// IMPORTANT NOTE: If anything is changed in here, need to make
	// corresponding changes to convertBlkHdr routine and
	// flmVerifyDiskStructOffsets routine.

#define F_BLK_HDR_ui32BlkAddr_OFFSET				0
#define F_BLK_HDR_ui32PrevBlkInChain_OFFSET		4
#define F_BLK_HDR_ui32NextBlkInChain_OFFSET		8
#define F_BLK_HDR_ui32PriorBlkImgAddr_OFFSET		12
#define F_BLK_HDR_ui64TransID_OFFSET				16
#define F_BLK_HDR_ui32BlkCRC_OFFSET					24
#define F_BLK_HDR_ui16BlkBytesAvail_OFFSET		28
#define F_BLK_HDR_ui8BlkFlags_OFFSET				30
#define F_BLK_HDR_ui8BlkType_OFFSET					31
} F_BLK_HDR;

#define SIZEOF_STD_BLK_HDR	sizeof( F_BLK_HDR)

/****************************************************************************
Desc:	Test to see if block is in non-native format.
****************************************************************************/
FINLINE FLMBOOL blkIsNonNativeFormat(
	F_BLK_HDR *	pBlkHdr)
{
#if XFLM_NATIVE_IS_LITTLE_ENDIAN
	return( (pBlkHdr->ui8BlkFlags & BLK_FORMAT_IS_LITTLE_ENDIAN)
				? FALSE
				: TRUE);
#else
	return( (pBlkHdr->ui8BlkFlags & BLK_FORMAT_IS_LITTLE_ENDIAN)
				? TRUE
				: FALSE);
#endif
}

/****************************************************************************
Desc:	Set block flags to indicate it is in native format.
****************************************************************************/
FINLINE void blkSetNativeFormat(
	F_BLK_HDR *	pBlkHdr)
{
#if XFLM_NATIVE_IS_LITTLE_ENDIAN
	pBlkHdr->ui8BlkFlags |= BLK_FORMAT_IS_LITTLE_ENDIAN;
#else
	pBlkHdr->ui8BlkFlags &= ~(BLK_FORMAT_IS_LITTLE_ENDIAN);
#endif
}

/****************************************************************************
Desc:	Test to see if a block type is a B-Tree block type.
****************************************************************************/
FINLINE FLMBOOL blkIsBTree(
	F_BLK_HDR *	pBlkHdr
	)
{
	return( (pBlkHdr->ui8BlkType != BT_FREE &&
				pBlkHdr->ui8BlkType != BT_LFH_BLK)
			  ? TRUE
			  : FALSE);
}

/****************************************************************************
Desc:	Test to see if a block type is a NEW B-Tree block type.
****************************************************************************/
FINLINE FLMBOOL blkIsNewBTree(
	F_BLK_HDR *	pBlkHdr
	)
{
	return( (pBlkHdr->ui8BlkType >= BT_LEAF)
				? TRUE
				: FALSE);
}

/****************************************************************************
Desc:	Determine where the block ends.
****************************************************************************/
FINLINE FLMUINT blkGetEnd(
	FLMUINT		uiBlockSize,
	FLMUINT		uiBlkHdrSize,
	F_BLK_HDR *	pBlkHdr
	)
{
	return( (FLMUINT)(blkIsNewBTree( pBlkHdr)
							? uiBlockSize
							: (FLMUINT)(pBlkHdr->ui16BlkBytesAvail >
											uiBlockSize - uiBlkHdrSize
											? uiBlkHdrSize
											: uiBlockSize -
												(FLMUINT)pBlkHdr->ui16BlkBytesAvail)));
}

FINLINE void setBlockEncrypted(
	F_BLK_HDR *	pBlkHdr
	)
{
	pBlkHdr->ui8BlkFlags |= BLK_IS_ENCRYPTED;
}

FINLINE void unsetBlockEncrypted(
	F_BLK_HDR *	pBlkHdr
	)
{
	pBlkHdr->ui8BlkFlags &= (~(BLK_IS_ENCRYPTED));
}

FINLINE FLMBOOL isEncryptedBlk(
	F_BLK_HDR *		pBlkHdr
	)
{
	return( (pBlkHdr->ui8BlkFlags & BLK_IS_ENCRYPTED) ? TRUE : FALSE);
}

/****************************************************************************
Desc:	 	B-Tree block header - on-disk format.
****************************************************************************/
typedef struct FlmBTreeBlkHdr
{
	F_BLK_HDR	stdBlkHdr;							// Standard block header
	FLMUINT16	ui16LogicalFile;					// BH_LOG_FILE_NUM
	FLMUINT16	ui16NumKeys;						// Number of keys
	FLMUINT8		ui8BlkLevel;						// BH_LEVEL
		#define BH_MAX_LEVELS		8 				// Max allowable b-tree levels
		#define MAX_LEVELS			BH_MAX_LEVELS
	FLMUINT8		ui8BTreeFlags;						// Flags for BTree
		#define BLK_IS_ROOT			0x01
		#define BLK_IS_INDEX			0x02
	FLMUINT16	ui16HeapSize;						// Contiguous available space
#define F_BTREE_BLK_HDR_stdBlkHdr_OFFSET					0
#define F_BTREE_BLK_HDR_ui16LogicalFile_OFFSET			32
#define F_BTREE_BLK_HDR_ui16NumKeys_OFFSET				34
#define F_BTREE_BLK_HDR_ui8BlkLevel_OFFSET				36
#define F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET				37
#define F_BTREE_BLK_HDR_ui16HeapSize_OFFSET				38
} F_BTREE_BLK_HDR;

/****************************************************************************
Desc:	 	Encrypted B-Tree block header - on-disk format.
****************************************************************************/
typedef struct
{
	F_BTREE_BLK_HDR		btree;
	FLMUINT64				ui64Reserved;			// Reserving 8 bytes to ensure the data segment
															// of the block is sized to an even 16 byte boundary
															// as needed by encryption.
} F_ENC_BTREE_BLK_HDR;

FINLINE FLMUINT sizeofBTreeBlkHdr(
	F_BTREE_BLK_HDR *		pBlkHdr)
{
	if (!isEncryptedBlk( (F_BLK_HDR *)pBlkHdr))
	{
			return sizeof( F_BTREE_BLK_HDR);
	}
	else
	{
		return sizeof( F_ENC_BTREE_BLK_HDR);
	}
}

/****************************************************************************
Desc:	 	Encrypted Data-only block header - on-disk format.
****************************************************************************/
typedef struct
{
	F_BLK_HDR		blk;
	FLMUINT32		ui32EncId;
	FLMBYTE			ucReserved[12];				// Reserving 12 bytes to ensure the data segment
															// of the block is sized to an even 16 byte boundary
															// as needed by encryption.
} F_ENC_DO_BLK_HDR;

FINLINE FLMUINT sizeofDOBlkHdr(
	F_BLK_HDR *		pBlkHdr
	)
{
	if (!isEncryptedBlk( pBlkHdr))
	{
			return sizeof( F_BLK_HDR);
	}
	else
	{
		return sizeof( F_ENC_DO_BLK_HDR);
	}
}

FINLINE FLMBOOL isRootBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	return( (pBlkHdr->ui8BTreeFlags & BLK_IS_ROOT) ? TRUE : FALSE);
}

FINLINE void setRootBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	pBlkHdr->ui8BTreeFlags |= BLK_IS_ROOT;
}

FINLINE void unsetRootBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	pBlkHdr->ui8BTreeFlags &= (~(BLK_IS_ROOT));
}

FINLINE FLMBOOL isIndexBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	return( (pBlkHdr->ui8BTreeFlags & BLK_IS_INDEX) ? TRUE : FALSE);
}

FINLINE eLFileType getBlkLfType(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	return( (eLFileType)((pBlkHdr->ui8BTreeFlags & BLK_IS_INDEX)
							? (eLFileType)XFLM_LF_INDEX
							: (eLFileType)XFLM_LF_COLLECTION));
}

FINLINE void setBlkLfType(
	F_BTREE_BLK_HDR *	pBlkHdr,
	FLMUINT				uiLfType
	)
{
	if (uiLfType == XFLM_LF_COLLECTION)
	{
		pBlkHdr->ui8BTreeFlags &= (~(BLK_IS_INDEX));
	}
	else
	{
		pBlkHdr->ui8BTreeFlags |= BLK_IS_INDEX;
	}
}

FINLINE FLMBOOL isContainerBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	return( (pBlkHdr->ui8BTreeFlags & BLK_IS_INDEX) ? FALSE : TRUE);
}

FINLINE void setIndexBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	pBlkHdr->ui8BTreeFlags |= BLK_IS_INDEX;
}

FINLINE void setContainerBlk(
	F_BTREE_BLK_HDR *	pBlkHdr
	)
{
	pBlkHdr->ui8BTreeFlags &= (~(BLK_IS_INDEX));
}

/****************************************************************************
Desc:	Get block header size.
****************************************************************************/
FINLINE FLMUINT blkHdrSize(
	F_BLK_HDR *	pBlkHdr
	)
{
	return( (pBlkHdr->ui8BlkType != BT_FREE &&
				pBlkHdr->ui8BlkType != BT_LFH_BLK)
				? (pBlkHdr->ui8BlkType == BT_DATA_ONLY
					? sizeofDOBlkHdr( pBlkHdr)
					: sizeofBTreeBlkHdr((F_BTREE_BLK_HDR *)pBlkHdr))
			  : SIZEOF_STD_BLK_HDR);
}

/****************************************************************************
Desc:	 	This is a union of all block header types - so that we can have
			something that gives us the largest block header type in one
			structure.  Reduce uses this.
****************************************************************************/
typedef struct FlmLargestBlkHdr
{
	union
	{
		F_BLK_HDR			stdBlkHdr;
		F_BTREE_BLK_HDR	BTreeBlkHdr;
	} all;
} F_LARGEST_BLK_HDR;

#define SIZEOF_LARGEST_BLK_HDR	sizeof( F_LARGEST_BLK_HDR)

/****************************************************************************
Desc:	 	Logical File (b-tree) header - on-disk format.
****************************************************************************/
typedef struct FlmLfHdrTag
{
	FLMUINT32	ui32LfNumber;
	FLMUINT32	ui32LfType;
	FLMUINT32	ui32RootBlkAddr;
	FLMUINT32	ui32EncId;
	FLMUINT64	ui64NextNodeId;
	FLMUINT64	ui64FirstDocId;
	FLMUINT64	ui64LastDocId;
	FLMBYTE		ucZeroes[ 24];			// Reserve 24 bytes of zero.

	// IMPORTANT NOTE: If anything is changed in here, need to make
	// corresponding changes to convertLfHdr routine and
	// flmVerifyDiskStructOffsets routine.

#define F_LF_HDR_ui32LfNumber_OFFSET				0
#define F_LF_HDR_ui32LfType_OFFSET					4
#define F_LF_HDR_ui32RootBlkAddr_OFFSET			8
#define F_LF_HDR_ui32EncId_OFFSET					12
#define F_LF_HDR_ui64NextNodeId_OFFSET				16
#define F_LF_HDR_ui64FirstDocId_OFFSET				24
#define F_LF_HDR_ui64LastDocId_OFFSET				32
#define F_LF_HDR_ucZeroes_OFFSET						40
} F_LF_HDR;

typedef struct
{
	FLMUINT		uiCollection;
	FLMUINT64	ui64Document;
	FLMUINT64	ui64NodeId;
} NODE_LIST_ITEM;

/****************************************************************************
Desc:	 	Node list
****************************************************************************/
class F_NodeList : public XF_RefCount, public XF_Base
{
public:

	F_NodeList()
	{
		m_pNodeTbl = NULL;
		m_uiNodeTblSize = 0;
		clearNodes();
	}

	~F_NodeList()
	{
		if (m_pNodeTbl)
		{
			f_free( &m_pNodeTbl);
		}
	}

	FINLINE void clearNodes( void)
	{
		m_uiNumNodes = 0;
		m_uiLastPosition = 0;
		m_uiLastCollection = 0;
		m_ui64LastDocument = 0;
		m_ui64LastNodeId = 0;
	}

	RCODE addNode(
		FLMUINT		uiCollection,
		FLMUINT64	ui64Document,
		FLMUINT64	ui64NodeId);

	FINLINE void removeNode(
		FLMUINT		uiPosition)
	{
		if( uiPosition < m_uiNumNodes)
		{
			if( uiPosition < m_uiNumNodes - 1)
			{
				f_memmove( &m_pNodeTbl[ uiPosition],
							  &m_pNodeTbl[ uiPosition + 1],
								  sizeof( NODE_LIST_ITEM) * (m_uiNumNodes - uiPosition));
			}
			m_uiNumNodes--;
		}

		m_uiLastPosition = 0;
		m_uiLastCollection = 0;
		m_ui64LastDocument = 0;
		m_ui64LastNodeId = 0;
	}

	void removeNode(
		FLMUINT		uiCollection,
		FLMUINT64	ui64Document,
		FLMUINT64	ui64NodeId);

	FINLINE FLMBOOL isNodeInList(
		FLMUINT		uiCollection,
		FLMUINT64	ui64Document,
		FLMUINT64	ui64NodeId)
	{
		FLMUINT	uiPos;

		return( findNode( uiCollection, ui64Document, ui64NodeId, &uiPos));
	}

	FLMBOOL findNode(
		FLMUINT		uiCollection,
		FLMUINT64	ui64Document,
		FLMUINT64	ui64NodeId,
		FLMUINT *	puiPos);

	FINLINE void getNode(
		FLMUINT			uiPosition,
		FLMUINT *		puiCollection,
		FLMUINT64 *		pui64Document,
		FLMUINT64 *		pui64NodeId)
	{
		if( uiPosition < m_uiNumNodes)
		{
			*puiCollection = m_pNodeTbl[ uiPosition].uiCollection;
			*pui64Document = m_pNodeTbl[ uiPosition].ui64Document;
			*pui64NodeId = m_pNodeTbl[ uiPosition].ui64NodeId;
		}
		else
		{
			*puiCollection = 0;
			*pui64Document = 0;
			*pui64NodeId = 0;
		}
	}

private:

	NODE_LIST_ITEM *	m_pNodeTbl;
	FLMUINT				m_uiNodeTblSize;
	FLMUINT				m_uiNumNodes;
	FLMUINT				m_uiLastPosition;
	FLMUINT				m_uiLastCollection;
	FLMUINT64			m_ui64LastDocument;
	FLMUINT64			m_ui64LastNodeId;

friend class F_Db;
friend class F_Database;
friend class F_Dict;
};

/***************************************************************************
Desc:		This is the notify request structure.  Notify requests are linked
			off of open requests for files or read requests for files so that
			when an operation is complete	that multiple threads are waiting
			on, all of them will be notified.
***************************************************************************/
typedef struct FNotify
{
	FNotify *		pNext;		// Pointer to next FNOTIFY structure in list.
	FLMUINT			uiThreadId;	// ID of thread requesting the notify
	RCODE  *			pRc;			// Pointer to a return code variable that is to
										// be filled in when the operation is completed.
										// The thread requesting notification supplies
										// the return code variable to be filled in.
	void *			pvUserData;	// Other user data that the notifier might use
										// to transfer other information to the waiter.
	F_SEM				hSem;			// Semaphore that will be signaled when the
										// operation is complete.
} FNOTIFY;

// Flags for the uiKrAction parameter - used in sorting/indexing.

#define KREF_DEL_KEYS			0x01
#define KREF_ADD_KEYS			0x02
#define KREF_INDEXING_ONLY		0x04
#define KREF_IN_MODIFY			0x10
#define KREF_MISSING_KEYS_OK	0x20

// Maximum length of a key.

#define MAX_KEY_SIZ				1024
#define MAX_ID_SIZE				256		// Cannot be more than 256 because
													// we can only use one byte to
													// represent the total ID size.
													
#define DEFAULT_KREF_TBL_SIZE					4096
#define DEFAULT_KREF_POOL_BLOCK_SIZE		8192

/****************************************************************************
Desc:		This structure is used to sort keys before the keys are actually
			added to an index.
****************************************************************************/
typedef struct Kref_Entry
{
	FLMBOOL		bDelete;					// Delete the key if TRUE
	FLMUINT		uiSequence;	  			// Sequence of updates within trans.
	FLMUINT		uiDataLen;				// Data length for this entry.  The
												// data, if any, comes after the key.
												// Note that there will be a null
												// terminating byte between the key and
												// the data.

	// Note: used uint16 below to reduce memory allocations.

	FLMUINT16	ui16IxNum;		  		// Index number
	FLMUINT16	ui16KeyLen;				// Key Length for this entry.  The key
												// comes immediately after this structure.
} KREF_ENTRY;

/****************************************************************************
Desc:		This is a temporary structure that is used when building compound
			keys.
****************************************************************************/
typedef struct Cdl
{
	F_DOMNode *	pNode;				// Node to be included in a compound key.
											// May be NULL.
	FLMUINT64	ui64ParentId;		// If pNode is NULL, this is a place
											// holder for a "missing" child node.  We
											// just need to remember the parent node id.
	FLMBOOL		bInNodeSubtree;	// Is this node subordinate to the original
											// node we are indexing on?
	Cdl *			pNext;				// Pointer to the next CDL entry.
} CDL;

/****************************************************************************
Desc:		This is a temporary structure that is used when building compound
			keys.
****************************************************************************/
typedef struct CdlHdrTag
{
	CDL *			pCdlList;
	FLMUINT64	ui64NodeId;
} CDL_HDR;

/****************************************************************************
Desc:		This is a temporary structure that is used when building compound
			keys.
****************************************************************************/
typedef struct IxContextTag
{
	F_Pool *			pPool;
	CDL_HDR *		pCdlTbl;
	CDL *				pCdlList;
	IxContextTag *	pNext;
	IxContextTag *	pPrev;
} IX_CONTEXT;

typedef struct IxdFixupTag
{
	FLMUINT			uiIndexNum;
	FLMUINT64		ui64LastDocIndexed;
	IxdFixupTag *	pNext;
} IXD_FIXUP;

#define	FTHREAD_ACTION_IDLE					0
#define	FTHREAD_ACTION_INDEX_OFFLINE		1

/***************************************************************************
Desc:		Contains elements for passing parms into the background thread.
***************************************************************************/
typedef struct F_BkgndIx
{
	F_Database *		pDatabase;
	FLMUINT				uiIndexingAction;
	XFLM_INDEX_STATUS	indexStatus;
	F_BkgndIx *			pPrev;
	F_BkgndIx *			pNext;
} F_BKGND_IX;

/****************************************************************************
Desc:	 	Structure used to pass information to the checkpoint thread for 3.x
			databases.
****************************************************************************/
typedef struct
{
	F_Database *		pDatabase;
	F_SuperFileHdl *	pSFileHdl;
	F_SEM					hWaitSem;
	XFLM_STATS			Stats;
	FLMBOOL				bStatsInitialized;
	FLMBOOL				bShuttingDown;
	FLMBOOL				bDoingCheckpoint;
	FLMUINT				uiStartTime;
	FLMBOOL				bForcingCheckpoint;
	FLMUINT				uiForceCheckpointStartTime;
	FLMINT				iForceCheckpointReason;
	FLMUINT				uiLogBlocksWritten;
	FLMBOOL				bWritingDataBlocks;
	FLMUINT				uiDataBlocksWritten;
	FLMUINT				uiStartWaitTruncateTime;
} CP_INFO;

#define MAX_WRITE_BUFFER_BYTES			(4 * 1024 * 1024)
#define MAX_PENDING_WRITES					(MAX_WRITE_BUFFER_BYTES / 4096)
#define MAX_LOG_BUFFER_SIZE				(256 * 1024)

typedef struct Tmp_Read_Stats
{
	XFLM_DISKIO_STAT		BlockReads;					// Statistics on block reads.
	XFLM_DISKIO_STAT		OldViewBlockReads;		// Statistics on old view block
																// reads.
	FLMUINT					uiBlockChkErrs;			// Number of times we had
																// check errors reading blocks.
	FLMUINT					uiOldViewBlockChkErrs;	// Number of times we had
																// check errors reading an
																// old view of a block.
} TMP_READ_STATS, * TMP_READ_STATS_p;

// Flags for F_Database->m_uiFlags

#define DBF_BEING_OPENED			0x01	// Flag indicating whether this database is
													// in the process of being opened.
#define DBF_BEING_CLOSED			0x02	// Database is being closed - cannot open.

/*****************************************************************************
Desc: Shared database object - only to be used internally
*****************************************************************************/
class F_Database : public XF_RefCount, public XF_Base
{
public:

	F_Database(
		FLMBOOL	bTempDb);

	~F_Database();

	void freeDatabase( void);

	RCODE setupDatabase(
		const char *	pszDbPath,
		const char *	pszDataDir);

	RCODE dbWriteLock(
		F_SEM					hWaitSem,
		XFLM_DB_STATS *	pDbStats = NULL,
		FLMUINT				uiTimeout = XFLM_NO_TIMEOUT);

	void dbWriteUnlock(
		XFLM_DB_STATS *	pDbStats = NULL);

	void shutdownDatabaseThreads( void);

	RCODE linkToBucket( void);			// was flmLinkFileToBucket

	void setMustCloseFlags(					// was flmSetMustCloseFlags
		RCODE		rcMustClose,
		FLMBOOL	bMutexLocked);

	FINLINE F_NameTable * getNameTable( void)
	{
		if( m_pDictList)
		{
			return( m_pDictList->getNameTable());
		}

		return( NULL);
	}

	FINLINE RCODE checkState(
		const char *	pszFileName,
		FLMINT			iLineNumber)

	{
		RCODE		rc = NE_XFLM_OK;

		if (m_bMustClose)
		{
			logMustCloseReason( pszFileName, iLineNumber);
			rc = RC_SET( NE_XFLM_MUST_CLOSE_DATABASE);
		}
		return( rc);
	}

	RCODE startCPThread( void);

	FLMBOOL tryCheckpoint(
		F_Thread *		pThread,
		CP_INFO *		pCPInfo);

	void newDatabaseFinish(
		RCODE				OpenRc);

	RCODE getExclAccess(
		const char *	pszFilePath);

	RCODE verifyOkToUse(
		FLMBOOL *		pbWaited);

	RCODE writeDbHdr(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		XFLM_DB_HDR *			pDbHdr,
		XFLM_DB_HDR *			pCPDbHdr,
		FLMBOOL					bIsCheckpoint);

	FINLINE char * getDbNamePtr( void)
	{
		return m_pszDbPath;
	}

	FINLINE XFLM_DB_HDR * getUncommittedDbHdr( void)
	{
		return &m_uncommittedDbHdr;
	}

	FINLINE FLMUINT getBlockSize( void)
	{
		return( m_uiBlockSize);
	}

	FINLINE FLMUINT getMaxFileSize( void)
	{
		return( m_uiMaxFileSize);
	}

	FINLINE FLMUINT getSigBitsInBlkSize( void)
	{
		return m_uiSigBitsInBlkSize;
	}

	FINLINE F_CachedBlock * getTransLogList( void)
	{
		return m_pTransLogList;
	}

	void releaseLogBlocks( void);

	FINLINE FLMUINT getDirtyCacheCount( void)
	{
		return m_uiDirtyCacheCount;
	}

	FINLINE void incrementDirtyCacheCount( void)
	{
		m_uiDirtyCacheCount++;
	}

	FINLINE void decrementDirtyCacheCount( void)
	{
		m_uiDirtyCacheCount--;
	}

	FLMBOOL neededByReadTrans(
		FLMUINT64			ui64LowTransId,
		FLMUINT64			ui64HighTransId);

	RCODE getBlock(
		F_Db *				pDb,
		LFILE *				pLFile,
		FLMUINT				uiBlkAddress,
		FLMUINT *			puiNumLooks,
		F_CachedBlock **	ppSCache);

	RCODE logPhysBlk(
		F_Db *				pDb,
		F_CachedBlock **	ppSCache,
		F_CachedBlock **	ppOldCache = NULL);

	RCODE createBlock(
		F_Db *				pDb,
		F_CachedBlock **	ppSCache);

	RCODE blockUseNextAvail(
		F_Db *				pDb,
		F_CachedBlock **	ppSCache);

	RCODE blockFree(
		F_Db *				pDb,
		F_CachedBlock *	pSCache);

	RCODE moveBtreeBlk(
		F_Db *				pDb,
		FLMUINT				uiBlkAddr,
		FLMUINT				uiLfNumber,
		eLFileType			eLfType);

	RCODE freeAvailBlk(
		F_Db *				pDb,
		FLMUINT				uiBlkAddr);

	RCODE moveLFHBlk(
		F_Db *				pDb,
		FLMUINT				uiBlkAddr);

	void unlinkTransLogBlocks( void);

	void freeBlockCache( void);

	void freeNodeCache( void);

	void freeModifiedBlocks(
		FLMUINT64			ui64CurrTransId);

	void freeModifiedNodes(
		F_Db *				pDb,
		FLMUINT64			ui64OlderTransId);

	void commitNodeCache( void);

	void getCPInfo(
		XFLM_CHECKPOINT_INFO *		pCheckpointInfo);

	FINLINE FLMUINT getFlags( void)
	{
		return( m_uiFlags);
	}

	// NOTE: This routine expects that the global mutex is locked when
	// it is called.

	FINLINE void incrOpenCount( void)
	{
		m_uiOpenIFDbCount++;
	}

	// NOTE: This routine expects that the global mutex is locked when
	// it is called.  It may temporarily unlock the mutex (when it
	// calls freeDatabase), but the mutex will be locked when it returns.

	FINLINE void decrOpenCount( void)
	{
		flmAssert( m_uiOpenIFDbCount);
		m_uiOpenIFDbCount--;
		if (!m_uiOpenIFDbCount)
		{
			freeDatabase();
		}
	}

	RCODE startMaintThread( void);

	FINLINE FLMBOOL inLimitedMode()
	{
		return m_bInLimitedMode;
	}

	RCODE encryptBlock(
		F_Dict *		pDict,
		FLMBYTE *	pucBuffer);

	RCODE decryptBlock(
		F_Dict *		pDict,
		FLMBYTE *	pucBuffer);
		
	FINLINE void lockMutex( void)
	{
		f_mutexLock( m_hMutex);
	}
		
	FINLINE void unlockMutex( void)
	{
		f_mutexUnlock( m_hMutex);
	}
		
private:

	void logMustCloseReason(
		const char *			pszFileName,
		FLMINT					iLineNumber);

	RCODE readDbHdr(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMBYTE *				pszPassword,
		FLMBOOL					bAllowLimited);

	RCODE physOpen(
		F_Db *					pDb,
		const char *			pszFilePath,
		const char *			pszRflDir,
		const char *			pszPassword,
		FLMUINT					uiOpenFlags,
		FLMBOOL					bNewDatabase,
		IF_RestoreClient *	pRestoreObj,
		IF_RestoreStatus *	pRestoreStatus);

	RCODE doRecover(
		F_Db *					pDb,
		IF_RestoreClient *	pRestoreObj,
		IF_RestoreStatus *	pRestoreStatus);

	RCODE outputNode(
		FLMBYTE *				pucKeyBuf,
		F_DOMNode *				pNode,
		FLMBOOL					bAdd,
		F_Btree *				pBTree);

	RCODE readTheBlock(
		F_Db *					pDb,
		TMP_READ_STATS *		pTmpReadStats,
		F_BLK_HDR *				pBlkHdr,
		FLMUINT					uiFilePos,
		FLMUINT					uiBlkAddress);

	RCODE readBlock(
		F_Db *					pDb,
		LFILE *					pLFile,
		FLMUINT					uiFilePos,
		FLMUINT					uiBlkAddress,
		FLMUINT64				ui64NewerBlkLowTransID,
		F_CachedBlock *		pSCache,
		FLMBOOL *				pbFoundVerRV,
		FLMBOOL *				pbDiscardRV);

	RCODE readIntoCache(
		F_Db *					pDb,
		LFILE *					pLFile,
		FLMUINT					uiBlkAddress,
		F_CachedBlock *		pPrevInVerList,
		F_CachedBlock *		pNextInVerList,
		F_CachedBlock **		ppSCacheRV,
		FLMBOOL *				pbGotFromDisk);

	void setBlkDirty(
		F_CachedBlock *		pSCache);

	RCODE allocBlocksArray(
		FLMUINT					uiNewSize,
		FLMBOOL					bOneArray);

	RCODE flushLogBlocks(
		F_SEM						hWaitSem,
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMBOOL					bIsCPThread,
		FLMUINT					uiMaxDirtyCache,
		FLMBOOL *				pbForceCheckpoint,
		FLMBOOL *				pbWroteAll);

	RCODE reduceNewBlocks(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMUINT *				puiBlocksFlushed);

	RCODE writeContiguousBlocks(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		F_IOBuffer *			pIOBuffer,
		FLMUINT					uiBlkAddress,
		FLMBOOL					bDoAsync);

	RCODE writeSortedBlocks(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMUINT					uiMaxDirtyCache,
		FLMUINT *				puiDirtyCacheLeft,
		FLMBOOL *				pbForceCheckpoint,
		FLMBOOL					bIsCPThread,
		FLMBOOL					bDoAsync,
		FLMUINT					uiNumSortedBlocks,
		FLMBOOL *				pbWroteAll);

	RCODE flushDirtyBlocks(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMUINT					uiMaxDirtyCache,
		FLMBOOL					bForceCheckpoint,
		FLMBOOL					bIsCPThread,
		FLMBOOL *				pbWroteAll);

	RCODE reduceDirtyCache(	
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl);

	RCODE finishCheckpoint(
		F_SEM						hWaitSem,
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMBOOL					bDoTruncate,
		FLMUINT					uiCPFileNum,
		FLMUINT					uiCPOffset,
		FLMUINT					uiCPStartTime,
		FLMUINT					uiTotalToWrite);

	RCODE doCheckpoint(
		F_SEM						hWaitSem,
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMBOOL					bDoTruncate,
		FLMBOOL					bForceCheckpoint,
		FLMINT					iForceReason,
		FLMUINT					uiCPFileNum,
		FLMUINT					uiCPOffset);

	RCODE lgFlushLogBuffer(
		XFLM_DB_STATS *		pDbStats,
		F_SuperFileHdl *		pSFileHdl,
		FLMBOOL					bDoAsync);

	RCODE lgOutputBlock(
		XFLM_DB_STATS *	pDbStats,
		F_SuperFileHdl *	pSFileHdl,
		F_CachedBlock *	pLogBlock,
		F_BLK_HDR *			pBlkHdr,
		FLMBOOL				bDoAsync,
		FLMUINT *			puiLogEofRV);

	FLMUINT lFileFindEmpty(
		FLMUINT				uiBlockSize,
		F_BLK_HDR *			pBlkHdr);

	RCODE lFileRead(
		F_Db *				pDb,
		LFILE *				pLFile,
		F_COLLECTION *		pCollection);

	RCODE lFileWrite(
		F_Db *				pDb,
		F_COLLECTION *		pCollection,
		LFILE *				pLFile);

	RCODE lFileCreate(
		F_Db *				pDb,
		LFILE *				pLFile,
		F_COLLECTION *		pCollection,
		FLMUINT				uiLfNum,
		eLFileType			eLfType,
		FLMBOOL				bCounts,
		FLMBOOL				bHaveData,
		FLMUINT				uiEncId = 0);

	RCODE startPendingInput(
		FLMUINT				uiPendingType,
		F_CachedNode *		pPendingNode);

	void endPendingInput( void);

	RCODE lFileDelete(
		F_Db *				pDb,
		F_COLLECTION *		pCollection,
		LFILE *				pLFile,
		FLMBOOL				bCounts,
		FLMBOOL				bHaveData);

	static RCODE maintenanceThread(
		F_Thread *			pThread);

	F_Database *			m_pNext;					// Next F_Database structure in in name hash
															// bucket, dependent store hash
															// bucket, or avail list.
	F_Database *			m_pPrev;					// Previous F_Database structure in name hash
															// bucket or dependent store hash
															// bucket.
	F_Query *				m_pFirstQuery;			// First query currently linked to this DB
	F_Query *				m_pLastQuery;			// Last query currently linked to this DB
	FLMUINT					m_uiBlockSize;			// Block size for database
	FLMUINT					m_uiDefaultLanguage;	// Default language for database.
	FLMUINT					m_uiMaxFileSize;		// Maximum file size for the database.
	FLMUINT					m_uiOpenIFDbCount;	// Number of IF_Dbs currently using this
															// database.  Does NOT count internal uses
	FLMBOOL					m_bTempDb;				// Is this a temporary database?  If so,
															// minimize writing out to disk.
	F_Db *					m_pFirstDb;				// List of ALL F_Db's associated with
															// this database.
	char *					m_pszDbPath;			// Database file name.
	char *					m_pszDataDir;			// Path for data files.
	F_Database *			m_pNextNUDatabase;	// Next F_Database structure in list of
															// unused databases.  When use count goes
															// to zero, the structure is linked
															// into a list of unused databases off of
															// the FSYSDATA structure.
	F_Database *			m_pPrevNUDatabase;	// Previous F_Database structure in list of
															// unused databases.
	F_CachedBlock *		m_pSCacheList;			// This is a pointer to a linked list
															// of all shared cache blocks
															// belonging to this database.
	F_CachedNode *			m_pFirstNode;			// First cached DOM node that belongs
															// to this database.  Also points to
															// the first dirty DOM node, if any.
	F_CachedNode *			m_pLastNode;			// Last cached DOM node that belongs
															// to this database.
	F_CachedNode *			m_pLastDirtyNode;		// Last dirty DOM node that belongs
															// to this database.
	F_CachedBlock *		m_pPendingWriteList;	// This is a pointer to a linked list
															// of all shared cache blocks
															// that are in the pending-write state.
	F_CachedBlock *		m_pLastDirtyBlk;		// Pointer to last dirty block in the
															// list.
	F_CachedBlock *		m_pFirstInLogList;	// First block that needs to be logged
	F_CachedBlock *		m_pLastInLogList;		// Last block that needs to be logged
	FLMUINT					m_uiLogListCount;		// Number of items in the log list
	F_CachedBlock *		m_pFirstInNewList;	// First new block that is dirty
	F_CachedBlock *		m_pLastInNewList;		// Last new block that is dirty
	FLMUINT					m_uiNewCount;			// Number of items in new list
	FLMUINT					m_uiDirtyCacheCount;	// Number of dirty blocks
	FLMUINT					m_uiLogCacheCount;	// Log blocks needing to be written.
	F_CachedBlock **		m_ppBlocksDone;		// List of blocks to be written to rollback
															// log or database.
	FLMUINT					m_uiBlocksDoneArraySize;// Size of ppBlocksDone array.
	FLMUINT					m_uiBlocksDone;		// Number of blocks currently in the
															// ppBlocksDone array.
	F_CachedBlock *		m_pTransLogList;		// This is a pointer to a linked list
															// of all shared cache blocks
															// belonging to this database that need
															// to be logged to the rollback log
															// for the current transaction.
	FNOTIFY *				m_pOpenNotifies;		// Pointer to a list of notifies to
															// perform when this database is finally
															// opened (points to a linked list of
															// FNOTIFY structures).
	FNOTIFY *				m_pCloseNotifies;		// Pointer to a list of notifies to
															// perform when this database is finally
															// closed (points to a linked list of
															// FNOTIFY structures).
	F_Dict *					m_pDictList;			// Pointer to linked list of
															// dictionaries currently being used
															// for this database.  The linked list
															// is a list of versions of the
															// dictionary.  When a version is no
															// longer used, it is removed from the
															// list.  Hence, the list is usually
															//	has only one member.
	FLMBOOL					m_bMustClose;			// The database is being forced to close
															// because of a critical error.
	RCODE						m_rcMustClose;			// Return code that caused bMustClose to
															// be set.
	F_Pool 					m_krefPool;				// Kref pool to be used during update
															// transactions.
	FLMUINT					m_uiSigBitsInBlkSize;// Significant bits in the database's
															// block size.
	FLMUINT					m_uiFileExtendSize;	// Bytes to extend files by.
	F_Rfl *					m_pRfl;					// Pointer RFL object.


	XFLM_DB_HDR				m_lastCommittedDbHdr;// This is the last committed DBheader.
	XFLM_DB_HDR				m_checkpointDbHdr;	// This is the DB header as of the start
															// of the last checkpoint.
	XFLM_DB_HDR				m_uncommittedDbHdr;	// This is the uncommitted DB header.
															// It is used by the current update
															// transaction.
	F_FileIdList *			m_pFileIdList;			// List of unique IDs that have been
															// assigned to the physical files that
															// are mananaged by the FFILE.
	F_IOBufferMgr *		m_pBufferMgr;

	F_IOBuffer *			m_pCurrLogBuffer;
	FLMUINT					m_uiCurrLogWriteOffset;	// Offset in current write buffer
	FLMUINT					m_uiCurrLogBlkAddr;
															// Address of first block in the current
															// buffer.
	XFLM_DB_HDR *			m_pDbHdrWriteBuf;		// Aligned buffer (on win32) for writing
															// the DB header.
	FLMBYTE *				m_pucUpdBuffer;		// Buffer for writing out records.
	FLMUINT					m_uiUpdBufferSize;	// Size of update buffer.
	FLMBYTE					m_ucIV [16];			// Used when outputting encrypted data.
	F_CachedNode *			m_pPendingInput;		// Node that is having its value
															// set across multiple calls.
	F_Btree *				m_pPendingBTree;		// B-Tree used by node that is having
															// its value set across multiple calls
	FLMBOOL					m_bUpdFirstBuf;
	FLMUINT					m_uiUpdByteCount;
	FLMUINT					m_uiUpdCharCount;
	FLMUINT					m_uiPendingType;
	FLMBYTE *				m_pucBTreeTmpBlk;		// Temporary buffer for F_Btree object
															// to use during a call that updates
															// a B-Tree - btInsertEntry,
															// btReplaceEntry, etc.  SHOULD NOT
															// be used in read/find operations!
	FLMBYTE *				m_pucBTreeTmpDefragBlk;
	FLMBYTE *				m_pucEntryArray;		// Temporary buffer for F_Btree object
															// to use during update operations.
	FLMBYTE *				m_pucSortedArray;		// Temporary buffer for F_Btree object
															// to use during update operations.
	FLMBYTE *				m_pucBtreeBuffer;		// Buffer used by the Btree during moves
															// between blocks.
	FLMBYTE *				m_pucReplaceStruct;	// Buffer used by the Btree to hold additional
															// replace information during updates *only*.
	ServerLockObject *	m_pDatabaseLockObj;	// Object for locking the database.
	ServerLockObject *	m_pWriteLockObj;		// Object for locking to do writing.
	IF_FileHdl *			m_pLockFileHdl;		// Lock file handle.
	FNOTIFY *				m_pLockNotifies;		// Pointer to a list of notifies to
															// perform when this database is finally
															// locked (points to a linked list of
															// FNOTIFY structures).
	FLMBOOL					m_bBeingLocked;		// Flag indicating whether or not this
															// database is in the process of being
															// locked for exclusive access.
	F_Db *					m_pFirstReadTrans;	// Pointer to first read transaction for
															// this database.
	F_Db *					m_pLastReadTrans;		// Pointer to last read transaction for
															// this database.
	F_Db *					m_pFirstKilledTrans;	// List of read transactions that have
															// been killed.
	FLMUINT					m_uiFirstLogBlkAddress;
															// Address of first block logged for the
															// current update transaction.

	FLMUINT					m_uiFirstLogCPBlkAddress;
															// Address of first block logged for the
															// current checkpoint.
	FLMUINT					m_uiLastCheckpointTime;
															// Last time we successfully completed a
															// checkpoint.
	F_Thread *				m_pCPThrd;				// Checkpoint thread.
	CP_INFO *				m_pCPInfo;				// Pointer to checkpoint thread's
															// information buffer - used for
															// communicating information to the
															// checkpoint thread.
	RCODE						m_CheckpointRc;		// Return code from last checkpoint
															// that was attempted.
	FLMUINT					m_uiBucket;				// Hash bucket this database is in.
															// 0xFFFF means it is not currently
															// in a bucket.
	FLMUINT					m_uiFlags;				// Flags for this database.
	FLMBOOL					m_bBackupActive;		// Backup is currently being run against the
															// database.
	F_NodeList				m_DocumentList;		// List of documents modified in a transaction
	F_Thread *				m_pMaintThrd;			// Background maintenance thread
	F_SEM						m_hMaintSem;			// Maintenance thread "work-to-do" semaphore
	FLMBYTE *				m_pszDbPasswd;			// The database encryption password
	F_CCS *					m_pWrappingKey;		// The database wrapping key
	FLMBOOL					m_bHaveEncKey;			// 
	FLMBOOL					m_bAllowLimitedMode;	// Is this database allowed to be opened in limited mode?
	FLMBOOL					m_bInLimitedMode;		// Has this database been opened in limited mode?
	RCODE						m_rcLimitedCode;
	F_MUTEX					m_hMutex;				// Database mutex.

friend class F_Db;
friend class F_Rfl;
friend class F_Btree;
friend class F_Dict;
friend class F_DbSystem;
friend class F_Backup;
friend class FFileItemId;
friend class F_DOMNode;
friend class F_BTreeIStream;
friend class F_DbRebuild;
friend class F_DbCheck;
friend class F_Query;
friend class F_BtResultSet;
friend class F_BtRSFactory;
friend class FSIndexCursor;
friend class FSCollectionCursor;
friend class F_CachedNode;
friend class F_CachedBlock;
friend class F_BlockCacheMgr;
friend class F_NodeCacheMgr;
friend class F_GlobalCacheMgr;
friend class F_QueryResultSet;
friend class F_BTreeInfo;
friend class F_NodeRelocator;
friend class F_NodeDataRelocator;
friend class F_NodeListRelocator;	
friend class F_AttrItemRelocator;	
friend class F_AttrBufferRelocator;	
friend class F_BlockRelocator;
};

/***************************************************************************
Desc:		This is the hash bucket header structure.  Each bucket header
			points to a list of items that belong to the bucket.
***************************************************************************/
typedef struct FBucket
{
	void *		pFirstInBucket;	// Pointer to first item in the bucket.
											// The type of structure being pointed to
											// depends on the usage of the hash bucket.
	FLMUINT		uiHashValue;		// Hash value for this bucket.
} FBUCKET;

typedef struct QueryHdrTag
{
	F_Query *		pQuery;
	QueryHdrTag *	pNext;
	QueryHdrTag *	pPrev;
} QUERY_HDR;

/***************************************************************************
Desc:		This is the FLAIM Event Structure.  It keeps track of a registered
			event callback function that has been registered for a particular
			event category.
***************************************************************************/
typedef struct F_Event
{
	IF_EventClient *	pEventClient;
	F_Event *			pNext;
	F_Event *			pPrev;
} FEVENT;

/***************************************************************************
Desc:		This is the FLAIM Event Header Structure.  It is the header for
			the list of events that have been registered for a particular
			event category.
***************************************************************************/
typedef struct F_Event_Hdr
{
	FEVENT *			pEventCBList;		// List of registered event callbacks.
	F_MUTEX			hMutex;				// Mutex to control access to the
												// the event list.
} FEVENT_HDR;

typedef enum
{
	HASH_SESSION_OBJ = 0,
	HASH_DB_OBJ
} eHashObjType;

/*===========================================================================
Desc: FLAIM object base class
===========================================================================*/
class F_HashObject : public XF_RefCount, public XF_Base
{
public:

#define F_INVALID_HASH_BUCKET				(~((FLMUINT)0))

	F_HashObject()
	{
		m_pNextInBucket = NULL;
		m_pPrevInBucket = NULL;
		m_pNextInGlobal = NULL;
		m_pPrevInGlobal = NULL;
		m_uiHashBucket = F_INVALID_HASH_BUCKET;
		m_ui32CRC = 0xFFFFFFFF;
	}

	virtual ~F_HashObject()
	{
		flmAssert( !m_pNextInBucket);
		flmAssert( !m_pPrevInBucket);
		flmAssert( !m_pNextInGlobal);
		flmAssert( !m_pPrevInGlobal);
		flmAssert( !getRefCount());
	}

	virtual void * getKey(
		FLMUINT *	puiKeyLen) = 0;

	FLMUINT getHashBucket( void)
	{
		return( m_uiHashBucket);
	}

	FLMUINT32 getKeyCRC( void)
	{
		return( m_ui32CRC);
	}

	FINLINE F_HashObject * getNextInGlobal( void)
	{
		return( m_pNextInGlobal);
	}

	virtual eHashObjType objectType( void) = 0;

protected:

	// Methods

	void setHashBucket(
		FLMUINT		uiHashBucket)
	{
		m_uiHashBucket = uiHashBucket;
	}

	void setKeyCRC(
		FLMUINT32	ui32CRC)
	{
		m_ui32CRC = ui32CRC;
	}

	// Data

	F_HashObject *		m_pNextInBucket;
	F_HashObject *		m_pPrevInBucket;
	F_HashObject *		m_pNextInGlobal;
	F_HashObject *		m_pPrevInGlobal;
	FLMUINT				m_uiHashBucket;
	FLMUINT32			m_ui32CRC;

friend class F_HashTable;
};

/*===========================================================================
Desc: FLAIM hash table
===========================================================================*/
class F_HashTable : public XF_RefCount, public XF_Base
{
public:

	F_HashTable();

	~F_HashTable();

	RCODE setupHashTable(
		FLMBOOL				bMultithreaded,
		FLMUINT				uiNumBuckets,
		FLMUINT32 *			pCRCTable);

	RCODE addObject(
		F_HashObject *		pObject);

	RCODE getNextObjectInGlobal(
		F_HashObject **	ppObject);

	RCODE getObject(
		void *				pvKey,
		FLMUINT				uiKeyLen,
		F_HashObject **	ppObject,
		FLMBOOL				bRemove = FALSE);

	RCODE removeObject(
		void *				pvKey,
		FLMUINT				uiKeyLen);

	RCODE removeObject(
		F_HashObject *		pObject);

private:

	// Methods

	FLMUINT getHashBucket(
		void *				pvKey,
		FLMUINT				uiLen,
		FLMUINT32 *			pui32KeyCRC = NULL);

	void linkObject(
		F_HashObject *		pObject,
		FLMUINT				uiBucket);

	void unlinkObject(
		F_HashObject *		pObject);

	RCODE findObject(
		void *				pvKey,
		FLMUINT				uiKeyLen,
		F_HashObject **	ppObject);

	// Data

	F_MUTEX 				m_hMutex;
	F_HashObject *		m_pGlobalList;
	F_HashObject **	m_ppHashTable;
	FLMUINT				m_uiBuckets;
	FLMUINT32 *			m_pCRCTable;
	FLMBOOL				m_bOwnCRCTable;
};

typedef struct node_loc
{
	FLMUINT32	ui32DatabaseId;
	FLMUINT32	ui32BlockAddr;
	FLMUINT32	ui32OffsetIndex;
	FLMUINT32	ui32Collection;
	FLMUINT64	ui64NodeId;
	node_loc *	pPrevInBucket;
	node_loc *	pNextInBucket;
} F_NODE_LOCATION;

/***************************************************************************
Desc:		This is the FLAIM Shared System Data Structure.  It is the anchor
			for all of the other shared structures.
***************************************************************************/
typedef struct FlmSystemData
{
	FBUCKET *				pDatabaseHashTbl;	// Database name hash table (array of FBUCKET).
#define FILE_HASH_ENTRIES		256

	F_MUTEX					hShareMutex;	// Mutex for controlling access to
													// various global items.
	F_MUTEX					hNodeCacheMutex;
													// Mutex for controlling access to
													// node cache.
	F_MUTEX					hBlockCacheMutex;
													// Mutex for controlling access to
													// block cache.
	F_FileHdlMgr *			pFileHdlMgr;	// Used to Manage all FileHdl objects

	FLMBOOL					bTempDirSet;	// TRUE if temporary directory has been set

	FLMBOOL					bOkToDoAsyncWrites;
													// OK To do async writes, if available.
	FLMBOOL					bOkToUseESM;	// OK to use Extended Server Memory,
													// if available
	ServerLockManager *	pServerLockMgr;
													// Pointer to server lock manager.
	FLMUINT					uiMaxCPInterval;
													// Maximum number of seconds to allow between
													// checkpoints
	F_GlobalCacheMgr *	pGlobalCacheMgr;
	F_BlockCacheMgr *		pBlockCacheMgr;
	F_NodeCacheMgr *		pNodeCacheMgr;	// Node cache manager
	FLMUINT					uiRehashAfterFailureBackoffTime;
													// Amount of time to wait after trying
													// to reallocate a cache manager's hash
													// table before trying again.
	F_NodePool *			pNodePool;		// Pool of nodes that can be re-used
	F_Thread *				pMonitorThrd;	// Monitor thread
	F_Thread *				pCacheCleanupThrd;
	XFLM_STATS				Stats;			// Statistics structure
	F_MUTEX					hStatsMutex;	// Mutex for statistics structure

	F_MUTEX					hQueryMutex;	// Mutex for managing query list
	QUERY_HDR *				pNewestQuery;	// Head of query list (newest)
	QUERY_HDR *				pOldestQuery;	// Tail of query list (oldest)
	FLMUINT					uiQueryCnt;		// Number of queries in the list
	FLMUINT					uiMaxQueries;	// Maximum number of queries to keep around
	FLMBOOL					bNeedToUnsetMaxQueries;
													// When TRUE, indicates that a call to stop
													// statistics should also stop saving
													// queries.
	FLMBOOL					bStatsInitialized;
													// Has statistics structure been
													// initialized?

	char						szTempDir[ F_PATH_MAX_SIZE];
													// Temporary working directory for
													// ResultSets, RecordCache
													// and other sub-systems that need
													// temporary files.  This is aligned
													// on a 4-byte boundary

	FLMUINT					uiMaxUnusedTime;
													// Maximum number of timer units to keep
													// unused structures in memory before
													// freeing them.
	FEVENT_HDR				EventHdrs [XFLM_MAX_EVENT_CATEGORIES];
	F_Pool *					pKRefPool;		// Memory Pool that is only used by
													// record updaters for key building

	FLMUINT					uiMaxFileSize;
	IF_LoggerClient *		pLogger;
	FLMUINT					uiPendingLogMessages;
	F_MUTEX					hLoggerMutex;

#ifdef FLM_LINUX
	FLMUINT					uiLinuxMajorVer;
	FLMUINT					uiLinuxMinorVer;
	FLMUINT					uiLinuxRevision;
#endif

#ifdef FLM_DEBUG
	// Variables for memory allocation tracking.

	FLMBOOL					bTrackLeaks;
	FLMBOOL					bLogLeaks;
	FLMBOOL					bStackWalk;
	FLMBOOL					bMemTrackingInitialized;
	FLMUINT					uiInitThreadId;
	F_MUTEX					hMemTrackingMutex;
	void **					ppvMemTrackingPtrs;
	FLMUINT					uiMemTrackingPtrArraySize;
	FLMUINT					uiMemNumPtrs;
	FLMUINT					uiMemNextPtrSlotToUse;
	FLMUINT					uiAllocCnt;
	#if defined( FLM_WIN)
		HANDLE				hMemProcess;
	#endif

	#ifdef DEBUG_SIM_OUT_OF_MEM
		FLMUINT				uiOutOfMemSimEnabledFlag;
		// We pick a random number for the flag so that it is hard to accidentally
		// turn this flag on by writing memory out-of-bounds.
		
		#define OUT_OF_MEM_SIM_ENABLED_FLAG 2149614134UL
		
		F_RandomGenerator	memSimRandomGen;
		FLMUINT			uiSimOutOfMemFailTotal;
		FLMUINT			uiSimOutOfMemFailSequence;
	#endif
#endif

	F_MUTEX			hIniMutex;
	F_ThreadMgr *	pThreadMgr;
	F_MUTEX			hHttpSessionMutex;
	F_BtPool *		pBtPool;
	F_XML *			pXml;
#ifdef FLM_NLM
	FLMBOOL			bUseNSSFileHdls;
#endif
} FLMSYSDATA;

#ifndef ALLOCATE_SYS_DATA
	extern FLMSYSDATA		gv_XFlmSysData;
#else
	FLMSYSDATA				gv_XFlmSysData;
#endif

FINLINE FLMBOOL f_isWhiteSpace(
	FLMBYTE	ucChar)
{
	return( ucChar == ASCII_SPACE || ucChar == ASCII_TAB ? TRUE : FALSE);
}

FINLINE FLMUNICODE flmConvertChar(
	FLMUNICODE	uzChar,
	FLMUINT		uiCompareRules)
{
	if (uzChar == ASCII_SPACE ||
		 (uzChar == ASCII_UNDERSCORE &&
		  (uiCompareRules & XFLM_COMP_NO_UNDERSCORES)) ||
		 (gv_XFlmSysData.pXml->isWhitespace( uzChar) &&
		  (uiCompareRules & XFLM_COMP_WHITESPACE_AS_SPACE)))
	{
		return( (FLMUNICODE)((uiCompareRules &
									 (XFLM_COMP_NO_WHITESPACE |
									  XFLM_COMP_IGNORE_LEADING_SPACE))
									 ? (FLMUNICODE)0
									 : (FLMUNICODE)ASCII_SPACE));
	}
	else if (uzChar == ASCII_DASH && (uiCompareRules & XFLM_COMP_NO_DASHES))
	{
		return( (FLMUNICODE)0);
	}
	else
	{
		return( uzChar);
	}
}

#if defined( FLM_WIN) || defined( FLM_NLM) || defined( FLM_LINUX)
	#pragma pack(pop)
#else
	#pragma pack()
#endif

#endif // FSTRUCTS_H
