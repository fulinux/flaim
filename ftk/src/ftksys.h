//------------------------------------------------------------------------------
// Desc:	Cross-platform macros, defines, etc.  Must visit this file
//			to port XFLAIM to another platform.
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
// $Id: ftk.h 3123 2006-01-24 17:19:50 -0700 (Tue, 24 Jan 2006) ahodgkinson $
//------------------------------------------------------------------------------

#ifndef FTKSYS_H
#define FTKSYS_H

	#include "ftk.h"

	#ifndef FLM_PLATFORM_CONFIGURED
		#error Platform not configured
	#endif
	
	class F_Thread;
	class F_ThreadMgr;
	class F_IOBuffer;
	class F_FileSystem;
	class F_ThreadMgr;
	class F_ResultSet;
	class F_ResultSetBlk;

	#define RS_BLOCK_SIZE				(1024 * 512)
	#define RS_POSITION_NOT_SET		FLM_MAX_UINT64
	#define RS_MAX_FIXED_ENTRY_SIZE	64
	
	// Cell sizes for buffer allocator
	
	#define CELL_SIZE_0			16
	#define CELL_SIZE_1			32
	#define CELL_SIZE_2			64
	#define CELL_SIZE_3			128
	#define CELL_SIZE_4			192
	#define CELL_SIZE_5			320
	#define CELL_SIZE_6			512
	#define CELL_SIZE_7			672
	#define CELL_SIZE_8			832
	#define CELL_SIZE_9			1088
	#define CELL_SIZE_10			1344
	#define CELL_SIZE_11			1760
	#define CELL_SIZE_12			2176
	#define CELL_SIZE_13			2848
	#define CELL_SIZE_14			3520
	#define CELL_SIZE_15			4608
	#define CELL_SIZE_16			5152
	#define CELL_SIZE_17			5696
	#define CELL_SIZE_18 		8164
	#define CELL_SIZE_19 		13068
	#define CELL_SIZE_20 		16340
	#define CELL_SIZE_21 		21796
	#define MAX_CELL_SIZE		CELL_SIZE_21
	
	#define NUM_BUF_ALLOCATORS	22

	/****************************************************************************
	Desc: Global data
	****************************************************************************/

	#ifndef ALLOCATE_SYS_DATA
		extern F_FileSystem *	gv_pFileSystem;
		extern F_ThreadMgr *		gv_pThreadMgr;
	#else
		F_FileSystem *				gv_pFileSystem;
		F_ThreadMgr *				gv_pThreadMgr;
	#endif
	
	FINLINE RCODE MapPlatformError(
		FLMINT		iError,
		RCODE			defaultRc);

	#define F_MULTI_FHDL_LIST_SIZE						8
	#define F_MULTI_FHDL_DEFAULT_MAX_FILE_SIZE		((FLMUINT)0xFFFFFFFF)
	
	typedef struct
	{
		IF_FileHdl *	pFileHdl;
		FLMUINT			uiFileNum;
		FLMBOOL			bDirty;
	} FH_INFO;

	typedef struct xmlChar
	{
		FLMBYTE		ucFlags;
	} XMLCHAR;
	
	#define FLM_MAX_KEY_SIZE		1024
	
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

	typedef struct FlmBTreeBlkHdr
	{
		F_BLK_HDR	stdBlkHdr;							// Standard block header
		FLMUINT16	ui16BtreeId;						// BH_LOG_FILE_NUM
		FLMUINT16	ui16NumKeys;						// Number of keys
		FLMUINT8		ui8BlkLevel;						// BH_LEVEL
			#define BH_MAX_LEVELS		8 				// Max allowable b-tree levels
			#define MAX_LEVELS			BH_MAX_LEVELS
		FLMUINT8		ui8BTreeFlags;						// Flags for BTree
			#define BLK_IS_ROOT			0x01
			#define BLK_IS_INDEX			0x02
		FLMUINT16	ui16HeapSize;						// Contiguous available space
	#define F_BTREE_BLK_HDR_stdBlkHdr_OFFSET					0
	#define F_BTREE_BLK_HDR_ui16BtreeId_OFFSET				32
	#define F_BTREE_BLK_HDR_ui16NumKeys_OFFSET				34
	#define F_BTREE_BLK_HDR_ui8BlkLevel_OFFSET				36
	#define F_BTREE_BLK_HDR_ui8BTreeFlags_OFFSET				37
	#define F_BTREE_BLK_HDR_ui16HeapSize_OFFSET				38
	} F_BTREE_BLK_HDR;

	enum BTREE_ERR_TYPE
	{
		NO_ERR = 0,  // FYI: Visual Studio already defines NOERROR
		BT_HEADER,
		KEY_ORDER,
		DUPLICATE_KEYS,
		INFINITY_MARKER,
		CHILD_BLOCK_ADDRESS,
		SCA_GET_BLOCK_FAILED,
		MISSING_OVERALL_DATA_LENGTH,
		NOT_DATA_ONLY_BLOCK,
		BAD_DO_BLOCK_LENGTHS,
		BAD_COUNTS,
		CATASTROPHIC_FAILURE = 999
	};
	
	typedef struct
	{
		FLMUINT			uiKeyCnt;
		FLMUINT			uiFirstKeyCnt;
		FLMUINT			uiBlkCnt;
		FLMUINT			uiBytesUsed;
		FLMUINT			uiDOBlkCnt;
		FLMUINT			uiDOBytesUsed;
	} BTREE_LEVEL_STATS;
	
	typedef struct
	{
		FLMUINT				uiBlkAddr;
		FLMUINT				uiBlockSize;
		FLMUINT				uiBlocksChecked;
		FLMUINT				uiAvgFreeSpace;
		FLMUINT				uiLevels;
		FLMUINT				uiNumKeys;
		FLMUINT64			ui64FreeSpace;
		BTREE_LEVEL_STATS	LevelStats[ BH_MAX_LEVELS];
		char					szMsg[ 64];
		BTREE_ERR_TYPE		type;
	}  BTREE_ERR_STRUCT;
	
	typedef struct
	{
		FLMUINT		uiParentLevel;
		FLMUINT		uiParentKeyLen;
		FLMUINT		uiParentChildBlkAddr;
		FLMUINT		uiNewKeyLen;
		FLMUINT		uiChildBlkAddr;
		FLMUINT		uiCounts;
		void *		pPrev;
		FLMBYTE		pucParentKey[ FLM_MAX_KEY_SIZE];
		FLMBYTE		pucNewKey[ FLM_MAX_KEY_SIZE];
	} BTREE_REPLACE_STRUCT;

	typedef struct
	{
		F_BTREE_BLK_HDR *			pBlkHdr;
		IF_Block *					pBlock;
		const FLMBYTE *			pucKeyBuf;
		FLMUINT						uiKeyBufSize;
		FLMUINT						uiKeyLen;
		FLMUINT						uiCurOffset;
		FLMUINT						uiLevel;
		FLMUINT16 *					pui16OffsetArray;
		FLMUINT32					ui32BlkAddr;
	} F_BTSK;

	typedef enum
	{
		ELM_INSERT_DO,
		ELM_INSERT,
		ELM_REPLACE_DO,
		ELM_REPLACE,
		ELM_REMOVE,
		ELM_BLK_MERGE,
		ELM_DONE
	} F_ELM_UPD_ACTION;

	// Represent the maximum size for data & key before needing two bytes to
	// store the length.
	
	#define ONE_BYTE_SIZE				0xFF
	
	// Flag definitions - BT_LEAF_DATA
	
	#define BTE_LEAF_DATA_OVHD			7		// Offset (2) Flags (1) OA Data (4)
	
	#define BTE_FLAG						0		// Offset to the FLAGS field
	#define BTE_FLAG_LAST_ELEMENT		0x04
	#define BTE_FLAG_FIRST_ELEMENT	0x08
	#define BTE_FLAG_DATA_BLOCK		0x10	// Data is stored in a Data-only Block
	#define BTE_FLAG_OA_DATA_LEN		0x20	// Overall data length
	#define BTE_FLAG_DATA_LEN			0x40
	#define BTE_FLAG_KEY_LEN			0x80
	
	// BT_LEAF (no data)
	
	#define BTE_LEAF_OVHD				4		// Offset (2) KeyLen (2)
	#define BTE_KEY_LEN					0
	#define BTE_KEY_START				2
	
	// BT_NON_LEAF_DATA
	
	#define BTE_NON_LEAF_OVHD			8		// Offset (2) Child Blk Addr (4) KeyLen (2)
	#define BTE_NL_CHILD_BLOCK_ADDR	0
	#define BTE_NL_KEY_LEN				4
	#define BTE_NL_KEY_START			6
	
	// BT_NON_LEAF_COUNTS
	
	#define BTE_NON_LEAF_COUNTS_OVHD	12		// Offset (2) Child Blk Addr (4) Counts (4) KeyLen (2)
	#define BTE_NLC_CHILD_BLOCK_ADDR	0
	#define BTE_NLC_COUNTS				4
	#define BTE_NLC_KEY_LEN				8
	#define BTE_NLC_KEY_START			10
	
	// Low water mark for coalescing blocks (as a percentage)
	
	#define BT_LOW_WATER_MARK			65
	
	FINLINE FLMBOOL bteKeyLenFlag( 
		FLMBYTE *			pucEntry)
	{
		return( (pucEntry[ BTE_FLAG] & BTE_FLAG_KEY_LEN) ? TRUE : FALSE);
	}
	
	FINLINE FLMBOOL bteDataLenFlag( 
		FLMBYTE *			pucEntry)
	{
		return( (pucEntry[ BTE_FLAG] & BTE_FLAG_DATA_LEN) ? TRUE : FALSE);
	}
	
	FINLINE FLMBOOL bteOADataLenFlag( 
		FLMBYTE *			pucEntry)
	{
		return( (pucEntry[ BTE_FLAG] & BTE_FLAG_OA_DATA_LEN) ? TRUE : FALSE);
	}
	
	FINLINE FLMBOOL bteDataBlockFlag( 
		FLMBYTE *			pucEntry)
	{
		return( (pucEntry[ BTE_FLAG] & BTE_FLAG_DATA_BLOCK) ? TRUE : FALSE);
	}
	
	FINLINE FLMBOOL bteFirstElementFlag(
		FLMBYTE *			pucEntry)
	{
		return( (pucEntry[ BTE_FLAG] & BTE_FLAG_FIRST_ELEMENT) ? TRUE : FALSE);
	}
	
	FINLINE FLMBOOL bteLastElementFlag(
		FLMBYTE *			pucEntry)
	{
		return( (pucEntry[ BTE_FLAG] & BTE_FLAG_LAST_ELEMENT) ? TRUE : FALSE);
	}
	
	FINLINE FLMUINT32 bteGetBlkAddr(
		const FLMBYTE *	pucEntry)
	{
		return( FB2UD( pucEntry));
	}
	
	FINLINE void bteSetEntryOffset(
		FLMUINT16 *			pui16OffsetArray,
		FLMUINT				uiOffsetIndex,
		FLMUINT				ui16Offset)
	{
		UW2FBA( ui16Offset, (FLMBYTE *)&pui16OffsetArray[ uiOffsetIndex]);
	}
	
	FINLINE FLMUINT16 bteGetEntryOffset( 
		const FLMUINT16 *	pui16OffsetArray,
		FLMUINT				uiOffsetIndex)
	{
		return( FB2UW( (FLMBYTE *)&pui16OffsetArray[ uiOffsetIndex]));
	}
	
	/****************************************************************************
	Desc:		Internal return code macros
	****************************************************************************/
	#ifdef FLM_DEBUG
		RCODE	flmMakeErr(
			RCODE				rc,
			const char *	pszFile,
			int				iLine,
			FLMBOOL			bAssert);
			
		#define RC_SET( rc) \
			flmMakeErr( rc, __FILE__, __LINE__, FALSE)
			
		#define RC_SET_AND_ASSERT( rc) \
			flmMakeErr( rc, __FILE__, __LINE__, TRUE)
			
		#define RC_UNEXPECTED_ASSERT( rc) \
			flmMakeErr( rc, __FILE__, __LINE__, TRUE)
	#else
		#define RC_SET( rc)							(rc)
		#define RC_SET_AND_ASSERT( rc)			(rc)
		#define RC_UNEXPECTED_ASSERT( rc)
	#endif

	#define F_SEM_WAITFOREVER			(0xFFFFFFFF)

	/****************************************************************************
	Desc:		NLM
	****************************************************************************/
	#if defined( FLM_NLM)
	
		#if defined( FLM_WATCOM_NLM)
			#pragma warning 007 9
	
			// Disable "Warning! W549: col(XX) 'sizeof' operand contains
			// compiler generated information"
			
			#pragma warning 549 9
			
			// Disable "Warning! W656: col(1) define this function inside its class
			// definition (may improve code quality)"
			
			#pragma warning 656 9
		#endif

		#include <stdio.h>
		#include <stdlib.h>
		#include <string.h>
		#include <pthread.h>
		#include <unistd.h>
		#include <errno.h>
		#include <library.h>
		#include <fcntl.h>
		#include <sys/stat.h>
		#include <sys/unistd.h>
		#include <glob.h>

		// The typedef for va_list in stdarg.h do not function properly when
		// a va_list is passed down multiple layers as a pointer (va_list *).
		// Therefore, the following definitions/typedefs were taken from a
		// "fixed" version of stdarg.h implemented by DS.

		// typedef unsigned long f_va_list;
		
		#define f_argsize(x) \
			((sizeof(x)+sizeof(int)-1) & ~(sizeof(int)-1))
			
		#define f_va_start(ap, parmN) \
			((void)((ap) = (unsigned long)&(parmN) + f_argsize(parmN)))
			
		#define f_va_arg(ap, type) \
			(*(type *)(((ap) += f_argsize(type)) - (f_argsize(type))))
			
		#define f_va_end(ap) ((void)0)
		#define FSTATIC

		#ifndef _SIZE_T
			#define _SIZE_T
			typedef unsigned int size_t;
		#endif

		#ifndef _WCHAR_T
			#define _WCHAR_T
			typedef unsigned short wchar_t;
		#endif

		#ifndef WCHAR
			#define WCHAR wchar_t
		#endif

		#ifndef LONG
			#define LONG unsigned long	
		#endif

		#ifndef BYTE
			#define BYTE unsigned char
		#endif

		#ifndef UINT
			#define UINT	unsigned int
		#endif
	
		typedef void * 			SEMAPHORE;
		typedef unsigned long 	ERROR;

		extern "C" SEMAPHORE kSemaphoreAlloc(
			BYTE *		pSemaName,
			UINT			SemaCount);
			
		extern "C" ERROR kSemaphoreFree(
			SEMAPHORE	SemaHandle);
			
		extern "C" ERROR kSemaphoreWait(
			SEMAPHORE	SemaHandle);
			
		extern "C" ERROR kSemaphoreTimedWait(
			SEMAPHORE	SemaHandle,
			UINT			MilliSecondTimeOut);
			
		extern "C" ERROR kSemaphoreSignal(
			SEMAPHORE	SemaHandle);
			
		extern "C" UINT kSemaphoreExamineCount(
			SEMAPHORE	SemaHandle);
	
		extern "C" MUTEX kMutexAlloc(
			BYTE *		MutexName);
			
		extern "C" ERROR kMutexFree(
			MUTEX			MutexHandle);
			
		extern "C" ERROR kMutexLock(
			MUTEX			MutexHandle);
			
		extern "C" ERROR kMutexUnlock(
			MUTEX			MutexHandle);
	
		// External Netware Symbols
		
		extern "C" FLMUINT f_getNLMHandle( void);
		
		extern "C" RCODE f_netwareStartup( void);

		extern "C" void f_netwareShutdown( void);
			
//		#define f_stricmp(str1,str2) \
//			strcasecmp((char *)(str1),(char *)(str2))
//
//		#define f_strnicmp(str1,str2,size_t) \
//			strncasecmp((char *)(str1),(char *)(str2),size_t)
//
//		#define f_memmove( dest, src, len) \
//			memmove( (void*)(dest), (void*)(src), len)
//
//		#define f_memset( src, chr, size) \
//			memset((void  *)(src),(chr),(size_t)(size))
//
//		#define f_memcmp( str1, str2, length) \
//			memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))
//
//		#define f_strcat( dest, src) \
//			strcat( (char*)(dest), (char*)(src))
//
//		#define f_strchr( str, value) \
//			strchr( (char*)str, (int)value)
//
//		#define f_strcmp( str1, str2) \
//			strcmp( (char*)(str1), (char*)(str2))
//
//		#define f_strcpy( dest, src) \
//			strcpy( (char*)(dest), (char*)(src))
//
//		#define f_strncpy( dest, src, length) \
//			strncpy( (char*)(dest), (char*)(src), (size_t)(length))
//
//		#define f_strlen( str) \
//			strlen( (char*)(str))
//
//		#define f_strncmp( str1, str2, size) \
//			strncmp( (char*)(str1), (char*)(str2), (size_t)(size))
//
//		#define f_strrchr( str, value ) \
//			strrchr( (char*)(str), (int)value)
//
//		#define f_strstr( str1, str2) \
//			(char *)strstr( (char*)(str1), (char*)(str2))
//
//		#define f_strncat( str1, str2, n) \
//			strncat( (char *)(str1), (char *)(str2), n)
//
//		#define f_strupr( str) \
//			strupr( (char *)(str))

	#endif

	/****************************************************************************
	Desc:	WIN
	****************************************************************************/
	#if defined( FLM_WIN)

		#ifndef WIN32_LEAN_AND_MEAN
			#define WIN32_LEAN_AND_MEAN
		#endif
	
		#ifndef WIN32_EXTRA_LEAN
			#define WIN32_EXTRA_LEAN
		#endif
	
		// Enable critical section and spin count API to be visible in header
		// file.
	
		#define _WIN32_WINNT	0x0403
	
		#pragma pack( push, enter_windows, 8)
			#include <windows.h>
			#include <time.h>
			#include <stdlib.h>
			#include <stddef.h>
			#include <rpc.h>
			#include <process.h>
			#include <winsock.h>
			#include <imagehlp.h>
			#include <malloc.h>
		#pragma pack( pop, enter_windows)
		
		// Conversion from XXX to YYY, possible loss of data
		#pragma warning( disable : 4244) 
	
		// Local variable XXX may be used without having been initialized
		#pragma warning( disable : 4701)
	
		// Function XXX not inlined
		#pragma warning( disable : 4710) 
		
		#define FSTATIC			static

		#define ENDLINE			ENDLINE_CRLF
//		#define f_va_list			va_list
		#define f_va_start		va_start
		#define f_va_arg			va_arg
		#define f_va_end			va_end

//		#define f_stricmp( str1, str2) \
//			_stricmp((char *)(str1), (char *)(str2))
//
//		#define f_strnicmp( str1, str2, size) \
//			_strnicmp((char *)(str1), (char *)(str2),(size_t)(size))
//
//		#define f_memmove( dest, src, length) \
//			memmove((void  *)(dest), (void  *)(src),(size_t)(length))
//
//		#define f_memset( src, chr, size) \
//			memset((void  *)(src),(chr),(size_t)(size))
//
//		#define f_memcmp( str1, str2, length) \
//			memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))
//
//		#define f_strcat( dest, src) \
//			strcat( (char*)(dest), (char*)(src))
//
//		#define f_strchr( str, value) \
//			strchr( (char*)str, (int)value)
//
//		#define f_strcmp( str1, str2) \
//			strcmp( (char*)(str1), (char*)(str2))
//
//		#define f_strcpy( dest, src) \
//			strcpy( (char*)(dest), (char*)(src))
//
//		#define f_strncpy( dest, src, length) \
//			strncpy( (char*)(dest), (char*)(src), (size_t)(length))
//
//		#define f_strlen( str) \
//			strlen( (char*)(str))
//
//		#define f_strncmp( str1, str2, size) \
//			strncmp( (char*)(str1), (char*)(str2), (size_t)(size))
//
//		#define f_strrchr( str, value ) \
//			strrchr( (char*)(str), (int)value)
//
//		#define f_strstr( str1, str2) \
//			(char *)strstr( (char*)(str1), (char*)(str2))
//
//		#define f_strncat( str1, str2, n) \
//			strncat( (char *)(str1), (char *)(str2), n)
//
//		#define f_strupr( str) \
//			_strupr( (char *)(str))

	#endif

	/****************************************************************************
	Desc:		UNIX
	****************************************************************************/
	#if defined( FLM_UNIX)

		#define FSTATIC		static

		#ifdef HAVE_CONFIG_H
			#include "config.h"
		#endif

		#ifdef FLM_AIX
			#ifndef _LARGE_FILES
				#define _LARGE_FILES
			#endif
			#include <stdio.h>
			#include <fcntl.h>
			#include <sys/vminfo.h>
		#endif

		#include <assert.h>
		#include <pthread.h>
		#include <errno.h>
		#include <glob.h>
		#include <limits.h>
		#include <netdb.h>
		#include <sys/types.h>
		#include <netinet/in.h>
		#include <arpa/nameser.h>
		#include <resolv.h>
		#include <stdarg.h>
		#include <stdlib.h>
		#include <string.h>
		#include <strings.h>
		#include <time.h>
		#include <unistd.h>
		#include <utime.h>
		#include <aio.h>
		#include <arpa/inet.h>
		#include <netinet/tcp.h>
		#include <sys/mman.h>
		#include <sys/resource.h>
		#include <sys/socket.h>
		#include <sys/stat.h>
		#include <sys/time.h>

		#ifdef FLM_AIX
			#include <sys/atomic_op.h>
		#endif

		#ifdef FLM_OSX
			#include <libkern/OSAtomic.h>
		#endif

//		#define f_stricmp(str1,str2) \
//			strcasecmp((char *)(str1),(char *)(str2))
//
//		#define f_strnicmp(str1,str2,size_t) \
//			strncasecmp((char *)(str1),(char *)(str2),size_t)
//
//		#define f_memmove( dest, src, len) \
//			memmove( (void*)(dest), (void*)(src), len)
//
//		#define f_memset( src, chr, size) \
//			memset((void  *)(src),(chr),(size_t)(size))
//
//		#define f_memcmp( str1, str2, length) \
//			memcmp((void  *)(str1), (void  *)(str2),(size_t)(length))
//
//		#define f_strcat( dest, src) \
//			strcat( (char*)(dest), (char*)(src))
//
//		#define f_strchr( str, value) \
//			strchr( (char*)str, (int)value)
//
//		#define f_strcmp( str1, str2) \
//			strcmp( (char*)(str1), (char*)(str2))
//
//		#define f_strcpy( dest, src) \
//			strcpy( (char*)(dest), (char*)(src))
//
//		#define f_strncpy( dest, src, length) \
//			strncpy( (char*)(dest), (char*)(src), (size_t)(length))
//
//		#define f_strlen( str) \
//			strlen( (char*)(str))
//
//		#define f_strncmp( str1, str2, size) \
//			strncmp( (char*)(str1), (char*)(str2), (size_t)(size))
//
//		#define f_strrchr( str, value ) \
//			strrchr( (char*)(str), (int)value)
//
//		#define f_strstr( str1, str2) \
//			(char *)strstr( (char*)(str1), (char*)(str2))
//
//		#define f_strncat( str1, str2, n) \
//			strncat( (char *)(str1), (char *)(str2), n)
//
//		#define f_strupr( str) \
//			strupr( (char *)(str))

//		#define f_va_list				va_list
		#define f_va_start			va_start
		#define f_va_arg				va_arg
		#define f_va_end				va_end

//		typedef pthread_mutex_t *	F_MUTEX;
//		typedef F_MUTEX *				F_MUTEX_p;
//		#define F_MUTEX_NULL			NULL
		
		typedef struct
		{
			pthread_mutex_t lock;
			pthread_cond_t  cond;
			int             count;
		} sema_t;
	
		typedef sema_t *				F_SEM;
		typedef F_SEM *				F_SEM_p;
		#define F_SEM_NULL			NULL

		typedef int						SOCKET;
		#define INVALID_SOCKET		-1
	
	#endif

	/****************************************************************************
	Desc: Cross-platform inline functions
	****************************************************************************/
//	FINLINE void f_memcpy(
//		void *			pvDest,
//		const void *	pvSrc,
//		FLMSIZET			iSize)
//	{
//		if( iSize == 1)
//		{
//			*((FLMBYTE *)pvDest) = *((FLMBYTE *)pvSrc);
//		}
//		else
//		{
//			(void)memcpy( pvDest, pvSrc, iSize);
//		}
//	}

	#if defined( __va_copy)
		#define  f_va_copy(to, from) __va_copy(to, from)
	#else
		#define f_va_copy(to, from)  ((to) = (from))
	#endif

	typedef struct IniLine
	{
		char *				pszParamName;
		char *				pszParamValue;	
		char *				pszComment;
		struct IniLine *	pPrev;
		struct IniLine *	pNext;
	} INI_LINE;

	/****************************************************************************
	Desc:		Internal base class
	****************************************************************************/
	class F_OSBase
	{
	public:

		F_OSBase()
		{ 
			m_refCnt = 1;	
		}

		virtual ~F_OSBase()
		{
		}

		FINLINE FLMUINT getRefCount( void)
		{
			return( m_refCnt);
		}

		void * operator new(
			FLMSIZET			uiSize);

	#ifdef FLM_DEBUG
		void * operator new(
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	#endif

		void operator delete(
			void *			ptr);

		void operator delete[](
			void *			ptr);

	#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
		void operator delete(
			void *			ptr,
			const char *,	// file
			int);				// line
	#endif

	#if defined( FLM_DEBUG) && !defined( FLM_WATCOM_NLM) && !defined( FLM_SOLARIS)
		void operator delete[](
			void *			ptr,
			const char *,	// file
			int);				// line
	#endif

		virtual FINLINE FLMINT FLMAPI AddRef( void)
		{
			return( ++m_refCnt);
		}

		virtual FINLINE FLMINT FLMAPI Release( void)
		{
			FLMINT		iRefCnt = --m_refCnt;

			if( !iRefCnt)
			{
				delete this;
			}

			return( iRefCnt);
		}

	protected:

		FLMATOMIC		m_refCnt;
	};

	/****************************************************************************
	Desc:		Base class
	****************************************************************************/
	class F_Base
	{
	public:
	
		F_Base()
		{
		}
	
		virtual ~F_Base()
		{
		}
	
		void * operator new(
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	
		void * operator new[](
			FLMSIZET			uiSize,
			const char *	pszFile,
			int				iLine);
	
		void operator delete(
			void *			ptr,
			const char *	file,
			int				line);
	
		void operator delete[](
			void *			ptr,
			const char *	file,
			int				line);
	};

	/****************************************************************************
	Desc:	This class is used to do pool memory allocations.
	****************************************************************************/
	class F_Pool : public IF_Pool, public F_Base
	{
	public:

		typedef struct PoolMemoryBlock
		{
			PoolMemoryBlock *		pPrevBlock;
			FLMUINT					uiBlockSize;
			FLMUINT					uiFreeOffset;
			FLMUINT					uiFreeSize;
		} MBLK;

		typedef struct
		{
			FLMUINT	uiAllocBytes;
			FLMUINT	uiCount;
		} POOL_STATS;

		F_Pool()
		{
			m_uiBytesAllocated = 0;
			m_pLastBlock = NULL;
			m_pPoolStats = NULL;
			m_uiBlockSize = 0;
		}

		virtual ~F_Pool();

		FINLINE void FLMAPI poolInit(
			FLMUINT			uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
		}

		void smartPoolInit(
			POOL_STATS *	pPoolStats);

		RCODE FLMAPI poolAlloc(
			FLMUINT			uiSize,
			void **			ppvPtr);

		RCODE FLMAPI poolCalloc(
  			FLMUINT			uiSize,
			void **			ppvPtr);

		void FLMAPI poolFree( void);

		void FLMAPI poolReset(
			void *			pvMark,
			FLMBOOL			bReduceFirstBlock = FALSE);

		FINLINE void * FLMAPI poolMark( void)
		{
			return (void *)(m_pLastBlock
								 ? (FLMBYTE *)m_pLastBlock + m_pLastBlock->uiFreeOffset
								 : NULL);
		}

		FINLINE FLMUINT FLMAPI getBlockSize( void)
		{
			return( m_uiBlockSize);
		}

		FINLINE FLMUINT FLMAPI getBytesAllocated( void)
		{
			return( m_uiBytesAllocated);
		}

	private:

		FINLINE void updateSmartPoolStats( void)
		{
			if (m_uiBytesAllocated)
			{
				if( (m_pPoolStats->uiAllocBytes + m_uiBytesAllocated) >= 0xFFFF0000)
				{
					m_pPoolStats->uiAllocBytes =
						(m_pPoolStats->uiAllocBytes / m_pPoolStats->uiCount) * 100;
					m_pPoolStats->uiCount = 100;
				}
				else
				{
					m_pPoolStats->uiAllocBytes += m_uiBytesAllocated;
					m_pPoolStats->uiCount++;
				}
				m_uiBytesAllocated = 0;
			}
		}

		FINLINE void setInitialSmartPoolBlkSize( void)
		{
			// Determine starting block size:
			// 1) average of bytes allocated / # of frees/resets (average size needed)
			// 2) add 10% - to minimize extra allocs 

			m_uiBlockSize = (m_pPoolStats->uiAllocBytes / m_pPoolStats->uiCount);
			m_uiBlockSize += (m_uiBlockSize / 10);

			if (m_uiBlockSize < 512)
			{
				m_uiBlockSize = 512;
			}
		}

		void freeToMark(
			void *		pvMark);

		PoolMemoryBlock *		m_pLastBlock;
		FLMUINT					m_uiBlockSize;
		FLMUINT					m_uiBytesAllocated;
		POOL_STATS *			m_pPoolStats;
	};

	/****************************************************************************
									FLAIM's Assert Layer

	This section contains prototypes and macros for FLAIM's assert. This layer
	enables FLAIM to redirect assert calls.
	****************************************************************************/

	#ifdef FLM_DEBUG

		#ifdef FLM_DBG_LOG
			void flmDbgLogFlush( void);
		#endif

		#if defined( FLM_WIN)
			#ifdef FLM_DBG_LOG
				#define flmAssert( exp) \
					(void)( (exp) || (flmDbgLogFlush(), DebugBreak(), 0))
			#else
				#define flmAssert( exp) \
					(void)( (exp) || (DebugBreak(), 0))
			#endif

		#elif defined( FLM_NLM)
			extern "C" void EnterDebugger(void);

			#ifdef FLM_DBG_LOG
				#define flmAssert( exp)	\
					(void)( (exp) || (flmDbgLogFlush(), EnterDebugger(), 0))
			#else
				#define flmAssert( exp) \
					(void)( (exp) || ( EnterDebugger(), 0))
			#endif

		#elif defined( FLM_UNIX)
			#ifdef FLM_DBG_LOG
				#define flmAssert( exp) \
					(void)( (exp) || (flmDbgLogFlush(), assert(0), 0))
			#else
				#define flmAssert( exp)	\
					(void)( (exp) || (assert(0), 0))
			#endif

		#else
			#define flmAssert( exp)
		#endif

	#else
		#define flmAssert( exp)
	#endif

	FLMUINT f_breakpoint(
		FLMUINT		uiBreakFlag);

	/****************************************************************************
										Random Numbers
	****************************************************************************/

	#define MAX_RANDOM  2147483646L

	class F_RandomGenerator : public IF_RandomGenerator, public F_Base
	{
	public:

		void FLMAPI randomize( void);

		void FLMAPI setSeed(
			FLMINT32		i32seed);

		FLMINT32 FLMAPI getInt32( void);
			
		FLMINT32 FLMAPI getInt32(
			FLMINT32 	i32Low,
			FLMINT32 	i32High);

		FLMBOOL FLMAPI getBoolean( void);

		FLMINT32 FLMAPI getSeed( void)
		{
			return( m_i32Seed);
		}

	private:

		FLMINT32			m_i32Seed;
	};

//	/**********************************************************************
//	Desc: Atomic Increment, Decrement, Exchange
//	Note:	Some of this code is derived from the Ximian source code contained
//			in that Mono project's atomic.h file. 
//	**********************************************************************/
//	#ifndef FLM_HAVE_ATOMICS
//		#define FLM_HAVE_ATOMICS
//	#endif
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	#if defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
//	extern "C" FLMINT32 sparc_atomic_add_32(
//		volatile FLMINT32 *		piTarget,
//		FLMINT32						iDelta);
//	#endif
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	#if defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
//	extern "C" FLMINT32 sparc_atomic_xchg_32(
//		volatile FLMINT32 *		piTarget,
//		FLMINT32						iNewValue);
//	#endif
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	#if defined( FLM_AIX)
//	FINLINE int aix_atomic_add(
//		volatile int *			piTarget,
//		int 						iDelta)
//	{
//		return( fetch_and_add( (int *)piTarget, iDelta) + iDelta);
//	}
//	#endif
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	FINLINE FLMINT32 _flmAtomicInc(
//		FLMATOMIC *			piTarget)
//	{
//		#if defined( FLM_NLM)
//		{
//			return( (FLMINT32)nlm_AtomicIncrement( (volatile LONG *)piTarget));
//		}
//		#elif defined( FLM_WIN)
//		{
//			return( (FLMINT32)InterlockedIncrement( (volatile LONG *)piTarget));
//		}
//		#elif defined( FLM_AIX)
//		{
//			return( (FLMINT32)aix_atomic_add( piTarget, 1));
//		}
//		#elif defined( FLM_OSX)
//		{
//			return( (FLMINT32)OSAtomicIncrement32( (int32_t *)piTarget));
//		}
//		#elif defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
//		{
//			return( sparc_atomic_add_32( piTarget, 1));
//		}
//		#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
//		{
//			FLMINT32 			i32Tmp;
//			
//			__asm__ __volatile__ (
//							"lock;"
//							"xaddl %0, %1"
//								: "=r" (i32Tmp), "=m" (*piTarget)
//								: "0" (1), "m" (*piTarget));
//		
//			return( i32Tmp + 1);
//		}
//		#else
//			#ifdef FLM_HAVE_ATOMICS
//				#undef FLM_HAVE_ATOMICS
//			#endif
//	
//			F_UNREFERENCED_PARM( piTarget);	
//	
//			flmAssert( 0);
//			return( 0);
//		#endif
//	}
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	FINLINE FLMINT32 _flmAtomicDec(
//		FLMATOMIC *			piTarget)
//	{
//		#if defined( FLM_NLM)
//		{
//			return( (FLMINT32)nlm_AtomicDecrement( (volatile LONG *)piTarget));
//		}
//		#elif defined( FLM_WIN)
//		{
//			return( (FLMINT32)InterlockedDecrement( (volatile LONG *)piTarget));
//		}
//		#elif defined( FLM_AIX)
//		{
//			return( (FLMINT32)aix_atomic_add( piTarget, -1));
//		}
//		#elif defined( FLM_OSX)
//		{
//			return( (FLMINT32)OSAtomicDecrement32( (int32_t *)piTarget));
//		}
//		#elif defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
//		{
//			return( sparc_atomic_add_32( piTarget, -1));
//		}
//		#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
//		{
//			FLMINT32				i32Tmp;
//			
//			__asm__ __volatile__ (
//							"lock;" 
//							"xaddl %0, %1"
//								: "=r" (i32Tmp), "=m" (*piTarget)
//								: "0" (-1), "m" (*piTarget));
//		
//			return( i32Tmp - 1);
//		}
//		#else
//			#ifdef FLM_HAVE_ATOMICS
//				#undef FLM_HAVE_ATOMICS
//			#endif
//	
//			F_UNREFERENCED_PARM( piTarget);
//				
//			flmAssert( 0);
//			return( 0);
//		#endif
//	}
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	FINLINE FLMINT32 _flmAtomicExchange(
//		FLMATOMIC *			piTarget,
//		FLMINT32				i32NewVal)
//	{
//		#if defined( FLM_NLM)
//		{
//			return( (FLMINT32)nlm_AtomicExchange( 
//				(volatile LONG *)piTarget, i32NewVal));
//		}
//		#elif defined( FLM_WIN)
//		{
//			return( (FLMINT32)InterlockedExchange( (volatile LONG *)piTarget,
//				i32NewVal));
//		}
//		#elif defined( FLM_AIX)
//		{
//			int		iOldVal;
//			
//			for( ;;)
//			{ 
//				iOldVal = (int)*piTarget;
//				
//				if( compare_and_swap( (int *)piTarget, &iOldVal, i32NewVal))
//				{
//					break;
//				}
//			}
//			
//			return( (FLMINT32)iOldVal);
//		}
//		#elif defined( FLM_OSX)
//		{
//			int32_t		iOldVal;
//
//			for( ;;)
//			{
//				iOldVal = (int32_t)*piTarget;
//
//				if( OSAtomicCompareAndSwap32( iOldVal, i32NewVal, 
//						(int32_t *)piTarget))
//				{
//					break;
//				}
//			}
//			
//			return( (FLMINT32)iOldVal);
//		}
//		#elif defined( FLM_SOLARIS) && defined( FLM_SPARC) && !defined( FLM_GNUC)
//		{
//			return( sparc_atomic_xchg_32( piTarget, i32NewVal));
//		}
//		#elif (defined( __i386__) || defined( __x86_64__)) && defined( FLM_GNUC)
//		{
//			FLMINT32 			i32OldVal;
//			
//			__asm__ __volatile__ (
//							"1:	lock;"
//							"		cmpxchgl %2, %0;"
//							"		jne 1b"
//								: "=m" (*piTarget), "=a" (i32OldVal)
//								: "r" (i32NewVal), "m" (*piTarget), "a" (*piTarget));
//		
//			return( i32OldVal);
//		}
//		#else
//			#ifdef FLM_HAVE_ATOMICS
//				#undef FLM_HAVE_ATOMICS
//			#endif
//	
//			F_UNREFERENCED_PARM( piTarget);
//			F_UNREFERENCED_PARM( i32NewVal);
//	
//			flmAssert( 0);
//			return( 0);
//		#endif
//	}
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	FINLINE FLMINT32 flmAtomicInc(
//		FLMATOMIC *		piTarget,
//		F_MUTEX			hMutex = F_MUTEX_NULL,
//		FLMBOOL			bMutexAlreadyLocked = FALSE)
//	{
//		#ifdef FLM_HAVE_ATOMICS
//		{
//			F_UNREFERENCED_PARM( bMutexAlreadyLocked);
//			F_UNREFERENCED_PARM( hMutex);
//			
//			return( _flmAtomicInc( piTarget));
//		}
//		#else
//		{
//			FLMINT32		i32NewVal;
//			
//			flmAssert( hMutex != F_MUTEX_NULL);
//	
//			if( !bMutexAlreadyLocked)
//			{
//				f_mutexLock( hMutex);
//			}
//			
//			i32NewVal = (FLMINT32)(++(*piTarget));
//			
//			if( !bMutexAlreadyLocked)
//			{
//				f_mutexUnlock( hMutex);
//			}
//			
//			return( i32NewVal);
//		}
//		#endif
//	}
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	FINLINE FLMINT32 flmAtomicDec(
//		FLMATOMIC *		piTarget,
//		F_MUTEX			hMutex = F_MUTEX_NULL,
//		FLMBOOL			bMutexAlreadyLocked = FALSE)
//	{
//		#ifdef FLM_HAVE_ATOMICS
//		{
//			F_UNREFERENCED_PARM( bMutexAlreadyLocked);
//			F_UNREFERENCED_PARM( hMutex);
//			
//			return( _flmAtomicDec( piTarget));
//		}
//		#else
//		{
//			FLMINT32		i32NewVal;
//			
//			flmAssert( hMutex != F_MUTEX_NULL);
//			
//			if( !bMutexAlreadyLocked)
//			{
//				f_mutexLock( hMutex);
//			}
//			
//			i32NewVal = (FLMINT32)(--(*piTarget));
//			
//			if( !bMutexAlreadyLocked)
//			{
//				f_mutexUnlock( hMutex);
//			}
//			
//			return( i32NewVal);
//		}
//		#endif
//	}
//	
//	/**********************************************************************
//	Desc:
//	**********************************************************************/
//	FINLINE FLMINT32 flmAtomicExchange(
//		FLMATOMIC *		piTarget,
//		FLMINT32			i32NewVal,
//		F_MUTEX			hMutex = F_MUTEX_NULL,
//		FLMBOOL			bMutexAlreadyLocked = FALSE)
//	{
//		#ifdef FLM_HAVE_ATOMICS
//		{
//			F_UNREFERENCED_PARM( bMutexAlreadyLocked);
//			F_UNREFERENCED_PARM( hMutex);
//			
//			return( _flmAtomicExchange( piTarget, i32NewVal));
//		}
//		#else
//		{
//			FLMINT32		i32OldVal;
//			
//			flmAssert( hMutex != F_MUTEX_NULL);
//			
//			if( !bMutexAlreadyLocked)
//			{
//				f_mutexLock( hMutex);
//			}
//			
//			i32OldVal = (FLMINT32)*piTarget;
//			*piTarget = i32NewVal;
//			
//			if( !bMutexAlreadyLocked)
//			{
//				f_mutexUnlock( hMutex);
//			}
//			
//			return( i32OldVal);
//		}
//		#endif
//	}

	/****************************************************************************
	Desc: Mutex and semaphore routines
	****************************************************************************/
//	#ifdef FLM_NLM
//		FINLINE RCODE f_mutexCreate(
//			F_MUTEX *	phMutex)
//		{
//			if( (*phMutex = (F_MUTEX)kMutexAlloc( (BYTE *)"NOVDB")) == F_MUTEX_NULL)
//			{
//				return RC_SET( NE_FLM_MEM);
//			}
//
//			return NE_FLM_OK;
//		}
//	
//		FINLINE void f_mutexDestroy(
//			F_MUTEX *	phMutex)
//		{
//			if (*phMutex != F_MUTEX_NULL)
//			{
//				if( kMutexFree( (MUTEX)(*phMutex)))
//				{
//					flmAssert( 0);
//				}
//				
//				*phMutex = F_MUTEX_NULL;
//			}
//		}
//	
//		FINLINE void f_mutexLock(
//			F_MUTEX		hMutex)
//		{
//			(void)kMutexLock( (MUTEX)hMutex);
//		}
//	
//		FINLINE void f_mutexUnlock(
//			F_MUTEX		hMutex)
//		{
//			(void)kMutexUnlock( (MUTEX)hMutex);
//		}
//	
//		FINLINE void f_assertMutexLocked(
//			F_MUTEX)
//		{
//		}
//
//		typedef SEMAPHORE				F_SEM;
//		typedef SEMAPHORE *			F_SEM_p;
//		#define F_SEM_NULL			0
//
//		FINLINE RCODE f_semCreate(
//			F_SEM *		phSem)
//		{
//			if( (*phSem = (F_SEM)kSemaphoreAlloc( (BYTE *)"NOVDB", 0)) == F_SEM_NULL)
//			{
//				return RC_SET( NE_FLM_MEM);
//			}
//	
//			return NE_FLM_OK;
//		}
//	
//		FINLINE void f_semDestroy(
//			F_SEM *		phSem)
//		{
//			if (*phSem != F_SEM_NULL)
//			{
//				(void)kSemaphoreFree( (SEMAPHORE)(*phSem));
//				*phSem = F_SEM_NULL;
//			}
//		}
//	
//		FINLINE RCODE f_semWait(
//			F_SEM			hSem,
//			FLMUINT		uiTimeout)
//		{
//			RCODE			rc = NE_FLM_OK;
//	
//			if( uiTimeout == F_SEM_WAITFOREVER)
//			{
//				if( kSemaphoreWait( (SEMAPHORE)hSem) != 0)
//				{
//					rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
//				}
//			}
//			else
//			{
//				if( kSemaphoreTimedWait( (SEMAPHORE)hSem, (UINT)uiTimeout) != 0)
//				{
//					rc = RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE);
//				}
//			}
//	
//			return( rc);
//		}
//	
//		FINLINE void f_semSignal(
//			F_SEM			hSem)
//		{
//			(void)kSemaphoreSignal( (SEMAPHORE)hSem);
//		}
//	
//	#elif defined( FLM_WIN)
//	
//		RCODE f_mutexCreate(
//			F_MUTEX *	phMutex);
//	
//		void f_mutexDestroy(
//			F_MUTEX *	phMutex);
//			
//		FINLINE void f_mutexLock(
//			F_MUTEX		hMutex)
//		{
//			while( flmAtomicExchange( 
//				&(((F_INTERLOCK *)hMutex)->locked), 1) != 0)
//			{
//		#ifdef FLM_DEBUG
//				flmAtomicInc( &(((F_INTERLOCK *)hMutex)->waitCount));
//		#endif
//				Sleep( 0);
//			}
//	
//		#ifdef FLM_DEBUG
//			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == 0);
//			((F_INTERLOCK *)hMutex)->uiThreadId = _threadid;
//			flmAtomicInc( &(((F_INTERLOCK *)hMutex)->lockedCount));
//		#endif
//		}
//		
//		FINLINE void f_mutexUnlock(
//			F_MUTEX		hMutex)
//		{
//			flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
//		#ifdef FLM_DEBUG
//			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
//			((F_INTERLOCK *)hMutex)->uiThreadId = 0;
//		#endif
//			flmAtomicExchange( &(((F_INTERLOCK *)hMutex)->locked), 0);
//		}
//	
//		FINLINE void f_assertMutexLocked(
//			F_MUTEX		hMutex)
//		{
//		#ifdef FLM_DEBUG
//			flmAssert( ((F_INTERLOCK *)hMutex)->locked == 1);
//			flmAssert( ((F_INTERLOCK *)hMutex)->uiThreadId == _threadid);
//		#else
//			F_UNREFERENCED_PARM( hMutex);
//		#endif
//		}
//		
//		FINLINE RCODE f_semCreate(
//			F_SEM *		phSem)
//		{
//			if( (*phSem = CreateSemaphore( (LPSECURITY_ATTRIBUTES)NULL,
//				0, 10000, NULL )) == NULL)
//			{
//				return( RC_SET( NE_FLM_COULD_NOT_CREATE_SEMAPHORE));
//			}
//	
//			return NE_FLM_OK;
//		}
//	
//		FINLINE void f_semDestroy(
//			F_SEM *		phSem)
//		{
//			if (*phSem != F_SEM_NULL)
//			{
//				CloseHandle( *phSem);
//				*phSem = F_SEM_NULL;
//			}
//		}
//	
//		FINLINE RCODE f_semWait(
//			F_SEM			hSem,
//			FLMUINT		uiTimeout)
//		{
//			if( WaitForSingleObject( hSem, uiTimeout ) == WAIT_OBJECT_0)
//			{
//				return( NE_FLM_OK);
//			}
//			else
//			{
//				return( RC_SET( NE_FLM_ERROR_WAITING_ON_SEMPAHORE));
//			}
//		}
//	
//		FINLINE void f_semSignal(
//			F_SEM			hSem)
//		{
//			(void)ReleaseSemaphore( hSem, 1, NULL);
//		}
//	#elif defined( FLM_UNIX)
//		RCODE f_mutexCreate(
//			F_MUTEX *	phMutex);
//	
//		void f_mutexDestroy(
//			F_MUTEX *	phMutex);
//		
//		FINLINE void f_mutexLock(
//			F_MUTEX		hMutex)
//		{
//			(void)pthread_mutex_lock( hMutex);
//		}
//	
//		FINLINE void f_mutexUnlock(
//			F_MUTEX		hMutex)
//		{
//			(void)pthread_mutex_unlock( hMutex);
//		}
//	
//		FINLINE void f_assertMutexLocked(
//			F_MUTEX)
//		{
//		}
//
//		int sema_signal(
//			sema_t *			sem);
//
//		void f_semDestroy(
//			F_SEM *	phSem);
//	
//		RCODE f_semCreate(
//			F_SEM *	phSem);
//	
//		RCODE f_semWait(
//			F_SEM		hSem,
//			FLMUINT	uiTimeout);
//	
//		FINLINE void f_semSignal(
//			F_SEM			hSem)
//		{
//			(void)sema_signal( hSem);
//		}
//
//	#endif

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IStream : public IF_IStream, public F_Base
	{
	public:

		F_IStream();

		virtual ~F_IStream();
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_OStream : public IF_OStream, public F_Base
	{
	public:

		F_OStream();

		virtual ~F_OStream();

	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_PosIStream : public IF_PosIStream, public F_Base
	{
	public:

		F_PosIStream();

		virtual ~F_PosIStream();

	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferIStream : public F_PosIStream
	{
	public:

		F_BufferIStream()
		{
			m_pucBuffer = NULL;
			m_uiBufferLen = 0;
			m_uiOffset = 0;
			m_bAllocatedBuffer = FALSE;
			m_bIsOpen = FALSE;
		}

		virtual ~F_BufferIStream();

		RCODE FLMAPI open(
			const FLMBYTE *	pucBuffer,
			FLMUINT				uiLength,
			FLMBYTE **			ppucAllocatedBuffer = NULL);

		FINLINE FLMUINT64 FLMAPI totalSize( void)
		{
			flmAssert( m_bIsOpen);
			return( m_uiBufferLen);
		}

		FINLINE FLMUINT64 FLMAPI remainingSize( void)
		{
			flmAssert( m_bIsOpen);
			return( m_uiBufferLen - m_uiOffset);
		}

		RCODE FLMAPI close( void);

		FINLINE RCODE FLMAPI positionTo(
			FLMUINT64		ui64Position)
		{
			flmAssert( m_bIsOpen);

			if( ui64Position < m_uiBufferLen)
			{
				m_uiOffset = (FLMUINT)ui64Position;
			}
			else
			{
				m_uiOffset = m_uiBufferLen;
			}

			return( NE_FLM_OK);
		}

		FINLINE FLMUINT64 FLMAPI getCurrPosition( void)
		{
			flmAssert( m_bIsOpen);
			return( m_uiOffset);
		}

		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
			
		FINLINE const FLMBYTE * getBuffer( void)
		{
			flmAssert( m_bIsOpen);
			return( m_pucBuffer);
		}
		
		FINLINE const FLMBYTE * getBufferAtCurrentOffset( void)
		{
			flmAssert( m_bIsOpen);
			return( m_pucBuffer ? &m_pucBuffer[ m_uiOffset] : NULL);
		}
		
		FINLINE void truncate(
			FLMUINT		uiOffset)
		{
			flmAssert( m_bIsOpen);
			flmAssert( uiOffset >= m_uiOffset);
			flmAssert( uiOffset <= m_uiBufferLen);
			
			m_uiBufferLen = uiOffset;
		}

		FINLINE FLMBOOL isOpen( void)
		{
			return( m_bIsOpen);
		}

	private:

		const FLMBYTE *	m_pucBuffer;
		FLMUINT				m_uiBufferLen;
		FLMUINT				m_uiOffset;
		FLMBOOL				m_bAllocatedBuffer;
		FLMBOOL				m_bIsOpen;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileIStream : public F_PosIStream
	{
	public:

		F_FileIStream()
		{
			m_pFileHdl = NULL;
			m_ui64FileOffset = 0;
		}

		virtual ~F_FileIStream()
		{
			if( m_pFileHdl)
			{
				m_pFileHdl->Release();
			}
		}

		RCODE FLMAPI open(
			const char *	pszPath);

		RCODE FLMAPI close( void);

		RCODE FLMAPI positionTo(
			FLMUINT64		ui64Position);

		FLMUINT64 FLMAPI totalSize( void);

		FLMUINT64 FLMAPI remainingSize( void);

		FLMUINT64 FLMAPI getCurrPosition( void);

		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

	private:

		IF_FileHdl *		m_pFileHdl;
		FLMUINT64			m_ui64FileOffset;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferedIStream : public F_PosIStream
	{
	public:

		F_BufferedIStream()
		{
			m_pIStream = NULL;
			m_pucBuffer = NULL;
		}

		virtual ~F_BufferedIStream()
		{
			close();
		}

		RCODE FLMAPI open(
			IF_IStream *		pIStream,
			FLMUINT				uiBufferSize);

		RCODE FLMAPI read(
			void *				pvBuffer,
			FLMUINT				uiBytesToRead,
			FLMUINT *			puiBytesRead);

		RCODE FLMAPI close( void);

		FINLINE FLMUINT64 FLMAPI totalSize( void)
		{
			if (!m_pIStream)
			{
				flmAssert( 0);
				return( 0);
			}

			return( m_uiBytesAvail);
		}

		FINLINE FLMUINT64 FLMAPI remainingSize( void)
		{
			if( !m_pIStream)
			{
				flmAssert( 0);
				return( 0);
			}

			return( m_uiBytesAvail - m_uiBufferOffset);
		}

		FINLINE RCODE FLMAPI positionTo(
			FLMUINT64		ui64Position)
		{
			if( !m_pIStream)
			{
				flmAssert( 0);
				return( RC_SET( NE_FLM_ILLEGAL_OP));
			}

			if( ui64Position < m_uiBytesAvail)
			{
				m_uiBufferOffset = (FLMUINT)ui64Position;
			}
			else
			{
				m_uiBufferOffset = m_uiBytesAvail;
			}

			return( NE_FLM_OK);
		}

		FINLINE FLMUINT64 FLMAPI getCurrPosition( void)
		{
			if( !m_pIStream)
			{
				flmAssert( 0);
				return( 0);
			}

			return( m_uiBufferOffset);
		}

	private:

		IF_IStream *			m_pIStream;
		FLMBYTE *				m_pucBuffer;
		FLMUINT					m_uiBufferSize;
		FLMUINT					m_uiBufferOffset;
		FLMUINT					m_uiBytesAvail;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferedOStream : public F_OStream
	{
	public:

		F_BufferedOStream()
		{
			m_pOStream = NULL;
			m_pucBuffer = NULL;
		}

		virtual ~F_BufferedOStream()
		{
			close();
		}

		RCODE FLMAPI open(
			IF_OStream *	pOStream,
			FLMUINT			uiBufferSize);

		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE FLMAPI close( void);

		RCODE FLMAPI flush( void);

	private:

		IF_OStream *		m_pOStream;
		FLMBYTE *			m_pucBuffer;
		FLMUINT				m_uiBufferSize;
		FLMUINT				m_uiBufferOffset;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileOStream : public F_OStream
	{
	public:

		F_FileOStream()
		{
			m_pFileHdl = NULL;
		}

		virtual ~F_FileOStream()
		{
			close();
		}

		RCODE FLMAPI open(
			const char *	pszFilePath,
			FLMBOOL			bTruncateIfExists);

		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE FLMAPI close( void);

	private:

		IF_FileHdl *		m_pFileHdl;
		FLMUINT64			m_ui64FileOffset;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_MultiFileIStream : public F_IStream
	{
	public:

		F_MultiFileIStream()
		{
			m_pIStream = NULL;
			m_bOpen = FALSE;
		}

		virtual ~F_MultiFileIStream()
		{
			close();
		}

		RCODE FLMAPI open(
			const char *	pszDirectory,
			const char *	pszBaseName);

		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

		RCODE FLMAPI close( void);

	private:

		RCODE rollToNextFile( void);

		IF_IStream *		m_pIStream;
		FLMBOOL				m_bOpen;
		FLMBOOL				m_bEndOfStream;
		FLMUINT				m_uiFileNum;
		FLMUINT64			m_ui64FileOffset;
		char 					m_szDirectory[ F_PATH_MAX_SIZE + 1];
		char	 				m_szBaseName[ F_PATH_MAX_SIZE + 1];
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_MultiFileOStream : public F_OStream
	{
	public:

		F_MultiFileOStream()
		{
			m_pOStream = NULL;
			m_bOpen = FALSE;
		}

		virtual ~F_MultiFileOStream()
		{
			close();
		}

		RCODE create(
			const char *	pszDirectory,
			const char *	pszBaseName,
			FLMUINT			uiMaxFileSize,
			FLMBOOL			bOkToOverwrite);

		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE FLMAPI close( void);

	private:

		RCODE rollToNextFile( void);

		RCODE processDirectory(
			const char *	pszDirectory,
			const char *	pszBaseName,
			FLMBOOL			bOkToDelete);

		F_OStream *		m_pOStream;
		FLMBOOL			m_bOpen;
		FLMUINT			m_uiFileNum;
		FLMUINT64		m_ui64MaxFileSize;
		FLMUINT64		m_ui64FileOffset;
		char 				m_szDirectory[ F_PATH_MAX_SIZE + 1];
		char 				m_szBaseName[ F_PATH_MAX_SIZE + 1];
		
		friend class F_DbSystem;
	};

	/****************************************************************************
	Desc:	Decodes an ASCII base64 stream to binary
	****************************************************************************/
	class F_Base64DecoderIStream : public F_IStream
	{
	public:

		F_Base64DecoderIStream()
		{
			m_pIStream = NULL;
			m_uiBufOffset = 0;
			m_uiAvailBytes = 0;
		}

		virtual ~F_Base64DecoderIStream()
		{
			close();
		}

		RCODE FLMAPI open(
			IF_IStream *	pIStream);
		
		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
			
		FINLINE RCODE FLMAPI close( void)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( m_pIStream)
			{
				if( m_pIStream->getRefCount() == 1)
				{
					rc = m_pIStream->close();
				}

				m_pIStream->Release();
				m_pIStream = NULL;
			}
			
			m_uiAvailBytes = 0;
			m_uiBufOffset = 0;
			
			return( rc);
		}
		
	private:

		IF_IStream *		m_pIStream;
		FLMUINT				m_uiBufOffset;
		FLMUINT				m_uiAvailBytes;
		FLMBYTE				m_ucBuffer[ 8];
		static FLMBYTE		m_ucDecodeTable[ 256];
	};

	/****************************************************************************
	Desc:	Encodes a binary input stream into ASCII base64.
	****************************************************************************/
	class F_Base64EncoderIStream : public F_IStream
	{
	public:

		F_Base64EncoderIStream()
		{
			m_pIStream = NULL;
		}

		virtual ~F_Base64EncoderIStream()
		{
			close();
		}

		RCODE FLMAPI open(
			IF_IStream *	pIStream,
			FLMBOOL			bLineBreaks);
		
		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

		FINLINE RCODE FLMAPI close( void)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( m_pIStream)
			{
				if( m_pIStream->getRefCount() == 1)
				{
					rc = m_pIStream->close();
				}

				m_pIStream->Release();
				m_pIStream = NULL;
			}
			
			return( rc);
		}
		
	private:

		IF_IStream *		m_pIStream;
		FLMBOOL				m_bInputExhausted;
		FLMBOOL				m_bLineBreaks;
		FLMBOOL				m_bPriorLineEnd;
		FLMUINT				m_uiBase64Count;
		FLMUINT				m_uiBufOffset;
		FLMUINT				m_uiAvailBytes;
		FLMBYTE 				m_ucBuffer[ 8];
		static FLMBYTE		m_ucEncodeTable[ 64];
	};

	typedef struct LZWODictItem
	{
		LZWODictItem *	pNext;
		FLMUINT16		ui16Code;
		FLMUINT16		ui16ParentCode;
		FLMBYTE			ucChar;
	} LZWODictItem;

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_CompressingOStream : public F_OStream
	{
	public:

		F_CompressingOStream()
		{
			m_pOStream = NULL;
			m_ppHashTbl = NULL;
			m_pool.poolInit( 64 * 1024);
		}

		virtual ~F_CompressingOStream()
		{
			close();
		}

		RCODE FLMAPI open(
			IF_OStream *	pOStream);

		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		RCODE FLMAPI close( void);

	private:

		FINLINE FLMUINT getHashBucket(
			FLMUINT16	ui16CurrentCode,
			FLMBYTE		ucChar)
		{
			return( ((((FLMUINT)ui16CurrentCode) << 8) | 
				((FLMUINT)ucChar)) % m_uiHashTblSize);
		}

		LZWODictItem * findDictEntry( 
			FLMUINT16		ui16CurrentCode,
			FLMBYTE			ucChar);

		IF_OStream *		m_pOStream;
		LZWODictItem **	m_ppHashTbl;
		FLMUINT				m_uiHashTblSize;
		FLMUINT				m_uiLastRatio;
		FLMUINT				m_uiBestRatio;
		FLMUINT				m_uiCurrentBytesIn;
		FLMUINT				m_uiTotalBytesIn;
		FLMUINT				m_uiCurrentBytesOut;
		FLMUINT				m_uiTotalBytesOut;
		FLMBOOL				m_bStopCompression;
		FLMUINT16			m_ui16CurrentCode;
		FLMUINT16			m_ui16FreeCode;
		F_Pool				m_pool;
	};

	typedef struct LZWIDictItem
	{
		LZWODictItem *	pNext;
		FLMUINT16		ui16ParentCode;
		FLMBYTE			ucChar;
	} LZWIDictItem;

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_UncompressingIStream : public F_IStream
	{
	public:

		F_UncompressingIStream()
		{
			m_pIStream = NULL;
			m_pDict = NULL;
			m_pucDecodeBuffer = NULL;
		}

		virtual ~F_UncompressingIStream()
		{
			close();
		}

		RCODE FLMAPI open(
			IF_IStream *	pIStream);

		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);

		RCODE FLMAPI close( void);
		
	private:

		RCODE readCode(
			FLMUINT16 *		pui16Code);

		RCODE decodeToBuffer(
			FLMUINT16		ui16Code);

		IF_IStream *		m_pIStream;
		LZWIDictItem *		m_pDict;
		FLMBYTE *			m_pucDecodeBuffer;
		FLMUINT				m_uiDecodeBufferSize;
		FLMUINT				m_uiDecodeBufferOffset;
		FLMUINT16			m_ui16FreeCode;
		FLMUINT16			m_ui16LastCode;
		FLMBOOL				m_bStopCompression;
		FLMBOOL				m_bEndOfStream;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class	F_TCPStream : public F_IStream, public F_OStream
	{
	public:
	
		F_TCPStream( void);
		
		virtual ~F_TCPStream( void);

		RCODE openConnection(
			const char *	pucHostAddress,
			FLMUINT			uiPort,
			FLMUINT			uiConnectTimeout	= 3,
			FLMUINT			uiDataTimeout = 15);

		RCODE FLMAPI read(
			void *			pvBuffer,
			FLMUINT			uiBytesToRead,
			FLMUINT *		puiBytesRead);
			
		RCODE FLMAPI write(
			const void *	pvBuffer,
			FLMUINT			uiBytesToWrite,
			FLMUINT *		puiBytesWritten);

		FINLINE RCODE socketPeekWrite(
			FLMINT		iTimeOut)
		{
			return( socketPeek( iTimeOut, FALSE));
		}

		FINLINE RCODE socketPeekRead( 
			FLMINT		iTimeOut)
		{
			return( socketPeek( iTimeOut, TRUE));
		};

		FINLINE const char * getName( void)
		{
			getLocalInfo();
			return( (const char *)m_pszName);
		};

		FINLINE const char * getAddr( void)
		{
			getLocalInfo();
			return( (const char *)m_pszIp);
		};

		FINLINE const char * getPeerName( void)
		{
			getRemoteInfo();
			return( (const char *)m_pszPeerName);
		};

		FINLINE const char * getPeerAddr( void)
		{
			getRemoteInfo();
			return( (const char *)m_pszPeerIp);
		};

		RCODE readNoWait(
			void *			pvBuffer,
			FLMUINT			uiCount,
			FLMUINT *		puiReadRead);

		RCODE readAll(
			void *			pvBuffer,
			FLMUINT			uiCount,
			FLMUINT *		puiBytesRead);

		RCODE	setTcpDelay(
			FLMBOOL			bOn);

		RCODE FLMAPI close( void);
	
	private:
	
		RCODE getLocalInfo( void);
		
		RCODE getRemoteInfo( void);

		RCODE socketPeek(
			FLMINT			iTimoutVal,
			FLMBOOL			bPeekRead);
	
	#ifndef FLM_UNIX
		WSADATA			m_wsaData;
	#endif
		FLMBOOL			m_bInitialized;
		SOCKET			m_iSocket;
		FLMUINT			m_uiIOTimeout;
		FLMBOOL			m_bConnected;
		char				m_pszIp[ 256];
		char				m_pszName[ 256];
		char				m_pszPeerIp[ 256];
		char				m_pszPeerName[ 256];
		unsigned long	m_ulRemoteAddr;
	};
	
	/****************************************************************************
												Misc.
	****************************************************************************/

	FINLINE FLMBOOL f_isHexChar(
		FLMBYTE		ucChar)
	{
		if( (ucChar >= '0' && ucChar <= '9') ||
			(ucChar >= 'A' && ucChar <= 'F') ||
			(ucChar >= 'a' && ucChar <= 'f'))
		{
			return( TRUE);
		}

		return( FALSE);
	}

	FINLINE FLMBOOL f_isHexChar(
		FLMUNICODE		uChar)
	{
		if( uChar > 127)
		{
			return( FALSE);
		}

		return( f_isHexChar( f_tonative( (FLMBYTE)uChar)));
	}

	FINLINE FLMBYTE f_getHexVal(
		FLMBYTE		ucChar)
	{
		if( ucChar >= '0' && ucChar <= '9')
		{
			return( (FLMBYTE)(ucChar - '0'));
		}
		else if( ucChar >= 'A' && ucChar <= 'F')
		{
			return( (FLMBYTE)((ucChar - 'A') + 10));
		}
		else if( ucChar >= 'a' && ucChar <= 'f')
		{
			return( (FLMBYTE)((ucChar - 'a') + 10));
		}

		return( 0);
	}

	FINLINE FLMBYTE f_getHexVal(
		FLMUNICODE	uChar)
	{
		return( f_getHexVal( f_tonative( (FLMBYTE)uChar)));
	}

	FINLINE FLMBOOL f_isValidHexNum(
		const FLMBYTE *	pszString)
	{
		if( *pszString == 0)
		{
			return( FALSE);
		}

		while( *pszString)
		{
			if( !f_isHexChar( *pszString))
			{
				return( TRUE);
			}

			pszString++;
		}

		return( TRUE);
	}

	/****************************************************************************
										Process ID Functions
	****************************************************************************/

	#if defined( FLM_WIN)

		FINLINE FLMUINT f_getpid( void)
		{ 
			return _getpid();
		}

	#elif defined( FLM_UNIX)

		pid_t getpid( void);

		FINLINE FLMUINT f_getpid( void)
		{ 
			return getpid();
		}

	#elif defined( FLM_NLM)

		FINLINE FLMUINT f_getpid() 
		{ 
			return( f_getNLMHandle());
		}
		
	#else
		#error "Unsupported Platform"
	#endif

	/****************************************************************************
											 f_sprintf
	****************************************************************************/

	typedef struct
	{
		FLMBYTE *	pszDestStr;
	} F_SPRINTF_INFO;

	// Percent formating prefixes

	#define FLM_PREFIX_NONE				0
	#define FLM_PREFIX_MINUS 			1
	#define FLM_PREFIX_PLUS				2
	#define FLM_PREFIX_POUND 			3

	// Width and Precision flags

	#define FLM_PRINTF_MINUS_FLAG		0x0001
	#define FLM_PRINTF_PLUS_FLAG		0x0002
	#define FLM_PRINTF_SPACE_FLAG		0x0004
	#define FLM_PRINTF_POUND_FLAG		0x0008
	#define FLM_PRINTF_ZERO_FLAG		0x0010
	#define FLM_PRINTF_SHORT_FLAG		0x0020
	#define FLM_PRINTF_LONG_FLAG		0x0040
	#define FLM_PRINTF_DOUBLE_FLAG	0x0080
	#define FLM_PRINTF_INT64_FLAG		0x0100
	#define FLM_PRINTF_COMMA_FLAG		0x0200

	void f_sprintfProcessFieldInfo(
		FLMBYTE **			ppszFormat,
		FLMUINT *			puiWidth,
		FLMUINT *			puiPrecision,
		FLMUINT *			puiFlags,
		f_va_list *			args);

	void f_sprintfStringFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfCharFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfErrorFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfNotHandledFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);

	void f_sprintfNumberFormatter(
		FLMBYTE				ucFormatChar,
		FLMUINT				uiWidth,
		FLMUINT				uiPrecision,
		FLMUINT				uiFlags,
		F_SPRINTF_INFO *	pInfo,
		f_va_list *			args);
		
	#if defined( FLM_WIN)
	
		typedef struct
		{
			 HANDLE					findHandle;
			 WIN32_FIND_DATA		findBuffer;
			 char 	   			szSearchPath[ F_PATH_MAX_SIZE];
			 FLMUINT					uiSearchAttrib;
		} F_IO_FIND_DATA;
	
		#define F_IO_FA_NORMAL			FILE_ATTRIBUTE_NORMAL		// Normal file
		#define F_IO_FA_RDONLY			FILE_ATTRIBUTE_READONLY		// Read only attribute
		#define F_IO_FA_HIDDEN			FILE_ATTRIBUTE_HIDDEN		// Hidden file
		#define F_IO_FA_SYSTEM			FILE_ATTRIBUTE_SYSTEM		// System file
		#define F_IO_FA_VOLUME			FILE_ATTRIBUTE_VOLUME		// Volume label
		#define F_IO_FA_DIRECTORY		FILE_ATTRIBUTE_DIRECTORY	// Directory
		#define F_IO_FA_ARCHIVE			FILE_ATTRIBUTE_ARCHIVE		// Archive
	
	#elif defined( FLM_UNIX) || defined( FLM_NLM)
	
		typedef struct _DirInfo
		{
			mode_t		mode_flag;
			struct stat	FileStat;
			char 			name[ F_PATH_MAX_SIZE+1];
			char			search_path[ F_PATH_MAX_SIZE+1];
			char			full_path[ F_PATH_MAX_SIZE];
			char			pattern_str[ F_PATH_MAX_SIZE];
			char			dirpath[ F_PATH_MAX_SIZE];
			glob_t      globbuf;
		} F_IO_FIND_DATA;
	
		#define F_IO_FA_NORMAL			0x01	// Normal file, no attributes
		#define F_IO_FA_RDONLY			0x02	// Read only attribute
		#define F_IO_FA_HIDDEN			0x04	// Hidden file
		#define F_IO_FA_SYSTEM			0x08	// System file
		#define F_IO_FA_VOLUME			0x10	// Volume label
		#define F_IO_FA_DIRECTORY		0x20	// Directory
		#define F_IO_FA_ARCHIVE			0x40	// Archive
	
	#else 
		#error Platform not supported
	#endif
	
	RCODE f_fileFindFirst(
		char *				pszSearchPath,
		FLMUINT				uiSearchAttrib,
		F_IO_FIND_DATA	*	find_data,
		char *				pszFoundPath,
		FLMUINT *			puiFoundAttrib);
	
	RCODE f_fileFindNext(
		F_IO_FIND_DATA *	pFindData,
		char *				pszFoundPath,
		FLMUINT *			puiFoundAttrib);
	
	void f_fileFindClose(
		F_IO_FIND_DATA *		pFindData);
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IOBufferMgr : public IF_IOBufferMgr, public F_Base
	{
	public:
	
		F_IOBufferMgr();
	
		virtual ~F_IOBufferMgr();
	
		RCODE FLMAPI waitForAllPendingIO( void);
	
		FINLINE void FLMAPI setMaxBuffers(
			FLMUINT			uiMaxBuffers)
		{
			m_uiMaxBuffers = uiMaxBuffers;
		}
	
		FINLINE void FLMAPI setMaxBytes(
			FLMUINT			uiMaxBytes)
		{
			m_uiMaxBufferBytesToUse = uiMaxBytes;
		}
	
		FINLINE void FLMAPI enableKeepBuffer( void)
		{
			m_bKeepBuffers = TRUE;
		}
	
		RCODE FLMAPI getBuffer(
			IF_IOBuffer **		ppIOBuffer,
			FLMUINT				uiBufferSize,
			FLMUINT				uiBlockSize);
	
		FINLINE FLMBOOL FLMAPI havePendingIO( void)
		{
			return( m_pFirstPending ? TRUE : FALSE);
		}
	
		FINLINE FLMBOOL FLMAPI haveUsed( void)
		{
			return( m_pFirstUsed ? TRUE : FALSE);
		}
	
	private:
	
		// Private methods and variables
	
		F_IOBuffer *		m_pFirstPending;
		F_IOBuffer *		m_pFirstAvail;
		F_IOBuffer *		m_pFirstUsed;
		FLMUINT				m_uiMaxBuffers;
		FLMUINT				m_uiMaxBufferBytesToUse;
		FLMUINT				m_uiBufferBytesInUse;
		FLMUINT				m_uiBuffersInUse;
		RCODE					m_completionRc;
		FLMBOOL				m_bKeepBuffers;
	
		void linkToList(
			F_IOBuffer **	ppListHead,
			F_IOBuffer *	pIOBuffer);
	
		void unlinkFromList(
			F_IOBuffer *	pIOBuffer);
	
	friend class F_IOBuffer;
	
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IOBuffer : public IF_IOBuffer, public F_Base
	{
	#define MAX_BUFFER_BLOCKS	16
	public:
	
		F_IOBuffer();
	
		virtual ~F_IOBuffer();
	
		RCODE FLMAPI setupBuffer(
			FLMUINT	uiBufferSize,
			FLMUINT	uiBlockSize);
	
		FINLINE FLMBYTE * FLMAPI getBuffer( void)
		{
			return( m_pucBuffer);
		}
	
		FINLINE FLMUINT FLMAPI getBufferSize( void)
		{
			return( m_uiBufferSize);
		}
	
		FINLINE FLMUINT FLMAPI getBlockSize( void)
		{
			return( m_uiBlockSize);
		}
	
		void FLMAPI notifyComplete(
			RCODE			rc);
	
		FINLINE void FLMAPI setCompletionCallback(
			WRITE_COMPLETION_CB 	fnCompletion)
		{
			m_fnCompletion = fnCompletion;
		}
	
		FINLINE void FLMAPI setCompletionCallbackData(
			FLMUINT	uiBlockNumber,
			void *	pvData)
		{
			flmAssert( uiBlockNumber < MAX_BUFFER_BLOCKS);
			m_UserData [uiBlockNumber] = pvData;
		}
	
		FINLINE void * FLMAPI getCompletionCallbackData(
			FLMUINT	uiBlockNumber)
		{
			flmAssert( uiBlockNumber < MAX_BUFFER_BLOCKS);
			return( m_UserData [uiBlockNumber]);
		}
	
		FINLINE RCODE FLMAPI getCompletionCode( void)
		{
			return( m_completionRc);
		}
	
		FINLINE eBufferMgrList FLMAPI getList( void)
		{
			return( m_eList);
		}
	
		void FLMAPI makePending( void);
	
	#ifdef FLM_WIN
		FINLINE OVERLAPPED * getOverlapped( void)
		{
			return( &m_Overlapped);
		}
	
		FINLINE void setFileHandle(
			HANDLE	FileHandle)
		{
			m_FileHandle = FileHandle;
		}
	#endif
	
	#if defined( FLM_LINUX) || defined( FLM_SOLARIS) || defined( FLM_OSX)
		FINLINE struct aiocb * getAIOStruct( void)
		{
			return( &m_aio);
		}
	#endif
	
	#ifdef FLM_NLM
		void signalComplete(
			RCODE		rc);
	#endif
	
	private:
	
		// Only called by the buffer manager
	
		RCODE setupIOBuffer(
			F_IOBufferMgr *	pIOBufferMgr);
	
		FLMBOOL isIOComplete( void);
	
		RCODE waitToComplete( void);
	
		// Private methods and variables
	
		F_IOBufferMgr *		m_pIOBufferMgr;
		FLMBYTE *				m_pucBuffer;
		void *					m_UserData [MAX_BUFFER_BLOCKS];
		FLMUINT					m_uiBufferSize;
		FLMUINT					m_uiBlockSize;
		eBufferMgrList			m_eList;
		FLMBOOL					m_bDeleteOnNotify;
	#ifdef FLM_WIN
		HANDLE					m_FileHandle;
		OVERLAPPED				m_Overlapped;
	#endif
	#if defined( FLM_LINUX) || defined( FLM_SOLARIS) || defined( FLM_OSX)
		struct aiocb			m_aio;
	#endif
	#ifdef FLM_NLM
		F_SEM						m_hSem;
	#endif
		F_IOBuffer *			m_pNext;
		F_IOBuffer *			m_pPrev;
		WRITE_COMPLETION_CB	m_fnCompletion;
		RCODE						m_completionRc;
		F_TMSTAMP				m_StartTime;
		FLMUINT64				m_ui64ElapMilli;
	
		friend class F_IOBufferMgr;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_DirHdl : public IF_DirHdl, public F_Base
	{
	public:
	
		F_DirHdl();
	
		virtual ~F_DirHdl()
		{
			if( m_bFindOpen)
			{
				f_fileFindClose( &m_FindData);
			}
		}
	
		RCODE openDir(
			const char *	pszDirName,
			const char *	pszPattern);
	
		RCODE createDir(
			const char *	pszDirName);
	
		RCODE removeDir(
			const char *	pszDirPath);
	
		RCODE FLMAPI next( void);
	
		const char * FLMAPI currentItemName( void);
	
		void FLMAPI currentItemPath(
			char *	pszPath);

			FLMUINT64 FLMAPI currentItemSize( void);
	
		FLMBOOL FLMAPI currentItemIsDir( void);
	
	private:
	
		char					m_szDirectoryPath[ F_PATH_MAX_SIZE];
		char					m_szPattern[ F_PATH_MAX_SIZE];
		FLMUINT32			m_ui32RefCount;
		RCODE					m_rc;
		FLMBOOL				m_bFirstTime;
		FLMBOOL				m_bFindOpen;
		FLMBOOL				m_EOF;
		char					m_szFileName[ F_PATH_MAX_SIZE];
		FLMUINT				m_uiAttrib;
		F_IO_FIND_DATA		m_FindData;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileSystem : public IF_FileSystem, public F_Base
	{
	public:
	
		F_FileSystem()
		{
		}
	
		virtual ~F_FileSystem()
		{
		}
	
		RCODE FLMAPI createFile(
			const char *	pszFileName,
			FLMUINT			uiIoFlags,
			IF_FileHdl **	ppFile);
	
		RCODE FLMAPI createBlockFile(
			const char *	pszFileName,
			FLMUINT			uiIoFlags,
			FLMUINT			uiBlockSize,
			IF_FileHdl **	ppFile);
	
		RCODE FLMAPI createUniqueFile(
			const char *	pszDirName,
			const char *	pszFileExtension,
			FLMUINT			uiIoFlags,
			IF_FileHdl **	ppFile);
	
		RCODE FLMAPI openFile(
			const char *	pszFileName,
			FLMUINT			uiIoFlags,
			IF_FileHdl **	ppFile);
	
		RCODE FLMAPI openBlockFile(
			const char *	pszFileName,
			FLMUINT			uiIoFlags,
			FLMUINT			uiBlockSize,
			IF_FileHdl **	ppFile);
	
		RCODE FLMAPI openDir(
			const char *	pszDirName,
			const char *	pszPattern,
			IF_DirHdl **	ppDir);
	
		RCODE FLMAPI createDir(
			const char *	pszDirName);
	
		RCODE FLMAPI removeDir(
			const char *	pszDirName,
			FLMBOOL			bClear = FALSE);
	
		RCODE FLMAPI doesFileExist(
			const char *	pszFileName);
	
		FLMBOOL FLMAPI isDir(
			const char *	pszFileName);
	
		RCODE FLMAPI getFileTimeStamp(
			const char *	pszFileName,
			FLMUINT *		puiTimeStamp);
	
		RCODE FLMAPI deleteFile(
			const char *	pszFileName);
	
		RCODE FLMAPI copyFile(
			const char *	pszSrcFileName,
			const char *	pszDestFileName,
			FLMBOOL			bOverwrite,
			FLMUINT64 *		pui64BytesCopied);
	
		RCODE FLMAPI renameFile(
			const char *	pszFileName,
			const char *	pszNewFileName);
	
		void FLMAPI pathParse(
			const char *	pszPath,
			char *			pszServer,
			char *			pszVolume,
			char *			pszDirPath,
			char *			pszFileName);
	
		RCODE FLMAPI pathReduce(
			const char *	pszSourcePath,
			char *			pszDestPath,
			char *			pszString);
	
		RCODE FLMAPI pathAppend(
			char *			pszPath,
			const char *	pszPathComponent);
	
		RCODE FLMAPI pathToStorageString(
			const char *	pszPath,
			char *			pszString);
	
		void FLMAPI pathCreateUniqueName(
			FLMUINT *		puiTime,
			char *			pszFileName,
			const char *	pszFileExt,
			FLMBYTE *		pHighChars,
			FLMBOOL			bModext);
	
		FLMBOOL FLMAPI doesFileMatch(
			const char *	pszFileName,
			const char *	pszTemplate);
	
		RCODE FLMAPI getSectorSize(
			const char *	pszFileName,
			FLMUINT *		puiSectorSize);
	
		RCODE setReadOnly(
			const char *	pszFileName,
			FLMBOOL			bReadOnly);
	
	private:
	
	#if defined( FLM_UNIX)
		RCODE unix_RenameSafe(
			const char *	pszSrcFile,
			const char *	pszDestFile);
	
		RCODE unix_TargetIsDir(
			const char	*	tpath,
			FLMBOOL *		isdir);
	#endif
	};
	
	/***************************************************************************
	Desc:
	***************************************************************************/
	#ifdef FLM_WIN
	class F_FileHdl : public IF_FileHdl, public F_Base
	{
	public:
	
		F_FileHdl();
	
		virtual ~F_FileHdl();
	
		RCODE FLMAPI close( void);
		
		RCODE FLMAPI create(
			const char *	pszFileName,
			FLMUINT			uiIoFlags);
	
		RCODE FLMAPI createUnique(
			const char *	pszDirName,
			const char *	pszFileExtension,
			FLMUINT			uiIoFlags);
	
		RCODE FLMAPI open(
			const char *	pszFileName,
			FLMUINT			uiIoFlags);
	
		RCODE FLMAPI flush( void);
	
		RCODE FLMAPI read(
			FLMUINT64		ui64Offset,
			FLMUINT			uiLength,
			void *			pvBuffer,
			FLMUINT *		puiBytesRead);
	
		RCODE FLMAPI seek(
			FLMUINT64		ui64Offset,
			FLMINT			iWhence,
			FLMUINT64 *		pui64NewOffset);
	
		RCODE FLMAPI size(
			FLMUINT64 *		pui64Size);
	
		RCODE FLMAPI tell(
			FLMUINT64 *		pui64Offset);
	
		RCODE FLMAPI truncate(
			FLMUINT64		ui64Size);
	
		RCODE FLMAPI write(
			FLMUINT64		ui64Offset,
			FLMUINT			uiLength,
			const void *	pvBuffer,
			FLMUINT *		puiBytesWritten);
	
		FINLINE RCODE FLMAPI sectorRead(
			FLMUINT64		ui64ReadOffset,
			FLMUINT			uiBytesToRead,
			void *			pvBuffer,
			FLMUINT *		puiBytesReadRV)
		{
			if (m_bDoDirectIO)
			{
				return( directRead( ui64ReadOffset, uiBytesToRead,
						pvBuffer, TRUE, puiBytesReadRV));
			}
			else
			{
				return( read( ui64ReadOffset, uiBytesToRead, pvBuffer, puiBytesReadRV));
			}
		}
	
		FINLINE RCODE FLMAPI sectorWrite(
			FLMUINT64		ui64WriteOffset,
			FLMUINT			uiBytesToWrite,
			const void *	pvBuffer,
			FLMUINT			uiBufferSize,
			void *			pvBufferObj,
			FLMUINT *		puiBytesWrittenRV,
			FLMBOOL			bZeroFill = TRUE)
		{
			F_UNREFERENCED_PARM( uiBufferSize);
			
			if (m_bDoDirectIO)
			{
				return( directWrite( ui64WriteOffset, uiBytesToWrite,
						pvBuffer, (F_IOBuffer *)pvBufferObj, TRUE,
						bZeroFill, puiBytesWrittenRV));
			}
			else
			{
				flmAssert( pvBufferObj == NULL);
				return( write( ui64WriteOffset, uiBytesToWrite, pvBuffer,
									puiBytesWrittenRV));
			}
		}
	
		FINLINE FLMBOOL FLMAPI canDoAsync( void)
		{
			return( m_bCanDoAsync);
		}
	
		FINLINE void FLMAPI setExtendSize(
			FLMUINT		uiExtendSize)
		{
			m_uiExtendSize = uiExtendSize;
		}
	
		FINLINE void FLMAPI setMaxAutoExtendSize(
			FLMUINT		uiMaxAutoExtendSize)
		{
			m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
		}
	
		FINLINE void setupFileHdl(
			FLMUINT			uiFileId,
			FLMBOOL			bDeleteOnRelease)
		{
			m_uiFileId = uiFileId;
			m_bDeleteOnRelease = bDeleteOnRelease;
		}
	
		FINLINE FLMUINT getFileId( void)
		{
			return( m_uiFileId);
		}
	
		FINLINE FLMUINT getSectorSize( void)
		{
			return( m_uiBytesPerSector);
		}
	
		FINLINE void setBlockSize(
			FLMUINT	uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
		}
	
		FINLINE HANDLE getFileHandle( void)
		{
			return m_FileHandle;
		}
	
	private:
	
		RCODE openOrCreate(
			const char *	pszFileName,
			FLMUINT			uiAccess,
			FLMBOOL			bCreateFlag);
	
		RCODE allocAlignBuffer( void);
	
		RCODE doOneRead(
			FLMUINT64		ui64Offset,
			FLMUINT			uiLength,
			void *			pvBuffer,
			FLMUINT *		puiBytesRead);
	
		RCODE directRead(
			FLMUINT64		uiOffset,
			FLMUINT			uiLength,
			void *			pvBuffer,
			FLMBOOL			bBuffHasFullSectors,
			FLMUINT *		puiBytesRead);
	
		RCODE directWrite(
			FLMUINT64		uiOffset,
			FLMUINT			uiLength,
			const void *	pvBuffer,
			F_IOBuffer *	pBufferObj,
			FLMBOOL			bBuffHasFullSectors,
			FLMBOOL			bZeroFill,
			FLMUINT *		puiBytesWritten);
	
		FINLINE FLMUINT64 roundToNextSector(
			FLMUINT64		ui64Bytes)
		{
			return( (ui64Bytes + m_ui64NotOnSectorBoundMask) & 
							m_ui64GetSectorBoundMask);
		}
	
		FINLINE FLMUINT64 truncateToPrevSector(
			FLMUINT64		ui64Offset)
		{
			return( ui64Offset & m_ui64GetSectorBoundMask);
		}
	
		RCODE extendFile(
			FLMUINT64		ui64EndOfLastWrite,
			FLMUINT			uiMaxBytesToExtend,
			FLMBOOL			bFlush);
	
		F_FileHdl *			m_pNext;
		F_FileHdl *			m_pPrev;
		FLMBOOL				m_bInList;
		FLMBOOL				m_bFileOpened;
		FLMUINT				m_uiAvailTime;
		FLMUINT				m_uiFileId;
		FLMBOOL				m_bDeleteOnRelease;
		FLMBOOL				m_bOpenedReadOnly;
		FLMBOOL				m_bOpenedExclusive;
		char *				m_pszFileName;
		HANDLE				m_FileHandle;
		FLMUINT				m_uiBlockSize;
		FLMUINT				m_uiBytesPerSector;
		FLMUINT64			m_ui64NotOnSectorBoundMask;
		FLMUINT64			m_ui64GetSectorBoundMask;
		FLMBOOL				m_bDoDirectIO;
		FLMUINT				m_uiExtendSize;
		FLMUINT				m_uiMaxAutoExtendSize;
		FLMBYTE *			m_pucAlignedBuff;
		FLMUINT				m_uiAlignedBuffSize;
		FLMUINT64			m_ui64CurrentPos;
		FLMBOOL				m_bCanDoAsync;
		OVERLAPPED			m_Overlapped;
	
		friend class F_FileHdlMgr;
		friend class F_FileHdlMgr;
	};
	#endif

	/***************************************************************************
	Desc:
	***************************************************************************/
	#ifdef FLM_UNIX
	class F_FileHdl : public IF_FileHdl, public F_Base
	{
	public:
	
		F_FileHdl();
	
		~F_FileHdl();
	
		RCODE FLMAPI setup(							
			FLMUINT				uiFileId);
	
		RCODE FLMAPI close( void);
													
		RCODE FLMAPI create(
			const char *		pszFileName,
			FLMUINT				uiIoFlags);
	
		RCODE FLMAPI createUnique(
			const char *		pszDirName,
			const char *		pszFileExtension,
			FLMUINT				uiIoFlags);
	
		RCODE FLMAPI open(
			const char *		pszFileName,
			FLMUINT				uiIoFlags);
	
		RCODE FLMAPI flush( void);
	
		RCODE FLMAPI read(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			void *				pvBuffer,
			FLMUINT *			puiBytesRead);
	
		RCODE FLMAPI seek(
			FLMUINT64			ui64Offset,
			FLMINT				iWhence,
			FLMUINT64 *			pui64NewOffset);
	
		RCODE FLMAPI size(
			FLMUINT64 *			pui64Size);
	
		RCODE FLMAPI tell(
			FLMUINT64 *			pui64Offset);
	
		RCODE FLMAPI truncate(
			FLMUINT64			ui64Size);
	
		RCODE FLMAPI write(
			FLMUINT64			ui64Offset,
			FLMUINT				uiLength,
			const void *		pvBuffer,
			FLMUINT *			puiBytesWritten);
	
		RCODE FLMAPI sectorRead(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,
			void *				pvBuffer,
			FLMUINT *			puiBytesReadRV);
	
		RCODE FLMAPI sectorWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			FLMUINT				uiBufferSize,
			void *				pvBufferObj,
			FLMUINT *			puiBytesWrittenRV,
			FLMBOOL				bZeroFill = TRUE)
		{
			if( m_bDoDirectIO)
			{
				return( DirectWrite( ui64WriteOffset, uiBytesToWrite, 
					pvBuffer, uiBufferSize, (F_IOBuffer *)pvBufferObj, 
					puiBytesWrittenRV, TRUE, bZeroFill));
			}
			else
			{
				return( Write( ui64WriteOffset, uiBytesToWrite, 
					pvBuffer, puiBytesWrittenRV));
			}
		}
	
		FLMBOOL FLMAPI canDoAsync( void);
	
		FINLINE FLMBOOL FLMAPI usingDirectIo( void)
		{
			return( m_bDoDirectIO);
		}
	
		FINLINE void FLMAPI setExtendSize(
			FLMUINT				uiExtendSize)
		{
			m_uiExtendSize = uiExtendSize;
		}
	
		FINLINE void FLMAPI setMaxAutoExtendSize(
			FLMUINT				uiMaxAutoExtendSize)
		{
			m_uiMaxAutoExtendSize = uiMaxAutoExtendSize;
		}
	
		RCODE FLMAPI lock( void);
	
		RCODE FLMAPI unlock( void);
	
		FINLINE void FLMAPI setBlockSize(
			FLMUINT				uiBlockSize)
		{
			m_uiBlockSize = uiBlockSize;
		}
		
		FINLINE FLMUINT FLMAPI getSectorSize( void)
		{
			return( m_uiBytesPerSector);
		}
	
		FINLINE void setupFileHdl(
			FLMUINT				uiFileId,
			FLMBOOL				bDeleteOnRelease)
		{
			m_uiFileId = uiFileId;
			m_bDeleteOnRelease = bDeleteOnRelease;
		}
	
		FINLINE FLMUINT getFileId( void)
		{
			return m_uiFileId;
		}
	
	private:
	
		RCODE openOrCreate(
			const char *		pszFileName,
			FLMUINT				uiAccess,
			FLMBOOL				bCreateFlag);
	
		FINLINE FLMUINT64 roundUpToSectorMultiple(
			FLMUINT64			ui64Bytes)
		{
			return( (ui64Bytes + m_ui64NotOnSectorBoundMask) &
						m_ui64GetSectorBoundMask);
		}
	
		FINLINE FLMUINT64 getSectorStartOffset(
			FLMUINT64			ui64Offset)
		{
			return( ui64Offset & m_ui64GetSectorBoundMask);
		}
	
		RCODE directRead(
			FLMUINT64			ui64ReadOffset,
			FLMUINT				uiBytesToRead,	
			void *				pvBuffer,
			FLMBOOL				bBuffHasFullSectors,
			FLMUINT *			puiBytesRead);
	
		RCODE directWrite(
			FLMUINT64			ui64WriteOffset,
			FLMUINT				uiBytesToWrite,
			const void *		pvBuffer,
			FLMUINT				uiBufferSize,
			F_IOBuffer *		pBufferObj,
			FLMUINT *			puiBytesWrittenRV,
			FLMBOOL				bBuffHasFullSectors,
			FLMBOOL				bZeroFill);
		
		RCODE allocAlignBuffer( void);
		
		F_FileHdl *				m_pNext;
		F_FileHdl *				m_pPrev;
		FLMBOOL					m_bInList;
		FLMBOOL					m_bFileOpened;
		FLMUINT					m_uiAvailTime;
		FLMUINT					m_uiFileId;
		FLMBOOL					m_bDeleteOnRelease;
		FLMBOOL					m_bOpenedReadOnly;
		FLMBOOL					m_bOpenedExclusive;
		char *					m_pszFileName;
		int				   	m_fd;
		FLMUINT					m_uiBlockSize;
		FLMUINT					m_uiBytesPerSector;
		FLMUINT64				m_ui64NotOnSectorBoundMask;
		FLMUINT64				m_ui64GetSectorBoundMask;
		FLMUINT64				m_ui64CurrentPos;
		FLMUINT					m_uiExtendSize;
		FLMUINT					m_uiMaxAutoExtendSize;
		FLMBOOL					m_bCanDoAsync;
		FLMBOOL					m_bDoDirectIO;
		FLMBYTE *				m_pucAlignedBuff;
		FLMUINT					m_uiAlignedBuffSize;
		
		friend class F_FileHdlMgr;
	};
	#endif
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_FileHdlMgr : public F_Base
	{
	public:
	
		F_FileHdlMgr();
	
		virtual ~F_FileHdlMgr()
		{
			if( m_hMutex != F_MUTEX_NULL)
			{
				lockMutex( FALSE);
				freeUsedList( TRUE);
				freeAvailList( TRUE);
				unlockMutex( FALSE);
				f_mutexDestroy( &m_hMutex);
			}
		}
	
		RCODE setupFileHdlMgr(
			FLMUINT			uiOpenThreshold = 64,
			FLMUINT			uiMaxAvailTime = 120);
	
		FINLINE void setOpenThreshold(
			FLMUINT			uiOpenThreshold)
		{
			if (m_bIsSetup)
			{
				lockMutex( FALSE);
				m_uiOpenThreshold = uiOpenThreshold;
				unlockMutex( FALSE);
			}
		}
	
		FINLINE void setMaxAvailTime(
			FLMUINT			uiMaxAvailTime)
		{
			if (m_bIsSetup)
			{
				lockMutex( FALSE);
				m_uiMaxAvailTime = uiMaxAvailTime;
				unlockMutex( FALSE);
			}
		}
	
		FINLINE FLMUINT getUniqueId( void)
		{
			FLMUINT	uiTemp;
	
			lockMutex( FALSE);
			uiTemp = ++m_uiFileIdCounter;
			unlockMutex( FALSE);
			return( uiTemp);
		}
	
		void findAvail(
			FLMUINT			uiFileId,
			FLMBOOL			bReadOnlyFlag,
			F_FileHdl **	ppFileHdl);
	
		void makeAvailAndRelease(
			FLMBOOL			bMutexAlreadyLocked,
			F_FileHdl *		pFileHdl);
	
		void removeFileHdls(
			FLMUINT			uiFileId);
	
		void checkAgedFileHdls(
			FLMUINT			uiMinSecondsOpened);
	
		void FINLINE releaseOneAvail(
			FLMBOOL			bMutexAlreadyLocked)
		{
			lockMutex( bMutexAlreadyLocked);
			if (m_pFirstAvail)
			{
				removeFromList( TRUE,
					m_pFirstAvail, &m_pFirstAvail, &m_pLastAvail, &m_uiNumAvail);
			}
			unlockMutex( bMutexAlreadyLocked);
		}
	
		FINLINE FLMUINT getOpenThreshold( void)
		{
			return m_uiOpenThreshold;
		}
	
		FINLINE FLMUINT getOpenedFiles( void)
		{
			FLMUINT		uiTemp;
	
			lockMutex( FALSE);
			uiTemp = m_uiNumUsed + m_uiNumAvail;
			unlockMutex( FALSE);
			return( uiTemp);
		}
	
		FINLINE FLMUINT getMaxAvailTime( void)
		{
			return m_uiMaxAvailTime;
		}
	
		void freeAvailList(
			FLMBOOL	bMutexAlreadyLocked);
	
		void freeUsedList(
			FLMBOOL	bMutexAlreadyLocked);
	
		FINLINE void insertInUsedList(
			FLMBOOL		bMutexAlreadyLocked,
			F_FileHdl *	pFileHdl,
			FLMBOOL		bInsertAtEnd)
		{
			insertInList( bMutexAlreadyLocked,
				pFileHdl, bInsertAtEnd,
				&m_pFirstUsed, &m_pLastUsed, &m_uiNumUsed);
		}
	
	private:
	
		void insertInList(
			FLMBOOL			bMutexAlreadyLocked,
			F_FileHdl *		pFileHdl,
			FLMBOOL			bInsertAtEnd,
			F_FileHdl **	ppFirst,
			F_FileHdl **	ppLast,
			FLMUINT *		puiCount);
	
		void removeFromList(
			FLMBOOL			bMutexAlreadyLocked,
			F_FileHdl *		pFileHdl,
			F_FileHdl **	ppFirst,
			F_FileHdl **	ppLast,
			FLMUINT *		puiCount);
	
		FINLINE void lockMutex(
			FLMBOOL	bMutexAlreadyLocked)
		{
			if (m_hMutex != F_MUTEX_NULL && !bMutexAlreadyLocked)
			{
				f_mutexLock( m_hMutex);
			}
		}
	
		FINLINE void unlockMutex(
			FLMBOOL	bMutexAlreadyLocked)
		{
			if (m_hMutex != F_MUTEX_NULL && !bMutexAlreadyLocked)
			{
				f_mutexUnlock( m_hMutex);
			}
		}
	
		F_MUTEX				m_hMutex;
		FLMUINT				m_uiOpenThreshold;
		FLMUINT				m_uiMaxAvailTime;
		F_FileHdl *			m_pFirstUsed;
		F_FileHdl *			m_pLastUsed;
		FLMUINT				m_uiNumUsed;
		F_FileHdl *			m_pFirstAvail;
		F_FileHdl *			m_pLastAvail;
		FLMUINT				m_uiNumAvail;
		FLMBOOL				m_bIsSetup;
		FLMUINT				m_uiFileIdCounter;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_MultiFileHdl : public IF_MultiFileHdl, public F_Base
	{
	public:
	
		F_MultiFileHdl(
			FLMUINT			uiMaxFileSize = F_MULTI_FHDL_DEFAULT_MAX_FILE_SIZE);
	
		virtual ~F_MultiFileHdl();
	
		void FLMAPI close(
			FLMBOOL			bDelete = FALSE);
	
	
		RCODE FLMAPI create(
			const char *	pszPath);
	
		RCODE FLMAPI createUnique(
			const char *	pszPath,
			const char *	pszFileExtension);
	
		RCODE FLMAPI deleteMultiFile(
			const char *	pszPath);
	
		RCODE FLMAPI open(
			const char *	pszPath);
	
		RCODE FLMAPI flush( void);
	
		RCODE FLMAPI read(
			FLMUINT64	ui64Offset,
			FLMUINT		uiLength,
			void *		pvBuffer,
			FLMUINT *	puiBytesRead);
	
		RCODE FLMAPI write(
			FLMUINT64	ui64Offset,
			FLMUINT		uiLength,
			void *		pvBuffer,
			FLMUINT *	puiBytesWritten);
	
		RCODE FLMAPI getPath(
			char *	pszFilePath);
	
		FINLINE RCODE FLMAPI size(
			FLMUINT64 *	pui64FileSize)
		{
			*pui64FileSize = m_ui64EOF;
			return( NE_FLM_OK);
		}
	
		RCODE FLMAPI truncate(
			FLMUINT64	ui64NewSize);
	
	private:
	
		RCODE getFileHdl(
			FLMUINT				uiFileNum,
			FLMBOOL				bGetForWrite,
			IF_FileHdl **		ppFileHdl);
	
		RCODE createLockFile(
			const char *		pszBasePath);
	
		FINLINE void releaseLockFile(
			const char *		pszBasePath,
			FLMBOOL				bDelete)
		{
	#ifndef FLM_UNIX
			F_UNREFERENCED_PARM( bDelete);
			F_UNREFERENCED_PARM( pszBasePath);
	#endif
	
			if( m_pLockFileHdl)
			{
	
				// Release the lock file
	
				(void)m_pLockFileHdl->close();
				m_pLockFileHdl->Release();
				m_pLockFileHdl = NULL;
	
	#ifdef FLM_UNIX
				if( bDelete)
				{
					char		szTmpPath[ F_PATH_MAX_SIZE];
	
					// Delete the lock file
	
					f_strcpy( szTmpPath, pszBasePath);
					gv_pFileSystem->pathAppend( szTmpPath, "64.LCK");
					gv_pFileSystem->Delete( szTmpPath);
				}
	#endif
			}
		}
	
		FINLINE void formatFileNum(
			FLMUINT	uiFileNum,
			char *	pszStr)
		{
			f_sprintf( pszStr, "%08X.64", (unsigned)uiFileNum);
		}
	
		RCODE getFileNum(
			const char *	pszFileName,
			FLMUINT *		puiFileNum);
	
		FINLINE void dataFilePath(
			FLMUINT		uiFileNum,
			char *		pszPath)
		{
			char	szFileName[ 13];
	
			f_strcpy( pszPath, m_szPath);
			formatFileNum( uiFileNum, szFileName);
			gv_pFileSystem->pathAppend( pszPath, szFileName);
		}
	
		FINLINE FLMUINT getFileNum(
			FLMUINT64		ui64Offset)
		{
			return( (FLMUINT)(ui64Offset / m_uiMaxFileSize));
		}
	
		FINLINE FLMUINT getFileOffset(
			FLMUINT64		ui64Offset)
		{
			return( (FLMUINT)(ui64Offset % m_uiMaxFileSize));
		}
	
		FH_INFO				m_pFileHdlList[ F_MULTI_FHDL_LIST_SIZE];
		char					m_szPath[ F_PATH_MAX_SIZE];
		FLMBOOL				m_bOpen;
		FLMUINT64			m_ui64EOF;
		FLMUINT				m_uiMaxFileSize;
		IF_FileHdl *		m_pLockFileHdl;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_ThreadMgr : public IF_ThreadMgr, public F_Base
	{
	public:
	
		F_ThreadMgr()
		{
			m_hMutex = F_MUTEX_NULL;
			m_pThreadList = NULL;
			m_uiNumThreads = 0;
		}
	
		virtual ~F_ThreadMgr();
	
		RCODE FLMAPI setupThreadMgr( void);
	
		void FLMAPI shutdownThreadGroup(
			FLMUINT			uiThreadGroup);
	
		void FLMAPI setThreadShutdownFlag(
			FLMUINT			uiThreadId);
	
		RCODE FLMAPI findThread(
			IF_Thread **	ppThread,
			FLMUINT			uiThreadGroup,
			FLMUINT			uiAppId = 0,
			FLMBOOL			bOkToFindMe = TRUE);
	
		RCODE FLMAPI getNextGroupThread(
			IF_Thread **	ppThread,
			FLMUINT			uiThreadGroup,
			FLMUINT *		puiThreadId);
	
		RCODE FLMAPI getThreadInfo(
			IF_Pool *			pPool,
			F_THREAD_INFO **	ppThreadInfo,
			FLMUINT *			puiNumThreads);
	
		FLMUINT FLMAPI getThreadGroupCount(
			FLMUINT			uiThreadGroup);
			
		inline void lockMutex( void)
		{
			f_mutexLock( m_hMutex);
		}
		
		inline void unlockMutex( void)
		{
			f_mutexUnlock( m_hMutex);
		}
	
		void unlinkThread(
			IF_Thread *		pThread,
			FLMBOOL			bMutexLocked);
	
	private:
	
		F_MUTEX			m_hMutex;
		F_Thread *		m_pThreadList;
		FLMUINT			m_uiNumThreads;
	
	friend class F_Thread;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_Thread : public IF_Thread, public F_Base
	{
	public:
	
		F_Thread()
		{
			m_hMutex = F_MUTEX_NULL;
			m_pszThreadName = NULL;
			m_pszThreadStatus = NULL;
			m_uiStatusBufLen = 0;
			m_pPrev = NULL;
			m_pNext = NULL;
			cleanupThread();
		}
	
		virtual ~F_Thread()
		{
			stopThread();
			cleanupThread();
		}
	
		FLMINT FLMAPI AddRef( void);
	
		FLMINT FLMAPI Release( void);
	
		RCODE FLMAPI startThread(
			F_THREAD_FUNC	fnThread,
			const char *	pszThreadName = NULL,
			FLMUINT			uiThreadGroup = 0,
			FLMUINT			uiAppId = 0,
			void *			pvParm1 = NULL,
			void *			pvParm2 = NULL,
			FLMUINT        uiStackSize = F_THREAD_DEFAULT_STACK_SIZE);
	
		void FLMAPI stopThread( void);
	
		FINLINE FLMUINT FLMAPI getThreadId( void)
		{
			return( m_uiThreadId);
		}
	
		FINLINE FLMBOOL FLMAPI getShutdownFlag( void)
		{
			return( m_bShutdown);
		}
	
		FINLINE RCODE FLMAPI getExitCode( void)
		{
			return( m_exitRc);
		}
	
		FINLINE void * FLMAPI getParm1( void)
		{
			return( m_pvParm1);
		}
	
		FINLINE void FLMAPI setParm1(
			void *		pvParm)
		{
			m_pvParm1 = pvParm;
		}
	
		FINLINE void * FLMAPI getParm2( void)
		{
			return( m_pvParm2);
		}
	
		FINLINE void FLMAPI setParm2(
			void *		pvParm)
		{
			m_pvParm2 = pvParm;
		}
	
		FINLINE void FLMAPI setShutdownFlag( void)
		{
			m_bShutdown = TRUE;
		}
	
		FINLINE FLMBOOL FLMAPI isThreadRunning( void)
		{
			return( m_bRunning);
		}
	
		void FLMAPI setThreadStatusStr(
			const char *	pszStatus);
	
		void FLMAPI setThreadStatus(
			const char *	pszBuffer, ...);
	
		void FLMAPI setThreadStatus(
			eThreadStatus	genericStatus);
	
		FINLINE void FLMAPI setThreadAppId(
			FLMUINT		uiAppId)
		{
			f_mutexLock( m_hMutex);
			m_uiAppId = uiAppId;
			f_mutexUnlock( m_hMutex);
		}
	
		FINLINE FLMUINT FLMAPI getThreadAppId( void)
		{
			return( m_uiAppId);
		}
	
		FINLINE FLMUINT FLMAPI getThreadGroup( void)
		{
			return( m_uiThreadGroup);
		}
	
		void FLMAPI cleanupThread( void);
	
		F_MUTEX				m_hMutex;
		F_Thread *			m_pPrev;
		F_Thread *			m_pNext;
		char *				m_pszThreadName;
		char *				m_pszThreadStatus;
		FLMUINT				m_uiStatusBufLen;
		FLMBOOL				m_bShutdown;
		F_THREAD_FUNC		m_fnThread;
		FLMBOOL				m_bRunning;
		FLMUINT				m_uiStackSize;
		void *				m_pvParm1;
		void *				m_pvParm2;
		FLMUINT				m_uiThreadId;
		FLMUINT				m_uiThreadGroup;
		FLMUINT				m_uiAppId;
		FLMUINT				m_uiStartTime;
		RCODE					m_exitRc;
	
	friend class F_ThreadMgr;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_IniFile : public IF_IniFile, public F_Base
	{
	public:
	
		F_IniFile();
		
		virtual ~F_IniFile();
		
		RCODE FLMAPI init( void);
		
		RCODE FLMAPI read(
			const char *		pszFileName);
			
		RCODE FLMAPI write( void);
	
		FLMBOOL FLMAPI getParam(
			const char *	pszParamName,
			FLMUINT *		puiParamVal);
		
		FLMBOOL FLMAPI getParam(
			const char *	pszParamName,
			FLMBOOL *		pbParamVal);
		
		FLMBOOL FLMAPI getParam(
			const char *	pszParamName,
			char **			ppszParamVal);
		
		RCODE FLMAPI setParam(
			const char *	pszParamName,
			FLMUINT 			uiParamVal);
	
		RCODE FLMAPI setParam(
			const char *	pszParamName,
			FLMBOOL			bParamVal);
	
		RCODE FLMAPI setParam(
			const char *	pszParamName,
			const char *	pszParamVal);
	
		FINLINE FLMBOOL FLMAPI testParam(
			const char *	pszParamName)
		{
			if( findParam( pszParamName))
			{
				return( TRUE);
			}
			
			return( FALSE);
		}
	
	private:
	
		RCODE readLine(
			char *			pucBuf,
			FLMUINT *		puiBytes,
			FLMBOOL *		pbMore);
	
		RCODE parseBuffer(
			char *			pucBuf,
			FLMUINT			uiNumButes);
	
		INI_LINE * findParam(
			const char *	pszParamName);
	
		RCODE setParamCommon( 
			INI_LINE **		ppLine,
			const char *	pszParamName);
	
		void fromAscii( 
			FLMUINT * 		puiVal,
			const char *	pszParamValue);
			
		void fromAscii(
			FLMBOOL *		pbVal,
			const char *	pszParamValue);
	
		RCODE toAscii( 
			char **			ppszParamValue,
			FLMUINT			puiVal);
			
		RCODE toAscii( 
			char **			ppszParamValue,
			FLMBOOL 			pbVal);
			
		RCODE toAscii(
			char **			ppszParamValue,
			const char * 	pszVal);
	
		FINLINE FLMBOOL isWhiteSpace(
			FLMBYTE			ucChar)
		{
			return( ucChar == 32 || ucChar == 9 ? TRUE : FALSE);
		}
		
		IF_Pool *			m_pPool;
		IF_FileHdl * 		m_pFileHdl;
		char *				m_pszFileName;
		INI_LINE *			m_pFirstLine;	
		INI_LINE *			m_pLastLine;
		FLMBOOL				m_bReady;
		FLMBOOL				m_bModified;
		FLMUINT				m_uiFileOffset;
	};

	/****************************************************************************
	Desc:
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
		
		FINLINE void FLMAPI truncateData(
			FLMUINT			uiSize)
		{
			if( uiSize < m_uiOffset)
			{
				m_uiOffset = uiSize;
			}
		}
		
		FINLINE RCODE FLMAPI allocSpace(
			FLMUINT		uiSize,
			void **		ppvPtr)
		{
			RCODE		rc = NE_FLM_OK;
			
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
		
		FINLINE RCODE FLMAPI appendData(
			const void *		pvData,
			FLMUINT				uiSize)
		{
			RCODE		rc = NE_FLM_OK;
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
			
		FINLINE RCODE FLMAPI appendByte(
			FLMBYTE		ucChar)
		{
			RCODE			rc = NE_FLM_OK;
			FLMBYTE *	pucTmp;
			
			if( RC_BAD( rc = allocSpace( 1, (void **)&pucTmp)))
			{
				goto Exit;
			}
			
			*pucTmp = ucChar;
			
		Exit:
		
			return( rc);
		}
		
		FINLINE RCODE FLMAPI appendUniChar(
			FLMUNICODE	uChar)
		{
			RCODE				rc = NE_FLM_OK;
			FLMUNICODE *	puTmp;
			
			if( RC_BAD( rc = allocSpace( sizeof( FLMUNICODE), (void **)&puTmp)))
			{
				goto Exit;
			}
			
			*puTmp = uChar;
			
		Exit:
		
			return( rc);
		}
		
		FINLINE FLMBYTE * FLMAPI getBufferPtr( void)
		{
			return( m_pucBuffer);
		}
		
		FINLINE FLMUNICODE * FLMAPI getUnicodePtr( void)
		{
			if( m_uiOffset >= sizeof( FLMUNICODE))
			{
				return( (FLMUNICODE *)m_pucBuffer);
			}
			
			return( NULL);
		}
		
		FINLINE FLMUINT FLMAPI getUnicodeLength( void)
		{
			if( m_uiOffset <= sizeof( FLMUNICODE))
			{
				return( 0);
			}
			
			return( (m_uiOffset >> 1) - 1);
		}
		
		FINLINE FLMUINT FLMAPI getDataLength( void)
		{
			return( m_uiOffset);
		}
		
		FINLINE RCODE FLMAPI copyFromBuffer(
			IF_DynaBuf *		pSource)
		{
			RCODE		rc = NE_FLM_OK;
			
			if( RC_BAD( rc = resizeBuffer( 
				((F_DynaBuf *)pSource)->m_uiBufferSize)))
			{
				goto Exit;
			}
			
			if( (m_uiOffset = ((F_DynaBuf *)pSource)->m_uiOffset) != 0)
			{
				f_memcpy( m_pucBuffer, ((F_DynaBuf *)pSource)->m_pucBuffer, 
					((F_DynaBuf *)pSource)->m_uiOffset);
			}
			
		Exit:
			
			return( rc);
		}		
		
	private:
	
		FINLINE RCODE resizeBuffer(
			FLMUINT		uiNewSize)
		{
			RCODE	rc = NE_FLM_OK;
			
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

	typedef struct
	{
		FLMUINT32	ui32Offset;
		FLMUINT32	ui32Length;
	} F_VAR_HEADER;
	
	typedef struct
	{
		FLMUINT64	ui64FilePos;
		FLMUINT		uiEntryCount;
		FLMUINT		uiBlockSize;
		FLMBOOL		bFirstBlock;
		FLMBOOL		bLastBlock;
	} F_BLOCK_HEADER;
	
	#define	RSBLK_UNSET_FILE_POS		(~((FLMUINT64)0))

	/****************************************************************************
	Desc: Result set block
	****************************************************************************/
	class	F_ResultSetBlk : public F_RefCount, public F_Base
	{
	public:
	
		F_ResultSetBlk();
	
		FINLINE ~F_ResultSetBlk()
		{
			if (m_pNext)
			{
				m_pNext->m_pPrev = m_pPrev;
			}
			
			if( m_pPrev)
			{
				m_pPrev->m_pNext = m_pNext;
			}
			
			if (m_pCompare)
			{
				m_pCompare->Release();
			}
		}
	
		void reset( void);
	
		void setup(
			IF_MultiFileHdl **		ppMultiFileHdl,
			IF_ResultSetCompare *	pCompare,
			FLMUINT						uiEntrySize,
			FLMBOOL						bFirstInList,
			FLMBOOL						bDropDuplicates,
			FLMBOOL						bEntriesInOrder);
	
		RCODE setBuffer(
			FLMBYTE *					pBuffer,
			FLMUINT						uiBufferSize = RS_BLOCK_SIZE);
	
		FINLINE FLMUINT bytesUsedInBuffer( void)
		{
			if (m_bEntriesInOrder)
			{
				return( m_BlockHeader.uiBlockSize);
			}
			else
			{
				return( m_BlockHeader.uiBlockSize - m_uiLengthRemaining);
			}
		}
	
		RCODE addEntry(
			FLMBYTE *	pEntry,
			FLMUINT		uiEntryLength );
	
		RCODE modifyEntry(
			FLMBYTE *	pEntry,
			FLMUINT		uiEntryLength = 0);
	
		FINLINE RCODE finalize(
			FLMBOOL		bForceWrite)
		{
			return( flush( TRUE, bForceWrite));
		}
	
		RCODE flush(
			FLMBOOL		bLastBlockInList,
			FLMBOOL		bForceWrite);
	
		RCODE getCurrent(
			FLMBYTE *	pBuffer,
			FLMUINT		uiBufferLength,
			FLMUINT *	puiReturnLength);
	
		FINLINE RCODE getNext(
			FLMBYTE *	pucBuffer,
			FLMUINT		uiBufferLength,
			FLMUINT *	puiReturnLength)
		{
			// Are we on the last entry or past the last entry?
	
			if (m_iEntryPos + 1 >= (FLMINT)m_BlockHeader.uiEntryCount)
			{
				m_iEntryPos = (FLMINT) m_BlockHeader.uiEntryCount;
				return RC_SET( NE_FLM_EOF_HIT);
			}
	
			m_iEntryPos++;
			return( copyCurrentEntry( pucBuffer, uiBufferLength, puiReturnLength));
		}
	
		RCODE getNextPtr(
			FLMBYTE **	ppBuffer,
			FLMUINT *	puiReturnLength);
	
		RCODE getPrev(
			FLMBYTE *	pBuffer,
			FLMUINT		uiBufferLength,
			FLMUINT *	puiReturnLength);
	
		FINLINE FLMUINT64 getPosition( void)
		{
			return( (!m_bPositioned ||
									m_iEntryPos == -1 ||
									m_iEntryPos == (FLMINT)m_BlockHeader.uiEntryCount
									? RS_POSITION_NOT_SET
									: m_ui64BlkEntryPosition + (FLMUINT64)m_iEntryPos));
		}
	
		RCODE setPosition(
			FLMUINT64	ui64Position );
	
		RCODE	findMatch(
			FLMBYTE *	pMatchEntry,
			FLMUINT		uiMatchEntryLength,
			FLMBYTE *	pFoundEntry,
			FLMUINT *	puiFoundEntryLength,
			FLMINT *		piCompare);
	
		void adjustState(
			FLMUINT		uiBlkBufferSize);
	
		RCODE truncate(
			FLMBYTE *	pszPath);
	
	private:
	
		RCODE addEntry(
			FLMBYTE *	pucEntry);
	
		void squeezeSpace( void);
	
		RCODE sortAndRemoveDups( void);
	
		void removeEntry(
			FLMBYTE *	pucEntry);
	
		RCODE quickSort(
			FLMUINT		uiLowerBounds,
			FLMUINT		uiUpperBounds);
	
		FINLINE RCODE entryCompare(
			FLMBYTE *	pucLeftEntry,
			FLMBYTE *	pucRightEntry,
			FLMINT *		piCompare)
		{
			RCODE			rc;
	
			if( m_bFixedEntrySize)
			{
				rc = m_pCompare->compare( pucLeftEntry,  m_uiEntrySize,
							pucRightEntry, m_uiEntrySize, piCompare);
			}
			else
			{
				rc = m_pCompare->compare(
							m_pucBlockBuf + ((F_VAR_HEADER *)pucLeftEntry)->ui32Offset,
							((F_VAR_HEADER *)pucLeftEntry)->ui32Length,
							m_pucBlockBuf + ((F_VAR_HEADER *)pucRightEntry)->ui32Offset,
							((F_VAR_HEADER *)pucRightEntry)->ui32Length,
							piCompare);
			}
			if (*piCompare == 0)
			{
				m_bDuplicateFound = TRUE;
			}
			
			return( rc);
		}
	
		RCODE copyCurrentEntry(
			FLMBYTE *	pBuffer,
			FLMUINT		uiBufferLength,
			FLMUINT *	puiReturnLength);
	
		RCODE compareEntry(
			FLMBYTE *	pMatchEntry,
			FLMUINT		uiMatchEntryLength,
			FLMUINT		uiEntryPos,
			FLMINT *		piCompare);
	
		RCODE write();
		
		RCODE read();
	
		F_BLOCK_HEADER					m_BlockHeader;
		IF_ResultSetCompare *		m_pCompare;
		FLMBYTE *						m_pucBlockBuf;
		FLMBYTE *						m_pucEndPoint;
		F_ResultSetBlk *				m_pNext;
		F_ResultSetBlk *				m_pPrev;
		IF_MultiFileHdl **			m_ppMultiFileHdl;
		FLMUINT64						m_ui64BlkEntryPosition;
		FLMUINT							m_uiLengthRemaining;
		FLMINT							m_iEntryPos;
		FLMUINT							m_uiEntrySize;
		FLMBOOL							m_bEntriesInOrder;
		FLMBOOL							m_bFixedEntrySize;
		FLMBOOL							m_bPositioned;
		FLMBOOL							m_bModifiedEntry;
		FLMBOOL							m_bDuplicateFound;
		FLMBOOL							m_bDropDuplicates;
		
		friend class F_ResultSet;
	};

	/*****************************************************************************
	Desc:	Result set
	*****************************************************************************/
	class F_ResultSet : public IF_ResultSet, public F_Base
	{
	public:
	
		F_ResultSet();
		
		F_ResultSet(
			FLMUINT		uiBlkSize);
	
		virtual ~F_ResultSet();
	
		RCODE FLMAPI setupResultSet(
			const char *				pszPath,
			IF_ResultSetCompare *	pCompare,
			FLMUINT						uiEntrySize,
			FLMBOOL						bDropDuplicates = TRUE,
			FLMBOOL						bEntriesInOrder = FALSE,
			const char *				pszInputFileName = NULL);	
	
		FINLINE void FLMAPI setSortStatus(
			IF_ResultSetSortStatus *	pSortStatus)
		{
			if (m_pSortStatus)
			{
				m_pSortStatus->Release();
				m_pSortStatus = NULL;
			}
			
			if ((m_pSortStatus = pSortStatus) != NULL)
			{
				m_pSortStatus->AddRef();
			}
		}
	
		FINLINE FLMUINT64 FLMAPI getTotalEntries( void)
		{
			F_ResultSetBlk	*	pBlk = m_pFirstRSBlk;
			FLMUINT64			ui64TotalEntries = 0;
	
			for( pBlk = m_pFirstRSBlk; pBlk; pBlk = pBlk->m_pNext)
			{
				ui64TotalEntries += pBlk->m_BlockHeader.uiEntryCount;
			}
			
			return( ui64TotalEntries);
		}
	
		RCODE FLMAPI addEntry(
			const void *	pvEntry,
			FLMUINT			uiEntryLength = 0);
	
		RCODE FLMAPI finalizeResultSet(
			FLMUINT64 *		pui64TotalEntries = NULL);
	
		RCODE FLMAPI getFirst(
			void *			pvEntryBuffer,
			FLMUINT			uiBufferLength = 0,
			FLMUINT *		puiEntryLength = NULL);
	
		RCODE FLMAPI getNext(
			void *			pvEntryBuffer,
			FLMUINT			uiBufferLength = 0,
			FLMUINT *		puiEntryLength = NULL);
	
		RCODE FLMAPI getLast(
			void *			pvEntryBuffer,
			FLMUINT			uiBufferLength = 0,
			FLMUINT *		puiEntryLength = NULL);
	
		RCODE FLMAPI getPrev(
			void *			pvEntryBuffer,
			FLMUINT			uiBufferLength = 0,
			FLMUINT *		puiEntryLength = NULL);
	
		RCODE FLMAPI getCurrent(
			void *			pvEntryBuffer,
			FLMUINT			uiBufferLength = 0,
			FLMUINT *		puiEntryLength = NULL);
	
		FINLINE RCODE FLMAPI modifyCurrent(
			const void *	pvEntry,
			FLMUINT			uiEntryLength = 0)
		{
			return( m_pCurRSBlk->modifyEntry( (FLMBYTE *)pvEntry, uiEntryLength));
		}
	
		FINLINE RCODE FLMAPI findMatch(
			const void *	pvMatchEntry,
			void *			pvFoundEntry)
		{
			return( findMatch( pvMatchEntry, m_uiEntrySize,
									pvFoundEntry, NULL));
		}
	
		RCODE FLMAPI findMatch(
			const void *	pvMatchEntry,
			FLMUINT			uiMatchEntryLength,
			void *			pvFoundEntry,
			FLMUINT *		puiFoundEntryLength);
			
		FINLINE FLMUINT64 FLMAPI getPosition( void)
		{
			return( (!m_pCurRSBlk
									? RS_POSITION_NOT_SET
									: m_pCurRSBlk->getPosition()));
		}
	
		RCODE FLMAPI setPosition(
			FLMUINT64		ui64Position);
	
		RCODE FLMAPI resetResultSet(
			FLMBOOL			bDelete = TRUE);
	
		RCODE FLMAPI flushToFile( void);
	
	private:
	
		FINLINE FLMUINT64 numberOfBlockChains( void)
		{
			FLMUINT64			ui64Count = 0;
			F_ResultSetBlk *	pBlk = m_pFirstRSBlk;
	
			for (; pBlk ; pBlk = pBlk->m_pNext)
			{
				if (pBlk->m_BlockHeader.bFirstBlock)
				{
					ui64Count++;
				}
			}
			
			return( ui64Count);
		}
	
		RCODE mergeSort();
	
		RCODE getNextPtr(
			F_ResultSetBlk **			ppCurBlk,
			FLMBYTE *	*				ppBuffer,
			FLMUINT *					puiReturnLength);
	
		RCODE unionBlkLists(
			F_ResultSetBlk *			pLeftBlk,
			F_ResultSetBlk *			pRightBlk = NULL);
	
		RCODE copyRemainingItems(
			F_ResultSetBlk *			pCurBlk);
	
		void closeFile(
			IF_MultiFileHdl **		ppMultiFileHdl,
			FLMBOOL						bDelete = TRUE);
	
		RCODE openFile(
			IF_MultiFileHdl **		ppMultiFileHdl);
	
		F_ResultSetBlk * selectMidpoint(
			F_ResultSetBlk *			pLowBlk,
			F_ResultSetBlk *			pHighBlk,
			FLMBOOL						bPickHighIfNeighbors);
	
		RCODE setupFromFile( void);
	
		IF_ResultSetCompare *		m_pCompare;
		IF_ResultSetSortStatus *	m_pSortStatus;
		FLMUINT64						m_ui64EstTotalUnits;
		FLMUINT64						m_ui64UnitsDone;
		FLMUINT							m_uiEntrySize;
		FLMUINT64						m_ui64TotalEntries;
		F_ResultSetBlk *				m_pCurRSBlk;
		F_ResultSetBlk *				m_pFirstRSBlk;
		F_ResultSetBlk *				m_pLastRSBlk;
		char								m_szIoDefaultPath[ F_PATH_MAX_SIZE];
		char								m_szIoFilePath1[ F_PATH_MAX_SIZE];
		char								m_szIoFilePath2[ F_PATH_MAX_SIZE];
		IF_MultiFileHdl *				m_pMultiFileHdl1;
		IF_MultiFileHdl *				m_pMultiFileHdl2;
		FLMBYTE *						m_pucBlockBuf1;
		FLMBYTE *						m_pucBlockBuf2;
		FLMBYTE *						m_pucBlockBuf3;
		FLMUINT							m_uiBlockBuf1Len;
		FLMBOOL							m_bFile1Opened;
		FLMBOOL							m_bFile2Opened;
		FLMBOOL							m_bOutput2ndFile;
		FLMBOOL							m_bInitialAdding;
		FLMBOOL							m_bFinalizeCalled;
		FLMBOOL							m_bSetupCalled;
		FLMBOOL							m_bDropDuplicates;
		FLMBOOL							m_bAppAddsInOrder;
		FLMBOOL							m_bEntriesInOrder;
		FLMUINT							m_uiBlkSize;
	
		friend class F_ResultSetBlk;
	};
	
	/****************************************************************************
	Desc: Logging
	****************************************************************************/

	void flmDbgLogInit( void);
	void flmDbgLogExit( void);
	void flmDbgLogFlush( void);

	/****************************************************************************
	Desc:	Timers
	****************************************************************************/
	#if defined( FLM_NLM)
	
		extern "C" void ConvertTicksToSeconds(
			LONG		ticks,
			LONG *	seconds,
			LONG *	tenthsOfSeconds);
	
		extern "C" void ConvertSecondsToTicks(
			LONG		seconds,
			LONG		tenthsOfSeconds,
			LONG *	ticks);
	
		#define FLM_GET_TIMER()	\
			(FLMUINT)GetCurrentTime()
	
		#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU) \
			ConvertSecondsToTicks( (LONG)(uiSeconds), 0, (LONG *)(&(uiTU)))
	
		#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds) \
		{ \
			LONG	udDummy; \
			ConvertTicksToSeconds( (LONG)(uiTU), (LONG *)(&(uiSeconds)), &udDummy); \
		}
	
		#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli) \
		{ \
			LONG	udTenths; \
			LONG	udSeconds; \
			ConvertTicksToSeconds( (LONG)(uiTU), (LONG *)(&(udSeconds)), &udTenths); \
			uiMilli = (FLMUINT)(udSeconds) * 1000 + (FLMUINT)udTenths * 100; \
		}
		
		#define FLM_MILLI_TO_TIMER_UNITS( uiMilliSeconds, uiTU) \
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
	
		#define FLM_GET_TIMER() \
			(FLMUINT) f_timeGetMilliTime()
			
		#define FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTU) \
			 ((uiTU) = ((uiSeconds) * 1000))
			 
		#define FLM_TIMER_UNITS_TO_SECS( uiTU, uiSeconds) \
			 ((uiSeconds) = ((uiTU) / 1000))
			 
		#define FLM_TIMER_UNITS_TO_MILLI( uiTU, uiMilli) \
			 ((uiMilli) = (uiTU))
			 
		#define FLM_MILLI_TO_TIMER_UNITS( uiMilli, uiTU) \
			 ((uiTU) = (uiMilli))

	#elif defined( FLM_WIN)
	
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
	Desc: XML
	****************************************************************************/
	class F_XML : public IF_XML, public virtual F_Base
	{
	public:
	
		F_XML();
	
		virtual ~F_XML();
		
		RCODE FLMAPI setup( void);
	
		FLMBOOL FLMAPI isPubidChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isQuoteChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isWhitespace(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isExtender(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isCombiningChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isNameChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isNCNameChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isIdeographic(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isBaseChar(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isDigit(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isLetter(
			FLMUNICODE		uChar);
	
		FLMBOOL FLMAPI isNameValid(
			FLMUNICODE *	puzName,
			FLMBYTE *		pszName);
	
	private:
	
		void setCharFlag(
			FLMUNICODE		uLowChar,
			FLMUNICODE		uHighChar,
			FLMUINT16		ui16Flag);
	
		XMLCHAR *			m_pCharTable;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_XMLNamespace : public F_RefCount, public F_Base
	{
	public:
	
		FINLINE F_XMLNamespace()
		{
			m_puzPrefix = NULL;
			m_puzURI = NULL;
			m_pNext = NULL;
		}
	
		virtual FINLINE ~F_XMLNamespace()
		{
			flmAssert( !m_pNext);
	
			if( m_puzPrefix)
			{
				f_free( &m_puzPrefix);
			}
	
			if( m_puzURI)
			{
				f_free( &m_puzURI);
			}
		}
	
		RCODE setPrefix(
			FLMUNICODE *		puzPrefix);
	
		RCODE setURI(
			FLMUNICODE *		puzURI);
	
		RCODE setup(
			FLMUNICODE *		puzPrefix,
			FLMUNICODE *		puzURI,
			F_XMLNamespace *	pNext);
	
		FINLINE FLMUNICODE * getPrefixPtr( void)
		{
			return( m_puzPrefix);
		}
	
		FINLINE FLMUNICODE * getURIPtr( void)
		{
			return( m_puzURI);
		}
	
	private:
	
		FLMUNICODE *		m_puzPrefix;
		FLMUNICODE *		m_puzURI;
		F_XMLNamespace *	m_pNext;
	
	friend class F_XMLNamespaceMgr;
	};
	
	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_XMLNamespaceMgr : public F_RefCount, public virtual F_Base
	{
	public:
	
		F_XMLNamespaceMgr();
	
		virtual ~F_XMLNamespaceMgr();
	
		RCODE findNamespace(
			FLMUNICODE *		puzPrefix,
			F_XMLNamespace **	ppNamespace,
			FLMUINT				uiMaxSearchSize = ~((FLMUINT)0));
	
		RCODE pushNamespace(
			FLMUNICODE *		puzPrefix,
			FLMUNICODE *		puzNamespaceURI);
	
		RCODE pushNamespace(
			F_XMLNamespace *	pNamespace);
	
		void popNamespaces(
			FLMUINT				uiCount);
	
		FLMUINT getNamespaceCount( void)
		{
			return( m_uiNamespaceCount);
		}
	
	private:
	
		F_XMLNamespace *			m_pFirstNamespace;
		FLMUINT						m_uiNamespaceCount;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_XMLParser : public F_XML, public F_XMLNamespaceMgr
	{
	public:
	
		F_XMLParser();
	
		~F_XMLParser();
	
		RCODE FLMAPI setup( void);
	
		void FLMAPI reset( void);
	
		RCODE FLMAPI import(
			IF_IStream *				pStream,
			FLMUINT						uiFlags,
			IF_DOMNode *				pNodeToLinkTo,
			eNodeInsertLoc				eInsertLoc,
			IF_DOMNode **				ppNewNode,
			FLM_IMPORT_STATS *		pImportStats);
	
		FINLINE void FLMAPI setStatusCallback(
			XML_STATUS_HOOK			fnStatus,
			void *						pvUserData)
		{
			m_fnStatus = fnStatus;
			m_pvCallbackData = pvUserData;
		}
	
	private:
	
		#define F_DEFAULT_NS_DECL		0x01
		#define F_PREFIXED_NS_DECL		0x02
	
		typedef struct xmlattr
		{
			FLMUINT				uiLineNum;
			FLMUINT				uiLineOffset;
			FLMUINT				uiLineFilePos;	
			FLMUINT				uiLineBytes;
			FLMUINT				uiValueLineNum;
			FLMUINT				uiValueLineOffset;
			FLMUNICODE *		puzPrefix;
			FLMUNICODE *		puzLocalName;
			FLMUNICODE *		puzVal;
			FLMUINT				uiFlags;
			xmlattr *			pPrev;
			xmlattr *			pNext;
		} XML_ATTR;
	
		// Methods
	
		RCODE getFieldTagAndType(
			FLMUNICODE *	puzName,
			FLMBOOL			bOkToAdd,
			FLMUINT *		puiTagNum,
			FLMUINT *		puiDataType);
	
		RCODE getByte(
			FLMBYTE *	pucByte);
			
		FINLINE void ungetByte(
			FLMBYTE	ucByte)
		{
			// Can only unget a single byte.
			
			flmAssert( !m_ucUngetByte);
			m_ucUngetByte = ucByte;
			m_importStats.uiChars--;
		}
			
		RCODE getLine( void);
		
		FINLINE FLMUNICODE getChar( void)
		{
			if (m_uiCurrLineOffset == m_uiCurrLineNumChars)
			{
				return( (FLMUNICODE)0);
			}
			else
			{
				return( m_puzCurrLineBuf [m_uiCurrLineOffset++]);
			}
		}
		
		FINLINE FLMUNICODE peekChar( void)
		{
			if (m_uiCurrLineOffset == m_uiCurrLineNumChars)
			{
				return( (FLMUNICODE)0);
			}
			else
			{
				return( m_puzCurrLineBuf [m_uiCurrLineOffset]);
			}
		}
		
		FINLINE void ungetChar( void)
		{
			flmAssert( m_uiCurrLineOffset);
			m_uiCurrLineOffset--;
		}
		
		RCODE getName(
			FLMUINT *		puiChars);
	
		RCODE getQualifiedName(
			FLMUINT *		puiChars,
			FLMUNICODE **	ppuzPrefix,
			FLMUNICODE **	ppuzLocal,
			FLMBOOL *		pbNamespaceDecl,
			FLMBOOL *		pbDefaultNamespaceDecl);
	
		void getNmtoken(
			FLMUINT *		puiChars);
	
		RCODE getPubidLiteral( void);
	
		RCODE getSystemLiteral( void);
	
		RCODE getElementValue(
			FLMUNICODE *	puBuf,
			FLMUINT *		puiMaxChars,
			FLMBOOL *		pbEntity);
	
		RCODE processEntityValue( void);
	
		RCODE getEntity(
			FLMUNICODE *	puBuf,
			FLMUINT *		puiChars,
			FLMBOOL *		pbTranslated,
			FLMUNICODE *	puTransChar);
	
		RCODE processReference(
			FLMUNICODE *	puChar = NULL);
	
		RCODE processCDATA(
			IF_DOMNode *	pParent,
			FLMUINT			uiSavedLineNum,
			FLMUINT     	uiSavedOffset,
			FLMUINT			uiSavedFilePos,
			FLMUINT			uiSavedLineBytes);
	
		RCODE processAttributeList( void);
	
		RCODE processComment(
			IF_DOMNode *	pParent,
			FLMUINT			uiSavedLineNum,
			FLMUINT     	uiSavedOffset,
			FLMUINT			uiSavedFilePos,
			FLMUINT			uiSavedLineBytes);
	
		RCODE processProlog( void);
	
		RCODE processXMLDecl( void);
	
		RCODE processVersion( void);
	
		RCODE processEncodingDecl( void);
	
		RCODE processSDDecl( void);
	
		RCODE processMisc( void);
	
		RCODE processDocTypeDecl( void);
	
		RCODE processPI(
			IF_DOMNode *	pParent,
			FLMUINT			uiSavedLineNum,
			FLMUINT     	uiSavedOffset,
			FLMUINT			uiSavedFilePos,
			FLMUINT			uiSavedLineBytes);
	
		RCODE processElement(
			IF_DOMNode *		pNodeToLinkTo,
			eNodeInsertLoc		eInsertLoc,
			IF_DOMNode **		ppNewNode);
	
		RCODE unicodeToNumber64(
			FLMUNICODE *		puzVal,
			FLMUINT64 *			pui64Val,
			FLMBOOL *			pbNeg);
	
		RCODE flushElementValue(
			IF_DOMNode *		pParent,
			FLMBYTE *			pucValue,
			FLMUINT				uiValueLen);
	
		RCODE getBinaryVal(
			FLMUINT *			puiLength);
	
		RCODE fixNamingTag(
			IF_DOMNode *		pNode);
	
		FLMBOOL lineHasToken(
			const char *	pszToken);
			
		RCODE processMarkupDecl( void);
	
		RCODE processPERef( void);
	
		RCODE processElementDecl( void);
	
		RCODE processEntityDecl( void);
	
		RCODE processNotationDecl( void);
	
		RCODE processAttListDecl( void);
	
		RCODE processContentSpec( void);
	
		RCODE processMixedContent( void);
	
		RCODE processChildContent( void);
	
		RCODE processAttDef( void);
	
		RCODE processAttType( void);
	
		RCODE processAttValue(
			XML_ATTR *	pAttr);
	
		RCODE processDefaultDecl( void);
	
		RCODE processID(
			FLMBOOL	bPublicId);
	
		RCODE processSTag(
			IF_DOMNode *		pNodeToLinkTo,	
			eNodeInsertLoc		eInsertLoc,
			FLMBOOL *			pbHasContent,
			IF_DOMNode **		ppElement);
	
		RCODE skipWhitespace(
			FLMBOOL	bRequired);
	
		RCODE resizeValBuffer(
			FLMUINT			uiSize);
	
		// Attribute management
	
		void resetAttrList( void)
		{
			m_pFirstAttr = NULL;
			m_pLastAttr = NULL;
			m_attrPool.poolReset( NULL);
		}
	
		RCODE allocAttribute(
			XML_ATTR **		ppAttr)
		{
			XML_ATTR *	pAttr = NULL;
			RCODE			rc = NE_FLM_OK;
	
			if( RC_BAD( rc = m_attrPool.poolCalloc( 
				sizeof( XML_ATTR), (void **)&pAttr)))
			{
				goto Exit;
			}
	
			if( (pAttr->pPrev = m_pLastAttr) == NULL)
			{
				m_pFirstAttr = pAttr;
			}
			else
			{
				m_pLastAttr->pNext = pAttr;
			}
	
			m_pLastAttr = pAttr;
	
		Exit:
	
			*ppAttr = pAttr;
			return( rc);
		}
	
		RCODE setPrefix(
			XML_ATTR *		pAttr,
			FLMUNICODE *	puzPrefix)
		{
			RCODE		rc = NE_FLM_OK;
			FLMUINT	uiStrLen;
	
			if( !puzPrefix)
			{
				pAttr->puzPrefix = NULL;
				goto Exit;
			}
	
			uiStrLen = f_unilen( puzPrefix);
	
			if( RC_BAD( rc = m_attrPool.poolAlloc( 
				sizeof( FLMUNICODE) * (uiStrLen + 1), (void **)&pAttr->puzPrefix)))
			{
				goto Exit;
			}
	
			f_memcpy( pAttr->puzPrefix, puzPrefix, 
				sizeof( FLMUNICODE) * (uiStrLen + 1));
	
		Exit:
	
			return( rc);
		}
	
		RCODE setLocalName(
			XML_ATTR *		pAttr,
			FLMUNICODE *	puzLocalName)
		{
			RCODE		rc = NE_FLM_OK;
			FLMUINT	uiStrLen;
	
			if( !puzLocalName)
			{
				pAttr->puzLocalName = NULL;
				goto Exit;
			}
	
			uiStrLen = f_unilen( puzLocalName);
	
			if( RC_BAD( rc = m_attrPool.poolAlloc( 
				sizeof( FLMUNICODE) * (uiStrLen + 1), 
				(void **)&pAttr->puzLocalName)))
			{
				goto Exit;
			}
	
			f_memcpy( pAttr->puzLocalName, puzLocalName,
				sizeof( FLMUNICODE) * (uiStrLen + 1));
	
		Exit:
	
			return( rc);
		}
	
		RCODE setUnicode(
			XML_ATTR	*		pAttr,
			FLMUNICODE *	puzUnicode)
		{
			RCODE		rc = NE_FLM_OK;
			FLMUINT	uiStrLen;
	
			if( !puzUnicode)
			{
				pAttr->puzVal = NULL;
				goto Exit;
			}
	
			uiStrLen = f_unilen( puzUnicode);
	
			if( RC_BAD( rc = m_attrPool.poolAlloc( 
				sizeof( FLMUNICODE) * (uiStrLen + 1), 
				(void **)&pAttr->puzVal)))
			{
				goto Exit;
			}
	
			f_memcpy( pAttr->puzVal, puzUnicode, 
				sizeof( FLMUNICODE) * (uiStrLen + 1));
	
		Exit:
	
			return( rc);
		}
	
		RCODE addAttributesToElement(
			IF_DOMNode *		pElement);
			
		FINLINE void setErrInfo(
			FLMUINT			uiErrLineNum,
			FLMUINT			uiErrLineOffset,
			XMLParseError	eErrorType,
			FLMUINT			uiErrLineFilePos,
			FLMUINT			uiErrLineBytes)
		{
			m_importStats.uiErrLineNum = uiErrLineNum;
			m_importStats.uiErrLineOffset = uiErrLineOffset;
			m_importStats.eErrorType = eErrorType;
			m_importStats.uiErrLineFilePos = uiErrLineFilePos;
			m_importStats.uiErrLineBytes = uiErrLineBytes;
		}
	
		FLMBYTE						m_ucUngetByte;
		FLMUNICODE *				m_puzCurrLineBuf;
		FLMUINT						m_uiCurrLineBufMaxChars;
		FLMUINT						m_uiCurrLineNumChars;
		FLMUINT						m_uiCurrLineOffset;
		FLMUINT						m_uiCurrLineNum;
		FLMUINT						m_uiCurrLineFilePos;
		FLMUINT						m_uiCurrLineBytes;
	#define FLM_XML_MAX_CHARS		128
		FLMUNICODE					m_uChars[ FLM_XML_MAX_CHARS];
		FLMBOOL						m_bSetup;
		IF_IStream *				m_pStream;
		FLMBYTE *					m_pucValBuf;
		FLMUINT						m_uiValBufSize; // Number of Unicode characters
		FLMUINT						m_uiFlags;
		FLMBOOL						m_bExtendDictionary;
		XMLEncoding					m_eXMLEncoding;
		XML_STATUS_HOOK			m_fnStatus;
		void *						m_pvCallbackData;
		FLM_IMPORT_STATS			m_importStats;
		F_Pool						m_tmpPool;
		XML_ATTR *					m_pFirstAttr;
		XML_ATTR *					m_pLastAttr;
		F_Pool 						m_attrPool;
	};
	
	#define FLM_XML_EXTEND_DICT_FLAG				0x00000001
	#define FLM_XML_COMPRESS_WHITESPACE_FLAG	0x00000002
	#define FLM_XML_TRANSLATE_ESC_FLAG			0x00000004
	
	FINLINE FLMBOOL isXMLNS(
		FLMUNICODE *	puzName)
	{
		return( (puzName [0] == FLM_UNICODE_x || puzName [0] == FLM_UNICODE_X) &&
				  (puzName [1] == FLM_UNICODE_m || puzName [1] == FLM_UNICODE_M) &&
				  (puzName [2] == FLM_UNICODE_l || puzName [2] == FLM_UNICODE_L) &&
				  (puzName [3] == FLM_UNICODE_n || puzName [3] == FLM_UNICODE_N) &&
				  (puzName [4] == FLM_UNICODE_s || puzName [4] == FLM_UNICODE_S)
				  ? TRUE
				  : FALSE);
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
	class F_NameTable : public IF_NameTable, public F_Base
	{
	public:
	
		F_NameTable();
	
		virtual ~F_NameTable();
	
		RCODE FLMAPI setupNameTable( void);
	
		void FLMAPI clearTable(
			FLMUINT					uiPoolBlkSize);
	
		RCODE FLMAPI getNextTagTypeAndNumOrder(
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
	
		RCODE FLMAPI getNextTagTypeAndNameOrder(
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
	
		RCODE FLMAPI getFromTagTypeAndName(
			FLMUINT					uiType,
			const FLMUNICODE *	puzTagName,
			const char *			pszTagName,
			FLMBOOL					bMatchNamespace,
			const FLMUNICODE *	puzNamespace = NULL,
			FLMUINT *				puiTagNum = NULL,
			FLMUINT *				puiDataType = NULL);
	
		RCODE FLMAPI getFromTagTypeAndNum(
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
	
		RCODE FLMAPI addTag(
			FLMUINT					uiType,
			FLMUNICODE *			puzTagName,
			const char *			pszTagName,
			FLMUINT					uiTagNum,
			FLMUINT					uiDataType = 0,
			FLMUNICODE *			puzNamespace = NULL,
			FLMBOOL					bCheckDuplicates = TRUE);
	
		void FLMAPI removeTag(
			FLMUINT	uiType,
			FLMUINT	uiTagNum);
	
		RCODE FLMAPI cloneNameTable(
			IF_NameTable *			pSrcNameTable);
	
		RCODE FLMAPI importFromNameTable(
			IF_NameTable *			pSrcNameTable);
	
		FLMINT FLMAPI AddRef( void);
	
		FLMINT FLMAPI Release( void);
		
	private:
	
		void sortTags( void);
	
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
		FLMUNICODE **				m_ppuzNamespaces;
		FLMUINT						m_uiNamespaceTblSize;
		FLMUINT						m_uiNumNamespaces;
		F_MUTEX						m_hRefMutex;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_Btree : public F_RefCount, public F_Base
	{
	public:
	
		F_Btree( void);
		
		virtual ~F_Btree( void);
	
		RCODE btCreate(
			IF_BlockMgr *				pBlockMgr,
			FLMUINT16					ui16BtreeId,
			FLMBOOL						bCounts,
			FLMBOOL						bData,
			FLMUINT *					puiRootBlkAddr);

	
		RCODE btOpen(
			IF_BlockMgr *				pBlockMgr,
			FLMUINT						uiRootBlkAddr,
			FLMBOOL						bCounts,
			FLMBOOL						bData,
			IF_ResultSetCompare *	pCompare = NULL);
	
		void btClose( void);
	
		RCODE btDeleteTree(
			IF_DeleteStatus *			ifpDeleteStatus);
	
		RCODE btGetBlockChains(
			FLMUINT *					puiBlockChains,
			FLMUINT *					puiNumLevels);
	
		RCODE btRemoveEntry(
			const FLMBYTE *			pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT						uiKeyLen);
	
		RCODE btInsertEntry(
			const FLMBYTE *			pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT						uiKeyLen,
			const FLMBYTE *			pucData,
			FLMUINT						uiDataLen,
			FLMBOOL						bFirst,
			FLMBOOL						bLast,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btReplaceEntry(
			const FLMBYTE *			pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT						uiKeyLen,
			const FLMBYTE *			pucData,
			FLMUINT						uiDataLen,
			FLMBOOL						bFirst,
			FLMBOOL						bLast,
			FLMBOOL						bTruncate = TRUE,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btLocateEntry(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT *					puiKeyLen,
			FLMUINT						uiMatch,
			FLMUINT *					puiPosition = NULL,
			FLMUINT *					puiDataLength = NULL,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btGetEntry(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyLen,
			FLMBYTE *					pucData,
			FLMUINT						uiDataBufSize,
			FLMUINT *					puiDataLen);
	
		RCODE btNextEntry(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT *					puiKeyLen,
			FLMUINT *					puiDataLength = NULL,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btPrevEntry(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT *					puiKeyLen,
			FLMUINT *					puiDataLength = NULL,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btFirstEntry(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT *					puiKeyLen,
			FLMUINT *					puiDataLength = NULL,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btLastEntry(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT *					puiKeyLen,
			FLMUINT *					puiDataLength = NULL,
			FLMUINT32 *					pui32BlkAddr = NULL,
			FLMUINT *					puiOffsetIndex = NULL);
	
		RCODE btSetReadPosition(
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyLen,
			FLMUINT						uiPosition);
	
		RCODE btGetReadPosition(
			FLMUINT *					puiPosition);
	
		RCODE btPositionTo(
			FLMUINT						uiPosition,
			FLMBYTE *					pucKey,
			FLMUINT						uiKeyBufSize,
			FLMUINT *					puiKeyLen);
	
		RCODE btGetPosition(
			FLMUINT *					puiPosition);
	
		RCODE btCheck(
			BTREE_ERR_STRUCT *		pErrStruct);
	
		RCODE btRewind( void);
	
		FINLINE void btRelease( void)
		{
			releaseBlocks( TRUE);
		}
	
		FINLINE void btResetBtree( void)
		{
			releaseBlocks( TRUE);
			m_bSetupForRead = FALSE;
			m_bSetupForWrite = FALSE;
			m_bSetupForReplace = FALSE;
			m_bOrigInDOBlocks = FALSE;
			m_bDataOnlyBlock = FALSE;
			m_ui32PrimaryBlkAddr = 0;
			m_ui32CurBlkAddr = 0;
			m_uiPrimaryOffset = 0;
			m_uiCurOffset = 0;
			m_uiDataLength = 0;
			m_uiPrimaryDataLen = 0;
			m_uiOADataLength = 0;
			m_uiDataRemaining = 0;
			m_uiOADataRemaining = 0;
			m_uiOffsetAtStart = 0;
			m_uiSearchLevel = BH_MAX_LEVELS;
		}
	
		RCODE btComputeCounts(
			F_Btree *					pUntilBtree,
			FLMUINT *					puiBlkCount,
			FLMUINT *					puiKeyCount,
			FLMBOOL *					pbTotalsEstimated,
			FLMUINT						uiAvgBlkFullness);
	
		FINLINE void btSetSearchLevel(
			FLMUINT						uiSearchLevel)
		{
			flmAssert( uiSearchLevel <= BH_MAX_LEVELS);
	
			btResetBtree();
	
			m_uiSearchLevel = uiSearchLevel;
		}
	
		RCODE btMoveBlock(
			FLMUINT32					ui32FromBlkAddr,
			FLMUINT32					ui32ToBlkAddr);
	
		FINLINE FLMBOOL btHasCounts( void)
		{
			return( m_bCounts);
		}
	
		FINLINE FLMBOOL btHasData( void)
		{
			return( m_bData);
		}
	
		FINLINE FLMBOOL btDbIsOpen( void)
		{
			return( m_bOpened);
		}
	
		FINLINE FLMBOOL btIsSetupForRead( void)
		{
			return( m_bSetupForRead);
		}
	
		FINLINE FLMBOOL btIsSetupForWrite( void)
		{
			return( m_bSetupForWrite);
		}
	
		FINLINE FLMBOOL btIsSetupForReplace( void)
		{
			return( m_bSetupForReplace);
		}
		
	private:
	
		RCODE btFreeBlockChain(
			FLMUINT					uiStartAddr,
			FLMUINT					uiBlocksToFree,
			FLMUINT *				puiBlocksFreed,
			FLMUINT *				puiEndAddr,
			IF_DeleteStatus *		ifpDeleteStatus);
			
		FINLINE FLMUINT calcEntrySize(
			FLMUINT					uiBlkType,
			FLMUINT					uiFlags,
			FLMUINT					uiKeyLen,
			FLMUINT					uiDataLen,
			FLMUINT					uiOADataLen)
		{
			switch( uiBlkType)
			{
				case BT_LEAF:
				{
					return( uiKeyLen + 2);
				}
	
				case BT_LEAF_DATA:
				{
					return( 1 +															// Flags
									(uiKeyLen > ONE_BYTE_SIZE ? 2 : 1) +		// KeyLen
									(uiDataLen > ONE_BYTE_SIZE ? 2 : 1) +		// DataLen
									(uiOADataLen &&									// OA DataLen
										(uiFlags & BTE_FLAG_FIRST_ELEMENT) ? 4 : 0) +
									uiKeyLen + uiDataLen);
				}
	
				case BT_NON_LEAF:
				case BT_NON_LEAF_COUNTS:
				{
					return( 4 +															// Child block address
							  (uiBlkType == BT_NON_LEAF_COUNTS ? 4 : 0) +	// Counts
								2 +														// Key length
								uiKeyLen);
				}
			}
	
			return( 0);
		}
	
		RCODE computeCounts(
			F_BTSK *					pFromStack,
			F_BTSK *					pUntilStack,
			FLMUINT *				puiBlockCount,
			FLMUINT *				puiKeyCount,
			FLMBOOL *				pbTotalsEstimated,
			FLMUINT					uiAvgBlkFullness);
	
		RCODE blockCounts(
			F_BTSK *					pStack,
			FLMUINT					uiFirstOffset,
			FLMUINT					uiLastOffset,
			FLMUINT *				puiKeyCount,
			FLMUINT *				puiElementCount);
	
		RCODE getStoredCounts(
			F_BTSK *					pFromStack,
			F_BTSK *					pUntilStack,
			FLMUINT *				puiBlockCount,
			FLMUINT *				puiKeyCount,
			FLMBOOL *				pbTotalsEstimated,
			FLMUINT					uiAvgBlkFullness);
	
		RCODE getCacheBlocks(
			F_BTSK *					pStack1,
			F_BTSK *					pStack2);
	
		FINLINE FLMUINT getAvgKeyCount(
			F_BTSK *					pFromStack,
			F_BTSK *					pUntilStack,
			FLMUINT					uiAvgBlkFullness);
	
		FINLINE FLMUINT getBlkEntryCount(
			FLMBYTE *				pBlk)
		{
			return ((F_BTREE_BLK_HDR *)pBlk)->ui16NumKeys;
		}
	
		FINLINE FLMUINT getBlkAvailSpace(
			FLMBYTE *				pBlk)
		{
			return ((F_BLK_HDR *)pBlk)->ui16BlkBytesAvail;
		}
	
		FLMUINT getEntryKeyLength(
			FLMBYTE *				pucEntry,
			FLMUINT					uiBlockType,
			const FLMBYTE **		ppucKeyRV);
	
		FLMUINT getEntrySize(
			FLMBYTE *				pBlk,
			FLMUINT					uiOffset,
			FLMBYTE **				ppucEntry = NULL);
	
		RCODE calcNewEntrySize(
			FLMUINT					uiKeyLen,
			FLMUINT					uiDataLen,
			FLMUINT *				puiEntrySize,
			FLMBOOL *				pbHaveRoom,
			FLMBOOL *				pbDefragBlk);
	
		RCODE extractEntryData(
			FLMBYTE *				pucKey,
			FLMUINT					uiKeyLen,
			FLMBYTE *				pucBuffer,
			FLMUINT					uiBufSiz,
			FLMUINT *				puiDataLen);
	
		RCODE updateEntry(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			F_ELM_UPD_ACTION		eAction,
			FLMBOOL					bTruncate = TRUE);
	
		RCODE insertEntry(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			F_ELM_UPD_ACTION *	peAction);
	
		RCODE storeEntry(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT					uiOADataLen,
			FLMUINT					uiChildBlkAddr,
			FLMUINT					uiCounts,
			FLMUINT					uiEntrySize,
			FLMBOOL *				pbLastEntry);
	
		RCODE removeEntry(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			FLMBOOL *				pbMoreToRemove,
			F_ELM_UPD_ACTION *	peAction);
	
		RCODE remove(
			FLMBOOL					bDeleteDOBlocks);
	
		RCODE removeRange(
			FLMUINT					uiStartElm,
			FLMUINT					uiEndElm,
			FLMBOOL					bDeleteDOBlocks);
	
		RCODE findEntry(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			FLMUINT					uiMatch,
			FLMUINT *				puiPosition = NULL,
			FLMUINT32 *				pui32BlkAddr = NULL,
			FLMUINT *				puiOffsetIndex = NULL);
	
		RCODE findInBlock(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			FLMUINT					uiMatch,
			FLMUINT *				uiPosition,
			FLMUINT32 *				ui32BlkAddr,
			FLMUINT *				uiOffsetIndex);
	
		RCODE scanBlock(
			F_BTSK *					pStack,
			FLMUINT					uiMatch);
	
		RCODE compareKeys(
			const FLMBYTE *		pucKey1,
			FLMUINT					uiKeyLen1,
			const FLMBYTE *		pucKey2,
			FLMUINT					uiKeyLen2,
			FLMINT *					piCompare);
	
		FINLINE RCODE compareBlkKeys(
			const FLMBYTE *		pucBlockKey,
			FLMUINT					uiBlockKeyLen,
			const FLMBYTE *		pucTargetKey,
			FLMUINT					uiTargetKeyLen,
			FLMINT *					piCompare)
		{
			flmAssert( uiBlockKeyLen);
	
			if( !m_pCompare && uiBlockKeyLen == uiTargetKeyLen)
			{
				*piCompare = f_memcmp( pucBlockKey, pucTargetKey, uiBlockKeyLen);
										
				return( NE_FLM_OK);
			}
	
			return( compareKeys( pucBlockKey, uiBlockKeyLen,
								pucTargetKey, uiTargetKeyLen, piCompare));
		}
	
		RCODE positionToEntry(
			FLMUINT					uiPosition);
	
		RCODE searchBlock(
			F_BTREE_BLK_HDR *		pBlkHdr,
			FLMUINT *				puiPrevCounts,
			FLMUINT					uiPosition,
			FLMUINT *				puiOffset);
	
		RCODE defragmentBlock(
			IF_Block **				ppBlock);
	
		RCODE advanceToNextElement(
			FLMBOOL					bAdvanceStack);
	
		RCODE backupToPrevElement(
			FLMBOOL					bBackupStack);
	
		RCODE replaceEntry(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			F_ELM_UPD_ACTION *	peAction,
			FLMBOOL					bTruncate = TRUE);
	
		RCODE replaceOldEntry(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT					uiOADataLen,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			F_ELM_UPD_ACTION *	peAction,
			FLMBOOL					bTruncate = TRUE);
	
		RCODE replaceByInsert(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			const FLMBYTE *		pucDataValue,
			FLMUINT					uiDataLen,
			FLMUINT					uiOADataLen,
			FLMUINT					uiFlags,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			F_ELM_UPD_ACTION *	peAction);
	
		RCODE replace(
			FLMBYTE *				pucEntry,
			FLMUINT					uiEntrySize,
			FLMBOOL *				pbLastEntry);
	
		RCODE buildAndStoreEntry(
			FLMUINT					uiBlkType,
			FLMUINT					uiFlags,
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			const FLMBYTE *		pucData,
			FLMUINT					uiDataLen,
			FLMUINT					uiOADataLen,
			FLMUINT					uiChildBlkAddr,
			FLMUINT					uiCounts,
			FLMBYTE *				pucBuffer,
			FLMUINT					uiBufferSize,
			FLMUINT *				puiEntrySize);
	
		RCODE moveEntriesToPrevBlk(
			FLMUINT					uiNewEntrySize,
			IF_Block **				ppPrevBlock,
			FLMBOOL *				pbEntriesWereMoved);
	
		RCODE moveEntriesToNextBlk(
			FLMUINT					uiEntrySize,
			FLMBOOL *				pbEntriesWereMoved);
	
		RCODE splitBlock(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT					uiOADataLen,
			FLMUINT 					uiChildBlkAddr,
			FLMUINT					uiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			FLMBOOL *				pbBlockSplit);
	
		RCODE createNewLevel( void);
	
		RCODE storeDataOnlyBlocks(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			FLMBOOL					bSaveKey,
			const FLMBYTE *		pucData,
			FLMUINT					uiDataLen);
	
		RCODE replaceDataOnlyBlocks(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			FLMBOOL					bSaveKey,
			const FLMBYTE *		pucData,
			FLMUINT					uiDataLen,
			FLMBOOL					bLast,
			FLMBOOL					bTruncate = TRUE);
	
		RCODE moveToPrev(
			FLMUINT					uiStart,
			FLMUINT					uiFinish,
			IF_Block **				ppPrevBlock);
	
		RCODE moveToNext(
			FLMUINT					uiStart,
			FLMUINT					uiFinish,
			IF_Block **				ppNextBlock);
	
		RCODE updateParentCounts(
			IF_Block *				pChildBlock,
			IF_Block **				ppParentBlock,
			FLMUINT					uiParentElm);
	
		FLMUINT countKeys(
			FLMBYTE *				pBlk);
	
		FLMUINT countRangeOfKeys(
			F_BTSK *					pFromStack,
			FLMUINT					uiFromOffset,
			FLMUINT					uiUntilOffset);
	
		RCODE moveStackToPrev(
			IF_Block *				pPrevBlock);
	
		RCODE moveStackToNext(
			IF_Block *				pBlock,
			FLMBOOL					bReleaseCurrent = TRUE);
	
		RCODE calcOptimalDataLength(
			FLMUINT					uiKeyLen,
			FLMUINT					uiDataLen,
			FLMUINT					uiBytesAvail,
			FLMUINT *				puiNewDataLen);
	
		// Performs an integrity check on a chain of data-only blocks
		
		RCODE verifyDOBlkChain(
			FLMUINT					uiDOAddr,
			FLMUINT					uiDataLength,
			BTREE_ERR_STRUCT *	localErrStruct);
	
		// Performs a check to verify that the counts in the DB match.
		RCODE verifyCounts(
			BTREE_ERR_STRUCT *	pErrStruct);
	
		void releaseBlocks(
			FLMBOOL					bResetStack);
	
		void releaseBtree( void);
	
		RCODE saveReplaceInfo(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen);
	
		RCODE restoreReplaceInfo(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts);
	
		FINLINE RCODE setReturnKey(
			FLMBYTE *				pucEntry,
			FLMUINT					uiBlockType,
			FLMBYTE *				pucKey,
			FLMUINT *				puiKeyLen,
			FLMUINT					uiKeyBufSize);
	
		RCODE setupReadState(
			F_BLK_HDR *				pBlkHdr,
			FLMBYTE *				pucEntry);
	
		RCODE removeRemainingEntries(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen);
	
		RCODE deleteEmptyBlock( void);
	
		RCODE removeDOBlocks(
			FLMUINT32				ui32OrigDOAddr);
	
		RCODE replaceMultiples(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			const FLMBYTE *		pucDataValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			F_ELM_UPD_ACTION *	peAction);
	
		RCODE replaceMultiNoTruncate(
			const FLMBYTE **		ppucKey,
			FLMUINT *				puiKeyLen,
			const FLMBYTE *		pucDataValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT *				puiChildBlkAddr,
			FLMUINT *				puiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			F_ELM_UPD_ACTION *	peAction);
	
		FINLINE RCODE getNextBlock(
			IF_Block **				ppBlock);
	
		FINLINE RCODE getPrevBlock(
			IF_Block **				ppBlock);
	
		FLMBOOL checkContinuedEntry(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			FLMBOOL *				pbLastElement,
			FLMBYTE *				pucEntry,
			FLMUINT					uiBlkType);
	
		RCODE updateCounts( void);
	
		RCODE storePartialEntry(
			const FLMBYTE *		pucKey,
			FLMUINT					uiKeyLen,
			const FLMBYTE *		pucValue,
			FLMUINT					uiLen,
			FLMUINT					uiFlags,
			FLMUINT					uiChildBlkAddr,
			FLMUINT 					uiCounts,
			const FLMBYTE **		ppucRemainingValue,
			FLMUINT *				puiRemainingLen,
			FLMBOOL					bNewBlock = FALSE);
	
		RCODE mergeBlocks(
			FLMBOOL					bLastEntry,
			FLMBOOL *				pbMergedWithPrev,
			FLMBOOL *				pbMergedWithNext,
			F_ELM_UPD_ACTION *	peAction);
	
		RCODE merge(
			IF_Block **				ppFromBlock,
			IF_Block **				ppToBlock);
	
		RCODE checkDownLinks( void);
	
		RCODE verifyChildLinks(
			IF_Block *				pParentBlock);
	
		RCODE combineEntries(
			F_BTREE_BLK_HDR *		pSrcBlkHdr,
			FLMUINT					uiSrcOffset,
			F_BTREE_BLK_HDR *		pDstBlkHdr,
			FLMUINT					uiDstOffset,
			FLMBOOL *				pbEntriesCombined,
			FLMUINT *				puiEntrySize,
			FLMBYTE *				pucTempBlk);
	
		RCODE moveBtreeBlock(
			FLMUINT32				ui32FromBlkAddr,
			FLMUINT32				ui32ToBlkAddr);
	
		RCODE moveDOBlock(
			FLMUINT32				ui32FromBlkAddr,
			FLMUINT32				ui32ToBlkAddr);
	
		IF_BlockMgr *				m_pBlockMgr;
		F_Pool *						m_pPool;
		FLMUINT						m_uiRootBlkAddr;
		FLMBOOL						m_bCounts;
		FLMBOOL						m_bData;
		FLMBOOL						m_bSetupForRead;
		FLMBOOL						m_bSetupForWrite;
		FLMBOOL						m_bSetupForReplace;
		FLMBOOL						m_bOpened;
		FLMBOOL						m_bDataOnlyBlock;
		FLMBOOL						m_bOrigInDOBlocks;
		FLMBOOL						m_bFirstRead;
		FLMBOOL						m_bStackSetup;
		F_BTSK *						m_pStack;
		BTREE_REPLACE_STRUCT *	m_pReplaceInfo;
		BTREE_REPLACE_STRUCT *	m_pReplaceStruct;
		const FLMBYTE *			m_pucDataPtr;
		IF_Block *					m_pBlock;
		F_BTREE_BLK_HDR *			m_pBlkHdr;
		FLMUINT						m_uiBlockSize;
		FLMUINT						m_uiDefragThreshold;
		FLMUINT						m_uiOverflowThreshold;
		FLMUINT						m_uiStackLevels;
		FLMUINT						m_uiRootLevel;
		FLMUINT						m_uiReplaceLevels;
		FLMUINT						m_uiDataLength;
		FLMUINT						m_uiPrimaryDataLen;
		FLMUINT						m_uiOADataLength;
		FLMUINT						m_uiDataRemaining;
		FLMUINT						m_uiOADataRemaining;
		FLMUINT						m_uiPrimaryOffset;
		FLMUINT						m_uiCurOffset;
		FLMUINT						m_uiSearchLevel;
		FLMUINT						m_uiOffsetAtStart;
		FLMUINT32					m_ui32PrimaryBlkAddr;
		FLMUINT32					m_ui32DOBlkAddr;
		FLMUINT32					m_ui32CurBlkAddr;
		F_BTSK						m_Stack[ BH_MAX_LEVELS];
		IF_ResultSetCompare *	m_pCompare;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_SlabManager : public IF_SlabManager, public F_Base
	{
	public:
	
		F_SlabManager();
	
		virtual ~F_SlabManager();
	
		RCODE FLMAPI setup(
			FLMUINT 				uiPreallocSize);
			
		RCODE FLMAPI allocSlab(
			void **				ppSlab,
			FLMBOOL				bMutexLocked);
			
		void FLMAPI freeSlab(
			void **				ppSlab,
			FLMBOOL				bMutexLocked);
			
		RCODE FLMAPI resize(
			FLMUINT 				uiNumBytes,
			FLMUINT *			puiActualSize = NULL,
			FLMBOOL				bMutexLocked = FALSE);
	
		FINLINE void FLMAPI incrementTotalBytesAllocated(
			FLMUINT					uiCount,
			FLMBOOL					bMutexLocked)
		{
			if( !bMutexLocked)
			{
				lockMutex();
			}
			
			m_uiTotalBytesAllocated += uiCount;	
			
			if( !bMutexLocked)
			{
				unlockMutex();
			}
		}
	
		FINLINE void FLMAPI decrementTotalBytesAllocated(
			FLMUINT					uiCount,
			FLMBOOL					bMutexLocked)
		{
			if( !bMutexLocked)
			{
				lockMutex();
			}
			
			flmAssert( m_uiTotalBytesAllocated >= uiCount);
			m_uiTotalBytesAllocated -= uiCount;	
			
			if( !bMutexLocked)
			{
				unlockMutex();
			}
		}
	
		FINLINE FLMUINT FLMAPI getSlabSize( void)
		{
			return( m_uiSlabSize);
		}
	
		FINLINE FLMUINT FLMAPI getTotalSlabs( void)
		{
			return( m_uiTotalSlabs);
		}
		
		FINLINE void FLMAPI lockMutex( void)
		{
			f_mutexLock( m_hMutex);
		}
		
		FINLINE void FLMAPI unlockMutex( void)
		{
			f_mutexUnlock( m_hMutex);
		}
		
		FINLINE FLMUINT FLMAPI totalBytesAllocated( void)
		{
			return( m_uiTotalBytesAllocated);
		}
	
		FINLINE FLMUINT FLMAPI availSlabs( void)
		{
			return( m_uiAvailSlabs);
		}
		
		void FLMAPI protectSlab(
			void *			pSlab);
	
		void FLMAPI unprotectSlab(
			void *			pSlab);
			
	private:
	
		void freeAllSlabs( void);
		
		void * allocSlabFromSystem( void);
		
		void releaseSlabToSystem(
			void *				pSlab);
	
		RCODE sortSlabList( void);
	
		typedef struct
		{
			void *				pPrev;
			void *				pNext;
		} SLABHEADER;
	
		static FLMINT FLMAPI slabAddrCompareFunc(
			void *		pvBuffer,
			FLMUINT		uiPos1,
			FLMUINT		uiPos2);
	
		static void FLMAPI slabAddrSwapFunc(
			void *		pvBuffer,
			FLMUINT		uiPos1,
			FLMUINT		uiPos2);
		
		F_MUTEX					m_hMutex;
		FLMUINT					m_uiTotalBytesAllocated;
		void *					m_pFirstInSlabList;
		void *					m_pLastInSlabList;
		FLMUINT					m_uiSlabSize;
		FLMUINT					m_uiTotalSlabs;
		FLMUINT					m_uiAvailSlabs;
		FLMUINT					m_uiInUseSlabs;
		FLMUINT					m_uiPreallocSlabs;
	#ifdef FLM_SOLARIS
		int						m_DevZero;
	#endif
	
	friend class F_FixedAlloc;
	};

	/****************************************************************************
	Desc:	Class to provide an efficient means of providing many allocations
			of a fixed size.
	****************************************************************************/
	class F_FixedAlloc : public IF_FixedAlloc, public F_Base
	{
	public:
	
		F_FixedAlloc();
	
		virtual ~F_FixedAlloc();
	
		RCODE FLMAPI setup(
			IF_Relocator *			pRelocator,
			IF_SlabManager *		pSlabManager,
			FLMBOOL					bMemProtect,
			FLMUINT					uiCellSize,
			FLM_SLAB_USAGE *		pUsageStats);
	
		FINLINE void * FLMAPI allocCell(
			IF_Relocator *		pRelocator,
			void *				pvInitialData = NULL,
			FLMUINT				uiDataSize = 0,
			FLMBOOL				bMutexLocked = FALSE)
		{
			void *	pvCell;
			
			flmAssert( pRelocator);
	
			if( !bMutexLocked)
			{
				m_pSlabManager->lockMutex();
			}
			
			if( (pvCell = getCell( pRelocator)) == NULL)
			{
				goto Exit;
			}
			
			if( uiDataSize == sizeof( FLMUINT *))
			{
				*((FLMUINT *)pvCell) = *((FLMUINT *)pvInitialData); 
			}
			else if( uiDataSize)
			{
				f_memcpy( pvCell, pvInitialData, uiDataSize);
			}
			
		Exit:
			
			if( !bMutexLocked)
			{
				m_pSlabManager->unlockMutex();
			}
			
			return( pvCell);
		}
	
		FINLINE void FLMAPI freeCell( 
			void *		ptr,
			FLMBOOL		bMutexLocked)
		{
			freeCell( ptr, bMutexLocked, FALSE, NULL);
		}
	
		void FLMAPI freeUnused( void);
	
		void FLMAPI freeAll( void);
	
		FINLINE FLMUINT FLMAPI getCellSize( void)
		{
			return( m_uiCellSize);
		}
		
		void FLMAPI defragmentMemory( void);
		
		void FLMAPI protectCell(
			void *					pvCell);
		
		void FLMAPI unprotectCell(
			void *					pvCell);
			
	private:
	
		typedef struct Slab
		{
			void *		pvAllocator;
			Slab *		pNext;
			Slab *		pPrev;
			Slab *		pNextSlabWithAvailCells;
			Slab *		pPrevSlabWithAvailCells;
			FLMBYTE *	pLocalAvailCellListHead;
			FLMUINT16	ui16NextNeverUsedCell;
			FLMUINT16	ui16AvailCellCount;
			FLMUINT16	ui16AllocatedCells;
	#ifdef FLM_CACHE_PROTECT	
			FLMUINT32	ui16UnprotectCount;
	#endif
		} SLAB;
	
		typedef struct CELLHEADER
		{
			SLAB *			pContainingSlab;
	#ifdef FLM_DEBUG
			FLMUINT *		puiStack;
	#endif
		} CELLHEADER;
	
		typedef struct CELLHEADER2
		{
			CELLHEADER		cellHeader;
			IF_Relocator *	pRelocator;
		} CELLHEADER2;
	
		typedef struct CellAvailNext
		{
			FLMBYTE *	pNextInList;
	#ifdef FLM_DEBUG
			FLMBYTE		szDebugPattern[ 8];
	#endif
		} CELLAVAILNEXT;
	
	#ifdef FLM_CACHE_PROTECT	
		void protectSlab(
			SLAB *			pSlab,
			FLMBOOL			bMutexLocked);
			
		void unprotectSlab(
			SLAB *			pSlab,
			FLMBOOL			bMutexLocked);
	#endif
			
		void * getCell(
			IF_Relocator *		pRelocator);
	
		SLAB * getAnotherSlab( void);
	
		static FINLINE FLMUINT getAllocAlignedSize(
			FLMUINT		uiAskedForSize)
		{
			return( (uiAskedForSize + FLM_ALLOC_ALIGN) & (~FLM_ALLOC_ALIGN));
		}
	
		void freeSlab( 
			SLAB *			pSlab);
	
		void freeCell(
			void *		pCell,
			FLMBOOL		bMutexLocked,
			FLMBOOL		bFreeIfEmpty,
			FLMBOOL *	pbFreedSlab);
	
	#ifdef FLM_DEBUG
		void testForLeaks( void);
	#endif
	
		FINLINE static FLMINT FLMAPI slabAddrCompareFunc(
			void *		pvBuffer,
			FLMUINT		uiPos1,
			FLMUINT		uiPos2)
		{
			SLAB *		pSlab1 = (((SLAB **)pvBuffer)[ uiPos1]);
			SLAB *		pSlab2 = (((SLAB **)pvBuffer)[ uiPos2]);
	
			flmAssert( pSlab1 != pSlab2);
	
			if( pSlab1 < pSlab2)
			{
				return( -1);
			}
	
			return( 1);
		}
	
		FINLINE static void FLMAPI slabAddrSwapFunc(
			void *		pvBuffer,
			FLMUINT		uiPos1,
			FLMUINT		uiPos2)
		{
			SLAB **		ppSlab1 = &(((SLAB **)pvBuffer)[ uiPos1]);
			SLAB **		ppSlab2 = &(((SLAB **)pvBuffer)[ uiPos2]);
			SLAB *		pTmp;
	
			pTmp = *ppSlab1;
			*ppSlab1 = *ppSlab2;
			*ppSlab2 = pTmp;
		}
	
		IF_SlabManager *		m_pSlabManager;
		SLAB *					m_pFirstSlab;
		SLAB *					m_pLastSlab;
		SLAB *					m_pFirstSlabWithAvailCells;
		SLAB *					m_pLastSlabWithAvailCells;
		IF_Relocator *			m_pRelocator;
		FLMBOOL					m_bAvailListSorted;
		FLMUINT					m_uiSlabsWithAvailCells;
		FLMUINT					m_uiSlabHeaderSize;
		FLMUINT					m_uiCellHeaderSize;
		FLMUINT					m_uiCellSize;
		FLMUINT					m_uiSizeOfCellAndHeader; 
		FLMUINT					m_uiTotalFreeCells;
		FLMUINT					m_uiCellsPerSlab;
		FLMUINT					m_uiSlabSize;
		FLM_SLAB_USAGE *		m_pUsageStats;
	#ifdef FLM_CACHE_PROTECT	
		FLMBOOL					m_bMemProtectionEnabled;
	#endif
		
	friend class F_BufferAlloc;
	friend class F_MultiAlloc;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_BufferAlloc : public IF_BufferAlloc, public F_Base
	{
	public:
	
		F_BufferAlloc()
		{
			f_memset( m_ppAllocators, 0, sizeof( m_ppAllocators));
			m_pSlabManager = NULL;
		}
	
		virtual ~F_BufferAlloc();
	
		RCODE FLMAPI setup(
			IF_SlabManager *		pSlabManager,
			FLMBOOL					bMemProtect,
			FLM_SLAB_USAGE *		pUsageStats);
	
		RCODE FLMAPI allocBuf(
			IF_Relocator *		pRelocator,
			FLMUINT				uiSize,
			void *				pvInitialData,
			FLMUINT				uiDataSize,
			FLMBYTE **			ppucBuffer,
			FLMBOOL *			pbAllocatedOnHeap = NULL);
	
		RCODE FLMAPI reallocBuf(
			IF_Relocator *		pRelocator,
			FLMUINT				uiOldSize,
			FLMUINT				uiNewSize,
			void *				pvInitialData,
			FLMUINT				uiDataSize,
			FLMBYTE **			ppucBuffer,
			FLMBOOL *			pbAllocatedOnHeap = NULL);
	
		void FLMAPI freeBuf(
			FLMUINT				uiSize,
			FLMBYTE **			ppucBuffer);
	
		FLMUINT FLMAPI getTrueSize(
			FLMUINT				uiSize,
			FLMBYTE *			pucBuffer);
	
		void FLMAPI defragmentMemory( void);
		
	private:
	
		IF_FixedAlloc * getAllocator(
			FLMUINT				uiSize);
	
		IF_SlabManager *		m_pSlabManager;
		IF_FixedAlloc *		m_ppAllocators[ NUM_BUF_ALLOCATORS];
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	class F_MultiAlloc : public IF_MultiAlloc, public F_Base
	{
	public:
	
		F_MultiAlloc()
		{
			m_pSlabManager = NULL;
			m_puiCellSizes = NULL;
			m_ppAllocators = NULL;
		}
	
		~F_MultiAlloc()
		{
			cleanup();
		}
	
		RCODE FLMAPI setup(
			F_SlabManager *		pSlabManager,
			FLMBOOL					bMemProtect,
			FLMUINT *				puiCellSizes,
			FLM_SLAB_USAGE *		pUsageStats);
	
		RCODE FLMAPI allocBuf(
			IF_Relocator *			pRelocator,
			FLMUINT					uiSize,
			FLMBYTE **				ppucBuffer,
			FLMBOOL					bMutexLocked = FALSE);
	
		RCODE FLMAPI reallocBuf(
			IF_Relocator *			pRelocator,
			FLMUINT					uiNewSize,
			FLMBYTE **				ppucBuffer,
			FLMBOOL					bMutexLocked = FALSE);
	
		FINLINE void FLMAPI freeBuf(
			FLMBYTE **				ppucBuffer)
		{
			if( ppucBuffer && *ppucBuffer)
			{
				getAllocator( *ppucBuffer)->freeCell( *ppucBuffer, FALSE);
				*ppucBuffer = NULL;
			}
		}
	
		void FLMAPI defragmentMemory( void);
	
		FINLINE FLMUINT FLMAPI getTrueSize(
			FLMBYTE *				pucBuffer)
		{
			return( getAllocator( pucBuffer)->getCellSize());
		}
	
		void FLMAPI protectBuffer(
			void *					pvBuffer,
			FLMBOOL					bMutexLocked = FALSE);
	
		void FLMAPI unprotectBuffer(
			void *					pvBuffer,
			FLMBOOL					bMutexLocked = FALSE);
	
		FINLINE void FLMAPI lockMutex( void)
		{
			m_pSlabManager->lockMutex();
		}
	
		FINLINE void FLMAPI unlockMutex( void)
		{
			m_pSlabManager->unlockMutex();
		}
			
	private:
	
		IF_FixedAlloc * getAllocator(
			FLMUINT					uiSize);
	
		IF_FixedAlloc * getAllocator(
			FLMBYTE *				pucBuffer);
	
		void cleanup( void);
	
		IF_SlabManager *			m_pSlabManager;
		FLMUINT *					m_puiCellSizes;
		IF_FixedAlloc **			m_ppAllocators;
	};

	/****************************************************************************
	Desc:
	****************************************************************************/
	FLMUINT flmGetFSBlockSize(
		FLMBYTE *			pszFileName);
		
	#if defined( FLM_LINUX)
		void flmGetLinuxKernelVersion(
			FLMUINT *		puiMajor,
			FLMUINT *		puiMinor,
			FLMUINT *		puiRevision);
			
		FLMUINT flmGetLinuxMaxFileSize( void);
		
		void flmGetLinuxMemInfo(
			FLMUINT64 *		pui64TotalMem,
			FLMUINT64 *		pui64AvailMem);
	#endif

#endif	// FTKSYS_H
