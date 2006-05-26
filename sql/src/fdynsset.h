//------------------------------------------------------------------------------
// Desc:	FLAIM Dynamic search result set class.
//
// Tabs:	3
//
//		Copyright (c) 1998-2000, 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fdynsset.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef FDYNSSET_H
#define FDYNSSET_H

/*****************************************************************************
*****
** 	Definitions
*****
*****************************************************************************/

/*
A block size of 8K will perform well in minimizing the number of reads
to obtain a block.  A 6K may perform better if the file is located
across the network.
*/

#define	DYNSSET_BLOCK_SIZE					0x4000
#define	DYNSSET_HASH_BUFFER_SIZE			0x2000
#define	DYNSSET_MIN_FIXED_ENTRY_SIZE		4
// Change ucZeros in fdynsset.cpp if this changes.
#define	DYNSSET_MAX_FIXED_ENTRY_SIZE		32
#define	DYNSSET_POSITION_NOT_SET			0xFFFFFFFF

#define	FBTREE_CACHE_BLKS			32
#define	FBTREE_END					0xFFFFFFFF
#define	FBTREE_MAX_LEVELS			4

typedef int (* FDYNSET_COMPARE_FUNC)(
	void *	pvData1,
	void *	pvData2,
	void *	pvUserData);


/*****************************************************************************
*****
** 	Forward References
*****
*****************************************************************************/

class FDynSearchSet;
class	FHashBlk;
class FBtreeBlk;
class	FBtreeRoot;
class	FBtreeNonLeaf;
class	FBtreeLeaf;

enum eBlkTypes
{
	ACCESS_HASH,
	ACCESS_BTREE_LEAF,
	ACCESS_BTREE_ROOT,
	ACCESS_BTREE_NON_LEAF
};
typedef enum eBlkTypes FBlkTypes;


/*===========================================================================
	Virtual FFixedBlk
===========================================================================*/
class	FFixedBlk : public F_Object
{
public:

	FFixedBlk();
	~FFixedBlk()
	{
	}

	/* virtual methods that must be implemented */

	FBlkTypes blkType()
	{
		return m_eBlkType;
	}

	virtual RCODE getCurrent(
		void *	pvEntryBuffer) = 0;

	virtual RCODE getFirst(
		void *	pvEntryBuffer) = 0;

	virtual RCODE getLast(
		void *	pvEntryBuffer) = 0;

	virtual RCODE getNext(
		void *	pvEntryBuffer) = 0;

	virtual FLMUINT getTotalEntries() = 0;

	virtual RCODE insert(
		void *	pvEntry) = 0;

	FINLINE FLMBOOL isDirty( void)
	{
		return( m_bDirty);
	}

	virtual RCODE search(
		void *	pvEntry,
		void *	pvFoundEntry = NULL) = 0;

	void setCompareFunc(
		FDYNSET_COMPARE_FUNC	fnCompare,
		void *					pvUserData)
	{
		m_fnCompare = fnCompare;
		m_pvUserData = pvUserData;
	}

protected:

	// Variables

	FDYNSET_COMPARE_FUNC	m_fnCompare;
	void *					m_pvUserData;
	FBlkTypes				m_eBlkType;
	FLMUINT					m_uiEntrySize;
	FLMUINT					m_uiNumSlots;
	FLMUINT					m_uiPosition;
	FLMBOOL					m_bDirty;
	FLMBYTE *				m_pucBlkBuf;
};


/*****************************************************************************
*****
** 	Result Set Class Definitions
**		Source: fdynsset.cpp
*****
*****************************************************************************/
class FDynSearchSet : public F_Object
{

public:

	FDynSearchSet()
	{
		m_fnCompare = NULL;
		m_pvUserData = NULL;
		m_uiEntrySize = 0;
		m_pAccess = NULL;
	}

	~FDynSearchSet()
	{
		if (m_pAccess)
		{
			m_pAccess->Release();
		}
	}

	RCODE setup(
		char *			pszTmpDir,
		FLMUINT			uiEntrySize);

	FINLINE void setCompareFunc(
		FDYNSET_COMPARE_FUNC	fnCompare,
		void *					pvUserData
		)
	{
		m_fnCompare = fnCompare;
		m_pvUserData = pvUserData;
		m_pAccess->setCompareFunc( fnCompare, pvUserData);
	}

	RCODE addEntry(
		void *	pvEntry);

	FINLINE RCODE findMatch(
		void *	pvMatchEntry,
		void *	pvFoundEntry
		)
	{
		return m_pAccess->search( pvMatchEntry, pvFoundEntry);
	}

	FINLINE FLMUINT getEntrySize( void)
	{
		return m_uiEntrySize;
	}

	FINLINE FLMUINT getTotalEntries( void)
	{
		return m_pAccess->getTotalEntries();
	}

private:

	// Variables

	FDYNSET_COMPARE_FUNC	m_fnCompare;
	void *					m_pvUserData;
	FLMUINT					m_uiEntrySize;
	FFixedBlk *				m_pAccess;
	char						m_szFileName [F_PATH_MAX_SIZE];
};

/*===========================================================================
								Block Header Definition

Desc:		Actually stored as the first section of each block.
			We can write this structure because the same process will
			read the block header i.e. portability is not a problem.
===========================================================================*/

typedef struct
{
	FLMUINT		uiBlkAddr;
	FLMUINT		uiPrevBlkAddr;
	FLMUINT		uiNextBlkAddr;
	FLMUINT		uiLEMAddr;
	FLMUINT		uiNumEntries;
} FixedBlkHdr;

/*===========================================================================
							Fixed Length HASH Access Method
===========================================================================*/
class	FHashBlk : public FFixedBlk
{
public:

	FHashBlk()
	{
		// Base class constructors are called before this constructor is.

		m_eBlkType = ACCESS_HASH;
		m_pucBlkBuf = m_ucHashBlk;
		f_memset( m_ucHashBlk, 0, sizeof( m_ucHashBlk));
		m_uiTotalEntries = 0;
	}

	~FHashBlk()
	{
		// Set to NULL so we don't free the block
		m_pucBlkBuf = NULL;
	}

	FINLINE RCODE setup(
		FLMUINT	uiEntrySize
		)
	{
		m_uiEntrySize = uiEntrySize;
		m_uiNumSlots = DYNSSET_HASH_BUFFER_SIZE / uiEntrySize;
		return( NE_SFLM_OK);
	}

	FINLINE RCODE getCurrent(
		void *	pvEntryBuffer
		)
	{

		// Position to the next/first entry.

		if (m_uiPosition == DYNSSET_POSITION_NOT_SET)
		{
			return RC_SET( NE_SFLM_NOT_FOUND);
		}

		f_memcpy( pvEntryBuffer, &m_pucBlkBuf[ m_uiPosition], m_uiEntrySize);
		return( NE_SFLM_OK);
	}

	FINLINE RCODE getFirst(
		void *	pvEntryBuffer
		)
	{
		m_uiPosition = DYNSSET_POSITION_NOT_SET;
		return( getNext( pvEntryBuffer));
	}

	RCODE getLast(
		void *	pvEntryBuffer);

	RCODE getNext(
		void *	pvEntryBuffer);

	FINLINE FLMUINT getTotalEntries( void)
	{
		return m_uiTotalEntries;
	}

	RCODE insert(
		void *	pvEntry);

	RCODE search(
		void *	pvEntry,
		void *	pvFoundEntry = NULL);

private:

	FLMUINT			m_uiTotalEntries;
	// Allocate the hash block to save 1 allocation.
	// We need to make the hash as fast as possible.
	FLMBYTE			m_ucHashBlk[ DYNSSET_HASH_BUFFER_SIZE];
};


/*===========================================================================
	Virtual FBtreeBlk
===========================================================================*/

// Leaf and non-leaf entry position.  Don't do any ++ or -- ! ! ! !

#define	ENTRY_POS(uiPos)	(m_pucBlkBuf + sizeof( FixedBlkHdr) + \
												(uiPos * (m_uiEntrySize+m_uiEntryOvhd)))

class	FBtreeBlk : public FFixedBlk
{
public:

	FBtreeBlk()
	{
	}

	~FBtreeBlk()
	{
		if (m_pucBlkBuf)
		{
			f_free( &m_pucBlkBuf);
		}
	}

	// virtual methods that must be implemented

	FINLINE RCODE getCurrent(
		void *	pvEntryBuffer
		)
	{
		// Position to the next/first entry.

		if (m_uiPosition == DYNSSET_POSITION_NOT_SET)
		{
			return RC_SET( NE_SFLM_NOT_FOUND);
		}

		f_memcpy( pvEntryBuffer, ENTRY_POS( m_uiPosition), m_uiEntrySize);
		return( NE_SFLM_OK);
	}

	FINLINE RCODE getFirst(
		void *	pvEntryBuffer
		)
	{
		m_uiPosition = DYNSSET_POSITION_NOT_SET;
		return( getNext( pvEntryBuffer));
	}

	RCODE getLast(
		void *	pvEntryBuffer);

	RCODE getNext(
		void *	pvEntryBuffer);

	RCODE readBlk(
		IF_FileHdl *	pFileHdl,
		FLMUINT			uiBlkAddr);

	void reset(
		FBlkTypes	eBlkType);

	RCODE split(
		FBtreeRoot *	pParent,
		FLMBYTE *		pucCurEntry,
		FLMUINT			uiCurBlkAddr,
		FLMBYTE *		pucParentEntry,
		FLMUINT *		puiNewBlkAddr);

	RCODE writeBlk(
		IF_FileHdl *	pFileHdl);

	// Virtual methods

	virtual FLMUINT getTotalEntries( void) = 0;

	virtual RCODE insert(
		void *	pvEntry) = 0;

	virtual RCODE search(
		void *	pvEntry,
		void *	pvFoundEntry = NULL) = 0;

	virtual RCODE searchEntry(
		void *		pvEntry,
		FLMUINT *	puiChildAddr = NULL,
		void *		pvFoundEntry = NULL);

	// Implemented as inline functions.
	// Even though these are b-tree specific - keep them here to
	// avoid having a b-tree block class.  Most are not used for hash.

	FINLINE FLMUINT blkAddr( void)
	{
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiBlkAddr);
	}

	FINLINE void blkAddr(
		FLMUINT	uiBlkAddr)
	{
		((FixedBlkHdr *)m_pucBlkBuf)->uiBlkAddr = uiBlkAddr;
		m_bDirty = TRUE;
	}

	// Get and set the number of entries in the block

	FINLINE FLMUINT entryCount( void)
	{
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiNumEntries);
	}

	FINLINE void entryCount(
		FLMUINT	uiNumEntries)
	{
		((FixedBlkHdr *)m_pucBlkBuf)->uiNumEntries = uiNumEntries;
		m_bDirty = TRUE;
	}

	RCODE insertEntry(
		void *	pvEntry,
		FLMUINT	uiChildAddr = FBTREE_END);


	// Get and set the last element marker in the block.

	FINLINE FLMUINT lemBlk( void)
	{
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiLEMAddr);
	}

	FINLINE void lemBlk(
		FLMUINT	uiLEMAddr)
	{
		((FixedBlkHdr *)m_pucBlkBuf)->uiLEMAddr = uiLEMAddr;
		m_bDirty = TRUE;
	}

	// Get and set the next block address element

	FINLINE FLMUINT nextBlk( void)
	{
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiNextBlkAddr);
	}

	FINLINE void nextBlk(
		FLMUINT	uiBlkAddr)
	{
		((FixedBlkHdr *)m_pucBlkBuf)->uiNextBlkAddr = uiBlkAddr;
		m_bDirty = TRUE;
	}

	// Get and set the previous block address element

	FINLINE FLMUINT prevBlk( void)
	{
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiPrevBlkAddr);
	}

	FINLINE void prevBlk(
		FLMUINT	uiBlkAddr)
	{
		((FixedBlkHdr *)m_pucBlkBuf)->uiPrevBlkAddr = uiBlkAddr;
		m_bDirty = TRUE;
	}

protected:

	// Variables

	FLMUINT				m_uiEntryOvhd;		// Overhead in the entry.
};


/*===========================================================================
							Fixed Length B-tree Leaf - may be a root
===========================================================================*/

class	FBtreeLeaf : public FBtreeBlk
{
public:

	FBtreeLeaf()
	{
		m_eBlkType = ACCESS_BTREE_LEAF;
		m_uiEntryOvhd = 0;
	}

	~FBtreeLeaf()
	{
	}

	RCODE setup(
		FLMUINT	uiEntrySize);

	FINLINE FLMUINT getTotalEntries( void)
	{
		return( (FLMUINT)entryCount());
	}

	FINLINE RCODE insert(
		void *	pvEntry)
	{
		return( insertEntry( pvEntry, FBTREE_END));
	}

	FINLINE RCODE search(
		void *	pvEntry,
		void *	pvFoundEntry = NULL)
	{
		return( searchEntry( pvEntry, NULL, pvFoundEntry));
	}

	RCODE split(
		FBtreeRoot *	pNewRoot);

};

typedef struct
{
	FLMUINT		uiBlkAddr;
	FLMUINT		uiLRUValue;
	FBtreeBlk *	pBlk;				// Points to leaf or non-leaf block
} FBTREE_CACHE;

/*===========================================================================
							Fixed Length B-tree non-leaf block
===========================================================================*/

class	FBtreeNonLeaf : public FBtreeBlk
{
public:

	FBtreeNonLeaf()
	{
		m_eBlkType = ACCESS_BTREE_NON_LEAF;
		m_uiEntryOvhd = sizeof( FLMUINT32);
	}

	~FBtreeNonLeaf()
	{
	}

	RCODE setup(
		FLMUINT	uiEntrySize);

	FINLINE FLMUINT getTotalEntries( void)
	{
		return( (FLMUINT) entryCount());
	}

	FINLINE RCODE insert(
		void *	// pvEntry
		)
	{
		return( NE_SFLM_OK);
	}

	FINLINE RCODE search(
		void *,	// pvEntry,
		void *	pvFoundEntry = NULL
		)
	{
		F_UNREFERENCED_PARM( pvFoundEntry);
		flmAssert( 0);
		return( NE_SFLM_OK);
	}
};

/*===========================================================================
							Fixed Length B-tree Non-Leaf Root
===========================================================================*/
class	FBtreeRoot : public FBtreeNonLeaf
{
public:
	FBtreeRoot();
	~FBtreeRoot();

	RCODE setup(
		FLMUINT		uiEntrySize,
		char *		pszFileName);


	void closeFile( void);

	FINLINE FLMUINT getTotalEntries( void)
	{
		return( m_uiTotalEntries);
	}

	RCODE insert(
		void *	pvEntry);

	RCODE newBlk(
		FBtreeBlk **	ppBlk,
		FBlkTypes		eBlkType);

	FINLINE FLMUINT newBlkAddr( void)
	{
		return( m_uiNewBlkAddr++);
	}

	RCODE newCacheBlk(
		FLMUINT			uiCachePos,
		FBtreeBlk **	ppBlk,
		FBlkTypes		eBlkType);

	RCODE openFile( void);

	RCODE readBlk(
		FLMUINT			uiBlkAddr,
		FBlkTypes		eBlkType,
		FBtreeBlk **	ppBlk);

	RCODE search(
		void *	pvEntry,
		void *	pvFoundEntry = NULL);

	RCODE setupTree(
		FLMBYTE *		pucMidEntry,		// If !NULL entry to insert into root.
		FBlkTypes		eBlkType,			// Leaf or non-leaf
		FBtreeBlk **	ppLeftBlk,			// (out)
		FBtreeBlk **	ppRightBlk);		// (out)

	RCODE split(
		void *		pvCurEntry,
		FLMUINT		uiCurChildAddr);

	RCODE writeBlk(
		FLMUINT		uiWritePos);

private:

	FLMUINT			m_uiLevels;			// Number of levels in the b-tree
	FLMUINT			m_uiTotalEntries;	// Count of total entries
	FLMUINT			m_uiNewBlkAddr;	// Next new blk addr.
	FLMUINT			m_uiHighestWrittenBlkAddr;

	IF_FileHdl *	m_pFileHdl;			// File handle or NULL if not open.
	char *			m_pszFileName;		// File created for result set or default.

	// Cache of 'n' blocks to apply LRU algorithm.

	FLMUINT			m_uiLRUCount;		// Count to find least rec. used.
	FBTREE_CACHE	m_CacheBlks[ FBTREE_CACHE_BLKS];

	// B-tree stack.
	FBtreeBlk *		m_BTStack[ FBTREE_MAX_LEVELS];
};

#endif		// ifndef FRSET_H
