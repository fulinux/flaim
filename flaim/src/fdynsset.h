//-------------------------------------------------------------------------
// Desc:	Dynamic result set - class definitions.
// Tabs:	3
//
//		Copyright (c) 1998-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fdynsset.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FDYNSSET_H
#define FDYNSSET_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

class FDynSearchSet;
class	FHashBlk;
class FBtreeBlk;
class	FBtreeRoot;
class	FBtreeNonLeaf;
class	FBtreeLeaf;

// A block size of 8K will perform well in minimizing the number of reads
// to obtain a block.  A 6K may perform better if the file is located
// across the network.

#define	DYNSSET_BLOCK_SIZE					0x4000
#define	DYNSSET_HASH_BUFFER_SIZE			0x2000
#define	DYNSSET_MIN_FIXED_ENTRY_SIZE		4

// Change ucZeros in fdynsset.cpp if this changes.

#define	DYNSSET_MAX_FIXED_ENTRY_SIZE		32
#define	DYNSSET_POSITION_NOT_SET			0xFFFFFFFF

#define	FBTREE_CACHE_BLKS			32
#define	FBTREE_END					0xFFFFFFFF
#define	FBTREE_MAX_LEVELS			4

// Default is to use the memcmp() function.

typedef int (* FDYNSET_COMPARE_FUNC_p)(
	const void *	vpData1,
	const void *	vpData2,
	size_t			UserValue);

typedef struct 
{
	FLMUINT		uiBlkAddr;
	FLMUINT		uiPrevBlkAddr;
	FLMUINT		uiNextBlkAddr;
	FLMUINT		uiLEMAddr;
	FLMUINT		uiNumEntries;
} FixedBlkHdr;

enum eBlkTypes
{
	ACCESS_HASH, 
	ACCESS_BTREE_LEAF, 
	ACCESS_BTREE_ROOT, 
	ACCESS_BTREE_NON_LEAF
};

typedef enum eBlkTypes FBlkTypes;

int DRNCompareFunc(
	const void *	vpData1,
	const void *	vpData2,
	size_t			UserValue);

/****************************************************************************
Desc:
****************************************************************************/
class	FFixedBlk : public F_Base
{
public:

	FFixedBlk();
	
	virtual ~FFixedBlk()
	{
	}

	FBlkTypes blkType() 
	{ 
		return m_eBlkType; 
	}

	virtual RCODE getCurrent(		// SUCCESS or FRC_NOT_FOUND
		void *		vpEntryBuffer) = 0;

	virtual RCODE getFirst(		// SUCCESS or FRC_EOF_HIT
		void *		vpEntryBuffer) = 0;

	virtual RCODE getLast(			// SUCCESS or FRC_EOF_HIT
		void *		vpEntryBuffer) = 0;

	virtual RCODE getNext(			// SUCCESS or FRC_EOF_HIT
		void *		vpEntryBuffer) = 0;

	virtual FLMUINT getTotalEntries() = 0;	// Total entries

	virtual RCODE insert(			// SUCCESS, FRC_FAILURE if full
		void *		vpEntry) = 0;

	FLMBOOL isDirty( void)
	{
		return( m_bDirty);
	}

	virtual RCODE search(
		void *		vpEntry,
		void *		vpFoundEntry = NULL) = 0;

	void	setCompareFunc(
		FDYNSET_COMPARE_FUNC_p	fnCompare,
		void *						UserValue) 
	{
		m_fnCompare = fnCompare;
		m_UserValue = UserValue;
	}

protected:

	FDYNSET_COMPARE_FUNC_p 		m_fnCompare;
	void *							m_UserValue;
	FBlkTypes						m_eBlkType;
	FLMUINT							m_uiEntrySize;
	FLMUINT							m_uiNumSlots;
	FLMUINT							m_uiPosition;
	FLMBOOL							m_bDirty;
	FLMBYTE *						m_pucBlkBuf;
};

/*****************************************************************************
Desc:
*****************************************************************************/
class FDynSearchSet : public F_Base
{
public:
	
	FDynSearchSet();
	
	virtual ~FDynSearchSet()
	{
		if( m_Access)
		{
			m_Access->Release();
		}
	}
	
	RCODE setup(
		const char *	pszIoPath,
		FLMUINT			uiEntrySize);

	FINLINE void setCompareFunc(
		FDYNSET_COMPARE_FUNC_p	fnCompare,
		void *   					UserValue)
	{
		m_fnCompare = fnCompare;
		m_UserValue = UserValue;
		m_Access->setCompareFunc( fnCompare, UserValue);
	}

	RCODE addEntry(
		void *		vpEntry);

	FINLINE RCODE findMatch(
		void *			vpEntry,
		void *			vpFoundEntry)
	{
		return m_Access->search( vpEntry, vpFoundEntry);
	}
	
	FINLINE FLMUINT getEntrySize()
	{ 
		return( m_uiEntrySize);
	}

	FINLINE FLMUINT getTotalEntries( void)
	{ 
		return( m_Access->getTotalEntries());
	}

private:

	FDYNSET_COMPARE_FUNC_p	m_fnCompare;
	void *						m_UserValue;
	FLMUINT						m_uiEntrySize;
	FFixedBlk *					m_Access;
	char							m_szFilePath[ F_PATH_MAX_SIZE];
};					

/****************************************************************************
Desc:
****************************************************************************/
class	FHashBlk : public FFixedBlk
{
public:

	FINLINE FHashBlk()
	{
		m_eBlkType = ACCESS_HASH;
		m_pucBlkBuf = m_ucHashBlk;
		f_memset( m_ucHashBlk, 0, sizeof( m_ucHashBlk));
		m_uiTotalEntries = 0;
	}
	
	FINLINE ~FHashBlk()
	{
		m_pucBlkBuf = NULL;
	}

	FINLINE RCODE setup(
		FLMUINT		uiEntrySize)
	{
		m_uiEntrySize = uiEntrySize;
		m_uiNumSlots = DYNSSET_HASH_BUFFER_SIZE / uiEntrySize;
		return( FERR_OK);
	}

	FINLINE RCODE getCurrent(
		void *			vpEntryBuffer)
	{
		if( m_uiPosition == DYNSSET_POSITION_NOT_SET)
		{
			return( RC_SET( FERR_NOT_FOUND));
		}

		f_memcpy( vpEntryBuffer, &m_pucBlkBuf[ m_uiPosition], m_uiEntrySize);
		return( FERR_OK);
	}
		
	FINLINE RCODE getFirst(
		void *			vpEntryBuffer)
	{
		m_uiPosition = DYNSSET_POSITION_NOT_SET;
		return getNext( vpEntryBuffer);
	}
		
	RCODE getLast( 
		void *		vpEntryBuffer);
		
	RCODE getNext(
		void *		vpEntryBuffer);

	FINLINE FLMUINT getTotalEntries( void) 
	{
		return( m_uiTotalEntries);
	}

	RCODE insert(
		void *		vpEntry);

	RCODE search(
		void *		vpEntry,
		void *		vpFoundEntry = NULL);

private:

	FLMUINT			m_uiTotalEntries;
	FLMBYTE			m_ucHashBlk[ DYNSSET_HASH_BUFFER_SIZE];
};

#define	ENTRY_POS(uiPos)	(m_pucBlkBuf + sizeof( FixedBlkHdr) + \
												(uiPos * (m_uiEntrySize+m_uiEntryOvhd)))

/****************************************************************************
Desc:
****************************************************************************/
class	FBtreeBlk : public FFixedBlk
{
public:

	FBtreeBlk()
	{
	}
	
	virtual ~FBtreeBlk()
	{
		if( m_pucBlkBuf)
		{
			f_free( &m_pucBlkBuf);
		}
	}

	FINLINE RCODE getCurrent(
		void *		vpEntryBuffer)
	{
		if( m_uiPosition == DYNSSET_POSITION_NOT_SET)
		{
			return( RC_SET( FERR_NOT_FOUND));
		}

		f_memcpy( vpEntryBuffer, ENTRY_POS( m_uiPosition), m_uiEntrySize);
		return( FERR_OK);
	}

	FINLINE RCODE getFirst(
		void *			vpEntryBuffer)
	{
		m_uiPosition = DYNSSET_POSITION_NOT_SET;
		return getNext( vpEntryBuffer);
	}
		
	RCODE getLast(
		void *			vpEntryBuffer);
		
	RCODE getNext(
		void *			vpEntryBuffer);

	RCODE readBlk(
		F_FileHdl *		pFileHdl,
		FLMUINT			uiBlkAddr);

	void reset(
		FBlkTypes		blkType);

	RCODE split(
		FBtreeRoot *	pParent,
		FLMBYTE *		pCurEntry,
		FLMUINT			uiCurBlkAddr,
		FLMBYTE *		pucParentEntry,
		FLMUINT *		puiNewBlkAddr);

	RCODE writeBlk(
		F_FileHdl *		pFileHdl);

	virtual FLMUINT getTotalEntries( void) = 0;

	virtual RCODE insert(
		void *			vpEntry) = 0;

	virtual RCODE search(
		void *			vpEntry,
		void *			vpFoundEntry = NULL) = 0;

	virtual RCODE searchEntry(
		void *			vpEntry,
		FLMUINT *		puiChildAddr = NULL,
		void *			vpFoundEntry = NULL);
	
	FINLINE FLMUINT blkAddr()
	{ 
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiBlkAddr); 
	}
	
	FINLINE void blkAddr( 
		FLMUINT			uiBlkAddr)
	{ 
		((FixedBlkHdr *)m_pucBlkBuf)->uiBlkAddr = uiBlkAddr;
		m_bDirty = TRUE;
	}

	FINLINE FLMUINT entryCount( void)
	{ 
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiNumEntries);
	}
	
	FINLINE void entryCount( 
		FLMUINT			uiNumEntries)
	{ 
		((FixedBlkHdr *)m_pucBlkBuf)->uiNumEntries = uiNumEntries;
		m_bDirty = TRUE;
	}

	RCODE insertEntry(
		void *			vpEntry,
		FLMUINT			uiChildAddr = FBTREE_END);

	FINLINE FLMUINT lemBlk( void)
	{ 
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiLEMAddr); 
	}
	
	FINLINE void lemBlk(
		FLMUINT			uiLEMAddr)
	{ 
		((FixedBlkHdr *)m_pucBlkBuf)->uiLEMAddr = uiLEMAddr;
		m_bDirty = TRUE;
	}
	
	FINLINE FLMUINT nextBlk( void)
	{ 
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiNextBlkAddr); 
	}
	
	FINLINE void nextBlk(
		FLMUINT		uiBlkAddr)
	{ 
		((FixedBlkHdr *)m_pucBlkBuf)->uiNextBlkAddr = uiBlkAddr;
		m_bDirty = TRUE;
	}

	FINLINE FLMUINT prevBlk( void)
	{ 
		return( ((FixedBlkHdr *)m_pucBlkBuf)->uiPrevBlkAddr); 
	}
	
	FINLINE void prevBlk( 
		FLMUINT		uiBlkAddr)
	{ 
		((FixedBlkHdr *)m_pucBlkBuf)->uiPrevBlkAddr = uiBlkAddr;
		m_bDirty = TRUE;
	}

protected:

	FLMUINT				m_uiEntryOvhd;
};

/****************************************************************************
Desc:
****************************************************************************/
class	FBtreeLeaf : public FBtreeBlk
{
public:

	FINLINE FBtreeLeaf()
	{
		m_eBlkType = ACCESS_BTREE_LEAF;
		m_uiEntryOvhd = 0;
	}
	
	virtual ~FBtreeLeaf()
	{
	}

	RCODE		setup(
		FLMUINT	uiEntrySize);

	FINLINE FLMUINT getTotalEntries( void) 
	{
		return (FLMUINT) entryCount();
	}

	FINLINE RCODE insert(
		void *	vpEntry)
	{
		return insertEntry( vpEntry, FBTREE_END);
	}

	FINLINE RCODE search(
		void *	vpEntry,
		void *	vpFoundEntry = NULL)
	{ 
		return searchEntry( vpEntry, NULL, vpFoundEntry);
	}

	RCODE split(
		FBtreeRoot *	pNewRoot);
};

typedef struct
{
	FLMUINT			uiBlkAddr;
	FLMUINT			uiLRUValue; 
	FBtreeBlk *		pBlk;
} FBTREE_CACHE;

/****************************************************************************
Desc:
****************************************************************************/
class	FBtreeNonLeaf : public FBtreeBlk
{
public:

	FBtreeNonLeaf()
	{
		m_eBlkType = ACCESS_BTREE_NON_LEAF;
		m_uiEntryOvhd = 4;
	}

	virtual ~FBtreeNonLeaf()
	{
	}
	
	RCODE setup(
		FLMUINT			uiEntrySize);

	FINLINE FLMUINT getTotalEntries( void) 
	{
		return( (FLMUINT) entryCount());
	}

	FINLINE RCODE insert( void *)
	{
		return( FERR_OK);
	}

	FINLINE RCODE search(
		void *	vpEntry,
		void *	vpFoundEntry = NULL)
	{
		F_UNREFERENCED_PARM( vpEntry);
		F_UNREFERENCED_PARM( vpFoundEntry);
		
		flmAssert(0); 
		return( FERR_OK); 
	}
};

/****************************************************************************
Desc:
****************************************************************************/
class	FBtreeRoot : public FBtreeNonLeaf
{
public:

	FBtreeRoot();
	
	virtual ~FBtreeRoot();

	RCODE setup(
		FLMUINT			uiEntrySize,
		const char *	pszIoPath);

	void closeFile( void);

	FLMUINT getTotalEntries( void)
	{
		return m_uiTotalEntries;
	}

	RCODE insert(
		void *	vpEntry);

	RCODE newBlk(
		FBtreeBlk **	ppBlk,
		FBlkTypes		blkType);

	FINLINE FLMUINT newBlkAddr( void)
	{
		return( m_uiNewBlkAddr++);
	}
	
	RCODE newCacheBlk(
		FLMUINT			uiCachePos,
		FBtreeBlk **	ppBlk,
		FBlkTypes		blkType);

	RCODE openFile( void);

	RCODE readBlk(
		FLMUINT			uiBlkAddr,
		FBlkTypes		blkType,
		FBtreeBlk **	ppBlk);

	RCODE search(
		void *			vpEntry,
		void *			vpFoundEntry = NULL);

	RCODE setupTree(
		FLMBYTE *		pMidEntry,
		FBlkTypes		BlkType,
		FBtreeBlk **	ppLeftBlk,
		FBtreeBlk **	ppRightBlk);

	RCODE split(
		void *			vpCurEntry,
		FLMUINT			uiCurChildAddr);

	RCODE writeBlk(
		FLMUINT			uiWritePos);

private:

	FLMUINT			m_uiLevels;
	FLMUINT			m_uiTotalEntries;
	FLMUINT			m_uiNewBlkAddr;
	FLMUINT			m_uiHighestWrittenBlkAddr;
	F_FileHdl *		m_pFileHdl;
	char 				m_szIoPath[ F_PATH_MAX_SIZE];
	FLMUINT			m_uiLRUCount;
	FBTREE_CACHE	m_CacheBlks[ FBTREE_CACHE_BLKS];
	FBtreeBlk *		m_BTStack[ FBTREE_MAX_LEVELS];
};

#include "fpackoff.h"

#endif
