//-------------------------------------------------------------------------
// Desc:	Dynamic result set - high level and hash implementation.
// Tabs:	3
//
//		Copyright (c) 1998-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fdynsset.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define	HASH_POS(vp)	(((FLMUINT)(FB2UD((FLMBYTE*)vp)) % m_uiNumSlots) * m_uiEntrySize)

static const FLMBYTE ucZeros [ DYNSSET_MAX_FIXED_ENTRY_SIZE ] = {
	0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
	0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0
};

/****************************************************************************
Desc:
****************************************************************************/
FDynSearchSet::FDynSearchSet()
{
	// Let's just initialize all member variables.

	m_fnCompare = NULL;
	m_UserValue = (void *) 4;
	m_uiEntrySize = 4;
	m_Access = NULL;
}

/****************************************************************************
Desc:    Setup the result set with input values.
			This method must be called and only called once.
****************************************************************************/
RCODE FDynSearchSet::setup(
	const char *	pszIoPath,
	FLMUINT			uiEntrySize)
{
	RCODE			rc = FERR_OK;
	FHashBlk *	pHashBlk;

	if( pszIoPath )
	{
		f_strcpy( m_szFilePath, pszIoPath);
	}
	else
	{
		f_memset( m_szFilePath, 0, F_PATH_MAX_SIZE);
	}
	
	m_uiEntrySize = uiEntrySize;

	if( (pHashBlk = f_new FHashBlk) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}
	
	pHashBlk->setup( uiEntrySize);
	m_Access = (FFixedBlk *) pHashBlk;
	m_UserValue = (void *) uiEntrySize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Add a fixed length entry to the dynamic search result set.
Notes:	This code will not work in UNIX land because of alignment issues.
****************************************************************************/
RCODE FDynSearchSet::addEntry(
	void *		vpEntry)
{
	RCODE			rc;

Add_Again:

	if( RC_OK( rc = m_Access->search( vpEntry)))
	{
		rc = RC_SET( FERR_EXISTS);
	}
	else if( rc == FERR_NOT_FOUND)
	{

		// Insert the entry.
		if( (rc = m_Access->insert( vpEntry)) == FERR_FAILURE)
		{
			// Find the type of access method implemented
			if( m_Access->blkType() == ACCESS_HASH)
			{
				FBtreeLeaf *	pBtreeBlk;
				FLMBYTE			ucEntryBuffer[ DYNSSET_MAX_FIXED_ENTRY_SIZE];

				// Go from a hash to a b-tree object. Alloc and move stuff over.

				if( (pBtreeBlk = f_new FBtreeLeaf) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
				pBtreeBlk->setup( m_uiEntrySize);
				pBtreeBlk->setCompareFunc( m_fnCompare, m_UserValue);
				for( rc = m_Access->getFirst( ucEntryBuffer ); 
						RC_OK(rc);
						rc = m_Access->getNext( ucEntryBuffer) )
				{
					// Call search to setup for insert.
					(void) pBtreeBlk->search( ucEntryBuffer);
					if( RC_BAD( rc = pBtreeBlk->insert( ucEntryBuffer)))
					{
						pBtreeBlk->Release();
						goto Exit;
					}
				}
				rc = FERR_OK;
				m_Access->Release();
				m_Access = pBtreeBlk;
				goto Add_Again;
			}
			else if( m_Access->blkType() == ACCESS_BTREE_LEAF)
			{
				FBtreeRoot *	pFullBtree;

				// Go from 1 block to 3 changing root blocks and free m_Access
				// All new splits will be taken care of automatically.

				if( (pFullBtree = f_new FBtreeRoot) == NULL)
				{
					rc = RC_SET( FERR_MEM);
					goto Exit;
				}
				
				if( RC_BAD( rc = pFullBtree->setup( m_uiEntrySize, m_szFilePath)))
				{
					pFullBtree->Release();
					goto Exit;
				}
				
				pFullBtree->setCompareFunc( m_fnCompare, m_UserValue);
				
				if( RC_BAD( rc = ((FBtreeLeaf *)m_Access)->split( pFullBtree)))
				{
					goto Exit;
				}
				
				m_Access->Release();
				m_Access = pFullBtree;
				goto Add_Again;
			}
			else
			{
				flmAssert(0);
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Find matching entry.  Position for Get* or for insert.
****************************************************************************/
RCODE FHashBlk::search(
	void *		vpEntry,
	void *		vpFoundEntry)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiHashPos = HASH_POS( vpEntry);
	FLMINT		iCompare;

	for(;;)
	{
		// If all zeros then setup to insert at this position.
		if( !f_memcmp( &m_pucBlkBuf[ uiHashPos], ucZeros, m_uiEntrySize))
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}

		if( m_fnCompare)
		{
			iCompare = m_fnCompare( vpEntry, &m_pucBlkBuf[ uiHashPos], 
								(size_t) m_UserValue);
		}
		else
		{
			iCompare = f_memcmp( vpEntry, &m_pucBlkBuf[ uiHashPos],
								(size_t) m_UserValue);
		}

		if( !iCompare)
		{
			// Found match.
			if( vpFoundEntry)
			{
				f_memcpy( vpFoundEntry, &m_pucBlkBuf[ uiHashPos], m_uiEntrySize);
			}
			break;
		}

		// Go to the next entry
		uiHashPos += m_uiEntrySize;
		if( uiHashPos >= DYNSSET_HASH_BUFFER_SIZE)
			uiHashPos = 0;
	}

Exit:
	m_uiPosition = uiHashPos;
	return( rc);
}

/****************************************************************************
Desc:		Insert the entry into the buffer.
****************************************************************************/
RCODE FHashBlk::insert(
	void *		vpEntry)
{
	RCODE			rc = FERR_OK;

	if( getTotalEntries() > ((m_uiNumSlots * 7) / 10))
	{
		rc = RC_SET( FERR_FAILURE);
		goto Exit;
	}

	f_memcpy( &m_pucBlkBuf[ m_uiPosition], vpEntry, m_uiEntrySize);
	m_uiTotalEntries++;

Exit:
	return( rc);
}

/****************************************************************************
Desc:		Return the next entry in the result set.  If the result set
			is not positioned then the first entry will be returned.
****************************************************************************/
RCODE FHashBlk::getNext(
	void *			vpEntryBuffer)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiHashPos;

	// Position to the next/first entry.
	
	if( m_uiPosition == DYNSSET_POSITION_NOT_SET)
	{
		uiHashPos = 0;
	}
	else
	{
		uiHashPos = m_uiPosition + m_uiEntrySize;
	}

	for( ; ; uiHashPos += m_uiEntrySize)
	{
		if( uiHashPos >= DYNSSET_HASH_BUFFER_SIZE)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}

		// If all zeros then setup to insert at this position.
		if( f_memcmp( &m_pucBlkBuf[ uiHashPos], ucZeros, m_uiEntrySize))
		{
			f_memcpy( vpEntryBuffer, &m_pucBlkBuf[ uiHashPos], m_uiEntrySize);
			m_uiPosition = uiHashPos;
			goto Exit;
		}
	}
Exit:
	return( rc);
}

/****************************************************************************
Desc:		Returns the last entry in the result set.  Not implemented.
****************************************************************************/
RCODE FHashBlk::getLast(
	void *			vpEntryBuffer)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiHashPos;

	// Position to the next/first entry.
	uiHashPos = DYNSSET_HASH_BUFFER_SIZE;

	for( ; ; )
	{
		uiHashPos -= m_uiEntrySize;

			// If all zeros then setup to insert at this position.
		if( f_memcmp( &m_pucBlkBuf[ uiHashPos], ucZeros, m_uiEntrySize))
		{
			f_memcpy( vpEntryBuffer, &m_pucBlkBuf[ uiHashPos], m_uiEntrySize);
			m_uiPosition = uiHashPos;
			goto Exit;
		}
		if( uiHashPos == 0)
		{
			rc = RC_SET( FERR_EOF_HIT);
			goto Exit;
		}
	}
Exit:
	return( rc);
}


int DRNCompareFunc(
	const void *	vpData1,
	const void *	vpData2,
	size_t			UserValue)
{
	F_UNREFERENCED_PARM(UserValue);
	if( *((FLMUINT *)vpData1) < *((FLMUINT *)vpData2))
		return -1;
	else if( *((FLMUINT *)vpData1) > *((FLMUINT *)vpData2))
		return 1;
	// else
	return 0;
}
