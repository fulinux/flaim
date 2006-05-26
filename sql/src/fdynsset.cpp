//------------------------------------------------------------------------------
// Desc:	Contains the methods for FDynSearchSet class.
//
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
// $Id: fdynsset.cpp 3112 2006-01-19 13:12:40 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"
#include "fdynsset.h"

#define HASH_POS(vp) \
			(((FLMUINT)(FB2UD((FLMBYTE*)vp)) % m_uiNumSlots) * m_uiEntrySize)

static const FLMBYTE ucZeros [ DYNSSET_MAX_FIXED_ENTRY_SIZE ] = {
	0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,
	0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0
};

/****************************************************************************
Desc:	Setup the result set with input values. This method must be
		called and only called once.
****************************************************************************/
RCODE FDynSearchSet::setup(
	char *	pszTmpDir,
	FLMUINT	uiEntrySize)
{
	RCODE			rc = NE_SFLM_OK;
	FHashBlk *	pHashBlk;

	// Set the input variables.

	if( pszTmpDir )
	{
		f_strcpy( m_szFileName, pszTmpDir);				// Dest <- src
	}
	else
	{
		f_memset( m_szFileName, 0, F_PATH_MAX_SIZE);
	}
	m_uiEntrySize = uiEntrySize;

	if ((pHashBlk = f_new FHashBlk) == NULL)
	{
		rc = RC_SET( NE_SFLM_MEM);
		goto Exit;
	}
	pHashBlk->setup( uiEntrySize);
	m_pAccess = (FFixedBlk *) pHashBlk;
	m_pvUserData = (void *) uiEntrySize;

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Add a fixed length entry to the dynamic search result set.
****************************************************************************/
RCODE FDynSearchSet::addEntry(
	void *		pvEntry)
{
	RCODE	rc = NE_SFLM_OK;

Add_Again:

	if (RC_OK( rc = m_pAccess->search( pvEntry)))
	{
		rc = RC_SET( NE_SFLM_EXISTS);
	}
	else if (rc == NE_SFLM_NOT_FOUND)
	{

		// Insert the entry.

		if ((rc = m_pAccess->insert( pvEntry)) == NE_SFLM_FAILURE)
		{
			// Find the type of access method implemented

			if (m_pAccess->blkType() == ACCESS_HASH)
			{
				FBtreeLeaf *	pBtreeBlk;
				FLMBYTE			ucEntryBuffer[ DYNSSET_MAX_FIXED_ENTRY_SIZE];

				// Go from a hash to a b-tree object. Alloc and move stuff over.

				if ((pBtreeBlk = f_new FBtreeLeaf) == NULL)
				{
					rc = RC_SET( NE_SFLM_MEM);
					goto Exit;
				}
				pBtreeBlk->setup( m_uiEntrySize);
				pBtreeBlk->setCompareFunc( m_fnCompare, m_pvUserData);
				for( rc = m_pAccess->getFirst( ucEntryBuffer ); 
						RC_OK(rc);
						rc = m_pAccess->getNext( ucEntryBuffer) )
				{
					// Call search to setup for insert.
					(void) pBtreeBlk->search( ucEntryBuffer);
					if (RC_BAD( rc = pBtreeBlk->insert( ucEntryBuffer)))
					{
						pBtreeBlk->Release();
						goto Exit;
					}
				}
				rc = NE_SFLM_OK;
				m_pAccess->Release();
				m_pAccess = pBtreeBlk;
				goto Add_Again;
			}
			else if( m_pAccess->blkType() == ACCESS_BTREE_LEAF)
			{
				FBtreeRoot *	pFullBtree;

				// Go from 1 block to 3 changing root blocks and free m_pAccess
				// All new splits will be taken care of automatically.

				if ((pFullBtree = f_new FBtreeRoot) == NULL)
				{
					rc = RC_SET( NE_SFLM_MEM);
					goto Exit;
				}
				if( RC_BAD( rc = pFullBtree->setup( m_uiEntrySize, m_szFileName)))
				{
					pFullBtree->Release();
					goto Exit;
				}
				pFullBtree->setCompareFunc( m_fnCompare, m_pvUserData);
				if (RC_BAD( rc = ((FBtreeLeaf *)m_pAccess)->split( pFullBtree)))
				{
					goto Exit;
				}
				m_pAccess->Release();
				m_pAccess = pFullBtree;
				goto Add_Again;
			}
			else
			{
				rc = RC_SET_AND_ASSERT( NE_SFLM_FAILURE);
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Find matching entry.  Position for Get* or for insert.
****************************************************************************/
RCODE FHashBlk::search(
	void *	pvEntry,
	void *	pvFoundEntry)
{
	RCODE			rc = NE_SFLM_OK;
	FLMUINT		uiHashPos = HASH_POS( pvEntry);
	FLMINT		iCompare;

	for (;;)
	{

		// If all zeros then setup to insert at this position.

		if (!f_memcmp( &m_pucBlkBuf[ uiHashPos], ucZeros, m_uiEntrySize))
		{
			rc = RC_SET( NE_SFLM_NOT_FOUND);
			goto Exit;
		}

		if (m_fnCompare)
		{
			iCompare = m_fnCompare( pvEntry, &m_pucBlkBuf[ uiHashPos], 
								m_pvUserData);
		}
		else
		{
			iCompare = f_memcmp( pvEntry, &m_pucBlkBuf[ uiHashPos],
								m_uiEntrySize);
		}

		if (iCompare == 0)
		{

			// Found match.

			if (pvFoundEntry)
			{
				f_memcpy( pvFoundEntry, &m_pucBlkBuf[ uiHashPos], m_uiEntrySize);
			}
			break;
		}

		// Go to the next entry

		uiHashPos += m_uiEntrySize;
		if (uiHashPos >= DYNSSET_HASH_BUFFER_SIZE)
		{
			uiHashPos = 0;
		}
	}

Exit:

	m_uiPosition = uiHashPos;
	return( rc);
}

/****************************************************************************
Desc:	Insert the entry into the buffer.
****************************************************************************/
RCODE FHashBlk::insert(
	void *		pvEntry)
{
	RCODE	rc = NE_SFLM_OK;

	if( getTotalEntries() > ((m_uiNumSlots * 7) / 10))
	{
		rc = RC_SET( NE_SFLM_FAILURE);
		goto Exit;
	}

	f_memcpy( &m_pucBlkBuf[ m_uiPosition], pvEntry, m_uiEntrySize);
	m_uiTotalEntries++;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Return the next entry in the result set.  If the result set
			is not positioned then the first entry will be returned.
****************************************************************************/
RCODE FHashBlk::getNext(
	void *	pvEntryBuffer)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiHashPos;

	// Position to the next/first entry.

	if (m_uiPosition == DYNSSET_POSITION_NOT_SET)
	{
		uiHashPos = 0;
	}
	else
	{
		uiHashPos = m_uiPosition + m_uiEntrySize;
	}

	for ( ; ; uiHashPos += m_uiEntrySize)
	{
		if (uiHashPos >= DYNSSET_HASH_BUFFER_SIZE)
		{
			rc = RC_SET( NE_SFLM_EOF_HIT);
			goto Exit;
		}

		// If all zeros then setup to insert at this position.

		if (f_memcmp( &m_pucBlkBuf[ uiHashPos], ucZeros, m_uiEntrySize))
		{
			f_memcpy( pvEntryBuffer, &m_pucBlkBuf[ uiHashPos], m_uiEntrySize);
			m_uiPosition = uiHashPos;
			goto Exit;
		}
	}

Exit:

	return rc;
}

/****************************************************************************
Desc:	Returns the last entry in the result set.
****************************************************************************/
RCODE FHashBlk::getLast(
	void *			pvEntryBuffer)
{
	RCODE		rc = NE_SFLM_OK;
	FLMUINT	uiHashPos;

	// Position to the next/first entry.

	uiHashPos = DYNSSET_HASH_BUFFER_SIZE;

	for( ; ; )
	{
		uiHashPos -= m_uiEntrySize;

		// If all zeros then setup to insert at this position.

		if (f_memcmp( &m_pucBlkBuf[ uiHashPos], ucZeros, m_uiEntrySize))
		{
			f_memcpy( pvEntryBuffer, &m_pucBlkBuf[ uiHashPos], m_uiEntrySize);
			m_uiPosition = uiHashPos;
			goto Exit;
		}
		if (uiHashPos == 0)
		{
			rc = RC_SET( NE_SFLM_EOF_HIT);
			goto Exit;
		}
	}

Exit:

	return( rc);
}
