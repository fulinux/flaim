//------------------------------------------------------------------------------
// Desc:	Header file for the B-Tree pool
//
// Tabs:	3
//
//		Copyright (c) 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: f_btpool.h 3108 2006-01-19 13:05:19 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#ifndef F_BTPOOL_H
#define F_BTPOOL_H

#include "f_btree.h"

class F_BtPool : public F_Object
{
public:
	F_BtPool( void)
	{
		m_pBtreeList = NULL;
		m_hMutex = F_MUTEX_NULL;
		m_bInitialized = FALSE;
	}

	~F_BtPool( void)
	{
		while (m_pBtreeList)
		{
			F_Btree *	pBtree;

			pBtree = m_pBtreeList;
			m_pBtreeList = m_pBtreeList->m_pNext;

			pBtree->Release();
		}

		if (m_hMutex != F_MUTEX_NULL)
		{
			f_mutexDestroy( &m_hMutex);
		}

		m_bInitialized = FALSE;
	}

	RCODE btpInit( void);

	RCODE btpReserveBtree(
		F_Btree **		ppBtree);

	void btpReturnBtree(
		F_Btree **		ppBtree);

private:

	F_Btree *		m_pBtreeList;
	F_MUTEX			m_hMutex;
	FLMBOOL			m_bInitialized;
};

#endif



