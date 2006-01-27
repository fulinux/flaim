//-------------------------------------------------------------------------
// Desc:	List class definitions.
// Tabs:	3
//
//		Copyright (c) 1997-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flist.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FLIST_H
#define FLIST_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define FLM_ALL_LISTS			0xFFFF

class F_ListMgr : public F_Base
{
public:

	F_ListMgr()
	{
		m_uiLNodeCnt = 0;
		m_pLNodes = NULL;
	}

	virtual ~F_ListMgr()
	{
		(void) ClearList( FLM_ALL_LISTS);
	}

	RCODE Setup(								// Finish the setup operation.
		F_ListNode *	pLNodes,				// LNodes to use
		FLMUINT			uiLNodeCnt);		// Number of lists that this obj will 
													// manage.

	void InsertAtFirst(						// Insert new list item at the first of 
													// the list.
		FLMUINT			uiList,				// Which list to insert this item into
		F_ListItem *	pNewFirstItem);	// New item to be inserted

	void InsertAtEnd(							// Insert the new list item at the end of 
													// list
		FLMUINT			uiList,				// Which list to insert this item into
		F_ListItem *	pNewLastItem);		// New item to be inserted

	F_ListItem * GetItem(					// Retrieve a specific item from the list
		FLMUINT		uiList,					// Which list to get item from
		FLMUINT		nth);						// Which item to retrieve (0 == first)

	void RemoveItem(							// Remove supplied item from the specified list
		FLMUINT				uiList,
		F_ListItem *		pItem);

	FINLINE FLMUINT GetListCount()			// Returns the number of lists that this 
	{												// object manages.
		return m_uiLNodeCnt;
	}

	FLMUINT GetCount(							// Returns the number of items within a list.
		FLMUINT		uiList);

	RCODE ClearList(							// Unlink all items from a specified list.
		FLMUINT		uiList = 0);			// Which list to clear. To clear all lists
													// pass in the FLM_ALL_LISTS define.

private:

	FLMUINT		m_uiLNodeCnt;				// Number of lists (F_ListNode lists) that this 
													// list object is managing.
	F_ListNode *		m_pLNodes;			// The Lists (LNODEs) that this object 
													// manages.
};

#include "fpackoff.h"

#endif
