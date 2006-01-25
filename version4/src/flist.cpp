//-------------------------------------------------------------------------
// Desc:	List class.
// Tabs:	3
//
//		Copyright (c) 1997-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flist.cpp 12263 2006-01-19 14:43:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/***************************************************************************
*																									*
*								     F_ListMgr Class 	         						*
*																									*
****************************************************************************/

/****************************************************************************
Desc:		Setup the pLNode array and set the count.
****************************************************************************/
RCODE F_ListMgr::Setup(
	F_ListNode *	pLNodes,
	FLMUINT			uiLNodeCnt)
{
	flmAssert( pLNodes && uiLNodeCnt );

	// Set the pLNodes array to null values.
	
	m_uiLNodeCnt = uiLNodeCnt;
	m_pLNodes = pLNodes;
	
	f_memset( pLNodes, 0, sizeof( F_ListNode) * uiLNodeCnt );
	return FERR_OK;
}

/****************************************************************************
Desc:		Insert an item into the first of a list.
****************************************************************************/
void F_ListMgr::InsertAtFirst(
	FLMUINT				uiList,					// List to insert this item.
	F_ListItem *		pNewFirstItem)			// New item to be inserted at the first.
{
	F_ListNode *		pLNode;
	
	// Check bounds with assert.  
	// This should be fine because uiList values are defined.
	
	flmAssert( uiList < m_uiLNodeCnt );
	
	pNewFirstItem->AddRef();

	pLNode = &m_pLNodes[ uiList ];
	
	if( pLNode->pNextItem == NULL)
	{
		// New last and first item.
		
		pLNode->pPrevItem = pNewFirstItem;
		pNewFirstItem->SetNextListItem( uiList, NULL);
	}
	else
	{
		// Add this new item to the first of the list. 
		pLNode->pNextItem->SetPrevListItem( uiList, pNewFirstItem );
		pNewFirstItem->SetNextListItem( uiList, pLNode->pNextItem);
	}

	pLNode->pNextItem = pNewFirstItem;
	pNewFirstItem->SetPrevListItem( uiList, NULL);
	pNewFirstItem->m_bInList = TRUE;

	// Increment the list's count element
	pLNode->uiListCount++;

	return;
}


/****************************************************************************
Desc:		Insert an item into the end of a list.
****************************************************************************/
void F_ListMgr::InsertAtEnd(
	FLMUINT				uiList,					// List to insert this item.
	F_ListItem *		pNewLastItem)			// New item to be inserted at the end.
{
	F_ListNode *		pLNode;
	
	// Check bounds with assert.  
	// This should be fine because uiList values are defined.
	
	flmAssert( uiList < m_uiLNodeCnt );
	
	pNewLastItem->AddRef();

	pLNode = &m_pLNodes[ uiList ];
	
	if( pLNode->pPrevItem == NULL)
	{
		// New last and first item.
		pLNode->pNextItem = pNewLastItem;
		pNewLastItem->SetPrevListItem( uiList, NULL);
	}
	else
	{
		// Add this new item to the end of the list. 
		pLNode->pPrevItem->SetNextListItem( uiList, pNewLastItem);
		pNewLastItem->SetPrevListItem( uiList, pLNode->pPrevItem);
	}

	pLNode->pPrevItem = pNewLastItem;
	pNewLastItem->SetNextListItem( uiList, NULL);
	pNewLastItem->m_bInList = TRUE;

	// Increment the list's count element
	pLNode->uiListCount++;

	return;
}

/****************************************************************************
Desc:		Obtain an item from the list.  Do not remove the item from the list.
****************************************************************************/
F_ListItem * F_ListMgr::GetItem(
	FLMUINT		uiList,					// List to get the item from.
	FLMUINT		nth)						// Which item to retrieve (0 == first)
{
	F_ListNode *		pLNode;
	F_ListItem *		pListItem;
	
	// Check bounds with assert.  
	
	flmAssert( uiList < m_uiLNodeCnt );

	pLNode = &m_pLNodes[ uiList ];
	
	pListItem = pLNode ? pLNode->pNextItem : NULL;
	
	while( nth-- )
	{
		pListItem = pListItem->GetNextListItem( uiList );
	}
	
	return pListItem;
}

/****************************************************************************
Desc: Remove the supplied ListItem object from the specified list.
****************************************************************************/
void F_ListMgr::RemoveItem(
	FLMUINT				uiList,
	F_ListItem *		pItem)
{
	F_ListNode *		pMgrLNode;				/* Manager's list node (head/tail pointers)*/
	F_ListItem *		pPrevItem;
	F_ListItem *		pNextItem;
	
	flmAssert( uiList < m_uiLNodeCnt);

	pMgrLNode = &m_pLNodes[ uiList];

	/* Get this item's Prev and Next items. */

	pPrevItem = pItem->GetPrevListItem( uiList);
	pNextItem = pItem->GetNextListItem( uiList);

	if( pPrevItem == NULL && pNextItem == NULL	
		&& pMgrLNode->pPrevItem != pItem
		&& pMgrLNode->pNextItem != pItem)
	{
		/* If the item is not within the list then skip to the end.
			Note: Need to also make sure this item is not the head or tail. */
		goto Exit;
	}

	/* Determine if this item is pointed to by the head or tail pointers
		that the list manager maintains. */

	if( pMgrLNode->pPrevItem == pItem)
	{
		pMgrLNode->pPrevItem = pItem->GetPrevListItem( uiList);
	}

	if( pMgrLNode->pNextItem == pItem)
	{
		pMgrLNode->pNextItem = pItem->GetNextListItem( uiList);
	}

	/* If there is a prev item - change it's next ptr to be items next ptr */
	if( pPrevItem != NULL)
	{
		pPrevItem->SetNextListItem( uiList, pItem->GetNextListItem( uiList));
	}

	/* If there is a next item - change it's prev ptr to be items prev ptr */
	if( pNextItem != NULL)
	{
		pNextItem->SetPrevListItem( uiList, pItem->GetPrevListItem( uiList));
	}

	/* Clear out this items prev and next links */
	pItem->SetPrevListItem( uiList, NULL);
	pItem->SetNextListItem( uiList, NULL);
	pItem->m_bInList = FALSE;

	/* This list no longer needs a reference to this object. */
	pItem->Release();

	/* Decrement this list's count element */
	pMgrLNode->uiListCount--;

Exit:
	return;
}

/****************************************************************************
Desc:		Unlink all items from a single specified list or all lists.
****************************************************************************/
RCODE F_ListMgr::ClearList(
	FLMUINT			uiList)						// List number or FLM_ALL_LISTS
{
	FLMUINT			uiListCnt;
	F_ListNode *	pLNode;

	// Check bounds with assert.  
	flmAssert( (FLM_ALL_LISTS == uiList) || (uiList < m_uiLNodeCnt));

	if( uiList == FLM_ALL_LISTS)
	{
		uiList = 0;
		uiListCnt = m_uiLNodeCnt;
		pLNode = m_pLNodes;
	}
	else
	{
		uiListCnt = 1;
		pLNode = &m_pLNodes[ uiList ];
	}
	
	for( ; uiListCnt--; pLNode++, uiList++)
	{
		F_ListItem *		pItem;
		F_ListItem *		pNextItem;
		
		// Go through the list Releasing every list item.
		
		for( pItem = pLNode->pNextItem; pItem; pItem = pNextItem)
		{
			pNextItem = pItem->GetNextListItem( uiList);
	
			(void) RemoveItem( uiList, pItem);
		}

		// At this point the ListCount should be at 0.
		flmAssert( pLNode->uiListCount == 0);

		// Clear the managers head and tail list pointers.
		pLNode->pNextItem = pLNode->pPrevItem = NULL;
	}
	return FERR_OK;
}

/****************************************************************************
Desc:		Obtain an the number of items within a list.
****************************************************************************/
FLMUINT F_ListMgr::GetCount(
	FLMUINT			uiList)					// List to get the item count from.
{
	FLMUINT			uiLNodeCnt;
	FLMUINT			uiCount = 0;
	F_ListNode *	pLNode;

	// Check bounds with assert.  
	flmAssert( (FLM_ALL_LISTS == uiList) || (uiList < m_uiLNodeCnt));

	if( uiList == FLM_ALL_LISTS)
	{
		uiLNodeCnt = m_uiLNodeCnt;
		pLNode = m_pLNodes;
	}
	else
	{
		uiLNodeCnt = 1;
		pLNode = &m_pLNodes[ uiList];
	}

	/* Calculate the count for the list[s] */

	for( ; uiLNodeCnt--; pLNode++)
	{
		uiCount += pLNode->uiListCount;
	}

	return uiCount;
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_ListItem::~F_ListItem()
{
#ifdef FLM_DEBUG
	FLMUINT			uiLoop;
	F_ListNode *	pTmpNd;

	flmAssert( m_bInList == FALSE);

	for( uiLoop = 0; uiLoop < m_uiLNodeCnt; uiLoop++)
	{
		pTmpNd = &m_pLNodes[ uiLoop];
		flmAssert( pTmpNd->pPrevItem == NULL && pTmpNd->pNextItem == NULL);
	}
#endif
}

/****************************************************************************
Desc:		Setup the pLNode array and set the count.
Visit:	We may want to add code in the future to check if Setup() has
			been previous called.  If so recode for uiLNodeCnt to be zero.
****************************************************************************/
RCODE F_ListItem::Setup(
	F_ListMgr *		pList,						// List manager to use
	F_ListNode *	pLNodes,						// Array of LNODEs to be used
	FLMUINT			uiLNodeCnt)					// Number of LNODEs supplied.
{
	flmAssert( pList != NULL);
	flmAssert( pLNodes != NULL);
	flmAssert( uiLNodeCnt != 0);

	m_pListMgr = pList;
	m_uiLNodeCnt = uiLNodeCnt;
	m_pLNodes = pLNodes;
	
	f_memset( pLNodes, 0, sizeof( F_ListNode) * uiLNodeCnt );
	return FERR_OK;
}

/****************************************************************************
Desc:		Remove this list item from all of the lists it is in.
****************************************************************************/
RCODE F_ListItem::RemoveFromList(	// Remove this list item from all lists.
	FLMUINT			uiList)				// Which list to remove item from
												// To remove item from all lists pass in
												// FLM_ALL_LISTS define.
{

	flmAssert( (uiList < m_uiLNodeCnt) || (uiList == FLM_ALL_LISTS));

	if( uiList == FLM_ALL_LISTS)
	{
		FLMUINT			uiListCnt = m_uiLNodeCnt;
		F_ListNode *	pLNode = m_pLNodes;

		uiList = 0;

		/* Remove this item from all lists. */

		for( ; uiListCnt--; uiList++, pLNode++)
		{
			m_pListMgr->RemoveItem( uiList, this);
		}
	}
	else
	{
		/* Remove item from a specific list. */
		m_pListMgr->RemoveItem( uiList, this);
	}
	
	return FERR_OK;
}

