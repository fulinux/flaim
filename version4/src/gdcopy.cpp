//-------------------------------------------------------------------------
// Desc:	Copy GEDCOM tree
// Tabs:	3
//
//		Copyright (c) 1990-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gdcopy.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/**************************************************************************
Name : GedNodeCopy
Area : GEDCOM
Desc : Allocates a new node, value, and tag and copies from the old node
		 its attached child(ren) and sibling(s).
Notes:	
****************************************************************************/
NODE * GedNodeCopy(
	POOL *		pPool,
	NODE *		node,
	NODE *		childList,
	NODE *		sibList)
{
	NODE *		newNd;
	FLMUINT		bias;
	FLMBYTE *		vp;
	RCODE			rc;
	HFDB			hDb;
	FLMUINT		uiContainer;
	FLMUINT		uiRecId;

	// If the node has source information, we need to copy it
	
	if( RC_OK( GedGetRecSource( node, &hDb, &uiContainer, &uiRecId)))
	{
		// The passed in node contains record source information,
		// so create a GEDCOM record source node
		
		if( RC_BAD( gedCreateSourceNode( pPool, GedTagNum( node), hDb,
													uiContainer, uiRecId, &newNd)))
		{
			return NULL;
		}
	}
	else
	{	
		// Create a normal (non-source) GEDCOM node
		
		if( (newNd = GedNodeMake( pPool, GedTagNum( node), &rc)) == NULL)
		{
			return( NULL);
		}
	}

	newNd->prior = NULL;
	newNd->next = childList;
	GedNodeLevelSet( newNd, 0);
	
	if( (vp = (FLMBYTE *)GedAllocSpace( pPool, newNd, GedValType( node),
		GedValLen( node), node->ui32EncId, GedEncLen( node))) != NULL)
	{
		f_memcpy( vp, GedValPtr( node), GedValLen( node));
		
		if (node->ui32EncFlags & FLD_HAVE_ENCRYPTED_DATA)
		{
			f_memcpy( GedEncPtr( newNd), GedEncPtr( node), GedEncLen( node));
		}
		
		newNd->ui32EncFlags = node->ui32EncFlags;
	}
	else
	{
		return( NULL);
	}

	if( childList)
	{
		childList->prior = newNd;
		for(										/* find end of sub-tree */
			bias = GedNodeLevel( childList) - 1
													/* 1st child level should be 1 */
		;	childList->next					/* continue to last in list */
		;	GedNodeLevelSub( childList, bias),	/* correct relative level */
			childList = childList->next	/* follow list */
		);
		GedNodeLevelSub( childList, bias);	/* correct last node in list */
		childList->next = sibList;
	}
	else
		childList = newNd;					/* no child(ren)--sib(s) link to newNd */

	if( sibList)								/* attach sibling(s) */
	{
		sibList->prior = childList;
		childList->next = sibList;
		for(										/* find end of sibList */
			bias = GedNodeLevel( sibList)			/* sib must be level 0 too */
		;	sibList->next
		;	GedNodeLevelSub( sibList, bias),		/* correct relative level */
			sibList = sibList->next
		);
		GedNodeLevelSub( sibList, bias);		/* correct last node in list */
	}

	return( newNd);
}

/*API~*********************************************************************
Desc:	Copies the entire contents of a tree.
****************************************************************************/
NODE * GedCopy(
	POOL *		pPool,
	FLMUINT		cnt,
	NODE *		tree)
{
	NODE *		oldNd;
	NODE *		newNd;
	NODE *		newRoot;
	FLMUINT		baseLevel;

	if( tree)
	{
		newRoot = newNd = GedNodeCopy( pPool, tree, NULL, NULL);
		if( newRoot)
		{
			for(
				baseLevel = GedNodeLevel( tree)
			;	(tree = tree->next) != NULL &&	/* follow linked list */
				(
					GedNodeLevel( tree) > baseLevel ||		/* process sub-tree */
					(
						GedNodeLevel( tree) == baseLevel &&	/* if sibling in forest AND */
						--cnt						/* count not expired, do next tree */
					)
				)
			;
			)
			{
				oldNd = newNd;					/* save for linking below */
				if( (newNd = GedNodeCopy( pPool, tree, NULL, NULL)) != NULL)
				{
					oldNd->next = newNd;		/* link up */
					newNd->prior = oldNd;
					GedNodeLevelSet( newNd, GedNodeLevel( tree) - baseLevel);
				}
				else
					return( NULL);
			}
		}
		return( newRoot);
	}

	return( NULL);
}

/*API~***********************************************************************
Name : GedClip
Area : GEDCOM/LINK
Desc : Unlinks a node or sub-tree from its parent and/or siblings.
Notes: Starting at the node specified by self, treeCnt sibling trees will
		 be unlinked from their parent node (if any), as well as from their
       previous and next sibling nodes (if any).  If the clipped siblings
       had a previous sibling and a next sibling, the previous sibling and
       the next sibling are reconnected as siblings.  If the clipped
       siblings had a parent, a next sibling, but no previous sibling, the
       next sibling be reconnected to the parent as the parent's first
       child.
*END************************************************************************/
NODE * 
		// A pointer to the input node/sub-tree (self) is returned.  This
      // allows GedClip to be used a a parameter to other functions which
      // require a NODE * parameter.
	GedClip(
 		FLMUINT	treeCnt,
			// [IN] Number of sibling trees to unlink.
		NODE *	self)
   		// [IN] Pointer to the node or sub-tree which is to be unlinked.
{
	NODE *	next;

	if( self)
	{
		FLMUINT	oldLevel = GedNodeLevel( self);

		GedNodeLevelSet( self, 0);			/* clipped tree now at level 0 */

		for(										/* skip to next sub-tree */
			next = self->next; 
			next &&								/* stop at end of sub-tree */
			(
				GedNodeLevel( next) > oldLevel ||	/* continue if child level */
				(
					GedNodeLevel( next) == oldLevel &&		/* if forest sibling, --treeCnt */
					--treeCnt					/* continue if treeCnt != 0 */
				)
			)
		;	GedNodeLevelSub( next, oldLevel),	// Adjust levels relative to new root
			next = next->next)
		{
			;
		}

		if( self->prior)
		{
			self->prior->next = next;		/* re-link the gap in old tree/forest */
		}

		if( next)
		{
			next->prior->next = NULL;		/* clipped tree's end must be made null*/
			next->prior = self->prior;		/* re-link the gap in old tree/forest */
		}
		self->prior = NULL;					/* clipped tree's head must now be root */
	}
	return( self);
}

