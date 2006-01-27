//-------------------------------------------------------------------------
// Desc:	GEDCOM tree traversal routines.
// Tabs:	3
//
//		Copyright (c) 1990-1993,1996-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gdtrvrs1.cpp 12308 2006-01-19 15:08:11 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~*********************************************************************
Name : GedSibNext
Area : GEDCOM
Desc : Returns a pointer to the sibling node that is after the input node.
		 NULL is returned if there is not a next sibling.
Notes: 
*END************************************************************************/
NODE *  
		// Pointer to the next sibling from node or NULL if the next
		// sibling does not exist.
	GedSibNext(
		NODE *		node)
			// [IN] Pointer to a GEDCOM node.
{
	FLMUINT	lev;

	if( node)
	{
		lev = GedNodeLevel( node);
		while(															/*skip children*/
			((node = node->next) != NULL) &&
			(GedNodeLevel( node) > lev)
		);
	}

	return(
		(node && (GedNodeLevel( node) == lev))
		?	node
		:	NULL);
}

/*API~*********************************************************************
Name : GedSibNext
Area : GEDCOM
Desc : Returns a pointer to the parent of the input node.
		 NULL is returned if there is not a parent.
Notes: 
*END************************************************************************/
NODE *  
		// Returns a pointer to the parent of the input node or NULL.
	GedParent(
		NODE *		node)
			// [IN] Pointer to a GEDCOM node.
{
	if( node)
	{
		FLMUINT	lev = GedNodeLevel( node);
		while(															/*skip nephews & siblings*/
			((node = node->prior) != NULL) &&
			(GedNodeLevel( node) >= lev)
		);
	}
	return( node);
}

/*API~*********************************************************************
Name : GedChild
Area : GEDCOM
Desc : Returns a pointer to the child of the input node.
		 NULL is returned if there is not a child.
Notes: 
*END************************************************************************/
NODE *  
		// Returns a pointer to the child of the input node or NULL.
	GedChild(
		NODE *		node
			// [IN] POinter to a GEDCOM node.
	)
{
	return(
		node &&
		node->next &&
		(GedNodeLevel( node->next) > GedNodeLevel( node))
		?	node->next
		:	NULL
	);
}

/*API~*********************************************************************
Name : GedNodeCreate
Area : GEDCOM
Desc : Returns a pointer to the previous sibling of the input node.
Notes: 
*END************************************************************************/
NODE *  
		// Returns a pointer to the previous sibling or NULL.
	GedSibPrev(
		NODE *		node)
			// [IN] Pointer to a GEDCOM node.
{
	FLMUINT	lev;

	if( node)
	{
		lev = GedNodeLevel( node);
		while(										/* skip nephews */
			((node = node->prior) != NULL) &&
			(GedNodeLevel( node) > lev)
		);
	}
	return(
		(node && (GedNodeLevel( node) == lev))
		?	node
		:	NULL
	);
}



