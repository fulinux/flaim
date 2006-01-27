//-------------------------------------------------------------------------
// Desc:	Find a node in a GEDCOM tree structure.
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
// $Id: gdfind.cpp 12307 2006-01-19 15:06:34 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~*********************************************************************
Name : GedFind
Area : GEDCOM
Desc : Gets the nth occurance of the node with a matching tag number.
		 Continues, if necessary, thru a limited number of sibling sub-trees.
		 If "treeCnt" is GED_FOREST(0), it continues thru virtually all sub-trees.
		 matches occur without regard to path (always assumes leading wild card)
Note:	
*END************************************************************************/
NODE * 
		// Returns a pointer to the nth occurance of the node with a matching 
		// tag or NULL if unsuccesful. If a value less than 0 is
		// given for nth, the first value will be returned. If a
		// value greater than the number of occurances is given,
		// NULL will be returned.
	GedFind(
		FLMUINT		treeCnt,
			// [IN] Maximum number of sibling trees to search.
		NODE *		nd,
			// [IN] GEDCOM tree or forest to search for matching tnum.
		FLMUINT		tnum,
			// [IN] Specific tag number to find.
		FLMINT 		nth  
			// [IN] The occurance of the tag number to find.
	)
{
	if( nd)
  	{
		FLMUINT	strtLvl = GedNodeLevel( nd);	/* Starting level */
		do
		{
			if( (tnum == GedTagNum( nd)) && (--nth < 1))
				return( nd);

		} while(
				(nd = nd->next) != NULL &&
				(
					GedNodeLevel( nd) > strtLvl ||
					(--treeCnt && GedNodeLevel( nd) == strtLvl)
		      )
		);
	}
	return( NULL);
}


/*API~*********************************************************************
Name : GedNodeCopy
Area : GEDCOM
Desc : Gets the "nth" occurance of the node with a matching path of tag numbers.
		 This path may be found in the first tree only, the entire forest or
		 within the first "treeCnt" of trees.
Notes: This routine does not support any wildcard tags.
VISIT: This code has some bugs, one of which is that the
		 path array may be accessed past the null terminator.
*END************************************************************************/
NODE *  
		//	Returns the nth occurance of the node at the end of the
		//path or NULL if unsuccesful. If a value less than 0 is
		//given for nth, the first value will be returned. If a
		//value greater than the number of occurances is given,
		//NULL will be returned.
	GedPathFind(
		FLMUINT		treeCnt,
			// [IN] Maximum number of sibling tress to search.
		NODE *		nd,
			// [IN] The input GEDCOM tree to search for matching path.
		FLMUINT *	puiPathArray,
			// [IN] A null terminated array of field numbers which make
			// up a path.
		FLMINT		nth
  			// [IN] The occurance of the matching path to return.  This
			// value is usualy one.
	)
{
	NODE *		node = nd;
	NODE *		savenode;
	FLMUINT *	path;

	if( nd && puiPathArray)
	{
		FLMUINT	uiLevel = GedNodeLevel( nd);
		for(;;)
		{
			path = puiPathArray + (GedNodeLevel( node) - uiLevel);
			savenode = node;
			if( *path == GedTagNum( node))				/* matching piece of path */
			{
				if( *(path + 1) == 0 && (--nth < 1))
					return( node);								/* complete match found */
				if( (node = GedChild( node)) != NULL)
					continue;									/* go down level for rest of path */
				node = savenode;
			}

			do
			{
				node = node->next;
			}
			while( node != NULL &&
				GedNodeLevel( node) > GedNodeLevel( savenode));			/* find next sibling/uncle/end */

			if(
				! node ||										/* end of tree */
				GedNodeLevel( node) < uiLevel ||						/* end of forest */
				(
					GedNodeLevel( node) == uiLevel &&
					!(--treeCnt)								/* end of partial forest limit */
				)
			)
				break;
		}
	}
	return( NULL);
}

