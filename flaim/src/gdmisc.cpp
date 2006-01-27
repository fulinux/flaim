//-------------------------------------------------------------------------
// Desc:	Miscellaneous GEDCOM routines.
// Tabs:	3
//
//		Copyright (c) 1992,1994-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gdmisc.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/* Offset from the end of the node to each record source element */
#define NODE_DRN_POS			0
#define NODE_CONTAINER_POS	(NODE_DRN_POS + sizeof( FLMUINT))
#define NODE_DB_POS			(NODE_CONTAINER_POS + sizeof( FLMUINT))

/*API~*********************************************************************
Name : GedNodeCreate
Area : GEDCOM
Desc : Allocates space for a new GEDCOM node.  Returns a pointer to the
		 node or NULL if an allocation error occurs.
Notes: 
****************************************************************************/
NODE * GedNodeCreate(
	POOL *		pPool,
	FLMUINT		tagNum,
	FLMUINT		id,
	RCODE *		rc)
{
	NODE *			nd;

	if( (nd = (NODE *)GedPoolAlloc( pPool,
							( sizeof(NODE) + (id ? sizeof(id) : 0)))) == NULL)
	{
		*rc = RC_SET( FERR_MEM);
	}
	else
	{
		f_memset( nd, '\0', sizeof( NODE));

		GedValTypeSet( nd, FLM_CONTEXT_TYPE);
		GedTagNumSet( nd, tagNum);

		if( id)
		{
			FLMBYTE *		ptr;
			GedValTypeSetFlag( nd, HAS_REC_ID);		/* Must set the ID before getting ptr */
			ptr = ((FLMBYTE *) nd) + sizeof(NODE);	/* If we call GedIdPtr */
			*((FLMUINT *)(ptr + NODE_DRN_POS)) = id;
		}
		*rc = FERR_OK;
	}

	return( nd);
}

/****************************************************************************
Desc:	This routine allocates space in a GEDCOM node for a value.  If the
		node already has the required space, nothing is done.  Otherwise,
		it calls the PoolAlloc routine to get the needed memory. This
		routine also sets the value length and type information.
Note: On FLM_TEXT_TYPE data type one extra byte will be allocated.
		This byte will be used for a NULL character.
NOTE: WARNING - If there is a length then the ptr value may be reused.
		This could cause problems with reusing GEDCOM memory and using
		the pool marker.
****************************************************************************/
void * GedAllocSpace(
	POOL *		pPool,
	NODE *		node,
	FLMUINT		valType,
	FLMUINT		size,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	FLMBYTE *	rPtr;							/* Return Pointer */
	FLMUINT		uiAllocSize = size;

	if( valType == FLM_TEXT_TYPE)
		uiAllocSize++;

	if( uiAllocSize <= sizeof( void *))
	{
		/* If the size is less than sizeof (void *), we use the space right */
		/* inside value pointer itself. */

		rPtr = (FLMBYTE *) &node->value;
	}

	/* BUG 10/1/96: Don't use wAllocSize here */

	else if( size <= GedValLen( node))
	{
		/* If there is already allocated space, just re-use it */

		rPtr = (FLMBYTE *)GedValPtr( node);
	}

	else
	{
		/* At this point, we know we have to allocate space elsewhere.
		** NOTE:  If we are unable to allocate the space required, DO
		** NOT modify the node -- return NULL immediately.
		*/

		if( (node->value = rPtr =
				(FLMBYTE *) GedPoolAlloc( pPool, uiAllocSize)) == NULL)
		{
			/* VISIT: 10/1/96: Comment above does not agree with this code. */

			node->ui32Length = 0;
			node->value = NULL;
			return( NULL);
		}
	}
	if( valType == FLM_TEXT_TYPE)
		rPtr[ size] = '\0';

	/* Now set the size and the data type */

	node->ui32Length = (FLMUINT32)size;
	GedSetType( node, valType);

	// If passed-in enc id is zero, use the node's enc id.

	if (!uiEncId)
	{
		flmAssert( !uiEncSize);
		if (size)
		{
			uiEncId = node->ui32EncId;
			uiEncSize = size + (16 - (size % 16));
		}
	}
	else
	{

		// We only should have an encryption ID if size is non-zero.
		// If size is non-zero, encryption size must also be non-zero.

		flmAssert( size);
		flmAssert( uiEncSize);
	}

	if (uiEncId)
	{
		if( uiEncSize > GedEncLen( node))
	  {
			if( (node->pucEncValue =
					(FLMBYTE *) GedPoolAlloc( pPool, uiEncSize)) == NULL)
			{
				node->ui32EncLength = 0;
				node->pucEncValue = NULL;
				return( NULL);
			}
		}
		node->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA | FLD_HAVE_ENCRYPTED_DATA;
		node->ui32EncId = (FLMUINT32)uiEncId;
		node->ui32EncLength = (FLMUINT32)uiEncSize;
	}


	return( rPtr);
}

/****************************************************************************
Desc:	return pointer to value.  The value may be store in the node if it's
		small enough in size to fit in the void * value pointer slot.
****************************************************************************/
void * GedValPtr(
	NODE *		nd)
{
	return(
		nd && nd->ui32Length
		?	GedValType( nd) == FLM_TEXT_TYPE
			?	nd->ui32Length < sizeof( void *)
				?	(void *) &nd->value
				:	(void *) nd->value
			:	nd->ui32Length > sizeof( void *)		/* non-text (no null terminator) */
				?	(void *) nd->value					/* value seperate from node */
				:	(void *) &nd->value					/* value in node's valuePtr space */
		:	(void *)NULL  									/* no node or value */
	);
}

/****************************************************************************
Desc:	return pointer to encryption value.
****************************************************************************/
void * GedEncPtr(
	NODE *		nd)
{
	return(
		nd && nd->ui32EncLength
			?	(void *)nd->pucEncValue
			:	(void *)NULL  							/* no node or encrypted value */
	);
}

/****************************************************************************
Desc:	Allows the user to set the ID (record number, or sub-record number)
		for the supplied field.
****************************************************************************/
RCODE GedPutRecId(
	POOL *		pPool,
	NODE **		ppNd,
	FLMUINT		uiId)
{
	NODE *		pNewNd;
	NODE *		pOldNd = *ppNd;
	FLMBYTE *	ptr;

	if( (pNewNd = (NODE *)GedPoolAlloc(pPool,
									sizeof( NODE) + sizeof( uiId))) == NULL)
	{
		*ppNd = NULL;
		return( RC_SET( FERR_MEM));
	}

	// Copy the contents of the existing node
	
	pNewNd->prior = pOldNd->prior;
	pNewNd->next = pOldNd->next;
	pNewNd->value = pOldNd->value;
	pNewNd->ui32Length = pOldNd->ui32Length;
	pNewNd->ui32EncId = pOldNd->ui32EncId;
	pNewNd->ui32EncLength = pOldNd->ui32EncLength;
	pNewNd->ui32EncFlags = pOldNd->ui32EncFlags;
	pNewNd->pucEncValue = pOldNd->pucEncValue;
	GedTagNumSet( pNewNd, GedTagNum( pOldNd));
	GedNodeLevelSet( pNewNd, GedNodeLevel( pOldNd));
	GedNodeTypeSet( pNewNd, (GedNodeType( pOldNd) | HAS_REC_ID));

	// Link in new node to parent and children/siblings
	
	if( pNewNd->prior)
	{
		pNewNd->prior->next = pNewNd;
	}
	
	if( pNewNd->next)
	{
		pNewNd->next->prior = pNewNd;
	}

	// Set the Ids value
	ptr = (FLMBYTE *)GedIdPtr( pNewNd );
	*((FLMUINT *)(ptr + NODE_DRN_POS)) = uiId;
	*ppNd = pNewNd;
	
	return( FERR_OK);
}

/****************************************************************************
Desc:	Will set the source information in a GEDCOM node.
****************************************************************************/
void gedSetRecSource(
	NODE *	pNode,      
	HFDB		hDb,
	FLMUINT	uiContainer,
	FLMUINT	uiDrn	)
{
	FLMBYTE *	pucPtr;

	pucPtr = ((FLMBYTE *) pNode) + sizeof( NODE);	/* Set pucPtr to end of node */
	if( uiDrn)
	{
		GedValTypeSetFlag( pNode, HAS_REC_ID);
		*((FLMUINT *)(pucPtr + NODE_DRN_POS)) = uiDrn;
	}
	if( uiContainer)
	{
		GedValTypeSetFlag( pNode, HAS_REC_SOURCE);
		*((FLMUINT *)(pucPtr + NODE_CONTAINER_POS)) = uiContainer;
	}
	if( hDb)
	{
		GedValTypeSetFlag( pNode, HAS_REC_SOURCE);
		*((HFDB *)(pucPtr + NODE_DB_POS)) = hDb;
	}
}

/****************************************************************************
Desc:	Will create a GEDCOM node that contains a FLAIM Database's HFDB,
		store number, container number, and record id (DRN).
****************************************************************************/
RCODE gedCreateSourceNode(
	POOL *		pPool,			/* Users allocation pool */
	FLMUINT  	uiFieldNum,		/* Tag Number */
	HFDB 			hDb,				/* FLAIM Database that record came from */
	FLMUINT		uiContainer,  	/* Container record came from */
	FLMUINT		uiRecId,			/* Record id (DRN) */
	NODE **		ppNode)			/* [out] newly created gedcom source node */
{
	NODE *		nd;
	RCODE			rc = FERR_OK;

	*ppNode = nd = (NODE *)GedPoolCalloc( pPool,
										( sizeof( NODE)
										+ sizeof( FLMUINT)	/* Record Id (DRN) */
										+ sizeof( FLMUINT)	/* Container Number */
										+ sizeof( HFDB))); 	/* Database handle */
	if( nd != NULL)
	{
		GedValTypeSet( nd, FLM_CONTEXT_TYPE);
		GedTagNumSet( nd, uiFieldNum);
		gedSetRecSource( nd, hDb, uiContainer, uiRecId);
	}
	else
	{
		rc = RC_SET( FERR_MEM);
	}

	return( rc);
}

/****************************************************************************
Desc:		Returns the FLAIM database source that this GEDCOM record is from.
Remarks:	The root node of each GEDCOM record returned from FLAIM will contain
			information about where the record came from. This information is
			known as its source information and includes:
				Memory Handle to FLAIM Database (HFDB)
				Store Number
				Container Number
				Record Id (DRN)

Note: 	Some GEDCOM Nodes may only contain the Record Id, in those cases
			calls to GedGetRecSource will return a NULL for the HFDB and 0's for
			the Store and Container number (meaning the record has not been
			assigned to a FLAIM database yet).
****************************************************************************/
RCODE GedGetRecSource(
	NODE *		pNode,		/* GEDCOM node to return FLAIM database source
										information. */
	HFDB *		phDb,			/* [out] FLAIM Database that this record came from.
										NOTE: This value is only valid while the
										database is actually open, and has no persistent
										capabilities. */
	FLMUINT *	puiContainer,/* [out] Database Container that this record is from.*/
	FLMUINT *	puiRecId)	/* [out] Database Record Id (DRN) that has been
										assigned to this record. This is unique only
										within a database's container. */
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	ptr =	((FLMBYTE *) pNode) + sizeof( NODE);	/* Set ptr to end of node */

	if( GedNodeType( pNode) & HAS_REC_SOURCE)
	{
		if( phDb)
		{
			*phDb = *((HFDB *)(ptr + NODE_DB_POS));
		}

		if( puiContainer)
		{
			*puiContainer = *((FLMUINT *)(ptr + NODE_CONTAINER_POS));
		}

		if( puiRecId)
		{
			*puiRecId = *((FLMUINT *)( ptr + NODE_DRN_POS));
		}
	}
	else if( GedNodeType( pNode) & HAS_REC_ID)
	{
		if( phDb)
		{
			*phDb = NULL;
		}

		if( puiContainer)
		{
			*puiContainer = 0;
		}

		if( puiRecId)
		{
			*puiRecId = *((FLMUINT *)( ptr + NODE_DRN_POS));
		}
	}
	else
	{	/* The record contains no record source, because the user may ignore
			the return code lets make sure everything is set to null/0. */

		if( phDb)
		{
			*phDb = NULL;
		}

		if( puiContainer)
		{
			*puiContainer = 0;
		}

		if( puiRecId)
		{
			*puiRecId = 0;
		}

		rc = RC_SET( FERR_NOT_FOUND);
		goto Exit;
	}

Exit:
	return( rc);
}


/*API~*********************************************************************
Desc:	Places the suppolied DRN into a context type node.
*END************************************************************************/
RCODE GedPutRecPtr(
	POOL *		pPool,
	NODE *		nd,
	FLMUINT 		drn,
	FLMUINT		uiEncId,
	FLMUINT		uiEncSize)
{
	void *		ptr;
	RCODE			rc = FERR_OK;

	/* Check for a null node being passed in */

	if( nd == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	if( (ptr = GedAllocSpace( pPool, nd, FLM_CONTEXT_TYPE,
							  sizeof(FLMUINT32), uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	UD2FBA( (FLMUINT32) drn, ptr);

	if (nd->ui32EncId)
	{
		nd->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:
	return( rc);
}

/*API~*********************************************************************
Name : GedGetRecPtr
Area : GEDCOM
Desc : Obtain the DRN (database record number) from a GEDCOM context type
		 node.  No conversion from other data types will be performed.
*END************************************************************************/
RCODE  // FERR_CONV_ILLEGAL - the input node (nd) is not a context type.
		// FERR_CONV_NULL_SRC - The input node (nd) is NULL.
	GedGetRecPtr(
		NODE *		nd,
			// [IN] Input GEDCOM node.
		FLMUINT *	drnRV
			// [OUT] Returns the DRN value on SUCCESS.
	)
{
	RCODE		rc = FERR_OK;

	*drnRV = (FLMUINT) 0xFFFFFFFF;							/* value for "no value" */

	if( nd == NULL)												/* Make sure we have a valid node */
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	if (nd->ui32EncId)
	{
		if (!(nd->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	if( GedValType(nd) != FLM_CONTEXT_TYPE)
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);	 	/* DIN's doesn't convert */
		goto Exit;
	}

	if( GedValLen( nd) == sizeof( FLMUINT32))
	{
		*drnRV = (FLMUINT)(FB2UD((FLMBYTE *) GedValPtr( nd)));
	}

Exit:
	return( rc);
}


/*API~*********************************************************************
Name : GedWalk
Area : GEDCOM
Desc : Traverses a tree or forest of GEDCOM tress.  For each GEDCOM node 
		 that is visited, a user specified callback function is called.

		The passed-in function needs to accept the following parameters:
			FLMUINT	level;		* current relative level of this node *
			NODE *	node;			* pointer to current node *
			void *	arg;			* user's passed-thru parameter *
		This passed-in function is repeatedly called until its return code is
		not SUCCESS, the tree/forest is completely processed, or the count
		expires.

Notes:This function can be extremely useful in many contexts. 									 
*END************************************************************************/
RCODE	GedWalk(
	FLMUINT		treeCnt,
		// [IN] treeCnt is the number of sibling trees to process.  
		// Pass in GED_TREE to walk through the input node and all
		// of its children.  Pass in GED_FOREST to walk through the input
		// node and all of its siblings as well as all children.  A number
		// may also be specified to limit the number of trees that are
		// walked through.  GED_TREE has a value of one.
	NODE * 		node,
		// [IN] Pointer to the first node of a GEDCOM tree or forest.
	GEDWALK_FUNC_p	func,
		// [IN] User specified callback function called on every GEDCOM
		// node that is visited.
	void *		arg)
		// [IN] Argument used as an argument for the callback function func().
{
	RCODE	rc;

	if( node)										/* non-null tree */
	{
		FLMUINT	baseLevel = GedNodeLevel( node);/* save to know when sub-tree's done */
		do
		{
			rc =										/* save return code for test & exit */
				(*func)(								/* passed-in function pointer */
					(GedNodeLevel( node) - baseLevel),/* node's relative level number */
					node,	arg);	
		}while(
			RC_OK( rc) &&							/* stop if( *func) != SUCCESS */
			(node = node->next) != NULL &&	/* stop if end of tree/forest */
			(
				GedNodeLevel( node) > baseLevel ||		/* continue while in sub-tree */
				(
					GedNodeLevel( node) == baseLevel &&/* if sibling, decrement treeCnt */
					--treeCnt						/* continue if treeCnt != 0 */
				)
			)											/* else, stop if no sibling trees */
		);
	}
	else
		rc = FERR_OK;								/* null tree always SUCCESS */

	return( rc);
}
