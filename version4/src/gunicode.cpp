//-------------------------------------------------------------------------
// Desc:	Get/Set Unicode into GEDCOM node.
// Tabs:	3
//
//		Copyright (c) 1999-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gunicode.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~*********************************************************************
Desc:	Copies and formats a Unicode string into a GEDCOM node.  The Unicode
		string must be in little endian format.  The Unicode string is 
		converted into an internal TEXT string in the GEDCOM node.  
		Unicode values that are not represented as WordPerfect 6.x characters
		are preserved as non-WP characters and will be droped if any other
		GEDCOM get routine is called other than GedGetUNICODE.
*END************************************************************************/
RCODE GedPutUNICODE(
	POOL *	 				pPool,
	NODE * 					node,
	const FLMUNICODE *	puzString,
	FLMUINT					uiEncId,
	FLMUINT					uiEncSize)
{
	FLMUINT		allocLength = 0;
	FLMBYTE *	outPtr;
	RCODE			rc = FERR_OK;

	/* Check for a null node being passed in */

	if( node == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	/*
	**  If the string is NULL or empty, call GedAllocSpace with a length
	**  of zero to set the node length to zero and node type to FLM_TEXT_TYPE.
	*/

	if( (puzString == NULL) || (*puzString == 0))
	{
		GedAllocSpace( pPool, node, FLM_TEXT_TYPE, 0, uiEncId, uiEncSize);
		return( FERR_OK);
	}

	/*  Two passes are needed on the data.
	**  The first pass is to determine the storage length
	**  The second pass is to store the string into FLAIMs internal text format
	*/

	allocLength = FlmGetUnicodeStorageLength( puzString);

	if( (outPtr = (FLMBYTE *)GedAllocSpace( pPool, node,
														 FLM_TEXT_TYPE, allocLength,
														 uiEncId, uiEncSize)) == NULL)
	{
		return( RC_SET( FERR_MEM));
	}

	if (RC_BAD( rc = FlmUnicode2Storage( puzString, &allocLength, outPtr)))
	{
		goto Exit;
	}

	if (node->ui32EncId)
	{
		node->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:
	return( rc);
}


/*API~*********************************************************************
Name : GedGetUNICODE
Area : GEDCOM
Desc : Get Unicode data from a GEDCOM text type node.  Also supports
		 conversions from number, date, time and time stamp types.
		 Most WordPerfect extended characters are converted to Unicode
		 character values.  All Unicode character values are preserved if
		 GedPutUNICODE() is used.  The Unicode representation is ALWAYS in
		 little endian byte order.  This may change if FLAIM moves to JAVA.
Notes:
*END************************************************************************/
RCODE // SUCCESS
		// FERR_CONV_ILLEGAL - the input node is not a text, number, date,
		// time or time stampe type.
		// FERR_CONV_NULL_SRC - the input node (nd) is NULL.
		// FERR_CONV_DEST_OVERFLOW - the number of bytes
	GedGetUNICODE(
		NODE *			node,
			// [IN] GEDCOM node that contains data.
		FLMUNICODE *	uniBuf,
			// [IN/OUT] Unicode buffer to hold the data. If uniBuf is NULL then
			// only a count return in bufLenRV is return that is the number
			// of bytes needed to contain the data.  Two bytes must be added
			// to this value to account for the two bytes of null termination.
		FLMUINT *		bufLenRV
			// [IN] Specified the number of bytes available in buffer including
			// the terminating two null bytes.
			// [OUT] Returns the number of Unicode bytes that are needed to
			// represent the data.  The two null termination bytes are not
			// included in this value.
	)
{
	FLMUINT		nodeType;
	RCODE			rc = FERR_OK;

	/* Check for a null node */

	if( node == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_SRC);
		goto Exit;
	}

	if (node->ui32EncId)
	{
		if (!(node->ui32EncFlags & FLD_HAVE_DECRYPTED_DATA))
		{
			rc = RC_SET( FERR_FLD_NOT_DECRYPTED);
			goto Exit;
		}
	}

	/* If the node is not a TEXT or a NUMBER node, return an error for now. */

	nodeType = GedValType( node);

	if( (nodeType == FLM_BINARY_TYPE) || (nodeType == FLM_CONTEXT_TYPE))
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	rc = FlmStorage2Unicode( nodeType, GedValLen( node), (const FLMBYTE *)GedValPtr( node),
			bufLenRV, uniBuf);

Exit:
	return( rc);
}

