//-------------------------------------------------------------------------
// Desc:	Set/get native strings into/from GEDCOM nodes.
// Tabs:	3
//
//		Copyright (c) 1992-1993,1995-1997,1999-2000,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gnative.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~*********************************************************************
Desc:	Copies and formats a native 8-bit null terminated string into a 
		GEDCOM node.  The code page
		It converts the string into an internal TEXT string in the GEDCOM node.
		ALL single byte, fixed length, and variable length function codes 
		are preserved as long as GedGetWP60 is called.  Other GedGet*() 
		calls will lose all formatting information.
Note: The code page parameter is currently not supported.  All character
		values of 0x80 and higher will be stored as non-character values.
		These values will be preserved if GedGetNATIVE is called, but will
		be dropped if any other GEDCOM get routine is called.  In addition,
		all character values under 0x20 are preserved as non-character values.
*END************************************************************************/
RCODE GedPutNATIVE(
	POOL *	 		pPool,
	NODE * 			node,
	const char *	nativeString,
	FLMUINT			uiEncId,
	FLMUINT			uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMUINT		allocLength;
	FLMBYTE *	outPtr;

	// Check for a null node being passed in

	if( node == NULL)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	/* If the string is NULL or empty, call GedAllocSpace with a length */
	/* of zero to set the node length to zero and node type to FLM_TEXT_TYPE. */

	if( (!nativeString) || (!(*nativeString)))
	{
		(void)GedAllocSpace( pPool, node, FLM_TEXT_TYPE, 0, uiEncId, uiEncSize);
		goto Exit;
	}

	// Determine the size of the buffer needed to store the string 
	
	if( RC_BAD( rc = FlmNative2Storage( nativeString, &allocLength, NULL)))
	{
		goto Exit;
	}

	if( (outPtr = (FLMBYTE *)GedAllocSpace( pPool, node,
														 FLM_TEXT_TYPE, allocLength,
													 	 uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Convert the string 

	if( RC_BAD( rc = FlmNative2Storage( nativeString, &allocLength, outPtr)))
	{
		goto Exit;
	}

	// Encrypted fields - only have decrypetd data at this point.
	
	if (node->ui32EncId)
	{
		node->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return( rc);
}

/*API~*********************************************************************
Desc : Places 8-bit text from a GEDCOM text type node into an output buffer.
		 The current text representation supports WordPerfect 6.x character
		 values, WordPerfect formatting codes, UNICODE character values 
		 and 8-bit character values that range from 0x80 to 0xFF.
*END************************************************************************/
RCODE GedGetNATIVE(
	NODE *			node,
	char *			pszBuffer,
	FLMUINT *		bufLenRV)
{
	RCODE				rc = FERR_OK;
	FLMBYTE *		ptr;
	FLMUINT			valLength;
	FLMUINT			nodeType;

	if( !node)
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
	
	// If the node is not a TEXT or a NUMBER node, return an error for now

	nodeType = (FLMBYTE)GedValType( node);
	if( (nodeType == FLM_BINARY_TYPE) || (nodeType == FLM_CONTEXT_TYPE))
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	ptr = (FLMBYTE *)GedValPtr( node);
	valLength = GedValLen( node);

	rc = FlmStorage2Native( nodeType, valLength, (const FLMBYTE *)ptr, bufLenRV, pszBuffer);

Exit:

	return( rc);
}
