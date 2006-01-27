//-------------------------------------------------------------------------
// Desc:	Get/set binary data in fields in gedcom records.
// Tabs:	3
//
//		Copyright (c) 1992-1993,1995-1997,1999-2000,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gbinary.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/*API~***********************************************************************
Name : GedGetBINARY
Area : GEDCOM
Desc : Gets binary data from a GEDCOM TEXT node or a GEDCOM
		 BINARY node.  If the node is a TEXT node, it converts each two
	    hex digits to one byte of BINARY.
Notes: If the buffer pointer is NULL, the routine just determines how
		 much buffer space is needed to return the data in binary
		 format.
*END************************************************************************/
RCODE // SUCCESS - No problems in processing
		// FERR_CONV_DEST_OVERFLOW - number of bytes specified by *bufLenRV is
		// not sufficient
		// FERR_CONV_ILLEGAL - the GEDCOM node is not a BINARY or TEXT type node
		// FERR_CONV_BAD_DIGIT - only hexadecimal values in a TEXT type node
		// are allowed for conversion.
	GedGetBINARY(
		NODE *		node,
			// [IN] Pointer to a GEDCOM node containing binary or hexadecimal
			// text data.
		void *		buffer,
			// [OUT} Pointer to the output buffer that will contain the binary
			// data.
		FLMUINT  *	bufLenRV
			// [IN] Specifies the length of buffer.
			// [OUT] Returns the number of bytes used in buffer.
	)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	ptr;
	FLMUINT		valLength;
	FLMUINT		outputData;
	FLMUINT		nodeType;

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
	/* If the node is not a BINARY or a TEXT node, return an error. */

	nodeType = GedValType( node);
	if( (nodeType != FLM_TEXT_TYPE) && (nodeType != FLM_BINARY_TYPE))
	{
		rc = RC_SET( FERR_CONV_ILLEGAL);
		goto Exit;
	}

	ptr = (FLMBYTE *)GedValPtr( node);
	valLength = GedValLen( node);
	if( nodeType == FLM_TEXT_TYPE)
	{
		rc = GedTextToBin( ptr, valLength, (FLMBYTE *)buffer, bufLenRV);
		goto Exit;
	}

	/* At this point we know the node is a BINARY node */

	outputData = ((buffer != NULL) && (*bufLenRV));
	if( (outputData) && (valLength))
	{
		if( valLength > *bufLenRV)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
		f_memcpy( buffer, ptr, valLength);
	}
	*bufLenRV = valLength;

Exit:
	return( rc);
}

/*API~*********************************************************************
Desc:	Copies a binary string of bytes into a GEDCOM node.
*END************************************************************************/
RCODE GedPutBINARY(
	POOL * 			pPool,
	NODE * 			node,
	const void *	pvData,
	FLMUINT			uiDataLen,
	FLMUINT			uiEncId,
	FLMUINT			uiEncSize)
{
	RCODE			rc = FERR_OK;
	FLMBYTE *	outPtr;

	// Check for a null node being passed in

	if( !node)
	{
		rc = RC_SET( FERR_CONV_NULL_DEST);
		goto Exit;
	}

	// If data pointer is NULL or length is zero, call GedAllocSpace with a
	// length of zero to set the node length to zero and node type to
	// FLM_BINARY_TYPE.

	if( pvData == NULL || !uiDataLen)
	{
		(void)GedAllocSpace( pPool, node, FLM_BINARY_TYPE, 0, uiEncId, uiEncSize);
		goto Exit;
	}

	// Allocate space in the node for the binary data

	if( (outPtr = (FLMBYTE *)GedAllocSpace( pPool, node,
		FLM_BINARY_TYPE, uiDataLen, uiEncId, uiEncSize)) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Set the node type and copy the data into the node

	f_memcpy( outPtr, pvData, uiDataLen);

	if (node->ui32EncId)
	{
		node->ui32EncFlags = FLD_HAVE_DECRYPTED_DATA;
	}

Exit:

	return( rc);
}
