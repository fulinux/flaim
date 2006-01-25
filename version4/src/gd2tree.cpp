//-------------------------------------------------------------------------
// Desc:	Import text GEDCOM buffer
// Tabs:	3
//
//		Copyright (c) 1990-1992,1994-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: gd2tree.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

extern FLMBYTE arr[];

#define f_isdigit(c) \
	((c) < 60 ?  (( ((FLMBYTE)(arr[(c) >> 3])) << ((c) & 0x07)) & 0x80) : 0)

FSTATIC RCODE tagValLenType(
	POOL * 				pPool,
	GED_STREAM_p		x,
	NODE **				node,
	F_NameTable *		pNameTable);

/*************************************************************************
Desc:	This function parses and builds one complete GEDCOM tree from a GEDCOM
		character buffer or file.  The beginning level number is used only to
		delimit the tree (possiblly in forest); the generated tree's level
		always start at zero.
Note:	The syntax for each GEDCOM "line" in the sub-tree is:

		[ {[#<comment>] <cr>} |
		 [
		  <level> [@<xref-id>@] <tag> [{@<xref-ptr>@} | {["] <value> ["]}] <cr>
			[ [{#<comment>} | {+ ["] <value> ["]}] <cr>]...
		 ]
		]...

		Where:

		<comment>	any character string
		  <level>	0 <= <level> < 32 (decimal) (delimited by whitespace/<cr>)
			 <tag>	any string of alphanumeric characters (and/or '_') and must
						not start with digit (delimited by whitespace/<cr>).
			  <cr>	carriage return (and/or linefeed)
				  #	1st non-whitespace character of line showing that a comment
						follows
				  +	1st non-whitespace character of line showing value continues
						from prior line.  Can not continue if xref used instead of
						value
				  "	quote mark may optionally preceed/follow string--leading or
						trailing spaces are included (significant) only if bounded
						by a leading/trailing quote mark
		  <value>	any character string.  data type is automatically converted
						to data dictionary type at time of storage
  <White space>	White space (tabs and/or spaces) is ignored except in
						strings.

***************************************************************************/
RCODE GedToTree(
	POOL *			pPool,
	F_FileHdl *		pFileHdl,
	char **			pBuf,
	FLMUINT			uiBufSize,
	NODE **			root,
	F_NameTable *	pNameTable)
{
	RCODE				rc = FERR_OK;
	GED_STREAM		x;
	FLMUINT      	level;
	FLMUINT			levelBase = 0;
	FLMUINT			levelPrior = 0;
	FLMBYTE			nextChar;
	NODE *			nd;
	NODE *			ndPrior = NULL;
	FLMUINT			startPos;

	x.pFileHdl = pFileHdl;
	x.pThis =
	x.pBuf = *pBuf;
	x.uiBufSize = uiBufSize;
	
	if( pFileHdl)
	{
		// Find 1st starting file position
		
		if( RC_OK( pFileHdl->Seek( 0L, F_IO_SEEK_CUR, &x.uiFilePos)))
		{
			x.pLast = x.pBuf;							/* set parms for forced 1st read */
			gedReadChar( &x, x.uiFilePos);					/* insure buffer has data in it */
		}
		else
			return( RC_SET( FERR_FILE_ER));
	}
	else
	{
		x.errorIO = 0;
		x.uiFilePos = 0;									/* uiFilePos is really just bufferPos */
		x.pLast = x.pBuf + (uiBufSize - 1);
		x.thisC = f_toascii( *x.pBuf);
	}

	for(;;)
	{
		gedSkipBlankLines( &x);
		startPos = x.uiFilePos;						/* save position of first of line */

		if( f_isdigit( x.thisC))					/* level number */
		{
			/* process level number */

			level = 0;
			do												/* compute level number */
			{
				level = x.thisC - ASCII_ZERO + (level * 10);
				nextChar = (FLMBYTE)(gedNextChar( &x));/* put next character in variable */
			} while( f_isdigit( nextChar));		/* to avoid problems with macro expansion */

			if( ! f_iswhitespace( x.thisC))
			{
				rc = RC_SET( FERR_BAD_FIELD_LEVEL);	/* not terminated properly */
				break;
			}

			if( level > GED_MAXLVLNUM)
			{
				rc = RC_SET( FERR_GED_MAXLVLNUM);
				break;
			}

			if( ndPrior)								/* not 1st time through */
			{
				if( levelBase >= level)				/* end sub-tree; more trees follow */
					goto successful;					/* done */
				else if( (levelPrior < level) &&	((levelPrior + 1) != level))
				{											/* bad level jump (too big: > 1) */
					rc = RC_SET( FERR_GED_SKIP_LEVEL);
					break;
				}
			}
			else
				levelBase = level;
			levelPrior = level;

			/* process node & value */

			rc = tagValLenType( pPool, &x, &nd, pNameTable);

			/* link node into tree */

			if( RC_OK( rc))
			{
				if( ndPrior)
					ndPrior->next = nd;				/* link prior to this */
				else
					*root = nd;
				nd->prior = ndPrior;					/* link this to prior */
				GedNodeLevelSet( nd, level - levelBase);
															/* make node's level relative to base */
				ndPrior = nd;
				continue;
			}
		}
		else if( x.thisC == '\0' || x.thisC == ASCII_CTRLZ)	/* end sub-tree; none follow */
		{
			if( x.errorIO)
				rc = RC_SET( FERR_FILE_ER);
			else if( ndPrior)
			{
successful:
				ndPrior->next = NULL;				/* finish last node's link */
				if( pFileHdl == NULL)
					*pBuf = x.pThis + (FLMINT32)(startPos - x.uiFilePos);
				x.uiFilePos = startPos;				/* this line part of next tree */
				rc = FERR_OK;
			}
			else
				rc = RC_SET( FERR_END);		/* only BLANKLINES in input line */
		}
		else
		{
			rc = RC_SET( FERR_BAD_FIELD_LEVEL);	/* missing level number */
		}
		break;											/* exit "for" loop */
	}

	if( RC_BAD( rc))
	{
		*root = NULL;
		if( pFileHdl == NULL)
			*pBuf = x.pThis;
	}
	if( pFileHdl)
	{
		/* this token had syntax error */
		pFileHdl->Seek( x.uiFilePos, F_IO_SEEK_SET, &x.uiFilePos);
	}
	return( rc);
}

/***************************************************************************
Desc:	Parse the tag, value, and length from a GEDCOM buffer, create a
		node, and populate it with these values.  Continuation lines and
		embedded comments are also handled.
***************************************************************************/
FSTATIC RCODE tagValLenType(
	POOL * 			pPool,
	GED_STREAM_p	x,
	NODE **			newNode,
	F_NameTable *  pNameTable)
{
	FLMUINT		startPos;
	RCODE			rc = FERR_OK;
	NODE *		nd;
	FLMUINT		drn = 0;
	FLMUINT		uiTagNum;
	char			tagBuf[ GED_MAXTAGLEN + 1];

	gedSkipWhiteSpaces( x);

	/* process optional xref-id */

	startPos = x->uiFilePos;
	if( x->thisC == ASCII_AT)						/* at-sign sequence begins */
	{
		int badDRN;
		for( badDRN = 0, gedNextChar( x); x->thisC != ASCII_AT; gedNextChar( x))
		{
			FLMUINT	priorDrn = drn;

			if( ! badDRN)
			{
				if( f_isdigit( x->thisC))
				{
					drn = (drn * 10) + x->thisC - ASCII_ZERO;
					badDRN = priorDrn != (drn / 10);
				}
				else
					badDRN = 1;
			}
		}
		if( badDRN)
			drn = 0;

		gedNextChar( x);
		if( f_iswhitespace( x->thisC))
			gedSkipWhiteSpaces( x);
		else
		{
			rc = RC_SET( FERR_GED_BAD_RECID);
			goto Exit;
		}
	}

	/* Determine the Tag Number and Build the NODE */

	startPos = x->uiFilePos;

	if( !gedCopyTag( x, tagBuf))
	{
		return( RC_SET( FERR_INVALID_TAG));
	}

	if( !pNameTable->getFromTagTypeAndName( NULL, tagBuf, 
		FLM_FIELD_TAG, &uiTagNum))
	{
		/* See if tag is the reserved tag with the number following */

		if( tagBuf[0] == f_toascii( 'T') &&
			 tagBuf[1] == f_toascii( 'A') &&
			 tagBuf[2] == f_toascii( 'G') &&
			 tagBuf[3] == f_toascii( '_'))
		{
			uiTagNum = f_atoi( &tagBuf[ 4]);
		}
		else
		{
			return( RC_SET( FERR_NOT_FOUND));
		}
	}

	if( (*newNode = nd = GedNodeCreate( pPool, uiTagNum, drn, &rc)) == NULL)
	{
		goto Exit;
	}

	gedSkipWhiteSpaces( x);

	/* alternate xref_ptr used instead of "value" */

	startPos = x->uiFilePos;
	if( x->thisC == ASCII_AT)
	{
		for( drn = 0; gedNextChar( x) != ASCII_AT;)
		{
			FLMUINT	priorDrn = drn;
			if( f_isdigit( x->thisC))
			{
				drn = (drn * 10) + x->thisC - ASCII_ZERO;
				if( priorDrn == (drn / 10))
					continue;					/* no overflow yet */
			}
			rc = RC_SET( FERR_GED_BAD_VALUE);	/* catch overflow & non-numeric */
			goto Exit;
		}
		gedNextChar( x);
		GedPutRecPtr( pPool, nd, drn);
		if( gedCopyValue( x, NULL))		/* test 2nd value & skip whitespace */
		{
			rc = RC_SET( FERR_GED_BAD_VALUE);	/* can't have both @xref@ and value */
			goto Exit;
		}
	}
	else
	{
		FLMINT	valLength;
		FLMUINT	tempPos = x->uiFilePos;

		if( (valLength = gedCopyValue( x, NULL)) > 0)
		{
			char * 	vp = (char *)GedAllocSpace( pPool, nd,
													FLM_TEXT_TYPE, valLength);
				
			if( vp)
			{
				gedReadChar( x, tempPos);
				gedCopyValue( x, vp);
			}
			else
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}
		}
	}

	startPos = x->uiFilePos;					/* successful: point to next token */

Exit:
	gedReadChar( x, startPos);					/* position to start of token */
	return( rc);
}

