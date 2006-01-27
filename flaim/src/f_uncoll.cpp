//-------------------------------------------------------------------------
// Desc:	Uncollation routines for converting from collated string to WP string.
// Tabs:	3
//
//		Copyright (c) 1992-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: f_uncoll.cpp 12245 2006-01-19 14:29:51 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/**-----------------------------------------
***		External tables
***		Could be far in another data segment
***----------------------------------------*/
/* From COLLATE1.C */

extern FLMUINT16		colToWPChr[];		/* Converts collated value to WP character */
extern FLMBYTE			ml1_COLtoD[];		/* Diacritic conversions */

extern FLMUINT16		HebArabColToWPChr[];
extern FLMUINT16		ArabSubColToWPChr[];

/**-----------------------------------------
***		Local Static Routine Prototypes
***----------------------------------------*/

/**----------------------------------------------------------------
***  The version using the table uses 34 ticks and 29 bytes ONLY 
***  because the turbo optimizer uses register variables better.  
***  The other version below uses 39 ticks and 33 bytes.
***  Macro not moved to other calls in f_tocoll or kycompnd because
***  these are rarely called areas right now.
***---------------------------------------------------------------*/


FSTATIC FLMUINT FWWSGetColStr(	/* Returns byte length of word string*/
	FLMBYTE *	fColStr,
	FLMUINT * 	fcStrLenRV,
	FLMBYTE *	wordStr,
	FLMUINT		fWPLang,
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbFirstSubstring);

FSTATIC FLMUINT FWWSCmbSubColBuf(
	FLMBYTE *	wordStr,
	FLMUINT *	wdStrLenRV,
	FLMBYTE *	subColBuf,
	FLMBOOL		hebrewArabicFlag	);

FSTATIC FLMUINT FWWSToMixed(
	FLMBYTE *	wordStr,
	FLMUINT		wdStrLen,
	FLMBYTE *	lowUpBitStr,
	FLMUINT		fWPLang	);

/**************************************************************************
Desc:	Get the Flaim collating string and convert back to a text string
Ret: 	Length of new wpStr
Notes:	Allocates the area for the word string buffer if will be over 256.
***************************************************************************/
FLMUINT FColStrToText(
	FLMBYTE *	fColStr,					/* Points to the Flaim collated string */
	FLMUINT *	fcStrLenRV,				/* Length of the Flaim collated string */
	FLMBYTE *	textStr,					/* Output string to build - TEXT string */
	FLMUINT	   fWPLang,					/* FLAIM WP language number */
	FLMBYTE *	postBuf,					/* Lower/upper POST buffer or NULL */
	FLMUINT *	postBytesRV,			/* Return next position to use in postBuf */
	FLMBOOL *	pbDataTruncated,		/* Sets to TRUE if data had been truncated */
	FLMBOOL *	pbFirstSubstring)		/* Sets to TRUE if first substring */
{
#define LOCAL_CHARS		150
	FLMBYTE		wordStr[ LOCAL_CHARS * 2 + LOCAL_CHARS / 5 ];	// Sample + 20%
	FLMBYTE *  	wsPtr = NULL;
	FLMBYTE *	wsAllocatedWsPtr = NULL;
	FLMUINT		wsLen;
	FLMUINT		textLen;
	FLMBYTE *	textPtr;

	if(  *fcStrLenRV > LOCAL_CHARS )			/* If won't fit allocate 1280 */
	{
		if( RC_BAD( f_alloc( MAX_KEY_SIZ * 2, &wsPtr)))
		{
			return( 0 );
		}
		wsAllocatedWsPtr = wsPtr;
	}
	else
		wsPtr = wordStr;
	
 	if( (fWPLang >= FIRST_DBCS_LANG) &&
 		 (fWPLang <= LAST_DBCS_LANG))
 	{
		wsLen = AsiaConvertColStr( fColStr, fcStrLenRV, wsPtr, 
									pbDataTruncated, pbFirstSubstring );
		if(  postBuf )
		{
			FLMUINT postBytes = *postBytesRV + 2;	/* Skip past marker */

			/* may change wsLen */
			postBytes += AsiaParseCase( wsPtr, &wsLen, &postBuf[ postBytes]);
			*postBytesRV = postBytes;
		}

	}
	else
	{
		wsLen = FWWSGetColStr( fColStr, fcStrLenRV, wsPtr, fWPLang, 
										pbDataTruncated, pbFirstSubstring );

		/* If a post buffer is sent - turn unflagged chars to lower case */
		if(  postBuf )
		{
			FLMUINT postBytes = *postBytesRV;
			/* Check if mixed case chars follow and always increment postBytes */
			if(  postBuf[ postBytes++ ] == (COLL_MARKER | SC_MIXED))
			{
				postBytes += FWWSToMixed( wsPtr, wsLen,
													&postBuf[ postBytes ], fWPLang);
			}
			*postBytesRV = postBytes;
		}
	}	
	/**-------------------------------------------
	***  Copy word string to TEXT string area
	***------------------------------------------*/

	wsLen >>= 1;									/* Convert # of bytes to # of words */
	textPtr = textStr;

	while( wsLen--)
	{
		register FLMBYTE	ch, cSet;

		/* Put the character in a local variable for speed */
		ch   = *wsPtr++;
		cSet = *wsPtr++;

		if( (!cSet) && (ch <= 127))
		{

			/**-----------------------------------------------------------
			***  Character set zero only needs one byte if the character
			***  is <= 127.  Otherwise, it is handled like all other
			***  extended characters below.
			***----------------------------------------------------------*/

			*textPtr++ = ch;
		}
		/**-----------------------------------------------------
		***  If the character set is > 63 it takes three bytes 
		***  to store, otherwise only two bytes are needed.
		***----------------------------------------------------*/
		else if( cSet < 63)
		{
			*textPtr++ = (FLMBYTE)(CHAR_SET_CODE | cSet);
			*textPtr++ = ch;
		}
		else if( cSet == 0xFF && ch == 0xFF)
		{
			*textPtr++ = UNICODE_CODE;
			*textPtr++ = *(wsPtr+1);		/* Character set */
			*textPtr++ = *wsPtr;				/* Character */
			wsPtr += 2;
			wsLen--;								/* Skip past 4 bytes for UNICODE */
		}
		else
		{
			*textPtr++ = EXT_CHAR_CODE;
			*textPtr++ = cSet;
			*textPtr++ = ch;
		}
	}

	textLen = (textPtr - textStr);		/* Compute total length */
	
	if( wsAllocatedWsPtr != NULL)
		f_free( &wsAllocatedWsPtr);

	return( textLen);
}

/*****************************************************************************
Desc:		Get the Flaim collating string and convert back to a WP word string
Ret:		Length of new WP word string
*****************************************************************************/
FSTATIC FLMUINT FWWSGetColStr(
	FLMBYTE *	fColStr,			  		/* Points to the Flaim collated string */
	FLMUINT *	fcStrLenRV,		  		/* Length of the Flaim collated string */
	FLMBYTE *	wordStr,			  		/* Output string to build - WP word string */
	FLMUINT		fWPLang,			  		/* FLAIM WP language number */
	FLMBOOL *	pbDataTruncated,		/* Set to TRUE if truncated */
	FLMBOOL *	pbFirstSubstring)		/* Sets to TRUE if first substring */
{
	FLMBYTE *	wsPtr  = wordStr;		/* Points to the word string data area */
	FLMUINT		length = *fcStrLenRV;/* May optimize as a register */
	FLMUINT		pos = 0;					/* Position in fColStr[] */
	FLMUINT		bitPos;					/* Computed bit position */
	FLMUINT		colChar;					/* Not portable if a FLMBYTE value */
	FLMUINT		wdStrLen;
	FLMBOOL		hebrewArabicFlag = 0;/* Set if hebrew/arabic language */

	/**
	***  WARNING:
	***  The code is duplicated for performance reasons.
	***  The US code below is much more optimized so
	***  any changes must be done twice.
	**/

	
	if( fWPLang != US_LANG)			/* Code for NON-US languages */
	{
		if( (fWPLang == AR_LANG ) ||		/* Arabic */
			 (fWPLang == FA_LANG ) ||		/* Farsi - persian */
			 (fWPLang == HE_LANG ) ||		/* Hebrew */
			 (fWPLang == UR_LANG ))			/* Urdu */
			hebrewArabicFlag++;				/* Add sindhi, pashto, kurdish, malay*/

		// MVSVISIT: will not work correctly on IBM390 - need to change toolkit tables.
		while( length && (fColStr[pos] > MAX_COL_OPCODE))
		{
			length--;
			colChar = (FLMUINT) fColStr[ pos++ ];
			switch( colChar)
			{
			case COLS9+4:							/* ch in spanish */
			case COLS9+11:							/* ch in czech */
				/* Put the WP char in the word string */
				UW2FBA( (FLMUINT16) 'C', wsPtr );
				wsPtr += 2;
				colChar = (FLMUINT) 'H';
				pos++;							/* move past second duplicate char */
				break;

			case COLS9+17:						/* ll in spanish */
				/* Put the WP char in the word string */
				UW2FBA( (FLMUINT16)'L', wsPtr );	
				wsPtr += 2;
				colChar = (FLMUINT)'L';
				pos++;							/* move past duplicate character */
				break;
			
			case COLS0:				/* Non collating character or OEM character */
				/* Actual character is in sub-collation area*/
				colChar = (FLMUINT) 0xFFFF;
				break;

			default:
				/* Watch out COLS10h has () around it for subtraction */
				if( hebrewArabicFlag && (colChar >= COLS10h))
				{
					colChar = (colChar < COLS10a)		/* Hebrew only? */
					 		? (FLMUINT) (0x900 + (colChar - (COLS10h)))	/* Hebrew */
					 		: (FLMUINT) (HebArabColToWPChr[ colChar - (COLS10a)]);	/* Arabic */
				} 
				else
				{
					colChar = (FLMUINT) colToWPChr[ colChar - COLLS ];
				}
				break;
			}
			UW2FBA( (FLMUINT16) colChar, wsPtr );		/* Put the WP char in the word string*/
			wsPtr += 2;
		}	/* end while */
	}	/* end if */
	else											/* US Sorting - optimized */
	{
		while( length && (fColStr[pos] > MAX_COL_OPCODE))
		{
			length--;
			/* Move in the WP value given uppercase collated value */
			colChar = (FLMUINT) fColStr[ pos++ ];

			if( colChar == COLS0)
			{
				colChar = (FLMUINT) 0xFFFF;
			}
			else
			{
				colChar = (FLMUINT) colToWPChr[ colChar - COLLS ];
			}
		UW2FBA( (FLMUINT16) colChar, wsPtr );			/* Put the WP char in the word string */
		wsPtr += 2;
		}
	}
	/* NULL Terminate the string */
	UW2FBA( (FLMUINT16)0, wsPtr);
	wdStrLen = pos + pos;					/* Multiply fcStrLen by 2 */

	/**--------------------------------------------------------------------
	***  Parse through the sub-collation and case information.
	***  Watch out for COMP POST indexes - don't have case info following.
	***  Here are values for some of the codes:
	***   [ 0x01] - end for fields - case info follows - for COMP POST ixs
	***   [ 0x02] - compound marker
	***   [ 0x03] - not really used at this time
	***   [ 0x04] - case information is all uppercase (IS,DK,GR)
	***   [ 0x05] - case bits follow
	***   [ 0x06] - case information is all uppercase
	***   [ 0x07] - beginning of sub-collation information
	***	[ 0x08] - first substring field that is made
	***	[ 0x09] - truncation marker for text and binary
	***
	***  Below are some cases to consider...
	***
	*** [ COLLATION][ 0x07 sub-collation][ 0x05 case info][ 0x02]
	*** [ COLLATION][ 0x07 sub-collation][ 0x05 case info]
	*** [ COLLATION][ 0x07 sub-collation][ 0x02]
	*** [ COLLATION][ 0x07 sub-collation][ 0x01]
	*** [ COLLATION][ 0x05 case info][ 0x02]
	*** [ COLLATION][ 0x05 case info] 
	*** [ COLLATION][ 0x02]
	*** [ COLLATION][ 0x01]
	***
	***  In the future still want[ 0x06] to be compressed out for uppercase
	***  only indexes.
	***-------------------------------------------------------------------*/

	// Check first substring before truncated
	if( length && fColStr[pos] == COLL_FIRST_SUBSTRING)
	{
		if( pbFirstSubstring)
			*pbFirstSubstring = TRUE;		// Don't need to initialize to FALSE.
		length--;
		pos++;
	}
	if( length && fColStr[pos] == COLL_TRUNCATED)
	{
		if( pbDataTruncated)
			*pbDataTruncated = TRUE;		// Don't need to initialize to FALSE.
		length--;
		pos++;
	}
	/**------------------------------
	***  Does sub-collation follow?
	***-----------------------------*/
	
	/* Still more to process - first work on the sub-collation (diacritics) */
	/* Hebrew/Arabic may have empty collation area */
	if( length && (fColStr[pos] == (COLL_MARKER | SC_SUB_COL)))
	{
		FLMUINT tempLen;
		/* Do another pass on the word string adding the diacritics */
		bitPos = FWWSCmbSubColBuf( wordStr, &wdStrLen, 
											&fColStr[++pos],
											hebrewArabicFlag );

		/* Move pos to next byte value */
		tempLen = BYTES_IN_BITS( bitPos );
		pos += tempLen;
		length -= tempLen + 1;				/* The 1 includes the 0x07 byte */
	}

	/**-------------------------------
	***  Does the case info follow?
	***------------------------------*/
	
	if( length && (fColStr[pos] > COMPOUND_MARKER))
	{
		/**----------------------------------------------------
		***  Take care of the lower and upper case conversion 
		***  If mixed case then convert using case bits
		***---------------------------------------------------*/

		if( fColStr[pos++] & SC_MIXED)		/* Increment pos here! */
		{
			/* Don't pre-increment pos on line below! */
			pos += FWWSToMixed( wordStr, wdStrLen, &fColStr[pos], fWPLang );
		}
		/* else 0x04 or 0x06 - all characters already in uppercase */

	}
	*fcStrLenRV = pos;          	/* pos should be on the 0x01 or 0x02 flag */
	return( wdStrLen);				/* Return the length of the word string */
}

/**************************************************************************
Desc: 	Combine the diacritic 5 bit values to an existing word string
Todo:		May want to check fwpCh6Cmbcar() for CY return value
***************************************************************************/
FSTATIC FLMUINT  FWWSCmbSubColBuf(
	FLMBYTE *	wordStr,						/* Existing word string to modify */
	FLMUINT *	wdStrLenRV,					/* Wordstring length in bytes */
	FLMBYTE *	subColBuf,					/* Diacritic values in 5 bit sets */
	FLMBOOL		hebrewArabicFlag)			/* Set if language is Hebrew or Arabic */
{
	FLMUINT 		subColBitPos = 0;
	FLMUINT 		numWords = *wdStrLenRV >> 1;
	FLMUINT16 	diac;
	FLMUINT16 	wpchar;
	FLMUINT		temp;

	/* For each word in the word string ... */
	while( numWords--)
	{
		/* label used for hebrew/arabic - additional subcollation can follow */
		/* This macro DOESN'T increment bitPos */
		if( TEST1BIT( subColBuf, subColBitPos))
		{
			/**--------------------------------------------
			*** If "11110" - unmappable unicode char - 0xFFFF is before it
			*** If "1110" then INDEX extended char is inserted
			*** If "110" then extended char follows that replaces collation
			*** If "10"  then take next 5 bits which
			*** contain the diacritic subcollation value.
			***-------------------------------------------*/
after_last_character:	
			subColBitPos++;						/* Eat the first 1 bit */
			if(  ! TEST1BIT( subColBuf, subColBitPos))
			{
				subColBitPos++;					/* Eat the 0 bit */
				diac = (FLMUINT16)(GETnBITS( 5, subColBuf, subColBitPos));
				subColBitPos += 5;

				if( (wpchar = FB2UW( wordStr )) < 0x100)	/* If not extended base..*/
				{

					/* Convert to WP diacritic and combine characters */
					fwpCh6Cmbcar( &wpchar, wpchar, (FLMUINT16) ml1_COLtoD[diac] );
					/* Even if cmbcar fails, wpchar is still set to a valid value */
					UW2FBA( wpchar, wordStr);
				}
				else if( (wpchar & 0xFF00) == 0x0D00)	/* arabic? */
				{
					wpchar = ArabSubColToWPChr[ diac ];
					UW2FBA( wpchar, wordStr);				
				}
				/* else diacritic is extra info */
				/* cmbcar should not handle extended chars for this design */	
			}
			else		/* "110"  or "1110" or "11110" */
			{
				subColBitPos++;					/* Eat the 2nd '1' bit */
				if( TEST1BIT( subColBuf, subColBitPos))	/* Test the 3rd bit */
				{
					/* 1110 - shift wpchars down 1 word and insert value below */
					subColBitPos++;					/* Eat the 3rd '1' bit */
					*wdStrLenRV += 2;					/* Return 2 more bytes */
					
					if( TEST1BIT( subColBuf, subColBitPos ))	/* Test 4th bit */
					{
						/* Unconvertable UNICODE character */
						/* The format will be 4 bytes, 0xFF, 0xFF, 2 byte Unicode */
						
						shiftN( wordStr, numWords + numWords + 4, 2 );
						subColBitPos++;				/* Eat the 4th '1' bit */
						wordStr += 2;					/* Skip the 0xFFFF for now */
					}
					else
					{
						/* Move down 2 byte NULL and rest of the 2 byte characters */
						/* The extended character does not have a 0xFF col value */

						shiftN( wordStr, numWords + numWords + 2, 2 );
						numWords++;						/* Increment because inserted */
						/* fall through reading the actual charater value */
					}	
				}
				subColBitPos++;						/* Skip past the zero bit */
				subColBitPos = (subColBitPos + 7) & (~7);	/*roundup to next byte*/
				temp = BYTES_IN_BITS( subColBitPos );		/* compute position */
				wordStr[1] = subColBuf[ temp ];				/* Character set */
				wordStr[0] = subColBuf[ temp + 1 ];			/* Character */

				subColBitPos += 16;
			}
		}
		else
			subColBitPos++;

		wordStr += 2;						/* Next WP character */
	}
	if( hebrewArabicFlag )
	{
		if( TEST1BIT( subColBuf, subColBitPos))	
		{
			/**--------------------------------------------------
			***  Hebrew/Arabic can have trailing accents that
			***  don't have a matching collation value.
			***  Keep looping in this case.
			***  Note that subColBitPos isn't incremented above.
			***-------------------------------------------------*/
			numWords = 0;						/* set so won't loop forever! */
			goto after_last_character;		/* process trailing bit */
		}
		subColBitPos++;						/* Eat the last '0' bit */
	}
	return( subColBitPos);
}

/**************************************************************************
Desc: 	Convert the word string to lower case chars given low/upp bit string
Out:	 	WP characters have modified to their original case
Ret:		Number of bytes used in the lower/upper buffer
Notes:	Only WP to lower case conversion is done here for each bit NOT set.
***************************************************************************/
FSTATIC FLMUINT  FWWSToMixed(
	FLMBYTE *	wordStr,			  	/* Existing word string to modify */
	FLMUINT		wdStrLen,		  	/* Length of the wordstring in bytes */
	FLMBYTE *	lowUpBitStr,	  	/* Lower/upper case bit string */
	FLMUINT		fWPLang)			  	/*Visit: Scott */
{
	FLMUINT		numWords;
	FLMUINT		tempWord;
	FLMBYTE		tempByte = 0;
	FLMBYTE		maskByte;
	FLMBYTE		xorByte;						/* Used to reverse GR, bits */
	
	xorByte = (fWPLang == US_LANG )		/* Do most common compare first */
						? (FLMBYTE)0
						: (fWPLang == GR_LANG)	/* Greek has uppercase first */
							? (FLMBYTE)0xFF
							: (FLMBYTE)0 ;

	/* For each word in the word string ... */
	for(  numWords = wdStrLen >> 1,		/* Total number of words in word string */
				maskByte = 0;								/* Force first time to get a byte */
				
				numWords--;									/* Test */

				wordStr += 2,								/* Next WP character - word */
				maskByte >>= 1 )						/* Next bit to mask and check */
	{
		if( maskByte == 0)							/* Time to get another byte */
		{
			tempByte = xorByte ^ *lowUpBitStr++;
			maskByte = 0x80;
		}

		if( ( tempByte & maskByte) == 0)	/* If lowercase conver - else is upper*/
		{
			/* Convert to lower case - COLL -> WP is already in upper case */
			tempWord = (FLMUINT) FB2UW( wordStr );
			if( (tempWord >= ASCII_UPPER_A) && (tempWord <= ASCII_UPPER_Z))		/*  yes */
				tempWord |= 0x20;
			else
			{
				FLMBYTE charVal = (FLMBYTE)(tempWord & 0xFF);
				FLMBYTE charSet = (FLMBYTE) (tempWord >> 8);
				
				/* check if charact within region of character set */
				if (	(( charSet == CHSMUL1) &&		/* Multinational 1 */
					    ((charVal >= 26) && (charVal <= 241)))
					 ||(( charSet == CHSGREK) &&		/* Greek */
						 ( charVal <= 69))
					 ||(( charSet == CHSCYR) &&		/* Cyrillic */
						 ( charVal <= 199))
					)
				{
					tempWord |= 0x01;		/* Set - don't increment */
				}
			}
			UW2FBA( (FLMUINT16) tempWord, wordStr );
		}
	}

	numWords = wdStrLen >> 1;
	return( BYTES_IN_BITS( numWords ));
}
