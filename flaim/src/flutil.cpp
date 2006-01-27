//-------------------------------------------------------------------------
// Desc:	Various utility functions
// Tabs:	3
//
//		Copyright (c) 1991-1993,1996-2000,2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flutil.cpp 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define HANDLE_NEGATIVE \
	if( value < 0) \
	{	*ptr++ = '-'; \
		absValue = (FLMUINT)(-(value)); \
	} \
	else absValue = (FLMUINT)value;

#define HANDLE_DNEGATIVE \
	if( value < 0) \
	{	*ptr++ = '-'; \
		absValue = (FLMUINT)(-(value)); \
	} \
	else absValue = (FLMUINT)value;

#define	PUSH_DIGITS( v) \
	{register FLMUINT reg = v; \
		do{ *sp++ = (FLMBYTE)((reg % 10) + '0'); \
		} while( reg /= 10); \
	}
	
#define	POP_DIGITS \
	while( stack < sp--)\
		*ptr++ = *sp; \
	*ptr = '\0';

#if defined( FLM_NLM) && defined ( __MWERKS__)
	void abort(void);
#endif

/****************************************************************************
Desc:  	Unsigned word to NATIVE value - null terminate the native string
****************************************************************************/
char * f_uwtoa(
	FLMUINT16	value,
	char *		ptr)
{
	char		  	stack[ 10];
	char *		sp = stack;

	PUSH_DIGITS( value);
	POP_DIGITS;

	return( ptr);
}

/****************************************************************************
Desc:  	Native to UDWORD value.  Supports 0x<HEX> codes. Non digits NOT ALLOWED
			NO LEADING SPACES ALLOWED ! ! !  No checks for overflow over 4 bytes!
****************************************************************************/
FLMUINT f_atoud(
	const char *  	pszBuf)
{
	FLMUINT			uiValue;
	FLMBOOL			bAllowHex = FALSE;

	if( *pszBuf == NATIVE_ZERO && 
		(*(pszBuf + 1) == NATIVE_LOWER_X || *(pszBuf + 1) == NATIVE_UPPER_X))
	{
		pszBuf += 2;
		bAllowHex = TRUE;
	}

	uiValue = 0;
	while( *pszBuf)
	{
		if( *pszBuf >= '0' && *pszBuf <= '9')
		{
			if( !bAllowHex)
			{
				uiValue *= 10;
			}
			else
			{
				uiValue <<= 4;
			}

			uiValue += (FLMUINT)(*pszBuf - '0');
		}
		else if( bAllowHex)
		{
			if( *pszBuf >= 'A' && *pszBuf <= 'F')
			{
				uiValue <<= 4;
				uiValue += (FLMUINT)(*pszBuf - 'A') + 10;
			}
			else if( *pszBuf >= 'a' && *pszBuf <= 'f')
			{
				uiValue <<= 4;
				uiValue += (FLMUINT)(*pszBuf - 'a') + 10;
			}
			else
			{
				break;
			}
		}
		else
		{
			break;
		}
		pszBuf++;
	}

	return( uiValue);
}

/****************************************************************************
Desc:  	Unsigned double (4 byte) number to native value & null terminate
****************************************************************************/
char * f_udtoa(
	FLMUINT		value,
	char *		ptr)
{
	char	 		stack[ 10];
	char *		sp = stack;

	PUSH_DIGITS( value);
	POP_DIGITS;

	return( ptr);
}

/****************************************************************************
Desc:  	Word to native value - null terminate the native string
****************************************************************************/
char * f_wtoa(
	FLMINT16		value,
	char *		ptr)
{
	char		  	stack[ 10];
	char *		sp = stack;
	FLMUINT		absValue;

	HANDLE_NEGATIVE;
	PUSH_DIGITS( absValue);
	POP_DIGITS;

	return( ptr);
}

/****************************************************************************
Desc:  	Double (4 byte) number to native value - null terminate the string
****************************************************************************/
char * f_dtoa(
	FLMINT		value,
	char *		ptr)
{
	char		  	stack[ 10];
	char *		sp = stack;
	FLMUINT		absValue;

	HANDLE_DNEGATIVE;
	PUSH_DIGITS( absValue);
	POP_DIGITS;

	return( ptr);
}

/****************************************************************************
Desc:  	Ascii to integer
****************************************************************************/
FLMINT f_atoi(
	const char *	ptr)
{
	return( f_atod( ptr));
}

/****************************************************************************
Desc:  	native to long
****************************************************************************/
FLMINT f_atol(
	const char *	ptr)
{
	return( f_atod( ptr));
}

/****************************************************************************
Desc:		Native to DWORD value.  Supports 0x<HEX> codes. Non digits NOT ALLOWED
			NO LEADING SPACES ALLOWED ! ! !  No checks for overflow over 4 bytes!
****************************************************************************/
FLMINT f_atod(
	const char *		pszBuf)
{
	FLMINT				iValue;
	FLMBOOL				bNeg = FALSE;

	if( *pszBuf == '-')
	{
		bNeg = TRUE;
		pszBuf++;
	}
	else if( *pszBuf == '+')
	{
		pszBuf++;
	}

	iValue = (FLMINT)f_atoud( pszBuf);
	return( bNeg ? -iValue : iValue);
}

/****************************************************************************
Desc:	Utility function to return the maximum size of a hex number
		represented as a string.
****************************************************************************/
FINLINE FLMUINT maxHexSize(
	FLMUINT			uiSizeOfPtr)
{
	return uiSizeOfPtr * 2;
}

/****************************************************************************
Desc:	Utility function to return the maximum size of a decimal number
		represented as a string.
****************************************************************************/
FINLINE FLMUINT maxDecimalSize(
	FLMUINT		uiSizeOfPtr)
{
	switch (uiSizeOfPtr)
	{
		case 4:
			return 10;
		case 8:
			return 20;
		default:
			flmAssert( 0);
			return 0;
	}
}

/****************************************************************************
Desc: Compares two Unicode strings
****************************************************************************/
FLMINT f_unicmp(
	const FLMUNICODE *	puzStr1,
	const FLMUNICODE *	puzStr2)
{
	while( *puzStr1 == *puzStr2 && *puzStr1)
	{ 
		puzStr1++;
		puzStr2++;
	}

	return( (FLMINT)*puzStr1 - (FLMINT)*puzStr2);
}

/****************************************************************************
Desc: Returns the length of a unicode string
****************************************************************************/
FLMUINT f_unilen(
	const FLMUNICODE *	puzStr)
{
	FLMUINT		uiLen = 0;

	if( !puzStr)
	{
		goto Exit;
	}

	while( *puzStr)
	{
		puzStr++;
		uiLen++;
	}

Exit:

	return( uiLen);
}

/****************************************************************************
Desc: Finds a substring
****************************************************************************/
FLMUNICODE * f_uniindex(
	const FLMUNICODE *	puzStr,
	const FLMUNICODE *	puzSearch)
{
	const FLMUNICODE *	puzSearchPos;
	const FLMUNICODE *	puzStrPos;
	FLMUNICODE				uFirstChar;	

	if( !puzStr || !puzSearch || (uFirstChar = *puzSearch) == 0)
	{
		return( NULL);
	}

	do
	{
		while( *puzStr)
		{
			if( *puzStr == uFirstChar)
			{
				break;
			}
			puzStr++;
		}

		if( *puzStr == 0)
		{
			return( NULL);
		}

		if( *(puzSearch + 1) == 0)
		{
			return( (FLMUNICODE *)puzStr);
		}

		for( puzSearchPos = puzSearch + 1, puzStrPos = puzStr + 1; 
			  *puzSearchPos == *puzStrPos; puzSearchPos++, puzStrPos++)
		{
			if( puzSearchPos[ 1] == 0)
			{
				return( (FLMUNICODE *)puzStr);
			}
		}

		puzStr++;
	} while( *puzStr);

	return( NULL);
}

/****************************************************************************
Desc: The equivalent of strncmp for unicode strings.
****************************************************************************/
FLMINT f_unincmp(
	const FLMUNICODE *	puzStr1,
	const FLMUNICODE *	puzStr2,
	FLMUINT					uiLen)
{
	if( !uiLen)
	{
		return( 0);
	}

	while( *puzStr1 == *puzStr2 && *puzStr1 && --uiLen)
	{
		puzStr1++;
		puzStr2++;
	}

	return( *puzStr1 - *puzStr2);
}

/****************************************************************************
Desc: Compares two strings, one Unicode and one native
****************************************************************************/
FLMINT f_uninativecmp(
	const FLMUNICODE *	puzStr1,
	const char *			pszStr2)
{
	while( *puzStr1 == ((FLMUNICODE)f_toascii( *pszStr2)) && *puzStr1)
	{
		puzStr1++;
		pszStr2++;
	}

	return( (FLMINT)*puzStr1 - (FLMINT)*pszStr2);
}

/****************************************************************************
Desc: Compares two strings, one Unicode and one native
****************************************************************************/
FLMINT f_uninativencmp(
	const FLMUNICODE *	puzStr1,
	const char *			pszStr2,
	FLMUINT					uiCount)
{
	if( !uiCount)
	{
		return( 0);
	}

	while( uiCount && 
		*puzStr1 == ((FLMUNICODE)f_toascii( *pszStr2)) && *puzStr1)
	{
		puzStr1++;
		pszStr2++;
		uiCount--;
	}

	return( uiCount ? ((FLMINT)*puzStr1 - (FLMINT)*pszStr2) : 0);
}

/****************************************************************************
Desc: Copies a unicode string
****************************************************************************/
FLMUNICODE * f_unicpy(
	FLMUNICODE *			puzDestStr,
	const FLMUNICODE *	puzSrcStr)
{
	const FLMUNICODE *	puzSrc = puzSrcStr;
	FLMUNICODE *			puzDest = puzDestStr;

	while( *puzSrc)
	{
		*puzDest++ = *puzSrc++;
	}

	*puzDest = 0;
	return( puzDestStr);
}

/****************************************************************************
Desc: Copies a native string into a Unicode buffer
****************************************************************************/
void f_nativetounistrcpy(
	FLMUNICODE *		puzDest,
	const char *		pszSrc)
{
	while( *pszSrc)
	{
		*puzDest++ = f_toascii( *pszSrc++);
	}
	*puzDest = 0;
}

/****************************************************************************
Desc: Abort function.  Required only when compiling with Codewarrior.  For
		some reason we don't have an implementation of this in any of the libraries
		we are linking against.
****************************************************************************/
#if defined( FLM_NLM) && defined( __MWERKS__)
void abort(void)
{
	flmAssert(0);
}
#endif

/***************************************************************************
Desc:		Sort an array of items
****************************************************************************/
void f_qsort(
	void *					pvBuffer,
	FLMUINT					uiLowerBounds,
	FLMUINT					uiUpperBounds,
	F_SORT_COMPARE_FUNC	fnCompare,
	F_SORT_SWAP_FUNC		fnSwap)
{
	FLMUINT			uiLBPos;
	FLMUINT			uiUBPos;
	FLMUINT			uiMIDPos;
	FLMUINT			uiCurrentPos;
	FLMUINT			uiLeftItems;
	FLMUINT			uiRightItems;
	FLMINT			iCompare;

Iterate_Larger_Half:

	uiUBPos = uiUpperBounds;
	uiLBPos = uiLowerBounds;
	uiMIDPos = (uiUpperBounds + uiLowerBounds + 1) / 2;
	uiCurrentPos = uiMIDPos;

	for (;;)
	{
		while (uiLBPos == uiMIDPos ||
					((iCompare = 
						fnCompare( pvBuffer, uiLBPos, uiCurrentPos)) < 0))
		{
			if( uiLBPos >= uiUpperBounds)
			{
				break;
			}
			uiLBPos++;
		}

		while( uiUBPos == uiMIDPos ||
					(((iCompare = 
						fnCompare( pvBuffer, uiCurrentPos, uiUBPos)) < 0)))
		{
			if (!uiUBPos)
			{
				break;
			}
			uiUBPos--;
		}
		
		if( uiLBPos < uiUBPos)
		{
			// Exchange [uiLBPos] with [uiUBPos].

			fnSwap( pvBuffer, uiLBPos, uiUBPos);
			uiLBPos++;
			uiUBPos--;
		}
		else
		{
			break;
		}
	}

	// Check for swap( LB, MID ) - cases 3 and 4

	if( uiLBPos < uiMIDPos )
	{

		// Exchange [uiLBPos] with [uiMIDPos]

		fnSwap( pvBuffer, uiMIDPos, uiLBPos);
		uiMIDPos = uiLBPos;
	}
	else if( uiMIDPos < uiUBPos )
	{
		// Exchange [uUBPos] with [uiMIDPos]

		fnSwap( pvBuffer, uiMIDPos, uiUBPos);
		uiMIDPos = uiUBPos;
	}

	// Check the left piece.

	uiLeftItems = (uiLowerBounds + 1 < uiMIDPos)
							? uiMIDPos - uiLowerBounds
							: 0;

	uiRightItems = (uiMIDPos + 1 < uiUpperBounds)
							? uiUpperBounds - uiMIDPos
							: 0;

	if( uiLeftItems < uiRightItems)
	{
		// Recurse on the LEFT side and goto the top on the RIGHT side.

		if( uiLeftItems)
		{
			f_qsort( pvBuffer, uiLowerBounds, uiMIDPos - 1, fnCompare, fnSwap);
		}

		uiLowerBounds = uiMIDPos + 1;
		goto Iterate_Larger_Half;
	}
	else if( uiLeftItems)
	{
		// Recurse on the RIGHT side and goto the top for the LEFT side.

		if (uiRightItems )
		{
			f_qsort( pvBuffer, uiMIDPos + 1, uiUpperBounds, fnCompare, fnSwap);
		}

		uiUpperBounds = uiMIDPos - 1;
		goto Iterate_Larger_Half;
	}
}

/***************************************************************************
Desc:
****************************************************************************/
FLMINT flmQSortUINTCompare(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	FLMUINT		uiLeft = *(((FLMUINT *)pvBuffer) + uiPos1);
	FLMUINT		uiRight = *(((FLMUINT *)pvBuffer) + uiPos2);

	if( uiLeft < uiRight)
	{
		return( -1);
	}
	else if( uiLeft > uiRight)
	{
		return( 1);
	}

	return( 0);
}

/***************************************************************************
Desc:
****************************************************************************/
void flmQSortUINTSwap(
	void *		pvBuffer,
	FLMUINT		uiPos1,
	FLMUINT		uiPos2)
{
	FLMUINT *	puiArray = (FLMUINT *)pvBuffer;
	FLMUINT		uiTmp = puiArray[ uiPos1];

	puiArray[ uiPos1] = puiArray[ uiPos2];
	puiArray[ uiPos2] = uiTmp;
}

/****************************************************************************
Desc:	Determine if a token is a number.
****************************************************************************/
FLMBOOL tokenIsNum(
	const char *	pszToken,
	FLMUINT *		puiNum)
{
	FLMBOOL		bIsNum = TRUE;
	FLMUINT		uiNum;
	FLMBOOL		bAllowHex = FALSE;

	if (*pszToken == 0)
	{
		bIsNum = FALSE;
		goto Exit;
	}

	if (*pszToken == '0' && 
		(*(pszToken + 1) == 'x' || *(pszToken + 1) == 'X'))
	{
		pszToken += 2;
		bAllowHex = TRUE;
	}

	uiNum = 0;
	while (*pszToken)
	{
		if (*pszToken >= '0' && *pszToken <= '9')
		{
			if (!bAllowHex)
			{
				if (uiNum > (FLMUINT)(-1) / 10)
				{

					// Number would overflow.

					bIsNum = FALSE;
					goto Exit;
				}
				else
				{
					uiNum *= 10;
				}
			}
			else
			{
				if (uiNum > (FLMUINT)(-1) >> 4)
				{

					// Number would overflow.

					bIsNum = FALSE;
					goto Exit;
				}
				uiNum <<= 4;
			}
			uiNum += (FLMUINT)(*pszToken - '0');
		}
		else if (bAllowHex)
		{
			if (uiNum > (FLMUINT)(-1) >> 4)
			{

				// Number would overflow.

				bIsNum = FALSE;
				goto Exit;
			}
			if (*pszToken >= 'A' && *pszToken <= 'F')
			{
				uiNum <<= 4;
				uiNum += (FLMUINT)(*pszToken - 'A') + 10;
			}
			else if (*pszToken >= 'a' && *pszToken <= 'f')
			{
				uiNum <<= 4;
				uiNum += (FLMUINT)(*pszToken - 'a') + 10;
			}
			else
			{
				bIsNum = FALSE;
				goto Exit;
			}
		}
		else
		{
			bIsNum = FALSE;
			goto Exit;
		}
		pszToken++;
	}

	*puiNum = uiNum;

Exit:

	return( bIsNum);
}
