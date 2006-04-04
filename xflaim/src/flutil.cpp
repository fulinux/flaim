//------------------------------------------------------------------------------
// Desc:	Routines to support NATIVE to/from internal numeric types and
// 		string comparision/shift routines.
//
// Tabs:	3
//
//		Copyright (c) 1991-1993, 1996-1998, 2000, 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flutil.cpp 3114 2006-01-19 13:22:45 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/*
*	Macros used by f_wtoa(), f_dtoa(), f_uwtoa(), and f_udtoa()
*/

#define HANDLE_NEGATIVE					/* output sign and make value positive */\
	if( value < 0) \
	{	*ptr++ = '-'; \
		absValue = (FLMUINT)(-(value)); \
	} \
	else absValue = (FLMUINT)value;

#define HANDLE_DNEGATIVE					/* output sign and make value positive */\
	if( value < 0) \
	{	*ptr++ = '-'; \
		absValue = (FLMUINT)(-(value)); \
	} \
	else absValue = (FLMUINT)value;

#define	PUSH_DIGITS( v) \
	{register FLMUINT reg = v; \
		do{ *sp++ = (char)((reg % 10) + '0'); /* convert and push */\
		} while( reg /= 10); \
	}

#define	POP_DIGITS \
	while( stack < sp--)					/* stack assumes post-inc & pre-dec */\
		*ptr++ = *sp; \
	*ptr = '\0';


/****************************************************************************
Desc:  	Unsigned word to NATIVE value - null terminate the native string
Return:	char pointer to the NULL byte in the native string
Notes: 	Radix not defined because it is not needed
****************************************************************************/
char * f_uwtoa(
	FLMUINT16	value,
	char *		ptr)
{
	char		stack[ 10];						/* max: 10 digits */
	char *	sp = stack;						/* stack pointer */

	PUSH_DIGITS( value);

	POP_DIGITS;

	return( ptr);								/* Return pointer to null */
}

/****************************************************************************
Desc:  	Native to UDWORD value.  Supports 0x<HEX> codes. Non digits NOT ALLOWED
			NO LEADING SPACES ALLOWED ! ! !  No checks for overflow over 4 bytes!
Return:	UDWORD value of what is being pointed to
Notes: 	This algorithm is NOT standard, assumes UNSIGNED char arithmetic
				so (20 - 30) should be 245 and NOT -10.
****************************************************************************/
FLMUINT f_atoud(
	char *  		pszBuf,
	FLMBOOL		bAllowUnprefixedHex)
{
	FLMUINT		uiValue;
	FLMBOOL		bAllowHex = FALSE;

	if( *pszBuf == NATIVE_ZERO &&
		(*(pszBuf + 1) == NATIVE_LOWER_X || *(pszBuf + 1) == NATIVE_UPPER_X))
	{
		pszBuf += 2;
		bAllowHex = TRUE;
	}
	else if( bAllowUnprefixedHex)
	{
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
Desc:  	Native to FLMUINT64 value.  Supports 0x<HEX> codes. Non digits 
			NOT ALLOWED NO LEADING SPACES ALLOWED! No checks for overflow
			over 4 bytes!
Return:	FLMUINT64 value of what is being pointed to
Notes: 	This algorithm is NOT standard, assumes UNSIGNED char arithmetic
			so (20 - 30) should be 245 and NOT -10.
****************************************************************************/
FLMUINT64 f_atou64(
	char *  	pszBuf)
{
	FLMUINT64	ui64Value;
	FLMBOOL		bAllowHex = FALSE;

	if( *pszBuf == NATIVE_ZERO &&
		(*(pszBuf + 1) == NATIVE_LOWER_X || *(pszBuf + 1) == NATIVE_UPPER_X))
	{
		pszBuf += 2;
		bAllowHex = TRUE;
	}

	ui64Value = 0;
	while( *pszBuf)
	{
		if( *pszBuf >= '0' && *pszBuf <= '9')
		{
			if( !bAllowHex)
			{
				ui64Value *= 10;
			}
			else
			{
				ui64Value <<= 4;
			}

			ui64Value += (FLMUINT64)(*pszBuf - '0');
		}
		else if( bAllowHex)
		{
			if( *pszBuf >= 'A' && *pszBuf <= 'F')
			{
				ui64Value <<= 4;
				ui64Value += (FLMUINT64)(*pszBuf - 'A') + 10;
			}
			else if( *pszBuf >= 'a' && *pszBuf <= 'f')
			{
				ui64Value <<= 4;
				ui64Value += (FLMUINT64)(*pszBuf - 'a') + 10;
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

	return( ui64Value);
}

/****************************************************************************
Desc:  	Unsigned double (4 byte) number to native value & null terminate
Return:	char pointer to the NULL byte in the native string
Notes: 	Radix not defined.
****************************************************************************/
char * f_udtoa(
	FLMUINT		value,
	char *		ptr)
{
	char 		stack[ 10];							/* max: 10 digits */
	char *	sp = stack;							/* stack pointer */

	PUSH_DIGITS( value);

	POP_DIGITS;

	return( ptr);									/* Return pointer to null */
}

/****************************************************************************
Desc:  	Word to native value - null terminate the native string
Return:	char pointer to the NULL byte in the native string
Notes: 	Radix not defined because it is not needed
****************************************************************************/
char * f_wtoa(
	FLMINT16		value,
	char *		ptr)
{
	char  		stack[ 10];							/* max: 10 digits values on stack */
	char *		sp = stack;							/* stack pointer */
	FLMUINT		absValue;          				/* algorithm assumes absolute value */

	HANDLE_NEGATIVE;

	PUSH_DIGITS( absValue);

	POP_DIGITS;

	return( ptr);									/* Return pointer to null terminator */
}

/****************************************************************************
Desc:  	Double (4 byte) number to native value - null terminate the string
Return:	char pointer to the NULL byte in the native string
****************************************************************************/
char * f_dtoa(
	FLMINT		value,
	char *		ptr)
{
	char			stack[ 10];		 				/* max: 10 digits values on stack */
	char *		sp = stack;						/* stack pointer */
	FLMUINT		absValue; 					/* algorithm assumes absolute value */

	HANDLE_DNEGATIVE;

	PUSH_DIGITS( absValue);

	POP_DIGITS;

	return( ptr);				  					/* Return pointer to null terminator */
}

/*****************************************************************************
Desc:	Convert unsigned 64 bit value to ASCII.
*****************************************************************************/
char * f_ui64toa(
	FLMUINT64	ui64Value,
	char *		pszAscii)
{
	char 		szStack [30];
	char *	pszStack = &szStack [0];
	
	do
	{
		*pszStack++ = (char)((ui64Value % 10) + '0');
	}
	while ((ui64Value /= 10) > 0);

	pszStack--;
	for (;;)
	{
		*pszAscii++ = *pszStack;
		if (pszStack == &szStack [0])
		{
			break;
		}
		pszStack--;
	}
	*pszAscii = 0;
	
	// Return pointer to terminating null character
	
	return( pszAscii);
}

/*****************************************************************************
Desc:	Convert signed 64 bit value to ASCII.
*****************************************************************************/
char * f_i64toa(
	FLMINT64	i64Value,
	char *	pszAscii)
{
	if (i64Value < 0)
	{
		*pszAscii++ = '-';
		i64Value = -i64Value;
	}
	return( f_ui64toa( (FLMUINT64)i64Value, pszAscii));
}

/****************************************************************************
Desc:	Ascii to integer
****************************************************************************/
FLMINT f_atoi(
	char *	pszStr)								/* Points to native number */
{
	return( f_atod( pszStr));
}

/****************************************************************************
Desc:	native to long
****************************************************************************/
FLMINT f_atol(
	char *	pszStr)								/* Points to native number */
{
	return( f_atod( pszStr));
}

/****************************************************************************
Desc:		native to DWORD value.  Supports 0x<HEX> codes. Non digits NOT ALLOWED
			NO LEADING SPACES ALLOWED ! ! !  No checks for overflow over 4 bytes!
Return:	DWORD value of what is being pointed to
Notes: 	This algorithm is NOT standard! Assumes UNSIGNED char arithmetic
			so (20 - 30) should be 245 and NOT -10.
****************************************************************************/
FLMINT f_atod(
	char *	pszBuf)
{
	FLMINT		iValue;
	FLMBOOL		bNeg = FALSE;

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
Desc: Copies a unicode string
****************************************************************************/
FLMUNICODE * f_unicpy(
	FLMUNICODE *	puzDestStr,
	FLMUNICODE *	puzSrcStr)
{
	FLMUNICODE *		puzSrc = puzSrcStr;
	FLMUNICODE *		puzDest = puzDestStr;

	while( *puzSrc)
	{
		*puzDest++ = *puzSrc++;
	}

	*puzDest = 0;
	return( puzDestStr);
}

/****************************************************************************
Desc: A rather trivial unicode monocase function.
****************************************************************************/
FLMUNICODE f_unitolower(
	FLMUNICODE		uChar)
{
	static const FLMUNICODE basicAlpha[ 0x600] =
	{
		0x0000, /* Monocases to self */
		0x0001, /* Monocases to self */
		0x0002, /* Monocases to self */
		0x0003, /* Monocases to self */
		0x0004, /* Monocases to self */
		0x0005, /* Monocases to self */
		0x0006, /* Monocases to self */
		0x0007, /* Monocases to self */
		0x0008, /* Monocases to self */
		0x0009, /* Monocases to self */
		0x000A, /* Monocases to self */
		0x000B, /* Monocases to self */
		0x000C, /* Monocases to self */
		0x000D, /* Monocases to self */
		0x000E, /* Monocases to self */
		0x000F, /* Monocases to self */
		0x0010, /* Monocases to self */
		0x0011, /* Monocases to self */
		0x0012, /* Monocases to self */
		0x0013, /* Monocases to self */
		0x0014, /* Monocases to self */
		0x0015, /* Monocases to self */
		0x0016, /* Monocases to self */
		0x0017, /* Monocases to self */
		0x0018, /* Monocases to self */
		0x0019, /* Monocases to self */
		0x001A, /* Monocases to self */
		0x001B, /* Monocases to self */
		0x001C, /* Monocases to self */
		0x001D, /* Monocases to self */
		0x001E, /* Monocases to self */
		0x001F, /* Monocases to self */
		0x0020, /* Monocases to self */
		0x0021, /* Monocases to self */
		0x0022, /* Monocases to self */
		0x0023, /* Monocases to self */
		0x0024, /* Monocases to self */
		0x0025, /* Monocases to self */
		0x0026, /* Monocases to self */
		0x0027, /* Monocases to self */
		0x0028, /* Monocases to self */
		0x0029, /* Monocases to self */
		0x002A, /* Monocases to self */
		0x002B, /* Monocases to self */
		0x002C, /* Monocases to self */
		0x002D, /* Monocases to self */
		0x002E, /* Monocases to self */
		0x002F, /* Monocases to self */
		0x0030, /* Monocases to self */
		0x0031, /* Monocases to self */
		0x0032, /* Monocases to self */
		0x0033, /* Monocases to self */
		0x0034, /* Monocases to self */
		0x0035, /* Monocases to self */
		0x0036, /* Monocases to self */
		0x0037, /* Monocases to self */
		0x0038, /* Monocases to self */
		0x0039, /* Monocases to self */
		0x003A, /* Monocases to self */
		0x003B, /* Monocases to self */
		0x003C, /* Monocases to self */
		0x003D, /* Monocases to self */
		0x003E, /* Monocases to self */
		0x003F, /* Monocases to self */
		0x0040, /* Monocases to self */
		0x0061, /* LATIN LETTER A */
		0x0062, /* LATIN LETTER B */
		0x0063, /* LATIN LETTER C */
		0x0064, /* LATIN LETTER D */
		0x0065, /* LATIN LETTER E */
		0x0066, /* LATIN LETTER F */
		0x0067, /* LATIN LETTER G */
		0x0068, /* LATIN LETTER H */
		0x0069, /* LATIN LETTER I */
		0x006A, /* LATIN LETTER J */
		0x006B, /* LATIN LETTER K */
		0x006C, /* LATIN LETTER L */
		0x006D, /* LATIN LETTER M */
		0x006E, /* LATIN LETTER N */
		0x006F, /* LATIN LETTER O */
		0x0070, /* LATIN LETTER P */
		0x0071, /* LATIN LETTER Q */
		0x0072, /* LATIN LETTER R */
		0x0073, /* LATIN LETTER S */
		0x0074, /* LATIN LETTER T */
		0x0075, /* LATIN LETTER U */
		0x0076, /* LATIN LETTER V */
		0x0077, /* LATIN LETTER W */
		0x0078, /* LATIN LETTER X */
		0x0079, /* LATIN LETTER Y */
		0x007A, /* LATIN LETTER Z */
		0x005B, /* Monocases to self */
		0x005C, /* Monocases to self */
		0x005D, /* Monocases to self */
		0x005E, /* Monocases to self */
		0x005F, /* Monocases to self */
		0x0060, /* Monocases to self */
		0x0061, /* Monocases to self */
		0x0062, /* Monocases to self */
		0x0063, /* Monocases to self */
		0x0064, /* Monocases to self */
		0x0065, /* Monocases to self */
		0x0066, /* Monocases to self */
		0x0067, /* Monocases to self */
		0x0068, /* Monocases to self */
		0x0069, /* Monocases to self */
		0x006A, /* Monocases to self */
		0x006B, /* Monocases to self */
		0x006C, /* Monocases to self */
		0x006D, /* Monocases to self */
		0x006E, /* Monocases to self */
		0x006F, /* Monocases to self */
		0x0070, /* Monocases to self */
		0x0071, /* Monocases to self */
		0x0072, /* Monocases to self */
		0x0073, /* Monocases to self */
		0x0074, /* Monocases to self */
		0x0075, /* Monocases to self */
		0x0076, /* Monocases to self */
		0x0077, /* Monocases to self */
		0x0078, /* Monocases to self */
		0x0079, /* Monocases to self */
		0x007A, /* Monocases to self */
		0x007B, /* Monocases to self */
		0x007C, /* Monocases to self */
		0x007D, /* Monocases to self */
		0x007E, /* Monocases to self */
		0x007F, /* Monocases to self */
		0x0080, /* Monocases to self */
		0x0081, /* Monocases to self */
		0x0082, /* Monocases to self */
		0x0083, /* Monocases to self */
		0x0084, /* Monocases to self */
		0x0085, /* Monocases to self */
		0x0086, /* Monocases to self */
		0x0087, /* Monocases to self */
		0x0088, /* Monocases to self */
		0x0089, /* Monocases to self */
		0x008A, /* Monocases to self */
		0x008B, /* Monocases to self */
		0x008C, /* Monocases to self */
		0x008D, /* Monocases to self */
		0x008E, /* Monocases to self */
		0x008F, /* Monocases to self */
		0x0090, /* Monocases to self */
		0x0091, /* Monocases to self */
		0x0092, /* Monocases to self */
		0x0093, /* Monocases to self */
		0x0094, /* Monocases to self */
		0x0095, /* Monocases to self */
		0x0096, /* Monocases to self */
		0x0097, /* Monocases to self */
		0x0098, /* Monocases to self */
		0x0099, /* Monocases to self */
		0x009A, /* Monocases to self */
		0x009B, /* Monocases to self */
		0x009C, /* Monocases to self */
		0x009D, /* Monocases to self */
		0x009E, /* Monocases to self */
		0x009F, /* Monocases to self */
		0x00A0, /* Monocases to self */
		0x00A1, /* Monocases to self */
		0x00A2, /* Monocases to self */
		0x00A3, /* Monocases to self */
		0x00A4, /* Monocases to self */
		0x00A5, /* Monocases to self */
		0x00A6, /* Monocases to self */
		0x00A7, /* Monocases to self */
		0x00A8, /* Monocases to self */
		0x00A9, /* Monocases to self */
		0x00AA, /* Monocases to self */
		0x00AB, /* Monocases to self */
		0x00AC, /* Monocases to self */
		0x00AD, /* Monocases to self */
		0x00AE, /* Monocases to self */
		0x00AF, /* Monocases to self */
		0x00B0, /* Monocases to self */
		0x00B1, /* Monocases to self */
		0x00B2, /* Monocases to self */
		0x00B3, /* Monocases to self */
		0x00B4, /* Monocases to self */
		0x00B5, /* Monocases to self */
		0x00B6, /* Monocases to self */
		0x00B7, /* Monocases to self */
		0x00B8, /* Monocases to self */
		0x00B9, /* Monocases to self */
		0x00BA, /* Monocases to self */
		0x00BB, /* Monocases to self */
		0x00BC, /* Monocases to self */
		0x00BD, /* Monocases to self */
		0x00BE, /* Monocases to self */
		0x00BF, /* Monocases to self */
		0x00E0, /* LATIN LETTER A GRAVE */
		0x00E1, /* LATIN LETTER A ACUTE */
		0x00E2, /* LATIN LETTER A CIRCUMFLEX */
		0x00E3, /* LATIN LETTER A TILDE */
		0x00E4, /* LATIN LETTER A DIAERESIS */
		0x00E5, /* LATIN LETTER A RING */
		0x00E6, /* LATIN LETTER A E */
		0x00E7, /* LATIN LETTER C CEDILLA */
		0x00E8, /* LATIN LETTER E GRAVE */
		0x00E9, /* LATIN LETTER E ACUTE */
		0x00EA, /* LATIN LETTER E CIRCUMFLEX */
		0x00EB, /* LATIN LETTER E DIAERESIS */
		0x00EC, /* LATIN LETTER I GRAVE */
		0x00ED, /* LATIN LETTER I ACUTE */
		0x00EE, /* LATIN LETTER I CIRCUMFLEX */
		0x00EF, /* LATIN LETTER I DIAERESIS */
		0x00F0, /* LATIN LETTER ETH */
		0x00F1, /* LATIN LETTER N TILDE */
		0x00F2, /* LATIN LETTER O GRAVE */
		0x00F3, /* LATIN LETTER O ACUTE */
		0x00F4, /* LATIN LETTER O CIRCUMFLEX */
		0x00F5, /* LATIN LETTER O TILDE */
		0x00F6, /* LATIN LETTER O DIAERESIS */
		0x00D7, /* Monocases to self */
		0x00F8, /* LATIN LETTER O SLASH */
		0x00F9, /* LATIN LETTER U GRAVE */
		0x00FA, /* LATIN LETTER U ACUTE */
		0x00FB, /* LATIN LETTER U CIRCUMFLEX */
		0x00FC, /* LATIN LETTER U DIAERESIS */
		0x00FD, /* LATIN LETTER Y ACUTE */
		0x00FE, /* LATIN LETTER THORN */
		0x00DF, /* Monocases to self */
		0x00E0, /* Monocases to self */
		0x00E1, /* Monocases to self */
		0x00E2, /* Monocases to self */
		0x00E3, /* Monocases to self */
		0x00E4, /* Monocases to self */
		0x00E5, /* Monocases to self */
		0x00E6, /* Monocases to self */
		0x00E7, /* Monocases to self */
		0x00E8, /* Monocases to self */
		0x00E9, /* Monocases to self */
		0x00EA, /* Monocases to self */
		0x00EB, /* Monocases to self */
		0x00EC, /* Monocases to self */
		0x00ED, /* Monocases to self */
		0x00EE, /* Monocases to self */
		0x00EF, /* Monocases to self */
		0x00F0, /* Monocases to self */
		0x00F1, /* Monocases to self */
		0x00F2, /* Monocases to self */
		0x00F3, /* Monocases to self */
		0x00F4, /* Monocases to self */
		0x00F5, /* Monocases to self */
		0x00F6, /* Monocases to self */
		0x00F7, /* Monocases to self */
		0x00F8, /* Monocases to self */
		0x00F9, /* Monocases to self */
		0x00FA, /* Monocases to self */
		0x00FB, /* Monocases to self */
		0x00FC, /* Monocases to self */
		0x00FD, /* Monocases to self */
		0x00FE, /* Monocases to self */
		0x00FF, /* Monocases to self */
		0x0101, /* LATIN LETTER A MACRON */
		0x0101, /* Monocases to self */
		0x0103, /* LATIN LETTER A BREVE */
		0x0103, /* Monocases to self */
		0x0105, /* LATIN LETTER A OGONEK */
		0x0105, /* Monocases to self */
		0x0107, /* LATIN LETTER C ACUTE */
		0x0107, /* Monocases to self */
		0x0109, /* LATIN LETTER C CIRCUMFLEX */
		0x0109, /* Monocases to self */
		0x010B, /* LATIN LETTER C DOT */
		0x010B, /* Monocases to self */
		0x010D, /* LATIN LETTER C HACEK */
		0x010D, /* Monocases to self */
		0x010F, /* LATIN LETTER D HACEK */
		0x010F, /* Monocases to self */
		0x0111, /* LATIN LETTER D BAR */
		0x0111, /* Monocases to self */
		0x0113, /* LATIN LETTER E MACRON */
		0x0113, /* Monocases to self */
		0x0115, /* LATIN LETTER E BREVE */
		0x0115, /* Monocases to self */
		0x0117, /* LATIN LETTER E DOT */
		0x0117, /* Monocases to self */
		0x0119, /* LATIN LETTER E OGONEK */
		0x0119, /* Monocases to self */
		0x011B, /* LATIN LETTER E HACEK */
		0x011B, /* Monocases to self */
		0x011D, /* LATIN LETTER G CIRCUMFLEX */
		0x011D, /* Monocases to self */
		0x011F, /* LATIN LETTER G BREVE */
		0x011F, /* Monocases to self */
		0x0121, /* LATIN LETTER G DOT */
		0x0121, /* Monocases to self */
		0x0123, /* LATIN LETTER G CEDILLA */
		0x0123, /* Monocases to self */
		0x0125, /* LATIN LETTER H CIRCUMFLEX */
		0x0125, /* Monocases to self */
		0x0127, /* LATIN LETTER H BAR */
		0x0127, /* Monocases to self */
		0x0129, /* LATIN LETTER I TILDE */
		0x0129, /* Monocases to self */
		0x012B, /* LATIN LETTER I MACRON */
		0x012B, /* Monocases to self */
		0x012D, /* LATIN LETTER I BREVE */
		0x012D, /* Monocases to self */
		0x012F, /* LATIN LETTER I OGONEK */
		0x012F, /* Monocases to self */
		0x0069, /* LATIN LETTER I DOT */
		0x0131, /* Monocases to self */
		0x0133, /* LATIN LETTER I J */
		0x0133, /* Monocases to self */
		0x0135, /* LATIN LETTER J CIRCUMFLEX */
		0x0135, /* Monocases to self */
		0x0137, /* LATIN LETTER K CEDILLA */
		0x0137, /* Monocases to self */
		0x0138, /* Monocases to self */
		0x013A, /* LATIN LETTER L ACUTE */
		0x013A, /* Monocases to self */
		0x013C, /* LATIN LETTER L CEDILLA */
		0x013C, /* Monocases to self */
		0x013E, /* LATIN LETTER L HACEK */
		0x013E, /* Monocases to self */
		0x0140, /* LATIN LETTER L WITH MIDDLE DOT */
		0x0140, /* Monocases to self */
		0x0142, /* LATIN LETTER L SLASH */
		0x0142, /* Monocases to self */
		0x0144, /* LATIN LETTER N ACUTE */
		0x0144, /* Monocases to self */
		0x0146, /* LATIN LETTER N CEDILLA */
		0x0146, /* Monocases to self */
		0x0148, /* LATIN LETTER N HACEK */
		0x0148, /* Monocases to self */
		0x0149, /* Monocases to self */
		0x014B, /* LATIN LETTER ENG */
		0x014B, /* Monocases to self */
		0x014D, /* LATIN LETTER O MACRON */
		0x014D, /* Monocases to self */
		0x014F, /* LATIN LETTER O BREVE */
		0x014F, /* Monocases to self */
		0x0151, /* LATIN LETTER O DOUBLE ACUTE */
		0x0151, /* Monocases to self */
		0x0153, /* LATIN LETTER O E */
		0x0153, /* Monocases to self */
		0x0155, /* LATIN LETTER R ACUTE */
		0x0155, /* Monocases to self */
		0x0157, /* LATIN LETTER R CEDILLA */
		0x0157, /* Monocases to self */
		0x0159, /* LATIN LETTER R HACEK */
		0x0159, /* Monocases to self */
		0x015B, /* LATIN LETTER S ACUTE */
		0x015B, /* Monocases to self */
		0x015D, /* LATIN LETTER S CIRCUMFLEX */
		0x015D, /* Monocases to self */
		0x015F, /* LATIN LETTER S CEDILLA */
		0x015F, /* Monocases to self */
		0x0161, /* LATIN LETTER S HACEK */
		0x0161, /* Monocases to self */
		0x0163, /* LATIN LETTER T CEDILLA */
		0x0163, /* Monocases to self */
		0x0165, /* LATIN LETTER T HACEK */
		0x0165, /* Monocases to self */
		0x0167, /* LATIN LETTER T BAR */
		0x0167, /* Monocases to self */
		0x0169, /* LATIN LETTER U TILDE */
		0x0169, /* Monocases to self */
		0x016B, /* LATIN LETTER U MACRON */
		0x016B, /* Monocases to self */
		0x016D, /* LATIN LETTER U BREVE */
		0x016D, /* Monocases to self */
		0x016F, /* LATIN LETTER U RING */
		0x016F, /* Monocases to self */
		0x0171, /* LATIN LETTER U DOUBLE ACUTE */
		0x0171, /* Monocases to self */
		0x0173, /* LATIN LETTER U OGONEK */
		0x0173, /* Monocases to self */
		0x0175, /* LATIN LETTER W CIRCUMFLEX */
		0x0175, /* Monocases to self */
		0x0177, /* LATIN LETTER Y CIRCUMFLEX */
		0x0177, /* Monocases to self */
		0x00FF, /* LATIN LETTER Y DIAERESIS */
		0x017A, /* LATIN LETTER Z ACUTE */
		0x017A, /* Monocases to self */
		0x017C, /* LATIN LETTER Z DOT */
		0x017C, /* Monocases to self */
		0x017E, /* LATIN LETTER Z HACEK */
		0x017E, /* Monocases to self */
		0x017F, /* Monocases to self */
		0x0180, /* Monocases to self */
		0x0253, /* LATIN LETTER B HOOK */
		0x0183, /* LATIN LETTER B TOPBAR */
		0x0183, /* Monocases to self */
		0x0185, /* LATIN LETTER TONE SIX */
		0x0185, /* Monocases to self */
		0x0254, /* LATIN LETTER OPEN O */
		0x0188, /* LATIN LETTER C HOOK */
		0x0188, /* Monocases to self */
		0x0256, /* LATIN LETTER AFRICAN D */
		0x0257, /* LATIN LETTER D HOOK */
		0x018C, /* LATIN LETTER D TOPBAR */
		0x018C, /* Monocases to self */
		0x018D, /* Monocases to self */
		0x01DD, /* LATIN LETTER TURNED E */
		0x0259, /* LATIN LETTER SCHWA */
		0x025B, /* LATIN LETTER EPSILON */
		0x0192, /* LATIN LETTER F HOOK */
		0x0192, /* Monocases to self */
		0x0260, /* LATIN LETTER G HOOK */
		0x0263, /* LATIN LETTER GAMMA */
		0x0195, /* Monocases to self */
		0x0269, /* LATIN LETTER IOTA */
		0x0268, /* LATIN LETTER BARRED I */
		0x0199, /* LATIN LETTER K HOOK */
		0x0199, /* Monocases to self */
		0x019A, /* Monocases to self */
		0x019B, /* Monocases to self */
		0x026F, /* LATIN LETTER TURNED M */
		0x0272, /* LATIN LETTER N HOOK */
		0x019E, /* Monocases to self */
		0x0275, /* LATIN LETTER BARRED O */
		0x01A1, /* LATIN LETTER O HORN */
		0x01A1, /* Monocases to self */
		0x01A3, /* LATIN LETTER O I */
		0x01A3, /* Monocases to self */
		0x01A5, /* LATIN LETTER P HOOK */
		0x01A5, /* Monocases to self */
		0x01A6, /* Monocases to self */
		0x01A8, /* LATIN LETTER TONE TWO */
		0x01A8, /* Monocases to self */
		0x0283, /* LATIN LETTER ESH */
		0x01AA, /* Monocases to self */
		0x01AB, /* Monocases to self */
		0x01AD, /* LATIN LETTER T HOOK */
		0x01AD, /* Monocases to self */
		0x0288, /* LATIN LETTER T RETROFLEX HOOK */
		0x01B0, /* LATIN LETTER U HORN */
		0x01B0, /* Monocases to self */
		0x028A, /* LATIN LETTER UPSILON */
		0x028B, /* LATIN LETTER SCRIPT V */
		0x01B4, /* LATIN LETTER Y HOOK */
		0x01B4, /* Monocases to self */
		0x01B6, /* LATIN LETTER Z BAR */
		0x01B6, /* Monocases to self */
		0x0292, /* LATIN LETTER YOGH */
		0x01B9, /* LATIN LETTER REVERSED YOGH */
		0x01B9, /* Monocases to self */
		0x01BA, /* Monocases to self */
		0x01BB, /* Monocases to self */
		0x01BD, /* LATIN LETTER TONE FIVE */
		0x01BD, /* Monocases to self */
		0x01BE, /* Monocases to self */
		0x01BF, /* Monocases to self */
		0x01C0, /* Monocases to self */
		0x01C1, /* Monocases to self */
		0x01C2, /* Monocases to self */
		0x01C3, /* Monocases to self */
		0x01C6, /* LATIN LETTER D Z HACEK */
		0x01C6, /* LATIN LETTER CAPITAL D SMALL Z HACEK */
		0x01C6, /* Monocases to self */
		0x01C9, /* LATIN LETTER CAPITAL L CAPTIAL J */
		0x01C9, /* LATIN LETTER CAPITAL L SMALL J */
		0x01C9, /* Monocases to self */
		0x01CC, /* LATIN LETTER CAPITAL N CAPITAL J */
		0x01CC, /* LATIN LETTER CAPITAL N SMALL J */
		0x01CC, /* Monocases to self */
		0x01CE, /* LATIN LETTER A HACEK */
		0x01CE, /* Monocases to self */
		0x01D0, /* LATIN LETTER I HACEK */
		0x01D0, /* Monocases to self */
		0x01D2, /* LATIN LETTER O HACEK */
		0x01D2, /* Monocases to self */
		0x01D4, /* LATIN LETTER U HACEK */
		0x01D4, /* Monocases to self */
		0x01D6, /* LATIN LETTER U DIAERESIS MACRON */
		0x01D6, /* Monocases to self */
		0x01D8, /* LATIN LETTER U DIAERESIS ACUTE */
		0x01D8, /* Monocases to self */
		0x01DA, /* LATIN LETTER U DIAERESIS HACEK */
		0x01DA, /* Monocases to self */
		0x01DC, /* LATIN LETTER U DIAERESIS GRAVE */
		0x01DC, /* Monocases to self */
		0x01DD, /* Monocases to self */
		0x01DF, /* LATIN LETTER A DIAERESIS MACRON */
		0x01DF, /* Monocases to self */
		0x01E1, /* LATIN LETTER A DOT MACRON */
		0x01E1, /* Monocases to self */
		0x01E3, /* LATIN LETTER A E MACRON */
		0x01E3, /* Monocases to self */
		0x01E5, /* LATIN LETTER G BAR */
		0x01E5, /* Monocases to self */
		0x01E7, /* LATIN LETTER G HACEK */
		0x01E7, /* Monocases to self */
		0x01E9, /* LATIN LETTER K HACEK */
		0x01E9, /* Monocases to self */
		0x01EB, /* LATIN LETTER O OGONEK */
		0x01EB, /* Monocases to self */
		0x01ED, /* LATIN LETTER O OGONEK MACRON */
		0x01ED, /* Monocases to self */
		0x01EF, /* LATIN LETTER YOGH HACEK */
		0x01EF, /* Monocases to self */
		0x01F0, /* Monocases to self */
		0x01F1, /* Monocases to self */
		0x01F2, /* Monocases to self */
		0x01F3, /* Monocases to self */
		0x01F4, /* Monocases to self */
		0x01F5, /* Monocases to self */
		0x01F6, /* Monocases to self */
		0x01F7, /* Monocases to self */
		0x01F8, /* Monocases to self */
		0x01F9, /* Monocases to self */
		0x01FA, /* Monocases to self */
		0x01FB, /* Monocases to self */
		0x01FC, /* Monocases to self */
		0x01FD, /* Monocases to self */
		0x01FE, /* Monocases to self */
		0x01FF, /* Monocases to self */
		0x0200, /* Monocases to self */
		0x0201, /* Monocases to self */
		0x0202, /* Monocases to self */
		0x0203, /* Monocases to self */
		0x0204, /* Monocases to self */
		0x0205, /* Monocases to self */
		0x0206, /* Monocases to self */
		0x0207, /* Monocases to self */
		0x0208, /* Monocases to self */
		0x0209, /* Monocases to self */
		0x020A, /* Monocases to self */
		0x020B, /* Monocases to self */
		0x020C, /* Monocases to self */
		0x020D, /* Monocases to self */
		0x020E, /* Monocases to self */
		0x020F, /* Monocases to self */
		0x0210, /* Monocases to self */
		0x0211, /* Monocases to self */
		0x0212, /* Monocases to self */
		0x0213, /* Monocases to self */
		0x0214, /* Monocases to self */
		0x0215, /* Monocases to self */
		0x0216, /* Monocases to self */
		0x0217, /* Monocases to self */
		0x0218, /* Monocases to self */
		0x0219, /* Monocases to self */
		0x021A, /* Monocases to self */
		0x021B, /* Monocases to self */
		0x021C, /* Monocases to self */
		0x021D, /* Monocases to self */
		0x021E, /* Monocases to self */
		0x021F, /* Monocases to self */
		0x0220, /* Monocases to self */
		0x0221, /* Monocases to self */
		0x0222, /* Monocases to self */
		0x0223, /* Monocases to self */
		0x0224, /* Monocases to self */
		0x0225, /* Monocases to self */
		0x0226, /* Monocases to self */
		0x0227, /* Monocases to self */
		0x0228, /* Monocases to self */
		0x0229, /* Monocases to self */
		0x022A, /* Monocases to self */
		0x022B, /* Monocases to self */
		0x022C, /* Monocases to self */
		0x022D, /* Monocases to self */
		0x022E, /* Monocases to self */
		0x022F, /* Monocases to self */
		0x0230, /* Monocases to self */
		0x0231, /* Monocases to self */
		0x0232, /* Monocases to self */
		0x0233, /* Monocases to self */
		0x0234, /* Monocases to self */
		0x0235, /* Monocases to self */
		0x0236, /* Monocases to self */
		0x0237, /* Monocases to self */
		0x0238, /* Monocases to self */
		0x0239, /* Monocases to self */
		0x023A, /* Monocases to self */
		0x023B, /* Monocases to self */
		0x023C, /* Monocases to self */
		0x023D, /* Monocases to self */
		0x023E, /* Monocases to self */
		0x023F, /* Monocases to self */
		0x0240, /* Monocases to self */
		0x0241, /* Monocases to self */
		0x0242, /* Monocases to self */
		0x0243, /* Monocases to self */
		0x0244, /* Monocases to self */
		0x0245, /* Monocases to self */
		0x0246, /* Monocases to self */
		0x0247, /* Monocases to self */
		0x0248, /* Monocases to self */
		0x0249, /* Monocases to self */
		0x024A, /* Monocases to self */
		0x024B, /* Monocases to self */
		0x024C, /* Monocases to self */
		0x024D, /* Monocases to self */
		0x024E, /* Monocases to self */
		0x024F, /* Monocases to self */
		0x0250, /* Monocases to self */
		0x0251, /* Monocases to self */
		0x0252, /* Monocases to self */
		0x0253, /* Monocases to self */
		0x0254, /* Monocases to self */
		0x0255, /* Monocases to self */
		0x0256, /* Monocases to self */
		0x0257, /* Monocases to self */
		0x0258, /* Monocases to self */
		0x0259, /* Monocases to self */
		0x025A, /* Monocases to self */
		0x025B, /* Monocases to self */
		0x025C, /* Monocases to self */
		0x025D, /* Monocases to self */
		0x025E, /* Monocases to self */
		0x025F, /* Monocases to self */
		0x0260, /* Monocases to self */
		0x0261, /* Monocases to self */
		0x0262, /* Monocases to self */
		0x0263, /* Monocases to self */
		0x0264, /* Monocases to self */
		0x0265, /* Monocases to self */
		0x0266, /* Monocases to self */
		0x0267, /* Monocases to self */
		0x0268, /* Monocases to self */
		0x0269, /* Monocases to self */
		0x026A, /* Monocases to self */
		0x026B, /* Monocases to self */
		0x026C, /* Monocases to self */
		0x026D, /* Monocases to self */
		0x026E, /* Monocases to self */
		0x026F, /* Monocases to self */
		0x0270, /* Monocases to self */
		0x0271, /* Monocases to self */
		0x0272, /* Monocases to self */
		0x0273, /* Monocases to self */
		0x0274, /* Monocases to self */
		0x0275, /* Monocases to self */
		0x0276, /* Monocases to self */
		0x0277, /* Monocases to self */
		0x0278, /* Monocases to self */
		0x0279, /* Monocases to self */
		0x027A, /* Monocases to self */
		0x027B, /* Monocases to self */
		0x027C, /* Monocases to self */
		0x027D, /* Monocases to self */
		0x027E, /* Monocases to self */
		0x027F, /* Monocases to self */
		0x0280, /* Monocases to self */
		0x0281, /* Monocases to self */
		0x0282, /* Monocases to self */
		0x0283, /* Monocases to self */
		0x0284, /* Monocases to self */
		0x0285, /* Monocases to self */
		0x0286, /* Monocases to self */
		0x0287, /* Monocases to self */
		0x0288, /* Monocases to self */
		0x0289, /* Monocases to self */
		0x028A, /* Monocases to self */
		0x028B, /* Monocases to self */
		0x028C, /* Monocases to self */
		0x028D, /* Monocases to self */
		0x028E, /* Monocases to self */
		0x028F, /* Monocases to self */
		0x0290, /* Monocases to self */
		0x0291, /* Monocases to self */
		0x0292, /* Monocases to self */
		0x0293, /* Monocases to self */
		0x0294, /* Monocases to self */
		0x0295, /* Monocases to self */
		0x0296, /* Monocases to self */
		0x0297, /* Monocases to self */
		0x0298, /* Monocases to self */
		0x0299, /* Monocases to self */
		0x029A, /* Monocases to self */
		0x029B, /* Monocases to self */
		0x029C, /* Monocases to self */
		0x029D, /* Monocases to self */
		0x029E, /* Monocases to self */
		0x029F, /* Monocases to self */
		0x02A0, /* Monocases to self */
		0x02A1, /* Monocases to self */
		0x02A2, /* Monocases to self */
		0x02A3, /* Monocases to self */
		0x02A4, /* Monocases to self */
		0x02A5, /* Monocases to self */
		0x02A6, /* Monocases to self */
		0x02A7, /* Monocases to self */
		0x02A8, /* Monocases to self */
		0x02A9, /* Monocases to self */
		0x02AA, /* Monocases to self */
		0x02AB, /* Monocases to self */
		0x02AC, /* Monocases to self */
		0x02AD, /* Monocases to self */
		0x02AE, /* Monocases to self */
		0x02AF, /* Monocases to self */
		0x02B0, /* Monocases to self */
		0x02B1, /* Monocases to self */
		0x02B2, /* Monocases to self */
		0x02B3, /* Monocases to self */
		0x02B4, /* Monocases to self */
		0x02B5, /* Monocases to self */
		0x02B6, /* Monocases to self */
		0x02B7, /* Monocases to self */
		0x02B8, /* Monocases to self */
		0x02B9, /* Monocases to self */
		0x02BA, /* Monocases to self */
		0x02BB, /* Monocases to self */
		0x02BC, /* Monocases to self */
		0x02BD, /* Monocases to self */
		0x02BE, /* Monocases to self */
		0x02BF, /* Monocases to self */
		0x02C0, /* Monocases to self */
		0x02C1, /* Monocases to self */
		0x02C2, /* Monocases to self */
		0x02C3, /* Monocases to self */
		0x02C4, /* Monocases to self */
		0x02C5, /* Monocases to self */
		0x02C6, /* Monocases to self */
		0x02C7, /* Monocases to self */
		0x02C8, /* Monocases to self */
		0x02C9, /* Monocases to self */
		0x02CA, /* Monocases to self */
		0x02CB, /* Monocases to self */
		0x02CC, /* Monocases to self */
		0x02CD, /* Monocases to self */
		0x02CE, /* Monocases to self */
		0x02CF, /* Monocases to self */
		0x02D0, /* Monocases to self */
		0x02D1, /* Monocases to self */
		0x02D2, /* Monocases to self */
		0x02D3, /* Monocases to self */
		0x02D4, /* Monocases to self */
		0x02D5, /* Monocases to self */
		0x02D6, /* Monocases to self */
		0x02D7, /* Monocases to self */
		0x02D8, /* Monocases to self */
		0x02D9, /* Monocases to self */
		0x02DA, /* Monocases to self */
		0x02DB, /* Monocases to self */
		0x02DC, /* Monocases to self */
		0x02DD, /* Monocases to self */
		0x02DE, /* Monocases to self */
		0x02DF, /* Monocases to self */
		0x02E0, /* Monocases to self */
		0x02E1, /* Monocases to self */
		0x02E2, /* Monocases to self */
		0x02E3, /* Monocases to self */
		0x02E4, /* Monocases to self */
		0x02E5, /* Monocases to self */
		0x02E6, /* Monocases to self */
		0x02E7, /* Monocases to self */
		0x02E8, /* Monocases to self */
		0x02E9, /* Monocases to self */
		0x02EA, /* Monocases to self */
		0x02EB, /* Monocases to self */
		0x02EC, /* Monocases to self */
		0x02ED, /* Monocases to self */
		0x02EE, /* Monocases to self */
		0x02EF, /* Monocases to self */
		0x02F0, /* Monocases to self */
		0x02F1, /* Monocases to self */
		0x02F2, /* Monocases to self */
		0x02F3, /* Monocases to self */
		0x02F4, /* Monocases to self */
		0x02F5, /* Monocases to self */
		0x02F6, /* Monocases to self */
		0x02F7, /* Monocases to self */
		0x02F8, /* Monocases to self */
		0x02F9, /* Monocases to self */
		0x02FA, /* Monocases to self */
		0x02FB, /* Monocases to self */
		0x02FC, /* Monocases to self */
		0x02FD, /* Monocases to self */
		0x02FE, /* Monocases to self */
		0x02FF, /* Monocases to self */
		0x0300, /* Monocases to self */
		0x0301, /* Monocases to self */
		0x0302, /* Monocases to self */
		0x0303, /* Monocases to self */
		0x0304, /* Monocases to self */
		0x0305, /* Monocases to self */
		0x0306, /* Monocases to self */
		0x0307, /* Monocases to self */
		0x0308, /* Monocases to self */
		0x0309, /* Monocases to self */
		0x030A, /* Monocases to self */
		0x030B, /* Monocases to self */
		0x030C, /* Monocases to self */
		0x030D, /* Monocases to self */
		0x030E, /* Monocases to self */
		0x030F, /* Monocases to self */
		0x0310, /* Monocases to self */
		0x0311, /* Monocases to self */
		0x0312, /* Monocases to self */
		0x0313, /* Monocases to self */
		0x0314, /* Monocases to self */
		0x0315, /* Monocases to self */
		0x0316, /* Monocases to self */
		0x0317, /* Monocases to self */
		0x0318, /* Monocases to self */
		0x0319, /* Monocases to self */
		0x031A, /* Monocases to self */
		0x031B, /* Monocases to self */
		0x031C, /* Monocases to self */
		0x031D, /* Monocases to self */
		0x031E, /* Monocases to self */
		0x031F, /* Monocases to self */
		0x0320, /* Monocases to self */
		0x0321, /* Monocases to self */
		0x0322, /* Monocases to self */
		0x0323, /* Monocases to self */
		0x0324, /* Monocases to self */
		0x0325, /* Monocases to self */
		0x0326, /* Monocases to self */
		0x0327, /* Monocases to self */
		0x0328, /* Monocases to self */
		0x0329, /* Monocases to self */
		0x032A, /* Monocases to self */
		0x032B, /* Monocases to self */
		0x032C, /* Monocases to self */
		0x032D, /* Monocases to self */
		0x032E, /* Monocases to self */
		0x032F, /* Monocases to self */
		0x0330, /* Monocases to self */
		0x0331, /* Monocases to self */
		0x0332, /* Monocases to self */
		0x0333, /* Monocases to self */
		0x0334, /* Monocases to self */
		0x0335, /* Monocases to self */
		0x0336, /* Monocases to self */
		0x0337, /* Monocases to self */
		0x0338, /* Monocases to self */
		0x0339, /* Monocases to self */
		0x033A, /* Monocases to self */
		0x033B, /* Monocases to self */
		0x033C, /* Monocases to self */
		0x033D, /* Monocases to self */
		0x033E, /* Monocases to self */
		0x033F, /* Monocases to self */
		0x0340, /* Monocases to self */
		0x0341, /* Monocases to self */
		0x0342, /* Monocases to self */
		0x0343, /* Monocases to self */
		0x0344, /* Monocases to self */
		0x0345, /* Monocases to self */
		0x0346, /* Monocases to self */
		0x0347, /* Monocases to self */
		0x0348, /* Monocases to self */
		0x0349, /* Monocases to self */
		0x034A, /* Monocases to self */
		0x034B, /* Monocases to self */
		0x034C, /* Monocases to self */
		0x034D, /* Monocases to self */
		0x034E, /* Monocases to self */
		0x034F, /* Monocases to self */
		0x0350, /* Monocases to self */
		0x0351, /* Monocases to self */
		0x0352, /* Monocases to self */
		0x0353, /* Monocases to self */
		0x0354, /* Monocases to self */
		0x0355, /* Monocases to self */
		0x0356, /* Monocases to self */
		0x0357, /* Monocases to self */
		0x0358, /* Monocases to self */
		0x0359, /* Monocases to self */
		0x035A, /* Monocases to self */
		0x035B, /* Monocases to self */
		0x035C, /* Monocases to self */
		0x035D, /* Monocases to self */
		0x035E, /* Monocases to self */
		0x035F, /* Monocases to self */
		0x0360, /* Monocases to self */
		0x0361, /* Monocases to self */
		0x0362, /* Monocases to self */
		0x0363, /* Monocases to self */
		0x0364, /* Monocases to self */
		0x0365, /* Monocases to self */
		0x0366, /* Monocases to self */
		0x0367, /* Monocases to self */
		0x0368, /* Monocases to self */
		0x0369, /* Monocases to self */
		0x036A, /* Monocases to self */
		0x036B, /* Monocases to self */
		0x036C, /* Monocases to self */
		0x036D, /* Monocases to self */
		0x036E, /* Monocases to self */
		0x036F, /* Monocases to self */
		0x0370, /* Monocases to self */
		0x0371, /* Monocases to self */
		0x0372, /* Monocases to self */
		0x0373, /* Monocases to self */
		0x0374, /* Monocases to self */
		0x0375, /* Monocases to self */
		0x0376, /* Monocases to self */
		0x0377, /* Monocases to self */
		0x0378, /* Monocases to self */
		0x0379, /* Monocases to self */
		0x037A, /* Monocases to self */
		0x037B, /* Monocases to self */
		0x037C, /* Monocases to self */
		0x037D, /* Monocases to self */
		0x037E, /* Monocases to self */
		0x037F, /* Monocases to self */
		0x0380, /* Monocases to self */
		0x0381, /* Monocases to self */
		0x0382, /* Monocases to self */
		0x0383, /* Monocases to self */
		0x0384, /* Monocases to self */
		0x0385, /* Monocases to self */
		0x03AC, /* GREEK LETTER ALPHA TONOS */
		0x0387, /* Monocases to self */
		0x03AD, /* GREEK LETTER EPSILON TONOS */
		0x03AE, /* GREEK LETTER ETA TONOS */
		0x03AF, /* GREEK LETTER IOTA TONOS */
		0x038B, /* Monocases to self */
		0x03CC, /* GREEK LETTER OMICRON TONOS */
		0x038D, /* Monocases to self */
		0x03CD, /* GREEK LETTER UPSILON TONOS */
		0x03CE, /* GREEK LETTER OMEGA TONOS */
		0x0390, /* Monocases to self */
		0x03B1, /* GREEK LETTER ALPHA */
		0x03B2, /* GREEK LETTER BETA */
		0x03B3, /* GREEK LETTER GAMMA */
		0x03B4, /* GREEK LETTER DELTA */
		0x03B5, /* GREEK LETTER EPSILON */
		0x03B6, /* GREEK LETTER ZETA */
		0x03B7, /* GREEK LETTER ETA */
		0x03B8, /* GREEK LETTER THETA */
		0x03B9, /* GREEK LETTER IOTA */
		0x03BA, /* GREEK LETTER KAPPA */
		0x03BB, /* GREEK LETTER LAMBDA */
		0x03BC, /* GREEK LETTER MU */
		0x03BD, /* GREEK LETTER NU */
		0x03BE, /* GREEK LETTER Xl */
		0x03BF, /* GREEK LETTER OMICRON */
		0x03C0, /* GREEK LETTER PI */
		0x03C1, /* GREEK LETTER RHO */
		0x03A2, /* Monocases to self */
		0x03C3, /* GREEK LETTER SIGMA */
		0x03C4, /* GREEK LETTER TAU */
		0x03C5, /* GREEK LETTER UPSILON */
		0x03C6, /* GREEK LETTER PHI */
		0x03C7, /* GREEK LETTER CHI */
		0x03C8, /* GREEK LETTER PSI */
		0x03C9, /* GREEK LETTER OMEGA */
		0x03CA, /* GREEK LETTER IOTA DIAERESIS */
		0x03CB, /* GREEK LETTER UPSILON DIAERESIS */
		0x03AC, /* Monocases to self */
		0x03AD, /* Monocases to self */
		0x03AE, /* Monocases to self */
		0x03AF, /* Monocases to self */
		0x03B0, /* Monocases to self */
		0x03B1, /* Monocases to self */
		0x03B2, /* Monocases to self */
		0x03B3, /* Monocases to self */
		0x03B4, /* Monocases to self */
		0x03B5, /* Monocases to self */
		0x03B6, /* Monocases to self */
		0x03B7, /* Monocases to self */
		0x03B8, /* Monocases to self */
		0x03B9, /* Monocases to self */
		0x03BA, /* Monocases to self */
		0x03BB, /* Monocases to self */
		0x03BC, /* Monocases to self */
		0x03BD, /* Monocases to self */
		0x03BE, /* Monocases to self */
		0x03BF, /* Monocases to self */
		0x03C0, /* Monocases to self */
		0x03C1, /* Monocases to self */
		0x03C2, /* Monocases to self */
		0x03C3, /* Monocases to self */
		0x03C4, /* Monocases to self */
		0x03C5, /* Monocases to self */
		0x03C6, /* Monocases to self */
		0x03C7, /* Monocases to self */
		0x03C8, /* Monocases to self */
		0x03C9, /* Monocases to self */
		0x03CA, /* Monocases to self */
		0x03CB, /* Monocases to self */
		0x03CC, /* Monocases to self */
		0x03CD, /* Monocases to self */
		0x03CE, /* Monocases to self */
		0x03CF, /* Monocases to self */
		0x03D0, /* Monocases to self */
		0x03D1, /* Monocases to self */
		0x03C5, /* GREEK LETTER UPSILON HOOK */
		0x03CD, /* GREEK LETTER UPSILON HOOK TONOS */
		0x03CB, /* GREEK LETTER UPSILON HOOK DIAERESIS */
		0x03D5, /* Monocases to self */
		0x03D6, /* Monocases to self */
		0x03D7, /* Monocases to self */
		0x03D8, /* Monocases to self */
		0x03D9, /* Monocases to self */
		0x03DB, /* GREEK LETTER STIGMA */
		0x03DB, /* Monocases to self */
		0x03DD, /* GREEK LETTER DIGAMMA */
		0x03DD, /* Monocases to self */
		0x03DF, /* GREEK LETTER KOPPA */
		0x03DF, /* Monocases to self */
		0x03E1, /* GREEK LETTER SAMPI */
		0x03E1, /* Monocases to self */
		0x03E3, /* GREEK LETTER SHEI */
		0x03E3, /* Monocases to self */
		0x03E5, /* GREEK LETTER FEI */
		0x03E5, /* Monocases to self */
		0x03E7, /* GREEK LETTER KHEI */
		0x03E7, /* Monocases to self */
		0x03E9, /* GREEK LETTER HORI */
		0x03E9, /* Monocases to self */
		0x03EB, /* GREEK LETTER GANGIA */
		0x03EB, /* Monocases to self */
		0x03ED, /* GREEK LETTER SHIMA */
		0x03ED, /* Monocases to self */
		0x03EF, /* GREEK LETTER DEI */
		0x03EF, /* Monocases to self */
		0x03F0, /* Monocases to self */
		0x03F1, /* Monocases to self */
		0x03F2, /* Monocases to self */
		0x03F3, /* Monocases to self */
		0x03F4, /* Monocases to self */
		0x03F5, /* Monocases to self */
		0x03F6, /* Monocases to self */
		0x03F7, /* Monocases to self */
		0x03F8, /* Monocases to self */
		0x03F9, /* Monocases to self */
		0x03FA, /* Monocases to self */
		0x03FB, /* Monocases to self */
		0x03FC, /* Monocases to self */
		0x03FD, /* Monocases to self */
		0x03FE, /* Monocases to self */
		0x03FF, /* Monocases to self */
		0x0400, /* Monocases to self */
		0x0451, /* CYRILLIC LETTER IO */
		0x0452, /* CYRILLIC LETTER DJE */
		0x0453, /* CYRILLIC LETTER GJE */
		0x0454, /* CYRILLIC LETTER E */
		0x0455, /* CYRILLIC LETTER DZE */
		0x0456, /* CYRILLIC LETTER I */
		0x0457, /* CYRILLIC LETTER YI */
		0x0458, /* CYRILLIC LETTER JE */
		0x0459, /* CYRILLIC LETTER LJE */
		0x045A, /* CYRILLIC LETTER NJE */
		0x045B, /* CYRILLIC LETTER TSHE */
		0x045C, /* CYRILLIC LETTER KJE */
		0x040D, /* Monocases to self */
		0x045E, /* CYRILLIC LETTER SHORT U */
		0x045F, /* CYRILLIC LETTER DZHE */
		0x0430, /* CYRILLIC LETTER A */
		0x0431, /* CYRILLIC LETTER BE */
		0x0432, /* CYRILLIC LETTER VE */
		0x0433, /* CYRILLIC LETTER GE */
		0x0434, /* CYRILLIC LETTER DE */
		0x0435, /* CYRILLIC LETTER IE */
		0x0436, /* CYRILLIC LETTER ZHE */
		0x0437, /* CYRILLIC LETTER ZE */
		0x0438, /* CYRILLIC LETTER II */
		0x0439, /* CYRILLIC LETTER SHORT II */
		0x043A, /* CYRILLIC LETTER KA */
		0x043B, /* CYRILLIC LETTER EL */
		0x043C, /* CYRILLIC LETTER EM */
		0x043D, /* CYRILLIC LETTER EN */
		0x043E, /* CYRILLIC LETTER O */
		0x043F, /* CYRILLIC LETTER PE */
		0x0440, /* CYRILLIC LETTER ER */
		0x0441, /* CYRILLIC LETTER ES */
		0x0442, /* CYRILLIC LETTER TE */
		0x0443, /* CYRILLIC LETTER U */
		0x0444, /* CYRILLIC LETTER EF */
		0x0445, /* CYRILLIC LETTER KHA */
		0x0446, /* CYRILLIC LETTER TSE */
		0x0447, /* CYRILLIC LETTER CHE */
		0x0448, /* CYRILLIC LETTER SHA */
		0x0449, /* CYRILLIC LETTER SHCHA */
		0x044A, /* CYRILLIC LETTER HARD SIGN */
		0x044B, /* CYRILLIC LETTER YERI */
		0x044C, /* CYRILLIC LETTER SOFT SIGN */
		0x044D, /* CYRILLIC LETTER REVERSED E */
		0x044E, /* CYRILLIC LETTER IU */
		0x044F, /* CYRILLIC LETTER IA */
		0x0430, /* Monocases to self */
		0x0431, /* Monocases to self */
		0x0432, /* Monocases to self */
		0x0433, /* Monocases to self */
		0x0434, /* Monocases to self */
		0x0435, /* Monocases to self */
		0x0436, /* Monocases to self */
		0x0437, /* Monocases to self */
		0x0438, /* Monocases to self */
		0x0439, /* Monocases to self */
		0x043A, /* Monocases to self */
		0x043B, /* Monocases to self */
		0x043C, /* Monocases to self */
		0x043D, /* Monocases to self */
		0x043E, /* Monocases to self */
		0x043F, /* Monocases to self */
		0x0440, /* Monocases to self */
		0x0441, /* Monocases to self */
		0x0442, /* Monocases to self */
		0x0443, /* Monocases to self */
		0x0444, /* Monocases to self */
		0x0445, /* Monocases to self */
		0x0446, /* Monocases to self */
		0x0447, /* Monocases to self */
		0x0448, /* Monocases to self */
		0x0449, /* Monocases to self */
		0x044A, /* Monocases to self */
		0x044B, /* Monocases to self */
		0x044C, /* Monocases to self */
		0x044D, /* Monocases to self */
		0x044E, /* Monocases to self */
		0x044F, /* Monocases to self */
		0x0450, /* Monocases to self */
		0x0451, /* Monocases to self */
		0x0452, /* Monocases to self */
		0x0453, /* Monocases to self */
		0x0454, /* Monocases to self */
		0x0455, /* Monocases to self */
		0x0456, /* Monocases to self */
		0x0457, /* Monocases to self */
		0x0458, /* Monocases to self */
		0x0459, /* Monocases to self */
		0x045A, /* Monocases to self */
		0x045B, /* Monocases to self */
		0x045C, /* Monocases to self */
		0x045D, /* Monocases to self */
		0x045E, /* Monocases to self */
		0x045F, /* Monocases to self */
		0x0461, /* CYRILLIC LETTER OMEGA */
		0x0461, /* Monocases to self */
		0x0463, /* CYRILLIC LETTER YAT */
		0x0463, /* Monocases to self */
		0x0465, /* CYRILLIC LETTER IOTIFIED E */
		0x0465, /* Monocases to self */
		0x0467, /* CYRILLIC LETTER LITTLE YUS */
		0x0467, /* Monocases to self */
		0x0469, /* CYRILLIC LETTER IOTIFIED LITTLE YUS */
		0x0469, /* Monocases to self */
		0x046B, /* CYRILLIC LETTER BIG YUS */
		0x046B, /* Monocases to self */
		0x046D, /* CYRILLIC LETTER IOTIFIED BIG YUS */
		0x046D, /* Monocases to self */
		0x046F, /* CYRILLIC LETTER KSI */
		0x046F, /* Monocases to self */
		0x0471, /* CYRILLIC LETTER PSI */
		0x0471, /* Monocases to self */
		0x0473, /* CYRILLIC LETTER FITA */
		0x0473, /* Monocases to self */
		0x0475, /* CYRILLIC LETTER IZHITSA */
		0x0475, /* Monocases to self */
		0x0477, /* CYRILLIC LETTER IZHITSA DOUBLE GRAVE */
		0x0477, /* Monocases to self */
		0x0479, /* CYRILLIC LETTER UK DIGRAPH */
		0x0479, /* Monocases to self */
		0x047B, /* CYRILLIC LETTER ROUND OMEGA */
		0x047B, /* Monocases to self */
		0x047D, /* CYRILLIC LETTER OMEGA TITLO */
		0x047D, /* Monocases to self */
		0x047F, /* CYRILLIC LETTER OT */
		0x047F, /* Monocases to self */
		0x0481, /* CYRILLIC LETTER KOPPA */
		0x0481, /* Monocases to self */
		0x0482, /* Monocases to self */
		0x0483, /* Monocases to self */
		0x0484, /* Monocases to self */
		0x0485, /* Monocases to self */
		0x0486, /* Monocases to self */
		0x0487, /* Monocases to self */
		0x0488, /* Monocases to self */
		0x0489, /* Monocases to self */
		0x048A, /* Monocases to self */
		0x048B, /* Monocases to self */
		0x048C, /* Monocases to self */
		0x048D, /* Monocases to self */
		0x048E, /* Monocases to self */
		0x048F, /* Monocases to self */
		0x0491, /* CYRILLIC LETTER GE WITH UPTURN */
		0x0491, /* Monocases to self */
		0x0493, /* CYRILLIC LETTER GE BAR */
		0x0493, /* Monocases to self */
		0x0495, /* CYRILLIC LETTER GE HOOK */
		0x0495, /* Monocases to self */
		0x0497, /* CYRILLIC LETTER ZHE WITH RIGHT DESCENDER */
		0x0497, /* Monocases to self */
		0x0499, /* CYRILLIC LETTER ZE CEDILLA */
		0x0499, /* Monocases to self */
		0x049B, /* CYRILLIC LETTER KA WITH RIGHT DESCENDER */
		0x049B, /* Monocases to self */
		0x049D, /* CYRILLIC LETTER KA VERTICAL BAR */
		0x049D, /* Monocases to self */
		0x049F, /* CYRILLIC LETTER KA BAR */
		0x049F, /* Monocases to self */
		0x04A1, /* CYRILLIC LETTER REVERSED GE KA */
		0x04A1, /* Monocases to self */
		0x04A3, /* CYRILLIC LETTER EN WITH RIGHT DESCENDER */
		0x04A3, /* Monocases to self */
		0x04A5, /* CYRILLIC LETTER EN GE */
		0x04A5, /* Monocases to self */
		0x04A7, /* CYRILLIC LETTER PE HOOK */
		0x04A7, /* Monocases to self */
		0x04A9, /* CYRILLIC LETTER O HOOK */
		0x04A9, /* Monocases to self */
		0x04AB, /* CYRILLIC LETTER ES CEDILLA */
		0x04AB, /* Monocases to self */
		0x04AD, /* CYRILLIC LETTER TE WITH RIGHT DESCENDER */
		0x04AD, /* Monocases to self */
		0x04AF, /* CYRILLIC LETTER STRAIGHT U */
		0x04AF, /* Monocases to self */
		0x04B1, /* CYRILLIC LETTER STRAIGHT U BAR */
		0x04B1, /* Monocases to self */
		0x04B3, /* CYRILLIC LETTER KHA WITH RIGHT DESCENDER */
		0x04B3, /* Monocases to self */
		0x04B5, /* CYRILLIC LETTER TE TSE */
		0x04B5, /* Monocases to self */
		0x04B7, /* CYRILLIC LETTER CHE WITH RIGHT DESCENDER */
		0x04B7, /* Monocases to self */
		0x04B9, /* CYRILLIC LETTER CHE VERTICAL BAR */
		0x04B9, /* Monocases to self */
		0x04BB, /* CYRILLIC LETTER H */
		0x04BB, /* Monocases to self */
		0x04BD, /* CYRILLIC LETTER IE HOOK */
		0x04BD, /* Monocases to self */
		0x04BF, /* CYRILLIC LETTER IE HOOK OGONEK */
		0x04BF, /* Monocases to self */
		0x04C0, /* Monocases to self */
		0x04C2, /* CYRILLIC LETTER SHORT ZHE */
		0x04C2, /* Monocases to self */
		0x04C4, /* CYRILLIC LETTER KA HOOK */
		0x04C4, /* Monocases to self */
		0x04C6, /* CYRILLIC LETTER KA OGONEK */
		0x04C6, /* Monocases to self */
		0x04C8, /* CYRILLIC LETTER EN HOOK */
		0x04C8, /* Monocases to self */
		0x04CA, /* CYRILLIC LETTER KHA OGONEK */
		0x04CA, /* Monocases to self */
		0x04CC, /* CYRILLIC LETTER CHE WITH LEFT DESCENDER */
		0x04CC, /* Monocases to self */
		0x04CD, /* Monocases to self */
		0x04CE, /* Monocases to self */
		0x04CF, /* Monocases to self */
		0x04D0, /* Monocases to self */
		0x04D1, /* Monocases to self */
		0x04D2, /* Monocases to self */
		0x04D3, /* Monocases to self */
		0x04D4, /* Monocases to self */
		0x04D5, /* Monocases to self */
		0x04D6, /* Monocases to self */
		0x04D7, /* Monocases to self */
		0x04D8, /* Monocases to self */
		0x04D9, /* Monocases to self */
		0x04DA, /* Monocases to self */
		0x04DB, /* Monocases to self */
		0x04DC, /* Monocases to self */
		0x04DD, /* Monocases to self */
		0x04DE, /* Monocases to self */
		0x04DF, /* Monocases to self */
		0x04E0, /* Monocases to self */
		0x04E1, /* Monocases to self */
		0x04E2, /* Monocases to self */
		0x04E3, /* Monocases to self */
		0x04E4, /* Monocases to self */
		0x04E5, /* Monocases to self */
		0x04E6, /* Monocases to self */
		0x04E7, /* Monocases to self */
		0x04E8, /* Monocases to self */
		0x04E9, /* Monocases to self */
		0x04EA, /* Monocases to self */
		0x04EB, /* Monocases to self */
		0x04EC, /* Monocases to self */
		0x04ED, /* Monocases to self */
		0x04EE, /* Monocases to self */
		0x04EF, /* Monocases to self */
		0x04F0, /* Monocases to self */
		0x04F1, /* Monocases to self */
		0x04F2, /* Monocases to self */
		0x04F3, /* Monocases to self */
		0x04F4, /* Monocases to self */
		0x04F5, /* Monocases to self */
		0x04F6, /* Monocases to self */
		0x04F7, /* Monocases to self */
		0x04F8, /* Monocases to self */
		0x04F9, /* Monocases to self */
		0x04FA, /* Monocases to self */
		0x04FB, /* Monocases to self */
		0x04FC, /* Monocases to self */
		0x04FD, /* Monocases to self */
		0x04FE, /* Monocases to self */
		0x04FF, /* Monocases to self */
		0x0500, /* Monocases to self */
		0x0501, /* Monocases to self */
		0x0502, /* Monocases to self */
		0x0503, /* Monocases to self */
		0x0504, /* Monocases to self */
		0x0505, /* Monocases to self */
		0x0506, /* Monocases to self */
		0x0507, /* Monocases to self */
		0x0508, /* Monocases to self */
		0x0509, /* Monocases to self */
		0x050A, /* Monocases to self */
		0x050B, /* Monocases to self */
		0x050C, /* Monocases to self */
		0x050D, /* Monocases to self */
		0x050E, /* Monocases to self */
		0x050F, /* Monocases to self */
		0x0510, /* Monocases to self */
		0x0511, /* Monocases to self */
		0x0512, /* Monocases to self */
		0x0513, /* Monocases to self */
		0x0514, /* Monocases to self */
		0x0515, /* Monocases to self */
		0x0516, /* Monocases to self */
		0x0517, /* Monocases to self */
		0x0518, /* Monocases to self */
		0x0519, /* Monocases to self */
		0x051A, /* Monocases to self */
		0x051B, /* Monocases to self */
		0x051C, /* Monocases to self */
		0x051D, /* Monocases to self */
		0x051E, /* Monocases to self */
		0x051F, /* Monocases to self */
		0x0520, /* Monocases to self */
		0x0521, /* Monocases to self */
		0x0522, /* Monocases to self */
		0x0523, /* Monocases to self */
		0x0524, /* Monocases to self */
		0x0525, /* Monocases to self */
		0x0526, /* Monocases to self */
		0x0527, /* Monocases to self */
		0x0528, /* Monocases to self */
		0x0529, /* Monocases to self */
		0x052A, /* Monocases to self */
		0x052B, /* Monocases to self */
		0x052C, /* Monocases to self */
		0x052D, /* Monocases to self */
		0x052E, /* Monocases to self */
		0x052F, /* Monocases to self */
		0x0530, /* Monocases to self */
		0x0561, /* ARMENIAN LETTER AYB */
		0x0562, /* ARMENIAN LETTER BEN */
		0x0563, /* ARMENIAN LETTER GIM */
		0x0564, /* ARMENIAN LETTER DA */
		0x0565, /* ARMENIAN LETTER ECH */
		0x0566, /* ARMENIAN LETTER ZA */
		0x0567, /* ARMENIAN LETTER EH */
		0x0568, /* ARMENIAN LETTER ET */
		0x0569, /* ARMENIAN LETTER TO */
		0x056A, /* ARMENIAN LETTER ZHE */
		0x056B, /* ARMENIAN LETTER INI */
		0x056C, /* ARMENIAN LETTER LIWN */
		0x056D, /* ARMENIAN LETTER XEH */
		0x056E, /* ARMENIAN LETTER CA */
		0x056F, /* ARMENIAN LETTER KEN */
		0x0570, /* ARMENIAN LETTER HO */
		0x0571, /* ARMENIAN LETTER JA */
		0x0572, /* ARMENIAN LETTER LAD */
		0x0573, /* ARMENIAN LETTER CHEH */
		0x0574, /* ARMENIAN LETTER MEN */
		0x0575, /* ARMENIAN LETTER YI */
		0x0576, /* ARMENIAN LETTER NOW */
		0x0577, /* ARMENIAN LETTER SHA */
		0x0578, /* ARMENIAN LETTER VO */
		0x0579, /* ARMENIAN LETTER CHA */
		0x057A, /* ARMENIAN LETTER PEH */
		0x057B, /* ARMENIAN LETTER JHEH */
		0x057C, /* ARMENIAN LETTER RA */
		0x057D, /* ARMENIAN LETTER SEH */
		0x057E, /* ARMENIAN LETTER VEW */
		0x057F, /* ARMENIAN LETTER TIWN */
		0x0580, /* ARMENIAN LETTER REH */
		0x0581, /* ARMENIAN LETTER CO */
		0x0582, /* ARMENIAN LETTER YIWN */
		0x0583, /* ARMENIAN LETTER PIWR */
		0x0584, /* ARMENIAN LETTER KEH */
		0x0585, /* ARMENIAN LETTER OH */
		0x0586, /* ARMENIAN LETTER FEH */
		0x0557, /* Monocases to self */
		0x0558, /* Monocases to self */
		0x0559, /* Monocases to self */
		0x055A, /* Monocases to self */
		0x055B, /* Monocases to self */
		0x055C, /* Monocases to self */
		0x055D, /* Monocases to self */
		0x055E, /* Monocases to self */
		0x055F, /* Monocases to self */
		0x0560, /* Monocases to self */
		0x0561, /* Monocases to self */
		0x0562, /* Monocases to self */
		0x0563, /* Monocases to self */
		0x0564, /* Monocases to self */
		0x0565, /* Monocases to self */
		0x0566, /* Monocases to self */
		0x0567, /* Monocases to self */
		0x0568, /* Monocases to self */
		0x0569, /* Monocases to self */
		0x056A, /* Monocases to self */
		0x056B, /* Monocases to self */
		0x056C, /* Monocases to self */
		0x056D, /* Monocases to self */
		0x056E, /* Monocases to self */
		0x056F, /* Monocases to self */
		0x0570, /* Monocases to self */
		0x0571, /* Monocases to self */
		0x0572, /* Monocases to self */
		0x0573, /* Monocases to self */
		0x0574, /* Monocases to self */
		0x0575, /* Monocases to self */
		0x0576, /* Monocases to self */
		0x0577, /* Monocases to self */
		0x0578, /* Monocases to self */
		0x0579, /* Monocases to self */
		0x057A, /* Monocases to self */
		0x057B, /* Monocases to self */
		0x057C, /* Monocases to self */
		0x057D, /* Monocases to self */
		0x057E, /* Monocases to self */
		0x057F, /* Monocases to self */
		0x0580, /* Monocases to self */
		0x0581, /* Monocases to self */
		0x0582, /* Monocases to self */
		0x0583, /* Monocases to self */
		0x0584, /* Monocases to self */
		0x0585, /* Monocases to self */
		0x0586, /* Monocases to self */
		0x0587, /* Monocases to self */
		0x0588, /* Monocases to self */
		0x0589, /* Monocases to self */
		0x058A, /* Monocases to self */
		0x058B, /* Monocases to self */
		0x058C, /* Monocases to self */
		0x058D, /* Monocases to self */
		0x058E, /* Monocases to self */
		0x058F, /* Monocases to self */
		0x0590, /* Monocases to self */
		0x0591, /* Monocases to self */
		0x0592, /* Monocases to self */
		0x0593, /* Monocases to self */
		0x0594, /* Monocases to self */
		0x0595, /* Monocases to self */
		0x0596, /* Monocases to self */
		0x0597, /* Monocases to self */
		0x0598, /* Monocases to self */
		0x0599, /* Monocases to self */
		0x059A, /* Monocases to self */
		0x059B, /* Monocases to self */
		0x059C, /* Monocases to self */
		0x059D, /* Monocases to self */
		0x059E, /* Monocases to self */
		0x059F, /* Monocases to self */
		0x05A0, /* Monocases to self */
		0x05A1, /* Monocases to self */
		0x05A2, /* Monocases to self */
		0x05A3, /* Monocases to self */
		0x05A4, /* Monocases to self */
		0x05A5, /* Monocases to self */
		0x05A6, /* Monocases to self */
		0x05A7, /* Monocases to self */
		0x05A8, /* Monocases to self */
		0x05A9, /* Monocases to self */
		0x05AA, /* Monocases to self */
		0x05AB, /* Monocases to self */
		0x05AC, /* Monocases to self */
		0x05AD, /* Monocases to self */
		0x05AE, /* Monocases to self */
		0x05AF, /* Monocases to self */
		0x05B0, /* Monocases to self */
		0x05B1, /* Monocases to self */
		0x05B2, /* Monocases to self */
		0x05B3, /* Monocases to self */
		0x05B4, /* Monocases to self */
		0x05B5, /* Monocases to self */
		0x05B6, /* Monocases to self */
		0x05B7, /* Monocases to self */
		0x05B8, /* Monocases to self */
		0x05B9, /* Monocases to self */
		0x05BA, /* Monocases to self */
		0x05BB, /* Monocases to self */
		0x05BC, /* Monocases to self */
		0x05BD, /* Monocases to self */
		0x05BE, /* Monocases to self */
		0x05BF, /* Monocases to self */
		0x05C0, /* Monocases to self */
		0x05C1, /* Monocases to self */
		0x05C2, /* Monocases to self */
		0x05C3, /* Monocases to self */
		0x05C4, /* Monocases to self */
		0x05C5, /* Monocases to self */
		0x05C6, /* Monocases to self */
		0x05C7, /* Monocases to self */
		0x05C8, /* Monocases to self */
		0x05C9, /* Monocases to self */
		0x05CA, /* Monocases to self */
		0x05CB, /* Monocases to self */
		0x05CC, /* Monocases to self */
		0x05CD, /* Monocases to self */
		0x05CE, /* Monocases to self */
		0x05CF, /* Monocases to self */
		0x05D0, /* Monocases to self */
		0x05D1, /* Monocases to self */
		0x05D2, /* Monocases to self */
		0x05D3, /* Monocases to self */
		0x05D4, /* Monocases to self */
		0x05D5, /* Monocases to self */
		0x05D6, /* Monocases to self */
		0x05D7, /* Monocases to self */
		0x05D8, /* Monocases to self */
		0x05D9, /* Monocases to self */
		0x05DA, /* Monocases to self */
		0x05DB, /* Monocases to self */
		0x05DC, /* Monocases to self */
		0x05DD, /* Monocases to self */
		0x05DE, /* Monocases to self */
		0x05DF, /* Monocases to self */
		0x05E0, /* Monocases to self */
		0x05E1, /* Monocases to self */
		0x05E2, /* Monocases to self */
		0x05E3, /* Monocases to self */
		0x05E4, /* Monocases to self */
		0x05E5, /* Monocases to self */
		0x05E6, /* Monocases to self */
		0x05E7, /* Monocases to self */
		0x05E8, /* Monocases to self */
		0x05E9, /* Monocases to self */
		0x05EA, /* Monocases to self */
		0x05EB, /* Monocases to self */
		0x05EC, /* Monocases to self */
		0x05ED, /* Monocases to self */
		0x05EE, /* Monocases to self */
		0x05EF, /* Monocases to self */
		0x05F0, /* Monocases to self */
		0x05F1, /* Monocases to self */
		0x05F2, /* Monocases to self */
		0x05F3, /* Monocases to self */
		0x05F4, /* Monocases to self */
		0x05F5, /* Monocases to self */
		0x05F6, /* Monocases to self */
		0x05F7, /* Monocases to self */
		0x05F8, /* Monocases to self */
		0x05F9, /* Monocases to self */
		0x05FA, /* Monocases to self */
		0x05FB, /* Monocases to self */
		0x05FC, /* Monocases to self */
		0x05FD, /* Monocases to self */
		0x05FE, /* Monocases to self */
		0x05FF, /* Monocases to self */
	};

	static const FLMUNICODE georgian[ 40] =
	{
		0x10D0, /* GEORGIAN LETTER AN */
		0x10D1, /* GEORGIAN LETTER BAN */
		0x10D2, /* GEORGIAN LETTER GAN */
		0x10D3, /* GEORGIAN LETTER DON */
		0x10D4, /* GEORGIAN LETTER EN */
		0x10D5, /* GEORGIAN LETTER VIN */
		0x10D6, /* GEORGIAN LETTER ZEN */
		0x10D7, /* GEORGIAN LETTER TAN */
		0x10D8, /* GEORGIAN LETTER IN */
		0x10D9, /* GEORGIAN LETTER KAN */
		0x10DA, /* GEORGIAN LETTER LAS */
		0x10DB, /* GEORGIAN LETTER MAN */
		0x10DC, /* GEORGIAN LETTER NAR */
		0x10DD, /* GEORGIAN LETTER ON */
		0x10DE, /* GEORGIAN LETTER PAR */
		0x10DF, /* GEORGIAN LETTER ZHAR */
		0x10E0, /* GEORGIAN LETTER RAE */
		0x10E1, /* GEORGIAN LETTER SAN */
		0x10E2, /* GEORGIAN LETTER TAR */
		0x10E3, /* GEORGIAN LETTER UN */
		0x10E4, /* GEORGIAN LETTER PHAR */
		0x10E5, /* GEORGIAN LETTER KHAR */
		0x10E6, /* GEORGIAN LETTER GHAN */
		0x10E7, /* GEORGIAN LETTER QAR */
		0x10E8, /* GEORGIAN LETTER SHIN */
		0x10E9, /* GEORGIAN LETTER CHIN */
		0x10EA, /* GEORGIAN LETTER CAN */
		0x10EB, /* GEORGIAN LETTER JIL */
		0x10EC, /* GEORGIAN LETTER CIL */
		0x10ED, /* GEORGIAN LETTER CHAR */
		0x10EE, /* GEORGIAN LETTER XAN */
		0x10EF, /* GEORGIAN LETTER JHAN */
		0x10F0, /* GEORGIAN LETTER HAE */
		0x10F1, /* GEORGIAN LETTER HE */
		0x10F2, /* GEORGIAN LETTER HIE */
		0x10F3, /* GEORGIAN LETTER WE */
		0x10F4, /* GEORGIAN LETTER HAR */
		0x10F5, /* GEORGIAN LETTER HOE */
	};

	static const FLMUNICODE circledLatin[26] =
	{
		0x24D0, /* CIRCLED LATIN LETTER A */
		0x24D1, /* CIRCLED LATIN LETTER B */
		0x24D2, /* CIRCLED LATIN LETTER C */
		0x24D3, /* CIRCLED LATIN LETTER D */
		0x24D4, /* CIRCLED LATIN LETTER E */
		0x24D5, /* CIRCLED LATIN LETTER F */
		0x24D6, /* CIRCLED LATIN LETTER G */
		0x24D7, /* CIRCLED LATIN LETTER H */
		0x24D8, /* CIRCLED LATIN LETTER I */
		0x24D9, /* CIRCLED LATIN LETTER J */
		0x24DA, /* CIRCLED LATIN LETTER K */
		0x24DB, /* CIRCLED LATIN LETTER L */
		0x24DC, /* CIRCLED LATIN LETTER M */
		0x24DD, /* CIRCLED LATIN LETTER N */
		0x24DE, /* CIRCLED LATIN LETTER O */
		0x24DF, /* CIRCLED LATIN LETTER P */
		0x24E0, /* CIRCLED LATIN LETTER Q */
		0x24E1, /* CIRCLED LATIN LETTER R */
		0x24E2, /* CIRCLED LATIN LETTER S */
		0x24E3, /* CIRCLED LATIN LETTER T */
		0x24E4, /* CIRCLED LATIN LETTER U */
		0x24E5, /* CIRCLED LATIN LETTER V */
		0x24E6, /* CIRCLED LATIN LETTER W */
		0x24E7, /* CIRCLED LATIN LETTER X */
		0x24E8, /* CIRCLED LATIN LETTER Y */
		0x24E9, /* CIRCLED LATIN LETTER Z */
	};

	static const FLMUNICODE compat[] =
	{
		0x2025,
		0x2014,
		0x2013,
		0x005F,
		0x005F,
		0x0028,
		0x0029,
		0x007B,
		0x007D,
		0x3014,
		0x3015,
		0x3010,
		0x3011,
		0x300A,
		0x300B,
		0x3008,
		0x3009,
		0x300C,
		0x300D,
		0x300E,
		0x300F,
		0xFE45,
		0xFE46,
		0xFE47,
		0xFE48,
		0x203E,
		0x203E,
		0x203E,
		0x203E,
		0x005F,
		0x005F,
		0x005F,
		0x002C,
		0x3001,
		0x002E,
		0xFE53,
		0x003B,
		0x003A,
		0x003F,
		0x0021,
		0x2014,
		0x0028,
		0x0029,
		0x007B,
		0x007D,
		0x3014,
		0x3015,
		0x0023,
		0x0026,
		0x002A,
		0x002B,
		0x002D,
		0x003C,
		0x003E,
		0x003D,
		0xFE67,
		0x005C,
		0x0024,
		0x0025,
		0x0040,
		0xFE6C,
		0xFE6D,
		0xFE6E,
		0xFE6F,
		0x064B,
		0x064B,
		0x064C,
		0xFE73,
		0x064D,
		0xFE75,
		0x064E,
		0x064E,
		0x064F,
		0x064F,
		0x0650,
		0x0650,
		0x0651,
		0x0651,
		0x0652,
		0x0652,
		0x0621,
		0x0622,
		0x0622,
		0x0623,
		0x0623,
		0x0624,
		0x0624,
		0x0625,
		0x0625,
		0x0626,
		0x0626,
		0x0626,
		0x0626,
		0x0627,
		0x0627,
		0x0628,
		0x0628,
		0x0628,
		0x0628,
		0x0629,
		0x0629,
		0x062A,
		0x062A,
		0x062A,
		0x062A,
		0x062B,
		0x062B,
		0x062B,
		0x062B,
		0x062C,
		0x062C,
		0x062C,
		0x062C,
		0x062D,
		0x062D,
		0x062D,
		0x062D,
		0x062E,
		0x062E,
		0x062E,
		0x062E,
		0x062F,
		0x062F,
		0x0630,
		0x0630,
		0x0631,
		0x0631,
		0x0632,
		0x0632,
		0x0633,
		0x0633,
		0x0633,
		0x0633,
		0x0634,
		0x0634,
		0x0634,
		0x0634,
		0x0635,
		0x0635,
		0x0635,
		0x0635,
		0x0636,
		0x0636,
		0x0636,
		0x0636,
		0x0637,
		0x0637,
		0x0637,
		0x0637,
		0x0638,
		0x0638,
		0x0638,
		0x0638,
		0x0639,
		0x0639,
		0x0639,
		0x0639,
		0x063A,
		0x063A,
		0x063A,
		0x063A,
		0x0641,
		0x0641,
		0x0641,
		0x0641,
		0x0642,
		0x0642,
		0x0642,
		0x0642,
		0x0643,
		0x0643,
		0x0643,
		0x0643,
		0x0644,
		0x0644,
		0x0644,
		0x0644,
		0x0645,
		0x0645,
		0x0645,
		0x0645,
		0x0646,
		0x0646,
		0x0646,
		0x0646,
		0x0647,
		0x0647,
		0x0647,
		0x0647,
		0x0648,
		0x0648,
		0x0649,
		0x0649,
		0x064A,
		0x064A,
		0x064A,
		0x064A,
		0xFEF5,
		0xFEF6,
		0xFEF7,
		0xFEF8,
		0xFEF9,
		0xFEFA,
		0xFEFB,
		0xFEFC,
		0xFEFD,
		0xFEFE,
		0xFEFE,
		0xFE00,
		0x0021,
		0x0022,
		0x0023,
		0x0024,
		0x0025,
		0x0026,
		0x0027,
		0x0028,
		0x0029,
		0x002A,
		0x002B,
		0x002C,
		0x002D,
		0x002E,
		0x002F,
		0x0030,
		0x0031,
		0x0032,
		0x0033,
		0x0034,
		0x0035,
		0x0036,
		0x0037,
		0x0038,
		0x0039,
		0x003A,
		0x003B,
		0x003C,
		0x003D,
		0x003E,
		0x003F,
		0x0040,
		0x0061,
		0x0062,
		0x0063,
		0x0064,
		0x0065,
		0x0066,
		0x0067,
		0x0068,
		0x0069,
		0x006A,
		0x006B,
		0x006C,
		0x006D,
		0x006E,
		0x006F,
		0x0070,
		0x0071,
		0x0072,
		0x0073,
		0x0074,
		0x0075,
		0x0076,
		0x0077,
		0x0078,
		0x0079,
		0x007A,
		0x005B,
		0x005C,
		0x005D,
		0x005E,
		0x005F,
		0x0060,
		0x0061,
		0x0062,
		0x0063,
		0x0064,
		0x0065,
		0x0066,
		0x0067,
		0x0068,
		0x0069,
		0x006A,
		0x006B,
		0x006C,
		0x006D,
		0x006E,
		0x006F,
		0x0070,
		0x0071,
		0x0072,
		0x0073,
		0x0074,
		0x0075,
		0x0076,
		0x0077,
		0x0078,
		0x0079,
		0x007A,
		0x007B,
		0x007C,
		0x007D,
		0x007E,
		0xFF5F,
		0xFF60,
		0x3002,
		0x300C,
		0x300D,
		0x3001,
		0x30FB,
		0x30F2,
		0x30A1,
		0x30A3,
		0x30A5,
		0x30A7,
		0x30A9,
		0x30E3,
		0x30E5,
		0x30E7,
		0x30C3,
		0x30FC,
		0x30A2,
		0x30A4,
		0x30A6,
		0x30A8,
		0x30AA,
		0x30AB,
		0x30AD,
		0x30AF,
		0x30B1,
		0x30B3,
		0x30B5,
		0x30B7,
		0x30B9,
		0x30BB,
		0x30BD,
		0x30BF,
		0x30C1,
		0x30C4,
		0x30C6,
		0x30C8,
		0x30CA,
		0x30CB,
		0x30CC,
		0x30CD,
		0x30CE,
		0x30CF,
		0x30D2,
		0x30D5,
		0x30D8,
		0x30DB,
		0x30DE,
		0x30DF,
		0x30E0,
		0x30E1,
		0x30E2,
		0x30E4,
		0x30E6,
		0x30E8,
		0x30E9,
		0x30EA,
		0x30EB,
		0x30EC,
		0x30ED,
		0x30EF,
		0x30F3,
		0x309B,
		0x309C,
		0x3164,
		0x3131,
		0x3132,
		0x3133,
		0x3134,
		0x3135,
		0x3136,
		0x3137,
		0x3138,
		0x3139,
		0x313A,
		0x313B,
		0x313C,
		0x313D,
		0x313E,
		0x313F,
		0x3140,
		0x3141,
		0x3142,
		0x3143,
		0x3144,
		0x3145,
		0x3146,
		0x3147,
		0x3148,
		0x3149,
		0x314A,
		0x314B,
		0x314C,
		0x314D,
		0x314E,
		0xFFBF,
		0xFFC0,
		0xFFC1,
		0x314F,
		0x3150,
		0x3151,
		0x3152,
		0x3153,
		0x3154,
		0xFFC8,
		0xFFC9,
		0x3155,
		0x3156,
		0x3157,
		0x3158,
		0x3159,
		0x315A,
		0xFFD0,
		0xFFD1,
		0x315B,
		0x315C,
		0x315D,
		0x315E,
		0x315F,
		0x3160,
		0xFFD8,
		0xFFD9,
		0x3161,
		0x3162,
		0x3163,
		0xFFDD,
		0xFFDE,
		0xFFDF,
		0x00A2,
		0x00A3,
		0x00AC,
		0x00AF,
		0x00A6,
		0x00A5,
		0x20A9
	};

	if( uChar < 0x600)
	{
		uChar = basicAlpha[ uChar];
	}
	else if( uChar < 0x10A0)
	{
		;
	}
	else if( uChar >= 0x10A0 && uChar <= 0x10C5)
	{
		uChar = georgian[ uChar - 0x10A0];
	}
	else if( uChar >= 0x24B6 && uChar <= 0x24CF)
	{
		uChar = circledLatin[ uChar - 0x24B6];
	}
	else if( uChar >= 0xFE30 && uChar <= 0xFFE6)
	{
		uChar = compat[ uChar - 0xFE30];
	}

	return( uChar);
}

/****************************************************************************
Desc: Compares two Unicode strings
****************************************************************************/
FLMINT f_unicmp(
	FLMUNICODE *	puzStr1,
	FLMUNICODE *	puzStr2)
{
	while( *puzStr1 == *puzStr2 && *puzStr1)
	{
		puzStr1++;
		puzStr2++;
	}

	return( (FLMINT)*puzStr1 - (FLMINT)*puzStr2);
}

/****************************************************************************
Desc: Performs a case-insensitive comparision of two Unicode strings
****************************************************************************/
FLMINT f_uniicmp(
	FLMUNICODE *	puzStr1,
	FLMUNICODE *	puzStr2)
{
	while( f_unitolower( *puzStr1) == f_unitolower( *puzStr2) && *puzStr1)
	{
		puzStr1++;
		puzStr2++;
	}

	return( (FLMINT)f_unitolower( *puzStr1) - (FLMINT)f_unitolower( *puzStr2));
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

/***************************************************************************
Desc:  Functions to test various properties of a unicode character
****************************************************************************/

#define UNICODE_DECIMAL_DIGIT_MASK		0x08
#define UNICODE_ALPHABETIC_MASK			0x04
#define UNICODE_UPPERCASE_MASK			0x02
#define UNICODE_LOWERCASE_MASK			0x01

const unsigned char UnicodeProperties[ 32768] = {
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,
   6, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,  96,   0,   0,   5,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  80,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  80,   0,   0,   0,   0,   5,   0,   0,  80,   0,   0,
 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,  96, 102, 102, 102, 101,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  80,  85,  85,  85,  85,
 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,  86,  86,  86,  86,
  86,  86,  86,  86,  85, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102,  86,  86,  85,
  86, 101, 101, 102,  86, 102,  85, 102, 102,  86, 101, 102, 101,  85, 102,  86, 101, 101, 101, 102,  86,  85, 101, 102,  86, 102,  86,  86, 101,  84, 101,  85,
  68,  68, 100,  86,  69, 100,  86,  86,  86,  86,  86,  86,  86,  86,  85, 101, 101, 101, 101, 101, 101, 101, 101, 101,  86,  69, 101, 102, 101, 101, 101, 101,
 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,  85,  80,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,
  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  84,  68,  68,  68,
  85,   0,   0,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,  85,  85,  80,   0,   0,   0,   0,  64,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  80,   0,   0,
   0,   0,   0,  96, 102,  96,  96, 102,  86, 102, 102, 102, 102, 102, 102, 102, 102,   6, 102, 102, 102, 102,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,
  85,  85,  85,  85,  85,  85,  85,  80,  85, 102, 101,  85, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,  85,  85, 101,   6,  86, 101,   0,   0,
 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,  85,  85,  85,  85,  85,  85,  85,  85,
  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
 101,   0,   0,   0,   0, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
 102,  86,  86,  86,  86,  86,  86,  80, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,   0, 101,   0,   0,   0,
 101, 101, 101, 101, 101, 101, 101, 101,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   6, 102, 102, 102, 102, 102, 102, 102,
 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,  96,   4,   0,   0,   0,   5,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,
  85,  85,  85,  85,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,   4,  68,   4,
   4,  64,  64,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,  68,  64,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,   0,   0,   0,   0,   0,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0, 136, 136, 136, 136, 136,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   4,  68,  68,  68,  64,   0,   4,  68,  68,  68,  64,   0,   4,  68, 136, 136, 136, 136, 136,  68,  64,   4,
   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
   0,   0,   0,   0,   0,   0,   4,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   4,  68,
  68,  68,  68,  68,  68,  68,  64,   0,  64,   0,   0,   0,  68,  68,  68,  68,  68,  68,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,
   4,  68,   4,  68,  68,  68,  64,   4,  64,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  64,  64,   0,  68,  68,   0,   4,  68,
  68,  68,  64,   4,  64,   4,  64,   0,   0,   0,   0,   4,   0,   0,  68,   4,  68,  68,   0, 136, 136, 136, 136, 136,  68,   0,   0,   0,   0,   0,   0,   0,
   4,  68,   4,  68,  68,  64,   0,   4,  64,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  64,  68,   4,  64,  68,   0,   0,  68,
  68,  64,   0,   4,  64,   4,  64,   0,   0,   0,   0,   0,   4,  68,  64,  64,   0,   0,   0, 136, 136, 136, 136, 136,  68,  68,  64,   0,   0,   0,   0,   0,
   4,  68,   4,  68,  68,  68,  68,   4,  68,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  64,  68,   4,  68,  68,   0,   4,  68,
  68,  68,  68,   4,  68,   4,  64,   0,  64,   0,   0,   0,   0,   0,   0,   0,  68,  68,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,
   4,  68,   4,  68,  68,  68,  64,   4,  64,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  64,  68,   4,  68,  68,   0,   4,  68,
  68,  68,   0,   4,  64,   4,  64,   0,   0,   0,   0,  68,   0,   0,  68,   4,  68,   0,   0, 136, 136, 136, 136, 136,   4,   0,   0,   0,   0,   0,   0,   0,
   0,  68,   4,  68,  68,  64,   0,  68,  64,  68,  68,   0,   4,  64,  64,  68,   0,   4,  64,   0,  68,  64,   0,  68,  68,  68,  68,   4,  68,   0,   0,  68,
  68,  64,   0,  68,  64,  68,  64,   0,   0,   0,   0,   4,   0,   0,   0,   0,   0,   0,   0,   8, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,
   4,  68,   4,  68,  68,  68,  64,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  68,  68,   4,  68,  68,   0,   0,  68,
  68,  68,  64,  68,  64,  68,  64,   0,   0,   0,   4,  64,   0,   0,   0,   0,  68,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,
   0,  68,   4,  68,  68,  68,  64,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  68,  68,   4,  68,  68,   0,   4,  68,
  68,  68,  64,  68,  64,  68,  64,   0,   0,   0,   4,  64,   0,   0,   0,  64,  68,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,
   0,  68,   4,  68,  68,  68,  64,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,  68,
  68,  68,   0,  68,  64,  68,  64,   0,   0,   0,   0,   4,   0,   0,   0,   0,  68,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,
   0,  68,   4,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   4,  68,  68,  68,  68,   4,   0,
  68,  68,  68,  64,   0,   0,   0,   4,  68,  68,  64,  64,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68,   0,   0,   0,   0,   0,   0,
   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,
  68,  68,  68,  64,   0,   0,   4,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   4,  64,  64,   4,  64,  64,   4,   0,   0,   0,  68,  68,   4,  68,  68,  68,   4,  68,   4,   4,   0,  68,   4,  68,  68,  68,  68,  68,  68,   4,  68,   0,
  68,  68,  64,  64,   0,   0,   4,   0, 136, 136, 136, 136, 136,   0,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  64,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   4,  68,  68,  68,  68,  68,  68,  68,
  68,   0,   0,   0,  68,  68,   0,   0,  68,  68,  68,  68,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   4,  68,  68,   4,  64,  68,  68,  68,  64,   0,  64,  64,   0,   0,   0,
 136, 136, 136, 136, 136,   0,   0,   0,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,
 102, 102, 102,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,
  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  64,  64,  68,  68,   0,  68,  68,  68,  64,  64,  68,  68,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  64,  64,  68,  68,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  64,  68,  68,   0,  68,  68,  68,  64,
  64,  68,  68,   0,  68,  68,  68,  64,  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  64,  64,  68,  68,   0,  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   0,   0,   0,   0,   8, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   0,   0,   0,
   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   4,  68,  68,  68,  64,   0,   0,   0,   0,
   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,  68,  64,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  64,  68,  68,  68,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  64,  68,  64,  68,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  64,   0,   0,   0,   0,   0,   0,   4,   0,   0,  64,   0, 136, 136, 136, 136, 136,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,  68,  68,  68,  68,  68,  68,   0,   0,  68,  68,  68,  68,  64,   0,   0,   0,
   0,   0,   0, 136, 136, 136, 136, 136,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,  68,  68,  64,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,
  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,  85,  85,  85,   0,   0, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,   0,   0,   0,
  85,  85,  85,  85, 102, 102, 102, 102,  85,  85,  85,   0, 102, 102, 102,   0,  85,  85,  85,  85, 102, 102, 102, 102,  85,  85,  85,  85, 102, 102, 102, 102,
  85,  85,  85,   0, 102, 102, 102,   0,  85,  85,  85,  85,   6,   6,   6,   6,  85,  85,  85,  85, 102, 102, 102, 102,  85,  85,  85,  85,  85,  85,  85,   0,
  85,  85,  85,  85,  68,  68,  68,  68,  85,  85,  85,  85,  68,  68,  68,  68,  85,  85,  85,  85,  68,  68,  68,  68,  85,  85,  80,  85, 102, 102,  64,  80,
   0,  85,  80,  85, 102, 102,  64,   0,  85,  85,   0,  85, 102, 102,   0,   0,  85,  85,  85,  85, 102, 102,  96,   0,   0,  85,  80,  85, 102, 102,  64,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,   0,   0,   0,   0,   0,   0,   5,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,  96,   0,   6,   0,  86, 102,  85, 102, 101,   6,   0,   6, 102, 102,   0,   0,   0,  96,  96,  96, 102, 102,   5, 102,   6,  84,  68,  69,   0,   5, 102,
   0,   0,   6,  85,  85,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 102, 102, 102, 102, 102, 102, 102, 102,  85,  85,  85,  85,  85,  85,  85,  85,
  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  34,  34,  34,  34,  34,
  34,  34,  34,  34,  34,  34,  34,  34,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   4,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,  68,  68,  68,  68,   0,   0,   0,   4,  68,  68,   0,  68,  68,  64,   0,
   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   4,  68,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,  68,  68,
   0,   0,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   4,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  64,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  85,  85,  85,  80,   0,   0,   0,   0,   0,   5,  85,  85,   0,   0,   4,  68,  68,  68,  68,  68,  64,  68,  68,  68,  68,  68,  68,  64,  68,  68,  64,  64,
  68,   4,  64,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,   0,
   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68,  68,  64,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,   0,
   0,   0,   0,   0,   0,   0,   0,   0, 136, 136, 136, 136, 136,   0,   0,   0,   6, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102,  96,   0,   0,
   5,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  85,  80,   0,   0,   0,   0,   0,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  68,  64,
   0,  68,  68,  68,   0,  68,  68,  68,   0,  68,  68,  68,   0,  68,  64,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

/***************************************************************************
Desc:
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::uniIsUpper(
	FLMUNICODE	uzChar)
{
	FLMBOOL	bRV;

	// Which nibble do we need to look at?
	if (uzChar & 0x1)
	{
		// Low nibble
		bRV = (UnicodeProperties[uzChar/2] & UNICODE_UPPERCASE_MASK) ? TRUE : FALSE;	
	}
	else
	{
		// High nibble
		bRV = ((UnicodeProperties[uzChar/2] >> 4) & UNICODE_UPPERCASE_MASK) ? TRUE : FALSE;
	}
	return bRV;
}

/***************************************************************************
Desc:
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::uniIsLower(
	FLMUNICODE	uzChar)
{
	FLMBOOL	bRV;

	// Which nibble do we need to look at?
	if (uzChar & 0x1)
	{
		// Low nibble
		bRV = (UnicodeProperties[uzChar/2] & UNICODE_LOWERCASE_MASK) ? TRUE : FALSE;	
	}
	else
	{
		// High nibble
		bRV = ((UnicodeProperties[uzChar/2] >> 4) & UNICODE_LOWERCASE_MASK) ? TRUE : FALSE;
	}
	return bRV;
}

/***************************************************************************
Desc:
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::uniIsAlpha(
	FLMUNICODE	uzChar)
{
	FLMBOOL	bRV;

	// Which nibble do we need to look at?
	if (uzChar & 0x1)
	{
		// Low nibble
		bRV = (UnicodeProperties[uzChar/2] & UNICODE_ALPHABETIC_MASK) ? TRUE : FALSE;	
	}
	else
	{
		// High nibble
		bRV = ((UnicodeProperties[uzChar/2] >> 4) & UNICODE_ALPHABETIC_MASK) ? TRUE : FALSE;
	}
	return bRV;
}

/***************************************************************************
Desc:
****************************************************************************/
FLMBOOL XFLMAPI F_DbSystem::uniIsDecimalDigit(
	FLMUNICODE	uzChar)
{
	FLMBOOL	bRV;

	// Which nibble do we need to look at?
	if (uzChar & 0x1)
	{
		// Low nibble
		bRV = (UnicodeProperties[uzChar/2] & UNICODE_DECIMAL_DIGIT_MASK) ? TRUE : FALSE;	
	}
	else
	{
		// High nibble
		bRV = ((UnicodeProperties[uzChar/2] >> 4) & UNICODE_DECIMAL_DIGIT_MASK) ? TRUE : FALSE;
	}
	return bRV;
}

/***************************************************************************
Desc:
****************************************************************************/
FLMUNICODE XFLMAPI F_DbSystem::uniToLower(
	FLMUNICODE	uzChar)
{
	return f_unitolower( uzChar);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE	XFLMAPI F_DbSystem::nextUCS2Char(
	const FLMBYTE **		ppszUTF8,
	const FLMBYTE *		pszEndOfUTF8String,
	FLMUNICODE *			puzChar)
{
	return flmGetCharFromUTF8Buf( ppszUTF8, pszEndOfUTF8String, puzChar);
}

/***************************************************************************
Desc:
****************************************************************************/
RCODE XFLMAPI F_DbSystem::numUCS2Chars(
	const FLMBYTE *	pszUTF8,
	FLMUINT *			puiNumChars)
{
	RCODE					rc = NE_XFLM_OK;
	FLMUINT				uiTemp = 0;
	FLMUNICODE			uzChar;
	
	for (;;)
	{
		if (RC_BAD( rc = flmGetCharFromUTF8Buf( &pszUTF8, NULL, &uzChar)))
		{
			goto Exit;
		}
		if (!uzChar)
		{
			break;
		}
		uiTemp++;
	}

Exit:

	*puiNumChars = uiTemp;
	return rc;
}
