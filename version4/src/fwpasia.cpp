//-------------------------------------------------------------------------
// Desc:	Collation for Asian languages.
// Tabs:	3
//
//		Copyright (c) 1991-1992,1994-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fwpasia.cpp 12301 2006-01-19 15:02:55 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#define	SET_CASE_BIT			0x01
#define	SET_KATAKANA_BIT		0x01
#define	SET_WIDTH_BIT			0x02
#define	COLS_ASIAN_MARKS		0x140


extern FLMBYTE		fwp_dia60Tbl[];		/* Diacritic conversions */

/**----------------------------------------------
***  Tables
***  The tables below were taken from the
***  following files:
***    XCH2COL.ASM 
***	 CMPWS.ASM   - k_diac (KanaSubColTbl[])
***---------------------------------------------*/

/**---------------------------------------------
***  Map special chars in CharSet (x24) to 
***  collation values
***--------------------------------------------*/

BYTE_WORD_TBL fwp_Ch24ColTbl[] =	/* Position in the table+1 is subColValue */
{
	{1,	COLLS+2},					/* comma */
	{2,	COLLS+1},					/* maru	 */
	{5,	COLS_ASIAN_MARKS+2},		/* chuuten */
	{10,	COLS_ASIAN_MARKS},		/* dakuten */
	{11,	COLS_ASIAN_MARKS+1},		/* handakuten */
	{43,	COLS2+2},					/* angled brackets */
	{44,	COLS2+3},					/* */
	{49,	COLS2+2},					/* pointy brackets */
	{50,	COLS2+3},	
	{51,	COLS2+2},					/* double pointy brackets */
	{52,	COLS2+3},	
	{53,	COLS1},						/* Japanese quotes */
	{54,	COLS1},
	{55,	COLS1},						/* hollow Japanese quotes */
	{56,	COLS1},
	{57,	COLS2+2},					/* filled rounded brackets */
	{58,	COLS2+3}	
};

/**-------------------------------------
***  Kana subcollation values
***  	 BIT 0: set if large char
***		 BIT 1: set if voiced
***		 BIT 2: set if half voiced
***  Note: 
***    To save space should be nibbles
***  IMPORTANT:
***    The '1' entries that do not have
***    a matching '0' entry have been
***    changed to zero to save space in
***    the subcollation area.
***    The original table is listed below.
***------------------------------------*/

FLMBYTE 	KanaSubColTbl[] = 
{
	0,1,0,1,0,1,0,1,0,1,				/* a    A   i   I   u   U   e   E   o   O */
	1,3,0,3,0,3,1,3,0,3,				/* KA  GA  KI  GI  KU  GU  KE  GE  KO  GO */
	0,3,0,3,0,3,0,3,0,3,				/* SA  ZA SHI  JI  SU  ZU  SE  ZE  SO  ZO */
	0,3,0,3,0,1,3,0,3,0,3,			/* TA  DA CHI  JI tsu TSU  ZU  TE DE TO DO*/
	0,0,0,0,0,							/* NA NI NU NE NO									*/
	0,3,5,0,3,5,0,3,5,				/* HA BA PA HI BI PI FU BU PU					*/
	0,3,5,0,3,5,						/* HE BE PE HO BO PO		*/
	0,0,0,0,0,							/* MA MI MU ME MO			*/
	0,1,0,1,0,1,						/* ya YA yu YU yo YO		*/
	0,0,0,0,0,							/* RA RI RU RE RO			*/
	0,1,0,0,0,							/* wa WA WI WE WO			*/		
	0,3,0,0								/*  N VU ka ke				*/
};

/**
***  Map katakana (CharSet x26) to collation values
***  kana collating values are two byte values
***  where the high byte is 0x01.
**/

FLMBYTE 	KanaColTbl[] = 
{
	 0, 0, 1, 1, 2, 2, 3, 3, 4, 4,/* a    A   i   I   u   U   e   E   o   O */
 	 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,/* KA  GA  KI  GI  KU  GU  KE  GE  KO  GO */
	10,10,11,11,12,12,13,13,14,14,/* SA  ZA SHI  JI  SU  ZU  SE  ZE  SO  ZO */
	15,15,16,16,17,17,17,18,18,19,19,/* TA DA CHI JI tsu TSU  ZU  TE DE TO DO*/
	20,21,22,23,24,					/* NA NI NU NE NO									*/
	25,25,25,26,26,26,27,27,27,	/* HA BA PA HI BI PI FU BU PU					*/
	28,28,28,29,29,29,				/* HE BE PE HO BO PO		*/
	30,31,32,33,34,					/* MA MI MU ME MO			*/
	35,35,36,36,37,37,				/* ya YA yu YU yo YO		*/
	38,39,40,41,42,					/* RA RI RU RE RO			*/
	43,43,44,45,46,					/* wa WA WI WE WO			*/
	47, 2, 5, 8							/*  N VU ka ke				*/
};


/**---------------------------------------
***  Map KataKana collated value to vowel
***  value for use for the previous char.
***--------------------------------------*/
FLMBYTE 	KanaColToVowel[] = 
{
	0,1,2,3,4,		/*  a   i   u  e  o */ 
	0,1,2,3,4,		/* ka  ki  ku ke ko */
	0,1,2,3,4,		/* sa shi  su se so */
	0,1,2,3,4,		/* ta chi tsu te to */
	0,1,2,3,4,		/* na  ni  nu ne no */
	0,1,2,3,4,		/* ha  hi  hu he ho */
	0,1,2,3,4,		/* ma  mi  mu me mo */
	0,2,4,			/* ya  yu  yo		  */
	0,1,2,3,4,		/* ra  ri  ru re ro */
	0,1,3,4,			/* wa  wi  we wo	  */
};

/**
***  Convert Zenkaku (double wide) to Hankaku (single wide)
***  Character set 0x24 maps to single wide chars in other char sets.
***  This enables collation values to be found on some symbols.
***  This is also used to convert symbols from hankaku to Zen24.
***  
**/

BYTE_WORD_TBL	Zen24ToHankaku[] = {
	{	0  ,0x0020 },		/* space */
	{	1  ,0x0b03 },		/* japanese comma */
	{	2  ,0x0b00 },		/* circle period */
	{	3  ,  44	 },		/* comma */
	{	4  ,  46	 },		/* period */
	{	5  ,0x0b04 },		/* center dot	 */
	{	6  ,  58	 },		/* colon */
	{	7  ,  59	 },		/* semicolon */
	{	8  ,  63	 },		/* question mark */
	{	9  ,  33	 },		/* exclamation mark */
	{	10 ,0x0b3d },		/* dakuten */
	{	11 ,0x0b3e },		/* handakuten */
	{	12 ,0x0106 },		/* accent mark */
	{	13 ,  96	 },		/* accent mark */
	{	14 ,0x0107 },		/* umlat */
	{	15 ,  94	 },		/* caret */
	{	16 ,0x0108 },		/* macron */
	{	17 ,  95	 },		/* underscore */
	{	27 ,0x0b0f },		/* extend vowel */
	{	28 ,0x0422 },		/* mdash */
	{	29 ,  45	 },		/* hyphen */
	{	30 ,  47  },     	/* slash */
	{	31 ,0x0607 },		/* backslash */
	{	32 , 126	 },		/* tilde */
	{	33 ,0x0611 },		/* doubleline */
	{	34 ,0x0609 },		/* line */
	{	37 ,0x041d },		/* left apostrophe */
	{	38 ,0x041c },		/* right apostrophe */
	{	39 ,0x0420 },		/* left quote */
	{	40 ,0x041f },		/* right quote */
	{	41 ,  40	 },		/* left paren */
	{	42 ,  41	 },		/* right paren */
	{	45 ,  91	 },		/* left bracket */
	{	46 ,  93	 },		/* right bracket */
	{	47 , 123	 },		/* left curly bracket */
	{	48 , 125	 },		/* right curly bracket */
	{	53 ,0x0b01 },		/* left j quote */
	{	54 ,0x0b02 },		/* right j quote */
	{	59 ,  43	 },		/* plus */
	{	60 ,0x0600 },		/* minus */
	{	61 ,0x0601 },		/* plus/minus */
	{	62 ,0x0627 },		/* times */
	{	63 ,0x0608 },		/* divide */
	{	64 ,  61	 },		/* equal */
	{	65 ,0x0663 },		/* unequal */
	{	66 ,  60	 },		/* less */
	{	67 ,  62	 },		/* greater */
	{	68 ,0x0602 },		/* less/equal */
	{	69 ,0x0603 },		/* greater/equal */
	{	70 ,0x0613 },		/* infinity */
	{	71 ,0x0666 },		/* traingle dots */
	{	72 ,0x0504 },		/* man */
	{	73 ,0x0505 },		/* woman */
	{	75 ,0x062d },		/* prime */
	{	76 ,0x062e },		/* double prime */
	{	78 ,0x040c },		/* yen */
	{	79 ,  36	 },		/* $ */
	{	80 ,0x0413 },		/* cent */
	{	81 ,0x040b },		/* pound */
	{	82 ,  37	 },		/* % */
	{	83 ,  35	 },		/* # */
	{	84 ,  38	 },		/* & */
	{	85 ,  42	 },		/* * */
	{	86 ,  64	 },		/* @ */
	{	87 ,0x0406 },		/* squiggle */
	{	89 ,0x06b8 },		/* filled star */
	{	90 ,0x0425 },		/* hollow circle */
	{	91 ,0x042c },		/* filled circle */
	{	93 ,0x065f },		/* hollow diamond */
	{	94 ,0x0660 },		/* filled diamond */
	{	95 ,0x0426 },		/* hollow box */
	{	96 ,0x042e },		/* filled box */
	{	97 ,0x0688 },		/* hollow triangle */
	{	99 ,0x0689 },		/* hollow upside down triangle */
	{	103,0x0615 },		/* right arrow */
	{	104,0x0616 },		/* left arrow */
	{	105,0x0617 },		/* up arrow */
	{	106,0x0622 },		/* down arrow */
	{	119,0x060f },		/*  */
	{	121,0x0645 },		/*  */
	{	122,0x0646 },
	{	123,0x0643 },
	{	124,0x0644 },
	{	125,0x0642 },		/* union */
	{	126,0x0610 },		/* intersection */
	{	135,0x0655 },
	{	136,0x0656 },
	{	138,0x0638 },		/* right arrow */
	{	139,0x063c },		/* left/right arrow */
	{	140,0x067a },		/*  */
	{	141,0x0679 },
	{	153,0x064f },		/* angle */
	{	154,0x0659 },
	{	155,0x065a },
	{	156,0x062c },
	{	157,0x062b },
	{	158,0x060e },
	{	159,0x06b0 },
	{	160,0x064d },
	{	161,0x064e },
	{	162,0x050e },		/* square root */
	{	164,0x0604 },
	{	175,0x0623 },		/* angstrom */
	{	176,0x044b },		/* percent */
	{	177,0x051b },		/* sharp */
	{	178,0x051c },		/* flat */
	{	179,0x0509 },		/* musical note	 */
	{	180,0x0427 },		/* dagger */
	{	181,0x0428 },		/* double dagger */
	{	182,0x0405 },		/* paragraph */
	{	187,0x068f }		/* big hollow circle */
};

/**
***	Maps CS26 to CharSet 11
***	Taken from Char.asm
***   Used to uncollate characters for FLAIM - placed here for consistency
***	0x80 - add dakuten
***   0xC0 - add handakuten
***   0xFF - no mapping exists
**/
FLMBYTE 	MapCS26ToCharSet11[ 86 ] = {
	0x06,	/* 0     a  */
	0x10,	/*	1     A  */ 
	0x07,	/*	2     i  */
	0x11,	/*	3     I  */ 
	0x08,	/*	4     u  */
	0x12,	/*	5     U  */ 
	0x09,	/*	6     e  */
	0x13,	/*	7     E  */ 
	0x0a,	/*	8     o  */
	0x14,	/*	9     O  */

	0x15,	/*	0x0a  KA */
	0x95,	/*       GA - 21 followed by 0x3D dakuten */
	
	0x16,	/* 0x0c  KI */
	0x96,	/*       GI */
	0x17,	/*	0x0e  KU */
	0x97,	/*       GU */
	0x18,	/* 0x10  KE */
	0x98,	/*       GE */
	0x19,	/* 0x12  KO */
	0x99,	/*       GO */

	0x1a,	/*	0x14  SA */
	0x9a,	/*       ZA */
	0x1b,	/*	0x16  SHI */
	0x9b,	/*       JI */
	0x1c,	/*	0x18  SU */
	0x9c,	/*       ZU */
	0x1d,	/*	0x1a  SE */
	0x9d,	/*       ZE */
	0x1e,	/*	0x1c  SO */
	0x9e,	/*       ZO */

	0x1f,	/*	0x1e  TA */
	0x9f,	/*       DA */
	0x20,	/*	0x20  CHI */
	0xa0,	/*       JI */
	0x0e,	/*	0x22  small tsu */
	0x21,	/*	0x23  TSU */
	0xa1,	/*       ZU */
	0x22,	/*	0x25  TE */
	0xa2,	/*       DE */
	0x23,	/*	0x27  TO */
	0xa3,	/*       DO */

	0x24,	/*	0x29  NA */
	0x25,	/*	0x2a  NI */
	0x26,	/* 0x2b  NU */
	0x27,	/*	0x2c  NE */
	0x28,	/*	0x2d  NO */

	0x29,	/*	0x2e  HA */
	0xa9,	/* 0x2f  BA */
	0xe9,	/* 0x30  PA */
	0x2a,	/*	0x31  HI */
	0xaa,	/* 0x32  BI */
	0xea,	/* 0x33  PI */
	0x2b,	/*	0x34  FU */
	0xab,	/* 0x35  BU */
	0xeb,	/* 0x36  PU */
	0x2c,	/*	0x37  HE */
	0xac,	/* 0x38  BE */
	0xec,	/* 0x39  PE */
	0x2d,	/*	0x3a  HO */
	0xad,	/* 0x3b  BO */
	0xed,	/* 0x3c  PO */

	0x2e,	/*	0x3d  MA */
	0x2f,	/*	0x3e  MI */
	0x30,	/*	0x3f  MU */
	0x31,	/*	0x40  ME */
	0x32,	/*	0x41  MO */

	0x0b,	/*	0x42  small ya */
	0x33,	/*	0x43  YA */
	0x0c,	/*	0x44  small yu */
	0x34,	/*	0x45  YU */
	0x0d,	/*	0x46  small yo */
	0x35,	/*	0x47  YO */

	0x36,	/*	0x48  RA */
	0x37,	/*	0x49  RI */
	0x38,	/*	0x4a  RU */
	0x39,	/*	0x4b  RE */
	0x3a,	/*	0x4c  RO */

	0xff,	/* 0x4d  small wa */
	0x3b,	/*	0x4e  WA */
	0xff,	/* 0x4f  WI */
	0xff,	/* 0x50  WE */
	0x05,	/*	0x51	WO */

	0x3c,	/*	0x52	N  */
	0xff,	/* 0x53  VU */
	0xff, /* 0x54  ka */
	0xff 	/* 0x55  ke */
};


/**
***  Conversion from single (Hankaku) to double (Zenkaku) wide characters
***  Used in HanToZenkaku()
**/

/* maps from charset 11 to CS24 (punctuation) (starting from 11,0) */

FLMBYTE  From0AToZen[] = {		/* ' changed because of windows */
	 	0, 	9,		40,	0x53, 		/* sp ! " # */
	 	0x4f, 0x52, 0x54,	38, 			/* $ % & ' */
	 											/* Was 187 for ! and 186 for ' */
		0x29,	0x2a,	0x55,	0x3b, 		/* ( ) * + */
		3,		0x1d,	4,		0x1e	 		/* , - . / */
  };
	
FLMBYTE  From0BToZen[] = {
		6,		7,		0x42,	0x40,			/* : ; < = */
		0x43,	8,		0x56					/* > ? @ */
  };

FLMBYTE  From0CToZen[] = {
		0x2d,	0x1f,	0x2e,	0x0f,	0x11,	0x0d	/* [ \ ] ^ _ ` */
  };
	
FLMBYTE  From0DToZen[] = {
		0x2f,	0x22,	0x30,	0x20 			/* { | } ~ */
  };

FLMBYTE  From8ToZen[] = {		/* Fast way to convert from 8 to zen */
	0x5e, 0x7e, 0x5f, 0x7f, 0x5f, 0xFF, 0x60, 0x80,
	0x61, 0x81, 0x62, 0x82, 0x63, 0x83, 0x64, 0x84,
	0x65, 0x85, 0x66, 0x86, 0x67, 0x87, 0x68, 0x88,
	0x69, 0x89, 0x6a, 0x8a, 0x6b, 0x8b, 0x6c, 0x8c,
	0x6d, 0x8d, 0x6e, 0x8e, 0x6f, 0x8f, 0x6f, 0xFF,
	0x70, 0x90, 0x71, 0x91, 0x72, 0x92, 0x73, 0x93,
	0x74, 0x94, 0x75, 0x95
  };

static FLMBYTE  From11AToZen[] = {		/* 11 to 24 punctuation except dash */
		2,			/* japanese period	*/
		0x35,		/* left bracket		*/
		0x36,		/* right bracket		*/
		0x01,		/* comma					*/
		0x05		/* chuuten				*/
  };

static FLMBYTE	From11BToZen[] = {		/* 11 to 26 (katakana) from 11,5 */
		0x51,										/* wo 									*/
		0,2,4,6,8,0x42,0x44,0x46,0x22,	/* small a i u e o ya yu yo tsu	*/
		0xFF, 1, 3, 5, 7, 9,					/* dash (x241b) a i u e o			*/
		0x0a, 0x0c, 0x0e, 0x10, 0x12,		/* ka ki ku ke ko						*/
		0x14, 0x16, 0x18, 0x1a, 0x1c,		/* sa shi su se so				*/
		0x1e, 0x20, 0x23, 0x25, 0x27,		/* ta chi tsu te to				*/
		0x29, 0x2a, 0x2b, 0x2c, 0x2d,		/* na ni nu ne no					*/
		0x2e, 0x31, 0x34, 0x37, 0x3a,		/* ha hi fu he ho					*/
		0x3d, 0x3e, 0x3f, 0x40, 0x41,		/* ma mi mu me mo					*/
		0x43, 0x45, 0x47,						/* ya yu yo							*/
		0x48, 0x49, 0x4a, 0x4b, 0x4c,		/* ra ri ru re ro					*/
		0x4e, 0x52								/* WA N								*/
  };												/* does not have wa WI WE VU ka ke */

/****************************************************************************
Desc:	Returns the collation value of the input Wp character.
		If in charset 11 will convert the character to Zenkaku (double wide).
In:	ui16WpChar - Char to collate off of - could be in CS0..14 or x24..up
		ui16NextWpChar - next WP char for CS11 voicing marks
		ui16PrevColValue - previous collating value - for repeat/vowel repeat
		pui16ColValue - returns 2 byte collation value
		pui16SubColVal - 0, 6 or 16 bit value for the latin sub collation
								or the kana size & vowel voicing
								001 - set if large (upper) character
								010 - set if voiced
								100 - set if half voiced
								
		pucCaseBits - returns 2 bits 
				Latin/Greek/Cyrillic
					01 - case bit set if character is uppercase
					10 - double wide character in CS 0x25xx, 0x26xx and 0x27xx
				Japanese
					00 - double wide hiragana 0x255e..25b0
					01 - double wide katakana 0x2600..2655
					10 - double wide symbols that map to charset 11
					11 - single wide katakana from charset 11
Ret:	0 - no valid collation value 
				high values set for pui16ColValue
				Sub-collation gets original WP character value
		1 - valid collation value
		2 - valid collation value and used the ui16NextWpChar
			
Notes:	Code taken from XCH2COL.ASM - routine xch2col_f
			also from CMPWS.ASM - routine getcase
Terms:			
	HANKAKU - single wide characters in charsets 0..14
	ZENKAKU - double wide characters in charsets 0x24..end of kanji
	KANJI   - collation values are 0x2900 less than WPChar value
	
****************************************************************************/
FLMUINT16 fwpAsiaGetCollation(
	FLMUINT16	ui16WpChar,				// WP char to get collation values
	FLMUINT16	ui16NextWpChar,		// Next WP char - for CS11 voicing marks
	FLMUINT16   ui16PrevColValue,		// Previous collating value
	FLMUINT16 *	pui16ColValue,			// Returns collation value
	FLMUINT16 * pui16SubColVal,		// Returns sub-collation value
	FLMBYTE *	pucCaseBits,		 	// Returns case bits value
	FLMUINT16	uiUppercaseFlag		// Set if to convert to uppercase
	)
{
	FLMUINT16	ui16ColValue;
	FLMUINT16	ui16SubColVal;
	FLMBYTE		ucCaseBits = 0;
	FLMBYTE		ucCharSet = ui16WpChar >> 8;
	FLMBYTE		ucCharVal = ui16WpChar & 0xFF;
	FLMUINT16	ui16Hankaku;
	FLMUINT		uiLoop;
	FLMUINT16	ui16ReturnValue = 1;

	ui16ColValue = ui16SubColVal = 0;

	// Kanji or above

	if (ucCharSet >= 0x2B)
	{

		// Puts 2 or above into high byte.

		ui16ColValue = ui16WpChar - 0x2900;

		// No subcollation or case bits need to be set

		goto	Exit;
	}

	// Single wide character? (HANKAKU)

	if (ucCharSet < 11)
	{
		// Get the values from a non-asian character
		// LATIN, GREEK or CYRILLIC
		// The width bit may have been set on a jump to
		// label from below.

Latin_Greek_Cyrillic:

		// YES: Pass US_LANG because this is what we want -
		// Prevents double character sorting.

		ui16ColValue = fwpGetCollation( ui16WpChar, US_LANG);

		if (uiUppercaseFlag || fwpIsUpper( ui16WpChar))
		{
			// Uppercase - set case bit

			ucCaseBits |= SET_CASE_BIT;
		}

		// Character for which there is no collation value?

		if (ui16ColValue == COLS0)
		{
			ui16ReturnValue = 0;
			if (!fwpIsUpper( ui16WpChar))
			{

				// Convert to uppercase

				ui16WpChar--;
			}
			ui16ColValue = 0xFFFF;
			ui16SubColVal = ui16WpChar;
		}
		else if (ucCharSet) 				// Don't bother with ascii
		{
			if (!fwpIsUpper( ui16WpChar))
			{

				// Convert to uppercase

				ui16WpChar--;
			}

        	if (ucCharSet == CHSMUL1)
			{
				FLMUINT16	ui16Base;
				FLMUINT16	ui16Diacritic;

				ui16SubColVal = !fwpCh6Brkcar( ui16WpChar, &ui16Base,
															&ui16Diacritic)
									  ? fwp_dia60Tbl[ ui16Diacritic & 0xFF]
									  : ui16WpChar;
			}
			else if (ucCharSet == CHSGREK) // GREEK
         {
         	if (ui16WpChar >= 0x834 ||		// [8,52] or above
            	 ui16WpChar == 0x804 ||		// [8,4] BETA Medial | Terminal
					 ui16WpChar == 0x826)		// [8,38] SIGMA terminal
					ui16SubColVal = ui16WpChar;
			}
			else if (ucCharSet == CHSCYR)	// CYRILLIC
			{
           	if (ui16WpChar >= 0xA90)		// [10, 144] or above
				{
              	ui16SubColVal = ui16WpChar;	// Dup collation values
				}
         }
         // else don't need a sub collation value
      }
		goto	Exit;
	}	

	// Single wide Japanese character?

 	if (ucCharSet == 11)
	{
		FLMUINT16	ui16KanaChar;

		// Convert charset 11 to Zenkaku (double wide) CS24 or CS26 hex.
		// All characters in charset 11 will convert to CS24 or CS26.
		// when combining the collation and the sub-collation values.

		if (HanToZenkaku( ui16WpChar, ui16NextWpChar, &ui16KanaChar ) == 2)
		{

			// Return 2

			ui16ReturnValue++;
		}

		ucCaseBits |= SET_WIDTH_BIT;	// Set so will allow to go back
		ui16WpChar = ui16KanaChar;		// If in CS24 will fall through to ZenKaku
		ucCharSet = ui16KanaChar >> 8;
		ucCharVal = ui16KanaChar & 0xFF;
	}
	
	if (ui16WpChar < 0x2400)
	{

		// In some other character set

		goto Latin_Greek_Cyrillic;
	}
	else if (ui16WpChar >= 0x255e &&	// Hiragana?
				ui16WpChar <= 0x2655)	// Katakana?
	{
		if (ui16WpChar >= 0x2600)
		{
			ucCaseBits |= SET_KATAKANA_BIT;
		}
			
		// HIRAGANA & KATAKANA
		//		Kana contains both hiragana and katakana.
		//		The tables contain the same characters in same order

		if (ucCharSet == 0x25)
		{

			// Change value to be in character set 26

			ucCharVal -= 0x5E;
		}
	
		ui16ColValue = 0x0100 + KanaColTbl[ ucCharVal ];
		ui16SubColVal = KanaSubColTbl[ ucCharVal ];
		goto Exit;
	}

	// ZenKaku - means any double wide character
	// Hankaku - single wide character

	//		Inputs:	0x2400..2559	symbols..latin  - Zenkaku
	//					0x265B..2750	greek..cyrillic - Zenkaku
	
	//	SET_WIDTH_BIT may have been set if original char
	// was in 11 and got converted to CS24. [1,2,5,27(extendedVowel),53,54]
	// Original chars from CS11 will have some collation value that when
	// combined with the sub-collation value will format a character in 
	// CS24.  The width bit will then convert back to CS11.

	if ((ui16Hankaku = ZenToHankaku( ui16WpChar, (FLMUINT16 *) 0 )) != 0)
	{
		if ((ui16Hankaku >> 8) != 11)			// if CharSet11 was a CS24 symbol
		{
			ui16WpChar = ui16Hankaku;			// May be CS24 symbol/latin/gk/cy
			ucCharSet = ui16WpChar >> 8;
			ucCharVal = ui16WpChar & 0xFF;
			ucCaseBits |= SET_WIDTH_BIT;		// Latin symbols double wide
			goto Latin_Greek_Cyrillic;
		}
	}

	// 0x2400..0x24bc Japanese symbols that cannot be converted to Hankaku.
	// All 6 original symbol chars from 11 will also be here.
	// First try to find a collation value of the symbol.
	// The sub-collation value will be the position in the CS24 table + 1.

	for (uiLoop = 0; 
		  uiLoop < (sizeof(fwp_Ch24ColTbl) / sizeof(BYTE_WORD_TBL));
	  	  uiLoop++ )
	{
		if (ucCharVal == fwp_Ch24ColTbl[ uiLoop].ByteValue)
		{
			if ((ui16ColValue = fwp_Ch24ColTbl[ uiLoop].WordValue) < 0x100)
			{

				// Don't save for chuuten, dakuten, handakuten

				ui16SubColVal = (FLMUINT16)(uiLoop + 1);
			}
			break;
		}
	}
	if (!ui16ColValue)
	{

		// Now see if it's a repeat or repeat-vowel character
		
		if( (((ucCharVal >= 0x12) && (ucCharVal <= 0x15)) ||
			   (ucCharVal == 0x17) ||
			   (ucCharVal == 0x18)) && 
		  		((ui16PrevColValue >> 8) == 1))
		{
			ui16ColValue = ui16PrevColValue;

			// Store original WP character

			ui16SubColVal = ui16WpChar;
		}
		else if( (ucCharVal == 0x1B) &&						// repeat vowel?
					(ui16PrevColValue >= 0x100) &&
					(ui16PrevColValue < COLS_ASIAN_MARKS))	// Previous kana char?
		{
			ui16ColValue = 0x0100 + KanaColToVowel[ ui16PrevColValue & 0xFF ];

			// Store original WP character

			ui16SubColVal = ui16WpChar;
		}
		else
		{
			ui16ReturnValue = 0;
			ui16ColValue = 0xFFFF;			// No collation value
			ui16SubColVal = ui16WpChar;	// Never have changed if gets here
		}
	}
	
Exit:

	// Set return values

	*pui16ColValue = ui16ColValue;
	*pui16SubColVal = ui16SubColVal;
	*pucCaseBits = ucCaseBits;

	return( ui16ReturnValue);
}

/****************************************************************************
Desc:	Convert a zenkaku (double wide) char to a hankaku (single wide) char
Ret:	Hankaku char or 0 if a conversion doesn't exist
Notes:	Taken from CHAR.ASM -  zen2han_f routine
****************************************************************************/
FLMUINT16 ZenToHankaku( 
	FLMUINT16	ui16WpChar,
	FLMUINT16 * DakutenOrHandakutenRV )
{
	FLMUINT16	ui16Hankaku = 0;
	FLMBYTE		ucCharSet = ui16WpChar >> 8;
	FLMBYTE		ucCharVal = ui16WpChar & 0xFF;
	FLMUINT		uiLoop;

	switch (ucCharSet)
	{
		// SYMBOLS

		case 0x24:
			for (uiLoop = 0;
				  uiLoop < (sizeof(Zen24ToHankaku) / sizeof(BYTE_WORD_TBL));
				  uiLoop++)
			{
				// List is sorted so table entry is more you are done

				if (Zen24ToHankaku [uiLoop].ByteValue >= ucCharVal)
				{
					if (Zen24ToHankaku [uiLoop].ByteValue == ucCharVal)
					{
						ui16Hankaku = Zen24ToHankaku [uiLoop].WordValue;
					}
					break;
				}
			}
			break;

		// ROMAN - 0x250F..2559
		// Hiragana - 0x255E..2580

		case 0x25:
			if (ucCharVal >= 0x0F && ucCharVal < 0x5E)
			{
				ui16Hankaku = ucCharVal + 0x21;
			}
			break;
			
		// Katakana - 0x2600..2655
		// Greek - 0x265B..2695

		case 0x26:
			if (ucCharVal <= 0x55)		// Katakana range
			{
				FLMBYTE		ucCS11CharVal;
				FLMUINT16	ui16NextWpChar = 0;
					
				if ((ucCS11CharVal = MapCS26ToCharSet11[ ucCharVal ]) != 0xFF)
				{
					if (ucCS11CharVal & 0x80)
					{
						if( ucCS11CharVal & 0x40)
						{

							// Handakuten voicing

							ui16NextWpChar = 0xB3E;
						}
						else
						{

							// Dakuten voicing

							ui16NextWpChar = 0xB3D;
						}
						ucCS11CharVal &= 0x3F;
					}
					ui16Hankaku = 0x0b00 + ucCS11CharVal;
					if( ui16NextWpChar && DakutenOrHandakutenRV )
					{
						*DakutenOrHandakutenRV = ui16NextWpChar;
					}
				}
			}
			else if (ucCharVal <= 0x95)	// Greek
			{
				FLMBYTE	ucGreekChar = ucCharVal;

				// Make a zero based number.

				ucGreekChar -= 0x5E;

				// Check for lowercase
				if( ucGreekChar >= 0x20)
				{

					// Convert to upper case for now

					ucGreekChar -= 0x20;
				}
				if (ucGreekChar >= 2)
				{
					ucGreekChar++;
				}
				if (ucGreekChar >= 19)
				{
					ucGreekChar++;
				}

				// Convert to character set 8
					
				ui16Hankaku = (ucGreekChar << 1) + 0x800;
				if (ucCharVal >= (0x5E + 0x20))
				{

					// Adjust to lower case character

					ui16Hankaku++;
				}
			}
			break;

		// Cyrillic
				
		case 0x27:

			// Uppercase?

			if (ucCharVal <= 0x20)
			{
				ui16Hankaku = (ucCharVal << 1) + 0xa00;
			}
			else if (ucCharVal >= 0x30 && ucCharVal <= 0x50)
			{

				// Lower case

				ui16Hankaku = ((ucCharVal - 0x30) << 1) + 0xa01;
			}
			break;
	}

	return( ui16Hankaku);
}

/****************************************************************************
Desc:		Convert a WPChar from hankaku (single wide) to zenkaku (double wide).
			1) Used to see if a char in CS11 can map to a double wide character
			2) Used to convert keys into original data.
Ret:		0 = no conversion
			1 = converted character to zenkaku
			2 = ui16NextWpChar dakuten or handakuten voicing got combined 
Notes:	Taken from char.asm - han2zen()
			From8ToZen could be taken out and placed in code.
****************************************************************************/
FLMUINT16 HanToZenkaku( 
	FLMUINT16	ui16WpChar,
	FLMUINT16	ui16NextWpChar,
	FLMUINT16 *	pui16Zenkaku)
{
	FLMUINT16	ui16Zenkaku = 0;
	FLMBYTE		ucCharSet = ui16WpChar >> 8;
	FLMBYTE		ucCharVal = ui16WpChar & 0xFF;
	FLMUINT		uiLoop;
	FLMUINT16	ui16CharsUsed = 1;

	switch( ucCharSet)
	{
		// Character set 0 - symbols

		case 0:

			// Invalid? - all others are used.

			if (ucCharVal < 0x20)
			{
				;
			}
			else if (ucCharVal <= 0x2F)
			{

				// Symbols A

				ui16Zenkaku = 0x2400 + From0AToZen[ ucCharVal - 0x20 ];
			}
			else if (ucCharVal <= 0x39)
			{

				// 0..9

				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if (ucCharVal <= 0x40)
			{

				// Symbols B

				ui16Zenkaku = 0x2400 + From0BToZen[ ucCharVal - 0x3A ];
			}
			else if (ucCharVal <= 0x5A)
			{
				
				// A..Z

				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if (ucCharVal <= 0x60)
			{
				
				// Symbols C

				ui16Zenkaku = 0x2400 + From0CToZen[ ucCharVal - 0x5B ];
			}
			else if (ucCharVal <= 0x7A)
			{

				// a..z

				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if (ucCharVal <= 0x7E)
			{
				
				// Symbols D

				ui16Zenkaku = 0x2400 + From0DToZen[ ucCharVal - 0x7B ];
			}
			break;

		// GREEK
			
		case 8:
			if ((ucCharVal >= sizeof( From8ToZen)) ||
				 ((ui16Zenkaku = 0x2600 + From8ToZen[ ucCharVal ]) == 0x26FF))
			{
				ui16Zenkaku = 0;
			}
			break;

		// CYRILLIC
		
		case 10:

			// Check range

			ui16Zenkaku = 0x2700 + (ucCharVal >> 1);	// Uppercase value

			// Convert to lower case?

			if( ucCharVal & 0x01)
			{
				ui16Zenkaku += 0x30;
			}
			break;

		// JAPANESE
			
		case 11:
			if (ucCharVal < 5)
			{
				ui16Zenkaku = 0x2400 + From11AToZen[ ucCharVal ];
			}
			else if (ucCharVal < 0x3D)		// katakana?
			{
				if ((ui16Zenkaku = 0x2600 +
							From11BToZen[ ucCharVal - 5 ]) == 0x26FF)
				{

					// Dash - convert to this

					ui16Zenkaku = 0x241b;
				}
				else
				{
					if (ui16NextWpChar == 0xB3D)		// dakuten? - voicing
					{

						// First check exception(s) then
						// check if voicing exists! - will NOT access out of table

						if ((ui16Zenkaku != 0x2652) &&	// is not 'N'?
							 (KanaSubColTbl[ ui16Zenkaku - 0x2600 + 1 ] == 3))
						{
							ui16Zenkaku++;

							// Return 2

							ui16CharsUsed++;
						}
					}
					else if (ui16NextWpChar == 0xB3E)	// handakuten? - voicing
					{

						// Check if voicing exists! - will NOT access out of table

						if (KanaSubColTbl [ui16Zenkaku - 0x2600 + 2 ] == 5)
						{
							ui16Zenkaku += 2;

							// Return 2

							ui16CharsUsed++;
						}
					}
				}
			}
			else if (ucCharVal == 0x3D)		// dakuten?
			{

				// Convert to voicing symbol

				ui16Zenkaku = 0x240A;
			}
			else if (ucCharVal == 0x3E)		// handakuten?
			{

				// Convert to voicing symbol

				ui16Zenkaku = 0x240B;
			}
			// else cannot convert
			break;

		// Other character sets
		// CS 1,4,5,6 - symbols
		
		default:

			// Instead of includes more tables from char.asm - look down the
			// Zen24Tohankaku[] table for a matching value - not much slower.

			for (uiLoop = 0;
				  uiLoop < (sizeof(Zen24ToHankaku) / sizeof(BYTE_WORD_TBL));
				  uiLoop++)
			{
				if (Zen24ToHankaku[ uiLoop].WordValue == ui16WpChar)
				{
					ui16Zenkaku = 0x2400 + Zen24ToHankaku[ uiLoop].ByteValue;
					break;
				}
			}
			break;
	}
	if (!ui16Zenkaku)
	{

		// Change return value

		ui16CharsUsed = 0;
	}
		
	*pui16Zenkaku = ui16Zenkaku;
	return( ui16CharsUsed);
}
