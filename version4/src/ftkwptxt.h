//-------------------------------------------------------------------------
// Desc:	WP text and character definitions.
// Tabs:	3
//
//		Copyright (c) 1991-1992,1994-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftkwptxt.h 12299 2006-01-19 15:01:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#ifndef FTKWPTXT_H
#define FTKWPTXT_H

#include "fpackon.h"

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

#define Upper_JP_A	0x2520
#define Upper_JP_Z	0x2539
#define Upper_KR_A	0x5420
#define Upper_KR_Z	0x5439
#define Upper_CS_A	0x82FC
#define Upper_CS_Z	0x8316
#define Upper_CT_A	0xA625
#define Upper_CT_Z	0xA63E

#define Lower_JP_a	0x2540
#define Lower_JP_z	0x2559
#define Lower_KR_a	0x5440
#define Lower_KR_z	0x5459
#define Lower_CS_a	0x82DC
#define Lower_CS_z	0x82F5
#define Lower_CT_a	0xA60B
#define Lower_CT_z	0xA624

/* character set #'s are same as high byte values except for algorithmic set. */

/* ------	character code high byte values for character sets */

#define	CHSASCI		0			/* ASCII */
#define	CHSMUL1		1			/* Multinational 1 */
#define	CHSMUL2		2			/* Multinational 2 */
#define	CHSBOXD		3			/* Box drawing */
#define	CHSSYM1		4			/* Typographic Symbols */
#define	CHSSYM2		5			/* Iconic Symbols */
#define	CHSMATH		6			/* Math */
#define	CHMATHX		7			/* Math Extension */
#define	CHSGREK		8			/* Greek */
#define	CHSHEB		9			/* Hebrew */
#define	CHSCYR		10			/* Cyrillic */
#define	CHSKANA		11			/* Japanese Kana */
#define	CHSUSER		12			/* User-defined */
#define  CHSARB1		13			/* Arabic */
#define  CHSARB2		14			/* Arabic script */

#define	NCHSETS		15			/* # of character sets (excluding asian) */
#define	ACHSETS		0x0E0		/* maximum character set value - asian	*/
#define	ACHSMIN		0x024		/* minimum character set value - asian	*/
#define	ACHCMAX		0x0FE		/* maxmimum character value in asian sets	*/


/* 
	# of characters in each character set. 
	CHANGING ANY OF THESE DEFINES WILL CAUSE BUGS!
*/
#define	ASC_N		95
#define	ML1_N		242
#define	ML2_N		145
#define	BOX_N		88
#define	TYP_N		103
#define	ICN_N		255
#define	MTH_N		238
#define	MTX_N		229
#define	GRK_N		219
#define	HEB_N		123
#define	CYR_N		250
#define	KAN_N		63
#define	USR_N		255
#define	ARB_N		196
#define	ARS_N		220

/*	
	TOTAL:	1447 WP + 255 User Characters
*/

#define	C_N 		ASC_N+ML1_N+ML2_N+BOX_N+MTH_N+MTX_N+TYP_N+ICN_N+GRK_N+\
                  HEB_N+CYR_N+KAN_N+USR_N+ARB_N+ARS_N

/*
	Definitions for diacritics.
*/

#define grave		0
#define centerd	1
#define tilde		2
#define circum		3	
#define crossb		4
#define slash		5	
#define acute		6
#define umlaut		7
#define macron		8

#define aposab		9
#define aposbes	10
#define aposba		11

#define ring		14
#define dota		15
#define dacute		16
#define cedilla	17
#define ogonek		18
#define caron		19
#define stroke		20

#define breve		22
#define dotlesi	239
#define dotlesj	25

#define gacute		83		/* greek acute */
#define gdia		84		/* greek diaeresis */
#define gactdia	85		/* acute diaeresis */
#define ggrvdia	86		/* grave diaeresis */
#define ggrave		87		/* greek grave */
#define gcircm		88		/* greek circumflex */
#define gsmooth	89		/* smooth breathing */
#define grough		90		/* rough breathing */
#define giota		91		/* iota subscript */
#define gsmact		92		/* smooth breathing acute */
#define grgact		93		/* rough breathing acute */
#define gsmgrv		94		/* smooth breathing grave */
#define grggrv		95		/* rough breathing grave */
#define gsmcir		96		/* smooth breathing circumflex */
#define grgcir		97		/* rough breathing circumflex */
#define gactio		98		/* acute iota */
#define ggrvio		99		/* grave iota */
#define gcirio		100	/* circumflex iota */
#define gsmio		101	/* smooth iota */
#define grgio		102	/* rough iota */
#define gsmaio		103	/* smooth acute iota */
#define grgaio		104	/* rough acute iota */
#define gsmgvio	105	/* smooth grave iota */
#define grggvio	106	/* rough grave iota */
#define gsmcio		107	/* smooth circumflex iota */
#define grgcio		108	/* rough circumflex iota */
#define ghprime	81		/* high prime */
#define glprime	82		/* low prime */

#define racute		200	/* russian acute */
#define rgrave		201	/* russian grave */
#define rrtdesc	204	/* russian right descender */
#define rogonek	205	/* russian ogonek */
#define rmacron	206	/* russian macron */

typedef struct base_diacrit_table
{
	FLMBYTE		base;
	FLMBYTE		diacrit;
} BASE_DIACRIT_TABLE, *BASE_DIACRIT_TABLEP;

typedef struct base_diacrit
{
	FLMUINT16	char_count;		/*# of characters in table.*/
	FLMUINT16	start_char;		/*start char.*/
	BASE_DIACRIT_TABLEP table;

} BASE_DIACRIT, *BASE_DIACRITP;

typedef struct cpose
{
	FLMBYTE		chars[2];
	FLMUINT16	wpchar;
} CPOSE;

/* Collating Sequence Equates */
/* NOTE:The collating sequence MUST start at 32 (20h).  This allows for the */
/* 	handling of nulls and control characters. */
/* 	(see \shar50\wp50lib\cmpws.asm) */

#define	COLLS	32				/*  first collating number (space/end of line) */
#define	COLS1	(COLLS+9)		/*  quotes */
#define	COLS2	(COLS1+5)		/*  parens */
#define	COLS3	(COLS2+6)		/*  money */
#define	COLS4	(COLS3+6)		/*  math ops */
#define	COLS5	(COLS4+8)		/*  math others */
#define	COLS6	(COLS5+14)		/*  others: %#&@\_|~ */
#define	COLS7	(COLS6+13)		/*  greek */
#define	COLS8	(COLS7+25)		/*  numbers */
#define	COLS9	(COLS8+10)		/*  alphabet */
									/* Three below will overlap each other */
#define	COLS10	(COLS9+60)	/*  cyrillic */
#define	COLS10h	(COLS9+42)	/*  hebrew - writes over european & cyrilic */
#define	COLS10a	(COLS10h+28)/*  arabic - inclusive from 198(C6)-252(FC) */

#define	COLS11	253		/*	End of list - arabic goes to the end */

#define	COLS0_ARABIC	COLS11	/* Set if arabic accent marking */
#define	COLS0_HEBREW	COLS11	/* Set if hebrew accent marking */

#define	COLSOEM	254		/*  OEM character in upper range - non-collatable*/
									/*  Phase COLSOEM out - not used! */
#define	COLS0_UNICODE 254	/*  Use this for UNICODE */									
#define	COLS0	255			/*  graphics/misc - chars without a collate value*/

/*
***	Language definitions - to get rid of testing "US" or multiple bytes
***	will define needed languages as a number with backward conversions.
*** Keep these defines synchronized with the table in wps6cmpc.c 
**/
#define	US_LANG	0		/* English, United States  */
#define	AF_LANG	1		/* Afrikaans               */
#define	AR_LANG	2		/* Arabic                  */
#define	CA_LANG	3		/* Catalan                 */
#define	HR_LANG	4		/* Croatian                */
#define	CZ_LANG	5		/* Czech                   */
#define	DK_LANG	6		/* Danish                  */
// JDP - SCO already defines NL_LANG as something else
#define	_NL_LANG	7		/* Dutch                   */
#define	OZ_LANG	8		/* English, Australia      */
#define	CE_LANG	9		/* English, Canada         */
#define	UK_LANG	10		/* English, United Kingdom */
#define	FA_LANG 	11		/* Farsi                   */
#define	SU_LANG	12		/* Finnish                 */
#define	CF_LANG	13		/* French, Canada          */
#define	FR_LANG	14		/* French, France          */
#define	GA_LANG	15		/* Galician                */
#define	DE_LANG	16		/* German, Germany         */
#define	SD_LANG	17		/* German, Switzerland     */
#define	GR_LANG	18		/* Greek                   */
#define	HE_LANG	19		/* Hebrew                  */
#define	HU_LANG	20		/* Hungarian               */
#define	IS_LANG	21		/* Icelandic               */
#define	IT_LANG	22		/* Italian                 */
#define	NO_LANG	23		/* Norwegian               */
#define	PL_LANG	24		/* Polish                  */
#define	BR_LANG	25		/* Portuguese, Brazil      */
#define	PO_LANG	26		/* Portuguese, Portugal    */
#define	RU_LANG	27		/* Russian                 */
#define	SL_LANG	28		/* Slovak                  */
#define	ES_LANG	29		/* Spanish                 */
#define	SV_LANG	30		/* Swedish                 */
#define	YK_LANG	31		/* Ukrainian               */
#define	UR_LANG	32		/* Urdu                    */
#define	TK_LANG	33		/* Turkey                  */
#define	JP_LANG	34		/* Japanese						*/
#define	KO_LANG	35		/* Korean                  */
#define	CT_LANG	36		/* Chinese-Traditional     */
#define	CS_LANG	37		/* Chinese-Simplified      */
#define	LA_LANG	38		/* another asian language  */

/* defines for languageID */
/*
WARNING: If adding new languages to LANGUAGE_ID, do NOT alter existing
	ID values. Add new languages to the end of the enumeration, just before
	ID_LANG_UNKNOWN.
*/
typedef enum {
	ID_LANG_AF,		/* Afrikaans */
	ID_LANG_AL,		/* Albanian */
	ID_LANG_AR,		/* Arabic */
	ID_LANG_BG,		/* Bulgarian */
	ID_LANG_CA,		/* Catalan */
	ID_LANG_CH,		/* Swiss */
	ID_LANG_CN_S, 	/* Chinese (Simplified)  */
	ID_LANG_CN_T,	/* Chinese (Traditional) */
	ID_LANG_CS,		/* Czech */
	ID_LANG_DA,		/* Danish */
	ID_LANG_DE,		/* German */
	ID_LANG_DE_CH,	/* German-Switzerland */
	ID_LANG_EN_AU,	/* Australia */
	ID_LANG_EN_UK,	/* U.K. English */
	ID_LANG_EN_US,	/* U.S. English */
	ID_LANG_ES,		/* Spanish, Castilian */
	ID_LANG_ES_EA,	/* Spanish, Latin America */
	ID_LANG_FR,		/* French */
	ID_LANG_FR_CA,	/* Canadian French */
	ID_LANG_GL,		/* Galician */
	ID_LANG_GR,		/* Greek */
	ID_LANG_HE,		/* Hebrew */
	ID_LANG_HR,		/* Croatian */
	ID_LANG_HU,		/* Hungarian */
	ID_LANG_IC,		/* Icelandic */
	ID_LANG_IT,		/* Italian */
	ID_LANG_JP,		/* Japanese */
	ID_LANG_KR,		/* Korean */
	ID_LANG_NE,		/* Dutch (Netherlands) */
	ID_LANG_NO,		/* Norwegian - Bokmal */
	ID_LANG_PL,		/* Polish */
	ID_LANG_PO,		/* Portugese */
	ID_LANG_PO_BR,	/* Brazilian Portugese */
	ID_LANG_RO,		/* Romanian */
	ID_LANG_RU,		/* Russian */
	ID_LANG_SK,		/* Slovak */
	ID_LANG_SL,		/* Slovenian */
	ID_LANG_SU,		/* Finnish */
	ID_LANG_SV,		/* Swedish */
	ID_LANG_TR,		/* Turkish */
	ID_LANG_AF_WIN,		/* Afrikaans */
	ID_LANG_AL_WIN,		/* Albanian */
	ID_LANG_AR_WIN,		/* Arabic */
	ID_LANG_BG_WIN,		/* Bulgarian */
	ID_LANG_CA_WIN,		/* Catalan */
	ID_LANG_CH_WIN,		/* Swiss */
	ID_LANG_CN_S_WIN, 	/* Chinese (Simplified)  */
	ID_LANG_CN_T_WIN,		/* Chinese (Traditional) */
	ID_LANG_CS_WIN,		/* Czech */
	ID_LANG_DA_WIN,		/* Danish */
	ID_LANG_DE_WIN,		/* German */
	ID_LANG_DE_CH_WIN,	/* German-Switzerland */
	ID_LANG_EN_AU_WIN,	/* Australia */
	ID_LANG_EN_UK_WIN,	/* U.K. English */
	ID_LANG_EN_US_WIN,	/* U.S. English */
	ID_LANG_ES_WIN,		/* Spanish, Castilian */
	ID_LANG_ES_EA_WIN,	/* Spanish, Latin America */
	ID_LANG_FR_WIN,		/* French */
	ID_LANG_FR_CA_WIN,	/* Canadian French */
	ID_LANG_GL_WIN,		/* Galician */
	ID_LANG_GR_WIN,		/* Greek */
	ID_LANG_HE_WIN,		/* Hebrew */
	ID_LANG_HR_WIN,		/* Croatian */
	ID_LANG_HU_WIN,		/* Hungarian */
	ID_LANG_IC_WIN,		/* Icelandic */
	ID_LANG_IT_WIN,		/* Italian */
	ID_LANG_JP_WIN,		/* Japanese */
	ID_LANG_KR_WIN,		/* Korean */
	ID_LANG_NE_WIN,		/* Dutch (Netherlands) */
	ID_LANG_NO_WIN,		/* Norwegian - Bokmal */
	ID_LANG_PL_WIN,		/* Polish */
	ID_LANG_PO_WIN,		/* Portugese */
	ID_LANG_PO_BR_WIN,	/* Brazilian Portugese */
	ID_LANG_RO_WIN,		/* Romanian */
	ID_LANG_RU_WIN,		/* Russian */
	ID_LANG_SK_WIN,		/* Slovak */
	ID_LANG_SL_WIN,		/* Slovenian */
	ID_LANG_SU_WIN,		/* Finnish */
	ID_LANG_SV_WIN,		/* Swedish */
	ID_LANG_TR_WIN,		/* Turkish */
	ID_LANG_AF_ISO,		/* Afrikaans */
	ID_LANG_AL_ISO,		/* Albanian */
	ID_LANG_AR_ISO,		/* Arabic */
	ID_LANG_BG_ISO,		/* Bulgarian */
	ID_LANG_CA_ISO,		/* Catalan */
	ID_LANG_CH_ISO,		/* Swiss */
	ID_LANG_CN_S_ISO, 	/* Chinese (Simplified)  */
	ID_LANG_CN_T_ISO,		/* Chinese (Traditional) */
	ID_LANG_CS_ISO,		/* Czech */
	ID_LANG_DA_ISO,		/* Danish */
	ID_LANG_DE_ISO,		/* German */
	ID_LANG_DE_CH_ISO,	/* German-Switzerland */
	ID_LANG_EN_AU_ISO,	/* Australia */
	ID_LANG_EN_UK_ISO,	/* U.K. English */
	ID_LANG_EN_US_ISO,	/* U.S. English */
	ID_LANG_ES_ISO,		/* Spanish, Castilian */
	ID_LANG_ES_EA_ISO,	/* Spanish, Latin America */
	ID_LANG_FR_ISO,		/* French */
	ID_LANG_FR_CA_ISO,	/* Canadian French */
	ID_LANG_GL_ISO,		/* Galician */
	ID_LANG_GR_ISO,		/* Greek */
	ID_LANG_HE_ISO,		/* Hebrew */
	ID_LANG_HR_ISO,		/* Croatian */
	ID_LANG_HU_ISO,		/* Hungarian */
	ID_LANG_IC_ISO,		/* Icelandic */
	ID_LANG_IT_ISO,		/* Italian */
	ID_LANG_JP_ISO,		/* Japanese */
	ID_LANG_KR_ISO,		/* Korean */
	ID_LANG_NE_ISO,		/* Dutch (Netherlands) */
	ID_LANG_NO_ISO,		/* Norwegian - Bokmal */
	ID_LANG_PL_ISO,		/* Polish */
	ID_LANG_PO_ISO,		/* Portugese */
	ID_LANG_PO_BR_ISO,	/* Brazilian Portugese */
	ID_LANG_RO_ISO,		/* Romanian */
	ID_LANG_RU_ISO,		/* Russian */
	ID_LANG_SK_ISO,		/* Slovak */
	ID_LANG_SL_ISO,		/* Slovenian */
	ID_LANG_SU_ISO,		/* Finnish */
	ID_LANG_SV_ISO,		/* Swedish */
	ID_LANG_TR_ISO,		/* Turkish */
	ID_LANG_RU_KOI8,		/* Russian */
	ID_LANG_UNKNOWN
} LANGUAGE_ID;

// We must not use TA_LANG - taiwanses is chinese simplified.
// You should re-use #39 (TA_LANG) -- it is NOT used for Taiwanese...
// #define	TA_LANG	39		/* Taiwanese					*/

/*
	LAST_LANG 	- Used for tables than contain items for each language.
	*_DBCS_LANG - Start and end points for double byte languages.
*/

#define	LAST_LANG 			(LA_LANG+1)		/* last language marker */
#define	FIRST_DBCS_LANG	(JP_LANG)
#define	LAST_DBCS_LANG		(LA_LANG)

/* VISIT: ACHSETS may be wrong.
	check for double wide asian characters */
#define isAsianSet(set) ((set) >= ACHSMIN  && (set) <= ACHSETS)

/**
***	State table information for double character sorting
**/

#define STATE1 1
#define STATE2 2
#define STATE3 3
#define STATE4 4
#define STATE5 5
#define STATE6 6
#define STATE7 7
#define STATE8 8
#define STATE9 9
#define STATE10 10
#define STATE11 11
#define AFTERC 12
#define AFTERH 13
#define AFTERL 14
#define INSTAE 15
#define INSTOE 16
#define INSTSG 17
#define INSTIJ 18
#define WITHAA 19

#define START_COL 12
#define START_ALL (START_COL+1)			/* all US and european */
#define START_DK (START_COL+2)			/* Danish */
#define START_IS (START_COL+3)			/* Icelandic */
#define START_NO (START_COL+4)			/* Norwegian */
#define START_SU (START_COL+5)			/* Finnish */
#define START_SV (START_COL+5)			/* Swedish */
#define START_YK (START_COL+6)			/* Ukrain */
#define START_TK (START_COL+7)			/* Turkish */
#define START_CZ (START_COL+8)			/* Czech */
#define START_SL (START_COL+8)			/* Slovak */

#define FIXUP_AREA_SIZE		24				/* Number of characters to fix up */

#define WPCH_HIMASK	   		0x00FF
#define WPCH_LOMASK	   		0xFF00
#define WPCH_MAX_COMPLEX		5

#define UTOWP60_ENTRIES			1502		/* number of entries in WP_UTOWP60[] */
													/* 2042 with WP user defined values in */
extern FLMBYTE			fwp_c60_max[];
extern FLMUINT16 *	WP60toUni[];
extern FLMUINT16 *	WP60toCpxUni[];
extern FLMUINT16		WP_UTOWP60[][2];

#define MULT60_ENTRIES			154
#define WP60toUni_MAX			15

extern FLMUINT16  WPCH_WP60UNI1[];
extern FLMUINT16  WPCH_WP60UNI2[];
extern FLMUINT16  WPCH_WPUNI3[];
extern FLMUINT16  WPCH_WPUNI4[];
extern FLMUINT16  WPCH_WP60UNI5[];
extern FLMUINT16  WPCH_WPUNI6[];
extern FLMUINT16  WPCH_WPUNI7[];
extern FLMUINT16  WPCH_WP60UNI8[];
extern FLMUINT16  WPCH_WP60UNI9[];
extern FLMUINT16  WPCH_WP60UNI10[];
extern FLMUINT16  WPCH_WP60UNI11[];
extern FLMUINT16  WPCH_WPUNI13[];
extern FLMUINT16  WPCH_WPUNI14[];

extern FLMUINT16  WPCH_CPXTAB1[][5];
extern FLMUINT16  WPCH_CPXGREEK[][5];
extern FLMUINT16  WPCH_CPXHEBREW[][5];
extern FLMUINT16  WPCH_CPXARABIC[][5];
extern FLMUINT16  WPCH_CPXARABIC2[][5];
extern FLMUINT16  WPCH_CPXCYRILLIC[][5];

extern BASE_DIACRIT *	fwp_car60_c[];
extern FLMBYTE				fwp_ml1_cb60[];
extern FLMBYTE				fwp_max_car60_size;

typedef	struct {
	FLMBYTE			key;
	FLMBYTE *		charPtr;
} TBL_B_TO_BP;

extern TBL_B_TO_BP		fwp_col60Tbl[];
extern FLMUINT16			fwp_indexi[];
extern FLMUINT16			fwp_indexj[];
extern FLMUINT16			fwp_valuea[];
extern TBL_B_TO_BP		fwp_HebArabicCol60Tbl[];

FLMBYTE GedTextObjType(
	FLMBYTE		c );

#define GedTextObjType(c) ( \
	(((c & ASCII_CHAR_MASK) == ASCII_CHAR_CODE) \
	 ? ASCII_CHAR_CODE \
	 : (((c & WHITE_SPACE_MASK) == WHITE_SPACE_CODE) \
			? WHITE_SPACE_CODE \
			: (((c & UNK_EQ_1_MASK) == UNK_EQ_1_CODE) \
				 ? UNK_EQ_1_CODE \
				 : (((c & CHAR_SET_MASK) == CHAR_SET_CODE) \
						 ? CHAR_SET_CODE \
						 : c \
					 ) \
				) \
		 ) \
	) \
)

/* Bit patterns for codes in internal TEXT type */
/* 
	ASCII_CHAR - 0x20..0x7E
	CHAR_SET	  - WP Char sets from 1 to 63
	WHITE_SPACE - 
*/
#define ASCII_CHAR_CODE		0x00	/* 0nnnnnnn */
#define ASCII_CHAR_MASK		0x80  /* 10000000 */
#define CHAR_SET_CODE	 	0x80	/* 10nnnnnn */
#define CHAR_SET_MASK	 	0xC0	/* 11000000 */
#define WHITE_SPACE_CODE	0xC0	/* 110nnnnn */
#define WHITE_SPACE_MASK	0xE0	/* 11100000 */

// UNK_GT_255 is an outdated code not part of 3x or newer
#define UNK_GT_255_CODE		0xE0	/* 11100nnn */
#define UNK_GT_255_MASK		0xF8	/* 11111000 */
#define UNK_EQ_1_CODE	 	0xF0	/* 11110nnn */
#define UNK_EQ_1_MASK	 	0xF8	/* 11111000 */

// UNK_LE_255 is an outdated code not part of 3x or newer
#define UNK_LE_255_CODE		0xF8	/* 11111nnn */
#define UNK_LE_255_MASK		0xF8	/* 11111000 */
#define EXT_CHAR_CODE	 	0xE8	/* 11101000 */
#define OEM_CODE			 	0xE9	/* 11101001 */
#define UNICODE_CODE			0xEA	/* 11101010 */

/* The Heart is what Novell NDS is using if a Unicode character */
/* cannot be converted into the current ANSI/ASCII code page    */

#define UNICODE_UNCONVERTABLE_CHAR	0x03		/* Heart */

/* Type codes to go with UNK_GT_255_CODE, UNK_EQ_1_CODE, and */
/* UNK_LE_255_CODE -- maximum of 8 */

#define WP60_TYPE		1
#define NATIVE_TYPE	2

// The HYPHEN's in 3x and 4x don't exist in the database.
#define HARD_HYPHEN		 	3
#define HARD_HYPHEN_EOL	 	4
#define HARD_HYPHEN_EOP	 	5
#define HARD_RETURN		 	7
#define NATIVE_TAB		 	12
#define NATIVE_LINEFEED	 	13

/**
***	Max Buffer sizes
**/

// GWBUG 30,645 - Had 200 and 70 - the 200 for max subcol was not
// enough for the sset and JP characters - computed wrong.
// This crashed the process that was building a key of sset characaters.

#define	MAX_SUBCOL_BUF		(500)	/* (((MAX_KEY_SIZ / 4) * 3 + fluff */
#define	MAX_LOWUP_BUF	  	(150) /* ((MAX_KEY_SIZ -(MAX_KEY_SIZ / 8)) / 8)*2*/


/*
***	Simple portable UWORD-->BYTE<--UWORD conversions
***	Word to High/Low byte convertsions - WP set # is high byte value
***
***	If there are portability problems then put #ifdef machine
*** or find a macro within swp.h or toolkit to use.
*** The definition works great for LITENDIN or other machines.
**/

#define F_GETLOWBYTE(w)		((FLMBYTE)(w))
#define F_GETHIGHBYTE(w)	((FLMBYTE)((w) >> 8))

/* FLAGS */
#define	HAD_SUB_COLLATION	0x01	/* Set if had sub-collating values-diacritics*/
#define	HAD_LOWER_CASE		0x02	/* Set if you hit a lowercase character */


#define	COMPOUND_MARKER		0x02		/* Compound key marker between each piece */
#define	END_COMPOUND_MARKER	0x01	/* Last of all compound markers - for post*/
																	/* This must be < CM because of multiple  */
																	/* COMPOUND_MARKERS that could appear     */
#define	NULL_KEY_MARKER		0x03
#define	COLL_FIRST_SUBSTRING	0x03	/* First substring marker */

#define	COLL_MARKER 		0x04		/* Marks place of sub-collation */

#define	SC_LOWER				0x00		/* Only lowercase characters exist */
#define	SC_MIXED				0x01		/* Lower/uppercase flags follow in next byte*/
#define	SC_UPPER				0x02		/* Only upper characters exist */
#define	SC_SUB_COL			0x03		/* Sub-collation follows (diacritics|extCh) */

#define	UNK_UNICODE_CODE	0xFFFE	/* Used for collation */

/* In the future a SUB_COL-1 value could mean leading information */

// Leave room for stuff before truncated value.
#define	COLL_TRUNCATED		0x0C		/* This key piece has been truncated from original*/

/* Max. opcode for any collation markers - was 7 before Nov 98 */
#define	MAX_COL_OPCODE		COLL_TRUNCATED	


#define	BYTES_IN_BITS(bits)	((bits + 7) >> 3)	/* Computes # of bytes */

#define	TEST1BIT( buf, bPos)	((((buf)[ (bPos) >> 3]) >> (7 - ((bPos) & 7))) & 1)
#define	GET1BIT( buf, bPos)	((((buf)[ (bPos) >> 3]) >> (7 - ((bPos) & 7))) & 1)

#define GETnBITS( n, bf, bit) \
(((unsigned int)( \
   ((unsigned char)bf[ (bit) >> 3] << 8)/* append high bits (byte 1) to ...	*/\
   | \
   (unsigned char)bf[ ((bit) >> 3) + 1]	/* ... overflow bits in 2nd byte		*/\
  ) >> (16 - (n) - ((bit) & 7))	  			/* reposition to low end of value		*/\
 ) & ((1 << (n)) - 1)				  			/* mask off high bits								*/\
)											  			/* return value */

#define	SET_BIT(buf,bPos)		((buf)[(bPos) >> 3] |=  (FLMBYTE)((1 << (7 - ((bPos) & 7)))))

#define	RESET_BIT(buf,bPos)	((buf)[(bPos) >> 3] &= (FLMBYTE)(~(1 << (7 - ((bPos) & 7)))))

#define	SETnBITS( n, bf, bit, v) \
	{	(bf)[ (bit) >> 3] |= 		  	/* 1st byte */\
			(FLMBYTE)(((v) << (8 - (n))) 		  	/* align to bit 0 */\
			>> \
			((bit) & 7)); 				  	/* re-align to actual bit position */\
		(bf)[ ((bit) >> 3) + 1] = 		/* 2nd byte */\
			(FLMBYTE)((v) \
			<< \
			(16 - (n) - ((bit) & 7))); 	/* align spill-over bits */\
	}


/* Defines for numeric collation/uncollation */

#define SIG_POS					0x80
#define COLLATED_DIGIT_OFFSET	0x05
#define COLLATED_NUM_EXP_BIAS 64
#define MIN_7BIT_EXP				0x08
#define MAX_7BIT_EXP				0x78

RCODE FTextToColStr(
	const FLMBYTE *	str,					/* Points to the internal TEXT string */
	FLMUINT				uiStrLen,			/* Length of the internal TEXT string */
	FLMBYTE *			colStr,				/* Output collated string */
	FLMUINT *			puiColStrLen,		/* Collated string length return value */
													/* Input value is MAX num of bytes in buffer*/
	FLMUINT				uiUppercaseFlag,	/* If set then treat string like uppercase */
	FLMUINT *			puiCollationLen,	/* Returns the collation bytes length */
	FLMUINT *			puiCaseLen,			/* Returns length of case area */
	FLMUINT				uiWPLang,			/* WP Language using Flaim language number */
	FLMUINT				uiCharLimit,
	FLMBOOL				bFirstSubstring,
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL *			pbDataTruncated);

FLMUINT FColStrToText( 				/* Returns strlen of null term output string */
	FLMBYTE *	ColStr, 				/* Points to the FLAIM collated string */
	FLMUINT *	puiColStrLenRV,	/* FLAIM Collated string length return value */
	FLMBYTE *	textStr,				/* Output text string buffer */
	FLMUINT		uiWPLang,			/* WP Language using Flaim language number */
	FLMBYTE *	postBuf,				/* Lower/upper POST buffer or NULL */
	FLMUINT *	postBytesRV,		/* Returns number bytes used in post buffer */
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbFirstSubstring);
	
RCODE AsiaFlmTextToColStr(
	const FLMBYTE *	Str,					/* Points to the internal TEXT string */
	FLMUINT 				uiStrLen,			/* Length of the internal TEXT string */
	FLMBYTE *			ColStr,				/* Output collated string */
	FLMUINT *			puiColStrLenRV,	/* Collated string length return value */
													/* Input value is MAX num of bytes in buffer*/
	FLMUINT 				uiUppercaseFlag,	/* Set if to convert to uppercase */
	FLMUINT *			puiCollationLen,	/* Returns the collation bytes length */
	FLMUINT *			puiCaseLenRV,		/* Returns length of case bytes */
	FLMUINT				uiCharLimit,
	FLMBOOL				bFirstSubstring,
	FLMBOOL *			pbDataTruncated);

FLMUINT AsiaConvertColStr(
	FLMBYTE *	CollatedStr,			/* Points to the Flaim collated string */
	FLMUINT *	puiCollatedStrLenRV,	/* Length of the Flaim collated string */
	FLMBYTE *	WordStr,			  		/* Output string to build - WP word string */
	FLMBOOL *	pbDataTruncated,
	FLMBOOL *	pbFirstSubstring);

FLMUINT AsiaParseSubCol(
	FLMBYTE *	WordStr,					/* Existing word string to modify */
	FLMUINT * 	puiWordStrLen,			/* Wordstring length in bytes */
	FLMBYTE *	SubColBuf);				/* Diacritic values in 5 bit sets */

FLMUINT AsiaParseCase(
	FLMBYTE *	WordStr,			  		/* Existing word string to modify */
	FLMUINT *	uiWordStrLenRV,	  	/* Length of the WordString in bytes */
	FLMBYTE *	pCaseBits);	  			/* Lower/upper case bit string */

typedef	struct
{
	FLMBYTE		ByteValue;
	FLMUINT16	WordValue;
} BYTE_WORD_TBL;

FLMUINT16 fwpCh6Upper(
	FLMUINT16			ui16WpChar);

FLMBOOL fwpIsUpper(	
	FLMUINT16			ui16Char);
			
FLMUINT16 fwpCh6Lower(
	FLMUINT16			ui16WpChar);
			
FLMBOOL fwpCh6Brkcar(	
	FLMUINT16			ui16WpChar, 
	FLMUINT16 *			pui16BaseChar,
	FLMUINT16 *			pui16DiacriticChar);

FLMBOOL fwpCh6Cmbcar(	
	FLMUINT16 *			pui16WpChar, 
	FLMUINT16			ui16BaseChar,
	FLMINT16				i16DiacriticChar);

FLMUINT16 fwpGetCollation(
	FLMUINT16			ui16WpChr,
	FLMUINT				uiLang);

FLMUINT16 fwpCheckDoubleCollation(
	FLMUINT16 *			pui16WpChar,
	FLMBOOL *			pbTwoIntoOne,
	const FLMBYTE **	ppucInputStr,
	FLMUINT				uiLanguage);

FLMUINT16 fwpAsiaGetCollation(
	FLMUINT16			WpChar,
	FLMUINT16			NextChar,
	FLMUINT16			PrevColValue,
	FLMUINT16 *			ColValueRV,
	FLMUINT16 *			SubColValRV,
	FLMBYTE *			CaseFlagsRV,
	FLMUINT16			UppercaseFlag);

FLMUINT16 ZenToHankaku( 
	FLMUINT16			WpChar,
	FLMUINT16 * 		DakutenOrHandakutenRV);

FLMUINT16 HanToZenkaku(
	FLMUINT16			WpChar,
	FLMUINT16			NextChar,
	FLMUINT16 *			ZenkakuRV );

#include "fpackoff.h"

#endif
