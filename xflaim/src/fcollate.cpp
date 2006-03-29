//------------------------------------------------------------------------------
// Desc:	Routines for building collation keys
//
// Tabs:	3
//
//		Copyright (c) 1993-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fcollate.cpp 3111 2006-01-19 13:10:50 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

#define shiftN(data,size,distance) \
		f_memmove((FLMBYTE *)(data) + (FLMINT)(distance), \
		(FLMBYTE *)(data), (size_t)(size))

typedef struct
{
	FLMBYTE		base;
	FLMBYTE		diacrit;
} BASE_DIACRIT_TABLE;

typedef struct
{
	FLMUINT16					char_count;		// # of characters in table
	FLMUINT16					start_char;		// start char.
	BASE_DIACRIT_TABLE *		table;

} BASE_DIACRIT;

typedef struct
{
	FLMBYTE			key;			// character key to search on
	FLMBYTE *		charPtr;		// character pointer for matched key
} TBL_B_TO_BP;

typedef struct
{
	FLMBYTE		ByteValue;
	FLMUINT16	WordValue;
} BYTE_WORD_TBL;

// Static functions

FSTATIC RCODE flmColStr2WPStr(
	const FLMBYTE *	pucColStr,
	FLMUINT				uiColStrLen,
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiLang,
	FLMUINT *			puiUnconvChars,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbFirstSubstring);

FSTATIC RCODE flmWPCmbSubColBuf(
	FLMBYTE *			pucWPStr,
	FLMUINT *			uiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,
	FLMBOOL				bHebrewArabic,
	FLMUINT *			puiSubColBitPos);

FSTATIC FLMUINT flmWPToMixed(
	FLMBYTE *			pucWPStr,
	FLMUINT				uiWPStrLen,
	const FLMBYTE *	pucLowUpBitStr,
	FLMUINT				uiLang);

FSTATIC FLMUINT16 flmWPZenToHankaku(
	FLMUINT16			ui16WpChar,
	FLMUINT16 * 		pui16DakutenOrHandakuten);

FSTATIC FLMUINT16 flmWPHanToZenkaku(
	FLMUINT16			ui16WpChar,
	FLMUINT16			ui16NextWpChar,
	FLMUINT16 *			pui16Zenkaku);

FSTATIC FLMBOOL flmWPCmbcar(
	FLMUINT16 *			pui16WpChar,
	FLMUINT16			ui16BaseChar,
	FLMINT16				ui16DiacriticChar);

FSTATIC RCODE flmAsiaColStr2WPStr(
	const FLMBYTE *	pucColStr,
	FLMUINT				uiColStrLen,
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT *			puiUnconvChars,
	FLMBOOL *			pbDataTruncated,
	FLMBOOL *			pbFirstSubstring);

FSTATIC RCODE flmAsiaParseSubCol(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,
	FLMUINT *			puiSubColBitPos);

FSTATIC RCODE flmAsiaParseCase(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucCaseBits,
	FLMUINT *			puiColBytesProcessed);

#define MAX_COL_OPCODE			COLL_TRUNCATED	// Max. opcode for any collation markers
#define WP_MAX_CAR60_SIZE		NCHSETS
#define SET_CASE_BIT				0x01
#define SET_KATAKANA_BIT		0x01
#define SET_WIDTH_BIT			0x02
#define COLS_ASIAN_MARKS		0x140
#define COLS_ASIAN_MARK_VAL	0x40			// Without 0x100

#define ASCTBLLEN					95
#define MNTBLLEN					219
#define SYMTBLLEN					9
#define GRKTBLLEN					219
#define CYRLTBLLEN				200
#define HEBTBL1LEN				27
#define HEBTBL2LEN				35
#define AR1TBLLEN					158
#define AR2TBLLEN					179

#define Upper_JP_A				0x2520
#define Upper_JP_Z				0x2539
#define Upper_KR_A				0x5420
#define Upper_KR_Z				0x5439
#define Upper_CS_A				0x82FC
#define Upper_CS_Z				0x8316
#define Upper_CT_A				0xA625
#define Upper_CT_Z				0xA63E

#define Lower_JP_a				0x2540
#define Lower_JP_z				0x2559
#define Lower_KR_a				0x5440
#define Lower_KR_z				0x5459
#define Lower_CS_a				0x82DC
#define Lower_CS_z				0x82F5
#define Lower_CT_a				0xA60B
#define Lower_CT_z				0xA624

// # of characters in each character set.
// CHANGING ANY OF THESE DEFINES WILL CAUSE BUGS!

#define ASC_N						95
#define ML1_N						242
#define ML2_N						145
#define BOX_N						88
#define TYP_N						103
#define ICN_N						255
#define MTH_N						238
#define MTX_N						229
#define GRK_N						219
#define HEB_N						123
#define CYR_N						250
#define KAN_N						63
#define USR_N						255
#define ARB_N						196
#define ARS_N						220

// TOTAL:	1447 WP + 255 User Characters

#define	C_N 						ASC_N + ML1_N + ML2_N + BOX_N +\
										MTH_N + MTX_N + TYP_N + ICN_N +\
										GRK_N + HEB_N + CYR_N + KAN_N +\
										USR_N + ARB_N + ARS_N

// State table constants for double character sorting

#define STATE1						1
#define STATE2						2
#define STATE3						3
#define STATE4						4
#define STATE5						5
#define STATE6						6
#define STATE7						7
#define STATE8						8
#define STATE9						9
#define STATE10					10
#define STATE11					11
#define AFTERC						12
#define AFTERH						13
#define AFTERL						14
#define INSTAE						15
#define INSTOE						16
#define INSTSG						17
#define INSTIJ						18
#define WITHAA						19

#define START_COL					12
#define START_ALL					(START_COL + 1)	// all US and european
#define START_DK					(START_COL + 2)	// Danish
#define START_IS					(START_COL + 3)	// Icelandic
#define START_NO					(START_COL + 4)	// Norwegian
#define START_SU					(START_COL + 5)	// Finnish
#define START_SV					(START_COL + 5)	// Swedish
#define START_YK					(START_COL + 6)	// Ukrain
#define START_TK					(START_COL + 7)	// Turkish
#define START_CZ					(START_COL + 8)	// Czech
#define START_SL					(START_COL + 8)	// Slovak

#define FIXUP_AREA_SIZE			24						// Number of characters to fix up

// Collation tables

/****************************************************************************
Desc:		Table of # of characters in each character set
****************************************************************************/
FLMBYTE fwp_c60_max[] =
{
	ASC_N,	// ascii
	ML1_N,	// multinational 1
	ML2_N,	// multinational 2
	BOX_N,	// line draw
	TYP_N,	// typographic
	ICN_N,	// icons
	MTH_N,	// math
	MTX_N,	// math extension
	GRK_N,	// Greek
	HEB_N,	// Hebrew
	CYR_N,	// Cyrillic - Russian
	KAN_N,	// Kana
	USR_N,	// user
	ARB_N,	// Arabic
	ARS_N,	// Arabic Script
};

/****************************************************************************
Desc:		Base character location table
				Bit mapped table.	(1) - corresponding base char is in same
				set as combined
				(0) - corresponding base char is in ascii set

Notes:		In the following table, the bits are numbered from left
				to right relative to each individual byte.
						EX. 00000000b   ;0-7
						bit#   01234567
****************************************************************************/
FLMBYTE fwp_ml1_cb60[] =
{
	0x00,    // 0-7
	0x00,    // 8-15
	0x00,    // 16-23
	0x00,    // 24-31
	0x00,    // 32-39
	0x00,    // 40-47
	0x55,    // 48-55
	0x00,    // 56-63
	0x00,    // 64-71
	0x00,    // 72-79
	0x00,    // 80-87
	0x00,    // 88-95
	0x00,    // 96-103
	0x00,    // 104-111
	0x00,    // 112-119
	0x00,    // 120-127
	0x14,    // 128-135
	0x44,    // 136-143
	0x00,    // 144-151
	0x00,    // 152-159
	0x00,    // 160-167
	0x00,    // 168-175
	0x00,    // 176-183
	0x00,    // 184-191
	0x00,    // 192-199
	0x00,    // 200-207
	0x00,    // 208-215
	0x00,    // 216-223
	0x00,    // 224-231
	0x04,    // 232-239
	0x00,    // 240-241
};

/****************************************************************************
Desc:		Format of index:
				2 words before = count.
				word before = start character.
				db code for base char.
				db code for diacritic
Notes:	Diacritical char is always in same set as composed char
			base is in same set if other table indicates, else in ASCII
****************************************************************************/
BASE_DIACRIT_TABLE fwp_ml1c_table[] =
{
	{'A',acute},
	{'a',acute},
	{'A',circum},
	{'a',circum},
	{'A',umlaut},
	{'a',umlaut},
	{'A',grave},
	{'a',grave},
	{'A',ring},
	{'a',ring},
	{0xff,0xff},      // no AE diagraph
	{0xff,0xff},      // no ae diagraph
	{'C',cedilla},
	{'c',cedilla},
	{'E',acute},
	{'e',acute},
	{'E',circum},
	{'e',circum},
	{'E',umlaut},
	{'e',umlaut},
	{'E',grave},
	{'e',grave},
	{'I',acute},
	{dotlesi,acute},
	{'I',circum},
	{dotlesi,circum},
	{'I',umlaut},
	{dotlesi,umlaut},
	{'I',grave},
	{dotlesi,grave},
	{'N',tilde},
	{'n',tilde},
	{'O',acute},
	{'o',acute},
	{'O',circum},
	{'o',circum},
	{'O',umlaut},
	{'o',umlaut},
	{'O',grave},
	{'o',grave},
	{'U',acute},
	{'u',acute},
	{'U',circum},
	{'u',circum},
	{'U',umlaut},
	{'u',umlaut},
	{'U',grave},
	{'u',grave},
	{'Y',umlaut},
	{'y',umlaut},
	{'A',tilde},
	{'a',tilde},
	{'D',crossb},
	{'d',crossb},
	{'O',slash},
	{'o',slash},
	{'O',tilde},
	{'o',tilde},
	{'Y',acute},
	{'y',acute},
	{0xff,0xff},		// no eth
	{0xff,0xff},		// no eth
	{0xff,0xff},		// no Thorn
	{0xff,0xff},		// no Thorn
	{'A',breve},
	{'a',breve},
	{'A',macron},
	{'a',macron},
	{'A',ogonek},
	{'a',ogonek},
	{'C',acute},
	{'c',acute},
	{'C',caron},
	{'c',caron},
	{'C',circum},
	{'c',circum},
	{'C',dota},
	{'c',dota},
	{'D',caron},
	{'d',caron},
	{'E',caron},
	{'e',caron},
	{'E',dota},
	{'e',dota},
	{'E',macron},
	{'e',macron},
	{'E',ogonek},
	{'e',ogonek},
	{'G',acute},
	{'g',acute},
	{'G',breve},
	{'g',breve},
	{'G',caron},
	{'g',caron},
	{'G',cedilla},
	{'g',aposab},
	{'G',circum},
	{'g',circum},
	{'G',dota},
	{'g',dota},
	{'H',circum},
	{'h',circum},
	{'H',crossb},
	{'h',crossb},
	{'I',dota},
	{dotlesi,dota},
	{'I',macron},
	{dotlesi,macron},
	{'I',ogonek},
	{'i',ogonek},
	{'I',tilde},
	{dotlesi,tilde},
	{0xff,0xff},		// no IJ digraph
	{0xff,0xff},		// no ij digraph
	{'J',circum},
	{dotlesj,circum},
	{'K',cedilla},
	{'k',cedilla},
	{'L',acute},
	{'l',acute},
	{'L',caron},
	{'l',caron},
	{'L',cedilla},
	{'l',cedilla},
	{'L',centerd},
	{'l',centerd},
	{'L',stroke},
	{'l',stroke},
	{'N',acute},
	{'n',acute},
	{'N',aposba},
	{'n',aposba},
	{'N',caron},
	{'n',caron},
	{'N',cedilla},
	{'n',cedilla},
	{'O',dacute},
	{'o',dacute},
	{'O',macron},
	{'o',macron},
	{0xff,0xff},		// OE digraph
	{0xff,0xff},		// oe digraph
	{'R',acute},
	{'r',acute},
	{'R',caron},
	{'r',caron},
	{'R',cedilla},
	{'r',cedilla},
	{'S',acute},
	{'s',acute},
	{'S',caron},
	{'s',caron},
	{'S',cedilla},
	{'s',cedilla},
	{'S',circum},
	{'s',circum},
	{'T',caron},
	{'t',caron},
	{'T',cedilla},
	{'t',cedilla},
	{'T',crossb},
	{'t',crossb},
	{'U',breve},
	{'u',breve},
	{'U',dacute},
	{'u',dacute},
	{'U',macron},
	{'u',macron},
	{'U',ogonek},
	{'u',ogonek},
	{'U',ring},
	{'u',ring},
	{'U',tilde},
	{'u',tilde},
	{'W',circum},
	{'w',circum},
	{'Y',circum},
	{'y',circum},
	{'Z',acute},
	{'z',acute},
	{'Z',caron},
	{'z',caron},
	{'Z',dota},
	{'z',dota},
	{0xff,0xff},		// no Eng
	{0xff,0xff},		// no eng
	{'D',macron},
	{'d',macron},
	{'L',macron},
	{'l',macron},
	{'N',macron},
	{'n',macron},
	{'R',grave},
	{'r',grave},
	{'S',macron},
	{'s',macron},
	{'T',macron},
	{'t',macron},
	{'Y',breve},
	{'y',breve},
	{'Y',grave},
	{'y',grave},
	{'D',aposbes},
	{'d',aposbes},
	{'O',aposbes},
	{'o',aposbes},
	{'U',aposbes},
	{'u',aposbes},
	{'E',breve},
	{'e',breve},
	{'I',breve},
	{dotlesi,breve},
	{0xff,0xff},		// no dotless I
	{0xff,0xff},		// no dotless i
	{'O',breve},
	{'o',breve}
};

/****************************************************************************
Desc:
****************************************************************************/
BASE_DIACRIT fwp_ml1c =
{
	216,    	// # of characters in table
	26,      // start char
	fwp_ml1c_table,
};

/****************************************************************************
Desc:		Format of index:
				2 words before = count.
				word before = start character.
				db	code for base char.
				db	code for diacritic
Notes:	Diacritical char is always in same set as composed char
			base is in same set
****************************************************************************/
static BASE_DIACRIT_TABLE fwp_grk_c_table[] =
{
	{  0, ghprime },					// ALPHA High Prime
	{  1, gacute },					// alpha acute
	{ 10, ghprime },					// EPSILON High Prime
	{ 11, gacute },					// epsilon Acute
	{ 14, ghprime },					// ETA High Prime
	{ 15, gacute },					// eta Acute
	{ 18, ghprime },					// IOTA High Prime
	{ 19, gacute },					// iota Acute
	{ 0xFF, 0xFF },					// IOTA Diaeresis
	{ 19, gdia },						// iota Diaeresis
	{ 30, ghprime },					// OMICRON High Prime
	{ 31, gacute },					// omicron Acute
	{ 42, ghprime },					// UPSILON High Prime
	{ 43, gacute },					// upsilon Acute
	{ 0xFF, 0xFF }, 					// UPSILON Diaeresis
	{ 43,gdia }, 						// upsilon Diaeresis
	{ 50,ghprime }, 					// OMEGA High Prime
	{ 51,gacute }, 					// omega Acute
	{ 0xFF, 0xFF },					// epsilon (Variant)
	{ 0xFF, 0xFF },					// theta (Variant)
	{ 0xFF, 0xFF },					// kappa (Variant)
	{ 0xFF, 0xFF },					// pi (Variant)
	{ 0xFF, 0xFF },					// rho (Variant)
	{ 0xFF, 0xFF },					// sigma (Variant)
	{ 0xFF, 0xFF },					// UPSILON (Variant)
	{ 0xFF, 0xFF },					// phi (Variant)
	{ 0xFF, 0xFF },					// omega (Variant)
	{ 0xFF, 0xFF },					// Greek Question Mark
	{ 0xFF, 0xFF },					// Greek Semicolon
	{ 0xFF, 0xFF },					// High Prime
	{ 0xFF, 0xFF },					// Low Prime
	{ 0xFF, 0xFF },					// Acute (Greek)
	{ 0xFF, 0xFF },					// Diaeresis (Greek)
	{ gacute,gdia },					// Acute Diaeresis
	{ ggrave, gdia },					// Grave Diaeresis
	{ 0xFF, 0xFF },					// Grave (Greek)
	{ 0xFF, 0xFF },					// Circumflex (Greek)
	{ 0xFF, 0xFF },					// Smooth Breathing
	{ 0xFF, 0xFF },					// Rough Breathing
	{ 0xFF, 0xFF },					// Iota Subscript
	{ gsmooth, gacute },				// Smooth Breathing Acute
	{ grough, gacute },				// Rough Breathing Acute
	{ gsmooth, ggrave },				// Smooth Breathing Grave
	{ grough, ggrave },				// Rough Breathing Grave
	{ gsmooth, gcircm },				// Smooth Breathing Circumflex
	{ grough, gcircm },				// Rough Breathing Circumflex
	{ gacute, giota },				// Acute w/Iota Subscript
	{ ggrave, giota },				// Grave w/Iota Subscript
	{ gcircm, giota },				// Circumflex w/Iota Subscript
	{ gsmooth, giota },				// Smooth Breathing w/Iota Subscript
	{ grough, giota },				// Rough Breathing w/Iota Subscript
	{ gsmact, giota },				// Smooth Breathing Acute w/Iota Subscript
	{ grgact, giota },				// Rough Breathing Acute w/Iota Subscript
	{ gsmgrv, giota },				// Smooth Breathing Grave w/Iota Subscript
	{ grggrv, giota },				// Rough Breathing Grave w/Iota Subscript
	{ gsmcir, giota },				// Smooth Breathing Circumflex w/Iota Sub
	{ grgcir, giota },				// Rough Breathing Circumflex w/Iota Sub
	{ 1, ggrave },						// alpha Grave
	{ 1, gcircm },						// alpha Circumflex
	{ 1, giota	 },					// alpha w/Iota
	{ 1, gactio },						// alpha Acute w/Iota
	{ 1, ggrvio },						// alpha Grave w/Iota
	{ 1, gcirio },						// alpha Circumflex w/Iota
	{ 1, gsmooth },					// alpha Smooth
	{ 1, gsmact },						// alpha Smooth Acute
	{ 1, gsmgrv },						// alpha Smooth Grave
	{ 1, gsmcir },						// alpha Smooth Circumflex
	{ 1, gsmio	 },					// alpha Smooth w/Iota
	{ 1, gsmaio },						// alpha Smooth Acute w/Iota
	{ 1, gsmgvio },					// alpha Smooth Grave w/Iota
	{ 1, gsmcio },						// alpha Smooth Circumflex w/Iota
	{ 1, grough },						// alpha Rough
	{ 1, grgact },						// alpha Rough Acute
	{ 1, grggrv },						// alpha Rough Grave
	{ 1, grgcir },						// alpha Rough Circumflex
	{ 1, grgio	 },					// alpha Rough w/Iota
	{ 1, grgaio },						// alpha Rough Acute w/Iota
	{ 1, grggvio },					// alpha Rough Grave w/Iota
	{ 1, grgcio },						// alpha Rough Circumflex w/Iota
	{ 11, ggrave },					// epsilon Grave
	{ 11, gsmooth },					// epsilon Smooth
	{ 11, gsmact },					// epsilon Smooth Acute
	{ 11, gsmgrv },					// epsilon Smooth Grave
	{ 11, grough },					// epsilon Rough
	{ 11, grgact },					// epsilon Rough Acute
	{ 11, grggrv },					// epsilon Rough Grave
	{ 15, ggrave },					// eta Grave
	{ 15, gcircm },					// eta Circumflex
	{ 15, giota },						// eta w/Iota
	{ 15, gactio },					// eta Acute w/Iota
	{ 15, ggrvio },					// eta Grave w/Iota
	{ 15, gcirio },					// eta Circumflex w/Iota
	{ 15, gsmooth },					// eta Smooth
	{ 15, gsmact },					// eta Smooth Acute
	{ 15, gsmgrv },					// eta Smooth Grave
	{ 15, gsmcir },					// eta Smooth Circumflex
	{ 15, gsmio },						// eta Smooth w/Iota
	{ 15, gsmaio },					// eta Smooth Acute w/Iota
	{ 15, gsmgvio },					// eta Smooth Grave w/Iota
	{ 15, gsmcio },					// eta Smooth Circumflex w/Iota
	{ 15, grough },					// eta Rough
	{ 15, grgact },					// eta Rough Acute
	{ 15, grggrv },					// eta Rough Grave
	{ 15, grgcir },					// eta Rough Circumflex
	{ 15, grgio },						// eta Rough w/Iota
	{ 15, grgaio },					// eta Rough Acute w/Iota
	{ 15, grggvio },					// eta Rough Grave w/Iota
	{ 15, grgcio },					// eta Rough Circumflex w/Iota
	{ 19, ggrave },					// iota Grave
	{ 19, gcircm },					// iota Circumflex
	{ 19, gactdia },					// iota Acute Diaeresis
	{ 19, ggrvdia },					// iota Grave Diaeresis
	{ 19, gsmooth },					// iota Smooth
	{ 19, gsmact },					// iota Smooth Acute
	{ 19, gsmgrv },					// iota Smooth Grave
	{ 19, gsmcir },					// iota Smooth Circumflex
	{ 19, grough },					// iota Rough
	{ 19, grgact },					// iota Rough Acute
	{ 19, grggrv },					// iota Rough Grave
	{ 19, grgcir },					// iota Rough Circumflex
	{ 31, ggrave },					// omicron Grave
	{ 31, gsmooth },					// omicron Smooth
	{ 31, gsmact },					// omicron Smooth Acute
	{ 31, gsmgrv },					// omicron Smooth Grave
	{ 31, grough },					// omicron Rough
	{ 31, grgact },					// omicron Rough Acute
	{ 31, grggrv },					// omicron Rough Grave
	{ 0xFF, 0xFF },					// rho rough
	{ 0xFF, 0xFF },					// rho smooth
	{ 43, ggrave },					// upsilon Grave
	{ 43, gcircm },					// upsilon Circumflex
	{ 43, gactdia },					// upsilon Acute Diaeresis
	{ 43, ggrvdia },					// upsilon Grave Diaeresis
	{ 43, gsmooth },					// upsilon Smooth
	{ 43, gsmact },					// upsilon Smooth Acute
	{ 43, gsmgrv },					// upsilon Smooth Grave
	{ 43, gsmcir },					// upsilon Smooth Circumflex
	{ 43, grough },					// upsilon Rough
	{ 43, grgact },					// upsilon Rough Acute
	{ 43, grggrv },					// upsilon Rough Grave
	{ 43, grgcir },					// upsilon Rough Circumflex
	{ 51, ggrave },					// omega Grave
	{ 51, gcircm },					// omega Circumflex
	{ 51, giota },						// omega w/Iota
	{ 51, gactio },					// omega Acute w/Iota
	{ 51, ggrvio },					// omega Grave w/Iota
	{ 51, gcirio },					// omega Circumflex w/Iota
	{ 51, gsmooth },					// omega Smooth
	{ 51, gsmact },					// omega Smooth Acute
	{ 51, gsmgrv },					// omega Smooth Grave
	{ 51, gsmcir },					// omega Smooth Circumflex
	{ 51, gsmio },						// omega Smooth w/Iota
	{ 51, gsmaio },					// omega Smooth Acute w/Iota
	{ 51, gsmgvio },					// omega Smooth Grave w/Iota
	{ 51, gsmcio },					// omega Smooth Circumflex w/Iota
	{ 51, grough },					// omega Rough
	{ 51, grgact },					// omega Rough Acute
	{ 51, grggrv },					// omega Rough Grave
	{ 51, grgcir },					// omega Rough Circumflex
	{ 51, grgio },						// omega Rough w/Iota
	{ 51, grgaio },					// omega Rough Acute w/Iota
	{ 51, grggvio },					// omega Rough Grave w/Iota
	{ 51, grgcio}						// omega Rough Circumflex w/Iota
};

/****************************************************************************
Desc:
****************************************************************************/
static BASE_DIACRIT fwp_grk_c =
{
	163,	// # of characters in table.
	52,	// start char.
	fwp_grk_c_table
};

/****************************************************************************
Desc:		Format of index:
				2 words before = count.
				word before = start character.
				db	code for base char.
				db code for diacritic
Notes:	Diacritical char is always in same set as composed char
			base is in same set
****************************************************************************/
static BASE_DIACRIT_TABLE  fwp_rus_c_table[] =
{
	{ 14, 204 },					// ZHE with right descender
	{ 15, 204 },					// zhe with right descender
	{ 0xFF, 0xFF},					// DZE
	{ 0xFF, 0xFF},					// dze
	{ 0xFF, 0xFF},					// Z
	{ 0xFF, 0xFF},					// z
	{ 18, 206 },					// II with macron
	{ 19, 206},						// ii with macron
	{ 0xFF, 0xFF},					// I
	{ 0xFF, 0xFF},					// i
	{ 0xFF, 0xFF},					// YI
	{ 0xFF, 0xFF},					// yi
	{ 0xFF, 0xFF},					// I ligature
	{ 0xFF, 0xFF},					// i ligature
	{ 0xFF, 0xFF},					// JE
	{ 0xFF, 0xFF},					// je
	{ 0xFF, 0xFF},					// KJE
	{ 0xFF, 0xFF},					// kje
	{ 22, 204},						// KA with right descender
	{ 23, 204},						// ka with right descender
	{ 22, 205 },					// KA ogonek
	{ 23, 205 },					// ka ogonek
	{ 0xFF, 0xFF},					// KA vertical bar
	{ 0xFF, 0xFF},					// ka vertical bar
	{ 0xFF, 0xFF},					// LJE
	{ 0xFF, 0xFF},					// lje
	{ 28, 204 },					// EN with right descender
	{ 29, 204 },					// en with right descender
	{ 0xFF, 0xFF},					// NJE
	{ 0xFF, 0xFF},					// nje
	{ 0xFF, 0xFF},					// ROUND OMEGA
	{ 0xFF, 0xFF},					// round omega
	{ 0xFF, 0xFF},					// OMEGA
	{ 0xFF, 0xFF},					// omega
	{ 0xFF, 0xFF},					// TSHE
	{ 0xFF, 0xFF},					// tshe
	{ 0xFF, 0xFF},					// SHORT U
	{ 0xFF, 0xFF},					// short u
	{ 40, 206},						// U with macron
	{ 41, 206 },					// u with macron
	{ 0xFF, 0xFF},					// STRAIGHT U
	{ 0xFF, 0xFF},					// straight u
	{ 0xFF, 0xFF},					// STRAIGHT U BAR
	{ 0xFF, 0xFF},					// straight u bar
	{ 0xFF, 0xFF},					// OU ligature
	{ 0xFF, 0xFF},					// ou ligature
	{ 44, 204 },					// KHA with right descender
	{ 45, 204 },					// kha with right descender
	{ 44, 205 },					// KHA ogonek
	{ 45, 205 },					// kha ogonek
	{ 0xFF, 0xFF},					// H
	{ 0xFF, 0xFF},					// h
	{ 0xFF, 0xFF},					// OMEGA titlo
	{ 0xFF, 0xFF},					// omega titlo
	{ 0xFF, 0xFF},					// DZHE
	{ 0xFF, 0xFF},					// dzhe
	{ 48, 204 },					// CHE with right descender
	{ 49, 204 },					// che with right descender
	{ 0xFF, 0xFF},					// CHE vertical bar
	{ 0xFF, 0xFF},					// che vertical bar
	{ 0xFF, 0xFF},					// SHCHA (variant)
	{ 0xFF, 0xFF},					// shcha (variant)
	{ 0xFF, 0xFF},					// YAT
	{ 0xFF, 0xFF},					// yat
	{ 0xFF, 0xFF},					// YUS BOLSHOI
	{ 0xFF, 0xFF},					// yus bolshoi
	{ 0xFF, 0xFF},					// BIG MALYI
	{ 0xFF, 0xFF},					// big malyi
	{ 0xFF, 0xFF},					// KSI
	{ 0xFF, 0xFF},					// ksi
	{ 0xFF, 0xFF},					// PSI
	{ 0xFF, 0xFF},					// psi
	{ 0xFF, 0xFF},					// FITA
	{ 0xFF, 0xFF},					// fita
	{ 0xFF, 0xFF},					// IZHITSA
	{ 0xFF, 0xFF},					// izhitsa
	{ 00, racute},					// Russian A acute
	{ 01, racute },				// Russian a acute
	{ 10, racute },				// Russian IE acute
	{ 11, racute },				// Russian ie acute
	{ 78, racute },				// Russian E acute
	{ 79, racute },				// Russian e acute
	{ 18, racute },				// Russian II acute
	{ 19, racute },				// Russian ii acute
	{ 88, racute },				// Russian I acute
	{ 89, racute },				// Russian i acute
	{ 90, racute },				// Russian YI acute
	{ 91, racute },				// Russian yi acute
	{ 30, racute },				// Russian O acute
	{ 31, racute },				// Russian o acute
	{ 40, racute },				// Russian U acute
	{ 41, racute },				// Russian u acute
	{ 56, racute },				// Russian YERI acute
	{ 57, racute },				// Russian yeri acute
	{ 60, racute },				// Russian REVERSED E acute
	{ 61, racute },				// Russian reversed e acute
	{ 62, racute },				// Russian IU acute
	{ 63, racute },				// Russian iu acute
	{ 64, racute },				// Russian IA acute
	{ 65, racute },				// Russian ia acute
	{ 00, rgrave },				// Russian A grave
	{ 01, rgrave },				// Russian a grave
	{ 10, rgrave },				// Russian IE grave
	{ 11, rgrave },				// Russian ie grave
	{ 12, rgrave },				// Russian YO grave
	{ 13, rgrave },				// Russian yo grave
	{ 18, rgrave },				// Russian I grave
	{ 19, rgrave },				// Russian i grave
	{ 30, rgrave },				// Russian O grave
	{ 31, rgrave },				// Russian o grave
	{ 40, rgrave },				// Russian U grave
	{ 41, rgrave },				// Russian u grave
	{ 56, rgrave },				// Russian YERI grave
	{ 57, rgrave },				// Russian yeri grave
	{ 60, rgrave },				// Russian REVERSED E grave
	{ 61, rgrave },				// Russian reversed e grave
	{ 62, rgrave },				// Russian IU grave
	{ 63, rgrave },				// Russian iu grave
	{ 64, rgrave },				// Russian IA grave
	{ 65, rgrave}					// Russian ia grave
};

/****************************************************************************
Desc:
****************************************************************************/
static BASE_DIACRIT fwp_rus_c =
{
	120,				// # of characters in table.
	156,				// start char.
	fwp_rus_c_table,
};

/****************************************************************************
Desc:		Table of pointers to character component tables.
****************************************************************************/
BASE_DIACRIT * fwp_car60_c[ NCHSETS] =
{
	(BASE_DIACRIT*)0,    // no composed characters for ascii.
	&fwp_ml1c,
	(BASE_DIACRIT*)0,    // no composed characters for multinational 2
	(BASE_DIACRIT*)0,    // no composed characters for line draw.
	(BASE_DIACRIT*)0,    // no composed characters for typographic.
	(BASE_DIACRIT*)0,    // no composed characters for icons.
	(BASE_DIACRIT*)0,    // no composed characters for math.
	(BASE_DIACRIT*)0,    // no composed characters for math extension.
	&fwp_grk_c,				// Greek
	(BASE_DIACRIT*)0,		// Hebrew
	&fwp_rus_c,				// Cyrillic - Russian
	(BASE_DIACRIT*)0,		// Hiragana or Katakana (Japanese)
	(BASE_DIACRIT*)0,		// no composed characters for user.
	(BASE_DIACRIT*)0,		// no composed characters for Arabic.
	(BASE_DIACRIT*)0,		// no composed characters for Arabic Script .
};

/****************************************************************************
Desc:		Map special chars in CharSet (x24) to collation values
****************************************************************************/
BYTE_WORD_TBL fwp_Ch24ColTbl[] =	// Position in the table+1 is subColValue
{
	{1,	COLLS+2},					// comma
	{2,	COLLS+1},					// maru
	{5,	COLS_ASIAN_MARKS+2},		// chuuten
	{10,	COLS_ASIAN_MARKS},		// dakuten
	{11,	COLS_ASIAN_MARKS+1},		// handakuten
	{43,	COLS2+2},					// angled brackets
	{44,	COLS2+3},					//
	{49,	COLS2+2},					// pointy brackets
	{50,	COLS2+3},
	{51,	COLS2+2},					// double pointy brackets
	{52,	COLS2+3},
	{53,	COLS1},						// Japanese quotes
	{54,	COLS1},
	{55,	COLS1},						// hollow Japanese quotes
	{56,	COLS1},
	{57,	COLS2+2},					// filled rounded brackets
	{58,	COLS2+3}
};

/****************************************************************************
Desc:		Kana subcollation values
		 		BIT 0: set if large char
				 BIT 1: set if voiced
				 BIT 2: set if half voiced
Notes:
			To save space should be nibbles
			IMPORTANT:
				The '1' entries that do not have
				a matching '0' entry have been
				changed to zero to save space in
				the subcollation area.
				The original table is listed below.
****************************************************************************/
FLMBYTE KanaSubColTbl[] =
{
	0,1,0,1,0,1,0,1,0,1,				// a    A   i   I   u   U   e   E   o   O
	1,3,0,3,0,3,1,3,0,3,				// KA  GA  KI  GI  KU  GU  KE  GE  KO  GO
	0,3,0,3,0,3,0,3,0,3,				// SA  ZA SHI  JI  SU  ZU  SE  ZE  SO  ZO
	0,3,0,3,0,1,3,0,3,0,3,			// TA  DA CHI  JI tsu TSU  ZU  TE DE TO DO
	0,0,0,0,0,							// NA NI NU NE NO
	0,3,5,0,3,5,0,3,5,				// HA BA PA HI BI PI FU BU PU
	0,3,5,0,3,5,						// HE BE PE HO BO PO
	0,0,0,0,0,							// MA MI MU ME MO
	0,1,0,1,0,1,						// ya YA yu YU yo YO
	0,0,0,0,0,							// RA RI RU RE RO
	0,1,0,0,0,							// wa WA WI WE WO
	0,3,0,0								//  N VU ka ke
};

/****************************************************************************
Desc:		Map katakana (CharSet x26) to collation values
			kana collating values are two byte values
			where the high byte is 0x01.
****************************************************************************/
FLMBYTE KanaColTbl[] =
{
	 0, 0, 1, 1, 2, 2, 3, 3, 4, 4,		// a    A   i   I   u   U   e   E   o   O
 	 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,		// KA  GA  KI  GI  KU  GU  KE  GE  KO  GO
	10,10,11,11,12,12,13,13,14,14,		// SA  ZA SHI  JI  SU  ZU  SE  ZE  SO  ZO
	15,15,16,16,17,17,17,18,18,19,19,	// TA DA CHI JI tsu TSU  ZU  TE DE TO DO
	20,21,22,23,24,							// NA NI NU NE NO
	25,25,25,26,26,26,27,27,27,			// HA BA PA HI BI PI FU BU PU
	28,28,28,29,29,29,						// HE BE PE HO BO PO
	30,31,32,33,34,							// MA MI MU ME MO
	35,35,36,36,37,37,						// ya YA yu YU yo YO
	38,39,40,41,42,							// RA RI RU RE RO
	43,43,44,45,46,							// wa WA WI WE WO
	47, 2, 5, 8									//  N VU ka ke
};

/****************************************************************************
Desc:		Map KataKana collated value to vowel value for
			use for the previous char.
****************************************************************************/
FLMBYTE KanaColToVowel[] =
{
	0,1,2,3,4,		//  a   i   u  e  o
	0,1,2,3,4,		// ka  ki  ku ke ko
	0,1,2,3,4,		// sa shi  su se so
	0,1,2,3,4,		// ta chi tsu te to
	0,1,2,3,4,		// na  ni  nu ne no
	0,1,2,3,4,		// ha  hi  hu he ho
	0,1,2,3,4,		// ma  mi  mu me mo
	0,2,4,			// ya  yu  yo
	0,1,2,3,4,		// ra  ri  ru re ro
	0,1,3,4,			// wa  wi  we wo
};

/****************************************************************************
Desc:		Convert Zenkaku (double wide) to Hankaku (single wide)
			Character set 0x24 maps to single wide chars in other char sets.
			This enables collation values to be found on some symbols.
			This is also used to convert symbols from hankaku to Zen24.
****************************************************************************/
BYTE_WORD_TBL Zen24ToHankaku[] =
{
	{	0  ,0x0020 },		// space
	{	1  ,0x0b03 },		// japanese comma
	{	2  ,0x0b00 },		// circle period
	{	3  ,  44	 },		// comma
	{	4  ,  46	 },		// period
	{	5  ,0x0b04 },		// center dot
	{	6  ,  58	 },		// colon
	{	7  ,  59	 },		// semicolon
	{	8  ,  63	 },		// question mark
	{	9  ,  33	 },		// exclamation mark
	{	10 ,0x0b3d },		// dakuten
	{	11 ,0x0b3e },		// handakuten
	{	12 ,0x0106 },		// accent mark
	{	13 ,  96	 },		// accent mark
	{	14 ,0x0107 },		// umlat
	{	15 ,  94	 },		// caret
	{	16 ,0x0108 },		// macron
	{	17 ,  95	 },		// underscore
	{	27 ,0x0b0f },		// extend vowel
	{	28 ,0x0422 },		// mdash
	{	29 ,  45	 },		// hyphen
	{	30 ,  47  },     	// slash
	{	31 ,0x0607 },		// backslash
	{	32 , 126	 },		// tilde
	{	33 ,0x0611 },		// doubleline
	{	34 ,0x0609 },		// line
	{	37 ,0x041d },		// left apostrophe
	{	38 ,0x041c },		// right apostrophe
	{	39 ,0x0420 },		// left quote
	{	40 ,0x041f },		// right quote
	{	41 ,  40	 },		// left paren
	{	42 ,  41	 },		// right paren
	{	45 ,  91	 },		// left bracket
	{	46 ,  93	 },		// right bracket
	{	47 , 123	 },		// left curly bracket
	{	48 , 125	 },		// right curly bracket
	{	53 ,0x0b01 },		// left j quote
	{	54 ,0x0b02 },		// right j quote
	{	59 ,  43	 },		// plus
	{	60 ,0x0600 },		// minus
	{	61 ,0x0601 },		// plus/minus
	{	62 ,0x0627 },		// times
	{	63 ,0x0608 },		// divide
	{	64 ,  61	 },		// equal
	{	65 ,0x0663 },		// unequal
	{	66 ,  60	 },		// less
	{	67 ,  62	 },		// greater
	{	68 ,0x0602 },		// less/equal
	{	69 ,0x0603 },		// greater/equal
	{	70 ,0x0613 },		// infinity
	{	71 ,0x0666 },		// traingle dots
	{	72 ,0x0504 },		// man
	{	73 ,0x0505 },		// woman
	{	75 ,0x062d },		// prime
	{	76 ,0x062e },		// double prime
	{	78 ,0x040c },		// yen
	{	79 ,  36	 },		// $
	{	80 ,0x0413 },		// cent
	{	81 ,0x040b },		// pound
	{	82 ,  37	 },		// %
	{	83 ,  35	 },		// #
	{	84 ,  38	 },		// &
	{	85 ,  42	 },		// *
	{	86 ,  64	 },		// @
	{	87 ,0x0406 },		// squiggle
	{	89 ,0x06b8 },		// filled star
	{	90 ,0x0425 },		// hollow circle
	{	91 ,0x042c },		// filled circle
	{	93 ,0x065f },		// hollow diamond
	{	94 ,0x0660 },		// filled diamond
	{	95 ,0x0426 },		// hollow box
	{	96 ,0x042e },		// filled box
	{	97 ,0x0688 },		// hollow triangle
	{	99 ,0x0689 },		// hollow upside down triangle
	{	103,0x0615 },		// right arrow
	{	104,0x0616 },		// left arrow
	{	105,0x0617 },		// up arrow
	{	106,0x0622 },		// down arrow
	{	119,0x060f },
	{	121,0x0645 },
	{	122,0x0646 },
	{	123,0x0643 },
	{	124,0x0644 },
	{	125,0x0642 },		// union
	{	126,0x0610 },		// intersection
	{	135,0x0655 },
	{	136,0x0656 },
	{	138,0x0638 },		// right arrow
	{	139,0x063c },		// left/right arrow
	{	140,0x067a },
	{	141,0x0679 },
	{	153,0x064f },		// angle
	{	154,0x0659 },
	{	155,0x065a },
	{	156,0x062c },
	{	157,0x062b },
	{	158,0x060e },
	{	159,0x06b0 },
	{	160,0x064d },
	{	161,0x064e },
	{	162,0x050e },		// square root
	{	164,0x0604 },
	{	175,0x0623 },		// angstrom
	{	176,0x044b },		// percent
	{	177,0x051b },		// sharp
	{	178,0x051c },		// flat
	{	179,0x0509 },		// musical note
	{	180,0x0427 },		// dagger
	{	181,0x0428 },		// double dagger
	{	182,0x0405 },		// paragraph
	{	187,0x068f }		// big hollow circle
};

/****************************************************************************
Desc:		Maps CS26 to CharSet 11
			Used to uncollate characters for FLAIM - placed here for consistency
				0x80 - add dakuten
				0xC0 - add handakuten
				0xFF - no mapping exists
****************************************************************************/
FLMBYTE MapCS26ToCharSet11[ 86] =
{
	0x06,	// 0     a
	0x10,	// 1     A
	0x07,	// 2     i
	0x11,	//	3     I
	0x08,	//	4     u
	0x12,	//	5     U
	0x09,	//	6     e
	0x13,	//	7     E
	0x0a,	//	8     o
	0x14,	//	9     O

	0x15,	//	0x0a  KA
	0x95,	//       GA - 21 followed by 0x3D dakuten

	0x16,	// 0x0c  KI
	0x96,	//       GI
	0x17,	//	0x0e  KU
	0x97,	//       GU
	0x18,	// 0x10  KE
	0x98,	//       GE
	0x19,	// 0x12  KO
	0x99,	//       GO

	0x1a,	//	0x14  SA
	0x9a,	//       ZA
	0x1b,	//	0x16  SHI
	0x9b,	//       JI
	0x1c,	//	0x18  SU
	0x9c,	//       ZU
	0x1d,	//	0x1a  SE
	0x9d,	//       ZE
	0x1e,	//	0x1c  SO
	0x9e,	//       ZO

	0x1f,	//	0x1e  TA
	0x9f,	//       DA
	0x20,	//	0x20  CHI
	0xa0,	//       JI
	0x0e,	//	0x22  small tsu
	0x21,	//	0x23  TSU
	0xa1,	//       ZU
	0x22,	//	0x25  TE
	0xa2,	//       DE
	0x23,	//	0x27  TO
	0xa3,	//       DO

	0x24,	//	0x29  NA
	0x25,	//	0x2a  NI
	0x26,	// 0x2b  NU
	0x27,	//	0x2c  NE
	0x28,	//	0x2d  NO

	0x29,	//	0x2e  HA
	0xa9,	// 0x2f  BA
	0xe9,	// 0x30  PA
	0x2a,	//	0x31  HI
	0xaa,	// 0x32  BI
	0xea,	// 0x33  PI
	0x2b,	//	0x34  FU
	0xab,	// 0x35  BU
	0xeb,	// 0x36  PU
	0x2c,	//	0x37  HE
	0xac,	// 0x38  BE
	0xec,	// 0x39  PE
	0x2d,	//	0x3a  HO
	0xad,	// 0x3b  BO
	0xed,	// 0x3c  PO

	0x2e,	//	0x3d  MA
	0x2f,	//	0x3e  MI
	0x30,	//	0x3f  MU
	0x31,	//	0x40  ME
	0x32,	//	0x41  MO

	0x0b,	//	0x42  small ya
	0x33,	//	0x43  YA
	0x0c,	//	0x44  small yu
	0x34,	//	0x45  YU
	0x0d,	// 0x46  small yo
	0x35,	//	0x47  YO

	0x36,	//	0x48  RA
	0x37,	//	0x49  RI
	0x38,	//	0x4a  RU
	0x39,	//	0x4b  RE
	0x3a,	//	0x4c  RO

	0xff,	// 0x4d  small wa
	0x3b,	//	0x4e  WA
	0xff,	// 0x4f  WI
	0xff,	// 0x50  WE
	0x05,	//	0x51	WO

	0x3c,	//	0x52	N
	0xff,	// 0x53  VU
	0xff, // 0x54  ka
	0xff 	// 0x55  ke
};

/****************************************************************************
Desc:		Conversion from single (Hankaku) to double (Zenkaku) wide characters
			Used in flmWPHanToZenkaku()
			Maps from charset 11 to CS24 (punctuation) (starting from 11,0)
****************************************************************************/
FLMBYTE From0AToZen[] =				// ' changed because of windows
{
 	0, 	9,		40,	0x53, 		// sp ! " #
 	0x4f, 0x52, 0x54,	38, 			// $ % & '
 											// Was 187 for ! and 186 for '
	0x29,	0x2a,	0x55,	0x3b, 		// ( ) * +
	3,		0x1d,	4,		0x1e	 		// , - . /
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE From0BToZen[] =
{
	6,		7,		0x42,	0x40,			// : ; < =
	0x43,	8,		0x56					// > ? @
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE From0CToZen[] =
{
	0x2d,	0x1f,	0x2e,	0x0f,	0x11,	0x0d	// [ BACKSLASH ] ^ _ `
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE From0DToZen[] =
{
	0x2f,	0x22,	0x30,	0x20 			// { | } ~
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE  From8ToZen[] =
{
	0x5e, 0x7e, 0x5f, 0x7f, 0x5f, 0xFF, 0x60, 0x80,
	0x61, 0x81, 0x62, 0x82, 0x63, 0x83, 0x64, 0x84,
	0x65, 0x85, 0x66, 0x86, 0x67, 0x87, 0x68, 0x88,
	0x69, 0x89, 0x6a, 0x8a, 0x6b, 0x8b, 0x6c, 0x8c,
	0x6d, 0x8d, 0x6e, 0x8e, 0x6f, 0x8f, 0x6f, 0xFF,
	0x70, 0x90, 0x71, 0x91, 0x72, 0x92, 0x73, 0x93,
	0x74, 0x94, 0x75, 0x95
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE From11AToZen[] =	// 11 to 24 punctuation except dash
{
	2,								// japanese period
	0x35,							// left bracket
	0x36,							// right bracket
	0x01,							// comma
	0x05							// chuuten
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE From11BToZen[] =				// 11 to 26 (katakana) from 11,5
{
	0x51,										// wo
	0,2,4,6,8,0x42,0x44,0x46,0x22,	// small a i u e o ya yu yo tsu
	0xFF, 1, 3, 5, 7, 9,					// dash (x241b) a i u e o
	0x0a, 0x0c, 0x0e, 0x10, 0x12,		// ka ki ku ke ko
	0x14, 0x16, 0x18, 0x1a, 0x1c,		// sa shi su se so
	0x1e, 0x20, 0x23, 0x25, 0x27,		// ta chi tsu te to
	0x29, 0x2a, 0x2b, 0x2c, 0x2d,		// na ni nu ne no
	0x2e, 0x31, 0x34, 0x37, 0x3a,		// ha hi fu he ho
	0x3d, 0x3e, 0x3f, 0x40, 0x41,		// ma mi mu me mo
	0x43, 0x45, 0x47,						// ya yu yo
	0x48, 0x49, 0x4a, 0x4b, 0x4c,		// ra ri ru re ro
	0x4e, 0x52								// WA N
};												// does not have wa WI WE VU ka ke

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT16 fwp_indexi[] =
{
	0,11,14,15,17,18,19,21,22,23,24,25,26,35,59
};

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT16 fwp_indexj[] =	// DOUBLE CHAR AREA - LANGUAGES
{
	XFLM_CA_LANG,	// Catalan (0)
	XFLM_CF_LANG,	// Canadian French
	XFLM_CZ_LANG,	// Czech
	XFLM_SL_LANG,	// Slovak
	XFLM_DE_LANG,	// German
	XFLM_SD_LANG,	// Swiss German
	XFLM_ES_LANG,	// Spanish (Spain)
	XFLM_FR_LANG,	// French
	XFLM_NL_LANG,	// Netherlands
	0xFFFF,			// DK_LANG,	Danish    - support for 'aa' -> a-ring out
	0xFFFF,			// NO_LANG,	Norwegian - support for 'aa' -> a-ring out
	0x0063,			// c						 - DOUBLE CHARACTERS - STATE ENTRIES
	0x006c,			// l
	0x0197,			// l with center dot
	0x0063,			// c
	0x0125,			// ae digraph
	0x01a7,			// oe digraph
	0x0068,			// h
	0x0068,			// h
	0x006c,			// l
	0x0101,			// center dot alone
	0x006c,			// l
	0x0117,			// ?	(for German)
	0x018b,			// ij digraph
	0x0000,			// was 'a' - will no longer map 'aa' to a-ring
	0x0000,			// was 'a'

	XFLM_CZ_LANG,	// SINGLE CHARS - LANGUAGES
	XFLM_DK_LANG,
	XFLM_NO_LANG,
	XFLM_SL_LANG,
	XFLM_TK_LANG,
	XFLM_SU_LANG,
	XFLM_IS_LANG,
	XFLM_SV_LANG,
	XFLM_YK_LANG,
						// SINGLE CHARS
	0x011e,			// A Diaeresis					- alternate collating sequences
	0x011f,			// a Diaeresis
	0x0122,			// A Ring						- 2
	0x0123,			// a Ring
	0x0124,			// AE Diagraph					- 4
	0x0125,			// ae diagraph
	0x013e,			// O Diaeresis					- 6
	0x013f,			// o Diaeresis
	0x0146,			// U Diaeresis					- 8
	0x0147,			// u Diaeresis
	0x0150,			// O Slash						- 10
	0x0151,			// o Slash

	0x0A3a,			// CYRILLIC SOFT SIGN		- 12
	0x0A3b,			// CYRILLIC soft sign
	0x01ee,			// dotless i - turkish		- 14
	0x01ef,			// dotless I - turkish
	0x0162,			// C Hacek/caron - 1,98		- 16
	0x0163,			// c Hacek/caron - 1,99
	0x01aa,			// R Hacek/caron - 1,170	- 18
	0x01ab,			// r Hacek/caron - 1,171
	0x01b0,			// S Hacek/caron - 1,176	- 20
	0x01b1,			// s Hacek/caron - 1,177
	0x01ce,			// Z Hacek/caron - 1,206	- 22
	0x01cf,			// z Hacek/caron - 1,207
};

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT16 fwp_valuea[] =
{
//	DOUBLE CHAR STATE VALUES
	STATE1,		// 00
	STATE3,
	STATE2,
	STATE2,
	STATE8,
	STATE8,
	STATE1,
	STATE3,
	STATE9,
	STATE10,		// No longer in use
	STATE10,		// No longer in use
	STATE4,
	STATE6,
	STATE6,
	STATE5,
	INSTAE,
	INSTOE,
	AFTERC,
	AFTERH,
	AFTERL,
	STATE7,
	STATE6,
	INSTSG,		// ss for German
	INSTIJ,
	STATE11,		// aa - no longer in use
	WITHAA,		// aa - no longer in use

// SINGLE CHARS - LANGUAGES
	START_CZ,	// Czech
	START_DK,	// Danish
	START_NO,	// Norwegian
	START_SL,	// Slovak
	START_TK,	// Turkish
	START_SU,	// Finnish
	START_IS,	// Icelandic
	START_SV,	// Swedish
	START_YK,	// Ukrainian

// SINGLE CHARS FIXUP AREAS
	COLS9,		COLS9,		COLS9,		COLS9,		// US & OTHERS
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9+45,	COLS9+45,	COLS9+55,	COLS9+55,	// DANISH
	COLS9+42,	COLS9+42,	COLS9+53,	COLS9+53,
	COLS9+30,	COLS9+30,	COLS9+49,	COLS9+49,	// Oct98 U Diaer no longer to y Diaer
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9,		COLS9,		// Icelandic
	COLS9+46,	COLS9+46,	COLS9+50,	COLS9+50,
	COLS9+30,	COLS9+30,	COLS9+54,	COLS9+54,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9+51,	COLS9+51,	// Norwegian
	COLS9+43,	COLS9+43,	COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+47,	COLS9+47,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9+48,	COLS9+48,	COLS9+44,	COLS9+44,	// Finnish/Swedish
	COLS9+1,		COLS9+1,		COLS9+52,	COLS9+52,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,	// Oct98 U Diaer no longer to y Diaer
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9,		COLS9,		// Ukrain
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS10+48,	COLS10+48,	COLS9+12,	COLS9+12,
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

	COLS9,		COLS9,		COLS9,		COLS9,		// Turkish
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS9+43,	COLS9+43,	COLS9+11,	COLS9+11,	// dotless i same as
	COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,	// the "CH" in Czech
	COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,	// works because char
																	// fails brkcar()

	COLS9,		COLS9,		COLS9,		COLS9,		// Czech / Slovak
	COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
	COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
	COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
	COLS9+5,		COLS9+5,		COLS9+26,	COLS9+26,	// carons
	COLS9+28,	COLS9+28,	COLS9+36,	COLS9+36
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE fwp_asc60Tbl[ ASCTBLLEN + 2] =
{
	0x20,			// initial character offset!!
	ASCTBLLEN,	// len of this table
	COLLS,		// <Spc>
	COLLS+5,		// !
	COLS1,		// "
	COLS6+1,		// #
	COLS3,		// $
	COLS6,		// %
	COLS6+2,		// &
	COLS1+1,		// '
	COLS2,		// (
	COLS2+1,		// )
	COLS4+2,		// *
	COLS4,		// +
	COLLS+2,		// ,
	COLS4+1,		// -
	COLLS+1,		// .
	COLS4+3,		// /
	COLS8,		// 0
	COLS8+1,		// 1
	COLS8+2,		// 2
	COLS8+3,		// 3
	COLS8+4,		// 4
	COLS8+5,		// 5
	COLS8+6,		// 6
	COLS8+7,		// 7
	COLS8+8,		// 8
	COLS8+9,		// 9
	COLLS+3,		// :
	COLLS+4,		// ;
	COLS5,		// <
	COLS5+2,		// =
	COLS5+4,		// >
	COLLS+7,		// ?
	COLS6+3,		// @
	COLS9,		// A
	COLS9+2,		// B
	COLS9+3,		// C
	COLS9+6,		// D
	COLS9+7,		// E
	COLS9+8,		// F
	COLS9+9,		// G
	COLS9+10,	// H
	COLS9+12,	// I
	COLS9+14,	// J
	COLS9+15,	// K
	COLS9+16,	// L
	COLS9+18,	// M
	COLS9+19,	// N
	COLS9+21,	// O
	COLS9+23,	// P
	COLS9+24,	// Q
	COLS9+25,	// R
	COLS9+27,	// S
	COLS9+29,	// T
	COLS9+30,	// U
	COLS9+31,	// V
	COLS9+32,	// W
	COLS9+33,	// X
	COLS9+34,	// Y
	COLS9+35,	// Z
	COLS9+40,	// [ (note: alphabetic - end of list)
	COLS6+4,		// Backslash
	COLS9+41,	// ] (note: alphabetic - end of list)
	COLS4+4,		// ^
	COLS6+5,		// _
	COLS1+2,		// `
	COLS9,		// a
	COLS9+2,		// b
	COLS9+3,		// c
	COLS9+6,		// d
	COLS9+7,		// e
	COLS9+8,		// f
	COLS9+9,		// g
	COLS9+10,	// h
	COLS9+12,	// i
	COLS9+14,	// j
	COLS9+15,	// k
	COLS9+16,	// l
	COLS9+18,	// m
	COLS9+19,	// n
	COLS9+21,	// o
	COLS9+23,	// p
	COLS9+24,	// q
	COLS9+25,	// r
	COLS9+27,	// s
	COLS9+29,	// t
	COLS9+30,	// u
	COLS9+31,	// v
	COLS9+32,	// w
	COLS9+33,	// x
	COLS9+34,	// y
	COLS9+35,	// z
	COLS2+4,		// {
	COLS6+6,		// |
	COLS2+5,		// }
	COLS6+7		// ~
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE fwp_mn60Tbl[ MNTBLLEN + 2] =		// multinational table
{
	23,			// initial character offset!!
	MNTBLLEN,	// len of this table
	COLS9+27,	// German Double s
	COLS9+15,	// Icelandic k
	COLS9+14,	// Dotless j

// IBM Charset

	COLS9,		// A Acute
	COLS9,		// a Acute
	COLS9,		// A Circumflex
	COLS9,		// a Circumflex
	COLS9,		// A Diaeresis or Umlaut
	COLS9,		// a Diaeresis or Umlaut
	COLS9,		// A Grave
	COLS9,		// a Grave
	COLS9,		// A Ring
	COLS9,		// a Ring
	COLS9+1,		// AE digraph
	COLS9+1,		// ae digraph
	COLS9+3,		// C Cedilla
	COLS9+3,		// c Cedilla
	COLS9+7,		// E Acute
	COLS9+7,		// e Acute
	COLS9+7,		// E Circumflex
	COLS9+7,		// e Circumflex
	COLS9+7,		// E Diaeresis or Umlaut
	COLS9+7,		// e Diaeresis or Umlaut
	COLS9+7,		// E Grave
	COLS9+7,		// e Grave
	COLS9+12,	// I Acute
	COLS9+12,	// i Acute
	COLS9+12,	// I Circumflex
	COLS9+12,	// i Circumflex
	COLS9+12,	// I Diaeresis or Umlaut
	COLS9+12,	// i Diaeresis or Umlaut
	COLS9+12,	// I Grave
	COLS9+12,	// i Grave
	COLS9+20,	// N Tilde
	COLS9+20,	// n Tilde
	COLS9+21,	// O Acute
	COLS9+21,	// o Acute
	COLS9+21,	// O Circumflex
	COLS9+21,	// o Circumflex
	COLS9+21,	// O Diaeresis or Umlaut
	COLS9+21,	// o Diaeresis or Umlaut
	COLS9+21,	// O Grave
	COLS9+21,	// o Grave
	COLS9+30,	// U Acute
	COLS9+30,	// u Acute
	COLS9+30,	// U Circumflex
	COLS9+30,	// u Circumflex
	COLS9+30,	// U Diaeresis or Umlaut
	COLS9+30,	// u Diaeresis or Umlaut
	COLS9+30,	// U Grave
	COLS9+30,	// u Grave
	COLS9+34,	// Y Diaeresis or Umlaut
	COLS9+34,	// y Diaeresis or Umlaut

// IBM foreign

	COLS9,		// A Tilde
	COLS9,		// a Tilde
	COLS9+6,		// D Cross Bar
	COLS9+6,		// d Cross Bar
	COLS9+21,	// O Slash
	COLS9+21,	// o Slash
	COLS9+21,	// O Tilde
	COLS9+21,	// o Tilde
	COLS9+34,	// Y Acute
	COLS9+34,	// y Acute
	COLS9+6,		// Uppercase Eth
	COLS9+6,		// Lowercase Eth
	COLS9+37,	// Uppercase Thorn
	COLS9+37,	// Lowercase Thorn

// Teletex chars

	COLS9,		// A Breve
	COLS9,		// a Breve
	COLS9,		// A Macron
	COLS9,		// a Macron
	COLS9,		// A Ogonek
	COLS9,		// a Ogonek
	COLS9+3,		// C Acute
	COLS9+3,		// c Acute
	COLS9+3,		// C Caron or Hachek
	COLS9+3,		// c Caron or Hachek
	COLS9+3,		// C Circumflex
	COLS9+3,		// c Circumflex
	COLS9+3,		// C Dot Above
	COLS9+3,		// c Dot Above
	COLS9+6,		// D Caron or Hachek (Apostrophe Beside)
	COLS9+6,		// d Caron or Hachek (Apostrophe Beside)
	COLS9+7,		// E Caron or Hachek
	COLS9+7,		// e Caron or Hachek
	COLS9+7,		// E Dot Above
	COLS9+7,		// e Dot Above
	COLS9+7,		// E Macron
	COLS9+7,		// e Macron
	COLS9+7,		// E Ogonek
	COLS9+7,		// e Ogonek
	COLS9+9,		// G Acute
	COLS9+9,		// g Acute
	COLS9+9,		// G Breve
	COLS9+9,		// g Breve
	COLS9+9,		// G Caron or Hachek
	COLS9+9,		// g Caron or Hachek
	COLS9+9,		// G Cedilla (Apostrophe Under)
	COLS9+9,		// g Cedilla (Apostrophe Over)
	COLS9+9,		// G Circumflex
	COLS9+9,		// g Circumflex
	COLS9+9,		// G Dot Above
	COLS9+9,		// g Dot Above
	COLS9+10,	// H Circumflex
	COLS9+10,	// h Circumflex
	COLS9+10,	// H Cross Bar
	COLS9+10,	// h Cross Bar
	COLS9+12,	// I Dot Above (Sharp Accent)
	COLS9+12,	// i Dot Above (Sharp Accent)
	COLS9+12,	// I Macron
	COLS9+12,	// i Macron
	COLS9+12,	// I Ogonek
	COLS9+12,	// i Ogonek
	COLS9+12,	// I Tilde
	COLS9+12,	// i Tilde
	COLS9+13,	// IJ Digraph
	COLS9+13,	// ij Digraph
	COLS9+14,	// J Circumflex
	COLS9+14,	// j Circumflex
	COLS9+15,	// K Cedilla (Apostrophe Under)
	COLS9+15,	// k Cedilla (Apostrophe Under)
	COLS9+16,	// L Acute
	COLS9+16,	// l Acute
	COLS9+16,	// L Caron or Hachek (Apostrophe Beside)
	COLS9+16,	// l Caron or Hachek (Apostrophe Beside)
	COLS9+16,	// L Cedilla (Apostrophe Under)
	COLS9+16,	// l Cedilla (Apostrophe Under)
	COLS9+16,	// L Center Dot
	COLS9+16,	// l Center Dot
	COLS9+16,	// L Stroke
	COLS9+16,	// l Stroke
	COLS9+19,	// N Acute
	COLS9+19,	// n Acute
	COLS9+19,	// N Apostrophe
	COLS9+19,	// n Apostrophe
	COLS9+19,	// N Caron or Hachek
	COLS9+19,	// n Caron or Hachek
	COLS9+19,	// N Cedilla (Apostrophe Under)
	COLS9+19,	// n Cedilla (Apostrophe Under)
	COLS9+21,	// O Double Acute
	COLS9+21,	// o Double Acute
	COLS9+21,	// O Macron
	COLS9+21,	// o Macron
	COLS9+22,	// OE digraph
	COLS9+22,	// oe digraph
	COLS9+25,	// R Acute
	COLS9+25,	// r Acute
	COLS9+25,	// R Caron or Hachek
	COLS9+25,	// r Caron or Hachek
	COLS9+25,	// R Cedilla (Apostrophe Under)
	COLS9+25,	// r Cedilla (Apostrophe Under)
	COLS9+27,	// S Acute
	COLS9+27,	// s Acute
	COLS9+27,	// S Caron or Hachek
	COLS9+27,	// s Caron or Hachek
	COLS9+27,	// S Cedilla
	COLS9+27,	// s Cedilla
	COLS9+27,	// S Circumflex
	COLS9+27,	// s Circumflex
	COLS9+29,	// T Caron or Hachek (Apostrophe Beside)
	COLS9+29,	// t Caron or Hachek (Apostrophe Beside)
	COLS9+29,	// T Cedilla (Apostrophe Under)
	COLS9+29,	// t Cedilla (Apostrophe Under)
	COLS9+29,	// T Cross Bar
	COLS9+29,	// t Cross Bar
	COLS9+30,	// U Breve
	COLS9+30,	// u Breve
	COLS9+30,	// U Double Acute
	COLS9+30,	// u Double Acute
	COLS9+30,	// U Macron
	COLS9+30,	// u Macron
	COLS9+30,	// U Ogonek
	COLS9+30,	// u Ogonek
	COLS9+30,	// U Ring
	COLS9+30,	// u Ring
	COLS9+30,	// U Tilde
	COLS9+30,	// u Tilde
	COLS9+32,	// W Circumflex
	COLS9+32,	// w Circumflex
	COLS9+34,	// Y Circumflex
	COLS9+34,	// y Circumflex
	COLS9+35,	// Z Acute
	COLS9+35,	// z Acute
	COLS9+35,	// Z Caron or Hachek
	COLS9+35,	// z Caron or Hachek
	COLS9+35,	// Z Dot Above
	COLS9+35,	// z Dot Above
	COLS9+19,	// Uppercase Eng
	COLS9+19,	// Lowercase Eng

// Other

	COLS9+6,		// D Macron
	COLS9+6,		// d Macron
	COLS9+16,	// L Macron
	COLS9+16,	// l Macron
	COLS9+19,	// N Macron
	COLS9+19,	// n Macron
	COLS9+25,	// R Grave
	COLS9+25,	// r Grave
	COLS9+27,	// S Macron
	COLS9+27,	// s Macron
	COLS9+29,	// T Macron
	COLS9+29,	// t Macron
	COLS9+34,	// Y Breve
	COLS9+34,	// y Breve
	COLS9+34,	// Y Grave
	COLS9+34,	// y Grave
	COLS9+6,		// D Apostrophe Beside
	COLS9+6,		// d Apostrophe Beside
	COLS9+21,	// O Apostrophe Beside
	COLS9+21,	// o Apostrophe Beside
	COLS9+30,	// U Apostrophe Beside
	COLS9+30,	// u Apostrophe Beside
	COLS9+7,		// E breve
	COLS9+7,		// e breve
	COLS9+12,	// I breve
	COLS9+12,	// i breve
	COLS9+12,	// dotless I
	COLS9+12,	// dotless i
	COLS9+21,	// O breve
	COLS9+21		// o breve
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE fwp_sym60Tbl[ SYMTBLLEN + 2] =
{
	11,			// initial character offset!!
	SYMTBLLEN,	// len of this table
	COLS3+2,		// pound
	COLS3+3,		// yen
	COLS3+4,		// pacetes
	COLS3+5,		// floren
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS3+1,		// cent
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE fwp_grk60Tbl[ GRKTBLLEN + 2] =
{
	0,					// starting offset
	GRKTBLLEN,		// length
	COLS7,			// Uppercase Alpha
	COLS7,			// Lowercase Alpha
	COLS7+1,			// Uppercase Beta
	COLS7+1,			// Lowercase Beta
	COLS7+1,			// Uppercase Beta Medial
	COLS7+1,			// Lowercase Beta Medial
	COLS7+2,			// Uppercase Gamma
	COLS7+2,			// Lowercase Gamma
	COLS7+3,			// Uppercase Delta
	COLS7+3,			// Lowercase Delta
	COLS7+4,			// Uppercase Epsilon
	COLS7+4,			// Lowercase Epsilon
	COLS7+5,			// Uppercase Zeta
	COLS7+5,			// Lowercase Zeta
	COLS7+6,			// Uppercase Eta
	COLS7+6,			// Lowercase Eta
	COLS7+7,			// Uppercase Theta
	COLS7+7,			// Lowercase Theta
	COLS7+8,			// Uppercase Iota
	COLS7+8,			// Lowercase Iota
	COLS7+9,			// Uppercase Kappa
	COLS7+9,			// Lowercase Kappa
	COLS7+10,		// Uppercase Lambda
	COLS7+10,		// Lowercase Lambda
	COLS7+11,		// Uppercase Mu
	COLS7+11,		// Lowercase Mu
	COLS7+12,		// Uppercase Nu
	COLS7+12,		// Lowercase Nu
	COLS7+13,		// Uppercase Xi
	COLS7+13,		// Lowercase Xi
	COLS7+14,		// Uppercase Omicron
	COLS7+14,		// Lowercase Omicron
	COLS7+15,		// Uppercase Pi
	COLS7+15,		// Lowercase Pi
	COLS7+16,		// Uppercase Rho
	COLS7+16,		// Lowercase Rho
	COLS7+17,		// Uppercase Sigma
	COLS7+17,		// Lowercase Sigma
	COLS7+17,		// Uppercase Sigma Terminal
	COLS7+17,		// Lowercase Sigma Terminal
	COLS7+18,		// Uppercase Tau
	COLS7+18,		// Lowercase Tau
	COLS7+19,		// Uppercase Upsilon
	COLS7+19,		// Lowercase Upsilon
	COLS7+20,		// Uppercase Phi
	COLS7+20,		// Lowercase Phi
	COLS7+21,		// Uppercase Chi
	COLS7+21,		// Lowercase Chi
	COLS7+22,		// Uppercase Psi
	COLS7+22,		// Lowercase Psi
	COLS7+23,		// Uppercase Omega
	COLS7+23,		// Lowercase Omega

// Other Modern Greek Characters [8,52]

	COLS7,			// Uppercase ALPHA Tonos high prime
	COLS7,			// Lowercase Alpha Tonos - acute
	COLS7+4,			// Uppercase EPSILON Tonos - high prime
	COLS7+4,			// Lowercase Epslion Tonos - acute
	COLS7+6,			// Uppercase ETA Tonos - high prime
	COLS7+6,			// Lowercase Eta Tonos - acute
	COLS7+8,			// Uppercase IOTA Tonos - high prime
	COLS7+8,			// Lowercase iota Tonos - acute
	COLS7+8,			// Uppercase IOTA Diaeresis
	COLS7+8,			// Lowercase iota diaeresis
	COLS7+14,		// Uppercase OMICRON Tonos - high prime
	COLS7+14,		// Lowercase Omicron Tonos - acute
	COLS7+19,		// Uppercase UPSILON Tonos - high prime
	COLS7+19,		// Lowercase Upsilon Tonos - acute
	COLS7+19,		// Uppercase UPSILON Diaeresis
	COLS7+19,		// Lowercase Upsilon diaeresis
	COLS7+23,		// Uppercase OMEGA Tonos - high prime
	COLS7+23,		// Lowercase Omega Tonso - acute

// Variants [8,70]

	COLS7+4,			// epsilon (variant)
	COLS7+7,			// theta (variant)
	COLS7+9,			// kappa (variant)
	COLS7+15,		// pi (variant)
	COLS7+16,		// rho (variant)
	COLS7+17,		// sigma (variant)
	COLS7+19,		// upsilon (variant)
	COLS7+20,		// phi (variant)
	COLS7+23,		// omega (variant)

// Greek Diacritic marks [8,79]

	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,
	COLS0,		// 8,108 end of diacritic marks

// Ancient Greek [8,109]

	COLS7,		// alpha grave
	COLS7,		// alpha circumflex
	COLS7,		// alpha w/iota
	COLS7,		// alpha acute w/iota
	COLS7,		// alpha grave w/iota
	COLS7,		// alpha circumflex w/Iota
	COLS7,		// alpha smooth
	COLS7,		// alpha smooth acute
	COLS7,		// alpha smooth grave
	COLS7,		// alpha smooth circumflex
	COLS7,		// alpha smooth w/Iota
	COLS7,		// alpha smooth acute w/Iota
	COLS7,		// alpha smooth grave w/Iota
	COLS7,		// alpha smooth circumflex w/Iota
// [8,123]
	COLS7,		// alpha rough
	COLS7,		// alpha rough acute
	COLS7,		// alpha rough grave
	COLS7,		// alpha rough circumflex
	COLS7,		// alpha rough w/Iota
	COLS7,		// alpha rough acute w/Iota
	COLS7,		// alpha rough grave w/Iota
	COLS7,		// alpha rough circumflex w/Iota
// [8,131]
	COLS7+4,		// epsilon grave
	COLS7+4,		// epsilon smooth
	COLS7+4,		// epsilon smooth acute
	COLS7+4,		// epsilon smooth grave
	COLS7+4,		// epsilon rough
	COLS7+4,		// epsilon rough acute
	COLS7+4,		// epsilon rough grave
// [8,138]
	COLS7+6,		// eta grave
	COLS7+6,		// eta circumflex
	COLS7+6,		// eta w/iota
	COLS7+6,		// eta acute w/iota
	COLS7+6,		// eta grave w/Iota
	COLS7+6,		// eta circumflex w/Iota
	COLS7+6,		// eta smooth
	COLS7+6,		// eta smooth acute
	COLS7+6,		// eta smooth grave
	COLS7+6,		// eta smooth circumflex
	COLS7+6,		// eta smooth w/Iota
	COLS7+6,		// eta smooth acute w/Iota
	COLS7+6,		// eta smooth grave w/Iota
	COLS7+6,		// eta smooth circumflex w/Iota
	COLS7+6,		// eta rough
	COLS7+6,		// eta rough acute
	COLS7+6,		// eta rough grave
	COLS7+6,		// eta rough circumflex
	COLS7+6,		// eta rough w/Iota
	COLS7+6,		// eta rough acute w/Iota
	COLS7+6,		// eta rough grave w/Iota
	COLS7+6,		// eta rough circumflex w/Iota
// [8,160]
	COLS7+8,		// iota grave
	COLS7+8,		// iota circumflex
	COLS7+8,		// iota acute diaeresis
	COLS7+8,		// iota grave diaeresis
	COLS7+8,		// iota smooth
	COLS7+8,		// iota smooth acute
	COLS7+8,		// iota smooth grave
	COLS7+8,		// iota smooth circumflex
	COLS7+8,		// iota rough
	COLS7+8,		// iota rough acute
	COLS7+8,		// iota rough grave
	COLS7+8,		// iota rough circumflex
// [8,172]
	COLS7+14,	// omicron grave
	COLS7+14,	// omicron smooth
	COLS7+14,	// omicron smooth acute
	COLS7+14,	// omicron smooth grave
	COLS7+14,	// omicron rough
	COLS7+14,	// omicron rough acute
	COLS7+14,	// omicron rough grave
// [8,179]
	COLS7+16,	// rho smooth
	COLS7+16,	// rho rough
// [8,181]
	COLS7+19,	// upsilon grave
	COLS7+19,	// upsilon circumflex
	COLS7+19,	// upsilon acute diaeresis
	COLS7+19,	// upsilon grave diaeresis
	COLS7+19,	// upsilon smooth
	COLS7+19,	// upsilon smooth acute
	COLS7+19,	// upsilon smooth grave
	COLS7+19,	// upsilon smooth circumflex
	COLS7+19,	// upsilon rough
	COLS7+19,	// upsilon rough acute
	COLS7+19,	// upsilon rough grave
	COLS7+19,	// upsilon rough circumflex
// [8,193]
	COLS7+23,	// omega grave
	COLS7+23,	// omega circumflex
	COLS7+23,	// omega w/Iota
	COLS7+23,	// omega acute w/Iota
	COLS7+23,	// omega grave w/Iota
	COLS7+23,	// omega circumflex w/Iota
	COLS7+23,	// omega smooth
	COLS7+23,	// omega smooth acute
	COLS7+23,	// omega smooth grave
	COLS7+23,	// omega smooth circumflex
	COLS7+23,	// omega smooth w/Iota
	COLS7+23,	// omega smooth acute w/Iota
	COLS7+23,	// omega smooth grave w/Iota
	COLS7+23,	// omega smooth circumflex w/Iota
	COLS7+23,	// omega rough
	COLS7+23,	// omega rough acute
	COLS7+23,	// omega rough grave
	COLS7+23,	// omega rough circumflex
	COLS7+23,	// omega rough w/Iota
	COLS7+23,	// omega rough acute w/Iota
	COLS7+23,	// omega rough grave w/Iota
	COLS7+23,	// omega rough circumflex w/Iota
// [8,215]
	COLS7+24,	// Uppercase Stigma--the number 6
	COLS7+24,	// Uppercase Digamma--Obsolete letter used as 6
	COLS7+24,	// Uppercase Koppa--Obsolete letter used as 90
	COLS7+24		// Uppercase Sampi--Obsolete letter used as 900
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE fwp_cyrl60Tbl[ CYRLTBLLEN + 2] =
{
	0,					// starting offset
	CYRLTBLLEN,		// len of table

	COLS10,			// Russian uppercase A
	COLS10,			// Russian lowercase A
	COLS10+1,		// Russian uppercase BE
	COLS10+1,		// Russian lowercase BE
	COLS10+2,		// Russian uppercase VE
	COLS10+2,		// Russian lowercase VE
	COLS10+3,		// Russian uppercase GHE
	COLS10+3,		// Russian lowercase GHE
	COLS10+5,		// Russian uppercase DE
	COLS10+5,		// Russian lowercase DE

	COLS10+8,		// Russian uppercase E
	COLS10+8,		// Russian lowercase E
	COLS10+9,		// Russian lowercase YO
	COLS10+9,		// Russian lowercase YO
	COLS10+11,		// Russian uppercase ZHE
	COLS10+11,		// Russian lowercase ZHE
	COLS10+12,		// Russian uppercase ZE
	COLS10+12,		// Russian lowercase ZE
	COLS10+14,		// Russian uppercase I
	COLS10+14,		// Russian lowercase I

	COLS10+17,		// Russian uppercase SHORT I
	COLS10+17,		// Russian lowercase SHORT I
	COLS10+19,		// Russian uppercase KA
	COLS10+19,		// Russian lowercase KA
	COLS10+20,		// Russian uppercase EL
	COLS10+20,		// Russian lowercase EL
	COLS10+22,		// Russian uppercase EM
	COLS10+22,		// Russian lowercase EM
	COLS10+23,		// Russian uppercase EN
	COLS10+23,		// Russian lowercase EN

	COLS10+25,		// Russian uppercase O
	COLS10+25,		// Russian lowercase O
	COLS10+26,		// Russian uppercase PE
	COLS10+26,		// Russian lowercase PE
	COLS10+27,		// Russian uppercase ER
	COLS10+27,		// Russian lowercase ER
	COLS10+28,		// Russian uppercase ES
	COLS10+28,		// Russian lowercase ES
	COLS10+29,		// Russian uppercase TE
	COLS10+29,		// Russian lowercase TE

	COLS10+32,		// Russian uppercase U
	COLS10+32,		// Russian lowercase U
	COLS10+34,		// Russian uppercase EF
	COLS10+34,		// Russian lowercase EF
	COLS10+35,		// Russian uppercase HA
	COLS10+35,		// Russian lowercase HA
	COLS10+36,		// Russian uppercase TSE
	COLS10+36,		// Russian lowercase TSE
	COLS10+37,		// Russian uppercase CHE
	COLS10+37,		// Russian lowercase CHE

	COLS10+39,		// Russian uppercase SHA
	COLS10+39,		// Russian lowercase SHA
	COLS10+40,		// Russian uppercase SHCHA
	COLS10+40,		// Russian lowercase SHCHA
	COLS10+41,		// Russian lowercase ER (also hard sign)
	COLS10+41,		// Russian lowercase ER (also hard sign)
	COLS10+42,		// Russian lowercase ERY
	COLS10+42,		// Russian lowercase ERY
	COLS10+43,		// Russian lowercase SOFT SIGN
	COLS10+43,		// Russian lowercase SOFT SIGN

	COLS10+45,		// Russian uppercase REVERSE E
	COLS10+45,		// Russian lowercase REVERSE E
	COLS10+46,		// Russian uppercase YU
	COLS10+46,		// Russian lowercase yu
	COLS10+47,		// Russian uppercase YA
	COLS10+47,		// Russian lowercase ya

	COLS0,			// Russian uppercase EH
	COLS0,			// Russian lowercase eh
	COLS10+7,		// Macedonian uppercase SOFT DJ
	COLS10+7,		// Macedonian lowercase soft dj

	COLS10+4,		// Ukrainian uppercase HARD G
	COLS10+4,		// Ukrainian lowercase hard g
	COLS0,			// GE bar
	COLS0,			// ge bar
	COLS10+6,		// Serbian uppercase SOFT DJ
	COLS10+6,		// Serbian lowercase SOFT DJ
	COLS0,			// IE (variant)
	COLS0,			// ie (variant)
	COLS10+10,		// Ukrainian uppercase YE
	COLS10+10,		// Ukrainian lowercase YE

	COLS0,			// ZHE with right descender
	COLS0,			// zhe with right descender
	COLS10+13,		// Macedonian uppercase ZELO
	COLS10+13,		// Macedonian lowercase ZELO
	COLS0,			// Old Slovanic uppercase Z
	COLS0,			// Old Slovanic uppercase z
	COLS0,			// II with macron
	COLS0,			// ii with mscron
	COLS10+15,		// Ukrainian uppercase I
	COLS10+15,		// Ukrainian lowercase I

	COLS10+16,		// Ukrainian uppercase I with Two Dots
	COLS10+16,		// Ukrainian lowercase I with Two Dots
	COLS0,			// Old Slovanic uppercase I ligature
	COLS0,			// Old Slovanic lowercase I ligature
	COLS10+18,		// Serbian--Macedonian uppercase JE
	COLS10+18,		// Serbian--Macedonian lowercase JE
	COLS10+31,		// Macedonian uppercase SOFT K
	COLS10+31,		// Macedonian lowercase SOFT K
	COLS0,			// KA with right descender
	COLS0,			// ka with right descender

	COLS0,			// KA ogonek
	COLS0,			// ka ogonek
	COLS0,			// KA vertical bar
	COLS0,			// ka vertical bar
	COLS10+21,		// Serbian--Macedonian uppercase SOFT L
	COLS10+21,		// Serbian--Macedonian lowercase SOFT L
	COLS0,			// EN with right descender
	COLS0,			// en with right descender
	COLS10+24,		// Serbian--Macedonian uppercase SOFT N
	COLS10+24,		// Serbian--Macedonian lowercase SOFT N

	COLS0,			// ROUND OMEGA
	COLS0,			// round omega
	COLS0,			// OMEGA
	COLS0,			// omega
	COLS10+30,		// Serbian uppercase SOFT T
	COLS10+30,		// Serbian lowercase SOFT T
	COLS10+33,		// Byelorussian uppercase SHORT U
	COLS10+33,		// Byelorussian lowercase SHORT U
	COLS0,			// U with macron
	COLS0,			// u with macron

	COLS0,			// STRAIGHT U
	COLS0,			// straight u
	COLS0,			// STRAIGHT U bar
	COLS0,			// straight u bar
	COLS0,			// OU ligature
	COLS0,			// ou ligature
	COLS0,			// KHA with right descender
	COLS0,			// kha with right descender
	COLS0,			// KHA ogonek
	COLS0,			// kha ogonek

	COLS0,			// H
	COLS0,			// h
	COLS0,			// OMEGA titlo
	COLS0,			// omega titlo
	COLS10+38,		// Serbian uppercase HARD DJ
	COLS10+38,		// Serbian lowercase HARD DJ
	COLS0,			// CHE with right descender
	COLS0,			// che with right descender
	COLS0,			// CHE vertical bar
	COLS0,			// che vertical bar

	COLS0,			// Old Slavonic SHCHA (variant)
	COLS0,			// old SLAVONIC shcha (variant)
	COLS10+44,		// Old Russian uppercase YAT
	COLS10+44,		// Old Russian lowercase YAT

// END OF UNIQUE COLLATED BYTES
// CHARACTERS BELOW MUST HAVE HAVE THEIR OWN
// SUB-COLLATION VALUE TO COMPARE CORRECTLY.

	COLS0,			// Old Bulgarian uppercase YUS
	COLS0,			// Old Bulgarian lowercase YUS
	COLS0,			// Old Slovanic uppercase YUS MALYI
	COLS0,			// Old Slovanic uppercase YUS MALYI
	COLS0,			// KSI
	COLS0,			// ksi

	COLS0,			// PSI
	COLS0,			// psi
	COLS0,			// Old Russian uppercase FITA
	COLS0,			// Old Russian lowercase FITA
	COLS0,			// Old Russian uppercase IZHITSA
	COLS0,			// Old Russian lowercase IZHITSA
	COLS0,			// Russian uppercase A acute
	COLS0,			// Russian lowercase A acute
	COLS10+8,		// Russian uppercase E acute
	COLS10+8,		// Russian lowercase E acute

// 160-below all characters are russian to 199

	COLS0,			// E acute
	COLS0,			// e acute
	COLS10+14,		// II acute
	COLS10+14,		// ii acute
	COLS0,			// I acute
	COLS0,			// i acute
	COLS0,			// YI acute
	COLS0,			// yi acute
	COLS10+25,		// O acute
	COLS10+25,		// o acute

	COLS10+32,		// U acute
	COLS10+32,		// u acute
	COLS10+42,		// YERI acute
	COLS10+42,		// YERI acute
	COLS10+45,		// REVERSED E acute
	COLS10+45,		// reversed e acute
	COLS10+46,		// YU acute
	COLS10+46,		// yu acute
	COLS10+47,		// YA acute
	COLS10+47,		// ya acute

	COLS10,			// A grave
	COLS10,			// a grave
	COLS10+8,		// E grave
	COLS10+8,		// e grave
	COLS10+9,		// YO grave
	COLS10+9,		// yo grave
	COLS10+14,		// I grave
	COLS10+14,		// i grave
	COLS10+25,		// O grave
	COLS10+25,		// o grave

	COLS10+32,		// U grave
	COLS10+32,		// u grave
	COLS10+42,		// YERI grave
	COLS10+42,		// yeri grave
	COLS10+45,		// REVERSED E grave
	COLS10+45,		// reversed e grave
	COLS10+46,		// IU (YU) grave
	COLS10+46,		// iu (yu) grave
	COLS10+47,		// ia (YA) grave
	COLS10+47,		// ia (ya) grave ******* [10,199]
};

/****************************************************************************
Desc:		The Hebrew characters are collated over the Russian characters
			Therefore sorting both Hebrew and Russian is impossible to do.
****************************************************************************/
FLMBYTE fwp_heb60TblA[ HEBTBL1LEN + 2] =
{
	0,					// starting offset
	HEBTBL1LEN,		// len of table
	COLS10h+0,		// Alef
	COLS10h+1,		// Bet
	COLS10h+2,		// Gimel
	COLS10h+3,		// Dalet
	COLS10h+4,		// He
	COLS10h+5,		// Vav
	COLS10h+6,		// Zayin
	COLS10h+7,		// Het
	COLS10h+8,		// Tet
	COLS10h+9,		// Yod
	COLS10h+10,		// Kaf (final) [9,10]
	COLS10h+11,		// Kaf
	COLS10h+12,		// Lamed
	COLS10h+13,		// Mem (final)
	COLS10h+14,		// Mem
	COLS10h+15,		// Nun (final)
	COLS10h+16,		// Nun
	COLS10h+17,		// Samekh
	COLS10h+18,		// Ayin
	COLS10h+19,		// Pe (final)
	COLS10h+20,		// Pe [9,20]
	COLS10h+21,		// Tsadi (final)
	COLS10h+22,		// Tsadi
	COLS10h+23,		// Qof
	COLS10h+24,		// Resh
	COLS10h+25,		// Shin
	COLS10h+26		// Tav [9,26]
};

/****************************************************************************
Desc:		This is the ANCIENT HEBREW SCRIPT piece.
			The actual value will be stored in the subcollation.
			This way we don't play diacritic/subcollation games.
****************************************************************************/
FLMBYTE fwp_heb60TblB[ HEBTBL2LEN + 2] =
{
	84,
	HEBTBL2LEN,

// [9,84]
	COLS10h+0,		// Alef Dagesh [9,84]
	COLS10h+1,		// Bet Dagesh
	COLS10h+1,		// Vez - looks like a bet
	COLS10h+2,		// Gimel Dagesh
	COLS10h+3,		// Dalet Dagesh
	COLS10h+4,		// He Dagesh
	COLS10h+5,		// Vav Dagesh [9,90]
	COLS10h+5,		// Vav Holem
	COLS10h+6,		// Zayin Dagesh
	COLS10h+7,		// Het Dagesh
	COLS10h+8,		// Tet Dagesh
	COLS10h+9,		// Yod Dagesh
	COLS10h+9,		// Yod Hiriq [9,96] - not on my list

	COLS10h+11,		// Kaf Dagesh
	COLS10h+10,		// Kaf Dagesh (final)
	COLS10h+10,		// Kaf Sheva (final)
	COLS10h+10,		// Kaf Tsere (final) [9,100]
	COLS10h+10,		// Kaf Segol (final)
	COLS10h+10,		// Kaf Patah (final)
	COLS10h+10,		// Kaf Qamats (final)
	COLS10h+10,		// Kaf Dagesh Qamats (final)
	COLS10h+12,		// Lamed Dagesh
	COLS10h+14,		// Mem Dagesh
	COLS10h+16,		// Nun Dagesh
	COLS10h+15,		// Nun Qamats (final)
	COLS10h+17,		// Samekh Dagesh
	COLS10h+20,		// Pe Dagesh [9,110]
	COLS10h+20,		// Fe - just guessing this is like Pe - was +21
	COLS10h+22,		// Tsadi Dagesh
	COLS10h+23,		// Qof Dagesh
	COLS10h+25,		// Sin (with sin dot)
	COLS10h+25,		// Sin Dagesh (with sin dot)
	COLS10h+25,		// Shin
	COLS10h+25,		// Shin Dagesh
	COLS10h+26		// Tav Dagesh [9,118]
};

/****************************************************************************
Desc:		The Arabic characters are collated OVER the Russian characters
			Therefore sorting both Arabic and Russian in the same database
			is not supported.

			Arabic starts with a bunch of accents/diacritic marks that are
			Actually placed OVER a preceeding character.  These accents are
			ignored while sorting the first pass - when collation == COLS0.

			There are 4 possible states for all/most arabic characters:
				?? - occurs as the only character in a word
				?? - appears at the first of the word
				?? - appears at the middle of a word
				?? - appears at the end of the word

			Usually only the simple version of the letter is stored.
			Therefore we should not have to worry about sub-collation
			of these characters.

			The arabic characters with diacritics differ however.  The alef has
			sub-collation values to sort correctly.  There is not any more room
			to add more collation values.  Some chars in CS14 are combined when
			urdu, pashto and sindhi characters overlap.
****************************************************************************/
FLMBYTE fwp_ar160Tbl[ AR1TBLLEN + 2] =
{
	38,				// starting offset
	AR1TBLLEN,		// len of table
// [13,38]
	COLLS+2,			// , comma
	COLLS+3,			// : colon
// [13,40]
	COLLS+7,			// ? question mark
	COLS4+2,			// * asterick
	COLS6,			// % percent
	COLS9+41,		// >> alphabetic - end of list)
	COLS9+40,		// << alphabetic - end of list)
	COLS2,			// (
	COLS2+1,			// )
// [13,47]
	COLS8+1,			// ?? One
	COLS8+2,			// ?? Two
	COLS8+3,			// ?? Three
// [13,50]
	COLS8+4,			// ?? Four
	COLS8+5,			// ?? Five
	COLS8+6,			// ?? Six
	COLS8+7,			// ?? Seven
	COLS8+8,			// ?? Eight
	COLS8+9,			// ?? Nine
	COLS8+0,			// ?? Zero
	COLS8+2,			// ?? Two (Handwritten)

	COLS10a+1,		// ?? alif
	COLS10a+1,		// ?? alif
// [13,60]
	COLS10a+2,		// ?? ba
	COLS10a+2,		// ?? ba
	COLS10a+2,		// ?? ba
	COLS10a+2,		// ?? ba
	COLS10a+6,		// ?? ta
	COLS10a+6,		// ?? ta
	COLS10a+6,		// ?? ta
	COLS10a+6,		// ?? ta
	COLS10a+8,		// ?? tha
	COLS10a+8,		// ?? tha
// [13,70]
	COLS10a+8,		// ?? tha
	COLS10a+8,		// ?? tha
	COLS10a+12,		// ?? jiim
	COLS10a+12,		// ?? jiim
	COLS10a+12,		// ?? jiim
	COLS10a+12,		// ?? jiim
	COLS10a+16,		// ?? Ha
	COLS10a+16,		// ?? Ha
	COLS10a+16,		// ?? Ha
	COLS10a+16,		// ?? Ha
// [13,80]
	COLS10a+17,		// ?? kha
	COLS10a+17,		// ?? kha
	COLS10a+17,		// ?? kha
	COLS10a+17,		// ?? kha
	COLS10a+20,		// ?? dal
	COLS10a+20,		// ?? dal
	COLS10a+22,		// ?? dhal
	COLS10a+22,		// ?? dhal
	COLS10a+27,		// ?? ra
	COLS10a+27,		// ?? ra
// [13,90]
	COLS10a+29,		// ?? ziin
	COLS10a+29,		// ?? ziin
	COLS10a+31,		// ?? siin
	COLS10a+31,		// ?? siin
	COLS10a+31,		// ?? siin
	COLS10a+31,		// ?? siin
	COLS10a+32,		// ?? shiin
	COLS10a+32,		// ?? shiin
	COLS10a+32,		// ?? shiin
	COLS10a+32,		// ?? shiin
// [13,100]
	COLS10a+34,		// ?? Sad
	COLS10a+34,		// ?? Sad
	COLS10a+34,		// ?? Sad
	COLS10a+34,		// ?? Sad
	COLS10a+35,		// ?? Dad
	COLS10a+35,		// ?? Dad
	COLS10a+35,		// ?? Dad
	COLS10a+35,		// ?? Dad
	COLS10a+36,		// ?? Ta
	COLS10a+36,		// ?? Ta
// [13,110]
	COLS10a+36,		// ?? Ta
	COLS10a+36,		// ?? Ta
	COLS10a+37,		// ?? Za
	COLS10a+37,		// ?? Za
	COLS10a+37,		// ?? Za
	COLS10a+37,		// ?? Za
	COLS10a+38,		// ?? 'ain
	COLS10a+38,		// ?? 'ain
	COLS10a+38,		// ?? 'ain
	COLS10a+38,		// ?? 'ain
// [13,120]
	COLS10a+39,		// ?? ghain
	COLS10a+39,		// ?? ghain
	COLS10a+39,		// ?? ghain
	COLS10a+39,		// ?? ghain
	COLS10a+40,		// ?? fa
	COLS10a+40,		// ?? fa
	COLS10a+40,		// ?? fa
	COLS10a+40,		// ?? fa
	COLS10a+42,		// ?? Qaf
	COLS10a+42,		// ?? Qaf
// [13,130]
	COLS10a+42,		// ?? Qaf
	COLS10a+42,		// ?? Qaf
	COLS10a+43,		// ?? kaf
	COLS10a+43,		// ?? kaf
	COLS10a+43,		// ?? kaf
	COLS10a+43,		// ?? kaf
	COLS10a+46,		// ?? lam
	COLS10a+46,		// ?? lam
	COLS10a+46,		// ?? lam
	COLS10a+46,		// ?? lam
// [13,140]
	COLS10a+47,		// ?? miim
	COLS10a+47,		// ?? miim
	COLS10a+47,		// ?? miim
	COLS10a+47,		// ?? miim
	COLS10a+48,		// ?? nuun
	COLS10a+48,		// ?? nuun
	COLS10a+48,		// ?? nuun
	COLS10a+48,		// ?? nuun
	COLS10a+49,		// ?? ha
	COLS10a+49,		// ?? ha
// [13,150]
	COLS10a+49,		// ?? ha
	COLS10a+49,		// ?? ha
						// ha is also 51 for non-arabic
	COLS10a+6, 		// ?? ta marbuuTah
	COLS10a+6, 		// ?? ta marbuuTah
	COLS10a+50,		// ?? waw
	COLS10a+50,		// ?? waw
	COLS10a+53,		// ?? ya
	COLS10a+53,		// ?? ya
	COLS10a+53,		// ?? ya
	COLS10a+53,		// ?? ya
// [13,160]
	COLS10a+52,		// ?? alif maqSuurah
	COLS10a+52,		// ?? ya   maqSuurah?
	COLS10a+52,		// ?? ya   maqSuurah?
	COLS10a+52,		// ?? alif maqSuurah

	COLS10a+0,		// ?? hamzah accent - never appears alone
// [13,165]

// Store the sub-collation as the actual
// character value from this point on

	COLS10a+1,		// ?? alif hamzah
	COLS10a+1,		// ?? alif hamzah
	COLS10a+1,		// ?? hamzah-under-alif
	COLS10a+1,		// ?? hamzah-under-alif
	COLS10a+1,		// ?? waw hamzah
// [13,170]
	COLS10a+1,		// ?? waw hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? ya hamzah
	COLS10a+1,		// ?? alif fatHataan
	COLS10a+1,		// ?? alif fatHataan
	COLS10a+1,		// ?? alif maddah
	COLS10a+1,		// ?? alif maddah
	COLS10a+1,		// ?? alif waSlah
// [13,180]
	COLS10a+1,		// ?? alif waSlah (final)

//  LIGATURES
//    Should NEVER be stored so will not worry
//    about breaking up into pieces for collation.
//  NOTE:
//    Let's store the "Lam" collation value (+42)
//    below and in the sub-collation store the
//    actual character.  This will sort real close.
//    The best implementation is to
//    break up ligatures into its base pieces.

	COLS10a+46,		// ?? lamalif
	COLS10a+46,		// ?? lamalif
	COLS10a+46,		// ?? lamalif hamzah
	COLS10a+46,		// ?? lamalif hamzah
	COLS10a+46,		// ?? hamzah-under-lamalif
	COLS10a+46,		// ?? hamzah-under-lamalif
	COLS10a+46,		// ?? lamalif fatHataan
	COLS10a+46,		// ?? lamalif fatHataan
	COLS10a+46,		// ?? lamalif maddah
// [13,190]
	COLS10a+46,		// ?? lamalif maddah
	COLS10a+46,		// ?? lamalif waSlah
	COLS10a+46,		// ?? lamalif waSlah
	COLS10a+46,		// ?? Allah - khaDalAlif
	COLS0_ARABIC,	// ?? taTwiil     - character extension - throw out
	COLS0_ARABIC	// ?? taTwiil 1/6 - character extension - throw out
};

/****************************************************************************
Desc:		Alef needs a subcollation table.
			If colval==COLS10a+1 & char>=165
			index through this table.  Otherwise
			the alef value is [13,58] and subcol
			value should be 7.  Alef maddah is default (0)

			Handcheck if colval==COLS10a+6
			Should sort:
				[13,152]..[13,153] - taa marbuuTah - nosubcoll
				[13,64] ..[13,67]  - taa - subcoll of 1
****************************************************************************/
FLMBYTE fwp_alefSubColTbl[] =
{
// [13,165]
	1,		// ?? alif hamzah
	1,		// ?? alif hamzah
	3,		// ?? hamzah-under-alif
	3,		// ?? hamzah-under-alif
	2,		// ?? waw hamzah
// [13,170]
	2,		// ?? waw hamzah
	4,		// ?? ya hamzah
	4,		// ?? ya hamzah
	4,		// ?? ya hamzah
	4,		// ?? ya hamzah
	5,		// ?? alif fatHataan
	5,		// ?? alif fatHataan
	0,		// ?? alif maddah
	0,		// ?? alif maddah
	6,		// ?? alif waSlah
// [13,180]
	6		// ?? alif waSlah (final)
};

/****************************************************************************
Desc:
****************************************************************************/
FLMBYTE fwp_ar260Tbl[ AR2TBLLEN + 2] =
{
	41,				// starting offset
	AR2TBLLEN,		// len of table
// [14,41]
	COLS8+4,			// Farsi and Urdu Four
	COLS8+4,			// Urdu Four
	COLS8+5,			// Farsi and Urdu Five
	COLS8+6,			// Farsi Six
	COLS8+6,			// Farsi and Urdu Six
	COLS8+7,			// Urdu Seven
	COLS8+8,			// Urdu Eight

	COLS10a+3,		// Sindhi bb - baa /w 2 dots below (67b)
	COLS10a+3,
	COLS10a+3,
	COLS10a+3,
	COLS10a+4,		// Sindhi bh - baa /w 4 dots below (680)
	COLS10a+4,
	COLS10a+4,
	COLS10a+4,
// [14,56]
	COLS10a+5,		// Malay, Kurdish, Pashto, Farsi, Sindhi, and Urdu p
	COLS10a+5,		// =peh - taa /w 3 dots below (67e)
	COLS10a+5,
	COLS10a+5,
	COLS10a+7,		// Urdu T - taa /w small tah
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,		// Pashto T - taa /w ring (forced to combine)
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,
	COLS10a+9,		// Sindhi th - taa /w 4 dots above (67f)
	COLS10a+9,
// [14,70]
	COLS10a+9,
	COLS10a+9,
	COLS10a+10,		// Sindhi Tr - taa /w 3 dots above (67d)
	COLS10a+10,
	COLS10a+10,
	COLS10a+10,
	COLS10a+11,		// Sindhi Th - taa /w 2 dots above (67a)
	COLS10a+11,
	COLS10a+11,
	COLS10a+11,
	COLS10a+13,		// Sindhi jj - haa /w 2 middle dots verticle (684)
	COLS10a+13,
	COLS10a+13,
	COLS10a+13,
	COLS10a+14,		// Sindhi ny - haa /w 2 middle dots (683)
	COLS10a+14,
	COLS10a+14,
	COLS10a+14,
// [14,88]
	COLS10a+15,		// Malay, Kurdish, Pashto, Farsi, Sindhi, and Urdu ch
	COLS10a+15,		// =tcheh (686)
	COLS10a+15,
	COLS10a+15,
	COLS10a+15,		// Sindhi chh - haa /w middle 4 dots (687)
	COLS10a+15,		// forced to combine
	COLS10a+15,
	COLS10a+15,
	COLS10a+18,		// Pashto ts - haa /w 3 dots above (685)
	COLS10a+18,
	COLS10a+18,
	COLS10a+18,
	COLS10a+19,		// Pashto dz - hamzah on haa (681)
	COLS10a+19,
	COLS10a+19,
	COLS10a+19,
// [14,104]
	COLS10a+21,		// Urdu D - dal /w small tah (688)
	COLS10a+21,
	COLS10a+21,		// Pashto D - dal /w ring (689) forced to combine
	COLS10a+21,
	COLS10a+23,		// Sindhi dh - dal /w 2 dots above (68c)
	COLS10a+23,
	COLS10a+24,		// Sindhi D - dal /w 3 dots above (68e)
	COLS10a+24,
	COLS10a+25,		// Sindhi Dr - dal /w dot below (68a)
	COLS10a+25,
	COLS10a+26,		// Sindhi Dh - dal /w 2 dots below (68d)
	COLS10a+26,
	COLS10a+28,		// Pashto r - ra /w ring (693)
	COLS10a+28,
// [14,118]
	COLS10a+28,		// Urdu R - ra /w small tah (691) forced to combine
	COLS10a+28,
	COLS10a+28,		// Sindhi r - ra /w 4 dots above (699) forced to combine
	COLS10a+28,
	COLS10a+27,		// Kurdish rolled r - ra /w 'v' below (695)
	COLS10a+27,
	COLS10a+27,
	COLS10a+27,
// [14,126]
	COLS10a+30,		// Kurdish, Pashto, Farsi, Sindhi, and Urdu Z
	COLS10a+30,		// = jeh - ra /w 3 dots above (698)
	COLS10a+30,		// Pashto zz - ra /w dot below & dot above (696)
	COLS10a+30,		// forced to combine
	COLS10a+30,		// Pashto g - not in unicode! - forced to combine
	COLS10a+30,
	COLS10a+33,		// Pashto x - seen dot below & above (69a)
	COLS10a+33,
	COLS10a+33,
	COLS10a+33,
	COLS10a+39,		// Malay ng - old maly ain /w 3 dots above (6a0)
	COLS10a+39,		// forced to combine
	COLS10a+39,
	COLS10a+39,
// [14,140]
	COLS10a+41,		// Malay p, Kurdish v - Farsi ? - fa /w 3 dots above
	COLS10a+41,		// = veh - means foreign words (6a4)
	COLS10a+41,
	COLS10a+41,
	COLS10a+41,		// Sindhi ph - fa /w 4 dots above (6a6) forced to combine
	COLS10a+41,
	COLS10a+41,
	COLS10a+41,
// [14,148]
	COLS10a+43,		// Misc k - open caf (6a9)
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,		// misc k - no unicode - forced to combine
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,		// Sindhi k - swash caf (various) (6aa) -forced to combine
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
// [14,160]
	COLS10a+44,		// Persian/Urdu g - gaf (6af)
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		// Persian/Urdu g - no unicode
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		// malay g - gaf /w ring (6b0)
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		// Sindhi ng  - gaf /w 2 dots above (6ba)
	COLS10a+44,		// forced to combine ng only
	COLS10a+44,
	COLS10a+44,
	COLS10a+45,		// Sindhi gg - gaf /w 2 dots vertical below (6b3)
	COLS10a+45,
	COLS10a+45,
	COLS10a+45,
// [14,180]
	COLS10a+46,		// Kurdish velar l - lam /w small v (6b5)
	COLS10a+46,
	COLS10a+46,
	COLS10a+46,
	COLS10a+46,		// Kurdish Lamalif with diacritic - no unicode
	COLS10a+46,
// [14,186]
	COLS10a+48,		// Urdu n - dotless noon (6ba)
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,		// Pashto N - noon /w ring (6bc) - forced to combine
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,		// Sindhi N - dotless noon/w small tah (6bb)
	COLS10a+48,		// forced to combine
	COLS10a+48,
	COLS10a+48,
	COLS10a+50,		// Kurdish o - waw /w small v (6c6)
	COLS10a+50,
// [14,200]
	COLS10a+50,		// Kurdish o - waw /w bar above (6c5)
	COLS10a+50,
	COLS10a+50,		// Kurdish o - waw /w 2 dots above (6ca)
	COLS10a+50,
// [14,204]
	COLS10a+51,		// Urdu h - no unicode
	COLS10a+51,
	COLS10a+51,
	COLS10a+51,
	COLS10a+52,		// Kurdish ? - ya /w small v (6ce)
	COLS10a+52,
	COLS10a+52,
	COLS10a+52,
// [14,212]
	COLS10a+54,		// Urdu y - ya barree (6d2)
	COLS10a+54,
	COLS10a+54,		// Malay ny - ya /w 3 dots below (6d1) forced to combine
	COLS10a+54,
	COLS10a+54,
	COLS10a+54,
// [14,218]
	COLS10a+51,		// Farsi hamzah - hamzah on ha (6c0) forced to combine
	COLS10a+51
};

/****************************************************************************
Desc:		If the bit position is set then save the character in the sub-col
			area.  The bit values are determined by looking at the
			FLAIM COLTBL1 to see which characters are combined with other
			Arabic characters.
****************************************************************************/
FLMBYTE fwp_ar2BitTbl[] =
{
	// Start at character 64
	// The only 'clean' areas uncollate to the correct place, they are...
						// 48..63
						// 68..91
						// 96..117
						// 126..127
						// 140..143
						// 160..163
						// 176..179
						// 212..213

	0xF0,				// 64..71
	0x00,				// 72..79
	0x00,				// 80..87
	0x0F,				// 88..95 - 92..95
	0x00,				// 96..103
	0x00,				// 104..111
	0x03,				// 112..119
	0xFC,				// 120..127
	0xFF,				// 128..135
	0xF0,				// 136..143 - 136..139
	0xFF,				// 144..151 - 144..147, 148..159
	0xFF,				// 152..159
	0x0F,				// 160..167 - 164..175
	0xFF,				// 168..175
	0x0F,				// 176..183 - 180..185
	0xFF,				// 184..191 - 186..197
	0xFF,				// 192..199 - 198..203
	0xFF,				// 200..207 - 204..207
	0xF3,				// 208..215 - 208..211 , 214..217
	0xF0				// 216..219 - 218..219
};

/****************************************************************************
Desc:		This table describes and gives addresses for collating 5.0
			character sets.  Each line corresponds with a character set.
***************************************************************************/
TBL_B_TO_BP fwp_col60Tbl[] =
{
	{CHSASCI, fwp_asc60Tbl},	// ascii - " " - "~"
	{CHSMUL1, fwp_mn60Tbl},		// multinational
	{CHSSYM1, fwp_sym60Tbl},	// symbols
	{CHSGREK, fwp_grk60Tbl},	// greek
	{CHSCYR,  fwp_cyrl60Tbl},	// Cyrillic - Russian
	{0xFF, 	 0}					// table terminator
};

/****************************************************************************
Desc:		This table is for sorting the hebrew/arabic languages.
			These values overlap the end of ASC/european and cyrillic tables.
****************************************************************************/
TBL_B_TO_BP fwp_HebArabicCol60Tbl[] =
{
	{CHSASCI,	fwp_asc60Tbl},		// ascii - " " - "~"
	{CHSMUL1,	fwp_mn60Tbl},		// multinational
	{CHSSYM1,	fwp_sym60Tbl},		// symbols
	{CHSGREK,	fwp_grk60Tbl},		// greek
	{CHSHEB,		fwp_heb60TblA},	// Hebrew
	{CHSHEB,		fwp_heb60TblB},	// Hebrew
	{CHSARB1,	fwp_ar160Tbl},		// Arabic Set 1
	{CHSARB2,	fwp_ar260Tbl},		// Arabic Set 2
	{0xff, 		0}						// table terminator
};

/****************************************************************************
Desc:		The diacritical to collated table translates the first 26
			characters of WP character set #1 into a 5 bit value for "correct"
			sorting sequence for that diacritical (DCV) - diacritic collated
			value.

			The attempt here is to convert the collated character value
			along with the DCV to form the original WP character.

			The diacriticals are in an order to fit the most languages.
			Czech, Swedish, and Finnish will have to manual reposition the
			ring above (assign it a value greater then the umlaut)

			This table is index by the diacritical value.
****************************************************************************/
FLMBYTE	fwp_dia60Tbl[] =
{
	2,			// grave		offset = 0
	16,		//	centerd	offset = 1
	7,			//	tilde		offset = 2
	4,			//	circum	offset = 3
	12,		//	crossb	offset = 4
	10,		//	slash		offset = 5
	1,			//	acute		offset = 6
	6,			//	umlaut	offset = 7
				// In SU, SV and CZ will = 9
	17,		//	macron	offset = 8
	18,		//	aposab	offset = 9
	19,		//	aposbes	offset = 10
	20,		//	aposba	offset = 11
	21,		//	aposbc	offset = 12
	22,		//	abosbl	offset = 13
	8,			// ring		offset = 14
	13,		//	dota		offset = 15
	23,		//	dacute	offset = 16
	11,		//	cedilla	offset = 17
	14,		//	ogonek	offset = 18
	5,			//	caron		offset = 19
	15,		//	stroke	offset = 20
	24,		//	bara 		offset = 21
	3,			//	breve		offset = 22
	0,			// dbls		offset = 23 sorts as 'ss'
	25,		//	dotlesi	offset = 24
	26			// dotlesj	offset = 25
};

/****************************************************************************
Desc:		This table defines the range of characters within the set
			which are case convertible.
****************************************************************************/
static FLMBYTE fwp_caseConvertableRange[] =
{
	26,241,		// Multinational 1
	0,0,			// Multinational 2
	0,0,			// Box Drawing
	0,0,			// Symbol 1
	0,0,			// Symbol 2
	0,0,			// Math 1
	0,0,			// Math 2
	0,69,			// Greek 1
	0,0,			// Hebrew
	0,199,		// Cyrillic
	0,0,			// Japanese Kana
	0,0,			// User-defined
	0,0,			// Not defined
	0,0,			// Not defined
	0,0,			// Not defined
};

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT16 colToWPChr[ COLS11 - COLLS] =
{
	0x20,			// colls	-	<Spc>
	0x2e,			// colls+1	-	.
	0x2c,			// colls+2	-	,
	0x3a,			// colls+3	-	:
	0x3b,			// colls+4	-	;
	0x21,			// colls+5	-	!
	0,				// colls+6	-	NO VALUE
	0x3f,			// colls+7	-	?
	0,				// colls+8	-	NO VALUE

	0x22,			// cols1		-	"
	0x27,			// cols1+1	-	'
	0x60,			// cols1+2	-	`
	0,				// cols1+3	-	NO VALUE
	0,				// cols1+4	-	NO VALUE

	0x28,			// cols2		-	(
	0x29,			// cols2+1	-	)
	0x5b,			// cols2+2	-	japanese angle brackets
	0x5d,			// cols2+3	-	japanese angle brackets
	0x7b,			// cols2+4	-	{
	0x7d,			// cols2+5	-	}

	0x24,			// cols3		-	$
	0x413,		// cols3+1	-	cent
	0x40b,		// cols3+2	-	pound
	0x40c,		// cols3+3	-	yen
	0x40d,		// cols3+4	-	pacetes
	0x40e,		// cols3+5	-	floren

	0x2b,			// cols4		-	+
	0x2d,			// cols4+1	-	-
	0x2a,			// cols4+2	-	*
	0x2f,			// cols4+3	-	/
	0x5e,			// cols4+4	-	^
	0,				// cols4+5	-	NO VALUE
	0,				// cols4+6	-	NO VALUE
	0,				// cols4+7	-	NO VALUE

	0x3c,			// cols5		-	<
	0,				// cols5+1	-	NO VALUE
	0x3d,			// cols5+2	-	=
	0,				// cols5+3	-	NO VALUE
	0x3e,			// cols5+4	-	>
	0,				// cols5+5	-	NO VALUE
	0,				// cols5+6	-	NO VALUE
	0,				// cols5+7	-	NO VALUE
	0,				// cols5+8	-	NO VALUE
	0,				// cols5+9	-	NO VALUE
	0,				// cols5+10	-	NO VALUE
	0,				// cols5+11	-	NO VALUE
	0,				// cols5+12	-	NO VALUE
	0,				// cols5+13	-	NO VALUE

	0x25,			// cols6		-	%
	0x23,			// cols6+1	-	#
	0x26,			// cols6+2	-	&
	0x40,			// cols6+3	-	@
	0x5c,			// cols6+4	-	Backslash
	0x5f,			// cols6+5	-	_
	0x7c,			// cols6+6	-	|
	0x7e,			// cols6+7	-	~
	0,				// cols6+8	- NO VALUE
	0,				// cols6+9	- NO VALUE
	0,				// cols6+10	- NO VALUE
	0,				// cols6+11	- NO VALUE
	0,				// cols6+12	- NO VALUE

	0x800,		// cols7		-	Uppercase Alpha
	0x802,		// cols7+1	-	Uppercase Beta
	0x806,		// cols7+2	-	Uppercase Gamma
	0x808,		// cols7+3	-	Uppercase Delta
	0x80a,		// cols7+4	-	Uppercase Epsilon
	0x80c,		// cols7+5	-	Uppercase Zeta
	0x80e,		// cols7+6	-	Uppercase Eta
	0x810,		// cols7+7	-	Uppercase Theta
	0x812,		// cols7+8	-	Uppercase Iota
	0x814,		// cols7+9	-	Uppercase Kappa
	0x816,		// cols7+10	-	Uppercase Lambda
	0x818,		// cols7+11	-	Uppercase Mu
	0x81a,		// cols7+12	-	Uppercase Nu
	0x81c,		// cols7+13	-	Uppercase Xi
	0x81e,		// cols7+14	-	Uppercase Omicron
	0x820,		// cols7+15	-	Uppercase Pi
	0x822,		// cols7+16	-	Uppercase Rho
	0x824,		// cols7+17	-	Uppercase Sigma
	0x828,		// cols7+18	-	Uppercase Tau
	0x82a,		// cols7+19	-	Uppercase Upsilon
	0x82c,		// cols7+20	-	Uppercase Phi
	0x82e,		// cols7+21	-	Uppercase Chi
	0x830,		// cols7+22	-	Uppercase Psi
	0x832,		// cols7+23	-	Uppercase Omega
	0,				// cols7+24 - NO VALUE

	0x30,			// cols8		-	0
	0x31,			// cols8+1	-	1
	0x32,			// cols8+2	-	2
	0x33,			// cols8+3	-	3
	0x34,			// cols8+4	-	4
	0x35,			// cols8+5	-	5
	0x36,			// cols8+6	-	6
	0x37,			// cols8+7	-	7
	0x38,			// cols8+8	-	8
	0x39,			// cols8+9	-	9

	0x41,			// cols9		-	A
	0x124,		// cols9+1	-	AE digraph
	0x42,			// cols9+2	-	B
	0x43,			// cols9+3	-	C
	0xffff,		// cols9+4	-	CH in spanish
	0x162,		// cols9+5	-	Holder for C caron in Czech
	0x44,			// cols9+6	-	D
	0x45,			// cols9+7	-	E
	0x46,			// cols9+8	-	F
	0x47,			// cols9+9	-	G
	0x48,			// cols9+10	-	H
	0xffff,		// cols9+11	-	CH in czech or dotless i in turkish
	0x49,			// cols9+12	-	I
	0x18a,		// cols9+13	-	IJ Digraph
	0x4a,			// cols9+14	-	J
	0x4b,			// cols9+15	-	K
	0x4c,			// cols9+16	-	L
	0xffff,		// cols9+17	-	LL in spanish
	0x4d,			// cols9+18	-	M
	0x4e,			// cols9+19	-	N
	0x138,		// cols9+20	-	N Tilde
	0x4f,			// cols9+21	-	O
	0x1a6,		// cols9+22	-	OE digraph
	0x50,			// cols9+23	-	P
	0x51,			// cols9+24	-	Q
	0x52,			// cols9+25	-	R
	0x1aa,		// cols9+26	-	Holder for R caron in Czech
	0x53,			// cols9+27	-	S
	0x1b0,		// cols9+28	-	Holder for S caron in Czech
	0x54,			// cols9+29	-	T
	0x55,			// cols9+30	-	U
	0x56,			// cols9+31	-	V

	0x57,			// cols9+32	-	W
	0x58,			// cols9+33	-	X
	0x59,			// cols9+34	-	Y
	0x5a,			// cols9+35	-	Z
	0x1ce,		// cols9+36	-	Holder for Z caron in Czech
	0x158,		// cols9+37	-	Uppercase Thorn
	0,				// cols9+38	-	???
	0,				// cols9+39	-	???
	0x5b,			// cols9+40	-	[ (note: alphabetic - end of list)
	0x5d,			// cols9+41	-	] (note: alphabetic - end of list)
// 0xAA - also start of Hebrew
	0x124,		// cols9+42	- AE diagraph - DK
	0x124,		// cols9+43 - AE diagraph - NO
	0x122,		// cols9+44 - A ring      - SW
	0x11E,		// cols9+45 - A diaeresis - DK
	0x124,		// cols9+46	- AE diagraph - IC
	0x150,		// cols9+47 - O slash     - NO
	0x11e,		// cols9+48	- A diaeresis - SW
	0x150,		// cols9+49	- O slash     - DK
	0x13E,		// cols9+50	- O Diaeresis - IC
	0x122,		// cols9+51	- A ring      - NO
	0x13E,		// cols9+52	- O Diaeresis - SW
	0x13E,		// cols9+53	- O Diaeresis - DK
	0x150,		// cols9+54 - O slash     - IC
	0x122,		// cols9+55	- A ring      - DK
	0x124,		// cols9+56	- AE diagraph future
	0x13E,		// cols9+57 - O Diaeresis future
	0x150,		// cols9+58 - O slash     future
	0,				// cols9+59 - NOT USED    future

	0xA00,		// cols10		-	Russian A
	0xA02,		// cols10+1		-	Russian BE
	0xA04,		// cols10+2		-	Russian VE
	0xA06,		// cols10+3		-	Russian GHE
	0xA46,		// cols10+4		-	Ukrainian HARD G
	0xA08,		// cols10+5		-	Russian DE
	0xA4a,		// cols10+6		-	Serbian SOFT DJ
	0xA44,		// cols10+7		-	Macedonian SOFT DJ
	0xA0a,		// cols10+8		-	Russian E
	0xA0c,		// cols10+9		-	Russian YO
	0xA4e,		// cols10+10	-	Ukrainian YE
	0xA0e,		// cols10+11	-	Russian ZHE
	0xA10,		// cols10+12	-	Russian ZE
	0xA52,		// cols10+13	-	Macedonian ZELO
	0xA12,		// cols10+14	-	Russian I
	0xA58,		// cols10+15	-	Ukrainian I
	0xA5a,		// cols10+16	-	Ukrainian I with Two dots
	0xA14,		// cols10+17	-	Russian SHORT I
	0xA5e,		// cols10+18	-	Serbian--Macedonian JE
	0xA16,		// cols10+19	-	Russian KA
	0xA18,		// cols10+20	-	Russian EL
	0xA68,		// cols10+21	-	Serbian--Macedonian SOFT L
	0xA1a,		// cols10+22	-	Russian EM
	0xA1c,		// cols10+23	-	Russian EN
	0xA6c,		// cols10+24	-	Serbian--Macedonian SOFT N
	0xA1e,		// cols10+25	-	Russian O
	0xA20,		// cols10+26	-	Russian PE
	0xA22,		// cols10+27	-	Russian ER
	0xA24,		// cols10+28	-	Russian ES
	0xA26,		// cols10+29	-	Russian TE
	0xA72,		// cols10+30	-	Serbian SOFT T
	0xA60,		// cols10+31	-	Macedonian SOFT K
	0xA28,		// cols10+32	-	Russian U
	0xA74,		// cols10+33	-	Byelorussian SHORT U
	0xA2a,		// cols10+34	-	Russian EF
	0xA2c,		// cols10+35	-	Russian HA
	0xA2e,		// cols10+36	-	Russian TSE
	0xA30,		// cols10+37	-	Russian CHE
	0xA86,		// cols10+38	-	Serbian HARD DJ
	0xA32,		// cols10+39	-	Russian SHA
	0xA34,		// cols10+40	-	Russian SHCHA
	0xA36,		// cols10+41	-	Russian ER (also hard
	0xA38,		// cols10+42	-	Russian ERY
	0xA3a,		// cols10+43	-	Russian SOFT SIGN
	0xA8e,		// cols10+44	-	Old Russian YAT
	0xA3c,		// cols10+45	-	Russian uppercase	REVERSE E
	0xA3e,		// cols10+46	-	Russian YU
	0xA40,		// cols10+47	-	Russian YA
	0xA3a,		// cols10+48	-	Russian SOFT SIGN - UKRAIN ONLY
 	0				// cols10+49	- future
};

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT16 HebArabColToWPChr[] =
{
	// Start at COLS10a+0
// [0]
	0x0D00 +164,	// hamzah
	0x0D00 + 58,	// [13,177] alef maddah
						// Read subcollation to get other alef values
	0x0D00 + 60,	// baa
	0x0E00 + 48,	// Sindhi bb
	0x0E00 + 52,	// Sindhi bh
	0x0E00 + 56,	// Misc p = peh
	0x0D00 +152,	// taa marbuuTah
						// subcollation of 1 is taa [13,64]
	0x0E00 + 60,	// Urdu T   [14,60]
						// Pashto T [14,64]
// [8]
	0x0D00 + 68,	// thaa
	0x0E00 + 68,	// Sindhi th
	0x0E00 + 72,	// Sindhi tr
	0x0E00 + 76,	// Sindhi Th
	0x0D00 + 72,	// jiim - jeem
	0x0E00 + 80,	// Sindhi jj
	0x0E00 + 84,	// Sindhi ny
	0x0E00 + 88,	// Misc ch
						// Sinhi chh [14,92]
// [16]
	0x0D00 + 76,	// Haa
	0x0D00 + 80,	// khaa
	0x0E00 + 96,	// Pashto ts
	0x0E00 +100,	// Pashto dz

	0x0D00 + 84,	// dal
	0x0E00 +104,	// Urdu D
						// Pashto D
	0x0D00 + 86,	// thal
	0x0E00 +108,	// Sindhi dh

// [24]
	0x0E00 +110,	// Sindhi D
	0x0E00 +112,	// Sindhi Dr
	0x0E00 +114,	// Sindhi Dh

	0x0D00 + 88,	// ra
						// Kurdish rolled r [14,122]
	0x0E00 +116,	// Pashto r [14,116] - must pick this!
						// Urdu R   [14,118]
						// Sindhi r [14,120]

	0x0D00 + 90,	// zain
	0x0E00 +126,	// Mizc Z=jeh [14,126]
						// Pashto zz  [14,128]
						// Pashto g   [14,130]

	0x0D00 + 92,	// seen

// [32]
	0x0D00 + 96,	// sheen
	0x0E00 +132,	// Pashto x
	0x0D00 +100,	// Sad
	0x0D00 +104,	// Dad
	0x0D00 +108,	// Tah
	0x0D00 +112,	// Za (dhah)
	0x0D00 +116,	// 'ain
	0x0D00 +120,	// ghain
						// malay ng [14,136]
// [40]
	0x0D00 +124,	// fa
	0x0E00 +140,	// Malay p, kurdish v = veh
						// Sindhi ph [14,144]
	0x0D00 +128,	// Qaf
	0x0D00 +132,	// kaf (caf)
						// Misc k  [14,148]
						// misc k - no unicode [14,152]
						// Sindhi k [14,156]

	0x0E00 +160,	// Persian/Urdu gaf
						// gaf - no unicode [14,164]
						// malay g [14,168]
						// Sindhi ng [14,172]
	0x0E00 +176,	// Singhi gg

	0x0D00 +136,	// lam - all ligature variants
						// Kurdish valar lam [14,180]
						// Kurdish lamalef - no unicode [14,184]

	0x0D00 +140,	// meem

// [48]
	0x0D00 +144,	// noon
						// Urdu n [14,186]
						// Pashto N [14,190]
						// Sindhi N [14,194]
	0x0D00 +148,	// ha - arabic language only!
	0x0D00 +154,	// waw
						// Kurdish o [14,198]
						// Kurdish o with bar [14,200]
						// Kurdish o with 2 dots [14,202]
	0x0D00 +148,	// ha - non-arabic language
						// Urdu h [14,204]
						// Farsi hamzah on ha [14,218]
	0x0D00 +160,	// alef maqsurah
						// Kurdish e - ya /w small v

	0x0D00 +156,	// ya
	0x0E00 +212		// Urdu ya barree
						// Malay ny [14,214]
};

/****************************************************************************
Desc:
****************************************************************************/
FLMUINT16 ArabSubColToWPChr[] =
{
	0x0D00 +177,	// Alef maddah - default value - here for documentation
	0x0D00 +165,	// Alef Hamzah
	0x0D00 +169,	// Waw hamzah
	0x0D00 +167,	// Hamzah under alef
	0x0D00 +171,	// ya hamzah
	0x0D00 +175,	// alef fathattan
	0x0D00 +179,	// alef waslah
	0x0D00 + 58,	// alef
	0x0D00 + 64		// taa - after taa marbuuTah
};

/****************************************************************************
Desc:		Turns a collated diacritic value into the original diacritic value
****************************************************************************/
FLMBYTE ml1_COLtoD[27] =
{
	23,		// dbls	sort value = 0  sorts as 'ss'
	6,			// acute	sort value = 1
	0,			// grave	sort value = 2
	22,		// breve	sort value = 3
	3,			// circum	sort value = 4
	19,		// caron	sort value = 5
	7,			// umlaut	sort value = 6
	2,			// tilde	sort value = 7
	14,		// ring	sort value = 8
	7,			// umlaut in SU,SV & CZ after ring = 9
	5,			// slash	sort value = 10
	17,	 	// cedilla	sort value = 11
	4,			// crossb	sort value = 12
	15,	 	// dota	sort value = 13
	18,	 	// ogonek	sort value = 14
	20,	 	// stroke	sort value = 15
	1, 	 	// centerd	sort value = 16
	8,			// macron	sort value = 17
	9,			// aposab	sort value = 18
	10,	 	// aposbes	sort value = 19
	11,	 	// aposba	sort value = 20
	12,	 	// aposbc	sort value = 21
	13,	 	// abosbl	sort value = 22
	16,	 	// dacute	sort value = 23
	21,	 	// bara 	sort value = 24
	24,	 	// dotlesi	sort value = 25
	25			// dotlesj	sort value = 26
};

/****************************************************************************
Desc:
Notes:		Only 48 values + 0x40, 0x41, 0x42 (169..171)
****************************************************************************/
FLMBYTE ColToKanaTbl[ 48] =
{
	 0,	// a=0, A=1
	 2,	// i=2, I=3
	 4,	// u=4, U=5, VU=83
	 6,	// e=6, E=7
 	 8,	// o=8, O=9
 	84,	// KA=10, GA=11, ka=84 - remember voicing table is optimized
 			//                       so that zero value is position and
 			//                       if voice=1 and no 0 is changed to 0
 	12,	// KI=12, GI=13
 	14,	// KU=14, GU=15
 	85,	// KE=16, GE=17, ke=85
 	18,	// KO=18, GO=19
 	20,	// SA=20, ZA=21
 	22,	// SHI=22, JI=23
 	24,	// SU=24, ZU=25
 	26,	// SE=26, ZE=27
 	28,	// SO=28, ZO=29
 	30,	// TA=30, DA=31
	32,	// CHI=32, JI=33
	34,	// tsu=34, TSU=35, ZU=36
	37,	// TE=37, DE=38
	39,	// TO=39, DO=40
	41,	// NA
	42,	// NI
	43,	// NU
	44,	// NE
	45,	// NO
	46,	// HA, BA, PA
	49,	// HI, BI, PI
	52,	// FU, BU, PU
	55,	// HE, BE, PE
	58,	// HO, BO, PO
	61,	// MA
	62,	// MI
	63,	// MU
	64,	// ME
	65,	// MO
	66,	// ya, YA
	68,	// yu, YU
	70,	// yo, YO
	72,	// RA
	73,	// RI
	74,	// RU
	75,	// RE
	76,	// RO
	77,	// wa, WA
	79,	// WI
	80,	// WE
	81,	// WO
	82		//  N
};

/****************************************************************************
Desc:
****************************************************************************/
static FLMBYTE fwp_langtbl[ LAST_LANG + LAST_LANG] =
{
	'U', 'S',	// English, United States
	'A', 'F',	// Afrikaans
	'A', 'R',	// Arabic
	'C', 'A',	// Catalan
	'H', 'R',	// Croatian
	'C', 'Z',	// Czech
	'D', 'K',	// Danish
	'N', 'L',	// Dutch
	'O', 'Z',	// English, Australia
	'C', 'E',	// English, Canada
	'U', 'K',	// English, United Kingdom
	'F', 'A',	// Farsi
	'S', 'U',	// Finnish
	'C', 'F',	// French, Canada
	'F', 'R',	// French, France
	'G', 'A',	// Galician
	'D', 'E',	// German, Germany
	'S', 'D',	// German, Switzerland
	'G', 'R',	// Greek
	'H', 'E',	// Hebrew
	'M', 'A',	// Hungarian
	'I', 'S',	// Icelandic
	'I', 'T',	// Italian
	'N', 'O',	// Norwegian
	'P', 'L',	// Polish
	'B', 'R',	// Portuguese, Brazil
	'P', 'O',	// Portuguese, Portugal
	'R', 'U',	// Russian
	'S', 'L',	// Slovak
	'E', 'S',	// Spanish
	'S', 'V',	// Swedish
	'Y', 'K',	// Ukrainian
	'U', 'R',	// Urdu
	'T', 'K',	// Turkey
	'J', 'P',	// Japanese
	'K', 'R',	// Korean
	'C', 'T',	// Chinese-Traditional
	'C', 'S',	// Chinese-Simplified
	'L', 'A'		// Future asian language
};

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT bytesInBits(
	FLMUINT		uiBits)
{
	return( (uiBits + 7) >> 3);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMBOOL testOneBit(
	const FLMBYTE *	pucBuf,
	FLMUINT				uiBit)
{
	return( (((pucBuf[ uiBit >> 3]) >> (7 - (uiBit & 7))) & 1)
		? TRUE
		: FALSE);
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE FLMUINT getNBits(
	FLMUINT				uiNumBits,
	const FLMBYTE *	pucBuf,
	FLMUINT				uiBit)
{
	return(((FLMUINT)(
		((FLMUINT)pucBuf[ uiBit >> 3] << 8) |		// append high bits (byte 1) to ...
		(FLMUINT)pucBuf[ (uiBit >> 3) + 1]) >>		// ... overflow bits in 2nd byte
		(16 - uiNumBits - (uiBit & 7))) &  			// reposition to low end of value
		((1 << uiNumBits) - 1));				  		// mask off high bits
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void setBit(
	FLMBYTE *	pucBuf,
	FLMUINT		uiBit)
{
	pucBuf[ uiBit >> 3] |= (FLMBYTE)(1 << (7 - (uiBit & 7)));
}

/****************************************************************************
Desc:
****************************************************************************/
FINLINE void setBits(
	FLMUINT		uiCount,
	FLMBYTE *	pucBuf,
	FLMUINT		uiBit,
	FLMUINT		uiVal)
{
	pucBuf[ uiBit >> 3] |= 		  					// 1st byte
			(FLMBYTE)((uiVal << (8 - uiCount)) 	// Align to bit 0
			>>
			(uiBit & 7)); 				  				// Re-align to actual bit position

	pucBuf[ (uiBit >> 3) + 1] = 					// 2nd byte
			(FLMBYTE)(uiVal
			<<
			(16 - uiCount - (uiBit & 7))); 		// Align spill-over bits
}

/****************************************************************************
Desc:		getNextCharState can be thought of as a 2 dimentional array with
			i and j as the row and column indicators respectively.  If a value
			exists at the intersection of i and j, it is returned.  Sparse array
			techniques are used to minimize memory usage.

Return:	0 = no valid next state
			non-zero = valid next state, offset for action, or collating value
****************************************************************************/
FINLINE FLMUINT16 getNextCharState(
	FLMUINT		i,
	FLMUINT		j)
{
	FLMUINT		k, x;

	for( k = fwp_indexi[ x =
			(i > START_COL) ? (START_ALL) : i ]; // adjust so don't use full tables
		  k <= (FLMUINT) (fwp_indexi[ x + 1] - 1);
		  k++ )
	{
		// FIXUP_AREA_SIZE should be 24.
		if(  j == fwp_indexj[ k])
		{
			return( fwp_valuea[ (i > START_COL)
				?	(k + (FIXUP_AREA_SIZE * (i - START_ALL)))
				: k]);
		}
	}

	return(0);
}

/****************************************************************************
Desc: Returns TRUE if the character is upper case, FALSE if lower case.
****************************************************************************/
FINLINE FLMBOOL charIsUpper(
	FLMUINT16	ui16Char)
{
	return( (FLMBOOL)((ui16Char < 0x7F)
							? (FLMBOOL)((ui16Char >= ASCII_LOWER_A &&
											 ui16Char <= ASCII_LOWER_Z)
											 ? (FLMBOOL)FALSE
											 : (FLMBOOL)TRUE)
							: flmWPIsUpper( ui16Char)));
}

/****************************************************************************
Desc:  	Convert a text string to a collated string.
			If NE_XFLM_CONV_DEST_OVERFLOW is returned the string is truncated as
			best as it can be.  The caller must decide to return the error up
			or deal with the truncation.
VISIT:	If the string is EXACTLY the length of the truncation
			length then it should, but doesn't, set the truncation flag.
			The code didn't match the design intent.  Fix next major
			version.
****************************************************************************/
RCODE flmUTF8ToColText(
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucCollatedStr,		// Returns collated string
	FLMUINT *			puiCollatedStrLen,	// Returns total collated string length
														// Input is maximum bytes in buffer
	FLMBOOL  			bCaseInsensitive,		// Set if to convert to uppercase
	FLMUINT *			puiCollationLen,		// Returns the collation bytes length
	FLMUINT *			puiCaseLen,				// Returns length of case bytes
	FLMUINT				uiLanguage,				// Language
	FLMUINT				uiCharLimit,			// Max number of characters in this key piece
	FLMBOOL				bFirstSubstring,		// TRUE is this is the first substring key
	FLMBOOL				bDataTruncated,		// TRUE if data is coming in truncated.
	FLMBOOL *			pbOriginalCharsLost,
	FLMBOOL *			pbDataTruncated)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT16	ui16Base;				// Value of the base character
	FLMUINT16	ui16SubColVal;			// Sub-collated value (diacritic)
	FLMUINT 		uiLength;				// Temporary variable for length
	FLMUINT 		uiTargetColLen = *puiCollatedStrLen - 8;	// 4=ovhd,4=worse char

	// Need to increase the buffer sizes to not overflow.
	// Characaters without COLL values will take up 3 bytes in
	// the ucSubColBuf[] and easily overflow the buffer.
	// Hard coded the values so as to minimize changes.

	FLMBYTE		ucSubColBuf[ MAX_SUBCOL_BUF + 301];	// Holds sub-collated values(diac)
	FLMBYTE		ucCaseBits[ MAX_CASE_BYTES + 81];	// Holds case bits
	FLMUINT16	ui16WpChr;			// Current WP character
	FLMUNICODE	uChar = 0;			// Current unconverted Unicode character
	FLMUNICODE	uChar2;
	FLMUINT16	ui16WpChr2;			// 2nd character if any; default 0 for US lang
	FLMUINT		uiColLen;			// Return value of collated length
	FLMUINT		uiSubColBitPos;	// Sub-collation bit position
	FLMUINT	 	uiCaseBitPos;		// Case bit position
	FLMUINT		uiFlags;				// Clear all bit flags
	FLMBOOL		bHebrewArabic = FALSE;	// Set if language is hebrew, arabic, farsi
	FLMBOOL		bTwoIntoOne;
	FLMUINT		uiUppercaseFlag;

	uiColLen = 0;
	uiSubColBitPos = 0;
	uiCaseBitPos = 0;
	uiFlags = 0;
	ui16WpChr2 = 0;

	// We don't want any single key piece to "pig out" more
	// than 256 bytes of the key

	if( uiTargetColLen > 256 - 8)
	{
		uiTargetColLen = 256 - 8;
	}

	// Code below sets ucSubColBuf[] and ucCaseBits[] values to zero.

	if (uiLanguage != XFLM_US_LANG)
	{
		if (uiLanguage == XFLM_AR_LANG ||		// Arabic
			 uiLanguage == XFLM_FA_LANG ||		// Farsi - persian
			 uiLanguage == XFLM_HE_LANG ||		// Hebrew
			 uiLanguage == XFLM_UR_LANG)			// Urdu
		{
			bHebrewArabic = TRUE;
		}
	}

	for (;;)
	{
		// Set the case bits and sub-collation bits to zero when
		// on the first bit of the byte.

		if (!(uiCaseBitPos & 0x07))
		{
			ucCaseBits [uiCaseBitPos >> 3] = 0;
		}
		if (!(uiSubColBitPos & 0x07))
		{
			ucSubColBuf [uiSubColBitPos >> 3] = 0;
		}

		ui16SubColVal = 0; // Default sub-collation value

		// Get the next character from the string.

		if( RC_BAD( rc = flmReadUTF8CharAsUnicode( pIStream, &uChar)))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
				break;
			}
			goto Exit;
		}

		// flmWPCheckDoubleCollation modifies ui16WpChr if a digraph or a double
		// character sequence is found.  If a double character is found, pucStr
		// is incremented past the next character and ui16WpChr2 is set to 1.
		// If a digraph is found, pucStr is not changed, but ui16WpChr
		// contains the first character and ui16WpChr2 contains the second
		// character of the digraph.

		if (uiLanguage != XFLM_US_LANG)
		{
			if( RC_BAD( rc = flmWPCheckDoubleCollation( 
				pIStream, FALSE, TRUE, &uChar, &uChar2, &bTwoIntoOne, uiLanguage)))
			{
				goto Exit;
			}
			if (!flmUnicodeToWP( uChar, &ui16WpChr))
			{
				ui16WpChr = UNK_UNICODE_CODE;
			}
			if (uChar2)
			{
				if (!flmUnicodeToWP( uChar2, &ui16WpChr2))
				{
					ui16WpChr2 = UNK_UNICODE_CODE;
				}
			}
			else
			{
				ui16WpChr2 = 0;
			}
		}
		else
		{

			// Convert the character to its WP equivalent

			if( !flmUnicodeToWP( uChar, &ui16WpChr))
			{
				ui16WpChr = UNK_UNICODE_CODE;
			}
		}

		// Save the case bit if not case-insensitive

		if (!bCaseInsensitive)
		{

			// charIsUpper returns TRUE if upper case, 0 if lower case.

			if (!charIsUpper( ui16WpChr))
			{
				uiFlags |= HAD_LOWER_CASE;
			}
			else
			{
				// Set if upper case.

				setBit( ucCaseBits, uiCaseBitPos);
			}
			uiCaseBitPos++;
		}

		// Handle non-collating characters with subcollating values,
		// Get the collated value from the WP character-if not collating value

		if ((pucCollatedStr[ uiColLen++] =
				(FLMBYTE)(flmWPGetCollation( ui16WpChr, uiLanguage))) >= COLS11)
		{
			FLMUINT	uiTemp;

			// If lower case, convert to upper case.

			if (!charIsUpper( ui16WpChr))
			{
				ui16WpChr &= ~1;
			}

			// No collating value given for this WP char.
			// Save original WP char (2 bytes) in subcollating
			// buffer.

			// 1110 is a new code that will store an insert over
			// the character OR a non-convertable unicode character.
			// Store with the same alignment as "store_extended_char"
			// below.

			// 11110 is code for unmappable UNICODE value.
			// A value 0xFE will be the collation value.  The sub-collation
			// value will be 0xFFFF followed by the UNICODE value.
			// Be sure to eat an extra case bit.

			// See specific Hebrew and Arabic comments in the
			//	switch statement below.

			// Set the next byte that follows in the sub collation buffer.

			ucSubColBuf [(uiSubColBitPos + 8) >> 3] = 0;
			if (bHebrewArabic && (pucCollatedStr[ uiColLen - 1] == COLS0_ARABIC))
			{
				// Store first bit of 1110, fall through & store remaining 3 bits

				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;

				// Don't store collation value

				uiColLen--;
			}
			else if( uChar)
			{
				ui16WpChr = uChar;
				uChar = 0;

				// Store 11 out of 11110

				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				if (!bCaseInsensitive)
				{
					ucCaseBits [(uiCaseBitPos + 7) >> 3] = 0;

					// Set upper case bit.

					setBit( ucCaseBits, uiCaseBitPos);
					uiCaseBitPos++;
				}
			}
store_extended_char:

			// Set the next byte that follows in the sub collation buffer.

			ucSubColBuf [(uiSubColBitPos + 8) >> 3] = 0;
			ucSubColBuf [(uiSubColBitPos + 16) >> 3] = 0;
			uiFlags |= HAD_SUB_COLLATION;

			// Set 110 bits in sub-collation - continued from above.
			// No need to explicitly set the zero, but must increment
			// for it.

			setBit( ucSubColBuf, uiSubColBitPos);
			uiSubColBitPos++;
			setBit( ucSubColBuf, uiSubColBitPos);
			uiSubColBitPos += 2;

			// store_aligned_word: This label is not referenced.
			// Go to the next byte boundary to write the character.

			uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
			uiTemp = bytesInBits( uiSubColBitPos);

			// Need to big-endian - so it will sort correctly.

			ucSubColBuf [uiTemp] = (FLMBYTE)(ui16WpChr >> 8);
			ucSubColBuf [uiTemp + 1] = (FLMBYTE)(ui16WpChr);
			uiSubColBitPos += 16;
			ucSubColBuf [uiSubColBitPos >> 3] = 0;
		}
		else
		{
			// Had a collation value
			// Add the lower/uppercase bit if a mixed case output.
			// If not lower ASCII set - check diacritic value for sub-collation

			if( !(ui16WpChr & 0xFF00))
			{
				// ASCII character set - set a single 0 bit - just need to
				// increment to do this.

				uiSubColBitPos++;
			}
			else
			{
				FLMBYTE	ucChar = (FLMBYTE)ui16WpChr;
				FLMBYTE	ucCharSet = (FLMBYTE)(ui16WpChr >> 8);

				// Convert char to uppercase because case information
				// is stored above.  This will help
				// ensure that the "ETA" doesn't sort before "eta"

				if( !charIsUpper( ui16WpChr))
				{
					ui16WpChr &= ~1;
				}

				switch( ucCharSet)
				{
					case CHSMUL1:	// Multinational 1
					{
						// If we cannot break down a char into base and
						// diacritic we cannot combine the charaacter
						// later when converting back the key.  In that case,
						// write the entire WP char in the sub-collation area.

						if( flmWPBrkcar( ui16WpChr, &ui16Base, &ui16SubColVal))
						{
							goto store_extended_char;
						}

						// Write the FLAIM diacritic sub-collation value.
						// Prefix is 2 bits "10".  Remember to leave
						// "111" alone for the future.
						// NOTE: The "unlaut" character must sort after the "ring"
						// character.

						ui16SubColVal = ((ui16SubColVal & 0xFF) == umlaut	&&
											  (uiLanguage == XFLM_SU_LANG ||
												uiLanguage == XFLM_SV_LANG ||
												uiLanguage == XFLM_CZ_LANG ||
												uiLanguage == XFLM_SL_LANG))
							?	(FLMUINT16)(fwp_dia60Tbl[ ring] + 1)
							:	(FLMUINT16)(fwp_dia60Tbl[ ui16SubColVal & 0xFF]);

store_sub_col:
						// Set the next byte that follows in the sub collation buffer.

						ucSubColBuf[ (uiSubColBitPos + 8) >> 3] = 0;
						uiFlags |= HAD_SUB_COLLATION;

						// Set the 10 bits - no need to explicitly set the zero, but
						// must increment for it.

						setBit( ucSubColBuf, uiSubColBitPos);
						uiSubColBitPos += 2;

						// Set sub-collation bits.

						setBits( 5, ucSubColBuf, uiSubColBitPos, ui16SubColVal);
						uiSubColBitPos += 5;
						break;
					}

					case CHSGREK:		// Greek
					{
						if (ucChar >= 52  ||			// Keep case bit for 52-69 else ignore
          				 ui16WpChr == 0x804 ||	// [ 8,4] BETA Medial | Terminal
							 ui16WpChr == 0x826) 	// [ 8,38] SIGMA terminal
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					case CHSCYR:
					{
						if (ucChar >= 144)
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;

						// Georgian covers 208-249 - no collation defined yet

						break;
					}

					case CHSHEB:		// Hebrew
					{
						// Three sections in Hebrew:
						//		0..26 - main characters
						//		27..83 - accents that apear over previous character
						//		84..118- dagesh (ancient) hebrew with accents

						// Because the ancient is only used for sayings & scriptures
						// we will support a collation value and in the sub-collation
						// store the actual character because sub-collation is in
						// character order.

            		if (ucChar >= 84)		// Save ancient - value 84 and above
						{
							goto store_extended_char;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					case CHSARB1:		// Arabic 1
					{
						// Three sections in Arabic:
						//		00..37  - accents that display OVER a previous character
						//		38..46  - symbols
						//		47..57  - numbers
						//		58..163 - characters
						//		164     - hamzah accent
						//		165..180- common characters with accents
						//		181..193- ligatures - common character combinations
						//		194..195- extensions - throw away when sorting

						if( ucChar <= 46)
						{
							goto store_extended_char;	// save original character
						}

						if( pucCollatedStr[ uiColLen - 1] == COLS10a + 1) // Alef?
						{
							ui16SubColVal = (ucChar >= 165)
								? (FLMUINT16)(fwp_alefSubColTbl[ ucChar - 165 ])
								: (FLMUINT16)7;			// Alef subcol value
							goto store_sub_col;
						}

						if (ucChar >= 181)			// Ligatures - char combination
						{
							goto store_extended_char;	// save original character
						}

						if (ucChar == 64)				// taa exception
						{
							ui16SubColVal = 8;
							goto store_sub_col;
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					case CHSARB2:			// Arabic 2
					{
						// There are some characters that share the same slot
						// Check the bit table if above character 64

						if (ucChar >= 64 &&
							 fwp_ar2BitTbl[(ucChar-64)>> 3] & (0x80 >> (ucChar&0x07)))
						{
							goto store_extended_char;	// Will save original
						}

						// No subcollation to worry about - set a zero bit by
						// incrementing the bit position.

						uiSubColBitPos++;
						break;
					}

					default:
					{
						// Increment bit position to set a zero bit.

						uiSubColBitPos++;
						break;
					}
				}
			}

			// Now let's worry about double character sorting

			if (ui16WpChr2)
			{
				if (pbOriginalCharsLost)
				{
					*pbOriginalCharsLost = TRUE;
				}

				// Set the next byte that follows in the sub collation buffer.

				ucSubColBuf[ (uiSubColBitPos + 7) >> 3] = 0;

				if (bTwoIntoOne)
				{

					// Sorts after character in ui16WpChr after call to
					// flmWPCheckDoubleCollation
					// Write the char 2 times so lower/upper bits are correct.
					// Could write infinite times because of collation rules.

					pucCollatedStr[ uiColLen] = ++pucCollatedStr[ uiColLen - 1];
					uiColLen++;

					// If original was upper case, set one more upper case bit

					if( !bCaseInsensitive)
					{
						ucCaseBits[ (uiCaseBitPos + 7) >> 3] = 0;
						if( !charIsUpper( ui16WpChr2))
						{
							uiFlags |= HAD_LOWER_CASE;
						}
						else
						{
							setBit( ucCaseBits, uiCaseBitPos);
						}
						uiCaseBitPos++;
					}

					// Take into account the diacritical space

					uiSubColBitPos++;
				}
				else
				{

					// We have a digraph, get second collation value

					pucCollatedStr[ uiColLen++] =
						(FLMBYTE)(flmWPGetCollation( ui16WpChr2, uiLanguage));

					// Normal case, assume no diacritics set

					uiSubColBitPos++;

					// If first was upper, set one more upper bit.

					if( !bCaseInsensitive)
					{
						ucCaseBits [(uiCaseBitPos + 7) >> 3] = 0;
						if (charIsUpper( ui16WpChr))
						{
							setBit( ucCaseBits, uiCaseBitPos);
						}
						uiCaseBitPos++;

						// no need to reset the uiFlags
					}
				}
			}
		}

		// Check to see if uiColLen is at some overflow limit.

		if (uiColLen >= uiCharLimit ||
			 uiColLen + bytesInBits( uiSubColBitPos) +
						  bytesInBits( uiCaseBitPos) >= uiTargetColLen)
		{

			// We hit the maximum number of characters.  See if we hit the
			// end of the string.

			if (RC_BAD( rc = flmReadUTF8CharAsUnicode( pIStream, &uChar)))
			{
				if (rc == NE_XFLM_EOF_HIT)
				{
					rc = NE_XFLM_OK;
				}
				else
				{
					goto Exit;
				}
			}
			else
			{
				bDataTruncated = TRUE;
			}
			break;
		}
	}

	if (puiCollationLen)
	{
		*puiCollationLen = uiColLen;
	}

	// Add the first substring marker - also serves as making the string non-null.

	if (bFirstSubstring)
	{
		pucCollatedStr[ uiColLen++] = COLL_FIRST_SUBSTRING;
	}

	if (bDataTruncated)
	{
		pucCollatedStr[ uiColLen++ ] = COLL_TRUNCATED;
	}

	// Return NOTHING if no values found

	if (!uiColLen && !uiSubColBitPos)
	{
		if (puiCaseLen)
		{
			*puiCaseLen = 0;
		}
		goto Exit;
	}

	// Store extra zero bit in the sub-collation area for Hebrew/Arabic

	if (bHebrewArabic)
	{
		uiSubColBitPos++;
	}

	// Done putting the string into 4 sections - build the COLLATED KEY
	// Don't set uiUppercaseFlag earlier than here because SC_LOWER may be zero

	uiUppercaseFlag = (uiLanguage == XFLM_GR_LANG)
								? SC_LOWER
								: SC_UPPER;

	// Did we write anything to the subcollation area?
	// The default terminating characters is (COLL_MARKER|SC_UPPER)

	if (uiFlags & HAD_SUB_COLLATION)
	{
		// Writes out a 0x7

		pucCollatedStr[ uiColLen++] = COLL_MARKER | SC_SUB_COL;

		// Move the sub-collation into the collating string

		uiLength = bytesInBits( uiSubColBitPos);
		f_memcpy( &pucCollatedStr[ uiColLen], ucSubColBuf, uiLength);
		uiColLen += uiLength;
	}

	// Move the upper/lower case stuff - force bits for Greek ONLY
	// This is such a small size that a memcpy is not worth it

	if( uiFlags & HAD_LOWER_CASE)
	{
		FLMUINT		uiNumBytes = bytesInBits( uiCaseBitPos);
		FLMBYTE *	pucCasePtr = ucCaseBits;

		// Output the 0x5

		pucCollatedStr[ uiColLen++] = (FLMBYTE)(COLL_MARKER | SC_MIXED);
		if( puiCaseLen)
		{
			*puiCaseLen = uiNumBytes + 1;
		}

		if( uiUppercaseFlag == SC_LOWER)
		{
			// Negate case bits for languages (like GREEK) that sort
			// upper case before lower case.

			while( uiNumBytes--)
			{
				pucCollatedStr[ uiColLen++] = ~(*pucCasePtr++);
			}
		}
		else
		{
			while( uiNumBytes--)
			{
				pucCollatedStr[ uiColLen++] = *pucCasePtr++;
			}
		}
	}
	else
	{
		// All characters are either upper or lower case, as determined
		// by uiUppercaseFlag.

		pucCollatedStr[ uiColLen++] = (FLMBYTE)(COLL_MARKER | uiUppercaseFlag);
		if( puiCaseLen)
		{
			*puiCaseLen = 1;
		}
	}

Exit:

	if( pbDataTruncated)
	{
		*pbDataTruncated = bDataTruncated;
	}

	*puiCollatedStrLen = uiColLen;
	return( rc);
}

/**************************************************************************
Desc:		Get the Flaim collating string and convert back to a text string
Ret:		Length of new wpStr
Notes:	Allocates the area for the word string buffer if will be over 256.
***************************************************************************/
RCODE flmColText2StorageText(
	const FLMBYTE *	pucColStr,				// Points to the collated string
	FLMUINT				uiColStrLen,			// Length of the collated string
	FLMBYTE *			pucStorageBuf,			// Output string to build - TEXT string
	FLMUINT *			puiStorageLen,			// In: Size of buffer, Out: Bytes used
	FLMUINT	   		uiLang,
	FLMBOOL *			pbDataTruncated,		// Sets to TRUE if data had been truncated
	FLMBOOL *			pbFirstSubstring)		// Sets to TRUE if first substring
{
#define LOCAL_CHARS		150
	FLMBYTE		ucWPStr[ LOCAL_CHARS * 2 + LOCAL_CHARS / 5 ];	// Sample + 20%
	FLMBYTE *  	pucWPPtr = NULL;
	FLMBYTE *	pucAllocatedWSPtr = NULL;
	FLMUINT		uiWPStrLen;
	FLMBYTE *	pucStoragePtr;
	FLMUINT		uiUnconvChars;
	FLMUINT		uiTmp;
	FLMUINT		uiMaxStorageBytes = *puiStorageLen;
	FLMUINT		uiMaxWPBytes;
	FLMUINT		uiStorageOffset;
	FLMBYTE		ucTmpSen[ 5];
	FLMBYTE *	pucTmpSen = &ucTmpSen[ 0];
	RCODE			rc = NE_XFLM_OK;

	if( uiColStrLen > LOCAL_CHARS)
	{
		// If it won't fit, allocate a new buffer

		if( RC_BAD( rc = f_alloc( XFLM_MAX_KEY_SIZE * 2, &pucWPPtr)))
		{
			goto Exit;
		}

		pucAllocatedWSPtr = pucWPPtr;
		uiMaxWPBytes = uiWPStrLen = XFLM_MAX_KEY_SIZE * 2;
	}
	else
	{
		pucWPPtr = &ucWPStr[ 0];
		uiMaxWPBytes = uiWPStrLen = sizeof( ucWPStr);
	}

 	if( (uiLang >= FIRST_DBCS_LANG) &&
 		 (uiLang <= LAST_DBCS_LANG))
 	{
		if( RC_BAD( rc = flmAsiaColStr2WPStr( pucColStr, uiColStrLen,
			pucWPPtr, &uiWPStrLen, &uiUnconvChars,
			pbDataTruncated, pbFirstSubstring)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = flmColStr2WPStr( pucColStr, uiColStrLen,
			pucWPPtr, &uiWPStrLen, uiLang, &uiUnconvChars,
			pbDataTruncated, pbFirstSubstring)))
		{
			goto Exit;
		}
	}

	// Copy word string to the storage string area

	uiWPStrLen >>= 1;	// Convert # of bytes to # of words
	pucStoragePtr = pucStorageBuf;
	uiStorageOffset = 0;

	// Encode the number of characters as a SEN.  If pucEncPtr is
	// NULL, the caller is only interested in the length of the encoded
	// string, so a temporary buffer is used to call flmEncodeSEN.

	uiTmp = flmEncodeSEN( uiWPStrLen - uiUnconvChars, &pucTmpSen);
	if( (uiStorageOffset + uiTmp) >= uiMaxStorageBytes)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}
	f_memcpy( pucStoragePtr, &ucTmpSen[ 0], uiTmp);
	uiStorageOffset += uiTmp;

	// Encode each of the WP characters into UTF-8

	while( uiWPStrLen--)
	{
		FLMBYTE			ucChar;
		FLMBYTE			ucCharSet;
		FLMUNICODE		uChar;

		// Put the character in a local variable for speed

		ucChar = *pucWPPtr++;
		ucCharSet = *pucWPPtr++;

		if( ucCharSet == 0xFF && ucChar == 0xFF)
		{
			uChar = (((FLMUNICODE)*(pucWPPtr + 1)) << 8) | *pucWPPtr;
			pucWPPtr += 2;
			uiWPStrLen--; // Skip past 4 bytes for UNICODE
		}
		else
		{
			if( RC_BAD( rc = flmWPToUnicode(
				(((FLMUINT16)ucCharSet) << 8) + ucChar, &uChar)))
			{
				goto Exit;
			}
		}

		uiTmp = uiMaxStorageBytes - uiStorageOffset;
		if( RC_BAD( rc = flmUni2UTF8( uChar,
			&pucStorageBuf[ uiStorageOffset], &uiTmp)))
		{
			goto Exit;
		}
		uiStorageOffset += uiTmp;
	}

	if( uiStorageOffset >= uiMaxStorageBytes)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	// Tack on a trailing NULL byte

	pucStorageBuf[ uiStorageOffset++] = 0;

	// Return the length of the storage buffer

	*puiStorageLen = uiStorageOffset;

Exit:

	if( pucAllocatedWSPtr)
	{
		f_free( &pucAllocatedWSPtr);
	}

	return( rc);
}

/*****************************************************************************
Desc:		Convert a collated string to a WP word string
*****************************************************************************/
FSTATIC RCODE flmColStr2WPStr(
	const FLMBYTE *	pucColStr,			  	// Points to the collated string
	FLMUINT				uiColStrLen,		  	// Length of the collated string
	FLMBYTE *			pucWPStr,			  	// Output string to build - WP word string
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiLang,
	FLMUINT *			puiUnconvChars,
	FLMBOOL *			pbDataTruncated,		// Set to TRUE if truncated
	FLMBOOL *			pbFirstSubstring)		// Sets to TRUE if first substring
{
	FLMBYTE *	pucWPPtr = pucWPStr;			// Points to the word string data area
	FLMBYTE *	pucWPEnd = &pucWPPtr[ *puiWPStrLen];
	FLMUINT		uiMaxWPBytes = *puiWPStrLen;
	FLMUINT		uiLength = uiColStrLen;		// May optimize as a register
	FLMUINT		uiPos = 0;						// Position in pucColStr
	FLMUINT		uiBitPos;						// Computed bit position
	FLMUINT		uiColChar;						// Not portable if a FLMBYTE value
	FLMUINT		uiWPStrLen;
	FLMUINT		uiUnconvChars = 0;
	FLMBOOL		bHebrewArabic = FALSE;
	RCODE			rc = NE_XFLM_OK;

	//  WARNING:
	//  The code is duplicated for performance reasons.
	//  The US code below is much more optimized so
	//  any changes must be done twice.

	if( uiLang == XFLM_US_LANG)
	{
		while( uiLength && (pucColStr[ uiPos] > MAX_COL_OPCODE))
		{
			uiLength--;

			// Move in the WP value given uppercase collated value

			uiColChar = (FLMUINT)pucColStr[ uiPos++];
			if( uiColChar == COLS0)
			{
				uiColChar = (FLMUINT)0xFFFF;
				uiUnconvChars++;
			}
			else
			{
				uiColChar = (FLMUINT)colToWPChr[ uiColChar - COLLS];
			}

			// Put the WP char in the word string

			if( pucWPPtr + 2 >= pucWPEnd)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			UW2FBA( (FLMUINT16)uiColChar, pucWPPtr);
			pucWPPtr += 2;
		}
	}
	else // Non-US collation
	{
		if( (uiLang == XFLM_AR_LANG ) ||		// Arabic
			 (uiLang == XFLM_FA_LANG ) ||		// Farsi - Persian
			 (uiLang == XFLM_HE_LANG ) ||		// Hebrew
			 (uiLang == XFLM_UR_LANG))			// Urdu
		{
			bHebrewArabic = TRUE;
		}

		while( uiLength && (pucColStr[ uiPos] > MAX_COL_OPCODE))
		{
			uiLength--;
			uiColChar = (FLMUINT)pucColStr[ uiPos++];

			switch( uiColChar)
			{
				case COLS9+4:		// ch in spanish
				case COLS9+11:		// ch in czech
				{
					// Put the WP char in the word string

					if( pucWPPtr + 2 >= pucWPEnd)
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					UW2FBA( (FLMUINT16) 'C', pucWPPtr);
					pucWPPtr += 2;
					uiColChar = (FLMUINT)'H';
					uiPos++;	// Move past second duplicate char
					break;
				}

				case COLS9+17:		// ll in spanish
				{
					// Put the WP char in the word string

					if( pucWPPtr + 2 >= pucWPEnd)
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					UW2FBA( (FLMUINT16)'L', pucWPPtr);
					pucWPPtr += 2;
					uiColChar = (FLMUINT)'L';
					uiPos++;	// Move past duplicate character
					break;
				}

				case COLS0:			// Non-collating character or OEM character
				{
					// Actual character is in sub-collation area

					uiColChar = (FLMUINT)0xFFFF;
					uiUnconvChars++;
					break;
				}

				default:
				{
					// Watch out COLS10h has () around it for subtraction

					if( bHebrewArabic && (uiColChar >= COLS10h))
					{
						uiColChar = (uiColChar < COLS10a)	// Hebrew only?
					 			? (FLMUINT) (0x900 + (uiColChar - (COLS10h)))	// Hebrew
					 			: (FLMUINT) (HebArabColToWPChr[ uiColChar - (COLS10a)]);	// Arabic
					}
					else
					{
						uiColChar = (FLMUINT)colToWPChr[ uiColChar - COLLS];
					}
					break;
				}
			}

			// Put the WP char in the word string

			if( pucWPPtr + 2 >= pucWPEnd)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			UW2FBA( (FLMUINT16)uiColChar, pucWPPtr);
			pucWPPtr += 2;
		}
	}

	// Terminate the string

	if( pucWPPtr + 2 >= pucWPEnd)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	UW2FBA( (FLMUINT16)0, pucWPPtr);
	uiWPStrLen = uiPos + uiPos;	// Multiply by 2

	// Parse through the sub-collation and case information.
	//  Here are values for some of the codes:
	//   [ 0x04] - case information is all uppercase (IS,DK,GR)
	//   [ 0x05] - case bits follow
	//   [ 0x06] - case information is all uppercase
	//   [ 0x07] - beginning of sub-collation information
	//   [ 0x08] - first substring field that is made
	//   [ 0x09] - truncation marker for text and binary
	//
	//  Below are some cases to consider...
	//
	// [ COLLATION][ 0x07 sub-collation][ 0x05 case info]
	// [ COLLATION][ 0x07 sub-collation][ 0x05 case info]
	// [ COLLATION][ 0x07 sub-collation]
	// [ COLLATION][ 0x07 sub-collation]
	// [ COLLATION][ 0x05 case info]
	// [ COLLATION][ 0x05 case info]
	// [ COLLATION]
	// [ COLLATION]
	//
	//  In the future still want[ 0x06] to be compressed out for uppercase
	//  only indexes.

	// Check first substring before truncated

	if( uiLength && pucColStr[ uiPos] == COLL_FIRST_SUBSTRING)
	{
		if( pbFirstSubstring)
		{
			*pbFirstSubstring = TRUE;	// Don't need to initialize to FALSE.
		}
		uiLength--;
		uiPos++;
	}

	// Is the key truncated?

	if( uiLength && pucColStr[ uiPos] == COLL_TRUNCATED)
	{
		if( pbDataTruncated)
		{
			*pbDataTruncated = TRUE;	// Don't need to initialize to FALSE.
		}
		uiLength--;
		uiPos++;
	}

	// Does sub-collation follow?
	// Still more to process - first work on the sub-collation (diacritics)
	// Hebrew/Arabic may have empty collation area

	if( uiLength && (pucColStr[ uiPos] == (COLL_MARKER | SC_SUB_COL)))
	{
		FLMUINT	uiTempLen;

		// Do another pass on the word string adding the diacritics

		if( RC_BAD( rc = flmWPCmbSubColBuf( pucWPStr, &uiWPStrLen, uiMaxWPBytes,
			&pucColStr[ ++uiPos], bHebrewArabic, &uiBitPos)))
		{
			goto Exit;
		}

		// Move pos to next byte value

		uiTempLen = bytesInBits( uiBitPos);
		uiPos += uiTempLen;
		uiLength -= uiTempLen + 1; // The 1 includes the 0x07 byte
	}

	// Does the case info follow?

	if( uiLength && (pucColStr[ uiPos] >= 0x04))
	{
		// Take care of the lower and upper case conversion
		// If mixed case then convert using case bits

		if( pucColStr[ uiPos++] & SC_MIXED)	// Increment pos here!
		{
			// Don't pre-increment pos on line below!
			uiPos += flmWPToMixed( pucWPStr, uiWPStrLen,
				&pucColStr[ uiPos], uiLang);
		}
		// else 0x04 or 0x06 - all characters already in uppercase
	}
	
	// Should end perfectly at the end of the collation buffer.

	if (uiPos != uiColStrLen)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}
	
	*puiWPStrLen = uiWPStrLen;
	*puiUnconvChars = uiUnconvChars;

Exit:

	return( rc);
}

/**************************************************************************
Desc: 	Combine the diacritic 5-bit values to an existing WP string
***************************************************************************/
FSTATIC RCODE flmWPCmbSubColBuf(
	FLMBYTE *			pucWPStr,					// Existing WP string to modify
	FLMUINT *			puiWPStrLen,				// WP string length in bytes
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,				// Diacritic values in 5 bit sets
	FLMBOOL				bHebrewArabic,				// Set if language is Hebrew or Arabic
	FLMUINT *			puiSubColBitPos)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT 		uiSubColBitPos = 0;
	FLMUINT 		uiNumChars = *puiWPStrLen >> 1;
	FLMUINT16 	ui16Diac;
	FLMUINT16 	ui16WPChar;
	FLMUINT		uiTemp;

	// For each character (two bytes) in the WP string ...

	while( uiNumChars--)
	{
		// Label used for hebrew/arabic - additional subcollation can follow
		// This macro DOESN'T increment bitPos

		if( testOneBit( pucSubColBuf, uiSubColBitPos))
		{
			// If "11110" - unmappable unicode char - 0xFFFF is before it
			// If "1110" then INDEX extended char is inserted
			// If "110" then extended char follows that replaces collation
			// If "10"  then take next 5 bits which
			// contain the diacritic subcollation value.

after_last_character:

			uiSubColBitPos++;	// Eat the first 1 bit
			if( !testOneBit( pucSubColBuf, uiSubColBitPos))
			{
				uiSubColBitPos++;	// Eat the 0 bit
				ui16Diac = (FLMUINT16)(getNBits( 5, pucSubColBuf, uiSubColBitPos));
				uiSubColBitPos += 5;

				// If not extended base

				if( (ui16WPChar = FB2UW( pucWPStr)) < 0x100)
				{
					// Convert to WP diacritic and combine characters

					flmWPCmbcar( &ui16WPChar, ui16WPChar,
						(FLMUINT16)ml1_COLtoD[ ui16Diac]);

					// Even if cmbcar fails, wpchar is still set to a valid value

					UW2FBA( ui16WPChar, pucWPStr);
				}
				else if( (ui16WPChar & 0xFF00) == 0x0D00)	// Arabic?
				{
					ui16WPChar = ArabSubColToWPChr[ ui16Diac];
					UW2FBA( ui16WPChar, pucWPStr);
				}
				// else diacritic is extra info
				// cmbcar should not handle extended chars for this design
			}
			else		// "110"  or "1110" or "11110"
			{
				uiSubColBitPos++;	// Eat the 2nd '1' bit
				if( testOneBit( pucSubColBuf, uiSubColBitPos))	// Test the 3rd bit
				{
					if( (*puiWPStrLen) + 2 > uiMaxWPBytes)
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					// 1110 - shift wpchars down 1 word and insert value below
					uiSubColBitPos++;			// Eat the 3rd '1' bit
					*puiWPStrLen += 2;		// Return 2 more bytes

					if( testOneBit( pucSubColBuf, uiSubColBitPos))	// Test 4th bit
					{
						// Unconvertable UNICODE character
						// The format will be 4 bytes, 0xFF, 0xFF, 2 byte Unicode

						shiftN( pucWPStr, uiNumChars + uiNumChars + 4, 2);
						uiSubColBitPos++;	// Eat the 4th '1' bit
						pucWPStr += 2;	// Skip the 0xFFFF for now
					}
					else
					{
						// Move down 2 byte NULL and rest of the 2 byte characters
						// The extended character does not have a 0xFF col value

						shiftN( pucWPStr, uiNumChars + uiNumChars + 2, 2);
						uiNumChars++;	// Increment because inserted

						// Fall through reading the actual charater value
					}
				}

				uiSubColBitPos++;	// Skip past the zero bit
				uiSubColBitPos = (uiSubColBitPos + 7) & (~7);	// roundup to next byte
				uiTemp = bytesInBits( uiSubColBitPos);				// compute position
				pucWPStr[ 1] = pucSubColBuf[ uiTemp];				// Character set
				pucWPStr[ 0] = pucSubColBuf[ uiTemp + 1];			// Character
				uiSubColBitPos += 16;
			}
		}
		else
		{
			uiSubColBitPos++;
		}

		pucWPStr += 2;	// Next WP character
	}

	if( bHebrewArabic)
	{
		if( testOneBit( pucSubColBuf, uiSubColBitPos))
		{
			// Hebrew/Arabic can have trailing accents that
			// don't have a matching collation value.
			// Keep looping in this case.
			// Note that subColBitPos isn't incremented above.

			uiNumChars = 0;	// Set so we won't loop forever!
			goto after_last_character;	// process trailing bit
		}
		uiSubColBitPos++;	// Eat the last '0' bit
	}

	*puiSubColBitPos = uiSubColBitPos;

Exit:

	return( rc);
}

/**************************************************************************
Desc: 	Convert the WP string to lower case chars given low/up bit string
Out:	 	WP characters that have been modified to their original case
Ret:		Number of bytes used in the lower/upper buffer
Notes:	Only WP to lower case conversion is done here for each bit NOT set.
***************************************************************************/
FSTATIC FLMUINT flmWPToMixed(
	FLMBYTE *			pucWPStr,			// Existing WP string to modify
	FLMUINT				uiWPStrLen,			// Length of the WP string in bytes
	const FLMBYTE *	pucLowUpBitStr,	// Lower/upper case bit string
	FLMUINT				uiLang)
{
	FLMUINT		uiNumChars;
	FLMUINT		uiTempWord;
	FLMBYTE		ucTempByte = 0;
	FLMBYTE		ucMaskByte;
	FLMBYTE		ucXorByte;	// Used to reverse GR, bits

	ucXorByte = (uiLang == XFLM_US_LANG)	// Do most common compare first
						? (FLMBYTE)0
						: (uiLang == XFLM_GR_LANG)	// Greek has uppercase first
							? (FLMBYTE)0xFF
							: (FLMBYTE)0 ;

	// For each character (two bytes) in the word string ...
	for( uiNumChars = uiWPStrLen >> 1,
				ucMaskByte = 0;							// Force first time to get a byte
				uiNumChars--;
				pucWPStr += 2,								// Next WP character - word
				ucMaskByte >>= 1)							// Next bit to mask and check
	{
		if( ucMaskByte == 0)
		{
			// Time to get another byte

			ucTempByte = ucXorByte ^ *pucLowUpBitStr++;
			ucMaskByte = 0x80;
		}

		// If lowercase convert, else is upper

		if( (ucTempByte & ucMaskByte) == 0)
		{
			// Convert to lower case - COLL -> WP is already in upper case

			uiTempWord = (FLMUINT) FB2UW( pucWPStr);
			if( uiTempWord >= ASCII_UPPER_A && uiTempWord <= ASCII_UPPER_Z)
			{
				uiTempWord |= 0x20;
			}
			else
			{
				FLMBYTE ucCharVal = (FLMBYTE)( uiTempWord & 0xFF);
				FLMBYTE ucCharSet = (FLMBYTE)( uiTempWord >> 8);

				// Check if charact within region of character set

				if( ((ucCharSet == CHSMUL1) &&
						((ucCharVal >= 26) && (ucCharVal <= 241))) ||
					((ucCharSet == CHSGREK) && (ucCharVal <= 69)) ||
					((ucCharSet == CHSCYR) && (ucCharVal <= 199)))
				{
					uiTempWord |= 0x01;		// Set the bit ... don't increment!
				}
			}
			UW2FBA( (FLMUINT16)uiTempWord, pucWPStr);
		}
	}

	uiNumChars = uiWPStrLen >> 1;
	return( bytesInBits( uiNumChars));
}

/****************************************************************************
Desc:	Converts a character to upper case (if possible)
****************************************************************************/
FLMUINT16 flmWPUpper(
	FLMUINT16	ui16WpChar)
{
	if( ui16WpChar < 256)
	{
		if( ui16WpChar >= ASCII_LOWER_A && ui16WpChar <= ASCII_LOWER_Z)
		{
			// Return ASCII upper case

			return( ui16WpChar & 0xdf);
		}
	}
	else
	{
		FLMBYTE	ucCharSet = ui16WpChar >> 8;

		if( ucCharSet == CHSMUL1)
		{
			FLMBYTE	ucChar = ui16WpChar & 0xFF;

			if( ucChar >= fwp_caseConvertableRange[ (CHSMUL1-1) * 2] &&
				 ucChar <= fwp_caseConvertableRange[ ((CHSMUL1-1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if( ucCharSet == CHSGREK)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSGREK-1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if( ucCharSet == CHSCYR)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSCYR-1) * 2) + 1])
			{
				return( ui16WpChar & 0xFFFE);
			}
		}
		else if( ui16WpChar >= Lower_JP_a)
		{
			// Possible double byte character set alphabetic character?

			if( ui16WpChar <= Lower_JP_z)
			{
				// Japanese?

				ui16WpChar = (ui16WpChar - Lower_JP_a) + Upper_JP_A;
			}
			else if( ui16WpChar >= Lower_KR_a && ui16WpChar <= Lower_KR_z)
			{
				// Korean?

				ui16WpChar = (ui16WpChar - Lower_KR_a) + Upper_KR_A;
			}
			else if( ui16WpChar >= Lower_CS_a && ui16WpChar <= Lower_CS_z)
			{
				// Chinese Simplified?

				ui16WpChar = (ui16WpChar - Lower_CS_a) + Upper_CS_A;
			}
			else if( ui16WpChar >= Lower_CT_a && ui16WpChar <= Lower_CT_z)
			{
				// Chinese Traditional?

				ui16WpChar = (ui16WpChar - Lower_CT_a) + Upper_CT_A;
			}
		}
	}

	// Return original character - original not in lower case.

	return( ui16WpChar);
}

/****************************************************************************
Desc:	Checks to see if WP character is upper case
****************************************************************************/
FLMBOOL flmWPIsUpper(
	FLMUINT16	ui16WpChar)
{
	FLMBYTE	ucChar;
	FLMBYTE	ucCharSet;

	// Get character

	ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

	// Test if ASCII character set

	if( !(ui16WpChar & 0xFF00))
	{
		return( (ucChar >= ASCII_LOWER_A && ucChar <= ASCII_LOWER_Z)
				  ? FALSE
				  : TRUE);
	}

	// Get the character set

	ucCharSet = (FLMBYTE) (ui16WpChar >> 8);

	// CHSMUL1 == Multinational 1 character set
	// CHSGREK == Greek character set
	// CHSCYR == Cyrillic character set

	if( (ucCharSet == CHSMUL1 && ucChar >= 26 && ucChar <= 241) ||
		 (ucCharSet == CHSGREK && ucChar <= 69) ||
		 (ucCharSet == CHSCYR && ucChar <= 199))
	{
		return( (ucChar & 1) ? FALSE : TRUE);
	}

	// Don't care that double ss is lower

	return( TRUE);
}

/****************************************************************************
Desc:	Converts a character to lower case (if possible)
****************************************************************************/
FLMUINT16 flmWPLower(
	FLMUINT16	ui16WpChar)
{
	if( ui16WpChar < 256)
	{
		if( ui16WpChar >= ASCII_UPPER_A && ui16WpChar <= ASCII_UPPER_Z)
		{
			return( ui16WpChar | 0x20);
		}
	}
	else
	{
		FLMBYTE	ucCharSet = ui16WpChar >> 8;

		if( ucCharSet == CHSMUL1)
		{
			FLMBYTE	ucChar = ui16WpChar & 0xFF;

			if( ucChar >= fwp_caseConvertableRange[ (CHSMUL1-1) * 2] &&
				 ucChar <= fwp_caseConvertableRange[ ((CHSMUL1-1) * 2) + 1] )
			{
				return( ui16WpChar | 1);
			}
		}
		else if( ucCharSet == CHSGREK)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSGREK-1) * 2) + 1])
			{
				return( ui16WpChar | 1);
			}
		}
		else if( ucCharSet == CHSCYR)
		{
			if( (ui16WpChar & 0xFF) <=
						fwp_caseConvertableRange[ ((CHSCYR-1) * 2) + 1])
			{
				return( ui16WpChar | 1);
			}
		}
		else if( ui16WpChar >= Upper_JP_A)
		{
			// Possible double byte character set alphabetic character?

			if( ui16WpChar <= Upper_JP_Z)
			{
				// Japanese?

				ui16WpChar = ui16WpChar - Upper_JP_A + Lower_JP_a;
			}
			else if( ui16WpChar >= Upper_KR_A && ui16WpChar <= Upper_KR_Z)
			{
				// Korean?

				ui16WpChar = ui16WpChar - Upper_KR_A + Lower_KR_a;
			}
			else if( ui16WpChar >= Upper_CS_A && ui16WpChar <= Upper_CS_Z)
			{
				// Chinese Simplified?

				ui16WpChar = ui16WpChar - Upper_CS_A + Lower_CS_a;
			}
			else if( ui16WpChar >= Upper_CT_A && ui16WpChar <= Upper_CT_Z)
			{
				// Chinese Traditional?

				ui16WpChar = ui16WpChar - Upper_CT_A + Lower_CT_a;
			}
		}
	}

	// Return original character, original not in upper case

	return( ui16WpChar);
}

/****************************************************************************
Desc:	Break a WP character into a base and a diacritical char.
Ret: 	TRUE - if not found
		FALSE - if found
****************************************************************************/
FLMBOOL flmWPBrkcar(
	FLMUINT16		ui16WpChar,
	FLMUINT16 *		pui16BaseChar,
	FLMUINT16 *		pui16DiacriticChar)
{
	BASE_DIACRIT *		pBaseDiacritic;
	FLMINT				iTableIndex;

	if( (pBaseDiacritic = fwp_car60_c[ HI(ui16WpChar)]) == 0)
	{
		return( TRUE);
	}

	iTableIndex = ((FLMBYTE)ui16WpChar) - pBaseDiacritic->start_char;
	if( iTableIndex < 0 ||
		 iTableIndex > pBaseDiacritic->char_count ||
		 pBaseDiacritic->table [iTableIndex].base == (FLMBYTE)0xFF)
	{
		return( TRUE);
	}

	if( (HI( ui16WpChar) != CHSMUL1) ||
		 ((fwp_ml1_cb60[ ((FLMBYTE) ui16WpChar) >> 3] >>
			(7 - (ui16WpChar & 0x07))) & 0x01))
	{

		// normal case, same base as same as characters

		*pui16BaseChar = (ui16WpChar & 0xFF00) |
								pBaseDiacritic->table [iTableIndex].base;
		*pui16DiacriticChar = (ui16WpChar & 0xFF00) |
								pBaseDiacritic->table[iTableIndex].diacrit;
	}
	else
	{

		// Multi-national where base is ascii value.

		*pui16BaseChar = pBaseDiacritic->table [iTableIndex].base;
		*pui16DiacriticChar = (ui16WpChar & 0xFF00) |
										pBaseDiacritic->table[iTableIndex].diacrit;
	}

	return( FALSE);
}

/****************************************************************************
Desc:	Take a base and a diacritic and compose a WP character.
		Note on base character: i's and j's must be dotless i's and j's (for
		those which use them) or they will not be found.
Ret: 	TRUE - if not found
		FALSE  - if found
Notes: ascii characters with diacriticals are in multi-national if anywhere;
		 all other base chars with diacritics are found in their own sets.
****************************************************************************/
FSTATIC FLMBOOL flmWPCmbcar(
	FLMUINT16 *	pui16WpChar,
	FLMUINT16	ui16BaseChar,
	FLMINT16		ui16DiacriticChar)
{
	FLMUINT						uiRemaining;
	FLMBYTE						ucCharSet;
	FLMBYTE						ucChar;
	BASE_DIACRIT *				pBaseDiacritic;
	BASE_DIACRIT_TABLE *		pTable;

	ucCharSet = HI( ui16BaseChar);
	if( ucCharSet > WP_MAX_CAR60_SIZE)
	{
		return( TRUE);
	}

	// Is base ASCII?  If so, look in multinational 1

	if( !ucCharSet)
	{
		ucCharSet = CHSMUL1;
	}

	if( (pBaseDiacritic = fwp_car60_c[ucCharSet]) == 0)
	{
		return( TRUE);
	}

	ucChar = LO( ui16BaseChar);
	ui16DiacriticChar = LO( ui16DiacriticChar);
	pTable = pBaseDiacritic->table;
	for( uiRemaining = pBaseDiacritic->char_count;
		  uiRemaining;
		  uiRemaining--, pTable++ )
	{
		// Same base?

		if( pTable->base == ucChar &&
			 (pTable->diacrit & 0x7F) == ui16DiacriticChar)
		{
			// Same diacritic?

			*pui16WpChar = (FLMUINT16) (((FLMUINT16) ucCharSet << 8) +
					(pBaseDiacritic->start_char +
					 (FLMUINT16)(pTable - pBaseDiacritic->table)));
			return( FALSE);
		}
	}

	return( TRUE);
}

/**************************************************************************
Desc:	Find the collating value of a WP character
ret:	Collating value (COLS0 is high value - undefined WP char)
***********************************************************************/
FLMUINT16 flmWPGetCollation(
	FLMUINT16	ui16WpChar,
	FLMUINT		uiLanguage)
{
	FLMUINT16		ui16State;
	FLMBYTE			ucCharVal;
	FLMBYTE			ucCharSet;
	FLMBOOL			bHebrewArabicFlag = FALSE;
	TBL_B_TO_BP *	pColTbl = fwp_col60Tbl;

	// State ONLY for non-US

	if( uiLanguage != XFLM_US_LANG)
	{
		if( uiLanguage == XFLM_AR_LANG ||		// Arabic
			 uiLanguage == XFLM_FA_LANG ||		// Farsi - persian
			 uiLanguage == XFLM_HE_LANG ||		// Hebrew
			 uiLanguage == XFLM_UR_LANG) 			// Urdu
		{
			pColTbl = fwp_HebArabicCol60Tbl;
			bHebrewArabicFlag = TRUE;
		}
		else
		{
			// check if uiLanguage candidate for alternate double collating

			ui16State = getNextCharState( START_COL, uiLanguage);
			if( 0 != (ui16State = getNextCharState( (ui16State
							?	ui16State		// look at special case languages
							:	START_ALL),		// look at US and European
							(FLMUINT) ui16WpChar)))
			{
				return( ui16State);
			}
		}
	}

	ucCharVal = (FLMBYTE)ui16WpChar;
	ucCharSet = (FLMBYTE)(ui16WpChar >> 8);

	do
	{
		if( pColTbl->key == ucCharSet)
		{
			FLMBYTE *	pucColVals;	// table of collating values

			pucColVals = pColTbl->charPtr;

			// Check if the value is in the range of collated chars
			// Above lower range of table?

			if (ucCharVal >= *pucColVals)
			{
				// Make value zero based to index

				ucCharVal -= *pucColVals++;

				// Below maximum number of table entries?

				if( ucCharVal < *pucColVals++)
				{
					// Return collated value.

					return( pucColVals[ ucCharVal]);
				}
			}
		}

		// Go to next table entry

		pColTbl++;
	} while( pColTbl->key != 0xFF);

	if( bHebrewArabicFlag)
	{
		if( ucCharSet == CHSHEB ||
			 ucCharSet == CHSARB1 ||
			 ucCharSet == CHSARB2)
		{
			// Same as COLS0_HEBREW

			return( COLS0_ARABIC);
		}
	}

	// Defaults for characters that don't have a collation value.

	return( COLS0);
}

/****************************************************************************
Desc:	Check for double characters that sort as 1 (like ch in Spanish) or
		1 character that should sort as 2 (like ? sorts as ae in French).
Return:	0 = nothing changes
			1 if sorting 2 characters as 1 - *pui16WpChar is the one character.
			second character value if 1 character sorts as 2,
			*pui16WpChar changes to first character in sequence
****************************************************************************/
RCODE flmWPCheckDoubleCollation(
	IF_PosIStream *	pIStream,
	FLMBOOL				bUnicodeStream,
	FLMBOOL				bAllowTwoIntoOne,
	FLMUNICODE *		puzChar,
	FLMUNICODE *		puzChar2,
	FLMBOOL *			pbTwoIntoOne,
	FLMUINT				uiLanguage)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT16	ui16CurState;
	FLMUINT16	ui16WpChar;
	FLMUNICODE	uzLastChar = 0;
	FLMUNICODE	uChar = *puzChar;
	FLMUNICODE	uDummy;
	FLMBOOL		bUpperFlag;
	FLMUINT64	ui64SavePosition = pIStream->getCurrPosition();

	if (!flmUnicodeToWP( *puzChar, &ui16WpChar))
	{
		ui16WpChar = UNK_UNICODE_CODE;
	}
	bUpperFlag = flmWPIsUpper( ui16WpChar);

	if ((ui16CurState = getNextCharState( 0, uiLanguage)) == 0)
	{
		*pbTwoIntoOne = FALSE;
		*puzChar2 = 0;
		goto Exit;
	}

	for (;;)
	{
		switch (ui16CurState)
		{
			case INSTSG:
				*puzChar = *puzChar2 = (FLMUNICODE)f_toascii( 's');
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case INSTAE:
				if (bUpperFlag)
				{
					*puzChar = (FLMUNICODE)f_toascii( 'A');
					*puzChar2 = (FLMUNICODE)f_toascii( 'E');
				}
				else
				{
					*puzChar = (FLMUNICODE)f_toascii( 'a');
					*puzChar2 = (FLMUNICODE)f_toascii( 'e');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case INSTIJ:
				if (bUpperFlag)
				{
					*puzChar = (FLMUNICODE)f_toascii( 'I');
					*puzChar2 = (FLMUNICODE)f_toascii( 'J');
				}
				else
				{
					*puzChar = (FLMUNICODE)f_toascii( 'i');
					*puzChar2 = (FLMUNICODE)f_toascii( 'j');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case INSTOE:
				if (bUpperFlag)
				{
					*puzChar = (FLMUNICODE)f_toascii( 'O');
					*puzChar2 = (FLMUNICODE)f_toascii( 'E');
				}
				else
				{
					*puzChar = (FLMUNICODE)f_toascii( 'o');
					*puzChar2 = (FLMUNICODE)f_toascii( 'e');
				}
				*pbTwoIntoOne = FALSE;
				goto Exit;
			case WITHAA:
				*puzChar = (FLMUNICODE)(bUpperFlag
													? (FLMUNICODE)0xC5
													: (FLMUNICODE)0xE5);
													
				if (RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
				{
					goto Exit;
				}

				if( bUnicodeStream)
				{
					rc = pIStream->read( &uDummy, sizeof( FLMUNICODE), NULL);
				}
				else
				{
					rc = flmReadUTF8CharAsUnicode( pIStream, &uDummy);
				}
				
				if( RC_BAD( rc))
				{
					if (rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}

				ui64SavePosition = pIStream->getCurrPosition();
				break;
			case AFTERC:
				*puzChar = (FLMUINT16)(bUpperFlag
													? (FLMUNICODE)f_toascii( 'C')
													: (FLMUNICODE)f_toascii( 'c'));
Position_After_2nd:

				if( bAllowTwoIntoOne)
				{
					*puzChar2 = uzLastChar;
					*pbTwoIntoOne = TRUE;

					if (RC_BAD( rc = pIStream->positionTo( ui64SavePosition)))
					{
						goto Exit;
					}
					
					if( bUnicodeStream)
					{
						rc = pIStream->read( &uChar, sizeof( FLMUNICODE), NULL);
					}
					else
					{
						rc = flmReadUTF8CharAsUnicode( pIStream, &uChar);
					}

					if (RC_BAD( rc))
					{
						if (rc == NE_XFLM_EOF_HIT)
						{
							rc = NE_XFLM_OK;
						}
						else
						{
							goto Exit;
						}
					}

					ui64SavePosition = pIStream->getCurrPosition();
				}
				goto Exit;
			case AFTERH:
				*puzChar = (FLMUINT16)(bUpperFlag
													? (FLMUNICODE)f_toascii( 'H')
													: (FLMUNICODE)f_toascii( 'h'));
				goto Position_After_2nd;
			case AFTERL:
				*puzChar = (FLMUINT16)(bUpperFlag
													? (FLMUNICODE)f_toascii( 'L')
													: (FLMUNICODE)f_toascii( 'l'));
				goto Position_After_2nd;
			default:
				// Handles STATE1 through STATE11 also
				break;
		}

		if ((ui16CurState = getNextCharState( ui16CurState,
									flmWPLower( ui16WpChar))) == 0)
		{
			break;
		}

		uzLastChar = uChar;
		
		if( bUnicodeStream)
		{
			rc = pIStream->read( &uChar, sizeof( FLMUNICODE), NULL);
		}
		else
		{
			rc = flmReadUTF8CharAsUnicode( pIStream, &uChar);
		}

		if (RC_BAD( rc))
		{
			if (rc == NE_XFLM_EOF_HIT)
			{
				rc = NE_XFLM_OK;
			}
			else
			{
				goto Exit;
			}
		}

		if (!flmUnicodeToWP( uChar, &ui16WpChar))
		{
			ui16WpChar = UNK_UNICODE_CODE;
		}
	}

Exit:

	if (RC_OK( rc))
	{
		rc = pIStream->positionTo( ui64SavePosition);
	}

	return( rc);
}

/****************************************************************************
Desc:	Returns the collation value of the input WP character.
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
FLMUINT16 flmWPAsiaGetCollation(
	FLMUINT16	ui16WpChar,				// WP char to get collation values
	FLMUINT16	ui16NextWpChar,		// Next WP char - for CS11 voicing marks
	FLMUINT16   ui16PrevColValue,		// Previous collating value
	FLMUINT16 *	pui16ColValue,			// Returns collation value
	FLMUINT16 * pui16SubColVal,		// Returns sub-collation value
	FLMBYTE *	pucCaseBits,		 	// Returns case bits value
	FLMBOOL		bUppercaseFlag)		// Set if to convert to uppercase
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

	if( ucCharSet >= 0x2B)
	{
		// Puts 2 or above into high byte.

		ui16ColValue = ui16WpChar - 0x2900;

		// No subcollation or case bits need to be set

		goto	Exit;
	}

	// Single wide character? (HANKAKU)

	if( ucCharSet < 11)
	{
		// Get the values from a non-asian character
		// LATIN, GREEK or CYRILLIC
		// The width bit may have been set on a jump to
		// label from below.

Latin_Greek_Cyrillic:

		// YES: Pass XFLM_US_LANG because this is what we want -
		// Prevents double character sorting.

		ui16ColValue = flmWPGetCollation( ui16WpChar, XFLM_US_LANG);

		if (bUppercaseFlag || flmWPIsUpper( ui16WpChar))
		{
			// Uppercase - set case bit

			ucCaseBits |= SET_CASE_BIT;
		}

		// Character for which there is no collation value?

		if( ui16ColValue == COLS0)
		{
			ui16ReturnValue = 0;
			if( !flmWPIsUpper( ui16WpChar))
			{
				// Convert to uppercase

				ui16WpChar--;
			}
			ui16ColValue = 0xFFFF;
			ui16SubColVal = ui16WpChar;
		}
		else if( ucCharSet) 				// Don't bother with ascii
		{
			if( !flmWPIsUpper( ui16WpChar))
			{
				// Convert to uppercase

				ui16WpChar--;
			}

        	if( ucCharSet == CHSMUL1)
			{
				FLMUINT16	ui16Base;
				FLMUINT16	ui16Diacritic;

				ui16SubColVal = !flmWPBrkcar( ui16WpChar, &ui16Base,
															&ui16Diacritic)
									  ? fwp_dia60Tbl[ ui16Diacritic & 0xFF]
									  : ui16WpChar;
			}
			else if( ucCharSet == CHSGREK) // GREEK
         {
         	if( ui16WpChar >= 0x834 ||		// [8,52] or above
            	 ui16WpChar == 0x804 ||		// [8,4] BETA Medial | Terminal
					 ui16WpChar == 0x826)		// [8,38] SIGMA terminal
				{
					ui16SubColVal = ui16WpChar;
				}
			}
			else if( ucCharSet == CHSCYR)	// CYRILLIC
			{
           	if( ui16WpChar >= 0xA90)		// [10, 144] or above
				{
              	ui16SubColVal = ui16WpChar;	// Dup collation values
				}
         }
         // else don't need a sub collation value
      }
		goto	Exit;
	}

	// Single wide Japanese character?

 	if( ucCharSet == 11)
	{
		FLMUINT16	ui16KanaChar;

		// Convert charset 11 to Zenkaku (double wide) CS24 or CS26 hex.
		// All characters in charset 11 will convert to CS24 or CS26.
		// when combining the collation and the sub-collation values.

		if( flmWPHanToZenkaku( ui16WpChar,
			ui16NextWpChar, &ui16KanaChar ) == 2)
		{
			// Return 2

			ui16ReturnValue++;
		}

		ucCaseBits |= SET_WIDTH_BIT;	// Set so will allow to go back
		ui16WpChar = ui16KanaChar;		// If in CS24 will fall through to ZenKaku
		ucCharSet = ui16KanaChar >> 8;
		ucCharVal = ui16KanaChar & 0xFF;
	}

	if( ui16WpChar < 0x2400)
	{
		// In some other character set

		goto Latin_Greek_Cyrillic;
	}
	else if( ui16WpChar >= 0x255e &&	// Hiragana?
				ui16WpChar <= 0x2655)	// Katakana?
	{
		if( ui16WpChar >= 0x2600)
		{
			ucCaseBits |= SET_KATAKANA_BIT;
		}

		// HIRAGANA & KATAKANA
		//		Kana contains both hiragana and katakana.
		//		The tables contain the same characters in same order

		if( ucCharSet == 0x25)
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

	if( (ui16Hankaku = flmWPZenToHankaku( ui16WpChar, NULL)) != 0)
	{
		if( (ui16Hankaku >> 8) != 11)			// if CharSet11 was a CS24 symbol
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

	for( uiLoop = 0;
		  uiLoop < (sizeof( fwp_Ch24ColTbl) / sizeof( BYTE_WORD_TBL));
	  	  uiLoop++ )
	{
		if( ucCharVal == fwp_Ch24ColTbl[ uiLoop].ByteValue)
		{
			if( (ui16ColValue = fwp_Ch24ColTbl[ uiLoop].WordValue) < 0x100)
			{
				// Don't save for chuuten, dakuten, handakuten

				ui16SubColVal = (FLMUINT16)(uiLoop + 1);
			}
			break;
		}
	}

	if( !ui16ColValue)
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
Desc:  	Convert a text string to a collated string.
****************************************************************************/
RCODE flmAsiaUTF8ToColText(
	IF_PosIStream *	pIStream,
	FLMBYTE *			pucColStr,			// Output collated string
	FLMUINT *			puiColStrLen,		// Collated string length return value
													// Input value is MAX num of bytes in buffer
	FLMBOOL				bCaseInsensitive,	// Set if to convert to uppercase
	FLMUINT *			puiCollationLen,	// Returns the collation bytes length
	FLMUINT *			puiCaseLen,			// Returns length of case bytes
	FLMUINT				uiCharLimit,		// Max number of characters in this key piece
	FLMBOOL				bFirstSubstring,	// TRUE is this is the first substring key
	FLMBOOL				bDataTruncated,	// Was input data already truncated.
	FLMBOOL *			pbDataTruncated)
{
	RCODE			rc = NE_XFLM_OK;
	FLMBOOL		bEndOfStr = FALSE;
	FLMUINT		uiLength;
	FLMUINT 		uiTargetColLen = *puiColStrLen - 12; // 6=ovhd,6=worst char
	FLMBYTE		ucSubColBuf[ MAX_SUBCOL_BUF + 1]; // Holds Sub-col values (diac)
	FLMBYTE		ucLowUpBuf[ MAX_CASE_BYTES + MAX_CASE_BYTES + 2]; // 2 case bits/wpchar
	FLMUINT		uiColLen;
	FLMUINT		uiSubColBitPos;
	FLMUINT 		uiLowUpBitPos;
	FLMUINT		uiFlags;
	FLMUNICODE	uChar;
	FLMUINT16	ui16NextWpChar;
	FLMUINT16	ui16ColValue;

	uiColLen = uiSubColBitPos = uiLowUpBitPos = uiFlags = 0;
	uChar = ui16ColValue = 0;

	// We don't want any single key piece to "pig out" more
	// than 256 bytes of the key

	if( uiTargetColLen > 256 - 12)
	{
		uiTargetColLen = 256 - 12;
	}

	// Make sure ucSubColBuf and ucLowUpBuf are set to 0

	f_memset( ucSubColBuf, 0, sizeof( ucSubColBuf));
	f_memset( ucLowUpBuf,  0, sizeof( ucLowUpBuf));

	ui16NextWpChar = 0;

	while( !bEndOfStr || ui16NextWpChar || uChar)
	{
		FLMUINT16	ui16WpChar;			// Current WP character
		FLMUINT16	ui16SubColVal;		// Sub-collated value (diacritic)
		FLMBYTE		ucCaseFlags;
		FLMUINT16	ui16CurWpChar;

		// Get the next character from the string.

		ui16WpChar = ui16NextWpChar;
		for( ui16NextWpChar = 0;
			  (!ui16WpChar || !ui16NextWpChar) &&
				  !uChar && !bEndOfStr;)
		{
			if (!bEndOfStr)
			{
				if( RC_BAD( rc = flmReadUTF8CharAsUnicode( pIStream, &uChar)))
				{
					if (rc == NE_XFLM_EOF_HIT)
					{
						rc = NE_XFLM_OK;
						bEndOfStr = TRUE;
					}
					else
					{
						goto Exit;
					}
				}
			}
			else
			{
				uChar = 0;
			}

			if( flmUnicodeToWP( uChar, &ui16CurWpChar))
			{
				uChar = 0;
			}

			if( !ui16WpChar)
			{
				ui16WpChar = ui16CurWpChar;
			}
			else
			{
				ui16NextWpChar = ui16CurWpChar;
			}
		}

		// If we didn't get a character, break out of the outer
		// processing loop.

		if( !ui16WpChar && !uChar)
		{
			break;
		}

		if( ui16WpChar)
		{
			if( flmWPAsiaGetCollation( ui16WpChar, ui16NextWpChar, ui16ColValue,
				&ui16ColValue, &ui16SubColVal, &ucCaseFlags, bCaseInsensitive) == 2)
			{
				// Took the ui16NextWpChar value
				// Force to skip this value

				ui16NextWpChar = 0;
			}
		}
		else // Use the uChar value for this pass
		{
			// This handles all of the UNICODE characters that could not
			// be converted to WP characters - which will include most
			// of the Asian characters.

			ucCaseFlags = 0;
			if( uChar < 0x20)
			{
				ui16ColValue = 0xFFFF;

				// Setting ui16SubColVal to a high code will ensure
				// that the code that the uChar value will be stored
				// in in the sub-collation area.

				ui16SubColVal = 0xFFFF;

				// NOTE: uChar SHOULD NOT be set to zero here.
				// It will be set to zero below.
			}
			else
			{
				ui16ColValue = uChar;
				ui16SubColVal = 0;
				uChar = 0;
			}
		}

		// Store the values in 2 bytes

		pucColStr[ uiColLen++] = (FLMBYTE)(ui16ColValue >> 8);
		pucColStr[ uiColLen++] = (FLMBYTE)(ui16ColValue & 0xFF);

		if( ui16SubColVal)
		{
			uiFlags |= HAD_SUB_COLLATION;
			if( ui16SubColVal <= 31)	// 5 bit - store bits 10
			{
				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos += 1 + 1; // Stores a zero
				setBits( 5, ucSubColBuf, uiSubColBitPos, ui16SubColVal);
				uiSubColBitPos += 5;
			}
			else	// 2 bytes - store bits 110 or 11110
			{
				FLMUINT		 uiTemp;

				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;
				setBit( ucSubColBuf, uiSubColBitPos);
				uiSubColBitPos++;

				if( !ui16WpChar && uChar) // Store as "11110"
				{
					ui16SubColVal = uChar;
					uChar = 0;
					setBit( ucSubColBuf, uiSubColBitPos);
					uiSubColBitPos++;
					setBit( ucSubColBuf, uiSubColBitPos);
					uiSubColBitPos++;
				}
				uiSubColBitPos++;	// Skip past the zero

				// Go to the next byte boundary to write the WP char
				uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
				uiTemp = bytesInBits( uiSubColBitPos);

				// Need to store HIGH-Low - PC format is Low-high!
				ucSubColBuf[ uiTemp ] = (FLMBYTE)(ui16SubColVal >> 8);
				ucSubColBuf[ uiTemp + 1] = (FLMBYTE)(ui16SubColVal);

				uiSubColBitPos += 16;
			}
		}
		else
		{
			uiSubColBitPos++;
		}

		// Save case information - always 2 bits worth for Asian

		if( ucCaseFlags & 0x02)
		{
			setBit( ucLowUpBuf, uiLowUpBitPos);
		}

		uiLowUpBitPos++;

		if( ucCaseFlags & 0x01)
		{
			setBit( ucLowUpBuf, uiLowUpBitPos);
		}
		uiLowUpBitPos++;

		// Check to see if uiColLen is within 1 byte of max

		if( (uiColLen >= uiCharLimit) ||
			 (uiColLen + bytesInBits( uiSubColBitPos) +
					 bytesInBits( uiLowUpBitPos) >= uiTargetColLen))
		{
			// Still something left?

			if (ui16NextWpChar || uChar)
			{
				bDataTruncated = TRUE;
			}
			else if (!bEndOfStr)
			{
				if (RC_BAD( rc = flmReadUTF8CharAsUnicode( pIStream, &uChar)))
				{
					if (rc == NE_XFLM_EOF_HIT)
					{
						bEndOfStr = TRUE;
						rc = NE_XFLM_OK;
					}
					else
					{
						goto Exit;
					}
				}
				else
				{
					bDataTruncated = TRUE;
				}
			}
			break; // Hit the max. number of characters
		}
	}

	if( puiCollationLen)
	{
		*puiCollationLen = uiColLen;
	}

	// Add the first substring marker - also serves
	// as making the string non-null.

	if( bFirstSubstring)
	{
		pucColStr[ uiColLen++] = 0;
		pucColStr[ uiColLen++] = COLL_FIRST_SUBSTRING;
	}

	if( bDataTruncated)
	{
		pucColStr[ uiColLen++] = 0;
		pucColStr[ uiColLen++] = COLL_TRUNCATED;
	}

	// Return NOTHING if no values found

	if( !uiColLen && !uiSubColBitPos)
	{
		if( puiCaseLen)
		{
			*puiCaseLen = 0;
		}
		goto Exit;
	}

	// Done putting the String into 3 sections - build the COLLATED KEY

	if( uiFlags & HAD_SUB_COLLATION)
	{
		pucColStr[ uiColLen++] = 0;
		pucColStr[ uiColLen++] = COLL_MARKER | SC_SUB_COL;

		// Move the Sub-collation (diacritics) into the collating string

		uiLength = (FLMUINT)(bytesInBits( uiSubColBitPos));
		f_memcpy( &pucColStr[ uiColLen], ucSubColBuf, uiLength);
		uiColLen += uiLength;
	}

	// Always represent the marker as 2 bytes and case bits in Asia

	pucColStr[ uiColLen++] = 0;
	pucColStr[ uiColLen++] = COLL_MARKER | SC_MIXED;

	uiLength = (FLMUINT)(bytesInBits( uiLowUpBitPos));
	f_memcpy( &pucColStr[ uiColLen ], ucLowUpBuf, uiLength);

	if( puiCaseLen)
	{
		*puiCaseLen = (FLMUINT)(uiLength + 2);
	}
	uiColLen += uiLength;

Exit:

	if( pbDataTruncated)
	{
		*pbDataTruncated = bDataTruncated;
	}

	*puiColStrLen = uiColLen;
	return( rc);
}

/***************************************************************************
Desc:		Get the original string from an asian collation string
Ret:		Length of the word string in bytes
****************************************************************************/
FSTATIC RCODE flmAsiaColStr2WPStr(
	const FLMBYTE *	pucColStr,			  	// Points to the collated string
	FLMUINT				uiColStrLen,		  	// Length of the collated string
	FLMBYTE *			pucWPStr,			  	// Output string to build - WP word string
	FLMUINT *			puiWPStrLen,
	FLMUINT *			puiUnconvChars,
	FLMBOOL *			pbDataTruncated,		// Set to TRUE if truncated
	FLMBOOL *			pbFirstSubstring)		// Sets to TRUE if first substring
{
	FLMBYTE *	pucWPStrPtr = pucWPStr;
	FLMBYTE *	pucWPEnd = &pucWPStr[ *puiWPStrLen];
	FLMUINT		uiLength = uiColStrLen;
	FLMUINT		uiMaxWPBytes = *puiWPStrLen;
	FLMUINT		uiColStrPos = 0;
	FLMBOOL		bHadExtended = FALSE;
	FLMUINT		uiWPStrLen;
	FLMUINT16	ui16ColChar;
	FLMUINT		uiUnconvChars = 0;
	FLMUINT		uiColBytesProcessed;
	RCODE			rc = NE_XFLM_OK;

	while( uiLength)
	{
		FLMBYTE	ucChar = pucColStr[ uiColStrPos + 1];
		FLMBYTE	ucCharSet = pucColStr[ uiColStrPos];

		ui16ColChar = (FLMUINT16)((ucCharSet << 8) + ucChar);
		if( ui16ColChar <= MAX_COL_OPCODE)
		{
			break;
		}

		uiColStrPos += 2;
		uiLength -= 2;
		if( ucCharSet == 0)	// Normal Latin/Greek/Cyrillic value
		{
			ui16ColChar = colToWPChr[ ucChar - COLLS];
		}
		else if( ucCharSet == 1)	// Katakana or Hiragana character
		{
			if( ucChar > sizeof( ColToKanaTbl))	// Special cases below
			{
				if( ucChar == COLS_ASIAN_MARK_VAL) // Dakuten
				{
					ui16ColChar = 0x240a;
				}
				else if( ucChar == COLS_ASIAN_MARK_VAL + 1)	// Handakuten
				{
					ui16ColChar = 0x240b;
				}
				else if( ucChar == COLS_ASIAN_MARK_VAL + 2)	// Chuuten
				{
					ui16ColChar = 0x2405;
				}
				else
				{
					ui16ColChar = 0xFFFF;	// Error
				}
			}
			else
			{
				ui16ColChar = (FLMUINT16)(0x2600 + ColToKanaTbl[ ucChar]);
			}
		}
		else if( ucCharSet != 0xFF || ucChar != 0xFF)	// Asian characters
		{
			// Insert zeroes that will be treated as a signal for
			// uncoverted unicode characters later on.  NOTE: Cannot
			// use 0xFFFF, because we need to be able to detect this
			// case in the sub-collation stuff, and we don't want
			// to confuse it with the 0xFFFF that may have been inserted
			// in another case.
			// THIS IS A REALLY BAD HACK, BUT IT IS THE BEST WE CAN DO
			// FOR NOW!

			if( pucWPStrPtr + 2 >= pucWPEnd)
			{
				rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
				goto Exit;
			}

			*pucWPStrPtr++ = 0;
			*pucWPStrPtr++ = 0;
			uiUnconvChars++;
			bHadExtended = TRUE;
		}
		// else, there is no collation value - found in sub-collation part

		if( pucWPStrPtr + 2 >= pucWPEnd)
		{
			rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		UW2FBA( ui16ColChar, pucWPStrPtr);	// Put the uncollation value back
		pucWPStrPtr += 2;
	}

	if( pucWPStrPtr + 2 >= pucWPEnd)
	{
		rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	UW2FBA( 0, pucWPStrPtr);	// Terminate the string
	uiWPStrLen = (FLMUINT)(pucWPStrPtr - pucWPStr);

	//  Parse through the sub-collation and case information.
	//  Here are values for some of the codes:
	//   [ 0x05] - case bits follow
	//   [ 0x06] - case information is all uppercase
	//   [ 0x07] - beginning of sub-collation information
	//   [ 0x08] - first substring field that is made
	//   [ 0x09] - truncation marker for text and binary
	//
	//  Asian chars the case information should always be there and not
	//  compressed out.  This is because the case information could change
	//  the actual width of the character from 0x26xx to charset 11.

	//  Does truncation marker or sub-collation follow?

	if( uiLength)
	{
		ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
									pucColStr[ uiColStrPos + 1]);

		// First substring is before truncated.
		if( ui16ColChar == COLL_FIRST_SUBSTRING)
		{
			if( pbFirstSubstring)
			{
				*pbFirstSubstring = TRUE;	// Don't need to initialize to FALSE.
			}

			uiLength -= 2;
			uiColStrPos += 2;
			ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
										pucColStr[ uiColStrPos + 1]);
		}

		if( ui16ColChar == COLL_TRUNCATED)
		{
			if( pbDataTruncated)
			{
				*pbDataTruncated = TRUE;	// Don't need to initialize to FALSE.
			}
			uiLength -= 2;
			uiColStrPos += 2;
			ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
										pucColStr[ uiColStrPos+1]);
		}

		if( ui16ColChar == (COLL_MARKER | SC_SUB_COL))
		{
			FLMUINT 	uiTempLen;

			// Do another pass on the word string adding diacritics/voicings

			uiColStrPos += 2;
			uiLength -= 2;
			if( RC_BAD( rc = flmAsiaParseSubCol( pucWPStr, &uiWPStrLen,
				uiMaxWPBytes, &pucColStr[ uiColStrPos], &uiTempLen)))
			{
				goto Exit;
			}

			uiColStrPos += uiTempLen;
			uiLength -= uiTempLen;
		}
		else
		{
			goto check_case;
		}
	}

	// Does the case info follow?

	if( uiLength)
	{
		ui16ColChar = (FLMUINT16)((pucColStr[ uiColStrPos] << 8) +
									pucColStr[ uiColStrPos + 1]);
check_case:

		if( ui16ColChar == (COLL_MARKER | SC_MIXED))
		{
			uiColStrPos += 2;

			if( RC_BAD( rc = flmAsiaParseCase( pucWPStr, &uiWPStrLen,
				uiMaxWPBytes, &pucColStr[ uiColStrPos], &uiColBytesProcessed)))
			{
				goto Exit;
			}

			uiColStrPos += uiColBytesProcessed;

			// Set bHadExtended to FALSE, because they will have
			// been taken care of in this pass.

			bHadExtended = FALSE;
		}
	}

	// Change embedded zeroes to 0xFFFFs

	if (bHadExtended)
	{
		FLMUINT		uiCnt;
		FLMBYTE *	pucTmp;

		for( uiCnt = 0, pucTmp = pucWPStr;
			  uiCnt < uiWPStrLen;
			  uiCnt += 2, pucTmp += 2)
		{
			if( FB2UW( pucTmp) == 0)
			{
				UW2FBA( 0xFFFF, pucTmp);
			}
		}
	}
	
	if (uiColStrLen != uiColStrPos)
	{
		rc = RC_SET_AND_ASSERT( NE_XFLM_DATA_ERROR);
		goto Exit;
	}

	*puiUnconvChars = uiUnconvChars;
	*puiWPStrLen = uiWPStrLen;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Combine the diacritic 5 and 16 bit values to an existing word string.
Ret:		FLMUINT - Number of bytes parsed
Notes:	For each bit in the sub-collation section:
	0 - no subcollation information
	10 - take next 5 bits - will tell about diacritics or japanese vowel
	110 - align to next byte & take word value as extended character

****************************************************************************/
FSTATIC RCODE flmAsiaParseSubCol(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucSubColBuf,
	FLMUINT *			puiSubColBitPos)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT 		uiSubColBitPos = 0;
	FLMUINT 		uiNumChars = *puiWPStrLen >> 1;
	FLMUINT16 	ui16Diac;
	FLMUINT16 	ui16WpChar;

	// For each character (16 bits) in the WP string ...

	while( uiNumChars--)
	{
		// Have to skip 0, because it is not accounted for
		// in the sub-collation bits.  It was inserted when we
		// encountered unconverted unicode characters (Asian).
		// Will be converted to something else later on.
		// SEE NOTE ABOVE.

		if( FB2UW( pucWPStr) == 0)
		{
			pucWPStr += 2;
			continue;
		}

		// This macro DOESN'T increment uiBitPos

		if( testOneBit( pucSubColBuf, uiSubColBitPos))
		{
			//  Bits 10 - take next 5 bits
			//  Bits 110 align and take next word
			//  Bits 11110 align and take unicode value

			uiSubColBitPos++;
			if( !testOneBit( pucSubColBuf, uiSubColBitPos))
			{
				uiSubColBitPos++;
				ui16Diac = (FLMUINT16)(getNBits( 5, pucSubColBuf, uiSubColBitPos));
				uiSubColBitPos += 5;

				if( (ui16WpChar = FB2UW( pucWPStr)) < 0x100)
				{
					if( (ui16WpChar >= 'A') && (ui16WpChar <= 'Z'))
					{
						// Convert to WP diacritic and combine characters

						flmWPCmbcar( &ui16WpChar, ui16WpChar,
							(FLMUINT16)ml1_COLtoD[ ui16Diac]);

						// Even if cmbcar fails, WpChar is still set to a valid value
					}
					else
					{
						// Symbols from charset 0x24

						ui16WpChar = (FLMUINT16)(0x2400 +
							fwp_Ch24ColTbl[ ui16Diac - 1 ].ByteValue);
					}
				}
				else if( ui16WpChar >= 0x2600)	// Katakana
				{
					//  Voicings - will allow to select original char
					//		000 - some 001 are changed to 000 to save space
					//		001 - set if large char (uppercase)
					//		010 - set if voiced
					//		100 - set if half voiced
					//
					//  Should NOT match voicing or wouldn't be here!

					FLMBYTE ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

					// Try exceptions first so don't access out of bounds

					if( ucChar == 84)
					{
						ui16WpChar = (FLMUINT16)(0x2600 +
												((ui16Diac == 1)
												? (FLMUINT16)10
												: (FLMUINT16)11));
					}
					else if( ucChar == 85)
					{
						ui16WpChar = (FLMUINT16)(0x2600 +
												((ui16Diac == 1)
												 ? (FLMUINT16)16
												 : (FLMUINT16)17));
					}

					// Try the next 2 slots, if not then
					// value is 83, 84 or 85

					else if( KanaSubColTbl[ ucChar + 1 ] == ui16Diac)
					{
						ui16WpChar++;
					}
					else if( KanaSubColTbl[ ucChar + 2 ] == ui16Diac)
					{
						ui16WpChar += 2;
					}
					else if( ucChar == 4) // Last exception
					{
						ui16WpChar = 0x2600 + 83;
					}

					// else, leave alone! - invalid storage
				}

				UW2FBA( ui16WpChar, pucWPStr);	// Set if changed or not
			}
			else		// "110"
			{
				FLMUINT    uiTemp;

				uiSubColBitPos++;	// Skip second '1'
				if( testOneBit( pucSubColBuf, uiSubColBitPos))	// 11?10 ?
				{
					if( (*puiWPStrLen) + 2 > uiMaxWPBytes)
					{
						rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
						goto Exit;
					}

					// Unconvertable UNICODE character
					// The format will be 4 bytes, 0xFF, 0xFF, 2 byte Unicode

					shiftN( pucWPStr,
						(FLMUINT16)(uiNumChars + uiNumChars + 4), 2);

					pucWPStr += 2;	// Skip the 0xFFFF for now
					uiSubColBitPos += 2;	// Skip next "11"
					(*puiWPStrLen) += 2;
				}
				uiSubColBitPos++;	// Skip the zero

				// Round up to next byte
				uiSubColBitPos = (uiSubColBitPos + 7) & (~7);
				uiTemp = bytesInBits( uiSubColBitPos);
				pucWPStr[ 1] = pucSubColBuf[ uiTemp];	// Character set
				pucWPStr[ 0] = pucSubColBuf[ uiTemp + 1];	// Character
				uiSubColBitPos += 16;
			}
		}
		else
		{
			uiSubColBitPos++;	// Be sure to increment this!
		}

		pucWPStr += 2; // Next WP character
	}

	*puiSubColBitPos = bytesInBits( uiSubColBitPos);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		The case bits for asia are:
				Latin/Greek/Cyrillic
					01 - case bit set if character is uppercase
					10 - double wide character in CS 0x25xx, 0x26xx and 0x27xx
				Japanese
					00 - double wide hiragana 0x255e..25b0
					01 - double wide katakana 0x2600..2655
					10 - single wide symbols from charset 11 that map to CS24??
					11 - single wide katakana from charset 11
Ret:
Notes:	This is tricky to really understand the inputs.
			This looks at the bits according to the current character value.
****************************************************************************/
FSTATIC RCODE flmAsiaParseCase(
	FLMBYTE *			pucWPStr,
	FLMUINT *			puiWPStrLen,
	FLMUINT				uiMaxWPBytes,
	const FLMBYTE *	pucCaseBits,
	FLMUINT *			puiColBytesProcessed)
{
	RCODE			rc = NE_XFLM_OK;
	FLMUINT		uiWPStrLen = *puiWPStrLen;
	FLMUINT		uiCharCnt;
	FLMUINT		uiExtraBytes = 0;
	FLMUINT16	ui16WpChar;
	FLMBYTE		ucTempByte = 0;
	FLMBYTE		ucMaskByte;

	// For each character (two bytes) in the string ...

	for( uiCharCnt = uiWPStrLen >> 1,	// Total number of words in word string
			ucMaskByte = 0;						// Force first time to get a byte
			uiCharCnt--;)
	{
		FLMBYTE	ucChar;
		FLMBYTE	ucCharSet;

		ui16WpChar = FB2UW( pucWPStr);	// Get the next character

		// Must skip any 0xFFFFs or zeroes that were inserted.

		if( ui16WpChar == 0xFFFF || ui16WpChar == 0)
		{
			// Put back 0xFFFF in case it was a zero.

			UW2FBA( 0xFFFF, pucWPStr);
			pucWPStr += 2;
			uiExtraBytes += 2;
			continue;
		}

		// Time to get another byte?

		if( ucMaskByte == 0)
		{
			ucTempByte = *pucCaseBits++;
			ucMaskByte = 0x80;
		}

		ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
		ucChar = (FLMBYTE)(ui16WpChar & 0xFF);

		// SINGLE WIDE - NORMAL CHARACTERS

		if( ui16WpChar < 0x2400)
		{
			// Convert to double wide?

			if( ucTempByte & ucMaskByte)
			{
				// Latin/greek/cyrillic
				// Convert to uppercase double wide char

				if( ucCharSet == 0) // Latin - uppercase
				{
					// May convert to 0x250F (Latin) or CS24

					if( ui16WpChar >= ASCII_UPPER_A && ui16WpChar <= ASCII_UPPER_Z)
					{
						// Convert to double wide

						ui16WpChar = (FLMUINT16)(ui16WpChar - 0x30 + 0x250F);
					}
					else
					{
						flmWPHanToZenkaku( ui16WpChar, 0, &ui16WpChar);
					}
				}
				else if( ucCharSet == 8)	// Greek
				{
					if( ucChar > 38)	// Adjust for spaces in Greek
					{
						ucChar -= 2;
					}

					if( ucChar > 4)
					{
						ucChar -= 2;
					}

					ui16WpChar = (FLMUINT16)((ucChar >> 1) + 0x265E);
				}
				else if( ucCharSet == 10)	// Cyrillic
				{
					ui16WpChar = (FLMUINT16)((ucChar >> 1) + 0x2700);
				}
				else
				{
					flmWPHanToZenkaku( ui16WpChar, 0, &ui16WpChar);
				}

				ucCharSet = (FLMBYTE)(ui16WpChar >> 8);
				ucChar = (FLMBYTE)(ui16WpChar & 0xFF);
			}

			ucMaskByte >>= 1; // Next bit

			// Change to lower case?

			if( (ucTempByte & ucMaskByte) == 0)
			{
				// Convert ui16WpChar to lower case

				switch( ucCharSet)
				{
					case	0:
						// Bit zero only if lower case

						ui16WpChar |= 0x20;
						break;

					case	1:
						// In upper/lower case region?

						if( ucChar >= 26)
						{
							ui16WpChar++;
						}
						break;

					case	8:
						// All lowercase after 69

						if( ucChar <= 69)
						{
							ui16WpChar++;
						}
						break;

					case	10:
						// No cases after 199

						if( ucChar <= 199)
						{
							ui16WpChar++;
						}
						break;

					case	0x25:
					case	0x26:
						// Should be double wide latin or Greek
						// Add offset to convert to lowercase

						ui16WpChar += 0x20;
						break;

					case	0x27:
						// Double wide cyrillic only
						// Add offset to convert to lowercase

						ui16WpChar += 0x30;
						break;
				}
			}
		}
		else // JAPANESE CHARACTERS
		{
			if( ucTempByte & ucMaskByte)	// Original chars from CharSet 11
			{
				if( ucCharSet == 0x26)	// Convert to Zen to Hankaku
				{
					FLMUINT16	ui16NextChar = 0;

					ui16WpChar = flmWPZenToHankaku( ui16WpChar, &ui16NextChar);
					if( ui16NextChar)	// Move everyone down
					{
						if( (*puiWPStrLen) + 2 > uiMaxWPBytes)
						{
							rc = RC_SET( NE_XFLM_CONV_DEST_OVERFLOW);
							goto Exit;
						}

						uiCharCnt++;
						shiftN( pucWPStr, uiCharCnt + uiCharCnt + 2, 2);
						UW2FBA( ui16WpChar, pucWPStr);
						pucWPStr += 2;
						ui16WpChar = ui16NextChar;	// This will be stored below

						// Adjust the length
						*puiWPStrLen = *puiWPStrLen + 2;
					}
				}
				else if( ucCharSet == 0x24)
				{
					ui16WpChar = flmWPZenToHankaku( ui16WpChar, NULL);
				}
				ucMaskByte >>= 1;	// Eat the next bit
			}
			else
			{
				ucMaskByte >>= 1;	// Next bit
				if( (ucTempByte & ucMaskByte) == 0)	// Convert to Hiragana?
				{
					// Kanji will also fall through here

					if( ucCharSet == 0x26)
					{
						// Convert to Hiragana
						ui16WpChar = (FLMUINT16)(0x255E + ucChar);
					}
				}
			}
		}
		UW2FBA( ui16WpChar, pucWPStr);
		pucWPStr += 2;
		ucMaskByte >>= 1;
	}

	uiCharCnt = uiWPStrLen - uiExtraBytes;	// Should be 2 bits for each character.
	*puiColBytesProcessed = bytesInBits( uiCharCnt);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Convert a zenkaku (double wide) char to a hankaku (single wide) char
Ret:		Hankaku char or 0 if a conversion doesn't exist
Notes:	Taken from CHAR.ASM -  zen2han_f routine
****************************************************************************/
FSTATIC FLMUINT16 flmWPZenToHankaku(
	FLMUINT16	ui16WpChar,
	FLMUINT16 * pui16DakutenOrHandakuten)
{
	FLMUINT16	ui16Hankaku = 0;
	FLMBYTE		ucCharSet = ui16WpChar >> 8;
	FLMBYTE		ucCharVal = ui16WpChar & 0xFF;
	FLMUINT		uiLoop;

	switch( ucCharSet)
	{
		// SYMBOLS

		case 0x24:
		{
			for( uiLoop = 0;
				  uiLoop < (sizeof( Zen24ToHankaku) / sizeof( BYTE_WORD_TBL));
				  uiLoop++)
			{
				// List is sorted so table entry is more you are done

				if( Zen24ToHankaku [uiLoop].ByteValue >= ucCharVal)
				{
					if( Zen24ToHankaku [uiLoop].ByteValue == ucCharVal)
					{
						ui16Hankaku = Zen24ToHankaku [uiLoop].WordValue;
					}
					break;
				}
			}
			break;
		}

		// ROMAN - 0x250F..2559
		// Hiragana - 0x255E..2580

		case 0x25:
		{
			if( ucCharVal >= 0x0F && ucCharVal < 0x5E)
			{
				ui16Hankaku = ucCharVal + 0x21;
			}
			break;
		}

		// Katakana - 0x2600..2655
		// Greek - 0x265B..2695

		case 0x26:
		{
			if( ucCharVal <= 0x55)		// Katakana range
			{
				FLMBYTE		ucCS11CharVal;
				FLMUINT16	ui16NextWpChar = 0;

				if( (ucCS11CharVal = MapCS26ToCharSet11[ ucCharVal ]) != 0xFF)
				{
					if( ucCS11CharVal & 0x80)
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
					if( ui16NextWpChar && pui16DakutenOrHandakuten)
					{
						*pui16DakutenOrHandakuten = ui16NextWpChar;
					}
				}
			}
			else if( ucCharVal <= 0x95)	// Greek
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

				if( ucGreekChar >= 2)
				{
					ucGreekChar++;
				}

				if (ucGreekChar >= 19)
				{
					ucGreekChar++;
				}

				// Convert to character set 8

				ui16Hankaku = (ucGreekChar << 1) + 0x800;
				if( ucCharVal >= (0x5E + 0x20))
				{
					// Adjust to lower case character

					ui16Hankaku++;
				}
			}
			break;
		}

		// Cyrillic

		case 0x27:
		{
			// Uppercase?

			if( ucCharVal <= 0x20)
			{
				ui16Hankaku = (ucCharVal << 1) + 0xa00;
			}
			else if( ucCharVal >= 0x30 && ucCharVal <= 0x50)
			{
				// Lower case

				ui16Hankaku = ((ucCharVal - 0x30) << 1) + 0xa01;
			}
			break;
		}
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
FSTATIC FLMUINT16 flmWPHanToZenkaku(
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
		{
			// Invalid? - all others are used.

			if( ucCharVal < 0x20)
			{
				;
			}
			else if( ucCharVal <= 0x2F)
			{
				// Symbols A
				ui16Zenkaku = 0x2400 + From0AToZen[ ucCharVal - 0x20 ];
			}
			else if( ucCharVal <= 0x39)
			{
				// 0..9
				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if( ucCharVal <= 0x40)
			{
				// Symbols B
				ui16Zenkaku = 0x2400 + From0BToZen[ ucCharVal - 0x3A ];
			}
			else if( ucCharVal <= 0x5A)
			{
				// A..Z
				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if( ucCharVal <= 0x60)
			{
				// Symbols C
				ui16Zenkaku = 0x2400 + From0CToZen[ ucCharVal - 0x5B ];
			}
			else if( ucCharVal <= 0x7A)
			{
				// a..z
				ui16Zenkaku = 0x2500 + (ucCharVal - 0x21);
			}
			else if( ucCharVal <= 0x7E)
			{
				// Symbols D
				ui16Zenkaku = 0x2400 + From0DToZen[ ucCharVal - 0x7B ];
			}
			break;
		}

		// GREEK

		case 8:
		{
			if( (ucCharVal >= sizeof( From8ToZen)) ||
				 ((ui16Zenkaku = 0x2600 + From8ToZen[ ucCharVal ]) == 0x26FF))
			{
				ui16Zenkaku = 0;
			}
			break;
		}

		// CYRILLIC

		case 10:
		{
			// Check range

			ui16Zenkaku = 0x2700 + (ucCharVal >> 1);	// Uppercase value

			// Convert to lower case?

			if( ucCharVal & 0x01)
			{
				ui16Zenkaku += 0x30;
			}
			break;
		}

		// JAPANESE

		case 11:
		{
			if( ucCharVal < 5)
			{
				ui16Zenkaku = 0x2400 + From11AToZen[ ucCharVal ];
			}
			else if( ucCharVal < 0x3D)		// katakana?
			{
				if( (ui16Zenkaku = 0x2600 +
							From11BToZen[ ucCharVal - 5 ]) == 0x26FF)
				{
					// Dash - convert to this
					ui16Zenkaku = 0x241b;
				}
				else
				{
					if( ui16NextWpChar == 0xB3D)		// dakuten? - voicing
					{
						// First check exception(s) then
						// check if voicing exists! - will NOT access out of table

						if( (ui16Zenkaku != 0x2652) &&	// is not 'N'?
							 (KanaSubColTbl[ ui16Zenkaku - 0x2600 + 1 ] == 3))
						{
							ui16Zenkaku++;

							// Return 2

							ui16CharsUsed++;
						}
					}
					else if( ui16NextWpChar == 0xB3E)	// handakuten? - voicing
					{
						// Check if voicing exists! - will NOT access out of table

						if( KanaSubColTbl [ui16Zenkaku - 0x2600 + 2 ] == 5)
						{
							ui16Zenkaku += 2;

							// Return 2

							ui16CharsUsed++;
						}
					}
				}
			}
			else if( ucCharVal == 0x3D)		// dakuten?
			{
				// Convert to voicing symbol

				ui16Zenkaku = 0x240A;
			}
			else if( ucCharVal == 0x3E)		// handakuten?
			{
				// Convert to voicing symbol

				ui16Zenkaku = 0x240B;
			}
			// else cannot convert

			break;
		}

		// Other character sets
		// CS 1,4,5,6 - symbols

		default:
		{
			// Instead of includes more tables from char.asm - look down the
			// Zen24Tohankaku[] table for a matching value - not much slower.

			for( uiLoop = 0;
				  uiLoop < (sizeof(Zen24ToHankaku) / sizeof(BYTE_WORD_TBL));
				  uiLoop++)
			{
				if( Zen24ToHankaku[ uiLoop].WordValue == ui16WpChar)
				{
					ui16Zenkaku = 0x2400 + Zen24ToHankaku[ uiLoop].ByteValue;
					break;
				}
			}
			break;
		}
	}

	if( !ui16Zenkaku)
	{
		// Change return value

		ui16CharsUsed = 0;
	}

	*pui16Zenkaku = ui16Zenkaku;
	return( ui16CharsUsed);
}

/****************************************************************************
Desc:		Converts a 2-byte language code into its corresponding language ID
****************************************************************************/
FLMUINT F_DbSystem::languageToNum(
	const char *	pszLanguage)
{
	FLMBYTE		ucFirstChar  = (FLMBYTE)(*pszLanguage);
	FLMBYTE		ucSecondChar = (FLMBYTE)(*(pszLanguage + 1));
	FLMUINT		uiTablePos;

	for( uiTablePos = 0; 
		uiTablePos < (LAST_LANG + LAST_LANG); uiTablePos += 2)
	{
		if( fwp_langtbl [uiTablePos]   == ucFirstChar &&
			 fwp_langtbl [uiTablePos+1] == ucSecondChar)
		{

			// Return uiTablePos div 2

			return( uiTablePos >> 1);
		}
	}

	// Language not found, return default US language

	return( XFLM_US_LANG);
}

/****************************************************************************
Desc:		Converts a language ID to its corresponding 2-byte language code
****************************************************************************/
void F_DbSystem::languageToStr(
	FLMINT	iLangNum,
	char *	pszLanguage)
{
	// iLangNum could be negative

	if( iLangNum < 0 || iLangNum >= LAST_LANG)
	{
		iLangNum = XFLM_US_LANG;
	}

	iLangNum += iLangNum;
	*pszLanguage++ = (char)fwp_langtbl [iLangNum ];
	*pszLanguage++ = (char)fwp_langtbl [iLangNum+1];
	*pszLanguage = 0;
}

/***************************************************************************
Desc:	Return the sub-collation value of a WP character.  Unconverted
		unicode values always have a sub-collation value of
		11110+UnicodeChar
***************************************************************************/
FLMUINT16 flmWPGetSubCol(
	FLMUINT16		ui16WPValue,		// [in] WP Character value.
	FLMUINT16		ui16ColValue,		// [in] Collation Value (for arabic)
	FLMUINT			uiLanguage)			// [in] WP Language ID.
{
	FLMUINT16		ui16SubColVal;
	FLMBYTE			ucCharVal;
	FLMBYTE			ucCharSet;
	FLMUINT16		ui16Base;

	// Easy case first - ascii characters.

	ui16SubColVal = 0;
	if (ui16WPValue <= 127)
	{
		goto Exit;
	}

	// From here down default ui16SubColVal is WP value.

	ui16SubColVal = ui16WPValue;
	ucCharVal = (FLMBYTE) ui16WPValue;
	ucCharSet = (FLMBYTE) (ui16WPValue >> 8);

	// Convert char to uppercase because case information
	// is stored above.  This will help
	// ensure that the "ETA" doesn't sort before "eta"
	// could use is lower code here for added performance.

	// This just happens to work with all WP character values.

	if (!flmWPIsUpper( ui16WPValue))
	{
		ui16WPValue &= ~1;
	}

	switch (ucCharSet)
	{
		case CHSMUL1:

			// If you cannot break down a char into base and
			// diacritic then you cannot combine the charaacter
			// later when converting back the key.  So, write
			// the entire WP char in the sub-collation area.
			// We can ONLY SUPPORT MULTINATIONAL 1 for brkcar()

			if (flmWPBrkcar( ui16WPValue, &ui16Base, &ui16SubColVal))
			{

				// WordPerfect character cannot be broken down.
				// If we had a collation value other than 0xFF (COLS0), don't
				// return a sub-collation value.  This will allow things like
				// upper and lower AE digraphs to compare properly.

				if (ui16ColValue != COLS0)
				{
					ui16SubColVal = 0;
				}
				goto Exit;
			}

			// Write the FLAIM diacritic sub-collation value.
			// Prefix is 2 bits "10".  Remember to leave
			// "111" alone for the future.
			// Bug 11/16/92 = was only writing a "1" and not "10"

			ui16SubColVal = (
					(ui16SubColVal & 0xFF) == umlaut
					&& ( (uiLanguage == XFLM_SU_LANG) ||
						  (uiLanguage == XFLM_SV_LANG) ||
						  (uiLanguage == XFLM_CZ_LANG) ||
						  (uiLanguage == XFLM_SL_LANG)
						)
					)
				?	(FLMUINT16)(fwp_dia60Tbl[ ring] + 1)	// umlaut must be after ring above
				:	(FLMUINT16)(fwp_dia60Tbl[ ui16SubColVal & 0xFF]);

			break;

		case CHSGREK:

			// Greek

			if( (ucCharVal >= 52)  ||		// Keep case bit for 52-69 else ignore
          	 (ui16WPValue == 0x804) ||	// [ 8,4] BETA Medial | Terminal
				 (ui16WPValue == 0x826)) 	// [ 8,38] SIGMA termainal
			{
				ui16SubColVal = ui16WPValue;
			}
			// else no subcollation to worry about
			break;

		case CHSCYR:
			if (ucCharVal >= 144)
			{
				ui16SubColVal = ui16WPValue;
			}
			// else no subcollation to worry about

			// VISIT: Georgian covers 208-249 - no collation defined yet
			break;

		case CHSHEB:	// Hebrew

			// Three sections in Hebrew:
			//		0..26 - main characters
			//		27..83 - accents that apear over previous character
			//		84..118- dagesh (ancient) hebrew with accents

			// Because the ancient is only used for sayings & scriptures
			// we will support a collation value and in the sub-collation
			// store the actual character because sub-collation is in
			// character order.

         if (ucCharVal >= 84) 		// Save ancient - value 84 and above
			{
				ui16SubColVal = ui16WPValue;
			}
			break;

		case CHSARB1:	// Arabic 1

			// Three sections in Arabic:
			//		00..37  - accents that display OVER a previous character
			//		38..46  - symbols
			//		47..57  - numbers
			//		58..163 - characters
			//		164     - hamzah accent
			//		165..180- common characters with accents
			//		181..193- ligatures - common character combinations
			//		194..195- extensions - throw away when sorting

			if (ucCharVal <= 46)
			{
				ui16SubColVal = ui16WPValue;
			}
			else
			{
				if (ui16ColValue == COLS10a+1)	// Alef?
				{
					ui16SubColVal = (ucCharVal >= 165)
						? (FLMUINT16)(fwp_alefSubColTbl[ ucCharVal - 165 ])
						: (FLMUINT16)7;		// Alef subcol value
				}
				else
				{
					if (ucCharVal >= 181)		// Ligatures - char combination
					{
						ui16SubColVal = ui16WPValue;
					}
					else if (ucCharVal == 64)	// taa exception
					{
						ui16SubColVal = 8;
					}
				}
			}
			break;

		case CHSARB2:			// Arabic 2

			// There are some characters that share the same slot
			// Check the bit table if above character 64

			if ((ucCharVal >= 64) &&
				 (fwp_ar2BitTbl[(ucCharVal-64)>> 3] & (0x80 >> (ucCharVal&0x07))))
			{
				ui16SubColVal = ui16WPValue;
			}
			break;

	}

Exit:

	return( ui16SubColVal);
}
