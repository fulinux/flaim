//-------------------------------------------------------------------------
// Desc:	WP collation tables.
// Tabs:	3
//
//		Copyright (c) 1991-2001,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fwpcoll.cpp 12301 2006-01-19 15:02:55 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

FLMUINT16 fwp_indexi[] = {0,11,14,15,17,18,19,21,22,23,24,25,26,35,59};

FLMUINT16 fwp_indexj[] = {
													/** DOUBLE CHAR AREA - LANGUAGES */
/*0*/	CA_LANG,	/* Catalan */
		CF_LANG,	/* Canadian French */
		CZ_LANG,	/* Czech */
		SL_LANG,	/* Slovak */
		DE_LANG,	/* German */
		SD_LANG,	/* Swiss German */
		ES_LANG,	/* Spanish (Spain) */
		FR_LANG,	/* French */
		_NL_LANG,	/* Netherlands */
		0xFFFF,	/* DK_LANG,	Danish    - support for 'aa' -> a-ring out */
		0xFFFF,	/* NO_LANG,	Norwegian - support for 'aa' -> a-ring out */
/*11*/0x0063,	/* c */						/* DOUBLE CHARACTERS - STATE ENTRIES */
		0x006c,	/* l */
		0x0197,	/* l with center dot */
/*14*/0x0063,	/* c */
/*15*/0x0125,	/* ae digraph  */
		0x01a7,	/* oe digraph */
/*17*/0x0068,	/* h */
/*18*/0x0068,	/* h */
/*19*/0x006c,	/* l */
		0x0101,	/* center dot alone */
/*21*/0x006c,	/* l */
/*22*/0x0117,	/* А	(for German) */
/*23*/0x018b,	/* ij digraph */
		0x0000,	/* was 'a' - will no longer map 'aa' to a-ring */
		0x0000,	/* was 'a' */
		
/*26*/CZ_LANG,									/* SINGLE CHARS - LANGUAGES */
		DK_LANG,
		NO_LANG,
		SL_LANG,
		TK_LANG,
		SU_LANG,
		IS_LANG,
		SV_LANG,
		YK_LANG,
													/* SINGLE CHARS */
/*35*/0x011e, /* A Diaeresis */		/* alternate collating sequences */
		0x011f, /* a Diaeresis  */
		0x0122, /* A Ring */							/* 2 */
		0x0123, /* a Ring  */
		0x0124, /* AE Diagraph */					/* 4 */
		0x0125, /* ae diagraph */
		0x013e, /* O Diaeresis */					/* 6 */
		0x013f, /* o Diaeresis */
		0x0146, /* U Diaeresis */					/* 8 */
		0x0147, /* u Diaeresis */
		0x0150, /* O Slash */						/* 10 */
		0x0151, /* o Slash */

		0x0A3a, /* CYRILLIC SOFT SIGN */			/* 12 */
		0x0A3b, /* CYRILLIC soft sign */
      0x01ee, /* dotless i - turkish */		/* 14 */
/*50*/0x01ef, /* dotless I - turkish */
		0x0162, /* C Hacek/caron - 1,98 */		/* 16 */
		0x0163, /* c Hacek/caron - 1,99 */
		0x01aa, /* R Hacek/caron - 1,170*/		/* 18 */
		0x01ab, /* r Hacek/caron - 1,171*/
		0x01b0, /* S Hacek/caron - 1,176*/		/* 20 */
		0x01b1, /* s Hacek/caron - 1,177*/
		0x01ce, /* Z Hacek/caron - 1,206*/		/* 22 */
/*58*/0x01cf, /* z Hacek/caron - 1,207*/
		}; 


FLMUINT16 fwp_valuea[] = {
/*00*/STATE1,									/* DOUBLE CHAR STATE VALUES */
		STATE3,
		STATE2,
      STATE2,
		STATE8,
		STATE8,
		STATE1,
		STATE3,
		STATE9,
		STATE10,		/* No longer in use */
/*10*/STATE10,		/* No longer in use */
		STATE4,
		STATE6,
		STATE6,
		STATE5,
		INSTAE,
		INSTOE,
		AFTERC,
		AFTERH,
      AFTERL,
/*20*/STATE7,
		STATE6,
		INSTSG,		/* ss for German */
		INSTIJ,
		STATE11,		/* aa - no longer in use */
		WITHAA,		/* aa - no longer in use */
													/* SINGLE CHARS - LANGUAGES */
/*26*/START_CZ,		/* Czech 		*/
		START_DK,		/* Danish 		*/
		START_NO,		/* Norwegian	*/
		START_SL,		/* Slovak 		*/
		START_TK,		/* Turkish 		*/
		START_SU,		/* Finnish  	*/
		START_IS,		/* Icelandic	*/
		START_SV,		/* Swedish 		*/
		START_YK,		/* Ukrainian	*/
													/* SINGLE CHARS FIXUP AREAS */
/*35*/COLS9,		COLS9,		COLS9,		COLS9,		/* US & OTHERS */
		COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
		COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
		COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,
		
		COLS9+45,	COLS9+45,	COLS9+55,	COLS9+55,	/* DANISH */
		COLS9+42,	COLS9+42,	COLS9+53,	COLS9+53,
		COLS9+30,	COLS9+30,	COLS9+49,	COLS9+49,	/* Oct98 U Diaer no longer to y Diaer */
		COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

		COLS9,		COLS9,		COLS9,		COLS9,		/* Icelandic */
		COLS9+46,	COLS9+46,	COLS9+50,	COLS9+50,
		COLS9+30,	COLS9+30,	COLS9+54,	COLS9+54,
		COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

		COLS9,		COLS9,		COLS9+51,	COLS9+51,	/* Norwegian */
		COLS9+43,	COLS9+43,	COLS9+21,	COLS9+21,
		COLS9+30,	COLS9+30,	COLS9+47,	COLS9+47,
		COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

		COLS9+48,	COLS9+48,	COLS9+44,	COLS9+44,	/* Finnish/Swedish*/
		COLS9+1,		COLS9+1,		COLS9+52,	COLS9+52,
		COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,	/* Oct98 U Diaer no longer to y Diaer */
		COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

		COLS9,		COLS9,		COLS9,		COLS9,		/* Ukrain */
		COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
		COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
		COLS10+48,	COLS10+48,	COLS9+12,	COLS9+12,
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,

		COLS9,		COLS9,		COLS9,		COLS9,		/* Turkish */
		COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
		COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
		COLS9+43,	COLS9+43,	COLS9+11,	COLS9+11,	/* dotless i same as */
		COLS9+3,		COLS9+3,		COLS9+25,	COLS9+25,	/* the "CH" in Czech */
		COLS9+27,	COLS9+27,	COLS9+35,	COLS9+35,	/* works because char*/
																		/* fails brkcar()    */

		COLS9,		COLS9,		COLS9,		COLS9,		/* Czech / Slovak */
		COLS9+1,		COLS9+1,		COLS9+21,	COLS9+21,
		COLS9+30,	COLS9+30,	COLS9+21,	COLS9+21,
		COLS10+43,	COLS10+43,	COLS9+12,	COLS9+12,
		COLS9+5,		COLS9+5,		COLS9+26,	COLS9+26,	/* carons */
		COLS9+28,	COLS9+28,	COLS9+36,	COLS9+36
	};

#define ASCTBLLEN 95	

FLMBYTE fwp_asc60Tbl[ASCTBLLEN+2] = {
	0x20,			/* initial character offset!! */
	ASCTBLLEN,	/* len of this table */
	COLLS,		/* <Spc> */
	COLLS+5,		/* ! */
	COLS1,		/* " */
	COLS6+1,		/* # */
	COLS3,		/* $ */
	COLS6,		/* % */
	COLS6+2,		/* & */
	COLS1+1,		/* ' */
	COLS2,		/* ( */
	COLS2+1,		/* ) */
	COLS4+2,		/* * */
	COLS4,		/* + */
	COLLS+2,		/* , */
	COLS4+1,		/* - */
	COLLS+1,		/* . */
	COLS4+3,		/* / */
	COLS8,		/* 0 */
	COLS8+1,		/* 1 */
	COLS8+2,		/* 2 */
	COLS8+3,		/* 3 */
	COLS8+4,		/* 4 */
	COLS8+5,		/* 5 */
	COLS8+6,		/* 6 */
	COLS8+7,		/* 7 */
	COLS8+8,		/* 8 */
	COLS8+9,		/* 9 */
	COLLS+3,		/* : */
	COLLS+4,		/* ; */
	COLS5,		/* < */
	COLS5+2,		/* = */
	COLS5+4,		/* > */
	COLLS+7,		/* ? */
	COLS6+3,		/* @ */
	COLS9,		/* A */
/*COLS9+1			Holder for AE digraph */
	COLS9+2,		/* B */
	COLS9+3,		/* C */
/*cols9+4			CH in spanish */
/*cols9+5			Holder for C caron in Czech */
	COLS9+6,		/* D */
	COLS9+7,		/* E */
	COLS9+8,		/* F */
	COLS9+9,		/* G */
	COLS9+10,	/* H */
/*cols9+11		CH in czech */
	COLS9+12,		/* I */
/*cols9+13		Holder for IJ digraph */
	COLS9+14,		/* J */
	COLS9+15,		/* K */
	COLS9+16,		/* L */
/*cols9+17		LL in spanish */
	COLS9+18,		/* M */
	COLS9+19,		/* N */
/*cols9+20		╔ */
	COLS9+21,		/* O */
/*cols9+22		Holder for OE digraph */
	COLS9+23,		/* P */
	COLS9+24,		/* Q */
	COLS9+25,		/* R */
/*cols9+26		Holder for R caron in Czech */
	COLS9+27,		/* S */
/*cols9+28		Holder for S caron in Czech */
	COLS9+29,		/* T */
	COLS9+30,		/* U */
	COLS9+31,		/* V */
	COLS9+32,		/* W */
	COLS9+33,		/* X */
	COLS9+34,		/* Y */
	COLS9+35,		/* Z */
/*cols9+36		Holder for Z caron in Czech */
	COLS9+40,		/* [ (note: alphabetic - end of list) */
	COLS6+4,		/* \ */
	COLS9+41,		/* ] (note: alphabetic - end of list) */
	COLS4+4,		/* ^ */
	COLS6+5,		/* _ */
	COLS1+2,		/* ` */
	COLS9,			/* a */
/*cols9+1			Holder for ae digraph */
	COLS9+2,		/* b */
	COLS9+3,		/* c */
/*cols9+4			ch in spanish */
/*cols9+5			Holder for c caron in Czech */
	COLS9+6,		/* d */
	COLS9+7,		/* e */
	COLS9+8,		/* f */
	COLS9+9,		/* g */
	COLS9+10,		/* h */
/*cols9+11		ch in czech */
	COLS9+12,		/* i */
/*cols9+13		Holder for ij digraph */
	COLS9+14,		/* j */
	COLS9+15,		/* k */
	COLS9+16,		/* l */
/*cols9+17		ll in spanish */
	COLS9+18,		/* m */
	COLS9+19,		/* n */
/*cols9+20		╔ */
	COLS9+21,		/* o */
/*cols9+22		Holder for oe digraph */
	COLS9+23,		/* p */
	COLS9+24,		/* q */
	COLS9+25,		/* r */
/*cols9+26		Holder for r caron in Czech */
	COLS9+27,		/* s */
/*cols9+28		Holder for s caron in Czech */
	COLS9+29,		/* t */
	COLS9+30,		/* u */
	COLS9+31,		/* v */
	COLS9+32,		/* w */
	COLS9+33,		/* x */
	COLS9+34,		/* y */
	COLS9+35,		/* z */
/*cols9+36		Holder for Z caron in Czech */
	COLS2+4,		/* { */
	COLS6+6,		/* | */
	COLS2+5,		/* } */
	COLS6+7			/* ~ */
};

#define MNTBLLEN 219

FLMBYTE fwp_mn60Tbl[MNTBLLEN+2] = {		/* multinational table	*/
	23,		/* initial character offset!! */
	MNTBLLEN,	/* len of this table */
	COLS9+27,	/* German Double s */
	COLS9+15,	/* Icelandic k */
	COLS9+14,	/* Dotless j */

/* IBM Charset */

	COLS9,		/* A Acute */
	COLS9,		/* a Acute */
	COLS9,		/* A Circumflex */
	COLS9,		/* a Circumflex */
	COLS9,		/* A Diaeresis or Umlaut */
	COLS9,		/* a Diaeresis or Umlaut */
	COLS9,		/* A Grave */
	COLS9,		/* a Grave */
	COLS9,		/* A Ring */
	COLS9,		/* a Ring */
	COLS9+1,		/* AE digraph */
	COLS9+1,		/* ae digraph */
	COLS9+3,		/* C Cedilla */
	COLS9+3,		/* c Cedilla */
	COLS9+7,		/* E Acute */
	COLS9+7,		/* e Acute */
	COLS9+7,		/* E Circumflex */
	COLS9+7,		/* e Circumflex */
	COLS9+7,		/* E Diaeresis or Umlaut */
	COLS9+7,		/* e Diaeresis or Umlaut */
	COLS9+7,		/* E Grave */
	COLS9+7,		/* e Grave */
	COLS9+12,	/* I Acute */
	COLS9+12,	/* i Acute */
	COLS9+12,	/* I Circumflex */
	COLS9+12,	/* i Circumflex */
	COLS9+12,	/* I Diaeresis or Umlaut */
	COLS9+12,	/* i Diaeresis or Umlaut */
	COLS9+12,	/* I Grave */
	COLS9+12,	/* i Grave */
	COLS9+20,	/* N Tilde */
	COLS9+20,	/* n Tilde */
	COLS9+21,	/* O Acute */
	COLS9+21,	/* o Acute */
	COLS9+21,	/* O Circumflex */
	COLS9+21,	/* o Circumflex */
	COLS9+21,	/* O Diaeresis or Umlaut */
	COLS9+21,	/* o Diaeresis or Umlaut */
	COLS9+21,	/* O Grave */
	COLS9+21,	/* o Grave */
	COLS9+30,	/* U Acute */
	COLS9+30,	/* u Acute */
	COLS9+30,	/* U Circumflex */
	COLS9+30,	/* u Circumflex */
	COLS9+30,	/* U Diaeresis or Umlaut */
	COLS9+30,	/* u Diaeresis or Umlaut */
	COLS9+30,	/* U Grave */
	COLS9+30,	/* u Grave */
	COLS9+34,	/* Y Diaeresis or Umlaut */
	COLS9+34,	/* y Diaeresis or Umlaut */

/* IBM foreign */

	COLS9,		/* A Tilde */
	COLS9,		/* a Tilde */
	COLS9+6,		/* D Cross Bar */
	COLS9+6,		/* d Cross Bar */
	COLS9+21,	/* O Slash */
	COLS9+21,	/* o Slash */
	COLS9+21,	/* O Tilde */
	COLS9+21,	/* o Tilde */
	COLS9+34,	/* Y Acute */
	COLS9+34,	/* y Acute */
	COLS9+6,		/* Uppercase Eth */
	COLS9+6,		/* Lowercase Eth */
	COLS9+37,	/* Uppercase Thorn */
	COLS9+37,	/* Lowercase Thorn */

/* Teletex chars */

	COLS9,		/* A Breve */
	COLS9,		/* a Breve */
	COLS9,		/* A Macron */
	COLS9,		/* a Macron */
	COLS9,		/* A Ogonek */
	COLS9,		/* a Ogonek */
	COLS9+3,		/* C Acute */
	COLS9+3,		/* c Acute */
	COLS9+3,		/* C Caron or Hachek */
	COLS9+3,		/* c Caron or Hachek */
	COLS9+3,		/* C Circumflex */
	COLS9+3,		/* c Circumflex */
	COLS9+3,		/* C Dot Above */
	COLS9+3,		/* c Dot Above */
	COLS9+6,		/* D Caron or Hachek (Apostrophe Beside) */
	COLS9+6,		/* d Caron or Hachek (Apostrophe Beside) */
	COLS9+7,		/* E Caron or Hachek */
	COLS9+7,		/* e Caron or Hachek */
	COLS9+7,		/* E Dot Above */
	COLS9+7,		/* e Dot Above */
	COLS9+7,		/* E Macron */
	COLS9+7,		/* e Macron */
	COLS9+7,		/* E Ogonek */
	COLS9+7,		/* e Ogonek */
	COLS9+9,		/* G Acute */
	COLS9+9,		/* g Acute */
	COLS9+9,		/* G Breve */
	COLS9+9,		/* g Breve */
	COLS9+9,		/* G Caron or Hachek */
	COLS9+9,		/* g Caron or Hachek */
	COLS9+9,		/* G Cedilla (Apostrophe Under) */
	COLS9+9,		/* g Cedilla (Apostrophe Over) */
	COLS9+9,		/* G Circumflex */
	COLS9+9,		/* g Circumflex */
	COLS9+9,		/* G Dot Above */
	COLS9+9,		/* g Dot Above */
	COLS9+10,	/* H Circumflex */
	COLS9+10,	/* h Circumflex */
	COLS9+10,	/* H Cross Bar */
	COLS9+10,	/* h Cross Bar */
	COLS9+12,	/* I Dot Above (Sharp Accent) */
	COLS9+12,	/* i Dot Above (Sharp Accent) */
	COLS9+12,	/* I Macron */
	COLS9+12,	/* i Macron */
	COLS9+12,	/* I Ogonek */
	COLS9+12,	/* i Ogonek */
	COLS9+12,	/* I Tilde */
	COLS9+12,	/* i Tilde */
	COLS9+13,	/* IJ Digraph */
	COLS9+13,	/* ij Digraph */
	COLS9+14,	/* J Circumflex */
	COLS9+14,	/* j Circumflex */
	COLS9+15,	/* K Cedilla (Apostrophe Under) */
	COLS9+15,	/* k Cedilla (Apostrophe Under) */
	COLS9+16,	/* L Acute */
	COLS9+16,	/* l Acute */
	COLS9+16,	/* L Caron or Hachek (Apostrophe Beside) */
	COLS9+16,	/* l Caron or Hachek (Apostrophe Beside) */
	COLS9+16,	/* L Cedilla (Apostrophe Under) */
	COLS9+16,	/* l Cedilla (Apostrophe Under) */
	COLS9+16,	/* L Center Dot */
	COLS9+16,	/* l Center Dot */
	COLS9+16,	/* L Stroke */
	COLS9+16,	/* l Stroke */
	COLS9+19,	/* N Acute */
	COLS9+19,	/* n Acute */
	COLS9+19,	/* N Apostrophe */
	COLS9+19,	/* n Apostrophe */
	COLS9+19,	/* N Caron or Hachek */
	COLS9+19,	/* n Caron or Hachek */
	COLS9+19,	/* N Cedilla (Apostrophe Under) */
	COLS9+19,	/* n Cedilla (Apostrophe Under) */
	COLS9+21,	/* O Double Acute */
	COLS9+21,	/* o Double Acute */
	COLS9+21,	/* O Macron */
	COLS9+21,	/* o Macron */
	COLS9+22,	/* OE digraph */
	COLS9+22,	/* oe digraph */
	COLS9+25,	/* R Acute */
	COLS9+25,	/* r Acute */
	COLS9+25,	/* R Caron or Hachek */
	COLS9+25,	/* r Caron or Hachek */
	COLS9+25,	/* R Cedilla (Apostrophe Under) */
	COLS9+25,	/* r Cedilla (Apostrophe Under) */
	COLS9+27,	/* S Acute */
	COLS9+27,	/* s Acute */
	COLS9+27,	/* S Caron or Hachek */
	COLS9+27,	/* s Caron or Hachek */
	COLS9+27,	/* S Cedilla */
	COLS9+27,	/* s Cedilla */
	COLS9+27,	/* S Circumflex */
	COLS9+27,	/* s Circumflex */
	COLS9+29,	/* T Caron or Hachek (Apostrophe Beside) */
	COLS9+29,	/* t Caron or Hachek (Apostrophe Beside) */
	COLS9+29,	/* T Cedilla (Apostrophe Under) */
	COLS9+29,	/* t Cedilla (Apostrophe Under) */
	COLS9+29,	/* T Cross Bar */
	COLS9+29,	/* t Cross Bar */
	COLS9+30,	/* U Breve */
	COLS9+30,	/* u Breve */
	COLS9+30,	/* U Double Acute */
	COLS9+30,	/* u Double Acute */
	COLS9+30,	/* U Macron */
	COLS9+30,	/* u Macron */
	COLS9+30,	/* U Ogonek */
	COLS9+30,	/* u Ogonek */
	COLS9+30,	/* U Ring */
	COLS9+30,	/* u Ring */
	COLS9+30,	/* U Tilde */
	COLS9+30,	/* u Tilde */
	COLS9+32,	/* W Circumflex */
	COLS9+32,	/* w Circumflex */
	COLS9+34,	/* Y Circumflex */
	COLS9+34,	/* y Circumflex */
	COLS9+35,	/* Z Acute */
	COLS9+35,	/* z Acute */
	COLS9+35,	/* Z Caron or Hachek */
	COLS9+35,	/* z Caron or Hachek */
	COLS9+35,	/* Z Dot Above */
	COLS9+35,	/* z Dot Above */
	COLS9+19,	/* Uppercase Eng */
	COLS9+19,	/* Lowercase Eng */

/* other */

	COLS9+6,		/* D Macron */
	COLS9+6,		/* d Macron */
	COLS9+16,	/* L Macron */
	COLS9+16,	/* l Macron */
	COLS9+19,	/* N Macron */
	COLS9+19,	/* n Macron */
	COLS9+25,	/* R Grave */
	COLS9+25,	/* r Grave */
	COLS9+27,	/* S Macron */
	COLS9+27,	/* s Macron */
	COLS9+29,	/* T Macron */
	COLS9+29,	/* t Macron */
	COLS9+34,	/* Y Breve */
	COLS9+34,	/* y Breve */
	COLS9+34,	/* Y Grave */
	COLS9+34,	/* y Grave */
	COLS9+6,		/* D Apostrophe Beside */
	COLS9+6,		/* d Apostrophe Beside */
	COLS9+21,	/* O Apostrophe Beside */
	COLS9+21,	/* o Apostrophe Beside */
	COLS9+30,	/* U Apostrophe Beside */
	COLS9+30,	/* u Apostrophe Beside */
	COLS9+7,		/* E breve */
	COLS9+7,		/* e breve */
	COLS9+12,	/* I breve */
	COLS9+12,	/* i breve */
	COLS9+12,	/* dotless I */
	COLS9+12,	/* dotless i */
	COLS9+21,	/* O breve */
	COLS9+21		/* o breve */
};

#define SYMTBLLEN 9

FLMBYTE fwp_sym60Tbl[SYMTBLLEN+2] = {
	11,			/* initial character offset!! */
	SYMTBLLEN,	/* len of this table */
	COLS3+2,		/* pound */
	COLS3+3,		/* yen */
	COLS3+4,		/* pacetes */
	COLS3+5,		/* floren */
	COLS0,	
	COLS0,
	COLS0,
	COLS0,
	COLS3+1,		/* cent */
};


/**----------------------------------------------
***  This is defined for the full greek table.
***---------------------------------------------*/

#define	GRKTBLLEN	219

FLMBYTE fwp_grk60Tbl[GRKTBLLEN+2] = {
	0,		/* starting offset */
	GRKTBLLEN,		/* length */
	COLS7,			/* Uppercase Alpha */
	COLS7,			/* Lowercase Alpha */
	COLS7+1,			/* Uppercase Beta */
	COLS7+1,			/* Lowercase Beta */
	COLS7+1,			/* Uppercase Beta Medial */
	COLS7+1,			/* Lowercase Beta Medial */
	COLS7+2,			/* Uppercase Gamma */
	COLS7+2,			/* Lowercase Gamma */
	COLS7+3,			/* Uppercase Delta */
	COLS7+3,			/* Lowercase Delta */
	COLS7+4,			/* Uppercase Epsilon */
	COLS7+4,			/* Lowercase Epsilon */
	COLS7+5,			/* Uppercase Zeta */
	COLS7+5,			/* Lowercase Zeta */
	COLS7+6,			/* Uppercase Eta */
	COLS7+6,			/* Lowercase Eta */
	COLS7+7,			/* Uppercase Theta */
	COLS7+7,			/* Lowercase Theta */
	COLS7+8,			/* Uppercase Iota */
	COLS7+8,			/* Lowercase Iota */
	COLS7+9,			/* Uppercase Kappa */
	COLS7+9,			/* Lowercase Kappa */
	COLS7+10,		/* Uppercase Lambda */
	COLS7+10,		/* Lowercase Lambda */
	COLS7+11,		/* Uppercase Mu */
	COLS7+11,		/* Lowercase Mu */
	COLS7+12,		/* Uppercase Nu */
	COLS7+12,		/* Lowercase Nu */
	COLS7+13,		/* Uppercase Xi */
	COLS7+13,		/* Lowercase Xi */
	COLS7+14,		/* Uppercase Omicron */
	COLS7+14,		/* Lowercase Omicron */
	COLS7+15,		/* Uppercase Pi */
	COLS7+15,		/* Lowercase Pi */
	COLS7+16,		/* Uppercase Rho */
	COLS7+16,		/* Lowercase Rho */
	COLS7+17,		/* Uppercase Sigma */
	COLS7+17,		/* Lowercase Sigma */
	COLS7+17,		/* Uppercase Sigma Terminal */
	COLS7+17,		/* Lowercase Sigma Terminal */
	COLS7+18,		/* Uppercase Tau */
	COLS7+18,		/* Lowercase Tau */
	COLS7+19,		/* Uppercase Upsilon */
	COLS7+19,		/* Lowercase Upsilon */
	COLS7+20,		/* Uppercase Phi */
	COLS7+20,		/* Lowercase Phi */
	COLS7+21,		/* Uppercase Chi */
	COLS7+21,		/* Lowercase Chi */
	COLS7+22,		/* Uppercase Psi */
	COLS7+22,		/* Lowercase Psi */
	COLS7+23,		/* Uppercase Omega */
	COLS7+23,		/* Lowercase Omega */

/* Other Modern Greek Characters [8,52] */

	COLS7,			/* Uppercase ALPHA Tonos high prime */
	COLS7,			/* Lowercase Alpha Tonos - acute */
	COLS7+4,			/* Uppercase EPSILON Tonos - high prime */
	COLS7+4,			/* Lowercase Epslion Tonos - acute */
	COLS7+6,			/* Uppercase ETA Tonos - high prime */
	COLS7+6,			/* Lowercase Eta Tonos - acute */
	COLS7+8,			/* Uppercase IOTA Tonos - high prime */
	COLS7+8,			/* Lowercase iota Tonos - acute */
	COLS7+8,			/* Uppercase IOTA Diaeresis */
	COLS7+8,			/* Lowercase iota diaeresis */
	COLS7+14,		/* Uppercase OMICRON Tonos - high prime */
	COLS7+14,		/* Lowercase Omicron Tonos - acute */
	COLS7+19,		/* Uppercase UPSILON Tonos - high prime */
	COLS7+19,		/* Lowercase Upsilon Tonos - acute */
	COLS7+19,		/* Uppercase UPSILON Diaeresis */
	COLS7+19,		/* Lowercase Upsilon diaeresis */
	COLS7+23,		/* Uppercase OMEGA Tonos - high prime */
	COLS7+23,		/* Lowercase Omega Tonso - acute */

/* Variants [8,70] */

	COLS7+4,			/* epsilon (variant) */
	COLS7+7,			/* theta (variant) */
	COLS7+9,			/* kappa (variant) */
	COLS7+15,		/* pi (variant) */
	COLS7+16,		/* rho (variant) */
	COLS7+17,		/* sigma (variant) */
	COLS7+19,		/* upsilon (variant) */
	COLS7+20,		/* phi (variant) */
	COLS7+23,		/* omega (variant) */
	
/* Greek Diacritic marks [8,79] */	

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
	COLS0,		/* 8,108 end of diacritic marks */

/* Ancient Greek [8,109] */
	
	COLS7,		/* alpha grave */
	COLS7,		/* alpha circumflex */
	COLS7,		/* alpha w/iota */
	COLS7,		/* alpha acute w/iota */
	COLS7,		/* alpha grave w/iota */
	COLS7,		/* alpha circumflex w/Iota */
	COLS7,		/* alpha smooth */
	COLS7,		/* alpha smooth acute */
	COLS7,		/* alpha smooth grave */
	COLS7,		/* alpha smooth circumflex */
	COLS7,		/* alpha smooth w/Iota */
	COLS7,		/* alpha smooth acute w/Iota */
	COLS7,		/* alpha smooth grave w/Iota */
	COLS7,		/* alpha smooth circumflex w/Iota */
/* [8,123] */
	COLS7,		/* alpha rough */
	COLS7,		/* alpha rough acute */
	COLS7,		/* alpha rough grave */
	COLS7,		/* alpha rough circumflex */
	COLS7,		/* alpha rough w/Iota */
	COLS7,		/* alpha rough acute w/Iota */
	COLS7,		/* alpha rough grave w/Iota */
	COLS7,		/* alpha rough circumflex w/Iota */
/* [8,131] */
	COLS7+4,		/* epsilon grave */
	COLS7+4,		/* epsilon smooth */
	COLS7+4,		/* epsilon smooth acute */
	COLS7+4,		/* epsilon smooth grave */
	COLS7+4,		/* epsilon rough */
	COLS7+4,		/* epsilon rough acute */
	COLS7+4,		/* epsilon rough grave */
/* [8,138] */
	COLS7+6,		/* eta grave */
	COLS7+6,		/* eta circumflex */
	COLS7+6,		/* eta w/iota */
	COLS7+6,		/* eta acute w/iota */
	COLS7+6,		/* eta grave w/Iota */
	COLS7+6,		/* eta circumflex w/Iota */
	COLS7+6,		/* eta smooth */
	COLS7+6,		/* eta smooth acute */
	COLS7+6,		/* eta smooth grave */
	COLS7+6,		/* eta smooth circumflex */
	COLS7+6,		/* eta smooth w/Iota */
	COLS7+6,		/* eta smooth acute w/Iota */
	COLS7+6,		/* eta smooth grave w/Iota */
	COLS7+6,		/* eta smooth circumflex w/Iota */
	COLS7+6,		/* eta rough */
	COLS7+6,		/* eta rough acute */
	COLS7+6,		/* eta rough grave */
	COLS7+6,		/* eta rough circumflex */
	COLS7+6,		/* eta rough w/Iota */
	COLS7+6,		/* eta rough acute w/Iota */
	COLS7+6,		/* eta rough grave w/Iota */
	COLS7+6,		/* eta rough circumflex w/Iota */
/* [8,160] */
	COLS7+8,		/* iota grave */
	COLS7+8,		/* iota circumflex */
	COLS7+8,		/* iota acute diaeresis */
	COLS7+8,		/* iota grave diaeresis */
	COLS7+8,		/* iota smooth */
	COLS7+8,		/* iota smooth acute */
	COLS7+8,		/* iota smooth grave */
	COLS7+8,		/* iota smooth circumflex */
	COLS7+8,		/* iota rough */
	COLS7+8,		/* iota rough acute */
	COLS7+8,		/* iota rough grave */
	COLS7+8,		/* iota rough circumflex */
/* [8,172] */
	COLS7+14,	/* omicron grave */
	COLS7+14,	/* omicron smooth */
	COLS7+14,	/* omicron smooth acute */
	COLS7+14,	/* omicron smooth grave */
	COLS7+14,	/* omicron rough */
	COLS7+14,	/* omicron rough acute */
	COLS7+14,	/* omicron rough grave */
/* [8,179] */
	COLS7+16,	/* rho smooth */
	COLS7+16,	/* rho rough */
/* [8,181] */
	COLS7+19,	/* upsilon grave */
	COLS7+19,	/* upsilon circumflex */
	COLS7+19,	/* upsilon acute diaeresis */
	COLS7+19,	/* upsilon grave diaeresis */
	COLS7+19,	/* upsilon smooth */
	COLS7+19,	/* upsilon smooth acute */
	COLS7+19,	/* upsilon smooth grave */
	COLS7+19,	/* upsilon smooth circumflex */
	COLS7+19,	/* upsilon rough */
	COLS7+19,	/* upsilon rough acute */
	COLS7+19,	/* upsilon rough grave */
	COLS7+19,	/* upsilon rough circumflex */
/* [8,193] */
	COLS7+23,	/* omega grave */
	COLS7+23,	/* omega circumflex */
	COLS7+23,	/* omega w/Iota */
	COLS7+23,	/* omega acute w/Iota */
	COLS7+23,	/* omega grave w/Iota */
	COLS7+23,	/* omega circumflex w/Iota */
	COLS7+23,	/* omega smooth */
	COLS7+23,	/* omega smooth acute */
	COLS7+23,	/* omega smooth grave */
	COLS7+23,	/* omega smooth circumflex */
	COLS7+23,	/* omega smooth w/Iota */
	COLS7+23,	/* omega smooth acute w/Iota */
	COLS7+23,	/* omega smooth grave w/Iota */
	COLS7+23,	/* omega smooth circumflex w/Iota */
	COLS7+23,	/* omega rough */
	COLS7+23,	/* omega rough acute */
	COLS7+23,	/* omega rough grave */
	COLS7+23,	/* omega rough circumflex */
	COLS7+23,	/* omega rough w/Iota */
	COLS7+23,	/* omega rough acute w/Iota */
	COLS7+23,	/* omega rough grave w/Iota */
	COLS7+23,	/* omega rough circumflex w/Iota */
/* [8,215] */
	COLS7+24,	/* Uppercase Stigma--the number 6 */
	COLS7+24,	/* Uppercase Digamma--Obsolete letter used as 6 */
	COLS7+24,	/* Uppercase Koppa--Obsolete letter used as 90 */
	COLS7+24		/* Uppercase Sampi--Obsolete letter used as 900 */
};

#define CYRLTBLLEN	200

FLMBYTE fwp_cyrl60Tbl[CYRLTBLLEN+2] = {
	0,						/* starting offset */
	CYRLTBLLEN,		/* len of table */
/*00*/
	COLS10,			/* Russian uppercase A */
	COLS10,			/* Russian lowercase A */
	COLS10+1,		/* Russian uppercase BE */
	COLS10+1,		/* Russian lowercase BE */
	COLS10+2,		/* Russian uppercase VE */
	COLS10+2,		/* Russian lowercase VE */
	COLS10+3,		/* Russian uppercase GHE */
	COLS10+3,		/* Russian lowercase GHE */
	COLS10+5,		/* Russian uppercase DE */
	COLS10+5,		/* Russian lowercase DE */
/*10*/
	COLS10+8,		/* Russian uppercase E */
	COLS10+8,		/* Russian lowercase E */
	COLS10+9,		/* Russian lowercase YO */
	COLS10+9,		/* Russian lowercase YO */
	COLS10+11,		/* Russian uppercase ZHE */
	COLS10+11,		/* Russian lowercase ZHE */
	COLS10+12,		/* Russian uppercase ZE */
	COLS10+12,		/* Russian lowercase ZE */
	COLS10+14,		/* Russian uppercase I */
	COLS10+14,		/* Russian lowercase I */
/*20*/
	COLS10+17,		/* Russian uppercase SHORT I */
	COLS10+17,		/* Russian lowercase SHORT I */
	COLS10+19,		/* Russian uppercase KA */
	COLS10+19,		/* Russian lowercase KA */
	COLS10+20,		/* Russian uppercase EL */
	COLS10+20,		/* Russian lowercase EL */
	COLS10+22,		/* Russian uppercase EM */
	COLS10+22,		/* Russian lowercase EM */
	COLS10+23,		/* Russian uppercase EN */
	COLS10+23,		/* Russian lowercase EN */
/*30*/
	COLS10+25,		/* Russian uppercase O */
	COLS10+25,		/* Russian lowercase O */
	COLS10+26,		/* Russian uppercase PE */
	COLS10+26,		/* Russian lowercase PE */
	COLS10+27,		/* Russian uppercase ER */
	COLS10+27,		/* Russian lowercase ER */
	COLS10+28,		/* Russian uppercase ES */
	COLS10+28,		/* Russian lowercase ES */
	COLS10+29,		/* Russian uppercase TE */
	COLS10+29,		/* Russian lowercase TE */
/*40*/
	COLS10+32,		/* Russian uppercase U */
	COLS10+32,		/* Russian lowercase U */
	COLS10+34,		/* Russian uppercase EF */
	COLS10+34,		/* Russian lowercase EF */
	COLS10+35,		/* Russian uppercase HA */
	COLS10+35,		/* Russian lowercase HA */
	COLS10+36,		/* Russian uppercase TSE */
	COLS10+36,		/* Russian lowercase TSE */
	COLS10+37,		/* Russian uppercase CHE */
	COLS10+37,		/* Russian lowercase CHE */
/*50*/
	COLS10+39,		/* Russian uppercase SHA */
	COLS10+39,		/* Russian lowercase SHA */
	COLS10+40,		/* Russian uppercase SHCHA */
	COLS10+40,		/* Russian lowercase SHCHA */
	COLS10+41,		/* Russian lowercase ER (also hard sign) */
	COLS10+41,		/* Russian lowercase ER (also hard sign) */
	COLS10+42,		/* Russian lowercase ERY */
	COLS10+42,		/* Russian lowercase ERY */
	COLS10+43,		/* Russian lowercase SOFT SIGN */
	COLS10+43,		/* Russian lowercase SOFT SIGN */
/*60*/
	COLS10+45,		/* Russian uppercase REVERSE E */
	COLS10+45,		/* Russian lowercase REVERSE E */
	COLS10+46,		/* Russian uppercase YU */
	COLS10+46,		/* Russian lowercase yu */
	COLS10+47,		/* Russian uppercase YA */
	COLS10+47,		/* Russian lowercase ya */
/*66*/
	COLS0,			/* Russian uppercase EH */
	COLS0,			/* Russian lowercase eh */
	COLS10+7,		/* Macedonian uppercase SOFT DJ */
	COLS10+7,		/* Macedonian lowercase soft dj */
/*70*/
	COLS10+4,		/* Ukrainian uppercase HARD G */
	COLS10+4,		/* Ukrainian lowercase hard g */
	COLS0,			/* GE bar */
	COLS0,			/* ge bar */
	COLS10+6,		/* Serbian uppercase SOFT DJ */
	COLS10+6,		/* Serbian lowercase SOFT DJ */
	COLS0,			/* IE (variant) */
	COLS0,			/* ie (variant) */
	COLS10+10,		/* Ukrainian uppercase YE */
	COLS10+10,		/* Ukrainian lowercase YE */
/*80*/
	COLS0,			/* ZHE with right descender */
	COLS0,			/* zhe with right descender */
	COLS10+13,		/* Macedonian uppercase ZELO */
	COLS10+13,		/* Macedonian lowercase ZELO */
	COLS0,			/* Old Slovanic uppercase Z */
	COLS0,			/* Old Slovanic uppercase z */
	COLS0,			/* II with macron */
	COLS0,			/* ii with mscron */
	COLS10+15,		/* Ukrainian uppercase I */
	COLS10+15,		/* Ukrainian lowercase I */
/*90*/
	COLS10+16,		/* Ukrainian uppercase I with Two Dots */
	COLS10+16,		/* Ukrainian lowercase I with Two Dots */
	COLS0,			/* Old Slovanic uppercase I ligature */
	COLS0,			/* Old Slovanic lowercase I ligature */
	COLS10+18,		/* Serbian--Macedonian uppercase JE */
	COLS10+18,		/* Serbian--Macedonian lowercase JE */
	COLS10+31,		/* Macedonian uppercase SOFT K */
	COLS10+31,		/* Macedonian lowercase SOFT K */
	COLS0,			/* KA with right descender */
	COLS0,			/* ka with right descender */
/*100*/
	COLS0,			/* KA ogonek */
	COLS0,			/* ka ogonek */
	COLS0,			/* KA vertical bar */
	COLS0,			/* ka vertical bar */
	COLS10+21,		/* Serbian--Macedonian uppercase SOFT L */
	COLS10+21,		/* Serbian--Macedonian lowercase SOFT L */
	COLS0,			/* EN with right descender */
	COLS0,			/* en with right descender */
	COLS10+24,		/* Serbian--Macedonian uppercase SOFT N */
	COLS10+24,		/* Serbian--Macedonian lowercase SOFT N */
/*110*/
	COLS0,			/* ROUND OMEGA */
	COLS0,			/* round omega */
	COLS0,			/* OMEGA */
	COLS0,			/* omega */
	COLS10+30,		/* Serbian uppercase SOFT T */
	COLS10+30,		/* Serbian lowercase SOFT T */
	COLS10+33,		/* Byelorussian uppercase SHORT U */
	COLS10+33,		/* Byelorussian lowercase SHORT U */
	COLS0,			/* U with macron */
	COLS0,			/* u with macron */
/*120*/
	COLS0,			/* STRAIGHT U */
	COLS0,			/* straight u */
	COLS0,			/* STRAIGHT U bar */
	COLS0,			/* straight u bar */
	COLS0,			/* OU ligature */
	COLS0,			/* ou ligature */
	COLS0,			/* KHA with right descender */
	COLS0,			/* kha with right descender */
	COLS0,			/* KHA ogonek */
	COLS0,			/* kha ogonek */
/*130*/	
	COLS0,			/* H */
	COLS0,			/* h */
	COLS0,			/* OMEGA titlo */
	COLS0,			/* omega titlo */
	COLS10+38,		/* Serbian uppercase HARD DJ */
	COLS10+38,		/* Serbian lowercase HARD DJ */
	COLS0,			/* CHE with right descender */
	COLS0,			/* che with right descender */
	COLS0,			/* CHE vertical bar */
	COLS0,			/* che vertical bar */
/*140*/
	COLS0,			/* Old Slavonic SHCHA (variant) */
	COLS0,			/* old SLAVONIC shcha (variant) */
	COLS10+44,		/* Old Russian uppercase YAT */
	COLS10+44,		/* Old Russian lowercase YAT */
/**----------------------------------------------
***  END OF UNIQUE COLLATED BYTES 
***  CHARACTERS BELOW MUST HAVE HAVE THEIR OWN
***  SUB-COLLATION VALUE TO COMPARE CORRECTLY.
***---------------------------------------------*/
	COLS0,			/* Old Bulgarian uppercase YUS */
	COLS0,			/* Old Bulgarian lowercase YUS */
	COLS0,			/* Old Slovanic uppercase YUS MALYI */
	COLS0,			/* Old Slovanic uppercase YUS MALYI */
	COLS0,			/* KSI */
	COLS0,			/* ksi */
/*150*/	
	COLS0,			/* PSI */
	COLS0,			/* psi */
	COLS0,			/* Old Russian uppercase FITA */
	COLS0,			/* Old Russian lowercase FITA */
	COLS0,			/* Old Russian uppercase IZHITSA */
	COLS0,			/* Old Russian lowercase IZHITSA */
	COLS0,			/* Russian uppercase A acute */
	COLS0,			/* Russian lowercase A acute */
	COLS10+8,		/* Russian uppercase E acute */
	COLS10+8,		/* Russian lowercase E acute */

/*160-below all characters are russian to 199*/	
	COLS0,			/* E acute */
	COLS0,			/* e acute */
	COLS10+14,		/* II acute */
	COLS10+14,		/* ii acute */
	COLS0,			/* I acute */
	COLS0,			/* i acute */
	COLS0,			/* YI acute */
	COLS0,			/* yi acute */
	COLS10+25,		/* O acute */
	COLS10+25,		/* o acute */
/*170*/
	COLS10+32,		/* U acute */
	COLS10+32,		/* u acute */
	COLS10+42,		/* YERI acute */
	COLS10+42,		/* YERI acute */
	COLS10+45,		/* REVERSED E acute */
	COLS10+45,		/* reversed e acute */
	COLS10+46,		/* YU acute */
	COLS10+46,		/* yu acute */
	COLS10+47,		/* YA acute */
	COLS10+47,		/* ya acute */
/*180*/
	COLS10,			/* A grave */
	COLS10,			/* a grave */
	COLS10+8,		/* E grave */
	COLS10+8,		/* e grave */
	COLS10+9,		/* YO grave */
	COLS10+9,		/* yo grave */
	COLS10+14,		/* I grave */
	COLS10+14,		/* i grave */
	COLS10+25,		/* O grave */
	COLS10+25,		/* o grave */
/*190*/
	COLS10+32,		/* U grave */
	COLS10+32,		/* u grave */
	COLS10+42,		/* YERI grave */
	COLS10+42,		/* yeri grave */
	COLS10+45,		/* REVERSED E grave */
	COLS10+45,		/* reversed e grave */
	COLS10+46,		/* IU (YU) grave */
	COLS10+46,		/* iu (yu) grave */
	COLS10+47,		/* ia (YA) grave */
	COLS10+47,		/* ia (ya) grave ******* [10,199] */
/***-----------------------------------------
***  What follows is for documentation only
***-----------------------------------------*/
/*200*/				/* Acute */
						/* Grave */
						/* Diaeresis */
						/* Breve */
						/* Right descender */
						/* ogonek */
						/* Macron */
						/* 207 - GEORGIAN CHARACTERS BELOW ***/
						/* Righ Quote Marks */
						/* Left Quote Marks */
/*210*/				/* An */
						/* Ban */
						/* Gan */
						/* Don */
						/* En */
						/* Vin */
						/* Zen */
						/* He */
						/* Tan */
						/* In */
/*220*/				/* Kan */
						/* Las */
						/* Man */
						/* Nar */
						/* Hie */
						/* On */
						/* Par */
						/* Zhar */
						/* Rae */
						/* San */
/*230*/				/* Tar */
						/* Un */
						/* We */
						/* Phar */
						/* Khar */
						/* Ghan */
						/* Qar */
						/* Shin */
						/* Chin */
						/* Can */
/*240*/				/* Jil */
						/* Cil */
						/* Char */
						/* Xan */
						/* Har */
						/* Jhan */
						/* Hae */
						/* Hoe */
						/* Fi */
						/* Un w/ Circumflex - 249 END */
};


/* The Hebrew characters are collated over the Russian characters	*/
/* Therefore sorting both Hebrew and Russian is impossible to do. */

#define	HEBTBL1LEN	27

/* #define HEBTBLLEN 119				27 in the first section */
											/* 57 in the accents section (wasted!)*/
											/* 35 in the ancient (dagesh) section */

FLMBYTE fwp_heb60TblA[HEBTBL1LEN+2] = {
	0,					/* starting offset */
	HEBTBL1LEN,		/* len of table */
	COLS10h+0,		/* Alef */
	COLS10h+1,		/* Bet */
	COLS10h+2,		/* Gimel */
	COLS10h+3,		/* Dalet */
	COLS10h+4,		/* He */
	COLS10h+5,		/* Vav */
	COLS10h+6,		/* Zayin */
	COLS10h+7,		/* Het */
	COLS10h+8,		/* Tet */
	COLS10h+9,		/* Yod */
	COLS10h+10,		/* Kaf (final) [9,10] */
	COLS10h+11,		/* Kaf */
	COLS10h+12,		/* Lamed */
	COLS10h+13,		/* Mem (final) */
	COLS10h+14,		/* Mem */
	COLS10h+15,		/* Nun (final) */
	COLS10h+16,		/* Nun */
	COLS10h+17,		/* Samekh */
	COLS10h+18,		/* Ayin */
	COLS10h+19,		/* Pe (final) */
	COLS10h+20,		/* Pe [9,20] */
	COLS10h+21,		/* Tsadi (final) */
	COLS10h+22,		/* Tsadi  */
	COLS10h+23,		/* Qof */
	COLS10h+24,		/* Resh */
	COLS10h+25,		/* Shin */
	COLS10h+26		/* Tav [9,26] */
};

	/**------------------------------------------------------
	***  This is the ANCIENT HEBREW SCRIPT piece.
	***  The actual value will be stored in the subcollation.
	***  This way we don't play diacritic/subcollation games.
	***		
	***-----------------------------------------------------*/
	
#define	HEBTBL2LEN	35

FLMBYTE fwp_heb60TblB[HEBTBL2LEN+2] = {
	84,				
	HEBTBL2LEN,		

/*[9,84]*/
	COLS10h+0,		/* Alef Dagesh [9,84] */
	COLS10h+1,		/* Bet Dagesh */
	COLS10h+1,		/* Vez - looks like a bet */
	COLS10h+2,		/* Gimel Dagesh */
	COLS10h+3,		/* Dalet Dagesh */
	COLS10h+4,		/* He Dagesh */
	COLS10h+5,		/* Vav Dagesh [9,90] */
	COLS10h+5,		/* Vav Holem */
	COLS10h+6,		/* Zayin Dagesh */
	COLS10h+7,		/* Het Dagesh */
	COLS10h+8,		/* Tet Dagesh */
	COLS10h+9,		/* Yod Dagesh  */
	COLS10h+9,		/* Yod Hiriq [9,96] - not on my list */
	
	COLS10h+11,		/* Kaf Dagesh */
	COLS10h+10,		/* Kaf Dagesh (final) */
	COLS10h+10,		/* Kaf Sheva (final)  */
	COLS10h+10,		/* Kaf Tsere (final) [9,100] */
	COLS10h+10,		/* Kaf Segol (final)  */
	COLS10h+10,		/* Kaf Patah (final)  */
	COLS10h+10,		/* Kaf Qamats (final) */
	COLS10h+10,		/* Kaf Dagesh Qamats (final) */
	COLS10h+12,		/* Lamed Dagesh */
	COLS10h+14,		/* Mem Dagesh */
	COLS10h+16,		/* Nun Dagesh */
	COLS10h+15,		/* Nun Qamats (final) */
	COLS10h+17,		/* Samekh Dagesh  */
	COLS10h+20,		/* Pe Dagesh [9,110] */
	COLS10h+20,		/* Fe - just guessing this is like Pe - was +21 */	
	COLS10h+22,		/* Tsadi Dagesh */
	COLS10h+23,		/* Qof Dagesh */
	COLS10h+25,		/* Sin (with sin dot) */
	COLS10h+25,		/* Sin Dagesh (with sin dot) */
	COLS10h+25,		/* Shin */
	COLS10h+25,		/* Shin Dagesh */
	COLS10h+26		/* Tav Dagesh [9,118] */
};

/**  The Arabic characters are collated OVER the Russian characters
***  Therefore sorting both Arabic and Russian in the same database
***  is not supported.
***
***  Arabic starts with a bunch of accents/diacritic marks that are
***  Actually placed OVER a preceeding character.  These accents are
***  ignored while sorting the first pass - when collation == COLS0.
***
***  There are 4 possible states for all/most arabic characters:
***		зы - occurs as the only character in a word 
***		ды - appears at the first of the word
***		дд - appears at the middle of a word
***		зд - appears at the end of the word
***
***  Usually only the simple version of the letter is stored.
***  Therefore we should not have to worry about sub-collation
***  of these characters.
***
***  The arabic characters with diacritics differ however.  The alef has
***  sub-collation values to sort correctly.  There is not any more room
***  to add more collation values.  Some chars in CS14 are combined when
***  urdu, pashto and sindhi characters overlap.
***
**/

#define AR1TBLLEN   	158					/* (195 - 38 + 1) */

FLMBYTE fwp_ar160Tbl[AR1TBLLEN+2] = {
	38,				/* starting offset */
	AR1TBLLEN,		/* len of table */
/*[13,38]*/
	COLLS+2,		/* , comma */
	COLLS+3,		/* : colon */
/*[13,40]*/
	COLLS+7,		/* ? question mark */
	COLS4+2,		/* * asterick */
	COLS6,		/* % percent */
	COLS9+41,	/* >> alphabetic - end of list) */
	COLS9+40,	/* << alphabetic - end of list) */
	COLS2,		/* (  */
	COLS2+1,		/* )  */
/*[13,47]*/
	COLS8+1,		/* зы One  */
	COLS8+2,		/* зы Two  */
	COLS8+3,		/* зы Three*/
/*[13,50]*/
	COLS8+4,		/* зы Four */
	COLS8+5,		/* зы Five */
	COLS8+6,		/* зы Six  */
	COLS8+7,		/* зы Seven */
	COLS8+8,		/* зы Eight */
	COLS8+9,		/* зы Nine */
	COLS8+0,		/* зы Zero */
	COLS8+2,		/* зы Two (Handwritten) */
	
	COLS10a+1,		/* зы alif */
	COLS10a+1,		/* зд alif */
/*[13,60]*/
	COLS10a+2,		/* зы ba   */
	COLS10a+2,		/* ды ba   */
	COLS10a+2,		/* дд ba   */
	COLS10a+2,		/* зд ba   */
	COLS10a+6,		/* зы ta   */
	COLS10a+6,		/* ды ta   */
	COLS10a+6,		/* дд ta   */
	COLS10a+6,		/* зд ta   */
	COLS10a+8,		/* зы tha  */
	COLS10a+8,		/* ды tha  */
/*[13,70]*/
	COLS10a+8,		/* дд tha  */
	COLS10a+8,		/* зд tha  */
	COLS10a+12,		/* зы jiim */
	COLS10a+12,		/* ды jiim */
	COLS10a+12,		/* дд jiim */
	COLS10a+12,		/* зд jiim */
	COLS10a+16,		/* зы Ha   */
	COLS10a+16,		/* ды Ha   */
	COLS10a+16,		/* дд Ha   */
	COLS10a+16,		/* зд Ha   */
/*[13,80]*/
	COLS10a+17,		/* зы kha  */
	COLS10a+17,		/* ды kha  */
	COLS10a+17,		/* дд kha  */
	COLS10a+17,		/* зд kha  */
	COLS10a+20,		/* зы dal  */
	COLS10a+20,		/* зд dal  */
	COLS10a+22,		/* зы dhal */
	COLS10a+22,		/* зд dhal */
	COLS10a+27,		/* зы ra   */
	COLS10a+27,		/* зд ra   */
/*[13,90]*/
	COLS10a+29,		/* зы ziin */
	COLS10a+29,		/* зд ziin */
	COLS10a+31,		/* зы siin */
	COLS10a+31,		/* ды siin */
	COLS10a+31,		/* дд siin */
	COLS10a+31,		/* зд siin */
	COLS10a+32,		/* зы shiin*/
	COLS10a+32,		/* ды shiin*/
	COLS10a+32,		/* дд shiin*/
	COLS10a+32,		/* зд shiin*/
/*[13,100]*/
	COLS10a+34,		/* зы Sad  */
	COLS10a+34,		/* ды Sad  */
	COLS10a+34,		/* дд Sad  */
	COLS10a+34,		/* зд Sad  */
	COLS10a+35,		/* зы Dad  */
	COLS10a+35,		/* ды Dad  */
	COLS10a+35,		/* дд Dad  */
	COLS10a+35,		/* зд Dad  */
	COLS10a+36,		/* зы Ta   */
	COLS10a+36,		/* ды Ta   */
/*[13,110]*/
	COLS10a+36,		/* дд Ta   */
	COLS10a+36,		/* зд Ta   */
	COLS10a+37,		/* зы Za   */
	COLS10a+37,		/* ды Za   */
	COLS10a+37,		/* дд Za   */
	COLS10a+37,		/* зд Za   */
	COLS10a+38,		/* зы 'ain */
	COLS10a+38,		/* ды 'ain */
	COLS10a+38,		/* дд 'ain */
	COLS10a+38,		/* зд 'ain */
/*[13,120]*/
	COLS10a+39,		/* зы ghain*/
	COLS10a+39,		/* ды ghain*/
	COLS10a+39,		/* дд ghain*/
	COLS10a+39,		/* зд ghain*/
	COLS10a+40,		/* зы fa   */
	COLS10a+40,		/* ды fa   */
	COLS10a+40,		/* дд fa   */
	COLS10a+40,		/* зд fa   */
	COLS10a+42,		/* зы Qaf  */
	COLS10a+42,		/* ды Qaf  */
/*[13,130]*/
	COLS10a+42,		/* дд Qaf  */
	COLS10a+42,		/* зд Qaf  */
	COLS10a+43,		/* зы kaf  */
	COLS10a+43,		/* ды kaf  */
	COLS10a+43,		/* дд kaf  */
	COLS10a+43,		/* зд kaf  */
	COLS10a+46,		/* зы lam  */
	COLS10a+46,		/* ды lam  */
	COLS10a+46,		/* дд lam  */
	COLS10a+46,		/* зд lam  */
/*[13,140]*/
	COLS10a+47,		/* зы miim */
	COLS10a+47,		/* ды miim */
	COLS10a+47,		/* дд miim */
	COLS10a+47,		/* зд miim */
	COLS10a+48,		/* зы nuun */
	COLS10a+48,		/* ды nuun */
	COLS10a+48,		/* дд nuun */
	COLS10a+48,		/* зд nuun */
	COLS10a+49,		/* зы ha   */
	COLS10a+49,		/* ды ha   */
/*[13,150]*/
	COLS10a+49,		/* дд ha   */
	COLS10a+49,		/* зд ha   */
						/* ha is also 51 for non-arabic */
	COLS10a+6, 		/* зы ta marbuuTah */
	COLS10a+6, 		/* зд ta marbuuTah */
	COLS10a+50,		/* зы waw  */
	COLS10a+50,		/* зд waw  */
	COLS10a+53,		/* зы ya   */
	COLS10a+53,		/* ды ya   */
	COLS10a+53,		/* дд ya   */
	COLS10a+53,		/* зд ya   */
/*[13,160]*/
	COLS10a+52,		/* зы alif maqSuurah */
	COLS10a+52,		/* ды ya   maqSuurah? */
	COLS10a+52,		/* дд ya   maqSuurah? */
	COLS10a+52,		/* зд alif maqSuurah */

	COLS10a+0,		/* зы hamzah accent - never appears alone */
/*[13,165]*/
/**----------------------------------------------------
***  Store the sub-collation as the actual 
***  character value from this point on
***---------------------------------------------------*/
	COLS10a+1,		/* зы alif hamzah    */
	COLS10a+1,		/* зд alif hamzah    */
	COLS10a+1,		/* зы hamzah-under-alif */
	COLS10a+1,		/* зд hamzah-under-alif */
	COLS10a+1,		/* зы waw hamzah */
/*[13,170]*/
	COLS10a+1,		/* зд waw hamzah */
	COLS10a+1,		/* зы ya hamzah  */
	COLS10a+1,		/* ды ya hamzah  */
	COLS10a+1,		/* дд ya hamzah  */
	COLS10a+1,		/* зд ya hamzah  */
	COLS10a+1,		/* зы alif fatHataan */
	COLS10a+1,		/* зд alif fatHataan */
	COLS10a+1,		/* зы alif maddah */
	COLS10a+1,		/* зд alif maddah */
	COLS10a+1,		/* зы alif waSlah */
/*[13,180]	*/
	COLS10a+1,		/* зд alif waSlah (final) */
/**---------------------------------------------------
***  LIGATURES
***    Should NEVER be stored so will not worry
***    about breaking up into pieces for collation.
***  NOTE:
***    Let's store the "Lam" collation value (+42)
***    below and in the sub-collation store the
***    actual character.  This will sort real close.
***    The best implementation is to 
***    break up ligatures into its base pieces.
***--------------------------------------------------*/
	COLS10a+46,		/* зы lamalif */
	COLS10a+46,		/* зд lamalif */
	COLS10a+46,		/* зы lamalif hamzah */
	COLS10a+46,		/* зд lamalif hamzah */
	COLS10a+46,		/* зы hamzah-under-lamalif */
	COLS10a+46,		/* зд hamzah-under-lamalif */
	COLS10a+46,		/* зы lamalif fatHataan */
	COLS10a+46,		/* зд lamalif fatHataan */
	COLS10a+46,		/* зы lamalif maddah */
/*[13,190]*/
	COLS10a+46,		/* зд lamalif maddah */
	COLS10a+46,		/* зы lamalif waSlah */
	COLS10a+46,		/* зд lamalif waSlah */
	COLS10a+46,		/* зы Allah - khaDalAlif */
	COLS0_ARABIC,	/* дд taTwiil     - character extension - throw out */
	COLS0_ARABIC	/* дд taTwiil 1/6 - character extension - throw out */
};

/**------------------------------------------
***  Alef needs a subcollation table.
***  If colval==COLS10a+1 & char>=165
***  index through this table.  Otherwise
***  the alef value is [13,58] and subcol
***  value should be 7.  Alef maddah is default (0)
***
***  Handcheck if colval==COLS10a+6
***  Should sort:
***      [13,152]..[13,153] - taa marbuuTah - nosubcoll
***      [13,64] ..[13,67]  - taa - subcoll of 1
***-----------------------------------------*/

FLMBYTE fwp_alefSubColTbl[] = {
/*[13,165]*/
	1,		/* зы alif hamzah    */
	1,		/* зд alif hamzah    */
	3,		/* зы hamzah-under-alif */
	3,		/* зд hamzah-under-alif */
	2,		/* зы waw hamzah */
/*[13,170]*/
	2,		/* зд waw hamzah */
	4,		/* зы ya hamzah  */
	4,		/* ды ya hamzah  */
	4,		/* дд ya hamzah  */
	4,		/* зд ya hamzah  */
	5,		/* зы alif fatHataan */
	5,		/* зд alif fatHataan */
	0,		/* зы alif maddah */
	0,		/* зд alif maddah */
	6,		/* зы alif waSlah */
/*[13,180]*/
	6		/* зд alif waSlah (final) */
};

#define AR2TBLLEN			179					/* 219 - 41 + 1 */

FLMBYTE fwp_ar260Tbl[AR2TBLLEN+2] = {
	41,				/* starting offset */
	AR2TBLLEN,		/* len of table */
/*[14,41]*/
	COLS8+4,			/* Farsi and Urdu Four */
	COLS8+4,			/* Urdu Four */
	COLS8+5,			/* Farsi and Urdu Five */
	COLS8+6,			/* Farsi Six */
	COLS8+6,			/* Farsi and Urdu Six */
	COLS8+7,			/* Urdu Seven */
	COLS8+8,			/* Urdu Eight */
	
	COLS10a+3,		/* Sindhi bb - baa /w 2 dots below (67b) */
	COLS10a+3,
	COLS10a+3,
	COLS10a+3,
	COLS10a+4,		/* Sindhi bh - baa /w 4 dots below (680) */
	COLS10a+4,
	COLS10a+4,
	COLS10a+4,
/*[14,56]*/
	COLS10a+5,		/* Malay, Kurdish, Pashto, Farsi, Sindhi, and Urdu p */
	COLS10a+5,		/* =peh - taa /w 3 dots below (67e) */
	COLS10a+5,
	COLS10a+5,
	COLS10a+7,		/* Urdu T - taa /w small tah */
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,		/* Pashto T - taa /w ring (forced to combine) */
	COLS10a+7,
	COLS10a+7,
	COLS10a+7,
	COLS10a+9,		/* Sindhi th - taa /w 4 dots above (67f) */
	COLS10a+9,
/*[14,70]*/
	COLS10a+9,
	COLS10a+9,
	COLS10a+10,		/* Sindhi Tr - taa /w 3 dots above (67d) */
	COLS10a+10,
	COLS10a+10,
	COLS10a+10,
	COLS10a+11,		/* Sindhi Th - taa /w 2 dots above (67a) */
	COLS10a+11,
	COLS10a+11,
	COLS10a+11,
	COLS10a+13,		/* Sindhi jj - haa /w 2 middle dots verticle (684) */
	COLS10a+13,		
	COLS10a+13,		
	COLS10a+13,		
	COLS10a+14,		/* Sindhi ny - haa /w 2 middle dots (683) */
	COLS10a+14,		
	COLS10a+14,		
	COLS10a+14,		
/*[14,88]*/
	COLS10a+15,		/* Malay, Kurdish, Pashto, Farsi, Sindhi, and Urdu ch */
	COLS10a+15,		/* =tcheh (686) */
	COLS10a+15,		
	COLS10a+15,		
	COLS10a+15,		/* Sindhi chh - haa /w middle 4 dots (687) */
	COLS10a+15,		/* forced to combine */
	COLS10a+15,		
	COLS10a+15,		
	COLS10a+18,		/* Pashto ts - haa /w 3 dots above (685) */
	COLS10a+18,		
	COLS10a+18,		
	COLS10a+18,		
	COLS10a+19,		/* Pashto dz - hamzah on haa (681) */
	COLS10a+19,		
	COLS10a+19,		
	COLS10a+19,		
/*[14,104]*/
	COLS10a+21,		/* Urdu D - dal /w small tah (688) */
	COLS10a+21,
	COLS10a+21,		/* Pashto D - dal /w ring (689) forced to combine */
	COLS10a+21,		
	COLS10a+23,		/* Sindhi dh - dal /w 2 dots above (68c) */
	COLS10a+23,		
	COLS10a+24,		/* Sindhi D - dal /w 3 dots above (68e) */
	COLS10a+24,		
	COLS10a+25,		/* Sindhi Dr - dal /w dot below (68a) */
	COLS10a+25,		
	COLS10a+26,		/* Sindhi Dh - dal /w 2 dots below (68d) */
	COLS10a+26,		
	COLS10a+28,		/* Pashto r - ra /w ring (693) */
	COLS10a+28,		
/*[14,118]*/
	COLS10a+28,		/* Urdu R - ra /w small tah (691) forced to combine */
	COLS10a+28,
	COLS10a+28,		/* Sindhi r - ra /w 4 dots above (699) forced to combine */
	COLS10a+28,		
	COLS10a+27,		/* Kurdish rolled r - ra /w 'v' below (695) */
	COLS10a+27,		
	COLS10a+27,		
	COLS10a+27,		
/*[14,126]*/
	COLS10a+30,		/* Kurdish, Pashto, Farsi, Sindhi, and Urdu Z */
	COLS10a+30,		/* = jeh - ra /w 3 dots above (698) */
	COLS10a+30,		/* Pashto zz - ra /w dot below & dot above (696) */
	COLS10a+30,		/* forced to combine */
	COLS10a+30,		/* Pashto g - not in unicode! - forced to combine */
	COLS10a+30,		
	COLS10a+33,		/* Pashto x - seen dot below & above (69a) */
	COLS10a+33,		
	COLS10a+33,		
	COLS10a+33,		
	COLS10a+39,		/* Malay ng - old maly ain /w 3 dots above (6a0) */
	COLS10a+39,		/* forced to combine */
	COLS10a+39,		
	COLS10a+39,		
/*[14,140]*/
	COLS10a+41,		/* Malay p, Kurdish v - Farsi ? - fa /w 3 dots above */
	COLS10a+41,		/* = veh - means foreign words (6a4) */
	COLS10a+41,		
	COLS10a+41,		
	COLS10a+41,		/* Sindhi ph - fa /w 4 dots above (6a6) forced to combine*/
	COLS10a+41,		
	COLS10a+41,		
	COLS10a+41,		
/*[14,148]*/
	COLS10a+43,		/* Misc k - open caf (6a9) */
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,		/* misc k - no unicode - forced to combine */
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,
	COLS10a+43,		/* Sindhi k - swash caf (various) (6aa) -forced to combine*/
	COLS10a+43,		
	COLS10a+43,		
	COLS10a+43,		
/*[14,160]*/
	COLS10a+44,		/* Persian/Urdu g - gaf (6af) */
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		/* Persian/Urdu g - no unicode */
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,
	COLS10a+44,		/* malay g - gaf /w ring (6b0) */
	COLS10a+44,		
	COLS10a+44,		
	COLS10a+44,		
	COLS10a+44,		/* Sindhi ng  - gaf /w 2 dots above (6ba) */
	COLS10a+44,		/* forced to combine ng only */
	COLS10a+44,		
	COLS10a+44,		
	COLS10a+45,		/* Sindhi gg - gaf /w 2 dots vertical below (6b3) */
	COLS10a+45,		
	COLS10a+45,		
	COLS10a+45,		
/*[14,180]*/
	COLS10a+46,		/* Kurdish velar l - lam /w small v (6b5) */
	COLS10a+46,		
	COLS10a+46,		
	COLS10a+46,		
	COLS10a+46,		/* Kurdish Lamalif with diacritic - no unicode */
	COLS10a+46,
/*[14,186]*/
	COLS10a+48,		/* Urdu n - dotless noon (6ba) */
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,
	COLS10a+48,		/* Pashto N - noon /w ring (6bc) - forced to combine */
	COLS10a+48,		
	COLS10a+48,		
	COLS10a+48,		
	COLS10a+48,		/* Sindhi N - dotless noon/w small tah (6bb) */
	COLS10a+48,		/* forced to combine */
	COLS10a+48,		
	COLS10a+48,		
	COLS10a+50,		/* Kurdish o - waw /w small v (6c6) */
	COLS10a+50,		
/*[14,200]*/
	COLS10a+50,		/* Kurdish o - waw /w bar above (6c5) */
	COLS10a+50,		
	COLS10a+50,		/* Kurdish o - waw /w 2 dots above (6ca) */
	COLS10a+50,		
/*[14,204]*/
	COLS10a+51,		/* Urdu h - no unicode */
	COLS10a+51,
	COLS10a+51,
	COLS10a+51,
	COLS10a+52,		/* Kurdish ┬ - ya /w small v (6ce) */
	COLS10a+52,		
	COLS10a+52,		
	COLS10a+52,		
/*[14,212]*/
	COLS10a+54,		/* Urdu y - ya barree (6d2) */
	COLS10a+54,		
	COLS10a+54,		/* Malay ny - ya /w 3 dots below (6d1) forced to combine */
	COLS10a+54,		
	COLS10a+54,		
	COLS10a+54,		
/*[14,218]*/
	COLS10a+51,		/* Farsi hamzah - hamzah on ha (6c0) forced to combine */
	COLS10a+51
};


/* If the bit position is set then save the character in the sub-col area */
/* The bit values are determined by looking at the FLAIM COLTBL1 to see */
/* which characters are combined with other arabic characters. */

FLMBYTE fwp_ar2BitTbl[] = {	
	/* Start at character 64 */
	/* The only 'clean' areas uncollate to the correct place, they are...*/
						/* 48..63 */
						/* 68..91 */
						/* 96..117 */
						/* 126..127 */
						/* 140..143 */
						/* 160..163 */
						/* 176..179 */
						/* 212..213 */
						
	0xF0,				/* 64..71 */
	0x00,				/* 72..79 */
	0x00,				/* 80..87 */
	0x0F,				/* 88..95 - 92..95 */
	0x00,				/* 96..103 */
	0x00,				/* 104..111 */
	0x03,				/* 112..119 */
	0xFC,				/* 120..127 */
	0xFF,				/* 128..135 */
	0xF0,				/* 136..143 - 136..139 */
	0xFF,				/* 144..151 - 144..147, 148..159 */
	0xFF,				/* 152..159 */
	0x0F,				/* 160..167 - 164..175 */
	0xFF,				/* 168..175 */
	0x0F,				/* 176..183 - 180..185 */
	0xFF,				/* 184..191 - 186..197 */
	0xFF,				/* 192..199 - 198..203 */
	0xFF,				/* 200..207 - 204..207 */
	0xF3,				/* 208..215 - 208..211 , 214..217 */
	0xF0				/* 216..219 - 218..219 */
};

/**------------------------------------------------------------------
***  Collating table  
***    This table describes and gives addresses for collating 5.0
***    character sets.  Each line corresponds with a character set.
***
***	The second table is for sorting the hebrew/arabic languages.
***	These values overlap the end of ASC/european and cyrillic tables.
***-----------------------------------------------------------------*/

TBL_B_TO_BP fwp_col60Tbl[] = {
	{CHSASCI, fwp_asc60Tbl},	/* ascii - " " - "~" */
	{CHSMUL1, fwp_mn60Tbl},		/* multinational */
	{CHSSYM1, fwp_sym60Tbl},	/* symbols */
	{CHSGREK, fwp_grk60Tbl},	/* greek */
	{CHSCYR,  fwp_cyrl60Tbl},	/* Cyrillic - Russian */
	{0xFF, 	 0}					/* table terminator */
};

TBL_B_TO_BP fwp_HebArabicCol60Tbl[] = {
	{CHSASCI, fwp_asc60Tbl},		/* ascii - " " - "~" */
	{CHSMUL1, fwp_mn60Tbl},		/* multinational */
	{CHSSYM1, fwp_sym60Tbl},		/* symbols */
	{CHSGREK, fwp_grk60Tbl},		/* greek */
	{CHSHEB,	fwp_heb60TblA},		/* Hebrew */
	{CHSHEB,	fwp_heb60TblB},		/* Hebrew */
	{CHSARB1,	fwp_ar160Tbl},	/* Arabic Set 1 */
	{CHSARB2,	fwp_ar260Tbl},	/* Arabic Set 2 */
	{0xff, 		0}				/* table terminator */
};


/********************************************************************
	The diacritical to collated table translates the first 26 characters of
	WP character set #1 into a 5 bit value for "correct" sorting
	sequence for that diacritical (DCV) - diacritic collated value.
	
	The attempt here is to convert the collated character value
	along with the DCV to form the original WP character.

	The diacriticals are in an order to fit the most languages.
	Czech, Swedish, and Finnish will have to manual reposition the
	ring above (assign it a value greater then the umlaut)

	This table is index by the diacritical value.
********************************************************************/

FLMBYTE	fwp_dia60Tbl[] =
{
	2,			/* grave		offset = 0	*/
	16,		/*	centerd	offset = 1	*/
	7,			/*	tilde		offset = 2	*/
	4,			/*	circum	offset = 3	*/
	12,		/*	crossb	offset = 4	*/
	10,		/*	slash		offset = 5	*/
	1,			/*	acute		offset = 6	*/
	6,			/*	umlaut	offset = 7	*/
				/* In SU, SV and CZ will = 9 */
	17,		/*	macron	offset = 8	*/
	18,		/*	aposab	offset = 9	*/
	19,		/*	aposbes	offset = 10	*/
	20,		/*	aposba	offset = 11	*/
	21,		/*	aposbc	offset = 12	*/
	22,		/*	abosbl	offset = 13	*/
	8,			/* ring		offset = 14	*/
	13,		/*	dota		offset = 15	*/
	23,		/*	dacute	offset = 16	*/
	11,		/*	cedilla	offset = 17	*/
	14,		/*	ogonek	offset = 18	*/
	5,			/*	caron		offset = 19	*/
	15,		/*	stroke	offset = 20	*/
	24,		/*	bara 		offset = 21	*/
	3,			/*	breve		offset = 22	*/
	0,			/* dbls		offset = 23 sorts as 'ss'	*/
	25,		/*	dotlesi	offset = 24	*/
	26			/* dotlesj	offset = 25	*/
};


