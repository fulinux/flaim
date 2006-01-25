//-------------------------------------------------------------------------
// Desc:	Collation tables to convert to/from WP characters.
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
// $Id: f_coltb1.cpp 12245 2006-01-19 14:29:51 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

/* COMMENTS FROM WPSRTR.ASM */
/* The code calls tskip() that skips all commas. - don't know why */


/* Better agree with COLLSx+y values in wp_char.h and COLTBL.C */
/* Character offset is COLS1 - length is COLS11 - COLLS = SHOULD BE 203 */

FLMUINT16	colToWPChr[ COLS11 - COLLS ] = {
	0x20,				/* colls	-	<Spc> */
	0x2e,				/* colls+1	-	. */
	0x2c,				/* colls+2	-	, */
	0x3a,				/* colls+3	-	: */
	0x3b,				/* colls+4	-	; */
	0x21,				/* colls+5	-	! */
	0,					/* colls+6	-	NO VALUE */
	0x3f,				/* colls+7	-	? */
	0,					/* colls+8	-	NO VALUE */

	0x22,				/* cols1		-	" */
	0x27,				/* cols1+1	-	' */
	0x60,				/* cols1+2	-	` */
	0,					/* cols1+3	-	NO VALUE */
	0,					/* cols1+4	-	NO VALUE */

	0x28,				/* cols2		-	( */
	0x29,				/* cols2+1	-	) */
	0x5b,				/* cols2+2	-	japanese angle brackets */
	0x5d,				/* cols2+3	-	japanese angle brackets */
	0x7b,				/* cols2+4	-	{ */
	0x7d,				/* cols2+5	-	} */
	
	0x24,				/* cols3		-	$ */
	0x413,			/* cols3+1	-	cent */
	0x40b,			/* cols3+2	-	pound */
	0x40c,			/* cols3+3	-	yen */
	0x40d,			/* cols3+4	-	pacetes */
	0x40e,			/* cols3+5	-	floren */

	0x2b,				/* cols4		-	+ */
	0x2d,				/* cols4+1	-	- */
	0x2a,				/* cols4+2	-	* */
	0x2f,				/* cols4+3	-	/ */
	0x5e,				/* cols4+4	-	^ */
	0,					/* cols4+5	-	NO VALUE */
	0,					/* cols4+6	-	NO VALUE */
	0,					/* cols4+7	-	NO VALUE */

	0x3c,				/* cols5		-	< */
	0,					/* cols5+1	-	NO VALUE */
	0x3d,				/* cols5+2	-	= */
	0,					/* cols5+3	-	NO VALUE */
	0x3e,				/* cols5+4	-	> */
	0,					/* cols5+5	-	NO VALUE */
	0,					/* cols5+6	-	NO VALUE */
	0,					/* cols5+7	-	NO VALUE */
	0,					/* cols5+8	-	NO VALUE */
	0,					/* cols5+9	-	NO VALUE */
	0,					/* cols5+10	-	NO VALUE */
	0,					/* cols5+11	-	NO VALUE */
	0,					/* cols5+12	-	NO VALUE */
	0,					/* cols5+13	-	NO VALUE */

	0x25,				/* cols6		-	% */
	0x23,				/* cols6+1	-	# */
	0x26,				/* cols6+2	-	& */
	0x40,				/* cols6+3	-	@ */
	0x5c,				/* cols6+4	-	\ */
	0x5f,				/* cols6+5	-	_ */
	0x7c,				/* cols6+6	-	| */
	0x7e,				/* cols6+7	-	~ */
	0,					/* cols6+8	- NO VALUE */
	0,					/* cols6+9	- NO VALUE */
	0,					/* cols6+10	- NO VALUE */
	0,					/* cols6+11	- NO VALUE */
	0,					/* cols6+12	- NO VALUE */
	
	0x800,			/* cols7		-	Uppercase Alpha */
	0x802,			/* cols7+1	-	Uppercase Beta */
	0x806,			/* cols7+2	-	Uppercase Gamma */
	0x808,			/* cols7+3	-	Uppercase Delta */
	0x80a,			/* cols7+4	-	Uppercase Epsilon */
	0x80c,			/* cols7+5	-	Uppercase Zeta */
	0x80e,			/* cols7+6	-	Uppercase Eta */
	0x810,			/* cols7+7	-	Uppercase Theta */
	0x812,			/* cols7+8	-	Uppercase Iota */
	0x814,			/* cols7+9	-	Uppercase Kappa */
	0x816,			/* cols7+10	-	Uppercase Lambda */
	0x818,			/* cols7+11	-	Uppercase Mu */
	0x81a,			/* cols7+12	-	Uppercase Nu */
	0x81c,			/* cols7+13	-	Uppercase Xi */
	0x81e,			/* cols7+14	-	Uppercase Omicron */
	0x820,			/* cols7+15	-	Uppercase Pi */
	0x822,			/* cols7+16	-	Uppercase Rho */
	0x824,			/* cols7+17	-	Uppercase Sigma */
	0x828,			/* cols7+18	-	Uppercase Tau */
	0x82a,			/* cols7+19	-	Uppercase Upsilon */
	0x82c,			/* cols7+20	-	Uppercase Phi */
	0x82e,			/* cols7+21	-	Uppercase Chi */
	0x830,			/* cols7+22	-	Uppercase Psi */
	0x832,			/* cols7+23	-	Uppercase Omega */
	0,					/* cols7+24 - NO VALUE */
	
	0x30,				/* cols8		-	0 */
	0x31,				/* cols8+1	-	1 */
	0x32,				/* cols8+2	-	2 */
	0x33,				/* cols8+3	-	3 */
	0x34,				/* cols8+4	-	4 */
	0x35,				/* cols8+5	-	5 */
	0x36,				/* cols8+6	-	6 */
	0x37,				/* cols8+7	-	7 */
	0x38,				/* cols8+8	-	8 */
	0x39,				/* cols8+9	-	9 */
/*0x80*/	
	0x41,				/* cols9		-	A */
	0x124,			/* cols9+1	-	AE digraph */
	0x42,				/* cols9+2	-	B */
	0x43,				/* cols9+3	-	C */
	0xffff,			/* cols9+4	-	CH in spanish */
	0x162,			/* cols9+5	-	Holder for C caron in Czech */
	0x44,				/* cols9+6	-	D */
	0x45,				/* cols9+7	-	E */
	0x46,				/* cols9+8	-	F */
	0x47,				/* cols9+9	-	G */
	0x48,				/* cols9+10	-	H */
	0xffff,			/* cols9+11	-	CH in czech or dotless i in turkish */
	0x49,				/* cols9+12	-	I */
	0x18a,			/* cols9+13	-	IJ Digraph */
	0x4a,				/* cols9+14	-	J */
	0x4b,				/* cols9+15	-	K */
	0x4c,				/* cols9+16	-	L */
	0xffff,			/* cols9+17	-	LL in spanish */
	0x4d,				/* cols9+18	-	M */
	0x4e,				/* cols9+19	-	N */
	0x138,			/* cols9+20	-	N Tilde */
	0x4f,				/* cols9+21	-	O */
	0x1a6,			/* cols9+22	-	OE digraph */
	0x50,				/* cols9+23	-	P */
	0x51,				/* cols9+24	-	Q */
	0x52,				/* cols9+25	-	R */
	0x1aa,			/* cols9+26	-	Holder for R caron in Czech */
	0x53,				/* cols9+27	-	S */
	0x1b0,			/* cols9+28	-	Holder for S caron in Czech */
	0x54,				/* cols9+29	-	T */
	0x55,				/* cols9+30	-	U */
	0x56,				/* cols9+31	-	V */
/*0xA0*/
	0x57,				/* cols9+32	-	W */
	0x58,				/* cols9+33	-	X */
	0x59,				/* cols9+34	-	Y */
	0x5a,				/* cols9+35	-	Z */
	0x1ce,			/* cols9+36	-	Holder for Z caron in Czech */
	0x158,			/* cols9+37	-	Uppercase Thorn */
	0,					/* cols9+38	-	??? */
	0,					/* cols9+39	-	??? */
	0x5b,				/* cols9+40	-	[ (note: alphabetic - end of list) */
	0x5d,				/* cols9+41	-	] (note: alphabetic - end of list) */
/*0xAA - also start of Hebrew */
	0x124,			/* cols9+42	- AE diagraph - DK */
	0x124,			/* cols9+43 - AE diagraph - NO */
	0x122,			/* cols9+44 - A ring      - SW */
	0x11E,			/* cols9+45 - A diaeresis - DK */
	0x124,			/* cols9+46	- AE diagraph - IC */
	0x150,			/* cols9+47 - O slash     - NO */
	0x11e,			/* cols9+48	- A diaeresis - SW */
	0x150,			/* cols9+49	- O slash     - DK */
	0x13E,			/* cols9+50	- O Diaeresis - IC */
	0x122,			/* cols9+51	- A ring      - NO */
	0x13E,			/* cols9+52	- O Diaeresis - SW */
	0x13E,			/* cols9+53	- O Diaeresis - DK */
	0x150,			/* cols9+54 - O slash     - IC */
	0x122,			/* cols9+55	- A ring      - DK */
	0x124,			/* cols9+56	- AE diagraph future */
	0x13E,			/* cols9+57 - O Diaeresis future */
	0x150,			/* cols9+58 - O slash     future */
	0,					/* cols9+59 - NOT USED    future */
		
	0xA00,			/* cols10			-	Russian A */
	0xA02,			/* cols10+1		-	Russian BE */
	0xA04,			/* cols10+2		-	Russian VE */
	0xA06,			/* cols10+3		-	Russian GHE */
	0xA46,			/* cols10+4		-	Ukrainian HARD G */
	0xA08,			/* cols10+5		-	Russian DE */
	0xA4a,			/* cols10+6		-	Serbian SOFT DJ */
	0xA44,			/* cols10+7		-	Macedonian SOFT DJ */
	0xA0a,			/* cols10+8		-	Russian E */
	0xA0c,			/* cols10+9		- Russian YO */
	0xA4e,			/* cols10+10	-	Ukrainian YE */
	0xA0e,			/* cols10+11	-	Russian ZHE */
	0xA10,			/* cols10+12	-	Russian ZE */
	0xA52,			/* cols10+13	-	Macedonian ZELO */
	0xA12,			/* cols10+14	-	Russian I */
	0xA58,			/* cols10+15	-	Ukrainian I */
	0xA5a,			/* cols10+16	-	Ukrainian I with Two dots */
	0xA14,			/* cols10+17	-	Russian SHORT I */
	0xA5e,			/* cols10+18	-	Serbian--Macedonian JE */
	0xA16,			/* cols10+19	-	Russian KA */
	0xA18,			/* cols10+20	-	Russian EL */
	0xA68,			/* cols10+21	-	Serbian--Macedonian SOFT L */
	0xA1a,			/* cols10+22	-	Russian EM */
	0xA1c,			/* cols10+23	-	Russian EN */
	0xA6c,			/* cols10+24	-	Serbian--Macedonian SOFT N */
	0xA1e,			/* cols10+25	-	Russian O */
	0xA20,			/* cols10+26	-	Russian PE */
	0xA22,			/* cols10+27	-	Russian ER */
	0xA24,			/* cols10+28	-	Russian ES */
	0xA26,			/* cols10+29	-	Russian TE */
	0xA72,			/* cols10+30	-	Serbian SOFT T */
	0xA60,			/* cols10+31	-	Macedonian SOFT K */
	0xA28,			/* cols10+32	-	Russian U */
	0xA74,			/* cols10+33	-	Byelorussian SHORT U */
	0xA2a,			/* cols10+34	-	Russian EF */
	0xA2c,			/* cols10+35	-	Russian HA */
	0xA2e,			/* cols10+36	-	Russian TSE */
	0xA30,			/* cols10+37	-	Russian CHE */
	0xA86,			/* cols10+38	-	Serbian HARD DJ */
	0xA32,			/* cols10+39	-	Russian SHA */
	0xA34,			/* cols10+40	-	Russian SHCHA */
	0xA36,			/* cols10+41	-	Russian ER (also hard */
	0xA38,			/* cols10+42	-	Russian ERY */
	0xA3a,			/* cols10+43	-	Russian SOFT SIGN */
	0xA8e,			/* cols10+44	-	Old Russian YAT	*/
	0xA3c,			/* cols10+45	-	Russian uppercase	REVERSE E */
	0xA3e,			/* cols10+46	-	Russian YU */
	0xA40,			/* cols10+47	-	Russian YA */
	0xA3a,			/* cols10+48	-	Russian SOFT SIGN - UKRAIN ONLY */
 	0					/* cols10+49	- future */
};

FLMUINT16	HebArabColToWPChr[ ] = {
	/* Start at COLS10a+0 */
/*[0]*/
	0x0D00 +164,	/* hamzah */
	0x0D00 + 58,	/* [13,177] alef maddah */
						/* Read subcollation to get other alef values */
	0x0D00 + 60,	/* baa */
	0x0E00 + 48,	/* Sindhi bb */
	0x0E00 + 52,	/* Sindhi bh */
	0x0E00 + 56,	/* Misc p = peh */
	0x0D00 +152,	/* taa marbuuTah */
						/* subcollation of 1 is taa [13,64] */
	0x0E00 + 60,	/* Urdu T   [14,60] */
						/* Pashto T [14,64] */
/*[8]*/
	0x0D00 + 68,	/* thaa */
	0x0E00 + 68,	/* Sindhi th */
	0x0E00 + 72,	/* Sindhi tr */
	0x0E00 + 76,	/* Sindhi Th */
	0x0D00 + 72,	/* jiim - jeem */
	0x0E00 + 80,	/* Sindhi jj */
	0x0E00 + 84,	/* Sindhi ny */
	0x0E00 + 88,	/* Misc ch */
						/* Sinhi chh [14,92] */
/*[16]*/
	0x0D00 + 76,	/* Haa */
	0x0D00 + 80,	/* khaa */
	0x0E00 + 96,	/* Pashto ts */
	0x0E00 +100,	/* Pashto dz */

	0x0D00 + 84,	/* dal */
	0x0E00 +104,	/* Urdu D */
						/* Pashto D */
	0x0D00 + 86,	/* thal */
	0x0E00 +108,	/* Sindhi dh */
/*[24]*/
	0x0E00 +110,	/* Sindhi D  */
	0x0E00 +112,	/* Sindhi Dr */
	0x0E00 +114,	/* Sindhi Dh */
	
	0x0D00 + 88,	/* ra */
						/* Kurdish rolled r [14,122] */
	0x0E00 +116,	/* Pashto r [14,116] - must pick this! */
						/* Urdu R   [14,118] */
						/* Sindhi r [14,120]  */
	
	0x0D00 + 90,	/* zain */
	0x0E00 +126,	/* Mizc Z=jeh [14,126] */
						/* Pashto zz  [14,128] */
						/* Pashto g   [14,130] */
	
	0x0D00 + 92,	/* seen */
/*[32]*/
	0x0D00 + 96,	/* sheen */
	0x0E00 +132,	/* Pashto x */
	0x0D00 +100,	/* Sad */
	0x0D00 +104,	/* Dad */
	0x0D00 +108,	/* Tah */
	0x0D00 +112,	/* Za (dhah) */
	0x0D00 +116,	/* 'ain */
	0x0D00 +120,	/* ghain */
						/* malay ng [14,136] */
/*[40]*/
	0x0D00 +124,	/* fa */
	0x0E00 +140,	/* Malay p, kurdish v = veh */
						/* Sindhi ph [14,144] */
	0x0D00 +128,	/* Qaf */
	0x0D00 +132,	/* kaf (caf) */
						/* Misc k  [14,148] */
						/* misc k - no unicode [14,152] */
						/* Sindhi k [14,156] */

	0x0E00 +160,	/* Persian/Urdu gaf */
						/* gaf - no unicode [14,164] */
						/* malay g [14,168] */
						/* Sindhi ng [14,172] */
	0x0E00 +176,	/* Singhi gg */

	0x0D00 +136,	/* lam - all ligature variants */
						/* Kurdish valar lam [14,180] */
						/* Kurdish lamalef - no unicode [14,184] */

	0x0D00 +140,	/* meem */
/*[48]*/
	0x0D00 +144,	/* noon */
						/* Urdu n [14,186] */
						/* Pashto N [14,190] */
						/* Sindhi N [14,194] */
	0x0D00 +148,	/* ha - arabic language only! */
	0x0D00 +154,	/* waw */
						/* Kurdish o [14,198] */
						/* Kurdish o with bar [14,200] */
						/* Kurdish o with 2 dots [14,202] */
	0x0D00 +148,	/* ha - non-arabic language */
						/* Urdu h [14,204] */
						/* Farsi hamzah on ha [14,218] */
	0x0D00 +160,	/* alef maqsurah */
						/* Kurdish e - ya /w small v */
						
	0x0D00 +156,	/* ya */
	0x0E00 +212		/* Urdu ya barree */
						/* Malay ny [14,214] */
};


FLMUINT16	ArabSubColToWPChr[] = {
	0x0D00 +177,	/* Alef maddah - default value - here for documentation */
	0x0D00 +165,	/* Alef Hamzah */
	0x0D00 +169,	/* Waw hamzah */
	0x0D00 +167,	/* Hamzah under alef */
	0x0D00 +171,	/* ya hamzah */
	0x0D00 +175,	/* alef fathattan */
	0x0D00 +179,	/* alef waslah */
	0x0D00 + 58,	/* alef */
	0x0D00 + 64		/* taa - after taa marbuuTah */
};


	/* Turns a collated diacritic value into the original diacritic value */
FLMBYTE ml1_COLtoD[27] = {
	23,		/* 	dbls	sort value = 0  sorts as 'ss'	*/
	6,			/*	acute	sort value = 1	*/
	0,			/* 	grave	sort value = 2	*/
	22,		/*	breve	sort value = 3 	*/
	3,			/*	circum	sort value = 4	*/
	19,		/*	caron	sort value = 5 	*/
	7,			/*	umlaut	sort value = 6	*/
	2,			/*	tilde	sort value = 7	*/
	14,		/* 	ring	sort value = 8 	*/
	7,			/*	umlaut in SU,SV & CZ after ring = 9 */
	5,			/*	slash	sort value = 10	*/
	17,	 	/*	cedilla	sort value = 11	*/
	4,			/*	crossb	sort value = 12	*/
	15,	 	/*	dota	sort value = 13	*/
	18,	 	/*	ogonek	sort value = 14	*/
	20,	 	/*	stroke	sort value = 15	*/
	1, 	 	/*	centerd	sort value = 16	*/
	8,			/*	macron	sort value = 17	*/
	9,			/*	aposab	sort value = 18	*/
	10,	 	/*	aposbes	sort value = 19	*/
	11,	 	/*	aposba	sort value = 20	*/
	12,	 	/*	aposbc	sort value = 21	*/
	13,	 	/*	abosbl	sort value = 22	*/
	16,	 	/*	dacute	sort value = 23	*/
	21,	 	/*	bara 	sort value = 24	*/
	24,	 	/*	dotlesi	sort value = 25	*/
	25			/* 	dotlesj	sort value = 26	*/
	};

