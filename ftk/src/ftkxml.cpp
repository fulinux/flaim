//------------------------------------------------------------------------------
//	Desc:	XML parser
//
// Tabs:	3
//
//		Copyright (c) 2000-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fxml.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "ftksys.h"

#define FLM_XML_BASE_CHAR			0x01
#define FLM_XML_IDEOGRAPHIC		0x02
#define FLM_XML_COMBINING_CHAR	0x04
#define FLM_XML_DIGIT				0x08
#define FLM_XML_EXTENDER			0x10
#define FLM_XML_WHITESPACE			0x20

typedef struct xmlChar
{
	FLMBYTE		ucFlags;
} XMLCHAR;
	
typedef struct
{
	char *			pszEntity;
	FLMUINT			uiValue;
} CharEntity;

static FLMUNICODE gv_puzXMLPrefix[] =
{
	FLM_UNICODE_x,
	FLM_UNICODE_m,
	FLM_UNICODE_l,
	0
};

FLMUNICODE gv_puzXMLNSURI[] =
{
	FLM_UNICODE_h,
	FLM_UNICODE_t,
	FLM_UNICODE_t,
	FLM_UNICODE_p,
	FLM_UNICODE_COLON,
	FLM_UNICODE_FSLASH,
	FLM_UNICODE_FSLASH,
	FLM_UNICODE_w,
	FLM_UNICODE_w,
	FLM_UNICODE_w,
	FLM_UNICODE_PERIOD,
	FLM_UNICODE_w,
	FLM_UNICODE_3,
	FLM_UNICODE_c,
	FLM_UNICODE_PERIOD,
	FLM_UNICODE_o,
	FLM_UNICODE_r,
	FLM_UNICODE_g,
	FLM_UNICODE_FSLASH,
	FLM_UNICODE_T,
	FLM_UNICODE_R,
	FLM_UNICODE_FSLASH,
	FLM_UNICODE_1,
	FLM_UNICODE_9,
	FLM_UNICODE_9,
	FLM_UNICODE_9,
	FLM_UNICODE_FSLASH,
	FLM_UNICODE_R,
	FLM_UNICODE_E,
	FLM_UNICODE_C,
	FLM_UNICODE_HYPHEN,
	FLM_UNICODE_x,
	FLM_UNICODE_m,
	FLM_UNICODE_l,
	FLM_UNICODE_HYPHEN,
	FLM_UNICODE_n,
	FLM_UNICODE_a,
	FLM_UNICODE_m,
	FLM_UNICODE_e,
	FLM_UNICODE_s,
	FLM_UNICODE_HYPHEN,
	FLM_UNICODE_1,
	FLM_UNICODE_9,
	FLM_UNICODE_9,
	FLM_UNICODE_9,
	FLM_UNICODE_0,
	FLM_UNICODE_1,
	FLM_UNICODE_1,
	FLM_UNICODE_4,
	0
};

typedef struct
{
	FLMUNICODE		uLowChar;
	FLMUNICODE		uHighChar;
	FLMUINT16		ui16Flag;
} CHAR_TBL;

static CHAR_TBL charTbl[] = 
{
	{ 0x0041, 0x005A, FLM_XML_BASE_CHAR},
	{ 0x0061, 0x007A, FLM_XML_BASE_CHAR},
	{ 0x00C0, 0x00D6, FLM_XML_BASE_CHAR},
	{ 0x00D8, 0x00F6, FLM_XML_BASE_CHAR},
	{ 0x00F8, 0x00FF, FLM_XML_BASE_CHAR},
	{ 0x0100, 0x0131, FLM_XML_BASE_CHAR},
	{ 0x0134, 0x013E, FLM_XML_BASE_CHAR},
	{ 0x0141, 0x0148, FLM_XML_BASE_CHAR},
	{ 0x014A, 0x017E, FLM_XML_BASE_CHAR},
	{ 0x0180, 0x01C3, FLM_XML_BASE_CHAR},
	{ 0x01CD, 0x01F0, FLM_XML_BASE_CHAR},
	{ 0x01F4, 0x01F5, FLM_XML_BASE_CHAR},
	{ 0x01FA, 0x0217, FLM_XML_BASE_CHAR},
	{ 0x0250, 0x02A8, FLM_XML_BASE_CHAR},
	{ 0x02BB, 0x02C1, FLM_XML_BASE_CHAR},
	{ 0x0386, 0x0386, FLM_XML_BASE_CHAR},
	{ 0x0388, 0x038A, FLM_XML_BASE_CHAR},
	{ 0x038C, 0x038C, FLM_XML_BASE_CHAR},
	{ 0x038E, 0x03A1, FLM_XML_BASE_CHAR},
	{ 0x03A3, 0x03CE, FLM_XML_BASE_CHAR},
	{ 0x03D0, 0x03D6, FLM_XML_BASE_CHAR},
	{ 0x03DA, 0x03DA, FLM_XML_BASE_CHAR},
	{ 0x03DC, 0x03DC, FLM_XML_BASE_CHAR},
	{ 0x03DE, 0x03DE, FLM_XML_BASE_CHAR},
	{ 0x03E0, 0x03E0, FLM_XML_BASE_CHAR},
	{ 0x03E2, 0x03F3, FLM_XML_BASE_CHAR},
	{ 0x0401, 0x040C, FLM_XML_BASE_CHAR},
	{ 0x040E, 0x044F, FLM_XML_BASE_CHAR},
	{ 0x0451, 0x045C, FLM_XML_BASE_CHAR},
	{ 0x045E, 0x0481, FLM_XML_BASE_CHAR},
	{ 0x0490, 0x04C4, FLM_XML_BASE_CHAR},
	{ 0x04C7, 0x04C8, FLM_XML_BASE_CHAR},
	{ 0x04CB, 0x04CC, FLM_XML_BASE_CHAR},
	{ 0x04D0, 0x04EB, FLM_XML_BASE_CHAR},
	{ 0x04EE, 0x04F5, FLM_XML_BASE_CHAR},
	{ 0x04F8, 0x04F9, FLM_XML_BASE_CHAR},
	{ 0x0531, 0x0556, FLM_XML_BASE_CHAR},
	{ 0x0559, 0x0559, FLM_XML_BASE_CHAR},
	{ 0x0561, 0x0586, FLM_XML_BASE_CHAR},
	{ 0x05D0, 0x05EA, FLM_XML_BASE_CHAR},
	{ 0x05F0, 0x05F2, FLM_XML_BASE_CHAR},
	{ 0x0621, 0x063A, FLM_XML_BASE_CHAR},
	{ 0x0641, 0x06B7, FLM_XML_BASE_CHAR},
	{ 0x06BA, 0x06BE, FLM_XML_BASE_CHAR},
	{ 0x06C0, 0x06CE, FLM_XML_BASE_CHAR},
	{ 0x06D0, 0x06D3, FLM_XML_BASE_CHAR},
	{ 0x06D5, 0x06D5, FLM_XML_BASE_CHAR},
	{ 0x06E5, 0x06E6, FLM_XML_BASE_CHAR},
	{ 0x0905, 0x0939, FLM_XML_BASE_CHAR},
	{ 0x093D, 0x093D, FLM_XML_BASE_CHAR},
	{ 0x0958, 0x0961, FLM_XML_BASE_CHAR},
	{ 0x0985, 0x098C, FLM_XML_BASE_CHAR},
	{ 0x098F, 0x0990, FLM_XML_BASE_CHAR},
	{ 0x0993, 0x09A8, FLM_XML_BASE_CHAR},
	{ 0x09AA, 0x09B0, FLM_XML_BASE_CHAR},
	{ 0x09B2, 0x09B2, FLM_XML_BASE_CHAR},
	{ 0x09B6, 0x09B9, FLM_XML_BASE_CHAR},
	{ 0x0061, 0x007A, FLM_XML_BASE_CHAR},
	{ 0x09DC, 0x09DD, FLM_XML_BASE_CHAR},
	{ 0x09DF, 0x09E1, FLM_XML_BASE_CHAR},
	{ 0x09F0, 0x09F1, FLM_XML_BASE_CHAR},
	{ 0x0A05, 0x0A0A, FLM_XML_BASE_CHAR},
	{ 0x0A0F, 0x0A10, FLM_XML_BASE_CHAR},
	{ 0x0A13, 0x0A28, FLM_XML_BASE_CHAR},
	{ 0x0A2A, 0x0A30, FLM_XML_BASE_CHAR},
	{ 0x0A32, 0x0A33, FLM_XML_BASE_CHAR},
	{ 0x0A35, 0x0A36, FLM_XML_BASE_CHAR},
	{ 0x0A38, 0x0A39, FLM_XML_BASE_CHAR},
	{ 0x0A59, 0x0A5C, FLM_XML_BASE_CHAR},
	{ 0x0A5E, 0x0A5E, FLM_XML_BASE_CHAR},
	{ 0x0A72, 0x0A74, FLM_XML_BASE_CHAR},
	{ 0x0A85, 0x0A8B, FLM_XML_BASE_CHAR},
	{ 0x0A8D, 0x0A8D, FLM_XML_BASE_CHAR},
	{ 0x0A8F, 0x0A91, FLM_XML_BASE_CHAR},
	{ 0x0A93, 0x0AA8, FLM_XML_BASE_CHAR},
	{ 0x0AAA, 0x0AB0, FLM_XML_BASE_CHAR},
	{ 0x0AB2, 0x0AB3, FLM_XML_BASE_CHAR},
	{ 0x0AB5, 0x0AB9, FLM_XML_BASE_CHAR},
	{ 0x0ABD, 0x0ABD, FLM_XML_BASE_CHAR},
	{ 0x0AE0, 0x0AE0, FLM_XML_BASE_CHAR},
	{ 0x0B05, 0x0B0C, FLM_XML_BASE_CHAR},
	{ 0x0B0F, 0x0B10, FLM_XML_BASE_CHAR},
	{ 0x0B13, 0x0B28, FLM_XML_BASE_CHAR},
	{ 0x0B2A, 0x0B30, FLM_XML_BASE_CHAR},
	{ 0x0B32, 0x0B33, FLM_XML_BASE_CHAR},
	{ 0x0B36, 0x0B39, FLM_XML_BASE_CHAR},
	{ 0x0B3D, 0x0B3D, FLM_XML_BASE_CHAR},
	{ 0x0B5C, 0x0B5D, FLM_XML_BASE_CHAR},
	{ 0x0B5F, 0x0B61, FLM_XML_BASE_CHAR},
	{ 0x0B85, 0x0B8A, FLM_XML_BASE_CHAR},
	{ 0x0B8E, 0x0B90, FLM_XML_BASE_CHAR},
	{ 0x0B92, 0x0B95, FLM_XML_BASE_CHAR},
	{ 0x0B99, 0x0B9A, FLM_XML_BASE_CHAR},
	{ 0x0B9C, 0x0B9C, FLM_XML_BASE_CHAR},
	{ 0x0B9E, 0x0B9F, FLM_XML_BASE_CHAR},
	{ 0x0BA3, 0x0BA4, FLM_XML_BASE_CHAR},
	{ 0x0BA8, 0x0BAA, FLM_XML_BASE_CHAR},
	{ 0x0BAE, 0x0BB5, FLM_XML_BASE_CHAR},
	{ 0x0BB7, 0x0BB9, FLM_XML_BASE_CHAR},
	{ 0x0C05, 0x0C0C, FLM_XML_BASE_CHAR},
	{ 0x0C0E, 0x0C10, FLM_XML_BASE_CHAR},
	{ 0x0C12, 0x0C28, FLM_XML_BASE_CHAR},
	{ 0x0C2A, 0x0C33, FLM_XML_BASE_CHAR},
	{ 0x0C35, 0x0C39, FLM_XML_BASE_CHAR},
	{ 0x0C60, 0x0C61, FLM_XML_BASE_CHAR},
	{ 0x0C85, 0x0C8C, FLM_XML_BASE_CHAR},
	{ 0x0C8E, 0x0C90, FLM_XML_BASE_CHAR},
	{ 0x0C92, 0x0CA8, FLM_XML_BASE_CHAR},
	{ 0x0CAA, 0x0CB3, FLM_XML_BASE_CHAR},
	{ 0x0CB5, 0x0CB9, FLM_XML_BASE_CHAR},
	{ 0x0CDE, 0x0CDE, FLM_XML_BASE_CHAR},
	{ 0x0CE0, 0x0CE1, FLM_XML_BASE_CHAR},
	{ 0x0D05, 0x0D0C, FLM_XML_BASE_CHAR},
	{ 0x0D0E, 0x0D10, FLM_XML_BASE_CHAR},
	{ 0x0D12, 0x0D28, FLM_XML_BASE_CHAR},
	{ 0x0D2A, 0x0D39, FLM_XML_BASE_CHAR},
	{ 0x0D60, 0x0D61, FLM_XML_BASE_CHAR},
	{ 0x0E01, 0x0E2E, FLM_XML_BASE_CHAR},
	{ 0x0E30, 0x0E30, FLM_XML_BASE_CHAR},
	{ 0x0E32, 0x0E33, FLM_XML_BASE_CHAR},
	{ 0x0E40, 0x0E45, FLM_XML_BASE_CHAR},
	{ 0x0E81, 0x0E82, FLM_XML_BASE_CHAR},
	{ 0x0E84, 0x0E84, FLM_XML_BASE_CHAR},
	{ 0x0E87, 0x0E88, FLM_XML_BASE_CHAR},
	{ 0x0E8A, 0x0E8A, FLM_XML_BASE_CHAR},
	{ 0x0E8D, 0x0E8D, FLM_XML_BASE_CHAR},
	{ 0x0E94, 0x0E97, FLM_XML_BASE_CHAR},
	{ 0x0E99, 0x0E9F, FLM_XML_BASE_CHAR},
	{ 0x0EA1, 0x0EA3, FLM_XML_BASE_CHAR},
	{ 0x0EA5, 0x0EA5, FLM_XML_BASE_CHAR},
	{ 0x0EA7, 0x0EA7, FLM_XML_BASE_CHAR},
	{ 0x0EAA, 0x0EAB, FLM_XML_BASE_CHAR},
	{ 0x0EAD, 0x0EAE, FLM_XML_BASE_CHAR},
	{ 0x0EB0, 0x0EB0, FLM_XML_BASE_CHAR},
	{ 0x0EB2, 0x0EB3, FLM_XML_BASE_CHAR},
	{ 0x0EBD, 0x0EBD, FLM_XML_BASE_CHAR},
	{ 0x0EC0, 0x0EC4, FLM_XML_BASE_CHAR},
	{ 0x0F40, 0x0F47, FLM_XML_BASE_CHAR},
	{ 0x0F49, 0x0F69, FLM_XML_BASE_CHAR},
	{ 0x10A0, 0x10C5, FLM_XML_BASE_CHAR},
	{ 0x10D0, 0x10F6, FLM_XML_BASE_CHAR},
	{ 0x1100, 0x1100, FLM_XML_BASE_CHAR},
	{ 0x1102, 0x1103, FLM_XML_BASE_CHAR},
	{ 0x1105, 0x1107, FLM_XML_BASE_CHAR},
	{ 0x1109, 0x1109, FLM_XML_BASE_CHAR},
	{ 0x110B, 0x110C, FLM_XML_BASE_CHAR},
	{ 0x110E, 0x1112, FLM_XML_BASE_CHAR},
	{ 0x113C, 0x113C, FLM_XML_BASE_CHAR},
	{ 0x113E, 0x113E, FLM_XML_BASE_CHAR},
	{ 0x1140, 0x1140, FLM_XML_BASE_CHAR},
	{ 0x114C, 0x114C, FLM_XML_BASE_CHAR},
	{ 0x114E, 0x114E, FLM_XML_BASE_CHAR},
	{ 0x1150, 0x1150, FLM_XML_BASE_CHAR},
	{ 0x1154, 0x1155, FLM_XML_BASE_CHAR},
	{ 0x1159, 0x1159, FLM_XML_BASE_CHAR},
	{ 0x115F, 0x1161, FLM_XML_BASE_CHAR},
	{ 0x1163, 0x1163, FLM_XML_BASE_CHAR},
	{ 0x1165, 0x1165, FLM_XML_BASE_CHAR},
	{ 0x1167, 0x1167, FLM_XML_BASE_CHAR},
	{ 0x1169, 0x1169, FLM_XML_BASE_CHAR},
	{ 0x116D, 0x116E, FLM_XML_BASE_CHAR},
	{ 0x1172, 0x1173, FLM_XML_BASE_CHAR},
	{ 0x1175, 0x1175, FLM_XML_BASE_CHAR},
	{ 0x119E, 0x119E, FLM_XML_BASE_CHAR},
	{ 0x11A8, 0x11A8, FLM_XML_BASE_CHAR},
	{ 0x11AB, 0x11AB, FLM_XML_BASE_CHAR},
	{ 0x11AE, 0x11AF, FLM_XML_BASE_CHAR},
	{ 0x11B7, 0x11B8, FLM_XML_BASE_CHAR},
	{ 0x11BA, 0x11BA, FLM_XML_BASE_CHAR},
	{ 0x11BC, 0x11C2, FLM_XML_BASE_CHAR},
	{ 0x11EB, 0x11EB, FLM_XML_BASE_CHAR},
	{ 0x11F0, 0x11F0, FLM_XML_BASE_CHAR},
	{ 0x11F9, 0x11F9, FLM_XML_BASE_CHAR},
	{ 0x1E00, 0x1E9B, FLM_XML_BASE_CHAR},
	{ 0x1EA0, 0x1EF9, FLM_XML_BASE_CHAR},
	{ 0x1F00, 0x1F15, FLM_XML_BASE_CHAR},
	{ 0x1F18, 0x1F1D, FLM_XML_BASE_CHAR},
	{ 0x1F20, 0x1F45, FLM_XML_BASE_CHAR},
	{ 0x1F48, 0x1F4D, FLM_XML_BASE_CHAR},
	{ 0x1F50, 0x1F57, FLM_XML_BASE_CHAR},
	{ 0x1F59, 0x1F59, FLM_XML_BASE_CHAR},
	{ 0x1F5B, 0x1F5B, FLM_XML_BASE_CHAR},
	{ 0x1F5D, 0x1F5D, FLM_XML_BASE_CHAR},
	{ 0x1F5F, 0x1F7D, FLM_XML_BASE_CHAR},
	{ 0x1F80, 0x1FB4, FLM_XML_BASE_CHAR},
	{ 0x1FB6, 0x1FBC, FLM_XML_BASE_CHAR},
	{ 0x1FBE, 0x1FBE, FLM_XML_BASE_CHAR},
	{ 0x1FC2, 0x1FC4, FLM_XML_BASE_CHAR},
	{ 0x1FC6, 0x1FCC, FLM_XML_BASE_CHAR},
	{ 0x1FD0, 0x1FD3, FLM_XML_BASE_CHAR},
	{ 0x1FD6, 0x1FDB, FLM_XML_BASE_CHAR},
	{ 0x1FE0, 0x1FEC, FLM_XML_BASE_CHAR},
	{ 0x1FF2, 0x1FF4, FLM_XML_BASE_CHAR},
	{ 0x1FF6, 0x1FFC, FLM_XML_BASE_CHAR},
	{ 0x2126, 0x2126, FLM_XML_BASE_CHAR},
	{ 0x212A, 0x212B, FLM_XML_BASE_CHAR},
	{ 0x212E, 0x212E, FLM_XML_BASE_CHAR},
	{ 0x2180, 0x2182, FLM_XML_BASE_CHAR},
	{ 0x3041, 0x3094, FLM_XML_BASE_CHAR},
	{ 0x30A1, 0x30FA, FLM_XML_BASE_CHAR},
	{ 0x3105, 0x312C, FLM_XML_BASE_CHAR},
	{ 0xAC00, 0xD7A3, FLM_XML_BASE_CHAR},

	{ 0x4E00, 0x9FA5, FLM_XML_IDEOGRAPHIC},
	{ 0x3007, 0x3007, FLM_XML_IDEOGRAPHIC},
	{ 0x3021, 0x3029, FLM_XML_IDEOGRAPHIC},

	{ 0x0300, 0x0345, FLM_XML_COMBINING_CHAR},
	{ 0x0360, 0x0361, FLM_XML_COMBINING_CHAR},
	{ 0x0483, 0x0486, FLM_XML_COMBINING_CHAR},
	{ 0x0591, 0x05A1, FLM_XML_COMBINING_CHAR},
	{ 0x05A3, 0x05B9, FLM_XML_COMBINING_CHAR},
	{ 0x05BB, 0x05BD, FLM_XML_COMBINING_CHAR},
	{ 0x05BF, 0x05BF, FLM_XML_COMBINING_CHAR},
	{ 0x05C1, 0x05C2, FLM_XML_COMBINING_CHAR},
	{ 0x05C4, 0x05C4, FLM_XML_COMBINING_CHAR},
	{ 0x064B, 0x0652, FLM_XML_COMBINING_CHAR},
	{ 0x0670, 0x0670, FLM_XML_COMBINING_CHAR},
	{ 0x06D6, 0x06DC, FLM_XML_COMBINING_CHAR},
	{ 0x06DD, 0x06DF, FLM_XML_COMBINING_CHAR},
	{ 0x06E0, 0x06E4, FLM_XML_COMBINING_CHAR},
	{ 0x06E7, 0x06E8, FLM_XML_COMBINING_CHAR},
	{ 0x06EA, 0x06ED, FLM_XML_COMBINING_CHAR},
	{ 0x0901, 0x0903, FLM_XML_COMBINING_CHAR},
	{ 0x093C, 0x093C, FLM_XML_COMBINING_CHAR},
	{ 0x093E, 0x094C, FLM_XML_COMBINING_CHAR},
	{ 0x094D, 0x094D, FLM_XML_COMBINING_CHAR},
	{ 0x0951, 0x0954, FLM_XML_COMBINING_CHAR},
	{ 0x0962, 0x0963, FLM_XML_COMBINING_CHAR},
	{ 0x0981, 0x0983, FLM_XML_COMBINING_CHAR},
	{ 0x09BC, 0x09BC, FLM_XML_COMBINING_CHAR},
	{ 0x09BE, 0x09BE, FLM_XML_COMBINING_CHAR},
	{ 0x09BF, 0x09BF, FLM_XML_COMBINING_CHAR},
	{ 0x09C0, 0x09C4, FLM_XML_COMBINING_CHAR},
	{ 0x09C7, 0x09C8, FLM_XML_COMBINING_CHAR},
	{ 0x09CB, 0x09CD, FLM_XML_COMBINING_CHAR},
	{ 0x09D7, 0x09D7, FLM_XML_COMBINING_CHAR},
	{ 0x09E2, 0x09E3, FLM_XML_COMBINING_CHAR},
	{ 0x0A02, 0x0A02, FLM_XML_COMBINING_CHAR},
	{ 0x0A3C, 0x0A3C, FLM_XML_COMBINING_CHAR},
	{ 0x0A3E, 0x0A3E, FLM_XML_COMBINING_CHAR},
	{ 0x0A3F, 0x0A3F, FLM_XML_COMBINING_CHAR},
	{ 0x0A40, 0x0A42, FLM_XML_COMBINING_CHAR},
	{ 0x0A47, 0x0A48, FLM_XML_COMBINING_CHAR},
	{ 0x0A4B, 0x0A4D, FLM_XML_COMBINING_CHAR},
	{ 0x0A70, 0x0A71, FLM_XML_COMBINING_CHAR},
	{ 0x0A81, 0x0A83, FLM_XML_COMBINING_CHAR},
	{ 0x0ABC, 0x0ABC, FLM_XML_COMBINING_CHAR},
	{ 0x0ABE, 0x0AC5, FLM_XML_COMBINING_CHAR},
	{ 0x0AC7, 0x0AC9, FLM_XML_COMBINING_CHAR},
	{ 0x0ACB, 0x0ACD, FLM_XML_COMBINING_CHAR},
	{ 0x0B01, 0x0B03, FLM_XML_COMBINING_CHAR},
	{ 0x0B3C, 0x0B3C, FLM_XML_COMBINING_CHAR},
	{ 0x0B3E, 0x0B43, FLM_XML_COMBINING_CHAR},
	{ 0x0B47, 0x0B48, FLM_XML_COMBINING_CHAR},
	{ 0x0B4B, 0x0B4D, FLM_XML_COMBINING_CHAR},
	{ 0x0B56, 0x0B57, FLM_XML_COMBINING_CHAR},
	{ 0x0B82, 0x0B83, FLM_XML_COMBINING_CHAR},
	{ 0x0BBE, 0x0BC2, FLM_XML_COMBINING_CHAR},
	{ 0x0BC6, 0x0BC8, FLM_XML_COMBINING_CHAR},
	{ 0x0BCA, 0x0BCD, FLM_XML_COMBINING_CHAR},
	{ 0x0BD7, 0x0BD7, FLM_XML_COMBINING_CHAR},
	{ 0x0C01, 0x0C03, FLM_XML_COMBINING_CHAR},
	{ 0x0C3E, 0x0C44, FLM_XML_COMBINING_CHAR},
	{ 0x0C46, 0x0C48, FLM_XML_COMBINING_CHAR},
	{ 0x0C4A, 0x0C4D, FLM_XML_COMBINING_CHAR},
	{ 0x0C55, 0x0C56, FLM_XML_COMBINING_CHAR},
	{ 0x0C82, 0x0C83, FLM_XML_COMBINING_CHAR},
	{ 0x0CBE, 0x0CC4, FLM_XML_COMBINING_CHAR},
	{ 0x0CC6, 0x0CC8, FLM_XML_COMBINING_CHAR},
	{ 0x0CCA, 0x0CCD, FLM_XML_COMBINING_CHAR},
	{ 0x0CD5, 0x0CD6, FLM_XML_COMBINING_CHAR},
	{ 0x0D02, 0x0D03, FLM_XML_COMBINING_CHAR},
	{ 0x0D3E, 0x0D43, FLM_XML_COMBINING_CHAR},
	{ 0x0D46, 0x0D48, FLM_XML_COMBINING_CHAR},
	{ 0x0D4A, 0x0D4D, FLM_XML_COMBINING_CHAR},
	{ 0x0D57, 0x0D57, FLM_XML_COMBINING_CHAR},
	{ 0x0E31, 0x0E31, FLM_XML_COMBINING_CHAR},
	{ 0x0E34, 0x0E3A, FLM_XML_COMBINING_CHAR},
	{ 0x0E47, 0x0E4E, FLM_XML_COMBINING_CHAR},
	{ 0x0EB1, 0x0EB1, FLM_XML_COMBINING_CHAR},
	{ 0x0EB4, 0x0EB9, FLM_XML_COMBINING_CHAR},
	{ 0x0EBB, 0x0EBC, FLM_XML_COMBINING_CHAR},
	{ 0x0EC8, 0x0ECD, FLM_XML_COMBINING_CHAR},
	{ 0x0F18, 0x0F19, FLM_XML_COMBINING_CHAR},
	{ 0x0F35, 0x0F35, FLM_XML_COMBINING_CHAR},
	{ 0x0F37, 0x0F37, FLM_XML_COMBINING_CHAR},
	{ 0x0F39, 0x0F39, FLM_XML_COMBINING_CHAR},
	{ 0x0F3E, 0x0F3E, FLM_XML_COMBINING_CHAR},
	{ 0x0F3F, 0x0F3F, FLM_XML_COMBINING_CHAR},
	{ 0x0F71, 0x0F84, FLM_XML_COMBINING_CHAR},
	{ 0x0F86, 0x0F8B, FLM_XML_COMBINING_CHAR},
	{ 0x0F90, 0x0F95, FLM_XML_COMBINING_CHAR},
	{ 0x0F97, 0x0F97, FLM_XML_COMBINING_CHAR},
	{ 0x0F99, 0x0FAD, FLM_XML_COMBINING_CHAR},
	{ 0x0FB1, 0x0FB7, FLM_XML_COMBINING_CHAR},
	{ 0x0FB9, 0x0FB9, FLM_XML_COMBINING_CHAR},
	{ 0x20D0, 0x20DC, FLM_XML_COMBINING_CHAR},
	{ 0x20E1, 0x20E1, FLM_XML_COMBINING_CHAR},
	{ 0x302A, 0x302F, FLM_XML_COMBINING_CHAR},
	{ 0x3099, 0x3099, FLM_XML_COMBINING_CHAR},
	{ 0x309A, 0x309A, FLM_XML_COMBINING_CHAR},

	{ 0x0030, 0x0039, FLM_XML_DIGIT},
	{ 0x0660, 0x0669, FLM_XML_DIGIT},
	{ 0x06F0, 0x06F9, FLM_XML_DIGIT},
	{ 0x0966, 0x096F, FLM_XML_DIGIT},
	{ 0x09E6, 0x09EF, FLM_XML_DIGIT},
	{ 0x0A66, 0x0A6F, FLM_XML_DIGIT},
	{ 0x0AE6, 0x0AEF, FLM_XML_DIGIT},
	{ 0x0B66, 0x0B6F, FLM_XML_DIGIT},
	{ 0x0BE7, 0x0BEF, FLM_XML_DIGIT},
	{ 0x0C66, 0x0C6F, FLM_XML_DIGIT},
	{ 0x0CE6, 0x0CEF, FLM_XML_DIGIT},
	{ 0x0D66, 0x0D6F, FLM_XML_DIGIT},
	{ 0x0E50, 0x0E59, FLM_XML_DIGIT},
	{ 0x0ED0, 0x0ED9, FLM_XML_DIGIT},
	{ 0x0F20, 0x0F29, FLM_XML_DIGIT},

	{ 0x00B7, 0x00B7, FLM_XML_EXTENDER},
	{ 0x02D0, 0x02D0, FLM_XML_EXTENDER},
	{ 0x02D1, 0x02D1, FLM_XML_EXTENDER},
	{ 0x0387, 0x0387, FLM_XML_EXTENDER},
	{ 0x0640, 0x0640, FLM_XML_EXTENDER},
	{ 0x0E46, 0x0E46, FLM_XML_EXTENDER},
	{ 0x0EC6, 0x0EC6, FLM_XML_EXTENDER},
	{ 0x3005, 0x3005, FLM_XML_EXTENDER},
	{ 0x3031, 0x3035, FLM_XML_EXTENDER},
	{ 0x309D, 0x309E, FLM_XML_EXTENDER},
	{ 0x30FC, 0x30FE, FLM_XML_EXTENDER},

	{ 0x0009, 0x0009, FLM_XML_WHITESPACE},
	{ 0x000A, 0x000A, FLM_XML_WHITESPACE},
	{ 0x000D, 0x000D, FLM_XML_WHITESPACE},
	{ 0x0020, 0x0020, FLM_XML_WHITESPACE},
   { 0, 0, 0}
};

//FSTATIC RCODE exportUniValue(
//	IF_OStream *	pOStream,
//	FLMUNICODE *	puzStr,
//	FLMUINT			uiStrChars,
//	FLMBOOL			bEncodeSpecialChars,
//	FLMUINT			uiIndentCount);
	
/****************************************************************************
Desc: XML
****************************************************************************/
class F_XML : public IF_XML, public virtual F_Base
{
public:

	F_XML();

	virtual ~F_XML();
	
	RCODE FLMAPI setup( void);

	FLMBOOL FLMAPI isPubidChar(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isQuoteChar(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isWhitespace(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isExtender(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isCombiningChar(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isNameChar(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isNCNameChar(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isIdeographic(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isBaseChar(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isDigit(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isLetter(
		FLMUNICODE		uChar);

	FLMBOOL FLMAPI isNameValid(
		FLMUNICODE *	puzName,
		FLMBYTE *		pszName);

private:

	void setCharFlag(
		FLMUNICODE		uLowChar,
		FLMUNICODE		uHighChar,
		FLMUINT16		ui16Flag);

	XMLCHAR *			m_pCharTable;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_XMLNamespace : public F_RefCount, public F_Base
{
public:

	FINLINE F_XMLNamespace()
	{
		m_puzPrefix = NULL;
		m_puzURI = NULL;
		m_pNext = NULL;
	}

	virtual FINLINE ~F_XMLNamespace()
	{
		flmAssert( !m_pNext);

		if( m_puzPrefix)
		{
			f_free( &m_puzPrefix);
		}

		if( m_puzURI)
		{
			f_free( &m_puzURI);
		}
	}

	RCODE setPrefix(
		FLMUNICODE *		puzPrefix);

	RCODE setURI(
		FLMUNICODE *		puzURI);

	RCODE setup(
		FLMUNICODE *		puzPrefix,
		FLMUNICODE *		puzURI,
		F_XMLNamespace *	pNext);

	FINLINE FLMUNICODE * getPrefixPtr( void)
	{
		return( m_puzPrefix);
	}

	FINLINE FLMUNICODE * getURIPtr( void)
	{
		return( m_puzURI);
	}

private:

	FLMUNICODE *		m_puzPrefix;
	FLMUNICODE *		m_puzURI;
	F_XMLNamespace *	m_pNext;

friend class F_XMLNamespaceMgr;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_XMLNamespaceMgr : public F_RefCount, public virtual F_Base
{
public:

	F_XMLNamespaceMgr();

	virtual ~F_XMLNamespaceMgr();

	RCODE findNamespace(
		FLMUNICODE *		puzPrefix,
		F_XMLNamespace **	ppNamespace,
		FLMUINT				uiMaxSearchSize = ~((FLMUINT)0));

	RCODE pushNamespace(
		FLMUNICODE *		puzPrefix,
		FLMUNICODE *		puzNamespaceURI);

	RCODE pushNamespace(
		F_XMLNamespace *	pNamespace);

	void popNamespaces(
		FLMUINT				uiCount);

	FLMUINT getNamespaceCount( void)
	{
		return( m_uiNamespaceCount);
	}

private:

	F_XMLNamespace *			m_pFirstNamespace;
	FLMUINT						m_uiNamespaceCount;
};

/****************************************************************************
Desc:
****************************************************************************/
class F_XMLParser : public F_XML, public F_XMLNamespaceMgr
{
public:

	F_XMLParser();

	~F_XMLParser();

	RCODE FLMAPI setup( void);

	void FLMAPI reset( void);

	RCODE FLMAPI import(
		IF_IStream *				pStream,
		FLMUINT						uiFlags,
		IF_DOMNode *				pNodeToLinkTo,
		eNodeInsertLoc				eInsertLoc,
		IF_DOMNode **				ppNewNode,
		FLM_IMPORT_STATS *		pImportStats);

	FINLINE void FLMAPI setStatusCallback(
		XML_STATUS_HOOK			fnStatus,
		void *						pvUserData)
	{
		m_fnStatus = fnStatus;
		m_pvCallbackData = pvUserData;
	}

private:

	#define F_DEFAULT_NS_DECL		0x01
	#define F_PREFIXED_NS_DECL		0x02

	typedef struct xmlattr
	{
		FLMUINT				uiLineNum;
		FLMUINT				uiLineOffset;
		FLMUINT				uiLineFilePos;	
		FLMUINT				uiLineBytes;
		FLMUINT				uiValueLineNum;
		FLMUINT				uiValueLineOffset;
		FLMUNICODE *		puzPrefix;
		FLMUNICODE *		puzLocalName;
		FLMUNICODE *		puzVal;
		FLMUINT				uiFlags;
		xmlattr *			pPrev;
		xmlattr *			pNext;
	} XML_ATTR;

	// Methods

	RCODE getFieldTagAndType(
		FLMUNICODE *	puzName,
		FLMBOOL			bOkToAdd,
		FLMUINT *		puiTagNum,
		FLMUINT *		puiDataType);

	RCODE getByte(
		FLMBYTE *	pucByte);
		
	FINLINE void ungetByte(
		FLMBYTE	ucByte)
	{
		// Can only unget a single byte.
		
		flmAssert( !m_ucUngetByte);
		m_ucUngetByte = ucByte;
		m_importStats.uiChars--;
	}
		
	RCODE getLine( void);
	
	FINLINE FLMUNICODE getChar( void)
	{
		if (m_uiCurrLineOffset == m_uiCurrLineNumChars)
		{
			return( (FLMUNICODE)0);
		}
		else
		{
			return( m_puzCurrLineBuf [m_uiCurrLineOffset++]);
		}
	}
	
	FINLINE FLMUNICODE peekChar( void)
	{
		if (m_uiCurrLineOffset == m_uiCurrLineNumChars)
		{
			return( (FLMUNICODE)0);
		}
		else
		{
			return( m_puzCurrLineBuf [m_uiCurrLineOffset]);
		}
	}
	
	FINLINE void ungetChar( void)
	{
		flmAssert( m_uiCurrLineOffset);
		m_uiCurrLineOffset--;
	}
	
	RCODE getName(
		FLMUINT *		puiChars);

	RCODE getQualifiedName(
		FLMUINT *		puiChars,
		FLMUNICODE **	ppuzPrefix,
		FLMUNICODE **	ppuzLocal,
		FLMBOOL *		pbNamespaceDecl,
		FLMBOOL *		pbDefaultNamespaceDecl);

	void getNmtoken(
		FLMUINT *		puiChars);

	RCODE getPubidLiteral( void);

	RCODE getSystemLiteral( void);

	RCODE getElementValue(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiMaxChars,
		FLMBOOL *		pbEntity);

	RCODE processEntityValue( void);

	RCODE getEntity(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiChars,
		FLMBOOL *		pbTranslated,
		FLMUNICODE *	puTransChar);

	RCODE processReference(
		FLMUNICODE *	puChar = NULL);

	RCODE processCDATA(
		IF_DOMNode *	pParent,
		FLMUINT			uiSavedLineNum,
		FLMUINT     	uiSavedOffset,
		FLMUINT			uiSavedFilePos,
		FLMUINT			uiSavedLineBytes);

	RCODE processAttributeList( void);

	RCODE processComment(
		IF_DOMNode *	pParent,
		FLMUINT			uiSavedLineNum,
		FLMUINT     	uiSavedOffset,
		FLMUINT			uiSavedFilePos,
		FLMUINT			uiSavedLineBytes);

	RCODE processProlog( void);

	RCODE processXMLDecl( void);

	RCODE processVersion( void);

	RCODE processEncodingDecl( void);

	RCODE processSDDecl( void);

	RCODE processMisc( void);

	RCODE processDocTypeDecl( void);

	RCODE processPI(
		IF_DOMNode *	pParent,
		FLMUINT			uiSavedLineNum,
		FLMUINT     	uiSavedOffset,
		FLMUINT			uiSavedFilePos,
		FLMUINT			uiSavedLineBytes);

	RCODE processElement(
		IF_DOMNode *		pNodeToLinkTo,
		eNodeInsertLoc		eInsertLoc,
		IF_DOMNode **		ppNewNode);

	RCODE unicodeToNumber64(
		FLMUNICODE *		puzVal,
		FLMUINT64 *			pui64Val,
		FLMBOOL *			pbNeg);

	RCODE flushElementValue(
		IF_DOMNode *		pParent,
		FLMBYTE *			pucValue,
		FLMUINT				uiValueLen);

	RCODE getBinaryVal(
		FLMUINT *			puiLength);

	RCODE fixNamingTag(
		IF_DOMNode *		pNode);

	FLMBOOL lineHasToken(
		const char *	pszToken);
		
	RCODE processMarkupDecl( void);

	RCODE processPERef( void);

	RCODE processElementDecl( void);

	RCODE processEntityDecl( void);

	RCODE processNotationDecl( void);

	RCODE processAttListDecl( void);

	RCODE processContentSpec( void);

	RCODE processMixedContent( void);

	RCODE processChildContent( void);

	RCODE processAttDef( void);

	RCODE processAttType( void);

	RCODE processAttValue(
		XML_ATTR *	pAttr);

	RCODE processDefaultDecl( void);

	RCODE processID(
		FLMBOOL	bPublicId);

	RCODE processSTag(
		IF_DOMNode *		pNodeToLinkTo,	
		eNodeInsertLoc		eInsertLoc,
		FLMBOOL *			pbHasContent,
		IF_DOMNode **		ppElement);

	RCODE skipWhitespace(
		FLMBOOL	bRequired);

	RCODE resizeValBuffer(
		FLMUINT			uiSize);

	// Attribute management

	void resetAttrList( void)
	{
		m_pFirstAttr = NULL;
		m_pLastAttr = NULL;
		m_pAttrPool->poolReset( NULL);
	}

	RCODE allocAttribute(
		XML_ATTR **		ppAttr)
	{
		XML_ATTR *	pAttr = NULL;
		RCODE			rc = NE_FLM_OK;

		if( RC_BAD( rc = m_pAttrPool->poolCalloc( 
			sizeof( XML_ATTR), (void **)&pAttr)))
		{
			goto Exit;
		}

		if( (pAttr->pPrev = m_pLastAttr) == NULL)
		{
			m_pFirstAttr = pAttr;
		}
		else
		{
			m_pLastAttr->pNext = pAttr;
		}

		m_pLastAttr = pAttr;

	Exit:

		*ppAttr = pAttr;
		return( rc);
	}

	RCODE setPrefix(
		XML_ATTR *		pAttr,
		FLMUNICODE *	puzPrefix)
	{
		RCODE		rc = NE_FLM_OK;
		FLMUINT	uiStrLen;

		if( !puzPrefix)
		{
			pAttr->puzPrefix = NULL;
			goto Exit;
		}

		uiStrLen = f_unilen( puzPrefix);

		if( RC_BAD( rc = m_pAttrPool->poolAlloc( 
			sizeof( FLMUNICODE) * (uiStrLen + 1), (void **)&pAttr->puzPrefix)))
		{
			goto Exit;
		}

		f_memcpy( pAttr->puzPrefix, puzPrefix, 
			sizeof( FLMUNICODE) * (uiStrLen + 1));

	Exit:

		return( rc);
	}

	RCODE setLocalName(
		XML_ATTR *		pAttr,
		FLMUNICODE *	puzLocalName)
	{
		RCODE		rc = NE_FLM_OK;
		FLMUINT	uiStrLen;

		if( !puzLocalName)
		{
			pAttr->puzLocalName = NULL;
			goto Exit;
		}

		uiStrLen = f_unilen( puzLocalName);

		if( RC_BAD( rc = m_pAttrPool->poolAlloc( 
			sizeof( FLMUNICODE) * (uiStrLen + 1), 
			(void **)&pAttr->puzLocalName)))
		{
			goto Exit;
		}

		f_memcpy( pAttr->puzLocalName, puzLocalName,
			sizeof( FLMUNICODE) * (uiStrLen + 1));

	Exit:

		return( rc);
	}

	RCODE setUnicode(
		XML_ATTR	*		pAttr,
		FLMUNICODE *	puzUnicode)
	{
		RCODE		rc = NE_FLM_OK;
		FLMUINT	uiStrLen;

		if( !puzUnicode)
		{
			pAttr->puzVal = NULL;
			goto Exit;
		}

		uiStrLen = f_unilen( puzUnicode);

		if( RC_BAD( rc = m_pAttrPool->poolAlloc( 
			sizeof( FLMUNICODE) * (uiStrLen + 1), 
			(void **)&pAttr->puzVal)))
		{
			goto Exit;
		}

		f_memcpy( pAttr->puzVal, puzUnicode, 
			sizeof( FLMUNICODE) * (uiStrLen + 1));

	Exit:

		return( rc);
	}

	RCODE addAttributesToElement(
		IF_DOMNode *		pElement);
		
	FINLINE void setErrInfo(
		FLMUINT			uiErrLineNum,
		FLMUINT			uiErrLineOffset,
		XMLParseError	eErrorType,
		FLMUINT			uiErrLineFilePos,
		FLMUINT			uiErrLineBytes)
	{
		m_importStats.uiErrLineNum = uiErrLineNum;
		m_importStats.uiErrLineOffset = uiErrLineOffset;
		m_importStats.eErrorType = eErrorType;
		m_importStats.uiErrLineFilePos = uiErrLineFilePos;
		m_importStats.uiErrLineBytes = uiErrLineBytes;
	}

	FLMBYTE						m_ucUngetByte;
	FLMUNICODE *				m_puzCurrLineBuf;
	FLMUINT						m_uiCurrLineBufMaxChars;
	FLMUINT						m_uiCurrLineNumChars;
	FLMUINT						m_uiCurrLineOffset;
	FLMUINT						m_uiCurrLineNum;
	FLMUINT						m_uiCurrLineFilePos;
	FLMUINT						m_uiCurrLineBytes;
#define FLM_XML_MAX_CHARS		128
	FLMUNICODE					m_uChars[ FLM_XML_MAX_CHARS];
	FLMBOOL						m_bSetup;
	IF_IStream *				m_pStream;
	FLMBYTE *					m_pucValBuf;
	FLMUINT						m_uiValBufSize; // Number of Unicode characters
	FLMUINT						m_uiFlags;
	FLMBOOL						m_bExtendDictionary;
	XMLEncoding					m_eXMLEncoding;
	XML_STATUS_HOOK			m_fnStatus;
	void *						m_pvCallbackData;
	FLM_IMPORT_STATS			m_importStats;
	XML_ATTR *					m_pFirstAttr;
	XML_ATTR *					m_pLastAttr;
	IF_Pool *					m_pTmpPool;
	IF_Pool * 					m_pAttrPool;
};

#define FLM_XML_EXTEND_DICT_FLAG				0x00000001
#define FLM_XML_COMPRESS_WHITESPACE_FLAG	0x00000002
#define FLM_XML_TRANSLATE_ESC_FLAG			0x00000004

FINLINE FLMBOOL isXMLNS(
	FLMUNICODE *	puzName)
{
	return( (puzName [0] == FLM_UNICODE_x || puzName [0] == FLM_UNICODE_X) &&
			  (puzName [1] == FLM_UNICODE_m || puzName [1] == FLM_UNICODE_M) &&
			  (puzName [2] == FLM_UNICODE_l || puzName [2] == FLM_UNICODE_L) &&
			  (puzName [3] == FLM_UNICODE_n || puzName [3] == FLM_UNICODE_N) &&
			  (puzName [4] == FLM_UNICODE_s || puzName [4] == FLM_UNICODE_S)
			  ? TRUE
			  : FALSE);
}
		
/****************************************************************************
Desc:
****************************************************************************/
F_XML::F_XML()
{
	m_pCharTable = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_XML::~F_XML()
{
	if( m_pCharTable)
	{
		f_free( &m_pCharTable);
	}
}

/****************************************************************************
Desc: Sets a character's type flag in the character lookup table
****************************************************************************/
void F_XML::setCharFlag(
	FLMUNICODE		uLowChar,
	FLMUNICODE		uHighChar,
	FLMUINT16		ui16Flag)
{
	FLMUINT		uiLoop;

	flmAssert( uLowChar <= uHighChar);

	for( uiLoop = (FLMUINT)uLowChar; uiLoop <= (FLMUINT)uHighChar; uiLoop++)
	{
		m_pCharTable[ uiLoop].ucFlags |= (FLMBYTE)ui16Flag;
	}
}

/****************************************************************************
Desc: Builds a character lookup table
****************************************************************************/
RCODE FLMAPI F_XML::setup( void)
{
	RCODE			rc = NE_FLM_OK;

	if( m_pCharTable)
	{
		f_free( &m_pCharTable);
	}

	if( RC_BAD( rc = f_calloc( sizeof( XMLCHAR) * 0xFFFF, &m_pCharTable)))
	{
		goto Exit;
	}

   for (FLMUINT uiLoop = 0; charTbl[uiLoop].ui16Flag; uiLoop++)
   {
      setCharFlag( charTbl[uiLoop].uLowChar,
                charTbl[uiLoop].uHighChar,
                charTbl[uiLoop].ui16Flag);
   }

Exit:

	return( rc);
}

/****************************************************************************
Desc: Returns TRUE if the character is a valid XML PubID character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isPubidChar(
	FLMUNICODE		uChar)
{
	if( uChar == FLM_UNICODE_SPACE ||
		uChar == FLM_UNICODE_LINEFEED ||
		(uChar >= FLM_UNICODE_a && uChar <= FLM_UNICODE_z) ||
		(uChar >= FLM_UNICODE_A && uChar <= FLM_UNICODE_Z) ||
		(uChar >= FLM_UNICODE_0 && uChar <= FLM_UNICODE_9) ||
		uChar == FLM_UNICODE_HYPHEN ||
		uChar == FLM_UNICODE_APOS ||
		uChar == FLM_UNICODE_LPAREN ||
		uChar == FLM_UNICODE_RPAREN ||
		uChar == FLM_UNICODE_PLUS ||
		uChar == FLM_UNICODE_COMMA ||
		uChar == FLM_UNICODE_PERIOD ||
		uChar == FLM_UNICODE_FSLASH ||
		uChar == FLM_UNICODE_COLON ||
		uChar == FLM_UNICODE_EQ ||
		uChar == FLM_UNICODE_QUEST ||
		uChar == FLM_UNICODE_SEMI ||
		uChar == FLM_UNICODE_BANG ||
		uChar == FLM_UNICODE_ASTERISK ||
		uChar == FLM_UNICODE_POUND ||
		uChar == FLM_UNICODE_ATSIGN ||
		uChar == FLM_UNICODE_DOLLAR ||
		uChar == FLM_UNICODE_UNDERSCORE ||
		uChar == FLM_UNICODE_PERCENT)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a single or double quote character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isQuoteChar(
	FLMUNICODE		uChar)
{
	if( uChar == FLM_UNICODE_QUOTE || uChar == FLM_UNICODE_APOS)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a whitespace character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isWhitespace(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_WHITESPACE) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is an extender character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isExtender(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_EXTENDER) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a combining character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isCombiningChar(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_COMBINING_CHAR) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a valid XML naming character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isNCNameChar(
	FLMUNICODE		uChar)
{
	if( isLetter( uChar) ||
		isDigit( uChar) ||
		uChar == FLM_UNICODE_PERIOD ||
		uChar == FLM_UNICODE_HYPHEN ||
		uChar == FLM_UNICODE_UNDERSCORE ||
		isCombiningChar( uChar) || isExtender( uChar))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a valid XML naming character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isNameChar(
	FLMUNICODE		uChar)
{
	if( isNCNameChar( uChar) ||
		uChar == FLM_UNICODE_COLON)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is an ideographic character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isIdeographic(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_IDEOGRAPHIC) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a base character
****************************************************************************/
FLMBOOL FLMAPI F_XML::isBaseChar(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_BASE_CHAR) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a digit
****************************************************************************/
FLMBOOL FLMAPI F_XML::isDigit(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_DIGIT) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a letter
****************************************************************************/
FLMBOOL FLMAPI F_XML::isLetter(
	FLMUNICODE		uChar)
{
	if( isBaseChar( uChar) || isIdeographic( uChar))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc: 	Returns TRUE if the name is a valid XML name
****************************************************************************/
FLMBOOL FLMAPI F_XML::isNameValid(
	FLMUNICODE *	puzName,
	FLMBYTE *		pszName)
{
	FLMBOOL			bValid = FALSE;

	if( puzName)
	{
		FLMUNICODE *	puzTmp;

		if( !isLetter( *puzName) && *puzName != FLM_UNICODE_UNDERSCORE &&
			*puzName != FLM_UNICODE_COLON)
		{
			goto Exit;
		}

		puzTmp = &puzName[ 1];
		while( *puzTmp)
		{
			if( !isNameChar( *puzTmp))
			{
				goto Exit;
			}
			puzTmp++;
		}
	}

	if( pszName)
	{
		FLMBYTE *	pszTmp;

		if( !isLetter( *pszName) && *pszName != FLM_UNICODE_UNDERSCORE &&
			*pszName != FLM_UNICODE_COLON)
		{
			goto Exit;
		}

		pszTmp = &pszName[ 1];
		while( *pszTmp)
		{
			if( !isNameChar( *pszTmp))
			{
				goto Exit;
			}
			pszTmp++;
		}
	}

	bValid = TRUE;

Exit:

	return( bValid);
}

/****************************************************************************
Desc:		Constructor
****************************************************************************/
F_XMLParser::F_XMLParser()
{
	m_uiValBufSize = 0;
	m_pucValBuf = NULL;
	m_bSetup = FALSE;
	m_fnStatus = NULL;
	m_pvCallbackData = NULL;
	m_puzCurrLineBuf = NULL;
	m_uiCurrLineBufMaxChars = 0;
	m_pTmpPool = NULL;
	m_pAttrPool = NULL;
	
	reset();
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_XMLParser::~F_XMLParser()
{
	reset();

	if( m_pucValBuf)
	{
		f_free( &m_pucValBuf);
	}
	
	if (m_puzCurrLineBuf)
	{
		f_free( &m_puzCurrLineBuf);
	}
	
	if( m_pTmpPool)
	{
		m_pTmpPool->Release();
	}
	
	if( m_pAttrPool)
	{
		m_pAttrPool->Release();
	}
}

/****************************************************************************
Desc:		Resets member variables so the object can be reused
****************************************************************************/
void FLMAPI F_XMLParser::reset( void)
{
	m_uiCurrLineNum = 0;
	m_uiCurrLineNumChars = 0;
	m_uiCurrLineOffset = 0;
	m_ucUngetByte = 0;
	m_uiCurrLineFilePos = 0;
	m_uiCurrLineBytes = 0;
	m_pStream = NULL;
	m_uiFlags = 0;
	m_eXMLEncoding = FLM_XML_USASCII_ENCODING;
	f_memset( &m_importStats, 0, sizeof( FLM_IMPORT_STATS));

	popNamespaces( getNamespaceCount());
	m_pTmpPool->poolReset( NULL);
	resetAttrList();
}

/****************************************************************************
Desc: 	Initializes the object (allocates buffers, etc.)
****************************************************************************/
RCODE FLMAPI F_XMLParser::setup( void)
{
	RCODE		rc = NE_FLM_OK;

	flmAssert( !m_bSetup);

	if( RC_BAD( rc = resizeValBuffer( 2048)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = F_XML::setup()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = FlmAllocPool( &m_pTmpPool)))
	{
		goto Exit;
	}
	
	m_pTmpPool->poolInit( 4096);
	
	if( RC_BAD( rc = FlmAllocPool( &m_pAttrPool)))
	{
		goto Exit;
	}
	
	m_pAttrPool->poolInit( 4096);
	m_bSetup = TRUE;

Exit:

	if( RC_BAD( rc))
	{
		if( m_pucValBuf)
		{
			f_free( &m_pucValBuf);
			m_pucValBuf = NULL;
		}
	}

	return( rc);
}

/****************************************************************************
Desc: 	Reads data from the input stream and builds a FLAIM record
****************************************************************************/
RCODE FLMAPI F_XMLParser::import(
	IF_IStream *			pStream,
	FLMUINT					uiFlags,
	IF_DOMNode *			pNodeToLinkTo,
	eNodeInsertLoc			eInsertLoc,
	IF_DOMNode **			ppNewNode,
	FLM_IMPORT_STATS *	pImportStats)
{
	RCODE				rc = NE_FLM_OK;

	// Reset the state of the parser

	reset();

	// Set up namespace support.  Un-prefixed names (NULL prefix) are
	// not bound to a namespace (NULL URI).  The 'xml' namespace prefix
	// is, by definition, bound to 'http://www.w3.org/XML/1998/namespace'

	if( RC_BAD( rc = pushNamespace( NULL, NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pushNamespace(
		gv_puzXMLPrefix, gv_puzXMLNSURI)))
	{
		goto Exit;
	}

	m_pStream = pStream;
	m_uiFlags = uiFlags;

	if( RC_BAD( rc = processProlog()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processElement( pNodeToLinkTo, eInsertLoc, ppNewNode)))
	{
		goto Exit;
	}

	// Call the status hook one last time

	m_importStats.uiDocuments++;
	if( m_fnStatus)
	{
		m_fnStatus( XML_STATS,
			(void *)&m_importStats, NULL, NULL, m_pvCallbackData);
	}

	// Tally and return the import stats

	if( pImportStats)
	{
		pImportStats->uiChars += m_importStats.uiChars;
		pImportStats->uiAttributes += m_importStats.uiAttributes;
		pImportStats->uiElements += m_importStats.uiElements;
		pImportStats->uiText += m_importStats.uiText;
		pImportStats->uiDocuments += m_importStats.uiDocuments;
	}

Exit:

	if( RC_BAD( rc) && pImportStats)
	{
		pImportStats->uiErrLineNum = m_importStats.uiErrLineNum
			? m_importStats.uiErrLineNum
			: m_uiCurrLineNum;

		pImportStats->uiErrLineOffset = m_importStats.uiErrLineOffset
			? m_importStats.uiErrLineOffset
			: m_uiCurrLineOffset;

		pImportStats->eErrorType = ( XMLParseError)( m_importStats.eErrorType);
		
		pImportStats->uiErrLineFilePos = m_importStats.uiErrLineFilePos;
		pImportStats->uiErrLineBytes = m_importStats.uiErrLineBytes;
		pImportStats->eXMLEncoding = m_importStats.eXMLEncoding;
	}

	return( rc);
}

/****************************************************************************
Desc: Process an XML prolog
****************************************************************************/
RCODE F_XMLParser::processProlog( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if (lineHasToken( "<?xml"))
	{
		if( RC_BAD( rc = processXMLDecl()))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = processMisc()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if (lineHasToken( "<!DOCTYPE"))
	{
		if( RC_BAD( rc = processDocTypeDecl()))
		{
			goto Exit;
		}

		if( RC_BAD( rc = processMisc()))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Converts a Unicode string to a number
****************************************************************************/
RCODE F_XMLParser::unicodeToNumber64(
	FLMUNICODE *		puzVal,
	FLMUINT64 *			pui64Val,
	FLMBOOL *			pbNeg)
{
	RCODE				rc = NE_FLM_OK;
	char				szTmpBuf[ 64];
	FLMUINT			uiLoop;
	FLMBOOL			bNeg = FALSE;
	FLMUNICODE		uChar;

	if( !puzVal)
	{
		*pui64Val = 0;
		*pbNeg = FALSE;
		goto Exit;
	}

	for( uiLoop = 0; uiLoop < sizeof( szTmpBuf); uiLoop++)
	{
		if( (uChar = puzVal[ uiLoop]) == 0)
		{
			break;
		}
		else if( uiLoop == 0 && uChar == FLM_UNICODE_HYPHEN)
		{
			bNeg = TRUE;
			continue;
		}

		szTmpBuf[ uiLoop] = (char)uChar;
	}

	if( uiLoop == sizeof( szTmpBuf))
	{
		rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	szTmpBuf[ uiLoop] = 0;
	*pui64Val = f_atou64( szTmpBuf);

	if( pbNeg)
	{
		*pbNeg = bNeg;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Creates a new text node and links it to the passed-in parent
****************************************************************************/
RCODE F_XMLParser::flushElementValue(
	IF_DOMNode *		pParent,
	FLMBYTE *			pucValue,
	FLMUINT				uiValueLen)
{
	RCODE					rc = NE_FLM_OK;
	IF_DOMNode *		pData = NULL;
	FLMUNICODE *		puzTextStart = (FLMUNICODE *)pucValue;
	eFlmDataType		parentDataType;

	if( !uiValueLen)
	{
		flmAssert( 0);
		goto Exit;
	}

//	if( RC_BAD( rc = pParent->createNode( m_pDb, DATA_NODE, 0,
//		FLM_LAST_CHILD, &pData)))
//	{
//		setErrInfo( m_uiCurrLineNum,
//				m_uiCurrLineOffset,
//				XML_ERR_CREATING_DATA_NODE,
//				m_uiCurrLineFilePos,
//				m_uiCurrLineBytes);
//		goto Exit;
//	}

	if( RC_BAD( rc = pParent->getDataType( &parentDataType)))
	{
		goto Exit;
	}

	switch( parentDataType)
	{
		case FLM_TEXT_TYPE:
		{
//			if( RC_BAD( rc = pData->setUnicode( m_pDb, puzTextStart)))
//			{
//				goto Exit;
//			}

			m_importStats.uiText++;
			if( m_fnStatus && (m_importStats.uiText % 50) == 0)
			{
				m_fnStatus( XML_STATS,
					(void *)&m_importStats, NULL, NULL, m_pvCallbackData);
			}

			break;
		}

		case FLM_NUMBER_TYPE:
		{
			FLMUINT64		ui64Val;
			FLMBOOL			bNeg;

			if( RC_BAD( rc = unicodeToNumber64( puzTextStart, &ui64Val, &bNeg)))
			{
				goto Exit;
			}

			if( !bNeg)
			{
//				if( RC_BAD( rc = pData->setUINT64( m_pDb, ui64Val)))
//				{
//					goto Exit;
//				}
			}
			else
			{
//				if( RC_BAD( rc = pData->setINT64( m_pDb, -((FLMINT64)ui64Val))))
//				{
//					goto Exit;
//				}
			}
			break;
		}

		case FLM_BINARY_TYPE:
		{
//			if( RC_BAD( rc = pData->setBinary( m_pDb, pucValue, uiValueLen)))
//			{
//				goto Exit;
//			}
			break;
		}

		default:
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
			goto Exit;
		}
	}

Exit:

	if( pData)
	{
		pData->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML element
****************************************************************************/
RCODE F_XMLParser::processElement(
	IF_DOMNode *		pNodeToLinkTo,
	eNodeInsertLoc		eInsertLoc,
	IF_DOMNode **		ppNewNode)
{
	RCODE							rc = NE_FLM_OK;
	FLMBOOL						bHasContent;
	FLMBOOL						bFlushedValue = FALSE;
	FLMUINT						uiChars;
	FLMUINT						uiOffset = 0;
	FLMUNICODE					uChar;
	IF_DOMNode *				pElement = NULL;
	F_XMLNamespace *			pNamespace = NULL;
	FLMUNICODE *				puzPrefix;
	FLMUNICODE *				puzLocal;
	FLMUINT						uiStartNSCount = getNamespaceCount();
	FLMUINT						uiTmp;
	FLMUINT						uiWhitespaceStartOffset = 0;
	FLMBOOL						bNamespaceDecl;
	FLMUINT						uiSavedLineNum =  0;
	FLMUINT                 uiSavedOffset = 0;
	FLMUINT						uiSavedFilePos = 0;
	FLMUINT						uiSavedLineBytes = 0;
	eFlmDataType				elmDataType;
	FLMUINT						uiElmNameId;

	if( RC_BAD( rc = processSTag( pNodeToLinkTo, eInsertLoc, 
		&bHasContent, &pElement)))
	{
		goto Exit;
	}
	
	if (ppNewNode)
	{
		*ppNewNode = pElement;
		(*ppNewNode)->AddRef();
	}

	if( !bHasContent)
	{
		goto Exit;
	}

	if( RC_BAD( rc = pElement->getNameId( &uiElmNameId)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pElement->getDataType( &elmDataType)))
	{
	 goto Exit;
	}
					
	for( ;;)
	{
		if ((uChar = getChar()) == 0)
		{
			uChar = ASCII_NEWLINE;
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
		}
		
		if( uChar == FLM_UNICODE_LT)
		{
			if( uiWhitespaceStartOffset)
			{
				// Set the offset to where the whitespace would
				// have started.
				
				flmAssert( uiWhitespaceStartOffset <= uiOffset);
				uiOffset = uiWhitespaceStartOffset;
				uiWhitespaceStartOffset = 0;
			}

			if( uiOffset)
			{
				// Flush the value

				if( pElement)
				{
					if( elmDataType == FLM_TEXT_TYPE ||
						 elmDataType == FLM_NUMBER_TYPE)
					{
						if( uiOffset + 1 >= m_uiValBufSize)
						{
							if( RC_BAD( rc = resizeValBuffer( uiOffset + 2)))
							{
								goto Exit;
							}
						}

						m_pucValBuf[ uiOffset] = 0;
						m_pucValBuf[ uiOffset + 1] = 0;
					}

					if( RC_BAD( rc = flushElementValue( 
						pElement, m_pucValBuf, uiOffset)))
					{
						goto Exit;
					}
				}

				bFlushedValue = TRUE;
				uiOffset = 0;
			}

			// Preserve start location for error handling if necessary

			uiSavedLineNum = m_uiCurrLineNum;
			uiSavedOffset = m_uiCurrLineOffset;
			uiSavedFilePos = m_uiCurrLineFilePos;
			uiSavedLineBytes = m_uiCurrLineBytes;

			if (lineHasToken( "?"))
			{
				if( RC_BAD( rc = processPI( pElement,
											uiSavedLineNum,
											uiSavedOffset,
											uiSavedFilePos,
											uiSavedLineBytes)))
				{
					goto Exit;
				}
			}
			else if (lineHasToken( "!--"))
			{
				if( RC_BAD( rc = processComment( pElement,
											uiSavedLineNum,
											uiSavedOffset,
											uiSavedFilePos,
											uiSavedLineBytes)))
				{
					goto Exit;
				}
			}
			else if (lineHasToken( "![CDATA["))
			{
				if( RC_BAD( rc = processCDATA( pElement,
											uiSavedLineNum,
											uiSavedOffset,
											uiSavedFilePos,
											uiSavedLineBytes)))
				{
					goto Exit;
				}
			}
			else if (lineHasToken( "/"))
			{
				break;
			}
			else if( isNameChar( peekChar()))
			{
				
				// Unget the "<" - because processElement expect to see
				// "<elementname"
				
				ungetChar();
				if( RC_BAD( rc = processElement( pElement, FLM_LAST_CHILD, NULL)))
				{
					goto Exit;
				}
			}
			else
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						XML_ERR_BAD_ELEMENT_NAME,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
		}
		else if( elmDataType == FLM_BINARY_TYPE)
		{
			ungetChar();

			if( RC_BAD( rc = getBinaryVal( &uiOffset)))
			{
				goto Exit;
			}
		}
		else if (uChar == FLM_UNICODE_AMP)
		{
			if( RC_BAD( rc = processReference( &uChar)))
			{
				goto Exit;
			}
			
			flmAssert( uChar);
			if (elmDataType != FLM_NODATA_TYPE)
			{
				*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
				uiOffset += sizeof( FLMUNICODE);
				uiWhitespaceStartOffset = 0;

				if( uiOffset >= m_uiValBufSize)
				{
					if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
					{
						goto Exit;
					}
				}
			}
		}
		else
		{
			if( elmDataType != FLM_NODATA_TYPE)
			{
				if( m_uiFlags & FLM_XML_COMPRESS_WHITESPACE_FLAG)
				{
					if( isWhitespace( uChar))
					{
						// If uiOffset is zero, this is still leading
						// white space, and we should ignore it.
						// Otherwise, we need to keep track of where
						// whitespace began.
						
						if( !uiOffset)
						{
							uChar = 0;
						}
						else
						{
							uiWhitespaceStartOffset = uiOffset;
						}
					}
					else
					{
						
						// Last character is not whitespace.
						
						uiWhitespaceStartOffset = 0;
					}
				}

				if( uChar)
				{
					*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
					uiOffset += sizeof( FLMUNICODE);
					if( uiOffset >= m_uiValBufSize)
					{
						if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
						{
							goto Exit;
						}
					}
				}
			}
		}
	}

	flmAssert( !uiOffset);


	uiSavedOffset = m_uiCurrLineOffset;
	if( RC_BAD( rc = getQualifiedName( &uiChars,
		&puzPrefix, &puzLocal, &bNamespaceDecl, NULL)))
	{
		goto Exit;
	}

	// Validate that the end tag matches the start tag

	if( pElement)
	{
		
		// Element names cannot be "xmlns" or begin with "xmlns:"
		
		if (bNamespaceDecl)
		{
			setErrInfo( m_uiCurrLineNum,
					 uiSavedOffset,
					XML_ERR_XMLNS_IN_ELEMENT_NAME,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
		if( puzPrefix)
		{
			FLMUINT		uiPrefixId1 = 0;
			FLMUINT		uiPrefixId2 = 0;

//			if( RC_BAD( rc = m_pDb->m_pDict->getPrefixId(
//				m_pDb, puzPrefix, &uiPrefixId1)))
//			{
//				goto Exit;
//			}

//			if( RC_BAD( rc = pElement->getPrefixId( m_pDb, &uiPrefixId2)))
//			{
//				goto Exit;
//			}

			if( uiPrefixId1 != uiPrefixId2)
			{
				setErrInfo( m_uiCurrLineNum,
						uiSavedOffset,
						XML_ERR_ELEMENT_NAME_MISMATCH,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = pElement->getPrefixId( &uiTmp)))
			{
				goto Exit;
			}

			if( uiTmp)
			{
				setErrInfo( m_uiCurrLineNum,
						uiSavedOffset,
						XML_ERR_ELEMENT_NAME_MISMATCH,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
		}

		if( RC_BAD( rc = findNamespace( puzPrefix, &pNamespace)))
		{
			if( rc == NE_FLM_NOT_FOUND)
			{
				setErrInfo( m_uiCurrLineNum, 
						uiSavedOffset,
						XML_ERR_PREFIX_NOT_DEFINED,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
			}
			goto Exit;
		}

//		if( RC_BAD( rc = m_pDb->getElementNameId(
//			pNamespace->getURIPtr(), puzLocal, &uiTmp)))
//		{
//			if( rc == NE_FLM_NOT_FOUND)
//			{
//				setErrInfo( m_uiCurrLineNum,
//						uiSavedOffset,
//						XML_ERR_ELEMENT_NAME_MISMATCH,
//						m_uiCurrLineFilePos,
//						m_uiCurrLineBytes);
//				rc = RC_SET( NE_FLM_INVALID_XML);
//			}
//			goto Exit;
//		}

		if( uiElmNameId != uiTmp)
		{
			setErrInfo( m_uiCurrLineNum,
						uiSavedOffset,
						XML_ERR_ELEMENT_NAME_MISMATCH,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}

	// Skip any whitespace after the name

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	// Get the ending ">"

	if( getChar() != FLM_UNICODE_GT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	if( pNamespace)
	{
		pNamespace->Release();
	}

	popNamespaces( getNamespaceCount() - uiStartNSCount);

	if( pElement)
	{
		pElement->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML STag
****************************************************************************/
RCODE F_XMLParser::processSTag(
	IF_DOMNode *		pNodeToLinkTo,
	eNodeInsertLoc		eInsertLoc,
	FLMBOOL *			pbHasContent,
	IF_DOMNode **		ppElement)
{
	RCODE					rc = NE_FLM_OK;
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	IF_DOMNode *		pElement = NULL;
	F_XMLNamespace *	pNamespace = NULL;
	FLMUNICODE *		puzTmpPrefix;
	FLMUNICODE *		puzPrefix = NULL;
	FLMUNICODE *		puzTmpLocal;
	FLMUNICODE *		puzLocal = NULL;
	FLMUINT				uiNameId = 0;
	FLMUINT				uiAllocSize;
	void *				pvMark = m_pTmpPool->poolMark();
	FLMBOOL				bNamespaceDecl;
	FLMUINT				uiSavedLineNum;
	FLMUINT           uiSavedOffset;
	FLMUINT				uiSavedFilePos;
	FLMUINT				uiSavedLineBytes;

	*pbHasContent = FALSE;

	if( getChar() != FLM_UNICODE_LT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_ELEMENT_LT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	uiSavedLineNum = m_uiCurrLineNum;
	uiSavedOffset = m_uiCurrLineOffset;
	uiSavedFilePos = m_uiCurrLineFilePos;
	uiSavedLineBytes = m_uiCurrLineBytes;
	if( RC_BAD( rc = getQualifiedName( &uiChars, &puzTmpPrefix, &puzTmpLocal,
		&bNamespaceDecl, NULL)))
	{
		goto Exit;
	}

	// Element names cannot be "xmlns" or begin with "xmlns:"

	if (bNamespaceDecl)
	{
		setErrInfo( uiSavedLineNum,
				uiSavedOffset,
				XML_ERR_XMLNS_IN_ELEMENT_NAME,
				uiSavedFilePos,
				uiSavedLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	uiAllocSize = (f_unilen( puzTmpLocal) + 1) * sizeof( FLMUNICODE);
	if( RC_BAD( rc = m_pTmpPool->poolAlloc( uiAllocSize, (void **)&puzLocal)))
	{
		goto Exit;
	}
	f_unicpy( puzLocal, puzTmpLocal);

	if( puzTmpPrefix)
	{

		// Need to save the prefix, because as parsing
		// continues, the scratch buffer will be overwritten

		uiAllocSize = (f_unilen( puzTmpPrefix) + 1) * sizeof( FLMUNICODE);
		if( RC_BAD( rc = m_pTmpPool->poolAlloc( uiAllocSize, (void **)&puzPrefix)))
		{
			goto Exit;
		}
		f_unicpy( puzPrefix, puzTmpPrefix);
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	// Read the attributes

	resetAttrList();

	uChar = peekChar();
	if( uChar != FLM_UNICODE_GT && uChar != FLM_UNICODE_FSLASH)
	{
		if( RC_BAD( rc = processAttributeList()))
		{
			goto Exit;
		}
	}

	// Find or create the element's name ID

	if( RC_BAD( rc = findNamespace( puzPrefix, &pNamespace)))
	{
		if( rc == NE_FLM_NOT_FOUND)
		{
			setErrInfo( uiSavedLineNum,
					uiSavedOffset,
					XML_ERR_PREFIX_NOT_DEFINED,
					uiSavedFilePos,
					uiSavedLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
		}
		goto Exit;
	}

//	if( RC_BAD( rc = m_pDb->getElementNameId(
//		pNamespace->getURIPtr(), puzLocal, &uiNameId)))
//	{
//		if( rc != NE_FLM_NOT_FOUND)
//		{
//			goto Exit;
//		}
//
//		if( !(m_uiFlags & FLM_XML_EXTEND_DICT_FLAG) ||
//			(pNamespace->getURIPtr() && 
//			 f_unicmp( pNamespace->getURIPtr(), gv_uzXFLAIMNamespace) == 0))
//		{
//			rc = RC_SET( NE_FLM_UNDEFINED_ELEMENT_NAME);
//			goto Exit;
//		}
//
//		// Automatically extend the schema
//
//		uiNameId = 0;
//		if( RC_BAD( rc = m_pDb->createElementDef(
//			pNamespace->getURIPtr(),
//			puzLocal, FLM_TEXT_TYPE, &uiNameId)))
//		{
//			goto Exit;
//		}
//	}

	// Create the element node

	if( pNodeToLinkTo)
	{
		if( RC_BAD( rc = pNodeToLinkTo->createNode( ELEMENT_NODE,
			uiNameId, eInsertLoc, &pElement)))
		{
			setErrInfo( uiSavedLineNum,
					uiSavedOffset,
					XML_ERR_CREATING_ELEMENT_NODE,
					uiSavedFilePos,
					uiSavedLineBytes);
			goto Exit;
		}
	}
	else
	{
//		if( RC_BAD( rc = m_pDb->createRootElement( uiNameId, &pElement)))
//		{
//			setErrInfo( uiSavedLineNum,
//					uiSavedOffset,
//					XML_ERR_CREATING_ROOT_ELEMENT,
//					uiSavedFilePos,
//					uiSavedLineBytes);
//			goto Exit;
//		}
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}
	
	// Need to end with ">" or "/>"

	uChar = getChar();
	if( uChar == FLM_UNICODE_GT)
	{
		*pbHasContent = TRUE;
	}
	else if( uChar == FLM_UNICODE_FSLASH)
	{
		if( getChar() != FLM_UNICODE_GT)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_GT,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	// Set the element's prefix

	if( RC_BAD( rc = addAttributesToElement( pElement)))
	{
		goto Exit;
	}

	if( puzPrefix)
	{
		if( RC_BAD( rc = pElement->setPrefix( puzPrefix)))
		{
			goto Exit;
		}
	}

	if( ppElement)
	{
		*ppElement = pElement;
		pElement = NULL;
	}

	m_importStats.uiElements++;
	if( m_fnStatus && (m_importStats.uiElements % 50) == 0)
	{
		m_fnStatus( XML_STATS,
			(void *)&m_importStats, NULL, NULL, m_pvCallbackData);
	}

Exit:

	if( pElement)
	{
		pElement->Release();
	}

	if( pNamespace)
	{
		pNamespace->Release();
	}

	m_pTmpPool->poolReset( pvMark);
	return( rc);
}

/****************************************************************************
Desc:		Processes an element's attributes
****************************************************************************/
RCODE F_XMLParser::processAttributeList( void)
{
	RCODE					rc = NE_FLM_OK;
	FLMUINT				uiChars;
	FLMUNICODE *		puzLocal;
	FLMUNICODE *		puzPrefix;
	XML_ATTR *			pAttr = NULL;
	FLMBOOL				bFoundDefaultNamespace = FALSE;
	FLMUINT				uiNamespaceCount = 0;
	FLMBOOL				bNamespaceDecl;
	FLMBOOL				bDefaultNamespaceDecl;
	FLMUINT				uiSavedLineNum;
	FLMUINT           uiSavedOffset;
	FLMUINT				uiSavedFilePos;
	FLMUINT				uiSavedLineBytes;

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
		if( !isNameChar( peekChar()))
		{
			break;
		}

		uiSavedLineNum = m_uiCurrLineNum;
		uiSavedOffset = m_uiCurrLineOffset;
		uiSavedFilePos = m_uiCurrLineFilePos;
		uiSavedLineBytes = m_uiCurrLineBytes;
		if( RC_BAD( rc = getQualifiedName( &uiChars,
			&puzPrefix, &puzLocal, &bNamespaceDecl,
			&bDefaultNamespaceDecl)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = allocAttribute( &pAttr)))
		{
			goto Exit;
		}
		pAttr->uiLineNum = uiSavedLineNum;
		pAttr->uiLineOffset = uiSavedOffset;
		pAttr->uiLineFilePos = uiSavedFilePos;
		pAttr->uiLineBytes = uiSavedLineBytes;

		if( RC_BAD( rc = setPrefix( pAttr, puzPrefix)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = setLocalName( pAttr, puzLocal)))
		{
			goto Exit;
		}
		if (bNamespaceDecl)
		{
			if (bDefaultNamespaceDecl)
			{
				pAttr->uiFlags |= F_DEFAULT_NS_DECL;
			}
			else
			{
				pAttr->uiFlags |= F_PREFIXED_NS_DECL;
			}
		}

		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
		
		// Attribute name must be followed by an "="

		if( getChar() != FLM_UNICODE_EQ)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_EQ,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}

		pAttr->uiValueLineNum = m_uiCurrLineNum;
		pAttr->uiValueLineOffset = m_uiCurrLineOffset;
		if( RC_BAD( rc = processAttValue( pAttr)))
		{
			goto Exit;
		}

		m_importStats.uiAttributes++;
		if( m_fnStatus && (m_importStats.uiAttributes % 50) == 0)
		{
			m_fnStatus( XML_STATS,
				(void *)&m_importStats, NULL, NULL, m_pvCallbackData);
		}
	}

	// Push any namespace declarations onto the stack

	for( pAttr = m_pFirstAttr; pAttr; pAttr = pAttr->pNext)
	{
		// Duplicate namespace declarations are not allowed within a single element.
		// So, multiple default namespace declarations or multiple uses of the same
		// prefix in will result in a syntax error.

		if( pAttr->uiFlags & F_DEFAULT_NS_DECL)
		{
			// Default namespace declaration

			if( bFoundDefaultNamespace)
			{
				setErrInfo( pAttr->uiLineNum,
						pAttr->uiLineOffset,
						XML_ERR_MULTIPLE_XMLNS_DECLS,
						pAttr->uiLineFilePos,
						pAttr->uiLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}

			if( !pAttr->puzVal || *pAttr->puzVal == 0)
			{
				// No namespace

				if( RC_BAD( rc = pushNamespace( NULL, NULL)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = pushNamespace( NULL, pAttr->puzVal)))
				{
					goto Exit;
				}
			}

			uiNamespaceCount++;
			bFoundDefaultNamespace = TRUE;
		}
		else if( pAttr->uiFlags & F_PREFIXED_NS_DECL)
		{
			// Check for a unique prefix within current element

			if( RC_OK( rc = findNamespace( &pAttr->puzLocalName [6], 
				NULL, uiNamespaceCount)))
			{
				setErrInfo( pAttr->uiLineNum,
						pAttr->uiLineOffset,
						XML_ERR_MULTIPLE_PREFIX_DECLS,
						pAttr->uiLineFilePos,
						pAttr->uiLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
			else if( rc != NE_FLM_NOT_FOUND)
			{
				goto Exit;
			}
			else
			{
				rc = NE_FLM_OK;
			}

			if( RC_BAD( rc = pushNamespace( 
				&pAttr->puzLocalName [6], pAttr->puzVal)))
			{
				goto Exit;
			}

			uiNamespaceCount++;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML declaration
****************************************************************************/
RCODE F_XMLParser::processXMLDecl( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUNICODE	uChar;

	// Have already eaten the "<?xml" - must have whitespace after it
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processVersion()))
	{
		goto Exit;
	}
	
	uChar = peekChar();
	if (!uChar || isWhitespace( uChar))
	{
		if (RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		goto Process_Question_Mark;
	}
	
	if (lineHasToken( "encoding"))
	{
		if( RC_BAD( rc = processEncodingDecl()))
		{
			goto Exit;
		}
		uChar = peekChar();
		if (!uChar || isWhitespace( uChar))
		{
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
		}
		else
		{
			goto Process_Question_Mark;
		}
	}
	
	if (lineHasToken( "standalone"))
	{
		if( RC_BAD( rc = processSDDecl()))
		{
			goto Exit;
		}
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
	}
	
Process_Question_Mark:

	if (!lineHasToken( "?>"))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_EXPECTING_QUEST_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML document type declaration
****************************************************************************/
RCODE F_XMLParser::processDocTypeDecl( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUNICODE	uChar;
	
	// Have already eaten the "<!DOCTYPE"
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}
	
	uChar = peekChar();
	if (!uChar || isWhitespace( uChar))
	{
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
	
		if (lineHasToken( "SYSTEM"))
		{
			if( RC_BAD( rc = processID( FALSE)))
			{
				goto Exit;
			}
			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
		}
		else if (lineHasToken( "PUBLIC"))
		{
			if( RC_BAD( rc = processID( TRUE)))
			{
				goto Exit;
			}
			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
		}
	}
	
	if( peekChar() == FLM_UNICODE_LBRACKET)
	{
		// Eat up the '['
		
		(void)getChar();

		for( ;;)
		{
			uChar = getChar();

			if( uChar == FLM_UNICODE_PERCENT)
			{
				if( RC_BAD( rc = processPERef()))
				{
					goto Exit;
				}
			}
			else if( uChar == FLM_UNICODE_RBRACKET)
			{
				if( RC_BAD( rc = skipWhitespace( FALSE)))
				{
					goto Exit;
				}
				break;
			}
			else if (isWhitespace( uChar))
			{
				if( RC_BAD( rc = skipWhitespace( FALSE)))
				{
					goto Exit;
				}
			}
			else
			{
				ungetChar();
				if( RC_BAD( rc = processMarkupDecl()))
				{
					goto Exit;
				}
			}
		}
	}

	if( getChar() != FLM_UNICODE_GT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	See if the current line has the specified token in it starting
		from the current offset.
****************************************************************************/
FLMBOOL F_XMLParser::lineHasToken(
	const char *	pszToken)
{
	FLMUINT			uiOffset;
	
	uiOffset = m_uiCurrLineOffset;
	while (uiOffset < m_uiCurrLineNumChars)
	{
		if (m_puzCurrLineBuf [uiOffset] != (FLMUNICODE)(*pszToken))
		{
			
			// Do NOT change m_uiCurrLineOffset if we return FALSE.
			
			return( FALSE);
		}
		pszToken++;
		uiOffset++;
		if (*pszToken == 0)
		{
			m_uiCurrLineOffset = uiOffset;
			return( TRUE);
		}
	}
	return( FALSE);
}

/****************************************************************************
Desc:		Processes an XML markup declaration
****************************************************************************/
RCODE F_XMLParser::processMarkupDecl( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if (lineHasToken( "<?"))
	{
		rc = processPI( NULL, 0, 0, 0, 0);
		goto Exit;
	}
	
	if (lineHasToken( "<!--"))
	{
		rc = processComment( NULL, 0, 0, 0, 0);
		goto Exit;
	}
	
	if (lineHasToken( "<!ENTITY"))
	{
		rc = processEntityDecl();
		goto Exit;
	}
	
	if (lineHasToken( "<!ELEMENT"))
	{
		rc = processElementDecl();
		goto Exit;
	}
	
	if (lineHasToken( "<!ATTLIST"))
	{
		rc = processAttListDecl();
		goto Exit;
	}
	
	if (lineHasToken( "<!NOTATION"))
	{
		rc = processNotationDecl();
		goto Exit;
	}
	
	setErrInfo( m_uiCurrLineNum,
			m_uiCurrLineOffset,
			XML_ERR_INVALID_XML_MARKUP,
			m_uiCurrLineFilePos,
			m_uiCurrLineBytes);
	rc = RC_SET( NE_FLM_INVALID_XML);

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML element declaration
****************************************************************************/
RCODE F_XMLParser::processElementDecl( void)
{
	RCODE		rc = NE_FLM_OK;

	// Have already eaten up the "<!ELEMENT" - must be followed by whitespace
	// before the name.
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processContentSpec()))
	{
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if( getChar() != FLM_UNICODE_GT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an attribute list declaration
****************************************************************************/
RCODE F_XMLParser::processAttListDecl( void)
{
	RCODE					rc = NE_FLM_OK;
	FLMUNICODE			uChar;
	FLMUINT				uiAttDefCount = 0;
	FLMUINT				uiAttListLineNum = m_uiCurrLineNum;
	FLMUINT				uiAttListLineOffset = m_uiCurrLineOffset - 9; 
	
	// Have already eaten up the "<!ATTLIST" - must be whitespace before
	// the name.
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}

	for( ;;)
	{
		uChar = peekChar();
		
		if (!uChar || isWhitespace( uChar))
		{
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			if( peekChar() == FLM_UNICODE_GT)
			{
				break;
			}
			if( RC_BAD( rc = processAttDef()))
			{
				goto Exit;
			}
	
			uiAttDefCount++;
		}
		else if (uChar == FLM_UNICODE_GT)
		{
			break;
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					XML_ERR_EXPECTING_GT,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}
	
	uChar = getChar();
	flmAssert( uChar == FLM_UNICODE_GT);
	if( !uiAttDefCount)
	{
		setErrInfo( uiAttListLineNum,
				uiAttListLineOffset,
				XML_ERR_MUST_HAVE_ONE_ATT_DEF,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes and entity declaration
****************************************************************************/
RCODE F_XMLParser::processEntityDecl( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUNICODE	uChar;
	FLMBOOL		bGEDecl = FALSE;

	// Have already eaten up the "<!ENTITY" - must be followed by whitespace
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( peekChar() == FLM_UNICODE_PERCENT)
	{
		uChar = getChar();

		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
	}
	else
	{
		bGEDecl = TRUE;
	}

	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}
	
	// Name must be followed by whitespace

	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	uChar = peekChar();
	if( isQuoteChar( uChar))
	{
		if( RC_BAD( rc = processEntityValue()))
		{
			goto Exit;
		}
	}
	else
	{
		if (lineHasToken( "SYSTEM"))
		{
			if( RC_BAD( rc = processID( TRUE)))
			{
				goto Exit;
			}
		}
		else if (lineHasToken( "PUBLIC"))
		{
			if( RC_BAD( rc = processID( FALSE)))
			{
				goto Exit;
			}
		}
		else
		{
			if (RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}
			goto Process_GT;
		}
		
		if (!isWhitespace( peekChar()))
		{
			goto Process_GT;
		}

		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
		
		if (!bGEDecl)
		{
			goto Process_GT;
		}

		if (!lineHasToken( "NDATA"))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					XML_ERR_EXPECTING_NDATA,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
		
		// Whitespace must be between the NDATA and the name
					
		if( RC_BAD( rc = skipWhitespace( TRUE)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = getName( NULL)))
		{
			goto Exit;
		}
	
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
	}

Process_GT:

	if( getChar() != FLM_UNICODE_GT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML ID
****************************************************************************/
RCODE F_XMLParser::processID(
	FLMBOOL	bPublicId)
{
	RCODE		rc = NE_FLM_OK;

	// Have already eaten the "SYSTEM" or "PUBLIC" token
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}
	
	if (bPublicId)
	{
	
		// Public ID
		
		if( RC_BAD( rc = getPubidLiteral()))
		{
			goto Exit;
		}
		
		// Must be whitespace if it wasn't a '>'
		
		if( RC_BAD( rc = skipWhitespace( TRUE)))
		{
			goto Exit;
		}
	}
	
	// Get the system ID
	
	if (RC_BAD( rc = getSystemLiteral()))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a notation declaration
****************************************************************************/
RCODE F_XMLParser::processNotationDecl( void)
{
	RCODE		rc = NE_FLM_OK;

	// Have already eaten up the "<!NOTATION" - must be whitespace before
	// the name.
	
	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}
	
	// Must be whitespace following the name

	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if (lineHasToken( "SYSTEM"))
	{
		if( RC_BAD( rc = processID( FALSE)))
		{
			goto Exit;
		}
	}
	else if (lineHasToken( "PUBLIC"))
	{
		if( RC_BAD( rc = processID( TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_EXPECTING_SYSTEM_OR_PUBLIC,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if( getChar() != FLM_UNICODE_GT)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_GT,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes and attribute definition
****************************************************************************/
RCODE F_XMLParser::processAttDef( void)
{
	RCODE		rc = NE_FLM_OK;

	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}
	
	// Must be whitespace between the name and type

	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processAttType()))
	{
		goto Exit;
	}
	
	// Must be whitespace between type and default decl.

	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processDefaultDecl()))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an attribute type
****************************************************************************/
RCODE F_XMLParser::processAttType( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUNICODE	uChar;
	FLMUINT		uiChars;
	
	if (lineHasToken( "CDATA"))
	{
	}
	else if (lineHasToken( "ID"))
	{
		if (lineHasToken( "REF"))
		{
			if (peekChar() == FLM_UNICODE_S)
			{
				(void)getChar();
			}
		}
	}
	else if (lineHasToken( "ENTIT"))
	{
		if (lineHasToken( "IES"))
		{
		}
		else if (peekChar() == FLM_UNICODE_Y)
		{
			(void)getChar();
		}
	}
	else if (lineHasToken( "NMTOKEN"))
	{
		if (peekChar() == FLM_UNICODE_S)
		{
			(void)getChar();
		}
	}
	else if (lineHasToken( "NOTATION"))
	{
		if( RC_BAD( rc = skipWhitespace( TRUE)))
		{
			goto Exit;
		}

		if (getChar() != FLM_UNICODE_LPAREN)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_LPAREN,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}

		for( ;;)
		{
			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getName( NULL)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}

			uChar = getChar();
			if( uChar == FLM_UNICODE_RPAREN)
			{
				break;
			}
			else if( uChar != FLM_UNICODE_PIPE)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_EXPECTING_RPAREN_OR_PIPE,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
		}
	}
	else if (peekChar() == FLM_UNICODE_LPAREN)
	{
		(void)getChar();
		for( ;;)
		{
			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}

			getNmtoken( &uiChars);

			if( !uiChars)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						XML_ERR_EXPECTING_NAME,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}

			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}

			uChar = getChar();
			if( uChar == FLM_UNICODE_RPAREN)
			{
				break;
			}
			else if( uChar != FLM_UNICODE_PIPE)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_EXPECTING_RPAREN_OR_PIPE,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
		}
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_INVALID_ATT_TYPE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a default declaration
****************************************************************************/
RCODE F_XMLParser::processDefaultDecl( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUNICODE	uChar;

	uChar = getChar();
	if (uChar == FLM_UNICODE_POUND)
	{
		if (lineHasToken( "FIXED"))
		{
			if( RC_BAD( rc = skipWhitespace( TRUE)))
			{
				goto Exit;
			}
			if( RC_BAD( rc = processAttValue( NULL)))
			{
				goto Exit;
			}
		}
		else if (lineHasToken( "IMPLIED"))
		{
			goto Exit;
		}
		else if (lineHasToken( "REQUIRED"))
		{
			goto Exit;
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_INVALID_DEFAULT_DECL,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}
	else if (isQuoteChar( uChar))
	{
		ungetChar();

		if( RC_BAD( rc = processAttValue( NULL)))
		{
			goto Exit;
		}
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
					XML_ERR_INVALID_DEFAULT_DECL,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a content specification
****************************************************************************/
RCODE F_XMLParser::processContentSpec( void)
{
	RCODE		rc = NE_FLM_OK;
	
	if (lineHasToken( "EMPTY") || lineHasToken( "ANY"))
	{
		goto Exit;
	}

	if (getChar() == FLM_UNICODE_LPAREN)
	{
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}
		
		if (lineHasToken( "#PCDATA"))
		{
			if( RC_BAD( rc = processMixedContent()))
			{
				goto Exit;
			}
		}
		else if (lineHasToken( "#"))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset + 1,
					XML_ERR_EXPECTING_PCDATA,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
		else
		{
			if( RC_BAD( rc = processChildContent()))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes mixed content
****************************************************************************/
RCODE F_XMLParser::processMixedContent( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUNICODE	uChar;
	FLMBOOL		bExpectingAsterisk = FALSE;
	
	// Have eaten up the "(<whitespace>#PCDATA"

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}

		uChar = getChar();
		if( uChar == FLM_UNICODE_RPAREN)
		{
			break;
		}
		else if( uChar == FLM_UNICODE_PIPE)
		{
			if( RC_BAD( rc = skipWhitespace( FALSE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getName( NULL)))
			{
				goto Exit;
			}

			bExpectingAsterisk = TRUE;
		}
		else
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_RPAREN_OR_PIPE,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}

	if( bExpectingAsterisk)
	{
		if( getChar() != FLM_UNICODE_ASTERISK)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_ASTERISK,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes child content
****************************************************************************/
RCODE F_XMLParser::processChildContent( void)
{
	RCODE					rc = NE_FLM_OK;
	FLMUNICODE			uChar;
	FLMUINT				uiItemCount = 0;
	FLMUINT				uiDelimCount = 0;
	FLMBOOL				bChoice = FALSE;
	FLMBOOL				bSeq = FALSE;

	// Have eaten up the "("
	
	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}

		uChar = getChar();
		if( uChar == FLM_UNICODE_LPAREN)
		{
			if( RC_BAD( rc = processChildContent()))
			{
				goto Exit;
			}

			uiItemCount++;
		}
		else if (uChar == FLM_UNICODE_RPAREN)
		{
			if( !uiItemCount || (uiItemCount - 1) != uiDelimCount)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_EMPTY_CONTENT_INVALID,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}

			break;
		}
		else if (uChar == FLM_UNICODE_PIPE)
		{
			if( bSeq)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_CANNOT_MIX_CHOICE_AND_SEQ,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
			bChoice = TRUE;
			uiDelimCount++;
		}
		else if (uChar == FLM_UNICODE_COMMA)
		{
			if (bChoice)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_CANNOT_MIX_CHOICE_AND_SEQ,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
			bSeq = TRUE;
			uiDelimCount++;
		}
		else
		{
			ungetChar();
			if( RC_BAD( rc = getName( NULL)))
			{
				goto Exit;
			}
			uiItemCount++;
			
			uChar = peekChar();
			if (uChar == FLM_UNICODE_QUEST ||
				 uChar == FLM_UNICODE_ASTERISK ||
				 uChar == FLM_UNICODE_PLUS)
			{
				
				// Eat up a "?", "*", or "+"
				
				(void)getChar();
			}
		}
	}

	uChar = peekChar();
	if( uChar == FLM_UNICODE_QUEST ||
		uChar == FLM_UNICODE_ASTERISK ||
		uChar == FLM_UNICODE_PLUS)
	{
		// Eat up a "?", "*", or "+"
				
		(void)getChar();
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a misc. declaration
****************************************************************************/
RCODE F_XMLParser::processMisc( void)
{
	RCODE		rc = NE_FLM_OK;

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			if( rc == NE_FLM_IO_END_OF_FILE || rc == NE_FLM_EOF_HIT)
			{
				rc = NE_FLM_OK;
			}
			
			goto Exit;
		}

		if (lineHasToken( "<!--"))
		{
			if (RC_BAD( rc = processComment( NULL, 0, 0, 0, 0)))
			{
				goto Exit;
			}
		}
		else if (lineHasToken( "<?"))
		{
			if( RC_BAD( rc = processPI( NULL, 0, 0, 0, 0)))
			{
				goto Exit;
			}
		}
		else
		{
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a processing instruction
****************************************************************************/
RCODE F_XMLParser::processPI(
	IF_DOMNode *	pParent,
	FLMUINT			uiSavedLineNum,
	FLMUINT     	uiSavedOffset,
	FLMUINT			uiSavedFilePos,
	FLMUINT			uiSavedLineBytes)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUINT			uiChars;
	FLMUINT			uiOffset = 0;
	IF_DOMNode *	pPI = NULL;
	FLMUINT			uiNameLineNum = m_uiCurrLineNum;
	FLMUINT     	uiNameOffset = m_uiCurrLineOffset;
	FLMUINT			uiNameFilePos = m_uiCurrLineFilePos;
	FLMUINT			uiNameLineBytes = m_uiCurrLineBytes;

	// Have already eaten up the "<?"

	if( RC_BAD( rc = getName( &uiChars)))
	{
		goto Exit;
	}

	if( uiChars >= 3 &&
		(m_uChars[ 0] == FLM_UNICODE_X ||
		m_uChars[ 0] == FLM_UNICODE_x) &&
		(m_uChars[ 1] == FLM_UNICODE_M ||
		m_uChars[ 1] == FLM_UNICODE_m) &&
		(m_uChars[ 2] == FLM_UNICODE_L ||
		m_uChars[ 2] == FLM_UNICODE_l))
	{
		setErrInfo( uiNameLineNum,
				uiNameOffset,
				XML_ERR_XML_ILLEGAL_PI_NAME,
				uiNameFilePos,
				uiNameLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	if( pParent)
	{
		if( RC_BAD( rc = pParent->createNode( PROCESSING_INSTRUCTION_NODE,
			0, FLM_LAST_CHILD, &pPI)))
		{
			setErrInfo( uiSavedLineNum,
					uiSavedOffset,
					XML_ERR_CREATING_PI_NODE,
					uiSavedFilePos,
					uiSavedLineBytes);
			goto Exit;
		}

		if( RC_BAD( rc = pPI->setUnicode( m_uChars)))
		{
			goto Exit;
		}
	}
	
	// Must be whitespace after the name

	if( RC_BAD( rc = skipWhitespace( TRUE)))
	{
		goto Exit;
	}

	for (;;)
	{
		if (lineHasToken( "?>"))
		{
			break;
		}
		if ((uChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			uChar = ASCII_NEWLINE;
		}

		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
		uiOffset += sizeof( FLMUNICODE);
		
		if( uiOffset >= m_uiValBufSize)
		{
			if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
			{
				goto Exit;
			}
		}
	}

	if( uiOffset && pPI)
	{
		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = 0;
		if( RC_BAD( rc = pPI->setUnicode( (FLMUNICODE *)m_pucValBuf)))
		{
			goto Exit;
		}
	}

Exit:

	if( pPI)
	{
		pPI->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Gets an XML name from the input stream
****************************************************************************/
RCODE F_XMLParser::getName(
	FLMUINT *	puiChars)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiCharCount = 0;
	FLMUNICODE		uChar;

	// Get the first character

	uChar = getChar();

	if( !isLetter( uChar) && uChar != FLM_UNICODE_UNDERSCORE)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_ILLEGAL_FIRST_NAME_CHAR,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	m_uChars[ uiCharCount++] = uChar;
	
	// Cannot go off of the current line
	
	for (;;)
	{
		if ((uChar = getChar()) == 0)
		{
			break;
		}
		if (!isNameChar( uChar))
		{
			ungetChar();
			break;
		}

		if (uiCharCount >= FLM_XML_MAX_CHARS)
		{
			rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		m_uChars [uiCharCount++] = uChar;
	}

	m_uChars[ uiCharCount] = 0;

Exit:

	if (puiChars)
	{
		*puiChars = uiCharCount;
	}
	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLParser::getQualifiedName(
	FLMUINT *		puiChars,
	FLMUNICODE **	ppuzPrefix,
	FLMUNICODE **	ppuzLocal,
	FLMBOOL *		pbNamespaceDecl,
	FLMBOOL *		pbDefaultNamespaceDecl
	)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiCharCount = 0;
	FLMUNICODE *	puzPrefix = NULL;
	FLMUNICODE *	puzLocal = &m_uChars [0];
	FLMUNICODE		uChar;
	FLMBOOL			bFoundColon = FALSE;

	*pbNamespaceDecl = FALSE;
	if (pbDefaultNamespaceDecl)
	{
		*pbDefaultNamespaceDecl = FALSE;
	}

	// Get the first character, then the rest of the name must
	// be on the same line.

	uChar = getChar();

	if( !isLetter( uChar) && uChar != FLM_UNICODE_UNDERSCORE)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_ILLEGAL_FIRST_NAME_CHAR,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	m_uChars[ uiCharCount++] = uChar;
	for (;;)
	{
		if ((uChar = getChar()) == 0)
		{
			break;
		}
		if (!isNameChar( uChar))
		{
			ungetChar();
			break;
		}

		if( uiCharCount >= FLM_XML_MAX_CHARS)
		{
			rc = RC_SET( NE_FLM_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		if( uChar == FLM_UNICODE_COLON)
		{
			if( bFoundColon)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_ILLEGAL_COLON_IN_NAME,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
			
			// If what we have so far is "xmlns", then don't put it
			// into the prefix - the xmlns: should simply be part of
			// the local name.
			
			if (uiCharCount != 5 || !isXMLNS( m_uChars))
			{
				uChar = 0;
				puzPrefix = &m_uChars [0];
				puzLocal = &m_uChars[ uiCharCount + 1];
			}
			else
			{
				*pbNamespaceDecl = TRUE;
			}
			bFoundColon = TRUE;
		}

		m_uChars[ uiCharCount++] = uChar;
	}

	m_uChars[ uiCharCount] = 0;

	*ppuzPrefix = puzPrefix;
	*ppuzLocal = puzLocal;
	
	if (!puzPrefix && !*pbNamespaceDecl && uiCharCount == 5 &&
		 isXMLNS( m_uChars))
	{
		*pbNamespaceDecl = TRUE;
		if (pbDefaultNamespaceDecl)
		{
			*pbDefaultNamespaceDecl = TRUE;
		}
	}

Exit:

	*puiChars = uiCharCount;
	return( rc);
}

/****************************************************************************
Desc:		Gets an XML Nmtoken from the input stream
****************************************************************************/
void F_XMLParser::getNmtoken(
	FLMUINT *		puiChars)
{
	FLMUINT		uiCharCount = 0;
	FLMUNICODE	uChar;

	for (;;)
	{
		if ((uChar = getChar()) == 0)
		{
			break;
		}

		if( !isNameChar( uChar))
		{
			ungetChar();
			break;
		}
		uiCharCount++;
	}
	*puiChars = uiCharCount;
}

/****************************************************************************
Desc:		Processes the XML version information encoded within the document
****************************************************************************/
RCODE F_XMLParser::processVersion( void)
{
	RCODE				rc = NE_FLM_OK;
	
	// Have already eaten the whitespace
	
	if (!lineHasToken( "version"))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_EXPECTING_VERSION,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if( getChar() != FLM_UNICODE_EQ)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_EQ,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}
	
	// Version must be '1.0' or "1.0"

	if (!lineHasToken( "'1.0'") && !lineHasToken( "\"1.0\""))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_INVALID_VERSION_NUM,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML encoding declaration
****************************************************************************/
RCODE F_XMLParser::processEncodingDecl( void)
{
	RCODE		rc = NE_FLM_OK;

	// Have already skipped the "encoding"

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if (getChar() != FLM_UNICODE_EQ)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_EQ,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}
	
	if (lineHasToken( "'UTF-8'") || lineHasToken( "\"UTF-8\"") ||
		 lineHasToken( "'utf-8'") || lineHasToken( "\"utf-8\""))
	{
		m_eXMLEncoding = FLM_XML_UTF8_ENCODING;
	}
	else if (lineHasToken( "'us-ascii'") || lineHasToken( "\"us-ascii\""))
	{
		m_eXMLEncoding = FLM_XML_USASCII_ENCODING;
	}
	else
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_UNSUPPORTED_ENCODING,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}
	m_importStats.eXMLEncoding = m_eXMLEncoding;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML SD declaration
****************************************************************************/
RCODE F_XMLParser::processSDDecl( void)
{
	RCODE		rc = NE_FLM_OK;

	// Have already eaten the "standalone"

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}

	if( getChar() != FLM_UNICODE_EQ)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_EQ,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( FALSE)))
	{
		goto Exit;
	}
	
	if (!lineHasToken( "'yes'") &&
		 !lineHasToken( "\"yes\"") &&
		 !lineHasToken( "'no'") &&
		 !lineHasToken( "\"no\""))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_EXPECTING_YES_OR_NO,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Get next byte from input stream.
****************************************************************************/
RCODE F_XMLParser::getByte(
	FLMBYTE *	pucByte)
{
	RCODE	rc = NE_FLM_OK;
	
	if (m_ucUngetByte)
	{
		*pucByte = m_ucUngetByte;
		m_ucUngetByte = 0;
	}
	else
	{
		if( RC_BAD( rc = m_pStream->read( (char *)pucByte, 1, NULL)))
		{
			goto Exit;
		}
	}
	m_importStats.uiChars++;
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:	Reads next line from the input stream.
****************************************************************************/
RCODE F_XMLParser::getLine( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucByte1;
	FLMBYTE		ucByte2;
	FLMBYTE		ucByte3;
	FLMUNICODE	uChar;
	
	m_uiCurrLineNumChars = 0;
	m_uiCurrLineOffset = 0;
	m_uiCurrLineFilePos = m_importStats.uiChars;	

	for (;;)
	{
		if( RC_BAD( rc = getByte( &ucByte1)))
		{
			if (rc == NE_FLM_EOF_HIT)
			{
				if (m_uiCurrLineNumChars)
				{
					rc = NE_FLM_OK;
				}
			}
			goto Exit;
		}
		
		// Keep count of the characters.
		
		if( m_fnStatus && (m_importStats.uiChars % 1024) == 0)
		{
			m_fnStatus( XML_STATS,
				(void *)&m_importStats, NULL, NULL, m_pvCallbackData);
		}
		
		// Convert CRLF->CR

		if( ucByte1 == ASCII_CR)
		{
			if( RC_BAD( rc = getByte( &ucByte1)))
			{
				if (rc == NE_FLM_EOF_HIT)
				{
					rc = NE_FLM_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}

			if( ucByte1 != ASCII_NEWLINE)
			{
				ungetByte( ucByte1);
			}
			
			// End of the line
			
			break;
		}
		else if (ucByte1 == ASCII_NEWLINE)
		{
			
			// End of the line
			
			break;
		}

		// Look for escape sequences
	
		if( m_uiFlags & FLM_XML_TRANSLATE_ESC_FLAG)
		{
			if( ucByte1 == ASCII_BACKSLASH)
			{
				if( RC_BAD( rc = getByte( &ucByte1)))
				{
					if (rc == NE_FLM_EOF_HIT)
					{
						rc = NE_FLM_OK;
						uChar = ASCII_BACKSLASH;
						goto Process_Char;
					}
					else
					{
						goto Exit;
					}
				}
	
				if( ucByte1 == ASCII_LOWER_N)
				{
					
					// End of line
					
					break;
				}
				else if( ucByte1 == ASCII_LOWER_T)
				{
					uChar = ASCII_TAB;
					goto Process_Char;
				}
				else if( ucByte1 == ASCII_BACKSLASH)
				{
					// No translation -- preserve backslash
					uChar = ASCII_BACKSLASH;
					goto Process_Char;
				}
				else
				{
					ungetByte( ucByte1);
					uChar = ASCII_BACKSLASH;
					goto Process_Char;
				}
			}
		}
	
		if( m_eXMLEncoding == FLM_XML_UTF8_ENCODING)
		{
			if( ucByte1 <= 0x7F)
			{
				uChar = (FLMUNICODE)ucByte1;
			}
			else
			{
	
				if( RC_BAD( rc = getByte( &ucByte2)))
				{
					if (rc == NE_FLM_EOF_HIT)
					{
						rc = RC_SET( NE_FLM_BAD_UTF8);
					}
					goto Exit;
				}
		
				if( (ucByte2 >> 6) != 0x02)
				{
					rc = RC_SET( NE_FLM_BAD_UTF8);
					goto Exit;
				}
		
				if( (ucByte1 >> 5) == 0x06)
				{
					uChar = ((FLMUNICODE)( ucByte1 - 0xC0) << 6) +
										(FLMUNICODE)(ucByte2 - 0x80);
				}
				else
				{
					if( RC_BAD( rc = getByte( &ucByte3)))
					{
						if (rc == NE_FLM_EOF_HIT)
						{
							rc = RC_SET( NE_FLM_BAD_UTF8);
						}
						goto Exit;
					}
			
					if( (ucByte3 >> 6) != 0x02 || (ucByte1 >> 4) != 0x0E)
					{
						rc = RC_SET( NE_FLM_BAD_UTF8);
						goto Exit;
					}
			
					uChar = ((FLMUNICODE)(ucByte1 - 0xE0) << 12) +
										((FLMUNICODE)(ucByte2 - 0x80) << 6) +
										(FLMUNICODE)(ucByte3 - 0x80);
				}
			}
		}
		else if( m_eXMLEncoding == FLM_XML_USASCII_ENCODING)
		{
			uChar = (FLMUNICODE)ucByte1;
		}
		else
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
			goto Exit;
		}

Process_Char:

		// We have a character, add it to the current line.
		
		if (m_uiCurrLineNumChars == m_uiCurrLineBufMaxChars)
		{
			// Allocate more space for the line buffer
			
			if (RC_BAD( rc = f_realloc(
						sizeof( FLMUNICODE) * (m_uiCurrLineBufMaxChars + 512),
						&m_puzCurrLineBuf)))
			{
				goto Exit;
			}
			m_uiCurrLineBufMaxChars += 512;
		}
		m_puzCurrLineBuf [m_uiCurrLineNumChars++] = uChar;
		m_uiCurrLineBytes = m_importStats.uiChars - m_uiCurrLineFilePos;
	}

	// Increment the line count

	m_uiCurrLineNum++;			
	m_importStats.uiLines++;
	if( m_fnStatus && (m_importStats.uiLines % 100) == 0)
	{
		m_fnStatus( XML_STATS,
			(void *)&m_importStats, NULL, NULL, m_pvCallbackData);
	}
	
Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML entity value
****************************************************************************/
RCODE F_XMLParser::processEntityValue( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUNICODE		uQuoteChar;

	uQuoteChar = getChar();
	
	// Caller should already have looked to make sure the
	// character is a quote character.
	
	flmAssert( isQuoteChar( uQuoteChar));

	for( ;;)
	{
		if ((uChar = getChar()) == 0)
		{
			// Quoted value cannot go to next line
			
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					XML_ERR_EXPECTING_QUOTE_BEFORE_EOL,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}

		if (uChar == uQuoteChar)
		{
			break;
		}
		else if( uChar == FLM_UNICODE_PERCENT)
		{
			if( RC_BAD( rc = processPERef()))
			{
				goto Exit;
			}
		}
		else if( uChar == FLM_UNICODE_AMP)
		{
			if( RC_BAD( rc = processReference( NULL)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML PERef
****************************************************************************/
RCODE F_XMLParser::processPERef( void)
{
	RCODE		rc = NE_FLM_OK;
	
	// Have already eaten the "%" character
	
	// Name must immediately follow on the same line as the %
	
	if( RC_BAD( rc = getName( NULL)))
	{
		goto Exit;
	}

	if( getChar() != FLM_UNICODE_SEMI)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_SEMI,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML reference
****************************************************************************/
RCODE F_XMLParser::processReference(
	FLMUNICODE *	puChar)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMBOOL			bHex = FALSE;
	FLMUINT			uiNum;
	FLMUINT			uiNumOffset;

	if( puChar)
	{
		*puChar = 0;
	}
	
	// Ampersand has already been processed.
	
	if (peekChar() == FLM_UNICODE_POUND)
	{
		uiNumOffset = m_uiCurrLineOffset - 1;
		(void)getChar();
		if (peekChar() == FLM_UNICODE_x)
		{
			(void)getChar();
			bHex = TRUE;
		}

		uiNum = 0;
		for (;;)
		{
			if ((uChar = getChar()) == 0)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset,
						XML_ERR_UNEXPECTED_EOL_IN_ENTITY,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}

			if (uChar == FLM_UNICODE_SEMI)
			{
				if (!uiNum || uiNum > 0xFFFF)
				{
					setErrInfo( m_uiCurrLineNum,
							uiNumOffset,
							XML_ERR_INVALID_CHARACTER_NUMBER,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_FLM_INVALID_XML);
					goto Exit;
				}
				if (puChar)
				{
					*puChar = (FLMUNICODE)uiNum;
				}
				break;
			}

			// Validate that the characters are valid dec/hex characters

			if( bHex)
			{
				if (uChar >= ASCII_ZERO && uChar <= ASCII_NINE)
				{
					uiNum <<= 4;
					uiNum += (FLMUINT)(uChar - ASCII_ZERO); 
				}
				else if (uChar >= ASCII_UPPER_A && uChar <= ASCII_UPPER_F)
				{
					uiNum <<= 4;
					uiNum += (FLMUINT)(uChar - ASCII_UPPER_A + 10); 
				}
				else if (uChar >= ASCII_LOWER_A && uChar <= ASCII_LOWER_F)
				{
					uiNum <<= 4;
					uiNum += (FLMUINT)(uChar - ASCII_LOWER_A + 10); 
				}
				else
				{
					setErrInfo( m_uiCurrLineNum,
							uiNumOffset,
							XML_ERR_INVALID_CHARACTER_NUMBER,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_FLM_INVALID_XML);
					goto Exit;
				}
			}
			else
			{
				if (uChar >= ASCII_ZERO && uChar <= ASCII_NINE)
				{
					uiNum *= 10;
					uiNum += (FLMUINT)(uChar - ASCII_ZERO); 
				}
				else
				{
					setErrInfo( m_uiCurrLineNum,
							uiNumOffset,
							XML_ERR_INVALID_CHARACTER_NUMBER,
							m_uiCurrLineFilePos,
							m_uiCurrLineBytes);
					rc = RC_SET( NE_FLM_INVALID_XML);
					goto Exit;
				}
			}
			
			// Cannot handle unicode characters more than 16 bits.
			
			if (uiNum > 0xFFFF)
			{
				setErrInfo( m_uiCurrLineNum,
						uiNumOffset,
						XML_ERR_INVALID_CHARACTER_NUMBER,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
		}
	}
	else if (lineHasToken( "lt;"))
	{
		if (puChar)
		{
			*puChar = FLM_UNICODE_LT;
		}
	}
	else if (lineHasToken( "gt;"))
	{
		if (puChar)
		{
			*puChar = FLM_UNICODE_GT;
		}
	}
	else if (lineHasToken( "amp;"))
	{
		if (puChar)
		{
			*puChar = FLM_UNICODE_AMP;
		}
	}
	else if (lineHasToken( "apos;"))
	{
		if (puChar)
		{
			*puChar = FLM_UNICODE_APOS;
		}
	}
	else if (lineHasToken( "quot;"))
	{
		if (puChar)
		{
			*puChar = FLM_UNICODE_QUOTE;
		}
	}
	else if (puChar)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_UNSUPPORTED_ENTITY,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
					
		// If we are expecting an entity that can be turned into
		// a character, we need to return an error.
		
		rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
		goto Exit;
	}
	else 
	{
		
		// Name must be on the same line as the '&'
		
		if( RC_BAD( rc = getName( NULL)))
		{
			goto Exit;
		}
		
		// Make sure we have a semicolon after the name
		
		if (getChar() != FLM_UNICODE_SEMI)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_SEMI,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an attribute value
****************************************************************************/
RCODE F_XMLParser::processAttValue(
	XML_ATTR *			pAttr)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUNICODE		uQuoteChar;
	FLMUINT			uiOffset = 0;
	
	// Must be on a single or double quote
	
	uQuoteChar = getChar();
	if (!isQuoteChar( uQuoteChar))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_QUOTE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	for( ;;)
	{
		if ((uChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			uChar = ASCII_NEWLINE;
		}
		if (uChar == uQuoteChar)
		{
			break;
		}
		else if( uChar == FLM_UNICODE_AMP)
		{
			if( RC_BAD( rc = processReference( &uChar)))
			{
				goto Exit;
			}

			flmAssert( uChar);
			*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
			uiOffset += sizeof( FLMUNICODE);
			if( uiOffset >= m_uiValBufSize)
			{
				if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
				{
					goto Exit;
				}
			}
		}
		else
		{
			*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
			uiOffset += sizeof( FLMUNICODE);
			if( uiOffset >= m_uiValBufSize)
			{
				if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
				{
					goto Exit;
				}
			}
		}
	}

	if( pAttr && uiOffset)
	{
		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = 0;
		if( RC_BAD( rc = setUnicode( pAttr, (FLMUNICODE *)m_pucValBuf)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Reads an XML system literal from the input stream
****************************************************************************/
RCODE F_XMLParser::getSystemLiteral( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uQuoteChar;
	FLMUNICODE		uChar;

	uQuoteChar = getChar();

	if (!isQuoteChar( uQuoteChar))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_QUOTE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	for (;;)
	{
		
		// Must terminate with quote character.
		
		if ((uChar = getChar()) == 0)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					XML_ERR_EXPECTING_QUOTE_BEFORE_EOL,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}

		if (uChar == uQuoteChar)
		{
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Reads an XML public ID literal from the input stream
****************************************************************************/
RCODE F_XMLParser::getPubidLiteral( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uQuoteChar;
	FLMUNICODE		uChar;

	uQuoteChar = getChar();

	if (!isQuoteChar( uQuoteChar))
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset - 1,
				XML_ERR_EXPECTING_QUOTE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

	for( ;;)
	{
		
		// Must terminate with quote character.
		
		if ((uChar = getChar()) == 0)
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset,
					XML_ERR_EXPECTING_QUOTE_BEFORE_EOL,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}

		if (uChar == uQuoteChar)
		{
			break;
		}
		if( !isPubidChar( uChar))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_INVALID_PUBLIC_ID_CHAR,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}
	}


Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML comment
****************************************************************************/
RCODE F_XMLParser::processComment(
	IF_DOMNode *	pParent,
	FLMUINT			uiSavedLineNum,
	FLMUINT     	uiSavedOffset,
	FLMUINT			uiSavedFilePos,
	FLMUINT			uiSavedLineBytes)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUINT			uiOffset;
	FLMUINT			uiMaxOffset;
	IF_DOMNode *	pComment = NULL;

	// Have already eaten up the "<!--"
	
	uiOffset = 0;
	uiMaxOffset = m_uiValBufSize;
	for( ;;)
	{
		
		// See if we have the termination of the comment
		
		if (lineHasToken( "-->"))
		{
			break;
		}
		
		if ((uChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			uChar = ASCII_NEWLINE;
		}
		
		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
		uiOffset += sizeof( FLMUNICODE);

		if( uiOffset >= uiMaxOffset)
		{
			if (RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
			{
				goto Exit;
			}
			uiMaxOffset = m_uiValBufSize;	
		}
	}

	if( pParent)
	{
		if( RC_BAD( rc = pParent->createNode( COMMENT_NODE, 0,
			FLM_LAST_CHILD, &pComment)))
		{
			setErrInfo( uiSavedLineNum,
					uiSavedOffset,
					XML_ERR_CREATING_COMMENT_NODE,
					uiSavedFilePos,
					uiSavedLineBytes);
			goto Exit;
		}

		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = 0;

		if( RC_BAD( rc = pComment->setUnicode( (FLMUNICODE *)m_pucValBuf)))
		{
			goto Exit;
		}

		pComment->Release();
		pComment = NULL;
	}

Exit:

	if( pComment)
	{
		pComment->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Processes a CDATA tag
****************************************************************************/
RCODE F_XMLParser::processCDATA(
	IF_DOMNode *	pParent,
	FLMUINT			uiSavedLineNum,
	FLMUINT     	uiSavedOffset,
	FLMUINT			uiSavedFilePos,
	FLMUINT			uiSavedLineBytes)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUINT			uiOffset = 0;
	IF_DOMNode *	pCData = NULL;

	// Have already eaten up the "<![CDATA["

	for( ;;)
	{
		if (lineHasToken( "]]>"))
		{
			break;
		}
		if ((uChar = getChar()) == 0)
		{
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			uChar = ASCII_NEWLINE;
		}

		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = uChar;
		uiOffset += sizeof( FLMUNICODE);

		if( uiOffset >= m_uiValBufSize)
		{
			if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
			{
				goto Exit;
			}
		}
	}

	if( pParent)
	{
		if( RC_BAD( rc = pParent->createNode( CDATA_SECTION_NODE, 0,
			FLM_LAST_CHILD, &pCData)))
		{
			setErrInfo( uiSavedLineNum,
					uiSavedOffset,
					XML_ERR_CREATING_CDATA_NODE,
					uiSavedFilePos,
					uiSavedLineBytes);
			goto Exit;
		}

		*((FLMUNICODE *)(&m_pucValBuf[ uiOffset])) = 0;
		if( RC_BAD( rc = pCData->setUnicode( (FLMUNICODE *)m_pucValBuf)))
		{
			goto Exit;
		}

		pCData->Release();
		pCData = NULL;
	}

Exit:

	if( pCData)
	{
		pCData->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:		Skips any whitespace characters in the input stream
****************************************************************************/
RCODE F_XMLParser::skipWhitespace(
	FLMBOOL 			bRequired)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUINT			uiCount = 0;

	for( ;;)
	{
		if ((uChar = getChar()) == 0)
		{
			uiCount++;
			if (RC_BAD( rc = getLine()))
			{
				goto Exit;
			}
			continue;
		}

		if( !isWhitespace( uChar))
		{
			ungetChar();
			break;
		}
		uiCount++;
	}

	if( !uiCount && bRequired)
	{
		setErrInfo( m_uiCurrLineNum,
				m_uiCurrLineOffset,
				XML_ERR_EXPECTING_WHITESPACE,
				m_uiCurrLineFilePos,
				m_uiCurrLineBytes);
		rc = RC_SET( NE_FLM_INVALID_XML);
		goto Exit;
	}

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_XMLParser::resizeValBuffer(
	FLMUINT			uiSize)
{
	RCODE			rc = NE_FLM_OK;

	if( uiSize == m_uiValBufSize)
	{
		goto Exit;
	}

	if( uiSize == ~((FLMUINT)0))
	{
		uiSize = m_uiValBufSize + 2048;
	}

	if( m_pucValBuf)
	{
		if( uiSize)
		{
			if( RC_BAD( rc = f_realloc( uiSize, &m_pucValBuf)))
			{
				goto Exit;
			}
		}
		else
		{
			f_free( &m_pucValBuf);
			m_pucValBuf = NULL;
		}
	}
	else
	{
		flmAssert( !m_pucValBuf);

		if( RC_BAD( rc = f_alloc( uiSize, &m_pucValBuf)))
		{
			goto Exit;
		}
	}

	m_uiValBufSize = uiSize;

Exit:

	return( rc);
}

/*****************************************************************************
Desc:
******************************************************************************/
RCODE F_XMLParser::getBinaryVal(
	FLMUINT *		puiLength)
{
	RCODE				rc = NE_FLM_OK;
	FLMUNICODE		uChar;
	FLMUNICODE		uChar2;
	FLMUINT			uiOffset = 0;
	FLMBOOL			bHavePreamble;

	flmAssert( *puiLength == 0);

	for( ;;)
	{
		bHavePreamble = FALSE;

		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}

Retry:

		uChar = getChar();
		if( !f_isHexChar( uChar))
		{
			if( uChar != FLM_UNICODE_LT)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_EXPECTING_HEX_DIGIT,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}
			ungetChar();
			break;
		}

		uChar2 = getChar();
		if( uChar == FLM_UNICODE_0 && 
			(uChar2 == FLM_UNICODE_X || uChar2 == FLM_UNICODE_x))
		{
			if( bHavePreamble)
			{
				setErrInfo( m_uiCurrLineNum,
						m_uiCurrLineOffset - 1,
						XML_ERR_EXPECTING_HEX_DIGIT,
						m_uiCurrLineFilePos,
						m_uiCurrLineBytes);
				rc = RC_SET( NE_FLM_INVALID_XML);
				goto Exit;
			}

			bHavePreamble = TRUE;
			goto Retry;
		}

		if( !f_isHexChar( uChar2))
		{
			setErrInfo( m_uiCurrLineNum,
					m_uiCurrLineOffset - 1,
					XML_ERR_EXPECTING_HEX_DIGIT,
					m_uiCurrLineFilePos,
					m_uiCurrLineBytes);
			rc = RC_SET( NE_FLM_INVALID_XML);
			goto Exit;
		}

		if( uiOffset >= m_uiValBufSize)
		{
			if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
			{
				goto Exit;
			}
		}

		m_pucValBuf[ uiOffset++] = 
			(f_getHexVal( uChar) << 4) | f_getHexVal( uChar2);

		if( RC_BAD( rc = skipWhitespace( FALSE)))
		{
			goto Exit;
		}

		uChar = getChar();
		if( uChar != FLM_UNICODE_COMMA)
		{
			ungetChar();
		}
	}

	*puiLength = uiOffset;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Constructor
****************************************************************************/
F_XMLNamespaceMgr::F_XMLNamespaceMgr()
{
	m_pFirstNamespace = NULL;
	m_uiNamespaceCount = 0;
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_XMLNamespaceMgr::~F_XMLNamespaceMgr()
{
	popNamespaces( m_uiNamespaceCount);
}

/****************************************************************************
Desc:
****************************************************************************/
void F_XMLNamespaceMgr::popNamespaces(
	FLMUINT			uiCount)
{
	F_XMLNamespace *		pTmpNamespace;

	flmAssert( uiCount <= m_uiNamespaceCount);

	while( uiCount && m_pFirstNamespace)
	{
		pTmpNamespace = m_pFirstNamespace;
		m_pFirstNamespace = m_pFirstNamespace->m_pNext;
		pTmpNamespace->m_pNext = NULL;
		pTmpNamespace->Release();
		m_uiNamespaceCount--;
		uiCount--;
	}
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLNamespaceMgr::findNamespace(
	FLMUNICODE *		puzPrefix,
	F_XMLNamespace **	ppNamespace,
	FLMUINT				uiMaxSearchSize)
{
	RCODE					rc = NE_FLM_OK;
	F_XMLNamespace *	pTmpNamespace = m_pFirstNamespace;

	while( pTmpNamespace)
	{
		if( !uiMaxSearchSize)
		{
			pTmpNamespace = NULL;
			break;
		}

		if( !puzPrefix && !pTmpNamespace->m_puzPrefix)
		{
			break;
		}
		else if( puzPrefix && pTmpNamespace->m_puzPrefix)
		{
			if( f_unicmp( puzPrefix, pTmpNamespace->m_puzPrefix) == 0)
			{
				break;
			}
		}

		pTmpNamespace = pTmpNamespace->m_pNext;
		uiMaxSearchSize--;
	}

	if( !pTmpNamespace)
	{
		rc = RC_SET( NE_FLM_NOT_FOUND);
		goto Exit;
	}

	if( ppNamespace)
	{
		if( *ppNamespace)
		{
			(*ppNamespace)->Release();
		}

		pTmpNamespace->AddRef();
		*ppNamespace = pTmpNamespace;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLNamespaceMgr::pushNamespace(
	FLMUNICODE *		puzPrefix,
	FLMUNICODE *		puzNamespaceURI)
{
	RCODE					rc = NE_FLM_OK;
	F_XMLNamespace *	pNewNamespace = NULL;

	if( (pNewNamespace = f_new F_XMLNamespace) == NULL)
	{
		rc = RC_SET( NE_FLM_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = pNewNamespace->setPrefix( puzPrefix)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pNewNamespace->setURI( puzNamespaceURI)))
	{
		goto Exit;
	}

	pNewNamespace->m_pNext = m_pFirstNamespace;
	m_pFirstNamespace = pNewNamespace;
	pNewNamespace = NULL;
	m_uiNamespaceCount++;

Exit:

	if( pNewNamespace)
	{
		pNewNamespace->Release();
	}

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLNamespaceMgr::pushNamespace(
	F_XMLNamespace *	pNamespace)
{
	flmAssert( m_pFirstNamespace != pNamespace &&
		!pNamespace->m_pNext);

	pNamespace->AddRef();
	pNamespace->m_pNext = m_pFirstNamespace;
	m_pFirstNamespace = pNamespace;
	m_uiNamespaceCount++;

	return( NE_FLM_OK);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLNamespace::setPrefix(
	FLMUNICODE *		puzPrefix)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLen;

	if( m_puzPrefix)
	{
		f_free( &m_puzPrefix);
	}

	if( puzPrefix)
	{
		uiLen = f_unilen( puzPrefix);
		if( RC_BAD( rc = f_alloc( sizeof( FLMUNICODE) * (uiLen + 1),
			&m_puzPrefix)))
		{
			goto Exit;
		}

		f_unicpy( m_puzPrefix, puzPrefix);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLNamespace::setURI(
	FLMUNICODE *		puzURI)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLen;

	if( m_puzURI)
	{
		f_free( &m_puzURI);
	}

	if( puzURI)
	{
		uiLen = f_unilen( puzURI);
		if( RC_BAD( rc = f_alloc( 
			sizeof( FLMUNICODE) * (uiLen + 1), &m_puzURI)))
		{
			goto Exit;
		}

		f_unicpy( m_puzURI, puzURI);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
RCODE F_XMLNamespace::setup(
	FLMUNICODE *		puzPrefix,
	FLMUNICODE *		puzURI,
	F_XMLNamespace *	pNext)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLen;

	flmAssert( !m_puzPrefix);
	flmAssert( !m_puzURI);
	flmAssert( !m_pNext);

	if( puzPrefix)
	{
		uiLen = f_unilen( puzPrefix);
		if( RC_BAD( rc = f_alloc( sizeof( FLMUNICODE) * (uiLen + 1),
			&m_puzPrefix)))
		{
			goto Exit;
		}

		f_unicpy( m_puzPrefix, puzPrefix);
	}

	if( puzURI)
	{
		uiLen = f_unilen( puzURI);
		if( RC_BAD( rc = f_alloc( sizeof( FLMUNICODE) * (uiLen + 1),
			&m_puzURI)))
		{
			goto Exit;
		}

		f_unicpy( m_puzURI, puzURI);
	}

	m_pNext = pNext;

Exit:

	return( rc);
}

/****************************************************************************
Desc:
****************************************************************************/
#if 0
RCODE F_XMLParser::addAttributesToElement(
	IF_DOMNode *		pElement)
{
	RCODE					rc = NE_FLM_OK;
	XML_ATTR *			pAttr;
	F_XMLNamespace *	pNamespace = NULL;
	IF_DOMNode *		pTmpNode = NULL;
	FLMUINT				uiNameId;
	eFlmDataType		attrDataType;
	
	// Make sure any prefixes (e.g., xmlns:xxxx) are added to the database
	// before they are used - in case they are used by the attributes
	// themselves.

	for( pAttr = m_pFirstAttr; pAttr; pAttr = pAttr->pNext)
	{
		if( pAttr->uiFlags & F_PREFIXED_NS_DECL)
		{
			FLMUINT	uiPrefixId;
			
			// Create the prefix (stored in &puzLocalName [6]) if it doesn't 
			// already exist

//			if( RC_BAD( rc = m_pDb->m_pDict->getPrefixId( m_pDb,
//				&pAttr->puzLocalName [6], &uiPrefixId)))
//			{
//				if( rc != NE_FLM_NOT_FOUND)
//				{
//					goto Exit;
//				}
//
//				uiPrefixId = 0;
//				if( RC_BAD( rc = m_pDb->createPrefixDef( TRUE,
//					&pAttr->puzLocalName [6], &uiPrefixId)))
//				{
//					goto Exit;
//				}
//			}
		}
	}
	
	// Add the attributes to the element
	//
	// NOTE: The XML namespace specification states that the names
	// of all unqualified attributes are assigned to the
	// appropriate per-element-type partition.  This means that
	// the combination of the attribute name with the parent
	// element's type and namespace name is used to uniquely
	// identify each unqualified attribute.
	//
	// For sake of clarity and useability, however, the parser
	// deviates from the namespace specification.  Each unprefixed
	// attribute encountered by the parser will inherit the
	// namespace of the parent element, even if the namespace
	// is a default namespace.

	for( pAttr = m_pFirstAttr; pAttr; pAttr = pAttr->pNext)
	{
		if( pAttr->uiFlags & F_DEFAULT_NS_DECL)
		{
			// Add the namespace declaration to the element

//			if( RC_BAD( rc = pElement->createAttribute( ATTR_XMLNS_TAG, 
//				&pTmpNode)))
//			{
//				goto Exit;
//			}
//
//			if( RC_BAD( rc = pTmpNode->setUnicode( pAttr->puzVal)))
//			{
//				goto Exit;
//			}

//			if( RC_BAD( rc = pTmpNode->addModeFlags( m_pDb, FDOM_READ_ONLY)))
//			{
//				goto Exit;
//			}
		}
		else if( pAttr->uiFlags & F_PREFIXED_NS_DECL)
		{
			// Find the attribute definition

//			if( RC_BAD( rc = m_pDb->getAttributeNameId(
//				NULL, pAttr->puzLocalName, &uiNameId)))
//			{
//				if( rc != NE_FLM_NOT_FOUND)
//				{
//					goto Exit;
//				}
//
//				if( !(m_uiFlags & FLM_XML_EXTEND_DICT_FLAG))
//				{
//					rc = RC_SET( NE_FLM_UNDEFINED_ATTRIBUTE_NAME);
//					goto Exit;
//				}
//
//				uiNameId = 0;
//				if( RC_BAD( rc = m_pDb->createAttributeDef(
//					NULL, pAttr->puzLocalName, FLM_TEXT_TYPE, &uiNameId,
//					&pTmpNode)))
//				{
//					goto Exit;
//				}
//			}

			// Add the namespace declaration to the element

//			if( RC_BAD( rc = pElement->createAttribute( m_pDb, uiNameId,
//										&pTmpNode)))
//			{
//				goto Exit;
//			}
//
//			if( RC_BAD( rc = pTmpNode->setUnicode( m_pDb, pAttr->puzVal)))
//			{
//				goto Exit;
//			}
//
//			if( RC_BAD( rc = pTmpNode->addModeFlags( m_pDb, FDOM_READ_ONLY)))
//			{
//				goto Exit;
//			}
		}
		else
		{
			if( pAttr->puzPrefix)
			{
				if( RC_BAD( rc = findNamespace( pAttr->puzPrefix, &pNamespace)))
				{
					if( rc == NE_FLM_NOT_FOUND)
					{
						setErrInfo( pAttr->uiLineNum, pAttr->uiLineOffset,
								XML_ERR_PREFIX_NOT_DEFINED, pAttr->uiLineFilePos,
								pAttr->uiLineBytes);
						rc = RC_SET( NE_FLM_INVALID_XML);
					}
					
					goto Exit;
				}
			}
			else
			{
				if( pNamespace)
				{
					pNamespace->Release();
				}
				pNamespace = NULL;
			}

//			if( RC_BAD( rc = m_pDb->getAttributeNameId(
//				pNamespace ? pNamespace->getURIPtr() : NULL,
//				pAttr->puzLocalName, &uiNameId)))
//			{
//				if( rc != NE_FLM_NOT_FOUND)
//				{
//					goto Exit;
//				}
//
//				if( !(m_uiFlags & FLM_XML_EXTEND_DICT_FLAG) ||
//					(pNamespace && 
//					 f_unicmp( pNamespace->getURIPtr(), gv_uzXFLAIMNamespace) == 0))
//				{
//					rc = RC_SET( NE_FLM_UNDEFINED_ATTRIBUTE_NAME);
//					goto Exit;
//				}
//
//				uiNameId = 0;
//				if( RC_BAD( rc = m_pDb->createAttributeDef(
//					pNamespace ? pNamespace->getURIPtr() : NULL,
//					pAttr->puzLocalName, FLM_TEXT_TYPE, &uiNameId)))
//				{
//					goto Exit;
//				}
//			}

//			if( RC_BAD( rc = pElement->createAttribute( m_pDb, uiNameId,
//										&pTmpNode)))
//			{
//				goto Exit;
//			}
//
//			if (pAttr->puzPrefix)
//			{
//				if( RC_BAD( rc = pTmpNode->setPrefix( m_pDb, pAttr->puzPrefix)))
//				{
//					if( rc == NE_FLM_NOT_FOUND)
//					{
//						setErrInfo( pAttr->uiLineNum,
//								pAttr->uiLineOffset,
//								XML_ERR_PREFIX_NOT_DEFINED,
//								pAttr->uiLineFilePos,
//								pAttr->uiLineBytes);
//						rc = RC_SET( NE_FLM_INVALID_XML);
//					}
//					goto Exit;
//				}
//			}
//
			if( RC_BAD( rc = pTmpNode->getDataType( &attrDataType)))
			{
				goto Exit;
			}

			switch( attrDataType)
			{
				case FLM_TEXT_TYPE:
				{
					if( RC_BAD( rc = pTmpNode->setUnicode( pAttr->puzVal)))
					{
						goto Exit;
					}
					break;
				}

				case FLM_NUMBER_TYPE:
				{
					FLMUINT64		ui64Val;
					FLMBOOL			bNeg;

					if( RC_BAD( rc = unicodeToNumber64( 
						pAttr->puzVal, &ui64Val, &bNeg)))
					{
						goto Exit;
					}

					if( !bNeg)
					{
						if( RC_BAD( rc = pTmpNode->setUINT64( ui64Val)))
						{
							goto Exit;
						}
					}
					else
					{
						if( RC_BAD( rc = pTmpNode->setINT64( -((FLMINT64)ui64Val))))
						{
							goto Exit;
						}
					}

					break;
				}

				case FLM_BINARY_TYPE:
				{
					FLMBOOL			bHavePreamble;
					FLMUNICODE *	puzStr = pAttr->puzVal;
					FLMUINT			uiOffset = 0;

					// Convert the Unicode value to binary

					while( puzStr && *puzStr)
					{
						bHavePreamble = FALSE;

						while( isWhitespace( *puzStr))
						{
							puzStr++;
						}

					Retry:

						if( !f_isHexChar( *puzStr))
						{
							break;
						}

						if( *puzStr == FLM_UNICODE_0 && 
							(puzStr[ 1] == FLM_UNICODE_X || puzStr[ 1] == FLM_UNICODE_x))
						{
							if( bHavePreamble)
							{
								setErrInfo( pAttr->uiValueLineNum,
										pAttr->uiValueLineOffset,
										XML_ERR_INVALID_BINARY_ATTR_VALUE,
										m_uiCurrLineFilePos,
										m_uiCurrLineBytes);
								rc = RC_SET( NE_FLM_INVALID_XML);
								goto Exit;
							}

							bHavePreamble = TRUE;
							puzStr += 2;
							goto Retry;
						}

						if( !f_isHexChar( puzStr[ 1]))
						{
							setErrInfo( pAttr->uiValueLineNum,
									pAttr->uiValueLineOffset,
									XML_ERR_INVALID_BINARY_ATTR_VALUE,
									m_uiCurrLineFilePos,
									m_uiCurrLineBytes);
							rc = RC_SET( NE_FLM_INVALID_XML);
							goto Exit;
						}

						if( uiOffset >= m_uiValBufSize)
						{
							if( RC_BAD( rc = resizeValBuffer( ~((FLMUINT)0))))
							{
								goto Exit;
							}
						}

						m_pucValBuf[ uiOffset++] = 
							(f_getHexVal( *puzStr) << 4) | f_getHexVal( puzStr[ 1]);

						puzStr += 2;

						while( isWhitespace( *puzStr))
						{
							puzStr++;
						}

						if( *puzStr == FLM_UNICODE_COMMA)
						{
							puzStr++;
						}
					}

					if( RC_BAD( rc = pTmpNode->setBinary( m_pucValBuf, uiOffset)))
					{
						goto Exit;
					}
					break;
				}

				default:
				{
					rc = RC_SET_AND_ASSERT( NE_FLM_NOT_IMPLEMENTED);
					goto Exit;
				}
			}
		}
	}

Exit:

	if( pTmpNode)
	{
		pTmpNode->Release();
	}

	if( pNamespace)
	{
		pNamespace->Release();
	}

	return( rc);
}
#endif

// Some forward declarations

class F_Element;
class F_Attribute;

/*****************************************************************************
Desc:	Keeps track of an attribute that we are going to output
*****************************************************************************/
class F_Attribute : public F_Base
{
public:
	F_Attribute(
		F_Element *	pElement)
	{
		m_uiTmpSpaceSize = sizeof( m_uzTmpSpace);
		m_puzName = &m_uzTmpSpace [0];
		reset( pElement);
	}
	
	~F_Attribute();
	
	FINLINE void reset(
		F_Element *	pElement)
	{
		m_uiNameChars = 0;
		m_bIsNamespaceDecl = FALSE;
		m_bDefaultNamespaceDecl = FALSE;
		m_uiNamespaceChars = 0;
		m_uiValueChars = 0;
		m_uiPrefixChars = 0;
		m_pElement = pElement;
	}

	RCODE allocNameSpace( void);
	
//	RCODE setupAttribute(
//		IF_Db *			pDb,
//		IF_DOMNode *	pNode);
		
	RCODE setPrefix( void);
	
	RCODE outputAttr(
		IF_OStream *	pOStream);
		
	FINLINE F_Attribute * getNext( void)
	{
		return( m_pNext);
	}
	
private:
	FLMUNICODE		m_uzTmpSpace [150];
	FLMUINT			m_uiTmpSpaceSize;
	FLMBOOL			m_bIsNamespaceDecl;
	FLMBOOL			m_bDefaultNamespaceDecl;
	FLMUNICODE *	m_puzName;
	FLMUINT			m_uiNameChars;
	FLMUNICODE *	m_puzNamespace;
	FLMUINT			m_uiNamespaceChars;
	FLMUNICODE *	m_puzValue;
	FLMUINT			m_uiValueChars;
	FLMUNICODE *	m_puzPrefix;
	FLMUINT			m_uiPrefixChars;
	F_Element *		m_pElement;
	F_Attribute *	m_pNext;
	
friend class F_Element;
};

/*****************************************************************************
Desc:	Destructor for F_Attribute class.
*****************************************************************************/
F_Attribute::~F_Attribute()
{
	if (m_puzName != &m_uzTmpSpace [0])
	{
		f_free( &m_puzName);
	}
}

/*****************************************************************************
Desc:	Keeps track of an element that we are going to output
*****************************************************************************/
class F_Element : public F_Base
{
public:
	F_Element(
		F_Element *		pParentElement,
		F_Attribute **	ppAvailAttrs,
		FLMUINT *		puiNextPrefixNum)
	{
		m_uiTmpSpaceSize = sizeof( m_uzTmpSpace);
		m_puzName = &m_uzTmpSpace [0];
		m_pFirstAttr = NULL;
		m_pLastAttr = NULL;
		m_pNext = NULL;
		m_uiIndentCount = 0;
		m_bIsDocumentRoot = FALSE;
		reset( pParentElement, ppAvailAttrs, puiNextPrefixNum);
	}
	
	~F_Element()
	{
		F_Attribute *	pAttr;
		F_Attribute *	pTmpAttr;
		
		// Delete all of the attributes
		
		pAttr = m_pFirstAttr;
		while (pAttr)
		{
			pTmpAttr = pAttr;
			pAttr = pAttr->m_pNext;
			delete pTmpAttr;
		}
		
		if (m_puzName != &m_uzTmpSpace [0])
		{
			f_free( &m_puzName);
		}
	}
	
	FINLINE void reset(
		F_Element *		pParentElement,
		F_Attribute **	ppAvailAttrs,
		FLMUINT *		puiNextPrefixNum)
	{
		m_uiNameChars = 0;
		m_uiNamespaceChars = 0;
		m_uiPrefixChars = 0;
		m_pParentElement = pParentElement;
		m_puiNextPrefixNum = puiNextPrefixNum;
		m_ppAvailAttrs = ppAvailAttrs;
		m_pNext = NULL;
		m_uiIndentCount = 0;
		m_bIsDocumentRoot = FALSE;
	}
	
	FINLINE void setIndentCount( 
		FLMUINT	uiIndentCount)
	{
		m_uiIndentCount = uiIndentCount;
	}

	FINLINE void setDocumentRoot(
		FLMBOOL	bIsDocumentRoot)
	{
		m_bIsDocumentRoot = bIsDocumentRoot;
	}

	RCODE allocAttr(
		F_Attribute **	ppAttr);
		
	FINLINE void makeAttrAvail(
		F_Attribute *	pAttr)
	{
		pAttr->m_pNext = *m_ppAvailAttrs;
		*m_ppAvailAttrs = pAttr;
	}

	FINLINE void makeAllAttrsAvail( void)
	{
		if (m_pFirstAttr)
		{
			m_pLastAttr->m_pNext = *m_ppAvailAttrs;
			*m_ppAvailAttrs = m_pFirstAttr;
			m_pFirstAttr = NULL;
			m_pLastAttr = NULL;
		}
	}
		
//	RCODE saveAttribute(
//		IF_Db *			pDb,
//		IF_DOMNode *	pNode);
		
	RCODE allocNameSpace( void);
	
//	RCODE setupElement(
//		IF_Db *			pDb,
//		IF_DOMNode *	pNode);
		
	RCODE addNamespaceDecl(
		FLMUNICODE *	puzPrefix,
		FLMUINT			uiPrefixChars,
		FLMUNICODE *	puzNamespace,
		FLMUINT			uiNamespaceChars,
		F_Attribute **	ppAttr);
		
	void genPrefix(
		FLMUNICODE *	puzPrefix,
		FLMUINT *		puiPrefixChars);
		
	RCODE findPrefix(
		FLMUNICODE *	puzNamespace,
		FLMUINT			uiNamespaceChars,
		FLMBOOL			bForElement,
		FLMUNICODE **	ppuzPrefix,
		FLMUINT *		puiPrefixChars);
		
	FINLINE RCODE setPrefix( void)
	{
		return( findPrefix( m_puzNamespace, m_uiNamespaceChars, TRUE,
						&m_puzPrefix, &m_uiPrefixChars));
	}
	
	FINLINE F_Element * getParentElement( void)
	{
		return( m_pParentElement);
	}

	RCODE outputElem(
		IF_OStream *	pOStream,
		FLMBOOL			bStartOfElement,
		FLMBOOL			bEndOfElement,
		FLMBOOL			bAddNewLine);

//	RCODE outputLocalData( 
//		IF_OStream *			pOStream,
//		IF_DOMNode *		  	pDbNode,
//		IF_Db *					ifpDb,
//		eExportFormatType		eFormatType,
//		FLMUINT					uiIndentCount);
		
	FINLINE F_Element * getNext( void)
	{
		return( m_pNext);
	}
	
	FINLINE void makeAvail(
		F_Element **	ppAvailElements)
	{
		m_pNext = *ppAvailElements;
		*ppAvailElements = this;
	}
	
private:
	FLMUNICODE		m_uzTmpSpace [100];
	FLMUINT			m_uiTmpSpaceSize;
	FLMUNICODE *	m_puzName;
	FLMUINT			m_uiNameChars;
	FLMUNICODE *	m_puzNamespace;
	FLMUINT			m_uiNamespaceChars;
	FLMUNICODE *	m_puzPrefix;
	FLMUINT			m_uiPrefixChars;
	F_Attribute *	m_pFirstAttr;
	F_Attribute *	m_pLastAttr;
	F_Element *		m_pParentElement;
	F_Element *		m_pNext;
	FLMUINT *		m_puiNextPrefixNum;
	F_Attribute **	m_ppAvailAttrs;
	FLMBOOL			m_bIsDocumentRoot;
	FLMUINT			m_uiIndentCount;
	
friend class F_Attribute;
};

/*****************************************************************************
Desc:	Allocate space to hold the name, namespace, and value for an attribute.
*****************************************************************************/
RCODE F_Attribute::allocNameSpace( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT			uiSpaceNeeded;
	FLMUNICODE *	puzTmp;
	
	uiSpaceNeeded = (m_uiNameChars +
						  m_uiNamespaceChars +
						  m_uiValueChars + 3) * sizeof( FLMUNICODE);
	if (uiSpaceNeeded > m_uiTmpSpaceSize)
	{
		if (RC_BAD( rc = f_alloc( uiSpaceNeeded, &puzTmp)))
		{
			goto Exit;
		}
		if (m_puzName != &m_uzTmpSpace [0])
		{
			f_free( &m_puzName);
		}
		m_puzName = puzTmp;
		m_uiTmpSpaceSize = uiSpaceNeeded;
	}
	m_puzNamespace = &m_puzName [m_uiNameChars + 1];
	m_puzValue = &m_puzNamespace [m_uiNamespaceChars + 1];

Exit:

	return( rc);
}
	
/*****************************************************************************
Desc:	Setup an attribute with its namespace, etc.
*****************************************************************************/
#if 0
RCODE F_Attribute::setupAttribute(
	IF_Db *			pDb,
	IF_DOMNode *	pNode
	)
{
	RCODE		rc = NE_FLM_OK;
	
	// Determine if the attribute is a namespace declaration
	
	if (RC_BAD( rc = pNode->isNamespaceDecl( pDb, &m_bIsNamespaceDecl)))
	{
		goto Exit;
	}
	
	// Get the length of the name of the attribute
			
	if (RC_BAD( rc = pNode->getLocalName( pDb, (FLMUNICODE *)NULL,
										0, &m_uiNameChars)))
	{
		goto Exit;
	}
	
	// If it is a namespace declaration, no need to get the namespace URI,
	// we already know what it is, and we will output it with an xmlns prefix
	// Otherwise, we need to get the namespace so we can determine a prefix,
	// if any.  If the namespace is the same namespace as the enclosing
	// element, we do not need to output a prefix.
			
	if (!m_bIsNamespaceDecl)
	{
				
		// Get the number of characters in the namespace of the attribute
				
		if (RC_BAD( rc = pNode->getNamespaceURI( pDb, (FLMUNICODE *)NULL,
											0, &m_uiNamespaceChars)))
		{
			goto Exit;
		}
	}
	
	// Get the number of characters in the attribute's value.
	
	if (RC_BAD( rc = pNode->getUnicodeChars( pDb, &m_uiValueChars)))
	{
		goto Exit;
	}
	
	// Allocate space for the name, namespace, and value
	
	if (RC_BAD( rc = allocNameSpace()))
	{
		goto Exit;
	}

	// Get the attribute name.
	
	if (RC_BAD( rc = pNode->getLocalName( pDb, m_puzName,
											(m_uiNameChars + 1) * sizeof( FLMUNICODE),
											&m_uiNameChars)))
	{
		goto Exit;
	}
	
	// Get the namespace, if necessary
	
	if (m_uiNamespaceChars)
	{
		if (RC_BAD( rc = pNode->getNamespaceURI( pDb, m_puzNamespace,
											(m_uiNamespaceChars + 1) * sizeof( FLMUNICODE),
											&m_uiNamespaceChars)))
		{
			goto Exit;
		}
	}
	
	// Get the value, if any
	
	if (m_uiValueChars)
	{
		if (RC_BAD( rc = pNode->getUnicode( pDb, m_puzValue,
									(m_uiValueChars + 1) * sizeof( FLMUNICODE),
									0, m_uiValueChars, &m_uiValueChars)))
		{
			goto Exit;
		}
	}
	
	// If it is a namespace declaration, the local name must either be
	// "xmlns" or begin with "xmlns:"
	
	if (m_bIsNamespaceDecl)
	{
		
		// Make sure name is "xmlns" or begins with "xmlns:"
		
		if (m_uiNameChars != 5 && m_uiNameChars <= 6)
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_NAMESPACE_DECL);
			goto Exit;
		}
		
		if (!isXMLNS( m_puzName))
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_NAMESPACE_DECL);
			goto Exit;
		}
		else if (m_uiNameChars == 5)
		{
			m_bDefaultNamespaceDecl = TRUE;
		}
		else if (m_puzName [5] != ':')
		{
			rc = RC_SET_AND_ASSERT( NE_FLM_INVALID_NAMESPACE_DECL);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}
#endif

/*****************************************************************************
Desc:	Set the prefix for an attribute.
*****************************************************************************/
#if 0
RCODE F_Attribute::setPrefix( void)
{
	RCODE		rc = NE_FLM_OK;
	
	// If this is a namespace declaration, there should be no prefix in the name.
	
	if (m_bIsNamespaceDecl)
	{
		flmAssert( !m_uiPrefixChars);
	}
	
	// Only need to set a prefix on an attribute if it has a namespace
	// Otherwise, leave it alone - no prefix.
	
	else if (m_uiNamespaceChars)
	{
	
			
		// See if we can find a namespace declaration in either
		// this element's attributes, or any of its parent element
		// attributes.
		
		if (RC_BAD( rc = m_pElement->findPrefix( m_puzNamespace,
										m_uiNamespaceChars, FALSE,
										&m_puzPrefix, &m_uiPrefixChars)))
		{
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}
#endif

/*****************************************************************************
Desc:	Export a unicode string to the string buffer - as UTF8.
*****************************************************************************/
#if 0
FSTATIC RCODE exportUniValue(
	IF_OStream *	pOStream,
	FLMUNICODE *	puzStr,
	FLMUINT			uiStrChars,
	FLMBOOL			bEncodeSpecialChars,
	FLMUINT			uiIndentCount
	)
{
	RCODE			rc = NE_FLM_OK;
	FLMBYTE		ucTmp [4];
	FLMUINT		uiLen;
	FLMUINT		uiCharOffset = 0;
	FLMUNICODE	uzChar;
	FLMBOOL		bIndent = FALSE;
	FLMUINT		uiICount = 0;

	while ( *puzStr && uiCharOffset < uiStrChars)
	{
		uzChar = *puzStr;
		
		// Handle encoding of special characters
		
		if (bEncodeSpecialChars)
		{
			if (uzChar == '<')
			{
				if (RC_BAD( rc = pOStream->write( (void *)"&lt;", 4)))
				{
					goto Exit;
				}
			}
			else if (uzChar == '>')
			{
				if (RC_BAD( rc = pOStream->write( (void *)"&gt;", 4)))
				{
					goto Exit;
				}
			}
			else if (uzChar == '&')
			{
				if (RC_BAD( rc = pOStream->write( (void *)"&amp;", 5)))
				{
					goto Exit;
				}
			}
			else if (uzChar == '\'')
			{
				if (RC_BAD( rc = pOStream->write( (void *)"&apos;", 6)))
				{
					goto Exit;
				}
			}
			else if (uzChar == '"')
			{
				if (RC_BAD( rc = pOStream->write( (void *)"&quot;", 6)))
				{
					goto Exit;
				}
			}
			else
			{
				goto Normal_Encoding;
			}
		}
		else
		{
			
Normal_Encoding:
		
			// Output the character as UTF8.
			
			if (uzChar <= 0x007F)
			{
				// New Line char found.  Need to indent.
				if( uzChar == ASCII_NEWLINE)
				{
					bIndent = TRUE;
				}
				ucTmp [0] = (FLMBYTE)uzChar;
				uiLen = 1;
			}
			else if (*puzStr <= 0x07FF)
			{
				ucTmp [0] = (FLMBYTE)(0xC0 | (FLMBYTE)(uzChar >> 6));
				ucTmp [1] = (FLMBYTE)(0x80 | (FLMBYTE)(uzChar & 0x003F));
				uiLen = 2;
			}
			else
			{
				ucTmp [0] = (FLMBYTE)(0xE0 | (FLMBYTE)(uzChar >> 12));
				ucTmp [1] = (FLMBYTE)(0x80 | (FLMBYTE)((uzChar & 0x0FC0) >> 6));
				ucTmp [2] = (FLMBYTE)(0x80 | (FLMBYTE)(uzChar & 0x003F));
				uiLen = 3;
			}
			if (RC_BAD( rc = pOStream->write( (void *)&ucTmp[0], uiLen)))
			{
				goto Exit;
			}

			if( bIndent && uiIndentCount)
			{
				for( uiICount = uiIndentCount; uiICount; uiICount--)
				{
					if (RC_BAD( rc = pOStream->write( (void *)"\t", 1)))
					{
						goto Exit;
					}
				}
				bIndent = FALSE;
			}
		}
		puzStr++;
		uiCharOffset++;
	}
	
Exit:

	return( rc);
}
#endif
	
/*****************************************************************************
Desc:	Output an attribute to the string buffer.
*****************************************************************************/
#if 0
RCODE F_Attribute::outputAttr(
	IF_OStream *		pOStream)
{
	RCODE		rc = NE_FLM_OK;
	
	if (RC_BAD( rc = pOStream->write( (void *)" ", 1)))
	{
		goto Exit;
	}
	if (m_uiPrefixChars)
	{
		if (RC_BAD( rc = exportUniValue( pOStream, m_puzPrefix, m_uiPrefixChars, FALSE, 0)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pOStream->write( (void *)":", 1)))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = exportUniValue( pOStream, m_puzName, m_uiNameChars, FALSE, 0)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pOStream->write( (void *)"=\"", 2)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = exportUniValue( pOStream, m_puzValue, m_uiValueChars, TRUE, 0)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = pOStream->write( (void *)"\"", 1)))
	{
		goto Exit;
	}
	
Exit:

	return( rc);
}
#endif
	
/*****************************************************************************
Desc:	Allocate a new attribute.
*****************************************************************************/
#if 0
RCODE F_Element::allocAttr(
	F_Attribute **	ppAttr
	)
{
	RCODE		rc = NE_FLM_OK;
	
	if ((*ppAttr = *m_ppAvailAttrs) != NULL)
	{
		*m_ppAvailAttrs = (*ppAttr)->m_pNext;
		(*ppAttr)->reset( this);
	}
	else
	{
		if ((*ppAttr = f_new F_Attribute( this)) == NULL)
		{
			rc = RC_SET( NE_FLM_MEM);
			goto Exit;
		}
	}
	
Exit:

	return( rc);
}
#endif
	
/*****************************************************************************
Desc:	Save an attribute in an element.  Put at end of list.
*****************************************************************************/
#if 0
RCODE F_Element::saveAttribute(
	IF_Db *			pDb,
	IF_DOMNode *	pNode
	)
{
	RCODE				rc = NE_FLM_OK;
	F_Attribute *	pAttr = NULL;

	if (RC_BAD( rc = allocAttr( &pAttr)))
	{
		goto Exit;
	}
	
	// Set up the attribute
	
	if (RC_BAD( rc = pAttr->setupAttribute( pDb, pNode)))
	{
		goto Exit;
	}
	
	// Put attribute at end of list of attributes.
	
	pAttr->m_pNext = NULL;
	if (m_pLastAttr)
	{
		m_pLastAttr->m_pNext = pAttr;
	}
	else
	{
		m_pFirstAttr = pAttr;
	}
	m_pLastAttr = pAttr;
	
	// Set pAttr to NULL so it won't be made available at exit.
	
	pAttr = NULL;
	
Exit:

	if (pAttr)
	{
		makeAttrAvail( pAttr);
	}

	return( rc);
}
#endif

/*****************************************************************************
Desc:	Allocate space for the element's name and namespace.
*****************************************************************************/
#if 0
RCODE F_Element::allocNameSpace( void)
{
	RCODE				rc = NE_FLM_OK;	
	FLMUINT			uiSpaceNeeded;
	FLMUNICODE *	puzTmp;
	
	uiSpaceNeeded = (m_uiNameChars + m_uiNamespaceChars + 2) * sizeof( FLMUNICODE);
	
	// Allocate space for the name and namespace
	
	if (uiSpaceNeeded > m_uiTmpSpaceSize)
	{
		
		if (RC_BAD( rc = f_alloc( uiSpaceNeeded, &puzTmp)))
		{
			goto Exit;
		}
		if (m_puzName != &m_uzTmpSpace [0])
		{
			f_free( &m_puzName);
		}
		m_puzName = puzTmp;
		m_uiTmpSpaceSize = uiSpaceNeeded;
	}
	m_puzNamespace = &m_puzName [m_uiNameChars + 1];
	
Exit:

	return( rc);
}
#endif
	
/*****************************************************************************
Desc:	Setup an element with its namespace, etc.
*****************************************************************************/
#if 0
RCODE F_Element::setupElement(
	IF_Db *			pDb,
	IF_DOMNode *	pNode
	)
{
	RCODE				rc = NE_FLM_OK;
	IF_DOMNode *	pAttrNode = NULL;
	F_Attribute *	pAttr;
	
	// Get the length of the name of the element
			
	if (RC_BAD( rc = pNode->getLocalName( pDb, (FLMUNICODE *)NULL,
										0, &m_uiNameChars)))
	{
		goto Exit;
	}
	
	// Get the number of characters in the namespace of the element
				
	if (RC_BAD( rc = pNode->getNamespaceURI( pDb, (FLMUNICODE *)NULL,
										0, &m_uiNamespaceChars)))
	{
		goto Exit;
	}
	
	if (RC_BAD( rc = allocNameSpace()))
	{
		goto Exit;
	}

	// Get the element name.
	
	if (RC_BAD( rc = pNode->getLocalName( pDb, m_puzName,
											(m_uiNameChars + 1) * sizeof( FLMUNICODE),
											&m_uiNameChars)))
	{
		goto Exit;
	}
	
	// Get the namespace, if necessary
	
	if (m_uiNamespaceChars)
	{
		if (RC_BAD( rc = pNode->getNamespaceURI( pDb, m_puzNamespace,
											(m_uiNamespaceChars + 1) * sizeof( FLMUNICODE),
											&m_uiNamespaceChars)))
		{
			goto Exit;
		}
	}
	
	// See if the node has any attributes.
			
	for (;;)
	{
		rc = (RCODE)(pAttrNode
						? pAttrNode->getNextSibling( pDb, &pAttrNode)
						: pNode->getFirstAttribute( pDb, &pAttrNode));
		if (RC_BAD( rc))
		{
			if (rc == NE_FLM_DOM_NODE_NOT_FOUND)
			{
				rc = NE_FLM_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}
		if (RC_BAD( rc = saveAttribute( pDb, pAttrNode)))
		{
			goto Exit;
		}
	}
	
	// Get the prefix for the element
	
	if (RC_BAD( rc = setPrefix()))
	{
		goto Exit;
	}
	
	// Set the prefix for every attribute
	
	pAttr = m_pFirstAttr;
	while (pAttr)
	{
		if (RC_BAD( rc = pAttr->setPrefix()))
		{
			goto Exit;
		}
		pAttr = pAttr->m_pNext;
	}
	
Exit:

	if (pAttrNode)
	{
		pAttrNode->Release();
	}

	return( rc);
}
#endif

/*****************************************************************************
Desc:	Add an attribute that is a namespace to an element.
*****************************************************************************/
#if 0
RCODE F_Element::addNamespaceDecl(
	FLMUNICODE *	puzPrefix,
	FLMUINT			uiPrefixChars,
	FLMUNICODE *	puzNamespace,
	FLMUINT			uiNamespaceChars,
	F_Attribute **	ppAttr
	)
{
	RCODE				rc = NE_FLM_OK;
	F_Attribute *	pAttr = NULL;
	
	// If uiPrefixChars is zero, we are being asked to create a default
	// namespace.  But that can only be output once in the element,
	// so make sure it is not already declared.  If it is, do nothing.
	
	if (!uiPrefixChars)
	{
		pAttr = m_pFirstAttr;
		while (pAttr && !pAttr->m_bDefaultNamespaceDecl)
		{
			pAttr = pAttr->m_pNext;
		}
		if (pAttr)
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = allocAttr( &pAttr)))
	{
		goto Exit;
	}
	pAttr->m_bIsNamespaceDecl = TRUE;
	
	// name will be "xmlns:<prefixname>" or, in the case of no namespace, "xmlns"
	
	if (!uiPrefixChars)
	{
		
		// "xmlns" - but make sure not already declared.
		
		pAttr->m_uiNameChars = 5;
		pAttr->m_bDefaultNamespaceDecl = TRUE;
	}
	else
	{
		
		// "xmlns:<prefixname>"
		
		pAttr->m_uiNameChars = uiPrefixChars + 6;
	}
	pAttr->m_uiNamespaceChars = 0;
	pAttr->m_uiValueChars = uiNamespaceChars;
	
	if (RC_BAD( rc = pAttr->allocNameSpace()))
	{
		goto Exit;
	}
	
	// Always output "xmlns" as the first part of the name

	f_memcpy( pAttr->m_puzName, gv_puzNamespaceDeclPrefix, 5 * sizeof( FLMUNICODE));
	if (uiPrefixChars)
	{
		pAttr->m_puzName [5] = ':';
		f_memcpy( &pAttr->m_puzName [6], puzPrefix, uiPrefixChars * sizeof( FLMUNICODE));
		pAttr->m_puzName [6 + uiPrefixChars] = 0;
	}
	else
	{
		pAttr->m_puzName [5] = 0;
	}
	if (uiNamespaceChars)
	{
		f_memcpy( pAttr->m_puzValue, puzNamespace, 
			uiNamespaceChars * sizeof( FLMUNICODE));
	}
	pAttr->m_puzValue [pAttr->m_uiValueChars] = 0;
	
	// Put new namespace decl at front of list of attributes.
	
	if ((pAttr->m_pNext = m_pFirstAttr) == NULL)
	{
		m_pLastAttr = pAttr;
	}
	m_pFirstAttr = pAttr;
	*ppAttr = pAttr;
	
	// Set pAttr to NULL so that it won't be made available at exit.
	
	pAttr = NULL;
		
Exit:

	if (pAttr)
	{
		makeAttrAvail( pAttr);
	}
	return( rc);
}
#endif

/*****************************************************************************
Desc:	Generate a random prefix, ensure that it is not defined anywhere
		in the path.
*****************************************************************************/
#if 0
void F_Element::genPrefix(
	FLMUNICODE *	puzPrefix,
	FLMUINT *		puiPrefixChars
	)
{
	FLMUINT			uiTmp;
	FLMUINT			uiPrefixChars;
	FLMUNICODE *	puzTmp;
	F_Attribute *	pAttr;
	F_Element *		pElement;

	puzPrefix [0] = 'p';
	puzPrefix [1] = 'r';
	puzPrefix [2] = 'f';
	puzPrefix [3] = 'x';	
	for (;;)
	{
		
		// Append the number in reverse digit order - it really doesn't matter
		// because we're just trying to generate a unique prefix number.
		
		puzTmp = &puzPrefix [4];
		uiPrefixChars = 4;
		uiTmp = *m_puiNextPrefixNum;
		do
		{
			*puzTmp++ = (FLMUNICODE)((uiTmp % 10) + '0');
			uiPrefixChars++;
			uiTmp /= 10;
		} while (uiTmp);
		
		// See if the prefix is defined.
		
		pAttr = m_pFirstAttr;
		pElement = this;
		while (pAttr)
		{
			if (pAttr->m_bIsNamespaceDecl &&
				 pAttr->m_uiNameChars > 6 &&
				 pAttr->m_uiNameChars - 6 == uiPrefixChars &&
				 f_memcmp( puzPrefix, &pAttr->m_puzName [6],
					uiPrefixChars * sizeof( FLMUNICODE)) == 0)
			{
				break;
			}
			if ((pAttr = pAttr->m_pNext) == NULL)
			{
				pElement = pElement->m_pParentElement;
				while (pElement && !pElement->m_pFirstAttr)
				{
					pElement = pElement->m_pParentElement;
				}
				if (!pElement)
				{
					break;
				}
				pAttr = pElement->m_pFirstAttr;
			}
		}
		
		// If the prefix was not defined, we can use it.
		
		if (!pAttr)
		{
			break;
		}
		(*m_puiNextPrefixNum)++;
	}
	puzPrefix [uiPrefixChars] = 0;
	*puiPrefixChars = uiPrefixChars;
}
#endif

/*****************************************************************************
Desc:	Find a prefix for a namespace
*****************************************************************************/
#if 0
RCODE F_Element::findPrefix(
	FLMUNICODE *	puzNamespace,
	FLMUINT			uiNamespaceChars,
	FLMBOOL			bForElement,
	FLMUNICODE **	ppuzPrefix,
	FLMUINT *		puiPrefixChars)
{
	RCODE				rc = NE_FLM_OK;
	F_Attribute *	pAttr = m_pFirstAttr;
	F_Element *		pElement = this;
	FLMUNICODE		uzPrefix [50];
	FLMUINT			uiPrefixChars;

	for (;;)
	{
		if ( pAttr)
		{
			if (pAttr->m_bIsNamespaceDecl &&
				uiNamespaceChars == pAttr->m_uiValueChars &&
		 		(!uiNamespaceChars ||
				f_memcmp( puzNamespace, pAttr->m_puzValue,
	 				uiNamespaceChars * sizeof( FLMUNICODE)) == 0))
			{
				
				// Don't set the prefix if it is the default namespace.
				
				if (!pAttr->m_bDefaultNamespaceDecl)
				{
					// Prefix comes after the "xmlns:"
					
					*ppuzPrefix = &pAttr->m_puzName [6];
					*puiPrefixChars = pAttr->m_uiNameChars - 6;
					goto Exit;
				}
				
				// Default namespace is only OK for elements,
				// but not attributes.  We don't want to count
				// attributes as having been "found" if they matched
				// the default namespace.  This routine is only called
				// for attributes if the attribute namepace is non-empty.
				
				else if (bForElement)
				{
					goto Exit;
				}
			}
			pAttr = pAttr->m_pNext;
		}
		if ( !pAttr)
		{
			pElement = pElement->m_pParentElement;
			while (pElement && !pElement->m_pFirstAttr)
			{
				pElement = pElement->m_pParentElement;
			}
			if (!pElement)
			{
				break;
			}
			pAttr = pElement->m_pFirstAttr;
		}
	}
	
	// If namespaces is empty, the only declaration that is legal is
	// a default namespace declaration.
	
	if (!uiNamespaceChars)
	{
		if (RC_BAD( rc = addNamespaceDecl( NULL, 0, NULL, 0, &pAttr)))
		{
			goto Exit;
		}
	}
	else
	{
		
		// Manufacture a prefix that is not used in the hierarchy yet.
			
		genPrefix( uzPrefix, &uiPrefixChars);
		if (RC_BAD( rc = addNamespaceDecl( uzPrefix, uiPrefixChars, puzNamespace,
									uiNamespaceChars, &pAttr)))
		{
			goto Exit;
		}
		
		*ppuzPrefix = &pAttr->m_puzName [6];
		*puiPrefixChars = pAttr->m_uiNameChars - 6;
	}
			
Exit:

	return( rc);
}
#endif

/*****************************************************************************
Desc:	Output the element name, with its attributes - this marks the
		beginning of the element.
*****************************************************************************/
#if 0
RCODE F_Element::outputElem(
	IF_OStream *	pOStream,
	FLMBOOL			bStartOfElement,
	FLMBOOL			bEndOfElement,
	FLMBOOL			bAddNewLine)
{
	RCODE				rc = NE_FLM_OK;
	F_Attribute *	pAttr;
	F_Attribute *	pPrevAttr;
	FLMUINT			uiIndentCount = 0;
	FLMBOOL			bEndNode;

	bEndNode = ( m_bIsDocumentRoot && !bStartOfElement);
	if( bAddNewLine && ( !m_bIsDocumentRoot || bEndNode))
	{
			if (RC_BAD( rc = pOStream->write( (void *)"\n", 1)))
			{
				goto Exit;
			}
			for( uiIndentCount = 0; uiIndentCount < m_uiIndentCount; uiIndentCount++)
			{
				if (RC_BAD( rc = pOStream->write( (void *)"\t", 1)))
				{
					goto Exit;
				}
			}		
	}

	// Output the element name
	if ( bStartOfElement)
	{

		if (RC_BAD( rc = pOStream->write( (void *)"<", 1)))
		{
			goto Exit;
		}
	}
	else
	{
		
		if (RC_BAD( rc = pOStream->write( (void *)"</", 2)))
		{
			goto Exit;
		}
	}
	
	if (m_uiPrefixChars)
	{
		if (RC_BAD( rc = exportUniValue( pOStream, m_puzPrefix, m_uiPrefixChars, FALSE, 0)))
		{
			goto Exit;
		}
		if (RC_BAD( rc = pOStream->write( (void *)":", 1)))
		{
			goto Exit;
		}
	}
	
	if (RC_BAD( rc = exportUniValue( pOStream, m_puzName, m_uiNameChars, FALSE, 0)))
	{
		goto Exit;
	}
	
	if (bStartOfElement)
	{
		
		// Output the attributes.  As we go, remove any attributes that are
		// not namespace declarations.  They are not needed after this.
		
		pPrevAttr = NULL;
		pAttr = m_pFirstAttr;
		while (pAttr)
		{
			if (RC_BAD( rc = pAttr->outputAttr( pOStream)))
			{
				goto Exit;
			}
			
			if (!pAttr->m_bIsNamespaceDecl)
			{
				if (pPrevAttr)
				{
					pPrevAttr->m_pNext = pAttr->m_pNext;
					makeAttrAvail( pAttr);
					pAttr = pPrevAttr->m_pNext;
				}
				else
				{
					m_pFirstAttr = pAttr->m_pNext;
					makeAttrAvail( pAttr);
					pAttr = m_pFirstAttr;
				}
				
				// See if we deleted the last attribute in the list.
				
				if (!pAttr)
				{
					m_pLastAttr = pPrevAttr;
				}
			}
			else
			{
				pPrevAttr = pAttr;
				pAttr = pAttr->m_pNext;
			}
		}
	}
	
	// Close out the element
	if (RC_BAD( rc = (RCODE)(bStartOfElement && bEndOfElement
					 ? pOStream->write( (void *)"/>", 2)
					 : pOStream->write( (void *)">", 1))))
	{
		goto Exit;
	}

	if ( bAddNewLine && bEndNode)
	{
		if (RC_BAD( rc = pOStream->write( (void *)"\n", 1)))
		{
			goto Exit;
		}
	}


Exit:

	return( rc);
}
#endif

/*****************************************************************************
Desc:	Output Data that is contained on an element node.
*****************************************************************************/
#if 0
RCODE F_Element::outputLocalData( 
	IF_OStream *				pOStream,
	IF_DOMNode *				pDbNode,
	IF_Db *						ifpDb,
	eExportFormatType		eFormatType,
	FLMUINT						uiIndentCount)
{
	RCODE					rc = NE_FLM_OK;
	FLMUNICODE			uzTmpData [150];
	FLMUNICODE *		puzData = &uzTmpData [0];
	FLMUINT				uiDataBufSize = sizeof( uzTmpData);
	FLMUINT				uiChars;
	 
	if (RC_BAD( rc = pDbNode->getUnicodeChars( ifpDb, &uiChars)))
	{
		 goto Exit;
	}

	
	if (uiDataBufSize < (uiChars + 1) * sizeof( FLMUNICODE))
	{
		FLMUNICODE *	puzNew;
		 
		if (RC_BAD( rc = f_alloc( (uiChars + 1) * sizeof( FLMUNICODE),
									 &puzNew)))
		{
			goto Exit;
		}
		if (puzData != &uzTmpData [0])
		{
		 f_free( &puzData);
		}
		puzData = puzNew;
		uiDataBufSize = (uiChars + 1) * sizeof( FLMUNICODE);
	}
	if (RC_BAD( rc = pDbNode->getUnicode( ifpDb, puzData,
						 uiDataBufSize, 0, uiChars, &uiChars)))
	{
		 goto Exit;
	}
	 
	// Output the value.
	if (RC_BAD( rc = exportUniValue( pOStream, puzData, uiChars, TRUE, 
		eFormatType >= FLM_EXPORT_INDENT_DATA ? uiIndentCount : 0)))
	{
		goto Exit;
	}

Exit:

	return ( rc);
}
#endif

/*****************************************************************************
Desc:	Outputs a UTF8 stream of XML, starting at the specified node.  Node and
		all of its descendant nodes are output.
*****************************************************************************/
#if 0
RCODE XFLMAPI F_Db::exportXML(
	IF_DOMNode *			pStartNode,
	IF_OStream *			pOStream,
	eExportFormatType		eFormatType)
{
	RCODE					rc = NE_FLM_OK;
	F_Element *			pAvailElements = NULL;
	F_Element *			pTmpElement;
	F_Attribute *		pAvailAttrs = NULL;
	F_Attribute *		pTmpAttr;
	FLMUNICODE			uzTmpData [150];
	FLMUNICODE *		puzData = &uzTmpData [0];
	FLMUINT				uiDataBufSize = sizeof( uzTmpData);
	IF_DOMNode *		pDbNode = NULL;
	eDomNodeType		ePrevNodeType;
	F_Element *			pCurrElement = NULL;
	FLMUINT				uiNextPrefixNum = 0;
	FLMBOOL				bStartOfDocument = TRUE;
	FLMBOOL				bShouldFormat = FALSE;	
	FLMBOOL				bIsDataLocal = FALSE;
	FLMUINT				uiIndentCount = 0;
	FLMUINT 				uiICount = 0;

	// This routine should only be called if the node type is element node.
	
	flmAssert( pStartNode->getNodeType() == ELEMENT_NODE);
	
	ePrevNodeType = ELEMENT_NODE;
	pDbNode = pStartNode;
	pDbNode->AddRef();
	
	for (;;)
	{
		// Output the current node, depending on its type.
		
		if( pDbNode->getNodeType() == ELEMENT_NODE)
		{
			if (pAvailElements)
			{
				pTmpElement = pAvailElements;
				pAvailElements = pAvailElements->getNext();
				pTmpElement->reset( pCurrElement, &pAvailAttrs, &uiNextPrefixNum);
			}
			else
			{
				if ((pTmpElement = f_new F_Element( pCurrElement, &pAvailAttrs,
													&uiNextPrefixNum)) == NULL)
				{
					rc = RC_SET( NE_FLM_MEM);
					goto Exit;
				}
			}
			
			pCurrElement = pTmpElement;
			
			if (RC_BAD(  rc = pCurrElement->setupElement( (IF_Db *)this, pDbNode)))
			{
				goto Exit;
			}

			if( eFormatType >= FLM_EXPORT_INDENT)
			{
				pCurrElement->setIndentCount(uiIndentCount);
			}
			
			if( pDbNode == pStartNode)
			{
				pCurrElement->setDocumentRoot( TRUE);
			}

			// Only want a New Line and tabs for Element if:
			// 1) New Line format is indicated
			// 2) Previous Element Was NOT Data
			bShouldFormat = ( (eFormatType >= FLM_EXPORT_NEW_LINE) && 
										(ePrevNodeType != DATA_NODE))
									? TRUE
									: FALSE;

			if( RC_BAD( rc = pDbNode->isDataLocalToNode( (IF_Db *)this,
				&bIsDataLocal)))
			{
				goto Exit;	
			}
			
			if( bIsDataLocal)
			{
				 if( RC_BAD( rc = pCurrElement->outputElem( pOStream, 
					 TRUE, FALSE, bShouldFormat)))
				 {
					  goto Exit;
				 }
				 
				 pCurrElement->outputLocalData( pOStream, 
											pDbNode,
											(IF_Db *)this,
											eFormatType,
											uiIndentCount);

			}

			if( RC_OK( rc = pDbNode->getFirstChild( (IF_Db *)this, &pDbNode)))
			{
				if( !bIsDataLocal && RC_BAD( rc = pCurrElement->outputElem( 
					pOStream, TRUE, FALSE, bShouldFormat)))
				{
					goto Exit;
				}

				bStartOfDocument = FALSE;
				uiIndentCount++;
				ePrevNodeType = ELEMENT_NODE;
            continue;
			}

			if( rc != NE_FLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			// Write out the "/>" for the element, because it had no
			// child nodes.

			if( bIsDataLocal)
			{
				 if( RC_BAD( rc = pCurrElement->outputElem( pOStream,
					 FALSE, TRUE, bShouldFormat)))
				 {
					  goto Exit;
				 }
			}
			else
			{
				 if( RC_BAD( rc = pCurrElement->outputElem( pOStream,
					 TRUE, TRUE, bShouldFormat)))
				 {
					  goto Exit;
				 }
			}
			
			// We are now done with this element
			
			ePrevNodeType = ELEMENT_NODE;
			pTmpElement = pCurrElement;
			pCurrElement = pCurrElement->getParentElement();
			pTmpElement->makeAvail( &pAvailElements);
			
			if( !pCurrElement)
			{
				break;
			}
				
Get_Element_Sibling:

			// See if we have a sibling.  Go up tree until we find
			// a node that has a sibling.
			
			for( ;;)
			{
				if( RC_OK( rc = pDbNode->getNextSibling( (IF_Db *)this, &pDbNode)))
				{
					break;
				}
				
				if( rc != NE_FLM_DOM_NODE_NOT_FOUND)
				{
					goto Exit;
				}

				// Need to close previous element

				if( uiIndentCount)
				{
					uiIndentCount--;
				}
				
				if( RC_BAD( rc = pCurrElement->outputElem( pOStream, FALSE, TRUE,
											eFormatType >= FLM_EXPORT_NEW_LINE 
													? TRUE 
													: FALSE)))
				{
					goto Exit;
				}
				
				if( RC_BAD( rc = pDbNode->getParentNode( (IF_Db *)this, &pDbNode)))
				{
					if( rc == NE_FLM_DOM_NODE_NOT_FOUND)
					{
						// There should be a parent node at this point!
					
						rc = RC_SET_AND_ASSERT( NE_FLM_DATA_ERROR);
					}
					
					goto Exit;
				}
				
				pTmpElement = pCurrElement;
				pCurrElement = pCurrElement->getParentElement();
				pTmpElement->makeAvail( &pAvailElements);
				
				if( !pCurrElement)
				{
					pDbNode->Release();
					pDbNode = NULL;
					goto Exit;
				}
			}
		}
		else
		{
			// Only output data, comment, and cdata nodes.
			
			if( pDbNode->getNodeType() == DATA_NODE || 
				 pDbNode->getNodeType() == COMMENT_NODE ||
				 pDbNode->getNodeType() == CDATA_SECTION_NODE)
			{
				FLMUINT	uiChars;

				if( RC_BAD( rc = pDbNode->getUnicodeChars( (IF_Db *)this, 
					&uiChars)))
				{
					goto Exit;
				}
				
				if( uiDataBufSize < (uiChars + 1) * sizeof( FLMUNICODE))
				{
					FLMUNICODE *	puzNew;
					
					if( RC_BAD( rc = f_alloc( (uiChars + 1) * sizeof( FLMUNICODE),
												&puzNew)))
					{
						goto Exit;
					}
					
					if( puzData != &uzTmpData [0])
					{
						f_free( &puzData);
					}
					
					puzData = puzNew;
					uiDataBufSize = (uiChars + 1) * sizeof( FLMUNICODE);
				}
				
				if( RC_BAD( rc = pDbNode->getUnicode( (IF_Db *)this, puzData,
										uiDataBufSize, 0, uiChars, &uiChars)))
				{
					goto Exit;
				}
				
				if( pDbNode->getNodeType() == DATA_NODE)
				{
					// Output the value
					
					if (RC_BAD( rc = exportUniValue( pOStream, puzData, uiChars, 
						TRUE, eFormatType >= FLM_EXPORT_INDENT_DATA 
												? uiIndentCount 
												: 0)))
					{
						goto Exit;
					}

					ePrevNodeType = DATA_NODE;
				}
				else if( pDbNode->getNodeType() == COMMENT_NODE)
				{
					//If Comment Node follows Data Node do not add new line
					
					if( eFormatType >= FLM_EXPORT_INDENT_DATA &&
							 ePrevNodeType != DATA_NODE)
					{
						if (RC_BAD( rc = pOStream->write( (void *)"\n", 1)))
						{
							goto Exit;
						}
						
                  for( uiICount = 0; uiICount < uiIndentCount; uiICount++)
						{
							if (RC_BAD( rc = pOStream->write( (void *)"\t", 1)))
							{
								goto Exit;
							}
						}		
					}

					// Output the beginning of a comment
					
					if (RC_BAD( rc = pOStream->write( (void *)"<!--", 4)))
					{
						goto Exit;
					}
					
					// Output the comment value - as is, no encoding
					
					if( RC_BAD( rc = exportUniValue( pOStream, puzData, uiChars,
						FALSE, eFormatType >= FLM_EXPORT_INDENT_DATA 
										? uiIndentCount 
										: 0)))
					{
						goto Exit;
					}
					
					// Output the end of the comment
					
					if( RC_BAD( rc = pOStream->write( (void *)"-->", 3)))
					{
						goto Exit;
					}
					
					ePrevNodeType = COMMENT_NODE;
				}
				else
				{
					// Output the beginning of a cdata section
					
					if( RC_BAD( rc = pOStream->write( (void *)"<![CDATA[", 9)))
					{
						goto Exit;
					}
					
					// Output the cdata value - as is, no encoding
					
					if( RC_BAD( rc = exportUniValue( pOStream, puzData,
						uiChars, FALSE, 0)))
					{
						goto Exit;
					}
					
					// Output the end of the cdata section
					
					if( RC_BAD( rc = pOStream->write( (void *)"]]>", 3)))
					{
						goto Exit;
					}
					
					ePrevNodeType = CDATA_SECTION_NODE;
				}
			}
			
			// Have a data node, or comment node probably
			// In any case, see if there are any sibling nodes.
			// If not, go back to enclosing element node.
			
			if( RC_OK( rc = pDbNode->getNextSibling( (IF_Db *)this, &pDbNode)))
			{
				continue;
			}
			
			if( rc != NE_FLM_DOM_NODE_NOT_FOUND)
			{
				goto Exit;
			}
			
			// Go back up to enclosing element
				
			if( RC_BAD( rc = pDbNode->getParentNode( (IF_Db *)this, &pDbNode)))
			{
				// There better be a parent node or we have a corruption!
					
				if( rc == NE_FLM_DOM_NODE_NOT_FOUND)
				{
					rc = RC_SET( NE_FLM_DATA_ERROR);
				}
				
				goto Exit;
			}
				
			// Parent node better be an element
				
			if( pDbNode->getNodeType() != ELEMENT_NODE)
			{
				rc = RC_SET_AND_ASSERT( NE_FLM_DATA_ERROR);
				goto Exit;
			}
			
			// If we were traversing the attributes of an element,
			// we need to now go back and get its child nodes.
			
			// Write out the </elementname> for the element
			
			if( RC_BAD( rc = pCurrElement->outputElem( pOStream, 
				FALSE, TRUE, FALSE)))
			{
				goto Exit;
			}
			
			// We are now done with this element
			
			if( uiIndentCount)
			{
					uiIndentCount--;
			}
			
			ePrevNodeType = ELEMENT_NODE;
			pTmpElement = pCurrElement;
			pCurrElement = pCurrElement->getParentElement();
			pTmpElement->makeAvail( &pAvailElements);
			
			if( !pCurrElement)
			{
				break;
			}
				
			goto Get_Element_Sibling;
		}
	}

Exit:

	if( puzData != &uzTmpData [0])
	{
		f_free( &puzData);
	}
	
	while( pCurrElement)
	{
		pTmpElement = pCurrElement;
		pCurrElement = pCurrElement->getParentElement();
		delete pTmpElement;
	}

	while( pAvailElements)
	{
		pTmpElement = pAvailElements;
		pAvailElements = pAvailElements->getNext();
		delete pTmpElement;
	}

	while( pAvailAttrs)
	{
		pTmpAttr = pAvailAttrs;
		pAvailAttrs = pAvailAttrs->getNext();
		delete pTmpAttr;
	}
	
	if( pDbNode)
	{
		pDbNode->Release();
	}

	return( rc);
}
#endif
