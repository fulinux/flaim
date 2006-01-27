//-------------------------------------------------------------------------
// Desc:	XML Wrapper
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
// $Id: fxml.cpp 12334 2006-01-23 12:45:35 -0700 (Mon, 23 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_UNIX
	#define CRLFSTR					"\n"
	#define CRLFSTRLEN				1
#else
	#define CRLFSTR					"\r\n"
	#define CRLFSTRLEN				2
#endif

#define F_XML_BASE_CHAR				0x01
#define F_XML_IDEOGRAPHIC			0x02
#define F_XML_COMBINING_CHAR		0x04
#define F_XML_DIGIT					0x08
#define F_XML_EXTENDER				0x10
#define F_XML_WHITESPACE			0x20

typedef struct
{
	char *			pszEntity;
	FLMUINT			uiValue;
} CharEntity;

typedef struct
{
	FLMUNICODE		uLowChar;
	FLMUNICODE		uHighChar;
	FLMUINT16		ui16Flag;
} CHAR_TBL;

extern char * FlmReservedTags[];

CHAR_TBL charTbl[] = 
{
	{0x0041, 0x005A, F_XML_BASE_CHAR},
	{0x0061, 0x007A, F_XML_BASE_CHAR},
	{0x00C0, 0x00D6, F_XML_BASE_CHAR},
	{0x00D8, 0x00F6, F_XML_BASE_CHAR},
	{0x00F8, 0x00FF, F_XML_BASE_CHAR},
	{0x0100, 0x0131, F_XML_BASE_CHAR},
	{0x0134, 0x013E, F_XML_BASE_CHAR},
	{0x0141, 0x0148, F_XML_BASE_CHAR},
	{0x014A, 0x017E, F_XML_BASE_CHAR},
	{0x0180, 0x01C3, F_XML_BASE_CHAR},
	{0x01CD, 0x01F0, F_XML_BASE_CHAR},
	{0x01F4, 0x01F5, F_XML_BASE_CHAR},
	{0x01FA, 0x0217, F_XML_BASE_CHAR},
	{0x0250, 0x02A8, F_XML_BASE_CHAR},
	{0x02BB, 0x02C1, F_XML_BASE_CHAR},
	{0x0386, 0x0386, F_XML_BASE_CHAR},
	{0x0388, 0x038A, F_XML_BASE_CHAR},
	{0x038C, 0x038C, F_XML_BASE_CHAR},
	{0x038E, 0x03A1, F_XML_BASE_CHAR},
	{0x03A3, 0x03CE, F_XML_BASE_CHAR},
	{0x03D0, 0x03D6, F_XML_BASE_CHAR},
	{0x03DA, 0x03DA, F_XML_BASE_CHAR},
	{0x03DC, 0x03DC, F_XML_BASE_CHAR},
	{0x03DE, 0x03DE, F_XML_BASE_CHAR},
	{0x03E0, 0x03E0, F_XML_BASE_CHAR},
	{0x03E2, 0x03F3, F_XML_BASE_CHAR},
	{0x0401, 0x040C, F_XML_BASE_CHAR},
	{0x040E, 0x044F, F_XML_BASE_CHAR},
	{0x0451, 0x045C, F_XML_BASE_CHAR},
	{0x045E, 0x0481, F_XML_BASE_CHAR},
	{0x0490, 0x04C4, F_XML_BASE_CHAR},
	{0x04C7, 0x04C8, F_XML_BASE_CHAR},
	{0x04CB, 0x04CC, F_XML_BASE_CHAR},
	{0x04D0, 0x04EB, F_XML_BASE_CHAR},
	{0x04EE, 0x04F5, F_XML_BASE_CHAR},
	{0x04F8, 0x04F9, F_XML_BASE_CHAR},
	{0x0531, 0x0556, F_XML_BASE_CHAR},
	{0x0559, 0x0559, F_XML_BASE_CHAR},
	{0x0561, 0x0586, F_XML_BASE_CHAR},
	{0x05D0, 0x05EA, F_XML_BASE_CHAR},
	{0x05F0, 0x05F2, F_XML_BASE_CHAR},
	{0x0621, 0x063A, F_XML_BASE_CHAR},
	{0x0641, 0x06B7, F_XML_BASE_CHAR},
	{0x06BA, 0x06BE, F_XML_BASE_CHAR},
	{0x06C0, 0x06CE, F_XML_BASE_CHAR},
	{0x06D0, 0x06D3, F_XML_BASE_CHAR},
	{0x06D5, 0x06D5, F_XML_BASE_CHAR},
	{0x06E5, 0x06E6, F_XML_BASE_CHAR},
	{0x0905, 0x0939, F_XML_BASE_CHAR},
	{0x093D, 0x093D, F_XML_BASE_CHAR},
	{0x0958, 0x0961, F_XML_BASE_CHAR},
	{0x0985, 0x098C, F_XML_BASE_CHAR},
	{0x098F, 0x0990, F_XML_BASE_CHAR},
	{0x0993, 0x09A8, F_XML_BASE_CHAR},
	{0x09AA, 0x09B0, F_XML_BASE_CHAR},
	{0x09B2, 0x09B2, F_XML_BASE_CHAR},
	{0x09B6, 0x09B9, F_XML_BASE_CHAR},
	{0x0061, 0x007A, F_XML_BASE_CHAR},
	{0x09DC, 0x09DD, F_XML_BASE_CHAR},
	{0x09DF, 0x09E1, F_XML_BASE_CHAR},
	{0x09F0, 0x09F1, F_XML_BASE_CHAR},
	{0x0A05, 0x0A0A, F_XML_BASE_CHAR},
	{0x0A0F, 0x0A10, F_XML_BASE_CHAR},
	{0x0A13, 0x0A28, F_XML_BASE_CHAR},
	{0x0A2A, 0x0A30, F_XML_BASE_CHAR},
	{0x0A32, 0x0A33, F_XML_BASE_CHAR},
	{0x0A35, 0x0A36, F_XML_BASE_CHAR},
	{0x0A38, 0x0A39, F_XML_BASE_CHAR},
	{0x0A59, 0x0A5C, F_XML_BASE_CHAR},
	{0x0A5E, 0x0A5E, F_XML_BASE_CHAR},
	{0x0A72, 0x0A74, F_XML_BASE_CHAR},
	{0x0A85, 0x0A8B, F_XML_BASE_CHAR},
	{0x0A8D, 0x0A8D, F_XML_BASE_CHAR},
	{0x0A8F, 0x0A91, F_XML_BASE_CHAR},
	{0x0A93, 0x0AA8, F_XML_BASE_CHAR},
	{0x0AAA, 0x0AB0, F_XML_BASE_CHAR},
	{0x0AB2, 0x0AB3, F_XML_BASE_CHAR},
	{0x0AB5, 0x0AB9, F_XML_BASE_CHAR},
	{0x0ABD, 0x0ABD, F_XML_BASE_CHAR},
	{0x0AE0, 0x0AE0, F_XML_BASE_CHAR},
	{0x0B05, 0x0B0C, F_XML_BASE_CHAR},
	{0x0B0F, 0x0B10, F_XML_BASE_CHAR},
	{0x0B13, 0x0B28, F_XML_BASE_CHAR},
	{0x0B2A, 0x0B30, F_XML_BASE_CHAR},
	{0x0B32, 0x0B33, F_XML_BASE_CHAR},
	{0x0B36, 0x0B39, F_XML_BASE_CHAR},
	{0x0B3D, 0x0B3D, F_XML_BASE_CHAR},
	{0x0B5C, 0x0B5D, F_XML_BASE_CHAR},
	{0x0B5F, 0x0B61, F_XML_BASE_CHAR},
	{0x0B85, 0x0B8A, F_XML_BASE_CHAR},
	{0x0B8E, 0x0B90, F_XML_BASE_CHAR},
	{0x0B92, 0x0B95, F_XML_BASE_CHAR},
	{0x0B99, 0x0B9A, F_XML_BASE_CHAR},
	{0x0B9C, 0x0B9C, F_XML_BASE_CHAR},
	{0x0B9E, 0x0B9F, F_XML_BASE_CHAR},
	{0x0BA3, 0x0BA4, F_XML_BASE_CHAR},
	{0x0BA8, 0x0BAA, F_XML_BASE_CHAR},
	{0x0BAE, 0x0BB5, F_XML_BASE_CHAR},
	{0x0BB7, 0x0BB9, F_XML_BASE_CHAR},
	{0x0C05, 0x0C0C, F_XML_BASE_CHAR},
	{0x0C0E, 0x0C10, F_XML_BASE_CHAR},
	{0x0C12, 0x0C28, F_XML_BASE_CHAR},
	{0x0C2A, 0x0C33, F_XML_BASE_CHAR},
	{0x0C35, 0x0C39, F_XML_BASE_CHAR},
	{0x0C60, 0x0C61, F_XML_BASE_CHAR},
	{0x0C85, 0x0C8C, F_XML_BASE_CHAR},
	{0x0C8E, 0x0C90, F_XML_BASE_CHAR},
	{0x0C92, 0x0CA8, F_XML_BASE_CHAR},
	{0x0CAA, 0x0CB3, F_XML_BASE_CHAR},
	{0x0CB5, 0x0CB9, F_XML_BASE_CHAR},
	{0x0CDE, 0x0CDE, F_XML_BASE_CHAR},
	{0x0CE0, 0x0CE1, F_XML_BASE_CHAR},
	{0x0D05, 0x0D0C, F_XML_BASE_CHAR},
	{0x0D0E, 0x0D10, F_XML_BASE_CHAR},
	{0x0D12, 0x0D28, F_XML_BASE_CHAR},
	{0x0D2A, 0x0D39, F_XML_BASE_CHAR},
	{0x0D60, 0x0D61, F_XML_BASE_CHAR},
	{0x0E01, 0x0E2E, F_XML_BASE_CHAR},
	{0x0E30, 0x0E30, F_XML_BASE_CHAR},
	{0x0E32, 0x0E33, F_XML_BASE_CHAR},
	{0x0E40, 0x0E45, F_XML_BASE_CHAR},
	{0x0E81, 0x0E82, F_XML_BASE_CHAR},
	{0x0E84, 0x0E84, F_XML_BASE_CHAR},
	{0x0E87, 0x0E88, F_XML_BASE_CHAR},
	{0x0E8A, 0x0E8A, F_XML_BASE_CHAR},
	{0x0E8D, 0x0E8D, F_XML_BASE_CHAR},
	{0x0E94, 0x0E97, F_XML_BASE_CHAR},
	{0x0E99, 0x0E9F, F_XML_BASE_CHAR},
	{0x0EA1, 0x0EA3, F_XML_BASE_CHAR},
	{0x0EA5, 0x0EA5, F_XML_BASE_CHAR},
	{0x0EA7, 0x0EA7, F_XML_BASE_CHAR},
	{0x0EAA, 0x0EAB, F_XML_BASE_CHAR},
	{0x0EAD, 0x0EAE, F_XML_BASE_CHAR},
	{0x0EB0, 0x0EB0, F_XML_BASE_CHAR},
	{0x0EB2, 0x0EB3, F_XML_BASE_CHAR},
	{0x0EBD, 0x0EBD, F_XML_BASE_CHAR},
	{0x0EC0, 0x0EC4, F_XML_BASE_CHAR},
	{0x0F40, 0x0F47, F_XML_BASE_CHAR},
	{0x0F49, 0x0F69, F_XML_BASE_CHAR},
	{0x10A0, 0x10C5, F_XML_BASE_CHAR},
	{0x10D0, 0x10F6, F_XML_BASE_CHAR},
	{0x1100, 0x1100, F_XML_BASE_CHAR},
	{0x1102, 0x1103, F_XML_BASE_CHAR},
	{0x1105, 0x1107, F_XML_BASE_CHAR},
	{0x1109, 0x1109, F_XML_BASE_CHAR},
	{0x110B, 0x110C, F_XML_BASE_CHAR},
	{0x110E, 0x1112, F_XML_BASE_CHAR},
	{0x113C, 0x113C, F_XML_BASE_CHAR},
	{0x113E, 0x113E, F_XML_BASE_CHAR},
	{0x1140, 0x1140, F_XML_BASE_CHAR},
	{0x114C, 0x114C, F_XML_BASE_CHAR},
	{0x114E, 0x114E, F_XML_BASE_CHAR},
	{0x1150, 0x1150, F_XML_BASE_CHAR},
	{0x1154, 0x1155, F_XML_BASE_CHAR},
	{0x1159, 0x1159, F_XML_BASE_CHAR},
	{0x115F, 0x1161, F_XML_BASE_CHAR},
	{0x1163, 0x1163, F_XML_BASE_CHAR},
	{0x1165, 0x1165, F_XML_BASE_CHAR},
	{0x1167, 0x1167, F_XML_BASE_CHAR},
	{0x1169, 0x1169, F_XML_BASE_CHAR},
	{0x116D, 0x116E, F_XML_BASE_CHAR},
	{0x1172, 0x1173, F_XML_BASE_CHAR},
	{0x1175, 0x1175, F_XML_BASE_CHAR},
	{0x119E, 0x119E, F_XML_BASE_CHAR},
	{0x11A8, 0x11A8, F_XML_BASE_CHAR},
	{0x11AB, 0x11AB, F_XML_BASE_CHAR},
	{0x11AE, 0x11AF, F_XML_BASE_CHAR},
	{0x11B7, 0x11B8, F_XML_BASE_CHAR},
	{0x11BA, 0x11BA, F_XML_BASE_CHAR},
	{0x11BC, 0x11C2, F_XML_BASE_CHAR},
	{0x11EB, 0x11EB, F_XML_BASE_CHAR},
	{0x11F0, 0x11F0, F_XML_BASE_CHAR},
	{0x11F9, 0x11F9, F_XML_BASE_CHAR},
	{0x1E00, 0x1E9B, F_XML_BASE_CHAR},
	{0x1EA0, 0x1EF9, F_XML_BASE_CHAR},
	{0x1F00, 0x1F15, F_XML_BASE_CHAR},
	{0x1F18, 0x1F1D, F_XML_BASE_CHAR},
	{0x1F20, 0x1F45, F_XML_BASE_CHAR},
	{0x1F48, 0x1F4D, F_XML_BASE_CHAR},
	{0x1F50, 0x1F57, F_XML_BASE_CHAR},
	{0x1F59, 0x1F59, F_XML_BASE_CHAR},
	{0x1F5B, 0x1F5B, F_XML_BASE_CHAR},
	{0x1F5D, 0x1F5D, F_XML_BASE_CHAR},
	{0x1F5F, 0x1F7D, F_XML_BASE_CHAR},
	{0x1F80, 0x1FB4, F_XML_BASE_CHAR},
	{0x1FB6, 0x1FBC, F_XML_BASE_CHAR},
	{0x1FBE, 0x1FBE, F_XML_BASE_CHAR},
	{0x1FC2, 0x1FC4, F_XML_BASE_CHAR},
	{0x1FC6, 0x1FCC, F_XML_BASE_CHAR},
	{0x1FD0, 0x1FD3, F_XML_BASE_CHAR},
	{0x1FD6, 0x1FDB, F_XML_BASE_CHAR},
	{0x1FE0, 0x1FEC, F_XML_BASE_CHAR},
	{0x1FF2, 0x1FF4, F_XML_BASE_CHAR},
	{0x1FF6, 0x1FFC, F_XML_BASE_CHAR},
	{0x2126, 0x2126, F_XML_BASE_CHAR},
	{0x212A, 0x212B, F_XML_BASE_CHAR},
	{0x212E, 0x212E, F_XML_BASE_CHAR},
	{0x2180, 0x2182, F_XML_BASE_CHAR},
	{0x3041, 0x3094, F_XML_BASE_CHAR},
	{0x30A1, 0x30FA, F_XML_BASE_CHAR},
	{0x3105, 0x312C, F_XML_BASE_CHAR},
	{0xAC00, 0xD7A3, F_XML_BASE_CHAR},
	{0x4E00, 0x9FA5, F_XML_IDEOGRAPHIC},
	{0x3007, 0x3007, F_XML_IDEOGRAPHIC},
	{0x3021, 0x3029, F_XML_IDEOGRAPHIC},
	{0x0300, 0x0345, F_XML_COMBINING_CHAR},
	{0x0360, 0x0361, F_XML_COMBINING_CHAR},
	{0x0483, 0x0486, F_XML_COMBINING_CHAR},
	{0x0591, 0x05A1, F_XML_COMBINING_CHAR},
	{0x05A3, 0x05B9, F_XML_COMBINING_CHAR},
	{0x05BB, 0x05BD, F_XML_COMBINING_CHAR},
	{0x05BF, 0x05BF, F_XML_COMBINING_CHAR},
	{0x05C1, 0x05C2, F_XML_COMBINING_CHAR},
	{0x05C4, 0x05C4, F_XML_COMBINING_CHAR},
	{0x064B, 0x0652, F_XML_COMBINING_CHAR},
	{0x0670, 0x0670, F_XML_COMBINING_CHAR},
	{0x06D6, 0x06DC, F_XML_COMBINING_CHAR},
	{0x06DD, 0x06DF, F_XML_COMBINING_CHAR},
	{0x06E0, 0x06E4, F_XML_COMBINING_CHAR},
	{0x06E7, 0x06E8, F_XML_COMBINING_CHAR},
	{0x06EA, 0x06ED, F_XML_COMBINING_CHAR},
	{0x0901, 0x0903, F_XML_COMBINING_CHAR},
	{0x093C, 0x093C, F_XML_COMBINING_CHAR},
	{0x093E, 0x094C, F_XML_COMBINING_CHAR},
	{0x094D, 0x094D, F_XML_COMBINING_CHAR},
	{0x0951, 0x0954, F_XML_COMBINING_CHAR},
	{0x0962, 0x0963, F_XML_COMBINING_CHAR},
	{0x0981, 0x0983, F_XML_COMBINING_CHAR},
	{0x09BC, 0x09BC, F_XML_COMBINING_CHAR},
	{0x09BE, 0x09BE, F_XML_COMBINING_CHAR},
	{0x09BF, 0x09BF, F_XML_COMBINING_CHAR},
	{0x09C0, 0x09C4, F_XML_COMBINING_CHAR},
	{0x09C7, 0x09C8, F_XML_COMBINING_CHAR},
	{0x09CB, 0x09CD, F_XML_COMBINING_CHAR},
	{0x09D7, 0x09D7, F_XML_COMBINING_CHAR},
	{0x09E2, 0x09E3, F_XML_COMBINING_CHAR},
	{0x0A02, 0x0A02, F_XML_COMBINING_CHAR},
	{0x0A3C, 0x0A3C, F_XML_COMBINING_CHAR},
	{0x0A3E, 0x0A3E, F_XML_COMBINING_CHAR},
	{0x0A3F, 0x0A3F, F_XML_COMBINING_CHAR},
	{0x0A40, 0x0A42, F_XML_COMBINING_CHAR},
	{0x0A47, 0x0A48, F_XML_COMBINING_CHAR},
	{0x0A4B, 0x0A4D, F_XML_COMBINING_CHAR},
	{0x0A70, 0x0A71, F_XML_COMBINING_CHAR},
	{0x0A81, 0x0A83, F_XML_COMBINING_CHAR},
	{0x0ABC, 0x0ABC, F_XML_COMBINING_CHAR},
	{0x0ABE, 0x0AC5, F_XML_COMBINING_CHAR},
	{0x0AC7, 0x0AC9, F_XML_COMBINING_CHAR},
	{0x0ACB, 0x0ACD, F_XML_COMBINING_CHAR},
	{0x0B01, 0x0B03, F_XML_COMBINING_CHAR},
	{0x0B3C, 0x0B3C, F_XML_COMBINING_CHAR},
	{0x0B3E, 0x0B43, F_XML_COMBINING_CHAR},
	{0x0B47, 0x0B48, F_XML_COMBINING_CHAR},
	{0x0B4B, 0x0B4D, F_XML_COMBINING_CHAR},
	{0x0B56, 0x0B57, F_XML_COMBINING_CHAR},
	{0x0B82, 0x0B83, F_XML_COMBINING_CHAR},
	{0x0BBE, 0x0BC2, F_XML_COMBINING_CHAR},
	{0x0BC6, 0x0BC8, F_XML_COMBINING_CHAR},
	{0x0BCA, 0x0BCD, F_XML_COMBINING_CHAR},
	{0x0BD7, 0x0BD7, F_XML_COMBINING_CHAR},
	{0x0C01, 0x0C03, F_XML_COMBINING_CHAR},
	{0x0C3E, 0x0C44, F_XML_COMBINING_CHAR},
	{0x0C46, 0x0C48, F_XML_COMBINING_CHAR},
	{0x0C4A, 0x0C4D, F_XML_COMBINING_CHAR},
	{0x0C55, 0x0C56, F_XML_COMBINING_CHAR},
	{0x0C82, 0x0C83, F_XML_COMBINING_CHAR},
	{0x0CBE, 0x0CC4, F_XML_COMBINING_CHAR},
	{0x0CC6, 0x0CC8, F_XML_COMBINING_CHAR},
	{0x0CCA, 0x0CCD, F_XML_COMBINING_CHAR},
	{0x0CD5, 0x0CD6, F_XML_COMBINING_CHAR},
	{0x0D02, 0x0D03, F_XML_COMBINING_CHAR},
	{0x0D3E, 0x0D43, F_XML_COMBINING_CHAR},
	{0x0D46, 0x0D48, F_XML_COMBINING_CHAR},
	{0x0D4A, 0x0D4D, F_XML_COMBINING_CHAR},
	{0x0D57, 0x0D57, F_XML_COMBINING_CHAR},
	{0x0E31, 0x0E31, F_XML_COMBINING_CHAR},
	{0x0E34, 0x0E3A, F_XML_COMBINING_CHAR},
	{0x0E47, 0x0E4E, F_XML_COMBINING_CHAR},
	{0x0EB1, 0x0EB1, F_XML_COMBINING_CHAR},
	{0x0EB4, 0x0EB9, F_XML_COMBINING_CHAR},
	{0x0EBB, 0x0EBC, F_XML_COMBINING_CHAR},
	{0x0EC8, 0x0ECD, F_XML_COMBINING_CHAR},
	{0x0F18, 0x0F19, F_XML_COMBINING_CHAR},
	{0x0F35, 0x0F35, F_XML_COMBINING_CHAR},
	{0x0F37, 0x0F37, F_XML_COMBINING_CHAR},
	{0x0F39, 0x0F39, F_XML_COMBINING_CHAR},
	{0x0F3E, 0x0F3E, F_XML_COMBINING_CHAR},
	{0x0F3F, 0x0F3F, F_XML_COMBINING_CHAR},
	{0x0F71, 0x0F84, F_XML_COMBINING_CHAR},
	{0x0F86, 0x0F8B, F_XML_COMBINING_CHAR},
	{0x0F90, 0x0F95, F_XML_COMBINING_CHAR},
	{0x0F97, 0x0F97, F_XML_COMBINING_CHAR},
	{0x0F99, 0x0FAD, F_XML_COMBINING_CHAR},
	{0x0FB1, 0x0FB7, F_XML_COMBINING_CHAR},
	{0x0FB9, 0x0FB9, F_XML_COMBINING_CHAR},
	{0x20D0, 0x20DC, F_XML_COMBINING_CHAR},
	{0x20E1, 0x20E1, F_XML_COMBINING_CHAR},
	{0x302A, 0x302F, F_XML_COMBINING_CHAR},
	{0x3099, 0x3099, F_XML_COMBINING_CHAR},
	{0x309A, 0x309A, F_XML_COMBINING_CHAR},
	{0x0030, 0x0039, F_XML_DIGIT},
	{0x0660, 0x0669, F_XML_DIGIT},
	{0x06F0, 0x06F9, F_XML_DIGIT},
	{0x0966, 0x096F, F_XML_DIGIT},
	{0x09E6, 0x09EF, F_XML_DIGIT},
	{0x0A66, 0x0A6F, F_XML_DIGIT},
	{0x0AE6, 0x0AEF, F_XML_DIGIT},
	{0x0B66, 0x0B6F, F_XML_DIGIT},
	{0x0BE7, 0x0BEF, F_XML_DIGIT},
	{0x0C66, 0x0C6F, F_XML_DIGIT},
	{0x0CE6, 0x0CEF, F_XML_DIGIT},
	{0x0D66, 0x0D6F, F_XML_DIGIT},
	{0x0E50, 0x0E59, F_XML_DIGIT},
	{0x0ED0, 0x0ED9, F_XML_DIGIT},
	{0x0F20, 0x0F29, F_XML_DIGIT},
	{0x00B7, 0x00B7, F_XML_EXTENDER},
	{0x02D0, 0x02D0, F_XML_EXTENDER},
	{0x02D1, 0x02D1, F_XML_EXTENDER},
	{0x0387, 0x0387, F_XML_EXTENDER},
	{0x0640, 0x0640, F_XML_EXTENDER},
	{0x0E46, 0x0E46, F_XML_EXTENDER},
	{0x0EC6, 0x0EC6, F_XML_EXTENDER},
	{0x3005, 0x3005, F_XML_EXTENDER},
	{0x3031, 0x3035, F_XML_EXTENDER},
	{0x309D, 0x309E, F_XML_EXTENDER},
	{0x30FC, 0x30FE, F_XML_EXTENDER},
	{0x0009, 0x0009, F_XML_WHITESPACE},
	{0x000A, 0x000A, F_XML_WHITESPACE},
	{0x000D, 0x000D, F_XML_WHITESPACE},
	{0x0020, 0x0020, F_XML_WHITESPACE},
	{0x0000, 0x0000, 0x0000}  // Mark the end.
};

// Function / Method Implementations

/****************************************************************************
Desc:		Constructor
****************************************************************************/
F_XML::F_XML()
{
	m_pCharTable = NULL;
	GedPoolInit( &m_tmpPool, 1024);
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_XML::~F_XML()
{
	if( m_pCharTable)
	{
		f_free( &m_pCharTable);
	}
	GedPoolFree( &m_tmpPool);
}

/****************************************************************************
Desc:		Sets a character's type flag in the character lookup table
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
Desc:		Builds a character lookup table
****************************************************************************/
RCODE F_XML::buildCharTable( void)
{
	RCODE			rc = FERR_OK;
	FLMUINT			uiLoop;

	if( m_pCharTable)
	{
		f_free( &m_pCharTable);
	}

	if( RC_BAD( rc = f_alloc( sizeof( XMLCHAR) * 0xFFFF, &m_pCharTable)))
	{
		goto Exit;
	}

	f_memset( m_pCharTable, 0, sizeof( XMLCHAR) * 0x0000FFFF);

	for (uiLoop = 0; charTbl[uiLoop].ui16Flag; uiLoop++)
	{
		setCharFlag( charTbl[uiLoop].uLowChar,
					 charTbl[uiLoop].uHighChar,
					 charTbl[uiLoop].ui16Flag);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a valid XML PubID character
****************************************************************************/
FINLINE FLMBOOL F_XML::isPubidChar(
	FLMUNICODE		uChar)
{
	if( uChar == F_XML_UNI_SPACE ||
		uChar == F_XML_UNI_LINEFEED ||
		(uChar >= F_XML_UNI_a && uChar <= F_XML_UNI_z) ||
		(uChar >= F_XML_UNI_A && uChar <= F_XML_UNI_Z) ||
		(uChar >= F_XML_UNI_0 && uChar <= F_XML_UNI_9) ||
		uChar == F_XML_UNI_HYPHEN ||
		uChar == F_XML_UNI_APOS ||
		uChar == F_XML_UNI_LPAREN ||
		uChar == F_XML_UNI_RPAREN ||
		uChar == F_XML_UNI_PLUS ||
		uChar == F_XML_UNI_COMMA ||
		uChar == F_XML_UNI_PERIOD ||
		uChar == F_XML_UNI_FSLASH ||
		uChar == F_XML_UNI_COLON ||
		uChar == F_XML_UNI_EQ ||
		uChar == F_XML_UNI_QUEST ||
		uChar == F_XML_UNI_SEMI ||
		uChar == F_XML_UNI_BANG ||
		uChar == F_XML_UNI_ASTERISK ||
		uChar == F_XML_UNI_POUND ||
		uChar == F_XML_UNI_ATSIGN ||
		uChar == F_XML_UNI_DOLLAR ||
		uChar == F_XML_UNI_UNDERSCORE ||
		uChar == F_XML_UNI_PERCENT)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a single or double quote character
****************************************************************************/
inline FLMBOOL F_XML::isQuoteChar(
	FLMUNICODE		uChar)
{
	if( uChar == F_XML_UNI_QUOTE ||
		uChar == F_XML_UNI_APOS)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a whitespace character
****************************************************************************/
FINLINE FLMBOOL F_XML::isWhitespace(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & F_XML_WHITESPACE) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is an extender character
****************************************************************************/
FINLINE FLMBOOL F_XML::isExtender(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & F_XML_EXTENDER) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a combining character
****************************************************************************/
FINLINE FLMBOOL F_XML::isCombiningChar(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & F_XML_COMBINING_CHAR) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a digit
****************************************************************************/
FINLINE FLMBOOL F_XML::isDigit(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & F_XML_DIGIT) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is an ideographic character
****************************************************************************/
FINLINE FLMBOOL F_XML::isIdeographic(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & F_XML_IDEOGRAPHIC) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a base character
****************************************************************************/
FINLINE FLMBOOL F_XML::isBaseChar(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & F_XML_BASE_CHAR) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a letter
****************************************************************************/
FINLINE FLMBOOL F_XML::isLetter(
	FLMUNICODE		uChar)
{
	if( isBaseChar( uChar) || isIdeographic( uChar))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a valid XML naming character
****************************************************************************/
FINLINE FLMBOOL F_XML::isNameChar(
	FLMUNICODE		uChar)
{
	if( isLetter( uChar) || 
		isDigit( uChar) ||
		uChar == F_XML_UNI_PERIOD ||
		uChar == F_XML_UNI_HYPHEN ||
		uChar == F_XML_UNI_UNDERSCORE ||
		uChar == F_XML_UNI_COLON ||
		isCombiningChar( uChar) || isExtender( uChar))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc: 	Returns TRUE if the name is a valid XML name
****************************************************************************/
FLMBOOL F_XML::isNameValid(
	const FLMUNICODE *	puzName,
	const char *			pszName)
{
	FLMBOOL			bValid = FALSE;

	if( puzName)
	{
		const FLMUNICODE *	puzTmp;

		if( !isLetter( *puzName) && *puzName != F_XML_UNI_UNDERSCORE &&
			*puzName != F_XML_UNI_COLON)
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
		const char *	pszTmp;

		if( !isLetter( *pszName) && *pszName != F_XML_UNI_UNDERSCORE &&
			*pszName != F_XML_UNI_COLON)
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
F_XMLExport::F_XMLExport()
{
	f_memset( m_szSpaces, 0x20, sizeof( m_szSpaces));
	m_uiTmpBufSize = 0;
	m_pszTmpBuf = NULL;
	m_pByteStream = NULL;
	m_bSetup = FALSE;
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_XMLExport::~F_XMLExport()
{
	if( m_pszTmpBuf)
	{
		f_free( &m_pszTmpBuf);
	}

	if( m_pByteStream)
	{
		m_pByteStream->Release();
	}
}

/****************************************************************************
Desc:		Initializes the object (allocates buffers, etc.)
****************************************************************************/
RCODE F_XMLExport::setup( void)
{
	RCODE			rc = FERR_OK;

	m_uiTmpBufSize = 64 * 1024;
	if( RC_BAD( rc = f_alloc( m_uiTmpBufSize, &m_pszTmpBuf)))
	{
		goto Exit;
	}

	if( (m_pByteStream = f_new FCS_BIOS) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	if( RC_BAD( rc = buildCharTable()))
	{
		goto Exit;
	}

	m_bSetup = TRUE;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Exports a FLAIM record object as XML text.
****************************************************************************/
RCODE F_XMLExport::exportRecord(
	F_NameTable *	pNameTable,
	FlmRecord *		pRec,
	FLMUINT			uiStartIndent,
	FLMUINT			uiIndentSize,
	POOL *			pPool,
	char **			ppszXML,
	FLMUINT *		puiBytes)
{
	FLMUINT		uiBytes;
	char			szNameBuf[ 256];
	FLMBYTE		szTmpBuf[ 256];
	FlmBlob *	pBlob = NULL;
	void *		pvCur;
	void *		pvPrev;
	void *		pvStart;
	void *		pvEnd;
	RCODE			rc = FERR_OK;

	if( ppszXML)
	{
		*ppszXML = NULL;
	}

	if( puiBytes)
	{
		*puiBytes = 0;
	}

	if( !pNameTable)
	{
		rc = RC_SET( FERR_INVALID_PARM);
		goto Exit;
	}

	if( RC_BAD( rc = m_pByteStream->reset()))
	{
		goto Exit;
	}

	pvCur = pRec->root();
	while( pvCur)
	{
		m_szSpaces[ (pRec->getLevel( pvCur) * 
			uiIndentSize) + uiStartIndent] = '\0';

		if( !pNameTable->getFromTagNum( pRec->getFieldID( pvCur),
			NULL, szNameBuf, sizeof( szNameBuf)))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( !isNameValid( NULL, szNameBuf))
		{
			f_sprintf( (char *)szNameBuf, "tag_%u",
				(unsigned)pRec->getFieldID( pvCur));
		}

		uiBytes = f_sprintf( (char *)szTmpBuf, "%s<%s>",
			m_szSpaces, szNameBuf);
		
		m_szSpaces[ (pRec->getLevel( pvCur) * 
			uiIndentSize) + uiStartIndent] = ' ';
		if( RC_BAD( rc = m_pByteStream->write( szTmpBuf, uiBytes)))
		{
			goto Exit;
		}

		if( pRec->getDataLength( pvCur))
		{
			FLMUINT			uiNum;
			FLMINT			iNum;

			switch( pRec->getDataType( pvCur))
			{
				case FLM_CONTEXT_TYPE:
				{
					if( RC_BAD( rc = pRec->getUINT( pvCur, &uiNum)))
					{
						goto Exit;
					}

					uiBytes = f_sprintf( (char *)szTmpBuf, "%u", (unsigned)uiNum);
					if( RC_BAD( rc = m_pByteStream->write( szTmpBuf, uiBytes)))
					{
						goto Exit;
					}
					break;
				}

				case FLM_TEXT_TYPE:
				{
					FLMUNICODE *		pUnicode = (FLMUNICODE *)m_pszTmpBuf;
					FLMUINT				uiUniSize = m_uiTmpBufSize;

					if( RC_BAD( rc = pRec->getUnicode( pvCur, 
						pUnicode, &uiUniSize)))
					{
						goto Exit;
					}

					uiBytes = 0;
					while( *pUnicode)
					{
						if( *pUnicode >= F_XML_UNI_SPACE && 
							 *pUnicode <= F_XML_UNI_TILDE &&
							 *pUnicode != F_XML_UNI_AMP && *pUnicode != F_XML_UNI_SEMI &&
							 *pUnicode != F_XML_UNI_GT && *pUnicode != F_XML_UNI_LT)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "%c", (char)*pUnicode);
						}
						else if( *pUnicode == F_XML_UNI_AMP)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&amp;");
						}
						else if( *pUnicode == F_XML_UNI_LT)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&lt;");
						}
						else if( *pUnicode == F_XML_UNI_GT)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&gt;");
						}
						else
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&#%u;",
								(unsigned)*pUnicode);
						}

						if( RC_BAD( rc = m_pByteStream->write( 
							szTmpBuf, uiBytes)))
						{
							goto Exit;
						}

						pUnicode++;
					}

					break;
				}

				case FLM_NUMBER_TYPE:
				{
					if( RC_BAD( rc = pRec->getUINT( pvCur, &uiNum)))
					{
						if( rc == FERR_CONV_NUM_UNDERFLOW)
						{
							if( RC_BAD( rc = pRec->getINT( pvCur, &iNum)))
							{
								goto Exit;
							}

							uiBytes = f_sprintf( (char *)szTmpBuf, "%d", 
								(int)iNum);
						}
						else
						{
							goto Exit;
						}
					}
					else
					{
						uiBytes = f_sprintf( (char *)szTmpBuf, "%u", 
							(unsigned)uiNum);
					}

					if( RC_BAD( rc = m_pByteStream->write( 
						szTmpBuf, uiBytes)))
					{
						goto Exit;
					}

					break;
				}

				case FLM_BINARY_TYPE:
				{
					FLMUINT		uiBinLen = m_uiTmpBufSize;
					FLMUINT		uiLoop;

					if( RC_BAD( rc = pRec->getBinary( pvCur, 
						m_pszTmpBuf, &uiBinLen)))
					{
						goto Exit;
					}

					for( uiLoop = 0; uiLoop < uiBinLen; uiLoop++)
					{
						FLMBYTE		ucHexChar1;
						FLMBYTE		ucHexChar2;

						ucHexChar1 = (m_pszTmpBuf[ uiLoop] & 0xF0) >> 4;
						if( ucHexChar1 <= 9)
						{
							ucHexChar1 += '0';
						}
						else if( ucHexChar1 >= 0x0A && ucHexChar1 <= 0x0F)
						{
							ucHexChar1 = (ucHexChar1 - 0x0A) + 'A';
						}

						ucHexChar2 = (m_pszTmpBuf[ uiLoop] & 0x0F);
						if( ucHexChar2 <= 9)
						{
							ucHexChar2 += '0';
						}
						else if( ucHexChar2 >= 0x0A && ucHexChar2 <= 0x0F)
						{
							ucHexChar2 = (ucHexChar2 - 0x0A) + 'A';
						}

						szTmpBuf[ 0] = ucHexChar1;
						szTmpBuf[ 1] = ucHexChar2;

						if( RC_BAD( rc = m_pByteStream->write( 
							szTmpBuf, 2)))
						{
							goto Exit;
						}
					}

					break;
				}

				case FLM_BLOB_TYPE:
				{
					char		szPath[ F_PATH_MAX_SIZE];
					char *	pszTmp;

					if( RC_BAD( rc = pRec->getBlob( pvCur, &pBlob)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = pBlob->buildFileName( szPath)))
					{
						goto Exit;
					}

					pszTmp = &szPath[ 0];
					while( *pszTmp)
					{
						if( *pszTmp >= F_XML_UNI_SPACE && 
							 *pszTmp <= F_XML_UNI_TILDE &&
							 *pszTmp != F_XML_UNI_AMP && *pszTmp != F_XML_UNI_SEMI &&
							 *pszTmp != F_XML_UNI_GT && *pszTmp != F_XML_UNI_LT)
						{
							szTmpBuf[ 0] = *pszTmp;
							uiBytes = 1;
						}
						else if( *pszTmp == F_XML_UNI_AMP)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&amp;");
						}
						else if( *pszTmp == F_XML_UNI_LT)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&lt;");
						}
						else if( *pszTmp == F_XML_UNI_GT)
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&gt;");
						}
						else
						{
							uiBytes = f_sprintf( (char *)szTmpBuf, "&#%u;",
								(unsigned)*pszTmp);
						}

						if( RC_BAD( rc = m_pByteStream->write( 
							szTmpBuf, uiBytes)))
						{
							goto Exit;
						}

						pszTmp++;
					}
					pBlob->Release();
					pBlob = NULL;
					break;
				}

				default:
				{
					flmAssert( 0);
					rc = RC_SET( FERR_NOT_IMPLEMENTED);
					goto Exit;
				}
			}
		}

		if( pRec->firstChild( pvCur))
		{
			uiBytes = f_sprintf( (char *)szTmpBuf, "%s", CRLFSTR);
		}
		else
		{
			uiBytes = f_sprintf( (char *)szTmpBuf, "</%s>%s",
				szNameBuf, CRLFSTR);
		}

		if( RC_BAD( rc = m_pByteStream->write( szTmpBuf, uiBytes)))
		{
			goto Exit;
		}

		pvPrev = pvCur;
		pvCur = pRec->next( pvCur);
		pvStart = pRec->parent( pvPrev);

		if( pvCur)
		{
			pvEnd = pvCur;
		}
		else
		{
			pvEnd = pRec->root();
		}

		while( pvStart && pRec->getLevel( pvStart) >= pRec->getLevel( pvEnd))
		{
			if( RC_BAD( rc = m_pByteStream->write( m_szSpaces, 
			(pRec->getLevel( pvStart) * uiIndentSize) + uiStartIndent)))
			{
				goto Exit;
			}

			if( !pNameTable->getFromTagNum( pRec->getFieldID( pvStart),
				NULL, szNameBuf, sizeof( szNameBuf)))
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			if( !isNameValid( NULL, szNameBuf))
			{
				f_sprintf( (char *)szNameBuf, "tag_%u",
					(unsigned)pRec->getFieldID( pvStart));
			}

			uiBytes = f_sprintf( (char *)szTmpBuf, "</%s>%s",
				szNameBuf, CRLFSTR);
			if( RC_BAD( rc = m_pByteStream->write( szTmpBuf, uiBytes)))
			{
				goto Exit;
			}

			if( pRec->getLevel( pvStart) == pRec->getLevel( pvEnd))
			{
				break;
			}

			pvStart = pRec->parent( pvStart);
		}
	}

	if( RC_BAD( rc = m_pByteStream->endMessage()))
	{
		goto Exit;
	}

	uiBytes = m_pByteStream->getAvailable();
	if( ppszXML)
	{
		if( (*ppszXML = (char *)GedPoolAlloc( pPool, uiBytes + 1)) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pByteStream->read( (FLMBYTE *)*ppszXML, uiBytes, &uiBytes)))
		{
			goto Exit;
		}

		(*ppszXML)[ uiBytes] = 0;
	}

	if( puiBytes)
	{
		*puiBytes = uiBytes;
	}

Exit:

	if( pBlob)
	{
		pBlob->Release();
	}

	m_pByteStream->reset();
	return( rc);
}

/****************************************************************************
Desc:		Constructor	
****************************************************************************/
F_XMLImport::F_XMLImport()
{
	reset();
	m_uiValBufSize = 0;
	m_puValBuf = NULL;
	m_bSetup = FALSE;
}

/****************************************************************************
Desc:		Destructor
****************************************************************************/
F_XMLImport::~F_XMLImport()
{
	if( m_puValBuf)
	{
		f_free( &m_puValBuf);
	}
}

/****************************************************************************
Desc:		Resets member variables so the object can be reused
****************************************************************************/
void F_XMLImport::reset( void)
{
	m_uiUngetPos = 0;
	m_ucUngetByte = 0;
	m_pStream = NULL;
	m_bSubset = FALSE;
	m_pNameTable = NULL;
	m_hDb = HFDB_NULL;
}

/****************************************************************************
Desc: 	Initializes the object (allocates buffers, etc.)
****************************************************************************/
RCODE F_XMLImport::setup( void)
{
	RCODE			rc = FERR_OK;

	flmAssert( !m_bSetup);

	m_uiValBufSize = 32 * 1024; // # of Unicode characters

	if( RC_BAD( rc = f_alloc(
		sizeof( FLMUNICODE) * m_uiValBufSize, &m_puValBuf)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = buildCharTable()))
	{
		goto Exit;
	}

	m_bSetup = TRUE;

Exit:

	return( rc);
}
/****************************************************************************
Desc: 	Returns a fields type and tag given its real name or
			its TAG_##### equivalent
****************************************************************************/
RCODE F_XMLImport::getFieldTagAndType(
	FLMUNICODE *	puzName,
	FLMUINT *		puiTagNum,
	FLMUINT *		puiDataType)
{
	RCODE			rc = FERR_OK;
	void *		pvMark = GedPoolMark( &m_tmpPool);
	char *		pszName;
	FLMUINT		uiTagNum;
	FLMUINT		uiType;
	FLMUINT		uiDataType;

	if( !m_pNameTable ||
		!m_pNameTable->getFromTagTypeAndName( puzName, NULL, 
			FLM_FIELD_TAG, puiTagNum, puiDataType))
	{
		// Convert the tag name to ASCII

		if( RC_BAD( rc = fcsConvertUnicodeToNative( &m_tmpPool, 
			m_uChars, &pszName)))
		{
			goto Exit;
		}

		if( f_strnicmp( pszName, "TAG_", 4) == 0)
		{
			uiTagNum = f_atoud( &pszName[ 4]);

			if( puiTagNum)
			{
				*puiTagNum = uiTagNum;
			}

			if( !m_pNameTable->getFromTagNum( uiTagNum, NULL, 
				NULL, 0, &uiType, &uiDataType) || uiType != FLM_FIELD_TAG)
			{
				rc = RC_SET( FERR_NOT_FOUND);
				goto Exit;
			}

			if( puiDataType)
			{
				*puiDataType = uiDataType;
			}
		}
		else
		{
			rc = RC_SET( FERR_NOT_FOUND);
			goto Exit;
		}
	}

Exit:

	GedPoolReset( &m_tmpPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc: 	Reads data from the input stream and builds a FLAIM record
****************************************************************************/
RCODE F_XMLImport::importDocument(
	HFDB				hDb,
	F_NameTable *	pNameTable,
	FCS_ISTM *		pStream,
	FLMBOOL			bSubset,
	FlmRecord **	ppRecord)
{
	FlmRecord *		pRec = NULL;
	void *			pvParent;
	RCODE				rc = FERR_OK;

	*ppRecord = NULL;

	reset();
	m_pStream = pStream;
	m_hDb = hDb;
	m_bSubset = bSubset;

	if( !pNameTable)
	{
		if( (m_pNameTable = f_new F_NameTable) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = m_pNameTable->setupFromDb( hDb)))
		{
			goto Exit;
		}
	}
	else
	{
		m_pNameTable = pNameTable;
	}

	if( (pRec = f_new FlmRecord) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	pvParent = NULL;
	if( !m_bSubset)
	{
		if( RC_BAD( rc = pRec->insertLast( 0,
			F_XML_DOCUMENT_TAG, FLM_TEXT_TYPE, &pvParent)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = processProlog( pRec, pvParent)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processElement( pRec, pvParent)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processMisc( pRec, pvParent)))
	{
		goto Exit;
	}

	*ppRecord = pRec;
	pRec = NULL;

Exit:

	if( pRec)
	{
		pRec->Release();
	}

	if( m_pNameTable != pNameTable)
	{
		m_pNameTable->Release();
		m_pNameTable = NULL;
	}

	return( rc);
}

/****************************************************************************
Desc: Process an XML prolog
****************************************************************************/
RCODE F_XMLImport::processProlog(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMBOOL				bTmp;
	RCODE					rc = FERR_OK;

	if( RC_BAD( rc = isXMLDecl( &bTmp)))
	{
		goto Exit;
	}

	if( bTmp)
	{
		if( RC_BAD( rc = processXMLDecl( pRec, pvParent)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = processMisc( pRec, pvParent)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = isDocTypeDecl( &bTmp)))
	{
		goto Exit;
	}

	if( bTmp)
	{
		if( RC_BAD( rc = processDocTypeDecl( pRec, pvParent)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = processMisc( pRec, pvParent)))
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
RCODE F_XMLImport::unicodeToNumber(
	const FLMUNICODE *	puzVal,
	FLMUINT *				puiVal,
	FLMBOOL *				pbNeg)
{
	char				szTmpBuf[ 32];
	FLMUINT			uiLoop;
	FLMBOOL			bNeg = FALSE;
	FLMUNICODE		uChar;
	RCODE				rc = FERR_OK;

	for( uiLoop = 0; uiLoop < sizeof( szTmpBuf); uiLoop++)
	{
		if( (uChar = puzVal[ uiLoop]) == 0)
		{
			break;
		}
		else if( uiLoop == 0 && uChar == F_XML_UNI_HYPHEN)
		{
			bNeg = TRUE;
			continue;
		}
		
		if( !((uChar >= '0' && uChar <= '9') ||
			(uChar >= 'A' && uChar <= 'F') ||
			(uChar >= 'a' && uChar <= 'f') ||
			uChar == 'X' || uChar == 'x'))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		szTmpBuf[ uiLoop] = (FLMBYTE)uChar;
	}

	if( uiLoop == sizeof( szTmpBuf))
	{
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	szTmpBuf[ uiLoop] = 0;
	*puiVal = f_atoud( szTmpBuf);

	if( pbNeg)
	{
		*pbNeg = bNeg;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Sets an element field's value
****************************************************************************/
RCODE F_XMLImport::setElementValue(
	FlmRecord *				pRec,
	void *					pvField,
	const FLMUNICODE *	puzValue)
{
	FLMUINT		uiDataType = pRec->getDataType( pvField);
	FLMUINT		uiVal;
	FLMBOOL		bNeg;
	FlmBlob *	pBlob = NULL;
	void *		pvMark = GedPoolMark( &m_tmpPool);
	RCODE			rc = FERR_OK;

	switch( uiDataType)
	{
		case FLM_TEXT_TYPE:
		{
			if( RC_BAD( rc = pRec->setUnicode( pvField, puzValue)))
			{
				goto Exit;
			}
			break;
		}

		case FLM_NUMBER_TYPE:
		{
			if( RC_BAD( rc = unicodeToNumber( puzValue, &uiVal, &bNeg)))
			{
				goto Exit;
			}

			if( !bNeg)
			{
				if( RC_BAD( rc = pRec->setUINT( pvField, uiVal)))
				{
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = pRec->setINT( pvField, -((FLMINT)uiVal))))
				{
					goto Exit;
				}
			}
			break;
		}

		case FLM_CONTEXT_TYPE:
		{
			if( RC_BAD( rc = unicodeToNumber( puzValue, &uiVal, &bNeg)))
			{
				goto Exit;
			}

			if( bNeg)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setRecPointer( pvField, uiVal)))
			{
				goto Exit;
			}

			break;
		}

		case FLM_BINARY_TYPE:
		{
			FLMBYTE *	pucCur = (FLMBYTE *)&puzValue[ 0];
			FLMBYTE		ucVal;
			FLMUINT		uiLoop;
			FLMBOOL		bShift = TRUE;
			FLMUINT		uiBytes = 0;
			FLMUNICODE	uChar;

			for( uiLoop = 0; puzValue[ uiLoop]; uiLoop++)
			{
				uChar = puzValue[ uiLoop];

				if( isWhitespace( uChar))
				{
					continue;
				}

				if( uChar >= F_XML_UNI_0 && uChar <= F_XML_UNI_9)
				{
					ucVal = (FLMBYTE)(uChar - F_XML_UNI_0);
				}
				else if( uChar >= F_XML_UNI_A && uChar <= F_XML_UNI_F)
				{
					ucVal = (FLMBYTE)((uChar - F_XML_UNI_A) + 0x0A);
				}
				else if( uChar >= F_XML_UNI_a && uChar <= F_XML_UNI_f)
				{
					ucVal = (FLMBYTE)((uChar - F_XML_UNI_a) + 0x0A);
				}
				else
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				if( bShift)
				{
					*pucCur = ucVal << 4;
					bShift = FALSE;
				}
				else
				{
					*pucCur |= ucVal;
					pucCur++;
					uiBytes++;
					bShift = TRUE;
				}
			}

			if( !bShift)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			if( uiBytes)
			{
				if( RC_BAD( rc = pRec->setBinary( pvField, (FLMBYTE *)&puzValue[ 0],
					uiBytes)))
				{
					goto Exit;
				}
			}

			break;
		}

		case FLM_BLOB_TYPE:
		{
			char *	pszPath;

			if( m_hDb == HFDB_NULL)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			if ( (pBlob = f_new FlmBlobImp) == NULL)
			{
				rc = RC_SET( FERR_MEM);
				goto Exit;
			}

			GedPoolReset( &m_tmpPool, pvMark);
			if( RC_BAD( rc = fcsConvertUnicodeToNative( &m_tmpPool, 
				puzValue, &pszPath)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pBlob->referenceFile( m_hDb, pszPath, TRUE)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setBlob( pvField, pBlob)))
			{
				goto Exit;
			}

			pBlob->Release();
			pBlob = NULL;
			break;
		}

		default:
		{
			flmAssert( 0);
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:

	if( pBlob)
	{
		pBlob->Release();
	}

	GedPoolReset( &m_tmpPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc:		Processes an XML element
****************************************************************************/
RCODE F_XMLImport::processElement(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMBOOL				bHasContent;
	FLMBOOL				bValueSet = FALSE;
	FLMUINT				uiLen;
	FLMUINT				uiChars;
	FLMUINT				uiOffset = 0;
	FLMUNICODE			uChar;
	void *				pvElementRoot;
	void *				pvField;
	RCODE					rc = FERR_OK;

	if( RC_BAD( rc = processSTag( pRec, pvParent,
		&bHasContent, &pvElementRoot)))
	{
		goto Exit;
	}

	if( bHasContent)
	{
		for( ;;)
		{
			if( RC_BAD( rc = getChar( &m_uChars[ 0])))
			{
				goto Exit;
			}

			if( m_uChars[ 0] == F_XML_UNI_LT)
			{
				if( uiOffset)
				{
					// Flush the value to the record

					m_puValBuf[ uiOffset] = 0;
					if( m_bSubset)
					{
						if( bValueSet)
						{
							rc = RC_SET( FERR_SYNTAX);
							goto Exit;
						}

						if( RC_BAD( rc = setElementValue( pRec, 
							pvElementRoot, m_puValBuf)))
						{
							goto Exit;
						}
					}
					else
					{
						if( RC_BAD( rc = pRec->insertLast( 
							pRec->getLevel( pvElementRoot) + 1,
							F_XML_ELEMENTVAL_TAG, FLM_TEXT_TYPE, &pvField)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
						{
							goto Exit;
						}
					}

					uiOffset = 0;
					bValueSet = TRUE;
				}

				if( RC_BAD( rc = getChar( &m_uChars[ 1])))
				{
					goto Exit;
				}
				
				if( m_uChars[ 1] == F_XML_UNI_QUEST)
				{
					if( RC_BAD( rc = ungetChars( m_uChars, 2)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = processPI( pRec, 
						m_bSubset ? NULL : pvElementRoot)))
					{
						goto Exit;
					}
				}
				else if( m_uChars[ 1] == F_XML_UNI_BANG)
				{
					uiChars = 2;
					if( RC_BAD( rc = getChars( &m_uChars[ 2], &uiChars)))
					{
						goto Exit;	
					}

					if( m_uChars[ 2] == F_XML_UNI_HYPHEN &&
						m_uChars[ 3] == F_XML_UNI_HYPHEN)
					{
						if( RC_BAD( rc = ungetChars( m_uChars, 4)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = processComment( pRec,
							m_bSubset ? NULL : pvElementRoot)))
						{
							goto Exit;
						}
					}
					else if( m_uChars[ 2] == F_XML_UNI_LBRACKET &&
						m_uChars[ 3] == F_XML_UNI_C)
					{
						uiChars = 5;
						if( RC_BAD( rc = getChars( &m_uChars[ 4], &uiChars)))
						{
							goto Exit;	
						}

						if( m_uChars[ 4] != F_XML_UNI_D ||
							m_uChars[ 5] != F_XML_UNI_A ||
							m_uChars[ 6] != F_XML_UNI_T ||
							m_uChars[ 7] != F_XML_UNI_A ||
							m_uChars[ 8] != F_XML_UNI_LBRACKET)
						{
							rc = RC_SET( FERR_SYNTAX);
							goto Exit;
						}

						if( RC_BAD( rc = ungetChars( m_uChars, 9)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = processCDATA( pRec,
							m_bSubset ? NULL : pvElementRoot)))
						{
							goto Exit;
						}
					}
					else
					{
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
					}
				}
				else if( isNameChar( m_uChars[ 1]))
				{
					if( RC_BAD( rc = ungetChars( m_uChars, 2)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = processElement( pRec, pvElementRoot)))
					{
						goto Exit;
					}
				}
				else if( m_uChars[ 1] == F_XML_UNI_FSLASH)
				{
					if( uiOffset)
					{
						// Flush the value to the record

						m_puValBuf[ uiOffset] = 0;
						if( m_bSubset)
						{
							if( bValueSet)
							{
								rc = RC_SET( FERR_SYNTAX);
								goto Exit;
							}

							if( RC_BAD( rc = setElementValue( pRec, 
								pvElementRoot, m_puValBuf)))
							{
								goto Exit;
							}
						}
						else
						{
							if( RC_BAD( rc = pRec->insertLast( 
								pRec->getLevel( pvElementRoot) + 1,
								F_XML_ELEMENTVAL_TAG, FLM_TEXT_TYPE, &pvField)))
							{
								goto Exit;
							}

							if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
							{
								goto Exit;
							}
						}

						uiOffset = 0;
						bValueSet = TRUE;
					}
					break;
				}
				else
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
			else
			{
				if( RC_BAD( rc = ungetChar( m_uChars[ 0])))
				{
					goto Exit;
				}

				for( ;;)
				{
					if( RC_BAD( rc = getChar( &uChar)))
					{
						goto Exit;
					}

					if( m_bSubset && uChar == F_XML_UNI_AMP)
					{
						if( RC_BAD( rc = processReference( pRec, NULL, &uChar)))
						{
							goto Exit;
						}

						m_puValBuf[ uiOffset++] = uChar;
						if( uiOffset >= m_uiValBufSize)
						{
							rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
							goto Exit;
						}
						continue;
					}
					
					if( uiOffset && (uChar == F_XML_UNI_LT ||
						uChar == F_XML_UNI_AMP))
					{
						if( uiOffset)
						{
							// Flush the value to the record

							m_puValBuf[ uiOffset] = 0;
							if( m_bSubset)
							{
								if( bValueSet)
								{
									rc = RC_SET( FERR_SYNTAX);
									goto Exit;
								}

								if( RC_BAD( rc = setElementValue( pRec, 
									pvElementRoot, m_puValBuf)))
								{
									goto Exit;
								}
							}
							else
							{
								if( RC_BAD( rc = pRec->insertLast( 
									pRec->getLevel( pvElementRoot) + 1,
									F_XML_ELEMENTVAL_TAG, FLM_TEXT_TYPE, &pvField)))
								{
									goto Exit;
								}

								if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
								{
									goto Exit;
								}
							}

							uiOffset = 0;
							bValueSet = TRUE;
						}
					}

					if( uChar == F_XML_UNI_LT)
					{
						if( RC_BAD( rc = ungetChar( uChar)))
						{
							goto Exit;
						}
						break;
					}
					else if( uChar == F_XML_UNI_AMP)
					{
						if( RC_BAD( rc = ungetChar( uChar)))
						{
							goto Exit;
						}

						if( RC_BAD( rc = processReference( pRec, pvElementRoot)))
						{
							goto Exit;
						}
					}
					else if( uChar == F_XML_UNI_LINEFEED && m_bSubset)
					{
						if( RC_BAD( rc = skipWhitespace( NULL)))
						{
							goto Exit;
						}
					}
					else
					{
						m_puValBuf[ uiOffset++] = uChar;
						if( uiOffset >= m_uiValBufSize)
						{
							rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
							goto Exit;
						}
					}
				}
			}
		}

		flmAssert( !uiOffset);
		flmAssert( m_uChars[ 0] == F_XML_UNI_LT);
		flmAssert( m_uChars[ 1] == F_XML_UNI_FSLASH);

		uiChars = F_XML_MAX_CHARS;
		if( RC_BAD( rc = getName( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		// Validate the end tag against the start tag

		if( m_bSubset)
		{
			FLMUINT		uiTagNum;
			FLMUINT		uiDataType;

			if( RC_BAD( rc = getFieldTagAndType( m_uChars, 
				&uiTagNum, &uiDataType)))
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			if( pRec->getFieldID( pvElementRoot) != uiTagNum ||
				pRec->getDataType( pvElementRoot) != uiDataType)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
		else
		{
			uiLen = m_uiValBufSize * sizeof( FLMUNICODE);
			if( RC_BAD( rc = pRec->getUnicode( pvElementRoot, 
				m_puValBuf, &uiLen)))
			{
				goto Exit;
			}

			if( f_unicmp( m_uChars, m_puValBuf) != 0)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}

		// Skip any whitespace after the name

		if( RC_BAD( rc = skipWhitespace( pvParent)))
		{
			goto Exit;
		}

		// Get the closing bracket character

		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		// Expecting a closing bracket
		
		if( uChar != F_XML_UNI_GT)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML STag
****************************************************************************/
RCODE F_XMLImport::processSTag(
	FlmRecord *		pRec,
	void *			pvParent,
	FLMBOOL *		pbHasContent,
	void **			ppvElementRoot)
{
	FLMUNICODE		uChar;
	FLMUINT			uiChars;
	FLMUINT			uiTagNum;
	FLMUINT			uiDataType;
	void *			pvMark = GedPoolMark( &m_tmpPool);
	void *			pvField;
	RCODE				rc = FERR_OK;

	*pbHasContent = FALSE;
	*ppvElementRoot = NULL;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_LT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_bSubset)
	{
		// Find the name in the name table

		if( RC_BAD( rc = getFieldTagAndType( m_uChars, 
			&uiTagNum, &uiDataType)))
		{
			if( rc == FERR_NOT_FOUND)
			{
				rc = RC_SET( FERR_SYNTAX);
			}
			goto Exit;
		}

		if( RC_BAD( rc = pRec->insertLast( 
			pvParent ? pRec->getLevel( pvParent) + 1 : 0,
			uiTagNum, uiDataType, &pvField)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ELEMENTNAME_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

	*ppvElementRoot = pvField;

	if( RC_BAD( rc = skipWhitespace( pvParent)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		if( RC_BAD( rc = processAttributes( pRec, 
			m_bSubset ? NULL : pvField)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( 
		m_bSubset ? NULL : pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_GT)
	{
		*pbHasContent = TRUE;
	}
	else if( uChar == F_XML_UNI_FSLASH)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_GT)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:

	GedPoolReset( &m_tmpPool, pvMark);
	return( rc);
}

/****************************************************************************
Desc:		Processes an element's attributes
****************************************************************************/
RCODE F_XMLImport::processAttributes(
	FlmRecord *			pRec,
	void *				pvParent)
{
	FLMUNICODE		uChar;
	FLMUINT			uiChars;
	void *			pvField = NULL;
	RCODE				rc = FERR_OK;

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( pvParent)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uChar)))
		{
			goto Exit;
		}
		
		if( !isNameChar( uChar))
		{
			break;
		}

		uiChars = F_XML_MAX_CHARS;
		if( RC_BAD( rc = getName( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		if( pvParent)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvParent) + 1,
				F_XML_ATTNAME_TAG, FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
			{
				goto Exit;
			}
		}

		if( RC_BAD( rc = skipWhitespace( pvParent)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_EQ)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( pvParent)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = processAttValue( pRec, pvField)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML declaration
****************************************************************************/
RCODE F_XMLImport::processXMLDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_XMLDECL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvParent)))
	{
		goto Exit;
	}

	uiChars = 5;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_QUEST ||
		m_uChars[ 2] != F_XML_UNI_x ||
		m_uChars[ 3] != F_XML_UNI_m ||
		m_uChars[ 4] != F_XML_UNI_l)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( pvField)
	{
		m_uChars[ 5] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, &m_uChars[ 2])))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvParent, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processVersion( pRec, pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
	{
		if( rc != FERR_SYNTAX)
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_QUEST)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_e)
	{
		if( RC_BAD( rc = processEncodingDecl( pRec, pvField)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
		{
			if( rc != FERR_SYNTAX)
			{
				goto Exit;
			}

			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar != F_XML_UNI_QUEST)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_s)
	{
		if( RC_BAD( rc = processSDDecl( pRec, pvField)))
		{
			goto Exit;
		}
	}	

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_QUEST)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML document type declaration
****************************************************************************/
RCODE F_XMLImport::processDocTypeDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	FLMBOOL				bTmp;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_DOCTYPEDECL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvParent)))
	{
		goto Exit;
	}

	uiChars = 9;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_D ||
		m_uChars[ 3] != F_XML_UNI_O ||
		m_uChars[ 4] != F_XML_UNI_C ||
		m_uChars[ 5] != F_XML_UNI_T ||
		m_uChars[ 6] != F_XML_UNI_Y ||
		m_uChars[ 7] != F_XML_UNI_P ||
		m_uChars[ 8] != F_XML_UNI_E)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}
	
	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( pvField)
	{
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	bTmp = FALSE;
	if( isWhitespace( uChar))
	{
		if( RC_BAD( rc = skipWhitespace( pvField)))
		{
			goto Exit;
		}
		bTmp = TRUE;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_S || uChar == F_XML_UNI_P)
	{
		if( !bTmp)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( RC_BAD( rc = processID( pRec, pvField, &bTmp)))
		{
			goto Exit;
		}

		if( bTmp)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( pvField)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uChar)))
		{
			goto Exit;
		}
	}

	if( uChar == F_XML_UNI_LBRACKET)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		for( ;;)
		{
			if( RC_BAD( rc = skipWhitespace( pvField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_PERCENT)
			{
				if( RC_BAD( rc = processPERef( pRec, pvField)))
				{
					goto Exit;
				}
			}
			else if( uChar == F_XML_UNI_RBRACKET)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}
				break;
			}
			else
			{
				if( RC_BAD( rc = processMarkupDecl( pRec, pvField)))
				{
					goto Exit;
				}
			}
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML markup declaration
****************************************************************************/
RCODE F_XMLImport::processMarkupDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUINT				uiChars;
	RCODE					rc = FERR_OK;

	uiChars = 10;
	rc = getChars( m_uChars, &uiChars);
	if( RC_BAD( rc) && rc != FERR_EOF_HIT)
	{
		goto Exit;
	}

	if( RC_BAD( rc = ungetChars( m_uChars, uiChars)))
	{
		goto Exit;
	}

	if( uiChars >= 10 &&
		m_uChars[ 0] == F_XML_UNI_LT &&
		m_uChars[ 1] == F_XML_UNI_BANG &&
		m_uChars[ 2] == F_XML_UNI_N &&
		m_uChars[ 3] == F_XML_UNI_O &&
		m_uChars[ 4] == F_XML_UNI_T &&
		m_uChars[ 5] == F_XML_UNI_A &&
		m_uChars[ 6] == F_XML_UNI_T &&
		m_uChars[ 7] == F_XML_UNI_I &&
		m_uChars[ 8] == F_XML_UNI_O &&
		m_uChars[ 9] == F_XML_UNI_N)
	{
		if( RC_BAD( rc = processNotationDecl( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else if( uiChars >= 9 &&
		m_uChars[ 0] == F_XML_UNI_LT &&
		m_uChars[ 1] == F_XML_UNI_BANG &&
		m_uChars[ 2] == F_XML_UNI_E &&
		m_uChars[ 3] == F_XML_UNI_L &&
		m_uChars[ 4] == F_XML_UNI_E &&
		m_uChars[ 5] == F_XML_UNI_M &&
		m_uChars[ 6] == F_XML_UNI_E &&
		m_uChars[ 7] == F_XML_UNI_N &&
		m_uChars[ 8] == F_XML_UNI_T)
	{
		if( RC_BAD( rc = processElementDecl( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else if( uiChars >= 9 &&
		m_uChars[ 0] == F_XML_UNI_LT &&
		m_uChars[ 1] == F_XML_UNI_BANG &&
		m_uChars[ 2] == F_XML_UNI_A &&
		m_uChars[ 3] == F_XML_UNI_T &&
		m_uChars[ 4] == F_XML_UNI_T &&
		m_uChars[ 5] == F_XML_UNI_L &&
		m_uChars[ 6] == F_XML_UNI_I &&
		m_uChars[ 7] == F_XML_UNI_S &&
		m_uChars[ 8] == F_XML_UNI_T)
	{
		if( RC_BAD( rc = processAttListDecl( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else if( uiChars >= 8 &&
		m_uChars[ 0] == F_XML_UNI_LT &&
		m_uChars[ 1] == F_XML_UNI_BANG &&
		m_uChars[ 2] == F_XML_UNI_E &&
		m_uChars[ 3] == F_XML_UNI_N &&
		m_uChars[ 4] == F_XML_UNI_T &&
		m_uChars[ 5] == F_XML_UNI_I &&
		m_uChars[ 6] == F_XML_UNI_T &&
		m_uChars[ 7] == F_XML_UNI_Y)
	{
		if( RC_BAD( rc = processEntityDecl( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else if( uiChars >= 4 &&
		m_uChars[ 0] == F_XML_UNI_LT &&
		m_uChars[ 1] == F_XML_UNI_BANG &&
		m_uChars[ 2] == F_XML_UNI_HYPHEN &&
		m_uChars[ 3] == F_XML_UNI_HYPHEN)
	{
		if( RC_BAD( rc = processComment( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else if( uiChars >= 2 &&
		m_uChars[ 0] == F_XML_UNI_LT &&
		m_uChars[ 1] == F_XML_UNI_QUEST)
	{
		if( RC_BAD( rc = processPI( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML element declaration
****************************************************************************/
RCODE F_XMLImport::processElementDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ELEMENTDECL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 9;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_E ||
		m_uChars[ 3] != F_XML_UNI_L ||
		m_uChars[ 4] != F_XML_UNI_E ||
		m_uChars[ 5] != F_XML_UNI_M ||
		m_uChars[ 6] != F_XML_UNI_E ||
		m_uChars[ 7] != F_XML_UNI_N ||
		m_uChars[ 8] != F_XML_UNI_T)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}
	
	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( pvField)
	{
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processContentSpec( pRec, pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an attribute list declaration
****************************************************************************/
RCODE F_XMLImport::processAttListDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	FLMUINT				uiAttDefCount = 0;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ATTLISTDECL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 9;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_A ||
		m_uChars[ 3] != F_XML_UNI_T ||
		m_uChars[ 4] != F_XML_UNI_T ||
		m_uChars[ 5] != F_XML_UNI_L ||
		m_uChars[ 6] != F_XML_UNI_I ||
		m_uChars[ 7] != F_XML_UNI_S ||
		m_uChars[ 8] != F_XML_UNI_T)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
	{
		goto Exit;
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( pvField)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar == F_XML_UNI_GT)
		{
			if( !uiAttDefCount)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
			break;
		}

		if( RC_BAD( rc = processAttDef( pRec, pvField)))
		{
			goto Exit;
		}

		uiAttDefCount++;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes and entity declaration
****************************************************************************/
RCODE F_XMLImport::processEntityDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	FLMBOOL				bTmp;
	FLMBOOL				bGEDecl = FALSE;
	void *				pvEntityField = NULL;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ENTITYDECL_TAG, FLM_TEXT_TYPE, &pvEntityField)))
		{
			goto Exit;
		}
	}

	uiChars = 8;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_E ||
		m_uChars[ 3] != F_XML_UNI_N ||
		m_uChars[ 4] != F_XML_UNI_T ||
		m_uChars[ 5] != F_XML_UNI_I ||
		m_uChars[ 6] != F_XML_UNI_T ||
		m_uChars[ 7] != F_XML_UNI_Y)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvEntityField, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_PERCENT)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( pvEntityField, TRUE)))
		{
			goto Exit;
		}
	}
	else
	{
		bGEDecl = TRUE;
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}
	
	if( pvEntityField)
	{
		if( RC_BAD( rc = pRec->setUnicode( pvEntityField, m_uChars)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvEntityField, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( isQuoteChar( uChar))
	{
		if( RC_BAD( rc = processEntityValue( pRec, pvEntityField)))
		{
			goto Exit;
		}
	}
	else
	{
		if( RC_BAD( rc = processID( pRec, pvEntityField, &bTmp)))
		{
			goto Exit;
		}

		if( bTmp)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( pvEntityField)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_GT)
		{
			if( bGEDecl)
			{
				uiChars = 6;
				if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
				{
					goto Exit;
				}

				if( m_uChars[ 0] != F_XML_UNI_N ||
					m_uChars[ 1] != F_XML_UNI_D ||
					m_uChars[ 2] != F_XML_UNI_A ||
					m_uChars[ 3] != F_XML_UNI_T ||
					m_uChars[ 4] != F_XML_UNI_A ||
					!isWhitespace( m_uChars[ 5]))
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				if( RC_BAD( rc = skipWhitespace( pvEntityField)))
				{
					goto Exit;
				}

				uiChars = F_XML_MAX_CHARS;
				if( RC_BAD( rc = getName( m_uChars, &uiChars)))
				{
					goto Exit;
				}

				if( pvEntityField)
				{
					if( RC_BAD( rc = pRec->insertLast( 
						pRec->getLevel( pvEntityField) + 1,
						F_XML_NDATADECL_TAG, FLM_TEXT_TYPE, &pvField)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
					{
						goto Exit;
					}
				}

				if( RC_BAD( rc = skipWhitespace( pvEntityField)))
				{
					goto Exit;
				}
			}
			else
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML ID
****************************************************************************/
RCODE F_XMLImport::processID(
	FlmRecord *		pRec,
	void *			pvParent,
	FLMBOOL *		pbPublicId)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvExtIdField = NULL;
	void *				pvLiteralField = NULL;
	RCODE					rc = FERR_OK;

	*pbPublicId = FALSE;

	uiChars = 7;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] == F_XML_UNI_S &&
		m_uChars[ 1] == F_XML_UNI_Y &&
		m_uChars[ 2] == F_XML_UNI_S &&
		m_uChars[ 3] == F_XML_UNI_T &&
		m_uChars[ 4] == F_XML_UNI_E &&
		m_uChars[ 5] == F_XML_UNI_M &&
		isWhitespace( m_uChars[ 6]))
	{
		if( RC_BAD( rc = skipWhitespace( NULL)))
		{
			goto Exit;
		}

		uiChars = F_XML_MAX_CHARS;
		if( RC_BAD( rc = getSystemLiteral( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		if( pvParent)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvParent) + 1,
				F_XML_EXTERNALID_TAG, FLM_TEXT_TYPE, &pvExtIdField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvExtIdField) + 1,
				F_XML_SYSLITERAL_TAG, FLM_TEXT_TYPE, &pvLiteralField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setUnicode( pvLiteralField, m_uChars)))
			{
				goto Exit;
			}
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_P &&
		m_uChars[ 1] == F_XML_UNI_U &&
		m_uChars[ 2] == F_XML_UNI_B &&
		m_uChars[ 3] == F_XML_UNI_L &&
		m_uChars[ 4] == F_XML_UNI_I &&
		m_uChars[ 5] == F_XML_UNI_C &&
		isWhitespace( m_uChars[ 6]))
	{
		if( RC_BAD( rc = skipWhitespace( NULL)))
		{
			goto Exit;
		}

		uiChars = F_XML_MAX_CHARS;
		if( RC_BAD( rc = getPubidLiteral( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = skipWhitespace( NULL, TRUE)))
		{
			if( rc != FERR_SYNTAX)
			{
				goto Exit;
			}

			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_GT)
			{
				*pbPublicId = TRUE;

				if( pvParent)
				{
					if( RC_BAD( rc = pRec->insertLast( 
						pRec->getLevel( pvParent) + 1,
						F_XML_PUBIDLITERAL_TAG, FLM_TEXT_TYPE, &pvLiteralField)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = pRec->setUnicode( pvLiteralField, m_uChars)))
					{
						goto Exit;
					}
				}
			}
			else
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
			goto Exit;
		}

		if( pvParent)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvParent) + 1,
				F_XML_EXTERNALID_TAG, FLM_TEXT_TYPE, &pvExtIdField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvExtIdField) + 1,
				F_XML_PUBIDLITERAL_TAG, FLM_TEXT_TYPE, &pvLiteralField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setUnicode( pvLiteralField, m_uChars)))
			{
				goto Exit;
			}
		}

		uiChars = F_XML_MAX_CHARS;
		if( RC_BAD( rc = getSystemLiteral( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		if( pvExtIdField)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvExtIdField) + 1,
				F_XML_SYSLITERAL_TAG, FLM_TEXT_TYPE, &pvLiteralField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setUnicode( pvLiteralField, m_uChars)))
			{
				goto Exit;
			}
		}
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a notation declaration
****************************************************************************/
RCODE F_XMLImport::processNotationDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	FLMBOOL				bTmp;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_NOTATIONDECL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 10;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_N ||
		m_uChars[ 3] != F_XML_UNI_O ||
		m_uChars[ 4] != F_XML_UNI_T ||
		m_uChars[ 5] != F_XML_UNI_A ||
		m_uChars[ 6] != F_XML_UNI_T ||
		m_uChars[ 7] != F_XML_UNI_I ||
		m_uChars[ 8] != F_XML_UNI_O ||
		m_uChars[ 9] != F_XML_UNI_N)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( pvField)
	{
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processID( pRec, pvField, &bTmp)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_GT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes and attribute definition
****************************************************************************/
RCODE F_XMLImport::processAttDef(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUINT				uiChars;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ATTRDEF_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( pvField)
	{
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processAttType( pRec, pvField)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvField, TRUE)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = processDefaultDecl( pRec, pvField)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an attribute type
****************************************************************************/
RCODE F_XMLImport::processAttType(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvType = NULL;
	void *				pvToken = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ATTTYPE_TAG, FLM_TEXT_TYPE, &pvType)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = getChar( &m_uChars[ 0])))
	{
		goto Exit;
	}

	if( m_uChars[ 0] == F_XML_UNI_C)
	{
		uiChars = 4;
		if( RC_BAD( rc = getChars( &m_uChars[ 1], &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 1] != F_XML_UNI_D ||
			m_uChars[ 2] != F_XML_UNI_A ||
			m_uChars[ 3] != F_XML_UNI_T ||
			m_uChars[ 4] != F_XML_UNI_A)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( pvType)
		{
			m_uChars[ 5] = 0;
			if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
			{
				goto Exit;
			}
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_I)
	{
		uiChars = 2;
		if( RC_BAD( rc = getChars( &m_uChars[ 1], &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 1] == F_XML_UNI_D &&
			isWhitespace( m_uChars[ 2]))
		{
			if( pvType)
			{
				m_uChars[ 2] = 0;
				if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
				{
					goto Exit;
				}
			}
		}
		else if( m_uChars[ 1] == F_XML_UNI_D &&
			m_uChars[ 2] == F_XML_UNI_R)
		{
			uiChars = 3;
			if( RC_BAD( rc = getChars( &m_uChars[ 3], &uiChars)))
			{
				goto Exit;
			}
			
			if( m_uChars[ 3] == F_XML_UNI_E &&
				m_uChars[ 4] == F_XML_UNI_F)
			{
				if( isWhitespace( m_uChars[ 5]))
				{
					if( pvType)
					{
						m_uChars[ 5] = 0;
						if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
						{
							goto Exit;
						}
					}
				}
				else if( m_uChars[ 5] == F_XML_UNI_S)
				{
					if( RC_BAD( rc = peekChar( &uChar)))
					{
						goto Exit;
					}

					if( isWhitespace( uChar))
					{
						if( pvType)
						{
							m_uChars[ 6] = 0;
							if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
							{
								goto Exit;
							}
						}
					}
					else
					{
						rc = RC_SET( FERR_SYNTAX);
						goto Exit;
					}
				}
				else
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_E)
	{
		uiChars = 6;
		if( RC_BAD( rc = getChars( &m_uChars[ 1], &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 1] == F_XML_UNI_N &&
			m_uChars[ 2] == F_XML_UNI_T &&
			m_uChars[ 3] == F_XML_UNI_I &&
			m_uChars[ 4] == F_XML_UNI_T)
		{
			if( m_uChars[ 5] == F_XML_UNI_Y &&
				isWhitespace( m_uChars[ 6]))
			{
				if( pvType)
				{
					m_uChars[ 6] = 0;
					if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
					{
						goto Exit;
					}
				}
			}
			else if( m_uChars[ 5] == F_XML_UNI_I &&
				m_uChars[ 6] == F_XML_UNI_E)
			{
				if( RC_BAD( rc = getChar( &m_uChars[ 7])))
				{
					goto Exit;
				}

				if( m_uChars[ 7] != F_XML_UNI_S)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}

				if( pvType)
				{
					m_uChars[ 8] = 0;
					if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
					{
						goto Exit;
					}
				}

				if( RC_BAD( rc = peekChar( &uChar)))
				{
					goto Exit;
				}

				if( !isWhitespace( uChar))
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
			else
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_N)
	{
		uiChars = 7;
		if( RC_BAD( rc = getChars( &m_uChars[ 1], &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 1] == F_XML_UNI_M &&
			m_uChars[ 2] == F_XML_UNI_T &&
			m_uChars[ 3] == F_XML_UNI_O &&
			m_uChars[ 4] == F_XML_UNI_K &&
			m_uChars[ 5] == F_XML_UNI_E &&
			m_uChars[ 6] == F_XML_UNI_N)
		{
			if( isWhitespace( m_uChars[ 7]))
			{
				if( pvType)
				{
					m_uChars[ 7] = 0;
					if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
					{
						goto Exit;
					}
				}
			}
			else if( m_uChars[ 7] == F_XML_UNI_S)
			{
				if( RC_BAD( rc = peekChar( &uChar)))
				{
					goto Exit;
				}

				if( isWhitespace( uChar))
				{
					if( pvType)
					{
						m_uChars[ 8] = 0;
						if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
						{
							goto Exit;
						}
					}
				}
				else
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
			else
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
		else if( m_uChars[ 1] == F_XML_UNI_O &&
			m_uChars[ 2] == F_XML_UNI_T &&
			m_uChars[ 3] == F_XML_UNI_A &&
			m_uChars[ 4] == F_XML_UNI_T &&
			m_uChars[ 5] == F_XML_UNI_I &&
			m_uChars[ 6] == F_XML_UNI_O &&
			m_uChars[ 7] == F_XML_UNI_N)
		{
			if( RC_BAD( rc = skipWhitespace( NULL, TRUE)))
			{
				goto Exit;
			}

			if( pvType)
			{
				m_uChars[ 8] = 0;
				if( RC_BAD( rc = pRec->setUnicode( pvType, m_uChars)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar != F_XML_UNI_LPAREN)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			for( ;;)
			{
				if( RC_BAD( rc = skipWhitespace( NULL)))
				{
					goto Exit;
				}

				uiChars = F_XML_MAX_CHARS;
				if( RC_BAD( rc = getName( m_uChars, &uiChars)))
				{
					goto Exit;
				}

				if( pvType)
				{
					if( RC_BAD( rc = pRec->insertLast( 
						pRec->getLevel( pvType) + 1,
						F_XML_ATTTYPE_TAG, FLM_TEXT_TYPE, &pvToken)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = pRec->setUnicode( pvToken, m_uChars)))
					{
						goto Exit;
					}
				}

				if( RC_BAD( rc = skipWhitespace( NULL)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}

				if( uChar == F_XML_UNI_RPAREN)
				{
					break;
				}
				else if( uChar != F_XML_UNI_PIPE)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
			}
		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_LPAREN)
	{
		if( pvType)
		{
			if( RC_BAD( rc = pRec->setNative( pvType, "ENUM")))
			{
				goto Exit;
			}
		}

		for( ;;)
		{
			if( RC_BAD( rc = skipWhitespace( NULL)))
			{
				goto Exit;
			}

			uiChars = F_XML_MAX_CHARS;
			if( RC_BAD( rc = getNmtoken( m_uChars, &uiChars)))
			{
				goto Exit;
			}

			if( !uiChars)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			if( pvType)
			{
				if( RC_BAD( rc = pRec->insertLast( 
					pRec->getLevel( pvType) + 1,
					F_XML_ATTTYPE_TAG, FLM_TEXT_TYPE, &pvToken)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRec->setUnicode( pvToken, m_uChars)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = skipWhitespace( NULL)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_RPAREN)
			{
				break;
			}
			else if( uChar != F_XML_UNI_PIPE)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a default declaration
****************************************************************************/
RCODE F_XMLImport::processDefaultDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	FLMBOOL				bAttValOk = TRUE;
	FLMBOOL				bShouldHaveVal = TRUE;
	RCODE					rc = FERR_OK;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_POUND)
	{
		uiChars = 6;
		if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 0] == F_XML_UNI_F &&
			m_uChars[ 1] == F_XML_UNI_I &&
			m_uChars[ 2] == F_XML_UNI_X &&
			m_uChars[ 3] == F_XML_UNI_E &&
			m_uChars[ 4] == F_XML_UNI_D &&
			isWhitespace( m_uChars[ 5]))
		{
		}
		else if( m_uChars[ 0] == F_XML_UNI_I &&
			m_uChars[ 1] == F_XML_UNI_M &&
			m_uChars[ 2] == F_XML_UNI_P &&
			m_uChars[ 3] == F_XML_UNI_L &&
			m_uChars[ 4] == F_XML_UNI_I &&
			m_uChars[ 5] == F_XML_UNI_E)
		{
			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar != F_XML_UNI_D)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			bShouldHaveVal = FALSE;
		}
		else if( m_uChars[ 0] == F_XML_UNI_R &&
			m_uChars[ 1] == F_XML_UNI_E &&
			m_uChars[ 2] == F_XML_UNI_Q &&
			m_uChars[ 3] == F_XML_UNI_U &&
			m_uChars[ 4] == F_XML_UNI_I &&
			m_uChars[ 5] == F_XML_UNI_R)
		{
			uiChars = 2;
			if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
			{
				goto Exit;
			}

			if( m_uChars[ 0] != F_XML_UNI_E ||
				m_uChars[ 1] != F_XML_UNI_D)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			bShouldHaveVal = FALSE;
		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( !bShouldHaveVal)
	{
		goto Exit;
	}
	
	if( uChar == F_XML_UNI_QUOTE ||
		uChar == F_XML_UNI_APOS)
	{
		if( !bAttValOk)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( RC_BAD( rc = ungetChar( uChar)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = processAttValue( pRec, pvParent)))
		{
			goto Exit;
		}
	}
	else
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a content specification
****************************************************************************/
RCODE F_XMLImport::processContentSpec(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUINT				uiChars;
	void *				pvField = NULL;
	FLMUNICODE			uChar;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_CONTENTSPEC_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = getChar( &m_uChars[ 0])))
	{
		goto Exit;
	}

	if( m_uChars[ 0] == F_XML_UNI_E)
	{
		uiChars = 4;
		if( RC_BAD( rc = getChars( &m_uChars[ 1], &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 1] != F_XML_UNI_M ||
			m_uChars[ 2] != F_XML_UNI_P ||
			m_uChars[ 3] != F_XML_UNI_T ||
			m_uChars[ 4] != F_XML_UNI_Y)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( pvField)
		{
			m_uChars[ 5] = 0;
			if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
			{
				goto Exit;
			}
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_A)
	{
		uiChars = 2;
		if( RC_BAD( rc = getChars( &m_uChars[ 1], &uiChars)))
		{
			goto Exit;
		}

		if( m_uChars[ 1] != F_XML_UNI_N ||
			m_uChars[ 2] != F_XML_UNI_Y)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( pvField)
		{
			m_uChars[ 3] = 0;
			if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
			{
				goto Exit;
			}
		}
	}
	else if( m_uChars[ 0] == F_XML_UNI_LPAREN)
	{
		if( RC_BAD( rc = skipWhitespace( NULL)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = peekChar( &uChar)))
		{
			goto Exit;
		}
		
		if( RC_BAD( rc = ungetChar( F_XML_UNI_LPAREN)))
		{
			goto Exit;
		}

		if( uChar == F_XML_UNI_POUND)
		{
			if( RC_BAD( rc = processMixedContent( pRec, pvParent)))
			{
				goto Exit;
			}
		}
		else
		{
			if( RC_BAD( rc = processChildContent( pRec, pvParent)))
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
RCODE F_XMLImport::processMixedContent(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvTop = NULL;
	void *				pvName = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_MIXED_TAG, FLM_TEXT_TYPE, &pvTop)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_LPAREN)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( pvTop)))
	{
		goto Exit;
	}

	uiChars = 7;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_POUND ||
		m_uChars[ 1] != F_XML_UNI_P ||
		m_uChars[ 2] != F_XML_UNI_C ||
		m_uChars[ 3] != F_XML_UNI_D ||
		m_uChars[ 4] != F_XML_UNI_A ||
		m_uChars[ 5] != F_XML_UNI_T ||
		m_uChars[ 6] != F_XML_UNI_A)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( pvTop)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar == F_XML_UNI_RPAREN)
		{
			break;
		}
		else if( uChar == F_XML_UNI_PIPE)
		{
			if( RC_BAD( rc = skipWhitespace( NULL)))
			{
				goto Exit;
			}

			uiChars = F_XML_MAX_CHARS;
			if( RC_BAD( rc = getName( m_uChars, &uiChars)))
			{
				goto Exit;
			}

			if( pvTop)
			{
				if( RC_BAD( rc = pRec->insertLast( 
					pRec->getLevel( pvTop) + 1,
					F_XML_NAME_TAG, FLM_TEXT_TYPE, &pvName)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRec->setUnicode( pvName, m_uChars)))
				{
					goto Exit;
				}
			}
		}
		else
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes child content
****************************************************************************/
RCODE F_XMLImport::processChildContent(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	FLMUINT				uiItemCount = 0;
	FLMUINT				uiDelimCount = 0;
	FLMBOOL				bChoice = FALSE;
	FLMBOOL				bSeq = FALSE;
	void *				pvTop = NULL;
	void *				pvName = NULL;
	void *				pvOccurs = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_UNKNOWN_TAG, FLM_TEXT_TYPE, &pvTop)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_LPAREN)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( NULL)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar == F_XML_UNI_LPAREN)
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = processChildContent( pRec, pvTop)))
			{
				goto Exit;
			}

			uiItemCount++;
		}
		else if( uChar == F_XML_UNI_RPAREN)
		{
			if( !uiItemCount || (uiItemCount - 1) != uiDelimCount)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			break;
		}
		else if( uChar == F_XML_UNI_PIPE)
		{
			if( bSeq)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
			bChoice = TRUE;
			uiDelimCount++;
		}
		else if( uChar == F_XML_UNI_COMMA)
		{
			if( bChoice)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
			bSeq = TRUE;
			uiDelimCount++;
		}
		else
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			uiChars = F_XML_MAX_CHARS;
			if( RC_BAD( rc = getName( m_uChars, &uiChars)))
			{
				goto Exit;
			}
			uiItemCount++;

			if( pvTop)
			{
				if( RC_BAD( rc = pRec->insertLast( 
					pRec->getLevel( pvTop) + 1,
					F_XML_NAME_TAG, FLM_TEXT_TYPE, &pvName)))
				{
					goto Exit;
				}

				if( RC_BAD( rc = pRec->setUnicode( pvName, m_uChars)))
				{
					goto Exit;
				}
			}

			if( RC_BAD( rc = peekChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_QUEST ||
				uChar == F_XML_UNI_ASTERISK ||
				uChar == F_XML_UNI_PLUS)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}

				if( pvName)
				{
					if( RC_BAD( rc = pRec->insertLast( 
						pRec->getLevel( pvName) + 1,
						F_XML_OCCURS_TAG, FLM_TEXT_TYPE, &pvOccurs)))
					{
						goto Exit;
					}

					m_uChars[ 0] = uChar;
					m_uChars[ 1] = 0;
					if( RC_BAD( rc = pRec->setUnicode( pvOccurs, m_uChars)))
					{
						goto Exit;
					}
				}
			}
		}
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_QUEST ||
		uChar == F_XML_UNI_ASTERISK ||
		uChar == F_XML_UNI_PLUS)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( pvTop)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvTop) + 1,
				F_XML_OCCURS_TAG, FLM_TEXT_TYPE, &pvOccurs)))
			{
				goto Exit;
			}

			m_uChars[ 0] = uChar;
			m_uChars[ 1] = 0;
			if( RC_BAD( rc = pRec->setUnicode( pvOccurs, m_uChars)))
			{
				goto Exit;
			}
		}
	}

	if( pvTop)
	{
		pRec->setFieldID( pvTop, 
			bChoice ? F_XML_CHOICE_TAG : F_XML_SEQ_TAG);
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a misc. declaration
****************************************************************************/
RCODE F_XMLImport::processMisc(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUINT		uiChars;
	RCODE			rc = FERR_OK;

	for( ;;)
	{
		if( RC_BAD( rc = skipWhitespace( NULL)))
		{
			if( rc == FERR_IO_END_OF_FILE || rc == FERR_EOF_HIT)
			{
				rc = FERR_OK;
			}
			goto Exit;
		}

		uiChars = 4;
		rc = getChars( m_uChars, &uiChars);
		if( RC_BAD( rc) && rc != FERR_EOF_HIT)
		{
			goto Exit;
		}

		if( RC_BAD( rc = ungetChars( m_uChars, uiChars)))
		{
			goto Exit;
		}

		if( uiChars >= 4 &&
			m_uChars[ 0] == F_XML_UNI_LT &&
			m_uChars[ 1] == F_XML_UNI_BANG &&
			m_uChars[ 2] == F_XML_UNI_HYPHEN &&
			m_uChars[ 3] == F_XML_UNI_HYPHEN)
		{
			if( RC_BAD( rc = processComment( pRec, pvParent)))
			{
				goto Exit;
			}
		}
		else if( uiChars >= 3 &&
			m_uChars[ 0] == F_XML_UNI_LT &&
			m_uChars[ 1] == F_XML_UNI_BANG &&
			m_uChars[ 2] == F_XML_UNI_QUEST)
		{
			if( RC_BAD( rc = processPI( pRec, pvParent)))
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
RCODE F_XMLImport::processPI(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	RCODE					rc = FERR_OK;

	F_UNREFERENCED_PARM( pRec);
	F_UNREFERENCED_PARM( pvParent);

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}
	
	if( uChar != F_XML_UNI_LT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}
	
	if( uChar != F_XML_UNI_QUEST)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( uiChars >= 3 && 
		(m_uChars[ 0] == F_XML_UNI_X ||
		m_uChars[ 0] == F_XML_UNI_x) &&
		(m_uChars[ 1] == F_XML_UNI_M ||
		m_uChars[ 1] == F_XML_UNI_m) &&
		(m_uChars[ 2] == F_XML_UNI_L ||
		m_uChars[ 2] == F_XML_UNI_l))
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_QUEST)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_GT)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL, TRUE)))
	{
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}
		
		if( uChar == F_XML_UNI_QUEST)
		{
			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_GT)
			{
				break;
			}
			
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			uChar = F_XML_UNI_QUEST;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Gets a single byte from the input stream
****************************************************************************/
RCODE F_XMLImport::getByte(
	FLMBYTE *		pucByte)
{
	RCODE			rc = FERR_OK;

	if( m_ucUngetByte)
	{
		*pucByte = m_ucUngetByte;
		m_ucUngetByte = 0;
	}
	else
	{
		if( RC_BAD( rc = m_pStream->read( pucByte, 1, NULL)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Pushes a single byte back into the input stream
****************************************************************************/
RCODE F_XMLImport::ungetByte(
	FLMBYTE 			ucByte)
{
	flmAssert( !m_ucUngetByte);
	m_ucUngetByte = ucByte;
	return( FERR_OK);
}

/****************************************************************************
Desc:		Pushes a single Unicode character back into the input stream
****************************************************************************/
RCODE F_XMLImport::ungetChar(
	FLMUNICODE 		uChar)
{
	RCODE		rc = FERR_OK;

	if( m_uiUngetPos >= F_XML_MAX_UNGET)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	m_puUngetBuf[ m_uiUngetPos++] = uChar;

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Pushes multiple Unicode characters back into the input stream
****************************************************************************/
RCODE F_XMLImport::ungetChars(
	FLMUNICODE *		puChars,
	FLMUINT				uiChars)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiLoop;

	for( uiLoop = 1; uiLoop <= uiChars; uiLoop++)
	{
		if( RC_BAD( rc = ungetChar( puChars[ uiChars - uiLoop])))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Returns the next character in the input stream without removing it
****************************************************************************/
RCODE F_XMLImport::peekChar(
	FLMUNICODE *		puChar)
{
	RCODE		rc = FERR_OK;

	if( m_uiUngetPos)
	{
		*puChar = m_puUngetBuf[ m_uiUngetPos - 1];
	}
	else
	{
		if( RC_BAD( rc = getChar( puChar)))
		{
			goto Exit;
		}

		if( RC_BAD( rc = ungetChar( *puChar)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Gets an XML name from the input stream
****************************************************************************/
RCODE F_XMLImport::getName(
	FLMUNICODE *	puzName,
	FLMUINT *		puiChars)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiMaxChars = *puiChars;
	FLMUINT			uiOffset = 0;
	FLMUNICODE		uChar;

	flmAssert( uiMaxChars >= 2);

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( !isLetter( uChar) && uChar != F_XML_UNI_UNDERSCORE &&
		uChar != F_XML_UNI_COLON)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	puzName[ uiOffset++] = uChar;

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( !isNameChar( uChar))
		{
			break;
		}

		if( uiOffset >= uiMaxChars)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		puzName[ uiOffset++] = uChar;
	}

	puzName[ uiOffset] = 0;

	if( RC_BAD( rc = ungetChar( uChar)))
	{
		goto Exit;
	}

Exit:

	*puiChars = uiOffset;
	return( rc);
}

/****************************************************************************
Desc:		Gets an XML Nmtoken from the input stream
****************************************************************************/
RCODE F_XMLImport::getNmtoken(
	FLMUNICODE *	puzName,
	FLMUINT *		puiChars)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiMaxChars = *puiChars;
	FLMUINT			uiOffset = 0;
	FLMUNICODE		uChar;

	flmAssert( uiMaxChars >= 2);

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( !isNameChar( uChar))
		{
			break;
		}

		if( uiOffset >= uiMaxChars)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		puzName[ uiOffset++] = uChar;
	}

	puzName[ uiOffset] = 0;

	if( RC_BAD( rc = ungetChar( uChar)))
	{
		goto Exit;
	}

Exit:

	*puiChars = uiOffset;
	return( rc);
}

/****************************************************************************
Desc:		Processes the XML version information encoded within the document
****************************************************************************/
RCODE F_XMLImport::processVersion(
	FlmRecord *		pRec,
	void *			pvParent)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiChars;
	FLMBOOL			bSingleQuote = FALSE;
	FLMUINT			uiOffset = 0;
	void *			pvField = NULL;
	FLMUNICODE		uChar;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_VERSION_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 7;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_v ||
		m_uChars[ 1] != F_XML_UNI_e ||
		m_uChars[ 2] != F_XML_UNI_r ||
		m_uChars[ 3] != F_XML_UNI_s ||
		m_uChars[ 4] != F_XML_UNI_i ||
		m_uChars[ 5] != F_XML_UNI_o ||
		m_uChars[ 6] != F_XML_UNI_n)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_EQ)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	m_uChars[ uiOffset++] = uChar;

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		m_uChars[ uiOffset++] = uChar;
		if( uiOffset >= F_XML_MAX_CHARS)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		if( (uChar == F_XML_UNI_APOS && bSingleQuote) ||
			(uChar == F_XML_UNI_QUOTE && !bSingleQuote))
		{
			break;
		}
		else if( !((uChar >= F_XML_UNI_A && uChar <= F_XML_UNI_Z) ||
			(uChar >= F_XML_UNI_a && uChar <= F_XML_UNI_z) ||
			(uChar >= F_XML_UNI_0 && uChar <= F_XML_UNI_9) ||
			uChar == F_XML_UNI_UNDERSCORE ||
			uChar == F_XML_UNI_PERIOD ||
			uChar == F_XML_UNI_COLON ||
			uChar == F_XML_UNI_HYPHEN))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( !uiOffset)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( pvField)
	{
		m_uChars[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML encoding declaration
****************************************************************************/
RCODE F_XMLImport::processEncodingDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiChars;
	FLMBOOL			bSingleQuote = FALSE;
	FLMUINT			uiOffset = 0;
	void *			pvField = NULL;
	FLMUNICODE		uChar;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ENCODING_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 8;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_e ||
		m_uChars[ 1] != F_XML_UNI_n ||
		m_uChars[ 2] != F_XML_UNI_c ||
		m_uChars[ 3] != F_XML_UNI_o ||
		m_uChars[ 4] != F_XML_UNI_d ||
		m_uChars[ 5] != F_XML_UNI_i ||
		m_uChars[ 6] != F_XML_UNI_n ||
		m_uChars[ 7] != F_XML_UNI_g)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_EQ)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	// VISIT: Make sure the encoding decl starts with
	// [A-Za-z]

	m_uChars[ uiOffset++] = uChar;

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		m_uChars[ uiOffset++] = uChar;
		if( uiOffset >= F_XML_MAX_CHARS)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}

		if( (uChar == F_XML_UNI_APOS && bSingleQuote) ||
			(uChar == F_XML_UNI_QUOTE && !bSingleQuote))
		{
			break;
		}
		else if( !((uChar >= F_XML_UNI_A && uChar <= F_XML_UNI_Z) ||
			(uChar >= F_XML_UNI_a && uChar <= F_XML_UNI_z) ||
			(uChar >= F_XML_UNI_0 && uChar <= F_XML_UNI_9) ||
			uChar == F_XML_UNI_UNDERSCORE ||
			uChar == F_XML_UNI_PERIOD ||
			uChar == F_XML_UNI_HYPHEN))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( !uiOffset)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( pvField)
	{
		m_uChars[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML SD declaration
****************************************************************************/
RCODE F_XMLImport::processSDDecl(
	FlmRecord *		pRec,
	void *			pvParent)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiChars;
	FLMBOOL			bSingleQuote = FALSE;
	FLMUINT			uiOffset = 0;
	void *			pvField = NULL;
	FLMUNICODE		uChar;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_SDDECL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 10;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_s ||
		m_uChars[ 1] != F_XML_UNI_t ||
		m_uChars[ 2] != F_XML_UNI_a ||
		m_uChars[ 3] != F_XML_UNI_n ||
		m_uChars[ 4] != F_XML_UNI_d ||
		m_uChars[ 5] != F_XML_UNI_a ||
		m_uChars[ 6] != F_XML_UNI_l ||
		m_uChars[ 7] != F_XML_UNI_o ||
		m_uChars[ 8] != F_XML_UNI_n ||
		m_uChars[ 9] != F_XML_UNI_e)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_EQ)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = skipWhitespace( NULL)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	m_uChars[ uiOffset++] = uChar;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_y)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_e)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_s)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		m_uChars[ uiOffset++] = F_XML_UNI_y;
		m_uChars[ uiOffset++] = F_XML_UNI_e;
		m_uChars[ uiOffset++] = F_XML_UNI_s;
	}
	else if( uChar == F_XML_UNI_n)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar != F_XML_UNI_o)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		m_uChars[ uiOffset++] = F_XML_UNI_n;
		m_uChars[ uiOffset++] = F_XML_UNI_o;
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( bSingleQuote)
	{
		if( uChar != F_XML_UNI_APOS)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}
	else
	{
		if( uChar != F_XML_UNI_QUOTE)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}
	}

	if( pvField)
	{
		m_uChars[ uiOffset++] = uChar;
		m_uChars[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Reads a UTF-8 encoded character from the input stream.
****************************************************************************/
RCODE	F_XMLImport::getChar(
	FLMUNICODE *	pChar)
{
	FLMBYTE		ucByte1;
	FLMBYTE		ucByte2;
	FLMBYTE		ucByte3;
	FLMBYTE		ucLoByte;
	FLMBYTE		ucHiByte;
	RCODE			rc = FERR_OK;

	if( m_uiUngetPos)
	{
		*pChar = m_puUngetBuf[ --m_uiUngetPos];
		goto Exit;
	}

	// Read and decode the bytes.

	if( RC_BAD( rc = getByte( &ucByte1)))
	{
		goto Exit;
	}

	// Convert CRLF->CR

	if( ucByte1 == (FLMBYTE)0x0D)
	{
		if( RC_BAD( rc = getByte( &ucByte1)))
		{
			goto Exit;
		}

		if( ucByte1 != (FLMBYTE)0x0A)
		{
			if( RC_BAD( rc = ungetByte( ucByte1)))
			{
				goto Exit;
			}

			ucByte1 = (FLMBYTE)0x0A;
		}
	}

	if( (ucByte1 & 0xC0) != 0xC0)
	{
		ucHiByte = 0;
		ucLoByte = ucByte1;
	}
	else
	{
		if( RC_BAD( rc = getByte( &ucByte2)))
		{
			goto Exit;
		}

		if( (ucByte1 & 0xE0) == 0xE0)
		{
			if( RC_BAD( rc = getByte( &ucByte3)))
			{
				goto Exit;
			}

			ucHiByte =
				(FLMBYTE)(((ucByte1 & 0x0F) << 4) | ((ucByte2 & 0x3C) >> 2));
			ucLoByte = (FLMBYTE)(((ucByte2 & 0x03) << 6) | (ucByte3 & 0x3F));
		}
		else
		{
			ucHiByte = (FLMBYTE)(((ucByte1 & 0x1C) >> 2));
			ucLoByte = (FLMBYTE)(((ucByte1 & 0x03) << 6) | (ucByte2 & 0x3F));
		}
	}

	*pChar = (FLMUNICODE)((((FLMUNICODE)(ucHiByte)) << 8) |
		((FLMUNICODE)(ucLoByte)));

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Reads multiple UTF-8 encoded characters from the stream.
****************************************************************************/
RCODE	F_XMLImport::getChars(
	FLMUNICODE *	uzChars,
	FLMUINT *		puiCount)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiCount = *puiCount;
	FLMUINT		uiLoop;

	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		if( RC_BAD( rc = getChar( &uzChars[ uiLoop])))
		{
			goto Exit;
		}
	}

Exit:

	*puiCount = uiLoop;
	return( rc);
}

/****************************************************************************
Desc:		Reads an XML character entity
****************************************************************************/
RCODE F_XMLImport::getCharEntity(
	FLMUNICODE *	puChar)
{
	FLMBYTE		ucByte;
	RCODE			rc = FERR_OK;

	F_UNREFERENCED_PARM( puChar);

	if( RC_BAD( rc = getByte( &ucByte)))
	{
		goto Exit;
	}

	if( ucByte == '&')
	{
		if( RC_BAD( rc = getByte( &ucByte)))
		{
			goto Exit;
		}

		if( ucByte == '#')
		{
			if( RC_BAD( rc = getByte( &ucByte)))
			{
				goto Exit;
			}
		}

		while( ucByte != ';')
		{
			if( RC_BAD( rc = getByte( &ucByte)))
			{
				goto Exit;
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML entity value
****************************************************************************/
RCODE F_XMLImport::processEntityValue(
	FlmRecord *			pRec,
	void *				pvParent)
{
	FLMUNICODE		uChar;
	FLMBOOL			bSingleQuote = FALSE;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( (uChar == F_XML_UNI_APOS && bSingleQuote) ||
			(uChar == F_XML_UNI_QUOTE && !bSingleQuote))
		{
			break;
		}
		else if( uChar == F_XML_UNI_PERCENT)
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = processPERef( pRec, pvParent)))
			{
				goto Exit;
			}
		}
		else if( uChar == F_XML_UNI_AMP)
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = processReference( pRec, pvParent, NULL)))
			{
				goto Exit;
			}
		}

		// VISIT: Add the character to the buffer
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML PERef
****************************************************************************/
RCODE F_XMLImport::processPERef(
	FlmRecord *		pRec,
	void *			pvParent)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_PEREF_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_PERCENT)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	uiChars = F_XML_MAX_CHARS;
	if( RC_BAD( rc = getName( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( pvField)
	{
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_SEMI)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML reference
****************************************************************************/
RCODE F_XMLImport::processReference(
	FlmRecord *		pRec,
	void *			pvParent,
	FLMUNICODE *	puChar)
{
	FLMUNICODE			uChar;
	FLMUINT				uiChars;
	void *				pvField = NULL;
	RCODE					rc = FERR_OK;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_AMP)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	if( RC_BAD( rc = peekChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_POUND)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}
		
		uiChars = 0;
		for( ;;)
		{
			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_SEMI)
			{
				if( RC_BAD( rc = ungetChar( uChar)))
				{
					goto Exit;
				}
				break;
			}

			// Validate that the characters are valid dec/hex characters

			if( !((uChar >= '0' && uChar <= '9') ||
				(uChar >= 'A' && uChar <= 'F') ||
				(uChar >= 'a' && uChar <= 'f')))
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}

			m_uChars[ uiChars++] = uChar;
			if( uiChars >= F_XML_MAX_CHARS)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}
		}

		m_uChars[ uiChars] = 0;

		if( pvParent)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvParent) + 1,
				F_XML_CHARREF_TAG, FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
			{
				goto Exit;
			}
		}

		if( puChar)
		{
			FLMUINT		uiVal;

			unicodeToNumber( m_uChars, &uiVal, NULL);
			*puChar = (FLMUNICODE)uiVal;
		}
	}
	else
	{
		uiChars = F_XML_MAX_CHARS;
		if( RC_BAD( rc = getName( m_uChars, &uiChars)))
		{
			goto Exit;
		}

		if( pvParent)
		{
			if( RC_BAD( rc = pRec->insertLast( 
				pRec->getLevel( pvParent) + 1,
				F_XML_ENTITYREF_TAG, FLM_TEXT_TYPE, &pvField)))
			{
				goto Exit;
			}

			if( RC_BAD( rc = pRec->setUnicode( pvField, m_uChars)))
			{
				goto Exit;
			}
		}

		if( puChar)
		{
			// Translate pre-defined references

			*puChar = 0;
			if( uiChars == 2)
			{
				if( m_uChars[ 0] == (FLMUNICODE)'l' && 
					m_uChars[ 1] == (FLMUNICODE)'t')
				{
					*puChar = F_XML_UNI_LT;
				}
				else if( m_uChars[ 0] == (FLMUNICODE)'g' && 
					m_uChars[ 1] == (FLMUNICODE)'t')
				{
					*puChar = F_XML_UNI_GT;
				}
			}
			else if( uiChars == 3)
			{
				if( m_uChars[ 0] == (FLMUNICODE)'a' && 
					m_uChars[ 1] == (FLMUNICODE)'m' &&
					m_uChars[ 2] == (FLMUNICODE)'p')
				{
					*puChar = F_XML_UNI_AMP;
				}
			}
			else if( uiChars == 4)
			{
				if( m_uChars[ 0] == (FLMUNICODE)'a' && 
					m_uChars[ 1] == (FLMUNICODE)'p' &&
					m_uChars[ 2] == (FLMUNICODE)'o' &&
					m_uChars[ 3] == (FLMUNICODE)'s')
				{
					*puChar = F_XML_UNI_APOS;
				}
				else if( m_uChars[ 0] == (FLMUNICODE)'q' && 
					m_uChars[ 1] == (FLMUNICODE)'u' &&
					m_uChars[ 2] == (FLMUNICODE)'o' &&
					m_uChars[ 3] == (FLMUNICODE)'t')
				{
					*puChar = F_XML_UNI_QUOTE;
				}
			}

			if( *puChar == 0)
			{
				rc = RC_SET( FERR_SYNTAX);
				goto Exit;
			}
		}
	}

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar != F_XML_UNI_SEMI)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an attribute value
****************************************************************************/
RCODE F_XMLImport::processAttValue(
	FlmRecord *			pRec,
	void *				pvParent)
{
	FLMUNICODE		uChar;
	FLMBOOL			bSingleQuote = FALSE;
	FLMUINT			uiOffset = 0;
	void *			pvField = NULL;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	m_puValBuf[ uiOffset++] = uChar;
	if( uiOffset >= m_uiValBufSize)
	{
		rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( (uChar == F_XML_UNI_APOS && bSingleQuote) ||
			(uChar == F_XML_UNI_QUOTE && !bSingleQuote))
		{
			m_puValBuf[ uiOffset++] = uChar;
			if( uiOffset >= m_uiValBufSize)
			{
				rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
				goto Exit;
			}
			break;
		}
		else if( uChar == F_XML_UNI_AMP)
		{
			if( pvParent && uiOffset)
			{
				if( RC_BAD( rc = pRec->insertLast( 
					pRec->getLevel( pvParent) + 1,
					F_XML_ATTVAL_TAG, FLM_TEXT_TYPE, &pvField)))
				{
					goto Exit;
				}

				m_puValBuf[ uiOffset] = 0;
				if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
				{
					goto Exit;
				}
				uiOffset = 0;
			}

			if( RC_BAD( rc = processReference( pRec, pvParent, NULL)))
			{
				goto Exit;
			}
		}
		else if( uChar == F_XML_UNI_PERCENT)
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		m_puValBuf[ uiOffset++] = uChar;
		if( uiOffset >= m_uiValBufSize)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	}

	if( pvParent && uiOffset)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_ATTVAL_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}

		m_puValBuf[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
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
RCODE F_XMLImport::getSystemLiteral(
	FLMUNICODE *		puBuf,
	FLMUINT *			puiMaxChars)
{
	FLMUNICODE		uChar;
	FLMUINT			uiMaxChars = *puiMaxChars;
	FLMBOOL			bSingleQuote = FALSE;
	FLMUINT			uiOffset = 0;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( (uChar == F_XML_UNI_APOS && bSingleQuote) ||
			(uChar == F_XML_UNI_QUOTE && !bSingleQuote))
		{
			break;
		}

		puBuf[ uiOffset++] = uChar;
		if( uiOffset >= uiMaxChars)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	} 

	puBuf[ uiOffset] = 0;

Exit:

	*puiMaxChars = uiOffset;
	return( rc);
}

/****************************************************************************
Desc:		Reads an XML public ID literal from the input stream
****************************************************************************/
RCODE F_XMLImport::getPubidLiteral(
	FLMUNICODE *		puBuf,
	FLMUINT *			puiMaxChars)
{
	FLMUNICODE		uChar;
	FLMUINT			uiMaxChars = *puiMaxChars;
	FLMBOOL			bSingleQuote = FALSE;
	FLMUINT			uiOffset = 0;
	RCODE				rc = FERR_OK;

	if( RC_BAD( rc = getChar( &uChar)))
	{
		goto Exit;
	}

	if( uChar == F_XML_UNI_APOS)
	{
		bSingleQuote = TRUE;
	}
	else if( uChar != F_XML_UNI_QUOTE)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( (uChar == F_XML_UNI_APOS && bSingleQuote) ||
			(uChar == F_XML_UNI_QUOTE && !bSingleQuote))
		{
			break;
		}
		else if( !isPubidChar( uChar))
		{
			rc = RC_SET( FERR_SYNTAX);
			goto Exit;
		}

		puBuf[ uiOffset++] = uChar;
		if( uiOffset >= uiMaxChars)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	} 

	puBuf[ uiOffset] = 0;

Exit:

	*puiMaxChars = uiOffset;
	return( rc);
}

/****************************************************************************
Desc:		Returns TRUE if the next item in the input stream is an XML
			declaration
****************************************************************************/
RCODE F_XMLImport::isXMLDecl(
	FLMBOOL *			pbIsXMLDecl)
{
	FLMUINT			uiChars;
	RCODE				rc = FERR_OK;

	*pbIsXMLDecl = FALSE;
	uiChars = 5;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		if( rc == FERR_IO_END_OF_FILE || rc == FERR_EOF_HIT)
		{
			if( uiChars)
			{
				rc = ungetChars( m_uChars, uiChars);
			}
			else
			{
				rc = FERR_OK;
			}
		}
		goto Exit;
	}

	if( m_uChars[ 0] == F_XML_UNI_LT && 
		m_uChars[ 1] == F_XML_UNI_QUEST &&
		m_uChars[ 2] == F_XML_UNI_x &&
		m_uChars[ 3] == F_XML_UNI_m &&
		m_uChars[ 4] == F_XML_UNI_l)
	{
		*pbIsXMLDecl = TRUE;
	}

	if( RC_BAD( rc = ungetChars( m_uChars, uiChars)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Returns TRUE if the next item in the input stream is an XML
			document type declaration
****************************************************************************/
RCODE F_XMLImport::isDocTypeDecl(
	FLMBOOL *			pbIsDocTypeDecl)
{
	FLMUINT			uiChars;
	RCODE				rc = FERR_OK;

	*pbIsDocTypeDecl = FALSE;
	uiChars = 9;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		if( rc == FERR_IO_END_OF_FILE || rc == FERR_EOF_HIT)
		{
			if( uiChars)
			{
				rc = ungetChars( m_uChars, uiChars);
			}
			else
			{
				rc = FERR_OK;
			}
		}
		goto Exit;
	}

	if( m_uChars[ 0] == F_XML_UNI_LT && 
		m_uChars[ 1] == F_XML_UNI_BANG &&
		m_uChars[ 2] == F_XML_UNI_D &&
		m_uChars[ 3] == F_XML_UNI_O &&
		m_uChars[ 4] == F_XML_UNI_C &&
		m_uChars[ 5] == F_XML_UNI_T &&
		m_uChars[ 6] == F_XML_UNI_Y &&
		m_uChars[ 7] == F_XML_UNI_P &&
		m_uChars[ 8] == F_XML_UNI_E)
	{
		*pbIsDocTypeDecl = TRUE;
	}

	if( RC_BAD( rc = ungetChars( m_uChars, uiChars)))
	{
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes an XML comment
****************************************************************************/
RCODE F_XMLImport::processComment(
	FlmRecord *			pRec,
	void *				pvParent)
{
	FLMUNICODE		uChar;
	FLMUNICODE		uChar1;
	FLMUNICODE		uChar2;
	FLMUINT			uiChars;
	FLMUINT			uiOffset;
	FLMUINT			uiMaxOffset;
	void *			pvField = NULL;
	RCODE				rc = FERR_OK;

	if( pvParent)
	{
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_COMMENT_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}
	}

	uiChars = 4;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT || 
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_HYPHEN ||
		m_uChars[ 3] != F_XML_UNI_HYPHEN)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	uiOffset = 0;
	uiMaxOffset = m_uiValBufSize;
	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar == F_XML_UNI_HYPHEN)
		{
			if( RC_BAD( rc = getChar( &uChar1)))
			{
				goto Exit;
			}

			if( uChar1 == F_XML_UNI_HYPHEN)
			{
				if( RC_BAD( rc = getChar( &uChar2)))
				{
					goto Exit;
				}

				if( uChar2 != F_XML_UNI_GT)
				{
					rc = RC_SET( FERR_SYNTAX);
					goto Exit;
				}
				break;
			}
			else
			{
				if( RC_BAD( rc = ungetChar( uChar1)))
				{
					goto Exit;
				}
			}
		}

		m_puValBuf[ uiOffset++] = uChar;
		if( uiOffset >= uiMaxOffset)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	}

	if( pvField)
	{
		m_puValBuf[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Processes a CDATA tag
****************************************************************************/
RCODE F_XMLImport::processCDATA(
	FlmRecord *			pRec,
	void *				pvParent)
{
	FLMUNICODE		uChar;
	FLMUINT			uiChars;
	FLMUINT			uiOffset = 0;
	void *			pvField = NULL;
	RCODE				rc = FERR_OK;

	uiChars = 9;
	if( RC_BAD( rc = getChars( m_uChars, &uiChars)))
	{
		goto Exit;
	}

	if( m_uChars[ 0] != F_XML_UNI_LT ||
		m_uChars[ 1] != F_XML_UNI_BANG ||
		m_uChars[ 2] != F_XML_UNI_LBRACKET ||
		m_uChars[ 3] != F_XML_UNI_C ||
		m_uChars[ 4] != F_XML_UNI_D ||
		m_uChars[ 5] != F_XML_UNI_A ||
		m_uChars[ 6] != F_XML_UNI_T ||
		m_uChars[ 7] != F_XML_UNI_A ||
		m_uChars[ 8] != F_XML_UNI_LBRACKET)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( uChar == F_XML_UNI_RBRACKET)
		{
			if( RC_BAD( rc = getChar( &uChar)))
			{
				goto Exit;
			}

			if( uChar == F_XML_UNI_RBRACKET)
			{
				if( RC_BAD( rc = getChar( &uChar)))
				{
					goto Exit;
				}

				if( uChar == F_XML_UNI_GT)
				{
					break;
				}
				else
				{
					if( RC_BAD( rc = ungetChar( uChar)))
					{
						goto Exit;
					}

					if( RC_BAD( rc = ungetChar( F_XML_UNI_RBRACKET)))
					{
						goto Exit;
					}
					
					uChar = F_XML_UNI_RBRACKET;
				}
			}
			else
			{
				if( RC_BAD( rc = ungetChar( uChar)))
				{
					goto Exit;
				}

				uChar = F_XML_UNI_RBRACKET;
			}
		}

		m_puValBuf[ uiOffset++] = uChar;
		if( uiOffset >= m_uiValBufSize)
		{
			rc = RC_SET( FERR_CONV_DEST_OVERFLOW);
			goto Exit;
		}
	} 

	if( pvParent)
	{
		m_puValBuf[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->insertLast( 
			pRec->getLevel( pvParent) + 1,
			F_XML_CDATA_TAG, FLM_TEXT_TYPE, &pvField)))
		{
			goto Exit;
		}

		m_puValBuf[ uiOffset] = 0;
		if( RC_BAD( rc = pRec->setUnicode( pvField, m_puValBuf)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:		Skips any whitespace characters in the input stream
****************************************************************************/
RCODE F_XMLImport::skipWhitespace(
	void *			pvParent,
	FLMBOOL 			bRequired)
{
	FLMUNICODE		uChar;
	FLMUINT			uiCount = 0;
	RCODE				rc = FERR_OK;

	F_UNREFERENCED_PARM( pvParent);

	for( ;;)
	{
		if( RC_BAD( rc = getChar( &uChar)))
		{
			goto Exit;
		}

		if( !isWhitespace( uChar))
		{
			if( RC_BAD( rc = ungetChar( uChar)))
			{
				goto Exit;
			}
			break;
		}
		uiCount++;
	}

	if( !uiCount && bRequired)
	{
		rc = RC_SET( FERR_SYNTAX);
		goto Exit;
	}

Exit:

	return( rc);
}
